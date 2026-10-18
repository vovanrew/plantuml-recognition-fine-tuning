"""The training recipe of the study, identical at 2B, 9B and 27B.

decisions/recipe/record.md, Decision. The values are the constants below; the functions apply them,
so the training script, the probe and the checkpoint rule (train/select_checkpoint.py) read one
place. Nothing here depends on the model size: the target list is every linear layer of the
language model, whatever their number (186 / 248 / 496 at 2B / 9B / 27B).

  Adapter      LoRA, rank 16, alpha 32, plain alpha/r scaling, dropout 0.05, no bias, float32
               adapter weights, on the 12 kinds of linear layer under `model.language_model.`
               (an explicit name list; the vision tower with its merger, the embeddings and
               `lm_head` carry no adapter and are frozen)
  Optimizer    torch AdamW, betas (0.9, 0.999), eps 1e-8, weight decay 0.01
  Schedule     learning rate 1e-4; linear warm-up over ceil(3% of the run's optimizer steps), then
               constant; no decay
  Clipping     global gradient norm 1.0; the norm before clipping is returned for the log
  Batch        16 pairs per optimizer step, one pair per forward and backward; no padding, packing
  Loss         cross-entropy on the answer tokens (target and `<|im_end|>`), summed over the step's
               16 pairs and divided by their number of answer tokens (`loss_scale`, with
               train/loss.py `backward`); the newline after `<|im_end|>` is not in the sequence
  Vision       the vision tower runs without gradient tracking, as the server runs it
  Order        one random permutation of the run's key list per epoch, seed ORDER_SEED + epoch,
               the same at every size
  Length       2 epochs; the adapter saved every half epoch, in `checkpoint_dir(step)` under the
               run's folder, the optimizer and schedule state with the last, so a run can be continued
  Probe        before any ladder run, at each size: this recipe on a run of 60 optimizer steps
               (warm-up 2 steps, then 1e-4) over the first 960 pairs of the 20,000's epoch-0 order;
               it fails on a non-finite loss or a mean loss of the last 10 steps above that of the
               first 10; its adapter is discarded
  Dev scoring  through train/dev_scoring.py with the scored runs' cap and deadline; the checkpoint
               rule is train/select_checkpoint.py

Importable without torch (the checkpoint rule runs under the scorer's Python 3.9): torch and PEFT
are imported inside the functions that need them.

Tests: train/tests/test_recipe.py
"""
from __future__ import annotations

import math
import random

# --- adapter ---------------------------------------------------------------------------------
RANK = 16
ALPHA = 32
DROPOUT = 0.05
USE_RSLORA = False
LANGUAGE_MODEL_PREFIX = "model.language_model."
TARGET_KINDS = ("gate_proj", "up_proj", "down_proj",                   # MLP, every layer
                "q_proj", "k_proj", "v_proj", "o_proj",                 # full-attention layers
                "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj")  # Gated DeltaNet
INIT_SEED = 42                       # torch seed set before the adapter is attached (lora_A draws)

# --- optimisation ----------------------------------------------------------------------------
LEARNING_RATE = 1e-4
BETAS = (0.9, 0.999)
EPS = 1e-8
WEIGHT_DECAY = 0.01
WARMUP_FRACTION = 0.03
CLIP_NORM = 1.0
PAIRS_PER_STEP = 16
END_WITH_NEWLINE = False             # build/examples.py end_ids(with_newline=False)
VISION_WITHOUT_GRAD = True
ORDER_SEED = 42

# --- length and checkpoints ------------------------------------------------------------------
EPOCHS = 2
CHECKPOINTS_PER_EPOCH = 2
SUBSET_PAIRS = {20000: 2500, 5000: 625, 1000: 125}   # pairs of a run -> largest rank per cell

# --- probe before the ladder -----------------------------------------------------------------
PROBE_STEPS = 60
PROBE_WINDOW = 10
CHECKPOINT_DIR = "step_{:05d}"

# --- dev scoring and the checkpoint rule -----------------------------------------------------
DEV_MAX_TOKENS = 5376                # the scored runs' cap (FINETUNE_PLAN.md §2)
DEV_TIMEOUT_S = 600                  # the scored runs' deadline (decisions/gpu-precision/record.md)
RULE_METRICS = ("csr", "element_f1_micro_full")
BOOTSTRAP_SEED = 20260614            # analysis/bootstrap.py SEED
BOOTSTRAP_RESAMPLES = 1000           # analysis/bootstrap.py N_RESAMPLES


# --------------------------------------------------------------------------- #
# data: which pairs, in which order, in which steps
# --------------------------------------------------------------------------- #
def run_rows(rows, pairs=20000):
    """The rows of a run of `pairs` pairs, in the given order (data/sets/train.json or
    data/examples/train.jsonl order): rank at most 2,500, 625 or 125 per cell."""
    top = SUBSET_PAIRS[pairs]
    chosen = [r for r in rows if r["rank"] <= top]
    if len(chosen) != pairs:
        raise ValueError(f"{len(chosen)} rows have rank <= {top}, expected {pairs}")
    return chosen


def epoch_order(items, epoch):
    """The run's items in the order of epoch `epoch` (0-based): a permutation drawn from
    random.Random(ORDER_SEED + epoch), so every epoch's order is computable on its own."""
    order = list(items)
    random.Random(ORDER_SEED + epoch).shuffle(order)
    return order


def steps_per_epoch(pairs):
    """Optimizer steps in one epoch; the last step of an epoch holds the remainder."""
    return math.ceil(pairs / PAIRS_PER_STEP)


def total_steps(pairs):
    return EPOCHS * steps_per_epoch(pairs)


def epoch_batches(items, epoch):
    """The optimizer steps of epoch `epoch`: lists of PAIRS_PER_STEP items, the last one shorter
    when the run's size is not a multiple of PAIRS_PER_STEP."""
    order = epoch_order(items, epoch)
    return [order[i:i + PAIRS_PER_STEP] for i in range(0, len(order), PAIRS_PER_STEP)]


def checkpoint_steps(pairs):
    """Optimizer steps (1-based, counted over the run) after which the adapter is saved: the
    middle and the end of every epoch. 20,000 pairs: 625, 1,250, 1,875, 2,500."""
    s = steps_per_epoch(pairs)
    steps = []
    for e in range(EPOCHS):
        steps += [e * s + math.ceil(s / CHECKPOINTS_PER_EPOCH), (e + 1) * s]
    return steps


def checkpoint_dir(step):
    """The folder name of the adapter saved after optimizer step `step`, inside the run's folder;
    train/select_checkpoint.py reads the step back from it."""
    return CHECKPOINT_DIR.format(step)


def probe_batches(keys):
    """The probe's optimizer steps: the first PROBE_STEPS steps of epoch 0 of the 20,000-pair run
    (`keys` in data/sets/train.json order), 960 pairs. The probe's schedule is `scheduler(opt,
    PROBE_STEPS)`: 2 warm-up steps, then the full learning rate for the other 58."""
    if len(keys) != 20000:
        raise ValueError(f"the probe draws from the 20,000-pair run, not {len(keys)} pairs")
    return epoch_batches(keys, 0)[:PROBE_STEPS]


def loss_scale(answer_tokens):
    """The factor for each pair's summed answer loss in one step: one over the step's answer
    tokens, so the step's loss is the mean over all its answer tokens (target and `<|im_end|>`)."""
    total = sum(answer_tokens)
    if total <= 0:
        raise ValueError("a step without answer tokens")
    return 1.0 / total


# --------------------------------------------------------------------------- #
# schedule
# --------------------------------------------------------------------------- #
def warmup_steps(total):
    return math.ceil(WARMUP_FRACTION * total)


def lr_factor(step, total):
    """Factor of LEARNING_RATE at optimizer step `step` (1-based) of a run of `total` steps:
    step / warm-up during the warm-up, 1 after it, to the end."""
    return min(1.0, step / warmup_steps(total))


def scheduler(optimizer, total):
    """LambdaLR whose factor before the k-th optimizer step (0-based k) is lr_factor(k + 1)."""
    import torch
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda k: lr_factor(k + 1, total))


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
def target_names(model):
    """Every torch.nn.Linear under `model.language_model.`, as full names. A linear layer of
    another kind there is refused, so an architecture the recipe was not written for stops here."""
    import torch
    names = [n for n, m in model.named_modules()
             if isinstance(m, torch.nn.Linear) and n.startswith(LANGUAGE_MODEL_PREFIX)]
    other = sorted({n.rsplit(".", 1)[1] for n in names} - set(TARGET_KINDS))
    if other or not names:
        raise ValueError(f"linear layers of kinds outside the recipe: {other}" if other
                         else "no linear layer under " + LANGUAGE_MODEL_PREFIX)
    return names


def lora_config(model):
    from peft import LoraConfig
    return LoraConfig(r=RANK, lora_alpha=ALPHA, lora_dropout=DROPOUT, bias="none",
                      use_rslora=USE_RSLORA, target_modules=target_names(model))


def attach(model):
    """The recipe's adapter on `model` (bf16 base), in place: torch seed INIT_SEED, then PEFT with
    its default float32 adapter weights. Every other parameter is frozen. Returns the PEFT model."""
    import torch
    from peft import get_peft_model
    torch.manual_seed(INIT_SEED)
    peft_model = get_peft_model(model, lora_config(model))
    trainable = {n: p for n, p in peft_model.named_parameters() if p.requires_grad}
    wrong = [n for n in trainable if ".lora_" not in n or LANGUAGE_MODEL_PREFIX not in n]
    if wrong or any(p.dtype != torch.float32 for p in trainable.values()):
        raise ValueError(f"trainable parameters outside the recipe: {wrong[:5]}")
    return peft_model


def vision_without_grad(model):
    """Run the vision tower (with its merger) without gradient tracking, as the server does
    (infer/server.py generates under torch.inference_mode). Works on the model or its PEFT
    wrapper; applied once.

    With the recipe's adapter the vision weights are frozen and the tower already tracks no
    gradient: its output in training equals the served output. A vision weight that tracks a
    gradient changes the kernels and the output (relative difference about 0.1 in bf16 on MPS,
    train/tests/test_recipe.py --live); under this wrapper the output stays the served one."""
    import torch
    core = model.get_base_model() if hasattr(model, "get_base_model") else model
    visual = core.model.visual
    if not getattr(visual, "_recipe_without_grad", False):
        visual.forward = torch.no_grad()(visual.forward)
        visual._recipe_without_grad = True
    return model


def optimizer(params):
    import torch
    return torch.optim.AdamW(params, lr=LEARNING_RATE, betas=BETAS, eps=EPS,
                             weight_decay=WEIGHT_DECAY)


def clip(params):
    """Clip the global gradient norm to CLIP_NORM; returns the norm before clipping."""
    import torch
    return float(torch.nn.utils.clip_grad_norm_(params, CLIP_NORM))


# --------------------------------------------------------------------------- #
# probe
# --------------------------------------------------------------------------- #
def probe_verdict(losses):
    """The probe's outcome over the per-step losses of PROBE_STEPS optimizer steps: passes unless
    a loss is not finite or the mean of the last PROBE_WINDOW steps exceeds that of the first."""
    if len(losses) != PROBE_STEPS:
        raise ValueError(f"{len(losses)} losses, the probe is {PROBE_STEPS} steps")
    if not all(math.isfinite(x) for x in losses):
        return {"passes": False, "reason": "a loss is not finite"}
    first = sum(losses[:PROBE_WINDOW]) / PROBE_WINDOW
    last = sum(losses[-PROBE_WINDOW:]) / PROBE_WINDOW
    return {"passes": last <= first, "mean_first": first, "mean_last": last,
            "reason": None if last <= first else "the loss rose over the probe"}


def settings():
    """The recipe's values, for the training log and the run manifest."""
    return {k: v for k, v in globals().items()
            if k.isupper() and isinstance(v, (int, float, str, bool, tuple, dict))}
