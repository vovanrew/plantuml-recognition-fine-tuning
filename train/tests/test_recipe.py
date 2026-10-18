"""Tests for train/recipe.py: the recipe's values, the adapter at the three sizes, the schedule, the
steps and checkpoints, the data order, the loss weighting, the probe.

Run from fine-tune/:
  .venv/bin/python train/tests/test_recipe.py            without weights (models on the meta device)
  HF_DEACTIVATE_ASYNC_LOAD=1 .venv/bin/python train/tests/test_recipe.py --live
      also loads Qwen3.5-2B from .local/hf: the vision tower's output in training under the recipe
      equals the server's, also with a vision weight left trainable, and a training backward
      reaches every adapter tensor and nothing else
(or with pytest: .venv/bin/python -m pytest <this file>)

The live test uses a loss-curve diagram, never the dev set.
"""
import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

FT = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
INSTRUMENT = FT.parent / "plantuml-recognition-evaluation"
os.environ.setdefault("HF_HOME", str(FT / ".local/hf"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
for sub in ("build", "infer", "train"):
    sys.path.insert(0, str(FT / sub))
import recipe as R  # noqa: E402

LORA_PARAMETERS = {"2B": 16_819_200, "9B": 43_278_336, "27B": 116_727_808}   # rank 16
MODULES = {"2B": 186, "9B": 248, "27B": 496}
# sha256 (first 16 hex digits) of the order of each epoch, one key per line; the same under Python
# 3.9.6 and 3.12.13
ORDERS = {20000: ("17cfaaf0132f1e4f", "5ca6958c172bcf79"), 5000: ("7404bde7ef60a055", "f13c825d480909bc"),
          1000: ("4a74ce4bb0c525a3", "86e753c8ff232fbd")}
_cache = {}


def train_keys():
    if "train" not in _cache:
        _cache["train"] = json.load(open(FT / "data/sets/train.json"))["diagrams"]
    return _cache["train"]


def meta_model(size):
    import torch
    from transformers import AutoConfig, AutoModelForImageTextToText
    cfg = AutoConfig.from_pretrained(FIXTURES / "configs" / size)
    with torch.device("meta"):
        return AutoModelForImageTextToText.from_config(cfg)


# --------------------------------------------------------------------------- #
def test_values_are_the_decision():
    assert (R.RANK, R.ALPHA, R.DROPOUT, R.USE_RSLORA) == (16, 32, 0.05, False)
    assert set(R.TARGET_KINDS) == {"gate_proj", "up_proj", "down_proj", "q_proj", "k_proj", "v_proj",
                                   "o_proj", "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b",
                                   "out_proj"} and len(R.TARGET_KINDS) == 12
    assert (R.LEARNING_RATE, R.BETAS, R.EPS, R.WEIGHT_DECAY) == (1e-4, (0.9, 0.999), 1e-8, 0.01)
    assert (R.WARMUP_FRACTION, R.CLIP_NORM, R.PAIRS_PER_STEP) == (0.03, 1.0, 16)
    assert (R.EPOCHS, R.CHECKPOINTS_PER_EPOCH, R.END_WITH_NEWLINE, R.VISION_WITHOUT_GRAD) == (2, 2, False, True)
    assert (R.PROBE_STEPS, R.PROBE_WINDOW) == (60, 10)
    assert R.checkpoint_dir(625) == "step_00625" and R.checkpoint_dir(2500) == "step_02500"
    assert (R.DEV_MAX_TOKENS, R.DEV_TIMEOUT_S) == (5376, 600)
    assert R.RULE_METRICS == ("csr", "element_f1_micro_full")
    settings = R.settings()
    assert settings["LEARNING_RATE"] == 1e-4 and settings["TARGET_KINDS"] == R.TARGET_KINDS


def test_bootstrap_constants_are_the_instruments():
    text = (INSTRUMENT / "analysis/bootstrap.py").read_text(encoding="utf-8")
    assert f"SEED = {R.BOOTSTRAP_SEED} " in text and f"N_RESAMPLES = {R.BOOTSTRAP_RESAMPLES} " in text


def test_the_dev_cap_is_the_scored_runs_cap():
    import dev_scoring
    assert dev_scoring.OUTPUT_CAP == R.DEV_MAX_TOKENS


def test_adapter_targets_and_parameters_at_three_sizes():
    for size in ("2B", "9B", "27B"):
        model = meta_model(size)
        names = R.target_names(model)
        assert len(names) == MODULES[size], (size, len(names))
        assert {n.rsplit(".", 1)[1] for n in names} == set(R.TARGET_KINDS), size
        assert all(n.startswith("model.language_model.") for n in names)
        peft_model = R.attach(model)
        trainable = {n: p for n, p in peft_model.named_parameters() if p.requires_grad}
        assert sum(p.numel() for p in trainable.values()) == LORA_PARAMETERS[size], size
        assert not any(".visual." in n or "lm_head" in n or "embed_tokens" in n for n in trainable)
        config = peft_model.peft_config["default"]
        assert (config.r, config.lora_alpha, config.lora_dropout, config.use_rslora, config.bias) == \
            (16, 32, 0.05, False, "none")


def test_an_unknown_linear_kind_is_refused():
    import torch

    class Inner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = torch.nn.ModuleDict({"mixer": torch.nn.Linear(4, 4)})

    class Fake(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Inner()

    try:
        R.target_names(Fake())
    except ValueError as exc:
        assert "mixer" in str(exc)
    else:
        raise AssertionError("an unknown kind was accepted")


def test_schedule_warm_up_then_constant():
    import torch
    total = R.total_steps(20000)
    assert total == 2500 and R.warmup_steps(total) == 75
    assert R.warmup_steps(R.total_steps(5000)) == 19 and R.warmup_steps(R.total_steps(1000)) == 4
    assert math.isclose(R.lr_factor(1, total), 1 / 75) and R.lr_factor(75, total) == 1.0
    assert all(R.lr_factor(k, total) == 1.0 for k in range(75, total + 1))
    p = torch.nn.Parameter(torch.zeros(1))
    opt = R.optimizer([p])
    sched = R.scheduler(opt, total)
    seen = []
    for _ in range(total):
        seen.append(opt.param_groups[0]["lr"])
        p.grad = torch.ones(1)
        opt.step()
        sched.step()
    expected = [R.LEARNING_RATE * R.lr_factor(k, total) for k in range(1, total + 1)]
    assert all(math.isclose(a, b, rel_tol=1e-12) for a, b in zip(seen, expected))
    assert seen[0] > 0 and seen[74] == R.LEARNING_RATE and seen[-1] == R.LEARNING_RATE


def test_steps_and_checkpoints():
    assert [R.steps_per_epoch(n) for n in (20000, 5000, 1000)] == [1250, 313, 63]
    assert R.checkpoint_steps(20000) == [625, 1250, 1875, 2500]
    assert R.checkpoint_steps(5000) == [157, 313, 470, 626]
    assert R.checkpoint_steps(1000) == [32, 63, 95, 126]


def test_run_rows_and_order():
    rows = train_keys()
    for pairs, top in ((20000, 2500), (5000, 625), (1000, 125)):
        run = R.run_rows(rows, pairs)
        assert len(run) == pairs and max(r["rank"] for r in run) == top
        keys = [r["key"] for r in run]
        for epoch in (0, 1):
            batches = R.epoch_batches(keys, epoch)
            flat = [k for b in batches for k in b]
            assert sorted(flat) == sorted(keys) and len(flat) == pairs
            assert len(batches) == R.steps_per_epoch(pairs)
            assert all(len(b) == 16 for b in batches[:-1]) and len(batches[-1]) == pairs - 16 * (len(batches) - 1)
        digests = tuple(hashlib.sha256("\n".join(R.epoch_order(keys, e)).encode()).hexdigest()[:16]
                        for e in (0, 1))
        assert digests == ORDERS[pairs], (pairs, digests)
    keys = [r["key"] for r in R.run_rows(rows, 20000)]
    assert len(R.epoch_batches(keys, 0)[-1]) == 16
    assert keys[:3] != R.epoch_order(keys, 0)[:3]      # the file order is by cell; the order is shuffled
    try:
        R.run_rows(rows[:100], 20000)
    except ValueError:
        pass
    else:
        raise AssertionError("a short list was accepted")


def test_loss_scale_is_the_mean_over_the_steps_answer_tokens():
    counts = [13, 328, 59]
    sums = [13 * 1.5, 328 * 0.5, 59 * 2.0]
    scale = R.loss_scale(counts)
    assert math.isclose(scale * sum(sums), sum(sums) / sum(counts))
    try:
        R.loss_scale([0, 0])
    except ValueError:
        pass
    else:
        raise AssertionError("a step without answer tokens was accepted")


def test_probe_runs_at_the_full_rate_on_the_first_960_pairs():
    import torch
    keys = [r["key"] for r in R.run_rows(train_keys(), 20000)]
    batches = R.probe_batches(keys)
    assert len(batches) == 60 and all(len(b) == 16 for b in batches)
    assert [k for b in batches for k in b] == R.epoch_order(keys, 0)[:960]
    assert R.warmup_steps(R.PROBE_STEPS) == 2
    p = torch.nn.Parameter(torch.zeros(1))
    opt = R.optimizer([p])
    sched = R.scheduler(opt, R.PROBE_STEPS)
    seen = []
    for _ in range(R.PROBE_STEPS):
        seen.append(opt.param_groups[0]["lr"])
        p.grad = torch.ones(1)
        opt.step()
        sched.step()
    assert seen[0] == R.LEARNING_RATE / 2 and all(v == R.LEARNING_RATE for v in seen[1:])
    try:
        R.probe_batches(keys[:5000])
    except ValueError:
        pass
    else:
        raise AssertionError("a probe over a subset was accepted")


def test_probe_verdict():
    falling = [2.0 - 0.02 * k for k in range(60)]
    assert R.probe_verdict(falling)["passes"]
    assert not R.probe_verdict(falling[::-1])["passes"]
    assert not R.probe_verdict(falling[:59] + [float("nan")])["passes"]
    assert R.probe_verdict([1.0] * 60)["passes"]           # equal means pass: not a rise
    try:
        R.probe_verdict(falling[:59])
    except ValueError:
        pass
    else:
        raise AssertionError("a probe of 59 steps was accepted")


def test_optimizer_and_clipping():
    import torch
    a, b = torch.nn.Parameter(torch.zeros(3)), torch.nn.Parameter(torch.zeros(4))
    opt = R.optimizer([a, b])
    g = opt.param_groups[0]
    assert (g["lr"], g["betas"], g["eps"], g["weight_decay"]) == (1e-4, (0.9, 0.999), 1e-8, 0.01)
    a.grad, b.grad = torch.tensor([3.0, 0.0, 0.0]), torch.tensor([4.0, 0.0, 0.0, 0.0])
    before = R.clip([a, b])
    after = float(torch.cat([a.grad, b.grad]).norm())
    assert math.isclose(before, 5.0, rel_tol=1e-6) and math.isclose(after, 1.0, rel_tol=1e-5)


# --------------------------------------------------------------------------- #
def live():
    """Qwen3.5-2B in bf16 on this machine's device: the vision tower's output under the recipe in
    training (gradients on) equals the server's (inference mode), and a training backward under
    gradient checkpointing gives every adapter tensor a gradient and no other parameter one."""
    import torch
    import examples as E
    import loss as L
    import server
    from transformers import AutoProcessor
    model_id, revision = "Qwen/Qwen3.5-2B", "15852e8c16360a2fea060d615a32b45270f8a8fc"
    device = server.pick_device()
    processor = AutoProcessor.from_pretrained(model_id, revision=revision)
    row = E.load_examples(FT / "data/examples/loss_curve.jsonl")[0]
    ids = E.target_ids(processor.tokenizer, row["target"])
    inputs, labels, n_prompt = E.example(processor, E.prompt_text(), E.user_turn.open_image(FT / row["picture"]),
                                         ids, E.end_ids(R.END_WITH_NEWLINE))
    model = server.load_model(model_id, revision, device)
    x = {k: (v.to(torch.bfloat16) if v.is_floating_point() else v).to(device) for k, v in inputs.items()}
    captured = []

    def tensors(out):
        if torch.is_tensor(out):
            return [out]
        if isinstance(out, (list, tuple)):
            return [t for o in out for t in tensors(o)]
        if hasattr(out, "values"):
            return [t for o in out.values() for t in tensors(o)]
        return []

    visual = model.model.visual
    hook = visual.register_forward_hook(lambda m, i, o: captured.append([t.detach().float().cpu() for t in tensors(o)]))

    with torch.inference_mode():                                     # the server's path
        model(**x, use_cache=False)
    served = captured.pop()

    peft_model = R.attach(model)
    model.gradient_checkpointing_enable()
    peft_model.train()
    vision_params = list(visual.parameters())

    def vision_output(trainable_vision):
        for q in vision_params:
            q.requires_grad_(trainable_vision)
        with torch.enable_grad():
            peft_model(**x, use_cache=False)
        for q in vision_params:
            q.requires_grad_(False)
        return captured.pop()

    def same(a, b):
        return len(a) == len(b) > 0 and all(torch.equal(p, q) for p, q in zip(a, b))

    def rel(a, b):
        return max(float((p - q).norm() / p.norm()) for p, q in zip(a, b))

    frozen_unwrapped = vision_output(False)          # the adapter froze the vision weights
    trainable_unwrapped = vision_output(True)        # a vision weight left trainable by mistake
    R.vision_without_grad(peft_model)
    R.vision_without_grad(peft_model)                # a second call changes nothing
    trained = vision_output(False)                   # the recipe
    trainable_wrapped = vision_output(True)
    hook.remove()
    print(f"     vision output against the served one, relative difference: frozen, no wrapper "
          f"{rel(served, frozen_unwrapped):.4g}; a trainable vision weight, no wrapper "
          f"{rel(served, trainable_unwrapped):.4g}; the recipe {rel(served, trained):.4g}; a trainable "
          f"vision weight under the wrapper {rel(served, trainable_wrapped):.4g}")
    assert same(served, trained) and same(served, trainable_wrapped)
    assert not any(q.requires_grad for q in vision_params)

    peft_model.zero_grad(set_to_none=True)
    total, n = L.backward(peft_model, x, n_prompt, scale=R.loss_scale([int((labels[0] != -100).sum())]))
    assert math.isfinite(total) and n == len(ids) + 1
    lora = [(name, p) for name, p in peft_model.named_parameters() if p.requires_grad]
    assert len(lora) == 2 * MODULES["2B"]
    with_grad = [name for name, p in lora if p.grad is not None and float(p.grad.abs().sum()) > 0]
    others = [name for name, p in peft_model.named_parameters() if not p.requires_grad and p.grad is not None]
    print(f"     backward: loss {total / n:.5f} over {n} answer tokens; adapter tensors with a gradient "
          f"{len(with_grad)} of {len(lora)}; other parameters with a gradient {len(others)}")
    # lora_A receives no gradient while lora_B is zero (PEFT's initialisation): every B must have one
    assert all(name in with_grad for name, _ in lora if "lora_B" in name) and not others


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    cli = ap.parse_args()
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print("ok  ", name, flush=True)
        except BaseException as exc:  # noqa: BLE001
            failed += 1
            print("FAIL", name, f"{type(exc).__name__}: {exc}", flush=True)
    if cli.live:
        try:
            live()
            print("ok   live", flush=True)
        except BaseException as exc:  # noqa: BLE001
            failed += 1
            print("FAIL live", f"{type(exc).__name__}: {exc}", flush=True)
    print(f"{len(tests) + int(cli.live) - failed} of {len(tests) + int(cli.live)} passed")
    sys.exit(1 if failed else 0)
