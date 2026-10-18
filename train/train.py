#!/usr/bin/env python3
"""The study's training script: one run of the recipe on one card.

decisions/pilot-run/record.md. The recipe is train/recipe.py, frozen as tag `recipe-v1`; its sha256
is checked before anything else (RECIPE_SHA256), and nothing here decides what the recipe decides.
For each optimizer step: the step's pairs (recipe.epoch_batches), each built by build/examples.py
as the server sees it (picture, newline, prompt; then the target and `<|im_end|>`) and
back-propagated one at a time through train/loss.py with the step's weight (recipe.loss_scale);
then clipping (the norm before clipping is logged), the optimizer step and the schedule step.

Modes:
  run      a run of --pairs 20000, 5000 or 1000, 2 epochs; the adapter saved after each step of
           recipe.checkpoint_steps in <out>/step_NNNNN/, the optimizer, schedule and random state
           in <out>/state.pt with the last. The pilot: --size 2B --pairs 1000.
  probe    the recipe's probe: 60 steps over the first 960 pairs of the 20,000's epoch-0 order
           under recipe.scheduler(opt, 60), judged by recipe.probe_verdict into <out>/probe.json;
           no adapter is kept.
  memory   one optimizer step over the 16 pairs of the 20,000 with the longest sequences, the
           peak memory of each pair and of the step into <out>/memory.json; no adapter is kept.

Before any weight loads, as infer/server.py: build/card_environment.py `require(device)` (the
library versions of requirements-card.txt; on the Mac the core packages) and
build/recurrent_layers.py `require(--recurrent-layers, device)` (default flash-linear-attention,
which needs CUDA; on the Mac `--recurrent-layers reference`, for local checks only).

The script's own choices, for memory, at every size: gradient checkpointing (non-reentrant), and
the loss computed in chunks of 512 answer positions (train/loss.py). The inputs go to the device as
the server sends them (no change of dtype).

Outputs in --out, which must not exist yet: run.json (what ran: the recipe's values and sha256,
the code's sha256, the examples' checksums, model and revision, environment, recurrent layers,
card, study commit, the orders' digests), log.jsonl (one line per optimizer step), and the files
of the mode.

Usage (from fine-tune/; HF_HOME holds the models at the pinned commits):
  python train/train.py run --size 2B --pairs 1000 --out runs/train/2B_1000
  python train/train.py probe --size 27B --out runs/probe/27B
  python train/train.py memory --size 27B --out runs/memory/27B

Tests: train/tests/test_train.py
"""
import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
FT = HERE.parent
for _p in (HERE, FT / "build"):
    sys.path.insert(0, str(_p))
import examples as E  # noqa: E402
import loss as L  # noqa: E402
import recipe as R  # noqa: E402

RECIPE_SHA256 = "1b21ee36ba45fc56cdbb9c9cc394aef462de88af6a623de54e3a75d8ce778596"   # tag recipe-v1
EXAMPLES = FT / "data" / "examples"
MEMORY_PAIRS = 16
CODE = ("train/train.py", "train/recipe.py", "train/loss.py", "train/models.py", "build/examples.py",
        "build/user_turn.py", "build/recurrent_layers.py", "build/card_environment.py",
        "requirements-card.txt")


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def check_recipe(path=HERE / "recipe.py"):
    got = E.sha256_file(path)
    if got != RECIPE_SHA256:
        raise SystemExit(f"{path} has sha256 {got}; the frozen recipe (tag recipe-v1) is {RECIPE_SHA256}")
    return got


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def load_rows(folder=EXAMPLES):
    """The 20,000 examples in data/sets/train.json order, after their checksum."""
    sums = dict(reversed(line.split("  ", 1)) for line in (folder / "SHA256SUMS").read_text().splitlines())
    got = E.sha256_file(folder / "train.jsonl")
    if got != sums.get("train.jsonl"):
        raise SystemExit(f"{folder / 'train.jsonl'} has sha256 {got}, SHA256SUMS says {sums.get('train.jsonl')}")
    return E.load_examples(folder / "train.jsonl")


def order_digest(keys):
    """sha256 (first 16 hex digits) of keys one per line, as train/tests/test_recipe.py pins them."""
    return hashlib.sha256("\n".join(keys).encode()).hexdigest()[:16]


def run_plan(rows, pairs):
    """(keys of the run in list order, optimizer steps over both epochs as lists of keys)."""
    keys = [r["key"] for r in R.run_rows(rows, pairs)]
    return keys, [batch for epoch in range(R.EPOCHS) for batch in R.epoch_batches(keys, epoch)]


def _pixels_ok(row):
    return E.pixels_sha256(FT / row["picture"]) == row["pixels_sha256"]


def check_pictures(rows, workers=8):
    """Every picture of `rows` hashes as the render manifest recorded it."""
    with ProcessPoolExecutor(workers) as pool:
        ok = list(pool.map(_pixels_ok, rows, chunksize=32))
    wrong = [r["key"] for r, good in zip(rows, ok) if not good]
    if wrong:
        raise SystemExit(f"{len(wrong)} pictures differ from the render manifest, e.g. {wrong[:3]}")
    return len(rows)


# --------------------------------------------------------------------------- #
# sequence lengths from the picture sizes
# --------------------------------------------------------------------------- #
def image_tokens(processor, width, height):
    """Positions the picture takes in the prompt: the processor's resize to multiples of
    patch × merge, then one position per merge × merge patches."""
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
    ip = processor.image_processor
    factor = ip.patch_size * ip.merge_size
    h, w = smart_resize(height, width, factor=factor, min_pixels=ip.size.shortest_edge,
                        max_pixels=ip.size.longest_edge)
    return (h // factor) * (w // factor)


def prompt_tokens(processor, prompt, width, height, _cache={}):
    """Length of the user turn for a picture of this size: the text's positions (measured once on a
    small picture) plus the picture's."""
    if "text" not in _cache:
        from PIL import Image
        n = int(E.user_turn.render(processor, prompt, Image.new("RGB", (512, 512)))["input_ids"].shape[1])
        _cache["text"] = n - image_tokens(processor, 512, 512)
    return _cache["text"] + image_tokens(processor, width, height)


def sequence_lengths(processor, prompt, rows):
    """{key: positions of the whole training sequence}: prompt, target, end tokens."""
    sizes = {k: v["size"] for k, v in E.manifest("train").items() if v["status"] == "ok"}
    ends = len(E.end_ids(R.END_WITH_NEWLINE))
    return {r["key"]: prompt_tokens(processor, prompt, *sizes[r["key"]])
            + len(E.target_ids(processor.tokenizer, r["target"])) + ends for r in rows}


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
def pick_device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def guards(device, layers):
    """The server's two guards, before any weight loads."""
    import card_environment
    import recurrent_layers
    return {"environment": card_environment.require(device),
            "recurrent_layers": recurrent_layers.require(layers, device)}


def load(size, device):
    """(processor, model with the recipe's adapter, trainable parameters), the model in training
    mode with gradient checkpointing."""
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from models import SIZES
    model_id, revision = SIZES[size]
    processor = AutoProcessor.from_pretrained(model_id, revision=revision)
    model = AutoModelForImageTextToText.from_pretrained(model_id, revision=revision,
                                                        device_map={"": device}, dtype=torch.bfloat16)
    model = R.attach(model)
    R.vision_without_grad(model)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()                                   # the adapter's dropout is active in training
    trainable = [p for p in model.parameters() if p.requires_grad]
    return processor, model, trainable


def memory(device):
    import torch
    if device != "cuda":
        return None
    return {"peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
            "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 1e9, 3),
            "card_total_gb": round(torch.cuda.mem_get_info()[1] / 1e9, 3)}


def reset_peak(device):
    import torch
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()


# --------------------------------------------------------------------------- #
# steps
# --------------------------------------------------------------------------- #
def train_steps(model, processor, trainable, prompt, rows_by_key, batches, opt, sched, device,
                out, log, checkpoints=(), first_step=1, per_pair=None):
    """Optimizer steps over `batches` (lists of keys). Appends one line per step to `log`, saves
    the adapter after each step in `checkpoints`, the state with the last of them. Returns the
    per-step mean losses."""
    import torch
    ends = E.end_ids(R.END_WITH_NEWLINE)
    tok = processor.tokenizer
    losses = []
    last_checkpoint = max(checkpoints) if checkpoints else None
    for step, batch in enumerate(batches, start=first_step):
        t0 = time.time()
        reset_peak(device)
        ids = {k: E.target_ids(tok, rows_by_key[k]["target"]) for k in batch}
        counts = [len(ids[k]) + len(ends) for k in batch]
        scale = R.loss_scale(counts)
        lr = opt.param_groups[0]["lr"]
        summed, positions = 0.0, 0
        for k in batch:
            row = rows_by_key[k]
            inputs, _, n_prompt = E.example(processor, prompt, E.user_turn.open_image(FT / row["picture"]),
                                            ids[k], ends)
            inputs = {name: v.to(device) for name, v in inputs.items()}
            total, n = L.backward(model, inputs, n_prompt, scale)
            summed += total
            positions += int(inputs["input_ids"].shape[1])
            if per_pair is not None:
                per_pair.append({"key": k, "positions": int(inputs["input_ids"].shape[1]),
                                 "answer_tokens": n, "memory": memory(device)})
            del inputs
        norm = R.clip(trainable)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        mean = summed / sum(counts)
        losses.append(mean)
        line = {"step": step, "pairs": len(batch), "answer_tokens": sum(counts),
                "positions": positions, "loss": mean, "grad_norm_before_clip": norm, "lr": lr,
                "seconds": round(time.time() - t0, 3), "memory": memory(device)}
        with open(log, "a", encoding="utf-8") as f:
            f.write(json.dumps(line) + "\n")
        if not math.isfinite(mean):
            raise SystemExit(f"step {step}: the loss is {mean}")
        if step in checkpoints:
            folder = Path(out) / R.checkpoint_dir(step)
            model.save_pretrained(str(folder))
            if step == last_checkpoint:
                state = {"step": step, "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                         "torch_rng": torch.get_rng_state(),
                         "cuda_rng": torch.cuda.get_rng_state_all() if device == "cuda" else None}
                torch.save(state, Path(out) / "state.pt")
    return losses


# --------------------------------------------------------------------------- #
# provenance
# --------------------------------------------------------------------------- #
def study_commit():
    head = subprocess.run(["git", "-C", str(FT), "rev-parse", "HEAD"], capture_output=True, text=True)
    dirty = subprocess.run(["git", "-C", str(FT), "status", "--porcelain", "--", *CODE],
                           capture_output=True, text=True)
    return {"commit": head.stdout.strip() or None,
            "changed_code_files": [l[3:] for l in dirty.stdout.splitlines()]}


def run_record(args, device, guarded, extra):
    import torch
    from models import SIZES
    return {"mode": args.mode, "size": args.size, "model": SIZES[args.size][0],
            "revision": SIZES[args.size][1], "device": device,
            "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
            **guarded, "recipe": R.settings(), "recipe_sha256": RECIPE_SHA256,
            "code_sha256": {c: E.sha256_file(FT / c) for c in CODE},
            "examples_sha256sums": (EXAMPLES / "SHA256SUMS").read_text().splitlines(),
            "prompt_sha256": E.PROMPT_SHA256, "gradient_checkpointing": "non-reentrant",
            "loss_chunk": L.CHUNK, "study": study_commit(), "started": now(),
            **extra}


def write(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)


# --------------------------------------------------------------------------- #
# modes
# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "probe", "memory"])
    ap.add_argument("--size", required=True, choices=["2B", "9B", "27B"])
    ap.add_argument("--pairs", type=int, choices=sorted(R.SUBSET_PAIRS), default=None,
                    help="run mode: the run's pairs (the pilot: 1000)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default=None, choices=["cuda", "mps", "cpu"])
    ap.add_argument("--recurrent-layers", default="flash-linear-attention",
                    choices=["flash-linear-attention", "reference"])
    ap.add_argument("--workers", type=int, default=8, help="processes for the picture check")
    args = ap.parse_args(argv)
    if (args.mode == "run") != (args.pairs is not None):
        raise SystemExit("--pairs is given in run mode and only there")
    check_recipe()
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty; a run is never written over")
    device = args.device or pick_device()
    guarded = guards(device, args.recurrent_layers)
    rows = load_rows()
    by_key = {r["key"]: r for r in rows}
    out.mkdir(parents=True, exist_ok=True)
    log = out / "log.jsonl"

    if args.mode == "run":
        keys, batches = run_plan(rows, args.pairs)
        total = R.total_steps(args.pairs)
        assert len(batches) == total
        checked = check_pictures([by_key[k] for k in keys], args.workers)
        extra = {"pairs": args.pairs, "steps": total, "checkpoint_steps": R.checkpoint_steps(args.pairs),
                 "order_digests": [order_digest(R.epoch_order(keys, e)) for e in range(R.EPOCHS)],
                 "pictures_checked": checked}
    elif args.mode == "probe":
        keys = [r["key"] for r in R.run_rows(rows, 20000)]
        batches, total = R.probe_batches(keys), R.PROBE_STEPS
        checked = check_pictures([by_key[k] for b in batches for k in b], args.workers)
        extra = {"steps": total, "pictures_checked": checked,
                 "probe_order_digest": order_digest([k for b in batches for k in b])}
    else:
        from transformers import AutoProcessor
        from models import SIZES
        model_id, revision = SIZES[args.size]
        processor = AutoProcessor.from_pretrained(model_id, revision=revision)
        lengths = sequence_lengths(processor, E.prompt_text(), rows)
        longest = sorted(lengths, key=lambda k: (-lengths[k], k))[:MEMORY_PAIRS]
        batches, total = [longest], 1
        checked = check_pictures([by_key[k] for k in longest], args.workers)
        extra = {"steps": 1, "pictures_checked": checked,
                 "longest_positions": [lengths[k] for k in longest]}
    record = run_record(args, device, guarded, extra)
    write(out / "run.json", record)

    processor, model, trainable = load(args.size, device)
    opt = R.optimizer(trainable)
    sched = R.scheduler(opt, total)
    per_pair = [] if args.mode == "memory" else None
    checkpoints = set(R.checkpoint_steps(args.pairs)) if args.mode == "run" else set()
    losses = train_steps(model, processor, trainable, E.prompt_text(), by_key, batches, opt, sched,
                         device, out, log, checkpoints=checkpoints, per_pair=per_pair)
    if args.mode == "probe":
        write(out / "probe.json", {"losses": losses, **R.probe_verdict(losses)})
    if args.mode == "memory":
        write(out / "memory.json", {"pairs": per_pair, "step": memory(device)})
    record["finished"] = now()
    write(out / "run.json", record)
    print(json.dumps({"mode": args.mode, "steps": len(losses), "last_loss": losses[-1]}), flush=True)


if __name__ == "__main__":
    main()
