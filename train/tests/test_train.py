"""Tests for train/train.py.

Run from fine-tune/:
  .venv/bin/python train/tests/test_train.py               without weights (the 2B processor only)
  HF_DEACTIVATE_ASYNC_LOAD=1 .venv/bin/python train/tests/test_train.py --live
      also trains Qwen3.5-2B from .local/hf for 2 optimizer steps of 2 pairs on this machine
      (reference path of the recurrent layers), with a checkpoint after each
(or with pytest: .venv/bin/python -m pytest <this file>; the live part runs as a script only)
"""
import argparse
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path

FT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", str(FT / ".local/hf"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
sys.path.insert(0, str(FT / "train"))
sys.path.insert(0, str(FT / "build"))
import train as T  # noqa: E402

MODEL, REVISION = "Qwen/Qwen3.5-2B", "15852e8c16360a2fea060d615a32b45270f8a8fc"
# the recipe's pinned orders of the 1,000-pair run (train/tests/test_recipe.py)
ORDERS_1000 = ("4a74ce4bb0c525a3", "86e753c8ff232fbd")
_cache = {}


def rows():
    if "rows" not in _cache:
        _cache["rows"] = T.load_rows()
    return _cache["rows"]


def processor():
    if "processor" not in _cache:
        from transformers import AutoProcessor
        _cache["processor"] = AutoProcessor.from_pretrained(MODEL, revision=REVISION)
    return _cache["processor"]


def refusal(argv):
    try:
        T.main(argv)
    except SystemExit as exc:
        return str(exc)
    return None


# --------------------------------------------------------------------------- #
# Without weights
# --------------------------------------------------------------------------- #
def test_the_recipe_is_the_frozen_file():
    assert T.check_recipe() == T.RECIPE_SHA256
    with tempfile.TemporaryDirectory() as tmp:
        other = Path(tmp) / "recipe.py"
        other.write_text((FT / "train/recipe.py").read_text() + "\n# changed\n")
        try:
            T.check_recipe(other)
        except SystemExit as exc:
            assert "recipe-v1" in str(exc)
        else:
            raise AssertionError("a changed recipe was accepted")


def test_the_pilot_is_the_recipes_1000_pair_run_in_the_recipes_order():
    keys, batches = T.run_plan(rows(), 1000)
    assert len(keys) == 1000 and len(batches) == 126 == T.R.total_steps(1000)
    assert [len(b) for b in batches[:63]] == [16] * 62 + [8] and [len(b) for b in batches[63:]] == [16] * 62 + [8]
    assert tuple(T.order_digest(T.R.epoch_order(keys, e)) for e in (0, 1)) == ORDERS_1000
    assert [k for b in batches[:63] for k in b] == T.R.epoch_order(keys, 0)
    assert T.R.checkpoint_steps(1000) == [32, 63, 95, 126]


def test_the_prompt_length_follows_from_the_picture_size():
    """For the 10 largest and 10 smallest pictures of the 20,000 and 10 others, the length computed
    from the picture size equals the length of the rendered turn."""
    proc, prompt = processor(), T.E.prompt_text()
    listed = {r["key"] for r in rows()}
    sizes = {k: v["size"] for k, v in T.E.manifest("train").items() if k in listed}
    by_area = sorted(sizes, key=lambda k: (sizes[k][0] * sizes[k][1], k))
    chosen = by_area[:10] + by_area[-10:] + by_area[5000:20000:1500]
    for key in chosen:
        picture = FT / T.E.picture_path("train", key)
        rendered = T.E.user_turn.render(proc, prompt, T.E.user_turn.open_image(picture))
        assert T.prompt_tokens(proc, prompt, *sizes[key]) == int(rendered["input_ids"].shape[1]), key


def test_the_memory_step_takes_the_longest_sequences():
    proc = processor()
    lengths = T.sequence_lengths(proc, T.E.prompt_text(), rows())
    assert len(lengths) == 20000
    longest = sorted(lengths, key=lambda k: (-lengths[k], k))[:T.MEMORY_PAIRS]
    assert min(lengths[k] for k in longest) >= sorted(lengths.values())[-T.MEMORY_PAIRS]
    print(f"     longest training sequence {lengths[longest[0]]} positions; 16th {lengths[longest[-1]]}")


def test_the_command_refuses_what_it_does_not_run():
    with tempfile.TemporaryDirectory() as tmp:
        assert "--pairs" in (refusal(["probe", "--size", "2B", "--pairs", "1000", "--out", tmp]) or "")
        assert "--pairs" in (refusal(["run", "--size", "2B", "--out", tmp]) or "")
        (Path(tmp) / "x").write_text("x")
        assert "never written over" in (refusal(["run", "--size", "2B", "--pairs", "1000", "--out", tmp]) or "")


def test_on_this_machine_only_the_reference_path_runs():
    device = T.pick_device()
    if device == "cuda":
        return
    assert "CUDA" in (refusal(["probe", "--size", "2B", "--out", tempfile.mkdtemp(), "--device", device]) or "")
    guarded = T.guards(device, "reference")
    assert guarded["recurrent_layers"]["chosen"] == "reference"


# --------------------------------------------------------------------------- #
# With weights (--live)
# --------------------------------------------------------------------------- #
def live():
    import torch
    device = T.pick_device()
    T.guards(device, "reference" if device != "cuda" else "flash-linear-attention")
    proc, model, trainable = T.load("2B", device)
    assert model.training and getattr(model.get_base_model().model.visual, "_recipe_without_grad", False)
    assert all(".lora_" in n for n, p in model.named_parameters() if p.requires_grad)
    by_key = {r["key"]: r for r in rows()}
    keys, batches = T.run_plan(rows(), 1000)
    two = [batches[0][:2], batches[0][2:4]]
    opt = T.R.optimizer(trainable)
    sched = T.R.scheduler(opt, T.R.total_steps(1000))           # the pilot's schedule: 4 warm-up steps
    out = Path(tempfile.mkdtemp())
    try:
        log = out / "log.jsonl"
        losses = T.train_steps(model, proc, trainable, T.E.prompt_text(), by_key, two, opt, sched,
                               device, out, log, checkpoints={1, 2})
        lines = [json.loads(l) for l in log.read_text().splitlines()]
        assert [l["step"] for l in lines] == [1, 2] and all(math.isfinite(x) for x in losses)
        assert [l["lr"] for l in lines] == [1e-4 * T.R.lr_factor(k, 126) for k in (1, 2)] == [2.5e-5, 5e-5]
        assert all(l["grad_norm_before_clip"] > 0 and l["pairs"] == 2 for l in lines)
        ends = len(T.E.end_ids(T.R.END_WITH_NEWLINE))
        for line, batch in zip(lines, two):
            assert line["answer_tokens"] == sum(len(T.E.target_ids(proc.tokenizer, by_key[k]["target"])) + ends
                                                for k in batch)
        saved = []
        for step in (1, 2):
            folder = out / T.R.checkpoint_dir(step)
            assert (folder / "adapter_model.safetensors").is_file() and (folder / "adapter_config.json").is_file()
            saved.append(T.E.sha256_file(folder / "adapter_model.safetensors"))
        assert saved[0] != saved[1]
        state = torch.load(out / "state.pt", weights_only=False)
        assert state["step"] == 2 and state["scheduler"]["last_epoch"] == 2 and state["optimizer"]["state"]
        config = json.loads((out / "step_00002/adapter_config.json").read_text())
        assert (config["r"], config["lora_alpha"], config["lora_dropout"]) == (16, 32, 0.05)
        print(f"ok   live: losses {[round(x, 5) for x in losses]}, norms "
              f"{[round(l['grad_norm_before_clip'], 4) for l in lines]}, seconds {[l['seconds'] for l in lines]}")
    finally:
        shutil.rmtree(out, ignore_errors=True)


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
        except BaseException as exc:  # noqa: BLE001 - SystemExit included
            failed += 1
            print("FAIL", name, f"{type(exc).__name__}: {exc}", flush=True)
    if cli.live:
        try:
            live()
        except BaseException as exc:  # noqa: BLE001
            failed += 1
            print("FAIL live", f"{type(exc).__name__}: {exc}", flush=True)
    print(f"{len(tests) + int(cli.live) - failed} of {len(tests) + int(cli.live)} passed")
    sys.exit(1 if failed else 0)
