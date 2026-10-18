"""Tests for infer/adapter_check.py.

Run from fine-tune/:
  .venv/bin/python -m pytest -q infer/tests/test_adapter_check.py            without weights
  HF_DEACTIVATE_ASYNC_LOAD=1 .venv/bin/python infer/tests/test_adapter_check.py --live --adapter <dir>
      the command itself with the 2B on this machine, on two training pictures
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

FT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", str(FT / ".local/hf"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import adapter_check as A  # noqa: E402

TRAINING_PICTURES = [FT / "data/sets/render/train/puml_images_1568" / f"{k}.png"
                     for k in ("4f40264104f318a1a05e37ab691c6238b08a3140", "2d6fac3911b034f6b26293a09f4cb8878aa45c2a")]


def row(differ, change):
    return {"answers_differ": differ, "max_abs_first_logit_change": change}


def test_it_passes_when_the_logits_change_everywhere_and_an_answer_somewhere():
    assert A.verdict([row(True, 3.1), row(False, 0.02)])[1]


def test_an_adapter_that_leaves_one_picture_s_logits_unchanged_fails():
    criteria, passed = A.verdict([row(True, 3.1), row(False, 0.0)])
    assert not passed and not criteria["first_logits_change_on_every_picture"]


def test_an_adapter_that_changes_no_answer_fails():
    criteria, passed = A.verdict([row(False, 0.5), row(False, 0.4)])
    assert not passed and not criteria["answer_changes_on_some_picture"]


def test_no_pictures_fails():
    assert not A.verdict([])[1]


def test_first_difference():
    assert A.first_difference([1, 2, 3], [1, 2, 3]) is None
    assert A.first_difference([1, 2, 3], [1, 9, 3]) == 1
    assert A.first_difference([1, 2], [1, 2, 3]) == 2


def live(adapter):
    out = Path(tempfile.mkdtemp()) / "adapter_check.json"
    cmd = [sys.executable, str(FT / "infer/adapter_check.py"), "--model", "Qwen/Qwen3.5-2B",
           "--revision", "15852e8c16360a2fea060d615a32b45270f8a8fc", "--adapter", str(adapter),
           "--pictures", *map(str, TRAINING_PICTURES), "--max-tokens", "48", "--recurrent-layers", "reference",
           "--out", str(out)]
    done = subprocess.run(cmd, capture_output=True, text=True)
    result = json.loads(out.read_text())
    assert done.returncode == 0 and result["passed"], done.stdout + done.stderr[-2000:]
    adapter_state = result["identity"]["adapter"]
    assert adapter_state["attached"] == "not merged" and adapter_state["merged_layers"] == 0
    assert adapter_state["lora_layers"] > 0
    print("ok   live:", [(Path(r["picture"]).name[:8], r["answers_differ"], r["max_abs_first_logit_change"])
                        for r in result["pictures"]])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--adapter", default=None)
    cli = ap.parse_args()
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failures = 0
    for name, fn in tests:
        try:
            fn()
            print("ok  ", name)
        except Exception as exc:  # noqa: BLE001 - a test runner reports every failure
            failures += 1
            print("FAIL", name, f"{type(exc).__name__}: {exc}")
    if cli.live:
        try:
            live(cli.adapter)
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print("FAIL live", f"{type(exc).__name__}: {exc}")
    print(f"{len(tests) + int(cli.live) - failures} of {len(tests) + int(cli.live)} passed")
    sys.exit(1 if failures else 0)
