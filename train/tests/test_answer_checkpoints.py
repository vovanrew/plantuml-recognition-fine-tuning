"""Tests for train/answer_checkpoints.sh, with a stand-in for the Python it runs.

Run from fine-tune/:
  .venv/bin/python train/tests/test_answer_checkpoints.py
(or with pytest: .venv/bin/python -m pytest <this file>)
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

FT = Path(__file__).resolve().parents[2]
SCRIPT = FT / "train" / "answer_checkpoints.sh"
STEPS = ["step_00032", "step_00063", "step_00095", "step_00126"]

# Stands in for the venv's python: records each call, holds dev_scoring's for a moment.
FAKE_PY = """#!/bin/bash
echo "$(date +%s.%N) start $*" >> "$CALLS"
if [ "$1" = train/dev_scoring.py ]; then sleep 1; fi
echo "$(date +%s.%N) end $*" >> "$CALLS"
"""


def run(at_once, steps=STEPS, make_dev=()):
    tmp = Path(tempfile.mkdtemp())
    checkout = tmp / "work" / "fine-tune"
    for s in steps:
        (checkout / "runs/train/2B_1000" / s).mkdir(parents=True)
    for s in make_dev:
        (checkout / "runs/dev/2B_1000" / s).mkdir(parents=True)
    fake = tmp / "python"
    fake.write_text(FAKE_PY)
    fake.chmod(0o755)
    calls = tmp / "calls.txt"
    done = subprocess.run(["bash", str(SCRIPT), "2B", "runs/train/2B_1000", "runs/dev/2B_1000", str(at_once), *steps],
                          cwd=checkout, env=dict(os.environ, PY=str(fake), CALLS=str(calls)),
                          capture_output=True, text=True, timeout=60)
    lines = calls.read_text().splitlines() if calls.exists() else []
    return done, lines, tmp / "work" / "results" / "logs"


def answers(lines):
    return [l.split() for l in lines if "train/dev_scoring.py" in l]


def test_each_checkpoint_is_answered_on_its_own_port_then_its_manifest_written():
    done, lines, logs = run(4)
    assert done.returncode == 0 and done.stdout.strip().endswith("all done"), done.stdout + done.stderr
    starts = [a for a in answers(lines) if a[1] == "start"]
    ports = sorted(a[a.index("--port") + 1] for a in starts)
    assert ports == ["8000", "8002", "8003", "8004"]
    for s in STEPS:
        assert any(a[a.index("--adapter") + 1].endswith(f"/runs/train/2B_1000/{s}") for a in starts)
        assert any(f"start infer/manifest.py write runs/dev/2B_1000/{s}/answers" in l for l in lines)
        assert (logs / f"answer_{s}.log").read_text().strip().endswith("exit 0")
    assert all(a[a.index("--timeout") + 1] == "600" for a in starts)


def test_no_more_servers_at_once_than_asked():
    done, lines, _ = run(2)
    assert done.returncode == 0
    running, most = 0, 0
    for a in answers(sorted(lines, key=lambda l: float(l.split()[0]))):
        running += 1 if a[1] == "start" else -1
        most = max(most, running)
    assert most == 2, most


def test_a_missing_checkpoint_or_an_existing_dev_folder_starts_nothing():
    for kwargs, words in (({"make_dev": ["step_00063"]}, "never written over"),):
        done, lines, _ = run(4, **kwargs)
        assert done.returncode == 2 and words in done.stdout and not lines, done.stdout
    tmp = Path(tempfile.mkdtemp())
    (tmp / "work" / "fine-tune").mkdir(parents=True)
    done = subprocess.run(["bash", str(SCRIPT), "2B", "runs/train/2B_1000", "runs/dev/2B_1000", "4", "step_00032"],
                          cwd=tmp / "work" / "fine-tune", capture_output=True, text=True, timeout=60)
    assert done.returncode == 2 and "not found" in done.stdout


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print("ok  ", name, flush=True)
        except BaseException as exc:  # noqa: BLE001
            failed += 1
            print("FAIL", name, f"{type(exc).__name__}: {exc}", flush=True)
    print(f"{len(tests) - failed} of {len(tests)} passed")
    sys.exit(1 if failed else 0)
