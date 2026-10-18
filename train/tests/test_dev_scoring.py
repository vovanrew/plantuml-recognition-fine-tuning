"""Tests for train/dev_scoring.py.

Run from fine-tune/:
  .venv/bin/python train/tests/test_dev_scoring.py            without weights
  HF_DEACTIVATE_ASYNC_LOAD=1 .venv/bin/python train/tests/test_dev_scoring.py --live
      also answers 2 diagrams with Qwen3.5-2B from .local/hf through the server and scores them
(or with pytest: .venv/bin/python -m pytest <this file>)

The tests use diagrams of the loss-curve split, never the dev set: no dev-set answer exists
before the recipe's checkpoint rule is frozen.
"""
import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

FT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", str(FT / ".local/hf"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(FT / "build"))
sys.path.insert(0, str(FT / "train"))
import dev_scoring as D  # noqa: E402
import examples as E  # noqa: E402


def loss_curve_set(folder, n, primary_type="class", tier=2):
    """A runner set file of the first n loss-curve diagrams of one cell."""
    full = E.runner_set("loss_curve")
    full["diagrams"] = [r for r in full["diagrams"]
                        if r["primary_type"] == primary_type and r["tier"] == tier][:n]
    path = Path(folder) / "set.json"
    path.write_text(json.dumps(full), encoding="utf-8")
    return path, full["diagrams"]


def score_args(answers, out, set_file, **changes):
    values = dict(step="score", answers=str(answers), out=str(out), set_file=str(set_file),
                  gt_dir=str(D.GT_DIR), render_jar=str(D.RENDER_JAR), java_home=D.JAVA_HOME,
                  python="/usr/bin/python3")
    values.update(changes)
    return argparse.Namespace(**values)


def test_the_scorer_on_this_machine_is_the_frozen_one():
    scorer = D.check_pins(python="/usr/bin/python3")
    assert scorer["commit"] == D.SCORER_COMMIT and scorer["python"] == "3.9.6"
    assert scorer["java"].startswith('openjdk version "26.0.1"') and "graphviz" in scorer["graphviz"]


def test_a_jar_other_than_the_pinned_one_is_refused():
    with tempfile.TemporaryDirectory() as tmp:
        other = Path(tmp) / "plantuml.jar"
        other.write_bytes(b"not the renderer")
        try:
            D.check_pins(render_jar=other)
        except D.Unpinned as exc:
            assert "pinned" in str(exc)
        else:
            raise AssertionError("an unpinned render JAR was accepted")


def test_the_adapter_is_served_under_another_name():
    assert D.served_name("Qwen/Qwen3.5-2B", None) == "Qwen/Qwen3.5-2B"
    assert D.served_name("Qwen/Qwen3.5-2B", "x") not in ("Qwen/Qwen3.5-2B", "Qwen3.5-2B")


def test_the_sources_scored_as_answers_get_full_marks():
    """Three loss-curve sources given as their own answers: every one compiles and Element F1 is
    1 on both populations. A missing answer counts as not compiled."""
    with tempfile.TemporaryDirectory() as tmp:
        set_file, rows = loss_curve_set(tmp, 4)
        answers = Path(tmp) / "answers"
        answers.mkdir()
        for r in rows[:3]:
            shutil.copy(D.GT_DIR / r["key"], answers / r["key"])
        result = D.score(score_args(answers, Path(tmp) / "scores", set_file))
        assert result["csr"] == {"n": 4, "compiled": 3, "csr": 0.75}, result["csr"]
        assert result["element_f1"]["compiled_only"]["n"] == 3
        assert result["element_f1"]["compiled_only"]["micro"] == 1.0
        assert result["element_f1"]["zeros_for_failed"]["n"] == 4
        assert result["scorer"]["render_jar_sha256"] == D.RENDER_SHA256
        assert result["answer_manifest"] is None


def live():
    """The base 2B answers 2 loss-curve diagrams through the server; the answers are scored."""
    with tempfile.TemporaryDirectory() as tmp:
        set_file, rows = loss_curve_set(tmp, 2)
        out = Path(tmp) / "run"
        manifest = D.main(["answer", "--size", "2B", "--out", str(out), "--timeout", "900",
                           "--max-tokens", "256", "--port", "8791", "--set-file", str(set_file),
                           "--images", str(FT / "data/sets/render/loss_curve/puml_images_1568"),
                           "--recurrent-layers", "reference"])
        assert manifest["answers_stored"] == 2 and manifest["runner_exit_code"] == 0, manifest
        server = manifest["server"]
        assert server["adapter"] is None and server["revision"] == "15852e8c16360a2fea060d615a32b45270f8a8fc"
        assert server["recurrent_layers"]["chosen"] == "reference"
        assert server["user_turn"]["rendered_sha256_frozen"] == E.user_turn.RENDERED_SHA256
        result = D.score(score_args(out / "answers", out / "scores", set_file))
        assert result["csr"]["n"] == 2 and result["answer_manifest"]["server"] == server
        print("ok   live:", json.dumps({"csr": result["csr"], "element_f1": result["element_f1"]}))


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
        except BaseException as exc:  # noqa: BLE001 - SystemExit included: a refusal is a failure here
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
