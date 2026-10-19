"""Tests for train/dev_scoring.py.

Run from fine-tune/:
  .venv/bin/python train/tests/test_dev_scoring.py            without weights
  HF_DEACTIVATE_ASYNC_LOAD=1 .venv/bin/python train/tests/test_dev_scoring.py --live [--adapter <dir>]
      also answers 2 diagrams with Qwen3.5-2B from .local/hf through the server and scores them;
      with --adapter, with that adapter attached, after its adapter check
(or with pytest: .venv/bin/python -m pytest <this file>)

The tests use diagrams of the loss-curve split, never the dev set: no dev-set answer exists
before the recipe's checkpoint rule is frozen.
"""
import argparse
import http.server
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
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


def test_relative_paths_score_as_absolute_ones():
    """The runners run from the benchmark's folder; relative paths given from elsewhere still
    name the answers here, and nothing is written into the benchmark's folder."""
    before = sorted(p.name for p in D.INSTRUMENT.iterdir())
    with tempfile.TemporaryDirectory() as tmp:
        set_file, rows = loss_curve_set(tmp, 4)
        answers = Path(tmp) / "answers"
        answers.mkdir()
        for r in rows[:3]:
            shutil.copy(D.GT_DIR / r["key"], answers / r["key"])
        cwd = os.getcwd()
        os.chdir(tmp)
        try:
            result = D.score(score_args("answers", "scores", "set.json"))
        finally:
            os.chdir(cwd)
        assert result["csr"] == {"n": 4, "compiled": 3, "csr": 0.75}, result["csr"]
        assert (Path(tmp) / "scores" / "scores.json").is_file()
    assert sorted(p.name for p in D.INSTRUMENT.iterdir()) == before


def test_the_adapter_check_pictures_are_three_rank_1_training_pairs():
    pictures = D.check_pictures()
    rows = {FT / r["picture"]: r for r in E.load_examples(FT / "data/examples/train.jsonl")}
    assert len(pictures) == D.ADAPTER_CHECK_PICTURES == 3
    assert all(rows[p]["rank"] == 1 and p.is_file() for p in pictures)
    assert len({(rows[p]["primary_type"], rows[p]["tier"]) for p in pictures}) == 3
    assert all("/render/train/" in str(p) for p in pictures)


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def answer_argv(out, port, adapter=None):
    argv = ["answer", "--size", "2B", "--out", str(out), "--timeout", "600", "--port", str(port)]
    return argv + (["--adapter", str(adapter)] if adapter else [])


def test_a_port_in_use_is_refused_before_anything_starts():
    with tempfile.TemporaryDirectory() as tmp, socket.socket() as taken:
        taken.bind(("127.0.0.1", 0))
        taken.listen()
        port = taken.getsockname()[1]
        try:
            D.main(answer_argv(Path(tmp) / "run", port))
        except SystemExit as exc:
            assert "in use" in str(exc), exc
        else:
            raise AssertionError("a port in use was accepted")
        assert not (Path(tmp) / "run").exists()


def test_a_failing_adapter_check_stops_before_any_answer():
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 1)

    def popen(*a, **k):
        raise AssertionError("the server was started after a failed adapter check")

    saved = D.subprocess.run, D.subprocess.Popen
    D.subprocess.run, D.subprocess.Popen = run, popen
    try:
        with tempfile.TemporaryDirectory() as tmp:
            try:
                D.main(answer_argv(Path(tmp) / "run", free_port(), adapter=Path(tmp) / "step_00032"))
            except SystemExit as exc:
                assert "adapter check failed" in str(exc), exc
            else:
                raise AssertionError("answers went ahead after a failed adapter check")
            assert not (Path(tmp) / "run" / "answers").exists()
    finally:
        D.subprocess.run, D.subprocess.Popen = saved
    assert len(calls) == 1 and str(D.ADAPTER_CHECK) in calls[0]
    assert calls[0][calls[0].index("--pictures") + 1:][:3] == [str(p) for p in D.check_pictures()]


class _Alive:
    returncode = None

    def poll(self):
        return None


def serve_once(body):
    """A server on a free port that answers every GET with `body`."""
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    httpd = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def test_a_port_that_answers_as_another_server_is_refused():
    for body, words in ((b"<html>nginx</html>", "something other"),
                        (json.dumps({"served_name": "Qwen/Qwen3.5-2B"}).encode(), "serves")):
        httpd = serve_once(body)
        try:
            D.wait_for(httpd.server_address[1], _Alive(), 5, "Qwen3.5-2B-adapter")
        except SystemExit as exc:
            assert words in str(exc), exc
        else:
            raise AssertionError(f"{body[:20]!r} was taken for the study's server")
        finally:
            httpd.shutdown()
    httpd = serve_once(json.dumps({"served_name": "Qwen3.5-2B-adapter"}).encode())
    try:
        assert D.wait_for(httpd.server_address[1], _Alive(), 5, "Qwen3.5-2B-adapter")["served_name"] == "Qwen3.5-2B-adapter"
    finally:
        httpd.shutdown()


def live(adapter=None):
    """The 2B answers 2 loss-curve diagrams through the server, the base model alone or with
    `adapter` attached (after its adapter check); the answers are scored."""
    with tempfile.TemporaryDirectory() as tmp:
        set_file, rows = loss_curve_set(tmp, 2)
        out = Path(tmp) / "run"
        argv = ["answer", "--size", "2B", "--out", str(out), "--timeout", "900",
                "--max-tokens", "256", "--port", "8791", "--set-file", str(set_file),
                "--images", str(FT / "data/sets/render/loss_curve/puml_images_1568"),
                "--recurrent-layers", "reference"]
        manifest = D.main(argv + (["--adapter", str(adapter)] if adapter else []))
        assert manifest["answers_stored"] == 2 and manifest["runner_exit_code"] == 0, manifest
        server = manifest["server"]
        assert server["revision"] == "15852e8c16360a2fea060d615a32b45270f8a8fc"
        if adapter:
            assert server["adapter"]["attached"] == "not merged" and manifest["adapter_check"]["passed"] is True
            assert (out / "adapter_check.json").is_file()
        else:
            assert server["adapter"] is None and manifest["adapter_check"] is None
        assert server["recurrent_layers"]["chosen"] == "reference"
        assert server["user_turn"]["rendered_sha256_frozen"] == E.user_turn.RENDERED_SHA256
        result = D.score(score_args(out / "answers", out / "scores", set_file))
        assert result["csr"]["n"] == 2 and result["answer_manifest"]["server"] == server
        print("ok   live:", json.dumps({"adapter": bool(adapter), "csr": result["csr"],
                                        "element_f1": result["element_f1"]}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--adapter", default=None, help="with --live: an adapter folder to answer with")
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
            live(cli.adapter)
        except BaseException as exc:  # noqa: BLE001
            failed += 1
            print("FAIL live", f"{type(exc).__name__}: {exc}", flush=True)
    print(f"{len(tests) + int(cli.live) - failed} of {len(tests) + int(cli.live)} passed")
    sys.exit(1 if failed else 0)
