"""Tests for infer/manifest.py, on run directories made here in the runner's format. No weights.

Run from fine-tune/:  .venv/bin/python -m pytest -q infer/tests/test_manifest.py
                      (or .venv/bin/python infer/tests/test_manifest.py)
"""
import copy
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import manifest as M  # noqa: E402

IDENTITY = {
    "served_name": "Qwen/Qwen3.5-2B", "model": "Qwen/Qwen3.5-2B",
    "revision": "15852e8c16360a2fea060d615a32b45270f8a8fc", "dtype": "bfloat16", "adapter": None,
    "decoding": "greedy", "eos_token_id": [248046, 248044], "chat_template_kwargs": {"enable_thinking": False},
    "user_turn": {"file_sha256": M.user_turn.FILE_SHA256, "rendered_sha256_frozen": M.user_turn.RENDERED_SHA256},
    "timing": False, "device": "cuda", "gpu": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
    "versions": {"torch": "2.14.1"}, "optional_kernels": {"flash-linear-attention": "0.5.2"},
    "recurrent_layers": {"chosen": "flash-linear-attention", "active": "flash-linear-attention"},
}
RUN_META = {"model": "Qwen/Qwen3.5-2B", "prompt": "P", "max_tokens": 5376, "timeout": 600, "temperature": 0,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}


def response(identity=None, rendered=None, finish="stop"):
    served = dict(identity or IDENTITY, rendered_sha256=rendered or M.user_turn.RENDERED_SHA256,
                  min_tokens=0, generate_seconds=1.0, first_token_seconds=0.1)
    return {"model": served["served_name"], "choices": [{"message": {"content": "@startuml\n@enduml"},
                                                          "finish_reason": finish}],
            "usage": {"prompt_tokens": 432, "completion_tokens": 9}, "x_served_by": served}


def run_dir(cells=None, meta=None):
    d = Path(tempfile.mkdtemp())
    (d / "run_meta.json").write_text(json.dumps(meta or RUN_META))
    for i, cell in enumerate(cells if cells is not None else [response(), response(), response()]):
        (d / f"key{i}.json").write_text(json.dumps(cell))
    return d


def failed(d, server=None):
    return {c["check"] for c in M.build(d, M.user_turn.RENDERED_SHA256, server)["checks"] if not c["passed"]}


def test_a_clean_run_passes_and_its_manifest_holds():
    d = run_dir()
    manifest = M.write(d, M.user_turn.RENDERED_SHA256)
    assert manifest["passed"] and manifest["summary"]["outcomes"] == {"ok": 3}
    assert (d / "manifest.json").exists() and M.check(d) == []


def test_a_run_that_mixes_two_configurations_fails():
    for field, value in (("revision", "0" * 40), ("optional_kernels", {"flash-linear-attention": None}),
                         ("gpu", "NVIDIA A100"), ("adapter", {"adapter_model.safetensors": "ab"})):
        other = dict(IDENTITY, **{field: value})
        assert "one_configuration" in failed(run_dir([response(), response(other)])), field


def test_an_answer_without_identity_fails():
    bare = response()
    del bare["x_served_by"]
    assert "one_configuration" in failed(run_dir([response(), bare]))


def test_a_run_on_the_reference_path_fails():
    reference = dict(IDENTITY, recurrent_layers={"chosen": "reference", "active": "reference"})
    assert failed(run_dir([response(reference)])) == {"recurrent_layers"}


def test_a_timing_server_fails():
    assert "not_timing_mode" in failed(run_dir([response(dict(IDENTITY, timing=True))]))


def test_another_rendered_turn_fails_unless_expected():
    d = run_dir([response(rendered="f" * 64)])
    assert "rendered_turn" in failed(d)
    assert all(c["passed"] for c in M.build(d, "f" * 64)["checks"] if c["check"] == "rendered_turn")


def test_another_user_turn_file_fails():
    other = copy.deepcopy(IDENTITY)
    other["user_turn"]["file_sha256"] = "e" * 64
    assert "user_turn_file" in failed(run_dir([response(other)]))


def test_decoding_and_deadline_are_read_from_run_meta():
    assert "decoding" in failed(run_dir(meta=dict(RUN_META, temperature=0.7)))
    assert "decoding" in failed(run_dir(meta=dict(RUN_META, max_tokens=4096)))
    assert "decoding" in failed(run_dir(meta=dict(RUN_META, extra_body=None)))
    assert "deadline" in failed(run_dir(meta=dict(RUN_META, timeout=90)))


def test_a_timed_out_cell_fails():
    cells = [response(), {"error": "timeout", "key": "k", "attempts": 1}]
    assert failed(run_dir(cells)) == {"no_timeouts"}


def test_the_runner_must_ask_for_the_served_name():
    assert "served_name" in failed(run_dir(meta=dict(RUN_META, model="Qwen3.5-2B-ft")))


def test_the_live_server_must_be_the_one_that_answered():
    assert "server_matches" not in failed(run_dir(), server=IDENTITY)
    assert "server_matches" in failed(run_dir(), server=dict(IDENTITY, revision="0" * 40))


def test_a_cell_changed_after_the_manifest_is_reported():
    d = run_dir()
    M.write(d, M.user_turn.RENDERED_SHA256)
    (d / "key1.json").write_text(json.dumps(response(finish="length")))
    assert M.check(d) == ["summary differs from manifest.json"]
    (d / "key9.json").write_text(json.dumps({"error": "timeout"}))
    assert "check no_timeouts fails: 1 cells timed out" in M.check(d)


def test_a_folder_without_run_meta_is_refused():
    d = Path(tempfile.mkdtemp())
    try:
        M.build(d, M.user_turn.RENDERED_SHA256)
    except SystemExit as exc:
        assert "run_meta.json" in str(exc)
    else:
        raise AssertionError("built")


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failures = 0
    for name, fn in tests:
        try:
            fn()
            print("ok  ", name)
        except Exception as exc:  # noqa: BLE001 - a test runner reports every failure
            failures += 1
            print("FAIL", name, f"{type(exc).__name__}: {exc}")
    print(f"{len(tests) - failures} of {len(tests)} passed")
    sys.exit(1 if failures else 0)
