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
    "environment": {"checked": "all", "packages_checked": 61},
    "files": {"infer/server.py": "1" * 64, "build/user_turn.py": "2" * 64, "requirements-card.txt": "3" * 64},
    "study_commit": {"commit": "f" * 40, "serving_files_equal_commit": True},
}
FROZEN = dict(IDENTITY["files"])        # the files at the freeze tag, as git would give them
RUN_META = {"model": "Qwen/Qwen3.5-2B", "prompt": "P", "max_tokens": 5376, "timeout": 600, "temperature": 0,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}


def response(identity=None, rendered=None, finish="stop"):
    served = dict(identity or IDENTITY, rendered_sha256=rendered or M.user_turn.RENDERED_SHA256,
                  min_tokens=0, generate_seconds=1.0, first_token_seconds=0.1)
    return {"model": served["served_name"], "choices": [{"message": {"content": "@startuml\n@enduml"},
                                                          "finish_reason": finish}],
            "usage": {"prompt_tokens": 432, "completion_tokens": 9}, "x_served_by": served}


def run_dir(cells=None, meta=None):
    """A run directory; `n_cells` is the number of cells written unless `meta` sets it."""
    d = Path(tempfile.mkdtemp())
    cells = cells if cells is not None else [response(), response(), response()]
    (d / "run_meta.json").write_text(json.dumps({"n_cells": len(cells), **(meta or RUN_META)}))
    for i, cell in enumerate(cells):
        (d / f"key{i}.json").write_text(json.dumps(cell))
    return d


def failed(d, server=None, frozen=FROZEN):
    return {c["check"] for c in M.build(d, M.user_turn.RENDERED_SHA256, server, frozen)["checks"] if not c["passed"]}


def test_a_clean_run_passes_and_its_manifest_holds():
    d = run_dir()
    manifest = M.write(d, M.user_turn.RENDERED_SHA256, frozen=FROZEN)
    assert manifest["passed"] and manifest["summary"]["outcomes"] == {"ok": 3}
    assert manifest["served_files"] == IDENTITY["files"] and manifest["freeze_tag"] == M.FREEZE_TAG
    assert (d / "manifest.json").exists() and M.check(d, frozen=FROZEN) == []


def test_a_run_that_mixes_two_configurations_fails():
    for field, value in (("revision", "0" * 40), ("optional_kernels", {"flash-linear-attention": None}),
                         ("versions", {"torch": "2.15.0"}), ("files", dict(IDENTITY["files"], **{"infer/server.py": "9" * 64})),
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


def test_a_cell_with_any_failure_record_fails():
    for error in ("timeout", "http_error", "network_error", "image_dropped", "bad_response"):
        cells = [response(), {"error": error, "key": "k", "attempts": 1}]
        assert failed(run_dir(cells)) == {"every_cell_answered"}, error


def test_served_files_other_than_the_frozen_ones_fail():
    assert failed(run_dir(), frozen=dict(FROZEN, **{"infer/server.py": "0" * 64})) == {"frozen_files"}
    assert failed(run_dir(), frozen={}) == {"frozen_files"}                  # the tag cannot be read
    bare = dict(IDENTITY)
    del bare["files"]
    assert "frozen_files" in failed(run_dir([response(bare)]))


def test_a_server_that_checked_only_the_core_packages_fails():
    mac = dict(IDENTITY, environment={"checked": "core", "packages_checked": 11})
    assert failed(run_dir([response(mac)])) == {"environment"}


def test_the_frozen_hashes_are_read_from_the_tag():
    hashes = M.frozen_hashes(["infer/server.py", "no/such/file"], tag="serving-v1")
    assert hashes["no/such/file"] is None
    assert hashes["infer/server.py"] == "7b5df19791977aeb87eccf24ddbf635c84a533a9dd43e372edd845b6af87fd76"


def test_the_runner_must_ask_for_the_served_name():
    assert "served_name" in failed(run_dir(meta=dict(RUN_META, model="Qwen3.5-2B-ft")))


def test_the_live_server_must_be_the_one_that_answered():
    assert "server_matches" not in failed(run_dir(), server=IDENTITY)
    assert "server_matches" in failed(run_dir(), server=dict(IDENTITY, revision="0" * 40))


def test_a_cell_changed_after_the_manifest_is_reported():
    d = run_dir()
    M.write(d, M.user_turn.RENDERED_SHA256, frozen=FROZEN)
    (d / "key1.json").write_text(json.dumps(response(finish="length")))
    assert M.check(d, frozen=FROZEN) == ["summary differs from manifest.json"]
    (d / "key9.json").write_text(json.dumps({"error": "timeout"}))
    assert any(p.startswith("check every_cell_answered fails") for p in M.check(d, frozen=FROZEN))


def test_a_run_stopped_part_way_fails():
    assert failed(run_dir(meta=dict(RUN_META, n_cells=1000))) == {"complete_run"}
    meta = dict(RUN_META)
    meta.pop("n_cells", None)
    d = run_dir()
    (d / "run_meta.json").write_text(json.dumps(meta))             # no n_cells recorded
    assert failed(d) == {"complete_run"}


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
