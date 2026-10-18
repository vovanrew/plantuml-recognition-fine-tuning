"""Tests for build/recurrent_layers.py. On the Mac flash-linear-attention is not installed, so the
cases with it are made by putting a stand-in module and stand-in versions in place.

Run from fine-tune/:  .venv/bin/python -m pytest -q build/tests/test_recurrent_layers.py
                      (or .venv/bin/python build/tests/test_recurrent_layers.py)
"""
import sys
import types
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import recurrent_layers as R  # noqa: E402

FLA_VERSIONS = {"flash-linear-attention": "0.5.2", "fla-core": "0.5.2", "einops": "0.8.2",
                "causal-conv1d": None, "triton": "3.8.0"}


@contextmanager
def installed(functions=R.FLA_FUNCTIONS, versions=None):
    """A stand-in `fla.ops.gated_delta_rule` with the given functions, and stand-in versions."""
    names = ["fla", "fla.ops", R.FLA_MODULE]
    saved = {n: sys.modules.get(n) for n in names}
    module = types.ModuleType(R.FLA_MODULE)
    for name in functions:
        setattr(module, name, lambda *a, **k: None)
    sys.modules.update({"fla": types.ModuleType("fla"), "fla.ops": types.ModuleType("fla.ops"),
                        R.FLA_MODULE: module})
    real_version = R.version
    R.version = lambda d: (versions or FLA_VERSIONS).get(d)
    try:
        yield
    finally:
        R.version = real_version
        for n, m in saved.items():
            if m is None:
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = m


def refusal(choice, device):
    try:
        R.require(choice, device)
    except SystemExit as exc:
        return str(exc)
    return None


def test_on_this_mac_the_reference_path_is_active_and_allowed():
    assert R.active() == R.REFERENCE
    state = R.require(R.REFERENCE, "mps")
    assert state["chosen"] == state["active"] == R.REFERENCE
    assert state["versions"]["flash-linear-attention"] is None


def test_the_kernels_are_refused_without_cuda():
    assert "needs CUDA" in refusal(R.FLA, "mps")
    with installed():
        assert "needs CUDA" in refusal(R.FLA, "cpu")


def test_the_kernels_are_refused_when_not_installed():
    assert "pins" in refusal(R.FLA, "cuda")


def test_the_kernels_at_the_pinned_versions_pass():
    with installed():
        assert R.active() == R.FLA
        state = R.require(R.FLA, "cuda")
        assert state["chosen"] == R.FLA and state["versions"]["triton"] == "3.8.0"


def test_other_versions_are_refused():
    with installed(versions=dict(FLA_VERSIONS, **{"flash-linear-attention": "0.5.3"})):
        assert "pins" in refusal(R.FLA, "cuda")


def test_a_package_without_both_functions_counts_as_the_reference_path():
    with installed(functions=("chunk_gated_delta_rule",)):
        assert R.active() == R.REFERENCE
        assert "transformers would use reference" in refusal(R.FLA, "cuda")


def test_the_reference_path_is_refused_when_the_kernels_would_be_used():
    with installed():
        assert "transformers would use flash-linear-attention" in refusal(R.REFERENCE, "cuda")


def test_causal_conv1d_installed_is_refused():
    with installed(versions=dict(FLA_VERSIONS, **{"causal-conv1d": "1.7.0"})):
        assert "causal-conv1d" in refusal(R.FLA, "cuda")


def test_an_unknown_choice_is_refused():
    assert "unknown" in refusal("fast", "cuda")


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
