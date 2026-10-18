"""Tests for build/card_environment.py.

Run from fine-tune/:  .venv/bin/python -m pytest -q build/tests/test_card_environment.py
                      (or .venv/bin/python build/tests/test_card_environment.py)
"""
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import card_environment as E  # noqa: E402


@contextmanager
def installed(versions):
    real = E.version
    E.version = lambda d: versions.get(E.normalise(d))
    try:
        yield
    finally:
        E.version = real


def refusal(device, path=E.LOCK):
    try:
        E.require(device, path)
    except SystemExit as exc:
        return str(exc)
    return None


def lock(text):
    path = Path(tempfile.mkdtemp()) / "requirements-card.txt"
    path.write_text(text)
    return path


def test_the_lock_file_pins_every_line_and_holds_the_decided_versions():
    pinned = E.pins()
    assert len(pinned) == 61
    assert pinned["torch"] == "2.14.1" and pinned["transformers"] == "5.18.0" and pinned["peft"] == "0.21.2"
    assert pinned["flash-linear-attention"] == "0.5.2" and pinned["fla-core"] == "0.5.2"
    assert pinned["einops"] == "0.8.2" and pinned["triton"] == "3.8.0"
    assert not {"causal-conv1d", "bitsandbytes", "kernels"} & set(pinned)


def test_this_mac_passes_on_the_core_packages():
    state = E.require("mps")
    assert state["checked"] == "core" and state["packages_checked"] == len(E.CORE)
    assert state["versions"]["torch"] == "2.14.1"


def test_on_cuda_every_pin_is_checked():
    # this Mac lacks the CUDA packages and the kernels, so a CUDA check refuses it
    message = refusal("cuda")
    assert "nvidia-cublas" in message and "triton" in message and "flash-linear-attention" in message


def test_a_card_with_exactly_the_pinned_versions_passes():
    with installed(E.pins()):
        state = E.require("cuda")
    assert state["checked"] == "all" and state["packages_checked"] == 61


def test_one_other_version_is_refused():
    for name, other in (("torch", "2.15.0"), ("transformers", "5.19.0"), ("peft", "0.22.0"), ("triton", "3.8.1")):
        versions = dict(E.pins(), **{name: other})
        with installed(versions):
            assert name in refusal("cuda"), name
    with installed(dict(E.pins(), transformers="5.19.0")):
        assert "transformers" in refusal("mps")          # a core package is checked on the Mac too


def test_a_missing_package_is_refused():
    versions = dict(E.pins())
    del versions["peft"]
    with installed(versions):
        assert "peft" in refusal("cuda")


def test_an_unpinned_line_is_refused():
    assert "not pinned" in refusal("cuda", lock("torch>=2.14\n"))


def test_names_are_compared_as_pip_compares_them():
    with installed({"jinja2": "3.1.6", "huggingface-hub": "2.2.0"}):
        assert E.require("cuda", lock("# a comment\nJinja2==3.1.6\nhuggingface_hub==2.2.0  # trailing\n"))


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
