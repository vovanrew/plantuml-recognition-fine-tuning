"""The library versions a process runs, against requirements-card.txt, and a guard that refuses a
process whose versions differ.

decisions/gpu-precision/record.md, Decision Part 2: training and serving "pinned to transformers
5.18.0 and torch 2.14.1 ... and peft 0.21.2", and "to hold fixed when the runs are made: the GPU
model, ... the container image and library versions". requirements-card.txt pins every package
that ran on the card. On CUDA every pin is checked. On the Mac, which runs no scored run and cannot
install the CUDA packages, only CORE is checked: the packages whose code loads the model, renders
the prompt or prepares the picture.

The server calls `require()` before loading a model; the training script is to call it the same
way. The result goes into the server's identity, so the manifest of a run records it.

Tests: build/tests/test_card_environment.py
"""
import hashlib
import importlib.metadata
import re
from pathlib import Path

LOCK = Path(__file__).resolve().parents[1] / "requirements-card.txt"
CORE = ("torch", "torchvision", "transformers", "peft", "accelerate", "tokenizers", "safetensors",
        "huggingface-hub", "pillow", "numpy", "jinja2")


def normalise(name):
    """A distribution name as pip compares it."""
    return re.sub(r"[-_.]+", "-", name).lower()


def pins(path=LOCK):
    """{normalised name: version} of every `name==version` line of the lock file."""
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            name, sep, ver = line.partition("==")
            if not sep:
                raise SystemExit(f"{path}: {line!r} is not pinned with ==")
            out[normalise(name)] = ver.strip()
    return out


def version(distribution):
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def require(device, path=LOCK):
    """The versions this process runs; SystemExit unless they equal the lock file's pins."""
    pinned = pins(path)
    names = sorted(pinned) if device == "cuda" else [n for n in CORE if n in pinned]
    wrong = {n: {"pinned": pinned[n], "installed": version(n)} for n in names if version(n) != pinned[n]}
    if wrong:
        raise SystemExit(f"installed versions differ from {Path(path).name}: {wrong}")
    return {"lock_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            "checked": "all" if device == "cuda" else "core", "packages_checked": len(names),
            "versions": {n: pinned[n] for n in CORE if n in pinned}}
