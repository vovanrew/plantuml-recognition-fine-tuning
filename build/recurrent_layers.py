"""Which implementation of Qwen3.5's recurrent (Gated DeltaNet) layers a process runs, and a guard
that refuses a process that would run the other one.

decisions/gpu-precision/record.md, Decision Part 2, "The implementation of the recurrent layers"
(author, 2026-10-18): the kernels of flash-linear-attention 0.5.2 in training and in serving, at
every size, in both conditions; causal-conv1d not installed, so its two functions stay on the
library's reference path.

transformers 5.18.0 decides this itself, once per function, when the model code is imported: it
uses `fla.ops.gated_delta_rule.<function>` if that imports, and its own reference PyTorch code
otherwise (transformers/integrations/hub_kernels.py, `use_kernel_func_from_hub_with_fallback`).
`active()` makes the same lookup; `require()` stops a process whose lookup does not give the
chosen implementation at the pinned versions (Triton included: it compiles the kernels). The
server calls it before loading a model; the training script is to call it the same way.

The kernels need CUDA. On the Mac the reference path is the only one: local checks run it and say
so in their identity; no scored run is made there. On a card the reference path is refused, except
for a diagnostic process that asks for it explicitly (`diagnostic=True`; the reversal check
compares one gradient with the reference path's) and is marked so; the server never asks for it.

Tests: build/tests/test_recurrent_layers.py
"""
import importlib
import importlib.metadata

FLA = "flash-linear-attention"
REFERENCE = "reference"
CHOICES = (FLA, REFERENCE)
PINNED = {"flash-linear-attention": "0.5.2", "fla-core": "0.5.2", "einops": "0.8.2", "triton": "3.8.0"}
FLA_MODULE = "fla.ops.gated_delta_rule"
FLA_FUNCTIONS = ("chunk_gated_delta_rule", "fused_recurrent_gated_delta_rule")


def version(distribution):
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def active():
    """The implementation transformers will use for the Gated DeltaNet functions in this process."""
    try:
        module = importlib.import_module(FLA_MODULE)
    except Exception:  # noqa: BLE001 - any failure to import sends transformers to its fallback too
        return REFERENCE
    return FLA if all(callable(getattr(module, name, None)) for name in FLA_FUNCTIONS) else REFERENCE


def describe():
    return {"active": active(),
            "versions": {d: version(d) for d in (*PINNED, "causal-conv1d")}}


def require(choice, device, diagnostic=False):
    """The description of this process's implementation; SystemExit unless it is `choice`."""
    if choice not in CHOICES:
        raise SystemExit(f"unknown implementation {choice!r}; one of {CHOICES}")
    if choice == REFERENCE and device == "cuda" and not diagnostic:
        raise SystemExit(f"on a card the study runs {FLA}; the reference path only in a diagnostic process")
    state = describe()
    if choice == FLA:
        if device != "cuda":
            raise SystemExit(f"{FLA} needs CUDA; this process runs on {device}")
        wrong = {d: state["versions"][d] for d, want in PINNED.items() if state["versions"][d] != want}
        if wrong:
            raise SystemExit(f"{FLA} pins {PINNED}; installed {wrong}")
    if state["active"] != choice:
        raise SystemExit(f"asked for the {choice} implementation; transformers would use {state['active']}")
    if state["versions"]["causal-conv1d"] is not None:
        raise SystemExit("causal-conv1d is installed; the decision leaves its functions on the reference path")
    return {"chosen": choice, "diagnostic": bool(diagnostic), **state}
