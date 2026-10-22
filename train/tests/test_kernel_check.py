"""Tests for train/kernel_check.py.

Run from fine-tune/:
  .venv/bin/python train/tests/test_kernel_check.py                 without weights
  HF_DEACTIVATE_ASYNC_LOAD=1 .venv/bin/python train/tests/test_kernel_check.py --live --adapter <dir>
      also runs `step` on the reference path (bfloat16, twice; float32, once) and `answer` with
      the 2B from .local/hf (the kernels need CUDA, so on the Mac only the reference path runs)
(or with pytest: .venv/bin/python -m pytest <this file>)
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
sys.path.insert(0, str(FT / "train"))
import kernel_check as K  # noqa: E402


def grads(seed, scale=1.0, noise=0.0, noise_seed=1):
    import torch
    g = torch.Generator().manual_seed(seed)
    base = {"a.lora_A": torch.randn(16, 32, generator=g), "a.lora_B": torch.randn(32, 16, generator=g)}
    if noise:
        n = torch.Generator().manual_seed(noise_seed)
        return {k: v * scale + noise * torch.randn(v.shape, generator=n) for k, v in base.items()}
    return {k: v * scale for k, v in base.items()}


def item(layers, dtype, gs, losses):
    return {"meta": {"layers": layers, "dtype": dtype, "losses": losses}, "grads": gs}


def setting(fla_noise=1e-3, ref_noise=1e-3, within_equal=True, across_equal=True, answers_equal=True,
            repeat_loss=0.5, other_process_loss=0.5):
    """A synthetic outcome: float32 truth, a bfloat16 reference and the kernels around it."""
    truth = grads(0)
    ref = grads(0, noise=ref_noise, noise_seed=2)
    fla = grads(0, noise=fla_noise, noise_seed=3)
    fla_repeat = fla if within_equal else grads(0, noise=fla_noise, noise_seed=4)
    fla_other = fla if across_equal else grads(0, noise=fla_noise, noise_seed=5)
    answers = [{"equal": answers_equal}, {"equal": True}]
    return (item("flash-linear-attention", "bfloat16", [fla, fla_repeat], [0.5, repeat_loss]),
            item("flash-linear-attention", "bfloat16", [fla_other], [other_process_loss]),
            item("reference", "bfloat16", [ref, ref], [0.5, 0.5]),
            item("reference", "float32", [truth], [0.5]), answers)


def test_all_three_pass_when_the_kernels_repeat_and_round_like_the_reference():
    v = K.verdicts(*setting())
    assert v["passed"] and v["request_repeat"]["passed"] and v["step_repeat"]["passed"] and v["gradient"]["passed"]
    assert v["step_repeat"]["within_process"]["differing_elements"] == 0


def test_a_step_that_does_not_repeat_fails_within_or_between_processes():
    within = K.verdicts(*setting(within_equal=False))
    assert not within["step_repeat"]["passed"] and within["step_repeat"]["within_process"]["differing_elements"] > 0
    across = K.verdicts(*setting(across_equal=False))
    assert not across["step_repeat"]["passed"] and across["step_repeat"]["within_process"]["equal"]
    assert not across["passed"]


def test_a_gradient_farther_from_float32_than_rounding_fails():
    v = K.verdicts(*setting(fla_noise=5e-3, ref_noise=1e-3))
    assert not v["gradient"]["passed"] and v["gradient"]["ratio"] > K.GRADIENT_FACTOR
    near = K.verdicts(*setting(fla_noise=1.05e-3, ref_noise=1e-3))
    assert near["gradient"]["passed"], near["gradient"]


def test_an_answer_that_differs_from_its_repeat_fails():
    v = K.verdicts(*setting(answers_equal=False))
    assert not v["request_repeat"]["passed"] and not v["passed"]


def test_compare_refuses_files_of_different_adapters_or_batches():
    import torch
    fla1, fla2, ref, f32, answers = setting()
    with tempfile.TemporaryDirectory() as tmp:
        files = []
        for name, it in (("fla_1", fla1), ("fla_2", fla2), ("ref", ref), ("f32", f32)):
            it["meta"].update(adapter_sha256="x", pairs=["k1", "k2"], device="cpu", gpu=None)
            if name == "f32":
                it["meta"]["adapter_sha256"] = "other"
            torch.save(it, Path(tmp) / f"{name}.pt")
            files.append(str(Path(tmp) / f"{name}.pt"))
        (Path(tmp) / "answers.json").write_text(json.dumps({"answers": answers}))
        try:
            K.main(["compare", "--steps", *files, "--answers", str(Path(tmp) / "answers.json"),
                    "--out", str(Path(tmp) / "out.json")])
        except SystemExit as exc:
            assert "one adapter and one batch" in str(exc)
        else:
            raise AssertionError("files of two adapters were compared")


def test_the_reference_process_keeps_the_kernels_from_importing():
    code = ("import sys; sys.path[:0] = ['train', 'build']; import kernel_check as K; K.block_fla(); "
            "import recurrent_layers; print(recurrent_layers.active()); "
            "import importlib\n"
            "try:\n importlib.import_module('fla.ops.gated_delta_rule'); print('imported')\n"
            "except ImportError: print('blocked')")
    done = subprocess.run([sys.executable, "-c", code], cwd=FT, capture_output=True, text=True)
    assert done.stdout.split() == ["reference", "blocked"], done.stdout + done.stderr
    late = ("import sys; sys.path[:0] = ['train']; import transformers, kernel_check as K\n"
            "try:\n K.block_fla(); print('blocked late')\nexcept SystemExit as e: print('refused')")
    done = subprocess.run([sys.executable, "-c", late], cwd=FT, capture_output=True, text=True)
    assert done.stdout.strip() == "refused", done.stdout + done.stderr


def test_the_written_criteria_are_the_ones_the_run_used():
    assert (K.GRADIENT_FACTOR, K.CHECK_PAIRS, K.CHECK_SEED, K.ANSWER_TOKENS) == (1.1, 4, 20261021, 512)


def test_the_step_criterion_needs_equal_losses_too():
    v = K.verdicts(*setting(repeat_loss=0.5000001))
    assert not v["step_repeat"]["passed"] and v["step_repeat"]["within_process"]["equal"]
    other = K.verdicts(*setting(other_process_loss=0.5000001))
    assert not other["step_repeat"]["passed"] and not other["step_repeat"]["between_processes"]["loss_equal"]


def test_compare_gives_each_file_its_role():
    import torch
    fla1, fla2, ref, f32, answers = setting(fla_noise=5e-3, ref_noise=1e-3)
    with tempfile.TemporaryDirectory() as tmp:
        files = []
        for name, it in (("ref", ref), ("f32", f32), ("fla_2", fla2), ("fla_1", fla1)):   # any order
            it["meta"].update(adapter_sha256="x", pairs=["k1"], device="cpu", gpu=None)
            torch.save(it, Path(tmp) / f"{name}.pt")
            files.append(str(Path(tmp) / f"{name}.pt"))
        (Path(tmp) / "answers.json").write_text(json.dumps({"answers": answers}))
        out = Path(tmp) / "out.json"
        K.main(["compare", "--steps", *files, "--answers", str(Path(tmp) / "answers.json"), "--out", str(out)])
        result = json.loads(out.read_text())
    g = result["gradient"]
    assert g["e_fla"] == K.distance(fla1["grads"][0], f32["grads"][0])
    assert g["e_ref"] == K.distance(ref["grads"][0], f32["grads"][0])
    assert not g["passed"] and result["step_repeat"]["passed"]


def test_configure_sets_the_training_scripts_configuration():
    import recipe
    calls = {}

    class Visual:
        def forward(self, x):
            return x

    class Core:
        def __init__(self):
            self.model = type("M", (), {"visual": Visual()})()

    class Model:
        core = Core()

        def get_base_model(self):
            return self.core

        def gradient_checkpointing_enable(self, **kwargs):
            calls["checkpointing"] = kwargs

        def train(self):
            calls["train"] = True

    model = K.configure(Model(), recipe)
    assert calls == {"checkpointing": {"gradient_checkpointing_kwargs": {"use_reentrant": False}}, "train": True}
    assert model.core.model.visual._recipe_without_grad is True


def test_answer_asks_every_picture_twice():
    import types
    asked = []

    def generate(model, inputs, max_tokens):
        asked.append((inputs, max_tokens))
        repeat = sum(1 for a in asked if a[0] == inputs) == 2
        return ([1, 2, 4] if repeat and inputs.endswith("caf1d5d8e6754721aec72cf30c97c5d66b6bcd47.png") else [1, 2, 3]), 0.0, 0.0

    class Rendered(str):
        def to(self, device):
            return self

    fake = types.ModuleType("server")
    fake.ROOT = FT
    fake.STATE = {}
    fake.start = lambda args: fake.STATE.update(model="m", processor="p", identity={"device": "cpu", "args": vars(args)})
    fake.generate = generate
    fake.user_turn = types.SimpleNamespace(render=lambda processor, prompt, image: Rendered(image),
                                           open_image=lambda picture: str(picture))
    saved = sys.modules.get("server")
    sys.modules["server"] = fake
    try:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "answers.json"
            K.main(["answer", "--adapter", "a", "--max-tokens", "7", "--out", str(out)])
            rows = json.loads(out.read_text())["answers"]
    finally:
        if saved is None:
            sys.modules.pop("server", None)
        else:
            sys.modules["server"] = saved
    assert len(asked) == 6 and all(m == 7 for _, m in asked)
    assert [r["equal"] for r in rows] == [True, False, True] and rows[1]["first_difference"] == 2


def test_the_reference_process_blocks_an_installed_fla():
    """With a stand-in flash-linear-attention importable, prepare("reference") still runs the
    reference path, as a diagnostic process; without the block the same process would not."""
    with tempfile.TemporaryDirectory() as tmp:
        pkg = Path(tmp) / "fla" / "ops" / "gated_delta_rule"
        pkg.mkdir(parents=True)
        (Path(tmp) / "fla" / "__init__.py").write_text("")
        (Path(tmp) / "fla" / "ops" / "__init__.py").write_text("")
        (pkg / "__init__.py").write_text("def chunk_gated_delta_rule(*a): pass\ndef fused_recurrent_gated_delta_rule(*a): pass\n")
        env = dict(os.environ, PYTHONPATH=tmp)
        control = subprocess.run([sys.executable, "-c", "import sys; sys.path.insert(0, 'build'); import recurrent_layers; "
                                  "print(recurrent_layers.active())"], cwd=FT, env=env, capture_output=True, text=True)
        assert control.stdout.strip() == "flash-linear-attention", control.stdout + control.stderr
        done = subprocess.run([sys.executable, "-c", "import sys; sys.path.insert(0, 'train'); import kernel_check as K; "
                               "g = K.prepare('reference', 'cpu')['recurrent_layers']; print(g['active'], g['diagnostic'])"],
                              cwd=FT, env=env, capture_output=True, text=True)
        assert done.stdout.strip() == "reference True", done.stdout + done.stderr


def live(adapter):
    """On the Mac: two repeats of the step on the reference path in bfloat16, one in float32, and
    the answer repeat with a short limit; the step must repeat bitwise within the process."""
    import torch
    with tempfile.TemporaryDirectory() as tmp:
        ref = Path(tmp) / "ref.pt"
        f32 = Path(tmp) / "f32.pt"
        for dtype, repeat, out in (("bfloat16", "2", ref), ("float32", "1", f32)):   # one process each, as on the card
            done = subprocess.run([sys.executable, str(FT / "train/kernel_check.py"), "step", "--layers", "reference",
                                   "--dtype", dtype, "--repeat", repeat, "--adapter", str(adapter), "--out", str(out)],
                                  cwd=FT, capture_output=True, text=True)
            assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]
        r, f = torch.load(ref, weights_only=False), torch.load(f32, weights_only=False)
        equal, differing, largest = K.bitwise(r["grads"][0], r["grads"][1])
        print(f"     reference bf16 within process: equal {equal}, differing {differing}, largest {largest}")
        assert r["meta"]["losses"][0] == r["meta"]["losses"][1] and equal
        assert r["meta"]["guards"]["recurrent_layers"]["active"] == "reference"
        assert r["meta"]["guards"]["recurrent_layers"]["diagnostic"] is True        # through prepare()
        assert r["meta"]["training_mode"] is True and r["meta"]["gradient_checkpointing"] is True  # configure()
        e_ref = K.distance(r["grads"][0], f["grads"][0])
        print(f"     e_ref {e_ref:.4g}, losses bf16 {r['meta']['losses'][0]:.6f}, f32 {f['meta']['losses'][0]:.6f}")
        assert 0 < e_ref < 1
        answers = Path(tmp) / "answers.json"
        done = subprocess.run([sys.executable, str(FT / "train/kernel_check.py"), "answer", "--adapter", str(adapter),
                               "--layers", "reference", "--max-tokens", "32", "--out", str(answers)],
                              cwd=FT, capture_output=True, text=True)
        assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]
        rows = json.loads(answers.read_text())["answers"]
        assert len(rows) == 3 and all(r_["equal"] for r_ in rows), rows
        print("ok   live:", json.dumps({"answers_equal": [r_["equal"] for r_ in rows], "e_ref": e_ref}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--adapter", default=None)
    cli = ap.parse_args()
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print("ok  ", name, flush=True)
        except BaseException as exc:  # noqa: BLE001
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
