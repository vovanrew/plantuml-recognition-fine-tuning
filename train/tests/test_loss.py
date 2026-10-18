"""Tests for train/loss.py: the chunked loss on the answer positions computes the library's own loss
and gradient.

Run from fine-tune/ (loads Qwen3.5-2B from .local/hf twice, float32 and bf16, on this machine's
device):
  HF_DEACTIVATE_ASYNC_LOAD=1 .venv/bin/python train/tests/test_loss.py
(or with pytest: .venv/bin/python -m pytest <this file>)

Two claims. In float32 the two computations give the same loss and gradient, so they compute the
same thing. In bf16, the precision of the study, the chunked gradient is no farther from the
float32 gradient than the library's own bf16 gradient is; the two bf16 gradients differ from each
other by rounding only.

The adapter is a test fixture: rank 4 on every linear layer of the language model, A drawn by
PEFT, B drawn from N(0, 0.001²) with a fixed seed, so that every adapter tensor receives a gradient
and the model stays near the base model. Both computations run in training mode without gradient
checkpointing; the test is of the loss computation, not of the training settings.
"""
import os
import sys
from pathlib import Path

FT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", str(FT / ".local/hf"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
sys.path.insert(0, str(FT / "build"))
sys.path.insert(0, str(FT / "infer"))
sys.path.insert(0, str(FT / "train"))
import torch  # noqa: E402

import examples as E  # noqa: E402
import loss as L  # noqa: E402

MODEL, REVISION = "Qwen/Qwen3.5-2B", "15852e8c16360a2fea060d615a32b45270f8a8fc"
CHUNK_TEST = 64          # small, so that the longer answer spans several chunks
B_STD = 1e-3
_cache = {}


def pairs():
    """Two pairs of the 1,000: the shortest class tier-1 target and the first sequence tier-3
    target, as (key, inputs, labels, prompt tokens) on the CPU."""
    if "pairs" not in _cache:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(MODEL, revision=REVISION)
        rows = E.load_examples(FT / "data/examples/train.jsonl", 125)
        short = min((r for r in rows if r["primary_type"] == "class" and r["tier"] == 1),
                    key=lambda r: len(r["target"]))
        longer = next(r for r in rows if r["primary_type"] == "sequence" and r["tier"] == 3)
        _cache["pairs"] = []
        for r in (short, longer):
            ids = E.target_ids(processor.tokenizer, r["target"])
            inputs, labels, n_prompt = E.example(processor, E.prompt_text(),
                                                 E.user_turn.open_image(FT / r["picture"]), ids,
                                                 E.end_ids(False))
            _cache["pairs"].append((r["key"], inputs, labels, n_prompt))
    return _cache["pairs"]


def gradients_of(dtype):
    """{key: {"library": (loss, gradient), "chunked": (loss, gradient)}} with the fixture adapter
    on the model loaded in `dtype`; gradients flattened, float32, on the CPU."""
    if dtype in _cache:
        return _cache[dtype]
    import server
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText
    device = server.pick_device()
    torch.manual_seed(0)
    model = AutoModelForImageTextToText.from_pretrained(MODEL, revision=REVISION,
                                                        device_map={"": device}, dtype=dtype)
    names = [n for n, m in model.named_modules()
             if isinstance(m, torch.nn.Linear) and n.startswith("model.language_model.")]
    model = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0,
                                             target_modules=names, init_lora_weights=False))
    with torch.no_grad():
        draw = torch.Generator().manual_seed(1)
        for n, p in model.named_parameters():
            if "lora_B" in n:
                p.copy_((torch.randn(p.shape, generator=draw) * B_STD).to(p.dtype))
    model.train()

    def flat():
        named = sorted((n, p) for n, p in model.named_parameters() if p.requires_grad)
        assert all(p.grad is not None for _, p in named)
        return torch.cat([p.grad.detach().float().flatten().cpu() for _, p in named])

    def zero():
        for p in model.parameters():
            p.grad = None

    out = {}
    for key, inputs, labels, n_prompt in pairs():
        x = {k: (v.to(dtype) if v.is_floating_point() else v).to(device) for k, v in inputs.items()}
        y = labels.to(device)
        answer = int((labels[0] != -100).sum())        # the library's loss is the mean over these
        zero()
        library = model(**x, labels=y, use_cache=False).loss
        library.backward()
        library_result = (float(library.detach()), flat())
        zero()
        total, n = L.backward(model, x, n_prompt, scale=1.0 / answer, chunk=CHUNK_TEST)
        assert n == answer, key
        chunked_result = (total / n, flat())
        zero()
        again, m = L.summed_loss(model, x, n_prompt, chunk=CHUNK_TEST)
        out[key] = {"library": library_result, "chunked": chunked_result, "answer_tokens": n,
                    "summed_without_gradient": (again, m), "summed": total}
    del model
    if device == "mps":
        torch.mps.empty_cache()
    _cache[dtype] = out
    return out


def relative(a, b):
    return float((a - b).norm() / a.norm())


def test_the_fixture_has_answers_of_one_and_of_several_chunks():
    lengths = [inputs["input_ids"].shape[1] - n_prompt for _, inputs, _, n_prompt in pairs()]
    assert lengths[0] <= CHUNK_TEST < 3 * CHUNK_TEST <= lengths[1], lengths


def test_in_float32_the_chunked_loss_and_gradient_are_the_librarys():
    for key, r in gradients_of(torch.float32).items():
        (lib_loss, lib_grad), (ch_loss, ch_grad) = r["library"], r["chunked"]
        assert abs(ch_loss - lib_loss) <= 1e-6 * max(1.0, lib_loss), (key, ch_loss, lib_loss)
        assert float(lib_grad.norm()) > 0
        assert relative(lib_grad, ch_grad) <= 1e-3, (key, relative(lib_grad, ch_grad))
        print(f"     float32 {key[:12]}: {r['answer_tokens']} answer tokens; loss {lib_loss:.7f} "
              f"library, {ch_loss:.7f} chunked; gradient difference {relative(lib_grad, ch_grad):.1e}")


def test_in_bf16_the_chunked_gradient_is_no_farther_from_float32_than_the_librarys():
    exact = gradients_of(torch.float32)
    for key, r in gradients_of(torch.bfloat16).items():
        reference = exact[key]["library"][1]
        (lib_loss, lib_grad), (ch_loss, ch_grad) = r["library"], r["chunked"]
        assert abs(ch_loss - lib_loss) <= 1e-5 * max(1.0, lib_loss), (key, ch_loss, lib_loss)
        lib_error, ch_error = relative(reference, lib_grad), relative(reference, ch_grad)
        assert ch_error <= 1.02 * lib_error, (key, ch_error, lib_error)
        print(f"     bf16 {key[:12]}: gradient off float32 by {lib_error:.4f} library, "
              f"{ch_error:.4f} chunked; between the two {relative(lib_grad, ch_grad):.4f}")


def test_the_loss_without_gradient_is_the_same_sum_in_float32():
    for key, r in gradients_of(torch.float32).items():
        again, m = r["summed_without_gradient"]
        assert m == r["answer_tokens"]
        assert abs(again - r["summed"]) <= 1e-5 * max(1.0, r["summed"]), (key, again, r["summed"])


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print("ok  ", name, flush=True)
        except Exception as exc:  # noqa: BLE001 - a test runner reports every failure
            failed += 1
            print("FAIL", name, f"{type(exc).__name__}: {exc}", flush=True)
    print(f"{len(tests) - failed} of {len(tests)} passed")
    sys.exit(1 if failed else 0)
