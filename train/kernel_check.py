#!/usr/bin/env python3
"""The reversal check of the recurrent layers' implementation, run on the card.

decisions/gpu-precision/record.md, Decision Part 2, "The implementation of the recurrent layers":
flash-linear-attention (the kernels) is reversed by "answers or training steps that do not repeat
under the kernels (the same request twice, the same step twice), or a training gradient that
differs from the reference path's by more than rounding". This script measures the three and
states each criterion's result; reversing the decision is the author's.

Modes, each one process, run from fine-tune/:
  step    --layers flash-linear-attention|reference --dtype bfloat16|float32 --repeat N
          --adapter <dir> --out <file.pt>
          One training step's loss and gradient, computed N times in this process from the same
          state: Qwen3.5-2B at its pinned revision with the adapter loaded as trainable, the
          recipe's vision tower without gradient, non-reentrant gradient checkpointing and training
          mode as train/train.py runs them, the torch seed CHECK_SEED set before each computation,
          the first CHECK_PAIRS pairs of the pilot's first optimizer step, the loss of train/loss.py
          weighted by recipe.loss_scale. With --layers reference the process is a diagnostic one:
          flash-linear-attention is kept from importing before transformers loads, and
          build/recurrent_layers.py require() is called with diagnostic=True.
  answer  --adapter <dir> --out <file.json>
          Greedy answers to the adapter check's three training pictures, each asked twice, through
          the server's own start-up and generate (the kernels), at most ANSWER_TOKENS tokens.
  compare --steps <file.pt>... --answers <file.json> --out <file.json>
          The measures and the criteria below over the files of the other two modes.

Criteria, written before the run:
  request repeat   every answer equals its repeat, token for token.
  step repeat      the loss and every gradient element of a repeat equal the first computation's,
                   bitwise: within one process and between two processes.
  gradient         rounding is the bfloat16 reference gradient's distance to the float32
                   reference gradient, e_ref = |g_ref - g_32| / |g_32| (norms over all adapter
                   parameters). The kernels' gradient passes when e_fla = |g_fla - g_32| / |g_32|
                   is at most GRADIENT_FACTOR * e_ref.

Tests: train/tests/test_kernel_check.py
"""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FT = HERE.parent

CHECK_SEED = 20261021
CHECK_PAIRS = 4
ANSWER_TOKENS = 512
GRADIENT_FACTOR = 1.1
SIZE = "2B"


def block_fla():
    """Keep flash-linear-attention from importing in this process, so transformers takes its
    reference path for the Gated DeltaNet functions. Must run before transformers is imported."""
    if "transformers" in sys.modules:
        raise SystemExit("block_fla() must run before transformers is imported")
    sys.modules["fla"] = None


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# step
# --------------------------------------------------------------------------- #
def step(args):
    if args.layers == "reference":
        block_fla()
    for p in (HERE, FT / "build"):
        sys.path.insert(0, str(p))
    import torch
    import card_environment
    import examples as E
    import loss as L
    import recipe as R
    import recurrent_layers
    import train as T
    from models import SIZES
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    device = T.pick_device()
    guards = {"environment": card_environment.require(device),
              "recurrent_layers": recurrent_layers.require(args.layers, device,
                                                           diagnostic=args.layers == "reference")}
    rows = T.load_rows()
    by_key = {r["key"]: r for r in rows}
    _, batches = T.run_plan(rows, 1000)
    batch = batches[0][:CHECK_PAIRS]
    model_id, revision = SIZES[SIZE]
    dtype = getattr(torch, args.dtype)
    processor = AutoProcessor.from_pretrained(model_id, revision=revision)
    base = AutoModelForImageTextToText.from_pretrained(model_id, revision=revision,
                                                       device_map={"": device}, dtype=dtype)
    model = PeftModel.from_pretrained(base, args.adapter, is_trainable=True)
    R.vision_without_grad(model)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if not named or any(".lora_" not in n for n, _ in named):
        raise SystemExit("the trainable parameters are not the adapter's")
    prompt = E.prompt_text()
    ends = E.end_ids(R.END_WITH_NEWLINE)
    ids = {k: E.target_ids(processor.tokenizer, by_key[k]["target"]) for k in batch}
    counts = [len(ids[k]) + len(ends) for k in batch]
    scale = R.loss_scale(counts)

    losses, grads, rng = [], [], []
    for _ in range(args.repeat):
        torch.manual_seed(CHECK_SEED)
        if device == "cuda":
            torch.cuda.manual_seed_all(CHECK_SEED)
        summed = 0.0
        for k in batch:
            inputs, _, n_prompt = E.example(processor, prompt, E.user_turn.open_image(FT / by_key[k]["picture"]),
                                            ids[k], ends)
            inputs = {name: v.to(device) for name, v in inputs.items()}
            total, _ = L.backward(model, inputs, n_prompt, scale)
            summed += total
        losses.append(summed / sum(counts))
        grads.append({n: p.grad.detach().float().cpu().clone() for n, p in named})
        rng.append(torch.cuda.get_rng_state().tolist() if device == "cuda" else None)
        model.zero_grad(set_to_none=True)
    meta = {"mode": "step", "layers": args.layers, "dtype": args.dtype, "device": device,
            "model": model_id, "revision": revision, "adapter": str(args.adapter),
            "adapter_sha256": sha256_file(Path(args.adapter) / "adapter_model.safetensors"),
            "pairs": batch, "seed": CHECK_SEED, "guards": guards, "losses": losses,
            "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"meta": meta, "grads": grads, "rng": rng}, args.out)
    print(json.dumps({k: meta[k] for k in ("layers", "dtype", "device", "losses")}), flush=True)


# --------------------------------------------------------------------------- #
# answer
# --------------------------------------------------------------------------- #
def answer(args):
    for p in (HERE, FT / "infer", FT / "build"):
        sys.path.insert(0, str(p))
    import server
    import dev_scoring as D
    from models import SIZES
    model_id, revision = SIZES[SIZE]
    server.start(argparse.Namespace(model=model_id, revision=revision, adapter=args.adapter,
                                    served_name=f"{model_id}+adapter", device=None, timing=False,
                                    recurrent_layers=args.layers))
    model, processor = server.STATE["model"], server.STATE["processor"]
    identity = server.STATE["identity"]
    prompt = (server.ROOT.parent / "plantuml-recognition-evaluation" / "prompts" / "zero_shot.txt"
              ).read_text(encoding="utf-8").strip()                    # as the runner reads it
    rows = []
    for picture in D.check_pictures():
        inputs = server.user_turn.render(processor, prompt, server.user_turn.open_image(picture)).to(identity["device"])
        first, _, _ = server.generate(model, inputs, args.max_tokens)
        second, _, _ = server.generate(model, inputs, args.max_tokens)
        rows.append({"picture": str(picture), "tokens": len(first), "repeat_tokens": len(second),
                     "equal": first == second,
                     "first_difference": next((i for i, (a, b) in enumerate(zip(first, second)) if a != b), None)})
    result = {"mode": "answer", "identity": identity, "max_tokens": args.max_tokens, "answers": rows}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(rows), flush=True)


# --------------------------------------------------------------------------- #
# compare
# --------------------------------------------------------------------------- #
def bitwise(a, b):
    """(equal, number of differing elements, largest absolute difference) of two gradient dicts."""
    import torch
    if a.keys() != b.keys():
        raise SystemExit("the two gradients are not over the same parameters")
    differing, largest = 0, 0.0
    for name in a:
        diff = (a[name] - b[name]).abs()
        differing += int((a[name] != b[name]).sum())
        largest = max(largest, float(diff.max()))
    return differing == 0, differing, largest


def norm(a):
    return math.sqrt(sum(float((t.double() ** 2).sum()) for t in a.values()))


def distance(a, b):
    """|a - b| / |b| over all parameters."""
    return math.sqrt(sum(float(((a[n].double() - b[n].double()) ** 2).sum()) for n in b)) / norm(b)


def cosine(a, b):
    dot = sum(float((a[n].double() * b[n].double()).sum()) for n in b)
    return dot / (norm(a) * norm(b))


def verdicts(fla_1, fla_2, ref, f32, answers, factor=GRADIENT_FACTOR):
    """The three criteria over loaded step files (dicts with "meta" and "grads") and the answer
    rows. fla_1 holds at least two repeats; fla_2 is a second process; ref holds the bfloat16
    reference path; f32 the float32 reference path."""
    within = bitwise(fla_1["grads"][0], fla_1["grads"][1])
    across = bitwise(fla_1["grads"][0], fla_2["grads"][0])
    losses_within = fla_1["meta"]["losses"][0] == fla_1["meta"]["losses"][1]
    losses_across = fla_1["meta"]["losses"][0] == fla_2["meta"]["losses"][0]
    ref_within = bitwise(ref["grads"][0], ref["grads"][1]) if len(ref["grads"]) > 1 else None
    g32 = f32["grads"][0]
    e_ref = distance(ref["grads"][0], g32)
    e_fla = distance(fla_1["grads"][0], g32)
    request = all(r["equal"] for r in answers)
    step = within[0] and across[0] and losses_within and losses_across
    gradient = e_fla <= factor * e_ref
    return {
        "request_repeat": {"passed": request, "answers": answers},
        "step_repeat": {"passed": step,
                        "within_process": {"loss_equal": losses_within, "equal": within[0],
                                           "differing_elements": within[1], "largest_difference": within[2]},
                        "between_processes": {"loss_equal": losses_across, "equal": across[0],
                                              "differing_elements": across[1], "largest_difference": across[2]},
                        "reference_within_process": None if ref_within is None else
                        {"equal": ref_within[0], "differing_elements": ref_within[1],
                         "largest_difference": ref_within[2]}},
        "gradient": {"passed": gradient, "e_fla": e_fla, "e_ref": e_ref, "ratio": e_fla / e_ref if e_ref else None,
                     "factor": factor, "fla_to_reference": distance(fla_1["grads"][0], ref["grads"][0]),
                     "cosine_fla_reference": cosine(fla_1["grads"][0], ref["grads"][0]),
                     "losses": {"fla": fla_1["meta"]["losses"][0], "reference": ref["meta"]["losses"][0],
                                "float32": f32["meta"]["losses"][0]}},
        "passed": request and step and gradient,
    }


def compare(args):
    import torch
    loaded = [torch.load(p, weights_only=False) for p in args.steps]
    pick = {}
    for item in loaded:
        m = item["meta"]
        key = (m["layers"], m["dtype"])
        pick.setdefault(key, []).append(item)
    fla = pick.get(("flash-linear-attention", "bfloat16"), [])
    ref = pick.get(("reference", "bfloat16"), [])
    f32 = pick.get(("reference", "float32"), [])
    if len(fla) != 2 or len(fla[0]["grads"]) < 2 or len(ref) != 1 or len(f32) != 1:
        raise SystemExit("compare needs two kernel processes (the first with two repeats), one bfloat16 "
                         "and one float32 reference process")
    if len({item["meta"]["adapter_sha256"] for item in loaded}) != 1 or len({tuple(item["meta"]["pairs"]) for item in loaded}) != 1:
        raise SystemExit("the step files are not of one adapter and one batch")
    answers = json.loads(Path(args.answers).read_text())["answers"]
    result = verdicts(fla[0], fla[1], ref[0], f32[0], answers)
    result["inputs"] = {"steps": [{"file": str(p), "sha256": sha256_file(p), **{k: item["meta"][k] for k in
                                   ("layers", "dtype", "device", "gpu", "losses", "adapter_sha256")}}
                                  for p, item in zip(args.steps, loaded)],
                        "answers": {"file": str(args.answers), "sha256": sha256_file(args.answers)},
                        "pairs": loaded[0]["meta"]["pairs"], "seed": CHECK_SEED,
                        "kernel_check_sha256": sha256_file(__file__)}
    Path(args.out).write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps({k: result[k]["passed"] for k in ("request_repeat", "step_repeat", "gradient")}
                     | {"passed": result["passed"]}), flush=True)
    return result


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    s = sub.add_parser("step")
    s.add_argument("--layers", required=True, choices=["flash-linear-attention", "reference"])
    s.add_argument("--dtype", required=True, choices=["bfloat16", "float32"])
    s.add_argument("--repeat", type=int, default=1)
    s.add_argument("--adapter", required=True)
    s.add_argument("--out", required=True)
    a = sub.add_parser("answer")
    a.add_argument("--adapter", required=True)
    a.add_argument("--layers", default="flash-linear-attention", choices=["flash-linear-attention", "reference"])
    a.add_argument("--max-tokens", type=int, default=ANSWER_TOKENS)
    a.add_argument("--out", required=True)
    c = sub.add_parser("compare")
    c.add_argument("--steps", nargs="+", required=True)
    c.add_argument("--answers", required=True)
    c.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    return {"step": step, "answer": answer, "compare": compare}[args.mode](args)


if __name__ == "__main__":
    main()
