"""The check "the adapter changes the output", run for an adapter before any scored run with it
(decisions/gpu-precision/record.md, Decision Part 2, "Serving software"; vllm #49354 reports an
adapter that silently changes nothing).

It loads the base model and the adapter exactly as the server does (`infer/server.py` `start`: the
same guards, bf16, the adapter attached as unmerged LoRA layers) and answers the same pictures
twice from that one loaded model: with the adapter, and with it switched off (`disable_adapter()`),
which is the base model the zero-shot condition serves. It passes when, on every picture, the
logits of the first answer token differ between the two, and on at least one picture the greedy
answers differ.

Use training pictures (data/sets/render/train/puml_images_1568/), not test pictures: the check
makes answers with a fine-tuned adapter, which must not exist for a test diagram before its
scored run.

Usage, from fine-tune/:
  python infer/adapter_check.py --model Qwen/Qwen3.5-27B --revision <commit> --adapter <dir> \
      --pictures <png> [<png> ...] --out <result.json>
  on the Mac: add --recurrent-layers reference (HF_DEACTIVATE_ASYNC_LOAD=1 on MPS)
Tests: infer/tests/test_adapter_check.py
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import server  # noqa: E402

PROMPT = server.ROOT.parent / "plantuml-recognition-evaluation" / "prompts" / "zero_shot.txt"


def first_difference(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def answer(model, inputs, max_tokens):
    ids, _, _ = server.generate(model, inputs, max_tokens)
    with torch.inference_mode():
        logits = model(**inputs).logits[0, -1].float().cpu()
    return ids, logits


def compare(model, processor, prompt, picture, device, max_tokens):
    inputs = server.user_turn.render(processor, prompt, server.user_turn.open_image(picture)).to(device)
    with_ids, with_logits = answer(model, inputs, max_tokens)
    with model.disable_adapter():
        base_ids, base_logits = answer(model, inputs, max_tokens)
    return {"picture": str(picture), "picture_sha256": hashlib.sha256(Path(picture).read_bytes()).hexdigest(),
            "prompt_tokens": int(inputs["input_ids"].shape[1]),
            "answers_differ": with_ids != base_ids, "first_differing_token": first_difference(with_ids, base_ids),
            "tokens_with_adapter": len(with_ids), "tokens_without": len(base_ids),
            "max_abs_first_logit_change": round(float((with_logits - base_logits).abs().max()), 6)}


def verdict(rows):
    criteria = {"pictures": len(rows) > 0,
                "first_logits_change_on_every_picture": all(r["max_abs_first_logit_change"] > 0 for r in rows),
                "answer_changes_on_some_picture": any(r["answers_differ"] for r in rows)}
    return criteria, all(criteria.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--pictures", nargs="+", required=True)
    ap.add_argument("--prompt", default=str(PROMPT))
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--device", default=None, choices=["cuda", "mps", "cpu"])
    ap.add_argument("--recurrent-layers", default=server.recurrent_layers.FLA,
                    choices=server.recurrent_layers.CHOICES)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    server.start(argparse.Namespace(model=args.model, revision=args.revision, adapter=args.adapter,
                                    served_name=f"{args.model}+adapter", device=args.device, timing=False,
                                    recurrent_layers=args.recurrent_layers))
    model, processor = server.STATE["model"], server.STATE["processor"]
    identity = server.STATE["identity"]
    prompt = Path(args.prompt).read_text(encoding="utf-8").strip()
    rows = [compare(model, processor, prompt, p, identity["device"], args.max_tokens) for p in args.pictures]
    criteria, passed = verdict(rows)
    result = {"check": "the adapter changes the output", "identity": identity,
              "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "max_tokens": args.max_tokens,
              "pictures": rows, "criteria": criteria, "passed": passed}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    for r in rows:
        print(f"{Path(r['picture']).name[:12]} answers differ {r['answers_differ']} (from token "
              f"{r['first_differing_token']}), first-logit change {r['max_abs_first_logit_change']}")
    print("passed" if passed else "FAILED", criteria, "->", args.out)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
