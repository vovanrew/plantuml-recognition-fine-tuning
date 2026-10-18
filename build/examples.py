#!/usr/bin/env python3
"""Training examples of the study: one pair of the frozen lists as the model sees it.

decisions/pilot-run/record.md. An example is the user turn the server builds for the pair's
picture (build/user_turn.py: the picture, a newline, the benchmark prompt; the model's own chat
template with the generation prompt and thinking off), then the token ids of the target, then the
end tokens. The labels are -100 on every position of the user turn, so the loss covers the target
and its end tokens only. Training appends to the served prompt, so the prefix the model is trained
on is the prompt it is served, token for token; build/tests/test_examples.py checks this through
infer/server.py's own request path.

The target is the text whose tokens element D4 of decisions/training-pool-rules counted and capped
at 5,376: build/pool.py `target_text` on the corpus source. No target is edited.

Which end tokens follow the target (`<|im_end|>` alone, or with the newline the template writes
after it) is a setting of the recipe; `end_ids` gives either.

Build (from fine-tune/, on the machine that holds the corpus):
  .venv/bin/python build/examples.py --out data/examples
writes train.jsonl (20,000 rows, in the order of data/sets/train.json), loss_curve.jsonl (200),
dev_runner_set.json (the 200 dev rows with the picture sizes infer_runner.py reads),
config.json and SHA256SUMS. A row of the .jsonl files: key, rank, primary_type, tier, picture
(path under fine-tune/), pixels_sha256 (from the render manifest, checked against the picture),
target, target_sha256. The training machine reads these files and the pictures; it needs no
corpus.

Tests: build/tests/test_examples.py
"""
import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
FT = HERE.parent
UNI = FT.parent
sys.path.insert(0, str(HERE))
import user_turn  # noqa: E402

SETS = FT / "data" / "sets"
PROMPT_FILE = UNI / "plantuml-recognition-evaluation" / "prompts" / "zero_shot.txt"
PROMPT_SHA256 = "097351ffce6134cb903322718facca5488f978766d46381de03934c8c9f01552"
IM_END, NEWLINE = 248046, 198            # <|im_end|> and "\n" of the Qwen3.5 tokenizer
TRAINING_SETS = ("train", "loss_curve")  # the lists a model trains on or draws a loss curve on


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def prompt_text(path=PROMPT_FILE):
    """The benchmark prompt as infer_runner.py reads it (`.strip()`); the file is pinned."""
    got = sha256_file(path)
    if got != PROMPT_SHA256:
        raise SystemExit(f"{path} has sha256 {got}, pinned {PROMPT_SHA256}")
    return Path(path).read_text(encoding="utf-8").strip()


# --------------------------------------------------------------------------- #
# The frozen lists and the pictures
# --------------------------------------------------------------------------- #
def set_rows(name, rank_at_most=None):
    """Rows of data/sets/<name>.json in their order; a subset is a bound on the rank."""
    rows = json.load(open(SETS / f"{name}.json", encoding="utf-8"))["diagrams"]
    return [r for r in rows if rank_at_most is None or r["rank"] <= rank_at_most]


def picture_path(name, key):
    """The picture of a key, relative to fine-tune/."""
    return f"data/sets/render/{name}/puml_images_1568/{key[:-len('.puml')]}.png"


def manifest(name):
    rows = {}
    with open(SETS / "render" / name / "manifest.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            rows[r["key"]] = r
    return rows


def pixels_sha256(path):
    """build/render.py's hash of a picture: its size and its RGB pixels."""
    from PIL import Image
    with Image.open(path) as im:
        rgb = im.convert("RGB")
    return hashlib.sha256(b"%dx%d:" % rgb.size + rgb.tobytes()).hexdigest()


def _pixels_of(path):
    return pixels_sha256(FT / path)


# --------------------------------------------------------------------------- #
# Targets
# --------------------------------------------------------------------------- #
def target_of(key, puml_dir=None):
    """The target of a key: the text D4 counted (build/pool.py)."""
    import pool
    path = Path(puml_dir or pool.PUML_DIR) / key
    return pool.target_text(pool.read_for_tokens(path))


def target_ids(tokenizer, target):
    return tokenizer(target, add_special_tokens=False)["input_ids"]


def end_ids(with_newline):
    """The tokens after the target that the loss covers: <|im_end|>, and the newline the
    template writes after it when `with_newline`."""
    return [IM_END, NEWLINE] if with_newline else [IM_END]


# --------------------------------------------------------------------------- #
# One example
# --------------------------------------------------------------------------- #
def example(processor, prompt, image, ids, ends):
    """(inputs, labels, number of prompt tokens) of one pair, tensors on the CPU, batch of one.

    `inputs` is user_turn.render of the prompt and the picture with `ids + ends` appended;
    `labels` equals the input ids with -100 on every prompt position."""
    import torch
    inputs = dict(user_turn.render(processor, prompt, image))
    n_prompt = int(inputs["input_ids"].shape[1])
    tail = torch.tensor([list(ids) + list(ends)], dtype=inputs["input_ids"].dtype)
    inputs["input_ids"] = torch.cat([inputs["input_ids"], tail], dim=1)
    inputs["attention_mask"] = torch.cat([inputs["attention_mask"], torch.ones_like(tail)], dim=1)
    if "mm_token_type_ids" in inputs:        # 0 marks a text position; the answer is text
        zeros = torch.zeros(tail.shape, dtype=inputs["mm_token_type_ids"].dtype)
        inputs["mm_token_type_ids"] = torch.cat([inputs["mm_token_type_ids"], zeros], dim=1)
    labels = inputs["input_ids"].clone()
    labels[:, :n_prompt] = -100
    return inputs, labels, n_prompt


def load_examples(path, rank_at_most=None):
    """Rows of an examples file; a subset is a bound on the rank."""
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    return [r for r in rows if rank_at_most is None or r["rank"] <= rank_at_most]


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
def build_rows(name, puml_dir=None, workers=8):
    """The examples of one list, every picture checked against its manifest hash."""
    shown = manifest(name)
    rows = []
    for r in set_rows(name):
        entry = shown.get(r["key"])
        if entry is None or entry["status"] != "ok":
            raise SystemExit(f"{name}: {r['key']} has no picture in the render manifest")
        target = target_of(r["key"], puml_dir)
        rows.append({"key": r["key"], "rank": r["rank"], "primary_type": r["primary_type"],
                     "tier": r["tier"], "picture": picture_path(name, r["key"]),
                     "pixels_sha256": entry["pixels_sha256"], "target": target,
                     "target_sha256": sha256_text(target)})
    with ProcessPoolExecutor(workers) as pool:
        got = list(pool.map(_pixels_of, [r["picture"] for r in rows], chunksize=64))
    wrong = [r["key"] for r, g in zip(rows, got) if g != r["pixels_sha256"]]
    if wrong:
        raise SystemExit(f"{name}: {len(wrong)} pictures differ from the manifest, e.g. {wrong[:3]}")
    return rows


def runner_set(name="dev"):
    """The rows of a list in the form infer_runner.py reads, with the picture's width and height."""
    shown = manifest(name)
    rows = []
    for r in set_rows(name):
        width, height = shown[r["key"]]["size"]
        rows.append({**r, "image_width": width, "image_height": height})
    return {"set": name, "from": f"data/sets/{name}.json; sizes from "
            f"data/sets/render/{name}/manifest.jsonl", "diagrams": rows}


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output folder, e.g. data/examples")
    ap.add_argument("--puml-dir", default=None, help="corpus sources (default build/pool.py PUML_DIR)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    import pool
    prompt_text()                                        # the pin is checked before anything is written
    counts = {}
    for name in TRAINING_SETS:
        rows = build_rows(name, args.puml_dir, args.workers)
        write_jsonl(out / f"{name}.jsonl", rows)
        counts[name] = len(rows)
        print(f"{name}: {len(rows)} examples", flush=True)
    with open(out / "dev_runner_set.json", "w", encoding="utf-8") as f:
        json.dump(runner_set("dev"), f, ensure_ascii=False, indent=1)
    config = {
        "decision": "decisions/pilot-run/record.md",
        "counts": counts,
        "sources": os.path.relpath(Path(args.puml_dir or pool.PUML_DIR), FT),
        "set_files_sha256": {f"{n}.json": sha256_file(SETS / f"{n}.json")
                             for n in (*TRAINING_SETS, "dev")},
        "prompt_sha256": PROMPT_SHA256,
        "code_sha256": {"build/examples.py": sha256_file(__file__),
                        "build/pool.py": sha256_file(HERE / "pool.py")},
        "extract_puml_source_sha256": pool.PINS["extract_puml_source"],
    }
    with open(out / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=1)
    names = sorted(p.name for p in out.iterdir() if p.is_file() and p.name != "SHA256SUMS")
    with open(out / "SHA256SUMS", "w", encoding="utf-8") as f:
        for name in names:
            f.write(f"{sha256_file(out / name)}  {name}\n")
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
