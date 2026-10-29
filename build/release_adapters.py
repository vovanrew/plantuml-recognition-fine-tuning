#!/usr/bin/env python3
"""Stage the study's adapters for Hugging Face, and check a staged or uploaded copy.

decisions/adapter-release/record.md, Decision, part 1. The released adapters are the five the
test-set runs served: one public model repository per base model, the 2B repository holding its
three adapters (20,000, 5,000 and 1,000 pairs), each adapter in its own subfolder. A repository
holds README.md (the model card, from build/release/model_card.md), LICENSE (Apache-2.0, the base
models' licence, build/release/apache-2.0.txt) and <subfolder>/adapter_model.safetensors and
adapter_config.json, copied byte for byte.

The adapter of a run is the checkpoint named in runs/test/<run>/adapter_check.json, and its files
must have the sha256 recorded there; the base revision is that of runs/train/<run>/run.json and
must equal the one the test run served.

Usage (from fine-tune/):
  python3 build/release_adapters.py stage <out folder>   # one folder per repository
  python3 build/release_adapters.py check <out folder>   # the staged copy against the runs
  python3 build/release_adapters.py check-hub            # the uploaded copy against the runs
Tests: build/tests/test_release_adapters.py
"""
import argparse
import hashlib
import json
import shutil
import string
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FT = HERE.parent
CARD = HERE / "release" / "model_card.md"
LICENCE = HERE / "release" / "apache-2.0.txt"
FILES = ("adapter_model.safetensors", "adapter_config.json")
OWNER = "vovanrew"
STUDY_URL = "https://github.com/vovanrew/plantuml-recognition-fine-tuning"
BENCHMARK_URL = "https://github.com/vovanrew/plantuml-recognition-evaluation"

# repository name -> base model, and its adapters in card order: (run, subfolder, pairs)
REPOS = {
    "Qwen3.5-2B-PlantUML-LoRA": ("Qwen/Qwen3.5-2B", [("2B_20000", "pairs-20000", 20000),
                                                      ("2B_5000", "pairs-5000", 5000),
                                                      ("2B_1000", "pairs-1000", 1000)]),
    "Qwen3.5-9B-PlantUML-LoRA": ("Qwen/Qwen3.5-9B", [("9B_20000", "pairs-20000", 20000)]),
    "Qwen3.5-27B-PlantUML-LoRA": ("Qwen/Qwen3.5-27B", [("27B_20000", "pairs-20000", 20000)]),
}


class Refused(Exception):
    pass


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def served(run, base=FT):
    """What the test run served: base model, revision, checkpoint folder and file sha256."""
    ident = json.loads((Path(base) / "runs" / "test" / run / "adapter_check.json").read_text())["identity"]
    trained = json.loads((Path(base) / "runs" / "train" / run / "run.json").read_text())
    if trained["revision"] != ident["revision"] or trained["model"] != ident["model"]:
        raise Refused(f"{run}: trained on {trained['model']}@{trained['revision']}, "
                      f"served {ident['model']}@{ident['revision']}")
    step = Path(ident["adapter"]["path"]).name
    return {"model": ident["model"], "revision": ident["revision"], "step": step,
            "folder": Path(base) / "runs" / "train" / run / step,
            "sha256": {f: ident["adapter"][f] for f in FILES}}


def card(repo, base=FT):
    """The model card of one repository."""
    model, adapters = REPOS[repo]
    infos = [(run, sub, pairs, served(run, base)) for run, sub, pairs in adapters]
    revisions = {i["revision"] for *_, i in infos}
    if len(revisions) != 1 or any(i["model"] != model for *_, i in infos):
        raise Refused(f"{repo}: adapters on different bases")
    rows = "\n".join(f"| `{sub}` | {pairs:,} | `{i['step']}` | `{i['sha256']['adapter_model.safetensors']}` |"
                     for run, sub, pairs, i in infos)
    many = len(infos) > 1
    return string.Template(CARD.read_text()).substitute(
        base=model, revision=revisions.pop(), repo_id=f"{OWNER}/{repo}", adapter_rows=rows,
        first_subfolder=infos[0][1], study_url=STUDY_URL, benchmark_url=BENCHMARK_URL,
        adapter_word="adapters" if many else "adapter", verb="turn" if many else "turns",
        They="They" if many else "It", are="are" if many else "is",
        model_word="models" if many else "model")


def stage(out, base=FT):
    out = Path(out)
    for repo, (_, adapters) in REPOS.items():
        d = out / repo
        if d.exists():
            raise Refused(f"{d} exists")
        d.mkdir(parents=True)
        (d / "README.md").write_text(card(repo, base))
        shutil.copyfile(LICENCE, d / "LICENSE")
        for run, sub, _ in adapters:
            info = served(run, base)
            (d / sub).mkdir()
            for f in FILES:
                shutil.copyfile(info["folder"] / f, d / sub / f)
    return check(out, base)


def check(out, base=FT):
    """Problems of a staged copy (empty when good)."""
    out, problems = Path(out), []
    for repo, (_, adapters) in REPOS.items():
        d = out / repo
        want = {"README.md", "LICENSE"} | {f"{sub}/{f}" for _, sub, _ in adapters for f in FILES}
        have = {p.relative_to(d).as_posix() for p in d.rglob("*") if p.is_file()} if d.is_dir() else set()
        problems += [f"{repo}: missing {p}" for p in sorted(want - have)]
        problems += [f"{repo}: not released {p}" for p in sorted(have - want)]
        if "README.md" in have and (d / "README.md").read_text() != card(repo, base):
            problems.append(f"{repo}: README.md is not the card")
        if "LICENSE" in have and (d / "LICENSE").read_bytes() != LICENCE.read_bytes():
            problems.append(f"{repo}: LICENSE is not the Apache-2.0 text")
        for run, sub, _ in adapters:
            info = served(run, base)
            for f in FILES:
                p = d / sub / f
                if p.is_file() and sha256_file(p) != info["sha256"][f]:
                    problems.append(f"{repo}/{sub}/{f}: sha256 is not the one {run} served")
    return problems


def check_hub(base=FT, api=None):
    """Problems of the uploaded copy: each adapter file's sha256 on the Hub, and the card."""
    if api is None:
        from huggingface_hub import HfApi
        api = HfApi()
    problems = []
    for repo, (_, adapters) in REPOS.items():
        repo_id = f"{OWNER}/{repo}"
        info = api.model_info(repo_id, files_metadata=True)
        files = {s.rfilename: s for s in info.siblings}
        for run, sub, _ in adapters:
            s = served(run, base)
            for f in FILES:
                entry = files.get(f"{sub}/{f}")
                if entry is None:
                    problems.append(f"{repo_id}: missing {sub}/{f}")
                    continue
                lfs = getattr(entry, "lfs", None)
                sha = getattr(lfs, "sha256", None) if lfs else None
                if sha is None:
                    from huggingface_hub import hf_hub_download
                    sha = sha256_file(hf_hub_download(repo_id, f"{sub}/{f}", revision=info.sha))
                if sha != s["sha256"][f]:
                    problems.append(f"{repo_id}/{sub}/{f}: sha256 on the Hub is not the one {run} served")
        for name in ("README.md", "LICENSE"):
            if name not in files:
                problems.append(f"{repo_id}: missing {name}")
    return problems


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("stage").add_argument("out")
    sub.add_parser("check").add_argument("out")
    sub.add_parser("check-hub")
    args = ap.parse_args(argv)
    try:
        if args.cmd == "stage":
            problems = stage(args.out)
        elif args.cmd == "check":
            problems = check(args.out)
        else:
            problems = check_hub()
    except Refused as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2
    for p in problems:
        print(p)
    print("OK" if not problems else f"{len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
