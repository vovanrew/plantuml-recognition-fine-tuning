#!/usr/bin/env python3
"""Pack the study's released runs into runs.zip, and check a runs.zip.

decisions/adapter-release/record.md, Decision, part 2. runs/ is local working output (plan §9);
runs.zip is the copy of it that the repository ships, as the benchmark ships data.zip. A reader
unpacks it at the repository root:  unzip runs.zip

What goes in (paths relative to fine-tune/):
  runs/test/**                    the test-set runs: answers, manifests, logs, scores, the tables
                                  in runs/test/_analysis/, and the data roots in runs/test/_root/
                                  (relative links only)
  runs/dev/**                     the dev answers and scores of every checkpoint of every run,
                                  except the PNG renders the CSR runner leaves in scores/csr/png/
                                  (no score reads them; the test runs keep none)
  runs/train/<run>/run.json, log.jsonl, selection.json
                                  the run's settings, its step log and its checkpoint choice; the
                                  adapters are on Hugging Face, and checkpoints and optimizer
                                  states are not released

The archive is the same bytes for the same files: entries sorted by path, every timestamp
1980-01-01 00:00, modes 0644 for files, links stored as links with their relative target.

Usage (from fine-tune/):
  python3 build/pack_runs.py               # writes runs.zip
  python3 build/pack_runs.py --check       # runs.zip against runs/, and every test run's
                                           # scores/manifest.json against the archive
Tests: build/tests/test_pack_runs.py
"""
import argparse
import hashlib
import json
import os
import stat
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
FT = HERE.parent
TRAIN_FILES = ("run.json", "log.jsonl", "selection.json")
DATE = (1980, 1, 1, 0, 0, 0)
FILE_MODE = stat.S_IFREG | 0o644
LINK_MODE = stat.S_IFLNK | 0o777


class Refused(Exception):
    pass


def _walk(top):
    """Every file and link under top, links not followed."""
    for dirpath, dirnames, filenames in os.walk(top, followlinks=False):
        here = Path(dirpath)
        for name in dirnames:
            p = here / name
            if p.is_symlink():
                yield p
        for name in filenames:
            yield here / name


def _excluded_dev(rel):
    parts = rel.split("/")
    return any(parts[i:i + 3] == ["scores", "csr", "png"] for i in range(len(parts) - 2))


def select(base=FT):
    """The released paths, relative to base, sorted. Refused if a train run lacks a file."""
    base = Path(base)
    out = []
    for p in _walk(base / "runs" / "test"):
        out.append(p.relative_to(base).as_posix())
    for p in _walk(base / "runs" / "dev"):
        rel = p.relative_to(base).as_posix()
        if not _excluded_dev(rel):
            out.append(rel)
    train = base / "runs" / "train"
    for run in sorted(d for d in train.iterdir() if d.is_dir() and not d.name.startswith(".")):
        for name in TRAIN_FILES:
            f = run / name
            if not f.is_file():
                raise Refused(f"{f} is missing")
            out.append(f.relative_to(base).as_posix())
    out = [r for r in out if Path(r).name != ".DS_Store"]
    return sorted(out)


def _entry(base, rel):
    """(ZipInfo, bytes) for one released path."""
    p = Path(base) / rel
    info = zipfile.ZipInfo(rel, date_time=DATE)
    info.create_system = 3
    if p.is_symlink():
        target = os.readlink(p)
        if os.path.isabs(target):
            raise Refused(f"{rel} links to an absolute path {target}")
        info.external_attr = LINK_MODE << 16
        info.compress_type = zipfile.ZIP_STORED
        return info, target.encode("utf-8")
    info.external_attr = FILE_MODE << 16
    info.compress_type = zipfile.ZIP_DEFLATED
    return info, p.read_bytes()


def pack(base=FT, out=None):
    """Write the archive; returns the number of entries."""
    base = Path(base)
    out = Path(out) if out else base / "runs.zip"
    rels = select(base)
    tmp = out.with_name(out.name + ".part")
    with zipfile.ZipFile(tmp, "w") as z:
        for rel in rels:
            info, data = _entry(base, rel)
            z.writestr(info, data, compresslevel=9 if info.compress_type == zipfile.ZIP_DEFLATED else None)
    os.replace(tmp, out)
    return len(rels)


def _is_link(info):
    return stat.S_ISLNK(info.external_attr >> 16)


def check(base=FT, archive=None):
    """Problems found (empty when the archive is good)."""
    base = Path(base)
    archive = Path(archive) if archive else base / "runs.zip"
    problems = []
    with zipfile.ZipFile(archive) as z:
        infos = {i.filename: i for i in z.infolist()}
        expected = select(base)
        missing = sorted(set(expected) - set(infos))
        extra = sorted(set(infos) - set(expected))
        problems += [f"missing from the archive: {r}" for r in missing]
        problems += [f"not a released path: {r}" for r in extra]
        for rel in expected:
            if rel not in infos:
                continue
            want_info, want = _entry(base, rel)
            got = z.read(infos[rel])
            if _is_link(infos[rel]) != _is_link(want_info):
                problems.append(f"link or file differs: {rel}")
            elif got != want:
                problems.append(f"bytes differ: {rel}")
        # each test run's score manifest against the archive's own bytes
        for rel in sorted(infos):
            parts = rel.split("/")
            if len(parts) == 5 and parts[:2] == ["runs", "test"] and parts[3:] == ["scores", "manifest.json"]:
                files = json.loads(z.read(infos[rel])).get("files", {})
                for sub, sha in sorted(files.items()):
                    member = f"runs/test/{parts[2]}/scores/{sub}"
                    if member not in infos:
                        problems.append(f"{rel} names {sub}, absent from the archive")
                    elif hashlib.sha256(z.read(infos[member])).hexdigest() != sha:
                        problems.append(f"{rel}: sha256 of {sub} differs")
    return problems


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", default=str(FT), help="folder holding runs/ (default: fine-tune/)")
    ap.add_argument("--zip", default=None, help="archive path (default: <base>/runs.zip)")
    ap.add_argument("--check", action="store_true", help="check the archive instead of writing it")
    args = ap.parse_args(argv)
    try:
        if args.check:
            problems = check(args.base, args.zip)
            for p in problems:
                print(p)
            print("runs.zip: OK" if not problems else f"runs.zip: {len(problems)} problem(s)")
            return 1 if problems else 0
        n = pack(args.base, args.zip)
        print(f"wrote {args.zip or Path(args.base) / 'runs.zip'}: {n} entries")
        return 0
    except Refused as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
