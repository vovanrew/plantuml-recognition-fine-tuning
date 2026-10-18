"""The manifest written beside each run of the study's server: what produced the run, and whether
every stored answer of it came from that one configuration.

decisions/gpu-precision/record.md, Decision Part 2, "Serving software": every response carries the
identity of what produced it, and a manifest is written beside each run. The runner
(`infer_runner.py` of the benchmark repository) is not changed: it stores each response verbatim,
`x_served_by` included, and `run_meta.json`. This file reads those and writes
`<run dir>/manifest.json`.

  write <run dir> [--server URL]   summarise the run, run the checks, write manifest.json;
                                   with --server, also compare the live server's identity
  check <run dir>                  recompute the summary and the checks and compare them with
                                   manifest.json; exit 1 on a failed check or a difference

Checks, each recorded with its result:
  one_configuration   every answered cell carries the same identity (the per-request fields
                      removed): a run cannot mix a base server and an adapter server, two
                      revisions, two GPU models or two implementations of the recurrent layers
  server_matches      with --server: the live server's identity is that one
  not_timing_mode     the server ran without --timing (no forced answer lengths)
  user_turn_file      the identity's hash of build/user_turn.py is the file's hash today
  rendered_turn       every answered cell carries the expected rendered-turn hash: by default the
                      frozen one of build/user_turn.py (picture, newline, prompts/zero_shot.txt);
                      a run with another prompt passes its own with --rendered-sha256
  decoding            run_meta: temperature 0, max_tokens 5,376, thinking off
  deadline            run_meta: the per-answer deadline is 600 s (author, 2026-10-17)
  no_timeouts         no cell ended at the deadline
  served_name         the runner asked for the name the server serves
  recurrent_layers    the run used the study's implementation of the recurrent layers,
                      flash-linear-attention (author, 2026-10-18); a run on the Mac fails it

Usage, from fine-tune/:
  .venv/bin/python infer/manifest.py write <run dir> --server http://127.0.0.1:8000
  .venv/bin/python infer/manifest.py check <run dir>
Tests: infer/tests/test_manifest.py
"""
import argparse
import datetime
import hashlib
import json
import subprocess
import sys
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "build"))
import recurrent_layers  # noqa: E402
import user_turn  # noqa: E402

PER_REQUEST = ("rendered_sha256", "min_tokens", "generate_seconds", "first_token_seconds")
DEADLINE_SECONDS = 600
MAX_TOKENS = 5376
NOT_CELLS = {"run_meta.json", "manifest.json"}


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity_of(served):
    """The configuration part of a response's `x_served_by`."""
    return {k: v for k, v in served.items() if k not in PER_REQUEST}


def canonical(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False)


def study_commit():
    """The study repository's commit, and whether build/ or infer/ differ from it."""
    try:
        def git(*args):
            return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True,
                                  check=True).stdout.strip()
        return {"commit": git("rev-parse", "HEAD"),
                "build_or_infer_differ_from_commit": git("status", "--porcelain", "--", "build", "infer") != ""}
    except (OSError, subprocess.CalledProcessError):
        return None


def summarise(run_dir):
    """Outcomes, configurations and rendered-turn hashes of the stored cells of one run."""
    outcomes, finish, rendered, names = Counter(), Counter(), Counter(), Counter()
    configurations = {}
    cells = sorted(p for p in Path(run_dir).glob("*.json") if p.name not in NOT_CELLS)
    for path in cells:
        record = json.loads(path.read_text(encoding="utf-8"))
        if "error" in record:
            outcomes[record["error"]] += 1
            continue
        served = record.get("x_served_by")
        if served is None:
            outcomes["ok_without_identity"] += 1
            continue
        outcomes["ok"] += 1
        finish[str(record["choices"][0].get("finish_reason"))] += 1
        names[str(record.get("model"))] += 1
        rendered[str(served.get("rendered_sha256"))] += 1
        identity = identity_of(served)
        configurations.setdefault(canonical(identity), {"identity": identity, "cells": 0})["cells"] += 1
    return {"cells": len(cells), "outcomes": dict(sorted(outcomes.items())),
            "finish_reason": dict(sorted(finish.items())), "model_names_answered": dict(sorted(names.items())),
            "rendered_sha256": dict(sorted(rendered.items())),
            "configurations": list(configurations.values())}


def checks(summary, run_meta, rendered_expected, server_identity=None):
    """(name, passed, detail) for every check of the module docstring."""
    configs = summary["configurations"]
    identity = configs[0]["identity"] if len(configs) == 1 else None
    answered = summary["outcomes"].get("ok", 0)
    extra = run_meta.get("extra_body") or {}
    out = [
        ("one_configuration", len(configs) == 1 and summary["outcomes"].get("ok_without_identity", 0) == 0,
         f"{len(configs)} configuration(s); {summary['outcomes'].get('ok_without_identity', 0)} answers without identity"),
        ("not_timing_mode", identity is not None and identity.get("timing") is False,
         f"timing {None if identity is None else identity.get('timing')}"),
        ("user_turn_file", identity is not None
         and identity.get("user_turn", {}).get("file_sha256") == user_turn.FILE_SHA256,
         f"identity {None if identity is None else identity.get('user_turn', {}).get('file_sha256')}, "
         f"file today {user_turn.FILE_SHA256}"),
        ("rendered_turn", answered > 0 and summary["rendered_sha256"].get(rendered_expected, 0) == answered,
         f"{summary['rendered_sha256'].get(rendered_expected, 0)} of {answered} answers carry {rendered_expected}"),
        ("decoding", run_meta.get("temperature") == 0 and run_meta.get("max_tokens") == MAX_TOKENS
         and extra.get("chat_template_kwargs") == {"enable_thinking": False},
         f"temperature {run_meta.get('temperature')}, max_tokens {run_meta.get('max_tokens')}, extra_body {extra}"),
        ("deadline", run_meta.get("timeout") == DEADLINE_SECONDS,
         f"timeout {run_meta.get('timeout')} s, the rule says {DEADLINE_SECONDS} s"),
        ("no_timeouts", summary["outcomes"].get("timeout", 0) == 0,
         f"{summary['outcomes'].get('timeout', 0)} cells timed out"),
        ("recurrent_layers", identity is not None
         and (identity.get("recurrent_layers") or {}).get("active") == recurrent_layers.FLA,
         f"active {None if identity is None else (identity.get('recurrent_layers') or {}).get('active')}, "
         f"the study's is {recurrent_layers.FLA}"),
        ("served_name", identity is not None and run_meta.get("model") == identity.get("served_name"),
         f"runner asked for {run_meta.get('model')!r}, server serves "
         f"{None if identity is None else identity.get('served_name')!r}"),
    ]
    if server_identity is not None:
        out.append(("server_matches", identity is not None and canonical(server_identity) == canonical(identity),
                    "the live server's identity equals the answers' one" if identity is not None
                    and canonical(server_identity) == canonical(identity) else "differs or no single identity"))
    return [{"check": n, "passed": bool(p), "detail": d} for n, p, d in out]


def build(run_dir, rendered_expected, server_identity=None):
    run_dir = Path(run_dir)
    meta_path = run_dir / "run_meta.json"
    if not meta_path.exists():
        raise SystemExit(f"{meta_path} not found: not a run directory of infer_runner.py")
    run_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    summary = summarise(run_dir)
    result = checks(summary, run_meta, rendered_expected, server_identity)
    return {
        "run_dir": str(run_dir), "run_meta_sha256": sha256_file(meta_path),
        "prompt_sha256": hashlib.sha256(run_meta.get("prompt", "").encode()).hexdigest(),
        "rendered_sha256_expected": rendered_expected,
        "files": {"infer/server.py": sha256_file(ROOT / "infer/server.py"),
                  "build/user_turn.py": user_turn.FILE_SHA256,
                  "build/recurrent_layers.py": sha256_file(ROOT / "build/recurrent_layers.py"),
                  "infer/manifest.py": sha256_file(Path(__file__))},
        "summary": summary, "checks": result, "passed": all(c["passed"] for c in result),
    }


def server_identity(url):
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(url.rstrip("/") + "/", timeout=10) as r:
        return json.loads(r.read())


def write(run_dir, rendered_expected, url=None):
    manifest = build(run_dir, rendered_expected, server_identity(url) if url else None)
    manifest.update(written=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    server_url=url, study_repository=study_commit())
    (Path(run_dir) / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    return manifest


def check(run_dir):
    """Failures: failed checks, and every field that differs from manifest.json."""
    stored = json.loads((Path(run_dir) / "manifest.json").read_text(encoding="utf-8"))
    now = build(run_dir, stored["rendered_sha256_expected"])
    problems = [f"check {c['check']} fails: {c['detail']}" for c in now["checks"] if not c["passed"]]
    for field in ("run_meta_sha256", "prompt_sha256", "files", "summary"):
        if canonical(now[field]) != canonical(stored[field]):
            problems.append(f"{field} differs from manifest.json")
    return problems


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    w = sub.add_parser("write")
    w.add_argument("run_dir")
    w.add_argument("--server", default=None, help="the URL of the server that answered the run")
    w.add_argument("--rendered-sha256", default=user_turn.RENDERED_SHA256)
    c = sub.add_parser("check")
    c.add_argument("run_dir")
    args = ap.parse_args()
    if args.command == "write":
        manifest = write(args.run_dir, args.rendered_sha256, args.server)
        for row in manifest["checks"]:
            print(("ok  " if row["passed"] else "FAIL"), row["check"], "-", row["detail"])
        print("wrote", Path(args.run_dir) / "manifest.json", "| passed" if manifest["passed"] else "| FAILED")
        sys.exit(0 if manifest["passed"] else 1)
    problems = check(args.run_dir)
    for p in problems:
        print("FAIL", p)
    print("manifest holds" if not problems else f"{len(problems)} problem(s)")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
