#!/usr/bin/env python3
"""Acceptance check of build/render.py: re-render the 1,000 test diagrams and compare with the
benchmark's images.

Compared, per diagram:
  final    <work>/render/puml_images_1568  against  baseline-evaluation/data/puml_images_1568
  raw      <work>/render/puml_images       against  baseline-evaluation/data/puml_images
           (the image after stages 1 and 2, before the downscale)
  stage 3  stage 3 applied to the benchmark's raw image, against the benchmark's final image
  stock    with --stock: the PNG file generate_images_hidpi.sh writes for the same source, against
           the file hash render.py recorded for its own stage 1 output

A source that calls %date() or %now() draws the time of the render, so its image changes from one
run to the next; such sources are listed as clock_sources and counted apart in the stock comparison.

The check passes when every diagram is rendered and both its sizes equal the benchmark's. Pixel
differences are counted and described, not judged: "share" is the share of pixels in which any
channel differs, "max" the largest difference in one channel on the 0-255 scale.

Usage (from fine-tune/):
  python3 build/acceptance_check.py --work <dir> --report <file.json> [--stock] [--limit N]
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops

sys.path.insert(0, str(Path(__file__).resolve().parent))
import render as R  # noqa: E402

BENCH = R.UNI / "baseline-evaluation/data"
_CLOCK = re.compile(r"%(date|now)\b")


def compare(pair):
    a_path, b_path = pair
    with Image.open(a_path) as a, Image.open(b_path) as b:
        a, b = a.convert("RGB"), b.convert("RGB")
    if a.size != b.size:
        return dict(same_size=False, sizes=[list(a.size), list(b.size)])
    diff = np.asarray(ImageChops.difference(a, b))
    top = int(diff.max())
    if top == 0:
        return dict(same_size=True, identical=True)
    return dict(same_size=True, identical=False, max=top,
                share=float((diff.max(axis=2) > 0).mean()))


def stage3_of_benchmark_raw(base):
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, base + ".png")
        R.stage_code()[1].standardize(str(BENCH / "puml_images" / (base + ".png")), out,
                                      R.stage_code()[1].LONG_EDGE)
        return compare((out, str(BENCH / "puml_images_1568" / (base + ".png"))))


def run_stock(keys, work, java_env):
    """generate_images_hidpi.sh as committed, over the same sources. Returns {key: sha256 or None}."""
    din, dout = work / "in", work / "out"
    shutil.rmtree(work, ignore_errors=True)
    din.mkdir(parents=True)
    for k in keys:
        shutil.copy(BENCH / "puml_files" / k, din / k)
    r = subprocess.run(["bash", str(R.STAGE1_SCRIPT), str(din), str(dout)], env=java_env,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    hashes = {}
    for k in keys:
        p = dout / (k[:-5] + ".png")
        hashes[k] = R.sha256_file(p) if p.is_file() else None
    errors = (dout / "errors.log").read_text(errors="replace") if (dout / "errors.log").is_file() else ""
    return r.returncode, hashes, errors.count("OutOfMemoryError")


def describe(rows, field):
    d = [r[field] for r in rows if r.get(field) and not r[field].get("identical", True)]
    if not d:
        return {}
    return dict(n=len(d), max_le_16=sum(1 for x in d if x["max"] <= 16),
                max_17_to_64=sum(1 for x in d if 16 < x["max"] <= 64),
                max_gt_64=sum(1 for x in d if x["max"] > 64),
                share_lt_0_05_percent=sum(1 for x in d if x["share"] < 0.0005),
                largest_share_percent=round(100 * max(x["share"] for x in d), 4))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--work", required=True, help="scratch directory; its render/ is rebuilt")
    ap.add_argument("--report", help="where to write the JSON report (default: <work>/report.json)")
    ap.add_argument("--stock", action="store_true", help="also run the stock stage 1 script and compare file hashes")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--heap-mb", type=int, default=32768)
    args = ap.parse_args()
    work = Path(args.work).resolve()
    diagrams = json.load(open(BENCH / "test_set.json"))["diagrams"]
    if args.limit:
        diagrams = diagrams[:args.limit]
    keys = [d["key"] for d in diagrams]

    t0 = time.time()
    shutil.rmtree(work / "render", ignore_errors=True)
    summary = R.render(keys, BENCH / "puml_files", work / "render", threads=args.threads,
                       heap_mb=args.heap_mb, keep_raw=True)
    render_seconds = round(time.time() - t0, 1)
    manifest = {r["key"]: r for r in R.read_manifest(work / "render/manifest.jsonl")}
    env = json.loads((work / "render/render_env.json").read_text())

    ok = [k for k in keys if manifest[k]["status"] == "ok"]
    name = lambda k: k[:-5] + ".png"
    with ProcessPoolExecutor(max_workers=6) as pool:
        final = dict(zip(ok, pool.map(compare, [(str(work / "render/puml_images_1568" / name(k)),
                                                 str(BENCH / "puml_images_1568" / name(k))) for k in ok])))
        raw = dict(zip(ok, pool.map(compare, [(str(work / "render/puml_images" / name(k)),
                                               str(BENCH / "puml_images" / name(k))) for k in ok])))
        stage3 = dict(zip(keys, pool.map(stage3_of_benchmark_raw, [k[:-5] for k in keys])))
    clock = [k for k in keys if _CLOCK.search((BENCH / "puml_files" / k).read_text(errors="replace"))]
    stock = None
    if args.stock:
        _, java_env = R.java_env(guard=False)   # the stock script calls dot itself
        stock_exit, stock_hashes, stock_oom = run_stock(keys, work / "stock", java_env)
        differing = [k for k in ok if stock_hashes[k] != manifest[k]["render_sha256"]]
        stock = dict(script_exit=stock_exit, out_of_memory_traces=stock_oom,
                     files_written=sum(1 for h in stock_hashes.values() if h),
                     same_bytes_as_render_py=len(ok) - len(differing),
                     differing_keys=differing[:20],
                     differing_keys_that_are_not_clock_sources=[k for k in differing if k not in clock][:20])

    rows, cells = [], {}
    for d in diagrams:
        k = d["key"]
        row = dict(key=k[:-5], type=d["primary_type"], tier=d["tier"], status=manifest[k]["status"],
                   attempt=manifest[k].get("attempt"), renderer_trace=manifest[k].get("renderer_trace"),
                   clock_source=k in clock, final=final.get(k), raw=raw.get(k), stage3=stage3[k])
        rows.append(row)
        for cell in ("all", d["primary_type"], f'{d["primary_type"]} tier {d["tier"]}'):
            c = cells.setdefault(cell, Counter())
            c["n"] += 1
            c["rendered"] += row["status"] == "ok"
            c["final_same_size"] += bool(row["final"] and row["final"]["same_size"])
            c["final_identical"] += bool(row["final"] and row["final"].get("identical"))
            c["raw_same_size"] += bool(row["raw"] and row["raw"]["same_size"])
            c["raw_identical"] += bool(row["raw"] and row["raw"].get("identical"))
            c["stage3_of_benchmark_raw_identical"] += bool(row["stage3"].get("identical"))
    n = len(keys)
    passed = cells["all"]["rendered"] == n and cells["all"]["final_same_size"] == n and cells["all"]["raw_same_size"] == n
    report = dict(
        passed=passed, render_seconds=render_seconds, render_summary=summary,
        environment={k: env[k] for k in ("renderer", "java", "graphviz", "dot_guard", "pillow", "platform", "stage_code", "parameters")},
        run=env["runs"][-1], cells={k: dict(v) for k, v in cells.items()},
        final_differences=describe(rows, "final"), raw_differences=describe(rows, "raw"),
        final_differences_by_type={t: describe([r for r in rows if r["type"] == t], "final") for t in ("class", "sequence")},
        stock=stock, clock_sources=clock,
        renderer_traces={k: manifest[k]["renderer_trace"] for k in keys if manifest[k].get("renderer_trace")},
        not_identical=[r for r in rows if r["status"] != "ok" or not (r["final"] or {}).get("identical")
                       or not (r["raw"] or {}).get("identical") or not r["stage3"].get("identical")])
    report_path = Path(args.report) if args.report else work / "report.json"
    report_path.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: report[k] for k in ("passed", "render_seconds", "cells", "final_differences",
                                             "raw_differences", "final_differences_by_type", "stock",
                                             "clock_sources", "renderer_traces")}, indent=1))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
