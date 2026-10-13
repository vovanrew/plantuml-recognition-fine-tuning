#!/usr/bin/env python3
"""Re-score the stored runs under scorer v1, every intermediate rule level, and scorer v2.

For each registered run and each level the two structural runners of the public scorer repo are
called with that level's switches. The level results go to
  <out>/_bridge/<level>/<metric>/<run>/<metric>_results.json
and the full v2 results (level `v2`, extracted blocks included) to
  <out>/<metric>/<run>/
`bridge_table.py` turns them into the bridge table.

Cumulative levels, in order (each adds one rule to the previous level):
  L0_v1            every switch off: scorer v1
  L1_whitespace    + whitespace runs collapsed            (Python)
  L2_unicode       + U+226A/U+226B stereotype brackets    (Python)
  L3a_markup       + names as drawn text, every line      (extractor: markup, spot letter)
  L3_names         + largest-font-size line(s) only       (extractor: package line)
  L4_direction     + one-arrowhead association directed   (extractor field + Python key)
  L5_first         + first diagram of a multi-diagram answer
  v2               + only the links and messages PlantUML draws are edges = scorer v2
                                                          (extractor; Decision, Amendment 7)
Leave-one-out levels: X_*. The six rules up to the first-diagram rule are left out of level
L5_first, one at a time, with every link and message an edge, as before Amendment 7; X_edges is
scorer v2 with the drawn-edge rule off (the same switches as L5_first).
Font levels (level L5_first with the base font size of unsized label text forced): F_*.
Every level but `v2` therefore passes `--undrawn-edges` to the extractor and gives what it gave
before the drawn-edge rule existed.

An extractor JAR built before the drawn-edge rule does not have the option. The script reads the
JAR's own usage text and, for such a JAR, runs the levels as they were then: no `L5_first`, no
`X_edges`, no `--undrawn-edges`, and `v2` is the first-diagram level.

Usage (from anywhere; Java on PATH):
  python3 bridge.py --scorer ../plantuml-recognition-evaluation \
      --data ../baseline-evaluation/data \
      --jar ../plantuml-recognition-evaluation/plantuml-stats-extractor-v2.jar \
      --out ../plantuml-recognition-evaluation/data_v2
"""
import argparse
import concurrent.futures
import json
import os
import shutil
import subprocess
import sys
import tempfile

PY_OFF = {"ws": "--no-collapse-whitespace", "br": "--no-unicode-brackets",
          "dir": "--no-association-direction", "first": "--last-diagram"}
EDGE_SWITCH = "--extractor-arg=--undrawn-edges"
EXTRACTOR_CLASS = "net.sourceforge.plantuml.stats.DiagramStatsExtractor"


def level(ws, br, names, direction, first, font=None, drawn_edges=False):
    """Runner switches for one level. names: raw | markup | drawn."""
    args = []
    if not ws:
        args.append(PY_OFF["ws"])
    if not br:
        args.append(PY_OFF["br"])
    if not direction:
        args += [PY_OFF["dir"], "--extractor-arg=--no-arrowheads"]
    if not first:
        args.append(PY_OFF["first"])
    if names != "drawn":
        args.append("--extractor-arg=--names=" + names)
    if font is not None:
        args.append("--extractor-arg=--name-font-size=%d" % font)
    if not drawn_edges:
        args.append(EDGE_SWITCH)
    return args


LEVELS = {
    "L0_v1":         level(False, False, "raw", False, False),
    "L1_whitespace": level(True, False, "raw", False, False),
    "L2_unicode":    level(True, True, "raw", False, False),
    "L3a_markup":    level(True, True, "markup", False, False),
    "L3_names":      level(True, True, "drawn", False, False),
    "L4_direction":  level(True, True, "drawn", True, False),
    "L5_first":      level(True, True, "drawn", True, True),
    "v2":            level(True, True, "drawn", True, True, drawn_edges=True),
    "X_whitespace":  level(False, True, "drawn", True, True),
    "X_unicode":     level(True, False, "drawn", True, True),
    "X_package":     level(True, True, "markup", True, True),
    "X_names":       level(True, True, "raw", True, True),
    "X_direction":   level(True, True, "drawn", False, True),
    "X_first":       level(True, True, "drawn", True, False),
    "X_edges":       level(True, True, "drawn", True, True),
    "F_12":          level(True, True, "drawn", True, True, font=12),
    "F_10":          level(True, True, "drawn", True, True, font=10),
}
METRICS = {"element_f1": "element_f1_runner.py", "relationship_f1": "relationship_f1_runner.py"}


def jar_has_drawn_edge_rule(jar):
    """True when the extractor JAR lists --undrawn-edges in its usage text."""
    proc = subprocess.run(["java", "-cp", jar, EXTRACTOR_CLASS], capture_output=True, text=True)
    return "--undrawn-edges" in proc.stderr + proc.stdout


def levels_before_drawn_edge_rule():
    """The levels as they were before the drawn-edge rule, for a JAR without the option."""
    return {name: [a for a in args if a != EDGE_SWITCH]
            for name, args in LEVELS.items() if name not in ("L5_first", "X_edges")}


def run_one(job):
    scorer, data, jar, out, run, name, metric = job
    final = name == "v2"
    dst = os.path.join(out, metric, run) if final else os.path.join(out, "_bridge", name, metric, run)
    work = dst if final else tempfile.mkdtemp(prefix="bridge_")
    os.makedirs(dst, exist_ok=True)
    cmd = [sys.executable, os.path.join("evaluation", METRICS[metric]),
           "--pred-dir", os.path.join(data, "runs", run),
           "--gt-dir", os.path.join(data, "puml_files"),
           "--test-set", os.path.join(data, "test_set.json"),
           "--csr", os.path.join(data, "csr", run, "csr_results.json"),
           "--out", work, "--jar", jar] + LEVELS[name]
    proc = subprocess.run(cmd, cwd=scorer, capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit("FAILED %s\n%s" % (" ".join(cmd), proc.stderr[-2000:]))
    if not final:
        shutil.copy(os.path.join(work, metric + "_results.json"), dst)
        shutil.rmtree(work)
    return run, name, metric


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scorer", required=True, help="checkout of plantuml-recognition-evaluation")
    ap.add_argument("--data", required=True, help="data root holding runs/, csr/, puml_files/, test_set.json")
    ap.add_argument("--jar", required=True, help="scorer-v2 extractor JAR")
    ap.add_argument("--out", required=True, help="v2 data root")
    ap.add_argument("--levels", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    scorer, data, jar, out = (os.path.abspath(p) for p in (args.scorer, args.data, args.jar, args.out))
    if not jar_has_drawn_edge_rule(jar):
        earlier = levels_before_drawn_edge_rule()
        LEVELS.clear()
        LEVELS.update(earlier)
        print("extractor JAR without --undrawn-edges: the %d levels of before the drawn-edge rule" % len(LEVELS),
              flush=True)
    if args.levels is None:
        args.levels = list(LEVELS)

    registry = json.load(open(os.path.join(scorer, "analysis", "model_registry.json")))["models"]
    runs = [m["run_dir"] for m in registry]
    jobs = [(scorer, data, jar, out, run, name, metric)
            for run in runs for name in args.levels for metric in METRICS]
    with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
        for i, (run, name, metric) in enumerate(pool.map(run_one, jobs), 1):
            print("[%3d/%d] %-14s %-16s %s" % (i, len(jobs), name, metric, run), flush=True)
    with open(os.path.join(out, "_bridge", "levels.json"), "w") as f:
        json.dump(LEVELS, f, indent=2)


if __name__ == "__main__":
    main()
