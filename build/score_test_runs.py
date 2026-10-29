#!/usr/bin/env python3
"""Score stored test-set runs under scorer v2, with scorer v1 alongside.

decisions/main-tables-intervals/record.md, Decision, "Scoring (1A, L2)". One command scores a run
of runs/test/<run>/ with the benchmark's own runners, unchanged:

  CSR -> Element F1, Relationship F1 (v2) -> chrF++ -> Element F1, Relationship F1 (v1)

Scorer v1 is route (i) of methodology/scorer-v2.md §7: the v2 code and the v2 extractor JAR with
the seven rule switches. CSR and chrF++ do not depend on the scorer version and are scored once.
The renderer, the test set and the test ground truth (the benchmark's data/puml_files) are passed
to the runners explicitly; the runners run from the benchmark's folder.

An answer on which the frozen extractor's JVM dies (a Java Error such as StackOverflowError, for
example a self-referential `!define`, escapes the extractor's per-file handler, which writes an
empty `parse_error` record for any other failure, and the runner exits) is held back from the
F1 runners' prediction folder of that scorer version, so the runner scores it as it scores every
non-parsing answer: an empty graph. CSR and chrF++ read every answer. The manifest names each
held-back key (`extractor_crashes`).

Refusals, before anything is scored:
  - any part of the scorer that is not the frozen one (train/dev_scoring.py check_pins);
  - a test set whose sha256 is not the frozen one;
  - an answers/ folder without exactly 1,000 .puml files whose stems equal the test-set keys;
  - a run manifest (answers/manifest.json) that does not pass;
  - an answer_manifest.json whose set_file_sha256 is not the frozen one;
  - an existing scores/ folder, unless a re-run gives the same bytes in every result file.

Output, per run (the loader's layout under each scorer version):
  runs/test/<run>/scores/v2/{csr,element_f1,relationship_f1,chrf}/<run>/<metric>_results.json
  runs/test/<run>/scores/v1/{element_f1,relationship_f1}/<run>/<metric>_results.json
  runs/test/<run>/scores/v1/{csr,chrf} -> ../v2/{csr,chrf}             (links)
  runs/test/<run>/scores/manifest.json                                  (sha256 of every result file)
and a data root of links per scorer version, which the benchmark's loader and leak gate read:
  runs/test/_root/{v2,v1}/test_set.json, puml_images_1568, runs/<run> -> the run's answers/,
  {csr,element_f1,relationship_f1,chrf}/<run> -> the run's score folder.

Usage (from fine-tune/):
  python3 build/score_test_runs.py 2B_zero_shot 2B_20000 ...
Tests: build/tests/test_score_test_runs.py
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
FT = HERE.parent
sys.path.insert(0, str(FT / "train"))
import dev_scoring  # noqa: E402

sys.path.insert(0, str(dev_scoring.INSTRUMENT / "evaluation"))
from element_f1_runner import EXTRACTOR_CLASS, graphs_from_output, write_extracted  # noqa: E402

INSTRUMENT = dev_scoring.INSTRUMENT
TEST_SET = INSTRUMENT / "data" / "test_set.json"
GT_DIR = INSTRUMENT / "data" / "puml_files"
IMAGES = INSTRUMENT / "data" / "puml_images_1568"
RUNS_DIR = FT / "runs" / "test"
ROOT = RUNS_DIR / "_root"
FROZEN_TEST_SET_SHA256 = "e233e56968f90c7411d7c6b27cdd5c81f801fcc47fd1c7b45d174918a440744c"
N_CELLS = 1000

# methodology/scorer-v2.md §7: with all seven, the v2 code and the v2 JAR are scorer v1.
V1_SWITCHES = ["--no-collapse-whitespace", "--no-unicode-brackets", "--no-association-direction",
               "--last-diagram", "--extractor-arg=--names=raw", "--extractor-arg=--no-arrowheads",
               "--extractor-arg=--undrawn-edges"]
F1_RUNNERS = {"element_f1": "element_f1_runner.py", "relationship_f1": "relationship_f1_runner.py"}
METRICS = ("csr", "element_f1", "relationship_f1", "chrf")
ONCE = ("csr", "chrf")          # scorer-independent: scored once, linked into v1
WORK = ".scoring"               # work folder inside the run folder, removed after each run


class Refused(SystemExit):
    """A run, a test set or a score folder the command does not score."""


def sha256_file(path):
    return dev_scoring.sha256_file(path)


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------- #
# refusals
# --------------------------------------------------------------------------- #
def test_set_keys(test_set):
    """The test set's keys without the .puml suffix; Refused unless it is the frozen file."""
    got = sha256_file(test_set) if Path(test_set).is_file() else None
    if got != FROZEN_TEST_SET_SHA256:
        raise Refused(f"test set {test_set} has sha256 {got}, frozen {FROZEN_TEST_SET_SHA256}")
    keys = [d["key"][:-5] if d["key"].endswith(".puml") else d["key"]
            for d in json.load(open(test_set, encoding="utf-8"))["diagrams"]]
    if len(keys) != N_CELLS or len(set(keys)) != N_CELLS:
        raise Refused(f"test set {test_set} does not hold {N_CELLS} distinct keys")
    return keys


def check_run(run_dir, keys):
    """What the run is scored from; Refused if it is not a complete, passed run of the frozen
    test set."""
    run_dir = Path(run_dir)
    answers = run_dir / "answers"
    if not answers.is_dir():
        raise Refused(f"{run_dir}: no answers/ folder")
    stems = sorted(p.stem for p in answers.glob("*.puml"))
    if len(stems) != N_CELLS:
        raise Refused(f"{answers}: {len(stems)} .puml files, not {N_CELLS}")
    if set(stems) != set(keys):
        only_run, only_set = sorted(set(stems) - set(keys)), sorted(set(keys) - set(stems))
        raise Refused(f"{answers}: .puml stems differ from the test-set keys "
                      f"({len(only_run)} not in the test set, e.g. {only_run[:2]}; "
                      f"{len(only_set)} keys without a .puml, e.g. {only_set[:2]})")
    manifest = answers / "manifest.json"
    if not manifest.is_file():
        raise Refused(f"{answers}: no run manifest (manifest.json)")
    if json.load(open(manifest, encoding="utf-8")).get("passed") is not True:
        raise Refused(f"{manifest}: the run manifest does not pass")
    answer_manifest = run_dir / "answer_manifest.json"
    if not answer_manifest.is_file():
        raise Refused(f"{run_dir}: no answer_manifest.json")
    set_sha = json.load(open(answer_manifest, encoding="utf-8")).get("set_file_sha256")
    if set_sha != FROZEN_TEST_SET_SHA256:
        raise Refused(f"{answer_manifest}: set_file_sha256 {set_sha}, frozen {FROZEN_TEST_SET_SHA256}")
    return {"answers": str(answers.resolve()), "run_manifest_sha256": sha256_file(manifest),
            "answer_manifest_sha256": sha256_file(answer_manifest), "set_file_sha256": set_sha}


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
def result_files(run):
    """Every result file of a run, relative to its scores/ folder: {relpath: (version, metric)}.
    The v1 CSR and chrF++ paths go through the links to v2."""
    out = {}
    for version in ("v2", "v1"):
        for metric in METRICS:
            out[f"{version}/{metric}/{run}/{metric}_results.json"] = (version, metric)
    return out


def _runner(python, script, arguments, env, log):
    command = [python, os.path.join("evaluation", script)] + arguments
    proc = subprocess.run(command, cwd=str(INSTRUMENT), env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True)
    with open(log, "a", encoding="utf-8") as f:
        f.write("$ " + " ".join(command) + "\n" + proc.stdout + "\n")
    if proc.returncode != 0:
        raise SystemExit(f"runner failed (exit {proc.returncode}): {' '.join(command)}\n"
                         f"{proc.stdout[-2000:]}")


def _extract(java, extractor_args, folder):
    return subprocess.run([java, "-cp", str(dev_scoring.EXTRACTOR_JAR), EXTRACTOR_CLASS,
                           *extractor_args, "--dir", str(folder)], capture_output=True, text=True)


def extractor_crashes(answers, work, extractor_args, java):
    """Stems of the answers on which the extractor's JVM dies when run on that answer alone,
    on the blocks the F1 runners give it (write_extracted). Every answer is first run in one
    batch, as the runners run it; only when the batch dies are the answers it did not reach
    tried one at a time, in order, until the one that dies alone is found, and the batch is run
    again without it."""
    work = Path(work)
    blocks = work / "crash_check"
    shutil.rmtree(blocks, ignore_errors=True)
    write_extracted(sorted(str(p) for p in Path(answers).glob("*.puml")), str(blocks))
    remaining, crashed = sorted(p.name for p in blocks.glob("*.puml")), []
    while remaining:
        batch = work / "crash_batch"
        shutil.rmtree(batch, ignore_errors=True)
        batch.mkdir()
        for name in remaining:
            shutil.copyfile(blocks / name, batch / name)
        proc = _extract(java, extractor_args, batch)
        if proc.returncode == 0:
            break
        reached = set(graphs_from_output(proc.stdout))
        left = [name for name in remaining if name[:-5] not in reached]
        found = None
        for name in left:
            single = work / "crash_single"
            shutil.rmtree(single, ignore_errors=True)
            single.mkdir()
            shutil.copyfile(blocks / name, single / name)
            if _extract(java, extractor_args, single).returncode != 0:
                found = name
                break
        if found is None:
            raise SystemExit(f"the extractor fails on {answers} but on no single answer:\n{proc.stderr[-2000:]}")
        crashed.append(found)
        remaining = [name for name in left if name != found]
    for folder in ("crash_check", "crash_batch", "crash_single"):
        shutil.rmtree(work / folder, ignore_errors=True)
    return sorted(name[:-5] for name in crashed)


def score_into(work, run, answers, test_set, gt_dir, render_jar, java_home, python):
    """Run the four runners, in the decided order, into <work>/<version>/<metric>/<run>/.
    Returns the seconds each step took and the answers held back from each version's F1
    runners (extractor_crashes)."""
    work = Path(work)
    java = str(Path(java_home) / "bin" / "java")
    crashes, f1_answers = {}, {}
    for version, switches in (("v2", []), ("v1", V1_SWITCHES)):
        extractor_args = [a.split("=", 1)[1] for a in switches if a.startswith("--extractor-arg=")]
        crashes[version] = extractor_crashes(answers, work, extractor_args, java)
        f1_answers[version] = answers
        if crashes[version]:
            held = work / f"{version}_f1_answers"
            held.mkdir()
            for p in sorted(Path(answers).glob("*.puml")):
                if p.stem not in crashes[version]:
                    os.symlink(p, held / p.name)
            f1_answers[version] = str(held)
    env = dict(os.environ, PATH=str(Path(java_home) / "bin") + os.pathsep + os.environ["PATH"])
    log = work / "runners.log"
    seconds = {}

    def out(version, metric):
        d = work / version / metric / run
        d.mkdir(parents=True, exist_ok=True)
        return str(d)

    csr = os.path.join(out("v2", "csr"), "csr_results.json")
    t = time.time()
    _runner(python, "csr_runner.py", ["--pred-dir", answers, "--test-set", test_set,
                                      "--out", out("v2", "csr"), "--jar", render_jar], env, log)
    seconds["csr"] = round(time.time() - t, 1)
    def common(version):
        return ["--pred-dir", f1_answers[version], "--gt-dir", gt_dir, "--test-set", test_set,
                "--csr", csr, "--jar", str(dev_scoring.EXTRACTOR_JAR)]
    for metric, script in F1_RUNNERS.items():
        t = time.time()
        _runner(python, script, common("v2") + ["--out", out("v2", metric)], env, log)
        seconds[f"v2_{metric}"] = round(time.time() - t, 1)
    t = time.time()
    _runner(python, "chrf_runner.py", ["--pred-dir", answers, "--gt-dir", gt_dir,
                                       "--test-set", test_set, "--csr", csr,
                                       "--out", out("v2", "chrf")], env, log)
    seconds["chrf"] = round(time.time() - t, 1)
    for metric, script in F1_RUNNERS.items():
        t = time.time()
        _runner(python, script, common("v1") + ["--out", out("v1", metric)] + V1_SWITCHES, env, log)
        seconds[f"v1_{metric}"] = round(time.time() - t, 1)
    return seconds, crashes


def install(work, scores, run):
    """Copy the result files (only) from the work folder into scores/ and link v1's CSR and
    chrF++ to v2's."""
    files = {}
    for rel, (version, metric) in result_files(run).items():
        if version == "v1" and metric in ONCE:
            continue
        dst = Path(scores) / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(work) / rel, dst)
        files[rel] = sha256_file(dst)
    links = {}
    for metric in ONCE:
        link = Path(scores) / "v1" / metric
        link.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(os.path.join("..", "v2", metric), link)
        links[f"v1/{metric}"] = os.readlink(link)
    return files, links


def differing(work, scores, run):
    """Result files of a re-run (in work) whose bytes differ from the stored ones."""
    out = []
    for rel, (version, metric) in result_files(run).items():
        if version == "v1" and metric in ONCE:
            stored = Path(scores) / "v1" / metric
            if not (stored.is_symlink() and os.readlink(stored) == os.path.join("..", "v2", metric)):
                out.append(f"v1/{metric} (not the link to v2)")
            continue
        new, old = Path(work) / rel, Path(scores) / rel
        if not old.is_file() or new.read_bytes() != old.read_bytes():
            out.append(rel)
    return out


# --------------------------------------------------------------------------- #
# data roots of links
# --------------------------------------------------------------------------- #
def _link(link, target):
    """A relative link at `link` to `target`; Refused if something else is already there."""
    link, target = Path(link), Path(target)
    rel = os.path.relpath(os.path.abspath(target), os.path.abspath(link.parent))
    if link.is_symlink():
        if os.readlink(link) != rel:
            raise Refused(f"{link} links to {os.readlink(link)}, not {rel}")
        return
    if link.exists():
        raise Refused(f"{link} exists and is not a link")
    link.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(rel, link)


def link_roots(root, run_dir, run, test_set, images):
    """Add one run to the data root of each scorer version."""
    scores = Path(run_dir) / "scores"
    for version in ("v2", "v1"):
        r = Path(root) / version
        r.mkdir(parents=True, exist_ok=True)
        _link(r / "test_set.json", test_set)
        _link(r / "puml_images_1568", images)
        _link(r / "runs" / run, Path(run_dir) / "answers")
        for metric in METRICS:
            _link(r / metric / run, scores / version / metric / run)


# --------------------------------------------------------------------------- #
# one run
# --------------------------------------------------------------------------- #
def score_run(run, runs_dir=RUNS_DIR, root=ROOT, test_set=TEST_SET, gt_dir=GT_DIR,
              images=IMAGES, render_jar=dev_scoring.RENDER_JAR, java_home=dev_scoring.JAVA_HOME,
              python="/usr/bin/python3", scorer=None):
    """Score one run (or confirm a byte-equal re-run) and link it into the data roots.
    `scorer` is check_pins()'s result; it is checked here when not given."""
    test_set, gt_dir, images, render_jar = (str(Path(p).resolve()) for p in
                                            (test_set, gt_dir, images, render_jar))
    if scorer is None:
        scorer = dev_scoring.check_pins(render_jar, dev_scoring.EXTRACTOR_JAR, java_home, python)
    keys = test_set_keys(test_set)
    run_dir = Path(runs_dir) / run
    info = check_run(run_dir, keys)
    scores = run_dir / "scores"
    rerun = scores.exists() or scores.is_symlink()
    # A fixed work folder: CSR's per-diagram error text carries the path of the file it compiled,
    # so a re-run is byte-equal only when it compiles from the same path.
    work = run_dir / WORK
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir()
    try:
        started = time.time()
        seconds, crashes = score_into(work, run, info["answers"], test_set, gt_dir, render_jar,
                                      java_home, python)
        if rerun:
            diff = differing(work, scores, run)
            if diff:
                raise Refused(f"{scores} exists and a re-run differs in {len(diff)} file(s): "
                              f"{diff[:4]}; move it away to score again")
            status = "byte-equal re-run; stored scores kept"
        else:
            scores.mkdir()
            files, links = install(work, scores, run)
            manifest = {"run": run, "scored": now(), **info,
                        "test_set": test_set, "test_set_sha256": FROZEN_TEST_SET_SHA256,
                        "gt_dir": gt_dir, "render_jar": render_jar, "scorer": scorer,
                        "v1_route": "v2 code and v2 extractor JAR with the seven rule switches "
                                    "(methodology/scorer-v2.md §7)",
                        "v1_switches": V1_SWITCHES,
                        "extractor_crashes": crashes,
                        "score_test_runs_sha256": sha256_file(__file__),
                        "files": files, "links": links, "seconds": seconds,
                        "seconds_total": round(time.time() - started, 1)}
            with open(scores / "manifest.json", "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=1)
            status = "scored"
    finally:
        shutil.rmtree(work, ignore_errors=True)
    link_roots(root, run_dir, run, test_set, images)
    return {"run": run, "status": status, "scores": str(scores), "seconds": seconds,
            "extractor_crashes": crashes}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="+", help="run names under --runs-dir, e.g. 2B_zero_shot")
    ap.add_argument("--runs-dir", default=str(RUNS_DIR))
    ap.add_argument("--root", default=None, help="data roots (default: <runs-dir>/_root)")
    ap.add_argument("--test-set", default=str(TEST_SET))
    ap.add_argument("--gt-dir", default=str(GT_DIR))
    ap.add_argument("--images", default=str(IMAGES))
    ap.add_argument("--render-jar", default=str(dev_scoring.RENDER_JAR))
    ap.add_argument("--java-home", default=dev_scoring.JAVA_HOME)
    ap.add_argument("--python", default="/usr/bin/python3",
                    help="the Python the runners run under (scorer v2 was measured under 3.9.6)")
    args = ap.parse_args(argv)
    root = args.root or str(Path(args.runs_dir) / "_root")
    scorer = dev_scoring.check_pins(str(Path(args.render_jar).resolve()), dev_scoring.EXTRACTOR_JAR,
                                    args.java_home, args.python)
    results = []
    for run in args.runs:
        r = score_run(run, args.runs_dir, root, args.test_set, args.gt_dir, args.images,
                      args.render_jar, args.java_home, args.python, scorer)
        held = {v: k for v, k in r["extractor_crashes"].items() if k}
        print(f"{run}: {r['status']} -> {r['scores']} ({sum(r['seconds'].values()):.0f} s)"
              + (f"; held back from the F1 runners (extractor crash): {held}" if held else ""), flush=True)
        results.append(r)
    return results


if __name__ == "__main__":
    main()
