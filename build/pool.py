#!/usr/bin/env python3
"""Build the training pool: elements D1 to D6 of decisions/training-pool-rules/record.md.

Steps, in this order (one stage name each; `--until` stops after a stage):

  inclusion    the inclusion filter of the test set (build_test_set.load_candidates)
  degeneracy   D2: a degeneracy label for every included diagram. Page count from the 96 DPI
               deposit; size predicted from the deposit image where that is safe; the diagrams
               of the uncertain groups rendered with build/render.py and labelled on padded_size
  d1           D1: the test set's stages in the code's order (degeneracy, normalized-code
               deduplication per type, type agreement with the extractor), the check that
               these stages rebuild the test set, then the test set's repositories removed
  d3           D3: sources with exactly one diagram block that the scorer reads as their
               labelled type (block isolated by csr_runner.extract_puml, network closed)
  d4           D4: targets of at most 5,376 tokens (Qwen/Qwen3.5-2B, slow tokenizer class)
  d6           D5, D6: near-duplicates of the 1,000 test diagrams removed (build/leakage.py)

The rules of the test set are run from plantuml-recognition-evaluation/util/build_test_set.py, loaded by
path; the scorer's block isolation from plantuml-recognition-evaluation/evaluation/csr_runner.py.
Both files and the two extractor JARs are pinned by sha256 (PINS); a file that differs stops
the run.

No JVM started here reaches the network (OFFLINE). With the network closed the extractor gives
no type for a source whose block reads a remote include. In the type check of D1 such a source
is counted as agreeing (its type is undetermined), and D3 then removes it.

The numbers of Implementation note 2 of the record are checked on the way (EXPECTED); a
difference stops the run. A count may exceed the record's value by at most the number of
sources of undetermined type, and the checks allow exactly that. `--d1-network open` runs the
type check of D1 as build_test_set.py ran it (remote includes are fetched) and checks the
numbers exactly.

Caches go to --work, artifacts to --out. Nothing is written into another repository.

Artifacts (--out): pool_after_d6.json (the pool: key, type, tier, repository), candidates.tsv
(every candidate of D1 with its outcome at D3, D4 and D6), funnel.json (counts by type and by
cell after every step, the counts of D6 and D9), d2_degeneracy_labels.tsv and d2_render/ (the
labels of D2 and the manifest of the renders behind them), d3_excluded.jsonl,
d3_extractor_v1_v2_differences.jsonl, d6_removed.jsonl, d6_maxima.tsv (per candidate the
largest value of each measure and the position of the test diagram in test_set.json),
config.json, SHA256SUMS.

Usage (from fine-tune/):
  python3 build/pool.py --work <scratch folder> --out data/pool
"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import pickle
import re
import shutil
import struct
import subprocess
import sys
import time
import types
from collections import Counter, defaultdict
from multiprocessing import get_context
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
UNI = HERE.parents[1]
METADATA = UNI / "herald/zenodo_content/uml_metadata_enriched.json"
PUML_DIR = UNI / "dataset/puml_files"
IMG96_DIR = UNI / "herald/zenodo_content/puml_images"
TEST_SET = UNI / "plantuml-recognition-evaluation/data/test_set.json"
TEST_PUML_DIR = UNI / "plantuml-recognition-evaluation/data/puml_files"
GT_TOKENS = UNI / "plantuml-recognition-evaluation/data/gt_token_stats.json"
BUILD_TEST_SET = UNI / "plantuml-recognition-evaluation/util/build_test_set.py"
CSR_RUNNER = UNI / "plantuml-recognition-evaluation/evaluation/csr_runner.py"
EXTRACTOR_V1 = UNI / "plantuml-recognition-evaluation/plantuml-stats-extractor-v1.jar"
EXTRACTOR_V2 = UNI / "plantuml-recognition-evaluation/plantuml-stats-extractor-v2.jar"
EXTRACTOR_CLASS = "net.sourceforge.plantuml.stats.DiagramStatsExtractor"
RENDER_PY = HERE / "render.py"
DEFAULT_JAVA_HOME = "/Library/Java/JavaVirtualMachines/temurin-26.jdk/Contents/Home"

PINS = {
    "build_test_set": "4be04fadf02b87f94a54591612b58a31f24adbb7ad0986b1e27abf635ec54d53",
    "extract_puml_source": "a9053594872aa83ad11a39d69630ee8d6bd623dfbde2be94ffc19ea795bed62a",
    "extractor_v1": "f6e76d4973ce12c23cb15dff6420726f7d2203a80c1761d1454b5a2c5a91777b",
    "extractor_v2": "7eb9436b790239e70a2ef40c72227954500b3721059812e2d50640d4acb88f1a",
}

TYPES = ("class", "sequence")
CELLS = [(t, tier) for t in TYPES for tier in (1, 2, 3, 4)]

# D2
SCALE_96_TO_300 = 3.125      # 300 / 96
MIN_SIDE = 512               # stage 2 pads a side below it
DEPOSIT_LIMIT = 4096         # PlantUML's default size limit, which the 96 DPI deposit kept
ASPECT_BAND = (7.8, 8.2)     # predicted padded aspect too near the limit of 8 to be trusted
_OWN_DPI = re.compile(r"\bdpi\b", re.I)
EMBEDDED = "{{"

# D4
TOKENIZER = "Qwen/Qwen3.5-2B"
TOKEN_CAP = 5376
HEADER_LINES = 3             # the World of Code header of a corpus file

# JVM properties that send every HTTP(S) and FTP request to a proxy nobody listens on
# (decisions/scorer-v2-freeze/scripts/corpus_extract.py). The JDK does not fall back to a
# direct connection (test_pool.py).
OFFLINE = ["-Dhttp.proxyHost=127.0.0.1", "-Dhttp.proxyPort=1", "-Dhttps.proxyHost=127.0.0.1",
           "-Dhttps.proxyPort=1", "-Dhttp.nonProxyHosts=", "-Dftp.proxyHost=127.0.0.1",
           "-Dftp.proxyPort=1"]
# A preprocessor line that reads from a URL (decisions/training-pool-rules/scripts/gate/
# render_handoff_counts.py).
REMOTE = re.compile(r"^[ \t]*!(includeurl\b|include(?:_many|_once|sub)?\s+<?https?://"
                    r"|theme\s+\S+\s+from\s+https?://)", re.M | re.I)
DOUBLE_AT = re.compile(r"^[ \t﻿]*@@startuml", re.M | re.I)
_LATER_DIAGRAM = re.compile(r"\.[A-Za-z]+_\d+$")   # the extractor's record of a further block

# Implementation note 2 of the record: numbers the build reproduces before it goes on.
EXPECTED = {
    "inclusion": {"class": 71323, "sequence": 35025},
    "degeneracy": {"multipage": 309, "clipped": 598, "aspect": 326},
    "rebuild_cells_equal": 7,
    "rebuild_class_T4": (16763, 16766),      # rebuilt, test_set.json
    "d1": {"class": 43233, "sequence": 24285},
}


class PoolError(RuntimeError):
    pass


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Code of the test set and of the scorer, loaded by path
# --------------------------------------------------------------------------- #
_loaded = {}


def _load_source(path, name):
    mod = types.ModuleType(name)
    mod.__file__ = str(path)
    exec(compile(Path(path).read_text(encoding="utf-8"), str(path), "exec"), mod.__dict__)
    return mod


def test_set_code(path=BUILD_TEST_SET):
    """The module build_test_set.py, the code the test set was built with."""
    key = ("build_test_set", str(path))
    if key not in _loaded:
        got = sha256_file(path)
        if got != PINS["build_test_set"]:
            raise PoolError(f"{path} has sha256 {got}, pinned {PINS['build_test_set']}")
        _loaded[key] = _load_source(path, "build_test_set")
    return _loaded[key]


def scorer_extract_puml(path=CSR_RUNNER):
    """csr_runner.extract_puml, the scorer's block isolation. Pinned by the text of the function,
    so that other edits of the scorer's file do not stop the build."""
    key = ("csr_runner", str(path))
    if key not in _loaded:
        fn = _load_source(path, "csr_runner").extract_puml
        got = hashlib.sha256(_function_source(path, "extract_puml").encode("utf-8")).hexdigest()
        if got != PINS["extract_puml_source"]:
            raise PoolError(f"extract_puml of {path} has source sha256 {got}, "
                            f"pinned {PINS['extract_puml_source']}")
        _loaded[key] = fn
    return _loaded[key]


def _function_source(path, name):
    """The text of a top-level function of a file: from its `def` line to the next top-level line."""
    lines = Path(path).read_text(encoding="utf-8").split("\n")
    start = next(i for i, l in enumerate(lines) if l.startswith(f"def {name}("))
    end = next((i for i in range(start + 1, len(lines)) if lines[i] and not lines[i][0].isspace()),
               len(lines))
    return "\n".join(lines[start:end]).rstrip() + "\n"


def check_jar(jar, pin):
    got = sha256_file(jar)
    if got != PINS[pin]:
        raise PoolError(f"{jar} has sha256 {got}, pinned {PINS[pin]}")


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
def load_records(metadata=METADATA):
    with open(metadata) as f:
        return json.load(f)["classifications"]


def load_test_set(path=TEST_SET):
    with open(path) as f:
        return json.load(f)


def tier_thresholds(test):
    """D10: the tier boundaries of test_set.json, {type: (q1, q2, q3)}."""
    q = test["quartile_thresholds"]
    return {t: (q[t]["q1"], q[t]["q2"], q[t]["q3"]) for t in TYPES}


def cell_counts(keys, records, thr):
    """{"class_T1": n, ...} for a list of keys, tiers by `thr`."""
    assign = test_set_code().assign_tier
    c = Counter()
    for k in keys:
        r = records[k]
        c[(r["primary_type"], assign(r["content_lines"], thr[r["primary_type"]]))] += 1
    return {f"{t}_T{tier}": c[(t, tier)] for t, tier in CELLS}


# --------------------------------------------------------------------------- #
# Inclusion
# --------------------------------------------------------------------------- #
def inclusion(records):
    """The inclusion filter of build_test_set.load_candidates on records already read: the same
    conditions in the same order. Returns ({type: {key: record}} in corpus order, the number
    dropped by each condition)."""
    elements_max = test_set_code().ELEMENTS_MAX
    out = {t: {} for t in TYPES}
    drops = {t: Counter() for t in TYPES}
    for key, rec in records.items():
        t = rec.get("primary_type")
        if t not in TYPES:
            continue
        if rec.get("secondary_types") != []:
            drops[t]["secondary_types"] += 1
            continue
        if rec.get("extraction_error") is not None:
            drops[t]["extraction_error"] += 1
            continue
        if rec.get("truncated") is True:
            drops[t]["truncated"] += 1
            continue
        if rec.get("elements_total", 0) > elements_max:
            drops[t]["elements_total_above_50"] += 1
            continue
        if not rec.get("repository"):
            drops[t]["no_repository"] += 1
            continue
        out[t][key] = rec
    return out, {t: dict(drops[t]) for t in TYPES}


# --------------------------------------------------------------------------- #
# D2: degeneracy labels
# --------------------------------------------------------------------------- #
def deposit_pages(img_dir, all_bases):
    """{base: [(w, h), ...]} for a folder of PNG pages: build_test_set.map_pages_to_bases, which
    reads a zip, applied to a folder. A stem equal to a base is that diagram's image; otherwise
    one trailing _NN is a page of the base before it. Returns (pages, orphan count)."""
    trailing = test_set_code()._TRAILING_PAGE
    pages, orphans = defaultdict(list), 0
    for name in sorted(os.listdir(img_dir)):
        if not name.endswith(".png"):
            continue
        stem = name[:-4]
        if stem in all_bases:
            owner = stem
        else:
            stripped = trailing.sub("", stem, count=1)
            if stripped != stem and stripped in all_bases:
                owner = stripped
            else:
                orphans += 1
                continue
        with open(os.path.join(img_dir, name), "rb") as fh:
            head = fh.read(26)
        if head[:8] != b"\x89PNG\r\n\x1a\n":
            continue
        pages[owner].append(struct.unpack(">II", head[16:24]))
    return dict(pages), orphans


def _norm_one(args):
    key, puml_dir = args
    norm = test_set_code().normalized_code(os.path.join(puml_dir, key))
    return key, hashlib.sha1(norm.encode("utf-8")).hexdigest(), norm


def normalized(keys, puml_dir=PUML_DIR, workers=8):
    """{key: (sha1 of the normalized code, normalized code)}; the code and the hash of
    build_test_set.build, stage 3."""
    with get_context("fork").Pool(workers) as pool:
        return {k: (h, n) for k, h, n in pool.imap(_norm_one, [(k, str(puml_dir)) for k in keys],
                                                   chunksize=500)}


def predicted_padded_size(w96, h96):
    """Size of the padded 300 DPI image predicted from the 96 DPI deposit image."""
    return (max(round(SCALE_96_TO_300 * w96), MIN_SIDE), max(round(SCALE_96_TO_300 * h96), MIN_SIDE))


def size_source(pages, norm):
    """Where the size of a diagram comes from: ("pages", []) when the deposit holds no page or
    more than one (the page count decides); ("render", reasons) for the uncertain groups;
    ("predicted", []) otherwise."""
    if len(pages) != 1:
        return "pages", []
    w, h = pages[0]
    pw, ph = predicted_padded_size(w, h)
    aspect = max(pw, ph) / min(pw, ph)
    reasons = []
    if _OWN_DPI.search(norm):
        reasons.append("own_dpi")
    if max(w, h) >= DEPOSIT_LIMIT:
        reasons.append("deposit_limit")
    if EMBEDDED in norm:
        reasons.append("embedded_diagram")
    if ASPECT_BAND[0] <= aspect <= ASPECT_BAND[1]:
        reasons.append("aspect_band")
    return ("render", reasons) if reasons else ("predicted", [])


def label_from_render(row):
    """The degeneracy label of a diagram rendered by build/render.py, from its manifest line."""
    label = test_set_code().degeneracy_label
    status = row["status"]
    if status == "ok":
        return label([tuple(row["padded_size"])])
    if status == "multi_page":
        return "multipage"
    if status == "no_image":
        return "no_image"
    raise PoolError(f"{row['key']}: render status {status!r} gives no size; the Decision does not "
                    "say how such a diagram is labelled")


def degeneracy_labels(keys, pages, norms, rendered):
    """{key: (label or None, source)} for the given keys. `rendered` is {key: manifest line}."""
    label = test_set_code().degeneracy_label
    out = {}
    for k in keys:
        p = pages.get(k[:-5], [])
        source, _ = size_source(p, norms[k][1])
        if source == "pages":
            out[k] = (label(p), source)
        elif source == "render":
            if k not in rendered:
                raise PoolError(f"{k} belongs to an uncertain group and has no render")
            out[k] = (label_from_render(rendered[k]), source)
        else:
            out[k] = (label([predicted_padded_size(*p[0])]), source)
    return out


def render_cwd(log_path):
    """Working directory of a render.py call: the folder `render_cwd` beside the render log.
    Files that PlantUML writes into the working directory of its JVM (svek.dot and svek.svg were
    seen; decisions/training-pool-rules/implement.md, section 20.1) go where the caller puts the
    log, not into the folder the build is started from."""
    cwd = Path(log_path).resolve().parent / "render_cwd"
    cwd.mkdir(parents=True, exist_ok=True)
    return cwd


def run_render(keys, out_dir, src_dir=PUML_DIR, log_path=None):
    """`python3 build/render.py --src-dir <sources> --keys <key list> --out <folder>` with its
    defaults, and the manifest lines of the keys. render.py is not called when the manifest
    of the folder already holds every key: a key is rendered once (Implementation note 7)."""
    out_dir = Path(out_dir).resolve()
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    key_file = out_dir.parent / (out_dir.name + "_keys.txt")
    key_file.write_text("".join(k + "\n" for k in keys))
    manifest = out_dir / "manifest.jsonl"
    done = {r["key"] for r in read_jsonl(manifest)} if manifest.is_file() else set()
    if not (set(keys) <= done and (out_dir / "summary.json").is_file()):
        cmd = [sys.executable, str(RENDER_PY), "--src-dir", str(Path(src_dir).resolve()), "--keys", str(key_file),
               "--out", str(out_dir)]
        log("render:", " ".join(cmd))
        log_path = log_path or (str(out_dir) + ".log")
        cwd = render_cwd(log_path)
        with open(log_path, "a") as fh:
            code = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=str(cwd)).returncode
        if code != 0:
            raise PoolError(f"render.py ended with exit code {code}; see {log_path}")
    rows = {r["key"]: r for r in read_jsonl(manifest)}
    return [rows[k] for k in keys]


# --------------------------------------------------------------------------- #
# D1: deduplication, type agreement, the test set's repositories
# --------------------------------------------------------------------------- #
def dedup(keys, sha1):
    """First occurrence of every normalized-code hash, in the order of `keys` (build_test_set.build,
    stage 3). Returns (kept keys, number dropped)."""
    seen, kept = set(), []
    for k in keys:
        h = sha1[k]
        if h in seen:
            continue
        seen.add(h)
        kept.append(k)
    return kept, len(keys) - len(kept)


def degeneracy_and_dedup(cands, labels, sha1):
    """Stages 2 and 3 of build_test_set.build: the diagrams without a degeneracy label, then,
    per type, the first occurrence of every normalized code among them, in corpus order.
    `cands` is {type: [keys in corpus order]}, `labels` {key: label or None}."""
    survivors = {t: [k for k in cands[t] if labels[k] is None] for t in TYPES}
    deduped, dups = {}, {}
    for t in TYPES:
        deduped[t], dups[t] = dedup(survivors[t], sha1)
    return survivors, deduped, dups


def type_agreement(keys, label, extracted, remote=()):
    """build_test_set.build, stage 3b, for the keys of one type: a key is kept when the
    extractor's diagram_type equals its label; a key without a type never agrees. `remote`
    holds the keys whose type was not computed because the network is closed and the source
    has a remote include: they are kept here (undetermined) and are for D3 to decide.
    Returns (kept keys, undetermined keys, Counter of the disagreeing types)."""
    kept, undetermined, disagree = [], [], Counter()
    for k in keys:
        pt = (extracted.get(k) or {}).get("diagram_type") or ""
        if pt == label:
            kept.append(k)
        elif not pt and k in remote:
            kept.append(k)
            undetermined.append(k)
        else:
            disagree[pt or "<unparsed>"] += 1
    return kept, undetermined, disagree


def candidate_pool(cands, labels, sha1, extracted, remote, repo_of, test_repos):
    """D1: the stages of build_test_set.build in the code's order (degeneracy labels;
    normalized-code deduplication per type, first occurrence in corpus order; type agreement),
    then every diagram of a test repository removed. Returns the keys after each stage."""
    survivors, deduped, dups = degeneracy_and_dedup(cands, labels, sha1)
    out = {"survivors": survivors, "deduped": deduped, "duplicates": dups,
           "agreed": {}, "undetermined": {}, "disagree": {}, "d1": {}}
    for t in TYPES:
        out["agreed"][t], out["undetermined"][t], d = type_agreement(deduped[t], t, extracted, remote)
        out["disagree"][t] = dict(d)
        out["d1"][t] = [k for k in out["agreed"][t] if repo_of[k] not in test_repos]
    return out


def java_bin(java_home=None):
    home = java_home or os.environ.get("JAVA_HOME") or DEFAULT_JAVA_HOME
    java = os.path.join(home, "bin", "java")
    if not os.path.isfile(java):
        raise PoolError(f"java not found at {java}; set JAVA_HOME")
    return java


def _parse_records(text):
    out = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line, strict=False))   # as build_test_set.parser_diagram_types
        except ValueError:
            pass
    return out


def _jvm(java, jar, args, cwd, timeout, offline):
    cmd = [java, "-Xmx4g", *(OFFLINE if offline else []), "-cp", str(jar), EXTRACTOR_CLASS, *args]
    try:
        p = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=timeout)
        return p.stdout.decode("utf-8", "replace"), ("ok" if p.returncode == 0 else f"exit:{p.returncode}")
    except subprocess.TimeoutExpired as e:
        return (e.stdout or b"").decode("utf-8", "replace"), "timeout"


def extract_dir(java, jar, directory, cwd, offline=True, batch_timeout=900, file_timeout=120):
    """{file name: {"diagram_type", "error"}} for the .puml files of a folder, one JVM over the
    folder (`--dir`, as build_test_set.parser_diagram_types). The record of a file is the one
    whose `file` equals its name; a file without a record gets error "no_record". If the JVM
    dies or runs out of time, the files without a record are run again, and the first of them
    alone (the run of decisions/scorer-v2-freeze/scripts/corpus_extract.py)."""
    files = sorted(f for f in os.listdir(directory) if f.endswith(".puml"))
    out, status = _jvm(java, jar, ["--dir", str(directory)], cwd, batch_timeout, offline)
    recs = {r["file"]: r for r in _parse_records(out) if "file" in r}
    todo = [f for f in files if f not in recs]
    while todo and status != "ok":
        out, status = _jvm(java, jar, [os.path.join(directory, f) for f in todo], cwd, batch_timeout, offline)
        recs.update({r["file"]: r for r in _parse_records(out) if "file" in r})
        rest = [f for f in todo if f not in recs]
        if status != "ok" and rest:
            culprit = rest[0]
            out1, status1 = _jvm(java, jar, [os.path.join(directory, culprit)], cwd, file_timeout, offline)
            got = {r["file"]: r for r in _parse_records(out1) if "file" in r}
            if status1 == "ok" and culprit in got:
                recs.update(got)
            else:
                recs[culprit] = {"file": culprit, "diagram_type": None,
                                 "error": "timeout" if status1 == "timeout" else "crash:" + status1}
            rest = rest[1:]
        todo = rest
    return {f: {"diagram_type": recs[f].get("diagram_type"), "error": recs[f].get("error")} if f in recs
            else {"diagram_type": None, "error": "no_record"} for f in files}


def extract_many(dirs, jar, cwd, offline=True, jobs=8, java_home=None):
    """extract_dir over several folders, `jobs` JVMs at a time. Returns one merged dict."""
    java = java_bin(java_home)
    os.makedirs(cwd, exist_ok=True)
    merged = {}
    with cf.ThreadPoolExecutor(max_workers=jobs) as ex:
        for res in ex.map(lambda d: extract_dir(java, jar, d, cwd, offline), dirs):
            merged.update(res)
    return merged


def link_batches(keys, src_dir, root, batch=500):
    """Folders of symlinks to the sources, `batch` per folder (the temporary folder of
    build_test_set.parser_diagram_types, in batches)."""
    shutil.rmtree(root, ignore_errors=True)
    dirs = []
    for i in range(0, len(keys), batch):
        d = os.path.join(root, f"b{i // batch:04d}")
        os.makedirs(d)
        for k in keys[i:i + batch]:
            os.symlink(os.path.abspath(os.path.join(src_dir, k)), os.path.join(d, k))
        dirs.append(d)
    return dirs


def has_remote_include(key, src_dir=PUML_DIR):
    with open(os.path.join(src_dir, key), "rb") as fh:
        return bool(REMOTE.search(fh.read().decode("utf-8", "replace")))


def rebuild_test_set(agreed, records, test):
    """Stages 4 and 5 of build_test_set.build on a candidate pool {type: [keys]}: quartile
    tiers, then the seeded sample with the repository limit. Returns (thresholds, available per
    cell, selected keys)."""
    code = test_set_code()
    thresholds, by_cell = {}, defaultdict(list)
    for t in TYPES:
        cl = np.array([records[k]["content_lines"] for k in agreed[t]])
        thresholds[t] = code.quartile_thresholds(cl)
        for k in agreed[t]:
            by_cell[(t, code.assign_tier(records[k]["content_lines"], thresholds[t]))].append((k, records[k]))
    selected, stats = code.sample_cells(by_cell, test["config"]["n_per_cell"], test["config"]["seed"])
    return thresholds, {f"{t}_T{tier}": stats[(t, tier)]["available"] for t, tier in CELLS}, [k for k, _ in selected]


# --------------------------------------------------------------------------- #
# D3: one diagram block, read by the scorer as its labelled type
# --------------------------------------------------------------------------- #
# PlantUML 1.2025.9's rule for where a diagram block starts and ends, written again from its
# source (utils/StartUtils.java beforeStartUml and startsWithSymbolAnd, core/DiagramType.java
# getTypeFromArobaseStart, preproc/UncommentReadLine.java, BlockUmlBuilder.java). The code is
# that of decisions/render-pipeline/scripts/verify_r2_corpus_block_starts.py.
_JAVA_WS = set(" \t\n\x0b\x0c\r\x1c\x1d\x1e\x1f        "
               "      　")
_WORD = re.compile(r"[A-Za-z0-9_~]")
_BLOCK_TYPES = ("bpm", "board", "creole", "chronology", "chen", "dot", "ditaa", "def", "ebnf", "flow",
                "files", "gantt", "git", "hcl", "jcckit", "json", "latex", "math", "mindmap", "nwdiag",
                "project", "regex", "salt", "uml", "wire", "wbs", "yaml")


def _symbol_and(keyword, text):
    for i, c in enumerate(text):
        if c in _JAVA_WS:
            continue
        if c not in "@\\":
            return False
        return text[i + 1:i + 1 + len(keyword)] == keyword
    return False


def _before_start(s):
    inside = False
    for i, ch in enumerate(s):
        if _symbol_and("start", s[i:]):
            return s[:i]
        if inside:
            if ch == ">":
                inside = False
            continue
        if ch == "<":
            inside = True
        elif _WORD.match(ch):
            return None
    return None


def _is_start(s):
    for i, c in enumerate(s):
        if c in _JAVA_WS:
            continue
        if c not in "@\\":
            return False
        rest = s[i + 1:].lower()
        return rest.startswith("start") and rest[5:].startswith(_BLOCK_TYPES)
    return False


def block_starts(text):
    """The first lines of the blocks PlantUML builds from a source, in order, each as
    (prefix cut from the line, line)."""
    header, current, found = None, None, []
    for raw in re.split(r"\r\n|\r|\n", text):
        tmp = _before_start(raw)
        if tmp is not None:
            header = tmp
        line = raw
        if header is not None and header.startswith(raw):
            line = ""
        elif header is not None and raw.startswith(header):
            line = raw[len(header):]
        if _is_start(line):
            current = (header or "", line)
        if _symbol_and("end", line) and current is not None:
            found.append(current)
            current = None
    return found


def read_source(path):
    """A corpus source as text, bytes kept (surrogateescape, no newline translation), as the
    scorer audit read it before isolating the block."""
    with open(path, encoding="utf-8", errors="surrogateescape", newline="") as f:
        return f.read()


def isolate_batches(keys, src_dir, root, batch=500):
    """The block of every source as csr_runner.extract_puml returns it, written under `root` in
    folders of `batch` files."""
    extract_puml = scorer_extract_puml()
    shutil.rmtree(root, ignore_errors=True)
    dirs = []
    for i in range(0, len(keys), batch):
        d = os.path.join(root, f"b{i // batch:04d}")
        os.makedirs(d)
        for k in keys[i:i + batch]:
            block = extract_puml(read_source(os.path.join(src_dir, k)))
            with open(os.path.join(d, k), "w", encoding="utf-8", errors="surrogateescape", newline="") as f:
                f.write(block)
        dirs.append(d)
    return dirs


def source_facts(key, src_dir=PUML_DIR):
    """What D3 reads from the source itself: the number of blocks under PlantUML's rule, and
    the properties that name the kind of an excluded source."""
    with open(os.path.join(src_dir, key), "rb") as fh:
        text = fh.read().decode("utf-8", "replace")
    starts = block_starts(text.lstrip("﻿"))
    return {"blocks": len(starts),
            "double_at": bool(DOUBLE_AT.search(text)),
            "prefixed_start": any(prefix.strip("".join(_JAVA_WS) + "\ufeff") for prefix, _ in starts),
            "remote_include": bool(REMOTE.search(text))}


def one_block_verdict(label, facts, record):
    """D3 for one candidate. Returns (kept, kind): kind is None for a kept candidate, else the
    kind of source excluded."""
    if facts["blocks"] != 1:
        return False, "several_blocks" if facts["blocks"] > 1 else "no_block"
    error, dtype = record.get("error"), record.get("diagram_type")
    if not error and dtype == label:
        return True, None
    if facts["double_at"]:
        return False, "double_at_startuml"
    if facts["prefixed_start"]:
        return False, "diagram_inside_a_comment"
    if facts["remote_include"]:
        return False, "remote_include"
    return False, "other_error" if error else "type_differs_on_the_block"


# --------------------------------------------------------------------------- #
# D4: token cap
# --------------------------------------------------------------------------- #
def target_text(text):
    """The text whose tokens D4 counts: what csr_runner.extract_puml returns for the file; when
    the file has no @startuml ... @enduml pair, the same after the three header lines are
    removed (Implementation note 4)."""
    extract_puml = scorer_extract_puml()
    s, e = text.find("@startuml"), text.rfind("@enduml")
    if s != -1 and e != -1 and e > s:
        return extract_puml(text)
    return extract_puml("".join(text.splitlines(keepends=True)[HEADER_LINES:]))


def read_for_tokens(path):
    """Text mode as plantuml-recognition-evaluation/util/measure_gt_tokens.py reads a test file (UTF-8,
    newlines translated); a byte that is not UTF-8 becomes U+FFFD."""
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


_tok = None


def tokenizer():
    """Qwen/Qwen3.5-2B, slow class (use_fast=False): the class that gives the counts of
    gt_token_stats.json on 1,000 of the 1,000 test files."""
    global _tok
    if _tok is None:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")           # the tokenizer files are on disk
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        import warnings
        warnings.filterwarnings("ignore")
        from transformers import AutoTokenizer
        _tok = AutoTokenizer.from_pretrained(TOKENIZER, use_fast=False)
    return _tok


def count_tokens(text):
    return len(tokenizer().encode(text, add_special_tokens=False))


def within_token_cap(n_tokens):
    """D4: a candidate whose target is longer than 5,376 tokens is dropped."""
    return n_tokens <= TOKEN_CAP


def _count_whole_file(path):
    with open(path, encoding="utf-8") as f:      # measure_gt_tokens.py: whole file, strict UTF-8
        return count_tokens(f.read())


def _count_target(path):
    return count_tokens(target_text(read_for_tokens(path)))


def tokenizer_self_check(workers=8, gt_tokens=GT_TOKENS, test_dir=TEST_PUML_DIR):
    """Number of test files whose token count differs from gt_token_stats.json (whole file, as
    recorded there)."""
    gt = json.load(open(gt_tokens))
    with get_context("fork").Pool(workers) as pool:
        got = pool.map(_count_whole_file, [os.path.join(test_dir, r["key"]) for r in gt["per_file"]], chunksize=20)
    return sum(1 for r, n in zip(gt["per_file"], got) if n != r["n_tokens"]), len(got)


def target_token_counts(keys, src_dir=PUML_DIR, workers=8):
    with get_context("fork").Pool(workers) as pool:
        got = pool.map(_count_target, [os.path.join(src_dir, k) for k in keys], chunksize=100)
    return dict(zip(keys, got))


# --------------------------------------------------------------------------- #
# Small file helpers
# --------------------------------------------------------------------------- #
def read_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
        f.write("\n")


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def cached(path, make):
    """The object stored at `path`, made and stored on the first call."""
    path = Path(path)
    if path.is_file():
        with open(path, "rb") as f:
            return pickle.load(f)
    obj = make()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=4)
    os.replace(tmp, path)
    return obj


def expect(name, got, want, slack=0):
    """A number of Implementation note 2. `slack` is the number of sources whose type is not
    computed with the network closed (a remote include): they are counted as agreeing, so a
    count may exceed the record's by at most that number."""
    if not (want <= got <= want + slack if slack else got == want):
        raise PoolError(f"check {name!r}: got {got}, the record expects {want}"
                        + (f" (at most {slack} more)" if slack else "") + " (Implementation note 2)")
    log(f"check {name}: {got}" + (" as expected" if got == want else
                                  f", the record has {want}; within the {slack} sources of undetermined type"))


def one_block_check(keys, label_of, src_dir, work, jar=EXTRACTOR_V1, jobs=8):
    """D3 for the keys given: {key: (kept, kind)}, the source facts and the extractor's records
    of the isolated blocks."""
    work = Path(work)
    dirs = isolate_batches(keys, src_dir, work / "iso")
    records = extract_many(dirs, jar, work / "empty_cwd", offline=True, jobs=jobs)
    facts = {k: source_facts(k, src_dir) for k in keys}
    return {k: one_block_verdict(label_of[k], facts[k], records[k]) for k in keys}, facts, records


# --------------------------------------------------------------------------- #
# The build
# --------------------------------------------------------------------------- #
STAGES = ("inclusion", "degeneracy", "d1", "d3", "d4", "d6")
SKETCHIT = "PlantUML diagram generated by SketchIt!"


def build(work, until="d6", workers=12, jobs=8, d1_network="closed", checks=True):
    """Run the stages up to `until`. Returns the state reached (a dict)."""
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    stop = STAGES.index(until)
    test_set_code()
    check_jar(EXTRACTOR_V1, "extractor_v1")
    check_jar(EXTRACTOR_V2, "extractor_v2")
    scorer_extract_puml()
    t_start = time.time()
    S = {"timing": {}, "funnel": [], "d1_network": d1_network}

    def step(name, by_type, **extra):
        keys = [k for t in TYPES for k in by_type[t]]
        entry = {"step": name, "class": len(by_type["class"]), "sequence": len(by_type["sequence"]),
                 "total": len(keys), "cells": cell_counts(keys, records, thr), **extra}
        S["funnel"].append(entry)
        log(f"[{name}] class {entry['class']:,}  sequence {entry['sequence']:,}  total {entry['total']:,}")

    # ---- inclusion -------------------------------------------------------- #
    t0 = time.time()
    records = load_records()
    test = load_test_set()
    thr = tier_thresholds(test)
    test_keys = [d["key"] for d in test["diagrams"]]
    test_repos = {d["repository"] for d in test["diagrams"]}
    cands, drops = inclusion(records)
    cand_keys = {t: list(cands[t]) for t in TYPES}
    S.update(records=records, test=test, thr=thr, test_keys=test_keys, test_repos=test_repos)
    step("inclusion", cand_keys, dropped=drops, metadata_records=len(records),
         labelled={t: sum(1 for r in records.values() if r.get("primary_type") == t) for t in TYPES})
    if checks:
        expect("inclusion", {t: len(cands[t]) for t in TYPES}, EXPECTED["inclusion"])
    S["timing"]["inclusion"] = time.time() - t0
    if stop < 1:
        return S

    # ---- degeneracy (D2) -------------------------------------------------- #
    t0 = time.time()
    incl = [k for t in TYPES for k in cands[t]]
    label_of = {k: records[k]["primary_type"] for k in incl}
    repo_of = {k: records[k]["repository"] for k in incl}
    all_bases = {k[:-5] for k in records}
    pages, orphans = cached(work / "pages96.pkl", lambda: deposit_pages(IMG96_DIR, all_bases))
    norms = cached(work / "norm.pkl", lambda: normalized(incl, PUML_DIR, workers))
    sources = {k: size_source(pages.get(k[:-5], []), norms[k][1]) for k in incl}
    to_render = [k for k in incl if sources[k][0] == "render"]
    reasons = Counter(r for k in to_render for r in sources[k][1])
    log(f"D2: deposit pages for {len(pages):,} diagrams ({orphans} orphan pages); size from the page "
        f"count {sum(1 for k in incl if sources[k][0] == 'pages'):,}, predicted "
        f"{sum(1 for k in incl if sources[k][0] == 'predicted'):,}, to render {len(to_render):,} {dict(reasons)}")
    t_render = time.time()
    manifest = run_render(to_render, work / "d2_render")
    S["timing"]["d2_render_call"] = time.time() - t_render
    rendered = {r["key"]: r for r in manifest}
    labelled = degeneracy_labels(incl, pages, norms, rendered)
    labels = {k: v[0] for k, v in labelled.items()}
    by_label = Counter(labels.values())
    survivors = {t: [k for k in cand_keys[t] if labels[k] is None] for t in TYPES}
    S.update(norms=norms, labelled=labelled, size_sources=sources, to_render=to_render, rendered=rendered,
             label_of=label_of, repo_of=repo_of)
    step("degeneracy_D2", survivors,
         labels={t: dict(Counter(str(labels[k]) for k in cand_keys[t])) for t in TYPES},
         labels_total={str(l): n for l, n in by_label.items()},
         label_source=dict(Counter(s for _, s in labelled.values())),
         render_keys=len(to_render), render_reasons=dict(reasons),
         render_status=dict(Counter(rendered[k]["status"] for k in to_render)),
         render_labels=dict(Counter(str(labels[k]) for k in to_render)))
    if checks:
        expect("degeneracy", {l: by_label[l] for l in EXPECTED["degeneracy"]}, EXPECTED["degeneracy"])
    S["timing"]["degeneracy"] = time.time() - t0
    if stop < 2:
        return S

    # ---- D1 --------------------------------------------------------------- #
    t0 = time.time()
    sha1 = {k: norms[k][0] for k in incl}
    _, deduped, dups = degeneracy_and_dedup(cand_keys, labels, sha1)
    step("deduplication", deduped, dropped=dups)
    dkeys = [k for t in TYPES for k in deduped[t]]
    offline = d1_network == "closed"

    def raw_types():
        dirs = link_batches(dkeys, PUML_DIR, work / "d1_links")
        res = extract_many(dirs, EXTRACTOR_V1, work / "empty_cwd", offline=offline, jobs=jobs)
        shutil.rmtree(work / "d1_links", ignore_errors=True)
        return res
    raw = cached(work / f"d1_types_v1_network_{d1_network}.pkl", raw_types)
    if any(k not in raw for k in dkeys):
        raise PoolError(f"the extractor cache lacks keys; delete {work}/d1_types_v1_network_{d1_network}.pkl")
    remote = {k for k in dkeys if not (raw[k]["diagram_type"] or "") and has_remote_include(k)} if offline else set()
    cp = candidate_pool(cand_keys, labels, sha1, raw, remote, repo_of, test_repos)
    step("type_agreement", cp["agreed"], dropped=cp["disagree"], d1_network=d1_network,
         type_undetermined_remote_include={t: len(cp["undetermined"][t]) for t in TYPES},
         raw_errors=dict(Counter(str(raw[k]["error"]) for k in dkeys if raw[k]["error"])))
    thresholds, available, selected = rebuild_test_set(cp["agreed"], records, test)
    ref = {c: v["available"] for c, v in test["cell_stats"].items()}
    in_pool = set(cp["agreed"]["class"]) | set(cp["agreed"]["sequence"])
    und = {k for t in TYPES for k in cp["undetermined"][t]}
    und_cells = cell_counts(sorted(und), records, {t: tuple(thresholds[t]) for t in TYPES})
    equal = [c for c in ref if available[c] == ref[c]]
    near = [c for c in ref if ref[c] <= available[c] <= ref[c] + und_cells[c]]
    S["rebuild"] = {"thresholds": {t: list(thresholds[t]) for t in TYPES}, "available": available,
                    "available_test_set_json": ref, "cells_equal": len(equal),
                    "cells_equal_up_to_sources_of_undetermined_type": len(near),
                    "cells_differing": {c: available[c] - ref[c] for c in ref if available[c] != ref[c]},
                    "keys_selected": len(selected),
                    "keys_identical_to_test_set": len(set(selected) & set(test_keys)),
                    "test_keys_with_a_degeneracy_label": sum(1 for k in test_keys if labels[k] is not None),
                    "test_keys_in_the_candidate_pool": sum(1 for k in test_keys if k in in_pool),
                    "type_undetermined_remote_include_in_available": und_cells}
    log("rebuild of the test set:", json.dumps(S["rebuild"]))
    if checks:
        expect("rebuild: tier boundaries", {t: tuple(thresholds[t]) for t in TYPES}, thr)
        expect("rebuild: cells equal to cell_stats" + (" up to the sources of undetermined type" if und else ""),
               len(near), EXPECTED["rebuild_cells_equal"])
        expect("rebuild: class_T4 in test_set.json", ref["class_T4"], EXPECTED["rebuild_class_T4"][1])
        expect("rebuild: class_T4 rebuilt", available["class_T4"], EXPECTED["rebuild_class_T4"][0],
               slack=und_cells["class_T4"])
    d1 = cp["d1"]
    d1keys = [k for t in TYPES for k in d1[t]]
    test_sha = {sha1[k] for k in test_keys}
    exact = sum(1 for k in d1keys if sha1[k] in test_sha)
    S.update(sha1=sha1, d1=d1, raw_types=raw, d1_undetermined=sorted(k for k in d1keys if k in und),
             d1_exact_matches_with_test=exact)
    step("test_repositories_removed_D1", d1, test_repositories=len(test_repos),
         type_undetermined_remote_include=len(S["d1_undetermined"]),
         candidates_with_the_normalized_code_of_a_test_diagram=exact)
    if checks:
        for t in TYPES:
            expect(f"D1 {t}", len(d1[t]), EXPECTED["d1"][t], slack=sum(1 for k in d1[t] if k in und))
        expect("D1: candidates with the normalized code of a test diagram", exact, 0)
    S["timing"]["d1"] = time.time() - t0
    if stop < 3:
        return S

    # ---- D3 --------------------------------------------------------------- #
    t0 = time.time()

    def isolated(jar):
        def make():
            dirs = isolate_batches(d1keys, PUML_DIR, work / "d3_iso")
            return extract_many(dirs, jar, work / "empty_cwd", offline=True, jobs=jobs)
        return make
    iso_v1 = cached(work / "d3_types_v1.pkl", isolated(EXTRACTOR_V1))
    iso_v2 = cached(work / "d3_types_v2.pkl", isolated(EXTRACTOR_V2))
    facts = cached(work / "d3_facts.pkl", lambda: {k: source_facts(k) for k in d1keys})
    if any(k not in iso_v1 or k not in iso_v2 or k not in facts for k in d1keys):
        raise PoolError(f"a D3 cache lacks keys; delete {work}/d3_*.pkl")
    d3, d3_excluded, kinds, jar_differs = {t: [] for t in TYPES}, [], Counter(), []
    for k in d1keys:
        kept, kind = one_block_verdict(label_of[k], facts[k], iso_v1[k])
        kept2, kind2 = one_block_verdict(label_of[k], facts[k], iso_v2[k])
        if kept != kept2 or iso_v1[k] != iso_v2[k]:
            jar_differs.append({"key": k, "v1": iso_v1[k], "v2": iso_v2[k], "kept_v1": kept, "kept_v2": kept2})
        if kept:
            d3[label_of[k]].append(k)
        else:
            kinds[kind] += 1
            d3_excluded.append({"key": k, "primary_type": label_of[k], "kind": kind, **facts[k],
                                "block_diagram_type": iso_v1[k]["diagram_type"], "block_error": iso_v1[k]["error"],
                                "raw_diagram_type": raw[k]["diagram_type"], "raw_error": raw[k]["error"]})
    S.update(d3=d3, d3_excluded=d3_excluded, d3_jar_differs=jar_differs)
    step("one_block_D3", d3, excluded_by_kind=dict(kinds),
         excluded={t: sum(1 for d in d3_excluded if d["primary_type"] == t) for t in TYPES},
         candidates_on_which_v1_and_v2_verdicts_differ=sum(1 for d in jar_differs if d["kept_v1"] != d["kept_v2"]),
         candidates_on_which_v1_and_v2_type_or_error_differ=len(jar_differs))
    S["timing"]["d3"] = time.time() - t0
    if stop < 4:
        return S

    # ---- D4 --------------------------------------------------------------- #
    t0 = time.time()
    bad, n = tokenizer_self_check(workers)
    log(f"D4: test files whose token count differs from gt_token_stats.json: {bad} of {n}")
    if bad:
        raise PoolError(f"the tokenizer gives another count than gt_token_stats.json on {bad} of {n} test files")
    d3keys = [k for t in TYPES for k in d3[t]]
    tokens = cached(work / "d4_tokens.pkl", lambda: target_token_counts(d3keys, PUML_DIR, workers))
    if any(k not in tokens for k in d3keys):
        raise PoolError(f"the token cache lacks keys; delete {work}/d4_tokens.pkl")
    d4 = {t: [k for k in d3[t] if within_token_cap(tokens[k])] for t in TYPES}
    over = [k for k in d3keys if not within_token_cap(tokens[k])]
    S.update(d4=d4, tokens=tokens, d4_dropped=over, tokenizer_self_check=[bad, n])
    step("token_cap_D4", d4, dropped={t: sum(1 for k in over if label_of[k] == t) for t in TYPES},
         dropped_cells=cell_counts(over, records, thr),
         longest_kept_target_tokens=max(tokens[k] for t in TYPES for k in d4[t]),
         longest_target_tokens=max(tokens[k] for k in d3keys),
         tokenizer_self_check_files_differing=bad, tokenizer_self_check_files=n)
    S["timing"]["d4"] = time.time() - t0
    if stop < 5:
        return S

    # ---- D5, D6 ----------------------------------------------------------- #
    import leakage   # build/leakage.py
    t0 = time.time()
    d4keys = [k for t in TYPES for k in d4[t]]
    sim = leakage.all_pairs(d4keys, PUML_DIR, test_keys, TEST_PUML_DIR, work / "d6", workers=workers)
    d6 = {t: [k for k in d4[t] if k not in sim["removed"]] for t in TYPES}
    S.update(d6=d6, sim=sim)
    step("near_duplicates_D6", d6, removed={t: sum(1 for k in sim["removed"] if label_of[k] == t) for t in TYPES})
    S["timing"]["d6"] = time.time() - t0
    S["timing"]["total"] = time.time() - t_start
    return S


# --------------------------------------------------------------------------- #
# Artifacts
# --------------------------------------------------------------------------- #
def environment():
    import platform
    import transformers
    import tokenizers
    java = subprocess.run([java_bin(), "-version"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return {"python": sys.version.split()[0], "numpy": np.__version__, "transformers": transformers.__version__,
            "tokenizers": tokenizers.__version__, "tokenizer_class": type(tokenizer()).__name__,
            "java": java.stdout.decode("utf-8", "replace").splitlines(), "platform": platform.platform()}


def d6_report(S):
    """Counts of D6 by arm and by cell, the largest values among the kept candidates, and D9."""
    import leakage
    code, records, thr, sim = test_set_code(), S["records"], S["thr"], S["sim"]
    keys = sim["keys"]
    cell = lambda k: (records[k]["primary_type"], code.assign_tier(records[k]["content_lines"], thr[records[k]["primary_type"]]))
    name = lambda c: f"{c[0]}_T{c[1]}"
    empty = lambda: {name(c): 0 for c in CELLS}
    flagged = {a: empty() for a in leakage.ARMS}
    combos, removed_cells = defaultdict(empty), empty()
    for i in np.nonzero(sim["removed_mask"])[0]:
        c = name(cell(keys[i]))
        arms = "".join(a for a in leakage.ARMS if sim["fired"][a][i])
        combos[arms][c] += 1
        removed_cells[c] += 1
        for a in arms:
            flagged[a][c] += 1
    total = lambda d: dict(d, total=sum(d.values()))
    kept = ~sim["removed_mask"]
    residual = {}
    for a in leakage.ARMS:
        vals = np.where(kept, sim["best"][a], -1.0)
        i = int(vals.argmax())
        residual[a] = {"value": float(vals[i]), "candidate": keys[i],
                       "test_key": sim["test_keys"][int(sim["arg"][a][i])],
                       "cell": name(cell(keys[i]))}
    # D9: class diagrams generated by the SketchIt! plugin
    norms = S["norms"]
    sketch = lambda k: SKETCHIT in norms[k][1]
    removed = set(sim["removed"])
    d9 = {}
    for tier in (1, 2, 3, 4):
        c = ("class", tier)
        before = [k for k in keys if cell(k) == c]
        sk = [k for k in before if sketch(k)]
        sk_removed = [k for k in sk if k in removed]
        test_cell = [d["key"] for d in S["test"]["diagrams"] if d["primary_type"] == "class" and d["tier"] == tier]
        d9[name(c)] = {"pool_before_D6": len(before), "sketchit_before_D6": len(sk),
                       "sketchit_removed": len(sk_removed), "sketchit_kept": len(sk) - len(sk_removed),
                       "pool_after_D6": len(before) - sum(1 for k in before if k in removed),
                       "test_cell": len(test_cell), "sketchit_in_test_cell": sum(1 for k in test_cell if sketch(k))}
    return {"candidates_compared": len(keys), "test_diagrams": len(sim["test_keys"]),
            "pairs": len(keys) * len(sim["test_keys"]),
            "removed": total(removed_cells),
            "flagged_by_arm": {a: total(flagged[a]) for a in leakage.ARMS},
            "removed_by_combination_of_arms": {arms: total(combos[arms]) for arms in sorted(combos)},
            "largest_value_among_kept": residual,
            "test_diagrams_with_a_removed_candidate": sim["test_side"],
            "d9_sketchit_class": d9}


def write_artifacts(S, out):
    """The artifacts of step 1 under `out`, and SHA256SUMS over them."""
    import leakage
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    code, records, thr, test = test_set_code(), S["records"], S["thr"], S["test"]
    label_of, repo_of, sim = S["label_of"], S["repo_of"], S["sim"]
    tier = lambda k: code.assign_tier(records[k]["content_lines"], thr[label_of[k]])
    work = Path(S["work"])

    # D2: labels and the manifest of the renders they come from
    rows = ["key\tprimary_type\tlabel\tsource\trender_reasons"]
    for k, (label, source) in S["labelled"].items():
        if label is not None or source == "render":
            rows.append("\t".join([k, label_of[k], str(label), source, ",".join(S["size_sources"][k][1])]))
    (out / "d2_degeneracy_labels.tsv").write_text("\n".join(rows) + "\n")
    (out / "d2_render").mkdir(exist_ok=True)
    for name in ("manifest.jsonl", "render_env.json", "summary.json"):
        shutil.copyfile(work / "d2_render" / name, out / "d2_render" / name)
    shutil.copyfile(work / "d2_render_keys.txt", out / "d2_render" / "keys.txt")

    # every candidate of D1 with its outcome at D3, D4 and D6
    d3_kind = {d["key"]: d["kind"] for d in S["d3_excluded"]}
    d4_dropped, final = set(S["d4_dropped"]), {k for t in TYPES for k in S["d6"][t]}
    rows = ["key\tprimary_type\ttier\tcontent_lines\trepository\td3\ttarget_tokens\td4\td6_arms\tin_pool"]
    for t in TYPES:
        for k in S["d1"][t]:
            tokens = S["tokens"].get(k, "")
            d4 = "" if k in d3_kind else ("dropped" if k in d4_dropped else "kept")
            arms = "".join(sim["removed"][k]["arms"]) if k in sim["removed"] else ""
            rows.append("\t".join(map(str, [k, t, tier(k), records[k]["content_lines"], repo_of[k],
                                            d3_kind.get(k, "kept"), tokens, d4, arms, int(k in final)])))
    (out / "candidates.tsv").write_text("\n".join(rows) + "\n")
    write_jsonl(out / "d3_excluded.jsonl", S["d3_excluded"])
    write_jsonl(out / "d3_extractor_v1_v2_differences.jsonl", S["d3_jar_differs"])

    # D6
    write_jsonl(out / "d6_removed.jsonl",
                [{"key": k, "primary_type": label_of[k], "tier": tier(k), "repository": repo_of[k], **v}
                 for k, v in sim["removed"].items()])
    rows = ["key\tjaccard_5gram_max\tjaccard_5gram_test\tedit_similarity_max\tedit_similarity_test\t"
            "lines_held_max\tlines_held_test"]
    for i, k in enumerate(sim["keys"]):
        rows.append("\t".join([k] + [x for a in leakage.ARMS
                                     for x in ("%.6f" % sim["best"][a][i], str(int(sim["arg"][a][i])))]))
    (out / "d6_maxima.tsv").write_text("\n".join(rows) + "\n")

    # the pool
    diagrams = [{"key": k, "primary_type": t, "tier": tier(k), "repository": repo_of[k]}
                for t in TYPES for k in S["d6"][t]]
    head = {"count": len(diagrams), "cells": cell_counts([d["key"] for d in diagrams], records, thr),
            "repositories": len({d["repository"] for d in diagrams})}
    with open(out / "pool_after_d6.json", "w", encoding="utf-8") as f:      # one diagram per line
        f.write(json.dumps(head, ensure_ascii=False)[:-1] + ', "diagrams": [\n')
        f.write(",\n".join(json.dumps(d, ensure_ascii=False) for d in diagrams) + "\n]}\n")

    report = d6_report(S)
    write_json(out / "funnel.json",
               {"steps": S["funnel"], "test_set_rebuild": S["rebuild"],
                "d1_type_undetermined_remote_include": S["d1_undetermined"], "d6": report})
    write_json(out / "config.json", {
        "decision": "decisions/training-pool-rules/record.md, elements D1 to D6",
        "inputs": {"metadata": {"path": str(METADATA.relative_to(UNI)), "sha256": sha256_file(METADATA)},
                   "test_set": {"path": str(TEST_SET.relative_to(UNI)), "sha256": sha256_file(TEST_SET)},
                   "sources": str(PUML_DIR.relative_to(UNI)), "deposit_images_96dpi": str(IMG96_DIR.relative_to(UNI)),
                   "test_sources": str(TEST_PUML_DIR.relative_to(UNI)),
                   "gt_token_stats": {"path": str(GT_TOKENS.relative_to(UNI)), "sha256": sha256_file(GT_TOKENS)}},
        "pins": PINS,
        "code": {name: sha256_file(HERE / name) for name in ("pool.py", "leakage.py", "render.py")},
        "parameters": {
            "tiers_content_lines": {t: list(thr[t]) for t in TYPES},
            "D1": {"type_check_network": S["d1_network"], "extractor": "extractor_v1"},
            "D2": {"scale": SCALE_96_TO_300, "min_side": MIN_SIDE, "deposit_limit": DEPOSIT_LIMIT,
                   "aspect_band": list(ASPECT_BAND), "own_dpi_pattern": _OWN_DPI.pattern, "embedded": EMBEDDED,
                   "clip_long_edge": code.CLIP_LONG_EDGE, "aspect_max": code.ASPECT_MAX},
            "D3": {"extractor": "extractor_v1", "cross_check": "extractor_v2", "jvm_options": OFFLINE,
                   "remote_include_pattern": REMOTE.pattern},
            "D4": {"tokenizer": TOKENIZER, "use_fast": False, "add_special_tokens": False, "token_cap": TOKEN_CAP},
            "D5_D6": {"shingle": leakage.SHINGLE, "a_jaccard_min": [leakage.A_NUM, leakage.A_DEN],
                      "b_edit_similarity_min": [leakage.B_NUM, leakage.B_DEN],
                      "c_lines_held_min": [leakage.C_NUM, leakage.C_DEN], "c_min_test_lines": leakage.C_MIN_LINES,
                      "test_index": "position in the diagrams list of test_set.json"}},
        "environment": environment()})

    names = sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file() and p.name != "SHA256SUMS")
    (out / "SHA256SUMS").write_text("".join(f"{sha256_file(out / n)}  {n}\n" for n in names))
    return report


def print_report(S, report):
    print("\nFunnel (diagrams)")
    for e in S["funnel"]:
        print(f"  {e['step']:<32} class {e['class']:>7,}  sequence {e['sequence']:>7,}  total {e['total']:>8,}")
    last = S["funnel"][-1]["cells"]
    print("\nCells after D6:", json.dumps(last))
    print("\nD6:", json.dumps(report, indent=1))
    print("\nSeconds:", json.dumps({k: round(v, 1) for k, v in S["timing"].items()}))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--work", required=True, help="scratch folder for caches and the D2 renders")
    ap.add_argument("--out", help="folder for the artifacts (data/pool); written after stage d6")
    ap.add_argument("--until", choices=STAGES, default="d6")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--jobs", type=int, default=8, help="extractor JVMs at a time")
    ap.add_argument("--d1-network", choices=("closed", "open"), default="closed",
                    help="closed: no JVM reaches the network; open: the type check of D1 as "
                         "build_test_set.py ran it (remote includes are fetched)")
    ap.add_argument("--no-checks", action="store_true", help="do not stop on the numbers of Implementation note 2")
    args = ap.parse_args()
    sys.path.insert(0, str(HERE))
    try:
        S = build(args.work, args.until, args.workers, args.jobs, args.d1_network, not args.no_checks)
        if args.until == "d6":
            if not args.out:
                raise PoolError("--out is needed to write the artifacts")
            S["work"] = args.work
            print_report(S, write_artifacts(S, args.out))
    except PoolError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
