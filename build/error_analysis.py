#!/usr/bin/env python3
"""Error analysis of the six main test-set runs: outcome counts before and after fine-tuning, a
blind hand-coded failure sample, and relation agreement on endpoint-matched edges.

decisions/error-analysis/record.md, Decision; methodology/analysis-protocol.md §16. Scorer v2 only.

Outcome classes, in order, from each run's stored v2 rows and answer file (never from has_pred):
provider drop (CSR error starts "no prediction"), compile fail (CSR not compiled), compiled
low-structural (per-diagram Element F1 or Relationship F1 < 0.5), ok. C2 = an answer with no
`@enduml`.

The sample: run x diagram type x tier x outcome class (compile fail, low-structural), one case per
cell, drawn with the benchmark sampler's select_cell and cell_seed_key (seed 20260614, the run
name as the model id). Cases are written as blinded packets (shuffled ids, run and condition
hidden, display-only lines stripped into a separate file) for a coder following
build/error_codebook.md. Codes come back as JSON (pass 1, pass 2, the author's resolutions); the
report gives raw agreement and Cohen's kappa on the primary code and the per-run sample counts.
No rate per 1,000 diagrams is derived from the sample.

Relation agreement: a ground-truth edge is endpoint-matched when it is paired one-to-one with an
answer edge on the same unordered pair of v2-normalized names (first diagram); within a pair,
same-relation edges pair first, the rest in sorted order. Population: the edges endpoint-matched
in both runs of a size, on the diagrams both runs compiled (paired by edge). Edges with a note
endpoint are left out. Five class relations; messages apart, as family agreement. Statistic: the
share whose answer relation equals the ground truth's, per run, and its difference; orientation
and the association arrowhead class on the agreeing edges; a ground truth x answer matrix.
Intervals: the main tables' paired draw (seed 20260614, 1,000 resamples, type-7 95%), diagrams
resampled and their edges re-pooled; an interval only with at least 100 edges in 30 diagrams on
the paired denominator. Every number is descriptive and outside the claim family of 12.

Usage (from fine-tune/):
  python3 build/error_analysis.py draw      # writes runs/test/_analysis/error_sample/ (keys, cases)
  python3 build/error_analysis.py report    # writes runs/test/_analysis/error_analysis.{json,md}
Tests: build/tests/test_error_analysis.py
"""
import argparse
import collections
import hashlib
import itertools
import json
import os
import random
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
FT = HERE.parent
sys.path.insert(0, str(HERE))
import main_tables as M  # noqa: E402  (puts the benchmark on sys.path)

INSTRUMENT = M.INSTRUMENT
sys.path.insert(0, str(INSTRUMENT / "evaluation"))
from analysis.bootstrap import N_RESAMPLES, SEED, make_resamples  # noqa: E402
from analysis.failure_sampler import cell_seed_key, select_cell  # noqa: E402
from csr_runner import extract_puml  # noqa: E402
from element_f1 import normalize  # noqa: E402
import element_f1 as ef  # noqa: E402
import relationship_f1 as rf  # noqa: E402

RUNS = FT / "runs" / "test"
OUT = M.OUT
SAMPLE_DIR = OUT / "error_sample"
CODEBOOK = HERE / "error_codebook.md"
JAR = INSTRUMENT / "plantuml-stats-extractor-v2.jar"
JAR_SHA256 = "2747a72a"  # prefix; the full hash is in methodology/scorer-v2.md §7
JAVA_BIN = "/Library/Java/JavaVirtualMachines/temurin-26.jdk/Contents/Home/bin"
TEST_SET = INSTRUMENT / "data" / "test_set.json"
GT_DIR = INSTRUMENT / "data" / "puml_files"
IMAGES = INSTRUMENT / "data" / "puml_images_1568"

PAIRS = {"2B": ("2B_zero_shot", "2B_20000"), "9B": ("9B_zero_shot", "9B_20000"),
         "27B": ("27B_zero_shot", "27B_20000")}
RUN_ORDER = [r for p in PAIRS.values() for r in p]
LOW = 0.5
FAILURES = ("compile_fail", "compiled_low_structural")
CLASSES = ("provider_drop",) + FAILURES + ("ok",)
CLASS_RELATIONS = ["inheritance", "composition", "aggregation", "dependency", "association"]
FLOOR_EDGES, FLOOR_DIAGRAMS = 100, 30
CODES = ["R", "C1", "C2", "C3", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "M"]
NOTE_TYPES = {"note", "tips"}  # the extractor's two note node types
FAILURE_ROWS = ["compile_fail", "c2_no_enduml", "c2_finish_length", "compiled_low_structural",
                "low_element_only", "low_relationship_only", "low_both"]
# Test-set items whose image does not show what the ground truth declares (Decision §2; found in
# the coding and by a check of every image whose source asks for a transparent background).
LIMITATIONS = [
    "Key `7b907a2193a6…`: the ground truth declares `Either --> \"type\" Either.SideType`, which its image does "
    "not draw.",
    "Keys `de434622c1f4…`, `73ece5e02bc2…` and `4ef701ef62d4…`: the source asks for a transparent background and "
    "the test image is rendered on black, so black arrows and labels drawn on it do not show (for `4ef701ef62d4…`, "
    "those outside its one filled box).",
]
DISPLAY_MAX_LINES = 120  # longer answers are shown as the first 90 and last 30 lines

# Display-only lines (Decision §3): removed from the shown answer, kept in <id>.stripped.txt.
_DISPLAY_ONLY = re.compile(
    r"^\s*(skinparam\b|title\b|hide\b|show\b|scale\b|left\s+to\s+right\s+direction\b|"
    r"top\s+to\s+bottom\s+direction\b|!|')", re.I)


# --------------------------------------------------------------------------- #
# stored rows and outcome classes
# --------------------------------------------------------------------------- #
def stored(run, metric, runs=RUNS):
    p = runs / run / "scores" / "v2" / metric / run / f"{metric}_results.json"
    return {d["key"]: d for d in json.load(open(p, encoding="utf-8"))["diagrams"]}


def answer_text(run, key, runs=RUNS):
    return (runs / run / "answers" / f"{key}.puml").read_text(encoding="utf-8", errors="replace")


def finish_reason(run, key, runs=RUNS):
    d = json.load(open(runs / run / "answers" / f"{key}.json", encoding="utf-8"))
    return d["choices"][0].get("finish_reason")


def outcome(csr, el, rel, low=LOW):
    """The record's class rule: the CSR row first, never has_pred."""
    if (csr.get("error") or "").startswith("no prediction"):
        return "provider_drop"
    if not csr["compiled"]:
        return "compile_fail"
    if el["f1"] < low or rel["f1"] < low:
        return "compiled_low_structural"
    return "ok"


def run_table(run, runs=RUNS):
    """{key: row} with the class and the mechanical facts each count needs."""
    csr, el, rel = stored(run, "csr", runs), stored(run, "element_f1", runs), stored(run, "relationship_f1", runs)
    out = {}
    for k in csr:
        text = answer_text(run, k, runs)
        out[k] = {"key": k, "class": outcome(csr[k], el[k], rel[k]), "csr": csr[k], "el": el[k], "rel": rel[k],
                  "no_enduml": "@enduml" not in text, "finish_reason": finish_reason(run, k, runs)}
    return out


def counts(table):
    """Per 1,000 diagrams (the test set holds 1,000, so counts are per 1,000), with shares of the
    run's failures beside them."""
    n = len(table)
    rows = list(table.values())
    cf = [r for r in rows if r["class"] == "compile_fail"]
    low = [r for r in rows if r["class"] == "compiled_low_structural"]
    fails = len(cf) + len(low)
    per = lambda x: round(1000 * x / n, 1)
    c = {
        "n_diagrams": n,
        "provider_drop": per(sum(r["class"] == "provider_drop" for r in rows)),
        "compile_fail": per(len(cf)),
        "c2_no_enduml": per(sum(r["no_enduml"] for r in cf)),
        "c2_finish_length": per(sum(r["finish_reason"] == "length" for r in cf)),
        "compiled_low_structural": per(len(low)),
        "low_element_only": per(sum(r["el"]["f1"] < LOW <= r["rel"]["f1"] for r in low)),
        "low_relationship_only": per(sum(r["rel"]["f1"] < LOW <= r["el"]["f1"] for r in low)),
        "low_both": per(sum(r["el"]["f1"] < LOW and r["rel"]["f1"] < LOW for r in low)),
        "ok": per(sum(r["class"] == "ok" for r in rows)),
        "failures": per(fails),
    }
    c["share_of_failures"] = {k: (round(c[k] / c["failures"], 4) if c["failures"] else None)
                              for k in FAILURE_ROWS}
    return c


def transitions(tz, tf):
    """Each diagram's zero-shot class against its fine-tuned class."""
    m = collections.Counter((tz[k]["class"], tf[k]["class"]) for k in tz)
    return {a: {b: m[(a, b)] for b in CLASSES} for a in CLASSES}


# --------------------------------------------------------------------------- #
# the sample
# --------------------------------------------------------------------------- #
def test_set():
    return {d["key"][:-5]: d for d in json.load(open(TEST_SET, encoding="utf-8"))["diagrams"]}


def draw_sample(tables, meta):
    """One case per run x type x tier x failure class, in RUN_ORDER and cell order."""
    cases = []
    for run in RUN_ORDER:
        cells = collections.defaultdict(list)
        for k, r in tables[run].items():
            if r["class"] in FAILURES:
                cells[(meta[k]["primary_type"], meta[k]["tier"], r["class"])].append({"key": k})
        for (typ, tier, cls) in sorted(cells, key=lambda c: (c[0], c[1], FAILURES.index(c[2]))):
            for row in select_cell(cells[(typ, tier, cls)], 1, cell_seed_key(run, typ, tier, cls)):
                cases.append({"run": run, "condition": condition(run), "key": row["key"],
                              "primary_type": typ, "tier": tier, "class": cls})
    order = list(range(len(cases)))
    random.Random(f"{SEED}|error-analysis|blind").shuffle(order)
    width = len(str(len(cases)))
    for n, i in enumerate(order, 1):
        cases[i]["case"] = f"E{n:0{width}d}"
    return sorted(cases, key=lambda c: c["case"])


def condition(run):
    for z, f in PAIRS.values():
        if run == z:
            return "zero_shot"
        if run == f:
            return "fine_tuned"
    raise KeyError(run)


def isolate(text):
    """The answer's PlantUML block for display: from the first @startuml to the last @enduml, or
    to the end when there is no @enduml (a trailing markdown fence dropped)."""
    s = text.find("@startuml")
    body = text[s:] if s != -1 else text
    e = body.rfind("@enduml")
    if e != -1:
        body = body[:e + len("@enduml")]
    else:
        body = re.sub(r"\n?```\s*$", "", body.rstrip())
    return body


def strip_display(block):
    """(shown lines, stripped lines). Removes display-only lines, `skinparam … {` blocks and
    `/' … '/` comment blocks."""
    shown, stripped = [], []
    depth, in_comment = 0, False
    for line in block.split("\n"):
        s = line.strip()
        if in_comment:
            stripped.append(line)
            in_comment = "'/" not in s
            continue
        if depth:
            stripped.append(line)
            depth += s.count("{") - s.count("}")
            depth = max(depth, 0)
            continue
        if s.startswith("/'"):
            stripped.append(line)
            in_comment = "'/" not in s[2:]
            continue
        if _DISPLAY_ONLY.match(line):
            stripped.append(line)
            if re.match(r"^\s*skinparam\b", line, re.I):
                depth = max(s.count("{") - s.count("}"), 0)
            continue
        shown.append(line)
    return shown, stripped


def clip(lines, limit=DISPLAY_MAX_LINES):
    if len(lines) <= limit:
        return lines
    head, tail = lines[:90], lines[-30:]
    return head + [f"... {len(lines) - 120} lines omitted ..."] + tail


def renderer_note(csr_row, text, no_enduml):
    err = csr_row.get("error") or ""
    parts = []
    if no_enduml:
        parts.append("The answer has no `@enduml`.")
    m = re.match(r"Error line (\d+)", err)
    if m:
        lines = extract_puml(text).split("\n")
        n = int(m.group(1))
        src = lines[n - 1] if 0 < n <= len(lines) else "(line not in the block)"
        parts.append(f"Renderer: error at line {n} of the answer block: `{src.strip()}`")
    elif err:
        parts.append(f"Renderer: {err.split(' in file:')[0]}")
    return " ".join(parts)


def _counter_lines(c):
    return ", ".join(f"`{k}`" + (f" ×{v}" if v > 1 else "") for k, v in sorted(c.items())) or "none"


def _edge_text(k):
    s, t, rel = k[0], k[1], k[2]
    return f"{s} -> {t} : {rel}" + (" (one arrowhead)" if len(k) > 3 else "")


def structural_diff(gt_rec, pred_rec):
    gn = collections.Counter(ef.names_from_record(gt_rec or {}))
    pn = collections.Counter(ef.names_from_record(pred_rec or {}))
    ge = collections.Counter(_edge_text(k) for k in rf.edges_from_record(gt_rec or {}))
    pe = collections.Counter(_edge_text(k) for k in rf.edges_from_record(pred_rec or {}))
    return {"nodes": {"matched": gn & pn, "missed": gn - pn, "spurious": pn - gn},
            "edges": {"matched": ge & pe, "missed": ge - pe, "spurious": pe - ge}}


def case_markdown(case, gt_text, shown, n_stripped, note, diff, el_f1, rel_f1):
    out = [f"# Case {case['case']}", "",
           f"Diagram type: {case['primary_type']} · tier {case['tier']} · outcome: "
           f"{'compile fail' if case['class'] == 'compile_fail' else 'compiled low-structural'}",
           f"Image: `{case['case']}.png`", "", "## Ground truth", "", "```plantuml", gt_text.rstrip(), "```", "",
           f"## Answer ({n_stripped} display-only lines removed; they are in `{case['case']}.stripped.txt`)", "",
           "```plantuml", *clip(shown), "```", ""]
    if case["class"] == "compile_fail":
        out += ["## Renderer", "", note or "Renderer: no PNG produced.", ""]
    else:
        out += ["## Structural diff (scorer v2)", "", f"Element F1 {el_f1:.3f} · Relationship F1 {rel_f1:.3f}", ""]
        if note:
            out += [note, ""]
        for part in ("nodes", "edges"):
            for kind in ("matched", "missed", "spurious"):
                c = diff[part][kind]
                out.append(f"- {part} {kind} ({sum(c.values())}): {_counter_lines(c)}")
        out.append("")
    return "\n".join(out)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --------------------------------------------------------------------------- #
# graphs (v2 JAR), checked against the stored rows
# --------------------------------------------------------------------------- #
def extract_all(runs=RUN_ORDER):
    """{"gt": {key: record}, run: {key: record}} re-extracted with the v2 JAR, the crash answers
    held back as build/score_test_runs.py holds them back; any mismatch with a stored v2 row stops."""
    from element_f1_runner import extract_graphs, write_extracted
    if not sha256(JAR).startswith(JAR_SHA256):
        raise SystemExit(f"{JAR}: not the frozen v2 JAR")
    os.environ["PATH"] = JAVA_BIN + os.pathsep + os.environ["PATH"]
    keys = sorted(test_set())
    graphs = {}
    with tempfile.TemporaryDirectory() as tmp:
        write_extracted([str(GT_DIR / f"{k}.puml") for k in keys], os.path.join(tmp, "gt"))
        graphs["gt"] = extract_graphs(os.path.join(tmp, "gt"), str(JAR))
        for run in runs:
            crashes = set(json.load(open(RUNS / run / "scores" / "manifest.json"))["extractor_crashes"]["v2"])
            d = os.path.join(tmp, run)
            write_extracted([str(RUNS / run / "answers" / f"{k}.puml") for k in keys if k not in crashes], d)
            graphs[run] = extract_graphs(d, str(JAR))
            check_graphs(graphs["gt"], graphs[run], stored(run, "element_f1"), stored(run, "relationship_f1"), keys)
    return graphs


def check_graphs(gt, pred, el, rel, keys):
    bad = []
    for k in keys:
        p = pred.get(k) or {}
        g = ef.prf(ef.names_from_record(gt[k]), ef.names_from_record(p))
        if (g["tp"], g["fp"], g["fn"]) != (el[k]["tp"], el[k]["fp"], el[k]["fn"]):
            bad.append((k, "element"))
        for r in [None] + rf.RELATIONS:
            g = ef.prf(rf.edges_from_record(gt[k], r), rf.edges_from_record(p, r))
            s = rel[k] if r is None else rel[k]["by_relation"][r]
            if (g["tp"], g["fp"], g["fn"]) != (s["tp"], s["fp"], s["fn"]):
                bad.append((k, r or "relationship"))
    if bad:
        raise SystemExit(f"{len(bad)} re-extracted rows differ from the stored v2 rows, e.g. {bad[:3]}")


# --------------------------------------------------------------------------- #
# relation agreement
# --------------------------------------------------------------------------- #
def edges_of(rec):
    """Edges of one record, names v2-normalized; `note` marks an edge with a note endpoint."""
    rec = rec or {}
    notes = {normalize(n["name"]) for n in rec.get("nodes") or [] if n.get("type") in NOTE_TYPES}
    out = []
    for e in rec.get("edges") or []:
        s, t = normalize(e.get("source", "")), normalize(e.get("target", ""))
        out.append({"s": s, "t": t, "rel": e["relation"], "ah": e.get("arrowhead"),
                    "note": s in notes or t in notes})
    return out


def _sort_key(e):
    return (e["rel"], e["s"], e["t"], e["ah"] or "")


def match(gt_edges, pred_edges):
    """One-to-one endpoint matching: [answer edge or None] aligned with gt_edges. Per unordered
    pair of names, same-relation edges pair first, the rest in sorted order."""
    pred_by_pair = collections.defaultdict(list)
    for e in pred_edges:
        pred_by_pair[frozenset((e["s"], e["t"]))].append(e)
    gt_by_pair = collections.defaultdict(list)
    for i, e in enumerate(gt_edges):
        gt_by_pair[frozenset((e["s"], e["t"]))].append(i)
    out = [None] * len(gt_edges)
    for pair, idx in gt_by_pair.items():
        free = sorted(pred_by_pair.get(pair, []), key=_sort_key)
        idx = sorted(idx, key=lambda i: _sort_key(gt_edges[i]))
        left = []
        for i in idx:
            j = next((n for n, p in enumerate(free) if p["rel"] == gt_edges[i]["rel"]), None)
            if j is None:
                left.append(i)
            else:
                out[i] = free.pop(j)
        for i in left:
            if free:
                out[i] = free.pop(0)
    return out


def oriented(e):
    """(tail, head) of a directed edge; None for an association with no or two arrowheads."""
    if e["rel"] == "association":
        if e["ah"] == "target":
            return (e["s"], e["t"])
        if e["ah"] == "source":
            return (e["t"], e["s"])
        return None
    return (e["s"], e["t"])


def same_orientation(g, a):
    """For an agreeing pair: the same canonical source and target; for an association, the same
    arrowhead class (one head, at the same node; or none / two heads)."""
    return oriented(g) == oriented(a)


def paired_edges(gt, pz, pf, joint_keys):
    """Per diagram key: the ground-truth edges endpoint-matched in both runs (note links out),
    each as (relation, answer relation zs, answer relation ft, orientation zs, orientation ft)."""
    out = {}
    for k in joint_keys:
        g = edges_of(gt.get(k))
        mz, mf = match(g, edges_of(pz.get(k))), match(g, edges_of(pf.get(k)))
        rows = []
        for e, a, b in zip(g, mz, mf):
            if e["note"] or a is None or b is None:
                continue
            rows.append((e["rel"], a["rel"], b["rel"],
                         a["rel"] == e["rel"] and same_orientation(e, a),
                         b["rel"] == e["rel"] and same_orientation(e, b)))
        out[k] = rows
    return out


def agreement(paired, keys, draws):
    """Per relation: support, each run's agreement share, Δ with the paired interval, the
    orientation rows and the ground truth x answer matrix. `keys` gives the draw's index order."""
    per = {}
    for rel in CLASS_RELATIONS + ["message"]:
        counts = {}
        for i, k in enumerate(keys):
            rows = [r for r in paired.get(k, []) if r[0] == rel]
            if rows:
                counts[i] = (len(rows), sum(r[1] == rel for r in rows), sum(r[2] == rel for r in rows),
                             sum(r[3] for r in rows), sum(r[4] for r in rows))
        per[rel] = counts
    mult = [collections.Counter(d) for d in draws]
    res = {}
    for rel, counts in per.items():
        n = sum(c[0] for c in counts.values())
        dia = len(counts)
        az, af = sum(c[1] for c in counts.values()), sum(c[2] for c in counts.values())
        oz, of_ = sum(c[3] for c in counts.values()), sum(c[4] for c in counts.values())
        pz, pf = (az / n if n else None), (af / n if n else None)
        delta = M.sub(pf, pz)
        row = {"edges": n, "diagrams": dia, "zs": pz, "ft": pf,
               "orientation_zs": (oz / az if az else None), "orientation_ft": (of_ / af if af else None)}
        if n >= FLOOR_EDGES and dia >= FLOOR_DIAGRAMS:
            vals = {"zs": [], "ft": [], "delta": []}
            for m in mult:
                bn = sum(c[0] * m[i] for i, c in counts.items() if i in m)
                bz = sum(c[1] * m[i] for i, c in counts.items() if i in m)
                bf = sum(c[2] * m[i] for i, c in counts.items() if i in m)
                z = bz / bn if bn else None
                f = bf / bn if bn else None
                vals["zs"].append(z)
                vals["ft"].append(f)
                vals["delta"].append(M.sub(f, z))
            row["ci"] = {"zs": M.interval(pz, vals["zs"]), "ft": M.interval(pf, vals["ft"]),
                         "delta": M.interval(delta, vals["delta"])}
        else:
            why = f"below the floor ({n} edges in {dia} diagrams; needs {FLOOR_EDGES} in {FLOOR_DIAGRAMS})"
            row["ci"] = {"zs": M.point_only(pz, why), "ft": M.point_only(pf, why), "delta": M.point_only(delta, why)}
        res[rel] = row
    matrix = {"zs": collections.Counter(), "ft": collections.Counter()}
    for rows in paired.values():
        for r in rows:
            matrix["zs"][f"{r[0]}->{r[1]}"] += 1
            matrix["ft"][f"{r[0]}->{r[2]}"] += 1
    res["_matrix"] = {c: dict(sorted(m.items())) for c, m in matrix.items()}
    return res


# --------------------------------------------------------------------------- #
# coder agreement and separability
# --------------------------------------------------------------------------- #
def kappa(a, b):
    """Raw agreement and Cohen's kappa of two label lists."""
    n = len(a)
    if not n:
        return {"n": 0, "agree": 0, "raw": None, "kappa": None}
    agree = sum(x == y for x, y in zip(a, b))
    po = agree / n
    ca, cb = collections.Counter(a), collections.Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return {"n": n, "agree": agree, "raw": po, "kappa": (po - pe) / (1 - pe) if pe < 1 else None}


SEPARABILITY_CUES = {
    "skinparam": r"^\s*skinparam\b", "ltr": r"left to right direction", "define": r"^\s*!",
    "comment": r"^\s*'", "hideshow": r"^\s*(hide|show)\b", "title": r"^\s*title\b",
    "direction_arrow": r"-(up|down|left|right)-", "note": r"^\s*note\b", "alias": r"\bas\s+\w",
    "quoted": r'"', "long": None, "noend": None,
}


def cue_features(shown_text):
    f = tuple(bool(re.search(p, shown_text, re.M | re.I)) if p else False
              for p in SEPARABILITY_CUES.values())
    names = list(SEPARABILITY_CUES)
    f = list(f)
    f[names.index("long")] = shown_text.count("\n") > 60
    f[names.index("noend")] = "@enduml" not in shown_text
    return tuple(f)


def separability(zs_feats, ft_feats):
    """Balanced accuracy of the best rule "zero-shot iff any of A and none of B" (|A| <= 3,
    |B| <= 2) over the cue list, searched in sample: an upper bound for that rule family."""
    names = list(SEPARABILITY_CUES)
    cz, cf = collections.Counter(zs_feats), collections.Counter(ft_feats)
    nz, nf = sum(cz.values()), sum(cf.values())
    if not nz or not nf:
        return None
    best = (0.5, (), ())
    idx = range(len(names))
    for a in range(0, 4):
        for A in itertools.combinations(idx, a):
            for b in range(0, 3):
                for B in itertools.combinations([i for i in idx if i not in A], b):
                    g = lambda x: (any(x[i] for i in A) if A else True) and not any(x[i] for i in B)
                    tpr = sum(v for x, v in cz.items() if g(x)) / nz
                    tnr = sum(v for x, v in cf.items() if not g(x)) / nf
                    ba = (tpr + tnr) / 2
                    if ba > best[0] + 1e-12:
                        best = (ba, A, B)
    return {"balanced_accuracy": round(best[0], 3), "zero_shot_if_any": [names[i] for i in best[1]],
            "and_none_of": [names[i] for i in best[2]], "n_zero_shot": nz, "n_fine_tuned": nf}


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #
def cmd_draw(args):
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out}: exists and is not empty; the sample is drawn once")
    meta = test_set()
    tables = {run: run_table(run) for run in RUN_ORDER}
    cases = draw_sample(tables, meta)
    graphs = extract_all()
    cdir = out / "cases"
    cdir.mkdir(parents=True)
    for c in cases:
        k, run = c["key"], c["run"]
        text = answer_text(run, k)
        shown, stripped = strip_display(isolate(text))
        r = tables[run][k]
        note = renderer_note(r["csr"], text, r["no_enduml"])
        diff = structural_diff(graphs["gt"].get(k), graphs[run].get(k))
        gt_text = (GT_DIR / f"{k}.puml").read_text(encoding="utf-8", errors="replace")
        (cdir / f"{c['case']}.md").write_text(
            case_markdown(c, gt_text, shown, len(stripped), note, diff, r["el"]["f1"], r["rel"]["f1"]),
            encoding="utf-8")
        (cdir / f"{c['case']}.stripped.txt").write_text("\n".join(stripped) + "\n", encoding="utf-8")
        shutil.copyfile(IMAGES / f"{k}.png", cdir / f"{c['case']}.png")
    keys = {"_doc": "The unblinded key of the error-analysis sample. Coders never read this file.",
            "seed": SEED, "codebook_sha256": sha256(CODEBOOK), "error_analysis_sha256": sha256(__file__),
            "n_cases": len(cases), "cases": cases}
    (out / "keys.json").write_text(json.dumps(keys, indent=1), encoding="utf-8")
    print(f"drawn: {len(cases)} cases into {cdir}; key {out / 'keys.json'}")


def load_codes(sample_dir):
    codes = {}
    for name in ("pass1", "pass2", "final"):
        p = sample_dir / f"codes_{name}.json"
        codes[name] = json.load(open(p, encoding="utf-8")) if p.exists() else {}
    return codes


def coding_summary(cases, codes):
    ids = [c["case"] for c in cases]
    p1, p2, fin = codes["pass1"], codes["pass2"], codes["final"]
    done = [i for i in ids if i in p1 and i in p2]
    summary = {"coded_pass1": sum(i in p1 for i in ids), "coded_pass2": sum(i in p2 for i in ids)}
    by_cond = {}
    for cond in ("zero_shot", "fine_tuned", "all"):
        sel = [c["case"] for c in cases if c["case"] in done and (cond == "all" or c["condition"] == cond)]
        by_cond[cond] = kappa([p1[i]["primary"] for i in sel], [p2[i]["primary"] for i in sel])
    summary["primary_agreement"] = by_cond
    disagree = [i for i in done if p1[i]["primary"] != p2[i]["primary"]]
    flagged = [i for i in done if p1[i].get("ask_author") or p2[i].get("ask_author")]
    summary["to_author"] = sorted(set(disagree) | set(flagged))
    unresolved = [i for i in summary["to_author"] if i not in fin]
    summary["unresolved"] = unresolved
    final = {}
    for i in done:
        if i in fin:
            final[i] = fin[i]
        elif i not in summary["to_author"]:
            final[i] = p1[i]
    summary["final_complete"] = len(final) == len(ids)
    per_run = {}
    for c in cases:
        if c["case"] in final:
            per_run.setdefault(c["run"], collections.Counter())[final[c["case"]]["primary"]] += 1
    summary["sample_primary_counts"] = {run: {code: per_run.get(run, {}).get(code, 0) for code in CODES}
                                        for run in RUN_ORDER}
    summary["final"] = final
    return summary


def cmd_report(args):
    t0 = time.time()
    sample_dir = Path(args.sample)
    keys = json.load(open(sample_dir / "keys.json", encoding="utf-8"))
    if sha256(CODEBOOK) != keys["codebook_sha256"]:
        raise SystemExit("the codebook changed after the sample was drawn; it is frozen")
    tables = {run: run_table(run) for run in RUN_ORDER}
    res = {"meta": {"seed": SEED, "n_resamples": N_RESAMPLES, "low_structural_cut": LOW,
                    "floor": [FLOOR_EDGES, FLOOR_DIAGRAMS], "codebook_sha256": keys["codebook_sha256"],
                    "error_analysis_sha256_at_draw": keys["error_analysis_sha256"],
                    "error_analysis_sha256": sha256(__file__), "jar_sha256": sha256(JAR)},
           "counts": {run: counts(tables[run]) for run in RUN_ORDER},
           "transitions": {size: transitions(tables[z], tables[f]) for size, (z, f) in PAIRS.items()}}
    graphs = extract_all()
    all_keys = sorted(test_set())
    draws = make_resamples(len(all_keys), N_RESAMPLES, SEED)
    res["agreement"] = {}
    for size, (z, f) in PAIRS.items():
        joint = [k for k in all_keys if tables[z][k]["csr"]["compiled"] and tables[f][k]["csr"]["compiled"]]
        paired = paired_edges(graphs["gt"], graphs[z], graphs[f], joint)
        res["agreement"][size] = dict(agreement(paired, all_keys, draws), n_joint=len(joint))
    seps = {}
    for size, (z, f) in PAIRS.items():
        fz = [cue_features("\n".join(strip_display(isolate(answer_text(z, k)))[0]))
              for k, r in tables[z].items() if r["class"] in FAILURES]
        ff = [cue_features("\n".join(strip_display(isolate(answer_text(f, k)))[0]))
              for k, r in tables[f].items() if r["class"] in FAILURES]
        seps[size] = separability(fz, ff)
    res["separability"] = seps
    coders = sample_dir / "coders.json"
    res["sample"] = {"n_cases": keys["n_cases"],
                     "coders": json.load(open(coders, encoding="utf-8")) if coders.exists() else None,
                     "cells": collections.Counter(c["run"] for c in keys["cases"]),
                     "coding": coding_summary(keys["cases"], load_codes(sample_dir))}
    res["meta"]["seconds"] = round(time.time() - t0, 1)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "error_analysis.json").write_text(json.dumps(res, indent=1, default=dict), encoding="utf-8")
    desc = sample_dir / "description.md"
    (out / "error_analysis.md").write_text(markdown(res, desc.read_text(encoding="utf-8") if desc.exists() else None),
                                           encoding="utf-8")
    print(f"written: {out / 'error_analysis.json'}, error_analysis.md ({res['meta']['seconds']} s)")
    return res


# --------------------------------------------------------------------------- #
# markdown
# --------------------------------------------------------------------------- #
def _p(x, d=3):
    return "—" if x is None else f"{x:.{d}f}"


def _ci(c):
    if c.get("ci") is None:
        return f"{_p(c['point'])} (point)"
    lo, hi = c["ci"]
    return f"{c['point']:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def markdown(res, description=None):
    L = ["# Error analysis of the six main test-set runs", "",
         "Scorer v2. `build/error_analysis.py`; `methodology/analysis-protocol.md` §16. Every number here is "
         "descriptive and outside the claim family of 12.", ""]
    L += ["## 1. Outcomes per 1,000 diagrams", "",
          "| Run | compile fail | of which no `@enduml` (C2) | `finish_reason` length | low-structural | "
          "El only / Rel only / both | ok | failures |",
          "|---|---|---|---|---|---|---|---|"]
    for run in RUN_ORDER:
        c = res["counts"][run]
        L.append(f"| `{run}` | {c['compile_fail']:g} | {c['c2_no_enduml']:g} | {c['c2_finish_length']:g} | "
                 f"{c['compiled_low_structural']:g} | {c['low_element_only']:g} / {c['low_relationship_only']:g} / "
                 f"{c['low_both']:g} | {c['ok']:g} | {c['failures']:g} |")
    L += ["", "Provider drop (R) is " + ", ".join(f"{res['counts'][r]['provider_drop']:g}" for r in RUN_ORDER)
          + " per 1,000 in the six runs.", "",
          "Shares of each run's own failures (not of the 1,000 diagrams):", "",
          "| Run | compile fail | no `@enduml` | `finish_reason` length | low-structural | El only | Rel only | both |",
          "|---|---|---|---|---|---|---|---|"]
    for run in RUN_ORDER:
        s = res["counts"][run]["share_of_failures"]
        L.append(f"| `{run}` | " + " | ".join(_p(s[k]) for k in FAILURE_ROWS) + " |")
    L += ["",
          "", "## 2. Transitions, zero-shot class → fine-tuned class (diagrams)", ""]
    short = {"provider_drop": "drop", "compile_fail": "compile fail", "compiled_low_structural": "low", "ok": "ok"}
    for size, t in res["transitions"].items():
        L += [f"**{size}**", "", "| zero-shot \\ fine-tuned | " + " | ".join(short[c] for c in CLASSES) + " |",
              "|---|" + "---|" * len(CLASSES)]
        for a in CLASSES:
            L.append(f"| {short[a]} | " + " | ".join(str(t[a][b]) for b in CLASSES) + " |")
        L.append("")
    L += ["## 3. Relation agreement on endpoint-matched edges", "",
          "Ground-truth edges endpoint-matched in both runs of a size, on the diagrams both compiled; note "
          "links left out. Share whose answer relation equals the ground truth's. Interval: paired bootstrap, "
          "95%, only with at least 100 edges in 30 diagrams. A realization arrow counts as inheritance; a "
          "dotted line with no head counts as dependency; message arrow styles are not distinguished.", "",
          "| Size | Relation | edges / diagrams | zero-shot | fine-tuned | Δ [95% CI] | orientation zs / ft |",
          "|---|---|---|---|---|---|---|"]
    n_ci = 0
    for size, a in res["agreement"].items():
        for rel in CLASS_RELATIONS + ["message"]:
            r = a[rel]
            n_ci += r["ci"]["delta"].get("ci") is not None
            label = rel if rel != "message" else "message (family agreement)"
            L.append(f"| {size} | {label} | {r['edges']} / {r['diagrams']} | {_p(r['zs'])} | {_p(r['ft'])} | "
                     f"{_ci(r['ci']['delta'])} | {_p(r['orientation_zs'])} / {_p(r['orientation_ft'])} |")
    L += ["", f"{n_ci} intervals on Δ, uncorrected, none in the claim family. Orientation: the share of agreeing "
          "edges with the same canonical source and target (association: the same arrowhead class).", ""]
    L += ["### Ground truth → answer relation, paired edges", ""]
    for size, a in res["agreement"].items():
        for cond in ("zs", "ft"):
            m = a["_matrix"][cond]
            L.append(f"- {size} {cond}: " + ", ".join(f"{k} {v}" for k, v in m.items()))
    L += ["", "## 4. The hand-coded sample", ""]
    s = res["sample"]
    cod = s["coding"]
    L.append(f"{s['n_cases']} cases, one per run × type × tier × failure class. Coded blind in two independent "
             f"passes (pass 1: {cod['coded_pass1']}, pass 2: {cod['coded_pass2']}).")
    cs = s.get("coders") or {}
    L.append(f"Coders: pass 1 {cs.get('pass1', 'not named')}; pass 2 {cs.get('pass2', 'not named')}; "
             f"disagreements and flagged cases: {cs.get('final', 'not named')}.")
    pa = cod["primary_agreement"]
    L += ["", "| Condition | cases | agree | raw | κ |", "|---|---|---|---|---|"]
    for cond in ("zero_shot", "fine_tuned", "all"):
        k = pa[cond]
        L.append(f"| {cond} | {k['n']} | {k['agree']} | {_p(k['raw'])} | {_p(k['kappa'])} |")
    L += ["", f"To the author: {len(cod['to_author'])} cases; unresolved: {len(cod['unresolved'])}.", ""]
    L += ["Sample primary codes per run (counts in the sample, not rates):", "",
          "| Run | " + " | ".join(CODES) + " |", "|---|" + "---|" * len(CODES)]
    for run in RUN_ORDER:
        L.append(f"| `{run}` | " + " | ".join(str(cod["sample_primary_counts"][run][c]) for c in CODES) + " |")
    L += ["", "Separability of the conditions on the displayed (stripped) answers, all failure cells: balanced "
          "accuracy of the best rule \"zero-shot if any of A and none of B\" over a fixed cue list, in sample.", ""]
    for size, sp in res["separability"].items():
        if sp:
            L.append(f"- {size}: {sp['balanced_accuracy']:.3f} (zero-shot if any of {sp['zero_shot_if_any']} and none of "
                     f"{sp['and_none_of']}; {sp['n_zero_shot']} zero-shot, {sp['n_fine_tuned']} fine-tuned cells)")
    L += ["", "## 5. Description", "", description.strip() if description else "(written after the coding)", "",
          "## Limitations", "", *[f"- {x}" for x in LIMITATIONS],
          "- Intervals cover diagram variance only, not decoding or training seed.", ""]
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("draw")
    d.add_argument("--out", default=str(SAMPLE_DIR))
    r = sub.add_parser("report")
    r.add_argument("--sample", default=str(SAMPLE_DIR))
    r.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)
    return cmd_draw(args) if args.cmd == "draw" else cmd_report(args)


if __name__ == "__main__":
    main()
