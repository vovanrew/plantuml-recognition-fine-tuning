#!/usr/bin/env python3
"""The main tables of Р.4: the six main test-set runs, each size's fine-tuned run against its
self-served zero-shot run, with paired bootstrap intervals.

decisions/main-tables-intervals/record.md, Decision. Reads the data roots that
build/score_test_runs.py writes (runs/test/_root/{v2,v1}) through the benchmark's own loader and
leak gate, with the study's registry (build/test_registry.json). The statistics the benchmark
does not compute (deltas, the delta of the population gap, differences of deltas, tiers, the
joint set, the Phase 0 check) are computed here from the benchmark's primitives, imported
unchanged: analysis.bootstrap make_resamples / model_stats / percentile, seed 20260614,
1,000 resamples, 95% type-7 percentile interval (CI_Q). One index draw is applied to every run
and both scorer versions, so every interval is paired by diagram.

Populations. Full-set is the stored per-diagram rows as the frozen runners write them (Р.3's
computation; no CSR mask). Compiled-only is each run's own compiled diagrams. The failed = 0
variant (a non-compiling answer scored 0 by the CSR flag) appears only in the supplementary
sensitivity table.

Reading rules fixed by the Decision:
  - claim family: Δ CSR and Δ compiled-only Element F1 micro for each size (6) and their three
    differences of deltas (6), 12 intervals at nominal 95%, no correction; everything else is
    descriptive. Δ gap is shown with intervals but is not a member.
  - a statistic on an empty population is undefined (None), never 0; a cell with any undefined
    resample gets a point only.
  - compiled-only tier cells: an interval only when both runs have at least 30 compiled diagrams
    in the tier. Per-relation cells: an interval only with at least 100 ground-truth edges and 30
    diagrams holding the relation in both runs' population.
  - Phase 0, on the diagrams both runs of a size compiled: Δ_v2 − Δ_v1 interval below 0: held;
    contains 0: no detectable change; above 0: did not hold.

Usage (from fine-tune/):
  python3 build/main_tables.py                     # writes runs/test/_analysis/
  python3 build/main_tables.py --root <root> --out <dir>
Tests: build/tests/test_main_tables.py
"""
import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
FT = HERE.parent
INSTRUMENT = FT.parent / "plantuml-recognition-evaluation"
sys.path.insert(0, str(INSTRUMENT))
from analysis.aggregate import POPULATIONS, RELATIONS, TIERS, TYPES, mean_macro, pool_micro  # noqa: E402
from analysis.bootstrap import CI_Q, N_RESAMPLES, SEED, make_resamples, model_stats, percentile  # noqa: E402
from analysis.leak_gate import count_reasoning_leaks, predictions_dir  # noqa: E402
from analysis.panel import load_eligible_models  # noqa: E402
from analysis.registry import load_registry  # noqa: E402

REGISTRY = HERE / "test_registry.json"
ROOT = FT / "runs" / "test" / "_root"
OUT = FT / "runs" / "test" / "_analysis"
N_CELLS = 1000

# Scout P4: each size's zero-shot and fine-tuned entries, fixed here.
SIZES = ["2B", "9B", "27B"]
PAIRS = {"2B": ("qwen3.5-2b-zs", "qwen3.5-2b-ft20k"),
         "9B": ("qwen3.5-9b-zs", "qwen3.5-9b-ft20k"),
         "27B": ("qwen3.5-27b-zs", "qwen3.5-27b-ft20k")}
DOD_PAIRS = [("2B", "9B"), ("9B", "27B"), ("2B", "27B")]

ZERO_EDGE_EXPECTED = 107        # diagrams whose v2 ground truth has no edges (challenge BLOCKS 1)
TIER_FLOOR = 30                 # compiled diagrams per run in a tier, for a compiled-only interval
REL_FLOOR_EDGES, REL_FLOOR_DIAGRAMS = 100, 30

CSR = "csr|csr"
EL = {"full": "element_f1|micro|zeros_for_failed", "co": "element_f1|micro|compiled_only",
      "gap": "gap|element_f1|micro"}
REL = {"full": "relationship_f1|micro|zeros_for_failed", "co": "relationship_f1|micro|compiled_only",
       "gap": "gap|relationship_f1|micro"}
GAPS = {f"gap|{m}|{lv}": (f"{m}|{lv}|compiled_only", f"{m}|{lv}|zeros_for_failed")
        for m in ("element_f1", "relationship_f1") for lv in ("micro", "macro")}
GAPS["gap|chrf|macro"] = ("chrf|macro|compiled_only", "chrf|macro|zeros_for_failed")
TYPE_ACC = "type_accuracy|accuracy|compiled_only"
NE = {"zeros_for_failed": "ne|relationship_f1|macro|zeros_for_failed",
      "compiled_only": "ne|relationship_f1|macro|compiled_only"}

LABEL = {CSR: "CSR",
         EL["full"]: "Element F1 micro, full-set", EL["co"]: "Element F1 micro, compiled-only",
         EL["gap"]: "gap Element F1 micro",
         REL["full"]: "Relationship F1 micro, full-set", REL["co"]: "Relationship F1 micro, compiled-only",
         REL["gap"]: "gap Relationship F1 micro",
         "element_f1|macro|zeros_for_failed": "Element F1 macro, full-set",
         "element_f1|macro|compiled_only": "Element F1 macro, compiled-only",
         "relationship_f1|macro|zeros_for_failed": "Relationship F1 macro, full-set",
         "relationship_f1|macro|compiled_only": "Relationship F1 macro, compiled-only",
         "gap|element_f1|macro": "gap Element F1 macro", "gap|relationship_f1|macro": "gap Relationship F1 macro",
         "chrf|macro|zeros_for_failed": "chrF++ macro, full-set",
         "chrf|macro|compiled_only": "chrF++ macro, compiled-only", "gap|chrf|macro": "gap chrF++ macro",
         TYPE_ACC: "type accuracy",
         NE["zeros_for_failed"]: "Relationship F1 macro, full-set, zero-edge diagrams left out",
         NE["compiled_only"]: "Relationship F1 macro, compiled-only, zero-edge diagrams left out"}


# --------------------------------------------------------------------------- #
# statistics on one row list
# --------------------------------------------------------------------------- #
def stats(rows):
    """model_stats (the benchmark's), with every statistic on an empty population undefined
    (None) instead of 0.0, plus the population gaps (compiled-only − full-set)."""
    s = dict(model_stats(rows)) if rows else {k: None for k in model_stats([_EMPTY_ROW])}
    for metric, field in (("element_f1", "element"), ("relationship_f1", "relationship")):
        for pop in POPULATIONS:
            pr = [r[field] for r in rows if pop == "zeros_for_failed" or r[field]["compiled"]]
            if not pr:
                s[f"{metric}|micro|{pop}"] = s[f"{metric}|macro|{pop}"] = None
            elif sum(x["tp"] + x["fp"] + x["fn"] for x in pr) == 0:
                s[f"{metric}|micro|{pop}"] = None
    if not any(r["chrf"]["compiled"] for r in rows):
        s["chrf|macro|compiled_only"] = None
    for g, (c, z) in GAPS.items():
        s[g] = sub(s[c], s[z])
    s["n"] = len(rows)
    s["n_compiled"] = sum(1 for r in rows if r["csr"]["compiled"])
    return s


_EMPTY_ROW = {"csr": {"compiled": False}, "chrf": {"score": 0.0, "compiled": False},
              "element": {"tp": 0, "fp": 0, "fn": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
                          "compiled": False, "type_accuracy": {"matched": 0, "correct": 0, "excluded": 0}},
              "relationship": {"tp": 0, "fp": 0, "fn": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
                               "compiled": False,
                               "by_relation": {rl: {"tp": 0, "fp": 0, "fn": 0} for rl in RELATIONS}}}


def macro_or_none(metric_rows):
    return mean_macro(metric_rows)["f1"] if metric_rows else None


def micro_or_none(metric_rows):
    m = pool_micro(metric_rows)
    return m["f1"] if metric_rows and m["tp"] + m["fp"] + m["fn"] else None


def sub(a, b):
    return None if a is None or b is None else a - b


def _zero(m):
    """One metric's row of a non-compiling answer, scored 0: no true or false positive, every
    ground-truth item a false negative, per-diagram P = R = F1 = 0."""
    out = dict(m, tp=0, fp=0, fn=m["tp"] + m["fn"], precision=0.0, recall=0.0, f1=0.0)
    if "by_relation" in m:
        out["by_relation"] = {rl: dict(c, tp=0, fp=0, fn=c["tp"] + c["fn"])
                              for rl, c in m["by_relation"].items()}
    return out


def failed_zero(row):
    """The failed = 0 variant of a row (CSR flag): only for the sensitivity table."""
    if row["csr"]["compiled"]:
        return row
    return dict(row, element=_zero(row["element"]), relationship=_zero(row["relationship"]))


def zero_edge_keys(rows):
    """Keys whose ground truth has no edges: tp + fn = 0 in the Relationship F1 row."""
    return {r["key"] for r in rows if r["relationship"]["tp"] + r["relationship"]["fn"] == 0}


def joint(rows_z, rows_f, index):
    """Element / Relationship F1 micro of both runs over the diagrams of `index` that both
    compiled (one fixed set of diagrams per size)."""
    keep = [i for i in index if rows_z[i]["csr"]["compiled"] and rows_f[i]["csr"]["compiled"]]
    out = {"n_joint": len(keep)}
    for name, field in (("element_f1", "element"), ("relationship_f1", "relationship")):
        out[f"{name}|zs"] = micro_or_none([rows_z[i][field] for i in keep])
        out[f"{name}|ft"] = micro_or_none([rows_f[i][field] for i in keep])
    return out


# --------------------------------------------------------------------------- #
# intervals
# --------------------------------------------------------------------------- #
def interval(point, values):
    """95% type-7 percentile interval of `values`; a point only when the point or any resample
    is undefined."""
    undefined = sum(1 for v in values if v is None)
    if point is None or undefined:
        return {"point": point, "ci": None, "excludes_zero": None, "n_undefined": undefined,
                "why_point_only": "undefined point" if point is None else f"{undefined} undefined resample(s)"}
    v = sorted(values)
    lo, hi = percentile(v, CI_Q[0]), percentile(v, CI_Q[1])
    return {"point": point, "ci": [lo, hi], "excludes_zero": bool(lo > 0 or hi < 0), "n_undefined": 0}


def point_only(point, why):
    return {"point": point, "ci": None, "excludes_zero": None, "why_point_only": why}


def phase0_outcome(ci):
    """The Phase 0 reading of Δ_v2 − Δ_v1 on the joint set (Decision, CHANGES 2)."""
    if ci is None:
        return "undefined"
    lo, hi = ci
    if hi < 0:
        return "held"
    if lo > 0:
        return "did not hold"
    return "no detectable change"


PHASE0_TEXT = {"held": "held (v2 makes the delta more conservative)",
               "no detectable change": "no detectable change",
               "did not hold": "did not hold", "undefined": "undefined"}


# --------------------------------------------------------------------------- #
# the analysis
# --------------------------------------------------------------------------- #
def check_rows(v2, v1):
    """The six runs (both scorer versions) are row-aligned on the same keys, and every metric's
    compiled flag is CSR's."""
    keys = None
    for version, panel in (("v2", v2), ("v1", v1)):
        for mid, rows in panel.items():
            k = [r["key"] for r in rows]
            keys = keys or k
            if k != keys:
                raise ValueError(f"{version} {mid}: rows are not aligned with the other runs")
            for r in rows:
                c = r["csr"]["compiled"]
                if r["element"]["compiled"] != c or r["relationship"]["compiled"] != c or r["chrf"]["compiled"] != c:
                    raise ValueError(f"{version} {mid} {r['key']}: a metric's compiled flag is not CSR's")
    for size, (z, f) in PAIRS.items():
        if z not in v2 or f not in v2 or z not in v1 or f not in v1:
            raise ValueError(f"{size}: pair {z} / {f} is not in the panel")
    return keys


def analyse(v2, v1, n_resamples=N_RESAMPLES, seed=SEED, zero_edge_expected=ZERO_EDGE_EXPECTED):
    """Every number of the Decision. v2 / v1: {registry id: loader rows}, each run's rows in the
    loader's (sorted-key) order."""
    t0 = time.time()
    check_rows(v2, v1)
    n = len(next(iter(v2.values())))
    draws = make_resamples(n, n_resamples, seed)
    identity = list(range(n))
    any_rows = next(iter(v2.values()))
    tier_of = [r["tier"] for r in any_rows]
    type_of = [r["primary_type"] for r in any_rows]

    # the zero-edge set is a property of the ground truth: the same in every v2 run
    zsets = {mid: zero_edge_keys(rows) for mid, rows in v2.items()}
    zero_edge = next(iter(zsets.values()))
    if any(s != zero_edge for s in zsets.values()):
        raise ValueError("the zero-edge ground-truth set differs between runs")
    if zero_edge_expected is not None and len(zero_edge) != zero_edge_expected:
        raise ValueError(f"{len(zero_edge)} zero-edge diagrams, expected {zero_edge_expected}")
    keep_ne = [r["key"] not in zero_edge for r in any_rows]

    def bundle(mid):
        r2, r1 = v2[mid], v1[mid]
        masked = [failed_zero(r) for r in r2]

        def f(d):
            rs = [r2[i] for i in d]
            ne = [r["relationship"] for i, r in zip(d, rs) if keep_ne[i]]
            out = {"v2": stats(rs), "v1": stats([r1[i] for i in d]),
                   "failed0": stats([masked[i] for i in d]),
                   "tier": {t: stats([r for i, r in zip(d, rs) if tier_of[i] == t]) for t in TIERS}}
            out["v2"][NE["zeros_for_failed"]] = macro_or_none(ne)
            out["v2"][NE["compiled_only"]] = macro_or_none([x for x in ne if x["compiled"]])
            return out
        return f

    point, boot = {}, {}
    for mid in v2:
        f = bundle(mid)
        point[mid] = f(identity)
        boot[mid] = [f(d) for d in draws]
    jpoint, jboot = {}, {}
    for size, (z, f) in PAIRS.items():
        def fj(d, z=z, f=f):
            return {"v2": joint(v2[z], v2[f], d), "v1": joint(v1[z], v1[f], d)}
        jpoint[size], jboot[size] = fj(identity), [fj(d) for d in draws]
    seconds_boot = time.time() - t0

    def vals(mid, *path):
        out = []
        for b in boot[mid]:
            for p in path:
                b = b[p]
            out.append(b)
        return out

    def pt(mid, *path):
        b = point[mid]
        for p in path:
            b = b[p]
        return b

    def run_cell(mid, *path):
        return interval(pt(mid, *path), vals(mid, *path))

    def delta(size, *path):
        z, f = PAIRS[size]
        d = interval(sub(pt(f, *path), pt(z, *path)),
                     [sub(b, a) for a, b in zip(vals(z, *path), vals(f, *path))])
        d.update(zs=pt(z, *path), ft=pt(f, *path))
        return d

    def delta_vals(size, *path):
        z, f = PAIRS[size]
        return [sub(b, a) for a, b in zip(vals(z, *path), vals(f, *path))]

    res = {"meta": {"seed": seed, "n_resamples": n_resamples, "ci_q": list(CI_Q), "n_rows": n,
                    "method": "paired percentile bootstrap (type-7), one index draw applied to all six "
                              "runs and both scorer versions (analysis.bootstrap.make_resamples)",
                    "pairs": PAIRS, "full_set": "stored per-diagram rows as the frozen runners write them "
                                                "(Р.3's computation, no CSR mask)",
                    "zero_edge_diagrams": len(zero_edge)},
           "runs": {}}
    for size, (z, f) in PAIRS.items():
        for mid, cond in ((z, "zero_shot"), (f, "fine_tuned")):
            res["runs"][mid] = {"size": size, "condition": cond, "n_compiled": pt(mid, "v2", "n_compiled")}

    # ---- headline decomposition --------------------------------------------------------- #
    headline, identity_dev = {}, 0.0
    for size in SIZES:
        z, f = PAIRS[size]
        h = {"n_compiled_zs": pt(z, "v2", "n_compiled"), "n_compiled_ft": pt(f, "v2", "n_compiled"),
             "csr": delta(size, "v2", CSR)}
        for name, m in (("element_f1", EL), ("relationship_f1", REL)):
            h[name] = {part: delta(size, "v2", m[part]) for part in ("co", "gap", "full")}
            # identity: Δ full-set = Δ compiled-only − Δ gap, in the point and in every resample
            pairs = [(h[name]["full"]["point"], h[name]["co"]["point"], h[name]["gap"]["point"])]
            pairs += list(zip(delta_vals(size, "v2", m["full"]), delta_vals(size, "v2", m["co"]),
                              delta_vals(size, "v2", m["gap"])))
            for full, co, gap in pairs:
                if None not in (full, co, gap):
                    identity_dev = max(identity_dev, abs(full - (co - gap)))
        headline[size] = h
    res["headline"] = headline
    res["identity_max_abs_deviation"] = identity_dev

    # ---- differences of deltas (5C) ------------------------------------------------------ #
    dod_stats = {"csr": CSR, "element_f1_co": EL["co"], "element_f1_gap": EL["gap"],
                 "relationship_f1_co": REL["co"], "relationship_f1_gap": REL["gap"]}
    dod, dodvals = {}, {}
    for a, b in DOD_PAIRS:
        for name, st in dod_stats.items():
            va, vb = delta_vals(a, "v2", st), delta_vals(b, "v2", st)
            v = [sub(x, y) for x, y in zip(va, vb)]
            p = sub(headline_delta(headline, a, name), headline_delta(headline, b, name))
            dod[f"{a}-{b}|{name}"] = interval(p, v)
            dodvals[(a, b, name)] = (p, v)
    linear_dev = 0.0
    for name in dod_stats:
        p13, v13 = dodvals[("2B", "27B", name)]
        p12, v12 = dodvals[("2B", "9B", name)]
        p23, v23 = dodvals[("9B", "27B", name)]
        for x, y, w in [(p13, p12, p23)] + list(zip(v13, v12, v23)):
            if None not in (x, y, w):
                linear_dev = max(linear_dev, abs(x - (y + w)))
    res["dod"] = dod
    res["dod_linear_dependence_max_abs_deviation"] = linear_dev

    # ---- claim family (6A) ---------------------------------------------------------------- #
    family = []
    for size in SIZES:
        family.append({"member": f"{size} Δ CSR", **headline[size]["csr"]})
        family.append({"member": f"{size} Δ Element F1 micro, compiled-only", **headline[size]["element_f1"]["co"]})
    for a, b in DOD_PAIRS:
        family.append({"member": f"{a}−{b} difference of Δ CSR", **dod[f"{a}-{b}|csr"]})
        family.append({"member": f"{a}−{b} difference of Δ Element F1 micro, compiled-only",
                       **dod[f"{a}-{b}|element_f1_co"]})
    res["claim_family"] = family

    # ---- main table, v1 companion, descriptive deltas ------------------------------------- #
    main_stats = [CSR, EL["full"], EL["co"], REL["full"], REL["co"], "chrf|macro|zeros_for_failed",
                  "chrf|macro|compiled_only", TYPE_ACC]
    v1_stats = [EL["full"], EL["co"], REL["full"], REL["co"], TYPE_ACC]
    main, comp = {}, {}
    for mid in v2:
        by_type = {t: stats([r for r, ty in zip(v2[mid], type_of) if ty == t]) for t in TYPES}
        by_type_v1 = {t: stats([r for r, ty in zip(v1[mid], type_of) if ty == t]) for t in TYPES}
        tiers_v1 = {t: stats([r for r, ti in zip(v1[mid], tier_of) if ti == t]) for t in TIERS}
        main[mid] = {"overall": {st: run_cell(mid, "v2", st) for st in main_stats},
                     "n_compiled": pt(mid, "v2", "n_compiled"),
                     "by_type": {t: _point_cells(by_type[t], main_stats) for t in TYPES},
                     "by_tier": {t: _point_cells(pt(mid, "tier", t), main_stats) for t in TIERS}}
        comp[mid] = {"overall": {st: run_cell(mid, "v1", st) for st in v1_stats},
                     "by_type": {t: _point_cells(by_type_v1[t], v1_stats) for t in TYPES},
                     "by_tier": {t: _point_cells(tiers_v1[t], v1_stats) for t in TIERS}}
    res["main_table"] = main
    res["v1_companion"] = comp
    res["main_table_deltas_by_type"] = {
        size: {t: {st: sub(main[PAIRS[size][1]]["by_type"][t][st]["point"],
                           main[PAIRS[size][0]]["by_type"][t][st]["point"]) for st in main_stats}
               for t in TYPES} for size in SIZES}
    desc_stats = [EL["full"], REL["full"], "chrf|macro|zeros_for_failed", "chrf|macro|compiled_only",
                  "gap|chrf|macro", TYPE_ACC]
    res["descriptive_deltas"] = {size: {st: delta(size, "v2", st) for st in desc_stats} for size in SIZES}
    res["v1_deltas"] = {size: {st: delta(size, "v1", st) for st in v1_stats + [EL["gap"], REL["gap"]]}
                        for size in SIZES}

    # ---- joint set (4B), headroom (4C) -------------------------------------------------- #
    jres, headroom = {}, {}
    for size in SIZES:
        z, f = PAIRS[size]
        e = {"n_joint": jpoint[size]["v2"]["n_joint"], "n_compiled_zs": pt(z, "v2", "n_compiled"),
             "n_compiled_ft": pt(f, "v2", "n_compiled")}
        for name in ("element_f1", "relationship_f1"):
            p = sub(jpoint[size]["v2"][f"{name}|ft"], jpoint[size]["v2"][f"{name}|zs"])
            v = [sub(b["v2"][f"{name}|ft"], b["v2"][f"{name}|zs"]) for b in jboot[size]]
            e[name] = dict(interval(p, v), zs=jpoint[size]["v2"][f"{name}|zs"], ft=jpoint[size]["v2"][f"{name}|ft"])
        jres[size] = e
        headroom[size] = {}
        for name, m in (("element_f1", EL), ("relationship_f1", REL)):
            zco = pt(z, "v2", m["co"])
            d = headline[size][name]["co"]["point"]
            headroom[size][name] = {"zs_compiled_only": zco, "delta_compiled_only": d,
                                    "share_closed": (d / (1 - zco)) if None not in (d, zco) and zco < 1 else None}
    res["joint_set"] = jres
    res["headroom"] = headroom

    # ---- tiers (7T-B + 7T-D, CHANGES 4) --------------------------------------------------- #
    tier_full = {"csr": CSR, "element_f1_full": EL["full"], "relationship_f1_full": REL["full"]}
    tier_co = {"element_f1_co": EL["co"], "relationship_f1_co": REL["co"]}
    tiers, tier_vals = {}, {}
    for size in SIZES:
        z, f = PAIRS[size]
        tiers[size] = {}
        for t in TIERS:
            cell = {"n": pt(z, "tier", t, "n"), "n_compiled_zs": pt(z, "tier", t, "n_compiled"),
                    "n_compiled_ft": pt(f, "tier", t, "n_compiled")}
            for name, st in tier_full.items():
                cell[name] = delta(size, "tier", t, st)
                tier_vals[(size, t, name)] = (cell[name], delta_vals(size, "tier", t, st))
            floor_ok = min(cell["n_compiled_zs"], cell["n_compiled_ft"]) >= TIER_FLOOR
            for name, st in tier_co.items():
                d = delta(size, "tier", t, st)
                if not floor_ok:
                    d = dict(point_only(d["point"], f"fewer than {TIER_FLOOR} compiled diagrams in a run"),
                             zs=d["zs"], ft=d["ft"])
                cell[name] = d
                tier_vals[(size, t, name)] = (d, delta_vals(size, "tier", t, st))
            tiers[size][t] = cell
        t4t1 = {}
        for name in list(tier_full) + list(tier_co):
            c4, v4 = tier_vals[(size, 4, name)]
            c1, v1_ = tier_vals[(size, 1, name)]
            p = sub(c4["point"], c1["point"])
            if c4["ci"] is None or c1["ci"] is None:
                t4t1[name] = point_only(p, "a tier cell has a point only")
            else:
                t4t1[name] = interval(p, [sub(a, b) for a, b in zip(v4, v1_)])
        tiers[size]["tier4_minus_tier1"] = t4t1
    res["tiers"] = tiers

    # ---- per relation (descriptive, support floor) ---------------------------------------- #
    per_rel = {}
    for size in SIZES:
        z, f = PAIRS[size]
        per_rel[size] = {}
        for rel in RELATIONS:
            for pop in POPULATIONS:
                st = f"relationship_f1::{rel}|micro|{pop}"
                sup = {mid: relation_support(v2[mid], rel, pop) for mid in (z, f)}
                floor_ok = all(s["edges"] >= REL_FLOOR_EDGES and s["diagrams"] >= REL_FLOOR_DIAGRAMS
                               for s in sup.values())
                d = delta(size, "v2", st)
                rz, rf = run_cell(z, "v2", st), run_cell(f, "v2", st)
                if not floor_ok:
                    why = f"below the support floor ({REL_FLOOR_EDGES} edges, {REL_FLOOR_DIAGRAMS} diagrams)"
                    d = dict(point_only(d["point"], why), zs=d["zs"], ft=d["ft"])
                    rz, rf = point_only(rz["point"], why), point_only(rf["point"], why)
                per_rel[size][f"{rel}|{pop}"] = {"support_zs": sup[z], "support_ft": sup[f],
                                                 "zs": rz, "ft": rf, "delta": d}
    res["per_relation"] = per_rel

    # ---- Phase 0 (CHANGES 2) -------------------------------------------------------------- #
    phase0 = {}
    for size in SIZES:
        phase0[size] = {"n_joint": jpoint[size]["v2"]["n_joint"]}
        for name in ("element_f1", "relationship_f1"):
            def dj(b, version):
                return sub(b[version][f"{name}|ft"], b[version][f"{name}|zs"])
            p2, p1 = dj(jpoint[size], "v2"), dj(jpoint[size], "v1")
            d = interval(sub(p2, p1), [sub(dj(b, "v2"), dj(b, "v1")) for b in jboot[size]])
            d.update(delta_v2=p2, delta_v1=p1, outcome=phase0_outcome(d["ci"]))
            phase0[size][name] = d
        m = {"element_f1": EL, "relationship_f1": REL}
        phase0[size]["beside"] = {f"{name}|{part}": {"v2": delta(size, "v2", m[name][part]),
                                                     "v1": delta(size, "v1", m[name][part])}
                                  for name in m for part in ("full", "co")}
    res["phase0"] = phase0

    # ---- supplementary: macro, zero-edge exclusion ---------------------------------------- #
    macro_stats = ["element_f1|macro|zeros_for_failed", "element_f1|macro|compiled_only",
                   "relationship_f1|macro|zeros_for_failed", "relationship_f1|macro|compiled_only",
                   NE["zeros_for_failed"], NE["compiled_only"], "gap|element_f1|macro", "gap|relationship_f1|macro"]
    res["macro"] = {"runs": {mid: {st: run_cell(mid, "v2", st) for st in macro_stats} for mid in v2},
                    "deltas": {size: {st: delta(size, "v2", st) for st in macro_stats} for size in SIZES},
                    "zero_edge_diagrams": len(zero_edge)}

    # ---- supplementary: failed = 0 sensitivity -------------------------------------------- #
    f0_stats = [EL["full"], REL["full"], "element_f1|macro|zeros_for_failed",
                "relationship_f1|macro|zeros_for_failed", EL["gap"], REL["gap"],
                "gap|element_f1|macro", "gap|relationship_f1|macro"]
    f0_runs = {}
    for mid in v2:
        failed = [r for r in v2[mid] if not r["csr"]["compiled"]]
        f0_runs[mid] = {
            "stats": {st: {"stored": pt(mid, "v2", st), "failed0": pt(mid, "failed0", st)} for st in f0_stats},
            "n_failed": len(failed),
            "failed_with_credit": {field: {"tp_gt_0": sum(1 for r in failed if r[field]["tp"] > 0),
                                           "f1_eq_1": sum(1 for r in failed if r[field]["f1"] == 1.0)}
                                   for field in ("element", "relationship")}}
    res["failed0"] = {"runs": f0_runs,
                      "deltas": {size: {st: {"stored": delta(size, "v2", st), "failed0": delta(size, "failed0", st)}
                                        for st in f0_stats} for size in SIZES}}
    res["meta"]["seconds_bootstrap"] = round(seconds_boot, 1)
    res["meta"]["seconds_total"] = round(time.time() - t0, 1)
    return res


def headline_delta(headline, size, name):
    h = headline[size]
    return {"csr": h["csr"], "element_f1_co": h["element_f1"]["co"], "element_f1_gap": h["element_f1"]["gap"],
            "relationship_f1_co": h["relationship_f1"]["co"],
            "relationship_f1_gap": h["relationship_f1"]["gap"]}[name]["point"]


def _point_cells(s, ids):
    out = {st: {"point": s[st]} for st in ids}
    out.update(n={"point": s["n"]}, n_compiled={"point": s["n_compiled"]})
    return out


def relation_support(rows, rel, pop):
    """Ground-truth edges of one relation, and diagrams holding it, in a run's population."""
    pr = [r["relationship"] for r in rows if pop == "zeros_for_failed" or r["relationship"]["compiled"]]
    counts = [c["by_relation"][rel]["tp"] + c["by_relation"][rel]["fn"] for c in pr]
    return {"edges": sum(counts), "diagrams": sum(1 for c in counts if c > 0)}


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def load(root, registry=REGISTRY):
    """{version: {id: rows}} through the benchmark's loader and leak gate. Refuses a run that is
    not loaded, is refused by the leak gate, or whose predictions folder the gate cannot see."""
    entries = load_registry(registry)
    ids = [e.id for e in entries]
    conds = {e.id: e.condition for e in entries}
    for size, (z, f) in PAIRS.items():
        if conds.get(z) != "zero_shot" or conds.get(f) != "fine_tuned":
            raise SystemExit(f"{registry}: pair {size} is not {z} zero_shot / {f} fine_tuned")
    out = {}
    for version in ("v2", "v1"):
        r = Path(root) / version
        eligible, pending, refused = load_eligible_models(entries, r)
        if pending or refused:
            raise SystemExit(f"{r}: pending {pending}, refused by the leak gate {refused}")
        for e in entries:
            pdir = predictions_dir(r, e.run_dir)
            n = len(list(pdir.glob("*.puml"))) if pdir.is_dir() else 0
            if n != N_CELLS:
                raise SystemExit(f"{pdir}: {n} predictions; the leak gate would pass silently")
            if count_reasoning_leaks(pdir):  # pragma: no cover - load_eligible_models refused it already
                raise SystemExit(f"{pdir}: reasoning leaks")
        out[version] = {m.entry.id: m.rows for m in eligible}
        if list(out[version]) != ids:
            raise SystemExit(f"{r}: loaded {list(out[version])}, registry {ids}")
    return out


# --------------------------------------------------------------------------- #
# markdown
# --------------------------------------------------------------------------- #
def f4(x):
    return "—" if x is None else f"{x:.4f}"


def d4(x):
    return "—" if x is None else f"{x:+.4f}"


def ci_text(c, signed=True):
    fmt = d4 if signed else f4
    if c.get("ci") is None:
        return f"{fmt(c['point'])} (point only)"
    lo, hi = c["ci"]
    return f"{fmt(c['point'])} [{fmt(lo)}, {fmt(hi)}]"


def markdown(res, displays):
    L = []
    w = L.append
    m = res["meta"]
    w("# Р.4 main tables: fine-tuned against self-served zero-shot, per size\n")
    w(f"Generated by `build/main_tables.py`. Scorer v2 unless marked v1. Paired percentile bootstrap, seed "
      f"{m['seed']}, {m['n_resamples']} resamples, 95% type-7 interval; one draw applied to all six runs and both "
      f"scorer versions. Full-set = {m['full_set']}; compiled-only = each run's own compiled diagrams, so the two "
      "runs of a size are compared on two different sets of diagrams. Deltas are fine-tuned − zero-shot. "
      "\"(point only)\" marks a cell without an interval (an undefined resample, or below a support floor).\n")
    w("**Claim family** (12 intervals, nominal 95%, no correction): Δ CSR and Δ compiled-only Element F1 micro for "
      "each size, and their differences of deltas for 2B−9B, 9B−27B and 2B−27B. Every other number is descriptive.\n")

    w("## 1. Headline: the decomposition of each size's gain\n")
    w("Δ full-set = Δ compiled-only − Δ gap (identity; largest deviation over the point and every resample "
      f"{res['identity_max_abs_deviation']:.1e}).\n")
    w("| Size | compiled zs / ft | Δ CSR | Δ Element F1 micro, compiled-only | Δ gap Element F1 micro | Δ Element F1 micro, full-set |")
    w("|---|---|---|---|---|---|")
    for s in SIZES:
        h = res["headline"][s]
        w(f"| {s} | {h['n_compiled_zs']} / {h['n_compiled_ft']} | {ci_text(h['csr'])} | {ci_text(h['element_f1']['co'])} "
          f"| {ci_text(h['element_f1']['gap'])} | {ci_text(h['element_f1']['full'])} |")
    w("")
    w("Relationship F1 beside it (descriptive):\n")
    w("| Size | Δ Relationship F1 micro, compiled-only | Δ gap Relationship F1 micro | Δ Relationship F1 micro, full-set |")
    w("|---|---|---|---|")
    for s in SIZES:
        h = res["headline"][s]["relationship_f1"]
        w(f"| {s} | {ci_text(h['co'])} | {ci_text(h['gap'])} | {ci_text(h['full'])} |")
    w("")

    w("## 2. Claim family\n")
    w("| Member | Point | 95% interval | Excludes 0 |")
    w("|---|---|---|---|")
    for c in res["claim_family"]:
        ci = "—" if c["ci"] is None else f"[{d4(c['ci'][0])}, {d4(c['ci'][1])}]"
        w(f"| {c['member']} | {d4(c['point'])} | {ci} | {'yes' if c['excludes_zero'] else 'no'} |")
    w("")

    w("## 3. Differences of deltas\n")
    w("Same draw for every size. 2B−27B = 2B−9B + 9B−27B in the point and in every resample (largest deviation "
      f"{res['dod_linear_dependence_max_abs_deviation']:.1e}), so two of the three intervals are independent. "
      "Three sizes give a direction, not a fit. Δ CSR and Δ compiled-only Element F1 are claim members; the rest "
      "is descriptive.\n")
    w("| Pair | Δ CSR | Δ Element F1 micro, compiled-only | Δ gap Element F1 micro | Δ Relationship F1 micro, compiled-only | Δ gap Relationship F1 micro |")
    w("|---|---|---|---|---|---|")
    for a, b in DOD_PAIRS:
        k = f"{a}-{b}"
        w(f"| {a}−{b} | {ci_text(res['dod'][k + '|csr'])} | {ci_text(res['dod'][k + '|element_f1_co'])} | "
          f"{ci_text(res['dod'][k + '|element_f1_gap'])} | {ci_text(res['dod'][k + '|relationship_f1_co'])} | "
          f"{ci_text(res['dod'][k + '|relationship_f1_gap'])} |")
    w("")

    main_cols = [CSR, EL["full"], EL["co"], REL["full"], REL["co"], "chrf|macro|zeros_for_failed",
                 "chrf|macro|compiled_only", TYPE_ACC]
    short = {CSR: "CSR", EL["full"]: "El F1 full", EL["co"]: "El F1 co", REL["full"]: "Rel F1 full",
             REL["co"]: "Rel F1 co", "chrf|macro|zeros_for_failed": "chrF++ full",
             "chrf|macro|compiled_only": "chrF++ co", TYPE_ACC: "type acc."}

    def table(block, cols, title, note):
        w(f"## {title}\n")
        w(note + "\n")
        w("| Run | compiled | " + " | ".join(short[c] for c in cols) + " |")
        w("|---|---|" + "---|" * len(cols))
        for mid, cell in block.items():
            nc = res["runs"][mid]["n_compiled"]
            w(f"| {displays[mid]} | {nc} | " + " | ".join(ci_text(cell["overall"][c], signed=False) for c in cols) + " |")
        w("")
        for scope, keys in (("by_type", TYPES), ("by_tier", TIERS)):
            w(f"{'By type' if scope == 'by_type' else 'By tier'} (points; n per cell "
              f"{'500' if scope == 'by_type' else '250'}; compiled count in brackets):\n")
            w("| Statistic: run | " + " | ".join(f"{scope[3:]} {k}" for k in keys) + " |")
            w("|---|" + "---|" * len(keys))
            for c in cols:
                for mid, cell in block.items():
                    vals_ = []
                    for k in keys:
                        x = cell[scope][k]
                        vals_.append(f"{f4(x[c]['point'])} ({x['n_compiled']['point']})")
                    w(f"| {short[c]}: {displays[mid]} | " + " | ".join(vals_) + " |")
            w("")

    table(res["main_table"], main_cols, "4. Main table (v2)",
          "Overall: each run's value with its own 95% interval. chrF++ is macro (0–100). Type accuracy is over "
          "compiled answers. Margins by type and by tier are points.")
    table(res["v1_companion"], [EL["full"], EL["co"], REL["full"], REL["co"], TYPE_ACC],
          "5. v1 companion (descriptive)", "The same runs under scorer v1 (route (i): v2 code and JAR with the seven "
          "rule switches). CSR and chrF++ do not depend on the scorer version.")

    w("## 6. Other deltas (descriptive)\n")
    dcols = [EL["full"], REL["full"], "chrf|macro|zeros_for_failed", "chrf|macro|compiled_only", "gap|chrf|macro", TYPE_ACC]
    w("| Size | " + " | ".join(LABEL[c] for c in dcols) + " |")
    w("|---|" + "---|" * len(dcols))
    for s in SIZES:
        w(f"| {s} | " + " | ".join(ci_text(res["descriptive_deltas"][s][c]) for c in dcols) + " |")
    w("")
    w("By type, points (Δ, fine-tuned − zero-shot):\n")
    w("| Size | Type | " + " | ".join(short[c] for c in main_cols) + " |")
    w("|---|---|" + "---|" * len(main_cols))
    for s in SIZES:
        for t in TYPES:
            w(f"| {s} | {t} | " + " | ".join(d4(res["main_table_deltas_by_type"][s][t][c]) for c in main_cols) + " |")
    w("")

    w("## 7. Sensitivity: the diagrams both runs compiled (joint set; descriptive)\n")
    w("One fixed set of diagrams per size. It is selected too: it holds the cells the weaker run compiled.\n")
    w("| Size | joint n | compiled zs / ft | Element F1 micro zs → ft | Δ Element F1 micro | Relationship F1 micro zs → ft | Δ Relationship F1 micro |")
    w("|---|---|---|---|---|---|---|")
    for s in SIZES:
        j = res["joint_set"][s]
        w(f"| {s} | {j['n_joint']} | {j['n_compiled_zs']} / {j['n_compiled_ft']} | {f4(j['element_f1']['zs'])} → "
          f"{f4(j['element_f1']['ft'])} | {ci_text(j['element_f1'])} | {f4(j['relationship_f1']['zs'])} → "
          f"{f4(j['relationship_f1']['ft'])} | {ci_text(j['relationship_f1'])} |")
    w("")

    w("## 8. Share of remaining headroom closed (point, descriptive)\n")
    w("Δ compiled-only / (1 − zero-shot compiled-only).\n")
    w("| Size | Element F1: zs compiled-only | Δ | share closed | Relationship F1: zs compiled-only | Δ | share closed |")
    w("|---|---|---|---|---|---|---|")
    for s in SIZES:
        e, r = res["headroom"][s]["element_f1"], res["headroom"][s]["relationship_f1"]
        w(f"| {s} | {f4(e['zs_compiled_only'])} | {d4(e['delta_compiled_only'])} | {f4(e['share_closed'])} | "
          f"{f4(r['zs_compiled_only'])} | {d4(r['delta_compiled_only'])} | {f4(r['share_closed'])} |")
    w("")

    w("## 9. Gains by tier (descriptive)\n")
    w(f"Intervals from the global draw filtered to the tier. Compiled-only cells get an interval only when both runs "
      f"have at least {TIER_FLOOR} compiled diagrams in the tier; a statistic on an empty population is undefined. "
      "Δ(tier 4) − Δ(tier 1) is read as \"consistent with\" an input limit, never \"shows\".\n")
    tcols = ["csr", "element_f1_full", "relationship_f1_full", "element_f1_co", "relationship_f1_co"]
    tname = {"csr": "Δ CSR", "element_f1_full": "Δ El F1 full", "relationship_f1_full": "Δ Rel F1 full",
             "element_f1_co": "Δ El F1 co", "relationship_f1_co": "Δ Rel F1 co"}
    w("| Size | Tier | compiled zs / ft | " + " | ".join(tname[c] for c in tcols) + " |")
    w("|---|---|---|" + "---|" * len(tcols))
    for s in SIZES:
        for t in TIERS:
            c = res["tiers"][s][t]
            w(f"| {s} | {t} | {c['n_compiled_zs']} / {c['n_compiled_ft']} | " + " | ".join(ci_text(c[k]) for k in tcols) + " |")
        c = res["tiers"][s]["tier4_minus_tier1"]
        w(f"| {s} | 4 − 1 | | " + " | ".join(ci_text(c[k]) for k in tcols) + " |")
    w("")

    w("## 10. Per-relation Relationship F1 (descriptive)\n")
    w(f"Micro. An interval only with at least {REL_FLOOR_EDGES} ground-truth edges and {REL_FLOOR_DIAGRAMS} diagrams "
      "holding the relation in both runs' population. An edge matches only when both endpoint names match too, so a "
      "rise does not isolate decoration reading; no decoration verdict is drawn.\n")
    w("| Size | Relation | Population | GT edges / diagrams zs; ft | zero-shot | fine-tuned | Δ |")
    w("|---|---|---|---|---|---|---|")
    for s in SIZES:
        for rel in RELATIONS:
            for pop in POPULATIONS[::-1]:
                c = res["per_relation"][s][f"{rel}|{pop}"]
                sz, sf = c["support_zs"], c["support_ft"]
                w(f"| {s} | {rel} | {'full-set' if pop == 'zeros_for_failed' else 'compiled-only'} | "
                  f"{sz['edges']} / {sz['diagrams']}; {sf['edges']} / {sf['diagrams']} | {ci_text(c['zs'], False)} | "
                  f"{ci_text(c['ft'], False)} | {ci_text(c['delta'])} |")
    w("")

    w("## 11. Phase 0: does v2 make the delta more conservative?\n")
    w("Read on the joint set: Δ_v2 − Δ_v1. Interval below 0: held; contains 0: no detectable change; above 0: did "
      "not hold.\n")
    w("| Size | joint n | Metric | Δ_v2 | Δ_v1 | Δ_v2 − Δ_v1 | Outcome |")
    w("|---|---|---|---|---|---|---|")
    for s in SIZES:
        for name in ("element_f1", "relationship_f1"):
            c = res["phase0"][s][name]
            w(f"| {s} | {res['phase0'][s]['n_joint']} | {'Element F1 micro' if name == 'element_f1' else 'Relationship F1 micro'} "
              f"| {d4(c['delta_v2'])} | {d4(c['delta_v1'])} | {ci_text(c)} | {PHASE0_TEXT[c['outcome']]} |")
    w("")
    w("Beside it, without a verdict: full-set and compiled-only deltas under each scorer.\n")
    w("| Size | Statistic | Δ under v2 | Δ under v1 |")
    w("|---|---|---|---|")
    for s in SIZES:
        for k, c in res["phase0"][s]["beside"].items():
            name, part = k.split("|")
            lab = ("Element F1 micro" if name == "element_f1" else "Relationship F1 micro") + \
                  (", full-set" if part == "full" else ", compiled-only")
            w(f"| {s} | {lab} | {ci_text(c['v2'])} | {ci_text(c['v1'])} |")
    w("")

    w("## S1. Supplementary: macro F1 (descriptive)\n")
    w(f"Beside macro Relationship F1, the same figure with the {res['macro']['zero_edge_diagrams']} diagrams whose "
      "v2 ground truth has no edges left out (an answer without edges scores 1 on them).\n")
    mcols = list(next(iter(res["macro"]["runs"].values())))
    w("| Run | " + " | ".join(LABEL[c] for c in mcols) + " |")
    w("|---|" + "---|" * len(mcols))
    for mid, cell in res["macro"]["runs"].items():
        w(f"| {displays[mid]} | " + " | ".join(ci_text(cell[c], False) for c in mcols) + " |")
    for s in SIZES:
        w(f"| Δ {s} | " + " | ".join(ci_text(res["macro"]["deltas"][s][c]) for c in mcols) + " |")
    w("")

    w("## S2. Supplementary: a non-compiling answer scored 0 (sensitivity; descriptive)\n")
    w("Stored = the frozen runners' rows (the numbers above); failed = 0 = every non-compiling answer forced to "
      "tp = fp = 0, fn = |GT|, per-diagram F1 = 0 by the CSR flag. The headline is not changed by it.\n")
    fcols = list(next(iter(res["failed0"]["runs"].values()))["stats"])
    w("| Run | failed | failed with tp > 0 (El / Rel) | failed with F1 = 1 (El / Rel) | " +
      " | ".join(f"{LABEL[c]}: stored → failed = 0" for c in fcols) + " |")
    w("|---|---|---|---|" + "---|" * len(fcols))
    for mid, r in res["failed0"]["runs"].items():
        fc = r["failed_with_credit"]
        w(f"| {displays[mid]} | {r['n_failed']} | {fc['element']['tp_gt_0']} / {fc['relationship']['tp_gt_0']} | "
          f"{fc['element']['f1_eq_1']} / {fc['relationship']['f1_eq_1']} | " +
          " | ".join(f"{f4(r['stats'][c]['stored'])} → {f4(r['stats'][c]['failed0'])}" for c in fcols) + " |")
    w("")
    w("| Size | Statistic | Δ stored | Δ failed = 0 |")
    w("|---|---|---|---|")
    for s in SIZES:
        for c in fcols:
            x = res["failed0"]["deltas"][s][c]
            w(f"| {s} | {LABEL[c]} | {ci_text(x['stored'])} | {ci_text(x['failed0'])} |")
    w("")
    return "\n".join(L)


# --------------------------------------------------------------------------- #
# figure
# --------------------------------------------------------------------------- #
def figure(res, path):
    """Before and after per size, with each run's 95% interval: CSR, Element F1 micro full-set and
    compiled-only, Relationship F1 micro compiled-only."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = [(CSR, "CSR"), (EL["full"], "Element F1 micro, full-set"),
              (EL["co"], "Element F1 micro, compiled-only"), (REL["co"], "Relationship F1 micro, compiled-only")]
    colours = {"zero_shot": "#2a78d6", "fine_tuned": "#eb6834"}     # dataviz reference slots 1, 2
    markers = {"zero_shot": "o", "fine_tuned": "s"}
    names = {"zero_shot": "zero-shot", "fine_tuned": "fine-tuned (20k)"}
    fig, axes = plt.subplots(1, len(panels), figsize=(13, 3.6), sharey=True)
    for ax, (st, title) in zip(axes, panels):
        for k, cond in enumerate(("zero_shot", "fine_tuned")):
            xs, ys, lo, hi = [], [], [], []
            for i, size in enumerate(SIZES):
                mid = PAIRS[size][k]
                c = res["main_table"][mid]["overall"][st]
                xs.append(i + (-0.09 if k == 0 else 0.09))
                ys.append(c["point"])
                lo.append(c["point"] - c["ci"][0] if c["ci"] else 0)
                hi.append(c["ci"][1] - c["point"] if c["ci"] else 0)
            ax.errorbar(xs, ys, yerr=[lo, hi], fmt=markers[cond], color=colours[cond], ms=7, lw=2, capsize=3,
                        label=names[cond], zorder=3)
        for i, size in enumerate(SIZES):
            z, f = PAIRS[size]
            ax.plot([i - 0.09, i + 0.09], [res["main_table"][z]["overall"][st]["point"],
                                           res["main_table"][f]["overall"][st]["point"]],
                    color="#b8b7b2", lw=1, zorder=1)
        ax.set_xticks(range(len(SIZES)))
        ax.set_xticklabels(SIZES)
        ax.set_title(title, fontsize=10, color="#0b0b0b")
        ax.set_ylim(0, 1.02)
        ax.grid(axis="y", color="#e6e5e0", lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors="#52514e")
    axes[0].legend(frameon=False, fontsize=9, loc="lower right")
    fig.text(0.5, -0.02, "Qwen3.5 size; each run's value with its bootstrap 95% interval (seed 20260614, 1,000 resamples, "
             "one draw for all runs), scorer v2", ha="center", fontsize=8, color="#52514e")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="#fcfcfb")
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=str(ROOT), help="data roots written by build/score_test_runs.py")
    ap.add_argument("--registry", default=str(REGISTRY))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)
    out = Path(args.out).resolve()
    if (INSTRUMENT / "analysis" / "out").resolve() in [out, *out.parents]:
        raise SystemExit("refused: never write into the benchmark's analysis/out")
    panel = load(args.root, args.registry)
    res = analyse(panel["v2"], panel["v1"])
    displays = {e.id: e.display for e in load_registry(args.registry)}
    res["meta"].update(root=str(Path(args.root).resolve()), registry=str(Path(args.registry).resolve()),
                       main_tables_sha256=_sha256(__file__))
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "main_tables.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    (out / "main_tables.md").write_text(markdown(res, displays), encoding="utf-8")
    figure(res, out / "before_after_by_size.png")
    print(f"written: {out / 'main_tables.json'}, main_tables.md, before_after_by_size.png "
          f"({res['meta']['seconds_total']} s)", flush=True)
    return res


def _sha256(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
