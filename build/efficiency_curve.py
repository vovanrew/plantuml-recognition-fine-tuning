#!/usr/bin/env python3
"""The 2B data-efficiency curve on the test set: zero-shot, and fine-tuned on 1,000, 5,000 and
20,000 pairs, every score shown with its paired interval.

decisions/efficiency-curve-analysis/record.md, Decision. Every score is shown; no statistic is
fixed in advance to decide whether the gain still rises, and the reading is written after the
scores are seen. Every number here is descriptive: none joins the claim family of the main
comparison (methodology/analysis-protocol.md §9), and each interval is read on its own at
nominal 95%.

What is computed, under scorer v2 unless marked:
  - each point: CSR; Element F1 and Relationship F1 micro, full-set and compiled-only, and their
    population gaps; chrF++ macro in both populations; type accuracy;
  - each point under scorer v1: the same F1 figures;
  - each point with every non-compiling answer scored 0 (the main comparison's sensitivity,
    analysis-protocol.md §4);
  - the differences between neighbouring points (1k - 0, 5k - 1k, 20k - 5k) on the v2 statistics;
  - 20k - 5k on Element F1 and Relationship F1 micro over the diagrams both runs compiled (the
    joint set, analysis-protocol.md §7).

The curve's one variable is the number of distinct pairs the recipe trains on. The recipe trains
two epochs, so the optimizer steps grow with the pairs (train/recipe.py, total_steps).

Everything statistical is build/main_tables.py's, imported unchanged: the loader and leak gate
(load), the row checks (check_rows), the statistics (stats, failed_zero, joint) and the interval
(interval), with the benchmark's seed 20260614, 1,000 resamples and type-7 95% interval. The draw
depends only on the number of diagrams, the resamples and the seed, so it is the main runs' draw.
Before writing, the 0 and 20k points are checked cell by cell against the published main tables
(runs/test/_analysis/main_tables.json): any difference is refused.

Usage (from fine-tune/):
  python3 build/efficiency_curve.py            # writes runs/test/_analysis/efficiency_curve.{json,md}
Tests: build/tests/test_efficiency_curve.py
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
FT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(FT / "train"))
import main_tables as M  # noqa: E402
import recipe  # noqa: E402
from analysis.bootstrap import CI_Q, N_RESAMPLES, SEED, make_resamples  # noqa: E402

REGISTRY = HERE / "curve_registry.json"
PUBLISHED = M.OUT / "main_tables.json"
OUT = M.OUT

# The curve's four points: label, registry id, distinct training pairs (0 = zero-shot).
POINTS = [("0", "qwen3.5-2b-zs", 0), ("1k", "qwen3.5-2b-ft1k", 1000),
          ("5k", "qwen3.5-2b-ft5k", 5000), ("20k", "qwen3.5-2b-ft20k", 20000)]
NEIGHBOURS = [("0", "1k"), ("1k", "5k"), ("5k", "20k")]
JOINT = ("5k", "20k")

V2_STATS = [M.CSR, M.EL["full"], M.EL["co"], M.EL["gap"], M.REL["full"], M.REL["co"], M.REL["gap"],
            "chrf|macro|zeros_for_failed", "chrf|macro|compiled_only", M.TYPE_ACC]
V1_STATS = [M.EL["full"], M.EL["co"], M.EL["gap"], M.REL["full"], M.REL["co"], M.REL["gap"]]
F0_STATS = [M.EL["full"], M.REL["full"], "element_f1|macro|zeros_for_failed",
            "relationship_f1|macro|zeros_for_failed", M.EL["gap"], M.REL["gap"],
            "gap|element_f1|macro", "gap|relationship_f1|macro"]
# the main comparison's headline Δ of the 2B (20k - 0), recomputed here only for the check
HEADLINE_PARTS = {"csr": M.CSR,
                  "element_f1": {"co": M.EL["co"], "gap": M.EL["gap"], "full": M.EL["full"]},
                  "relationship_f1": {"co": M.REL["co"], "gap": M.REL["gap"], "full": M.REL["full"]}}


def _cell(point, values):
    """main_tables.interval without its excludes_zero flag: the curve fixes no verdict."""
    c = M.interval(point, values)
    c.pop("excludes_zero", None)
    return c


def steps_of(pairs):
    return recipe.total_steps(pairs) if pairs else 0


def curve(v2, v1, n_resamples=N_RESAMPLES, seed=SEED, zero_edge_expected=M.ZERO_EDGE_EXPECTED):
    """Every number of the curve. v2 / v1: {registry id: loader rows}, every run's rows in the
    loader's (sorted-key) order; the four POINTS must be among them."""
    t0 = time.time()
    M.check_rows(v2, v1)
    ids = {label: mid for label, mid, _ in POINTS}
    missing = [mid for mid in ids.values() if mid not in v2 or mid not in v1]
    if missing:
        raise ValueError(f"curve points not in the panel: {missing}")
    zsets = {mid: M.zero_edge_keys(rows) for mid, rows in v2.items()}
    zero_edge = next(iter(zsets.values()))
    if any(s != zero_edge for s in zsets.values()):
        raise ValueError("the zero-edge ground-truth set differs between runs")
    if zero_edge_expected is not None and len(zero_edge) != zero_edge_expected:
        raise ValueError(f"{len(zero_edge)} zero-edge diagrams, expected {zero_edge_expected}")

    n = len(v2[ids["0"]])
    draws = make_resamples(n, n_resamples, seed)
    identity = list(range(n))

    def bundle(mid):
        r2, r1 = v2[mid], v1[mid]
        masked = [M.failed_zero(r) for r in r2]
        return lambda d: {"v2": M.stats([r2[i] for i in d]), "v1": M.stats([r1[i] for i in d]),
                          "failed0": M.stats([masked[i] for i in d])}

    point, boot = {}, {}
    for label, mid in ids.items():
        f = bundle(mid)
        point[label], boot[label] = f(identity), [f(d) for d in draws]
    a, b = JOINT

    def fj(d):
        return M.joint(v2[ids[a]], v2[ids[b]], d)
    jpoint, jboot = fj(identity), [fj(d) for d in draws]

    def cell(label, version, st):
        return _cell(point[label][version][st], [x[version][st] for x in boot[label]])

    def diff(lo, hi, version, st):
        d = _cell(M.sub(point[hi][version][st], point[lo][version][st]),
                       [M.sub(y[version][st], x[version][st]) for x, y in zip(boot[lo], boot[hi])])
        d.update({lo: point[lo][version][st], hi: point[hi][version][st]})
        return d

    res = {"meta": {"seed": seed, "n_resamples": n_resamples, "ci_q": list(CI_Q), "n_rows": n,
                    "method": "paired percentile bootstrap (type-7); one index draw applied to the four "
                              "points and to every scorer version (analysis.bootstrap.make_resamples), "
                              "the same draw as the main comparison's",
                    "reading": "descriptive: no statistic decides in advance; not in the claim family of "
                               "analysis-protocol.md §9; each interval read on its own at nominal 95%",
                    "variable": "distinct training pairs under the recipe (two epochs); the optimizer "
                                "steps grow with the pairs",
                    "full_set": "stored per-diagram rows as the frozen runners write them (Р.3's "
                                "computation, no CSR mask)",
                    "zero_edge_diagrams": len(zero_edge)},
           "points": {}, "differences": {}}
    for label, mid, pairs in POINTS:
        res["points"][label] = {
            "id": mid, "pairs": pairs, "steps": steps_of(pairs),
            "n_compiled": point[label]["v2"]["n_compiled"],
            "v2": {st: cell(label, "v2", st) for st in V2_STATS},
            "v1": {st: cell(label, "v1", st) for st in V1_STATS},
            "failed0": {st: {"stored": cell(label, "v2", st), "failed0": cell(label, "failed0", st)}
                        for st in F0_STATS}}
    for lo, hi in NEIGHBOURS:
        res["differences"][f"{hi}-{lo}"] = {st: diff(lo, hi, "v2", st) for st in V2_STATS}
    j = {"pair": f"{b}-{a}", "n_joint": jpoint["n_joint"],
         "n_compiled": {a: point[a]["v2"]["n_compiled"], b: point[b]["v2"]["n_compiled"]}}
    for name in ("element_f1", "relationship_f1"):
        p = M.sub(jpoint[f"{name}|ft"], jpoint[f"{name}|zs"])
        v = [M.sub(x[f"{name}|ft"], x[f"{name}|zs"]) for x in jboot]
        j[name] = dict(_cell(p, v), **{a: jpoint[f"{name}|zs"], b: jpoint[f"{name}|ft"]})
    res["joint_set"] = j

    # the 2B headline Δ (20k - 0), only to check against the published main tables
    check = {"csr": diff("0", "20k", "v2", M.CSR)}
    for name in ("element_f1", "relationship_f1"):
        check[name] = {part: diff("0", "20k", "v2", st) for part, st in HEADLINE_PARTS[name].items()}
    res["_headline_2B"] = check
    res["meta"]["seconds_total"] = round(time.time() - t0, 1)
    return res


def _same(ours, published):
    """A cell's point and interval, exactly (the curve's cells carry no excludes_zero flag)."""
    return ours["point"] == published["point"] and ours.get("ci") == published.get("ci")


def check_published(res, published):
    """The 0 and 20k points equal the published main tables, cell by cell; refuses otherwise.
    Returns the number of cells compared."""
    bad, n = [], 0
    for label in ("0", "20k"):
        p = res["points"][label]
        mid = p["id"]
        main, comp = published["main_table"][mid], published["v1_companion"][mid]
        if p["n_compiled"] != main["n_compiled"]:
            bad.append(f"{label} n_compiled {p['n_compiled']} != {main['n_compiled']}")
        for st in (x for x in V2_STATS if x in main["overall"]):
            n += 1
            if not _same(p["v2"][st], main["overall"][st]):
                bad.append(f"{label} v2 {st}")
        for st in (x for x in V1_STATS if x in comp["overall"]):
            n += 1
            if not _same(p["v1"][st], comp["overall"][st]):
                bad.append(f"{label} v1 {st}")
        for st, c in published["failed0"]["runs"][mid]["stats"].items():
            n += 2
            if p["failed0"][st]["failed0"]["point"] != c["failed0"]:
                bad.append(f"{label} failed0 {st}")
            if p["failed0"][st]["stored"]["point"] != c["stored"]:
                bad.append(f"{label} stored {st}")
    head = published["headline"]["2B"]
    ours = res["_headline_2B"]
    n += 1
    if not _same(ours["csr"], head["csr"]):
        bad.append("headline 2B csr")
    for name in ("element_f1", "relationship_f1"):
        for part in ("co", "gap", "full"):
            n += 1
            if not _same(ours[name][part], head[name][part]):
                bad.append(f"headline 2B {name} {part}")
    if bad:
        raise SystemExit("the 0 and 20k points differ from the published main tables: " + "; ".join(bad[:10]))
    return n


# --------------------------------------------------------------------------- #
# markdown
# --------------------------------------------------------------------------- #
def markdown(res):
    L = []
    w = L.append
    m = res["meta"]
    pts = res["points"]
    labels = [label for label, _, _ in POINTS]
    w("# The 2B data-efficiency curve on the test set\n")
    w(f"Generated by `build/efficiency_curve.py`. Scorer v2 unless marked v1. Paired percentile bootstrap, seed "
      f"{m['seed']}, {m['n_resamples']} resamples, 95% type-7 interval; one draw applied to the four points and "
      "both scorer versions, the main comparison's draw. Every number is descriptive: none is in the claim "
      "family of the main comparison, and each interval is read on its own. Full-set = stored per-diagram "
      "rows (Р.3's computation); compiled-only = each run's own compiled diagrams, so two points are compared "
      "on two different sets of diagrams. \"(point only)\" marks a cell without an interval.\n")
    w("The variable is the number of distinct training pairs under the recipe, which trains two epochs: the "
      "optimizer steps grow with the pairs ("
      + ", ".join(f"{pts[x]['pairs']:,} pairs: {pts[x]['steps']:,} steps" for x in labels[1:]) + ").\n")

    w("## Points\n")
    w("| Statistic | " + " | ".join(f"{x} pairs" for x in labels) + " |")
    w("|---|" + "---|" * len(labels))
    w("| compiled diagrams | " + " | ".join(str(pts[x]["n_compiled"]) for x in labels) + " |")
    for st in V2_STATS:
        w(f"| {M.LABEL[st]} | " + " | ".join(M.ci_text(pts[x]["v2"][st], signed=st.startswith("gap"))
                                               for x in labels) + " |")
    w("")

    w("## Differences between neighbouring points\n")
    keys = list(res["differences"])
    w("| Statistic | " + " | ".join(keys) + " |")
    w("|---|" + "---|" * len(keys))
    for st in V2_STATS:
        w(f"| {M.LABEL[st]} | " + " | ".join(M.ci_text(res["differences"][k][st]) for k in keys) + " |")
    w("")

    j = res["joint_set"]
    w(f"## {j['pair']} on the joint set\n")
    a, b = JOINT
    w(f"Diagrams both runs compiled: {j['n_joint']} ({a}: {j['n_compiled'][a]} compiled, "
      f"{b}: {j['n_compiled'][b]}).\n")
    w(f"| Statistic | {a} | {b} | {j['pair']} |")
    w("|---|---|---|---|")
    for name, label in (("element_f1", "Element F1 micro"), ("relationship_f1", "Relationship F1 micro")):
        c = j[name]
        w(f"| {label} | {M.f4(c[a])} | {M.f4(c[b])} | {M.ci_text(c)} |")
    w("")

    w("## Scorer v1\n")
    w("| Statistic | " + " | ".join(f"{x} pairs" for x in labels) + " |")
    w("|---|" + "---|" * len(labels))
    for st in V1_STATS:
        w(f"| {M.LABEL[st]} | " + " | ".join(M.ci_text(pts[x]["v1"][st], signed=st.startswith("gap"))
                                               for x in labels) + " |")
    w("")

    w("## Non-compiling answers scored 0 (sensitivity)\n")
    w("Stored → failed = 0.\n")
    w("| Statistic | " + " | ".join(f"{x} pairs" for x in labels) + " |")
    w("|---|" + "---|" * len(labels))
    for st in F0_STATS:
        w(f"| {M.LABEL[st]} | " + " | ".join(f"{M.f4(pts[x]['failed0'][st]['stored']['point'])} → "
                                               f"{M.f4(pts[x]['failed0'][st]['failed0']['point'])}"
                                               for x in labels) + " |")
    w("")
    return "\n".join(L)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=str(M.ROOT))
    ap.add_argument("--registry", default=str(REGISTRY))
    ap.add_argument("--published", default=str(PUBLISHED))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)
    out = Path(args.out).resolve()
    if (M.INSTRUMENT / "analysis" / "out").resolve() in [out, *out.parents]:
        raise SystemExit("refused: never write into the benchmark's analysis/out")
    published = json.loads(Path(args.published).read_text())
    panel = M.load(args.root, args.registry)
    res = curve(panel["v2"], panel["v1"])
    n_checked = check_published(res, published)
    del res["_headline_2B"]
    res["meta"].update(root=str(Path(args.root).resolve()), registry=str(Path(args.registry).resolve()),
                       cells_checked_against_published=n_checked,
                       efficiency_curve_sha256=sha256(__file__), main_tables_sha256=sha256(M.__file__),
                       registry_sha256=sha256(args.registry), published_sha256=sha256(args.published))
    out.mkdir(parents=True, exist_ok=True)
    (out / "efficiency_curve.json").write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n")
    (out / "efficiency_curve.md").write_text(markdown(res))
    print(f"wrote {out}/efficiency_curve.{{json,md}}; {n_checked} published cells equal "
          f"({res['meta']['seconds_total']} s)")


if __name__ == "__main__":
    main()
