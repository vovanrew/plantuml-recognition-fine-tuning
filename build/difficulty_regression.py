#!/usr/bin/env python3
"""The difficulty regression: which properties of a test diagram predict compile failure and the
per-diagram Element F1 of the six main runs, before and after fine-tuning, and whether the data
separate the size of a diagram from the crowding of its picture.

decisions/difficulty-regression/record.md, Decision and Implementation notes. Scorer v2 only.

Models. One model per run and outcome part, the two diagram types pooled with common slopes:
  - Model A: type + z log content_lines + z log standardized area (the area of the 1,568 px input
    image; for most large diagrams, the aspect ratio);
  - Model B: type + z log content_lines + z log shrink (original area / standardized area, 1 for an
    image whose original long side is at most 1,568 px: how far the image was shrunk).
  z is standardized within type, with constants taken once from the 1,000 test diagrams.
Parts. Compile success (the v2 CSR flag) on all 1,000 diagrams of a run, by Firth-penalized
logistic regression; per-diagram v2 Element F1 on the run's own compiled diagrams, by fractional
logit (Bernoulli quasi-likelihood, logit link) fitted by IRLS without penalty. Both are fitted
by hand-written numpy (Newton-Raphson with step-halving).

Read quantity. For a property, every diagram of the run's population gets the 25th, then the
75th percentile of its own type's test-set values (other terms as observed); the predicted
probabilities are averaged over the population and 75th - 25th is taken. Percentiles are fixed
from the test set, not re-estimated in a resample. Coefficients are shown in a table; no
difference of coefficients is computed.

Intervals. The paired bootstrap of methodology/analysis-protocol.md §5, with the index draw of
build/main_tables.py (make_resamples, seed 20260614, 1,000 resamples, type-7 95%) applied to the
loader's rows (main_tables.load), imported unchanged. A compile model whose minority outcome in
the full run is below 25 gives points only (event floor); a resample whose fit does not converge
in the iteration limit makes the cells of that fit point-only (§5). The repository-level to
diagram-level resampling spread of the read crowding differences is stated per run, not used.

Reading. Descriptive and outside the claim family of 12 (§9). A read list of 12: per crowding
axis, the crowding difference in predicted compile rate after fine-tuning per size and its
fine-tuned - zero-shot difference. Per size and read quantity, a crowding verdict only where the
two axes' intervals lie on the same side of 0; otherwise the fixed sentence "the data do not
separate size from crowding". The total number of intervals is stated.

Before writing, each run's CSR (point and interval) and compiled count are checked against the
published runs/test/_analysis/main_tables.json; any difference is refused.

Usage (from fine-tune/):
  python3 build/difficulty_regression.py   # writes runs/test/_analysis/difficulty_regression.{json,md}
Tests: build/tests/test_difficulty_regression.py
"""
import argparse
import collections
import hashlib
import json
import math
import random
import struct
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FT = HERE.parent
sys.path.insert(0, str(HERE))
import main_tables as M  # noqa: E402
from analysis.aggregate import csr_rate  # noqa: E402  (main_tables put the benchmark on the path)
from analysis.bootstrap import CI_Q, N_RESAMPLES, SEED  # noqa: E402

PUBLISHED = M.OUT / "main_tables.json"
ERROR_ANALYSIS = M.OUT / "error_analysis.json"
TEST_SET = M.INSTRUMENT / "data" / "test_set.json"
IMAGES = M.INSTRUMENT / "data" / "puml_images_1568"
OUT = M.OUT

LONG_SIDE = 1568            # the input standard: longest side of the standardized image
EVENT_FLOOR = 25            # minority outcome count of a compile model, below which points only
MAX_ITER = 100              # Newton iterations; a fit not converged within them is undefined
TOL = 1e-8                  # convergence: largest absolute Newton step below this
MAX_HALVINGS = 30
Q_LO, Q_HI = 0.25, 0.75

REFERENCE_TYPE = M.TYPES[0]                 # "class": the intercept
OTHER_TYPES = M.TYPES[1:]                   # ["sequence"]: one indicator each
MODELS = {
    "A": {"crowding": "std_area",
          "name": "image area at 1,568 px (for most large diagrams, the aspect ratio)",
          "column": "z log standardized area",
          "toward_crowding": -1,          # less area at the same size = more crowded
          "crowded_end": "25th percentile of standardized area"},
    "B": {"crowding": "shrink",
          "name": "how far the image was shrunk",
          "column": "z log shrink",
          "toward_crowding": +1,          # more shrink = smaller glyphs
          "crowded_end": "75th percentile of shrink"},
}
PARTS = {"compile": "compile success, all diagrams, Firth-penalized logistic",
         "element_f1": "per-diagram Element F1, the run's compiled diagrams, fractional logit (IRLS)"}
PROPS = ("size", "crowding")
SIZE_COLUMN = "z log content_lines"
NOT_SEPARATED = "the data do not separate size from crowding"
CEILING_SENTENCE = "a run with few failures cannot show a large difference in compile rate"


# --------------------------------------------------------------------------- #
# fitting
# --------------------------------------------------------------------------- #
def _sigmoid(eta):
    return 1.0 / (1.0 + np.exp(-np.clip(eta, -35.0, 35.0)))


def _objective(X, y, b, firth):
    """Bernoulli (quasi-)log-likelihood with the logit link, y in [0, 1]; with firth, plus
    0.5 log det of the Fisher information (Jeffreys prior)."""
    eta = X @ b
    ll = float(np.sum(y * eta - np.logaddexp(0.0, eta)))
    if firth:
        p = _sigmoid(eta)
        w = p * (1.0 - p)
        sign, logdet = np.linalg.slogdet(X.T @ (X * w[:, None]))
        if sign <= 0:
            return -math.inf
        ll += 0.5 * logdet
    return ll


def fit_logit(X, y, firth=False, max_iter=MAX_ITER, tol=TOL):
    """Newton-Raphson for the logit-link Bernoulli likelihood (y binary: logistic regression;
    y in [0, 1]: the fractional logit's quasi-likelihood, Papke & Wooldridge 1996). With firth,
    the score is Firth's (1993) modified score U* = X'(y - p + h (1/2 - p)), h the diagonal of
    the weighted hat matrix. A step that lowers the (penalized) objective is halved. Returns
    (beta, converged): converged when the largest absolute Newton step falls below tol within
    max_iter iterations and every estimate is finite."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    b = np.zeros(X.shape[1])
    obj = _objective(X, y, b, firth)
    for _ in range(max_iter):
        p = _sigmoid(X @ b)
        w = p * (1.0 - p)
        try:
            inv = np.linalg.inv(X.T @ (X * w[:, None]))
        except np.linalg.LinAlgError:
            return b, False
        r = y - p
        if firth:
            h = np.einsum("ij,jk,ik->i", X, inv, X) * w
            r = r + h * (0.5 - p)
        step = inv @ (X.T @ r)
        if not np.all(np.isfinite(step)):
            return b, False
        if np.max(np.abs(step)) < tol:
            b = b + step
            return b, bool(np.all(np.isfinite(b)))
        t = 1.0
        for _ in range(MAX_HALVINGS):
            nb = b + t * step
            nobj = _objective(X, y, nb, firth)
            if nobj >= obj - 1e-10 * (1.0 + abs(obj)):
                break
            t /= 2.0
        b, obj = nb, nobj
    return b, False


def predicted_levels(beta, X, col, lo, hi):
    """Mean predicted probability over the rows of X with column `col` set to `lo`, then to
    `hi` (per-row arrays or scalars), every other term as observed; and hi - lo."""
    a = X.copy()
    a[:, col] = lo
    c = X.copy()
    c[:, col] = hi
    p_lo = float(np.mean(_sigmoid(a @ beta)))
    p_hi = float(np.mean(_sigmoid(c @ beta)))
    return {"p25": p_lo, "p75": p_hi, "diff": p_hi - p_lo}


# --------------------------------------------------------------------------- #
# predictors
# --------------------------------------------------------------------------- #
def png_size(path):
    """(width, height) from a PNG's IHDR chunk."""
    with open(path, "rb") as f:
        head = f.read(24)
    if head[:8] != b"\x89PNG\r\n\x1a\n" or head[12:16] != b"IHDR":
        raise ValueError(f"{path}: not a PNG")
    return struct.unpack(">II", head[16:24])


def predictor_table(test_set=TEST_SET, images=IMAGES):
    """{bare key: predictors} for every test diagram: type, content_lines, the original render's
    size (test-set metadata), the standardized image's size (its PNG header), repository, and the
    two crowding axes. Refuses an image that does not follow the 1,568 px standard."""
    ts = json.loads(Path(test_set).read_text())["diagrams"]
    out = {}
    for d in ts:
        key = d["key"][:-5] if d["key"].endswith(".puml") else d["key"]
        sw, sh = png_size(Path(images) / f"{key}.png")
        ow, oh = d["image_width"], d["image_height"]
        if max(sw, sh) != min(LONG_SIDE, max(ow, oh)):
            raise SystemExit(f"{key}: standardized long side {max(sw, sh)}, expected "
                             f"min({LONG_SIDE}, {max(ow, oh)})")
        out[key] = {"type": d["primary_type"], "tier": d["tier"], "content_lines": d["content_lines"],
                    "orig_w": ow, "orig_h": oh, "std_w": sw, "std_h": sh, "repository": d["repository"]}
    return out


def crowding_values(rec):
    """Standardized area (MP) and shrink (original area / standardized area; 1 when the original
    long side is at most 1,568 px) of one diagram."""
    std_area = rec["std_w"] * rec["std_h"] / 1e6
    if max(rec["orig_w"], rec["orig_h"]) <= LONG_SIDE:
        shrink = 1.0
    else:
        shrink = (rec["orig_w"] * rec["orig_h"]) / (rec["std_w"] * rec["std_h"])
    return {"size": float(rec["content_lines"]), "std_area": std_area, "shrink": shrink}


def design(table, keys):
    """The model matrices of A and B for the rows `keys` (loader order), the standardization
    constants and the 25th / 75th percentile values, the latter two from every diagram of
    `table` once (the 1,000 test diagrams). Percentiles are type-7 (the benchmark's), of the
    property's own values within type, then put on the model's z log scale."""
    raw = {k: crowding_values(rec) for k, rec in table.items()}
    consts, pct = {}, {}
    for t in M.TYPES:
        ks = [k for k in table if table[k]["type"] == t]
        if not ks:
            raise ValueError(f"no test diagram of type {t}")
        consts[t], pct[t] = {}, {}
        for prop in ("size", "std_area", "shrink"):
            v = np.array([raw[k][prop] for k in ks], float)
            lv = np.log(v)
            mu, sd = float(lv.mean()), float(lv.std())
            if not sd > 0:
                raise ValueError(f"{t} {prop}: no spread")
            consts[t][prop] = {"mean_log": mu, "sd_log": sd, "n": len(ks)}
            s = sorted(v.tolist())
            q = {"p25": M.percentile(s, Q_LO), "p75": M.percentile(s, Q_HI)}
            pct[t][prop] = {"value": q, "z": {k: (math.log(x) - mu) / sd for k, x in q.items()}}

    def z(k, prop):
        c = consts[table[k]["type"]][prop]
        return (math.log(raw[k][prop]) - c["mean_log"]) / c["sd_log"]

    base = [[1.0] + [float(table[k]["type"] == t) for t in OTHER_TYPES] for k in keys]
    size_col = len(base[0])
    X, q = {}, {}
    for m, spec in MODELS.items():
        X[m] = np.array([b + [z(k, "size"), z(k, spec["crowding"])] for b, k in zip(base, keys)])
        q[m] = {}
        for prop, name in (("size", "size"), ("crowding", spec["crowding"])):
            q[m][prop] = {e: np.array([pct[table[k]["type"]][name]["z"][e] for k in keys]) for e in ("p25", "p75")}
    columns = {m: ["intercept (class)"] + list(OTHER_TYPES) + [SIZE_COLUMN, MODELS[m]["column"]] for m in MODELS}
    return {"X": X, "q": q, "col": {"size": size_col, "crowding": size_col + 1}, "columns": columns,
            "constants": consts, "percentiles": {t: {p: pct[t][p]["value"] for p in pct[t]} for t in pct}}


# --------------------------------------------------------------------------- #
# resampling
# --------------------------------------------------------------------------- #
def repository_resamples(repos, n_resamples=N_RESAMPLES, seed=SEED):
    """Resamples of repositories with replacement (every diagram of a drawn repository enters),
    from one random.Random(seed), as many repositories per draw as there are."""
    rng = random.Random(seed)
    by_repo = collections.defaultdict(list)
    for i, r in enumerate(repos):
        by_repo[r].append(i)
    names = sorted(by_repo)
    return [[i for r in rng.choices(names, k=len(names)) for i in by_repo[r]] for _ in range(n_resamples)]


def _cell(point, values, sign=1):
    """main_tables.interval (a point only when the point or any resample is undefined), without
    its excludes_zero flag: no cell outside the agreement rule carries a verdict."""
    s = (lambda x: None if x is None else sign * x)
    c = M.interval(s(point), [s(v) for v in values])
    c.pop("excludes_zero", None)
    return c


def _floor_cell(point, why, sign=1):
    c = M.point_only(None if point is None else sign * point, why)
    c.pop("excludes_zero", None)
    return c


# --------------------------------------------------------------------------- #
# the analysis
# --------------------------------------------------------------------------- #
def fit_stat(X, y, comp, q, col, part, idx, max_iter=MAX_ITER):
    """One fit on the rows `idx` (with repetition) of a run: the coefficients and the predicted
    25th / 75th percentile levels for size and crowding; None when the fit does not converge."""
    idx = np.asarray(idx, dtype=int)
    if part == "element_f1":
        idx = idx[comp[idx]]
    if len(idx) == 0:
        return None
    XX, yy = X[idx], y[idx]
    beta, ok = fit_logit(XX, yy, firth=(part == "compile"), max_iter=max_iter)
    if not ok:
        return None
    out = {"coef": [float(x) for x in beta]}
    for prop in PROPS:
        out[prop] = predicted_levels(beta, XX, col[prop], q[prop]["p25"][idx], q[prop]["p75"][idx])
    return out


def agreement(cell_a, cell_b):
    """The Decision's rule on two crowding-oriented intervals of the same read quantity: a
    crowding verdict only where both lie on the same side of 0; otherwise the fixed sentence."""
    a, b = cell_a.get("ci"), cell_b.get("ci")
    if a is not None and b is not None:
        if a[0] > 0 and b[0] > 0:
            return {"verdict": "crowding", "side": "above 0", "text": "both axes' intervals lie above 0"}
        if a[1] < 0 and b[1] < 0:
            return {"verdict": "crowding", "side": "below 0", "text": "both axes' intervals lie below 0"}
    return {"verdict": "not separated", "side": None, "text": NOT_SEPARATED}


def count_intervals(res):
    """Interval-bearing cells of the coefficient, predicted-difference and Δ tables (the read
    list's 12 are among them): (with an interval, point only)."""
    cells = []
    for mid in res["coefficients"]:
        for m in MODELS:
            for part in PARTS:
                cells += list(res["coefficients"][mid][m][part].values())
                cells += [res["predicted_differences"][mid][m][part][p]["diff"] for p in PROPS]
    for size in res["deltas"]:
        for m in MODELS:
            for part in PARTS:
                cells += [res["deltas"][size][m][part][p] for p in PROPS]
    with_ci = sum(1 for c in cells if c.get("ci") is not None)
    return with_ci, len(cells) - with_ci


def analyse(v2, table, n_resamples=N_RESAMPLES, seed=SEED, event_floor=EVENT_FLOOR, max_iter=MAX_ITER,
            pairs=None):
    """Every number of the Decision. v2: {registry id: loader rows} for the six main runs, every
    run's rows in the loader's (sorted-key) order. table: predictor_table()."""
    t0 = time.time()
    pairs = pairs or M.PAIRS
    sizes = list(pairs)
    keys = [r["key"] for r in next(iter(v2.values()))]
    for mid, rows in v2.items():
        if [r["key"] for r in rows] != keys:
            raise ValueError(f"{mid}: rows are not aligned with the other runs")
    missing = [k for k in keys if k not in table]
    if missing:
        raise ValueError(f"{len(missing)} row keys have no predictors, e.g. {missing[0]}")
    if len(table) != len(keys):
        raise ValueError(f"predictor table holds {len(table)} diagrams, rows {len(keys)}")
    rows0 = next(iter(v2.values()))
    bad = [r["key"] for r in rows0 if r["primary_type"] != table[r["key"]]["type"]]
    if bad:
        raise ValueError(f"type differs between rows and test set for {bad[0]}")

    D = design(table, keys)
    n = len(keys)
    draws = M.make_resamples(n, n_resamples, seed)
    rdraws = repository_resamples([table[k]["repository"] for k in keys], n_resamples, seed)
    identity = np.arange(n)

    fits, events = {}, {}
    for mid, rows in v2.items():
        comp = np.array([bool(r["csr"]["compiled"]) for r in rows])
        f1 = np.array([float(r["element"]["f1"]) for r in rows])
        n_comp = int(comp.sum())
        minority = min(n_comp, n - n_comp)
        floored = minority < event_floor
        events[mid] = {"n": n, "n_compiled": n_comp, "n_failed": n - n_comp, "compile_minority": minority,
                       "event_floor": event_floor, "below_event_floor": floored,
                       "f1_compiled_eq_1": int(np.sum(f1[comp] == 1.0)),
                       "f1_compiled_eq_0": int(np.sum(f1[comp] == 0.0)),
                       "f1_compiled_between": int(np.sum((f1[comp] > 0) & (f1[comp] < 1)))}
        fits[mid] = {}
        for m in MODELS:
            fits[mid][m] = {}
            for part in PARTS:
                y = comp.astype(float) if part == "compile" else f1
                X, q = D["X"][m], D["q"][m]

                def stat(idx, X=X, y=y, q=q, part=part):
                    return fit_stat(X, y, comp, q, D["col"], part, idx, max_iter)
                f = {"point": stat(identity), "floored": part == "compile" and floored}
                if not f["floored"]:
                    f["boot"] = [stat(d) for d in draws]
                    if part == "compile":
                        f["repo"] = [stat(d) for d in rdraws]
                fits[mid][m][part] = f

    def get(s, *path):
        if s is None:
            return None
        for p in path:
            s = s[p]
        return s

    def floor_why(mid):
        e = events[mid]
        return f"event floor: {e['compile_minority']} minority outcomes < {event_floor}"

    def cell(mid, m, part, *path, sign=1):
        f = fits[mid][m][part]
        p = get(f["point"], *path)
        if f["floored"]:
            return _floor_cell(p, floor_why(mid), sign)
        return _cell(p, [get(b, *path) for b in f["boot"]], sign)

    def delta(size, m, part, *path, sign=1):
        z, ft = pairs[size]
        fz, ff = fits[z][m][part], fits[ft][m][part]
        p = M.sub(get(ff["point"], *path), get(fz["point"], *path))
        if fz["floored"] or ff["floored"]:
            why = floor_why(z if fz["floored"] else ft)
            c = _floor_cell(p, why, sign)
        else:
            c = _cell(p, [M.sub(get(b, *path), get(a, *path)) for a, b in zip(fz["boot"], ff["boot"])], sign)
        return c

    res = {"meta": {"seed": seed, "n_resamples": n_resamples, "ci_q": list(CI_Q), "n_rows": n,
                    "method": "paired percentile bootstrap (type-7) with the index draw of build/main_tables.py "
                              "(analysis.bootstrap.make_resamples) on the loader's rows; one draw for all six runs",
                    "reading": "descriptive: outside the claim family of 12 (analysis-protocol.md §9); a read list "
                               "of 12 intervals; every other number descriptive",
                    "scorer": "v2", "event_floor": event_floor, "max_iter": max_iter, "tol": TOL,
                    "models": {m: {"crowding": s["name"], "columns": D["columns"][m]} for m, s in MODELS.items()},
                    "parts": PARTS,
                    "standardization": D["constants"], "percentiles": D["percentiles"],
                    "pairs": pairs},
           "events": events, "coefficients": {}, "predicted_differences": {}, "deltas": {}}

    for mid in v2:
        res["coefficients"][mid], res["predicted_differences"][mid] = {}, {}
        for m in MODELS:
            res["coefficients"][mid][m], res["predicted_differences"][mid][m] = {}, {}
            for part in PARTS:
                res["coefficients"][mid][m][part] = {name: cell(mid, m, part, "coef", j)
                                                     for j, name in enumerate(D["columns"][m])}
                res["predicted_differences"][mid][m][part] = {
                    prop: {"p25": get(fits[mid][m][part]["point"], prop, "p25"),
                           "p75": get(fits[mid][m][part]["point"], prop, "p75"),
                           "diff": cell(mid, m, part, prop, "diff")} for prop in PROPS}
    for size in sizes:
        res["deltas"][size] = {m: {part: {prop: delta(size, m, part, prop, "diff") for prop in PROPS}
                                   for part in PARTS} for m in MODELS}

    # ---- the read list of 12 and the agreement rule ------------------------------------- #
    read, verdicts = [], {}
    for size in sizes:
        z, ft = pairs[size]
        verdicts[size] = {}
        for qty in ("fine_tuned", "delta"):
            by_axis = {}
            for m, spec in MODELS.items():
                sgn = spec["toward_crowding"]
                c = (cell(ft, m, "compile", "crowding", "diff", sign=sgn) if qty == "fine_tuned"
                     else delta(size, m, "compile", "crowding", "diff", sign=sgn))
                by_axis[m] = c
                read.append({"axis": m, "size": size, "quantity": qty, "run": ft if qty == "fine_tuned" else f"{ft} − {z}",
                             **c})
            verdicts[size][qty] = agreement(by_axis["A"], by_axis["B"])
    order = {m: i for i, m in enumerate(MODELS)}
    read.sort(key=lambda r: (order[r["axis"]], r["quantity"] != "fine_tuned", sizes.index(r["size"])))
    res["read_list"] = read
    res["agreement"] = verdicts

    # ---- repository-to-diagram resampling spread of the read crowding differences ----- #
    ratio = {}
    for mid in v2:
        ratio[mid] = {}
        for m in MODELS:
            f = fits[mid][m]["compile"]
            if f["floored"]:
                ratio[mid][m] = {"ratio": None, "why": floor_why(mid)}
                continue
            dv = [get(b, "crowding", "diff") for b in f["boot"]]
            rv = [get(b, "crowding", "diff") for b in f["repo"]]
            dd, rr = [x for x in dv if x is not None], [x for x in rv if x is not None]
            sd_d = float(np.std(dd, ddof=1)) if len(dd) > 1 else None
            sd_r = float(np.std(rr, ddof=1)) if len(rr) > 1 else None
            ratio[mid][m] = {"sd_diagram": sd_d, "sd_repository": sd_r,
                             "ratio": (sd_r / sd_d) if sd_d and sd_r is not None else None,
                             "undefined_diagram": len(dv) - len(dd), "undefined_repository": len(rv) - len(rr)}
    res["repository_ratio"] = ratio
    res["ceiling"] = {mid: {"n_failed_per_1000": events[mid]["n_failed"] * 1000 / n,
                            "text": CEILING_SENTENCE} for mid in v2}
    with_ci, point_only = count_intervals(res)
    res["interval_count"] = {"with_interval": with_ci, "point_only": point_only, "read": len(read),
                             "read_with_interval": sum(1 for r in read if r.get("ci") is not None)}
    res["point_only_cells"] = point_only_cells(res)
    res["meta"]["seconds_total"] = round(time.time() - t0, 1)
    return res


def point_only_cells(res):
    out = []
    for mid, by_m in res["coefficients"].items():
        for m, by_p in by_m.items():
            for part, cells in by_p.items():
                for name, c in cells.items():
                    if c.get("ci") is None:
                        out.append({"cell": f"{mid} {m} {part} coefficient {name}", "why": c.get("why_point_only")})
                for prop in PROPS:
                    c = res["predicted_differences"][mid][m][part][prop]["diff"]
                    if c.get("ci") is None:
                        out.append({"cell": f"{mid} {m} {part} {prop} difference", "why": c.get("why_point_only")})
    for size, by_m in res["deltas"].items():
        for m, by_p in by_m.items():
            for part, cells in by_p.items():
                for prop, c in cells.items():
                    if c.get("ci") is None:
                        out.append({"cell": f"{size} Δ {m} {part} {prop} difference", "why": c.get("why_point_only")})
    return out


# --------------------------------------------------------------------------- #
# checks and stated lines
# --------------------------------------------------------------------------- #
def check_published(v2, published, n_resamples=N_RESAMPLES, seed=SEED):
    """Each run's compiled count and CSR (point and interval, on the main draw) equal the
    published main tables, exactly; refuses otherwise. Returns the number of values compared."""
    bad, n_checked = [], 0
    meta = published.get("meta", {})
    if meta.get("seed") != seed or meta.get("n_resamples") != n_resamples:
        bad.append(f"draw: published seed {meta.get('seed')} / {meta.get('n_resamples')} resamples")
    n = len(next(iter(v2.values())))
    draws = M.make_resamples(n, n_resamples, seed)
    for mid, rows in v2.items():
        csr_rows = [r["csr"] for r in rows]
        n_comp = csr_rate(csr_rows)["compiled"]
        csr = M.interval(csr_rate(csr_rows)["csr"], [csr_rate([csr_rows[i] for i in d])["csr"] for d in draws])
        pub_run = published.get("runs", {}).get(mid)
        pub_main = published.get("main_table", {}).get(mid)
        if pub_run is None or pub_main is None:
            bad.append(f"{mid} not in the published tables")
            continue
        for label, ours, theirs in (("n_compiled (runs)", n_comp, pub_run["n_compiled"]),
                                    ("n_compiled (main_table)", n_comp, pub_main["n_compiled"]),
                                    ("CSR point", csr["point"], pub_main["overall"][M.CSR]["point"]),
                                    ("CSR interval", csr["ci"], pub_main["overall"][M.CSR]["ci"])):
            n_checked += 1
            if ours != theirs:
                bad.append(f"{mid} {label}: {ours} != {theirs}")
    if bad:
        raise SystemExit("differs from the published main tables: " + "; ".join(bad[:10]))
    return n_checked


def coverage_line(error_counts, run_dirs, pairs=None):
    """After fine-tuning, the compiled low-structural cells that involve Element F1 (Element F1
    only + both) of all compiled low-structural cells, per size, from error_analysis.json counts."""
    pairs = pairs or M.PAIRS
    out = {}
    for size, (_, ft) in pairs.items():
        c = error_counts[run_dirs[ft]]
        out[size] = {"element_involved": int(c["low_element_only"] + c["low_both"]),
                     "compiled_low_structural": int(c["compiled_low_structural"])}
    sizes = list(pairs)
    text = ("After fine-tuning, Element F1 is involved in only "
            + " / ".join(str(out[s]["element_involved"]) for s in sizes) + " of the "
            + " / ".join(str(out[s]["compiled_low_structural"]) for s in sizes)
            + " compiled low-structural cells (" + " / ".join(sizes)
            + "; runs/test/_analysis/error_analysis.json counts), so the regression does not describe low "
              "relationship accuracy.")
    return {"by_size": out, "text": text}


# --------------------------------------------------------------------------- #
# markdown
# --------------------------------------------------------------------------- #
def _f(x, nd=4):
    return "—" if x is None else f"{x:.{nd}f}"


def _ci(c, nd=4):
    fmt = (lambda x: "—" if x is None else f"{x:+.{nd}f}")
    if c.get("ci") is None:
        return f"{fmt(c['point'])} (point only)"
    return f"{fmt(c['point'])} [{fmt(c['ci'][0])}, {fmt(c['ci'][1])}]"


def markdown(res, displays):
    L = []
    w = L.append
    m = res["meta"]
    ic = res["interval_count"]
    pairs = m["pairs"]
    sizes = list(pairs)
    runs = list(res["coefficients"])
    w("# Difficulty regression: diagram size and crowding against compile success and Element F1\n")
    w(f"Generated by `build/difficulty_regression.py`. Scorer v2. One model per run and outcome part, the two "
      "diagram types pooled with common slopes; content lines and the crowding axis enter as z of their log, "
      "standardized within type with constants from the 1,000 test diagrams. Model A's crowding axis is the "
      f"{MODELS['A']['name']}; Model B's is {MODELS['B']['name']} (original area / standardized area, 1 for an "
      "image not downscaled). Compile success is fitted on all 1,000 diagrams of a run by Firth-penalized "
      "logistic regression; per-diagram Element F1 on the run's own compiled diagrams by fractional logit (IRLS). "
      "Relationship F1 is not an outcome.\n")
    w(f"Paired percentile bootstrap, seed {m['seed']}, {m['n_resamples']} resamples, 95% type-7 interval, with the "
      "index draw of `build/main_tables.py` on the loader's rows, one draw for all six runs. Each size has one "
      "training seed, so the intervals cover diagram variance only. A compile model whose minority outcome in "
      f"the run is below {m['event_floor']} gives points only; a resample whose fit does not converge within "
      f"{m['max_iter']} iterations makes the cells of that fit point-only.\n")
    w("Every number is descriptive and outside the claim family of 12 of the main comparison. The read list holds "
      f"12 intervals. The report holds **{ic['with_interval']} intervals** in all (coefficients, predicted "
      f"differences and their fine-tuned − zero-shot differences; the read list's 12 are among them), and "
      f"{ic['point_only']} interval cells with a point only.\n")
    w("A predicted difference sets the property of every diagram of the run's population to the 25th, then the "
      "75th percentile of its own type's test-set values, keeps the other terms as observed, and averages the "
      "predicted probabilities: 75th − 25th. Percentiles are fixed from the test set.\n")

    w("## 1. Read list\n")
    w("The crowding difference in predicted compile rate is the predicted compile rate at the more crowded "
      f"quartile end minus that at the less crowded end: for Model A, standardized area at its 25th minus its 75th "
      "percentile (less area at the same size is more crowded), the negative of the Model A crowding entry of §3; "
      "for Model B, shrink at its 75th minus its 25th percentile, the Model B crowding entry of §3.\n")
    w("| Axis | Quantity | " + " | ".join(sizes) + " |")
    w("|---|---|" + "---|" * len(sizes))
    for ax in MODELS:
        for qty, label in (("fine_tuned", "after fine-tuning"), ("delta", "fine-tuned − zero-shot")):
            cells = {r["size"]: r for r in res["read_list"] if r["axis"] == ax and r["quantity"] == qty}
            w(f"| {ax}: {MODELS[ax]['name']} | {label} | " + " | ".join(_ci(cells[s]) for s in sizes) + " |")
    w("")
    w("Agreement rule: a crowding verdict for a size only where both axes' intervals lie on the same side of 0; "
      f"otherwise \"{NOT_SEPARATED}\".\n")
    w("| Size | After fine-tuning | Fine-tuned − zero-shot |")
    w("|---|---|---|")
    for s in sizes:
        a = res["agreement"][s]
        w(f"| {s} | {a['fine_tuned']['text']} | {a['delta']['text']} |")
    w("")

    w("## 2. Ceiling and coverage\n")
    w(f"The predicted differences are on the probability scale: {CEILING_SENTENCE}. Compile failures per 1,000 "
      "diagrams:\n")
    w("| Run | compile failures per 1,000 | compiled | Element F1 = 1 / between / = 0 among compiled |")
    w("|---|---|---|---|")
    for mid in runs:
        e = res["events"][mid]
        w(f"| {displays.get(mid, mid)} | {res['ceiling'][mid]['n_failed_per_1000']:g} | {e['n_compiled']} | "
          f"{e['f1_compiled_eq_1']} / {e['f1_compiled_between']} / {e['f1_compiled_eq_0']} |")
    w("")
    w(res["coverage"]["text"] + "\n")

    w("## 3. Predicted differences (descriptive)\n")
    w("Predicted level at the 25th and at the 75th percentile (points), and 75th − 25th with its interval. "
      "Compile: predicted compile rate over all 1,000 diagrams. Element F1: predicted mean per-diagram Element F1 "
      "over the run's compiled diagrams.\n")
    w("| Run | Model | Part | Property | at 25th | at 75th | 75th − 25th |")
    w("|---|---|---|---|---|---|---|")
    for mid in runs:
        for ax in MODELS:
            for part in PARTS:
                for prop in PROPS:
                    c = res["predicted_differences"][mid][ax][part][prop]
                    pname = "content lines" if prop == "size" else MODELS[ax]["crowding"].replace("std_area", "standardized area")
                    w(f"| {displays.get(mid, mid)} | {ax} | {part} | {pname} | {_f(c['p25'])} | {_f(c['p75'])} | "
                      f"{_ci(c['diff'])} |")
    w("")

    w("## 4. Fine-tuned − zero-shot differences of the predicted differences (descriptive)\n")
    w("Paired by diagram on the shared draw. The Element F1 part compares each run's own compiled diagrams, two "
      "different sets.\n")
    w("| Size | Model | Part | Property | Δ (75th − 25th) |")
    w("|---|---|---|---|---|")
    for s in sizes:
        for ax in MODELS:
            for part in PARTS:
                for prop in PROPS:
                    pname = "content lines" if prop == "size" else MODELS[ax]["crowding"].replace("std_area", "standardized area")
                    w(f"| {s} | {ax} | {part} | {pname} | {_ci(res['deltas'][s][ax][part][prop])} |")
    w("")

    w("## 5. Coefficients (descriptive)\n")
    w("Logit scale, per SD of the z log predictor within type. No difference of coefficients is computed.\n")
    for ax in MODELS:
        cols = res["meta"]["models"][ax]["columns"]
        w(f"Model {ax} ({MODELS[ax]['name']}):\n")
        w("| Run | Part | " + " | ".join(cols) + " |")
        w("|---|---|" + "---|" * len(cols))
        for mid in runs:
            for part in PARTS:
                cc = res["coefficients"][mid][ax][part]
                w(f"| {displays.get(mid, mid)} | {part} | " + " | ".join(_ci(cc[c], 3) for c in cols) + " |")
        w("")

    w("## 6. Repository-level against diagram-level resampling (stated, not used)\n")
    w("SD of the crowding difference in predicted compile rate under 1,000 resamples of repositories (every "
      "diagram of a drawn repository, seed as §5) divided by its SD under the diagram resamples of the intervals.\n")
    w("| Run | " + " | ".join(f"Model {ax}" for ax in MODELS) + " |")
    w("|---|" + "---|" * len(MODELS))
    for mid in runs:
        w(f"| {displays.get(mid, mid)} | " + " | ".join(_f(res["repository_ratio"][mid][ax]["ratio"], 3)
                                                       for ax in MODELS) + " |")
    w("")

    w("## 7. Events and point-only cells\n")
    w(f"Event floor {m['event_floor']} on the compile minority outcome (failures or compiles, whichever is fewer).\n")
    w("| Run | failures | compiles | minority | below floor |")
    w("|---|---|---|---|---|")
    for mid in runs:
        e = res["events"][mid]
        w(f"| {displays.get(mid, mid)} | {e['n_failed']} | {e['n_compiled']} | {e['compile_minority']} | "
          f"{'yes' if e['below_event_floor'] else 'no'} |")
    w("")
    if res["point_only_cells"]:
        w("Point-only cells:\n")
        for c in res["point_only_cells"]:
            w(f"- {c['cell']}: {c['why']}")
        w("")
    else:
        w("No point-only cell.\n")
    return "\n".join(L)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=str(M.ROOT))
    ap.add_argument("--registry", default=str(M.REGISTRY))
    ap.add_argument("--published", default=str(PUBLISHED))
    ap.add_argument("--error-analysis", default=str(ERROR_ANALYSIS))
    ap.add_argument("--test-set", default=str(TEST_SET))
    ap.add_argument("--images", default=str(IMAGES))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)
    out = Path(args.out).resolve()
    if (M.INSTRUMENT / "analysis" / "out").resolve() in [out, *out.parents]:
        raise SystemExit("refused: never write into the benchmark's analysis/out")
    published = json.loads(Path(args.published).read_text())
    if published.get("meta", {}).get("main_tables_sha256") != sha256(M.__file__):
        raise SystemExit("the published main tables were not written by this build/main_tables.py")
    error_counts = json.loads(Path(args.error_analysis).read_text())["counts"]
    panel = M.load(args.root, args.registry)
    M.check_rows(panel["v2"], panel["v1"])
    v2 = panel["v2"]
    n_checked = check_published(v2, published)
    table = predictor_table(args.test_set, args.images)
    res = analyse(v2, table)
    entries = M.load_registry(args.registry)
    displays = {e.id: e.display for e in entries}
    res["coverage"] = coverage_line(error_counts, {e.id: e.run_dir for e in entries})
    res["meta"].update(root=str(Path(args.root).resolve()), registry=str(Path(args.registry).resolve()),
                       values_checked_against_published=n_checked,
                       difficulty_regression_sha256=sha256(__file__), main_tables_sha256=sha256(M.__file__),
                       published_sha256=sha256(args.published), error_analysis_sha256=sha256(args.error_analysis),
                       test_set_sha256=sha256(args.test_set))
    out.mkdir(parents=True, exist_ok=True)
    (out / "difficulty_regression.json").write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n")
    (out / "difficulty_regression.md").write_text(markdown(res, displays))
    print(f"wrote {out}/difficulty_regression.{{json,md}}; {n_checked} published values equal; "
          f"{res['interval_count']['with_interval']} intervals ({res['meta']['seconds_total']} s)")
    return res


if __name__ == "__main__":
    main()
