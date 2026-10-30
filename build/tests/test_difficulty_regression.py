"""Tests for build/difficulty_regression.py, on synthetic data and small synthetic panels (no scored
run is read).

Run from fine-tune/:  python3 -m pytest build/tests/test_difficulty_regression.py -q
"""
import copy
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import pytest

FT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(FT / "build"))
sys.path.insert(0, str(FT / "build" / "tests"))
import difficulty_regression as R  # noqa: E402
import main_tables as M  # noqa: E402
import test_main_tables as T  # noqa: E402

from analysis.bootstrap import make_resamples  # noqa: E402

N_RES = 40


# --------------------------------------------------------------------------- #
# fitting against known answers
# --------------------------------------------------------------------------- #
def table_2x2(a, b, c, d):
    """x = 0: a events, b non-events; x = 1: c events, d non-events."""
    x = [0] * (a + b) + [1] * (c + d)
    y = [1] * a + [0] * b + [1] * c + [0] * d
    return np.column_stack([np.ones(len(x)), x]), np.array(y, float)


def test_firth_on_a_2x2_table_adds_one_half_to_every_cell():
    """With one binary covariate the model is saturated and Firth's estimate is the empirical
    logit with 1/2 added to each cell (Haldane-Anscombe)."""
    X, y = table_2x2(30, 70, 12, 88)
    b, ok = R.fit_logit(X, y, firth=True)
    assert ok
    b0 = math.log(30.5 / 70.5)
    assert b[0] == pytest.approx(b0, abs=1e-9)
    assert b[1] == pytest.approx(math.log(12.5 / 88.5) - b0, abs=1e-9)


def test_firth_stays_finite_on_a_separated_table_where_ml_does_not_converge():
    X, y = table_2x2(15, 35, 0, 50)
    b_ml, ok_ml = R.fit_logit(X, y)
    assert not ok_ml                                     # the ML estimate drifts to -infinity
    b, ok = R.fit_logit(X, y, firth=True)
    assert ok and np.all(np.isfinite(b))
    b0 = math.log(15.5 / 35.5)
    assert b[1] == pytest.approx(math.log(0.5 / 50.5) - b0, abs=1e-9)


def test_ml_on_a_2x2_table_is_the_empirical_logit():
    X, y = table_2x2(30, 70, 12, 88)
    b, ok = R.fit_logit(X, y)
    assert ok
    assert b[0] == pytest.approx(math.log(30 / 70), abs=1e-9)
    assert b[1] == pytest.approx(math.log(12 / 88) - math.log(30 / 70), abs=1e-9)


def test_firth_is_close_to_ml_on_a_large_non_separated_sample():
    rng = np.random.default_rng(20260614)
    n = 5000
    X = np.column_stack([np.ones(n), rng.integers(0, 2, n), rng.normal(size=(n, 2))])
    y = (rng.random(n) < 1 / (1 + np.exp(-(X @ np.array([2.0, 0.5, -0.7, 0.4]))))).astype(float)
    b_ml, ok_ml = R.fit_logit(X, y)
    b_f, ok_f = R.fit_logit(X, y, firth=True)
    assert ok_ml and ok_f
    assert np.max(np.abs(b_ml - b_f)) < 0.01
    assert np.max(np.abs(b_ml - b_f)) > 0                  # the penalty is there, and small


def test_the_fractional_logit_recovers_the_group_means_with_exact_zeros_and_ones():
    """Saturated binary covariate: the quasi-likelihood solution is logit of each group's mean
    of y in [0, 1], exact 0 and 1 included."""
    y0 = np.array([1.0, 1.0, 1.0, 0.5, 0.25, 0.0, 1.0, 0.8])
    y1 = np.array([1.0, 1.0, 1.0, 1.0, 0.9, 0.0, 1.0, 1.0, 1.0, 0.6])
    X = np.column_stack([np.ones(len(y0) + len(y1)), [0] * len(y0) + [1] * len(y1)])
    b, ok = R.fit_logit(X, np.r_[y0, y1])
    assert ok
    lg = (lambda m: math.log(m / (1 - m)))
    assert b[0] == pytest.approx(lg(y0.mean()), abs=1e-9)
    assert b[1] == pytest.approx(lg(y1.mean()) - lg(y0.mean()), abs=1e-9)


def test_the_fractional_logit_is_undefined_when_one_group_is_all_ones():
    X = np.column_stack([np.ones(8), [0, 0, 0, 0, 1, 1, 1, 1]])
    _, ok = R.fit_logit(X, np.array([0.5, 1, 0.2, 1, 1, 1, 1, 1.0]))
    assert not ok


# --------------------------------------------------------------------------- #
# predicted differences
# --------------------------------------------------------------------------- #
def test_the_predicted_difference_averages_over_the_population_with_the_property_set():
    X = np.array([[1.0, 0.0, 0.3, -1.0], [1.0, 1.0, -0.2, 0.5], [1.0, 1.0, 1.1, 2.0]])
    beta = np.array([0.4, -0.3, 0.9, -0.6])
    lo, hi = np.array([-0.7, -0.5, -0.5]), np.array([0.6, 0.8, 0.8])   # per-row: each row's type
    got = R.predicted_levels(beta, X, 3, lo, hi)
    s = (lambda e: 1 / (1 + math.exp(-e)))
    p_lo = sum(s(0.4 - 0.3 * X[i, 1] + 0.9 * X[i, 2] - 0.6 * lo[i]) for i in range(3)) / 3
    p_hi = sum(s(0.4 - 0.3 * X[i, 1] + 0.9 * X[i, 2] - 0.6 * hi[i]) for i in range(3)) / 3
    assert got["p25"] == pytest.approx(p_lo) and got["p75"] == pytest.approx(p_hi)
    assert got["diff"] == pytest.approx(p_hi - p_lo)
    assert X[0, 3] == -1.0                                  # the input is not changed


# --------------------------------------------------------------------------- #
# predictors and design
# --------------------------------------------------------------------------- #
def synthetic_table(rows, seed=5):
    """Predictors for the keys of a synthetic panel: content lines grow with tier, original
    renders of varied size and aspect, the 1,568 px standard applied; repositories of 1-3."""
    rng = random.Random(seed)
    out = {}
    for i, r in enumerate(rows):
        lines = max(1, int(math.exp(rng.gauss(2.0 + 0.5 * r["tier"], 0.4))))
        long_side = int(math.exp(rng.gauss(7.0 + 0.3 * r["tier"], 0.5)))
        aspect = 1 + 3 * rng.random()
        ow, oh = (long_side, max(1, int(long_side / aspect))) if rng.random() < 0.5 else \
                 (max(1, int(long_side / aspect)), long_side)
        f = min(1.0, R.LONG_SIDE / max(ow, oh))
        sw, sh = (ow, oh) if f == 1.0 else (round(ow * f), round(oh * f))
        if max(ow, oh) > R.LONG_SIDE:
            sw, sh = (R.LONG_SIDE, sh) if ow >= oh else (sw, R.LONG_SIDE)
        out[r["key"]] = {"type": r["primary_type"], "tier": r["tier"], "content_lines": lines, "orig_w": ow,
                         "orig_h": oh, "std_w": sw, "std_h": sh, "repository": f"repo{i // 3 if i % 7 else i}"}
    return out


def test_shrink_is_one_for_an_image_not_downscaled_and_the_area_ratio_otherwise():
    small = {"content_lines": 9, "orig_w": 1568, "orig_h": 400, "std_w": 1568, "std_h": 400}
    big = {"content_lines": 9, "orig_w": 3136, "orig_h": 800, "std_w": 1568, "std_h": 400}
    assert R.crowding_values(small)["shrink"] == 1.0
    assert R.crowding_values(big)["shrink"] == pytest.approx(4.0)
    assert R.crowding_values(big)["std_area"] == pytest.approx(1568 * 400 / 1e6)


def test_the_design_standardizes_within_type_and_fixes_type_percentiles():
    v2, _ = T.panel()
    rows = next(iter(v2.values()))
    table = synthetic_table(rows)
    keys = [r["key"] for r in rows]
    D = R.design(table, keys)
    for m in R.MODELS:
        X = D["X"][m]
        assert X.shape == (len(keys), 4)
        for t in M.TYPES:
            sel = np.array([table[k]["type"] == t for k in keys])
            assert X[sel, 2].mean() == pytest.approx(0, abs=1e-12) and X[sel, 2].std() == pytest.approx(1)
            assert X[sel, 3].mean() == pytest.approx(0, abs=1e-12) and X[sel, 3].std() == pytest.approx(1)
        assert list(X[:, 1]) == [float(table[k]["type"] == "sequence") for k in keys]
    # the 25th percentile of class content lines (type 7), on the z log scale
    cls = sorted(float(table[k]["content_lines"]) for k in keys if table[k]["type"] == "class")
    c = D["constants"]["class"]["size"]
    q25 = M.percentile(cls, 0.25)
    i = next(j for j, k in enumerate(keys) if table[k]["type"] == "class")
    assert D["q"]["A"]["size"]["p25"][i] == pytest.approx((math.log(q25) - c["mean_log"]) / c["sd_log"])
    assert D["percentiles"]["class"]["size"]["p25"] == q25


def test_png_size_reads_the_header_and_a_nonstandard_image_is_refused(tmp_path):
    def png(path, w, h):
        path.write_bytes(b"\x89PNG\r\n\x1a\n" + (13).to_bytes(4, "big") + b"IHDR" + w.to_bytes(4, "big")
                         + h.to_bytes(4, "big") + b"\x08\x06\x00\x00\x00")
    imgs = tmp_path / "img"
    imgs.mkdir()
    png(imgs / "a.png", 1568, 392)
    png(imgs / "b.png", 700, 500)
    assert R.png_size(imgs / "a.png") == (1568, 392)
    ts = {"diagrams": [
        {"key": "a.puml", "primary_type": "class", "tier": 1, "content_lines": 5, "repository": "r",
         "image_width": 3136, "image_height": 784},
        {"key": "b.puml", "primary_type": "class", "tier": 1, "content_lines": 5, "repository": "r",
         "image_width": 700, "image_height": 500}]}
    (tmp_path / "ts.json").write_text(json.dumps(ts))
    t = R.predictor_table(tmp_path / "ts.json", imgs)
    assert t["a"]["std_w"] == 1568 and t["b"]["orig_w"] == 700
    ts["diagrams"][1]["image_width"] = 900                   # long side 900 kept, but the PNG is 700
    (tmp_path / "ts.json").write_text(json.dumps(ts))
    with pytest.raises(SystemExit, match="standardized long side"):
        R.predictor_table(tmp_path / "ts.json", imgs)


def test_repository_resamples_draw_whole_repositories_from_the_seed():
    repos = ["r1", "r1", "r2", "r3", "r3", "r3"]
    a = R.repository_resamples(repos, 5, 7)
    assert a == R.repository_resamples(repos, 5, 7)
    groups = {"r1": [0, 1], "r2": [2], "r3": [3, 4, 5]}
    for d in a:
        i = 0
        while i < len(d):
            g = next(g for g in groups.values() if g[0] == d[i])
            assert d[i:i + len(g)] == g
            i += len(g)


# --------------------------------------------------------------------------- #
# the analysis on a synthetic panel
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def panel():
    v2, v1 = T.panel()
    return v2, v1, synthetic_table(next(iter(v2.values())))


@pytest.fixture(scope="module")
def result(panel):
    v2, _, table = panel
    return R.analyse(v2, table, n_resamples=N_RES, event_floor=5)


def test_every_run_model_and_part_has_its_coefficients_and_predicted_differences(result):
    assert list(result["coefficients"]) == [mid for p in M.PAIRS.values() for mid in p]
    for mid in result["coefficients"]:
        for m in R.MODELS:
            for part in R.PARTS:
                assert list(result["coefficients"][mid][m][part]) == result["meta"]["models"][m]["columns"]
                assert set(result["predicted_differences"][mid][m][part]) == {"size", "crowding"}
    assert "difference" not in json.dumps(result["coefficients"])    # no difference of coefficients


def test_the_read_list_holds_twelve_crowding_compile_intervals(result):
    read = result["read_list"]
    assert len(read) == 12
    assert {(r["axis"], r["size"], r["quantity"]) for r in read} == {
        (m, s, q) for m in R.MODELS for s in M.SIZES for q in ("fine_tuned", "delta")}


def test_the_read_list_is_oriented_toward_crowding(result):
    """Model A: area at the 25th minus the 75th percentile (the negative of the table entry);
    Model B: shrink at the 75th minus the 25th (the table entry)."""
    for r in result["read_list"]:
        s = r["size"]
        ft = M.PAIRS[s][1]
        src = (result["predicted_differences"][ft][r["axis"]]["compile"]["crowding"]["diff"]
               if r["quantity"] == "fine_tuned" else result["deltas"][s][r["axis"]]["compile"]["crowding"])
        sign = -1 if r["axis"] == "A" else 1
        assert r["point"] == pytest.approx(sign * src["point"])
        if src["ci"] is not None:
            lo, hi = sorted(sign * x for x in src["ci"])
            assert r["ci"] == pytest.approx([lo, hi])


def test_a_coefficient_interval_is_paired_on_the_main_draw(panel, result):
    """The size coefficient of 9B fine-tuned, Model B, compile part, refitted by hand on the
    draw of main_tables (make_resamples, seed 20260614)."""
    v2, _, table = panel
    mid = M.PAIRS["9B"][1]
    rows = v2[mid]
    keys = [r["key"] for r in rows]
    X = R.design(table, keys)["X"]["B"]
    y = np.array([float(r["csr"]["compiled"]) for r in rows])
    vals = []
    for d in make_resamples(len(rows), N_RES, R.SEED):
        b, ok = R.fit_logit(X[d], y[d], firth=True)
        vals.append(b[2] if ok else None)
    b, _ = R.fit_logit(X, y, firth=True)
    exp = M.interval(b[2], vals)
    got = result["coefficients"][mid]["B"]["compile"][R.SIZE_COLUMN]
    assert got["point"] == pytest.approx(exp["point"])
    assert got["ci"] == pytest.approx(exp["ci"])


def test_a_delta_is_fine_tuned_minus_zero_shot_resample_by_resample(result):
    for s in M.SIZES:
        z, f = M.PAIRS[s]
        d = result["deltas"][s]["A"]["element_f1"]["size"]
        assert d["point"] == pytest.approx(result["predicted_differences"][f]["A"]["element_f1"]["size"]["diff"]["point"]
                                           - result["predicted_differences"][z]["A"]["element_f1"]["size"]["diff"]["point"])


def test_the_interval_count_covers_every_interval_cell(result):
    ic = result["interval_count"]
    # 6 runs x 2 models x 2 parts x (4 coefficients + 2 differences) + 3 sizes x 2 x 2 x 2 deltas
    assert ic["with_interval"] + ic["point_only"] == 6 * 2 * 2 * 6 + 3 * 2 * 2 * 2 == 168
    assert ic["point_only"] == len(result["point_only_cells"])
    assert ic["read"] == 12


def test_the_repository_ratio_is_stated_per_run_and_axis(result):
    for mid in result["coefficients"]:
        for m in R.MODELS:
            r = result["repository_ratio"][mid][m]
            assert r["ratio"] == pytest.approx(r["sd_repository"] / r["sd_diagram"])


def test_no_cell_carries_a_verdict_flag(result):
    assert "excludes_zero" not in json.dumps({k: result[k] for k in ("coefficients", "predicted_differences",
                                                                     "deltas", "read_list")})


def test_the_ceiling_is_each_runs_failures_per_1000(panel, result):
    v2, _, _ = panel
    for mid, rows in v2.items():
        fails = sum(1 for r in rows if not r["csr"]["compiled"])
        assert result["ceiling"][mid]["n_failed_per_1000"] == pytest.approx(fails * 1000 / len(rows))
        assert result["ceiling"][mid]["text"] == R.CEILING_SENTENCE


# --------------------------------------------------------------------------- #
# the event floor
# --------------------------------------------------------------------------- #
def with_failures(rows, n_fail):
    """The rows with exactly n_fail compile failures, spread over the run."""
    out = copy.deepcopy(rows)
    step = len(out) // n_fail
    fail = set(range(0, step * n_fail, step))
    for i, r in enumerate(out):
        r["csr"]["compiled"] = i not in fail
    return out


@pytest.mark.parametrize("n_fail,floored", [(24, True), (25, False)])
def test_the_event_floor_gives_points_only_below_25_minority_outcomes(panel, n_fail, floored):
    v2, _, table = panel
    v2 = dict(v2)
    ft = M.PAIRS["27B"][1]
    v2[ft] = with_failures(v2[ft], n_fail)
    res = R.analyse(v2, table, n_resamples=8)
    assert res["events"][ft]["compile_minority"] == n_fail
    assert res["events"][ft]["below_event_floor"] is floored
    cells = list(res["coefficients"][ft]["A"]["compile"].values())
    cells += [res["predicted_differences"][ft][m]["compile"][p]["diff"] for m in R.MODELS for p in R.PROPS]
    cells += [res["deltas"]["27B"][m]["compile"][p] for m in R.MODELS for p in R.PROPS]
    cells += [r for r in res["read_list"] if r["size"] == "27B"]
    if floored:
        assert all(c["ci"] is None and "event floor" in c["why_point_only"] for c in cells)
        assert all(a["verdict"] == "not separated" for a in res["agreement"]["27B"].values())
        assert res["repository_ratio"][ft]["A"]["ratio"] is None
        # the Element F1 part has no floor
        assert res["predicted_differences"][ft]["A"]["element_f1"]["size"]["diff"].get("why_point_only") is None
    else:
        assert all(c["ci"] is not None for c in cells)


def test_a_non_converging_resample_makes_the_cell_point_only(panel):
    """A run whose compiled diagrams all score Element F1 = 1 except in one type: the fractional
    logit diverges, and every Element F1 cell of that run is undefined, never an interval."""
    v2, _, table = panel
    v2 = dict(v2)
    ft = M.PAIRS["2B"][1]
    rows = copy.deepcopy(v2[ft])
    for r in rows:
        r["element"]["f1"] = 1.0 if r["primary_type"] == "class" else r["element"]["f1"]
    v2[ft] = rows
    res = R.analyse(v2, table, n_resamples=8, event_floor=5)
    c = res["predicted_differences"][ft]["A"]["element_f1"]["size"]["diff"]
    assert c["ci"] is None and c["point"] is None
    assert res["deltas"]["2B"]["A"]["element_f1"]["size"]["ci"] is None


# --------------------------------------------------------------------------- #
# the agreement rule
# --------------------------------------------------------------------------- #
def ci(lo, hi):
    return {"point": (lo + hi) / 2, "ci": [lo, hi]}


@pytest.mark.parametrize("a,b,verdict,side", [
    (ci(0.01, 0.05), ci(0.002, 0.03), "crowding", "above 0"),
    (ci(-0.05, -0.01), ci(-0.2, -0.001), "crowding", "below 0"),
    (ci(0.01, 0.05), ci(-0.01, 0.03), "not separated", None),
    (ci(0.01, 0.05), ci(-0.05, -0.01), "not separated", None),
    (ci(0.0, 0.05), ci(0.01, 0.03), "not separated", None),
    (ci(0.01, 0.05), {"point": 0.02, "ci": None}, "not separated", None),
])
def test_the_agreement_rule(a, b, verdict, side):
    got = R.agreement(a, b)
    assert got["verdict"] == verdict and got["side"] == side
    if verdict == "not separated":
        assert got["text"] == "the data do not separate size from crowding"


# --------------------------------------------------------------------------- #
# the check against the published main tables, and the stated lines
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def published(panel):
    v2, v1, _ = panel
    return json.loads(json.dumps(M.analyse(v2, v1, n_resamples=N_RES, zero_edge_expected=2)))


def test_the_published_main_tables_agree(panel, published):
    v2, _, _ = panel
    assert R.check_published(v2, published, n_resamples=N_RES) == 6 * 4


@pytest.mark.parametrize("edit,match", [
    (lambda p: p["runs"][M.PAIRS["9B"][1]].update(n_compiled=p["runs"][M.PAIRS["9B"][1]]["n_compiled"] + 1),
     "n_compiled"),
    (lambda p: p["main_table"][M.PAIRS["2B"][0]]["overall"][M.CSR].update(point=0.5), "CSR point"),
    (lambda p: p["main_table"][M.PAIRS["27B"][1]]["overall"][M.CSR]["ci"].__setitem__(0, 0.0), "CSR interval"),
    (lambda p: p["meta"].update(seed=1), "draw"),
])
def test_a_difference_from_the_published_tables_is_refused(panel, published, edit, match):
    v2, _, _ = panel
    bad = copy.deepcopy(published)
    edit(bad)
    with pytest.raises(SystemExit, match=match):
        R.check_published(v2, bad, n_resamples=N_RES)


def test_the_coverage_line_is_read_from_the_error_analysis_counts():
    counts = {"2B_20000": {"low_element_only": 3, "low_both": 10, "compiled_low_structural": 67},
              "9B_20000": {"low_element_only": 2, "low_both": 7, "compiled_low_structural": 37},
              "27B_20000": {"low_element_only": 1, "low_both": 6, "compiled_low_structural": 31}}
    run_dirs = {M.PAIRS["2B"][1]: "2B_20000", M.PAIRS["9B"][1]: "9B_20000", M.PAIRS["27B"][1]: "27B_20000"}
    got = R.coverage_line(counts, run_dirs)
    assert "13 / 9 / 7 of the 67 / 37 / 31" in got["text"]
    assert got["by_size"]["27B"] == {"element_involved": 7, "compiled_low_structural": 31}


def test_the_markdown_states_the_count_the_read_list_and_the_rule(result):
    res = dict(result, coverage={"text": "COVERAGE LINE"})
    md = R.markdown(res, {})
    assert f"**{result['interval_count']['with_interval']} intervals**" in md
    assert "## 1. Read list" in md and "COVERAGE LINE" in md
    assert "the data do not separate size from crowding" in md
    assert "outside the claim family of 12" in md
    assert R.CEILING_SENTENCE in md


def test_misaligned_rows_or_missing_predictors_are_refused(panel):
    v2, _, table = panel
    bad = dict(v2)
    bad[M.PAIRS["2B"][0]] = list(reversed(bad[M.PAIRS["2B"][0]]))
    with pytest.raises(ValueError, match="not aligned"):
        R.analyse(bad, table, n_resamples=2)
    t = dict(table)
    t.pop(next(iter(t)))
    with pytest.raises(ValueError, match="no predictors"):
        R.analyse(v2, t, n_resamples=2)


def test_the_benchmarks_analysis_out_is_refused():
    with pytest.raises(SystemExit, match="analysis/out"):
        R.main(["--out", str(M.INSTRUMENT / "analysis" / "out" / "x")])


# --------------------------------------------------------------------------- #
# second pass (after verification): the Δ pairing, a non-converging resample, the fixed
# percentiles, and the sha256 in the json meta
# --------------------------------------------------------------------------- #
def _hand_crowding_diffs(v2, table, mid, model, part, draws, q=None):
    """The crowding 75th − 25th of one run, refitted by hand on each draw (None when a fit does
    not converge), with the per-row percentile values `q` (default: the full test set's)."""
    rows = v2[mid]
    keys = [r["key"] for r in rows]
    D = R.design(table, keys)
    X = D["X"][model]
    q = q or D["q"][model]["crowding"]
    comp = np.array([bool(r["csr"]["compiled"]) for r in rows])
    y = comp.astype(float) if part == "compile" else np.array([float(r["element"]["f1"]) for r in rows])
    out = []
    for d in [list(range(len(rows)))] + draws:
        idx = np.asarray(d)
        if part == "element_f1":
            idx = idx[comp[idx]]
        b, ok = R.fit_logit(X[idx], y[idx], firth=(part == "compile"))
        out.append(R.predicted_levels(b, X[idx], 3, q["p25"][idx], q["p75"][idx])["diff"] if ok else None)
    return out[0], out[1:]


def test_the_delta_interval_is_built_resample_by_resample_on_the_shared_draw(panel, result):
    """2B, Model A, compile, crowding: the Δ interval is the percentile interval of the
    per-resample differences fine-tuned − zero-shot on one draw, not the difference of the two
    runs' bounds."""
    v2, _, table = panel
    z, f = M.PAIRS["2B"]
    draws = make_resamples(len(v2[z]), N_RES, R.SEED)
    pz, vz = _hand_crowding_diffs(v2, table, z, "A", "compile", draws)
    pf, vf = _hand_crowding_diffs(v2, table, f, "A", "compile", draws)
    paired = M.interval(pf - pz, [b - a for a, b in zip(vz, vf)])
    got = result["deltas"]["2B"]["A"]["compile"]["crowding"]
    assert got["point"] == pytest.approx(paired["point"])
    assert got["ci"] == pytest.approx(paired["ci"])
    cz, cf = M.interval(pz, vz)["ci"], M.interval(pf, vf)["ci"]
    unpaired = [cf[0] - cz[1], cf[1] - cz[0]]
    naive = [cf[0] - cz[0], cf[1] - cz[1]]
    assert got["ci"] != pytest.approx(unpaired) and got["ci"] != pytest.approx(naive)


def test_a_converged_point_with_one_non_converging_resample_is_point_only(panel):
    """Every compiled class answer of 2B fine-tuned scores Element F1 = 1 except one at 0.5: the
    full-sample fractional logit converges, and a resample that leaves that diagram out cannot
    (class all ones). The cell keeps its point and has no interval."""
    v2, _, table = panel
    v2 = dict(v2)
    ft = M.PAIRS["2B"][1]
    rows = copy.deepcopy(v2[ft])
    cls = [i for i, r in enumerate(rows) if r["primary_type"] == "class" and r["csr"]["compiled"]]
    for i in cls:
        rows[i]["element"]["f1"] = 1.0
    rows[cls[0]]["element"]["f1"] = 0.5
    v2[ft] = rows
    draws = make_resamples(len(rows), 8, R.SEED)
    left_out = sum(1 for d in draws if cls[0] not in d)
    assert 0 < left_out < len(draws)                          # some resamples miss it, some keep it
    res = R.analyse(v2, table, n_resamples=8, event_floor=5)
    c = res["predicted_differences"][ft]["A"]["element_f1"]["size"]["diff"]
    assert c["point"] is not None and c["ci"] is None
    assert c["n_undefined"] == left_out
    assert "undefined resample" in c["why_point_only"]
    assert res["deltas"]["2B"]["A"]["element_f1"]["size"]["ci"] is None
    assert any(p["cell"].startswith(f"{ft} A element_f1") for p in res["point_only_cells"])


def test_the_percentiles_stay_fixed_from_the_test_set_in_every_resample(panel, result):
    """9B fine-tuned, Model B, Element F1, crowding: the interval equals the hand refit with the
    full test set's per-type percentiles on every resample, and differs from one that re-estimates
    the percentiles within each resample."""
    v2, _, table = panel
    mid = M.PAIRS["9B"][1]
    draws = make_resamples(len(v2[mid]), N_RES, R.SEED)
    p, vals = _hand_crowding_diffs(v2, table, mid, "B", "element_f1", draws)
    fixed = M.interval(p, vals)
    got = result["predicted_differences"][mid]["B"]["element_f1"]["crowding"]["diff"]
    assert got["point"] == pytest.approx(fixed["point"]) and got["ci"] == pytest.approx(fixed["ci"])

    # the alternative the Decision rules out: percentiles re-estimated on each resample's diagrams
    rows = v2[mid]
    keys = [r["key"] for r in rows]
    D = R.design(table, keys)
    c = D["constants"]
    types = [table[k]["type"] for k in keys]
    raw = np.array([R.crowding_values(table[k])["shrink"] for k in keys])
    comp = np.array([bool(r["csr"]["compiled"]) for r in rows])
    f1 = np.array([float(r["element"]["f1"]) for r in rows])
    X = D["X"]["B"]
    moved = []
    for d in draws:
        d = np.asarray(d)
        qz = {}
        for t in M.TYPES:
            s = sorted(raw[d][[types[i] == t for i in d]].tolist())
            qz[t] = {e: (math.log(M.percentile(s, qq)) - c[t]["shrink"]["mean_log"]) / c[t]["shrink"]["sd_log"]
                     for e, qq in (("p25", R.Q_LO), ("p75", R.Q_HI))}
        idx = d[comp[d]]
        b, ok = R.fit_logit(X[idx], f1[idx])
        lo = np.array([qz[types[i]]["p25"] for i in idx])
        hi = np.array([qz[types[i]]["p75"] for i in idx])
        moved.append(R.predicted_levels(b, X[idx], 3, lo, hi)["diff"] if ok else None)
    assert None not in moved and None not in vals
    assert got["ci"] != pytest.approx(M.interval(p, moved)["ci"])
    # and the per-row values handed to every fit are the full test set's two per-type values
    for e in ("p25", "p75"):
        per_type = {t: D["percentiles"][t]["shrink"][e] for t in M.TYPES}
        zq = D["q"]["B"]["crowding"][e]
        for i, k in enumerate(keys):
            t = table[k]["type"]
            assert zq[i] == pytest.approx((math.log(per_type[t]) - c[t]["shrink"]["mean_log"]) / c[t]["shrink"]["sd_log"])


def test_the_json_meta_records_the_sha256_of_the_script_and_of_main_tables(panel, published, tmp_path, monkeypatch):
    """main() on a synthetic panel, written to a tmp folder: meta holds the sha256 of
    build/difficulty_regression.py and build/main_tables.py as they are on disk."""
    import hashlib
    v2, v1, table = panel
    pub = copy.deepcopy(published)
    pub["meta"]["main_tables_sha256"] = hashlib.sha256(Path(M.__file__).read_bytes()).hexdigest()
    (tmp_path / "main_tables.json").write_text(json.dumps(pub))
    counts = {e.run_dir: {"low_element_only": 1, "low_both": 2, "compiled_low_structural": 9}
              for e in M.load_registry(M.REGISTRY)}
    (tmp_path / "error_analysis.json").write_text(json.dumps({"counts": counts}))
    real_check, real_analyse = R.check_published, R.analyse
    monkeypatch.setattr(M, "load", lambda root, registry=None: {"v2": v2, "v1": v1})
    monkeypatch.setattr(R, "predictor_table", lambda *a, **k: table)
    monkeypatch.setattr(R, "check_published", lambda v, p: real_check(v, p, n_resamples=N_RES))
    monkeypatch.setattr(R, "analyse", lambda v, t: real_analyse(v, t, n_resamples=4, event_floor=5))
    R.main(["--published", str(tmp_path / "main_tables.json"),
            "--error-analysis", str(tmp_path / "error_analysis.json"), "--out", str(tmp_path / "out")])
    meta = json.loads((tmp_path / "out" / "difficulty_regression.json").read_text())["meta"]
    assert meta["difficulty_regression_sha256"] == hashlib.sha256((FT / "build" / "difficulty_regression.py").read_bytes()).hexdigest()
    assert meta["main_tables_sha256"] == hashlib.sha256((FT / "build" / "main_tables.py").read_bytes()).hexdigest()
    assert meta["published_sha256"] == hashlib.sha256((tmp_path / "main_tables.json").read_bytes()).hexdigest()
    assert (tmp_path / "out" / "difficulty_regression.md").exists()
