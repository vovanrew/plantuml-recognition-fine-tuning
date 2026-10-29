"""Tests for build/efficiency_curve.py, on small synthetic panels (no scored run is read).

Run from fine-tune/:  python3 -m pytest build/tests/test_efficiency_curve.py -q
"""
import copy
import json
import sys
from pathlib import Path

import pytest

FT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(FT / "build"))
sys.path.insert(0, str(FT / "build" / "tests"))
import efficiency_curve as C  # noqa: E402
import main_tables as M  # noqa: E402
import test_main_tables as T  # noqa: E402

from analysis.bootstrap import make_resamples  # noqa: E402

N_RES = 60
IDS = {label: mid for label, mid, _ in C.POINTS}


def eight():
    """The six main runs of test_main_tables.panel() and the 2B on 1k and 5k (v2, v1)."""
    v2, v1 = T.panel()
    for mid, (seed, pc, q) in {IDS["1k"]: (31, 0.6, 0.8), IDS["5k"]: (32, 0.8, 0.85)}.items():
        v2[mid] = T.run_rows(seed, pc, q, empty_gt={"k0003", "k0007"})
        v1[mid] = [dict(r, element=T.metric(r["element"]["tp"], r["element"]["fp"] + 1, r["element"]["fn"],
                                            r["csr"]["compiled"], ta=r["element"]["type_accuracy"]))
                   for r in v2[mid]]
    return v2, v1


@pytest.fixture(scope="module")
def panel8():
    return eight()


@pytest.fixture(scope="module")
def result(panel8):
    v2, v1 = panel8
    return C.curve(v2, v1, n_resamples=N_RES, zero_edge_expected=2)


@pytest.fixture(scope="module")
def published(panel8):
    """What build/main_tables.py publishes for the six main runs of the same panel."""
    v2, v1 = panel8
    six = {mid for pair in M.PAIRS.values() for mid in pair}
    return json.loads(json.dumps(M.analyse({k: v for k, v in v2.items() if k in six},
                                           {k: v for k, v in v1.items() if k in six},
                                           n_resamples=N_RES, zero_edge_expected=2)))


def test_the_four_points_in_order_with_their_pairs_and_steps(result):
    assert list(result["points"]) == ["0", "1k", "5k", "20k"]
    assert [p["pairs"] for p in result["points"].values()] == [0, 1000, 5000, 20000]
    assert [p["steps"] for p in result["points"].values()] == [0, 126, 626, 2500]


def test_every_decided_statistic_is_shown_at_every_point(result):
    for p in result["points"].values():
        assert set(p["v2"]) == set(C.V2_STATS)
        assert set(p["v1"]) == set(C.V1_STATS)
        assert set(p["failed0"]) == set(C.F0_STATS)


def test_the_differences_are_the_three_neighbours_on_every_v2_statistic(result):
    assert list(result["differences"]) == ["1k-0", "5k-1k", "20k-5k"]
    for d in result["differences"].values():
        assert set(d) == set(C.V2_STATS)


def test_a_difference_is_paired_by_diagram_on_the_shared_draw(panel8, result):
    """20k - 5k on CSR, recomputed by hand from the shared draw."""
    v2, _ = panel8
    r5, r20 = v2[IDS["5k"]], v2[IDS["20k"]]
    draws = make_resamples(len(r5), N_RES, C.SEED)

    def csr(rows, d):
        return sum(rows[i]["csr"]["compiled"] for i in d) / len(d)
    ident = range(len(r5))
    expected = M.interval(csr(r20, ident) - csr(r5, ident), [csr(r20, d) - csr(r5, d) for d in draws])
    got = result["differences"]["20k-5k"][M.CSR]
    assert got["point"] == pytest.approx(expected["point"])
    assert got["ci"] == pytest.approx(expected["ci"])
    assert got["5k"] == pytest.approx(csr(r5, ident)) and got["20k"] == pytest.approx(csr(r20, ident))


def test_the_points_use_the_main_comparisons_draw(result, published):
    """The 0 and 20k points equal what main_tables publishes for the same panel, exactly."""
    assert C.check_published(result, published) > 30


def test_a_difference_from_the_published_tables_is_refused(result, published):
    bad = copy.deepcopy(published)
    cell = bad["main_table"][IDS["20k"]]["overall"][M.CSR]
    cell["point"] += 1e-12
    with pytest.raises(SystemExit, match="20k v2 csr"):
        C.check_published(result, bad)
    bad = copy.deepcopy(published)
    bad["headline"]["2B"]["element_f1"]["co"]["ci"][0] -= 1e-9
    with pytest.raises(SystemExit, match="headline 2B element_f1 co"):
        C.check_published(result, bad)


def test_the_joint_set_is_the_diagrams_both_runs_compiled(panel8, result):
    v2, _ = panel8
    r5, r20 = v2[IDS["5k"]], v2[IDS["20k"]]
    keep = [i for i in range(len(r5)) if r5[i]["csr"]["compiled"] and r20[i]["csr"]["compiled"]]
    j = result["joint_set"]
    assert j["pair"] == "20k-5k" and j["n_joint"] == len(keep)
    e5 = M.micro_or_none([r5[i]["element"] for i in keep])
    e20 = M.micro_or_none([r20[i]["element"] for i in keep])
    assert j["element_f1"]["point"] == pytest.approx(e20 - e5)
    assert j["element_f1"]["5k"] == pytest.approx(e5)
    r5_ = M.micro_or_none([r5[i]["relationship"] for i in keep])
    r20_ = M.micro_or_none([r20[i]["relationship"] for i in keep])
    assert j["relationship_f1"]["point"] == pytest.approx(r20_ - r5_)
    assert j["relationship_f1"]["5k"] == pytest.approx(r5_)
    assert j["relationship_f1"]["20k"] == pytest.approx(r20_)


def test_the_failed_equals_zero_cells_score_every_non_compiling_answer_zero(panel8, result):
    v2, _ = panel8
    rows = v2[IDS["1k"]]
    expected = M.stats([M.failed_zero(r) for r in rows])[M.EL["full"]]
    cell = result["points"]["1k"]["failed0"][M.EL["full"]]
    assert cell["failed0"]["point"] == pytest.approx(expected)
    assert cell["stored"]["point"] == pytest.approx(M.stats(rows)[M.EL["full"]])


def test_only_the_decided_statistics_and_no_verdict_flag(result):
    assert "gap|chrf|macro" not in C.V2_STATS and M.TYPE_ACC not in C.V1_STATS
    cells = [c for p in result["points"].values() for v in ("v2", "v1") for c in p[v].values()]
    cells += [c for d in result["differences"].values() for c in d.values()]
    cells += [result["joint_set"][k] for k in ("element_f1", "relationship_f1")]
    assert cells and all("excludes_zero" not in c for c in cells)


def test_a_point_with_no_compiled_answer_is_undefined_not_zero(panel8):
    v2, v1 = copy.deepcopy(panel8[0]), copy.deepcopy(panel8[1])
    dead = [T.row(r["key"], r["primary_type"], r["tier"], False, (0, 0, r["element"]["tp"] + r["element"]["fn"]),
                  (0, 0, r["relationship"]["tp"] + r["relationship"]["fn"]), "association", chrf=0.0)
            for r in v2[IDS["1k"]]]
    v2[IDS["1k"]] = v1[IDS["1k"]] = dead
    res = C.curve(v2, v1, n_resamples=N_RES, zero_edge_expected=2)
    co = res["points"]["1k"]["v2"][M.EL["co"]]
    assert co["point"] is None and co["ci"] is None
    assert res["differences"]["5k-1k"][M.EL["co"]]["point"] is None


def test_a_missing_curve_point_is_refused(panel8):
    v2, v1 = (dict(x) for x in panel8)
    del v2[IDS["5k"]], v1[IDS["5k"]]
    with pytest.raises(ValueError, match="qwen3.5-2b-ft5k"):
        C.curve(v2, v1, n_resamples=N_RES, zero_edge_expected=2)


def test_the_markdown_shows_every_section(result):
    md = C.markdown(result)
    for heading in ("## Points", "## Differences between neighbouring points", "## 20k-5k on the joint set",
                    "## Scorer v1", "## Non-compiling answers scored 0"):
        assert heading in md
    assert "not in the claim" in md or "none is in the claim family" in md
    assert "5,000 pairs: 626 steps" in md


def test_the_registry_holds_the_six_main_entries_unchanged_and_the_two_points():
    curve_ids = [e.id for e in M.load_registry(C.REGISTRY)]
    main = json.loads(M.REGISTRY.read_text())["models"]
    mine = json.loads(C.REGISTRY.read_text())["models"]
    assert mine[:6] == main
    assert set(IDS.values()) <= set(curve_ids)


def test_load_reads_the_eight_runs_through_the_frozen_loader(tmp_path, monkeypatch):
    monkeypatch.setattr(M, "REGISTRY", C.REGISTRY)      # write_root lays out every registry entry
    reg = T.write_root(tmp_path / "root")
    out = M.load(tmp_path / "root", C.REGISTRY)
    assert list(out["v2"]) == [e.id for e in reg] and len(reg) == 8


def test_the_benchmarks_analysis_out_is_refused():
    with pytest.raises(SystemExit, match="analysis/out"):
        C.main(["--out", str(M.INSTRUMENT / "analysis" / "out" / "x")])
