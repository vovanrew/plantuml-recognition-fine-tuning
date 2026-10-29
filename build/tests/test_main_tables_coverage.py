"""More tests for build/main_tables.py and build/score_test_runs.py, on fixtures whose expected
values are worked out by hand (decisions/main-tables-intervals/verify.md, Failures 1-7).

Run from fine-tune/:  python3 -m pytest build/tests/test_main_tables_coverage.py -q
No scored run is read.
"""
import os
import sys
from pathlib import Path

import pytest

FT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(FT / "build"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import main_tables as M  # noqa: E402
import score_test_runs as S  # noqa: E402
from test_main_tables import metric  # noqa: E402

N = 16


def f1(tp, fp, fn):
    """Micro F1 from pooled counts, written out here independently of the benchmark."""
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


# --------------------------------------------------------------------------- #
# a 16-diagram panel worked out by hand
#
# key i: tier 1 + i // 4; class when i % 4 in (0, 1), sequence when in (2, 3).
# zero-shot (2B): compiled {0,1,2,4,6,8,10,12,14}; a compiled row has Element (tp, fp, fn) =
#   (3, 1, 1), Relationship (1, 0, 1), chrF++ 60, type accuracy matched 3 / correct 2; a failed
#   row (0, 0, 4), (0, 0, 2), chrF++ 30 (stored raw), except failed key 3, whose Element row
#   keeps a parsed answer: (2, 0, 2).
# fine-tuned (2B): compiled every key but {2, 3, 13}; a compiled row (4, 1, 0), (2, 0, 0),
#   chrF++ 80, matched 4 / correct 4; a failed row (0, 0, 4), (0, 0, 2), chrF++ 10.
# v1: every compiled row has one Element false positive more.
# --------------------------------------------------------------------------- #
ZS_COMPILED = {0, 1, 2, 4, 6, 8, 10, 12, 14}
FT_COMPILED = set(range(N)) - {2, 3, 13}
JOINT = ZS_COMPILED & FT_COMPILED           # {0, 1, 4, 6, 8, 10, 12, 14}


def hand_row(i, compiled, el, rel, chrf, ta):
    key = f"k{i:04d}"
    by_rel = {rl: {"tp": 0, "fp": 0, "fn": 0} for rl in M.RELATIONS}
    by_rel["association"] = {"tp": rel[0], "fp": rel[1], "fn": rel[2]}
    return {"key": key, "primary_type": "class" if i % 4 in (0, 1) else "sequence", "tier": 1 + i // 4,
            "csr": {"key": key, "compiled": compiled},
            "element": dict(metric(*el, compiled, ta={"matched": ta[0], "correct": ta[1], "excluded": 0}), key=key),
            "relationship": dict(metric(*rel, compiled, by_relation=by_rel), key=key),
            "chrf": {"key": key, "compiled": compiled, "score": chrf}}


def zs_rows(perfect=False):
    rows = []
    for i in range(N):
        if i in ZS_COMPILED:
            rows.append(hand_row(i, True, (4, 0, 0) if perfect else (3, 1, 1), (1, 0, 1), 60.0, (3, 2)))
        else:
            rows.append(hand_row(i, False, (2, 0, 2) if i == 3 else (0, 0, 4), (0, 0, 2), 30.0, (0, 0)))
    return rows


def ft_rows():
    return [hand_row(i, True, (4, 1, 0), (2, 0, 0), 80.0, (4, 4)) if i in FT_COMPILED
            else hand_row(i, False, (0, 0, 4), (0, 0, 2), 10.0, (0, 0)) for i in range(N)]


def v1_of(rows):
    out = []
    for r in rows:
        e = r["element"]
        out.append(dict(r, element=dict(metric(e["tp"], e["fp"] + int(r["csr"]["compiled"]), e["fn"],
                                               e["compiled"], ta=e["type_accuracy"]), key=r["key"])))
    return out


@pytest.fixture(scope="module")
def hand():
    v2 = {}
    for size in M.SIZES:
        z, f = M.PAIRS[size]
        v2[z] = zs_rows(perfect=(size == "9B"))       # the 9B zero-shot run is perfect where it compiles
        v2[f] = ft_rows()
    v1 = {mid: v1_of(rows) for mid, rows in v2.items()}
    return M.analyse(v2, v1, n_resamples=40, zero_edge_expected=0), v2


ZS, FTID = M.PAIRS["2B"]


# --------------------------------------------------------------------------- #
# Failure 1: the main table and the v1 companion
# --------------------------------------------------------------------------- #
def test_the_main_table_holds_the_decided_statistics_and_margins(hand):
    res, _ = hand
    main = res["main_table"]
    assert set(main) == {mid for pair in M.PAIRS.values() for mid in pair}
    expected_ids = {"csr|csr", "element_f1|micro|zeros_for_failed", "element_f1|micro|compiled_only",
                    "relationship_f1|micro|zeros_for_failed", "relationship_f1|micro|compiled_only",
                    "chrf|macro|zeros_for_failed", "chrf|macro|compiled_only",
                    "type_accuracy|accuracy|compiled_only"}
    for cell in main.values():
        assert set(cell["overall"]) == expected_ids
        assert set(cell["by_type"]) == {"class", "sequence"} and set(cell["by_tier"]) == {1, 2, 3, 4}
        for margin in list(cell["by_type"].values()) + list(cell["by_tier"].values()):
            assert set(margin) == expected_ids | {"n", "n_compiled"}
    z = main[ZS]
    want = {"csr|csr": 9 / 16,
            "element_f1|micro|zeros_for_failed": f1(29, 9, 35),
            "element_f1|micro|compiled_only": f1(27, 9, 9),
            "relationship_f1|micro|zeros_for_failed": f1(9, 0, 23),
            "relationship_f1|micro|compiled_only": f1(9, 0, 9),
            "chrf|macro|zeros_for_failed": 9 * 60 / 16,
            "chrf|macro|compiled_only": 60.0,
            "type_accuracy|accuracy|compiled_only": 18 / 27}
    for st, v in want.items():
        assert z["overall"][st]["point"] == pytest.approx(v, abs=1e-12), st
        assert z["overall"][st]["ci"] is not None
    assert z["n_compiled"] == 9
    f = main[FTID]["overall"]
    assert f["csr|csr"]["point"] == pytest.approx(13 / 16)
    assert f["element_f1|micro|zeros_for_failed"]["point"] == pytest.approx(f1(52, 13, 12))
    assert f["element_f1|micro|compiled_only"]["point"] == pytest.approx(f1(52, 13, 0))
    assert f["relationship_f1|micro|zeros_for_failed"]["point"] == pytest.approx(f1(26, 0, 6))
    assert f["chrf|macro|zeros_for_failed"]["point"] == pytest.approx(65.0)
    assert f["type_accuracy|accuracy|compiled_only"]["point"] == pytest.approx(1.0)
    # margins of the zero-shot run
    cls, seq = z["by_type"]["class"], z["by_type"]["sequence"]
    assert (cls["n"]["point"], cls["n_compiled"]["point"]) == (8, 5)
    assert cls["csr|csr"]["point"] == pytest.approx(5 / 8)
    assert cls["element_f1|micro|zeros_for_failed"]["point"] == pytest.approx(f1(15, 5, 17))
    assert cls["relationship_f1|micro|zeros_for_failed"]["point"] == pytest.approx(f1(5, 0, 11))
    assert cls["chrf|macro|zeros_for_failed"]["point"] == pytest.approx(37.5)
    assert seq["csr|csr"]["point"] == pytest.approx(4 / 8)
    assert seq["element_f1|micro|zeros_for_failed"]["point"] == pytest.approx(f1(14, 4, 18))
    assert seq["relationship_f1|micro|zeros_for_failed"]["point"] == pytest.approx(f1(4, 0, 12))
    t1, t4 = z["by_tier"][1], z["by_tier"][4]
    assert (t1["n"]["point"], t1["n_compiled"]["point"]) == (4, 3)
    assert t1["csr|csr"]["point"] == pytest.approx(3 / 4)
    assert t1["element_f1|micro|zeros_for_failed"]["point"] == pytest.approx(f1(11, 3, 5))
    assert t1["relationship_f1|micro|zeros_for_failed"]["point"] == pytest.approx(f1(3, 0, 5))
    assert t4["csr|csr"]["point"] == pytest.approx(2 / 4)
    assert t4["element_f1|micro|zeros_for_failed"]["point"] == pytest.approx(f1(6, 2, 10))
    ft_m = main[FTID]
    assert ft_m["by_tier"][1]["element_f1|micro|zeros_for_failed"]["point"] == pytest.approx(f1(8, 2, 8))
    assert ft_m["by_tier"][4]["element_f1|micro|zeros_for_failed"]["point"] == pytest.approx(f1(12, 3, 4))
    assert ft_m["by_type"]["class"]["element_f1|micro|zeros_for_failed"]["point"] == pytest.approx(f1(28, 7, 4))
    assert ft_m["by_type"]["sequence"]["element_f1|micro|zeros_for_failed"]["point"] == pytest.approx(f1(24, 6, 8))
    # the type-margin deltas are the margin points' differences
    d = res["main_table_deltas_by_type"]["2B"]["class"]["element_f1|micro|zeros_for_failed"]
    assert d == pytest.approx(f1(28, 7, 4) - f1(15, 5, 17))


def test_the_v1_companion_holds_element_relationship_and_type_accuracy_under_v1(hand):
    res, _ = hand
    comp = res["v1_companion"]
    ids = {"element_f1|micro|zeros_for_failed", "element_f1|micro|compiled_only",
           "relationship_f1|micro|zeros_for_failed", "relationship_f1|micro|compiled_only",
           "type_accuracy|accuracy|compiled_only"}
    for cell in comp.values():
        assert set(cell["overall"]) == ids
        assert set(cell["by_type"]) == {"class", "sequence"} and set(cell["by_tier"]) == {1, 2, 3, 4}
    z, f = comp[ZS], comp[FTID]
    assert z["overall"]["element_f1|micro|compiled_only"]["point"] == pytest.approx(f1(27, 18, 9))
    assert z["overall"]["element_f1|micro|zeros_for_failed"]["point"] == pytest.approx(f1(29, 18, 35))
    assert z["overall"]["relationship_f1|micro|compiled_only"]["point"] == pytest.approx(f1(9, 0, 9))
    assert z["overall"]["type_accuracy|accuracy|compiled_only"]["point"] == pytest.approx(18 / 27)
    assert z["by_type"]["class"]["element_f1|micro|zeros_for_failed"]["point"] == pytest.approx(f1(15, 10, 17))
    assert f["overall"]["element_f1|micro|compiled_only"]["point"] == pytest.approx(f1(52, 26, 0))
    assert f["by_tier"][4]["element_f1|micro|zeros_for_failed"]["point"] == pytest.approx(f1(12, 6, 4))
    # v1 differs from v2 where the fixture says it does, and only there
    assert z["overall"]["element_f1|micro|compiled_only"]["point"] != \
        res["main_table"][ZS]["overall"]["element_f1|micro|compiled_only"]["point"]
    assert z["overall"]["relationship_f1|micro|zeros_for_failed"]["point"] == \
        res["main_table"][ZS]["overall"]["relationship_f1|micro|zeros_for_failed"]["point"]


# --------------------------------------------------------------------------- #
# Failures 2-5
# --------------------------------------------------------------------------- #
def test_the_headroom_share_is_the_delta_over_the_zero_shot_headroom(hand):
    res, _ = hand
    e, r = res["headroom"]["2B"]["element_f1"], res["headroom"]["2B"]["relationship_f1"]
    assert e["zs_compiled_only"] == pytest.approx(0.75)
    assert e["delta_compiled_only"] == pytest.approx(8 / 9 - 3 / 4)
    assert e["share_closed"] == pytest.approx((8 / 9 - 3 / 4) / (1 - 3 / 4))        # 5/9
    assert e["share_closed"] == pytest.approx(5 / 9)
    assert r["zs_compiled_only"] == pytest.approx(2 / 3) and r["share_closed"] == pytest.approx(1.0)
    # a zero-shot run already at 1 where it compiles has no headroom: undefined, not a number
    e9 = res["headroom"]["9B"]["element_f1"]
    assert e9["zs_compiled_only"] == pytest.approx(1.0) and e9["share_closed"] is None


def test_the_joint_set_is_the_diagrams_both_runs_compiled(hand):
    res, v2 = hand
    z, f = v2[ZS], v2[FTID]
    members = {z[i]["key"] for i in range(N) if M.joint(z, f, [i])["n_joint"] == 1}
    assert members == {f"k{i:04d}" for i in JOINT}
    assert members == {"k0000", "k0001", "k0004", "k0006", "k0008", "k0010", "k0012", "k0014"}
    j = res["joint_set"]["2B"]
    assert j["n_joint"] == 8 and (j["n_compiled_zs"], j["n_compiled_ft"]) == (9, 13)
    assert j["element_f1"]["zs"] == pytest.approx(f1(24, 8, 8))
    assert j["element_f1"]["ft"] == pytest.approx(f1(32, 8, 0))
    assert j["element_f1"]["point"] == pytest.approx(8 / 9 - 3 / 4)
    assert j["relationship_f1"]["point"] == pytest.approx(1 - 2 / 3)
    assert j["element_f1"]["ci"] is not None
    assert res["phase0"]["2B"]["n_joint"] == 8


def test_the_joint_set_delta_uses_only_the_joint_diagrams():
    """Make the diagrams only the fine-tuned run compiled (5, 7, 9, 11, 15) poor answers,
    (2, 2, 2): the fine-tuned run's own compiled-only F1 drops, its joint-set F1 does not."""
    ft = [hand_row(i, True, (2, 2, 2), (1, 1, 1), 80.0, (2, 1)) if i in FT_COMPILED - ZS_COMPILED else r
          for i, r in enumerate(ft_rows())]
    v2 = {}
    for size in M.SIZES:
        z, f = M.PAIRS[size]
        v2[z], v2[f] = zs_rows(), ft
    res = M.analyse(v2, {mid: v1_of(rows) for mid, rows in v2.items()}, n_resamples=20, zero_edge_expected=0)
    j, h = res["joint_set"]["2B"], res["headline"]["2B"]
    assert j["n_joint"] == 8
    assert j["element_f1"]["ft"] == pytest.approx(f1(32, 8, 0))
    assert j["element_f1"]["point"] == pytest.approx(f1(32, 8, 0) - f1(24, 8, 8))
    assert j["relationship_f1"]["point"] == pytest.approx(f1(16, 0, 0) - f1(8, 0, 8))
    assert h["element_f1"]["co"]["ft"] == pytest.approx(f1(32 + 10, 8 + 10, 10))   # 8 joint + 5 fine-tuned-only
    assert h["element_f1"]["co"]["point"] != pytest.approx(j["element_f1"]["point"])


def test_the_headline_states_both_compiled_counts(hand):
    res, _ = hand
    for size in M.SIZES:
        h = res["headline"][size]
        assert (h["n_compiled_zs"], h["n_compiled_ft"]) == (9, 13)
    assert res["runs"][ZS]["n_compiled"] == 9 and res["runs"][FTID]["n_compiled"] == 13
    assert "| 2B | 9 / 13 |" in M.markdown(res, {mid: mid for mid in res["runs"]})


def test_the_failed_equals_zero_table_gives_each_size_both_deltas(hand):
    res, _ = hand
    d = res["failed0"]["deltas"]["2B"]
    assert set(d) == {"element_f1|micro|zeros_for_failed", "relationship_f1|micro|zeros_for_failed",
                      "element_f1|macro|zeros_for_failed", "relationship_f1|macro|zeros_for_failed",
                      "gap|element_f1|micro", "gap|relationship_f1|micro",
                      "gap|element_f1|macro", "gap|relationship_f1|macro"}
    ft_full = f1(52, 13, 12)
    el = d["element_f1|micro|zeros_for_failed"]
    assert el["stored"]["point"] == pytest.approx(ft_full - f1(29, 9, 35))      # key 3 keeps its parsed tp
    assert el["failed0"]["point"] == pytest.approx(ft_full - f1(27, 9, 37))     # key 3 scored 0
    gap = d["gap|element_f1|micro"]
    ft_gap = f1(52, 13, 0) - ft_full
    assert gap["stored"]["point"] == pytest.approx(ft_gap - (f1(27, 9, 9) - f1(29, 9, 35)))
    assert gap["failed0"]["point"] == pytest.approx(ft_gap - (f1(27, 9, 9) - f1(27, 9, 37)))
    mac = d["element_f1|macro|zeros_for_failed"]
    ft_macro = 13 * f1(4, 1, 0) / 16
    assert mac["stored"]["point"] == pytest.approx(ft_macro - (9 * 0.75 + f1(2, 0, 2)) / 16)
    assert mac["failed0"]["point"] == pytest.approx(ft_macro - 9 * 0.75 / 16)
    rel = d["relationship_f1|micro|zeros_for_failed"]                           # no failed credit there
    assert rel["stored"]["point"] == pytest.approx(rel["failed0"]["point"])
    for c in d.values():
        assert c["stored"]["ci"] is not None and c["failed0"]["ci"] is not None
    # the stored Δ is the headline's
    assert el["stored"]["point"] == pytest.approx(res["headline"]["2B"]["element_f1"]["full"]["point"])


# --------------------------------------------------------------------------- #
# Failure 6: the per-relation floor, condition by condition
#
# 200 diagrams. GT edges / diagrams holding the relation:
#   inheritance  99 / 34   rows 0-33   (edges alone below)
#   composition 120 / 29   rows 34-62  (diagrams alone below)
#   aggregation 100 / 30   rows 63-92  (exactly at the floor: passes)
#   dependency  120 / 40   rows 93-132; the zero-shot run fails rows 126-132, so its compiled
#                          population holds 99 / 33 (zero-shot alone below, on edges)
#   association 120 / 40   rows 133-172; the fine-tuned run fails rows 166-172 (fine-tuned alone
#                          below, on edges)
#   message     200 / 100  rows 100-199
# --------------------------------------------------------------------------- #
def floor_rows(failed):
    gt = [dict() for _ in range(200)]
    for i in range(0, 34):
        gt[i]["inheritance"] = 3 if i <= 30 else 2
    for i in range(34, 63):
        gt[i]["composition"] = 5 if i <= 37 else 4
    for i in range(63, 93):
        gt[i]["aggregation"] = 4 if i < 73 else 3
    for i in range(93, 133):
        gt[i]["dependency"] = 3
    for i in range(133, 173):
        gt[i]["association"] = 3
    for i in range(100, 200):
        gt[i]["message"] = 2
    rows = []
    for i, edges in enumerate(gt):
        compiled = i not in failed
        by_rel = {rl: {"tp": 0, "fp": 0, "fn": 0} for rl in M.RELATIONS}
        for rl, n in edges.items():
            if not compiled:
                by_rel[rl] = {"tp": 0, "fp": 0, "fn": n}
            elif i % 2:
                by_rel[rl] = {"tp": n - 1, "fp": 1, "fn": 1}
            else:
                by_rel[rl] = {"tp": n, "fp": 0, "fn": 0}
        tp, fp, fn = (sum(c[k] for c in by_rel.values()) for k in ("tp", "fp", "fn"))
        key = f"k{i:04d}"
        rows.append({"key": key, "primary_type": "class" if i % 2 == 0 else "sequence", "tier": 1 + i // 50,
                     "csr": {"key": key, "compiled": compiled},
                     "element": dict(metric(1 if compiled else 0, 0, 0 if compiled else 1, compiled,
                                            ta={"matched": 1, "correct": 1, "excluded": 0}), key=key),
                     "relationship": dict(metric(tp, fp, fn, compiled, by_relation=by_rel), key=key),
                     "chrf": {"key": key, "compiled": compiled, "score": 50.0 if compiled else 0.0}})
    return rows


@pytest.fixture(scope="module")
def floors():
    v2 = {}
    for size in M.SIZES:
        z, f = M.PAIRS[size]
        v2[z] = floor_rows(failed=set(range(126, 133)))
        v2[f] = floor_rows(failed=set(range(166, 173)))
    return M.analyse(v2, v2, n_resamples=50, zero_edge_expected=0)["per_relation"]["2B"]


def is_point_only(cell):
    return cell["delta"]["ci"] is None and cell["zs"]["ci"] is None and cell["ft"]["ci"] is None \
        and "support floor" in cell["delta"]["why_point_only"]


def has_intervals(cell):
    return cell["delta"]["ci"] is not None and cell["zs"]["ci"] is not None and cell["ft"]["ci"] is not None


def test_the_floor_on_edges_alone(floors):
    """99 edges in 34 diagrams in both runs: point only (a floor on diagrams alone would pass it)."""
    c = floors["inheritance|zeros_for_failed"]
    assert c["support_zs"] == c["support_ft"] == {"edges": 99, "diagrams": 34}
    assert is_point_only(c)


def test_the_floor_on_diagrams_alone(floors):
    """120 edges in 29 diagrams in both runs: point only (a floor on edges alone would pass it)."""
    c = floors["composition|zeros_for_failed"]
    assert c["support_zs"] == c["support_ft"] == {"edges": 120, "diagrams": 29}
    assert is_point_only(c)


def test_the_floor_is_at_least_100_edges_and_30_diagrams(floors):
    """Exactly 100 edges in 30 diagrams passes (a strict > floor would not)."""
    for pop in ("zeros_for_failed", "compiled_only"):
        c = floors[f"aggregation|{pop}"]
        assert c["support_zs"] == c["support_ft"] == {"edges": 100, "diagrams": 30}
        assert has_intervals(c)


def test_the_floor_needs_both_runs(floors):
    """One run below, the other above: point only, whichever run is below (a floor on one run,
    on either run, or on the fine-tuned run alone would pass one of the two)."""
    dep, asc = floors["dependency|compiled_only"], floors["association|compiled_only"]
    assert dep["support_zs"] == {"edges": 99, "diagrams": 33} and dep["support_ft"] == {"edges": 120, "diagrams": 40}
    assert asc["support_zs"] == {"edges": 120, "diagrams": 40} and asc["support_ft"] == {"edges": 99, "diagrams": 33}
    assert is_point_only(dep) and is_point_only(asc)
    # the same relations in the full set, where both runs hold 120 / 40, keep their intervals
    for rel in ("dependency", "association", "message"):
        c = floors[f"{rel}|zeros_for_failed"]
        assert c["support_zs"]["edges"] >= 100 and has_intervals(c)


# --------------------------------------------------------------------------- #
# Failure 7: the data root's link to the test images
# --------------------------------------------------------------------------- #
def test_the_data_roots_link_the_benchmarks_test_images(tmp_path):
    run_dir = tmp_path / "runs" / "r"
    (run_dir / "answers").mkdir(parents=True)
    for version, metric_ in [(v, m) for v in ("v2", "v1") for m in S.METRICS]:
        (run_dir / "scores" / version / metric_ / "r").mkdir(parents=True, exist_ok=True)
    root = tmp_path / "runs" / "_root"
    S.link_roots(root, run_dir, "r", S.TEST_SET, S.IMAGES)
    assert S.IMAGES == S.INSTRUMENT / "data" / "puml_images_1568"
    for version in ("v2", "v1"):
        link = root / version / "puml_images_1568"
        assert link.is_symlink() and not os.path.isabs(os.readlink(link))
        assert link.resolve() == S.IMAGES.resolve()
        assert len(list(link.glob("*.png"))) == 1000
        assert (root / version / "test_set.json").resolve() == S.TEST_SET.resolve()
        assert (root / version / "runs" / "r").resolve() == (run_dir / "answers").resolve()
    S.link_roots(root, run_dir, "r", S.TEST_SET, S.IMAGES)            # again: the same links, accepted
    with pytest.raises(S.Refused, match="links to"):
        S.link_roots(root, run_dir, "r", S.TEST_SET, tmp_path)        # another image folder: refused
