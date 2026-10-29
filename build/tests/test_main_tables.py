"""Tests for build/main_tables.py, on small synthetic panels (no scored run is read).

Run from fine-tune/:  python3 -m pytest build/tests/test_main_tables.py -q
"""
import json
import random
import sys
from pathlib import Path

import pytest

FT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(FT / "build"))
import main_tables as M  # noqa: E402

from analysis.bootstrap import make_resamples  # noqa: E402  (main_tables put the benchmark on the path)

TEST_SET = M.INSTRUMENT / "data" / "test_set.json"


# --------------------------------------------------------------------------- #
# synthetic rows, in the loader's joined form
# --------------------------------------------------------------------------- #
def prf(tp, fp, fn):
    if tp + fp + fn == 0:
        return 1.0, 1.0, 1.0                      # the runners' empty-both convention
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def metric(tp, fp, fn, compiled, by_relation=None, ta=None):
    p, r, f = prf(tp, fp, fn)
    m = {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f, "compiled": compiled,
         "has_pred": True}
    if by_relation is not None:
        m["by_relation"] = by_relation
    if ta is not None:
        m["type_accuracy"] = ta
    return m


def row(key, ptype, tier, compiled, el, rel, relation, chrf=50.0):
    by_rel = {rl: {"tp": 0, "fp": 0, "fn": 0} for rl in M.RELATIONS}
    by_rel[relation] = {"tp": rel[0], "fp": rel[1], "fn": rel[2]}
    return {"key": key, "primary_type": ptype, "tier": tier,
            "csr": {"key": key, "compiled": compiled},
            "element": dict(metric(*el, compiled, ta={"matched": el[0], "correct": el[0], "excluded": 0}), key=key),
            "relationship": dict(metric(*rel, compiled, by_relation=by_rel), key=key),
            "chrf": {"key": key, "compiled": compiled, "score": chrf}}


def run_rows(seed, p_compile, quality, n_per_cell=25, compiled_in=None, empty_gt=()):
    """A synthetic run: 2 types x 4 tiers x n_per_cell diagrams. compiled_in: {tier: exact
    number of compiled diagrams in that tier} overrides p_compile there. empty_gt: keys whose
    ground truth has no edges."""
    rng = random.Random(seed)
    rows, i = [], 0
    for tier in M.TIERS:
        cell_index = 0
        for ptype in M.TYPES:
            for _ in range(n_per_cell):
                key = f"k{i:04d}"
                if compiled_in and tier in compiled_in:
                    compiled = cell_index < compiled_in[tier]
                else:
                    compiled = rng.random() < p_compile
                cell_index += 1
                gt_el, gt_rel = 2 + tier, (0 if key in empty_gt else 1 + tier)
                tp_el = sum(rng.random() < quality for _ in range(gt_el)) if compiled else 0
                tp_rel = sum(rng.random() < quality for _ in range(gt_rel)) if compiled else 0
                fp = rng.randint(0, 1) if compiled else 0
                relation = ("message" if ptype == "sequence" else
                            ("composition" if i % 10 == 0 else "association"))
                rows.append(row(key, ptype, tier, compiled, (tp_el, fp, gt_el - tp_el),
                                (tp_rel, fp, gt_rel - tp_rel), relation, chrf=100 * quality * compiled))
                i += 1
    return rows


def panel(**override):
    """Six runs (v2) and their v1 copies (one fp more on every compiled class row)."""
    spec = {"2B": (0.3, 0.6, 0.9, 0.85), "9B": (0.5, 0.7, 0.95, 0.9), "27B": (0.7, 0.8, 0.95, 0.9)}
    v2, v1 = {}, {}
    for k, (size, (pz, qz, pf, qf)) in enumerate(spec.items()):
        z, f = M.PAIRS[size]
        v2[z] = override.get(z) or run_rows(10 + k, pz, qz, empty_gt={"k0003", "k0007"})
        v2[f] = override.get(f) or run_rows(20 + k, pf, qf, empty_gt={"k0003", "k0007"})
    for mid, rows in v2.items():
        v1[mid] = [dict(r, element=metric(r["element"]["tp"], r["element"]["fp"] + (r["primary_type"] == "class"
                                                                                        and r["csr"]["compiled"]),
                                          r["element"]["fn"], r["csr"]["compiled"],
                                          ta=r["element"]["type_accuracy"]))
                   for r in rows]
    return v2, v1


@pytest.fixture(scope="module")
def result():
    v2, v1 = panel()
    return M.analyse(v2, v1, n_resamples=60, zero_edge_expected=2)


# --------------------------------------------------------------------------- #
# tests
# --------------------------------------------------------------------------- #
def test_the_decomposition_identity_holds_in_the_point_and_every_resample(result):
    assert result["identity_max_abs_deviation"] < 1e-12
    for size in M.SIZES:
        for name in ("element_f1", "relationship_f1"):
            h = result["headline"][size][name]
            assert h["full"]["point"] == pytest.approx(h["co"]["point"] - h["gap"]["point"], abs=1e-12)


def test_the_differences_of_deltas_are_linearly_dependent(result):
    assert result["dod_linear_dependence_max_abs_deviation"] < 1e-12
    d = result["dod"]
    for name in ("csr", "element_f1_co", "element_f1_gap", "relationship_f1_co"):
        assert d[f"2B-27B|{name}"]["point"] == pytest.approx(
            d[f"2B-9B|{name}"]["point"] + d[f"9B-27B|{name}"]["point"], abs=1e-12)
    h = result["headline"]
    assert d["2B-27B|csr"]["point"] == pytest.approx(h["2B"]["csr"]["point"] - h["27B"]["csr"]["point"])


def test_one_draw_is_shared_by_every_run():
    """A fine-tuned run equal to its zero-shot run gives Δ = 0 in every resample, which holds only
    when both runs are resampled with the same indices."""
    v2, v1 = panel()
    z, f = M.PAIRS["2B"]
    v2[f], v1[f] = v2[z], v1[z]
    res = M.analyse(v2, v1, n_resamples=40, zero_edge_expected=2)
    for c in (res["headline"]["2B"]["csr"], res["headline"]["2B"]["element_f1"]["co"],
              res["headline"]["2B"]["relationship_f1"]["gap"], res["joint_set"]["2B"]["element_f1"]):
        assert c["point"] == 0 and c["ci"] == [0, 0]
    assert make_resamples(100, 5, M.SEED) == make_resamples(100, 5, M.SEED)


def test_the_claim_family_has_twelve_members(result):
    fam = result["claim_family"]
    assert len(fam) == 12
    assert sum("Δ CSR" in c["member"] and "difference" not in c["member"] for c in fam) == 3
    assert sum("difference" in c["member"] for c in fam) == 6
    assert not any("gap" in c["member"] or "Relationship" in c["member"] for c in fam)


def test_a_statistic_on_an_empty_population_is_undefined_not_zero():
    rows = run_rows(1, 0.0, 0.5)                     # nothing compiles
    s = M.stats(rows)
    for st in ("element_f1|micro|compiled_only", "element_f1|macro|compiled_only",
               "relationship_f1|micro|compiled_only", "chrf|macro|compiled_only",
               "type_accuracy|accuracy|compiled_only", "gap|element_f1|micro"):
        assert s[st] is None, st
    assert s["csr|csr"] == 0.0 and s["element_f1|micro|zeros_for_failed"] == 0.0
    assert all(v is None for k, v in M.stats([]).items() if k not in ("n", "n_compiled"))


def test_a_cell_with_an_undefined_resample_gets_a_point_only():
    c = M.interval(0.5, [0.4, None, 0.6])
    assert c["ci"] is None and c["n_undefined"] == 1 and c["point"] == 0.5
    assert M.interval(None, [0.1, 0.2])["ci"] is None
    assert M.interval(0.1, [0.0, 0.1, 0.2])["ci"] == [pytest.approx(0.005), pytest.approx(0.195)]


def test_the_compiled_only_tier_floor():
    """29 compiled in tier 4 of the 2B zero-shot run: a point only; 30 at the 9B: an interval."""
    z2, z9 = M.PAIRS["2B"][0], M.PAIRS["9B"][0]
    v2, v1 = panel(**{z2: run_rows(10, 0.3, 0.6, n_per_cell=50, compiled_in={4: 29}, empty_gt={"k0003", "k0007"}),
                      z9: run_rows(11, 0.5, 0.7, n_per_cell=50, compiled_in={4: 30}, empty_gt={"k0003", "k0007"})})
    for mid, rows in v2.items():                    # every run on the 400-row panel
        if len(rows) != 400:
            v2[mid] = run_rows(hash(mid) % 97, 0.9, 0.9, n_per_cell=50, empty_gt={"k0003", "k0007"})
            v1[mid] = v2[mid]
    v1[z2], v1[z9] = v2[z2], v2[z9]
    res = M.analyse(v2, v1, n_resamples=60, zero_edge_expected=2)
    t2, t9 = res["tiers"]["2B"][4], res["tiers"]["9B"][4]
    assert t2["n_compiled_zs"] == 29 and t9["n_compiled_zs"] == 30
    assert t2["element_f1_co"]["ci"] is None and "fewer than 30" in t2["element_f1_co"]["why_point_only"]
    assert t2["element_f1_co"]["point"] is not None
    assert t9["element_f1_co"]["ci"] is not None
    assert t2["csr"]["ci"] is not None                # full-set tier cells keep their interval
    assert res["tiers"]["2B"]["tier4_minus_tier1"]["element_f1_co"]["ci"] is None
    assert res["tiers"]["2B"]["tier4_minus_tier1"]["csr"]["ci"] is not None


def test_an_empty_compiled_tier_is_undefined_not_zero():
    z = M.PAIRS["2B"][0]
    v2, v1 = panel(**{z: run_rows(10, 0.3, 0.6, compiled_in={4: 0}, empty_gt={"k0003", "k0007"})})
    v1[z] = v2[z]
    res = M.analyse(v2, v1, n_resamples=20, zero_edge_expected=2)
    cell = res["tiers"]["2B"][4]["element_f1_co"]
    assert cell["zs"] is None and cell["point"] is None and cell["ci"] is None


def test_the_per_relation_support_floor(result):
    """Composition sits in 1 of 10 class diagrams (fewer than 30 diagrams, 100 edges): point only.
    Message is in every sequence diagram (100 diagrams, 350 edges): an interval in the full set."""
    comp = result["per_relation"]["2B"]["composition|zeros_for_failed"]
    msg = result["per_relation"]["2B"]["message|zeros_for_failed"]
    assert comp["support_zs"]["diagrams"] < 30 and comp["delta"]["ci"] is None
    assert "support floor" in comp["delta"]["why_point_only"] and comp["zs"]["ci"] is None
    assert msg["support_zs"] == {"edges": 350, "diagrams": 100} and msg["delta"]["ci"] is not None
    rows = run_rows(3, 1.0, 0.5)
    assert M.relation_support(rows, "message", "zeros_for_failed") == {"edges": 350, "diagrams": 100}


def test_the_zero_edge_diagrams_are_derived_from_the_ground_truth_and_left_out(result):
    v2, v1 = panel()
    mid = M.PAIRS["9B"][1]
    assert M.zero_edge_keys(v2[mid]) == {"k0003", "k0007"}
    keep = [r["relationship"] for r in v2[mid] if r["key"] not in {"k0003", "k0007"}]
    expected = sum(r["f1"] for r in keep) / len(keep)
    assert result["macro"]["runs"][mid][M.NE["zeros_for_failed"]]["point"] == pytest.approx(expected)
    with pytest.raises(ValueError, match="expected 107"):
        M.analyse(v2, v1, n_resamples=2)


def test_the_failed_equals_zero_variant_is_a_sensitivity_only():
    rows = run_rows(5, 0.4, 0.8, empty_gt={"k0003", "k0007"})
    failed = next(r for r in rows if not r["csr"]["compiled"])
    failed["element"].update(tp=2, fn=failed["element"]["fn"] - 2, f1=1.0)      # a parsed failed answer
    z = M.failed_zero(failed)["element"]
    assert (z["tp"], z["fp"], z["fn"], z["f1"]) == (0, 0, failed["element"]["tp"] + failed["element"]["fn"], 0.0)
    masked = M.stats([M.failed_zero(r) for r in rows])
    # with failed = 0, macro full-set = CSR x macro compiled-only exactly
    assert masked["element_f1|macro|zeros_for_failed"] == pytest.approx(
        masked["csr|csr"] * masked["element_f1|macro|compiled_only"])
    stored = M.stats(rows)
    assert stored["element_f1|micro|zeros_for_failed"] > masked["element_f1|micro|zeros_for_failed"]
    # the analysis keeps the stored rows in the headline and shows failed = 0 beside them
    z2 = M.PAIRS["2B"][0]
    v2, v1 = panel(**{z2: rows})
    v1[z2] = rows
    res = M.analyse(v2, v1, n_resamples=10, zero_edge_expected=2)
    f0 = res["failed0"]["runs"][z2]["stats"]["element_f1|micro|zeros_for_failed"]
    assert f0["stored"] == pytest.approx(stored["element_f1|micro|zeros_for_failed"])
    assert f0["failed0"] == pytest.approx(masked["element_f1|micro|zeros_for_failed"])
    assert res["headline"]["2B"]["element_f1"]["full"]["zs"] == pytest.approx(f0["stored"])
    assert res["failed0"]["runs"][z2]["failed_with_credit"]["element"]["tp_gt_0"] == 1


def test_phase0_outcome_labels():
    assert M.phase0_outcome([-0.02, -0.001]) == "held"
    assert M.phase0_outcome([-0.02, 0.0]) == "no detectable change"
    assert M.phase0_outcome([-0.01, 0.01]) == "no detectable change"
    assert M.phase0_outcome([0.0, 0.0]) == "no detectable change"
    assert M.phase0_outcome([0.001, 0.02]) == "did not hold"
    assert M.phase0_outcome(None) == "undefined"


def test_phase0_is_read_on_the_joint_set(result):
    for size in M.SIZES:
        p = result["phase0"][size]
        assert p["n_joint"] == result["joint_set"][size]["n_joint"]
        for name in ("element_f1", "relationship_f1"):
            assert p[name]["delta_v2"] == pytest.approx(result["joint_set"][size][name]["point"])
            assert p[name]["outcome"] == M.phase0_outcome(p[name]["ci"])
        assert set(p["beside"]) == {f"{n}|{q}" for n in ("element_f1", "relationship_f1") for q in ("full", "co")}
    # v1 adds a false positive to compiled class rows only: Element F1 Δ changes, Relationship F1 Δ does not
    assert result["phase0"]["2B"]["relationship_f1"]["point"] == pytest.approx(0)


def test_the_markdown_and_figure_are_written(result, tmp_path):
    displays = {mid: mid for mid in result["runs"]}
    md = M.markdown(result, displays)
    for heading in ("## 1. Headline", "## 2. Claim family", "## 11. Phase 0", "## S1.", "## S2."):
        assert heading in md
    M.figure(result, tmp_path / "f.png")
    assert (tmp_path / "f.png").stat().st_size > 10000


def test_the_benchmarks_analysis_out_is_refused():
    with pytest.raises(SystemExit, match="analysis/out"):
        M.main(["--out", str(M.INSTRUMENT / "analysis" / "out" / "x")])


# --------------------------------------------------------------------------- #
# loading through the benchmark's loader and leak gate
# --------------------------------------------------------------------------- #
def write_root(root):
    """A data root of six synthetic runs over the 1,000 test-set keys, in the loader's layout."""
    keys = sorted(d["key"][:-5] for d in json.loads(TEST_SET.read_text())["diagrams"])
    reg = M.load_registry(M.REGISTRY)
    for version in ("v2", "v1"):
        r = root / version
        r.mkdir(parents=True)
        (r / "test_set.json").symlink_to(TEST_SET)
        for e in reg:
            rows = [row(k, "class", 1, True, (1, 0, 0), (1, 0, 0), "association") for k in keys]
            for name, field in (("csr", "csr"), ("element_f1", "element"), ("relationship_f1", "relationship"),
                                ("chrf", "chrf")):
                d = r / name / e.run_dir
                d.mkdir(parents=True)
                (d / f"{name}_results.json").write_text(json.dumps(
                    {"summary": {}, "diagrams": [x[field] for x in rows]}))
            p = r / "runs" / e.run_dir
            p.mkdir(parents=True)
            for k in keys:
                (p / (k + ".puml")).write_text("@startuml\n@enduml\n")
    return reg


def test_load_reads_six_runs_and_refuses_a_run_the_leak_gate_cannot_see(tmp_path):
    reg = write_root(tmp_path / "root")
    out = M.load(tmp_path / "root")
    assert list(out["v2"]) == [e.id for e in reg] and len(out["v1"][reg[0].id]) == 1000
    pdir = tmp_path / "root" / "v1" / "runs" / reg[3].run_dir
    for p in pdir.glob("*.puml"):
        p.unlink()
    pdir.rmdir()
    with pytest.raises(SystemExit, match="pass silently"):
        M.load(tmp_path / "root")


def test_load_refuses_a_run_with_reasoning_leaks(tmp_path):
    reg = write_root(tmp_path / "root")
    first = sorted((tmp_path / "root" / "v2" / "runs" / reg[0].run_dir).glob("*.puml"))[0]
    first.write_text("<think>a</think>@startuml\n@enduml\n")
    with pytest.raises(SystemExit, match="refused by the leak gate"):
        M.load(tmp_path / "root")
