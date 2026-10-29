"""Tests for build/error_analysis.py, on synthetic inputs (no scored run is read).

Run from fine-tune/:  python3 -m pytest build/tests/test_error_analysis.py -q
"""
import collections
import sys
from pathlib import Path

FT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(FT / "build"))
import error_analysis as E  # noqa: E402

from analysis.bootstrap import make_resamples  # noqa: E402
from analysis.failure_sampler import cell_seed_key, select_cell  # noqa: E402


def csr(compiled, error=None):
    return {"compiled": compiled, "error": error}


def f1(x, has_pred=True):
    return {"f1": x, "has_pred": has_pred}


# --------------------------------------------------------------------------- outcome classes
def test_outcome_order_and_cut():
    assert E.outcome(csr(False, "no prediction for key"), f1(0), f1(0)) == "provider_drop"
    assert E.outcome(csr(False, "no PNG produced"), f1(0), f1(0)) == "compile_fail"
    assert E.outcome(csr(True), f1(0.49), f1(1)) == "compiled_low_structural"
    assert E.outcome(csr(True), f1(1), f1(0.49)) == "compiled_low_structural"
    assert E.outcome(csr(True), f1(0.5), f1(0.5)) == "ok"


def test_has_pred_false_is_not_a_drop():
    # the held-back extractor-crash answer: a model answer that does not render
    assert E.outcome(csr(False, "no PNG produced"), f1(0, False), f1(0, False)) == "compile_fail"


def row(cls, el=1.0, rel=1.0, no_end=False, fr="stop"):
    return {"class": cls, "el": {"f1": el}, "rel": {"f1": rel}, "no_enduml": no_end, "finish_reason": fr}


def test_counts_per_1000_and_shares():
    t = {f"k{i}": row("ok") for i in range(1000)}
    for i in range(10):
        t[f"k{i}"] = row("compile_fail", 0, 0, no_end=i < 4, fr="length" if i < 3 else "stop")
    for i in range(10, 20):
        t[f"k{i}"] = row("compiled_low_structural", el=0.3 if i < 12 else 0.9, rel=0.2 if i >= 11 else 0.9)
    c = E.counts(t)
    assert c["compile_fail"] == 10 and c["c2_no_enduml"] == 4 and c["c2_finish_length"] == 3
    assert c["compiled_low_structural"] == 10
    assert (c["low_element_only"], c["low_relationship_only"], c["low_both"]) == (1, 8, 1)
    assert c["ok"] == 980 and c["failures"] == 20
    assert c["share_of_failures"]["compile_fail"] == 0.5


def test_transitions_sum_to_classes():
    tz = {"a": row("compile_fail"), "b": row("ok"), "c": row("compiled_low_structural")}
    tf = {"a": row("ok"), "b": row("ok"), "c": row("compile_fail")}
    t = E.transitions(tz, tf)
    assert t["compile_fail"]["ok"] == 1 and t["ok"]["ok"] == 1 and t["compiled_low_structural"]["compile_fail"] == 1
    assert sum(sum(r.values()) for r in t.values()) == 3


# --------------------------------------------------------------------------- the sample
def test_draw_one_per_cell_with_the_benchmark_seed():
    meta = {f"k{i}": {"primary_type": "class" if i % 2 else "sequence", "tier": 1 + (i // 2) % 4} for i in range(96)}
    tables = {}
    for run in E.RUN_ORDER:
        tables[run] = {k: row("compile_fail" if i % 3 == 0 else "compiled_low_structural" if i % 3 == 1 else "ok")
                       for i, k in enumerate(meta)}
    cases = E.draw_sample(tables, meta)
    cells = collections.Counter((c["run"], c["primary_type"], c["tier"], c["class"]) for c in cases)
    assert max(cells.values()) == 1
    assert len(cases) == len(cells) == 6 * 16
    # the case chosen in a cell is select_cell's, with cell_seed_key(run, type, tier, class)
    c0 = next(c for c in cases if c["run"] == "9B_20000" and c["primary_type"] == "class" and c["tier"] == 2
              and c["class"] == "compile_fail")
    pool = [{"key": k} for k, r in tables["9B_20000"].items()
            if meta[k]["primary_type"] == "class" and meta[k]["tier"] == 2 and r["class"] == "compile_fail"]
    assert select_cell(pool, 1, cell_seed_key("9B_20000", "class", 2, "compile_fail"))[0]["key"] == c0["key"]
    ids = [c["case"] for c in cases]
    assert len(set(ids)) == len(ids) and ids == sorted(ids)
    # blind ids are shuffled: not in run order
    assert [c["run"] for c in cases] != sorted((c["run"] for c in cases), key=E.RUN_ORDER.index)
    assert E.draw_sample(tables, meta) == cases  # deterministic


def test_empty_cell_is_skipped():
    meta = {"a": {"primary_type": "class", "tier": 1}, "b": {"primary_type": "class", "tier": 1}}
    tables = {run: {"a": row("compile_fail"), "b": row("ok")} for run in E.RUN_ORDER}
    cases = E.draw_sample(tables, meta)
    assert len(cases) == 6 and all(c["class"] == "compile_fail" for c in cases)


def test_isolate_and_strip_display():
    text = ("Here is the diagram:\n```plantuml\n@startuml\nskinparam classAttributeIconSize 0\n"
            "skinparam class {\n  BackgroundColor White\n}\n' a comment\n/' block\ncomment '/\n"
            "title Shop\n!define X Y\nleft to right direction\nhide empty members\n"
            "class A\nA *-- B\n@enduml\n```\nDone.")
    block = E.isolate(text)
    assert block.startswith("@startuml") and block.endswith("@enduml")
    shown, stripped = E.strip_display(block)
    assert shown == ["@startuml", "class A", "A *-- B", "@enduml"]
    assert len(stripped) == 11


def test_isolate_without_enduml_drops_fence():
    assert E.isolate("```\n@startuml\nclass A\n```") == "@startuml\nclass A"


def test_renderer_note_maps_the_error_line():
    text = "@startuml\nclass A\nA -> -> B\n@enduml"
    note = E.renderer_note(csr(False, "Error line 3 in file: /x/y.puml"), text, False)
    assert "line 3" in note and "`A -> -> B`" in note
    assert "no `@enduml`" in E.renderer_note(csr(False, "no PNG produced"), "@startuml\nx", True)


# --------------------------------------------------------------------------- matching and agreement
def e(s, t, rel, ah=None, note=False):
    return {"s": s, "t": t, "rel": rel, "ah": ah, "note": note}


def test_match_is_one_to_one_same_relation_first():
    gt = [e("a", "b", "composition"), e("a", "b", "inheritance"), e("c", "d", "association")]
    pred = [e("b", "a", "inheritance"), e("a", "b", "dependency")]
    m = E.match(gt, pred)
    assert m[1]["rel"] == "inheritance"          # same relation pairs first, either orientation
    assert m[0]["rel"] == "dependency"           # the rest pair in sorted order
    assert m[2] is None                          # no answer edge on the pair


def test_match_does_not_reuse_an_answer_edge():
    gt = [e("a", "b", "composition"), e("a", "b", "composition")]
    m = E.match(gt, [e("a", "b", "composition")])
    assert sum(x is not None for x in m) == 1


def test_orientation():
    assert E.same_orientation(e("c", "p", "inheritance"), e("c", "p", "inheritance"))
    assert not E.same_orientation(e("c", "p", "inheritance"), e("p", "c", "inheritance"))
    assert E.same_orientation(e("a", "b", "association", "target"), e("b", "a", "association", "source"))
    assert not E.same_orientation(e("a", "b", "association", "target"), e("a", "b", "association", "none"))
    assert E.same_orientation(e("a", "b", "association", "none"), e("b", "a", "association", "both"))


def rec(edges, notes=()):
    names = {x for ed in edges for x in ed[:2]} | set(notes)
    return {"nodes": [{"name": n, "type": "note" if n in notes else "class"} for n in sorted(names)],
            "edges": [{"source": s, "target": t, "relation": r, "arrowhead": "none"} for s, t, r in edges]}


def test_paired_edges_keeps_edges_matched_in_both_runs_and_drops_notes():
    gt = {"k": rec([("A", "B", "composition"), ("A", "C", "composition"), ("N", "A", "dependency")], notes=("N",))}
    pz = {"k": rec([("A", "B", "inheritance"), ("N", "A", "dependency")], notes=("N",))}
    pf = {"k": rec([("A", "B", "composition"), ("A", "C", "composition"), ("N", "A", "dependency")], notes=("N",))}
    p = E.paired_edges(gt, pz, pf, ["k"])
    assert p["k"] == [("composition", "inheritance", "composition", False, True)]


def synthetic_paired(n_diagrams, per, zs_agree, ft_agree, rel="composition"):
    out = {}
    for i in range(n_diagrams):
        rows = []
        for j in range(per):
            rows.append((rel, rel if (i * per + j) % 10 < zs_agree * 10 else "inheritance",
                         rel if (i * per + j) % 10 < ft_agree * 10 else "inheritance", True, True))
        out[f"k{i:03d}"] = rows
    return out


def test_agreement_point_only_below_the_floor():
    keys = [f"k{i:03d}" for i in range(50)]
    paired = synthetic_paired(20, 3, 0.2, 0.8)               # 60 edges in 20 diagrams
    a = E.agreement(paired, keys, make_resamples(len(keys), 40, 1))
    r = a["composition"]
    assert (r["edges"], r["diagrams"]) == (60, 20)
    assert r["ci"]["delta"]["ci"] is None and "below the floor" in r["ci"]["delta"]["why_point_only"]
    assert abs(r["ft"] - r["zs"] - r["ci"]["delta"]["point"]) < 1e-12


def test_agreement_interval_above_the_floor():
    keys = [f"k{i:03d}" for i in range(60)]
    paired = synthetic_paired(40, 3, 0.2, 0.8)               # 120 edges in 40 diagrams
    a = E.agreement(paired, keys, make_resamples(len(keys), 200, 7))
    r = a["composition"]
    assert r["edges"] == 120 and r["diagrams"] == 40
    assert abs(r["zs"] - 0.2) < 1e-9 and abs(r["ft"] - 0.8) < 1e-9
    lo, hi = r["ci"]["delta"]["ci"]
    assert 0 < lo <= 0.6 <= hi
    assert a["inheritance"]["edges"] == 0 and a["inheritance"]["zs"] is None
    assert a["_matrix"]["zs"]["composition->inheritance"] == 96


# --------------------------------------------------------------------------- coders
def test_kappa():
    k = E.kappa(["C1", "C1", "S1", "S1"], ["C1", "C1", "S1", "S1"])
    assert k["raw"] == 1.0 and k["kappa"] == 1.0
    k = E.kappa(["C1", "S1", "C1", "S1"], ["C1", "C1", "S1", "S1"])
    assert k["raw"] == 0.5 and abs(k["kappa"]) < 1e-12


def test_coding_summary_sends_disagreements_and_flags_to_the_author():
    cases = [{"case": "E1", "run": "2B_zero_shot", "condition": "zero_shot"},
             {"case": "E2", "run": "2B_20000", "condition": "fine_tuned"},
             {"case": "E3", "run": "2B_20000", "condition": "fine_tuned"}]
    codes = {"pass1": {"E1": {"primary": "C1"}, "E2": {"primary": "S1"}, "E3": {"primary": "S4", "ask_author": True}},
             "pass2": {"E1": {"primary": "C1"}, "E2": {"primary": "S2"}, "E3": {"primary": "S4"}},
             "final": {}}
    s = E.coding_summary(cases, codes)
    assert s["to_author"] == ["E2", "E3"] and s["unresolved"] == ["E2", "E3"]
    assert not s["final_complete"]
    codes["final"] = {"E2": {"primary": "S2"}, "E3": {"primary": "S4"}}
    s = E.coding_summary(cases, codes)
    assert s["final_complete"] and s["unresolved"] == []
    assert s["sample_primary_counts"]["2B_20000"]["S2"] == 1
    assert s["primary_agreement"]["fine_tuned"]["n"] == 2


def test_separability_bounds():
    on = E.cue_features("@startuml\nskinparam x\nclass A\n@enduml")
    off = E.cue_features("@startuml\nclass A\n@enduml")
    assert E.separability([on] * 10, [off] * 10)["balanced_accuracy"] == 1.0
    assert E.separability([off] * 10, [off] * 10)["balanced_accuracy"] == 0.5


def test_markdown_renders_with_points_only():
    keys = [f"k{i:03d}" for i in range(10)]
    agr = E.agreement(synthetic_paired(5, 2, 0.5, 1.0), keys, make_resamples(10, 20, 1))
    agr["n_joint"] = 5
    t = {f"k{i}": row("ok") for i in range(1000)}
    res = {"counts": {run: E.counts(t) for run in E.RUN_ORDER},
           "transitions": {s: E.transitions(t, t) for s in E.PAIRS},
           "agreement": {s: agr for s in E.PAIRS},
           "separability": {s: None for s in E.PAIRS},
           "sample": {"n_cases": 0, "coding": E.coding_summary([], {"pass1": {}, "pass2": {}, "final": {}})}}
    md = E.markdown(res)
    assert "(point)" in md and "0 intervals on Δ" in md


# --------------------------------------------------------------------------- behaviours pinned after verification
def test_a_reextracted_row_that_differs_from_the_stored_one_stops_the_run():
    import pytest
    gt = {"k": rec([("A", "B", "composition")])}
    pred = {"k": rec([("A", "B", "composition")])}
    el = {"k": {"tp": 2, "fp": 0, "fn": 0}}
    by = {r: {"tp": 0, "fp": 0, "fn": 0} for r in E.rf.RELATIONS}
    by["composition"] = {"tp": 1, "fp": 0, "fn": 0}
    rel = {"k": {"tp": 1, "fp": 0, "fn": 0, "by_relation": by}}
    E.check_graphs(gt, pred, el, rel, ["k"])                      # equal: passes
    el["k"]["tp"] = 1
    with pytest.raises(SystemExit):
        E.check_graphs(gt, pred, el, rel, ["k"])


def test_the_report_refuses_a_changed_codebook(tmp_path):
    import argparse
    import json
    import pytest
    (tmp_path / "keys.json").write_text(json.dumps({"codebook_sha256": "0" * 64}))
    with pytest.raises(SystemExit, match="codebook changed"):
        E.cmd_report(argparse.Namespace(sample=str(tmp_path), out=str(tmp_path / "out")))


def test_a_case_packet_hides_run_condition_and_scores_the_diff():
    case = {"case": "E07", "run": "9B_20000", "condition": "fine_tuned", "key": "abc123",
            "primary_type": "class", "tier": 2, "class": "compiled_low_structural"}
    diff = E.structural_diff(rec([("A", "B", "composition")]), rec([("A", "B", "inheritance")]))
    md = E.case_markdown(case, "@startuml\nA *-- B\n@enduml", ["@startuml", "A --|> B", "@enduml"], 2, "", diff,
                         1.0, 0.0)
    for hidden in ("9B", "20000", "fine", "zero", "abc123"):
        assert hidden not in md
    assert "edges missed (1): `a -> b : composition`" in md
    assert "edges spurious (1): `a -> b : inheritance`" in md
    cf = dict(case, **{"class": "compile_fail"})
    assert "## Renderer" in E.case_markdown(cf, "x", ["x"], 0, "Renderer: no PNG produced", None, 0, 0)


def test_edge_names_go_through_the_v2_normalizer():
    r = {"nodes": [{"name": "Order  <<entity>>", "type": "class"}, {"name": "N1", "type": "note"}],
         "edges": [{"source": "Order  <<entity>>", "target": "Line\tItem", "relation": "composition"},
                   {"source": "N1", "target": "Order  <<entity>>", "relation": "dependency"}]}
    es = E.edges_of(r)
    assert (es[0]["s"], es[0]["t"], es[0]["note"]) == ("order", "line item", False)
    assert es[1]["note"] is True


def test_messages_are_reported_apart_from_the_class_relations():
    keys = [f"k{i:03d}" for i in range(40)]
    paired = synthetic_paired(40, 3, 0.5, 1.0, rel="message")
    a = E.agreement(paired, keys, make_resamples(len(keys), 50, 3))
    assert a["message"]["edges"] == 120
    assert all(a[r]["edges"] == 0 for r in E.CLASS_RELATIONS)
    assert "message" not in E.CLASS_RELATIONS


def test_the_diagram_floor_alone_gives_a_point():
    keys = [f"k{i:03d}" for i in range(30)]
    paired = synthetic_paired(20, 6, 0.2, 0.8)                     # 120 edges, only 20 diagrams
    r = E.agreement(paired, keys, make_resamples(len(keys), 40, 1))["composition"]
    assert r["edges"] >= E.FLOOR_EDGES and r["diagrams"] < E.FLOOR_DIAGRAMS
    assert r["ci"]["delta"]["ci"] is None


def test_the_r_row_is_kept():
    t = {f"k{i}": row("ok") for i in range(1000)}
    t["k0"] = row("provider_drop")
    assert E.counts(t)["provider_drop"] == 1
    assert "R" in E.CODES and E.CODES[0] == "R"
    s = E.coding_summary([], {"pass1": {}, "pass2": {}, "final": {}})
    assert all(v["R"] == 0 for v in s["sample_primary_counts"].values())


def test_every_failure_row_has_a_share():
    t = {f"k{i}": row("ok") for i in range(1000)}
    for i in range(4):
        t[f"k{i}"] = row("compile_fail", 0, 0, no_end=True, fr="length")
    for i in range(4, 8):
        t[f"k{i}"] = row("compiled_low_structural", el=0.9, rel=0.1)
    s = E.counts(t)["share_of_failures"]
    assert set(s) == set(E.FAILURE_ROWS)
    assert s["c2_no_enduml"] == 0.5 and s["low_relationship_only"] == 0.5 and s["low_both"] == 0.0
