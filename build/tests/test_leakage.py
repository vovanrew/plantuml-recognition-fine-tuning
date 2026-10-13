"""Tests for build/leakage.py: elements D5 and D6 of decisions/training-pool-rules.

Run from fine-tune/:  python3 -m pytest build/tests/test_leakage.py -q
The tests of the all-pairs computation read the corpus and the 1,000 test sources; the tests of
the artifacts read data/pool and are skipped when it does not exist.
"""
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pool as P  # noqa: E402
import leakage as L  # noqa: E402

DATA = Path(__file__).resolve().parents[2] / "data/pool"
HEADER = "' Blob ID: 0\n' Original Path: /a.puml\n' Source: World of Code\n\n"


def src(body):
    return HEADER + "@startuml\n" + body + "\n@enduml\n"


# --------------------------------------------------------------------------- #
# D5: the text
# --------------------------------------------------------------------------- #
def test_the_text_is_the_normalized_code_without_the_wrapper_lines(tmp_path):
    source = src("class A {\n  +run()\n}\n' a comment\n/' a block\n comment '/\nA  -->   B")
    f = tmp_path / "a.puml"
    f.write_text(source)
    assert P.test_set_code().normalized_code(str(f)) == "@startuml class A { +run() } A --> B @enduml"
    assert L.wrapper_free_text(source) == "class A { +run() } A --> B"


@pytest.mark.parametrize("line", ["@startuml", "@startuml diagram_name", "  @enduml  ", "@StartUML", "@ENDUML",
                                  "@startuml{scale=2}"])
def test_a_line_that_begins_with_the_wrapper_is_dropped(line):
    assert L.wrapper_free_text("class A\n" + line + "\nclass B\n") == "class A class B"


def test_a_line_that_only_holds_the_wrapper_word_stays():
    assert L.wrapper_free_text("note: see @startuml\n﻿@startuml\nclass A") == "note: see @startuml ﻿@startuml class A"


def test_a_source_of_wrapper_lines_only_has_an_empty_text():
    assert L.wrapper_free_text("@startuml\n\n@enduml\n") == ""


def test_the_kept_lines_rebuild_the_normalized_code_of_the_test_set():
    code = P.test_set_code()
    names = sorted(os.listdir(P.TEST_PUML_DIR))[::5]
    assert len(names) == 200
    for name in names:
        path = os.path.join(P.TEST_PUML_DIR, name)
        rebuilt = code._WS.sub(" ", " ".join(L.source_lines(L.read_text(path)))).strip()
        assert rebuilt == code.normalized_code(path)


# --------------------------------------------------------------------------- #
# D6: the measures on made-up pairs
# --------------------------------------------------------------------------- #
def test_a_text_shorter_than_five_characters_is_its_own_5_gram_set():
    assert L.shingles("A->B") == {"A->B"}
    assert L.shingles("") == {""}
    assert L.shingles("abcdef") == {"abcde", "bcdef"}


def test_arm_a_fires_at_a_jaccard_index_of_exactly_0_7():
    assert L.arm_a(7, 8, 9)              # 7 / 10
    assert not L.arm_a(69, 80, 88)       # 69 / 99 = 0.697
    assert L.arm_a(10, 10, 10)


def test_arm_b_fires_at_an_edit_similarity_of_exactly_0_8():
    a, b = "abcdefghij", "abcdefghXY"
    assert L.lcs_length(a, b) == 8
    assert L.arm_b(8, 10, 10)            # 2 * 8 / 20
    assert not L.arm_b(7, 10, 10)
    assert not L.arm_b(799, 1000, 1000)


def test_arm_c_needs_a_test_diagram_of_ten_distinct_lines_and_eight_tenths_of_them():
    assert L.arm_c(8, 10)
    assert not L.arm_c(7, 10)
    assert not L.arm_c(9, 9)             # all lines of a test diagram of nine lines
    assert L.arm_c(12, 15) and not L.arm_c(11, 15)


def test_an_exact_copy_fires_every_arm():
    body = "\n".join(f"class C{i}" for i in range(12))
    r = L.compare(src(body), src(body))
    assert r["arms"] == ["a", "b", "c"] and r["dropped"]
    assert r["jaccard_5gram"] == r["edit_similarity"] == r["lines_held"] == 1.0


def test_a_copy_with_one_word_changed_is_caught_by_the_5_gram_arm():
    test = "\n".join(f"Alice -> Bob: message number {i}" for i in range(12))
    cand = test.replace("number 3", "numbre 3")
    r = L.compare(src(cand), src(test))
    assert "a" in r["arms"] and r["jaccard_5gram"] >= 0.7


def test_a_respaced_copy_is_caught_by_the_edit_arm_only():
    test = "\n".join(f"A{i}->B{i}:m{i}" for i in range(12))
    cand = "\n".join(f"A{i} -> B{i} : m{i}" for i in range(12))
    r = L.compare(src(cand), src(test))
    assert r["jaccard_5gram"] < 0.7 and r["lines_held"] < 0.8
    assert r["edit_similarity"] >= 0.8 and r["arms"] == ["b"]


def test_a_test_diagram_inside_a_larger_candidate_is_caught_by_the_containment_arm_only():
    test = "\n".join(f"class Piece{i}" for i in range(10))
    extra = "\n".join(f"Controller{i} --> View{i} : draws the board {i * 7919}" for i in range(40))
    r = L.compare(src(test + "\n" + extra), src(test))
    assert r["jaccard_5gram"] < 0.7 and r["edit_similarity"] < 0.8
    assert r["lines_held"] == 1.0 and r["arms"] == ["c"]
    # eight of the ten lines are enough, seven are not
    eight = "\n".join(f"class Piece{i}" for i in range(8))
    seven = "\n".join(f"class Piece{i}" for i in range(7))
    assert L.compare(src(eight + "\n" + extra), src(test))["arms"] == ["c"]
    assert L.compare(src(seven + "\n" + extra), src(test))["arms"] == []


def test_containment_gives_no_verdict_against_a_test_diagram_of_nine_lines():
    test = "\n".join(f"class Piece{i}" for i in range(9))
    extra = "\n".join(f"Controller{i} --> View{i} : draws the board {i * 7919}" for i in range(40))
    r = L.compare(src(test + "\n" + extra), src(test))
    assert r["lines_held"] is None and r["arms"] == []


def test_lone_braces_and_the_wrapper_are_not_lines_and_a_repeated_line_counts_once():
    source = src("class A\n{\n+run()\n}\nclass A\nclass B {\n}")
    assert L.line_set(source) == {"class A", "+run()", "class B {"}
    # nine distinct lines, with braces and the wrapper eleven and more: still no verdict
    test = src("\n".join(f"class Piece{i}\n{{\n}}" for i in range(9)))
    assert len(L.line_set(test)) == 9
    assert L.compare(test, test)["lines_held"] is None


def test_whitespace_inside_a_line_is_collapsed_for_the_line_set():
    assert L.line_set("A   ->  B :  hi") == {"A -> B : hi"}


def test_two_tiny_sources_are_compared_without_the_wrapper():
    # with the wrapper these two share most of their text; without it they share nothing
    r = L.compare("@startuml\nhide circle\n@enduml", "@startuml\nclass test\n@enduml")
    assert r["jaccard_5gram"] == 0.0 and r["edit_similarity"] < 0.5 and r["arms"] == []


def test_identical_texts_shorter_than_five_characters_are_a_match():
    r = L.compare("@startuml\nA->B\n@enduml", "@startuml\nA->B\n@enduml")
    assert r["jaccard_5gram"] == 1.0 and r["arms"] == ["a", "b"]


def test_a_pair_with_an_empty_text_gets_no_verdict_from_the_first_two_arms():
    empty = "@startuml\n@enduml\n"
    other = src("class A")
    for cand, test in ((empty, other), (other, empty), (empty, empty)):
        r = L.compare(cand, test)
        assert r["jaccard_5gram"] is None and r["edit_similarity"] is None and r["arms"] == []
    # the containment arm still reads the candidate's lines
    test = src("\n".join(f"class Piece{i}" for i in range(10)))
    assert L.compare(empty, test)["lines_held"] == 0.0


def test_the_edit_measure_does_not_depend_on_the_order_of_its_arguments():
    rng = random.Random(7)
    for _ in range(200):
        a = "".join(rng.choice("abcd ->:") for _ in range(rng.randint(0, 60)))
        b = "".join(rng.choice("abcd ->:") for _ in range(rng.randint(0, 60)))
        assert L.lcs_length(a, b) == L.lcs_length(b, a) == L.lcs_length_dp(a, b)
    one = src("\n".join(f"A{i}->B{i}:m{i}" for i in range(12)))
    two = src("\n".join(f"A{i} -> B{i} : m{i}" for i in range(11)))
    forward, back = L.compare(one, two), L.compare(two, one)
    assert forward["edit_similarity"] == back["edit_similarity"]
    assert forward["jaccard_5gram"] == back["jaccard_5gram"]
    assert ("b" in forward["arms"]) == ("b" in back["arms"])


# --------------------------------------------------------------------------- #
# The computation over all pairs
# --------------------------------------------------------------------------- #
def test_the_packed_lcs_equals_the_recurrence_on_made_up_texts():
    rng = random.Random(11)
    alphabet = "ab ->:é中\U0001f600"
    texts = ["", "a", "aaaa", "abab" * 30] + [
        "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 80))) for _ in range(60)]
    packed = L.PackedLCS(texts)
    queries = ["", "zzzz", "a" * 200, "\U0001f600中"] + [
        "".join(rng.choice(alphabet + "z") for _ in range(rng.randint(0, 120))) for _ in range(40)]
    for q in queries:
        got = packed.lengths_against(q)
        assert list(got) == [L.lcs_length_dp(q, t) for t in texts]


def test_the_set_index_counts_intersections():
    sets = [set("abc"), set(), set("cde"), set("a")]
    index = L.SetIndex(sets)
    for items in (set("ac"), set(), set("xyz"), set("abcde")):
        assert list(index.intersections(items)) == [len(items & s) for s in sets]


@pytest.fixture(scope="module")
def benchmark_sources():
    test = P.load_test_set()
    keys = [d["key"] for d in test["diagrams"]]
    return keys, [L.read_text(os.path.join(P.TEST_PUML_DIR, k)) for k in keys]


@pytest.fixture(scope="module")
def sample(benchmark_sources):
    """48 corpus sources, every 3,000th file name, measured against the 1,000 test diagrams."""
    names = sorted(f for f in os.listdir(P.PUML_DIR) if f.endswith(".puml"))[::3000]
    m = L.measure(names, P.PUML_DIR, benchmark_sources[1], workers=4, chunk=4)
    return names, m


def test_all_pairs_equal_the_pair_by_pair_computation(sample, benchmark_sources):
    """Every pair of the sample: the three integers of the matrix computation against sets and
    the bit-parallel LCS of one pair, and the verdicts against compare()."""
    names, m = sample
    assert len(names) == 48 and m["lcs"].shape == (48, 1000)
    t_text = [L.wrapper_free_text(s) for s in benchmark_sources[1]]
    t_grams = [L.shingles(t) for t in t_text]
    t_lines = [L.line_set(s) for s in benchmark_sources[1]]
    arm, val = L.verdicts(m)
    pairs = 0
    for i, name in enumerate(names):
        source = L.read_text(os.path.join(P.PUML_DIR, name))
        text, lines = L.wrapper_free_text(source), L.line_set(source)
        grams = L.shingles(text)
        assert (m["cand_len"][i], m["cand_grams"][i], m["cand_lines"][i]) == (len(text), len(grams), len(lines))
        for j in range(1000):
            assert m["inter_grams"][i, j] == len(grams & t_grams[j])
            assert m["inter_lines"][i, j] == len(lines & t_lines[j])
            pairs += 1
        for j in range(0, 1000, 7):      # 143 test diagrams per candidate
            assert m["lcs"][i, j] == L.lcs_length(text, t_text[j])
        for j in range(i, 1000, 97):     # the verdict function against compare()
            r = L.compare(source, benchmark_sources[1][j])
            assert [a for a in L.ARMS if arm[a][i, j]] == r["arms"]
            for a, name_ in zip(L.ARMS, ("jaccard_5gram", "edit_similarity", "lines_held")):
                assert val[a][i, j] == pytest.approx(r[name_] or 0.0, abs=1e-12)
    assert pairs == 48000


def test_the_lcs_of_all_pairs_equals_the_textbook_recurrence_on_a_sample(sample, benchmark_sources):
    names, m = sample
    t_text = [L.wrapper_free_text(s) for s in benchmark_sources[1]]
    rng = random.Random(5)
    checked = 0
    while checked < 150:
        i, j = rng.randrange(len(names)), rng.randrange(1000)
        if m["cand_len"][i] * len(t_text[j]) > 250_000:
            continue
        text = L.wrapper_free_text(L.read_text(os.path.join(P.PUML_DIR, names[i])))
        assert m["lcs"][i, j] == L.lcs_length_dp(text, t_text[j])
        checked += 1


def test_one_process_and_several_give_the_same_matrices(sample, benchmark_sources):
    names, m = sample
    one = L.measure(names[:6], P.PUML_DIR, benchmark_sources[1], workers=1)
    for k in ("inter_grams", "lcs", "inter_lines"):
        assert (one[k] == m[k][:6]).all()


def test_no_test_diagram_has_an_empty_text(benchmark_sources):
    assert all(L.wrapper_free_text(s) for s in benchmark_sources[1])


# --------------------------------------------------------------------------- #
# The artifacts of D6
# --------------------------------------------------------------------------- #
needs_artifacts = pytest.mark.skipif(not (DATA / "d6_removed.jsonl").is_file(), reason="data/pool not built")


@needs_artifacts
def test_every_removed_candidate_is_removed_again_pair_by_pair(benchmark_sources):
    keys, sources = benchmark_sources
    index = {k: i for i, k in enumerate(keys)}
    names = {"a": "jaccard_5gram", "b": "edit_similarity", "c": "lines_held"}
    rows = P.read_jsonl(DATA / "d6_removed.jsonl")
    assert rows
    for row in rows:
        cand = L.read_text(os.path.join(P.PUML_DIR, row["key"]))
        for arm in row["arms"]:
            r = L.compare(cand, sources[index[row[f"{arm}_test_key"]]])
            assert arm in r["arms"]
            assert r[names[arm]] == pytest.approx(row[f"{arm}_value"], abs=1e-12)


@needs_artifacts
def test_a_sample_of_kept_candidates_fires_no_arm_against_any_test_diagram(benchmark_sources):
    pool = json.load(open(DATA / "pool_after_d6.json"))["diagrams"]
    picked = [d["key"] for d in pool[::len(pool) // 24]][:24]
    m = L.measure(picked, P.PUML_DIR, benchmark_sources[1], workers=4, chunk=4)
    arm, val = L.verdicts(m)
    assert not any(arm[a].any() for a in L.ARMS)
    maxima = {}
    with open(DATA / "d6_maxima.tsv") as fh:
        next(fh)
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if f[0] in picked:
                maxima[f[0]] = f[1:]
    for i, k in enumerate(picked):
        for n, a in enumerate(L.ARMS):
            assert float(maxima[k][2 * n]) == pytest.approx(val[a][i].max(), abs=1e-6)
            assert val[a][i, int(maxima[k][2 * n + 1])] == pytest.approx(val[a][i].max(), abs=1e-12)


@needs_artifacts
def test_the_removed_list_and_the_candidates_table_agree():
    removed = {r["key"]: "".join(r["arms"]) for r in P.read_jsonl(DATA / "d6_removed.jsonl")}
    table = {}
    with open(DATA / "candidates.tsv") as fh:
        head = next(fh).rstrip("\n").split("\t")
        for line in fh:
            f = dict(zip(head, line.rstrip("\n").split("\t")))
            table[f["key"]] = f
    assert {k: f["d6_arms"] for k, f in table.items() if f["d6_arms"]} == removed
    pool = {d["key"] for d in json.load(open(DATA / "pool_after_d6.json"))["diagrams"]}
    assert pool == {k for k, f in table.items() if f["in_pool"] == "1"}
    assert pool == {k for k, f in table.items() if f["d3"] == "kept" and f["d4"] == "kept" and not f["d6_arms"]}


@needs_artifacts
def test_the_counts_of_d6_and_d9_are_recounted_from_the_lists():
    funnel = json.load(open(DATA / "funnel.json"))["d6"]
    removed = P.read_jsonl(DATA / "d6_removed.jsonl")
    cell = lambda r: f"{r['primary_type']}_T{r['tier']}"
    count = lambda rows: {c: sum(1 for r in rows if cell(r) == c) for c in funnel["removed"] if c != "total"}
    assert dict(count(removed), total=len(removed)) == funnel["removed"]
    for arm in L.ARMS:
        rows = [r for r in removed if arm in r["arms"]]
        assert dict(count(rows), total=len(rows)) == funnel["flagged_by_arm"][arm]
        threshold = {"a": 0.7, "b": 0.8, "c": 0.8}[arm]
        assert all((r[f"{arm}_value"] >= threshold - 1e-12) == (arm in r["arms"]) for r in removed)
        assert all(r[f"{arm}_test_diagrams_at_or_above"] > 0 for r in rows)
    for arms, cells in funnel["removed_by_combination_of_arms"].items():
        rows = [r for r in removed if "".join(r["arms"]) == arms]
        assert dict(count(rows), total=len(rows)) == cells
    # the largest values among the kept candidates, from the table of maxima
    gone = {r["key"] for r in removed}
    best = [0.0, 0.0, 0.0]
    with open(DATA / "d6_maxima.tsv") as fh:
        next(fh)
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if f[0] not in gone:
                best = [max(b, float(f[1 + 2 * n])) for n, b in enumerate(best)]
    for n, arm in enumerate(L.ARMS):
        assert funnel["largest_value_among_kept"][arm]["value"] == pytest.approx(best[n], abs=1e-6)
    assert best[0] < 0.7 and best[1] < 0.8 and best[2] < 0.8
    # D9: class tier 1, recounted from the sources
    code = P.test_set_code()
    before = []
    with open(DATA / "candidates.tsv") as fh:
        head = next(fh).rstrip("\n").split("\t")
        for line in fh:
            f = dict(zip(head, line.rstrip("\n").split("\t")))
            if f["primary_type"] == "class" and f["tier"] == "1" and f["d4"] == "kept":
                before.append(f["key"])
    sketch = [k for k in before if P.SKETCHIT in code.normalized_code(os.path.join(P.PUML_DIR, k))]
    d9 = funnel["d9_sketchit_class"]["class_T1"]
    assert (len(before), len(sketch), sum(1 for k in sketch if k in gone)) == \
        (d9["pool_before_D6"], d9["sketchit_before_D6"], d9["sketchit_removed"])
    assert d9["sketchit_kept"] == d9["sketchit_before_D6"] - d9["sketchit_removed"]
