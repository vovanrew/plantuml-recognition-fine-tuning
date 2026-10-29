"""Tests for build/score_test_runs.py and build/test_registry.json.

Run from fine-tune/:  python3 -m pytest build/tests/test_score_test_runs.py -q
The refusal tests read file names only. The layout test scores a stand-in run, the benchmark's
published Qwen3.5-9B run (1,000 answers) relabelled, never a study test run; it needs Java,
Graphviz and the pinned JARs, and takes about half a minute (it scores twice).
"""
import json
import os
import shutil
import sys
from pathlib import Path

import pytest

FT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(FT / "build"))
import score_test_runs as S  # noqa: E402

BENCH = S.INSTRUMENT
sys.path.insert(0, str(BENCH))
from analysis.leak_gate import count_reasoning_leaks, predictions_dir  # noqa: E402
from analysis.loader import load_model  # noqa: E402
from analysis.registry import load_registry  # noqa: E402

STANDIN = BENCH / "data" / "runs" / "Qwen_Qwen3.5-9B_20260613T202001Z"   # 1,000 published answers
KEYS = S.test_set_keys(S.TEST_SET)


def make_run(runs_dir, name, n=1000, passed=True, set_sha=S.FROZEN_TEST_SET_SHA256, rename=False):
    """A stand-in run folder: links to the published answers, a run manifest, an answer manifest."""
    run = Path(runs_dir) / name
    answers = run / "answers"
    answers.mkdir(parents=True)
    for k in KEYS[:n]:
        os.symlink(STANDIN / (k + ".puml"), answers / (k + ".puml"))
    if rename:
        (answers / (KEYS[0] + ".puml")).rename(answers / "not_a_test_key.puml")
    if passed is not None:
        (answers / "manifest.json").write_text(json.dumps({"passed": passed}))
    (run / "answer_manifest.json").write_text(json.dumps({"set_file_sha256": set_sha}))
    return run


# --------------------------------------------------------------------------- #
# refusals
# --------------------------------------------------------------------------- #
def test_a_complete_passed_run_is_accepted(tmp_path):
    info = S.check_run(make_run(tmp_path, "r"), KEYS)
    assert info["set_file_sha256"] == S.FROZEN_TEST_SET_SHA256


def test_a_run_without_1000_predictions_is_refused(tmp_path):
    with pytest.raises(S.Refused, match="999 .puml files"):
        S.check_run(make_run(tmp_path, "r", n=999), KEYS)


def test_a_missing_answers_folder_is_refused(tmp_path):
    (tmp_path / "r").mkdir()
    with pytest.raises(S.Refused, match="no answers/ folder"):
        S.check_run(tmp_path / "r", KEYS)


def test_stems_other_than_the_test_set_keys_are_refused(tmp_path):
    with pytest.raises(S.Refused, match="stems differ from the test-set keys"):
        S.check_run(make_run(tmp_path, "r", rename=True), KEYS)


def test_a_run_manifest_that_does_not_pass_is_refused(tmp_path):
    with pytest.raises(S.Refused, match="does not pass"):
        S.check_run(make_run(tmp_path, "r", passed=False), KEYS)


def test_a_run_without_a_run_manifest_is_refused(tmp_path):
    with pytest.raises(S.Refused, match="no run manifest"):
        S.check_run(make_run(tmp_path, "r", passed=None), KEYS)


def test_another_set_file_is_refused(tmp_path):
    with pytest.raises(S.Refused, match="set_file_sha256"):
        S.check_run(make_run(tmp_path, "r", set_sha="0" * 64), KEYS)


def test_a_test_set_other_than_the_frozen_one_is_refused(tmp_path):
    other = tmp_path / "test_set.json"
    other.write_text(S.TEST_SET.read_text() + "\n")
    with pytest.raises(S.Refused, match="frozen"):
        S.test_set_keys(other)


def test_the_test_set_and_ground_truth_are_the_benchmarks():
    assert S.TEST_SET == BENCH / "data" / "test_set.json" and len(KEYS) == 1000
    assert all((S.GT_DIR / (k + ".puml")).is_file() for k in KEYS)


def test_v1_is_the_seven_switches_of_scorer_v2_section_7():
    text = (FT / "methodology" / "scorer-v2.md").read_text()
    block = text.split("Seven\nswitches")[1].split("```")[1].split()
    assert sorted(block) == sorted(S.V1_SWITCHES) and len(S.V1_SWITCHES) == 7


# --------------------------------------------------------------------------- #
# layout, re-run, data roots, leak gate (scores a stand-in)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def scored(tmp_path_factory):
    runs = tmp_path_factory.mktemp("runs")
    make_run(runs, "9B_standin")
    result = S.score_run("9B_standin", runs_dir=runs, root=runs / "_root")
    return runs, result


def test_the_scores_are_in_the_loaders_layout_with_a_manifest(scored):
    runs, result = scored
    assert result["status"] == "scored"
    scores = runs / "9B_standin" / "scores"
    manifest = json.loads((scores / "manifest.json").read_text())
    for rel in S.result_files("9B_standin"):
        assert (scores / rel).is_file(), rel
    assert os.readlink(scores / "v1" / "csr") == os.path.join("..", "v2", "csr")
    assert os.readlink(scores / "v1" / "chrf") == os.path.join("..", "v2", "chrf")
    assert set(manifest["files"]) == {r for r, (v, m) in S.result_files("9B_standin").items()
                                      if not (v == "v1" and m in S.ONCE)}
    for rel, sha in manifest["files"].items():
        assert S.sha256_file(scores / rel) == sha
    assert manifest["scorer"]["commit"] == S.dev_scoring.SCORER_COMMIT
    assert manifest["v1_switches"] == S.V1_SWITCHES
    assert not (runs / "9B_standin" / S.WORK).exists()
    v1 = json.loads((scores / "v1/element_f1/9B_standin/element_f1_results.json").read_text())
    v2 = json.loads((scores / "v2/element_f1/9B_standin/element_f1_results.json").read_text())
    assert v1["scorer"]["extractor_args"] == [a.split("=", 1)[1] for a in S.V1_SWITCHES if "extractor-arg" in a]
    assert v2["scorer"]["extractor_args"] == [] and v2["scorer"]["first_diagram"] is True


def test_the_stand_in_reproduces_the_published_rows(scored):
    """The command scores as Р.3 did: the stand-in's rows equal the published v2 and v1 rows."""
    runs, _ = scored
    scores = runs / "9B_standin" / "scores"

    def rows(p):
        return {d["key"]: (d.get("tp"), d.get("fp"), d.get("fn"), d["compiled"])
                for d in json.loads(Path(p).read_text())["diagrams"]}
    src = STANDIN.name
    for version, base in (("v2", "data_v2"), ("v1", "data")):
        for m in ("element_f1", "relationship_f1"):
            assert rows(scores / version / m / "9B_standin" / f"{m}_results.json") == \
                rows(BENCH / base / m / src / f"{m}_results.json")
    assert rows(scores / "v2/csr/9B_standin/csr_results.json") == rows(BENCH / "data/csr" / src / "csr_results.json")


def test_the_data_roots_load_and_the_leak_gate_reads_the_answers(scored, tmp_path):
    runs, _ = scored
    entry = load_registry(FT / "build" / "test_registry.json")[0]
    entry = type(entry)(**{**entry.__dict__, "run_dir": "9B_standin"})
    for version in ("v2", "v1"):
        root = runs / "_root" / version
        md = load_model(entry, root)
        assert md is not None and len(md.rows) == 1000
        pdir = predictions_dir(root, "9B_standin")
        assert pdir.is_dir() and len(list(pdir.glob("*.puml"))) == 1000
        assert count_reasoning_leaks(pdir) == 0
    # the gate is no longer silent: a planted <think> in the run's answers is counted through the root
    planted = runs / "9B_standin" / "answers" / (KEYS[0] + ".puml")
    original = os.readlink(planted)
    planted.unlink()
    planted.write_text("<think>x</think>\n@startuml\n@enduml\n")
    try:
        assert count_reasoning_leaks(predictions_dir(runs / "_root" / "v2", "9B_standin")) == 1
    finally:
        planted.unlink()
        os.symlink(original, planted)


def test_a_byte_equal_rerun_is_accepted_and_a_differing_one_refused(scored):
    runs, _ = scored
    again = S.score_run("9B_standin", runs_dir=runs, root=runs / "_root")
    assert again["status"].startswith("byte-equal")
    target = runs / "9B_standin" / "scores" / "v2/chrf/9B_standin/chrf_results.json"
    keep = target.read_bytes()
    target.write_bytes(keep + b" ")
    try:
        with pytest.raises(S.Refused, match="re-run differs"):
            S.score_run("9B_standin", runs_dir=runs, root=runs / "_root")
    finally:
        target.write_bytes(keep)


def test_a_root_link_to_somewhere_else_is_refused(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    S._link(tmp_path / "root" / "x", tmp_path / "a")
    S._link(tmp_path / "root" / "x", tmp_path / "a")       # the same link again: accepted
    with pytest.raises(S.Refused, match="links to"):
        S._link(tmp_path / "root" / "x", tmp_path / "b")


# --------------------------------------------------------------------------- #
# the study's registry
# --------------------------------------------------------------------------- #
def test_the_study_registry_loads_with_three_zero_shot_and_three_fine_tuned_entries():
    import main_tables as M
    reg = load_registry(FT / "build" / "test_registry.json")
    assert len(reg) == 6
    assert sorted(e.condition for e in reg) == ["fine_tuned"] * 3 + ["zero_shot"] * 3
    assert all(e.arm == "qwen" and e.status == "scored" and not e.supplementary for e in reg)
    assert {e.run_dir for e in reg} == {f"{s}_{c}" for s in ("2B", "9B", "27B") for c in ("zero_shot", "20000")}
    by_id = {e.id: e for e in reg}
    for size, (z, f) in M.PAIRS.items():
        assert by_id[z].condition == "zero_shot" and by_id[f].condition == "fine_tuned"
        assert by_id[z].run_dir == f"{size}_zero_shot" and by_id[f].run_dir == f"{size}_20000"
        assert by_id[z].params_total_b == by_id[f].params_total_b
    published = {e.id for e in load_registry(BENCH / "analysis" / "model_registry.json")}
    assert not published & set(by_id)


# --------------------------------------------------------------------------- #
# an answer that kills the extractor's JVM
# --------------------------------------------------------------------------- #
def test_an_answer_that_kills_the_extractor_is_scored_as_an_empty_graph(tmp_path):
    """A self-referential function-like !define overflows the extractor's stack, an Error its per-file handler
    does not catch, so the frozen runner would exit 1. The answer is held back from the F1
    runners and scored as every non-parsing answer is: tp = fp = 0, fn = |GT|. CSR and chrF++
    still read it."""
    answers = tmp_path / "answers"
    answers.mkdir()
    bad, good = KEYS[0], KEYS[1:3]
    (answers / (bad + ".puml")).write_text((FT / "build/tests/fixtures/self_referential_define.puml").read_text())
    for k in good:
        os.symlink(STANDIN / (k + ".puml"), answers / (k + ".puml"))
    java = str(Path(S.dev_scoring.JAVA_HOME) / "bin" / "java")
    assert S.extractor_crashes(answers, tmp_path, [], java) == [bad]
    v1_args = [a.split("=", 1)[1] for a in S.V1_SWITCHES if a.startswith("--extractor-arg=")]
    assert S.extractor_crashes(answers, tmp_path, v1_args, java) == [bad]
    work = tmp_path / "work"
    work.mkdir()
    seconds, crashes = S.score_into(work, "r", str(answers), str(S.TEST_SET), str(S.GT_DIR),
                                    str(S.dev_scoring.RENDER_JAR), S.dev_scoring.JAVA_HOME, "/usr/bin/python3")
    assert crashes == {"v2": [bad], "v1": [bad]}
    for version in ("v2", "v1"):
        for m in ("element_f1", "relationship_f1"):
            rows = {d["key"]: d for d in json.loads((work / version / m / "r" / f"{m}_results.json")
                                                    .read_text())["diagrams"]}
            r = rows[bad]
            assert (r["tp"], r["fp"]) == (0, 0) and r["has_pred"] is False
            assert any(rows[k]["tp"] > 0 for k in good)
    chrf = {d["key"]: d for d in json.loads((work / "v2/chrf/r/chrf_results.json").read_text())["diagrams"]}
    assert chrf[bad]["has_pred"] is True
