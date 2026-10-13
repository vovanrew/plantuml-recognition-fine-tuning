"""Tests for build/pool.py: elements D1 to D4 of decisions/training-pool-rules.

Run from fine-tune/:  python3 -m pytest build/tests/test_pool.py -q
The tests that start the extractor need Java and the sibling repositories. The tests of the
artifacts read data/pool and the corpus, and are skipped when data/pool does not exist.
"""
import hashlib
import http.server
import json
import os
import struct
import subprocess
import sys
import threading
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pool as P  # noqa: E402

FT = Path(__file__).resolve().parents[2]
DATA = FT / "data/pool"
HEADER = "' Blob ID: 0\n' Original Path: /a.puml\n' Source: World of Code\n\n"
needs_artifacts = pytest.mark.skipif(not (DATA / "candidates.tsv").is_file(), reason="data/pool not built")


def png_header(w, h):
    return b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + struct.pack(">II", w, h) + b"\x08\x06\x00\x00\x00"


# --------------------------------------------------------------------------- #
# Pins
# --------------------------------------------------------------------------- #
def test_pinned_files_are_the_ones_on_disk():
    assert P.sha256_file(P.BUILD_TEST_SET) == P.PINS["build_test_set"]
    assert P.sha256_file(P.EXTRACTOR_V1) == P.PINS["extractor_v1"]
    assert P.sha256_file(P.EXTRACTOR_V2) == P.PINS["extractor_v2"]
    source = P._function_source(P.CSR_RUNNER, "extract_puml")
    assert hashlib.sha256(source.encode()).hexdigest() == P.PINS["extract_puml_source"]
    assert source.startswith("def extract_puml(text):") and source.rstrip().endswith('strip("`").strip()')


def test_the_v1_extractor_is_the_jar_of_the_test_set():
    fork = P.UNI / "plantuml/build/libs/plantuml-1.2025.9.jar"   # build_test_set.EXTRACTOR_JAR
    if not fork.is_file():
        pytest.skip("extractor fork JAR not built")
    assert P.sha256_file(fork) == P.PINS["extractor_v1"]


def test_a_changed_test_set_script_stops_the_build(tmp_path):
    changed = tmp_path / "build_test_set.py"
    changed.write_text(P.BUILD_TEST_SET.read_text() + "\n# edited\n")
    with pytest.raises(P.PoolError, match="pinned"):
        P.test_set_code(changed)


def test_a_changed_block_isolation_stops_the_build(tmp_path):
    changed = tmp_path / "csr_runner.py"
    changed.write_text(P.CSR_RUNNER.read_text().replace('text.rfind("@enduml")', 'text.find("@enduml")'))
    with pytest.raises(P.PoolError, match="extract_puml"):
        P.scorer_extract_puml(changed)


def test_another_extractor_jar_is_refused():
    with pytest.raises(P.PoolError, match="pinned"):
        P.check_jar(P.EXTRACTOR_V2, "extractor_v1")


# --------------------------------------------------------------------------- #
# Inclusion
# --------------------------------------------------------------------------- #
def made_up_records():
    ok = dict(primary_type="class", secondary_types=[], extraction_error=None, elements_total=3,
              content_lines=5, repository="r")
    return {
        "a.puml": dict(ok),
        "b.puml": dict(ok, primary_type="sequence"),
        "c.puml": dict(ok, primary_type="state"),
        "d.puml": dict(ok, secondary_types=["sequence"]),
        "e.puml": dict(ok, extraction_error="boom"),
        "f.puml": dict(ok, truncated=True),
        "g.puml": dict(ok, elements_total=51),
        "h.puml": dict(ok, elements_total=50),
        "i.puml": dict(ok, repository=None),
        "j.puml": dict(ok, primary_type="sequence", repository=""),
        "k.puml": dict(ok, truncated=False),
    }


def test_inclusion_is_the_filter_of_the_test_set(tmp_path):
    records = made_up_records()
    meta = tmp_path / "meta.json"
    meta.write_text(json.dumps({"classifications": records}))
    reference, _ = P.test_set_code().load_candidates(str(meta))
    got, drops = P.inclusion(records)
    assert {t: list(got[t]) for t in P.TYPES} == {t: list(reference[t]) for t in P.TYPES}
    assert list(got["class"]) == ["a.puml", "h.puml", "k.puml"] and list(got["sequence"]) == ["b.puml"]
    assert drops == {"class": {"secondary_types": 1, "extraction_error": 1, "truncated": 1,
                               "elements_total_above_50": 1, "no_repository": 1},
                     "sequence": {"no_repository": 1}}


def test_tiers_are_those_of_the_test_set():
    thr = P.tier_thresholds(P.load_test_set())
    assert thr == {"class": (11.0, 24.0, 49.0), "sequence": (12.0, 22.0, 39.0)}
    records = {f"{t}{n}.puml": dict(primary_type=t, content_lines=n)
               for t in P.TYPES for n in (11, 12, 13, 22, 23, 24, 25, 39, 40, 49, 50)}
    cells = P.cell_counts(list(records), records, thr)
    assert cells == {"class_T1": 1, "class_T2": 5, "class_T3": 4, "class_T4": 1,
                     "sequence_T1": 2, "sequence_T2": 2, "sequence_T3": 4, "sequence_T4": 3}


# --------------------------------------------------------------------------- #
# D2: degeneracy labels
# --------------------------------------------------------------------------- #
def test_deposit_pages_are_grouped_by_the_rule_of_the_test_set(tmp_path):
    files = {"aaa.png": (100, 50), "aaa_001.png": (100, 60), "bbb_01.png": (30, 40), "bbb.png": (10, 20),
             "ccc_002.png": (5, 5), "zzz.png": (7, 7)}
    folder = tmp_path / "img"
    folder.mkdir()
    with zipfile.ZipFile(tmp_path / "img.zip", "w") as zf:
        for name in sorted(files):
            (folder / name).write_bytes(png_header(*files[name]))
            zf.writestr("puml_images/" + name, png_header(*files[name]))
    (folder / "notes.txt").write_text("x")
    bases = {"aaa", "bbb", "bbb_01", "ccc"}      # bbb_01 is a diagram of its own, not a page of bbb
    pages, orphans = P.deposit_pages(folder, bases)
    reference, ref_orphans = P.test_set_code().map_pages_to_bases(str(tmp_path / "img.zip"), bases)
    assert pages == dict(reference) and orphans == ref_orphans == 1
    assert pages == {"aaa": [(100, 50), (100, 60)], "bbb": [(10, 20)], "bbb_01": [(30, 40)], "ccc": [(5, 5)]}


def test_the_size_is_predicted_at_300_over_96_and_padded_to_512():
    assert P.predicted_padded_size(219, 84) == (684, 512)      # test key 000133b3: 685 x 512 in the benchmark
    assert P.predicted_padded_size(1000, 2000) == (3125, 6250)
    assert P.predicted_padded_size(100, 100) == (512, 512)


@pytest.mark.parametrize("pages, norm, source, reasons", [
    ([(300, 200)], "@startuml class A @enduml", "predicted", []),
    ([(300, 200)], "@startuml skinparam dpi 200 class A @enduml", "render", ["own_dpi"]),
    ([(300, 200)], "@startuml skinparam DPI 200 @enduml", "render", ["own_dpi"]),
    ([(300, 200)], "@startuml class dpiValue @enduml", "predicted", []),
    ([(4096, 300)], "@startuml class A @enduml", "render", ["deposit_limit"]),
    ([(900, 4096)], "@startuml class A @enduml", "render", ["deposit_limit"]),
    ([(4095, 600)], "@startuml class A @enduml", "predicted", []),
    ([(300, 200)], "@startuml note left {{ Bob -> Alice }} end note @enduml", "render", ["embedded_diagram"]),
    ([(1280, 160)], "@startuml class A @enduml", "render", ["aspect_band"]),        # 4000 x 512: 7.81
    ([(1343, 160)], "@startuml class A @enduml", "render", ["aspect_band"]),        # 4197 x 512: 8.197
    ([(1344, 160)], "@startuml class A @enduml", "predicted", []),                  # 4200 x 512: 8.203
    ([(1270, 160)], "@startuml class A @enduml", "predicted", []),                  # 3969 x 512: 7.75
    ([(4096, 100)], "@startuml skinparam dpi 96 {{ }} @enduml", "render", ["own_dpi", "deposit_limit", "embedded_diagram"]),
    ([], "@startuml skinparam dpi 200 @enduml", "pages", []),
    ([(300, 200), (300, 100)], "@startuml skinparam dpi 200 @enduml", "pages", []),
])
def test_only_the_uncertain_groups_are_rendered(pages, norm, source, reasons):
    assert P.size_source(pages, norm) == (source, reasons)


def test_a_rendered_diagram_is_labelled_on_its_padded_size():
    row = lambda **kw: dict(key="k.puml", **kw)
    assert P.label_from_render(row(status="ok", padded_size=[16384, 9000], render_size=[16384, 9000])) == "clipped"
    assert P.label_from_render(row(status="ok", padded_size=[16383, 9000])) is None
    assert P.label_from_render(row(status="ok", padded_size=[4096, 512], render_size=[4096, 300])) is None   # 8.0
    assert P.label_from_render(row(status="ok", padded_size=[4097, 512], render_size=[4097, 300])) == "aspect"
    assert P.label_from_render(row(status="ok", padded_size=[512, 5000])) == "aspect"
    assert P.label_from_render(row(status="multi_page", pages=3)) == "multipage"
    assert P.label_from_render(row(status="no_image", reason="syntax_error")) == "no_image"
    for status in ("crash", "timeout", "postprocess_error", "missing_source"):
        with pytest.raises(P.PoolError, match="gives no size"):
            P.label_from_render(row(status=status))


def test_every_diagram_gets_its_label_from_the_pages_the_prediction_or_the_render():
    norms = {k: ("", "@startuml class A @enduml") for k in "abcdefg"}
    norms = {k + ".puml": v for k, v in norms.items()}
    norms["g.puml"] = ("", "@startuml skinparam dpi 300 @enduml")
    pages = {"a": [(300, 200)],               # predicted, usable
             "b": [(3000, 100)],              # predicted 9375 x 512: aspect 18
             "c": [(300, 200), (300, 200)],   # two pages
             "e": [(4096, 2000)],             # at the deposit's limit: rendered
             "f": [(4096, 100)],
             "g": [(300, 200)]}               # own dpi: rendered
    rendered = {"e.puml": dict(key="e.puml", status="ok", padded_size=[16384, 8000]),
                "f.puml": dict(key="f.puml", status="ok", padded_size=[9000, 512]),
                "g.puml": dict(key="g.puml", status="ok", padded_size=[512, 512])}
    labels = P.degeneracy_labels(sorted(norms), pages, norms, rendered)
    assert labels == {"a.puml": (None, "predicted"), "b.puml": ("aspect", "predicted"),
                      "c.puml": ("multipage", "pages"), "d.puml": ("no_image", "pages"),
                      "e.puml": ("clipped", "render"), "f.puml": ("aspect", "render"),
                      "g.puml": (None, "render")}
    with pytest.raises(P.PoolError, match="no render"):
        P.degeneracy_labels(["e.puml"], pages, norms, {})


# --------------------------------------------------------------------------- #
# D1: the stages of the test set, then its repositories removed
# --------------------------------------------------------------------------- #
def made_up_corpus():
    """Keys in corpus order, with (type, repository, normalized-code hash, degeneracy label,
    extractor type)."""
    rows = [("c1", "class", "TEST", "h1", None, "class"),          # in a test repository
            ("c2", "class", "A", "h1", None, "class"),             # later copy of c1
            ("c3", "class", "A", "h2", "clipped", "class"),        # degenerate first copy
            ("c4", "class", "B", "h2", None, "class"),             # its later copy
            ("c5", "class", "A", "h3", None, "component"),         # first copy, another type
            ("c6", "class", "B", "h3", None, "class"),             # its later copy
            ("c7", "class", "TEST", "h4", None, "class"),
            ("c8", "class", "C", "h5", None, "class"),
            ("c9", "class", "C", "h6", "multipage", "class"),
            ("s1", "sequence", "A", "h1", None, "sequence"),       # same code as c1, other type
            ("s2", "sequence", "D", "h7", None, ""),               # no type
            ("s3", "sequence", "D", "h8", None, "")]               # no type, remote include
    keys = {t: [r[0] for r in rows if r[1] == t] for t in P.TYPES}
    return (keys, {r[0]: r[4] for r in rows}, {r[0]: r[3] for r in rows},
            {r[0]: {"diagram_type": r[5] or None, "error": None if r[5] else "parse_error"} for r in rows},
            {r[0]: r[2] for r in rows})


def test_the_stages_run_in_the_order_of_the_test_set_and_its_repositories_leave_last():
    keys, labels, sha1, extracted, repo_of = made_up_corpus()
    cp = P.candidate_pool(keys, labels, sha1, extracted, {"s3"}, repo_of, {"TEST"})
    # degeneracy first: a degenerate first copy does not hide its later copy (c4 stays)
    assert cp["survivors"]["class"] == ["c1", "c2", "c4", "c5", "c6", "c7", "c8"]
    # deduplication over the whole corpus, per type, before the test repositories are removed:
    # c2 goes because c1 came first; s1 stays because it is another type
    assert cp["deduped"] == {"class": ["c1", "c4", "c5", "c7", "c8"], "sequence": ["s1", "s2", "s3"]}
    assert cp["duplicates"] == {"class": 2, "sequence": 0}
    # type agreement after deduplication: c6 was already dropped as the copy of c5
    assert cp["agreed"] == {"class": ["c1", "c4", "c7", "c8"], "sequence": ["s1", "s3"]}
    assert cp["disagree"] == {"class": {"component": 1}, "sequence": {"<unparsed>": 1}}
    assert cp["undetermined"] == {"class": [], "sequence": ["s3"]}
    # the test repositories leave last
    assert cp["d1"] == {"class": ["c4", "c8"], "sequence": ["s1", "s3"]}


def test_no_candidate_keeps_the_normalized_code_of_a_diagram_of_a_test_repository_that_came_first():
    keys, labels, sha1, extracted, repo_of = made_up_corpus()
    cp = P.candidate_pool(keys, labels, sha1, extracted, set(), repo_of, {"TEST"})
    test_hashes = {sha1[k] for k in ("c1", "c7")}
    assert not [k for k in cp["d1"]["class"] if sha1[k] in test_hashes]
    # in the plan's order (repositories removed before deduplication) the copy c2 would stay
    plan = [k for k in keys["class"] if repo_of[k] != "TEST" and labels[k] is None]
    assert "c2" in P.dedup(plan, sha1)[0]


def test_deduplication_keeps_the_first_occurrence_in_the_order_given():
    sha1 = {"a": "x", "b": "y", "c": "x", "d": "x", "e": "z"}
    assert P.dedup(["a", "b", "c", "d", "e"], sha1) == (["a", "b", "e"], 2)
    assert P.dedup(["d", "c", "b", "a"], sha1) == (["d", "b"], 2)


def test_a_source_whose_type_needs_the_network_is_left_to_d3():
    extracted = {"a": {"diagram_type": None, "error": "parse_error"},
                 "b": {"diagram_type": None, "error": "parse_error"},
                 "c": {"diagram_type": "component", "error": None},
                 "d": {"diagram_type": "class", "error": None}}
    kept, undetermined, disagree = P.type_agreement(["a", "b", "c", "d", "e"], "class", extracted, remote={"a", "c"})
    assert kept == ["a", "d"] and undetermined == ["a"]
    assert dict(disagree) == {"<unparsed>": 2, "component": 1}      # b, e (no record), c


def test_the_rebuild_of_the_test_set_runs_the_sampler_of_the_test_set():
    code = P.test_set_code()
    records = {f"{t[0]}{i:03d}.puml": dict(primary_type=t, content_lines=1 + (i * 7) % 60, repository=f"r{i % 9}")
               for t in P.TYPES for i in range(120)}
    agreed = {t: [k for k in records if records[k]["primary_type"] == t] for t in P.TYPES}
    test = {"config": {"n_per_cell": 6, "seed": 42}}
    thresholds, available, selected = P.rebuild_test_set(agreed, records, test)
    assert sum(available.values()) == 240
    assert len(selected) == 45        # nine repositories, at most five diagrams of each across the set
    by_cell = {}
    for t in P.TYPES:
        thr = code.quartile_thresholds([records[k]["content_lines"] for k in agreed[t]])
        assert thr == thresholds[t]
        for k in agreed[t]:
            by_cell.setdefault((t, code.assign_tier(records[k]["content_lines"], thr)), []).append((k, records[k]))
    reference, _ = code.sample_cells(by_cell, 6, 42)
    assert selected == [k for k, _ in reference]
    per_repo = {}
    for k in selected:
        per_repo[records[k]["repository"]] = per_repo.get(records[k]["repository"], 0) + 1
    assert max(per_repo.values()) <= code.REPO_CAP == 5


def test_a_number_of_the_record_is_checked_exactly_unless_types_are_undetermined():
    P.expect("x", 5, 5)
    with pytest.raises(P.PoolError, match="Implementation note 2"):
        P.expect("x", 6, 5)
    P.expect("x", 6, 5, slack=1)          # one source whose type the closed network leaves open
    P.expect("x", 5, 5, slack=1)
    for got in (4, 7):
        with pytest.raises(P.PoolError):
            P.expect("x", got, 5, slack=1)
    assert P.EXPECTED == {"inclusion": {"class": 71323, "sequence": 35025},
                          "degeneracy": {"multipage": 309, "clipped": 598, "aspect": 326},
                          "rebuild_cells_equal": 7, "rebuild_class_T4": (16763, 16766),
                          "d1": {"class": 43233, "sequence": 24285}}


def test_render_py_is_not_called_again_when_the_manifest_holds_every_key(tmp_path, monkeypatch):
    out = tmp_path / "render"
    out.mkdir()
    rows = [dict(key="a.puml", status="ok", padded_size=[600, 512]), dict(key="b.puml", status="multi_page", pages=2)]
    (out / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (out / "summary.json").write_text("{}")
    called = []
    monkeypatch.setattr(P.subprocess, "run", lambda *a, **k: called.append(a) or (_ for _ in ()).throw(AssertionError))
    assert P.run_render(["b.puml", "a.puml"], out) == [rows[1], rows[0]]
    assert not called and (tmp_path / "render_keys.txt").read_text() == "b.puml\na.puml\n"
    with pytest.raises(AssertionError):
        P.run_render(["a.puml", "c.puml"], out)       # a key without a line: render.py is called
    assert called and called[0][0][1:4] == [str(P.RENDER_PY), "--src-dir", str(P.PUML_DIR)]


def test_render_py_runs_with_absolute_paths_in_render_cwd_beside_its_log(tmp_path, monkeypatch):
    """The call runs in `render_cwd` beside its log, not in the folder the build is started from, and
    its three paths are absolute."""
    seen = []

    class Done:
        returncode = 0

    def fake_run(cmd, **kw):
        seen.append((cmd, kw))
        out = Path(cmd[cmd.index("--out") + 1])
        out.mkdir(parents=True, exist_ok=True)
        (out / "manifest.jsonl").write_text(json.dumps(dict(key="a.puml", status="ok", padded_size=[600, 512])) + "\n")
        return Done()
    monkeypatch.setattr(P.subprocess, "run", fake_run)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "src").mkdir()
    rows = P.run_render(["a.puml"], Path("work/render"), src_dir=Path("src"))
    assert [r["key"] for r in rows] == ["a.puml"]
    cmd, kw = seen[0]
    here = tmp_path.resolve()
    assert cmd[2:] == ["--src-dir", str(here / "src"), "--keys", str(here / "work/render_keys.txt"),
                       "--out", str(here / "work/render")]
    assert kw["cwd"] == str(here / "work/render_cwd") and (here / "work/render_cwd").is_dir()
    assert kw["cwd"] != str(Path(P.__file__).resolve().parents[1])      # not fine-tune/ itself


SOURCES = {
    "class_ok.puml": ("class", HEADER + "@startuml\nclass A\nclass B\nA --> B\n@enduml\n"),
    "sequence_ok.puml": ("sequence", HEADER + "@startuml\nAlice -> Bob: hi\nBob --> Alice: ok\n@enduml\n"),
    "bom.puml": ("class", HEADER + "\ufeff@startuml\nclass A\nA --> B\n@enduml\n"),
    "two_blocks.puml": ("class", HEADER + "@startuml\nclass A\n@enduml\n\n@startuml\nclass B\n@enduml\n"),
    "double_at.puml": ("class", HEADER + "@@startuml\nclass A\nA --> B\n@@enduml\n"),
    "in_comment.puml": ("class", HEADER + "/**\n * @startuml\n * class A\n * A --> B\n * @enduml\n */\n"),
    "remote.puml": ("sequence", HEADER + "@startuml\n!include https://nonexistent-host.invalid/style.puml\n"
                                         "Alice -> Bob: hi\n@enduml\n"),
    "include_outside.puml": ("sequence", HEADER + "!includeurl https://nonexistent-host.invalid/style.puml\n"
                                                  "@startuml\nAlice -> Bob: hi\n@enduml\n"),
    "other_type.puml": ("class", HEADER + "@startuml\nAlice -> Bob: hi\n@enduml\n"),
    "no_block.puml": ("class", HEADER + "class A\nA --> B\n"),
    "syntax_error.puml": ("class", HEADER + "@startuml\nclass A {\n this is not ] plantuml [ at all -->\n@enduml\n"),
}


@pytest.fixture(scope="module")
def made_up_sources(tmp_path_factory):
    root = tmp_path_factory.mktemp("sources")
    for name, (_, text) in SOURCES.items():
        (root / name).write_text(text, encoding="utf-8")
    return root


def test_the_extractor_types_a_raw_file_and_reads_only_its_first_record(made_up_sources, tmp_path):
    dirs = P.link_batches(sorted(SOURCES), made_up_sources, tmp_path / "links", batch=4)
    assert len(dirs) == 3 and all(os.path.islink(os.path.join(dirs[0], f)) for f in os.listdir(dirs[0]))
    got = P.extract_many(dirs, P.EXTRACTOR_V1, tmp_path / "cwd", offline=True, jobs=2)
    assert sorted(got) == sorted(SOURCES)                    # no record under the name of a further block
    assert got["class_ok.puml"] == {"diagram_type": "class", "error": None}
    assert got["sequence_ok.puml"] == {"diagram_type": "sequence", "error": None}
    assert got["two_blocks.puml"]["diagram_type"] == "class"
    assert got["double_at.puml"]["diagram_type"] == "class"       # the raw file parses; its isolated block does not
    assert got["in_comment.puml"]["diagram_type"] == "class"
    assert got["other_type.puml"]["diagram_type"] == "sequence"
    assert got["remote.puml"] == {"diagram_type": None, "error": "parse_error"}
    assert got["include_outside.puml"] == {"diagram_type": "sequence", "error": None}
    assert not got["no_block.puml"]["diagram_type"] and got["no_block.puml"]["error"]
    kept, undetermined, _ = P.type_agreement(
        sorted(SOURCES), "sequence", got, remote={k for k in SOURCES if P.has_remote_include(k, made_up_sources)})
    assert undetermined == ["remote.puml"]
    assert kept == ["include_outside.puml", "other_type.puml", "remote.puml", "sequence_ok.puml"]


def test_the_extractor_run_equals_the_run_of_the_test_set_code(made_up_sources, tmp_path, monkeypatch):
    keys = [k for k in sorted(SOURCES) if "remote" not in k and "include" not in k]
    monkeypatch.setenv("PATH", os.path.dirname(P.java_bin()) + os.pathsep + os.environ["PATH"])
    reference = P.test_set_code().parser_diagram_types(keys, str(made_up_sources), str(P.EXTRACTOR_V1))
    got = P.extract_many(P.link_batches(keys, made_up_sources, tmp_path / "links"), P.EXTRACTOR_V1,
                         tmp_path / "cwd", offline=True)
    assert {k: got[k]["diagram_type"] or "" for k in keys} == {k: reference.get(k, "") for k in keys}


# --------------------------------------------------------------------------- #
# D3: one diagram block, read by the scorer as its labelled type
# --------------------------------------------------------------------------- #
def test_the_block_rule_is_the_one_written_from_plantuml_source():
    sys.path.insert(0, str(FT / "decisions/render-pipeline/scripts"))
    import verify_r2_corpus_block_starts as original
    cases = [text for _, text in SOURCES.values()] + [
        "@startuml\nA\n@enduml", "\\startuml\nA\n\\enduml", "@startsalt\n{\n}\n@endsalt", "@STARTUML\nA\n@enduml",
        "' @startuml\n' A\n' @enduml", "@startuml\nA", "@enduml\n@startuml", "text @startuml\nA\n@enduml",
        "<b>@startuml\nA\n@enduml", "@startuml\r\nA\r\n@enduml\r\n@startuml\rB\r@enduml", "@startfoo\nA\n@endfoo",
        "\u3000@startuml\nA\n@enduml", "@startuml a\n@enduml\n@startuml b\n@enduml\n@startuml c\n@enduml"]
    names = sorted(os.listdir(P.PUML_DIR))[::700]
    cases += [open(os.path.join(P.PUML_DIR, n), "rb").read().decode("utf-8", "replace") for n in names]
    assert len(cases) > 200
    for text in cases:
        assert [line for _, line in P.block_starts(text)] == original.blocks(text)
    count = lambda text: len(P.block_starts(text))
    assert [count(c) for c in cases[len(SOURCES):len(SOURCES) + 13]] == [1, 1, 1, 1, 1, 0, 0, 0, 1, 2, 0, 1, 3]


def test_the_block_is_isolated_as_the_scorer_isolates_it(made_up_sources, tmp_path):
    (made_up_sources / "latin1.puml").write_bytes(HEADER.encode() + b"@startuml\r\nclass Caf\xe9\r\n@enduml\r\ntail\r\n")
    dirs = P.isolate_batches(["class_ok.puml", "double_at.puml", "in_comment.puml", "latin1.puml", "no_block.puml"],
                             made_up_sources, tmp_path / "iso")
    read = lambda name: (Path(dirs[0]) / name).read_bytes()
    assert read("class_ok.puml") == b"@startuml\nclass A\nclass B\nA --> B\n@enduml"
    assert read("double_at.puml") == b"@startuml\nclass A\nA --> B\n@@enduml"         # cut from the second @
    assert read("in_comment.puml") == b"@startuml\n * class A\n * A --> B\n * @enduml"
    assert read("latin1.puml") == b"@startuml\r\nclass Caf\xe9\r\n@enduml"            # bytes kept
    assert read("no_block.puml") == (HEADER + "class A\nA --> B").encode()            # the scorer's fallback
    (made_up_sources / "latin1.puml").unlink()


def test_d3_keeps_a_source_of_one_block_that_the_scorer_reads_as_its_type(made_up_sources, tmp_path):
    keys = sorted(SOURCES)
    label_of = {k: SOURCES[k][0] for k in keys}
    verdict, facts, records = P.one_block_check(keys, label_of, made_up_sources, tmp_path)
    assert {k: v for k, v in verdict.items() if v[0]} == {
        "class_ok.puml": (True, None), "sequence_ok.puml": (True, None), "bom.puml": (True, None),
        "include_outside.puml": (True, None)}
    assert {k: v[1] for k, v in verdict.items() if not v[0]} == {
        "two_blocks.puml": "several_blocks", "double_at.puml": "double_at_startuml",
        "in_comment.puml": "diagram_inside_a_comment", "remote.puml": "remote_include",
        "other_type.puml": "type_differs_on_the_block", "no_block.puml": "no_block",
        "syntax_error.puml": "other_error"}
    assert facts["two_blocks.puml"]["blocks"] == 2 and facts["no_block.puml"]["blocks"] == 0
    assert records["double_at.puml"]["error"] == "no_block" and records["in_comment.puml"]["error"] == "no_block"
    assert records["remote.puml"]["error"] == "parse_error"


def test_both_extractor_jars_give_the_same_verdict_on_the_made_up_sources(made_up_sources, tmp_path):
    keys = sorted(SOURCES)
    label_of = {k: SOURCES[k][0] for k in keys}
    v1, _, r1 = P.one_block_check(keys, label_of, made_up_sources, tmp_path / "v1", jar=P.EXTRACTOR_V1)
    v2, _, r2 = P.one_block_check(keys, label_of, made_up_sources, tmp_path / "v2", jar=P.EXTRACTOR_V2)
    assert v1 == v2 and r1 == r2


class _Count(http.server.BaseHTTPRequestHandler):
    hits = 0

    def do_GET(self):
        type(self).hits += 1
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"participant Carol\n")

    def log_message(self, *a):
        pass


def test_the_jvm_options_close_the_network(tmp_path):
    """A JVM started with OFFLINE reaches no HTTP server, also not by falling back to a direct
    connection when the proxy does not answer."""
    server = http.server.HTTPServer(("127.0.0.1", 0), _Count)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}/x.puml"
    fetch = tmp_path / "Fetch.java"
    fetch.write_text(
        "import java.net.*; import java.io.*;\n"
        "public class Fetch { public static void main(String[] a) throws Exception {\n"
        "  try { URLConnection c = new URI(a[0]).toURL().openConnection(); c.setConnectTimeout(5000);\n"
        "    try (InputStream in = c.getInputStream()) { System.out.println(\"FETCHED\"); }\n"
        "  } catch (Exception e) { System.out.println(\"FAILED \" + e); } } }\n")
    run = lambda options: subprocess.run([P.java_bin(), *options, str(fetch), url], capture_output=True,
                                         text=True, timeout=120).stdout
    try:
        _Count.hits = 0
        assert run([]).startswith("FETCHED") and _Count.hits == 1      # the server can be reached
        assert run(P.OFFLINE).startswith("FAILED") and _Count.hits == 1
    finally:
        server.shutdown()


def test_remote_includes_are_recognised():
    yes = ["!includeurl https://x.org/a.puml", "  !include https://x.org/a.puml", "!include <https://x.org/a>",
           "!include_many http://x.org/a", "!includesub https://x.org/a!part", "!theme spacelab from https://x.org/themes",
           "!INCLUDEURL HTTPS://X.ORG/A"]
    no = ["!include <tupadr3/common>", "!include common.puml", "!theme spacelab", "' !includeurl https://x.org/a",
          "note: see !include https://x.org", "Alice -> Bob: GET https://x.org"]
    assert all(P.REMOTE.search("@startuml\n" + line + "\n@enduml") for line in yes)
    assert not any(P.REMOTE.search("@startuml\n" + line + "\n@enduml") for line in no)


# --------------------------------------------------------------------------- #
# D4: token cap
# --------------------------------------------------------------------------- #
def test_the_tokenizer_is_the_slow_class_and_reproduces_the_record_on_test_files():
    assert type(P.tokenizer()).__name__ == "Qwen2Tokenizer"
    gt = json.load(open(P.GT_TOKENS))
    assert gt["tokenizer"]["tokenizer_class"] == "Qwen2Tokenizer" and gt["tokenizer"]["add_special_tokens"] is False
    rows = gt["per_file"][::25] + [r for r in gt["per_file"] if r["key"].startswith("e9a06725c4d8")]
    assert len(rows) == 41
    for r in rows:
        assert P._count_whole_file(os.path.join(P.TEST_PUML_DIR, r["key"])) == r["n_tokens"]
    assert rows[-1]["n_tokens"] == 189        # the file on which the fast class gives 133


def test_the_target_is_the_block_the_scorer_isolates():
    assert P.target_text(HEADER + "@startuml\nclass A\n@enduml\n\ntrailing text\n") == "@startuml\nclass A\n@enduml"
    assert P.target_text(HEADER + "x @startuml\nA\n@enduml\n@startuml\nB\n@enduml\n") == "@startuml\nA\n@enduml\n@startuml\nB\n@enduml"


def test_a_target_without_a_block_pair_loses_the_three_header_lines():
    assert P.target_text(HEADER + "@startuml\nclass A\n") == "@startuml\nclass A"
    assert P.target_text(HEADER + "\\startuml\nclass A\n\\enduml\n") == "\\startuml\nclass A\n\\enduml"
    assert P.target_text(HEADER + "@enduml\nclass A\n@startuml\n") == "@enduml\nclass A\n@startuml"


def test_the_header_is_not_counted_and_the_cap_is_5376_tokens(tmp_path):
    assert P.TOKEN_CAP == 5376
    body = "@startuml\n" + "class Alpha\n" * 40 + "@enduml"
    f = tmp_path / "a.puml"
    f.write_text(HEADER + body + "\n")
    assert P._count_target(str(f)) == P.count_tokens(body) < P._count_whole_file(str(f))
    long = tmp_path / "long.puml"
    long.write_text(HEADER + "@startuml\n" + "class Alpha\n" * 3000 + "@enduml\n")
    counts = P.target_token_counts(["a.puml", "long.puml"], tmp_path, workers=2)
    assert counts["a.puml"] <= P.TOKEN_CAP < counts["long.puml"]
    assert [k for k in counts if P.within_token_cap(counts[k])] == ["a.puml"]
    assert P.within_token_cap(5376) and not P.within_token_cap(5377)


def test_a_source_that_is_not_utf8_is_counted_with_replacement_characters(tmp_path):
    f = tmp_path / "latin1.puml"
    f.write_bytes(HEADER.encode() + b"@startuml\r\nclass Caf\xe9\r\n@enduml\r\n")
    assert P.read_for_tokens(str(f)).endswith("@startuml\nclass Caf\ufffd\n@enduml\n")
    assert P._count_target(str(f)) == P.count_tokens("@startuml\nclass Caf\ufffd\n@enduml")


# --------------------------------------------------------------------------- #
# The artifacts of step 1
# --------------------------------------------------------------------------- #
def table():
    rows = {}
    with open(DATA / "candidates.tsv") as fh:
        head = next(fh).rstrip("\n").split("\t")
        for line in fh:
            f = dict(zip(head, line.rstrip("\n").split("\t")))
            rows[f["key"]] = f
    return rows


@needs_artifacts
def test_the_artifacts_have_the_recorded_sha256():
    lines = (DATA / "SHA256SUMS").read_text().splitlines()
    assert len(lines) >= 11
    for line in lines:
        digest, name = line.split("  ", 1)
        assert P.sha256_file(DATA / name) == digest
    listed = {line.split("  ", 1)[1] for line in lines}
    assert listed == {str(p.relative_to(DATA)) for p in DATA.rglob("*") if p.is_file() and p.name != "SHA256SUMS"}


@needs_artifacts
def test_the_funnel_the_table_and_the_pool_file_agree():
    funnel = json.load(open(DATA / "funnel.json"))
    steps = {s["step"]: s for s in funnel["steps"]}
    rows = table()
    pool = json.load(open(DATA / "pool_after_d6.json"))
    assert steps["test_repositories_removed_D1"]["total"] == len(rows)
    assert steps["one_block_D3"]["total"] == sum(1 for f in rows.values() if f["d3"] == "kept")
    assert steps["token_cap_D4"]["total"] == sum(1 for f in rows.values() if f["d4"] == "kept")
    assert steps["near_duplicates_D6"]["total"] == pool["count"] == len(pool["diagrams"])
    assert steps["near_duplicates_D6"]["cells"] == pool["cells"]
    for s in funnel["steps"]:
        assert sum(s["cells"].values()) == s["total"] == s["class"] + s["sequence"]
    assert funnel["d6"]["removed"]["total"] == steps["token_cap_D4"]["total"] - pool["count"]
    assert sum(v["total"] for v in funnel["d6"]["removed_by_combination_of_arms"].values()) == funnel["d6"]["removed"]["total"]
    for d in pool["diagrams"]:
        f = rows[d["key"]]
        assert (f["primary_type"], int(f["tier"]), f["repository"]) == (d["primary_type"], d["tier"], d["repository"])
        assert int(f["target_tokens"]) <= P.TOKEN_CAP
    assert max(int(f["target_tokens"]) for f in rows.values() if f["d4"] == "kept") == \
        steps["token_cap_D4"]["longest_kept_target_tokens"]


@needs_artifacts
def test_the_pool_follows_the_metadata_and_holds_no_diagram_of_a_test_repository():
    records, test = P.load_records(), P.load_test_set()
    thr, code = P.tier_thresholds(test), P.test_set_code()
    test_repos = {d["repository"] for d in test["diagrams"]}
    assert len(test_repos) == 815
    included, _ = P.inclusion(records)
    for key, f in table().items():
        r = records[key]
        assert key in included[f["primary_type"]]
        assert r["repository"] == f["repository"] and r["repository"] not in test_repos
        assert code.assign_tier(r["content_lines"], thr[r["primary_type"]]) == int(f["tier"])


@needs_artifacts
def test_no_candidate_has_the_normalized_code_of_a_test_diagram_or_of_another_candidate():
    rows = table()
    test_keys = [d["key"] for d in P.load_test_set()["diagrams"]]
    norms = P.normalized(list(rows) + test_keys, P.PUML_DIR, workers=8)
    test_hashes = {norms[k][0] for k in test_keys}
    assert len(test_hashes) == 1000
    assert not [k for k in rows if norms[k][0] in test_hashes]
    for t in P.TYPES:
        hashes = [norms[k][0] for k, f in rows.items() if f["primary_type"] == t]
        assert len(hashes) == len(set(hashes))


@needs_artifacts
def test_the_labels_of_the_rendered_diagrams_come_from_the_render_manifest():
    manifest = {r["key"]: r for r in P.read_jsonl(DATA / "d2_render/manifest.jsonl")}
    keys = (DATA / "d2_render/keys.txt").read_text().split()
    assert set(keys) == set(manifest)
    seen = 0
    with open(DATA / "d2_degeneracy_labels.tsv") as fh:
        next(fh)
        for line in fh:
            key, _, label, source, _ = line.rstrip("\n").split("\t")
            if source == "render":
                seen += 1
                assert str(P.label_from_render(manifest[key])) == label
    assert seen == len(keys)
    rows = table()
    assert not [k for k in keys if k in rows and P.label_from_render(manifest[k]) is not None]


@needs_artifacts
def test_every_excluded_source_of_d3_is_excluded_again(tmp_path):
    excluded = P.read_jsonl(DATA / "d3_excluded.jsonl")
    kept = [k for k, f in table().items() if f["d3"] == "kept"][::1500]
    keys = [d["key"] for d in excluded] + kept
    label_of = {d["key"]: d["primary_type"] for d in excluded}
    label_of.update({k: table()[k]["primary_type"] for k in kept})
    verdict, _, _ = P.one_block_check(keys, label_of, P.PUML_DIR, tmp_path)
    assert {d["key"]: (False, d["kind"]) for d in excluded} == {k: v for k, v in verdict.items() if not v[0]}
    assert all(verdict[k] == (True, None) for k in kept)


@needs_artifacts
def test_the_token_counts_of_the_table_are_recounted_on_a_sample():
    rows = table()
    counted = [k for k, f in rows.items() if f["target_tokens"]]
    longest = sorted(counted, key=lambda k: -int(rows[k]["target_tokens"]))[:40]
    sample = longest + counted[::2000]
    again = P.target_token_counts(sample, P.PUML_DIR, workers=4)
    assert {k: int(rows[k]["target_tokens"]) for k in sample} == again
    assert all((rows[k]["d4"] == "kept") == P.within_token_cap(again[k]) for k in sample)
    assert any(rows[k]["d4"] == "dropped" for k in longest)
