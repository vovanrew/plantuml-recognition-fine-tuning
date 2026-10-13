"""Tests for build/render.py.

Run from fine-tune/:  python3 -m pytest build/tests -q
The tests that start PlantUML need Java, Graphviz and the sibling repositories; the two that
render the largest test diagrams take about a minute together.
"""
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import render as R  # noqa: E402

FIX = Path(__file__).parent / "fixtures"
BENCH = R.UNI / "baseline-evaluation/data"


def manifest(out):
    return {r["key"]: r for r in R.read_manifest(Path(out) / "manifest.jsonl")}


# --------------------------------------------------------------------------- #
# Pins
# --------------------------------------------------------------------------- #
def test_pinned_files_are_the_ones_on_disk():
    paths = dict(renderer_jar=R.RENDERER_JAR, stage1_script=R.STAGE1_SCRIPT,
                 stage2_script=R.STAGE2_SCRIPT, stage3_script=R.STAGE3_SCRIPT)
    assert {k: R.sha256_file(p) for k, p in paths.items()} == R.PINS
    R.check_pins()


def test_the_instrument_repository_holds_the_same_stage3_file():
    public = R.UNI / "plantuml-recognition-evaluation/util/standardize_images.py"
    assert R.sha256_file(public) == R.PINS["stage3_script"]


def test_a_changed_stage_file_stops_the_run(tmp_path):
    changed = tmp_path / "postprocess_images.py"
    changed.write_text(R.STAGE2_SCRIPT.read_text() + "\n# edited\n")
    with pytest.raises(R.RenderError, match="stage2_script"):
        R.check_pins(stage2=changed)


def test_another_plantuml_jar_is_refused():
    fork = R.UNI / "plantuml/build/libs/plantuml-1.2025.9.jar"   # the extractor fork, same file name
    if not fork.is_file():
        pytest.skip("extractor fork JAR not built")
    with pytest.raises(R.RenderError, match="renderer_jar"):
        R.check_pins(jar=fork)


# --------------------------------------------------------------------------- #
# Stage 1 command
# --------------------------------------------------------------------------- #
def test_java_command_is_the_stock_command_with_heap_and_threads_as_arguments():
    script = R.STAGE1_SCRIPT.read_text()
    default = lambda name: re.search(rf"^{name}=(\d+)$", script, re.M).group(1)
    assert int(default("DEFAULT_DPI")) == R.DPI
    assert int(default("DEFAULT_LIMIT_SIZE")) == R.LIMIT_SIZE
    assert 'echo "skinparam dpi ${DPI}" > "${CONFIG_FILE}"' in script
    stock = ["java"]
    for piece in re.findall(r'^\s*plantuml_cmd="\$\{plantuml_cmd\} (.*)"$', script, re.M):
        stock += piece.replace('\\"', "").split()
    values = {"${LIMIT_SIZE}": str(R.LIMIT_SIZE), "${HEAP_MB}": "8192", "${PLANTUML_JAR}": "JAR",
              "${OUTPUT_DIR}": "OUT", "${CONFIG_FILE}": "CFG", "${INPUT_DIR}": "IN"}
    for var, value in values.items():
        stock = [tok.replace(var, value) for tok in stock]
    assert stock == R.plantuml_command("java", "JAR", "IN", "OUT", "CFG", threads="auto", heap_mb=8192)


# --------------------------------------------------------------------------- #
# Stages 2 and 3
# --------------------------------------------------------------------------- #
def test_stage_code_is_loaded_from_the_two_original_files():
    postprocess, standardize = R.stage_code()
    assert postprocess.__file__ == str(R.STAGE2_SCRIPT) and postprocess.MIN_SIDE == 512
    assert standardize.__file__ == str(R.STAGE3_SCRIPT) and standardize.LONG_EDGE == 1568


def test_stage2_pads_a_small_image_on_its_corner_colour(tmp_path):
    p = tmp_path / "small.png"
    im = Image.new("RGB", (100, 60), (10, 20, 30))
    im.putpixel((5, 5), (255, 0, 0))
    im.save(p)
    assert R.stage_code()[0].postprocess_one(p) == "padded"
    out = Image.open(p)
    assert (out.size, out.mode) == ((512, 512), "RGB")
    assert out.getpixel((0, 0)) == (10, 20, 30) and out.getpixel((511, 511)) == (10, 20, 30)
    assert out.getpixel((206 + 5, 226 + 5)) == (255, 0, 0)


def test_stage2_pads_only_the_short_side(tmp_path):
    p = tmp_path / "wide.png"
    Image.new("RGB", (800, 100), "white").save(p)
    assert R.stage_code()[0].postprocess_one(p) == "padded"
    assert Image.open(p).size == (800, 512)


def test_stage2_converts_to_rgb_and_leaves_a_conformant_file_untouched(tmp_path):
    p = tmp_path / "palette.png"
    Image.new("P", (600, 700)).save(p)
    assert R.stage_code()[0].postprocess_one(p) == "converted"
    assert (Image.open(p).size, Image.open(p).mode) == ((600, 700), "RGB")
    before = p.read_bytes()
    assert R.stage_code()[0].postprocess_one(p) == "unchanged"
    assert p.read_bytes() == before


def test_stage3_downscales_the_long_edge_to_1568_and_never_upscales(tmp_path):
    standardize = R.stage_code()[1]
    Image.new("RGB", (3136, 1000), "white").save(tmp_path / "big.png")
    Image.new("RGB", (600, 512), "white").save(tmp_path / "small.png")
    assert standardize.standardize(tmp_path / "big.png", tmp_path / "big_out.png", 1568) == ((3136, 1000), (1568, 500))
    assert standardize.standardize(tmp_path / "small.png", tmp_path / "small_out.png", 1568) == ((600, 512), (600, 512))


def test_stage3_flattens_transparency_on_white(tmp_path):
    Image.new("RGBA", (600, 600), (0, 0, 0, 0)).save(tmp_path / "clear.png")
    R.stage_code()[1].standardize(tmp_path / "clear.png", tmp_path / "out.png", 1568)
    out = Image.open(tmp_path / "out.png")
    assert out.mode == "RGB" and out.getpixel((300, 300)) == (255, 255, 255)


def test_stages_2_and_3_turn_a_benchmark_raw_image_into_the_benchmark_image(tmp_path):
    keys = sorted(f for f in (BENCH / "puml_images").iterdir() if f.suffix == ".png")[:8]
    for raw in keys:
        work = tmp_path / raw.name
        shutil.copy(raw, work)
        added = R._finish_one((str(work), str(tmp_path / ("out_" + raw.name))))
        assert "error" not in added
        assert added["pixels_sha256"] == R.pixels_sha256(BENCH / "puml_images_1568" / raw.name)


# --------------------------------------------------------------------------- #
# Reading the renderer's outputs
# --------------------------------------------------------------------------- #
def test_outputs_are_assigned_to_keys_by_the_test_set_rule(tmp_path):
    for name in ("a.png", "b.png", "b_001.png", "b_002.png", "c_01.png", "c_01_001.png",
                 "myname.png", "dpi.cfg"):
        (tmp_path / name).write_bytes(b"")
    pages, stray = R.map_outputs(["a", "b", "c_01", "d"], tmp_path)
    assert pages == {"a": ["a.png"], "b": ["b.png", "b_001.png", "b_002.png"],
                     "c_01": ["c_01.png", "c_01_001.png"], "d": []}
    assert stray == ["myname.png"]


def test_stderr_is_read_for_traces_and_per_file_errors():
    stderr = (FIX / "oom_stderr.txt").read_text() + "\n".join([
        "Error line 2 in file: /w/in/broken.puml",
        "Warning: no image in /w/in/nodiagram.puml",
        "Some diagram description contains errors",
        'Exception in thread "main" java.lang.StackOverflowError',
        "Caused by: java.lang.IllegalStateException: inner",
        "\tat net.sourceforge.plantuml.Run.main(Run.java:1)", ""])
    traces, syntax, nodiagram = R.read_stderr(stderr)
    assert traces == ["java.lang.OutOfMemoryError", "java.lang.OutOfMemoryError", "java.lang.StackOverflowError"]
    assert syntax == {"broken.puml"} and nodiagram == {"nodiagram.puml"}
    assert R.read_stderr("Error: Dot executable does not exist\n") == ([], set(), set())
    assert R.read_stderr((FIX / "logged_exception_stderr.txt").read_text())[0] == ["java.lang.IllegalStateException"]


def test_a_trace_is_attributed_to_the_source_read_before_it():
    found = R.attribute_traces((FIX / "verbose_stderr.txt").read_text())
    assert found == {"7b907a2193a605d50cf838ce9ddf8fd1ff761ac0.puml": ["java.lang.IllegalStateException"]}
    assert R.attribute_traces("java.lang.OutOfMemoryError: Java heap space\n") == {None: ["java.lang.OutOfMemoryError"]}


@pytest.mark.parametrize("picture", [
    "crash_picture_481x330.png",              # out of memory
    "crash_picture_946x711.png",              # out of memory, longer trace
    "crash_picture_stack_overflow_top.png",   # top 400 rows of a 771 x 10,378 px picture
    "crash_picture_graphviz_top.png",         # Graphviz output PlantUML could not parse
])
def test_the_crash_picture_is_recognised_whatever_the_error(picture):
    assert R.is_crash_image(FIX / picture)


def test_each_crash_mark_alone_recognises_the_picture(monkeypatch):
    for mark in R.CRASH_MARKS:
        monkeypatch.setattr(R, "CRASH_MARKS", (mark,))
        assert all(R.is_crash_image(p) for p in FIX.glob("crash_picture_*.png"))


def test_an_ordinary_image_is_not_a_crash_picture(tmp_path):
    Image.new("RGB", (600, 400), "white").save(tmp_path / "blank.png")
    Image.new("RGB", (100, 10), "white").save(tmp_path / "tiny.png")
    assert not R.is_crash_image(tmp_path / "blank.png")
    assert not R.is_crash_image(tmp_path / "tiny.png")
    for raw in sorted((BENCH / "puml_images").glob("*.png"))[:40]:
        assert not R.is_crash_image(raw)


def test_key_lists_are_read_in_three_layouts(tmp_path):
    (tmp_path / "k.txt").write_text("a.puml\n\nb.puml\n")
    (tmp_path / "k.json").write_text(json.dumps(["a.puml", "b.puml"]))
    (tmp_path / "t.json").write_text(json.dumps({"diagrams": [{"key": "a.puml"}, {"key": "b.puml"}]}))
    assert R.read_keys(tmp_path / "k.txt") == R.read_keys(tmp_path / "k.json") \
        == R.read_keys(tmp_path / "t.json") == ["a.puml", "b.puml"]


# --------------------------------------------------------------------------- #
# The decision to render a diagram again alone (no JVM: run_plantuml is replaced)
# --------------------------------------------------------------------------- #
class InlinePool:
    map = staticmethod(lambda fn, items: [fn(i) for i in items])


OOM = "java.lang.OutOfMemoryError: Java heap space\n\tat java.desktop/java.awt.image.DataBufferInt.<init>(DataBufferInt.java:76)\n"
LOGGED = "java.lang.IllegalStateException\n\tat net.sourceforge.plantuml.svek.Bibliotekon.getNodeUid(Bibliotekon.java:137)\n"


def fake_renderer(calls, batch_stderr="", batch_exit=0, solo_stderr="", solo_exit=0, crash_for=(),
                  corrupt_for=(), logged_for=(), named_block_for=()):
    """Stands in for run_plantuml: one PNG per source. The batch run writes the crash picture for
    the keys in crash_for and an unreadable file for the keys in corrupt_for; the -verbose run
    reports a logged exception for the keys in logged_for; the keys in named_block_for get a second
    PNG under another name, as a named @startuml block does. An exit of None is a timeout."""
    def run(java, env, jar, in_dir, out_dir, threads, heap_mb, timeout, verbose=False):
        out_dir.mkdir(parents=True, exist_ok=True)
        sources = sorted(in_dir.glob("*.puml"))
        kind = "attribution" if verbose else "alone" if threads == 1 else "batch"
        calls.append((kind, [p.name for p in sources]))
        for src in sources:
            if src.name in crash_for and kind == "batch":
                shutil.copy(FIX / "crash_picture_481x330.png", out_dir / (src.stem + ".png"))
            elif src.name in corrupt_for and kind == "batch":
                (out_dir / (src.stem + ".png")).write_bytes(b"not a png")
            else:
                Image.new("RGBA", (700, 600), "white").save(out_dir / (src.stem + ".png"))
            if src.name in named_block_for:
                Image.new("RGBA", (700, 600), "white").save(out_dir / (src.stem + "block.png"))
        if kind == "batch":
            return batch_exit, batch_stderr
        if kind == "alone":
            return solo_exit, solo_stderr
        return 0, "".join(f"(0.1 - 60 Mo) 48 Mo - Reading file: /w/in/{p.name}\n"
                          + (LOGGED if p.name in logged_for else "") for p in sources)
    return run


def run_fake_batch(tmp_path, monkeypatch, sources=None, **kw):
    src = tmp_path / "src"
    src.mkdir()
    keys = ["a.puml", "b.puml", "c.puml"]
    for k in keys:
        (src / k).write_text((sources or {}).get(k, "@startuml\nclass A\n@enduml\n"))
    calls = []
    monkeypatch.setattr(R, "run_plantuml", fake_renderer(calls, **kw))
    ctx = dict(java="java", env={}, jar="JAR", threads=3, heap_mb=1, solo_heap_mb=1,
               batch_timeout=1, solo_timeout=1, log=lambda s: None)
    rows, pngs = R.render_batch(keys, src, tmp_path / "work", ctx, InlinePool())
    return rows, pngs, [kind for kind, _ in calls]


def test_a_clean_batch_is_rendered_once(tmp_path, monkeypatch):
    rows, pngs, calls = run_fake_batch(tmp_path, monkeypatch)
    assert calls == ["batch"]
    assert rows == {k: dict(status="ok", attempt="batch", render_size=[700, 600], render_mode="RGBA",
                            render_sha256=rows[k]["render_sha256"]) for k in ("a.puml", "b.puml", "c.puml")}
    assert set(pngs) == set(rows)


def test_a_crash_picture_is_replaced_by_a_render_alone(tmp_path, monkeypatch):
    rows, pngs, calls = run_fake_batch(tmp_path, monkeypatch, batch_stderr=OOM, crash_for={"b.puml"})
    assert calls == ["batch", "alone"]
    assert [rows[k]["attempt"] for k in ("a.puml", "b.puml", "c.puml")] == ["batch", "solo", "batch"]
    assert rows["b.puml"]["status"] == "ok" and not R.is_crash_image(pngs["b.puml"])


def test_a_crash_picture_without_any_trace_is_still_caught(tmp_path, monkeypatch):
    rows, _, calls = run_fake_batch(tmp_path, monkeypatch, crash_for={"b.puml"})
    assert calls == ["batch", "alone"] and rows["b.puml"]["attempt"] == "solo"


def test_a_fatal_trace_without_its_crash_picture_sends_the_whole_batch_to_single_renders(tmp_path, monkeypatch):
    rows, _, calls = run_fake_batch(tmp_path, monkeypatch, batch_stderr=OOM)
    assert calls == ["batch", "alone", "alone", "alone"]
    assert [r["attempt"] for r in rows.values()] == ["solo"] * 3


def test_an_abnormal_exit_sends_the_whole_batch_to_single_renders(tmp_path, monkeypatch):
    rows, _, calls = run_fake_batch(tmp_path, monkeypatch, batch_exit=1)
    assert calls == ["batch", "alone", "alone", "alone"]
    assert [r["attempt"] for r in rows.values()] == ["solo"] * 3


def test_a_batch_that_times_out_is_rendered_one_diagram_at_a_time(tmp_path, monkeypatch):
    rows, _, calls = run_fake_batch(tmp_path, monkeypatch, batch_exit=None)
    assert calls == ["batch", "alone", "alone", "alone"]
    assert [r["attempt"] for r in rows.values()] == ["solo"] * 3


def test_a_diagram_that_times_out_alone_is_dropped(tmp_path, monkeypatch):
    rows, pngs, _ = run_fake_batch(tmp_path, monkeypatch, crash_for={"b.puml"}, solo_exit=None)
    assert rows["b.puml"] == dict(status="timeout") and set(pngs) == {"a.puml", "c.puml"}


def test_an_unreadable_output_is_rendered_again_alone(tmp_path, monkeypatch):
    rows, pngs, calls = run_fake_batch(tmp_path, monkeypatch, corrupt_for={"b.puml"})
    assert calls == ["batch", "alone"]
    assert (rows["b.puml"]["status"], rows["b.puml"]["attempt"]) == ("ok", "solo")
    assert Image.open(pngs["b.puml"]).size == (700, 600)


def test_stage_errors_are_reported_not_raised(tmp_path):
    (tmp_path / "bad.png").write_bytes(b"not a png")
    assert "error" in R._inspect_one(tmp_path / "bad.png")
    assert "error" in R._finish_one((str(tmp_path / "bad.png"), str(tmp_path / "out.png")))
    assert not (tmp_path / "out.png").exists()


def test_a_diagram_that_crashes_alone_is_dropped(tmp_path, monkeypatch):
    rows, pngs, _ = run_fake_batch(tmp_path, monkeypatch, batch_stderr=OOM, crash_for={"b.puml"},
                                   solo_stderr=OOM)
    assert rows["b.puml"] == dict(status="crash", detail="java.lang.OutOfMemoryError")
    assert set(pngs) == {"a.puml", "c.puml"}


def test_a_logged_exception_keeps_the_diagram_and_is_recorded_on_it(tmp_path, monkeypatch):
    rows, pngs, calls = run_fake_batch(tmp_path, monkeypatch, batch_stderr=LOGGED, logged_for={"b.puml"})
    assert calls == ["batch", "attribution"]
    assert [(r["status"], r["attempt"]) for r in rows.values()] == [("ok", "batch")] * 3
    assert rows["b.puml"]["renderer_trace"] == ["java.lang.IllegalStateException"]
    assert "renderer_trace" not in rows["a.puml"] and "renderer_trace" not in rows["c.puml"]
    assert set(pngs) == set(rows)


def test_a_png_without_an_owner_sends_the_whole_batch_to_single_renders(tmp_path, monkeypatch):
    # The three sources have the same text: which of them wrote the second PNG shows only alone.
    rows, pngs, calls = run_fake_batch(tmp_path, monkeypatch, named_block_for={"b.puml"})
    assert calls == ["batch", "alone", "alone", "alone"]
    assert rows["b.puml"] == dict(status="multi_page", pages=2) and set(pngs) == {"a.puml", "c.puml"}
    assert [(rows[k]["status"], rows[k]["attempt"]) for k in ("a.puml", "c.puml")] == [("ok", "solo")] * 2


def test_a_logged_exception_in_a_render_alone_is_recorded_too(tmp_path, monkeypatch):
    rows, _, calls = run_fake_batch(tmp_path, monkeypatch, batch_stderr=OOM, crash_for={"b.puml"},
                                    solo_stderr=LOGGED)
    assert calls == ["batch", "alone"]
    assert (rows["b.puml"]["status"], rows["b.puml"]["attempt"]) == ("ok", "solo")
    assert rows["b.puml"]["renderer_trace"] == ["java.lang.IllegalStateException"]


# --------------------------------------------------------------------------- #
# With PlantUML
# --------------------------------------------------------------------------- #
SOURCES = {
    "plain.puml": "' h1\n' h2\n' h3\n\n@startuml\nclass A\nclass B\nA --> B\n@enduml\n",
    "named.puml": "@startuml myname\nclass A\nA --> B\n@enduml\n",
    "paged.puml": "@startuml\nAlice -> Bob: hi\nnewpage\nBob -> Alice: yo\n@enduml\n",
    "twoblock.puml": "@startuml\nclass A\n@enduml\n\n@startuml\nclass B\n@enduml\n",
    "unnamed_then_named.puml": "@startuml\nclass A\n@enduml\n\n@startuml second\nclass B\n@enduml\n",
    "named_then_unnamed.puml": "@startuml first\nclass A\n@enduml\n\n@startuml\nclass B\n@enduml\n",
    # spellings of a block start that PlantUML accepts: upper case, a backslash, a comment prefix
    "second_upper.puml": "@startuml\nclass A\n@enduml\n\n@STARTUML upper\nclass B\n@enduml\n",
    "second_backslash.puml": "@startuml\nclass A\n@enduml\n\n\\startuml slash\nclass B\n\\enduml\n",
    "second_commented.puml": "@startuml\nclass A\n@enduml\n\n// @startuml commented\n// class B\n// @enduml\n",
    "broken.puml": "@startuml\nclass A {\nthis is ((( not valid ->>> \n@enduml\n",
    "nodiagram.puml": "just text, no diagram\n",
    "abc_001.puml": "@startuml\nAlice -> Bob: hi\n@enduml\n",
}


@pytest.fixture(scope="module")
def small_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("small")
    src = root / "src"
    src.mkdir()
    for name, text in SOURCES.items():
        (src / name).write_text(text)
    keys = sorted(SOURCES) + ["absent.puml"]
    summary = R.render(keys, src, root / "out", threads=2, heap_mb=2048, batch_size=3, keep_raw=True,
                       log=lambda s: None)
    return root, src, keys, summary


def test_each_source_gets_one_manifest_line_with_its_outcome(small_run):
    root, _, keys, summary = small_run
    rows = manifest(root / "out")
    assert list(rows) == keys
    outcome = {k: (r["status"], r.get("reason", r.get("pages"))) for k, r in rows.items()}
    assert outcome == {
        "abc_001.puml": ("ok", None), "plain.puml": ("ok", None),
        "absent.puml": ("missing_source", None),
        "broken.puml": ("no_image", "syntax_error"),
        "nodiagram.puml": ("no_image", "no_diagram"),
        "named.puml": ("no_image", "named_output"),
        "paged.puml": ("multi_page", 2), "twoblock.puml": ("multi_page", 2),
        "unnamed_then_named.puml": ("multi_page", 2), "named_then_unnamed.puml": ("multi_page", 2),
        "second_upper.puml": ("multi_page", 2), "second_backslash.puml": ("multi_page", 2),
        "second_commented.puml": ("multi_page", 2),
    }
    assert summary["status"] == {"ok": 2, "missing_source": 1, "no_image": 3, "multi_page": 7}
    assert summary["requested_not_in_manifest"] == 0


def test_only_one_page_diagrams_get_an_image(small_run):
    root = small_run[0]
    for folder in ("puml_images_1568", "puml_images"):
        assert sorted(p.name for p in (root / "out" / folder).iterdir()) == ["abc_001.png", "plain.png"]
    assert not (root / "out" / "_work").exists()


def test_an_ok_line_carries_sizes_and_hashes_of_every_stage(small_run):
    root = small_run[0]
    for key in ("plain.puml", "abc_001.puml"):
        row = manifest(root / "out")[key]
        name = key[:-5] + ".png"
        w, h = row["render_size"]
        # both share a batch of three with a source that leaves a PNG under another name
        assert min(w, h) < 512 and row["stage2"] == "padded" and row["attempt"] == "solo"
        assert len(row["render_sha256"]) == 64 and row["render_mode"]
        assert row["padded_size"] == [max(w, 512), max(h, 512)] == list(Image.open(root / "out/puml_images" / name).size)
        final = Image.open(root / "out/puml_images_1568" / name)
        assert row["size"] == row["padded_size"] == list(final.size) and final.mode == "RGB"
        assert row["pixels_sha256"] == R.pixels_sha256(root / "out/puml_images_1568" / name)


def test_the_environment_is_recorded(small_run):
    root = small_run[0]
    env = json.loads((root / "out/render_env.json").read_text())
    assert env["renderer"]["sha256"] == R.PINS["renderer_jar"]
    assert env["renderer"]["version"] == "PlantUML version 1.2025.9"
    assert "graphviz version" in env["graphviz"] and env["java"] and env["pillow"] and env["platform"]
    assert [env["stage_code"][s]["sha256"] for s in ("stage2", "stage3")] == [
        R.PINS["stage2_script"], R.PINS["stage3_script"]]
    assert {k: env["parameters"][k] for k in ("dpi", "limit_size", "min_side", "long_edge")} == {
        "dpi": 300, "limit_size": 16384, "min_side": 512, "long_edge": 1568}
    assert env["runs"][0]["threads"] == 2 and env["runs"][0]["heap_mb"] == 2048
    assert env["dot_guard"]["sha256"] == R.sha256_file(R.DOT_GUARD)
    assert Path(env["dot_guard"]["real_dot"]).name == "dot"


def test_a_second_run_renders_only_new_keys_and_another_environment_is_refused(small_run):
    root, src, keys, _ = small_run
    before = (root / "out/manifest.jsonl").read_text()
    log = []
    R.render(keys, src, root / "out", threads=2, heap_mb=2048, log=log.append)
    assert (root / "out/manifest.jsonl").read_text() == before and "0 to render" in log[0]
    (src / "new.puml").write_text(SOURCES["abc_001.puml"])
    R.render(keys + ["new.puml"], src, root / "out", threads=2, heap_mb=2048, log=lambda s: None)
    after = (root / "out/manifest.jsonl").read_text()
    assert after.startswith(before) and len(after.splitlines()) == len(before.splitlines()) + 1
    assert manifest(root / "out")["new.puml"]["status"] == "ok"
    assert len(json.loads((root / "out/render_env.json").read_text())["runs"]) == 3

    env_path = root / "out/render_env.json"
    env = json.loads(env_path.read_text())
    env["graphviz"] = "dot - graphviz version 2.43.0 (0)"
    env_path.write_text(json.dumps(env))
    with pytest.raises(R.RenderError, match="graphviz"):
        R.render(keys, src, root / "out", log=lambda s: None)


def test_the_renderer_is_stopped_at_the_timeout(tmp_path):
    (tmp_path / "in").mkdir()
    (tmp_path / "in/a.puml").write_text(SOURCES["plain.puml"])
    java, env = R.java_env()
    code, _ = R.run_plantuml(java, env, R.RENDERER_JAR, tmp_path / "in", tmp_path / "out", 1, 512, timeout=0.05)
    assert code is None


def test_the_run_stops_when_plantuml_cannot_find_graphviz(tmp_path, monkeypatch):
    monkeypatch.setenv("GRAPHVIZ_DOT", "/nonexistent/dot")
    (tmp_path / "a.puml").write_text(SOURCES["plain.puml"])
    with pytest.raises(R.RenderError, match="Graphviz"):
        R.render(["a.puml"], tmp_path, tmp_path / "out", log=lambda s: None)
    assert not (tmp_path / "out").exists()


def test_the_folder_of_the_test_images_is_refused_as_output(tmp_path):
    before = sorted(p.name for p in (BENCH / "puml_images_1568").iterdir())
    with pytest.raises(R.RenderError, match="test images"):
        R.render(["a.puml"], tmp_path, BENCH)
    with pytest.raises(R.RenderError, match="test images"):
        R.render(["a.puml"], tmp_path, BENCH / "../data")
    link = tmp_path / "elsewhere"
    link.symlink_to(BENCH, target_is_directory=True)
    spellings = [link, link / "not_there" / "..", BENCH / "not_there" / "..",
                 BENCH / "not_there" / "deeper" / ".." / "..", BENCH.parent / "not_there" / ".." / "data"]
    if (BENCH.parent / "DATA").exists():   # a volume that does not tell letter case apart
        spellings += [BENCH.parent / "DATA", BENCH.parent / "DATA" / "not_there" / ".."]
    for out in spellings:
        with pytest.raises(R.RenderError, match="test images"):
            R.render(["a.puml"], tmp_path, out)
    assert not (BENCH / "not_there").exists() and not (BENCH.parent / "not_there").exists()
    assert sorted(p.name for p in (BENCH / "puml_images_1568").iterdir()) == before
    assert not (BENCH / "manifest.jsonl").exists() and not (BENCH / "render_env.json").exists()


def test_keys_are_rendered_in_batches_of_the_given_size(tmp_path, monkeypatch):
    batches = []

    def batch(keys, src_dir, work, ctx, pool):
        batches.append(list(keys))
        return {k: dict(status="missing_source") for k in keys}, {}
    monkeypatch.setattr(R, "render_batch", batch)
    keys = [f"k{i}.puml" for i in range(7)]
    summary = R.render(keys, tmp_path, tmp_path / "out", batch_size=3, log=lambda s: None)
    assert batches == [keys[0:3], keys[3:6], keys[6:7]]
    assert summary["status"] == {"missing_source": 7} and list(manifest(tmp_path / "out")) == keys


def test_bad_key_lists_are_refused(tmp_path):
    with pytest.raises(R.RenderError, match="ending in .puml"):
        R.render(["a.png"], tmp_path, tmp_path / "out")
    with pytest.raises(R.RenderError, match="duplicate"):
        R.render(["a.puml", "a.puml"], tmp_path, tmp_path / "out")


# --------------------------------------------------------------------------- #
# Graphviz that ends without output
# --------------------------------------------------------------------------- #
GRAPH = b"digraph { a -> b; b -> c }"
# 14 classes in a chain: more than 1,000 bytes of dot text, so the stand-in dot dies on it
CHAIN = ("".join(f"class C{i} {{\n  +field{i}: int\n  +method{i}()\n}}\n" for i in range(14))
         + "".join(f"C{i} --> C{i + 1}\n" for i in range(13)))


def run_guard(real, *args, stdin=GRAPH):
    return subprocess.run([str(R.DOT_GUARD), *args], input=stdin, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, env=dict(os.environ, RENDER_REAL_DOT=str(real)))


def test_the_dot_guard_passes_the_real_dot_on_unchanged():
    java, env = R.java_env()
    assert env["GRAPHVIZ_DOT"] == str(R.DOT_GUARD)
    assert "GRAPHVIZ_DOT" not in R.java_env(guard=False)[1]
    real = env["RENDER_REAL_DOT"]
    os.environ["GRAPHVIZ_DOT"] = str(R.DOT_GUARD)        # a caller that names the script itself
    try:
        assert R.java_env()[1]["RENDER_REAL_DOT"] == real
        assert "GRAPHVIZ_DOT" not in R.java_env(guard=False)[1]
    finally:
        del os.environ["GRAPHVIZ_DOT"]
    for args, stdin in ((["-Tsvg"], GRAPH), (["-V"], b"")):
        direct = subprocess.run([real, *args], input=stdin, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        guarded = run_guard(real, *args, stdin=stdin)
        assert (guarded.returncode, guarded.stdout, guarded.stderr) == (direct.returncode, direct.stdout, direct.stderr)
    assert direct.stdout == b"" and b"graphviz version" in direct.stderr      # -V
    assert b"<svg" in run_guard(real, "-Tsvg").stdout


def test_the_dot_guard_reports_a_dot_that_ends_without_output(tmp_path):
    for code in (139, 0):
        silent = tmp_path / f"dot{code}"
        silent.write_text(f"#!/bin/sh\ncat > /dev/null\nexit {code}\n")
        silent.chmod(0o755)
        r = run_guard(silent, "-Tsvg")
        assert (r.returncode, r.stdout) == (code, b"Error: dot ended without output (exit code %d)\n" % code)


def dying_dot(monkeypatch, tmp_path, deaths=None):
    """Graphviz replaced by fixtures/dot_dying_silently.sh for the renders that follow."""
    monkeypatch.setenv("DYING_DOT_REAL", R.java_env()[1]["RENDER_REAL_DOT"])
    monkeypatch.setenv("DYING_DOT_STATE", str(tmp_path / "dot_state"))
    monkeypatch.setenv("GRAPHVIZ_DOT", str(FIX / "dot_dying_silently.sh"))
    if deaths is not None:
        monkeypatch.setenv("DYING_DOT_DEATHS", str(deaths))


def test_a_diagram_on_which_graphviz_ends_without_output_never_reaches_the_image_set(tmp_path, monkeypatch):
    # Without the guard PlantUML draws its "dot/GraphViz has crashed" picture for each of these, in
    # the diagram's scale and colours (black on black under the theme), with exit code 0.
    settings = {"chain": "", "scaled": "scale 2\n", "titled": "title The title\n",
                "dark": "skinparam backgroundColor #222222\n", "themed": "!theme cyborg\n"}
    src = tmp_path / "src"
    src.mkdir()
    for name, first in settings.items():
        (src / f"{name}.puml").write_text("@startuml\n" + first + CHAIN + "@enduml\n")
    (src / "plain.puml").write_text(SOURCES["plain.puml"])
    dying_dot(monkeypatch, tmp_path)
    R.render([f"{n}.puml" for n in settings] + ["plain.puml"], src, tmp_path / "out", threads=2,
             heap_mb=2048, log=lambda s: None)
    rows = manifest(tmp_path / "out")
    assert {k: r["status"] for k, r in rows.items()} == {
        **{f"{n}.puml": "crash" for n in settings}, "plain.puml": "ok"}
    assert [p.name for p in (tmp_path / "out/puml_images_1568").iterdir()] == ["plain.png"]


def test_a_diagram_on_which_graphviz_died_once_is_rendered_again_alone(tmp_path, monkeypatch):
    src = tmp_path / "src"
    src.mkdir()
    (src / "chain.puml").write_text("@startuml\n" + CHAIN + "@enduml\n")
    (src / "plain.puml").write_text(SOURCES["plain.puml"])
    keys = ["chain.puml", "plain.puml"]
    R.render(keys, src, tmp_path / "control", threads=2, heap_mb=2048, log=lambda s: None)
    dying_dot(monkeypatch, tmp_path, deaths=1)
    R.render(keys, src, tmp_path / "out", threads=2, heap_mb=2048, log=lambda s: None)
    control, rows = manifest(tmp_path / "control"), manifest(tmp_path / "out")
    assert [(rows[k]["status"], rows[k]["attempt"]) for k in keys] == [("ok", "solo"), ("ok", "batch")]
    assert control["chain.puml"]["attempt"] == "batch"
    for k in keys:   # the drawing of the working dot, not the one of PlantUML's other layout mode
        assert rows[k]["render_sha256"] == control[k]["render_sha256"]
    assert len(list((tmp_path / "dot_state").glob("death.*"))) == 1


# The two largest test diagrams (141 and 71 megapixels at 300 DPI) cannot be drawn in a 96 MB heap.
HEAVY = ["b98c5fc6649e2bb3ceff143dbc060d141396ff62.puml", "fa6d6a0cba01276779f8a97b6857b3e3efbf67e1.puml"]
LIGHT = "000133b38e70cfb70834681221553363e9f37714.puml"


TRACED = "7b907a2193a605d50cf838ce9ddf8fd1ff761ac0.puml"   # PlantUML logs an exception for one relationship


def test_a_test_diagram_with_a_logged_exception_is_rendered_as_in_the_benchmark(tmp_path):
    R.render([LIGHT, TRACED], BENCH / "puml_files", tmp_path / "out", threads=2, heap_mb=2048,
             log=lambda s: None)
    rows = manifest(tmp_path / "out")
    assert (rows[TRACED]["status"], rows[TRACED]["attempt"]) == ("ok", "batch")
    assert rows[TRACED]["renderer_trace"] == ["java.lang.IllegalStateException"]
    assert "renderer_trace" not in rows[LIGHT]
    for k in (LIGHT, TRACED):
        assert rows[k]["pixels_sha256"] == R.pixels_sha256(BENCH / "puml_images_1568" / (k[:-5] + ".png"))
    assert json.loads((tmp_path / "out/summary.json").read_text())["renderer_trace"] == 1


def test_a_stack_overflow_crash_picture_never_reaches_the_image_set(tmp_path):
    n = 4000
    (tmp_path / "deep.puml").write_text(
        "@startuml\n" + "".join(f"package p{i} {{\n" for i in range(n)) + "class A\n" + "}\n" * n + "@enduml\n")
    (tmp_path / "plain.puml").write_text(SOURCES["plain.puml"])
    R.render(["deep.puml", "plain.puml"], tmp_path, tmp_path / "out", threads=2, heap_mb=2048,
             log=lambda s: None)
    rows = manifest(tmp_path / "out")
    assert rows["deep.puml"] == dict(key="deep.puml", status="crash", detail="java.lang.StackOverflowError")
    assert rows["plain.puml"]["status"] == "ok"
    assert [p.name for p in (tmp_path / "out/puml_images_1568").iterdir()] == ["plain.png"]


def test_out_of_memory_crash_pictures_never_reach_the_image_set(tmp_path):
    R.render(HEAVY + [LIGHT], BENCH / "puml_files", tmp_path / "out", threads=3, heap_mb=96,
             log=lambda s: None)
    rows = manifest(tmp_path / "out")
    assert [rows[k]["status"] for k in HEAVY] == ["crash", "crash"]
    assert rows[LIGHT]["status"] == "ok"
    assert [p.name for p in (tmp_path / "out/puml_images_1568").iterdir()] == [LIGHT[:-5] + ".png"]


def test_a_diagram_that_crashed_in_the_batch_is_recovered_by_the_render_alone(tmp_path):
    R.render(HEAVY + [LIGHT], BENCH / "puml_files", tmp_path / "out", threads=3, heap_mb=96,
             solo_heap_mb=8192, log=lambda s: None)
    rows = manifest(tmp_path / "out")
    for k in HEAVY:
        name = k[:-5] + ".png"
        assert (rows[k]["status"], rows[k]["attempt"]) == ("ok", "solo")
        assert rows[k]["padded_size"] == list(Image.open(BENCH / "puml_images" / name).size)
        assert rows[k]["size"] == list(Image.open(BENCH / "puml_images_1568" / name).size)
        assert not R.is_crash_image(tmp_path / "out/puml_images_1568" / name)
