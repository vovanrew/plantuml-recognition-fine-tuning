#!/usr/bin/env python3
"""Render PlantUML sources to the benchmark's image standard.

One command for the three stages that produced the test images
(plantuml-recognition-evaluation/data/puml_images_1568):

  1. PlantUML 1.2025.9 with `skinparam dpi 300` and PLANTUML_LIMIT_SIZE 16384: the java command of
     dataset/plantuml-in-wild-dataset/phase2/4.generate_images/generate_images_hidpi.sh, with the
     heap and the thread count set here instead of 8 GB and `--threads auto`.
  2. Pad any side below 512 px and convert to RGB: `postprocess_one` of
     dataset/plantuml-in-wild-dataset/phase2/5.postprocess_images/postprocess_images.py.
  3. Flatten transparency on white and downscale the long edge to 1,568 px:
     `standardize` of plantuml-recognition-evaluation/util/standardize_images.py.

Stages 2 and 3 run the code of those two files, loaded by path. The renderer JAR and both files are
pinned by sha256 (PINS); a file that differs stops the run.

What stage 1 adds to the stock script:
  - sources are rendered in bounded batches, each into its own directory;
  - PlantUML answers an error it cannot recover from (out of memory, stack overflow) with a
    "PlantUML has crashed" picture: a valid PNG written in place of the diagram, with exit code 0.
    Every output is tested for that picture and the renderer's stderr for Java traces; a diagram
    that got the picture is rendered again alone and dropped if it gets it again;
  - PlantUML also logs some exceptions and still draws the diagram, without the element that
    raised it. Such a trace is attributed to its diagram and recorded as `renderer_trace`;
  - Graphviz dot is called through dot_guard.sh, which passes its output on unchanged and reports
    a dot that ended without output. PlantUML answers that with the same crash picture; without
    the script it would draw the diagram in another layout mode or draw a failure picture that
    cannot be told from a diagram;
  - a diagram counts as rendered only if its source produced exactly one image, under its own
    name. A source with a second block that PlantUML writes under the block's name is not one.

Output directory:
  puml_images_1568/<base>.png   one image per diagram with status "ok"
  puml_images/<base>.png        the image after stage 2, only with --keep-raw
  manifest.jsonl                one line per key: status, sizes at each stage, hashes, traces
  render_env.json               renderer, Java, Graphviz, Pillow, stage code hashes, parameters
  summary.json                  counts per status

A run resumes: keys already in manifest.jsonl are skipped, and the environment must equal the one
in render_env.json.

Usage (from fine-tune/):
  python3 build/render.py --src-dir <dir of <key>> --keys <file> --out <dir>
"""
import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import types
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import PIL
from PIL import Image, PngImagePlugin

Image.MAX_IMAGE_PIXELS = None   # trusted corpus; renders reach 16,384 px on a side
PngImagePlugin.MAX_TEXT_CHUNK = 100 * 1024 * 1024   # as stage 2 sets it: PlantUML embeds the source

UNI = Path(__file__).resolve().parents[2]
_STAGE1_DIR = UNI / "dataset/plantuml-in-wild-dataset/phase2/4.generate_images"
RENDERER_JAR = _STAGE1_DIR / "plantuml-1.2025.9.jar"
STAGE1_SCRIPT = _STAGE1_DIR / "generate_images_hidpi.sh"
STAGE2_SCRIPT = UNI / "dataset/plantuml-in-wild-dataset/phase2/5.postprocess_images/postprocess_images.py"
STAGE3_SCRIPT = UNI / "plantuml-recognition-evaluation/util/standardize_images.py"
TEST_IMAGES = UNI / "plantuml-recognition-evaluation/data"   # the benchmark's puml_images and puml_images_1568
DEFAULT_JAVA_HOME = "/Library/Java/JavaVirtualMachines/temurin-26.jdk/Contents/Home"
DOT_GUARD = Path(__file__).resolve().parent / "dot_guard.sh"

# sha256 of the files the test images were made with
PINS = {
    "renderer_jar": "280f4aa97f4f3b7af39358dd0d79f65e353b54a1630b5f121a13b1651162185d",
    "stage1_script": "68722e410a71b125fb91663c4505362a4ef544a03f8a1a5574b39fef5b300e0e",
    "stage2_script": "53742a3d8ec4414bbc4829a3b5d5c92f21be5754ffdafb2bb5c6fb8925c4b3fc",
    "stage3_script": "726d80a77092e817d517807e77043c61a55f2e9b659631b8a3f01aff21b8caf3",
}

DPI = 300
LIMIT_SIZE = 16384

# Two text lines of the crash picture, each binarised at 128 inside a box and hashed (MD5). The
# first box holds "An error has occured : ja", the start of line 1; the second holds
# "PlantUML (1.2025.9) has crashed.", line 4. Neither depends on the error: both hashes are the
# same on out-of-memory and on stack-overflow pictures. The first is the signature of the 528
# crash pictures of decisions/training-pool-rules (challenge finding 10).
CRASH_MARKS = (((0, 0, 150, 20), "baa68a88a4079e94808f72adce4f2d17"),
               ((0, 40, 200, 62), "dfcbc4863a4a4bf9230244c08d72a060"))

_TRAILING_PAGE = re.compile(r"_\d+$")   # as in plantuml-recognition-evaluation/util/build_test_set.py
_TRACE = re.compile(r'^(?:Exception in thread "[^"]*" )?((?:[\w$]+\.)+[\w$]*(?:Error|Exception))\b', re.M)
_SYNTAX_ERROR = re.compile(r"^Error line \d+ in file: (.*)$", re.M)
_NO_DIAGRAM = re.compile(r"^Warning: no image in (.*)$", re.M)
_READING = re.compile(r" - Reading file: (.*)$")   # -verbose, one line per source
_FATAL = ("OutOfMemoryError", "StackOverflowError")
_NORMAL_EXIT = (0, 200)                 # 200: some source had a syntax error


class RenderError(RuntimeError):
    pass


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check_pins(jar=RENDERER_JAR, stage2=STAGE2_SCRIPT, stage3=STAGE3_SCRIPT):
    for name, path in (("renderer_jar", jar), ("stage2_script", stage2), ("stage3_script", stage3)):
        if not Path(path).is_file():
            raise RenderError(f"{name} not found: {path}")
        got = sha256_file(path)
        if got != PINS[name]:
            raise RenderError(f"{name} {path} has sha256 {got}, pinned {PINS[name]}")


# --------------------------------------------------------------------------- #
# Stages 2 and 3: the code of the two original files
# --------------------------------------------------------------------------- #
_stage_cache = {}


def _load_source(path, name, future_annotations=False):
    src = Path(path).read_text(encoding="utf-8")
    if future_annotations:   # the file's type hints need Python 3.10 without it
        src = "from __future__ import annotations\n" + src
    mod = types.ModuleType(name)
    mod.__file__ = str(path)
    exec(compile(src, str(path), "exec"), mod.__dict__)
    return mod


def stage_code(stage2=STAGE2_SCRIPT, stage3=STAGE3_SCRIPT):
    """(postprocess module, standardize module), loaded once per process."""
    key = (str(stage2), str(stage3))
    if key not in _stage_cache:
        _stage_cache[key] = (_load_source(stage2, "stage2_postprocess", future_annotations=True),
                             _load_source(stage3, "stage3_standardize"))
    return _stage_cache[key]


def pixels_sha256(path):
    with Image.open(path) as im:
        rgb = im.convert("RGB")
    return hashlib.sha256(b"%dx%d:" % rgb.size + rgb.tobytes()).hexdigest()


def is_crash_image(path):
    with Image.open(path) as im:
        for box, md5 in CRASH_MARKS:
            if im.size[0] < box[2] or im.size[1] < box[3]:
                continue
            dark = np.asarray(im.crop(box).convert("L")) < 128
            if hashlib.md5(dark.tobytes()).hexdigest() == md5:
                return True
    return False


def _inspect_one(path):
    """Stage 1 output as PlantUML wrote it: size, mode, file hash, crash-picture test."""
    try:
        with Image.open(path) as im:
            size, mode = list(im.size), im.mode
        return dict(render_size=size, render_mode=mode, render_sha256=sha256_file(path),
                    crash=is_crash_image(path))
    except Exception as e:   # an unreadable PNG is not a rendered diagram
        return dict(error=repr(e))


def _finish_one(args):
    """Stages 2 and 3 for one image. Returns the manifest fields they add."""
    raw_path, final_path = args
    try:
        postprocess, standardize = stage_code()
        action = postprocess.postprocess_one(Path(raw_path))
        padded, final = standardize.standardize(raw_path, final_path, standardize.LONG_EDGE)
        return dict(stage2=action, padded_size=list(padded), size=list(final),
                    pixels_sha256=pixels_sha256(final_path))
    except Exception as e:
        return dict(error=repr(e))


# --------------------------------------------------------------------------- #
# Stage 1: PlantUML
# --------------------------------------------------------------------------- #
def java_env(java_home=None, guard=True):
    """(java, environment). With guard=False PlantUML finds dot itself, as under the stock script."""
    home = java_home or os.environ.get("JAVA_HOME") or DEFAULT_JAVA_HOME
    java = os.path.join(home, "bin", "java")
    if not os.path.isfile(java):
        raise RenderError(f"java not found at {java}; set JAVA_HOME (the /usr/bin/java stub fails "
                          "in non-interactive shells)")
    path = os.pathsep.join([os.path.join(home, "bin"), "/opt/homebrew/bin", os.environ.get("PATH", "")])
    env = dict(os.environ, JAVA_HOME=home, PATH=path)
    # PlantUML reaches Graphviz through dot_guard.sh; the dot it would have called is handed on.
    # Without a dot nothing is set, and the start-up check of environment() stops the run.
    named = env.get("GRAPHVIZ_DOT")
    if named and os.path.exists(named) and os.path.samefile(named, DOT_GUARD):
        del env["GRAPHVIZ_DOT"]   # a caller that names the script itself: it is not the dot behind it
        named = None
    dot = named or shutil.which("dot", path=path)
    if guard and dot and os.access(dot, os.X_OK):
        env.update(GRAPHVIZ_DOT=str(DOT_GUARD), RENDER_REAL_DOT=dot)
    return java, env


def plantuml_command(java, jar, in_dir, out_dir, cfg, threads, heap_mb):
    """The java command of generate_images_hidpi.sh; only -Xmx and --threads are set here."""
    return [java, f"-DPLANTUML_LIMIT_SIZE={LIMIT_SIZE}", f"-Xmx{heap_mb}m", "-jar", str(jar),
            "--threads", str(threads), "--output-dir", str(out_dir), "-tpng", "-stdrpt",
            "--no-error-image", "-config", str(cfg), f"{in_dir}/*.puml"]


def run_plantuml(java, env, jar, in_dir, out_dir, threads, heap_mb, timeout, verbose=False):
    """Returns (exit code or None on timeout, stderr text)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = out_dir / "dpi.cfg"
    cfg.write_text(f"skinparam dpi {DPI}\n")
    cmd = plantuml_command(java, jar, in_dir, out_dir, cfg, threads, heap_mb)
    if verbose:
        cmd.insert(-1, "-verbose")
    try:
        r = subprocess.run(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                           timeout=timeout)
        return r.returncode, r.stderr.decode("utf-8", "replace")
    except subprocess.TimeoutExpired as e:
        return None, (e.stderr or b"").decode("utf-8", "replace")


def map_outputs(bases, out_dir):
    """{base: [png names]} and the PNGs no base owns.

    Same ownership rule as build_test_set.map_pages_to_bases: a stem equal to a base is that
    diagram's image; otherwise one trailing _NN is a page of the base before it.
    """
    pages = {b: [] for b in bases}
    stray = []
    for name in sorted(os.listdir(out_dir)):
        if not name.endswith(".png"):
            continue
        stem = name[:-4]
        if stem in pages:
            pages[stem].append(name)
            continue
        stripped = _TRAILING_PAGE.sub("", stem, count=1)
        if stripped != stem and stripped in pages:
            pages[stripped].append(name)
        else:
            stray.append(name)
    return pages, stray


def read_stderr(stderr):
    """(Java trace class names, keys with a syntax error, keys with no diagram)."""
    return (_TRACE.findall(stderr),
            {os.path.basename(p.strip()) for p in _SYNTAX_ERROR.findall(stderr)},
            {os.path.basename(p.strip()) for p in _NO_DIAGRAM.findall(stderr)})


def attribute_traces(verbose_stderr):
    """{key: [trace class names]} from the stderr of a one-thread -verbose run, in which every
    trace follows the "Reading file" line of its source. Traces before the first source go to None."""
    owner, found = None, {}
    for line in verbose_stderr.splitlines():
        reading = _READING.search(line)
        if reading:
            owner = os.path.basename(reading.group(1).strip())
            continue
        trace = _TRACE.match(line)
        if trace:
            found.setdefault(owner, []).append(trace.group(1))
    return found


def environment(java, env, jar, stage2=STAGE2_SCRIPT, stage3=STAGE3_SCRIPT):
    """Versions of everything that can change a pixel. Stops if PlantUML cannot run Graphviz:
    without it a class diagram renders to a notice that is a valid PNG."""
    def out(cmd):
        r = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        return r.returncode, r.stdout.decode("utf-8", "replace").strip()

    code, testdot = out([java, "-jar", str(jar), "-testdot"])
    if code != 0 or "Installation seems OK" not in testdot:
        raise RenderError(f"PlantUML cannot run Graphviz (exit {code}): {testdot}")
    graphviz = next((l.split("Dot version:", 1)[1].strip() for l in testdot.splitlines()
                     if l.startswith("Dot version:")), "")
    postprocess, standardize = stage_code(stage2, stage3)
    return {
        "renderer": {"jar": str(jar), "sha256": sha256_file(jar),
                     "version": out([java, "-jar", str(jar), "-version"])[1].split("(")[0].strip()},
        "java": out([java, "-version"])[1].splitlines(),
        "graphviz": graphviz,
        "dot_guard": {"path": str(DOT_GUARD), "sha256": sha256_file(DOT_GUARD),
                      "real_dot": env.get("RENDER_REAL_DOT")},
        "pillow": PIL.__version__,
        "platform": platform.platform(),
        "stage_code": {"stage2": {"path": str(stage2), "sha256": sha256_file(stage2)},
                       "stage3": {"path": str(stage3), "sha256": sha256_file(stage3)}},
        "parameters": {"dpi": DPI, "limit_size": LIMIT_SIZE, "min_side": postprocess.MIN_SIDE,
                       "long_edge": standardize.LONG_EDGE,
                       "plantuml_flags": plantuml_command("java", "JAR", "IN", "OUT", "CFG", "T", "H")},
    }


def _identity(env):
    """What must be equal for two runs to belong to one image set (file locations may differ)."""
    return {"renderer": (env["renderer"]["sha256"], env["renderer"]["version"]),
            "java": env["java"], "graphviz": env["graphviz"],
            "dot_guard": env.get("dot_guard", {}).get("sha256"), "pillow": env["pillow"],
            "platform": env["platform"],
            "stage_code": [env["stage_code"][s]["sha256"] for s in ("stage2", "stage3")],
            "parameters": env["parameters"]}


def environment_differences(a, b):
    ia, ib = _identity(a), _identity(b)
    return [k for k in ia if ia[k] != ib[k]]


# --------------------------------------------------------------------------- #
# One batch
# --------------------------------------------------------------------------- #
def _no_image_reason(key, syntax, nodiagram, stray=()):
    if key in syntax:
        return "syntax_error"
    if key in nodiagram:
        return "no_diagram"
    return "named_output" if stray else "unknown"


def _is_fatal(traces):
    return any(t.endswith(_FATAL) for t in traces)


def _render_solo(key, src_dir, work, ctx):
    """One diagram, one JVM, one thread. A Java trace here belongs to this diagram."""
    base = key[:-5]
    din, dout = work / "in", work / "out"
    din.mkdir(parents=True)
    shutil.copy(Path(src_dir) / key, din / key)
    code, stderr = run_plantuml(ctx["java"], ctx["env"], ctx["jar"], din, dout, 1,
                                ctx["solo_heap_mb"], ctx["solo_timeout"])
    if code is None:
        return dict(status="timeout"), None
    traces, syntax, nodiagram = read_stderr(stderr)
    pages, stray = map_outputs([base], dout)
    names = pages[base]
    if code not in _NORMAL_EXIT or _is_fatal(traces):
        return dict(status="crash", detail=traces[0] if traces else f"exit {code}"), None
    logged = dict(renderer_trace=traces) if traces else {}
    if len(names) == 1 and not stray:
        info = _inspect_one(dout / names[0])
        if info.get("error"):
            return dict(status="crash", detail=info["error"]), None
        if info.pop("crash"):
            return dict(status="crash", detail=traces[0] if traces else "crash picture"), None
        return dict(status="ok", attempt="solo", **info, **logged), dout / names[0]
    if names:   # alone, a PNG under another name is this source's too: a block with a name of its own
        return dict(status="multi_page", pages=len(names) + len(stray), **logged), None
    return dict(status="no_image", reason=_no_image_reason(key, syntax, nodiagram, stray), **logged), None


def render_batch(keys, src_dir, work, ctx, pool):
    """Stage 1 for one batch. Returns {key: manifest fields}, {key: path of its one PNG}."""
    din, dout = work / "in", work / "out"
    din.mkdir(parents=True)
    rows, pngs, present = {}, {}, []
    for k in keys:
        src = Path(src_dir) / k
        if src.is_file():
            shutil.copy(src, din / k)
            present.append(k)
        else:
            rows[k] = dict(status="missing_source")
    if not present:
        return rows, pngs
    code, stderr = run_plantuml(ctx["java"], ctx["env"], ctx["jar"], din, dout, ctx["threads"],
                                ctx["heap_mb"], ctx["batch_timeout"])
    traces, syntax, nodiagram = read_stderr(stderr)
    pages, stray = map_outputs([k[:-5] for k in present], dout)
    single = [k for k in present if len(pages[k[:-5]]) == 1]
    infos = dict(zip(single, pool.map(_inspect_one, [str(dout / pages[k[:-5]][0]) for k in single])))
    crashed = [k for k in single if infos[k].get("error") or infos[k].get("crash")]
    fatal = [t for t in traces if t.endswith(_FATAL)]
    # A trace names no file. Every fatal trace should have left a crash picture; if one did not,
    # or the JVM did not end normally, any diagram of the batch may be affected. A PNG under a
    # name no key has is a further image of some source of the batch (a block with a name of its
    # own); whose it is shows only where a source is rendered alone.
    if code not in _NORMAL_EXIT or len(fatal) > len(crashed) or stray:
        suspects = list(present)
    else:
        unexplained = [k for k in present if not pages[k[:-5]] and k not in syntax and k not in nodiagram]
        suspects = crashed + unexplained
    solo = set(suspects)
    # Traces beyond the crash pictures are exceptions PlantUML logged while it went on drawing.
    # A one-thread -verbose pass over the diagrams kept from the batch says whose they are.
    logged, kept = {}, [k for k in present if k not in solo]
    if kept and len(traces) > len(crashed):
        ain = work / "attribution/in"
        ain.mkdir(parents=True)
        for k in kept:
            shutil.copy(din / k, ain / k)
        _, verbose_stderr = run_plantuml(ctx["java"], ctx["env"], ctx["jar"], ain, work / "attribution/out",
                                         1, ctx["heap_mb"], ctx["batch_timeout"], verbose=True)
        logged = attribute_traces(verbose_stderr)
        for k in [k for k in kept if _is_fatal(logged.get(k, []))]:
            kept.remove(k)
            suspects.append(k)
    unowned = len(traces) - len(crashed) - sum(len(v) for k, v in logged.items() if k in kept)
    ctx["log"](f"    exit {code}, Java traces {len(traces)}, crash pictures {len(crashed)}, "
               f"stray outputs {len(stray)}, rendered again alone {len(suspects)}, "
               f"traces attributed {sum(len(v) for k, v in logged.items() if k is not None)}"
               + (f", traces without an owner {unowned}" if kept and unowned > 0 else ""))
    for k in kept:
        names = pages[k[:-5]]
        extra = dict(renderer_trace=logged[k]) if logged.get(k) else {}
        if len(names) == 1:
            info = infos[k]
            info.pop("crash")
            rows[k] = dict(status="ok", attempt="batch", **info, **extra)
            pngs[k] = dout / names[0]
        elif len(names) > 1:
            rows[k] = dict(status="multi_page", pages=len(names), **extra)
        else:
            rows[k] = dict(status="no_image", reason=_no_image_reason(k, syntax, nodiagram), **extra)
    for i, k in enumerate(suspects):
        rows[k], png = _render_solo(k, src_dir, work / "solo" / f"{i:05d}", ctx)
        if png is not None:
            pngs[k] = png
    return rows, pngs


# --------------------------------------------------------------------------- #
# The run
# --------------------------------------------------------------------------- #
def read_keys(path):
    """A key list: one key per line, or JSON (a list of keys, a list of records with "key", or
    the test set's layout {"diagrams": [{"key": ...}]})."""
    text = Path(path).read_text(encoding="utf-8")
    if str(path).endswith(".json"):
        data = json.loads(text)
        if isinstance(data, dict):
            data = data["diagrams"]
        return [d["key"] if isinstance(d, dict) else d for d in data]
    return [l.strip() for l in text.splitlines() if l.strip()]


def read_manifest(path):
    if not Path(path).is_file():
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def render(keys, src_dir, out_dir, threads=3, heap_mb=32768, solo_heap_mb=None, batch_size=200,
           workers=None, keep_raw=False, batch_timeout=3600, solo_timeout=900, jar=RENDERER_JAR,
           java_home=None, log=print):
    """Render `keys` (names ending in .puml, files in src_dir) into out_dir. Returns the summary."""
    out_dir = Path(out_dir)
    keys = list(keys)
    bad = [k for k in keys if not k.endswith(".puml") or "/" in k]
    if bad:
        raise RenderError(f"keys must be file names ending in .puml: {bad[:3]}")
    if len(set(keys)) != len(keys):
        raise RenderError("duplicate keys")
    target = out_dir.resolve()   # also a path through a folder that does not exist yet, then ".."
    if target == TEST_IMAGES.resolve() or (target.exists() and os.path.samefile(target, TEST_IMAGES)):
        raise RenderError(f"{out_dir} holds the test images; they are not rendered again in place")
    check_pins(jar)
    java, env = java_env(java_home)
    env_now = environment(java, env, jar)

    final_dir, raw_dir, work_root = out_dir / "puml_images_1568", out_dir / "puml_images", out_dir / "_work"
    final_dir.mkdir(parents=True, exist_ok=True)
    if keep_raw:
        raw_dir.mkdir(exist_ok=True)
    run = dict(started=time.strftime("%Y-%m-%dT%H:%M:%S%z"), threads=threads, heap_mb=heap_mb,
               solo_heap_mb=solo_heap_mb or heap_mb, batch_size=batch_size,
               render_py_sha256=sha256_file(__file__))
    env_path, manifest_path = out_dir / "render_env.json", out_dir / "manifest.jsonl"
    if env_path.is_file():
        env_old = json.loads(env_path.read_text())
        differs = environment_differences(env_old, env_now)
        if differs:
            raise RenderError(f"{env_path} records another environment (differs in {differs}); "
                              "one image set is rendered in one environment")
        env_now["runs"] = env_old.get("runs", []) + [run]
    else:
        env_now["runs"] = [run]
    env_path.write_text(json.dumps(env_now, indent=1) + "\n")

    done = {r["key"] for r in read_manifest(manifest_path)}
    todo = [k for k in keys if k not in done]
    log(f"{len(keys)} keys, {len(keys) - len(todo)} already in the manifest, {len(todo)} to render")
    ctx = dict(java=java, env=env, jar=jar, threads=threads, heap_mb=heap_mb,
               solo_heap_mb=solo_heap_mb or heap_mb, batch_timeout=batch_timeout,
               solo_timeout=solo_timeout, log=log)
    shutil.rmtree(work_root, ignore_errors=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers or min(6, os.cpu_count() or 1)) as pool, \
            open(manifest_path, "a", encoding="utf-8") as manifest:
        for n, start in enumerate(range(0, len(todo), batch_size)):
            batch = todo[start:start + batch_size]
            work = work_root / f"batch_{n:05d}"
            log(f"  batch {n}: {len(batch)} keys ({start + len(batch)}/{len(todo)}, {time.time() - t0:.0f} s)")
            rows, pngs = render_batch(batch, src_dir, work, ctx, pool)
            ok = [k for k in batch if k in pngs]
            jobs = [(str(pngs[k]), str(final_dir / (k[:-5] + ".png"))) for k in ok]
            for k, added in zip(ok, pool.map(_finish_one, jobs)):
                if added.get("error"):
                    rows[k] = dict(status="postprocess_error", detail=added["error"])
                    (final_dir / (k[:-5] + ".png")).unlink(missing_ok=True)
                    continue
                rows[k].update(added)
                if keep_raw:
                    shutil.move(str(pngs[k]), raw_dir / (k[:-5] + ".png"))
            for k in batch:
                manifest.write(json.dumps(dict(key=k, **rows[k])) + "\n")
            manifest.flush()
            shutil.rmtree(work, ignore_errors=True)
    shutil.rmtree(work_root, ignore_errors=True)
    return summarize(out_dir, keys, log)


def summarize(out_dir, keys=None, log=print):
    """Counts per status, checked against the files on disk; written to summary.json."""
    out_dir = Path(out_dir)
    rows = read_manifest(out_dir / "manifest.jsonl")
    by_key = {r["key"]: r for r in rows}
    if len(by_key) != len(rows):
        raise RenderError("manifest.jsonl holds a key twice")
    expected = {r["key"][:-5] + ".png" for r in rows if r["status"] == "ok"}
    on_disk = {f for f in os.listdir(out_dir / "puml_images_1568") if f.endswith(".png")}
    if expected != on_disk:
        raise RenderError(f"images and manifest disagree: {len(expected - on_disk)} images missing, "
                          f"{len(on_disk - expected)} without an ok line")
    summary = dict(keys=len(rows), status=dict(Counter(r["status"] for r in rows)),
                   no_image_reason=dict(Counter(r["reason"] for r in rows if r["status"] == "no_image")),
                   rendered_alone=sum(1 for r in rows if r.get("attempt") == "solo"),
                   renderer_trace=sum(1 for r in rows if r.get("renderer_trace")),
                   stage2=dict(Counter(r["stage2"] for r in rows if r["status"] == "ok")))
    if keys is not None:
        summary["requested_not_in_manifest"] = sum(1 for k in keys if k not in by_key)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    log(json.dumps(summary))
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src-dir", required=True, help="directory holding the .puml sources")
    ap.add_argument("--keys", help="key list (text or JSON); default: every *.puml in --src-dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--heap-mb", type=int, default=32768)
    ap.add_argument("--solo-heap-mb", type=int, help="heap for a diagram rendered again alone (default: --heap-mb)")
    ap.add_argument("--batch-size", type=int, default=200)
    ap.add_argument("--workers", type=int, help="processes for stages 2 and 3 (default: up to 6)")
    ap.add_argument("--keep-raw", action="store_true", help="keep the images after stage 2 in puml_images/")
    ap.add_argument("--limit", type=int, default=0, help="smoke: only the first N keys")
    args = ap.parse_args()
    keys = read_keys(args.keys) if args.keys else sorted(
        f for f in os.listdir(args.src_dir) if f.endswith(".puml"))
    if args.limit:
        keys = keys[:args.limit]
    try:
        summary = render(keys, args.src_dir, args.out, threads=args.threads, heap_mb=args.heap_mb,
                         solo_heap_mb=args.solo_heap_mb, batch_size=args.batch_size,
                         workers=args.workers, keep_raw=args.keep_raw)
    except RenderError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0 if summary.get("requested_not_in_manifest", 0) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
