#!/usr/bin/env python3
"""Dev-set scoring of a checkpoint: answers through the study's server, scores under scorer v2.

decisions/pilot-run/record.md. Two steps, which may run on different machines:

  answer   on the machine that holds the model. With an adapter, first runs infer/adapter_check.py
           for it on three training pictures (check_pictures) and refuses to go on unless it
           passes (methodology/compute-and-serving.md §8, rule 3). Refuses a --port something
           already listens on. Starts infer/server.py with the checkpoint's adapter attached, not
           merged (or the base model alone), runs the benchmark's own evaluation/infer_runner.py
           over the 200 dev diagrams (data/examples/dev_runner_set.json, pictures under
           data/sets/render/dev/puml_images_1568/), stops the server, and writes the answers and
           answer_manifest.json (the server's identity, the runner's arguments, the adapter
           check's result). Before the server stops, writes the run manifest with
           infer/manifest.py write --server, so it records all its checks; exits 1 if the
           manifest does not pass. Several checkpoints can be answered at once on one card, each with its
           own server and port (train/answer_checkpoints.sh).
  score    on a machine with the scorer's Java and Graphviz. CSR and Element F1 of the answers
           under the tagged scorer v2 against the dev sources, through the benchmark's own
           evaluation/csr_runner.py and evaluation/element_f1_runner.py; writes scores.json.
           Runs only if the scorer's evaluation code equals tag `scorer-v2`, the extractor JAR and
           the render JAR have their pinned sha256, and Graphviz answers. Its paths are made
           absolute first: the runners run from the benchmark's folder.

The output cap and the per-answer deadline of a dev scoring are arguments: the recipe fixes them
(train/recipe.py DEV_MAX_TOKENS, DEV_TIMEOUT_S). Nothing here chooses a checkpoint: that is
train/select_checkpoint.py, which reads the folders `score` writes.

Usage (from fine-tune/):
  .venv/bin/python train/dev_scoring.py answer --size 2B --adapter <dir> --out <dir> --timeout <s>
  python3 train/dev_scoring.py score --answers <dir>/answers --out <dir>/scores
The scorer's runners run under --python (default /usr/bin/python3, the Python 3.9.6 scorer v2 was
measured under); scores.json records its version.

Tests: train/tests/test_dev_scoring.py
"""
import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
FT = HERE.parent
UNI = FT.parent
INSTRUMENT = UNI / "plantuml-recognition-evaluation"
SERVER = FT / "infer" / "server.py"
ADAPTER_CHECK = FT / "infer" / "adapter_check.py"
MANIFEST = FT / "infer" / "manifest.py"
ADAPTER_CHECK_PICTURES = 3
DEV_SET = FT / "data" / "examples" / "dev_runner_set.json"
DEV_IMAGES = FT / "data" / "sets" / "render" / "dev" / "puml_images_1568"
GT_DIR = UNI / "herald" / "zenodo_content" / "puml_files"
JAVA_HOME = "/Library/Java/JavaVirtualMachines/temurin-26.jdk/Contents/Home"

# Scorer v2 (methodology/scorer-v2.md §7) and the renderer CSR compiles with.
SCORER_TAG, SCORER_COMMIT = "scorer-v2", "f460b3529b18b12de7225a8d2f6ee2014835dae5"
EXTRACTOR_JAR = INSTRUMENT / "plantuml-stats-extractor-v2.jar"
EXTRACTOR_SHA256 = "2747a72aec9206e424ddf9ed659f9eceb96730444a68b1fe99094317400f555b"
RENDER_JAR = UNI / "plantuml-1.2025.9.jar"
RENDER_SHA256 = "280f4aa97f4f3b7af39358dd0d79f65e353b54a1630b5f121a13b1651162185d"
OUTPUT_CAP = 5376


class Unpinned(SystemExit):
    """A scoring input that is not the frozen one."""


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------- #
# answer
# --------------------------------------------------------------------------- #
def served_name(model, adapter):
    """The name requests ask for: the model's own, or another one with an adapter, so that a
    server left running with the other condition refuses the run (infer/server.py)."""
    return model if adapter is None else model.split("/")[-1] + "-adapter"


def port_free(port):
    """Refuse a port something already listens on (on the card, the image's nginx holds 8001):
    the identity read from it would not be the study's server's."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", port))
        except OSError:
            raise SystemExit(f"port {port} is in use; give another --port") from None


def wait_for(port, process, seconds, name):
    url = f"http://127.0.0.1:{port}/"
    for _ in range(seconds):
        if process.poll() is not None:
            raise SystemExit(f"the server exited during start with code {process.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                identity = json.loads(r.read())
        except OSError:
            time.sleep(1)
            continue
        except ValueError:
            raise SystemExit(f"port {port} answered with something other than the study's server") from None
        if identity.get("served_name") != name:
            raise SystemExit(f"port {port} serves {identity.get('served_name')!r}, not {name!r}")
        return identity
    raise SystemExit(f"the server did not answer on port {port} within {seconds} s")


def check_pictures(examples=FT / "data" / "examples" / "train.jsonl"):
    """The adapter check's pictures: the rank-1 training pair of each of the first
    ADAPTER_CHECK_PICTURES type × tier cells, in data/examples/train.jsonl order. Training
    pictures, so no fine-tuned answer to a dev or test diagram exists before its scored run."""
    cells, pictures = set(), []
    with open(examples, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            cell = (row["primary_type"], row["tier"])
            if row["rank"] == 1 and cell not in cells:
                cells.add(cell)
                pictures.append(FT / row["picture"])
                if len(pictures) == ADAPTER_CHECK_PICTURES:
                    break
    return pictures


def adapter_check(model, revision, adapter, out, recurrent_layers=None):
    """infer/adapter_check.py for the adapter on this machine, before any answer of a scored run
    (methodology/compute-and-serving.md §8, rule 3). Refuses to go on unless it passes."""
    result = Path(out) / "adapter_check.json"
    command = [sys.executable, str(ADAPTER_CHECK), "--model", model, "--revision", revision,
               "--adapter", str(adapter), "--pictures", *map(str, check_pictures()), "--out", str(result)]
    if recurrent_layers:
        command += ["--recurrent-layers", recurrent_layers]
    with open(Path(out) / "adapter_check.log", "a", encoding="utf-8") as log:
        code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
    if code != 0 or not result.is_file():
        raise SystemExit(f"the adapter check failed for {adapter} (exit {code}; {Path(out) / 'adapter_check.log'}); "
                         "no answer is made")
    return {"file": result.name, "sha256": sha256_file(result), "passed": json.loads(result.read_text())["passed"]}


def write_manifest(answers, port, out):
    """infer/manifest.py write for the answers while their server still runs (--server), so the
    manifest records every check of methodology/compute-and-serving.md §5, `server_matches`
    included."""
    with open(Path(out) / "manifest.log", "a", encoding="utf-8") as log:
        code = subprocess.run([sys.executable, str(MANIFEST), "write", str(answers), "--server",
                               f"http://127.0.0.1:{port}"], stdout=log, stderr=subprocess.STDOUT,
                              env=dict(os.environ, NO_PROXY="127.0.0.1,localhost")).returncode
    written = Path(answers) / "manifest.json"
    return {"file": "answers/manifest.json", "passed": code == 0 and written.is_file(),
            "checks": len(json.loads(written.read_text())["checks"]) if written.is_file() else 0}


def answer(args):
    sys.path.insert(0, str(HERE))
    from models import SIZES
    model, revision = SIZES[args.size]
    out = Path(args.out)
    port_free(args.port)
    out.mkdir(parents=True, exist_ok=True)
    checked = adapter_check(model, revision, args.adapter, out, args.recurrent_layers) if args.adapter else None
    answers = out / "answers"
    answers.mkdir(parents=True, exist_ok=True)
    rows = json.load(open(args.set_file, encoding="utf-8"))["diagrams"]
    name = served_name(model, args.adapter)
    command = [sys.executable, str(SERVER), "--model", model, "--revision", revision,
               "--port", str(args.port)]
    if args.adapter:
        command += ["--adapter", str(args.adapter), "--served-name", name]
    if args.recurrent_layers:
        command += ["--recurrent-layers", args.recurrent_layers]
    env = dict(os.environ, NO_PROXY="127.0.0.1,localhost")
    started = now()
    with open(out / "server.log", "a", encoding="utf-8") as log:
        server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
    try:
        identity = wait_for(args.port, server, args.start_seconds, name)
        runner = [sys.executable, str(INSTRUMENT / "evaluation" / "infer_runner.py"),
                  "--base-url", f"http://127.0.0.1:{args.port}/v1", "--model", name,
                  "--test-set", str(args.set_file), "--images", str(args.images),
                  "--n", str(len(rows)), "--max-tokens", str(args.max_tokens),
                  "--timeout", str(args.timeout), "--run-dir", str(answers),
                  "--key-env", "LOCAL_KEY",
                  "--extra-body", json.dumps({"chat_template_kwargs": {"enable_thinking": False}})]
        with open(out / "runner.log", "a", encoding="utf-8") as log:
            code = subprocess.run(runner, stdout=log, stderr=subprocess.STDOUT,
                                  env=dict(env, LOCAL_KEY="local")).returncode
        run_manifest = write_manifest(answers, args.port, out)
    finally:
        server.terminate()
        try:
            server.wait(timeout=60)
        except subprocess.TimeoutExpired:
            server.kill()
    stored = sorted(p.stem for p in answers.glob("*.puml"))
    manifest = {"step": "answer", "started": started, "finished": now(), "size": args.size,
                "set_file": str(args.set_file), "set_file_sha256": sha256_file(args.set_file),
                "images": str(args.images), "diagrams": len(rows), "answers_stored": len(stored),
                "max_tokens": args.max_tokens, "timeout_seconds": args.timeout,
                "runner_exit_code": code, "server": identity, "adapter_check": checked,
                "run_manifest": run_manifest, "dev_scoring_sha256": sha256_file(__file__)}
    with open(out / "answer_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    print(f"{len(stored)} of {len(rows)} answers stored in {answers}", flush=True)
    return manifest


# --------------------------------------------------------------------------- #
# score
# --------------------------------------------------------------------------- #
def check_pins(render_jar=RENDER_JAR, extractor_jar=EXTRACTOR_JAR, java_home=JAVA_HOME,
               python=sys.executable):
    """The identity of the scorer used, or Unpinned if any part is not the frozen one."""
    commit = subprocess.run(["git", "-C", str(INSTRUMENT), "rev-parse", f"{SCORER_TAG}^{{commit}}"],
                            capture_output=True, text=True).stdout.strip()
    if commit != SCORER_COMMIT:
        raise Unpinned(f"tag {SCORER_TAG} is on {commit or 'nothing'}, pinned {SCORER_COMMIT}")
    same = subprocess.run(["git", "-C", str(INSTRUMENT), "diff", "--quiet", SCORER_TAG, "--",
                           "evaluation", "prompts"]).returncode
    if same != 0:
        raise Unpinned(f"evaluation/ or prompts/ of {INSTRUMENT} differ from tag {SCORER_TAG}")
    for path, pin in ((extractor_jar, EXTRACTOR_SHA256), (render_jar, RENDER_SHA256)):
        got = sha256_file(path) if Path(path).is_file() else None
        if got != pin:
            raise Unpinned(f"{path} has sha256 {got}, pinned {pin}")
    java = Path(java_home) / "bin" / "java"
    java_version = subprocess.run([str(java), "-version"], capture_output=True, text=True)
    dot = subprocess.run(["dot", "-V"], capture_output=True, text=True)
    py = subprocess.run([python, "-c", "import sys; print(sys.version.split()[0])"],
                        capture_output=True, text=True)
    if java_version.returncode != 0 or dot.returncode != 0 or py.returncode != 0:
        raise Unpinned("Java, Graphviz or the scorer's Python does not answer")
    return {"tag": SCORER_TAG, "commit": SCORER_COMMIT, "extractor_jar_sha256": EXTRACTOR_SHA256,
            "render_jar_sha256": RENDER_SHA256,
            "java": java_version.stderr.splitlines()[0], "graphviz": dot.stderr.strip(),
            "python": py.stdout.strip()}


def score(args):
    # The runners run from the benchmark's folder: a relative path would resolve there.
    for name in ("answers", "out", "set_file", "gt_dir", "render_jar"):
        setattr(args, name, str(Path(getattr(args, name)).resolve()))
    scorer = check_pins(args.render_jar, EXTRACTOR_JAR, args.java_home, args.python)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PATH=str(Path(args.java_home) / "bin") + os.pathsep + os.environ["PATH"])
    py = args.python
    subprocess.run([py, "evaluation/csr_runner.py", "--pred-dir", str(args.answers),
                    "--test-set", str(args.set_file), "--out", str(out / "csr"),
                    "--jar", str(args.render_jar)],
                   cwd=INSTRUMENT, env=env, check=True, stdout=subprocess.DEVNULL)
    subprocess.run([py, "evaluation/element_f1_runner.py", "--pred-dir", str(out / "csr" / "extracted"),
                    "--gt-dir", str(args.gt_dir), "--test-set", str(args.set_file),
                    "--csr", str(out / "csr" / "csr_results.json"), "--out", str(out / "element_f1"),
                    "--jar", str(EXTRACTOR_JAR)],
                   cwd=INSTRUMENT, env=env, check=True, stdout=subprocess.DEVNULL)
    csr = json.load(open(out / "csr" / "csr_results.json"))["summary"]
    ef1 = json.load(open(out / "element_f1" / "element_f1_results.json"))["summary"]
    manifest = Path(args.answers).parent / "answer_manifest.json"
    result = {"step": "score", "finished": now(), "answers": str(args.answers),
              "set_file": str(args.set_file), "set_file_sha256": sha256_file(args.set_file),
              "csr": {k: csr[k] for k in ("n", "compiled", "csr")},
              "element_f1": {pop: {"n": ef1[pop]["n"], "micro": ef1[pop]["micro"]["f1"],
                                   "macro": ef1[pop]["macro"]["f1"]}
                             for pop in ("zeros_for_failed", "compiled_only") if pop in ef1},
              "scorer": scorer, "dev_scoring_sha256": sha256_file(__file__),
              "answer_manifest": json.load(open(manifest)) if manifest.is_file() else None}
    with open(out / "scores.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    print(json.dumps({"csr": result["csr"], "element_f1": result["element_f1"]}), flush=True)
    return result


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="step", required=True)
    a = sub.add_parser("answer")
    a.add_argument("--size", required=True, choices=["2B", "9B", "27B"])
    a.add_argument("--adapter", default=None, help="checkpoint folder; omitted: the base model")
    a.add_argument("--out", required=True)
    a.add_argument("--timeout", type=int, required=True, help="per-answer deadline, seconds")
    a.add_argument("--max-tokens", type=int, default=OUTPUT_CAP)
    a.add_argument("--port", type=int, default=8000)
    a.add_argument("--start-seconds", type=int, default=900)
    a.add_argument("--set-file", default=str(DEV_SET))
    a.add_argument("--images", default=str(DEV_IMAGES))
    a.add_argument("--recurrent-layers", default=None,
                   help="passed to the server; omitted, the server's default (the study's choice) "
                        "applies. On the Mac only `reference` runs, for local checks")
    s = sub.add_parser("score")
    s.add_argument("--answers", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--set-file", default=str(DEV_SET))
    s.add_argument("--gt-dir", default=str(GT_DIR))
    s.add_argument("--render-jar", default=str(RENDER_JAR))
    s.add_argument("--java-home", default=JAVA_HOME)
    s.add_argument("--python", default="/usr/bin/python3",
                   help="the Python the scorer runs under (scorer v2 was measured under 3.9.6)")
    args = ap.parse_args(argv)
    return answer(args) if args.step == "answer" else score(args)


def cli(argv=None):
    """The command's exit status: 1 when `answer`'s run manifest does not pass (a failed runner
    included), else 0."""
    result = main(argv)
    if isinstance(result, dict) and result.get("step") == "answer" and not result["run_manifest"]["passed"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(cli())
