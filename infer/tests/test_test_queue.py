"""Tests for infer/test_queue.sh, infer/test_runs.txt and the parts of infer/test_session.sh that run
without a pod (files, check, follow), with a stand-in for the Python the queue runs and a local
folder in place of the pod.

Run from fine-tune/:
  .venv/bin/python infer/tests/test_test_queue.py
(or with pytest: .venv/bin/python -m pytest <this file>)
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

FT = Path(__file__).resolve().parents[2]
INSTRUMENT = FT.parent / "plantuml-recognition-evaluation"
SESSION = FT / "infer" / "test_session.sh"
MAIN = ["2B_zero_shot", "2B_20000", "9B_zero_shot", "9B_20000", "27B_zero_shot", "27B_20000"]
CURVE = ["2B_1000", "2B_5000"]
ORDER = ["2B_zero_shot", "2B_20000", "2B_1000", "2B_5000", "9B_zero_shot", "9B_20000",
         "27B_zero_shot", "27B_20000"]

# Stands in for the venv's python. download_models.py and dev_scoring.py are recorded with their
# arguments. FAIL="<id>:<n>" makes the first <n> answers of run <id> exit 1.
FAKE_PY = """#!/bin/bash
if [ "$1" = -c ]; then exec "$REAL_PY" "$@"; fi
echo "$(date +%s.%N) start $*" >> "$CALLS"
code=0
if [ "$1" = train/dev_scoring.py ]; then
    out=$(echo "$*" | sed 's/.*--out \\([^ ]*\\).*/\\1/'); id=${out##*/}
    n=$(cat "$out/.attempts" 2>/dev/null || echo 0); n=$((n + 1)); echo $n > "$out/.attempts"
    for f in $FAIL; do [ "${f%%:*}" = "$id" ] && [ $n -le "${f#*:}" ] && code=1; done
    sleep 0.3
fi
echo "$(date +%s.%N) end $*" >> "$CALLS"
exit $code
"""


def write_exe(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(0o755)


def git(co, *args):
    subprocess.run(["git", "-C", str(co), *args], check=True, capture_output=True)


def checkout(pictures=1000, test_set=INSTRUMENT / "data" / "test_set.json"):
    tmp = Path(tempfile.mkdtemp())
    co = tmp / "work" / "fine-tune"
    (co / "infer").mkdir(parents=True)
    for f in ("test_queue.sh", "test_runs.txt"):
        shutil.copy(FT / "infer" / f, co / "infer" / f)
    for line in (FT / "infer" / "test_runs.txt").read_text().splitlines():
        parts = line.split()
        if parts and not parts[0].startswith("#") and parts[2] != "-":
            for f in ("adapter_config.json", "adapter_model.safetensors"):
                (co / parts[2]).mkdir(parents=True, exist_ok=True)
                (co / parts[2] / f).write_text("{}")
    inst = tmp / "work" / "plantuml-recognition-evaluation" / "data"
    (inst / "puml_images_1568").mkdir(parents=True)
    shutil.copy(test_set, inst / "test_set.json")
    for i in range(pictures):
        (inst / "puml_images_1568" / f"{i:040x}.png").write_bytes(b"")
    git(co, "init", "-q")
    git(co, "add", "infer")
    git(co, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "c")
    write_exe(tmp / "python", FAKE_PY)
    return tmp, co


def run(tmp, co, *ids, fail="", log="test_queue.log"):
    env = dict(os.environ, PY=str(tmp / "python"), CALLS=str(tmp / "calls.txt"), FAIL=fail,
               REAL_PY=sys.executable)
    done = subprocess.run(["bash", "infer/test_queue.sh", *ids], cwd=co, env=env,
                          capture_output=True, text=True, timeout=120)
    calls = tmp / "calls.txt"
    lines = calls.read_text().splitlines() if calls.exists() else []
    qlog = tmp / "work/results/logs" / log
    return done, lines, qlog.read_text().splitlines() if qlog.exists() else []


def calls_of(lines, kind="start"):
    return [l.split()[2:] for l in lines if l.split()[1] == kind]


def answers(lines):
    return [c for c in calls_of(lines) if c[0] == "train/dev_scoring.py"]


def arg(call, name):
    return call[call.index(name) + 1] if name in call else None


def run_id(call):
    return arg(call, "--out").rsplit("/", 1)[1]


def most_at_once(lines, size):
    running, most = 0, 0
    for l in sorted(lines, key=lambda l: float(l.split()[0])):
        a = l.split()
        if a[2] == "train/dev_scoring.py" and arg(a, "--size") == size:
            running += 1 if a[1] == "start" else -1
            most = max(most, running)
    return most


# --------------------------------------------------------------------------- #
# the run list
# --------------------------------------------------------------------------- #
def runs_listed():
    rows = []
    for line in (FT / "infer" / "test_runs.txt").read_text().splitlines():
        parts = line.split()
        if parts and not parts[0].startswith("#"):
            rows.append(parts)
    return rows


def test_the_list_holds_the_six_main_runs_and_the_two_of_the_curve():
    rows = runs_listed()
    assert [r[0] for r in rows] == ORDER
    assert all(len(r) == 3 for r in rows)
    for size in ("2B", "9B", "27B"):
        assert [size, "-"] == next(r[1:] for r in rows if r[0] == f"{size}_zero_shot")
        assert [size, f"runs/train/{size}_20000/step_02500"] == next(r[1:] for r in rows if r[0] == f"{size}_20000")
    assert next(r[1:] for r in rows if r[0] == "2B_1000") == ["2B", "runs/train/2B_1000/step_00126"]
    assert next(r[1:] for r in rows if r[0] == "2B_5000") == ["2B", "runs/train/2B_5000/step_00626"]


def test_the_adapters_listed_are_the_selected_ones():
    # runs/ is kept out of the repository: on a machine without it there is nothing to compare.
    for rid, adapter in (("2B_20000", "step_02500"), ("9B_20000", "step_02500"),
                         ("27B_20000", "step_02500"), ("2B_5000", "step_00626")):
        sel = FT / "runs" / "train" / rid / "selection.json"
        if not sel.exists():
            continue
        folder = json.loads(sel.read_text())["selected_folder"]
        assert folder.endswith(f"runs/dev/{rid}/{adapter}/scores"), (rid, folder)


def test_the_benchmark_reads_every_stored_test_run_as_1000_cells():
    # The benchmark's run reader (analysis/run_level.py) on each stored run folder, the run
    # manifest beside the cells included. Where runs/test/ is absent there is nothing to read.
    stored = [rid for rid in ORDER if (FT / "runs" / "test" / rid / "answers").is_dir()]
    if not stored:
        return
    code = ("import json, sys; sys.path.insert(0, sys.argv[1]); "
            "from analysis.run_level import read_run_usage; "
            "print(json.dumps({r: read_run_usage(f'{sys.argv[2]}/{r}/answers')['n_cells'] "
            "for r in sys.argv[3:]}))")
    out = subprocess.run([sys.executable, "-I", "-c", code, str(INSTRUMENT),
                          str(FT / "runs" / "test"), *stored],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-2000:]
    assert json.loads(out.stdout) == {rid: 1000 for rid in stored}


# --------------------------------------------------------------------------- #
# the queue
# --------------------------------------------------------------------------- #
def test_every_run_is_answered_on_the_test_set_by_size_and_says_all_done():
    tmp, co = checkout()
    done, lines, log = run(tmp, co)
    assert done.returncode == 0 and log[-1].endswith("all done"), (log, done.stdout, done.stderr)
    order = [c[0] if c[0] != "train/dev_scoring.py" else run_id(c) for c in calls_of(lines)]
    assert order[0] == "train/download_models.py" and order[5] == "train/download_models.py"
    assert sorted(order[1:5]) == sorted(ORDER[:4])
    assert order[6:] == ["9B_zero_shot", "9B_20000", "train/download_models.py", "27B_zero_shot", "27B_20000"]
    work = tmp / "work"
    inst = work / "plantuml-recognition-evaluation" / "data"
    for c in answers(lines):
        rid = run_id(c)
        assert c[:2] == ["train/dev_scoring.py", "answer"]
        assert arg(c, "--size") == rid.split("_")[0]
        assert arg(c, "--out") == f"runs/test/{rid}"
        assert arg(c, "--timeout") == "600"
        assert Path(arg(c, "--set-file")).resolve() == (inst / "test_set.json").resolve()
        assert Path(arg(c, "--images")).resolve() == (inst / "puml_images_1568").resolve()
        adapter = arg(c, "--adapter")
        if rid.endswith("zero_shot"):
            assert adapter is None
        else:
            listed = next(r[2] for r in runs_listed() if r[0] == rid)
            assert adapter == f"{co.resolve()}/{listed}" or Path(adapter).resolve() == (co / listed).resolve()
        assert (work / "results" / "queue" / f"test_{rid}.done").exists()
        assert (work / "results" / "logs" / f"test_{rid}.log").read_text().splitlines()[-1] == "exit 0"
    assert sorted(run_id(c) for c in answers(lines)) == sorted(ORDER)
    for s in ("2B", "9B", "27B"):
        assert (work / "results" / "queue" / f"download_{s}.done").exists()


def test_the_2b_runs_go_side_by_side_on_their_own_ports_the_others_one_at_a_time():
    tmp, co = checkout()
    _, lines, _ = run(tmp, co)
    assert most_at_once(lines, "2B") == 4
    assert most_at_once(lines, "9B") == 1 and most_at_once(lines, "27B") == 1
    ports = {run_id(c): arg(c, "--port") for c in answers(lines)}
    assert sorted(ports[r] for r in ORDER[:4]) == ["8000", "8002", "8003", "8004"]
    assert {ports[r] for r in ORDER[4:]} == {"8000"}


def test_the_ids_limit_the_queue_and_an_unknown_id_stops_it():
    tmp, co = checkout()
    done, lines, log = run(tmp, co, "27B_20000", "9B_zero_shot")
    assert done.returncode == 0, log
    assert [run_id(c) for c in answers(lines)] == ["9B_zero_shot", "27B_20000"]
    tmp, co = checkout()
    done, lines, log = run(tmp, co, "27B_5000")
    assert done.returncode == 1 and "27B_5000 is not a run" in log[-1] and not lines


def test_a_run_that_fails_once_is_resumed_with_the_same_command():
    tmp, co = checkout()
    done, lines, log = run(tmp, co, "9B_20000", fail="9B_20000:1")
    assert done.returncode == 0 and log[-1].endswith("all done"), log
    tries = answers(lines)
    assert len(tries) == 2 and tries[0] == tries[1]
    text = (tmp / "work/results/logs/test_9B_20000.log").read_text()
    assert "attempt 1" in text and "attempt 2" in text and text.splitlines()[-1] == "exit 0"


def test_a_run_that_fails_twice_stops_the_queue_after_its_size():
    tmp, co = checkout()
    done, lines, log = run(tmp, co, fail="2B_1000:2")
    assert done.returncode == 1 and log[-1].endswith("stopped: a run of the 2B failed twice"), log
    marks = tmp / "work/results/queue"
    assert not (marks / "test_2B_1000.done").exists()
    for rid in ("2B_zero_shot", "2B_20000", "2B_5000"):
        assert (marks / f"test_{rid}.done").exists()
    assert {arg(c, "--size") for c in answers(lines)} == {"2B"}
    assert sum(run_id(c) == "2B_1000" for c in answers(lines)) == 2


def test_a_restart_skips_the_finished_runs_and_resumes_its_own_folder():
    tmp, co = checkout()
    run(tmp, co, fail="2B_1000:2")
    (tmp / "calls.txt").unlink()
    done, lines, log = run(tmp, co)
    assert done.returncode == 0 and log[-1].endswith("all done"), log
    assert [run_id(c) for c in answers(lines)][0] == "2B_1000"
    assert "2B_zero_shot" not in [run_id(c) for c in answers(lines)]


def test_a_folder_of_another_checkout_is_refused_before_anything_runs():
    tmp, co = checkout()
    (co / "runs/test/27B_20000").mkdir(parents=True)
    (co / "runs/test/27B_20000/queue_commit.txt").write_text("0" * 40 + "\n")
    done, lines, log = run(tmp, co)
    assert done.returncode == 1 and "runs/test/27B_20000 exists from another checkout" in log[-1]
    assert not lines
    tmp, co = checkout()
    (co / "runs/test/2B_5000").mkdir(parents=True)
    done, lines, log = run(tmp, co)
    assert done.returncode == 1 and "runs/test/2B_5000 exists" in log[-1] and not lines


def test_another_test_set_or_a_missing_picture_or_adapter_stops_the_queue():
    other = Path(tempfile.mkdtemp()) / "test_set.json"
    other.write_text(json.dumps({"diagrams": []}))
    tmp, co = checkout(test_set=other)
    done, lines, log = run(tmp, co)
    assert done.returncode == 1 and "not the frozen test set" in log[-1] and not lines
    tmp, co = checkout(pictures=999)
    done, lines, log = run(tmp, co)
    assert done.returncode == 1 and "holds 999 pictures, not 1000" in log[-1] and not lines
    tmp, co = checkout()
    (co / "runs/train/9B_20000/step_02500/adapter_model.safetensors").unlink()
    done, lines, log = run(tmp, co)
    assert done.returncode == 1 and "9B_20000: runs/train/9B_20000/step_02500/adapter_model.safetensors not found" in log[-1]
    assert not lines


# --------------------------------------------------------------------------- #
# the Mac's session
# --------------------------------------------------------------------------- #
def test_files_are_the_listed_adapters_the_check_pictures_and_the_examples():
    out = subprocess.run(["zsh", str(SESSION), "files"], capture_output=True, text=True, check=True).stdout.split()
    adapters = [r[2] for r in runs_listed() if r[2] != "-"]
    want = [f"{a}/{f}" for a in adapters for f in ("adapter_config.json", "adapter_model.safetensors")]
    assert out[:len(want)] == want
    pictures = out[len(want):-2]
    assert len(pictures) == 3 and all(p.startswith("data/sets/render/train/puml_images_1568/") for p in pictures)
    assert out[-2:] == ["data/examples/train.jsonl", "data/examples/loss_curve.jsonl"]


# Stands in for the Python that checks: `infer/manifest.py check <dir>` exits as BAD says (an id in
# BAD fails); `-c` goes to the real Python (it reads n_cells).
FAKE_CHECK = """#!/bin/bash
if [ "$1" = -c ]; then exec "$REAL_PY" "$@"; fi
echo "$*" >> "$CHECKS"
for b in $BAD; do case "$3" in *"/$b/answers") exit 1 ;; esac; done
exit 0
"""


def pod_folder(ids, n_cells=1000, last="all done"):
    pod = Path(tempfile.mkdtemp())
    (pod / "results/queue").mkdir(parents=True)
    (pod / "results/logs").mkdir(parents=True)
    for rid in ids:
        a = pod / "fine-tune/runs/test" / rid / "answers"
        a.mkdir(parents=True)
        (a / "run_meta.json").write_text(json.dumps({"n_cells": n_cells}))
        (pod / "results/queue" / f"test_{rid}.done").touch()
    (pod / "results/logs/test_queue.log").write_text(f"2026-10-26T00:00:00Z {last}\n")
    return pod


def follow(pod, local, ids, bad=""):
    tmp = Path(tempfile.mkdtemp())
    write_exe(tmp / "check", FAKE_CHECK)
    ft = tmp / "ft"
    (ft / "infer").mkdir(parents=True)
    (ft / "train").mkdir()
    shutil.copy(SESSION, ft / "infer" / "test_session.sh")
    shutil.copy(FT / "infer" / "test_runs.txt", ft / "infer" / "test_runs.txt")
    env = dict(os.environ, TEST_POD_ROOT=str(pod), TEST_LOCAL=str(local), CHECK_PY=str(tmp / "check"),
               REAL_PY=sys.executable, CHECKS=str(tmp / "checks.txt"), BAD=bad, TEST_IDS=" ".join(ids),
               FOLLOW_SECONDS="1")
    done = subprocess.run(["zsh", str(ft / "infer/test_session.sh"), "follow"], env=env,
                          capture_output=True, text=True, timeout=60)
    checks = (tmp / "checks.txt").read_text().splitlines() if (tmp / "checks.txt").exists() else []
    return done, ft, checks


def test_follow_fetches_and_checks_every_finished_run_once_then_ends_with_the_queue():
    pod = pod_folder(ORDER)
    local = Path(tempfile.mkdtemp())
    done, ft, checks = follow(pod, local, ORDER)
    assert done.returncode == 0, (done.stdout, done.stderr)
    assert len(checks) == 8 and all(c.startswith("infer/manifest.py check ") for c in checks)
    for rid in ORDER:
        assert (local / "checked" / f"{rid}.done").exists()
        assert (ft / "runs/test" / rid / "answers/run_meta.json").exists()
    assert (local / "follow.log").read_text().splitlines()[-1].endswith("follow ends: the queue is done")


def test_follow_ends_with_4_on_a_failed_manifest_or_a_run_not_of_1000_cells():
    pod = pod_folder(["2B_zero_shot"])
    done, _, _ = follow(pod, Path(tempfile.mkdtemp()), ["2B_zero_shot"], bad="2B_zero_shot")
    assert done.returncode == 4
    pod = pod_folder(["2B_zero_shot"], n_cells=25)
    local = Path(tempfile.mkdtemp())
    done, _, _ = follow(pod, local, ["2B_zero_shot"])
    assert done.returncode == 4 and "not a run of 1000 cells" in (local / "follow.log").read_text()


def test_follow_ends_with_1_when_the_queue_stopped():
    pod = pod_folder(["2B_zero_shot"], last="stopped: a run of the 2B failed twice")
    done, _, _ = follow(pod, Path(tempfile.mkdtemp()), ["2B_zero_shot"])
    assert done.returncode == 1


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok", t.__name__)
    print(f"{len(tests)} tests passed")
