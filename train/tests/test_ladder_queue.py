"""Tests for train/ladder_queue.sh and the parts of train/ladder_session.sh that run without a pod
(files, pack, follow), with stand-ins for the Python the queue and the scoring run and a local
folder in place of the pod.

Run from fine-tune/:
  .venv/bin/python train/tests/test_ladder_queue.py
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
QUEUE = FT / "train" / "ladder_queue.sh"
SESSION = FT / "train" / "ladder_session.sh"
STEPS = ["step_00625", "step_01250", "step_01875", "step_02500"]

# Stands in for the venv's python. `-c` (the queue reading the checkpoint steps) goes to the real
# Python; the study's scripts are recorded. FAIL="<script>:<word>" makes the call fail whose
# arguments contain <word>; a failing train.py leaves its run folder behind, as a crash would.
FAKE_PY = """#!/bin/bash
if [ "$1" = -c ]; then exec "$REAL_PY" "$@"; fi
echo "$(date +%s.%N) start $*" >> "$CALLS"
fail=0
case "$FAIL" in "$1":*) w=${FAIL#*:}; case " $* " in *"$w"*) fail=1 ;; esac ;; esac
if [ "$1" = train/train.py ]; then
    out=${@: -1}; mkdir -p "$out"
    [ $fail = 1 ] && exit 3
    for s in $STEPS; do mkdir -p "$out/$s"; done
fi
if [ "$1" = train/dev_scoring.py ]; then sleep 0.3; fi
echo "$(date +%s.%N) end $*" >> "$CALLS"
[ $fail = 1 ] && exit 4
exit 0
"""


def checkout():
    tmp = Path(tempfile.mkdtemp())
    co = tmp / "work" / "fine-tune"
    (co / "train").mkdir(parents=True)
    for f in ("ladder_queue.sh", "answer_checkpoints.sh", "recipe.py"):
        shutil.copy(FT / "train" / f, co / "train" / f)
    fake = tmp / "python"
    fake.write_text(FAKE_PY)
    fake.chmod(0o755)
    return tmp, co


def run(tmp, co, *sizes, fail=""):
    calls = tmp / "calls.txt"
    env = dict(os.environ, PY=str(tmp / "python"), CALLS=str(calls), FAIL=fail,
               REAL_PY=sys.executable, STEPS=" ".join(STEPS))
    done = subprocess.run(["bash", "train/ladder_queue.sh", *sizes], cwd=co, env=env,
                          capture_output=True, text=True, timeout=120)
    lines = calls.read_text().splitlines() if calls.exists() else []
    log = (tmp / "work/results/logs/queue.log")
    return done, lines, log.read_text().splitlines() if log.exists() else []


def calls_of(lines, kind="start"):
    return [l.split()[2:] for l in lines if l.split()[1] == kind]


def most_at_once(lines, size):
    running, most = 0, 0
    for l in sorted(lines, key=lambda l: float(l.split()[0])):
        a = l.split()
        if a[2] == "train/dev_scoring.py" and a[a.index("--size") + 1] == size:
            running += 1 if a[1] == "start" else -1
            most = max(most, running)
    return most


def test_the_ladder_runs_in_order_and_says_all_done():
    tmp, co = checkout()
    done, lines, log = run(tmp, co)
    assert done.returncode == 0 and log[-1].endswith("all done"), log
    order = [(c[0], c[c.index("--size") + 1] if "--size" in c else c[-1]) for c in calls_of(lines)]
    want = []
    for s in ("2B", "9B", "27B"):
        want += [("train/download_models.py", s), ("train/train.py", s)] + [("train/dev_scoring.py", s)] * 4
    assert order == want, order
    for c in calls_of(lines):
        if c[0] == "train/train.py":
            s = c[c.index("--size") + 1]
            assert c[:2] == ["train/train.py", "run"] and c[c.index("--pairs") + 1] == "20000"
            assert c[c.index("--out") + 1] == f"runs/train/{s}_20000"
    results = tmp / "work" / "results"
    for s in ("2B", "9B", "27B"):
        for name in ("download", "train", "answer"):
            assert (results / "queue" / f"{name}_{s}.done").exists()
        kept = sorted(p.name for p in (results / "logs" / f"answers_{s}").iterdir())
        assert kept == [f"answer_{c}.log" for c in STEPS]
    assert not list((results / "logs").glob("answer_step_*.log"))


def test_the_checkpoints_come_from_the_recipe_with_the_deadline():
    tmp, co = checkout()
    _, lines, _ = run(tmp, co, "2B")
    answers = [c for c in calls_of(lines) if c[0] == "train/dev_scoring.py"]
    adapters = sorted(c[c.index("--adapter") + 1].rsplit("/", 1)[1] for c in answers)
    assert adapters == STEPS
    for c in answers:
        assert "/runs/train/2B_20000/" in c[c.index("--adapter") + 1]
        assert c[c.index("--out") + 1].startswith("runs/dev/2B_20000/")
        assert c[c.index("--timeout") + 1] == "600"


def test_four_servers_at_the_2B_and_one_at_the_9B_and_27B():
    tmp, co = checkout()
    done, lines, _ = run(tmp, co)
    assert done.returncode == 0
    assert most_at_once(lines, "2B") == 4
    assert most_at_once(lines, "9B") == 1
    assert most_at_once(lines, "27B") == 1


def test_a_failed_training_stops_the_queue_and_is_never_trained_over():
    tmp, co = checkout()
    done, lines, log = run(tmp, co, fail="train/train.py:9B")
    assert done.returncode == 1 and "stopped: train 9B exited with 3" in log[-1], log
    assert not any(c[0] == "train/dev_scoring.py" and "9B" in c for c in calls_of(lines))
    assert not any("27B" in c for c in calls_of(lines))
    # started again: the 2B is skipped, and the 9B's folder left by the crash stops it
    done, lines2, log = run(tmp, co)
    new = calls_of(lines2)[len(calls_of(lines)):]
    assert done.returncode == 1 and "runs/train/9B_20000 exists without" in log[-1], log
    assert not any(c[0] == "train/train.py" for c in new), new
    assert any("download 2B: done before, skipped" in l for l in log)
    # the operator frees the folder: the queue goes on from the 9B
    shutil.rmtree(co / "runs/train/9B_20000")
    done, lines3, log = run(tmp, co)
    new = calls_of(lines3)[len(calls_of(lines2)):]
    assert done.returncode == 0 and log[-1].endswith("all done"), log
    trained = [c[c.index("--size") + 1] for c in new if c[0] == "train/train.py"]
    assert trained == ["9B", "27B"], trained


def test_a_failed_answer_stops_the_queue_and_names_the_checkpoint():
    tmp, co = checkout()
    done, lines, log = run(tmp, co, fail="train/dev_scoring.py:9B_20000/step_01250")
    assert done.returncode == 1 and "stopped: answer 9B" in log[-1], log
    out = (tmp / "work/results/logs/answer_9B.log").read_text()
    assert "checkpoint step_01250: exit 4" in out, out
    assert not (tmp / "work/results/queue/answer_9B.done").exists()
    assert (tmp / "work/results/logs/answers_9B/answer_step_01250.log").exists()
    assert not any("27B" in c for c in calls_of(lines))


def test_an_existing_dev_folder_without_its_mark_stops_before_answering():
    tmp, co = checkout()
    (co / "runs/dev/2B_20000/step_00625").mkdir(parents=True)
    done, lines, log = run(tmp, co, "2B")
    assert done.returncode == 1 and "runs/dev/2B_20000 exists without" in log[-1], log
    assert not any(c[0] == "train/dev_scoring.py" for c in calls_of(lines))


def test_an_unknown_size_starts_nothing():
    tmp, co = checkout()
    done, lines, log = run(tmp, co, "2B", "4B")
    assert done.returncode == 1 and "unknown size 4B" in log[-1] and not lines, log


def test_files_lists_every_picture_of_the_training_and_dev_lists():
    done = subprocess.run(["zsh", str(SESSION), "files"], capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    files = done.stdout.split()
    assert len(files) == len(set(files)) == 20202
    train = json.load(open(FT / "data/sets/train.json"))["diagrams"]
    dev = json.load(open(FT / "data/sets/dev.json"))["diagrams"]
    want = {f"data/sets/render/train/puml_images_1568/{r['key'][:-5]}.png" for r in train}
    want |= {f"data/sets/render/dev/puml_images_1568/{r['key'][:-5]}.png" for r in dev}
    want |= {"data/examples/train.jsonl", "data/examples/loss_curve.jsonl"}
    assert set(files) == want


def test_pack_refuses_uncommitted_code():
    tmp = Path(tempfile.mkdtemp())
    repo = tmp / "fine-tune"
    (repo / "train").mkdir(parents=True)
    shutil.copy(SESSION, repo / "train" / "ladder_session.sh")
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-C", str(repo)]
    subprocess.run(git[:5] + ["init", "-q", str(repo)], check=True)
    subprocess.run(git + ["add", "."], check=True)
    subprocess.run(git + ["commit", "-qm", "t"], check=True)
    (repo / "train" / "new.py").write_text("x = 1\n")
    done = subprocess.run(["zsh", str(repo / "train" / "ladder_session.sh"), "pack"],
                          env=dict(os.environ, LADDER_LOCAL=str(tmp / "local")),
                          capture_output=True, text=True, timeout=60)
    assert done.returncode == 1 and "commit these first" in done.stdout and "new.py" in done.stdout
    assert not (tmp / "local").exists()



# ---------------------------------------------------------------------------------------------- #
# ladder_session.sh follow: fetch, score each finished size once, end with the queue
# ---------------------------------------------------------------------------------------------- #
FAKE_SCORE_PY = """#!/bin/bash
echo "$*" >> "$CALLS"
if [ -n "$FAIL" ]; then case " $* " in *"$FAIL"*) exit 9 ;; esac; fi
if [ "$1" = train/select_checkpoint.py ]; then
    while [ $# -gt 0 ]; do [ "$1" = --out ] && { mkdir -p "$(dirname "$2")"; echo '{}' > "$2"; }; shift; done
fi
exit 0
"""
RSYNC_SHIM = """#!/bin/bash
echo "$*" >> "$RSYNC_CALLS"
exec /usr/bin/rsync "$@"
"""


def session_copy():
    """A fine-tune/ with the driver and the recipe only, a stand-in pod folder, a scoring stand-in."""
    tmp = Path(tempfile.mkdtemp())
    ft = tmp / "uni" / "fine-tune"
    (ft / "train").mkdir(parents=True)
    for f in ("ladder_session.sh", "recipe.py"):
        shutil.copy(FT / "train" / f, ft / "train" / f)
    pod = tmp / "pod"
    (pod / "results" / "logs").mkdir(parents=True)
    (pod / "results" / "queue").mkdir()
    (pod / "fine-tune" / "runs").mkdir(parents=True)
    for name, text in (("score_py", FAKE_SCORE_PY), ("bin/rsync", RSYNC_SHIM)):
        (tmp / name).parent.mkdir(exist_ok=True)
        (tmp / name).write_text(text)
        (tmp / name).chmod(0o755)
    return tmp, ft, pod


def pod_state(pod, done_sizes, last_line):
    for s in done_sizes:
        for c in STEPS:
            answers = pod / "fine-tune" / "runs" / "dev" / f"{s}_20000" / c / "answers"
            answers.mkdir(parents=True, exist_ok=True)
            (answers / "a.puml").write_text("@startuml\n@enduml\n")
        (pod / "results" / "queue" / f"answer_{s}.done").touch()
    (pod / "results" / "logs" / "queue.log").write_text(f"2026-10-24T00:00:00Z {last_line}\n")


def follow(tmp, ft, pod, fail="", **env):
    calls = tmp / "score_calls.txt"
    e = dict(os.environ, LADDER_POD_ROOT=str(pod), LADDER_LOCAL=str(tmp / "local"),
             SCORE_PY=str(tmp / "score_py"), CALLS=str(calls), FAIL=fail, FOLLOW_SECONDS="0",
             RSYNC_CALLS=str(tmp / "rsync_calls.txt"), PATH=f"{tmp / 'bin'}:{os.environ['PATH']}")
    e.update(env)
    done = subprocess.run(["zsh", str(ft / "train" / "ladder_session.sh"), "follow"], env=e,
                          capture_output=True, text=True, timeout=120)
    lines = calls.read_text().splitlines() if calls.exists() else []
    return done, lines


def scored_sizes(lines):
    return [l.split("--out runs/train/")[1].split("_")[0] for l in lines if l.startswith("train/select_checkpoint.py")]


def test_follow_scores_every_size_in_order_and_ends_with_the_queue():
    tmp, ft, pod = session_copy()
    pod_state(pod, ["2B", "9B", "27B"], "all done")
    done, lines = follow(tmp, ft, pod)
    assert done.returncode == 0, done.stdout + done.stderr
    assert scored_sizes(lines) == ["2B", "9B", "27B"]
    for s in ("2B", "9B", "27B"):
        scores = [l for l in lines if l.startswith("train/dev_scoring.py") and f"/{s}_20000/" in l]
        assert [l.split("--answers ")[1].split()[0] for l in scores] == [f"runs/dev/{s}_20000/{c}/answers" for c in STEPS]
        select = next(l for l in lines if l.startswith("train/select_checkpoint.py") and f"{s}_20000" in l)
        assert "--pairs 20000" in select
        assert select.split()[-4:] == [f"runs/dev/{s}_20000/{c}/scores" for c in STEPS]
        assert (tmp / "local" / "scored" / f"{s}.done").exists()
    # the fetch copied the runs into this fine-tune/ and the queue's files beside the session
    assert (ft / "runs/dev/27B_20000/step_02500/answers/a.puml").exists()
    assert (tmp / "local/results/queue/answer_27B.done").exists()


def test_fetch_copies_the_queue_marks_before_the_runs():
    tmp, ft, pod = session_copy()
    pod_state(pod, ["2B"], "all done")
    follow(tmp, ft, pod)
    # the client calls only: a local rsync starts its own `--server` side through PATH too
    rsync = [l for l in (tmp / "rsync_calls.txt").read_text().splitlines() if not l.startswith("--server")]
    assert "/results/" in rsync[0] and "/fine-tune/runs/" in rsync[1], rsync


def test_follow_ends_when_the_queue_stops_after_scoring_what_is_finished():
    tmp, ft, pod = session_copy()
    pod_state(pod, ["2B"], "stopped: train 9B exited with 3 (results/logs/train_9B.log)")
    done, lines = follow(tmp, ft, pod)
    assert done.returncode == 1 and "the queue stopped: train 9B" in done.stdout, done.stdout
    assert scored_sizes(lines) == ["2B"]
    # started again once the queue has gone on: the 2B is not scored a second time
    pod_state(pod, ["9B", "27B"], "all done")
    done, lines2 = follow(tmp, ft, pod)
    assert done.returncode == 0 and scored_sizes(lines2[len(lines):]) == ["9B", "27B"]


def test_follow_ends_when_the_pod_cannot_be_reached():
    tmp, ft, pod = session_copy()
    done, lines = follow(tmp, ft, tmp / "no_pod", FOLLOW_MAX_FAILS="2")
    assert done.returncode == 3 and "could not be reached" in done.stdout and not lines, done.stdout
    assert done.stdout.count("fetch failed") == 2


def test_follow_ends_when_scoring_fails_and_does_not_mark_the_size():
    tmp, ft, pod = session_copy()
    pod_state(pod, ["2B", "9B"], "train 27B: start")
    done, lines = follow(tmp, ft, pod, fail="9B_20000/step_01250")
    assert done.returncode == 4 and "scoring or selection of 9B failed" in done.stdout, done.stdout
    assert (tmp / "local/scored/2B.done").exists() and not (tmp / "local/scored/9B.done").exists()


def test_follow_ends_at_the_stop_time_while_the_queue_runs():
    tmp, ft, pod = session_copy()
    pod_state(pod, [], "train 2B: start")
    done, lines = follow(tmp, ft, pod, FOLLOW_STOP_AT="1")
    assert done.returncode == 5 and "stop time" in done.stdout and not lines, done.stdout

if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print("ok  ", name, flush=True)
        except BaseException as exc:  # noqa: BLE001
            failed += 1
            print("FAIL", name, f"{type(exc).__name__}: {exc}", flush=True)
    print(f"{len(tests) - failed} of {len(tests)} passed")
    sys.exit(1 if failed else 0)
