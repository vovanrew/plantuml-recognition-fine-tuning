"""Tests for train/ladder_queue.sh, train/ladder_chain.sh and the parts of train/ladder_session.sh
that run without a pod (files, pack, follow, chain, chain-mac), with stand-ins for the Python the
queue and the scoring run and a local folder in place of the pod.

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
import time
from pathlib import Path

FT = Path(__file__).resolve().parents[2]
SESSION = FT / "train" / "ladder_session.sh"
STEPS = ["step_00625", "step_01250", "step_01875", "step_02500"]
STEPS_5K = ["step_00157", "step_00313", "step_00470", "step_00626"]
LADDER = ("2B_20000", "9B_20000", "27B_20000")

# Stands in for the venv's python. `-c` (reading the checkpoint steps) goes to the real Python;
# the study's scripts are recorded. train.py makes the run's checkpoint folders, read from the
# recipe for its --pairs. FAIL="<script>:<word>" makes the call fail whose arguments contain
# <word>; a failing train.py leaves its run folder behind, as a crash would.
FAKE_PY = """#!/bin/bash
if [ "$1" = -c ]; then exec "$REAL_PY" "$@"; fi
echo "$(date +%s.%N) start $*" >> "$CALLS"
fail=0
case "$FAIL" in "$1":*) w=${FAIL#*:}; case " $* " in *"$w"*) fail=1 ;; esac ;; esac
if [ "$1" = train/train.py ]; then
    out=${@: -1}; mkdir -p "$out"
    [ $fail = 1 ] && exit 3
    pairs=$(echo "$*" | sed 's/.*--pairs \\([0-9]*\\).*/\\1/')
    for s in $("$REAL_PY" -c "import sys; sys.path.insert(0, 'train'); import recipe as R; print(' '.join(R.checkpoint_dir(s) for s in R.checkpoint_steps($pairs)))"); do
        mkdir -p "$out/$s"
    done
fi
if [ "$1" = train/dev_scoring.py ]; then sleep 0.3; fi
echo "$(date +%s.%N) end $*" >> "$CALLS"
[ $fail = 1 ] && exit 4
exit 0
"""


def write_exe(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(0o755)


def checkout():
    tmp = Path(tempfile.mkdtemp())
    co = tmp / "work" / "fine-tune"
    (co / "train").mkdir(parents=True)
    for f in ("ladder_queue.sh", "answer_checkpoints.sh", "recipe.py"):
        shutil.copy(FT / "train" / f, co / "train" / f)
    write_exe(tmp / "python", FAKE_PY)
    return tmp, co


def queue_env(tmp, fail="", **extra):
    return dict(os.environ, PY=str(tmp / "python"), CALLS=str(tmp / "calls.txt"), FAIL=fail,
                REAL_PY=sys.executable, **extra)


def run(tmp, co, *runs, fail="", log="queue.log", **extra):
    env = queue_env(tmp, fail, **extra)
    done = subprocess.run(["bash", "train/ladder_queue.sh", *runs], cwd=co, env=env,
                          capture_output=True, text=True, timeout=120)
    calls = tmp / "calls.txt"
    lines = calls.read_text().splitlines() if calls.exists() else []
    qlog = tmp / "work/results/logs" / log
    return done, lines, qlog.read_text().splitlines() if qlog.exists() else []


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
    for rid in LADDER:
        for name in ("download", "train", "answer"):
            assert (results / "queue" / f"{name}_{rid}.done").exists()
        kept = sorted(p.name for p in (results / "logs" / f"answers_{rid}").iterdir())
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


def test_a_5000_pair_run_has_its_own_folders_marks_and_log():
    tmp, co = checkout()
    run(tmp, co, "2B")                            # the 2B's 20,000-pair run is marked done
    done, lines, log = run(tmp, co, "2B:5000", log="queue_2B_5000.log", QUEUE_LOG="queue_2B_5000.log")
    assert done.returncode == 0 and log[-1].endswith("all done"), log
    new = calls_of(lines)[6:]
    train = [c for c in new if c[0] == "train/train.py"]
    assert len(train) == 1 and train[0][train[0].index("--pairs") + 1] == "5000"
    assert train[0][train[0].index("--out") + 1] == "runs/train/2B_5000"
    answers = [c for c in new if c[0] == "train/dev_scoring.py"]
    assert sorted(c[c.index("--adapter") + 1].rsplit("/", 1)[1] for c in answers) == STEPS_5K
    assert all(c[c.index("--out") + 1].startswith("runs/dev/2B_5000/") for c in answers)
    assert most_at_once(lines[len(lines) - 2 * len(new):], "2B") == 4
    results = tmp / "work/results"
    assert (results / "queue/answer_2B_5000.done").exists()
    assert sorted(p.name for p in (results / "logs/answers_2B_5000").iterdir()) == [f"answer_{c}.log" for c in STEPS_5K]
    # the ladder's log is not written to
    assert not any("2B_5000" in l for l in (results / "logs/queue.log").read_text().splitlines())


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
    assert done.returncode == 1 and "stopped: train 9B_20000 exited with 3" in log[-1], log
    assert not any(c[0] == "train/dev_scoring.py" and "9B" in c for c in calls_of(lines))
    assert not any("27B" in c for c in calls_of(lines))
    done, lines2, log = run(tmp, co)
    new = calls_of(lines2)[len(calls_of(lines)):]
    assert done.returncode == 1 and "runs/train/9B_20000 exists without" in log[-1], log
    assert not any(c[0] == "train/train.py" for c in new), new
    assert any("download 2B_20000: done before, skipped" in l for l in log)
    shutil.rmtree(co / "runs/train/9B_20000")
    done, lines3, log = run(tmp, co)
    new = calls_of(lines3)[len(calls_of(lines2)):]
    assert done.returncode == 0 and log[-1].endswith("all done"), log
    trained = [c[c.index("--size") + 1] for c in new if c[0] == "train/train.py"]
    assert trained == ["9B", "27B"], trained


def test_a_failed_answer_stops_the_queue_and_names_the_checkpoint():
    tmp, co = checkout()
    done, lines, log = run(tmp, co, fail="train/dev_scoring.py:9B_20000/step_01250")
    assert done.returncode == 1 and "stopped: answer 9B_20000" in log[-1], log
    out = (tmp / "work/results/logs/answer_9B_20000.log").read_text()
    assert "checkpoint step_01250: exit 4" in out, out
    assert not (tmp / "work/results/queue/answer_9B_20000.done").exists()
    assert (tmp / "work/results/logs/answers_9B_20000/answer_step_01250.log").exists()
    assert not any("27B" in c for c in calls_of(lines))


def test_an_existing_dev_folder_without_its_mark_stops_before_answering():
    tmp, co = checkout()
    (co / "runs/dev/2B_20000/step_00625").mkdir(parents=True)
    done, lines, log = run(tmp, co, "2B")
    assert done.returncode == 1 and "runs/dev/2B_20000 exists without" in log[-1], log
    assert not any(c[0] == "train/dev_scoring.py" for c in calls_of(lines))


def test_an_unknown_size_or_run_length_starts_nothing():
    for runs, words in ((("2B", "4B"), "unknown size 4B"), (("2B:3000",), "no run of 3000 pairs")):
        tmp, co = checkout()
        done, lines, log = run(tmp, co, *runs)
        assert done.returncode == 1 and words in log[-1] and not lines, log


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


def git(repo, *args):
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "-C", str(repo), *args],
                   check=True, capture_output=True, text=True)


def test_pack_refuses_uncommitted_code():
    tmp = Path(tempfile.mkdtemp())
    repo = tmp / "fine-tune"
    (repo / "train").mkdir(parents=True)
    shutil.copy(SESSION, repo / "train" / "ladder_session.sh")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "t")
    (repo / "train" / "new.py").write_text("x = 1\n")
    for step in ("pack", "chain"):
        done = subprocess.run(["zsh", str(repo / "train" / "ladder_session.sh"), step, *(["2B:5000"] if step == "chain" else [])],
                              env=dict(os.environ, LADDER_LOCAL=str(tmp / "local"), LADDER_POD_ROOT=str(tmp / "pod")),
                              capture_output=True, text=True, timeout=60)
        assert done.returncode == 1 and "commit these first" in done.stdout and "new.py" in done.stdout, done.stdout
    assert not (tmp / "local" / "pack").exists() and not (tmp / "pod").exists()


# ---------------------------------------------------------------------------------------------- #
# ladder_session.sh follow: fetch, score each finished run once, end with the queue
# ---------------------------------------------------------------------------------------------- #
FAKE_SCORE_PY = """#!/bin/bash
echo "$*" >> "$SCORE_CALLS"
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
    write_exe(tmp / "score_py", FAKE_SCORE_PY)
    write_exe(tmp / "bin" / "rsync", RSYNC_SHIM)
    return tmp, ft, pod


def pod_state(pod, ids, last_line, log="queue.log", legacy=False):
    for rid in ids:
        steps = STEPS_5K if rid.endswith("_5000") else STEPS
        for c in steps:
            answers = pod / "fine-tune" / "runs" / "dev" / rid / c / "answers"
            answers.mkdir(parents=True, exist_ok=True)
            (answers / "a.puml").write_text("@startuml\n@enduml\n")
        mark = rid.split("_")[0] if legacy else rid
        (pod / "results" / "queue" / f"answer_{mark}.done").touch()
    (pod / "results" / "logs" / log).write_text(f"2026-10-24T00:00:00Z {last_line}\n")


def session_env(tmp, pod, fail="", **env):
    e = dict(os.environ, LADDER_POD_ROOT=str(pod), LADDER_LOCAL=str(tmp / "local"),
             SCORE_PY=str(tmp / "score_py"), SCORE_CALLS=str(tmp / "score_calls.txt"), FAIL=fail,
             FOLLOW_SECONDS="0", RSYNC_CALLS=str(tmp / "rsync_calls.txt"),
             PATH=f"{tmp / 'bin'}:{os.environ['PATH']}")
    e.update(env)
    return e


def follow(tmp, ft, pod, fail="", **env):
    done = subprocess.run(["zsh", str(ft / "train" / "ladder_session.sh"), "follow"],
                          env=session_env(tmp, pod, fail, **env), capture_output=True, text=True, timeout=120)
    calls = tmp / "score_calls.txt"
    return done, calls.read_text().splitlines() if calls.exists() else []


def scored_runs(lines):
    return [l.split("--out runs/train/")[1].split("/")[0] for l in lines if l.startswith("train/select_checkpoint.py")]


def test_follow_scores_every_run_in_order_and_ends_with_the_queue():
    tmp, ft, pod = session_copy()
    pod_state(pod, LADDER, "all done")
    done, lines = follow(tmp, ft, pod)
    assert done.returncode == 0, done.stdout + done.stderr
    assert scored_runs(lines) == list(LADDER)
    for rid in LADDER:
        scores = [l for l in lines if l.startswith("train/dev_scoring.py") and f"/{rid}/" in l]
        assert [l.split("--answers ")[1].split()[0] for l in scores] == [f"runs/dev/{rid}/{c}/answers" for c in STEPS]
        select = next(l for l in lines if l.startswith("train/select_checkpoint.py") and rid in l)
        assert "--pairs 20000" in select
        assert select.split()[-4:] == [f"runs/dev/{rid}/{c}/scores" for c in STEPS]
        assert (tmp / "local" / "scored" / f"{rid}.done").exists()
    assert (ft / "runs/dev/27B_20000/step_02500/answers/a.puml").exists()
    assert (tmp / "local/results/queue/answer_27B_20000.done").exists()


def test_follow_reads_the_first_sessions_marks_by_size():
    tmp, ft, pod = session_copy()
    pod_state(pod, ["2B_20000"], "stopped: train 9B exited with 3", legacy=True)
    done, lines = follow(tmp, ft, pod)
    assert done.returncode == 1 and scored_runs(lines) == ["2B_20000"], done.stdout
    (tmp / "local/scored/2B_20000.done").unlink()
    (tmp / "local/scored/2B.done").touch()            # scored by the first session's follow
    done, lines2 = follow(tmp, ft, pod)
    assert scored_runs(lines2[len(lines):]) == []


def test_follow_scores_a_5000_pair_run_by_its_own_log():
    tmp, ft, pod = session_copy()
    pod_state(pod, ["2B_5000"], "all done", log="queue_2B_5000.log")
    (pod / "results/logs/queue.log").write_text("2026-10-24T00:00:00Z train 27B_20000: start\n")
    done, lines = follow(tmp, ft, pod, LADDER_RUNS="2B:5000", QUEUE_LOG="queue_2B_5000.log")
    assert done.returncode == 0, done.stdout
    assert scored_runs(lines) == ["2B_5000"]
    select = next(l for l in lines if l.startswith("train/select_checkpoint.py"))
    assert "--pairs 5000" in select and select.split()[-4:] == [f"runs/dev/2B_5000/{c}/scores" for c in STEPS_5K]


def test_fetch_copies_the_queue_marks_before_the_runs():
    tmp, ft, pod = session_copy()
    pod_state(pod, ["2B_20000"], "all done")
    follow(tmp, ft, pod)
    rsync = [l for l in (tmp / "rsync_calls.txt").read_text().splitlines() if not l.startswith("--server")]
    assert "/results/" in rsync[0] and "/fine-tune/runs/" in rsync[1], rsync


def test_follow_ends_when_the_queue_stops_after_scoring_what_is_finished():
    tmp, ft, pod = session_copy()
    pod_state(pod, ["2B_20000"], "stopped: train 9B_20000 exited with 3 (results/logs/train_9B_20000.log)")
    done, lines = follow(tmp, ft, pod)
    assert done.returncode == 1 and "the queue stopped: train 9B_20000" in done.stdout, done.stdout
    assert scored_runs(lines) == ["2B_20000"]
    pod_state(pod, ["9B_20000", "27B_20000"], "all done")
    done, lines2 = follow(tmp, ft, pod)
    assert done.returncode == 0 and scored_runs(lines2[len(lines):]) == ["9B_20000", "27B_20000"]


def test_follow_ends_when_the_pod_cannot_be_reached():
    tmp, ft, pod = session_copy()
    done, lines = follow(tmp, ft, tmp / "no_pod", FOLLOW_MAX_FAILS="2")
    assert done.returncode == 3 and "could not be reached" in done.stdout and not lines, done.stdout
    assert done.stdout.count("fetch failed") == 2


def test_follow_ends_when_scoring_fails_and_does_not_mark_the_run():
    tmp, ft, pod = session_copy()
    pod_state(pod, ["2B_20000", "9B_20000"], "train 27B_20000: start")
    done, lines = follow(tmp, ft, pod, fail="9B_20000/step_01250")
    assert done.returncode == 4 and "scoring or selection of 9B_20000 failed" in done.stdout, done.stdout
    assert (tmp / "local/scored/2B_20000.done").exists() and not (tmp / "local/scored/9B_20000.done").exists()


def test_follow_ends_at_the_stop_time_while_the_queue_runs():
    tmp, ft, pod = session_copy()
    pod_state(pod, [], "train 2B_20000: start")
    done, lines = follow(tmp, ft, pod, FOLLOW_STOP_AT="1")
    assert done.returncode == 5 and "stop time" in done.stdout and not lines, done.stdout


# ---------------------------------------------------------------------------------------------- #
# chain: the next runs start once the queue before has ended, on the newer commit
# ---------------------------------------------------------------------------------------------- #
def wait_for(cond, seconds=60):
    end = time.time() + seconds
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.2)
    return False


def chain_setup():
    """This Mac's repository (two commits, the tag on the first) and a pod checked out at the first."""
    tmp = Path(tempfile.mkdtemp())
    mac = tmp / "uni" / "fine-tune"
    for f in ("ladder_session.sh", "ladder_chain.sh", "ladder_queue.sh", "answer_checkpoints.sh",
              "study_bundle.sh", "recipe.py"):
        (mac / "train").mkdir(parents=True, exist_ok=True)
        shutil.copy(FT / "train" / f, mac / "train" / f)
    (mac / "infer").mkdir()
    (mac / "infer" / "manifest.py").write_text('FREEZE_TAG = "serving-v3"\n')
    (mac / ".gitignore").write_text("runs/\n.local/\n")
    subprocess.run(["git", "init", "-q", str(mac)], check=True)
    git(mac, "add", ".")
    git(mac, "commit", "-qm", "first")
    git(mac, "tag", "serving-v3")
    pod = tmp / "pod"
    (pod / "results" / "logs").mkdir(parents=True)
    subprocess.run(["git", "clone", "-q", str(mac), str(pod / "fine-tune")], check=True)
    (mac / "train" / "later.txt").write_text("the newer commit\n")
    git(mac, "add", ".")
    git(mac, "commit", "-qm", "second")
    second = subprocess.run(["git", "-C", str(mac), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    write_exe(tmp / "python", FAKE_PY)
    write_exe(tmp / "score_py", FAKE_SCORE_PY)
    write_exe(tmp / "bin" / "rsync", RSYNC_SHIM)
    write_exe(tmp / "backup.sh", '#!/bin/bash\necho "LADDER_RUNS=$LADDER_RUNS" >> "$BACKUP_CALLS"\n')
    return tmp, mac, pod, second


def test_chain_waits_for_the_queue_then_moves_the_checkout_and_starts_the_runs():
    tmp, mac, pod, second = chain_setup()
    logs = pod / "results" / "logs"
    (logs / "queue.log").write_text("2026-10-25T00:00:00Z answer 27B_20000: start\n")
    local = tmp / "uni" / "fine-tune" / ".local" / "ladder_session"
    local.mkdir(parents=True)
    # stands in for the running follow: detached, as follow is, so that it is reaped when it ends
    sleeper = int(subprocess.run(["bash", "-c", "nohup sleep 3 > /dev/null 2>&1 & echo $!"],
                                 capture_output=True, text=True).stdout)
    (local / "follow.pid").write_text(f"{sleeper}\n")
    (local / "follow.log").write_text("2026-10-25T00:00:00Z follow ends: the queue is done\n")
    env = queue_env(tmp)
    env.update(session_env(tmp, pod))
    env.update(LADDER_LOCAL=str(local), CHAIN_SECONDS="1", LADDER_BACKUP=str(tmp / "backup.sh"),
               BACKUP_CALLS=str(tmp / "backup_calls.txt"), FOLLOW_SECONDS="1")
    done = subprocess.run(["zsh", str(mac / "train" / "ladder_session.sh"), "chain", "2B:5000"], env=env,
                          capture_output=True, text=True, timeout=60)
    assert done.returncode == 0 and "chained 2B_5000" in done.stdout, done.stdout + done.stderr
    head = lambda: subprocess.run(["git", "-C", str(pod / "fine-tune"), "rev-parse", "HEAD"],
                                  capture_output=True, text=True).stdout.strip()
    time.sleep(2)
    assert head() != second and not (logs / "queue_2B_5000.log").exists()   # the earlier queue still runs
    with open(logs / "queue.log", "a") as f:
        f.write("2026-10-25T01:00:00Z all done\n")
    assert wait_for(lambda: (logs / "queue_2B_5000.log").exists()
                    and (logs / "queue_2B_5000.log").read_text().rstrip().endswith("all done")), \
        (logs / f"chain_2B_5000.log").read_text()
    assert head() == second
    chain_log = (logs / "chain_2B_5000.log").read_text()
    assert f"checkout {second}" in chain_log and "queue started" in chain_log
    trained = [c for c in calls_of((tmp / "calls.txt").read_text().splitlines()) if c[0] == "train/train.py"]
    assert len(trained) == 1 and "--pairs" in trained[0] and trained[0][trained[0].index("--pairs") + 1] == "5000"
    # this Mac: once the running follow has ended with the queue done, a follow for 2B_5000 and its backup
    assert wait_for(lambda: (tmp / "backup_calls.txt").exists())
    assert (tmp / "backup_calls.txt").read_text().strip() == "LADDER_RUNS=2B:5000"
    assert wait_for(lambda: "scored and selected 2B_5000" in (local / "follow.log").read_text())
    assert wait_for(lambda: (local / "follow.log").read_text().rstrip().endswith("the queue is done"))
    assert int((local / "follow.pid").read_text()) != sleeper
    mac_log = (local / "chain_2B_5000.log").read_text()
    assert "follow" in mac_log and "started for 2B:5000" in mac_log, mac_log


def test_chain_starts_nothing_when_the_queue_before_stopped():
    tmp, mac, pod, second = chain_setup()
    logs = pod / "results" / "logs"
    (logs / "queue.log").write_text("2026-10-25T00:00:00Z stopped: answer 27B_20000 exited with 1\n")
    done = subprocess.run(["bash", str(mac / "train" / "ladder_chain.sh"), str(pod), "2B_5000", "queue.log", "2B:5000"],
                          env=dict(queue_env(tmp), CHAIN_SECONDS="1"), capture_output=True, text=True, timeout=60)
    assert done.returncode == 1
    assert "stopped; nothing started" in (logs / "chain_2B_5000.log").read_text()
    assert not (logs / "queue_2B_5000.log").exists()


def test_chain_mac_starts_nothing_when_follow_ended_otherwise():
    tmp, mac, pod, second = chain_setup()
    local = tmp / "local"
    local.mkdir()
    (local / "follow.pid").write_text("999999\n")
    (local / "follow.log").write_text("2026-10-25T00:00:00Z follow ends: the queue stopped: answer 27B_20000 exited with 1\n")
    done = subprocess.run(["zsh", str(mac / "train" / "ladder_session.sh"), "chain-mac", "2B_5000", "2B:5000"],
                          env=dict(os.environ, LADDER_LOCAL=str(local), CHAIN_SECONDS="1",
                                   LADDER_BACKUP=str(tmp / "backup.sh"), BACKUP_CALLS=str(tmp / "backup_calls.txt")),
                          capture_output=True, text=True, timeout=60)
    assert done.returncode == 1 and (local / "follow.pid").read_text().strip() == "999999"
    assert not (tmp / "backup_calls.txt").exists()
    assert "nothing started" in (local / "chain_2B_5000.log").read_text()


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
