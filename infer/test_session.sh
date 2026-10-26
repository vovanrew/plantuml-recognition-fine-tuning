#!/bin/zsh
# The card session of the runs on the test set, driven from this Mac over SSH, one step per call
# (decisions/main-inference-runs/record.md). The runs themselves go on the card unattended under
# infer/test_queue.sh. Nothing here creates, stops or deletes the pod: that is done through Runpod on
# the author's word.
#
#   test_session.sh files                        on the Mac: the files of the study the pack carries
#   test_session.sh pack                         on the Mac, before the pod exists
#   POD_HOST=<ip> POD_PORT=<port> test_session.sh <step>
#   (or with .local/test_session/pod.env holding POD_HOST=, POD_PORT=, POD_ID=)
#
# TEST_IDS limits `start` and `follow` to those runs of infer/test_runs.txt (default: all);
# QUEUE_LOG names the queue's log under results/logs/ (default test_queue.log).
#
# `follow` is started detached, so that it outlives the session that started it:
#   nohup zsh infer/test_session.sh follow >> .local/test_session/follow.out 2>&1 &
#   echo $! > .local/test_session/follow.pid
#
# Steps:
#   files     print the study's files the pack carries that git does not hold: the adapters of
#             infer/test_runs.txt (adapter_config.json, adapter_model.safetensors), the three
#             training pictures of the adapter check (train/dev_scoring.py check_pictures),
#             data/examples/train.jsonl and loss_curve.jsonl (data/examples/SHA256SUMS lists them)
#   pack      refuse unless the session's code is committed, the benchmark's evaluation/ and
#             prompts/ equal its tag scorer-v2, and its data/test_set.json and
#             data/puml_images_1568/ equal its HEAD; the commit and the study's tags as a git
#             bundle, the files above as data.tar, the benchmark's evaluation/, prompts/,
#             data/test_set.json and data/puml_images_1568/ as instrument.tar (its commit in
#             instrument_commit.txt), train/pod_setup.sh; checksums. Into .local/test_session/pack/
#   push      copy the pack to the pod (rsync, resumable) and check it
#   setup     on the pod: train/pod_setup.sh (checkout, data, pinned environment, the card's record)
#   start     start infer/test_queue.sh under nohup on the pod; returns at once
#   status    the queue's log, the answers so far per run, the card's use
#   fetch     copy results/ and then runs/test/ from the pod to this Mac (.local/test_session/results/,
#             runs/test/). results/ first: a run's mark copied by a fetch means its answers were
#             complete before that fetch copied runs/test/
#   follow    until the queue ends: fetch every FOLLOW_SECONDS (default 1800); when a run's mark
#             (results/queue/test_<id>.done) has arrived, check it on this Mac (check_run below), once
#             (.local/test_session/checked/<id>.done). Exits 0 when the queue's last line is
#             "all done" (after checking every run), 1 when it is "stopped: …", 3 after
#             FOLLOW_MAX_FAILS (default 6) fetches in a row failed, 4 when a check failed, 5 when
#             FOLLOW_STOP_AT (epoch seconds) is passed. Its log: .local/test_session/follow.log
#   check <id>
#             check_run alone: infer/manifest.py check on runs/test/<id>/answers, and its run_meta.json
#             started with n_cells 1000
#   run <cmd> any command on the pod, from the checkout
#
# For the tests only: TEST_POD_ROOT=<folder> makes fetch read <folder>/results and
# <folder>/fine-tune/runs/test from this machine instead of the pod; CHECK_PY replaces the Python
# that checks (default .venv/bin/python); TEST_LOCAL replaces .local/test_session.
set -eu
FT=${0:A:h:h}
U=${FT:h}
INSTRUMENT=$U/plantuml-recognition-evaluation
LOCAL=${TEST_LOCAL:-$FT/.local/test_session}
PACK=$LOCAL/pack
KEY=${POD_KEY:-$HOME/.ssh/id_ed25519_runpod}
WORK=${POD_WORK:-/workspace/test}
CHECK_PY=${CHECK_PY:-$FT/.venv/bin/python}
QLOG=${QUEUE_LOG:-test_queue.log}
N_CELLS=1000
step=${1:?step}; shift

ids() {   # the runs of infer/test_runs.txt, or TEST_IDS
    if [[ -n ${TEST_IDS:-} ]]; then echo ${=TEST_IDS}; return; fi
    awk '$1 !~ /^#/ && NF == 3 { print $1 }' $FT/infer/test_runs.txt
}

files() {
    (cd $FT && /usr/bin/python3 - <<'EOF'
import sys
sys.path.insert(0, "train")
from pathlib import Path
import dev_scoring
files = []
for line in open("infer/test_runs.txt", encoding="utf-8"):
    parts = line.split()
    if not parts or parts[0].startswith("#"):
        continue
    if parts[2] != "-":
        files += [f"{parts[2]}/adapter_config.json", f"{parts[2]}/adapter_model.safetensors"]
ft = Path.cwd().resolve()
files += [str(p.resolve().relative_to(ft)) for p in dev_scoring.check_pictures()]
files += ["data/examples/train.jsonl", "data/examples/loss_curve.jsonl"]
assert len(files) == len(set(files)), files
print("\n".join(files))
EOF
    )
}

uncommitted() { git -C $FT status --porcelain -- train build infer requirements-card.txt data/examples data/sets }

packtar() {   # a plain tar made in <dir> without the macOS metadata (as train/ladder_session.sh packtar)
    local dir=$1 out=$2; shift 2
    (cd $dir && COPYFILE_DISABLE=1 tar --no-mac-metadata --no-xattrs -cf $out "$@")
}

check_run() {   # check_run <id>: the run's manifest passes on this Mac, and it was started with all 1,000 cells
    local id=$1 answers=$FT/runs/test/$1/answers
    [[ -f $answers/run_meta.json ]] || { echo "$id: no run_meta.json"; return 1 }
    (cd $FT && $CHECK_PY infer/manifest.py check $answers) || { echo "$id: manifest check failed"; return 1 }
    $CHECK_PY -c "import json, sys; n = json.load(open(sys.argv[1]))['n_cells']; sys.exit(0 if n == $N_CELLS else f'n_cells {n}, not $N_CELLS')" \
        $answers/run_meta.json || { echo "$id: not a run of $N_CELLS cells"; return 1 }
    echo "$id: checked"
}

if [[ $step == files ]]; then files; exit 0; fi
if [[ $step == check ]]; then check_run ${1:?id}; exit $?; fi

if [[ $step == pack ]]; then
    changed=$(uncommitted)
    if [[ -n $changed ]]; then
        echo "commit these first; the pod runs a checkout of a commit:"; echo $changed; exit 1
    fi
    if ! git -C $INSTRUMENT diff --quiet scorer-v2 -- evaluation prompts; then
        echo "$INSTRUMENT: evaluation/ or prompts/ differ from tag scorer-v2"; exit 1
    fi
    if [[ -n $(git -C $INSTRUMENT status --porcelain -- data/test_set.json data/puml_images_1568) ]]; then
        echo "$INSTRUMENT: data/test_set.json or data/puml_images_1568/ differ from its HEAD"; exit 1
    fi
    mkdir -p $PACK
    files > $LOCAL/data_files.txt
    missing=$(cd $FT && while read -r f; do [[ -f $f ]] || echo $f; done < $LOCAL/data_files.txt)
    if [[ -n $missing ]]; then echo "missing files:"; echo $missing | head; exit 1; fi
    bash $FT/train/study_bundle.sh create $FT $PACK/study.bundle
    git -C $FT rev-parse HEAD > $PACK/commit.txt
    git -C $INSTRUMENT rev-parse HEAD > $PACK/instrument_commit.txt
    packtar $FT $PACK/data.tar -T $LOCAL/data_files.txt
    packtar $INSTRUMENT $PACK/instrument.tar evaluation prompts data/test_set.json data/puml_images_1568
    cp $FT/train/pod_setup.sh $PACK/
    (cd $PACK && shasum -a 256 commit.txt instrument_commit.txt study.bundle data.tar instrument.tar pod_setup.sh > SHA256SUMS)
    echo "$(wc -l < $LOCAL/data_files.txt | tr -d ' ') study files"; ls -la $PACK; exit 0
fi

# The pod's address, written when the pod is created: POD_HOST=…, POD_PORT=…, POD_ID=…
if [[ -z ${TEST_POD_ROOT:-} && -z ${POD_HOST:-} && -f $LOCAL/pod.env ]]; then source $LOCAL/pod.env; fi
if [[ -n ${TEST_POD_ROOT:-} ]]; then
    SRC=$TEST_POD_ROOT; RSH=(); WORK=$TEST_POD_ROOT
    remote() { (cd $WORK/fine-tune && bash -c "$1") }
else
    BASE_OPTS=(-i $KEY -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new
               -o UserKnownHostsFile=$LOCAL/known_hosts -o ServerAliveInterval=30 -o ConnectTimeout=30)
    SSH_OPTS=($BASE_OPTS -p ${POD_PORT:?set POD_PORT})
    POD=root@${POD_HOST:?set POD_HOST}
    SRC=$POD:$WORK; RSH=(-e "ssh ${SSH_OPTS[*]}")
    ENV="export HF_HOME=$WORK/hf PYTHONUNBUFFERED=1; cd $WORK/fine-tune"
    remote() { ssh $SSH_OPTS $POD "$ENV; $1" }
fi

fetch() {
    mkdir -p $FT/runs/test $LOCAL/results
    rsync -a $RSH $SRC/results/ $LOCAL/results/ && rsync -a $RSH $SRC/fine-tune/runs/test/ $FT/runs/test/
}

follow() {
    local every=${FOLLOW_SECONDS:-1800} max_fails=${FOLLOW_MAX_FAILS:-6} fails=0 last id
    local log=$LOCAL/follow.log
    mkdir -p $LOCAL/checked
    say() { echo "$(date -u +%FT%TZ) $*" | tee -a $log }
    say "follow: $(ids | tr '\n' ' ')every $every s, queue log $QLOG"
    while true; do
        if fetch >> $log 2>&1; then
            fails=0
        else
            fails=$((fails + 1)); say "fetch failed ($fails in a row)"
            if (( fails >= max_fails )); then say "follow ends: the pod could not be reached"; return 3; fi
        fi
        for id in $(ids); do
            if [[ -e $LOCAL/results/queue/test_$id.done && ! -e $LOCAL/checked/$id.done ]]; then
                if check_run $id >> $log 2>&1; then touch $LOCAL/checked/$id.done; say "checked $id"
                else say "follow ends: the check of $id failed"; return 4; fi
            fi
        done
        last=$(tail -n 1 $LOCAL/results/logs/$QLOG 2>/dev/null || true)
        case $last in
            *"all done") say "follow ends: the queue is done"; return 0 ;;
            *"stopped:"*) say "follow ends: the queue ${last#* }"; return 1 ;;
        esac
        if [[ -n ${FOLLOW_STOP_AT:-} ]] && (( $(date +%s) >= FOLLOW_STOP_AT )); then
            say "follow ends: the stop time is reached"; return 5
        fi
        sleep $every
    done
}

case $step in
  push)
    ssh $SSH_OPTS $POD "mkdir -p $WORK/pack"
    rsync -a --partial --progress $RSH \
        $PACK/{commit.txt,instrument_commit.txt,study.bundle,data.tar,instrument.tar,pod_setup.sh,SHA256SUMS} $POD:$WORK/pack/
    ssh $SSH_OPTS $POD "cd $WORK/pack && sha256sum --quiet -c SHA256SUMS && echo pack ok" ;;
  setup)
    ssh $SSH_OPTS $POD "bash $WORK/pack/pod_setup.sh $WORK 2>&1 | tee $WORK/pack/setup.log; exit \${PIPESTATUS[0]}"
    remote "echo \$(ls ../plantuml-recognition-evaluation/data/puml_images_1568 | wc -l) test pictures" ;;
  start)
    remote "QUEUE_LOG=$QLOG nohup bash infer/test_queue.sh ${TEST_IDS:-} < /dev/null > $WORK/results/logs/test_queue.out 2>&1 &"
    echo "started; follow with: test_session.sh status" ;;
  status)
    remote "tail -n 12 $WORK/results/logs/$QLOG 2>/dev/null; for d in runs/test/*; do [ -d \$d/answers ] && echo \$d \$(ls \$d/answers | grep -c '[.]puml\$') answers; done; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader; df -h $WORK | tail -1" ;;
  fetch)     fetch; ls -la $FT/runs/test ;;
  follow)    follow ;;
  run)       remote "$*" ;;
  *)         echo "unknown step $step"; exit 2 ;;
esac
