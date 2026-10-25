#!/bin/zsh
# The card session of the training runs, driven from this Mac over SSH, one step per call
# (decisions/ladder-training/card_session_plan.md; decisions/data-efficiency-runs/implement.md). The
# runs themselves go on the card unattended under train/ladder_queue.sh. Nothing here creates,
# stops or deletes the pod: that is done through Runpod on the author's word.
#
#   ladder_session.sh files                        on the Mac: the data files the pack carries
#   ladder_session.sh pack                         on the Mac, before the pod exists
#   POD_HOST=<ip> POD_PORT=<port> ladder_session.sh <step>
#   (or with .local/ladder_session/pod.env holding POD_HOST=, POD_PORT=, POD_ID=)
#
# A run is <size>:<pairs> (a bare <size> means <size>:20000) and its id <size>_<pairs>.
# LADDER_RUNS lists the runs `follow` scores (default "2B:20000 9B:20000 27B:20000", the ladder);
# QUEUE_LOG names the queue's log under results/logs/ (default queue.log).
#
# `follow` is started detached, so that it outlives the session that started it:
#   nohup zsh train/ladder_session.sh follow >> .local/ladder_session/follow.out 2>&1 &
#   echo $! > .local/ladder_session/follow.pid
#
# Steps:
#   files     print the data files of the pack, one per line: the 20,000 training pictures of
#             data/sets/train.json, the 200 dev pictures, data/examples/train.jsonl and
#             loss_curve.jsonl (the files git does not hold)
#   pack      refuse unless the session's code is committed; the commit and the study's tags as a
#             git bundle, the data files above as data.tar, the benchmark's evaluation/ and
#             prompts/, train/pod_setup.sh; checksums. Into .local/ladder_session/pack/
#   push      copy the pack to the pod (rsync, resumable: data.tar is about 4 GB) and check it
#   setup     on the pod: train/pod_setup.sh (checkout, data, pinned environment, the card's record)
#   start     start train/ladder_queue.sh under nohup on the pod; returns at once.
#             start <run> ... limits it to those runs
#   status    the queue's log, each run's last training step, the answers so far, the card's use
#   fetch     copy results/ and then runs/ from the pod to this Mac (.local/ladder_session/results/,
#             runs/). results/ first: a run's answer mark copied by a fetch means its answers were
#             complete before that fetch copied runs/
#   follow    until the queue ends: fetch every FOLLOW_SECONDS (default 1800); when a run of
#             LADDER_RUNS has its answers complete (results/queue/answer_<id>.done has arrived),
#             score its four checkpoints and apply the checkpoint rule on this Mac (score_run
#             below), once (.local/ladder_session/scored/<id>.done). Exits 0 when the queue's last
#             line is "all done" (after scoring every run), 1 when it is "stopped: …", 3 after
#             FOLLOW_MAX_FAILS (default 6) fetches in a row failed, 4 when scoring or selection
#             failed, 5 when FOLLOW_STOP_AT (epoch seconds) is passed. Its log:
#             .local/ladder_session/follow.log
#   chain <run> ...
#             queue further runs behind the queue that runs now (QUEUE_LOG), on the commit now
#             checked out on this Mac, which must be committed: the commit as a bundle and
#             train/ladder_chain.sh to the pod's pack_<tag>/ (tag: the run ids joined by "-"),
#             train/ladder_chain.sh started there (it waits for the queue's "all done", moves the
#             checkout, starts the runs with the log queue_<tag>.log), and `chain-mac` started here
#             under nohup. Returns at once. Logs: the pod's results/logs/chain_<tag>.log and
#             .local/ladder_session/chain_<tag>.log
#   chain-mac <tag> <run> ...
#             on this Mac: wait until the running `follow` (follow.pid) has ended with "the queue is
#             done", then start `follow` for the chained runs (LADDER_RUNS, QUEUE_LOG=queue_<tag>.log)
#             with caffeinate, and .local/backup/pack_ladder.sh for them. Ends 0 once both are
#             started, 1 when `follow` ended otherwise (nothing is started)
#   run <cmd> any command on the pod, from the checkout
#
# For the tests only: LADDER_POD_ROOT=<folder> makes fetch read <folder>/results and
# <folder>/fine-tune/runs from this machine, and `chain` copy and run there, instead of the pod;
# SCORE_PY replaces the Python that scores (default /usr/bin/python3, the scorer's 3.9);
# CHAIN_SECONDS (60) is the waiters' interval; LADDER_BACKUP names the backup script.
set -eu
FT=${0:A:h:h}
U=${FT:h}
LOCAL=${LADDER_LOCAL:-$FT/.local/ladder_session}
PACK=$LOCAL/pack
KEY=${POD_KEY:-$HOME/.ssh/id_ed25519_runpod}
WORK=${POD_WORK:-/workspace/ladder}
SCORE_PY=${SCORE_PY:-/usr/bin/python3}
RUNS=(${=${LADDER_RUNS:-2B:20000 9B:20000 27B:20000}})
QLOG=${QUEUE_LOG:-queue.log}
step=${1:?step}; shift

run_id() { local s=${1%%:*} p=20000; [[ $1 != $s ]] && p=${1#*:}; echo ${s}_$p }

files() {
    (cd $FT && /usr/bin/python3 - <<'EOF'
import json
rows = json.load(open("data/sets/train.json"))["diagrams"]
dev = json.load(open("data/sets/dev.json"))["diagrams"]
pic = lambda name, key: f"data/sets/render/{name}/puml_images_1568/{key[:-len('.puml')]}.png"
files = [pic("train", r["key"]) for r in rows] + [pic("dev", r["key"]) for r in dev]
files += ["data/examples/train.jsonl", "data/examples/loss_curve.jsonl"]
assert len(files) == len(set(files)) == 20202, len(files)
print("\n".join(files))
EOF
    )
}

uncommitted() { git -C $FT status --porcelain -- train build infer requirements-card.txt data/examples data/sets }

if [[ $step == files ]]; then files; exit 0; fi

if [[ $step == pack ]]; then
    changed=$(uncommitted)
    if [[ -n $changed ]]; then
        echo "commit these first; the pod runs a checkout of a commit:"; echo $changed; exit 1
    fi
    mkdir -p $PACK
    files > $LOCAL/data_files.txt
    missing=$(cd $FT && while read -r f; do [[ -f $f ]] || echo $f; done < $LOCAL/data_files.txt)
    if [[ -n $missing ]]; then echo "missing data files:"; echo $missing | head; exit 1; fi
    bash $FT/train/study_bundle.sh create $FT $PACK/study.bundle
    git -C $FT rev-parse HEAD > $PACK/commit.txt
    (cd $FT && tar -cf $PACK/data.tar -T $LOCAL/data_files.txt)
    (cd $U/plantuml-recognition-evaluation && tar -cf $PACK/instrument.tar evaluation prompts)
    cp $FT/train/pod_setup.sh $PACK/
    (cd $PACK && shasum -a 256 commit.txt study.bundle data.tar instrument.tar pod_setup.sh > SHA256SUMS)
    echo "$(wc -l < $LOCAL/data_files.txt | tr -d ' ') data files"; ls -la $PACK; exit 0
fi

checkpoints() {   # checkpoints <pairs>
    (cd $FT && /usr/bin/python3 -c "import sys; sys.path.insert(0, 'train'); import recipe as R; print(' '.join(R.checkpoint_dir(s) for s in R.checkpoint_steps($1)))")
}

score_run() {   # score_run <size> <pairs>: the four checkpoints under scorer v2, then the checkpoint rule
    local s=$1 p=$2 c scores=()
    for c in $(checkpoints $p); do
        (cd $FT && $SCORE_PY train/dev_scoring.py score --answers runs/dev/${s}_$p/$c/answers \
            --out runs/dev/${s}_$p/$c/scores) || return 1
        scores+=(runs/dev/${s}_$p/$c/scores)
    done
    [[ ${#scores} -eq 4 ]] || return 1
    (cd $FT && $SCORE_PY train/select_checkpoint.py --pairs $p \
        --out runs/train/${s}_$p/selection.json $scores) || return 1
}

answered() {   # answered <id> <size> <pairs>: its answer mark has arrived (the ladder's first
               # session marked 20,000-pair runs by size alone)
    [[ -e $LOCAL/results/queue/answer_$1.done ]] && return 0
    [[ $3 == 20000 && -e $LOCAL/results/queue/answer_$2.done ]]
}

scored() {   # scored <id> <size> <pairs>: as answered, for this Mac's mark
    [[ -e $LOCAL/scored/$1.done ]] && return 0
    [[ $3 == 20000 && -e $LOCAL/scored/$2.done ]]
}

chain_mac() {   # chain_mac <tag> <run> ...
    local tag=$1; shift
    local log=$LOCAL/chain_$tag.log pid last
    say_chain() { echo "$(date -u +%FT%TZ) $*" >> $log }
    pid=$(cat $LOCAL/follow.pid)
    say_chain "chain-mac $tag: waiting for follow ($pid) to end"
    while kill -0 $pid 2>/dev/null; do sleep ${CHAIN_SECONDS:-60}; done
    last=$(tail -n 1 $LOCAL/follow.log 2>/dev/null || true)
    if [[ $last != *"the queue is done" ]]; then
        say_chain "chain-mac $tag ends: follow ended with: ${last#* }; nothing started"; return 1
    fi
    LADDER_RUNS="$*" QUEUE_LOG=queue_$tag.log nohup zsh $FT/train/ladder_session.sh follow \
        >> $LOCAL/follow.out 2>&1 < /dev/null &
    pid=$!; echo $pid > $LOCAL/follow.pid
    nohup caffeinate -i -w $pid > /dev/null 2>&1 < /dev/null &
    echo $! > $LOCAL/caffeinate.pid
    LADDER_RUNS="$*" nohup bash ${LADDER_BACKUP:-$FT/.local/backup/pack_ladder.sh} \
        >> $LOCAL/backup.out 2>&1 < /dev/null &
    echo $! > $LOCAL/backup.pid
    say_chain "chain-mac $tag ends: follow ($pid) and the backup ($(cat $LOCAL/backup.pid)) started for $*"
}

if [[ $step == chain-mac ]]; then chain_mac "$@"; exit $?; fi

# The pod's address, written when the pod is created, so that any later session (or a restart of
# `follow`) reaches the same pod without being told: POD_HOST=…, POD_PORT=…, POD_ID=…
if [[ -z ${LADDER_POD_ROOT:-} && -z ${POD_HOST:-} && -f $LOCAL/pod.env ]]; then source $LOCAL/pod.env; fi
if [[ -n ${LADDER_POD_ROOT:-} ]]; then
    SRC=$LADDER_POD_ROOT; RSH=(); WORK=$LADDER_POD_ROOT
    remote() { (cd $WORK/fine-tune && bash -c "$1") }
    copy_to_pod() { mkdir -p $WORK/$2 && cp $1 $WORK/$2/ }
else
    BASE_OPTS=(-i $KEY -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new
               -o UserKnownHostsFile=$LOCAL/known_hosts -o ServerAliveInterval=30 -o ConnectTimeout=30)
    SSH_OPTS=($BASE_OPTS -p ${POD_PORT:?set POD_PORT}); SCP_OPTS=($BASE_OPTS -P $POD_PORT)
    POD=root@${POD_HOST:?set POD_HOST}
    SRC=$POD:$WORK; RSH=(-e "ssh ${SSH_OPTS[*]}")
    ENV="export HF_HOME=$WORK/hf PYTHONUNBUFFERED=1; cd $WORK/fine-tune"
    remote() { ssh $SSH_OPTS $POD "$ENV; $1" }
    copy_to_pod() { ssh $SSH_OPTS $POD "mkdir -p $WORK/$2" && scp -q $SCP_OPTS $1 $POD:$WORK/$2/ }
fi

fetch() {
    mkdir -p $FT/runs $LOCAL/results
    rsync -a $RSH $SRC/results/ $LOCAL/results/ && rsync -a $RSH $SRC/fine-tune/runs/ $FT/runs/
}

follow() {
    local every=${FOLLOW_SECONDS:-1800} max_fails=${FOLLOW_MAX_FAILS:-6} fails=0 last r id s p
    local log=$LOCAL/follow.log
    mkdir -p $LOCAL/scored
    say() { echo "$(date -u +%FT%TZ) $*" | tee -a $log }
    say "follow: ${RUNS[*]}, every $every s, queue log $QLOG"
    while true; do
        if fetch >> $log 2>&1; then
            fails=0
        else
            fails=$((fails + 1)); say "fetch failed ($fails in a row)"
            if (( fails >= max_fails )); then say "follow ends: the pod could not be reached"; return 3; fi
        fi
        for r in $RUNS; do
            id=$(run_id $r); s=${id%%_*}; p=${id#*_}
            if answered $id $s $p && ! scored $id $s $p; then
                say "scoring $id"
                if score_run $s $p >> $log 2>&1; then touch $LOCAL/scored/$id.done; say "scored and selected $id"
                else say "follow ends: scoring or selection of $id failed"; return 4; fi
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

chain() {   # chain <run> ...: see the header
    (( $# > 0 )) || { echo "no run given"; return 2; }
    local changed=$(uncommitted) r ids=() tag cp=$LOCAL/pack_code
    if [[ -n $changed ]]; then echo "commit these first; the pod runs a checkout of a commit:"; echo $changed; return 1; fi
    for r in "$@"; do ids+=($(run_id $r)); done
    tag=${(j:-:)ids}
    mkdir -p $cp
    git -C $FT bundle create $cp/study.bundle HEAD --tags 2>/dev/null
    git -C $FT rev-parse HEAD > $cp/commit.txt
    cp $FT/train/ladder_chain.sh $cp/
    for f in study.bundle commit.txt ladder_chain.sh; do copy_to_pod $cp/$f pack_$tag; done
    remote "nohup bash $WORK/pack_$tag/ladder_chain.sh $WORK $tag $QLOG $* < /dev/null > $WORK/results/logs/chain_$tag.out 2>&1 &"
    nohup zsh $FT/train/ladder_session.sh chain-mac $tag "$@" < /dev/null > $LOCAL/chain_$tag.out 2>&1 &
    echo $! > $LOCAL/chain_mac.pid
    echo "chained $tag behind $QLOG at $(cat $cp/commit.txt): the pod's results/logs/chain_$tag.log, $LOCAL/chain_$tag.log"
}

case $step in
  push)
    ssh $SSH_OPTS $POD "mkdir -p $WORK/pack"
    rsync -a --partial --progress $RSH \
        $PACK/{commit.txt,study.bundle,data.tar,instrument.tar,pod_setup.sh,SHA256SUMS} $POD:$WORK/pack/
    ssh $SSH_OPTS $POD "cd $WORK/pack && sha256sum --quiet -c SHA256SUMS && echo pack ok" ;;
  setup)
    ssh $SSH_OPTS $POD "bash $WORK/pack/pod_setup.sh $WORK 2>&1 | tee $WORK/pack/setup.log; exit \${PIPESTATUS[0]}"
    remote "echo \$(ls data/sets/render/train/puml_images_1568 | wc -l) training pictures, \$(ls data/sets/render/dev/puml_images_1568 | wc -l) dev pictures" ;;
  start)
    remote "QUEUE_LOG=$QLOG nohup bash train/ladder_queue.sh $* < /dev/null > $WORK/results/logs/queue.out 2>&1 &"
    echo "started; follow with: ladder_session.sh status" ;;
  status)
    remote "tail -n 8 $WORK/results/logs/$QLOG 2>/dev/null; for r in runs/train/*_*; do [ -f \$r/log.jsonl ] && echo \$r step \$(tail -n 1 \$r/log.jsonl | cut -c1-120); done; for d in runs/dev/*_*/step_*; do [ -d \$d/answers ] && echo \$d \$(ls \$d/answers | grep -c '[.]puml\$') answers; done; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader; df -h $WORK | tail -1" ;;
  fetch)     fetch; ls -la $FT/runs/* ;;
  follow)    follow ;;
  chain)     chain "$@" ;;
  run)       remote "$*" ;;
  *)         echo "unknown step $step"; exit 2 ;;
esac
