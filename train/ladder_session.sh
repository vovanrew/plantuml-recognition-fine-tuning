#!/bin/zsh
# The ladder's card session, driven from this Mac over SSH, one step per call
# (decisions/ladder-training/card_session_plan.md). The runs themselves go on the card unattended
# under train/ladder_queue.sh. Nothing here creates, stops or deletes the pod: that is done
# through Runpod on the author's word.
#
#   ladder_session.sh files                        on the Mac: the data files the pack carries
#   ladder_session.sh pack                         on the Mac, before the pod exists
#   POD_HOST=<ip> POD_PORT=<port> ladder_session.sh <step>
#   (or with .local/ladder_session/pod.env holding POD_HOST=, POD_PORT=, POD_ID=)
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
#             start <size> ... limits it to those sizes
#   status    the queue's log, the training's last step, the answers so far, the card's use
#   fetch     copy results/ and then runs/ from the pod to this Mac (.local/ladder_session/results/,
#             runs/). results/ first: a size's answer mark copied by a fetch means its answers were
#             complete before that fetch copied runs/
#   follow    until the queue ends: fetch every FOLLOW_SECONDS (default 1800); when a size's
#             answers are complete (its results/queue/answer_<size>.done has arrived), score its
#             four checkpoints and apply the checkpoint rule on this Mac (score_size below), once.
#             Exits 0 when the queue's last line is "all done" (after scoring every size), 1 when
#             it is "stopped: …", 3 after FOLLOW_MAX_FAILS (default 6) fetches in a row failed,
#             4 when scoring or selection failed, 5 when FOLLOW_STOP_AT (epoch seconds) is passed.
#             Its log: .local/ladder_session/follow.log
#   run <cmd> any command on the pod, from the checkout
#
# For the tests only: LADDER_POD_ROOT=<folder> makes fetch read <folder>/results and
# <folder>/fine-tune/runs from this machine instead of the pod; SCORE_PY replaces the Python that
# scores (default /usr/bin/python3, the scorer's 3.9).
set -eu
FT=${0:A:h:h}
U=${FT:h}
LOCAL=${LADDER_LOCAL:-$FT/.local/ladder_session}
PACK=$LOCAL/pack
KEY=${POD_KEY:-$HOME/.ssh/id_ed25519_runpod}
WORK=${POD_WORK:-/workspace/ladder}
SCORE_PY=${SCORE_PY:-/usr/bin/python3}
PAIRS=20000
step=${1:?step}; shift

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

if [[ $step == files ]]; then files; exit 0; fi

if [[ $step == pack ]]; then
    changed=$(git -C $FT status --porcelain -- train build infer requirements-card.txt data/examples data/sets)
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

# The pod's address, written when the pod is created, so that any later session (or a restart of
# `follow`) reaches the same pod without being told: POD_HOST=…, POD_PORT=…, POD_ID=…
if [[ -z ${LADDER_POD_ROOT:-} && -z ${POD_HOST:-} && -f $LOCAL/pod.env ]]; then source $LOCAL/pod.env; fi
if [[ -n ${LADDER_POD_ROOT:-} ]]; then
    SRC=$LADDER_POD_ROOT; RSH=()
else
    SSH_OPTS=(-i $KEY -p ${POD_PORT:?set POD_PORT} -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new
              -o UserKnownHostsFile=$LOCAL/known_hosts -o ServerAliveInterval=30 -o ConnectTimeout=30)
    POD=root@${POD_HOST:?set POD_HOST}
    SRC=$POD:$WORK; RSH=(-e "ssh ${SSH_OPTS[*]}")
fi
ENV="export HF_HOME=$WORK/hf PYTHONUNBUFFERED=1; cd $WORK/fine-tune"
remote() { ssh $SSH_OPTS $POD "$ENV; $1" }

fetch() {
    mkdir -p $FT/runs $LOCAL/results
    rsync -a $RSH $SRC/results/ $LOCAL/results/ && rsync -a $RSH $SRC/fine-tune/runs/ $FT/runs/
}

checkpoints() {
    (cd $FT && /usr/bin/python3 -c "import sys; sys.path.insert(0, 'train'); import recipe as R; print(' '.join(R.checkpoint_dir(s) for s in R.checkpoint_steps($PAIRS)))")
}

score_size() {   # score_size <size>: the four checkpoints under scorer v2, then the checkpoint rule
    local s=$1 c scores=()
    for c in $(checkpoints); do
        (cd $FT && $SCORE_PY train/dev_scoring.py score --answers runs/dev/${s}_$PAIRS/$c/answers \
            --out runs/dev/${s}_$PAIRS/$c/scores) || return 1
        scores+=(runs/dev/${s}_$PAIRS/$c/scores)
    done
    (cd $FT && $SCORE_PY train/select_checkpoint.py --pairs $PAIRS \
        --out runs/train/${s}_$PAIRS/selection.json $scores) || return 1
}

follow() {
    local every=${FOLLOW_SECONDS:-1800} max_fails=${FOLLOW_MAX_FAILS:-6} fails=0 last s
    local log=$LOCAL/follow.log
    mkdir -p $LOCAL/scored
    say() { echo "$(date -u +%FT%TZ) $*" | tee -a $log }
    say "follow: every $every s"
    while true; do
        if fetch >> $log 2>&1; then
            fails=0
        else
            fails=$((fails + 1)); say "fetch failed ($fails in a row)"
            if (( fails >= max_fails )); then say "follow ends: the pod could not be reached"; return 3; fi
        fi
        for s in 2B 9B 27B; do
            if [[ -e $LOCAL/results/queue/answer_$s.done && ! -e $LOCAL/scored/$s.done ]]; then
                say "scoring $s"
                if score_size $s >> $log 2>&1; then touch $LOCAL/scored/$s.done; say "scored and selected $s"
                else say "follow ends: scoring or selection of $s failed"; return 4; fi
            fi
        done
        last=$(tail -n 1 $LOCAL/results/logs/queue.log 2>/dev/null || true)
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
        $PACK/{commit.txt,study.bundle,data.tar,instrument.tar,pod_setup.sh,SHA256SUMS} $POD:$WORK/pack/
    ssh $SSH_OPTS $POD "cd $WORK/pack && sha256sum --quiet -c SHA256SUMS && echo pack ok" ;;
  setup)
    ssh $SSH_OPTS $POD "bash $WORK/pack/pod_setup.sh $WORK 2>&1 | tee $WORK/pack/setup.log; exit \${PIPESTATUS[0]}"
    remote "echo \$(ls data/sets/render/train/puml_images_1568 | wc -l) training pictures, \$(ls data/sets/render/dev/puml_images_1568 | wc -l) dev pictures" ;;
  start)
    remote "nohup bash train/ladder_queue.sh $* < /dev/null > $WORK/results/logs/queue.out 2>&1 &"
    echo "started; follow with: ladder_session.sh status" ;;
  status)
    remote "tail -n 8 $WORK/results/logs/queue.log 2>/dev/null; for r in runs/train/*_20000; do [ -f \$r/log.jsonl ] && echo \$r step \$(tail -n 1 \$r/log.jsonl | cut -c1-120); done; for d in runs/dev/*_20000/step_*; do [ -d \$d/answers ] && echo \$d \$(ls \$d/answers | grep -c '[.]puml\$') answers; done; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader; df -h $WORK | tail -1" ;;
  fetch)     fetch; ls -la $FT/runs/* ;;
  follow)    follow ;;
  run)       remote "$*" ;;
  *)         echo "unknown step $step"; exit 2 ;;
esac
