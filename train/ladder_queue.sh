#!/bin/bash
# On the card: training runs one after another, unattended (decisions/ladder-training/
# card_session_plan.md). For each run <size>:<pairs> in the order given (default the ladder,
# 2B:20000 9B:20000 27B:20000; a bare <size> means <size>:20000):
#   download   the model at its pinned commit (train/download_models.py; quick when cached)
#   train      train/train.py run --size <size> --pairs <pairs> --out runs/train/<size>_<pairs>
#   answer     train/answer_checkpoints.sh: the run's four checkpoints (train/recipe.py
#              checkpoint_steps(<pairs>)) on the 200 dev diagrams, each after its adapter check,
#              the run manifest beside each; several servers at once only at the 2B (4); one at a
#              time at the 9B (no gain side by side) and the 27B (one server's weights fill the card)
# Scoring and the checkpoint rule run on the Mac (train/ladder_session.sh follow).
#
# Run from the checkout, under nohup (train/ladder_session.sh `start` does):
#   [QUEUE_LOG=<name>] bash train/ladder_queue.sh [<size>[:<pairs>] ...]
#
# A run's id is <size>_<pairs>. Each finished step leaves results/queue/<step>_<id>.done; a later
# start skips it, so the queue can be started again after a stop. It stops at the first failure and
# writes the reason to its log, results/logs/$QUEUE_LOG (default queue.log). A run or dev folder
# that exists without its .done is never written over: a run stopped part way is made again from
# step 0 in a folder the operator frees by hand (the recipe saves the optimizer state with the last
# checkpoint only). The step's own log is results/logs/<step>_<id>.log, the answers' logs
# results/logs/answers_<id>/. The log's last line is "all done" or "stopped: <reason>".
set -u
RUNS=("$@"); [ ${#RUNS[@]} -gt 0 ] || RUNS=(2B:20000 9B:20000 27B:20000)
WORK=$(cd .. && pwd)
PY=${PY:-$WORK/venv/bin/python}
LOGS=$WORK/results/logs
MARKS=$WORK/results/queue
QLOG=$LOGS/${QUEUE_LOG:-queue.log}
mkdir -p "$LOGS" "$MARKS"

say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$QLOG"; }
stop() { say "stopped: $*"; exit 1; }

checkpoints() {   # checkpoints <pairs>: the run's checkpoint folders, from the recipe itself; fails
                  # for a run length the recipe does not define (SUBSET_PAIRS)
    "$PY" -c "import sys; sys.path.insert(0, 'train'); import recipe as R; assert $1 in R.SUBSET_PAIRS; print(' '.join(R.checkpoint_dir(s) for s in R.checkpoint_steps($1)))"
}

IDS=(); SIZES=(); PAIRS=()
for r in "${RUNS[@]}"; do
    s=${r%%:*}; p=20000; [ "$r" != "$s" ] && p=${r#*:}
    case $s in 2B|9B|27B) ;; *) stop "unknown size $s" ;; esac
    steps=$(checkpoints "$p" 2>/dev/null) || stop "train/recipe.py has no run of $p pairs"
    [ -n "$steps" ] || stop "train/recipe.py gave no checkpoint steps for $p pairs"
    IDS+=("${s}_$p"); SIZES+=("$s"); PAIRS+=("$p")
done

step() {   # step <name> <id> <command...>: run once, mark when it exits 0
    local name=$1 id=$2; shift 2
    if [ -e "$MARKS/${name}_$id.done" ]; then say "$name $id: done before, skipped"; return 0; fi
    say "$name $id: start"
    "$@" >> "$LOGS/${name}_$id.log" 2>&1
    local code=$?
    [ $code -eq 0 ] || stop "$name $id exited with $code (results/logs/${name}_$id.log)"
    touch "$MARKS/${name}_$id.done"
    say "$name $id: done"
}

answer() {   # answer <size> <pairs>: the four checkpoints, then every answer log must end in "exit 0"
    local size=$1 pairs=$2 id=${1}_$2 at_once=1 steps
    [ "$size" = 2B ] && at_once=4
    steps=$(checkpoints "$pairs") || return 1
    # shellcheck disable=SC2086
    bash train/answer_checkpoints.sh "$size" "runs/train/$id" "runs/dev/$id" "$at_once" $steps || return $?
    mkdir -p "$LOGS/answers_$id"
    local bad=0 c f
    for c in $steps; do
        f=$LOGS/answer_$c.log
        if [ -f "$f" ] && [ "$(tail -n 1 "$f")" = "exit 0" ]; then mv "$f" "$LOGS/answers_$id/"
        else echo "checkpoint $c: $( [ -f "$f" ] && tail -n 1 "$f" || echo 'no log')"; bad=1
            [ -f "$f" ] && mv "$f" "$LOGS/answers_$id/"
        fi
    done
    return $bad
}

say "queue: ${IDS[*]}"
for i in "${!IDS[@]}"; do
    id=${IDS[$i]}; s=${SIZES[$i]}; p=${PAIRS[$i]}
    step download "$id" "$PY" train/download_models.py "$WORK/results/downloads_$s.json" "$s"
    if [ ! -e "$MARKS/train_$id.done" ] && [ -e "runs/train/$id" ]; then
        stop "runs/train/$id exists without results/queue/train_$id.done: a run stopped part way; free the folder by hand"
    fi
    step train "$id" "$PY" train/train.py run --size "$s" --pairs "$p" --out "runs/train/$id"
    if [ ! -e "$MARKS/answer_$id.done" ] && [ -e "runs/dev/$id" ]; then
        stop "runs/dev/$id exists without results/queue/answer_$id.done; resume each checkpoint by hand (train/answer_checkpoints.sh header)"
    fi
    step answer "$id" answer "$s" "$p"
done
say "all done"
