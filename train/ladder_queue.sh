#!/bin/bash
# On the card: the ladder's runs one after another, unattended (decisions/ladder-training/
# card_session_plan.md). For each size, in the plan's order 2B → 9B → 27B:
#   download   the model at its pinned commit (train/download_models.py; quick when cached)
#   train      train/train.py run --size <s> --pairs 20000 --out runs/train/<s>_20000
#   answer     train/answer_checkpoints.sh: the run's four checkpoints on the 200 dev diagrams,
#              each after its adapter check, the run manifest beside each; several servers at
#              once only at the 2B (4); one at a time at the 9B (no gain side by side) and the 27B
#              (one server's weights fill the card)
# Scoring and the checkpoint rule run on the Mac after `fetch` (methodology/training-recipe.md §2).
#
# Run from the checkout, under nohup (train/ladder_session.sh `start` does):
#   bash train/ladder_queue.sh [<size> ...]          default: 2B 9B 27B
#
# Each finished step leaves results/queue/<step>_<size>.done; a later start skips it, so the
# queue can be started again after a stop. It stops at the first failure and writes the reason to
# results/logs/queue.log. A run folder that exists without its .done is never trained over: a run
# stopped part way is made again from step 0 in a folder the operator frees by hand (the recipe
# saves the optimizer state with the last checkpoint only). Logs: results/logs/queue.log, the
# step's own log in results/logs/<step>_<size>.log, the answers' logs in
# results/logs/answers_<size>/. The queue's last line is "all done" or "stopped: <reason>".
set -u
SIZES=("$@"); [ ${#SIZES[@]} -gt 0 ] || SIZES=(2B 9B 27B)
WORK=$(cd .. && pwd)
PY=${PY:-$WORK/venv/bin/python}
LOGS=$WORK/results/logs
MARKS=$WORK/results/queue
PAIRS=20000
mkdir -p "$LOGS" "$MARKS"

say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOGS/queue.log"; }
stop() { say "stopped: $*"; exit 1; }

for s in "${SIZES[@]}"; do
    case $s in 2B|9B|27B) ;; *) stop "unknown size $s" ;; esac
done

# The checkpoint folders of a 20,000-pair run, from the recipe itself.
STEPS=$("$PY" -c "import sys; sys.path.insert(0, 'train'); import recipe as R; print(' '.join(R.checkpoint_dir(s) for s in R.checkpoint_steps($PAIRS)))") \
    || stop "could not read the checkpoint steps from train/recipe.py"
[ -n "$STEPS" ] || stop "train/recipe.py gave no checkpoint steps"

step() {   # step <name> <size> <command...>: run once, mark when it exits 0
    local name=$1 size=$2; shift 2
    if [ -e "$MARKS/${name}_$size.done" ]; then say "$name $size: done before, skipped"; return 0; fi
    say "$name $size: start"
    "$@" >> "$LOGS/${name}_$size.log" 2>&1
    local code=$?
    [ $code -eq 0 ] || stop "$name $size exited with $code (results/logs/${name}_$size.log)"
    touch "$MARKS/${name}_$size.done"
    say "$name $size: done"
}

answer() {   # answer <size>: the four checkpoints, then every answer log must end in "exit 0"
    local size=$1 at_once=1
    [ "$size" = 2B ] && at_once=4
    # shellcheck disable=SC2086
    bash train/answer_checkpoints.sh "$size" "runs/train/${size}_$PAIRS" "runs/dev/${size}_$PAIRS" "$at_once" $STEPS \
        || return $?
    mkdir -p "$LOGS/answers_$size"
    local bad=0
    for c in $STEPS; do
        local f=$LOGS/answer_$c.log
        if [ -f "$f" ] && [ "$(tail -n 1 "$f")" = "exit 0" ]; then mv "$f" "$LOGS/answers_$size/"
        else echo "checkpoint $c: $( [ -f "$f" ] && tail -n 1 "$f" || echo 'no log')"; bad=1
            [ -f "$f" ] && mv "$f" "$LOGS/answers_$size/"
        fi
    done
    return $bad
}

say "queue: ${SIZES[*]}; checkpoints $STEPS"
for s in "${SIZES[@]}"; do
    run=runs/train/${s}_$PAIRS
    step download "$s" "$PY" train/download_models.py "$WORK/results/downloads_$s.json" "$s"
    if [ ! -e "$MARKS/train_$s.done" ] && [ -e "$run" ]; then
        stop "$run exists without results/queue/train_$s.done: a run stopped part way; free the folder by hand"
    fi
    step train "$s" "$PY" train/train.py run --size "$s" --pairs $PAIRS --out "$run"
    if [ ! -e "$MARKS/answer_$s.done" ] && [ -e "runs/dev/${s}_$PAIRS" ]; then
        stop "runs/dev/${s}_$PAIRS exists without results/queue/answer_$s.done; resume each checkpoint by hand (train/answer_checkpoints.sh header)"
    fi
    step answer "$s" answer "$s"
done
say "all done"
