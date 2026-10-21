#!/bin/bash
# On the card: the dev answers of a training run's checkpoints, several at once, each checkpoint
# answered by its own server on its own port (decisions/pilot-run/card_session_plan.md). Every
# server is the same checkout with only the adapter different, and answers one request at a time;
# running several side by side changes no answer, only the wall-clock. train/dev_scoring.py `answer` runs the adapter check first, refuses
# a port in use, and writes the run manifest while its server still runs.
#
# Run from the checkout, under nohup (train/card_session.sh `answer` does):
#   bash train/answer_checkpoints.sh <size> <run folder> <dev folder> <at once> <checkpoint>...
#   e.g.  bash train/answer_checkpoints.sh 2B runs/train/2B_1000 runs/dev/2B_1000 4 \
#             step_00032 step_00063 step_00095 step_00126
# <at once>: how many servers share the card. Each holds the base model: 2B about 6 GB, so all
# four fit; 27B 55.6 GB of weights, so one.
# Ports 8000, 8002, 8003, …: 8001 is taken by the container image's nginx.
# Logs: <work>/results/logs/answer_<checkpoint>.log, each ending in "exit <code>"; this script's
# own log ends in "all done". A checkpoint folder that exists is refused; an interrupted
# checkpoint is resumed by its own dev_scoring.py answer command (same pod, same checkout), which
# re-attempts the missing cells in its folder.
set -u
SIZE=${1:?size}; RUN=${2:?run folder}; DEV=${3:?dev folder}; AT_ONCE=${4:?at once}; shift 4
[ $# -gt 0 ] || { echo "no checkpoint given"; exit 2; }
WORK=$(cd .. && pwd)
PY=${PY:-$WORK/venv/bin/python}
LOGS=$WORK/results/logs
mkdir -p "$LOGS"

one() {
    local step=$1 port=$2
    {
        "$PY" train/dev_scoring.py answer --size "$SIZE" --adapter "$(pwd)/$RUN/$step" \
            --out "$DEV/$step" --timeout 600 --port "$port"
        echo "exit $?"
    } > "$LOGS/answer_$step.log" 2>&1
}

for step in "$@"; do
    [ -d "$RUN/$step" ] || { echo "$RUN/$step not found"; exit 2; }
    [ ! -e "$DEV/$step" ] || { echo "$DEV/$step exists; a run is never written over"; exit 2; }
done
i=0
for step in "$@"; do
    while [ "$(jobs -rp | wc -l)" -ge "$AT_ONCE" ]; do sleep 1; done
    port=$((8000 + i)); [ $port -ge 8001 ] && port=$((port + 1))
    echo "$(date -u +%FT%TZ) $step on port $port"
    one "$step" "$port" &
    i=$((i + 1))
done
wait
echo "$(date -u +%FT%TZ) all done"
