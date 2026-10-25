#!/bin/bash
# On the card: start a further queue of runs once the queue before it has ended, on a newer commit
# (decisions/data-efficiency-runs/implement.md). Waits until the last line of the earlier queue's
# log is "all done", then moves the checkout to the commit of the pack (fetched from the pack's
# git bundle, the study's tags with it), checks the freeze tag (train/study_bundle.sh check), and
# starts train/ladder_queue.sh on the runs given, with its own log. The earlier queue's runs were
# answered from the earlier commit, and nothing of the checkout changes while it runs.
#
#   nohup bash <work>/pack_<tag>/ladder_chain.sh <work> <tag> <earlier queue log> <run> ... &
#     <work>/pack_<tag>/        study.bundle and commit.txt (train/ladder_session.sh chain puts them)
#     <earlier queue log>       its name under <work>/results/logs/, e.g. queue.log
#     <run>                     <size>:<pairs>, as train/ladder_queue.sh takes them
# Its log: <work>/results/logs/chain_<tag>.log; the new queue's: results/logs/queue_<tag>.log.
# Ends 0 once the new queue is started, 1 when the earlier queue stopped (nothing is started),
# 2 when the checkout or the tag check failed. Checks every CHAIN_SECONDS (60).
set -u
WORK=${1:?work folder}; TAG=${2:?tag}; PREV=${3:?earlier queue log}; shift 3
[ $# -gt 0 ] || { echo "no run given"; exit 2; }
PACK=$WORK/pack_$TAG
LOG=$WORK/results/logs/chain_$TAG.log
mkdir -p "$WORK/results/logs"
say() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }

say "chain $TAG: waiting for $PREV to end; then $*"
while true; do
    last=$(tail -n 1 "$WORK/results/logs/$PREV" 2>/dev/null || true)
    case $last in
        *"all done") break ;;
        *"stopped:"*) say "chain $TAG ends: $PREV stopped; nothing started"; exit 1 ;;
    esac
    sleep "${CHAIN_SECONDS:-60}"
done
commit=$(cat "$PACK/commit.txt")
cd "$WORK/fine-tune" || { say "chain $TAG ends: no checkout"; exit 2; }
if ! { git fetch -q "$PACK/study.bundle" HEAD && git fetch -q "$PACK/study.bundle" 'refs/tags/*:refs/tags/*' \
       && git checkout -q "$commit" && [ "$(git rev-parse HEAD)" = "$commit" ] \
       && bash train/study_bundle.sh check . >> "$LOG" 2>&1; }; then
    say "chain $TAG ends: the checkout of $commit failed"; exit 2
fi
say "checkout $commit"
QUEUE_LOG=queue_$TAG.log nohup bash train/ladder_queue.sh "$@" < /dev/null > "$WORK/results/logs/queue_$TAG.out" 2>&1 &
say "chain $TAG ends: queue started (pid $!, log results/logs/queue_$TAG.log)"
