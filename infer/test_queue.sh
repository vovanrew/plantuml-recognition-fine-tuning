#!/bin/bash
# On the card: the runs of infer/test_runs.txt on the 1,000 test diagrams, unattended
# (decisions/main-inference-runs/record.md). Each run is a scored run of
# methodology/compute-and-serving.md §8, made by train/dev_scoring.py `answer` with the test set:
# the adapter check first when the run has an adapter, the study's server, the benchmark's runner
# unchanged with --n 1000 (every row of data/test_set.json), --timeout 600, and the run manifest
# written while the server still runs. Nothing is scored here.
#
# Run from the checkout, under nohup (infer/test_session.sh `start` does):
#   [QUEUE_LOG=<name>] bash infer/test_queue.sh [<id> ...]
# With ids, only those runs of test_runs.txt, still in the file's order.
#
# By size, in the file's order: the model is downloaded at its pinned commit
# (train/download_models.py), then its runs are answered: side by side at the 2B (each server
# about 6 GB, one port each), one at a time at the 9B (no gain side by side) and the 27B (one
# server's weights fill the card). Each server answers one request at a time.
#
# A run's answers go to runs/test/<id>/answers, the benchmark's run format (one <key stem>.json and
# .puml per cell, run_meta.json), with manifest.json beside them; dev_scoring.py's own files
# (answer_manifest.json, adapter_check.json, server.log, runner.log) to runs/test/<id>/. A run that
# ends with a failure is resumed once at once, with the same command: the runner then re-attempts
# its missing and failed cells (rule 5 of §8). A run folder that exists without its .done mark is
# resumed the same way, but only by the checkout that started it (runs/test/<id>/queue_commit.txt).
# Each finished run leaves results/queue/test_<id>.done; a later start skips it. When a run fails
# twice, the queue stops once the other runs of its size have ended, and writes the reason to its log, results/logs/$QUEUE_LOG (default
# test_queue.log), whose last line is "all done" or "stopped: <reason>". The runs' own logs:
# results/logs/test_<id>.log, each ending in "exit <code>".
set -u
WORK=$(cd .. && pwd)
PY=${PY:-$WORK/venv/bin/python}
INSTRUMENT=${INSTRUMENT:-$WORK/plantuml-recognition-evaluation}
TEST_SET=$INSTRUMENT/data/test_set.json
TEST_IMAGES=$INSTRUMENT/data/puml_images_1568
TEST_SET_SHA256=e233e56968f90c7411d7c6b27cdd5c81f801fcc47fd1c7b45d174918a440744c
N_CELLS=1000
LIST=${TEST_RUNS:-infer/test_runs.txt}
LOGS=$WORK/results/logs
MARKS=$WORK/results/queue
QLOG=$LOGS/${QUEUE_LOG:-test_queue.log}
mkdir -p "$LOGS" "$MARKS"

say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$QLOG"; }
stop() { say "stopped: $*"; exit 1; }

sha256() { if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }

# The runs: every line of the list, or the ids asked for, in the list's order.
IDS=(); SIZES=(); ADAPTERS=()
while read -r id size adapter rest; do
    case $id in ''|'#'*) continue ;; esac
    [ -n "$adapter" ] && [ -z "$rest" ] || stop "$LIST: a line is not <id> <size> <adapter>: $id $size $adapter $rest"
    if [ $# -gt 0 ]; then
        keep=0; for want in "$@"; do [ "$want" = "$id" ] && keep=1; done
        [ $keep = 1 ] || continue
    fi
    case $size in 2B|9B|27B) ;; *) stop "$id: unknown size $size" ;; esac
    if [ "$adapter" != - ]; then
        for f in adapter_config.json adapter_model.safetensors; do
            [ -f "$adapter/$f" ] || stop "$id: $adapter/$f not found"
        done
    fi
    IDS+=("$id"); SIZES+=("$size"); ADAPTERS+=("$adapter")
done < "$LIST"
for want in "$@"; do
    found=0; for id in ${IDS[@]+"${IDS[@]}"}; do [ "$id" = "$want" ] && found=1; done
    [ $found = 1 ] || stop "$want is not a run of $LIST"
done
[ ${#IDS[@]} -gt 0 ] || stop "no run to answer"

# The test set and its pictures, as the benchmark holds them.
[ -f "$TEST_SET" ] || stop "$TEST_SET not found"
[ "$(sha256 "$TEST_SET")" = "$TEST_SET_SHA256" ] || stop "$TEST_SET is not the frozen test set (sha256 $TEST_SET_SHA256)"
pictures=$(find "$TEST_IMAGES" -maxdepth 1 -name '*.png' 2>/dev/null | wc -l | tr -d ' ')
[ "$pictures" = "$N_CELLS" ] || stop "$TEST_IMAGES holds $pictures pictures, not $N_CELLS"
COMMIT=$(git rev-parse HEAD) || stop "not a git checkout"

answer_one() {   # answer_one <id> <size> <adapter> <port>: two attempts at most; the log ends in "exit <code>"
    local id=$1 size=$2 adapter=$3 port=$4 out=runs/test/$1 code=1 attempt
    local command=("$PY" train/dev_scoring.py answer --size "$size" --out "$out" --timeout 600
                   --port "$port" --set-file "$TEST_SET" --images "$TEST_IMAGES")
    [ "$adapter" != - ] && command+=(--adapter "$(pwd)/$adapter")
    {
        mkdir -p "$out"
        [ -f "$out/queue_commit.txt" ] || echo "$COMMIT" > "$out/queue_commit.txt"
        for attempt in 1 2; do
            echo "$(date -u +%FT%TZ) attempt $attempt: ${command[*]}"
            "${command[@]}"; code=$?
            [ $code = 0 ] && break
        done
        echo "exit $code"
    } >> "$LOGS/test_$id.log" 2>&1
}

say "test queue: ${IDS[*]} at $COMMIT"
for i in "${!IDS[@]}"; do   # refuse before anything runs: a folder left by another checkout
    id=${IDS[$i]}
    [ -e "$MARKS/test_$id.done" ] && continue
    if [ -e "runs/test/$id" ]; then
        [ -f "runs/test/$id/queue_commit.txt" ] && [ "$(cat "runs/test/$id/queue_commit.txt")" = "$COMMIT" ] \
            || stop "runs/test/$id exists from another checkout (or none recorded); it is not resumed here"
    fi
done

done_sizes=()
for i in "${!IDS[@]}"; do
    s=${SIZES[$i]}
    case " ${done_sizes[*]+${done_sizes[*]}} " in *" $s "*) continue ;; esac
    done_sizes+=("$s")
    if [ -e "$MARKS/download_$s.done" ]; then say "download $s: done before, skipped"
    else
        say "download $s: start"
        "$PY" train/download_models.py "$WORK/results/downloads_$s.json" "$s" >> "$LOGS/download_$s.log" 2>&1 \
            || stop "download $s failed (results/logs/download_$s.log)"
        touch "$MARKS/download_$s.done"; say "download $s: done"
    fi
    at_once=1; [ "$s" = 2B ] && at_once=4
    group=(); port=8000
    for j in "${!IDS[@]}"; do
        [ "${SIZES[$j]}" = "$s" ] || continue
        id=${IDS[$j]}
        if [ -e "$MARKS/test_$id.done" ]; then say "test $id: done before, skipped"; continue; fi
        while [ "$(jobs -rp | wc -l)" -ge "$at_once" ]; do sleep 1; done
        say "test $id: start on port $port"
        answer_one "$id" "$s" "${ADAPTERS[$j]}" "$port" &
        group+=("$id")
        port=$((port + 1)); [ $port = 8001 ] && port=8002   # 8001 is the image's nginx
        [ "$at_once" = 1 ] && port=8000
    done
    wait
    bad=0
    for id in ${group[@]+"${group[@]}"}; do
        last=$(tail -n 1 "$LOGS/test_$id.log" 2>/dev/null)
        if [ "$last" = "exit 0" ]; then touch "$MARKS/test_$id.done"; say "test $id: done"
        else say "test $id: ${last:-no log} (results/logs/test_$id.log)"; bad=1; fi
    done
    [ $bad = 0 ] || stop "a run of the $s failed twice"
done
say "all done"
