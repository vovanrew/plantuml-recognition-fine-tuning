#!/bin/zsh
# A training session on the rented card, driven from this Mac over SSH, one step per call
# (decisions/pilot-run/card_session_plan.md). Nothing here creates, stops or deletes the pod: that
# is done through Runpod on the author's word.
#
#   card_session.sh pack                          on the Mac, before the pod exists
#   POD_HOST=<ip> POD_PORT=<port> card_session.sh <step>
#
# Steps, in the plan's order:
#   pack        refuse unless the session's code is committed; pack the commit as a git bundle, the
#               files git does not hold (data/examples/*.jsonl, the pictures the session reads) and
#               the benchmark's evaluation/ and prompts/; checksums
#   push        copy the pack to the pod and check it
#   setup       on the pod: the checkout, the data, the pinned environment, the card's record
#   download    the 2B now, the 9B and 27B in the background;  downloads: how far they are
#   probe <s>   the recipe's probe at size <s> (2B, 9B, 27B)
#   memory      one step of the 27B over the 16 longest training sequences
#   pilot       the 2B on the 1,000-pair run (126 steps, four checkpoints)
#   answer      the 200 dev diagrams answered by each checkpoint in turn, one server per checkpoint,
#               the run manifest written beside each
#   fetch       copy runs/ and the session's records to this Mac
#   run <cmd>   any command on the pod, from the checkout
# After fetch, on the Mac:  score and select (card_session_plan.md, "After the session").
set -eu
FT=${0:A:h:h}
U=${FT:h}
LOCAL=$FT/.local/card_session
KEY=${POD_KEY:-$HOME/.ssh/id_ed25519_runpod}
WORK=${POD_WORK:-/workspace/pilot}
PILOT=runs/train/2B_1000
DEV=runs/dev/2B_1000
step=${1:?step}; shift

if [[ $step == pack ]]; then
    mkdir -p $LOCAL
    changed=$(git -C $FT status --porcelain -- train build infer requirements-card.txt data/examples data/sets)
    if [[ -n $changed ]]; then
        echo "commit these first; the pod runs a checkout of a commit:"; echo $changed; exit 1
    fi
    git -C $FT rev-parse HEAD > $LOCAL/commit.txt
    git -C $FT bundle create $LOCAL/study.bundle HEAD 2>/dev/null
    cd $FT
    HF_HOME=$FT/.local/hf HF_HUB_OFFLINE=1 TRANSFORMERS_VERBOSITY=error $FT/.venv/bin/python - $LOCAL/data_files.txt <<'EOF'
import sys
sys.path.insert(0, "train"); sys.path.insert(0, "build")
import train as T
from transformers import AutoProcessor
from models import SIZES
rows = T.load_rows()
by_key = {r["key"]: r for r in rows}
keys20 = [r["key"] for r in T.R.run_rows(rows, 20000)]
pilot, _ = T.run_plan(rows, 1000)
probe = [k for b in T.R.probe_batches(keys20) for k in b]
processor = AutoProcessor.from_pretrained(SIZES["2B"][0], revision=SIZES["2B"][1])
lengths = T.sequence_lengths(processor, T.E.prompt_text(), rows)
longest = sorted(lengths, key=lambda k: (-lengths[k], k))[:T.MEMORY_PAIRS]
files = sorted({by_key[k]["picture"] for k in set(pilot) | set(probe) | set(longest)})
files += [T.E.picture_path("dev", r["key"]) for r in T.E.set_rows("dev")]
files += ["data/examples/train.jsonl", "data/examples/loss_curve.jsonl"]
open(sys.argv[1], "w").write("\n".join(files) + "\n")
print(len(files), "data files")
EOF
    (cd $FT && tar -cf $LOCAL/data.tar -T $LOCAL/data_files.txt)
    (cd $U/plantuml-recognition-evaluation && tar -cf $LOCAL/instrument.tar evaluation prompts)
    cp $FT/train/pod_setup.sh $LOCAL/
    (cd $LOCAL && shasum -a 256 commit.txt study.bundle data.tar instrument.tar pod_setup.sh > SHA256SUMS)
    ls -la $LOCAL; exit 0
fi

SSH_OPTS=(-i $KEY -p ${POD_PORT:?set POD_PORT} -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new
          -o UserKnownHostsFile=$LOCAL/known_hosts -o ServerAliveInterval=30)
POD=root@${POD_HOST:?set POD_HOST}
ENV="export HF_HOME=$WORK/hf PYTHONUNBUFFERED=1; cd $WORK/fine-tune"
PY=$WORK/venv/bin/python
remote() { ssh $SSH_OPTS $POD "$ENV; $1" }
logged() { remote "$1 2>&1 | tee -a $WORK/results/logs/$2.log; exit \${PIPESTATUS[0]}" }

case $step in
  push)
    ssh $SSH_OPTS $POD "mkdir -p $WORK/pack"
    scp -q -i $KEY -P $POD_PORT -o IdentitiesOnly=yes -o UserKnownHostsFile=$LOCAL/known_hosts \
        $LOCAL/{commit.txt,study.bundle,data.tar,instrument.tar,pod_setup.sh,SHA256SUMS} $POD:$WORK/pack/
    ssh $SSH_OPTS $POD "cd $WORK/pack && sha256sum --quiet -c SHA256SUMS && echo pack ok" ;;
  setup)     ssh $SSH_OPTS $POD "bash $WORK/pack/pod_setup.sh $WORK 2>&1 | tee $WORK/pack/setup.log" ;;
  download)
    logged "$PY train/download_models.py $WORK/results/downloads_2B.json 2B" download_2B
    remote "nohup $PY train/download_models.py $WORK/results/downloads_9B_27B.json 9B 27B > $WORK/results/logs/download.log 2>&1 &" ;;
  downloads) remote "tail -n 3 $WORK/results/logs/download.log; du -sh $WORK/hf; df -h $WORK | tail -1" ;;
  probe)     logged "$PY train/train.py probe --size ${1:?size} --out runs/probe/$1" probe_$1 ;;
  memory)    logged "$PY train/train.py memory --size 27B --out runs/memory/27B" memory_27B ;;
  pilot)     logged "$PY train/train.py run --size 2B --pairs 1000 --out $PILOT" pilot ;;
  answer)
    for s in 00032 00063 00095 00126; do
      logged "$PY train/dev_scoring.py answer --size 2B --adapter $WORK/fine-tune/$PILOT/step_$s --out $DEV/step_$s --timeout 600 && $PY infer/manifest.py write $DEV/step_$s/answers" answer_$s
    done ;;
  fetch)
    mkdir -p $FT/runs $LOCAL/results
    rsync -a -e "ssh ${SSH_OPTS[*]}" $POD:$WORK/fine-tune/runs/ $FT/runs/
    rsync -a -e "ssh ${SSH_OPTS[*]}" $POD:$WORK/results/ $LOCAL/results/
    ls -la $FT/runs/* ;;
  run)       remote "$*" ;;
  *)         echo "unknown step $step"; exit 2 ;;
esac
