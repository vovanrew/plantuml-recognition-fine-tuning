#!/bin/bash
# Prepare a rented card for a training session (decisions/pilot-run/record.md, Part 2): the study
# repository as a checkout of the packed commit, the files it needs that git does not hold, the
# benchmark's runner and prompt, the pinned Python environment, a record of the card.
#
# Run on the card, from the folder train/card_session.sh copied the pack into:
#   bash pod_setup.sh <work folder>
# Layout it makes:  <work>/fine-tune (checkout), <work>/plantuml-recognition-evaluation
# (evaluation/ and prompts/ only), <work>/venv, <work>/hf (model cache), <work>/results.
set -euo pipefail
WORK=${1:?work folder}
PACK=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$WORK/results/logs" "$WORK/hf"

echo "== pack"
(cd "$PACK" && sha256sum --quiet -c SHA256SUMS) && echo "pack files match SHA256SUMS"

echo "== study checkout"
if [ ! -d "$WORK/fine-tune/.git" ]; then
    git clone --quiet "$PACK/study.bundle" "$WORK/fine-tune"
fi
git -C "$WORK/fine-tune" checkout --quiet "$(cat "$PACK/commit.txt")"
git -C "$WORK/fine-tune" rev-parse HEAD | tee "$WORK/results/study_commit.txt"
tag=$(sed -n 's/^FREEZE_TAG = "\(.*\)"/\1/p' "$WORK/fine-tune/infer/manifest.py")
git -C "$WORK/fine-tune" rev-parse -q --verify "refs/tags/$tag" >/dev/null \
    || { echo "tag $tag is not in the checkout; infer/manifest.py's frozen_files reads it"; exit 1; }
echo "tag $tag present"
tar -xf "$PACK/data.tar" -C "$WORK/fine-tune"
mkdir -p "$WORK/plantuml-recognition-evaluation"
tar -xf "$PACK/instrument.tar" -C "$WORK/plantuml-recognition-evaluation"
(cd "$WORK/fine-tune/data/examples" && sha256sum --quiet -c SHA256SUMS) && echo "data/examples match SHA256SUMS"

echo "== card"
nvidia-smi | tee "$WORK/results/nvidia_smi.txt"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv | tee "$WORK/results/card.csv"
{ uname -a; python3 --version; nproc; free -g | head -2; df -h "$WORK"; } | tee "$WORK/results/machine.txt"

echo "== environment"
if [ ! -x "$WORK/venv/bin/python" ]; then
    python3 -m venv "$WORK/venv"
fi
"$WORK/venv/bin/python" -m pip install --quiet --upgrade pip
"$WORK/venv/bin/python" -m pip install --quiet --no-deps -r "$WORK/fine-tune/requirements-card.txt"
"$WORK/venv/bin/python" -m pip freeze > "$WORK/results/pip_freeze.txt"
cd "$WORK/fine-tune"
"$WORK/venv/bin/python" - <<'EOF' | tee "$WORK/results/environment.txt"
import json, sys
sys.path.insert(0, "build")
import card_environment, recurrent_layers, torch
print(json.dumps(card_environment.require("cuda")))
print(json.dumps(recurrent_layers.require(recurrent_layers.FLA, "cuda")))
x = torch.randn(4, 4, device="cuda", dtype=torch.bfloat16)
print("card", torch.cuda.get_device_name(0), "| bf16 matmul ok:", bool(torch.isfinite(x @ x).all()))
EOF
echo "setup done"
