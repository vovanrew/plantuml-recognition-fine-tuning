# Fixtures of train/tests

- `configs/<size>/config.json`: `config.json` of Qwen/Qwen3.5-2B, 9B and 27B at the pinned commits
  of `train/models.py` (sha256 `ed1c1723…15b4`, `d0883072…2b05`, `f8d190c5…174c`); the models are
  built from them on the meta device, without weights.
- `checkpoint_scores/{bf16,nf4_mlp}/`: CSR and Element F1 result files of two arms of the precision
  test of `decisions/gpu-precision/` (Qwen3.5-2B zero-shot, 200 test diagrams, scored under scorer
  v2), in the layout `train/dev_scoring.py score` writes. Their stored paired comparison
  (`nf4_mlp` − `bf16`, seed 20260614, 1,000 resamples): CSR +0.015, 95% interval [−0.060, +0.085];
  Element F1 micro, full set, −0.0191, [−0.1091, +0.0700].
