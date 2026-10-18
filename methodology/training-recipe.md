# Training Recipe

Methodological record for how the study trains its adapters and how it picks one adapter per
training run. The recipe trains Qwen3.5-2B, Qwen3.5-9B and Qwen3.5-27B on the training sample of
`methodology/training-set-construction.md` §4. Paths are relative to `fine-tune/`, and the scorer
repository is `../plantuml-recognition-evaluation`. The analysis protocol of the benchmark,
`../plantuml-recognition-evaluation/methodology/analysis-protocol.md`, is called the baseline
protocol below.

## 1. The recipe

One recipe trains all three sizes: the same data, values, length and checkpoint rule at 2B, 9B and
27B. No value in it depends on the model size. `train/recipe.py` holds every value of §1 and §3 as
a constant, together with the functions that apply them.

### 1.1 Adapter

LoRA adapters sit on every linear layer of the language model, on the bf16 base, and nowhere else.

| Element | Value |
|---|---|
| Targets | every `torch.nn.Linear` under `model.language_model.`, of 12 kinds: `gate_proj`, `up_proj`, `down_proj` (MLP, every layer); `q_proj`, `k_proj`, `v_proj`, `o_proj` (full-attention layers); `in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`, `out_proj` (Gated DeltaNet layers) |
| Rank, alpha, scaling | rank 16, `lora_alpha` 32, plain α/r scaling (factor 2.0) |
| Dropout, bias | dropout 0.05; no bias adapters |
| Adapter weights | float32, PEFT's default on a bf16 base |
| Initial values | drawn after `torch.manual_seed(42)`, set before PEFT attaches the adapter |
| Frozen, without adapters | the vision tower with its merger, the embeddings and `lm_head` |

The targets reach PEFT as an explicit list of names, so no layer of the vision tower receives an
adapter. A linear layer of a kind outside the 12 under the language model stops the run. The
library does not load the multi-token-prediction head (`mtp.*`).

| Size | Target modules | Trainable parameters |
|---|---|---|
| Qwen3.5-2B | 186 | 16,819,200 |
| Qwen3.5-9B | 248 | 43,278,336 |
| Qwen3.5-27B | 496 | 116,727,808 |

The counts were taken on the three models built without weights from their pinned configurations
(`train/tests/fixtures/configs/`), with the adapter attached by `train/recipe.py`.

### 1.2 Optimisation

| Element | Value |
|---|---|
| Optimizer | `torch.optim.AdamW`, betas (0.9, 0.999), eps 1e-8, weight decay 0.01, each passed explicitly |
| Learning rate | 1e-4 |
| Schedule | linear warm-up over the first ⌈0.03 × S⌉ optimizer steps of a run of S steps, then constant to the end, with no decay; at step k (from 1) of a warm-up of W steps the rate is 1e-4 × k / W |
| Gradient clipping | global norm 1.0; the norm before clipping is returned for the log of every step |
| Effective batch | 16 pairs per optimizer step, one pair per forward and backward, gradients accumulated over the 16; no padding, no packing |
| Data order | in epoch e (from 0), the run's keys in `data/sets/train.json` order, shuffled by Python's `random.Random(42 + e)`; the same order at every size |

The rate stays constant after the warm-up, so each saved adapter is the end of a complete shorter
run and the four checkpoints of §1.5 are fair candidates. `train/tests/test_recipe.py` pins the
order of both epochs of each run size by a digest, and the digests were the same under Python 3.9.6
and 3.12.

### 1.3 Loss

The loss is the cross-entropy on the answer tokens only, which are the target and its closing
`<|im_end|>`. A training sequence ends at `<|im_end|>`, and the newline that the chat template
writes after it is not part of the sequence. The per-token losses are summed over the 16 pairs of a
step and divided by the number of answer tokens in them (`loss_scale` in `train/recipe.py`). This
weighting makes every answer token and every `<|im_end|>` weigh the same.

Under this weighting each cell of the training sample weighs in proportion to its answer tokens.
For each cell, the share below is the sum over its 2,500 pairs of `target_tokens` plus one
`<|im_end|>` (`data/sets/train.json`), divided by the 6,231,671 answer tokens of one epoch.

| Share of the answer tokens of an epoch | Tier 1 | Tier 2 | Tier 3 | Tier 4 |
|---|---|---|---|---|
| Class | 2.4% | 5.4% | 10.7% | 32.6% |
| Sequence | 3.4% | 6.9% | 11.5% | 27.2% |

Tier 4 together carries 59.8% of the weight and tier 1 carries 5.8%, while each tier holds 25% of
the sampled pairs.

### 1.4 Vision tower

In training the vision tower and its merger run without gradient tracking, so the picture features
trained on are those the server computes. The adapter of §1.1 leaves every vision weight frozen,
and the tower's forward also runs under `torch.no_grad`. In the live test of
`train/tests/test_recipe.py` (Qwen3.5-2B, one pair of the loss-curve split), the vision output of a
training forward equalled the served output bit for bit. With a vision weight left trainable, the
output differed from the served one by a relative 0.1021 without `torch.no_grad` and by 0 under it.

### 1.5 Length and checkpoints

Every size trains for 2 epochs, and the checkpoint rule of §2 picks the adapter within each run.
Every size thus gets the same training and its own choice on its own dev scores. The recipe saves
the adapter every half epoch, and it saves the optimizer and schedule state with the last
checkpoint, so a run can be continued. Each adapter goes to the folder `step_` plus its step in
five digits, inside the run's folder (`step_00625` to `step_02500` for a main run).

| Run | Steps per epoch | Steps | Warm-up steps | Adapter saved after step |
|---|---|---|---|---|
| 20,000 pairs, every size | 1,250 | 2,500 | 75 | 625, 1,250, 1,875, 2,500 |
| 5,000 pairs, 2B (optional) | 313 | 626 | 19 | 157, 313, 470, 626 |
| 1,000 pairs, 2B (optional) | 63 | 126 | 4 | 32, 63, 95, 126 |

The 5,000 and 1,000 runs are the nested subsets of `methodology/training-set-construction.md` §4.
Their sizes are not multiples of 16, so the last step of each of their epochs holds the remaining
8 pairs. In an epoch of S steps the middle checkpoint follows the epoch's step ⌈S / 2⌉.

## 2. The checkpoint rule

The checkpoint rule picks one of the 4 saved adapters of a run as the run's adapter, from their
scores on the dev set.

### 2.1 Dev scoring

Each of the 4 checkpoints of a run is scored on the 200 diagrams of the dev set (`data/sets/dev.json`;
`methodology/training-set-construction.md` §4), with the settings of the scored runs. The study's
server (`infer/server.py`) serves the bf16 base with the checkpoint's adapter attached, not merged.
It decodes greedily with `max_tokens` 5,376, `enable_thinking: false` and a deadline of 600 s per
answer. It runs on the card model of the scored runs, with the recurrent-layer implementation of
the main runs. `train/dev_scoring.py` produces and scores the answers under scorer v2, as frozen
in `methodology/scorer-v2.md` §7.

The rule reads two metrics: CSR, and Element F1 micro over the full set, where a failed answer
scores 0 (baseline protocol, §1).

### 2.2 The rule

The last checkpoint is taken, unless an earlier one is better on the dev set. An earlier checkpoint
is better when both of these hold:

- the 95% interval of earlier minus last excludes zero in the earlier one's favour, on CSR or on
  Element F1;
- its point difference on the other metric is not negative.

The interval is the paired percentile bootstrap of the baseline protocol, §5, over the 200 dev
diagrams. It is computed by the scorer repository's `analysis/bootstrap.py`, with seed 20260614 and
1,000 resamples. If several earlier checkpoints are better, the latest of them is taken. The rule
is the same at every size and for every run.

```
python3 train/select_checkpoint.py --pairs <pairs of the run> --out <run folder>/selection.json \
    <score folder of each checkpoint, in step order>
```

The command runs under the scorer's Python 3.9, without torch. It writes to `selection.json` the
choice, every number the choice rests on, and the sha256 of its inputs, of itself and of
`train/recipe.py`.

### 2.3 What the rule refuses

The command stops without a choice when its inputs do not show the dev scoring of §2.1. It refuses:

- result files not scored by scorer v2, or not over the diagrams of the set file, by default the 200
  dev diagrams of `data/examples/dev_runner_set.json`;
- answers made without an attached adapter, at another cap or deadline, by a failed runner or by
  a server in timing mode;
- answers not made in bf16 on the study's card, NVIDIA RTX PRO 6000 Blackwell Server Edition;
- a server identity that does not name the model, revision, dtype, device, card, library versions,
  recurrent-layer implementation or timing mode;
- checkpoints answered by servers that differ in anything but the adapter;
- checkpoints that are not the run's 4 checkpoints in step order from one run folder, or two
  checkpoints that are the same adapter;
- bootstrap or aggregation code of the scorer repository (`analysis/bootstrap.py`,
  `analysis/aggregate.py`) that differs from tag `scorer-v2`.

Scorer v2 is pinned by its tag, its commit and its extractor JAR (`methodology/scorer-v2.md` §7),
and by the renderer JAR (`methodology/training-set-construction.md` §1.1).

## 3. The probe before the ladder

Before the first ladder run, a probe runs the recipe for 60 optimizer steps at each of the three
sizes on the card. A fault of the recipe thus shows before any ladder run, and a change it causes
applies to all sizes. The probe is the recipe on a run of 60 steps: 2 warm-up steps (5e-5 at step
1), then 1e-4 for the other 58. It trains on the first 960 pairs of the epoch-0 order of the
20,000-pair run (`probe_batches` in `train/recipe.py`), and its adapter is discarded.

The probe logs the loss and the norm before clipping at every step. It fails on a non-finite loss,
or when the mean loss of its last 10 steps exceeds that of its first 10 (`probe_verdict`). A failed
probe stops the ladder, and any change it brings to the recipe applies to all three sizes. After
ladder training starts, no outcome of a run changes the recipe.

## 4. Frozen artifacts

The recipe and the rule are tagged `recipe-v1` in the study repository.

| File | Role | sha256 |
|---|---|---|
| `train/recipe.py` | the recipe: every value of §1 and §3, and the functions that apply them | `1b21ee36ba45fc56cdbb9c9cc394aef462de88af6a623de54e3a75d8ce778596` |
| `train/select_checkpoint.py` | the checkpoint rule of §2 | `ba1d50f015882f9051b3e5c37493dac9764f3fa9b230bfd8bc775bd8cf6f74d2` |

The rule runs with these two files, the scorer repository and its set file (by default
`data/examples/dev_runner_set.json`), and it imports nothing else of the study. Both files were fixed before any dev-set output of a fine-tuned checkpoint existed.

The tests lie under `train/tests/`. `test_recipe.py` checks every value, the adapter at the three
sizes, the schedule, the steps, the orders and the probe. `test_select_checkpoint.py` checks the
rule and each refusal of §2.3, on the fixtures in `train/tests/fixtures/`. The tests also import
the data step, the loss and the dev scoring of the training script (`build/examples.py`,
`train/loss.py`, `train/dev_scoring.py`).

```
.venv/bin/python train/tests/test_recipe.py                                    12 of 12 passed
HF_DEACTIVATE_ASYNC_LOAD=1 .venv/bin/python train/tests/test_recipe.py --live  13 of 13 passed
/usr/bin/python3 train/tests/test_select_checkpoint.py   (Python 3.9.6)        14 of 14 passed
```

The `--live` test loads the Qwen3.5-2B weights; the libraries are those of `requirements-card.txt`
(torch 2.14.1, transformers 5.18.0, PEFT 0.21.2). The fixtures of the rule are two stored scorings
of Qwen3.5-2B on 200 diagrams. On them the rule's metrics equalled the runners' own summaries.
Its interval reproduced the stored paired comparison: CSR +0.015, 95% interval [−0.060, +0.085];
Element F1 micro, full set, −0.0191, [−0.1091, +0.0700].
