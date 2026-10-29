# Training Recipe

Methodological record for how the study trains its adapters, how it picks one adapter per
training run, how a pilot run checks the pipeline, the ladder that makes the study's three
adapters, and the 5,000-pair run of the data-efficiency curve. The recipe trains Qwen3.5-2B, Qwen3.5-9B and Qwen3.5-27B on the training sample of
`methodology/training-set-construction.md` §4. Paths are relative to `fine-tune/`, and the scorer
repository is `../plantuml-recognition-evaluation`. The analysis protocol of the benchmark,
`../plantuml-recognition-evaluation/methodology/analysis-protocol.md`, is called the baseline
protocol below.

## 1. The recipe

One recipe trains all three sizes: the same data, values, length and checkpoint rule at 2B, 9B and
27B. No value in it depends on the model size. `train/recipe.py` holds every value of §1 and of
the probe's rule in §3 as a constant, together with the functions that apply them.

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
weighting makes every answer token and every `<|im_end|>` weigh the same. §5.1 gives how a pair
becomes this sequence.

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
| 5,000 pairs, 2B (§8) | 313 | 626 | 19 | 157, 313, 470, 626 |
| 1,000 pairs, 2B (the pilot, §6) | 63 | 126 | 4 | 32, 63, 95, 126 |

The 5,000 and 1,000 runs are the nested subsets of `methodology/training-set-construction.md` §4.
Their sizes are not multiples of 16, so the last step of each of their epochs holds the remaining
8 pairs. In an epoch of S steps the middle checkpoint follows the epoch's step ⌈S / 2⌉.

## 2. The checkpoint rule

The checkpoint rule picks one of the 4 saved adapters of a run as the run's adapter, from their
scores on the dev set.

### 2.1 Dev scoring

Each of the 4 checkpoints of a run is scored on the 200 diagrams of the dev set (`data/sets/dev.json`;
`methodology/training-set-construction.md` §4), with the settings of the scored runs. The dev
scoring of a checkpoint is a scored run and follows the rules of
`methodology/compute-and-serving.md` §8. The study's server (`infer/server.py`) serves the bf16
base with the checkpoint's adapter attached, not merged. It decodes greedily with `max_tokens`
5,376, `enable_thinking: false` and a deadline of 600 s per answer. It runs on the card model of
the scored runs, with the recurrent-layer implementation of the main runs.

`train/dev_scoring.py` makes and scores the answers in two steps, which may run on two machines.

- `answer` runs on the card. It first runs the adapter check of
  `methodology/compute-and-serving.md` §6 on three training pictures: the rank-1 pair of each of
  the first three cells of `data/examples/train.jsonl`. It makes no answer unless the check passes.
  It then starts the server and runs the benchmark's runner over
  `data/examples/dev_runner_set.json`. That file holds the 200 rows of `dev.json` with the picture
  sizes the runner reads, taken from the dev render manifest.
- `score` runs on a machine with the scorer's Java and Graphviz. Without Graphviz a class diagram
  renders to a notice image, which CSR would count as compiled. It scores CSR and Element F1 under
  scorer v2 (`methodology/scorer-v2.md` §7) against the dev sources of `methodology/scorer-v2.md`
  §5. It refuses to run unless the scorer repository's `evaluation/` and `prompts/` equal tag
  `scorer-v2` and the extractor and renderer JARs have their pinned sha256 (§2.3).

`train/answer_checkpoints.sh` answers the checkpoints of a run side by side on one card, which
saves card time. Each checkpoint has its own server on its own port, and each server answers one
request at a time (`methodology/compute-and-serving.md` §4). How many servers share the card is an
argument, bounded by the card's memory. Before its server stops, `answer` writes the run's
manifest with `infer/manifest.py write --server`, so the manifest records every check of
`methodology/compute-and-serving.md` §5, `server_matches` included. A run whose manifest does not
pass ends with a failure.

```
bash train/answer_checkpoints.sh <size> <run folder> <dev folder> <servers at once> <checkpoint> ...
python3 train/dev_scoring.py score --answers <dev folder>/<checkpoint>/answers \
    --out <dev folder>/<checkpoint>/scores
```

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
and by the renderer JAR of `methodology/training-set-construction.md` §1.1. The renderer is
identified by that JAR's sha256 wherever the file lies. Dev scoring reads a copy at
`../plantuml-1.2025.9.jar` by default and refuses it unless its sha256 is the pinned one.

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

`train/train.py probe --size <size> --out <folder>` runs the probe (§5.3) and writes its losses and
verdict to `probe.json`. The probe passed at each of the three sizes on the study's card, so the
recipe stands as §1 gives it.

## 4. Frozen artifacts

The recipe and the rule are tagged `recipe-v1` in the study repository.

| File | Role | sha256 |
|---|---|---|
| `train/recipe.py` | the recipe: every value of §1 and of the probe's rule in §3, and the functions that apply them | `1b21ee36ba45fc56cdbb9c9cc394aef462de88af6a623de54e3a75d8ce778596` |
| `train/select_checkpoint.py` | the checkpoint rule of §2 | `ba1d50f015882f9051b3e5c37493dac9764f3fa9b230bfd8bc775bd8cf6f74d2` |

The rule runs with these two files, the scorer repository and its set file (by default
`data/examples/dev_runner_set.json`), and it imports nothing else of the study. Both files were fixed before any dev-set output of a fine-tuned checkpoint existed.

The tests lie under `train/tests/`. `test_recipe.py` checks every value, the adapter at the three
sizes, the schedule, the steps, the orders and the probe. `test_select_checkpoint.py` checks the
rule and each refusal of §2.3, on the fixtures in `train/tests/fixtures/`. The tests also import
the data step, the loss and the dev scoring of the training script (`build/examples.py`,
`train/loss.py`, `train/dev_scoring.py`).

```
.venv/bin/python train/tests/test_recipe.py
HF_DEACTIVATE_ASYNC_LOAD=1 .venv/bin/python train/tests/test_recipe.py --live
/usr/bin/python3 train/tests/test_select_checkpoint.py   (Python 3.9.6)
```

The `--live` test loads the Qwen3.5-2B weights; the libraries are those of `requirements-card.txt`
(torch 2.14.1, transformers 5.18.0, PEFT 0.21.2). The fixtures of the rule are two stored scorings
of Qwen3.5-2B on 200 diagrams. On them the rule's metrics equalled the runners' own summaries.
Its interval reproduced the stored paired comparison: CSR +0.015, 95% interval [−0.060, +0.085];
Element F1 micro, full set, −0.0191, [−0.1091, +0.0700].

The training script of §5 and the dev scoring of §2.1 are not part of the tag. Every training run
records the sha256 of each file of its code in its `run.json`.

The data step (§5.1) pins the benchmark prompt of the user turn,
`../plantuml-recognition-evaluation/prompts/zero_shot.txt`, by its sha256 in `build/examples.py`:
`097351ffce6134cb903322718facca5488f978766d46381de03934c8c9f01552`. It writes the checksums of the
files it makes (`train.jsonl`, `loss_curve.jsonl`, `dev_runner_set.json` and `config.json`) to
`data/examples/SHA256SUMS`. That file is tracked, so the checksums of the two `.jsonl` files, which
git ignores, stay in the repository. The training script checks `train.jsonl` against it before
any weights load (§5.3).

The tests of §5 and of the dev scoring of §2.1 lie under `train/tests/` and `build/tests/`:

```
HF_DEACTIVATE_ASYNC_LOAD=1 .venv/bin/python -m pytest -q -p no:cacheprovider train/tests build/tests/test_user_turn.py \
    build/tests/test_examples.py build/tests/test_recurrent_layers.py build/tests/test_card_environment.py \
    infer/tests
HF_DEACTIVATE_ASYNC_LOAD=1 .venv/bin/python train/tests/test_dev_scoring.py --live --adapter <checkpoint folder>
```

## 5. The training script

`train/train.py` trains one run of the recipe on one card. It builds every pair of every step as a
training example (§5.1) and computes the loss of §1.3 on it (§5.2).

### 5.1 The training example

A training example is one pair of the training sample as the model receives it in serving,
followed by the answer it is trained to give. `build/examples.py` builds it.

**The pairs.** `build/examples.py --out data/examples` writes the pairs of `data/sets/train.json`
to `data/examples/train.jsonl`, in list order, each with its picture path and its target text. The
training machine thus needs this file and the pictures, not the corpus. A run takes its pairs from
this file, a subset by its rank bound (`methodology/training-set-construction.md` §7), and never
from the image folder. The picture of a key is
`data/sets/render/train/puml_images_1568/<key without .puml>.png`.

**The user turn.** The prompt part of the example is `build/user_turn.py` `render` of the pair's
picture and the benchmark prompt. This is the function the server calls, so the prompt part is the
user turn of `methodology/compute-and-serving.md` §4.1, with the generation prompt and thinking off.
The prompt is `prompts/zero_shot.txt` of the benchmark
(`../plantuml-recognition-evaluation/methodology/benchmark-protocol.md` §1). It is read with
`.strip()`, as the benchmark's runner reads it, and checked against its sha256 (§4).

**The target.** The target is the text whose tokens rule 4 of
`methodology/training-set-construction.md` §2 counted. It is the diagram block that the scorer's
`extract_puml` isolates from the source (`build/pool.py` `target_text`). For the 6 training
sources without an `@startuml` … `@enduml` pair in lower case, the block is isolated from the
source less its three header lines. No target is edited. Its token ids come from the processor's
tokenizer at the model's pinned revision, without special tokens. With this tokenizer no target
has more tokens than rule 4 counted, so every target stays within the 5,376-token cap
(`build/tests/test_examples.py`). The tokenizer normalizes text to Unicode NFC, so the one target of
the 20,000 that is not in NFC trains as its NFC form.

**The sequence and its labels.** After the target comes `<|im_end|>` (token 248046), and the
sequence ends there (§1.3). The labels equal the input ids, with -100 on every position of the user
turn. The loss thus covers the target and its `<|im_end|>`, and nothing of the prompt.

**Byte-equality with serving.** Training appends the answer to the served prompt, so the prefix a
model trains on is the prompt it is served, token for token. `build/tests/test_examples.py` checks
this through the server's own request path, for one pair of each cell. The inputs that
`infer/server.py` builds from the runner's request equal the example's prefix in every tensor:
`input_ids`, `attention_mask`, `mm_token_type_ids`, `pixel_values` and `image_grid_thw`. With the
template's newline added after `<|im_end|>`, the example equals the model's own chat template
applied to the two-turn conversation, with the target as the assistant turn. This is the
byte-equality test of `FINETUNE_PLAN.md`, Phase B.

### 5.2 The loss in chunks

`train/loss.py` computes the loss of §1.3 in chunks of 512 answer positions. The hidden states that
predict the answer pass through the output layer one chunk at a time. The logits over the
vocabulary therefore never exist for the whole sequence, which saves memory at the longest
sequences. In float32 the loss and its gradient equal those of the library's own loss, with -100 on
the prompt, up to rounding. In bf16 the chunked gradient is no farther from the float32 gradient
than the library's own (`train/tests/test_loss.py`).

### 5.3 A run

```
python train/train.py run --size <2B|9B|27B> --pairs <20000|5000|1000> --out <run folder>
python train/train.py probe --size <size> --out <folder>
python train/train.py memory --size <size> --out <folder>
```

Before any weights load, the script checks `train/recipe.py` against its sha256 at `recipe-v1`
(§4). It checks `data/examples/train.jsonl` against `data/examples/SHA256SUMS`, and every picture
of the run against its render manifest. It calls the server's two guards
(`methodology/compute-and-serving.md` §3.1, §3.2), so training runs the library versions and the
recurrent-layer implementation of serving. It then writes `run.json`: the recipe's values and
sha256, the code's sha256, the examples' checksums, the model, the card and both guards' results.

The script uses non-reentrant gradient checkpointing at every size, so one configuration serves
the whole ladder. It moves the inputs to the card as the server does, without a change of dtype.
It writes one line per optimizer step to `log.jsonl`, with the loss, the norm before clipping and
the learning rate. It never writes into a folder that holds files.

The script keeps torch's default algorithms and does not enforce deterministic ones. A training
step is therefore not repeatable bit for bit on the card. The backward of torch's Flash Attention
in the full-attention layers is not deterministic. Each run trains once, with one seed.

## 6. The pilot

The pilot is the 1,000-pair run of Qwen3.5-2B in the table of §1.5, made once before the ladder.
It checks that the pipeline runs end to end on the card: the training of §5, the dev scoring of
§2.1 and the rule of §2.2. It fixes no value of the recipe. On the same card, a memory step checks
the 27B under the recipe at the longest training sequences (§6.2).

### 6.1 The run

`train/train.py run --size 2B --pairs 1000` trains on the 1,000-pair subset for the recipe's 126
optimizer steps over 2 epochs. It saves the adapter after steps 32, 63, 95 and 126 (§1.5). The four
checkpoints are answered on the 200 dev diagrams side by side on the card, each after its adapter
check, and scored under scorer v2 (§2.1). `train/select_checkpoint.py --pairs 1000` then applies
the rule of §2.2. The exit criterion is CSR > 0 and Element F1 > 0 on the dev set for the selected
adapter (`FINETUNE_PLAN.md`, Phase B). The selected adapter met the exit criterion. It is the
1,000-pair point of the data-efficiency curve (§8).

### 6.2 Memory at the longest sequences

`train/train.py memory --size 27B --out <folder>` runs one optimizer step of the recipe. Its 16
pairs (§1.2) are the 16 pairs of the 20,000 with the longest training sequences. Their lengths are
computed from the picture sizes of the render manifest and the target tokens, and the script writes
them to `run.json` before any weights load. The longest has 7,132 positions, the tokens of its
prompt, its target and `<|im_end|>`. The card holds the 27B under the recipe at these sequences, so
the one card model of `methodology/compute-and-serving.md` §2 serves the whole ladder.

## 7. The ladder

The ladder is the three runs of the recipe on the 20,000-pair training sample, one at each of
Qwen3.5-2B, Qwen3.5-9B and Qwen3.5-27B. Each run is the first row of the table of §1.5: 2 epochs,
2,500 optimizer steps, and adapters saved after steps 625, 1,250, 1,875 and 2,500. The recipe is
that of §1 at tag `recipe-v1`, unchanged after the probe of §3, and each size trains once (§5.3).
The runs train one after another on the card of `methodology/compute-and-serving.md` §2, in the
order 2B, 9B, 27B. A fault of the pipeline thus shows at the smallest size before a larger run
starts.

```
python train/train.py run --size <2B|9B|27B> --pairs 20000 --out runs/train/<size>_20000
```

The four checkpoints of each run are answered on the 200 dev diagrams and scored under scorer v2,
as §2.1 gives. The four of a run are answered on one pod, by servers equal in everything but the
adapter (§2.3). At the 2B the four servers share the card; at the 9B and the 27B one server
answers at a time. Either way each server answers one request at a time
(`methodology/compute-and-serving.md` §4). The manifests of the 12 dev runs passed every check of
`methodology/compute-and-serving.md` §5, so each is a scored run under §8 of that file.

`train/select_checkpoint.py --pairs 20000` applies the rule of §2.2 to each run. At every size it
took the last checkpoint, step 2,500, because no earlier checkpoint was better. The adapter in
`runs/train/<size>_20000/step_02500` is therefore the fine-tuned model of that size.

A run's folder `runs/train/<size>_20000` holds `run.json`, the step log `log.jsonl`, the four
checkpoints, the optimizer state `state.pt` and the choice with its numbers, `selection.json`. The
dev answers and scores of each checkpoint lie in `runs/dev/<size>_20000/<checkpoint>/`.

## 8. The 5,000-pair run

The 5,000-pair run trains Qwen3.5-2B under the recipe on the 5,000-pair subset, for the
data-efficiency curve of `methodology/analysis-protocol.md` §15. It is the second row of the table
of §1.5: 2 epochs, 626 optimizer steps, 19 warm-up steps, and adapters saved after steps 157, 313,
470 and 626. The recipe is that of §1 at tag `recipe-v1`, as for the ladder. The run therefore
differs from the 2B's ladder run in the number of distinct pairs, and its steps follow from that
number. It trains once, on the card of `methodology/compute-and-serving.md` §2, after the ladder.

```
python train/train.py run --size 2B --pairs 5000 --out runs/train/2B_5000
```

The four checkpoints are answered on the 200 dev diagrams on one pod, side by side. Their servers
are equal in everything but the adapter, and the answers are scored under scorer v2 (§2.1, §2.3).
`train/select_checkpoint.py --pairs 5000` applies the rule of §2.2. It took the last checkpoint,
step 626, because no earlier checkpoint was better. The run's folder `runs/train/2B_5000` holds
the files a ladder run holds (§7), and the dev answers and scores of each checkpoint lie in
`runs/dev/2B_5000/<checkpoint>/`.

The curve's two inner points are the last checkpoint of each run: the pilot's
`runs/train/2B_1000/step_00126` (§6.1) and `runs/train/2B_5000/step_00626`, the one the rule took. Each
answers the 1,000 test diagrams as the main runs do (`methodology/compute-and-serving.md` §10).
Both are scored by the command of `methodology/analysis-protocol.md` §2, under scorer v2 and v1,
as the main runs are.
