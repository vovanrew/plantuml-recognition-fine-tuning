# Compute and Serving

Methodological record for the platform that every run of the study uses. It covers the precision,
the card, and the software with its pinned environment. It also covers the server that answers
every scored request, the manifest written beside each run, and the cost and time of the plan. It
holds for the study's self-served runs what
`../plantuml-recognition-evaluation/methodology/benchmark-protocol.md` §2 holds for the benchmark's
hosted runs; that file is called the benchmark protocol below. The
inference runner, its image-ingestion check, its retries and its failure records are those of the
benchmark protocol, §2. The decoding settings are those of its §2 and §3: greedy, temperature 0,
`max_tokens` 5,376, `chat_template_kwargs: {"enable_thinking": false}`. Paths are relative to
`fine-tune/`, and the scorer repository is `../plantuml-recognition-evaluation`.

## 1. Precision

### 1.1 The weights

Every model of the study is used in its released bf16 weights, with no quantization. This holds at
2B, 9B and 27B, in training and in serving, and in both conditions. The server loads the base with
`dtype=torch.bfloat16` and no quantization configuration, and every response reports the dtype
(§4.3). bitsandbytes is not installed on the card (§3.1).

### 1.2 The bf16 load against the hosted run

On 2026-10-10 the released bf16 weights of Qwen3.5-2B (revision
`15852e8c16360a2fea060d615a32b45270f8a8fc`) answered 200 test diagrams zero-shot on an Apple M5 Pro
(device `mps`), served by a test server over the same libraries on the reference path of §3.2. The
200 diagrams are `pick(diagrams, 200)` of the scorer repository's `evaluation/infer_runner.py` over
its `data/test_set.json` (sha256 `e233e569…744c`). That is every 5th of the 1,000 in file order, 25 in
each of the 8 type × tier cells.

The load ran torch 2.14.1 and transformers 5.18.0. It decoded greedily and stopped at
`eos_token_id` [248046, 248044]. The unchanged runner was called with `--n 200 --max-tokens 5376
--timeout 3600` and thinking off. The user turn was the picture, a newline and the prompt text. It
answered 200 of 200 diagrams. The reference is the benchmark's stored hosted Qwen3.5-2B run on the
same diagrams, `data/runs/Qwen_Qwen3.5-2B_20260613T153655Z` of the scorer repository, with 197
answers and 3 timeouts.

Both answer sets are scored under scorer v2 at its tag (`methodology/scorer-v2.md` §7). CSR and
Element F1 over the full set count a failed answer as 0.

| Answer set | CSR | Element F1 micro, full set | Element F1 macro, full set |
|---|---|---|---|
| bf16, self-served | 45 of 200 (22.5%) | 0.2840 | 0.1991 |
| hosted Qwen3.5-2B | 50 of 200 (25.0%) | 0.2917 | 0.2186 |

The difference uses the paired percentile bootstrap of
`../plantuml-recognition-evaluation/methodology/analysis-protocol.md` §5 (`analysis/bootstrap.py`,
seed 20260614, 1,000 resamples, 95% interval). CSR is a share of the 200 diagrams.

| Difference | CSR (share) | Element F1 micro, full set | Compiled in both / bf16 only / hosted only | Identical answers |
|---|---|---|---|---|
| bf16 − hosted | −0.025 [−0.070, +0.020] | −0.0077 [−0.0690, +0.0532] | 36 / 9 / 14 | 50 of 197 |

Neither interval excludes zero.

## 2. The card

### 2.1 Card and provider

Every run of the study runs on one card model, the NVIDIA RTX PRO 6000 Blackwell Server Edition, a
96 GB card rented in Runpod's Secure Cloud. This holds for training and serving, at every size and
in both conditions. The server reports the card model in every response, and the manifest fails a
run that mixes two (§5). On 2026-10-12 one such card, in data centre US-NE-1, reported 101.976 GB
(GB = 10⁹ bytes). The card measurements of §2.2, §2.3, §3.2 and §7 were made on it that day, with
Qwen3.5-2B at revision `15852e8c16360a2fea060d615a32b45270f8a8fc`, Qwen3.5-9B at
`c202236235762e1c871ad0ccb60c8ee5ba337b9a` and Qwen3.5-27B at
`fc05daec18b0a78c049392ed2e771dde82bdf654`.

### 2.2 Memory of the 27B training step

One AdamW optimizer step of Qwen3.5-27B was measured in bf16, with LoRA adapters and gradient
checkpointing. It ran on one synthetic pair of 7,874 tokens, the longest prompt of the test set with
an answer at the cap. Three ways of computing the loss were measured, first on the library's
reference path and then with the kernels of §3.2.

| Loss computed over | Peak allocated, GB | Peak reserved, GB | Step, s | Peak allocated with the kernels, GB | Step with the kernels, s |
|---|---|---|---|---|---|
| answer positions, in chunks | 68.557 | 72.029 | 27.0 | 65.858 | 40.1 (first step of the process, kernels compiled) |
| answer positions at once | 77.766 | 85.327 | 26.1 | 77.764 | 7.9 |
| every position | 85.124 | 94.009 | 26.1 | 85.122 | 8.0 |

The largest peak reserved, 94.009 GB, is below the card's 101.976 GB. The 27B thus trains in bf16 on
one card at the longest sequence, under each of the three computations.

### 2.3 Ten training steps of the 2B

Ten training steps of Qwen3.5-2B with pictures ran in transformers and PEFT on the same card, on the
reference path. The mean loss of the pairs fell from 0.93225 to 0.55623, and the saved adapter,
loaded on the base again, gave 0.55623. A step took 0.23 to 0.61 s, and peak memory was 6.139 GB.

## 3. Software

### 3.1 Libraries and the pinned environment

Training and serving run Hugging Face transformers 5.18.0 and PEFT 0.21.2 on torch 2.14.1, under
Python 3.12. The training script and the server are the study's own (`methodology/training-recipe.md`;
§4 below).

`requirements-card.txt` is the environment of every run on the card. It pins 61 packages with `==`.
They are the packages that ran on the card on 2026-10-12 after flash-linear-attention was installed,
Triton 3.8.0 and the CUDA libraries included. Its 61 pins equal the freeze of that environment, with
0 differing. It is installed without dependency resolution into a fresh virtual environment of the
container image's Python:

```
python3 -m venv <work>/venv && <work>/venv/bin/python -m pip install --no-deps -r requirements-card.txt
```

The file's header names the container image, `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`
(whose own torch is not used), and host CUDA 13.0 or later. The server's identity does not carry the
image. causal-conv1d, bitsandbytes and kernels are not installed.

`build/card_environment.py` refuses a process whose installed versions differ from the file. On CUDA
its `require(device)` checks every pin. Elsewhere it checks the 11 core packages, whose code loads
the model, renders the prompt or prepares the picture. They are torch, torchvision, transformers,
peft, accelerate, tokenizers, safetensors, huggingface-hub, pillow, numpy and jinja2. The server
calls it before any weights load and puts the result in its identity (§4.3). With the card's
versions stood in on the Mac, the server refused torch 2.15.0, transformers 5.19.0, peft 0.22.0 and
nvidia-cublas 13.1.1.4, each before any weights loaded.

### 3.2 The recurrent layers

The recurrent layers of Qwen3.5, its Gated DeltaNet layers, run on the kernels of
flash-linear-attention 0.5.2, with fla-core 0.5.2, einops 0.8.2 and Triton 3.8.0. This holds in
training and in serving, at every size and in both conditions. causal-conv1d is not installed, so
its two functions stay on the library's reference PyTorch path.

transformers 5.18.0 picks the implementation itself, once per function, when the model code is
imported. It uses `chunk_gated_delta_rule` and `fused_recurrent_gated_delta_rule` of
`fla.ops.gated_delta_rule` if that module imports, and its own reference code otherwise.
`build/recurrent_layers.py` makes the same lookup (`active()`). Its `require(choice, device)` stops a
process when:

- the active implementation is not the one asked for;
- the kernels are asked for without CUDA, or with any of the four packages at another version;
- causal-conv1d is installed;
- the reference path is asked for on CUDA by a process that does not declare itself a diagnostic
  (`diagnostic=True`, recorded in the result). The server cannot declare it.

The server calls `require` before any weights load and states the active implementation in its
identity. The Mac has no CUDA, so local checks there run the reference path and say so, and no
scored run is made there.

On 2026-10-12 the card checks ran first on the reference path. flash-linear-attention was then
installed on the same machine, with fla-core 0.5.2 and einops 0.8.2. Under the kernels, 7 of 8
answers of 256 tokens were identical to the reference path's, at the 2B and at the 27B. The one that
differed did so from token 100 at the 2B and from token 51 at the 27B (positions counted from 1).
For scale, the same code on the Mac and on the card gave 6 of 8 identical answers over the first 48
tokens. That was at the 2B on the reference path, and the other two differed from tokens 36 and 38.
The 9B was not run with the kernels. Memory and speed are in §2.2 and §7.

## 4. The server

`infer/server.py` is a thin OpenAI-compatible server over transformers and PEFT. It serves one model
in bf16 on one device. The zero-shot condition is the base model. The fine-tuned condition is the
same base with the adapter attached as unmerged LoRA layers (`--adapter`, with a `--served-name`
that differs from the base model's). A process answers one request at a time: `make_http_server`
builds a single-threaded `HTTPServer`, and the next request waits.

### 4.1 What it answers

The server answers `POST /v1/chat/completions` as the runner sends it. The request holds one user
message with one text part and at most one picture, as a base64 `data:` URL. It may set
`temperature` 0, `max_tokens` (or `max_completion_tokens`) and `chat_template_kwargs:
{"enable_thinking": false}`. `build/user_turn.py` builds the user turn: the picture, one newline and
the prompt text, rendered by the model's own chat template with thinking off. The arrangement is the
`prompt-arrangement` item's (`FINETUNE_PLAN.md` §10). For `prompts/zero_shot.txt` and one picture,
the rendered turn has sha256 `44a5cc713a568d729f5839c9f2903ef09327b890b8c9f4b756c765e6585141e8` at
the three sizes. A text-only request, the runner's ingestion baseline, renders to 93 tokens.

Decoding is greedy and stops at `eos_token_id` [248046, 248044] at every size. `finish_reason` is
`stop` when the last token is one of these and `length` otherwise. `usage` gives the prompt and
completion tokens. `GET /` returns the server's identity.

### 4.2 What it refuses

The server does not start when:

- the revision is not a full 40-character commit hash;
- an adapter is given and the served name equals the base model's name;
- the installed versions differ from `requirements-card.txt` (§3.1);
- the process would run the other implementation of the recurrent layers (§3.2);
- the loaded adapter has no LoRA layers, or any of them is merged.

The first four are checked before any weights load. A request is refused with HTTP 400, never
adjusted, when it:

- names a model other than the served one;
- sets a temperature other than 0, or `chat_template_kwargs` other than `{"enable_thinking": false}`;
- carries a field outside `model`, `messages`, `max_tokens`, `max_completion_tokens`, `temperature`,
  `chat_template_kwargs` and `min_tokens`;
- sets `min_tokens` on a server not started with `--timing`;
- holds a user turn outside the domain of `build/user_turn.py`, such as two pictures, two text parts,
  no text, a second message or another role.

`--timing` accepts `min_tokens`, which forces an answer to a length so that it can be timed at the
cap. The identity shows the mode, and the manifest fails a run made in it.

### 4.3 What every response records

Every response carries `x_served_by`, the identity of what produced it. `GET /` returns the same
identity without the per-request fields.

| Field | Content |
|---|---|
| `served_name`, `model`, `revision`, `dtype` | the name served, the model, its commit, `bfloat16` |
| `adapter` | in the fine-tuned condition: its path, the sha256 of `adapter_model.safetensors` and `adapter_config.json`, the numbers of LoRA layers and of merged ones, and `attached`: `not merged` |
| `decoding`, `eos_token_id`, `chat_template_kwargs` | `greedy`, [248046, 248044], `{"enable_thinking": false}` |
| `user_turn` | the sha256 of `build/user_turn.py` and the frozen rendered-turn hash |
| `recurrent_layers` | the result of §3.2: the implementation asked for and the active one, the diagnostic flag, the versions of the five packages |
| `environment` | the result of §3.1: the sha256 of `requirements-card.txt`, whether all pins or the core ones were checked, how many, and the core versions |
| `files` | the sha256 of the five serving files: `infer/server.py`, `build/user_turn.py`, `build/recurrent_layers.py`, `build/card_environment.py`, `requirements-card.txt` |
| `study_commit` | the study repository's commit, and whether the serving files equal it |
| `timing` | whether `--timing` is on |
| `device`, `gpu`, `driver`, `versions`, `optional_kernels` | the device; on CUDA the card model and driver; the versions of Python, torch and its CUDA build, transformers, PEFT, tokenizers and Pillow; those of flash-linear-attention, causal-conv1d and kernels |
| per request | `rendered_sha256`, `min_tokens`, `generate_seconds`, `first_token_seconds` |

With the 2B adapter of the live test (§10), the identity read 186 LoRA layers and 0 merged.

On the Mac, with the 2B on the reference path, the frozen server reproduced the stored `bf16` answers
of the precision test (§1.2) on 4 of 4 test diagrams. The answers matched character for character,
with the same `usage`. The card measurements of 2026-10-12 ran the server's first version (sha256
`8528698153a5d3ba24fca1eac5724249ea4c1c529b0e84fbe4b4a3f1db92f6a2`). That version lacks the guards
of §3.1 and §3.2 and the check of the adapter's LoRA layers. Its code that loads the model, renders
the turn and decodes is unchanged in the frozen version.

## 5. The run manifest

`infer/manifest.py` writes `manifest.json` into a run directory of the runner, from `run_meta.json`
and every stored response.

```
.venv/bin/python infer/manifest.py write <run dir> --server http://127.0.0.1:8000
.venv/bin/python infer/manifest.py check <run dir>
```

`write` summarises the run and records each check below with its result. With `--server` it also
compares the live server's identity. `check` recomputes the summary and the checks and compares them
with `manifest.json`. It exits with 1 on a failed check or a difference. The manifest records the
sha256 of `run_meta.json`, of the prompt and of `infer/manifest.py`. It also records the served files
and the study commit from the identity, the freeze tag, and the counts of outcomes, stop reasons,
rendered-turn hashes and configurations.

| Check | Passes when |
|---|---|
| `one_configuration` | every answered cell carries the same identity, per-request fields removed, and no answer lacks one |
| `server_matches` | with `--server`: the live server's identity is that one |
| `frozen_files` | the five serving files the server ran equal those of the tag `serving-v3` |
| `environment` | the server checked every pin of `requirements-card.txt` |
| `recurrent_layers` | the active implementation was flash-linear-attention |
| `not_timing_mode` | the server ran without `--timing` |
| `user_turn_file` | the identity's hash of `build/user_turn.py` is the file's hash |
| `rendered_turn` | every answered cell carries the expected rendered-turn hash: the frozen one, or the one given with `--rendered-sha256` for a run with another prompt, which the manifest records |
| `decoding` | `run_meta.json` sets temperature 0, `max_tokens` 5,376 and thinking off |
| `deadline` | `run_meta.json` sets the deadline to 600 s |
| `every_cell_answered` | no cell ended in a failure record of the runner (`timeout`, `http_error`, `network_error`, `image_dropped`, `bad_response`) |
| `complete_run` | the run directory holds one stored cell, answer or failure record, for each of the `n_cells` of `run_meta.json` |
| `served_name` | the runner asked for the name the server serves |

`one_configuration` compares the whole identity. Cells answered from two study commits, two drivers
or two adapter paths therefore fail it, even when the serving files and the adapter's hashes are the
same. A run on the Mac fails `environment` and `recurrent_layers`, since the Mac checks only the
core packages and runs the reference path. `frozen_files` reads the tag with `git show`, so
`manifest.py` runs in a checkout of the study repository that holds the tag. The runner rewrites
`n_cells` on every start, resumes included, so `complete_run` measures a run against the selection
of its last start (§8).

## 6. The adapter check

`infer/adapter_check.py` checks that an adapter changes the output. It loads the base and the
adapter through the server's own start-up, with the same guards: bf16, the adapter attached as
unmerged LoRA layers. It answers the same pictures twice from that one loaded model. The first
answer is with the adapter. The second is with the adapter switched off by `disable_adapter()`.
The check passes when the logits of the
first answer token differ on every picture and the greedy answers differ on at least one. The
pictures are training pictures, so no fine-tuned answer to a test diagram exists before its scored
run.

```
.venv/bin/python infer/adapter_check.py --model Qwen/Qwen3.5-27B --revision <commit> --adapter <dir> \
    --pictures <png> [<png> ...] --out <result.json>
```

By default it runs the flash-linear-attention implementation, 128 answer tokens and the prompt
`../plantuml-recognition-evaluation/prompts/zero_shot.txt`. The result file records the server's
identity, the prompt's sha256, the comparison on every picture and the verdict.

On the Mac with the 2B on the reference path, an adapter from ten training steps changed the answers
on 3 of 3 training pictures (48 answer tokens). The first-logit changes were 8.15625, 7.320312 and
7.625. A copy of that adapter with all 186 `lora_B` matrices set to zero stayed attached and
unmerged. It gave changes of 0.0, 0.0 and 0.0 and failed the check with exit code 1.

## 7. Speed and the deadline

The speeds were measured on the card on 2026-10-12, through the server, one request at a time.
Decoding speed comes from 8 test diagrams, one per type × tier cell, with answers limited to 256
tokens. The answer at the cap is one answer forced to 5,376 tokens by `--timing`, on the longest
prompt of the test set (2,497 tokens).

| Size | Decode, tokens/s per stream (median), reference path | Same, flash-linear-attention | Answer at the cap, wall-clock s, reference path | Same, flash-linear-attention |
|---|---|---|---|---|
| Qwen3.5-2B | 83.39 | 87.67 | 65.804 | 61.389 |
| Qwen3.5-9B | 59.92 | not run | 90.847 | not run |
| Qwen3.5-27B | 21.86 | 23.00 | 252.287 | 237.439 |

The kernels decoded 5.1% faster at the 2B and 5.2% faster at the 27B.

The runner's deadline is 600 s per answer (`--timeout 600`), at every size and in both conditions.
It is 600 ÷ 252.287 = 2.38 times the slowest answer at the cap. The manifest's `deadline` check reads
it from `run_meta.json`, and `every_cell_answered` fails a run with a timed-out cell (§5).

## 8. Scored runs

A scored run is a run whose answers the study scores: the dev scoring of a checkpoint
(`methodology/training-recipe.md` §2.1) and every run on the test set. Every scored run follows
these rules.

1. It runs on the card of §2.1 in the environment of §3.1, never on the Mac.
2. It is served, and its manifest written, from one checkout of one commit of the study
   repository whose five serving files equal those of the tag `serving-v3`, with the tag present in
   that checkout, since `frozen_files` reads it. The adapter stays at one path, and the run stays on
   one pod.
3. Before a scored run with an adapter, `infer/adapter_check.py` passes for that adapter on the card.
4. Its command fixes the selection before the run starts and states the `n_cells` the run must
   reach. The selection is all 1,000 test diagrams (the runner's `--n 1000`) or the dev set's key
   list.
5. An interruption is resumed with the same command plus `--run-dir <its run directory>`, on the
   same pod and checkout. The runner then re-attempts the missing and failed cells (benchmark
   protocol, §2).
6. A run that cannot be finished on the same pod is made again from the start in a new run
   directory. This applies when the pod is lost or a driver or the checkout changed. The earlier
   part is kept and is not scored.
7. The manifest's `n_cells` is compared with the one stated beforehand.

On 2026-10-18 the rules were run on the Mac, with the 2B on the reference path. One checkout of
`serving-v3` served 4 test diagrams (`n_cells` 4) and wrote every manifest.

| Case | Stored of `n_cells` | Checks failed |
|---|---|---|
| runner killed after its first stored cell, then resumed under a new server process from the same checkout | 4 of 4 | `environment` and `recurrent_layers`, the two that every Mac run fails |
| the interrupted part, kept as it was | 1 of 4 | `complete_run`, and the two of the Mac |
| the run made again in a new run directory | 4 of 4 | the two of the Mac |
| the interrupted part resumed under a server from another checkout | 4 of 4 | `one_configuration`, and with it `frozen_files`, `server_matches`, `not_timing_mode`, `user_turn_file` and `served_name`, plus the two of the Mac |

The resumed answers equalled those of the run made again, in text and in `usage`, on 4 of 4
diagrams.

## 9. Cost and time

The estimate is for the configuration of this entry: flash-linear-attention, one request at a time,
on one card. Card time is priced at USD 2.49 per hour, the Secure Cloud price of the card read on
2026-10-12. Storage is not in the figures.

| Plan | Epochs | Training, GPU-h | Inference, GPU-h | Total, GPU-h | Days of one card | USD |
|---|---|---|---|---|---|---|
| as written | 1 | 29.2 | 107.3 | 136.5 | 5.7 | 340 |
| as written | 2 | 58.4 | 107.3 | 165.7 | 6.9 | 413 |
| draft cut | 1 | 5.2 | 41.3 | 46.5 | 1.9 | 116 |
| draft cut | 2 | 10.4 | 41.3 | 51.7 | 2.2 | 129 |

The plans are those of 2026-10-12. The plan as written trains the 2B, 9B and 27B on 20,000 pairs,
with the 5,000 and 1,000 subsets at the 2B. It runs the few-shot control and two robustness variants
at every size. The draft cut trains the 2B and 27B on 5,000 pairs, with a 1,000 subset at the 2B, and
runs the control and one robustness variant at the 2B only. The few-shot control is in the figures
although the plan dropped the controls on 2026-10-18 (`FINETUNE_PLAN.md` §6). The recipe trains for 2 epochs (`methodology/training-recipe.md`
§1.5). Inference of the plan as written is 34.2 GPU-hours at the 2B, 30.5 at the 9B and 42.6 at the
27B.

What is measured:

- inference: the decode rate per stream at each size, on the reference path (§7);
- training: the 2B's time per token, the median of steps 2 to 10 of §2.3, and the 27B's step at
  7,874 tokens with the kernels, every position (§2.2).

What is extrapolated or assumed:

- training time is proportional to sequence length, at a mean pair length;
- the 9B's training time per token lies between the 2B's and the 27B's in proportion to its
  parameter count;
- the 2B's training rate is that of the reference path, since the 2B did not train with the
  kernels;
- inference uses the reference path's rates, about 5% slower than the kernels' (§7);
- the completion tokens of the zero-shot runs are those of the stored hosted runs, and those of the
  fine-tuned runs and of dev scoring are assumed;
- each training run has four dev scorings of 200 diagrams, and each size has half a main run of
  headroom for a failed run (the plan's §8 counts one failed run per size).

The 9B was not run with the kernels, so no row rests on a 9B measurement under them.

## 10. Frozen artifacts

The serving configuration is tagged `serving-v3` in the study repository. The annotated tag object
`537a3b149fe02fc7f48926eeafd4ed7f03dae293` points at commit
`e47bd5e65d56a3a70626543f052e73c376cd211f`. No scored run existed when the tag was set.

| File | Role | sha256 |
|---|---|---|
| `infer/server.py` | the server (§4) | `7cb31590262d47c1f908aa18a1b6abdf0ffcabbf1c62c41059282bb31b72278d` |
| `build/user_turn.py` | the user turn of `prompt-arrangement` (§4.1) | `d075641adac54d3ac37e21e0ae10a57f739e715fb5788b661bbe84e36ba02561` |
| `build/recurrent_layers.py` | the recurrent-layer guard (§3.2) | `77cdb3a192c41e4efe33d60fb54af0afed8213dfd474b5d218512695c8565a75` |
| `build/card_environment.py` | the version guard (§3.1) | `5fe071b5bd209ca0e953be1ec6c8045af7676e0cc4aaad4751c5bda2defb9aa2` |
| `requirements-card.txt` | the pinned environment (§3.1) | `9c07234514f4df497fa682fa5a954e91d40d9241cd7c395767ce6a5009d0a874` |
| `infer/manifest.py` | the run manifest (§5) | `dd71b2d176c5ce17e1ccf8b480a7312e30b31b13530a15264ced90366eb178f3` |
| `infer/adapter_check.py` | the adapter check (§6) | `eb488b029449e070488542b410c36ba45b6e44386f7f0dd727dd9e5de52b8145` |

`serving-v1` (on commit `bf6644f`) and `serving-v2` (on commit `8070e53`) are earlier freezes, and no
scored run used either.

The tests lie under `build/tests/` and `infer/tests/`. The tests without weights stand in the card's
versions and a module in place of `fla.ops.gated_delta_rule`, since the Mac cannot install the CUDA
packages. The live tests load Qwen3.5-2B on the Mac, on the reference path, with an adapter from ten
training steps.

```
.venv/bin/python -m pytest -q build/tests/test_card_environment.py build/tests/test_recurrent_layers.py \
    build/tests/test_user_turn.py infer/tests/test_server.py infer/tests/test_manifest.py \
    infer/tests/test_adapter_check.py                                                     69 passed
HF_DEACTIVATE_ASYNC_LOAD=1 .venv/bin/python infer/tests/test_server.py --live --adapter <dir>        16 of 16 passed
HF_DEACTIVATE_ASYNC_LOAD=1 .venv/bin/python infer/tests/test_adapter_check.py --live --adapter <dir> 6 of 6 passed
```
