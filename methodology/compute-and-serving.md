# Compute and Serving

Methodological record for the platform that every run of the study uses: the precision, the card,
the software, the server that answers every scored request, the manifest written beside each run,
and the rules a scored run follows. It holds for the study's self-served runs what
`../plantuml-recognition-evaluation/methodology/benchmark-protocol.md` §2 holds for the benchmark's
hosted runs; that file is called the benchmark protocol below. The inference runner, its
image-ingestion check, its retries and its failure records are those of the benchmark protocol, §2.
The decoding settings are those of its §2 and §3: greedy, temperature 0, `max_tokens` 5,376,
`chat_template_kwargs: {"enable_thinking": false}`. Paths are relative to `fine-tune/`, and the
scorer repository is `../plantuml-recognition-evaluation`.

## 1. Precision

Every model is used in its released bf16 weights, with no quantization: Qwen3.5-2B, Qwen3.5-9B and
Qwen3.5-27B, in training and in serving, in both conditions. Both conditions thus use the model as
released, and no compression enters the comparison. The server loads the base with
`dtype=torch.bfloat16` and no quantization configuration, and every response reports the dtype
(§4.3).

## 2. The card

Every run uses one card model, the NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB), rented in
Runpod's Secure Cloud: training and serving, at every size, in both conditions. The card holds the
27B in bf16 for training at the longest training sequence, so one card model serves the whole
ladder. The models are Qwen3.5-2B at revision `15852e8c16360a2fea060d615a32b45270f8a8fc`,
Qwen3.5-9B at `c202236235762e1c871ad0ccb60c8ee5ba337b9a` and Qwen3.5-27B at
`fc05daec18b0a78c049392ed2e771dde82bdf654`. The server reports the card model in every response,
and the manifest fails a run that mixes two (§5).

## 3. Software

### 3.1 Libraries and the pinned environment

Training and serving run Hugging Face transformers 5.18.0 and PEFT 0.21.2 on torch 2.14.1, under
Python 3.12. The training script and the server are the study's own (`methodology/training-recipe.md`;
§4 below), so the picture, the prompt and the loss are each a line of the study's code.

`requirements-card.txt` is the environment of every run on the card. It pins all 61 packages of
that environment with `==`, Triton 3.8.0 and the CUDA libraries included, and is installed without
dependency resolution into a fresh virtual environment of the container image's Python:

```
python3 -m venv <work>/venv && <work>/venv/bin/python -m pip install --no-deps -r requirements-card.txt
```

The image is `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` (its own torch is not used), on a
host with CUDA 13.0 or later. causal-conv1d, bitsandbytes and kernels are not installed.

`build/card_environment.py` refuses a process whose installed versions differ from the file. On CUDA
it checks every pin. Elsewhere it checks the 11 core packages whose code loads the model, renders
the prompt or prepares the picture: torch, torchvision, transformers, peft, accelerate, tokenizers,
safetensors, huggingface-hub, pillow, numpy and jinja2. The server calls it before any weights load
and puts the result in its identity (§4.3).

### 3.2 The recurrent layers

The Gated DeltaNet layers of Qwen3.5 run on the kernels of flash-linear-attention 0.5.2, with
fla-core 0.5.2, einops 0.8.2 and Triton 3.8.0, in training and in serving, at every size and in both
conditions. The kernels make training and decoding faster than the library's reference PyTorch
code, and one implementation holds in both conditions. causal-conv1d is not installed, so its two
functions stay on the reference path.

transformers 5.18.0 picks the implementation itself when the model code is imported: it uses
`chunk_gated_delta_rule` and `fused_recurrent_gated_delta_rule` of `fla.ops.gated_delta_rule` if that
module imports, and its own reference code otherwise. `build/recurrent_layers.py` makes the same
lookup (`active()`). Its `require(choice, device)` stops a process when:

- the active implementation is not the one asked for;
- the kernels are asked for without CUDA, or with any of the four packages at another version;
- causal-conv1d is installed;
- the reference path is asked for on CUDA by a process that does not declare itself a diagnostic
  (`diagnostic=True`, recorded in the result). The server cannot declare it.

The server calls `require` before any weights load and states the active implementation in its
identity. The kernels need CUDA, so local checks on a Mac run the reference path and say so; no
scored run is made there.

## 4. The server

`infer/server.py` is a thin OpenAI-compatible server over transformers and PEFT. It serves one model
in bf16 on one device. The zero-shot condition is the base model. The fine-tuned condition is the
same base with the adapter attached as unmerged LoRA layers (`--adapter`, with a `--served-name`
that differs from the base model's), so the two conditions differ in the adapter alone. A process
answers one request at a time: `make_http_server` builds a single-threaded `HTTPServer`.

### 4.1 What it answers

The server answers `POST /v1/chat/completions` as the runner sends it: one user message with one
text part and at most one picture, as a base64 `data:` URL, with `temperature` 0, `max_tokens` (or
`max_completion_tokens`) and `chat_template_kwargs: {"enable_thinking": false}`.
`build/user_turn.py` builds the user turn: the picture, one newline and the prompt text, rendered by
the model's own chat template with thinking off. For `prompts/zero_shot.txt` and one picture, the
rendered turn has sha256 `44a5cc713a568d729f5839c9f2903ef09327b890b8c9f4b756c765e6585141e8` at the
three sizes.

This turn is the study's one arrangement of picture and prompt text. It is the same at the three
sizes and in both conditions, and every training example starts with it
(`methodology/training-recipe.md` §5.1). It is chosen to match the order of the benchmark's
published hosted runs of the three sizes (the benchmark protocol, §2). The bridge table (`FINETUNE_PLAN.md`,
Phase E) compares those published predictions with the self-served zero-shot run. Its serving
step thus does not also change the layout of the prompt. The runner sends the text part first
and runs unchanged: the server takes the two parts in either order and builds this turn from them.

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

A request is refused with HTTP 400, never adjusted, when it:

- names a model other than the served one;
- sets a temperature other than 0, or `chat_template_kwargs` other than `{"enable_thinking": false}`;
- carries a field outside `model`, `messages`, `max_tokens`, `max_completion_tokens`, `temperature`,
  `chat_template_kwargs` and `min_tokens`;
- sets `min_tokens` on a server not started with `--timing`;
- holds a user turn outside the domain of `build/user_turn.py`, such as two pictures, two text parts,
  no text, a second message or another role.

`--timing` accepts `min_tokens`, which forces an answer to a length so that it can be timed. The
identity shows the mode, and the manifest fails a run made in it.

### 4.3 What every response records

Every response carries `x_served_by`, the identity of what produced it:

| Field | Content |
|---|---|
| `served_name`, `model`, `revision`, `dtype` | the name served, the model, its commit, `bfloat16` |
| `adapter` | in the fine-tuned condition: its path, the sha256 of `adapter_model.safetensors` and `adapter_config.json`, the numbers of LoRA layers and of merged ones, and `attached` |
| `decoding`, `eos_token_id`, `chat_template_kwargs` | `greedy`, [248046, 248044], `{"enable_thinking": false}` |
| `user_turn` | the sha256 of `build/user_turn.py` and the frozen rendered-turn hash |
| `recurrent_layers` | the result of §3.2 |
| `environment` | the result of §3.1: the sha256 of `requirements-card.txt`, whether all pins or the core ones were checked, and the core versions |
| `files` | the sha256 of the five serving files: `infer/server.py`, `build/user_turn.py`, `build/recurrent_layers.py`, `build/card_environment.py`, `requirements-card.txt` |
| `study_commit` | the study repository's commit, and whether the serving files equal it |
| `timing` | whether `--timing` is on |
| `device`, `gpu`, `driver`, `versions`, `optional_kernels` | the device; on CUDA the card model and driver; the library versions |
| per request | `rendered_sha256`, `min_tokens`, `generate_seconds`, `first_token_seconds` |

## 5. The run manifest

`infer/manifest.py` writes `manifest.json` into a run directory of the runner, from `run_meta.json`
and every stored response, and checks it again later:

```
.venv/bin/python infer/manifest.py write <run dir> --server http://127.0.0.1:8000
.venv/bin/python infer/manifest.py check <run dir>
```

`write` records each check below with its result; with `--server` it also compares the live
server's identity. `check` recomputes everything and exits with 1 on a failed check or a difference.

| Check | Passes when |
|---|---|
| `one_configuration` | every answered cell carries the same identity, per-request fields removed |
| `server_matches` | with `--server`: the live server's identity is that one |
| `frozen_files` | the five serving files the server ran equal those of the tag `serving-v3` |
| `environment` | the server checked every pin of `requirements-card.txt` |
| `recurrent_layers` | the active implementation was flash-linear-attention |
| `not_timing_mode` | the server ran without `--timing` |
| `user_turn_file` | the identity's hash of `build/user_turn.py` is the file's hash |
| `rendered_turn` | every answered cell carries the frozen rendered-turn hash, or the one given with `--rendered-sha256` for a run with another prompt |
| `decoding` | `run_meta.json` sets temperature 0, `max_tokens` 5,376 and thinking off |
| `deadline` | `run_meta.json` sets the deadline to 600 s |
| `every_cell_answered` | no cell ended in a failure record of the runner |
| `complete_run` | the run directory holds one stored cell for each of the `n_cells` of `run_meta.json` |
| `served_name` | the runner asked for the name the server serves |

`one_configuration` compares the whole identity, so cells answered from two study commits, two
drivers or two adapter paths fail it. `frozen_files` reads the tag with `git show`, so `manifest.py`
runs in a checkout of the study repository that holds the tag.

## 6. The adapter check

`infer/adapter_check.py` checks that an adapter changes the output, since an adapter that changed
nothing would make the fine-tuned condition the zero-shot one. It loads the base and the adapter
through the server's own start-up, with the same guards, and answers the same pictures twice from
that one loaded model: with the adapter, and with it switched off by `disable_adapter()`. It passes
when the logits of the first answer token differ on every picture and the greedy answers differ on
at least one. The pictures are training pictures, so no fine-tuned answer to a test diagram exists
before its scored run.

```
.venv/bin/python infer/adapter_check.py --model Qwen/Qwen3.5-27B --revision <commit> --adapter <dir> \
    --pictures <png> [<png> ...] --out <result.json>
```

## 7. The deadline

The runner's deadline is 600 s per answer (`--timeout 600`), at every size and in both conditions.
It is the deadline the benchmark's Qwen3.5 runs were completed with (the benchmark protocol, §2).
The manifest's `deadline` check reads it from `run_meta.json`.

## 8. Scored runs

A scored run is a run whose answers the study scores: the dev scoring of a checkpoint
(`methodology/training-recipe.md` §2.1) and every run on the test set. Every scored run follows
these rules.

1. It runs on the card of §2 in the environment of §3.1.
2. It is served, and its manifest written, from one checkout of one commit of the study
   repository whose five serving files equal those of the tag `serving-v3`, with the tag present in
   that checkout. The adapter stays at one path, and the run stays on one pod.
3. Before a scored run with an adapter, `infer/adapter_check.py` passes for that adapter on the card.
4. Its command fixes the selection before the run starts and states the `n_cells` the run must
   reach: all 1,000 test diagrams (the runner's `--n 1000`) or the dev set's key list.
5. An interruption is resumed with the same command plus `--run-dir <its run directory>`, on the
   same pod and checkout. The runner then re-attempts the missing and failed cells.
6. A run that cannot be finished on the same pod is made again from the start in a new run
   directory; the earlier part is not scored.
7. The manifest passes every check, and its `n_cells` equals the one stated beforehand.

## 9. Frozen artifacts

The serving configuration is tagged `serving-v3` in the study repository (annotated tag object
`537a3b149fe02fc7f48926eeafd4ed7f03dae293`, commit `e47bd5e65d56a3a70626543f052e73c376cd211f`).

| File | Role | sha256 |
|---|---|---|
| `infer/server.py` | the server (§4) | `7cb31590262d47c1f908aa18a1b6abdf0ffcabbf1c62c41059282bb31b72278d` |
| `build/user_turn.py` | the user turn of `prompt-arrangement` (§4.1) | `d075641adac54d3ac37e21e0ae10a57f739e715fb5788b661bbe84e36ba02561` |
| `build/recurrent_layers.py` | the recurrent-layer guard (§3.2) | `77cdb3a192c41e4efe33d60fb54af0afed8213dfd474b5d218512695c8565a75` |
| `build/card_environment.py` | the version guard (§3.1) | `5fe071b5bd209ca0e953be1ec6c8045af7676e0cc4aaad4751c5bda2defb9aa2` |
| `requirements-card.txt` | the pinned environment (§3.1) | `9c07234514f4df497fa682fa5a954e91d40d9241cd7c395767ce6a5009d0a874` |
| `infer/manifest.py` | the run manifest (§5) | `dd71b2d176c5ce17e1ccf8b480a7312e30b31b13530a15264ced90366eb178f3` |
| `infer/adapter_check.py` | the adapter check (§6) | `eb488b029449e070488542b410c36ba45b6e44386f7f0dd727dd9e5de52b8145` |

The tests lie under `build/tests/` and `infer/tests/`:

```
.venv/bin/python -m pytest -q build/tests/test_card_environment.py build/tests/test_recurrent_layers.py \
    build/tests/test_user_turn.py infer/tests/test_server.py infer/tests/test_manifest.py \
    infer/tests/test_adapter_check.py
```
