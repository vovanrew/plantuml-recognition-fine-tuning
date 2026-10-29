# Fine-tuning plan — Qwen3.5 dense ladder on image → PlantUML

Status: draft for author review · 2026-09-21
Feeds: dissertation Розділ 4 and the fine-tuning paper
Benchmark repo: `../plantuml-recognition-evaluation` (test set, prompt, scorer, analysis pipeline).
`data/` comes from its `data.zip`; the 1,000 standardized test images are tracked in that repository
as `data/puml_images_1568/` since 2026-10-13. They are not yet in a Zenodo record (ASSESSMENT §4e).

---

## 1. Question and claim

How far does domain adaptation move an open multimodal family on real-world UML diagrams, and what
does the gain consist of: reliability (the output compiles) or structure (the output is right)?

The study fine-tunes Qwen3.5-2B, 9B and 27B with one fixed recipe and re-runs the published benchmark
before and after. The comparison is within-family: each size against itself. The claim is a
measurement, decomposed into Δ CSR, Δ compiled-only structural F1 and Δ population gap. It is not
"a bigger SFT dataset" and not "first to fine-tune for UML".

## 2. What is carried over unchanged from the benchmark

| Item | Value |
|---|---|
| Test set | the 1,000 diagrams of `data/test_set.json`, class/sequence × 4 tiers × 125 |
| Test images | `data/puml_images_1568/` — long edge 1,568 px, Lanczos, PNG, no upscaling |
| Prompt | `prompts/zero_shot.txt`, verbatim, same for both conditions |
| Decoding | temperature 0, `max_tokens` 5,376, `enable_thinking: false` |
| Metrics | CSR, Element F1, Relationship F1, per-relation Relationship F1, type accuracy, chrF++ |
| Populations | full-set (failed = 0) and compiled-only, side by side ("failed = 0" superseded by `decisions/main-tables-intervals/`, 2026-10-29: full-set over all 1,000 diagrams, computed as in Р.3 from the stored per-diagram rows, so an answer that does not render but that the extractor parses keeps its parsed counts. `methodology/analysis-protocol.md` §4.) |
| Statistics | paired percentile bootstrap, seed 20260614, 1,000 resamples |

**Input standard: 1,568 px is re-used** for training images, dev scoring and both evaluation
conditions. No resolution study.

Published zero-shot results for the ladder (hosted, scorer v1; full-set / compiled-only):

| Model | CSR | Element F1 | Relationship F1 | chrF++ | Type acc |
|---|---|---|---|---|---|
| Qwen3.5-2B | 20.1% | 0.246 / 0.878 | 0.114 / 0.481 | 8.96 / 65.12 | 0.758 |
| Qwen3.5-9B | 43.1% | 0.491 / 0.844 | 0.201 / 0.439 | 24.40 / 61.17 | 0.756 |
| Qwen3.5-27B | 60.1% | 0.676 / 0.917 | 0.485 / 0.722 | 38.41 / 65.65 | 0.886 |

## 3. Design

- **3 sizes × 2 conditions** (zero-shot, fine-tuned) × 1,000 diagrams = 6 main runs on one
  self-hosted stack.
- **One change per comparison.** Anything not under test is identical in both arms: images, prompt,
  decoding, precision, serving stack, scorer.
- **One recipe across the ladder.** Same data, mix, precision, hyperparameters, epoch count and
  checkpoint rule at 2B, 9B and 27B. Ablations run outside the ladder at a single size.
- **Frontier models and the MoE are context, not competitors.** If shown, they appear as reference
  lines from the stored hosted predictions, re-scored under scorer v2. No new frontier inference.
- **One seed per size.** Stated as a limitation; the bootstrap covers diagram variance only.

## 4. Phases

### Phase 0 — Scorer v2

The scorer changes, where they go and why: **`PLAN_PREREQUISITE.md`**. Three name-matching fixes
(whitespace in Python; markup and the bold-name-plus-package pattern in the extractor fork); note
counting and the two prediction-side cases stay as they are. (As planned; what was decided and built is in
`decisions/scorer-v2-freeze/record.md` and `methodology/scorer-v2.md` §2 to §4, which hold where
they differ from this paragraph: five name rules, two edge rules and the first-diagram rule.)

Rules for this study:

- Why v2 belongs here: v1 credits reproducing the source's styling, which a model trained on the same
  corpus learns to imitate. Expected effect: v2 raises the zero-shot baseline at least as much as the
  fine-tuned score, so the delta gets more conservative. Verify per fix in the bridge table. (Measured on the published zero-shot predictions:
  `methodology/scorer-v2.md` §6. Bridge table cut 2026-10-28, author: `decisions/bridge-table/`; the
  expectation on both conditions is checked by `main-tables-intervals` from the v1 numbers reported
  alongside. "From the v1 numbers reported alongside" superseded by `decisions/main-tables-intervals/`,
  2026-10-29: the expectation is read on the joint set, the diagrams both runs of a size compiled, as
  Δ_v2 − Δ_v1 with a paired interval; the full-set and compiled-only v1 deltas stand beside without a
  verdict. `methodology/analysis-protocol.md` §13.)
- **Frozen and tagged before any fine-tuned output is scored**, dev-set checkpoint scoring included.
  Record the JAR hash and scorer tag here once frozen.
  Frozen 2026-10-13: tag `scorer-v2` of `../plantuml-recognition-evaluation` (commit `f460b35`);
  extractor fork commit `549fe8ec08`; extractor JAR sha256
  `2747a72aec9206e424ddf9ed659f9eceb96730444a68b1fe99094317400f555b`. No fine-tuned output had been
  scored.
- Applied identically to both conditions; v1 numbers are reported alongside.
- An artifact discovered after the freeze goes into a separate sensitivity analysis on both
  conditions, not into v2.
- v1 stays runnable; scoring is offline, so any run can be scored under both.

Output: tagged scorer v2, and the stored 8 runs re-scored under v2.

### Phase A — Training set

**Rule: the training pool is built under the same construction rules as the test set, and its images
come from the same rendering pipeline.** (`../plantuml-recognition-evaluation/methodology/test-set-construction.md` §3–§9.)

Pool construction, in order (as planned; what was decided and built is in
`decisions/training-pool-rules/record.md` and `methodology/training-set-construction.md` §2–§8,
which hold where they differ from this list):

1. Inclusion: `primary_type ∈ {class, sequence}`, no secondary types, extraction succeeded, not
   truncated, `elements_total ≤ 50`, repository known.
2. Remove every repository reserved by the test set (repo-disjoint).
3. Degeneracy exclusions: multi-page `newpage`, renderer-clipped (long edge ≥ 16,384 px), aspect
   ratio > 8:1.
4. Normalized-code deduplication (comments removed, whitespace collapsed, SHA-1).
5. Type-agreement filter: `parser.diagram_type == primary_type`.
6. Drop targets longer than the 5,376-token output cap.
7. **Leakage, measured:** near-duplicate similarity of every candidate against all 1,000 test
   diagrams; drop above a stated threshold; report the residual maximum.

Expected pool: roughly 70,000 diagrams, naturally ~63/37 class/sequence. (Built: 66,379 diagrams,
64.01% class; `methodology/training-set-construction.md` §2.)

Sampling:

- **N ≈ 20,000**, 50/50 class/sequence, balanced over the four tiers using the test set's
  `content_lines` quartile boundaries.
- Data-efficiency subsets **1k ⊂ 5k ⊂ 20k**, nested, same mix.
- **Dev set: 200 diagrams** (25 per cell), repo-disjoint from both train and test, for checkpoint
  scoring. Separate small validation split for loss curves.
- Commit the key lists, seed and mix as artifacts, the way the test set was.

Images and targets:

- `brew install graphviz` first. Without it class diagrams render to a "Cannot find Graphviz" notice
  that is a valid PNG; the failure is silent.
- Render with the pipeline that produced `data/puml_images/`, then `util/standardize_images.py`
  (1,568 px). The PNGs published with the corpus are a different, lower-resolution render and must
  not be used.
- **Acceptance check:** re-render the 1,000 test-set diagrams through the training pipeline and confirm the
  output dimensions equal their `data/puml_images_1568/` counterparts.
- Target = PlantUML source from `@startuml` to `@enduml`, with the three-line World of Code header
  stripped. (As built by `pilot-run`: the block the scorer's `extract_puml` isolates;
  `methodology/training-recipe.md` §5.1.)
- Every training image must compile and render to exactly one page; log and drop the rest.

Output: `train_{1k,5k,20k}.jsonl`, `dev_200.jsonl`, images, leakage report, final counts per cell.
(Built by `training-pool-rules`: `data/sets/train.json` with the subsets as ranks, `dev.json`,
`loss_curve.json`, `d8_report.json`, `counts.json`, the images under `data/sets/render/`, and
`data/pool/`. The `.jsonl` files of examples are the formatting step of Phase B, built by
`pilot-run`: `data/examples/`, `methodology/training-recipe.md` §5.1.)

### Phase B — Formatting and pilot

- Messages format: the benchmark prompt + image as the user turn, the bare PlantUML block as the
  assistant turn. Loss on the assistant turn only.
- Chat template pinned with `enable_thinking=false`. **Byte-equality test:** the templated prefix
  produced at training time equals the one the inference runner sends ("the one the inference
  runner sends" superseded by `decisions/prompt-arrangement/`, 2026-10-11: the one the server builds
  from the runner's request; the user turn is picture, newline, text, built by one function that
  training and the server both call).
- Pilot on 1–2k examples at 2B (Colab is enough; "Colab is enough" superseded by `decisions/gpu-precision/`, 2026-10-11: every run on the one card model; "1–2k" superseded by `decisions/recipe/`, 2026-10-18: the 2B on the 1,000-pair subset, run by `pilot-run`). Exit criterion: the pipeline runs end to end and
  the dev set scores non-zero CSR and Element F1 — not "the output looks like PlantUML".
- The pilot fixes the epoch count and confirms memory headroom at the longest sequences. ("Fixes the
  epoch count" is superseded by `decisions/recipe/`, 2026-10-18: 2 epochs at every size, the
  adapter picked within each run on the dev set; the pilot keeps its exit criterion.)

### Phase C — Training

| Setting | Proposed value (confirm in pilot; the values are fixed by `decisions/recipe/`, 2026-10-18: `methodology/training-recipe.md`) |
|---|---|
| Method | LoRA on the bf16 base, no quantization, **same at all three sizes** (author, 2026-10-11; "QLoRA, 4-bit NF4 base, bf16 compute" is superseded by `decisions/gpu-precision/`) |
| Adapters | all linear layers of the language model; vision encoder frozen |
| Rank / alpha / dropout | 16 / 32 / 0.05 |
| Learning rate / schedule | 1e-4, constant after a 3% warm-up ("cosine" superseded by `decisions/recipe/`, 2026-10-18) |
| Effective batch | 16 sequences ("16–32" superseded by `decisions/recipe/`, 2026-10-18) |
| Epochs | 2, identical at all sizes; the checkpoint picked within each run by one rule ("fixed by the pilot (1 or 2)" superseded by `decisions/recipe/`, 2026-10-18) |
| Hardware | one GPU class for the whole ladder |

- Order: 2B → 9B → 27B.
- **Checkpoint selection on real metrics, not validation loss:** score the 200-diagram dev set with
  CSR + Element F1 under frozen scorer v2 at each checkpoint. Same rule at all sizes.
- **Data-efficiency curve at 2B:** train on 1k, 5k and 20k, report where the curve saturates. Two
  extra short runs. (Narrowed by `decisions/data-efficiency-sizes/`, 2026-10-25: the 2B only; the
  1k run is the pilot's, one new 5k run; the selected 1k and 5k adapters each get a test-set run,
  and the curve is reported on the test set, 0 / 1k / 5k / 20k. Reporting by
  `decisions/efficiency-curve-analysis/`, 2026-10-29: every score shown, the reading of where the
  curve saturates written after the scores are seen; `methodology/analysis-protocol.md` §15.)
- **Ablation, only if Weeks 6–8 come in early:** one run at one size with the vision encoder
  unfrozen. It is the only test of whether relation decorations (filled vs hollow diamond, dashed
  vs solid) are recoverable on the model side.

Output: three adapters (+2 data-efficiency adapters), training logs, dev-set score per checkpoint.

### Phase D — Inference

- **6 main runs** on one stack, identical decoding, raw responses stored immutably in the benchmark's
  run format. The fine-tuned condition serves the same bf16 base with the adapter attached, not merged
  ("the same 4-bit base with the adapter attached" is superseded by `decisions/gpu-precision/`,
  2026-10-11; attached was decided there the same day).
- The zero-shot condition is **re-run on this stack**, not taken from the published table. Expect
  movement: the published 2B run lost 25 cells to a wall-clock deadline on no-stop generations, a
  failure mode that does not exist when self-serving.
- **Two inference-only controls on the zero-shot models:**
  1. a few-shot / format-constrained prompt;
  2. a mechanical repair pass that closes an unterminated block.
  If either recovers most of the reliability gain, the claim becomes "fine-tuning buys structure,
  not compilation".
  (Cut from this study, author, 2026-10-18: the controls are not part of the paper; they would be a
  separate thesis. `decisions/controls-definition/`.)
- **Robustness run:** the same 1,000 diagrams re-rendered under (a) a different PlantUML theme and
  (b) documentation-like degradation (JPEG compression, screenshot-scale resampling, ±3° rotation,
  mild noise). Both conditions, scored against the unchanged ground truth. Any training-time
  augmentation stays train-only.
- `analysis/registry.py` needs a `condition` field; `_VALID_ARM = {"frontier","qwen"}` raises on the
  fine-tuned runs today.

### Phase E — Evaluation

Everything under scorer v2, full framework, both populations.

- **Data-efficiency curve at the 2B** (optional): 0 / 1k / 5k / 20k pairs on the test set, the 0
  and 20k points from the 2B's main runs (`decisions/data-efficiency-sizes/`, 2026-10-25); every
  score descriptive, the reading written after (`decisions/efficiency-curve-analysis/`, 2026-10-29).

- **Bridge table** (opens Р.4): per size, published predictions under v1 → same predictions under
  v2 → self-served bf16 zero-shot under v2 ("4-bit" superseded by `decisions/gpu-precision/`,
  2026-10-11). Separates the scorer effect from the serving effect.
  One sentence on whether any published conclusion (ranking, population gap) moves under v2.
  (Cut from this study, author, 2026-10-28: the gain is measured within the study's own runs, and
  the baseline is the self-served zero-shot run, stated once in the paper; Р.3's hosted numbers are
  not compared. `decisions/bridge-table/`.)
- **Main table:** size × condition × type × tier. ("× type × tier" superseded by
  `decisions/main-tables-intervals/`, 2026-10-29: size × condition, overall and at the type and tier
  margins, not their cross; margin cells are points. `methodology/analysis-protocol.md` §10.)
- **Decomposition of every gain:** Δ CSR, Δ compiled-only Element / Relationship F1, Δ population
  gap. Published gaps on Element F1 are +0.632 (2B), +0.354 (9B), +0.241 (27B); the headline result
  is how far fine-tuning collapses them. (Since the bridge table's cut, 2026-10-28: the gaps the
  headline uses are those of the self-served zero-shot runs; the published ones are context only.)
  ("Δ compiled-only Element / Relationship F1" superseded in part by
  `decisions/main-tables-intervals/`, 2026-10-29: Element F1 carries the claim; Relationship F1
  stands beside it and is descriptive; Δ gap is shown with its interval but is not a separate member
  of the claim family. `methodology/analysis-protocol.md` §6, §9.)
- **Intervals:** paired bootstrap on each before/after delta, plus a separate paired resample for
  the **difference of deltas** ("2B gains more than 27B"). Three sizes give a direction, not a fit.
  ("A separate paired resample" superseded by `decisions/main-tables-intervals/`, 2026-10-29: the
  differences of deltas come from the same draw as the Δ, so they are paired by diagram.
  `methodology/analysis-protocol.md` §8.)
- **Per-relation Relationship F1** before and after — the test of the composition / aggregation /
  dependency hypothesis. ("The test of the composition / aggregation / dependency hypothesis"
  superseded by `decisions/main-tables-intervals/`, 2026-10-29: per-relation results are descriptive,
  with paired intervals above a support floor and no decoration verdict, because an edge matches only
  when both endpoint names match too; the direct test, relation agreement on endpoint-matched edges,
  goes to `error-analysis`. `methodology/analysis-protocol.md` §11.) (Done by `error-analysis`, 2026-10-29: `methodology/analysis-protocol.md` §16.)
- **Gains by tier.** Flat gains on the densest tier point at the input, not the model. ("Point at the
  input, not the model" superseded by `decisions/main-tables-intervals/`, 2026-10-29: Δ(tier 4) −
  Δ(tier 1) is reported with its interval, and a smaller gain on tier 4 is worded as consistent with
  a limit set by the input, never as showing one; the tiers differ in diagram size and in crowding
  together. `methodology/analysis-protocol.md` §12.)
- **Robustness** reported next to the main result, same metrics.
- **Error analysis** on a stratified failure sample, mapped onto the existing taxonomy so before and
  after compare like with like. (Done by `error-analysis`, 2026-10-29: a blind hand-coded sample of 95 failure cases and
  relation agreement on endpoint-matched edges, `methodology/analysis-protocol.md` §16.)

## 5. Schedule

| Weeks | Work |
|---|---|
| 1–2 | Phase 0 (scorer v2, frozen) · Phase A (pool, leakage, renders, dev set) · decide framework and GPU |
| 3–5 | Phase B pilot · Phase C ladder 2B → 9B → 27B · data-efficiency runs |
| 6–8 | Phase D six runs, robustness · `registry.py` change · Phase E tables and intervals |
| 9–12 | Error analysis · optional unfrozen-encoder ablation · write-up (Р.4, paper, arXiv preprint) |

## 6. Out of scope

| Cut | Reason |
|---|---|
| Input-resolution study | changes a second variable; 1,568 px keeps the link to the published benchmark |
| Constrained decoding with a PlantUML grammar | weeks of work |
| RL stage (GRPO, rendered-visual reward) | scope; one sentence of justification in the text |
| Text-only → VL staged pre-training | not needed for the claim |
| The other 7 diagram types | class + sequence = 76.2% of the corpus |
| Inference-only controls (few-shot / format-constrained prompt, mechanical repair pass) | not part of the paper; a separate thesis (author, 2026-10-18) |
| Bridge table (Р.3's published zero-shot numbers, re-scored under v2, against the study's self-served zero-shot run) | the gain is measured within the study's own runs; the baseline is the self-served zero-shot run, stated once in the paper (author, 2026-10-28) |
| Classifier rates per 1,000 of compile-fail categories (C1 / C2 / C3) | the classifier's raw rates are wrong in size and the chapter does not need them; `error-analysis`'s hand sample and §16 counts describe the failures (author, 2026-10-29) |
| Hand-labelled in-the-wild image set | manual ground truth is the expensive part; theme + degradation covers the threat |

## 7. Open questions

Questions not yet promoted to an item. Promoting one means adding a row to §10 and, if its answer
would invalidate scored results or break a comparison, giving it the frozen tier. The seven open
items listed here before 2026-09-22 are now rows in §10 (`training-stack`, `serving-stack`,
`gpu-precision`, `leakage-threshold`, `render-pipeline`, `robustness-parameters`,
`reference-lines`, `adapter-release`).

- [x] Whether the assessment's weeks 9–12 additions belong to this plan. Decided by the author,
      2026-10-04: yes, as optional items in §10 (`human-rater-validation`, `difficulty-regression`,
      `edit-readiness-rate`, `prototype`), decided case by case after the core results.

- Hand-offs from `training-pool-rules` (recorded 2026-10-11; details in its record, Open threads),
  for items that have no record yet:
  - `pilot-run` (taken up by `pilot-run`, recorded 2026-10-20: `methodology/training-recipe.md`
    §5.1; for the pilot the pictures went in the session's pack): read the training set from
    `data/sets/train.json`, not from the image folder,
    although since 2026-10-12 the folder holds exactly the 20,000 images of the list (the 19 of replaced
    keys are in `data/replaced_images/train/`). How a target is written for the 6 training sources
    that the scorer reads from the whole file (1 in the 5,000 subset). One training diagram holds
    a debug pragma line in its target (`a416dd4f5567…`, in the 20,000 and the 5,000). Since Amendment 2 of
    2026-10-11 the near-duplicate rule is also run between the training pool and the loss-curve
    split; the split is not compared with the dev set (12 of its 200 diagrams reach a threshold
    against a dev diagram). The images are not in either repository: the training machine gets
    them from the private Hugging Face dataset `vovanrew/plantuml-finetune-backup`, whose
    `README.md` says where each archive unpacks and whose `SHA256SUMS` checks them
    (`training-pool-rules` record, Open threads, 2026-10-17).
  - `adapter-release`: whether the training set is distributed as a key list or as the stored
    images (a second render does not repeat 11 of 20,000 images; author, 2026-10-10: decided
    later). The images are backed up since 2026-10-17 in a private Hugging Face dataset,
    `vovanrew/plantuml-finetune-backup`, which is not the release. 7 training sources name a
    picture at a URL, which the render fetched: 2 stored images show the picture and 4 show
    "Cannot decode" in its place; the sets are kept as built (author, 2026-10-17).

## 8. Budget

Three ladder trainings, two data-efficiency runs (one new training and two test-set runs since
`decisions/data-efficiency-sizes/`, 2026-10-25: USD 19.64–36.35), headroom for one failed run per size, and 6 main +
robustness inference runs: **$300–800**.

## 9. Proposed layout of this folder

```
fine-tune/
  FINETUNE_PLAN.md
  data/          train/dev key lists, jsonl, leakage report   (images gitignored; the freeze is
                 `data/*/SHA256SUMS`; the folder is a git repository since 2026-10-13)
  build/         pool construction, rendering, formatting; assembly of the scorer-v2 extractor
                 JAR and the v1-to-v2 bridge of the stored runs; scoring of the test-set runs,
                 the main tables and the study's registry (`main-tables-intervals`)
  train/         the recipe, one file for every size ("configs (one per size)" superseded by
                 `decisions/recipe/`, 2026-10-18), the checkpoint rule, launch scripts
  infer/         the study's server, the run manifest and the adapter check (the runner is the benchmark's, unchanged);
                 the runs on the test set (run list, card queue, session driver)
  runs/          local working output of a card session and its scores and tables, kept out of
                 the repository; a finished run that is not final moves to its item's folder
                 under `decisions/`; final runs are
                 published by `adapter-release` ("raw outputs, immutable" superseded, author,
                 2026-10-19)
  decisions/     one folder per §10 item: record.md plus the role outputs (plan-item skill);
                 kept out of the repository
  methodology/   facts about what exists, written after verification
```

`decisions/` is kept out of the repository (author, 2026-10-12). It holds the history of each
decision, and that history is not what a reader reproduces. What is reproducible is the methodology,
from `build/`, `train/`, `infer/` and the frozen files of `data/`. Anything a result, a test, a
later item or a methodology entry depends on therefore lives in those folders; an adapter a later
item builds on is restored from the private backup into `runs/`, not read from `decisions/`. A script
under `decisions/<slug>/scripts/` measures or checks for a role: nothing outside `decisions/`
imports it, runs it or is produced by it, and a methodology entry names no command or file under
`decisions/`. A script that comes to be needed is moved out.

Scoring and analysis stay in `../plantuml-recognition-evaluation` so both conditions go through one
instrument. (Superseded in part by `decisions/main-tables-intervals/`, 2026-10-29: the instrument's
runners, loader and bootstrap run unchanged, so both conditions still go through one instrument; the
study's scoring and analysis commands are in `build/`, and their outputs are in `runs/test/`.
`methodology/analysis-protocol.md` §2, §3, §14.)

## 10. Items

One row per unit of work the `plan-item` skill runs (`.claude/skills/plan-item/`). A row holds
identity and status only: the reasons live in `decisions/<slug>/record.md`, the facts in
`methodology/`. An item that is only implementation has no entry there; its facts are in its record
and in the code. Promote a question from §7 by adding a row. Never delete a row; mark it `cut`,
`merged into <slug>` or `superseded by <slug>`. Status mirrors the record's status line
(`framing → scouted → challenged → decided → implemented → verified → recorded`).

Tier: **frozen** if changing the item later would invalidate scored results or break a comparison;
**implementation** otherwise. When in doubt, frozen.

Path (author, 2026-10-04): **core** items are the training set, the basic fine-tune and the
comparison that carries the chapter's claim; they are worked in order and none is dropped.
**optional** items are candidate future work: each is decided case by case, depending on time,
once the core results exist. An optional item that is not done moves to §6 with its reason.

| Slug | Tier | Path | Decides or builds | Plan §§ | Status | Record |
|---|---|---|---|---|---|---|
| scorer-v2-freeze | frozen | core | what is inside v2 (name fixes, sensitivities, limitations, multi-block rule); freeze mechanics: tags, JAR hashes, v1 kept runnable, bridge panel | Phase 0; `PLAN_PREREQUISITE.md` | recorded 2026-10-13 (verified 2026-10-12, `verify.md`, tenth pass; Parts 1 to 4 and Amendments 1 to 7; tag `scorer-v2` on scorer commit `f460b35`, fork commit `549fe8ec08`, extractor JAR sha256 `2747a72a…555b`) | `decisions/scorer-v2-freeze/` · `methodology/scorer-v2.md` §1–§7 |
| training-pool-rules | frozen | core | inclusion, repository disjointness, degeneracy exclusions, normalized dedup, type agreement, token cap; pool count re-measured | Phase A 1–6 | recorded 2026-10-11, Amendments 1 and 2 included (verified 2026-10-11, `verify.md`, sixth verification, of Amendment 2; the fifth is `verify_2026-10-11_fifth_pass.md`; four earlier verifications failed on 2026-10-10: `verify_2026-10-10_fail.md`, `_fail_2.md`, `_fail_3.md`, `_fail_4.md`; decided 2026-10-10, Amendment 1 2026-10-10) | `decisions/training-pool-rules/` · `methodology/training-set-construction.md` §2–§8 |
| leakage-threshold | frozen | core | the near-duplicate similarity measure, the drop threshold, the residual maximum reported | Phase A 7 | merged into training-pool-rules 2026-10-04 | `decisions/training-pool-rules/` |
| train-size-mix | frozen | core | N, class/sequence mix, tier balance, nested 1k ⊂ 5k ⊂ 20k, seed; key lists committed | Phase A sampling | merged into training-pool-rules 2026-10-04 | `decisions/training-pool-rules/` |
| dev-set | frozen | core | 200-diagram cell design, disjoint from train and test, validation split for loss curves; its ground truth audited for the scorer-v2 artifact classes before the freeze | Phase A sampling; Phase 0 | merged into training-pool-rules 2026-10-04 | `decisions/training-pool-rules/` · audit: `decisions/scorer-v2-freeze/dev_audit.md` · `methodology/scorer-v2.md` §5 |
| render-pipeline | implementation (was frozen; the pipeline is fixed by the test set, this item reproduces it) | core | the three existing render stages behind one command; versions recorded; acceptance check over the 1,000 test diagrams | Phase A images | recorded 2026-10-09 (verified 2026-10-09, `verify.md`, fourth verification; three earlier builds failed: `verify_2026-10-05_fail.md`, `verify_2026-10-08_fail.md`, `verify_2026-10-09_fail.md`) | `decisions/render-pipeline/` · `methodology/training-set-construction.md` §1 |
| training-stack | frozen | core | one training framework for the whole ladder | Phase C | merged into gpu-precision 2026-10-04 | `decisions/gpu-precision/` |
| serving-stack | frozen | core | one serving stack for both conditions, bf16 base with adapter attached ("4-bit" superseded by `decisions/gpu-precision/`, 2026-10-11) | Phase D | merged into gpu-precision 2026-10-04 | `decisions/gpu-precision/` |
| gpu-precision | frozen | core | the compute platform: provider, one GPU class, one precision at all sizes (bf16; "4-bit" superseded 2026-10-11), training framework, serving stack; 27B fits at the longest sequences; cost of the plan and of the cut | Phase B; Phase C; Phase D; §8 | recorded 2026-10-18 (verified 2026-10-18, `verify.md`, fifth pass; all five passes are in that file, the first, second and fourth failed): released bf16 weights, no quantization, at every size; one card model, NVIDIA RTX PRO 6000 Blackwell Server Edition, in Runpod Secure Cloud; transformers 5.18.0 + PEFT 0.21.2 on torch 2.14.1, every package pinned in `requirements-card.txt`; recurrent layers on flash-linear-attention 0.5.2; the study's own server, adapter attached and not merged, one request at a time; answer deadline 600 s; run manifest and adapter check; frozen as tag `serving-v3` on commit `e47bd5e` | `decisions/gpu-precision/` · `methodology/compute-and-serving.md` §1–§9 |
| prompt-arrangement | frozen | core | the order of picture and prompt text in the user turn (picture first, as the stored hosted runs appear to have been rendered, or text first, as the official template renders the runner's request), one arrangement in training and in serving, at all sizes and in both conditions | §2; Phase B; Phase D; Phase E | recorded 2026-10-22 (verified 2026-10-22, `verify.md`; decided 2026-10-11): picture, newline, text, at all sizes and in both conditions, one function for training and the server, rendered turn sha256 `44a5cc71…41e8`; the order of the published hosted runs; confirmed on the first card session, 2026-10-12: the more likely rendering on 18 of 20 cells at the 9B and 20 of 20 at the 27B | `decisions/prompt-arrangement/` · `methodology/compute-and-serving.md` §4.1 |
| recipe | frozen | core | adapter targets, rank/alpha/dropout, learning rate and schedule, effective batch; the epoch count and the rule that fixes it; checkpoint selection on dev-set CSR + Element F1 under frozen v2; identical at all sizes (absorbs `epoch-checkpoint-rule`, author, 2026-10-17) | Phase B; Phase C | recorded 2026-10-18 (verified 2026-10-18, `verify.md`, fourth pass; the first three failed: `verify_2026-10-18_fail.md`, `_fail_2.md`, `_fail_3.md`): 2 epochs at every size, constant learning rate 1e-4 after a 3% warm-up, adapter picked within each run on dev CSR and Element F1 under v2 (the last checkpoint unless an earlier one is resolvably better); rank 16, alpha 32, dropout 0.05 on the language model's linear layers; effective batch 16 | `decisions/recipe/` · `methodology/training-recipe.md` §1–§4 |
| epoch-checkpoint-rule | frozen | core | epoch count fixed by the pilot; checkpoint selection on dev-set CSR + Element F1 under frozen v2; same rule at all sizes | Phase B; Phase C | merged into recipe 2026-10-17 | `decisions/recipe/` |
| controls-definition | frozen | core | the few-shot / format-constrained prompt and the mechanical repair pass, fixed before they are scored | Phase D | cut 2026-10-18 (author): not part of the paper; a separate thesis (candidate B in `../theses/next-theses-probes/CANDIDATES.md`) | `decisions/controls-definition/` |
| robustness-parameters | frozen | optional | the alternative theme and the degradation parameters, fixed before scoring | Phase D | — |  |
| reference-lines | frozen | optional | whether frontier and MoE lines appear; if so, re-scored under v2 from stored predictions | §3; Phase E | — |  |
| adapter-release | frozen | optional | HuggingFace repository names, licence, what is deposited | §7 (former) | — |  |
| graphviz-render-check | implementation | core | Graphviz installed; class diagrams render to diagrams, not the notice image | Phase A images | merged into render-pipeline 2026-10-04 (Graphviz 16.1.0 present; 500 of 500 class test diagrams re-render to benchmark dimensions) | `decisions/render-pipeline/` |
| registry-condition-field | implementation | core | `analysis/registry.py` gains `condition`; `_VALID_ARM` accepts the fine-tuned runs | Phase D | recorded 2026-10-10 (verified 2026-10-10, `verify.md`); no methodology entry (author, 2026-10-11); rule 5, who registers the six runs, superseded 2026-10-26: `main-tables-intervals` (author) | `decisions/registry-condition-field/` |
| pilot-run | implementation | core | messages format, chat template pinned, byte-equality test; the training script; the recipe's 60-step probe at the three sizes and the 27B's memory at the longest sequence; the pilot: the 2B on the 1,000-pair subset under the recipe (2 epochs, 126 steps, four checkpoints scored on dev), exit criterion met ("1–2k" superseded by `decisions/recipe/`, 2026-10-18) | Phase B | recorded 2026-10-22 (verified 2026-10-22, `verify.md`, seventh pass; the kernels' reversal check run on the card, the decision not reversed, author 2026-10-21): the training example is the served user turn, then the target and `<|im_end|>`, loss on the answer only, byte-equal to what the server builds; the probe passes at the three sizes; the 27B fits under the recipe at the longest training sequences; pilot dev CSR 0.740, Element F1 0.7616 at the selected step 126; exit criterion met | `decisions/pilot-run/` · `methodology/training-recipe.md` §2.1, §2.3, §3–§6 |
| ladder-training | implementation | core | 2B → 9B → 27B under the recipe; adapters, logs, dev-set score per checkpoint | Phase C | recorded 2026-10-26 (verified 2026-10-26, `verify.md`, second pass; implemented 2026-10-25; each size selects step 2,500; dev CSR / Element F1: 2B 0.915 / 0.900, 9B 0.980 / 0.968, 27B 0.975 / 0.956) | `decisions/ladder-training/` · `methodology/training-recipe.md` §7 |
| data-efficiency-runs | implementation | optional | 1k and 5k at 2B (1k: the pilot's run, `pilot-run`, 2026-10-20); the selected 1k and 5k adapters each get a test-set run, the curve 0 / 1k / 5k / 20k on the test set at the 2B only (`data-efficiency-sizes`, 2026-10-25) | Phase C | recorded 2026-10-29 (verified 2026-10-29, `verify.md`, second pass; the 5,000-pair run 2026-10-25, step 626 selected; the two test-set runs, made by `main-inference-runs` 2026-10-26 and 27, scored 2026-10-29; the curve as `efficiency-curve-analysis` decided) | `decisions/data-efficiency-runs/` · `methodology/training-recipe.md` §8 |
| data-efficiency-sizes | frozen | optional | at which sizes the 1k and 5k data-efficiency runs are trained and evaluated: the 2B only, as planned, or also the 9B and the 27B; whether the 2B alone is enough for the study. Evaluation only (author, 2026-10-25): no implementation, no methodology entry; the answer sets the scope of `data-efficiency-runs` | Phase C; §8; row data-efficiency-runs | recorded 2026-10-26; no methodology entry (decided 2026-10-25: A + test, the 2B only (1k, new 5k, 20k), the 1k and 5k adapters also run on the test set); superseded in part by `efficiency-curve-analysis`, 2026-10-29: the reading fixed before the 5,000-pair result exists (author: every score shown, the reading written after) | `decisions/data-efficiency-sizes/` |
| efficiency-curve-analysis | frozen | optional | before the 1,000- and 5,000-pair test runs are scored: how the 2B's test curve 0 / 1k / 5k / 20k is computed and read. Every score shown (the main table's v2 statistics at each point, v1 F1 and failed = 0 beside; the neighbour differences with paired intervals, 20k − 5k also on the joint set), all descriptive and outside the claim family; no verdict rule fixed in advance, the reading written after the scores are seen (author, 2026-10-29; supersedes in part `data-efficiency-sizes`); its own registry and script, `build/main_tables.py`, `build/test_registry.json` and their sha256 unchanged | Phase C; Phase E; rows data-efficiency-runs, data-efficiency-sizes, main-tables-intervals | recorded 2026-10-29 (verified 2026-10-29, `verify.md`, second pass) | `decisions/efficiency-curve-analysis/` · `methodology/analysis-protocol.md` §15 |
| unfrozen-encoder-ablation | implementation | optional | one run at one size with the vision encoder unfrozen | Phase C | conditional: only if weeks 6–8 come in early |  |
| main-inference-runs | implementation | core | 6 runs, raw responses stored in the benchmark's run format; its tooling and card session also make the two test-set runs of `data-efficiency-runs` | Phase D | recorded 2026-10-28 (verified 2026-10-28, `verify.md`, second pass; tooling 2026-10-26; the card session 2026-10-26 to 2026-10-28: eight runs on the test set, the six main and the two of `data-efficiency-runs`, each manifest passing with `n_cells` 1,000; the port check; the benchmark's run reader skips the run manifest) | `decisions/main-inference-runs/` · `methodology/compute-and-serving.md` §10 |
| controls-and-robustness-runs | implementation | core | the two control runs and the robustness run, scored under v2 | Phase D | split 2026-10-04 into control-runs and robustness-runs |  |
| bridge-table | frozen (was implementation; Frame 2026-10-28: it fixes what is reported before the self-served zero-shot runs are scored) | core | per size: published v1 → same predictions under v2 → self-served bf16 zero-shot under v2 ("4-bit" superseded by `decisions/gpu-precision/`, 2026-10-11) | Phase E | cut 2026-10-28 (author): the gain is measured within the study's own runs, the baseline is the self-served zero-shot run, stated once in the paper; Р.3's hosted numbers are not compared (framed, scouted and challenged before the cut) | `decisions/bridge-table/` |
| main-tables-intervals | frozen (was implementation; Frame 2026-10-28: it fixes what is reported, the interval family and the reading rules before the six runs are scored) | core | main table, gain decomposition, paired intervals including the difference of deltas, per-relation, per-tier; the six main runs' registry entries (`analysis/model_registry.json`, `condition`; "`analysis/model_registry.json`" superseded by `decisions/main-tables-intervals/`, 2026-10-29: the study's own `build/test_registry.json`, the published registry unchanged), written with their score files (author, 2026-10-26, in `main-inference-runs`; supersedes `registry-condition-field` rule 5 on that point) | Phase E | recorded 2026-10-29 (verified 2026-10-29, `verify.md`, second pass) | `decisions/main-tables-intervals/` · `methodology/analysis-protocol.md` §1–§14 |
| error-analysis | frozen (changed from implementation at Frame, 2026-10-29: it selects the failure sample, judges outputs and defines a reported statistic) | core | stratified failure sample mapped onto the existing taxonomy, before and after; relation agreement on endpoint-matched edges, the direct per-relation test (from `main-tables-intervals`, 2026-10-29) | Phase E | recorded 2026-10-29 (verified 2026-10-29, `verify.md`, third pass; decided 2026-10-29, author: option 1, a blind hand-worked sample of 95 failure cases, and relation agreement on edges paired across both runs; option 2, the classifier census, split into `failure-classifier-rates`, cut 2026-10-29) | `decisions/error-analysis/` · `methodology/analysis-protocol.md` §16 |
| failure-classifier-rates | frozen | optional | the baseline-errors thesis classifier's compile-fail branch ported into `build/`, frozen and hashed before it labels any fine-tuned answer; C1 / C2 / C3 rates per 1,000 diagrams before and after over the 1,964 compile-fail cells of the six main runs (the low-structural cells keep `error-analysis`'s hand sample and mechanical split); validated on about 80 held-out cases stratified on classifier code × condition, with agreement reported per condition (split from `error-analysis` option 2, author, 2026-10-29; run in parallel with it) | Phase E; row error-analysis | cut 2026-10-29 (author, after Implement): its validation shows the raw rates wrong in size (zero-shot C3 about half the corrected figure) with intervals around them, and the chapter does not need them; `error-analysis` already states the compile-fail mix (decided 2026-10-29: compile-fail codes only) | `decisions/failure-classifier-rates/` |
| control-runs | implementation | core | the mechanical-repair control and the few-shot control, scored under v2 | Phase D | cut 2026-10-18 with `controls-definition` | |
| robustness-runs | implementation | optional | the alternative-theme run first; image degradation after the fine-tune results, if time allows | Phase D | — | |
| human-rater-validation | implementation | optional | people rate a sample of predictions; agreement between their ratings and the metrics, under v1 and v2 | ASSESSMENT §6 weeks 9–12 | — | |
| difficulty-regression | implementation | optional | which diagram properties predict compile failure and low Element F1, before and after fine-tuning. Publication target (author, 2026-10-10, provisional): one conference thesis holding both, submitted after the Р.4 article; the article keeps gains by tier and one sentence. Thesis candidate R in `../theses/next-theses-probes/CANDIDATES.md` | ASSESSMENT §6 weeks 9–12 | — | |
| edit-readiness-rate | implementation | optional | share of predictions that compile and reach a stated structural threshold | ASSESSMENT §6 weeks 9–12 | — | |
| prototype | implementation | optional | demo: image in, PlantUML out, live render, edit | ASSESSMENT §6 weeks 9–12 | — | |
| benchmark-paths | implementation | core | every benchmark file the study reads is read from the public repository `../plantuml-recognition-evaluation`: code paths, the pin of the test-set rules, the six frozen records and both checksum lists re-frozen with the new path labels and code hashes | header; §9 | recorded 2026-10-15 (verified 2026-10-15, `verify.md`, third pass; the first two passes failed) | `decisions/benchmark-paths/` · `methodology/training-set-construction.md` §7 (re-freeze of 2026-10-14); paths and hashes in §1.1, §2, §7 |
