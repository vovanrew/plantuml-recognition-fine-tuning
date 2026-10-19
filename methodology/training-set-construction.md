# Training Set Construction

Methodological record for how the training and dev sets of the fine-tuning study are built. The
rules of the test set are in `../plantuml-recognition-evaluation/methodology/test-set-construction.md`, called
the baseline record below. Paths are relative to `fine-tune/`. §1 records the rendering of images;
§2 to §8 record the pool of candidates, the sets drawn from it and their frozen artifacts.

## 1. Image rendering

The settings (DPI, sizes, heap, threads, batch size) are constants of the code named in §1.1.

### 1.1 The command

One command renders PlantUML sources to images:

```
python3 build/render.py --src-dir <sources> --keys <key list> --out <folder>
```

It runs the three stages that produced the benchmark's test images. The reason is the rule of
`FINETUNE_PLAN.md`, Phase A: training images come from the same rendering pipeline as the test
images.

| Stage | What it does | Code that runs |
|---|---|---|
| 1 | PlantUML 1.2025.9 renders each source to a PNG, with `skinparam dpi 300` passed through `-config`, `PLANTUML_LIMIT_SIZE=16384` and `--no-error-image` | the java command of `generate_images_hidpi.sh`, built by `plantuml_command()` |
| 2 | Pads any side below 512 px on a canvas of the colour of pixel (0, 0), and converts to RGB | `postprocess_one` of `postprocess_images.py` |
| 3 | Flattens transparency on white and downscales the long edge to 1,568 px with Lanczos, never upscaling | `standardize` of `standardize_images.py` |

Stage 3 applies the working standard of the baseline record, §7. The image after stage 2 is called
the raw image, and the image after stage 3 the final image.

Stages 2 and 3 run the code of the two original files, loaded by path. Stage 1 runs the java
command of `generate_images_hidpi.sh`, called the stock script below. `render.py` sets only the heap
size and the thread count of that command. Its defaults are a heap of 32,768 MB, 3 threads and
batches of 200 sources. The renderer JAR and the stage 2 and stage 3 files are pinned by sha256, and
a file that differs stops the run.

| File | Role | sha256 |
|---|---|---|
| `build/render.py` | the command | `a23d0f29e47564ed7d1bf2cefb195e4ff79a56429b384b815c728b8f1b9b85f0` |
| `build/dot_guard.sh` | script between PlantUML and Graphviz `dot` (§1.3) | `5d2209008f64d362c55a6b78da443d169a484b3ed0f7d4771ff9767f14ff9113` |
| `build/acceptance_check.py` | acceptance check (§1.5) | `e53e04bfe851d68d47b34e7df290b2fc34a7027104825379bf4b58acba4509a7` |
| `../dataset/plantuml-in-wild-dataset/phase2/4.generate_images/plantuml-1.2025.9.jar` | renderer, pinned | `280f4aa97f4f3b7af39358dd0d79f65e353b54a1630b5f121a13b1651162185d` |
| `../dataset/plantuml-in-wild-dataset/phase2/4.generate_images/generate_images_hidpi.sh` | the stock script; `render.py` does not run it, and a test compares its java command with `plantuml_command()` | `68722e410a71b125fb91663c4505362a4ef544a03f8a1a5574b39fef5b300e0e` |
| `../dataset/plantuml-in-wild-dataset/phase2/5.postprocess_images/postprocess_images.py` | stage 2, pinned | `53742a3d8ec4414bbc4829a3b5d5c92f21be5754ffdafb2bb5c6fb8925c4b3fc` |
| `../plantuml-recognition-evaluation/util/standardize_images.py` | stage 3, pinned | `726d80a77092e817d517807e77043c61a55f2e9b659631b8a3f01aff21b8caf3` |

`render.py` writes only under its output folder. Final images go to `puml_images_1568/`, and raw
images to `puml_images/` when `--keep-raw` is given. The folder that holds the test images,
`../plantuml-recognition-evaluation/data`, is refused as output folder.

### 1.2 Framework and outcomes

The command works within a framework of one diagram per source file. For a source with more than
one diagram block its behaviour is not determined. None of the 1,000 test sources has more than one
block.

Within this framework a source gets an image only if it compiles and renders to exactly one page
(the one-image rule of the baseline record, §4). Every key gets one line
in `manifest.jsonl` with its status, and only a key with status `ok` has an image. A key is the file
name of the source (baseline record, §2).

| Status | Outcome |
|---|---|
| `ok` | one image, written to `puml_images_1568/` |
| `multi_page` | the source rendered to more than one page |
| `no_image` | PlantUML wrote no image under the key's name; the line gives the reason (`syntax_error`, `no_diagram`, `named_output`, `unknown`) |
| `crash` | the renderer failed on the diagram when it was rendered alone (§1.3) |
| `timeout` | the render of the diagram alone did not end within the time limit |
| `missing_source` | the source file is not in the source folder |
| `postprocess_error` | stage 2 or stage 3 raised an error |

`summary.json` holds the count of keys per status. The command checks that the image folder holds
exactly the images of the `ok` keys.

### 1.3 What stage 1 adds to the stock script

PlantUML answers some failures with a picture in place of the diagram, and the stock script counts
the file as rendered. `render.py` adds three checks for this reason.

**Crash picture.** When PlantUML runs out of memory or overflows its stack, it writes a "PlantUML
has crashed" picture in place of the diagram. This crash picture is a valid PNG. `render.py` tests
every stage 1 output for it: `is_crash_image()` compares two fixed regions of the image with the
hashes in `CRASH_MARKS`. A diagram that got the picture is rendered again alone, and dropped with
status `crash` if it gets the picture again. If the renderer exits abnormally, times out, logs
more fatal errors than there are crash pictures, or leaves a PNG under a name no key has, every
diagram of the batch is rendered again alone.

**Graphviz `dot` behind a script.** PlantUML does not report a `dot` that ends without output. It
draws the diagram in another layout mode or, if `dot` fails again, a failure picture in the
diagram's own scale and colours. `build/dot_guard.sh` therefore stands between PlantUML and `dot`.
It hands every call and its output on unchanged, and prints one error line when `dot` returns no
output. PlantUML then draws the crash picture, and the rule above applies.

**Graphviz present.** Before it makes an output folder, the command runs PlantUML's Graphviz test
(`-testdot`) and stops if the test fails. Without Graphviz a class diagram renders to a notice that
is a valid PNG.

### 1.4 What is recorded with every render

`render_env.json` in the output folder records the versions and settings that can change a pixel:

| Group | Fields |
|---|---|
| Renderer | path, sha256 and version of the JAR |
| Environment | the Java version, the Graphviz version as PlantUML reports it, the Pillow version, the platform |
| Graphviz script | the sha256 of `dot_guard.sh` and the path of the `dot` behind it |
| Stages 2 and 3 | the sha256 of both files |
| Parameters | DPI, size limit, minimum side, long edge, the flags of the java command |
| Each run | start time, threads, heap, batch size, the sha256 of `render.py` |

A run into a folder that already holds a manifest renders only the keys without a line. It stops if
its environment differs from the recorded one, so one image set comes from one environment.

The manifest line of an `ok` key holds the size and sha256 of the stage 1 PNG, the stage 2 action
and the size of the raw image. It also holds the size and the pixel sha256 of the final image, and
says whether the diagram was rendered in its batch or alone. When PlantUML logs an exception for a
diagram and still draws it, without the element that raised the exception, the diagram is kept. The
exception is recorded on its line (`renderer_trace`).

### 1.5 Acceptance check on the 1,000 test diagrams

`build/acceptance_check.py` renders the 1,000 test diagrams (baseline record, §1) through
`render.py` and compares every image with the benchmark's. It passes when all diagrams are rendered
and every raw and final image has the pixel dimensions of the benchmark's image. Equal dimensions
are the criterion `FINETUNE_PLAN.md` sets for this check (Phase A). The check passes, and its
report gives the counts below:

```
python3 build/acceptance_check.py --work <scratch> --report <file> --stock
```

All 1,000 final images and all 1,000 raw images have the pixel dimensions of the benchmark's images.

| Count of test diagrams | All | Class | Sequence |
|---|---|---|---|
| Rendered with status `ok` | 1,000 | 500 | 500 |
| Final image identical to the benchmark's, pixel for pixel | 775 | 284 | 491 |
| Raw image identical to the benchmark's, pixel for pixel | 774 | 283 | 491 |

No test diagram was rendered again alone (0 of 1,000), and 1 carried a logged exception.

**Size of the differences.** 225 final images differed from the benchmark's. Their largest
difference in one colour channel, on the scale 0 to 255, was at most 16 in 218 images, 17 to 64 in 5
and above 64 in 2. The share of differing pixels was below 0.05% in 182 of them and at most 0.6361%.
The 226 raw images that differed held a median of 72 differing pixels and at most 8,136.

**Stage 3.** Applied to the benchmark's raw images, stage 3 gave the benchmark's final images on
1,000 of 1,000.

**Stage 1 against the stock script.** The stock script's PNG has the sha256 that `render.py`
records for its own stage 1 output on 999 of the 1,000 test diagrams. The other diagram prints the
time of the render.

### 1.6 Environment of the renders

Every render records its environment in `render_env.json` (§1.4). The three sets
(`data/sets/render/<set>/render_env.json`) and the acceptance check were rendered in this one:

| Component | Version |
|---|---|
| PlantUML | 1.2025.9 (the JAR of §1.1) |
| Java | Temurin 26.0.1+8 |
| Graphviz | 16.1.0 (20260904.0139) |
| Pillow | 11.3.0 |
| Python | 3.9.6 |
| Operating system | macOS 26.4 arm64 |

## 2. The training pool

`build/pool.py` builds the pool, and `build/sample.py` draws and renders the sets; §7 lists both
files with their sections. The counts of §2 and §3 are those of `data/pool/funnel.json`, and those
of §4 to §8 of `data/sets/counts.json` and `data/sets/d8_report.json`.

Four rules build the pool. The table below gives the order of the steps. A diagram that has passed
the steps so far is called a candidate.

**Rule 1: the stages of the test set, then its repositories removed.** The test set's own code
(`../plantuml-recognition-evaluation/util/build_test_set.py`) runs its stages over the whole corpus: inclusion, degeneracy labels,
deduplication and type agreement (baseline record, §3, §4, §8 and §9). Type agreement uses the
extractor JAR that typed the test set (§7). Every diagram of the 815 repositories of the test set is
removed after these stages, 33,690 diagrams (baseline record, §6). This is the code path of the test set, and with
the repositories removed last no candidate has the normalized code of a test diagram.

**Rule 2: degeneracy labels from the published images.** The labels of the baseline record, §4, are
defined on the 300 DPI image, and the corpus is published with 96 DPI images. The labels are those
of the test set's code: more than one page, whatever the cause, and the aspect ratio of the padded
image. The page count is
read from the published images, and the size of the raw image (§1.1) is predicted as
`max(round(3.125 × side), 512)` px per side. Where the prediction is not safe, the diagram is
rendered with the command of §1.1 and labelled on its raw image. The conditions are in
`data/pool/config.json` (`D2`). Of the 106,348 included diagrams 1,853 were rendered. All 598 diagrams labelled as clipped got the label
from a render. Three class diagrams of tier 4 that the test set's archive kept against this rule
carry a label here. The class counts from this step on are therefore lower than those of the
baseline record.

**Rule 3: one diagram block, read as its labelled type.** The pool is defined for a source that
holds exactly one diagram block and that the scorer reads as its labelled type. Blocks are counted
by PlantUML's own rule for the start of a block. The block is isolated as the scorer isolates it
(`extract_puml` of `../plantuml-recognition-evaluation/evaluation/csr_runner.py`), and the extractor runs on it with the network closed. A
candidate is kept when the extractor reports no error and its `diagram_type` equals the label
`primary_type`. For a source outside this framework the rules that follow are not determined, and
it is not a candidate. As ground truth such a file is an empty graph, or it depends on what a URL
returns on the day.

**Rule 4: token cap.** A candidate whose target is longer than 5,376 tokens is dropped: 5,376
tokens is the output cap of `FINETUNE_PLAN.md`, Phase A. The length counted is that of the
diagram block as rule 3 isolates it. It is counted with the tokenizer of `Qwen/Qwen3.5-2B`, with
the settings recorded in `data/pool/config.json` (`D4` and `environment`). On the 1,000 test files this
count equalled the count stored in `../plantuml-recognition-evaluation/data/gt_token_stats.json` in every
case.

| After step | Class | Sequence | Total |
|---|---|---|---|
| Inclusion | 71,323 | 35,025 | 106,348 |
| Degeneracy labels (rule 2) | 70,474 | 34,641 | 105,115 |
| Deduplication | 68,456 | 33,189 | 101,645 |
| Type agreement | 68,329 | 32,880 | 101,209 |
| Test repositories removed | 43,234 | 24,285 | 67,519 |
| Rule 3 | 43,200 | 24,230 | 67,430 |
| Rule 4 | 43,184 | 24,226 | 67,410 |
| Near-duplicates of the test set (§3) | 42,492 | 23,887 | 66,379 |

Every extractor run of the build has the network closed, so the pool does not depend on what a URL
returns. At the type-agreement step 42 sources that read a URL therefore got no type and stayed,
and rule 3 removed all 42. The candidates after the last step are called the pool: 66,379 diagrams
of 21,190 repositories, 64.01% of them class diagrams.

## 3. Near-duplicates of the test set

The last step of §2 removes every candidate that is a near-duplicate of a test diagram.

**Texts compared.** The comparison text is the normalized code of the deduplication step (baseline
record, §8) without the lines that begin with `@startuml` or `@enduml`. Every diagram
holds these lines, and they are dropped in any letter case.
The text is taken twice: from the whole file, and from the diagram block alone as rule 3 isolates
it. Both sides of a pair are the same kind of text. The block is added because PlantUML does not
draw text outside it: for 1,213 of the 66,379 pool diagrams the two texts differ. With both texts
the rule holds for the file and for the diagram that the model sees.

**Measures.** A candidate is a near-duplicate when one of three measures reaches its threshold
against any of the 1,000 test diagrams, on either text.

| | Measure | Threshold | What it finds |
|---|---|---|---|
| (a) | Jaccard index of the two sets of character 5-grams | ≥ 0.7 | a copy with a line edited |
| (b) | edit similarity on characters, 2·LCS(a, b) / (\|a\| + \|b\|), where LCS is the longest common subsequence | ≥ 0.8 | a copy that was retyped or spaced differently |
| (c) | share of the distinct lines of a test diagram that the candidate holds; only for a test diagram of at least 10 distinct lines, a lone `{` or `}` not counted | ≥ 0.8 | a test diagram inside a larger candidate |

**What the rule removed.** Every measure is computed exactly for every pair. On the whole file the
rule compared 67,410 candidates with the 1,000 test diagrams and removed 1,031. On the block alone
it compared the 66,379 pool diagrams with them and flagged 4 more, all sequence diagrams of tier 1.
These 4 stay in the pool file and are excluded from every draw of §4.

**What remains.** Among the candidates that were not flagged, the largest values against a test
diagram were 0.697959 for (a), 0.799465 for (b) and 0.769231 for (c), on both texts.

**Stated consequence.** The rule removes most of the small class diagrams written by the SketchIt!
plugin, a kind of diagram that the test set contains. They are identified by the plugin's footer
text or its title form. The test set holds 5 of them among the 125 diagrams of class tier 1. In the
same tier the candidates held 367 of them before this step. The rule removed 314 (85.6%), and 53
remain among the 7,691 pool diagrams of the tier (0.69%). By the footer text alone, the count
stored in `data/pool/funnel.json`, the four figures are 4, 350, 314 and 36.

## 4. Sampling

The sets copy the design of the test set: equal cells, where a cell is one diagram type in one tier
(baseline record, §5.3), eight cells in all. The tier boundaries are those of the test set and are
not computed again on the pool. Every draw follows the sampler of the test set, `sample_cells` of
`build_test_set.py`: a seeded permutation of each cell, read in order (baseline record, §10). The
three lists equal what that function itself gives with the excluded keys passed over. Every draw
starts its own generator with seed 42. The sets are drawn in the order
below.

**Dev set.** The dev set holds 200 diagrams, 25 per cell, drawn from the pool with the test set's
limit of 5 diagrams per repository. It is drawn first, so it is frozen before the training list
exists, by the procedure of the test set. Its 195 repositories are reserved: their 2,682 pool
diagrams (4.04% of the pool) cannot enter the training sample.

**Loss-curve split.** The split holds 200 diagrams, 25 per cell, drawn in the same way from the
2,482 reserved diagrams outside the dev set. It is the validation split for loss curves
(`FINETUNE_PLAN.md`, Phase A). Drawn from reserved repositories, it costs the training pool
nothing.

**The rule of §3 against the dev set.** The rule is run again between the 63,697 pool diagrams
outside the dev repositories and the 200 dev diagrams. On the whole file it removed 51
candidates. The 63,646 diagrams left, of 20,986 repositories, are the training pool. On the block
alone the rule flagged 2 more candidates, which are excluded from the training draw. Among the
candidates that were not flagged, the largest values against a dev diagram were 0.656863,
0.798742 and 0.727273, on both texts.

**The rule of §3 against the loss-curve split.** The rule is also run, on both texts, between the
training pool and the 200 diagrams of the loss-curve split. A loss curve on the split is then read
on diagrams the model has not seen. The rule flagged 47 candidates, 46 of them not excluded
before, and they are excluded from the training draw. Among the candidates that were not
excluded, the largest values against a loss-curve diagram were 0.602362, 0.795556 and 0.774194,
on both texts.

**Training sample.** The sample holds 20,000 diagrams, 2,500 per cell, drawn from the training pool
with the same sampler and without a limit per repository (§8). The 5,000 and 1,000 subsets are the
first 625 and the first 125 diagrams of each cell's order, so the 1,000 lie inside the 5,000.

**Exclusion list.** A key that must leave a set changes no candidate list and no permutation. It
goes on an exclusion list, the sampler passes it over, and the next key of the cell's order enters
at the last rank. In the training draw the other keys keep their order, so the subsets stay
nested. The keys replaced in this way are in §5.

## 5. Rendering of the sets

Every diagram of the three sets is rendered with the command of §1.1 at its defaults. A key is
dropped, and replaced through the exclusion list of §4, when its status is not `ok` or when its raw
image carries a degeneracy label. The dev and loss-curve diagrams are rendered and their targets
counted before their lists are frozen. All 20,400 keys of the final lists had status `ok` and no
degeneracy label. Twenty training keys were replaced. One had a render with status `crash`. The
rule of §3 flagged 2 on the block alone against a dev diagram and 17 against a loss-curve diagram
(§4). No dev or loss-curve key was replaced.

A clock source prints the time of the render (`%date`). A diagram with a logged exception is drawn
without the element that raised it (§1.4). Both kinds are kept: the test set holds one of each
(§1.5), and dropping them would be a filter the test set did not have.

| Set | Repositories | Largest number from one repository | Clock sources | Logged exceptions |
|---|---|---|---|---|
| Dev set | 195 | 2 | 0 | 1 |
| Loss-curve split | 86 | 5 | 0 | 0 |
| Training sample | 10,158 | 56 | 14 | 25 |
| 5,000 subset | 3,748 | 17 | 3 | 9 |
| 1,000 subset | 920 | 4 | 1 | 1 |

## 6. What the sets guarantee

The statements below were measured on the final key lists.

- **Repositories.** The test set, the dev set and the training sample share no repository, pair by
  pair. All 86 repositories of the loss-curve split are dev repositories.
- **Keys.** No key is in two of the three sets, and no key of the test set is in a set.
- **Near-duplicates.** No measure of §3 reaches its threshold for a key of the three sets against a
  test diagram. None does for a training key against a dev diagram or a loss-curve diagram. Both
  hold on either text. The loss-curve split is not compared with the dev set: both come from the same
  repositories, and 12 of its 200 diagrams reach a threshold against a dev diagram.
- **Images.** No image of the three sets is pixel-identical to one of the 1,000 test images, and no
  dev image to a training image.
- **Targets.** Every target is within 5,376 tokens.

No threshold separates a copy from a look-alike. The nearest training diagram of every test diagram
and of every dev diagram is therefore stored, for each measure and each text. With these tables the
results of the evaluation can be computed again without the test diagrams that have a close
training neighbour. The table counts the diagrams by the value of their nearest diagram of the
20,000.

| Nearest training diagram | Whole file: ≥ 0.5 | ≥ 0.6 | ≥ 0.7 | Block alone: ≥ 0.5 | ≥ 0.6 | ≥ 0.7 |
|---|---|---|---|---|---|---|
| Test diagrams, (a) | 36 | 15 | 0 | 36 | 16 | 0 |
| Test diagrams, (b) | 351 | 137 | 55 | 350 | 137 | 56 |
| Test diagrams, (c) | 22 | 8 | 1 | 22 | 8 | 1 |
| Dev diagrams, (a) | 4 | 2 | 0 | 4 | 2 | 0 |
| Dev diagrams, (b) | 78 | 35 | 12 | 79 | 35 | 12 |
| Dev diagrams, (c) | 5 | 2 | 1 | 5 | 2 | 1 |

Measure (c) has a value for 791 test diagrams and 155 dev diagrams on the whole file, and for 789
and 154 on the block alone. No value reaches 0.8. Inside the 20,000 the largest values were
0.697959, 0.799443 and 0.705882 against the test set, and 0.656863, 0.798742 and 0.727273 against
the dev set. Against the loss-curve split they were 0.561224, 0.795506 and 0.741935. Each holds on
both texts.

## 7. Frozen artifacts

The freeze is two lists of sha256 values, `data/pool/SHA256SUMS` (13 files) and
`data/sets/SHA256SUMS` (29 files). Both verified with
`shasum -a 256 -c`. The code is under `build/`, with its tests under `build/tests/`.

The two lists name every file. The table gives the files a reader needs first.

| Artifact | Files |
|---|---|
| The pool | `data/pool/pool_after_d6.json` |
| Key lists, each key with its cell and its rank in the cell | `data/sets/dev.json`, `loss_curve.json`, `train.json`; the subsets are ranks 1 to 625 and 1 to 125 |
| Keys removed or excluded, with the reason | `data/pool/d3_excluded.jsonl`, `d6_removed.jsonl`; `data/sets/d7_removed.jsonl`, `block_excluded.jsonl`, `loss_curve_excluded.jsonl`, `dropped.jsonl` |
| Nearest training diagrams (§6) | `data/sets/neighbours_test.tsv`, `neighbours_dev.tsv` (whole file); `neighbours_test_block.tsv`, `neighbours_dev_block.tsv` (block alone) |

| File | sha256 |
|---|---|
| `data/pool/SHA256SUMS` | `76a150ab091fab652f5112923462bb2043b332cfd8beb7922002aefc9b15eeaf` |
| `data/sets/SHA256SUMS` | `2d3f1b5101826a6092c81311816228a2d287075c22932c3ac45ba5f06a90f96a` |
| `build/pool.py` (§2, and §3 on the whole file) | `cc7f9bd16e47f8e0c7ac9f06714bb46739fc684cc4adb6f0d81a90cd4cd59980` |
| `build/leakage.py` (the measures of §3) | `d71399b23527b5541c5c0afdab9eb471bed7efd4108bc1c516252269c79fb824` |
| `build/sample.py` (§3 on the block alone, §4 to §6) | `b9ed9c982ca69bdabcc63a9269b9c5b1198cb8c9d5b376999f2f3b66514ce35b` |
| `../plantuml-recognition-evaluation/plantuml-stats-extractor-v1.jar` (the extractor of rules 1 and 3) | `f6e76d4973ce12c23cb15dff6420726f7d2203a80c1761d1454b5a2c5a91777b` |
| `../plantuml-recognition-evaluation/util/build_test_set.py` (the test set's code of rule 1 and §4, pinned in `build/pool.py`) | `4be04fadf02b87f94a54591612b58a31f24adbb7ad0986b1e27abf635ec54d53` |

**Images.** The images lie under `data/sets/render/<set>/puml_images_1568/` and are not in the two
lists. Each image is fixed by the pixel hash (`pixels_sha256`) of its line in the render manifest of
its set (`data/sets/render/<set>/manifest.jsonl`, §1.4), which the lists hold. The training images
are these stored files.

The training folder holds the images of the 20,000 keys of `train.json` and no other. The 19
images of the keys replaced under the rule of §3 (§5) were moved to `data/replaced_images/train/`
after the verification, and their lines stay in the manifest. The render command checks a folder
against its manifest, so it runs on this folder again only with the 19 put back.

**The code that built the data.** The data were built by commit `5b91797` of the study repository.
The files hashed here differ from it only in the lines that name the benchmark folder and in two
pins: those of the test set's code and of the scorer-v2 extractor. No key list, manifest or image
depends on them. The copy of the test set's code in the table above is the public repository's. It
differs from the copy the build ran only in the defaults `DATASET` and `EXTRACTOR_JAR`, which only
the file's own `main()` uses. The other benchmark files the code reads have the same bytes in both
copies.

## 8. Rules that differ from the test set's

The plan's rule is that the training pool is built by the construction rules of the test set
(`FINETUNE_PLAN.md`, Phase A). Two rules of this record differ from the test set's.

**Filters applied to the pool only.** Rules 3 and 4 of §2 and the near-duplicate rule of §3 and §4
are not applied to the test set. Rules 3 and 4 would remove no test diagram: all 1,000 are inside
the framework of rule 3, and the longest test target has 3,546 tokens. The near-duplicate rule
compares a candidate with the test set, the dev set and the loss-curve split, so it can apply to
the pool only. Its effect
on one kind of diagram is the consequence stated in §3.

**No limit per repository in the training sample.** The dev set and the loss-curve split keep the
test set's limit of 5 diagrams per repository (baseline record, §6), and the training sample has
none. In the sample 513 repositories give more than 5 diagrams, 4,990 of the 20,000 diagrams. A limit of 5
would cut into them and lower the share of projects with many diagrams. The test set's limit is 5
diagrams per repository in a set of 1,000. In the same proportion it is 100 in a set of 20,000, and
the largest repository gives 56.
