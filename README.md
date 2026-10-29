# Fine-tuning for image-to-PlantUML: Qwen3.5 at 2B, 9B and 27B

How far does fine-tuning move an open multimodal model family on real-world UML diagrams, and what
does the gain consist of: reliability (the output compiles) or structure (the output is right)?

This repository holds the plan, the code and the frozen data lists of a study that fine-tunes
Qwen3.5-2B, 9B and 27B with one fixed recipe and runs the published
[image-to-PlantUML benchmark](https://github.com/vovanrew/plantuml-recognition-evaluation) before
and after. The comparison is within the family: each size against itself, on the same 1,000 test
diagrams, with the same images, prompt, decoding and scorer in both conditions.

## Status

Work in progress, state of 2026-10-29.

- Done: scorer v2 is frozen; the training, dev and loss-curve sets are drawn and frozen; the image
  pipeline passes its acceptance check on the 1,000 test diagrams; the compute platform is
  frozen (tag `serving-v3`); the prompt arrangement, the training recipe and the checkpoint rule
  are frozen; the 2B pilot runs end to end; the three trainings (2B, 9B and 27B on the 20,000
  training pairs) and the 2B's 5,000-pair run are trained, each adapter picked on the dev set; the
  eight runs on the 1,000 test diagrams (the six main runs and two of the data-efficiency curve) are
  answered and stored; the six main runs are scored under scorer v2 and scorer v1, and the main
  tables are built; the error analysis of the six main runs is done.
- The two data-efficiency runs are scored and the 2B's curve (0 / 1k / 5k / 20k pairs) is built;
  its reading is not yet written.

[FINETUNE_PLAN.md](FINETUNE_PLAN.md) §10 lists every step with its status.

## What is here

| Path | Content |
|---|---|
| [FINETUNE_PLAN.md](FINETUNE_PLAN.md) | The plan: question, design, phases, item table |
| [methodology/](methodology/) | What was built and how: [scorer v2](methodology/scorer-v2.md), [training set construction](methodology/training-set-construction.md), [training recipe](methodology/training-recipe.md), [compute and serving](methodology/compute-and-serving.md), [analysis protocol](methodology/analysis-protocol.md) |
| `build/` | Training pool, near-duplicate check against the test set, sampling, image rendering, the user turn, the scorer v1 to v2 bridge, the scoring of the test-set runs and the main tables (`score_test_runs.py`, `main_tables.py`, the study's registry `test_registry.json`), the data-efficiency curve (`efficiency_curve.py`, its registry `curve_registry.json`), the error analysis (`error_analysis.py`, its codebook `error_codebook.md`), the release (`pack_runs.py`, `release_adapters.py`, the model card template in `release/`); tests in `build/tests/` |
| `infer/` | The server used for both conditions, the run manifest, the adapter check, and the runs on the test set; tests in `infer/tests/` |
| `train/` | The training recipe and the checkpoint rule; tests in `train/tests/` |
| `data/pool/`, `data/sets/` | Frozen outputs of `build/`, each folder with a `SHA256SUMS` list |
| `runs.zip` | Every test-set and dev-set run with its scores, the tables, and each training run's settings, step log and checkpoint choice (see [Release](#release)) |
| [LICENSE](LICENSE), [LICENSE-DATA](LICENSE-DATA), [CITATION.cff](CITATION.cff) | Licences of the code and of the data; how to cite |

## Data

- `data/sets/train.json` lists 20,000 training diagrams; its 5,000 and 1,000 subsets are ranks
  within it. `dev.json` and `loss_curve.json` list 200 diagrams each.
- A key is a source file name in the UML-in-the-Wild corpus
  ([10.5281/zenodo.18952371](https://doi.org/10.5281/zenodo.18952371)). The sources are not copied
  here.
- No diagram from a repository of the test set is in the pool, and candidates close to a test
  diagram are removed. `data/pool/d6_removed.jsonl` and `data/sets/neighbours_test.tsv` hold the
  evidence.
- The images are not in this repository (20,400 files, about 4 GB). Each one is pinned by a pixel
  hash in `data/sets/render/<set>/manifest.jsonl`.

## Release

| What | Where |
|---|---|
| Adapters, one repository per base model | Hugging Face: [Qwen3.5-2B-PlantUML-LoRA](https://huggingface.co/vovanrew/Qwen3.5-2B-PlantUML-LoRA) (20,000, 5,000 and 1,000 pairs), [Qwen3.5-9B-PlantUML-LoRA](https://huggingface.co/vovanrew/Qwen3.5-9B-PlantUML-LoRA), [Qwen3.5-27B-PlantUML-LoRA](https://huggingface.co/vovanrew/Qwen3.5-27B-PlantUML-LoRA) |
| Runs, scores and tables | `runs.zip` in this repository |
| Extractor JARs of scorer v1 and v2 | Assets of the fork's release [v1.2025.9-stats](https://github.com/vovanrew/plantuml/releases/tag/v1.2025.9-stats) |
| Training set | The key lists in `data/sets/`; the sources are in the corpus |

**Adapters.** Each repository holds the adapter the study served for its test-set runs, in a
subfolder per training size (`pairs-20000`, `pairs-5000`, `pairs-1000`), with a model card that
names the base model's revision and shows how to load it. They are under Apache-2.0, the licence
of the base models. `build/release_adapters.py check-hub` compares each file on the Hub with the
sha256 its test run recorded.

**Runs.** Unpack at the repository root:

```
unzip runs.zip
python3 build/pack_runs.py --check
```

`runs/test/<run>/` holds the run's answers, manifests, logs and scores (`scores/v2`, `scores/v1`);
`runs/test/_analysis/` holds the tables; `runs/dev/<run>/<checkpoint>/` holds the dev answers and
scores of every checkpoint; `runs/train/<run>/` holds `run.json`, `log.jsonl` and `selection.json`.
`runs/test/_root/` links each run into the layout the benchmark's loader reads; its links to the
test set and the test images resolve when the benchmark sits next to this repository (see
[Running the code](#running-the-code)). The tables rebuild from the stored scores with
`build/main_tables.py` and `build/efficiency_curve.py`. Re-scoring an answer needs the extractor
JAR of its scorer version, with the sha256 that `methodology/scorer-v2.md` §7 pins. Files inside
the runs keep the paths of the machine they were made on.

**Training set.** `data/sets/train.json` lists the 20,000 training diagrams by their key in the
corpus, and `data/sets/render/train/manifest.jsonl` the pixel sha256 of each image. The images are
not distributed: each is a render of a source that keeps its repository's licence. Rendering the
sources again with the pipeline of `methodology/training-set-construction.md` repeats 19,989 of the
20,000 images; the training command checks every image against the manifest and stops on the
other 11.

## Running the code

The code expects this repository next to its inputs, under one parent folder:

```
plantuml-recognition-fine-tuning/     this repository; the plan and methodology call it fine-tune/
plantuml-recognition-evaluation/      the benchmark: scorer, prompt, test set and its 1,000 images;
                                      data.zip unpacked; the two extractor JARs at its root
plantuml/                             the extractor fork, branch stats-extractor-graph
dataset/                              the corpus sources and the rendering scripts
herald/zenodo_content/                the corpus metadata and images
```

The acceptance check and some render tests also read the original renders of the test images,
`data/puml_images/` in the benchmark folder, which are not published.

It needs Python 3.12 with numpy, Pillow and pytest, a JDK, Graphviz and PlantUML 1.2025.9. The
versions used are recorded in `data/sets/render/train/render_env.json`.

```
python3 -m pytest build/tests -q
```

The tests of the user turn, the server, the manifest and the adapter check need the libraries of
[requirements-card.txt](requirements-card.txt), the environment of every run on the card; locally
the core packages listed in `build/card_environment.py` are enough.

Working records of each decision are kept outside this repository, so paths that start with
`decisions/` do not resolve here.

## Related

- Benchmark and scorer: [plantuml-recognition-evaluation](https://github.com/vovanrew/plantuml-recognition-evaluation), [10.5281/zenodo.21670236](https://doi.org/10.5281/zenodo.21670236)
- Corpus: UML-in-the-Wild, [10.5281/zenodo.18952371](https://doi.org/10.5281/zenodo.18952371)
- Extractor fork: [vovanrew/plantuml](https://github.com/vovanrew/plantuml)

## Licence

The code (`build/`, `infer/`, `train/`) is under the MIT Licence ([LICENSE](LICENSE)). The data and
documentation, `runs.zip` included, are under CC BY 4.0 ([LICENSE-DATA](LICENSE-DATA)). The
PlantUML sources that the key lists name, and any render of them, keep the licence of the
repository each source came from. The adapters are under Apache-2.0.
