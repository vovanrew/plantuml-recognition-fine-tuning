# Fine-tuning for image-to-PlantUML: Qwen3.5 at 2B, 9B and 27B

How far does fine-tuning move an open multimodal model family on real-world UML diagrams, and what
does the gain consist of: reliability (the output compiles) or structure (the output is right)?

This repository holds the plan, the code and the frozen data lists of a study that fine-tunes
Qwen3.5-2B, 9B and 27B with one fixed recipe and runs the published
[image-to-PlantUML benchmark](https://github.com/vovanrew/plantuml-recognition-evaluation) before
and after. The comparison is within the family: each size against itself, on the same 1,000 test
diagrams, with the same images, prompt, decoding and scorer in both conditions.

## Status

Work in progress, state of 2026-10-22. There are no results here yet.

- Done: scorer v2 is frozen; the training, dev and loss-curve sets are drawn and frozen; the image
  pipeline passes its acceptance check on the 1,000 test diagrams; the compute platform is
  frozen (tag `serving-v3`); the prompt arrangement, the training recipe and the checkpoint rule
  are frozen; the 2B pilot runs end to end.
- Not done: the three trainings and the evaluation runs.

[FINETUNE_PLAN.md](FINETUNE_PLAN.md) §10 lists every step with its status.

## What is here

| Path | Content |
|---|---|
| [FINETUNE_PLAN.md](FINETUNE_PLAN.md) | The plan: question, design, phases, item table |
| [methodology/](methodology/) | What was built and how: [scorer v2](methodology/scorer-v2.md), [training set construction](methodology/training-set-construction.md), [training recipe](methodology/training-recipe.md), [compute and serving](methodology/compute-and-serving.md) |
| `build/` | Training pool, near-duplicate check against the test set, sampling, image rendering, the user turn, the scorer v1 to v2 bridge; tests in `build/tests/` |
| `infer/` | The server used for both conditions, the run manifest and the adapter check; tests in `infer/tests/` |
| `train/` | The training recipe and the checkpoint rule; tests in `train/tests/` |
| `data/pool/`, `data/sets/` | Frozen outputs of `build/`, each folder with a `SHA256SUMS` list |

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

No licence file has been added yet. The PlantUML sources that the key lists name were mined from
open-source repositories, and each remains under the licence of its repository.
