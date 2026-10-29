# Analysis Protocol

Methodological record for the analysis of the main comparison of Р.4. It says how the six main runs
on the test set are scored, and how each number of the main tables is computed and read. §15 does
the same for the 2B's data-efficiency curve. The file holds
for this study what `../plantuml-recognition-evaluation/methodology/analysis-protocol.md` holds for
the benchmark; that file is called the baseline protocol below. The populations, aggregation,
stratification, per-relation counts, bootstrap, population gap and eligibility rule are those of
the baseline protocol, §1 to §6 and §10, except where a section below states otherwise. The
metrics are scored under scorer v2 (`methodology/scorer-v2.md`). Paths are relative to
`fine-tune/`, and the scorer repository is `../plantuml-recognition-evaluation`.

## 1. The comparison

The analysis compares each size with itself. The fine-tuned run of the 2B, 9B and 27B is compared
with the same size's zero-shot run, answered on the same stack
(`methodology/compute-and-serving.md` §10). The two runs of a size differ in the adapter alone, so
each difference measures the fine-tune. Every difference is fine-tuned − zero-shot and is written Δ
below. Sections 2 to 14 cover the six main runs. The 2B's two data-efficiency runs are scored by
the same command (§2), and §15 gives the curve they belong to.

## 2. Scoring

`build/score_test_runs.py` scores a run under scorer v2 and, alongside, under scorer v1, with the
benchmark's runners unchanged. Scorer v1 is the v2 code and the v2 extractor JAR with the seven rule
switches of `methodology/scorer-v2.md` §7, so one pinned scorer gives both versions. CSR and chrF++
do not depend on the scorer version and are scored once. Before scoring, the command checks the
pins of scorer v2 with `check_pins` of `train/dev_scoring.py`.

**Input contract.** The command scores a run stored as `methodology/compute-and-serving.md` §10.2
describes, and refuses any other input. The runners would score a missing answer as an empty one,
so a partial run must never reach them. The command refuses:

- a test set whose sha256 is not `e233e56968f90c7411d7c6b27cdd5c81f801fcc47fd1c7b45d174918a440744c`;
- an `answers/` folder without exactly 1,000 `.puml` files whose names are the test-set keys;
- a run manifest `answers/manifest.json` that is missing or does not pass;
- an `answer_manifest.json` whose `set_file_sha256` is not the frozen one;
- an existing score folder, unless the new scoring gives the same bytes in every result file.

**Answers the extractor cannot read.** An answer on which the frozen extractor fails is scored as an
empty graph, as a non-parsing answer is. It has no true or false positive, and every ground-truth
element and edge is a false negative. One rule for every unreadable answer keeps the frozen scorer unchanged
and treats both conditions alike. The extractor writes an empty record for an answer it cannot
parse. An answer that stops the extractor's Java process (a Java `Error`, such as a stack overflow)
is held back from that scorer version's F1 input. The unchanged runner then scores it as a missing
answer, which is the same empty graph. CSR and chrF++ still read the answer. The score manifest
lists each held-back answer under `extractor_crashes`.

**Output.** Scores go to `runs/test/<run>/scores/v2/` and `scores/v1/`, in the layout of the
benchmark's loader (`<metric>/<run>/<metric>_results.json`). `scores/manifest.json` holds the
sha256 of every result file and of the command. For each scorer version the command builds a data
root of links, `runs/test/_root/v2/` and `runs/test/_root/v1/`. Each root holds `test_set.json`,
the test pictures, `runs/<run>` (the run's `answers/`) and the metric folders, so the benchmark's
loader, leak gate and builders read it unchanged.

## 3. Registry and pairs

`build/test_registry.json` lists the six runs in the format of the benchmark's
`analysis/model_registry.json`. Each entry has `condition` (`zero_shot` or `fine_tuned`),
`arm: "qwen"`, an id `qwen3.5-{2b,9b,27b}-{zs,ft20k}` and the run name as `run_dir`. The fine-tune
results belong to this study, so the benchmark's registry and every output built from it stay as
published. The pairs of a zero-shot and a fine-tuned entry are fixed in `build/main_tables.py`
(`PAIRS`), so a pair never depends on the order of the registry.

`build/main_tables.py` loads the runs through the benchmark's loader and leak gate (baseline
protocol §10). It also refuses a run whose `runs/<run_dir>` holds other than exactly 1,000
predictions, because the leak gate counts a missing folder as zero leaks. It checks that the six
runs align diagram by diagram, and that every metric's compiled flag equals the CSR flag (baseline
protocol §1).

## 4. Populations

The full-set figures are pooled from the per-diagram rows as the frozen runners write them, as
Р.3's published figures were. No compilation mask is applied in the analysis. Р.4 compares each
size with itself, so it keeps the computation of Р.3. In these rows, an answer that does not render
but that the extractor parses keeps its parsed counts. The full-set figure therefore does not score
every failed answer 0, as the baseline protocol, §1, describes `zeros_for_failed`.

A supplementary table gives the figures with every answer that does not render (CSR flag false)
scored 0, beside the stored ones. It covers each run and each Δ: Element F1 and Relationship F1,
micro and macro, full-set, and their population gaps. The headline uses the stored figures.
Compiled-only is each run's own compiled diagrams (baseline protocol §1).

## 5. Intervals

Every interval is a paired percentile bootstrap (baseline protocol §5): seed 20260614, 1,000
resamples, the 2.5th and 97.5th percentiles (type 7). The constants and functions are imported
unchanged from the benchmark's `analysis/bootstrap.py`. One draw of diagram indices serves all six
runs and both scorer versions. Every Δ, difference of deltas and v2 − v1 contrast is therefore
paired by diagram. Each size has one training seed, so the intervals cover diagram variance only,
not the variance of training.

A statistic on an empty population is undefined, not 0, because an empty population has no score.
When the point or any resample of a cell is undefined, the cell gets a point only. An interval over
the defined resamples alone would hold only for the draws that happened to fill the population.

## 6. The headline: the decomposition of each size's gain

The headline of each size is three Δ under scorer v2, each with its interval:

- Δ CSR;
- Δ compiled-only Element F1 micro;
- Δ population gap Element F1 micro (gap = compiled-only − full-set, baseline protocol §6).

The three parts say whether the gain lies in compiling, in structure where the answer compiles, or
in which diagrams compile. Relationship F1 micro, with the same three parts, stands beside it. Δ
full-set follows from the identity Δ full-set = Δ compiled-only − Δ gap. The identity holds in the
point and in every resample, and `build/main_tables.py` checks it.

Both compiled counts are stated for each size. Δ compiled-only compares two different sets of
diagrams, the ones each run compiled, and is described as such.

## 7. Joint set and headroom

The joint set of a size is the set of diagrams that both its runs compiled. A sensitivity gives Δ
Element F1 and Δ Relationship F1 micro on the joint set, with intervals and the size of the set.
On the joint set, both runs are scored on the same diagrams.

The share of remaining headroom closed is Δ compiled-only / (1 − zero-shot compiled-only). It is a
point and descriptive, and it is undefined when the zero-shot compiled-only score is 1. It shows
whether a small compiled-only gain is small because little headroom was left.

## 8. Differences of deltas

A difference of deltas compares the gains of two sizes. The study reports 2B − 9B, 9B − 27B and
2B − 27B on Δ CSR, Δ compiled-only Element F1 micro and Δ gap Element F1 micro, with Relationship F1
beside. They come from the same draw as the Δ, so they are paired by diagram across the runs. The
three are linearly dependent (2B − 27B = 2B − 9B + 9B − 27B), and the tables state it. Three sizes
give a direction, not a fit: the differences say which size gains more, and no curve is fitted.

## 9. The claim family

The claim rests on a stated family of 12 intervals:

- Δ CSR and Δ compiled-only Element F1 micro for each size (6);
- the three differences of deltas of each of these two statistics (6).

The family is read at nominal 95% without correction. It is stated so that the reader knows how
many intervals the claim reads. Δ gap is shown with its interval but is not a separate member: it
moves with CSR and with compiled-only quality, which the family already measures. Every other
number is descriptive: Relationship F1, chrF++, type accuracy, macro figures, per-relation and
per-tier results, scorer v1 and the sensitivities.

## 10. Tables

**Main table.** Size × condition under scorer v2, overall and at the type and tier margins. It holds
CSR; Element F1 and Relationship F1 micro, full-set and compiled-only; chrF++ macro in both
populations; and type accuracy. Overall cells carry intervals. Margin cells and per-type Δ are
points; the tier Δ with intervals are those of §12.

**Scorer v1 companion.** Element F1, Relationship F1 and type accuracy under scorer v1, so v1
numbers stand beside v2 for both conditions (plan, Phase 0).

**Supplementary macro table.** Element F1 and Relationship F1 macro in both populations, with their
gaps and Δ. Beside macro Relationship F1 stands the same figure without the 107 diagrams whose
ground truth has no edges under scorer v2, for both conditions. On such a diagram, macro credits a
compiled answer that has no edges. The set is read from the ground truth: no true positive and no
false negative in the Relationship F1 row. `build/main_tables.py` checks that it is the same 107
diagrams in every run.

**Supplementary failed = 0 table.** As in §4.

## 11. Per-relation Relationship F1

Per-relation Relationship F1 micro is reported for the six relations (baseline protocol §4), before
and after, in both populations, with its Δ. It is descriptive. A relation gets intervals only when,
in the population of both runs, it has at least 100 ground-truth edges and at least 30 diagrams
that hold it. Below this floor, its run cells and its Δ are points, because few diagrams make the
interval unstable. An edge matches only when its relation and both endpoint names match. A rise in
a relation's F1 therefore does not isolate the reading of its decoration, and the per-relation
results carry no verdict on it.

## 12. Tiers

Δ per tier is reported on CSR and on full-set Element F1 and Relationship F1 micro. Its interval
comes from the global draw restricted to the tier's diagrams. A compiled-only tier Δ gets an
interval only when both runs compiled at least 30 diagrams of the tier; otherwise it is a point.
An empty population is undefined, not 0 (§5).

Δ(tier 4) − Δ(tier 1) is reported with its interval where both tier cells have one. A smaller gain
on tier 4 is worded as consistent with a limit set by the input, never as showing one. The tiers
differ in diagram size and in the crowding of the picture together (baseline protocol §9), so a
tier contrast cannot separate the two.

## 13. The scorer check of Phase 0

Plan Phase 0 expects scorer v2 to raise the zero-shot score at least as much as the fine-tuned one,
so that Δ under v2 is no larger than Δ under v1. The check is read on the joint set of each size
(§7), where both versions score the same answers on the same diagrams. The statistic is
Δ_v2 − Δ_v1 for Element F1 and Relationship F1 micro, with its paired interval:

| Interval | Reading |
|---|---|
| Upper bound below 0 | held: v2 makes the Δ more conservative |
| Lower bound above 0 | did not hold |
| Otherwise, an endpoint at exactly 0 included | no detectable change |

The full-set and compiled-only Δ under v1 stand beside those under v2, without a verdict. In these
populations, the v2 − v1 contrast moves with CSR as well as with the scorer.

## 14. Outputs and frozen code

`build/main_tables.py` writes `runs/test/_analysis/main_tables.json` and `main_tables.md`, and one
figure, `before_after_by_size.png`, which shows each size before and after with intervals. The
benchmark's master and CI builders may also run on the study's registry and data root, for tables
only. They then write to an `--out-dir` outside the benchmark's `analysis/out`, so its published
outputs stay as they are. No benchmark file is changed.

| File | sha256 |
|---|---|
| `build/score_test_runs.py` | `60cd57ac1635621b12bb7091f83215821077d7b0746fcc24410a7c97fce8a40a` |
| `build/main_tables.py` | `ac38533dd2bc66061c277f3bdb7af4b201a9c015218403c7a00984786a6abf12` |
| `build/test_registry.json` | `0cbe0cd550e8bd3a7ad2c00c5c693ee8aaf11dcb325cbd67b3fd64d83def3e2a` |

Each score manifest records the sha256 of the scoring command in `score_test_runs_sha256`.
`main_tables.json` records that of the analysis code in `meta.main_tables_sha256`.

## 15. The data-efficiency curve

The data-efficiency curve shows how the 2B's scores on the test set change with the number of
training pairs. It has four points: the zero-shot run (0 pairs) and the runs fine-tuned on 1,000,
5,000 and 20,000 pairs (`methodology/training-recipe.md` §6 to §8). The 0 and 20,000 points are the
2B's two main runs. The 1,000 and 5,000 points are the two data-efficiency runs, scored by the
command of §2 as the main runs are. The curve covers the 2B only.

**The variable.** The curve's one variable is the number of distinct pairs the recipe trains on.
The recipe trains 2 epochs at every run length, so the optimizer steps and the warm-up grow with
the pairs: 126, 626 and 2,500 steps (`methodology/training-recipe.md` §1.5). The curve therefore
reads as more pairs under the recipe, not as more data at a fixed amount of training. Each adapter
is its run's last checkpoint. Each point has one training seed, so its
intervals cover diagram variance only (§5).

**What each point shows.** Each point shows its number of compiled diagrams and the statistics of
the main table (§10) under scorer v2. These are CSR; Element F1 and Relationship F1 micro, full-set
and compiled-only, with their population gaps; chrF++ macro in both populations; and type accuracy.
Beside them stand the same six F1 figures under scorer v1, and the failed = 0 figures of §4. The
populations are those of §4.

**Differences.** A difference is named by the pairs of its two points, with 1k for 1,000. The
curve gives 1k − 0, 5k − 1k and 20k − 5k on every v2 statistic, each with its paired interval. The 20k − 5k difference on Element F1 and Relationship F1 micro is also given on the joint
set of the two runs (§7), with the size of that set. A compiled-only difference compares two
different sets of diagrams, as in §6, and the joint set compares the same diagrams.

**Intervals and reading.** Every interval is that of §5. The draw of diagram indices depends only on
the number of diagrams, the resamples and the seed, so it is the main comparison's draw. Every point
and difference is therefore paired by diagram with the main runs. Every number of the curve is
descriptive. None joins the claim family of §9, and each interval is read on its own at nominal 95%.
No statistic is fixed in advance to decide whether the gain still rises between 5,000 and 20,000
pairs, and no interval carries a verdict. The reading is written after the scores are seen, from the
full table, so that no number is selected before the reader sees it.

**Agreement with the main tables.** The 0 and 20,000 points must equal the main table's numbers,
so that each run has one value per statistic in Р.4. Before it writes, `build/efficiency_curve.py`
compares these two points and the 2B's headline Δ (§6) with `runs/test/_analysis/main_tables.json`,
cell by cell and exactly. It refuses on any difference.

**Registry, code and outputs.** `build/curve_registry.json` holds the six entries of
`build/test_registry.json`, unchanged and in order, because the loader of `build/main_tables.py`
requires every pair of §3. It adds the two data-efficiency runs as `qwen3.5-2b-ft1k` (`run_dir`
`2B_1000`) and `qwen3.5-2b-ft5k` (`2B_5000`), with `condition: fine_tuned`.
`build/efficiency_curve.py` loads the eight runs with the loader, leak gate and checks of
`build/main_tables.py` (§3). It computes every statistic and interval with that file's functions,
imported unchanged. The main tables therefore hold the six main runs only, and the files and
sha256 of §14 stay as recorded.

```
python3 build/efficiency_curve.py
```

The command writes `runs/test/_analysis/efficiency_curve.json` and `efficiency_curve.md`. The json
records in `meta` the sha256 of the script, of `build/main_tables.py`, of the registry and of the
published `main_tables.json`.

| File | sha256 |
|---|---|
| `build/efficiency_curve.py` | `dfe8845248120032ac458c75e8c806530b10a65b422bddfce41b57d866775b05` |
| `build/curve_registry.json` | `528fc4b1095eea2b5786e7dcd4c10d24fd3fc956e8678efed4a3e7b36b8c3f12` |
