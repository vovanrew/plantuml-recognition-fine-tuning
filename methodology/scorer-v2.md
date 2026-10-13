# Scorer v2

Methodological record for the structural scorer of the fine-tuning study. The metrics are defined in
`../plantuml-recognition-evaluation/methodology/evaluation-framework.md` at the tag `scorer-v1`, the
published methodology of the benchmark, called the baseline record below.
Paths are relative to `fine-tune/`, and the scorer repository is
`../plantuml-recognition-evaluation`.

## 1. What scorer v2 is

Scorer v2 computes Element F1, Relationship F1 and type accuracy for every output of this study. It
keeps the definitions of the baseline record (§2 to §4) and changes how a name and an edge are read
(§2 to §4 below). CSR and chrF++ are those of the baseline record (§1 and §5).

The rules share one reason. A model sees only the image, so a name or an edge is scored when the
image shows it. Element F1 then measures whether a model read a name, not whether it copied the
styling of the source. Styling is credited by chrF++ and the entity type by type accuracy. Every
rule is applied in the same way to the ground truth and to an answer.

Scorer v1 is the scorer of the benchmark, and every published number is a v1 number. It stays
runnable, and scoring is offline over stored answers, so any run can be scored under both versions
(§7). The stored runs are the eight registered runs of the benchmark under
`../baseline-evaluation/data/`: the seven published models and Claude Sonnet 4.6, a supplementary
run.

## 2. Names

A node is identified by its display name, compared after lowercasing, trimming and the removal of
stereotype tokens (baseline record, §2.1 and §3). Scorer v2 adds five rules, and §5 counts the
ground-truth names that each one changes.

**Whitespace.** The scorer collapses every run of whitespace inside a name to one space. The number
of spaces between two words cannot be read from an image. `Spring  Application`, typed with two
spaces, equals `Spring Application`.

**Drawn text.** A name is the text that PlantUML draws for the label. The extractor passes the label
through PlantUML's own markup parser, under the skin of the diagram, and keeps the text of the drawn
lines. PlantUML reads markup only when it draws, so a tag or an icon token stands in the source and
not in the image. An actor typed as the sprite `<$actor>` above `Payer` is named `Payer`. The
parser decides what is markup, so a generic such as `List<Pharmacy>` stays text.

PlantUML draws a label on one of three paths, by kind of element, and the name is read on the same
path.

| Element | How the label is read | Example |
|---|---|---|
| Class-like element | With the setting of PlantUML's class header: a pair of double underscores is text, and a leading `*`, `**` or `#` starts no list item | `client_new_connection_SYNC__NO_MSG__AT_MOST` keeps its underscores |
| Note, use case, rectangle-like element, circle, childless group that has a symbol | The label is cut at separator lines first. A separator line is a horizontal line that PlantUML draws, not text. Its title is text, unless it holds only separator characters | The lines `API Gateway`, `===`, `/api/v2/petstore` give `API Gateway /api/v2/petstore` |
| Every other label: sequence participant, object, map, childless package without a symbol | Whole, with every kind of markup on. A lone `==` line is the heading `=` that PlantUML draws | The lines `PkgTop`, `==`, `PkgBottom` of a childless package give `=`: the heading is the largest line |

A tip is a note on a class member. It keeps the name of scorer v1, its owner and side, as in
`NumbersStack$$$LEFT`.

**Largest line.** When the lines of a label differ in font size, the line or lines at the largest
size are the name. Sources generated from code write the name of a class at one size and its package
under it at a smaller one. `<b><size:14>ConquestMapPart</b>` above `<size:10>soen6441riskgame.enums`
is `ConquestMapPart`.

Text without a size of its own starts from the base size of the diagram's own style. That is
PlantUML's skin default for the element (class name 14 pt, participant 14 pt, note 13 pt), unless the
source changes it with `skinparam` or `<style>`. The largest line is then the line that PlantUML
draws largest in that diagram, in a ground-truth file and in an answer alike. `<size:10>Components::`
above the unsized `DimensionComponent` is `DimensionComponent`, because the default of 14 pt is the
larger size. The base size was checked for a class name, a participant and a note, each without an
override and with three. In 12 of 12 cases it equalled the size at which the stock renderer
(`../baseline-evaluation/plantuml-1.2025.9.jar`) drew the unsized line.

**Spot letter.** The letter of a stereotype spot is not part of the name. A spot such as
`<< (C,#ADD1B2) >>` sets the letter and the colour of the circle that PlantUML draws beside the name,
so the letter belongs to an icon. For `participant "Producer" as UpstreamProducer << (C,#ADD1B2) >>`
scorer v1 reads `C  Producer` and scorer v2 reads `Producer`.

**Unicode stereotype brackets.** `≪X≫`, written with U+226A and U+226B, is a stereotype token, as
`«X»` and `<<X>>` are, and is removed from the match key. A source can write a stereotype with these
two characters, and the reason of the baseline record (§2.1) holds for it unchanged.
`≪Factory≫ factory : RepositoryFactory` matches as `factory : RepositoryFactory`.

## 3. Edges

**Association direction.** An association with exactly one arrowhead is a directed edge from tail to
head. A plain line, and a line with an arrowhead at both ends, stay undirected. The image shows the
arrowhead as it shows a diamond, so a reversed, a dropped or an added arrowhead is a miss. For the
ground truth `Order --> Customer`, the answers `Customer --> Order` and `Order -- Customer` are
misses, and `Customer <-- Order` is a match. This rule supersedes the sentence of the baseline
record, §4, that an association is undirected and matches when drawn in either direction. The other
relations keep the direction rule of that section.

**Drawn edges.** An edge is counted when PlantUML draws it. The extractor leaves out every link and
every message that PlantUML marks as not drawn:

- a link that PlantUML adds itself to arrange three or more elements of a group that have no
  relationship;
- the link that PlantUML adds between the points of two association classes on the same pair of
  classes;
- a link or a message that the source writes as `[hidden]`, and a link to an element that the source
  hides or removes.

The image shows no line for these, so a model cannot read them. With them counted, an answer that
lists unconnected elements in the order of the source matches edges without reading a relationship.
Six unconnected interfaces in one package give 0 edges, where scorer v1 counts 5 associations. The
baseline record (§4) counts every link and every sequence message, and this rule narrows it.

The link of a tip to its owner stays an edge, because the image shows the pointer of the note to
the class member. Note counting is that of the baseline record (§2.1). The rule holds in every
diagram that the extractor reads as elements and links, and for the messages of a sequence diagram.
It removes no node, so Element F1 and type accuracy do not depend on it.

Every left-out edge was looked for in an SVG render of its diagram, made with the stock renderer
inside the v1 JAR of §7. None was drawn: 0 of 172 in the ground truth of the test set, 0 of 36 in
that of the dev set and 0 of 991 in the answers of the eight stored runs.

## 4. Answers with several diagrams

Element F1, Relationship F1 and type accuracy are computed on the first diagram of an answer that
holds several. PlantUML's online editor displays only the first diagram of such a file, and scorer
v1 keeps the last one. In the answer blocks under
`../baseline-evaluation/data/element_f1/<run>/pred_extracted/`, 6 of the 1,000 answers of Claude
Opus 4.6 held several diagrams, and no answer of the other 7 runs did. In one of the 6 (key
`819e9bd5`) the third diagram does not parse: Element F1 was 0.000 under v1 and is 1.000 on the
first diagram. No ground-truth file of the test set or the dev set holds several diagrams.

## 5. Ground truth under scorer v2

The counts come from the extractor, run with the v1 JAR and the v2 JAR of §7 over the ground-truth
sources of the two sets:

```
java -cp <jar> net.sourceforge.plantuml.stats.DiagramStatsExtractor --dir <folder of sources>
```

The test sources are `../baseline-evaluation/data/puml_files/` with the keys of
`../baseline-evaluation/data/test_set.json`, and the dev sources are
`../herald/zenodo_content/puml_files/` with the keys of `data/sets/dev.json`. The rows by rule are
from `build/audit_gt.py` and are not a partition.

| Count (a pair is nodes / diagrams or edges / diagrams) | Test set | Dev set |
|---|---|---|
| Diagrams; nodes | 1,000; 5,742 | 200; 1,097 |
| Nodes whose match key differs between v1 and v2 | 190 / 60 | 15 / 6 |
| By rule: drawn text differs from the typed label, counted on match keys | 144 / 42 | 4 / 2 |
| By rule: of these, a spot letter | 25 / 4 | 0 |
| By rule: the largest-line rule drops a line | 39 / 12 | 0 |
| By rule: Unicode brackets removed | 11 / 1 | 0 |
| By rule: a whitespace run collapsed, in the v2 name | 46 / 19 | 10 / 3 |
| Edges, with every link and message an edge → under scorer v2 | 9,437 → 9,265 | 1,797 → 1,761 |
| Edges left out | 172 / 39 | 36 / 12 |
| Left out: links that PlantUML adds between unconnected elements | 148 / 31 | 33 / 10 |
| Left out: links between two association-class points | 2 / 2 | 0 |
| Left out: links written `[hidden]` | 20 / 4 | 0 |
| Left out: messages written `[hidden]` | 2 / 2 | 3 / 2 |
| Links of tips, kept | 6 / 5 | 2 / 2 |
| Association edges | 1,014 → 862 | 182 → 149 |
| Association edges with exactly one arrowhead, with every link an edge | 586 of 1,014 (57.8%), in 153 diagrams | 91 of 182 (50.0%), in 22 diagrams |
| Diagrams without any edge | 97 → 107 | 18 → 19 |

The association edges left out of the test set carry no arrowhead. Its 862 association edges under
v2 have the arrowhead at the target in 471, at the source in 115, at neither end in 276 and at both
ends in 0. On a diagram without any edge an answer without edges scores 1 (baseline record, §4), so
the last row enters every macro Relationship F1.

**Audit of the dev set.** The ground truth of the dev set was audited before the freeze, because
checkpoints are selected on it. The extractor read all 200 diagrams without an error, 100 as class
and 100 as sequence diagrams. After the rules of §2 no name held markup, a spot letter or an
unpaired stereotype bracket, and no name had lost a word: nothing was left over.

## 6. From scorer v1 to scorer v2 on the stored runs

The eight stored runs were scored again with the two runners of §7, under scorer v2 and at every
cumulative level between v1 and v2 (`build/bridge.py`). `build/bridge_table.py` forms the
differences of the table below, and `build/spot_effect.py` gives its two spot-letter rows. The result files lie under
`../plantuml-recognition-evaluation/data_v2/`: `element_f1/<run>/` and `relationship_f1/<run>/` hold
scorer v2, and `_bridge/<level>/` holds each level, with its switches in `_bridge/levels.json`. The
v1 result files lie under `../baseline-evaluation/data/`.

The table gives micro F1 of the three dense Qwen sizes as full-set / compiled-only. Full-set is the
zeros-for-failed population of the baseline record (§3). Each row with a plus sign is the change
that one rule adds to the rows above it. The two rows "of which the spot letter" are scorer v2 with
every link and message an edge, minus the same with the letter put back into the names. Type
accuracy is over compiled answers.

| | Qwen3.5-2B | Qwen3.5-9B | Qwen3.5-27B |
|---|---|---|---|
| **Element F1**, scorer v1 | 0.2461 / 0.8785 | 0.4909 / 0.8444 | 0.6761 / 0.9170 |
| + whitespace | +0.0000 / +0.0000 | +0.0029 / +0.0050 | +0.0007 / +0.0009 |
| + drawn text, the spot letter included | +0.0000 / +0.0000 | +0.0034 / +0.0058 | +0.0087 / +0.0119 |
| &nbsp;&nbsp;of which the spot letter | +0.0000 / +0.0000 | +0.0007 / +0.0012 | +0.0031 / +0.0042 |
| + largest line | +0.0000 / +0.0000 | +0.0041 / +0.0071 | +0.0028 / +0.0039 |
| Element F1, scorer v2 | 0.2461 / 0.8785 | 0.5013 / 0.8623 | 0.6883 / 0.9337 |
| v2 minus v1 | +0.0000 / +0.0000 | +0.0104 / +0.0179 | +0.0122 / +0.0167 |
| **Relationship F1**, scorer v1 | 0.1142 / 0.4807 | 0.2007 / 0.4388 | 0.4852 / 0.7223 |
| + whitespace | +0.0000 / +0.0000 | +0.0003 / +0.0007 | +0.0004 / +0.0006 |
| + drawn text, the spot letter included | +0.0000 / +0.0000 | +0.0026 / +0.0056 | +0.0047 / +0.0070 |
| &nbsp;&nbsp;of which the spot letter | +0.0000 / +0.0000 | +0.0011 / +0.0025 | +0.0010 / +0.0015 |
| + largest line | +0.0000 / +0.0000 | +0.0011 / +0.0025 | +0.0006 / +0.0009 |
| + association direction | -0.0022 / -0.0093 | -0.0093 / -0.0203 | -0.0065 / -0.0098 |
| + drawn edges | +0.0022 / +0.0095 | -0.0030 / -0.0001 | -0.0025 / +0.0014 |
| Relationship F1, scorer v2 | 0.1142 / 0.4810 | 0.1924 / 0.4271 | 0.4819 / 0.7224 |
| v2 minus v1 | -0.0000 / +0.0002 | -0.0083 / -0.0117 | -0.0033 / +0.0001 |
| **Type accuracy**, v1 → v2 | 0.7585 → 0.7585 | 0.7561 → 0.7527 | 0.8863 → 0.8830 |

The rows left out are +0.0000 in every cell: Unicode brackets and first diagram for both metrics,
association direction and drawn edges for Element F1. On the association relation alone, over the
eight runs and full-set, the direction rule changed micro F1 by -0.1223 to -0.0183 and the
drawn-edge rule by -0.0464 to +0.0077.

**Rankings.** Full-set rankings of the seven published models are unchanged under v2 in all four
tables (Element F1 and Relationship F1, micro and macro). In the compiled-only tables three adjacent
pairs whose v1 intervals include zero change order
(`../baseline-evaluation/analysis/out/ci_table_pairwise.csv`):

- Element F1, macro: Qwen3.5-2B and Qwen3.5-9B;
- Relationship F1, micro: Qwen3.5-27B and Claude Opus 4.6;
- Relationship F1, macro: Qwen3.5-27B and Qwen3.5-397B-A17B.

No other pair changes order against v1, at v2 or at any level before it. The macro averages, the
other runs and the figures per relation are in the result files.

## 7. Frozen artifacts

| Artifact | Scorer v2 | Scorer v1 |
|---|---|---|
| Scorer repository, `../plantuml-recognition-evaluation` | tag `scorer-v2` on commit `f460b35` | tag `scorer-v1` on commit `8f85611` |
| Extractor fork `vovanrew/plantuml`, branch `stats-extractor-graph` (`../plantuml`) | commit `549fe8ec08` | commit `7d2292dab1` |
| Extractor JAR | `plantuml-stats-extractor-v2.jar`, sha256 `2747a72aec9206e424ddf9ed659f9eceb96730444a68b1fe99094317400f555b` | `plantuml-stats-extractor-v1.jar`, sha256 `f6e76d4973ce12c23cb15dff6420726f7d2203a80c1761d1454b5a2c5a91777b` |
| Result files of the stored runs | `../plantuml-recognition-evaluation/data_v2/` | `../baseline-evaluation/data/element_f1/`, `relationship_f1/` |

The message of a tag names the fork commit and the sha256 of the JAR, so one tag identifies the whole
scorer. The two JARs and `data_v2/` lie in the folder of the scorer repository and are not tracked
by git. The commit, the fork commit and the JAR of scorer v2 were fixed before any output of a
fine-tuned model was scored.

**The v2 JAR.** The file is the v1 JAR with the two classes changed on the fork,
`DiagramStatsExtractor` and `CreoleHorizontalLine`, recompiled and replaced in place
(`build/build_v2_jar.sh <fork checkout> <v1 JAR> <output JAR>`). It is not a
Gradle build, and the sha256 names this file, not the commit. Assembled twice from an export of
commit `549fe8ec08`, it was the same file both times (25,337,575 bytes). 2 of its 10,115 entries
differ from the v1 JAR.

**Runners and switches.** `evaluation/element_f1_runner.py` and
`evaluation/relationship_f1_runner.py` of the scorer repository score under v2 by default. Seven
switches, given to both runners, turn every rule off:

```
--no-collapse-whitespace --no-unicode-brackets --no-association-direction --last-diagram
--extractor-arg=--names=raw --extractor-arg=--no-arrowheads --extractor-arg=--undrawn-edges
```

With all seven the v2 code and the v2 JAR are scorer v1. With its three options the extractor
printed byte for byte the output of the v1 JAR on 17 of 17 folders (10,239 records). With all seven
the runners gave the per-diagram counts of the stored v1 result files in 16 of 16 files (16,000
rows).

Each result file records its rules, its extractor options and the sha256 of the JAR in a `scorer`
block. All 272 result files of `data_v2/` carry the sha256 of the v2 JAR. The test suite of
commit `f460b35` gave 543 passed, 1 skipped with that JAR (`python3 -m pytest -q`, with
`PLANTUML_EXTRACTOR_JAR` set to it and `data.zip` unpacked). The measurements ran under Java Temurin 26.0.1 and Python
3.9.6.
