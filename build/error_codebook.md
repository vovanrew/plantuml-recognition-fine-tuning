# Coding instructions: failure cases of the test-set runs

These are the instructions every coder follows for the error analysis of the six main test-set runs
(`build/error_analysis.py`; `methodology/analysis-protocol.md` §16). They are frozen before the first
case is coded: `build/error_analysis.py draw` records their sha256, and `report` refuses a codebook
that differs.

## What you see

Each case is one failed answer of an image-to-PlantUML model, in a file `cases/<id>.md`:

- the input image the model was shown (`cases/<id>.png`);
- the ground-truth PlantUML the image was rendered from;
- the model's answer. Its PlantUML block is shown with display-only lines removed: `skinparam`,
  `title`, comments, `hide`/`show`, layout directives (`left to right direction`, `top to bottom
  direction`, `scale`) and preprocessor lines (`!define`, `!include`, …). The removed lines are in
  `cases/<id>.stripped.txt`. Open that file only for a compile failure whose cause the shown
  answer does not explain; say so in `opened_stripped`;
- the outcome: **compile fail** (the answer does not render) or **compiled low-structural** (it
  renders, but per-diagram Element F1 or Relationship F1 is below 0.5);
- for a compile fail, the renderer's message and, where it names a line, that line's text;
- for a compiled low-structural case, the structural diff: matched, missed and spurious node names,
  and matched, missed and spurious edges, as scorer v2 keys them (names lower-cased and normalized;
  an edge is `source -> target : relation`).

You are not told which model, which size or which condition (before or after fine-tuning) produced
the answer. Do not try to infer it, and do not record a guess.

## The codes

Give every case **one primary code**: the defect that, fixed alone, would recover most of what the
case lost. Give any other defect present as a **secondary** code. Use only these codes.

| Code | Name | Use when |
|---|---|---|
| R | Provider drop / non-response | there is no answer at all. Expected to be empty in this study. |
| C1 | Syntax / compile error | the answer does not render because of invalid syntax, whether one bad token on correct content or a malformed answer throughout. |
| C2 | Runaway / truncation | the answer has no `@enduml` (it ran on, usually looping, until the token limit). The case file states this; it is mechanical. |
| C3 | Diagram-type / family confusion | the answer is written in the wrong diagram family, e.g. a sequence diagram drawn with rectangles, components or class arrows. Primary when the family swap causes the failure, whether or not the answer renders. |
| S1 | Wrong relation type | the endpoints are right but the relation's decoration is wrong: a diamond, triangle, dashed line or arrowhead drawn as another (composition as inheritance, association as dependency, …). A reversed edge or a wrong association arrowhead also counts here. |
| S2 | Missed relations | edges present in the ground truth are absent as edges in the answer, including messages written as notes instead of arrows (add secondary `note`). |
| S3 | Hallucinated elements | the answer invents nodes or edges the image does not show: enum members or stereotypes promoted to classes, extra participants, members or labels moved into `note` blocks (add secondary `note`), spurious edges on a ground truth with no edges. |
| S4 | OCR / text / name misread | a name is misread, so it fails the name match and cascades to its edges: misspelling, dropped or added word, dropped `:` of `:Instance`, a typed circle letter kept as part of the name (`C Admin` for `Admin`), accents, an identifier for a display string. |
| S5 | Type / stereotype confusion | the element kind is wrong (interface as class, enum as class, participant as actor). Usually secondary. |
| S6 | Control-flow loss | `alt` / `opt` / `loop` / `ref` / `group` structure lost in a sequence diagram. Usually secondary. |
| S7 | Missed elements | nodes the image shows are absent from the answer, not because of a misread name. |
| M | Measurement artifact | the answer is right and the score is wrong, for one of two reasons only: (a) the ground truth declares something its image does not draw, or draws it otherwise (a degenerate ground truth); (b) the extractor splits a faithfully transcribed name, e.g. a dotted name read as a namespace path. Anything else you would call M: write `M?` as primary and set `ask_author`. |
| note | (secondary only) | the answer uses `note` blocks for content the ground truth draws as members, labels or messages. |

Rules:

- A typed letter, a dropped or added word, or any other name change the image does not show is
  S4, never M.
- Notes in the answer are model behaviour, never M: S3 with `note` when they add nodes or edges,
  S2 with `note` when they replace messages.
- A compile fail is C1, C2 or C3 as primary. C2 whenever the case says the answer has no
  `@enduml`. A structural defect visible in a compile fail may be a secondary.
- If two codes are equally dominant, or none fits, choose the closer one and set `ask_author`.

## What to write

One JSON object per case, all cases of your batch in one file:

```json
{"E07": {"primary": "S1", "secondary": ["S4"], "reason": "both compositions drawn as inheritance; one name misspelled", "opened_stripped": false, "ask_author": false}}
```

`reason` is one sentence of evidence from the case: the line, edge or name that shows the defect.
