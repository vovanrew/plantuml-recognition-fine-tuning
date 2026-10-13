#!/usr/bin/env python3
"""Audit a set of ground-truth PlantUML files for the name artifacts scorer v2 handles, and for
anything left over that it does not.

Runs the scorer-v2 extractor three times over the isolated blocks (`--names=raw`, `--names=markup`,
default) and reports, as nodes / diagrams:
  - what each v2 name rule changes, counted on match keys (markup and spot letter; largest-size
    line; Unicode brackets; whitespace runs);
  - residue left in v2 names (markup-like tokens, unpaired brackets, a lone letter before the name
    in a diagram that uses the spot syntax);
  - names where the drawn text lost a word that is not markup vocabulary (candidates for dropped
    text, to be read by hand);
  - association edges by arrowhead end; files with several diagrams; extractor errors.

Written for the test set (reproduces the counts in implement.md) and for the dev-set and training
pool ground truth, which must be audited before scorer v2 is tagged.

Usage (Java on PATH):
  python3 audit_gt.py --scorer ../plantuml-recognition-evaluation \
      --jar ../plantuml-recognition-evaluation/plantuml-stats-extractor-v2.jar \
      --gt-dir ../baseline-evaluation/data/puml_files [--keys ../baseline-evaluation/data/test_set.json]
"""
import argparse
import collections
import json
import os
import re
import subprocess
import sys
import tempfile

CLS = "net.sourceforge.plantuml.stats.DiagramStatsExtractor"
# words that markup itself contributes to a raw name
MARKUP_WORDS = re.compile(
    r"^(b|i|u|s|w|size|color|font|back|plain|sub|sup|img|U|\d+|[0-9A-Fa-f]{4,6}|[a-z]+)$")
RESIDUE = [
    ("creole/html tag", re.compile(r"</?(b|i|u|s|w|size|color|font|back|plain|sub|sup)\b[^<>]*>", re.I)),
    ("icon/emoji/sprite token", re.compile(r"<&[\w-]+>|<:[\w+-]+:>|<\$\w+>")),
    ("<U+XXXX> escape", re.compile(r"<U\+[0-9A-Fa-f]{4,6}>")),
    ("creole inline marker", re.compile(r"\*\*|(?<!:)//|__|~~|\"\"")),
    ("link brackets", re.compile(r"\[\[|\]\]")),
    ("empty angle pair", re.compile(r"<>")),
]
SPOT_SOURCE = re.compile(r"<<\s*\(\s*[A-Za-z]\s*,")
LONE_LETTER = re.compile(r"^[A-Za-z] \s*\S")


def extract(jar, directory, *opts):
    out = subprocess.run(["java", "-cp", jar, CLS, *opts, "--dir", directory],
                         capture_output=True, text=True, check=True).stdout
    recs = [json.loads(line, strict=False) for line in out.split("\n") if line.strip()]
    return {r["file"]: r for r in recs}


def nd(pairs):
    """'N nodes / D diagrams' for a list of (file, ...) tuples."""
    return "%d nodes / %d diagrams" % (len(pairs), len({p[0] for p in pairs}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scorer", required=True, help="checkout of plantuml-recognition-evaluation")
    ap.add_argument("--jar", required=True, help="scorer-v2 extractor JAR")
    ap.add_argument("--gt-dir", required=True)
    ap.add_argument("--keys", default="", help="test_set.json-style key list; default: every .puml in --gt-dir")
    ap.add_argument("--examples", type=int, default=5)
    args = ap.parse_args()

    sys.path.insert(0, os.path.join(os.path.abspath(args.scorer), "evaluation"))
    import element_f1 as ef
    from element_f1_runner import write_extracted

    if args.keys:
        keys = [d["key"][:-5] for d in json.load(open(args.keys))["diagrams"]]
    else:
        keys = sorted(f[:-5] for f in os.listdir(args.gt_dir) if f.endswith(".puml"))
    work = tempfile.mkdtemp(prefix="audit_gt_")
    write_extracted([os.path.join(args.gt_dir, k + ".puml") for k in keys], work)
    sources = {k + ".puml": open(os.path.join(work, k + ".puml"), encoding="utf-8").read() for k in keys}

    raw = extract(args.jar, work, "--names=raw")
    markup = extract(args.jar, work, "--names=markup")
    drawn = extract(args.jar, work)
    assert set(raw) == set(markup) == set(drawn)

    first = sorted(f for f in drawn if f in sources)
    later = sorted(f for f in drawn if f not in sources)
    errors = [(f, drawn[f]["error"]) for f in first if drawn[f]["error"]]
    n_nodes = sum(len(drawn[f]["nodes"]) for f in first)
    n_edges = sum(len(drawn[f]["edges"]) for f in first)
    print("files %d | nodes %d | edges %d | extractor errors %d | files with several diagrams %d"
          % (len(first), n_nodes, n_edges, len(errors), len({f.rsplit("_", 1)[0] for f in later})))
    for f, e in errors[:args.examples]:
        print("    error", f, e)

    by_markup, by_spot, by_size, by_unicode, by_ws, by_any, lost_text = [], [], [], [], [], [], []
    residue = collections.defaultdict(list)
    types = collections.Counter()
    for f in first:
        for r, m, d in zip(raw[f]["nodes"], markup[f]["nodes"], drawn[f]["nodes"]):
            types[d["type"]] += 1
            spot = bool(LONE_LETTER.match(r["name"]) and SPOT_SOURCE.search(sources[f]))
            if spot and not m["name"].startswith(r["name"][:2]):
                by_spot.append((f, r["name"], m["name"]))
            if ef.normalize(r["name"]) != ef.normalize(m["name"]):
                by_markup.append((f, r["name"], m["name"]))
                words = re.findall(r"\w+", r["name"])[1 if spot else 0:]
                lost = collections.Counter(words) - collections.Counter(re.findall(r"\w+", m["name"]))
                odd = [t for t in lost if not MARKUP_WORDS.match(t)]
                if odd:
                    lost_text.append((f, r["name"], m["name"], odd))
            if ef.normalize(m["name"]) != ef.normalize(d["name"]):
                by_size.append((f, m["name"], d["name"]))
            if ef.normalize(r["name"], ef.V1) != ef.normalize(d["name"]):
                by_any.append((f, r["name"], d["name"]))
            if ef.normalize(d["name"]) != ef.normalize(d["name"], ef.Rules(unicode_brackets=False)):
                by_unicode.append((f, d["name"]))
            if ef.normalize(d["name"]) != ef.normalize(d["name"], ef.Rules(collapse_whitespace=False)):
                by_ws.append((f, d["name"]))
            for label, pattern in RESIDUE:
                if pattern.search(d["name"]):
                    residue[label].append((f, d["type"], d["name"]))
            if re.search(r"[≪≫«»]", ef.normalize(d["name"])):
                residue["unpaired stereotype bracket"].append((f, d["type"], d["name"]))
            if LONE_LETTER.match(d["name"]) and SPOT_SOURCE.search(sources[f]):
                residue["lone letter in a diagram with spot syntax"].append((f, d["type"], d["name"]))

    def show(title, items, fmt):
        print("\n%s: %s" % (title, nd(items)))
        for it in items[:args.examples]:
            print("    " + fmt(it))

    cut = lambda s: repr(s[:90])
    print("\n== what the v2 name rules change (match keys, i.e. after normalize()) ==")
    show("match key differs between scorer v1 and scorer v2 (any rule)", by_any,
         lambda it: "%s %s -> %s" % (it[0][:12], cut(it[1]), cut(it[2])))
    show("drawn text differs from the typed label (markup, icons, spot letter)", by_markup,
         lambda it: "%s %s -> %s" % (it[0][:12], cut(it[1]), cut(it[2])))
    show("  of which: stereotype spot letter removed", by_spot,
         lambda it: "%s %s -> %s" % (it[0][:12], cut(it[1]), cut(it[2])))
    show("largest-size line rule drops a line", by_size,
         lambda it: "%s %s -> %s" % (it[0][:12], cut(it[1]), cut(it[2])))
    show("Unicode stereotype brackets stripped by normalize()", by_unicode, lambda it: "%s %s" % (it[0][:12], cut(it[1])))
    show("whitespace runs collapsed by normalize()", by_ws, lambda it: "%s %s" % (it[0][:12], cut(it[1])))

    print("\n== left over after v2 (read these by hand) ==")
    if not residue:
        print("    none")
    for label, items in sorted(residue.items()):
        by_type = collections.Counter(t for _, t, _ in items)
        show("%s %s" % (label, dict(by_type)), items, lambda it: "%s %s %s" % (it[0][:12], it[1], cut(it[2])))
    show("drawn text lost a word that is not markup vocabulary", lost_text,
         lambda it: "%s lost %s: %s -> %s" % (it[0][:12], it[3], cut(it[1]), cut(it[2])))

    print("\n== edges ==")
    heads = collections.Counter()
    diagrams_with = collections.defaultdict(set)
    relations = collections.Counter()
    for f in first:
        for e in drawn[f]["edges"]:
            relations[e["relation"]] += 1
            if e["relation"] == "association":
                heads[e["arrowhead"]] += 1
                diagrams_with[e["arrowhead"]].add(f)
    print("by relation:", dict(relations.most_common()))
    print("association edges by arrowhead end:",
          {k: "%d edges / %d diagrams" % (v, len(diagrams_with[k])) for k, v in heads.most_common()})
    one = heads["source"] + heads["target"]
    print("directed under v2 (exactly one arrowhead): %d of %d association edges in %d diagrams"
          % (one, sum(heads.values()), len(diagrams_with["source"] | diagrams_with["target"])))
    print("\nnode types:", dict(types.most_common()))


if __name__ == "__main__":
    main()
