#!/usr/bin/env python3
"""Build the v1 -> v2 bridge table from the level results written by bridge.py.

Reads   <out>/spot_effect.json                              (written by spot_effect.py; or --spot)
        <v2 root>/_bridge/<level>/<metric>/<run>/<metric>_results.json
        <v2 root>/<metric>/<run>/<metric>_results.json      (level `v2`)
        <v1 root>/<metric>/<run>/<metric>_results.json      (the stored scorer-v1 results)
Writes  bridge.md (tables) and bridge.json (every number) next to the decision record
        (another file stem with --name).

A data root that holds level `L5_first` was written after the drawn-edge rule (Decision,
Amendment 7): the table then has one more cumulative column after "+ first diagram", one more
leave-one-out column and section 5b, and the leave-one-out and font levels are compared with
`L5_first`, the level they are taken from. A data root without it gives the table as before.

All scores are F1 on a 0-1 scale. "full-set" is the zeros-for-failed population (1,000 diagrams
per run); "compiled-only" is the diagrams whose prediction compiled. Deltas are differences of
those scores.

Usage:
  python3 bridge_table.py --scorer ../plantuml-recognition-evaluation \
      --v1 ../plantuml-recognition-evaluation/data --v2 ../plantuml-recognition-evaluation/data_v2 --out ..
"""
import argparse
import json
import os

CUMULATIVE = ["L0_v1", "L1_whitespace", "L2_unicode", "L3a_markup", "L3_names", "L4_direction", "v2"]
STEP_LABEL = {
    "L1_whitespace": "+ whitespace (Python)",
    "L2_unicode": "+ Unicode brackets (Python)",
    "L3a_markup": "+ drawn text: markup, spot letter (extractor)",
    "L3_names": "+ largest-size line: package line (extractor)",
    "L4_direction": "+ association direction",
    "v2": "+ first diagram",
}
LEAVE_ONE_OUT = [("X_whitespace", "whitespace"), ("X_unicode", "Unicode brackets"),
                 ("X_names", "extractor names (markup, spot letter, package line)"),
                 ("X_package", "package line only"), ("X_direction", "association direction"),
                 ("X_first", "first diagram")]
# Amendment 7: one more cumulative level after the first-diagram level
CUMULATIVE_7 = CUMULATIVE[:-1] + ["L5_first", "v2"]
STEP_LABEL_7 = dict(STEP_LABEL, L5_first="+ first diagram", v2="+ drawn edges (extractor)")
RELATIONS = ["inheritance", "composition", "aggregation", "dependency", "association", "message"]
POPS = [("zeros_for_failed", "full-set"), ("compiled_only", "compiled-only")]
QWEN = ["qwen3.5-2b", "qwen3.5-9b", "qwen3.5-27b"]


def load(root, metric, run):
    return json.load(open(os.path.join(root, metric, run, metric + "_results.json")))


def score(res, metric, pop, avg):
    block = res["summary"][pop]
    if metric == "relationship_f1":
        block = block["all"]
    return block[avg]["f1"]


def flat(summary, prefix=""):
    """Every number of a summary as {path: value}."""
    out = {}
    for k, v in summary.items():
        if isinstance(v, dict):
            out.update(flat(v, prefix + k + "."))
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[prefix + k] = v
    return out


def f4(x):
    return "%.4f" % x


def d4(x):
    return "%+.4f" % x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scorer", required=True)
    ap.add_argument("--v1", required=True, help="data root of the stored scorer-v1 results")
    ap.add_argument("--v2", required=True, help="scorer-v2 data root written by bridge.py")
    ap.add_argument("--out", required=True, help="directory for bridge.md and bridge.json; "
                    "spot_effect.json (from spot_effect.py) is read from it")
    ap.add_argument("--name", default="bridge", help="file stem of the two output files")
    ap.add_argument("--spot", default="", help="spot_effect.json, if not the one in --out")
    args = ap.parse_args()
    spot = json.load(open(args.spot or os.path.join(args.out, "spot_effect.json")))

    models = json.load(open(os.path.join(args.scorer, "analysis", "model_registry.json")))["models"]
    reported = [m for m in models if not m["supplementary"]]
    bridge = os.path.join(args.v2, "_bridge")
    levels = sorted(d for d in os.listdir(bridge) if os.path.isdir(os.path.join(bridge, d))) + ["v2"]
    amend7 = "L5_first" in levels
    cumul = CUMULATIVE_7 if amend7 else CUMULATIVE
    label = STEP_LABEL_7 if amend7 else STEP_LABEL
    base = "L5_first" if amend7 else "v2"     # the level the X_* and F_* levels are taken from

    # the full-v2 level is the v2 data root itself
    res = {lv: {m["id"]: {mt: load(args.v2 if lv == "v2" else os.path.join(bridge, lv), mt, m["run_dir"])
                          for mt in ("element_f1", "relationship_f1")} for m in models}
           for lv in levels}
    num = {lv: {mid: {"%s.%s.%s" % (mt, pop, avg): score(r[mt], mt, pop, avg)
                      for mt in ("element_f1", "relationship_f1")
                      for pop, _ in POPS for avg in ("micro", "macro")}
                for mid, r in by_model.items()} for lv, by_model in res.items()}
    out = {"levels": levels, "scores": num}
    L = []
    w = L.append

    jar = {r[mt]["scorer"]["extractor_jar_sha256"] for by in res.values() for r in by.values() for mt in r}
    w("# Bridge table: scorer v1 -> scorer v2 on the stored runs\n")
    w("Generated by `scripts/bridge_table.py` from the level results of `scripts/bridge.py`. "
      "Extractor JAR sha256: `%s`. Scores are F1 on a 0-1 scale; full-set = zeros-for-failed, "
      "1,000 diagrams per run; compiled-only = diagrams whose prediction compiled. "
      "Claude Sonnet 4.6 is the supplementary run and is outside the ranking check.\n" % ", ".join(sorted(jar)))
    if amend7:
        w("Level `L5_first` is every rule up to the first-diagram rule, with every link and every "
          "message an edge; scorer v2 adds the drawn-edge rule to it (Decision, Amendment 7). Every "
          "level but `v2` is run with the extractor option `--undrawn-edges`.\n")

    # 1. v1 reproduction
    w("## 1. Every switch off reproduces the stored scorer-v1 results\n")
    w("Level `L0_v1` (v2 code and v2 JAR, every rule switch off) against the stored v1 result files: "
      "largest absolute difference over every number of the summary, and per-diagram rows whose "
      "tp/fp/fn differ.\n")
    w("| Run | Element F1: max abs diff | rows differing | Relationship F1: max abs diff | rows differing |")
    w("|---|---|---|---|---|")
    out["v1_reproduction"] = {}
    for m in models:
        cells = []
        for mt in ("element_f1", "relationship_f1"):
            old, new = load(args.v1, mt, m["run_dir"]), res["L0_v1"][m["id"]][mt]
            a, b = flat(old["summary"]), flat(new["summary"])
            assert set(a) == set(b), (m["id"], mt, set(a) ^ set(b))
            mx = max(abs(a[k] - b[k]) for k in a)
            rows_old = {d["key"]: (d["tp"], d["fp"], d["fn"]) for d in old["diagrams"]}
            rows_new = {d["key"]: (d["tp"], d["fp"], d["fn"]) for d in new["diagrams"]}
            assert set(rows_old) == set(rows_new)
            nd = sum(1 for k in rows_old if rows_old[k] != rows_new[k])
            cells += ["%.1e" % mx, "%d of %d" % (nd, len(rows_old))]
            out["v1_reproduction"]["%s.%s" % (m["id"], mt)] = {"max_abs_diff": mx, "rows_differing": nd}
        w("| %s | %s |" % (m["display"], " | ".join(cells)))
    w("")

    # 2. cumulative tables
    def cumulative(mt, avg, title):
        for pop, pop_label in POPS:
            w("### %s, %s\n" % (title, pop_label))
            if amend7:
                w("| Run | v1 | + whitespace | + Unicode brackets | + markup, spot letter | + package line | "
                  "+ association direction | + first diagram | + drawn edges = v2 | v2 - v1 |")
                w("|---|---|---|---|---|---|---|---|---|---|")
            else:
                w("| Run | v1 | + whitespace | + Unicode brackets | + markup, spot letter | + package line | "
                  "+ association direction | + first diagram = v2 | v2 - v1 |")
                w("|---|---|---|---|---|---|---|---|---|")
            for m in models:
                v = [num[lv][m["id"]]["%s.%s.%s" % (mt, pop, avg)] for lv in cumul]
                w("| %s | %s | %s |" % (m["display"], " | ".join(f4(x) for x in v), d4(v[-1] - v[0])))
            w("")

    w("## 2. Cumulative levels, every run\n")
    w("Each column adds one rule to the column on its left.\n")
    cumulative("element_f1", "micro", "Element F1, micro")
    cumulative("relationship_f1", "micro", "Relationship F1, micro")
    cumulative("element_f1", "macro", "Element F1, macro")
    cumulative("relationship_f1", "macro", "Relationship F1, macro")

    # 3. per-step deltas, three Qwen sizes
    w("## 3. Effect of each rule, the three dense Qwen sizes\n")
    w("Difference between a cumulative level and the level before it, micro F1, full-set / compiled-only. "
      "The spot-letter row is part of the drawn-text step above it, not a further step: the rule has no "
      "switch in the extractor, so its effect is scorer v2 minus scorer v2 with the letter put back "
      "(`scripts/spot_effect.py`, section 4b).\n")
    disp = {m["id"]: m["display"] for m in models}
    out["qwen_steps"] = {}
    for mt, title in (("element_f1", "Element F1"), ("relationship_f1", "Relationship F1")):
        w("### %s\n" % title)
        w("| Step | %s |" % " | ".join(disp[q] for q in QWEN))
        w("|---|" + "---|" * len(QWEN))
        for prev, lv in zip(cumul, cumul[1:]):
            cells = []
            for q in QWEN:
                ds = [num[lv][q]["%s.%s.micro" % (mt, pop)] - num[prev][q]["%s.%s.micro" % (mt, pop)]
                      for pop, _ in POPS]
                cells.append("%s / %s" % (d4(ds[0]), d4(ds[1])))
                out["qwen_steps"]["%s.%s.%s" % (mt, lv, q)] = ds
            w("| %s | %s |" % (label[lv], " | ".join(cells)))
            if lv == "L3a_markup":
                w("| &nbsp;&nbsp;of which: spot letter | %s |" % " | ".join(
                    "%s / %s" % (d4(spot[q][mt][0]), d4(spot[q][mt][1])) for q in QWEN))
        tot = []
        for q in QWEN:
            ds = [num["v2"][q]["%s.%s.micro" % (mt, pop)] - num["L0_v1"][q]["%s.%s.micro" % (mt, pop)]
                  for pop, _ in POPS]
            tot.append("%s / %s" % (d4(ds[0]), d4(ds[1])))
        w("| **v2 - v1** | %s |" % " | ".join(tot))
        w("")

    # 4. leave-one-out
    if amend7:
        w("## 4. Leave-one-out: one rule switched off\n")
        w("The score with every rule on minus the score with that rule off. Micro F1, full-set / "
          "compiled-only. The first six columns are taken at level `L5_first` (every rule up to the "
          "first-diagram rule, every link and message an edge), as before the drawn-edge rule existed: "
          "`L5_first` minus `L5_first` with that rule off. The last column is scorer v2 minus scorer v2 "
          "with the drawn-edge rule off, which is `L5_first` again. The Unicode-bracket rule acts only "
          "once the extractor decodes `<U+226A>`, so its effect shows here and not in its cumulative "
          "column.\n")
    else:
        w("## 4. Leave-one-out: scorer v2 with one rule switched off\n")
        w("v2 score minus the score with that rule off: the rule's effect in the presence of every other "
          "rule. Micro F1, full-set / compiled-only. The Unicode-bracket rule acts only once the extractor "
          "decodes `<U+226A>`, so its effect shows here and not in its cumulative column.\n")
    # (level with the rule off, column title, level with every rule on)
    loo = [(lv, title, base) for lv, title in LEAVE_ONE_OUT] + ([("X_edges", "drawn edges", "v2")] if amend7 else [])
    for mt, title in (("element_f1", "Element F1"), ("relationship_f1", "Relationship F1")):
        w("### %s\n" % title)
        w("| Run | %s |" % " | ".join(t for _, t, _ in loo))
        w("|---|" + "---|" * len(loo))
        for m in models:
            cells = []
            for lv, _, on in loo:
                ds = [num[on][m["id"]]["%s.%s.micro" % (mt, pop)] - num[lv][m["id"]]["%s.%s.micro" % (mt, pop)]
                      for pop, _ in POPS]
                cells.append("%s / %s" % (d4(ds[0]), d4(ds[1])))
            w("| %s | %s |" % (m["display"], " | ".join(cells)))
        w("")

    # 4b. spot letter
    w("## 4b. Spot letter: scorer v2 against scorer v2 with the letter put back\n")
    w("The letter of a stereotype spot (`<< (C,#ADD1B2) >>`) is removed inside the drawn-text step. "
      "`scripts/spot_effect.py` restores it outside the scorer in every name that had one; the difference "
      "is the rule's own effect. Micro F1, full-set / compiled-only; names restored are node names over "
      "the 1,000 diagrams of a run.\n")
    w("| Run | Ground-truth names | Prediction names | Element F1 | Relationship F1 |")
    w("|---|---|---|---|---|")
    out["spot_letter"] = spot
    for m in models:
        sp = spot[m["id"]]
        w("| %s | %d | %d | %s / %s | %s / %s |" % (
            m["display"], sp["gt_names_restored"], sp["pred_names_restored"],
            d4(sp["element_f1"][0]), d4(sp["element_f1"][1]),
            d4(sp["relationship_f1"][0]), d4(sp["relationship_f1"][1])))
    w("")

    # 5. association stratum
    w("## 5. Association stratum before and after the direction rule\n")
    w("Micro F1 of the `association` relation alone, levels `L3_names` -> `L4_direction`, full-set / "
      "compiled-only; `gt` and `pred` are the association edge counts over the full set.\n")
    w("| Run | gt | pred | undirected (before) | directed (after) | delta |")
    w("|---|---|---|---|---|---|")
    out["association"] = {}
    for m in models:
        b = res["L3_names"][m["id"]]["relationship_f1"]["summary"]
        a = res["L4_direction"][m["id"]]["relationship_f1"]["summary"]
        bs = [b[pop]["by_relation"]["association"]["f1"] for pop, _ in POPS]
        as_ = [a[pop]["by_relation"]["association"]["f1"] for pop, _ in POPS]
        sup = a["zeros_for_failed"]["by_relation"]["association"]
        out["association"][m["id"]] = {"before": bs, "after": as_}
        w("| %s | %d | %d | %s / %s | %s / %s | %s / %s |" % (
            m["display"], sup["support_gt"], sup["support_pred"], f4(bs[0]), f4(bs[1]),
            f4(as_[0]), f4(as_[1]), d4(as_[0] - bs[0]), d4(as_[1] - bs[1])))
    w("")

    # 5b. every relation before and after the drawn-edge rule
    if amend7:
        w("## 5b. Relations before and after the drawn-edge rule\n")
        w("Levels `L5_first` -> `v2`. First table: the `association` relation alone, micro F1, full-set / "
          "compiled-only; `gt` and `pred` are the association edge counts over the full set, before -> "
          "after. Second table: micro F1 of every relation, `v2` minus `L5_first`, full-set / "
          "compiled-only.\n")
        w("| Run | gt | pred | every link an edge (before) | drawn edges (after) | delta |")
        w("|---|---|---|---|---|---|")
        out["relations_drawn_edges"] = {}
        for m in models:
            b = res["L5_first"][m["id"]]["relationship_f1"]["summary"]
            a = res["v2"][m["id"]]["relationship_f1"]["summary"]
            out["relations_drawn_edges"][m["id"]] = {
                rel: {"before": [b[pop]["by_relation"][rel]["f1"] for pop, _ in POPS],
                      "after": [a[pop]["by_relation"][rel]["f1"] for pop, _ in POPS],
                      "support_gt": [x["zeros_for_failed"]["by_relation"][rel]["support_gt"] for x in (b, a)],
                      "support_pred": [x["zeros_for_failed"]["by_relation"][rel]["support_pred"] for x in (b, a)]}
                for rel in RELATIONS}
            r = out["relations_drawn_edges"][m["id"]]["association"]
            w("| %s | %d -> %d | %d -> %d | %s / %s | %s / %s | %s / %s |" % (
                m["display"], r["support_gt"][0], r["support_gt"][1], r["support_pred"][0], r["support_pred"][1],
                f4(r["before"][0]), f4(r["before"][1]), f4(r["after"][0]), f4(r["after"][1]),
                d4(r["after"][0] - r["before"][0]), d4(r["after"][1] - r["before"][1])))
        w("")
        w("| Run | %s |" % " | ".join(RELATIONS))
        w("|---|" + "---|" * len(RELATIONS))
        for m in models:
            r = out["relations_drawn_edges"][m["id"]]
            w("| %s | %s |" % (m["display"], " | ".join(
                "%s / %s" % (d4(r[rel]["after"][0] - r[rel]["before"][0]), d4(r[rel]["after"][1] - r[rel]["before"][1]))
                for rel in RELATIONS)))
        w("")

    # 6. type accuracy
    w("## 6. Type accuracy, v1 -> v2\n")
    w("Share of name-matched pairs whose entity type agrees, compiled-only; `correct / denominator`.\n")
    w("| Run | v1 | v2 | delta |")
    w("|---|---|---|---|")
    out["type_accuracy"] = {}
    for m in models:
        t1 = res["L0_v1"][m["id"]]["element_f1"]["summary"]["type_accuracy"]
        t2 = res["v2"][m["id"]]["element_f1"]["summary"]["type_accuracy"]
        out["type_accuracy"][m["id"]] = {"v1": t1["accuracy"], "v2": t2["accuracy"]}
        w("| %s | %s (%d / %d) | %s (%d / %d) | %s |" % (
            m["display"], f4(t1["accuracy"]), t1["correct"], t1["denominator"],
            f4(t2["accuracy"]), t2["correct"], t2["denominator"], d4(t2["accuracy"] - t1["accuracy"])))
    w("")

    # 7. base font size
    w("## 7. Base font size of unsized label text\n")
    if amend7:
        w("Scorer v2 takes it from each diagram's own style. Forcing it to a constant, at level `L5_first` "
          "(every link and message an edge, as before the drawn-edge rule existed): largest absolute "
          "difference to `L5_first` over all eight summary scores of a run, and per-diagram rows whose "
          "tp/fp/fn differ (Element F1 + Relationship F1 rows).\n")
    else:
        w("Scorer v2 takes it from each diagram's own style. Forcing it to a constant: largest absolute "
          "difference to v2 over all eight summary scores of a run, and per-diagram rows whose tp/fp/fn differ "
          "(Element F1 + Relationship F1 rows).\n")
    w("| Run | forced 12 pt: max abs diff | rows differing | forced 10 pt: max abs diff | rows differing |")
    w("|---|---|---|---|---|")
    out["font"] = {}
    for m in models:
        cells = []
        for lv in ("F_12", "F_10"):
            mx = max(abs(num[lv][m["id"]][k] - num[base][m["id"]][k]) for k in num[base][m["id"]])
            nd = 0
            for mt in ("element_f1", "relationship_f1"):
                r2 = {d["key"]: (d["tp"], d["fp"], d["fn"]) for d in res[base][m["id"]][mt]["diagrams"]}
                rf = {d["key"]: (d["tp"], d["fp"], d["fn"]) for d in res[lv][m["id"]][mt]["diagrams"]}
                nd += sum(1 for k in r2 if r2[k] != rf[k])
            cells += ["%.1e" % mx, str(nd)]
            out["font"]["%s.%s" % (m["id"], lv)] = {"max_abs_diff": mx, "rows_differing": nd}
        w("| %s | %s |" % (m["display"], " | ".join(cells)))
    w("")

    # 8. ranking
    w("## 8. Ranking of the seven registered runs\n")
    w("Order by score, best first, at v1 and at v2, for each table. A swap is a pair of runs whose "
      "order differs from v1; intermediate levels are checked too.\n")
    w("| Table | Order at v1 | Order at v2 | Pairs swapped at v2 | Pairs swapped at an intermediate level only |")
    w("|---|---|---|---|---|")
    out["ranking"] = {}
    ids = [m["id"] for m in reported]

    def order(lv, key):
        return sorted(ids, key=lambda i: -num[lv][i][key])

    def swapped(lv, key):
        base = order("L0_v1", key)
        now = order(lv, key)
        pos = {i: n for n, i in enumerate(now)}
        return {(a, b) for n, a in enumerate(base) for b in base[n + 1:] if pos[a] > pos[b]}

    for mt, mt_label in (("element_f1", "Element F1"), ("relationship_f1", "Relationship F1")):
        for avg in ("micro", "macro"):
            for pop, pop_label in POPS:
                key = "%s.%s.%s" % (mt, pop, avg)
                final = swapped("v2", key)
                mid = set()
                for lv in cumul[1:-1]:
                    mid |= {(lv, p) for p in swapped(lv, key) if p not in final}
                fmt = lambda pairs: "; ".join("%s <-> %s" % (disp[a], disp[b]) for a, b in sorted(pairs)) or "none"
                mid_txt = "; ".join("%s <-> %s (%s)" % (disp[a], disp[b], lv) for lv, (a, b) in sorted(mid)) or "none"
                out["ranking"][key] = {"v1": order("L0_v1", key), "v2": order("v2", key),
                                       "swapped_v2": sorted(final), "swapped_mid": sorted(mid)}
                w("| %s, %s, %s | %s | %s | %s | %s |" % (
                    mt_label, avg, pop_label, " > ".join(disp[i] for i in order("L0_v1", key)),
                    " > ".join(disp[i] for i in order("v2", key)), fmt(final), mid_txt))
    w("")

    # 9. population gap
    w("## 9. Population gap, v1 -> v2\n")
    w("compiled-only minus full-set, Element F1 micro and macro.\n")
    w("| Run | micro: v1 | v2 | change | macro: v1 | v2 | change |")
    w("|---|---|---|---|---|---|---|")
    for m in models:
        cells = []
        for avg in ("micro", "macro"):
            g = [num[lv][m["id"]]["element_f1.compiled_only." + avg] - num[lv][m["id"]]["element_f1.zeros_for_failed." + avg]
                 for lv in ("L0_v1", "v2")]
            cells += [f4(g[0]), f4(g[1]), d4(g[1] - g[0])]
        w("| %s | %s |" % (m["display"], " | ".join(cells)))
    w("")

    with open(os.path.join(args.out, args.name + ".md"), "w") as f:
        f.write("\n".join(L))
    with open(os.path.join(args.out, args.name + ".json"), "w") as f:
        json.dump(out, f, indent=1, default=list)
    print("\n".join(L))


if __name__ == "__main__":
    main()
