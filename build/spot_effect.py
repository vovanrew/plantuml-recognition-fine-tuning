#!/usr/bin/env python3
"""Effect of the spot-letter rule alone: scorer v2 against scorer v2 with the letter put back.

The extractor removes the letter of a stereotype spot (`<< (C,#ADD1B2) >>`) as part of reading the
drawn text, so the rule has no switch. This script restores the letter outside the scorer: a name
gets its letter back when the typed label starts with a lone letter, the diagram source uses the
spot syntax, and the drawn name no longer starts with that letter. Everything else is scorer v2.

Reads the isolated blocks under <v2 root>/element_f1/<run>/{gt_extracted,pred_extracted}.
Writes spot_effect.json and spot_effect.md to --out: per run, v2 minus restored (micro F1 on a 0-1
scale, full-set / compiled-only) and the number of names restored. bridge_table.py reads the JSON
and puts the row into the bridge table.

Usage (Java on PATH):
  python3 spot_effect.py --scorer ../plantuml-recognition-evaluation \
      --jar ../plantuml-recognition-evaluation/plantuml-stats-extractor-v2.jar \
      --v2 ../plantuml-recognition-evaluation/data_v2 --csr-root ../baseline-evaluation/data/csr --out ..
"""
import argparse
import copy
import json
import os
import re
import subprocess
import sys

CLS = "net.sourceforge.plantuml.stats.DiagramStatsExtractor"
SPOT_SOURCE = re.compile(r"<<\s*\(\s*[A-Za-z]\s*,")
LONE_LETTER = re.compile(r"^([A-Za-z]) (?=\s|<<)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scorer", required=True)
    ap.add_argument("--jar", required=True)
    ap.add_argument("--v2", required=True)
    ap.add_argument("--csr-root", required=True)
    ap.add_argument("--out", required=True, help="directory for spot_effect.json and spot_effect.md")
    args = ap.parse_args()
    scorer = os.path.abspath(args.scorer)
    sys.path.insert(0, os.path.join(scorer, "evaluation"))
    import element_f1 as ef
    import relationship_f1 as rf
    from element_f1_runner import graphs_from_output

    def graphs(directory, *opts):
        out = subprocess.run(["java", "-cp", args.jar, CLS, *opts, "--dir", os.path.abspath(directory)],
                             capture_output=True, text=True, check=True).stdout
        return graphs_from_output(out)

    def restored(directory):
        """(v2 graphs, v2 graphs with spot letters put back, number of names restored)."""
        drawn, raw = graphs(directory), graphs(directory, "--names=raw")
        back, n = copy.deepcopy(drawn), 0
        for key, rec in back.items():
            source = open(os.path.join(directory, key + ".puml"), encoding="utf-8").read()
            if not SPOT_SOURCE.search(source):
                continue

            def fix(typed, name):
                m = LONE_LETTER.match(typed)
                return m.group(1) + " " + name if m and not name.startswith(m.group(1) + " ") else name

            for node, node_raw in zip(rec["nodes"], raw[key]["nodes"]):
                new = fix(node_raw["name"], node["name"])
                n += new != node["name"]
                node["name"] = new
            for edge, edge_raw in zip(rec["edges"], raw[key]["edges"]):
                edge["source"] = fix(edge_raw["source"], edge["source"])
                edge["target"] = fix(edge_raw["target"], edge["target"])
        return drawn, back, n

    def micro(module, gt, pred, keys, compiled):
        rows = {r["key"]: r for r in module.compute(gt, pred, keys)}
        return (module.aggregate([rows[k] for k in keys])["micro"]["f1"],
                module.aggregate([rows[k] for k in compiled])["micro"]["f1"])

    models = json.load(open(os.path.join(scorer, "analysis", "model_registry.json")))["models"]
    lines = [
        "# Effect of the spot-letter rule alone\n",
        "Scorer v2 minus scorer v2 with the letter of a stereotype spot put back into the name "
        "(`scripts/spot_effect.py`). Micro F1 on a 0-1 scale, full-set / compiled-only; "
        "\"names restored\" counts node names over the 1,000 diagrams of a run.\n",
        "| Run | GT names restored | prediction names restored | Element F1: v2 - restored | "
        "Relationship F1: v2 - restored |",
        "|---|---|---|---|---|"]
    out = {}
    for m in models:
        base = os.path.join(args.v2, "element_f1", m["run_dir"])
        gt, gt_back, n_gt = restored(os.path.join(base, "gt_extracted"))
        pred, pred_back, n_pred = restored(os.path.join(base, "pred_extracted"))
        keys = sorted(gt)
        csr = json.load(open(os.path.join(args.csr_root, m["run_dir"], "csr_results.json")))
        compiled = [d["key"] for d in csr["diagrams"] if d.get("compiled") and d["key"] in gt]
        cells = []
        out[m["id"]] = {"gt_names_restored": n_gt, "pred_names_restored": n_pred}
        for name, module in (("element_f1", ef), ("relationship_f1", rf)):
            v2 = micro(module, gt, pred, keys, compiled)
            old = micro(module, gt_back, pred_back, keys, compiled)
            out[m["id"]][name] = [v2[0] - old[0], v2[1] - old[1]]
            cells.append("%+.4f / %+.4f" % (v2[0] - old[0], v2[1] - old[1]))
        lines.append("| %s | %d | %d | %s |" % (m["display"], n_gt, n_pred, " | ".join(cells)))
    with open(os.path.join(args.out, "spot_effect.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(args.out, "spot_effect.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
