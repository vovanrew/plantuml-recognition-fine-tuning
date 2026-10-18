#!/usr/bin/env python3
"""The checkpoint rule of the recipe: which saved adapter of one training run is the run's adapter.

decisions/recipe/record.md, Decision. Input: the dev scorings of a run's checkpoints, in training
order, each a folder written by `train/dev_scoring.py score` (csr/csr_results.json,
element_f1/element_f1_results.json). The last checkpoint is taken, unless an earlier one is better:
the paired percentile bootstrap over the dev diagrams (the benchmark's analysis/bootstrap.py, seed
20260614, 1,000 resamples) gives a 95% interval of earlier - last that lies above zero on CSR or on
Element F1 (micro, full set: a failed answer scores 0), and the earlier one's point difference on
the other metric is not negative. If several earlier checkpoints qualify, the latest of them is
taken. The same rule at every size.

Refused: result files that do not cover the same diagrams, the dev set's 200 by default; Element F1
not scored by the frozen scorer v2 (its JAR sha256 and rules); the benchmark's bootstrap or
aggregation code differing from tag `scorer-v2`; and, from each folder's scores.json and the answer
manifest it carries (`train/dev_scoring.py`): a scorer other than tag `scorer-v2` with its two
pinned JARs, answers not made with an adapter attached, at a cap other than 5,376 tokens, a deadline
other than 600 s, a failed runner, or in timing mode; answers not made in bf16 on the study's card;
a server identity without the model, revision, dtype, device, card, versions, recurrent-layer
implementation or timing mode; checkpoints answered by servers that differ in anything but the
adapter; checkpoints that are not the run's checkpoints in order (the adapter
folders `step_00625`, `step_01250`, `step_01875`, `step_02500` of one run folder for 20,000 pairs;
`recipe.checkpoint_steps`), or two that are the same adapter.

Usage (from fine-tune/, under the scorer's Python):
  python3 train/select_checkpoint.py [--pairs 20000] --out <run>/selection.json <scores of step 625> ... <scores of the last step>
Writes the decision and every number it rests on to --out.

Tests: train/tests/test_select_checkpoint.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
FT = HERE.parent
INSTRUMENT = FT.parent / "plantuml-recognition-evaluation"
DEV_SET = FT / "data" / "examples" / "dev_runner_set.json"
SCORER_TAG = "scorer-v2"
ANALYSIS_FILES = ("analysis/bootstrap.py", "analysis/aggregate.py")
EXTRACTOR_SHA256 = "2747a72aec9206e424ddf9ed659f9eceb96730444a68b1fe99094317400f555b"
V2_SCORER = {"rules": {"collapse_whitespace": True, "unicode_brackets": True,
                       "association_direction": True},
             "first_diagram": True, "extractor_args": [], "extractor_jar_sha256": EXTRACTOR_SHA256}

# The scorer a dev scoring must have used (methodology/scorer-v2.md §7; the render JAR of the render
# pipeline), as train/dev_scoring.py checks it before scoring; written here so that this file and
# train/recipe.py are the whole rule (train/tests/test_select_checkpoint.py keeps the two equal).
SCORER_PINS = {"tag": "scorer-v2", "commit": "f460b3529b18b12de7225a8d2f6ee2014835dae5",
               "extractor_jar_sha256": EXTRACTOR_SHA256,
               "render_jar_sha256": "280f4aa97f4f3b7af39358dd0d79f65e353b54a1630b5f121a13b1651162185d"}
# The card model of every scored run (decisions/gpu-precision/record.md, Decision Part 2), as
# infer/server.py reports it.
CARD = "NVIDIA RTX PRO 6000 Blackwell Server Edition"
SERVER_FIELDS = ("model", "revision", "dtype", "device", "gpu", "versions", "recurrent_layers", "timing")

sys.path.insert(0, str(HERE))
import recipe  # noqa: E402


class Refused(SystemExit):
    """An input the rule does not accept."""


def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def check_instrument(instrument=INSTRUMENT):
    """The bootstrap and aggregation code equal tag scorer-v2; the bootstrap's seed and number of
    resamples are the recipe's. Returns the analysis module functions used."""
    same = subprocess.run(["git", "-C", str(instrument), "diff", "--quiet", SCORER_TAG, "--",
                           *ANALYSIS_FILES]).returncode
    if same != 0:
        raise Refused(f"{ANALYSIS_FILES} of {instrument} differ from tag {SCORER_TAG}")
    if str(instrument) not in sys.path:
        sys.path.insert(0, str(instrument))
    from analysis import aggregate, bootstrap
    if (bootstrap.SEED, bootstrap.N_RESAMPLES) != (recipe.BOOTSTRAP_SEED, recipe.BOOTSTRAP_RESAMPLES):
        raise Refused(f"bootstrap seed and resamples {bootstrap.SEED}, {bootstrap.N_RESAMPLES} are "
                      f"not the recipe's {recipe.BOOTSTRAP_SEED}, {recipe.BOOTSTRAP_RESAMPLES}")
    return aggregate, bootstrap


def dev_keys(set_file=DEV_SET):
    keys = [d["key"] for d in json.load(open(set_file))["diagrams"]]
    return sorted(k[:-5] if k.endswith(".puml") else k for k in keys)


def load(folder, keys=None):
    """Per-diagram CSR and Element F1 rows of one checkpoint's dev scoring, in key order."""
    folder = Path(folder)
    csr_file, el_file = folder / "csr" / "csr_results.json", folder / "element_f1" / "element_f1_results.json"
    csr = {d["key"]: d for d in json.load(open(csr_file))["diagrams"]}
    el_all = json.load(open(el_file))
    scorer = {k: el_all["scorer"].get(k) for k in V2_SCORER}
    if scorer != V2_SCORER:
        raise Refused(f"{el_file} was not scored by scorer v2: {scorer}")
    el = {d["key"]: d for d in el_all["diagrams"]}
    if set(csr) != set(el):
        raise Refused(f"{folder}: the CSR and Element F1 files cover different diagrams")
    if keys is not None and sorted(csr) != list(keys):
        raise Refused(f"{folder}: {len(csr)} diagrams, not the {len(keys)} of the set")
    order = sorted(csr)
    scores_file = folder / "scores.json"
    scores = json.load(open(scores_file)) if scores_file.is_file() else None
    return {"folder": str(folder), "keys": order,
            "csr": [csr[k] for k in order], "element": [el[k] for k in order], "scores": scores,
            "files_sha256": {"csr": sha256_file(csr_file), "element_f1": sha256_file(el_file),
                             "scores": sha256_file(scores_file) if scores else None}}


def check_run(checkpoints, pairs=20000):
    """How the checkpoints were answered and scored, and that they are the run's checkpoints in
    training order. Returns the adapter of each, or Refused."""
    steps = recipe.checkpoint_steps(pairs)
    if len(checkpoints) != len(steps):
        raise Refused(f"{len(checkpoints)} checkpoints given; a run of {pairs} pairs has {len(steps)}")
    servers, adapters = [], []
    for c, step in zip(checkpoints, steps):
        where = c["folder"]
        scores = c["scores"]
        if scores is None:
            raise Refused(f"{where}: no scores.json")
        scorer = {k: scores["scorer"].get(k) for k in SCORER_PINS}
        if scorer != SCORER_PINS:
            raise Refused(f"{where}: scored by {scorer}, not the pinned scorer v2")
        manifest = scores.get("answer_manifest")
        if manifest is None:
            raise Refused(f"{where}: no answer manifest")
        if (manifest["max_tokens"], manifest["timeout_seconds"]) != (recipe.DEV_MAX_TOKENS, recipe.DEV_TIMEOUT_S):
            raise Refused(f"{where}: cap {manifest['max_tokens']}, deadline {manifest['timeout_seconds']} s; "
                          f"the recipe's are {recipe.DEV_MAX_TOKENS} and {recipe.DEV_TIMEOUT_S} s")
        if manifest["runner_exit_code"] != 0:
            raise Refused(f"{where}: the runner exited with {manifest['runner_exit_code']}")
        if manifest["set_file_sha256"] != scores["set_file_sha256"]:
            raise Refused(f"{where}: answered and scored on different set files")
        server = dict(manifest["server"])
        missing = [k for k in SERVER_FIELDS + ("adapter",) if k not in server]
        if missing:
            raise Refused(f"{where}: the server's identity lacks {missing}")
        adapter = server.pop("adapter")
        server.pop("served_name", None)
        if server["timing"] is not False:
            raise Refused(f"{where}: answered by a server in timing mode")
        if (server["device"], server["gpu"], server["dtype"]) != ("cuda", CARD, "bfloat16"):
            raise Refused(f"{where}: answered on {server['device']} {server['gpu']} in {server['dtype']}, "
                          f"not on the study's card ({CARD}) in bfloat16")
        if not server["recurrent_layers"]:
            raise Refused(f"{where}: the server does not say which recurrent-layer implementation ran")
        if not adapter or adapter.get("attached") != "not merged":
            raise Refused(f"{where}: answered without an attached adapter")
        name = Path(adapter["path"]).name
        if name != recipe.checkpoint_dir(step):
            raise Refused(f"{where}: adapter folder {name}, expected {recipe.checkpoint_dir(step)} "
                          f"(the checkpoints of a {pairs}-pair run, in training order)")
        servers.append(server)
        adapters.append(adapter)
    if any(s != servers[0] for s in servers):
        raise Refused("the checkpoints were answered by servers that differ in more than the adapter")
    if len({Path(a["path"]).parent for a in adapters}) != 1:
        raise Refused("the adapters are not the checkpoints of one run folder")
    if len({a["adapter_model.safetensors"] for a in adapters}) != len(adapters):
        raise Refused("two checkpoints are the same adapter")
    if len({c["scores"]["set_file_sha256"] for c in checkpoints}) != 1:
        raise Refused("the checkpoints were scored on different set files")
    return adapters


def statistics(aggregate, csr_rows, element_rows):
    return {"csr": aggregate.csr_rate(csr_rows)["csr"],
            "element_f1_micro_full": aggregate.pool_micro(element_rows)["f1"]}


def select(checkpoints, aggregate, bootstrap):
    """The rule over loaded checkpoints (training order, the last is the end of the run)."""
    if len(checkpoints) < 1:
        raise Refused("no checkpoint")
    keys = checkpoints[0]["keys"]
    if any(c["keys"] != keys for c in checkpoints):
        raise Refused("the checkpoints were scored on different diagrams")
    resamples = bootstrap.make_resamples(len(keys), recipe.BOOTSTRAP_RESAMPLES, recipe.BOOTSTRAP_SEED)

    def resampled(c):
        return [statistics(aggregate, [c["csr"][i] for i in idx], [c["element"][i] for i in idx])
                for idx in resamples]

    points = [statistics(aggregate, c["csr"], c["element"]) for c in checkpoints]
    last, last_values = len(checkpoints) - 1, resampled(checkpoints[-1])
    comparisons, qualifying = [], []
    for i in range(last):
        values = resampled(checkpoints[i])
        diff = {m: bootstrap.diff_ci([v[m] for v in values], [v[m] for v in last_values],
                                     points[i][m], points[last][m]) for m in recipe.RULE_METRICS}
        csr, f1 = diff["csr"], diff["element_f1_micro_full"]
        qualifies = ((csr["ci_low"] > 0.0 and f1["diff_point"] >= 0.0)
                     or (f1["ci_low"] > 0.0 and csr["diff_point"] >= 0.0))
        comparisons.append({"checkpoint": i, "earlier_minus_last": diff, "qualifies": qualifies})
        if qualifies:
            qualifying.append(i)
    chosen = qualifying[-1] if qualifying else last
    return {"selected": chosen, "selected_folder": checkpoints[chosen]["folder"],
            "qualifying": qualifying, "points": points, "comparisons": comparisons,
            "n_diagrams": len(keys)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("scores", nargs="+", help="score folders of the run's checkpoints, in training order")
    ap.add_argument("--out", required=True, help="selection.json to write")
    ap.add_argument("--set-file", default=str(DEV_SET),
                    help="the set the checkpoints were scored on (keys must match)")
    ap.add_argument("--pairs", type=int, default=20000, choices=sorted(recipe.SUBSET_PAIRS),
                    help="pairs of the run, which sets its checkpoint steps")
    args = ap.parse_args(argv)
    aggregate, bootstrap = check_instrument()
    keys = dev_keys(args.set_file)
    checkpoints = [load(f, keys) for f in args.scores]
    adapters = check_run(checkpoints, args.pairs)
    if any(c["scores"]["set_file_sha256"] != sha256_file(args.set_file) for c in checkpoints):
        raise Refused(f"the checkpoints were not scored on {args.set_file}")
    result = select(checkpoints, aggregate, bootstrap)
    result["steps"] = recipe.checkpoint_steps(args.pairs)
    result["adapters"] = adapters
    result.update({
        "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rule": "decisions/recipe/record.md, Decision; train/select_checkpoint.py",
        "set_file": args.set_file, "set_file_sha256": sha256_file(args.set_file),
        "pairs": args.pairs,
        "inputs": [{"folder": c["folder"], **c["files_sha256"]} for c in checkpoints],
        "seed": recipe.BOOTSTRAP_SEED, "n_resamples": recipe.BOOTSTRAP_RESAMPLES,
        "instrument_tag": SCORER_TAG,
        "select_checkpoint_sha256": sha256_file(__file__),
        "recipe_sha256": sha256_file(HERE / "recipe.py"),
        "python": sys.version.split()[0]})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    print(json.dumps({"selected": result["selected"], "selected_folder": result["selected_folder"],
                      "qualifying": result["qualifying"]}), flush=True)
    return result


if __name__ == "__main__":
    main()
