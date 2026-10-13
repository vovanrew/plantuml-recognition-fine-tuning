#!/usr/bin/env python3
"""The dev set, the loss-curve split and the training sample: elements D7 and D10 to D16,
Amendment 1 (A1 to A5) and Amendment 2 (A6 to A8) of decisions/training-pool-rules/record.md,
drawn from the pool that build/pool.py leaves after D6 (data/pool/pool_after_d6.json).

Stages, in the order of Implementation notes 6 and 14 (`--until` stops after a stage):

  dev          A3 first: rule D6 (build/leakage.py) on the diagram block alone between the pool
               after D6 and the test diagrams; a flagged candidate goes on the exclusion list of
               the three draws. Then D11: 25 diagrams per cell from the pool after D6, the
               sampler of the test set with its limit of 5 per repository, seed 42; rendered and
               replaced as D14 says
  loss_curve   D12: 25 per cell, same sampler and seed, from the other pool diagrams of the dev
               repositories; rendered and replaced as D14 says; the targets of both sets counted
               against the token cap
  d7           D7: rule D6 on the whole file between the pool without the dev repositories and
               the 200 dev diagrams; a flagged candidate leaves the training pool. Then A4: the
               rule on the block alone between the training pool after D7 and the dev diagrams;
               a flagged candidate goes on the exclusion list before the training draw. Then A6:
               the rule on the whole file and on the block alone between the same training pool
               and the diagrams of the loss-curve split; a candidate flagged on either text goes
               on the exclusion list before the training draw
  train        D13: 2,500 per cell from that pool, one seeded permutation per cell, seed 42, no
               repository limit; rendered and replaced as D14 says. The first 625 and the first
               125 diagrams of each cell's order are the 5,000 and the 1,000 subsets
  report       D8 on the whole file and A5 on the block alone, A8 (the final sample against the
               loss-curve split, on both texts), D15, the checks, the artifacts of D16

The sampler (draw) is build_test_set.sample_cells with two additions: the limit per repository is
an argument (None: no limit), and a key on the exclusion list is passed over. Each of the three
draws starts its own generator with the seed (Implementation note 12).

D14. A drawn key is rendered with build/render.py (defaults). A key whose status is not "ok", or
whose rendered padded_size carries a degeneracy label of the test set's code, goes on the
exclusion list, and the sampler runs again over the same lists (Implementation note 6): the
candidate lists and the permutations do not change. The exclusion list is one list for the three
draws. The key list passed to every call of render.py is saved (render_calls/).

Amendment 1. The block of a source is what csr_runner.extract_puml returns for it
(pool.isolate_batches); the blocks of the pool and of the test diagrams are written under
<work>/blocks and measured block against block with leakage.all_pairs, the run of D6 and D7. The
pool file, the candidate lists and the permutations are not changed (A2): the flagged keys go on
the exclusion list. When --out already holds dev.json or loss_curve.json, the build stops before
a render if the draw of that set gives other keys (Implementation note 14).

Amendment 2. A6 goes the way of A3 and A4 (RULES names, for each rule, its reference set and the
texts on which it is measured; apply() in build() runs a rule): leakage.all_pairs between the
training pool after D7 and the loss-curve diagrams, once on the files and once on the blocks. The
flagged keys go on the exclusion list (A7); nothing else changes.

Artifacts (--out, default data/sets): dev.json, loss_curve.json, train.json (key lists: key,
type, tier, repository, rank in the cell, target tokens), config.json, counts.json (count per
cell at every step, replacements, checks, D15), dropped.jsonl (the keys D14 drops),
block_excluded.jsonl (the keys A3 and A4 exclude), loss_curve_excluded.jsonl (the keys A6
excludes), d7_removed.jsonl, d8_report.json,
neighbours_test.tsv, neighbours_dev.tsv (whole file), neighbours_test_block.tsv,
neighbours_dev_block.tsv (block alone), render_calls/, render/<set>/ (manifest.jsonl,
render_env.json, summary.json and the images in puml_images_1568/, written by render.py),
SHA256SUMS (every file but the images).

Usage (from fine-tune/):
  python3 build/sample.py --work <scratch folder> --out data/sets
"""
import argparse
import json
import os
import platform
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import pool as P
import leakage as L

HERE = P.HERE
POOL_DIR = HERE.parent / "data/pool"
DEFAULT_OUT = HERE.parent / "data/sets"

SEED = 42
N_DEV = 25                   # D11, per cell
N_LOSS = 25                  # D12, per cell
N_TRAIN = 2500               # D13, per cell
PREFIXES = {"train_20k": 2500, "train_5k": 625, "train_1k": 125}   # D13: first diagrams of each cell's order
REPO_FLAG = 100              # D13 would be reversed by a repository with more diagrams in the sample
CLOCK = re.compile(r"%(date|now)\b")     # D15: decisions/render-pipeline/implement.md, section 4.6
STAGES = ("dev", "loss_curve", "d7", "train", "report")
SETS = ("dev", "loss_curve", "train")
MEASURES = {"a": "jaccard_5gram", "b": "edit_similarity", "c": "lines_held"}
LEVELS = (0.5, 0.6, 0.7, 0.8)
TEXTS = ("block", "whole_file")               # the two texts of a pair, as an exclusion record names them
RULES = {"A3": ("test", ("block",)),          # Amendments 1 and 2: the rule, its reference set and the texts on
         "A4": ("dev", ("block",)),           # which it is measured (D6 and D7 are measured on the whole file)
         "A6": ("loss_curve", ("whole_file", "block"))}
AMENDMENTS = {"amendment_1": ("A3", "A4"), "amendment_2": ("A6",)}                     # in the order decided
EXCLUDED = {"amendment_1": "block_excluded", "amendment_2": "loss_curve_excluded"}    # their exclusion records
REPORT_TEXT = {"whole_file": "whole_file", "block": "block_alone"}                    # a text in d8_report.json
FROZEN = ("dev", "loss_curve")                # Implementation note 14: lists that must come out as written


class SampleError(RuntimeError):
    pass


def cell_name(cell):
    return f"{cell[0]}_T{cell[1]}"


def cell_of(d):
    return (d["primary_type"], d["tier"])


def count_cells(diagrams):
    c = Counter(cell_of(d) for d in diagrams)
    return {cell_name(x): c[x] for x in P.CELLS}


# --------------------------------------------------------------------------- #
# D10 to D13: the sampler
# --------------------------------------------------------------------------- #
def cells_of(diagrams):
    """{(type, tier): [(key, record)]}, each list in the order of `diagrams` (the `by_cell` of
    build_test_set.build; the pool file lists every type in corpus order)."""
    out = {c: [] for c in P.CELLS}
    for d in diagrams:
        out[cell_of(d)].append((d["key"], d))
    return out


def draw(by_cell, n_per_cell, seed, repo_cap=None, excluded=()):
    """build_test_set.sample_cells with the limit per repository as an argument (None: no limit)
    and an exclusion list. One generator; the cells in the fixed order class T1..T4, sequence
    T1..T4; one permutation per cell, drawn over the whole list of the cell. A key of `excluded`
    is passed over: it is not picked and does not count for its repository. Returns
    (selected [(key, record)], per-cell statistics)."""
    code = P.test_set_code()
    excluded = excluded if isinstance(excluded, (set, frozenset)) else set(excluded)
    rng = np.random.default_rng(seed)
    repo_count = defaultdict(int)
    selected, stats = [], {}
    for t in code.TYPES:
        for tier in (1, 2, 3, 4):
            keys = list(by_cell.get((t, tier), []))
            order = rng.permutation(len(keys))
            picked, passed = [], 0
            for idx in order:
                key, rec = keys[idx]
                if key in excluded:
                    passed += 1
                    continue
                repo = rec["repository"]
                if repo_cap is not None and repo_count[repo] >= repo_cap:
                    continue
                picked.append((key, rec))
                repo_count[repo] += 1
                if len(picked) >= n_per_cell:
                    break
            selected.extend(picked)
            stats[(t, tier)] = {"available": len(keys), "selected": len(picked), "excluded_passed_over": passed}
    return selected, stats


def cell_orders(by_cell, seed):
    """{cell: [keys in the order of the cell's permutation]}: the orders draw() walks."""
    code = P.test_set_code()
    rng = np.random.default_rng(seed)
    out = {}
    for t in code.TYPES:
        for tier in (1, 2, 3, 4):
            keys = list(by_cell.get((t, tier), []))
            out[(t, tier)] = [keys[i][0] for i in rng.permutation(len(keys))]
    return out


def ranked(selected, tokens=None):
    """The key list of a draw: one record per key with its cell and its rank in the cell (1 is
    the first key of the cell's order)."""
    out, rank = [], Counter()
    for key, rec in selected:
        rank[cell_of(rec)] += 1
        row = {"key": key, "primary_type": rec["primary_type"], "tier": rec["tier"],
               "repository": rec["repository"], "rank": rank[cell_of(rec)]}
        if tokens is not None:
            row["target_tokens"] = tokens[key]
        out.append(row)
    return out


def prefix(diagrams, n):
    """D13: the first n diagrams of each cell's order."""
    return [d for d in diagrams if d["rank"] <= n]


def reserve(pool, dev_keys):
    """D11, D12: (the dev repositories, the other pool diagrams of those repositories, the pool
    without those repositories). A repository is a dev repository when it has a dev diagram."""
    dev_keys = set(dev_keys)
    repos = {d["repository"] for d in pool if d["key"] in dev_keys}
    others = [d for d in pool if d["repository"] in repos and d["key"] not in dev_keys]
    rest = [d for d in pool if d["repository"] not in repos]
    return repos, others, rest


# --------------------------------------------------------------------------- #
# D14: render, drop, replace
# --------------------------------------------------------------------------- #
def render_outcome(row):
    """None for a rendered diagram that stays; else the reason it is dropped: a status other
    than "ok", or a degeneracy label of build_test_set.degeneracy_label on its padded_size."""
    if row["status"] != "ok":
        return "status:" + row["status"]
    label = P.test_set_code().degeneracy_label([tuple(row["padded_size"])])
    return None if label is None else "degenerate:" + label


def read_manifest(render_dir):
    path = Path(render_dir) / "manifest.jsonl"
    return {r["key"]: r for r in P.read_jsonl(path)} if path.is_file() else {}


def run_render_py(key_file, render_dir, src_dir, log_path):
    """`python3 build/render.py --src-dir <sources> --keys <key list> --out <folder>`, defaults.
    The three paths are passed as absolute paths: render.py hands --out to PlantUML's
    --output-dir, and PlantUML resolves a relative one against the folder of the sources, so
    that render.py finds no image and gives every key the status no_image."""
    cmd = [sys.executable, str(P.RENDER_PY), "--src-dir", str(Path(src_dir).resolve()),
           "--keys", str(Path(key_file).resolve()), "--out", str(Path(render_dir).resolve())]
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    cwd = P.render_cwd(log_path)        # beside the log, never fine-tune/: PlantUML drops debug files there
    with open(log_path, "a") as fh:
        fh.write(f"# {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(cmd)}\n")
        fh.flush()
        t0 = time.time()
        code = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=str(cwd)).returncode
        fh.write(f"# exit {code}, {time.time() - t0:.1f} s\n")
    if code != 0:
        raise SampleError(f"render.py ended with exit code {code}; see {log_path}")


def settle(name, by_cell, n_per_cell, repo_cap, excluded, dropped, render_dir, calls_dir, src_dir,
           render, seed=SEED, log=P.log):
    """One draw with its replacements (D14, Implementation note 6): draw; render the drawn keys
    that have no manifest line (the key list of the call is saved first); put every drawn key
    that is dropped on the exclusion list; draw again over the same lists; until no drawn key is
    dropped. `excluded` (a set) and `dropped` (a list of records) are extended in place. A cell
    that cannot supply n_per_cell stops the build. Returns (selected, statistics, the keys of
    the first draw)."""
    calls_dir = Path(calls_dir)
    first = None
    while True:
        selected, stats = draw(by_cell, n_per_cell, seed, repo_cap, excluded)
        short = {cell_name(c): f"{s['selected']} of {s['available']}" for c, s in stats.items()
                 if s["selected"] < n_per_cell}
        if short:
            raise SampleError(f"{name}: cells that cannot supply {n_per_cell} diagrams: {short}")
        if first is None:
            first = [k for k, _ in selected]
        manifest = read_manifest(render_dir)
        todo = [k for k, _ in selected if k not in manifest]
        if todo:
            calls_dir.mkdir(parents=True, exist_ok=True)
            n = len(list(calls_dir.glob(f"{name}_call_*.txt"))) + 1
            key_file = calls_dir / f"{name}_call_{n:02d}.txt"
            key_file.write_text("".join(k + "\n" for k in todo))
            log(f"{name}: render call {n}, {len(todo):,} keys")
            render(key_file, render_dir, src_dir)
            manifest = read_manifest(render_dir)
            missing = [k for k in todo if k not in manifest]
            if missing:
                raise SampleError(f"{name}: {len(missing)} keys have no manifest line after the render")
        new = [(k, rec, render_outcome(manifest[k])) for k, rec in selected]
        new = [(k, rec, why) for k, rec, why in new if why]
        if not new:
            return selected, stats, first
        for k, rec, why in new:
            excluded.add(k)
            dropped.append({"key": k, "set": name, "primary_type": rec["primary_type"], "tier": rec["tier"],
                            "repository": rec["repository"], "reason": why, "status": manifest[k]["status"],
                            "padded_size": manifest[k].get("padded_size")})
        log(f"{name}: dropped {len(new)} ({dict(Counter(why for _, _, why in new))}); drawing again")


def draw_from_manifest(by_cell, n_per_cell, seed, repo_cap, excluded, manifest):
    """The keys settle() ends with when it has nothing to render: the draw, repeated with every
    drawn key that its manifest line drops put on a copy of the exclusion list. A drawn key
    without a manifest line stays in the list returned. Nothing is rendered or written."""
    excluded = set(excluded)
    while True:
        selected, _ = draw(by_cell, n_per_cell, seed, repo_cap, excluded)
        out = {k for k, _ in selected if k in manifest and render_outcome(manifest[k])}
        if not out:
            return [k for k, _ in selected]
        excluded |= out


def replacement_summary(name, first, final, dropped):
    """What the replacements of one draw did: dropped keys by cell and reason (a replacement
    that is dropped itself counts too), the keys of the final list that the first draw did not
    hold, and whether every other key of the first draw is still in its cell, in the same order,
    before the new keys."""
    gone = {d["key"] for d in dropped if d["set"] == name}
    by_cell = {cell_name(c): {} for c in P.CELLS}
    for d in dropped:
        if d["set"] == name:
            cell = by_cell[cell_name(cell_of(d))]
            cell[d["reason"]] = cell.get(d["reason"], 0) + 1
    final_cell = {d["key"]: cell_of(d) for d in final}
    kept_first = [k for k in first if k not in gone]
    lost = [k for k in kept_first if k not in final_cell]
    same = not lost
    for c in P.CELLS:          # in every cell: the kept keys of the first draw, in their order, then the new keys
        now = [d["key"] for d in final if cell_of(d) == c]
        before = [k for k in kept_first if final_cell.get(k) == c]
        same = same and now[:len(before)] == before
    first_set = set(first)
    new = [k for k in final_cell if k not in first_set]
    return {"drawn_first": len(first), "dropped": len(gone), "dropped_of_the_first_draw": len(gone & first_set),
            "new_keys": len(new),
            "dropped_by_reason": dict(Counter(d["reason"] for d in dropped if d["set"] == name)),
            "dropped_by_cell": by_cell,
            "first_draw_keys_no_longer_drawn_that_were_not_dropped": len(lost),
            "every_other_key_of_the_first_draw_keeps_its_place": bool(same)}


# --------------------------------------------------------------------------- #
# D7, D8: rule D6 against another list; nearest training diagram of a reference diagram
# --------------------------------------------------------------------------- #
def near_duplicates(cand_keys, cand_dir, ref_keys, ref_dir, work, workers=8):
    """Rule D6 between a candidate list and a reference list: leakage.all_pairs, the run of
    step 1 against the test set, with the reference list in the place of the test diagrams."""
    return L.all_pairs(cand_keys, cand_dir, ref_keys, ref_dir, work, workers=workers)


def arm_report(sim, cell_of_key):
    """Counts by arm and by cell of what a run of rule D6 removes, and the largest value of each
    measure among the candidates it keeps (the counts of pool.d6_report, without D9)."""
    keys = sim["keys"]
    empty = lambda: {cell_name(c): 0 for c in P.CELLS}
    flagged = {a: empty() for a in L.ARMS}
    combos, removed = defaultdict(empty), empty()
    for i in np.nonzero(sim["removed_mask"])[0]:
        c = cell_name(cell_of_key[keys[i]])
        arms = "".join(a for a in L.ARMS if sim["fired"][a][i])
        combos[arms or "none"][c] += 1          # "none": removed by the mask given, no arm in this run (rule_report)
        removed[c] += 1
        for a in arms:
            flagged[a][c] += 1
    total = lambda d: dict(d, total=sum(d.values()))
    kept = ~sim["removed_mask"]
    residual = {}
    for a in L.ARMS:
        vals = np.where(kept, sim["best"][a], -1.0)
        i = int(vals.argmax()) if len(vals) else -1
        residual[a] = None if i < 0 else {
            "value": float(vals[i]), "candidate": keys[i],
            "reference": sim["test_keys"][int(sim["arg"][a][i])], "cell": cell_name(cell_of_key[keys[i]])}
    return {"candidates_compared": len(keys), "reference_diagrams": len(sim["test_keys"]),
            "pairs": len(keys) * len(sim["test_keys"]),
            "removed": total(removed),
            "flagged_by_arm": {a: total(flagged[a]) for a in L.ARMS},
            "removed_by_combination_of_arms": {arms: total(combos[arms]) for arms in sorted(combos)},
            "largest_value_among_kept": residual,
            "reference_diagrams_with_a_removed_candidate": sim["test_side"]}


def nearest_per_reference(m, rows=None, chunk=4096):
    """For every reference diagram (a column of the matrices of leakage.measure) the largest
    value of each measure over the candidate rows given, and the row that gives it: every pair
    is read, none is skipped. On a tie the first row in the order given. A column whose largest
    value is 0 gets row -1. `rows` are row numbers of the matrices (default: all)."""
    idx = np.arange(len(m["cand_len"])) if rows is None else np.asarray(rows, dtype=np.int64)
    nt = len(m["test_len"])
    best = {a: np.zeros(nt) for a in L.ARMS}
    arg = {a: np.full(nt, -1, dtype=np.int64) for a in L.ARMS}
    cols = np.arange(nt)
    for lo in range(0, len(idx), chunk):
        part = idx[lo:lo + chunk]
        _, val = L.verdicts(m, part)
        for a in L.ARMS:
            j = val[a].argmax(axis=0)
            top = val[a][j, cols]
            better = top > best[a]
            best[a][better] = top[better]
            arg[a][better] = lo + j[better]          # position in `rows`
    return best, arg


def neighbour_table(ref, m, best, arg, cand_keys):
    """Rows of a neighbour table: for each reference diagram its nearest diagram of `cand_keys`
    on each measure. (c) has no value for a reference diagram of fewer than 10 distinct lines."""
    rows = []
    for j, d in enumerate(ref):
        n_lines = int(m["test_lines"][j])
        row = {"key": d["key"], "primary_type": d["primary_type"], "tier": d["tier"], "distinct_lines": n_lines}
        for a, name in MEASURES.items():
            if a == "c" and n_lines < L.C_MIN_LINES:
                row[name], row[name + "_train_key"] = None, None
            else:
                row[name] = float(best[a][j])
                row[name + "_train_key"] = cand_keys[int(arg[a][j])] if arg[a][j] >= 0 else None
        rows.append(row)
    return rows


def neighbour_summary(rows, cell_by_key):
    """The largest value of each measure over a neighbour table (first row on a tie), and the
    number of reference diagrams whose nearest training diagram is at or above each level."""
    out = {"reference_diagrams": len(rows)}
    for name in MEASURES.values():
        valued = [r for r in rows if r[name] is not None]
        top = max(valued, key=lambda r: r[name]) if valued else None
        out[name] = {
            "reference_diagrams_with_a_value": len(valued),
            "largest": None if top is None else {
                "value": top[name], "train_key": top[name + "_train_key"], "reference": top["key"],
                "cell_of_the_train_key": cell_name(cell_by_key[top[name + "_train_key"]])
                if top[name + "_train_key"] else None},
            "reference_diagrams_at_or_above": {str(x): sum(1 for r in valued if r[name] >= x) for x in LEVELS}}
    return out


# --------------------------------------------------------------------------- #
# Amendments 1 and 2: rule D6 on the diagram block alone, and against the loss-curve split
# --------------------------------------------------------------------------- #
def write_blocks(keys, src_dir, folder):
    """The diagram block of every source as csr_runner.extract_puml returns it, one file per key
    under the name of the key (pool.isolate_batches, the isolation of D3, with all keys in one
    batch). `folder` is emptied first. Returns the folder that holds the files."""
    keys = list(keys)
    dirs = P.isolate_batches(keys, str(src_dir), str(folder), batch=max(len(keys), 1))
    if not dirs:
        os.makedirs(os.path.join(str(folder), "b0000"))
        dirs = [os.path.join(str(folder), "b0000")]
    return Path(dirs[0])


def pair_values(cand_text, ref_text):
    """The arms and the three values of leakage.compare for one pair of texts."""
    r = L.compare(cand_text, ref_text)
    return {"arms": r["arms"], **{name: r[name] for name in MEASURES.values()}}


def flagged(rule, sims, by_key, cand_dirs, ref_dirs):
    """The exclusion records of one rule of RULES, one per candidate that a run of rule D6 flags
    on a text the rule measures, in the order of the candidate list. `sims` holds the run of
    leakage.all_pairs on each such text, `cand_dirs` and `ref_dirs` the folder of each text of
    TEXTS. A record names every reference diagram against which an arm fires on a measured text,
    with the arms and the three values of that pair on the two blocks and on the two whole files
    (leakage.compare); `arms` are the arms that fire for the candidate on a measured text. The
    record of a rule that measures more than one text also names the texts on which the
    candidate is flagged (`flagged_on`); that of a rule that measures the block alone has the
    fields of Amendment 1. The build stops when leakage.compare on a pair does not give the arms
    the matrices give on a measured text. `drawn_without_the_amendment` is filled in by build()."""
    ref_set, texts = RULES[rule]
    keys, ref_keys = sims[texts[0]]["keys"], sims[texts[0]]["test_keys"]
    if any(sims[t]["keys"] != keys or sims[t]["test_keys"] != ref_keys for t in texts):
        raise SampleError(f"{rule}: the runs on its texts do not have the same key lists")
    rows = []
    for i in np.nonzero(np.any([sims[t]["removed_mask"] for t in texts], axis=0))[0]:
        key = keys[int(i)]
        arm = {t: L.verdicts(sims[t]["matrices"], np.array([int(i)], dtype=np.int64))[0] for t in texts}
        pairs = []
        for j in np.nonzero(np.any([arm[t][a][0] for t in texts for a in L.ARMS], axis=0))[0]:
            ref = ref_keys[int(j)]
            pair = {"reference_key": ref}
            for t in TEXTS:
                pair[t] = pair_values(L.read_text(os.path.join(cand_dirs[t], key)), L.read_text(os.path.join(ref_dirs[t], ref)))
                fired = [a for a in L.ARMS if arm[t][a][0, j]] if t in texts else None
                if t in texts and pair[t]["arms"] != fired:
                    raise SampleError(f"{rule}: {key} against {ref}, {t}: the matrices give the arms {fired}, "
                                      f"leakage.compare gives {pair[t]['arms']}")
            pairs.append(pair)
        d = by_key[key]
        row = {"key": key, "primary_type": d["primary_type"], "tier": d["tier"], "repository": d["repository"],
               "rule": rule, "reference_set": ref_set}
        if len(texts) > 1:
            row["flagged_on"] = [t for t in texts if sims[t]["removed_mask"][i]]
        rows.append(dict(row, arms=[a for a in L.ARMS if any(sims[t]["fired"][a][i] for t in texts)], pairs=pairs,
                         drawn_without_the_amendment=None))
    return rows


def block_report(sim, cell_of_key):
    """arm_report() for a run of the rule on the blocks, with the field names of an exclusion:
    a flagged candidate stays in the pool and goes on the exclusion list (A2)."""
    names = {"removed": "excluded", "removed_by_combination_of_arms": "excluded_by_combination_of_arms",
             "largest_value_among_kept": "largest_value_among_the_candidates_not_excluded",
             "reference_diagrams_with_a_removed_candidate": "reference_diagrams_with_an_excluded_candidate"}
    return {names.get(k, k): v for k, v in arm_report(sim, cell_of_key).items()}


def rule_report(rule, sims, cell_of_key):
    """block_report() of the run on each text a rule of RULES measures. The excluded candidates
    are those flagged on any of its texts: one flagged on another text only is counted in the
    report of this text under the arms "none", and is not among the candidates not excluded."""
    mask = np.any([sim["removed_mask"] for sim in sims.values()], axis=0)
    return {t: block_report(dict(sim, removed_mask=mask), cell_of_key) for t, sim in sims.items()}


def exclusion_effect(without, final, keys):
    """What the keys an amendment puts on the exclusion list change in one draw. `without` is
    the ranked key list the draw gives over the same candidate lists when those keys are taken
    off the exclusion list, `final` the ranked list with them on it. Returns the excluded keys
    that `without` holds, the keys of `final` that `without` does not hold, and whether in every
    cell `final` begins with the other keys of `without` in their order."""
    held = [d for d in without if d["key"] in keys]
    before = {d["key"] for d in without}
    enter = [d for d in final if d["key"] not in before]
    same = True
    for c in P.CELLS:
        was = [d["key"] for d in without if cell_of(d) == c and d["key"] not in keys]
        now = [d["key"] for d in final if cell_of(d) == c]
        same = same and now[:len(was)] == was
    short = lambda ds: [{"key": d["key"], "cell": cell_name(cell_of(d)), "rank": d["rank"]} for d in ds]
    return {"excluded_keys_the_draw_holds_without_the_exclusion": short(held), "keys_that_enter": short(enter),
            "every_other_key_keeps_its_place": bool(same)}


def arms_fired(m, rows, chunk=4096):
    """For the candidate rows given of the matrices of leakage.measure: the number of rows on
    which each arm of D6 fires against at least one reference diagram."""
    rows = np.asarray(list(rows), dtype=np.int64)
    n = {a: 0 for a in L.ARMS}
    for lo in range(0, len(rows), chunk):
        arm, _ = L.verdicts(m, rows[lo:lo + chunk])
        for a in L.ARMS:
            n[a] += int(arm[a].any(axis=1).sum())
    return n


def frozen_lists(out):
    """{set: [keys]} of the dev and loss-curve key lists already written under `out`; a set
    without a file is left out."""
    got = {}
    for name in FROZEN:
        path = Path(out) / f"{name}.json"
        if path.is_file():
            with open(path) as f:
                got[name] = [d["key"] for d in json.load(f)["diagrams"]]
    return got


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
def load_pool(pool_dir=POOL_DIR):
    """The pool after D6 and the target token counts of D4, from the artifacts of step 1; both
    files are checked against the SHA256SUMS of that folder. Returns (diagrams, tokens, sha256)."""
    pool_dir = Path(pool_dir)
    sums = dict(reversed(l.split("  ", 1)) for l in (pool_dir / "SHA256SUMS").read_text().splitlines())
    got = {}
    for name in ("pool_after_d6.json", "candidates.tsv", "config.json"):
        got[name] = P.sha256_file(pool_dir / name)
        if got[name] != sums[name]:
            raise SampleError(f"{pool_dir / name} has sha256 {got[name]}, SHA256SUMS has {sums[name]}")
    got["SHA256SUMS"] = P.sha256_file(pool_dir / "SHA256SUMS")
    with open(pool_dir / "pool_after_d6.json") as f:
        pool = json.load(f)["diagrams"]
    tokens = {}
    with open(pool_dir / "candidates.tsv") as fh:
        head = next(fh).rstrip("\n").split("\t")
        ki, ti = head.index("key"), head.index("target_tokens")
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if f[ti]:
                tokens[f[ki]] = int(f[ti])
    return pool, tokens, got


def check_pool_order(pool, records):
    """The pool file lists class then sequence, each in corpus order, with the tier and the
    repository of the metadata: the order in which build_test_set.build fills `by_cell`."""
    code, thr = P.test_set_code(), P.tier_thresholds(P.load_test_set())
    pos = {k: i for i, k in enumerate(records)}
    last = {t: -1 for t in P.TYPES}
    seen_sequence = False
    for d in pool:
        r = records[d["key"]]
        t = d["primary_type"]
        seen_sequence |= t == "sequence"
        if (t == "class" and seen_sequence) or pos[d["key"]] <= last[t] or r["primary_type"] != t \
                or r["repository"] != d["repository"] or code.assign_tier(r["content_lines"], thr[t]) != d["tier"]:
            raise SampleError(f"the pool file does not follow the metadata at {d['key']}")
        last[t] = pos[d["key"]]


def read_source(key, src_dir):
    with open(os.path.join(src_dir, key), "rb") as fh:
        return fh.read().decode("utf-8", "replace")


# --------------------------------------------------------------------------- #
# The build
# --------------------------------------------------------------------------- #
def set_facts(diagrams):
    repos = Counter(d["repository"] for d in diagrams)
    top = sorted(repos.items(), key=lambda kv: (-kv[1], kv[0]))[:3]
    return {"diagrams": len(diagrams), "cells": count_cells(diagrams), "repositories": len(repos),
            "largest_repository": {"repository": top[0][0], "diagrams": top[0][1]} if top else None,
            "three_largest_repositories": [{"repository": r, "diagrams": n} for r, n in top],
            "repositories_with_more_than_5_diagrams": sum(1 for n in repos.values() if n > 5)}


def build(pool, tokens, test, out, work, src_dir=P.PUML_DIR, test_dir=P.TEST_PUML_DIR,
          n_dev=N_DEV, n_loss=N_LOSS, n_train=N_TRAIN, prefixes=None, until="report", workers=12,
          render=None, seed=SEED, log=P.log, frozen=None):
    """Run the stages up to `until` on a pool (the diagram records of pool_after_d6.json) and
    return the state reached. Renders go to <out>/render/<set>, the key list of each render call
    to <out>/render_calls; caches, blocks and render logs to `work`. `frozen` ({set: [keys]}):
    the build stops before a render when the draw of such a set gives another list."""
    out, work = Path(out).resolve(), Path(work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    prefixes = dict(prefixes or PREFIXES)
    stop = STAGES.index(until)
    code = P.test_set_code()
    cap = code.REPO_CAP
    if render is None:
        render = lambda key_file, render_dir, src: run_render_py(
            key_file, render_dir, src, work / f"render_{Path(render_dir).name}.log")
    render_dir = {s: out / "render" / s for s in SETS}
    calls_dir = out / "render_calls"
    test_keys = [d["key"] for d in test["diagrams"]]
    test_repos = {d["repository"] for d in test["diagrams"]}
    excluded, dropped = set(), []
    records = {a: [] for a in AMENDMENTS}       # Amendments 1 and 2: the exclusion records of each, and the
    added = {a: set() for a in AMENDMENTS}      # keys it puts on the exclusion list
    S = {"steps": [], "dropped": dropped, "timing": {}, "pool": pool, "test": test, "tokens": tokens,
         "sizes": {"dev": n_dev, "loss_curve": n_loss, "train": n_train}, "prefixes": prefixes, "seed": seed,
         "repo_cap": cap, "draws": {}, "render_dir": render_dir, "src_dir": str(src_dir),
         "excluded": excluded, **{EXCLUDED[a]: records[a] for a in AMENDMENTS}, "sims": {}, "rules": {}}

    def step(name, diagrams, **extra):
        entry = {"step": name, "total": len(diagrams), "cells": count_cells(diagrams),
                 "repositories": len({d["repository"] for d in diagrams}), **extra}
        S["steps"].append(entry)
        log(f"[{name}] {entry['total']:,} diagrams, {entry['repositories']:,} repositories, "
            f"cells {list(entry['cells'].values())}")

    def as_frozen(name, candidates, n, limit):
        # Implementation note 14: compared before settle() renders anything for the set
        if not frozen or name not in frozen:
            return
        now = draw_from_manifest(cells_of(candidates), n, seed, limit, excluded, read_manifest(render_dir[name]))
        if now != list(frozen[name]):
            gone = [k for k in frozen[name] if k not in set(now)]
            new = [k for k in now if k not in set(frozen[name])]
            raise SampleError(f"{name}: the draw does not give the frozen list: {len(gone)} keys leave {gone[:5]}, "
                              f"{len(new)} keys enter {new[:5]}" + ("" if gone or new else " (another order)")
                              + "; nothing was rendered for this set")

    def one_draw(name, candidates, n, limit):
        t0 = time.time()
        by_cell = cells_of(candidates)
        selected, stats, first = settle(name, by_cell, n, limit, excluded, dropped,
                                        render_dir[name], calls_dir, src_dir, render, seed, log)
        final = ranked(selected, tokens)
        # A2, A5, A7, A8: for each amendment, the latest first, the same draw with its keys and those of a later
        # amendment taken off the exclusion list (the list before the amendment) against the list after it
        effects, after, off = {}, final, set()
        for a in reversed(list(AMENDMENTS)):
            off |= added[a]
            before = ranked(draw(by_cell, n, seed, limit, excluded - off)[0])
            effects[a] = exclusion_effect(before, after, added[a])
            if limit is None and not effects[a]["every_other_key_keeps_its_place"]:
                raise SampleError(f"{name}: a key that {a} does not exclude changes its place: {json.dumps(effects[a])}")
            held = {d["key"]: {"set": name, "rank": d["rank"]}
                    for d in effects[a]["excluded_keys_the_draw_holds_without_the_exclusion"]}
            for row in records[a]:
                if row["drawn_without_the_amendment"] is None and row["key"] in held:
                    row["drawn_without_the_amendment"] = held[row["key"]]
            after = before
        S["draws"][name] = dict(replacement_summary(name, first, final, dropped),
                                available={cell_name(c): s["available"] for c, s in stats.items()},
                                excluded_keys_passed_over={cell_name(c): s["excluded_passed_over"] for c, s in stats.items()},
                                **{a: effects[a] for a in AMENDMENTS})
        S["timing"][name] = time.time() - t0
        return final

    in_test_repo = [d["key"] for d in pool if d["repository"] in test_repos]
    if in_test_repo or set(test_keys) & {d["key"] for d in pool}:
        raise SampleError(f"the pool holds {len(in_test_repo)} diagrams of test repositories")
    step("pool_after_D6", pool)

    def apply(rule, cand_keys, ref_keys, ref_dirs):
        """One rule of RULES: rule D6 between the candidates and the reference diagrams on each text the rule
        measures; a candidate flagged on any of them gets a record and goes on the exclusion list."""
        t0 = time.time()
        texts = RULES[rule][1]
        sims = {t: near_duplicates(cand_keys, pool_dirs[t], ref_keys, ref_dirs[t],
                                   work / (rule.lower() if len(texts) == 1 else f"{rule.lower()}_{t}"), workers) for t in texts}
        rows = flagged(rule, sims, by_key, pool_dirs, ref_dirs)
        amendment = next(a for a, rules in AMENDMENTS.items() if rule in rules)
        records[amendment].extend(rows)
        added[amendment].update(r["key"] for r in rows if r["key"] not in excluded)
        excluded.update(r["key"] for r in rows)
        S["sims"][rule], S["rules"][rule] = sims, rule_report(rule, sims, cell_by_key)
        log(f"[{rule}] {len(rows):,} of {len(cand_keys):,} candidates flagged against the {RULES[rule][0]} diagrams "
            f"({', '.join(texts)})")
        S["timing"][rule.lower()] = time.time() - t0
        return sims

    # ---- Amendment 1, A3: rule D6 on the block alone, the pool against the test set ---- #
    t0 = time.time()
    by_key = {d["key"]: d for d in pool}
    cell_by_key = {d["key"]: cell_of(d) for d in pool}
    pool_keys = [d["key"] for d in pool]
    pool_dirs = {"block": write_blocks(pool_keys, src_dir, work / "blocks/pool"), "whole_file": src_dir}
    test_dirs = {"block": write_blocks(test_keys, test_dir, work / "blocks/test"), "whole_file": test_dir}
    S.update(cell_by_key=cell_by_key, block_dirs={"pool": str(pool_dirs["block"]), "test": str(test_dirs["block"])})
    S["timing"]["blocks"] = time.time() - t0
    sim_a3 = apply("A3", pool_keys, test_keys, test_dirs)["block"]       # passed over in the three draws

    # ---- dev set (D11, D14) ------------------------------------------------ #
    as_frozen("dev", pool, n_dev, cap)
    dev = one_draw("dev", pool, n_dev, cap)
    dev_repos, others, rest = reserve(pool, [d["key"] for d in dev])
    S.update(dev=dev, dev_repos=dev_repos)
    step("dev_set_D11", dev)
    step("pool_diagrams_of_the_dev_repositories", [d for d in pool if d["repository"] in dev_repos])
    step("pool_without_the_dev_repositories", rest)
    if stop < 1:
        return S

    # ---- loss-curve split (D12, D14) ---------------------------------------- #
    step("other_pool_diagrams_of_the_dev_repositories", others)
    as_frozen("loss_curve", others, n_loss, cap)
    loss = one_draw("loss_curve", others, n_loss, cap)
    S["loss_curve"] = loss
    step("loss_curve_split_D12", loss)
    # D14: the targets of both sets counted again, before the lists are frozen
    t0 = time.time()
    counted = [d["key"] for d in dev + loss]
    again = P.target_token_counts(counted, src_dir, workers=min(workers, 4))
    differ = [k for k in counted if again[k] != tokens[k]]
    over = [k for k in counted if not P.within_token_cap(again[k])]
    if differ or over:
        raise SampleError(f"token counts: {len(differ)} differ from the table of D4, {len(over)} above the cap")
    S["token_check"] = {"keys_counted_again": len(counted), "differing_from_the_table_of_D4": 0, "above_the_cap": 0,
                        "longest_target_tokens": {"dev": max(again[d["key"]] for d in dev),
                                                  "loss_curve": max(again[d["key"]] for d in loss)},
                        "token_cap": P.TOKEN_CAP}
    S["timing"]["token_check"] = time.time() - t0
    if stop < 2:
        return S

    # ---- D7 ------------------------------------------------------------------ #
    t0 = time.time()
    dev_keys = [d["key"] for d in dev]
    rest_keys = [d["key"] for d in rest]
    sim7 = near_duplicates(rest_keys, src_dir, dev_keys, src_dir, work / "d7", workers)
    train_pool = [d for d in rest if d["key"] not in sim7["removed"]]
    S.update(sim7=sim7, d7=arm_report(sim7, cell_by_key), train_pool=train_pool)
    step("training_pool_after_D7", train_pool, removed=S["d7"]["removed"])
    S["timing"]["d7"] = time.time() - t0

    # ---- Amendment 1, A4, and Amendment 2, A6: the training pool after D7 against the dev set on the block alone,
    # and against the loss-curve split on the whole file and on the block alone. Their keys go on the list after
    # the dev and loss-curve draws, before the training draw ---- #
    train_pool_keys = [d["key"] for d in train_pool]
    sim_a4 = apply("A4", train_pool_keys, dev_keys, pool_dirs)["block"]
    sims_a6 = apply("A6", train_pool_keys, [d["key"] for d in loss], pool_dirs)
    if stop < 3:
        return S

    # ---- training sample (D13, D14) ---------------------------------------------- #
    train = one_draw("train", train_pool, n_train, None)
    S["train"] = train
    step("training_sample_D13", train)
    if stop < 4:
        return S

    # ---- report (D8, A5, A8, D15, checks) ----------------------------------------- #
    t0 = time.time()
    train_keys = [d["key"] for d in train]
    loss_keys = [d["key"] for d in loss]
    # whole file: the final training sample and the two other sets against the test set, all pairs
    sim_t = near_duplicates(train_keys, src_dir, test_keys, test_dir, work / "d8_test", workers)
    sim_o = near_duplicates(dev_keys + loss_keys, src_dir, test_keys, test_dir, work / "d8_test_dev_loss", workers)
    best_t, arg_t = nearest_per_reference(sim_t["matrices"])
    row7 = {k: i for i, k in enumerate(sim7["keys"])}
    rows = [row7[k] for k in train_keys]
    best_d, arg_d = nearest_per_reference(sim7["matrices"], rows)
    # block alone: the rows of the final lists in the matrices of A3 (pool x test) and A4 (training pool x dev);
    # the candidates of A6 are those of A4, so its rows are rows4, on both texts
    row3 = {k: i for i, k in enumerate(sim_a3["keys"])}
    row4 = {k: i for i, k in enumerate(sim_a4["keys"])}
    rows3, rows4 = [row3[k] for k in train_keys], [row4[k] for k in train_keys]
    best_bt, arg_bt = nearest_per_reference(sim_a3["matrices"], rows3)
    best_bd, arg_bd = nearest_per_reference(sim_a4["matrices"], rows4)
    # the arms of D6 on the keys of the final lists, on either text
    fired = {"whole_file": {"train_against_test": arms_fired(sim_t["matrices"], range(len(train_keys))),
                            "dev_against_test": arms_fired(sim_o["matrices"], range(len(dev_keys))),
                            "loss_curve_against_test": arms_fired(sim_o["matrices"], range(len(dev_keys), len(dev_keys) + len(loss_keys))),
                            "train_against_dev": arms_fired(sim7["matrices"], rows)},
             "block_alone": {"train_against_test": arms_fired(sim_a3["matrices"], rows3),
                             "dev_against_test": arms_fired(sim_a3["matrices"], [row3[k] for k in dev_keys]),
                             "loss_curve_against_test": arms_fired(sim_a3["matrices"], [row3[k] for k in loss_keys]),
                             "train_against_dev": arms_fired(sim_a4["matrices"], rows4)}}
    for t, sim in sims_a6.items():
        fired[REPORT_TEXT[t]]["train_against_loss_curve"] = arms_fired(sim["matrices"], rows4)
    if any(n for text in fired.values() for pair in text.values() for n in pair.values()):
        raise SampleError(f"keys of a set on which an arm of D6 fires: {json.dumps(fired)}")
    S["neighbours_test"] = neighbour_table(test["diagrams"], sim_t["matrices"], best_t, arg_t, train_keys)
    S["neighbours_dev"] = neighbour_table(dev, sim7["matrices"], best_d, arg_d, train_keys)
    S["neighbours_test_block"] = neighbour_table(test["diagrams"], sim_a3["matrices"], best_bt, arg_bt, train_keys)
    S["neighbours_dev_block"] = neighbour_table(dev, sim_a4["matrices"], best_bd, arg_bd, train_keys)
    for t, sim in sims_a6.items():              # A8: the largest values go into the report; no table is written
        S["neighbours_loss_curve" + ("_block" if t == "block" else "")] = neighbour_table(
            loss, sim["matrices"], *nearest_per_reference(sim["matrices"], rows4), train_keys)
    sample_report = lambda text, suffix: {
        "diagrams": len(train_keys),
        "training_diagrams_on_which_an_arm_fires": {"test": fired[text]["train_against_test"],
                                                    "dev": fired[text]["train_against_dev"],
                                                    "loss_curve": fired[text]["train_against_loss_curve"]},
        "against_the_test_set": neighbour_summary(S["neighbours_test" + suffix], cell_by_key),
        "against_the_dev_set": neighbour_summary(S["neighbours_dev" + suffix], cell_by_key),
        "against_the_loss_curve_split": neighbour_summary(S["neighbours_loss_curve" + suffix], cell_by_key)}
    S["d8"] = {"D7": S["d7"],
               "training_sample": sample_report("whole_file", ""),
               "block_alone": {"A3": S["rules"]["A3"]["block"], "A4": S["rules"]["A4"]["block"],
                               "training_sample": sample_report("block_alone", "_block")},
               "A6": {REPORT_TEXT[t]: report for t, report in S["rules"]["A6"].items()},
               "set_keys_on_which_an_arm_fires": fired}
    S["timing"]["d8"] = time.time() - t0

    S["sets"] = {"dev": dev, "loss_curve": loss, **{name: prefix(train, n) for name, n in prefixes.items()}}
    manifests = {s: read_manifest(render_dir[s]) for s in SETS}
    which = lambda name: "train" if name.startswith("train") else name
    S["d15"], S["render"] = {}, {}
    for name, diagrams in S["sets"].items():
        man = manifests[which(name)]
        clock = [d["key"] for d in diagrams if CLOCK.search(read_source(d["key"], src_dir))]
        trace = [d["key"] for d in diagrams if man[d["key"]].get("renderer_trace")]
        S["d15"][name] = {"diagrams": len(diagrams), "clock_sources": len(clock), "renderer_trace": len(trace),
                          "clock_source_keys": clock, "renderer_trace_keys": trace}
    for s in SETS:
        keys = [d["key"] for d in S["sets"][s if s != "train" else max(prefixes, key=prefixes.get)]]
        man = manifests[s]
        calls = sorted(calls_dir.glob(f"{s}_call_*.txt")) if calls_dir.is_dir() else []
        S["render"][s] = {
            "keys_in_the_manifest": len(man), "status_in_the_manifest": dict(Counter(r["status"] for r in man.values())),
            "render_calls": [len(c.read_text().split()) for c in calls],
            "keys_of_the_set": len(keys),
            "status_of_the_keys_of_the_set": dict(Counter(man[k]["status"] for k in keys)),
            "attempt_of_the_keys_of_the_set": dict(Counter(man[k].get("attempt") for k in keys)),
            "keys_of_the_set_with_a_degeneracy_label": sum(1 for k in keys if render_outcome(man[k])),
            "manifest_keys_not_in_the_set": len(set(man) - set(keys))}
    S["checks"] = checks(S)
    S["timing"]["report"] = time.time() - t0
    return S


def checks(S):
    """The checks of the brief on the final sets; a failed check stops the build."""
    test, sets, prefixes = S["test"], S["sets"], S["prefixes"]
    biggest = max(prefixes, key=prefixes.get)
    dev, loss, train = sets["dev"], sets["loss_curve"], sets[biggest]
    repos = lambda ds: {d["repository"] for d in ds}
    keys = lambda ds: {d["key"] for d in ds}
    test_repos, test_keys = repos(test["diagrams"]), keys(test["diagrams"])
    shared = {"dev_and_test": len(repos(dev) & test_repos), "train_and_test": len(repos(train) & test_repos),
              "train_and_dev": len(repos(train) & repos(dev)), "loss_curve_and_train": len(repos(loss) & repos(train)),
              "loss_curve_and_test": len(repos(loss) & test_repos)}
    in_two = len(keys(dev) & keys(loss)) + len(keys(dev) & keys(train)) + len(keys(loss) & keys(train))
    in_a_set = keys(dev) | keys(loss) | keys(train)
    order = sorted(prefixes, key=prefixes.get)
    nested = {f"{a}_in_{b}": keys(sets[a]) <= keys(sets[b]) for a, b in zip(order, order[1:])}
    equal_cells = {name: sorted(set(count_cells(sets[name]).values())) == [n] for name, n in prefixes.items()}
    sizes = S["sizes"]
    per_repo = lambda ds: max(Counter(d["repository"] for d in ds).values())
    out = {"shared_repositories": shared,
           "loss_curve_repositories_outside_the_dev_repositories": len(repos(loss) - repos(dev)),
           "keys_in_two_sets": in_two,
           "test_keys_in_a_set": len(test_keys & in_a_set),
           "excluded_keys_in_a_set": len({d["key"] for d in S["dropped"]} & in_a_set),
           **{name + "_keys_in_a_set": len({r["key"] for r in S[name]} & in_a_set) for name in EXCLUDED.values()},
           "nested": nested, "eight_equal_cells": equal_cells,
           "dev_cells": sorted(set(count_cells(dev).values())) == [sizes["dev"]],
           "loss_curve_cells": sorted(set(count_cells(loss).values())) == [sizes["loss_curve"]],
           "largest_number_of_diagrams_of_one_repository": {
               "dev": per_repo(dev), "loss_curve": per_repo(loss), **{name: per_repo(sets[name]) for name in prefixes}},
           "repository_limit_of_dev_and_loss_curve": S["repo_cap"],
           "all_set_keys_rendered_ok_without_a_degeneracy_label": all(
               r["status_of_the_keys_of_the_set"] == {"ok": r["keys_of_the_set"]}
               and r["keys_of_the_set_with_a_degeneracy_label"] == 0 for r in S["render"].values()),
           "targets_within_the_token_cap": all(d["target_tokens"] <= P.TOKEN_CAP for ds in (dev, loss, train) for d in ds),
           "d13_reversing_condition_more_than_100_of_one_repository": per_repo(train) > REPO_FLAG}
    failed = [k for k, v in shared.items() if v] + [k for k in (
        "loss_curve_repositories_outside_the_dev_repositories", "keys_in_two_sets", "test_keys_in_a_set",
        "excluded_keys_in_a_set", *(name + "_keys_in_a_set" for name in EXCLUDED.values())) if out[k]]
    failed += [k for k, v in {**nested, **{"cells_" + n: v for n, v in equal_cells.items()}}.items() if not v]
    failed += [k for k in ("dev_cells", "loss_curve_cells", "all_set_keys_rendered_ok_without_a_degeneracy_label",
                           "targets_within_the_token_cap") if not out[k]]
    if max(out["largest_number_of_diagrams_of_one_repository"][s] for s in ("dev", "loss_curve")) > S["repo_cap"]:
        failed.append("repository limit of the dev set or the loss-curve split")
    if failed:
        raise SampleError(f"checks failed: {failed}; {json.dumps(out)}")
    return out


# --------------------------------------------------------------------------- #
# Artifacts
# --------------------------------------------------------------------------- #
def _fmt(x):
    return "" if x is None else ("%.6f" % x if isinstance(x, float) else str(x))


def write_key_list(path, head, diagrams):
    with open(path, "w", encoding="utf-8") as f:                 # one diagram per line
        f.write(json.dumps(head, ensure_ascii=False)[:-1] + ', "diagrams": [\n')
        f.write(",\n".join(json.dumps(d, ensure_ascii=False) for d in diagrams) + "\n]}\n")


def write_artifacts(S, out, inputs=None):
    """The artifacts of D16 under `out`, and SHA256SUMS over every file there but the images.
    Nothing written here depends on the time of the run."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    sizes, prefixes, cap = S["sizes"], S["prefixes"], S["repo_cap"]
    biggest = max(prefixes, key=prefixes.get)
    lists = {"dev": ("D11", S["dev"], sizes["dev"], cap), "loss_curve": ("D12", S["loss_curve"], sizes["loss_curve"], cap),
             "train": ("D13", S["train"], sizes["train"], None)}
    for name, (element, diagrams, n, limit) in lists.items():
        head = {"set": name, "decision": f"decisions/training-pool-rules/record.md, {element}", "seed": S["seed"],
                "n_per_cell": n, "repository_limit": limit, "count": len(diagrams), "cells": count_cells(diagrams),
                "repositories": len({d["repository"] for d in diagrams}),
                "rank": "position of the key in the order of its cell, from 1"}
        if name == "train":
            head["subsets"] = {s: {"rank_at_most": n_, "count": len(prefix(diagrams, n_))}
                               for s, n_ in sorted(prefixes.items(), key=lambda kv: -kv[1]) if s != biggest}
        write_key_list(out / f"{name}.json", head, diagrams)

    P.write_jsonl(out / "dropped.jsonl", S["dropped"])
    for name in EXCLUDED.values():
        P.write_jsonl(out / f"{name}.jsonl", S[name])
    removed = S["sim7"]["removed"]
    rename = lambda k: k.replace("_test_key", "_dev_key").replace("_test_diagrams_", "_dev_diagrams_")
    by_key = {d["key"]: d for d in S["pool"]}
    P.write_jsonl(out / "d7_removed.jsonl",
                  [{"key": k, "primary_type": by_key[k]["primary_type"], "tier": by_key[k]["tier"],
                    "repository": by_key[k]["repository"], **{rename(f): v for f, v in row.items()}}
                   for k, row in removed.items()])
    P.write_json(out / "d8_report.json", S["d8"])
    cols = ["key", "primary_type", "tier", "distinct_lines"] + [c for n in MEASURES.values() for c in (n, n + "_train_key")]
    for table in ("neighbours_test", "neighbours_dev", "neighbours_test_block", "neighbours_dev_block"):
        lines = ["\t".join(cols)] + ["\t".join(_fmt(r[c]) for c in cols) for r in S[table]]
        (out / f"{table}.tsv").write_text("\n".join(lines) + "\n")

    of_rule = lambda rule: next(iter(S["rules"][rule].values()))      # the three fields below are the same on every text
    P.write_json(out / "counts.json", {
        "steps": S["steps"],
        "draws": S["draws"],
        **{a: {**{rule: {k: of_rule(rule)[k] for k in ("candidates_compared", "reference_diagrams", "excluded")}
                  for rule in rules},
               "excluded_keys": len({r["key"] for r in S[EXCLUDED[a]]}),
               "excluded_keys_a_draw_holds_without_the_exclusion": {
                   name: len(S["draws"][name][a]["excluded_keys_the_draw_holds_without_the_exclusion"]) for name in SETS}}
           for a, rules in AMENDMENTS.items()},
        "sets": {name: set_facts(ds) for name, ds in S["sets"].items()},
        "dev_repositories": len(S["dev_repos"]),
        "token_check": S["token_check"],
        "checks": S["checks"],
        "D15": S["d15"],
        "render": S["render"]})
    P.write_json(out / "config.json", {
        "decision": "decisions/training-pool-rules/record.md, elements D7 and D10 to D16, Amendment 1 (A1 to A5), "
                    "Amendment 2 (A6 to A8)",
        "inputs": inputs or {},
        "code": {name: P.sha256_file(HERE / name) for name in ("sample.py", "pool.py", "leakage.py", "render.py")},
        "parameters": {
            "seed": S["seed"],
            "cells": [cell_name(c) for c in P.CELLS],
            "D10_tiers_content_lines": "quartile_thresholds of test_set.json; the tier of each key is that of data/pool/pool_after_d6.json",
            "D11_dev": {"n_per_cell": sizes["dev"], "repository_limit": cap, "drawn_from": "pool after D6"},
            "D12_loss_curve": {"n_per_cell": sizes["loss_curve"], "repository_limit": cap,
                               "drawn_from": "the other pool diagrams of the dev repositories"},
            "D13_train": {"n_per_cell": sizes["train"], "repository_limit": None,
                          "drawn_from": "pool after D6 without the dev repositories and without the candidates D7 flags",
                          "subsets_first_keys_of_each_cell": prefixes},
            "sampler": "build_test_set.sample_cells; a generator of its own for each draw; a key of the exclusion "
                       "list is passed over and does not count for its repository",
            "D7": {"shingle": L.SHINGLE, "a_jaccard_min": [L.A_NUM, L.A_DEN], "b_edit_similarity_min": [L.B_NUM, L.B_DEN],
                   "c_lines_held_min": [L.C_NUM, L.C_DEN], "c_min_reference_lines": L.C_MIN_LINES,
                   "reference": "the dev diagrams"},
            "Amendment_1": {
                "A1_text": "the block csr_runner.extract_puml returns for the file (pool.isolate_batches), on both sides "
                           "of a pair; the measures and thresholds are those of D7 above (build/leakage.py)",
                "extract_puml_source_sha256": P.PINS["extract_puml_source"],
                "A2": "a flagged key goes on the exclusion list; the pool file, the candidate lists and the "
                      "permutations are not changed",
                "A3": {"candidates": "pool after D6", "reference": "the test diagrams",
                       "excluded_from": ["dev", "loss_curve", "train"]},
                "A4": {"candidates": "training pool after D7", "reference": "the dev diagrams", "excluded_from": ["train"]},
                "A5_neighbours": "neighbours_test_block.tsv and neighbours_dev_block.tsv, as D8_neighbours below; "
                                 "distinct_lines is the number of distinct lines of the block"},
            "Amendment_2": {
                "A6": {"candidates": "training pool after D7", "reference": "the diagrams of the loss-curve split",
                       "texts": list(RULES["A6"][1]), "excluded_from": ["train"]},
                "A6_texts": "whole_file: the file, the text of D7; block: the text of A1; a candidate that an arm "
                            "flags on either text is excluded",
                "A7": "as A2",
                "A8": "loss_curve_excluded.jsonl; in d8_report.json the key A6 and, in training_sample and in "
                      "block_alone.training_sample, against_the_loss_curve_split (no table of it is written)"},
            "D14": {"render": "build/render.py, defaults", "dropped": "status other than ok, or a degeneracy label on padded_size",
                    "clip_long_edge": P.test_set_code().CLIP_LONG_EDGE, "aspect_max": P.test_set_code().ASPECT_MAX,
                    "token_cap": P.TOKEN_CAP},
            "D15": {"clock_pattern": CLOCK.pattern, "logged_exception": "renderer_trace of the render manifest"},
            "D8_neighbours": "largest value over all training diagrams, every pair computed; first training key in "
                             "the order of train.json on a tie; no key when the value is 0; lines_held only for a "
                             "reference diagram of at least 10 distinct lines"},
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__, "platform": platform.platform()}})

    names = sorted(str(p.relative_to(out)) for p in out.rglob("*")
                   if p.is_file() and p.name != "SHA256SUMS" and "puml_images_1568" not in p.parts
                   and "_work" not in p.parts)
    (out / "SHA256SUMS").write_text("".join(f"{P.sha256_file(out / n)}  {n}\n" for n in names))
    return names


def print_report(S):
    print("\nCells at every step (diagrams)")
    for e in S["steps"]:
        print(f"  {e['step']:<46} {e['total']:>7,}  repositories {e['repositories']:>6,}  {list(e['cells'].values())}")
    print("\nDraws:", json.dumps(S["draws"], indent=1))
    print("\nDropped keys:", json.dumps(S["dropped"], indent=1))
    for a, name in EXCLUDED.items():
        print(f"\n{a}, the keys it excludes ({name}.jsonl):", json.dumps(S[name], indent=1))
    print("\nToken check:", json.dumps(S["token_check"]))
    print("\nSets:", json.dumps({name: set_facts(ds) for name, ds in S["sets"].items()}, indent=1))
    print("\nD8:", json.dumps(S["d8"], indent=1))
    print("\nD15:", json.dumps(S["d15"], indent=1))
    print("\nRender:", json.dumps(S["render"], indent=1))
    print("\nChecks:", json.dumps(S["checks"], indent=1))
    print("\nSeconds:", json.dumps({k: round(v, 1) for k, v in S["timing"].items()}))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--work", required=True, help="scratch folder for the caches and the render logs")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="folder for the artifacts and the renders (data/sets)")
    ap.add_argument("--until", choices=STAGES, default="report")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    try:
        pool, tokens, sums = load_pool()
        check_pool_order(pool, P.load_records())
        test = P.load_test_set()
        S = build(pool, tokens, test, args.out, args.work, until=args.until, workers=args.workers,
                  frozen=frozen_lists(args.out))
        if args.until == "report":
            inputs = {"pool": {"path": "data/pool/pool_after_d6.json", "sha256": sums["pool_after_d6.json"]},
                      "candidates": {"path": "data/pool/candidates.tsv", "sha256": sums["candidates.tsv"]},
                      "pool_config": {"path": "data/pool/config.json", "sha256": sums["config.json"],
                                      "parameters": json.load(open(POOL_DIR / "config.json"))["parameters"]},
                      "pool_sha256sums": {"path": "data/pool/SHA256SUMS", "sha256": sums["SHA256SUMS"]},
                      "test_set": {"path": str(P.TEST_SET.relative_to(P.UNI)), "sha256": P.sha256_file(P.TEST_SET)},
                      "sources": str(P.PUML_DIR.relative_to(P.UNI)),
                      "test_sources": str(P.TEST_PUML_DIR.relative_to(P.UNI))}
            write_artifacts(S, args.out, inputs)
            print_report(S)
        else:
            print(json.dumps({"steps": S["steps"], "draws": S["draws"], "dropped": S["dropped"],
                              **{name: S[name] for name in EXCLUDED.values()}, "d7": S.get("d7"), "rules": S["rules"],
                              "seconds": {k: round(v, 1) for k, v in S["timing"].items()}}, indent=1))
    except (SampleError, P.PoolError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
