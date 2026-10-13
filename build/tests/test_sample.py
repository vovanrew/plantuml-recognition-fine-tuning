"""Tests for build/sample.py: elements D7 and D10 to D16, Amendment 1 (A1 to A5) and Amendment 2
(A6 to A8) of decisions/training-pool-rules.

Run from fine-tune/:  python3 -m pytest build/tests/test_sample.py -q
The tests of a made-up pool write their sources to a temporary folder and use a renderer that
writes manifest lines only. One test calls build/render.py (Java, Graphviz). The test that
rebuilds the draw of the test set reads the caches of build/pool.py and is skipped unless
POOL_WORK names their folder. The tests of the artifacts read data/pool, data/sets and the corpus,
and are skipped when data/sets does not exist.
"""
import json
import os
import pickle
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pool as P  # noqa: E402
import leakage as L  # noqa: E402
import sample as S  # noqa: E402

FT = Path(__file__).resolve().parents[2]
SETS = FT / "data/sets"
HEADER = "' Blob ID: 0\n' Original Path: /a.puml\n' Source: World of Code\n\n"
CAP = 5
ZERO = {"a": 0, "b": 0, "c": 0}
NO_ARM = {"test": ZERO, "dev": ZERO, "loss_curve": ZERO}          # training diagrams on which an arm fires
PAIRS = ("train_against_test", "dev_against_test", "loss_curve_against_test", "train_against_dev", "train_against_loss_curve")
NO_EFFECT = {"excluded_keys_the_draw_holds_without_the_exclusion": [], "keys_that_enter": [], "every_other_key_keeps_its_place": True}
needs_artifacts = pytest.mark.skipif(not (SETS / "SHA256SUMS").is_file(), reason="data/sets not built")


def made_up_cells(rng, sizes=(0, 1, 7, 30, 120), repos=(1, 3, 10, 40, 200)):
    n_repo = rng.choice(repos)
    return {c: [(f"{c[0][0]}{c[1]}_{i}.puml", {"key": f"{c[0][0]}{c[1]}_{i}.puml", "primary_type": c[0],
                                                  "tier": c[1], "repository": f"r{rng.randrange(n_repo)}"})
                for i in range(rng.choice(sizes))] for c in P.CELLS}


def keys_by_cell(selected):
    out = {c: [] for c in P.CELLS}
    for k, rec in selected:
        out[S.cell_of(rec)].append(k)
    return out


# --------------------------------------------------------------------------- #
# D10 to D13: the sampler
# --------------------------------------------------------------------------- #
def test_the_sampler_is_sample_cells_of_the_test_set():
    code = P.test_set_code()
    assert code.REPO_CAP == CAP
    rng = random.Random(1)
    for case in range(200):
        by_cell = made_up_cells(rng)
        n, seed = rng.choice([1, 5, 25, 125]), rng.choice([0, 42, 7, 20261004])
        reference, ref_stats = code.sample_cells(by_cell, n, seed)
        got, stats = S.draw(by_cell, n, seed, repo_cap=code.REPO_CAP)
        assert got == reference
        assert {c: (s["available"], s["selected"]) for c, s in stats.items()} == \
            {c: (s["available"], s["selected"]) for c, s in ref_stats.items()}


def step_1_candidates(work):
    """The candidate pool of the test set as step 1 rebuilds it (build/pool.py, stage d1), from
    the caches of that build and the labels of data/pool."""
    records, test = P.load_records(), P.load_test_set()
    cands, _ = P.inclusion(records)
    cand_keys = {t: list(cands[t]) for t in P.TYPES}
    norms = pickle.load(open(work / "norm.pkl", "rb"))
    sha1 = {k: v[0] for k, v in norms.items()}
    labels = {k: None for t in P.TYPES for k in cands[t]}
    with open(FT / "data/pool/d2_degeneracy_labels.tsv") as fh:
        next(fh)
        for line in fh:
            k, _, label, _, _ = line.rstrip("\n").split("\t")
            labels[k] = None if label == "None" else label
    raw = pickle.load(open(work / "d1_types_v1_network_closed.pkl", "rb"))
    repo_of = {k: records[k]["repository"] for k in labels}
    _, deduped, _ = P.degeneracy_and_dedup(cand_keys, labels, sha1)
    dkeys = [k for t in P.TYPES for k in deduped[t]]
    remote = {k for k in dkeys if not (raw[k]["diagram_type"] or "") and P.has_remote_include(k)}
    cp = P.candidate_pool(cand_keys, labels, sha1, raw, remote, repo_of, {d["repository"] for d in test["diagrams"]})
    return cp["agreed"], records, test


def test_the_sampler_rebuilds_the_draw_of_the_test_set_as_step_1_does():
    work = Path(os.environ.get("POOL_WORK", "/nonexistent"))
    if not (work / "norm.pkl").is_file() or not (work / "d1_types_v1_network_closed.pkl").is_file():
        pytest.skip("POOL_WORK does not name the caches of build/pool.py")
    code = P.test_set_code()
    agreed, records, test = step_1_candidates(work)
    thresholds, available, selected = P.rebuild_test_set(agreed, records, test)      # step 1: sample_cells
    by_cell = {c: [] for c in P.CELLS}
    for t in P.TYPES:
        for k in agreed[t]:
            by_cell[(t, code.assign_tier(records[k]["content_lines"], thresholds[t]))].append((k, records[k]))
    got, stats = S.draw(by_cell, test["config"]["n_per_cell"], test["config"]["seed"], repo_cap=code.REPO_CAP)
    assert [k for k, _ in got] == selected and len(selected) == 1000
    assert {S.cell_name(c): s["available"] for c, s in stats.items()} == available
    funnel = json.load(open(FT / "data/pool/funnel.json"))["test_set_rebuild"]
    assert available == funnel["available"]
    assert len({k for k, _ in got} & {d["key"] for d in test["diagrams"]}) == funnel["keys_identical_to_test_set"] == 790


def test_a_draw_is_determined_by_its_seed_and_starts_its_own_generator():
    by_cell = made_up_cells(random.Random(2), sizes=(120,), repos=(200,))
    one = S.draw(by_cell, 25, 42, CAP)
    np.random.default_rng(42).permutation(1000)        # another generator used in between
    S.draw(by_cell, 25, 7, CAP)
    assert S.draw(by_cell, 25, 42, CAP) == one
    assert S.draw(by_cell, 25, 43, CAP)[0] != one[0]
    assert S.cell_orders(by_cell, 42) == S.cell_orders(by_cell, 42)
    # the uncapped draw of a cell is the head of the cell's permutation
    orders = S.cell_orders(by_cell, 42)
    assert keys_by_cell(S.draw(by_cell, 25, 42, None)[0]) == {c: orders[c][:25] for c in P.CELLS}
    assert all(sorted(orders[c]) == sorted(k for k, _ in by_cell[c]) for c in P.CELLS)


def test_cells_are_type_by_tier_in_the_fixed_order_and_keep_the_order_of_the_pool():
    assert P.CELLS == [("class", 1), ("class", 2), ("class", 3), ("class", 4),
                       ("sequence", 1), ("sequence", 2), ("sequence", 3), ("sequence", 4)]
    assert P.tier_thresholds(P.load_test_set()) == {"class": (11.0, 24.0, 49.0), "sequence": (12.0, 22.0, 39.0)}
    pool = [dict(key=f"k{i}", primary_type=t, tier=tier, repository="r")
            for i, (t, tier) in enumerate([("class", 2), ("sequence", 4), ("class", 2), ("class", 1), ("sequence", 4)])]
    cells = S.cells_of(pool)
    assert list(cells) == P.CELLS
    assert [k for k, _ in cells[("class", 2)]] == ["k0", "k2"] and [k for k, _ in cells[("sequence", 4)]] == ["k1", "k4"]
    selected, _ = S.draw(cells, 2, 42, None)
    assert [S.cell_of(r) for _, r in selected] == [("class", 1), ("class", 2), ("class", 2), ("sequence", 4), ("sequence", 4)]
    assert [d["rank"] for d in S.ranked(selected)] == [1, 1, 2, 1, 2]


def test_the_limit_of_five_per_repository_holds_across_a_capped_draw_and_not_in_an_uncapped_one():
    rng = random.Random(4)
    by_cell = {c: [(f"{c[0][0]}{c[1]}_{i}.puml", {"primary_type": c[0], "tier": c[1],
                                                   "repository": "big" if i % 2 else f"r{rng.randrange(60)}"})
                   for i in range(80)] for c in P.CELLS}
    capped, _ = S.draw(by_cell, 25, 42, CAP)
    free, stats = S.draw(by_cell, 25, 42, None)
    per_repo = lambda sel: Counter(rec["repository"] for _, rec in sel)
    assert len(capped) == len(free) == 200
    assert max(per_repo(capped).values()) == CAP and per_repo(capped)["big"] == CAP      # across the set, not per cell
    assert per_repo(free)["big"] > 60
    # a limit that a repository reaches in an early cell holds in the later cells
    assert all(rec["repository"] != "big" for _, rec in capped[25:])


def test_an_excluded_key_is_replaced_by_the_next_key_of_its_cell_and_nothing_else_changes():
    by_cell = made_up_cells(random.Random(5), sizes=(60,), repos=(200,))
    orders = S.cell_orders(by_cell, 42)
    for cap in (None, CAP):
        first, _ = S.draw(by_cell, 10, 42, cap)
        before = keys_by_cell(first)
        assert max(Counter(rec["repository"] for _, rec in first).values()) < CAP      # the limit binds nowhere here
        cell = ("class", 3)
        out = before[cell][4]
        again, stats = S.draw(by_cell, 10, 42, cap, excluded={out})
        after = keys_by_cell(again)
        nxt = orders[cell][orders[cell].index(before[cell][-1]) + 1]      # the key after the last one drawn
        assert after[cell] == [k for k in before[cell] if k != out] + [nxt]
        assert all(after[c] == before[c] for c in P.CELLS if c != cell)
        assert stats[cell]["excluded_passed_over"] == 1 and stats[cell]["available"] == 60
        assert S.cell_orders(by_cell, 42) == orders
    # several keys, uncapped: every cell is the head of its unchanged permutation without the excluded keys
    excluded = {orders[c][i] for c in P.CELLS for i in (0, 3, 11, 59)}
    again, _ = S.draw(by_cell, 10, 42, None, excluded=excluded)
    assert keys_by_cell(again) == {c: [k for k in orders[c] if k not in excluded][:10] for c in P.CELLS}


def test_the_list_is_not_shortened_before_the_permutation_is_drawn():
    by_cell = made_up_cells(random.Random(6), sizes=(60,), repos=(200,))
    first, _ = S.draw(by_cell, 10, 42, None)
    out = first[3][0]
    shortened = {c: [kr for kr in rows if kr[0] != out] for c, rows in by_cell.items()}
    with_list, _ = S.draw(by_cell, 10, 42, None, excluded={out})
    assert [k for k, _ in S.draw(shortened, 10, 42, None)[0]] != [k for k, _ in with_list]
    assert len(set(k for k, _ in with_list) & set(k for k, _ in first)) == 79


def test_an_excluded_key_does_not_count_for_the_limit_of_its_repository():
    cell = ("class", 1)
    rows = [(f"k{i}.puml", {"primary_type": "class", "tier": 1, "repository": "one" if i < 30 else f"r{i}"})
            for i in range(40)]
    by_cell = {cell: rows}
    first = [k for k, _ in S.draw(by_cell, 12, 42, CAP)[0]]
    ones = [k for k in S.cell_orders(by_cell, 42)[cell] if int(k[1:-5]) < 30]
    assert len(first) == 12
    assert [k for k in first if k in ones] == ones[:5]
    again = [k for k, _ in S.draw(by_cell, 12, 42, CAP, excluded={ones[0]})[0]]
    assert [k for k in again if k in ones] == ones[1:6]         # the sixth diagram of the repository comes in


def test_the_other_diagrams_of_the_dev_repositories_leave_the_training_pool():
    pool = [dict(key=f"k{i}", primary_type="class", tier=1, repository=f"r{i % 4}") for i in range(12)]
    repos, others, rest = S.reserve(pool, ["k1", "k6"])
    assert repos == {"r1", "r2"}
    assert [d["key"] for d in others] == ["k2", "k5", "k9", "k10"]
    assert [d["key"] for d in rest] == ["k0", "k3", "k4", "k7", "k8", "k11"]
    assert not {d["repository"] for d in rest} & repos


def test_the_subsets_are_the_first_keys_of_each_cell_and_nested():
    by_cell = made_up_cells(random.Random(8), sizes=(300,), repos=(40,))
    train = S.ranked(S.draw(by_cell, 100, 42, None)[0])
    full, mid, small = S.prefix(train, 100), S.prefix(train, 25), S.prefix(train, 5)
    assert full == train and len(mid) == 200 and len(small) == 40
    assert {d["key"] for d in small} < {d["key"] for d in mid} < {d["key"] for d in full}
    assert set(S.count_cells(mid).values()) == {25} and set(S.count_cells(small).values()) == {5}
    orders = S.cell_orders(by_cell, 42)
    for c in P.CELLS:
        assert [d["key"] for d in small if S.cell_of(d) == c] == orders[c][:5]
        assert [d["key"] for d in mid if S.cell_of(d) == c] == orders[c][:25]
    assert S.PREFIXES == {"train_20k": 2500, "train_5k": 625, "train_1k": 125} and S.SEED == 42
    assert (S.N_DEV, S.N_LOSS, S.N_TRAIN) == (25, 25, 2500)


# --------------------------------------------------------------------------- #
# D14: render, drop, replace
# --------------------------------------------------------------------------- #
def test_a_rendered_key_is_dropped_for_its_status_or_for_a_degeneracy_label_on_its_padded_size():
    row = lambda **kw: dict(key="k.puml", **kw)
    assert S.render_outcome(row(status="ok", padded_size=[1568, 900])) is None
    assert S.render_outcome(row(status="ok", padded_size=[4096, 512])) is None                  # aspect 8.0
    assert S.render_outcome(row(status="ok", padded_size=[4097, 512])) == "degenerate:aspect"
    assert S.render_outcome(row(status="ok", padded_size=[16384, 9000])) == "degenerate:clipped"
    assert S.render_outcome(row(status="ok", padded_size=[16383, 9000])) is None
    # the label is taken on the padded size, not on the size PlantUML wrote
    assert S.render_outcome(row(status="ok", render_size=[4096, 100], padded_size=[4096, 512])) is None
    for status in ("multi_page", "no_image", "crash", "timeout", "missing_source", "postprocess_error"):
        assert S.render_outcome(row(status=status)) == "status:" + status
    # a logged exception and a diagram rendered alone are kept (D15)
    assert S.render_outcome(row(status="ok", padded_size=[900, 900], renderer_trace=["x.Exception"], attempt="solo")) is None


class FakeRender:
    """Writes a manifest line per key, as build/render.py does, and no image."""

    def __init__(self, rows=None, first_call=None):
        self.rows, self.calls = dict(rows or {}), []
        self.first_call = dict(first_call or {})      # {set: (position in its first call, manifest fields)}

    def __call__(self, key_file, render_dir, src_dir):
        keys = Path(key_file).read_text().split()
        name = Path(render_dir).name
        if name in self.first_call:
            position, fields = self.first_call.pop(name)
            self.rows[keys[position]] = fields
        self.calls.append((name, keys))
        Path(render_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(render_dir) / "manifest.jsonl", "a") as fh:
            for k in keys:
                row = dict(status="ok", attempt="batch", padded_size=[900, 700])
                row.update(self.rows.get(k, {}))
                fh.write(json.dumps(dict(key=k, **row)) + "\n")


def test_a_degenerate_render_is_replaced_by_the_next_key_of_the_cell(tmp_path):
    by_cell = made_up_cells(random.Random(9), sizes=(40,), repos=(200,))
    orders = S.cell_orders(by_cell, 42)
    clipped, pages = orders[("class", 2)][1], orders[("sequence", 4)][0]
    second = orders[("class", 2)][6]                       # the replacement of `clipped` is degenerate too
    render = FakeRender({clipped: dict(padded_size=[16384, 9000]), pages: dict(status="multi_page", pages=2),
                         second: dict(padded_size=[512, 5000])})
    excluded, dropped = set(), []
    selected, stats, first = S.settle("train", by_cell, 6, None, excluded, dropped, tmp_path / "render/train",
                                      tmp_path / "render_calls", tmp_path, render, log=lambda *a: None)
    final = keys_by_cell(selected)
    assert first == [k for c in P.CELLS for k in orders[c][:6]]
    assert final[("class", 2)] == [k for k in orders[("class", 2)][:8] if k not in (clipped, second)]
    assert final[("sequence", 4)] == orders[("sequence", 4)][1:7]
    assert all(final[c] == orders[c][:6] for c in P.CELLS if c not in (("class", 2), ("sequence", 4)))
    assert excluded == {clipped, pages, second}
    assert [(d["key"], d["reason"], d["set"]) for d in dropped] == [
        (clipped, "degenerate:clipped", "train"), (pages, "status:multi_page", "train"), (second, "degenerate:aspect", "train")]
    # three calls: the draw, the two replacements of the first round, the replacement of the second
    assert [len(keys) for _, keys in render.calls] == [48, 2, 1]
    assert render.calls[1][1] == [orders[("class", 2)][6], orders[("sequence", 4)][6]]
    saved = sorted((tmp_path / "render_calls").iterdir())
    assert [p.name for p in saved] == ["train_call_01.txt", "train_call_02.txt", "train_call_03.txt"]
    assert [p.read_text().split() for p in saved] == [keys for _, keys in render.calls]
    summary = S.replacement_summary("train", first, S.ranked(selected), dropped)
    assert summary["dropped"] == 3 and summary["dropped_of_the_first_draw"] == summary["new_keys"] == 2
    assert summary["every_other_key_of_the_first_draw_keeps_its_place"]
    assert summary["first_draw_keys_no_longer_drawn_that_were_not_dropped"] == 0
    assert summary["dropped_by_cell"]["class_T2"] == {"degenerate:clipped": 1, "degenerate:aspect": 1}
    assert summary["dropped_by_reason"] == {"degenerate:clipped": 1, "status:multi_page": 1, "degenerate:aspect": 1}
    # a second run renders nothing and ends with the same list
    render2 = FakeRender()
    again, _, _ = S.settle("train", by_cell, 6, None, set(), [], tmp_path / "render/train",
                           tmp_path / "render_calls", tmp_path, render2, log=lambda *a: None)
    assert again == selected and not render2.calls and len(list((tmp_path / "render_calls").iterdir())) == 3


def test_a_cell_that_cannot_supply_its_number_stops_the_build(tmp_path):
    by_cell = made_up_cells(random.Random(10), sizes=(30,), repos=(200,))
    by_cell[("sequence", 2)] = by_cell[("sequence", 2)][:4]
    with pytest.raises(S.SampleError, match="sequence_T2"):
        S.settle("loss_curve", by_cell, 5, CAP, set(), [], tmp_path / "r", tmp_path / "c", tmp_path, FakeRender(),
                 log=lambda *a: None)
    # the limit per repository can empty a cell as well
    one_repo = {c: [(k, dict(r, repository="same")) for k, r in rows] for c, rows in by_cell.items()}
    with pytest.raises(S.SampleError, match="class_T2"):
        S.settle("loss_curve", one_repo, 5, CAP, set(), [], tmp_path / "r2", tmp_path / "c2", tmp_path, FakeRender(),
                 log=lambda *a: None)


def test_render_py_is_called_with_its_defaults(tmp_path, monkeypatch):
    seen = []

    class Done:
        returncode = 0
    monkeypatch.setattr(S.subprocess, "run", lambda cmd, **kw: seen.append((cmd, kw)) or Done())
    S.run_render_py(tmp_path / "keys.txt", tmp_path / "out", P.PUML_DIR, tmp_path / "log.txt")
    cmd, kw = seen[0]
    assert cmd == [sys.executable, str(P.RENDER_PY), "--src-dir", str(P.PUML_DIR), "--keys", str(tmp_path / "keys.txt"),
                   "--out", str(tmp_path / "out")]
    # the working directory is the folder render_cwd beside the log, not fine-tune/ itself
    assert kw["cwd"] == str((tmp_path / "render_cwd").resolve()) and (tmp_path / "render_cwd").is_dir()
    assert kw["cwd"] != str(FT.resolve())
    # relative paths are made absolute: PlantUML resolves a relative output folder against the sources
    monkeypatch.chdir(tmp_path)
    S.run_render_py(Path("calls/keys.txt"), Path("data/sets/render/dev"), Path("src"), tmp_path / "log.txt")
    here = Path.cwd().resolve()
    assert seen[1][0][2:] == ["--src-dir", str(here / "src"), "--keys", str(here / "calls/keys.txt"),
                              "--out", str(here / "data/sets/render/dev")]
    Done.returncode = 1
    with pytest.raises(S.SampleError, match="exit code 1"):
        S.run_render_py(tmp_path / "keys.txt", tmp_path / "out", P.PUML_DIR, tmp_path / "log.txt")


def test_render_py_drops_a_source_of_two_pages_and_a_degenerate_image(tmp_path, monkeypatch):
    """The real command on 40 made-up sources, five per cell: one renders to two pages, one to
    an image whose padded size is wider than 8:1. The folders are given as relative paths."""
    try:
        P.java_bin()
    except P.PoolError:
        pytest.skip("no Java")
    monkeypatch.chdir(tmp_path)
    tmp_path = Path(".")
    src = tmp_path / "src"
    src.mkdir()
    by_cell = {c: [(f"{c[0][0]}{c[1]}_{i}.puml", dict(primary_type=c[0], tier=c[1], repository=f"r{c[1]}{i}"))
                   for i in range(5)] for c in P.CELLS}
    orders = S.cell_orders(by_cell, 42)
    cell = ("sequence", 1)
    order = orders[cell]
    pages, wide = order[0], order[2]                 # both are in the first draw of three
    for c in P.CELLS:
        for i, (name, _) in enumerate(by_cell[c]):
            body = f"Alice{i} -> Bob{c[1]}: hello {i}\nBob{c[1]} --> Alice{i}: fine"
            if name == pages:
                body = "Alice -> Bob: one\nnewpage\nAlice -> Bob: two"
            if name == wide:
                body = "\n".join(f"participant P{j:02d}" for j in range(40))
            (src / name).write_text(HEADER + "@startuml\n" + body + "\n@enduml\n")
    render = lambda key_file, render_dir, s: S.run_render_py(key_file, render_dir, s, tmp_path / "render.log")
    excluded, dropped = set(), []
    selected, _, first = S.settle("dev", by_cell, 3, CAP, excluded, dropped, tmp_path / "render/dev",
                                  tmp_path / "render_calls", src, render, log=lambda *a: None)
    manifest = S.read_manifest(tmp_path / "render/dev")
    assert first == [k for c in P.CELLS for k in orders[c][:3]] and len(manifest) == 26
    assert manifest[pages]["status"] == "multi_page" and manifest[pages]["pages"] == 2
    row = manifest[wide]
    assert row["status"] == "ok" and max(row["padded_size"]) / min(row["padded_size"]) > 8
    assert {d["key"]: d["reason"] for d in dropped} == {pages: "status:multi_page", wide: "degenerate:aspect"}
    final = keys_by_cell(selected)
    assert final[cell] == [order[1], order[3], order[4]]
    assert all(final[c] == orders[c][:3] for c in P.CELLS if c != cell)
    assert [p.read_text().split() for p in sorted((tmp_path / "render_calls").iterdir())] == [first, order[3:]]
    images = {p.name for p in (tmp_path / "render/dev/puml_images_1568").iterdir()}
    assert {k[:-5] + ".png" for k, _ in selected} <= images
    assert all(manifest[k]["status"] == "ok" and S.render_outcome(manifest[k]) is None for k, _ in selected)


# --------------------------------------------------------------------------- #
# The build on a made-up pool
# --------------------------------------------------------------------------- #
def made_up_source(rng, kind, lines):
    word = lambda: "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(rng.randint(5, 11))).capitalize()
    if kind == "class":
        body = [f"class {word()} {{\n  +{word()}(): {word()}\n}}" for _ in range(lines)]
        body += [f"{word()} --> {word()} : {word()}" for _ in range(lines)]
    else:
        body = [f"{word()} -> {word()}: {word()} {word()}" for _ in range(2 * lines)]
    return HEADER + "@startuml\n" + "\n".join(body) + "\n@enduml\n"


@pytest.fixture(scope="module")
def made_up(tmp_path_factory):
    """A pool of 8 cells x 36 diagrams in 60 repositories, with sources, and a test set of 4."""
    root = tmp_path_factory.mktemp("made_up")
    src, test_dir = root / "src", root / "test"
    src.mkdir()
    test_dir.mkdir()
    rng = random.Random(21)
    pool = []
    for t in P.TYPES:
        for i in range(36 * 4):
            tier = 1 + i % 4
            key = f"{t[0]}{i:03d}.puml"
            pool.append(dict(key=key, primary_type=t, tier=tier, repository=f"repo{rng.randrange(60)}"))
            (src / key).write_text(made_up_source(rng, t, 5 + 2 * tier))
    test = {"diagrams": []}
    for i in range(4):
        key = f"t{i}.puml"
        t = P.TYPES[i % 2]
        (test_dir / key).write_text(made_up_source(rng, t, 8))
        test["diagrams"].append(dict(key=key, primary_type=t, tier=1 + i, repository=f"testrepo{i}"))
    tokens = P.target_token_counts([d["key"] for d in pool], src, workers=2)
    return dict(root=root, src=src, test_dir=test_dir, pool=pool, test=test, tokens=tokens)


SIZES = dict(n_dev=2, n_loss=1, n_train=8, prefixes={"train_20k": 8, "train_5k": 4, "train_1k": 2})


def run(made_up, out, render=None, **kw):
    return S.build(made_up["pool"], made_up["tokens"], made_up["test"], out, out.parent / (out.name + "_work"),
                   src_dir=made_up["src"], test_dir=made_up["test_dir"], workers=2, render=render or FakeRender(),
                   log=lambda *a: None, **dict(SIZES, **kw))


@pytest.fixture(scope="module")
def plain(made_up):
    out = made_up["root"] / "plain"
    render = FakeRender()
    return run(made_up, out, render), out, render


def test_the_three_draws_follow_the_order_of_work(made_up, plain):
    st, out, render = plain
    pool = made_up["pool"]
    keys = lambda ds: [d["key"] for d in ds]
    # D11: the dev set is the capped draw from the whole pool
    dev = S.draw(S.cells_of(pool), 2, 42, CAP)[0]
    assert keys(st["dev"]) == [k for k, _ in dev] and len(dev) == 16
    # every diagram of a dev repository is out of the training pool; the loss-curve split is the
    # capped draw, by a generator of its own, from the other diagrams of those repositories (D12)
    repos, others, rest = S.reserve(pool, keys(st["dev"]))
    assert st["dev_repos"] == repos == {d["repository"] for d in st["dev"]}
    assert keys(st["loss_curve"]) == [k for k, _ in S.draw(S.cells_of(others), 1, 42, CAP)[0]]
    assert {d["repository"] for d in st["loss_curve"]} <= repos
    assert not set(keys(st["loss_curve"])) & set(keys(st["dev"]))
    # D13: the training sample is the uncapped draw from the pool without the dev repositories
    # and without what D7 flags
    assert keys(st["train_pool"]) == [d["key"] for d in rest if d["key"] not in st["sim7"]["removed"]]
    assert not {d["repository"] for d in st["train_pool"]} & repos
    assert keys(st["train"]) == [k for k, _ in S.draw(S.cells_of(st["train_pool"]), 8, 42, None)[0]]
    assert len(st["train"]) == 64 and [d["rank"] for d in st["train"][:8]] == list(range(1, 9))
    # render calls: dev, loss-curve split, training sample, in that order, with their key lists saved
    assert [(name, len(k)) for name, k in render.calls] == [("dev", 16), ("loss_curve", 8), ("train", 64)]
    assert sorted(p.name for p in (out / "render_calls").iterdir()) == ["dev_call_01.txt", "loss_curve_call_01.txt",
                                                                        "train_call_01.txt"]
    assert (out / "render_calls/train_call_01.txt").read_text().split() == keys(st["train"])
    steps = [s["step"] for s in st["steps"]]
    assert steps == ["pool_after_D6", "dev_set_D11", "pool_diagrams_of_the_dev_repositories",
                     "pool_without_the_dev_repositories", "other_pool_diagrams_of_the_dev_repositories",
                     "loss_curve_split_D12", "training_pool_after_D7", "training_sample_D13"]
    by_step = {s["step"]: s for s in st["steps"]}
    assert by_step["pool_after_D6"]["total"] == 288
    assert by_step["pool_diagrams_of_the_dev_repositories"]["total"] + by_step["pool_without_the_dev_repositories"]["total"] == 288
    assert by_step["other_pool_diagrams_of_the_dev_repositories"]["total"] == by_step["pool_diagrams_of_the_dev_repositories"]["total"] - 16
    assert st["checks"]["shared_repositories"] == dict(dev_and_test=0, train_and_test=0, train_and_dev=0,
                                                       loss_curve_and_train=0, loss_curve_and_test=0)
    assert st["checks"]["nested"] == {"train_1k_in_train_5k": True, "train_5k_in_train_20k": True}
    assert st["token_check"]["keys_counted_again"] == 24


def test_d7_removes_a_planted_near_duplicate_of_a_dev_diagram(made_up, plain, tmp_path):
    st, _, _ = plain
    dev_keys = [d["key"] for d in st["dev"]]
    assert not st["sim7"]["removed"] and st["d7"]["removed"]["total"] == 0
    # two keys of the training sample get the text of a dev diagram: one with a word changed
    # (5-gram arm), one re-spaced (edit arm only)
    victims = [d["key"] for d in st["train"] if d["primary_type"] == "sequence"][:2]
    src = tmp_path / "src"
    shutil.copytree(made_up["src"], src)
    dev_seq = next(k for k in dev_keys if k.startswith("s"))
    text = (made_up["src"] / dev_seq).read_text()
    lines = text.split("\n")
    (src / victims[0]).write_text("\n".join(lines[:6] + [lines[6] + "x"] + lines[7:]))
    (src / victims[1]).write_text(text.replace(" -> ", "->").replace(": ", " : "))
    made = dict(made_up, src=src)
    st2 = run(made, tmp_path / "out")
    assert [d["key"] for d in st2["dev"]] == dev_keys                 # the draw reads no source
    assert set(st2["sim7"]["removed"]) == set(victims)
    assert "a" in st2["sim7"]["removed"][victims[0]]["arms"]
    assert st2["sim7"]["removed"][victims[1]]["arms"] == ["b"]
    assert st2["sim7"]["removed"][victims[0]]["a_test_key"] == dev_seq
    assert not set(victims) & {d["key"] for d in st2["train_pool"]} and not set(victims) & {d["key"] for d in st2["train"]}
    assert st2["d7"]["removed"]["total"] == 2 and st2["d7"]["flagged_by_arm"]["b"]["total"] == 2
    assert len(st2["train"]) == 64
    # the training sample is the draw over the pool without them: D7 holds inside the sample
    assert [d["key"] for d in st2["train"]] == [k for k, _ in S.draw(S.cells_of(st2["train_pool"]), 8, 42, None)[0]]
    assert st2["d8"]["training_sample"]["training_diagrams_on_which_an_arm_fires"] == NO_ARM
    S.write_artifacts(st2, tmp_path / "out")
    rows = P.read_jsonl(tmp_path / "out/d7_removed.jsonl")
    assert {r["key"] for r in rows} == set(victims) and rows[0]["a_dev_key"] == dev_seq and "a_test_key" not in rows[0]


def test_a_dropped_key_is_replaced_in_each_of_the_three_draws(made_up, plain, tmp_path):
    st, _, _ = plain
    dev_out = st["dev"][3]["key"]
    render = FakeRender(first_call={"dev": (3, dict(padded_size=[16384, 600])),
                                    "loss_curve": (5, dict(status="no_image", reason="syntax_error")),
                                    "train": (9, dict(padded_size=[600, 5000]))})
    st2 = run(made_up, tmp_path / "out", render)
    assert [(d["set"], d["reason"]) for d in st2["dropped"]] == [
        ("dev", "degenerate:clipped"), ("loss_curve", "status:no_image"), ("train", "degenerate:aspect")]
    assert st2["dropped"][0]["key"] == dev_out and (st2["dropped"][2]["primary_type"], st2["dropped"][2]["tier"]) == ("class", 2)
    dropped = {d["key"] for d in st2["dropped"]}
    pool = made_up["pool"]
    keys = lambda name: [d["key"] for d in st2[name]]
    # each list is the draw over its unchanged candidate list with the exclusion list
    assert keys("dev") == [k for k, _ in S.draw(S.cells_of(pool), 2, 42, CAP, dropped)[0]]
    repos, others, rest = S.reserve(pool, keys("dev"))
    assert keys("loss_curve") == [k for k, _ in S.draw(S.cells_of(others), 1, 42, CAP, dropped)[0]]
    assert keys("train") == [k for k, _ in S.draw(S.cells_of(st2["train_pool"]), 8, 42, None, dropped)[0]]
    assert [name for name, _ in render.calls] == ["dev", "dev", "loss_curve", "loss_curve", "train", "train"]
    assert [len(k) for _, k in render.calls] == [16, 1, 8, 1, 64, 1]
    for name in ("dev", "loss_curve", "train"):
        assert not set(keys(name)) & dropped
        draw = st2["draws"][name]
        assert (draw["dropped"], draw["dropped_of_the_first_draw"], draw["new_keys"]) == (1, 1, 1)
        assert draw["every_other_key_of_the_first_draw_keeps_its_place"]
        assert st2["render"][name]["status_of_the_keys_of_the_set"] == {"ok": len(keys(name))}
        assert st2["render"][name]["manifest_keys_not_in_the_set"] == 1
    assert st2["draws"]["train"]["dropped_by_cell"]["class_T2"] == {"degenerate:aspect": 1}
    assert set(S.count_cells(st2["train"]).values()) == {8} and st2["checks"]["excluded_keys_in_a_set"] == 0
    assert st2["checks"]["nested"] == {"train_1k_in_train_5k": True, "train_5k_in_train_20k": True}
    # the 1k and 5k subsets are prefixes of the order after the replacement
    order = [k for k in S.cell_orders(S.cells_of(st2["train_pool"]), 42)[("class", 2)] if k not in dropped]
    assert [d["key"] for d in st2["train"] if S.cell_of(d) == ("class", 2)] == order[:8]
    assert [d["key"] for d in st2["sets"]["train_1k"] if S.cell_of(d) == ("class", 2)] == order[:2]


def test_clock_sources_and_logged_exceptions_are_counted_and_kept(made_up, plain, tmp_path):
    st, _, _ = plain
    src = tmp_path / "src"
    shutil.copytree(made_up["src"], src)
    clock_dev, clock_train = st["dev"][0]["key"], st["train"][1]["key"]          # rank 2 of its cell: in 1k, 5k, 20k
    trace_train, trace_loss = st["train"][7]["key"], st["loss_curve"][2]["key"]   # rank 8: only in 20k
    for k, line in ((clock_dev, 'footer %date("yyyy")'), (clock_train, "title %date()"), (st["train"][2]["key"], "note: %DATE% x")):
        (src / k).write_text((src / k).read_text().replace("@enduml", line + "\n@enduml"))
    assert st["train"][1]["rank"] == 2 and st["train"][7]["rank"] == 8
    made = dict(made_up, src=src, tokens=P.target_token_counts([d["key"] for d in made_up["pool"]], src, workers=2))
    render = FakeRender({trace_train: dict(renderer_trace=["java.lang.IllegalStateException"]),
                         trace_loss: dict(renderer_trace=["java.lang.NullPointerException"])})
    st2 = run(made, tmp_path / "out", render)
    assert [d["key"] for d in st2["train"]] == [d["key"] for d in st["train"]]       # none is dropped
    count = {name: (v["clock_sources"], v["renderer_trace"]) for name, v in st2["d15"].items()}
    assert count == {"dev": (1, 0), "loss_curve": (0, 1), "train_20k": (1, 1), "train_5k": (1, 0), "train_1k": (1, 0)}
    assert st2["d15"]["train_20k"]["clock_source_keys"] == [clock_train]
    assert st2["d15"]["train_20k"]["renderer_trace_keys"] == [trace_train]
    assert S.CLOCK.pattern == r"%(date|now)\b" and not S.CLOCK.search("%DATE%") and S.CLOCK.search("%now()")


def test_the_nearest_training_diagram_is_found_over_all_pairs(made_up, plain):
    st, _, _ = plain
    text = lambda key, folder: L.read_text(os.path.join(folder, key))
    train = [(d["key"], text(d["key"], made_up["src"])) for d in st["train"]]
    names = {"a": "jaccard_5gram", "b": "edit_similarity", "c": "lines_held"}
    for side, ref, folder in (("test", made_up["test"]["diagrams"], made_up["test_dir"]),
                              ("dev", st["dev"], made_up["src"])):
        table = st[f"neighbours_{side}"]
        assert [r["key"] for r in table] == [d["key"] for d in ref]
        for row in table:
            pairs = [(k, L.compare(source, text(row["key"], folder))) for k, source in train]
            assert row["distinct_lines"] == len(L.line_set(text(row["key"], folder)))
            for arm, name in names.items():
                values = [(r[name] or 0.0) for _, r in pairs]
                top = max(values)
                if arm == "c" and row["distinct_lines"] < 10:
                    assert row[name] is None and row[name + "_train_key"] is None
                    continue
                assert row[name] == pytest.approx(top, abs=1e-12)
                first = next(k for (k, _), v in zip(pairs, values) if v == top) if top > 0 else None
                assert row[name + "_train_key"] == first
        summary = st["d8"]["training_sample"][f"against_the_{side}_set"]
        for name in names.values():
            valued = [r[name] for r in table if r[name] is not None]
            assert summary[name]["largest"]["value"] == max(valued)
            assert summary[name]["reference_diagrams_at_or_above"]["0.5"] == sum(1 for v in valued if v >= 0.5)


def test_the_nearest_row_of_a_column_is_the_first_on_a_tie_and_none_at_zero():
    m = {"cand_len": np.array([10, 10, 10, 0]), "cand_grams": np.array([6, 6, 6, 1]), "cand_lines": np.array([3, 3, 3, 0]),
         "test_len": np.array([10, 10, 10]), "test_grams": np.array([6, 6, 6]), "test_lines": np.array([10, 12, 4]),
         "inter_grams": np.array([[3, 0, 6], [3, 0, 2], [1, 0, 6], [0, 0, 0]], dtype=np.int32),
         "lcs": np.array([[5, 0, 9], [5, 0, 2], [2, 0, 9], [0, 0, 0]], dtype=np.int32),
         "inter_lines": np.array([[2, 0, 4], [5, 0, 1], [5, 0, 4], [0, 0, 0]], dtype=np.int32)}
    best, arg = S.nearest_per_reference(m, chunk=2)             # two chunks: the tie crosses a chunk
    assert list(arg["a"]) == [0, -1, 0] and list(best["a"]) == pytest.approx([3 / 9, 0.0, 1.0])
    assert list(arg["b"]) == [0, -1, 0] and list(best["b"]) == pytest.approx([0.5, 0.0, 0.9])
    assert list(arg["c"]) == [1, -1, -1] and list(best["c"]) == pytest.approx([0.5, 0.0, 0.0])   # column 2: 4 lines
    best, arg = S.nearest_per_reference(m, rows=[2, 1])          # only these rows, in this order
    assert list(arg["a"]) == [1, -1, 0] and list(arg["c"]) == [0, -1, -1]
    ref = [dict(key=f"t{j}", primary_type="class", tier=1) for j in range(3)]
    table = S.neighbour_table(ref, m, best, arg, ["x", "y"])
    assert [(r["jaccard_5gram_train_key"], r["lines_held_train_key"], r["lines_held"]) for r in table] == [
        ("y", "x", 0.5), (None, None, 0.0), ("x", None, None)]


# --------------------------------------------------------------------------- #
# Amendment 1: rule D6 on the diagram block alone
# --------------------------------------------------------------------------- #
VALUES = ("jaccard_5gram", "edit_similarity", "lines_held")
RECORD = {"key", "primary_type", "tier", "repository", "rule", "reference_set", "arms", "pairs",
          "drawn_without_the_amendment"}


def block_of(text):
    return P.scorer_extract_puml()(text)


def prose(rng, lines=30):
    """Lines of words to stand outside a block."""
    word = lambda: "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(rng.randint(4, 9)))
    return "".join(" ".join(word() for _ in range(8)) + "\n" for _ in range(lines))


def outside(source, text):
    """The file `source` with `text` between its header and its block."""
    assert source.count("@startuml") == 1
    return source.replace("@startuml", text + "@startuml")


def respaced_copy(source, rng):
    """A file whose block is the block of `source` with other spacing, behind lines of words."""
    return HEADER + prose(rng) + block_of(source).replace(" -> ", "->").replace(": ", " : ") + "\n"


def planted(made_up, tmp_path, sources, test_sources=None):
    """The made-up pool with some sources replaced ({key: text}), in a folder of its own."""
    src = tmp_path / "src"
    shutil.copytree(made_up["src"], src)
    for k, text in sources.items():
        (src / k).write_text(text)
    made = dict(made_up, src=src)
    if test_sources:
        test_dir = tmp_path / "test"
        shutil.copytree(made_up["test_dir"], test_dir)
        for k, text in test_sources.items():
            (test_dir / k).write_text(text)
        made["test_dir"] = test_dir
    return made


def pair_on_both_texts(cand, ref):
    """leakage.compare of two files on their blocks and on the whole files, as an exclusion record holds it."""
    pick = lambda r: {"arms": r["arms"], **{name: r[name] for name in VALUES}}
    return {"block": pick(L.compare(block_of(cand), block_of(ref))), "whole_file": pick(L.compare(cand, ref))}


def test_rule_d6_on_the_block_alone_flags_a_copy_that_text_outside_the_block_hides(tmp_path):
    foo = "@startuml\nclass Foo {\n}\n@enduml\n"
    talk = "@startuml\nAlice -> Bob: Authentication Request\nBob --> Alice: Authentication Response\n@enduml\n"
    drawing = "".join(f"  {a:<12}{b:>12}\n" for a, b in (("Alice", "Bob"), ("|", "|"), ("|---request-->", "|"), ("|", "|"),
                                                           ("|<--response--", "|"), ("|", "|"), ("first", "second")))
    depot = "@startuml\nclass Depot\nclass Truck\nDepot --> Truck : loads\n@enduml\n"
    note = prose(random.Random(31), 12)
    ref = {"r_foo.puml": HEADER + foo,
           "r_talk.puml": HEADER + talk + drawing,                       # the reference has text outside its block
           "r_note.puml": HEADER + note + "@startuml\nparticipant Kitchen\nKitchen -> Waiter: the order is ready\n@enduml\n",
           "r_depot.puml": HEADER + depot}
    cand = {"c_foo.puml": HEADER + "The class Foo of the first exercise, drawn below as a diagram.\n" + foo,
            "c_talk.puml": HEADER + talk,
            "c_note.puml": HEADER + note + "@startuml\nclass Stock\nclass Order\n@enduml\n",   # the same text outside another block
            "c_depot.puml": HEADER + depot}
    # a pair that differs only by text outside the block: no arm on the whole files, (a) and (b) on the blocks
    for c, r in (("c_foo.puml", "r_foo.puml"), ("c_talk.puml", "r_talk.puml")):
        whole, block = L.compare(cand[c], ref[r]), L.compare(block_of(cand[c]), block_of(ref[r]))
        assert whole["arms"] == [] and whole["jaccard_5gram"] < 0.7 and whole["edit_similarity"] < 0.8
        assert block["arms"] == ["a", "b"] and block["jaccard_5gram"] == block["edit_similarity"] == 1.0
    # both sides are blocks: the block of the candidate against the whole reference file is not flagged
    assert L.compare(block_of(cand["c_talk.puml"]), ref["r_talk.puml"])["arms"] == []
    # the same text outside two other blocks: flagged on the whole files, not on the blocks
    assert L.compare(cand["c_note.puml"], ref["r_note.puml"])["arms"] == ["a", "b", "c"]
    assert L.compare(block_of(cand["c_note.puml"]), block_of(ref["r_note.puml"]))["arms"] == []

    src, ref_dir = tmp_path / "src", tmp_path / "ref"
    src.mkdir()
    ref_dir.mkdir()
    for folder, files in ((src, cand), (ref_dir, ref)):
        for k, text in files.items():
            (folder / k).write_text(text)
    cand_keys, ref_keys = sorted(cand), sorted(ref)
    whole = S.near_duplicates(cand_keys, src, ref_keys, ref_dir, tmp_path / "whole", workers=1)
    assert set(whole["removed"]) == {"c_note.puml", "c_depot.puml"}
    # the blocks are written as csr_runner.extract_puml returns them, one file per key, and measured block against block
    cand_blocks = S.write_blocks(cand_keys, src, tmp_path / "blocks/cand")
    ref_blocks = S.write_blocks(ref_keys, ref_dir, tmp_path / "blocks/ref")
    assert sorted(os.listdir(cand_blocks)) == cand_keys and sorted(os.listdir(ref_blocks)) == ref_keys
    assert all((cand_blocks / k).read_text() == block_of(cand[k]) for k in cand_keys)
    assert all((ref_blocks / k).read_text() == block_of(ref[k]) for k in ref_keys)
    assert (ref_blocks / "r_talk.puml").read_text() == talk.rstrip("\n") and "Kitchen" in (ref_blocks / "r_note.puml").read_text()
    block = S.near_duplicates(cand_keys, cand_blocks, ref_keys, ref_blocks, tmp_path / "block", workers=1)
    assert set(block["removed"]) == {"c_foo.puml", "c_talk.puml", "c_depot.puml"}
    assert block["removed"]["c_talk.puml"]["a_test_key"] == "r_talk.puml" and block["removed"]["c_talk.puml"]["arms"] == ["a", "b"]
    # A1: flagged when an arm fires on the whole file or on the block alone; the pair flagged on the whole file stays flagged
    assert set(whole["removed"]) | set(block["removed"]) == set(cand_keys)
    assert S.block_report(block, {k: ("class", 1) for k in cand_keys})["excluded"]["total"] == 3
    # the exclusion records
    by_key = {k: dict(key=k, primary_type="class", tier=1, repository="repo") for k in cand_keys}
    cand_dirs, ref_dirs = {"block": cand_blocks, "whole_file": src}, {"block": ref_blocks, "whole_file": ref_dir}
    rows = S.flagged("A3", {"block": block}, by_key, cand_dirs, ref_dirs)
    assert [r["key"] for r in rows] == ["c_depot.puml", "c_foo.puml", "c_talk.puml"] and all(set(r) == RECORD for r in rows)
    one = pair_on_both_texts(cand["c_foo.puml"], ref["r_foo.puml"])
    assert rows[1] == {"key": "c_foo.puml", "primary_type": "class", "tier": 1, "repository": "repo", "rule": "A3",
                       "reference_set": "test", "arms": ["a", "b"], "drawn_without_the_amendment": None,
                       "pairs": [dict(reference_key="r_foo.puml", **one)]}
    assert one["block"] == {"arms": ["a", "b"], "jaccard_5gram": 1.0, "edit_similarity": 1.0, "lines_held": None}
    assert one["whole_file"]["arms"] == [] and 0 < one["whole_file"]["edit_similarity"] < 0.8
    assert rows[0]["pairs"][0]["whole_file"]["arms"] == ["a", "b"]                # flagged on both texts
    assert S.flagged("A4", {"block": block}, by_key, cand_dirs, ref_dirs)[2]["reference_set"] == "dev"
    assert S.RULES == {"A3": ("test", ("block",)), "A4": ("dev", ("block",)), "A6": ("loss_curve", ("whole_file", "block"))}
    # A6 is measured on both texts: a record names the texts on which its candidate is flagged; the record of a
    # rule that measures the block alone has no such field
    both = {"whole_file": whole, "block": block}
    rows6 = S.flagged("A6", both, by_key, cand_dirs, ref_dirs)
    assert [(r["key"], r["flagged_on"], r["arms"]) for r in rows6] == [
        ("c_depot.puml", ["whole_file", "block"], ["a", "b"]), ("c_foo.puml", ["block"], ["a", "b"]),
        ("c_note.puml", ["whole_file"], ["a", "b", "c"]), ("c_talk.puml", ["block"], ["a", "b"])]
    assert all(set(r) == RECORD | {"flagged_on"} and (r["rule"], r["reference_set"]) == ("A6", "loss_curve") for r in rows6)
    assert rows6[1]["pairs"] == rows[1]["pairs"] and rows6[2]["pairs"] == [
        dict(reference_key="r_note.puml", **pair_on_both_texts(cand["c_note.puml"], ref["r_note.puml"]))]
    assert rows6[2]["pairs"][0]["block"]["arms"] == [] and rows6[2]["pairs"][0]["whole_file"]["arms"] == ["a", "b", "c"]
    # its report, text by text: the excluded candidates are those flagged on either text
    report = S.rule_report("A6", both, {k: ("class", 1) for k in cand_keys})
    assert report["whole_file"]["excluded"]["total"] == report["block"]["excluded"]["total"] == 4
    combos = lambda text: {arms: v["total"] for arms, v in report[text]["excluded_by_combination_of_arms"].items()}
    assert combos("whole_file") == {"ab": 1, "abc": 1, "none": 2} and combos("block") == {"ab": 3, "none": 1}
    assert report["block"]["flagged_by_arm"]["a"]["total"] == 3 and report["whole_file"]["flagged_by_arm"]["c"]["total"] == 1
    assert S.rule_report("A3", {"block": block}, {k: ("class", 1) for k in cand_keys})["block"] == \
        S.block_report(block, {k: ("class", 1) for k in cand_keys})


def test_a3_excludes_a_block_copy_of_a_test_diagram_from_the_three_draws(made_up, plain, tmp_path):
    st, _, _ = plain
    pool, rng = made_up["pool"], random.Random(32)
    ref = (made_up["test_dir"] / "t1.puml").read_text()
    keys = lambda ds: [d["key"] for d in ds]
    drawn = lambda candidates, n, cap, out: [k for k, _ in S.draw(S.cells_of(candidates), n, 42, cap, out)[0]]
    # four pool keys get a file whose block is a re-spaced copy of a test diagram behind lines of words: one
    # that the dev draw holds, one of the loss-curve draw, one of the training draw, one that no draw reaches
    v_dev = st["dev"][5]["key"]                                          # class T3, rank 2
    dev = drawn(pool, 2, CAP, {v_dev})
    _, others, rest = S.reserve(pool, dev)
    v_loss = drawn(others, 1, CAP, {v_dev})[6]                           # sequence T3
    loss = drawn(others, 1, CAP, {v_dev, v_loss})
    v_train = drawn(rest, 8, None, set())[11]                            # class T2, rank 4
    v_none = S.cell_orders(S.cells_of(rest), 42)[("sequence", 3)][-1]
    four = {v_dev, v_loss, v_train, v_none}
    assert len(four) == 4 and v_none not in drawn(rest, 8, None, set())
    made = planted(made_up, tmp_path, {k: respaced_copy(ref, rng) for k in sorted(four)},
                   {"t1.puml": outside(ref, prose(rng))})                # the test diagram has text outside its block too
    ref = (made["test_dir"] / "t1.puml").read_text()
    for k in four:                                                       # not flagged on the whole file, (b) on the block
        assert L.compare(block_of((made["src"] / k).read_text()), ref)["arms"] == []      # nor block against whole file
        both = pair_on_both_texts((made["src"] / k).read_text(), ref)
        assert both["whole_file"]["arms"] == [] and both["block"]["arms"] == ["b"]
    render = FakeRender()
    st2 = run(made, tmp_path / "out", render)
    rows = {r["key"]: r for r in st2["block_excluded"]}
    assert set(rows) == four and len(st2["block_excluded"]) == 4
    for k, r in rows.items():
        assert set(r) == RECORD and (r["rule"], r["reference_set"], r["arms"]) == ("A3", "test", ["b"])
        assert r["pairs"] == [dict(reference_key="t1.puml", **pair_on_both_texts((made["src"] / k).read_text(), ref))]
        assert (r["primary_type"], r["tier"], r["repository"]) == next(
            (d["primary_type"], d["tier"], d["repository"]) for d in pool if d["key"] == k)
    # each list is the draw over its unchanged candidate list with the four keys passed over
    assert keys(st2["dev"]) == dev and keys(st2["loss_curve"]) == loss and not st2["sim7"]["removed"]
    assert keys(st2["train_pool"]) == keys(rest) and {v_train, v_none} <= set(keys(st2["train_pool"]))
    assert keys(st2["train"]) == drawn(rest, 8, None, four)
    assert not four & set(keys(st2["dev"]) + keys(st2["loss_curve"]) + keys(st2["train"]))
    assert st2["checks"]["block_excluded_keys_in_a_set"] == 0 and st2["excluded"] == four
    assert not any(k in four for _, called in render.calls for k in called)          # none is rendered
    # the record says which draw holds the key without the exclusion, and at which rank
    assert rows[v_dev]["drawn_without_the_amendment"] == {"set": "dev", "rank": 2}
    assert rows[v_loss]["drawn_without_the_amendment"] == {"set": "loss_curve", "rank": 1}
    assert rows[v_train]["drawn_without_the_amendment"] == {"set": "train", "rank": 4}
    assert rows[v_none]["drawn_without_the_amendment"] is None
    effect = {name: st2["draws"][name]["amendment_1"] for name in S.SETS}
    assert effect["dev"]["excluded_keys_the_draw_holds_without_the_exclusion"] == [dict(key=v_dev, cell="class_T3", rank=2)]
    assert effect["loss_curve"]["excluded_keys_the_draw_holds_without_the_exclusion"] == [dict(key=v_loss, cell="sequence_T3", rank=1)]
    order = S.cell_orders(S.cells_of(rest), 42)[("class", 2)]
    assert effect["train"] == {"excluded_keys_the_draw_holds_without_the_exclusion": [dict(key=v_train, cell="class_T2", rank=4)],
                               "keys_that_enter": [dict(key=order[8], cell="class_T2", rank=8)],
                               "every_other_key_keeps_its_place": True}
    assert [d["key"] for d in st2["train"] if S.cell_of(d) == ("class", 2)] == [k for k in order[:9] if k != v_train]
    assert st2["d8"]["block_alone"]["A3"]["excluded"]["total"] == 4
    assert st2["d8"]["block_alone"]["A3"]["candidates_compared"] == len(pool)
    assert st2["d8"]["block_alone"]["A3"]["flagged_by_arm"]["b"]["total"] == 4
    assert st2["d8"]["block_alone"]["A4"]["excluded"]["total"] == 0
    # the artifact
    S.write_artifacts(st2, tmp_path / "out")
    written = P.read_jsonl(tmp_path / "out/block_excluded.jsonl")
    assert written == st2["block_excluded"] and [r["key"] for r in written] == [d["key"] for d in pool if d["key"] in four]
    assert set(written[0]["pairs"][0]) == {"reference_key", "block", "whole_file"}
    assert set(written[0]["pairs"][0]["block"]) == set(written[0]["pairs"][0]["whole_file"]) == {"arms", *VALUES}
    counts = json.load(open(tmp_path / "out/counts.json"))["amendment_1"]
    assert counts["A3"]["excluded"]["total"] == 4 and counts["A4"]["excluded"]["total"] == 0 and counts["excluded_keys"] == 4
    assert counts["excluded_keys_a_draw_holds_without_the_exclusion"] == {"dev": 1, "loss_curve": 1, "train": 1}
    assert P.read_jsonl(tmp_path / "out/dropped.jsonl") == []


def test_a4_excludes_a_block_copy_of_a_dev_diagram_from_the_training_draw_only(made_up, plain, tmp_path):
    st, _, _ = plain
    rng = random.Random(33)
    keys = lambda ds: [d["key"] for d in ds]
    dev_seq = next(k for k in keys(st["dev"]) if k.startswith("s"))
    ref = (made_up["src"] / dev_seq).read_text()
    orders = S.cell_orders(S.cells_of(st["train_pool"]), 42)
    cell = ("sequence", 1)
    v = orders[cell][2]                                                  # rank 3 of its cell in the training draw
    v_none = orders[("class", 4)][-1]                                    # in the training pool, not reached by the draw
    v_loss = st["loss_curve"][3]["key"]                                  # a key of the loss-curve split: not in the training pool
    assert v in keys(st["train"]) and v_none not in keys(st["train"]) and v_none in keys(st["train_pool"])
    made = planted(made_up, tmp_path, dict({k: respaced_copy(ref, rng) for k in (v, v_none, v_loss)},
                                           **{dev_seq: outside(ref, prose(rng))}))      # the dev diagram has text outside its block too
    made["tokens"] = dict(made_up["tokens"], **P.target_token_counts([v_loss], made["src"], workers=1))
    ref = (made["src"] / dev_seq).read_text()
    for k in (v, v_none, v_loss):
        assert L.compare(block_of((made["src"] / k).read_text()), ref)["arms"] == []      # block against whole file: no arm
        both = pair_on_both_texts((made["src"] / k).read_text(), ref)
        assert both["whole_file"]["arms"] == [] and both["block"]["arms"] == ["b"]
    render = FakeRender()
    st2 = run(made, tmp_path / "out", render)
    # D7, on the whole file, flags none of them; A4 flags the two of the training pool
    assert not st2["sim7"]["removed"] and keys(st2["train_pool"]) == keys(st["train_pool"])
    rows = {r["key"]: r for r in st2["block_excluded"]}
    assert set(rows) == {v, v_none}
    for k, r in rows.items():
        assert (r["rule"], r["reference_set"], r["arms"]) == ("A4", "dev", ["b"])
        assert r["pairs"] == [dict(reference_key=dev_seq, **pair_on_both_texts((made["src"] / k).read_text(), ref))]
    assert rows[v]["drawn_without_the_amendment"] == {"set": "train", "rank": 3}
    assert rows[v_none]["drawn_without_the_amendment"] is None
    # the dev set and the loss-curve split are those of the pool without the planted files; the loss-curve key stays
    assert keys(st2["dev"]) == keys(st["dev"]) and keys(st2["loss_curve"]) == keys(st["loss_curve"])
    assert v_loss in keys(st2["loss_curve"])
    passed = lambda name: st2["draws"][name]["excluded_keys_passed_over"]
    assert set(passed("dev").values()) == set(passed("loss_curve").values()) == {0}       # the list is empty in both draws
    assert passed("train") == {S.cell_name(c): int(c == cell) for c in P.CELLS}
    assert st2["draws"]["dev"]["amendment_1"] == st2["draws"]["loss_curve"]["amendment_1"] == NO_EFFECT
    # the drawn key is replaced by the next key of its cell's order; every other pick and the permutations are unchanged
    before, after = {c: [d["key"] for d in st["train"] if S.cell_of(d) == c] for c in P.CELLS}, \
                    {c: [d["key"] for d in st2["train"] if S.cell_of(d) == c] for c in P.CELLS}
    assert before[cell] == orders[cell][:8] and after[cell] == [k for k in orders[cell][:9] if k != v]
    assert all(after[c] == before[c] for c in P.CELLS if c != cell)
    assert S.cell_orders(S.cells_of(st2["train_pool"]), 42) == orders
    assert keys(st2["train"]) == [k for k, _ in S.draw(S.cells_of(st2["train_pool"]), 8, 42, None, {v, v_none})[0]]
    assert st2["draws"]["train"]["amendment_1"] == {
        "excluded_keys_the_draw_holds_without_the_exclusion": [dict(key=v, cell="sequence_T1", rank=3)],
        "keys_that_enter": [dict(key=orders[cell][8], cell="sequence_T1", rank=8)], "every_other_key_keeps_its_place": True}
    assert [d["rank"] for d in st2["train"] if S.cell_of(d) == cell] == list(range(1, 9))
    # the replacement is the one key rendered beside the keys of the first draw
    assert [(name, len(called)) for name, called in render.calls] == [("dev", 16), ("loss_curve", 8), ("train", 64)]
    assert v not in render.calls[2][1] and orders[cell][8] in render.calls[2][1]
    assert st2["d8"]["block_alone"]["A4"]["excluded"] == dict({S.cell_name(c): 0 for c in P.CELLS}, sequence_T1=1, class_T4=1, total=2)
    assert st2["d8"]["block_alone"]["A4"]["candidates_compared"] == len(st["train_pool"])
    assert st2["d8"]["block_alone"]["A4"]["reference_diagrams"] == 16
    assert st2["d8"]["set_keys_on_which_an_arm_fires"] == {text: {pair: ZERO for pair in PAIRS}
                                                           for text in ("whole_file", "block_alone")}
    # A6 flags the same two keys on the block (the planted loss-curve key has their block); Amendment 1 has excluded
    # them already, so no draw holds them before Amendment 2 and Amendment 2 changes no list
    assert {r["key"]: (r["rule"], r["flagged_on"], r["drawn_without_the_amendment"]) for r in st2["loss_curve_excluded"]} == {
        v: ("A6", ["block"], None), v_none: ("A6", ["block"], None)}
    assert all(r["pairs"][0]["reference_key"] == v_loss for r in st2["loss_curve_excluded"])
    assert all(st2["draws"][name]["amendment_2"] == NO_EFFECT for name in S.SETS)


def test_an_excluded_key_that_no_draw_holds_changes_nothing(made_up, plain, tmp_path):
    st, out, render = plain
    rng = random.Random(35)
    keys = lambda ds: [d["key"] for d in ds]
    orders = S.cell_orders(S.cells_of(st["train_pool"]), 42)
    far_test, far_dev, far_loss = orders[("sequence", 2)][-1], orders[("class", 1)][-1], orders[("sequence", 4)][-1]
    dev_seq = next(k for k in keys(st["dev"]) if k.startswith("s"))
    loss_seq = next(k for k in keys(st["loss_curve"]) if k.startswith("s"))
    made = planted(made_up, tmp_path, {
        far_test: respaced_copy((made_up["test_dir"] / "t3.puml").read_text(), rng),
        far_dev: respaced_copy((made_up["src"] / dev_seq).read_text(), rng),
        far_loss: respaced_copy((made_up["src"] / loss_seq).read_text(), rng)})
    render2 = FakeRender()
    st2 = run(made, tmp_path / "out", render2)
    assert [(r["key"], r["rule"], r["drawn_without_the_amendment"]) for r in st2["block_excluded"]] == [
        (far_test, "A3", None), (far_dev, "A4", None)]
    assert [(r["key"], r["rule"], r["flagged_on"], r["drawn_without_the_amendment"]) for r in st2["loss_curve_excluded"]] == [
        (far_loss, "A6", ["block"], None)]
    for name in S.SETS:
        assert st2[name] == st[name]                                     # keys, ranks, token counts
        assert st2["draws"][name]["amendment_1"] == st2["draws"][name]["amendment_2"] == NO_EFFECT
    assert render2.calls == render.calls and keys(st2["train_pool"]) == keys(st["train_pool"])
    S.write_artifacts(st2, tmp_path / "out")
    S.write_artifacts(st, out)
    for name in ("dev.json", "loss_curve.json", "train.json", "dropped.jsonl", "neighbours_test.tsv", "neighbours_dev.tsv"):
        assert (tmp_path / "out" / name).read_bytes() == (out / name).read_bytes()
    assert (out / "block_excluded.jsonl").read_text() == "" and len(P.read_jsonl(tmp_path / "out/block_excluded.jsonl")) == 2
    assert (out / "loss_curve_excluded.jsonl").read_text() == "" and len(P.read_jsonl(tmp_path / "out/loss_curve_excluded.jsonl")) == 1


def test_a_candidate_flagged_on_the_whole_file_stays_removed_whatever_its_block_gives(made_up, plain, tmp_path):
    st, _, _ = plain
    rng = random.Random(34)
    dev_seq = next(d["key"] for d in st["dev"] if d["primary_type"] == "sequence")
    shared, both = [d["key"] for d in st["train"] if d["primary_type"] == "sequence"][:2]
    note = prose(rng, 100)
    dev_text = (made_up["src"] / dev_seq).read_text()
    lines = dev_text.split("\n")
    made = planted(made_up, tmp_path, {
        dev_seq: outside(dev_text, note),                                          # the dev diagram, with text outside its block
        shared: outside((made_up["src"] / shared).read_text(), note),              # the same text outside another diagram
        both: outside("\n".join(lines[:6] + [lines[6] + "x"] + lines[7:]), note)})  # the dev file with a word changed in the block
    block = lambda k: block_of((made["src"] / k).read_text())
    assert L.compare(block(shared), block(dev_seq))["arms"] == [] and "a" in L.compare(block(both), block(dev_seq))["arms"]
    st2 = run(made, tmp_path / "out")
    assert [d["key"] for d in st2["dev"]] == [d["key"] for d in st["dev"]]
    removed = st2["sim7"]["removed"]
    assert set(removed) == {shared, both} and removed[shared]["arms"] == ["a", "b", "c"]
    assert removed[shared]["a_test_key"] == dev_seq
    pool_keys = {d["key"] for d in st2["train_pool"]}
    assert not {shared, both} & pool_keys and not {shared, both} & {d["key"] for d in st2["train"]}
    # A4 reads the training pool after D7: neither is among its candidates, and it flags nothing more
    assert st2["block_excluded"] == [] and st2["rules"]["A4"]["block"]["candidates_compared"] == len(pool_keys)
    assert not {shared, both} & set(st2["sims"]["A4"]["block"]["keys"])
    assert [d["key"] for d in st2["train"]] == [k for k, _ in S.draw(S.cells_of(st2["train_pool"]), 8, 42, None)[0]]


def test_the_build_stops_before_a_render_when_a_frozen_list_would_change(made_up, plain, tmp_path):
    st, out, _ = plain
    rng = random.Random(36)
    keys = lambda ds: [d["key"] for d in ds]
    frozen = {"dev": keys(st["dev"]), "loss_curve": keys(st["loss_curve"])}
    ref = (made_up["test_dir"] / "t1.puml").read_text()
    # the lists as written: the build runs through and gives them again
    render = FakeRender()
    same = run(made_up, tmp_path / "same", render, frozen=frozen)
    assert keys(same["dev"]) == frozen["dev"] and keys(same["loss_curve"]) == frozen["loss_curve"] and len(render.calls) == 3
    # a dev key that A3 excludes: stopped before the dev render
    (tmp_path / "a").mkdir()
    made = planted(made_up, tmp_path / "a", {st["dev"][5]["key"]: respaced_copy(ref, rng)})
    render = FakeRender()
    with pytest.raises(S.SampleError, match="dev: the draw does not give the frozen list: 1 keys leave"):
        run(made, tmp_path / "a/out", render, frozen=frozen)
    assert render.calls == [] and not (tmp_path / "a/out/render").exists()
    # a loss-curve key that A3 excludes: the dev set is rendered, the loss-curve split is not
    (tmp_path / "b").mkdir()
    made = planted(made_up, tmp_path / "b", {st["loss_curve"][6]["key"]: respaced_copy(ref, rng)})
    render = FakeRender()
    with pytest.raises(S.SampleError, match="loss_curve: the draw does not give the frozen list"):
        run(made, tmp_path / "b/out", render, frozen=frozen)
    assert [name for name, _ in render.calls] == ["dev"]
    # the frozen lists are read from the key lists of the output folder
    assert S.frozen_lists(tmp_path / "nothing") == {}
    S.write_artifacts(st, out)
    assert S.frozen_lists(out) == frozen and S.FROZEN == ("dev", "loss_curve")


def test_the_draw_from_the_manifest_is_the_settled_draw_without_a_render(tmp_path):
    by_cell = made_up_cells(random.Random(9), sizes=(40,), repos=(200,))
    orders = S.cell_orders(by_cell, 42)
    clipped, second = orders[("class", 2)][1], orders[("class", 2)][6]
    render = FakeRender({clipped: dict(padded_size=[16384, 9000]), second: dict(padded_size=[512, 5000])})
    selected, _, _ = S.settle("dev", by_cell, 6, CAP, set(), [], tmp_path / "render/dev", tmp_path / "calls", tmp_path,
                              render, log=lambda *a: None)
    manifest = S.read_manifest(tmp_path / "render/dev")
    assert S.draw_from_manifest(by_cell, 6, 42, CAP, set(), manifest) == [k for k, _ in selected]
    # with an empty manifest it is the plain draw; a key on the exclusion list is passed over
    assert S.draw_from_manifest(by_cell, 6, 42, CAP, set(), {}) == [k for k, _ in S.draw(by_cell, 6, 42, CAP)[0]]
    out = orders[("sequence", 1)][0]
    assert S.draw_from_manifest(by_cell, 6, 42, CAP, {out}, manifest) == [
        k for k, _ in S.draw(by_cell, 6, 42, CAP, {out, clipped, second})[0]]


def test_the_block_alone_neighbour_tables_are_found_over_all_pairs_of_blocks(made_up, plain, tmp_path):
    st, _, _ = plain
    rng = random.Random(37)
    text = lambda key, folder: L.read_text(os.path.join(folder, key))
    # text outside the block of every fifth training diagram, of two dev diagrams and of two test diagrams
    with_text = [d["key"] for d in st["train"]][::5] + [st["dev"][1]["key"], st["dev"][10]["key"]]
    made = planted(made_up, tmp_path,
                   {k: outside((made_up["src"] / k).read_text(), prose(rng, 12)) for k in with_text},
                   {k: outside((made_up["test_dir"] / k).read_text(), prose(rng, 12)) for k in ("t1.puml", "t2.puml")})
    st2 = run(made, tmp_path / "out")
    for name in S.SETS:
        assert st2[name] == st[name]
    assert st2["block_excluded"] == []
    train = [(d["key"], text(d["key"], made["src"])) for d in st2["train"]]
    names = {"a": "jaccard_5gram", "b": "edit_similarity", "c": "lines_held"}
    for side, ref, folder in (("test", made["test"]["diagrams"], made["test_dir"]), ("dev", st2["dev"], made["src"])):
        whole, table = st2[f"neighbours_{side}"], st2[f"neighbours_{side}_block"]
        assert [r["key"] for r in table] == [d["key"] for d in ref] and [set(r) for r in table] == [set(r) for r in whole]
        for row, whole_row in zip(table, whole):
            source = text(row["key"], folder)
            pairs = [(k, L.compare(block_of(s), block_of(source))) for k, s in train]
            assert row["distinct_lines"] == len(L.line_set(block_of(source)))
            added = 12 if row["key"] in with_text + ["t1.puml", "t2.puml"] else 0       # the lines outside the block
            assert whole_row["distinct_lines"] == len(L.line_set(source)) == row["distinct_lines"] + added
            for arm, name in names.items():
                values = [(r[name] or 0.0) for _, r in pairs]
                top = max(values)
                if arm == "c" and row["distinct_lines"] < 10:
                    assert row[name] is None and row[name + "_train_key"] is None
                    continue
                assert row[name] == pytest.approx(top, abs=1e-12)
                first = next(k for (k, _), v in zip(pairs, values) if v == top) if top > 0 else None
                assert row[name + "_train_key"] == first
        summary = st2["d8"]["block_alone"]["training_sample"][f"against_the_{side}_set"]
        for name in names.values():
            valued = [r[name] for r in table if r[name] is not None]
            assert summary[name]["largest"]["value"] == max(valued)
            assert summary[name]["reference_diagrams_at_or_above"]["0.5"] == sum(1 for v in valued if v >= 0.5)
    # the tables on the blocks equal the tables of the pool without the text outside the blocks
    assert st2["neighbours_test_block"] == st["neighbours_test"] == st["neighbours_test_block"]
    assert st2["neighbours_dev_block"] == st["neighbours_dev"] and st2["neighbours_test"] != st["neighbours_test"]
    assert st2["d8"]["block_alone"]["training_sample"]["training_diagrams_on_which_an_arm_fires"] == NO_ARM
    # the files: beside the tables of the whole file, with the same columns and one row per reference diagram
    S.write_artifacts(st2, tmp_path / "out")
    for side, n in (("test", 4), ("dev", 16)):
        whole = (tmp_path / f"out/neighbours_{side}.tsv").read_text().splitlines()
        block = (tmp_path / f"out/neighbours_{side}_block.tsv").read_text().splitlines()
        assert block[0] == whole[0] and len(block) == len(whole) == n + 1
        assert [l.split("\t")[0] for l in block] == [l.split("\t")[0] for l in whole]
    report = json.load(open(tmp_path / "out/d8_report.json"))
    assert set(report) == {"D7", "training_sample", "block_alone", "A6", "set_keys_on_which_an_arm_fires"}
    assert set(report["block_alone"]) == {"A3", "A4", "training_sample"}
    assert set(report["block_alone"]["training_sample"]) == set(report["training_sample"])
    config = json.load(open(tmp_path / "out/config.json"))["parameters"]["Amendment_1"]
    assert config["A3"]["excluded_from"] == ["dev", "loss_curve", "train"] and config["A4"]["excluded_from"] == ["train"]
    assert config["extract_puml_source_sha256"] == P.PINS["extract_puml_source"]


# --------------------------------------------------------------------------- #
# Amendment 2: rule D6 between the training pool and the loss-curve split, on both texts
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def loss_copies(made_up, plain, tmp_path_factory):
    """The made-up pool with three candidates of the training pool planted as near-duplicates of a
    loss-curve diagram: one on the whole file only (the same long text outside two other blocks),
    one on the block alone only (the block re-spaced, behind other words), both drawn; and one on
    the block alone that the training draw does not reach."""
    st, _, _ = plain
    tmp_path, rng = tmp_path_factory.mktemp("loss_copies"), random.Random(41)
    source = lambda k: (made_up["src"] / k).read_text()
    orders = S.cell_orders(S.cells_of(st["train_pool"]), 42)
    loss_note = next(d["key"] for d in st["loss_curve"] if d["primary_type"] == "class")
    loss_seq = next(d["key"] for d in st["loss_curve"] if d["primary_type"] == "sequence")
    v_whole, v_block, v_none = orders[("class", 2)][1], orders[("sequence", 3)][5], orders[("sequence", 1)][-1]
    note = prose(rng, 300)
    made = planted(made_up, tmp_path, {loss_note: outside(source(loss_note), note), v_whole: outside(source(v_whole), note),
                                       v_block: respaced_copy(source(loss_seq), rng), v_none: respaced_copy(source(loss_seq), rng)})
    render = FakeRender()
    st2 = run(made, tmp_path / "out", render)
    return dict(st=st2, made=made, out=tmp_path / "out", render=render, orders=orders, loss_note=loss_note, loss_seq=loss_seq,
                v_whole=v_whole, v_block=v_block, v_none=v_none)


def test_a6_excludes_a_copy_of_a_loss_curve_diagram_on_the_whole_file_or_on_the_block_alone_from_the_training_draw(plain, loss_copies):
    st, c = plain[0], loss_copies
    st2, orders, v_whole, v_block, v_none = c["st"], c["orders"], c["v_whole"], c["v_block"], c["v_none"]
    keys = lambda ds: [d["key"] for d in ds]
    source = lambda k: (c["made"]["src"] / k).read_text()
    assert v_whole in keys(st["train"]) and v_block in keys(st["train"]) and v_none not in keys(st["train"])
    # the planted pairs: one fires on the whole files only, the others on the blocks only
    one = pair_on_both_texts(source(v_whole), source(c["loss_note"]))
    assert one["whole_file"]["arms"] == ["a", "b", "c"] and one["block"]["arms"] == []
    for k in (v_block, v_none):
        both = pair_on_both_texts(source(k), source(c["loss_seq"]))
        assert both["whole_file"]["arms"] == [] and both["block"]["arms"] == ["b"]
    # D7 and Amendment 1 flag none of them; the candidate list of the training draw and its permutations are unchanged
    assert not st2["sim7"]["removed"] and st2["block_excluded"] == [] and keys(st2["train_pool"]) == keys(st["train_pool"])
    assert S.cell_orders(S.cells_of(st2["train_pool"]), 42) == orders and {v_whole, v_block, v_none} <= set(keys(st2["train_pool"]))
    # A6 flags the three; the two drawn keys are replaced by the next key of their cell's order, nothing else changes
    assert {r["key"] for r in st2["loss_curve_excluded"]} == {v_whole, v_block, v_none} == st2["excluded"]
    before = {cell: [d["key"] for d in st["train"] if S.cell_of(d) == cell] for cell in P.CELLS}
    after = {cell: [d["key"] for d in st2["train"] if S.cell_of(d) == cell] for cell in P.CELLS}
    changed = {("class", 2): v_whole, ("sequence", 3): v_block}
    for cell in P.CELLS:
        assert before[cell] == orders[cell][:8]
        assert after[cell] == ([k for k in orders[cell][:9] if k != changed[cell]] if cell in changed else before[cell])
    assert keys(st2["train"]) == [k for k, _ in S.draw(S.cells_of(st2["train_pool"]), 8, 42, None, {v_whole, v_block, v_none})[0]]
    assert all([d["rank"] for d in st2["train"] if S.cell_of(d) == cell] == list(range(1, 9)) for cell in P.CELLS)
    assert st2["draws"]["train"]["amendment_2"] == {
        "excluded_keys_the_draw_holds_without_the_exclusion": [dict(key=v_whole, cell="class_T2", rank=2),
                                                               dict(key=v_block, cell="sequence_T3", rank=6)],
        "keys_that_enter": [dict(key=orders[("class", 2)][8], cell="class_T2", rank=8),
                            dict(key=orders[("sequence", 3)][8], cell="sequence_T3", rank=8)],
        "every_other_key_keeps_its_place": True}
    assert st2["draws"]["train"]["amendment_1"] == NO_EFFECT
    assert st2["draws"]["train"]["excluded_keys_passed_over"] == {S.cell_name(cell): int(cell in changed) for cell in P.CELLS}
    # the subsets are the first keys of the order after the replacement: the key at rank 2 leaves the smallest one too
    assert [d["key"] for d in st2["sets"]["train_1k"] if S.cell_of(d) == ("class", 2)] == [orders[("class", 2)][0], orders[("class", 2)][2]]
    assert st2["checks"]["nested"] == {"train_1k_in_train_5k": True, "train_5k_in_train_20k": True}
    assert st2["checks"]["loss_curve_excluded_keys_in_a_set"] == st2["checks"]["block_excluded_keys_in_a_set"] == 0
    # D14: the keys that enter are rendered with the draw; the excluded keys are not rendered
    called = c["render"].calls
    assert [(name, len(k)) for name, k in called] == [("dev", 16), ("loss_curve", 8), ("train", 64)]
    assert not {v_whole, v_block, v_none} & set(called[2][1]) and {orders[("class", 2)][8], orders[("sequence", 3)][8]} <= set(called[2][1])
    # no arm of D6 on a training key against a loss-curve diagram, on either text
    assert st2["d8"]["set_keys_on_which_an_arm_fires"] == {text: {pair: ZERO for pair in PAIRS} for text in ("whole_file", "block_alone")}


def test_a6_leaves_the_dev_set_and_the_loss_curve_split_as_they_are(plain, loss_copies):
    st, st2 = plain[0], loss_copies["st"]
    for name in ("dev", "loss_curve"):
        assert st2[name] == st[name]                                      # keys, ranks, token counts
        assert st2["draws"][name]["amendment_1"] == st2["draws"][name]["amendment_2"] == NO_EFFECT
        assert set(st2["draws"][name]["excluded_keys_passed_over"].values()) == {0}      # the list is empty in both draws
    assert {loss_copies["loss_note"], loss_copies["loss_seq"]} <= {d["key"] for d in st2["loss_curve"]}
    assert st2["dev_repos"] == st["dev_repos"] and loss_copies["render"].calls[:2] == plain[2].calls[:2]


def test_the_records_and_the_report_of_amendment_2(plain, loss_copies):
    st, c = plain[0], loss_copies
    st2, out, v_whole, v_block, v_none = c["st"], c["out"], c["v_whole"], c["v_block"], c["v_none"]
    source = lambda k: (c["made"]["src"] / k).read_text()
    rows = {r["key"]: r for r in st2["loss_curve_excluded"]}
    # A8: key, cell, the text or texts, the arms, the loss-curve diagram with the three values, whether it was drawn
    want = {v_whole: (["whole_file"], ["a", "b", "c"], c["loss_note"], {"set": "train", "rank": 2}),
            v_block: (["block"], ["b"], c["loss_seq"], {"set": "train", "rank": 6}),
            v_none: (["block"], ["b"], c["loss_seq"], None)}
    by_key = {d["key"]: d for d in c["made"]["pool"]}
    for k, (texts, arms, ref, drawn) in want.items():
        r = rows[k]
        assert set(r) == RECORD | {"flagged_on"} and (r["rule"], r["reference_set"]) == ("A6", "loss_curve")
        assert (r["primary_type"], r["tier"], r["repository"]) == tuple(by_key[k][f] for f in ("primary_type", "tier", "repository"))
        assert (r["flagged_on"], r["arms"], r["drawn_without_the_amendment"]) == (texts, arms, drawn)
        assert r["pairs"] == [dict(reference_key=ref, **pair_on_both_texts(source(k), source(ref)))]
    # the report of the rule, text by text, and the largest values inside the final sample against the split
    cells = dict({S.cell_name(cell): 0 for cell in P.CELLS}, class_T2=1, sequence_T3=1, sequence_T1=1, total=3)
    report = st2["d8"]["A6"]
    assert set(report) == {"whole_file", "block_alone"}
    for text, combos in (("whole_file", {"abc": 1, "none": 2}), ("block_alone", {"b": 2, "none": 1})):
        assert report[text]["excluded"] == cells and report[text]["candidates_compared"] == len(st["train_pool"])
        assert report[text]["reference_diagrams"] == 8 and report[text]["pairs"] == 8 * len(st["train_pool"])
        assert {arms: v["total"] for arms, v in report[text]["excluded_by_combination_of_arms"].items()} == combos
    train = [(d["key"], source(d["key"])) for d in st2["train"]]
    for part, table, read in ((st2["d8"]["training_sample"], st2["neighbours_loss_curve"], lambda t: t),
                              (st2["d8"]["block_alone"]["training_sample"], st2["neighbours_loss_curve_block"], block_of)):
        assert part["training_diagrams_on_which_an_arm_fires"] == NO_ARM
        assert [r["key"] for r in table] == [d["key"] for d in st2["loss_curve"]]
        summary = part["against_the_loss_curve_split"]
        assert summary["reference_diagrams"] == 8
        pairs = {r["key"]: [L.compare(read(text), read(source(r["key"]))) for _, text in train] for r in table}
        for name in VALUES:
            valued = [r for r in table if r[name] is not None]
            for r in valued:                                              # every pair of the final sample, on this text
                assert r[name] == pytest.approx(max(p[name] or 0.0 for p in pairs[r["key"]]), abs=1e-12)
            top = max(valued, key=lambda r: r[name])
            assert (summary[name]["largest"]["value"], summary[name]["largest"]["reference"]) == (top[name], top["key"])
            assert summary[name]["largest"]["train_key"] == top[name + "_train_key"]
            assert summary[name]["reference_diagrams_with_a_value"] == len(valued)
        assert summary["jaccard_5gram"]["largest"]["value"] < 0.7 and summary["edit_similarity"]["largest"]["value"] < 0.8
    # the artifacts
    names = S.write_artifacts(st2, out)
    assert "loss_curve_excluded.jsonl" in names and "loss_curve_excluded.jsonl" in (out / "SHA256SUMS").read_text()
    written = P.read_jsonl(out / "loss_curve_excluded.jsonl")
    assert written == st2["loss_curve_excluded"]
    assert [r["key"] for r in written] == [d["key"] for d in st2["train_pool"] if d["key"] in rows]     # in the order of the pool
    assert list(written[0]) == ["key", "primary_type", "tier", "repository", "rule", "reference_set", "flagged_on", "arms",
                                "pairs", "drawn_without_the_amendment"]
    assert set(written[0]["pairs"][0]) == {"reference_key", "block", "whole_file"}
    assert set(written[0]["pairs"][0]["block"]) == set(written[0]["pairs"][0]["whole_file"]) == {"arms", *VALUES}
    assert (out / "block_excluded.jsonl").read_text() == ""
    counts = json.load(open(out / "counts.json"))
    assert counts["amendment_2"] == {
        "A6": {"candidates_compared": len(st["train_pool"]), "reference_diagrams": 8, "excluded": cells}, "excluded_keys": 3,
        "excluded_keys_a_draw_holds_without_the_exclusion": {"dev": 0, "loss_curve": 0, "train": 2}}
    assert counts["amendment_1"]["excluded_keys"] == 0 and list(counts["amendment_1"]) == [
        "A3", "A4", "excluded_keys", "excluded_keys_a_draw_holds_without_the_exclusion"]
    saved = json.load(open(out / "d8_report.json"))
    assert saved["A6"] == json.loads(json.dumps(report)) and "against_the_loss_curve_split" in saved["training_sample"]
    config = json.load(open(out / "config.json"))
    assert config["parameters"]["Amendment_2"]["A6"] == {
        "candidates": "training pool after D7", "reference": "the diagrams of the loss-curve split",
        "texts": ["whole_file", "block"], "excluded_from": ["train"]}
    assert not (out / "neighbours_loss_curve.tsv").exists()              # A8 asks for the largest values, not for a table


# --------------------------------------------------------------------------- #
# D16: the artifacts
# --------------------------------------------------------------------------- #
def snapshot(folder):
    return {str(p.relative_to(folder)): p.read_bytes() for p in sorted(Path(folder).rglob("*")) if p.is_file()}


def test_the_artifacts_are_byte_identical_on_a_second_write_and_on_a_second_run(made_up, tmp_path):
    out = tmp_path / "out"
    render = FakeRender({})
    st = run(made_up, out, render)
    names = S.write_artifacts(st, out, {"pool": {"path": "made up"}})
    first = snapshot(out)
    assert set(names) == set(first) - {"SHA256SUMS"}
    assert {"dev.json", "loss_curve.json", "train.json", "config.json", "counts.json", "dropped.jsonl", "d7_removed.jsonl",
            "d8_report.json", "neighbours_test.tsv", "neighbours_dev.tsv", "render/dev/manifest.jsonl",
            "render/loss_curve/manifest.jsonl", "render/train/manifest.jsonl", "render_calls/dev_call_01.txt",
            "render_calls/loss_curve_call_01.txt", "render_calls/train_call_01.txt",
            "block_excluded.jsonl", "neighbours_test_block.tsv", "neighbours_dev_block.tsv",
            "loss_curve_excluded.jsonl"} == set(names)
    S.write_artifacts(st, out, {"pool": {"path": "made up"}})
    assert snapshot(out) == first
    # a second run of the build in the same folder renders nothing and writes the same bytes
    render2 = FakeRender({})
    st2 = run(made_up, out, render2)
    S.write_artifacts(st2, out, {"pool": {"path": "made up"}})
    assert not render2.calls and snapshot(out) == first
    for line in (out / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        assert P.sha256_file(out / name) == digest
    # the key lists: type, tier, repository, rank in the cell; the subsets are read from the rank
    train = json.load(open(out / "train.json"))
    assert train["count"] == 64 and train["n_per_cell"] == 8 and train["repository_limit"] is None and train["seed"] == 42
    assert train["subsets"] == {"train_5k": {"rank_at_most": 4, "count": 32}, "train_1k": {"rank_at_most": 2, "count": 16}}
    assert set(train["diagrams"][0]) == {"key", "primary_type", "tier", "repository", "rank", "target_tokens"}
    by_key = {d["key"]: d for d in made_up["pool"]}
    for d in train["diagrams"]:
        assert (d["primary_type"], d["tier"], d["repository"]) == tuple(by_key[d["key"]][f] for f in ("primary_type", "tier", "repository"))
        assert d["target_tokens"] == made_up["tokens"][d["key"]]
    dev = json.load(open(out / "dev.json"))
    assert dev["repository_limit"] == CAP and dev["n_per_cell"] == 2 and dev["count"] == 16
    counts = json.load(open(out / "counts.json"))
    assert [s["step"] for s in counts["steps"]][0] == "pool_after_D6" and counts["sets"]["train_1k"]["diagrams"] == 16
    config = json.load(open(out / "config.json"))
    assert config["parameters"]["seed"] == 42 and config["parameters"]["D13_train"]["repository_limit"] is None
    assert config["code"]["sample.py"] == P.sha256_file(P.HERE / "sample.py")


def test_images_are_left_out_of_the_checksum_list(made_up, plain, tmp_path):
    st, _, _ = plain
    out = tmp_path / "out"
    (out / "render/dev/puml_images_1568").mkdir(parents=True)
    (out / "render/dev/puml_images_1568/a.png").write_bytes(b"x")
    (out / "render/dev/manifest.jsonl").write_text("")
    names = S.write_artifacts(st, out)
    assert "render/dev/manifest.jsonl" in names and not [n for n in names if n.endswith(".png")]


# --------------------------------------------------------------------------- #
# The artifacts of step 2 (data/sets)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def frozen():
    lists = {name: json.load(open(SETS / f"{name}.json")) for name in S.SETS}
    pool, tokens, _ = S.load_pool()
    return dict(lists=lists, pool=pool, tokens=tokens, test=P.load_test_set(),
                dropped=P.read_jsonl(SETS / "dropped.jsonl"), d7=P.read_jsonl(SETS / "d7_removed.jsonl"),
                block=P.read_jsonl(SETS / "block_excluded.jsonl"), loss=P.read_jsonl(SETS / "loss_curve_excluded.jsonl"),
                counts=json.load(open(SETS / "counts.json")))


@needs_artifacts
def test_the_step_2_artifacts_have_the_recorded_sha256():
    lines = (SETS / "SHA256SUMS").read_text().splitlines()
    for line in lines:
        digest, name = line.split("  ", 1)
        assert P.sha256_file(SETS / name) == digest
    listed = {line.split("  ", 1)[1] for line in lines}
    on_disk = {str(p.relative_to(SETS)) for p in SETS.rglob("*")
               if p.is_file() and p.name != "SHA256SUMS" and "puml_images_1568" not in p.parts}
    assert listed == on_disk
    assert {"dev.json", "loss_curve.json", "train.json", "config.json", "counts.json", "dropped.jsonl",
            "d7_removed.jsonl", "d8_report.json", "neighbours_test.tsv", "neighbours_dev.tsv",
            "block_excluded.jsonl", "neighbours_test_block.tsv", "neighbours_dev_block.tsv",
            "loss_curve_excluded.jsonl"} <= listed
    assert all(f"render/{s}/{f}" in listed for s in S.SETS for f in ("manifest.jsonl", "render_env.json", "summary.json"))
    # the code the configuration records is the code on disk
    config = json.load(open(SETS / "config.json"))
    assert config["code"] == {name: P.sha256_file(P.HERE / name) for name in ("sample.py", "pool.py", "leakage.py", "render.py")}
    assert config["inputs"]["pool"]["sha256"] == P.sha256_file(FT / "data/pool/pool_after_d6.json")
    assert config["parameters"]["seed"] == 42 and config["parameters"]["D11_dev"] == {
        "n_per_cell": 25, "repository_limit": 5, "drawn_from": "pool after D6"}
    amendment = config["parameters"]["Amendment_1"]
    assert amendment["A3"] == {"candidates": "pool after D6", "reference": "the test diagrams",
                               "excluded_from": ["dev", "loss_curve", "train"]}
    assert amendment["A4"] == {"candidates": "training pool after D7", "reference": "the dev diagrams", "excluded_from": ["train"]}
    assert amendment["extract_puml_source_sha256"] == P.PINS["extract_puml_source"]
    assert config["parameters"]["Amendment_2"]["A6"] == {
        "candidates": "training pool after D7", "reference": "the diagrams of the loss-curve split",
        "texts": ["whole_file", "block"], "excluded_from": ["train"]}
    for name in S.SETS:                      # every render ran with the render.py on disk, at its defaults
        runs = json.load(open(SETS / "render" / name / "render_env.json"))["runs"]
        assert all((r["render_py_sha256"], r["threads"], r["heap_mb"], r["batch_size"]) ==
                   (config["code"]["render.py"], 3, 32768, 200) for r in runs)


@needs_artifacts
def test_the_frozen_key_lists_are_the_three_draws_over_the_pool_and_the_exclusion_list(frozen):
    """The lists follow from the pool file, the exclusion list, the list of D7 and the keys of
    Amendments 1 and 2 alone: those of A3 are passed over in the three draws, those of A4 and of
    A6 in the training draw."""
    pool, lists = frozen["pool"], frozen["lists"]
    a3 = {r["key"] for r in frozen["block"] if r["rule"] == "A3"}
    a4 = {r["key"] for r in frozen["block"] if r["rule"] == "A4"}
    assert {r["rule"] for r in frozen["block"]} <= {"A3", "A4"} and a3 <= {d["key"] for d in pool}
    excluded = {d["key"] for d in frozen["dropped"]} | a3
    keys = lambda name: [d["key"] for d in lists[name]["diagrams"]]
    assert keys("dev") == [k for k, _ in S.draw(S.cells_of(pool), 25, 42, CAP, excluded)[0]]
    repos, others, rest = S.reserve(pool, keys("dev"))
    assert keys("loss_curve") == [k for k, _ in S.draw(S.cells_of(others), 25, 42, CAP, excluded)[0]]
    flagged = {d["key"] for d in frozen["d7"]}
    assert flagged <= {d["key"] for d in rest}
    train_pool = [d for d in rest if d["key"] not in flagged]
    a6 = {r["key"] for r in frozen["loss"]}
    assert {r["rule"] for r in frozen["loss"]} == {"A6"}
    assert a4 | a6 <= {d["key"] for d in train_pool}           # the candidate list holds them: it is not shortened
    selected, _ = S.draw(S.cells_of(train_pool), 2500, 42, None, excluded | a4 | a6)
    assert keys("train") == [k for k, _ in selected]
    assert lists["train"]["diagrams"] == S.ranked(selected, frozen["tokens"])
    steps = {s["step"]: s for s in frozen["counts"]["steps"]}
    assert steps["pool_after_D6"]["cells"] == S.count_cells(pool)
    assert steps["pool_without_the_dev_repositories"]["cells"] == S.count_cells(rest)
    assert steps["other_pool_diagrams_of_the_dev_repositories"]["cells"] == S.count_cells(others)
    assert steps["training_pool_after_D7"]["cells"] == S.count_cells(train_pool)
    assert steps["training_pool_after_D7"]["removed"]["total"] == len(flagged)


@needs_artifacts
def test_the_frozen_sets_are_disjoint_nested_and_balanced(frozen):
    lists, test = frozen["lists"], frozen["test"]
    dev, loss, train = (lists[n]["diagrams"] for n in ("dev", "loss_curve", "train"))
    repos = lambda ds: {d["repository"] for d in ds}
    keys = lambda ds: {d["key"] for d in ds}
    test_repos, test_keys = repos(test["diagrams"]), keys(test["diagrams"])
    assert (len(dev), len(loss), len(train)) == (200, 200, 20000)
    assert len(keys(dev)) == 200 and len(keys(loss)) == 200 and len(keys(train)) == 20000
    assert not repos(dev) & test_repos and not repos(train) & test_repos and not repos(loss) & test_repos
    assert not repos(train) & repos(dev) and not repos(loss) & repos(train)
    assert repos(loss) <= repos(dev)
    assert not keys(dev) & keys(loss) and not keys(dev) & keys(train) and not keys(loss) & keys(train)
    assert not test_keys & (keys(dev) | keys(loss) | keys(train))
    assert set(S.count_cells(dev).values()) == {25} and set(S.count_cells(loss).values()) == {25}
    k5, k1 = S.prefix(train, 625), S.prefix(train, 125)
    assert keys(k1) < keys(k5) < keys(train)
    assert set(S.count_cells(train).values()) == {2500} and set(S.count_cells(k5).values()) == {625}
    assert set(S.count_cells(k1).values()) == {125}
    for c in P.CELLS:
        assert [d["rank"] for d in train if S.cell_of(d) == c] == list(range(1, 2501))
    per_repo = lambda ds: max(Counter(d["repository"] for d in ds).values())
    assert per_repo(dev) <= CAP and per_repo(loss) <= CAP
    assert CAP < per_repo(train) <= S.REPO_FLAG == 100      # no limit in the training draw; D13's reversing condition not met
    checks = frozen["counts"]["checks"]["largest_number_of_diagrams_of_one_repository"]
    assert checks == {"dev": per_repo(dev), "loss_curve": per_repo(loss), "train_20k": per_repo(train),
                      "train_5k": per_repo(k5), "train_1k": per_repo(k1)}
    # type, tier and repository are those of the pool file; every target is within the token cap
    by_key = {d["key"]: d for d in frozen["pool"]}
    for d in dev + loss + train:
        p = by_key[d["key"]]
        assert (d["primary_type"], d["tier"], d["repository"]) == (p["primary_type"], p["tier"], p["repository"])
        assert d["target_tokens"] == frozen["tokens"][d["key"]] <= P.TOKEN_CAP


@needs_artifacts
def test_every_key_of_a_frozen_set_has_an_image_and_no_degeneracy_label(frozen):
    for name in S.SETS:
        manifest = S.read_manifest(SETS / "render" / name)
        images = set(os.listdir(SETS / "render" / name / "puml_images_1568"))
        keys = [d["key"] for d in frozen["lists"][name]["diagrams"]]
        for k in keys:
            row = manifest[k]
            assert row["status"] == "ok" and S.render_outcome(row) is None and k[:-5] + ".png" in images
        dropped = {d["key"] for d in frozen["dropped"] if d["set"] == name}
        # Implementation note 15: a key that was drawn and rendered before an amendment excluded it keeps its line
        replaced = {r["key"] for r in frozen["block"] + frozen["loss"]
                    if (r["drawn_without_the_amendment"] or {}).get("set") == name}
        assert set(manifest) == set(keys) | dropped | replaced and not replaced & set(keys)
        assert all(manifest[k]["status"] == "ok" for k in replaced)
        # Implementation note 18: the images of the replaced keys are moved out of the folder of the set,
        # which holds the images of the keys of the list and no other
        side = FT / "data/replaced_images" / name
        moved = set(os.listdir(side)) if side.is_dir() else set()
        assert moved == {k[:-5] + ".png" for k in replaced}
        assert images == {k[:-5] + ".png" for k in keys}
        for d in frozen["dropped"]:
            if d["set"] == name:
                assert S.render_outcome(manifest[d["key"]]) == d["reason"]
        called = [k for p in sorted((SETS / "render_calls").glob(f"{name}_call_*.txt")) for k in p.read_text().split()]
        assert sorted(called) == sorted(manifest)                     # every key was passed to render.py once


@needs_artifacts
def test_the_token_counts_of_the_dev_set_and_the_loss_curve_split_are_counted_again(frozen):
    keys = [d["key"] for n in ("dev", "loss_curve") for d in frozen["lists"][n]["diagrams"]]
    again = P.target_token_counts(keys, P.PUML_DIR, workers=4)
    assert again == {k: frozen["tokens"][k] for k in keys}
    assert max(again.values()) <= P.TOKEN_CAP


@needs_artifacts
def test_every_candidate_d7_removes_is_removed_again_pair_by_pair_and_none_is_in_a_set(frozen):
    names = {"a": "jaccard_5gram", "b": "edit_similarity", "c": "lines_held"}
    dev_keys = {d["key"] for d in frozen["lists"]["dev"]["diagrams"]}
    in_sets = {d["key"] for n in S.SETS for d in frozen["lists"][n]["diagrams"]}
    assert frozen["d7"]
    for row in frozen["d7"]:
        assert row["key"] not in in_sets
        cand = L.read_text(os.path.join(P.PUML_DIR, row["key"]))
        for arm in row["arms"]:
            assert row[f"{arm}_dev_key"] in dev_keys
            r = L.compare(cand, L.read_text(os.path.join(P.PUML_DIR, row[f"{arm}_dev_key"])))
            assert arm in r["arms"] and r[names[arm]] == pytest.approx(row[f"{arm}_value"], abs=1e-12)


def check_neighbour_tables(frozen, suffix, read):
    """The tables neighbours_test<suffix>.tsv and neighbours_dev<suffix>.tsv against leakage.compare
    on the texts `read(folder, key)` gives."""
    names = ("jaccard_5gram", "edit_similarity", "lines_held")
    train = [d["key"] for d in frozen["lists"]["train"]["diagrams"]]
    rng = random.Random(3)
    others = [read(P.PUML_DIR, k) for k in rng.sample(train, 100)]
    for side, folder, n in (("test", P.TEST_PUML_DIR, 1000), ("dev", P.PUML_DIR, 200)):
        rows = []
        with open(SETS / f"neighbours_{side}{suffix}.tsv") as fh:
            head = next(fh).rstrip("\n").split("\t")
            rows = [dict(zip(head, line.rstrip("\n").split("\t"))) for line in fh]
        assert len(rows) == n
        ordered = sorted(rows, key=lambda r: -float(r["edit_similarity"]))
        for row in ordered[:6] + rng.sample(rows, 6):
            ref = read(folder, row["key"])
            assert int(row["distinct_lines"]) == len(L.line_set(ref))
            for name in names:
                if row[name] == "":
                    assert name == "lines_held" and int(row["distinct_lines"]) < 10
                    continue
                if row[name + "_train_key"]:
                    assert row[name + "_train_key"] in set(train)
                    r = L.compare(read(P.PUML_DIR, row[name + "_train_key"]), ref)
                    assert (r[name] or 0.0) == pytest.approx(float(row[name]), abs=1e-6)
                else:
                    assert float(row[name]) == 0.0
            for source in others:
                r = L.compare(source, ref)
                assert not r["arms"]
                for name in names:
                    if row[name] != "":
                        assert (r[name] or 0.0) <= float(row[name]) + 1e-6


@needs_artifacts
def test_the_neighbour_tables_hold_the_pair_values_and_no_sampled_training_diagram_is_nearer(frozen):
    check_neighbour_tables(frozen, "", lambda folder, key: L.read_text(os.path.join(folder, key)))


# --------------------------------------------------------------------------- #
# The artifacts of Amendment 1 (data/sets)
# --------------------------------------------------------------------------- #
@needs_artifacts
def test_the_block_alone_neighbour_tables_hold_the_pair_values_and_no_sampled_training_diagram_is_nearer(frozen):
    check_neighbour_tables(frozen, "_block", lambda folder, key: block_of(L.read_text(os.path.join(folder, key))))
    for side in ("test", "dev"):                 # beside the tables of the whole file: same columns, same reference keys
        whole = (SETS / f"neighbours_{side}.tsv").read_text().splitlines()
        block = (SETS / f"neighbours_{side}_block.tsv").read_text().splitlines()
        assert block[0] == whole[0] and [l.split("\t")[0] for l in block] == [l.split("\t")[0] for l in whole]


@needs_artifacts
def test_the_dev_set_and_the_loss_curve_split_are_the_files_frozen_before_amendment_1(frozen):
    assert P.sha256_file(SETS / "dev.json") == "3bb1e474562e6c0a654c82e702f311fa83dbd492b500fda99d63cfac84790dcf"
    assert P.sha256_file(SETS / "loss_curve.json") == "897dc6d402f9e960ad184e785d2a07490751506cf7dfe01c6f69fa8b56f6c6d6"
    assert S.frozen_lists(SETS) == {n: [d["key"] for d in frozen["lists"][n]["diagrams"]] for n in ("dev", "loss_curve")}
    for name in ("dev", "loss_curve"):
        assert frozen["counts"]["draws"][name]["amendment_1"] == frozen["counts"]["draws"][name]["amendment_2"] == NO_EFFECT


@needs_artifacts
def test_every_key_amendment_1_excludes_is_flagged_again_on_its_block_pair_by_pair_and_none_is_in_a_set(frozen):
    lists, pool = frozen["lists"], frozen["pool"]
    in_sets = {d["key"] for n in S.SETS for d in lists[n]["diagrams"]}
    test_keys = {d["key"] for d in frozen["test"]["diagrams"]}
    dev_keys = {d["key"] for d in lists["dev"]["diagrams"]}
    _, _, rest = S.reserve(pool, dev_keys)
    train_pool = {d["key"] for d in rest} - {d["key"] for d in frozen["d7"]}
    by_key = {d["key"]: d for d in pool}
    assert frozen["block"] and len({(r["rule"], r["key"]) for r in frozen["block"]}) == len(frozen["block"])
    for row in frozen["block"]:
        assert set(row) == RECORD and row["key"] not in in_sets and row["pairs"]
        p = by_key[row["key"]]
        assert (row["primary_type"], row["tier"], row["repository"]) == (p["primary_type"], p["tier"], p["repository"])
        assert row["reference_set"] == S.RULES[row["rule"]][0]
        refs, folder = (test_keys, P.TEST_PUML_DIR) if row["rule"] == "A3" else (dev_keys, P.PUML_DIR)
        assert row["rule"] == "A3" or row["key"] in train_pool
        cand = L.read_text(os.path.join(P.PUML_DIR, row["key"]))
        arms = set()
        for pair in row["pairs"]:
            assert set(pair) == {"reference_key", "block", "whole_file"} and pair["reference_key"] in refs
            again = pair_on_both_texts(cand, L.read_text(os.path.join(folder, pair["reference_key"])))
            assert again == {"block": pair["block"], "whole_file": pair["whole_file"]}
            assert pair["block"]["arms"] and pair["whole_file"]["arms"] == []       # flagged on the block alone
            arms |= set(pair["block"]["arms"])
        assert sorted(arms) == row["arms"]
    # the rank of a key that was drawn is its place in the training draw without the keys of the amendment (and
    # without those of Amendment 2, which came after it); the list after Amendment 1 has its keys passed over
    excluded = {d["key"] for d in frozen["dropped"]}
    cells = S.cells_of([d for d in rest if d["key"] in train_pool])
    without = S.ranked(S.draw(cells, 2500, 42, None, excluded)[0])
    rank = {d["key"]: d["rank"] for d in without}
    for row in frozen["block"]:
        drawn = row["drawn_without_the_amendment"]
        assert drawn == ({"set": "train", "rank": rank[row["key"]]} if row["key"] in rank else None)
    train = S.ranked(S.draw(cells, 2500, 42, None, excluded | {r["key"] for r in frozen["block"]})[0])
    effect = frozen["counts"]["draws"]["train"]["amendment_1"]
    held = [d for d in without if d["key"] in {r["key"] for r in frozen["block"]}]
    assert [(e["key"], e["rank"]) for e in effect["excluded_keys_the_draw_holds_without_the_exclusion"]] == [
        (d["key"], d["rank"]) for d in held]
    new = [d for d in train if d["key"] not in rank]
    assert [(e["key"], e["rank"]) for e in effect["keys_that_enter"]] == [(d["key"], d["rank"]) for d in new]
    assert len(new) == len(held) and all(d["rank"] > 2500 - len(held) for d in new)
    assert effect["every_other_key_keeps_its_place"] is True
    for c in P.CELLS:                            # every other key keeps its place; the new keys close the cell
        was = [d["key"] for d in without if S.cell_of(d) == c and d["key"] not in {h["key"] for h in held}]
        now = [d["key"] for d in train if S.cell_of(d) == c]
        assert now[:len(was)] == was and set(now[len(was):]) == {d["key"] for d in new if S.cell_of(d) == c}


@needs_artifacts
def test_the_reports_of_amendment_1_agree_with_the_exclusion_records(frozen):
    report = json.load(open(SETS / "d8_report.json"))
    counts = frozen["counts"]["amendment_1"]
    by_key = {d["key"]: d for d in frozen["pool"]}
    for rule in ("A3", "A4"):
        keys = [r["key"] for r in frozen["block"] if r["rule"] == rule]
        cells = Counter(S.cell_name(S.cell_of(by_key[k])) for k in keys)
        want = dict({S.cell_name(c): cells[S.cell_name(c)] for c in P.CELLS}, total=len(keys))
        assert report["block_alone"][rule]["excluded"] == counts[rule]["excluded"] == want
        assert report["block_alone"][rule]["candidates_compared"] == counts[rule]["candidates_compared"]
    assert counts["A3"]["candidates_compared"] == len(frozen["pool"]) and counts["A3"]["reference_diagrams"] == 1000
    steps = {s["step"]: s for s in frozen["counts"]["steps"]}
    assert counts["A4"]["candidates_compared"] == steps["training_pool_after_D7"]["total"] and counts["A4"]["reference_diagrams"] == 200
    assert counts["excluded_keys"] == len({r["key"] for r in frozen["block"]})
    assert frozen["counts"]["checks"]["block_excluded_keys_in_a_set"] == 0
    # no arm of D6 on a key of the three sets, on the whole file or on the block alone
    assert report["set_keys_on_which_an_arm_fires"] == {text: {pair: ZERO for pair in PAIRS}
                                                        for text in ("whole_file", "block_alone")}
    for part in (report["training_sample"], report["block_alone"]["training_sample"]):
        assert part["diagrams"] == 20000 and part["training_diagrams_on_which_an_arm_fires"] == NO_ARM
        for side, n in (("test", 1000), ("dev", 200)):
            summary = part[f"against_the_{side}_set"]
            assert summary["reference_diagrams"] == n
            assert summary["jaccard_5gram"]["largest"]["value"] < 0.7 and summary["edit_similarity"]["largest"]["value"] < 0.8
            assert summary["lines_held"]["largest"]["value"] < 0.8
            assert all(summary[name]["reference_diagrams_at_or_above"]["0.8"] == 0 for name in VALUES)
    # the largest values of the report are those of the tables
    for suffix, part in (("", report["training_sample"]), ("_block", report["block_alone"]["training_sample"])):
        for side in ("test", "dev"):
            with open(SETS / f"neighbours_{side}{suffix}.tsv") as fh:
                head = next(fh).rstrip("\n").split("\t")
                rows = [dict(zip(head, line.rstrip("\n").split("\t"))) for line in fh]
            for name in VALUES:
                valued = [float(r[name]) for r in rows if r[name] != ""]
                summary = part[f"against_the_{side}_set"][name]
                assert summary["reference_diagrams_with_a_value"] == len(valued)
                assert max(valued) == pytest.approx(summary["largest"]["value"], abs=1e-6)
                for level in ("0.5", "0.6", "0.7"):
                    near = sum(1 for v in valued if abs(v - float(level)) < 1e-6)       # six decimals in the table
                    assert abs(sum(1 for v in valued if v >= float(level)) - summary["reference_diagrams_at_or_above"][level]) <= near


# --------------------------------------------------------------------------- #
# The artifacts of Amendment 2 (data/sets)
# --------------------------------------------------------------------------- #
@needs_artifacts
def test_every_key_amendment_2_excludes_is_flagged_again_pair_by_pair_and_the_training_list_changes_by_those_keys_only(frozen):
    lists, pool, rows = frozen["lists"], frozen["pool"], frozen["loss"]
    in_sets = {d["key"] for n in S.SETS for d in lists[n]["diagrams"]}
    loss_keys = {d["key"] for d in lists["loss_curve"]["diagrams"]}
    _, _, rest = S.reserve(pool, [d["key"] for d in lists["dev"]["diagrams"]])
    train_pool = [d for d in rest if d["key"] not in {r["key"] for r in frozen["d7"]}]
    by_key = {d["key"]: d for d in pool}
    a6 = {r["key"] for r in rows}
    assert rows and len(a6) == len(rows)
    assert [r["key"] for r in rows] == [d["key"] for d in train_pool if d["key"] in a6]       # candidates of the training pool, in its order
    for row in rows:
        assert set(row) == RECORD | {"flagged_on"} and (row["rule"], row["reference_set"]) == ("A6", "loss_curve")
        assert row["key"] not in in_sets and row["pairs"]
        p = by_key[row["key"]]
        assert (row["primary_type"], row["tier"], row["repository"]) == (p["primary_type"], p["tier"], p["repository"])
        cand = L.read_text(os.path.join(P.PUML_DIR, row["key"]))
        arms = {text: set() for text in ("whole_file", "block")}
        for pair in row["pairs"]:
            assert set(pair) == {"reference_key", "block", "whole_file"} and pair["reference_key"] in loss_keys
            again = pair_on_both_texts(cand, L.read_text(os.path.join(P.PUML_DIR, pair["reference_key"])))
            assert again == {"block": pair["block"], "whole_file": pair["whole_file"]}
            assert pair["block"]["arms"] or pair["whole_file"]["arms"]
            for text in arms:
                arms[text] |= set(pair[text]["arms"])
        assert row["flagged_on"] == [text for text in ("whole_file", "block") if arms[text]]
        assert row["arms"] == sorted(arms["whole_file"] | arms["block"])
    # A7: the list before the amendment is the training draw with the keys of D14 and of Amendment 1 passed over
    excluded = {d["key"] for d in frozen["dropped"]} | {r["key"] for r in frozen["block"]}
    cells = S.cells_of(train_pool)
    before = S.ranked(S.draw(cells, 2500, 42, None, excluded)[0])
    rank = {d["key"]: d["rank"] for d in before}
    for row in rows:
        assert row["drawn_without_the_amendment"] == ({"set": "train", "rank": rank[row["key"]]} if row["key"] in rank else None)
    train = lists["train"]["diagrams"]
    held = [d for d in before if d["key"] in a6]
    new = [d for d in train if d["key"] not in rank]
    effect = frozen["counts"]["draws"]["train"]["amendment_2"]
    assert [(e["key"], e["rank"]) for e in effect["excluded_keys_the_draw_holds_without_the_exclusion"]] == [
        (d["key"], d["rank"]) for d in held]
    assert [(e["key"], e["rank"]) for e in effect["keys_that_enter"]] == [(d["key"], d["rank"]) for d in new]
    assert effect["every_other_key_keeps_its_place"] is True and len(new) == len(held) > 0
    assert {d["key"] for d in before} - {d["key"] for d in train} == {d["key"] for d in held}
    # in every cell: the other keys keep their place, and the next keys of the cell's order that are on no list close it
    orders = S.cell_orders(cells, 42)
    for c in P.CELLS:
        was = [d["key"] for d in before if S.cell_of(d) == c]
        kept = [k for k in was if k not in a6]
        now = [d["key"] for d in train if S.cell_of(d) == c]
        assert now[:len(kept)] == kept and len(now) == len(was) == 2500
        after_last = orders[c][orders[c].index(was[-1]) + 1:]
        assert now[len(kept):] == [k for k in after_last if k not in excluded | a6][:len(was) - len(kept)]
    for n in (625, 125):                         # a subset loses the excluded keys it held and no other key
        gone = {d["key"] for d in before if d["rank"] <= n} - {d["key"] for d in train if d["rank"] <= n}
        assert gone == {d["key"] for d in held if d["rank"] <= n}
    # the records of Amendment 1 are the file of before the amendment
    assert P.sha256_file(SETS / "block_excluded.jsonl") == "5d5b36be66984f1486fafb9fca435a3a18673dc6a3d41238a55c8849303e3480"


@needs_artifacts
def test_the_reports_of_amendment_2_agree_with_the_exclusion_records(frozen):
    report, rows = json.load(open(SETS / "d8_report.json")), frozen["loss"]
    counts = frozen["counts"]["amendment_2"]
    by_key = {d["key"]: d for d in frozen["pool"]}
    cells = Counter(S.cell_name(S.cell_of(by_key[r["key"]])) for r in rows)
    want = dict({S.cell_name(c): cells[S.cell_name(c)] for c in P.CELLS}, total=len(rows))
    steps = {s["step"]: s for s in frozen["counts"]["steps"]}
    assert counts == {"A6": {"candidates_compared": steps["training_pool_after_D7"]["total"], "reference_diagrams": 200, "excluded": want},
                      "excluded_keys": len(rows),
                      "excluded_keys_a_draw_holds_without_the_exclusion": {
                          "dev": 0, "loss_curve": 0, "train": sum(1 for r in rows if r["drawn_without_the_amendment"])}}
    assert frozen["counts"]["checks"]["loss_curve_excluded_keys_in_a_set"] == 0
    assert set(report["A6"]) == {"whole_file", "block_alone"}
    for text, name in (("whole_file", "whole_file"), ("block", "block_alone")):
        r = report["A6"][name]
        assert r["excluded"] == want and r["candidates_compared"] == counts["A6"]["candidates_compared"] and r["reference_diagrams"] == 200
        on = [x for x in rows if text in x["flagged_on"]]
        for arm in L.ARMS:
            assert r["flagged_by_arm"][arm]["total"] == sum(1 for x in on if any(arm in pair[text]["arms"] for pair in x["pairs"]))
        assert r["excluded_by_combination_of_arms"].get("none", {"total": 0})["total"] == len(rows) - len(on)
        assert r["reference_diagrams_with_an_excluded_candidate"]["any"] == len(
            {pair["reference_key"] for x in rows for pair in x["pairs"] if pair[text]["arms"]})
    # A8: the largest values inside the final 20,000 against the loss-curve split, on both texts
    train = [d["key"] for d in frozen["lists"]["train"]["diagrams"]]
    loss = [d["key"] for d in frozen["lists"]["loss_curve"]["diagrams"]]
    rng = random.Random(5)
    whole = lambda key: L.read_text(os.path.join(P.PUML_DIR, key))
    for part, read in ((report["training_sample"], whole), (report["block_alone"]["training_sample"], lambda key: block_of(whole(key)))):
        assert part["diagrams"] == 20000 and part["training_diagrams_on_which_an_arm_fires"] == NO_ARM
        summary = part["against_the_loss_curve_split"]
        assert summary["reference_diagrams"] == 200
        assert summary["jaccard_5gram"]["largest"]["value"] < 0.7 and summary["edit_similarity"]["largest"]["value"] < 0.8
        assert summary["lines_held"]["largest"]["value"] < 0.8
        assert all(summary[name]["reference_diagrams_at_or_above"]["0.8"] == 0 for name in VALUES)
        for name in VALUES:                      # the pair that gives the largest value, computed again from the corpus
            top = summary[name]["largest"]
            assert top["train_key"] in set(train) and top["reference"] in set(loss)
            r = L.compare(read(top["train_key"]), read(top["reference"]))
            assert r[name] == pytest.approx(top["value"], abs=1e-12) and not r["arms"]
        refs = [read(k) for k in rng.sample(loss, 12)]
        for source in (read(k) for k in rng.sample(train, 100)):       # no sampled pair is above it or fires an arm
            for ref in refs:
                r = L.compare(source, ref)
                assert not r["arms"] and all((r[name] or 0.0) <= summary[name]["largest"]["value"] + 1e-12 for name in VALUES)
