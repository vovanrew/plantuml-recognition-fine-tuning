"""Tests for build/rater_check.py, on synthetic inputs (no scored run is read).

Run from fine-tune/:  python3 -m pytest build/tests/test_rater_check.py -q
"""
import argparse
import collections
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

FT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(FT / "build"))
import rater_check as RC  # noqa: E402


def sc(ef, rf, compiled=(True, True), v1=None):
    """Scores of one diagram at one size: ef and rf are (zero-shot, fine-tuned)."""
    v1 = v1 or (ef, rf)
    return {"compiled": compiled, "v2": {"ef": ef, "rf": rf}, "v1": {"ef": v1[0], "rf": v1[1]}}


# --------------------------------------------------------------------------- pool and direction
def test_direction_element_first_then_relationship():
    assert RC.direction(sc((0.5, 1.0), (1.0, 0.0))) == "fine_tuned"     # EF decides even against RF
    assert RC.direction(sc((1.0, 0.5), (0.0, 1.0))) == "zero_shot"
    assert RC.direction(sc((1.0, 1.0), (0.2, 0.8))) == "fine_tuned"     # EF tie: RF decides
    assert RC.direction(sc((1.0, 1.0), (0.9, 0.3))) == "zero_shot"
    assert RC.direction(sc((1.0, 1.0), (0.5, 0.5))) is None


def test_pool_needs_both_compiled_and_a_difference():
    table = {
        "both_differ": sc((0.5, 1.0), (1.0, 1.0)),
        "one_failed": sc((0.0, 1.0), (1.0, 1.0), compiled=(False, True)),   # empty-edge RF = 1 kind
        "both_failed": sc((0.0, 0.0), (1.0, 0.0), compiled=(False, False)),
        "tie": sc((1.0, 1.0), (1.0, 1.0)),
    }
    assert set(RC.pool(table)) == {"both_differ"}


# --------------------------------------------------------------------------- the draw
def test_split_types_even_as_allowed():
    assert RC.split_types(17, {"class": 70, "sequence": 68}) == {"class": 9, "sequence": 8}
    assert RC.split_types(3, {"class": 1, "sequence": 2}) == {"class": 1, "sequence": 2}
    assert RC.split_types(8, {"class": 10, "sequence": 2}) == {"class": 6, "sequence": 2}
    assert RC.split_types(4, {"sequence": 9}) == {"class": 0, "sequence": 4}
    with pytest.raises(ValueError):
        RC.split_types(5, {"class": 2, "sequence": 2})


def synthetic_pools(n_ft=40, n_zs=30):
    pools, meta, i = {}, {}, 0
    for size in RC.SIZES:
        pools[size] = {}
        for d, n in (("fine_tuned", n_ft), ("zero_shot", n_zs)):
            for j in range(n):
                k = f"{i:040x}"
                i += 1
                meta[k] = {"primary_type": RC.TYPES[j % 2], "tier": 1 + j % 4}
                pools[size][k] = sc((0.4, 0.9) if d == "fine_tuned" else (0.9, 0.4), (1.0, 1.0))
    return pools, meta


def test_draw_quotas_ids_and_sides():
    pools, meta = synthetic_pools()
    pairs = RC.draw(pools, meta)
    assert len(pairs) == 60
    assert [p["pair"] for p in pairs] == [f"P{n:02d}" for n in range(1, 61)]
    cells = collections.Counter((p["size"], p["direction"]) for p in pairs)
    for size in RC.SIZES:
        assert cells[(size, "zero_shot")] == RC.QUOTA[size]
        assert cells[(size, "fine_tuned")] == RC.PER_SIZE - RC.QUOTA[size]
    assert sum(RC.QUOTA.values()) == 20
    for p in pairs:
        assert RC.direction(pools[p["size"]][p["key"]]) == p["direction"]
        assert {p["A"], p["B"]} == {"zero_shot", "fine_tuned"}
        assert p["primary_type"] == meta[p["key"]]["primary_type"]
    assert len({(p["size"], p["key"]) for p in pairs}) == 60
    sides = collections.Counter(p["A"] for p in pairs)
    assert 15 <= sides["zero_shot"] <= 45                                 # both sides used


def test_draw_is_deterministic_and_seeded():
    pools, meta = synthetic_pools()
    assert RC.draw(pools, meta) == RC.draw(pools, meta)
    assert RC.draw(pools, meta) != RC.draw(pools, meta, seed=1)


def test_rater_orders_are_permutations_and_differ():
    pools, meta = synthetic_pools()
    pairs = RC.draw(pools, meta)
    o1, o2 = RC.rater_order(pairs, "R1"), RC.rater_order(pairs, "R2")
    assert sorted(o1) == sorted(o2) == [p["pair"] for p in pairs]
    assert o1 != o2


# --------------------------------------------------------------------------- the sheet is blind
def test_sheet_names_no_run_condition_or_key(tmp_path):
    pools, meta = synthetic_pools()
    pairs = RC.draw(pools, meta)[:4]
    images, render = tmp_path / "images", tmp_path / "render"
    images.mkdir(); render.mkdir()
    for p in pairs:
        Image.new("RGB", (8, 8), "white").save(images / f"{p['key']}.png")
        for cond in RC.CONDS:
            colour = "red" if cond == "zero_shot" else "blue"
            Image.new("RGB", (8, 8), colour).save(render / (RC.source_name(p["size"], cond, p["key"])[:-5] + ".png"))
    sheet = tmp_path / "R1"
    RC.build_sheet(pairs, "R1", sheet, render, images=images)
    files = sorted(f.name for f in (sheet / "img").iterdir())
    assert files == sorted(f"{p['pair']}_{s}.png" for p in pairs for s in "oab")
    html = (sheet / "index.html").read_text()
    banned = ["zero", "shot", "fine", "tuned", "20000", "2B", "9B", "27B"] + [p["key"] for p in pairs]
    for word in banned:
        assert word not in html and not any(word in f for f in files), word
    assert RC.QUESTION in html and RC.GUIDANCE in html
    # the A picture is the render of the condition the key file names as A
    for p in pairs:
        a = Image.open(sheet / "img" / f"{p['pair']}_a.png").getpixel((0, 0))
        assert a == ((255, 0, 0) if p["A"] == "zero_shot" else (0, 0, 255))
    ids = re.search(r"const IDS = (\[.*?\]);", html).group(1)
    assert ids.count('"P') == 4


# --------------------------------------------------------------------------- ratings and numbers
def two_pairs():
    return [
        {"pair": "P01", "size": "2B", "key": "k1", "primary_type": "class", "direction": "fine_tuned",
         "A": "zero_shot", "B": "fine_tuned",
         "scores": {"v2": {"ef": {"zero_shot": 0.5, "fine_tuned": 1.0}, "rf": {"zero_shot": 1.0, "fine_tuned": 1.0}},
                    "v1": {"ef": {"zero_shot": 0.5, "fine_tuned": 1.0}, "rf": {"zero_shot": 0.0, "fine_tuned": 1.0}}}},
        {"pair": "P02", "size": "27B", "key": "k2", "primary_type": "sequence", "direction": "zero_shot",
         "A": "zero_shot", "B": "fine_tuned",
         "scores": {"v2": {"ef": {"zero_shot": 0.9, "fine_tuned": 0.6}, "rf": {"zero_shot": 0.2, "fine_tuned": 0.8}},
                    "v1": {"ef": {"zero_shot": 0.9, "fine_tuned": 0.6}, "rf": {"zero_shot": 0.2, "fine_tuned": 0.8}}}},
    ]


def write_csv(path, rows):
    path.write_text("pair,choice,comment\n" + "".join(f'{a},{b},"{c}"\n' for a, b, c in rows))
    return path


def test_read_ratings_validates(tmp_path):
    pairs = two_pairs()
    ok = RC.read_ratings(write_csv(tmp_path / "ok.csv", [("P01", "B", "x, y"), ("P02", "same", "")]), pairs)
    assert ok["P01"] == {"choice": "B", "comment": "x, y"} and ok["P02"]["choice"] == "same"
    for bad in ([("P01", "B", "")],                                    # missing pair
                [("P01", "B", ""), ("P02", "C", "")],                 # bad choice
                [("P01", "B", ""), ("P01", "A", ""), ("P02", "A", "")],   # twice
                [("P01", "B", ""), ("P02", "A", ""), ("P09", "A", "")]):  # unknown
        with pytest.raises(ValueError):
            RC.read_ratings(write_csv(tmp_path / "bad.csv", bad), pairs)


def test_agreement_counts_and_wilson():
    pairs = two_pairs()
    # R1 picks B (fine-tuned) on P01, A (zero-shot) on P02
    r = {"P01": {"choice": "B", "comment": ""}, "P02": {"choice": "A", "comment": ""}}
    ef = RC.agreement(pairs, r, RC.by_metric("v2", "ef"))
    assert (ef["pairs"], ef["agree"], ef["against"], ef["same"]) == (2, 2, 0, 0)
    rf = RC.agreement(pairs, r, RC.by_metric("v2", "rf"))                 # P01 ties on v2 RF: out
    assert (rf["pairs"], rf["agree"], rf["against"]) == (1, 0, 1)
    rf1 = RC.agreement(pairs, r, RC.by_metric("v1", "rf"))                # v1 RF differs on P01
    assert (rf1["pairs"], rf1["agree"], rf1["against"]) == (2, 1, 1)
    lo, hi = RC.wilson(8, 10)
    assert round(lo, 4) == 0.4902 and round(hi, 4) == 0.9433                # textbook value
    assert RC.wilson(0, 0) is None
    same = RC.agreement(pairs, {"P01": {"choice": "same"}, "P02": {"choice": "same"}}, RC.by_metric("v2", "ef"))
    assert same["same"] == 2 and same["picks"] == 0 and same["share"] is None


def test_report_kappa_v1v2_and_against_list():
    pairs = two_pairs()
    r1 = {"P01": {"choice": "B", "comment": ""}, "P02": {"choice": "B", "comment": "arrow"}}
    r2 = {"P01": {"choice": "B", "comment": ""}, "P02": {"choice": "B", "comment": "labels"}}
    rep = RC.report(pairs, {"R1": r1, "R2": r2})
    assert rep["between_raters"]["agree"] == 2
    assert rep["v1_v2_prefer_differently"] == {"ef": [], "rf": ["P01"]}
    against = rep["both_against_direction"]
    assert [p["pair"] for p in against] == ["P02"]                        # both picked fine-tuned, metric zero-shot
    assert against[0]["comments"] == {"R1": "arrow", "R2": "labels"}
    assert (rep["by_direction"]["R1"]["zero_shot"]["agree"], rep["by_direction"]["R1"]["zero_shot"]["against"]) == (0, 1)
    md = RC.report_md(rep)
    assert "| P02 | 27B | sequence | zero-shot |  |  |" in md                # empty until the author writes them


def test_differences_fill_the_table_and_must_match_the_list(tmp_path):
    pairs = two_pairs()
    r = {"P01": {"choice": "B", "comment": ""}, "P02": {"choice": "B", "comment": ""}}
    rep = RC.report(pairs, {"R1": r, "R2": r})
    f = tmp_path / "differences.csv"
    f.write_text('pair,differs,kind\nP02,"Fine-tuned misses one arrow, zero-shot two",Scores split\n', encoding="utf-8")
    RC.add_differences(rep, RC.read_differences(f, rep["both_against_direction"]))
    md = RC.report_md(rep)
    assert "| P02 | 27B | sequence | zero-shot | Fine-tuned misses one arrow, zero-shot two | Scores split |" in md
    f.write_text("pair,differs,kind\nP02,x,k\nP01,y,k\n", encoding="utf-8")           # a pair not in the list
    with pytest.raises(SystemExit):
        RC.read_differences(f, rep["both_against_direction"])
    f.write_text("pair,differs,kind\nP02, ,k\n", encoding="utf-8")                   # no description
    with pytest.raises(SystemExit):
        RC.read_differences(f, rep["both_against_direction"])


# --------------------------------------------------------------------------- added after verify 1
def six_pairs():
    """Two directions x sizes x types, with A sides mixed, for the report's groupings."""
    out = []
    spec = [("P01", "2B", "class", "fine_tuned", "zero_shot"), ("P02", "2B", "sequence", "fine_tuned", "fine_tuned"),
            ("P03", "9B", "class", "zero_shot", "zero_shot"), ("P04", "9B", "sequence", "fine_tuned", "fine_tuned"),
            ("P05", "27B", "class", "zero_shot", "fine_tuned"), ("P06", "27B", "sequence", "zero_shot", "zero_shot")]
    for pid, size, typ, d, a in spec:
        hi, lo = 1.0, 0.5
        ef = {"zero_shot": hi, "fine_tuned": lo} if d == "zero_shot" else {"zero_shot": lo, "fine_tuned": hi}
        rf = {"zero_shot": 1.0, "fine_tuned": 1.0} if pid != "P06" else {"zero_shot": 0.0, "fine_tuned": 1.0}
        out.append({"pair": pid, "size": size, "key": pid.lower(), "primary_type": typ, "direction": d,
                    "A": a, "B": "fine_tuned" if a == "zero_shot" else "zero_shot",
                    "scores": {"v2": {"ef": ef, "rf": rf}, "v1": {"ef": ef, "rf": rf}}})
    return out


def test_direction_split_uses_the_pair_direction_on_each_group():
    pairs = six_pairs()
    # pick the direction's answer everywhere except P05 (picks the other) and P06 ("same")
    r = {}
    for p in pairs:
        side = "A" if p["A"] == p["direction"] else "B"
        r[p["pair"]] = {"choice": side, "comment": ""}
    r["P05"] = {"choice": "B" if r["P05"]["choice"] == "A" else "A", "comment": ""}
    r["P06"] = {"choice": "same", "comment": ""}
    rep = RC.report(pairs, {"R1": r})
    ft, zs = rep["by_direction"]["R1"]["fine_tuned"], rep["by_direction"]["R1"]["zero_shot"]
    assert (ft["pairs"], ft["agree"], ft["against"], ft["same"]) == (3, 3, 0, 0)
    assert (zs["pairs"], zs["agree"], zs["against"], zs["same"]) == (3, 1, 1, 1)
    # counts by size and type, v2 Element F1, without intervals in the markdown
    assert (rep["by_size"]["R1"]["27B"]["ef"]["agree"], rep["by_size"]["R1"]["27B"]["ef"]["against"],
            rep["by_size"]["R1"]["27B"]["ef"]["same"]) == (0, 1, 1)
    assert (rep["by_type"]["R1"]["class"]["ef"]["agree"], rep["by_type"]["R1"]["class"]["ef"]["against"]) == (2, 1)
    md = RC.report_md(rep)
    assert "| R1 | 27B | 0 / 1 / 1 | 0 / 0 / 1 |" in md
    counts = md.split("## Counts by size and by type")[1].split("## v1 and v2")[0]
    assert "[" not in counts                                                # no intervals there


def test_kappa_is_on_the_raters_own_labels():
    pairs = six_pairs()
    choices1 = ["A", "A", "B", "B", "A", "same"]
    choices2 = ["A", "A", "A", "A", "A", "A"]
    r1 = {p["pair"]: {"choice": c, "comment": ""} for p, c in zip(pairs, choices1)}
    r2 = {p["pair"]: {"choice": c, "comment": ""} for p, c in zip(pairs, choices2)}
    k = RC.report(pairs, {"R1": r1, "R2": r2})["between_raters"]
    # by hand on A / B / same: po = 3/6; R2 says A every time, so pe = (3/6)(6/6) = 1/2 and kappa = 0
    assert k["agree"] == 3 and abs(k["kappa"]) < 1e-12
    # relabelled as zero-shot / fine-tuned / same the same choices give kappa 1/7, so the label set is pinned
    import error_analysis as E
    cond = E.kappa([RC.picked(p, c) for p, c in zip(pairs, choices1)],
                   [RC.picked(p, c) for p, c in zip(pairs, choices2)])
    assert abs(cond["kappa"] - 1 / 7) < 1e-12


def test_write_sources_is_extract_puml_of_the_answer(tmp_path):
    pools, meta = synthetic_pools()
    pairs = RC.draw(pools, meta)[:2]
    texts = {}

    def answer(run, key):
        texts[(run, key)] = f"Here it is:\n```\n@startuml\nclass {run[:2]}{key[:4]}\n@enduml\n```\nDone."
        return texts[(run, key)]
    names = RC.write_sources(pairs, tmp_path / "src", answer=answer)
    assert len(names) == 4 and len(set(names)) == 4
    for p in pairs:
        for cond in RC.CONDS:
            run = RC.run_of(p["size"], cond)
            got = (tmp_path / "src" / RC.source_name(p["size"], cond, p["key"])).read_text()
            assert got == RC.extract_puml(texts[(run, p["key"])]) + "\n"
            assert got.startswith("@startuml") and got.rstrip().endswith("@enduml")


def test_render_check_counts_and_stops_on_anything_but_one_picture():
    names = ["a.puml", "b.puml", "c.puml"]
    ok = {n: {"status": "ok"} for n in names}
    assert RC.render_check(names, ok) == {"answers": 3, "ok": 3, "multi_page": 0, "other": {}}
    for bad in ({"status": "multi_page", "pages": 2}, {"status": "no_image"}, {"status": "crash"}):
        with pytest.raises(SystemExit):
            RC.render_check(names, dict(ok, b=None) | {"b.puml": bad})
    with pytest.raises(SystemExit):
        RC.render_check(names, {"a.puml": {"status": "ok"}})                # missing rows


def test_draw_refuses_a_second_draw(tmp_path):
    (tmp_path / "keys.json").write_text("{}")
    with pytest.raises(SystemExit):
        RC.cmd_draw(argparse.Namespace(out=str(tmp_path)))
    assert (tmp_path / "keys.json").read_text() == "{}"


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_page_save_format_reads_back(tmp_path):
    """The page's own toCSV, run in node, writes a file read_ratings accepts."""
    pairs = two_pairs()
    js = re.search(r"function toCSV\(ids, ans\) \{.*?\n\}", RC.PAGE, re.S).group(0)
    ans = {"P02": {"choice": "same", "comment": 'has "quotes", and a comma'}, "P01": {"choice": "A"}}
    prog = js + "\nprocess.stdout.write(toCSV(" + json.dumps(["P02", "P01"]) + ", " + json.dumps(ans) + "));"
    out = subprocess.run(["node", "-e", prog], capture_output=True, text=True, check=True).stdout
    assert out.splitlines()[0] == "pair,choice,comment"
    path = tmp_path / "R1.csv"
    path.write_text(out)
    got = RC.read_ratings(path, pairs)
    assert got == {"P01": {"choice": "A", "comment": ""},
                   "P02": {"choice": "same", "comment": 'has "quotes", and a comma'}}
    # an unanswered pair leaves an empty choice, which the reader refuses
    out2 = subprocess.run(["node", "-e", js + "\nprocess.stdout.write(toCSV(['P01','P02'], {P01: {choice: 'B'}}));"],
                          capture_output=True, text=True, check=True).stdout
    path.write_text(out2)
    with pytest.raises(ValueError):
        RC.read_ratings(path, pairs)


def test_page_is_ukrainian_and_shortcuts_use_physical_keys():
    html = RC.sheet_html("R1", ["P01"])
    assert '<html lang="uk">' in html and "Оригінал" in html and "Зберегти відповіді" in html
    for code, choice in (("KeyA", "A"), ("KeyB", "B"), ("KeyS", "same")):
        assert f'e.code === "{code}") pick("{choice}")' in html          # ф / и / і in the Ukrainian layout
    assert "e.key.toLowerCase" not in html
    assert "e.metaKey || e.ctrlKey || e.altKey) return;" in html             # Cmd+S does not answer
    for english in ("Original", "Save answers", "is closer", "About the same", "Previous"):
        assert english not in html
