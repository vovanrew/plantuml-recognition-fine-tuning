#!/usr/bin/env python3
"""Human-rater check of the metrics: paired before/after judgements on 60 test diagrams.

decisions/human-rater-validation/record.md, Decision and design.md (approved 2026-10-30).

For one test diagram at one model size, a rater sees the test image and the renders of that size's
zero-shot and fine-tuned answers, unlabelled and in random order as A and B, and answers: A, B, or
about the same. The report says how often each rater picks the answer the metric scores higher,
and how far the two raters agree.

Pool: per size, the diagrams whose zero-shot and fine-tuned answers both compiled (stored v2 CSR)
and whose v2 Element F1 or v2 Relationship F1 differ. A pair's direction is the condition the
metric prefers: by Element F1 when it differs, otherwise by Relationship F1.

Sample: 60 pairs, 20 per size; per size, QUOTA pairs where the metric prefers the zero-shot answer
and the rest where it prefers the fine-tuned one. Within a size x direction cell, class and
sequence are split as evenly as the cell allows (class takes the odd one). Each cell is drawn with
random.Random(f"{SEED}|rater-check|<size>|<direction>|<type>").sample over the sorted keys
(SEED 20260614, the study's seed). Pair ids P01..P60 come from one seeded shuffle; which answer is
A comes from a second seeded stream; each rater sees the pairs in a seeded order of their own.

Pictures: the answer's PlantUML block (the scorer's extract_puml) rendered by build/render.py, the
pipeline that made the test images; the test image itself as the original. A sample answer that
does not render to exactly one picture stops the build.

Numbers (all descriptive, scorer v2 with v1 beside, outside the claim family; no pass mark):
per rater and metric, the share of A/B picks that chose the answer with the higher score, on the
pairs where that metric differs, with a Wilson 95% interval, and the count of "same"; the same
agreement with the pair's direction on the pairs of each direction; Cohen's kappa and raw
agreement between the raters on A / B / same; the sample pairs where v1 and v2 prefer different answers; the pairs where both raters chose
against the pair's direction, for the author to describe; counts by size and by type.

Usage (from fine-tune/):
  python3 build/rater_check.py draw     # rater_check/keys.json (once; before anything is rendered)
  python3 build/rater_check.py render   # the 120 answer pictures, rater_check/render/
  python3 build/rater_check.py sheets   # one rating page per rater: runs/rater_sheets/R1, R2 and their .zip
  python3 build/rater_check.py report   # reads rater_check/ratings/R1.csv, R2.csv; writes report.{json,md}
The author's description of each pair both raters chose against the metric (design §5.5) is kept in
rater_check/differences.csv (pair, differs, kind); the report fills its table from that file and
refuses one whose pairs are not exactly the report's list.
Output folder: runs/test/_analysis/rater_check/ (keys, renders, ratings, report; packed into runs.zip
with the other analyses). The rating pages are copies of its pictures for the raters, so they are
written outside runs/test/ and not released.
Tests: build/tests/test_rater_check.py
"""
import argparse
import collections
import csv
import hashlib
import json
import math
import random
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FT = HERE.parent
sys.path.insert(0, str(HERE))
import error_analysis as E  # noqa: E402  (puts the benchmark and its evaluation code on sys.path)

from analysis.bootstrap import SEED  # noqa: E402
from csr_runner import extract_puml  # noqa: E402

RUNS = E.RUNS
OUT = E.OUT / "rater_check"
SHEETS = FT / "runs" / "rater_sheets"
IMAGES = E.IMAGES
SIZES = ("2B", "9B", "27B")
PAIRS = E.PAIRS                                   # size -> (zero-shot run, fine-tuned run)
QUOTA = {"2B": 3, "9B": 8, "27B": 9}              # pairs per size where the metric prefers zero-shot
PER_SIZE = 20
TYPES = ("class", "sequence")
CONDS = ("zero_shot", "fine_tuned")
METRICS = {"ef": "element_f1", "rf": "relationship_f1"}
VERSIONS = ("v2", "v1")
RATERS = ("R1", "R2")                             # R1 the author, R2 the second rater
CHOICES = ("A", "B", "same")
DIFFERENCES = "differences.csv"                  # the author's lines for design §5.5
Z95 = 1.959963984540054

# The raters are Ukrainian, so the page is in Ukrainian (author, 2026-10-30). The approved English
# wording, of which these are the translation (design.md §1):
#   "Which drawing shows the original diagram more faithfully: A, B, or about the same?"
#   "Judge the content: which boxes there are, their names and members, the messages and their text,
#    and what is connected to what, with which kind of arrow. Ignore layout, position and colour."
QUESTION = "Який малюнок точніше відтворює оригінальну діаграму: A, B чи приблизно однаково?"
GUIDANCE = ("Оцінюйте зміст: які є блоки, їхні назви та члени (атрибути й методи), повідомлення та "
            "їхній текст, а також що з чим з'єднано і яким видом стрілки. Не зважайте на "
            "розташування, положення та колір.")


# --------------------------------------------------------------------------- #
# stored scores and the pool
# --------------------------------------------------------------------------- #
def stored(run, version, metric, runs=RUNS):
    p = runs / run / "scores" / version / metric / run / f"{metric}_results.json"
    return {d["key"]: d for d in json.load(open(p, encoding="utf-8"))["diagrams"]}


def size_scores(size, runs=RUNS):
    """{key: {"compiled": (zs, ft), "v2": {"ef": (zs, ft), "rf": ...}, "v1": {...}}} for one size."""
    zs, ft = PAIRS[size]
    csr = {r: stored(r, "v2", "csr", runs) for r in (zs, ft)}
    sc = {(v, m, r): stored(r, v, METRICS[m], runs) for v in VERSIONS for m in METRICS for r in (zs, ft)}
    out = {}
    for k in csr[zs]:
        out[k] = {"compiled": (csr[zs][k]["compiled"], csr[ft][k]["compiled"])}
        for v in VERSIONS:
            out[k][v] = {m: (sc[(v, m, zs)][k]["f1"], sc[(v, m, ft)][k]["f1"]) for m in METRICS}
    return out


def direction(scores, version="v2"):
    """The condition the metric prefers, Element F1 first; None when both metrics tie."""
    for m in ("ef", "rf"):
        z, f = scores[version][m]
        if z != f:
            return "zero_shot" if z > f else "fine_tuned"
    return None


def in_pool(scores):
    return all(scores["compiled"]) and direction(scores) is not None


def pool(table):
    return {k: s for k, s in table.items() if in_pool(s)}


# --------------------------------------------------------------------------- #
# the draw
# --------------------------------------------------------------------------- #
def split_types(n, avail):
    """How many class and sequence pairs a cell of n takes: as even as availability allows."""
    c = min(avail.get("class", 0), (n + 1) // 2)
    s = min(avail.get("sequence", 0), n - c)
    c = min(avail.get("class", 0), n - s)
    if c + s != n:
        raise ValueError(f"cell needs {n} pairs, has {avail}")
    return {"class": c, "sequence": s}


def draw(pools, meta, seed=SEED):
    """pools: {size: {key: scores}}. Returns the 60 pair records, sorted by pair id."""
    pairs = []
    for size in SIZES:
        quota = {"zero_shot": QUOTA[size], "fine_tuned": PER_SIZE - QUOTA[size]}
        for d in CONDS:
            cell = collections.defaultdict(list)
            for k, s in pools[size].items():
                if direction(s) == d:
                    cell[meta[k]["primary_type"]].append(k)
            take = split_types(quota[d], {t: len(v) for t, v in cell.items()})
            for t in TYPES:
                rng = random.Random(f"{seed}|rater-check|{size}|{d}|{t}")
                for k in rng.sample(sorted(cell[t]), take[t]):
                    s = pools[size][k]
                    pairs.append({"size": size, "key": k, "primary_type": t, "tier": meta[k]["tier"],
                                  "direction": d,
                                  "scores": {v: {m: {"zero_shot": s[v][m][0], "fine_tuned": s[v][m][1]}
                                                 for m in METRICS} for v in VERSIONS}})
    order = list(range(len(pairs)))
    random.Random(f"{seed}|rater-check|ids").shuffle(order)
    width = len(str(len(pairs)))
    for n, i in enumerate(order, 1):
        pairs[i]["pair"] = f"P{n:0{width}d}"
    pairs.sort(key=lambda p: p["pair"])
    ab = random.Random(f"{seed}|rater-check|ab")
    for p in pairs:
        p["A"] = "zero_shot" if ab.random() < 0.5 else "fine_tuned"
        p["B"] = "fine_tuned" if p["A"] == "zero_shot" else "zero_shot"
    return pairs


def rater_order(pairs, rater, seed=SEED):
    ids = [p["pair"] for p in pairs]
    random.Random(f"{seed}|rater-check|order|{rater}").shuffle(ids)
    return ids


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_keys(out=OUT):
    return json.load(open(out / "keys.json", encoding="utf-8"))


def cmd_draw(args):
    out = Path(args.out)
    if (out / "keys.json").exists():
        raise SystemExit(f"{out / 'keys.json'} exists; the sample is drawn once")
    meta = E.test_set()
    pools = {size: pool(size_scores(size)) for size in SIZES}
    pairs = draw(pools, meta)
    pool_counts = {size: dict(collections.Counter(direction(s) for s in pools[size].values()))
                   for size in SIZES}
    out.mkdir(parents=True, exist_ok=True)
    keys = {"_doc": "The unblinded key of the rater-check sample. Raters never read this file.",
            "seed": SEED, "rater_check_sha256": sha256(__file__), "quota_zero_shot": QUOTA,
            "per_size": PER_SIZE, "pool": pool_counts, "n_pairs": len(pairs), "pairs": pairs}
    (out / "keys.json").write_text(json.dumps(keys, indent=1) + "\n", encoding="utf-8")
    print(f"drawn: {len(pairs)} pairs; pool {pool_counts}; key {out / 'keys.json'}")


# --------------------------------------------------------------------------- #
# pictures
# --------------------------------------------------------------------------- #
def run_of(size, cond):
    return PAIRS[size][CONDS.index(cond)]


def source_name(size, cond, key):
    return f"{run_of(size, cond)}__{key}.puml"


def write_sources(pairs, src, answer=E.answer_text):
    """Each sample answer's PlantUML block (the scorer's extract_puml) as <run>__<key>.puml."""
    src.mkdir(parents=True, exist_ok=True)
    names = []
    for p in pairs:
        for cond in CONDS:
            name = source_name(p["size"], cond, p["key"])
            (src / name).write_text(extract_puml(answer(run_of(p["size"], cond), p["key"])) + "\n",
                                    encoding="utf-8")
            names.append(name)
    return names


def render_check(names, rows):
    """Counts of render status over the sample answers. Any answer that is not one picture
    (several pages, no image, a crash) stops the build: the raters see one picture per answer."""
    status = collections.Counter(rows.get(n, {}).get("status", "missing") for n in names)
    check = {"answers": len(names), "ok": status["ok"], "multi_page": status["multi_page"],
             "other": {k: v for k, v in status.items() if k not in ("ok", "multi_page")}}
    if check["ok"] != len(names):
        raise SystemExit(f"sample answers that are not one picture: {check}")
    return check


def cmd_render(args):
    import render as R
    out = Path(args.out)
    pairs = load_keys(out)["pairs"]
    names = write_sources(pairs, out / "render" / "src")
    R.render(names, out / "render" / "src", out / "render", threads=args.threads, heap_mb=args.heap_mb)
    rows = {r["key"]: r for r in R.read_manifest(out / "render" / "manifest.jsonl")}
    check = render_check(names, rows)
    (out / "render" / "check.json").write_text(json.dumps(check, indent=1) + "\n", encoding="utf-8")
    print(f"rendered {len(names)} answers: {check}")


# --------------------------------------------------------------------------- #
# rating sheets
# --------------------------------------------------------------------------- #
PAGE = """<!doctype html>
<html lang="uk"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Оцінювання діаграм</title>
<style>
:root { --bg:#fff; --fg:#1d1d1f; --muted:#666; --line:#d0d0d0; --accent:#1f5fbf; --sel:#e6eefc; }
@media (prefers-color-scheme: dark) { :root { --bg:#1b1b1d; --fg:#ececec; --muted:#a0a0a0;
  --line:#3a3a3d; --accent:#7aa7ff; --sel:#25324a; } }
body { margin:0; background:var(--bg); color:var(--fg); font:15px/1.45 system-ui, sans-serif; }
header, main, footer { max-width:1400px; margin:0 auto; padding:10px 16px; }
h1 { font-size:18px; margin:4px 0; } .muted { color:var(--muted); }
.q { font-weight:600; }
.pics { display:grid; grid-template-columns:repeat(3, minmax(0, 1fr)); gap:12px; align-items:start; }
body.stacked .pics { grid-template-columns:1fr; }
@media (max-width:700px) { .pics { grid-template-columns:1fr; } }
figure { margin:0; border:1px solid var(--line); border-radius:6px; padding:6px; background:#fff; }
figcaption { color:#333; font-weight:600; margin-bottom:4px; }
figure img { display:block; max-width:100%; margin:0 auto; cursor:zoom-in; }
.choices { display:flex; gap:10px; margin:10px 0; flex-wrap:wrap; }
button { font:inherit; padding:8px 16px; border:1px solid var(--line); border-radius:6px;
  background:var(--bg); color:var(--fg); cursor:pointer; }
button.on { background:var(--sel); border-color:var(--accent); font-weight:600; }
textarea { width:100%; box-sizing:border-box; min-height:44px; font:inherit; background:var(--bg);
  color:var(--fg); border:1px solid var(--line); border-radius:6px; padding:6px; }
nav { display:flex; gap:10px; align-items:center; flex-wrap:wrap; margin-top:8px; }
</style></head><body>
<header>
<h1>Оцінювання діаграм, оцінювач __RATER__</h1>
<p class="q">__QUESTION__</p>
<p class="muted">__GUIDANCE__ Натисніть на малюнок, щоб відкрити його в повному розмірі;
кнопка «Малюнки один під одним» показує їх на всю ширину. Клавіші: A, B, S (приблизно однаково),
стрілки — перехід між парами (працюють і в українській розкладці). Відповіді зберігаються в цьому
браузері, поки ви працюєте; наприкінці натисніть <b>Зберегти відповіді</b> й надішліть файл автору.</p>
</header>
<main>
<p><b id="pos"></b> <span class="muted" id="done"></span></p>
<div class="choices">
<button data-c="A">A ближче</button><button data-c="B">B ближче</button>
<button data-c="same">Приблизно однаково</button></div>
<textarea id="note" placeholder="Коментар (необов'язково)"></textarea>
<nav><button id="prev">Назад</button><button id="next">Далі</button>
<button id="save">Зберегти відповіді</button><button id="layout">Малюнки один під одним</button>
<span class="muted" id="msg"></span></nav>
<div class="pics" style="margin-top:12px">
<figure><figcaption>Оригінал</figcaption><img id="o" alt="оригінал"></figure>
<figure><figcaption>A</figcaption><img id="a" alt="малюнок A"></figure>
<figure><figcaption>B</figcaption><img id="b" alt="малюнок B"></figure>
</div>
</main>
<script>
const RATER = "__RATER__";
const IDS = __IDS__;
const STORE = "rating-" + RATER;
let ans = {}; try { ans = JSON.parse(localStorage.getItem(STORE) || "{}"); } catch (e) { ans = {}; }
let i = 0;
function keep() { try { localStorage.setItem(STORE, JSON.stringify(ans)); } catch (e) {} }
function show() {
  const id = IDS[i];
  for (const s of ["o", "a", "b"]) { const el = document.getElementById(s);
    el.src = "img/" + id + "_" + s + ".png"; el.onclick = () => window.open(el.src, "_blank"); }
  document.getElementById("pos").textContent = "Пара " + (i + 1) + " з " + IDS.length;
  const n = IDS.filter(x => ans[x] && ans[x].choice).length;
  document.getElementById("done").textContent = "(відповідей: " + n + ")";
  const a = ans[id] || {};
  document.querySelectorAll(".choices button").forEach(b => b.classList.toggle("on", b.dataset.c === a.choice));
  document.getElementById("note").value = a.comment || "";
}
function pick(c) { const id = IDS[i]; ans[id] = Object.assign(ans[id] || {}, {choice: c}); keep(); show(); }
function move(d) { i = Math.max(0, Math.min(IDS.length - 1, i + d)); show(); window.scrollTo(0, 0); }
document.querySelectorAll(".choices button").forEach(b => b.onclick = () => pick(b.dataset.c));
document.getElementById("note").oninput = e => { const id = IDS[i];
  ans[id] = Object.assign(ans[id] || {}, {comment: e.target.value}); keep(); };
function layout(stacked) { document.body.classList.toggle("stacked", stacked);
  document.getElementById("layout").textContent = stacked ? "Малюнки поруч" : "Малюнки один під одним";
  try { localStorage.setItem(STORE + "-stacked", stacked ? "1" : ""); } catch (e) {} }
let stacked = false; try { stacked = !!localStorage.getItem(STORE + "-stacked"); } catch (e) {}
layout(stacked);
document.getElementById("layout").onclick = () => layout(!document.body.classList.contains("stacked"));
document.getElementById("prev").onclick = () => move(-1);
document.getElementById("next").onclick = () => move(1);
document.addEventListener("keydown", e => { if (e.target.tagName === "TEXTAREA" || e.metaKey || e.ctrlKey || e.altKey) return;
  // e.code is the physical key, so A / B / S work in the Ukrainian layout too (where they type ф / и / і)
  if (e.code === "KeyA") pick("A"); else if (e.code === "KeyB") pick("B"); else if (e.code === "KeyS") pick("same");
  else if (e.key === "ArrowRight") move(1); else if (e.key === "ArrowLeft") move(-1); });
function toCSV(ids, ans) {
  const q = s => '"' + String(s || "").replace(/"/g, '""') + '"';
  return ["pair,choice,comment"].concat(ids.map(x => [x, (ans[x] || {}).choice || "",
    q((ans[x] || {}).comment)].join(","))).join("\\n") + "\\n";
}
document.getElementById("save").onclick = () => {
  const missing = IDS.filter(x => !(ans[x] && ans[x].choice));
  const url = URL.createObjectURL(new Blob([toCSV(IDS, ans)], {type: "text/csv"}));
  const link = document.createElement("a"); link.href = url; link.download = RATER + ".csv"; link.click();
  document.getElementById("msg").textContent = missing.length ?
    "Збережено; пар без відповіді: " + missing.length + "." : "Збережено всі " + IDS.length + " відповідей.";
};
show();
</script></body></html>
"""


def sheet_html(rater, ids):
    return (PAGE.replace("__RATER__", rater).replace("__IDS__", json.dumps(ids))
            .replace("__QUESTION__", QUESTION).replace("__GUIDANCE__", GUIDANCE))


def build_sheet(pairs, rater, sheet_dir, render_dir, images=IMAGES):
    """One rater's folder: index.html and img/<pair>_{o,a,b}.png. Nothing in it names a run,
    a condition or a test key."""
    img = sheet_dir / "img"
    img.mkdir(parents=True, exist_ok=True)
    for p in pairs:
        shutil.copyfile(images / f"{p['key']}.png", img / f"{p['pair']}_o.png")
        for side in ("A", "B"):
            name = source_name(p["size"], p[side], p["key"])[:-5] + ".png"
            shutil.copyfile(render_dir / name, img / f"{p['pair']}_{side.lower()}.png")
    (sheet_dir / "index.html").write_text(sheet_html(rater, rater_order(pairs, rater)), encoding="utf-8")


def cmd_sheets(args):
    out, sheets = Path(args.out), Path(args.sheets)
    pairs = load_keys(out)["pairs"]
    for rater in RATERS:
        d = sheets / rater
        if d.exists():
            shutil.rmtree(d)
        build_sheet(pairs, rater, d, out / "render" / "puml_images_1568")
        shutil.make_archive(str(sheets / f"rating_{rater}"), "zip", root_dir=sheets, base_dir=rater)
        print(f"sheet {rater}: {d / 'index.html'} and {sheets / f'rating_{rater}.zip'}")


# --------------------------------------------------------------------------- #
# the report
# --------------------------------------------------------------------------- #
def read_ratings(path, pairs):
    """{pair: {"choice", "comment"}}; every sample pair answered once with A, B or same."""
    ids = {p["pair"] for p in pairs}
    out = {}
    with open(path, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            pid, choice = row["pair"].strip(), row["choice"].strip()
            if pid not in ids:
                raise ValueError(f"{path}: unknown pair {pid}")
            if pid in out:
                raise ValueError(f"{path}: pair {pid} twice")
            if choice not in CHOICES:
                raise ValueError(f"{path}: pair {pid} has choice {choice!r}")
            out[pid] = {"choice": choice, "comment": (row.get("comment") or "").strip()}
    missing = sorted(ids - set(out))
    if missing:
        raise ValueError(f"{path}: {len(missing)} pairs without an answer, e.g. {missing[:3]}")
    return out


def picked(pair, choice):
    """The condition a choice names, or "same"."""
    return "same" if choice == "same" else pair[choice]


def higher(pair, version, metric):
    s = pair["scores"][version][metric]
    if s["zero_shot"] == s["fine_tuned"]:
        return None
    return "zero_shot" if s["zero_shot"] > s["fine_tuned"] else "fine_tuned"


def wilson(k, n, z=Z95):
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def agreement(pairs, ratings, prefer, select=None):
    """On the pairs where `prefer(pair)` names a condition (and `select` holds): A/B picks of that
    condition, their share among A/B picks with a Wilson interval, and the count of "same"."""
    n_pairs = agree = against = same = 0
    for p in pairs:
        h = prefer(p)
        if h is None or (select and not select(p)):
            continue
        n_pairs += 1
        c = picked(p, ratings[p["pair"]]["choice"])
        if c == "same":
            same += 1
        elif c == h:
            agree += 1
        else:
            against += 1
    n = agree + against
    ci = wilson(agree, n)
    return {"pairs": n_pairs, "picks": n, "agree": agree, "against": against, "same": same,
            "share": agree / n if n else None, "ci95": list(ci) if ci else None}


def by_metric(version, metric):
    return lambda p: higher(p, version, metric)


def by_direction(p):
    return p["direction"]


def v1_v2_differ(pairs, metric):
    return sorted(p["pair"] for p in pairs if higher(p, "v2", metric) != higher(p, "v1", metric))


def report(pairs, ratings):
    rep = {"n_pairs": len(pairs), "raters": sorted(ratings), "agreement": {}, "by_direction": {},
           "by_size": {}, "by_type": {}}
    for r, rt in ratings.items():
        rep["agreement"][r] = {v: {m: agreement(pairs, rt, by_metric(v, m)) for m in METRICS} for v in VERSIONS}
        # design §5.2: agreement with the pair's direction on the 40 and the 20 pairs
        rep["by_direction"][r] = {d: agreement(pairs, rt, by_direction, lambda p, d=d: p["direction"] == d)
                                  for d in CONDS}
        rep["by_size"][r] = {s: {m: agreement(pairs, rt, by_metric("v2", m), lambda p, s=s: p["size"] == s)
                                 for m in METRICS} for s in SIZES}
        rep["by_type"][r] = {t: {m: agreement(pairs, rt, by_metric("v2", m), lambda p, t=t: p["primary_type"] == t)
                                 for m in METRICS} for t in TYPES}
    if len(ratings) == 2:
        a, b = (ratings[r] for r in sorted(ratings))
        # design §5.3: kappa on the raters' own labels, A / B / same
        rep["between_raters"] = E.kappa([a[p["pair"]]["choice"] for p in pairs],
                                        [b[p["pair"]]["choice"] for p in pairs])
    rep["v1_v2_prefer_differently"] = {m: v1_v2_differ(pairs, m) for m in METRICS}
    rep["both_against_direction"] = [
        {"pair": p["pair"], "size": p["size"], "key": p["key"], "primary_type": p["primary_type"],
         "direction": p["direction"], "scores_v2": p["scores"]["v2"],
         "comments": {r: ratings[r][p["pair"]]["comment"] for r in sorted(ratings)}}
        for p in pairs
        if len(ratings) == 2 and all(picked(p, ratings[r][p["pair"]]["choice"]) not in (p["direction"], "same")
                                     for r in ratings)]
    return rep


def read_differences(path, against):
    """The author's lines, one per pair both raters chose against the metric: pair -> (differs, kind).

    Refuses a file whose pairs are not exactly the report's list, or a row without a description.
    """
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or set(rows[0]) != {"pair", "differs", "kind"}:
        raise SystemExit(f"{path}: expected columns pair, differs, kind")
    got = [r["pair"] for r in rows]
    want = [p["pair"] for p in against]
    if sorted(got) != sorted(want) or len(set(got)) != len(got):
        raise SystemExit(f"{path}: pairs {sorted(got)} are not the report's list {sorted(want)}")
    for r in rows:
        if not r["differs"].strip():
            raise SystemExit(f"{path}: {r['pair']} has no description")
    return {r["pair"]: (r["differs"].strip(), r["kind"].strip()) for r in rows}


def add_differences(rep, diffs):
    for p in rep["both_against_direction"]:
        p["differs"], p["kind"] = diffs[p["pair"]]
    return rep


def _fmt(a):
    if not a["picks"]:
        return f"– (pairs {a['pairs']}, same {a['same']})"
    lo, hi = a["ci95"]
    return (f"{a['agree']}/{a['picks']} = {a['share']:.3f} [{lo:.3f}, {hi:.3f}]; same {a['same']} "
            f"(pairs {a['pairs']})")


def _counts(a):
    return f"{a['agree']} / {a['against']} / {a['same']}"


def report_md(rep):
    L = ["# Human-rater check: report", "",
         "All numbers descriptive, outside the claim family. Agreement: of the pairs where the metric "
         "differs and the rater picked A or B, those where the pick is the answer the metric scores "
         "higher; Wilson 95% interval.", ""]
    L += ["## Agreement with the metric", "", "| Rater | Version | Element F1 | Relationship F1 |", "|---|---|---|---|"]
    for r in rep["raters"]:
        for v in VERSIONS:
            a = rep["agreement"][r][v]
            L.append(f"| {r} | {v} | {_fmt(a['ef'])} | {_fmt(a['rf'])} |")
    L += ["", "## By the pair's direction (v2)", "",
          "Agreement with the answer the metric prefers (Element F1 when it differs, otherwise "
          "Relationship F1), on the pairs of each direction.", "",
          "| Rater | Metric prefers | Agreement |", "|---|---|---|"]
    for r in rep["raters"]:
        for d in CONDS:
            L.append(f"| {r} | {d.replace('_', '-')} | {_fmt(rep['by_direction'][r][d])} |")
    if "between_raters" in rep:
        k = rep["between_raters"]
        kap = "–" if k["kappa"] is None else f"{k['kappa']:.3f}"
        L += ["", "## Between the raters", "",
              f"Same answer on {k['agree']} of {k['n']} pairs ({k['raw']:.3f}); Cohen's kappa {kap} "
              "(A / B / about the same)."]
    L += ["", "## Counts by size and by type (v2; agree / against / same)", "",
          "| Rater | Group | Element F1 | Relationship F1 |", "|---|---|---|---|"]
    for r in rep["raters"]:
        for s in SIZES:
            a = rep["by_size"][r][s]
            L.append(f"| {r} | {s} | {_counts(a['ef'])} | {_counts(a['rf'])} |")
        for t in TYPES:
            a = rep["by_type"][r][t]
            L.append(f"| {r} | {t} | {_counts(a['ef'])} | {_counts(a['rf'])} |")
    d = rep["v1_v2_prefer_differently"]
    L += ["", "## v1 and v2", "",
          f"Sample pairs where v1 and v2 prefer different answers: Element F1 {len(d['ef'])} "
          f"({', '.join(d['ef']) or 'none'}), Relationship F1 {len(d['rf'])} ({', '.join(d['rf']) or 'none'})."]
    L += ["", "## Pairs where both raters chose against the metric", "",
          "| Pair | Size | Type | Metric prefers | What differs (written by the author) | Kind |",
          "|---|---|---|---|---|---|"]
    for p in rep["both_against_direction"]:
        L.append(f"| {p['pair']} | {p['size']} | {p['primary_type']} | {p['direction'].replace('_', '-')} "
                 f"| {p.get('differs', '')} | {p.get('kind', '')} |")
    return "\n".join(L) + "\n"


def cmd_report(args):
    out = Path(args.out)
    pairs = load_keys(out)["pairs"]
    ratings = {}
    for r in RATERS:
        p = out / "ratings" / f"{r}.csv"
        if p.exists():
            ratings[r] = read_ratings(p, pairs)
    if not ratings:
        raise SystemExit(f"no ratings in {out / 'ratings'}")
    rep = report(pairs, ratings)
    if (out / DIFFERENCES).exists():
        add_differences(rep, read_differences(out / DIFFERENCES, rep["both_against_direction"]))
    rep["rater_check_sha256"] = sha256(__file__)
    (out / "report.json").write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
    (out / "report.md").write_text(report_md(rep), encoding="utf-8")
    print(f"report: {out / 'report.md'} (raters {sorted(ratings)})")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("draw", "render", "sheets", "report"):
        sp = sub.add_parser(name)
        sp.add_argument("--out", default=str(OUT))
        if name == "sheets":
            sp.add_argument("--sheets", default=str(SHEETS))
        if name == "render":
            sp.add_argument("--threads", type=int, default=3)
            sp.add_argument("--heap-mb", type=int, default=8192)
    args = ap.parse_args()
    {"draw": cmd_draw, "render": cmd_render, "sheets": cmd_sheets, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
