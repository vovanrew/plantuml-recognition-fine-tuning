"""Tests for train/select_checkpoint.py, the checkpoint rule of the recipe.

Run from fine-tune/ under the scorer's Python (the rule runs there) or the training environment:
  python3 train/tests/test_select_checkpoint.py
(or with pytest: .venv/bin/python -m pytest <this file>)

The rule's interval reproduces a stored paired comparison of the instrument's bootstrap (fixtures:
two arms of the precision test of decisions/gpu-precision/, 200 test diagrams, scored under scorer
v2; train/tests/fixtures/README.md). Its decisions are tested on made-up checkpoint scorings. No
dev-set answer is read: none exists before the rule is frozen.
"""
import copy
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path

FT = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "checkpoint_scores"
sys.path.insert(0, str(FT / "train"))
import select_checkpoint as S  # noqa: E402

_cache = {}


def modules():
    if "m" not in _cache:
        _cache["m"] = S.check_instrument()
    return _cache["m"]


def fixture(name):
    return S.load(FIXTURES / name)


def write(folder, csr_rows, element_rows, scorer=None):
    """A score folder in the layout of train/dev_scoring.py score."""
    folder = Path(folder)
    (folder / "csr").mkdir(parents=True)
    (folder / "element_f1").mkdir(parents=True)
    (folder / "csr" / "csr_results.json").write_text(json.dumps({"summary": {}, "diagrams": csr_rows}))
    (folder / "element_f1" / "element_f1_results.json").write_text(json.dumps(
        {"scorer": scorer or S.V2_SCORER, "summary": {}, "diagrams": element_rows}))
    return folder


def made_up(folder, compiled, names_found, n=200, names_per_diagram=5):
    """A checkpoint scoring of n diagrams: the first `compiled` compile; the first `names_found`
    diagrams find all their names, the others none (a failed or wrong answer)."""
    keys = [f"{i:040x}" for i in range(n)]
    csr = [{"key": k, "compiled": i < compiled} for i, k in enumerate(keys)]
    el = [{"key": k, "tp": names_per_diagram if i < names_found else 0, "fp": 0,
           "fn": 0 if i < names_found else names_per_diagram, "precision": 0.0, "recall": 0.0, "f1": 0.0,
           "compiled": i < compiled} for i, k in enumerate(keys)]
    return write(folder, csr, el)


SET_SHA = "5" * 64


def provenance(folder, step, run="/runs/2B", set_sha=SET_SHA, **changes):
    """The scores.json that train/dev_scoring.py score writes, with the answer manifest of a
    checkpoint answered at the recipe's settings; `changes` replaces manifest or server fields."""
    server = {"served_name": "Qwen3.5-2B-adapter", "model": "Qwen/Qwen3.5-2B",
              "revision": "15852e8c16360a2fea060d615a32b45270f8a8fc", "dtype": "bfloat16",
              "adapter": {"path": f"{run}/{S.recipe.checkpoint_dir(step)}", "attached": "not merged",
                          "adapter_model.safetensors": f"{step:064d}", "adapter_config.json": "c" * 64},
              "decoding": "greedy", "eos_token_id": [248046, 248044],
              "recurrent_layers": {"chosen": "fla", "active": "fla"}, "timing": False,
              "device": "cuda", "gpu": S.CARD,
              "versions": {"python": "3.12.3", "torch": "2.14.1", "transformers": "5.18.0", "peft": "0.21.2"}}
    manifest = {"max_tokens": 5376, "timeout_seconds": 600, "runner_exit_code": 0,
                "set_file_sha256": set_sha, "server": server}
    for k, v in changes.items():
        (server if k in server else manifest)[k] = v
    scores = {"set_file_sha256": set_sha, "scorer": dict(S.SCORER_PINS, java="Temurin 26.0.1"),
              "answer_manifest": manifest}
    (Path(folder) / "scores.json").write_text(json.dumps(scores))
    return folder


def made_up_run(tmp, quality=((100, 100),) * 4, steps=(625, 1250, 1875, 2500), every=False, **changes):
    """Four checkpoint scorings of one run with their scores.json; `changes` go to the last one,
    or to every one with `every`."""
    folders = []
    for i, ((compiled, names), step) in enumerate(zip(quality, steps)):
        f = made_up(Path(tmp) / f"s{i}", compiled, names)
        provenance(f, step, **(changes if every or i == len(steps) - 1 else {}))
        folders.append(f)
    return folders


def decide(folders):
    aggregate, bootstrap = modules()
    return S.select([S.load(f) for f in folders], aggregate, bootstrap)


# --------------------------------------------------------------------------- #
def test_the_instrument_code_is_at_the_tag():
    aggregate, bootstrap = modules()
    assert (bootstrap.SEED, bootstrap.N_RESAMPLES) == (20260614, 1000)


def test_the_rules_metrics_equal_the_runners_summaries():
    for name in ("bf16", "nf4_mlp"):
        c = fixture(name)
        aggregate, _ = modules()
        points = S.statistics(aggregate, c["csr"], c["element"])
        summary = json.load(open(FIXTURES / name / "element_f1" / "element_f1_results.json"))["summary"]
        assert math.isclose(points["element_f1_micro_full"], summary["zeros_for_failed"]["micro"]["f1"])
        csr = json.load(open(FIXTURES / name / "csr" / "csr_results.json"))["summary"]
        assert math.isclose(points["csr"], csr["csr"])


def test_reproduces_the_stored_paired_interval():
    result = decide([FIXTURES / "nf4_mlp", FIXTURES / "bf16"])
    diff = result["comparisons"][0]["earlier_minus_last"]
    assert [round(diff["csr"]["ci_low"], 4), round(diff["csr"]["ci_high"], 4)] == [-0.06, 0.085]
    assert round(diff["csr"]["diff_point"], 4) == 0.015
    f1 = diff["element_f1_micro_full"]
    assert [round(f1["ci_low"], 4), round(f1["ci_high"], 4)] == [-0.1091, 0.07]
    assert round(f1["diff_point"], 4) == -0.0191
    assert result["selected"] == 1 and result["qualifying"] == []


def test_the_last_checkpoint_by_default():
    with tempfile.TemporaryDirectory() as tmp:
        folders = [made_up(Path(tmp) / f"c{i}", compiled=100 + d, names_found=100 + d)
                   for i, d in enumerate((-4, 3, -2, 0))]          # differences within noise
        result = decide(folders)
        assert result["selected"] == 3 and result["qualifying"] == []


def test_an_earlier_checkpoint_clearly_better_on_csr_is_taken():
    with tempfile.TemporaryDirectory() as tmp:
        earlier = made_up(Path(tmp) / "e", compiled=150, names_found=100)
        last = made_up(Path(tmp) / "l", compiled=100, names_found=100)
        result = decide([earlier, last])
        assert result["qualifying"] == [0] and result["selected"] == 0


def test_an_earlier_checkpoint_clearly_better_on_element_f1_is_taken():
    with tempfile.TemporaryDirectory() as tmp:
        earlier = made_up(Path(tmp) / "e", compiled=100, names_found=150)
        last = made_up(Path(tmp) / "l", compiled=100, names_found=100)
        assert decide([earlier, last])["selected"] == 0


def test_better_on_one_metric_but_worse_on_the_other_is_not_taken():
    with tempfile.TemporaryDirectory() as tmp:
        earlier = made_up(Path(tmp) / "e", compiled=150, names_found=95)
        last = made_up(Path(tmp) / "l", compiled=100, names_found=100)
        result = decide([earlier, last])
        c = result["comparisons"][0]["earlier_minus_last"]
        assert c["csr"]["ci_low"] > 0 and c["element_f1_micro_full"]["diff_point"] < 0
        assert result["selected"] == 1


def test_better_on_element_f1_but_worse_on_csr_is_not_taken():
    with tempfile.TemporaryDirectory() as tmp:
        earlier = made_up(Path(tmp) / "e", compiled=95, names_found=150)
        last = made_up(Path(tmp) / "l", compiled=100, names_found=100)
        result = decide([earlier, last])
        c = result["comparisons"][0]["earlier_minus_last"]
        assert c["element_f1_micro_full"]["ci_low"] > 0 and c["csr"]["diff_point"] < 0
        assert result["selected"] == 1


def test_of_several_qualifying_the_latest_is_taken():
    with tempfile.TemporaryDirectory() as tmp:
        folders = [made_up(Path(tmp) / "a", 150, 150), made_up(Path(tmp) / "b", 160, 160),
                   made_up(Path(tmp) / "c", 155, 150), made_up(Path(tmp) / "d", 100, 100)]
        result = decide(folders)
        assert result["qualifying"] == [0, 1, 2] and result["selected"] == 2


def refused(fn):
    try:
        fn()
    except S.Refused:
        return True
    return False


def test_refusals():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        good = made_up(tmp / "good", 100, 100)
        fewer = made_up(tmp / "fewer", 100, 100, n=199)
        assert refused(lambda: decide([fewer, good]))                      # different diagrams
        other_jar = dict(S.V2_SCORER, extractor_jar_sha256="0" * 64)
        csr = json.load(open(good / "csr" / "csr_results.json"))["diagrams"]
        el = json.load(open(good / "element_f1" / "element_f1_results.json"))["diagrams"]
        assert refused(lambda: S.load(write(tmp / "v1jar", csr, el, other_jar)))
        v1_rules = copy.deepcopy(S.V2_SCORER)
        v1_rules["rules"]["collapse_whitespace"] = False
        assert refused(lambda: S.load(write(tmp / "v1rules", csr, el, v1_rules)))
        assert refused(lambda: S.load(write(tmp / "split", csr[:-1], el)))  # files cover different keys
        assert refused(lambda: S.load(good, keys=sorted(r["key"] for r in csr)[:-1]))
        assert refused(lambda: S.select([], *modules()))


def test_a_run_is_checked_before_the_rule():
    with tempfile.TemporaryDirectory() as tmp:
        assert len(S.check_run([S.load(f) for f in made_up_run(tmp)])) == 4
    bad = {"three checkpoints": dict(steps=(625, 1250, 1875)),
           "out of order": dict(steps=(625, 1875, 1250, 2500)),
           "the 5,000 run's steps": dict(steps=(157, 313, 470, 626)),
           "another cap": dict(max_tokens=4096), "another deadline": dict(timeout_seconds=300),
           "a failed runner": dict(runner_exit_code=1), "no adapter": dict(adapter=None),
           "another revision": dict(revision="0" * 40), "timing mode": dict(timing=True),
           "another set file": dict(set_file_sha256="6" * 64),
           "another recurrent-layer path at one checkpoint": dict(recurrent_layers={"chosen": "reference",
                                                                                     "active": "reference"})}
    for what, change in bad.items():
        with tempfile.TemporaryDirectory() as tmp:
            steps = change.pop("steps", (625, 1250, 1875, 2500))
            folders = made_up_run(tmp, quality=((100, 100),) * len(steps), steps=steps, **change)
            assert refused(lambda: S.check_run([S.load(f) for f in folders])), what
    # the same at every checkpoint, so that the servers agree and only the check of the field refuses
    everywhere = {"another card": dict(gpu="NVIDIA H100 80GB HBM3"), "the Mac": dict(device="mps", gpu=None),
                  "float32": dict(dtype="float32"), "timing mode": dict(timing=True),
                  "no recurrent-layer implementation named": dict(recurrent_layers={}),
                  "recurrent-layer field empty": dict(recurrent_layers=None),
                  "another cap": dict(max_tokens=4096), "another deadline": dict(timeout_seconds=300),
                  "a failed runner": dict(runner_exit_code=1), "no adapter": dict(adapter=None)}
    for what, change in everywhere.items():
        with tempfile.TemporaryDirectory() as tmp:
            folders = made_up_run(tmp, every=True, **change)
            assert refused(lambda: S.check_run([S.load(f) for f in folders])), what
    with tempfile.TemporaryDirectory() as tmp:
        folders = made_up_run(tmp)
        provenance(folders[-1], 2500, run="/runs/other")                    # another run folder
        assert refused(lambda: S.check_run([S.load(f) for f in folders]))
        provenance(folders[-1], 2500, adapter={"path": "/runs/2B/step_02500", "attached": "not merged",
                                               "adapter_model.safetensors": f"{1875:064d}",
                                               "adapter_config.json": "c" * 64})   # the same adapter twice
        assert refused(lambda: S.check_run([S.load(f) for f in folders]))
        os.remove(Path(folders[0]) / "scores.json")
        assert refused(lambda: S.check_run([S.load(f) for f in folders]))
    with tempfile.TemporaryDirectory() as tmp:
        folders = made_up_run(tmp)
        scores = json.load(open(Path(folders[1]) / "scores.json"))
        scores["scorer"]["commit"] = "8f85611"                              # scorer v1's commit
        (Path(folders[1]) / "scores.json").write_text(json.dumps(scores))
        assert refused(lambda: S.check_run([S.load(f) for f in folders]))
    for field in S.SERVER_FIELDS:                                          # a field missing at every checkpoint
        with tempfile.TemporaryDirectory() as tmp:
            folders = made_up_run(tmp)
            for f in folders:
                scores = json.load(open(Path(f) / "scores.json"))
                del scores["answer_manifest"]["server"][field]
                (Path(f) / "scores.json").write_text(json.dumps(scores))
            assert refused(lambda: S.check_run([S.load(f) for f in folders])), field


def test_a_5000_pair_run_is_accepted_with_its_steps():
    with tempfile.TemporaryDirectory() as tmp:
        folders = made_up_run(tmp, steps=(157, 313, 470, 626))
        assert len(S.check_run([S.load(f) for f in folders], pairs=5000)) == 4
        assert refused(lambda: S.check_run([S.load(f) for f in folders], pairs=20000))


def test_the_scorer_pins_are_dev_scorings():
    import dev_scoring
    assert S.SCORER_PINS == {"tag": dev_scoring.SCORER_TAG, "commit": dev_scoring.SCORER_COMMIT,
                             "extractor_jar_sha256": dev_scoring.EXTRACTOR_SHA256,
                             "render_jar_sha256": dev_scoring.RENDER_SHA256}


def test_command_line_writes_the_decision_and_its_inputs():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        keys = [f"{i:040x}" for i in range(200)]
        set_file = tmp / "set.json"
        set_file.write_text(json.dumps({"diagrams": [{"key": k + ".puml"} for k in keys]}))
        set_sha = S.sha256_file(set_file)
        folders = []
        for i, ((compiled, names), step) in enumerate(zip(((150, 150), (100, 100), (101, 99), (100, 100)),
                                                          (625, 1250, 1875, 2500))):
            folders.append(provenance(made_up(tmp / f"s{i}", compiled, names), step, set_sha=set_sha))
        out = tmp / "run" / "selection.json"
        S.main([str(f) for f in folders] + ["--out", str(out), "--set-file", str(set_file)])
        result = json.load(open(out))
        assert result["selected"] == 0 and result["qualifying"] == [0] and result["n_diagrams"] == 200
        assert result["steps"] == [625, 1250, 1875, 2500] and result["pairs"] == 20000
        assert [a["path"].rsplit("/", 1)[1] for a in result["adapters"]] == \
            ["step_00625", "step_01250", "step_01875", "step_02500"]
        assert result["seed"] == 20260614 and result["n_resamples"] == 1000
        assert len(result["inputs"]) == 4 and len(result["select_checkpoint_sha256"]) == 64
        assert len(result["recipe_sha256"]) == 64
        other = tmp / "other.json"
        other.write_text(json.dumps({"diagrams": [{"key": k + ".puml"} for k in keys], "x": 1}))
        assert refused(lambda: S.main([str(f) for f in folders] + ["--out", str(tmp / "y.json"),
                                                                    "--set-file", str(other)]))
        assert refused(lambda: S.main([str(f) for f in folders] + ["--out", str(tmp / "x.json")]))  # not the dev set


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print("ok  ", name, flush=True)
        except BaseException as exc:  # noqa: BLE001 - SystemExit included: a refusal is a failure here
            failed += 1
            print("FAIL", name, f"{type(exc).__name__}: {exc}", flush=True)
    print(f"{len(tests) - failed} of {len(tests)} passed")
    sys.exit(1 if failed else 0)
