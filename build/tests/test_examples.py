"""Tests for build/examples.py: the training example of a pair, against the server's request path
and the model's own template, and the examples files against the frozen lists.

Run from fine-tune/:  .venv/bin/python build/tests/test_examples.py
(or with pytest: .venv/bin/python -m pytest <this file>)

Reads, without model weights: the Qwen3.5-2B processor in .local/hf at the pinned revision, the
benchmark prompt, data/sets/, data/examples/ and the pictures under data/sets/render/. The test that
no target is edited also reads the corpus sources (build/pool.py PUML_DIR).
"""
import base64
import json
import os
import sys
from pathlib import Path

FT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", str(FT / ".local/hf"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
sys.path.insert(0, str(FT / "build"))
sys.path.insert(0, str(FT / "infer"))
import examples as E  # noqa: E402

MODEL, REVISION = "Qwen/Qwen3.5-2B", "15852e8c16360a2fea060d615a32b45270f8a8fc"
EXAMPLES = FT / "data" / "examples"
TOKEN_CAP = 5376
# Targets the scorer reads from the whole file: no lower-case @startuml ... @enduml pair
# (decisions/pilot-run/record.md, Context).
WHOLE_FILE = {"66acafcb0bc017e7faa53176da369956c47e4fea.puml",
              "7e679373b7a79ff77416ec504b0fcef93c75597f.puml",
              "c7c58b4a73463f2976de75839a156317ce1d80b0_25.puml",
              "c7c58b4a73463f2976de75839a156317ce1d80b0_27.puml",
              "c922d54ce5036852db069e2385d59981a95d6fc5.puml",
              "d37c9d77a29fcbdd463b971acc16a16e1c3be008_01.puml"}
_cache = {}


def processor():
    if "processor" not in _cache:
        from transformers import AutoProcessor
        _cache["processor"] = AutoProcessor.from_pretrained(MODEL, revision=REVISION)
    return _cache["processor"]


def rows(name):
    if name not in _cache:
        _cache[name] = E.load_examples(EXAMPLES / f"{name}.jsonl")
    return _cache[name]


def first_of_each_cell():
    """The rank-1 example of each of the 8 type and tier cells."""
    return [r for r in rows("train") if r["rank"] == 1]


def built(row, with_newline):
    proc = processor()
    image = E.user_turn.open_image(FT / row["picture"])
    ids = E.target_ids(proc.tokenizer, row["target"])
    return E.example(proc, E.prompt_text(), image, ids, E.end_ids(with_newline)), ids


def runner_request(picture):
    """The request infer_runner.py sends for a picture: the text part first, then the picture."""
    url = "data:image/png;base64," + base64.b64encode((FT / picture).read_bytes()).decode()
    return {"model": MODEL, "max_tokens": 16, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": E.prompt_text()},
                {"type": "image_url", "image_url": {"url": url}}]}]}


def served_inputs(request):
    """The model inputs infer/server.py builds from a request, taken where it calls generate."""
    import server as S
    seen = {}

    def capture(model, inputs, max_tokens, min_tokens=0):
        seen.update({k: v.clone() for k, v in inputs.items()})
        return [E.IM_END], 0.0, 0.0

    S.STATE.clear()
    S.STATE.update(served_name=MODEL, timing=False, processor=processor(), model=None,
                   identity={"device": "cpu"}, served=0)
    real, S.generate = S.generate, capture
    try:
        S.complete(request)
    finally:
        S.generate = real
    return seen


# --------------------------------------------------------------------------- #
# One example
# --------------------------------------------------------------------------- #
def test_the_prompt_is_the_pinned_file_as_the_runner_reads_it():
    text = E.prompt_text()
    assert text == E.PROMPT_FILE.read_text(encoding="utf-8").strip()
    assert E.sha256_file(E.PROMPT_FILE) == E.PROMPT_SHA256


def test_the_end_tokens_are_im_end_and_the_templates_newline():
    tok = processor().tokenizer
    assert tok.convert_tokens_to_ids("<|im_end|>") == E.IM_END
    assert tok.encode("\n", add_special_tokens=False) == [E.NEWLINE]
    assert E.end_ids(False) == [E.IM_END] and E.end_ids(True) == [E.IM_END, E.NEWLINE]


def test_the_labels_cover_the_target_and_its_end_tokens_only():
    for with_newline in (False, True):
        for row in first_of_each_cell():
            (inputs, labels, n_prompt), ids = built(row, with_newline)
            ends = E.end_ids(with_newline)
            n = inputs["input_ids"].shape[1]
            assert n == n_prompt + len(ids) + len(ends)
            assert (labels[0, :n_prompt] == -100).all()
            assert labels[0, n_prompt:].tolist() == ids + ends == inputs["input_ids"][0, n_prompt:].tolist()
            for k in ("attention_mask", "mm_token_type_ids"):
                if k in inputs:
                    assert inputs[k].shape == inputs["input_ids"].shape, k
            assert inputs["attention_mask"][0, n_prompt:].tolist() == [1] * (n - n_prompt)
            if "mm_token_type_ids" in inputs:
                assert inputs["mm_token_type_ids"][0, n_prompt:].tolist() == [0] * (n - n_prompt)


def test_the_training_prefix_is_what_the_server_builds_from_the_runners_request():
    """The byte-equality test of plan Phase B, through infer/server.py's own request path."""
    for row in first_of_each_cell():
        (inputs, _, n_prompt), _ = built(row, False)
        served = served_inputs(runner_request(row["picture"]))
        assert served["input_ids"].shape[1] == n_prompt
        assert set(served) == set(inputs), (sorted(served), sorted(inputs))
        for k, v in served.items():
            # per-position tensors are compared on the prompt; the picture's tensors whole
            mine = inputs[k] if k in ("pixel_values", "image_grid_thw") else inputs[k][:, :n_prompt]
            assert mine.shape == v.shape and mine.dtype == v.dtype and (mine == v).all(), (row["key"], k)


def test_the_example_is_the_models_own_template_of_the_two_turn_conversation():
    """With the newline, an example is the chat template's rendering of the user turn and the
    target as the assistant turn, token for token; without it, the same less its last token."""
    proc = processor()
    for row in first_of_each_cell():
        image = E.user_turn.open_image(FT / row["picture"])
        conversation = [E.user_turn.message(E.prompt_text(), image),
                        {"role": "assistant", "content": [{"type": "text", "text": row["target"]}]}]
        whole = proc.apply_chat_template(conversation, tokenize=True, return_dict=True,
                                         return_tensors="pt", enable_thinking=False)
        (with_nl, _, _), _ = built(row, True)
        (without, _, _), _ = built(row, False)
        assert with_nl["input_ids"].tolist() == whole["input_ids"].tolist(), row["key"]
        assert without["input_ids"].tolist() == whole["input_ids"][:, :-1].tolist(), row["key"]
        assert (with_nl["pixel_values"] == whole["pixel_values"]).all()


# --------------------------------------------------------------------------- #
# The examples files
# --------------------------------------------------------------------------- #
def test_the_examples_follow_the_frozen_lists():
    for name, count in (("train", 20000), ("loss_curve", 200)):
        listed, made = E.set_rows(name), rows(name)
        assert len(made) == len(listed) == count, name
        for a, b in zip(listed, made):
            assert (a["key"], a["rank"], a["primary_type"], a["tier"]) == \
                   (b["key"], b["rank"], b["primary_type"], b["tier"])
            assert b["picture"] == E.picture_path(name, b["key"]) and (FT / b["picture"]).is_file()
            assert b["target_sha256"] == E.sha256_text(b["target"])
    assert len(E.load_examples(EXAMPLES / "train.jsonl", 125)) == 1000
    assert len(E.load_examples(EXAMPLES / "train.jsonl", 625)) == 5000


def test_the_pictures_are_the_rendered_ones():
    """Pixel hashes of the 8 first pictures and of the 200 loss-curve pictures equal the render
    manifest; the build checks all 20,200."""
    for name, chosen in (("train", first_of_each_cell()), ("loss_curve", rows("loss_curve"))):
        shown = E.manifest(name)
        for r in chosen:
            assert r["pixels_sha256"] == shown[r["key"]]["pixels_sha256"]
            assert E.pixels_sha256(FT / r["picture"]) == r["pixels_sha256"], r["key"]


def test_no_target_is_edited():
    """Every target equals build/pool.py target_text on its corpus source."""
    for name in ("train", "loss_curve"):
        for r in rows(name):
            assert r["target"] == E.target_of(r["key"]), r["key"]


def test_the_targets_read_from_the_whole_file_drop_the_header_only():
    import pool
    found = {r["key"]: r for r in rows("train") if r["key"] in WHOLE_FILE}
    assert set(found) == WHOLE_FILE
    for key, r in found.items():
        source = pool.read_for_tokens(Path(pool.PUML_DIR) / key)
        assert "' Source: World of Code" not in r["target"]
        assert r["target"] == "".join(source.splitlines(keepends=True)[3:]).strip().strip("`").strip()
        assert not ("@startuml" in r["target"] and "@enduml" in r["target"])


def test_every_target_round_trips_through_the_tokenizer_and_stays_under_the_cap():
    """The training tokenizer decodes each target's ids back to the target in Unicode NFC (its
    normaliser), never gives more tokens than D4 counted with the slow class, and so never more
    than the output cap. Measured on 2026-10-17: 1 target is not in NFC (rank 883 of the 20,000);
    16 targets holding Thai text get fewer tokens than D4 counted, none in the 1,000."""
    import unicodedata
    tok = processor().tokenizer
    counted = {r["key"]: r["target_tokens"] for n in ("train", "loss_curve") for r in E.set_rows(n)}
    not_nfc, fewer = 0, 0
    for name in ("train", "loss_curve"):
        for r in rows(name):
            ids = E.target_ids(tok, r["target"])
            nfc = unicodedata.normalize("NFC", r["target"])
            assert tok.decode(ids) == nfc, r["key"]
            assert len(ids) <= counted[r["key"]] <= TOKEN_CAP, r["key"]
            not_nfc += nfc != r["target"]
            fewer += len(ids) < counted[r["key"]]
    assert (not_nfc, fewer) == (1, 16), (not_nfc, fewer)


def test_the_dev_runner_set_is_dev_json_with_the_picture_sizes():
    from PIL import Image
    made = json.load(open(EXAMPLES / "dev_runner_set.json"))["diagrams"]
    listed = E.set_rows("dev")
    assert len(made) == len(listed) == 200
    for a, b in zip(listed, made):
        assert {k: b[k] for k in a} == a
        with Image.open(FT / E.picture_path("dev", a["key"])) as im:
            assert im.size == (b["image_width"], b["image_height"]), a["key"]


def test_the_examples_files_match_their_checksums():
    lines = (EXAMPLES / "SHA256SUMS").read_text().splitlines()
    names = sorted(p.name for p in EXAMPLES.iterdir() if p.is_file() and p.name != "SHA256SUMS")
    assert [line.split("  ", 1)[1] for line in lines] == names
    for line in lines:
        digest, name = line.split("  ", 1)
        assert E.sha256_file(EXAMPLES / name) == digest, name
    config = json.load(open(EXAMPLES / "config.json"))
    assert config["code_sha256"]["build/examples.py"] == E.sha256_file(FT / "build/examples.py")


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print("ok  ", name, flush=True)
        except Exception as exc:  # noqa: BLE001 - a test runner reports every failure
            failed += 1
            print("FAIL", name, f"{type(exc).__name__}: {exc}", flush=True)
    print(f"{len(tests) - failed} of {len(tests)} passed")
    sys.exit(1 if failed else 0)
