"""Tests for build/user_turn.py: the user turn of decisions/prompt-arrangement (picture, newline,
text), its rendering at the three sizes, and the requests it refuses.

Run from fine-tune/:  .venv/bin/python build/tests/test_user_turn.py
(pytest collects the same functions; the environment that holds transformers has no pytest.)

Reads, without model weights: the Qwen3.5-2B processor in .local/hf at the pinned revision, the
chat templates of the three sizes saved by the Scout of prompt-arrangement
(decisions/prompt-arrangement/scripts/out/hf_small_files, hashes in its SHA256SUMS; tokenizer.json
and preprocessor_config.json are byte-identical at the three sizes), the benchmark prompt and one
standardized test image.
"""
import base64
import hashlib
import os
import sys
from pathlib import Path

FT = Path(__file__).resolve().parents[2]
UNI = FT.parent
os.environ.setdefault("HF_HOME", str(FT / ".local/hf"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import user_turn as U  # noqa: E402

MODEL, REVISION = "Qwen/Qwen3.5-2B", "15852e8c16360a2fea060d615a32b45270f8a8fc"
TEMPLATES = FT / "decisions/prompt-arrangement/scripts/out/hf_small_files"
TEMPLATE_SHA256 = {"2B": "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80",
                   "9B": "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715",
                   "27B": "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715"}
PROMPT_PATH = UNI / "plantuml-recognition-evaluation/prompts/zero_shot.txt"
PROMPT_SHA256 = "097351ffce6134cb903322718facca5488f978766d46381de03934c8c9f01552"
IMAGE = UNI / "baseline-evaluation/data/puml_images_1568/000133b38e70cfb70834681221553363e9f37714.png"
# What the official template renders for the runner's text-only opening call (Scout of
# prompt-arrangement, out/render_arrangements_2b.json, arrangement `text_only`): 93 tokens.
TEXT_ONLY_SHA256 = "3415139c468464294201ba306a80f8e3152c75cdf9746e0dd7b021ea2433a390"
TEXT_ONLY_TOKENS = 93
# Token ids of the turn for IMAGE, equal at the three sizes (out/train_prefix_equals_serving_tokens.json).
IMAGE_TURN_TOKENS = 432
IMAGE_TURN_IDS_SHA256 = "5d49849a034ebfdd1776723fa69ea5e4d5e10100a29164960b35dcd3a1401482"

_cache = {}


def prompt():
    raw = PROMPT_PATH.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == PROMPT_SHA256
    return raw.decode("utf-8").strip()          # infer_runner.py: PROMPT = _f.read().strip()


def processor():
    if "processor" not in _cache:
        from transformers import AutoProcessor
        _cache["processor"] = AutoProcessor.from_pretrained(MODEL, revision=REVISION)
    return _cache["processor"]


def template(size):
    raw = (TEMPLATES / size / "chat_template.jinja").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == TEMPLATE_SHA256[size]
    return raw.decode("utf-8")


def data_url(path=IMAGE):
    return "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode()


def runner_request(text=None, picture=True):
    """The messages infer_runner.py sends: the text part first, the picture second."""
    parts = [{"type": "text", "text": prompt() if text is None else text}]
    if picture:
        parts.append({"type": "image_url", "image_url": {"url": data_url()}})
    return [{"role": "user", "content": parts}]


def refused(messages):
    try:
        U.from_request(messages)
    except U.Refused:
        return True
    return False


# --------------------------------------------------------------------------- #
# The arrangement
# --------------------------------------------------------------------------- #
def test_the_turn_is_the_picture_then_a_newline_and_the_text():
    picture = object()
    assert U.content("P", picture) == [{"type": "image", "image": picture}, {"type": "text", "text": "\nP"}]
    assert U.message("P", picture) == {"role": "user", "content": U.content("P", picture)}


def test_a_turn_without_a_picture_is_its_text_alone():
    assert U.content("P") == [{"type": "text", "text": "P"}]


def test_the_file_hash_is_the_hash_of_the_file():
    assert U.FILE_SHA256 == hashlib.sha256(Path(U.__file__).read_bytes()).hexdigest()


# --------------------------------------------------------------------------- #
# The rendering, at the three sizes
# --------------------------------------------------------------------------- #
def test_the_rendered_turn_has_the_frozen_hash_at_the_three_sizes():
    for size in ("2B", "9B", "27B"):
        assert U.rendered_sha256(processor(), prompt(), chat_template=template(size)) == U.RENDERED_SHA256, size
    assert U.rendered_sha256(processor(), prompt()) == U.RENDERED_SHA256     # the processor's own template


def test_the_rendered_turn_is_this_string():
    expected = ("<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>\n" + prompt()
                + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert hashlib.sha256(expected.encode()).hexdigest() == U.RENDERED_SHA256


def test_a_text_only_request_is_rendered_as_the_official_template_renders_it():
    for size in ("2B", "9B", "27B"):
        t = template(size)
        assert U.rendered_sha256(processor(), prompt(), picture=False, chat_template=t) == TEXT_ONLY_SHA256, size
        ids = U.render(processor(), prompt(), chat_template=t)["input_ids"][0].tolist()
        official = processor().apply_chat_template(
            [{"role": "user", "content": [{"type": "text", "text": prompt()}]}], chat_template=t,
            add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
            enable_thinking=False)["input_ids"][0].tolist()
        assert ids == official and len(ids) == TEXT_ONLY_TOKENS, size


def test_the_token_ids_of_a_picture_turn_are_the_same_at_the_three_sizes():
    image = U.open_image(IMAGE)
    for size in ("2B", "9B", "27B"):
        ids = U.render(processor(), prompt(), image, chat_template=template(size))["input_ids"][0].tolist()
        assert len(ids) == IMAGE_TURN_TOKENS, size
        assert hashlib.sha256(__import__("json").dumps(ids).encode()).hexdigest() == IMAGE_TURN_IDS_SHA256, size


def test_thinking_is_off_whatever_the_template_default():
    # the 9B and 27B templates open a think block unless the switch is passed
    t = template("27B")
    default = processor().apply_chat_template(
        [U.message(prompt())], chat_template=t, add_generation_prompt=True, tokenize=False)
    assert hashlib.sha256(default.encode()).hexdigest() != TEXT_ONLY_SHA256
    assert U.rendered_sha256(processor(), prompt(), picture=False, chat_template=t) == TEXT_ONLY_SHA256


# --------------------------------------------------------------------------- #
# Requests
# --------------------------------------------------------------------------- #
def test_the_runners_request_gives_the_text_and_the_picture_bytes():
    text, picture = U.from_request(runner_request())
    assert text == prompt() and picture == IMAGE.read_bytes()


def test_the_order_of_the_parts_in_the_request_does_not_matter():
    request = runner_request()
    request[0]["content"].reverse()
    assert U.from_request(request) == U.from_request(runner_request())


def test_a_text_only_request_gives_the_text_unchanged():
    assert U.from_request(runner_request(picture=False)) == (prompt(), None)
    assert U.from_request([{"role": "user", "content": " two  spaces\n"}]) == (" two  spaces\n", None)


def test_the_request_route_and_the_file_route_give_the_same_model_inputs():
    text, picture = U.from_request(runner_request())
    served = U.render(processor(), text, U.open_image(picture))
    trained = U.render(processor(), prompt(), U.open_image(IMAGE))
    assert set(served.keys()) == set(trained.keys())
    for key in served.keys():
        assert served[key].equal(trained[key]), key


def test_other_requests_are_refused():
    picture = {"type": "image_url", "image_url": {"url": data_url()}}
    text = {"type": "text", "text": "P"}
    assert refused([{"role": "user", "content": [text, picture, picture]}])            # two pictures
    assert refused([{"role": "user", "content": [text, text, picture]}])               # two text parts
    assert refused([{"role": "user", "content": [picture]}])                           # no text
    assert refused([{"role": "user", "content": []}])
    assert refused([{"role": "system", "content": "S"}, {"role": "user", "content": [text]}])
    assert refused([{"role": "user", "content": [text]}, {"role": "assistant", "content": "A"}])
    assert refused([{"role": "assistant", "content": [text]}])
    assert refused([])
    assert refused("P")
    assert refused([{"role": "user", "content": [text, {"type": "video_url", "video_url": {"url": "x"}}]}])
    assert refused([{"role": "user", "content": [text, {"type": "input_audio"}]}])
    assert refused([{"role": "user", "content": [text, "a bare string"]}])
    assert refused([{"role": "user", "content": [{"type": "text", "text": 5}]}])
    assert refused([{"role": "user", "content": None}])


def test_a_picture_that_is_not_a_base64_data_url_is_refused():
    text = {"type": "text", "text": "P"}
    for url in ("https://example.org/a.png", "file:///a.png", "data:image/png;base64,not base64!", "", None):
        assert refused([{"role": "user", "content": [text, {"type": "image_url", "image_url": {"url": url}}]}]), url
    assert refused([{"role": "user", "content": [text, {"type": "image_url", "image_url": "data:x"}]}])
    assert refused([{"role": "user", "content": [text, {"type": "image_url"}]}])


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print("ok  ", name)
        except Exception as exc:  # noqa: BLE001 - a test runner reports every failure
            failed += 1
            print("FAIL", name, f"{type(exc).__name__}: {exc}")
    print(f"{len(tests) - failed} of {len(tests)} passed")
    sys.exit(1 if failed else 0)
