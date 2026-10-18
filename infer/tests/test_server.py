"""Tests for infer/server.py: what the server refuses, and, with weights, what it answers.

Run from fine-tune/:
  .venv/bin/python infer/tests/test_server.py                 without weights
  HF_DEACTIVATE_ASYNC_LOAD=1 .venv/bin/python infer/tests/test_server.py --live [--adapter <dir>]
      loads Qwen3.5-2B from .local/hf on this machine's device and serves real requests; with
      --adapter also serves the same requests with that adapter attached.
"""
import argparse
import base64
import hashlib
import json
import os
import sys
import threading
import time
import urllib.request
from pathlib import Path

FT = Path(__file__).resolve().parents[2]
UNI = FT.parent
os.environ.setdefault("HF_HOME", str(FT / ".local/hf"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import server as S  # noqa: E402

MODEL, REVISION = "Qwen/Qwen3.5-2B", "15852e8c16360a2fea060d615a32b45270f8a8fc"
PROMPT = (UNI / "plantuml-recognition-evaluation/prompts/zero_shot.txt").read_text(encoding="utf-8").strip()
# A test diagram whose bf16 answer in the precision test of gpu-precision stops after 38 tokens;
# that test ran on this Mac with these libraries and this arrangement, so its stored answer is
# the reference the study's server must reproduce. The stored response is copied unchanged into
# fixtures/ (from decisions/gpu-precision/precision_test/runs/bf16/, 2026-10-17).
KEY = "46701d87f11434bfe1cae06f2715ae74a91e0fcd"
IMAGE = UNI / "plantuml-recognition-evaluation/data/puml_images_1568" / (KEY + ".png")
REFERENCE = Path(__file__).resolve().parent / "fixtures/precision_test_bf16_46701d87.json"
TEXT_ONLY_TOKENS = 93                                # the runner's text-only baseline


def body(picture=True, **changes):
    """The request infer_runner.py sends, with fields changed or (value None) removed."""
    parts = [{"type": "text", "text": PROMPT}]
    if picture:
        url = "data:image/png;base64," + base64.b64encode(IMAGE.read_bytes()).decode()
        parts.append({"type": "image_url", "image_url": {"url": url}})
    request = {"model": MODEL, "messages": [{"role": "user", "content": parts}], "max_tokens": 16,
               "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    request.update(changes)
    return {k: v for k, v in request.items() if v is not None}


def refusal(request):
    try:
        S.complete(request)
    except S.Refused as exc:
        return str(exc)
    return None


def stub(served_name=MODEL, timing=False):
    """State enough for the checks that come before the model is used."""
    S.STATE.clear()
    S.STATE.update(served_name=served_name, timing=timing)


def args(**changes):
    values = dict(model=MODEL, revision=REVISION, adapter=None, served_name=None, device=None, timing=False,
                  recurrent_layers="reference")       # the only implementation this Mac can run
    values.update(changes)
    return argparse.Namespace(**values)


# --------------------------------------------------------------------------- #
# Without weights: every refusal happens before the model is touched
# --------------------------------------------------------------------------- #
def test_the_stop_tokens_are_pinned():
    assert S.EOS == [248046, 248044]          # <|im_end|>, <|endoftext|>


def test_another_model_name_is_refused():
    stub()
    assert "serves" in refusal(body(model="Qwen/Qwen3.5-9B"))
    assert "serves" in refusal(body(model=None))
    stub(served_name="Qwen3.5-2B-ft")           # an adapter server refuses the base model's name
    assert "serves" in refusal(body())


def test_a_temperature_other_than_zero_is_refused():
    stub()
    assert "temperature" in refusal(body(temperature=0.7))


def test_thinking_switched_on_is_refused():
    stub()
    assert "chat_template_kwargs" in refusal(body(chat_template_kwargs={"enable_thinking": True}))
    assert "chat_template_kwargs" in refusal(body(chat_template_kwargs={}))


def test_fields_the_server_does_not_implement_are_refused():
    stub()
    for extra in ({"stream": True}, {"top_p": 0.9}, {"stop": ["@enduml"]}, {"n": 2}, {"seed": 1}, {"tools": []}):
        assert "does not implement" in refusal(body(**extra)), extra


def test_the_output_limit_must_be_a_positive_integer():
    stub()
    for value in (0, -1, "16", 1.5, True):
        assert "max_tokens" in refusal(body(max_tokens=value)), value
    assert "max_tokens" in refusal(body(max_tokens=None))


def test_a_forced_answer_length_is_refused_outside_timing():
    stub()
    assert "--timing" in refusal(body(min_tokens=8))
    stub(timing=True)                           # accepted there: the refusal that follows is about the turn
    assert "exactly one message" in refusal(body(min_tokens=8, messages=[]))
    assert "min_tokens" in refusal(body(min_tokens=17))


def test_a_turn_outside_the_arrangement_is_refused():
    stub()
    request = body()
    request["messages"][0]["content"].append(request["messages"][0]["content"][1])
    assert "at most one picture" in refusal(request)
    assert "role" in refusal(body(messages=[{"role": "system", "content": "S"}]))


def test_an_adapter_needs_its_own_served_name():
    for bad in (args(adapter="x"), args(adapter="x", served_name=MODEL)):
        try:
            S.start(bad)
        except SystemExit as exc:
            assert "--served-name" in str(exc)
        else:
            raise AssertionError("started")


def test_the_revision_must_be_a_full_commit():
    for bad in ("main", REVISION[:12], REVISION.upper()):
        try:
            S.start(args(revision=bad))
        except SystemExit as exc:
            assert "commit" in str(exc)
        else:
            raise AssertionError("started")


def test_one_request_is_answered_at_a_time():
    active, most = [0], [0]
    lock = threading.Lock()
    real = S.complete

    def slow(body):
        with lock:
            active[0] += 1
            most[0] = max(most[0], active[0])
        time.sleep(0.3)
        with lock:
            active[0] -= 1
        return {"ok": True}

    S.complete = slow
    httpd = S.make_http_server("127.0.0.1", 0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{httpd.server_address[1]}/v1/chat/completions"
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        def post():
            opener.open(urllib.request.Request(url, data=b"{}", headers={"Content-Type": "application/json"}),
                        timeout=10).read()
        clients = [threading.Thread(target=post) for _ in range(3)]
        t0 = time.time()
        for c in clients:
            c.start()
        for c in clients:
            c.join()
        assert most[0] == 1 and time.time() - t0 >= 0.9, (most[0], time.time() - t0)
    finally:
        S.complete = real
        httpd.shutdown()
        httpd.server_close()


def test_the_attachment_is_read_from_the_model():
    import torch.nn as nn
    from peft import LoraConfig, get_peft_model

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(4, 4)

        def forward(self, x):
            return self.proj(x)

    assert S.adapter_state(Tiny())["attached"] == "absent"
    model = get_peft_model(Tiny(), LoraConfig(r=2, target_modules=["proj"]))
    assert S.adapter_state(model) == {"lora_layers": 1, "merged_layers": 0, "attached": "not merged"}
    model.merge_adapter()
    assert S.adapter_state(model) == {"lora_layers": 1, "merged_layers": 1, "attached": "merged"}


def test_the_reference_path_is_refused_on_a_card():
    try:
        S.start(args(device="cuda"))
    except SystemExit as exc:
        assert "diagnostic" in str(exc) or "differ from requirements-card.txt" in str(exc)
    else:
        raise AssertionError("started")


def test_the_study_s_recurrent_layers_are_refused_where_they_cannot_run():
    # this Mac: no CUDA, no flash-linear-attention; refused before any weights load
    try:
        S.start(args(recurrent_layers="flash-linear-attention"))
    except SystemExit as exc:
        assert "flash-linear-attention" in str(exc)
    else:
        raise AssertionError("started")


# --------------------------------------------------------------------------- #
# With weights (--live)
# --------------------------------------------------------------------------- #
def live(adapter):
    S.STATE.clear()
    S.start(args())
    text_only = S.complete(body(picture=False, max_tokens=4))
    assert text_only["usage"]["prompt_tokens"] == TEXT_ONLY_TOKENS
    reference = json.loads(REFERENCE.read_text())
    base = S.complete(body(max_tokens=128))
    served = base["x_served_by"]
    assert base["usage"] == reference["usage"], (base["usage"], reference["usage"])
    assert base["choices"][0]["message"]["content"] == reference["choices"][0]["message"]["content"]
    assert served["rendered_sha256"] == S.user_turn.RENDERED_SHA256 == served["user_turn"]["rendered_sha256_frozen"]
    assert served["user_turn"]["file_sha256"] == S.user_turn.FILE_SHA256
    assert served["revision"] == REVISION and served["dtype"] == "bfloat16" and served["adapter"] is None
    assert served["eos_token_id"] == S.EOS and served["versions"]["transformers"] and served["device"]
    assert served["recurrent_layers"]["chosen"] == served["recurrent_layers"]["active"] == "reference"
    assert served["environment"]["checked"] == "core" and served["environment"]["versions"]["torch"] == "2.14.1"
    for rel in S.SERVING_FILES:
        assert served["files"][rel] == hashlib.sha256((FT / rel).read_bytes()).hexdigest(), rel
    assert served["study_commit"]["commit"]
    assert served["generate_seconds"] > 0 and served["first_token_seconds"] > 0
    # the answer stops at <|im_end|>, and nothing of the stop token reaches the text
    assert base["choices"][0]["finish_reason"] == "stop", base["choices"][0]
    assert base["choices"][0]["message"]["content"].rstrip().endswith("@enduml")
    assert "<|" not in base["choices"][0]["message"]["content"]
    assert S.complete(body(max_tokens=128))["choices"][0]["message"] == base["choices"][0]["message"]   # deterministic
    assert S.complete(body(max_tokens=5))["choices"][0]["finish_reason"] == "length"
    print("ok   live: base model")
    if adapter is None:
        return
    S.STATE.clear()
    S.start(args(adapter=adapter, served_name="Qwen3.5-2B-ft"))
    assert "serves" in refusal(body(max_tokens=128))                     # the base name is refused
    tuned = S.complete(body(max_tokens=128, model="Qwen3.5-2B-ft"))
    identity = tuned["x_served_by"]["adapter"]
    for name in ("adapter_model.safetensors", "adapter_config.json"):
        assert identity[name] == hashlib.sha256((Path(adapter) / name).read_bytes()).hexdigest()
    assert identity["attached"] == "not merged" and identity["merged_layers"] == 0 and identity["lora_layers"] == 186
    assert tuned["usage"]["prompt_tokens"] == reference["usage"]["prompt_tokens"]
    assert tuned["choices"][0]["message"]["content"] != base["choices"][0]["message"]["content"], \
        "the adapter does not change the answer"
    print("ok   live: adapter attached, the answer changes")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--adapter", default=None)
    cli = ap.parse_args()
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print("ok  ", name)
        except Exception as exc:  # noqa: BLE001 - a test runner reports every failure
            failed += 1
            print("FAIL", name, f"{type(exc).__name__}: {exc}")
    if cli.live:
        try:
            live(cli.adapter)
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print("FAIL live", f"{type(exc).__name__}: {exc}")
    print(f"{len(tests) + int(cli.live) - failed} of {len(tests) + int(cli.live)} passed")
    sys.exit(1 if failed else 0)
