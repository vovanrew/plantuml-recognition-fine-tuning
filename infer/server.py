"""The study's server: a thin OpenAI-compatible endpoint over transformers + PEFT.

decisions/gpu-precision/record.md, Decision Part 2, "Serving software". One model in bf16, one
request at a time. The zero-shot condition is the base model; the fine-tuned condition is the
same base with the adapter attached, not merged (`--adapter`).

It answers the request `infer_runner.py` sends (`POST /v1/chat/completions`: one user message
with a text part and at most one picture, `temperature` 0, `max_tokens`, top-level
`chat_template_kwargs`). The user turn is built by build/user_turn.py (picture, newline, text;
thinking off). Decoding is greedy and stops at `eos_token_id` [248046, 248044] at every size.

Refused with HTTP 400, never silently adjusted: a model name other than the served one, a
temperature other than 0, `chat_template_kwargs` other than `{"enable_thinking": false}`, a
request field this server does not implement, and a user turn outside the domain of
build/user_turn.py. With an adapter the served name must differ from the base model's name, so a
base server left on a fine-tuned run's port refuses the run.

Every response carries `x_served_by`: model revision, adapter hashes, stop tokens, device and GPU
model, library versions, the hash of build/user_turn.py and of the rendered turn, and the
generation time. `GET /` returns the same identity without the per-request fields.

`--timing` accepts `min_tokens` (an answer forced to a length, for measuring the time of an
answer at the output cap). It is shown in the identity and is not used for scored runs.

First version, 2026-10-12, for the first session on the rented card. Not tagged; the manifest
written beside each run is not built yet.

Usage:
  python infer/server.py --model Qwen/Qwen3.5-2B --revision <commit> --port 8000
  python infer/server.py --model Qwen/Qwen3.5-2B --revision <commit> --adapter <dir> \
      --served-name Qwen3.5-2B-ft --port 8001
On Apple MPS set HF_DEACTIVATE_ASYNC_LOAD=1 (the 4-thread weight copy crashed or hung there).
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import sys
import time
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.generation.streamers import BaseStreamer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "build"))
import user_turn  # noqa: E402

# <|im_end|>, <|endoftext|>: the 27B's generation_config.json. The 2B and 9B ship none and would
# stop at <|endoftext|> only (challenge.md, CHANGES 2).
EOS = [248046, 248044]
REQUEST_FIELDS = {"model", "messages", "max_tokens", "max_completion_tokens", "temperature",
                  "chat_template_kwargs", "min_tokens"}
STATE = {}


class Refused(ValueError):
    """A request this server does not serve; answered with HTTP 400."""


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def version(distribution):
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def environment(device):
    """Where and with what an answer was produced."""
    return {
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "cuda": torch.version.cuda, "transformers": version("transformers"),
                     "peft": version("peft"), "tokenizers": version("tokenizers"),
                     "pillow": version("pillow")},
        # Which implementation of the recurrent layers is active: with neither package the
        # library's reference PyTorch path runs (transformers integrations/hub_kernels.py).
        "optional_kernels": {"flash-linear-attention": version("flash-linear-attention"),
                             "causal-conv1d": version("causal-conv1d"),
                             "kernels": version("kernels")},
    }


def adapter_identity(adapter):
    if adapter is None:
        return None
    adapter = Path(adapter)
    return {"path": str(adapter), "attached": "not merged",
            "adapter_model.safetensors": sha256_file(adapter / "adapter_model.safetensors"),
            "adapter_config.json": sha256_file(adapter / "adapter_config.json")}


def load_model(model_id, revision, device, adapter=None):
    """The base model in bf16 on one device, with the adapter attached when one is given."""
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, revision=revision, device_map={"": device}, dtype=torch.bfloat16)
    if adapter is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter))
    return model.eval()


class FirstTokenClock(BaseStreamer):
    """Time from the start of `generate` to the first new token."""

    def __init__(self):
        self.start, self.calls, self.first = time.time(), 0, None

    def put(self, value):
        self.calls += 1                       # the first call hands over the prompt
        if self.calls == 2:
            self.first = time.time() - self.start

    def end(self):
        pass


def generate(model, inputs, max_tokens, min_tokens=0):
    """Greedy decoding of one prompt: (new token ids, seconds, seconds to the first token)."""
    n_prompt = int(inputs["input_ids"].shape[1])
    clock = FirstTokenClock()
    extra = {"min_new_tokens": min_tokens} if min_tokens else {}
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False,
                             eos_token_id=EOS, temperature=None, top_p=None, top_k=None,
                             streamer=clock, **extra)
    seconds = time.time() - clock.start
    return out[0, n_prompt:].tolist(), seconds, clock.first


def complete(body):
    unknown = sorted(set(body) - REQUEST_FIELDS)
    if unknown:
        raise Refused(f"request fields this server does not implement: {unknown}")
    if body.get("model") != STATE["served_name"]:
        raise Refused(f"this server serves {STATE['served_name']!r}, not {body.get('model')!r}")
    if body.get("temperature", 0) != 0:
        raise Refused("this server decodes greedily; temperature must be 0")
    if body.get("chat_template_kwargs", {"enable_thinking": False}) != {"enable_thinking": False}:
        raise Refused('chat_template_kwargs must be {"enable_thinking": false} or absent')
    max_tokens = body.get("max_tokens", body.get("max_completion_tokens"))
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 1:
        raise Refused("max_tokens must be a positive integer")
    min_tokens = body.get("min_tokens", 0)
    if min_tokens and not STATE["timing"]:
        raise Refused("min_tokens is accepted only by a server started with --timing")
    if not isinstance(min_tokens, int) or isinstance(min_tokens, bool) or not 0 <= min_tokens <= max_tokens:
        raise Refused("min_tokens must be an integer between 0 and max_tokens")
    try:
        text, picture = user_turn.from_request(body.get("messages"))
    except user_turn.Refused as exc:
        raise Refused(str(exc)) from None
    processor, device = STATE["processor"], STATE["identity"]["device"]
    image = user_turn.open_image(picture) if picture is not None else None
    inputs = user_turn.render(processor, text, image).to(device)
    n_prompt = int(inputs["input_ids"].shape[1])
    new_ids, seconds, first = generate(STATE["model"], inputs, max_tokens, min_tokens)
    del inputs
    if device == "mps":
        torch.mps.empty_cache()
    STATE["served"] += 1
    rendered = user_turn.rendered_sha256(processor, text, picture=picture is not None)
    return {
        "id": f"chatcmpl-{STATE['served']}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": STATE["served_name"],
        "choices": [{"index": 0,
                     "message": {"role": "assistant",
                                 "content": processor.decode(new_ids, skip_special_tokens=True)},
                     "finish_reason": "stop" if new_ids and new_ids[-1] in EOS else "length"}],
        "usage": {"prompt_tokens": n_prompt, "completion_tokens": len(new_ids),
                  "total_tokens": n_prompt + len(new_ids)},
        "x_served_by": {**STATE["identity"], "rendered_sha256": rendered,
                        "min_tokens": min_tokens, "generate_seconds": round(seconds, 3),
                        "first_token_seconds": None if first is None else round(first, 3)},
    }


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, payload):
        data = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):  # noqa: N802 - http.server API
        self._send(200, STATE["identity"])

    def do_POST(self):  # noqa: N802 - http.server API
        if not self.path.rstrip("/").endswith("/chat/completions"):
            self._send(404, {"error": f"no route {self.path}"})
            return
        try:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if not isinstance(body, dict):
                raise Refused("the request body must be a JSON object")
            payload = complete(body)
        except (Refused, json.JSONDecodeError) as exc:
            self._send(400, {"error": str(exc)})
            return
        except Exception as exc:  # noqa: BLE001 - reported to the client and the log
            traceback.print_exc()
            self._send(500, {"error": f"{type(exc).__name__}: {exc}"})
            return
        self._send(200, payload)

    def log_message(self, fmt, *args):  # one line per request on stderr is enough
        sys.stderr.write("%s %s\n" % (time.strftime("%H:%M:%S"), fmt % args))


def start(args):
    """Load everything and fill STATE; raises SystemExit on a configuration that must not run."""
    served_name = args.served_name or args.model
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        raise SystemExit("--revision must be a full commit hash, not a branch or a tag")
    if args.adapter and served_name == args.model:
        raise SystemExit("with --adapter, --served-name is required and must differ from --model")
    device = args.device or pick_device()
    processor = AutoProcessor.from_pretrained(args.model, revision=args.revision)
    model = load_model(args.model, args.revision, device, args.adapter)
    STATE.update(processor=processor, model=model, served_name=served_name, served=0,
                 timing=args.timing)
    STATE["identity"] = {
        "served_name": served_name, "model": args.model, "revision": args.revision,
        "dtype": str(next(model.parameters()).dtype).replace("torch.", ""),
        "adapter": adapter_identity(args.adapter), "decoding": "greedy", "eos_token_id": EOS,
        "chat_template_kwargs": {"enable_thinking": False},
        "user_turn": {"file_sha256": user_turn.FILE_SHA256,
                      "rendered_sha256_frozen": user_turn.RENDERED_SHA256},
        "timing": args.timing, **environment(device),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", required=True, help="the commit of the model repository, in full")
    ap.add_argument("--adapter", default=None, help="directory of a PEFT adapter; attached, not merged")
    ap.add_argument("--served-name", default=None, help="the name requests must ask for (default: --model)")
    ap.add_argument("--device", default=None, choices=["cuda", "mps", "cpu"])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--timing", action="store_true", help="accept min_tokens; not for scored runs")
    args = ap.parse_args()
    start(args)
    print("serving", json.dumps(STATE["identity"]), flush=True)
    HTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
