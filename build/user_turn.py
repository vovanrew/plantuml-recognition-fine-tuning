"""The user turn of the study: the picture, one newline, the prompt text.

decisions/prompt-arrangement/record.md, Decision of 2026-10-11. The training data step and the
server both build the turn with this file, so the two cannot differ. `fine-tune/` is not under
version control; the frozen objects are the sha256 of this file (FILE_SHA256) and the sha256 of
the turn rendered for the benchmark prompt (RENDERED_SHA256), and the server puts both in every
response.

The arrangement is defined for a user turn with one text part and at most one picture. A turn
without a picture is its text alone (the runner's opening call). `from_request` refuses anything
else; the arrangement of a request with several pictures belongs to `controls-definition`.

Rendering uses the model's own chat template with a generation prompt and `enable_thinking=False`.
The 9B and 27B templates think by default, so the switch is set here and not left to the request.

Tests: build/tests/test_user_turn.py
"""
import base64
import binascii
import hashlib
import io

from PIL import Image

# sha256 of the turn rendered for prompts/zero_shot.txt of the instrument repository (sha256
# 097351ff…1552, read with .strip() as infer_runner.py does) and one picture, the picture left
# as the template's single placeholder. Equal at the 2B, 9B and 27B templates.
RENDERED_SHA256 = "44a5cc713a568d729f5839c9f2903ef09327b890b8c9f4b756c765e6585141e8"

with open(__file__, "rb") as _f:
    FILE_SHA256 = hashlib.sha256(_f.read()).hexdigest()


class Refused(ValueError):
    """A request outside the domain the arrangement is defined for."""


def open_image(source):
    """A picture as the processor receives it, from a path or from the bytes of a file."""
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    return Image.open(source).convert("RGB")


def content(text, image=None):
    """Content of the user turn for `apply_chat_template`: picture, newline, text."""
    if image is None:
        return [{"type": "text", "text": text}]
    return [{"type": "image", "image": image}, {"type": "text", "text": "\n" + text}]


def message(text, image=None):
    return {"role": "user", "content": content(text, image)}


def from_request(messages):
    """(text, picture bytes or None) of a chat-completions request; Refused outside the domain.

    Accepted: one message, role `user`, whose content is a string or a list holding exactly one
    `text` part and at most one `image_url` part with a base64 `data:` URL, in either order.
    """
    if not isinstance(messages, list) or len(messages) != 1:
        raise Refused("exactly one message is expected")
    turn = messages[0]
    if not isinstance(turn, dict) or turn.get("role") != "user":
        raise Refused("the message must have the role `user`")
    parts = turn.get("content")
    if isinstance(parts, str):
        parts = [{"type": "text", "text": parts}]
    if not isinstance(parts, list):
        raise Refused("the content must be a string or a list of parts")
    texts, pictures = [], []
    for part in parts:
        kind = part.get("type") if isinstance(part, dict) else None
        if kind == "text" and isinstance(part.get("text"), str):
            texts.append(part["text"])
        elif kind == "image_url":
            pictures.append(_picture_bytes(part))
        else:
            raise Refused(f"unsupported content part {kind!r}")
    if len(texts) != 1:
        raise Refused(f"exactly one text part is expected, got {len(texts)}")
    if len(pictures) > 1:
        raise Refused(f"at most one picture is expected, got {len(pictures)}")
    return texts[0], (pictures[0] if pictures else None)


def _picture_bytes(part):
    image_url = part.get("image_url")
    url = image_url.get("url") if isinstance(image_url, dict) else None
    if not isinstance(url, str) or not url.startswith("data:") or ";base64," not in url:
        raise Refused("the picture must be a base64 `data:` URL")
    try:
        return base64.b64decode(url.split(";base64,", 1)[1], validate=True)
    except (binascii.Error, ValueError) as exc:
        raise Refused(f"the picture is not valid base64: {exc}") from None


def render(processor, text, image=None, chat_template=None):
    """Model inputs of the turn: the model's own template, generation prompt, thinking off."""
    return processor.apply_chat_template(
        [message(text, image)], chat_template=chat_template, add_generation_prompt=True,
        tokenize=True, return_dict=True, return_tensors="pt", enable_thinking=False)


def rendered_sha256(processor, text, picture=True, chat_template=None):
    """sha256 of the rendered turn as a string, the picture left as the template's placeholder.
    For the benchmark prompt with a picture this is RENDERED_SHA256."""
    image = Image.new("RGB", (32, 32)) if picture else None
    rendered = processor.apply_chat_template(
        [message(text, image)], chat_template=chat_template, add_generation_prompt=True,
        tokenize=False, enable_thinking=False)
    return hashlib.sha256(rendered.encode()).hexdigest()
