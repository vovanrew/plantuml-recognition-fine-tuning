"""The training loss of the study: cross-entropy on the answer positions only, computed in chunks.

decisions/pilot-run/record.md. The logits of the whole sequence are never formed. The hidden
states at the positions that predict the answer go through the output layer `chunk` positions at a
time; each chunk's loss is back-propagated to those hidden states and its logits freed; then the
gradient gathered on the hidden states is back-propagated through the model once. The loss and the
gradient are those of the library's own loss with -100 on the prompt positions
(train/tests/test_loss.py). At a 7,874-token sequence of the 27B, under placeholder adapter
settings, this computation left 29.9 GB of the card free, against 8.0 GB for the library's own
(decisions/gpu-precision/implement.md, check 2).

`backward` returns the sum of the per-token losses of one example and its number of answer
tokens. How examples are weighted inside an optimizer step is the caller's: the gradient
back-propagated is that of `scale` times the sum (for a mean over the answer tokens of one
example, `scale` is 1 / tokens).

Tests: train/tests/test_loss.py
"""
import torch
import torch.nn.functional as F

CHUNK = 512


def core(model):
    """The transformers model underneath a PEFT wrapper, adapters in place."""
    return model.get_base_model() if hasattr(model, "get_base_model") else model


def answer_hidden(model, inputs, n_prompt):
    """Hidden states at the positions that predict the answer tokens, and those tokens.

    No cache is made: a training forward keeps no decoding state."""
    hidden = core(model).model(**inputs, use_cache=False)[0]
    return hidden[0, n_prompt - 1:-1], inputs["input_ids"][0, n_prompt:]


def backward(model, inputs, n_prompt, scale, chunk=CHUNK):
    """Back-propagate `scale` × (summed cross-entropy of the answer tokens) of one example.
    Returns (summed loss, number of answer tokens)."""
    hidden, targets = answer_hidden(model, inputs, n_prompt)
    head = core(model).lm_head
    leaf = hidden.detach().requires_grad_(True)
    n, total = targets.shape[0], 0.0
    for start in range(0, n, chunk):
        part = F.cross_entropy(head(leaf[start:start + chunk]).float(), targets[start:start + chunk],
                               reduction="sum")
        (part * scale).backward()
        total += float(part.detach())
    hidden.backward(leaf.grad)
    return total, n


def summed_loss(model, inputs, n_prompt, chunk=CHUNK):
    """The same sum without a gradient: (summed loss, number of answer tokens)."""
    with torch.no_grad():
        hidden, targets = answer_hidden(model, inputs, n_prompt)
        head = core(model).lm_head
        total = sum(float(F.cross_entropy(head(hidden[s:s + chunk]).float(), targets[s:s + chunk],
                                          reduction="sum"))
                    for s in range(0, targets.shape[0], chunk))
    return total, int(targets.shape[0])
