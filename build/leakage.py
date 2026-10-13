#!/usr/bin/env python3
"""Near-duplicates of the test diagrams: elements D5 and D6 of
decisions/training-pool-rules/record.md.

D5  The text of a diagram is the normalized code of build_test_set.normalized_code without
    the lines that begin with @startuml or @enduml (wrapper_free_text).
D6  A candidate is dropped when, against any test diagram,
      (a) the Jaccard index of their character 5-gram sets is >= 0.7, or
      (b) their edit similarity 2*LCS(a, b) / (|a| + |b|) on characters is >= 0.8, or
      (c) it holds at least 80% of the distinct lines of a test diagram that has at least 10
          distinct lines (lone { and } not counted).
    A 5-gram set of a text shorter than 5 characters is the set holding the text. A pair in
    which one text is empty gets no verdict from (a) and (b) (Implementation note 5).

all_pairs() computes the three quantities for every candidate against every test diagram, with
no shortlist and no bound: the 5-gram and line intersections through a 0/1 matrix of the test
side, the LCS length with the bit-parallel recurrence of Allison and Dix run on all test
diagrams at once (PackedLCS). The three integer matrices are kept, and every verdict is taken
on integers (10*I >= 7*U, 5*LCS >= 2*(|a|+|b|), 5*I >= 4*n).

Usage: called by build/pool.py, stage d6.
"""
import os
import time
from collections import Counter
from multiprocessing import get_context
from pathlib import Path

import numpy as np

import pool as P

SHINGLE = 5
A_NUM, A_DEN = 7, 10         # (a) Jaccard >= 0.7
B_NUM, B_DEN = 4, 5          # (b) edit similarity >= 0.8
C_NUM, C_DEN = 4, 5          # (c) share of the test diagram's lines >= 0.8
C_MIN_LINES = 10             # (c) only against a test diagram with at least this many distinct lines
ARMS = ("a", "b", "c")


# --------------------------------------------------------------------------- #
# D5: the text
# --------------------------------------------------------------------------- #
def source_lines(text):
    """The lines build_test_set.normalized_code keeps: block comments removed, every line
    stripped, blank lines and lines that begin with ' dropped."""
    code = P.test_set_code()
    out = []
    for line in code._BLOCK_COMMENT.sub(" ", text).splitlines():
        s = line.strip()
        if s and not s.startswith("'"):
            out.append(s)
    return out


def is_wrapper(line):
    low = line.lower()
    return low.startswith("@startuml") or low.startswith("@enduml")


def wrapper_free_text(text):
    """D5: the normalized code without the lines that begin with @startuml or @enduml."""
    ws = P.test_set_code()._WS
    return ws.sub(" ", " ".join(l for l in source_lines(text) if not is_wrapper(l))).strip()


def line_set(text):
    """The distinct lines of D6 (c): whitespace collapsed inside each kept line; the wrapper
    lines and lone { and } not counted (lineset() of the Challenger's b15 script)."""
    ws = P.test_set_code()._WS
    out = set()
    for line in source_lines(text):
        s = ws.sub(" ", line)
        if is_wrapper(s) or s in ("{", "}"):
            continue
        out.add(s)
    return out


def read_text(path):
    """A source as build_test_set.normalized_code reads it."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


# --------------------------------------------------------------------------- #
# D6: the three measures on one pair
# --------------------------------------------------------------------------- #
def shingles(s, n=SHINGLE):
    return {s[i:i + n] for i in range(len(s) - n + 1)} if len(s) >= n else {s}


def lcs_length(a, b):
    """Length of the longest common subsequence of two strings (bit-parallel, Allison and Dix)."""
    masks = {}
    for i, ch in enumerate(b):
        masks[ch] = masks.get(ch, 0) | (1 << i)
    full = (1 << len(b)) - 1
    v = full
    for ch in a:
        m = masks.get(ch)
        if m:
            u = v & m
            v = ((v + u) | (v - u)) & full
    return len(b) - bin(v).count("1")


def lcs_length_dp(a, b):
    """The same by the textbook recurrence; the reference the tests compare with."""
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


def arm_a(inter, n_cand, n_test):
    return A_DEN * inter >= A_NUM * (n_cand + n_test - inter)


def arm_b(lcs, len_cand, len_test):
    return B_DEN * 2 * lcs >= B_NUM * (len_cand + len_test)


def arm_c(inter, n_test_lines):
    return n_test_lines >= C_MIN_LINES and C_DEN * inter >= C_NUM * n_test_lines


def compare(cand_source, test_source):
    """The three measures and the verdict of D6 for one pair of sources (file contents).
    A value is None where the pair gets no verdict from that arm."""
    a, b = wrapper_free_text(cand_source), wrapper_free_text(test_source)
    la, lb = line_set(cand_source), line_set(test_source)
    out = {"jaccard_5gram": None, "edit_similarity": None, "lines_held": None, "arms": []}
    if a and b:
        sa, sb = shingles(a), shingles(b)
        inter = len(sa & sb)
        out["jaccard_5gram"] = inter / (len(sa) + len(sb) - inter)
        if arm_a(inter, len(sa), len(sb)):
            out["arms"].append("a")
        lcs = lcs_length(a, b)
        out["edit_similarity"] = 2 * lcs / (len(a) + len(b))
        if arm_b(lcs, len(a), len(b)):
            out["arms"].append("b")
    if len(lb) >= C_MIN_LINES:
        inter = len(la & lb)
        out["lines_held"] = inter / len(lb)
        if arm_c(inter, len(lb)):
            out["arms"].append("c")
    out["dropped"] = bool(out["arms"])
    return out


# --------------------------------------------------------------------------- #
# LCS of one text against many texts at once
# --------------------------------------------------------------------------- #
class PackedLCS:
    """LCS length of a text against every one of a fixed list of texts in one pass.

    The texts are laid side by side in one long integer, each followed by one guard bit. The
    recurrence V = (V + (V & M)) | (V - (V & M)) is the one of lcs_length; a carry of the
    addition can leave a text only into its guard bit, which is cleared after every step, so
    the texts do not touch each other. Texts are ordered so that those holding rare characters
    lie in the low bits, which keeps the masks of rare characters short."""

    def __init__(self, texts):
        n = len(texts)
        freq = Counter(ch for s in texts for ch in set(s))
        rarity = [min((freq[ch] for ch in s), default=n + 1) for s in texts]
        self.order = np.array(sorted(range(n), key=lambda j: (rarity[j], j)), dtype=np.int64)
        self.lengths = np.array([len(texts[j]) for j in self.order], dtype=np.int64)
        self.offsets = np.concatenate([[0], np.cumsum(self.lengths + 1)[:-1]]).astype(np.int64)
        self.nbits = int((self.lengths + 1).sum())
        self.nbytes = (self.nbits + 7) // 8
        codes = np.full(self.nbytes * 8, -1, dtype=np.int64)        # -1: guard bit or padding
        for off, j in zip(self.offsets, self.order):
            s = texts[j]
            if s:
                codes[off:off + len(s)] = np.frombuffer(s.encode("utf-32-le", "surrogatepass"), dtype="<u4")
        self.full = self._to_int(codes >= 0)
        self.masks = {chr(c): self._to_int(codes == c) for c in np.unique(codes[codes >= 0])}

    @staticmethod
    def _to_int(bits):
        return int.from_bytes(np.packbits(bits, bitorder="little").tobytes(), "little")

    def lengths_against(self, text):
        """LCS length of `text` with every text of the list, in the order the list was given."""
        v, full, masks = self.full, self.full, self.masks
        for ch in text:
            m = masks.get(ch)
            if m is not None:
                u = v & m
                v = ((v + u) | (v - u)) & full
        bits = np.unpackbits(np.frombuffer(v.to_bytes(self.nbytes, "little"), dtype=np.uint8), bitorder="little")
        ones = np.add.reduceat(bits, self.offsets) if len(self.offsets) else np.zeros(0, dtype=np.int64)
        out = np.empty(len(self.order), dtype=np.int64)
        out[self.order] = self.lengths - ones
        return out


# --------------------------------------------------------------------------- #
# Set intersections of one item set against many
# --------------------------------------------------------------------------- #
class SetIndex:
    """|A ∩ B_j| for a set A against every set B_j of a fixed list: a 0/1 matrix item x list
    position; only an item that occurs in some B_j can be in an intersection."""

    def __init__(self, sets):
        self.vocab = {}
        for s in sets:
            for x in s:
                if x not in self.vocab:
                    self.vocab[x] = len(self.vocab)
        self.matrix = np.zeros((max(len(self.vocab), 1), len(sets)), dtype=np.uint8)
        for j, s in enumerate(sets):
            if s:
                self.matrix[[self.vocab[x] for x in s], j] = 1
        self.sizes = np.array([len(s) for s in sets], dtype=np.int64)

    def intersections(self, items):
        ids = [self.vocab[x] for x in items if x in self.vocab]
        if not ids:
            return np.zeros(self.matrix.shape[1], dtype=np.int64)
        return self.matrix[ids].sum(axis=0, dtype=np.int64)


# --------------------------------------------------------------------------- #
# All pairs
# --------------------------------------------------------------------------- #
class TestSide:
    def __init__(self, sources):
        self.texts = [wrapper_free_text(s) for s in sources]
        self.lines = [line_set(s) for s in sources]
        self.len = np.array([len(t) for t in self.texts], dtype=np.int64)
        self.grams = SetIndex([shingles(t) for t in self.texts])
        self.line_index = SetIndex(self.lines)
        self.lcs = PackedLCS(self.texts)


def candidate_row(source, side):
    """For one candidate source: (text length, 5-gram set size, line set size, 5-gram
    intersections, LCS lengths, line intersections), each of the last three against every test
    diagram."""
    text = wrapper_free_text(source)
    grams = shingles(text)
    lines = line_set(source)
    return (len(text), len(grams), len(lines), side.grams.intersections(grams),
            side.lcs.lengths_against(text), side.line_index.intersections(lines))


_G = {}


def _work(span):
    lo, hi = span
    side, keys, src = _G["side"], _G["keys"], _G["src"]
    nt = len(side.texts)
    sizes = np.zeros((hi - lo, 3), dtype=np.int64)
    mats = np.zeros((3, hi - lo, nt), dtype=np.int32)
    for i in range(lo, hi):
        ln, ng, nl, i5, lcs, il = candidate_row(read_text(os.path.join(src, keys[i])), side)
        sizes[i - lo] = (ln, ng, nl)
        mats[0, i - lo], mats[1, i - lo], mats[2, i - lo] = i5, lcs, il
    return lo, hi, sizes, mats


def measure(cand_keys, cand_dir, test_sources, workers=8, chunk=64, progress=None):
    """The three integer matrices candidates x test diagrams, and the sizes on both sides."""
    side = TestSide(test_sources)
    n, nt = len(cand_keys), len(test_sources)
    sizes = np.zeros((n, 3), dtype=np.int64)
    mats = np.zeros((3, n, nt), dtype=np.int32)
    _G.update(side=side, keys=list(cand_keys), src=str(cand_dir))
    spans = [(i, min(i + chunk, n)) for i in range(0, n, chunk)]
    t0, done = time.time(), 0
    if workers <= 1:
        results = map(_work, spans)
    else:
        pool = get_context("fork").Pool(workers)
        results = pool.imap_unordered(_work, spans)
    for lo, hi, s, m in results:
        sizes[lo:hi] = s
        mats[:, lo:hi] = m
        done += hi - lo
        if progress and (done % 5000 < chunk or done == n):
            progress(f"  D6: {done:,} of {n:,} candidates, {time.time() - t0:.0f} s")
    if workers > 1:
        pool.close()
        pool.join()
    return {"cand_len": sizes[:, 0], "cand_grams": sizes[:, 1], "cand_lines": sizes[:, 2],
            "test_len": side.len, "test_grams": side.grams.sizes, "test_lines": side.line_index.sizes,
            "inter_grams": mats[0], "lcs": mats[1], "inter_lines": mats[2]}


def verdicts(m, rows=None):
    """Boolean matrices of the three arms and float matrices of the three values, for the rows
    given (all by default). A value is 0 where the pair gets no verdict from the arm."""
    sl = slice(None) if rows is None else rows
    cl, cg = m["cand_len"][sl, None], m["cand_grams"][sl, None]
    tl, tg, tn = m["test_len"][None, :], m["test_grams"][None, :], m["test_lines"][None, :]
    i5 = m["inter_grams"][sl].astype(np.int64)
    lcs = m["lcs"][sl].astype(np.int64)
    il = m["inter_lines"][sl].astype(np.int64)
    both = (cl > 0) & (tl > 0)
    union = cg + tg - i5
    arm = {"a": both & (A_DEN * i5 >= A_NUM * union),
           "b": both & (B_DEN * 2 * lcs >= B_NUM * (cl + tl)),
           "c": (tn >= C_MIN_LINES) & (C_DEN * il >= C_NUM * tn)}
    with np.errstate(divide="ignore", invalid="ignore"):
        val = {"a": np.where(both, i5 / np.maximum(union, 1), 0.0),
               "b": np.where(both, 2 * lcs / np.maximum(cl + tl, 1), 0.0),
               "c": np.where(tn >= C_MIN_LINES, il / np.maximum(tn, 1), 0.0)}
    return arm, val


def all_pairs(cand_keys, cand_dir, test_keys, test_dir, work, workers=8, progress=P.log):
    """D6 over the candidates given. Returns a dict with, per candidate, the largest value of
    each measure and the test diagram that gives it, the arms that fire, and the set of
    candidates removed. The integer matrices are stored under `work` and reused when the
    candidate and test lists are the same."""
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    cand_keys, test_keys = list(cand_keys), list(test_keys)
    stamp = P.hashlib.sha256(("\n".join(cand_keys) + "\n--\n" + "\n".join(test_keys)).encode()).hexdigest()
    cache = work / "matrices.npz"
    m = None
    if cache.is_file():
        z = np.load(cache, allow_pickle=False)
        if str(z["stamp"]) == stamp:
            m = {k: z[k] for k in z.files if k != "stamp"}
    if m is None:
        sources = [read_text(os.path.join(test_dir, k)) for k in test_keys]
        m = measure(cand_keys, cand_dir, sources, workers=workers, progress=progress)
        np.savez(cache, stamp=stamp, **m)
    n = len(cand_keys)
    best = {a: np.zeros(n) for a in ARMS}
    arg = {a: np.zeros(n, dtype=np.int64) for a in ARMS}
    hits = {a: np.zeros(n, dtype=np.int64) for a in ARMS}
    per_test = {a: np.zeros(len(test_keys), dtype=np.int64) for a in ARMS + ("any",)}
    for lo in range(0, n, 4096):
        rows = slice(lo, min(lo + 4096, n))
        arm, val = verdicts(m, rows)
        for a in ARMS:
            arg[a][rows] = val[a].argmax(axis=1)
            best[a][rows] = val[a].max(axis=1)
            hits[a][rows] = arm[a].sum(axis=1)
            per_test[a] += arm[a].sum(axis=0)
        per_test["any"] += (arm["a"] | arm["b"] | arm["c"]).sum(axis=0)
    fired = {a: hits[a] > 0 for a in ARMS}
    removed_mask = fired["a"] | fired["b"] | fired["c"]
    removed = {}
    for i in np.nonzero(removed_mask)[0]:
        removed[cand_keys[i]] = {
            "arms": [a for a in ARMS if fired[a][i]],
            **{f"{a}_value": float(best[a][i]) for a in ARMS},
            **{f"{a}_test_key": test_keys[int(arg[a][i])] for a in ARMS},
            **{f"{a}_test_diagrams_at_or_above": int(hits[a][i]) for a in ARMS}}
    test_side = {a: int((per_test[a] > 0).sum()) for a in per_test}
    return {"keys": cand_keys, "test_keys": test_keys, "best": best, "arg": arg, "hits": hits,
            "fired": fired, "removed_mask": removed_mask, "removed": removed, "matrices": m,
            "test_side": test_side, "per_test": per_test}
