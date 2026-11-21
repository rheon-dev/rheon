"""Task 507: half-space trees (Tan, Ting and Liu 2011) in the form river builds
them, ported to flat arrays, with the construction's draws taken from a
supplied uniform source.

**Construction** (river's ``make_padded_tree``, pre-order: a node, then its
left subtree, then its right), two uniforms per internal node::

    feature  q = bisect_right(cumsum(b_i - a_i), u1 * total)   (wider ranges likelier)
    split    lo = a_q + 0.15 (b_q - a_q),  hi = b_q - 0.15 (b_q - a_q)
             s  = lo + (hi - lo) * u2
    left child works on [a_q, s), right child on [s, b_q)

Node ``i``'s children are ``2i+1`` and ``2i+2``; a tree of height ``h`` has
``2^(h+1) - 1`` nodes, ``2^h - 1`` of them internal.

**Learning** a row of weight ``w > 0``: along the row's path (``x_q < s`` goes
left, else right; a NaN goes right) every node's latest mass ``l += w``. After
``window_size`` learned rows the window turns: ``r = l, l = 0`` at every node.

**Score** (before the row is learned; 0 during the first window)::

    per tree, from the root at depth 0:  S += r_node * 2^depth, stop after a node
    with r_node < 0.1 * window_size; at the leaf S += r_leaf * 2^h
    score = 1 - (sum over trees of S) / (n_trees * window_size * (2^(h+1) - 1))

Weights: river counts rows; this port adds ``w`` (unit weights give river's
integers exactly). A zero-weight row learns nothing and does not count toward
the window: rule 9 (``n`` is the learned-row count).
"""

from __future__ import annotations

import bisect
import random
from collections.abc import Callable
from itertools import accumulate

import numpy as np

import philox as P

PADDING = 0.15
LANE_BUILD = 2


class Trees:
    def __init__(self, limits: list[tuple[float, float]], n_trees: int, height: int, window_size: int,
                 uniform: Callable[[], float]) -> None:
        self.d = len(limits)
        self.n_trees, self.h, self.window = n_trees, height, window_size
        n_int = 2**height - 1
        n_all = 2 ** (height + 1) - 1
        self.feature = np.zeros((n_trees, n_int), dtype=np.int32)
        self.split = np.zeros((n_trees, n_int))
        self.l = np.zeros((n_trees, n_all))
        self.r = np.zeros((n_trees, n_all))
        self.counter = 0
        self.first_window = True
        self.n = 0
        for t in range(n_trees):
            self._build(t, 0, list(limits), height, uniform)

    def _build(self, t: int, i: int, lim: list[tuple[float, float]], height: int, uniform) -> None:
        if height == 0:
            return
        cum = list(accumulate(b - a for a, b in lim))
        total = cum[-1] + 0.0
        q = bisect.bisect(cum, uniform() * total, 0, self.d - 1)
        a, b = lim[q]
        lo, hi = a + PADDING * (b - a), b - PADDING * (b - a)
        s = lo + (hi - lo) * uniform()
        self.feature[t, i], self.split[t, i] = q, s
        keep = lim[q]
        lim[q] = (keep[0], s)
        self._build(t, 2 * i + 1, lim, height - 1, uniform)
        lim[q] = (s, keep[1])
        self._build(t, 2 * i + 2, lim, height - 1, uniform)
        lim[q] = keep

    def _child(self, t: int, i: int, x: np.ndarray) -> int:
        return 2 * i + 1 if x[self.feature[t, i]] < self.split[t, i] else 2 * i + 2

    def learn(self, x: np.ndarray, w: float = 1.0) -> None:
        if w <= 0.0:
            return
        for t in range(self.n_trees):
            i = 0
            for _ in range(self.h):
                self.l[t, i] += w
                i = self._child(t, i, x)
            self.l[t, i] += w
        self.n += 1
        self.counter += 1
        if self.counter == self.window:
            self.r[:] = self.l
            self.l[:] = 0.0
            self.first_window = False
            self.counter = 0

    def score(self, x: np.ndarray) -> float:
        if self.first_window:
            return 0.0
        limit = 0.1 * self.window
        total = 0.0
        for t in range(self.n_trees):
            i, pw, s = 0, 1, 0.0
            stopped = False
            for _ in range(self.h):
                r = self.r[t, i]
                s += r * pw
                if r < limit:
                    stopped = True
                    break
                i = self._child(t, i, x)
                pw <<= 1
            if not stopped:
                s += self.r[t, i] * pw
            total += s
        return 1 - total / (self.n_trees * self.window * (2 ** (self.h + 1) - 1))

    def state_bytes(self, keep_trees: bool) -> int:
        n_int = 2**self.h - 1
        n_all = 2 ** (self.h + 1) - 1
        masses = self.n_trees * n_all * 2 * 8
        trees = self.n_trees * n_int * (4 + 8) if keep_trees else 0
        return masses + trees + 8 + 8 + 1  # + the window counter, n, the first-window flag


def state_bytes_formula(n_trees: int, h: int, keep_trees: bool) -> int:
    n_int = 2**h - 1
    n_all = 2 ** (h + 1) - 1
    return n_trees * n_all * 16 + (n_trees * n_int * 12 if keep_trees else 0) + 17


# --- uniform sources -----------------------------------------------------------------


class Recorder(random.Random):
    """``random.Random(seed)`` that logs every ``random()`` it serves. HST's
    construction calls only ``choices`` (one ``random()`` a call with weights)
    and ``uniform`` (``a + (b - a) * random()``), so the log is the whole of
    river's draws, in order, and the generator's output is unchanged."""

    def __init__(self, seed) -> None:
        self.log: list[float] = []
        super().__init__(seed)

    def random(self) -> float:  # type: ignore[override]
        v = super().random()
        self.log.append(v)
        return v


class Served(random.Random):
    """A ``random.Random`` whose ``random()`` serves a given list, in order."""

    def __init__(self, values) -> None:
        self.values = list(values)
        self.i = 0
        super().__init__(0)

    def random(self) -> float:  # type: ignore[override]
        v = self.values[self.i]
        self.i += 1
        return v


def philox_build_uniforms(seed: int, group: int, count: int) -> list[float]:
    """The construction's uniforms: lane 2, learned-row count 0 (the trees exist
    before any row), words in order."""
    s = P.DrawStream(seed, group, lane=LANE_BUILD)
    return s.uniforms(0, count).tolist()


def from_list(vals: list[float]) -> Callable[[], float]:
    it = iter(vals)
    return lambda: next(it)


# --- river, driven ---------------------------------------------------------------------


def river_hst(n_trees, height, window, limits_list, rng: random.Random):
    from river import anomaly

    names = [f"x{i:02d}" for i in range(len(limits_list))]
    m = anomaly.HalfSpaceTrees(n_trees=n_trees, height=height, window_size=window,
                               limits=dict(zip(names, limits_list, strict=True)), seed=0)
    m.rng = rng
    return m, names


def river_tree_arrays(m, height):
    """River's nodes in heap order: (feature index, split) of each internal node."""
    n_int = 2**height - 1
    feats = np.zeros((len(m.trees), n_int), dtype=np.int32)
    splits = np.zeros((len(m.trees), n_int))
    for t, root in enumerate(m.trees):
        stack = [(root, 0)]
        while stack:
            node, i = stack.pop()
            if i >= n_int:
                continue
            feats[t, i] = int(node.feature[1:])
            splits[t, i] = node.threshold
            stack.append((node.children[0], 2 * i + 1))
            stack.append((node.children[1], 2 * i + 2))
    return feats, splits


def river_masses(m, height):
    n_all = 2 ** (height + 1) - 1
    L = np.zeros((len(m.trees), n_all))
    R = np.zeros((len(m.trees), n_all))
    for t, root in enumerate(m.trees):
        stack = [(root, 0)]
        while stack:
            node, i = stack.pop()
            L[t, i], R[t, i] = node.l_mass, node.r_mass
            if i < 2**height - 1:
                stack.append((node.children[0], 2 * i + 1))
                stack.append((node.children[1], 2 * i + 2))
    return L, R


def make_data(T: int, d: int, seed: int, limits) -> np.ndarray:
    g = np.random.default_rng(seed)
    lo = np.array([a for a, _ in limits])
    hi = np.array([b for _, b in limits])
    centre = lo + (hi - lo) * g.uniform(0.3, 0.7, d)
    X = centre + (hi - lo) * 0.08 * g.normal(size=(T, d))
    out = g.random(T) < 0.02
    X[out] = lo + (hi - lo) * g.random((int(out.sum()), d))
    return X


def compare(T, d, n_trees, height, window, seed, limits=None, source="river"):
    """source 'river': river's own Mersenne Twister draws, recorded, served to the
    port. source 'philox': the stream's lane-2 uniforms served to river and to the port."""
    limits = limits or [(0.0, 1.0)] * d
    X = make_data(T, d, seed, limits)
    if source == "river":
        rec = Recorder(seed)
        m, names = river_hst(n_trees, height, window, limits, rec)
        plain, _ = river_hst(n_trees, height, window, limits, random.Random(seed))
    else:
        vals = philox_build_uniforms(seed, P.group_hash("g"), 2 * n_trees * (2**height - 1))
        m, names = river_hst(n_trees, height, window, limits, Served(vals))
        plain = None
    s_river = np.empty(T)
    s_port = np.empty(T)
    port = None
    for t in range(T):
        xd = dict(zip(names, map(float, X[t]), strict=True))
        s_river[t] = m.score_one(xd)
        if plain is not None:
            plain.score_one(xd)
        m.learn_one(xd)
        if plain is not None:
            plain.learn_one(xd)
        if port is None:
            draws = rec.log if source == "river" else vals
            port = Trees(list(limits), n_trees, height, window, from_list(list(draws)))
            s_port[t] = 0.0  # the first row is scored in the first window
        else:
            s_port[t] = port.score(X[t])
        port.learn(X[t])
    fr, sr = river_tree_arrays(m, height)
    L, R = river_masses(m, height)
    same_plain = None
    if plain is not None:
        fp, sp = river_tree_arrays(plain, height)
        same_plain = bool(np.array_equal(fp, fr) and sp.tobytes() == sr.tobytes())
    return {
        "draws": 2 * n_trees * (2**height - 1),
        "features equal": bool(np.array_equal(fr, port.feature)),
        "splits equal (bits)": bool(sr.tobytes() == port.split.tobytes()),
        "masses equal at the end": bool(L.tobytes() == port.l.tobytes() and R.tobytes() == port.r.tobytes()),
        "rows": T,
        "scores equal to the bit": int(np.sum(s_river.view(np.uint64) == s_port.view(np.uint64))),
        "max |score diff|": float(np.max(np.abs(s_river - s_port))),
        "recording changes nothing": same_plain,
    }


# --- chunking, save/resume, zero-weight rows, state ----------------------------------------


def build(limits, n_trees, height, window, seed, group) -> Trees:
    vals = philox_build_uniforms(seed, group, 2 * n_trees * (2**height - 1))
    return Trees(list(limits), n_trees, height, window, from_list(vals))


def save(t: Trees) -> bytes:
    """Masses, window counter, n, first-window flag. The trees are not saved: they
    are a function of (seed, group, limits, n_trees, height) and are rebuilt."""
    import struct

    return struct.pack("<QQ?", t.counter, t.n, t.first_window) + t.l.tobytes() + t.r.tobytes()


def load(b: bytes, limits, n_trees, height, window, seed, group) -> Trees:
    import struct

    t = build(limits, n_trees, height, window, seed, group)
    t.counter, t.n, t.first_window = struct.unpack_from("<QQ?", b, 0)
    m = t.l.size
    t.l = np.frombuffer(b, "<f8", m, 17).reshape(t.l.shape).copy()
    t.r = np.frombuffer(b, "<f8", m, 17 + 8 * m).reshape(t.r.shape).copy()
    return t


def invariance(T: int = 3000, d: int = 3, n_trees: int = 10, height: int = 6, window: int = 100):
    import hashlib

    import streamcheck as S

    limits = [(0.0, 1.0)] * d
    X = make_data(T, d, 5, limits)
    g = np.random.default_rng(6)
    w = g.choice([0.0, 1.0, 1.0, 0.5, 2.0], size=T)
    w[:3] = 0.0  # zero-weight rows first
    grp = P.group_hash("sensor-1")
    digests = set()
    for _, ch in S.chunkings(T).items():
        for resume in (False, True):
            t = build(limits, n_trees, height, window, 11, grp)
            h = hashlib.sha256()
            i = 0
            for size in ch:
                for j in range(i, i + size):
                    h.update(P.f64_bytes(t.score(X[j])))
                    t.learn(X[j], float(w[j]))
                i += size
                if resume:
                    t = load(save(t), limits, n_trees, height, window, 11, grp)
            digests.add(h.hexdigest()[:16])
    # zero weight: a row of weight 0 leaves every byte of the state as it was
    t = build(limits, n_trees, height, window, 11, grp)
    before = save(t)
    t.learn(X[0], 0.0)
    zero_ok = save(t) == before
    return len(S.chunkings(T)) * 2, len(digests), sorted(digests)[0], zero_ok


def state_table():
    rows = []
    for n_trees in [10, 25, 50, 100]:
        for h in [4, 6, 8, 10, 12]:
            rows.append((n_trees, h, n_trees * (2 ** (h + 1) - 1), state_bytes_formula(n_trees, h, False),
                         state_bytes_formula(n_trees, h, True)))
    return rows
