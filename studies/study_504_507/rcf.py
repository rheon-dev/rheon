"""Task 507: the robust random cut tree (Guha, Mishra, Roy and Schrijvers 2016),
written from the paper on flat arrays, with rrcf's conventions where the paper
leaves a choice, and its draws from a supplied source.

**Insert** a point ``p`` into a tree with root ``v`` (Guha et al., Algorithm
InsertPoint)::

    if a leaf holds exactly p: its count n += 1, no draw (rrcf's duplicate rule)
    at node v with bounding box [lo, hi]:
        lo' = min(lo, p), hi' = max(hi, p), span_i = hi'_i - lo'_i, S = cumsum(span)
        r = 0 + (sum(span) - 0) * u                    (one uniform; numpy's sum)
        q = first i with S_i >= r;  c = lo'_q + S_q - r   (rrcf's mapping of r)
        if c <= lo_q:   new branch (q, c) with p's leaf on the left,  v on the right
        elif c >= hi_q: new branch (q, c) with v on the left,  p's leaf on the right
        else:           descend: p_{q_v} <= c_v goes left, else right; repeat
    bounding boxes of the ancestors grow to cover p

The paper's natural reading of ``r`` puts the cut at ``lo'_q + (r - S_{q-1})``;
rrcf measures it from the other end of the span. Both are uniform on the span:
this port takes rrcf's, so that the trees can match.

**Forget** (first in, first out by learned rows): drop the leaf's count; at 0
the leaf's parent branch is replaced by the sibling and the ancestors'
counts and boxes are recomputed.

**Score** of a leaf ``x`` (collusive displacement)::

    CoDisp(x) = max over the ancestors' path, node v from x up to the root's child,
                of n(sibling of v) / n(v)

and a forest's score is the mean over its trees.
"""

from __future__ import annotations

import numpy as np

import philox as P

LANE_CUT = 3


class LazyUniforms:
    """Uniforms of one (tree, row): word ``j`` is word ``j % 4`` of block
    ``((tree << 32) | j // 4, lane 3, n, slot)``; blocks made on demand."""

    def __init__(self, stream: P.DrawStream, n: int, tree: int) -> None:
        self.s, self.n, self.tree = stream, n, tree
        self.buf: list[int] = []
        self.j = 0

    def __call__(self) -> float:
        if self.j >= len(self.buf):
            b = len(self.buf) // 4
            self.buf.extend(P.philox4x64(((self.tree << 32) | b, LANE_CUT, self.n, self.s.slot), self.s.key))
        w = self.buf[self.j]
        self.j += 1
        return P.u01(w)


class Tree:
    """mode 'paper': at a leaf the box is the leaf's point, so the cut is drawn
    over the span of the two points. mode 'rrcf': rrcf 0.4.4's arithmetic at a
    leaf, where the leaf's box is one row (shape 1 x d), so ``bbox_hat[0]`` and
    ``bbox_hat[-1]`` are the same row and both end as ``max(leaf, p)``: every span
    is 0, the draw is multiplied by 0, and the cut is dimension 0 at
    ``max(leaf_0, p_0)``."""

    def __init__(self, d: int, mode: str = "paper") -> None:
        self.d = d
        self.mode = mode
        self.leaf_cuts = 0
        self.root = -1
        self.leaf = []  # bool
        self.par: list[int] = []
        self.lft: list[int] = []
        self.rgt: list[int] = []
        self.q: list[int] = []
        self.c: list[float] = []
        self.cnt: list[int] = []
        self.lo: list[np.ndarray] = []
        self.hi: list[np.ndarray] = []
        self.free: list[int] = []
        self.of: dict[int, int] = {}  # point index -> leaf node

    def _node(self) -> int:
        if self.free:
            i = self.free.pop()
            return i
        for a in (self.leaf, self.par, self.lft, self.rgt, self.q, self.c, self.cnt, self.lo, self.hi):
            a.append(None)
        return len(self.leaf) - 1

    def _new_leaf(self, x: np.ndarray) -> int:
        i = self._node()
        self.leaf[i], self.par[i], self.lft[i], self.rgt[i] = True, -1, -1, -1
        self.q[i], self.c[i], self.cnt[i] = -1, 0.0, 1
        self.lo[i], self.hi[i] = x.copy(), x.copy()
        return i

    def _query(self, x: np.ndarray) -> int:
        v = self.root
        while not self.leaf[v]:
            v = self.lft[v] if x[self.q[v]] <= self.c[v] else self.rgt[v]
        return v

    def _box_up(self, v: int) -> None:
        while v != -1:
            a, b = self.lft[v], self.rgt[v]
            self.lo[v] = np.minimum(self.lo[a], self.lo[b])
            self.hi[v] = np.maximum(self.hi[a], self.hi[b])
            v = self.par[v]

    def _count_up(self, v: int, inc: int) -> None:
        while v != -1:
            self.cnt[v] += inc
            v = self.par[v]

    def insert(self, x: np.ndarray, index: int, uniform) -> None:
        x = np.asarray(x, dtype=float)
        if self.root == -1:
            self.root = self._new_leaf(x)
            self.of[index] = self.root
            return
        near = self._query(x)
        if np.array_equal(self.lo[near], x):
            self._count_up(near, 1)
            self.of[index] = near
            return
        v = self.root
        parent, side = -1, 0
        while True:
            lo, hi = self.lo[v], self.hi[v]
            if self.leaf[v]:
                self.leaf_cuts += 1
            if self.leaf[v] and self.mode == "rrcf":
                lo2 = hi2 = np.maximum(hi, x)
            else:
                lo2, hi2 = np.minimum(lo, x), np.maximum(hi, x)
            span = hi2 - lo2
            S = np.cumsum(span)
            r = 0 + (span.sum() - 0) * uniform()  # numpy's sum (pairwise from 8 terms), as rrcf
            q = int(np.argmax(S >= r))
            c = lo2[q] + S[q] - r
            if c <= lo[q] or c >= hi[q]:
                new = self._new_leaf(x)
                br = self._node()
                self.leaf[br], self.q[br], self.c[br] = False, q, c
                if c <= lo[q]:
                    self.lft[br], self.rgt[br] = new, v
                else:
                    self.lft[br], self.rgt[br] = v, new
                self.cnt[br] = self.cnt[v] + 1
                self.par[br] = parent
                self.par[v] = br
                self.par[new] = br
                if parent == -1:
                    self.root = br
                elif side == 0:
                    self.lft[parent] = br
                else:
                    self.rgt[parent] = br
                self._count_up(parent, 1)
                self._box_up(br)
                self.of[index] = new
                return
            parent = v
            if x[self.q[v]] <= self.c[v]:
                v, side = self.lft[v], 0
            else:
                v, side = self.rgt[v], 1

    def forget(self, index: int) -> None:
        v = self.of.pop(index)
        if self.cnt[v] > 1:
            self._count_up(v, -1)
            return
        if v == self.root:
            self.root = -1
            self.free.append(v)
            return
        p = self.par[v]
        sib = self.rgt[p] if self.lft[p] == v else self.lft[p]
        g = self.par[p]
        self.par[sib] = g
        if g == -1:
            self.root = sib
        elif self.lft[g] == p:
            self.lft[g] = sib
        else:
            self.rgt[g] = sib
        self.free.extend([v, p])
        self._count_up(g, -1)
        self._box_up(g)

    def codisp(self, index: int) -> float:
        v = self.of[index]
        if v == self.root:
            return 0
        best = None
        while self.par[v] != -1:
            p = self.par[v]
            sib = self.rgt[p] if self.lft[p] == v else self.lft[p]
            r = self.cnt[sib] / self.cnt[v]
            best = r if best is None else max(best, r)
            v = p
        return best

    def branches(self) -> list[tuple[int, float, int]]:
        """(cut dimension, cut, count) of every branch, pre-order from the root."""
        out = []
        st = [self.root] if self.root != -1 else []
        while st:
            v = st.pop()
            if not self.leaf[v]:
                out.append((self.q[v], self.c[v], self.cnt[v]))
                st.append(self.rgt[v])
                st.append(self.lft[v])
        return out


def rrcf_branches(tree) -> list[tuple[int, float, int]]:
    import rrcf

    out = []
    st = [tree.root] if tree.root is not None else []
    while st:
        v = st.pop()
        if isinstance(v, rrcf.Branch):
            out.append((int(v.q), float(v.p), int(v.n)))
            st.append(v.r)
            st.append(v.l)
    return out


# --- draw sources handed to rrcf -----------------------------------------------------------


class ServedRS:
    """Stands in for rrcf's ``self.rng`` (a ``RandomState``): ``uniform(low, high)``
    is numpy's legacy ``low + (high - low) * u`` on a served ``u``."""

    def __init__(self) -> None:
        self.src = None

    def uniform(self, low, high):
        return low + (high - low) * self.src()


class RecordingRS:
    """``RandomState(seed)`` whose ``uniform`` is computed as numpy's is
    (``low + (high - low) * random_sample()``, one double of MT19937), logging u."""

    def __init__(self, seed: int) -> None:
        self.rs = np.random.RandomState(seed)
        self.log: list[float] = []

    def uniform(self, low, high):
        u = self.rs.random_sample()
        self.log.append(u)
        return low + (high - low) * u


# --- the streaming forest, run against rrcf -------------------------------------------------


def make_stream(T: int, shingle: int, seed: int) -> np.ndarray:
    g = np.random.default_rng(seed)
    t = np.arange(T + shingle)
    y = np.sin(2 * np.pi * t / 50) + 0.1 * g.normal(size=t.size)
    for k in g.choice(np.arange(100, T), size=max(1, T // 300), replace=False):
        y[k : k + 5] += 3.0  # anomalies
    y[200:210] = y[200]  # a flat run: exact duplicate shingles... of a constant only if shingle <= 10
    return np.array([y[i : i + shingle] for i in range(T)])


def run(T=1000, shingle=4, num_trees=8, tree_size=64, seed=3, source="philox", weights=None, mode="rrcf"):
    """Feed rrcf and the port the same points with the same draws. ``source``:
    'philox' (the stream's lane 3, served to both) or 'mt' (rrcf's own MT19937
    draws through a ``RandomState(seed)``, recorded and served to the port)."""
    import rrcf

    X = make_stream(T, shingle, seed)
    stream = P.DrawStream(seed, P.group_hash("rcf"))
    forest = [rrcf.RCTree() for _ in range(num_trees)]
    ports = [Tree(shingle, mode) for _ in range(num_trees)]
    plain = [rrcf.RCTree(random_state=1000 + t) for t in range(num_trees)] if source == "mt" else None
    if source == "philox":
        served = [ServedRS() for _ in range(num_trees)]
        for tr, s in zip(forest, served, strict=True):
            tr.rng = s
    else:
        rec = [RecordingRS(1000 + t) for t in range(num_trees)]
        for tr, r in zip(forest, rec, strict=True):
            tr.rng = r
    s_ref = np.zeros(T)
    s_port = np.zeros(T)
    n = 0
    fifo: list[int] = []
    for i in range(T):
        x = X[i]
        w = 1.0 if weights is None else weights[i]
        if w > 0 and len(fifo) == tree_size:
            old = fifo.pop(0)
            for tr in forest:
                tr.forget_point(old)
            for pt in ports:
                pt.forget(old)
            if plain:
                for tr in plain:
                    tr.forget_point(old)
        for t in range(num_trees):
            if source == "philox":
                served[t].src = LazyUniforms(stream, n, t)
                forest[t].insert_point(x, index=i)
                src = LazyUniforms(stream, n, t)
            else:
                k0 = len(rec[t].log)
                forest[t].insert_point(x, index=i)
                vals = iter(rec[t].log[k0:])
                src = lambda v=vals: next(v)  # noqa: E731
                if plain:
                    plain[t].insert_point(x, index=i)
            ports[t].insert(x, i, src)
            s_ref[i] += forest[t].codisp(i) / num_trees
            s_port[i] += ports[t].codisp(i) / num_trees
        if w > 0:
            fifo.append(i)
            n += 1
        else:  # a zero-weight row is scored and then taken back out
            for tr in forest:
                tr.forget_point(i)
            for pt in ports:
                pt.forget(i)
            if plain:
                for tr in plain:
                    tr.forget_point(i)
    trees_equal = all(rrcf_branches(f) == p.branches() for f, p in zip(forest, ports, strict=True))
    plain_equal = None
    if plain:
        plain_equal = all(rrcf_branches(a) == rrcf_branches(b) for a, b in zip(plain, forest, strict=True))
    misrouted = sum(
        sum(pt._query(pt.lo[leaf]) != leaf for leaf in set(pt.of.values())) for pt in ports
    )
    return {
        "s_ref": s_ref,
        "s_port": s_port,
        "leaf cuts": sum(p.leaf_cuts for p in ports),
        "insertions": T * num_trees,
        "leaves a query does not reach (end)": int(misrouted),
        "rows": T,
        "trees equal at the end (cut, value bits, count)": trees_equal,
        "scores equal to the bit": int(np.sum(s_ref.view(np.uint64) == s_port.view(np.uint64))),
        "max |score diff|": float(np.max(np.abs(s_ref - s_port))),
        "recording changes nothing": plain_equal,
        "learned rows": n,
    }


def labels(T: int, shingle: int, seed: int) -> np.ndarray:
    """Rows whose shingle holds an injected anomaly (make_stream's draws, replayed)."""
    g = np.random.default_rng(seed)
    t = np.arange(T + shingle)
    g.normal(size=t.size)
    lab = np.zeros(T + shingle, dtype=bool)
    for k in g.choice(np.arange(100, T), size=max(1, T // 300), replace=False):
        lab[k : k + 5] = True
    return np.array([lab[i : i + shingle].any() for i in range(T)])


def separation_violations_rrcf(tree) -> tuple[int, int]:
    """Branches of an rrcf tree where some point below the left child has
    x_q > c or some point below the right child has x_q <= c (the paper's cut
    separates them); and the branch count."""
    import rrcf

    bad = 0
    total = 0

    def pts(v):
        out = []
        tree.map_leaves(v, op=lambda leaf, acc: acc.append(leaf.x), acc=out)
        return np.array(out)

    st = [tree.root]
    while st:
        v = st.pop()
        if isinstance(v, rrcf.Branch):
            total += 1
            L, R_ = pts(v.l), pts(v.r)
            bad += bool((L[:, v.q] > v.p).any() or (R_[:, v.q] <= v.p).any())
            st.extend([v.l, v.r])
    return bad, total


def separation_violations_port(t: Tree) -> tuple[int, int]:
    bad = total = 0
    for v in range(len(t.leaf)):
        if v in t.free or t.leaf[v] or t.leaf[v] is None:
            continue
        total += 1
        a, b = t.lft[v], t.rgt[v]
        bad += bool(t.hi[a][t.q[v]] > t.c[v] or t.lo[b][t.q[v]] <= t.c[v])
    return bad, total


def modes_table(T=1500, shingle=4, num_trees=20, tree_size=128, seed=3):
    """rrcf against the port in each mode on one stream: agreement, the paper's
    separation, and the anomaly ranking (AUC of the score for the injected rows)."""
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score

    lab = labels(T, shingle, seed)
    out = []
    for mode in ["rrcf", "paper"]:
        r = run(T, shingle, num_trees, tree_size, seed, mode=mode)
        out.append({
            "mode": mode,
            "rows equal to rrcf's score (bits)": r["scores equal to the bit"],
            "Spearman vs rrcf": float(spearmanr(r["s_ref"], r["s_port"]).statistic),
            "AUC rrcf": float(roc_auc_score(lab, r["s_ref"])),
            "AUC port": float(roc_auc_score(lab, r["s_port"])),
            "insertions reaching a leaf": r["leaf cuts"],
            "insertions": r["insertions"],
            "leaves a query misses (end)": r["leaves a query does not reach (end)"],
        })
    return out


def violations_table(T=800, shingle=4, num_trees=8, tree_size=64, seed=3):
    import rrcf

    X = make_stream(T, shingle, seed)
    stream = P.DrawStream(seed, P.group_hash("rcf"))
    forest = [rrcf.RCTree() for _ in range(num_trees)]
    sv = [ServedRS() for _ in range(num_trees)]
    for tr, s in zip(forest, sv, strict=True):
        tr.rng = s
    paper = [Tree(shingle, "paper") for _ in range(num_trees)]
    for i in range(T):
        if i >= tree_size:
            for tr in forest:
                tr.forget_point(i - tree_size)
            for p in paper:
                p.forget(i - tree_size)
        for t in range(num_trees):
            sv[t].src = LazyUniforms(stream, i, t)
            forest[t].insert_point(X[i], index=i)
            paper[t].insert(X[i], i, LazyUniforms(stream, i, t))
    a = [separation_violations_rrcf(f) for f in forest]
    b = [separation_violations_port(p) for p in paper]
    return (sum(x for x, _ in a), sum(y for _, y in a)), (sum(x for x, _ in b), sum(y for _, y in b))


def invariance(T=800, shingle=4, num_trees=6, tree_size=64, seed=4):
    """The paper-mode forest alone, fed in each chunking, with a pickle round trip
    of the whole state after every chunk; zero-weight rows (the first among them)
    are scored and taken back out."""
    import hashlib
    import pickle

    import streamcheck as S

    X = make_stream(T, shingle, seed)
    g = np.random.default_rng(8)
    w = g.choice([0.0, 1.0, 1.0, 1.0], size=T)
    w[0] = 0.0
    stream = P.DrawStream(seed, P.group_hash("rcf"))
    digests = set()
    for _, ch in S.chunkings(T).items():
        for resume in (False, True):
            st = {"trees": [Tree(shingle, "paper") for _ in range(num_trees)], "fifo": [], "n": 0}
            h = hashlib.sha256()
            i = 0
            for size in ch:
                for j in range(i, i + size):
                    trees, fifo, n = st["trees"], st["fifo"], st["n"]
                    if w[j] > 0 and len(fifo) == tree_size:
                        old = fifo.pop(0)
                        for t in trees:
                            t.forget(old)
                    sc = 0.0
                    for k, t in enumerate(trees):
                        t.insert(X[j], j, LazyUniforms(stream, n, k))
                        sc += t.codisp(j) / num_trees
                    h.update(P.f64_bytes(sc))
                    if w[j] > 0:
                        fifo.append(j)
                        st["n"] = n + 1
                    else:
                        for t in trees:
                            t.forget(j)
                i += size
                if resume:
                    st = pickle.loads(pickle.dumps(st))
            digests.add(h.hexdigest()[:16])
    return len(S.chunkings(T)) * 2, len(digests), sorted(digests)[0]


def state_bytes(num_trees: int, tree_size: int, d: int) -> int:
    """Per tree: ``tree_size`` branches of (cut dimension u32, cut f64, count u32,
    box 2d f64, parent/left/right u32) and ``tree_size + 1`` leaves of (parent,
    count, ring slot: u32); once per forest the ring of the last ``tree_size``
    learned points (d f64 and a row id u64 each) and the learned-row count."""
    branch = 4 + 8 + 4 + 16 * d + 12
    leaf = 12
    return num_trees * (tree_size * branch + (tree_size + 1) * leaf) + tree_size * (8 * d + 8) + 8


def state_table():
    rows = []
    for nt, ts in [(10, 64), (20, 128), (40, 256), (100, 256)]:
        for d in [1, 4, 8, 16]:
            rows.append((nt, ts, d, state_bytes(nt, ts, d)))
    return rows
