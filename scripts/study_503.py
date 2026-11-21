"""Study 503: Hoeffding trees against river under a memory cap, in numpy.

The record is docs/records/STUDY-503-TREES.md. Two trees are written here from
their definitions, beside river's (0.27.0) on the same rows:

* the Hoeffding tree classifier (VFDT, Domingos and Hulten 2000): a leaf keeps
  the class weights ``n_c`` and, per feature and class, a Gaussian observer
  (weight, mean, sum of squares about the mean, min, max). Every
  ``grace_period`` of learned weight a leaf scores, per feature, ``n_splits``
  equally spaced thresholds between the feature's min and max with the
  criterion ``G`` (information gain, Gini or Hellinger) on the class weights
  each side (the Gaussian CDF splits a class's weight at a threshold), and
  splits on the best when

      G(best) - G(second) > eps  or  eps < tau,
      eps = sqrt(R^2 ln(1/delta) / (2 n)),   n = the leaf's weight,

  with ``R = log2(max(classes seen, 2))`` for information gain and Hellinger
  and 1 for Gini. Leaf predictions: majority class (``mc``), naive Bayes over
  the Gaussian observers (``nb``), or whichever of the two has been right on
  more of the leaf's learned weight (``nba``).

* the Hoeffding tree regressor (FIMT without the linear leaves, Ikonomovska,
  Gama and Dzeroski 2011): a leaf keeps the target's weight, mean and sum of
  squares; an observer per feature keeps, per candidate threshold, the target
  statistics of the rows at or below it (river's E-BST, its truncated form
  TE-BST, or the quantization observer QO of Mastelini and de Carvalho 2021).
  The merit of a threshold is the variance reduction

      VR = var(S) - sum_side n_side / n var(S_side)

  and a leaf splits when VR(second) / VR(best) < 1 - eps or eps < tau, with
  ``R = 1``. The leaf predicts the target mean (``leaf_prediction='mean'``).

The arithmetic follows river's own order of operations where it is a choice
(the Welford update of its Rust ``Gaussian``, its entropy and Gini forms, its
naive Bayes normalisation, the fdlibm ``erf`` its Rust backend links), so the
row-by-row comparison can be made to the bit; where river differs from the
definitions (the null split's merit, zero-weight rows, its memory manager),
the tree here has a switch and the record says which setting was compared.

numpy holds the packed state (the byte counts of part 3 are its sizes) and
the data; the per-row recursion runs on Python floats, which are IEEE
doubles with the same rounding as the Rust a build would use.

Run (the study's venv holds numpy and river; the record says how it was made):

    .venv-503/bin/python scripts/study_503.py part1 --out <dir>   (classifier vs river)
    .venv-503/bin/python scripts/study_503.py part2 --out <dir>   (regressor, observers)
    .venv-503/bin/python scripts/study_503.py part3 --out <dir>   (memory cap)
    .venv-503/bin/python scripts/study_503.py part4 --out <dir>   (determinism, weights, cost)

Each part writes <dir>/<part>.json; a rerun prints the cached result unless
--force is given. Data are generated from seeded numpy generators.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys
import time

import numpy as np

# --------------------------------------------------------------------------- libm

# river's Rust backend links the `libm` crate (0.2.16) for `erf`, a port of
# FreeBSD's msun s_erf.c, whose `exp` is msun's e_exp.c. glibc's erf (Python's
# math.erf) differs from it in the last bit of erf on about 2% of arguments
# (part 1 measures it), so the two are ported here, operation for operation.

_ERX = 8.45062911510467529297e-01
_EFX8 = 1.02703333676410069053e00
_PP = (
    1.28379167095512558561e-01,
    -3.25042107247001499370e-01,
    -2.84817495755985104766e-02,
    -5.77027029648944159157e-03,
    -2.37630166566501626084e-05,
)
_QQ = (
    3.97917223959155352819e-01,
    6.50222499887672944485e-02,
    5.08130628187576562776e-03,
    1.32494738004321644526e-04,
    -3.96022827877536812320e-06,
)
_PA = (
    -2.36211856075265944077e-03,
    4.14856118683748331666e-01,
    -3.72207876035701323847e-01,
    3.18346619901161753674e-01,
    -1.10894694282396677476e-01,
    3.54783043256182359371e-02,
    -2.16637559486879084300e-03,
)
_QA = (
    1.06420880400844228286e-01,
    5.40397917702171048937e-01,
    7.18286544141962662868e-02,
    1.26171219808761642112e-01,
    1.36370839120290507362e-02,
    1.19844998467991074170e-02,
)
_RA = (
    -9.86494403484714822705e-03,
    -6.93858572707181764372e-01,
    -1.05586262253232909814e01,
    -6.23753324503260060396e01,
    -1.62396669462573470355e02,
    -1.84605092906711035994e02,
    -8.12874355063065934246e01,
    -9.81432934416914548592e00,
)
_SA = (
    1.96512716674392571292e01,
    1.37657754143519042600e02,
    4.34565877475229228821e02,
    6.45387271733267880336e02,
    4.29008140027567833386e02,
    1.08635005541779435134e02,
    6.57024977031928170135e00,
    -6.04244152148580987438e-02,
)
_RB = (
    -9.86494292470009928597e-03,
    -7.99283237680523006574e-01,
    -1.77579549177547519889e01,
    -1.60636384855821916062e02,
    -6.37566443368389627722e02,
    -1.02509513161107724954e03,
    -4.83519191608651397019e02,
)
_SB = (
    3.03380607434824582924e01,
    3.25792512996573918826e02,
    1.53672958608443695994e03,
    3.19985821950859553908e03,
    2.55305040643316442583e03,
    4.74528541206955367215e02,
    -2.24409524465858183362e01,
)
_LN2HI = 6.93147180369123816490e-01
_LN2LO = 1.90821492927058770002e-10
_INVLN2 = 1.44269504088896338700e00
_EP = (
    1.66666666666666019037e-01,
    -2.77777777770155933842e-03,
    6.61375632143793436117e-05,
    -1.65339022054652515390e-06,
    4.13813679705723846039e-08,
)


def _hi(x: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", x))[0] >> 32


def _zero_low(x: float) -> float:
    b = struct.unpack("<Q", struct.pack("<d", x))[0] & 0xFFFFFFFF00000000
    return struct.unpack("<d", struct.pack("<Q", b))[0]


def msun_exp(x: float) -> float:
    """msun's e_exp.c as the libm crate has it (no x87 path)."""
    hx = _hi(x)
    sign = hx >> 31
    hx &= 0x7FFFFFFF
    if hx >= 0x4086232B:
        if x != x:
            return x
        if x > 709.782712893383973096:
            return math.inf
        if x < -745.13321910194110842:
            return 0.0
    if hx > 0x3FD62E42:
        if hx >= 0x3FF0A2B2:  # noqa: SIM108 (msun's structure, kept)
            k = int(_INVLN2 * x + (0.5, -0.5)[sign])
        else:
            k = 1 - sign - sign
        hi = x - k * _LN2HI
        lo = k * _LN2LO
        x = hi - lo
    elif hx > 0x3E300000:
        k = 0
        hi = x
        lo = 0.0
    else:
        return 1.0 + x
    xx = x * x
    p1, p2, p3, p4, p5 = _EP
    c = x - xx * (p1 + xx * (p2 + xx * (p3 + xx * (p4 + xx * p5))))
    y = 1.0 + (x * c / (2.0 - c) - lo + hi)
    return y if k == 0 else math.ldexp(y, k)


def _erfc1(x: float) -> float:
    s = abs(x) - 1.0
    a = _PA
    q_ = _QA
    p = a[0] + s * (a[1] + s * (a[2] + s * (a[3] + s * (a[4] + s * (a[5] + s * a[6])))))
    q = 1.0 + s * (q_[0] + s * (q_[1] + s * (q_[2] + s * (q_[3] + s * (q_[4] + s * q_[5])))))
    return 1.0 - _ERX - p / q


def _erfc2(ix: int, x: float) -> float:
    if ix < 0x3FF40000:
        return _erfc1(x)
    x = abs(x)
    s = 1.0 / (x * x)
    if ix < 0x4006DB6D:
        a, b = _RA, _SA
        r = a[0] + s * (
            a[1] + s * (a[2] + s * (a[3] + s * (a[4] + s * (a[5] + s * (a[6] + s * a[7])))))
        )
        big = 1.0 + s * (
            b[0]
            + s * (b[1] + s * (b[2] + s * (b[3] + s * (b[4] + s * (b[5] + s * (b[6] + s * b[7]))))))
        )
    else:
        a, b = _RB, _SB
        r = a[0] + s * (a[1] + s * (a[2] + s * (a[3] + s * (a[4] + s * (a[5] + s * a[6])))))
        big = 1.0 + s * (
            b[0] + s * (b[1] + s * (b[2] + s * (b[3] + s * (b[4] + s * (b[5] + s * b[6])))))
        )
    z = _zero_low(x)
    return msun_exp(-z * z - 0.5625) * msun_exp((z - x) * (z + x) + r / big) / x


def msun_erf(x: float) -> float:
    """msun's s_erf.c as the libm crate has it."""
    ix = _hi(x)
    sign = ix >> 31
    ix &= 0x7FFFFFFF
    if ix >= 0x7FF00000:
        return 1.0 - 2.0 * sign + 1.0 / x
    if ix < 0x3FEB0000:
        if ix < 0x3E300000:
            return 0.125 * (8.0 * x + _EFX8 * x)
        z = x * x
        p, q = _PP, _QQ
        r = p[0] + z * (p[1] + z * (p[2] + z * (p[3] + z * p[4])))
        s = 1.0 + z * (q[0] + z * (q[1] + z * (q[2] + z * (q[3] + z * q[4]))))
        return x + x * (r / s)
    if ix < 0x40180000:  # noqa: SIM108 (msun's structure, kept)
        y = 1.0 - _erfc2(ix, x)
    else:
        y = 1.0 - 2.2250738585072014e-308
    return -y if sign else y


# --------------------------------------------------------------------------- helpers

HALF_LOG_TAU = 0.9189385332046727  # river's literal 0.918_938_533_204_672_7, ln(2 pi) / 2
SQRT_2 = 1.4142135623730951
NEG_INF = -math.inf


def ulps(a: float, b: float) -> int:
    """Units in the last place between two doubles (0 if both are equal or both NaN)."""
    if a == b or (a != a and b != b):
        return 0
    ia = struct.unpack("<q", struct.pack("<d", a))[0]
    ib = struct.unpack("<q", struct.pack("<d", b))[0]
    ia = ia if ia >= 0 else -(ia & 0x7FFFFFFFFFFFFFFF)
    ib = ib if ib >= 0 else -(ib & 0x7FFFFFFFFFFFFFFF)
    return abs(ia - ib)


def naive_sum(v) -> float:
    s = 0.0
    for a in v:
        s += a
    return s


# --------------------------------------------------------------------------- Gaussian observer

# An observer is a list [n, mean, S, min, max]: the weight seen, the weighted
# mean, the weighted sum of squared deviations (Welford; variance S / (n - 1),
# river's ddof = 1 on the weight), and the min and max of the values seen.


def g_new() -> list:
    return [0.0, 0.0, 0.0, 0.0, 0.0]


def g_update(o: list, x: float, w: float, zero_moves_range: bool) -> None:
    """One weighted value. A zero weight changes nothing unless `zero_moves_range`
    (river: a zero-weight value still moves min and max, and on an observer
    with no weight yet it replaces them)."""
    if w == 0.0 and not zero_moves_range:
        return
    if o[0] == 0.0:
        o[3] = x
        o[4] = x
    else:
        o[3] = min(o[3], x)
        o[4] = max(o[4], x)
    if w == 0.0:
        return
    m_old = o[1]
    n = o[0] + w
    o[0] = n
    ratio = w / n
    upd = m_old + ratio * (x - m_old)
    o[1] = upd if math.isfinite(upd) else m_old * (1.0 - ratio) + x * ratio
    c = w * (x - m_old) * (x - o[1])
    if c != c and x != m_old:
        o[2] = math.inf
    elif c > 0.0:
        o[2] += c


def g_var(o: list) -> float:
    return o[2] / (o[0] - 1.0) if o[0] > 1.0 else 0.0


def g_log_pdf(o: list, x: float) -> float:
    v = g_var(o)
    if v > 0.0:
        d = x - o[1]
        return d * d / (-2.0 * v) - HALF_LOG_TAU - 0.5 * math.log(v)
    return NEG_INF


def g_cdf(o: list, x: float, erf) -> float:
    sd = math.sqrt(g_var(o))
    if sd > 0.0:
        return 0.5 * (1.0 + erf((x - o[1]) / sd / SQRT_2))
    return 0.0


# --------------------------------------------------------------------------- criteria


def entropy(counts) -> float:
    e = 0.0
    t = 0.0
    for c in counts:
        if c > 0.0:
            e -= c * math.log2(c)
            t += c
    if t == 0.0:
        return 0.0
    return (e + t * math.log2(t)) / t


def gini_native(counts) -> float:
    t = naive_sum(counts)
    if t == 0.0:
        return 1.0
    return 1.0 - naive_sum((c / t) * (c / t) for c in counts)


def split_merit(counts, left, right, criterion: str, form: str, sum_=naive_sum) -> float:
    lw = sum_(left)
    rw = sum_(right)
    tot = lw + rw
    if criterion == "info_gain":
        return entropy(counts) - (lw * entropy(left) + rw * entropy(right)) / tot
    if criterion == "gini":
        if form == "python":  # river's Python GiniSplitCriterion
            g = 0.0
            for wside, side in ((lw, left), (rw, right)):
                gi = 1.0
                if wside != 0.0:
                    for v in side:
                        p = v / wside
                        gi -= p * p
                g += (wside / tot) * gi
            return 1.0 - g
        return 1.0 - (lw * gini_native(left) + rw * gini_native(right)) / tot
    if criterion == "hellinger":
        tn = left[0] + right[0]
        tp = left[1] + right[1]
        if tn == 0.0 or tp == 0.0:
            return 0.0
        a = math.sqrt(left[0] / tn) - math.sqrt(left[1] / tp)
        b = math.sqrt(right[0] / tn) - math.sqrt(right[1] / tp)
        return math.sqrt(a * a + b * b)
    raise ValueError(criterion)


# --------------------------------------------------------------------------- classifier


class CLeaf:
    __slots__ = ("counts", "obs", "depth", "last", "mc", "nb", "active")

    def __init__(self, depth: int, counts: list, sum_=None):
        self.counts = counts
        self.obs = None  # k * C observers, allocated at the leaf's first learned row
        self.depth = depth
        self.last = (sum_ or naive_sum)(counts)
        self.mc = 0.0
        self.nb = 0.0
        self.active = True

    def total(self) -> float:
        return naive_sum(self.counts)


class HTClassifier:
    """The Hoeffding tree classifier with Gaussian observers.

    Options beyond river's parameters:
      null_merit   the merit of not splitting: -inf (river, whose null split
                   only wins when no feature has a candidate) or 0.0 (the
                   paper's X_0, whose information gain is 0).
      zero_weight  'skip' (a zero-weight row changes no state; the library's
                   rule) or 'river' (it moves the Gaussian observers' min and
                   max, as river does).
      erf          msun_erf (river's Rust backend) or math.erf (glibc).
      nb_form      'native' (river's Rust backend: exp(v - max) / sum) or
                   'python' (river's Python path: exp(v - logsumexp)).
    """

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        grace_period: float = 200.0,
        delta: float = 1e-7,
        tau: float = 0.05,
        split_criterion: str = "info_gain",
        leaf_prediction: str = "nba",
        nb_threshold: float = 0.0,
        n_splits: int = 10,
        min_branch_fraction: float = 0.01,
        max_share_to_split: float = 0.99,
        merit_preprune: bool = True,
        max_depth: int | None = None,
        null_merit: float = NEG_INF,
        zero_weight: str = "skip",
        erf=msun_erf,
        nb_form: str = "native",
        gini_form: str = "native",
        depth_rule: str = "native",
    ):
        self.k = n_features
        self.C = n_classes
        self.grace = float(grace_period)
        self.delta = delta
        self.tau = tau
        self.crit = split_criterion
        self.pred_kind = leaf_prediction
        self.nb_threshold = float(nb_threshold)
        self.n_splits = n_splits
        self.mbf = min_branch_fraction
        self.max_share = max_share_to_split
        self.preprune = merit_preprune
        self.max_depth = max_depth if max_depth is not None else 980
        self.null_merit = null_merit
        self.zero_weight = zero_weight
        self.erf = erf
        self.nb_form = nb_form
        self.gini_form = gini_form
        self.depth_rule = depth_rule  # 'python': a leaf at max_depth is deactivated when it learns
        # river's Python path sums class weights with Python's sum() (compensated
        # since 3.12); its Rust backend adds them in order. Equal for 2 classes.
        self._sum = sum if nb_form == "python" else naive_sum
        self.labels: dict = {}  # label -> class index, in first-seen order (as river)
        self.leaves = [CLeaf(0, [0.0] * n_classes)]
        # a branch: [feature, threshold, left, right, left_weight, right_weight];
        # a node id >= 0 is a branch, ~i a leaf.
        self.branches: list = []
        self.root = ~0
        self.weight_seen = 0.0
        self.n_split_attempts = 0
        # memory cap (part 3): None, or (max_bytes, layout)
        self.cap = None
        self.n_inactive_by_cap = 0

    # ---- class bookkeeping
    def class_index(self, y) -> int:
        try:
            return self.labels[y]
        except KeyError:
            i = len(self.labels)
            if i >= self.C:
                raise ValueError("more classes than declared") from None
            self.labels[y] = i
            return i

    def sorted_labels(self):
        return sorted(self.labels)

    # ---- traversal
    def leaf_of(self, x) -> int:
        node = self.root
        br = self.branches
        while node >= 0:
            b = br[node]
            node = b[2] if x[b[0]] <= b[1] else b[3]
        return ~node

    # ---- leaf predictions
    def _nb_votes(self, leaf: CLeaf, x):
        tot = self._sum(leaf.counts)
        C = self.C
        votes = [NEG_INF] * C
        obs = leaf.obs
        for c in range(C):
            cw = leaf.counts[c]
            if cw <= 0.0:
                continue
            v = math.log(cw / tot)
            if obs is not None:
                for f in range(self.k):
                    lp = g_log_pdf(obs[f * C + c], x[f])
                    if math.isfinite(lp):
                        v += lp
            votes[c] = v
        return votes, tot

    def _nb_proba(self, leaf: CLeaf, x):
        votes, tot = self._nb_votes(leaf, x)
        C = self.C
        if tot == 0.0:
            return [0.0] * C
        mx = NEG_INF
        for v in votes:
            mx = max(mx, v)
        if not math.isfinite(mx):
            return [0.0] * C
        if self.nb_form == "python":
            lse = mx + math.log(sum(math.exp(v - mx) for v in votes if math.isfinite(v)))
            return [math.exp(v - lse) if math.isfinite(v) else 0.0 for v in votes]
        norm = naive_sum(math.exp(v - mx) if math.isfinite(v) else 0.0 for v in votes)
        return [math.exp(v - mx) / norm if math.isfinite(v) else 0.0 for v in votes]

    def _nb_class(self, leaf: CLeaf, x, order):
        votes, tot = self._nb_votes(leaf, x)
        if tot == 0.0:
            return None
        best, bv = None, NEG_INF
        for c in order:
            if leaf.counts[c] <= 0.0:
                continue
            if votes[c] > bv:
                best, bv = c, votes[c]
        return best

    def _mc_proba(self, leaf: CLeaf):
        tot = self._sum(leaf.counts)
        if tot == 0.0:
            return [0.0] * self.C
        if self.nb_form == "python":  # river's normalize_values_in_dict: c * (1 / total)
            inv = 1 / tot
            return [c * inv for c in leaf.counts]
        return [c / tot for c in leaf.counts]

    def _use_nb(self, leaf: CLeaf) -> bool:
        if not leaf.active:
            return False
        if self.pred_kind == "mc":
            return False
        if self.pred_kind == "nb":
            return self._sum(leaf.counts) >= self.nb_threshold
        return leaf.nb >= leaf.mc

    def predict_proba(self, x) -> dict:
        """{label: probability} over the labels seen, in sorted order ({} before any row)."""
        if not self.labels:
            return {}
        leaf = self.leaves[self.leaf_of(x)]
        p = self._nb_proba(leaf, x) if self._use_nb(leaf) else self._mc_proba(leaf)
        return {lab: p[self.labels[lab]] for lab in self.sorted_labels()}

    def predict(self, x):
        if not self.labels:
            return None
        leaf = self.leaves[self.leaf_of(x)]
        order = [self.labels[lab] for lab in self.sorted_labels()]
        if self._use_nb(leaf):
            c = self._nb_class(leaf, x, order)
            c = 0 if c is None else c
        else:
            c = order[0]
            for cand in order:
                if leaf.counts[cand] > leaf.counts[c]:
                    c = cand
        inv = {i: lab for lab, i in self.labels.items()}
        return inv[c]

    # ---- learning
    def learn(self, x, y, w: float = 1.0) -> None:
        c = self.class_index(y)
        self.weight_seen += w
        if w == 0.0 and self.zero_weight == "skip":
            return
        node = self.root
        parent, right = -1, False
        br = self.branches
        while node >= 0:
            b = br[node]
            parent = node
            if x[b[0]] <= b[1]:
                b[4] += w
                node, right = b[2], False
            else:
                b[5] += w
                node, right = b[3], True
        li = ~node
        leaf = self.leaves[li]
        C = self.C
        if leaf.active and self.pred_kind == "nba":
            maj = 0
            for cand in range(C):
                if leaf.counts[cand] > leaf.counts[maj]:
                    maj = cand
            if self._sum(leaf.counts) == 0.0 or maj == c:
                leaf.mc += w
            nbc = self._nb_class(leaf, x, range(C))
            if nbc is not None and nbc == c:
                leaf.nb += w
        leaf.counts[c] += w
        if leaf.active:
            if leaf.obs is None:
                leaf.obs = [g_new() for _ in range(self.k * C)]
            zr = self.zero_weight == "river"
            for f in range(self.k):
                g_update(leaf.obs[f * C + c], x[f], w, zr)
        if leaf.active and self.depth_rule == "python" and leaf.depth >= self.max_depth:
            leaf.active = False
            leaf.obs = None
            return
        if (
            leaf.active
            and leaf.depth < self.max_depth
            and self._sum(leaf.counts) - leaf.last >= self.grace
            and not self._attempt_split(li, parent, right)
        ):
            leaf.last = self._sum(leaf.counts)

    def _best_candidate(self, leaf: CLeaf, f: int):
        C = self.C
        obs = leaf.obs[f * C : (f + 1) * C]
        lo = math.inf
        hi = NEG_INF
        for o in obs:
            if o[0] > 0.0:
                lo = min(lo, o[3])
                hi = max(hi, o[4])
        if not math.isfinite(lo) or hi <= lo:
            return None
        bin_size = (hi - lo) / (self.n_splits + 1.0)
        best = None
        for i in range(1, self.n_splits + 1):
            if math.isfinite(bin_size):
                t = lo + bin_size * i
            else:
                r = i / (self.n_splits + 1.0)
                t = lo * (1.0 - r) + hi * r
            left = [0.0] * C
            right = [0.0] * C
            lpres = [False] * C
            rpres = [False] * C
            for ci in range(C):
                o = obs[ci]
                wt = o[0]
                if wt == 0.0:
                    continue
                if t < o[3]:
                    right[ci] = wt
                    rpres[ci] = True
                elif t >= o[4]:
                    left[ci] = wt
                    lpres[ci] = True
                else:
                    lpres[ci] = rpres[ci] = True
                    left[ci] = g_cdf(o, t, self.erf) * wt
                    right[ci] = wt - left[ci]
            lw = self._sum(left)
            rw = self._sum(right)
            tot = lw + rw
            if tot == 0.0 or lw / tot <= self.mbf or rw / tot <= self.mbf:
                continue
            if (
                self.crit == "hellinger"
                and self.nb_form == "python"
                and not (lpres[0] and lpres[1] and rpres[0] and rpres[1])
            ):
                m = 0.0  # river's Python Hellinger: a class absent from a side -> KeyError -> 0
            else:
                m = split_merit(leaf.counts, left, right, self.crit, self.gini_form, self._sum)
            if best is None or m > best[0]:
                best = (m, f, t, left, right)
        return best

    def _attempt_split(self, li: int, parent: int, right: bool) -> bool:
        self.n_split_attempts += 1
        leaf = self.leaves[li]
        tot = self._sum(leaf.counts)
        maj = 0.0
        observed = 0
        for v in leaf.counts:
            maj = max(maj, v)
            observed += v > 0.0
        if observed < 2:
            return False
        if maj / tot > self.max_share:
            leaf.active = False
            leaf.obs = None
            self._python_manager()
            return False
        sugg = []  # (merit, candidate or None); the null split first, as river
        if self.preprune:
            sugg.append((self.null_merit, None))
        for f in range(self.k):
            if any(leaf.obs[f * self.C + ci][0] > 0.0 for ci in range(self.C)):
                cand = self._best_candidate(leaf, f)
                sugg.append((NEG_INF, None) if cand is None else (cand[0], cand))
        sugg.sort(key=lambda s: s[0])  # stable, as Rust's sort_by and Python's sort
        if len(sugg) < 2:
            should = len(sugg) > 0
        else:
            best_m, second_m = sugg[-1][0], sugg[-2][0]
            rng_ = 1.0 if self.crit == "gini" else math.log2(float(max(observed, 2)))
            bound = math.sqrt(
                (rng_ * rng_ * math.log(1.0 / self.delta)) / (2.0 * self._sum(leaf.counts))
            )
            should = best_m - second_m > bound or bound < self.tau
        if not should:
            return False
        best = sugg[-1][1]
        if best is None:
            leaf.active = False
            leaf.obs = None
            self._python_manager()
            return False
        if self.cap is not None and not self._may_grow():
            return False
        _, f, t, left, right_counts = best
        new_l = CLeaf(leaf.depth + 1, left, self._sum)
        new_r = CLeaf(leaf.depth + 1, right_counts, self._sum)
        self.leaves[li] = new_l
        self.leaves.append(new_r)
        bid = len(self.branches)
        self.branches.append(
            [f, t, ~li, ~(len(self.leaves) - 1), naive_sum(left), naive_sum(right_counts)]
        )
        if parent < 0:
            self.root = bid
        elif right:
            self.branches[parent][3] = bid
        else:
            self.branches[parent][2] = bid
        self._python_manager()
        if self.cap is not None:
            self._enforce_cap()
        return True

    def _python_manager(self) -> None:
        """river's Python path runs its memory manager after every decided split
        attempt (a split, a null split winning, a dominated leaf). With its size
        estimates still 0 (first measured at memory_estimate_period rows) it
        deactivates nothing and reactivates every inactive leaf above max_depth
        with empty observers: a deactivation there only resets the observers."""
        if self.depth_rule != "python":
            return
        for lf in self.leaves:
            if not lf.active and lf.depth < self.max_depth:
                lf.active = True
                lf.obs = None

    # ---- structure, for the comparison with river
    def preorder(self) -> list:
        out = []
        stack = [self.root]
        while stack:
            n = stack.pop()
            if n >= 0:
                b = self.branches[n]
                out.append((b[0], b[1]))
                stack.append(b[3])
                stack.append(b[2])
            else:
                out.append(None)
        return out

    def n_nodes(self) -> int:
        return len(self.branches) + len(self.leaves)

    # ---- memory cap (part 3) -------------------------------------------------
    def _may_grow(self) -> bool:
        return True

    def _enforce_cap(self) -> None:
        return None


# --------------------------------------------------------------------------- streams


def random_tree_concept(k: int, n_classes: int, depth: int, rng):
    """A random axis-aligned tree over N(0, 1) features: (feature, threshold) at
    each of 2^depth - 1 internal nodes (heap order), a class at each leaf."""
    n_int = 2**depth - 1
    feats = rng.integers(0, k, n_int)
    thr = rng.normal(0.0, 0.7, n_int)
    leaf_class = rng.integers(0, n_classes, 2**depth)
    return feats, thr, leaf_class


def label_by_concept(X, concept, depth):
    feats, thr, leaf_class = concept
    node = np.zeros(len(X), dtype=np.int64)
    for _ in range(depth):
        f = feats[node]
        go_right = X[np.arange(len(X)), f] > thr[node]
        node = 2 * node + 1 + go_right
    return leaf_class[node - (2**depth - 1)]


def clf_stream(
    n: int,
    k: int,
    n_classes: int,
    seed: int,
    depth: int = 6,
    noise: float = 0.05,
    weights: str = "one",
):
    rng = np.random.default_rng(seed)
    concept = random_tree_concept(k, n_classes, depth, rng)
    X = rng.standard_normal((n, k))
    y = label_by_concept(X, concept, depth)
    flip = rng.random(n) < noise
    y = np.where(flip, rng.integers(0, n_classes, n), y)
    if weights == "one":
        w = np.ones(n)
    elif weights == "uniform":
        w = rng.uniform(0.2, 3.0, n)
    else:
        raise ValueError(weights)
    return X, y.astype(int), w


# --------------------------------------------------------------------------- river runners


def river_classifier(native: bool, **kw):
    from river import tree

    if not native:
        # river runs its Python path when max_size is not its default 100 MiB;
        # 1e9 MiB keeps its memory manager from ever binding.
        kw.setdefault("max_size", 1e9)
    m = tree.HoeffdingTreeClassifier(**kw)
    return m


def river_preorder(model) -> list:
    from river.tree.nodes.branch import DTBranch

    root = model._root_for_inspection()
    out = []
    stack = [root]
    while stack:
        n = stack.pop()
        if isinstance(n, DTBranch):
            out.append((n.feature, n.threshold))
            stack.append(n.children[1])
            stack.append(n.children[0])
        else:
            out.append(None)
    return out


def run_prequential_clf(model, X, y, w, mine: bool):
    """Predict, then learn, row by row. Returns the probability rows (labels
    sorted), the predicted labels and the node count after each row."""
    n = len(X)
    probas, preds, nodes = [], [], np.zeros(n, dtype=np.int64)
    t0 = time.perf_counter()
    for i in range(n):
        if mine:
            xi = X[i].tolist()
            p = model.predict_proba(xi)
            preds.append(model.predict(xi))
            model.learn(xi, int(y[i]), float(w[i]))
            nodes[i] = model.n_nodes()
        else:
            xi = dict(enumerate(X[i].tolist()))
            p = model.predict_proba_one(xi)
            preds.append(model.predict_one(xi) if p else None)
            model.learn_one(xi, int(y[i]), w=float(w[i]))
            nodes[i] = model.n_nodes or 1
        probas.append(p)
    return probas, preds, nodes, time.perf_counter() - t0


def compare_rows(pa, pb) -> dict:
    """Probability rows (dicts) compared to the bit."""
    n_diff = 0
    worst_abs = 0.0
    worst_ulp = 0
    first = None
    for i, (a, b) in enumerate(zip(pa, pb, strict=True)):
        if a.keys() != b.keys():
            n_diff += 1
            first = i if first is None else first
            continue
        d = False
        for key in a:
            if a[key] != b[key]:
                d = True
                worst_abs = max(worst_abs, abs(a[key] - b[key]))
                worst_ulp = max(worst_ulp, ulps(a[key], b[key]))
        if d:
            n_diff += 1
            first = i if first is None else first
    return {"rows_differ": n_diff, "first_row": first, "max_abs": worst_abs, "max_ulps": worst_ulp}


def split_rows(nodes) -> list:
    return [int(i) for i in np.flatnonzero(np.diff(np.concatenate([[1], nodes])) != 0)]


def compare_trees(mine_pre, river_pre) -> dict:
    if len(mine_pre) != len(river_pre):
        same_shape = False
    else:
        same_shape = all(
            (a is None) == (b is None) for a, b in zip(mine_pre, river_pre, strict=True)
        )
    thr = [
        (a, b) for a, b in zip(mine_pre, river_pre, strict=False) if a is not None and b is not None
    ]
    feat_same = all(a[0] == b[0] for a, b in thr)
    thr_ulps = max((ulps(a[1], b[1]) for a, b in thr), default=0)
    return {
        "nodes_mine": len(mine_pre),
        "nodes_river": len(river_pre),
        "same_shape": same_shape,
        "features_same": feat_same,
        "threshold_max_ulps": thr_ulps,
    }


# --------------------------------------------------------------------------- part 1

CLF_CASES = [
    # name, k, C, stream kw, tree kw
    ("A_defaults", 5, 2, {"depth": 6}, {}),
    ("B_gini_g50", 10, 2, {"depth": 6}, {"grace_period": 50, "split_criterion": "gini"}),
    ("C_3class_mc", 5, 3, {"depth": 5}, {"grace_period": 50, "leaf_prediction": "mc"}),
    ("D_nb_g100", 8, 2, {"depth": 6}, {"grace_period": 100, "leaf_prediction": "nb"}),
    ("E_weighted", 5, 2, {"depth": 6, "weights": "uniform"}, {"grace_period": 50}),
    ("F_3class_nba", 6, 3, {"depth": 6}, {"grace_period": 50, "delta": 1e-5}),
    ("G_many_leaves", 10, 2, {"depth": 9, "noise": 0.02}, {"grace_period": 50, "tau": 0.1}),
    # river runs these on its Python path only (Hellinger has no native form)
    ("H_hellinger", 5, 2, {"depth": 6}, {"grace_period": 50, "split_criterion": "hellinger"}),
    # max_depth: river's Python path deactivates a leaf at the depth (it then
    # predicts its majority class); its Rust backend keeps it active (naive Bayes)
    ("I_max_depth_2", 5, 2, {"depth": 6}, {"grace_period": 50, "max_depth": 2}),
    # near-pure leaves: river's Rust backend deactivates a leaf whose majority
    # share passes max_share_to_split; its Python path only resets its observers
    ("J_near_pure", 5, 2, {"depth": 3, "noise": 0.004}, {"grace_period": 50}),
]


def _erf_gap(n: int) -> dict:
    """glibc's erf (math.erf) and the msun port against river's Rust cdf."""
    from river.proba import Gaussian

    g = Gaussian._from_state(n=3.0, m=0.0, var=0.5, ddof=1)
    sd = math.sqrt(0.5)
    rng = np.random.default_rng(7)
    v = np.concatenate([rng.standard_normal(n) * 2.0, rng.uniform(-8.0, 8.0, n)]).tolist()
    glibc_bad = msun_bad = 0
    glibc_abs = 0.0
    for x in v:
        r = g.cdf(x)
        z = x / sd / SQRT_2
        a = 0.5 * (1.0 + math.erf(z))
        b = 0.5 * (1.0 + msun_erf(z))
        if a != r:
            glibc_bad += 1
            glibc_abs = max(glibc_abs, abs(a - r))
        msun_bad += b != r
    # erf itself: glibc against the port
    erf_ulps = max(ulps(math.erf(x / sd / SQRT_2), msun_erf(x / sd / SQRT_2)) for x in v)
    return {
        "n": len(v),
        "glibc_cdf_differ": glibc_bad,
        "glibc_cdf_max_abs": glibc_abs,
        "msun_cdf_differ": msun_bad,
        "erf_glibc_vs_msun_max_ulps": erf_ulps,
    }


def _clf_case(name, k, C, skw, tkw, n, n_py, seed):
    X, y, w = clf_stream(n, k, C, seed, **skw)
    out = {"case": name, "k": k, "classes": C, "rows": n, "stream": skw, "tree": tkw}
    # river's Rust backend (its default whenever the settings allow it)
    rn = river_classifier(True, **tkw)
    pr, cr, nr, tr = run_prequential_clf(rn, X, y, w, mine=False)
    if rn._native_model is None:
        out["native"] = None
        return _clf_python_only(out, X, y, w, k, C, tkw, n_py)
    mine = HTClassifier(k, C, **tkw)
    pm, cm, nm, tm = run_prequential_clf(mine, X, y, w, mine=True)
    out["native"] = {
        "proba": compare_rows(pm, pr),
        "pred_differ": int(sum(a != b for a, b in zip(cm, cr, strict=True))),
        "split_rows_same": split_rows(nm) == split_rows(nr),
        "n_splits": len(split_rows(nr)),
        "tree": compare_trees(mine.preorder(), river_preorder(rn)),
        "accuracy_river": float(np.mean(np.array(cr[1:]) == y[1:])),
        "seconds_river": tr,
        "seconds_mine": tm,
    }
    # the same with glibc's erf: what the port buys
    mg = HTClassifier(k, C, erf=math.erf, **tkw)
    pg, cg, ng, _ = run_prequential_clf(mg, X, y, w, mine=True)
    out["native_glibc_erf"] = {
        "proba": compare_rows(pg, pr),
        "split_rows_same": split_rows(ng) == split_rows(nr),
        "tree": compare_trees(mg.preorder(), river_preorder(rn)),
    }
    # river's Python path, on the first n_py rows
    m = min(n, n_py)
    rp = river_classifier(False, **tkw)
    pp, cp, npy, tp = run_prequential_clf(rp, X[:m], y[:m], w[:m], mine=False)
    assert rp._native_model is None
    form = {"nb_form": "python", "gini_form": "python", "depth_rule": "python"}
    mp = HTClassifier(k, C, **tkw, **form)
    pm2, cm2, nm2, _ = run_prequential_clf(mp, X[:m], y[:m], w[:m], mine=True)
    out["python_path"] = {
        "rows": m,
        "proba": compare_rows(pm2, pp),
        "pred_differ": int(sum(a != b for a, b in zip(cm2, cp, strict=True))),
        "split_rows_same": split_rows(nm2) == split_rows(npy),
        "tree": compare_trees(mp.preorder(), river_preorder(rp)),
        "seconds_river": tp,
        # river's two paths against each other
        "python_vs_native_proba": compare_rows(pp, pr[:m]),
        "python_vs_native_split_rows_same": split_rows(npy) == split_rows(nr[:m]),
    }
    # the paper's null split (merit 0) against river's (-inf): how the tree moves
    m0 = HTClassifier(k, C, null_merit=0.0, **tkw)
    _, c0, n0, _ = run_prequential_clf(m0, X, y, w, mine=True)
    out["null_merit_0"] = {
        "n_splits": len(split_rows(n0)),
        "first_split_row": (split_rows(n0) or [None])[0],
        "first_split_row_river": (split_rows(nr) or [None])[0],
        "accuracy": float(np.mean(np.array(c0[1:]) == y[1:])),
    }
    return out


def _clf_python_only(out, X, y, w, k, C, tkw, n_py):
    m = min(len(X), n_py)
    rp = river_classifier(False, **tkw)
    pp, cp, npy, tp = run_prequential_clf(rp, X[:m], y[:m], w[:m], mine=False)
    mp = HTClassifier(k, C, nb_form="python", gini_form="python", depth_rule="python", **tkw)
    pm, cm, nm, _ = run_prequential_clf(mp, X[:m], y[:m], w[:m], mine=True)
    out["python_path"] = {
        "rows": m,
        "proba": compare_rows(pm, pp),
        "pred_differ": int(sum(a != b for a, b in zip(cm, cp, strict=True))),
        "split_rows_same": split_rows(nm) == split_rows(npy),
        "n_splits": len(split_rows(npy)),
        "tree": compare_trees(mp.preorder(), river_preorder(rp)),
        "seconds_river": tp,
    }
    return out


def _one_feature_case(n: int, seed: int) -> dict:
    """river's null split has merit -inf, so with one feature the first split
    attempt always splits (the second best is the null split)."""
    X, y, w = clf_stream(n, 1, 2, seed, depth=3, noise=0.45)
    rn = river_classifier(True, grace_period=50)
    _, _, nr, _ = run_prequential_clf(rn, X, y, w, mine=False)
    m = HTClassifier(1, 2, grace_period=50)
    _, _, nm, _ = run_prequential_clf(m, X, y, w, mine=True)
    m0 = HTClassifier(1, 2, grace_period=50, null_merit=0.0)
    _, _, n0, _ = run_prequential_clf(m0, X, y, w, mine=True)
    return {
        "rows": n,
        "noise": 0.45,
        "splits_river": len(split_rows(nr)),
        "splits_mine": len(split_rows(nm)),
        "first_split_river": (split_rows(nr) or [None])[0],
        "same_split_rows": split_rows(nm) == split_rows(nr),
        "splits_null_merit_0": len(split_rows(n0)),
    }


def part1(args) -> dict:
    res = {"erf": _erf_gap(100_000), "cases": []}
    for i, (name, k, C, skw, tkw) in enumerate(CLF_CASES):
        r = _clf_case(name, k, C, skw, tkw, args.rows, args.py_rows, 100 + i)
        res["cases"].append(r)
        print(
            name,
            json.dumps((r["native"] or {}).get("proba")),
            "py",
            json.dumps(r["python_path"]["proba"]),
            flush=True,
        )
    res["one_feature"] = _one_feature_case(4000, 5)
    return res


def summarise1(res) -> None:
    print("erf:", json.dumps(res["erf"]))
    for r in res["cases"]:
        if r["native"] is None:
            py = r["python_path"]
            print(
                f"{r['case']}: k={r['k']} C={r['classes']} (river's Python path only)",
                json.dumps(py),
            )
            continue
        nv, g, py, n0 = r["native"], r["native_glibc_erf"], r["python_path"], r["null_merit_0"]
        print(f"{r['case']}: k={r['k']} C={r['classes']} rows={r['rows']} splits={nv['n_splits']}")
        print(
            "  native:",
            json.dumps(nv["proba"]),
            "pred_differ",
            nv["pred_differ"],
            "split_rows_same",
            nv["split_rows_same"],
            json.dumps(nv["tree"]),
            f"acc={nv['accuracy_river']:.4f}",
            f"t river {nv['seconds_river']:.1f}s mine {nv['seconds_mine']:.1f}s",
        )
        print(
            "  glibc erf:",
            json.dumps(g["proba"]),
            "split_rows_same",
            g["split_rows_same"],
            json.dumps(g["tree"]),
        )
        print(
            "  python path:",
            json.dumps(py["proba"]),
            "pred_differ",
            py["pred_differ"],
            "split_rows_same",
            py["split_rows_same"],
            json.dumps(py["tree"]),
        )
        print(
            "  river python vs native:",
            json.dumps(py["python_vs_native_proba"]),
            "split rows same",
            py["python_vs_native_split_rows_same"],
        )
        print("  null merit 0:", json.dumps(n0))
    print("one feature:", json.dumps(res["one_feature"]))


# --------------------------------------------------------------------------- regressor

# A target statistic is a list [n, mean, S] with river's Python `stats.Var`
# arithmetic (variance S / (n - 1) for n > 1, else 0), and its parallel
# combination (Chan et al.) for adding and removing a block of rows.


def v_new() -> list:
    return [0.0, 0.0, 0.0]


def v_update(v: list, y: float, w: float) -> None:
    m_old = v[1]
    v[0] += w
    v[1] += (w / v[0]) * (y - v[1])  # a first row of weight 0 divides 0 by 0 (river raises)
    v[2] += w * (y - m_old) * (y - v[1])


def v_get(v: list) -> float:
    return v[2] / (v[0] - 1) if v[0] > 1 else 0.0


def v_iadd(a: list, b: list) -> list:
    S = a[2] + b[2] + (a[1] - b[1]) ** 2 * a[0] * b[0] / (a[0] + b[0])
    old_n = a[0]
    a[0] += b[0]
    a[1] = (old_n * a[1] + b[0] * b[1]) / a[0]
    a[2] = S
    return a


def v_isub(a: list, b: list) -> list:
    old_n = a[0]
    a[0] -= b[0]
    if a[0] > 0:
        a[1] = (old_n * a[1] - b[0] * b[1]) / a[0]
    else:
        a[0] = 0.0
        a[1] = 0.0
    a[2] = a[2] - b[2] - (a[1] - b[1]) ** 2 * a[0] * b[0] / (a[0] + b[0])
    return a


def vr_merit(pre: list, left: list, right: list, min_samples: int) -> float:
    """Variance reduction, river's order: var(pre) - nl/n var(l) - nr/n var(r);
    0 when a side has fewer than min_samples_split weight."""
    if left[0] >= min_samples and right[0] >= min_samples:
        n = pre[0]
        vr = v_get(pre)
        vr -= left[0] / n * v_get(left)
        vr -= right[0] / n * v_get(right)
        return vr
    return 0.0


class EBST:
    """river's extended binary search tree observer: a node per distinct value
    (rounded to `digits` decimals for TE-BST), holding the target statistics of
    the rows at or below its value within its subtree. Bounded only by the
    number of distinct (rounded) values: E-BST grows with the stream."""

    __slots__ = ("root", "digits")

    def __init__(self, digits: int | None):
        self.root = None  # node: [value, Var, left, right]
        self.digits = digits

    def update(self, x: float, y: float, w: float) -> None:
        if self.digits is not None:
            x = round(x, self.digits)
        if self.root is None:
            self.root = [x, v_new(), None, None]
            v_update(self.root[1], y, w)
            return
        cur = self.root
        ante, is_right = None, False
        while cur is not None:
            ante = cur
            if x == cur[0]:
                v_update(cur[1], y, w)
                return
            if x < cur[0]:
                v_update(cur[1], y, w)
                cur, is_right = cur[2], False
            else:
                cur, is_right = cur[3], True
        node = [x, v_new(), None, None]
        v_update(node[1], y, w)
        if is_right:
            ante[3] = node
        else:
            ante[2] = node

    def n_nodes(self) -> int:
        n, stack = 0, [self.root] if self.root else []
        while stack:
            nd = stack.pop()
            n += 1
            stack.extend(c for c in (nd[2], nd[3]) if c is not None)
        return n

    def best(self, pre: list, min_samples: int, alias_bug: bool):
        """(merit, threshold, left stats, right stats) or None, river's in-order walk."""
        if self.root is None:
            return None
        cand = None
        aux = v_new()
        ops = [("d", self.root)]
        while ops:
            act, nd = ops.pop()
            if act == "d":
                if nd[3] is not None:
                    ops.append(("s", nd))
                    ops.append(("d", nd[3]))
                    ops.append(("a", nd))
                ops.append(("p", nd))
                if nd[2] is not None:
                    ops.append(("d", nd[2]))
            elif act == "a":
                v_iadd(aux, nd[1])
            elif act == "s":
                v_isub(aux, nd[1])
            else:
                left = v_iadd(list(nd[1]), aux)
                right = v_isub(list(pre), left)
                m = vr_merit(pre, left, right, min_samples)
                if cand is None or m > cand[0]:
                    cand = (m, nd[0], left, right)
        return cand

    def remove_bad_splits(self, pre, min_samples, ratio, best_vr, eps) -> None:
        """FIMT-DD's pruning of the E-BST after a failed split attempt, as river
        has it (a node whose left subtree is clean is kept with its subtree)."""
        if self.root is None:
            return
        aux = v_new()
        stack = [[self.root, None, True, 0, False]]
        child_bad = False
        while stack:
            fr = stack[-1]
            nd, par, is_left, phase, _ = fr
            if phase == 0:
                if nd[2] is not None:
                    fr[3] = 1
                    stack.append([nd[2], nd, True, 0, False])
                    continue
                child_bad = True
                fr[3] = 1
                continue
            if phase == 1:
                if not child_bad:
                    stack.pop()
                    child_bad = False
                    continue
                if nd[3] is not None:
                    v_iadd(aux, nd[1])
                    fr[3] = 2
                    fr[4] = True
                    stack.append([nd[3], nd, False, 0, False])
                    continue
                child_bad = True
                fr[3] = 2
                continue
            if fr[4]:
                v_isub(aux, nd[1])
            is_bad = child_bad
            stack.pop()
            if is_bad:
                left = v_iadd(list(nd[1]), aux)
                right = v_isub(list(pre), left)
                m = vr_merit(pre, left, right, min_samples)
                if m / best_vr < ratio - 2 * eps:
                    nd[2] = None
                    nd[3] = None
                    if par is None:
                        self.root = None
                    elif is_left:
                        par[2] = None
                    else:
                        par[3] = None
                    child_bad = True
                    continue
            child_bad = False


class QO:
    """The quantization observer: slot floor(x / radius) keeps the weight and
    mean of x and the target statistics of its rows. Candidate thresholds are
    midpoints between consecutive slots' x means. Bounded by the feature's
    range over the radius, and here also by `max_slots` (a value whose slot is
    new once the cap is full goes to the nearest existing slot).

    river (0.27) evaluates, at the threshold between slots i-1 and i, the
    partition {slots <= i} | {slots > i}, and the left statistics it hands the
    new leaf are its running accumulator, which by the end of the walk holds
    every slot (the left child starts with the parent's statistics).
    `river_quirks=True` reproduces both; False evaluates the partition the
    threshold makes and copies the left statistics."""

    __slots__ = ("radius", "slots", "max_slots")

    def __init__(self, radius: float, max_slots: int | None = None):
        self.radius = radius
        self.slots: dict = {}  # index -> [x_n, x_mean, Var]
        self.max_slots = max_slots

    def update(self, x: float, y: float, w: float) -> None:
        i = math.floor(x / self.radius)
        sl = self.slots.get(i)
        if sl is None:
            if self.max_slots is not None and len(self.slots) >= self.max_slots:
                i = min(self.slots, key=lambda j: (abs(j - i), j))
                sl = self.slots[i]
            else:
                xs = [0.0, 0.0]
                xs[0] += w
                xs[1] += (w / xs[0]) * (x - xs[1])
                v = v_new()
                v_update(v, y, w)
                self.slots[i] = [xs[0], xs[1], v]
                return
        sl[0] += w
        sl[1] += (w / sl[0]) * (x - sl[1])
        v_update(sl[2], y, w)

    def best(self, pre: list, min_samples: int, river_quirks: bool):
        if len(self.slots) == 1:
            return None
        cand = None
        aux = v_new()
        prev_x = None
        prev_aux = None
        for i in sorted(self.slots):
            sl = self.slots[i]
            x = sl[1]
            if not river_quirks:
                prev_aux = list(aux)
            v_iadd(aux, sl[2])
            if prev_x is None:
                prev_x = x
                continue
            left = aux if river_quirks else prev_aux
            right = v_isub(list(pre), left)
            m = vr_merit(pre, left, right, min_samples)
            if cand is None or m > cand[0]:
                cand = (m, (prev_x + x) / 2.0, left, right)
            prev_x = x
        if cand is not None and not river_quirks:
            return cand
        return cand  # with the quirk, cand[2] is `aux`, mutated to the total by now

    def n_slots(self) -> int:
        return len(self.slots)


class RLeaf:
    __slots__ = ("stats", "obs", "depth", "last", "active")

    def __init__(self, depth: int, stats: list):
        self.stats = stats
        self.obs = None
        self.depth = depth
        self.last = stats[0]
        self.active = True


class HTRegressor:
    """The Hoeffding tree regressor with leaf_prediction='mean' (river's
    HoeffdingTreeRegressor on its only code path, which is Python).

    splitter: ('tebst', digits) | ('ebst',) | ('qo', radius, max_slots)."""

    def __init__(
        self,
        n_features: int,
        grace_period: float = 200.0,
        delta: float = 1e-7,
        tau: float = 0.05,
        min_samples_split: int = 5,
        merit_preprune: bool = True,
        max_depth: int | None = None,
        splitter=("tebst", 1),
        null_merit: float = NEG_INF,
        zero_weight: str = "skip",
        river_quirks: bool = True,
        prune_ebst: bool = True,
    ):
        self.k = n_features
        self.grace = float(grace_period)
        self.delta = delta
        self.tau = tau
        self.min_samples = min_samples_split
        self.preprune = merit_preprune
        self.max_depth = max_depth if max_depth is not None else 980
        self.splitter = splitter
        self.null_merit = null_merit
        self.zero_weight = zero_weight
        self.river_quirks = river_quirks
        self.prune_ebst = prune_ebst
        self.leaves = [RLeaf(0, v_new())]
        self.branches: list = []  # [feature, threshold, left, right]
        self.root = ~0
        self.weight_seen = 0.0
        self.cap = None

    def _new_obs(self):
        kind = self.splitter[0]
        if kind == "tebst":
            return EBST(self.splitter[1])
        if kind == "ebst":
            return EBST(None)
        if kind == "qo":
            return QO(self.splitter[1], self.splitter[2] if len(self.splitter) > 2 else None)
        raise ValueError(kind)

    def leaf_of(self, x) -> int:
        node = self.root
        br = self.branches
        while node >= 0:
            b = br[node]
            node = b[2] if x[b[0]] <= b[1] else b[3]
        return ~node

    def predict(self, x) -> float:
        return self.leaves[self.leaf_of(x)].stats[1]

    def learn(self, x, y: float, w: float = 1.0) -> None:
        self.weight_seen += w
        if w == 0.0 and self.zero_weight == "skip":
            return
        node = self.root
        parent, right = -1, False
        br = self.branches
        while node >= 0:
            b = br[node]
            parent = node
            if x[b[0]] <= b[1]:
                node, right = b[2], False
            else:
                node, right = b[3], True
        li = ~node
        leaf = self.leaves[li]
        v_update(leaf.stats, y, w)
        if leaf.active:
            if leaf.obs is None:
                leaf.obs = [self._new_obs() for _ in range(self.k)]
            for f in range(self.k):
                leaf.obs[f].update(x[f], y, w)
            if leaf.depth >= self.max_depth:
                leaf.active = False
                leaf.obs = None
            elif leaf.stats[0] - leaf.last >= self.grace:
                self._attempt_split(li, parent, right)
                leaf.last = leaf.stats[0]

    def _attempt_split(self, li: int, parent: int, right: bool) -> bool:
        leaf = self.leaves[li]
        pre = leaf.stats
        sugg = []
        if self.preprune:
            sugg.append((self.null_merit, None))
        for f in range(self.k):
            c = leaf.obs[f].best(pre, self.min_samples, self.river_quirks)
            sugg.append((NEG_INF, None) if c is None else (c[0], (f,) + tuple(c[1:])))
        sugg.sort(key=lambda s: s[0])
        should = False
        eps = None
        if len(sugg) < 2:
            should = len(sugg) > 0
        else:
            eps = math.sqrt((1.0 * 1.0 * math.log(1.0 / self.delta)) / (2.0 * pre[0]))
            best_m, second_m = sugg[-1][0], sugg[-2][0]
            if best_m > 0.0 and (second_m / best_m < 1 - eps or eps < self.tau):
                should = True
        if should:
            best = sugg[-1][1]
            if best is None:
                leaf.active = False
                leaf.obs = None
            else:
                if self.cap is not None and not self._may_grow():
                    return False
                f, t, lstats, rstats = best
                self.leaves[li] = RLeaf(leaf.depth + 1, lstats)
                self.leaves.append(RLeaf(leaf.depth + 1, rstats))
                bid = len(self.branches)
                self.branches.append([f, t, ~li, ~(len(self.leaves) - 1)])
                if parent < 0:
                    self.root = bid
                elif right:
                    self.branches[parent][3] = bid
                else:
                    self.branches[parent][2] = bid
            # river's memory manager after a split attempt that decided (see the classifier)
            for lf in self.leaves:
                if not lf.active and lf.depth < self.max_depth:
                    lf.active = True
                    lf.obs = None
            if self.cap is not None:
                self._enforce_cap()
            return best is not None
        if (
            self.prune_ebst
            and len(sugg) >= 2
            and sugg[-1][0] > 0
            and sugg[-2][0] > 0
            and self.splitter[0] in ("tebst", "ebst")
        ):
            ratio = sugg[-2][0] / sugg[-1][0]
            for ob in leaf.obs:
                ob.remove_bad_splits(pre, self.min_samples, ratio, sugg[-1][0], eps)
        return False

    def preorder(self) -> list:
        out = []
        stack = [self.root]
        while stack:
            n = stack.pop()
            if n >= 0:
                b = self.branches[n]
                out.append((b[0], b[1]))
                stack.append(b[3])
                stack.append(b[2])
            else:
                out.append(None)
        return out

    def n_nodes(self) -> int:
        return len(self.branches) + len(self.leaves)

    def _may_grow(self) -> bool:
        return True

    def _enforce_cap(self) -> None:
        return None


def reg_stream(
    n: int,
    k: int,
    seed: int,
    kind: str = "tree",
    noise: float = 0.3,
    weights: str = "one",
    depth: int = 6,
):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, k))
    if kind == "tree":
        feats, thr, _ = random_tree_concept(k, 2, depth, rng)
        leaf_val = rng.normal(0.0, 1.0, 2**depth)
        node = np.zeros(n, dtype=np.int64)
        for _ in range(depth):
            f = feats[node]
            node = 2 * node + 1 + (X[np.arange(n), f] > thr[node])
        mu = leaf_val[node - (2**depth - 1)]
    elif kind == "friedman":  # Friedman #1 on N(0,1) features mapped to (0, 1)
        U = 0.5 * (1.0 + np.tanh(X / 1.5))
        mu = (
            10 * np.sin(np.pi * U[:, 0] * U[:, 1 % k])
            + 20 * (U[:, 2 % k] - 0.5) ** 2
            + 10 * U[:, 3 % k]
            + 5 * U[:, 4 % k]
        ) / 5.0
    else:
        raise ValueError(kind)
    y = mu + noise * rng.standard_normal(n)
    w = np.ones(n) if weights == "one" else rng.uniform(0.2, 3.0, n)
    return X, y, w


def river_regressor(**kw):
    from river import tree

    kw.setdefault("leaf_prediction", "mean")
    return tree.HoeffdingTreeRegressor(**kw)


def river_splitter(spec):
    from river.tree import splitter as sp

    if spec[0] == "tebst":
        return sp.TEBSTSplitter(digits=spec[1])
    if spec[0] == "ebst":
        return sp.EBSTSplitter()
    if spec[0] == "qo":
        return sp.QOSplitter(radius=spec[1])
    raise ValueError(spec)


def run_prequential_reg(model, X, y, w, mine: bool):
    n = len(X)
    pred = np.zeros(n)
    nodes = np.zeros(n, dtype=np.int64)
    t0 = time.perf_counter()
    for i in range(n):
        if mine:
            xi = X[i].tolist()
            pred[i] = model.predict(xi)
            model.learn(xi, float(y[i]), float(w[i]))
            nodes[i] = model.n_nodes()
        else:
            xi = dict(enumerate(X[i].tolist()))
            pred[i] = model.predict_one(xi)
            model.learn_one(xi, float(y[i]), w=float(w[i]))
            nodes[i] = model.n_nodes or 1
    return pred, nodes, time.perf_counter() - t0


def compare_pred(a, b) -> dict:
    d = np.flatnonzero(a != b)
    return {
        "rows_differ": int(len(d)),
        "first_row": int(d[0]) if len(d) else None,
        "max_abs": float(np.max(np.abs(a - b))) if len(d) else 0.0,
        "max_ulps": max((ulps(float(a[i]), float(b[i])) for i in d), default=0),
    }


def river_obs_sizes(model) -> dict:
    """The observers river holds at its active leaves: E-BST nodes or QO slots."""
    from river.tree.splitter import EBSTSplitter, QOSplitter

    sizes = []
    for leaf in model._find_leaves():
        if not leaf.is_active():
            continue
        for sp in leaf.splitters.values():
            if isinstance(sp, EBSTSplitter):
                n, stack = 0, [sp._root] if sp._root else []
                while stack:
                    nd = stack.pop()
                    n += 1
                    stack.extend(c for c in (nd._left, nd._right) if c is not None)
                sizes.append(n)
            elif isinstance(sp, QOSplitter):
                sizes.append(len(sp._quantizer))
    return {
        "observers": len(sizes),
        "max": max(sizes, default=0),
        "mean": float(np.mean(sizes)) if sizes else 0.0,
    }


def mine_obs_sizes(model) -> dict:
    sizes = []
    for leaf in model.leaves:
        if leaf.active and leaf.obs is not None:
            for ob in leaf.obs:
                sizes.append(ob.n_nodes() if isinstance(ob, EBST) else ob.n_slots())
    return {
        "observers": len(sizes),
        "max": max(sizes, default=0),
        "mean": float(np.mean(sizes)) if sizes else 0.0,
    }


REG_CASES = [
    # name, k, stream kw, tree kw (river's names), splitter
    ("RA_tebst_default", 5, {"kind": "tree"}, {}, ("tebst", 1)),
    ("RB_tebst_g50", 8, {"kind": "friedman"}, {"grace_period": 50}, ("tebst", 1)),
    ("RC_ebst_g100", 5, {"kind": "tree"}, {"grace_period": 100}, ("ebst",)),
    ("RD_qo_g100", 5, {"kind": "tree"}, {"grace_period": 100}, ("qo", 0.25)),
    ("RE_qo_friedman", 8, {"kind": "friedman"}, {"grace_period": 50}, ("qo", 0.25)),
    (
        "RF_weighted_tebst",
        5,
        {"kind": "tree", "weights": "uniform"},
        {"grace_period": 50},
        ("tebst", 1),
    ),
    (
        "RG_weighted_qo",
        5,
        {"kind": "tree", "weights": "uniform"},
        {"grace_period": 50},
        ("qo", 0.25),
    ),
]


def _reg_case(name, k, skw, tkw, spl, n, seed):
    X, y, w = reg_stream(n, k, seed, **skw)
    out = {"case": name, "k": k, "rows": n, "stream": skw, "tree": tkw, "splitter": list(spl)}
    rv = river_regressor(splitter=river_splitter(spl), **tkw)
    pr, nr, tr = run_prequential_reg(rv, X, y, w, mine=False)
    mine = HTRegressor(k, splitter=spl, **tkw)
    pm, nm, tm = run_prequential_reg(mine, X, y, w, mine=True)
    out["river"] = {
        "pred": compare_pred(pm, pr),
        "split_rows_same": split_rows(nm) == split_rows(nr),
        "n_splits": len(split_rows(nr)),
        "tree": compare_trees(mine.preorder(), river_preorder(rv)),
        "mse_river": float(np.mean((pr[1:] - y[1:]) ** 2)),
        "obs_river": river_obs_sizes(rv),
        "obs_mine": mine_obs_sizes(mine),
        "seconds_river": tr,
        "seconds_mine": tm,
    }
    # the definitions without river's quirks (QO's partition and aliasing, the null split)
    d = HTRegressor(k, splitter=spl, river_quirks=False, **tkw)
    pd_, nd_, _ = run_prequential_reg(d, X, y, w, mine=True)
    out["definition"] = {
        "n_splits": len(split_rows(nd_)),
        "mse": float(np.mean((pd_[1:] - y[1:]) ** 2)),
        "pred_vs_river": compare_pred(pd_, pr),
    }
    # river's own default leaves ('adaptive': the mean or a linear model per leaf)
    ra = river_regressor(splitter=river_splitter(spl), leaf_prediction="adaptive", **tkw)
    pa, na, _ = run_prequential_reg(ra, X, y, w, mine=False)
    out["river_adaptive"] = {
        "mse": float(np.mean((pa[1:] - y[1:]) ** 2)),
        "n_splits": len(split_rows(na)),
        "split_rows_same_as_mean": split_rows(na) == split_rows(nr),
    }
    return out


def _qo_quirk_demo(seed: int) -> dict:
    """The left child's statistics after river's first QO split, against the
    rows that the threshold actually sends left."""
    X, y, w = reg_stream(3000, 3, seed, kind="tree")
    rv = river_regressor(splitter=river_splitter(("qo", 0.25)), grace_period=100)
    first = None
    for i in range(len(X)):
        rv.learn_one(dict(enumerate(X[i].tolist())), float(y[i]))
        if (rv.n_nodes or 1) > 1:
            first = i
            break
    if first is None:
        return {"split": False}
    root = rv._root
    f, t = root.feature, root.threshold
    seen = X[: first + 1]
    goes_left = seen[:, f] <= t
    left_leaf, right_leaf = root.children
    return {
        "split_row": first,
        "feature": f,
        "threshold": t,
        "rows_seen": first + 1,
        "left_rows_by_threshold": int(goes_left.sum()),
        "left_child_n_river": left_leaf.stats.mean.n,
        "right_rows_by_threshold": int((~goes_left).sum()),
        "right_child_n_river": right_leaf.stats.mean.n,
        "left_mean_by_threshold": float(y[: first + 1][goes_left].mean()),
        "left_child_mean_river": left_leaf.stats.mean.get(),
        "parent_mean": float(y[: first + 1].mean()),
    }


def _gaussian_obs_vs_hist(n: int, seed: int) -> dict:
    """River's classification observers: GaussianSplitter (5 floats per class
    and feature) against HistogramSplitter (n_bins per class and feature), on
    the same stream, river's Python path, for accuracy and size."""
    from river.tree import splitter as sp

    X, y, w = clf_stream(n, 5, 2, seed, depth=6)
    out = {}
    for name, s in (
        ("gaussian", sp.GaussianSplitter()),
        ("histogram_256", sp.HistogramSplitter(n_bins=256, n_splits=32)),
        ("histogram_32", sp.HistogramSplitter(n_bins=32, n_splits=16)),
    ):
        m = river_classifier(False, grace_period=50, splitter=s, leaf_prediction="mc")
        _, preds, nodes, t = run_prequential_clf(m, X, y, w, mine=False)
        bins = []
        for leaf in m._find_leaves():
            if leaf.is_active():
                for o in leaf.splitters.values():
                    if hasattr(o, "hists"):
                        bins.extend(len(h) for h in o.hists.values())
        out[name] = {
            "accuracy": float(np.mean(np.array(preds[1:]) == y[1:])),
            "n_splits": len(split_rows(nodes)),
            "seconds": t,
            "bins_max": max(bins, default=0),
            "bins_mean": float(np.mean(bins)) if bins else 0.0,
        }
    return out


def part2(args) -> dict:
    res = {"cases": []}
    for i, (name, k, skw, tkw, spl) in enumerate(REG_CASES):
        r = _reg_case(name, k, skw, tkw, spl, args.reg_rows, 200 + i)
        res["cases"].append(r)
        print(name, json.dumps(r["river"]["pred"]), "splits", r["river"]["n_splits"], flush=True)
    res["qo_quirk"] = _qo_quirk_demo(9)
    res["one_feature"] = _reg_one_feature(3000, 11)
    res["clf_observers"] = _gaussian_obs_vs_hist(args.rows // 2, 12)
    return res


def _reg_one_feature(n: int, seed: int) -> dict:
    X, y, w = reg_stream(n, 1, seed, kind="tree", noise=3.0)
    rv = river_regressor(splitter=river_splitter(("tebst", 1)), grace_period=50)
    _, nr, _ = run_prequential_reg(rv, X, y, w, mine=False)
    m = HTRegressor(1, grace_period=50)
    _, nm, _ = run_prequential_reg(m, X, y, w, mine=True)
    m0 = HTRegressor(1, grace_period=50, null_merit=0.0)
    _, n0, _ = run_prequential_reg(m0, X, y, w, mine=True)
    return {
        "rows": n,
        "noise_sd": 3.0,
        "splits_river": len(split_rows(nr)),
        "splits_mine": len(split_rows(nm)),
        "same_split_rows": split_rows(nm) == split_rows(nr),
        "splits_null_merit_0": len(split_rows(n0)),
    }


def summarise2(res) -> None:
    for r in res["cases"]:
        rv, d, a = r["river"], r["definition"], r["river_adaptive"]
        print(
            f"{r['case']}: k={r['k']} rows={r['rows']} splitter={r['splitter']}",
            f"splits={rv['n_splits']}",
        )
        print(
            "  vs river:",
            json.dumps(rv["pred"]),
            "split_rows_same",
            rv["split_rows_same"],
            json.dumps(rv["tree"]),
            f"mse={rv['mse_river']:.4f}",
            f"t river {rv['seconds_river']:.1f}s mine {rv['seconds_mine']:.1f}s",
        )
        print(
            "  observer sizes river",
            json.dumps(rv["obs_river"]),
            "mine",
            json.dumps(rv["obs_mine"]),
        )
        print("  definition:", json.dumps(d))
        print("  river adaptive leaves:", json.dumps(a))
    print("QO quirk:", json.dumps(res["qo_quirk"]))
    print("one feature:", json.dumps(res["one_feature"]))
    print("classification observers:", json.dumps(res["clf_observers"]))


# --------------------------------------------------------------------------- main

PARTS = {
    "part1": (part1, summarise1),
    "part2": (part2, summarise2),
}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("part", choices=sorted(PARTS))
    ap.add_argument("--out", default=os.path.join(".cache", "study_503"))
    ap.add_argument("--rows", type=int, default=20000)
    ap.add_argument("--py-rows", type=int, default=20000)
    ap.add_argument("--reg-rows", type=int, default=20000)
    ap.add_argument("--force", action="store_true", help="recompute a part whose result exists")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"{args.part}.json")
    run, summarise = PARTS[args.part]
    if os.path.exists(path) and not args.force:
        with open(path) as fh:
            res = json.load(fh)
        print(f"(cached: {path})")
    else:
        t0 = time.perf_counter()
        res = run(args)
        res["_seconds"] = time.perf_counter() - t0
        with open(path, "w") as fh:
            json.dump(res, fh, indent=1)
    summarise(res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
