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
import io
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
    __slots__ = ("counts", "obs", "depth", "last", "mc", "nb", "active", "capped")

    def __init__(self, depth: int, counts: list, sum_=None):
        self.counts = counts
        self.obs = None  # k * C observers, allocated at the leaf's first learned row
        self.depth = depth
        self.last = (sum_ or naive_sum)(counts)
        self.mc = 0.0
        self.nb = 0.0
        self.active = True
        self.capped = False  # deactivated by the memory cap (may be reactivated)

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
        self.rows_learned = 0

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
        self.weight_seen += w
        if w == 0.0 and self.zero_weight == "skip":
            return  # learns nothing, not even the label
        c = self.class_index(y)
        self.rows_learned += 1
        if self.cap is not None and self.rows_learned % self.recheck == 0:
            self._enforce_cap()
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
    def struct_bytes(self, n_leaves: int, n_branches: int) -> int:
        return clf_struct_bytes(self.C, n_leaves, n_branches)

    def obs_reserve(self) -> int:
        return 40 * self.k * self.C

    def promise(self, leaf) -> float:
        # river's: the weight the majority class gets wrong
        return self._sum(leaf.counts) - max(leaf.counts)


# --------------------------------------------------------------------------- state and cap

# The packed layout a build would save (little-endian, no padding). Its length
# is the state size reported in part 3; `pack` and `unpack` are also the save
# and resume of part 4.
#
#   classifier  header 40 + 8 C   (version, k, C, leaves, branches, root,
#                                  weight seen, rows learned, C labels)
#               branch 36         (feature u32, threshold, left i32, right i32,
#                                  left and right weight)
#               leaf   8 C + 27   (C class weights, last attempt, mc and nb
#                                  correct weight, depth u16, flags u8)
#               observers 40 k C  (per active leaf that has learned: k x C x
#                                  [W, mean, S, min, max])
#   regressor   header 40, branch 20 (no weights: nothing routes by them),
#               leaf 35 (W, mean, S, last attempt, depth, flags),
#               QO observer 2 + 36 per slot (slot index i32, x mean, W, y mean, y S),
#               E-BST observer 4 + 33 per node (value, W, mean, S, child flags).

CLF_BRANCH, REG_BRANCH = 36, 20


def clf_struct_bytes(C: int, n_leaves: int, n_branches: int) -> int:
    return 40 + 8 * C + CLF_BRANCH * n_branches + (8 * C + 27) * n_leaves


def reg_struct_bytes(n_leaves: int, n_branches: int) -> int:
    return 40 + REG_BRANCH * n_branches + 35 * n_leaves


def clf_state_bytes(t) -> int:
    n_obs = sum(1 for lf in t.leaves if lf.obs is not None)
    return clf_struct_bytes(t.C, len(t.leaves), len(t.branches)) + 40 * t.k * t.C * n_obs


def reg_state_bytes(t) -> int:
    b = reg_struct_bytes(len(t.leaves), len(t.branches))
    for lf in t.leaves:
        if lf.obs is not None:
            for ob in lf.obs:
                b += 2 + 36 * ob.n_slots() if isinstance(ob, QO) else 4 + 33 * ob.n_nodes()
    return b


def cap_set(self, max_bytes: int, recheck: int = 1000, promise: str = "river") -> None:
    """Bound the packed state by `max_bytes`. A split is refused when the
    structure alone (every leaf inactive) would pass the cap; after every split
    and every `recheck` learned rows, the leaves are ranked by promise and only
    as many keep observers as the bytes left by the structure pay for at their
    largest (40 k C for the Gaussian observers, k (2 + 36 max_slots) for QO).
    A leaf deactivated by the cap keeps its class weights (target statistics)
    and predicts from them; a leaf ranked back in gets empty observers."""
    self.cap = int(max_bytes)
    self.recheck = int(recheck)
    self.promise_kind = promise
    self._enforce_cap()


def cap_may_grow(self) -> bool:
    return self.struct_bytes(len(self.leaves) + 1, len(self.branches) + 1) <= self.cap


def cap_enforce(self) -> None:
    budget = self.cap - self.struct_bytes(len(self.leaves), len(self.branches))
    max_active = max(0, budget // self.obs_reserve())
    elig = [i for i, lf in enumerate(self.leaves) if lf.active or lf.capped]
    elig.sort(key=lambda i: self.promise(self.leaves[i]))  # stable: ties keep leaf order
    cutoff = len(elig) - min(max_active, len(elig))
    for i in elig[:cutoff]:
        lf = self.leaves[i]
        if lf.active:
            lf.active = False
            lf.capped = True
            lf.obs = None
    for i in elig[cutoff:]:
        lf = self.leaves[i]
        if lf.capped:
            lf.active = True
            lf.capped = False
            lf.obs = None


def _flags(lf) -> int:
    return (1 if lf.active else 0) | (2 if lf.obs is not None else 0) | (4 if lf.capped else 0)


def pack_clf(t: HTClassifier) -> bytes:
    out = io.BytesIO()
    labels = [-1] * t.C
    for lab, i in t.labels.items():
        labels[i] = int(lab)
    out.write(
        struct.pack(
            "<IIIIIidQ",
            503,
            t.k,
            t.C,
            len(t.leaves),
            len(t.branches),
            t.root,
            t.weight_seen,
            t.rows_learned,
        )
    )
    out.write(struct.pack(f"<{t.C}q", *labels))
    for b in t.branches:
        out.write(struct.pack("<Idiidd", b[0], b[1], b[2], b[3], b[4], b[5]))
    for lf in t.leaves:
        out.write(
            struct.pack(f"<{t.C + 3}dHB", *lf.counts, lf.last, lf.mc, lf.nb, lf.depth, _flags(lf))
        )
    for lf in t.leaves:
        if lf.obs is not None:
            out.write(np.asarray(lf.obs, dtype="<f8").tobytes())
    return out.getvalue()


def unpack_clf(buf: bytes, **params) -> HTClassifier:
    v, k, C, L, B, root, ws, rl = struct.unpack_from("<IIIIIidQ", buf, 0)
    assert v == 503
    t = HTClassifier(k, C, **params)
    off = 40
    labels = struct.unpack_from(f"<{C}q", buf, off)
    off += 8 * C
    t.labels = {lab: i for i, lab in enumerate(labels) if lab != -1}
    t.root, t.weight_seen, t.rows_learned = root, ws, rl
    t.branches = []
    for _ in range(B):
        f, th, le, ri, lw, rw = struct.unpack_from("<Idiidd", buf, off)
        off += CLF_BRANCH
        t.branches.append([f, th, le, ri, lw, rw])
    t.leaves = []
    has_obs = []
    for _ in range(L):
        vals = struct.unpack_from(f"<{C + 3}dHB", buf, off)
        off += 8 * C + 27
        lf = CLeaf(vals[C + 3], list(vals[:C]))
        lf.last, lf.mc, lf.nb = vals[C], vals[C + 1], vals[C + 2]
        fl = vals[C + 4]
        lf.active, lf.capped = bool(fl & 1), bool(fl & 4)
        has_obs.append(bool(fl & 2))
        t.leaves.append(lf)
    n_o = 5 * k * C
    for lf, h in zip(t.leaves, has_obs, strict=True):
        if h:
            a = np.frombuffer(buf, dtype="<f8", count=n_o, offset=off)
            off += 8 * n_o
            lf.obs = a.reshape(k * C, 5).tolist()
    assert off == len(buf)
    return t


def pack_reg(t: HTRegressor) -> bytes:
    out = io.BytesIO()
    out.write(
        struct.pack(
            "<IIIIIidQ",
            504,
            t.k,
            0,
            len(t.leaves),
            len(t.branches),
            t.root,
            t.weight_seen,
            t.rows_learned,
        )
    )
    for b in t.branches:
        out.write(struct.pack("<Idii", b[0], b[1], b[2], b[3]))
    for lf in t.leaves:
        out.write(struct.pack("<4dHB", *lf.stats, lf.last, lf.depth, _flags(lf)))
    for lf in t.leaves:
        if lf.obs is None:
            continue
        for ob in lf.obs:
            if isinstance(ob, QO):
                out.write(struct.pack("<H", len(ob.slots)))
                for i in sorted(ob.slots):
                    xn, xm, v = ob.slots[i]
                    assert xn == v[0]
                    out.write(struct.pack("<i4d", i, xm, v[0], v[1], v[2]))
            else:
                nodes = []
                stack = [ob.root] if ob.root is not None else []
                while stack:
                    nd = stack.pop()
                    nodes.append(nd)
                    stack.extend(c for c in (nd[3], nd[2]) if c is not None)
                out.write(struct.pack("<I", len(nodes)))
                for nd in nodes:
                    fl = (1 if nd[2] is not None else 0) | (2 if nd[3] is not None else 0)
                    out.write(struct.pack("<4dB", nd[0], *nd[1], fl))
    return out.getvalue()


def unpack_reg(buf: bytes, **params) -> HTRegressor:
    v, k, _, L, B, root, ws, rl = struct.unpack_from("<IIIIIidQ", buf, 0)
    assert v == 504
    t = HTRegressor(k, **params)
    off = 40
    t.root, t.weight_seen, t.rows_learned = root, ws, rl
    t.branches = []
    for _ in range(B):
        f, th, le, ri = struct.unpack_from("<Idii", buf, off)
        off += REG_BRANCH
        t.branches.append([f, th, le, ri])
    t.leaves, has_obs = [], []
    for _ in range(L):
        n, m, S_, last, depth, fl = struct.unpack_from("<4dHB", buf, off)
        off += 35
        lf = RLeaf(depth, [n, m, S_])
        lf.last = last
        lf.active, lf.capped = bool(fl & 1), bool(fl & 4)
        has_obs.append(bool(fl & 2))
        t.leaves.append(lf)
    for lf, h in zip(t.leaves, has_obs, strict=True):
        if not h:
            continue
        lf.obs = []
        for _ in range(k):
            ob = t._new_obs()
            if isinstance(ob, QO):
                (ns,) = struct.unpack_from("<H", buf, off)
                off += 2
                for _ in range(ns):
                    i, xm, n, ym, yS = struct.unpack_from("<i4d", buf, off)
                    off += 36
                    ob.slots[i] = [n, xm, [n, ym, yS]]
            else:
                (nn,) = struct.unpack_from("<I", buf, off)
                off += 4

                def read(off_):
                    val, n, m, S_, fl = struct.unpack_from("<4dB", buf, off_)
                    return [val, [n, m, S_], None, None], fl, off_ + 33

                if nn:
                    root_nd, fl, off = read(off)
                    ob.root = root_nd
                    stack = [(root_nd, fl)]
                    # preorder, left before right: rebuild with an explicit stack
                    pending = []
                    nd, fl = root_nd, fl
                    while True:
                        if fl & 1:
                            if fl & 2:
                                pending.append(nd)
                            child, cfl, off = read(off)
                            nd[2] = child
                            nd, fl = child, cfl
                            continue
                        if fl & 2:
                            child, cfl, off = read(off)
                            nd[3] = child
                            nd, fl = child, cfl
                            continue
                        if not pending:
                            break
                        par = pending.pop()
                        child, cfl, off = read(off)
                        par[3] = child
                        nd, fl = child, cfl
                    del stack
            lf.obs.append(ob)
    assert off == len(buf), (off, len(buf))
    return t


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


def _pow2(d: float) -> float:
    return d**2  # river's `** 2` is C pow(d, 2.0): not always d * d (part 4)


SQUARE = {"f": _pow2}


def v_iadd(a: list, b: list) -> list:
    S = a[2] + b[2] + SQUARE["f"](a[1] - b[1]) * a[0] * b[0] / (a[0] + b[0])
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
    a[2] = a[2] - b[2] - SQUARE["f"](a[1] - b[1]) * a[0] * b[0] / (a[0] + b[0])
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
    __slots__ = ("stats", "obs", "depth", "last", "active", "capped")

    def __init__(self, depth: int, stats: list):
        self.stats = stats
        self.obs = None
        self.depth = depth
        self.last = stats[0]
        self.active = True
        self.capped = False


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
        self.rows_learned = 0

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
        self.rows_learned += 1
        if self.cap is not None and self.rows_learned % self.recheck == 0:
            self._enforce_cap()
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

    def struct_bytes(self, n_leaves: int, n_branches: int) -> int:
        return reg_struct_bytes(n_leaves, n_branches)

    def obs_reserve(self) -> int:
        if self.splitter[0] != "qo" or len(self.splitter) < 3 or self.splitter[2] is None:
            raise ValueError("a byte cap needs a bounded observer: ('qo', radius, max_slots)")
        return self.k * (2 + 36 * self.splitter[2])

    def promise(self, leaf) -> float:
        if self.promise_kind == "sse":
            return leaf.stats[2]  # the leaf's sum of squared deviations
        return -leaf.depth  # river's: shallow leaves first


for _cls, _sb in ((HTClassifier, clf_state_bytes), (HTRegressor, reg_state_bytes)):
    _cls.set_cap = cap_set
    _cls._may_grow = cap_may_grow
    _cls._enforce_cap = cap_enforce
    _cls.state_bytes = _sb


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


def _observer_growth(seed: int) -> dict:
    """Observer size at one leaf that never splits, against rows learned: E-BST
    keeps a node per distinct value, TE-BST per distinct rounded value, QO a
    slot per occupied interval of width radius."""
    X, y, _ = reg_stream(40_000, 1, seed, kind="tree")
    obs = {"ebst": EBST(None), "tebst_1": EBST(1), "tebst_2": EBST(2), "qo_0.25": QO(0.25)}
    marks = [1_000, 5_000, 20_000, 40_000]
    out = {name: [] for name in obs}
    for i in range(len(X)):
        for o in obs.values():
            o.update(float(X[i, 0]), float(y[i]), 1.0)
        if i + 1 in marks:
            for name, o in obs.items():
                out[name].append(o.n_nodes() if isinstance(o, EBST) else o.n_slots())
    out["rows"] = marks
    return out


def _rounding_forms(n: int, seed: int) -> dict:
    """TE-BST rounds with Python's round(x, digits) (correctly rounded decimal);
    the obvious Rust form (x * 10^d).round() / 10^d is not the same function."""
    rng = np.random.default_rng(seed)
    v = (rng.standard_normal(n) * 3.0).tolist()
    out = {}
    for d in (1, 2, 3):
        sc = 10.0**d
        diff = 0
        for x in v:
            a = round(x, d)
            b = math.floor(abs(x) * sc + 0.5) / sc
            b = math.copysign(b, x)
            diff += a != b
        out[f"digits_{d}"] = diff
    out["n"] = n
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
    res["observer_growth"] = _observer_growth(13)
    res["rounding"] = _rounding_forms(200_000, 14)
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
    print("observer growth at one leaf:", json.dumps(res["observer_growth"]))
    print("rounding forms that differ from round(x, d):", json.dumps(res["rounding"]))


# --------------------------------------------------------------------------- part 3

CAPS = [None, 1 << 20, 256 << 10, 64 << 10]


def _cap_name(c) -> str:
    return "none" if c is None else f"{c >> 10} KB"


def _river_size(model) -> int:
    from river.tree.utils import calculate_object_size

    return int(calculate_object_size(model))


def _run_capped_clf(X, y, tkw, cap, n_tail, check_every=2000):
    k = X.shape[1]
    t = HTClassifier(k, 2, **tkw)
    if cap is not None:
        t.set_cap(cap)
    hits = np.zeros(len(X), dtype=bool)
    worst = 0
    t0 = time.perf_counter()
    for i in range(len(X)):
        xi = X[i].tolist()
        if i:
            hits[i] = t.predict(xi) == y[i]
        t.learn(xi, int(y[i]), 1.0)
        if i % check_every == 0 or i == len(X) - 1:
            worst = max(worst, t.state_bytes())
    b = pack_clf(t)
    assert len(b) == t.state_bytes()
    return {
        "cap": _cap_name(cap),
        "accuracy": float(hits[1:].mean()),
        "accuracy_tail": float(hits[-n_tail:].mean()),
        "leaves": len(t.leaves),
        "active": sum(lf.active for lf in t.leaves),
        "capped": sum(lf.capped for lf in t.leaves),
        "bytes_end": len(b),
        "bytes_max_checked": worst,
        "within_cap": cap is None or worst <= cap,
        "seconds": time.perf_counter() - t0,
    }


def _run_river_clf(X, y, tkw, cap, n_tail, period):
    kw = dict(tkw)
    kw["max_size"] = 1e9 if cap is None else cap / 2**20
    kw["memory_estimate_period"] = period
    m = river_classifier(False, **kw)
    hits = np.zeros(len(X), dtype=bool)
    t0 = time.perf_counter()
    for i in range(len(X)):
        xi = dict(enumerate(X[i].tolist()))
        if i:
            hits[i] = m.predict_one(xi) == y[i]
        m.learn_one(xi, int(y[i]))
    return {
        "cap": _cap_name(cap),
        "accuracy": float(hits[1:].mean()),
        "accuracy_tail": float(hits[-n_tail:].mean()),
        "leaves": m.n_leaves,
        "active": m.n_active_leaves,
        "river_size_end": _river_size(m),
        "seconds": time.perf_counter() - t0,
    }


def _run_capped_reg(X, y, tkw, spl, cap, promise, n_tail, check_every=2000):
    k = X.shape[1]
    t = HTRegressor(k, splitter=spl, **tkw)
    if cap is not None:
        t.set_cap(cap, promise=promise)
    err = np.zeros(len(X))
    worst = 0
    t0 = time.perf_counter()
    for i in range(len(X)):
        xi = X[i].tolist()
        err[i] = (t.predict(xi) - y[i]) ** 2
        t.learn(xi, float(y[i]), 1.0)
        if i % check_every == 0 or i == len(X) - 1:
            worst = max(worst, t.state_bytes())
    b = pack_reg(t)
    assert len(b) == t.state_bytes()
    return {
        "cap": _cap_name(cap),
        "promise": promise,
        "mse": float(err[1:].mean()),
        "mse_tail": float(err[-n_tail:].mean()),
        "leaves": len(t.leaves),
        "active": sum(lf.active for lf in t.leaves),
        "bytes_end": len(b),
        "bytes_max_checked": worst,
        "within_cap": cap is None or worst <= cap,
        "seconds": time.perf_counter() - t0,
    }


def _run_river_reg(X, y, tkw, radius, cap, n_tail, period):
    kw = dict(tkw)
    kw["max_size"] = 1e9 if cap is None else cap / 2**20
    kw["memory_estimate_period"] = period
    m = river_regressor(splitter=river_splitter(("qo", radius)), **kw)
    err = np.zeros(len(X))
    t0 = time.perf_counter()
    for i in range(len(X)):
        xi = dict(enumerate(X[i].tolist()))
        err[i] = (m.predict_one(xi) - y[i]) ** 2
        m.learn_one(xi, float(y[i]))
    return {
        "cap": _cap_name(cap),
        "mse": float(err[1:].mean()),
        "mse_tail": float(err[-n_tail:].mean()),
        "leaves": m.n_leaves,
        "active": m.n_active_leaves,
        "river_size_end": _river_size(m),
        "seconds": time.perf_counter() - t0,
    }


def _state_table(seed: int) -> dict:
    """The packed size of grown trees against the formula, and river's own
    measure (Python object sizes) of river's trees grown on the same rows."""
    rows = []
    for k, C in ((5, 2), (20, 2), (20, 5)):
        X, y, w = clf_stream(20_000, k, C, seed + k + C, depth=8, noise=0.02)
        tkw = {"grace_period": 50, "tau": 0.1}
        t = HTClassifier(k, C, **tkw)
        for i in range(len(X)):
            t.learn(X[i].tolist(), int(y[i]), 1.0)
        b = pack_clf(t)
        n_obs = sum(lf.obs is not None for lf in t.leaves)
        formula = clf_struct_bytes(C, len(t.leaves), len(t.branches)) + 40 * k * C * n_obs
        rp = river_classifier(False, **tkw)
        for i in range(len(X)):
            rp.learn_one(dict(enumerate(X[i].tolist())), int(y[i]))
        rows.append(
            {
                "tree": "classifier, Gaussian",
                "k": k,
                "classes": C,
                "leaves": len(t.leaves),
                "with_observers": n_obs,
                "packed": len(b),
                "formula": formula,
                "river_leaves": rp.n_leaves,
                "river_object_size": _river_size(rp),
            }
        )
    for spl in (("qo", 0.25, None), ("tebst", 1), ("ebst",)):
        k = 10
        X, y, w = reg_stream(20_000, k, seed + 50, kind="tree")
        tkw = {"grace_period": 100}
        t = HTRegressor(k, splitter=spl, **tkw)
        for i in range(len(X)):
            t.learn(X[i].tolist(), float(y[i]), 1.0)
        b = pack_reg(t)
        sizes = mine_obs_sizes(t)
        rv = river_regressor(splitter=river_splitter(spl), **tkw)
        for i in range(len(X)):
            rv.learn_one(dict(enumerate(X[i].tolist())), float(y[i]))
        rows.append(
            {
                "tree": f"regressor, {spl[0]}",
                "k": k,
                "leaves": len(t.leaves),
                "with_observers": sum(lf.obs is not None for lf in t.leaves),
                "bins_mean": sizes["mean"],
                "bins_max": sizes["max"],
                "packed": len(b),
                "formula": reg_state_bytes(t),
                "river_leaves": rv.n_leaves,
                "river_object_size": _river_size(rv),
            }
        )
    return {"rows": rows}


def hyperplane_stream(n: int, k: int, seed: int, informative: int = 10, noise: float = 0.05):
    """y = 1[a'x > 0] with `informative` nonzero coefficients, labels redrawn
    with probability `noise`: a boundary a tree approximates better with every
    leaf it can afford."""
    rng = np.random.default_rng(seed)
    a = np.zeros(k)
    a[:informative] = rng.standard_normal(informative)
    X = rng.standard_normal((n, k))
    y = (X @ a > 0).astype(int)
    flip = rng.random(n) < noise
    y = np.where(flip, rng.integers(0, 2, n), y)
    return X, y, np.ones(n)


def part3(args) -> dict:
    res = {"state": _state_table(300)}
    print("state table done", flush=True)
    n = args.cap_rows
    n_tail = n // 4
    tkw = {"grace_period": 50, "tau": 0.1}
    res["clf"] = []
    for name, (X, y, _) in (
        ("random tree, depth 10", clf_stream(n, 50, 2, 301, depth=10, noise=0.05)),
        ("hyperplane, 10 of 50 informative", hyperplane_stream(n, 50, 303)),
    ):
        d = {"stream": name, "rows": n, "k": 50, "classes": 2, "tree": tkw, "mine": [], "river": []}
        for cap in CAPS:
            r = _run_capped_clf(X, y, tkw, cap, n_tail)
            d["mine"].append(r)
            print("clf mine", name, json.dumps(r), flush=True)
        for cap in CAPS:
            r = _run_river_clf(X, y, tkw, cap, n_tail, args.mem_period)
            d["river"].append(r)
            print("clf river", name, json.dumps(r), flush=True)
        res["clf"].append(d)
    # regressor: k = 20, QO radius 0.25 at most 32 slots a feature
    rkw = {"grace_period": 50}
    spl = ("qo", 0.25, 32)
    res["reg"] = []
    for name, (X, y, _) in (
        ("random tree, depth 10", reg_stream(n, 20, 302, kind="tree", depth=10)),
        ("Friedman #1, 5 of 20 informative", reg_stream(n, 20, 304, kind="friedman")),
    ):
        d = {
            "stream": name,
            "rows": n,
            "k": 20,
            "splitter": list(spl),
            "tree": rkw,
            "mine": [],
            "river": [],
        }
        for cap in CAPS:
            for promise in ("river", "sse") if cap is not None else ("river",):
                r = _run_capped_reg(X, y, rkw, spl, cap, promise, n_tail)
                d["mine"].append(r)
                print("reg mine", name, json.dumps(r), flush=True)
        for cap in CAPS:
            r = _run_river_reg(X, y, rkw, 0.25, cap, n_tail, args.mem_period)
            d["river"].append(r)
            print("reg river", name, json.dumps(r), flush=True)
        res["reg"].append(d)
    return res


def summarise3(res) -> None:
    print("state table:")
    for r in res["state"]["rows"]:
        print("  ", json.dumps(r))
    for kind in ("clf", "reg"):
        for d in res[kind]:
            print(f"{kind}: {d['stream']} rows={d['rows']} k={d['k']} tree={d['tree']}")
            for r in d["mine"]:
                print("  mine ", json.dumps(r))
            for r in d["river"]:
                print("  river", json.dumps(r))


# --------------------------------------------------------------------------- part 4


def _chunks(n: int, n_chunks: int, rng) -> list:
    """n_chunks chunk boundaries at random rows (1 chunk: the whole stream)."""
    if n_chunks <= 1:
        return [0, n]
    cuts = np.sort(rng.choice(np.arange(1, n), size=n_chunks - 1, replace=False))
    return [0, *cuts.tolist(), n]


def _feed(make, pack, unpack, X, y, w, n_chunks: int, resume: bool, seed: int, reg: bool):
    """Prequential over chunks; with `resume`, the state is packed to bytes and
    a new tree unpacked from them at every chunk boundary."""
    rng = np.random.default_rng(seed)
    bounds = _chunks(len(X), n_chunks, rng)
    t = make()
    preds = []
    for a, b in zip(bounds[:-1], bounds[1:], strict=True):
        Xc, yc, wc = X[a:b].tolist(), y[a:b].tolist(), w[a:b].tolist()
        for xi, yi, wi in zip(Xc, yc, wc, strict=True):
            if reg:
                preds.append(t.predict(xi))
            else:
                p = t.predict_proba(xi)
                preds.append(tuple(sorted(p.items())))
            t.learn(xi, yi if reg else int(yi), wi)
        if resume:
            t = unpack(pack(t))
    return pack(t), preds


def _determinism(n: int) -> dict:
    out = {}
    X, y, w = clf_stream(n, 8, 3, 401, depth=7, weights="uniform")
    w[::17] = 0.0  # zero-weight rows among them
    params = {"grace_period": 50, "tau": 0.1}

    def mk_c(cap=None):
        def make():
            t = HTClassifier(8, 3, **params)
            if cap:
                t.set_cap(cap, recheck=500)
            return t

        def unpack(b):
            t = unpack_clf(b, **params)
            if cap:
                t.cap, t.recheck, t.promise_kind = cap, 500, "river"
            return t

        return make, unpack

    Xr, yr, wr = reg_stream(n, 8, 402, kind="friedman", weights="uniform")
    wr[::17] = 0.0
    wr[0] = 0.0  # a zero-weight first row
    rparams = {"grace_period": 50}

    def mk_r(spl, cap=None):
        def make():
            t = HTRegressor(8, splitter=spl, **rparams)
            if cap:
                t.set_cap(cap, recheck=500, promise="sse")
            return t

        def unpack(b):
            t = unpack_reg(b, splitter=spl, **rparams)
            if cap:
                t.cap, t.recheck, t.promise_kind = cap, 500, "sse"
            return t

        return make, unpack

    configs = {
        "classifier": (*mk_c(), pack_clf, X, y, w, False),
        "classifier, cap 3 KB": (*mk_c(3 << 10), pack_clf, X, y, w, False),
        "regressor, TE-BST": (*mk_r(("tebst", 1)), pack_reg, Xr, yr, wr, True),
        "regressor, E-BST": (*mk_r(("ebst",)), pack_reg, Xr, yr, wr, True),
        "regressor, QO 32 slots, cap 64 KB": (
            *mk_r(("qo", 0.25, 32), 64 << 10),
            pack_reg,
            Xr,
            yr,
            wr,
            True,
        ),
    }
    for name, (make, unpack, pack, Xs, ys, ws, reg) in configs.items():
        ref_state, ref_pred = _feed(make, pack, unpack, Xs, ys, ws, 1, False, 0, reg)
        rows = []
        for n_chunks in (7, 1000):
            for resume in (False, True):
                st, pr = _feed(make, pack, unpack, Xs, ys, ws, n_chunks, resume, n_chunks, reg)
                rows.append(
                    {
                        "chunks": n_chunks,
                        "resume_each_chunk": resume,
                        "state_identical": st == ref_state,
                        "predictions_identical": pr == ref_pred,
                    }
                )
        t_ref = make()
        for xi, yi, wi in zip(Xs.tolist(), ys.tolist(), ws.tolist(), strict=True):
            t_ref.learn(xi, yi if reg else int(yi), wi)
        out[name] = {
            "state_bytes": len(ref_state),
            "leaves": len(t_ref.leaves),
            "capped_leaves": sum(lf.capped for lf in t_ref.leaves),
            "runs": rows,
        }
    return out


def _river_resume(n: int) -> dict:
    """river itself: pickled mid-stream, loaded, continued (its Rust and Python
    classifier paths and its regressor)."""
    import pickle

    out = {}
    X, y, w = clf_stream(n, 5, 2, 403, depth=6)
    Xr, yr, _ = reg_stream(n, 5, 404)
    for name, make, Xs, ys in (
        ("classifier, Rust path", lambda: river_classifier(True, grace_period=50), X, y),
        ("classifier, Python path", lambda: river_classifier(False, grace_period=50), X, y),
        ("regressor, TE-BST", lambda: river_regressor(grace_period=50), Xr, yr),
    ):
        reg = name.startswith("regressor")

        def run(cut, make=make, Xs=Xs, ys=ys, reg=reg):
            m = make()
            pr = []
            for i in range(len(Xs)):
                if i == cut:
                    m = pickle.loads(pickle.dumps(m))
                xi = dict(enumerate(Xs[i].tolist()))
                pr.append(
                    m.predict_one(xi) if reg else tuple(sorted(m.predict_proba_one(xi).items()))
                )
                m.learn_one(xi, float(ys[i]) if reg else int(ys[i]))
            return pr

        out[name] = {"rows": n, "cut": n // 2, "identical": run(None) == run(n // 2)}
    return out


def _zero_weight(n: int) -> dict:
    """A zero-weight row learns nothing here; what it does in river."""
    rng = np.random.default_rng(405)
    out = {}
    # the stream, and the same stream with zero-weight rows spliced in (first row too)
    X, y, w = clf_stream(n, 5, 2, 406, depth=6)
    n_zero = n // 10
    Z = rng.standard_normal((n_zero, 5)) * 3.0  # wide values: they move a min or a max
    zy = rng.integers(0, 2, n_zero)
    pos = np.sort(rng.choice(np.arange(1, n), size=n_zero - 1, replace=False))
    pos = np.concatenate([[0], pos])
    Xz = np.insert(X, pos, Z, axis=0)
    yz = np.insert(y, pos, zy)
    wz = np.insert(w, pos, 0.0)
    keep = wz > 0
    tkw = {"grace_period": 50}
    # the tree here: the spliced stream's state equals the plain stream's, to the byte
    a = HTClassifier(5, 2, **tkw)
    b = HTClassifier(5, 2, **tkw)
    pa, pb = [], []
    for i in range(len(X)):
        pa.append(a.predict_proba(X[i].tolist()))
        a.learn(X[i].tolist(), int(y[i]), 1.0)
    for i in range(len(Xz)):
        p = b.predict_proba(Xz[i].tolist())
        if keep[i]:
            pb.append(p)
        b.learn(Xz[i].tolist(), int(yz[i]), float(wz[i]))
    sa, sb = pack_clf(a), pack_clf(b)
    out["mine_classifier"] = {
        "zero_rows": n_zero,
        "state_identical": sa == sb,
        "predictions_identical": pa == pb,
        "weight_seen_equal": a.weight_seen == b.weight_seen,
    }
    # river (Rust path): it refuses a zero weight
    ra = river_classifier(True, **tkw)
    qa = []
    for i in range(len(X)):
        xi = dict(enumerate(X[i].tolist()))
        qa.append(ra.predict_proba_one(xi))
        ra.learn_one(xi, int(y[i]), w=1.0)
    ta = river_preorder(ra)
    rb = river_classifier(True, **tkw)
    for i in range(5):
        rb.learn_one(dict(enumerate(X[i].tolist())), int(y[i]), w=1.0)
    try:
        rb.learn_one(dict(enumerate(X[5].tolist())), int(y[5]), w=0.0)
        out["river_classifier_rust"] = "accepted"
    except ValueError as e:
        out["river_classifier_rust"] = f"ValueError: {e}"
    # river's Python path on the spliced stream: a zero-weight row adds its class
    # to the leaf's table with weight 0, whose naive Bayes vote is then 0.0
    rp = river_classifier(False, **tkw)
    qp = []
    for i in range(len(Xz)):
        xi = dict(enumerate(Xz[i].tolist()))
        p = rp.predict_proba_one(xi)
        if keep[i]:
            qp.append(p)
        rp.learn_one(xi, int(yz[i]), w=float(wz[i]))
    out["river_classifier_python"] = {
        "predictions_differ_from_plain": compare_rows(qa, qp)["rows_differ"],
        "same_tree_as_plain": river_preorder(rp) == ta,
    }
    # river's regressor: a zero-weight first row, and later ones
    Xr, yr, _ = reg_stream(2000, 3, 407)
    res = {}
    for name, spl in (("tebst", ("tebst", 1)), ("qo", ("qo", 0.25))):
        m = river_regressor(splitter=river_splitter(spl))
        try:
            m.learn_one(dict(enumerate(Xr[0].tolist())), float(yr[0]), w=0.0)
            res[f"{name}_first_row"] = "ok"
        except ZeroDivisionError as e:
            res[f"{name}_first_row"] = f"ZeroDivisionError: {e}"
        m = river_regressor(splitter=river_splitter(spl))
        err = None
        for i in range(1, 1000):
            m.learn_one(dict(enumerate(Xr[i].tolist())), float(yr[i]))
        for i in range(1000, 2000):
            try:
                m.learn_one(dict(enumerate((Xr[i] * 4.0).tolist())), float(yr[i]), w=0.0)
            except ZeroDivisionError as e:
                err = f"row {i}: ZeroDivisionError: {e}"
                break
        res[f"{name}_later_rows"] = err or "ok"
    # the adaptive leaf's faded errors move on a zero-weight row
    m = river_regressor(leaf_prediction="adaptive", grace_period=100_000)
    for i in range(1, 300):
        m.learn_one(dict(enumerate(Xr[i].tolist())), float(yr[i]))
    leaf = m._root if not hasattr(m._root, "children") else None
    before = (leaf._fmse_mean, leaf._fmse_model) if leaf is not None else None
    m.learn_one(dict(enumerate(Xr[400].tolist())), float(yr[400]) + 10.0, w=0.0)
    after = (leaf._fmse_mean, leaf._fmse_model) if leaf is not None else None
    res["adaptive_faded_errors"] = {"before": before, "after_zero_weight_row": after}
    out["river_regressor"] = res
    # the regressor here: zero-weight rows, first included, learn nothing
    Xr2, yr2, _ = reg_stream(n, 5, 408, kind="friedman")
    wr = np.ones(n)
    wr[0] = 0.0
    wr[rng.choice(np.arange(1, n), n // 10, replace=False)] = 0.0
    for spl in (("tebst", 1), ("qo", 0.25, 32)):
        a = HTRegressor(5, splitter=spl, grace_period=50)
        b = HTRegressor(5, splitter=spl, grace_period=50)
        for i in range(n):
            if wr[i] > 0:
                a.learn(Xr2[i].tolist(), float(yr2[i]), 1.0)
            b.learn(Xr2[i].tolist(), float(yr2[i]), float(wr[i]))
        out[f"mine_regressor_{spl[0]}"] = {"state_identical": pack_reg(a) == pack_reg(b)}
    return out


def _weights(n: int) -> dict:
    """Integer weights against duplicated rows, and a rescaling of every weight."""
    out = {}
    X, y, _ = clf_stream(n, 5, 2, 409, depth=6)
    rng = np.random.default_rng(410)
    wi = rng.integers(1, 4, n).astype(float)
    tkw = {"grace_period": 50}
    a = HTClassifier(5, 2, **tkw)
    for i in range(n):
        a.learn(X[i].tolist(), int(y[i]), wi[i])
    b = HTClassifier(5, 2, **tkw)
    for i in range(n):
        for _ in range(int(wi[i])):
            b.learn(X[i].tolist(), int(y[i]), 1.0)
    Xt, yt, _ = clf_stream(20_000, 5, 2, 409, depth=6)  # same concept (seed), new rows

    def acc(t) -> float:  # on 5,000 rows of the same concept that were not learned
        rows = zip(Xt[-5000:].tolist(), yt[-5000:], strict=True)
        return float(np.mean([t.predict(r) == c for r, c in rows]))

    out["integer_vs_duplicated"] = {
        "rows": n,
        "nodes_weighted": a.n_nodes(),
        "nodes_duplicated": b.n_nodes(),
        "same_tree": a.preorder() == b.preorder(),
        "thresholds_max_ulps": compare_trees(a.preorder(), b.preorder())["threshold_max_ulps"],
        "accuracy_weighted": acc(a),
        "accuracy_duplicated": acc(b),
    }
    for scale in (0.5, 2.0, 10.0):
        c = HTClassifier(5, 2, **tkw)
        for i in range(n):
            c.learn(X[i].tolist(), int(y[i]), scale)
        d = HTClassifier(5, 2, grace_period=50 * scale)
        for i in range(n):
            d.learn(X[i].tolist(), int(y[i]), scale)
        out[f"all_weights_x{scale}"] = {
            "nodes": c.n_nodes(),
            "nodes_at_weight_1": HTClassifier_nodes(X, y, tkw),
            "nodes_with_grace_scaled": d.n_nodes(),
            "same_tree_with_grace_scaled": d.preorder() == _plain_tree(X, y, tkw).preorder(),
        }
    return out


def _plain_tree(X, y, tkw):
    t = HTClassifier(X.shape[1], 2, **tkw)
    for i in range(len(X)):
        t.learn(X[i].tolist(), int(y[i]), 1.0)
    return t


def HTClassifier_nodes(X, y, tkw) -> int:  # noqa: N802
    return _plain_tree(X, y, tkw).n_nodes()


def _time_rows(fn, budget_s: float) -> float:
    """Seconds per row of fn(i) over as many rows as fit the budget (at least 2,000)."""
    t0 = time.perf_counter()
    i = 0
    while True:
        fn(i)
        i += 1
        if i >= 2000 and time.perf_counter() - t0 > budget_s:
            break
    return (time.perf_counter() - t0) / i


def _cost(budget_s: float) -> dict:
    """Per row (predict, then learn) on a grown tree: river's Rust classifier,
    river's Python paths, and the tree here (Python)."""
    out = []
    for k in (5, 20, 50):
        X, y, _ = clf_stream(60_000, k, 2, 411 + k, depth=8, noise=0.02)
        warm = 30_000
        rows = {"k": k}
        for name, make, mine in (
            ("river_rust_classifier", lambda: river_classifier(True, grace_period=200), False),
            ("river_python_classifier", lambda: river_classifier(False, grace_period=200), False),
            ("here_classifier", lambda k=k: HTClassifier(k, 2, grace_period=200), True),
        ):
            m = make()
            xs = [X[i].tolist() for i in range(len(X))]
            xd = [dict(enumerate(r)) for r in xs]
            for i in range(warm):
                if mine:
                    m.learn(xs[i], int(y[i]))
                else:
                    m.learn_one(xd[i], int(y[i]))

            def step(i, m=m, mine=mine, xs=xs, xd=xd, y=y, warm=warm):
                j = warm + i % (len(xs) - warm)
                if mine:
                    m.predict_proba(xs[j])
                    m.learn(xs[j], int(y[j]))
                else:
                    m.predict_proba_one(xd[j])
                    m.learn_one(xd[j], int(y[j]))

            rows[name + "_us"] = 1e6 * _time_rows(step, budget_s)
            rows[name + "_leaves"] = m.n_leaves if not mine else len(m.leaves)
        out.append(rows)
    # the regressor (river has only its Python path)
    for k in (5, 20):
        X, y, _ = reg_stream(60_000, k, 420 + k, kind="friedman")
        warm = 30_000
        rows = {"k": k}
        for name, make, mine in (
            ("river_python_regressor_tebst", lambda: river_regressor(), False),
            (
                "river_python_regressor_qo",
                lambda: river_regressor(splitter=river_splitter(("qo", 0.25))),
                False,
            ),
            ("here_regressor_qo32", lambda k=k: HTRegressor(k, splitter=("qo", 0.25, 32)), True),
        ):
            m = make()
            xs = [X[i].tolist() for i in range(len(X))]
            xd = [dict(enumerate(r)) for r in xs]
            for i in range(warm):
                if mine:
                    m.learn(xs[i], float(y[i]))
                else:
                    m.learn_one(xd[i], float(y[i]))

            def step(i, m=m, mine=mine, xs=xs, xd=xd, y=y, warm=warm):
                j = warm + i % (len(xs) - warm)
                if mine:
                    m.predict(xs[j])
                    m.learn(xs[j], float(y[j]))
                else:
                    m.predict_one(xd[j])
                    m.learn_one(xd[j], float(y[j]))

            rows[name + "_us"] = 1e6 * _time_rows(step, budget_s)
        out.append(rows)
    return {"rows": out}


def _square_forms(n: int, n_rows: int) -> dict:
    """How often C's pow(d, 2) differs from d * d, and what d * d in the
    regressor's parallel variance formulas does to the comparison with river."""
    rng = np.random.default_rng(412)
    v = np.concatenate([rng.standard_normal(n), rng.standard_normal(n) * 1e3]).tolist()
    out = {"values": len(v), "pow_ne_mul": int(sum((x**2) != (x * x) for x in v))}
    cases = []
    try:
        SQUARE["f"] = lambda d: d * d
        for i, (name, k, skw, tkw, spl) in enumerate(REG_CASES):
            X, y, w = reg_stream(n_rows, k, 200 + i, **skw)
            rv = river_regressor(splitter=river_splitter(spl), **tkw)
            pr, nr, _ = run_prequential_reg(rv, X, y, w, mine=False)
            m = HTRegressor(k, splitter=spl, **tkw)
            pm, nm, _ = run_prequential_reg(m, X, y, w, mine=True)
            cases.append(
                {
                    "case": name,
                    "pred": compare_pred(pm, pr),
                    "split_rows_same": split_rows(nm) == split_rows(nr),
                    "tree": compare_trees(m.preorder(), river_preorder(rv)),
                }
            )
    finally:
        SQUARE["f"] = _pow2
    out["regressor_with_mul"] = cases
    return out


def part4(args) -> dict:
    res = {"determinism": _determinism(args.det_rows)}
    print("determinism", json.dumps(res["determinism"]), flush=True)
    res["river_resume"] = _river_resume(5000)
    print("river resume", json.dumps(res["river_resume"]), flush=True)
    res["zero_weight"] = _zero_weight(10_000)
    print("zero weight", json.dumps(res["zero_weight"]), flush=True)
    res["weights"] = _weights(10_000)
    print("weights", json.dumps(res["weights"]), flush=True)
    res["square"] = _square_forms(1_000_000, args.reg_rows)
    print("square", json.dumps(res["square"]), flush=True)
    res["cost"] = _cost(args.time_budget)
    return res


def summarise4(res) -> None:
    for name, d in res["determinism"].items():
        print(f"determinism, {name} ({d['state_bytes']} bytes):")
        for r in d["runs"]:
            print("   ", json.dumps(r))
    print("river resume:", json.dumps(res["river_resume"]))
    for name, d in res["zero_weight"].items():
        print(f"zero weight, {name}:", json.dumps(d))
    for name, d in res["weights"].items():
        print(f"weights, {name}:", json.dumps(d))
    sq = res["square"]
    print(f"pow(d, 2) != d * d on {sq['pow_ne_mul']} of {sq['values']} values")
    for r in sq["regressor_with_mul"]:
        print("  regressor with d * d vs river:", json.dumps(r))
    for r in res["cost"]["rows"]:
        print(
            "cost per row (us):",
            json.dumps({a: round(b, 1) if isinstance(b, float) else b for a, b in r.items()}),
        )


# --------------------------------------------------------------------------- main

PARTS = {
    "part1": (part1, summarise1),
    "part2": (part2, summarise2),
    "part3": (part3, summarise3),
    "part4": (part4, summarise4),
}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("part", choices=sorted(PARTS))
    ap.add_argument("--out", default=os.path.join(".cache", "study_503"))
    ap.add_argument("--rows", type=int, default=20000)
    ap.add_argument("--py-rows", type=int, default=20000)
    ap.add_argument("--reg-rows", type=int, default=20000)
    ap.add_argument("--cap-rows", type=int, default=200_000)
    ap.add_argument(
        "--mem-period", type=int, default=5000, help="river's memory_estimate_period in part 3"
    )
    ap.add_argument("--det-rows", type=int, default=20_000)
    ap.add_argument("--time-budget", type=float, default=3.0, help="seconds per timed case")
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
