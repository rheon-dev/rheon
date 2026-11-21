"""Study 508: five change detectors written from their papers, in plain Python floats.

Every detector reads one value a row and answers, for that row, whether it
flags a drift (and, for DDM and EDDM, a warning). The defaults follow
``river.drift`` 0.27.0 wherever river departs from the paper, so that the two
can be compared flag for flag and bit for bit; each such departure is a named
option whose paper form is also given. Python floats are IEEE doubles and each
expression below is written in the order river evaluates it, so the
comparisons can be exact.

Rules every detector keeps (the library's, not river's):

* ``update(x, w)``: a row of weight ``w == 0`` advances nothing -- not the row
  count, not the check cadence, not the random draw -- and flags nothing. Any
  ``w > 0`` is one row. River takes no weights; the checkable form is river on
  the stream with the zero-weight rows removed.
* ``save()`` returns the whole state as plain values and ``load`` restores it,
  so a detector saved mid-stream continues to the bit; ``state_arrays()``
  returns the state as the arrays a Rust layout would hold, for counting bytes.
* After a drift is flagged, the next row (of positive weight) starts from an
  empty state, as river does (``reset_on_drift=True``).

The update equations
--------------------

ADWIN (Bifet and Gavalda 2007, ADWIN2). The window ``W`` is an exponential
histogram: level ``i`` holds at most ``M`` (``max_buckets``) buckets of
``2^i`` rows each, a bucket being the pair (sum of its values, sum of squared
deviations about its own mean). A row enters level 0 as ``(x, 0)``; when a
level holds ``M + 1`` buckets its two oldest merge into one of the next level,
``(t0 + t1, v0 + v1 + n (t0/n - t1/n)^2 / 2)``, ``n = 2^i``. The window's
totals are kept incrementally:

    width += 1;  V += (width - 1) (x - S/(width - 1))^2 / width;  S += x

Every ``clock`` rows (once ``width > grace_period``) each split of the window
at a bucket boundary into an older part ``W0`` (``n0`` rows, mean ``mu0``) and
a newer ``W1`` is tested, oldest boundary first, and the cut is made when

    |mu0 - mu1| > eps_cut,
    eps_cut = sqrt(2 (1/m) (V/width) d') + (2/3) d' (1/m),   d' = ln(2 ln(width) / delta)
    1/m = 1/(n0 - L + 1) + 1/(n1 - L + 1)                    (L = min_window_length)

with both ``n0, n1 >= L``. A cut drops the oldest bucket and restarts the scan
on the shrunken window; the row is flagged if any cut was made. Dropping a
bucket ``(u, v)`` of ``n`` rows: ``width -= n; S -= u;
V -= v + n width (u/n - S/width)^2 / (n + width)``.

The bound is the paper's Bernstein form (section 4: ``eps = sqrt(2/m sigma^2
ln(2/delta')) + 2/(3m) ln(2/delta')`` with ``delta' = delta / ln(width)``),
with ``1/m`` shifted by ``L`` as MOA and river shift it. ``bound="hoeffding"``
gives the paper's Theorem 3.1 form instead, for values in [0, 1]:
``eps = sqrt((1/(2m)) ln(4/delta'))``, ``delta' = delta/width``, ``1/m =
1/n0 + 1/n1``. ``max_width`` caps the window (the oldest bucket is dropped
while ``width > max_width``), which neither the paper nor river has: without
it the state grows with the logarithm of the stream's length.

Page-Hinkley (Page 1954; Hinkley 1971; Mouss et al. 2004 for the two-sided
form). With the running mean ``xbar_t`` of the rows so far (this row included):

    m_t = alpha m_{t-1} + (x_t - xbar_t - delta);   M_t = min_{s<=t} m_s;   up:   m_t - M_t > lambda
    u_t = alpha u_{t-1} + (x_t - xbar_t + delta);   U_t = max_{s<=t} u_s;   down: U_t - u_t > lambda

tested once ``t >= min_instances``. ``alpha = 1`` is the paper; river's default
``alpha = 1 - 1e-4`` fades the sums. ``mean="recursive"`` (river) keeps
``xbar += (x - xbar)/t``; ``mean="sum"`` keeps ``sum/t``.

DDM (Gama et al. 2004). ``x_t`` is 1 for an error. ``p_t`` the error rate after
``t`` rows, ``s_t = sqrt(p_t (1 - p_t) / t)``. Once ``t > warm_start``:
if ``p_t + s_t <= p_min + s_min`` the minimum moves to ``(p_t, s_t)``; warning when
``p_t + s_t > p_min + 2 s_min``, drift when ``p_t + s_t > p_min + 3 s_min``.
``mean="recursive"`` (river) keeps ``p += (x - p)/t``; ``mean="count"`` is
the paper's ``errors / t`` (one rounding).

EDDM (Baena-Garcia et al. 2006). On each error the distance (rows) since the
previous error (or since the start) enters a running mean ``p'`` and sample
standard deviation ``s'`` (Welford). Once ``t > warm_start``: if ``p' + 2 s'``
exceeds its maximum the maximum moves; otherwise, once more than
``warm_start`` errors have been seen, ``level = (p' + 2 s') / max``; drift when
``level < beta``, warning when ``level < alpha``.

KSWIN (Raab et al. 2020). A ring of the last ``window_size`` values. Once full,
each row draws ``r = stat_size`` values at random, without replacement, from
the older ``window_size - r`` and compares them with the newest ``r`` by the
two-sample Kolmogorov-Smirnov statistic ``D = max_x |F_old(x) - F_new(x)|``.
Equal sizes make ``D = h / r`` for an integer ``h``. ``test="river"`` flags
when the exact p-value ``P(D_{r,r} >= h/r) <= alpha`` and ``D > 0.1``; the
exact p-value is the Gnedenko-Korolyuk sum
``2 sum_{j>=1} (-1)^{j+1} C(2r, r - j h) / C(2r, r)``, computed here in integers
and rounded once. ``test="paper"`` flags when ``D > sqrt(-ln(alpha) / r)``
(the paper's eq. for two windows of ``r``). ``draw="random"`` is river's draw
(CPython's ``random.Random(seed).sample``; re-seeded at each reset);
``draw="all"`` compares the newest ``r`` with all ``n = window_size - r`` older
values (no random stream in the state) and flags when ``D`` exceeds the
asymptotic critical value ``sqrt(-ln(alpha/2)/2) sqrt((n + r)/(n r))``.
"""

from __future__ import annotations

import math
import random
from collections import deque
from fractions import Fraction

import numpy as np

# ---------------------------------------------------------------------------
# ADWIN


class Adwin:
    def __init__(
        self,
        delta: float = 0.002,
        clock: int = 32,
        max_buckets: int = 5,
        min_window_length: int = 5,
        grace_period: int = 10,
        bound: str = "bernstein",
        reset_on_drift: bool = True,
        max_width: int | None = None,
    ) -> None:
        self.delta = delta
        self.clock = clock
        self.M = max_buckets
        self.L = float(min_window_length)
        self.grace = float(grace_period)
        self.bound = bound
        self.reset_on_drift = reset_on_drift
        self.max_width = max_width
        self.n_detections = 0
        self.drift = False
        self._reset()

    def _reset(self) -> None:
        self.total = 0.0
        self.var = 0.0  # sum of squared deviations (river's ``variance``)
        self.width = 0.0
        self.tick = 0
        # levels[i] = [totals list, variances list]; oldest bucket first
        self.levels: list[list[list[float]]] = [[[], []]]
        self.max_levels = 1

    # -- the window ---------------------------------------------------------

    def _insert(self, x: float) -> None:
        lv0 = self.levels[0]
        lv0[0].append(x)
        lv0[1].append(0.0)
        self.width += 1.0
        inc = 0.0
        if self.width > 1.0:
            dv = x - self.total / (self.width - 1.0)
            inc = (self.width - 1.0) * dv * dv / self.width
        self.var += inc
        self.total += x
        self._compress()

    def _compress(self) -> None:
        M = self.M
        idx = 0
        levels = self.levels
        while True:
            t, v = levels[idx]
            if len(t) != M + 1:
                break
            n1 = float(1 << idx)
            t0, t1 = t[0], t[1]
            v0, v1 = v[0], v[1]
            mu1 = t0 / n1
            mu2 = t1 / n1
            total12 = t0 + t1
            temp = n1 * 0.5 * (mu1 - mu2) * (mu1 - mu2)
            v12 = v0 + v1 + temp
            if idx + 1 >= len(levels):
                levels.append([[], []])
                if len(levels) > self.max_levels:
                    self.max_levels = len(levels)
            levels[idx + 1][0].append(total12)
            levels[idx + 1][1].append(v12)
            del t[:2]
            del v[:2]
            if len(levels[idx + 1][0]) <= M:
                break
            idx += 1

    def _delete_oldest(self) -> None:
        last = len(self.levels) - 1
        n = float(1 << last)
        t, v = self.levels[last]
        u, vv = t[0], v[0]
        mu = u / n
        self.width -= n
        self.total -= u
        muw = self.total / self.width
        inc = vv + n * self.width * (mu - muw) * (mu - muw) / (n + self.width)
        self.var -= inc
        del t[0]
        del v[0]
        if not t:
            self.levels.pop()

    def _cut(self, n0: float, n1: float, dmean: float) -> bool:
        if self.bound == "bernstein":
            dp = math.log(2.0 * math.log(self.width) / self.delta)
            m_recip = 1.0 / (n0 - self.L + 1.0) + 1.0 / (n1 - self.L + 1.0)
            eps = math.sqrt(2.0 * m_recip * (self.var / self.width) * dp) + 2.0 / 3.0 * dp * m_recip
        else:  # Theorem 3.1, values in [0, 1]
            dp = self.delta / self.width
            m_recip = 1.0 / n0 + 1.0 / n1
            eps = math.sqrt(m_recip / 2.0 * math.log(4.0 / dp))
        return abs(dmean) > eps

    def _detect(self) -> bool:
        change = False
        self.tick += 1
        if self.tick % self.clock == 0 and self.width > self.grace:
            reduce = True
            while reduce:
                reduce = False
                n0 = 0.0
                n1 = self.width
                u0 = 0.0
                u1 = self.total
                levels = self.levels
                done = False
                for idx in range(len(levels) - 1, -1, -1):
                    size = float(1 << idx)
                    t = levels[idx][0]
                    cnt = len(t)
                    for k in range(cnt):
                        n0 += size
                        n1 -= size
                        tk = t[k]
                        u0 += tk
                        u1 -= tk
                        if idx == 0 and k == cnt - 1:
                            done = True
                            break
                        dmean = (u0 / n0) - (u1 / n1)
                        if n1 >= self.L and n0 >= self.L and self._cut(n0, n1, dmean):
                            reduce = True
                            change = True
                            if self.width > 0.0:
                                self._delete_oldest()
                                done = True
                                break
                    if done:
                        break
        return change

    # -- the public row -------------------------------------------------------

    def update(self, x: float, w: float = 1.0) -> bool:
        if w == 0.0:
            return False
        if self.drift and self.reset_on_drift:
            self._reset()
        self._insert(float(x))
        if self.max_width is not None:
            while self.width > self.max_width:
                self._delete_oldest()
        self.drift = self._detect()
        if self.drift:
            self.n_detections += 1
        return self.drift

    @property
    def estimation(self) -> float:
        return 0.0 if self.width == 0 else self.total / self.width

    def n_buckets(self) -> int:
        return sum(len(lv[0]) for lv in self.levels)

    def save(self) -> dict:
        return {
            "total": self.total,
            "var": self.var,
            "width": self.width,
            "tick": self.tick,
            "n_detections": self.n_detections,
            "drift": self.drift,
            "levels": [[list(t), list(v)] for t, v in self.levels],
        }

    def load(self, s: dict) -> None:
        self.total, self.var, self.width = s["total"], s["var"], s["width"]
        self.tick, self.n_detections, self.drift = s["tick"], s["n_detections"], s["drift"]
        self.levels = [[list(t), list(v)] for t, v in s["levels"]]

    def state_arrays(self) -> dict[str, np.ndarray]:
        """River's layout: each level a fixed ``M + 1`` totals and variances and a count."""
        nl = len(self.levels)
        tot = np.zeros((nl, self.M + 1))
        var = np.zeros((nl, self.M + 1))
        cnt = np.zeros(nl, dtype=np.int64)
        for i, (t, v) in enumerate(self.levels):
            tot[i, : len(t)] = t
            var[i, : len(v)] = v
            cnt[i] = len(t)
        scal = np.array([self.total, self.var, self.width, self.tick, self.n_detections, self.drift], dtype=np.float64)
        return {"totals": tot, "variances": var, "counts": cnt, "scalars": scal}


# ---------------------------------------------------------------------------
# Page-Hinkley


class PageHinkley:
    def __init__(
        self,
        min_instances: int = 30,
        delta: float = 0.005,
        threshold: float = 50.0,
        alpha: float = 1 - 0.0001,
        mode: str = "both",
        mean: str = "recursive",
    ) -> None:
        self.min_instances = min_instances
        self.delta = delta
        self.threshold = threshold
        self.alpha = alpha
        self.mode = mode
        self.mean_form = mean
        self.drift = False
        self._reset()

    def _reset(self) -> None:
        self.n = 0.0
        self.mean = 0.0
        self.sum = 0.0
        self.s_inc = 0.0
        self.s_dec = 0.0
        self.min_inc = math.inf
        self.max_dec = -math.inf

    def update(self, x: float, w: float = 1.0) -> bool:
        if w == 0.0:
            return False
        if self.drift:
            self._reset()
            self.drift = False
        self.n += 1.0
        if self.mean_form == "recursive":
            self.mean += (1.0 / self.n) * (x - self.mean)
        else:
            self.sum += x
            self.mean = self.sum / self.n
        dev = x - self.mean
        self.s_inc = self.alpha * self.s_inc + dev - self.delta
        self.s_dec = self.alpha * self.s_dec + dev + self.delta
        if self.s_inc < self.min_inc:
            self.min_inc = self.s_inc
        if self.s_dec > self.max_dec:
            self.max_dec = self.s_dec
        if self.n >= self.min_instances:
            up = self.s_inc - self.min_inc
            down = self.max_dec - self.s_dec
            if self.mode == "up":
                self.drift = up > self.threshold
            elif self.mode == "down":
                self.drift = down > self.threshold
            else:
                self.drift = up > self.threshold or down > self.threshold
        return self.drift

    _FIELDS = ("n", "mean", "sum", "s_inc", "s_dec", "min_inc", "max_dec", "drift")

    def save(self) -> dict:
        return {f: getattr(self, f) for f in self._FIELDS}

    def load(self, s: dict) -> None:
        for f in self._FIELDS:
            setattr(self, f, s[f])

    def state_arrays(self) -> dict[str, np.ndarray]:
        fields = [f for f in self._FIELDS if f != "sum" or self.mean_form == "sum"]
        return {"scalars": np.array([float(getattr(self, f)) for f in fields])}


# ---------------------------------------------------------------------------
# DDM


class DDM:
    def __init__(
        self,
        warm_start: int = 30,
        warning_threshold: float = 2.0,
        drift_threshold: float = 3.0,
        mean: str = "recursive",
    ) -> None:
        self.warm_start = warm_start
        self.warn_t = warning_threshold
        self.drift_t = drift_threshold
        self.mean_form = mean
        self.drift = False
        self.warning = False
        self._reset()

    def _reset(self) -> None:
        self.n = 0.0
        self.errors = 0.0
        self.p = 0.0
        self.p_min = math.inf
        self.s_min = math.inf
        self.ps_min = math.inf

    def update(self, x: float, w: float = 1.0) -> bool:
        if w == 0.0:
            return False
        if self.drift:
            self._reset()
            self.drift = False
            self.warning = False
        self.n += 1.0
        if self.mean_form == "recursive":
            self.p += (1.0 / self.n) * (x - self.p)
        else:
            self.errors += x
            self.p = self.errors / self.n
        p, n = self.p, self.n
        s = math.sqrt(p * (1 - p) / n)
        if n > self.warm_start:
            if p + s <= self.ps_min:
                self.p_min = p
                self.s_min = s
                self.ps_min = self.p_min + self.s_min
            self.warning = p + s > self.p_min + self.warn_t * self.s_min
            if p + s > self.p_min + self.drift_t * self.s_min:
                self.drift = True
                self.warning = False
        return self.drift

    _FIELDS = ("n", "errors", "p", "p_min", "s_min", "ps_min", "drift", "warning")

    def save(self) -> dict:
        return {f: getattr(self, f) for f in self._FIELDS}

    def load(self, s: dict) -> None:
        for f in self._FIELDS:
            setattr(self, f, s[f])

    def state_arrays(self) -> dict[str, np.ndarray]:
        fields = [f for f in self._FIELDS if f != ("errors" if self.mean_form == "recursive" else "p")]
        return {"scalars": np.array([float(getattr(self, f)) for f in fields])}


# ---------------------------------------------------------------------------
# EDDM


class EDDM:
    def __init__(self, warm_start: int = 30, alpha: float = 0.95, beta: float = 0.9) -> None:
        self.warm_start = warm_start
        self.alpha = alpha
        self.beta = beta
        self.drift = False
        self.warning = False
        self._reset()

    def _reset(self) -> None:
        self.n = 0
        self.last_error = 0
        self.n_errors = 0
        self.dn = 0.0  # Welford over the distances: count, mean, sum of squares
        self.dmean = 0.0
        self.dS = 0.0
        self.p2s_max = -1.0

    def update(self, x: float, w: float = 1.0) -> bool:
        if w == 0.0:
            return False
        if self.drift:
            self._reset()
            self.drift = False
            self.warning = False
        self.n += 1
        if x == 1:
            self.n_errors += 1
            d = self.n - self.last_error
            old = self.dmean
            self.dn += 1.0
            self.dmean += (1.0 / self.dn) * (d - self.dmean)
            self.dS += 1.0 * (d - old) * (d - self.dmean)
            if self.n > self.warm_start:
                pi = self.dmean
                var = self.dS / (self.dn - 1) if self.dn > 1 else 0.0
                si = math.sqrt(var)
                p2s = pi + 2 * si
                if p2s > self.p2s_max:
                    self.p2s_max = p2s
                elif self.n_errors > self.warm_start:
                    level = p2s / self.p2s_max
                    if level < self.beta:
                        self.drift = True
                    elif level < self.alpha:
                        self.warning = True
                    else:
                        self.warning = False
            self.last_error = self.n
        return self.drift

    _FIELDS = ("n", "last_error", "n_errors", "dn", "dmean", "dS", "p2s_max", "drift", "warning")

    def save(self) -> dict:
        return {f: getattr(self, f) for f in self._FIELDS}

    def load(self, s: dict) -> None:
        for f in self._FIELDS:
            setattr(self, f, s[f])

    def state_arrays(self) -> dict[str, np.ndarray]:
        return {"scalars": np.array([float(getattr(self, f)) for f in self._FIELDS])}


# ---------------------------------------------------------------------------
# KSWIN


_KS_P: dict[tuple[int, int], float] = {}


def ks_pvalue_equal(r: int, h: int) -> float:
    """``P(D_{r,r} >= h/r)``, two-sided, by the Gnedenko-Korolyuk sum in integers, rounded once."""
    key = (r, h)
    if key not in _KS_P:
        if h == 0:
            _KS_P[key] = 1.0
        else:
            num = 0
            j = 1
            while r - j * h >= 0:
                num += (-1) ** (j + 1) * math.comb(2 * r, r - j * h)
                j += 1
            _KS_P[key] = float(Fraction(2 * num, math.comb(2 * r, r)))
    return _KS_P[key]


def ks_h(a: list[float], b: list[float]) -> int:
    """``r D`` for two samples of equal size ``r``: the largest gap in counts at or below a pooled value."""
    sa = sorted(a)
    sb = sorted(b)
    r = len(sa)
    i = j = 0
    h = 0
    while i < r and j < r:
        v = sa[i] if sa[i] <= sb[j] else sb[j]
        while i < r and sa[i] == v:
            i += 1
        while j < r and sb[j] == v:
            j += 1
        g = i - j if i >= j else j - i
        if g > h:
            h = g
    return h


class KSWIN:
    def __init__(
        self,
        alpha: float = 0.005,
        window_size: int = 100,
        stat_size: int = 30,
        seed: int = 0,
        test: str = "river",
        draw: str = "random",
    ) -> None:
        self.alpha = alpha
        self.W = window_size
        self.r = stat_size
        self.seed = seed
        self.test = test
        self.draw = draw
        self.drift = False
        self.crit = math.sqrt(-math.log(alpha) / stat_size)
        n, m = window_size - stat_size, stat_size
        self.crit_all = math.sqrt(-0.5 * math.log(alpha / 2.0) * (n + m) / (n * m))
        self._reset()

    def _reset(self) -> None:
        self.n = 0
        self.window: deque[float] = deque(maxlen=self.W)
        self.rng = random.Random(self.seed)
        self.p_value = 0.0
        self.stat = 0.0
        self.tested = False

    def update(self, x: float, w: float = 1.0) -> bool:
        if w == 0.0:
            return False
        if self.drift:
            self._reset()
            self.drift = False
        self.n += 1
        win = self.window
        win.append(x)
        if len(win) < self.W:
            self.tested = False
            return False
        self.tested = True
        r, W = self.r, self.W
        if self.draw == "random":
            old = [win[i] for i in self.rng.sample(range(W - r), r)]
        else:
            old = [win[i] for i in range(W - r)]
        new = [win[i] for i in range(W - r, W)]
        if self.draw == "random":
            h = ks_h(old, new)
            self.stat = h / r
            if self.test == "river":
                self.p_value = ks_pvalue_equal(r, h)
                self.drift = self.p_value <= self.alpha and self.stat > 0.1
            else:
                self.drift = self.stat > self.crit
        else:
            self.stat = ks_stat(old, new)
            self.drift = self.stat > self.crit_all
        return self.drift

    def save(self) -> dict:
        return {
            "n": self.n,
            "window": list(self.window),
            "rng": self.rng.getstate(),
            "p_value": self.p_value,
            "stat": self.stat,
            "drift": self.drift,
        }

    def load(self, s: dict) -> None:
        self.n = s["n"]
        self.window = deque(s["window"], maxlen=self.W)
        self.rng = random.Random()
        self.rng.setstate(s["rng"])
        self.p_value, self.stat, self.drift = s["p_value"], s["stat"], s["drift"]

    def state_arrays(self) -> dict[str, np.ndarray]:
        out = {
            "window": np.array(self.window, dtype=np.float64),
            "scalars": np.array([self.n, self.p_value, self.stat, self.drift], dtype=np.float64),
        }
        if self.draw == "random":
            # MT19937: 624 words of 32 bits and the position
            st = self.rng.getstate()[1]
            out["rng"] = np.array(st[:-1], dtype=np.uint32)
            out["rng_pos"] = np.array([st[-1]], dtype=np.uint32)
        return out


def ks_stat(a: list[float], b: list[float]) -> float:
    """Two-sample KS statistic for unequal sizes: ``max |i/na - j/nb|`` over pooled values."""
    sa = sorted(a)
    sb = sorted(b)
    na, nb = len(sa), len(sb)
    i = j = 0
    d = 0.0
    while i < na and j < nb:
        v = sa[i] if sa[i] <= sb[j] else sb[j]
        while i < na and sa[i] == v:
            i += 1
        while j < nb and sb[j] == v:
            j += 1
        g = abs(i / na - j / nb)
        if g > d:
            d = g
    return d


DETECTORS = {"adwin": Adwin, "ph": PageHinkley, "ddm": DDM, "eddm": EDDM, "kswin": KSWIN}


def run(det, xs, ws=None) -> np.ndarray:
    """Feed a stream; the per-row drift flag."""
    out = np.zeros(len(xs), dtype=bool)
    upd = det.update
    if ws is None:
        for i, x in enumerate(xs):
            out[i] = upd(x)
    else:
        for i, (x, w) in enumerate(zip(xs, ws)):
            out[i] = upd(x, w)
    return out
