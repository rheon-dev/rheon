"""Task 518: window operators. Oracle: Polars itself.

A hard window of clock length `window` over the rows' stamps, as `rolling_*_by`:

    closed="right"  (t - window, t]      "left"  [t - window, t)
    closed="both"   [t - window, t]      "none"  (t - window, t)

The row's own value is in the window for `right` and `both`, out of it for `left` and `none`.
A zero-weight row advances the clock (it evicts) and enters no window (a null in the oracle).
Weights and decay do not apply: a hard window counts each row once. (A weighted quantile would
be a different quantity and has no oracle.)

State, in rows held `R` (the rows inside the window, capped by `max_rows`):
    rolling min / max   monotone deque of (stamp, value):                16 R bytes at most
    quantile / rank     ring of (stamp, value) + sorted values:          24 R bytes
    `ewm_mean`          `W` and the mean per column, no ring:             8 (1 + k) bytes
"""

from __future__ import annotations

import bisect
import math
from collections import deque

import numpy as np
import polars as pl

from .common import lam_of, table


def _evict_bound(t, window, closed):
    """Rows with stamp <= bound have left (right, none) or < bound (left, both)."""
    return t - window


def _gone(ti, t, window, closed):
    return ti <= t - window if closed in ("right", "none") else ti < t - window


class RollingExtreme:
    """`rolling_min_by` / `rolling_max_by`: a monotone deque, O(1) amortised, exact."""

    def __init__(self, window, closed="right", min_samples=1, kind="min"):
        self.window, self.closed, self.min_samples, self.kind = window, closed, min_samples, kind
        self.dq: deque = deque()  # (stamp, value, count_before_mask)
        self.ring: deque = deque()  # all live stamps, to count samples
        self.max_held = 0

    def step(self, t, v, live=True):
        sgn = 1.0 if self.kind == "min" else -1.0
        while self.dq and _gone(self.dq[0][0], t, self.window, self.closed):
            self.dq.popleft()
        while self.ring and _gone(self.ring[0], t, self.window, self.closed):
            self.ring.popleft()
        include = self.closed in ("right", "both")
        if include and live:
            self._push(t, sgn * v)
        n = len(self.ring)
        res = sgn * self.dq[0][1] if (self.dq and n >= self.min_samples) else math.nan
        if not include and live:
            self._push(t, sgn * v)
        self.max_held = max(self.max_held, len(self.dq))
        return res

    def _push(self, t, v):
        while self.dq and self.dq[-1][1] >= v:
            self.dq.pop()
        self.dq.append((t, v))
        self.ring.append(t)


class RollingSorted:
    """`rolling_quantile_by` and `rolling_rank_by`: a ring plus a sorted list of values."""

    def __init__(self, window, closed="right", min_samples=1):
        self.window, self.closed, self.min_samples = window, closed, min_samples
        self.ring: deque = deque()  # (stamp, value)
        self.sorted: list = []
        self.max_held = 0

    def _evict(self, t):
        while self.ring and _gone(self.ring[0][0], t, self.window, self.closed):
            _, old = self.ring.popleft()
            del self.sorted[bisect.bisect_left(self.sorted, old)]

    def step(self, t, v, live, read):
        self._evict(t)
        include = self.closed in ("right", "both")
        if include and live:
            self._push(t, v)
        res = read(self.sorted, v) if len(self.sorted) >= self.min_samples else math.nan
        if not include and live:
            self._push(t, v)
        self.max_held = max(self.max_held, len(self.ring))
        return res

    def _push(self, t, v):
        self.ring.append((t, v))
        bisect.insort(self.sorted, v)


def quantile_read(q, interpolation):
    def read(w, _v):
        m = len(w)
        if m == 0:
            return math.nan
        h = (m - 1) * q
        if interpolation == "nearest":
            return w[int(math.floor(h + 0.5))]
        if interpolation == "higher":
            return w[int(math.ceil(h))]
        if interpolation == "lower":
            return w[int(math.floor(h))]
        if interpolation == "midpoint":
            return (w[int(math.floor(h))] + w[int(math.ceil(h))]) / 2
        if interpolation == "linear":
            lo, hi = int(math.floor(h)), int(math.ceil(h))
            return w[lo] + (h - lo) * (w[hi] - w[lo])
        if interpolation == "equiprobable":
            return w[max(int(math.ceil(m * q)) - 1, 0)]
        raise ValueError(interpolation)

    return read


def rank_read(method):
    def read(w, v):
        lo = bisect.bisect_left(w, v)
        hi = bisect.bisect_right(w, v)
        if method == "average":
            return (lo + 1 + hi) / 2
        if method == "min":
            return lo + 1
        if method == "max":
            return hi
        if method == "dense":
            return len(set(w[: hi])) if False else _dense(w, v)
        raise ValueError(method)

    return read


def _dense(w, v):
    # number of distinct values <= v
    out, prev = 0, None
    for x in w:
        if x > v:
            break
        if x != prev:
            out += 1
            prev = x
    return out


def stamps(seed, n, hi=4):
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.integers(1, hi, n)).astype(np.int64)


def _series(seed, n, zero_frac=0.1, ties=True):
    rng = np.random.default_rng(seed + 31)
    t = stamps(seed, n)
    v = rng.standard_normal(n)
    if ties:
        v = v.round(1)
    live = rng.random(n) >= zero_frac
    live[0] = False  # a zero-weight first row
    return t, v, live


def _pl(df, expr):
    return df.select(expr).to_series().to_numpy()


def report(seeds: int = 20) -> None:
    n = 300
    window = 25
    # ---- min / max
    rows = []
    for kind in ("min", "max"):
        for closed in ("right", "left", "both", "none"):
            for ms in (1, 3):
                worst = 0.0
                mism = 0
                for sd in range(seeds):
                    t, v, live = _series(sd, n)
                    vn = np.where(live, v, np.nan)
                    df = pl.DataFrame({"t": t, "v": pl.Series(vn).fill_nan(None)})
                    f = getattr(pl.col("v"), f"rolling_{kind}_by")
                    ref = _pl(df, f("t", f"{window}i", min_samples=ms, closed=closed))
                    ref = np.where(ref == None, np.nan, ref).astype(float)  # noqa: E711
                    op = RollingExtreme(window, closed, ms, kind)
                    got = np.array([op.step(t[i], v[i], live[i]) for i in range(n)])
                    both_nan = np.isnan(got) & np.isnan(ref)
                    mism += int((np.isnan(got) != np.isnan(ref)).sum())
                    d = np.where(both_nan, 0.0, np.abs(got - ref))
                    worst = max(worst, float(np.nanmax(d)))
                rows.append([f"{kind}, closed={closed}, min_samples={ms}", worst, mism])
    table("518a `rolling_min_by` / `rolling_max_by` (integer stamps, window 25, 10% zero-weight rows as nulls, a zero-weight first row, ties)",
          ["case", "largest gap", "null-pattern mismatches"], rows)

    # ---- quantile
    rows = []
    for interp in ("nearest", "higher", "lower", "midpoint", "linear", "equiprobable"):
        worst = 0.0
        mism = 0
        for closed in ("right", "both", "left"):
            for q in (0.1, 0.5, 0.9):
                for sd in range(max(3, seeds // 4)):
                    t, v, live = _series(sd, n)
                    vn = np.where(live, v, np.nan)
                    df = pl.DataFrame({"t": t, "v": pl.Series(vn).fill_nan(None)})
                    ref = _pl(df, pl.col("v").rolling_quantile_by("t", f"{window}i", quantile=q, interpolation=interp, closed=closed))
                    ref = np.where(ref == None, np.nan, ref).astype(float)  # noqa: E711
                    op = RollingSorted(window, closed)
                    rd = quantile_read(q, interp)
                    got = np.array([op.step(t[i], v[i], live[i], rd) for i in range(n)])
                    mism += int((np.isnan(got) != np.isnan(ref)).sum())
                    worst = max(worst, float(np.nanmax(np.where(np.isnan(got) & np.isnan(ref), 0, np.abs(got - ref)))))
        rows.append([interp, worst, mism])
    table("518b `rolling_quantile_by` over q in {.1,.5,.9} and closed in {right,both,left}", ["interpolation", "largest gap", "null-pattern mismatches"], rows)

    # ---- rank
    rows = []
    for method in ("average", "min", "max", "dense"):
        worst = 0.0
        mism = 0
        for closed in ("right", "both"):
            for sd in range(max(3, seeds // 4)):
                t, v, live = _series(sd, n)
                vn = np.where(live, v, np.nan)
                df = pl.DataFrame({"t": t, "v": pl.Series(vn).fill_nan(None)})
                ref = _pl(df, pl.col("v").rolling_rank_by("t", f"{window}i", method=method, closed=closed))
                ref = np.where(ref == None, np.nan, ref).astype(float)  # noqa: E711
                op = RollingSorted(window, closed)
                rd = rank_read(method)
                got = np.array([op.step(t[i], v[i], live[i], rd) if live[i] else math.nan for i in range(n)])
                # a null row's own rank is null in the oracle
                mism += int((np.isnan(got) != np.isnan(ref)).sum())
                worst = max(worst, float(np.nanmax(np.where(np.isnan(got) | np.isnan(ref), 0, np.abs(got - ref)))))
        rows.append([method, worst, mism])
    table("518c `rolling_rank_by` (the row's rank in its window; `random` is not reproducible, not compared)",
          ["method", "largest gap", "null-pattern mismatches"], rows)

    # ---- ewm_mean, count-weighted
    rows = []
    for adjust in (True, False):
        for hl in (3.0, 40.0):
            worst = 0.0
            for sd in range(seeds):
                rng = np.random.default_rng(sd)
                x = rng.standard_normal(400) + 5.0
                live = rng.random(400) >= 0.1
                live[0] = False
                vn = np.where(live, x, np.nan)
                df = pl.DataFrame({"v": pl.Series(vn).fill_nan(None)})
                ref = _pl(df, pl.col("v").ewm_mean(half_life=hl, adjust=adjust, ignore_nulls=False, min_samples=1))
                ref = np.where(ref == None, np.nan, ref).astype(float)  # noqa: E711
                lam = 0.5 ** (1.0 / hl)
                al = 1.0 - lam
                W, m, y, g = 0.0, 0.0, None, 0
                got = []
                for i in range(400):
                    if adjust:
                        # mean form, `lam` per row: W' = lam W + 1, m' = m + (1/W') (x - m)
                        Wo = lam * W
                        if live[i]:
                            W2 = Wo + 1.0
                            m = x[i] if Wo == 0.0 else m + (1.0 / W2) * (x[i] - m)
                            W = W2
                        else:
                            W = Wo  # a time step that learns nothing
                        got.append(m if W > 0 else math.nan)
                    else:
                        # fixed mass on the row: y' = (lam^g y + alpha x) / (lam^g + alpha), g rows since the last
                        g += 1
                        if live[i]:
                            if y is None:
                                y = x[i]
                            else:
                                o = lam**g
                                y = (o * y + al * x[i]) / (o + al)
                            g = 0
                        got.append(math.nan if y is None else y)
                got = np.array(got)
                # until the first live row the oracle is null
                first = int(np.argmax(live))
                got[:first] = math.nan
                worst = max(worst, float(np.nanmax(np.abs(got - ref) / np.abs(ref))))
            rows.append([f"adjust={adjust}, half_life={hl} rows", worst])
    table("518d count-weighted `ewm_mean` (decay per row, a zero-weight row is a null with `ignore_nulls=False`)",
          ["case", "largest relative gap"], rows)

    # ---- state / cap
    print("\n518 state added, by the rows held R inside the window (a clock window holds as many "
          "rows as the arrival rate puts in it, so it needs `max_rows`):")
    wt = []
    for kind, f in (("min/max deque", lambda r: 16 * r), ("quantile / rank ring + sorted", lambda r: 24 * r)):
        wt.append([kind, f(100), f(1000), f(10000), f(100000)])
    table("518e bytes at R = 100 / 1000 / 10000 / 100000", ["operator", "R=100", "R=1000", "R=10000", "R=100000"], wt)
    print("  256 KB per-group cap: min/max R <= 16384, quantile/rank R <= 10922; `ewm_mean` 8(1+k): k=20: 168 bytes, no ring.")
    # measured max deque length on a monotone stream (worst case) vs random
    t = np.arange(5000, dtype=np.int64)
    worst_inc = RollingExtreme(200, "right", 1, "min")
    for i in range(5000):
        worst_inc.step(t[i], float(i), True)
    rnd = RollingExtreme(200, "right", 1, "min")
    rng = np.random.default_rng(0)
    for i in range(5000):
        rnd.step(t[i], rng.standard_normal(), True)
    print(f"  deque held, window 200 rows: monotone increasing (worst case, min) {worst_inc.max_held}; random {rnd.max_held}")
