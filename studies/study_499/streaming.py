"""One-pass forms of the estimators, each a state of numpy arrays fed rows in
chunks. The state is everything that survives between chunks: its size is
``sum(a.nbytes)`` over the dict, which the write-up's formulas are checked
against; the configuration (scales, weights, window) is fixed before the
first row and not counted. The kernels are numba-compiled so that the cost
per tick is a compiled loop's, not the interpreter's.

Feeding one chunk or a thousand must give the same bits (the library's
chunk invariance); ``compare.py`` checks it for each.
"""

from __future__ import annotations

import math

import estimators as E
import numba
import numpy as np


def nbytes(state: dict) -> int:
    return int(sum(a.nbytes for a in state.values()))


def chunks(n: int, parts: int, seed: int = 0):
    """``parts`` contiguous slices of ``range(n)`` at seeded cut points."""
    if parts <= 1:
        return [slice(0, n)]
    rng = np.random.default_rng(seed)
    cuts = np.sort(rng.choice(np.arange(1, n), size=min(parts - 1, n - 1), replace=False))
    edges = np.concatenate([[0], cuts, [n]])
    return [slice(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:], strict=True)]


# --------------------------------------------------------------------------
# the scale estimators on a synchronous grid: rows are price vectors
# --------------------------------------------------------------------------


@numba.njit(cache=False)
def _lags_feed(ring, cnt, acc, prices, lags, wts, which):
    """For each row ``p_i``: for each scale ``K = lags[s]`` with ``i >= K``,
    ``acc[which[s]] += wts[s] * (p_i - p_{i-K})(p_i - p_{i-K})'``. ``ring``
    holds the last ``len(ring)`` prices; ``cnt[0]`` the prices seen."""
    L, d = ring.shape
    for r in range(prices.shape[0]):
        i = cnt[0]
        p = prices[r]
        for s in range(lags.shape[0]):
            K = lags[s]
            if i >= K:
                q = ring[(i - K) % L]
                a = acc[which[s]]
                w = wts[s]
                for u in range(d):
                    du = p[u] - q[u]
                    for v in range(d):
                        a[u, v] += w * du * (p[v] - q[v])
        ring[i % L] = p
        cnt[0] = i + 1


def tsrc_new(d: int, K: int, J: int = 1) -> dict:
    return {
        "ring": np.zeros((K, d)),
        "cnt": np.zeros(1, np.int64),
        "acc": np.zeros((2, d, d)),  # sum of K-differences' outer products; J's
    }


def tsrc_feed(st: dict, prices: np.ndarray, K: int, J: int = 1) -> None:
    lags = np.array([K, J], np.int64)
    _lags_feed(
        st["ring"], st["cnt"], st["acc"], prices, lags, np.ones(2), np.array([0, 1], np.int64)
    )


def tsrc_close(st: dict, K: int, J: int = 1, n_conv: str = "hfhd") -> np.ndarray:
    n = int(st["cnt"][0]) - 1
    m = n + 1 if n_conv == "hfhd" else n
    nk, nj = (m - K + 1) / K, (m - J + 1) / J
    return st["acc"][0] / K - nk / nj * st["acc"][1] / J


def msrc_new(d: int, M: int, N: int) -> dict:
    return {
        "ring": np.zeros((max(N + M, 1), d)),
        "cnt": np.zeros(1, np.int64),
        # the weighted sum over the M scales, and the two end scales' raw sums
        "acc": np.zeros((3, d, d)),
    }


def _msrc_cfg(M: int, N: int):
    if M == 1:
        return np.array([1], np.int64), np.ones(1), np.zeros(1, np.int64)
    Ks, a = E.msrc_weights(M, N)
    lags = np.concatenate([Ks.astype(np.int64), [N + 1, N + M]])
    wts = np.concatenate([a / Ks, [1.0, 1.0]])
    which = np.concatenate([np.zeros(M, np.int64), [1, 2]])
    return lags, wts, which


def msrc_feed(st: dict, prices: np.ndarray, M: int, N: int) -> None:
    lags, wts, which = _msrc_cfg(M, N)
    _lags_feed(st["ring"], st["cnt"], st["acc"], prices, lags, wts, which)


def msrc_close(st: dict, M: int, N: int) -> np.ndarray:
    if M == 1:
        return st["acc"][0].copy()
    n = int(st["cnt"][0])  # hfhd's n: the count of prices
    zeta = (M + N) * (N + 1) / ((n + 1) * (M - 1))
    return st["acc"][0] + zeta * (st["acc"][1] / (N + 1) - st["acc"][2] / (N + M))


# --------------------------------------------------------------------------
# pre-averaging
# --------------------------------------------------------------------------


@numba.njit(cache=False)
def _mrc_feed(ring, last, cnt, acc, prices, w, first):
    """Rows are prices; returns ``r_j = p_j - p_{j-1}`` enter a ring of the
    last ``k - 1``. Each full window gives ``Ybar = sum_l w[l] r_{.+l}`` and
    ``acc[0] += Ybar Ybar'``; every return gives ``acc[1] += r r'``.
    ``cnt``: prices seen, windows formed. ``first`` windows are skipped
    (hfhd's convention skips one)."""
    km1, d = ring.shape
    ybar = np.empty(d)
    for row in range(prices.shape[0]):
        p = prices[row]
        if cnt[0] > 0:
            j = cnt[0] - 1  # this return's index from 0
            slot = j % km1
            for u in range(d):
                ring[slot, u] = p[u] - last[u]
            r = ring[slot]
            for u in range(d):
                for v in range(d):
                    acc[1, u, v] += r[u] * r[v]
            if j + 1 >= km1:
                win = j + 1 - km1  # the window's index from 0
                if win >= first:
                    for u in range(d):
                        ybar[u] = 0.0
                    for li in range(km1):
                        q = ring[(win + li) % km1]
                        for u in range(d):
                            ybar[u] += w[li] * q[u]
                    for u in range(d):
                        for v in range(d):
                            acc[0, u, v] += ybar[u] * ybar[v]
                    cnt[1] += 1
        for u in range(d):
            last[u] = p[u]
        cnt[0] += 1


def mrc_new(d: int, k: int) -> dict:
    return {
        "ring": np.zeros((k - 1, d)),
        "last": np.zeros(d),
        "cnt": np.zeros(2, np.int64),
        "acc": np.zeros((2, d, d)),
    }


def mrc_feed(st: dict, prices: np.ndarray, k: int, windows: str = "paper") -> None:
    w = E.g_min(np.arange(1, k) / k)
    _mrc_feed(
        st["ring"], st["last"], st["cnt"], st["acc"], prices, w, 0 if windows == "paper" else 1
    )


def mrc_close(st, k, bias_correction, psi1_upto_k=True, rescale=True, theta=None):
    n = int(st["cnt"][0]) - 1
    p2 = E.psi2(k)
    est = n / (n - k + 2) / (p2 * k) * st["acc"][0]
    if bias_correction:
        th = k / math.sqrt(n) if theta is None else theta
        b = E.psi1(k, psi1_upto_k) / (th**2 * p2)
        est = est - b * st["acc"][1] / (2 * n)
        if rescale:
            est = est / (1.0 - b / (2 * n))
    return est


# --------------------------------------------------------------------------
# realised kernel at a fixed bandwidth, no jitter (rheon's rcov has the jitter)
# --------------------------------------------------------------------------


@numba.njit(cache=False)
def _krvm_feed(ring, last, cnt, acc, prices, w, per_lag):
    """Returns enter a ring of the last ``L = len(w) - 1``. Each return ``x``
    adds ``w[0] x x'`` and, for each lag ``h`` the ring holds,
    ``w[h] (x y' + y x')`` with ``y`` the return ``h`` back. ``per_lag``
    keeps each lag's ``Gamma_h`` apart (what an automatic bandwidth, chosen
    at close, needs); otherwise the weighted sum is the one accumulator."""
    L, d = ring.shape
    x = np.empty(d)
    for row in range(prices.shape[0]):
        p = prices[row]
        if cnt[0] > 0:
            j = cnt[0] - 1
            for u in range(d):
                x[u] = p[u] - last[u]
            a0 = acc[0]
            for u in range(d):
                for v in range(d):
                    a0[u, v] += (1.0 if per_lag else w[0]) * x[u] * x[v]
            top = min(L, j)
            for h in range(1, top + 1):
                y = ring[(j - h) % L]
                if per_lag:
                    a = acc[h]
                    for u in range(d):
                        for v in range(d):
                            a[u, v] += x[u] * y[v]
                else:
                    wh = w[h]
                    for u in range(d):
                        for v in range(d):
                            a0[u, v] += wh * (x[u] * y[v] + y[u] * x[v])
            if L > 0:
                ring[j % L] = x
        for u in range(d):
            last[u] = p[u]
        cnt[0] += 1


def krvm_weights(H: float, kernel: str, form: str, lags: int | None = None) -> np.ndarray:
    """``w[h]`` for ``h = 0 .. L``: Parzen's support ends where its weight is
    0; the quadratic spectral's does not, so ``lags`` truncates it."""
    kern = E.KERNELS[kernel]
    out = [1.0]
    h = 1
    while True:
        if lags is not None and h > lags:
            break
        wh = float(kern(h / (H + 1))) if form == "bnhls" else float(kern((h - 1) / H))
        if wh == 0.0 and kernel == "parzen":
            break
        out.append(wh)
        h += 1
        if lags is None and h > 10**6:
            raise ValueError("unbounded kernel: give lags")
    return np.array(out)


def krvm_new(d: int, w: np.ndarray, per_lag: bool = False) -> dict:
    L = len(w) - 1
    return {
        "ring": np.zeros((L, d)),
        "last": np.zeros(d),
        "cnt": np.zeros(1, np.int64),
        "acc": np.zeros((L + 1 if per_lag else 1, d, d)),
    }


def krvm_feed(st, prices, w, per_lag=False):
    _krvm_feed(st["ring"], st["last"], st["cnt"], st["acc"], prices, w, per_lag)


def krvm_close(st, w, per_lag=False):
    if not per_lag:
        return st["acc"][0].copy()
    out = st["acc"][0].copy()
    for h in range(1, len(w)):
        out += w[h] * (st["acc"][h] + st["acc"][h].T)
    return out


# --------------------------------------------------------------------------
# a long frame of ticks: one row per (stamp, asset, log-price), time-ordered
# --------------------------------------------------------------------------


def long_frame(times: list[np.ndarray], prices: list[np.ndarray]):
    t = np.concatenate(times)
    a = np.concatenate([np.full(len(x), i, np.int64) for i, x in enumerate(times)])
    p = np.concatenate(prices)
    o = np.lexsort((a, t))
    return t[o].astype(np.int64), a[o], p[o]


@numba.njit(cache=False)
def _rt_feed(grp, last, seen, out_t, out_v, n_out, t, a, p, final):
    """Refresh time. A stamp's rows are applied together (``grp[0]`` is the
    open stamp; ``grp[1]`` whether one is open): only when a later stamp
    arrives, or at ``final``, is the open stamp tested, and if every asset
    has ticked since the last refresh it becomes one, with each asset's
    latest price. One stamp of look-ahead is the only delay."""
    k = last.shape[0]
    for r in range(t.shape[0] + (1 if final else 0)):
        closing = r == t.shape[0]
        if grp[1] == 1 and (closing or t[r] != grp[0]):
            allseen = True
            for j in range(k):
                if seen[j] == 0:
                    allseen = False
            if allseen:
                out_t[n_out[0]] = grp[0]
                for j in range(k):
                    out_v[n_out[0], j] = last[j]
                    seen[j] = 0
                n_out[0] += 1
            grp[1] = 0
        if closing:
            break
        grp[0] = t[r]
        grp[1] = 1
        last[a[r]] = p[r]
        seen[a[r]] = 1


def rt_new(k: int) -> dict:
    return {
        "grp": np.zeros(2, np.int64),
        "last": np.zeros(k),
        "seen": np.zeros(k, np.uint8),
        "n_out": np.zeros(1, np.int64),
    }


def rt_run(t, a, p, k, parts=1):
    st = rt_new(k)
    out_t = np.zeros(len(t), np.int64)
    out_v = np.zeros((len(t), k))
    sl = chunks(len(t), parts)
    for i, s in enumerate(sl):
        _rt_feed(
            st["grp"],
            st["last"],
            st["seen"],
            out_t,
            out_v,
            st["n_out"],
            t[s],
            a[s],
            p[s],
            i == len(sl) - 1,
        )
    n = int(st["n_out"][0])
    return out_t[:n], out_v[:n], st


@numba.njit(cache=False)
def _hy_feed(st, cum, has, hy, t, a, p, final):
    """Hayashi-Yoshida for the pair (A = asset 0, B = asset 1), one pass.

    ``cum[j]``: asset j's running sum of returns since its first tick (its
    path, built as ``np.cumsum(np.diff(p))`` is). ``st``: 0 the open stamp,
    1 whether one is open, 2/3 A's/B's tick in it (flags), 4/5 their
    prices, 6 A's last price, 7 B's last price, 8 A's anchor (B's path as
    of A's last tick), 9 the first pending A return, 10 its anchor, 11 the
    sum of later pending A returns (anchored at B's path now), 12 the count
    pending (0, 1, 2 = more).

    A's return over ``(s, e]`` meets B's returns from the first with
    ``tb > s`` to the first with ``tb >= e``; their sum is B's path at the
    first tick at or after ``e`` less its path at the last tick at or before
    ``s``. So a closed A return waits, pending, for B's next tick. Within a
    stamp: A's return closes, then B's tick completes what is pending, then
    A's anchor is read, which is hfhd's (and the definition's) treatment
    of a shared stamp."""
    for r in range(t.shape[0] + (1 if final else 0)):
        closing = r == t.shape[0]
        if st[1] == 1.0 and (closing or t[r] != st[0]):
            if st[2] == 1.0:  # A ticked at this stamp
                if has[0] == 1:
                    x = st[4] - st[6]
                    cum[0] += x
                    hy[1] += x * x
                    if st[12] == 0.0:
                        st[9] = x
                        st[10] = st[8]
                        st[12] = 1.0
                    else:
                        st[11] += x
                        st[12] = 2.0
                st[6] = st[4]
                has[0] = 1
            if st[3] == 1.0:  # B ticked
                ynew = cum[1]
                if has[1] == 1:
                    y = st[5] - st[7]
                    ynew = cum[1] + y
                    hy[2] += y * y
                if st[12] >= 1.0:
                    hy[0] += st[9] * (ynew - st[10])
                if st[12] == 2.0:
                    hy[0] += st[11] * (ynew - cum[1])
                st[9] = 0.0
                st[11] = 0.0
                st[12] = 0.0
                cum[1] = ynew
                st[7] = st[5]
                has[1] = 1
            if st[2] == 1.0:
                st[8] = cum[1]
            st[1] = 0.0
            st[2] = 0.0
            st[3] = 0.0
        if closing:
            if st[12] >= 1.0:
                hy[0] += st[9] * (cum[1] - st[10])
            if st[12] == 2.0:
                hy[0] += st[11] * (cum[1] - cum[1])
            st[12] = 0.0
            break
        st[0] = t[r]
        st[1] = 1.0
        if a[r] == 0:
            st[2] = 1.0
            st[4] = p[r]
        else:
            st[3] = 1.0
            st[5] = p[r]


def hy_new() -> dict:
    return {
        "st": np.zeros(13),
        "cum": np.zeros(2),
        "has": np.zeros(2, np.uint8),
        "hy": np.zeros(3),  # the covariance, A's and B's realised variance
    }


def hy_run(t, a, p, parts=1):
    st = hy_new()
    sl = chunks(len(t), parts)
    for i, s in enumerate(sl):
        _hy_feed(st["st"], st["cum"], st["has"], st["hy"], t[s], a[s], p[s], i == len(sl) - 1)
    return st
