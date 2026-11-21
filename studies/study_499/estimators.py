"""Batch definitions of the realised estimators, written from the papers.

Every function takes **log-prices** (hfhd's input), not returns, so that the
two sides start from the same numbers. A price matrix is ``(n + 1) x k``:
``n`` returns behind it. Each function names the paper's equation it
writes, and each place where hfhd's convention differs is a flag whose
default is the paper's; ``compare.py`` measures each flag's effect and
checks the hfhd setting of every flag against hfhd itself.

Nothing here calls hfhd.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import integrate

# --------------------------------------------------------------------------
# refresh time (Barndorff-Nielsen, Hansen, Lunde and Shephard 2011, Section 2.3;
# the idea is Ait-Sahalia, Mykland and Zhang's "previous tick" sampling)
# --------------------------------------------------------------------------


def refresh_time(times: list[np.ndarray], prices: list[np.ndarray]):
    """BNHLS 2011 (their Definition 1): ``tau_1 = max_i t_i[0]`` and
    ``tau_{j+1} = max_i min{t in t_i : t > tau_j}``; each asset's value at
    ``tau_j`` is its previous tick, the last price stamped at or before it.

    Returns the refresh stamps and an ``(N, k)`` matrix of previous ticks."""
    k = len(times)
    out_t = []
    tau = max(int(t[0]) for t in times)
    while True:
        out_t.append(tau)
        nxt = []
        for t in times:
            pos = np.searchsorted(t, tau, side="right")
            if pos == len(t):
                nxt = None
                break
            nxt.append(int(t[pos]))
        if nxt is None:
            break
        tau = max(nxt)
    stamps = np.asarray(out_t, dtype=np.int64)
    vals = np.empty((len(stamps), k))
    for j, (t, p) in enumerate(zip(times, prices, strict=True)):
        vals[:, j] = p[np.searchsorted(t, stamps, side="right") - 1]
    return stamps, vals


# --------------------------------------------------------------------------
# two scales (Zhang, Mykland and Ait-Sahalia 2005), covariances by Zhang 2011
# --------------------------------------------------------------------------


def avg_rv(p: np.ndarray, K: int) -> np.ndarray:
    """``[Y, Y]^(K) = (1/K) sum_{i=K}^{n} (Y_i - Y_{i-K})(Y_i - Y_{i-K})'``:
    the average of the K sparse realised covariances on offset grids
    (ZMA 2005, Eq. 5 and Section 3.2)."""
    d = p[K:] - p[:-K]
    return d.T @ d / K


def tsrc(p: np.ndarray, K: int, J: int = 1, n_conv: str = "paper", adjust: bool = False):
    """ZMA 2005's two-scale estimator (their Eq. 55 with the slow scale ``K``
    and, after Ait-Sahalia, Mykland and Zhang 2011, a fast scale ``J``):

        TSRV = [Y,Y]^(K) - (nbar_K / nbar_J) [Y,Y]^(J),   nbar_K = (n - K + 1) / K

    with ``n`` the number of returns. ``n_conv = "hfhd"`` puts the number
    of prices, ``n + 1``, where the paper has ``n``. ``adjust`` multiplies
    by ``(1 - nbar_K / nbar_J)^-1``, ZMA's small-sample adjustment
    (their Section 4.2), which hfhd does not apply."""
    n = p.shape[0] - 1
    m = n + 1 if n_conv == "hfhd" else n
    nk = (m - K + 1) / K
    nj = (m - J + 1) / J
    est = avg_rv(p, K) - nk / nj * avg_rv(p, J)
    if adjust:
        est = est / (1.0 - nk / nj)
    return est


# --------------------------------------------------------------------------
# multi scale (Zhang 2006), shifted scales and the edge term of Tao, Wang
# and Chen 2013, which is the form hfhd writes
# --------------------------------------------------------------------------


def msrc_weights(M: int, N: int) -> tuple[np.ndarray, np.ndarray]:
    """Scales ``K_i = N + i`` and Zhang's optimal weights
    ``a_i = K_i (K_i - Kbar) / (M var(K))`` (Zhang 2006, Eq. 18 for
    ``K_i = i``): they sum to 1 and ``sum a_i / K_i = 0``, which cancels the
    noise term ``2 n omega^2 / K`` of every scale."""
    Ks = N + np.arange(1, M + 1, dtype=float)
    kbar = Ks.mean()
    var = (Ks**2).mean() - kbar**2
    a = Ks * (Ks - kbar) / (M * var)
    return Ks, a


def msrc(p: np.ndarray, M: int, N: int, edge: bool = True, n_conv: str = "hfhd"):
    """``MSRV = sum_i a_i [Y,Y]^(K_i) + zeta ([Y,Y]^(K_1) - [Y,Y]^(K_M))``,
    ``zeta = (M + N)(N + 1) / ((n + 1)(M - 1))``: Zhang 2006's estimator on
    the scales ``N + 1 .. N + M`` with the end-effect term of Tao, Wang and
    Chen 2013. ``n_conv = "hfhd"`` reads ``n`` as the number of prices (what
    hfhd passes), ``"returns"`` as the number of returns. ``M = 1`` is the
    plain realised covariance, as hfhd has it (``N`` ignored)."""
    if M == 1:
        return avg_rv(p, 1)
    Ks, a = msrc_weights(M, N)
    n = p.shape[0] if n_conv == "hfhd" else p.shape[0] - 1
    s = np.zeros((p.shape[1], p.shape[1]))
    for K, ai in zip(Ks.astype(int), a, strict=True):
        s += ai * avg_rv(p, K)
    if edge:
        zeta = (M + N) * (N + 1) / ((n + 1) * (M - 1))
        s += zeta * (avg_rv(p, N + 1) - avg_rv(p, N + M))
    return s


def msrc_hfhd_weights(M: int, N: int) -> np.ndarray:
    """hfhd's closed form ``12 (m + N)(m - M/2 - 1/2) / (M (M^2 - 1))``."""
    m = np.arange(1, M + 1, dtype=float)
    return 12 * (m + N) * (m - M / 2 - 1 / 2) / (M * (M**2 - 1))


# --------------------------------------------------------------------------
# pre-averaging (Christensen, Kinnebrock and Podolskij 2010)
# --------------------------------------------------------------------------


def g_min(x):
    return np.minimum(x, 1.0 - x)


def psi1(k: int, upto_k: bool = True) -> float:
    """CKP's ``psi_1^k = k sum_{i=1}^{k} (g(i/k) - g((i-1)/k))^2``;
    ``upto_k = False`` stops at ``k - 1``, as hfhd does."""
    i = np.arange(1, k + 1 if upto_k else k, dtype=float)
    return float(k * np.sum((g_min(i / k) - g_min((i - 1) / k)) ** 2))


def psi2(k: int) -> float:
    """CKP's ``psi_2^k = (1/k) sum_{i=1}^{k-1} g(i/k)^2``."""
    i = np.arange(1, k, dtype=float)
    return float(np.sum(g_min(i / k) ** 2) / k)


def preavg(r: np.ndarray, k: int, first: int = 0) -> np.ndarray:
    """``Ybar_i = sum_{j=1}^{k-1} g(j/k) r_{i+j}`` (CKP 2010, Eq. 8), returns
    indexed from 1, for ``i = first .. n - k + 1``: ``first = 0`` is the
    paper's ``n - k + 2`` windows, ``first = 1`` hfhd's (its first window
    starts at the second return)."""
    n = r.shape[0]
    w = g_min(np.arange(1, k) / k)
    starts = np.arange(first, n - k + 2)
    out = np.zeros((len(starts), r.shape[1]))
    for j in range(1, k):
        out += w[j - 1] * r[starts + j - 1]
    return out


def mrc(
    p: np.ndarray,
    k: int,
    bias_correction: bool,
    windows: str = "paper",
    psi1_upto_k: bool = True,
    rescale: bool = True,
    theta: float | None = None,
):
    """CKP 2010's modulated realised covariance (Eq. 9) and, with
    ``bias_correction``, its bias-corrected form (Theorem 1, Eq. 12)::

        MRC = n/(n - k + 2) * 1/(psi_2 k) * sum_i Ybar_i Ybar_i'
        MRC - psi_1/(theta^2 psi_2) * (1/(2n)) sum_j r_j r_j',  theta = k / sqrt(n)

    ``rescale`` divides the corrected form by ``1 - psi_1/(2 psi_2 k^2)``,
    CKP's footnote 1 (the share of the integrated covariance the bias term
    takes away). ``theta`` given puts that number in the bias term in place
    of the window's own ``k / sqrt(n)``. hfhd: ``windows = "hfhd"``,
    ``psi1_upto_k = False``, ``rescale = False`` and ``theta`` its input."""
    r = np.diff(p, axis=0)
    n = r.shape[0]
    ybar = preavg(r, k, first=0 if windows == "paper" else 1)
    p2 = psi2(k)
    est = n / (n - k + 2) / (p2 * k) * (ybar.T @ ybar)
    if bias_correction:
        if theta is None:
            theta = k / math.sqrt(n)
        b = psi1(k, psi1_upto_k) / (theta**2 * p2)
        est = est - b * (r.T @ r) / (2 * n)
        if rescale:
            est = est / (1.0 - b / (2 * n))
    return est


def mrc_k(n: int, theta: float, bias_correction: bool) -> int:
    """The window CKP recommend: ``ceil(theta sqrt(n))`` with the bias
    correction, ``ceil(theta n^0.6)`` without it (their Section 3.4)."""
    if bias_correction:
        return int(math.ceil(math.sqrt(n) * theta))
    return int(math.ceil(n**0.6 * theta))


# --------------------------------------------------------------------------
# realised kernels (Barndorff-Nielsen, Hansen, Lunde and Shephard 2011)
# --------------------------------------------------------------------------


def parzen(x):
    x = np.abs(np.asarray(x, dtype=float))
    return np.where(x <= 0.5, 1 - 6 * x**2 + 6 * x**3, np.where(x <= 1, 2 * (1 - x) ** 3, 0.0))


def qs(x):
    """BNHLS 2011's quadratic spectral, Table 1: ``3/x^2 (sin x / x - cos x)``."""
    x = np.abs(np.asarray(x, dtype=float))
    with np.errstate(invalid="ignore", divide="ignore"):
        y = 3.0 / x**2 * (np.sin(x) / x - np.cos(x))
    return np.where(x == 0, 1.0, y)


KERNELS = {"parzen": parzen, "quadratic_spectral": qs}


def c_star(name: str) -> float:
    """BNHLS's ``c* = (k''(0)^2 / k_.^{0,0})^{1/5}``, with ``k_.^{0,0} =
    int_0^inf k(x)^2 dx`` (one side), by quadrature: 0.269 for Parzen and
    ``3 pi / 5`` for the quadratic spectral."""
    kern = KERNELS[name]
    if name == "parzen":
        k2 = 12.0**2
        kk = integrate.quad(lambda x: float(kern(x)) ** 2, 0, 1, points=[0.5])[0]
    else:
        k2 = 0.2**2  # k(x) = 1 - x^2/10 + ..., so k''(0) = -1/5
        kk = integrate.quad(lambda x: float(kern(x)) ** 2, 0, np.inf, limit=500)[0]
    return (k2 / kk) ** 0.2


def jitter(p: np.ndarray, m: int) -> np.ndarray:
    """BNHLS's end jitter on the prices: the first price replaced by the mean
    of the first ``m``, the last by the mean of the last ``m``, the ``m - 1``
    inside each end dropped. ``m = 1`` changes nothing."""
    n = p.shape[0] - 1
    return np.vstack([p[:m].mean(axis=0), p[m : n - m + 1], p[n - m + 1 :].mean(axis=0)])


def gamma(x: np.ndarray, h: int) -> np.ndarray:
    """``Gamma_h = sum_{j=h+1}^{n} x_j x_{j-h}'``."""
    if h == 0:
        return x.T @ x
    return x[h:].T @ x[:-h]


def krvm(
    p: np.ndarray, H: float, kernel: str = "parzen", m: int = 1, form: str = "bnhls", lags=None
):
    """The multivariate realised kernel.

    ``form = "bnhls"`` (BNHLS 2011, Eq. 2): ``sum_{h=-n}^{n} k(h/(H+1)) Gamma_h``
    over the jittered prices' returns. ``form = "flat_top"`` is hfhd's:
    ``Gamma_0 + sum_{h>=1} k((h-1)/H)(Gamma_h + Gamma_h')`` (the flat-top
    weights of Barndorff-Nielsen et al. 2008, which give lag 1 the weight
    1). ``lags`` truncates the sum (the quadratic spectral kernel has
    unbounded support)."""
    x = np.diff(jitter(p, m), axis=0)
    n = x.shape[0]
    kern = KERNELS[kernel]
    top = n - 1 if lags is None else min(lags, n - 1)
    out = gamma(x, 0)
    for h in range(1, top + 1):
        w = float(kern(h / (H + 1))) if form == "bnhls" else float(kern((h - 1) / H))
        if w == 0.0 and kernel == "parzen":
            break
        g = gamma(x, h)
        out = out + w * (g + g.T)
    return out


def noise_and_iv(p: np.ndarray, stride: int = 20):
    """BNHLS's inputs to the bandwidth, per asset: ``omega^2 = RV / (2n)`` on
    every return (biased upward, as they accept) and the integrated variance
    as the realised variance on a ``stride``-return grid averaged over the
    ``stride`` offsets (their Section 4.1)."""
    r = np.diff(p, axis=0)
    n = r.shape[0]
    omega2 = (r**2).sum(axis=0) / (2 * n)
    sparse = []
    for o in range(stride):
        q = p[o::stride]
        sparse.append((np.diff(q, axis=0) ** 2).sum(axis=0))
    return omega2, np.mean(sparse, axis=0)


def bandwidth(n: int, iv: float, omega2: float, kernel: str, exact: bool = True) -> float:
    """``H = c* xi^{4/5} n^{3/5}``, ``xi^2 = omega^2 / IV`` (BNHLS 2011,
    Section 4.1). ``exact = False`` is hfhd's ``get_bandwidth``: ``c*``
    rounded to 3.51 or 0.46, and the result truncated to an integer."""
    xi2 = omega2 / iv
    if exact:
        return c_star(kernel) * xi2 ** (2 / 5) * n ** (3 / 5)
    c = 3.51 if kernel == "parzen" else 0.46
    return float(int(c * xi2 ** (2 / 5) * n ** (3 / 5)))


# --------------------------------------------------------------------------
# Hayashi-Yoshida (Hayashi and Yoshida 2005); pre-averaged after
# Christensen, Kinnebrock and Podolskij 2010, Section 4, and Christensen,
# Podolskij and Vetter 2013
# --------------------------------------------------------------------------


def hy_bruteforce(ta, pa, tb, pb) -> float:
    """``sum_i sum_j dX_i dY_j 1{(ta_{i-1}, ta_i] and (tb_{j-1}, tb_j] meet}``
    over every pair: the definition itself, ``O(n_a n_b)``."""
    dx, dy = np.diff(pa), np.diff(pb)
    lo = np.maximum(ta[:-1, None], tb[None, :-1])
    hi = np.minimum(ta[1:, None], tb[None, 1:])
    return float(dx @ (lo < hi).astype(float) @ dy)


def hy(ta, pa, tb, pb) -> float:
    """The same sum, ``O(n log n)``: the returns of B that meet ``(s, e]``
    are those from the first with ``tb_j > s`` to the first with
    ``tb_j >= e``, so their sum telescopes to ``Y(first tb >= e) - Y(last
    tb <= s)`` (the first and last of B's prices where those run out)."""
    s, e = ta[:-1], ta[1:]
    lo = np.searchsorted(tb, s, side="right") - 1  # last tb <= s
    hi = np.searchsorted(tb, e, side="left")  # first tb >= e
    lo = np.clip(lo, 0, len(tb) - 1)
    hi = np.clip(hi, 0, len(tb) - 1)
    # B's path as a running sum of its returns, not its log-prices: a
    # difference of two log-prices near 4.6 carries a rounding of 5e-16,
    # five parts in 10^12 of a return of 1e-4.
    cum = np.concatenate([[0.0], np.cumsum(np.diff(pb))])
    return float(np.diff(pa) @ (cum[hi] - cum[lo]))


def hy_preavg(ta, pa, tb, pb, k: int, convention: str = "paper", psi: str = "limit") -> float:
    """The pre-averaged Hayashi-Yoshida estimator::

        1/(psi_HY k)^2 sum_i sum_j Abar_i Bbar_j 1{I_i and J_j meet}

    ``Abar_i = sum_{l=1}^{k-1} g(l/k) dX_{i+l}``. ``convention = "paper"``
    (Christensen, Podolskij and Vetter 2013): windows ``i = 0 .. n - k`` on
    each asset, ``I_i = (t_i, t_{i+k}]``. ``convention = "hfhd"``
    reproduces hfhd's index choices: A's windows ``i = k .. n - k + 1``
    with ``I_i = (t_{i-1}, t_{i+k-1}]``, B's windows ``0 .. n - k + 1`` with
    the same shift and no lower end for B's first ``k``, and A the asset
    with fewer ticks. ``psi = "limit"`` is ``psi_HY = 1/4`` (hfhd);
    ``"exact"`` the finite-sample ``(1/k) sum g(l/k)``."""
    w = g_min(np.arange(1, k) / k)
    norm = (k / 4.0) if psi == "limit" else float(w.sum())

    def bars(p, first, last):
        d = np.diff(p)
        return np.correlate(d, w, mode="valid")[first : last + 1] / norm

    if convention == "paper":
        na, nb = len(ta) - 1, len(tb) - 1
        A = bars(pa, 0, na - k)
        B = bars(pb, 0, nb - k)
        a_lo, a_hi = ta[: na - k + 1], ta[k:]
        b_lo, b_hi = tb[: nb - k + 1], tb[k:]
        # B's windows that meet (s, e]: b_hi > s and b_lo < e, a run of j
        # since both ends rise with j.
        j0 = np.searchsorted(b_hi, a_lo, side="right")
        j1 = np.searchsorted(b_lo, a_hi, side="left")
        cum = np.concatenate([[0.0], np.cumsum(B)])
        return float(A @ (cum[np.maximum(j1, j0)] - cum[j0]))
    # hfhd: the slower asset is A
    if len(ta) > len(tb):
        ta, pa, tb, pb = tb, pb, ta, pa
    na, nb = len(ta) - 1, len(tb) - 1
    A = bars(pa, 0, na - k + 1)  # window m uses returns m+1 .. m+k-1
    B = bars(pb, 0, nb - k + 1)
    a_new = ta[k - 1 :]  # window m's stamp: t_{m+k-1}
    b_new = tb[k - 1 :]
    total = 0.0
    for i in range(k, len(A)):
        s, e = a_new[i - k], a_new[i]
        j0 = np.searchsorted(b_new, s, side="right")
        j1 = np.searchsorted(b_new, e, side="left")
        total += A[i] * B[j0 : j1 + k].sum()
    return total


def hy_k(n_a: int, n_b: int, theta: float) -> int:
    """hfhd's window from ``theta``: ``ceil(theta sqrt(nbar))``, ``nbar`` the
    mean count of prices of the two assets."""
    return int(math.ceil(math.sqrt((n_a + n_b) / 2) * theta))


def hy_preavg_bruteforce(ta, pa, tb, pb, k: int) -> float:
    """The paper's pre-averaged sum over every pair of windows, ``O(n_a n_b)``:
    the check of :func:`hy_preavg`'s run search on a short stretch."""
    w = g_min(np.arange(1, k) / k)
    norm = k / 4.0
    A = np.correlate(np.diff(pa), w, mode="valid")[: len(ta) - k] / norm
    B = np.correlate(np.diff(pb), w, mode="valid")[: len(tb) - k] / norm
    a_lo, a_hi = ta[: len(A)], ta[k : k + len(A)]
    b_lo, b_hi = tb[: len(B)], tb[k : k + len(B)]
    meet = np.maximum(a_lo[:, None], b_lo[None, :]) < np.minimum(a_hi[:, None], b_hi[None, :])
    return float(A @ meet.astype(float) @ B)
