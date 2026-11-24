"""Oracle scan for PLAN 554 (ewm_moments): which third-party call computes
each quantity, and how far it sits from the definition written longhand.

Not a test and not run by the gate: the scan behind
docs/records/EWM-MOMENTS-DESIGN.md. Every library it calls is in the dev
group:

    uv run python scripts/oracle_scan_554.py

Every definition below is written longhand from the paper, over plain
Python floats summed with ``math.fsum``, so it shares no arithmetic with any
library it checks. The weights are frequency weights normalised by their
sum, ``E_w[f] = sum_i w_i f(x_i) / sum_i w_i`` (Pebay, Terriberry, Kolla and
Bennett 2016, "Numerically stable, scalable formulas for parallel and online
computation of higher-order multivariate central moments with arbitrary
weights"; Finch 2009, "Incremental calculation of weighted mean and
variance"). The exponential weights are ``w_i = lam^(n-1-i)`` times a row's
own weight, one of which is 0 (a zero-weight row learns nothing).
"""

from __future__ import annotations

import math
import sys
from math import comb, fsum

import numpy as np
import polars as pl
import river
import river.stats as rs
import scipy
import scipy.stats as ss
import statsmodels
import statsmodels.stats.moment_helpers as mh
import statsmodels.tsa.stattools as st
from statsmodels.stats.weightstats import DescrStatsW

N = 40
LAM = 0.9
ROWS: list[tuple[str, str, str, str, float]] = []


def gap(lib: float, ref: float) -> float:
    """Relative gap, against the reference's magnitude (absolute below 1)."""
    return abs(lib - ref) / max(abs(ref), 1.0)


def record(quantity: str, call: str, version: str, exact: str, lib, ref) -> None:
    lib = np.atleast_1d(np.asarray(lib, dtype=float))
    ref = np.atleast_1d(np.asarray(ref, dtype=float))
    g = max(gap(a, b) for a, b in zip(lib, ref, strict=True))
    ROWS.append((quantity, call, version, exact, g))


# ---------------------------------------------------------------- longhand


def wmean(x, w, p=1):
    return fsum(wi * xi**p for xi, wi in zip(x, w, strict=False)) / fsum(w)


def wcentral(x, w, p):
    m = wmean(x, w)
    return fsum(wi * (xi - m) ** p for xi, wi in zip(x, w, strict=False)) / fsum(w)


def wcross(x, y, w, a, b):
    mx, my = wmean(x, w), wmean(y, w)
    num = fsum(wi * (xi - mx) ** a * (yi - my) ** b for xi, yi, wi in zip(x, y, w, strict=False))
    return num / fsum(w)


def cumulants_from_raw(mu_raw):
    """kappa_n = mu'_n - sum_{m=1}^{n-1} C(n-1, m-1) kappa_m mu'_{n-m}."""
    mu = [1.0, *mu_raw]
    k = [0.0]
    for n in range(1, len(mu)):
        k.append(mu[n] - fsum(comb(n - 1, m - 1) * k[m] * mu[n - m] for m in range(1, n)))
    return k[1:]


def cumulants_from_central(mean, mu):
    """mu[p] central moments, p = 2..6: the closed forms up to 6."""
    m2, m3, m4 = mu[2], mu[3], mu[4]
    out = [mean, m2, m3, m4 - 3 * m2**2]
    if 5 in mu:
        out.append(mu[5] - 10 * m3 * m2)
    if 6 in mu:
        out.append(mu[6] - 15 * mu[4] * m2 - 10 * m3**2 + 30 * m2**3)
    return out


def kstats(x):
    """Fisher's unbiased k-statistics k1..k4 from power sums."""
    n = len(x)
    s1, s2, s3, s4 = (fsum(v**r for v in x) for r in (1, 2, 3, 4))
    k1 = s1 / n
    k2 = (n * s2 - s1**2) / (n * (n - 1))
    k3 = (2 * s1**3 - 3 * n * s1 * s2 + n**2 * s3) / (n * (n - 1) * (n - 2))
    k4 = (
        -6 * s1**4
        + 12 * n * s1**2 * s2
        - 3 * n * (n - 1) * s2**2
        - 4 * n * (n + 1) * s1 * s3
        + n**2 * (n + 1) * s4
    ) / (n * (n - 1) * (n - 2) * (n - 3))
    return [k1, k2, k3, k4]


def acov(x, k, adjusted=False):
    n = len(x)
    m = fsum(x) / n
    s = fsum((x[t] - m) * (x[t - k] - m) for t in range(k, n))
    return s / ((n - k) if adjusted else n)


def ccov_lead(x, y, k):
    """(1/n) sum_t (x_{t+k} - xbar)(y_t - ybar): x leads y by k."""
    n = len(x)
    mx, my = fsum(x) / n, fsum(y) / n
    return fsum((x[t + k] - mx) * (y[t] - my) for t in range(n - k)) / n


def pearson(a, b):
    w = [1.0] * len(a)
    return wcross(a, b, w, 1, 1) / math.sqrt(wcentral(a, w, 2) * wcentral(b, w, 2))


# ------------------------------------------- the online update, longhand


class PebayEW:
    """Weighted, decayed central co-moments of a pair (x, y) up to orders
    (A, B), one row at a time: the binomial shift of Pebay et al. (2016).

    Decay is a scaling of every past weight by lam, so a row is the merge
    of (lam*W, lam*M) with the one-row summary (w, x, y). M[i][j] is the
    unnormalised sum_i w_i (x_i - mx)^i (y_i - my)^j; M[0][0] = W.
    """

    def __init__(self, A, B):
        self.A, self.B = A, B
        self.M = [[0.0] * (B + 1) for _ in range(A + 1)]
        self.mx = self.my = 0.0

    def update(self, x, y, w, lam):
        W0 = lam * self.M[0][0]
        Wn = W0 + w
        if Wn == 0.0:  # rule 9: a zero-weight first row; nothing to learn
            for i in range(self.A + 1):
                for j in range(self.B + 1):
                    self.M[i][j] *= lam
            return
        b = w / Wn
        a = W0 / Wn
        dx, dy = x - self.mx, y - self.my
        old = [[lam * v for v in row] for row in self.M]
        new = [[0.0] * (self.B + 1) for _ in range(self.A + 1)]
        for p in range(self.A + 1):
            for q in range(self.B + 1):
                s = []
                for i in range(p + 1):
                    for j in range(q + 1):
                        if (p - i, q - j) in ((1, 0), (0, 1)):
                            continue  # a first central moment is 0
                        s.append(
                            comb(p, i)
                            * comb(q, j)
                            * old[p - i][q - j]
                            * (-b * dx) ** i
                            * (-b * dy) ** j
                        )
                s.append(w * (a * dx) ** p * (a * dy) ** q)
                new[p][q] = fsum(s)
        new[0][0] = Wn
        self.M = new
        self.mx += b * dx
        self.my += b * dy

    def central(self, p, q):
        return self.M[p][q] / self.M[0][0]


def main() -> None:
    rng = np.random.default_rng(554)
    x = (3.0 + rng.gamma(2.0, 1.5, N)).tolist()
    y = (0.6 * np.array(x) + rng.standard_normal(N)).tolist()
    ones = [1.0] * N
    own = [1.0] * N
    own[17] = 0.0  # a zero-weight row
    ew = [LAM ** (N - 1 - i) * own[i] for i in range(N)]
    xa = np.array(x)
    v_np, v_sp, v_sm = np.__version__, scipy.__version__, statsmodels.__version__
    v_pl, v_rv = pl.__version__, river.__version__

    # raw and central moments, unweighted
    for p in (2, 3, 4, 5, 6):
        record(
            f"raw moment p={p}",
            f"scipy.stats.moment(x, order={p}, center=0)",
            v_sp,
            "exact",
            ss.moment(xa, order=p, center=0),
            wmean(x, ones, p),
        )
        record(
            f"central moment p={p}",
            f"scipy.stats.moment(x, order={p})",
            v_sp,
            "exact (1/n, plug-in)",
            ss.moment(xa, order=p),
            wcentral(x, ones, p),
        )

    # weighted raw and central: numpy computes the sum, the definition is by hand
    ewa = np.array(ew)
    for p in (2, 3, 4, 6):
        record(
            f"EW raw moment p={p}",
            f"numpy.average(x**{p}, weights=w)",
            v_np,
            "exact",
            np.average(xa**p, weights=ewa),
            wmean(x, ew, p),
        )
        m = np.average(xa, weights=ewa)
        record(
            f"EW central moment p={p}",
            f"numpy.average((x-m)**{p}, weights=w)",
            v_np,
            "exact",
            np.average((xa - m) ** p, weights=ewa),
            wcentral(x, ew, p),
        )
    d = DescrStatsW(xa, weights=ewa)
    record(
        "EW variance",
        "DescrStatsW(x, weights=w).var  (ddof=0)",
        v_sm,
        "exact (order 2 only)",
        d.var,
        wcentral(x, ew, 2),
    )
    record(
        "EW covariance",
        "numpy.cov(x, y, aweights=w, bias=True)[0,1]",
        v_np,
        "exact",
        np.cov(xa, np.array(y), aweights=ewa, bias=True)[0, 1],
        wcross(x, y, ew, 1, 1),
    )
    lam_w = [LAM ** (N - 1 - i) for i in range(N)]
    alpha = 1 - LAM
    record(
        "EW mean (no own weights)",
        f"pl.col.ewm_mean(alpha={alpha:g}, adjust=True)",
        v_pl,
        "exact",
        pl.Series(x).ewm_mean(alpha=alpha, adjust=True)[-1],
        wmean(x, lam_w),
    )
    record(
        "EW variance (no own weights)",
        f"pl.col.ewm_var(alpha={alpha:g}, adjust=True, bias=True)",
        v_pl,
        "exact (bias=True); bias=False is the reliability-weight correction",
        pl.Series(x).ewm_var(alpha=alpha, adjust=True, bias=True)[-1],
        wcentral(x, lam_w, 2),
    )

    # cumulants
    mu_raw = [wmean(x, ones, p) for p in range(1, 7)]
    mu_c = {p: wcentral(x, ones, p) for p in range(2, 7)}
    ref_k = cumulants_from_central(mu_raw[0], mu_c)
    record(
        "cumulants k1..k6 (plug-in)",
        "statsmodels.stats.moment_helpers.mnc2cum(raw[1..6])",
        v_sm,
        "exact (plug-in, not unbiased)",
        mh.mnc2cum(mu_raw),
        ref_k,
    )
    record(
        "cumulants k2..k6 from central",
        "statsmodels.stats.moment_helpers.mc2cum([0, mu2..mu6])",
        v_sm,
        "exact (plug-in); fed central moments, no cancellation",
        mh.mc2cum([0.0] + [mu_c[p] for p in range(2, 7)])[1:],
        ref_k[1:],
    )
    record(
        "cumulants k1..k6 (recursion)",
        "longhand recursion vs closed forms",
        "-",
        "exact",
        cumulants_from_raw(mu_raw),
        ref_k,
    )
    record(
        "k-statistics k1..k4",
        "scipy.stats.kstat(x, n) for n=1..4",
        v_sp,
        "exact for the unbiased k-statistic",
        [ss.kstat(xa, n) for n in (1, 2, 3, 4)],
        kstats(x),
    )
    record(
        "k-statistic k4 vs plug-in cumulant",
        "scipy.stats.kstat(x, 4)",
        v_sp,
        "NOT the plug-in cumulant (unbiased)",
        ss.kstat(xa, 4),
        ref_k[3],
    )
    kk = kstats(x)
    record(
        "var of k2",
        "scipy.stats.kstatvar(x, 2)",
        v_sp,
        "exact: (2n k2^2 + (n-1) k4)/(n(n+1)), unbiased",
        ss.kstatvar(xa, 2),
        (2 * N * kk[1] ** 2 + (N - 1) * kk[3]) / (N * (N + 1)),
    )

    # standardized: skew, kurtosis
    g1 = mu_c[3] / mu_c[2] ** 1.5
    g2 = mu_c[4] / mu_c[2] ** 2 - 3
    G1 = g1 * math.sqrt(N * (N - 1)) / (N - 2)
    G2 = ((N + 1) * g2 + 6) * (N - 1) / ((N - 2) * (N - 3))
    record(
        "skew g1 (plug-in)",
        "scipy.stats.skew(x, bias=True)",
        v_sp,
        "exact",
        ss.skew(xa, bias=True),
        g1,
    )
    record(
        "skew G1 (adjusted)",
        "scipy.stats.skew(x, bias=False)",
        v_sp,
        "exact for G1, not g1",
        ss.skew(xa, bias=False),
        G1,
    )
    record(
        "excess kurtosis g2",
        "scipy.stats.kurtosis(x, fisher=True, bias=True)",
        v_sp,
        "exact",
        ss.kurtosis(xa, bias=True),
        g2,
    )
    record(
        "excess kurtosis G2",
        "scipy.stats.kurtosis(x, bias=False)",
        v_sp,
        "exact for G2, not g2",
        ss.kurtosis(xa, bias=False),
        G2,
    )
    for p in (3, 4, 5):
        record(
            f"standardized p={p}",
            f"scipy.stats.moment(x,{p})/moment(x,2)**{p}/2",
            v_sp,
            "exact",
            ss.moment(xa, order=p) / ss.moment(xa, order=2) ** (p / 2),
            mu_c[p] / mu_c[2] ** (p / 2),
        )
    record(
        "skew g1 via mc2mvsk",
        "statsmodels...moment_helpers.mc2mvsk([m,m2,m3,m4])[2]",
        v_sm,
        "exact (g1; kurt is excess g2)",
        mh.mc2mvsk([mu_raw[0], mu_c[2], mu_c[3], mu_c[4]])[2:],
        [g1, g2],
    )
    W = 10
    tail = x[-W:]
    tc = {p: wcentral(tail, [1.0] * W, p) for p in (2, 3, 4)}
    record(
        "rolling skew g1, window 10",
        "pl.col.rolling_skew(10, bias=True)",
        v_pl,
        "exact (g1)",
        pl.Series(x).rolling_skew(W, bias=True)[-1],
        tc[3] / tc[2] ** 1.5,
    )
    record(
        "rolling excess kurtosis g2, window 10",
        "pl.col.rolling_kurtosis(10, bias=True)",
        v_pl,
        "exact (g2)",
        pl.Series(x).rolling_kurtosis(W, bias=True)[-1],
        tc[4] / tc[2] ** 2 - 3,
    )
    sk, ku = rs.Skew(bias=True), rs.Kurtosis(bias=True)
    for v in x:
        sk.update(v)
        ku.update(v)
    record("skew g1, online", "river.stats.Skew(bias=True)", v_rv, "exact (g1)", sk.get(), g1)
    record(
        "excess kurtosis g2, online",
        "river.stats.Kurtosis(bias=True)",
        v_rv,
        "exact (g2)",
        ku.get(),
        g2,
    )

    # cross-moments, lagged cross-moments, autocovariance, autocorrelation
    ya = np.array(y)
    for a, b in ((1, 1), (2, 1), (2, 2), (3, 1)):
        mx, my = xa.mean(), ya.mean()
        record(
            f"cross central E[dx^{a} dy^{b}]",
            f"numpy.mean((x-mx)**{a}*(y-my)**{b})",
            v_np,
            "exact",
            np.mean((xa - mx) ** a * (ya - my) ** b),
            wcross(x, y, ones, a, b),
        )
    for k in (1, 2, 3):
        record(
            f"autocovariance lag {k} (1/n, full mean)",
            f"statsmodels acovf(x, fft=False)[{k}]",
            v_sm,
            "exact (biased, 1/n)",
            st.acovf(xa, fft=False)[k],
            acov(x, k),
        )
        record(
            f"autocovariance lag {k} (1/(n-k))",
            f"statsmodels acovf(x, adjusted=True, fft=False)[{k}]",
            v_sm,
            "exact (adjusted)",
            st.acovf(xa, adjusted=True, fft=False)[k],
            acov(x, k, adjusted=True),
        )
        record(
            f"autocorrelation lag {k}",
            f"statsmodels acf(x, nlags=3, fft=False)[{k}]",
            v_sm,
            "exact (acovf_k/acovf_0, full-sample mean)",
            st.acf(xa, nlags=3, fft=False)[k],
            acov(x, k) / acov(x, 0),
        )
        rho_pair = pearson(x[k:], x[:-k])
        record(
            f"autocorrelation lag {k}, Pearson of pairs",
            "longhand Pearson(x_t, x_{t-k}) vs acf",
            v_sm,
            "NOT acf: pair means and pair variances",
            st.acf(xa, nlags=3, fft=False)[k],
            rho_pair,
        )
        ac = rs.AutoCorr(lag=k)
        for v in x:
            ac.update(v)
        record(
            f"autocorrelation lag {k}, online",
            f"river.stats.AutoCorr(lag={k})",
            v_rv,
            "Pearson of pairs, not acf",
            ac.get(),
            rho_pair,
        )
        sx, sy = np.std(xa), np.std(ya)
        record(
            f"cross-correlation lag {k}",
            f"statsmodels ccf(x, y, adjusted=False, fft=False)[{k}]",
            v_sm,
            "exact: x LEADS y (x_{t+k} with y_t), 1/n, full means",
            st.ccf(xa, ya, adjusted=False, fft=False)[k],
            ccov_lead(x, y, k) / (sx * sy),
        )
        # the block's lagged cross-moment E[x_t^a y_{t-k}^b]: pairs from row k on
        mx, my = fsum(x[k:]) / (N - k), fsum(y[:-k]) / (N - k)
        record(
            f"lagged cross E[dx_t dy_(t-k)], k={k}",
            "numpy.cov(x[k:], y[:-k], bias=True)[0,1]",
            v_np,
            "exact for the pairs' own means; NOT ccovf (full means, 1/n)",
            np.cov(xa[k:], ya[:-k], bias=True)[0, 1],
            fsum((x[t] - mx) * (y[t - k] - my) for t in range(k, N)) / (N - k),
        )

    # the online update, longhand, against the batch weighted definition
    for A, B in ((6, 0), (2, 2), (3, 1)):
        acc = PebayEW(A, B)
        for xi, yi, wi in zip(x, y, own, strict=False):
            acc.update(xi, yi, wi, LAM)
        lib = [acc.central(p, q) for p in range(A + 1) for q in range(B + 1) if p + q >= 2]
        ref = [wcross(x, y, ew, p, q) for p in range(A + 1) for q in range(B + 1) if p + q >= 2]
        record(
            f"EW central co-moments to ({A},{B}), online",
            "Pebay binomial shift, longhand",
            "-",
            "exact (to rounding)",
            lib,
            ref,
        )
    # a zero-weight FIRST row (rule 9): nothing poisoned
    acc = PebayEW(4, 0)
    for xi, wi in zip([5.0, *x], [0.0, *own], strict=False):
        acc.update(xi, 0.0, wi, LAM)
    record(
        "zero-weight first row, online to p=4",
        "Pebay binomial shift, longhand",
        "-",
        "exact (no NaN)",
        [acc.central(p, 0) for p in (2, 3, 4)],
        [wcentral(x, ew, p) for p in (2, 3, 4)],
    )

    # river's EW variance is not the adjusted EW form
    ev = rs.EWVar(fading_factor=alpha)
    for v in x:
        ev.update(v)
    record(
        "EW variance, online",
        f"river.stats.EWVar(fading_factor={alpha:g})",
        v_rv,
        "NOT exact: unadjusted EW recursion seeded at the first row",
        ev.get(),
        wcentral(x, lam_w, 2),
    )

    width = max(len(r[0]) for r in ROWS)
    cw = max(len(r[1]) for r in ROWS)
    for q, call, ver, ex, g in ROWS:
        print(f"{q:<{width}}  {call:<{cw}}  {ver:<7}  {g:9.2e}  {ex}")

    # numerical stability at a level far above the spread
    print("\nstability: x = 1e6 + N(0,1), n = 10000, central m4 (relative error vs fsum two-pass)")
    z = (1e6 + rng.standard_normal(10_000)).tolist()
    zw = [1.0] * len(z)
    ref4 = wcentral(z, zw, 4)
    n = len(z)
    s = [sum(v**r for v in z) / n for r in range(5)]  # naive raw power sums, plain summation
    m = s[1]
    naive4 = s[4] - 4 * m * s[3] + 6 * m * m * s[2] - 3 * m**4
    acc = PebayEW(4, 0)
    for v in z:
        acc.update(v, 0.0, 1.0, 1.0)
    print(f"  raw power sums: {abs(naive4 - ref4) / ref4:.2e}")
    print(f"  Pebay online:   {abs(acc.central(4, 0) - ref4) / ref4:.2e}")
    # a window by subtraction: last 500 rows = all rows minus the prefix, inverse merge
    win = 500
    pre = PebayEW(4, 0)
    for v in z[:-win]:
        pre.update(v, 0.0, 1.0, 1.0)
    W_ab, W_a = acc.M[0][0], pre.M[0][0]
    W_b = W_ab - W_a
    m_b = (acc.mx * W_ab - pre.mx * W_a) / W_b
    dlt = m_b - pre.mx
    Mb = {}
    for p in (2, 3, 4):  # invert M_p^{AB} = M_p^A + M_p^B + sum_k C(p,k) dlt^k [...] for M_p^B
        cross = fsum(
            comb(p, k)
            * dlt**k
            * (pre.M[p - k][0] * (-W_b / W_ab) ** k + Mb.get(p - k, 0.0) * (W_a / W_ab) ** k)
            for k in range(1, p - 1)
        )
        Mb[p] = (
            acc.M[p][0]
            - pre.M[p][0]
            - cross
            - dlt**p * (W_a * (-W_b / W_ab) ** p + W_b * (W_a / W_ab) ** p)
        )
    ref_w4 = wcentral(z[-win:], [1.0] * win, 4)
    direct = PebayEW(4, 0)
    for v in z[-win:]:
        direct.update(v, 0.0, 1.0, 1.0)
    print(
        f"  window of 500 by subtraction (inverse merge): {abs(Mb[4] / W_b - ref_w4) / ref_w4:.2e}"
    )
    print(
        "  window of 500 by merge of its own rows:       "
        f"{abs(direct.central(4, 0) - ref_w4) / ref_w4:.2e}"
    )
    print(
        f"\npython {sys.version.split()[0]}, numpy {v_np}, scipy {v_sp}, "
        f"statsmodels {v_sm}, polars {v_pl}, river {v_rv}"
    )


if __name__ == "__main__":
    main()
