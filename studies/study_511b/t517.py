"""Task 517: `ew_cov` outputs, the uncentered covariance and a Gaussian log-likelihood.

Beside `ew_cov`'s mean-form recursion (`EwMoments`): everything below is a read of the
accumulator it already keeps, so no new state.

    uncentered second moment   R = C + m m'                (divisor W: the weighted mean of x x')
    Gaussian log-likelihood    ll = -0.5 (k ln 2 pi + ln|S| + q),   q = d' S^-1 d,  d = x - m
        with S = c (C + s * prior * I), the matrix `mahal` already inverts, `c` the unbiased
        factor (1 where `bias`), read BEFORE the row folds in (out-of-sample, rule 2):
        `q` is the square of `mahal`.
    RiskMetrics EWMA           S' = lam S + (1 - lam) x x', the zero-mean recursion with a
        fixed weight on the row; `R` is the same with the row's weight `1/W`.
    rcov(kind="plain")         sum x x' = n R at lam = 1, unit weights.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.stats import multivariate_normal
from sklearn.covariance import EmpiricalCovariance

from .common import EwMoments, dt_stream, lam_of, ulps, stream, table, two_pass


def _run(x, dt, w, hl, upto=None):
    k = x.shape[1]
    ew = EwMoments(k)
    out = []
    for t in range(len(x) if upto is None else upto + 1):
        lam = 1.0 if t == 0 else lam_of(dt[t], hl)
        if t == 0:
            lam = 1.0
        # read BEFORE the row folds in (the log-likelihood's side), keep the copy
        out.append((ew.W, ew.Q, ew.m.copy(), ew.C.copy(), ew.s, ew.unbiased_factor()))
        ew.update(x[t], lam, w[t])
    return ew, out


def loglik(x, m, C, s, c, prior, bias):
    """The row's Gaussian log-density under the pre-update fit. `S = c (C + s prior I)`."""
    k = len(x)
    S = (C + s * prior * np.eye(k)) * (1.0 if bias else c)
    sign, logdet = np.linalg.slogdet(S)
    d = x - m
    q = float(d @ np.linalg.solve(S, d))
    return -0.5 * (k * np.log(2 * np.pi) + logdet + q), q


def report(seeds: int = 20) -> None:
    warnings.filterwarnings("ignore")
    rel_rows = []
    # --- uncentered second moment vs sklearn assume_centered=True (expanding, unit weights)
    for k in (2, 5, 20):
        gap = 0.0
        for sd in range(seeds):
            x, _, _, _ = stream(sd, 300, k, offset=0.0)
            ew = EwMoments(k)
            for t in range(len(x)):
                ew.update(x[t], 1.0, 1.0)
            ref = EmpiricalCovariance(assume_centered=True).fit(x).covariance_
            gap = max(gap, np.max(np.abs(ew.raw() - ref) / np.max(np.abs(ref))))
            # plain = n R
            gap = max(gap, np.max(np.abs(len(x) * ew.raw() - x.T @ x) / np.max(np.abs(x.T @ x))))
        rel_rows.append([f"lam=1 unit w, k={k}", gap])
    table("517a uncentered covariance vs sklearn `EmpiricalCovariance(assume_centered=True)` and `sum x x'` (relative to max entry)",
          ["case", "largest gap"], rel_rows)

    # --- decayed, weighted, zero-weight rows, offset 1e3: against two-pass definition
    rows = []
    for k in (2, 5, 20):
        g = 0.0
        gl = 0.0
        for sd in range(seeds):
            x, _, _, _ = stream(sd, 250, k, offset=1e3)
            dt, w = dt_stream(sd, 250)
            ew = EwMoments(k)
            for t in range(len(x)):
                ew.update(x[t], 1.0 if t == 0 else lam_of(dt[t], 20.0), w[t])
                if t in (60, 120, 249):
                    W, Q, m, C, raw = two_pass(x, dt, w, 20.0, t)
                    g = max(g, np.max(np.abs(ew.raw() - raw)) / np.max(np.abs(raw)))
        rows.append([f"half-life 20, w in [.5,2], 10% zero w, zero first row, offset 1e3, k={k}", g])
    table("517b uncentered covariance vs the two-pass weighted `sum omega x x' / sum omega` (relative to max entry)",
          ["case", "largest gap"], rows)

    # --- window 60 by retraction vs sklearn on the window
    rows = []
    for k in (2, 5, 20):
        g = 0.0
        for sd in range(seeds):
            x, _, _, _ = stream(sd, 300, k)
            ew = EwMoments(k)
            for t in range(len(x)):
                ew.update(x[t], 1.0, 1.0)
                if t >= 60:
                    ew.update(x[t - 60], 1.0, -1.0)
                if t >= 100 and t % 40 == 0:
                    ref = EmpiricalCovariance(assume_centered=True).fit(x[t - 59 : t + 1]).covariance_
                    g = max(g, np.max(np.abs(ew.raw() - ref)) / np.max(np.abs(ref)))
        rows.append([f"window 60 by retraction, k={k}", g])
    table("517c windowed uncentered covariance vs sklearn on the window", ["case", "largest gap"], rows)

    # --- RiskMetrics EWMA transient
    rows = []
    lam = 0.94
    x, _, _, _ = stream(1, 3000, 3)
    ew = EwMoments(3)
    S = None
    gaps = {}
    for t in range(len(x)):
        ew.update(x[t], lam, 1.0)
        S = np.outer(x[t], x[t]) if S is None else lam * S + (1 - lam) * np.outer(x[t], x[t])
        if t + 1 in (5, 20, 50, 100, 200, 1000):
            gaps[t + 1] = np.max(np.abs(ew.raw() - S)) / np.max(np.abs(S))
    # seeded with the weight 1/(1-lam) the mean form gives it
    ew2 = EwMoments(3)
    ew2.W = 1.0 / (1 - lam)
    S2 = np.zeros((3, 3))
    for t in range(len(x)):
        ew2.update(x[t], lam, 1.0)
        S2 = lam * S2 + (1 - lam) * np.outer(x[t], x[t])
    rows.append(["lam .94, seeded x1 x1'", *[gaps[n] for n in (5, 20, 50, 100, 200, 1000)]])
    rows.append(["lam .94, mean form seeded at W0 = 1/(1-lam), S0 = 0 (final row)", np.max(np.abs(ew2.raw() - S2)) / np.max(np.abs(S2)), "", "", "", "", ""])
    table("517d uncentered covariance vs RiskMetrics' EWMA `S' = lam S + (1-lam) x x'` (relative to max entry)",
          ["case", "t=5", "t=20", "t=50", "t=100", "t=200", "t=1000"], rows)

    # --- Gaussian log-likelihood vs scipy logpdf, sklearn score on one row
    rows = []
    for bias in (True, False):
        for prior in (0.0, 1e-3):
            for hl in (np.inf, 30.0):
                g = 0.0
                gs = 0.0
                q_g = 0.0
                for sd in range(seeds):
                    k = 4
                    x, _, _, _ = stream(sd, 200, k)
                    dt, w = dt_stream(sd, 200, zero_first=hl != np.inf, zero_frac=0.1 if hl != np.inf else 0.0)
                    ew = EwMoments(k)
                    for t in range(len(x)):
                        W, Q, m, C, s, c = ew.W, ew.Q, ew.m.copy(), ew.C.copy(), ew.s, ew.unbiased_factor()
                        if t >= 30 and W > 0 and (bias or np.isfinite(c)):
                            ll, q = loglik(x[t], m, C, s, c, prior, bias)
                            S = (C + s * prior * np.eye(k)) * (1.0 if bias else c)
                            ref = multivariate_normal.logpdf(x[t], mean=m, cov=S, allow_singular=True)
                            g = max(g, abs(ll - ref) / max(1.0, abs(ref)))
                            q_ref = -2.0 * ref - k * np.log(2 * np.pi) - np.linalg.slogdet(S)[1]
                            q_g = max(q_g, abs(q - q_ref) / max(1.0, abs(q_ref)))
                            if hl == np.inf and prior == 0.0 and bias:
                                ec = EmpiricalCovariance().fit(x[:t])
                                # one-row score: -0.5(d'Pd + k ln 2pi - ln|P|)
                                gs = max(gs, abs(ec.score(x[t][None, :]) - ll) / max(1.0, abs(ll)))
                        lam = 1.0 if (t == 0 or hl == np.inf) else lam_of(dt[t], hl)
                        ew.update(x[t], lam, 1.0 if hl == np.inf else w[t])
                rows.append([f"bias={bias}, prior={prior}, hl={hl}", g, q_g, gs if (hl == np.inf and prior == 0.0 and bias) else ""])
    table("517e Gaussian log-likelihood vs `scipy.stats.multivariate_normal.logpdf` (relative, floor 1); squared `mahal` vs the quadratic form; sklearn one-row `score`",
          ["case", "ll gap", "q gap (rel)", "sklearn `score` gap"], rows)

    # --- zero weight rows: the folded stream
    rows = []
    g = 0.0
    for sd in range(seeds):
        k = 3
        x, _, _, _ = stream(sd, 150, k)
        dt, w = dt_stream(sd, 150, zero_first=True, zero_frac=0.2)
        ew = EwMoments(k)
        for t in range(len(x)):
            ew.update(x[t], 1.0 if t == 0 else lam_of(dt[t], 25.0), w[t])
        keep = [t for t in range(len(x)) if w[t] > 0]
        # same stream with the zero-weight rows removed and their gaps folded into the next kept row
        ew2 = EwMoments(k)
        carry = 0.0
        first = True
        for t in range(len(x)):
            carry += 0.0 if t == 0 else dt[t]
            if w[t] > 0:
                ew2.update(x[t], 1.0 if first else lam_of(carry, 25.0), w[t])
                first = False
                carry = 0.0
        g = max(g, ulps(ew.raw(), ew2.raw()), 0.0)
    rows.append(["20% zero-weight rows and a zero-weight first row, R (ulps)", g])
    table("517f zero-weight rows against the same stream with their gaps folded forward", ["case", "gap"], rows)
    from .t517_state import state_note
    state_note()
