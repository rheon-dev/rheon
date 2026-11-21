# ruff: noqa: E501  (long report strings)
"""Task 496 study, part 2: k = 5 and 20, forgetting, zero-weight rows, costs.

Builds on `scripts/study_496.py` (part 1: the model, the three forms, the k = 2
checks). This part generalizes the filter and the three forms to

    P_{t|t-1} = P_{t-1|t-1} / lam + Q        (lam = 1: part 1's random walk)
    s_t       = x_t' P_{t|t-1} x_t + R / w_t  (w_t = 0: no update at all)

and runs, each section well under 30 seconds:

    uv run python scripts/study_496_part2.py --section k       # 1: k = 5, 20
    uv run python scripts/study_496_part2.py --section forget  # 2: forgetting
    uv run python scripts/study_496_part2.py --section zero    # 3: zero-weight rows
    uv run python scripts/study_496_part2.py --section cost    # 4: cost and state

Python and numpy only; data come from seeded generators.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
from statsmodels.tsa.statespace.kalman_smoother import KalmanSmoother

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from study_496 import (  # noqa: E402
    HAVE_FILTERPY,
    gap_sd,
    rel,
    simulate,
)

if HAVE_FILTERPY:
    from filterpy.kalman import KalmanFilter

R = 1.0
LAGS = [1, 10, 100]


# ---------------------------------------------------------------- generalized filter and full RTS


def filter_gen(x, y, w, Q, R, P0, lam=1.0):
    """Filtered b_{t|t}, P_{t|t} and predictions, with forgetting and row weights.

    A zero-weight row skips the update entirely (it is not multiplied by a zero
    gain, so a NaN or infinite target on such a row cannot reach the state).
    """
    n, k = x.shape
    b = np.zeros(k)
    P = P0.copy()
    bf = np.empty((n, k))
    Pf = np.empty((n, k, k))
    pred = np.empty(n)
    for t in range(n):
        if t > 0:
            P = P / lam + Q
        z = x[t]
        pred[t] = z @ b
        if w[t] > 0:
            Pz = P @ z
            s = z @ Pz + R / w[t]
            g = Pz / s
            b = b + g * (y[t] - pred[t])
            P = P - np.outer(g, Pz)
            P = 0.5 * (P + P.T)
        bf[t] = b
        Pf[t] = P
    return bf, Pf, pred


def gains_gen(Pf, Q, lam=1.0):
    Pp = Pf / lam + Q  # P_{s+1|s}
    return np.linalg.solve(Pp, Pf).transpose(0, 2, 1), Pp


def full_rts_gen(bf, Pf, Q, lam=1.0):
    n = bf.shape[0]
    G, Pp = gains_gen(Pf, Q, lam)
    bs, Ps = bf.copy(), Pf.copy()
    for s in range(n - 2, -1, -1):
        bs[s] = bf[s] + G[s] @ (bs[s + 1] - bf[s])
        Ps[s] = Pf[s] + G[s] @ (Ps[s + 1] - Pp[s]) @ G[s].T
    return bs, Ps


def ring_vec_gen(bf, Pf, Q, L, lam=1.0):
    """Form (a) for every emission row at once; row i is the estimate for row i."""
    n = bf.shape[0]
    G, Pp = gains_gen(Pf, Q, lam)
    t = np.arange(L, n)
    xs, Ps = bf[t].copy(), Pf[t].copy()
    for j in range(1, L + 1):
        s = t - j
        xs = bf[s] + np.einsum("nij,nj->ni", G[s], xs - bf[s])
        Ps = Pf[s] + G[s] @ (Ps - Pp[s]) @ G[s].transpose(0, 2, 1)
    return xs, Ps


def pad(n, L, m, c):
    k = m.shape[1]
    pm = np.full((n, k), np.nan)
    pc = np.full((n, k, k), np.nan)
    pm[: n - L], pc[: n - L] = m, c
    return pm, pc


# ---------------------------------------------------------------- streaming forms


class Ring:
    """(a) the ring; `cache_g` keeps G_s (computed once, when row s is filtered)."""

    def __init__(self, k, L, Q, R, P0, lam=1.0, cache_g=False):
        self.k, self.L, self.Q, self.R, self.lam, self.cache_g = k, L, Q, R, lam, cache_g
        self.b, self.P = np.zeros(k), P0.copy()
        self.rb, self.rP = np.zeros((L, k)), np.zeros((L, k, k))
        self.rG = np.zeros((L, k, k)) if cache_g else None
        self.t = 0

    def step(self, z, y, w=1.0):
        Q, L, lam = self.Q, self.L, self.lam
        if self.t > 0:
            self.P = self.P / lam + Q
        pred = z @ self.b
        if w > 0:
            Pz = self.P @ z
            s = z @ Pz + self.R / w
            g = Pz / s
            self.b = self.b + g * (y - pred)
            P = self.P - np.outer(g, Pz)
            self.P = 0.5 * (P + P.T)
        t, out = self.t, None
        if t >= L:
            xs, Ps = self.b, self.P
            for j in range(1, L + 1):
                i = (t - j) % L
                bs, Pss = self.rb[i], self.rP[i]
                Pp = Pss / lam + Q
                G = self.rG[i] if self.cache_g else np.linalg.solve(Pp, Pss).T
                xs = bs + G @ (xs - bs)
                Ps = Pss + G @ (Ps - Pp) @ G.T
            out = (t - L, xs, Ps)
        if L > 0:
            i = t % L
            self.rb[i], self.rP[i] = self.b, self.P
            if self.cache_g:
                self.rG[i] = np.linalg.solve(self.P / lam + Q, self.P).T
        self.t += 1
        return pred, out


class Augmented:
    """(b) full augmented state; forgetting inflates the head block only."""

    def __init__(self, k, L, Q, R, P0, lam=1.0):
        self.k, self.L, self.Q, self.R, self.lam = k, L, Q, R, lam
        m = k * (L + 1)
        self.a, self.P = np.zeros(m), np.zeros((m, m))
        self.P[:k, :k] = P0
        self.t = 0

    def step(self, z, y, w=1.0):
        k, Q = self.k, self.Q
        if self.t > 0:
            a, P = self.a, self.P
            na = np.empty_like(a)
            na[:k], na[k:] = a[:k], a[:-k]
            nP = np.empty_like(P)
            nP[k:, k:] = P[:-k, :-k]
            nP[:k, k:] = P[:k, :-k]
            nP[k:, :k] = P[:-k, :k]
            nP[:k, :k] = P[:k, :k] / self.lam + Q
            self.a, self.P = na, nP
        pred = z @ self.a[:k]
        if w > 0:
            Ph = self.P[:, :k] @ z
            s = z @ Ph[:k] + self.R / w
            g = Ph / s
            self.a = self.a + g * (y - pred)
            P = self.P - np.outer(g, Ph)
            self.P = 0.5 * (P + P.T)
        out = None
        if self.t >= self.L:
            lo = k * self.L
            out = (self.t - self.L, self.a[lo:].copy(), self.P[lo:, lo:].copy())
        self.t += 1
        return pred, out


class Reduced:
    """(b') reduced augmented; `forget_all` divides every block by lam (a different model)."""

    def __init__(self, k, L, Q, R, P0, lam=1.0, forget_all=False):
        self.k, self.L, self.Q, self.R, self.lam, self.forget_all = k, L, Q, R, lam, forget_all
        self.a = np.zeros((L + 1, k))
        self.C = np.zeros((L + 1, k, k))
        self.D = np.zeros((L + 1, k, k))
        self.C[0] = self.D[0] = P0
        self.t = 0

    def step(self, z, y, w=1.0):
        Q, lam = self.Q, self.lam
        if self.t > 0:
            self.a[1:] = self.a[:-1].copy()
            self.C[1:] = self.C[:-1].copy()
            self.D[1:] = self.D[:-1].copy()
            if self.forget_all:
                self.C[1:] /= lam
                self.D[1:] /= lam
            self.C[0] = self.C[0] / lam + Q
            self.D[0] = self.C[0]
        a, C, D = self.a, self.C, self.D
        pred = z @ a[0]
        if w > 0:
            Ph = C @ z
            s = z @ Ph[0] + self.R / w
            g = Ph / s
            a += g * (y - pred)
            C -= np.einsum("ji,l->jil", g, Ph[0])
            D -= np.einsum("ji,jl->jil", g, Ph)
            C[0] = 0.5 * (C[0] + C[0].T)
            D[0] = C[0]
        out = None
        if self.t >= self.L:
            out = (self.t - self.L, a[self.L].copy(), 0.5 * (D[self.L] + D[self.L].T))
        self.t += 1
        return pred, out


def run(sm, x, y, w=None):
    n, k = x.shape
    if w is None:
        w = np.ones(n)
    means = np.full((n, k), np.nan)
    covs = np.full((n, k, k), np.nan)
    preds = np.empty(n)
    for t in range(n):
        preds[t], out = sm.step(x[t], y[t], w[t])
        if out is not None:
            s, m, c = out
            means[s], covs[s] = m, c
    return means, covs, preds


# ---------------------------------------------------------------- third-party oracles


def sm_smoother(x, y, w, Q, R, P0, state_cov=None):
    """statsmodels KalmanSmoother; w = 0 rows are missing (NaN), R_t = R / w_t.

    state_cov, if given, is (k, k, n): the noise between row t and t + 1 at [..., t].
    """
    n, k = x.shape
    yy = np.where(w > 0, y, np.nan).reshape(n, 1).copy()
    ks = KalmanSmoother(k_endog=1, k_states=k, k_posdef=k)
    ks.bind(yy)
    ks["design"] = x.T.reshape(1, k, n).copy()
    ks["obs_cov"] = (R / np.where(w > 0, w, 1.0)).reshape(1, 1, n).copy()
    ks["transition"] = np.eye(k)
    ks["selection"] = np.eye(k)
    ks["state_cov"] = Q if state_cov is None else state_cov.copy()
    ks.initialize_known(np.zeros(k), P0)
    res = ks.smooth()
    return (
        res.smoothed_state.T,
        res.smoothed_state_cov.transpose(2, 0, 1),
        res.filtered_state.T,
        res.filtered_state_cov.transpose(2, 0, 1),
    )


def fp_smoother(x, y, w, Q, R, P0, lam=1.0):
    """filterpy: forgetting is its fading-memory `alpha` (alpha^2 = 1 / lam) in the
    filter; its RTS smoother has no alpha, so it is given Qs[t+1] = (alpha^2 - 1) P_t + Q
    from filterpy's own filtered covariances. w = 0 rows pass z = None (no update)."""
    n, k = x.shape
    kf = KalmanFilter(dim_x=k, dim_z=1)
    kf.x = np.zeros((k, 1))
    kf.P = P0.copy()
    kf.F = np.eye(k)
    kf.Q = Q
    kf.R = np.array([[R]])
    kf.alpha = 1.0 / np.sqrt(lam)
    Hs = [x[t].reshape(1, k) for t in range(n)]
    Rs = [np.array([[R / w[t]]]) if w[t] > 0 else np.array([[R]]) for t in range(n)]
    zs = [np.array([[y[t]]]) if w[t] > 0 else None for t in range(n)]
    mu, cov, _, _ = kf.batch_filter(zs, Hs=Hs, Rs=Rs, update_first=True)
    a2 = 1.0 / lam
    Qs = [Q] + [(a2 - 1.0) * cov[t - 1] + Q for t in range(1, n)]
    xs, Ps, _, _ = kf.rts_smoother(mu, cov, Qs=Qs)
    return xs.reshape(n, k), Ps, mu.reshape(n, k), cov


# ---------------------------------------------------------------- sizes


def sizes(k, L):
    tri = k * (k + 1) // 2
    m = k * (L + 1)
    return {
        "ring": (k + tri) + L * (k + tri),
        "ring, G cached": (k + tri) + L * (k + tri + k * k),
        "augmented": m + m * (m + 1) // 2,
        "reduced": m + tri + L * (k * k + tri),
    }


# ---------------------------------------------------------------- 1. k = 5, 20


def section_k():
    n, seeds = 2000, [0, 1, 2, 3, 4]
    print(f"filterpy available: {HAVE_FILTERPY}")
    print(
        "\n## Exactness on the prefix: emitted value vs statsmodels / filterpy (q=1e-3, seed 0, N=2000)"
    )
    for k in [5, 20]:
        P0 = np.eye(k) * 10.0
        q = 1e-3
        Q = np.eye(k) * q
        x, y = simulate(n, k, q, R, 0)
        w = np.ones(n)
        bf, Pf, _ = filter_gen(x, y, w, Q, R, P0)
        for L in LAGS:
            vm, vc = pad(n, L, *ring_vec_gen(bf, Pf, Q, L))
            wsm = wfp = 0.0
            t0 = time.perf_counter()
            for t in [L, 500, 1234, 1999]:
                s = t - L
                sm_m, sm_c, _, _ = sm_smoother(x[: t + 1], y[: t + 1], w[: t + 1], Q, R, P0)
                wsm = max(wsm, rel(vm[s], sm_m[s]), rel(vc[s], sm_c[s]))
                if HAVE_FILTERPY:
                    fm, fc, _, _ = fp_smoother(x[: t + 1], y[: t + 1], w[: t + 1], Q, R, P0)
                    wfp = max(wfp, rel(vm[s], fm[s]), rel(vc[s], fc[s]))
            print(
                f"k={k} L={L}: max rel diff (mean and cov, rows t in {{L,500,1234,1999}}) vs statsmodels {wsm:.1e}, "
                f"vs filterpy {wfp:.1e}  ({time.perf_counter() - t0:.1f} s)"
            )

    print(
        "\n## Forms agree; gap to the full smoother (N=2000, 5 seeds; RMS and max over rows >= 500)"
    )
    print(
        "| k | q | L | ring vs aug | ring vs reduced | ring stream vs vectorized (seed 0) "
        "| gap max, rows >= 500 | gap RMS, rows >= 500 | RMS predicted by the covariances |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    for k in [5, 20]:
        P0 = np.eye(k) * 10.0
        for q in [1e-2, 1e-3]:
            Q = np.eye(k) * q
            for L in LAGS:
                t0 = time.perf_counter()
                run_aug = k * (L + 1) <= 600
                gaps, d_aug, d_red, d_vec = [], 0.0, 0.0, 0.0
                for seed in seeds:
                    x, y = simulate(n, k, q, R, seed)
                    w = np.ones(n)
                    bf, Pf, _ = filter_gen(x, y, w, Q, R, P0)
                    bs, Ps = full_rts_gen(bf, Pf, Q)
                    lm, lc = pad(n, L, *ring_vec_gen(bf, Pf, Q, L))
                    if run_aug:
                        am, ac, _ = run(Augmented(k, L, Q, R, P0), x, y)
                        d_aug = max(d_aug, rel(lm, am), rel(lc, ac))
                    em, ec, _ = run(Reduced(k, L, Q, R, P0), x, y)
                    d_red = max(d_red, rel(lm, em), rel(lc, ec))
                    if seed == 0:
                        rm, rc, _ = run(Ring(k, L, Q, R, P0), x, y)
                        d_vec = rel(lm, rm), rel(lc, rc)
                        d_vec = max(d_vec)
                    gaps.append(gap_sd(lm[: n - L], lc[: n - L], bs[: n - L], Ps[: n - L]))
                g = np.array(gaps)
                aug = f"{d_aug:.1e}" if run_aug else "not run"
                print(
                    f"| {k} | {q:g} | {L} | {aug} | {d_red:.1e} | {d_vec:.1e} | "
                    f"{g[:, 1].max():.3g} | {g[:, 2].max():.3g} | {g[:, 3].max():.3g} |"
                    f"  <!-- {time.perf_counter() - t0:.1f} s -->"
                )


# ---------------------------------------------------------------- 2. forgetting


def section_forget():
    n, k = 2000, 5
    P0 = np.eye(k) * 10.0
    print(f"filterpy available: {HAVE_FILTERPY}")
    print("\n## Forgetting: P_{t|t-1} = P_{t-1|t-1} / lam + Q (k=5, N=2000)")
    print(
        "| q | lam | L | ring vs hand full RTS on prefix | ring vs statsmodels on prefix | ring vs filterpy on prefix "
        "| ring vs reduced | ring vs aug | reduced, all blocks / lam, vs ring | gap RMS, rows >= 500 (5 seeds) | RMS predicted by the covariances |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for q in [1e-3, 0.0]:
        Q = np.eye(k) * q
        for lam in [0.999, 0.99]:
            for L in LAGS:
                t0 = time.perf_counter()
                gaps = []
                for seed in range(5):
                    # with q = 0 the data still drift (q = 1e-3); only the model forgets
                    x, y = simulate(n, k, q if q > 0 else 1e-3, R, seed)
                    w = np.ones(n)
                    bf, Pf, _ = filter_gen(x, y, w, Q, R, P0, lam)
                    bs, Ps = full_rts_gen(bf, Pf, Q, lam)
                    lm, lc = pad(n, L, *ring_vec_gen(bf, Pf, Q, L, lam))
                    gaps.append(gap_sd(lm[: n - L], lc[: n - L], bs[: n - L], Ps[: n - L]))
                    if seed:
                        continue
                    rm, rc, _ = run(Ring(k, L, Q, R, P0, lam), x, y)
                    d_hand = d_sm = d_fp = 0.0
                    for t in [L, 500, 1234, 1999]:
                        s = t - L
                        hb, hP = full_rts_gen(bf[: t + 1], Pf[: t + 1], Q, lam)
                        d_hand = max(d_hand, rel(rm[s], hb[s]), rel(rc[s], hP[s]))
                        sc = np.empty((k, k, t + 1))
                        sc[:] = ((1.0 / lam - 1.0) * Pf[: t + 1] + Q).transpose(1, 2, 0)
                        sm_m, sm_c, _, _ = sm_smoother(
                            x[: t + 1], y[: t + 1], w[: t + 1], Q, R, P0, sc
                        )
                        d_sm = max(d_sm, rel(rm[s], sm_m[s]), rel(rc[s], sm_c[s]))
                        if HAVE_FILTERPY:
                            fm, fc, _, _ = fp_smoother(
                                x[: t + 1], y[: t + 1], w[: t + 1], Q, R, P0, lam
                            )
                            d_fp = max(d_fp, rel(rm[s], fm[s]), rel(rc[s], fc[s]))
                    em, ec, _ = run(Reduced(k, L, Q, R, P0, lam), x, y)
                    d_red = max(rel(rm, em), rel(rc, ec))
                    if L <= 10:
                        am, ac, _ = run(Augmented(k, L, Q, R, P0, lam), x, y)
                        d_aug = f"{max(rel(rm, am), rel(rc, ac)):.1e}"
                    else:
                        d_aug = "n/a"
                    fm2, _, _ = run(Reduced(k, L, Q, R, P0, lam, forget_all=True), x, y)
                    d_all = rel(rm, fm2)
                g = np.array(gaps)
                print(
                    f"| {q:g} | {lam} | {L} | {d_hand:.1e} | {d_sm:.1e} | {d_fp:.1e} | {d_red:.1e} | {d_aug} | "
                    f"{d_all:.2g} | {g[:, 2].max():.3g} | {g[:, 3].max():.3g} |  <!-- {time.perf_counter() - t0:.1f} s -->"
                )
    # q = 0: G = lam I exactly
    Q0 = np.zeros((k, k))
    x, y = simulate(n, k, 1e-3, R, 0)
    bf, Pf, _ = filter_gen(x, y, np.ones(n), Q0, R, P0, 0.99)
    G, _ = gains_gen(Pf, Q0, 0.99)
    print(f"\nq=0, lam=0.99: max |G_s - lam I| = {np.max(np.abs(G - 0.99 * np.eye(k))):.1e}")
    # the filtered covariances of the three filters under forgetting
    x, y = simulate(n, k, 1e-3, R, 0)
    w = np.ones(n)
    Q = np.eye(k) * 1e-3
    bf, Pf, _ = filter_gen(x, y, w, Q, R, P0, 0.99)
    sc = np.empty((k, k, n))
    sc[:] = ((1.0 / 0.99 - 1.0) * Pf + Q).transpose(1, 2, 0)
    _, _, smf, smP = sm_smoother(x, y, w, Q, R, P0, sc)
    line = f"q=1e-3, lam=0.99 filtered: own vs statsmodels mean {rel(bf, smf):.1e} cov {rel(Pf, smP):.1e}"
    if HAVE_FILTERPY:
        _, _, fpm, fpP = fp_smoother(x, y, w, Q, R, P0, 0.99)
        line += f"; own vs filterpy (alpha) mean {rel(bf, fpm):.1e} cov {rel(Pf, fpP):.1e}"
    print(line)


# ---------------------------------------------------------------- 3. zero-weight rows


def section_zero():
    n, k, q = 2000, 5, 1e-3
    P0 = np.eye(k) * 10.0
    Q = np.eye(k) * q
    x, y = simulate(n, k, q, R, 0)
    rng = np.random.default_rng(7)
    w = np.ones(n)
    w[0] = 0.0  # the first row
    w[rng.choice(np.arange(1, n), 200, replace=False)] = 0.0  # 10 % scattered
    w[700:721] = 0.0  # a run of 21
    zero = w == 0
    y_nan = np.where(zero, np.nan, y)  # a zero-weight target may be anything, NaN included
    y_big = np.where(zero, 1e300, y)
    print(f"## Zero-weight rows (k=5, q=1e-3, N=2000): {int(zero.sum())} rows, row 0 among them")

    bf, Pf, pf = filter_gen(x, y_nan, w, Q, R, P0)
    # the dropped stream: zero-weight rows removed, the clock carried as Q * gap
    keep = np.flatnonzero(~zero)
    gap = np.diff(np.concatenate([[-1], keep]))  # clock units from the previous kept row
    bd = np.zeros(k)
    Pd = P0.copy()
    bfd = np.empty((len(keep), k))
    Pfd = np.empty((len(keep), k, k))
    for i, t in enumerate(keep):
        # the first kept row's prior: P0 advanced over the zero rows before it
        if gap[i] > 1 or i > 0:
            Pd = Pd + Q * (gap[i] if i > 0 else gap[i] - 1)
        z = x[t]
        Pz = Pd @ z
        g = Pz / (z @ Pz + R)
        bd = bd + g * (y[t] - z @ bd)
        Pd = Pd - np.outer(g, Pz)
        Pd = 0.5 * (Pd + Pd.T)
        bfd[i], Pfd[i] = bd, Pd
    print(
        f"filter: kept rows vs the dropped stream with the clock carried: mean {rel(bf[keep], bfd):.1e}, "
        f"cov {rel(Pf[keep], Pfd):.1e}"
    )

    for L in LAGS:
        t0 = time.perf_counter()
        res = {}
        for name, cls in [("ring", Ring), ("reduced", Reduced), ("augmented", Augmented)]:
            if name == "augmented" and L > 10:
                continue
            res[name] = run(cls(k, L, Q, R, P0), x, y_nan, w)
        rm, rc, rp = res["ring"]
        nan_any = any(
            np.isnan(m[: n - L]).any() or np.isnan(c[: n - L]).any() or np.isnan(p).any()
            for m, c, p in res.values()
        )
        rm2, rc2, rp2 = run(Ring(k, L, Q, R, P0), x, y_big, w)
        same_y = (
            np.array_equal(rm, rm2, equal_nan=True)  # NaN only in the unemitted last L rows
            and np.array_equal(rc, rc2, equal_nan=True)
            and np.array_equal(rp, rp2)
        )
        d_red = max(rel(rm, res["reduced"][0]), rel(rc, res["reduced"][1]))
        d_aug = (
            f"{max(rel(rm, res['augmented'][0]), rel(rc, res['augmented'][1])):.1e}"
            if "augmented" in res
            else "n/a"
        )
        vm, vc = pad(n, L, *ring_vec_gen(bf, Pf, Q, L))
        d_vec = max(rel(rm, vm), rel(rc, vc))
        # oracle: statsmodels (missing y) and filterpy (z = None) on the prefix
        d_sm = d_fp = 0.0
        for t in [L, L + 1, 705, 720, 1234, 1999]:
            s = t - L
            sm_m, sm_c, _, _ = sm_smoother(x[: t + 1], y[: t + 1], w[: t + 1], Q, R, P0)
            d_sm = max(d_sm, rel(rm[s], sm_m[s]), rel(rc[s], sm_c[s]))
            if HAVE_FILTERPY:
                fm, fc, _, _ = fp_smoother(x[: t + 1], y[: t + 1], w[: t + 1], Q, R, P0)
                d_fp = max(d_fp, rel(rm[s], fm[s]), rel(rc[s], fc[s]))
        # vs the dropped stream: the emission for row r at t = r + L conditions on the
        # kept rows <= t; run the dropped stream's RTS from its last kept row <= t back to r.
        Gd = np.linalg.solve(Pfd + Q, Pfd).transpose(0, 2, 1)  # one clock unit of Q
        pos = np.cumsum(~zero) - 1  # index of the last kept row <= t in the dropped stream
        d_drop = 0.0
        interp = 0.0
        for r in range(n - L):
            t = r + L
            if zero[r]:
                continue
            i_r, i_t = pos[r], pos[t]
            m, P = bfd[i_t], Pfd[i_t]
            for i in range(i_t - 1, i_r - 1, -1):
                hq = Q * gap[i + 1]
                Pp = Pfd[i] + hq
                G = Gd[i] if gap[i + 1] == 1 else np.linalg.solve(Pp, Pfd[i]).T
                m = bfd[i] + G @ (m - bfd[i])
                P = Pfd[i] + G @ (P - Pp) @ G.T
            d_drop = max(d_drop, rel(rm[r], m), rel(rc[r], P))
        # what a zero-weight row emits: one RTS step from the row after it, from the
        # predicted pair; between two kept rows, a linear interpolation in clock time
        # of the neighbours' smoothed means given the same y_1..y_t.
        isolated = [
            r for r in np.flatnonzero(zero[1 : n - L - 1]) + 1 if not (zero[r - 1] or zero[r + 1])
        ]
        for r in isolated[:20]:
            t = r + L
            hb, _ = full_rts_gen(bf[: t + 1], Pf[: t + 1], Q)
            interp = max(interp, rel(rm[r], 0.5 * (hb[r - 1] + hb[r + 1])))
            if interp > 1e-12:
                break
        print(
            f"L={L}: NaN anywhere (emitted, preds, all forms, y = NaN on zero rows): {nan_any}; "
            f"zero-row y = NaN vs 1e300 bit-identical: {same_y}; ring vs reduced {d_red:.1e}, vs aug {d_aug}, "
            f"vs vectorized {d_vec:.1e}; vs statsmodels (missing y) {d_sm:.1e}, vs filterpy (z=None) {d_fp:.1e}; "
            f"kept rows vs the dropped stream with clock carried {d_drop:.1e}; "
            f"isolated zero row = midpoint of neighbours' smoothed means {interp:.1e}  ({time.perf_counter() - t0:.1f} s)"
        )
    # the first row: emitted value at t = L for row 0
    L = 10
    rm, rc, _ = run(Ring(k, L, Q, R, P0), x, y_nan, w)
    hb, hP = full_rts_gen(bf[: L + 1], Pf[: L + 1], Q)
    G0 = np.linalg.solve(P0 + Q, P0).T
    step = G0 @ hb[1]  # b_{0|0} = 0
    print(
        f"row 0 zero-weight, L=10: emitted = G_0 b_(1|10) with b_(0|0) = 0, G_0 = P0 (P0 + Q)^-1: "
        f"{rel(rm[0], step):.1e}; = full RTS on y_1..y_10 at row 0: {rel(rm[0], hb[0]):.1e}"
    )


# ---------------------------------------------------------------- 4. cost and state


def section_cost():
    print(
        "## Per-row cost (numpy, one Python step per row, timed after the first L rows) and state size"
    )
    print("| k | L | form | us/row | rows timed | floats in state | bytes |")
    print("|---|---|---|---|---|---|---|")
    q = 1e-3
    for k in [5, 20]:
        P0 = np.eye(k) * 10.0
        Q = np.eye(k) * q
        x, y = simulate(2200, k, q, R, 0)
        for L in LAGS:
            sz = sizes(k, L)
            for name in ["ring", "ring, G cached", "augmented", "reduced"]:
                if name == "augmented" and k * (L + 1) > 600:
                    m = k * (L + 1)
                    print(
                        f"| {k} | {L} | {name} | not run | 0 | {sz[name]} | {8 * sz[name]:,} "
                        f"(formula; a full square covariance would be {8 * (m * m + m):,}) |"
                    )
                    continue
                if name == "ring":
                    sm = Ring(k, L, Q, R, P0)
                elif name == "ring, G cached":
                    sm = Ring(k, L, Q, R, P0, cache_g=True)
                elif name == "augmented":
                    sm = Augmented(k, L, Q, R, P0)
                else:
                    sm = Reduced(k, L, Q, R, P0)
                for t in range(L):
                    sm.step(x[t], y[t])
                nt = (
                    300
                    if (name.startswith("ring") and L == 100) or name == "augmented" and L == 100
                    else 1000
                )
                t1 = time.perf_counter()
                for t in range(L, L + nt):
                    sm.step(x[t], y[t])
                us = (time.perf_counter() - t1) / nt * 1e6
                print(f"| {k} | {L} | {name} | {us:.0f} | {nt} | {sz[name]} | {8 * sz[name]:,} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--section", choices=["k", "forget", "zero", "cost"], required=True)
    args = ap.parse_args()
    {"k": section_k, "forget": section_forget, "zero": section_zero, "cost": section_cost}[
        args.section
    ]()


if __name__ == "__main__":
    main()
