# ruff: noqa: E501  (long report strings)
"""Task 496 study: a fixed-lag smoother beside `kalman` (parts 1 to 3, one script).

Random-walk-coefficient regression, with an optional forgetting factor and row
weights (part 2):

    beta_t = beta_{t-1} + eta_t,   eta_t ~ N(0, Q)
    y_t    = x_t' beta_t + eps_t,  eps_t ~ N(0, R / w_t)    (w_t = 0: no update at all)
    beta_1 ~ N(0, P0)              (the prior at the first row; no Q before it)
    P_{t|t-1} = P_{t-1|t-1} / lam + Q                         (lam = 1: no forgetting)

At row t the lag-L smoother emits E[beta_{t-L} | y_1..y_t] (and its covariance).
Three forms are built and checked against each other, against a full
Rauch-Tung-Striebel (RTS) smoother on the prefix y_1..y_t (statsmodels and
filterpy), and the gap to the full smoother on all N rows is measured:

(a) ring:      the last L filtered means and covariances, and a backward RTS
               pass of length L at each row (optionally with G_s cached);
(b) augmented: the state [beta_t, ..., beta_{t-L}] with a shift transition,
               filtered as an ordinary Kalman filter (full covariance);
(b') reduced augmented: only the blocks (b) actually reads -- the first block
               column cov(beta_{t-j}, beta_t) and the diagonal blocks.

Python and numpy only; data are generated from seeded generators. Run:

    uv run python scripts/study_496.py --quick            # every check, short (about a minute)
    uv run python scripts/study_496.py                    # every check, the recorded sizes
    uv run python scripts/study_496.py --section seeds --seeds 50 --workers 3
                                                          # the tolerance table alone
    uv run python scripts/study_496.py --section exact --section zero   # chosen sections

Sections: full, exact, forms, contraction, oos, forget, zero, cost, seeds.
Every section records pass/fail checks; the run exits 1 if any fails.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from statsmodels.tsa.statespace.kalman_smoother import KalmanSmoother

try:
    from filterpy.kalman import KalmanFilter

    HAVE_FILTERPY = True
except ImportError:  # pragma: no cover - reported in the output
    HAVE_FILTERPY = False

R = 1.0
LAGS = [1, 10, 100]
BURN = 500  # rows from the start excluded from the steady-state columns

# ---------------------------------------------------------------- checks

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(ok), detail))
    if not ok:
        print(f"  CHECK FAILED: {name} {detail}")


# ---------------------------------------------------------------- data


def simulate(n: int, k: int, q: float, r: float, seed: int):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, k))
    beta = np.cumsum(rng.standard_normal((n, k)) * np.sqrt(q), axis=0) + rng.standard_normal(k)
    y = np.einsum("tk,tk->t", x, beta) + rng.standard_normal(n) * np.sqrt(r)
    return x, y


# ---------------------------------------------------------------- filter and full RTS


def filter_gen(x, y, w, Q, R, P0, lam=1.0):
    """Filtered b_{t|t}, P_{t|t} and the out-of-sample predictions.

    A zero-weight row skips the update entirely (it is not multiplied by a zero
    gain, so a NaN or infinite target on such a row cannot reach the state).
    With lam = 1 and w = 1 every operation is part 1's, bit for bit.
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
        pred[t] = z @ b  # before the update with y_t
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
    """G_s = P_{s|s} P_{s+1|s}^{-1}, fixed once row s is filtered; and P_{s+1|s}."""
    Pp = Pf / lam + Q
    # G = Pf Pp^{-1}  <=>  G' = Pp^{-1} Pf (both symmetric)
    return np.linalg.solve(Pp, Pf).transpose(0, 2, 1), Pp


def full_rts_gen(bf, Pf, Q, lam=1.0):
    n = bf.shape[0]
    G, Pp = gains_gen(Pf, Q, lam)
    bs, Ps = bf.copy(), Pf.copy()
    for s in range(n - 2, -1, -1):
        bs[s] = bf[s] + G[s] @ (bs[s + 1] - bf[s])
        Ps[s] = Pf[s] + G[s] @ (Ps[s + 1] - Pp[s]) @ G[s].T
    return bs, Ps


def ring_vec_gen(bf, Pf, Q, L, lam=1.0, cov=True):
    """Form (a) for every emission row at once (the same arithmetic per row).

    Row i of each output is the estimate for row i (= t - L)."""
    n = bf.shape[0]
    G, Pp = gains_gen(Pf, Q, lam)
    t = np.arange(L, n)
    xs = bf[t].copy()
    Ps = Pf[t].copy() if cov else None
    for j in range(1, L + 1):
        s = t - j
        xs = bf[s] + np.einsum("nij,nj->ni", G[s], xs - bf[s])
        if cov:
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
    """(a) the ring of the last L filtered (b, P); `cache_g` keeps G_s.

    Per row: the filter's step, then a backward RTS pass over the ring. The
    ring holds b_{s|s} and P_{s|s} for s = t-L..t-1; row t's own filtered
    state is the filter's. Emits (row t-L, mean, cov) once t >= L.
    """

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
            for j in range(1, L + 1):  # s = t-j, newest first
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
    """(b) state [beta_t, ..., beta_{t-L}], size m = k(L+1); full m x m covariance.

    Forgetting inflates the head block only."""

    def __init__(self, k, L, Q, R, P0, lam=1.0):
        self.k, self.L, self.Q, self.R, self.lam = k, L, Q, R, lam
        m = k * (L + 1)
        self.a, self.P = np.zeros(m), np.zeros((m, m))
        # The lagged slots start at zero with zero covariance: their gain is 0
        # until a real state shifts in, and none is read before row L.
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
            Ph = self.P[:, :k] @ z  # P H', H = [z', 0, ..., 0]
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
    """(b') only the blocks of (b) that are read.

    C_j = cov(beta_{t-j}, beta_t | y_1..t)  (k x k, j = 1..L)
    D_j = cov(beta_{t-j} | y_1..t)          (k x k symmetric, j = 1..L)
    a_j = E[beta_{t-j} | y_1..t]            (k, j = 1..L)
    plus the filter's own b = a_0 and P = D_0 = C_0.
    `forget_all` divides every block by lam (a different model, kept as a foil).
    """

    def __init__(self, k, L, Q, R, P0, lam=1.0, forget_all=False):
        self.k, self.L, self.Q, self.R, self.lam, self.forget_all = k, L, Q, R, lam, forget_all
        self.a = np.zeros((L + 1, k))
        self.C = np.zeros((L + 1, k, k))
        self.D = np.zeros((L + 1, k, k))
        self.C[0] = P0
        self.D[0] = P0
        self.t = 0

    def step(self, z, y, w=1.0):
        Q, lam = self.Q, self.lam
        if self.t > 0:
            # shift: beta_{t-j} <- beta_{(t-1)-(j-1)}; cov(beta_{t-j}, beta_t) =
            # cov(beta_{t-j}, beta_{t-1}) because eta_t is independent of the past.
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
            Ph = C @ z  # (L+1, k): cov(beta_{t-j}, beta_t) z
            s = z @ Ph[0] + self.R / w
            g = Ph / s
            a += g * (y - pred)
            # C_j <- C_j - g_j (P z)'  ;  D_j <- D_j - g_j (C_j z)'
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
    from filterpy's own filtered covariances. w = 0 rows pass z = None (no update).
    update_first: row 1's prior is P0 (no Q before the first row)."""
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


# ---------------------------------------------------------------- summaries


def rel(a, b):
    return float(np.nanmax(np.abs(a - b)) / max(1.0, float(np.nanmax(np.abs(b)))))


def gap_sd(lag_mean, lag_cov, full_mean, full_cov):
    """|lag - full| / sd_full per row and coefficient, summarized.

    Returns (max over all rows, max over rows >= BURN, RMS over rows >= BURN,
    the RMS the covariances predict over rows >= BURN). Given y_1..y_{s+L},
    the full smoother's mean differs from the lag-L mean by a zero-mean
    Gaussian of covariance P^L_s - P^full_s, so the predicted RMS in SD units
    is sqrt(mean((P^L_ii - P^full_ii) / P^full_ii)).
    """
    var = np.einsum("nii->ni", full_cov)
    z = np.abs(lag_mean - full_mean) / np.sqrt(var)
    excess = (np.einsum("nii->ni", lag_cov) - var) / var
    return (
        float(np.max(z)),
        float(np.max(z[BURN:])),
        float(np.sqrt(np.mean(z[BURN:] ** 2))),
        float(np.sqrt(np.mean(excess[BURN:]))),
    )


def sizes(k, L):
    """Floats in state (f64: 8 bytes each), the filter's own (b, P) included."""
    tri = k * (k + 1) // 2
    m = k * (L + 1)
    return {
        # the filter's (b, P sym) plus L slots of (b, P sym)
        "ring": (k + tri) + L * (k + tri),
        # and G_s (k x k, not symmetric) in each slot
        "ring, G cached": (k + tri) + L * (k + tri + k * k),
        # mean k(L+1) and the symmetric covariance of the whole stacked state
        "augmented": m + m * (m + 1) // 2,
        # a_0..a_L, P sym, C_1..C_L full, D_1..D_L sym
        "reduced": m + tri + L * (k * k + tri),
    }


def tolerance(max_gap: float) -> float:
    """1.5 times the largest gap over seeds, rounded up to a multiple of 0.05."""
    return round(0.05 * math.ceil(round(1.5 * max_gap / 0.05, 9)), 2)


# ---------------------------------------------------------------- sections


class Cfg:
    def __init__(self, quick: bool, seeds: int, n_seeds_n: int, workers: int):
        self.quick = quick
        self.seeds = [0] if quick else [0, 1, 2, 3, 4]
        # prefix rows at which the emitted value is compared with a library smoother
        self.rows = (lambda L: [L, 1999]) if quick else (lambda L: [L, 500, 1234, 1999])
        self.tol_seeds = seeds
        self.tol_n = n_seeds_n
        self.workers = workers


ORACLE: dict[tuple[int, int], tuple[float, float]] = {}  # (k, L) -> (statsmodels, filterpy)


def section_full(cfg: Cfg) -> None:
    """Part 1: own numpy RTS against both libraries on all rows (k = 2)."""
    k, n = 2, 2000
    P0 = np.eye(k) * 10.0
    print(f"\n## Full RTS smoother: own numpy vs statsmodels vs filterpy (k={k}, N={n}, seed 0)")
    for q in [1e-2, 1e-3, 1e-5]:
        Q = np.eye(k) * q
        x, y = simulate(n, k, q, R, 0)
        w = np.ones(n)
        bf, Pf, _ = filter_gen(x, y, w, Q, R, P0)
        bs, Ps = full_rts_gen(bf, Pf, Q)
        sm_m, sm_c, sm_f, _ = sm_smoother(x, y, w, Q, R, P0)
        d = [rel(bf, sm_f), rel(bs, sm_m), rel(Ps, sm_c)]
        line = f"q={q:g}: filtered vs sm {d[0]:.1e}; smoothed mean vs sm {d[1]:.1e}, cov vs sm {d[2]:.1e}"
        if HAVE_FILTERPY:
            fp_m, fp_c, _, _ = fp_smoother(x, y, w, Q, R, P0)
            d += [rel(bs, fp_m), rel(Ps, fp_c)]
            line += f"; mean vs filterpy {d[3]:.1e}, cov vs filterpy {d[4]:.1e}"
        print(line)
        check(f"full RTS vs libraries q={q:g}", max(d) < 1e-12, f"{max(d):.1e}")


def section_exact(cfg: Cfg) -> None:
    """Parts 1 and 2: the emitted value against a library smoother run on the prefix."""
    n, q = 2000, 1e-3
    print(f"filterpy available: {HAVE_FILTERPY}")
    print(
        f"\n## Exactness on the prefix: emitted value vs statsmodels / filterpy (q={q:g}, seed 0, N={n})"
    )
    print("| k | L | vs statsmodels | vs filterpy | rows t |")
    print("|---|---|---|---|---|")
    for k in [2, 5, 20]:
        P0 = np.eye(k) * 10.0
        Q = np.eye(k) * q
        x, y = simulate(n, k, q, R, 0)
        w = np.ones(n)
        bf, Pf, _ = filter_gen(x, y, w, Q, R, P0)
        for L in LAGS:
            if k == 2:  # part 1 checked the streaming ring itself
                vm, vc, _ = run(Ring(k, L, Q, R, P0), x, y)
            else:  # the batched arithmetic, checked against the stream in `forms`
                vm, vc = pad(n, L, *ring_vec_gen(bf, Pf, Q, L))
            wsm = wfp = 0.0
            for t in cfg.rows(L):
                s = t - L
                sm_m, sm_c, _, _ = sm_smoother(x[: t + 1], y[: t + 1], w[: t + 1], Q, R, P0)
                wsm = max(wsm, rel(vm[s], sm_m[s]), rel(vc[s], sm_c[s]))
                if HAVE_FILTERPY:
                    fm, fc, _, _ = fp_smoother(x[: t + 1], y[: t + 1], w[: t + 1], Q, R, P0)
                    wfp = max(wfp, rel(vm[s], fm[s]), rel(vc[s], fc[s]))
            ORACLE[(k, L)] = (wsm, wfp)
            print(f"| {k} | {L} | {wsm:.1e} | {wfp:.1e} | {cfg.rows(L)} |")
            check(f"exact on prefix k={k} L={L}", max(wsm, wfp) < 1e-11, f"{wsm:.1e} {wfp:.1e}")


def section_forms(cfg: Cfg) -> None:
    """Parts 1 and 2: the forms agree; the gap to the full smoother in posterior SD."""
    print(
        "\n## Forms agree; gap to the full smoother "
        f"({len(cfg.seeds)} seed(s); max over rows, RMS over rows >= {BURN})"
    )
    print(
        "| k | q | N | L | ring vs aug | ring vs reduced | ring stream vs vectorized (seed 0) "
        "| gap max, all rows | gap max, rows >= 500 | gap RMS, rows >= 500 | RMS predicted by the covariances |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    cases = []
    for q in [1e-2, 1e-3, 1e-5]:
        for n in [2000] if cfg.quick else [2000, 10000]:
            cases.append((2, q, n))
    for k in [5, 20]:
        for q in [1e-2, 1e-3]:
            cases.append((k, q, 2000))
    aug_cap = 250 if cfg.quick else 600
    for k, q, n in cases:
        P0 = np.eye(k) * 10.0
        Q = np.eye(k) * q
        for L in LAGS:
            t0 = time.perf_counter()
            run_aug = k * (L + 1) <= aug_cap
            # the streaming ring is checked at N = 2000 (where it is cheap enough);
            # --quick checks it where L <= 10 and at k = 2
            run_stream = n == 2000 and (not cfg.quick or L <= 10 or k == 2)
            gaps, d_aug, d_red, d_vec = [], 0.0, 0.0, None
            for seed in cfg.seeds:
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
                if run_stream and seed == 0:
                    rm, rc, _ = run(Ring(k, L, Q, R, P0), x, y)
                    d_vec = max(rel(lm, rm), rel(lc, rc))
                gaps.append(gap_sd(lm[: n - L], lc[: n - L], bs[: n - L], Ps[: n - L]))
            g = np.array(gaps)
            aug = f"{d_aug:.1e}" if run_aug else "not run"
            vec = f"{d_vec:.1e}" if d_vec is not None else "n/a"
            print(
                f"| {k} | {q:g} | {n} | {L} | {aug} | {d_red:.1e} | {vec} | {g[:, 0].max():.3g} | "
                f"{g[:, 1].max():.3g} | {g[:, 2].max():.3g} | {g[:, 3].max():.3g} |"
                f"  <!-- {time.perf_counter() - t0:.1f} s -->"
            )
            check(f"forms agree k={k} q={q:g} N={n} L={L}", max(d_aug, d_red) < 1e-12)
            if d_vec is not None:
                check(
                    f"ring stream vs vectorized k={k} q={q:g} L={L}", d_vec < 1e-13, f"{d_vec:.1e}"
                )
            ratio = g[:, 2].max() / g[:, 3].max()
            check(
                f"covariances predict the gap RMS k={k} q={q:g} N={n} L={L}",
                0.7 < ratio < 1.4,
                f"ratio {ratio:.2f}",
            )


def section_contraction(cfg: Cfg) -> None:
    """Part 1: ||G_s ... G_{s+L-1}|| -- the gap is that product times the
    full-vs-filtered difference at s+L."""
    k, n = 2, 2000
    P0 = np.eye(k) * 10.0
    print(
        "\n## Contraction of the smoother gains (median over rows of ||G_s..G_{s+L-1}||_2, k=2, seed 0, N=2000)"
    )
    for q in [1e-2, 1e-3, 1e-5]:
        Q = np.eye(k) * q
        x, y = simulate(n, k, q, R, 0)
        bf, Pf, _ = filter_gen(x, y, np.ones(n), Q, R, P0)
        G, _ = gains_gen(Pf, Q)
        res, med = [], []
        for L in LAGS:
            norms = []
            for s in range(200, n - L, 7):
                M = np.eye(k)
                for j in range(L):
                    M = M @ G[s + j]
                norms.append(np.linalg.norm(M, 2))
            med.append(float(np.median(norms)))
            res.append(f"L={L}: {med[-1]:.3g}")
        rho = np.median([np.max(np.abs(np.linalg.eigvals(G[s]))) for s in range(200, n)])
        print(
            f"q={q:g}: "
            + ", ".join(res)
            + f"; median spectral radius of one G {rho:.4f} (1/(1-rho) = {1 / (1 - rho):.0f} rows)"
        )
        check(f"gain products shrink with L q={q:g}", med[0] > med[1] > med[2] and rho < 1)


def section_oos(cfg: Cfg) -> None:
    """Part 1: perturb y after row t0; emissions at rows <= t0 and preds <= t0 unchanged."""
    k, q, t0 = 2, 1e-3, 1000
    n = 1200 if cfg.quick else 2000
    P0 = np.eye(k) * 10.0
    Q = np.eye(k) * q
    print(f"\n## Out-of-sample: perturb y after row t0 = {t0} (k=2, N={n}, q=1e-3)")
    x, y = simulate(n, k, q, R, 0)
    y2 = y.copy()
    y2[t0 + 1 :] += np.random.default_rng(99).standard_normal(n - t0 - 1) * 5.0
    y3 = y.copy()
    y3[t0] += 5.0
    for name, cls in [("ring", Ring), ("augmented", Augmented), ("reduced", Reduced)]:
        for L in LAGS:
            m1, c1, p1 = run(cls(k, L, Q, R, P0), x, y)
            m2, c2, p2 = run(cls(k, L, Q, R, P0), x, y2)
            m3, _, p3 = run(cls(k, L, Q, R, P0), x, y3)
            # emitted on rows <= t0 are the estimates for rows <= t0 - L
            e = t0 - L + 1
            same = (
                np.array_equal(m1[:e], m2[:e])
                and np.array_equal(c1[:e], c2[:e])
                and np.array_equal(p1[: t0 + 1], p2[: t0 + 1])
            )
            moved_after = not np.array_equal(m1[e:], m2[e:], equal_nan=True)
            own_moves = not np.array_equal(m1[e - 1], m3[e - 1])  # emitted at row t0 sees y_t0
            pred_t0_same = p1[t0] == p3[t0]
            print(
                f"{name} L={L}: emitted rows <= t0 bit-identical under future perturbation: {same}; "
                f"later emissions move: {moved_after}; emission at t0 moves with y_t0: {own_moves}; "
                f"pred at t0 unaffected by y_t0: {pred_t0_same}"
            )
            check(
                f"out of sample {name} L={L}",
                same and moved_after and own_moves and pred_t0_same,
            )


def section_forget(cfg: Cfg) -> None:
    """Part 2: exponential forgetting, the head block inflated alone."""
    n, k = 2000, 5
    P0 = np.eye(k) * 10.0
    lams = [0.99] if cfg.quick else [0.999, 0.99]
    print(f"\n## Forgetting: P_{{t|t-1}} = P_{{t-1|t-1}} / lam + Q (k={k}, N={n})")
    print(
        "| q | lam | L | ring vs hand full RTS on prefix | ring vs statsmodels on prefix | ring vs filterpy on prefix "
        f"| ring vs reduced | ring vs aug | reduced, all blocks / lam, vs ring | gap RMS, rows >= 500 ({len(cfg.seeds)} seed(s)) | RMS predicted by the covariances |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for q in [1e-3, 0.0]:
        Q = np.eye(k) * q
        for lam in lams:
            for L in LAGS:
                t0 = time.perf_counter()
                gaps = []
                for seed in cfg.seeds:
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
                    for t in cfg.rows(L):
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
                        d_aug_v = max(rel(rm, am), rel(rc, ac))
                        d_aug = f"{d_aug_v:.1e}"
                    else:
                        d_aug_v, d_aug = 0.0, "n/a"
                    fm2, _, _ = run(Reduced(k, L, Q, R, P0, lam, forget_all=True), x, y)
                    d_all = rel(rm, fm2)
                g = np.array(gaps)
                print(
                    f"| {q:g} | {lam} | {L} | {d_hand:.1e} | {d_sm:.1e} | {d_fp:.1e} | {d_red:.1e} | {d_aug} | "
                    f"{d_all:.2g} | {g[:, 2].max():.3g} | {g[:, 3].max():.3g} |  <!-- {time.perf_counter() - t0:.1f} s -->"
                )
                check(
                    f"forgetting exact q={q:g} lam={lam} L={L}",
                    d_hand < 1e-13 and max(d_sm, d_fp) < 1e-11 and max(d_red, d_aug_v) < 1e-12,
                    f"{d_hand:.1e} {d_sm:.1e} {d_fp:.1e} {d_red:.1e}",
                )
                check(f"all blocks / lam differs q={q:g} lam={lam} L={L}", d_all > 1e-6)
    # q = 0: G = lam I exactly
    Q0 = np.zeros((k, k))
    x, y = simulate(n, k, 1e-3, R, 0)
    bf, Pf, _ = filter_gen(x, y, np.ones(n), Q0, R, P0, 0.99)
    G, _ = gains_gen(Pf, Q0, 0.99)
    dg = float(np.max(np.abs(G - 0.99 * np.eye(k))))
    print(f"\nq=0, lam=0.99: max |G_s - lam I| = {dg:.1e}")
    check("q=0: G = lam I", dg < 1e-12, f"{dg:.1e}")
    # the filtered covariances of the three filters under forgetting
    w = np.ones(n)
    Q = np.eye(k) * 1e-3
    bf, Pf, _ = filter_gen(x, y, w, Q, R, P0, 0.99)
    sc = np.empty((k, k, n))
    sc[:] = ((1.0 / 0.99 - 1.0) * Pf + Q).transpose(1, 2, 0)
    _, _, smf, smP = sm_smoother(x, y, w, Q, R, P0, sc)
    d = [rel(bf, smf), rel(Pf, smP)]
    line = f"q=1e-3, lam=0.99 filtered: own vs statsmodels mean {d[0]:.1e} cov {d[1]:.1e}"
    if HAVE_FILTERPY:
        _, _, fpm, fpP = fp_smoother(x, y, w, Q, R, P0, 0.99)
        d += [rel(bf, fpm), rel(Pf, fpP)]
        line += f"; own vs filterpy (alpha) mean {d[2]:.1e} cov {d[3]:.1e}"
    print(line)
    check("forgetting filter vs libraries", max(d) < 1e-12, f"{max(d):.1e}")


def section_zero(cfg: Cfg) -> None:
    """Part 2: zero-weight rows, the first row among them, targets NaN."""
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
    print(f"\n## Zero-weight rows (k=5, q=1e-3, N=2000): {int(zero.sum())} rows, row 0 among them")

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
    d_f = max(rel(bf[keep], bfd), rel(Pf[keep], Pfd))
    print(f"filter: kept rows vs the dropped stream with the clock carried: {d_f:.1e}")
    check("zero rows: filter vs dropped stream", d_f < 1e-13, f"{d_f:.1e}")

    rows = (lambda L: [L, 705, 1999]) if cfg.quick else (lambda L: [L, L + 1, 705, 720, 1234, 1999])
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
        d_aug_v = (
            max(rel(rm, res["augmented"][0]), rel(rc, res["augmented"][1]))
            if "augmented" in res
            else 0.0
        )
        d_aug = f"{d_aug_v:.1e}" if "augmented" in res else "n/a"
        vm, vc = pad(n, L, *ring_vec_gen(bf, Pf, Q, L))
        d_vec = max(rel(rm, vm), rel(rc, vc))
        # oracle: statsmodels (missing y) and filterpy (z = None) on the prefix
        d_sm = d_fp = 0.0
        for t in rows(L):
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
        check(f"zero rows no NaN L={L}", not nan_any)
        check(f"zero rows target ignored L={L}", same_y)
        check(
            f"zero rows exact L={L}",
            max(d_red, d_aug_v, d_vec, d_drop, interp) < 1e-12 and max(d_sm, d_fp) < 1e-11,
            f"{d_red:.1e} {d_vec:.1e} {d_sm:.1e} {d_fp:.1e} {d_drop:.1e} {interp:.1e}",
        )
    # the first row: emitted value at t = L for row 0
    L = 10
    rm, rc, _ = run(Ring(k, L, Q, R, P0), x, y_nan, w)
    hb, hP = full_rts_gen(bf[: L + 1], Pf[: L + 1], Q)
    G0 = np.linalg.solve(P0 + Q, P0).T
    step = G0 @ hb[1]  # b_{0|0} = 0
    d0, d1 = rel(rm[0], step), rel(rm[0], hb[0])
    print(
        f"row 0 zero-weight, L=10: emitted = G_0 b_(1|10) with b_(0|0) = 0, G_0 = P0 (P0 + Q)^-1: "
        f"{d0:.1e}; = full RTS on y_1..y_10 at row 0: {d1:.1e}"
    )
    check("zero first row", max(d0, d1) < 1e-14, f"{d0:.1e} {d1:.1e}")


def section_cost(cfg: Cfg) -> None:
    """Parts 1 and 2: per-row cost (one interpreted step per row) and state size."""
    print(
        "\n## Per-row cost (numpy, one Python step per row, timed after the first L rows) and state size"
    )
    print("| k | L | form | us/row | rows timed | floats in state | bytes |")
    print("|---|---|---|---|---|---|---|")
    q = 1e-3
    aug_cap = 250 if cfg.quick else 600
    for k in [2, 5, 20]:
        P0 = np.eye(k) * 10.0
        Q = np.eye(k) * q
        x, y = simulate(2200, k, q, R, 0)
        for L in LAGS:
            sz = sizes(k, L)
            for name in ["ring", "ring, G cached", "augmented", "reduced"]:
                if name == "augmented" and k * (L + 1) > aug_cap:
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
                slow = (name.startswith("ring") or name == "augmented") and L == 100
                nt = (50 if slow else 200) if cfg.quick else (300 if slow else 1000)
                t1 = time.perf_counter()
                for t in range(L, L + nt):
                    sm.step(x[t], y[t])
                us = (time.perf_counter() - t1) / nt * 1e6
                print(f"| {k} | {L} | {name} | {us:.0f} | {nt} | {sz[name]} | {8 * sz[name]:,} |")
    # the formulas, in closed form, against the counts above
    for k in [2, 5, 20]:
        for L in LAGS:
            tri = k * (k + 1) // 2
            sz = sizes(k, L)
            check(
                f"state formula k={k} L={L}",
                sz["reduced"] == sz["ring, G cached"] == (k + tri) + L * (k + tri + k * k)
                and sz["ring"] == (L + 1) * (k + tri),
            )


# ---------------------------------------------------------------- the tolerance table


TOL_KS = [5, 20]
TOL_QS = [1e-2, 1e-3]


def _seed_gaps(args):
    """One seed: per (k, q, L) the gap summary at N rows (max all, max >= BURN, RMS, predicted)."""
    seed, n = args
    out = {}
    for k in TOL_KS:
        P0 = np.eye(k) * 10.0
        for q in TOL_QS:
            Q = np.eye(k) * q
            x, y = simulate(n, k, q, R, seed)
            bf, Pf, _ = filter_gen(x, y, np.ones(n), Q, R, P0)
            bs, Ps = full_rts_gen(bf, Pf, Q)
            for L in LAGS:
                lm, lc = ring_vec_gen(bf, Pf, Q, L)
                out[(k, q, L)] = gap_sd(lm, lc, bs[: n - L], Ps[: n - L])
    return seed, out


def section_seeds(cfg: Cfg) -> None:
    """Part 3: the gap at rows >= BURN over many seeds, and the proposed tolerance."""
    n, ns = (2000, 1) if cfg.quick else (cfg.tol_n, cfg.tol_seeds)
    print(
        f"\n## Gap to the full smoother over {ns} seed(s), N={n}, rows >= {BURN} "
        "(posterior SD of the full smoother); tolerance = 1.5 x max over seeds, rounded up to 0.05"
    )
    t0 = time.perf_counter()
    jobs = [(s, n) for s in range(ns)]
    results = {}
    if cfg.workers > 1 and ns > 1:
        with ProcessPoolExecutor(max_workers=cfg.workers) as ex:
            for seed, out in ex.map(_seed_gaps, jobs):
                results[seed] = out
                print(f"  seed {seed} done ({time.perf_counter() - t0:.0f} s)", flush=True)
    else:
        for job in jobs:
            seed, out = _seed_gaps(job)
            results[seed] = out
            print(f"  seed {seed} done ({time.perf_counter() - t0:.0f} s)", flush=True)
    print(
        "| k | q | L | gap max, rows >= 500: max over seeds (seed) | median over seeds "
        "| gap RMS, rows >= 500, worst seed | RMS predicted, worst seed | gap max, all rows | proposed tolerance |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    per_kl: dict[tuple[int, int], float] = {}
    for k in TOL_KS:
        for q in TOL_QS:
            for L in LAGS:
                g = np.array([results[s][(k, q, L)] for s in range(ns)])
                mx = float(g[:, 1].max())
                tol = tolerance(mx)
                per_kl[(k, L)] = max(per_kl.get((k, L), 0.0), tol)
                print(
                    f"| {k} | {q:g} | {L} | {mx:.3g} ({int(g[:, 1].argmax())}) | {np.median(g[:, 1]):.3g} | "
                    f"{g[:, 2].max():.3g} | {g[:, 3].max():.3g} | {g[:, 0].max():.3g} | {tol:.2f} |"
                )
                check(
                    f"tolerance covers every seed k={k} q={q:g} L={L}", bool((g[:, 1] <= tol).all())
                )
    print("\nper k and L (the larger over q):")
    for (k, L), tol in sorted(per_kl.items()):
        print(f"  k={k} L={L}: {tol:.2f}")
    print(f"({time.perf_counter() - t0:.0f} s)")


SECTIONS = {
    "full": section_full,
    "exact": section_exact,
    "forms": section_forms,
    "contraction": section_contraction,
    "oos": section_oos,
    "forget": section_forget,
    "zero": section_zero,
    "cost": section_cost,
    "seeds": section_seeds,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument(
        "--quick", action="store_true", help="every check, short (one seed, fewer rows)"
    )
    ap.add_argument(
        "--section",
        action="append",
        choices=list(SECTIONS),
        help="run only these sections (repeatable); default: all",
    )
    ap.add_argument(
        "--seeds", type=int, default=5, help="seeds for the tolerance table (default 5)"
    )
    ap.add_argument(
        "--n", type=int, default=10000, help="rows for the tolerance table (default 10000)"
    )
    ap.add_argument("--workers", type=int, default=1, help="processes for the tolerance table")
    args = ap.parse_args()
    cfg = Cfg(args.quick, args.seeds, args.n, args.workers)
    print(f"filterpy available: {HAVE_FILTERPY}; quick: {args.quick}")
    t_all = time.perf_counter()
    for name in args.section or list(SECTIONS):
        t0 = time.perf_counter()
        SECTIONS[name](cfg)
        print(f"[section {name}: {time.perf_counter() - t0:.1f} s]", flush=True)
    if ORACLE:
        print("\n## Largest relative difference to each oracle on the prefix, per k and L")
        for (k, L), (a, b) in sorted(ORACLE.items()):
            print(f"  k={k} L={L}: statsmodels {a:.1e}, filterpy {b:.1e}")
    failed = [c for c in CHECKS if not c[1]]
    print(
        f"\n{len(CHECKS) - len(failed)} of {len(CHECKS)} checks passed "
        f"in {time.perf_counter() - t_all:.0f} s"
    )
    for name, _, detail in failed:
        print(f"FAILED: {name} {detail}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
