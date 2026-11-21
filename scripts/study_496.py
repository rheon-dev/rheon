# ruff: noqa: E501  (long report strings)
"""Task 496 study, part 1: a fixed-lag smoother beside `kalman`.

Random-walk-coefficient regression

    beta_t = beta_{t-1} + eta_t,   eta_t ~ N(0, Q)
    y_t    = x_t' beta_t + eps_t,  eps_t ~ N(0, R)
    beta_1 ~ N(0, P0)              (the prior at the first row; no Q before it)

At row t the lag-L smoother emits E[beta_{t-L} | y_1..y_t] (and its covariance).
Two exact forms are built and checked against each other, against a full
Rauch-Tung-Striebel (RTS) smoother on the prefix y_1..y_t (statsmodels and
filterpy), and the gap to the full smoother on all N rows is measured:

(a) ring:      the last L filtered means and covariances, and a backward RTS
               pass of length L at each row;
(b) augmented: the state [beta_t, ..., beta_{t-L}] with a shift transition,
               filtered as an ordinary Kalman filter (full covariance);
(b') reduced augmented: only the blocks (b) actually reads -- the first block
               column cov(beta_{t-j}, beta_t) and the diagonal blocks.

Python and numpy only; data are generated from seeded generators. Run:

    uv run python scripts/study_496.py            # everything (about 30 s)
    uv run python scripts/study_496.py --quick    # one seed, N = 2000
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from statsmodels.tsa.statespace.kalman_smoother import KalmanSmoother

try:
    from filterpy.kalman import KalmanFilter

    HAVE_FILTERPY = True
except ImportError:  # pragma: no cover - reported in the output
    HAVE_FILTERPY = False


# ---------------------------------------------------------------- data


def simulate(n: int, k: int, q: float, r: float, seed: int):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, k))
    beta = np.cumsum(rng.standard_normal((n, k)) * np.sqrt(q), axis=0) + rng.standard_normal(k)
    y = np.einsum("tk,tk->t", x, beta) + rng.standard_normal(n) * np.sqrt(r)
    return x, y


# ---------------------------------------------------------------- filter + full RTS


def kalman_filter(x, y, Q, R, P0):
    """Filtered means/covs b_{t|t}, P_{t|t} and the out-of-sample predictions."""
    n, k = x.shape
    b = np.zeros(k)
    P = P0.copy()
    bf = np.empty((n, k))
    Pf = np.empty((n, k, k))
    pred = np.empty(n)
    for t in range(n):
        if t > 0:
            P = P + Q  # predict (random walk: mean unchanged)
        z = x[t]
        pred[t] = z @ b  # before the update with y_t
        Pz = P @ z
        s = z @ Pz + R
        g = Pz / s
        b = b + g * (y[t] - pred[t])
        P = P - np.outer(g, Pz)
        P = 0.5 * (P + P.T)
        bf[t] = b
        Pf[t] = P
    return bf, Pf, pred


def rts_gains(Pf, Q):
    """G_s = P_{s|s} (P_{s|s} + Q)^{-1}, fixed once row s is filtered."""
    Pp = Pf + Q
    # G = Pf Pp^{-1}  <=>  G' = Pp^{-1} Pf (both symmetric)
    return np.linalg.solve(Pp, Pf).transpose(0, 2, 1)


def full_rts(bf, Pf, Q):
    n = bf.shape[0]
    G = rts_gains(Pf, Q)
    bs = bf.copy()
    Ps = Pf.copy()
    for s in range(n - 2, -1, -1):
        bs[s] = bf[s] + G[s] @ (bs[s + 1] - bf[s])
        Ps[s] = Pf[s] + G[s] @ (Ps[s + 1] - (Pf[s] + Q)) @ G[s].T
    return bs, Ps


# ---------------------------------------------------------------- (a) ring, streaming


class RingSmoother:
    """Streaming lag-L smoother: a ring of the last L filtered (b, P).

    Per row: the filter's step, then a backward RTS pass over the ring. The
    ring holds b_{s|s} and P_{s|s} for s = t-L..t-1; row t's own filtered
    state is the filter's. Emits (row t-L, mean, cov) once t >= L.
    """

    def __init__(self, k, L, Q, R, P0):
        self.k, self.L, self.Q, self.R = k, L, Q, R
        self.b = np.zeros(k)
        self.P = P0.copy()
        self.rb = np.zeros((L, k))
        self.rP = np.zeros((L, k, k))
        self.t = 0  # rows seen

    def step(self, z, y):
        Q, R, L = self.Q, self.R, self.L
        if self.t > 0:
            self.P = self.P + Q
        pred = z @ self.b
        Pz = self.P @ z
        s = z @ Pz + R
        g = Pz / s
        self.b = self.b + g * (y - pred)
        P = self.P - np.outer(g, Pz)
        self.P = 0.5 * (P + P.T)
        t = self.t
        out = None
        if t >= L:
            xs, Ps = self.b, self.P
            for j in range(1, L + 1):  # s = t-j, newest first
                i = (t - j) % L
                bs, Pss = self.rb[i], self.rP[i]
                Pp = Pss + Q
                G = np.linalg.solve(Pp, Pss).T
                xs = bs + G @ (xs - bs)
                Ps = Pss + G @ (Ps - Pp) @ G.T
            out = (t - L, xs, Ps)
        if L > 0:
            i = t % L
            self.rb[i] = self.b
            self.rP[i] = self.P
        self.t += 1
        return pred, out


def ring_vectorized(bf, Pf, Q, L, cov=True):
    """Form (a) for every emission row at once (the same arithmetic per row)."""
    n = bf.shape[0]
    G = rts_gains(Pf, Q)
    t = np.arange(L, n)
    xs = bf[t].copy()
    Ps = Pf[t].copy() if cov else None
    for j in range(1, L + 1):
        s = t - j
        xs = bf[s] + np.einsum("nij,nj->ni", G[s], xs - bf[s])
        if cov:
            Ps = Pf[s] + G[s] @ (Ps - (Pf[s] + Q)) @ G[s].transpose(0, 2, 1)
    return xs, Ps  # row i of each is the estimate for row i (= t - L)


# ---------------------------------------------------------------- (b) augmented, streaming


class AugmentedSmoother:
    """State [beta_t, beta_{t-1}, ..., beta_{t-L}], size m = k(L+1); full m x m covariance."""

    def __init__(self, k, L, Q, R, P0):
        self.k, self.L, self.Q, self.R = k, L, Q, R
        m = k * (L + 1)
        self.m = m
        self.a = np.zeros(m)
        self.P = np.zeros((m, m))
        # The lagged slots start at zero with zero covariance: their gain is 0
        # until a real state shifts in, and none is read before row L.
        self.P[:k, :k] = P0
        self.t = 0

    def step(self, z, y):
        k, Q, R = self.k, self.Q, self.R
        if self.t > 0:
            a, P = self.a, self.P
            na = np.empty_like(a)
            na[:k] = a[:k]
            na[k:] = a[:-k]
            nP = np.empty_like(P)
            nP[k:, k:] = P[:-k, :-k]
            nP[:k, k:] = P[:k, :-k]
            nP[k:, :k] = P[:-k, :k]
            nP[:k, :k] = P[:k, :k] + Q
            self.a, self.P = na, nP
        pred = z @ self.a[:k]
        Ph = self.P[:, :k] @ z  # P H', H = [z', 0, ..., 0]
        s = z @ Ph[:k] + R
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


class ReducedAugmentedSmoother:
    """(b'): only the blocks of (b) that are read.

    C_j = cov(beta_{t-j}, beta_t | y_1..t)  (k x k, j = 1..L)
    D_j = cov(beta_{t-j} | y_1..t)          (k x k symmetric, j = 1..L)
    a_j = E[beta_{t-j} | y_1..t]            (k, j = 1..L)
    plus the filter's own b = a_0 and P = D_0 = C_0.
    """

    def __init__(self, k, L, Q, R, P0):
        self.k, self.L, self.Q, self.R = k, L, Q, R
        self.a = np.zeros((L + 1, k))
        self.C = np.zeros((L + 1, k, k))  # C[0] is P
        self.D = np.zeros((L + 1, k, k))  # D[0] is P
        self.C[0] = P0
        self.D[0] = P0
        self.t = 0

    def step(self, z, y):
        Q, R = self.Q, self.R
        if self.t > 0:
            # shift: beta_{t-j} <- beta_{(t-1)-(j-1)}; cov(beta_{t-j}, beta_t) =
            # cov(beta_{t-j}, beta_{t-1}) because eta_t is independent of the past.
            self.a[1:] = self.a[:-1].copy()
            self.C[1:] = self.C[:-1].copy()
            self.D[1:] = self.D[:-1].copy()
            self.C[0] = self.C[0] + Q
            self.D[0] = self.C[0]
        a, C, D = self.a, self.C, self.D
        pred = z @ a[0]
        Ph = C @ z  # (L+1, k): cov(beta_{t-j}, beta_t) z
        s = z @ Ph[0] + R
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


def run_stream(cls, x, y, Q, R, P0, L):
    n, k = x.shape
    sm = cls(k, L, Q, R, P0)
    means = np.full((n, k), np.nan)
    covs = np.full((n, k, k), np.nan)
    preds = np.empty(n)
    for t in range(n):
        preds[t], out = sm.step(x[t], y[t])
        if out is not None:
            s, m, c = out
            means[s], covs[s] = m, c
    return means, covs, preds


# ---------------------------------------------------------------- third-party smoothers


def statsmodels_smoother(x, y, Q, R, P0):
    n, k = x.shape
    ks = KalmanSmoother(k_endog=1, k_states=k, k_posdef=k)
    ks.bind(y.reshape(n, 1).copy())
    ks["design"] = x.T.reshape(1, k, n).copy()
    ks["obs_cov"] = np.array([[R]])
    ks["transition"] = np.eye(k)
    ks["selection"] = np.eye(k)
    ks["state_cov"] = Q
    ks.initialize_known(np.zeros(k), P0)
    res = ks.smooth()
    return res.smoothed_state.T, res.smoothed_state_cov.transpose(2, 0, 1), res.filtered_state.T


def filterpy_smoother(x, y, Q, R, P0):
    n, k = x.shape
    kf = KalmanFilter(dim_x=k, dim_z=1)
    kf.x = np.zeros((k, 1))
    kf.P = P0.copy()
    kf.F = np.eye(k)
    kf.Q = Q
    kf.R = np.array([[R]])
    Hs = [x[t].reshape(1, k) for t in range(n)]
    # update_first: row 1's prior is P0 (no Q before the first row), as above.
    mu, cov, _, _ = kf.batch_filter(y.reshape(n, 1, 1), Hs=Hs, update_first=True)
    xs, Ps, _, _ = kf.rts_smoother(mu, cov)
    return xs.reshape(n, k), Ps


# ---------------------------------------------------------------- the study


def rel(a, b):
    return float(np.nanmax(np.abs(a - b)) / max(1.0, float(np.nanmax(np.abs(b)))))


BURN = 500  # rows from the start excluded from the steady-state columns


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    k = 2
    R = 1.0
    P0 = np.eye(k) * 10.0
    lags = [1, 10, 100]
    Ns = [2000] if args.quick else [2000, 10000]
    seeds = [0] if args.quick else [0, 1, 2, 3, 4]
    qs = [1e-2, 1e-3, 1e-5]

    print(f"filterpy available: {HAVE_FILTERPY}")

    # 1. Full smoother: own RTS vs statsmodels vs filterpy (N = 2000, seed 0).
    print("\n## Full RTS smoother: own numpy vs statsmodels vs filterpy (k=2, N=2000, seed 0)")
    for q in qs:
        Q = np.eye(k) * q
        x, y = simulate(2000, k, q, R, 0)
        bf, Pf, _ = kalman_filter(x, y, Q, R, P0)
        bs, Ps = full_rts(bf, Pf, Q)
        sm_m, sm_c, sm_f = statsmodels_smoother(x, y, Q, R, P0)
        line = (
            f"q={q:g}: filtered vs sm {rel(bf, sm_f):.1e}; smoothed mean vs sm {rel(bs, sm_m):.1e}, "
            f"cov vs sm {rel(Ps, sm_c):.1e}"
        )
        if HAVE_FILTERPY:
            fp_m, fp_c = filterpy_smoother(x, y, Q, R, P0)
            line += f"; mean vs filterpy {rel(bs, fp_m):.1e}, cov vs filterpy {rel(Ps, fp_c):.1e}"
        print(line)

    # 2. Exactness: lag-L at row s == full smoother on the prefix y_1..y_{s+L}, at s.
    print(
        "\n## Exactness: lag-L emitted value vs a third-party smoother run on the prefix (q=1e-3, N=2000)"
    )
    q = 1e-3
    Q = np.eye(k) * q
    x, y = simulate(2000, k, q, R, 0)
    for L in lags:
        rm, rc, _ = run_stream(RingSmoother, x, y, Q, R, P0, L)
        worst_sm = worst_fp = 0.0
        for t in [L, 500, 1234, 1999]:
            s = t - L
            sm_m, sm_c, _ = statsmodels_smoother(x[: t + 1], y[: t + 1], Q, R, P0)
            worst_sm = max(worst_sm, rel(rm[s], sm_m[s]), rel(rc[s], sm_c[s]))
            if HAVE_FILTERPY:
                fp_m, fp_c = filterpy_smoother(x[: t + 1], y[: t + 1], Q, R, P0)
                worst_fp = max(worst_fp, rel(rm[s], fp_m[s]), rel(rc[s], fp_c[s]))
        print(
            f"L={L}: max rel diff (mean and cov, 4 rows) vs statsmodels {worst_sm:.1e}, vs filterpy {worst_fp:.1e}"
        )

    # 3. Forms agree; gap to the full smoother in posterior SD.
    print(
        "\n## Forms agree, and the gap to the full smoother (max over rows s in 1..N-L, both coefficients)"
    )
    print(
        "| q | N | L | ring vs aug | ring vs reduced | ring stream vs vectorized "
        "| gap max, all rows | gap max, rows >= 500 | gap RMS, rows >= 500 | RMS predicted by the covariances |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|")
    for q in qs:
        Q = np.eye(k) * q
        for n in Ns:
            for L in lags:
                gaps, d_aug, d_red, d_vec = [], 0.0, 0.0, 0.0
                for seed in seeds:
                    x, y = simulate(n, k, q, R, seed)
                    bf, Pf, _ = kalman_filter(x, y, Q, R, P0)
                    bs, Ps = full_rts(bf, Pf, Q)
                    vm, vc = ring_vectorized(bf, Pf, Q, L)
                    lag_m = np.full((n, k), np.nan)
                    lag_c = np.full((n, k, k), np.nan)
                    lag_m[: n - L], lag_c[: n - L] = vm, vc
                    am, ac, _ = run_stream(AugmentedSmoother, x, y, Q, R, P0, L)
                    em, ec, _ = run_stream(ReducedAugmentedSmoother, x, y, Q, R, P0, L)
                    d_aug = max(d_aug, rel(lag_m, am), rel(lag_c, ac))
                    d_red = max(d_red, rel(lag_m, em), rel(lag_c, ec))
                    if n == 2000:
                        rm, rc, _ = run_stream(RingSmoother, x, y, Q, R, P0, L)
                        d_vec = max(d_vec, rel(lag_m, rm), rel(lag_c, rc))
                    gaps.append(gap_sd(lag_m[: n - L], lag_c[: n - L], bs[: n - L], Ps[: n - L]))
                vec = f"{d_vec:.1e}" if n == 2000 else "n/a"
                g = np.array(gaps)  # (seeds, 4)
                print(
                    f"| {q:g} | {n} | {L} | {d_aug:.1e} | {d_red:.1e} | {vec} | "
                    f"{g[:, 0].max():.3g} | {g[:, 1].max():.3g} | {g[:, 2].max():.3g} | {g[:, 3].max():.3g} |"
                )

    # 3b. The rate: ||G_s ... G_{s+L-1}|| -- the gap is that product times the
    # full-vs-filtered difference at s+L.
    print(
        "\n## Contraction of the smoother gains (median over rows of ||G_s..G_{s+L-1}||_2, seed 0, N=2000)"
    )
    for q in qs:
        Q = np.eye(k) * q
        x, y = simulate(2000, k, q, R, 0)
        bf, Pf, _ = kalman_filter(x, y, Q, R, P0)
        G = rts_gains(Pf, Q)
        res = []
        for L in lags:
            norms = []
            for s in range(200, 2000 - L, 7):
                M = np.eye(k)
                for j in range(L):
                    M = M @ G[s + j]
                norms.append(np.linalg.norm(M, 2))
            res.append(f"L={L}: {np.median(norms):.3g}")
        rho = np.median([np.max(np.abs(np.linalg.eigvals(G[s]))) for s in range(200, 2000)])
        print(
            f"q={q:g}: "
            + ", ".join(res)
            + f"; median spectral radius of one G {rho:.4f} (1/(1-rho) = {1 / (1 - rho):.0f} rows)"
        )

    # 4. Out-of-sample: perturb y_{t+1..}; emissions at rows <= t and preds <= t unchanged.
    print("\n## Out-of-sample: perturb y after row t0 = 1000 (N=2000, q=1e-3)")
    q = 1e-3
    Q = np.eye(k) * q
    x, y = simulate(2000, k, q, R, 0)
    t0 = 1000
    y2 = y.copy()
    y2[t0 + 1 :] += np.random.default_rng(99).standard_normal(2000 - t0 - 1) * 5.0
    y3 = y.copy()
    y3[t0] += 5.0
    for name, cls in [
        ("ring", RingSmoother),
        ("augmented", AugmentedSmoother),
        ("reduced", ReducedAugmentedSmoother),
    ]:
        for L in lags:
            m1, c1, p1 = run_stream(cls, x, y, Q, R, P0, L)
            m2, c2, p2 = run_stream(cls, x, y2, Q, R, P0, L)
            m3, _, p3 = run_stream(cls, x, y3, Q, R, P0, L)
            # emitted on rows <= t0 are the estimates for rows <= t0 - L
            e = t0 - L + 1
            same = (
                np.array_equal(m1[:e], m2[:e])
                and np.array_equal(c1[:e], c2[:e])
                and np.array_equal(p1[: t0 + 1], p2[: t0 + 1])
            )
            moved_after = not np.array_equal(m1[e:], m2[e:])
            own_moves = not np.array_equal(m1[e - 1], m3[e - 1])  # emitted at row t0 sees y_t0
            pred_t0_same = p1[t0] == p3[t0]
            print(
                f"{name} L={L}: emitted rows <= t0 bit-identical under future perturbation: {same}; "
                f"later emissions move: {moved_after}; emission at t0 moves with y_t0: {own_moves}; "
                f"pred at t0 unaffected by y_t0: {pred_t0_same}"
            )

    # 5. Per-row cost and state size.
    print("\n## Per-row cost (k=2, N=2000, q=1e-3; numpy, one Python step per row) and state size")
    print("| form | L | us/row | floats in state | bytes |")
    print("|---|---|---|---|---|")
    x, y = simulate(2000, k, 1e-3, R, 0)
    Q = np.eye(k) * 1e-3
    tri = k * (k + 1) // 2
    for L in lags:
        m = k * (L + 1)
        sizes = {
            # the filter's own (b, P) plus L slots of (b, P sym)
            "ring": (k + tri) + L * (k + tri),
            # mean k(L+1) and the symmetric covariance of the whole stacked state
            "augmented": m + m * (m + 1) // 2,
            # a_0..a_L, C_1..C_L full, D_1..D_L sym, plus P sym
            "reduced": m + tri + L * (k * k + tri),
        }
        for name, cls in [
            ("ring", RingSmoother),
            ("augmented", AugmentedSmoother),
            ("reduced", ReducedAugmentedSmoother),
        ]:
            t1 = time.perf_counter()
            run_stream(cls, x, y, Q, R, P0, L)
            us = (time.perf_counter() - t1) / 2000 * 1e6
            print(f"| {name} | {L} | {us:.0f} | {sizes[name]} | {8 * sizes[name]} |")


if __name__ == "__main__":
    main()
