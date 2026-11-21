"""Task 524: regression variants (part 1: evidence-maximized ridge, averaged `sgd` and `pa`, `positive=True` lasso).

Each beside the recursion its model already keeps: the mean-form Gram (`EwMoments` over `(x, y)`), the per-row
`sgd` step `b -= lr (p - y) z w`, `pa`'s `tau`, the standardized coordinate descent of `lasso.rs`.

* BayesianRidge at solve time (MacKay 1992 / Tipping 2001), on the centred Gram `G = n C_xx`, `n` Kish's count with the
  weights scaled so `sum w' = n` (sklearn 1.9.1 uses `sum(sample_weight)` for `n`, the source says): iterate
      coef = (G + (lam/alpha) I)^-1 Xty,  sse = yy - 2 coef'Xty + coef' G coef,
      gamma = sum alpha e_i / (lam + alpha e_i),  lam = (gamma + 2 l1)/(|coef|^2 + 2 l2),  alpha = (n - gamma + 2 a1)/(sse + 2 a2)
  to the same stopping rule, `pred_std = sqrt(1/alpha + (x - m)' Sigma (x - m))`, `Sigma = (alpha G + lam I)^-1`.
* averaged sgd / pa: the EW mean of the iterates, `bbar' = bbar + (w/W') (b' - bbar)`.
* `positive=True`: the coordinate step clipped at 0.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
from sklearn.linear_model import BayesianRidge, ElasticNet, SGDRegressor

from .common import EwMoments, dt_stream, lam_of, table


# ------------------------------------------------------------------ evidence-maximized ridge
def bayes_ridge_from_moments(ew: EwMoments, k, a1=1e-6, a2=1e-6, l1=1e-6, l2=1e-6, tol=1e-3, max_iter=300, init=None):
    """Evidence maximization from the accumulator over (x_1..x_k, y). Returns coef, intercept, alpha, lam, Sigma, iters, m."""
    W = ew.W
    n = W * W / ew.Q  # Kish
    C = ew.C
    Cxx, cxy, cyy = C[:k, :k], C[:k, k], C[k, k]
    G = n * Cxx
    Xty = n * cxy
    yy = n * cyy
    e, V = np.linalg.eigh(G)
    e = np.maximum(e, 0.0)
    alpha = 1.0 / (cyy + np.finfo(float).eps) if init is None else init[0]
    lam = 1.0 if init is None else init[1]

    def coef_of(alpha, lam):
        return V @ ((V.T @ Xty) / (e + lam / alpha))

    old = None
    for it in range(max_iter):
        coef = coef_of(alpha, lam)
        sse = yy - 2 * coef @ Xty + coef @ G @ coef
        gamma = float(np.sum(alpha * e / (lam + alpha * e)))
        lam = (gamma + 2 * l1) / (coef @ coef + 2 * l2)
        alpha = (n - gamma + 2 * a1) / (sse + 2 * a2)
        if it != 0 and np.sum(np.abs(old - coef)) < tol:
            break
        old = coef.copy()
    coef = coef_of(alpha, lam)
    Sigma = V @ np.diag(1.0 / (alpha * e + lam)) @ V.T
    m = ew.m
    return coef, m[k] - m[:k] @ coef, alpha, lam, Sigma, it + 1, m[:k]


def stream_xy(sd, n, k, offset=0.0):
    rng = np.random.default_rng(sd)
    a = rng.standard_normal((k, k)) * 0.4 + np.eye(k)
    x = rng.standard_normal((n, k)) @ a + offset
    beta = rng.standard_normal(k)
    y = 0.4 + x @ beta + 1.3 * rng.standard_normal(n)
    return x, y


def run_moments(x, y, dt, w, hl):
    n, k = x.shape
    ew = EwMoments(k + 1)
    for t in range(n):
        ew.update(np.concatenate([x[t], [y[t]]]), 1.0 if (t == 0 or math.isinf(hl)) else lam_of(dt[t], hl), w[t])
    return ew


def omega_weights(dt, w, hl):
    n = len(w)
    lam = [1.0 if (t == 0 or math.isinf(hl)) else lam_of(dt[t], hl) for t in range(n)]
    om = np.empty(n)
    acc = 1.0
    for t in range(n - 1, -1, -1):
        om[t] = w[t] * acc
        acc *= lam[t]
    return om


def report_ridge(seeds):
    rows = []
    for label, hl, weighted, zeros, priors in (
        ("lam=1, unit w, default priors 1e-6", math.inf, False, False, 1e-6),
        ("lam=1, unit w, priors 0 (pure evidence)", math.inf, False, False, 0.0),
        ("lam=1, w in [.5,2], priors 1e-6", math.inf, True, False, 1e-6),
        ("half-life 50, w, 10% zero w, zero first row, priors 1e-6", 50.0, True, True, 1e-6),
    ):
        g = {"coef": 0.0, "icpt": 0.0, "alpha": 0.0, "lam": 0.0, "std": 0.0}
        for sd in range(seeds):
            k, n = 6, 160
            x, y = stream_xy(sd, n, k)
            dt, w = dt_stream(sd, n, zero_first=zeros, zero_frac=0.1 if zeros else 0.0)
            if not weighted:
                w = np.ones(n)
            ew = run_moments(x, y, dt, w, hl)
            coef, icpt, alpha, lam, Sigma, it, m = bayes_ridge_from_moments(ew, k, priors, priors, priors, priors, tol=1e-14, max_iter=5000)
            om = omega_weights(dt, w, hl)
            nK = om.sum() ** 2 / (om**2).sum()
            sw = om * (nK / om.sum())  # sum = Kish n
            ref = BayesianRidge(alpha_1=priors, alpha_2=priors, lambda_1=priors, lambda_2=priors, tol=1e-14, max_iter=5000).fit(x, y, sample_weight=sw)
            g["coef"] = max(g["coef"], float(np.max(np.abs(coef - ref.coef_)) / np.max(np.abs(ref.coef_))))
            g["icpt"] = max(g["icpt"], abs(icpt - ref.intercept_) / max(1.0, abs(ref.intercept_)))
            g["alpha"] = max(g["alpha"], abs(alpha - ref.alpha_) / ref.alpha_)
            g["lam"] = max(g["lam"], abs(lam - ref.lambda_) / ref.lambda_)
            xq = np.random.default_rng(sd + 1).standard_normal((20, k))
            std_mine = np.sqrt(1.0 / alpha + np.einsum("ij,jl,il->i", xq - m, Sigma, xq - m))
            _, std_ref = ref.predict(xq, return_std=True)
            g["std"] = max(g["std"], float(np.max(np.abs(std_mine - std_ref) / std_ref)))
        rows.append([label, g["coef"], g["icpt"], g["alpha"], g["lam"], g["std"]])
    table("524a evidence-maximized ridge from the accumulator vs `BayesianRidge(alpha_1, alpha_2, lambda_1, lambda_2, tol=1e-14, max_iter=5000).fit(X, y, sample_weight=w')`, "
          "`w'` the EW weights scaled to sum to Kish's `n` (k = 6, 160 rows; relative gaps; `pred_std` = `predict(X, return_std=True)`)",
          ["case", "coef", "intercept", "alpha_", "lambda_", "pred_std"], rows)

    # defaults: tol 1e-3 (stopping rule), warm start
    g = 0.0
    it_cold = it_warm = 0
    for sd in range(seeds):
        k, n = 6, 160
        x, y = stream_xy(sd, n, k)
        ew = run_moments(x, y, np.ones(n), np.ones(n), math.inf)
        coef, icpt, alpha, lam, Sigma, it, _ = bayes_ridge_from_moments(ew, k)
        ref = BayesianRidge().fit(x, y)
        g = max(g, float(np.max(np.abs(coef - ref.coef_)) / np.max(np.abs(ref.coef_))), abs(alpha - ref.alpha_) / ref.alpha_)
        assert it == ref.n_iter_, (it, ref.n_iter_)
        # warm start one row later
        ew2 = run_moments(x[: n], y[: n], np.ones(n), np.ones(n), math.inf)
        x2, y2 = stream_xy(sd + 77, 1, k)
        ew2.update(np.concatenate([x2[0], [y2[0]]]), 1.0, 1.0)
        c0 = bayes_ridge_from_moments(ew2, k)
        c1 = bayes_ridge_from_moments(ew2, k, init=(alpha, lam))
        it_cold += c0[5]
        it_warm += c1[5]
    print(f"\n524a' at sklearn's defaults (`tol=1e-3`, `max_iter=300`) the same stopping rule gives the same iteration count and the same fit: gap {g:.1e} (iterations equal on every seed). "
          f"Warm-starting `(alpha, lam)` from the previous solve (one row later): mean iterations {it_warm / seeds:.1f} against {it_cold / seeds:.1f} cold.")


# ------------------------------------------------------------------ averaged sgd / pa
def sgd_iterates(x, y, w, dt, hl, lr, intercept=True):
    """rheon's plain `sgd` step (no standardize, no ridge): `b -= lr (p - y) z w`. Returns the list of iterates after each learned row."""
    n, k = x.shape
    b = np.zeros(k + (1 if intercept else 0))
    its = []
    for t in range(n):
        if w[t] == 0.0:
            its.append(None)
            continue
        z = np.concatenate([[1.0], x[t]]) if intercept else x[t]
        p = z @ b
        b = b - lr * (p - y[t]) * z * w[t]
        its.append(b.copy())
    return its


def pa_iterates(x, y, w, C, eps):
    """PA-I without an intercept: `tau = min(C, loss/|z|^2)`, `b += min(w,1) tau sign(y-p) z`; sklearn's `pa1` has no row weight cap, so w = 1 here."""
    n, k = x.shape
    b = np.zeros(k)
    its = []
    for t in range(n):
        p = x[t] @ b
        loss = max(0.0, abs(y[t] - p) - eps)
        tau = min(C, loss / float(x[t] @ x[t]))
        b = b + tau * np.sign(y[t] - p) * x[t]
        its.append(b.copy())
    return its


class Averager:
    """EW mean of the iterates in the mean form: W' = lam W + w, bbar += (w/W')(b - bbar)."""

    def __init__(self, k):
        self.W = 0.0
        self.bbar = np.zeros(k)

    def update(self, b, lam, w):
        Wo = lam * self.W
        W2 = Wo + w
        self.W = W2
        if w == 0.0 or W2 == 0.0:
            return
        self.bbar = self.bbar + (w / W2) * (b - self.bbar)


def report_average(seeds):
    rows = []
    for k in (3, 10):
        g_c = g_i = 0.0
        for sd in range(seeds):
            n = 300
            x, y = stream_xy(sd, n, k)
            w = np.ones(n)
            lr = 0.01
            its = sgd_iterates(x, y, w, np.ones(n), math.inf, lr)
            av = Averager(k + 1)
            for t in range(n):
                av.update(its[t], 1.0, 1.0)
            ref = SGDRegressor(loss="squared_error", penalty=None, alpha=0.0, learning_rate="constant", eta0=lr, average=True, fit_intercept=True, shuffle=False, tol=None, max_iter=1)
            for t in range(n):
                ref.partial_fit(x[t : t + 1], y[t : t + 1])
            g_i = max(g_i, abs(av.bbar[0] - ref.intercept_[0]))
            g_c = max(g_c, float(np.max(np.abs(av.bbar[1:] - ref.coef_))))
        rows.append([f"sgd, k = {k}, lr .01, half_life inf, unit w", g_c, g_i])
    # pa
    for k in (3, 10):
        g_c = 0.0
        for sd in range(seeds):
            n = 300
            x, y = stream_xy(sd, n, k)
            its = pa_iterates(x, y, np.ones(n), 1.0, 0.1)
            av = Averager(k)
            for t in range(n):
                av.update(its[t], 1.0, 1.0)
            ref = SGDRegressor(loss="epsilon_insensitive", epsilon=0.1, penalty=None, learning_rate="pa1", eta0=1.0, average=True, fit_intercept=False, shuffle=False, tol=None, max_iter=1)
            for t in range(n):
                ref.partial_fit(x[t : t + 1], y[t : t + 1])
            g_c = max(g_c, float(np.max(np.abs(av.bbar - ref.coef_))))
        rows.append([f"pa (PA-I, C = 1, eps = .1, no intercept), k = {k}, half_life inf", g_c, ""])
    table("524b averaged iterates vs `SGDRegressor(average=True)` row by row with `partial_fit` (C1: `PassiveAggressiveRegressor` is deprecated since 1.8, so the pa oracle is "
          "`SGDRegressor(loss=\"epsilon_insensitive\", penalty=None, learning_rate=\"pa1\", eta0=C)`; 300 rows; absolute gaps; `pa` without an intercept, since rheon puts the intercept inside `|z|^2` and sklearn outside)",
          ["case", "coef gap", "intercept gap"], rows)

    # decay / weights / zero-weight rows against the two-pass average of recorded iterates
    g = g_fold = 0.0
    for sd in range(seeds):
        n, k = 250, 4
        x, y = stream_xy(sd, n, k)
        dt, w = dt_stream(sd, n)
        its = sgd_iterates(x, y, w, dt, 40.0, 0.01)
        av = Averager(k + 1)
        for t in range(n):
            av.update(its[t] if its[t] is not None else av.bbar, 1.0 if t == 0 else lam_of(dt[t], 40.0), w[t])
        om = omega_weights(dt, w, 40.0)
        idx = [t for t in range(n) if w[t] > 0]
        ref = sum(om[t] * its[t] for t in idx) / om.sum()
        g = max(g, float(np.max(np.abs(av.bbar - ref)) / np.max(np.abs(ref))))
        # fold: drop zero-weight rows, carry their gaps
        av2 = Averager(k + 1)
        carry, first = 0.0, True
        for t in range(n):
            carry += 0.0 if t == 0 else dt[t]
            if w[t] > 0:
                av2.update(its[t], 1.0 if first else lam_of(carry, 40.0), w[t])
                first, carry = False, 0.0
        g_fold = max(g_fold, float(np.max(np.abs(av.bbar - av2.bbar)) / np.max(np.abs(ref))))
    table("524b' the decay-weighted average (half-life 40, row weights, 10% zero-weight rows, a zero-weight first row; the iterates are rheon's `sgd` step): vs the two-pass `sum omega b_i / sum omega`, and vs the stream with zero-weight rows removed and gaps folded",
          ["comparison", "relative gap"], [["two-pass definition", g], ["gaps folded", g_fold]])
    # state
    print("\n524b state added: one k_total-vector per target, `8 (k + 1)` bytes with an intercept (k = 20: 168; k = 100: 808) and the shared weight `W` (8); the iterate `b` is kept as it is, so the average doubles an `sgd` target's coefficient state.")


# ------------------------------------------------------------------ positive lasso
def cd_enet(G, c, lam_pen, l1_ratio, positive, tol=1e-15, max_sweeps=20000):
    """Coordinate descent on `1/2 b'Gb - c'b + lam l1 |b|_1 + lam (1-l1)/2 |b|^2` (`lasso.rs`'s update); `positive` clips at 0."""
    k = len(c)
    b = np.zeros(k)
    for _ in range(max_sweeps):
        d = 0.0
        for i in range(k):
            rho = c[i] - G[i] @ b + G[i, i] * b[i]
            if positive:
                new = max(rho - lam_pen * l1_ratio, 0.0) / (G[i, i] + lam_pen * (1 - l1_ratio))
            else:
                new = np.sign(rho) * max(abs(rho) - lam_pen * l1_ratio, 0.0) / (G[i, i] + lam_pen * (1 - l1_ratio))
            d = max(d, abs(new - b[i]))
            b[i] = new
        if d < tol:
            break
    return b


def report_positive(seeds):
    rows = []
    for label, hl, weighted, zeros, l1r in (
        ("lam=1, unit w, lasso", math.inf, False, False, 1.0),
        ("lam=1, unit w, elastic net l1_ratio .5", math.inf, False, False, 0.5),
        ("half-life 50, w, 10% zero w, zero first row, lasso", 50.0, True, True, 1.0),
        ("half-life 50, w, zero w, elastic net .5", 50.0, True, True, 0.5),
    ):
        g = 0.0
        g_unc = 0.0
        npos_active = 0
        for sd in range(seeds):
            k, n = 8, 200
            x, y = stream_xy(sd, n, k)
            dt, w = dt_stream(sd, n, zero_first=zeros, zero_frac=0.1 if zeros else 0.0)
            if not weighted:
                w = np.ones(n)
            ew = run_moments(x, y, dt, w, hl)
            m, C = ew.m, ew.C
            s = np.sqrt(np.diag(C[:k, :k]))
            Gz = C[:k, :k] / np.outer(s, s)  # correlation matrix of the standardized features
            cz = C[:k, k] / s
            lam_pen = 0.15
            for positive in (True, False):
                bz = cd_enet(Gz, cz, lam_pen, l1r, positive)
                b = bz / s
                om = omega_weights(dt, w, hl)
                Xs = x / s
                est = ElasticNet(alpha=lam_pen, l1_ratio=l1r, positive=positive, tol=1e-14, max_iter=100000, fit_intercept=True) if l1r < 1 else None
                from sklearn.linear_model import Lasso

                est = est if est is not None else Lasso(alpha=lam_pen, positive=positive, tol=1e-14, max_iter=100000, fit_intercept=True)
                est.fit(Xs, y, sample_weight=om)
                gap = float(np.max(np.abs(bz - est.coef_)))
                if positive:
                    assert np.all(bz >= 0)
                    icpt_mine = m[k] - (m[:k] / s) @ bz
                    g = max(g, gap, abs(icpt_mine - est.intercept_) / max(1.0, abs(est.intercept_)))
                    npos_active += int(np.sum(bz == 0.0))
                else:
                    g_unc = max(g_unc, gap)
        rows.append([label, g, g_unc, npos_active / seeds])
    table("524c `positive=True` by the clipped coordinate step vs `Lasso(positive=True)` / `ElasticNet(positive=True, l1_ratio)` fitted on the standardized features with `sample_weight` = the EW weights "
          "(`tol=1e-14`; alpha .15; k = 8, 200 rows; coefficient gap in the standardized coordinates; the unconstrained step vs sklearn as the control; mean number of coefficients held at 0 by the constraint)",
          ["case", "positive: coef and intercept gap", "unconstrained control", "zeros per fit"], rows)
    print("\n524c state added: 0 (the constraint is in the solve; the warm start is the same k-vector of coefficients). Per-feature penalty weights were not built: scikit-learn has no such parameter (C1: MISSING; its column-rescaling device is an identity, not an oracle).")


def report(seeds: int = 20) -> None:
    warnings.filterwarnings("ignore")
    report_ridge(seeds)
    report_average(seeds)
    report_positive(seeds)
