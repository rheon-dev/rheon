"""Task 524 part 2: redescending norms for `huber`, the `sgd` elastic net with scikit-learn's `optimal` schedule,
and Vowpal Wabbit's importance-invariant step.

* Norms. `huber` reweights each row by the robust weight of its *prior* (out-of-sample) residual `u = r/(c s)`: `w(u) = psi(u)/u`.
  A redescending norm changes the weight function only (Tukey: `(1 - u^2)^2` for `|u| < 1`, else 0, so a row far outside learns
  nothing: a zero-weight row for the Gram). Exact part: each weight function against `statsmodels.robust.norms.*.weights`.
  Statistical part: the one-pass reweighted fit against the converged `RLM`.
* `sgd` elastic net (Tsuruoka's cumulative penalty, as `SGDRegressor`) and the `optimal` schedule `eta_t = 1/(alpha (t0 + t - 1))`,
  `t0 = 1/(eta_0 alpha)`, `eta_0 = sqrt(1/sqrt(alpha))` for the squared loss (the source's `typw / max(1, gradient(1, -typw))`):
      b <- b (1 - (1 - l1r) eta alpha);  b += -eta (p - y) w z;  u += l1r eta alpha;  truncate each b_j by (u + q_j) / (u - q_j);  q_j += b_j' - b_j
* importance-invariant step (Karampatziakis and Langford): instead of `b += lr w (y - p) z`,
      b += (y - p) (1 - exp(-lr w |z|^2)) / |z|^2 z,   so `p' = y + (p - y) exp(-lr w |z|^2)`: a heavy row cannot overshoot its target.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
from scipy import stats

from .common import EwMoments, table


# ---------------------------------------------------------------- weight functions
def w_huber(u, c=1.345):
    a = np.abs(u)
    return np.where(a <= c, 1.0, c / np.maximum(a, 1e-300))


def w_tukey(u, c=4.685):
    a = np.abs(u) / c
    return np.where(a < 1, (1 - a**2) ** 2, 0.0)


def w_andrews(u, a=1.339):
    z = np.abs(u) / a
    return np.where(z <= np.pi, np.sinc(z / np.pi), 0.0)  # sin(z)/z


def w_hampel(u, a=2.0, b=4.0, c=8.0):
    z = np.abs(u)
    out = np.where(z <= a, 1.0, np.where(z <= b, a / np.maximum(z, 1e-300), np.where(z <= c, a * (c - z) / (np.maximum(z, 1e-300) * (c - b)), 0.0)))
    return out


def w_ramsay(u, a=0.3):
    z = np.abs(u)
    return np.exp(-a * z)


class OnePass:
    """One-pass reweighted least squares from the mean-form Gram, fixed scale `s`, the first `warm` rows at weight 1."""

    def __init__(self, k, weight_fn, c_scale, s, warm):
        self.k, self.fn, self.cs, self.s, self.warm = k, weight_fn, c_scale, s, warm
        self.ew = EwMoments(k + 1)
        self.beta = np.zeros(k)
        self.icpt = 0.0
        self.n = 0

    def solve(self):
        C = self.ew.C
        k = self.k
        try:
            self.beta = np.linalg.solve(C[:k, :k] + 1e-12 * np.eye(k), C[:k, k])
        except np.linalg.LinAlgError:
            return
        self.icpt = self.ew.m[k] - self.ew.m[:k] @ self.beta

    def step(self, x, y, w=1.0):
        r = y - (self.icpt + x @ self.beta)
        rw = 1.0 if self.n < self.warm else float(self.fn(r / self.s))
        self.ew.update(np.concatenate([x, [y]]), 1.0, w * rw)
        self.n += 1
        self.solve()


def contaminated(sd, n, k, frac, scale=8.0):
    rng = np.random.default_rng(sd)
    x = rng.standard_normal((n, k))
    beta = np.arange(1, k + 1, dtype=float)
    noise = rng.standard_normal(n)
    out = rng.random(n) < frac
    noise[out] = scale * rng.standard_normal(out.sum()) + 6.0  # one-sided heavy outliers
    return x, 2.0 + x @ beta + noise, beta


def report_norms(seeds):
    import statsmodels.api as sm
    from statsmodels.robust import norms

    # exact weight functions
    u = np.linspace(-12, 12, 4801)
    rows = []
    for name, fn, nrm in (
        ("HuberT(t=1.345)", w_huber, norms.HuberT(t=1.345)),
        ("TukeyBiweight(c=4.685)", w_tukey, norms.TukeyBiweight(c=4.685)),
        ("AndrewWave(a=1.339)", w_andrews, norms.AndrewWave(a=1.339)),
        ("Hampel(2, 4, 8)", w_hampel, norms.Hampel(a=2.0, b=4.0, c=8.0)),
        ("RamsayE(a=0.3)", w_ramsay, norms.RamsayE(a=0.3)),
    ):
        ref = nrm.weights(u)
        rows.append([name, float(np.max(np.abs(fn(u) - ref))), int(np.sum(ref == 0)), "yes" if np.any(ref == 0) else "no"])
    table("524d the weight functions `w(u) = psi(u)/u` vs `statsmodels.robust.norms.*.weights` on 4801 points in [-12, 12] (`TukeyBiweight`: C1: the name is not `TukeyBisquare`)",
          ["norm", "largest gap", "points with weight exactly 0", "redescending"], rows)

    # one-pass vs converged RLM, fixed scale
    class FixedRLM(sm.RLM):
        _fixed = 1.0

        def _estimate_scale(self, resid, scale_est=None):
            return FixedRLM._fixed

    rows = []
    for n in (400, 3000):
        for frac in (0.0, 0.15):
            err = {"ols": [], "huber one-pass": [], "huber RLM": [], "tukey one-pass": [], "tukey RLM": [], "gap tukey": [], "gap huber": []}
            for sd in range(max(8, seeds // 2)):
                k = 3
                x, y, beta = contaminated(sd, n, k, frac)
                s = 1.0  # the clean noise scale, as a fixed `huber_scale`
                FixedRLM._fixed = s
                for tag, fn, c, nrm in (("huber", w_huber, 1.345, norms.HuberT(t=1.345)), ("tukey", w_tukey, 4.685, norms.TukeyBiweight(c=4.685))):
                    op = OnePass(k, fn, 1.0, s, warm=3 * (k + 1))
                    for t in range(n):
                        op.step(x[t], y[t])
                    rlm = FixedRLM(y, sm.add_constant(x), M=nrm).fit(update_scale=False, maxiter=200, tol=1e-12)
                    err[f"{tag} one-pass"].append(float(np.max(np.abs(op.beta - beta))))
                    err[f"{tag} RLM"].append(float(np.max(np.abs(rlm.params[1:] - beta))))
                    err[f"gap {tag}"].append(float(np.max(np.abs(op.beta - rlm.params[1:]))))
                err["ols"].append(float(np.max(np.abs(np.linalg.lstsq(sm.add_constant(x), y, rcond=None)[0][1:] - beta))))
            rows.append([f"n = {n}, {int(frac * 100)}% one-sided outliers", *(float(np.mean(err[k])) for k in ("ols", "huber one-pass", "huber RLM", "tukey one-pass", "tukey RLM", "gap huber", "gap tukey"))])
    table("524d' one-pass reweighting (the prior residual's weight, the fit from the first `3 (k+1)` rows at weight 1, scale fixed at the clean noise's 1) vs the converged `RLM(M=...)` with the same fixed scale: "
          "mean over 8-10 seeds of the largest coefficient error against the truth (k = 3), and the gap between the one-pass and the RLM coefficients",
          ["stream", "OLS", "Huber one-pass", "Huber RLM", "Tukey one-pass", "Tukey RLM", "gap Huber", "gap Tukey"], rows)


# ---------------------------------------------------------------- sgd elastic net, optimal schedule
def sgd_enet(x, y, w, alpha, l1r, sched="optimal", eta0=0.01, power_t=0.25):
    n, k = x.shape
    b = np.zeros(k)
    icpt = 0.0
    q = np.zeros(k)
    u = 0.0
    t = 1.0
    typw = math.sqrt(1.0 / math.sqrt(alpha))
    t0 = 1.0 / (typw * alpha)  # squared loss: typw / max(1, grad) with grad < 1
    for i in range(n):
        if w[i] == 0.0:  # a zero-weight row: the schedule counts learned rows, nothing moves
            continue
        eta = 1.0 / (alpha * (t0 + t - 1)) if sched == "optimal" else (eta0 / t**power_t if sched == "invscaling" else eta0)
        p = x[i] @ b + icpt
        upd = -eta * (p - y[i]) * w[i]
        b = b * max(0.0, 1.0 - (1.0 - l1r) * eta * alpha)
        b = b + upd * x[i]
        icpt += upd
        if l1r > 0:
            u += l1r * eta * alpha
            for j in range(k):
                z = b[j]
                if z > 0:
                    b[j] = max(0.0, z - (u + q[j]))
                elif z < 0:
                    b[j] = min(0.0, z + (u - q[j]))
                q[j] += b[j] - z
        t += 1
    return b, icpt


def report_sgd(seeds):
    from sklearn.linear_model import SGDRegressor

    rows = []
    for sched in ("optimal", "invscaling", "constant"):
        for pen, l1r in (("l2", 0.0), ("l1", 1.0), ("elasticnet", 0.4)):
            g_c = g_i = 0.0
            nz = 0
            for sd in range(seeds):
                rng = np.random.default_rng(sd)
                n, k = 300, 6
                x = 0.3 * rng.standard_normal((n, k))
                y = x @ rng.standard_normal(k) * 3 + 0.3 + 0.2 * rng.standard_normal(n)
                w = rng.uniform(0.5, 2.0, n)
                alpha = 0.05
                b, ic = sgd_enet(x, y, w, alpha, l1r, sched)
                ref = SGDRegressor(loss="squared_error", penalty=pen, alpha=alpha, l1_ratio=l1r if pen == "elasticnet" else 0.15, learning_rate=sched, eta0=0.01, power_t=0.25,
                                   fit_intercept=True, shuffle=False, tol=None, max_iter=1)
                ref.partial_fit(x, y, sample_weight=w)  # ONE call: u and q (the cumulative penalty) are per call in scikit-learn
                g_c = max(g_c, float(np.max(np.abs(b - ref.coef_))))
                g_i = max(g_i, abs(ic - ref.intercept_[0]))
                nz += int(np.sum(b == 0.0))
            rows.append([f"{sched}, {pen}", g_c, g_i, nz / seeds])
    table("524e `sgd` with the elastic net's cumulative penalty and scikit-learn's schedules vs `SGDRegressor(penalty, alpha=.05, learning_rate, shuffle=False).partial_fit(X, y, sample_weight)` (one call: scikit-learn "
          "re-creates the cumulative penalty `u, q` on every call, so a row-by-row `partial_fit` loses the L1 memory); row weights in [.5, 2]; absolute gaps; coefficients held at 0 per fit",
          ["schedule, penalty", "coef gap", "intercept gap", "zeros"], rows)
    # zero weights: skip
    g = 0.0
    for sd in range(seeds):
        rng = np.random.default_rng(sd)
        n, k = 200, 4
        x = 0.3 * rng.standard_normal((n, k))
        y = x @ np.ones(k) + 0.1 * rng.standard_normal(n)
        w = rng.uniform(0.5, 2.0, n)
        w[rng.random(n) < 0.15] = 0.0
        w[0] = 0.0
        a = sgd_enet(x, y, w, 0.05, 0.4, "optimal")
        keep = w > 0
        b = sgd_enet(x[keep], y[keep], w[keep], 0.05, 0.4, "optimal")
        g = max(g, float(np.max(np.abs(a[0] - b[0]))), abs(a[1] - b[1]))
    print(f"\n524e' zero-weight rows (15%, first included) are skipped by the schedule (`t` counts learned rows) and by the penalty: against the stream without them, gap {g:.1e}")
    print("524e state added: `q`, one `k`-vector (the cumulative penalty), `u`, the schedule's `t`: `8 (k + 2)` bytes a target (k = 20: 176) under an L1 part; 16 bytes under `optimal` with L2 only; `sgd` has no coefficient decay, so decay does not enter.")


# ---------------------------------------------------------------- VW importance-invariant step
def invariant_run(Z, y, w, lr, invariant=True):
    k = Z.shape[1]
    b = np.zeros(k)
    preds = []
    for t in range(len(y)):
        p = Z[t] @ b
        preds.append(p)
        if w[t] == 0.0:
            continue
        zz = float(Z[t] @ Z[t])
        if invariant:
            b = b + (y[t] - p) * (-math.expm1(-lr * w[t] * zz)) / zz * Z[t]
        else:
            b = b + lr * w[t] * (y[t] - p) * Z[t]
    return np.array(preds), b


def report_vw(seeds):
    import vowpalwabbit as vw

    rows = []
    for lr in (0.05, 0.5):
        for hi in (2.0, 20.0):
            g_pred = g_w = 0.0
            for sd in range(max(4, seeds // 4)):
                rng = np.random.default_rng(sd)
                n, k = 200, 4
                x = rng.standard_normal((n, k))
                y = 1.0 + x @ rng.standard_normal(k) + 0.3 * rng.standard_normal(n)
                w = rng.uniform(0.5, hi, n)
                w[rng.random(n) < 0.1] = 0.0
                Z = np.column_stack([np.ones(n), x])
                # VW: `--sgd --invariant` (plain SGD with the importance-invariant update; its squared loss is (p - y)^2, so its rate is lr/2 of the 1/2 (p - y)^2 form), explicit constant
                m = vw.Workspace(f"--sgd --invariant --quiet --noconstant -l {lr / 2} --power_t 0 --initial_t 0 -b 20 --loss_function squared --min_prediction -1e6 --max_prediction 1e6")
                pv = []
                for t in range(n):
                    ex = f"{float(y[t])!r} {float(w[t])!r} |f " + " ".join(f"a{i}:{float(Z[t, i])!r}" for i in range(k + 1))
                    pv.append(m.predict(ex))
                    if w[t] > 0:
                        m.learn(ex)
                wv = np.array([m.predict("0 |f " + f"a{i}:1") for i in range(k + 1)])
                m.finish()
                preds, b = invariant_run(Z, y, w, lr)
                g_pred = max(g_pred, float(np.max(np.abs(preds - np.array(pv)) / (1 + np.abs(preds)))))
                g_w = max(g_w, float(np.max(np.abs(b - wv) / (1 + np.abs(b)))))
            rows.append([f"lr = {lr}, weights in [.5, {hi}], 10% zero-weight rows", g_pred, g_w])
    table("524f the importance-invariant step vs Vowpal Wabbit 9.11.9 `--sgd --invariant` (C1: `--invariant` is the safe / importance-aware update; `--sgd` switches off the adaptive and normalized updates), row-by-row predictions and final weights (relative; VW keeps float32 weights)",
          ["case", "predictions", "weights"], rows)
    # the property, and the small-step limit
    rng = np.random.default_rng(3)
    Z = np.column_stack([np.ones(50), rng.standard_normal((50, 3))])
    y = rng.standard_normal(50)
    p1, b1 = invariant_run(Z, y, np.full(50, 1e-4), 1.0, True)
    p0, b0 = invariant_run(Z, y, np.full(50, 1e-4), 1.0, False)
    resid_inv = 0.0
    plain_over = 0
    for t in range(50):
        zz = float(Z[t] @ Z[t])
        h = 1e6
        bn = (y[t]) * (-math.expm1(-h * zz)) / zz * Z[t]
        resid_inv = max(resid_inv, abs(y[t] - Z[t] @ bn))
        bp = h * y[t] * Z[t]
        plain_over += float((y[t] - Z[t] @ bp) * y[t] < 0)
    print(f"\n524f' small-step limit: at `lr w |z|^2` ~ 1e-4 the invariant and the plain step differ by {float(np.max(np.abs(b1 - b0))):.1e} in the coefficients (first order); "
          f"at a huge weight (1e6) the plain step overshoots the target on {int(plain_over)}/50 rows, the invariant one leaves a residual of at most {resid_inv:.1e} (`p' = y + (p - y) e^(-lr w |z|^2)`: it moves to the target and stops).")
    print("524f state added: 0. A zero-weight row is a no-op (`1 - e^0 = 0`). It changes nothing by default (`invariant = false`); at `lr w |z|^2 << 1` the two agree to the first order.")


def report(seeds: int = 20) -> None:
    warnings.filterwarnings("ignore")
    report_norms(seeds)
    report_sgd(seeds)
    report_vw(seeds)
