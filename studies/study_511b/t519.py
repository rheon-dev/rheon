"""Task 519: `kalman` -- `coef_prior`, the innovation log-likelihood and the Mahalanobis distance.

The recursion it extends (`kalman.rs`, in units of the noise, a random-walk-beta filter): per target,
coefficient mean `b`, covariance `P`, the clock `D` since `P` last took an observation. A row of clock
step `d`, weight `w`, features `z` (intercept first):

    D  += d                          (every row, a zero-weight row included)
    on a row that observes (w > 0):  P += q D^2 I ; D = 0 ;  Pz = P z ;  s = z'Pz + 1/w ;  nu = y - z'b
        k = Pz / s ;  b += k nu ;  P -= k (Pz)'
    the first observation is the diffuse step on the intercept (Durbin & Koopman 5.2): b_0 += nu,
        P_00 = s, P_0i = -(Pz)_i, the slopes' block unmoved.

New outputs, read on the row's out-of-sample innovation before the update:

    innovation log-likelihood   ll = -0.5 (ln 2 pi + ln(R s) + nu^2 / (R s))     R the noise estimate
    Mahalanobis distance        mahal = |nu| / sqrt(R s)    (sqrt, as filterpy's `mahalanobis`)
    coef_prior                  b_0 = coef_prior  (slopes; the intercept's prior is moot: see report)

Oracle: filterpy's `KalmanFilter` (`x`, `P`, `log_likelihood`, `mahalanobis`), per C1 `mahalanobis` is the
distance `sqrt(y' S^-1 y)`, not the squared form.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
from filterpy.kalman import KalmanFilter

from .common import stream, table


class KalmanRW:
    def __init__(self, k, p0, q, prior=None, noise=1.0, intercept=True):
        self.k, self.intercept = k, intercept
        kt = k + (1 if intercept else 0)
        self.kt = kt
        self.b = np.zeros(kt)
        if prior is not None:
            self.b[(1 if intercept else 0):] = prior  # the intercept's prior is moot (diffuse)
        self.P = np.zeros((kt, kt))
        s0 = 1 if intercept else 0
        for i in range(s0, kt):
            self.P[i, i] = p0
        self.q, self.D, self.R = q, 0.0, noise
        self.seen = not intercept
        self.n_obs = 0

    def step(self, x, y, d, w):
        """Returns (pred, ll, mahal); ll and mahal are NaN on the diffuse row and on a zero-weight row."""
        z = np.concatenate([[1.0], x]) if self.intercept else np.asarray(x, float)
        pred = float(z @ self.b)
        self.D += d
        if w == 0.0:
            return pred, math.nan, math.nan
        if self.seen:
            self.P = self.P + self.q * self.D**2 * np.eye(self.kt)
        else:  # first observation: the intercept's row and column are still 0 and stay out of Q
            for i in range(1, self.kt):
                self.P[i, i] += self.q * self.D**2
        self.D = 0.0
        Pz = self.P @ z
        s = float(z @ Pz) + 1.0 / w
        nu = y - pred
        if not self.seen:
            # diffuse step on the intercept
            self.b[0] += nu
            self.P[0, 0] = s
            self.P[0, 1:] = -Pz[1:]
            self.P[1:, 0] = -Pz[1:]
            self.seen = True
            self.n_obs += 1
            return pred, math.nan, math.nan
        S = self.R * s
        ll = -0.5 * (math.log(2 * math.pi) + math.log(S) + nu * nu / S)
        mahal = abs(nu) / math.sqrt(S)
        kgain = Pz / s
        self.b = self.b + kgain * nu
        self.P = self.P - np.outer(kgain, Pz)
        self.n_obs += 1
        return pred, ll, mahal


def run_filterpy(x, y, dt, w, p0, q, prior, R, kappa=None):
    """filterpy on the same rows. P in absolute units: R * (noise-unit P). `kappa` None: no intercept."""
    n, k = x.shape
    kt = k + (0 if kappa is None else 1)
    kf = KalmanFilter(dim_x=kt, dim_z=1)
    kf.F = np.eye(kt)
    s0 = 0 if kappa is None else 1
    kf.x = np.zeros((kt, 1))
    if prior is not None:
        kf.x[s0:, 0] = prior
    P0 = np.zeros((kt, kt))
    for i in range(s0, kt):
        P0[i, i] = R * p0
    if kappa is not None:
        P0[0, 0] = R * kappa
    kf.P = P0
    out = []
    D = 0.0
    for t in range(n):
        D += dt[t]
        if w[t] == 0.0:
            out.append((float(np.concatenate([[1.0] if kappa is not None else [], x[t]]) @ kf.x[:, 0]), math.nan, math.nan, kf.x[:, 0].copy()))
            continue
        z = np.concatenate([[1.0] if kappa is not None else [], x[t]])
        pred = float(z @ kf.x[:, 0])
        kf.predict(Q=R * q * D**2 * np.eye(kt))
        D = 0.0
        kf.update(np.array([y[t]]), R=np.array([[R / w[t]]]), H=z[None, :])
        out.append((pred, float(kf.log_likelihood), float(kf.mahalanobis), kf.x[:, 0].copy()))
    return out


def _case(sd, n, k, hetero_w, zeros, p0=4.0, q=1e-4, R=0.7, intercept=True):
    x, y, dt, w = stream(sd, n, k)
    dt = dt.copy()
    dt[0] = 0.0
    if not hetero_w:
        w = np.ones(n)
    if zeros:
        rng = np.random.default_rng(sd + 5)
        w = w.copy()
        w[rng.random(n) < 0.1] = 0.0
        w[0] = 0.0
    prior = np.random.default_rng(sd + 11).standard_normal(k) * 0.5
    return x, y, dt, w, prior, p0, q, R


def report(seeds: int = 20) -> None:
    warnings.filterwarnings("ignore")
    rows = []
    # --- without an intercept: exact against filterpy
    for label, hetero_w, zeros, q in (("q = 0, unit w", False, False, 0.0), ("q = 1e-4, w in [.5,2]", True, False, 1e-4), ("q = 1e-4, w, 10% zero w, zero first row", True, True, 1e-4)):
        g = {"x": 0.0, "ll": 0.0, "mahal": 0.0, "pred": 0.0}
        for sd in range(seeds):
            for k in (2, 8):
                x, y, dt, w, prior, p0, _, R = _case(sd, 200, k, hetero_w, zeros, q=q)
                kal = KalmanRW(k, p0, q, prior, noise=R, intercept=False)
                ref = run_filterpy(x, y, dt, w, p0, q, prior, R, kappa=None)
                for t in range(len(x)):
                    pred, ll, mh = kal.step(x[t], y[t], dt[t], w[t])
                    rp, rll, rmh, rb = ref[t]
                    g["pred"] = max(g["pred"], abs(pred - rp))
                    if not math.isnan(rll):
                        g["ll"] = max(g["ll"], abs(ll - rll) / max(1.0, abs(rll)))
                        g["mahal"] = max(g["mahal"], abs(mh - rmh) / max(1.0, rmh))
                    else:
                        assert math.isnan(ll)
                g["x"] = max(g["x"], float(np.max(np.abs(kal.b - ref[-1][3]))))
        rows.append([label, g["x"], g["pred"], g["ll"], g["mahal"]])
    table("519a no intercept, `coef_prior` on every coefficient, vs filterpy (k = 2 and 8, 200 rows; absolute for x and pred, relative for ll and mahal)",
          ["case", "x (coef)", "pred", "log_likelihood", "mahalanobis (sqrt)"], rows)

    # --- with the diffuse intercept: filterpy with a large intercept variance kappa
    rows = []
    for kappa in (1e4, 1e6, 1e8, 1e10):
        g = {"x": 0.0, "ll": 0.0, "mahal": 0.0}
        for sd in range(seeds):
            k = 4
            x, y, dt, w, prior, p0, q, R = _case(sd, 200, k, True, True)
            kal = KalmanRW(k, p0, q, prior, noise=R)
            ref = run_filterpy(x, y, dt, w, p0, q, prior, R, kappa=kappa)
            first = True
            for t in range(len(x)):
                pred, ll, mh = kal.step(x[t], y[t], dt[t], w[t])
                rp, rll, rmh, rb = ref[t]
                if w[t] > 0 and first:
                    first = False  # the diffuse row: ll withheld here, finite in the filterpy limit
                    continue
                if not math.isnan(rll):
                    g["ll"] = max(g["ll"], abs(ll - rll) / max(1.0, abs(rll)))
                    g["mahal"] = max(g["mahal"], abs(mh - rmh) / max(1.0, rmh))
            g["x"] = max(g["x"], float(np.max(np.abs(kal.b - ref[-1][3]))))
        rows.append([f"intercept prior variance kappa = {kappa:.0e}", g["x"], g["ll"], g["mahal"]])
    table("519b the diffuse intercept against filterpy at a large prior variance (k = 4, 200 rows, w, 10% zero w, zero first row)",
          ["case", "x (coef), abs", "log_likelihood, rel", "mahalanobis, rel"], rows)

    # --- coef_prior: the intercept's prior is moot; the slopes' prior is not; closed-form ridge toward it
    rows = []
    for p0 in (0.5, 4.0, 100.0):
        g = 0.0
        for sd in range(seeds):
            k = 4
            x, y, dt, w, prior, _, _, R = _case(sd, 150, k, True, False)
            kal = KalmanRW(k, p0, 0.0, prior, noise=R)
            for t in range(len(x)):
                kal.step(x[t], y[t], 0.0, w[t])
            Z = np.column_stack([np.ones(len(x)), x])
            A = Z.T @ (w[:, None] * Z)
            pen = np.diag([0.0] + [1.0 / p0] * k)
            rhs = Z.T @ (w * y) + pen @ np.concatenate([[0.0], prior])
            ref = np.linalg.solve(A + pen, rhs)
            g = max(g, float(np.max(np.abs(kal.b - ref))))
        rows.append([f"p0 = {p0}", g])
    table("519c `coef_prior` at `q = 0`: the filter's final coefficients vs the closed form `argmin sum w (y - z'b)^2 + (1/p0) |b_slopes - prior|^2` (intercept unpenalized)",
          ["slope prior variance", "largest gap (abs)"], rows)
    # the intercept prior is moot
    g = 0.0
    x, y, dt, w, prior, p0, q, R = _case(0, 100, 3, True, False)
    a = KalmanRW(3, p0, q, prior, R)
    b = KalmanRW(3, p0, q, prior, R)
    b.b[0] = 123.0  # a different intercept prior
    for t in range(100):
        a.step(x[t], y[t], dt[t], w[t])
        b.step(x[t], y[t], dt[t], w[t])
    print(f"\n519 intercept prior set to 123 vs 0: final coefficient gap {np.max(np.abs(a.b - b.b)):.1e} (the diffuse step discards it; "
          f"rounding only: it moves with `|nu|`'s size, here 1e2)")

    # --- zero-weight rows vs folded
    g = {"b": 0.0, "P": 0.0}
    for sd in range(seeds):
        k = 3
        x, y, dt, w, prior, p0, q, R = _case(sd, 150, k, True, True, q=3e-4)
        a = KalmanRW(k, p0, q, prior, R)
        for t in range(len(x)):
            a.step(x[t], y[t], dt[t], w[t])
        b = KalmanRW(k, p0, q, prior, R)
        carry = 0.0
        for t in range(len(x)):
            carry += dt[t]
            if w[t] > 0:
                b.step(x[t], y[t], carry, w[t])
                carry = 0.0
        g["b"] = max(g["b"], float(np.max(np.abs(a.b - b.b))))
        g["P"] = max(g["P"], float(np.max(np.abs(a.P - b.P))))
    table("519d zero-weight rows (10%, first row included) vs the same stream with their gaps folded into the next row",
          ["quantity", "largest gap (abs)"], [["b", g["b"]], ["P", g["P"]]])
    print("\n519 state added: 0 bytes (nu, s and the noise estimate are computed by the step; `coef_prior` is "
          "configuration, `8 k_total` bytes per target if the state file carries it, k = 20: 168 bytes).")
