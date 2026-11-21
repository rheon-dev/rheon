"""ARMA(1,1) online: a recursive prediction-error estimator in numpy, one row
at a time, with optional exponential decay (docs/records/ORACLE-STUDY-485.md,
part 4).

The model, zero mean::

    y_t = phi * y_{t-1} + e_t + theta * e_{t-1},   e_t ~ (0, sigma^2)

Row ``t``'s prediction is made before ``y_t`` is read, from the current
parameters and the kept residual::

    yhat_t = phi * y_{t-1} + theta * ehat_{t-1},   e_t = y_t - yhat_t

The gradient of the prediction with respect to ``beta = (phi, theta)`` is
carried by its own recursion at the current parameters (the moving-average
term makes it a filtered regressor, not the raw lagged values)::

    psi_t = (y_{t-1}, ehat_{t-1}) - theta * psi_{t-1}

and each row takes one Gauss-Newton step on the squared one-step error,
weighted by the decay ``lam = 2^(-1/half_life)`` (``lam = 1`` with no decay)::

    W_t    = lam * W_{t-1} + 1,          gamma_t = 1 / W_t
    R_t    = R_{t-1} + gamma_t * (psi_t psi_t' - R_{t-1})
    beta_t = Proj(beta_{t-1} + gamma_t * R_t^{-1} psi_t e_t)
    s2_t   = s2_{t-1} + gamma_t * (e_t^2 - s2_{t-1})

``Proj`` clips ``|phi|`` and ``|theta|`` to ``1 - margin`` (stationary and
invertible). The kept residual is the a-posteriori one,
``ehat_t = y_t - phi_t y_{t-1} - theta_t ehat_{t-1}`` (Ljung and Soderstrom's
choice for stability, (mem)). The start, in two phases of ``warm`` rows and
with nothing buffered: ``y_{-1} = ehat_{-1} = 0`` and ``beta = (0, 0)``, where
the two regressors are the same column (``ehat = y``) and ``R`` is singular, so
the first phase only accumulates the lag-1 autocorrelation ``rho``; then
``beta = (rho, 0)`` (the AR(1) moment estimate) and the second phase
accumulates ``R`` and ``s2`` with ``beta`` held. Steps start after ``2 warm``
rows. Every row emits its prediction at the parameters it holds.

Request A11, remedy (c), a robust score (``robust_c``; 0: off): the step's
error is Huber-clipped in units of the running SD, ``e -> s * clip(e/s, -c, c)``
with ``s = sqrt(s2_{t-1})``, and divided by the running fraction of rows not
clipped (the clipped error's slope, so the step keeps Gauss-Newton's scale).
``s2`` stays the mean of the unclipped ``e^2``. The clip is symmetric, so no
centring is needed under symmetric noise.

State per target: ``beta`` (2), ``R`` (3 distinct), ``psi`` (2), ``y_{t-1}``,
``ehat_{t-1}``, ``s2``, ``W``, the row count and the two sums of the first
phase: 14 floats, 112 bytes.

numpy only; statsmodels and scipy (through statsmodels) are the oracles,
imported only by the comparison functions and the tests.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

TRUTH = (0.6, 0.3, 1.0)  # phi, theta, sigma


def simulate(
    n: int,
    phi: float,
    theta: float,
    sigma: float,
    seed: int,
    burn: int = 1000,
    nu: float | None = None,
):
    """A zero-mean Gaussian ARMA(1,1) series of ``n`` rows after ``burn``
    discarded rows started at zero. With ``nu``, the innovations are Student-t
    with ``nu`` degrees of freedom scaled to standard deviation ``sigma``."""
    rng = np.random.default_rng(seed)
    e = sigma * (
        rng.standard_normal(n + burn)
        if nu is None
        else rng.standard_t(nu, n + burn) * math.sqrt((nu - 2.0) / nu)
    )
    y = np.empty(n + burn)
    prev_y = 0.0
    prev_e = 0.0
    for t in range(n + burn):
        y[t] = phi * prev_y + e[t] + theta * prev_e
        prev_y, prev_e = y[t], e[t]
    return y[burn:]


def residuals(phi: float, theta: float, y: np.ndarray) -> np.ndarray:
    """The fixed-parameter residuals from a zero start, row by row."""
    out = np.empty(y.shape[0])
    prev_y = prev_e = 0.0
    for t in range(y.shape[0]):
        out[t] = y[t] - phi * prev_y - theta * prev_e
        prev_y, prev_e = float(y[t]), float(out[t])
    return out


@dataclass
class OnlineArma:
    """The recursive prediction-error estimator. ``update(y)`` takes one row."""

    half_life: float = math.inf
    warm: int = 50
    margin: float = 1e-2
    robust_c: float = 0.0

    beta: np.ndarray = field(default_factory=lambda: np.zeros(2))
    R: np.ndarray = field(default_factory=lambda: np.zeros((2, 2)))
    psi: np.ndarray = field(default_factory=lambda: np.zeros(2))
    prev_y: float = 0.0
    prev_e: float = 0.0
    s2: float = 0.0
    W: float = 0.0
    n_seen: int = 0
    syy: float = 0.0
    syy1: float = 0.0
    rob_m: float = 0.0
    rob_n: int = 0
    lam: float = field(init=False)

    pred: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.lam = 1.0 if math.isinf(self.half_life) else 2.0 ** (-1.0 / self.half_life)

    def update(self, y: float) -> None:
        phi, theta = self.beta
        x = np.array([self.prev_y, self.prev_e])
        yhat = phi * self.prev_y + theta * self.prev_e
        self.pred.append(yhat)
        e = y - yhat
        self.n_seen += 1
        if self.n_seen <= self.warm:
            # phase 1: the lag-1 autocorrelation at beta = 0
            self.syy += y * y
            self.syy1 += y * self.prev_y
            self.prev_e = y
            self.prev_y = y
            if self.n_seen == self.warm:
                cap = 1.0 - self.margin
                rho = self.syy1 / self.syy if self.syy > 0.0 else 0.0
                self.beta = np.array([min(max(rho, -cap), cap), 0.0])
            return
        psi = x - theta * self.psi
        e_step = e
        if self.robust_c > 0.0 and self.s2 > 0.0:
            sd = math.sqrt(self.s2)
            u = e / sd
            self.rob_n += 1
            self.rob_m += ((1.0 if abs(u) < self.robust_c else 0.0) - self.rob_m) / self.rob_n
            e_step = sd * min(max(u, -self.robust_c), self.robust_c)
            if self.rob_m > 0.0:
                e_step /= self.rob_m
        self.W = self.lam * self.W + 1.0
        gamma = 1.0 / self.W
        self.R += gamma * (np.outer(psi, psi) - self.R)
        self.s2 += gamma * (e * e - self.s2)
        if self.n_seen > 2 * self.warm:
            step = gamma * np.linalg.solve(self.R, psi * e_step)
            cap = 1.0 - self.margin
            self.beta = np.clip(self.beta + step, -cap, cap)
        phi, theta = self.beta
        self.prev_e = y - phi * self.prev_y - theta * self.prev_e
        self.prev_y = y
        self.psi = psi

    @property
    def params(self) -> np.ndarray:
        """``(phi, theta, sigma2)``, as statsmodels orders them."""
        return np.array([self.beta[0], self.beta[1], self.s2])


def run(y: np.ndarray, **kw: object) -> OnlineArma:
    est = OnlineArma(**kw)  # type: ignore[arg-type]
    for v in y:
        est.update(float(v))
    return est


def sarimax_fit(y: np.ndarray):  # noqa: ANN201
    """The oracle: statsmodels' exact-likelihood fit. ``SARIMAX(order=(1,0,1),
    trend="n")`` with its defaults: stationary initialization, stationarity and
    invertibility enforced, L-BFGS, ``cov_type="opg"``."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    model = SARIMAX(y, order=(1, 0, 1), trend="n")
    return model, model.fit(disp=False)


def weighted_css_fit(y: np.ndarray, half_life: float) -> np.ndarray:
    """The decayed oracle: ``(phi, theta)`` minimizing the weighted conditional
    sum of squares ``sum_t lam^(T-1-t) e_t^2`` (zero start), by scipy; then
    ``sigma2`` as the weighted mean of ``e^2``. The definition is written here;
    scipy computes the residuals (``lfilter``) and the minimum."""
    from scipy.optimize import minimize
    from scipy.signal import lfilter

    n = y.shape[0]
    lam = 1.0 if math.isinf(half_life) else 2.0 ** (-1.0 / half_life)
    w = lam ** np.arange(n - 1, -1, -1, dtype=float)

    def resid(b: np.ndarray) -> np.ndarray:
        return lfilter([1.0, -b[0]], [1.0, b[1]], y)

    def loss(b: np.ndarray) -> float:
        e = resid(b)
        return float(np.sum(w * e * e) / np.sum(w))

    res = minimize(loss, x0=np.zeros(2), method="L-BFGS-B", bounds=[(-0.99, 0.99)] * 2)
    e = resid(res.x)
    return np.array([res.x[0], res.x[1], float(np.sum(w * e * e) / np.sum(w))])


NAMES = ("phi", "theta", "sigma2")


def compare(
    n: int,
    seed: int,
    half_life: float = math.inf,
    truth: tuple[float, float, float] = TRUTH,
    nu: float | None = None,
    **kw: object,
) -> dict[str, float]:
    """The online estimator against the batch fits on one simulated series.

    Gaps are in statsmodels' standard errors. Under decay they are also given
    against the weighted fit, in the unweighted SEs scaled by ``sqrt(n/n_eff)``,
    ``n_eff = (sum w)^2 / sum w^2``: an approximation to the weighted fit's own
    SEs, which no library computes. ``truth`` and ``nu`` (Student-t noise)
    default to part 4's Gaussian series."""
    phi, theta, sigma = truth
    y = simulate(n, phi, theta, sigma, seed, nu=nu)
    _, res = sarimax_fit(y)
    b = np.asarray(res.params, dtype=float)
    se = np.asarray(res.bse, dtype=float)
    est = run(y, half_life=half_life, **kw)
    o = est.params
    wfit = weighted_css_fit(y, half_life)
    lam = 1.0 if math.isinf(half_life) else 2.0 ** (-1.0 / half_life)
    w = lam ** np.arange(n - 1, -1, -1, dtype=float)
    n_eff = float(np.sum(w) ** 2 / np.sum(w * w))
    se_w = se * math.sqrt(n / n_eff)
    pred = np.asarray(est.pred)
    tail = slice(n // 2, n)
    # one-step predictions at the batch parameters, zero start, same rows
    e_b = residuals(b[0], b[1], y)
    yhat_b = y - e_b
    out: dict[str, float] = {
        "n": float(n),
        "seed": float(seed),
        "half_life": half_life if math.isfinite(half_life) else -1.0,
        "n_eff": n_eff,
        "sm_converged": float(bool(res.mle_retvals.get("converged", False))),
        "llf_batch": float(res.llf),
        "online_mse_second_half": float(np.mean((y[tail] - pred[tail]) ** 2)),
        "batch_mse_second_half": float(np.mean(e_b[tail] ** 2)),
        "pred_rms_gap_second_half": float(np.sqrt(np.mean((pred[tail] - yhat_b[tail]) ** 2))),
    }
    for i, name in enumerate(NAMES):
        out[f"{name}_batch"] = float(b[i])
        out[f"{name}_online"] = float(o[i])
        out[f"{name}_weighted"] = float(wfit[i])
        out[f"{name}_batch_se"] = float(se[i])
        out[f"{name}_diff_in_se"] = float((o[i] - b[i]) / se[i])
        out[f"{name}_vs_weighted_in_se_w"] = float((o[i] - wfit[i]) / se_w[i])
        out[f"{name}_weighted_vs_batch_in_se"] = float((wfit[i] - b[i]) / se[i])
    return out


if __name__ == "__main__":
    import sys

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 485
    for k, v in compare(n, seed).items():
        print(f"{k:34s} {v: .10g}")
