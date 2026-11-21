"""DCC(1,1) online, two series with GARCH(1,1) margins: the correlation
recursion with ``(a, b)`` learned one row at a time, in numpy; the batch fit
written from the definition and minimized by scipy; the fifty-seed grid
(docs/records/ORACLE-STUDY-485.md, part 6).

    python dcc.py OUT.jsonl            # the grid, resuming after the last case written
    python dcc.py OUT.jsonl --summary  # its table
    python dcc.py --one N SEED         # one case, every field

The model (Engle 2002), on standardized residuals ``z_t = r_t / sigma_t``::

    Q_t = (1 - a - b) * Qbar + a * z_{t-1} z_{t-1}' + b * Q_{t-1}
    R_t = diag(Q_t)^(-1/2) Q_t diag(Q_t)^(-1/2)
    l_t = -1/2 * (ln|R_t| + z_t' R_t^{-1} z_t - z_t' z_t)        (second stage)

With two series ``R_t`` has one free entry, ``rho_t = q12 / sqrt(q11 q22)``,
and the composite likelihood over pairs is the full likelihood.

**Online.** Stage one: each margin is the online GARCH(1,1) of ``garch.py``
with variance targeting (its part-3 variant that passed), and ``z_t`` uses the
margin's variance for row ``t`` computed before row ``t`` was read. Stage two,
per row once both margins emit: ``Qbar`` is the running mean of ``z z'`` over
the rows before this one (online targeting, the scan's form (b)); ``rho_t`` is
emitted before ``z_t`` is read; then one Fisher-scoring step on ``l_t``::

    dQ_t/da = z_{t-1} z_{t-1}' - Qbar + b * dQ_{t-1}/da
    dQ_t/db = Q_{t-1}          - Qbar + b * dQ_{t-1}/db
    g_t     = dl_t/drho * grad rho_t,     I_t = (1 + rho^2)/(1 - rho^2)^2 * grad rho grad rho'
    H_t     = H_{t-1} + (I_t - H_{t-1}) / (t + t0)
    (a, b)  = Proj((a, b) + H_t^{-1} g_t / (t + t0))

``Proj``: ``a, b >= 0``, ``a + b <= 1 - 1e-4`` (Euclidean). The first ``warm``
DCC rows build ``H`` with ``(a, b)`` held at ``(0.03, 0.85)``. State past the
margins (two GARCH states, 128 bytes each): ``Qbar`` and ``Q`` (3 + 3), two
``dQ`` (6), the last ``z`` (2), ``(a, b)``, ``H`` (3), the count: 20 floats.

**Batch** (the oracle): arch's GARCH(1,1) fit per margin (zero mean, normal),
``z`` from its in-sample conditional volatility, ``Qbar`` the sample mean of
``z z'``, ``Q_0 = Qbar``, and ``(a, b)`` maximizing the summed ``l_t`` by
``scipy.optimize.minimize(method="SLSQP")`` under ``a, b >= 0``,
``a + b <= 1 - 1e-4``. The recursion is ``scipy.signal.lfilter``. Standard
errors: the robust sandwich ``H^{-1} J'J H^{-1}`` of the second stage, ``H``
and the per-row scores ``J`` by statsmodels' numerical differentiation. It
ignores the first stage's estimation error, so it is an approximation, not a
library's own standard error: no library here fits DCC (arch 9.0.0 has no
multivariate model).
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import garch  # noqa: E402

TRUTH = (0.05, 0.90)  # a, b
MARGIN = (0.02, 0.08, 0.90)  # omega, alpha, beta of each margin
RHO_BAR = 0.5  # the off-diagonal of Qbar
SEEDS = range(485000, 485050)
SIZES = (2000, 10000)
MARGIN_EPS = 1e-4
NAMES = ("a", "b")


def simulate(n: int, seed: int, burn: int = 1000) -> np.ndarray:
    """``n`` rows of two returns: GARCH(1,1) margins, DCC(1,1) correlation."""
    a, b = TRUTH
    w, al, be = MARGIN
    rng = np.random.default_rng(seed)
    eta = rng.standard_normal((n + burn, 2))
    qbar = np.array([[1.0, RHO_BAR], [RHO_BAR, 1.0]])
    q = qbar.copy()
    s2 = np.full(2, w / (1.0 - al - be))
    prev_r2 = s2.copy()
    prev_z = np.zeros(2)
    out = np.empty((n + burn, 2))
    for t in range(n + burn):
        if t > 0:
            q = (1.0 - a - b) * qbar + a * np.outer(prev_z, prev_z) + b * q
        rho = q[0, 1] / math.sqrt(q[0, 0] * q[1, 1])
        z = np.array([eta[t, 0], rho * eta[t, 0] + math.sqrt(1.0 - rho * rho) * eta[t, 1]])
        s2 = w + al * prev_r2 + be * s2
        r = np.sqrt(s2) * z
        out[t] = r
        prev_r2 = r * r
        prev_z = z
    return out[burn:]


def rho_path(a: float, b: float, z: np.ndarray, qbar: np.ndarray) -> np.ndarray:
    """``rho_t`` for every row, ``Q_0 = Qbar``, by ``lfilter`` on each entry."""
    from scipy.signal import lfilter

    n = z.shape[0]
    out = {}
    for i, j in ((0, 0), (1, 1), (0, 1)):
        x = z[:, i] * z[:, j]
        u = np.empty(n)
        u[0] = qbar[i, j]
        u[1:] = (1.0 - a - b) * qbar[i, j] + a * x[:-1]
        out[(i, j)] = lfilter([1.0], [1.0, -b], u)
    return out[(0, 1)] / np.sqrt(out[(0, 0)] * out[(1, 1)])


def loglik_rows(rho: np.ndarray, z: np.ndarray) -> np.ndarray:
    """The second-stage log-likelihood of each row, two series."""
    d = 1.0 - rho * rho
    s = z[:, 0] ** 2 + z[:, 1] ** 2
    p = z[:, 0] * z[:, 1]
    return -0.5 * (np.log(d) + (s - 2.0 * rho * p) / d - s)


def batch_fit(r: np.ndarray) -> dict[str, object]:
    """The oracle (module docstring)."""
    from scipy.optimize import minimize
    from statsmodels.tools.numdiff import approx_fprime, approx_hess

    z = np.empty_like(r)
    for i in range(2):
        _, res = garch.arch_fit(r[:, i])
        z[:, i] = r[:, i] / np.asarray(res.conditional_volatility)
    qbar = z.T @ z / z.shape[0]

    def rows(theta: np.ndarray) -> np.ndarray:
        return loglik_rows(rho_path(theta[0], theta[1], z, qbar), z)

    def neg(theta: np.ndarray) -> float:
        return -float(np.sum(rows(theta)))

    fit = minimize(
        neg,
        x0=np.array([0.03, 0.85]),
        method="SLSQP",
        bounds=[(0.0, 1.0), (0.0, 1.0)],
        constraints=[{"type": "ineq", "fun": lambda t: 1.0 - MARGIN_EPS - t[0] - t[1]}],
        options={"ftol": 1e-12, "maxiter": 500},
    )
    theta = fit.x
    hess = approx_hess(theta, lambda t: float(np.sum(rows(t))))
    scores = approx_fprime(theta, rows)
    hinv = np.linalg.inv(-hess)
    cov = hinv @ (scores.T @ scores) @ hinv
    return {
        "theta": theta,
        "se": np.sqrt(np.diag(cov)),
        "se_classic": np.sqrt(np.diag(hinv)),
        "success": bool(fit.success),
        "z": z,
        "qbar": qbar,
        "llf": -float(fit.fun),
    }


def project(a: float, b: float) -> tuple[float, float]:
    a, b = max(a, 0.0), max(b, 0.0)
    cap = 1.0 - MARGIN_EPS
    if a + b > cap:
        shift = 0.5 * (a + b - cap)
        a, b = a - shift, b - shift
        if a < 0.0:
            a, b = 0.0, cap
        elif b < 0.0:
            a, b = cap, 0.0
    return a, b


@dataclass
class OnlineDcc:
    """Stage two, online. ``update(z)`` takes one row of standardized residuals."""

    a0: float = 0.03
    b0: float = 0.85
    warm: int = 100
    t0: float = 100.0

    theta: np.ndarray = field(init=False)
    qbar: np.ndarray = field(default_factory=lambda: np.zeros((2, 2)))
    q: np.ndarray = field(default_factory=lambda: np.zeros((2, 2)))
    dqa: np.ndarray = field(default_factory=lambda: np.zeros((2, 2)))
    dqb: np.ndarray = field(default_factory=lambda: np.zeros((2, 2)))
    prev_z: np.ndarray = field(default_factory=lambda: np.zeros(2))
    H: np.ndarray = field(default_factory=lambda: np.zeros((2, 2)))
    n_seen: int = 0
    n_steps: int = 0
    pred: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.theta = np.array([self.a0, self.b0])

    def update(self, z: np.ndarray) -> None:
        a, b = self.theta
        if self.n_seen == 0:
            # no earlier row: Q starts at the identity, its derivatives at zero
            self.q = np.eye(2)
        else:
            zz = np.outer(self.prev_z, self.prev_z)
            q_prev = self.q
            self.q = (1.0 - a - b) * self.qbar + a * zz + b * q_prev
            self.dqa = zz - self.qbar + b * self.dqa
            self.dqb = q_prev - self.qbar + b * self.dqb
        q11, q22, q12 = self.q[0, 0], self.q[1, 1], self.q[0, 1]
        s = math.sqrt(q11 * q22)
        rho = q12 / s
        self.pred.append(rho)
        # the row's score and information
        d = 1.0 - rho * rho
        z1, z2 = float(z[0]), float(z[1])
        p, ss = z1 * z2, z1 * z1 + z2 * z2
        dl = rho / d + (p * d - rho * (ss - 2.0 * rho * p)) / (d * d)
        grad = np.array(
            [
                dq[0, 1] / s - 0.5 * rho * (dq[0, 0] / q11 + dq[1, 1] / q22)
                for dq in (self.dqa, self.dqb)
            ]
        )
        info = (1.0 + rho * rho) / (d * d) * np.outer(grad, grad)
        self.n_seen += 1
        if self.n_seen <= self.warm:
            self.H += (info - self.H) / self.n_seen
        else:
            self.n_steps += 1
            gamma = 1.0 / (self.n_steps + self.t0)
            self.H += gamma * (info - self.H)
            step = gamma * np.linalg.solve(self.H, dl * grad)
            self.theta = np.array(project(*(self.theta + step)))
        self.qbar += (np.outer(z, z) - self.qbar) / self.n_seen
        self.prev_z = np.asarray(z, dtype=float).copy()


def online(r: np.ndarray, **kw: object) -> tuple[OnlineDcc, np.ndarray, np.ndarray]:
    """Both stages online. Returns the DCC state, the rows it saw (those where
    both margins emit) and the standardized residuals it was fed."""
    s2 = np.column_stack([np.asarray(garch.run(r[:, i], target="running").pred) for i in range(2)])
    rows = np.flatnonzero(np.all(np.isfinite(s2), axis=1))
    z = r[rows] / np.sqrt(s2[rows])
    est = OnlineDcc(**kw)  # type: ignore[arg-type]
    for row in z:
        est.update(row)
    return est, rows, z


def compare(n: int, seed: int) -> dict[str, float]:
    r = simulate(n, seed)
    bf = batch_fit(r)
    th_b, se = bf["theta"], bf["se"]
    est, rows, _ = online(r)
    th_o = est.theta
    z_b, qbar_b = bf["z"], bf["qbar"]
    rho_b = rho_path(th_b[0], th_b[1], z_b, qbar_b)
    rho_bo = rho_path(th_o[0], th_o[1], z_b, qbar_b)
    pred = np.full(n, np.nan)
    pred[rows] = est.pred
    tail = slice(n // 2, n)
    llf_o = float(np.sum(loglik_rows(rho_bo, z_b)))
    out: dict[str, float] = {
        "n": float(n),
        "seed": float(seed),
        "batch_success": float(bf["success"]),
        "llf_batch": bf["llf"],
        "llf_diff_per_row": (bf["llf"] - llf_o) / n,
        "rho_gap_median_second_half": float(np.median(np.abs(pred[tail] - rho_b[tail]))),
        "rho_gap_p90_second_half": float(np.percentile(np.abs(pred[tail] - rho_b[tail]), 90)),
        "first_dcc_row": float(rows[0]),
    }
    for i, name in enumerate(NAMES):
        out[f"{name}_batch"] = float(th_b[i])
        out[f"{name}_online"] = float(th_o[i])
        out[f"{name}_batch_se"] = float(se[i])
        out[f"{name}_batch_se_classic"] = float(bf["se_classic"][i])
        out[f"{name}_diff_in_se"] = float((th_o[i] - th_b[i]) / se[i])
    return out


# ---------------------------------------------------------------------------
# k series (request A10, part 4). The two-series code above is unchanged;
# this path takes the truth, k and the noise as options. With k = 2 its step
# is the one above (the information below reduces to (1 + rho^2)/(1 - rho^2)^2
# grad grad'), up to rounding.
#
# The second stage, k series, the full Gaussian likelihood (Engle 2002):
#
#     l_t   = -1/2 * (ln|R_t| + z_t' R_t^{-1} z_t - z_t' z_t)
#     dR    = D^{-1/2} dQ D^{-1/2} - 1/2 (L R + R L),   D = diag(Q), L = diag(dQ_ii / Q_ii)
#     dl    = 1/2 tr((R^{-1} z z' R^{-1} - R^{-1}) dR)
#     I_ij  = 1/2 tr(R^{-1} dR_i R^{-1} dR_j)                  (its information)
#
# Noise: with ``nu`` the innovation is multivariate Student-t, ``eta = L_t x
# sqrt((nu - 2)/w)``, ``x`` standard normal, ``w ~ chi2(nu)``, ``L_t`` the
# Cholesky factor of ``R_t``: correlation ``R_t``, each margin t(nu) at unit
# variance.


def simulate_k(
    n: int,
    seed: int,
    k: int,
    ab: tuple[float, float],
    rho_bar: float,
    margin: tuple[float, float, float] = MARGIN,
    nu: float | None = None,
    burn: int = 1000,
) -> np.ndarray:
    """``n`` rows of ``k`` returns: GARCH(1,1) margins, DCC(1,1) correlation,
    ``Qbar`` with every off-diagonal ``rho_bar``."""
    a, b = ab
    w, al, be = margin
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n + burn, k))
    scale = np.ones(n + burn) if nu is None else np.sqrt((nu - 2.0) / rng.chisquare(nu, n + burn))
    qbar = np.full((k, k), rho_bar)
    np.fill_diagonal(qbar, 1.0)
    q = qbar.copy()
    s2 = np.full(k, w / (1.0 - al - be))
    prev_r2 = s2.copy()
    prev_z = np.zeros(k)
    out = np.empty((n + burn, k))
    for t in range(n + burn):
        if t > 0:
            q = (1.0 - a - b) * qbar + a * np.outer(prev_z, prev_z) + b * q
        d = 1.0 / np.sqrt(np.diag(q))
        rmat = q * np.outer(d, d)
        z = np.linalg.cholesky(rmat) @ x[t] * scale[t]
        s2 = w + al * prev_r2 + be * s2
        r = np.sqrt(s2) * z
        out[t] = r
        prev_r2 = r * r
        prev_z = z
    return out[burn:]


def chi2_cdf(x: float, k: int) -> float:
    """The chi-squared CDF, ``k`` degrees of freedom: the regularized lower
    incomplete gamma ``P(k/2, x/2)`` by its series (numpy-free; the test holds
    it to scipy's)."""
    a, h = 0.5 * k, 0.5 * x
    term = total = 1.0 / a
    j = 0
    while term > 1e-17 * total:
        j += 1
        term *= h / (a + j)
        total += term
    return total * math.exp(-h + a * math.log(h) - math.lgamma(a))


def huber_radius(c: float, k: int) -> float:
    """The squared Mahalanobis radius ``q`` with ``P(chi2_k > q)`` equal to
    ``P(|z| > c)`` for one standard normal: the univariate clip at ``c``
    carried to ``k`` dimensions at the same tail probability (``k = 1``:
    ``q = c^2``)."""
    p = math.erf(c / math.sqrt(2.0))
    lo, hi = 0.0, 4.0 * c * c * k + 100.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if chi2_cdf(mid, k) < p else (lo, mid)
    return 0.5 * (lo + hi)


# request A11, remedy (c): the robust score, both stages or the second only
REMEDIES: dict[str, dict[str, dict[str, object]]] = {
    "huber2": {"stage2": {"robust_c": 2.0}, "margins": {"robust_c": 2.0}},
    "huber2_s2": {"stage2": {"robust_c": 2.0}, "margins": {}},
}


def r_path_k(a: float, b: float, z: np.ndarray, qbar: np.ndarray) -> np.ndarray:
    """``R_t`` for every row, ``(n, k, k)``, ``Q_0 = Qbar``, by ``lfilter``."""
    from scipy.signal import lfilter

    n, k = z.shape
    zz = z[:, :, None] * z[:, None, :]
    u = np.empty((n, k, k))
    u[0] = qbar
    u[1:] = (1.0 - a - b) * qbar + a * zz[:-1]
    q = lfilter([1.0], [1.0, -b], u, axis=0)
    d = 1.0 / np.sqrt(np.einsum("tii->ti", q))
    return q * d[:, :, None] * d[:, None, :]


def loglik_rows_k(rmat: np.ndarray, z: np.ndarray) -> np.ndarray:
    """The second-stage log-likelihood of each row, ``k`` series."""
    _, logdet = np.linalg.slogdet(rmat)
    quad = np.einsum("ti,ti->t", z, np.linalg.solve(rmat, z[:, :, None])[:, :, 0])
    return -0.5 * (logdet + quad - np.einsum("ti,ti->t", z, z))


INFEASIBLE = 1e12


def batch_fit_k(r: np.ndarray) -> dict[str, object]:
    """The oracle, ``k`` series: ``batch_fit``'s, with the full likelihood."""
    from scipy.optimize import minimize
    from statsmodels.tools.numdiff import approx_fprime, approx_hess

    k = r.shape[1]
    z = np.empty_like(r)
    for i in range(k):
        _, res = garch.arch_fit(r[:, i])
        z[:, i] = r[:, i] / np.asarray(res.conditional_volatility)
    qbar = z.T @ z / z.shape[0]

    def rows(theta: np.ndarray) -> np.ndarray:
        return loglik_rows_k(r_path_k(theta[0], theta[1], z, qbar), z)

    def neg(theta: np.ndarray) -> float:
        # SLSQP's line search and finite differences can step outside the
        # admissible set, where Q loses its positive diagonal; a large finite
        # value there (never at the optimum, which is admissible) keeps it going
        a, b = float(theta[0]), float(theta[1])
        if a < 0.0 or b < 0.0 or a + b >= 1.0:
            return INFEASIBLE
        v = -float(np.sum(rows(theta)))
        return v if math.isfinite(v) else INFEASIBLE

    fit = minimize(
        neg,
        x0=np.array([0.03, 0.85]),
        method="SLSQP",
        bounds=[(0.0, 1.0), (0.0, 1.0)],
        constraints=[{"type": "ineq", "fun": lambda t: 1.0 - MARGIN_EPS - t[0] - t[1]}],
        options={"ftol": 1e-12, "maxiter": 500},
    )
    theta = fit.x
    hess = approx_hess(theta, lambda t: float(np.sum(rows(t))))
    scores = approx_fprime(theta, rows)
    hinv = np.linalg.inv(-hess)
    cov = hinv @ (scores.T @ scores) @ hinv
    return {
        "theta": theta,
        "se": np.sqrt(np.diag(cov)),
        "se_classic": np.sqrt(np.diag(hinv)),
        "success": bool(fit.success),
        "nit": int(fit.nit),
        "z": z,
        "qbar": qbar,
        "llf": -float(fit.fun),
    }


@dataclass
class OnlineDccK:
    """Stage two online, ``k`` series. ``update(z)`` takes one row.

    ``robust_c`` (request A11, remedy (c); 0: off): the score's ``u u'``,
    ``u = R^{-1} z``, is Huber-weighted by ``w = min(1, q / s)``,
    ``s = z' R^{-1} z`` the squared Mahalanobis distance and ``q =
    huber_radius(robust_c, k)``, and ``R^{-1}`` is scaled by ``kappa``, the
    running mean of ``min(s, q) / k``, so the score is mean zero at the truth
    for any elliptical noise. The information matrix is not rescaled by the
    clipped score's slope (unlike garch.py and arma.py, where it is a
    univariate mean; here it was not derived)."""

    k: int = 3
    a0: float = 0.03
    b0: float = 0.85
    warm: int = 100
    t0: float = 100.0
    robust_c: float = 0.0
    rob_q: float = field(init=False)
    rob_kappa: float = 0.0

    theta: np.ndarray = field(init=False)
    qbar: np.ndarray = field(init=False)
    q: np.ndarray = field(init=False)
    dqa: np.ndarray = field(init=False)
    dqb: np.ndarray = field(init=False)
    prev_z: np.ndarray = field(init=False)
    H: np.ndarray = field(default_factory=lambda: np.zeros((2, 2)))
    n_seen: int = 0
    n_steps: int = 0
    n_projected: int = 0
    pred: list[np.ndarray] = field(default_factory=list)

    def __post_init__(self) -> None:
        k = self.k
        self.theta = np.array([self.a0, self.b0])
        self.qbar = np.zeros((k, k))
        self.q = np.zeros((k, k))
        self.dqa = np.zeros((k, k))
        self.dqb = np.zeros((k, k))
        self.prev_z = np.zeros(k)
        self.rob_q = huber_radius(self.robust_c, k) if self.robust_c > 0.0 else math.inf

    def update(self, z: np.ndarray) -> None:
        a, b = self.theta
        z = np.asarray(z, dtype=float)
        if self.n_seen == 0:
            self.q = np.eye(self.k)
        else:
            zz = np.outer(self.prev_z, self.prev_z)
            q_prev = self.q
            self.q = (1.0 - a - b) * self.qbar + a * zz + b * q_prev
            self.dqa = zz - self.qbar + b * self.dqa
            self.dqb = q_prev - self.qbar + b * self.dqb
        dg = np.diag(self.q)
        d = 1.0 / np.sqrt(dg)
        rmat = self.q * np.outer(d, d)
        self.pred.append(rmat[np.triu_indices(self.k, 1)])
        rinv = np.linalg.inv(rmat)
        u = rinv @ z
        if self.robust_c > 0.0:
            s = float(z @ u)
            self.rob_kappa += (min(s, self.rob_q) / self.k - self.rob_kappa) / (self.n_seen + 1)
            w = 1.0 if s <= self.rob_q else self.rob_q / s
            m = w * np.outer(u, u) - self.rob_kappa * rinv
        else:
            m = np.outer(u, u) - rinv
        drs = []
        for dq in (self.dqa, self.dqb):
            lam = np.diag(dq) / dg
            drs.append(dq * np.outer(d, d) - 0.5 * (lam[:, None] * rmat + rmat * lam[None, :]))
        g = np.array([0.5 * float(np.sum(m * dr)) for dr in drs])
        prods = [rinv @ dr for dr in drs]
        info = np.array([[0.5 * float(np.sum(pi * pj.T)) for pj in prods] for pi in prods])
        self.last_g, self.last_info = g, info
        self.n_seen += 1
        if self.n_seen <= self.warm:
            self.H += (info - self.H) / self.n_seen
        else:
            self.n_steps += 1
            gamma = 1.0 / (self.n_steps + self.t0)
            self.H += gamma * (info - self.H)
            proposal = self.theta + gamma * np.linalg.solve(self.H, g)
            projected = np.array(project(*proposal))
            self.n_projected += int(not np.array_equal(projected, proposal))
            self.theta = projected
        self.qbar += (np.outer(z, z) - self.qbar) / self.n_seen
        self.prev_z = z.copy()


def online_k(
    r: np.ndarray, margins: dict[str, object] | None = None, **kw: object
) -> tuple[OnlineDccK, np.ndarray, np.ndarray]:
    """Both stages online, ``k`` series (``online``'s, with ``OnlineDccK``);
    ``margins`` adds options to each margin's ``garch.OnlineGarch``."""
    k = r.shape[1]
    mkw = {"target": "running", **(margins or {})}
    s2 = np.column_stack([np.asarray(garch.run(r[:, i], **mkw).pred) for i in range(k)])
    rows = np.flatnonzero(np.all(np.isfinite(s2), axis=1))
    z = r[rows] / np.sqrt(s2[rows])
    est = OnlineDccK(k=k, **kw)  # type: ignore[arg-type]
    for row in z:
        est.update(row)
    return est, rows, z


def compare_k(
    n: int,
    seed: int,
    k: int,
    ab: tuple[float, float],
    rho_bar: float,
    nu: float | None,
    remedy: str = "",
) -> dict[str, float]:
    r = simulate_k(n, seed, k, ab, rho_bar, nu=nu)
    bf = batch_fit_k(r)
    th_b, se = bf["theta"], bf["se"]
    rem = REMEDIES[remedy] if remedy else {"stage2": {}, "margins": {}}
    est, rows, _ = online_k(r, rem["margins"], **rem["stage2"])
    th_o = est.theta
    z_b, qbar_b = bf["z"], bf["qbar"]
    iu = np.triu_indices(k, 1)
    rho_b = r_path_k(th_b[0], th_b[1], z_b, qbar_b)[:, iu[0], iu[1]]
    llf_o = float(np.sum(loglik_rows_k(r_path_k(th_o[0], th_o[1], z_b, qbar_b), z_b)))
    pred = np.full((n, len(iu[0])), np.nan)
    pred[rows] = np.asarray(est.pred)
    tail = slice(n // 2, n)
    gap = np.abs(pred[tail] - rho_b[tail])
    per_pair = np.median(gap, axis=0)
    cap = 1.0 - MARGIN_EPS
    out: dict[str, float] = {
        "n": float(n),
        "seed": float(seed),
        "k": float(k),
        "nu": nu or -1.0,
        "a_true": ab[0],
        "b_true": ab[1],
        "rho_bar": rho_bar,
        "batch_success": float(bf["success"]),
        "batch_nit": float(bf["nit"]),
        "batch_at_sum_cap": float(th_b[0] + th_b[1] >= cap - 1e-6),
        "batch_at_zero": float(min(th_b) <= 1e-6),
        "online_at_constraint": float(th_o[0] + th_o[1] >= cap - 1e-9 or min(th_o) <= 0.0),
        "online_steps_projected": float(est.n_projected),
        "llf_batch": bf["llf"],
        "llf_diff_per_row": (bf["llf"] - llf_o) / n,
        "rho_gap_median_second_half": float(np.median(gap)),
        "rho_gap_worst_pair_median": float(per_pair.max()),
        "rho_gap_p90_second_half": float(np.percentile(gap, 90)),
        "first_dcc_row": float(rows[0]),
    }
    if remedy:
        out["remedy"] = remedy  # type: ignore[assignment]
    for i, name in enumerate(NAMES):
        out[f"{name}_batch"] = float(th_b[i])
        out[f"{name}_online"] = float(th_o[i])
        out[f"{name}_batch_se"] = float(se[i])
        out[f"{name}_batch_se_classic"] = float(bf["se_classic"][i])
        out[f"{name}_diff_in_se"] = float((th_o[i] - th_b[i]) / se[i])
    return out


def grid_k(
    out: Path,
    n: int,
    k: int,
    ab: tuple[float, float],
    rho_bar: float,
    nu: float | None,
    shard: tuple[int, int] = (0, 1),
    remedy: str = "",
) -> None:
    finished = {(int(r["seed"]), int(r["n"])) for r in load(out)}
    seeds = [s for i, s in enumerate(SEEDS) if i % shard[1] == shard[0]]
    with out.open("a", encoding="utf-8") as f:
        for seed in seeds:
            if (seed, n) in finished:
                continue
            f.write(json.dumps(compare_k(n, seed, k, ab, rho_bar, nu, remedy)) + "\n")
            f.flush()
            print(f"dcc k {k}: n {n} seed {seed} done", flush=True)


def summary_k(outs: list[Path]) -> None:
    """The criterion, the worst seed, the correlation gap and the section 15.2
    bounds (45 of 50 within 2 SE; every seed within 4 SE; per-seed median
    correlation gap at most 0.03 over the second half)."""
    recs = [r for out in outs for r in load(out)]
    for n in sorted({int(r["n"]) for r in recs}):
        rows = [r for r in recs if int(r["n"]) == n]
        ok = sum(r["batch_success"] == 1 for r in rows)
        print(f"rows {n}: {len(rows)} seeds; scipy success {ok}")
        for name in NAMES:
            v = np.abs([r[f"{name}_diff_in_se"] for r in rows])
            med, p90 = np.median(v), np.quantile(v, 0.9)
            print(f"  {name} in SE: median {med:.3g}  p90 {p90:.3g}  max {v.max():.3g}")
        within2 = sum(all(abs(r[f"{p}_diff_in_se"]) <= 2 for p in NAMES) for r in rows)
        within4 = sum(all(abs(r[f"{p}_diff_in_se"]) <= 4 for p in NAMES) for r in rows)
        g = np.array([r["rho_gap_median_second_half"] for r in rows])
        gw = np.array([r["rho_gap_worst_pair_median"] for r in rows])
        worst = max(rows, key=lambda r: max(abs(r[f"{p}_diff_in_se"]) for p in NAMES))
        print(f"  within 2 SE {within2}; within 4 SE {within4}")
        print(
            f"  rho gap (per-seed median, all pairs): median {np.median(g):.3g} max {g.max():.3g}"
        )
        print(f"  rho gap (per-seed worst pair): median {np.median(gw):.3g} max {gw.max():.3g}")
        ng, ngw = int((g <= 0.03).sum()), int((gw <= 0.03).sum())
        print(f"  seeds with rho gap <= 0.03: {ng} (worst pair: {ngw})")
        both = sum(
            all(abs(r[f"{p}_diff_in_se"]) <= 4 for p in NAMES)
            and r["rho_gap_worst_pair_median"] <= 0.03
            for r in rows
        )
        print(f"  every per-seed bound (4 SE and worst-pair gap 0.03): {both}")
        print(
            f"  worst seed {int(worst['seed'])}: a {worst['a_diff_in_se']:+.3g} SE, "
            f"b {worst['b_diff_in_se']:+.3g} SE; "
            f"batch ({worst['a_batch']:.4f}, {worst['b_batch']:.4f}) "
            f"online ({worst['a_online']:.4f}, {worst['b_online']:.4f})"
        )
        for key in ("batch_at_sum_cap", "batch_at_zero", "online_at_constraint"):
            hit = [int(r["seed"]) for r in rows if r[key]]
            print(f"  {key}: {hit}")
        proj = np.array([r["online_steps_projected"] for r in rows])
        nproj = int((proj > 0).sum())
        print(f"  online steps projected: seeds with any {nproj}, max {proj.max():.0f}")
        llf = np.array([r["llf_diff_per_row"] for r in rows])
        ratio = np.median([r["a_batch_se"] / r["a_batch_se_classic"] for r in rows])
        print(
            f"  llf gap per row median {np.median(llf):.3g} max {llf.max():.3g} min {llf.min():.3g}"
        )
        print(f"  robust/classic SE of a, median {ratio:.3f}")
        verdict = "passes" if within2 >= 0.9 * len(rows) else "fails"
        print(f"  criterion: {verdict}")


def load(out: Path) -> list[dict]:
    if not out.exists():
        return []
    return [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines() if x.strip()]


def grid(out: Path) -> None:
    finished = {(int(r["seed"]), int(r["n"])) for r in load(out)}
    with out.open("a", encoding="utf-8") as f:
        for n in SIZES:
            for seed in SEEDS:
                if (seed, n) in finished:
                    continue
                f.write(json.dumps(compare(n, seed)) + "\n")
                f.flush()
                print(f"dcc: n {n} seed {seed} done", flush=True)


def summary(out: Path) -> None:
    recs = load(out)
    print("| rows | a: median / max / over 3 | b | all within 2 SE | rho gap: median / max |")
    print("|---|---|---|---|---|")
    for n in SIZES:
        rows = [r for r in recs if int(r["n"]) == n]
        cells = []
        for name in NAMES:
            a = np.abs([r[f"{name}_diff_in_se"] for r in rows])
            cells.append(f"{np.median(a):.3g} / {a.max():.3g} / {int((a > 3).sum())}")
        within2 = sum(all(abs(r[f"{k}_diff_in_se"]) <= 2 for k in NAMES) for r in rows)
        g = np.array([r["rho_gap_median_second_half"] for r in rows])
        print(
            f"| {n:,} | "
            + " | ".join(cells)
            + f" | {within2} | {np.median(g):.3g} / {g.max():.3g} |"
        )
        llf = np.array([r["llf_diff_per_row"] for r in rows])
        ok = sum(r["batch_success"] == 1 for r in rows)
        ratio = np.median([r["a_batch_se"] / r["a_batch_se_classic"] for r in rows])
        print(
            f"<!-- n {n}: seeds {len(rows)}, scipy success {ok}; llf gap per row median "
            f"{np.median(llf):.3g} max {llf.max():.3g} min {llf.min():.3g}; "
            f"robust/classic SE of a, median {ratio:.3f} -->"
        )


def _opt(name: str, default: str | None = None) -> str | None:
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


if __name__ == "__main__":
    if "--k" in sys.argv:
        # the k-series path (request A10, part 4):
        #   dcc.py OUT.jsonl --k 3 [--a A --b B --rho-bar R --nu NU --rows N --shard I/K]
        #     [--remedy NAME]   (request A11: an entry of REMEDIES)
        #   dcc.py OUT1.jsonl [OUT2.jsonl ...] --k 3 --summary
        if "--summary" in sys.argv:
            outs = []
            for x in sys.argv[1:]:
                if x.startswith("--"):
                    break
                outs.append(Path(x))
            summary_k(outs)
        else:
            nu_s = _opt("--nu")
            si, sk = (int(x) for x in str(_opt("--shard", "0/1")).split("/"))
            grid_k(
                Path(sys.argv[1]),
                int(str(_opt("--rows", "10000"))),
                int(str(_opt("--k"))),
                (float(str(_opt("--a", "0.05"))), float(str(_opt("--b", "0.93")))),
                float(str(_opt("--rho-bar", "0.5"))),
                None if nu_s is None else float(nu_s),
                (si, sk),
                str(_opt("--remedy", "")),
            )
    elif sys.argv[1] == "--one":
        for k, v in compare(int(sys.argv[2]), int(sys.argv[3])).items():
            print(f"{k:34s} {v: .10g}")
    elif "--summary" in sys.argv[2:]:
        summary(Path(sys.argv[1]))
    else:
        grid(Path(sys.argv[1]))
