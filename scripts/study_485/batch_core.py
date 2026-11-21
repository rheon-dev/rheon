"""A batch optimizer the core can hold: projected Newton with backtracking
on a bounded buffer, in numpy (docs/records/ORACLE-STUDY-485.md, request A12
part 1). It replaces the library fits (arch, statsmodels, scipy's SLSQP) that
stood in for the batch start of remedy (a).

**The method.** Each objective ``f(theta)`` over the buffered rows is a sum of
per-row terms whose gradient and Gauss-Newton (Fisher) matrix come from the
same sensitivity recursion the online estimator carries, run once over the
buffer at fixed parameters::

    f, g, H = objective(theta)            # value, gradient, Gauss-Newton matrix
    M       = -(central difference of the analytic g, step 1e-5 max(|theta_j|, 0.01)),
              symmetrized: the observed curvature; if it is not positive definite,
              or a probe is not finite, M = H (Gauss-Newton)
    d       = (M + 1e-10 tr(M)/p I)^{-1} g    # ascent direction (f is maximized)
    theta'  = Proj(theta + s d),  s = 1, 1/2, 1/4, ...  (at most 30 halvings)
    accept the first s with f(theta') >= f + 1e-4 g'(theta' - theta)   (Armijo)

``Proj`` is the model's own admissible region (the online projection).
When no step length passes along ``d``, two more directions are tried in
turn before stopping: the reduced Newton step on the coordinates the
projection does not block (a coordinate is blocked when a small move along
``d`` in it alone is cut by ``Proj`` to less than half), then the gradient
scaled by ``1 / diag(M)``. At a bound (``alpha = 0`` on a GARCH buffer under
t(5)) the full direction points out of the region and its projection is no
ascent; the reduced one is.
Gauss-Newton alone (``M = H``) was tried first: under heavy-tailed data the
expected information is mis-scaled against the likelihood's curvature, and on
a DCC buffer under t(5) it zigzagged along the ``a + b`` ridge at a linear
rate of about 0.86 and met the iteration cap; the observed curvature, from
``2p`` extra gradient passes per iteration, converges in a few steps.

**Start.** The best point of a small fixed grid by ``f`` (``GRIDS``), so the
first step is not taken from a far corner; arch does the same for its fits.

**Stopping rule.** Stop after the iteration that moves every parameter by at
most ``1e-8`` (absolute) and ``f`` by at most ``1e-10 (1 + |f|)``; or when no
step length passes the Armijo test (the projected direction no longer
increases ``f``); or at ``MAX_ITER = 50`` iterations. ``Fit`` records which.

Costs: each iteration is one pass over the buffer for ``(f, g, H)``, ``2p``
passes for the curvature (``p = 2``) and one pass per step length tried;
state is the buffer (``W`` rows) and a handful of floats. numpy only: no
library optimizer, no library model.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

MAX_ITER = 50
MAX_HALVINGS = 30
ARMIJO = 1e-4
TOL_X = 1e-8
TOL_F = 1e-10
RIDGE = 1e-10
LN_2PI = math.log(2.0 * math.pi)

Objective = Callable[[np.ndarray, bool], tuple[float, np.ndarray, np.ndarray]]


@dataclass
class Fit:
    theta: np.ndarray
    value: float
    iterations: int
    stop: str  # "converged", "no ascent", "iteration cap"
    evaluations: int
    newton_steps: int = 0


def maximize(
    objective: Objective,
    project: Callable[[np.ndarray], np.ndarray],
    grid: list[tuple[float, float]],
) -> Fit:
    """Projected Newton with Armijo backtracking (module docstring)."""
    evals = 0
    newton = 0
    best, best_f = None, -math.inf
    for point in grid:
        th = project(np.asarray(point, dtype=float))
        f, _, _ = objective(th, False)
        evals += 1
        if math.isfinite(f) and f > best_f:
            best, best_f = th, f
    assert best is not None, "no admissible finite start"
    theta = best
    f, g, H = objective(theta, True)
    evals += 1
    stop = "iteration cap"
    it = 0
    while it < MAX_ITER:
        it += 1
        p = H.shape[0]
        M = _curvature(objective, theta, p)
        evals += 2 * p
        if M is None:
            M = H
        else:
            newton += 1
        ridge = RIDGE * max(float(np.trace(M)) / p, 1e-300)
        d = np.linalg.solve(M + ridge * np.eye(p), g)
        accepted = None
        for direction in _directions(d, g, M, ridge, theta, project):
            s = 1.0
            for _ in range(MAX_HALVINGS + 1):
                cand = project(theta + s * direction)
                fc, _, _ = objective(cand, False)
                evals += 1
                if math.isfinite(fc) and fc >= f + ARMIJO * float(g @ (cand - theta)):
                    accepted = (cand, fc)
                    break
                s *= 0.5
            if accepted is not None:
                break
        if accepted is None:
            stop = "no ascent"
            break
        cand, fc = accepted
        moved = float(np.max(np.abs(cand - theta)))
        df = abs(fc - f)
        theta = cand
        f, g, H = objective(theta, True)
        evals += 1
        if moved <= TOL_X and df <= TOL_F * (1.0 + abs(f)):
            stop = "converged"
            break
    return Fit(
        theta=theta, value=f, iterations=it, stop=stop, evaluations=evals, newton_steps=newton
    )


def _directions(
    d: np.ndarray,
    g: np.ndarray,
    M: np.ndarray,
    ridge: float,
    theta: np.ndarray,
    project: Callable[[np.ndarray], np.ndarray],
):  # noqa: ANN202
    """The directions tried in turn: Newton, reduced Newton on the coordinates
    the projection does not block, the diagonally scaled gradient."""
    yield d
    p = d.shape[0]
    free = []
    for j in range(p):
        e = np.zeros(p)
        e[j] = 1e-6 * d[j]
        moved = abs(float(project(theta + e)[j] - theta[j]))
        free.append(moved >= 0.5 * abs(e[j]) and d[j] != 0.0)
    f_idx = [j for j in range(p) if free[j]]
    if f_idx and len(f_idx) < p:
        red = np.zeros(p)
        sub = M[np.ix_(f_idx, f_idx)] + ridge * np.eye(len(f_idx))
        red[f_idx] = np.linalg.solve(sub, g[f_idx])
        yield red
    yield g / np.maximum(np.diag(M), 1e-300)


def _curvature(objective: Objective, theta: np.ndarray, p: int) -> np.ndarray | None:
    """Minus the Jacobian of the analytic gradient by central differences,
    symmetrized; None when a probe is not finite or the result is not
    positive definite."""
    M = np.empty((p, p))
    for j in range(p):
        h = 1e-5 * max(abs(float(theta[j])), 1e-2)
        e = np.zeros(p)
        e[j] = h
        fp, gp, _ = objective(theta + e, True)
        fm, gm, _ = objective(theta - e, True)
        if not (math.isfinite(fp) and math.isfinite(fm)):
            return None
        M[:, j] = -(gp - gm) / (2.0 * h)
    M = 0.5 * (M + M.T)
    if not np.all(np.linalg.eigvalsh(M) > 0.0):
        return None
    return M


# ---- targeted GARCH(1,1): (alpha, beta), omega = v (1 - alpha - beta)

GARCH_GRID = [(a, p - a) for a in (0.01, 0.05, 0.1, 0.2) for p in (0.5, 0.7, 0.9, 0.98)]


def garch_vt_objective(r: np.ndarray, start: float, margin: float = 1e-4) -> Objective:
    """The Gaussian log-likelihood of the targeted GARCH on ``r``, ``v`` the
    mean of ``r^2`` over ``r``, the filter started at ``start`` (the backcast),
    as ``vt_garch.vt_fit`` defines it; sensitivities ``d sigma2 / d(alpha,
    beta) = (r2_{t-1} - v, sigma2_{t-1} - v) + beta d_{t-1}``."""
    r2 = r * r
    v = float(np.mean(r2))

    def f(theta: np.ndarray, grad: bool) -> tuple[float, np.ndarray, np.ndarray]:
        a, b = float(theta[0]), float(theta[1])
        omega = v * (1.0 - a - b)
        prev_r2 = prev_s2 = start
        da = db = 0.0
        ll = 0.0
        g = np.zeros(2)
        H = np.zeros((2, 2))
        for t in range(r2.shape[0]):
            s2 = omega + a * prev_r2 + b * prev_s2
            if not s2 > 0.0:
                return -math.inf, g, H
            ll -= 0.5 * (LN_2PI + math.log(s2) + r2[t] / s2)
            if grad:
                da, db = prev_r2 - v + b * da, prev_s2 - v + b * db
                dv = np.array([da, db])
                g += 0.5 * (r2[t] / s2 - 1.0) * dv / s2
                H += 0.5 * np.outer(dv, dv) / (s2 * s2)
            prev_r2, prev_s2 = float(r2[t]), s2
        return ll, g, H

    return f


def garch_vt_project(margin: float = 1e-4) -> Callable[[np.ndarray], np.ndarray]:
    def proj(theta: np.ndarray) -> np.ndarray:
        import garch

        return garch.project(np.array([1.0, theta[0], theta[1]]), 0.0, margin)[1:]

    return proj


def garch_vt_fit(r: np.ndarray, start: float, margin: float = 1e-4) -> Fit:
    return maximize(garch_vt_objective(r, start, margin), garch_vt_project(margin), GARCH_GRID)


# ---- ETS(A,A,N): (alpha, beta), known initial states, least squares

ETS_GRID = [(a, a * q) for a in (0.1, 0.3, 0.5, 0.7, 0.9) for q in (0.05, 0.2, 0.5)]


def ets_objective(y: np.ndarray, l0: float, b0: float) -> Objective:
    """``-1/2 sum e_t^2`` (statsmodels' concentrated likelihood is monotone in
    it), gradient ``sum psi_t e_t``, Gauss-Newton ``sum psi_t psi_t'``, the
    sensitivities of ``ets.py``."""

    def f(theta: np.ndarray, grad: bool) -> tuple[float, np.ndarray, np.ndarray]:
        a, b = float(theta[0]), float(theta[1])
        lev, tr = l0, b0
        Sl = np.zeros(2)
        Sb = np.zeros(2)
        sse = 0.0
        g = np.zeros(2)
        H = np.zeros((2, 2))
        for t in range(y.shape[0]):
            e = float(y[t]) - (lev + tr)
            sse += e * e
            if grad:
                psi = Sl + Sb
                g += psi * e
                H += np.outer(psi, psi)
                Sl = (1.0 - a) * psi + np.array([e, 0.0])
                Sb = Sb - b * psi + np.array([0.0, e])
            lev, tr = lev + tr + a * e, tr + b * e
        return -0.5 * sse, g, H

    return f


def ets_fit(y: np.ndarray, l0: float, b0: float) -> Fit:
    import ets

    return maximize(
        ets_objective(y, l0, b0),
        lambda th: np.array(ets.project(float(th[0]), float(th[1]))),
        ETS_GRID,
    )


# ---- ARMA(1,1): (phi, theta), conditional least squares from a zero start

ARMA_GRID = [(p, q) for p in (-0.5, 0.0, 0.5, 0.9) for q in (-0.5, 0.0, 0.5)]


def arma_objective(y: np.ndarray) -> Objective:
    """``-1/2 sum e_t^2`` with ``e_t = y_t - phi y_{t-1} - theta e_{t-1}`` from
    a zero start, ``psi_t = (y_{t-1}, e_{t-1}) - theta psi_{t-1}``: the
    conditional least squares whose limit is the exact likelihood's."""

    def f(theta: np.ndarray, grad: bool) -> tuple[float, np.ndarray, np.ndarray]:
        phi, th = float(theta[0]), float(theta[1])
        prev_y = prev_e = 0.0
        psi = np.zeros(2)
        sse = 0.0
        g = np.zeros(2)
        H = np.zeros((2, 2))
        for t in range(y.shape[0]):
            e = float(y[t]) - phi * prev_y - th * prev_e
            sse += e * e
            if grad:
                psi = np.array([prev_y, prev_e]) - th * psi
                g += psi * e
                H += np.outer(psi, psi)
            prev_y, prev_e = float(y[t]), e
        return -0.5 * sse, g, H

    return f


def arma_fit(y: np.ndarray, margin: float = 1e-2) -> Fit:
    cap = 1.0 - margin
    return maximize(arma_objective(y), lambda th: np.clip(th, -cap, cap), ARMA_GRID)


def digamma(x: float) -> float:
    """``psi(x)`` for ``x > 0``: the recurrence up to 6, then the asymptotic
        series through ``x^-10`` (absolute error at most about 1e-11, largest at
    small ``x``: 8.8e-12 at ``x = 1``; a test holds it to scipy's within 2e-11).
    It enters only the gradient in ``nu``."""
    acc = 0.0
    while x < 6.0:
        acc -= 1.0 / x
        x += 1.0
    x2 = 1.0 / (x * x)
    return (
        acc
        + math.log(x)
        - 0.5 / x
        - x2 * (1.0 / 12 - x2 * (1.0 / 120 - x2 * (1.0 / 252 - x2 * (1.0 / 240 - x2 / 132))))
    )


# ---- DCC(1,1) second stage, k series: (a, b), the full Gaussian likelihood

DCC_GRID = [(a, b) for a in (0.01, 0.03, 0.05, 0.1) for b in (0.8, 0.85, 0.9, 0.95)]


def dcc_objective(z: np.ndarray, qbar: np.ndarray) -> Objective:
    """The full second-stage log-likelihood with ``Q_0 = Qbar`` (the oracle's
    ``r_path_k``), its score and information as ``dcc.OnlineDccK`` computes
    them per row (``dQ_0 = 0``)."""
    n, k = z.shape

    def f(theta: np.ndarray, grad: bool) -> tuple[float, np.ndarray, np.ndarray]:
        a, b = float(theta[0]), float(theta[1])
        q = qbar.copy()
        dqa = np.zeros((k, k))
        dqb = np.zeros((k, k))
        ll = 0.0
        g = np.zeros(2)
        H = np.zeros((2, 2))
        for t in range(n):
            if t > 0:
                zz = np.outer(z[t - 1], z[t - 1])
                q_prev = q
                q = (1.0 - a - b) * qbar + a * zz + b * q_prev
                if grad:
                    dqa = zz - qbar + b * dqa
                    dqb = q_prev - qbar + b * dqb
            dg = np.diag(q)
            if not np.all(dg > 0.0):
                return -math.inf, g, H
            d = 1.0 / np.sqrt(dg)
            rmat = q * np.outer(d, d)
            sign, logdet = np.linalg.slogdet(rmat)
            if sign <= 0:
                return -math.inf, g, H
            rinv = np.linalg.inv(rmat)
            u = rinv @ z[t]
            ll -= 0.5 * (logdet + float(z[t] @ u) - float(z[t] @ z[t]))
            if grad:
                m = np.outer(u, u) - rinv
                drs = []
                for dq in (dqa, dqb):
                    lam = np.diag(dq) / dg
                    drs.append(
                        dq * np.outer(d, d) - 0.5 * (lam[:, None] * rmat + rmat * lam[None, :])
                    )
                g += np.array([0.5 * float(np.sum(m * dr)) for dr in drs])
                prods = [rinv @ dr for dr in drs]
                H += np.array([[0.5 * float(np.sum(pi * pj.T)) for pj in prods] for pi in prods])
        return ll, g, H

    return f


# ---- DCC(1,1) second stage with multivariate Student-t noise: (a, b, nu)

NU_MIN, NU_MAX = 2.1, 200.0
DCC_T_GRID = [(a, b, v) for a, b in DCC_GRID for v in (4.0, 8.0, 20.0)]


def dcc_t_row(z: np.ndarray, rmat: np.ndarray, nu: float) -> tuple[float, float, float]:
    """One row's multivariate-t log density with correlation ``rmat`` and unit
    variances (scale matrix ``R (nu - 2) / nu``), its Mahalanobis ``s`` and
    its derivative in ``nu``."""
    k = z.shape[0]
    sign, logdet = np.linalg.slogdet(rmat)
    s = float(z @ np.linalg.solve(rmat, z))
    h = 0.5 * (nu + k)
    ll = (
        math.lgamma(h)
        - math.lgamma(0.5 * nu)
        - 0.5 * k * math.log(math.pi * (nu - 2.0))
        - 0.5 * logdet
        - h * math.log1p(s / (nu - 2.0))
    )
    dnu = (
        0.5 * digamma(h)
        - 0.5 * digamma(0.5 * nu)
        - 0.5 * k / (nu - 2.0)
        - 0.5 * math.log1p(s / (nu - 2.0))
        + h * s / ((nu - 2.0) * (nu - 2.0 + s))
    )
    return ll if sign > 0 else -math.inf, s, dnu


def dcc_t_objective(z: np.ndarray, qbar: np.ndarray) -> Objective:
    """The multivariate-t second-stage log-likelihood in ``(a, b, nu)``, ``Q``
    as ``dcc_objective``. Score in ``(a, b)``: ``1/2 tr((w u u' - R^{-1}) dR)``
    with ``w = (nu + k) / (nu - 2 + s)``; in ``nu``: ``dcc_t_row``'s. Its
    Gauss-Newton fallback is the sum of the rows' score outer products."""
    n, k = z.shape

    def f(theta: np.ndarray, grad: bool) -> tuple[float, np.ndarray, np.ndarray]:
        a, b, nu = (float(x) for x in theta)
        q = qbar.copy()
        dqa = np.zeros((k, k))
        dqb = np.zeros((k, k))
        ll = 0.0
        g = np.zeros(3)
        H = np.zeros((3, 3))
        if not nu > 2.0:
            return -math.inf, g, H
        for t in range(n):
            if t > 0:
                zz = np.outer(z[t - 1], z[t - 1])
                q_prev = q
                q = (1.0 - a - b) * qbar + a * zz + b * q_prev
                if grad:
                    dqa = zz - qbar + b * dqa
                    dqb = q_prev - qbar + b * dqb
            dg = np.diag(q)
            if not np.all(dg > 0.0):
                return -math.inf, g, H
            d = 1.0 / np.sqrt(dg)
            rmat = q * np.outer(d, d)
            lr, s, dnu = dcc_t_row(z[t], rmat, nu)
            if not math.isfinite(lr):
                return -math.inf, g, H
            ll += lr
            if grad:
                rinv = np.linalg.inv(rmat)
                u = rinv @ z[t]
                m = (nu + k) / (nu - 2.0 + s) * np.outer(u, u) - rinv
                row = np.empty(3)
                for i, dq in enumerate((dqa, dqb)):
                    lam = np.diag(dq) / dg
                    dr = dq * np.outer(d, d) - 0.5 * (lam[:, None] * rmat + rmat * lam[None, :])
                    row[i] = 0.5 * float(np.sum(m * dr))
                row[2] = dnu
                g += row
                H += np.outer(row, row)
        return ll, g, H

    return f


def dcc_t_fit(z: np.ndarray) -> Fit:
    import dcc

    qbar = z.T @ z / z.shape[0]

    def proj(th: np.ndarray) -> np.ndarray:
        a, b = dcc.project(float(th[0]), float(th[1]))
        return np.array([a, b, min(max(float(th[2]), NU_MIN), NU_MAX)])

    return maximize(dcc_t_objective(z, qbar), proj, DCC_T_GRID)


def dcc_fit(z: np.ndarray) -> Fit:
    import dcc

    qbar = z.T @ z / z.shape[0]
    return maximize(
        dcc_objective(z, qbar),
        lambda th: np.array(dcc.project(float(th[0]), float(th[1]))),
        DCC_GRID,
    )
