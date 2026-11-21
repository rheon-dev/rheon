"""Expectiles, scalar and regression: one-pass online forms and exact refits.

Notation. A row is ``(z, y, w)``: ``z`` the regressors with the constant
column first (``p`` of them), ``y`` the target, ``w >= 0`` the row's weight.
``lam`` in ``(0, 1]`` is the decay a row step applies to everything before
it. ``tau`` in ``(0, 1)`` is the expectile level. The asymmetric weight of a
residual ``r = y - z'beta`` is (scipy's convention at ``r = 0``)::

    a(r) = tau        if r > 0
         = 1 - tau    if r <= 0

The expectile regression at ``tau`` minimises ``sum v_i a(r_i) r_i^2``, ``v_i``
the row's weight decayed to now; its stationarity condition is
``sum v_i a(r_i) r_i z_i = 0``. The loss is convex, piecewise quadratic and
C1, so for a fixed assignment of sides the minimiser is a weighted least
squares fit, and the minimiser of the loss is the fit whose sides agree with
its own residuals (Newey and Powell 1987; Efron 1991's IRLS).

**One pass** (``OnlineExpectile``). The row's side is judged by its residual
against the fit *before* it (the prediction, out of sample by rule 2), and the
row then enters decayed mean-form sums at weight ``w a``::

    S'  = lam S + w a                      (S: the asymmetric weight)
    f   = w a / S'                         (guarded: w a = 0 -> no update)
    G'  = G + f (z z' - G)                 (mean Gram)
    c'  = c + f (z y - c)                  (mean cross-moment)
    W'  = lam W + w                        (n_eff, the raw weight: rule 8)
    beta = G'^-1 c'

A row's update cannot move its own residual across zero. From
``S'G' beta' = lam S G beta + w a z y`` and ``S G beta = S c``,
``beta' - beta = w a (S'G')^-1 z r``, so ``r_post = r (1 - w a h)`` with
``h = z'(S'G')^-1 z``, and ``0 <= w a h <= 1`` because ``S'G'`` holds
``w a z z'``. The residual after the row keeps the sign of the one before it
(or reaches 0, where the row is interpolated and its weight does not matter),
whatever ``lam`` is, so judging the side after the row gives the same side,
and the one pass needs no inner iteration on the row itself. What one pass cannot do
is re-judge an *earlier* row whose side the fit has since crossed; that needs
the rows, which the refit and the window keep.

**Refit** (``irls``). Over kept rows with weights ``v``: start from sides at
a given fit, solve the weighted least squares, re-judge every side, repeat
until no side changes (an exact fixed point: the WLS fit of a self-consistent
assignment is the loss's minimiser). At ``tau = 1/2`` one solve is the answer. A full Newton step can in principle
cycle on a piecewise quadratic; a step that does not lower the loss is halved
(the loss is convex, so a halving always helps). It also stops when the
fitted values move by no more than ``tol`` of the residual scale or by no
more than their own rounding (``8 eps max|fitted|``: rows whose residual is at
rounding level flip sides with no effect on the fit, and an exact fit has a
residual scale of 0), when a side pattern comes back with no lower loss
than it had (a cycle of rows at rounding level trading sides: the lowest
loss met is kept), and after ``max_iter`` passes, which is reported (a solve
failure).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

TOL = 1e-12
EPS = float(np.finfo(float).eps)
MAX_ITER = 50


def asym(r: np.ndarray | float, tau: float) -> np.ndarray:
    return np.where(np.asarray(r) > 0.0, tau, 1.0 - tau)


def loss(beta: np.ndarray, Z: np.ndarray, y: np.ndarray, v: np.ndarray, tau: float) -> float:
    r = y - Z @ beta
    return float(np.sum(v * asym(r, tau) * r * r))


def wls(Z: np.ndarray, y: np.ndarray, v: np.ndarray) -> np.ndarray | None:
    """Weighted least squares by the normal equations (the longhand)."""
    if not np.any(v > 0.0):
        return None
    G = (Z * v[:, None]).T @ Z
    c = (Z * v[:, None]).T @ y
    return chol_solve(G, c)


def chol_solve(G: np.ndarray, c: np.ndarray) -> np.ndarray | None:
    """G^-1 c by Cholesky; None when G is not positive definite to working
    precision (a pivot under 1e-12 of the largest diagonal: no fit)."""
    try:
        L = np.linalg.cholesky(G)
    except np.linalg.LinAlgError:
        return None
    if np.min(np.diag(L)) ** 2 < 1e-12 * np.max(np.diag(G)):
        return None
    u = np.linalg.solve(L, c)
    return np.linalg.solve(L.T, u)


@dataclass
class IrlsResult:
    beta: np.ndarray | None
    passes: int  # WLS solves after the first
    converged: bool
    reason: str  # "sides", "fit", "cap", "empty"
    halvings: int = 0


def irls(
    Z: np.ndarray,
    y: np.ndarray,
    v: np.ndarray,
    tau: float,
    beta0: np.ndarray | None = None,
    tol: float = TOL,
    max_iter: int = MAX_ITER,
) -> IrlsResult:
    """The exact expectile regression of the kept rows, warm-started at beta0."""
    if beta0 is None:
        beta = wls(Z, y, v)  # every side at 1/2: least squares
        if beta is None:
            return IrlsResult(None, 0, False, "empty")
    else:
        beta = beta0
    if tau == 0.5:  # every side weighs 1/2: the least-squares fit is the answer
        return IrlsResult(wls(Z, y, v), 1, True, "sides")
    sides = (y - Z @ beta) > 0.0
    halvings = 0
    f_old = loss(beta, Z, y, v, tau)
    # The loss's and the fitted values' rounding: an exact fit (residuals at
    # rounding level) must not read as a fit that still moves.
    f_floor = 8 * EPS * float(np.sum(v * y * y))
    seen = {sides.tobytes(): (f_old, beta)}
    for it in range(1, max_iter + 1):
        a = np.where(sides, tau, 1.0 - tau)
        new = wls(Z, y, v * a)
        if new is None:
            return IrlsResult(beta, it, False, "empty", halvings)
        step = new - beta
        f_new = loss(new, Z, y, v, tau)
        h = 1.0
        while f_new > f_old + max(f_old, f_floor) * 8 * EPS and h > 1e-6:
            h *= 0.5
            halvings += 1
            new = beta + h * step
            f_new = loss(new, Z, y, v, tau)
        fitted = Z @ new
        r_new = y - fitted
        moved = float(np.max(np.abs(Z @ (new - beta))))
        scale = float(np.sqrt(np.sum(v * r_new * r_new) / max(np.sum(v), 1e-300)))
        rounding = 8 * EPS * float(np.max(np.abs(fitted)))
        new_sides = r_new > 0.0
        beta, f_old = new, f_new
        if h == 1.0 and np.array_equal(new_sides, sides):
            return IrlsResult(beta, it, True, "sides", halvings)
        if moved <= max(tol * scale, rounding):
            return IrlsResult(beta, it, True, "fit", halvings)
        key = new_sides.tobytes()
        # A side pattern met before with no lower loss since: a cycle that
        # makes no progress (rows at rounding level trading sides). A pattern
        # met again at a lower loss is a damped path still descending.
        if key in seen and f_new >= seen[key][0] - max(f_new, f_floor) * 8 * EPS:
            f_best, b_best = min([*seen.values(), (f_new, beta)], key=lambda e: e[0])
            return IrlsResult(b_best, it, True, "cycle", halvings)
        seen[key] = (f_new, beta)
        sides = new_sides
    return IrlsResult(beta, max_iter, False, "cap", halvings)


def expectile_exact(
    y: np.ndarray, tau: float, w: np.ndarray | None = None, compensated: bool = True
) -> float:
    """The scalar expectile by sorting: exact to rounding, no root finder.

    On the segment ``[y_j, y_{j+1})`` the first-order condition is linear, so
    ``t = (tau S_U + (1 - tau) S_L) / (tau W_U + (1 - tau) W_L)`` with ``L`` the
    values at or below ``t`` and ``U`` those above; the segment whose ``t``
    lands inside it is the answer.
    """
    y = np.asarray(y, float)
    w = np.ones_like(y) if w is None else np.asarray(w, float)
    keep = w > 0.0
    y, w = y[keep], w[keep]
    if y.size == 0:
        return float("nan")
    o = np.argsort(y, kind="stable")
    y, w = y[o], w[o]
    WL = np.cumsum(w)
    SL = np.cumsum(w * y)
    WU, SU = WL[-1] - WL, SL[-1] - SL
    t = (tau * SU + (1.0 - tau) * SL) / (tau * WU + (1.0 - tau) * WL)
    hi = np.append(y[1:], np.inf)
    ok = (t >= y) & (t < hi)
    if ok.any():
        j = int(np.argmax(ok))
    else:  # rounding left t a hair outside every segment: take the nearest
        j = int(np.argmin(np.maximum(y - t, 0.0) + np.maximum(t - hi, 0.0)))
    if not compensated:
        return float(t[j])
    # The running sums lose up to ~1e-13 relative under steep decay (the
    # upper sums are total minus lower); the segment's own sums, each
    # correctly rounded (fsum), put t within an ulp or two of the root.
    lo, up = slice(0, j + 1), slice(j + 1, None)
    num = tau * math.fsum(w[up] * y[up]) + (1.0 - tau) * math.fsum(w[lo] * y[lo])
    den = tau * math.fsum(w[up]) + (1.0 - tau) * math.fsum(w[lo])
    return num / den


@dataclass
class OnlineExpectile:
    """One pass, decayed mean form; p regressors (constant first)."""

    p: int
    tau: float
    lam: float = 1.0
    guard: bool = True
    S: float = 0.0  # asymmetric weight, decayed
    W: float = 0.0  # raw weight, decayed (n_eff)
    rows: int = 0  # rows with positive weight, counted one each
    G: np.ndarray = field(default=None)  # type: ignore[assignment]
    c: np.ndarray = field(default=None)  # type: ignore[assignment]
    beta: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.G is None:
            self.G = np.zeros((self.p, self.p))
            self.c = np.zeros(self.p)

    def predict(self, z: np.ndarray) -> float:
        return float("nan") if self.beta is None else float(z @ self.beta)

    def step(self, z: np.ndarray, y: float, w: float) -> tuple[float, float]:
        """Predict, then learn. Returns (prediction, n_eff before the row)."""
        pred = self.predict(z)
        n_eff = self.W
        r = y - pred
        a = 0.5 if np.isnan(pred) else float(asym(r, self.tau))
        wa = w * a
        S = self.lam * self.S + wa
        self.W = self.lam * self.W + w
        if self.guard and wa == 0.0:
            self.S = S  # advance the clock, learn nothing
            return pred, n_eff
        f = wa / S
        self.S = S
        self.G = self.G + f * (np.outer(z, z) - self.G)
        self.c = self.c + f * (z * y - self.c)
        if w > 0.0:
            self.rows += 1
        if not np.all(np.isfinite(self.G)):
            self.beta = np.full(self.p, np.nan)
        elif self.rows >= self.p:
            b = chol_solve(self.G, self.c)
            if b is not None:
                self.beta = b
        return pred, n_eff

    def state(self) -> np.ndarray:
        return np.concatenate([[self.S, self.W], self.G[np.triu_indices(self.p)], self.c])


def run_online(Z, y, w, tau, lam=1.0, guard=True, refit_rows=0):
    """One pass over the stream; with ``refit_rows`` the first rows are refit
    exactly once the stream reaches them, the sums rebuilt at the converged
    sides (as ``quantile_refit_rows`` does for ``quantile``)."""
    m = OnlineExpectile(Z.shape[1], tau, lam, guard)
    preds = np.empty(len(y))
    neff = np.empty(len(y))
    info = None
    for t in range(len(y)):
        preds[t], neff[t] = m.step(Z[t], y[t], w[t])
        if refit_rows and t + 1 == refit_rows:
            v = w[: t + 1] * lam ** np.arange(t, -1, -1.0)
            info = irls(Z[: t + 1], y[: t + 1], v, tau, m.beta)
            if info.beta is not None:
                m = rebuild(Z[: t + 1], y[: t + 1], w[: t + 1], tau, lam, info.beta, m)
    return m, preds, neff, info


def rebuild(Z, y, w, tau, lam, beta, m: OnlineExpectile) -> OnlineExpectile:
    """The one-pass sums rebuilt row by row with each side judged at beta."""
    a = asym(y - Z @ beta, tau)
    n = OnlineExpectile(m.p, tau, lam, m.guard)
    for t in range(len(y)):
        wa = w[t] * a[t]
        n.S = lam * n.S + wa
        n.W = lam * n.W + w[t]
        if wa == 0.0:
            continue
        f = wa / n.S
        n.G = n.G + f * (np.outer(Z[t], Z[t]) - n.G)
        n.c = n.c + f * (Z[t] * y[t] - n.c)
        n.rows += 1
    n.beta = np.linalg.solve(n.G, n.c)
    return n


def run_window(Z, y, w, tau, window, lam=1.0, tol=TOL, max_iter=MAX_ITER):
    """Exact expectile regression of the last ``window`` rows (decayed by lam
    within the window), refit by warm-started IRLS on every row."""
    n, p = Z.shape
    preds = np.full(n, np.nan)
    neff = np.zeros(n)
    passes = np.zeros(n, int)
    fails = 0
    halv = 0
    beta = None
    for t in range(n):
        lo = max(0, t - window + 1)
        if beta is not None:
            preds[t] = Z[t] @ beta
        lo_prev = max(0, t - window)
        neff[t] = float(np.sum(w[lo_prev:t] * lam ** np.arange(t - 1 - lo_prev, -1, -1.0)))
        v = w[lo : t + 1] * lam ** np.arange(t - lo, -1, -1.0)
        if np.count_nonzero(v) < p:
            continue
        res = irls(Z[lo : t + 1], y[lo : t + 1], v, tau, beta, tol, max_iter)
        if res.beta is None:
            continue
        passes[t] = res.passes
        fails += not res.converged
        halv += res.halvings
        beta = res.beta
    return beta, preds, neff, passes, fails, halv


def scalar_online(y, w, tau, lam=1.0):
    """The scalar one-pass expectile: t = sum v a y / sum v a, sides judged at
    the expectile before the row; guarded for zero weight."""
    S = 0.0
    m = 0.0
    out = np.full(len(y), np.nan)
    have = False
    for i in range(len(y)):
        if have:
            out[i] = m
        a = float(asym(y[i] - m, tau)) if have else 0.5
        wa = w[i] * a
        S = lam * S + wa
        if wa == 0.0:
            continue
        m = m + (wa / S) * (y[i] - m)
        have = True
    return (m if have else float("nan")), out
