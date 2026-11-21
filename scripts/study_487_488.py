"""Study 487 and 488: online SARIMA and seasonal Holt-Winters ETS against statsmodels.

Two numpy studies (the record is docs/records/STUDY-487-488.md).

**487, SARIMA(p,d,q)(P,D,Q,s)** (Box and Jenkins; Box, Jenkins, Reinsel and
Ljung 2015, ch. 9)::

    phi(B) Phi(B^s) w_t = theta(B) Theta(B^s) e_t,   w_t = (1 - B)^d (1 - B^s)^D y_t

with ``phi(B) = 1 - phi_1 B - ...``, ``Phi(B^s) = 1 - Phi_1 B^s - ...``,
``theta(B) = 1 + theta_1 B + ...``, ``Theta(B^s) = 1 + Theta_1 B^s + ...``
(statsmodels' signs). Write ``A(B) = phi(B) Phi(B^s)`` and
``C(B) = theta(B) Theta(B^s)``. The online estimator keeps a ring of the last
``d + sD`` targets for the differencing and runs the recursive
prediction-error method (Ljung and Soderstrom 1983) on ``w``. Row ``t``'s
prediction is made before ``y_t`` is read::

    what_t = -sum_{k>=1} A_k w_{t-k} + sum_{j>=1} C_j ehat_{t-j}
    yhat_t = what_t - sum_{k>=1} delta_k y_{t-k}        (delta(B) = (1-B)^d (1-B^s)^D)

The gradient ``psi_t = -d e_t / d beta`` is carried by four filtered signals,
each a ring, at the current parameters::

    u_t = Phi(B^s) w_t / C(B)      psi_{phi_i}   = u_{t-i}
    v_t = phi(B) w_t / C(B)        psi_{Phi_i}   = v_{t-s i}
    r_t = Theta(B^s) e_t / C(B)    psi_{theta_j} = r_{t-j}
    z_t = theta(B) e_t / C(B)      psi_{Theta_j} = z_{t-s j}

(``x / C(B)`` is the recursion ``x_t - sum_j C_j xf_{t-j}``.) Each row takes
one Gauss-Newton step, gain ``1/W_t``::

    R_t    = R_{t-1} + (psi_t psi_t' - R_{t-1}) / W_t
    beta_t = Proj(beta_{t-1} + R_t^{-1} psi_t e_t / W_t)
    s2_t   = s2_{t-1} + (e_t^2 - s2_{t-1}) / W_t

``Proj`` clips each coefficient to ``[-0.99, 0.99]``: with each polynomial of
order at most one (the study's specs), that is stationarity and invertibility.
The kept residual is the a-posteriori one. Start (nothing buffered): the first
``d + sD`` rows fill the differencing ring and emit nothing, each with its
stated reason; then ``warm`` rows at ``beta = 0`` accumulate the lag-1 and
lag-``s`` autocorrelations of ``w``, which start ``phi_1`` and ``Phi_1``
(the moment estimates; with ``phi = theta = 0`` the two regressors are the same
column); then ``warm`` rows build ``R`` and ``s2`` at held ``beta``; steps
follow. Every row after the differencing ring emits.

The oracle: ``statsmodels.tsa.statespace.sarimax.SARIMAX`` (settings in
``sarimax_fit``) fitted on the same rows.

**488, ETS** (Hyndman, Koehler, Ord and Snyder 2008, ch. 2 and Table 2.2) is
added by parts 3 and 4.

Data are generated from seeded numpy generators; nothing is read from disk.
numpy only in the estimators; statsmodels and scipy are imported only in the
oracles and the data generator (``scipy.signal.lfilter``).

Run (the study's venv holds numpy, scipy, pandas and statsmodels)::

    .venv-487/bin/python scripts/study_487_488.py part1   (SARIMA, s = 7)

Each grid case is appended to ``<out>/<grid>.jsonl`` as it finishes, so a rerun
resumes; ``<out>/<part>.json`` holds the part's summary.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass

import numpy as np

SEEDS = list(range(485000, 485050))
N_ROWS = 10_000
MARGIN = 1e-2
CAP = 1.0 - MARGIN


# --------------------------------------------------------------------------
# 487: SARIMA


@dataclass(frozen=True)
class Spec:
    """A SARIMA spec and its truth (each polynomial of order 0 or 1)."""

    name: str
    p: int
    d: int
    q: int
    P: int
    D: int
    Q: int
    s: int
    truth: tuple[float, ...]  # statsmodels' order: ar, ma, seasonal ar, seasonal ma, sigma2

    def __post_init__(self) -> None:
        assert all(o in (0, 1) for o in (self.p, self.q, self.P, self.Q))
        assert len(self.truth) == self.n_params + 1

    @property
    def n_params(self) -> int:
        return self.p + self.q + self.P + self.Q

    @property
    def label(self) -> str:
        return f"SARIMA({self.p},{self.d},{self.q})({self.P},{self.D},{self.Q},{self.s})"

    @property
    def names(self) -> list[str]:
        out = []
        out += ["ar.L1"] * self.p
        out += ["ma.L1"] * self.q
        out += [f"ar.S.L{self.s}"] * self.P
        out += [f"ma.S.L{self.s}"] * self.Q
        return out + ["sigma2"]


def polys(spec: Spec, beta: np.ndarray):
    """``A(B) = phi(B) Phi(B^s)`` and ``C(B) = theta(B) Theta(B^s)`` as lists
    of ``(lag, coefficient)`` for lags >= 1, from ``beta`` in statsmodels'
    order."""
    i = 0
    phi = beta[i] if spec.p else 0.0
    i += spec.p
    th = beta[i] if spec.q else 0.0
    i += spec.q
    Phi = beta[i] if spec.P else 0.0
    i += spec.P
    Th = beta[i] if spec.Q else 0.0
    s = spec.s
    a = []
    if spec.p:
        a.append((1, -phi))
    if spec.P:
        a.append((s, -Phi))
    if spec.p and spec.P:
        a.append((s + 1, phi * Phi))
    c = []
    if spec.q:
        c.append((1, th))
    if spec.Q:
        c.append((s, Th))
    if spec.q and spec.Q:
        c.append((s + 1, th * Th))
    return a, c, (phi, th, Phi, Th)


def delta_poly(d: int, D: int, s: int) -> list[tuple[int, float]]:
    """``(1 - B)^d (1 - B^s)^D`` as ``(lag, coefficient)`` for lags >= 1."""
    poly = np.zeros(d + s * D + 1)
    poly[0] = 1.0
    for _ in range(d):
        poly = poly - np.concatenate([[0.0], poly[:-1]])
    for _ in range(D):
        poly = poly - np.concatenate([np.zeros(s), poly[:-s]])
    return [(k, float(poly[k])) for k in range(1, poly.shape[0]) if poly[k] != 0.0]


class Ring:
    """A fixed-length ring of floats; ``lag(k)`` is the value written ``k``
    rows ago (``1 <= k <= len``), zero before it was written."""

    __slots__ = ("buf", "pos", "n")

    def __init__(self, n: int) -> None:
        self.n = n
        self.buf = np.zeros(max(n, 1))
        self.pos = 0

    def lag(self, k: int) -> float:
        assert 1 <= k <= self.n
        return float(self.buf[(self.pos - k) % self.n])

    def push(self, x: float) -> None:
        if self.n:
            self.buf[self.pos] = x
            self.pos = (self.pos + 1) % self.n


def ring_lengths(spec: Spec) -> dict[str, int]:
    """Each ring's length: the largest lag any recursion reads from it."""
    s = spec.s
    le = spec.q + s * spec.Q
    return {
        "y (differencing)": spec.d + s * spec.D,
        "w (AR lags)": spec.p + s * spec.P,
        "e (MA lags)": le,
        "u (psi_phi)": max(spec.p, le) if spec.p else 0,
        "v (psi_Phi)": max(s * spec.P, le) if spec.P else 0,
        "r (psi_theta)": max(spec.q, le) if spec.q else 0,
        "z (psi_Theta)": max(s * spec.Q, le) if spec.Q else 0,
    }


def state_floats(spec: Spec) -> dict[str, int]:
    """The estimator's state in floats: the rings, ``beta``, ``R`` (distinct
    entries), ``s2``, the gain's weight, the row count and the two warm sums
    with the lag-``s`` one."""
    n = spec.n_params
    out = dict(ring_lengths(spec))
    out["beta"] = n
    out["R (distinct)"] = n * (n + 1) // 2
    out["scalars (s2, W, t, warm sums)"] = 6
    return out


class OnlineSarima:
    """The recursive prediction-error estimator. ``update(y)`` takes one row
    and returns its prediction (made before ``y`` is read; NaN while the
    differencing ring fills, with the reason in ``reasons``)."""

    def __init__(self, spec: Spec, warm: int | None = None, batch_start: int = 0) -> None:
        self.spec = spec
        self.batch_start = batch_start
        self.buffer: list[float] = []
        self.started = not batch_start
        self.batch_fit = None
        self.warm = warm if warm is not None else max(50, 2 * spec.s)
        L = ring_lengths(spec)
        self.ry = Ring(L["y (differencing)"])
        self.rw = Ring(L["w (AR lags)"])
        self.re = Ring(L["e (MA lags)"])
        self.ru = Ring(L["u (psi_phi)"])
        self.rv = Ring(L["v (psi_Phi)"])
        self.rr = Ring(L["r (psi_theta)"])
        self.rz = Ring(L["z (psi_Theta)"])
        self.delta = delta_poly(spec.d, spec.D, spec.s)
        n = spec.n_params
        self.beta = np.zeros(n)
        self.R = np.zeros((n, n))
        self.s2 = 0.0
        self.W = 0.0
        self.t_y = 0  # rows read
        self.t = 0  # rows of w
        self.syy = self.sy1 = self.sys = 0.0
        self.reasons: dict[int, str] = {}
        self._set_polys()

    def _set_polys(self) -> None:
        self.a, self.c, self.coef = polys(self.spec, self.beta)

    def _psi(self) -> np.ndarray:
        sp = self.spec
        out = []
        if sp.p:
            out.append(self.ru.lag(1))
        if sp.q:
            out.append(self.rr.lag(1))
        if sp.P:
            out.append(self.rv.lag(sp.s))
        if sp.Q:
            out.append(self.rz.lag(sp.s))
        return np.array(out)

    def _what(self) -> float:
        acc = 0.0
        for k, ak in self.a:
            acc -= ak * self.rw.lag(k)
        for j, cj in self.c:
            acc += cj * self.re.lag(j)
        return acc

    def update(self, y: float) -> float:
        row = self.t_y
        self.t_y += 1
        sp = self.spec
        if row < self.ry.n:
            self.ry.push(y)
            self.reasons[row] = (
                f"filling the differencing ring ({self.ry.n} rows: d = {sp.d}, "
                f"D = {sp.D}, s = {sp.s}); no differenced value yet"
            )
            return math.nan
        base = 0.0
        for k, dk in self.delta:
            base -= dk * self.ry.lag(k)
        self.ry.push(y)
        w = y - base
        if not self.started:
            self.buffer.append(w)
            self.reasons[row] = (
                f"buffering {self.batch_start} differenced rows for the batch start "
                "(fitted by the core's optimizer); no parameters yet"
            )
            if len(self.buffer) == self.batch_start:
                self._start()
            return math.nan
        what = self._what()
        pred = base + what
        self._row_w(w, what)
        return pred

    def _start(self) -> None:
        """Remedy (a) of study 485: fit the buffer's conditional least squares
        by the core-held optimizer (``batch_core.maximize``, projected Newton),
        then replay the buffer at the fitted ``beta`` with ``beta`` held,
        building ``R``, ``s2`` and the rings; steps follow at gain ``1/W``,
        ``W`` counting the buffer."""
        rows = np.asarray(self.buffer)
        self.batch_fit = css_fit(self.spec, rows)
        self.beta = np.clip(self.batch_fit.theta, -CAP, CAP)
        self._set_polys()
        for w in rows:
            what = self._what()
            e = float(w) - what
            psi = self._psi()
            self.W += 1.0
            g = 1.0 / self.W
            self.R += g * (np.outer(psi, psi) - self.R)
            self.s2 += g * (e * e - self.s2)
            self._advance(float(w), e)
        self.t = len(self.buffer)
        self.buffer = []
        self.started = True

    def _row_w(self, w: float, what: float) -> None:
        sp = self.spec
        e = w - what
        t = self.t
        self.t += 1
        warm = self.warm
        if self.batch_start:
            psi = self._psi()
            self.W += 1.0
            g = 1.0 / self.W
            self.R += g * (np.outer(psi, psi) - self.R)
            self.s2 += g * (e * e - self.s2)
            step = g * np.linalg.solve(self.R, psi * e)
            self.beta = np.clip(self.beta + step, -CAP, CAP)
        elif t < warm:
            # phase 1: the lag-1 and lag-s autocorrelations at beta = 0
            self.syy += w * w
            if sp.p:
                self.sy1 += w * self.rw.lag(1)
            if sp.P:
                self.sys += w * self.rw.lag(sp.s)
            if t == warm - 1:
                rho1 = self.sy1 / self.syy if self.syy > 0.0 else 0.0
                rhos = self.sys / self.syy if self.syy > 0.0 else 0.0
                i = 0
                if sp.p:
                    self.beta[i] = min(max(rho1, -CAP), CAP)
                i += sp.p + sp.q
                if sp.P:
                    self.beta[i] = min(max(rhos, -CAP), CAP)
        else:
            psi = self._psi()
            self.W += 1.0
            g = 1.0 / self.W
            self.R += g * (np.outer(psi, psi) - self.R)
            self.s2 += g * (e * e - self.s2)
            if t >= 2 * warm:
                step = g * np.linalg.solve(self.R, psi * e)
                self.beta = np.clip(self.beta + step, -CAP, CAP)
        self._set_polys()
        # the a-posteriori residual, the filtered signals, the rings
        e_post = w - self._what()
        self._advance(w, e_post)

    def _advance(self, w: float, e: float) -> None:
        sp = self.spec
        phi, th, Phi, Th = self.coef
        s = sp.s

        def over_c(x: float, ring: Ring) -> float:
            for j, cj in self.c:
                x -= cj * ring.lag(j)
            return x

        if sp.p:
            x = w - (Phi * self.rw.lag(s) if sp.P else 0.0)
            self.ru.push(over_c(x, self.ru))
        if sp.P:
            x = w - (phi * self.rw.lag(1) if sp.p else 0.0)
            self.rv.push(over_c(x, self.rv))
        if sp.q:
            x = e + (Th * self.re.lag(s) if sp.Q else 0.0)
            self.rr.push(over_c(x, self.rr))
        if sp.Q:
            x = e + (th * self.re.lag(1) if sp.q else 0.0)
            self.rz.push(over_c(x, self.rz))
        self.rw.push(w)
        self.re.push(e)

    @property
    def params(self) -> np.ndarray:
        return np.concatenate([self.beta, [self.s2]])

    def state_bytes(self) -> int:
        return 8 * sum(state_floats(self.spec).values())

    def allocated_floats(self) -> int:
        """Floats the implementation holds (a check on ``state_floats``)."""
        n = self.spec.n_params
        rings = sum(r.n for r in (self.ry, self.rw, self.re, self.ru, self.rv, self.rr, self.rz))
        return rings + n + n * (n + 1) // 2 + 6


def css_objective(spec: Spec, w: np.ndarray):  # noqa: ANN201
    """``-1/2 sum e_t^2`` over ``w`` from a zero start, its gradient
    ``sum psi_t e_t`` and Gauss-Newton matrix ``sum psi_t psi_t'``, by the
    estimator's own recursions at fixed parameters (``batch_core``'s
    objective form)."""

    def f(theta: np.ndarray, grad: bool):  # noqa: ANN202
        est = OnlineSarima(spec, warm=10**9)
        est.beta = np.asarray(theta, dtype=float).copy()
        est._set_polys()
        n = spec.n_params
        g = np.zeros(n)
        H = np.zeros((n, n))
        sse = 0.0
        for v in w:
            what = est._what()
            e = float(v) - what
            sse += e * e
            if grad:
                psi = est._psi()
                g += psi * e
                H += np.outer(psi, psi)
            est._advance(float(v), e)
        if not math.isfinite(sse):
            return -math.inf, g, H
        return -0.5 * sse, g, H

    return f


def css_fit(spec: Spec, w: np.ndarray):  # noqa: ANN201
    """The batch start's fit: ``batch_core.maximize`` (study 485's core-held
    projected Newton) on the conditional least squares, the projection the
    online clip, the start grid ``{-0.5, 0, 0.5}`` per coefficient."""
    import itertools

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "study_485"))
    import batch_core

    grid = [tuple(g) for g in itertools.product((-0.5, 0.0, 0.5), repeat=spec.n_params)]
    return batch_core.maximize(css_objective(spec, w), lambda th: np.clip(th, -CAP, CAP), grid)


def fixed_predictions(spec: Spec, beta: np.ndarray, w: np.ndarray) -> np.ndarray:
    """One-step predictions of ``w`` at fixed parameters from a zero start
    (the conditional recursion), and the residuals."""
    a, c, _ = polys(spec, beta)
    n = w.shape[0]
    e = np.zeros(n)
    what = np.zeros(n)
    for t in range(n):
        acc = 0.0
        for k, ak in a:
            if t - k >= 0:
                acc -= ak * w[t - k]
        for j, cj in c:
            if t - j >= 0:
                acc += cj * e[t - j]
        what[t] = acc
        e[t] = w[t] - acc
    return what


def simulate_sarima(spec: Spec, n: int, seed: int, burn: int = 2000) -> np.ndarray:
    """``n`` rows of ``y``: the ARMA ``w`` by the multiplied polynomials after
    ``burn`` discarded rows, integrated by ``1/delta(B)`` from zeros."""
    from scipy.signal import lfilter

    rng = np.random.default_rng(seed)
    sigma = math.sqrt(spec.truth[-1])
    e = sigma * rng.standard_normal(n + burn)
    a, c, _ = polys(spec, np.asarray(spec.truth[:-1], dtype=float))
    L = max([k for k, _ in a + c] + [0]) + 1
    ap = np.zeros(L)
    cp = np.zeros(L)
    ap[0] = cp[0] = 1.0
    for k, v in a:
        ap[k] = v
    for k, v in c:
        cp[k] = v
    w = lfilter(cp, ap, e)[burn:]
    delta = delta_poly(spec.d, spec.D, spec.s)
    y = np.zeros(n)
    for t in range(n):
        acc = w[t]
        for k, dk in delta:
            if t - k >= 0:
                acc -= dk * y[t - k]
        y[t] = acc
    return y


SARIMAX_SETTINGS = {
    "trend": "n",
    "simple_differencing": True,
    "enforce_stationarity": True,
    "enforce_invertibility": True,
    "concentrate_scale": False,
    "method": "lbfgs",
    "maxiter": 200,
    "cov_type": "opg",
    "low_memory": "fit(low_memory=True, cov_type='none'), then filter(cov_type='opg') "
    "storing no covariance arrays (identical to the default path, part 2)",
}


LEAN = 0  # set by _lean() at first use


def sarimax_fit(spec: Spec, y: np.ndarray, start_params=None):  # noqa: ANN201
    """The oracle: ``SARIMAX(y, order=(p,d,q), seasonal_order=(P,D,Q,s),
    trend="n", simple_differencing=True)`` (so it sees the same differenced
    rows the online estimator sees; the first ``d + sD`` rows are dropped),
    stationarity and invertibility enforced (statsmodels' defaults), the
    default stationary initialization of the ARMA state, exact Gaussian
    likelihood by the Kalman filter, ``fit(method="lbfgs", maxiter=200,
    cov_type="opg", disp=False)``."""
    import warnings

    from statsmodels.tsa.statespace.sarimax import SARIMAX

    global LEAN
    LEAN = _lean()
    st = SARIMAX_SETTINGS
    model = SARIMAX(
        y,
        order=(spec.p, spec.d, spec.q),
        seasonal_order=(spec.P, spec.D, spec.Q, spec.s),
        trend=st["trend"],
        simple_differencing=st["simple_differencing"],
        enforce_stationarity=st["enforce_stationarity"],
        enforce_invertibility=st["enforce_invertibility"],
        concentrate_scale=st["concentrate_scale"],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = model.fit(
            start_params=start_params,
            method=st["method"],
            maxiter=st["maxiter"],
            cov_type="none",
            low_memory=True,
            disp=False,
        )
        res = model.filter(fit.params, cov_type=st["cov_type"], conserve_memory=LEAN)
    res.mle_retvals_fit = fit.mle_retvals
    return model, res


def _lean() -> int:
    """statsmodels' memory flags that keep the likelihood and the forecast
    means and drop every stored covariance, gain and smoothing array: at
    ``s = 168`` (170 states) the default results store ``n r^2`` floats per
    covariance array, 2.3 GB each at 10,000 rows, and four fits at once ran
    out of the machine's 15 GB. ``fit(low_memory=True, cov_type="none")`` then
    ``filter(params, cov_type="opg", conserve_memory=LEAN)`` gives the default
    path's parameters, standard errors, log-likelihood and forecasts to the
    bit (checked on four ``s = 7`` fits, part 2)."""
    from statsmodels.tsa.statespace import kalman_filter as kf

    return (
        kf.MEMORY_NO_PREDICTED_COV
        | kf.MEMORY_NO_FILTERED_COV
        | kf.MEMORY_NO_GAIN
        | kf.MEMORY_NO_SMOOTHING
        | kf.MEMORY_NO_STD_FORECAST
        | kf.MEMORY_NO_FORECAST_COV
    )


def batch_window(spec: Spec) -> int:
    """The batch start's buffer: 500 differenced rows (study 485's), or three
    seasons when that is longer."""
    return max(500, 3 * spec.s)


def sarima_variants(spec: Spec, plain_only: bool) -> dict[str, dict]:
    """The online forms run on each case: ``""`` the plain start (§3), and
    ``"batch."`` the batch start of ``batch_window(spec)`` rows."""
    out: dict[str, dict] = {"": {}}
    if not plain_only:
        out["batch."] = {"batch_start": batch_window(spec)}
    return out


def sarima_case(
    spec: Spec,
    seed: int,
    n: int = N_ROWS,
    tol0: bool = True,
    start: str = "default",
    plain_only: bool = True,
) -> dict:
    """One series: each online form against SARIMAX on the same rows. Field
    names of the plain form are bare; the batch start's carry ``batch.``."""
    y = simulate_sarima(spec, n, seed)
    runs = {}
    for pre, kw in sarima_variants(spec, plain_only).items():
        t0 = time.perf_counter()
        est = OnlineSarima(spec, **kw)
        pred = np.array([est.update(float(v)) for v in y])
        runs[pre] = (est, pred, time.perf_counter() - t0)
    t0 = time.perf_counter()
    # start="online": SARIMAX's L-BFGS starts from the plain online estimate
    sp0 = runs[""][0].params if start == "online" else None
    model, res = sarimax_fit(spec, y, start_params=sp0)
    t_batch = time.perf_counter() - t0
    b = np.asarray(res.params, dtype=float)
    se = np.asarray(res.bse, dtype=float)
    L = runs[""][0].ry.n
    w = np.asarray(model.endog, dtype=float).ravel()
    # the fixed-parameter recursion at the batch parameters against SARIMAX's filter
    fres = model.filter(b, conserve_memory=LEAN)
    fc = np.asarray(fres.forecasts, dtype=float).ravel()
    mine = fixed_predictions(spec, b[:-1], w)
    tail_w = slice(w.shape[0] - 1000, w.shape[0])
    # and against the filter with its steady-state switch off (tolerance 0):
    # statsmodels stops updating P once it has converged to 1e-19, which costs
    # about 1e-9 in the forecasts; with the switch off the filter runs O(r^3)
    # on every row
    gap_tol0 = math.nan
    if tol0:
        tol_saved = model.ssm.tolerance
        model.ssm.tolerance = 0.0
        fc0 = np.asarray(model.filter(b, conserve_memory=LEAN).forecasts, dtype=float).ravel()
        model.ssm.tolerance = tol_saved
        gap_tol0 = float(np.max(np.abs(mine[tail_w] - fc0[tail_w])))
    base = y[L:] - w  # y_t - w_t
    yhat_b = base + fc
    half = slice(n // 2, n)
    yb = np.concatenate([np.full(L, math.nan), yhat_b])[half]
    out = {
        "spec": spec.name,
        "seed": seed,
        "n": n,
        "sm_converged": bool(res.mle_retvals_fit.get("converged", False)),
        "sm_iterations": int(res.mle_retvals_fit.get("iterations", -1)),
        "sm_start": start,
        "llf_batch": float(res.llf),
        "filter_gap_last1000": float(np.max(np.abs(mine[tail_w] - fc[tail_w]))),
        "filter_gap_tol0_last1000": gap_tol0,
        "w_abs_max_last1000": float(np.max(np.abs(w[tail_w]))),
        "t_batch_s": t_batch,
    }
    for i, nm in enumerate(spec.names):
        out[f"{nm}_batch"] = float(b[i])
        out[f"{nm}_se"] = float(se[i])
    for pre, (est, pred, t_online) in runs.items():
        o = est.params
        yh = pred[half]
        out[pre + "silent_rows"] = int(np.sum(np.isnan(pred)))
        out[pre + "silent_explained"] = int(
            sum(1 for i in range(n) if math.isnan(pred[i]) and i in est.reasons)
        )
        out[pre + "llf_online_params"] = float(model.loglike(o)) if o[-1] > 0 else math.nan
        out[pre + "pred_rms_gap_second_half"] = float(np.sqrt(np.mean((yh - yb) ** 2)))
        out[pre + "online_mse_second_half"] = float(np.mean((y[half] - yh) ** 2))
        out[pre + "batch_mse_second_half"] = float(np.mean((y[half] - yb) ** 2))
        out[pre + "t_online_s"] = t_online
        out[pre + "state_bytes"] = est.state_bytes()
        if est.batch_fit is not None:
            out[pre + "start_fit_stop"] = est.batch_fit.stop
            out[pre + "start_fit_iterations"] = est.batch_fit.iterations
        for i, nm in enumerate(spec.names):
            out[f"{pre}{nm}_online"] = float(o[i])
            out[f"{pre}{nm}_gap_se"] = float((o[i] - b[i]) / se[i])
        out[pre + "max_abs_gap_se"] = float(max(abs(out[f"{pre}{nm}_gap_se"]) for nm in spec.names))
    return out


# --------------------------------------------------------------------------
# the grid runner (resumable, one JSON line per case)


def _load(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _worker(args):  # noqa: ANN001, ANN202
    fn, a, kw = args
    return fn(*a, **kw)


def run_grid(path: str, fn, cases: list[tuple], key, workers: int = 4) -> list[dict]:  # noqa: ANN001
    """Run ``fn(*case)`` for each case not yet in ``path``; append each result
    as it arrives. ``key(row)`` and ``key(case)`` identify a case."""
    done = {key(r) for r in _load(path)}
    todo = [c for c in cases if key(c) not in done]
    if todo:
        print(f"{path}: {len(done)} cached, {len(todo)} to run", flush=True)
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        from multiprocessing import get_context

        with get_context("fork").Pool(workers) as pool, open(path, "a") as fh:
            for res in pool.imap_unordered(_worker, [(fn, c, {}) for c in todo]):
                fh.write(json.dumps(res) + "\n")
                fh.flush()
    return _load(path)


def summarise_sarima(rows: list[dict], spec: Spec, pre: str = "") -> dict:
    rows = [r for r in rows if r["spec"] == spec.name and pre + "max_abs_gap_se" in r]
    out: dict = {"spec": spec.name, "variant": pre or "plain.", "cases": len(rows)}
    if not rows:
        return out
    for nm in spec.names:
        g = np.abs([r[f"{pre}{nm}_gap_se"] for r in rows])
        out[nm] = {
            "median": float(np.median(g)),
            "max": float(np.max(g)),
            "over2": int(np.sum(g > 2)),
            "worst_seed": int(rows[int(np.argmax(g))]["seed"]),
            "worst_signed": float(rows[int(np.argmax(g))][f"{pre}{nm}_gap_se"]),
        }
    out["all_within_2"] = int(sum(1 for r in rows if r[pre + "max_abs_gap_se"] <= 2.0))
    out["sm_converged"] = int(sum(r["sm_converged"] for r in rows))
    out["sm_iterations_max"] = int(max(r["sm_iterations"] for r in rows))
    for f in (
        "filter_gap_last1000",
        "filter_gap_tol0_last1000",
        pre + "pred_rms_gap_second_half",
        pre + "t_online_s",
        "t_batch_s",
    ):
        v = np.array([r[f] for r in rows if f in r and not math.isnan(r[f])])
        out[f.removeprefix(pre)] = {
            "median": float(np.median(v)) if v.size else math.nan,
            "max": float(np.max(v)) if v.size else math.nan,
            "cases": int(v.size),
        }
    ratio = np.array(
        [r[pre + "online_mse_second_half"] / r[pre + "batch_mse_second_half"] for r in rows]
    )
    out["mse_ratio_second_half"] = {"median": float(np.median(ratio)), "max": float(np.max(ratio))}
    dll = np.array([r["llf_batch"] - r[pre + "llf_online_params"] for r in rows])
    out["llf_gap_total"] = {
        "median": float(np.median(dll)),
        "max": float(np.max(dll)),
        "min": float(np.min(dll)),
    }
    out["silent_rows"] = sorted({r[pre + "silent_rows"] for r in rows})
    out["silent_unexplained"] = int(
        sum(r[pre + "silent_rows"] - r[pre + "silent_explained"] for r in rows)
    )
    out["state_bytes"] = rows[0][pre + "state_bytes"]
    if pre + "start_fit_stop" in rows[0]:
        out["start_fits"] = {
            "converged": int(sum(r[pre + "start_fit_stop"] == "converged" for r in rows)),
            "iterations_max": int(max(r[pre + "start_fit_iterations"] for r in rows)),
        }
    return out


def print_sarima(summ: dict, spec: Spec) -> None:
    print(f"== {spec.name}: {spec.label} [{summ.get('variant', 'plain.')[:-1]}]")
    if not summ["cases"]:
        print("   no cases yet")
        return
    if "start_fits" in summ:
        print(f"   start fits: {summ['start_fits']}")
    print(f"   truth {spec.truth}; cases {summ['cases']}; state {summ['state_bytes']} B")
    for nm in spec.names:
        g = summ[nm]
        print(
            f"   {nm:10s} |gap| in SE: median {g['median']:.3g}  max {g['max']:.3g} "
            f"(seed {g['worst_seed']}, {g['worst_signed']:+.3g})  over 2: {g['over2']}"
        )
    print(f"   all within 2 SE: {summ['all_within_2']} of {summ['cases']}")
    print(
        f"   SARIMAX converged {summ['sm_converged']}, iterations max {summ['sm_iterations_max']}; "
        f"filter gap last 1000 rows median {summ['filter_gap_last1000']['median']:.3g} "
        f"max {summ['filter_gap_last1000']['max']:.3g}; with the steady-state switch off "
        f"median {summ['filter_gap_tol0_last1000']['median']:.3g} "
        f"max {summ['filter_gap_tol0_last1000']['max']:.3g} "
        f"({summ['filter_gap_tol0_last1000']['cases']} cases)"
    )
    pg, mr, lg = (
        summ[k] for k in ("pred_rms_gap_second_half", "mse_ratio_second_half", "llf_gap_total")
    )
    print(
        f"   pred RMS gap 2nd half median {pg['median']:.3g} max {pg['max']:.3g}; "
        f"MSE ratio median {mr['median']:.5f} max {mr['max']:.5f}; "
        f"llf gap median {lg['median']:.3g} max {lg['max']:.3g}"
    )
    to, tb = summ["t_online_s"], summ["t_batch_s"]
    print(
        f"   silent rows {summ['silent_rows']}, unexplained {summ['silent_unexplained']}; "
        f"seconds online {to['median']:.2f}, SARIMAX {tb['median']:.2f} (max {tb['max']:.2f})"
    )


# --------------------------------------------------------------------------
# checks: gradient, out-of-sample, scale, admissibility, the state count


def sarima_checks(spec: Spec, seed: int = 487) -> dict:
    """Held on one series: the carried gradient equals the derivative of the
    zero-start conditional recursion (central difference); the prediction of
    row t does not move when row t changes; scaling y by c leaves the
    coefficients and scales s2 by c^2; every row's coefficients are admissible;
    the allocated floats equal the state count."""
    y = simulate_sarima(spec, 3000, seed)
    out: dict = {"spec": spec.name}
    # gradient at fixed parameters: run the estimator's recursion with steps off
    est = OnlineSarima(spec, warm=10**9)
    beta = np.asarray(spec.truth[:-1], dtype=float) * 0.9
    est.beta = beta.copy()
    est._set_polys()
    L = est.ry.n
    w = y[L:].copy()
    for k, dk in est.delta:
        w -= dk * y[L - k : y.shape[0] - k]
    g = np.zeros(spec.n_params)
    sse = 0.0
    for v in w:
        what = est._what()
        e = float(v) - what
        psi = est._psi()
        g += psi * e
        sse += e * e
        est._advance(float(v), e)

    def css(b: np.ndarray) -> float:
        wh = fixed_predictions(spec, b, w)
        return float(np.sum((w - wh) ** 2))

    assert abs(sse - css(beta)) <= 1e-9 * sse
    num = np.zeros_like(g)
    for j in range(spec.n_params):
        h = 1e-6
        bp = beta.copy()
        bm = beta.copy()
        bp[j] += h
        bm[j] -= h
        num[j] = -(css(bp) - css(bm)) / (4 * h)  # -d(1/2 sse)/d beta = sum psi e
    out["gradient_rel_err"] = float(np.max(np.abs(g - num)) / np.max(np.abs(num)))
    # out-of-sample
    a = OnlineSarima(spec)
    b2 = OnlineSarima(spec)
    t_change = 1500
    moved = 0.0
    for t, v in enumerate(y):
        pa = a.update(float(v))
        pb = b2.update(float(v) + (100.0 if t == t_change else 0.0))
        if t <= t_change and not (math.isnan(pa) and math.isnan(pb)):
            moved = max(moved, abs(pa - pb))
    out["out_of_sample_max_move_up_to_changed_row"] = moved
    # scale equivariance and admissibility
    c = 37.0
    e1 = OnlineSarima(spec)
    e2 = OnlineSarima(spec)
    worst = 0.0
    for v in y:
        e1.update(float(v))
        e2.update(c * float(v))
        worst = max(worst, float(np.max(np.abs(e1.beta))))
    p1, p2 = e1.params, e2.params
    out["scale_coef_rel"] = float(
        np.max(np.abs(p1[:-1] - p2[:-1]) / np.maximum(np.abs(p1[:-1]), 1e-12))
    )
    out["scale_s2_rel"] = float(abs(p2[-1] / (c * c) - p1[-1]) / p1[-1])
    out["max_abs_coef_any_row"] = worst
    out["allocated_floats"] = e1.allocated_floats()
    out["state_floats"] = sum(state_floats(spec).values())
    return out


# --------------------------------------------------------------------------
# parts


SPECS = {
    "s7_full": Spec("s7_full", 1, 1, 1, 1, 1, 1, 7, (0.5, 0.3, -0.3, -0.5, 1.0)),
    "s7_airline": Spec("s7_airline", 0, 1, 1, 0, 1, 1, 7, (-0.4, -0.6, 1.0)),
    "s24_full": Spec("s24_full", 1, 1, 1, 1, 1, 1, 24, (0.5, 0.3, -0.3, -0.5, 1.0)),
    "s24_airline": Spec("s24_airline", 0, 1, 1, 0, 1, 1, 24, (-0.4, -0.6, 1.0)),
    "s168_full": Spec("s168_full", 1, 1, 1, 1, 1, 1, 168, (0.5, 0.3, -0.3, -0.5, 1.0)),
    "s168_airline": Spec("s168_airline", 0, 1, 1, 0, 1, 1, 168, (-0.4, -0.6, 1.0)),
}


def part1(args) -> dict:  # noqa: ANN001
    res: dict = {"sarimax_settings": SARIMAX_SETTINGS, "seeds": [SEEDS[0], SEEDS[-1]], "n": N_ROWS}
    res["checks"] = [sarima_checks(SPECS[k]) for k in ("s7_full", "s7_airline")]
    path = os.path.join(args.out, "sarima.jsonl")
    cases = [(SPECS[k], seed) for k in ("s7_full", "s7_airline") for seed in SEEDS[: args.seeds]]
    rows = run_grid(
        path,
        sarima_case,
        cases,
        key=lambda x: (x["spec"], x["seed"]) if isinstance(x, dict) else (x[0].name, x[1]),
        workers=args.workers,
    )
    res["summary"] = [summarise_sarima(rows, SPECS[k]) for k in ("s7_full", "s7_airline")]
    return res


P2_SPECS = ("s24_full", "s24_airline", "s168_airline", "s168_full")
TABLE_S = (4, 7, 12, 24, 52, 96, 168, 336, 672, 1440, 8760)
TABLE_FORMS = {
    "(0,1,1)(0,1,1,s) airline": (0, 1, 1, 0, 1, 1),
    "(1,1,1)(1,1,1,s)": (1, 1, 1, 1, 1, 1),
    "(1,0,1)(1,0,1,s)": (1, 0, 1, 1, 0, 1),
    "(1,1,1)(1,1,0,s)": (1, 1, 1, 1, 1, 0),
}


def state_table() -> list[dict]:
    """The estimator's state in bytes against ``s``: each ring, ``beta``,
    ``R`` and the scalars, checked against the floats the implementation
    allocates; beside it the batch start's buffer (held only until the start
    is fitted) and the exact Kalman filter's state and covariance
    ``8 (r + r(r+1)/2)``, ``r = max(p + sP, q + sQ + 1)`` (scan §3), which the
    fixed-parameter oracle runs."""
    out = []
    for form, (p, d, q, P, D, Q) in TABLE_FORMS.items():
        for s in TABLE_S:
            n = p + q + P + Q
            spec = Spec("t", p, d, q, P, D, Q, s, tuple([0.1] * n + [1.0]))
            parts = state_floats(spec)
            est = OnlineSarima(spec)
            assert est.allocated_floats() == sum(parts.values())
            r = max(p + s * P, q + s * Q + 1)
            out.append(
                {
                    "form": form,
                    "s": s,
                    "rings_floats": {k: v for k, v in parts.items() if k in ring_lengths(spec)},
                    "ring_floats_total": sum(ring_lengths(spec).values()),
                    "other_floats": sum(parts.values()) - sum(ring_lengths(spec).values()),
                    "state_bytes": 8 * sum(parts.values()),
                    "batch_start_buffer_bytes": 8 * batch_window(spec),
                    "kalman_r": r,
                    "kalman_bytes": 8 * (r + r * (r + 1) // 2),
                }
            )
    return out


def part2(args) -> dict:  # noqa: ANN001
    res: dict = {
        "sarimax_settings": SARIMAX_SETTINGS,
        "seeds": [SEEDS[0], SEEDS[-1]],
        "n": N_ROWS,
        "batch_window": {k: batch_window(SPECS[k]) for k in P2_SPECS},
    }
    res["state_table"] = state_table()
    path = os.path.join(args.out, "sarima.jsonl")
    cases = [(SPECS[k], seed) for k in P2_SPECS for seed in SEEDS[: args.seeds]]
    key = lambda x: (x["spec"], x["seed"]) if isinstance(x, dict) else (x[0].name, x[1])  # noqa: E731
    if args.no_run:
        rows = _load(path)
    else:
        rows = run_grid(path, sarima_case_p2, cases, key=key, workers=args.workers)
    res["summary"] = [
        summarise_sarima(rows, SPECS[k], pre) for k in P2_SPECS for pre in ("", "batch.")
    ]
    have = {key(r) for r in rows}
    res["complete"] = all(key(c) in have for c in cases)
    return res


def sarima_case_p2(spec: Spec, seed: int) -> dict:
    return sarima_case(spec, seed, plain_only=False)


def summarise2(res: dict) -> None:
    print("SARIMAX settings:", res["sarimax_settings"])
    print("batch windows:", res["batch_window"], "complete:", res["complete"])
    print("state table (bytes):")
    for row in res["state_table"]:
        print(
            f"   {row['form']:28s} s={row['s']:5d} rings {row['ring_floats_total']:6d} floats "
            f"+ {row['other_floats']:3d}: {row['state_bytes']:7d} B; start buffer "
            f"{row['batch_start_buffer_bytes']:6d} B; Kalman r={row['kalman_r']:5d} "
            f"{row['kalman_bytes']:10d} B"
        )
    for summ in res["summary"]:
        print_sarima(summ, SPECS[summ["spec"]])


def summarise1(res: dict) -> None:
    print("SARIMAX settings:", res["sarimax_settings"])
    for c in res["checks"]:
        print("checks", c)
    for summ in res["summary"]:
        print_sarima(summ, SPECS[summ["spec"]])


# --------------------------------------------------------------------------
# 488: seasonal Holt-Winters ETS, additive error (Hyndman et al. 2008, Table 2.2)

EPS = 1e-4  # statsmodels' margin on alpha, beta/alpha and gamma/(1 - alpha)
PHI_BOUNDS = (0.8, 0.98)  # statsmodels' default bounds on the damping
ETS_L0, ETS_B0 = 10.0, 0.1


def ets_initial_seasonal(m: int) -> np.ndarray:
    """The known initial seasonal states ``s_{-m}, ..., s_{-1}`` in time order:
    a sine of amplitude 2 over the season (study 485's E3), summing to zero."""
    j = np.arange(1, m + 1)
    return 2.0 * np.sin(2.0 * math.pi * j / m)


@dataclass(frozen=True)
class EtsSpec:
    """ETS(A,A,A) or ETS(A,Ad,A) with season ``m`` and its truth
    ``(alpha, beta, gamma[, phi], sigma2)`` in ``ETSModel``'s parameterization
    (``beta`` moves the trend by ``beta e``, not the classical ``beta* = beta/alpha``;
    ``gamma`` moves the season by ``gamma e``)."""

    name: str
    damped: bool
    m: int
    truth: tuple[float, ...]
    seasonal: str = "add"

    @property
    def n_params(self) -> int:
        return 4 if self.damped else 3

    @property
    def label(self) -> str:
        t = "Ad" if self.damped else "A"
        s = "M" if self.seasonal == "mul" else "A"
        return f"ETS(A,{t},{s}), m = {self.m}"

    @property
    def names(self) -> list[str]:
        out = ["smoothing_level", "smoothing_trend", "smoothing_seasonal"]
        if self.damped:
            out.append("damping_trend")
        return out + ["sigma2"]


def ets_project(spec: EtsSpec, th: np.ndarray) -> np.ndarray:
    """statsmodels' admissible box: ``alpha`` in ``[eps, 1 - eps]``,
    ``beta/alpha`` and ``gamma/(1 - alpha)`` in ``[eps, 1 - eps]``, ``phi`` in
    ``[0.8, 0.98]``."""
    out = np.array(th, dtype=float)
    a = min(max(out[0], EPS), 1.0 - EPS)
    out[0] = a
    out[1] = min(max(out[1], EPS * a), (1.0 - EPS) * a)
    out[2] = min(max(out[2], EPS * (1.0 - a)), (1.0 - EPS) * (1.0 - a))
    if spec.damped:
        out[3] = min(max(out[3], PHI_BOUNDS[0]), PHI_BOUNDS[1])
    return out


def ets_state_floats(spec: EtsSpec) -> dict[str, int]:
    """The estimator's state in floats: level, trend and the season ring,
    their sensitivities to the ``k`` parameters, ``theta``, ``R`` (distinct),
    ``s2``, ``W``, the row count and the ring position."""
    k, m = spec.n_params, spec.m
    return {
        "level, trend": 2,
        "season ring": m,
        "sensitivities of level and trend": 2 * k,
        "sensitivities of the season ring": m * k,
        "theta": k,
        "R (distinct)": k * (k + 1) // 2,
        "scalars (s2, W, t, position)": 4,
    }


class OnlineEts:
    """The recursive prediction-error estimator of ``(alpha, beta, gamma[,
    phi])`` with the initial states known. ``update(y)`` returns row ``t``'s
    prediction, made before ``y`` is read.

    Additive seasonality::

        yhat_t = l + phi b + s_{t-m},          e_t = y_t - yhat_t
        l <- l + phi b + alpha e,   b <- phi b + beta e,   s_t = s_{t-m} + gamma e

    ``phi = 1`` without damping. The sensitivities ``D. = d(.)/d theta`` at the
    current parameters, with ``psi_t = d yhat_t / d theta = Dl + phi Db
    + b u_phi + Ds_{t-m}`` (``u_x`` the unit vector of parameter ``x``)::

        Dl <- Dl + phi Db + b u_phi + e u_alpha - alpha psi
        Db <- phi Db + b u_phi + e u_beta - beta psi
        Ds_t = Ds_{t-m} + e u_gamma - gamma psi

    and one Gauss-Newton step per row, gain ``1/W``, projected onto
    statsmodels' box (``ets_project``). The states move at the parameters held
    before the step. Multiplicative seasonality (part 4) is in ``_row``.
    """

    def __init__(self, spec: EtsSpec, warm: int | None = None, batch_start: int = 0) -> None:
        self.spec = spec
        k, m = spec.n_params, spec.m
        self.warm = warm if warm is not None else max(50, 2 * m)
        self.batch_start = batch_start
        self.started = not batch_start
        self.buffer: list[float] = []
        self.batch_fit = None
        self.reasons: dict[int, str] = {}
        self.l, self.b = ETS_L0, ETS_B0
        self.s = ets_initial_seasonal_for(spec)
        self.Dl = np.zeros(k)
        self.Db = np.zeros(k)
        self.Ds = np.zeros((m, k))
        self.pos = 0
        start = [0.1, 0.01, 0.01] + ([0.98] if spec.damped else [])
        self.theta = ets_project(spec, np.array(start))  # statsmodels' default start
        self.R = np.zeros((k, k))
        self.s2 = 0.0
        self.W = 0.0
        self.t = 0

    def _psi(self) -> np.ndarray:
        sp = self.spec
        phi = self.theta[3] if sp.damped else 1.0
        if sp.seasonal == "mul":
            sv = self.s[self.pos]
            base = self.l + phi * self.b
            psi = (self.Dl + phi * self.Db) * sv + base * self.Ds[self.pos]
            if sp.damped:
                psi[3] += self.b * sv
            return psi
        psi = self.Dl + phi * self.Db + self.Ds[self.pos]
        if sp.damped:
            psi[3] += self.b
        return psi

    def predict(self) -> float:
        sp = self.spec
        phi = self.theta[3] if sp.damped else 1.0
        if sp.seasonal == "mul":
            return (self.l + phi * self.b) * self.s[self.pos]
        return self.l + phi * self.b + self.s[self.pos]

    def update(self, y: float) -> float:
        row = self.t
        if not self.started:
            self.t += 1
            self.buffer.append(y)
            self.reasons[row] = (
                f"buffering {self.batch_start} rows for the batch start (fitted by "
                "the core's optimizer); no parameters yet"
            )
            if len(self.buffer) == self.batch_start:
                self._start()
            return math.nan
        pred = self.predict()
        self._row(y, step=True)
        return pred

    def _start(self) -> None:
        """Remedy (a) of study 485: fit the buffer by the core-held optimizer,
        then replay it from the known initial states at the fitted parameters,
        held, building the states, their sensitivities, ``R`` and ``s2``."""
        rows = np.asarray(self.buffer)
        self.batch_fit = ets_css_fit(self.spec, rows)
        self.theta = ets_project(self.spec, self.batch_fit.theta)
        for v in rows:
            self._row(float(v), step=False, replay=True)
        self.buffer = []
        self.started = True

    def _row(self, y: float, step: bool, replay: bool = False) -> None:
        sp = self.spec
        k = sp.n_params
        th = self.theta
        a, bt, g = th[0], th[1], th[2]
        phi = th[3] if sp.damped else 1.0
        yhat = self.predict()
        psi = self._psi()
        e = y - yhat
        if not replay:
            self.t += 1
        if True:  # R and s2 accumulate on every row, the warm rows included
            self.W += 1.0
            w = 1.0 / self.W
            self.R += w * (np.outer(psi, psi) - self.R)
            self.s2 += w * (e * e - self.s2)
        sv = self.s[self.pos]
        Dsv = self.Ds[self.pos]
        u = np.eye(k)
        lb = self.l + phi * self.b
        Dlb = self.Dl + phi * self.Db
        if sp.damped:
            Dlb = Dlb + self.b * u[3]
        if sp.seasonal == "mul":
            # e = y - (l + phi b) s;  l <- lb + alpha e / s;  b <- phi b + beta e / s;
            # s_t = s_{t-m} + gamma e / (l + phi b)   (Hyndman et al. 2008, Table 2.2)
            de = -psi
            nl = lb + a * e / sv
            Dnl = Dlb + (e / sv) * u[0] + a * (de / sv - e * Dsv / (sv * sv))
            nb = phi * self.b + bt * e / sv
            Dnb = phi * self.Db + (e / sv) * u[1] + bt * (de / sv - e * Dsv / (sv * sv))
            if sp.damped:
                Dnb = Dnb + self.b * u[3]
            ns = sv + g * e / lb
            Dns = Dsv + (e / lb) * u[2] + g * (de / lb - e * Dlb / (lb * lb))
        else:
            nl = lb + a * e
            Dnl = Dlb + e * u[0] - a * psi
            nb = phi * self.b + bt * e
            Dnb = phi * self.Db + e * u[1] - bt * psi
            if sp.damped:
                Dnb = Dnb + self.b * u[3]
            ns = sv + g * e
            Dns = Dsv + e * u[2] - g * psi
        self.l, self.b = nl, nb
        self.Dl, self.Db = Dnl, Dnb
        self.s[self.pos] = ns
        self.Ds[self.pos] = Dns
        self.pos = (self.pos + 1) % sp.m
        if step and not self.batch_start and self.t <= self.warm:
            return
        if step:
            stepv = (1.0 / self.W) * np.linalg.solve(self.R, psi * e)
            self.theta = ets_project(sp, self.theta + stepv)

    @property
    def params(self) -> np.ndarray:
        return np.concatenate([self.theta, [self.s2]])

    def state_bytes(self) -> int:
        return 8 * sum(ets_state_floats(self.spec).values())

    def allocated_floats(self) -> int:
        k = self.spec.n_params
        return (
            2 + self.s.size + self.Dl.size + self.Db.size + self.Ds.size + k + k * (k + 1) // 2 + 4
        )


def ets_initial_seasonal_for(spec: EtsSpec) -> np.ndarray:
    s = ets_initial_seasonal(spec.m)
    if spec.seasonal == "mul":
        return 1.0 + 0.2 * s / 2.0  # factors 0.8 to 1.2, mean 1
    return s


def ets_fixed(spec: EtsSpec, theta: np.ndarray, y: np.ndarray) -> np.ndarray:
    """One-step predictions at fixed parameters from the known initial states."""
    est = OnlineEts(spec)
    est.theta = np.array(theta, dtype=float)
    out = np.empty(y.shape[0])
    for t, v in enumerate(y):
        out[t] = est.predict()
        est._row(float(v), step=False)
    return out


def ets_objective(spec: EtsSpec, y: np.ndarray):  # noqa: ANN201
    """``-1/2 sum e_t^2`` over ``y`` from the known initial states, gradient
    ``sum psi_t e_t``, Gauss-Newton ``sum psi_t psi_t'``: statsmodels'
    concentrated likelihood of the additive-error forms is monotone in it."""

    def f(theta: np.ndarray, grad: bool):  # noqa: ANN202
        est = OnlineEts(spec)
        est.theta = np.array(theta, dtype=float)
        k = spec.n_params
        g = np.zeros(k)
        H = np.zeros((k, k))
        sse = 0.0
        for v in y:
            yhat = est.predict()
            e = float(v) - yhat
            sse += e * e
            if grad:
                psi = est._psi()
                g += psi * e
                H += np.outer(psi, psi)
            est._row(float(v), step=False)
            if spec.seasonal == "mul" and not (est.l + est.b > 0.0 and est.s.min() > 0.0):
                return -math.inf, g, H
        if not math.isfinite(sse):
            return -math.inf, g, H
        return -0.5 * sse, g, H

    return f


def ets_grid(spec: EtsSpec) -> list[tuple]:
    import itertools

    pts = []
    for a, bs, gs in itertools.product((0.1, 0.3, 0.6), (0.05, 0.3), (0.05, 0.3)):
        base = (a, a * bs, (1.0 - a) * gs)
        if spec.damped:
            pts += [base + (phi,) for phi in (0.85, 0.95)]
        else:
            pts.append(base)
    return pts


def ets_css_fit(spec: EtsSpec, y: np.ndarray):  # noqa: ANN201
    """The batch start's fit by ``batch_core.maximize`` (study 485's core-held
    projected Newton), the projection ``ets_project``, the start grid
    ``alpha in {0.1, 0.3, 0.6}`` x ``beta/alpha in {0.05, 0.3}`` x
    ``gamma/(1 - alpha) in {0.05, 0.3}`` (x ``phi in {0.85, 0.95}``)."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "study_485"))
    import batch_core

    return batch_core.maximize(
        ets_objective(spec, y), lambda th: ets_project(spec, th), ets_grid(spec)
    )


def simulate_ets(spec: EtsSpec, n: int, seed: int) -> np.ndarray:
    """``n`` rows from the model at its truth, from the known initial states;
    Gaussian errors (scaled by the level for nothing: additive error)."""
    rng = np.random.default_rng(seed)
    sigma = math.sqrt(spec.truth[-1])
    est = OnlineEts(spec)
    est.theta = np.array(spec.truth[:-1], dtype=float)
    y = np.empty(n)
    eps = sigma * rng.standard_normal(n)
    for t in range(n):
        y[t] = est.predict() + eps[t]
        est._row(float(y[t]), step=False)
    return y


def etsmodel_fit(spec: EtsSpec, y: np.ndarray):  # noqa: ANN201
    """The oracle: ``ETSModel(y, error="add", trend="add", damped_trend=...,
    seasonal=..., seasonal_periods=m, initialization_method="known",
    initial_level=10, initial_trend=0.1, initial_seasonal=...)``,
    ``fit(disp=False)`` with its defaults (L-BFGS-B on the concentrated
    likelihood, ``beta/alpha`` and ``gamma/(1 - alpha)`` fitted in
    ``[1e-4, 1 - 1e-4]``, ``phi`` in ``[0.8, 0.98]``, 1,000 iterations at
    most; ``bse`` from the numerical Hessian, ``cov_type="approx"``)."""
    import warnings

    from statsmodels.tsa.exponential_smoothing.ets import ETSModel

    model = ETSModel(
        y,
        error="add",
        trend="add",
        damped_trend=spec.damped,
        seasonal=spec.seasonal,
        seasonal_periods=spec.m,
        initialization_method="known",
        initial_level=ETS_L0,
        initial_trend=ETS_B0,
        initial_seasonal=ets_initial_seasonal_for(spec),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = model.fit(disp=False)
    return model, res


def ets_window(spec: EtsSpec) -> int:
    return max(500, 3 * spec.m)


def ets_case(spec: EtsSpec, seed: int, n: int = N_ROWS) -> dict:
    """One series: the plain and the batch start against ``ETSModel``. The
    gaps of ``alpha, beta, gamma, phi`` are in ``ETSModel``'s own SEs; for
    ``sigma2`` (concentrated out by statsmodels) in the Gaussian closed form
    ``mse sqrt(2/n)`` around ``ETSResults.mse``."""
    y = simulate_ets(spec, n, seed)
    runs = {}
    for pre, kw in (("", {}), ("batch.", {"batch_start": ets_window(spec)})):
        t0 = time.perf_counter()
        est = OnlineEts(spec, **kw)
        pred = np.array([est.update(float(v)) for v in y])
        runs[pre] = (est, pred, time.perf_counter() - t0)
    t0 = time.perf_counter()
    model, res = etsmodel_fit(spec, y)
    t_batch = time.perf_counter() - t0
    b = np.concatenate([np.asarray(res.params, dtype=float), [float(res.mse)]])
    se = np.concatenate([np.asarray(res.bse, dtype=float), [float(res.mse) * math.sqrt(2.0 / n)]])
    fitted = np.asarray(res.fittedvalues, dtype=float)
    mine = ets_fixed(spec, b[:-1], y)
    diff = np.abs(mine - fitted)
    # in ulps of the largest |prediction| (a per-row ulp is meaningless where
    # the seasonal prediction crosses zero)
    ulps = diff / np.spacing(np.abs(fitted).max())
    half = slice(n // 2, n)
    out = {
        "spec": spec.name,
        "seed": seed,
        "n": n,
        "sm_converged": bool(res.mle_retvals.get("converged", False)),
        "sm_iterations": int(res.mle_retvals.get("iterations", -1)),
        "llf_batch": float(res.llf),
        "fixed_gap_abs_max": float(diff.max()),
        "fixed_gap_rel_max": float((diff / np.abs(fitted)).max()),
        "fixed_gap_ulps_max": float(ulps.max()),
        "fixed_gap_ulps_median": float(np.median(ulps)),
        "abs_fitted_max": float(np.abs(fitted).max()),
        "t_batch_s": t_batch,
    }
    for i, nm in enumerate(spec.names):
        out[f"{nm}_batch"] = float(b[i])
        out[f"{nm}_se"] = float(se[i])
    for pre, (est, pred, t_online) in runs.items():
        o = est.params
        yh = pred[half]
        out[pre + "silent_rows"] = int(np.sum(np.isnan(pred)))
        out[pre + "silent_explained"] = int(
            sum(1 for i in range(n) if math.isnan(pred[i]) and i in est.reasons)
        )
        out[pre + "llf_online_params"] = float(model.loglike(o[:-1]))
        out[pre + "pred_rms_gap_second_half"] = float(np.sqrt(np.mean((yh - fitted[half]) ** 2)))
        out[pre + "online_mse_second_half"] = float(np.mean((y[half] - yh) ** 2))
        out[pre + "batch_mse_second_half"] = float(np.mean((y[half] - fitted[half]) ** 2))
        out[pre + "t_online_s"] = t_online
        out[pre + "state_bytes"] = est.state_bytes()
        if est.batch_fit is not None:
            out[pre + "start_fit_stop"] = est.batch_fit.stop
            out[pre + "start_fit_iterations"] = est.batch_fit.iterations
        for i, nm in enumerate(spec.names):
            out[f"{pre}{nm}_online"] = float(o[i])
            out[f"{pre}{nm}_gap_se"] = float((o[i] - b[i]) / se[i])
        out[pre + "max_abs_gap_se"] = float(max(abs(out[f"{pre}{nm}_gap_se"]) for nm in spec.names))
    return out


ETS_TRUTHS = {
    # A: study 485's alpha and beta with gamma 0.1; outside the admissible
    # (forecastable) region at m = 24 and 168 (``ets_radius``)
    "": ((0.3, 0.1, 0.1), 0.9),
    # B: a smaller beta and gamma, inside the region at m = 7 and 24 and within
    # 1e-5 of its edge at m = 168
    "_b": ((0.3, 0.01, 0.05), 0.9),
}
ETS_SPECS: dict[str, EtsSpec] = {}
for _suf, (_tr, _phi) in ETS_TRUTHS.items():
    for _m in (7, 24, 168):
        ETS_SPECS[f"aaa_m{_m}{_suf}"] = EtsSpec(f"aaa_m{_m}{_suf}", False, _m, _tr + (1.0,))
        ETS_SPECS[f"aada_m{_m}{_suf}"] = EtsSpec(f"aada_m{_m}{_suf}", True, _m, _tr + (_phi, 1.0))


def ets_discount(theta: tuple, m: int) -> np.ndarray:
    """The discount matrix ``D = F - g w'`` of the additive-seasonal
    error-correction form (Hyndman et al. 2008, ch. 3 and 10), state
    ``(l, b, s_{t-1}, ..., s_{t-m})``; ``theta = (alpha, beta, gamma, phi)``."""
    a, b, g, phi = theta
    n = 2 + m
    F = np.zeros((n, n))
    F[0, 0], F[0, 1], F[1, 1] = 1.0, phi, phi
    F[2, n - 1] = 1.0
    for i in range(3, n):
        F[i, i - 1] = 1.0
    w = np.zeros(n)
    w[0], w[1], w[n - 1] = 1.0, phi, 1.0
    gv = np.zeros(n)
    gv[0], gv[1], gv[2] = a, b, g
    return F - np.outer(gv, w)


def ets_radius(theta: tuple, m: int) -> float:
    """The largest eigenvalue modulus of ``D`` after the one exact unit root
    (the unobservable direction: level up by c, every seasonal state down by
    c). Below 1 the model is forecastable (the admissible region); above 1 the
    filter's sensitivities to its parameters, and its response to a wrong
    start, grow without bound."""
    ev = np.linalg.eigvals(ets_discount(theta, m))
    ev = np.delete(ev, int(np.argmin(np.abs(ev - 1.0))))
    return float(np.max(np.abs(ev)))


def ets_checks(spec: EtsSpec, seed: int = 488) -> dict:
    """Held on one series of 3,000 rows: the carried gradient against a central
    difference of the sum of squares; the prediction of row t does not move
    when row t changes; scale equivariance (additive forms: states and target
    by 37, parameters unmoved, ``s2`` by ``37^2``) is not run, since the known
    initial states would have to be scaled too; admissibility on every row;
    the allocated floats against the state count."""
    y = simulate_ets(spec, 3000, seed)
    th = np.array(spec.truth[:-1]) * 0.9
    f = ets_objective(spec, y)
    _, g, _ = f(th, True)
    num = np.zeros_like(g)
    for j in range(spec.n_params):
        h = 1e-6
        tp, tm = th.copy(), th.copy()
        tp[j] += h
        tm[j] -= h
        num[j] = (f(tp, False)[0] - f(tm, False)[0]) / (2 * h)
    out = {
        "spec": spec.name,
        "gradient_rel_err": float(np.max(np.abs(g - num)) / np.max(np.abs(num))),
    }
    a, b2 = OnlineEts(spec), OnlineEts(spec)
    moved = 0.0
    worst_adm = 0.0
    for t, v in enumerate(y):
        pa = a.update(float(v))
        pb = b2.update(float(v) + (100.0 if t == 1500 else 0.0))
        if t <= 1500:
            moved = max(moved, abs(pa - pb))
        worst_adm = max(worst_adm, float(np.max(np.abs(ets_project(spec, a.theta) - a.theta))))
    out["out_of_sample_max_move_up_to_changed_row"] = moved
    out["projection_moves_on_any_row"] = worst_adm
    out["allocated_floats"] = a.allocated_floats()
    out["state_floats"] = sum(ets_state_floats(spec).values())
    return out


def ets_state_table(seasonal: str = "add") -> list[dict]:
    out = []
    for damped in (False, True):
        for m in TABLE_S:
            k = 4 if damped else 3
            spec = EtsSpec("t", damped, m, tuple([0.1] * k + [1.0]), seasonal)
            parts = ets_state_floats(spec)
            assert OnlineEts(spec).allocated_floats() == sum(parts.values())
            out.append(
                {
                    "form": spec.label.split(", m =")[0],
                    "m": m,
                    "state_bytes": 8 * sum(parts.values()),
                    "fixed_filter_bytes": 8 * (2 + m),
                    "batch_start_buffer_bytes": 8 * ets_window(spec),
                }
            )
    return out


P3_SPECS = tuple(
    f"{f}_m{m}{suf}" for suf in ("", "_b") for m in (7, 24, 168) for f in ("aaa", "aada")
)


def part3(args) -> dict:  # noqa: ANN001
    res: dict = {"seeds": [SEEDS[0], SEEDS[-1]], "n": N_ROWS}
    res["checks"] = [ets_checks(ETS_SPECS[k]) for k in ("aaa_m7", "aada_m24")]
    res["state_table"] = ets_state_table()
    res["radius"] = {
        k: ets_radius(
            tuple(ETS_SPECS[k].truth[:3])
            + ((ETS_SPECS[k].truth[3],) if ETS_SPECS[k].damped else (1.0,)),
            ETS_SPECS[k].m,
        )
        - 1.0
        for k in P3_SPECS
    }
    path = os.path.join(args.out, "ets.jsonl")
    cases = [(ETS_SPECS[k], seed) for k in P3_SPECS for seed in SEEDS[: args.seeds]]
    key = lambda x: (x["spec"], x["seed"]) if isinstance(x, dict) else (x[0].name, x[1])  # noqa: E731
    rows = (
        _load(path)
        if args.no_run
        else run_grid(path, ets_case, cases, key=key, workers=args.workers)
    )
    res["summary"] = [
        summarise_ets(rows, ETS_SPECS[k], pre) for k in P3_SPECS for pre in ("", "batch.")
    ]
    have = {key(r) for r in rows}
    res["complete"] = all(key(c) in have for c in cases)
    return res


def summarise_ets(rows: list[dict], spec: EtsSpec, pre: str) -> dict:
    out = summarise_sarima(rows, spec, pre)  # type: ignore[arg-type]
    rows = [r for r in rows if r["spec"] == spec.name and pre + "max_abs_gap_se" in r]
    if rows:
        for f in ("fixed_gap_abs_max", "fixed_gap_rel_max", "fixed_gap_ulps_max"):
            v = np.array([r[f] for r in rows])
            out[f] = {"median": float(np.median(v)), "max": float(np.max(v))}
    return out


def summarise3(res: dict) -> None:
    for c in res["checks"]:
        print("checks", c)
    print("complete:", res["complete"])
    print("truth's discount radius minus 1:", {k: f"{v:+.2e}" for k, v in res["radius"].items()})
    print("state table (bytes):")
    for row in res["state_table"]:
        print(
            f"   {row['form']:18s} m={row['m']:5d} {row['state_bytes']:9d} B; fixed filter "
            f"{row['fixed_filter_bytes']:7d} B; start buffer {row['batch_start_buffer_bytes']:7d} B"
        )
    for summ in res["summary"]:
        print_sarima(summ, ETS_SPECS[summ["spec"]])  # type: ignore[arg-type]
        if summ.get("cases"):
            fa, fr, fu = (
                summ[k]["max"]
                for k in ("fixed_gap_abs_max", "fixed_gap_rel_max", "fixed_gap_ulps_max")
            )
            print(
                f"   fixed-parameter gap to fittedvalues: abs max {fa:.3g}, "
                f"rel max {fr:.3g}, ulps of the largest |prediction| max {fu:.3g}"
            )


def versions() -> dict:
    out = {"python": sys.version.split()[0]}
    for mod in ("numpy", "scipy", "statsmodels", "pandas"):
        try:
            out[mod] = __import__(mod).__version__
        except ImportError:
            out[mod] = None
    return out


PARTS = {
    "part1": (part1, summarise1),
    "part2": (part2, summarise2),
    "part3": (part3, summarise3),
}


def main(argv=None) -> int:  # noqa: ANN001
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("part", choices=sorted(PARTS))
    ap.add_argument("--out", default=os.path.join(".cache", "study_487_488"))
    ap.add_argument("--seeds", type=int, default=len(SEEDS), help="the first N seeds")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--force", action="store_true", help="recompute the part's summary")
    ap.add_argument("--no-run", action="store_true", help="summarise the cached cases; run nothing")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"{args.part}.json")
    run, summarise = PARTS[args.part]
    if os.path.exists(path) and not args.force:
        with open(path) as fh:
            res = json.load(fh)
        print(f"(cached: {path})")
    else:
        t0 = time.perf_counter()
        res = run(args)
        res["versions"] = versions()
        res["_seconds"] = time.perf_counter() - t0
        if res.get("complete", True):
            with open(path, "w") as fh:
                json.dump(res, fh, indent=1)
    summarise(res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
