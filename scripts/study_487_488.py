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

    def __init__(self, spec: Spec, warm: int | None = None) -> None:
        self.spec = spec
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
        what = self._what()
        pred = base + what
        self._row_w(w, what)
        return pred

    def _row_w(self, w: float, what: float) -> None:
        sp = self.spec
        e = w - what
        t = self.t
        self.t += 1
        warm = self.warm
        if t < warm:
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
}


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
        res = model.fit(
            start_params=start_params,
            method=st["method"],
            maxiter=st["maxiter"],
            cov_type=st["cov_type"],
            disp=False,
        )
    return model, res


def sarima_case(spec: Spec, seed: int, n: int = N_ROWS, tol0: bool = True) -> dict:
    """One series: the online estimator against SARIMAX on the same rows."""
    y = simulate_sarima(spec, n, seed)
    t0 = time.perf_counter()
    est = OnlineSarima(spec)
    pred = np.array([est.update(float(v)) for v in y])
    t_online = time.perf_counter() - t0
    t0 = time.perf_counter()
    model, res = sarimax_fit(spec, y)
    t_batch = time.perf_counter() - t0
    b = np.asarray(res.params, dtype=float)
    se = np.asarray(res.bse, dtype=float)
    o = est.params
    L = est.ry.n
    w = model.endog[:, 0] if model.endog.ndim == 2 else np.asarray(model.endog)
    w = np.asarray(w, dtype=float).ravel()
    # the fixed-parameter recursion at the batch parameters against SARIMAX's filter
    fres = model.filter(b)
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
        fc0 = np.asarray(model.filter(b).forecasts, dtype=float).ravel()
        model.ssm.tolerance = tol_saved
        gap_tol0 = float(np.max(np.abs(mine[tail_w] - fc0[tail_w])))
    base = y[L:] - w  # y_t - w_t
    yhat_b = base + fc
    half = slice(n // 2, n)
    yh = pred[half]
    yb = np.concatenate([np.full(L, math.nan), yhat_b])[half]
    out = {
        "spec": spec.name,
        "seed": seed,
        "n": n,
        "silent_rows": int(np.sum(np.isnan(pred))),
        "silent_explained": int(
            sum(1 for i in range(n) if math.isnan(pred[i]) and i in est.reasons)
        ),
        "sm_converged": bool(res.mle_retvals.get("converged", False)),
        "sm_iterations": int(res.mle_retvals.get("iterations", -1)),
        "llf_batch": float(res.llf),
        "llf_online_params": float(model.loglike(o)) if o[-1] > 0 else math.nan,
        "filter_gap_last1000": float(np.max(np.abs(mine[tail_w] - fc[tail_w]))),
        "filter_gap_tol0_last1000": gap_tol0,
        "w_abs_max_last1000": float(np.max(np.abs(w[tail_w]))),
        "pred_rms_gap_second_half": float(np.sqrt(np.mean((yh - yb) ** 2))),
        "online_mse_second_half": float(np.mean((y[half] - yh) ** 2)),
        "batch_mse_second_half": float(np.mean((y[half] - yb) ** 2)),
        "t_online_s": t_online,
        "t_batch_s": t_batch,
        "state_bytes": est.state_bytes(),
    }
    for i, nm in enumerate(spec.names):
        out[f"{nm}_batch"] = float(b[i])
        out[f"{nm}_online"] = float(o[i])
        out[f"{nm}_se"] = float(se[i])
        out[f"{nm}_gap_se"] = float((o[i] - b[i]) / se[i])
    out["max_abs_gap_se"] = float(max(abs(out[f"{nm}_gap_se"]) for nm in spec.names))
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


def summarise_sarima(rows: list[dict], spec: Spec) -> dict:
    rows = [r for r in rows if r["spec"] == spec.name]
    out: dict = {"spec": spec.name, "cases": len(rows)}
    for nm in spec.names:
        g = np.abs([r[f"{nm}_gap_se"] for r in rows])
        out[nm] = {
            "median": float(np.median(g)),
            "max": float(np.max(g)),
            "over2": int(np.sum(g > 2)),
            "worst_seed": int(rows[int(np.argmax(g))]["seed"]),
            "worst_signed": float(rows[int(np.argmax(g))][f"{nm}_gap_se"]),
        }
    out["all_within_2"] = int(sum(1 for r in rows if r["max_abs_gap_se"] <= 2.0))
    out["sm_converged"] = int(sum(r["sm_converged"] for r in rows))
    out["sm_iterations_max"] = int(max(r["sm_iterations"] for r in rows))
    for f in (
        "filter_gap_last1000",
        "filter_gap_tol0_last1000",
        "pred_rms_gap_second_half",
        "t_online_s",
        "t_batch_s",
    ):
        v = np.array([r[f] for r in rows if f in r and not math.isnan(r[f])])
        out[f] = {
            "median": float(np.median(v)) if v.size else math.nan,
            "max": float(np.max(v)) if v.size else math.nan,
            "cases": int(v.size),
        }
    ratio = np.array([r["online_mse_second_half"] / r["batch_mse_second_half"] for r in rows])
    out["mse_ratio_second_half"] = {"median": float(np.median(ratio)), "max": float(np.max(ratio))}
    dll = np.array([r["llf_batch"] - r["llf_online_params"] for r in rows])
    out["llf_gap_total"] = {"median": float(np.median(dll)), "max": float(np.max(dll))}
    out["silent_rows"] = sorted({r["silent_rows"] for r in rows})
    out["silent_unexplained"] = int(sum(r["silent_rows"] - r["silent_explained"] for r in rows))
    out["state_bytes"] = rows[0]["state_bytes"]
    return out


def print_sarima(summ: dict, spec: Spec) -> None:
    print(
        f"== {spec.name}: SARIMA({spec.p},{spec.d},{spec.q})({spec.P},{spec.D},{spec.Q},{spec.s})"
    )
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


def summarise1(res: dict) -> None:
    print("SARIMAX settings:", res["sarimax_settings"])
    for c in res["checks"]:
        print("checks", c)
    for summ in res["summary"]:
        print_sarima(summ, SPECS[summ["spec"]])


def versions() -> dict:
    out = {"python": sys.version.split()[0]}
    for mod in ("numpy", "scipy", "statsmodels", "pandas"):
        try:
            out[mod] = __import__(mod).__version__
        except ImportError:
            out[mod] = None
    return out


PARTS = {"part1": (part1, summarise1)}


def main(argv=None) -> int:  # noqa: ANN001
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("part", choices=sorted(PARTS))
    ap.add_argument("--out", default=os.path.join(".cache", "study_487_488"))
    ap.add_argument("--seeds", type=int, default=len(SEEDS), help="the first N seeds")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--force", action="store_true", help="recompute the part's summary")
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
        with open(path, "w") as fh:
            json.dump(res, fh, indent=1)
    summarise(res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
