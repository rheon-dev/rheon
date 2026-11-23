"""``garch`` with inputs, GARCH-X, against ``scipy``'s batch maximum
likelihood (task 559, part 4).

``arch`` has no input in its variance equation, and ``rugarch``, which has
(``external.regressors``), is R's: it was measured against the definition
once (docs/records/HORIZONS.md §16), and the repository's CI runs no R. So
the estimates' oracle is ``scipy.optimize`` maximizing the Gaussian
likelihood written out here from the definition:

    s_t = ω + α e²_{t-1} + γ 1[e_{t-1} < 0] e²_{t-1} + β s_{t-1} + π'x_t
    l   = -½ Σ (ln 2π + ln s_t + e_t² / s_t)

every lag before the first row the backcast, the ``0.94^i``-weighted mean of
the first 75 squared returns (``arch``'s, which the bank keeps), and GJR's
asymmetric lag half of it. Given the returns the recursion is linear in
``s``, a first-order filter of ``u_t = ω + α e²_{t-1} + γ ... + π'x_t``, so
``scipy.signal.lfilter`` runs it. EGARCH's ``z`` reads ``s``, so it runs as a
loop.

Two checks. The start's fit, ``coef`` on the row that closes the start, is
the maximum of the same likelihood on the same rows: no more than 1e-8 nats
below ``scipy``'s, and each parameter within 1e-4 of its standard error
(measured: within 5.4e-6). The online estimate after 20,000 rows is
within 1.5 of the batch fit's standard errors (the inverse of the
likelihood's curvature at it, by central differences) of ``scipy``'s fit of
those rows: the recursive prediction-error method agrees with the batch fit
to first order. Measured over five seeds of each form: at most 0.88.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest
from scipy.optimize import minimize
from scipy.signal import lfilter

from rheon import polars as rp

TIER = "mixed"

SQRT_2_PI = math.sqrt(2.0 / math.pi)
LN_2PI = math.log(2.0 * math.pi)

TRUTH = {
    "garch": [0.02, 0.05, 0.85, 0.1, 0.04],
    "gjr": [0.02, 0.03, 0.06, 0.85, 0.1, 0.04],
    "egarch": [-0.02, 0.15, -0.08, 0.95, 0.05, -0.03],
}


def _inputs(vol: str, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = np.zeros(2)
    out = np.empty((n, 2))
    for t in range(n):
        a = 0.9 * a + math.sqrt(1 - 0.81) * rng.standard_normal(2)
        out[t] = a if vol == "egarch" else 0.5 * a * a
    return out


def _simulate(vol: str, theta: list[float], x: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    k = 3 if vol == "garch" else 4
    e, s, ln_s = 0.0, 1.0, 0.0
    out = np.empty(len(x))
    for t in range(len(x)):
        px = float(np.dot(theta[k:], x[t]))
        if vol == "egarch":
            z = 0.0 if t == 0 else e / math.sqrt(s)
            news = 0.0 if t == 0 else theta[1] * (abs(z) - SQRT_2_PI) + theta[2] * z
            ln_s = theta[0] + news + theta[3] * ln_s + px
            s = math.exp(ln_s)
        else:
            g = theta[2] if k == 4 and e < 0 else 0.0
            s = theta[0] + (theta[1] + g) * e * e + theta[k - 1] * s + px
        e = math.sqrt(s) * rng.standard_normal()
        out[t] = e
    return out


def _backcast(e: np.ndarray) -> float:
    w = 0.94 ** np.arange(min(75, len(e)))
    return float(np.dot(w, e[: len(w)] ** 2) / w.sum())


def _variances(vol: str, theta: np.ndarray, e: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Each row's variance by the definition, from the backcast."""
    b = _backcast(e)
    k = 3 if vol == "garch" else 4
    px = x @ theta[k:]
    if vol == "egarch":
        out = np.empty(len(e))
        ln_s, z = math.log(b), None
        for t in range(len(e)):
            news = 0.0 if z is None else theta[1] * (abs(z) - SQRT_2_PI) + theta[2] * z
            ln_s = theta[0] + news + theta[3] * ln_s + px[t]
            out[t] = math.exp(ln_s)
            z = e[t] / math.sqrt(out[t])
        return out
    e2 = np.concatenate([[b], e[:-1] ** 2])
    u = theta[0] + theta[1] * e2 + px
    if k == 4:
        neg = np.concatenate([[0.5 * b], np.where(e[:-1] < 0, e[:-1] ** 2, 0.0)])
        u = u + theta[2] * neg
    beta = theta[k - 1]
    return lfilter([1.0], [1.0, -beta], u, zi=[beta * b])[0]


def _loglik(vol: str, theta: np.ndarray, e: np.ndarray, x: np.ndarray) -> float:
    try:
        # a trial point of the search can take EGARCH's variance past the
        # double range: no likelihood there
        with np.errstate(all="ignore"):
            s = _variances(vol, theta, e, x)
    except (OverflowError, ValueError):
        return -1e300
    if not np.all(np.isfinite(s)) or np.any(s <= 0):
        return -1e300
    return float(-0.5 * np.sum(LN_2PI + np.log(s) + e * e / s))


def _scipy_fit(vol: str, e: np.ndarray, x: np.ndarray, start: list[float]):
    """``scipy``'s maximum of the likelihood, its log-likelihood and the
    standard errors from the inverse of its curvature, by central
    differences."""
    k = 3 if vol == "garch" else 4
    neg = lambda th: -_loglik(vol, th, e, x)  # noqa: E731
    if vol == "egarch":
        bounds = [
            (None, None),
            (0.0, 1.0),
            (-1.0, 1.0),
            (0.0, 1 - 1e-4),
            (None, None),
            (None, None),
        ]
        cons = [{"type": "ineq", "fun": lambda th: th[1] - abs(th[2])}]
    else:
        bounds = [(1e-8, None)] + [(0.0, 1.0)] * (k - 1) + [(0.0, None)] * 2
        p = (lambda th: th[1] + 0.5 * th[2] + th[3]) if k == 4 else (lambda th: th[1] + th[2])
        cons = [{"type": "ineq", "fun": lambda th: 1 - 1e-4 - p(th)}]
    best = None
    for x0 in (np.array(start), np.array(start) * 0.8 + 0.01):
        r = minimize(
            neg,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={"ftol": 1e-14, "maxiter": 1000},
        )
        if best is None or r.fun < best.fun:
            best = r
    th = best.x
    n = len(th)
    h = np.zeros((n, n))
    step = 1e-4 * np.maximum(np.abs(th), 1e-2)
    for i in range(n):
        for j in range(n):
            pp, pm, mp, mm = (th.copy() for _ in range(4))
            pp[i] += step[i]
            pp[j] += step[j]
            pm[i] += step[i]
            pm[j] -= step[j]
            mp[i] -= step[i]
            mp[j] += step[j]
            mm[i] -= step[i]
            mm[j] -= step[j]
            h[i, j] = (neg(pp) - neg(pm) - neg(mp) + neg(mm)) / (4 * step[i] * step[j])
    se = np.sqrt(np.diag(np.linalg.inv(h)))
    return th, -best.fun, se


def _bank(vol: str, e: np.ndarray, x: np.ndarray, start_rows: int) -> list:
    df = pl.DataFrame({"y": e, "x1": x[:, 0], "x2": x[:, 1]})
    spec = rp.spec.garch(
        "m", targets=["y"], features=["x1", "x2"], vol=vol, start_rows=start_rows, coef_every=0
    )
    return rp.ModelBank([spec]).fit_predict(df).unnest("m")["coef"].to_list()


@pytest.mark.parametrize("vol", ["garch", "gjr", "egarch"])
def test_the_start_is_scipys_maximum_of_the_same_likelihood(vol):
    n0 = 2000
    x = _inputs(vol, n0 + 5, 101)
    e = _simulate(vol, TRUTH[vol], x, 103)
    coef = _bank(vol, e, x, n0)
    got = np.array(coef[n0 - 1])
    assert coef[n0 - 2] is None or all(v is None for v in coef[n0 - 2]), (
        "the start closes on its last row"
    )
    th, f, se = _scipy_fit(vol, e[:n0], x[:n0], TRUTH[vol])
    f_bank = _loglik(vol, got, e[:n0], x[:n0])
    # measured 2026-11-23 (Linux x86-64): the bank 1.8e-12 to 1.8e-11 nats
    # above scipy's stop, every parameter within 5.4e-6 of a standard error
    assert f_bank >= f - 1e-8, (f_bank, f, got, th)
    assert np.all(np.abs(got - th) <= 1e-4 * se), (got, th, se)


@pytest.mark.extended(reason="a grid: three forms, three seeds of 20,000 rows, each a scipy fit")
@pytest.mark.parametrize("vol", ["garch", "gjr", "egarch"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_online_estimate_is_the_batch_fit_to_first_order(vol, seed):
    n = 20_000
    x = _inputs(vol, n, 200 + seed)
    e = _simulate(vol, TRUTH[vol], x, 300 + seed)
    got = np.array(_bank(vol, e, x, 2000)[-1])
    th, _, se = _scipy_fit(vol, e, x, TRUTH[vol])
    gap = np.abs(got - th) / se
    # measured 2026-11-23 over seeds 0 to 4: at most 0.88 (EGARCH's beta,
    # seed 2), 0.45 under GARCH and GJR
    assert np.all(gap < 1.5), (vol, seed, gap, got, th, se)
