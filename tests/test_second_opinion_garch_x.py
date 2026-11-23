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

With Student's t and Hansen's skewed t the start is held the same way
against ``scipy``'s maximum of the likelihood written out here, the input
term as above and each row's density the standardized one the model's
documentation states,

    c        = Γ((ν+1)/2) / (√(π (ν−2)) Γ(ν/2))
    a        = 4 λ c (ν−2)/(ν−1),   b = √(1 + 3λ² − a²)
    m        = (b z + a) / (1 + sign(z + a/b) λ)
    l        = Σ [ln b + ln c − (ν+1)/2 ln(1 + m²/(ν−2)) − ½ ln s_t]

with ``z = e_t/√s_t`` and ``λ = 0`` for the t. ``scipy.special.gammaln``
gives ``ln Γ``, and ``scipy.stats.t`` holds the t's density to the
definition. ``rugarch``'s ``std`` and ``sstd`` (the latter Fernández and
Steel's skew, not Hansen's) were not run.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest
from scipy import stats
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.signal import lfilter
from scipy.special import gammaln

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
    return th, -best.fun, _standard_errors(neg, th)


def _standard_errors(neg, th: np.ndarray) -> np.ndarray:  # noqa: ANN001
    """The standard errors at ``th`` from the inverse of the curvature of
    ``neg``, the negative log-likelihood, by central differences."""
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
    return np.sqrt(np.diag(np.linalg.inv(h)))


def _bank(vol: str, e: np.ndarray, x: np.ndarray, start_rows: int, dist: str = "normal") -> list:
    df = pl.DataFrame({"y": e, "x1": x[:, 0], "x2": x[:, 1]})
    spec = rp.spec.garch(
        "m",
        targets=["y"],
        features=["x1", "x2"],
        vol=vol,
        dist=dist,
        start_rows=start_rows,
        coef_every=0,
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


#: The shapes of the t and skewed-t truths: ``ν`` (and ``λ``).
SHAPES = {"t": [6.0], "skewt": [6.0, -0.2]}


def _log_density(z: np.ndarray, nu: float, lam: float) -> np.ndarray:
    """Hansen's standardized skewed t at ``(ν, λ)``, written from its
    definition (the module docstring); ``λ = 0`` is Student's t."""
    c = math.exp(gammaln((nu + 1) / 2) - gammaln(nu / 2)) / math.sqrt(math.pi * (nu - 2))
    a = 4 * lam * c * (nu - 2) / (nu - 1)
    b = math.sqrt(1 + 3 * lam * lam - a * a)
    m = (b * z + a) / np.where(z > -a / b, 1 + lam, 1 - lam)
    return math.log(b) + math.log(c) - 0.5 * (nu + 1) * np.log1p(m * m / (nu - 2))


def _shaped_loglik(vol: str, dist: str, x_all: np.ndarray, e: np.ndarray, x: np.ndarray) -> float:
    """The t's or skewed t's log-likelihood at ``x_all``, the variance
    equation's parameters (the inputs' last) and then the shape."""
    k = (3 if vol == "garch" else 4) + x.shape[1]
    theta, shape = x_all[:k], x_all[k:]
    nu = float(shape[0])
    lam = float(shape[1]) if dist == "skewt" else 0.0
    if not (2.05 <= nu <= 300 and abs(lam) < 1):
        return -1e300
    try:
        with np.errstate(all="ignore"):
            s = _variances(vol, theta, e, x)
    except (OverflowError, ValueError):
        return -1e300
    if not np.all(np.isfinite(s)) or np.any(s <= 0):
        return -1e300
    return float(np.sum(_log_density(e / np.sqrt(s), nu, lam) - 0.5 * np.log(s)))


def _scipy_shaped_fit(vol: str, dist: str, e: np.ndarray, x: np.ndarray, start: list[float]):
    """``scipy``'s SLSQP maximum of :func:`_shaped_loglik`, its value and
    its standard errors, under GARCH's or GJR's set, ``ν ∈ [2.05, 300]``
    and ``|λ| ≤ 0.99``."""
    k = 3 if vol == "garch" else 4
    neg = lambda th: -_shaped_loglik(vol, dist, th, e, x)  # noqa: E731
    bounds = [(1e-8, None)] + [(0.0, 1.0)] * (k - 1) + [(0.0, None)] * x.shape[1]
    bounds += [(2.05, 300.0)] + ([(-0.99, 0.99)] if dist == "skewt" else [])
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
    return th, -best.fun, _standard_errors(neg, th)


def _shaped_innovations(dist: str, n: int, seed: int) -> np.ndarray:
    """Unit-variance draws: the t's by ``numpy``, the skewed t's by
    inverting its distribution function on a grid of its density."""
    rng = np.random.default_rng(seed)
    nu = SHAPES[dist][0]
    if dist == "t":
        return rng.standard_t(nu, n) * math.sqrt((nu - 2) / nu)
    grid = np.linspace(-40.0, 40.0, 400_001)
    cdf = np.cumsum(np.exp(_log_density(grid, nu, SHAPES[dist][1])))
    return np.interp(rng.uniform(size=n), cdf / cdf[-1], grid)


def _simulate_with(vol: str, theta: list[float], x: np.ndarray, z: np.ndarray) -> np.ndarray:
    """:func:`_simulate`'s GARCH or GJR process driven by the draws ``z``."""
    k = 3 if vol == "garch" else 4
    e, s = 0.0, 1.0
    out = np.empty(len(x))
    for t in range(len(x)):
        g = theta[2] if k == 4 and e < 0 else 0.0
        s = theta[0] + (theta[1] + g) * e * e + theta[k - 1] * s + float(np.dot(theta[k:], x[t]))
        e = math.sqrt(s) * z[t]
        out[t] = e
    return out


def _moment(v: float, p: int, nu: float, lam: float) -> float:
    """``v^p f(v)``, the integrand of the density's ``p``-th moment."""
    return v**p * math.exp(_log_density(np.array([v]), nu, lam)[0])


def test_the_shaped_density_is_its_definition():
    """The hand-written density is the t's of ``scipy.stats`` at ``λ = 0``,
    rescaled to unit variance, and with a skew still a density of mean 0
    and variance 1."""
    z = np.linspace(-6.0, 6.0, 121)
    for nu in (2.5, 6.0, 40.0):
        r = math.sqrt(nu / (nu - 2))
        want = stats.t.logpdf(z * r, nu) + math.log(r)
        assert np.allclose(_log_density(z, nu, 0.0), want, rtol=0, atol=1e-13), nu
    for nu, lam in ((6.0, -0.2), (5.0, 0.5), (30.0, -0.8)):
        # integrated on each side of the branch point -a/b
        c = math.exp(gammaln((nu + 1) / 2) - gammaln(nu / 2)) / math.sqrt(math.pi * (nu - 2))
        a = 4 * lam * c * (nu - 2) / (nu - 1)
        kink = -a / math.sqrt(1 + 3 * lam * lam - a * a)
        sides = ((-np.inf, kink), (kink, np.inf))
        moments = [
            sum(quad(_moment, lo, hi, args=(p, nu, lam), limit=200)[0] for lo, hi in sides)
            for p in (0, 1, 2)
        ]
        assert np.allclose(moments, [1.0, 0.0, 1.0], rtol=0, atol=1e-8), (nu, lam, moments)


@pytest.mark.parametrize(("vol", "dist"), [("garch", "t"), ("gjr", "t"), ("garch", "skewt")])
def test_the_shaped_start_is_scipys_maximum_of_the_same_likelihood(vol, dist):
    n0 = 2000
    x = _inputs(vol, n0 + 5, 111)
    truth = TRUTH[vol] + SHAPES[dist]
    e = _simulate_with(vol, TRUTH[vol], x, _shaped_innovations(dist, n0 + 5, 113))
    coef = _bank(vol, e, x, n0, dist)
    got = np.array(coef[n0 - 1])
    assert len(got) == len(truth), got
    assert coef[n0 - 2] is None or all(v is None for v in coef[n0 - 2]), (
        "the start closes on its last row"
    )
    th, f, se = _scipy_shaped_fit(vol, dist, e[:n0], x[:n0], truth)
    f_bank = _shaped_loglik(vol, dist, got, e[:n0], x[:n0])
    # measured 2026-11-23 (Linux x86-64), seeds 111 to 151 of each form
    # (two of the fifteen starts waited, the t's at seed 131): the bank
    # 2.3e-12 to 5.0e-11 nats above scipy's stop, every parameter within
    # 9.7e-6 of a standard error
    assert f_bank >= f - 1e-8, (f_bank, f, got, th)
    assert np.all(np.abs(got - th) <= 1e-4 * se), (got, th, se, np.abs(got - th) / se)
