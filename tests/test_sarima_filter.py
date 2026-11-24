"""The SARIMA(X) filter at fixed parameters against statsmodels' ``SARIMAX``.

The core's filter (``_rheon.sarima_filter``: the model's own recursion over
one series, with no clock, weight or group) predicts each slot's
differenced value ``ŵ_t`` from the slots before it. The oracle is
``SARIMAX(order, seasonal_order, trend="n", simple_differencing=True)``
with its ``filter`` at the same parameters: the same state-space form, the
same stationary start, the same differencing. Its steady-state switch is
off (``ssm.tolerance = 0``), so both run the full recursion on every row.

The error is read in units in the last place of the largest ``|w|`` of the
series, ``|ŵ_t - ŵ_t^SARIMAX| / spacing(max_t |w_t|)``, over the rows past
the first 500 differenced ones: within 8 (the bound the design study
measured for a season up to 24). Every row from the first is held within
``1e-13`` of the largest ``|w|``: both filters are exact from the first
row and start from the same stationary covariance, but each solves its
Lyapunov equation its own way, and statsmodels' solution misses the
equation by up to about ``1.6e-14`` of its scale (the core's doubling by
under ``2e-15``), a gap that fades as the filter forgets its start. At
``(1,0,1)(1,0,1,24)`` that is about 100 ulps on the first rows and 2 by row
500; elsewhere the two agree within 5 ulps from the first row. Past the
start the two differ only in the order of their sums: statsmodels' dense
BLAS products against the core's two terms an entry of the companion
matrix.

A null value is a slot not observed, which ``SARIMAX`` reads as a ``NaN``
and predicts through, so seasonal positions stay aligned.
"""

from __future__ import annotations

import numpy as np
import pytest
from statsmodels.tsa.statespace.sarimax import SARIMAX

import ledger
from rheon import _rheon as native

TIER = "mixed"

#: The study's bound for a season up to 24: 8 ulps of the largest ``|w|``,
#: past the first ``BURN`` differenced rows.
ULPS = 8.0
BURN = 500

#: Every row's gap from the first, as a share of the largest ``|w|``: the
#: two starts' solutions of the Lyapunov equation differ by up to about
#: 1.3e-14 of it on the grid's specs (statsmodels' the less exact).
FIRST = 1e-13

#: Each case: ``(order, seasonal_order, params)``, statsmodels' flat order.
SPECS = {
    "airline_7": ((0, 1, 1), (0, 1, 1, 7), [-0.4, -0.6, 1.0]),
    "full_24": ((1, 1, 1), (1, 1, 1, 24), [0.5, -0.3, 0.4, -0.5, 2.0]),
    "arma_24": ((1, 0, 1), (1, 0, 1, 24), [0.6, 0.2, 0.7, -0.4, 0.5]),
    "seasonal_ar_24": ((1, 1, 1), (1, 1, 0, 24), [0.3, 0.4, -0.5, 1.5]),
    "order_two": ((2, 1, 2), (0, 0, 0, 0), [1.1, -0.35, 0.3, 0.2, 1.0]),
    "season_two_ma": ((0, 1, 1), (0, 1, 2, 12), [0.4, -0.5, 0.2, 1.0]),
}


def _sim(order, seasonal_order, params, n: int, seed: int, k_exog: int = 0):
    """``n`` values of the model with its parameters, after a burn-in: the
    ARMA part by its two polynomials, then integrated, then the
    regressors' part added. Returns ``(y, exog)``."""
    p, d, q = order
    sp, sd, sq, s = seasonal_order
    beta = np.asarray(params[:k_exog])
    rest = params[k_exog:]
    ar, ma = rest[:p], rest[p : p + q]
    sar, sma = rest[p + q : p + q + sp], rest[p + q + sp : p + q + sp + sq]
    sigma2 = rest[-1]
    s = max(s, 1)

    def lag(coefs, step, sign):
        out = np.zeros(len(coefs) * step + 1)
        out[0] = 1.0
        out[step::step] = sign * np.asarray(coefs)
        return out

    a = np.polymul(lag(ar, 1, -1.0)[::-1], lag(sar, s, -1.0)[::-1])[::-1]
    c = np.polymul(lag(ma, 1, 1.0)[::-1], lag(sma, s, 1.0)[::-1])[::-1]
    rng = np.random.default_rng(seed)
    burn = 1000
    e = np.sqrt(sigma2) * rng.standard_normal(n + burn)
    u = np.zeros(n + burn)
    for t in range(n + burn):
        acc = sum(c[j] * e[t - j] for j in range(len(c)) if t >= j)
        acc -= sum(a[i] * u[t - i] for i in range(1, len(a)) if t >= i)
        u[t] = acc
    y = u[burn:].copy()
    for _ in range(sd):
        for t in range(s, n):
            y[t] += y[t - s]
    for _ in range(d):
        y = np.cumsum(y)
    exog = rng.standard_normal((n, k_exog)) if k_exog else None
    if exog is not None:
        y = y + exog @ beta
    return y + 50.0, exog


def _depth(order, seasonal_order) -> int:
    return order[1] + seasonal_order[1] * seasonal_order[3]


def _sarimax(y, order, seasonal_order, params, exog=None, steady=False):
    """statsmodels' one-step predictions of the differenced series, and the
    differenced series itself."""
    model = SARIMAX(
        y,
        exog=exog,
        order=order,
        seasonal_order=seasonal_order,
        trend="n",
        simple_differencing=True,
    )
    if not steady:
        model.ssm.tolerance = 0
    res = model.filter(np.asarray(params, dtype=float))
    return np.asarray(res.forecasts[0]), np.asarray(model.endog[:, 0])


def _ours(y, order, seasonal_order, params, exog=None):
    ys = [None if np.isnan(v) else float(v) for v in y]
    xs = None if exog is None else exog.tolist()
    w_hat, y_hat = native.sarima_filter(ys, order, seasonal_order, list(params), xs)
    return np.asarray(w_hat), np.asarray(y_hat)


def _ulps(order, seasonal_order, params, n, seed, k_exog=0, missing=(), burn=BURN):
    """The worst gap between the two filters' ``ŵ_t`` past ``burn`` rows, in
    ulps of the largest ``|w|``; the worst over every row, as a share of
    it; and ``ŷ_t``'s against ``ŵ_t + y_t - w_t`` past ``burn`` rows, in
    ulps of the largest ``|y|``, on the rows fully observed."""
    y, exog = _sim(order, seasonal_order, params, n, seed, k_exog)
    y[list(missing)] = np.nan
    theirs, w = _sarimax(y, order, seasonal_order, params, exog)
    w_hat, y_hat = _ours(y, order, seasonal_order, params, exog)
    depth = _depth(order, seasonal_order)
    assert np.isnan(w_hat[:depth]).all() and np.isnan(y_hat[:depth]).all()
    mine = w_hat[depth:]
    assert np.isfinite(mine).all()
    scale = np.nanmax(np.abs(w))
    gap = np.abs(mine - theirs)
    w_err = float(np.max(gap[burn:]) / np.spacing(scale))
    rel = float(np.max(gap) / scale)
    back = theirs + (y[depth:] - w)
    seen = np.isfinite(back)
    seen[:burn] = False
    y_gap = np.abs(y_hat[depth:][seen] - back[seen])
    y_err = float(np.max(y_gap) / np.spacing(np.nanmax(np.abs(y))))
    return w_err, rel, y_err


def _holds(w_err, rel, y_err, what):
    assert w_err <= ULPS, f"{what}: {w_err:.2f} ulps"
    assert rel <= FIRST, f"{what}: {rel:.2e} of the largest |w| on an early row"
    assert y_err <= ULPS, f"{what}: {y_err:.2f} ulps on y"


@pytest.mark.parametrize("spec", sorted(SPECS))
def test_the_filter_is_sarimax_to_the_last_bits(spec):
    """``ŵ_t`` within 8 ulps of the largest ``|w|`` of ``SARIMAX.filter``'s
    past the first 500 rows, and within ``1e-13`` of it from the first, at
    seasons up to 24; ``ŷ_t`` is the differencing put back, within the same
    of the largest ``|y|``."""
    order, seasonal, params = SPECS[spec]
    _holds(*_ulps(order, seasonal, params, 2000, seed=1), spec)


def test_regressors_enter_differenced_as_sarimax_takes_them():
    """With ``exog=``, ``SARIMAX``'s simple differencing differences the
    regressors as well, and ``x_t' beta`` enters the observation."""
    order, seasonal = (1, 1, 1), (0, 1, 1, 7)
    params = [0.8, -1.3, 0.4, -0.3, -0.5, 1.2]
    _holds(*_ulps(order, seasonal, params, 2000, seed=2, k_exog=2), "exog")


def test_a_missing_value_is_predicted_through_as_sarimax_does():
    """A null value keeps its slot: the filter skips its update, and every
    differenced value that reads it, as ``SARIMAX`` does with a ``NaN``, so
    the seasonal positions after it stay where they were. Runs of missing
    slots, inside the first season too."""
    order, seasonal, params = SPECS["full_24"]
    missing = [3, 30, 31, 32, 200, *range(400, 450), 1201]
    _holds(*_ulps(order, seasonal, params, 2000, seed=3, missing=missing), "missing")


def test_the_steady_state_switch_moves_sarimax_by_little():
    """statsmodels' default filter stops updating ``P`` once it has
    converged to its tolerance; the core never stops. The gap that leaves
    is small against the series, under ``5e-9`` of the largest ``|w|``."""
    order, seasonal, params = SPECS["full_24"]
    y, _ = _sim(order, seasonal, params, 3000, seed=4)
    theirs, w = _sarimax(y, order, seasonal, params, steady=True)
    w_hat, _ = _ours(y, order, seasonal, params)
    depth = _depth(order, seasonal)
    gap = np.max(np.abs(w_hat[depth:] - theirs)) / np.max(np.abs(w))
    assert gap < 5e-9, gap


@pytest.mark.extended(reason="SARIMAX's dense filter at 170 states takes over two minutes")
def test_a_season_of_168_is_sarimax_too():
    """At ``s = 168`` (hourly data, a weekly season) the state has 170
    entries; every row from the first within 8 ulps (6.2 at worst over
    2,000 rows when measured, 3.5 over the first 50)."""
    order, seasonal = (1, 1, 1), (1, 1, 1, 168)
    params = [0.5, -0.3, 0.4, -0.5, 2.0]
    _holds(*_ulps(order, seasonal, params, 600, seed=5, burn=0), "s = 168")


GRID_SEEDS = range(50)
GRID_SPECS = ("airline_7", "full_24", "arma_24")

ledger.guard(
    "filter_grid",
    lambda: [
        _ours(*_sim(*SPECS[spec], 2000, seed)[:1], *SPECS[spec])[0]
        for spec in GRID_SPECS
        for seed in GRID_SEEDS
    ],
    oracles=["statsmodels"],
)


@pytest.mark.oracle_grid(reason="fifty seeds of three specs against SARIMAX, about a minute")
def test_fifty_seeds_of_each_spec_are_within_the_bound():
    """The worst of fifty seeds of each spec up to ``s = 24``, 2,000 rows
    each, within the bounds; the ledger holds the worst and the 95th
    percentile past the first 500 rows, in ulps, so the agreement only
    tightens (under 8 ulps it is round-off, the ledger's floor)."""
    for spec in GRID_SPECS:
        order, seasonal, params = SPECS[spec]
        runs = [_ulps(order, seasonal, params, 2000, seed) for seed in GRID_SEEDS]
        ledger.record("filter_grid", f"{spec}_ulps", [r[0] for r in runs], floor=ULPS)
        for seed, r in zip(GRID_SEEDS, runs, strict=True):
            _holds(*r, f"{spec}, seed {seed}")
