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
500; elsewhere the two agree within 5 ulps from the first row. Which side
is off there is settled by the exact filter, in 40 digits: the core's
predictions are within 4 ulps of it from the first row, statsmodels' up
to 100. Past the start the two differ only in the order of their sums: statsmodels' dense
BLAS products against the core's two terms an entry of the companion
matrix.

A null value is a slot not observed, which ``SARIMAX`` reads as a ``NaN``
and predicts through, so seasonal positions stay aligned.
"""

from __future__ import annotations

import mpmath as mp
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


def _exact(w, order, seasonal_order, params, rows, digits=40):
    """The one-step predictions of the first ``rows`` slots, exactly: the
    stationary covariance as its defining series ``sigma2 sum_j (T^j R)
    (T^j R)'``, summed until a term is under ``10^-(digits + 5)`` of the
    sum, and the Kalman filter, both in mpmath at ``digits`` digits, from
    the parameters as the doubles they are. Shares no arithmetic with
    either double-precision filter."""
    mp.mp.dps = digits
    p, _, q = order
    sp, _, sq, s = seasonal_order
    par = [mp.mpf(float(v)) for v in params]
    ar, ma = par[:p], par[p : p + q]
    sar, sma = par[p + q : p + q + sp], par[p + q + sp : p + q + sp + sq]

    def poly(coefs, step, sign):
        out = [mp.mpf(0)] * (len(coefs) * step + 1)
        out[0] = mp.mpf(1)
        for i, c in enumerate(coefs):
            out[(i + 1) * step] = sign * c
        return out

    def mul(a, b):
        out = [mp.mpf(0)] * (len(a) + len(b) - 1)
        for i, x in enumerate(a):
            for j, y in enumerate(b):
                out[i + j] += x * y
        return out

    a = mul(poly(ar, 1, -1), poly(sar, s, -1))
    c = mul(poly(ma, 1, 1), poly(sma, s, 1))
    r = max(len(a) - 1, len(c))
    tcol = [-a[i + 1] if i + 1 < len(a) else mp.mpf(0) for i in range(r)]
    big_r = [c[i] if i < len(c) else mp.mpf(0) for i in range(r)]
    sigma2 = par[-1]

    def t_vec(v):
        return [tcol[i] * v[0] + (v[i + 1] if i + 1 < r else 0) for i in range(r)]

    def t_p_t(pm):
        m = [
            [tcol[i] * pm[0][j] + (pm[i + 1][j] if i + 1 < r else 0) for j in range(r)]
            for i in range(r)
        ]
        return [
            [m[i][0] * tcol[j] + (m[i][j + 1] if j + 1 < r else 0) for j in range(r)]
            for i in range(r)
        ]

    pm = [[mp.mpf(0)] * r for _ in range(r)]
    v = list(big_r)
    tol = mp.mpf(10) ** (-(digits + 5))
    while True:
        for i in range(r):
            vi = v[i] * sigma2
            for k in range(r):
                pm[i][k] += vi * v[k]
        if max(abs(x) for x in v) ** 2 < tol:
            break
        v = t_vec(v)
    state = [mp.mpf(0)] * r
    out = []
    for wt in w[:rows]:
        out.append(state[0])
        if np.isfinite(wt):
            f = pm[0][0]
            gain = [pm[i][0] / f for i in range(r)]
            innov = mp.mpf(float(wt)) - state[0]
            state = [state[i] + gain[i] * innov for i in range(r)]
            row0 = list(pm[0])
            pm = [[pm[i][k] - gain[i] * row0[k] for k in range(r)] for i in range(r)]
        state = t_vec(state)
        pm = t_p_t(pm)
        for i in range(r):
            for k in range(r):
                pm[i][k] += big_r[i] * sigma2 * big_r[k]
    return out


def _blp(w, order, seasonal_order, params, digits=40):
    """The one-step predictions of every slot of ``w``, exactly, by another
    route than ``_exact``'s: the best linear predictor of a stationary
    series from its finite past is what the Kalman filter from the
    stationary start computes, and Durbin-Levinson computes it from the
    autocovariances. Those are ``sigma2 sum_j psi_j psi_{j+h}`` over the
    MA(infinity) weights ``psi`` (summed until a whole span of them is under
    ``10^-(digits + 5)``) up to the larger polynomial's degree, and the AR
    recursion past it. All in mpmath at ``digits`` digits, from the
    parameters and ``w`` as the doubles they are. Its cost is the
    polynomials' degree times the ``psi`` weights' length plus the square
    of the rows, where ``_exact``'s is the square of the states a row, so
    it reaches ``s = 168`` (170 states) in seconds."""
    mp.mp.dps = digits
    p, _, q = order
    sp, _, sq, s = seasonal_order
    par = [mp.mpf(float(v)) for v in params]
    ar, ma = par[:p], par[p : p + q]
    sar, sma = par[p + q : p + q + sp], par[p + q + sp : p + q + sp + sq]

    def poly(coefs, step, sign):
        out = [mp.mpf(0)] * (len(coefs) * step + 1)
        out[0] = mp.mpf(1)
        for i, c in enumerate(coefs):
            out[(i + 1) * step] = sign * c
        return out

    def mul(a, b):
        out = [mp.mpf(0)] * (len(a) + len(b) - 1)
        for i, x in enumerate(a):
            for j, y in enumerate(b):
                if x and y:
                    out[i + j] += x * y
        return out

    a = mul(poly(ar, 1, -1), poly(sar, s, -1))
    c = mul(poly(ma, 1, 1), poly(sma, s, 1))
    sigma2 = par[-1]
    phi = [(i, -a[i]) for i in range(1, len(a)) if a[i]]
    span = max(len(a), len(c))
    tol = mp.mpf(10) ** (-(digits + 5))
    psi = []
    while True:
        j = len(psi)
        v = c[j] if j < len(c) else mp.mpf(0)
        for i, f in phi:
            if j >= i:
                v += f * psi[j - i]
        psi.append(v)
        if len(psi) > span and max(abs(x) for x in psi[-span:]) ** 2 < tol:
            break
    n = len(w)
    top = min(span - 1, n - 1)
    gam = [sigma2 * mp.fdot(psi[: len(psi) - h], psi[h:]) for h in range(top + 1)]
    for h in range(top + 1, n):
        gam.append(mp.fsum(f * gam[h - i] for i, f in phi))
    x = [mp.mpf(float(v)) for v in w]
    out, coef, var = [mp.mpf(0)], [], gam[0]
    for t in range(1, n):
        k = (gam[t] - mp.fdot(coef, gam[t - 1 : 0 : -1])) / var
        coef = [cj - k * coef[t - 2 - j] for j, cj in enumerate(coef)] + [k]
        var *= 1 - k * k
        out.append(mp.fdot(coef, x[t - 1 :: -1]))
    return out


@pytest.mark.extended(reason="the exact filter in 40 digits takes about 10 s")
def test_the_first_rows_are_the_exact_filter():
    """On ``(1,0,1)(1,0,1,24)`` statsmodels' filter and the core's part by
    about 100 ulps of the largest ``|w|`` on the first rows. The exact
    filter (``_exact``, in 40 digits) says which is off: the core's is
    within 4 ulps of it from the first row (measured over fifty seeds and
    every one of 2,000 rows: 3.5 at worst, 2.6 past row 500), statsmodels'
    up to 100 ulps on the first rows (its stationary covariance is 5.4e-14
    of its scale from the exact one) and 2.4 past row 500. ``_blp``, the
    reference the ``s = 168`` test reads, is the same predictions by
    another route: within ``1e-20`` ulps of ``_exact`` here (8e-25 when
    measured)."""
    order, seasonal, params = SPECS["arma_24"]
    y, _ = _sim(order, seasonal, params, 2000, seed=1)
    _, w = _sarimax(y, order, seasonal, params)
    w_hat, _ = _ours(y, order, seasonal, params)
    rows = 200
    exact = _exact(w, order, seasonal, params, rows)
    unit = np.spacing(np.max(np.abs(w)))
    gap = [
        float(abs(mp.mpf(float(a)) - b)) / unit for a, b in zip(w_hat[:rows], exact, strict=True)
    ]
    worst = int(np.argmax(gap))
    assert max(gap) <= 4.0, f"{max(gap):.2f} ulps from the exact filter at row {worst}"
    other = _blp(w[:rows], order, seasonal, params)
    agree = max(float(abs(a - b)) / unit for a, b in zip(other, exact, strict=True))
    assert agree <= 1e-20, f"the two exact routes part by {agree:.1e} ulps"


@pytest.mark.extended(reason="SARIMAX's dense filter and the 40-digit reference, about 8 s")
def test_a_season_of_168_is_sarimax_too():
    """At ``s = 168`` (hourly data, a weekly season) the state has 170
    entries. The core is held to the exact predictions (``_blp``, 40
    digits, every one of the 431 differenced rows from the first): within
    4 ulps of the largest ``|w|``, and ``ŷ_t`` within 4 of the largest
    ``|y|`` (0.88 and 0.95 when measured). statsmodels is the second
    opinion, and the inexact side here: its predictions are up to 12.5
    ulps from the exact ones (row 130 when measured; its stationary
    covariance, a Lyapunov solution at 170 states, is the less exact
    start), so it is held to 13 against them, and the two filters to each
    other within the sum of their bounds. An 80-bit Kalman filter from the
    defining series of the start measured the same: the core 0.88 ulps
    off, statsmodels 12.5."""
    order, seasonal = (1, 1, 1), (1, 1, 1, 168)
    params = [0.5, -0.3, 0.4, -0.5, 2.0]
    y, _ = _sim(order, seasonal, params, 600, seed=5)
    theirs, w = _sarimax(y, order, seasonal, params)
    w_hat, y_hat = _ours(y, order, seasonal, params)
    depth = _depth(order, seasonal)
    assert np.isnan(w_hat[:depth]).all() and np.isnan(y_hat[:depth]).all()
    mine = w_hat[depth:]
    exact = _blp(w, order, seasonal, params)
    unit = np.spacing(np.max(np.abs(w)))

    def ulps(got):
        return [float(abs(mp.mpf(float(a)) - b)) / unit for a, b in zip(got, exact, strict=True)]

    ours, sm = ulps(mine), ulps(theirs)
    assert max(ours) <= 4.0, f"the core: {max(ours):.2f} ulps at row {int(np.argmax(ours))}"
    assert max(sm) <= 13.0, f"statsmodels: {max(sm):.2f} ulps at row {int(np.argmax(sm))}"
    both = float(np.max(np.abs(mine - theirs)) / unit)
    assert both <= 4.0 + 13.0, f"the two filters part by {both:.2f} ulps"
    y_unit = np.spacing(np.max(np.abs(y)))
    back = [
        b + mp.mpf(float(v)) - mp.mpf(float(u)) for b, v, u in zip(exact, y[depth:], w, strict=True)
    ]
    y_err = max(
        float(abs(mp.mpf(float(a)) - b)) / y_unit for a, b in zip(y_hat[depth:], back, strict=True)
    )
    assert y_err <= 4.0, f"the core's y: {y_err:.2f} ulps"


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
