"""`garch`'s forecasts at horizons against `arch`'s (task 559).

`arch` forecasts from the variance and the residual it has at an origin;
here it is handed the bank's own, so the library computes the forecast from
the state the bank forecast from. Each row ``t`` of the bank reads the
parameters as they stood after row ``t - 1`` (``coef`` written after each row,
``coef_every=0``) and the lags of row ``t - 1``: its return and the variance it
reported. A subclass of `arch`'s process returns that variance from
`_one_step_forecast` in place of `arch`'s own filter, and `arch`'s
`_analytic_forecast` (GARCH, GJR, FIGARCH) or `_simulation_forecast` (EGARCH)
runs from it at `start = t - 1`: its column `h.h` is the bank's horizon `h`
on row `t`, its `h.1` the row's own variance.

**Why not to the bit.** `arch` squares each forecast's square root through
numpy's vectorized `power` (`np.power(np.sqrt(f), 2)` with an array of
exponents), which on x86-64 with AVX-512 differs from the correctly rounded
square in about 5.7% of values (measured 2026-11-23, numpy 2.5.2, on 10⁷
draws), and from `f` itself where `sqrt` rounds. So `arch`'s last bits are its
platform's and the bank's are not; the bank is held to the recursion
written out to the bit (`tests/test_garch.py`) and to `arch` within a bound in
units of the last place, set from the measured gap. FIGARCH's sum over its
weights is numpy's `dot`, in BLAS's order.

**EGARCH** has no analytic forecast in `arch` past one row: its simulation is
the oracle, with enough draws to resolve the difference between the mean of
the variance and the exponential of the log-variance's mean (the convexity
Nelson's closed form carries): the bank within 4 of the simulation's
standard errors at every horizon, and the exponential of the log-variance's
mean more than 10 away.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest
from arch.univariate import EGARCH, FIGARCH, GARCH

from rheon import polars as rp

TIER = "mixed"

HORIZONS = [1, 2, 5, 22, 100]
N0 = 2000
ROWS = (2100, 2300, 2599)
#: The one-step variance's gap to `arch`'s, in units of its last place.
ULPS = 4


def _bound(vol: str, h: int) -> float:
    """The gap allowed to `arch`'s forecast at horizon `h`, in units of the
    last place of `arch`'s value (and of the sum's). Measured 2026-11-23 on
    Linux x86-64, the largest over 79 rows (15 for FIGARCH) from row 2,050 to
    2,599: GARCH 0, 1, 4, 15 and 37 at horizons 1, 2, 5, 22 and 100, GJR 0,
    2, 4, 13 and 32, their sums at most 30. The bank forms the persistence
    `P = α + γ/2 + β` once and steps `ω + P f`, as the single horizon did,
    where `arch` adds `ω`, `α f`, `γ f/2` and `β f` a step: `P`'s one
    rounding is carried into every step, so the gap grows about linearly in
    the horizon, and the bound is `2 + 0.75 h`, about twice the measured.
    FIGARCH 19, 15, 16, 26 and 20, its sums at most 19, flat in the horizon:
    `arch`'s 1,000-term dot product is in BLAS's order, the bank's in the
    lags'; the bound is 64."""
    return 64.0 if vol == "figarch" else 2.0 + 0.75 * h


def _series(vol: str, n: int, seed: int) -> np.ndarray:
    """The form's process at the truths `tests/test_second_opinion_garch.py`
    holds the estimator at, normal innovations, 1,000 rows discarded."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n + 1000)
    out = np.empty(n + 1000)
    if vol == "figarch":
        from arch.univariate.recursions import figarch_weights

        lam = figarch_weights(np.array([0.2, 0.4, 0.5]), 1, 1, 1000)
        w = 0.03 / (1 - 0.5)
        e2 = np.full(1000, 1.0)
        for t in range(n + 1000):
            s = w + lam @ e2
            out[t] = math.sqrt(s) * z[t]
            e2 = np.concatenate(([out[t] ** 2], e2[:-1]))
        return out[1000:]
    w, a, g, b = (0.03, 0.05, 0.10, 0.87) if vol == "gjr" else (0.05, 0.08, 0.0, 0.9)
    if vol == "egarch":
        w, a, g, b = 0.0, 0.15, -0.08, 0.97
        ln_s, zz = 0.0, 0.0
        for t in range(n + 1000):
            ln_s = w + a * (abs(zz) - math.sqrt(2 / math.pi)) + g * zz + b * ln_s
            zz = z[t]
            out[t] = math.exp(0.5 * ln_s) * zz
        return out[1000:]
    s, e = w / (1 - a - g / 2 - b), 0.0
    for t in range(n + 1000):
        s = w + (a + (g if e < 0 else 0.0)) * e * e + b * s
        e = math.sqrt(s) * z[t]
        out[t] = e
    return out[1000:]


def _bank(vol: str, ys: np.ndarray, horizons: list[int] = HORIZONS, **kw) -> pl.DataFrame:
    spec = rp.spec.garch(
        "m", targets=["y"], vol=vol, start_rows=N0, horizons=horizons, coef_every=0, **kw
    )
    return rp.ModelBank([spec]).fit_predict(pl.DataFrame({"y": ys})).unnest("m")


def _handed(base: type, sigma2: np.ndarray) -> object:
    """`arch`'s process `base`, its one-step forecast the bank's variances."""

    class Handed(base):  # type: ignore[misc, valid-type]
        def _one_step_forecast(self, parameters, resids, backcast, var_bounds, horizon, start):
            return sigma2[: resids.shape[0]], np.zeros((resids.shape[0] - start, horizon))

    return Handed


def _ulps(got: float, want: float) -> float:
    return abs(got - want) / math.ulp(want)


@pytest.mark.parametrize("vol", ["garch", "gjr", "figarch"])
def test_the_forecasts_are_archs_analytic_forecast(vol):
    ys = _series(vol, 2600, 559)
    out = _bank(vol, ys, **({"truncation": 1000} if vol == "figarch" else {}))
    var = out["variance_y"].fill_null(1.0).to_numpy()
    coef = out["coef"].to_list()
    worst = 0.0
    for t in ROWS:
        theta = np.asarray(coef[t - 1], dtype=float)
        if vol == "figarch":
            proc = _handed(FIGARCH, var)(p=1, q=1, truncation=1000)
        else:
            proc = _handed(GARCH, var)(p=1, o=1 if vol == "gjr" else 0, q=1)
        resids = ys[:t]
        bounds = np.column_stack([np.zeros(t), np.full(t, np.inf)])
        fc = proc._analytic_forecast(theta, resids, 1.0, bounds, t - 1, max(HORIZONS)).forecasts
        for h in HORIZONS:
            got = out[f"variance_h{h}_y"][t]
            want = float(fc[0, h - 1])
            worst = max(worst, _ulps(got, want))
            assert _ulps(got, want) <= _bound(vol, h), (vol, t, h, got, want)
            want_sum = math.fsum(fc[0, :h])
            got_sum = out[f"variance_sum_h{h}_y"][t]
            assert _ulps(got_sum, want_sum) <= _bound(vol, h), (vol, t, h, got_sum, want_sum)
        # horizon 1 is the row's own variance, and arch's h.1 from row t - 1
        assert out["variance_h1_y"][t] == out["variance_y"][t]
        assert _ulps(out["variance_y"][t], float(fc[0, 0])) <= _bound(vol, 1)
    print(f"{vol}: worst gap to arch, {worst:.1f} units in the last place")


def _egarch_against_simulation(draws: int, seed: int, resolve: bool) -> None:
    ys = _series("egarch", 2600, 560)
    hs = [1, 2, 3, 5, 10]
    out = _bank("egarch", ys, horizons=hs)
    var = out["variance_y"].fill_null(1.0).to_numpy()
    coef = out["coef"].to_list()
    gen = np.random.default_rng(seed)
    for t in ROWS:
        theta = np.asarray(coef[t - 1], dtype=float)
        proc = _handed(EGARCH, var)(p=1, o=1, q=1)
        bounds = np.column_stack([np.zeros(t), np.full(t, np.inf)])
        f = proc._simulation_forecast(
            theta, ys[:t], 1.0, bounds, t - 1, max(hs), draws, gen.standard_normal
        )
        paths = f.forecast_paths[0]
        # numpy's mean down the draws adds them in order, a rounding a draw;
        # `fsum` rounds once
        mean = np.array([math.fsum(c) for c in paths.T]) / draws
        se = paths.std(axis=0, ddof=1) / math.sqrt(draws)
        ln_mean = np.log(paths).mean(axis=0)
        # the first step is no draw: the row's own variance, by arch's ln and exp
        assert _ulps(out["variance_y"][t], float(mean[0])) <= ULPS, t
        for h in hs[1:]:
            got = out[f"variance_h{h}_y"][t]
            assert abs(got - mean[h - 1]) <= 4 * se[h - 1], (t, h, got, mean[h - 1], se[h - 1])
            if resolve:
                # the draws resolve the convexity: exp(E ln s) lies far outside
                gap = abs(math.exp(ln_mean[h - 1]) - mean[h - 1])
                assert gap > 10 * se[h - 1], (t, h, gap, se[h - 1])


def test_egarch_is_archs_simulation_at_a_few_draws():
    """The bank's EGARCH forecasts within the sampling error of `arch`'s
    simulation, 20,000 draws."""
    _egarch_against_simulation(20_000, 7, resolve=False)


@pytest.mark.extended(
    reason="a million draws of arch's EGARCH simulation at three rows: about 6 s and 200 MB"
)
def test_egarch_is_archs_simulation_with_draws_that_resolve_the_convexity():
    """A million draws: the simulation's standard error is under a tenth of
    the gap between the variance's mean and the exponential of the
    log-variance's mean, and the bank sits within 4 of them."""
    _egarch_against_simulation(1_000_000, 8, resolve=True)
