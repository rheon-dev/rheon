"""`dist_fit(family="t")` against scipy: the bank's `loc`, `scale` and `df`
against the maximum of the weighted log likelihood of the same rows.

scipy has no weighted t fit (`scipy.stats.t.fit` takes no weights), so the
reference is the definition written by hand and computed by scipy: the
objective `Σ ω_i log f(y_i; loc, scale, df)`, each `log f` from
`scipy.stats.t.logpdf`, maximized by `scipy.optimize.minimize` (L-BFGS-B)
over `(loc, ln scale, η = 1/df)`, `η` in `[0, 1]`, from several starts, the
best kept. `η = 0` is the normal, whose `log f` is `scipy.stats.norm.logpdf`.

The weights are the bank's: the fit a row reports is read before the row is
learned, from every earlier row `i` at `w_i 0.5 ** ((t_last - t_i) /
half_life)`, its own weight times the decay over the clock since it,
measured to the last learned row (a factor common to every row, which the
maximum does not see). The clock has irregular gaps, a few long, and the
rows unequal weights.

The bank's fit is recursive: each row moves it to the maximum of the rows
before it, as a quadratic, plus the row's own exact log density. It is not
the exact maximum, and the tolerances below are how far from it the
recursion was measured to be, rounded up.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from scipy import optimize, stats

from rheon import polars as rp

TIER = "essential"

HALF_LIFE = 1500.0

#: The largest gaps from scipy's maximum over every case and row below,
#: measured on Linux x86-64: `|loc - loc*| / scale*` 1.1e-3, `|scale /
#: scale* - 1|` 6.2e-3, `|η - η*|` 9.6e-3 (`η* ` from 0.06 to 0.37), and the
#: log likelihood the recursion's fit gives up against the maximum, 0.090,
#: where sampling noise alone puts the maximum 7.8 times `Σω²/Σω` (about 1
#: here) above the truth at 95%. Each held at about twice its measurement.
LOC_TOL = 2e-3
SCALE_TOL = 1.5e-2
ETA_TOL = 2e-2
SHORTFALL_TOL = 0.2


def _stream(seed: int, df: float, n: int = 6000) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    gaps = rng.exponential(1.0, n)
    gaps[rng.random(n) < 0.03] *= 15.0  # a few long gaps on the clock
    return pl.DataFrame(
        {
            "t": np.cumsum(gaps),
            "w": rng.uniform(0.2, 3.0, n),
            "y": 1.5 + 0.8 * rng.standard_t(df, n),
        }
    )


def _fit(df: pl.DataFrame, floc: float | None) -> pl.DataFrame:
    model: dict[str, object] = {"type": "dist_fit", "family": "t"}
    if floc is not None:
        model["floc"] = floc
    spec = {
        "name": "d",
        "model": model,
        "targets": ["y"],
        "features": [],
        "half_life": HALF_LIFE,
        "clock": "t",
        "gap_cap": 1e6,  # past every gap: each decays in full
        "weight": "w",
    }
    return rp.ModelBank([spec]).fit_predict(df).unnest("d")


def _log_density(y: np.ndarray, loc: float, scale: float, eta: float) -> np.ndarray:
    if eta <= 0.0:
        return stats.norm.logpdf(y, loc, scale)
    return stats.t.logpdf(y, 1.0 / eta, loc, scale)


def _maximum(
    y: np.ndarray, om: np.ndarray, floc: float | None
) -> tuple[tuple[float, float, float], float]:
    """scipy's maximum of `Σ ω log f` over `(loc, ln scale, η)`, and the
    maximum's value."""

    def minus(p: np.ndarray) -> float:
        loc = p[0] if floc is None else floc
        return -float(np.sum(om * _log_density(y, loc, np.exp(p[-2]), p[-1]))) / om.sum()

    best = None
    for eta in (0.01, 0.2, 0.5):
        x0 = [float(np.median(y)), float(np.log(np.std(y))), eta]
        bounds: list[tuple[float | None, float | None]] = [(None, None), (None, None), (0.0, 1.0)]
        if floc is not None:
            x0, bounds = x0[1:], bounds[1:]
        r = optimize.minimize(
            minus,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"ftol": 1e-15, "gtol": 1e-12, "maxiter": 10_000},
        )
        if best is None or r.fun < best.fun:
            best = r
    assert best is not None and best.success, best
    p = best.x
    loc = float(p[0]) if floc is None else floc
    return (loc, float(np.exp(p[-2])), float(p[-1])), -best.fun * om.sum()


@pytest.mark.parametrize(
    ("seed", "df", "floc"), [(0, 5.0, None), (1, 3.0, None), (2, 10.0, None), (3, 5.0, 1.5)]
)
def test_loc_scale_and_df_are_the_weighted_maximum(
    seed: int, df: float, floc: float | None
) -> None:
    frame = _stream(seed, df)
    out = _fit(frame, floc)
    t, w, y = (frame[c].to_numpy() for c in ("t", "w", "y"))
    for n in (2000, 4000, len(frame) - 1):
        assert out["withheld_reason_y"][n] is None, n
        loc, scale, dof = out["loc_y"][n], out["scale_y"][n], out["df_y"][n]
        om = w[:n] * 0.5 ** ((t[n - 1] - t[:n]) / HALF_LIFE)
        (m_loc, m_scale, m_eta), top = _maximum(y[:n], om, floc)
        assert abs(loc - m_loc) / m_scale <= LOC_TOL, (n, loc, m_loc)
        assert abs(scale / m_scale - 1.0) <= SCALE_TOL, (n, scale, m_scale)
        assert abs(1.0 / dof - m_eta) <= ETA_TOL, (n, 1.0 / dof, m_eta)
        given_up = top - float(np.sum(om * _log_density(y[:n], loc, scale, 1.0 / dof)))
        assert -1e-9 <= given_up <= SHORTFALL_TOL, (n, given_up)
        if floc is not None:
            assert loc == floc


def test_the_quantiles_shortfall_and_scores_are_scipys_at_the_fit() -> None:
    """Every field that reads the tail is `scipy.stats.t`'s at the reported
    `loc`, `scale` and `df`: `ppf`, the shortfall by `expect` below the
    quantile over the level, `cdf` and `logpdf`."""
    frame = _stream(4, 4.0, n=3000)
    spec = rp.spec.dist_fit(
        "d",
        targets=["y"],
        family="t",
        quantiles=[0.01, 0.5, 0.95],
        es_levels=[0.05],
        half_life=HALF_LIFE,
        clock="t",
        gap_cap=1e6,
        weight="w",
    )
    out = rp.ModelBank([spec]).fit_predict(frame).unnest("d")
    ready = out.filter(pl.col("withheld_reason_y").is_null())
    assert ready.height > 2500
    rows = ready[::50]
    loc, scale, dof, y = (rows[c].to_numpy() for c in ("loc_y", "scale_y", "df_y", "y"))
    for p in (0.01, 0.5, 0.95):
        np.testing.assert_allclose(
            rows[f"q{p}_y"].to_numpy(), stats.t.ppf(p, dof, loc, scale), rtol=1e-11, atol=1e-12
        )
    want = [
        stats.t.expect(lambda x: x, args=(d,), loc=m, scale=s, ub=stats.t.ppf(0.05, d, m, s)) / 0.05
        for m, s, d in zip(loc, scale, dof, strict=True)
    ]
    np.testing.assert_allclose(rows["es0.05_y"].to_numpy(), want, rtol=1e-7)
    np.testing.assert_allclose(
        rows["pit_y"].to_numpy(), stats.t.cdf(y, dof, loc, scale), rtol=1e-11
    )
    np.testing.assert_allclose(
        rows["log_score_y"].to_numpy(), stats.t.logpdf(y, dof, loc, scale), rtol=1e-12
    )
