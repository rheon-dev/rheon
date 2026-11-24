"""`dist_fit(family="skewt")` against arch: the bank's `loc`, `scale`, `df`
and `lambda` against the maximum of the weighted log likelihood of the same
rows.

Neither scipy nor arch has a weighted fit of Hansen's skewed t (scipy has
no Hansen skewed t at all; arch's `SkewStudent` is a GARCH innovation with
no fit of its own), so the reference is the definition written by hand and
computed by the libraries: the objective `Σ ω_i log f(y_i; loc, scale, df,
λ)`, each `log f` from arch's `SkewStudent.loglikelihood(individual=True)`
at the residual `y - loc` and variance `scale²`, maximized by
`scipy.optimize.minimize` (L-BFGS-B) over `(loc, ln scale, η = 1/df, λ)`
from several starts, the best kept. The quantiles, shortfall, `pit` and
`log_score` are arch's `ppf`, `partial_moment`, `cdf` and log likelihood at
the reported fit.

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
from arch.univariate.distribution import SkewStudent
from scipy import optimize

from rheon import polars as rp

TIER = "essential"

HALF_LIFE = 1500.0
#: `η`'s bounds: `df` from 2.05 (the variance exists above 2) to 10,000,
#: short of the normal's tail at `η = 0`, which arch's density does not take.
ETA_BOUNDS = (1e-4, 1.0 / 2.05)
LAM_BOUNDS = (-0.99, 0.99)

#: The largest gaps from the maximum over every case and row below, measured
#: on Linux x86-64: `|loc - loc*| / scale*` 4.7e-3, `|scale / scale* - 1|`
#: 1.7e-2, `|η - η*|` 1.1e-2 (`η*` from 0.09 to 0.25), `|λ - λ*|` 4.6e-3
#: (`λ*` from -0.61 to 0.44), and the log likelihood the recursion's fit
#: gives up against the maximum, 0.12, where sampling noise alone puts the
#: maximum 9.5 times `Σω²/Σω` (1.0 to 1.3 here) above the truth at 95%. The
#: largest are at the first row checked, 2,000; each is held at about twice
#: its measurement.
LOC_TOL = 1e-2
SCALE_TOL = 3.5e-2
ETA_TOL = 2.5e-2
LAM_TOL = 1e-2
SHORTFALL_TOL = 0.25

DIST = SkewStudent()


def _stream(seed: int, df: float, lam: float, n: int = 6000) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    gaps = rng.exponential(1.0, n)
    gaps[rng.random(n) < 0.03] *= 15.0  # a few long gaps on the clock
    draws = np.asarray(SkewStudent(seed=rng).simulate([df, lam])(n))
    return pl.DataFrame(
        {
            "t": np.cumsum(gaps),
            "w": rng.uniform(0.2, 3.0, n),
            "y": 1.5 + 0.8 * draws,
        }
    )


def _fit(df: pl.DataFrame, floc: float | None) -> pl.DataFrame:
    model: dict[str, object] = {"type": "dist_fit", "family": "skewt"}
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


def _log_density(y: np.ndarray, loc: float, scale: float, eta: float, lam: float) -> np.ndarray:
    """arch's log density of each `y`, at the mean `loc` and standard
    deviation `scale`."""
    resid = y - loc
    return DIST.loglikelihood(
        np.array([1.0 / eta, lam]), resid, np.full_like(resid, scale * scale), individual=True
    )


def _maximum(
    y: np.ndarray, om: np.ndarray, floc: float | None
) -> tuple[tuple[float, float, float, float], float]:
    """scipy's maximum of `Σ ω log f` over `(loc, ln scale, η, λ)`, and the
    maximum's value."""

    def minus(p: np.ndarray) -> float:
        loc = p[0] if floc is None else floc
        lp = _log_density(y, loc, float(np.exp(p[-3])), p[-2], p[-1])
        return -float(np.sum(om * lp)) / om.sum()

    best = None
    for eta, lam in ((0.02, 0.0), (0.2, -0.5), (0.3, 0.5)):
        x0 = [float(np.mean(y)), float(np.log(np.std(y))), eta, lam]
        bounds: list[tuple[float | None, float | None]] = [
            (None, None),
            (None, None),
            ETA_BOUNDS,
            LAM_BOUNDS,
        ]
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
    return (loc, float(np.exp(p[-3])), float(p[-2]), float(p[-1])), -best.fun * om.sum()


@pytest.mark.parametrize(
    ("seed", "df", "lam", "floc"),
    [(0, 5.0, -0.3, None), (1, 4.0, 0.4, None), (2, 10.0, -0.6, None), (3, 6.0, 0.3, 1.5)],
)
def test_loc_scale_df_and_lambda_are_the_weighted_maximum(
    seed: int, df: float, lam: float, floc: float | None
) -> None:
    frame = _stream(seed, df, lam)
    out = _fit(frame, floc)
    t, w, y = (frame[c].to_numpy() for c in ("t", "w", "y"))
    for n in (2000, 4000, len(frame) - 1):
        assert out["withheld_reason_y"][n] is None, n
        loc, scale = out["loc_y"][n], out["scale_y"][n]
        eta, skew = 1.0 / out["df_y"][n], out["lambda_y"][n]
        om = w[:n] * 0.5 ** ((t[n - 1] - t[:n]) / HALF_LIFE)
        (m_loc, m_scale, m_eta, m_lam), top = _maximum(y[:n], om, floc)
        assert abs(loc - m_loc) / m_scale <= LOC_TOL, (n, loc, m_loc)
        assert abs(scale / m_scale - 1.0) <= SCALE_TOL, (n, scale, m_scale)
        assert abs(eta - m_eta) <= ETA_TOL, (n, eta, m_eta)
        assert abs(skew - m_lam) <= LAM_TOL, (n, skew, m_lam)
        given_up = top - float(np.sum(om * _log_density(y[:n], loc, scale, eta, skew)))
        assert -1e-9 <= given_up <= SHORTFALL_TOL, (n, given_up)
        if floc is not None:
            assert loc == floc


def test_the_quantiles_shortfall_and_scores_are_archs_at_the_fit() -> None:
    """Every field that reads the shape is arch's `SkewStudent` at the
    reported `loc`, `scale`, `df` and `lambda`: `ppf`, the shortfall by
    `partial_moment` of order 1 below the quantile over the level, `cdf`
    and the log likelihood."""
    frame = _stream(4, 4.5, -0.4, n=3000)
    spec = rp.spec.dist_fit(
        "d",
        targets=["y"],
        family="skewt",
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
    for row in ready[::50].iter_rows(named=True):
        loc, scale, y = row["loc_y"], row["scale_y"], row["y"]
        par = np.array([row["df_y"], row["lambda_y"]])
        for p in (0.01, 0.5, 0.95):
            want = loc + scale * float(DIST.ppf(p, par))
            assert abs(row[f"q{p}_y"] - want) <= 1e-11 * max(abs(want), 1.0), (p, row)
        q = float(DIST.ppf(0.05, par))
        es = loc + scale * DIST.partial_moment(1, q, par) / 0.05
        assert abs(row["es0.05_y"] - es) <= 1e-10 * abs(es), row
        pit = float(DIST.cdf(np.array([(y - loc) / scale]), par)[0])
        assert abs(row["pit_y"] - pit) <= 1e-11 * max(pit, 1e-300), row
        ls = float(_log_density(np.array([y]), loc, scale, 1.0 / par[0], par[1])[0])
        assert abs(row["log_score_y"] - ls) <= 1e-12 * max(abs(ls), 1.0), row
