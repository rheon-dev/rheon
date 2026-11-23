"""`dist_fit` against statsmodels: the bank's `loc` and `scale` against
`DescrStatsW`'s weighted mean and standard deviation of the same rows.

The fit a row reports is read before the row is learned, from every earlier
row `i` at the weight `w_i 0.5 ** ((t_last - t_i) / half_life)`: its own
weight times the decay over the clock since it, measured to the last learned
row (a factor common to every row, which the weighted mean and variance do
not see). `DescrStatsW(ddof=0)` computes the normal's weighted
maximum-likelihood fit from those weights directly, two-pass, with none of
the bank's recursion: the mean, and the root of the weighted mean square
about it. The clock has irregular gaps and the rows unequal weights, so a
recursion that decayed per row rather than per clock unit, or dropped a
weight, would be off by far more than the tolerance.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from statsmodels.stats.weightstats import DescrStatsW

from rheon import polars as rp

TIER = "essential"

HALF_LIFE = 40.0

#: The largest relative gap of `loc` and of `scale` from statsmodels', over
#: every row of every case below (relative to the larger of the value and 1).
#: Measured at 6.8e-16 (`loc`) and 7.5e-16 (`scale`) on Linux x86-64, a few
#: roundings; held at 1e-14, the measurement rounded up by about a factor
#: of 13, room for the decay factor's last bits from another OS's libm.
RTOL = 1e-14


def _stream(seed: int, level: float, n: int = 1500) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    gaps = rng.exponential(1.0, n)
    gaps[rng.random(n) < 0.05] *= 20.0  # a few long gaps on the clock
    return pl.DataFrame(
        {
            "t": np.cumsum(gaps),
            "w": rng.uniform(0.2, 3.0, n),
            "y": level + 2.0 * rng.standard_normal(n),
        }
    )


def _fit(df: pl.DataFrame) -> pl.DataFrame:
    spec = {
        "name": "d",
        "model": {"type": "dist_fit", "quantiles": [0.05], "es_levels": [0.05]},
        "targets": ["y"],
        "features": [],
        "half_life": HALF_LIFE,
        "clock": "t",
        "gap_cap": 1e6,  # past every gap: each decays in full
        "weight": "w",
    }
    return rp.ModelBank([spec]).fit_predict(df).unnest("d")


@pytest.mark.parametrize(("seed", "level"), [(0, 0.0), (1, 5.0), (2, -1000.0)])
def test_loc_and_scale_are_statsmodels_weighted_fit(seed: int, level: float) -> None:
    df = _stream(seed, level)
    out = _fit(df)
    t, w, y = (df[c].to_numpy() for c in ("t", "w", "y"))
    loc, scale = out["loc_y"].to_numpy(), out["scale_y"].to_numpy()
    worst = [0.0, 0.0]
    for n in range(2, len(df)):
        ow = w[:n] * 0.5 ** ((t[n - 1] - t[:n]) / HALF_LIFE)
        ref = DescrStatsW(y[:n], weights=ow, ddof=0)
        for k, (got, want) in enumerate(((loc[n], ref.mean), (scale[n], ref.std))):
            gap = abs(got - want) / max(abs(want), 1.0)
            worst[k] = max(worst[k], gap)
    assert worst[0] <= RTOL, f"loc off statsmodels by {worst[0]:.3g} relative"
    assert worst[1] <= RTOL, f"scale off statsmodels by {worst[1]:.3g} relative"


def test_the_first_two_rows_say_why_they_report_nothing() -> None:
    out = _fit(_stream(3, 1.0))
    reasons = out["withheld_reason_y"].cast(pl.String).to_list()
    assert reasons[:3] == ["no_weight", "degenerate_variance", None]
    assert out["loc_y"][:2].is_null().all() or out["loc_y"][:2].is_nan().all()
    assert out["scale_y"][2] > 0.0
