"""``rp.clock_drift`` against statsmodels' ``WLS``, and ``rp.clock_step``
against statsmodels' ``DescrStatsW`` (task 564).

The slope of the lag ``d = arrival - event`` on the arrival time, each row
weighing ``2 ** (-(t_now - t_i) / half_life)`` on the stream's clock, is the
weighted least-squares slope of ``d`` on ``arrival`` with a constant. Only
the weights are written here, from the definition; statsmodels fits the
line. Its intercept's mean is checked too: the fitted line passes through
the weighted means, so ``intercept + slope * a_bar`` is the lag's weighted
mean, which ``rp.clock_lag`` gives.

The tolerance: ``WLS`` solves by a pseudo-inverse of the weighted design, and
the operator updates two sums about running means, so the two share no
arithmetic. On arrivals measured from the first one, the slope agrees to
1e-9 relatively here, on every row, at each half-life.

A clock step's mean and variance on a temporal clock are the weighted mean
and variance of the steps so far, each weighing ``2 ** (-(t_now - t_i) /
half_life)``: the weights are written here, and ``DescrStatsW`` computes the
statistics (``ddof=0``, the plug-in form; the corrected form divides it by
``1 - sum w**2 / W**2``, Kish's correction, written here from its
definition). ``DescrStatsW`` sums in two passes and the operator updates
about a running mean, sharing no arithmetic: they agree to 1e-9 relatively.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest
import statsmodels.api as sm
from statsmodels.stats.weightstats import DescrStatsW

from rheon import polars as rp

TIER = "essential"


def _frame(n: int, seed: int) -> pl.DataFrame:
    """Arrivals on an uneven clock, a lag that grows 4 ms a second and
    wobbles, and two rows with no event time."""
    rng = np.random.default_rng(seed)
    start = datetime(2026, 3, 2, 14, 0)
    steps = np.cumsum(rng.integers(50, 3000, n))
    arrival = [start + timedelta(milliseconds=int(s)) for s in steps]
    lag_ms = 25 + 0.004 * steps + rng.normal(0, 2, n)
    event: list[datetime | None] = [
        a - timedelta(microseconds=round(1000 * float(m)))
        for a, m in zip(arrival, lag_ms, strict=True)
    ]
    event[4] = event[60] = None
    return pl.DataFrame({"arrival": arrival, "event": event}).with_columns(
        pl.col("arrival", "event").dt.cast_time_unit("ns")
    )


@pytest.mark.parametrize("half_life", ["5s", "1m", None])
def test_the_drift_is_statsmodels_wls(half_life: str | None) -> None:
    df = _frame(200, 17)
    h = float("inf") if half_life is None else {"5s": 5.0, "1m": 60.0}[half_life]
    span = float("inf") if half_life is None else half_life
    out = rp.stream.with_windows(
        df,
        lag=rp.clock_lag("arrival", "event", half_life=span),
        drift=rp.clock_drift("arrival", "event", half_life=span),
        clock="arrival",
        gap_cap="1h",
    )
    a_ns = df["arrival"].dt.epoch("ns").to_numpy().astype(np.int64)
    e_ns = df["event"].dt.epoch("ns").fill_null(0).to_numpy().astype(np.int64)
    has = df["event"].is_not_null().to_numpy()
    first = a_ns[has][0]
    a = (a_ns - first) / 1e9
    d = (a_ns - e_ns) / 1e9
    checked = 0
    for i in range(len(df)):
        rows = np.flatnonzero(has[: i + 1])
        if len(rows) < 2:
            assert out["drift"][i] is None, i
            continue
        w = np.exp2(-((a_ns[i] - a_ns[rows]) / 1e9) / h)
        fit = sm.WLS(d[rows], sm.add_constant(a[rows]), weights=w).fit()
        intercept, slope = fit.params
        assert out["drift"][i] == pytest.approx(slope, rel=1e-9), i
        a_bar = np.average(a[rows], weights=w)
        assert out["lag"][i] == pytest.approx(intercept + slope * a_bar, rel=1e-9), i
        checked += 1
    assert checked > 190
    # The lag grows 4 ms a second; the slope over the stream reads it.
    if half_life is None:
        assert out["drift"][-1] == pytest.approx(0.004, rel=0.05)


@pytest.mark.parametrize("half_life", ["2s", "30s", None])
def test_a_clock_step_is_statsmodels_descrstatsw(half_life: str | None) -> None:
    rng = np.random.default_rng(23)
    n = 200
    start = datetime(2026, 3, 2, 14, 0)
    # Repeats (a step of 0) and uneven steps, on a clock of nanoseconds.
    steps_ms = np.cumsum(rng.choice([0, 1, 250, 1000, 4000], size=n))
    df = pl.DataFrame(
        {"t": [start + timedelta(milliseconds=int(s)) for s in steps_ms]}
    ).with_columns(pl.col("t").dt.cast_time_unit("ns"))
    h = float("inf") if half_life is None else {"2s": 2.0, "30s": 30.0}[half_life]
    span = float("inf") if half_life is None else half_life
    out = rp.stream.with_windows(
        df,
        m=rp.clock_step("t", half_life=span),
        b=rp.clock_step("t", half_life=span, stat="var", bias=True),
        u=rp.clock_step("t", half_life=span, stat="std"),
        clock="t",
        gap_cap="1h",
    )
    t_ns = df["t"].dt.epoch("ns").to_numpy().astype(np.int64)
    s = np.diff(t_ns) / 1e9
    assert out["m"][0] is None and out["b"][0] is None
    for i in range(1, n):
        w = np.exp2(-((t_ns[i] - t_ns[1 : i + 1]) / 1e9) / h)
        d = DescrStatsW(s[:i], weights=w, ddof=0)
        assert out["m"][i] == pytest.approx(d.mean, rel=1e-9, abs=1e-15), i
        assert out["b"][i] == pytest.approx(d.var, rel=1e-9, abs=1e-15), i
        r2 = float(np.sum(w**2) / np.sum(w) ** 2)
        if i == 1:
            assert out["u"][i] is None
        else:
            assert out["u"][i] == pytest.approx(np.sqrt(d.var / (1 - r2)), rel=1e-9), i
