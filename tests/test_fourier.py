"""Fourier seasonality terms as Polars expressions (docs/PLAN.md tasks 489).

``fourier`` reduces the phase in integers, ``r_i = (i (t - t0)) mod P``, so
each harmonic's ``u_i = r_i / P`` is the correctly rounded quotient of the
same rational that statsmodels' ``Fourier(n, 1).in_sample`` divides at row
``r_i / s`` of a clock with step ``s`` and ``n = P / s``: the terms are held
to that lookup to 0 ulps (the study's §3.1 cases, its §3.5 for why the
reduction). ``calendar_fourier`` is statsmodels' ``CalendarFourier`` to the
bit, on ns and us indexes, at the instants around every month's start
(§3.3). Both are statsmodels' public ``in_sample``; the sines compare to
the bit because numpy's ``sin`` and Rust's are the platform's libm here.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import polars as pl
import pytest
from statsmodels.tsa.deterministic import CalendarFourier, Fourier

from rheon.polars._fourier import calendar_fourier, fourier

TIER = "essential"

NS = 10**9
#: 2026-01-01T00:00, a Thursday.
T2026 = 1_767_225_600 * NS


def _bits_differ(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.sum(a.view(np.int64) != b.view(np.int64)))


def _lookup(t: np.ndarray, t0: int, step: int, n: int, order: int) -> np.ndarray:
    """statsmodels' terms by the reduced position: harmonic i at row
    ``(i b) mod n`` of ``Fourier(n, 1).in_sample(RangeIndex(n))``."""
    grid = Fourier(n, 1).in_sample(pd.RangeIndex(n)).to_numpy()
    b = np.mod(t - t0, n * step) // step
    out = np.empty((t.size, 2 * order))
    for i in range(1, order + 1):
        row = np.mod(b * i, n)
        out[:, 2 * i - 2] = grid[row, 0]
        out[:, 2 * i - 1] = grid[row, 1]
    return out


@pytest.mark.parametrize(
    ("period", "n", "order", "step", "rows"),
    [
        ("1d", 24, 6, 3600 * NS, 200_000),  # hourly rows, a daily season
        ("1w", 7, 3, 86400 * NS, 20_000),  # daily rows, a weekly season from Monday
        ("1w", 168, 10, 3600 * NS, 200_000),
        ("1d", 1440, 8, 60 * NS, 525_600),  # a year of minutes, past 2**53 ns from t0
        ("60ns", 60, 5, 1, 200_000),  # a 1 ns step
    ],
)
def test_fourier_is_statsmodels_at_the_reduced_position(period, n, order, step, rows):
    t = T2026 + np.arange(rows, dtype=np.int64) * step
    frame = pl.DataFrame({"t": pl.Series(t).cast(pl.Datetime("ns"))})
    mine = frame.select(fourier("t", period=period, order=order)).to_numpy()
    t0 = 4 * 86400 * NS if period == "1w" else 0  # Monday 1970-01-05 for whole weeks
    assert _bits_differ(mine, _lookup(t, t0, step, n, order)) == 0


def test_fourier_on_an_integer_clock_and_an_origin():
    """An integer clock in its own units, the period an int, a negative
    clock reduced by the floor mod, and an origin that moves phase 0."""
    t = np.arange(-5_000, 45_000, dtype=np.int64) * 3
    frame = pl.DataFrame({"t": t})
    mine = frame.select(fourier("t", period=36, order=4, origin=7 * 3)).to_numpy()
    assert _bits_differ(mine, _lookup(t, 21, 3, 12, 4)) == 0
    names = frame.select(fourier("t", period=36, order=2)).columns
    assert names == ["t_sin1_36", "t_cos1_36", "t_sin2_36", "t_cos2_36"]


def test_fourier_on_a_date_and_a_us_clock_is_the_ns_clock():
    days = pl.date_range(dt.date(2023, 12, 25), dt.date(2026, 3, 1), eager=True).alias("t")
    as_date = pl.DataFrame({"t": days}).select(fourier("t", period="1w", order=3)).to_numpy()
    as_us = (
        pl.DataFrame({"t": days.cast(pl.Datetime("us"))})
        .select(fourier("t", period="7d", order=3))
        .to_numpy()
    )
    t = days.cast(pl.Datetime("ns")).dt.epoch("ns").to_numpy()
    assert _bits_differ(as_date, as_us) == 0
    assert _bits_differ(as_date, _lookup(t, 4 * 86400 * NS, 86400 * NS, 7, 3)) == 0


def _calendar_instants() -> np.ndarray:
    rng = np.random.default_rng(489)
    lo = np.datetime64("2019-01-01", "ns").astype(np.int64)
    hi = np.datetime64("2027-01-01", "ns").astype(np.int64)
    t = rng.integers(lo, hi, size=50_000, dtype=np.int64)
    starts = np.arange("2019-01", "2027-01", dtype="datetime64[M]").astype("datetime64[ns]")
    edge = np.concatenate([starts.astype(np.int64) + k for k in (-1, 0, 1)])
    extra = np.array(
        ["2024-02-29T23:59:59.999999999", "2024-12-31T23:59:59.999999999", "2020-12-31T12:00"],
        dtype="datetime64[ns]",
    ).astype(np.int64)
    return np.unique(np.concatenate([t, edge, extra]))


@pytest.mark.parametrize("unit", ["ns", "us"])
@pytest.mark.parametrize(
    ("freq", "sm_freq", "order"),
    [
        ("1h", "h", 2),
        ("1d", "D", 4),
        ("1w", "W", 3),
        ("1mo", "ME", 3),
        ("1q", "QE", 2),
        ("1y", "YE", 6),
    ],
)
def test_calendar_fourier_is_statsmodels_to_the_bit(unit, freq, sm_freq, order):
    t = _calendar_instants()
    if unit == "us":
        t = np.unique(np.floor_divide(t, 1000))
    index = pd.DatetimeIndex(t.astype(f"datetime64[{unit}]"))
    frame = pl.DataFrame({"t": pl.Series(t).cast(pl.Datetime(unit))})
    mine = frame.select(calendar_fourier("t", freq=freq, order=order)).to_numpy()
    oracle = CalendarFourier(sm_freq, order).in_sample(index).to_numpy()
    assert _bits_differ(mine, oracle) == 0


def test_a_zoned_clock_is_read_on_its_wall_clock():
    """A zoned clock gives the terms of its naive wall clock, across both
    daylight-saving changes of the year."""
    wall = pl.datetime_range(
        dt.datetime(2026, 1, 1), dt.datetime(2026, 12, 31), "37m", eager=True
    ).alias("t")
    zoned = wall.dt.replace_time_zone("Europe/Paris", ambiguous="earliest", non_existent="null")
    keep = zoned.is_not_null()
    naive = pl.DataFrame({"t": wall.filter(keep)})
    aware = pl.DataFrame({"t": zoned.filter(keep)})
    for build in (
        lambda: fourier("t", period="1d", order=3),
        lambda: calendar_fourier("t", freq="1d", order=3),
        lambda: calendar_fourier("t", freq="1w", order=2),
    ):
        assert _bits_differ(aware.select(build()).to_numpy(), naive.select(build()).to_numpy()) == 0


def test_terms_stream_and_ignore_chunking():
    t = T2026 + np.arange(30_000, dtype=np.int64) * 61 * NS
    frame = pl.DataFrame({"t": pl.Series(t).cast(pl.Datetime("ns"))})
    exprs = [*fourier("t", period="1h", order=2), *calendar_fourier("t", freq="1d", order=2)]
    whole = frame.select(exprs)
    streamed = frame.lazy().select(exprs).collect(engine="streaming")
    chunked = pl.concat([frame.slice(i, 997) for i in range(0, frame.height, 997)], rechunk=False)
    assert whole.equals(streamed)
    assert whole.equals(chunked.select(exprs))


def test_a_null_clock_gives_null_terms():
    frame = pl.DataFrame({"t": [dt.datetime(2026, 1, 1, 6), None]})
    out = frame.select(
        fourier("t", period="1d", order=1) + calendar_fourier("t", freq="1d", order=1)
    )
    assert out.row(0) == pytest.approx((1.0, 0.0, 1.0, 0.0), abs=1e-15)
    assert out.row(1) == (None, None, None, None)


def test_refusals():
    with pytest.raises(pl.exceptions.InvalidOperationError):
        pl.DataFrame({"f": [1.5]}).select(fourier("f", period=24, order=1))
    with pytest.raises(pl.exceptions.InvalidOperationError):
        pl.DataFrame({"i": [1]}).select(fourier("i", period="1d", order=1))
    with pytest.raises(ValueError, match="calendar_fourier"):
        fourier("t", period="1mo", order=1)
    with pytest.raises(ValueError, match="2\\*\\*63"):
        fourier("t", period="365d", order=300)
    with pytest.raises(ValueError, match="at least 1"):
        fourier("t", period="1d", order=0)
    with pytest.raises(TypeError, match="period"):
        fourier("t", period=24.0, order=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="time zone"):
        fourier("t", period="1d", order=1, origin=dt.datetime(2026, 1, 1, tzinfo=dt.UTC))
    with pytest.raises(ValueError, match="fixed span"):
        calendar_fourier("t", freq="2d", order=1)
