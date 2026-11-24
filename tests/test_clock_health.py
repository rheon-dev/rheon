"""``rp.run_length``, ``rp.clock_lag`` and ``rp.clock_drift`` (task 564), the
row operators a stream's health is read from.

The run length is held to Polars' ``rle``: the run lengths, read row by row
as each row's place in its run, on the series with every value the window
engine does not read (NaN, an infinity, beyond 1e100) set to null, since the
engine reads those as missing. Under a ``tol`` above 0, which ``rle`` does not
take, it is held to its definition, written with Polars' own expressions.
The lag is held to numpy's weighted mean; the slope to statsmodels' ``WLS`` in
``test_second_opinion_clock_health.py``.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import numpy as np
import polars as pl
import pytest

from rheon import polars as rp

TIER = "essential"

BIG = 1e100


def _values(n: int, seed: int) -> pl.Series:
    """Values that hold for stretches, with nulls, NaNs, infinities, a value
    past 1e100, both zeros and repeats of each, in runs of 1 to 5."""
    rng = np.random.default_rng(seed)
    pool: list[float | None] = [
        1.0,
        2.0,
        0.0,
        -0.0,
        None,
        float("nan"),
        float("inf"),
        -float("inf"),
        BIG,
        np.nextafter(BIG, np.inf),
    ]
    out: list[float | None] = []
    while len(out) < n:
        out += [pool[rng.integers(len(pool))]] * int(rng.integers(1, 6))
    return pl.Series("x", out[:n], dtype=pl.Float64)


def _rle_rows(s: pl.Series) -> list[float]:
    """Polars' ``rle`` read row by row: each row's place in its run."""
    lengths = s.rle().struct.field("len").to_list()
    return [float(i + 1) for n in lengths for i in range(n)]


def _missing_as_null(s: pl.Series) -> pl.Series:
    """The engine's reading: a value that is not finite or is beyond 1e100
    is missing, as a null is."""
    return pl.select(pl.when(s.is_finite() & (s.abs() <= BIG)).then(s).otherwise(None)).to_series()


@pytest.mark.parametrize("chunk_size", [1, 7, None])
def test_a_run_length_is_polars_rle_row_by_row(chunk_size: int | None) -> None:
    s = _values(500, 7)
    df = s.to_frame()
    kwargs: dict[str, Any] = {} if chunk_size is None else {"chunk_size": chunk_size}
    out = rp.stream.with_windows(df, n=rp.run_length("x"), **kwargs)
    assert out["n"].to_list() == _rle_rows(_missing_as_null(s))


def test_nulls_are_runs_of_their_own_as_in_rle() -> None:
    # Both sides of each rule, against rle itself: two nulls are one run; a
    # null next to a value ends it on both sides; 0.0 and -0.0 are one value.
    s = pl.Series("x", [1.0, 1.0, None, None, 1.0, 0.0, -0.0, 2.0, None])
    out = rp.stream.with_windows(s.to_frame(), n=rp.run_length("x"))
    assert out["n"].to_list() == _rle_rows(s) == [1, 2, 1, 2, 1, 1, 2, 1, 1]
    # A NaN is missing here, so it joins a run of nulls where rle keeps it
    # apart; a value at 1e100 is a value, and the next double past it is not.
    s = pl.Series("x", [None, float("nan"), None, BIG, BIG, np.nextafter(BIG, np.inf)])
    out = rp.stream.with_windows(s.to_frame(), n=rp.run_length("x"))
    assert out["n"].to_list() == [1.0, 2.0, 3.0, 1.0, 2.0, 1.0]
    assert _rle_rows(s) == [1.0, 1.0, 1.0, 1.0, 2.0, 1.0]
    assert out["n"].to_list() == _rle_rows(_missing_as_null(s))


def _creeping_runs(x: pl.Series, tol: float) -> list[float]:
    """The definition under a ``tol``, in Polars' expressions: a row starts
    a run unless it and the row before are both missing, or both numbers at
    most ``tol`` apart; each row's place in its run."""
    df = pl.DataFrame({"x": _missing_as_null(x)})
    prev = pl.col("x").shift(1)
    same = (pl.col("x").is_null() & prev.is_null()) | ((pl.col("x") - prev).abs() <= tol)
    starts = (~same.fill_null(False)).cast(pl.Int64)
    starts = pl.when(pl.int_range(pl.len()) == 0).then(1).otherwise(starts)
    return (
        df.with_columns(run=starts.cum_sum())
        .select(pl.int_range(pl.len()).over("run") + 1)
        .to_series()
        .cast(pl.Float64)
        .to_list()
    )


@pytest.mark.parametrize("tol", [0.0, 0.5, 1.0, 3.0])
def test_a_run_under_a_tolerance_is_its_definition(tol: float) -> None:
    s = _values(400, 11)
    out = rp.stream.with_windows(s.to_frame(), n=rp.run_length("x", tol=tol), chunk_size=13)
    assert out["n"].to_list() == _creeping_runs(s, tol)
    if tol == 0.0:
        assert out["n"].to_list() == _rle_rows(_missing_as_null(s))


def test_tol_is_inclusive() -> None:
    # Both sides: a step of exactly tol stays in the run, the next double
    # past it ends the run.
    edge = np.nextafter(1.5, np.inf)
    df = pl.DataFrame({"x": [1.0, 1.5, 1.0, edge]})
    out = rp.stream.with_windows(df, n=rp.run_length("x", tol=0.5))
    assert out["n"].to_list() == [1.0, 2.0, 3.0, 1.0]


def test_a_run_is_each_groups_and_restarts_at_a_session_not_at_a_gap() -> None:
    df = pl.DataFrame(
        {
            "t": [0.0, 1.0, 2.0, 100.0, 101.0, 102.0, 103.0],
            "g": ["a", "b", "a", "a", "b", "a", "a"],
            "s": [1, 1, 1, 1, 1, 2, 2],
            "x": [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
        }
    )
    out = rp.stream.with_windows(
        df, n=rp.run_length("x"), clock="t", gap_cap=10.0, group="g", session="s"
    )
    # a: 1, 2, then a gap of 98 keeps the run (3), a new session starts one.
    assert out["n"].to_list() == [1.0, 1.0, 2.0, 3.0, 2.0, 1.0, 2.0]


def test_a_run_is_saved_and_resumed_at_every_row(tmp_path: Any) -> None:
    s = _values(60, 3)
    df = pl.DataFrame({"t": np.arange(60.0), "x": s})
    ops = {
        "n": rp.run_length("x"),
        "lag": rp.clock_lag("t", pl.col("t") - 0.01 * pl.col("t"), half_life=5.0),
        "drift": rp.clock_drift("t", pl.col("t") - 0.01 * pl.col("t"), half_life=5.0),
    }
    one = rp.stream.with_windows(df, **ops, clock="t", gap_cap=10.0)
    for at in (1, 2, 17, 33, 59):
        state = tmp_path / f"w{at}.state"
        first = rp.stream.with_windows(df[:at], **ops, clock="t", gap_cap=10.0, save_state=state)
        second = rp.stream.with_windows(df[at:], **ops, clock="t", gap_cap=10.0, load_state=state)
        assert pl.concat([first, second]).equals(one), at


def _drift_frame(n: int, seed: int) -> pl.DataFrame:
    """Arrivals on an uneven clock with a lag that grows and wobbles, and a
    row without an event time."""
    rng = np.random.default_rng(seed)
    start = datetime(2026, 1, 5, 9, 30)
    steps = np.cumsum(rng.integers(100, 2000, n))
    arrival = [start + timedelta(milliseconds=int(s)) for s in steps]
    lag_ms = 40 + 0.004 * steps + rng.normal(0, 3, n)
    event: list[datetime | None] = [
        a - timedelta(milliseconds=float(m)) for a, m in zip(arrival, lag_ms, strict=True)
    ]
    event[9] = None
    return pl.DataFrame({"arrival": arrival, "event": event}).with_columns(
        pl.col("arrival", "event").dt.cast_time_unit("ns")
    )


def _weights(df: pl.DataFrame, upto: int, half_life_s: float) -> np.ndarray:
    t = df["arrival"][: upto + 1].dt.epoch("ns").to_numpy().astype(np.int64)
    return np.exp2(-((t[-1] - t) / 1e9) / half_life_s)


def test_the_lag_is_numpys_weighted_mean() -> None:
    df = _drift_frame(150, 5)
    out = rp.stream.with_windows(
        df,
        lag=rp.clock_lag("arrival", "event", half_life="10s"),
        clock="arrival",
        gap_cap="1h",
    )
    d = (df["arrival"] - df["event"]).dt.total_nanoseconds().to_numpy() / 1e9
    has = ~np.isnan(d)
    for i in range(len(df)):
        w = _weights(df, i, 10.0) * has[: i + 1]
        want = np.average(np.nan_to_num(d[: i + 1]), weights=w)
        assert out["lag"][i] == pytest.approx(want, rel=1e-12, abs=1e-15), i


def test_the_slope_is_withheld_while_the_arrivals_have_no_spread() -> None:
    # Both sides: one row, then rows at the one arrival, have no slope; the
    # first row at another arrival has one.
    df = pl.DataFrame({"a": [5.0, 5.0, 5.0, 6.0, 6.0], "e": [4.0, 4.5, 3.0, 5.0, 5.5]})
    out = rp.stream.with_windows(
        df,
        lag=rp.clock_lag("a", "e", half_life=float("inf")),
        drift=rp.clock_drift("a", "e", half_life=float("inf")),
        clock="a",
        gap_cap=10.0,
    )
    assert out["drift"].to_list()[:3] == [None, None, None]
    assert out["drift"].to_list()[3] is not None
    assert out["lag"].to_list()[0] == 1.0
    # The lag is withheld until a row has both times.
    df = pl.DataFrame({"a": [1.0, 2.0, 3.0], "e": [None, None, 2.5]})
    out = rp.stream.with_windows(
        df, lag=rp.clock_lag("a", "e", half_life=4.0), clock="a", gap_cap=10.0
    )
    assert out["lag"].to_list() == [None, None, 0.5]


def test_the_drift_restarts_at_a_gap_past_gap_cap() -> None:
    # Both sides of the cap: a step of exactly gap_cap keeps the sums, a step
    # past it starts them again, so the slope is withheld at the next row.
    rows = {"a": [0.0, 1.0, 2.0, 12.0, 22.5, 23.0], "e": [0.0, 0.5, 1.0, 6.0, 11.0, 11.0]}
    out = rp.stream.with_windows(
        pl.DataFrame(rows),
        drift=rp.clock_drift("a", "e", half_life=float("inf")),
        clock="a",
        gap_cap=10.0,
    )
    got = out["drift"].to_list()
    assert got[3] == pytest.approx(0.5) and got[4] is None and got[5] is not None


def test_what_the_operators_refuse() -> None:
    df = pl.DataFrame({"t": [1.0, 2.0], "x": [1.0, 2.0], "s": ["a", "b"]})
    with pytest.raises(ValueError, match="tol must be finite and at least 0"):
        rp.run_length("x", tol=-0.1)
    with pytest.raises(ValueError, match="tol must be finite and at least 0"):
        rp.run_length("x", tol=float("nan"))
    with pytest.raises(TypeError, match="tol must be a number"):
        rp.run_length("x", tol="0")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="holds the operator increment"):
        rp.run_length(rp.increment("x"))
    with pytest.raises(ValueError, match="another window operator"):
        rp.ewm_mean(rp.run_length("x"), half_life=1.0)
    with pytest.raises(TypeError, match="half_life is required"):
        rp.clock_lag("t", "x", half_life=None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="half_life must be above 0"):
        rp.clock_drift("t", "x", half_life=0.0)
    with pytest.raises(rp.WrongTypeError, match="must be a number or a temporal column"):
        rp.stream.with_windows(df, n=rp.run_length("s"))
    dt = df.with_columns(pl.datetime(2026, 1, 1).alias("ts"))
    with pytest.raises(ValueError, match="both be temporal or both be numbers"):
        rp.stream.with_windows(dt, d=rp.clock_drift("ts", "x", half_life=1.0))
