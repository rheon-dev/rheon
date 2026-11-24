"""A zoned ``Datetime`` clock keeps its zone through every path that takes a
temporal clock.

Polars' Rust crate built without its ``timezones`` feature casts a column to
``Datetime(unit, zone)`` and returns it naive, without a word; a column of
nulls alone keeps the zone, so the two no longer subtract. So each public
path that reads a temporal clock -- ``rp.stream.merge`` (with ``carry``),
``rp.stream.refresh_time`` (with ``age``, and a pair grid's held points),
a bank's clock (under an ``embargo``), ``with_windows``, and a pipeline
checkpoint -- runs here on a clock in Europe/Amsterdam across the spring
change of 2024-03-31, and is held to: the zone on every clock it returns,
and a run resumed from its state equal to the one run, to the bit.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from polars.testing import assert_frame_equal

from rheon import polars as rp

TIER = "essential"

ZONE = "Europe/Amsterdam"
ZONED = pl.Datetime("us", ZONE)


def clock(n: int, start_minute: int = 0, step: str = "7m") -> pl.Series:
    """``n`` instants ``step`` apart from 00:30 local, across the change at
    02:00 (clocks forward to 03:00)."""
    start = pl.datetime(2024, 3, 31, 0, 30 + start_minute, time_zone=ZONE)
    out = pl.select(
        pl.datetime_ranges(start, start + pl.duration(days=1), step, time_zone=ZONE)
        .explode()
        .head(n)
    ).to_series()
    return out.cast(ZONED).alias("t")


def test_merge_with_carry_keeps_the_zone_and_resumes(tmp_path):
    quotes = pl.DataFrame({"t": clock(30), "bid": [float(i) if i % 3 else None for i in range(30)]})
    trades = pl.DataFrame({"t": clock(20, 3, "11m"), "px": [float(i) for i in range(20)]})
    frames = {"quotes": quotes, "trades": trades}
    whole = rp.stream.merge(frames, clock="t", carry=["bid", "px"])
    assert whole.schema["t"] == ZONED
    assert whole.schema["bid_age"] == pl.Duration("us")
    want = pl.concat([f.with_columns(source=pl.lit(k)) for k, f in frames.items()], how="diagonal")
    assert_frame_equal(whole.select("t"), want.sort("t", maintain_order=True).select("t"))
    # Cut inside a tie, so the state holds zoned rows; then resumed.
    at = trades["t"][8]
    first = {k: f.filter(pl.col("t") <= at) for k, f in frames.items()}
    rest = {k: f.filter(pl.col("t") > at) for k, f in frames.items()}
    state = tmp_path / "merge.state"
    out1 = rp.stream.merge(first, clock="t", carry=["bid", "px"], save_state=state)
    out2 = rp.stream.merge(rest, clock="t", carry=["bid", "px"], load_state=state)
    assert_frame_equal(pl.concat([out1, out2]), whole, check_exact=True)


def ticks(n: int = 240, seed: int = 5) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    return pl.DataFrame(
        {
            "series": rng.choice(["a", "b", "c"], n).tolist(),
            "t": clock(n, step="1m"),
            "v": 100.0 + np.cumsum(0.1 * rng.standard_normal(n)),
            "i": np.arange(n),
        }
    )


def test_refresh_time_with_ages_keeps_the_zone_and_a_held_pair_point_resumes(tmp_path):
    df = ticks()
    kw = {"series": "series", "names": ["a", "b", "c"], "clock": "t", "value": "v"}
    kw |= {"pairs": True, "age": True, "keep": ["i"]}
    whole = rp.stream.refresh_time(df, **kw)
    assert whole.schema["time_refresh"] == ZONED
    assert whole.schema["a_age"] == pl.Duration("us")
    state = tmp_path / "grid.state"
    i = whole["i"]
    # A slice ending among one tick's pair points: the state holds the rest,
    # with the tick's zoned clock.
    n = next(p for p in range(1, whole.height) if i[p] == i[p - 1] and p > whole.height // 2)
    head = rp.stream.refresh_time(df.lazy(), save_state=state, **kw).head(n).collect()
    rest = rp.stream.refresh_time(df.slice(head["i"][-1] + 1), load_state=state, **kw)
    assert rest.schema["time_refresh"] == ZONED
    assert_frame_equal(pl.concat([head, rest]), whole, check_exact=True)


def frame(n: int = 200, seed: int = 6) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    return pl.DataFrame({"t": clock(n), "x": x, "y": 2 * x + 0.1 * rng.standard_normal(n)})


def test_a_banks_clock_keeps_the_zone_and_resumes_under_an_embargo():
    df = frame()
    spec = rp.spec.ewridge(
        "m", targets=["y"], features=["x"], clock="t", gap_cap="1h", half_life="1h", embargo="20m"
    )
    full = rp.ModelBank([spec])
    whole = full.fit_predict(df)
    assert whole.schema["t"] == ZONED
    # The clocks a bank reports back are in the clock's own dtype, zone too.
    assert full.groups().schema["clock_last"] == ZONED
    assert full.groups()["clock_last"][0] == df["t"][-1]
    summary = full.summary()
    for name in ("clock_min", "clock_max", "clock_last"):
        if name in summary.columns:
            assert summary.schema[name] == ZONED, name
    bank = rp.ModelBank([spec])
    out1 = bank.fit_predict(df.head(97))
    out2 = rp.ModelBank.load_bytes(bank.save_bytes()).fit_predict(df.slice(97))
    got = pl.concat([out1, out2]).unnest("m").drop("coef", "support_coef")
    assert got.equals(whole.unnest("m").drop("coef", "support_coef"))


def test_with_windows_keeps_the_zone_and_resumes(tmp_path):
    df = frame()

    def run(lf: pl.DataFrame, **state: object) -> pl.DataFrame:
        return rp.stream.with_windows(
            lf,
            clock="t",
            gap_cap="1h",
            back=rp.ewm_mean("x", half_life="20m", window_size="1h"),
            ahead=rp.ewm_mean("y", half_life="20m", window_size="30m", reverse=True),
            **state,  # type: ignore[arg-type]
        )

    whole = run(df)
    assert whole.schema["t"] == ZONED
    state = tmp_path / "windows.state"
    out1 = run(df.head(120), save_state=state)
    out2 = run(df.slice(120), load_state=state)
    assert_frame_equal(pl.concat([out1, out2]), whole, check_exact=True)


def test_a_zoned_pipeline_resumed_from_a_checkpoint_is_one_run(tmp_path):
    grid_kw = {"series": "series", "names": ["a", "b"], "clock": "t", "value": "v"}
    df = ticks().filter(pl.col("series") != "c")

    def windows(lf: pl.DataFrame, **state: object) -> pl.DataFrame:
        return rp.stream.with_windows(
            lf,
            clock="time_refresh",
            gap_cap="2h",
            level=rp.ewm_mean("a_value", half_life="10m"),
            **state,  # type: ignore[arg-type]
        )

    spec = rp.spec.ewridge(
        "m",
        targets=["b_value"],
        features=["a_value", "level"],
        clock="time_refresh",
        gap_cap="2h",
        half_life="1h",
    )
    want = rp.ModelBank([spec]).fit_predict(windows(rp.stream.refresh_time(df, **grid_kw)))
    assert want.schema["time_refresh"] == ZONED
    first, second = df.head(110), df.slice(110)
    grid_state, win_state = tmp_path / "grid.state", tmp_path / "windows.state"
    bank = rp.ModelBank([spec])
    out1 = bank.fit_predict(
        windows(
            rp.stream.refresh_time(first, **grid_kw, save_state=grid_state), save_state=win_state
        )
    )
    path = tmp_path / "run.checkpoint"
    rp.stream.save_checkpoint(path, {"grid": grid_state, "windows": win_state, "bank": bank})
    stages = rp.stream.load_checkpoint(path)
    grid2 = rp.stream.refresh_time(second, **grid_kw, load_state=stages["grid"])
    out2 = rp.ModelBank.load_bytes(stages["bank"]).fit_predict(
        windows(grid2, load_state=stages["windows"])
    )
    got = pl.concat([out1, out2]).unnest("m").drop("coef", "support_coef")
    assert got.equals(want.unnest("m").drop("coef", "support_coef"))
