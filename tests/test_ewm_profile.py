"""``rp.ewm_profile``: an exponentially weighted mean, variance or standard
deviation of a column by each row's phase of a period, read at the row's
own phase.

Held to Polars on one chunk: the phase as ``t − t.dt.truncate(period)`` on
the wall clock (a zoned clock read through ``dt.replace_time_zone(None)``),
cross-checked row by row against the standard library's ``zoneinfo``; and
each row's value as a ``group_by`` over the phase of its group's earlier
rows, with the decayed weights written out. And what a window run owes:
chunk invariance, a save and load at every stamp, zero weights in no clock,
an unseen phase withheld by its weight, the state bound and its refusal.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
import pytest

from rheon import polars as rp

TIER = "essential"

#: Welford's form in input order, decayed at each step, against Polars'
#: two-pass sums of the weights written out, relative.
TOL = 1e-10

DAY = 86_400 * 10**9


def close(a: pl.Series, b: pl.Series, tol: float = TOL) -> None:
    an, bn = a.is_null().to_numpy(), b.is_null().to_numpy()
    assert np.array_equal(an, bn), (int(an.sum()), int(bn.sum()), np.flatnonzero(an != bn)[:5])
    x, y = a.drop_nulls().to_numpy(), b.drop_nulls().to_numpy()
    err = np.max(np.abs(x - y) / np.maximum(np.abs(y), 1.0)) if len(x) else 0.0
    assert err < tol, err


def stream(days: int, seed: int, *, zone: str | None = None, groups: int = 3) -> pl.DataFrame:
    """Rows every 20 to 100 minutes over ``days`` days, some at a repeated
    stamp, each in one of ``groups`` groups, with a daily shape, nulls and
    zero weights."""
    rng = np.random.default_rng(seed)
    t0 = datetime(2026, 3, 25, tzinfo=UTC)
    rows = []
    t = t0
    end = t0 + timedelta(days=days)
    while t < end:
        for _ in range(1 + int(rng.uniform() < 0.15)):
            hour = t.hour + t.minute / 60
            x = (
                None
                if rng.uniform() < 0.05
                else float(np.sin(hour / 24 * 2 * np.pi) * 3 + rng.normal())
            )
            w = 0.0 if rng.uniform() < 0.08 else float(rng.uniform(0.3, 2.0))
            rows.append((t, f"g{rng.integers(groups)}", x, w))
        t += timedelta(minutes=int(rng.integers(20, 100)))
    df = pl.DataFrame(rows, schema=["t", "g", "x", "w"], orient="row")
    df = df.with_columns(pl.col("t").dt.cast_time_unit("us"))
    if zone is not None:
        df = df.with_columns(pl.col("t").dt.convert_time_zone(zone))
    else:
        df = df.with_columns(pl.col("t").dt.replace_time_zone(None))
    return df


def phase_ns(t: pl.Expr, period: str, offset_ns: int = 0) -> pl.Expr:
    """The wall-clock phase in nanoseconds, by Polars: ``t − truncate(t)``
    on the zone dropped, less the offset, mod the period."""
    wall = t.dt.replace_time_zone(None)
    p = pl.duration(nanoseconds=_ns(period)).dt.total_nanoseconds()
    raw = (wall - wall.dt.truncate(period)).dt.total_nanoseconds()
    return (raw - offset_ns) % p


#: The periods the tests read, in nanoseconds.
PERIOD_NS = {"1d": DAY, "1w": 7 * DAY}


def _ns(period: str) -> int:
    return PERIOD_NS[period]


def oracle(
    df: pl.DataFrame,
    *,
    bucket: pl.Expr,
    age: pl.Expr,
    half_life: float,
    closed: str = "right",
    min_samples: int = 1,
    bias: bool = False,
    weight: str | None = "w",
    window: float | None = None,
) -> pl.DataFrame:
    """Per row: the decayed weighted mean, variance and standard deviation
    of its group's earlier rows in its bucket -- a join on the group and
    the bucket, then a ``group_by`` of the row. ``age`` reads the clock of
    the row read (``t``) and of the row it weighs (``t_right``)."""
    d = df.with_row_index("i").with_columns(b=bucket)
    if weight is None:
        d = d.with_columns(w=pl.lit(1.0))
    members = d.filter(pl.col("x").is_not_null() & (pl.col("w") > 0)).select(
        "g", "b", "t", "x", "w"
    )
    j = d.select("i", "g", "b", "t").join(members, on=["g", "b"], how="inner")
    near = closed in ("right", "both")
    later = pl.col("t_right") <= pl.col("t") if near else pl.col("t_right") < pl.col("t")
    j = j.filter(later)
    if window is not None:
        # In integer nanoseconds, as the operator decides the edge: a float
        # age can land a last bit past a whole window.
        far = closed in ("left", "both")
        ns = (pl.col("t") - pl.col("t_right")).dt.total_nanoseconds()
        edge = round(window * 1e9)
        j = j.filter(ns <= edge if far else ns < edge)
    j = j.with_columns(
        wd=pl.col("w") * pl.lit(2.0).pow(-age / half_life)
        if np.isfinite(half_life)
        else pl.col("w")
    )
    mean = (pl.col("wd") * pl.col("x")).sum() / pl.col("wd").sum()
    agg = j.group_by("i").agg(
        n=pl.len(),
        v1=pl.col("wd").sum(),
        v2=(pl.col("wd") ** 2).sum(),
        mean=mean,
        m2=(pl.col("wd") * (pl.col("x") - mean) ** 2).sum(),
    )
    out = d.select("i").join(agg, on="i", how="left", maintain_order="left")
    seen = pl.col("n") >= min_samples
    var = (
        pl.col("m2") / pl.col("v1")
        if bias
        else pl.when(pl.col("v1") ** 2 - pl.col("v2") > 0).then(
            pl.col("m2") / pl.col("v1") * pl.col("v1") ** 2 / (pl.col("v1") ** 2 - pl.col("v2"))
        )
    )
    return out.select(
        mean=pl.when(seen).then("mean"),
        var=pl.when(seen).then(var),
        std=pl.when(seen).then(var.sqrt()),
    )


def run(df: pl.DataFrame, **exprs: Any) -> pl.DataFrame:
    kw: dict[str, Any] = {"clock": "t", "gap_cap": "30d", "group": "g"}
    kw.update(exprs.pop("_kw", {}))
    return rp.stream.with_windows(df.lazy(), **exprs, **kw).collect()


TIME_AGE = (pl.col("t") - pl.col("t_right")).dt.total_nanoseconds() / 1e9


@pytest.mark.parametrize("closed", ["right", "left"])
def test_the_profile_is_a_group_by_over_the_phase(closed: str):
    """On a naive Datetime: every statistic, weighted and decayed, is
    Polars' ``group_by`` over the hour of the day of each group's rows."""
    df = stream(12, seed=1)
    h = 2 * 86_400.0
    kw = {"period": "1d", "buckets": 24, "half_life": "2d", "weight": "w", "closed": closed}
    out = run(
        df,
        m=rp.ewm_profile("x", **kw),
        v=rp.ewm_profile("x", stat="var", **kw),
        s=rp.ewm_profile("x", stat="std", **kw),
        b=rp.ewm_profile("x", stat="var", bias=True, **kw),
    )
    bucket = phase_ns(pl.col("t"), "1d") * 24 // DAY
    want = oracle(df, bucket=bucket, age=TIME_AGE, half_life=h, closed=closed)
    close(out["m"], want["mean"])
    close(out["v"], want["var"])
    close(out["s"], want["std"])
    wb = oracle(df, bucket=bucket, age=TIME_AGE, half_life=h, closed=closed, bias=True)
    close(out["b"], wb["var"])


@pytest.mark.parametrize("zone", ["Europe/Amsterdam", "America/New_York", "Asia/Kolkata"])
def test_a_zoned_clock_reads_the_wall_clock_across_both_daylight_saving_changes(zone: str):
    """Four stretches of four days in 2026 cross both changes in Amsterdam
    (29 March, 25 October) and in New York (8 March, 1 November), and
    Kolkata's half-hour offset has none. Each row's bucket is its wall
    clock's hour, as Polars reads it and as ``zoneinfo`` does row by row."""
    rng = np.random.default_rng(2)
    starts = [datetime(2026, d, m, tzinfo=UTC) for d, m in [(3, 6), (3, 27), (10, 23), (10, 30)]]
    rows = []
    for s in starts:
        for k in range(4 * 24 * 2):
            t = s + timedelta(minutes=30 * k + int(rng.integers(0, 29)))
            rows.append((t, "a", float(rng.normal() + (t.hour % 7)), 1.0))
    df = pl.DataFrame(rows, schema=["t", "g", "x", "w"], orient="row").with_columns(
        pl.col("t").dt.cast_time_unit("us").dt.convert_time_zone(zone)
    )
    # zoneinfo, row by row: the hour of the wall clock.
    tz = ZoneInfo(zone)
    by_zoneinfo = [
        (t.astimezone(tz).hour * 60 + t.astimezone(tz).minute) * 24 // 1440
        for t in df["t"].dt.convert_time_zone("UTC").to_list()
    ]
    polars_bucket = df.select(b=phase_ns(pl.col("t"), "1d") * 24 // DAY)["b"].to_list()
    assert polars_bucket == by_zoneinfo
    # The wall clock and the elapsed time part on a change's day.
    elapsed = df.select(
        b=(pl.col("t") - pl.col("t").dt.truncate("1d")).dt.total_nanoseconds() * 24 // DAY
    )["b"].to_list()
    assert (polars_bucket != elapsed) == (zone != "Asia/Kolkata")
    # One stretch across the months between the two changes.
    out = run(
        df,
        m=rp.ewm_profile("x", period="1d", buckets=24, half_life=float("inf")),
        _kw={"gap_cap": "400d"},
    )
    want = oracle(
        df, bucket=pl.Series(by_zoneinfo), age=TIME_AGE, half_life=float("inf"), weight=None
    )
    close(out["m"], want["mean"])


def test_a_week_starts_on_monday_and_an_offset_moves_it():
    """A week is Monday to Sunday, as Polars truncates one; ``offset="6d"``
    starts it on Sunday, and ``offset="9h30m"`` a day at 09:30. On a
    ``Date`` clock too."""
    df = stream(30, seed=3, groups=1)
    out = run(
        df,
        wk=rp.ewm_profile("x", period="1w", buckets=7, half_life=float("inf")),
        s=rp.ewm_profile("x", period="1w", buckets=7, half_life=float("inf"), offset="6d"),
        o=rp.ewm_profile("x", period="1d", buckets=24, half_life=float("inf"), offset="9h30m"),
    )
    week = phase_ns(pl.col("t"), "1w")
    for col, bucket in [
        ("wk", week * 7 // (7 * DAY)),
        ("s", phase_ns(pl.col("t"), "1w", 6 * DAY) * 7 // (7 * DAY)),
        ("o", phase_ns(pl.col("t"), "1d", 9 * 3600 * 10**9 + 1800 * 10**9) * 24 // DAY),
    ]:
        want = oracle(df, bucket=bucket, age=TIME_AGE, half_life=float("inf"), weight=None)
        close(out[col], want["mean"])
    # Monday is day 0 of Polars' week.
    assert df.select((week // DAY == pl.col("t").dt.weekday() - 1).all()).item()
    dates = (
        df.with_columns(t=pl.col("t").dt.date())
        .group_by("t", "g", maintain_order=True)
        .agg(pl.col("x").drop_nulls().first(), pl.col("w").first())
    )
    d_out = run(dates, d=rp.ewm_profile("x", period="1w", buckets=7, half_life="10d"))
    day_age = (pl.col("t") - pl.col("t_right")).dt.total_days() * 86_400.0
    want = oracle(
        dates,
        bucket=pl.col("t").dt.weekday() - 1,
        age=day_age,
        half_life=10 * 86_400.0,
        weight=None,
    )
    close(d_out["d"], want["mean"])


def test_a_numeric_an_integer_and_a_row_clock():
    """A float clock's phase is its remainder; an integer clock's in
    integers; without a clock, the group's rows of weight above 0."""
    rng = np.random.default_rng(4)
    n = 600
    t = np.cumsum(rng.integers(1, 5, n))
    df = pl.DataFrame(
        {
            "t": t,
            "g": rng.choice(["a", "b"], n),
            "x": rng.normal(size=n) + (t % 10),
            "w": np.where(rng.uniform(size=n) < 0.1, 0.0, rng.uniform(0.5, 2.0, n)),
        }
    )
    kw = {"period": 10, "buckets": 5, "half_life": 40.0, "weight": "w", "stat": "var"}
    num_age = (pl.col("t") - pl.col("t_right")).cast(pl.Float64)
    for frame in (df, df.with_columns(pl.col("t").cast(pl.Float64))):
        out = run(frame, v=rp.ewm_profile("x", **kw), _kw={"gap_cap": 1e9})
        want = oracle(frame, bucket=(pl.col("t") % 10) * 5 // 10, age=num_age, half_life=40.0)
        close(out["v"], want["var"])
    # The row clock: each group's rows of weight above 0, from 0.
    rows = df.drop("t").with_columns(
        t=((pl.col("w") > 0).cast(pl.Int64).cum_sum() - 1).clip(0).over("g")
    )
    out = rp.stream.with_windows(rows.drop("t"), v=rp.ewm_profile("x", **kw), group="g")
    want = oracle(rows, bucket=(pl.col("t") % 10) * 5 // 10, age=num_age, half_life=40.0)
    close(out["v"], want["var"])


def test_an_unseen_phase_is_withheld_and_min_samples_counts_its_rows():
    """A phase no row has reached is null; ``min_samples`` withholds one
    holding fewer rows with a value and a weight, as Polars'
    ``rolling_*_by`` counts them, however far back they are."""
    df = pl.DataFrame(
        {
            "t": [datetime(2026, 1, d, h) for d, h in [(5, 8), (5, 9), (6, 8), (20, 8), (20, 10)]],
            "g": ["a"] * 5,
            "x": [1.0, 2.0, 3.0, 5.0, 7.0],
            "w": [1.0, 1.0, 0.0, 1.0, 1.0],
        }
    )
    kw = {"period": "1d", "buckets": 24, "half_life": "1d", "weight": "w"}
    out = run(
        df,
        m=rp.ewm_profile("x", closed="left", **kw),
        two=rp.ewm_profile("x", closed="left", min_samples=2, **kw),
        r2=rp.ewm_profile("x", min_samples=2, **kw),
    )
    assert out["m"].to_list() == [None, None, 1.0, 1.0, None]
    # The 08:00 phase holds one row before the 20th (the 6th's weighs 0).
    assert out["two"].to_list() == [None] * 5
    # Under "right" the 20th's own row is the second.
    assert out["r2"][3] is not None and out["r2"][0] is None


def test_a_zero_weight_row_reads_its_phase_and_adds_nothing():
    df = stream(6, seed=5)
    kw = {"period": "1d", "buckets": 12, "half_life": "1d", "weight": "w"}
    exprs = {"m": rp.ewm_profile("x", **kw), "v": rp.ewm_profile("x", stat="std", **kw)}
    full = run(df, **exprs)
    kept = run(df.filter(pl.col("w") > 0), **exprs)
    assert full.filter(pl.col("w") > 0).equals(kept)


def exprs() -> dict[str, pl.Expr]:
    kw = {"period": "1d", "buckets": 24, "half_life": "3d", "weight": "w"}
    return {
        "m": rp.ewm_profile("x", **kw),
        "l": rp.ewm_profile("x", stat="std", closed="left", **kw),
        "e": rp.ewm_mean("x", half_life="1h"),
    }


@pytest.mark.parametrize("rows", [1, 2, 7, 64, 10_000])
def test_chunking_changes_nothing(rows: int):
    df = stream(8, seed=6, zone="Europe/Amsterdam")
    kw = {"clock": "t", "gap_cap": "30d", "group": "g"}
    one = rp.stream.with_windows(df.lazy(), **exprs(), **kw, chunk_size=100_000).collect()
    out = rp.stream.with_windows(df.lazy(), **exprs(), **kw, chunk_size=rows).collect()
    assert out.equals(one)


def test_a_save_and_load_at_every_stamp_is_one_run(tmp_path: Path):
    df = stream(3, seed=7, zone="America/New_York")
    kw = {"clock": "t", "gap_cap": "30d", "group": "g"}
    one = rp.stream.with_windows(df, **exprs(), **kw)
    starts = [i for i in range(1, df.height) if df["t"][i] != df["t"][i - 1]]
    for at in starts[::3]:
        state = tmp_path / f"p{at}.state"
        first = rp.stream.with_windows(df[:at], **exprs(), **kw, save_state=state)
        second = rp.stream.with_windows(df[at:], **exprs(), **kw, load_state=state)
        assert pl.concat([first, second]).equals(one), at


def test_the_state_is_bounded_by_the_buckets(tmp_path: Path):
    """A run of 5 days and one of 200 save states of the same size, give or
    take the last rows' clock: 56 bytes a bucket a group."""
    sizes = []
    for days in (5, 200):
        df = stream(days, seed=8)
        state = tmp_path / f"s{days}.state"
        rp.stream.with_windows(
            df,
            m=rp.ewm_profile("x", period="1d", buckets=24, half_life="3d", closed="left"),
            clock="t",
            gap_cap="30d",
            group="g",
            save_state=state,
        )
        sizes.append(state.stat().st_size)
    assert abs(sizes[0] - sizes[1]) < 400, sizes


def test_a_profile_past_its_cap_is_refused_with_its_bytes():
    df = stream(1, seed=9)
    with pytest.raises(Exception, match=r"holds 1344 bytes a group, 24 buckets of 56 bytes"):
        run(
            df, m=rp.ewm_profile("x", period="1d", buckets=24, half_life="1d", max_state_bytes=1000)
        )
    run(df, m=rp.ewm_profile("x", period="1d", buckets=24, half_life="1d", max_state_bytes=1344))


def test_the_refusals():
    df = stream(1, seed=10)
    with pytest.raises(TypeError, match="period is required"):
        rp.ewm_profile("x", period=None, buckets=24, half_life="1d")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="buckets must be a whole number from 1"):
        rp.ewm_profile("x", period="1d", buckets=0, half_life="1d")
    with pytest.raises(TypeError, match="bias is given"):
        rp.ewm_profile("x", period="1d", buckets=24, half_life="1d", bias=True)
    with pytest.raises(ValueError, match="stat must be"):
        rp.ewm_profile("x", period="1d", buckets=24, half_life="1d", stat="median")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="min_samples must be an integer of at least 1"):
        rp.ewm_profile("x", period="1d", buckets=24, half_life="1d", min_samples=0)
    with pytest.raises(Exception, match="month has no fixed length"):
        run(df, m=rp.ewm_profile("x", period="1mo", buckets=12, half_life="1d"))
    with pytest.raises(Exception, match="needs a temporal clock"):
        rp.stream.with_windows(
            df.drop("t").with_columns(t=pl.int_range(pl.len())),
            m=rp.ewm_profile("x", period="1d", buckets=24, half_life="1h"),
            clock="t",
            gap_cap="10h",
        )
    with pytest.raises(Exception, match="ewm_profile, which reads the rows before each row"):
        ahead = rp.ewm_mean("x", half_life=1.0, window_size=5.0, reverse=True)
        rp.spec.ewridge(
            "m",
            targets=[
                (ahead - rp.ewm_profile("x", period=24, buckets=24, half_life=5.0)).alias("c")
            ],
            features=["x"],
            half_life=10.0,
            embargo=5.0,
        )


# ---- a window, and a quantile ------------------------------------------------


@pytest.mark.parametrize("closed", ["right", "left", "both", "none"])
def test_a_window_drops_the_rows_further_back(closed: str):
    """Under ``window_size`` each phase reads its rows inside the window
    alone, the far end kept under ``"left"`` and ``"both"``: Polars'
    ``group_by`` over the phase of the rows inside it, and ``min_samples``
    counted inside it."""
    df = stream(16, seed=21)
    kw = {
        "period": "1d",
        "buckets": 24,
        "half_life": "2d",
        "window_size": "3d",
        "weight": "w",
        "closed": closed,
    }
    out = run(
        df,
        m=rp.ewm_profile("x", **kw),
        v=rp.ewm_profile("x", stat="var", **kw),
        m3=rp.ewm_profile("x", min_samples=3, **kw),
    )
    bucket = phase_ns(pl.col("t"), "1d") * 24 // DAY
    args: dict[str, Any] = {
        "bucket": bucket,
        "age": TIME_AGE,
        "half_life": 2 * 86_400.0,
        "closed": closed,
        "window": 3 * 86_400.0,
    }
    want = oracle(df, **args)
    close(out["m"], want["mean"])
    close(out["v"], want["var"])
    close(out["m3"], oracle(df, min_samples=3, **args)["mean"])


def test_a_window_on_a_row_exactly_its_span_back():
    """A row exactly ``window_size`` back is in the window under ``"left"``
    and ``"both"`` and out under ``"right"`` and ``"none"``, decided in
    integers on an integer clock."""
    df = pl.DataFrame({"t": [0, 24, 48, 72], "g": ["a"] * 4, "x": [1.0, 2.0, 4.0, 8.0]})
    kw = {"period": 24, "buckets": 1, "half_life": float("inf"), "window_size": 48}
    out = run(
        df,
        r=rp.ewm_profile("x", closed="right", **kw),
        b=rp.ewm_profile("x", closed="both", **kw),
        _kw={"gap_cap": 1000},
    )
    assert out["r"].to_list() == [1.0, 1.5, 3.0, 6.0]
    assert np.allclose(out["b"].to_list(), [1.0, 1.5, 7.0 / 3.0, 14.0 / 3.0], rtol=1e-15)


def weighted_quantile(
    df: pl.DataFrame, q: float, bucket: pl.Expr, h: float, closed: str, window: float | None
) -> list[float | None]:
    """NumPy's weighted ``inverted_cdf`` of each row's phase's earlier
    rows, the decayed weights written out."""
    d = df.with_row_index("i").with_columns(b=bucket)
    m = d.filter(pl.col("x").is_not_null() & (pl.col("w") > 0)).select("g", "b", "t", "x", "w")
    j = d.select("i", "g", "b", "t").join(m, on=["g", "b"], how="inner")
    near = closed in ("right", "both")
    j = j.filter(pl.col("t_right") <= pl.col("t") if near else pl.col("t_right") < pl.col("t"))
    if window is not None:
        far = closed in ("left", "both")
        ns = (pl.col("t") - pl.col("t_right")).dt.total_nanoseconds()
        edge = round(window * 1e9)
        j = j.filter(ns <= edge if far else ns < edge)
    j = j.with_columns(wd=pl.col("w") * pl.lit(2.0).pow(-TIME_AGE / h))
    out: list[float | None] = [None] * df.height
    for (i,), part in j.group_by("i"):
        x, w = part["x"].to_numpy(), part["wd"].to_numpy()
        out[int(i)] = float(np.quantile(x, q, method="inverted_cdf", weights=w))
    return out


#: ``ewm_quantile``'s sketch: within ``tanh(1/128)`` of the exact quantile,
#: relatively.
ALPHA = float(np.tanh(1 / 128))


@pytest.mark.parametrize(
    ("closed", "window"), [("right", None), ("left", None), ("right", 3.0), ("both", 3.0)]
)
def test_the_quantile_is_numpys_weighted_inverted_cdf_within_the_sketch(
    closed: str, window: float | None
):
    df = stream(14, seed=22).with_columns(x=pl.col("x") + 20.0)
    kw: dict[str, Any] = {
        "period": "1d",
        "buckets": 12,
        "half_life": "2d",
        "weight": "w",
        "closed": closed,
    }
    if window is not None:
        kw["window_size"] = f"{int(window)}d"
    for q in (0.1, 0.5, 0.9):
        out = run(df, q=rp.ewm_profile("x", stat="quantile", quantile=q, **kw))
        want = weighted_quantile(
            df,
            q,
            phase_ns(pl.col("t"), "1d") * 12 // DAY,
            2 * 86_400.0,
            closed,
            None if window is None else window * 86_400.0,
        )
        got = out["q"].to_list()
        for i, (a, b) in enumerate(zip(got, want, strict=True)):
            assert (a is None) == (b is None), (q, i, a, b)
            if a is not None and b is not None:
                assert abs(a - b) <= ALPHA * abs(b) * (1 + 1e-9), (q, i, a, b)


def windowed() -> dict[str, pl.Expr]:
    kw = {"period": "1d", "buckets": 6, "half_life": "2d", "weight": "w", "window_size": "2d"}
    return {
        "m": rp.ewm_profile("x", **kw),
        "q": rp.ewm_profile("x", stat="quantile", quantile=0.7, closed="left", **kw),
        "s": rp.ewm_profile(
            "x", stat="quantile", quantile=0.3, period="1d", buckets=6, half_life="1d"
        ),
    }


@pytest.mark.parametrize("rows", [1, 3, 50, 10_000])
def test_a_window_and_a_quantile_change_nothing_by_chunking(rows: int):
    df = stream(9, seed=23, zone="Europe/Amsterdam")
    kw = {"clock": "t", "gap_cap": "30d", "group": "g"}
    one = rp.stream.with_windows(df.lazy(), **windowed(), **kw, chunk_size=100_000).collect()
    out = rp.stream.with_windows(df.lazy(), **windowed(), **kw, chunk_size=rows).collect()
    assert out.equals(one)


def test_a_window_and_a_quantile_save_and_load_at_every_stamp(tmp_path: Path):
    df = stream(4, seed=24)
    kw = {"clock": "t", "gap_cap": "30d", "group": "g"}
    one = rp.stream.with_windows(df, **windowed(), **kw)
    starts = [i for i in range(1, df.height) if df["t"][i] != df["t"][i - 1]]
    for at in starts[::4]:
        state = tmp_path / f"w{at}.state"
        first = rp.stream.with_windows(df[:at], **windowed(), **kw, save_state=state)
        second = rp.stream.with_windows(df[at:], **windowed(), **kw, load_state=state)
        assert pl.concat([first, second]).equals(one), at


def test_a_window_bounds_the_state(tmp_path: Path):
    """Under a window the state holds the rows inside it: a run of 6 days and
    one of 120 save states of about one size."""
    sizes = []
    for days in (6, 120):
        df = stream(days, seed=25)
        state = tmp_path / f"w{days}.state"
        rp.stream.with_windows(
            df, **windowed(), clock="t", gap_cap="30d", group="g", save_state=state
        )
        sizes.append(state.stat().st_size)
    assert sizes[1] < 2 * sizes[0], sizes


def test_a_window_past_half_its_cap_warns_once_and_past_it_is_refused():
    """One bucket of 56 bytes, and 80 bytes a row in the window: a cap of
    400 holds four rows; the second takes it past half, which warns once a
    run, and a fifth row inside the window is refused by row. Rows that
    leave the window free their bytes."""
    from rheon import RheonWarning

    def rows(n: int, step: float = 1.0) -> pl.DataFrame:
        return pl.DataFrame({"t": [step * i for i in range(n)], "g": ["a"] * n, "x": [1.0] * n})

    kw: dict[str, Any] = {
        "period": 100.0,
        "buckets": 1,
        "half_life": 50.0,
        "window_size": 10.0,
        "max_state_bytes": 400,
    }
    with pytest.warns(RheonWarning, match="past half of its max_state_bytes of 400") as caught:
        run(rows(4), m=rp.ewm_profile("x", **kw), _kw={"gap_cap": 1e9})
    assert len([w for w in caught if issubclass(w.category, RheonWarning)]) == 1
    with pytest.raises(
        Exception, match=r"would hold 456 bytes for one group at row 4 .*max_state_bytes of 400"
    ):
        run(rows(5), m=rp.ewm_profile("x", **kw), _kw={"gap_cap": 1e9})
    run(rows(20, step=20.0), m=rp.ewm_profile("x", **kw), _kw={"gap_cap": 1e9})
