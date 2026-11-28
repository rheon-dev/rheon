"""A value carried forward and its age on the clock: ``carry`` in
``rp.stream.merge`` and ``age`` in ``rp.stream.refresh_time``.

The oracle is Polars itself, on single chunks. A carried column is the
backward ``join_asof`` of each merged row onto the rows of its group where
the column has a value -- matched on the merged row's position, so a tie on
the clock goes as the merge orders it, and on the clock itself where no two
rows share one -- and equally ``forward_fill`` over the group. Its age is
Polars' own subtraction of the two clocks, ``pl.col(clock) - clock_then``,
in that expression's dtype. A grid's age is the backward ``join_asof`` of
each point's completing tick onto the series' ticks with a value. Each is
held equal to the bit, with Hypothesis drawing the streams, their number,
the clock's dtype, ties, nulls and groups, and the chunking, which changes
nothing (hard rule 3). Then the properties that make each a streaming
stage: a value carried within its day under ``restart_after_step_back``, a
saved state and a slice that resume to the one run's rows, and the
refusals.
"""

from __future__ import annotations

import os
import tempfile
import warnings
from datetime import datetime

import polars as pl
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from polars.testing import assert_frame_equal

from rheon import ColumnNotFoundError, WrongTypeError
from rheon import polars as rp
from tiers import examples

TIER = "essential"

PROPERTY = settings(
    max_examples=examples(60, essential=10),
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)

CLOCK_DTYPES = [
    pl.Int64(),
    pl.Int32(),
    pl.UInt64(),
    pl.Float64(),
    pl.Float32(),
    pl.Datetime("ns"),
    pl.Datetime("us", "Europe/Amsterdam"),
    pl.Date(),
    pl.Duration("ms"),
]


def clock(ticks: list[int], dtype: pl.DataType) -> pl.Series:
    """A clock of ``dtype`` at the integer ``ticks``."""
    raw = pl.Series("t", ticks, dtype=pl.Int64)
    if dtype.is_float():
        return (raw / 4).cast(dtype)
    if dtype.is_integer():
        return raw.cast(dtype)
    if isinstance(dtype, pl.Date):
        return (raw + 19_800).cast(pl.Int32).cast(pl.Date)
    if isinstance(dtype, pl.Duration):
        return raw.cast(dtype)
    out = (raw * 60_000_000_000 + 1_711_843_200_000_000_000).cast(pl.Datetime("ns"))
    out = out.cast(pl.Datetime(dtype.time_unit))
    if dtype.time_zone is not None:
        out = out.dt.replace_time_zone("UTC").dt.convert_time_zone(dtype.time_zone)
    return out


def sorted_merge(frames: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Polars' stable sort of the streams' rows, concatenated in order: the
    merge without ``carry`` (held to it in ``test_stream_merge.py``)."""
    tagged = [f.with_columns(pl.lit(k).alias("source")) for k, f in frames.items()]
    out = pl.concat(tagged, how="diagonal").sort("t", maintain_order=True)
    columns: list[str] = []
    for f in frames.values():
        columns += [c for c in f.columns if c not in columns]
    return out.select(*columns, "source")


def asof(left: pl.DataFrame, right: pl.DataFrame, on: str, by: list[str]) -> pl.DataFrame:
    """Polars' backward ``join_asof``. Both sides are sorted on ``on``
    across their groups, so within each: the warning that Polars cannot
    check it under ``by`` says nothing here."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Sortedness of columns cannot be checked")
        return left.join_asof(right, on=on, by=by or None, strategy="backward")


def by_asof(merged: pl.DataFrame, carry: list[str], by: list[str], on: str) -> pl.DataFrame:
    """Each carried column as the backward ``join_asof`` of every merged row
    onto the rows of its group with a value, matched on ``on``; its age the
    row's clock less the clock matched."""
    out = merged.with_row_index("_i")
    for c in carry:
        right = out.filter(pl.col(c).is_not_null()).select(
            "_i" if on == "_i" else pl.col("t"),
            *by,
            pl.col("t").alias("_then"),
            pl.col(c).alias("_value"),
        )
        joined = asof(out, right, on, by)
        out = joined.with_columns(
            pl.col("_value").alias(c), (pl.col("t") - pl.col("_then")).alias(f"{c}_age")
        ).drop("_then", "_value")
    return out.drop("_i")


def by_forward_fill(merged: pl.DataFrame, carry: list[str], by: list[str]) -> pl.DataFrame:
    """Each carried column filled forward over its group, Polars'
    ``forward_fill``, and its age from the clock filled forward the same."""

    def over(e: pl.Expr) -> pl.Expr:
        return e.over(by) if by else e

    return merged.with_columns(
        *[over(pl.col(c).forward_fill()) for c in carry],
        *[
            (
                pl.col("t")
                - over(pl.when(pl.col(c).is_not_null()).then(pl.col("t")).forward_fill())
            ).alias(f"{c}_age")
            for c in carry
        ],
    )


@st.composite
def streams(
    draw: st.DrawFn, distinct: bool = False
) -> tuple[dict[str, pl.DataFrame], list[str], pl.DataType]:
    """One to four streams on one clock dtype, ties within and across them
    unless ``distinct``; ``x`` in every stream, ``only_<i>`` in some, ``s``
    (text) in the first; each null now and then; a group ``g``."""
    dtype = draw(st.sampled_from(CLOCK_DTYPES))
    n = draw(st.integers(1, 4))
    if distinct:
        ticks = sorted(draw(st.sets(st.integers(0, 200), max_size=60)))
        owner = [draw(st.integers(0, n - 1)) for _ in ticks]
        per = [[t for t, o in zip(ticks, owner, strict=True) if o == i] for i in range(n)]
    else:
        per = []
        for _ in range(n):
            steps = draw(st.lists(st.integers(0, 2), max_size=20))
            at, out = 0, []
            for s in steps:
                at += s
                out.append(at)
            per.append(out)
    frames: dict[str, pl.DataFrame] = {}
    carry = ["x"]
    for i, ticks in enumerate(per):
        m = len(ticks)
        nulls = draw(st.lists(st.booleans(), min_size=m, max_size=m))
        cols: dict[str, pl.Series] = {
            "t": clock(ticks, dtype),
            "g": pl.Series("g", [draw(st.sampled_from("ab")) for _ in range(m)], dtype=pl.String),
            "x": pl.Series(
                "x", [None if z else i * 100 + r for r, z in enumerate(nulls)], dtype=pl.Int64
            ),
        }
        if i == 0:
            cols["s"] = pl.Series(
                "s", [None if z else f"s{r}" for r, z in enumerate(nulls[::-1])], dtype=pl.String
            )
            carry.append("s")
        elif draw(st.booleans()):
            more = draw(st.lists(st.booleans(), min_size=m, max_size=m))
            cols[f"only_{i}"] = pl.Series(
                f"only_{i}", [None if z else r + 0.5 for r, z in enumerate(more)], dtype=pl.Float64
            )
            carry.append(f"only_{i}")
        frames[f"s{i}"] = pl.DataFrame(cols)
    return frames, carry, dtype


@PROPERTY
@given(streams(), st.booleans(), st.integers(1, 7))
def test_carry_is_polars_join_asof_and_forward_fill_to_the_bit(drawn, grouped, chunk_size):
    """Fed in chunks of any size, each carried column and its age are the
    backward ``join_asof`` of the merged rows on their position, and
    ``forward_fill`` over the group, on single chunks."""
    frames, carry, _ = drawn
    by = ["g"] if grouped else []
    got = rp.stream.merge(
        frames, clock="t", group=by[0] if by else None, carry=carry, chunk_size=chunk_size
    )
    merged = sorted_merge(frames)
    assert_frame_equal(got, by_asof(merged, carry, by, on="_i"), check_exact=True)
    assert_frame_equal(got, by_forward_fill(merged, carry, by), check_exact=True)
    lazy = rp.stream.merge(
        {k: f.lazy() for k, f in frames.items()},
        clock="t",
        group=by[0] if by else None,
        carry=carry,
        chunk_size=chunk_size,
    )
    assert_frame_equal(lazy.collect(), got, check_exact=True)


@PROPERTY
@given(streams(distinct=True), st.booleans(), st.integers(1, 7))
def test_on_distinct_clocks_carry_is_join_asof_on_the_clock(drawn, grouped, chunk_size):
    """Where no two rows share a clock, the backward ``join_asof`` on the
    clock itself: the last value at or before each row's clock."""
    frames, carry, _ = drawn
    by = ["g"] if grouped else []
    got = rp.stream.merge(
        frames, clock="t", group=by[0] if by else None, carry=carry, chunk_size=chunk_size
    )
    assert_frame_equal(got, by_asof(sorted_merge(frames), carry, by, on="t"), check_exact=True)


def test_a_null_group_carries_its_own_values():
    a = pl.DataFrame({"t": [1, 2, 3, 4], "g": [None, "k", None, "k"], "x": [1, 2, None, None]})
    got = rp.stream.merge({"a": a}, clock="t", group="g", carry="x")
    assert got["x"].to_list() == [1, 2, 1, 2]
    assert got["x_age"].to_list() == [0, 0, 2, 2]
    assert_frame_equal(got, by_forward_fill(sorted_merge({"a": a}), ["x"], ["g"]))


def test_before_any_value_the_value_and_its_age_are_null():
    """Not a count of rows: until its group has a value, a row has none."""
    quotes = pl.DataFrame({"t": [3.0, 5.0], "bid": [None, 10.0]})
    trades = pl.DataFrame({"t": [1.0, 4.0, 6.0], "px": [9.0, 9.5, 9.7]})
    got = rp.stream.merge({"quotes": quotes, "trades": trades}, clock="t", carry=["bid"])
    assert got["bid"].to_list() == [None, None, None, 10.0, 10.0]
    assert got["bid_age"].to_list() == [None, None, None, 0.0, 1.0]


def test_a_value_is_carried_within_its_day():
    """Under ``restart_after_step_back`` each day's rows carry that day's
    values alone: the oracle groups by the day as well."""
    day1 = pl.DataFrame({"t": [600, 900, 1500], "d": [1, 1, 1], "bid": [1.0, None, 2.0]})
    day2 = pl.DataFrame({"t": [960, 1200, 1700], "d": [2, 2, 2], "bid": [None, 3.0, None]})
    quotes = pl.concat([day1, day2])
    trades = pl.DataFrame({"t": [700, 1600, 1000, 1650], "d": [1, 1, 2, 2], "px": [5, 6, 7, 8]})
    frames = {"quotes": quotes, "trades": trades}
    got = rp.stream.merge(frames, clock="t", carry="bid", restart_after_step_back=60, chunk_size=2)
    merged = sorted_merge({k: f for k, f in frames.items()}).sort("d", maintain_order=True)
    want = by_forward_fill(merged, ["bid"], ["d"])
    assert_frame_equal(got, want)
    # Day 2's first rows, at 960 and 1000, come after day 1's 2.0 at 1500
    # and find nothing: that value is the other day's.
    assert got.filter(pl.col("d") == 2)["bid"].to_list()[:2] == [None, None]


def test_staleness_is_a_threshold_on_the_age():
    """The how-to's example: trades priced against a quote no older than a
    second, written in Polars."""
    at = [
        datetime(2024, 1, 2, 9, 30, s, us)
        for s, us in [(0, 0), (1, 0), (4, 0), (1, 500_000), (3, 0)]
    ]
    quotes = pl.DataFrame({"t": at[:3], "bid": [99.9, 100.0, 100.2]})
    trades = pl.DataFrame({"t": at[3:], "price": [100.05, 100.4]})
    merged = rp.stream.merge({"quotes": quotes, "trades": trades}, clock="t", carry="bid")
    fresh = merged.filter(pl.col("source") == "trades").with_columns(
        pl.when(pl.col("bid_age") <= pl.duration(seconds=1)).then(pl.col("bid")).alias("bid_fresh")
    )
    assert fresh["bid"].to_list() == [100.0, 100.0]
    assert fresh["bid_fresh"].to_list() == [100.0, None]


def cut(frames: dict[str, pl.DataFrame], at: pl.Expr, ties: dict[str, int]):
    """Each stream's rows before ``at``, and ``ties[name]`` of its rows at
    it; then the rest."""
    first, rest = {}, {}
    for k, f in frames.items():
        n = f.filter(pl.col("t") < at).height + ties.get(k, 0)
        first[k], rest[k] = f.head(n), f.slice(n)
    return first, rest


@PROPERTY
@given(streams(), st.booleans(), st.integers(0, 30), st.integers(1, 5), st.data())
def test_a_run_cut_at_any_clock_resumes_to_the_one_run(drawn, grouped, at_step, chunk_size, data):
    frames, carry, dtype = drawn
    group = "g" if grouped else None
    whole = rp.stream.merge(frames, clock="t", group=group, carry=carry)
    at = pl.lit(clock([at_step], dtype))
    ties = {
        k: data.draw(st.integers(0, f.filter(pl.col("t") == at).height)) for k, f in frames.items()
    }
    first, rest = cut(frames, at, ties)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "merge.state")
        out1 = rp.stream.merge(
            first, clock="t", group=group, carry=carry, chunk_size=chunk_size, save_state=path
        )
        out2 = rp.stream.merge(
            rest, clock="t", group=group, carry=carry, chunk_size=chunk_size, load_state=path
        )
    assert_frame_equal(pl.concat([out1, out2]), whole, check_exact=True)


def test_a_slice_saves_the_values_carried_so_far(tmp_path):
    frames = {
        "a": pl.DataFrame({"t": [1.0, 2.0, 2.0, 5.0], "x": [0, None, 2, None], "g": list("pqpq")}),
        "b": pl.DataFrame({"t": [0.5, 2.0, 3.0], "y": ["u", None, "w"], "g": list("qpp")}),
    }
    kw = {"clock": "t", "group": "g", "carry": ["x", "y"]}
    whole = rp.stream.merge(frames, **kw)
    state = tmp_path / "s"
    for n in range(1, whole.height + 1):
        for chunk_size in (1, 3):
            lazy = rp.stream.merge(
                {k: f.lazy() for k, f in frames.items()},
                chunk_size=chunk_size,
                save_state=state,
                **kw,
            )
            head = lazy.head(n).collect()
            took = head["source"].value_counts()
            counts = dict(zip(took["source"], took["count"], strict=True))
            rest = {k: f.slice(counts.get(k, 0)) for k, f in frames.items()}
            tail = rp.stream.merge(rest, load_state=state, **kw)
            assert_frame_equal(pl.concat([head, tail]), whole)


def test_a_filter_on_the_age_is_pushed_into_the_merge():
    frames = {
        "a": pl.DataFrame({"t": [1, 2, 6], "x": [1, None, None]}),
        "b": pl.DataFrame({"t": [3, 4], "y": [7, 8]}),
    }
    lazy = rp.stream.merge({k: f.lazy() for k, f in frames.items()}, clock="t", carry="x")
    got = lazy.filter(pl.col("x_age") <= 2).select("t", "x").collect()
    assert got.to_dicts() == [{"t": 1, "x": 1}, {"t": 2, "x": 1}, {"t": 3, "x": 1}]


@pytest.mark.parametrize(
    ("carry", "error", "words"),
    [
        ("nope", ColumnNotFoundError, 'carry names column "nope", which no input has'),
        ("t", ValueError, 'carry names "t", the clock column'),
        (["x", "x"], ValueError, 'carry names column "x" twice'),
        ("w", ValueError, 'its age in a column named "w_age"'),
        ([1], WrongTypeError, "carry must name columns"),
    ],
)
def test_a_bad_carry_is_refused_when_the_query_is_built(carry, error, words):
    a = pl.DataFrame({"t": [1.0], "x": [1], "w": [1.0], "w_age": [0.0]})
    with pytest.raises(error, match=words):
        rp.stream.merge({"a": a.lazy()}, clock="t", carry=carry)


def test_a_state_carries_only_into_the_carry_it_was_saved_with(tmp_path):
    a = pl.DataFrame({"t": [1.0, 2.0], "x": [1, None], "y": ["p", None]})
    state = tmp_path / "s"
    rp.stream.merge({"a": a}, clock="t", save_state=state)
    with pytest.raises(ValueError, match=r'saved carrying \[\], but this call carries \["x"\]'):
        rp.stream.merge({"a": a}, clock="t", carry="x", load_state=state)
    rp.stream.merge({"a": a}, clock="t", carry=["x", "y"], save_state=state)
    with pytest.raises(ValueError, match=r'saved carrying \["x", "y"\]'):
        rp.stream.merge({"a": a}, clock="t", carry=["y", "x"], load_state=state)
    got = rp.stream.merge(
        {"a": pl.DataFrame({"t": [3.0], "x": [None], "y": [None]}, schema=a.schema)},
        clock="t",
        carry=["x", "y"],
        load_state=state,
    )
    # The row at 2, held at the save, then the row at 3: each carries the
    # values of 1 across the save.
    assert got.rows() == [(2.0, 1, "p", "a", 1.0, 1.0), (3.0, 1, "p", "a", 2.0, 2.0)]


# --- refresh_time(age=True) ---------------------------------------------------


def usable(v: pl.Expr) -> pl.Expr:
    """A value the grid takes: not null, NaN, infinite or past 1e100."""
    return v.is_not_null() & (v.abs() <= 1e100)


@st.composite
def ticks(draw: st.DrawFn, distinct: bool = False) -> tuple[pl.DataFrame, pl.DataType]:
    """Three series in a random order on a clock of any dtype, ties unless
    ``distinct``, values missing now and then, two groups; ``i`` the row."""
    dtype = draw(st.sampled_from(CLOCK_DTYPES))
    n = draw(st.integers(0, 80))
    if distinct:
        at = sorted(draw(st.sets(st.integers(0, 400), min_size=n, max_size=n)))
    else:
        at, t = [], 0
        for _ in range(n):
            t += draw(st.integers(0, 2))
            at.append(t)
    values = [
        draw(st.one_of(st.none(), st.just(float("nan")), st.floats(-5, 5, allow_nan=False)))
        if draw(st.integers(0, 5)) == 0
        else float(r)
        for r in range(n)
    ]
    df = pl.DataFrame(
        {
            "series": [draw(st.sampled_from("abc")) for _ in range(n)],
            "t": clock(at, dtype),
            "v": pl.Series("v", values, dtype=pl.Float64),
            "g": [draw(st.sampled_from("pq")) for _ in range(n)],
            "i": pl.Series("i", range(n), dtype=pl.Int64),
        },
        schema_overrides={"series": pl.String, "g": pl.String},
    )
    return df, dtype


def grid_oracle(
    df: pl.DataFrame, grid: pl.DataFrame, series: list[str], cols: list[str], by: list[str], on: str
) -> pl.DataFrame:
    """Each series' age, and its value, as the backward ``join_asof`` of the
    point's completing tick (``i``, or its clock) onto the series' ticks
    with a value."""
    out = grid
    left = "i" if on == "i" else "time_refresh"
    for s, col in zip(series, cols, strict=True):
        right = df.filter((pl.col("series") == s) & usable(pl.col("v"))).select(
            pl.col("i" if on == "i" else "t").alias(left), *by, _then=pl.col("t"), _v=pl.col("v")
        )
        out = (
            asof(out, right, left, by)
            .with_columns(
                (pl.col("time_refresh") - pl.col("_then")).alias(f"{col}_age"),
                pl.col("_v").alias(f"_{col}_oracle_value"),
            )
            .drop("_then", "_v")
        )
    return out


@PROPERTY
@given(ticks(), st.booleans(), st.integers(1, 9))
def test_each_age_is_polars_join_asof_backward(drawn, grouped, chunk_size):
    df, _ = drawn
    by = ["g"] if grouped else []
    kw = {"series": "series", "names": ["a", "b", "c"], "clock": "t", "value": "v"}
    kw |= {"group": by[0] if by else None, "keep": ["i"]}
    got = rp.stream.refresh_time(df, age=True, chunk_size=chunk_size, **kw)
    plain = rp.stream.refresh_time(df, **kw)
    ages = ["a_age", "b_age", "c_age"]
    assert_frame_equal(got.drop(ages), plain, check_exact=True)
    want = grid_oracle(df, plain, ["a", "b", "c"], ["a", "b", "c"], by, on="i")
    assert_frame_equal(got, want.select(got.columns), check_exact=True)


@PROPERTY
@given(ticks(), st.booleans(), st.integers(1, 9))
def test_each_pairs_age_is_polars_join_asof_backward(drawn, grouped, chunk_size):
    df, _ = drawn
    by = ["g"] if grouped else []
    kw = {"series": "series", "names": ["a", "b", "c"], "clock": "t", "value": "v"}
    kw |= {"group": by[0] if by else None, "keep": ["i"], "pairs": True}
    got = rp.stream.refresh_time(df, age=True, chunk_size=chunk_size, **kw)
    plain = rp.stream.refresh_time(df, **kw)
    assert_frame_equal(got.drop("a_age", "b_age"), plain, check_exact=True)
    parts = []
    for pair, rows in plain.with_row_index("_r").group_by("pair", maintain_order=True):
        a, b = str(pair[0]).split("|")
        parts.append(grid_oracle(df, rows, [a, b], ["a", "b"], by, on="i"))
    if parts:
        want = pl.concat(parts).sort("_r")
        assert_frame_equal(got, want.select(got.columns), check_exact=True)


@PROPERTY
@given(ticks(distinct=True), st.integers(1, 9))
def test_on_distinct_clocks_each_value_and_age_is_join_asof_on_the_clock(drawn, chunk_size):
    df, _ = drawn
    kw = {"series": "series", "names": ["a", "b", "c"], "clock": "t", "value": "v"}
    got = rp.stream.refresh_time(df, age=True, chunk_size=chunk_size, **kw)
    want = grid_oracle(df, got.drop("a_age", "b_age", "c_age"), *[["a", "b", "c"]] * 2, [], "t")
    for s in "abc":
        assert_frame_equal(
            got.select(f"{s}_value", f"{s}_age"),
            want.select(pl.col(f"_{s}_oracle_value").alias(f"{s}_value"), f"{s}_age"),
            check_exact=True,
        )


@pytest.mark.parametrize("pairs", [False, True])
def test_a_grid_with_ages_resumes_and_slices_to_the_one_run(tmp_path, pairs):
    """Two runs through a saved state, and a slice resumed after the tick
    behind its last point (a pair grid's slice inside a tick holds that
    tick's other points, with their ages), give the one run's grid."""
    df = pl.DataFrame(
        {
            "series": list("abcabbcaacbcabca"),
            "t": [float(r // 2) for r in range(16)],
            "v": [float(r) for r in range(16)],
            "i": list(range(16)),
        }
    )
    kw = {"series": "series", "names": ["a", "b", "c"], "clock": "t", "value": "v"}
    kw |= {"age": True, "pairs": pairs, "keep": ["i"]}
    whole = rp.stream.refresh_time(df, **kw)
    state = tmp_path / "s"
    for split in (1, 5, 9, 15):
        first = rp.stream.refresh_time(df.head(split), save_state=state, **kw)
        rest = rp.stream.refresh_time(df.slice(split), load_state=state, **kw)
        assert_frame_equal(pl.concat([first, rest]), whole)
    for n in range(1, whole.height + 1):
        lazy = rp.stream.refresh_time(df.lazy(), chunk_size=2, save_state=state, **kw)
        head = lazy.head(n).collect()
        rest = rp.stream.refresh_time(df.slice(head["i"][-1] + 1), load_state=state, **kw)
        assert_frame_equal(pl.concat([head, rest]), whole)


def test_a_state_gives_ages_only_where_it_was_saved_with_them(tmp_path):
    df = pl.DataFrame({"s": ["a", "b", "a"], "t": [1.0, 2.0, 3.0], "v": [1.0, 2.0, 3.0]})
    kw = {"series": "s", "names": ["a", "b"], "clock": "t", "value": "v"}
    state = tmp_path / "s"
    rp.stream.refresh_time(df, save_state=state, **kw)
    with pytest.raises(ValueError, match="saved with age = False, but this call gives age = True"):
        rp.stream.refresh_time(df, age=True, load_state=state, **kw)
    rp.stream.refresh_time(df, age=True, save_state=state, **kw)
    with pytest.raises(ValueError, match="saved with age = True, but this call gives age = False"):
        rp.stream.refresh_time(df, load_state=state, **kw)
    # a's tick at 3, saved, is b's next point's value: aged across the save.
    got = rp.stream.refresh_time(
        pl.DataFrame({"s": ["b"], "t": [4.5], "v": [5.0]}), age=True, load_state=state, **kw
    )
    assert got.select("a_age", "b_age").row(0) == (1.5, 0.0)
