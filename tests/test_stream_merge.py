"""``rp.stream.merge``: N streams, each in clock order, merged into one.

The oracle is Polars itself, a third party's computation of the same order:
the streams concatenated in the order given, each row tagged with its
stream, and sorted by the clock with a stable sort,
``pl.concat(frames, how="diagonal").sort(clock, maintain_order=True)``, on
single chunks; and for streams of one schema, ``merge_sorted`` chained from
the first stream, whose left frame wins a tie. Each is held equal to the
bit. Hypothesis draws the streams -- their number, their clocks' dtype, ties
within and across them -- and the chunking, which never changes the output
(hard rule 3). The rest is the properties that make the merge usable as a
streaming stage: the disorder rules a bank applies, the refusals named, a
slice and a saved state that resume to the one run's rows, and a pipeline
checkpoint that carries the state as a stage.
"""

from __future__ import annotations

import os
import tempfile
from datetime import date, datetime, timedelta

import polars as pl
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from polars.testing import assert_frame_equal

from rheon import ColumnNotFoundError, WrongTypeError
from rheon import polars as rp
from tiers import examples

TIER = "essential"

#: How a refusal made while the query runs arrives: inside Polars'
#: ``ComputeError`` under py-polars 1.x, the ``ValueError`` itself under 2.0.
_REFUSAL = (pl.exceptions.ComputeError, ValueError)

PROPERTY = settings(
    max_examples=examples(60, essential=10),
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)


def oracle(frames: dict[str, pl.DataFrame], clock: str, source: str = "source") -> pl.DataFrame:
    """Polars' stable sort of the streams' rows, concatenated in order."""
    tagged = [f.with_columns(pl.lit(name).alias(source)) for name, f in frames.items()]
    out = pl.concat(tagged, how="diagonal").sort(clock, maintain_order=True)
    columns: list[str] = []
    for f in frames.values():
        columns += [c for c in f.columns if c not in columns]
    return out.select(*columns, source)


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


def clock_series(steps: list[int], dtype: pl.DataType) -> pl.Series:
    """A sorted clock of ``dtype`` from non-negative integer ``steps``."""
    ticks: list[int] = []
    t = 0
    for s in steps:
        t += s
        ticks.append(t)
    if dtype.is_float():
        return pl.Series("t", [x / 4 for x in ticks], dtype=dtype)
    if dtype.is_integer():
        return pl.Series("t", ticks, dtype=dtype)
    if isinstance(dtype, pl.Date):
        return pl.Series("t", [date(2024, 1, 1) + timedelta(days=x) for x in ticks], dtype=pl.Date)
    if isinstance(dtype, pl.Duration):
        return pl.Series("t", ticks, dtype=pl.Int64).cast(dtype)
    base = datetime(2024, 3, 31, 0, 0)
    out = pl.Series("t", [base + timedelta(minutes=17 * x) for x in ticks])
    out = out.cast(pl.Datetime(dtype.time_unit))
    if dtype.time_zone is not None:
        out = out.dt.replace_time_zone("UTC").dt.convert_time_zone(dtype.time_zone)
    return out


@st.composite
def streams(draw: st.DrawFn) -> tuple[dict[str, pl.DataFrame], pl.DataType]:
    """Two to six streams on one clock dtype, ties within and across them,
    each with columns of its own and some in common."""
    dtype = draw(st.sampled_from(CLOCK_DTYPES))
    n = draw(st.integers(1, 6))
    out: dict[str, pl.DataFrame] = {}
    for i in range(n):
        steps = draw(st.lists(st.integers(0, 2), max_size=25))
        t = clock_series(steps, dtype)
        cols = {
            "t": t,
            "x": pl.Series("x", [i * 1000 + r for r in range(len(steps))], dtype=pl.Int64),
        }
        if draw(st.booleans()):
            cols[f"only_{i}"] = pl.Series(
                f"only_{i}", [float(r) for r in range(len(steps))], dtype=pl.Float64
            )
        if draw(st.booleans()):
            cols["w"] = pl.Series("w", [float(r % 2) for r in range(len(steps))], dtype=pl.Float64)
        out[f"s{i}"] = pl.DataFrame(cols)
    return out, dtype


@PROPERTY
@given(streams(), st.integers(1, 7))
def test_the_merge_is_polars_stable_sort_to_the_bit(drawn, chunk_size):
    """Polars' sort of the concatenated streams, on single chunks, is the
    merge of the streams fed in chunks of any size."""
    frames, _ = drawn
    got = rp.stream.merge(frames, clock="t", chunk_size=chunk_size)
    assert_frame_equal(got, oracle(frames, "t"), check_exact=True)
    lazy = rp.stream.merge(
        {k: f.lazy() for k, f in frames.items()}, clock="t", chunk_size=chunk_size
    )
    assert_frame_equal(lazy.collect(), got, check_exact=True)


@PROPERTY
@given(streams())
def test_merge_sorted_chained_from_the_first_stream_is_the_merge(drawn):
    """Polars' ``merge_sorted``, two frames at a time from the first, whose
    left frame wins a tie: the same rows in the same order. Eager: the
    streaming engine's ``merge_sorted`` (a ``LazyFrame``'s, collected by
    default) breaks a tie across its two frames in no fixed order --
    ``[0, 0]`` merged with ``[0]`` comes out left, right, left under
    py-polars 2.1.0 -- so it is no oracle for ties."""
    frames, _ = drawn
    same = {k: f.select("t", "x") for k, f in frames.items()}
    tagged = [f.with_columns(pl.lit(k).alias("source")) for k, f in same.items()]
    chained = tagged[0]
    for f in tagged[1:]:
        chained = chained.merge_sorted(f, key="t")
    assert_frame_equal(rp.stream.merge(same, clock="t", chunk_size=3), chained, check_exact=True)


def test_two_streams_of_ties_go_to_the_first_then_keep_their_own_order():
    a = pl.DataFrame({"t": [1, 2, 2], "x": [0, 1, 2]})
    b = pl.DataFrame({"t": [0, 2, 2, 3], "x": [10, 11, 12, 13]})
    out = rp.stream.merge(a, b, clock="t", chunk_size=1)
    assert out["x"].to_list() == [10, 0, 1, 2, 11, 12, 13]
    assert out["source"].to_list() == ["1", "0", "0", "0", "1", "1", "1"]
    assert_frame_equal(out, a.merge_sorted(b, key="t").with_columns(out["source"]))


def test_the_kind_of_frame_it_was_given_comes_back():
    a = pl.DataFrame({"t": [1.0]})
    assert isinstance(rp.stream.merge(a, a, clock="t"), pl.DataFrame)
    assert isinstance(rp.stream.merge(a.lazy(), a, clock="t"), pl.LazyFrame)
    assert isinstance(rp.stream.merge({"a": a}, clock="t"), pl.DataFrame)
    assert rp.stream.merge(a, clock="t", source="from").columns == ["t", "from"]


def test_streams_read_from_files_stream_and_slice(tmp_path):
    """Files scanned lazily, a projection, a filter and a slice of the
    output, each as Polars would apply it to the merged frame."""
    frames = {
        "quotes": pl.DataFrame({"t": [1.0, 2.0, 4.0, 6.0], "bid": [9.9, 10.0, 10.1, 10.0]}),
        "trades": pl.DataFrame({"t": [2.0, 3.0, 5.0], "price": [10.05, 10.2, 10.1]}),
    }
    for k, f in frames.items():
        f.write_parquet(tmp_path / f"{k}.parquet")
    lazy = rp.stream.merge(
        {k: pl.scan_parquet(tmp_path / f"{k}.parquet") for k in frames}, clock="t", chunk_size=2
    )
    want = oracle(frames, "t")
    assert_frame_equal(lazy.collect(), want)
    assert_frame_equal(lazy.head(3).collect(), want.head(3))
    assert_frame_equal(lazy.select("t", "source").collect(), want.select("t", "source"))
    assert_frame_equal(lazy.filter(pl.col("t") > 2).collect(), want.filter(pl.col("t") > 2))


def test_a_zero_weight_row_is_a_row_like_any_other():
    a = pl.DataFrame({"t": [1.0, 2.0, 3.0], "w": [1.0, 0.0, 1.0], "y": [0.5, 0.7, 0.9]})
    b = pl.DataFrame({"t": [2.0, 2.5], "w": [0.0, 0.0], "y": [1.0, 2.0]})
    out = rp.stream.merge(a, b, clock="t")
    assert out["w"].to_list() == [1.0, 0.0, 0.0, 0.0, 1.0]
    assert out.height == 5


@pytest.mark.parametrize(
    ("frames", "error", "words"),
    [
        (
            [pl.DataFrame({"t": [1]}), pl.DataFrame({"t": [1.0]})],
            WrongTypeError,
            'the clocks are not of one kind: input "0"\'s clock column "t" is Int64 and '
            'input "1"\'s is Float64',
        ),
        (
            [
                pl.DataFrame({"t": [datetime(2024, 1, 1)]}).with_columns(
                    pl.col("t").dt.replace_time_zone("UTC")
                ),
                pl.DataFrame({"t": [datetime(2024, 1, 1)]}),
            ],
            WrongTypeError,
            "the clocks are not of one kind",
        ),
        (
            [pl.DataFrame({"t": [1], "a": [1.0]}), pl.DataFrame({"t": [1], "a": ["x"]})],
            WrongTypeError,
            'column "a" is Float64 in input "0" and String in input "1"',
        ),
        ([pl.DataFrame({"t": [1]}), pl.DataFrame({"u": [1]})], ColumnNotFoundError, "clock"),
        ([pl.DataFrame({"t": ["a"]})], WrongTypeError, "which is no clock"),
        ([pl.DataFrame({"t": [1], "source": [1]})], ValueError, "Pass another source="),
        ([], ValueError, "no input is given"),
        ([pl.DataFrame({"t": [1]}), 3], WrongTypeError, "stream '1' is a int, not a frame"),
    ],
)
def test_a_bad_call_is_refused_by_name_when_the_query_is_built(frames, error, words):
    with pytest.raises(error, match=None) as e:
        rp.stream.merge(*frames, clock="t")
    assert words in str(e.value)


def test_a_stream_without_the_group_column_and_a_bad_name_are_refused():
    a = pl.DataFrame({"t": [1], "g": ["A"]})
    with pytest.raises(ColumnNotFoundError, match='input "1" has no group column "g"'):
        rp.stream.merge(a, pl.DataFrame({"t": [1]}), clock="t", group="g")
    with pytest.raises(WrongTypeError, match="a stream's name must be text"):
        rp.stream.merge({1: a}, clock="t")


@pytest.mark.parametrize(
    ("t", "words"),
    [
        (
            [0.0, 2.0, 1.5],
            'input "0": clock column "t" steps back by 0.5 at row 2 (counting from 0)',
        ),
        ([0.0, None, 1.5], 'input "0": clock column "t" is null at row 1'),
        ([0.0, float("nan")], "is NaN at row 1"),
    ],
)
def test_a_bad_clock_is_refused_naming_its_stream_and_row(t, words):
    with pytest.raises(_REFUSAL) as e:
        rp.stream.merge(pl.DataFrame({"t": t}), clock="t", chunk_size=1)
    assert words in str(e.value)
    assert "restart_after_step_back" in str(e.value) or "Drop the rows" in str(e.value)


def test_a_stream_that_starts_again_merges_after_every_row_before_it():
    """Two days of two streams, each day's clock from 0: under
    ``restart_after_step_back`` the second day follows the first."""
    a = pl.DataFrame({"t": [1.0, 5.0, 9.0, 0.5, 4.0], "x": [0, 1, 2, 3, 4]})
    b = pl.DataFrame({"t": [2.0, 8.0, 1.0, 6.0], "x": [10, 11, 12, 13]})
    out = rp.stream.merge(a, b, clock="t", restart_after_step_back=1, chunk_size=2)
    assert out["x"].to_list() == [0, 10, 1, 11, 2, 3, 12, 4, 13]
    with pytest.raises(_REFUSAL, match="no more than restart_after_step_back = 9"):
        rp.stream.merge(a, b, clock="t", restart_after_step_back=9).height  # noqa: B018
    # A bank under the same setting reads the merged rows in order.
    spec = rp.spec.ewridge(
        "m",
        targets=["x"],
        features=["t"],
        clock="t",
        gap_cap=10.0,
        half_life=5.0,
        restart_after_step_back=1.0,
    )
    rp.ModelBank([spec]).fit_predict(out.with_columns(pl.col("x").cast(pl.Float64)))


def test_restart_after_step_back_takes_a_duration_on_a_temporal_clock():
    day = [datetime(2024, 1, 1, 9), datetime(2024, 1, 1, 17), datetime(2024, 1, 1, 9, 30)]
    a = pl.DataFrame({"t": day})
    out = rp.stream.merge(a, clock="t", restart_after_step_back="1h")
    assert out.height == 3
    with pytest.raises(_REFUSAL, match="steps back by 7h30m at row 2"):
        rp.stream.merge(a, clock="t", restart_after_step_back="8h").height  # noqa: B018
    with pytest.raises(ValueError, match="must be a duration"):
        rp.stream.merge(a, clock="t", restart_after_step_back=3600)
    with pytest.raises(ValueError, match="is a duration"):
        rp.stream.merge(pl.DataFrame({"t": [1.0]}), clock="t", restart_after_step_back="1h")
    with pytest.raises(ValueError, match="0 or more"):
        rp.stream.merge(pl.DataFrame({"t": [1.0]}), clock="t", restart_after_step_back=-1)


def test_group_checks_each_groups_merged_rows_as_a_bank_would():
    """A group whose rows step back by little where its stream started
    again is refused, naming the group; another group is not."""
    a = pl.DataFrame({"g": ["A", "B", "A"], "t": [10.0, 20.0, 9.5]})
    with pytest.raises(_REFUSAL, match='the merged rows of g = "A" step back by 0.5'):
        rp.stream.merge(a, clock="t", group="g", restart_after_step_back=1).height  # noqa: B018
    b = pl.DataFrame({"g": ["A", "B", "A"], "t": [10.0, 20.0, 1.0]})
    assert rp.stream.merge(b, clock="t", group="g", restart_after_step_back=1).height == 3


def cut(frames: dict[str, pl.DataFrame], at: float, ties: dict[str, int]):
    """Each stream's rows below ``at``, and ``ties[name]`` of its rows at
    it; then the rest."""
    first, rest = {}, {}
    for k, f in frames.items():
        n = f.filter(pl.col("t") < at).height + ties.get(k, 0)
        first[k], rest[k] = f.head(n), f.slice(n)
    return first, rest


def test_two_runs_cut_at_one_clock_give_the_one_run(tmp_path):
    """A cut inside the tie at 2: the state holds the rows at the last
    clock, and the resumed run returns them in their place."""
    frames = {
        "a": pl.DataFrame({"t": [1.0, 2.0, 3.0], "x": [0, 1, 2]}),
        "b": pl.DataFrame({"t": [2.0, 2.0, 4.0], "x": [10, 11, 12]}),
        "c": pl.DataFrame({"t": [0.5, 2.0, 2.0, 5.0], "x": [20, 21, 22, 23]}),
    }
    whole = rp.stream.merge(frames, clock="t")
    first, rest = cut(frames, 2.0, {"a": 1, "b": 1, "c": 2})
    state = tmp_path / "merge.state"
    for chunk_size in (1, 2, 50):
        out1 = rp.stream.merge(first, clock="t", chunk_size=chunk_size, save_state=state)
        assert out1["x"].to_list() == [20, 0], "the rows at 2 are held"
        out2 = rp.stream.merge(rest, clock="t", chunk_size=chunk_size, load_state=state)
        assert_frame_equal(pl.concat([out1, out2]), whole)


@PROPERTY
@given(streams(), st.integers(0, 40), st.integers(1, 5), st.data())
def test_any_cut_at_one_clock_resumes_to_the_one_run(drawn, at_step, chunk_size, data):
    frames, dtype = drawn
    whole = rp.stream.merge(frames, clock="t")
    at = clock_series([at_step], dtype)[0]
    ties = {
        k: data.draw(st.integers(0, f.filter(pl.col("t") == at).height)) for k, f in frames.items()
    }
    first, rest = cut(frames, at, ties)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "merge.state")
        out1 = rp.stream.merge(first, clock="t", chunk_size=chunk_size, save_state=path)
        out2 = rp.stream.merge(rest, clock="t", chunk_size=chunk_size, load_state=path)
    assert_frame_equal(pl.concat([out1, out2]), whole, check_exact=True)


def test_a_resumed_run_on_streams_cut_at_different_clocks_is_refused(tmp_path):
    a = pl.DataFrame({"t": [1.0, 3.0]})
    b = pl.DataFrame({"t": [2.0, 4.0, 5.0]})
    state = tmp_path / "s"
    rp.stream.merge({"a": a.head(1), "b": b}, clock="t", save_state=state)
    with pytest.raises(_REFUSAL, match="Cut every input at the same clock"):
        rp.stream.merge({"a": a.slice(1), "b": b.clear()}, clock="t", load_state=state)


def test_a_slice_saves_the_state_after_its_rows_and_each_stream_resumes_after_them(tmp_path):
    frames = {
        "a": pl.DataFrame({"t": [1.0, 2.0, 2.0, 5.0], "x": [0, 1, 2, 3]}),
        "b": pl.DataFrame({"t": [0.5, 2.0, 3.0], "x": [10, 11, 12]}),
    }
    whole = rp.stream.merge(frames, clock="t")
    state = tmp_path / "s"
    # From 1: Polars answers a slice of no rows without running the source.
    for n in range(1, whole.height + 1):
        for chunk_size in (1, 3):
            lazy = rp.stream.merge(
                {k: f.lazy() for k, f in frames.items()},
                clock="t",
                chunk_size=chunk_size,
                save_state=state,
            )
            head = lazy.head(n).collect()
            took = head["source"].value_counts()
            counts = dict(zip(took["source"], took["count"], strict=True))
            rest = {k: f.slice(counts.get(k, 0)) for k, f in frames.items()}
            tail = rp.stream.merge(rest, clock="t", load_state=state)
            assert_frame_equal(pl.concat([head, tail]), whole)


def test_a_state_loads_only_into_the_merge_it_holds(tmp_path):
    a = pl.DataFrame({"t": [1.0, 2.0]})
    state = tmp_path / "s"
    rp.stream.merge({"a": a, "b": a}, clock="t", save_state=state)
    with pytest.raises(ValueError, match=r"saved merging the inputs \[\"a\", \"b\"\]"):
        rp.stream.merge({"b": a, "a": a}, clock="t", load_state=state)
    with pytest.raises(ValueError, match="saved on a clock of dtype Float64"):
        rp.stream.merge(
            {"a": a.cast({"t": pl.Int64}), "b": a.cast({"t": pl.Int64})},
            clock="t",
            load_state=state,
        )
    grid = tmp_path / "g"
    rp.stream.refresh_time(
        pl.DataFrame({"s": ["a", "b"], "t": [1.0, 2.0], "v": [1.0, 2.0]}),
        series="s",
        names=["a", "b"],
        clock="t",
        value="v",
        save_state=grid,
    )
    with pytest.raises(ValueError, match="a refresh_time run's state, not a merge's"):
        rp.stream.merge({"a": a, "b": a}, clock="t", load_state=grid)


def test_the_state_is_a_stage_of_a_pipeline_checkpoint(tmp_path):
    frames = {
        "a": pl.DataFrame({"t": [1.0, 2.0, 3.0]}),
        "b": pl.DataFrame({"t": [1.5, 2.0, 2.5]}),
    }
    whole = rp.stream.merge(frames, clock="t")
    first, rest = cut(frames, 2.0, {"a": 1})
    state = tmp_path / "merge.state"
    out1 = rp.stream.merge(first, clock="t", save_state=state)
    rp.stream.save_checkpoint(tmp_path / "run.checkpoint", {"merge": state})
    stages = rp.stream.load_checkpoint(tmp_path / "run.checkpoint")
    assert list(stages) == ["merge"]
    out2 = rp.stream.merge(rest, clock="t", load_state=stages["merge"])
    assert_frame_equal(pl.concat([out1, out2]), whole)


@pytest.mark.parametrize("dirty", [float("nan"), 1.5])
def test_a_slice_reads_no_chunk_past_its_last_row(dirty):
    """Review 2026-11-23, MERGE-1: a slice whose last row ended a stream's
    chunk pulled that stream's next chunk and pushed it before it stopped,
    so a row past the slice -- a clock that is not a number, or a step back
    -- refused a slice that never reaches it. The slice reads each stream
    up to its last row returned, so it returns the rows."""
    a = pl.DataFrame({"t": [1.0, 2.0, dirty]})
    b = pl.DataFrame({"t": [10.0, 20.0]})
    got = rp.stream.merge(a.lazy(), b.lazy(), clock="t", chunk_size=2).head(2).collect()
    assert got["t"].to_list() == [1.0, 2.0]


def test_restart_after_step_back_is_exact_past_104_days_and_2_to_53():
    """Review 2026-11-23, MERGE-2: `restart_after_step_back` stays exact in
    nanoseconds past 104.25 days (`2^53` ns) and exact in integers past `2^53`,
    and huge integers like `10**400` do not raise `OverflowError`."""
    ns_105d = 105 * 86_400 * 1_000_000_000
    t0 = 1_700_000_000_000_000_000
    # Step back of exactly 105d1ns with threshold "105d1ns": must be REFUSED
    # (inclusive), not restarted because float(105d1ns) rounded down to 105d.
    a = pl.DataFrame({"t": pl.Series([t0, t0 - (ns_105d + 1)], dtype=pl.Datetime("ns"))})
    b = pl.DataFrame({"t": pl.Series([t0], dtype=pl.Datetime("ns"))})
    with pytest.raises(_REFUSAL, match="steps back"):
        rp.stream.merge(a, b, clock="t", restart_after_step_back="105d1ns")
    # Step back of 105d2ns with threshold "105d1ns": strictly past -> restarts.
    a_restart = pl.DataFrame({"t": pl.Series([t0, t0 - (ns_105d + 2)], dtype=pl.Datetime("ns"))})
    got = rp.stream.merge(a_restart, b, clock="t", restart_after_step_back="105d1ns")
    assert got.height == 3

    # Step back of 105d4ns with threshold "105d3ns": strictly past -> restarts
    # (previously float(105d3ns) rounded up to 105d4ns and falsely refused).
    a_4 = pl.DataFrame({"t": pl.Series([t0, t0 - (ns_105d + 4)], dtype=pl.Datetime("ns"))})
    got4 = rp.stream.merge(a_4, b, clock="t", restart_after_step_back="105d3ns")
    assert got4.height == 3

    # Int64 clock past 2^53: step back of (1<<53)+1 with threshold (1<<53)+1 is refused,
    # while (1<<53)+2 restarts cleanly.
    big = (1 << 53) + 1
    ai_eq = pl.DataFrame({"t": pl.Series([big + 10, 10], dtype=pl.Int64)})
    ai_gt = pl.DataFrame({"t": pl.Series([big + 10, 9], dtype=pl.Int64)})
    bi = pl.DataFrame({"t": pl.Series([big + 10], dtype=pl.Int64)})
    with pytest.raises(_REFUSAL, match="steps back"):
        rp.stream.merge(ai_eq, bi, clock="t", restart_after_step_back=big)
    assert rp.stream.merge(ai_gt, bi, clock="t", restart_after_step_back=big).height == 3
    # Huge integer (10**400): refuses any finite step back instead of OverflowError.
    with pytest.raises(_REFUSAL, match="steps back"):
        rp.stream.merge(ai_gt, bi, clock="t", restart_after_step_back=10**400)
    # And succeeds on sorted inputs with 10**400.
    ok_i = pl.DataFrame({"t": pl.Series([1, 2], dtype=pl.Int64)})
    assert rp.stream.merge(ok_i, ok_i, clock="t", restart_after_step_back=10**400).height == 4


def test_merge_refuses_unreadable_group_dtype_at_build():
    """Review 2026-11-23, MERGE-4 (1): an unreadable `group` dtype is refused when
    the merge is built, not mid-stream with a raw Polars error."""
    for s in [
        pl.Series("g", [[1, 2], [3]]),
        pl.Series("g", [b"a", b"b"]),
        pl.Series("g", [{"a": 1}, {"a": 2}]),
    ]:
        df = pl.DataFrame({"t": [1.0, 2.0]}).with_columns(s)
        with pytest.raises(TypeError, match='group column "g"'):
            rp.stream.merge(df.lazy(), df.lazy(), clock="t", group="g")
