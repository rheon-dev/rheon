"""``rp.stream.bars``: threshold-triggered bars over a stream.

The oracles, each a third party's computation of the same bars:

- **Polars' own construction** for ``"count"`` and ``"sum"``: the running
  total by ``cum_sum`` per group, a row's bar ``floor(S_before / θ)``, and
  Polars' ``group_by`` over those bars running the operator's own
  aggregations. On integers and on a dyadic grid every partial sum is exact
  in Polars too, so the bars, their aggregates, ``thresholds`` and
  ``carry`` are equal to the bit.
- **The definition in** ``fractions.Fraction`` for general doubles, where a
  double total drifts: the bars, ``thresholds`` and ``carry`` (the exact
  remainder rounded once) equal to the bit; and for ``"imbalance"``, whose
  reset at each bar no cumulative sum expresses, the definition as a loop.
- **bar-forge** (test-only, MIT), the one library of those checked that
  computes the same bars and installs beside the tests' NumPy: its tick bars
  are ``"count"``, its fixed-threshold imbalance bars ``"imbalance"`` over its
  own tick-rule signs, and its volume and dollar bars, which reset at each
  bar, ``"imbalance"`` of a quantity that is never negative. Its sums are
  plain doubles in row order, exact on the dyadic grid, so its bars and
  aggregates are equal to the bit.

The rest is what makes the operator a streaming stage: one chunk or many
give the same bars to the bit, a saved state resumes to the one run's bars,
rows of weight 0 are in no bar, each cut on both sides of its setting, and
the caps on memory.
"""

from __future__ import annotations

import math
import os
import random
import re
import tempfile
import warnings
from fractions import Fraction

import bar_forge
import polars as pl
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from polars.testing import assert_frame_equal

from rheon import ColumnNotFoundError, ReadinessWarning
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

#: The aggregations the oracle runs through Polars' own ``group_by``.
AGGS = {
    "open": pl.col("price").first(),
    "high": pl.col("price").max(),
    "low": pl.col("price").min(),
    "close": pl.col("price").last(),
    "volume": pl.col("size").sum(),
    "buys": pl.when(pl.col("side") > 0).then(pl.col("size")).otherwise(0).sum(),
    "n": pl.len(),
    "sides": pl.col("side").count(),
}


def trades(seed: int, n: int = 600, groups: int = 3, big: bool = False) -> pl.DataFrame:
    """A seeded stream on a dyadic grid: integer sizes, prices on 1/64, so
    every sum the oracles make is exact."""
    rng = random.Random(seed)
    sizes = [rng.randint(1, 100) for _ in range(n)]
    if big:
        for i in rng.sample(range(n), n // 40):
            sizes[i] = rng.randint(300, 2500)
    return pl.DataFrame(
        {
            "t": list(range(n)),
            "sym": [rng.choice("ABC"[:groups]) for _ in range(n)],
            "price": [rng.randint(640, 1279) / 64 for _ in range(n)],
            "size": sizes,
            "side": [rng.choice((-1, 1)) for _ in range(n)],
        }
    )


def sum_oracle(
    df: pl.DataFrame, theta: float | int, q: pl.Expr, group: str | None = "sym"
) -> pl.DataFrame:
    """The bars of ``kind="sum"`` (``kind="count"`` with ``q = 1``) by Polars'
    ``cum_sum`` and ``group_by``: the rows of bar ``k`` are those with
    ``floor(S_before / θ) = k``, and the open bar has crossed nothing."""
    by = [group] if group else []
    s_after = q.cum_sum().over(group) if group else q.cum_sum()
    s_before = s_after - q
    out = (
        df.with_row_index("_row")
        .with_columns(_bar=s_before // theta, _after=s_after // theta, _s=s_after)
        .group_by([*by, "_bar"], maintain_order=True)
        .agg(
            time_open=pl.col("t").first(),
            time_close=pl.col("t").last(),
            thresholds=(pl.col("_after").last() - pl.col("_bar").first()).cast(pl.Int64),
            carry=(pl.col("_s").last() - pl.col("_after").last() * theta).cast(pl.Float64),
            _close=pl.col("_row").last(),
            **AGGS,
        )
        .filter(pl.col("thresholds") > 0)
        .sort("_close")
    )
    bar = pl.int_range(pl.len(), dtype=pl.Int64)
    out = out.with_columns(bar=bar.over(group) if group else bar, complete=pl.lit(True))
    return out.select(
        *by, "bar", "time_open", "time_close", "thresholds", "carry", "complete", *AGGS
    )


def run(df: pl.DataFrame, chunk_size: int | None = None, **kw: object) -> pl.DataFrame:
    kw.setdefault("clock", "t")
    out = rp.stream.bars(df.lazy(), chunk_size=chunk_size, **kw).collect()  # type: ignore[arg-type]
    assert isinstance(out, pl.DataFrame)
    return out


@pytest.mark.parametrize("seed", [1, 2, 3])
@pytest.mark.parametrize("big", [False, True])
def test_sum_bars_are_polars_cum_sum_construction(seed: int, big: bool) -> None:
    df = trades(seed, big=big)
    for theta in (250, 1000):
        got = run(df, kind="sum", quantity="size", threshold=theta, group="sym", **AGGS)
        want = sum_oracle(df, theta, pl.col("size"))
        assert_frame_equal(got, want)
    # Dollar bars on the dyadic grid: every product and partial sum exact.
    dollars = pl.col("price") * pl.col("size")
    got = run(df, kind="sum", quantity=dollars, threshold=7500.0, group="sym", **AGGS)
    assert_frame_equal(got, sum_oracle(df, 7500.0, dollars))


def test_count_bars_are_polars_construction() -> None:
    df = trades(4)
    got = run(df, kind="count", threshold=7, group="sym", **AGGS)
    want = sum_oracle(df, 7, pl.col("size") * 0 + 1).drop("carry")
    assert_frame_equal(got, want)


def fraction_bars(values: list[float], theta: float, reset: bool) -> list[tuple[int, int, float]]:
    """The definition, exact: under ``reset`` the imbalance bar's loop,
    otherwise the multiples of θ on the running total. Each bar is (closing
    row, thresholds, carry rounded once)."""
    th = Fraction(theta)
    total = Fraction(0)
    out = []
    for i, v in enumerate(values):
        total += Fraction(v) if math.isfinite(v) and abs(v) <= 1e100 else 0
        if abs(total) >= th:
            m = math.floor(abs(total) / th)
            if reset:
                out.append((i, m, 0.0))
                total = Fraction(0)
            else:
                total -= m * th
                out.append((i, m, float(total)))
    return out


@PROPERTY
@given(
    values=st.lists(st.floats(0, 1e6), min_size=1, max_size=200),
    theta=st.floats(1e-3, 1e5),
    chunk=st.integers(1, 50),
)
def test_sum_of_general_doubles_is_the_exact_definition(
    values: list[float], theta: float, chunk: int
) -> None:
    df = pl.DataFrame({"t": range(len(values)), "q": values})
    want = fraction_bars(values, theta, reset=False)
    if any(m >= 2**53 for _, m, _ in want):
        return
    got = run(df, chunk, kind="sum", quantity="q", threshold=theta)
    assert got["time_close"].to_list() == [i for i, _, _ in want]
    assert got["thresholds"].to_list() == [m for _, m, _ in want]
    assert got["carry"].to_list() == [c for _, _, c in want]


@PROPERTY
@given(
    values=st.lists(st.floats(-1e6, 1e6), min_size=1, max_size=200),
    theta=st.floats(1e-2, 1e5),
    chunk=st.integers(1, 50),
)
def test_imbalance_is_the_definition_as_a_loop(
    values: list[float], theta: float, chunk: int
) -> None:
    df = pl.DataFrame({"t": range(len(values)), "q": values})
    want = fraction_bars(values, theta, reset=True)
    if any(m >= 2**53 for _, m, _ in want):
        return
    got = run(df, chunk, kind="imbalance", quantity="q", threshold=theta)
    assert got["time_close"].to_list() == [i for i, _, _ in want]
    assert got["thresholds"].to_list() == [m for _, m, _ in want]
    starts = ([0] + [i + 1 for i, _, _ in want])[: len(want)]
    assert got["time_open"].to_list() == starts


def test_a_double_total_would_drift_and_the_bars_do_not() -> None:
    # 0.1 + 0.2 in doubles is 0.30000000000000004; the exact sum of the two
    # doubles is just below it, so no bar of that threshold closes.
    df = pl.DataFrame({"t": [0, 1], "q": [0.1, 0.2]})
    assert run(df, kind="sum", quantity="q", threshold=0.30000000000000004).height == 0
    assert run(df, kind="sum", quantity="q", threshold=0.3)["time_close"].to_list() == [1]


@pytest.mark.parametrize("kind", ["sum", "imbalance"])
def test_both_sides_of_the_threshold(kind: str) -> None:
    df = pl.DataFrame({"t": [0, 1, 2], "q": [4.0, 6.0, 1.0]})
    at = run(df, kind=kind, quantity="q", threshold=10.0)
    assert at["time_close"].to_list() == [1]
    above = run(df, kind=kind, quantity="q", threshold=math.nextafter(10.0, math.inf))
    assert above["time_close"].to_list() == [2]
    counted = run(df, kind="count", threshold=3)
    assert counted["time_close"].to_list() == [2]
    assert run(df, kind="count", threshold=4).height == 0


def test_a_row_larger_than_the_threshold_stays_whole() -> None:
    df = pl.DataFrame({"t": [1, 2, 3, 4, 5], "q": [4.0, 35.0, 1.0, 0.0, 10.0]})
    out = run(df, kind="sum", quantity="q", threshold=10.0, rows=pl.len(), total=pl.col("q").sum())
    # 4 + 35 = 39: three thresholds and 9 carried; then 9 + 1 = 10.
    assert out.select("time_open", "time_close", "thresholds", "carry", "rows", "total").rows() == [
        (1, 2, 3, 9.0, 2, 39.0),
        (3, 3, 1, 0.0, 1, 1.0),
        (4, 5, 1, 0.0, 2, 10.0),
    ]
    imb = run(df.with_columns(-pl.col("q")), kind="imbalance", quantity="q", threshold=10.0)
    assert imb.select("time_close", "thresholds").rows() == [(2, 3), (5, 1)]


def test_a_negative_quantity_is_refused_under_sum_naming_the_row() -> None:
    df = pl.DataFrame({"t": [0, 1, 2], "q": [1.0, 2.0, -0.5]})
    with pytest.raises(_REFUSAL, match="row 2"):
        run(df, kind="sum", quantity="q", threshold=10.0)


def test_missing_quantities_add_nothing_but_their_rows_count() -> None:
    df = pl.DataFrame(
        {
            "t": [0, 1, 2, 3, 4],
            "q": [5.0, None, float("nan"), float("inf"), 5.0],
            "p": [1.0, 7.0, 3.0, 2.0, 4.0],
        }
    )
    out = run(df, kind="sum", quantity="q", threshold=10.0, high=pl.col("p").max(), n=pl.len())
    assert out.select("time_close", "high", "n").rows() == [(4, 7.0, 5)]


@PROPERTY
@given(seed=st.integers(0, 10_000), chunks=st.lists(st.integers(1, 97), min_size=1, max_size=8))
def test_any_chunking_gives_the_same_bars_and_state_to_the_bit(
    seed: int, chunks: list[int]
) -> None:
    df = trades(seed, n=300, big=True).with_columns(dollars=pl.col("price") * pl.col("size") / 3.0)
    with tempfile.TemporaryDirectory() as d:
        kw = dict(
            kind="sum", quantity="dollars", threshold=1234.5, group="sym", final=False, **AGGS
        )
        one = run(df, save_state=os.path.join(d, "one"), **kw)
        for size in chunks:
            many = run(df, size, save_state=os.path.join(d, "many"), **kw)
            assert_frame_equal(one, many)
            with open(os.path.join(d, "one"), "rb") as a, open(os.path.join(d, "many"), "rb") as b:
                assert a.read() == b.read()


@pytest.mark.parametrize("kind", ["count", "sum", "imbalance"])
def test_a_saved_state_resumes_to_the_one_runs_bars_after_every_row(kind: str) -> None:
    df = trades(7, n=120, big=True).with_columns(signed=pl.col("size") * pl.col("side"))
    kw: dict[str, object] = {"kind": kind, "group": "sym", **AGGS}
    if kind != "count":
        kw["quantity"] = "size" if kind == "sum" else "signed"
    kw["threshold"] = 5 if kind == "count" else 400.0
    whole = run(df, **kw)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "state")
        for cut in range(0, df.height + 1, 7):
            first = run(df.head(cut), save_state=path, **kw)
            rest = run(df.slice(cut), load_state=path, **kw)
            assert_frame_equal(pl.concat([first, rest]), whole)


def test_a_slice_saves_the_state_after_the_row_of_its_last_bar() -> None:
    df = trades(8, n=150)
    kw: dict[str, object] = {
        "kind": "sum",
        "quantity": "size",
        "threshold": 300,
        "group": "sym",
        **AGGS,
    }
    whole = run(df, **kw)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "state")
        for n in range(1, whole.height + 1):
            for chunk in (1, 13, None):
                head = (
                    rp.stream.bars(df.lazy(), clock="t", chunk_size=chunk, save_state=path, **kw)  # type: ignore[arg-type]
                    .head(n)
                    .collect()
                )
                assert_frame_equal(head, whole.head(n))
                # The clock is the row's index: the rest starts after the
                # row that closed the n-th bar.
                after = int(head["time_close"][-1]) + 1
                rest = run(df.slice(after), load_state=path, **kw)
                assert_frame_equal(pl.concat([head, rest]), whole)


@PROPERTY
@given(seed=st.integers(0, 10_000), chunk=st.integers(1, 40))
def test_rows_of_weight_0_are_in_no_bar(seed: int, chunk: int) -> None:
    rng = random.Random(seed)
    df = trades(seed, n=200).with_columns(
        w=pl.Series([rng.choice([0, 1, 1]) for _ in range(200)]),
    )
    # A row of weight 0 holds values that would show if it were read.
    weighted = df.with_columns(
        price=pl.when(pl.col("w") == 0).then(1e6).otherwise(pl.col("price")),
        size=pl.when(pl.col("w") == 0).then(99_999).otherwise(pl.col("size")),
    )
    kw: dict[str, object] = {
        "kind": "sum",
        "quantity": "size",
        "threshold": 250,
        "group": "sym",
        **AGGS,
    }
    with_zeros = run(weighted, chunk, weight="w", **kw)
    without = run(df.filter(pl.col("w") == 1), chunk, **kw)
    assert_frame_equal(with_zeros, without)


def test_a_weight_other_than_0_or_1_is_refused() -> None:
    df = pl.DataFrame({"t": [0, 1], "q": [1.0, 2.0], "w": [1.0, 0.5]})
    with pytest.raises(_REFUSAL, match="row 1"):
        run(df, kind="sum", quantity="q", threshold=10.0, weight="w")


def test_final_emits_the_open_bars_and_saves_none_open() -> None:
    df = pl.DataFrame({"t": [0, 1, 2, 3], "g": ["a", "b", "a", "b"], "q": [6.0, 3.0, 6.0, 1.0]})
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "s")
        out = run(
            df, kind="sum", quantity="q", threshold=10.0, group="g", final=True, save_state=path
        )
        assert out.select(
            "g", "time_open", "time_close", "thresholds", "carry", "complete"
        ).rows() == [
            ("a", 0, 2, 1, 2.0, True),
            ("b", 1, 3, 0, 4.0, False),
        ]
        # Resumed: the open bar was emitted, so nothing of it is left.
        more = pl.DataFrame({"t": [4], "g": ["b"], "q": [9.0]})
        assert (
            run(more, kind="sum", quantity="q", threshold=10.0, group="g", load_state=path).height
            == 0
        )
    assert run(df, kind="sum", quantity="q", threshold=10.0, group="g").height == 1


def cut_run(df: pl.DataFrame, **kw: object) -> list[tuple[object, ...]]:
    out = run(df, kind="count", threshold=3, n=pl.len(), **kw)
    return out.select("time_open", "time_close", "complete", "n").rows()


def test_gap_cap_cuts_past_it_and_not_at_it() -> None:
    df = pl.DataFrame({"t": [0.0, 1.0, 3.0, 4.0, 5.0]})
    # A step of 2.0 is past a cap of 1.5, and not past one of 2.0.
    assert cut_run(df, gap_cap=1.5) == [(0.0, 1.0, False, 2), (3.0, 5.0, True, 3)]
    assert cut_run(df, gap_cap=2.0) == [(0.0, 3.0, True, 3)]
    assert cut_run(df, gap_cap=1.5, partial="drop") == [(3.0, 5.0, True, 3)]


def test_max_span_cuts_past_it_and_not_at_it() -> None:
    df = pl.DataFrame({"t": [0, 1, 2, 3, 4, 5]})
    # The last two rows are a bar still open: nothing after them cuts it.
    assert cut_run(df, max_span=1) == [(0, 1, False, 2), (2, 3, False, 2)]
    assert cut_run(df, max_span=2) == [(0, 2, True, 3), (3, 5, True, 3)]


def test_session_cuts_at_a_new_session_only() -> None:
    df = pl.DataFrame({"t": [0, 1, 2, 3, 4], "day": [1, 1, 2, 2, 2]})
    assert cut_run(df, session="day") == [(0, 1, False, 2), (2, 4, True, 3)]
    same = df.with_columns(day=pl.lit(1))
    assert cut_run(same, session="day") == [(0, 2, True, 3)]


def test_restart_after_step_back_cuts_past_it_and_refuses_at_it() -> None:
    df = pl.DataFrame({"t": [10, 11, 2, 3, 4]})
    assert cut_run(df, restart_after_step_back=8) == [(10, 11, False, 2), (2, 4, True, 3)]
    with pytest.raises(_REFUSAL, match="row 2"):
        cut_run(df, restart_after_step_back=9)
    with pytest.raises(_REFUSAL, match="row 2"):
        cut_run(df)


def test_a_cut_drops_the_carry() -> None:
    df = pl.DataFrame({"t": [0.0, 1.0, 5.0, 6.0], "q": [7.0, 7.0, 7.0, 7.0]})
    out = run(df, kind="sum", quantity="q", threshold=10.0, gap_cap=2.0)
    # 14: one bar and 4 carried; the gap drops the 4, so 7 + 7 closes next.
    assert out.select("time_close", "thresholds", "carry", "complete").rows() == [
        (1.0, 1, 4.0, True),
        (6.0, 1, 4.0, True),
    ]


def test_monotone_group_close_cuts_and_drops_each_group_before() -> None:
    df = pl.DataFrame({"t": [0, 1, 2, 3, 4], "day": [1, 1, 2, 2, 2]})
    out = run(df, kind="count", threshold=3, group="day", group_close="monotone", n=pl.len())
    assert out.select("day", "time_close", "complete", "n").rows() == [
        (1, 1, False, 2),
        (2, 4, True, 3),
    ]
    back = pl.DataFrame({"t": [0, 1, 2], "day": [2, 1, 1]})
    with pytest.raises(_REFUSAL, match="row 1"):
        run(back, kind="count", threshold=3, group="day", group_close="monotone")


def test_the_caps_on_memory() -> None:
    df = pl.DataFrame({"t": range(6), "g": ["a", "b", "c", "d", "e", "f"]})
    with pytest.raises(ValueError, match="max_state_bytes"):
        run(df, kind="count", threshold=2, group="g", max_state_bytes=64)
    # The bytes a group holds, as the refusal of the first one names them.
    with pytest.raises(_REFUSAL, match="max_total_bytes") as refused:
        run(df, kind="count", threshold=2, group="g", max_total_bytes=1)
    found = re.search(r"hold up to (\d+) bytes", str(refused.value))
    assert found is not None
    one = int(found.group(1))
    with pytest.raises(_REFUSAL, match="row 2"):
        run(df, kind="count", threshold=2, group="g", max_total_bytes=2 * one + one // 2)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run(df, kind="count", threshold=2, group="g", max_total_bytes=6 * one)
    assert [issubclass(w.category, ReadinessWarning) for w in caught] == [True]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        run(df, kind="count", threshold=2, group="g", max_total_bytes=12 * one)


@pytest.mark.parametrize(
    "dtype", [pl.Int8, pl.Int32, pl.Int64, pl.UInt16, pl.UInt64, pl.Float32, pl.Float64]
)
def test_aggregates_match_polars_dtypes_and_nulls(dtype: pl.DataType) -> None:
    rng = random.Random(5)
    n = 200
    df = pl.DataFrame(
        {
            "t": range(n),
            "x": pl.Series([rng.choice([None, rng.randint(0, 100)]) for _ in range(n)]).cast(dtype),
            "b": [rng.choice([None, True, False]) for _ in range(n)],
            "d": pl.Series([rng.randint(0, 10**6) for _ in range(n)]).cast(pl.Datetime("us")),
        }
    )
    aggs = {
        "first": pl.col("x").first(),
        "last": pl.col("x").last(),
        "min": pl.col("x").min(),
        "max": pl.col("x").max(),
        "sum": pl.col("x").sum(),
        "count": pl.col("x").count(),
        "bsum": pl.col("b").sum(),
        "bmax": pl.col("b").max(),
        "dmin": pl.col("d").min(),
        "dlast": pl.col("d").last(),
    }
    got = run(df, kind="count", threshold=9, **aggs).drop(
        "bar", "time_open", "time_close", "thresholds", "complete"
    )
    want = (
        df.with_columns(_bar=pl.int_range(pl.len()) // 9)
        .group_by("_bar", maintain_order=True)
        .agg(**aggs, _n=pl.len())
        .filter(pl.col("_n") == 9)
        .drop("_bar", "_n")
    )
    assert_frame_equal(got, want)


def test_float_aggregates_with_nan_and_infinities_are_polars() -> None:
    df = pl.DataFrame(
        {
            "t": range(8),
            "x": [
                1.0,
                float("nan"),
                0.5,
                None,
                float("nan"),
                float("nan"),
                float("inf"),
                -float("inf"),
            ],
        }
    )
    aggs = {"mn": pl.col("x").min(), "mx": pl.col("x").max(), "s": pl.col("x").sum()}
    got = run(df, kind="count", threshold=2, **aggs).select(*aggs)
    want = (
        df.with_columns(_bar=pl.int_range(pl.len()) // 2)
        .group_by("_bar", maintain_order=True)
        .agg(**aggs)
        .drop("_bar")
    )
    assert_frame_equal(got, want)


def test_a_float_sum_is_rounded_once_as_fsum() -> None:
    rng = random.Random(11)
    xs = [rng.uniform(-1e6, 1e6) * 10 ** rng.randint(-8, 8) for _ in range(400)]
    df = pl.DataFrame({"t": range(400), "x": xs})
    got = run(df, kind="count", threshold=50, s=pl.col("x").sum())["s"].to_list()
    assert got == [math.fsum(xs[i : i + 50]) for i in range(0, 400, 50)]


def test_refusals_at_the_call() -> None:
    df = pl.DataFrame({"t": [0, 1], "q": [1.0, 2.0]})
    with pytest.raises(ValueError, match="kind"):
        run(df, kind="volume", threshold=1.0)
    with pytest.raises(ValueError, match="takes no quantity"):
        run(df, kind="count", quantity="q", threshold=1)
    with pytest.raises(ValueError, match="none is given"):
        run(df, kind="sum", threshold=1.0)
    with pytest.raises(ValueError, match="whole number"):
        run(df, kind="count", threshold=1.5)
    with pytest.raises(ValueError, match="above 0"):
        run(df, kind="sum", quantity="q", threshold=0.0)
    with pytest.raises(ColumnNotFoundError):
        run(df, kind="sum", quantity="nope", threshold=1.0)
    with pytest.raises(ValueError, match="aggregation"):
        run(df, kind="count", threshold=1, m=pl.col("q").mean())
    with pytest.raises(ValueError, match="no aggregation"):
        run(df, kind="count", threshold=1, m=pl.col("q") + 1)
    with pytest.raises(ValueError, match="own column"):
        run(df, kind="count", threshold=1, bar=pl.len())


def test_a_state_of_another_call_is_refused() -> None:
    df = pl.DataFrame({"t": [0, 1], "q": [1.0, 2.0]})
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "s")
        run(df, kind="sum", quantity="q", threshold=10.0, save_state=path)
        with pytest.raises(ValueError, match="threshold"):
            run(df, kind="sum", quantity="q", threshold=11.0, load_state=path)
        with pytest.raises(ValueError, match="kind"):
            run(df, kind="imbalance", quantity="q", threshold=10.0, load_state=path)


def test_a_threshold_column_is_read_at_each_row() -> None:
    df = pl.DataFrame({"t": [0, 1, 2, 3], "q": [6.0, 0.0, 3.0, 3.0], "th": [10.0, 5.0, 4.0, None]})
    out = run(df.head(3), kind="sum", quantity="q", threshold="th")
    assert out.select("time_close", "thresholds", "carry").rows() == [(1, 1, 1.0), (2, 1, 0.0)]
    with pytest.raises(_REFUSAL, match="row 3"):
        run(df, kind="sum", quantity="q", threshold="th")


# --- bar-forge, the third-party library that computes the same bars --------


def forge_stream(seed: int) -> tuple[pl.DataFrame, list[bar_forge.Trade]]:
    df = trades(seed, n=800, groups=1, big=True).with_columns(pl.col("size").cast(pl.Float64))
    rows = df.select("t", "price", "size").rows()
    return df, [bar_forge.Trade(float(t), p, s) for t, p, s in rows]


def forge_frame(bars: list[bar_forge.Bar]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "time_open": [float(b.start_time) for b in bars],
            "time_close": [float(b.end_time) for b in bars],
            "open": [b.open for b in bars],
            "high": [b.high for b in bars],
            "low": [b.low for b in bars],
            "close": [b.close for b in bars],
            "volume": [b.volume for b in bars],
            "notional": [b.notional for b in bars],
            "n": pl.Series([b.trade_count for b in bars], dtype=pl.UInt32),
        }
    )


FORGE_AGGS = {
    "open": pl.col("price").first(),
    "high": pl.col("price").max(),
    "low": pl.col("price").min(),
    "close": pl.col("price").last(),
    "volume": pl.col("size").sum(),
    "notional": (pl.col("price") * pl.col("size")).sum(),
    "n": pl.len(),
}


def ours(df: pl.DataFrame, **kw: object) -> pl.DataFrame:
    out = run(df.with_columns(pl.col("t").cast(pl.Float64)), **kw, **FORGE_AGGS)
    return out.select("time_open", "time_close", *FORGE_AGGS).with_columns(
        pl.col("n").cast(pl.UInt32)
    )


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_bar_forge_tick_bars_are_count_bars(seed: int) -> None:
    df, ts = forge_stream(seed)
    assert_frame_equal(
        ours(df, kind="count", threshold=25), forge_frame(bar_forge.tick_bars(ts, 25))
    )


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_bar_forge_volume_and_dollar_bars_are_imbalance_of_a_positive_quantity(seed: int) -> None:
    df, ts = forge_stream(seed)
    assert_frame_equal(
        ours(df, kind="imbalance", quantity="size", threshold=500.0),
        forge_frame(bar_forge.volume_bars(ts, 500.0)),
    )
    assert_frame_equal(
        ours(df, kind="imbalance", quantity=pl.col("price") * pl.col("size"), threshold=7500.0),
        forge_frame(bar_forge.dollar_bars(ts, 7500.0)),
    )


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_bar_forge_imbalance_bars_over_its_tick_rule_signs(seed: int) -> None:
    df, ts = forge_stream(seed)
    signed = df.with_columns(sign=pl.Series(bar_forge.tick_rule_signs(ts), dtype=pl.Float64))
    assert_frame_equal(
        ours(signed, kind="imbalance", quantity=pl.col("sign") * pl.col("size"), threshold=300.0),
        forge_frame(bar_forge.volume_imbalance_bars(ts, 300.0)),
    )
    assert_frame_equal(
        ours(signed, kind="imbalance", quantity="sign", threshold=7.0),
        forge_frame(bar_forge.tick_imbalance_bars(ts, 7.0)),
    )
