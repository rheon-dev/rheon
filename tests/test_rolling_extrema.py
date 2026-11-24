"""``rp.ops.rolling_max`` and ``rp.ops.rolling_min``: the largest and smallest value
of a window on the stream's clock, and since an event (task 561).

Held to Polars itself, the library that has the operator: a backward window
to ``rolling_max_by`` / ``rolling_min_by`` bit for bit, under every
``closed``, on integer, float-valued and datetime clocks with repeated
stamps; a forward window to the same over the reversed stream; the running
form with ``since`` to ``rolling_*_by`` over each run, with a window longer
than the run. numpy's ``fmax.accumulate`` over each run is a second
opinion. The rest is the stream's own rules: chunks of 1 and of the whole
input give the same bits, a save and a load at any row is one run, NaN and
infinities are read as missing (Polars' rolling functions let a NaN make the
window NaN and count an infinity), and the bound on a window's bytes refuses
on both sides of it.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import numpy as np
import polars as pl
import pytest

from polars_version import INSTALLED, needs_polars
from rheon import polars as rp

TIER = "essential"

CLOSED = ("right", "left", "both", "none")
MIRROR = {"right": "left", "left": "right", "both": "both", "none": "none"}


def bits(s: pl.Series) -> list[Any]:
    """A column as the bits of each value, null as None, so ``-0.0`` and
    ``0.0`` differ."""
    return [None if v is None else np.float64(v).view(np.uint64).item() for v in s.to_list()]


def frame(n: int, seed: int, *, nulls: bool = False, zeros: bool = False) -> pl.DataFrame:
    """An integer clock with repeated stamps (a third of the rows), values
    over a few levels, with ``zeros`` both signed zeros and with ``nulls``
    nulls."""
    rng = np.random.default_rng(seed)
    t = np.cumsum(rng.choice([0, 0, 1, 1, 2, 3], n))
    if zeros:
        x: list[float | None] = list(rng.choice([0.0, -0.0, 1.0, -1.0], n))
    else:
        x = list(np.round(rng.standard_normal(n) * 4, 1))
    if nulls:
        x = [None if rng.random() < 0.15 else v for v in x]
    return pl.DataFrame({"t": t.astype(np.int64), "x": pl.Series(x, dtype=pl.Float64)})


def polars_extremum(df: pl.DataFrame, fn: str, by: str, w: Any, closed: str) -> pl.Series:
    expr = getattr(pl.col("x"), f"{fn}_by")(by, window_size=w, closed=closed)
    return df.select(expr.alias("p"))["p"]


def rheon_extremum(
    df: pl.DataFrame, fn: str, clock: str, gap_cap: Any = 1e12, **kw: Any
) -> pl.Series:
    op = getattr(rp.ops, fn)("x", **kw)
    return rp.stream.with_windows(df, out=op, clock=clock, gap_cap=gap_cap)["out"]


@pytest.mark.parametrize("closed", CLOSED)
@pytest.mark.parametrize("fn", ["rolling_max", "rolling_min"])
def test_a_backward_window_is_polars_rolling_by(fn: str, closed: str) -> None:
    for seed in range(4):
        df = frame(300, seed)
        for w in (1, 2, 5, 40):
            want = polars_extremum(df, fn, "t", f"{w}i", closed)
            got = rheon_extremum(df, fn, "t", window_size=w, closed=closed)
            assert bits(got) == bits(want), (seed, w)


def test_the_400_frames_of_both_zeros_are_polars_to_the_bit() -> None:
    """Values from {0.0, -0.0, 1.0, -1.0}, repeated stamps, windows of 1 to
    4: the earliest row of a window wins a tie, for max and min alike."""
    for seed in range(400):
        df = frame(int(np.random.default_rng(seed).integers(5, 40)), 1000 + seed, zeros=True)
        w = 1 + seed % 4
        for closed in ("right", "left"):
            for fn in ("rolling_max", "rolling_min"):
                want = polars_extremum(df, fn, "t", f"{w}i", closed)
                got = rheon_extremum(df, fn, "t", window_size=w, closed=closed)
                assert bits(got) == bits(want), (seed, closed, fn)


@needs_polars("1.38.1", "rolling_*_by refuses a series with nulls before 1.38.1")
def test_a_null_counts_for_nothing_and_gets_the_window_as_polars_gives_it() -> None:
    for seed in range(100):
        df = frame(40, 2000 + seed, nulls=True, zeros=seed % 2 == 0)
        w = 1 + seed % 5
        for closed in CLOSED:
            for fn in ("rolling_max", "rolling_min"):
                want = polars_extremum(df, fn, "t", f"{w}i", closed)
                got = rheon_extremum(df, fn, "t", window_size=w, closed=closed)
                assert bits(got) == bits(want), (seed, closed, fn)


def test_a_datetime_clock_and_a_duration_window_are_polars_rolling_by() -> None:
    df = frame(400, 9)
    base = datetime(2024, 1, 2)
    df = df.with_columns(
        ts=pl.Series([base + timedelta(milliseconds=250 * int(v)) for v in df["t"]])
    )
    for closed in CLOSED:
        for fn in ("rolling_max", "rolling_min"):
            want = polars_extremum(df, fn, "ts", "750ms", closed)
            got = rheon_extremum(df, fn, "ts", gap_cap="1000d", window_size="750ms", closed=closed)
            assert bits(got) == bits(want), (closed, fn)


def test_nan_and_infinity_are_missing_where_polars_lets_them_in() -> None:
    """The difference from Polars, on both sides: on an input holding NaN and
    infinities rheon gives what Polars gives on the input with them nulled,
    and not what Polars gives on the input as it is."""
    df = pl.DataFrame(
        {
            "t": np.arange(8, dtype=np.int64),
            "x": [1.0, float("nan"), 2.0, float("inf"), 0.5, float("-inf"), 3.0, 1e101],
        }
    )
    nulled = df.with_columns(
        x=pl.when(pl.col("x").is_finite() & (pl.col("x").abs() <= 1e100)).then(pl.col("x"))
    )
    for fn in ("rolling_max", "rolling_min"):
        got = rheon_extremum(df, fn, "t", window_size=3)
        raw = polars_extremum(df, fn, "t", "3i", "right")
        assert bits(got) != bits(raw)
        if INSTALLED >= (1, 38, 1):
            assert bits(got) == bits(polars_extremum(nulled, fn, "t", "3i", "right"))
    assert rheon_extremum(df, "rolling_max", "t", window_size=3).to_list()[3] == 2.0


@pytest.mark.parametrize("closed", CLOSED)
def test_a_forward_window_is_polars_over_the_reversed_stream(closed: str) -> None:
    for seed in range(3):
        df = frame(200, 50 + seed, zeros=seed == 0)
        rev = df.reverse().with_columns(t=-pl.col("t"))
        for fn in ("rolling_max", "rolling_min"):
            got = rheon_extremum(
                df, fn, "t", window_size=3, closed=closed, reverse=True, partial="keep"
            )
            want = polars_extremum(rev, fn, "t", "3i", MIRROR[closed]).reverse()
            # The windows still open at the input's end are null, and kept
            # ones are Polars'.
            closed_by_end = (df["t"] + 3 < df["t"].max()).to_list()
            g, p = bits(got), bits(want)
            for i, (a, b) in enumerate(zip(g, p, strict=True)):
                if closed_by_end[i]:
                    assert a == b, (seed, fn, i)


def runs(df: pl.DataFrame) -> pl.DataFrame:
    """Each row's run: a stamp holding an event, at any of its rows, starts
    one."""
    ev = pl.col("e").any().over("t")
    first = pl.col("t") != pl.col("t").shift(1)
    return df.with_columns(run=(ev & first.fill_null(True)).cast(pl.Int64).cum_sum())


@pytest.mark.parametrize("closed", CLOSED)
def test_the_running_form_since_an_event_is_polars_over_each_run(closed: str) -> None:
    for seed in range(4):
        df = frame(300, 70 + seed, zeros=seed % 2 == 1)
        rng = np.random.default_rng(seed)
        df = runs(df.with_columns(e=pl.Series(rng.random(len(df)) < 0.08)))
        for fn in ("rolling_max", "rolling_min"):
            got = rheon_extremum(df, fn, "t", since="e", closed=closed)
            expr = getattr(pl.col("x"), f"{fn}_by")("t", window_size="100000i", closed=closed)
            want = df.select(expr.over("run").alias("p"))["p"]
            assert bits(got) == bits(want), (seed, fn)


def test_the_running_maximum_is_numpys_over_each_run() -> None:
    """A second opinion: numpy's ``fmax.accumulate`` over each run's values
    (a missing one as ``-inf``), read at each stamp's last row, since the
    operator gives every row of a stamp the stamp's value."""
    df = frame(500, 3)
    rng = np.random.default_rng(3)
    df = runs(df.with_columns(e=pl.Series(rng.random(len(df)) < 0.05)))
    got = rheon_extremum(df, "rolling_max", "t", since="e").to_numpy()
    x, run, t = df["x"].to_numpy(), df["run"].to_numpy(), df["t"].to_numpy()
    for r in np.unique(run):
        idx = np.flatnonzero(run == r)
        acc = np.fmax.accumulate(x[idx])
        last = np.r_[t[idx][1:] != t[idx][:-1], True]
        np.testing.assert_array_equal(got[idx][last], acc[last])


def mixed() -> dict[str, pl.Expr]:
    return {
        "back": rp.ops.rolling_max("x", window_size=4.0),
        "left": rp.ops.rolling_min("x", window_size=3.0, closed="left", partial="null"),
        "ahead": rp.ops.rolling_max("x", window_size=2.5, reverse=True),
        "run": rp.ops.rolling_max("x", since="e"),
        "run_left": rp.ops.rolling_min("x", since="e", closed="left"),
        "win_since": rp.ops.rolling_min("x", window_size=5.0, since="e", partial="null"),
        "dd": pl.col("x") - rp.ops.rolling_max("x", since="e"),
    }


def grouped(n: int, seed: int) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.cumsum(rng.choice([0.0, 0.5, 1.0, 3.0, 30.0], n, p=[0.3, 0.3, 0.3, 0.09, 0.01]))
    x = np.round(rng.standard_normal(n) * 3, 1)
    x[rng.random(n) < 0.1] = np.nan
    return pl.DataFrame(
        {
            "t": t,
            "x": x,
            "e": rng.random(n) < 0.05,
            "g": [f"g{k}" for k in rng.integers(0, 3, n)],
        }
    ).with_columns(pl.col("x").fill_nan(None))


@pytest.mark.parametrize("rows", [1, 7, 600])
def test_chunking_changes_no_bit(rows: int) -> None:
    df = grouped(1500, 5)
    kw: dict[str, Any] = {"clock": "t", "gap_cap": 20.0, "group": "g"}
    one = rp.stream.with_windows(df, **mixed(), **kw, chunk_size=100_000)
    out = rp.stream.with_windows(df, **mixed(), **kw, chunk_size=rows)
    assert out.equals(one)
    for c in mixed():
        assert bits(out[c]) == bits(one[c]), c
        assert one[c].null_count() < 0.6 * len(one), c


def test_a_save_and_load_at_any_row_is_one_run(tmp_path: Any) -> None:
    df = grouped(300, 6)
    kw: dict[str, Any] = {"clock": "t", "gap_cap": 20.0, "group": "g"}
    one = rp.stream.with_windows(df, **mixed(), **kw)
    # Cut between two stamps, as a resume needs: the first such row at or
    # after each place.
    t = df["t"].to_list()
    cuts = sorted(
        {next(i for i in range(at, len(t)) if t[i] != t[i - 1]) for at in (1, 2, 37, 151, 290)}
    )
    assert len(cuts) == 5
    for at in cuts:
        state = tmp_path / f"x{at}.state"
        first = rp.stream.with_windows(df[:at], **mixed(), **kw, save_state=state)
        second = rp.stream.with_windows(df[at:], **mixed(), **kw, load_state=state)
        both = pl.concat([first, second])
        for c in mixed():
            assert bits(both[c]) == bits(one[c]), (at, c)


def test_a_gap_past_the_cap_starts_the_running_form_over() -> None:
    df = pl.DataFrame({"t": [0.0, 1.0, 100.0, 101.0], "x": [5.0, 1.0, 2.0, 3.0]})
    out = rp.stream.with_windows(df, m=rp.ops.rolling_max("x"), clock="t", gap_cap=10.0)
    assert out["m"].to_list() == [5.0, 5.0, 2.0, 3.0]
    wide = rp.stream.with_windows(df, m=rp.ops.rolling_max("x"), clock="t", gap_cap=1000.0)
    assert wide["m"].to_list() == [5.0, 5.0, 5.0, 5.0]


def test_since_is_each_groups_own() -> None:
    df = pl.DataFrame(
        {
            "t": [0.0, 0.0, 1.0, 1.0, 2.0, 2.0],
            "g": ["a", "b", "a", "b", "a", "b"],
            "x": [9.0, 9.0, 1.0, 1.0, 2.0, 2.0],
            "e": [False, False, True, False, False, False],
        }
    )
    out = rp.stream.with_windows(
        df, m=rp.ops.rolling_max("x", since="e"), clock="t", gap_cap=10.0, group="g"
    )
    assert out["m"].to_list() == [9.0, 9.0, 1.0, 9.0, 2.0, 9.0]


def test_the_row_clock_bound_refuses_on_both_sides_of_the_cap() -> None:
    """Without a clock column a window of ``w`` rows holds at most ``w + 2``
    rows of 112 bytes (5 shared doubles, 8 for the operator, 1 for its
    extremum), so a cap of exactly that is taken and one byte less refused,
    when the query is built."""
    df = pl.DataFrame({"x": np.arange(50, dtype=float)})
    w, row = 10, 8 * (5 + 8 + 1)
    need = (w + 2) * row
    ok = rp.stream.with_windows(df, m=rp.ops.rolling_max("x", window_size=w, max_state_bytes=need))
    assert ok["m"].to_list()[-1] == 49.0
    with pytest.raises(
        Exception,
        match=r"holds up to 12 rows of 112 bytes each a group, 1344 bytes, past "
        r"max_state_bytes of 1343",
    ):
        rp.stream.with_windows(
            df, m=rp.ops.rolling_max("x", window_size=w, max_state_bytes=need - 1)
        )


def test_a_burst_past_the_cap_on_a_clock_is_refused_at_its_row() -> None:
    row = 8 * (5 + 8 + 1)
    burst = pl.DataFrame({"t": [0.0] * 20, "x": np.arange(20, dtype=float)})
    op = rp.ops.rolling_max("x", window_size=1.0, max_state_bytes=10 * row)
    with pytest.raises(Exception, match=r"at row 10 \(counting from 0\).*holds 11 rows"):
        rp.stream.with_windows(burst, m=op, clock="t", gap_cap=5.0)
    ten = rp.stream.with_windows(burst[:10], m=op, clock="t", gap_cap=5.0)
    assert ten["m"].to_list() == [9.0] * 10


def test_the_parameters_are_checked_by_name() -> None:
    with pytest.raises(ValueError, match="since cannot be combined with reverse=True yet"):
        rp.ops.rolling_max("x", window_size=2.0, reverse=True, since="e")
    with pytest.raises(ValueError, match="reverse=True needs window_size"):
        rp.ops.rolling_min("x", reverse=True)
    with pytest.raises(ValueError, match="max_state_bytes must be an integer above 0"):
        rp.ops.rolling_max("x", window_size=2.0, max_state_bytes=0)
    with pytest.raises(ValueError, match="at most 67108864"):
        rp.ops.rolling_max("x", window_size=2.0, max_state_bytes=(64 << 20) + 1)
    with pytest.raises(TypeError):
        rp.ops.rolling_max("x", window_size=2.0, max_state_bytes=1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="partial needs window_size"):
        rp.ops.rolling_max("x", partial="null")
    with pytest.raises(ValueError, match="holds another window operator"):
        rp.ops.rolling_max("x", since=rp.ops.ewm_mean("y", half_life=1.0) > 0)
    df = pl.DataFrame({"t": [0.0, 1.0], "x": [1.0, 2.0], "e": [1, 0]})
    with pytest.raises(Exception, match="since must be a boolean"):
        rp.stream.with_windows(df, m=rp.ops.rolling_max("x", since="e"), clock="t", gap_cap=5.0)


def test_drawdown_is_the_formula_around_the_peak() -> None:
    df = pl.DataFrame(
        {
            "t": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "equity": [1.0, 3.0, 2.0, 4.0, 1.0, 2.0],
            "new_period": [False, False, False, False, True, False],
        }
    )
    peak = rp.ops.rolling_max("equity", since="new_period")
    out = rp.stream.with_windows(df, peak=peak, dd=pl.col("equity") - peak, clock="t", gap_cap=10.0)
    assert out["peak"].to_list() == [1.0, 3.0, 3.0, 4.0, 1.0, 2.0]
    assert out["dd"].to_list() == [0.0, 0.0, -1.0, 0.0, 0.0, 0.0]


@pytest.mark.xfail(
    strict=True,
    reason=(
        "rule 9 for formula targets: a spec's window core reads every row of the frame, a row "
        "of weight 0 included, so its value moves a target's extremum and it counts a unit on "
        "the row clock. Recorded, not fixed here: the rule belongs to the engine, for every "
        "operator at once."
    ),
)
def test_a_weight_zero_row_moves_no_formula_targets_extremum() -> None:
    """A row of weight 0 learns nothing and counts in no clock (rule 9): the
    targets of the rows after it, and so every prediction, should be those
    of the frame with its value changed."""
    rng = np.random.default_rng(4)
    n = 200
    x0 = rng.standard_normal(n)
    y = x0 + rng.standard_normal(n)
    w = np.ones(n)
    w[100] = 0.0
    df = pl.DataFrame({"x0": x0, "y": y, "w": w})
    big = df.with_columns(y=pl.when(pl.int_range(n) == 100).then(1e6).otherwise(pl.col("y")))
    target = rp.ops.rolling_max("y", window_size=5.0).alias("hi")

    def preds(frame: pl.DataFrame) -> list[Any]:
        spec = rp.spec.ewridge("m", targets=[target], features=["x0"], half_life=50.0, weight="w")
        return rp.ModelBank([spec]).fit_predict(frame)["m"].struct.field("pred_hi").to_list()

    assert preds(big) == preds(df)
