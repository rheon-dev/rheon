"""``rp.ops.clock_step`` (task 564): a decayed statistic of the steps between
each time and the one before it, a stream's cadence read as a column.

On a stream with no clock column every row is one unit of the clock, and the
mean, variance and standard deviation of the steps are Polars' ``ewm_mean``,
``ewm_var`` and ``ewm_std`` (``adjust=True``, nulls decaying by their place)
of ``diff()``. On a clock they are held to statsmodels' ``DescrStatsW`` in
``test_second_opinion_clock_health.py``. A count is the weight of the steps
of one sign: numpy sums the decayed weights written from the definition.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import numpy as np
import polars as pl
import pytest

from polars_version import INSTALLED
from rheon import polars as rp

TIER = "essential"

INF = float("inf")


def _times(n: int, seed: int) -> pl.Series:
    """A time on uneven steps that repeats (a step of 0), steps back now and
    then, and has nulls."""
    rng = np.random.default_rng(seed)
    steps = rng.choice([0.0, 0.25, 1.0, 3.5, -2.0], size=n, p=[0.2, 0.3, 0.3, 0.1, 0.1])
    t: list[float | None] = list(np.cumsum(steps) + 1e6)
    for i in rng.choice(n, size=n // 25, replace=False):
        t[i] = None
    return pl.Series("t", t, dtype=pl.Float64)


def _polars(s: pl.Series, stat: str, half_life: float, bias: bool) -> list[float | None]:
    """Polars' own decayed statistic of the steps, where a step is formed."""
    d = s.diff()
    kw: dict[str, Any] = {"half_life": half_life, "adjust": True, "ignore_nulls": False}
    if stat == "mean":
        r = d.ewm_mean(**kw)
    elif stat == "var":
        r = d.ewm_var(**kw, bias=bias)
    else:
        r = d.ewm_std(**kw, bias=bias)
    return [None if x is None else v for x, v in zip(d.to_list(), r.to_list(), strict=True)]


@pytest.mark.parametrize("stat", ["mean", "var", "std"])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("half_life", [0.7, 5.0, 40.0])
def test_the_steps_statistics_are_polars_ewm_of_diff(
    stat: str, bias: bool, half_life: float
) -> None:
    if stat == "mean" and not bias:
        pytest.skip("a mean takes no bias")
    s = _times(400, 11)
    op = (
        rp.ops.clock_step("t", half_life=half_life)
        if stat == "mean"
        else rp.ops.clock_step("t", half_life=half_life, stat=stat, bias=bias)  # type: ignore[arg-type]
    )
    got = rp.stream.with_windows(s.to_frame(), v=op)["v"].to_list()
    want = _polars(s, stat, half_life, bias)
    formed = s.diff().is_not_null().to_list()
    first_formed = formed.index(True)
    compared = 0
    for i, (g, w) in enumerate(zip(got, want, strict=True)):
        if s[i] is None:
            assert g is None, i
        elif not formed[i]:
            continue  # no step here: Polars gives null, the operator what stands
        elif i == first_formed and not bias and stat != "mean" and INSTALLED < (1, 44, 1):
            # Polars' ewm_var/ewm_std(bias=False) returned 0.0 instead of null
            # on the first non-null observation before 1.44 (tests/test_windows.py).
            assert g is None, i
        elif w is None or np.isnan(w):
            assert g is None, i  # both withhold: one step carries the weight
        else:
            assert g == pytest.approx(w, rel=1e-9, abs=1e-12), i
            compared += 1
    # Polars withholds a corrected spread on a step of its own alone.
    assert compared >= sum(formed) - (0 if bias or stat == "mean" else 1)


def test_a_count_is_the_decayed_weight_of_its_steps() -> None:
    s = _times(300, 5)
    h = 6.0
    out = rp.stream.with_windows(
        s.to_frame(),
        zeros=rp.ops.clock_step("t", half_life=h, stat="zeros"),
        backs=rp.ops.clock_step("t", half_life=h, stat="backs"),
    )
    d = s.diff().to_numpy()
    for i in range(len(s)):
        if s[i] is None:
            assert out["zeros"][i] is None and out["backs"][i] is None
            continue
        w = np.exp2(-(i - np.arange(i + 1)) / h)
        step = d[: i + 1]
        formed = ~np.isnan(step)
        zeros = float(np.sum(w[formed & (np.nan_to_num(step) == 0.0)]))
        backs = float(np.sum(w[formed & (np.nan_to_num(step) < 0.0)]))
        assert out["zeros"][i] == pytest.approx(zeros, rel=1e-12, abs=1e-300), i
        assert out["backs"][i] == pytest.approx(backs, rel=1e-12, abs=1e-300), i


def test_a_count_is_by_sign_on_both_sides_of_zero() -> None:
    # A step of exactly 0, of -0.0, and the smallest either side of it.
    tiny = 5e-324
    s = pl.Series("t", [0.0, 0.0, -0.0, tiny, 0.0, 1.0], dtype=pl.Float64)
    out = rp.stream.with_windows(
        s.to_frame(),
        zeros=rp.ops.clock_step("t", half_life=INF, stat="zeros"),
        backs=rp.ops.clock_step("t", half_life=INF, stat="backs"),
    )
    assert out["zeros"].to_list() == [0.0, 1.0, 2.0, 2.0, 2.0, 2.0]
    assert out["backs"].to_list() == [0.0, 0.0, 0.0, 0.0, 1.0, 1.0]


def test_a_missing_time_reports_null_and_restarts_the_chain() -> None:
    # A null and a NaN: each gives null, and the next time takes no step
    # across it, so the mean of the steps 1 and 1 holds until the step 4.
    s = pl.Series("t", [0.0, 1.0, None, 9.0, 10.0, float("nan"), 50.0, 54.0])
    out = rp.stream.with_windows(s.to_frame(), m=rp.ops.clock_step("t", half_life=INF))
    assert out["m"].to_list() == [None, 1.0, None, 1.0, 1.0, None, 1.0, 2.0]


def test_a_spread_is_withheld_while_one_step_carries_the_weight() -> None:
    # Both sides: one step has a biased spread of 0 and no corrected one; a
    # second step of weight gives one.
    s = pl.Series("t", [0.0, 2.0, 3.0], dtype=pl.Float64)
    out = rp.stream.with_windows(
        s.to_frame(),
        b=rp.ops.clock_step("t", half_life=INF, stat="var", bias=True),
        u=rp.ops.clock_step("t", half_life=INF, stat="var"),
    )
    assert out["b"].to_list() == [None, 0.0, 0.25]
    assert out["u"].to_list() == [None, None, 0.5]


def test_the_steps_are_each_groups_and_restart_at_a_gap_and_a_session() -> None:
    df = pl.DataFrame(
        {
            "t": [0.0, 1.0, 2.0, 4.0, 15.0, 16.0, 17.0, 19.0],
            "g": ["a", "b", "a", "a", "a", "b", "a", "a"],
            "s": [1, 1, 1, 1, 1, 1, 2, 2],
        }
    )
    out = rp.stream.with_windows(
        df,
        m=rp.ops.clock_step("t", half_life=INF),
        clock="t",
        gap_cap=12.0,
        group="g",
        session="s",
    )
    # a: steps 2, 2 and 11, within the cap; the session change starts again;
    # then a step of 2. b: a step of 15, past the cap, starts again.
    assert out["m"].to_list() == [None, None, 2.0, 2.0, 5.0, None, None, 2.0]
    # A step of exactly gap_cap keeps the steps.
    out = rp.stream.with_windows(
        pl.DataFrame({"t": [0.0, 1.0, 11.0]}),
        m=rp.ops.clock_step("t", half_life=INF),
        clock="t",
        gap_cap=10.0,
    )
    assert out["m"].to_list() == [None, 1.0, 5.5]


def test_a_temporal_time_steps_in_seconds() -> None:
    start = datetime(2026, 3, 1, 9, 0)
    ts = [start + timedelta(microseconds=m) for m in (0, 1, 500_001, 500_001)]
    df = pl.DataFrame({"ts": ts}).with_columns(pl.col("ts").dt.cast_time_unit("ns"))
    out = rp.stream.with_windows(
        df,
        m=rp.ops.clock_step("ts", half_life=INF),
        z=rp.ops.clock_step("ts", half_life=INF, stat="zeros"),
        clock="ts",
        gap_cap="1h",
    )
    m = out["m"].to_list()
    assert m[0] is None
    assert m[1:] == pytest.approx([1e-6, 0.500001 / 2, 0.500001 / 3], rel=1e-12)
    assert out["z"].to_list() == [0.0, 0.0, 0.0, 1.0]


@pytest.mark.parametrize("chunk_size", [1, 3, 64])
def test_chunking_changes_no_bit(chunk_size: int) -> None:
    s = _times(500, 3)
    df = pl.DataFrame({"c": np.arange(500.0) * 0.5, "t": s, "g": ["a", "b"] * 250})
    ops = {
        "m": rp.ops.clock_step("t", half_life=4.0),
        "sd": rp.ops.clock_step("t", half_life=4.0, stat="std"),
        "z": rp.ops.clock_step("t", half_life=4.0, stat="zeros"),
        "b": rp.ops.clock_step("t", half_life=4.0, stat="backs"),
    }
    kw: dict[str, Any] = {"clock": "c", "gap_cap": 10.0, "group": "g"}
    one = rp.stream.with_windows(df, **ops, **kw)
    many = rp.stream.with_windows(df, **ops, **kw, chunk_size=chunk_size)
    assert many.equals(one)


def test_a_run_is_saved_and_resumed_at_every_row(tmp_path: Any) -> None:
    s = _times(60, 9)
    df = pl.DataFrame({"c": np.arange(60.0), "t": s})
    ops = {
        "m": rp.ops.clock_step("t", half_life=5.0),
        "v": rp.ops.clock_step("t", half_life=5.0, stat="var", bias=True),
        "z": rp.ops.clock_step("t", half_life=5.0, stat="zeros"),
    }
    one = rp.stream.with_windows(df, **ops, clock="c", gap_cap=10.0)
    for at in range(1, 60):
        state = tmp_path / f"w{at}.state"
        first = rp.stream.with_windows(df[:at], **ops, clock="c", gap_cap=10.0, save_state=state)
        second = rp.stream.with_windows(df[at:], **ops, clock="c", gap_cap=10.0, load_state=state)
        assert pl.concat([first, second]).equals(one), at


def test_what_a_clock_step_refuses() -> None:
    df = pl.DataFrame({"t": [1.0, 2.0], "s": ["a", "b"]})
    with pytest.raises(TypeError, match="half_life is required"):
        rp.ops.clock_step("t", half_life=None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="half_life must be above 0"):
        rp.ops.clock_step("t", half_life=0.0)
    with pytest.raises(ValueError, match='stat must be "mean"'):
        rp.ops.clock_step("t", half_life=1.0, stat="median")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match='only stat="var" or "std" takes it'):
        rp.ops.clock_step("t", half_life=1.0, bias=True)
    with pytest.raises(TypeError, match="bias must be True or False"):
        rp.ops.clock_step("t", half_life=1.0, stat="std", bias=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="holds the operator increment"):
        rp.ops.clock_step(rp.ops.increment("t"), half_life=1.0)
    with pytest.raises(ValueError, match="another window operator"):
        rp.ops.ewm_mean(rp.ops.clock_step("t", half_life=1.0), half_life=1.0)
    with pytest.raises(rp.WrongTypeError, match="must be a number or a temporal column"):
        rp.stream.with_windows(df, m=rp.ops.clock_step("s", half_life=1.0))
