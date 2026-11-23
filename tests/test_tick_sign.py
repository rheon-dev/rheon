"""``rp.tick_sign``: the tick rule as a window operator.

A row's sign is +1 on a price above the last one, -1 below it, and over a
step of 0 the sign of the last step that was not 0. A row with no step (the
first of its group and session, a missing price) has no sign, and so do the
rows at the first price until it first moves.

The oracles, each a third party's computation of the same signs:

- **bar-forge** (test-only, MIT), whose ``tick_rule_signs`` is the tick rule
  over a list of trades. It signs a first trade +1 by convention, where the
  operator leaves it null; with that one fill the two are equal row for row.
- **Polars' own construction** over the whole frame: ``diff``, 0 read as
  null, ``sign``, and the last sign filled forward, ``.over`` the group. It
  needs the whole column at once; the operator reads one row at a time.

The rest is what makes it a streaming stage: one chunk or many give the same
signs to the bit, a saved state resumes to the one run's signs, a session
change starts the rule over, and a row of weight 0 under a weighted spec's
clock policy moves the sign as any row does.
"""

from __future__ import annotations

import random
from pathlib import Path

import bar_forge
import polars as pl
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from polars.testing import assert_series_equal

from rheon import polars as rp
from tiers import examples

TIER = "essential"

CLOCK = {"clock": "t", "gap_cap": 1e9}

PROPERTY = settings(
    max_examples=examples(60, essential=10),
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)


def ticks(seed: int, n: int = 500, groups: int = 1, nulls: bool = False) -> pl.DataFrame:
    """A seeded trade stream on a price grid of 1/64: about half the steps
    are 0, so the carried sign is read on most rows."""
    rng = random.Random(seed)
    price = {g: 100.0 for g in range(groups)}
    rows = []
    for i in range(n):
        g = rng.randrange(groups)
        price[g] += rng.choice([0, 0, 0, 1, -1, 2, -2]) / 64
        p: float | None = price[g]
        if nulls and rng.random() < 0.08:
            p = None
        rows.append((float(i), f"g{g}", p, float(rng.randint(1, 100))))
    return pl.DataFrame(rows, schema=["t", "g", "price", "size"], orient="row")


def polars_tick_sign(price: pl.Expr) -> pl.Expr:
    """The tick rule in Polars over a whole column: the sign of ``diff``,
    with a step of 0 read as null and the last sign filled forward."""
    d = price.diff()
    return pl.when(d != 0).then(d.sign()).otherwise(None).forward_fill().cast(pl.Float64)


def signed(df: pl.DataFrame, **kw: object) -> pl.Series:
    out = rp.stream.with_windows(df, s=rp.tick_sign("price"), **CLOCK, **kw)
    assert isinstance(out, pl.DataFrame)
    return out["s"]


# --------------------------------------------------------------------------
# The rule, row by row


def test_the_first_row_and_the_rows_at_its_price_have_no_sign() -> None:
    """No step, no sign: the first row, and each row after it at the same
    price, until the price first moves."""
    df = pl.DataFrame({"t": [0.0, 1.0, 2.0, 3.0], "price": [10.0, 10.0, 10.0, 9.5]})
    assert signed(df).to_list() == [None, None, None, -1.0]


@pytest.mark.parametrize(("first", "want"), [(10.5, 1.0), (9.5, -1.0)])
def test_a_step_of_zero_carries_the_last_sign_on_either_side(first: float, want: float) -> None:
    """Up then flat holds +1; down then flat holds -1: both sides of the
    zero-step rule, over several flat rows."""
    df = pl.DataFrame({"t": [0.0, 1.0, 2.0, 3.0, 4.0], "price": [10.0, first, first, first, first]})
    assert signed(df).to_list() == [None, want, want, want, want]


def test_a_step_of_zero_after_a_reversal_carries_the_new_sign() -> None:
    df = pl.DataFrame(
        {"t": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0], "price": [10.0, 11.0, 11.0, 10.0, 10.0, 10.0]}
    )
    assert signed(df).to_list() == [None, 1.0, 1.0, -1.0, -1.0, -1.0]


def test_a_missing_price_has_no_sign_and_the_next_step_is_from_the_last_price() -> None:
    """A null, NaN or infinite price is skipped, as an increment skips it:
    the carried sign stands across it, and the next row's step is from the
    last valued price, so 10 -> (null) -> 10 is a step of 0, not a move."""
    for missing in [None, float("nan"), float("inf")]:
        df = pl.DataFrame(
            {"t": [0.0, 1.0, 2.0, 3.0, 4.0], "price": [9.0, 10.0, missing, 10.0, 9.0]},
            schema={"t": pl.Float64, "price": pl.Float64},
        )
        assert signed(df).to_list() == [None, 1.0, None, 1.0, -1.0], missing


def test_an_integer_price_is_signed_in_integers() -> None:
    """A step of 1 on an integer past 2**53 is a move: the step is taken in
    integers, as an increment's is, so it is not lost to rounding."""
    big = 2**60
    df = pl.DataFrame(
        {"t": [0.0, 1.0, 2.0, 3.0], "price": [big, big + 1, big + 1, big]},
        schema={"t": pl.Float64, "price": pl.Int64},
    )
    assert signed(df).to_list() == [None, 1.0, 1.0, -1.0]


def test_a_session_change_starts_the_rule_over() -> None:
    """The first row of a session has no step, and the sign carried from the
    session before is gone: the rule starts over, as an increment does."""
    df = pl.DataFrame(
        {
            "t": [0.0, 1.0, 2.0, 3.0, 4.0],
            "day": [1, 1, 2, 2, 2],
            "price": [10.0, 11.0, 11.0, 11.0, 10.0],
        }
    )
    assert signed(df, session="day").to_list() == [None, 1.0, None, None, -1.0]


def test_each_group_carries_its_own_sign() -> None:
    df = pl.DataFrame(
        {
            "t": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "g": ["a", "b", "a", "b", "a", "b"],
            "price": [10.0, 20.0, 11.0, 19.0, 11.0, 19.0],
        }
    )
    assert signed(df, group="g").to_list() == [None, None, 1.0, -1.0, 1.0, -1.0]


def test_a_tick_sign_inside_an_operator_signs_the_flow() -> None:
    """Signed volume summed with no decay is the running order-flow
    imbalance: the tick sign is an operator's input, as an increment is."""
    df = ticks(4, n=200)
    out = rp.stream.with_windows(
        df,
        flow=rp.ewm_sum(rp.tick_sign("price") * pl.col("size"), half_life=float("inf")),
        **CLOCK,
    )
    want = (polars_tick_sign(pl.col("price")) * pl.col("size")).fill_null(0.0).cum_sum()
    assert_series_equal(out["flow"], df.select(flow=want)["flow"])


def test_a_tick_sign_of_a_tick_sign_or_of_a_window_is_refused() -> None:
    df = ticks(5, n=10)
    with pytest.raises(ValueError, match="holds another window operator"):
        rp.tick_sign(rp.ewm_sum("price", half_life=1.0))
    nested = "a tick sign's input holds another tick sign"
    with pytest.raises((pl.exceptions.ComputeError, ValueError), match=nested):
        rp.stream.with_windows(df, s=rp.tick_sign(rp.tick_sign("price")), **CLOCK)
    with pytest.raises(
        (pl.exceptions.ComputeError, ValueError), match="a tick sign's input holds an increment"
    ):
        rp.stream.with_windows(df, s=rp.tick_sign(rp.increment("price")), **CLOCK)
    with pytest.raises(TypeError, match="column name or a pl.Expr"):
        rp.tick_sign(3.0)  # type: ignore[arg-type]


def test_a_text_price_is_refused_as_the_wrong_type() -> None:
    df = pl.DataFrame({"t": [0.0, 1.0], "price": ["a", "b"]})
    with pytest.raises(
        (pl.exceptions.ComputeError, TypeError),
        match="a tick sign's input must be a number or a temporal column",
    ):
        rp.stream.with_windows(df, s=rp.tick_sign("price"), **CLOCK)


# --------------------------------------------------------------------------
# The oracles


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_bar_forge_tick_rule_signs_with_its_first_trade_filled(seed: int) -> None:
    """bar-forge's ``tick_rule_signs`` signs a first trade, and the trades
    at its price, +1; that fill on the operator's nulls gives its signs."""
    df = ticks(seed)
    trades = [bar_forge.Trade(t, p, s) for t, p, s in df.select("t", "price", "size").rows()]
    want = pl.Series("s", bar_forge.tick_rule_signs(trades), dtype=pl.Float64)
    got = signed(df)
    assert got.null_count() >= 1
    assert_series_equal(got.fill_null(1.0), want)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_polars_diff_sign_forward_fill_per_group(seed: int) -> None:
    """Polars' whole-column construction ``.over`` the group, to the bit,
    nulls included."""
    df = ticks(seed, groups=4)
    want = df.select(s=polars_tick_sign(pl.col("price")).over("g"))["s"]
    assert_series_equal(signed(df, group="g"), want)
    flat = df.select((pl.col("price").diff().over("g") == 0).sum()).item()
    assert flat > 100, "the carried sign is read on many rows"


# --------------------------------------------------------------------------
# The stream: chunks, state, weights


@given(seed=st.integers(0, 10_000), size=st.integers(1, 97))
@PROPERTY
def test_one_chunk_or_many_give_the_same_signs(seed: int, size: int) -> None:
    df = ticks(seed, n=300, groups=3, nulls=True)
    one = signed(df, group="g", chunk_size=df.height)
    many = signed(df, group="g", chunk_size=size)
    assert_series_equal(many, one)
    assert_series_equal(one, _oracle_with_nulls(df))


def _oracle_with_nulls(df: pl.DataFrame) -> pl.Series:
    """Polars' construction where a missing price is skipped: over the
    valued rows alone, per group, and null where the price is missing."""
    rows = df.with_row_index("i")
    valued = (
        rows.filter(pl.col("price").is_not_null())
        .with_columns(s=polars_tick_sign(pl.col("price")).over("g"))
        .select("i", "s")
    )
    return rows.join(valued, on="i", how="left").sort("i")["s"]


@pytest.mark.parametrize("at", [1, 2, 37, 150, 299])
def test_a_saved_state_resumes_to_the_one_runs_signs(at: int, tmp_path: Path) -> None:
    """Saved after ``at`` rows and loaded for the rest, the signs are the
    one run's: the last price and the carried sign of each group are in the
    state, so a step of 0 across the cut carries the sign from before it."""
    df = ticks(11, n=300, groups=3, nulls=True)
    whole = signed(df, group="g")
    state = tmp_path / "s.state"
    head = signed(df[:at], group="g", save_state=state)
    rest = signed(df[at:], group="g", load_state=state)
    assert_series_equal(pl.concat([head, rest]), whole)


def test_a_step_of_zero_across_a_resume_carries_the_sign(tmp_path: Path) -> None:
    df = pl.DataFrame({"t": [0.0, 1.0, 2.0, 3.0], "price": [10.0, 9.0, 9.0, 9.0]})
    state = tmp_path / "s.state"
    assert signed(df[:2], save_state=state).to_list() == [None, -1.0]
    assert signed(df[2:], load_state=state).to_list() == [-1.0, -1.0]


def test_a_row_of_weight_zero_moves_the_sign_as_any_row() -> None:
    """Under a weighted spec's clock policy (``like=``), a row of weight 0
    advances the clock and learns nothing in a model; its price is still the
    stream's last price. The sign looks back only, so no weight touches it:
    the signs are the call's without the weight, rows of weight 0 included,
    the first row among them."""
    df = ticks(21, n=200).with_columns(
        w=pl.Series([0.0 if i % 7 in (0, 3) else 1.0 for i in range(200)])
    )
    spec = rp.spec.ewridge(
        "m",
        targets=["size"],
        features=["price"],
        weight="w",
        half_life=10.0,
        clock="t",
        gap_cap=1e9,
    )
    lent = rp.stream.with_windows(df, s=rp.tick_sign("price"), like=spec)
    assert isinstance(lent, pl.DataFrame)
    assert df["w"][0] == 0.0
    assert_series_equal(lent["s"], signed(df))
    assert_series_equal(lent["s"], df.select(s=polars_tick_sign(pl.col("price")))["s"])
