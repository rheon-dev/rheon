"""hfvol's realized variance, bipower variation and jump variation as the
second opinion on ``realized`` (docs/PLAN.md task 499). hfvol applies no
small-sample factor, as the papers and ``realized`` do not, so the two are
held to the rounding of the sums: a few units in the last place.

The days: a stochastic-volatility day with five planted jumps and one
without (so the jump variation ``max(RV - BV, 0)`` is met on both sides of
its floor), at 390, 2,340 and 23,400 returns; BTCUSDT's one-minute closes;
ETCUSDT's trades as 10-second returns.
"""

from __future__ import annotations

import math

import hfvol
import numpy as np
import polars as pl
import pytest

import reference_realized as ref
from data import public_intraday_or_skip, public_quotes_and_trades_or_skip
from rheon import polars as rp

TIER = "essential"


def _whole_day(r: np.ndarray, **kw) -> dict:
    """``realized``'s fields of a day: the row after its last return reads
    them all."""
    df = pl.DataFrame({"r": [*map(float, r), None]})
    spec = rp.spec.realized("v", targets=["r"], half_life=float("inf"), **kw)
    return rp.ModelBank([spec]).fit_predict(df).unnest("v").row(len(r), named=True)


def _ulps(got: float, want: float) -> float:
    return abs(got - want) / np.spacing(abs(want))


def _days() -> list[tuple[str, np.ndarray]]:
    days = [
        (f"sv {n} {'jumps' if j else 'none'}", ref.sv_returns(n, 31 + n, j))
        for n in (390, 2_340, 23_400)
        for j in (True, False)
    ]
    return days


@pytest.mark.parametrize(("name", "r"), _days(), ids=[d[0] for d in _days()])
def test_rv_bv_and_the_jump_variation_are_hfvols(name, r):
    got = _whole_day(r)
    assert _ulps(got["rv_r"], hfvol.realized_variance(r)) <= 4, name
    assert _ulps(got["bv_r"], hfvol.bipower_variation(r)) <= 8, name
    continuous, jump = hfvol.jump_variation(r)
    assert _ulps(got["bv_r"], continuous) <= 8, name
    ours = max(got["rv_r"] - got["bv_r"], 0.0)
    assert abs(ours - jump) <= 16 * np.spacing(got["rv_r"]), (name, ours, jump)


def test_both_sides_of_the_jump_variations_floor_are_met():
    """The floor binds on a day where BV comes out above RV, and not on a
    day with jumps: hfvol's jump variation is 0 on one and positive on the
    other, as ``realized``'s ``rv - bv`` is negative and positive."""
    signs = set()
    # Returns of one size and alternating signs: each product is RV's mean
    # square, so BV is π/2 of RV and the floor binds.
    even = 0.001 * np.where(np.arange(390) % 2 == 0, 1.0, -1.0)
    for r in [d for _, d in _days()] + [even * (1 + 0.01 * np.sin(np.arange(390)))]:
        got = _whole_day(r)
        _, jump = hfvol.jump_variation(r)
        diff = got["rv_r"] - got["bv_r"]
        signs.add(diff > 0)
        assert (jump == 0.0) == (diff <= 0), (diff, jump)
    assert signs == {True, False}


def test_real_days_are_hfvols():
    """BTCUSDT's minutes and ETCUSDT's trades in 10 seconds, log returns."""
    btc = public_intraday_or_skip()
    r_btc = np.diff(np.log(btc["close"].to_numpy()))
    trades = public_quotes_and_trades_or_skip().filter(pl.col("kind") == "trade")
    r_etc = (
        trades.group_by_dynamic("ts", every="10s")
        .agg(pl.col("price").last())
        .select(pl.col("price").log().diff())
        .drop_nulls()
        .to_series()
        .to_numpy()
    )
    for name, r in (("BTCUSDT", r_btc), ("ETCUSDT", r_etc)):
        got = _whole_day(r)
        assert _ulps(got["rv_r"], hfvol.realized_variance(r)) <= 4, name
        assert _ulps(got["bv_r"], hfvol.bipower_variation(r)) <= 8, name
        assert math.isfinite(got["bns_r"]), name


def test_a_window_of_a_day_is_hfvols_on_its_returns():
    """A window of 390 rows over three days of 390 returns: at each day's
    end the window holds that day's returns alone, ``W - 1`` products, as
    hfvol's bipower variation of the day does."""
    days = [ref.sv_returns(390, s, s == 1) for s in range(3)]
    r = np.concatenate(days)
    df = pl.DataFrame({"r": [*r, None]})
    spec = rp.spec.realized("v", targets=["r"], half_life=float("inf"), window_rows=390)
    out = rp.ModelBank([spec]).fit_predict(df).unnest("v")
    for d, day in enumerate(days):
        row = out.row(390 * (d + 1), named=True)
        assert _ulps(row["bv_r"], hfvol.bipower_variation(day)) <= 8, d
        assert _ulps(row["rv_r"], hfvol.realized_variance(day)) <= 4, d
