"""``realized`` through the bank: every field against the papers' estimators
written out in numpy (``tests/reference_realized.py``) on realistic days --
a stochastic-volatility day with and without planted jumps, Binance's
BTCUSDT one-minute closes and ETCUSDT trades on a temporal clock -- under
no decay, a decay with unequal weights, a window of rows and a window of
clock units at either edge; Lee and Mykland's statistic and its flag on
either side of the critical value; and what the bank owes every model:
chunk invariance, a save mid-stream, zero-weight rows in no clock, nulls,
its refusals and its output schema (docs/PLAN.md task 499).

hfvol, a library's, is the second opinion on RV and BV
(``tests/test_second_opinion_realized.py``); R's ``highfrequency`` and
``yuima`` are in ``tests/r_oracle/``.
"""

from __future__ import annotations

import io
import math

import numpy as np
import polars as pl
import pytest

import reference_realized as ref
from data import public_intraday_or_skip, public_quotes_and_trades_or_skip
from rheon import polars as rp

TIER = "essential"

INF = float("inf")
FIELDS = ("rv", "bv", "tv", "tq", "bns")
EPS = np.finfo(float).eps


def _ulps(got: float, want: float) -> float:
    if math.isnan(got) and math.isnan(want):
        return 0.0
    return abs(got - want) / np.spacing(abs(want))


def _z_gap(got: float, want: dict) -> float:
    """The test statistic's gap in units of ``cond · ε``: it is a
    difference of RV and BV, so their rounding is amplified by ``cond =
    (RV + BV) / |RV - BV|``."""
    cond = (want["rv"] + want["bv"]) / abs(want["rv"] - want["bv"])
    return abs(got - want["bns"]) / (cond * EPS * max(abs(want["bns"]), 1.0))


#: The largest gaps the core's tests measured to the same definitions, in
#: ulps (``Z`` in units of ``cond · ε``), with room: the cube roots against
#: the papers' powers cost TV and TQ their ten to twenty.
ONE_PASS = {"rv": 4, "bv": 4, "tv": 32, "tq": 64, "bns": 16}


def _frame(r, **cols) -> pl.DataFrame:
    """The returns, then one row with no return, which reads the whole."""
    return pl.DataFrame({"r": [*map(float, r), None], **cols})


def _fit(spec: dict, df: pl.DataFrame) -> pl.DataFrame:
    return rp.ModelBank([spec]).fit_predict(df).unnest("v")


def _check(row: dict, want: dict, bounds: dict, label: str) -> None:
    for f in FIELDS:
        got = row[f"{f}_r"]
        gap = _z_gap(got, want) if f == "bns" else _ulps(got, want[f])
        assert gap <= bounds[f], f"{label}: {f} {got} against {want[f]}, {gap:.1f}"


@pytest.mark.parametrize(("n", "jumps"), [(2_340, True), (23_400, True), (23_400, False)])
def test_one_pass_is_the_papers_measures_of_every_prefix(n, jumps):
    """Without a decay a row's fields are the papers' measures of the returns
    before it: ``n - 1`` products, ``n - 2`` triples."""
    r = ref.sv_returns(n, 7, jumps)
    out = _fit(rp.spec.realized("v", targets=["r"], half_life=INF), _frame(r))
    for i in [*range(5, n, 997), n]:
        _check(out.row(i, named=True), ref.measures(r[:i]), ONE_PASS, f"row {i}")
    whole = out.row(n, named=True)
    if jumps:
        assert whole["bns_r"] > 1.645 and whole["rv_r"] > whole["bv_r"]
    else:
        assert abs(whole["bns_r"]) < 1.645


@pytest.mark.parametrize("w", [50, 390])
def test_a_window_of_rows_is_the_papers_measures_of_its_returns(w):
    """A window of ``W`` returns holds ``W - 1`` products and ``W - 2``
    triples: a row's fields are the papers' on the ``W`` returns before it,
    and its weight is theirs."""
    r = ref.sv_returns(3_000, 11, True)
    spec = rp.spec.realized("v", targets=["r"], half_life=INF, window_rows=w)
    out = _fit(spec, _frame(r))
    for i in [*range(3, 3_000, 211), w - 1, w, w + 1, 3_000]:
        row = out.row(i, named=True)
        _check(row, ref.measures(r[max(i - w, 0) : i]), ONE_PASS, f"row {i}")
        assert row["weight_sum"] == min(i, w)


@pytest.mark.parametrize("h", [30.0, 300.0])
def test_a_decay_is_the_weighted_definition_with_kishs_count(h):
    """Under a decay each term is weighted by its row's weight and decayed by
    its age, and ``n`` is Kish's effective count of the decayed weights: an
    irregular clock in 64ths, so the ages carry no rounding of their own,
    and weights in [0.5, 1.5]."""
    rng = np.random.default_rng(3)
    n = 4_000
    r = ref.sv_returns(n, 13, True)
    t = np.cumsum(1 + np.floor(rng.uniform(0, 256, n))) / 64
    w = rng.uniform(0.5, 1.5, n)
    df = _frame(r, t=[*t, t[-1] + 1], w=[*w, 1.0])
    spec = rp.spec.realized("v", targets=["r"], half_life=h, clock="t", gap_cap=1e9, weight="w")
    out = _fit(spec, df)
    bounds = {"rv": 64, "bv": 64, "tv": 64, "tq": 128, "bns": 32}
    for i in range(5, n + 1, 331):
        want = ref.weighted(t[:i], r[:i], w[:i], h)
        _check(out.row(i, named=True), want, bounds, f"h {h} row {i}")


@pytest.mark.parametrize("closed", ["right", "both"])
def test_a_clock_window_on_trades_holds_the_rows_of_its_span(closed):
    """ETCUSDT's trades of a day, the last price of each 10 seconds as a log
    return on a temporal clock; a window of ``"30m"`` holds the returns
    whose stamps are less than 30 minutes before the row before (or exactly
    30 under ``closed="both"``), and its fields are the papers' measures of
    them."""
    trades = public_quotes_and_trades_or_skip().filter(pl.col("kind") == "trade")
    bars = (
        trades.group_by_dynamic("ts", every="10s")
        .agg(pl.col("price").last())
        .with_columns(r=pl.col("price").log().diff())
        .drop_nulls("r")
    )
    spec = rp.spec.realized(
        "v",
        targets=["r"],
        half_life=INF,
        window_size="30m",
        closed=closed,
        clock="ts",
        gap_cap="1d",
    )
    out = _fit(spec, bars.select("ts", "r"))
    ts = bars["ts"].dt.epoch("us").to_numpy()
    r = bars["r"].to_numpy()
    span = 30 * 60 * 1_000_000
    checked = 0
    for i in range(400, len(r), 397):
        age = ts[i - 1] - ts[:i]
        held = (age < span) | ((age == span) if closed == "both" else False)
        row = out.row(i, named=True)
        assert row["weight_sum"] == held.sum(), i
        _check(row, ref.measures(r[:i][held]), ONE_PASS, f"row {i}")
        checked += 1
    assert checked > 15


def test_btcusdt_minutes_are_the_papers_measures_and_the_test_is_finite():
    """A day of BTCUSDT's one-minute closes, its 1,439 log returns measured
    to the papers' definitions over the day."""
    day = public_intraday_or_skip()
    r = np.diff(np.log(day["close"].to_numpy()))
    out = _fit(rp.spec.realized("v", targets=["r"], half_life=INF), _frame(r))
    whole = out.row(len(r), named=True)
    _check(whole, ref.measures(r), ONE_PASS, "the day")
    assert math.isfinite(whole["bns_r"])


def test_lee_myklands_statistic_and_its_flag():
    """``lm`` is the paper's ``L_i`` from the ``K - 1`` returns before the
    row, to the rounding of the products; the flag is the paper's test at
    ``lm_beta``, with ``n`` the window's count of returns, on either side
    of its critical value; the largest planted jumps are flagged."""
    n, k, w = 2_340, 200, 390
    r = ref.sv_returns(n, 17, True)
    spec = rp.spec.realized("v", targets=["r"], half_life=INF, window_rows=w, lm_window=k)
    out = _fit(spec, _frame(r))
    want = ref.lee_mykland(r, k)
    lm = out["lm_r"].to_numpy()[:n]
    assert max(_ulps(lm[i], want[i]) for i in range(k - 1, n)) <= 8
    flag = out["lm_jump_r"].to_list()[:n]
    sides = {True: 0, False: 0}
    for i in range(k - 1, n):
        count = min(i, w)
        edge = ref.critical(count, 0.99)
        if abs(abs(lm[i]) - edge) > 1e-9 * edge:
            assert flag[i] == (abs(lm[i]) > edge), i
            sides[flag[i]] += 1
    assert sides[True] >= 4 and sides[False] > 1_000, sides
    # The two largest jumps, 10 and 12 local deviations, are flagged; the
    # 6 and 7, measured against the 200 returns before them on a U-shaped
    # day, are flagged where the paper's statistic says, as above.
    for at in (0.3, 0.7):
        assert flag[int(at * n)], at
    # A lower confidence flags more rows, the statistic itself unmoved.
    lax = _fit(
        rp.spec.realized(
            "v", targets=["r"], half_life=INF, window_rows=w, lm_window=k, lm_beta=0.5
        ),
        _frame(r),
    )
    assert lax["lm_r"].equals(out["lm_r"])
    assert lax["lm_jump_r"].sum() > out["lm_jump_r"].sum()


KWS = [
    {"half_life": 40.0, "lm_window": 30},
    {"half_life": INF, "window_rows": 60},
    {"half_life": INF, "window_size": 25.0},
    {"half_life": INF, "window_size": 25.0, "closed": "both", "lm_window": 12},
]


def _stream(n: int, seed: int) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.cumsum(np.floor(rng.uniform(0, 3, n)) * 0.5)
    w = np.where(rng.uniform(size=n) < 0.1, 0.0, rng.uniform(0.5, 1.5, n))
    a = ref.sv_returns(n, seed, True)
    b = ref.sv_returns(n, seed + 1, False)
    return pl.DataFrame(
        {
            "g": np.where(np.arange(n) % 3 == 0, "x", "y"),
            "t": t,
            "w": w,
            "a": a,
            "b": [None if i % 7 == 3 else v for i, v in enumerate(b)],
        }
    )


def _spec(**kw) -> dict:
    return rp.spec.realized(
        "v", targets=["a", "b"], clock="t", gap_cap=1e9, group="g", weight="w", **kw
    )


@pytest.mark.parametrize("kw", KWS)
def test_chunk_invariance(kw):
    df = _stream(600, 3)
    s = _spec(**kw)
    one = rp.ModelBank([s]).fit_predict(df)
    for size in (1, 7, 97):
        bank = rp.ModelBank([s])
        many = pl.concat([bank.fit_predict(df.slice(i, size)) for i in range(0, 600, size)])
        assert one.equals(many, null_equal=True), size


@pytest.mark.parametrize("kw", KWS)
def test_a_save_goes_on_to_the_bit(kw):
    df = _stream(500, 4)
    s = _spec(**kw)
    for cut in (1, 37, 250, 499):
        a = rp.ModelBank([s])
        a.fit_predict(df.slice(0, cut))
        buf = io.BytesIO()
        a.save(buf)
        b = rp.ModelBank.load_bytes(buf.getvalue(), specs=[s])
        rest = df.slice(cut)
        assert a.fit_predict(rest).equals(b.fit_predict(rest), null_equal=True), cut


@pytest.mark.parametrize("kw", KWS)
def test_a_zero_weight_row_counts_in_no_clock(kw):
    """Rows of weight 0, each carrying a return, give every other row, to the
    bit, the output of the stream without them: no product, no window, no
    clock and no weight takes them. They are still read."""
    df = _stream(500, 5)
    full = rp.ModelBank([_spec(**kw)]).fit_predict(df).unnest("v")
    mask = (df["w"] > 0).to_numpy()
    thin = rp.ModelBank([_spec(**kw)]).fit_predict(df.filter(pl.col("w") > 0)).unnest("v")
    for f in rp.spec.output_fields(_spec(**kw)):
        if f in ("settled_frac", "withheld_reason"):
            continue
        assert full[f].filter(pl.Series(mask)).equals(thin[f], null_equal=True), (kw, f)
    zero = full.filter(pl.Series(~mask))
    assert zero["rv_a"].drop_nulls().len() > 20, "a zero-weight row is still read"


def test_an_absent_return_is_its_series_own():
    """A null return is that target's alone: the other target reads as it
    reads alone, and the target's products reach across the null to its
    last return."""
    n = 800
    a, b = ref.sv_returns(n, 21, True), ref.sv_returns(n, 22, False)
    nulls = np.arange(n) % 5 == 2
    df = pl.DataFrame(
        {"a": [*a, None], "b": [*[None if m else v for m, v in zip(nulls, b, strict=True)], None]}
    )
    both = rp.ModelBank([rp.spec.realized("v", targets=["a", "b"], half_life=INF)]).fit_predict(df)
    alone = rp.ModelBank([rp.spec.realized("v", targets=["a"], half_life=INF)]).fit_predict(
        df.select("a")
    )
    for f in ("rv", "bv", "tv", "tq", "bns", "lm", "lm_jump"):
        assert (
            both["v"]
            .struct.field(f"{f}_a")
            .equals(alone["v"].struct.field(f"{f}_a"), null_equal=True)
        ), f
    whole = both["v"].struct.unnest().row(n, named=True)
    want = ref.measures(b[~nulls])
    for f in FIELDS:
        got = whole[f"{f}_b"]
        gap = _z_gap(got, want) if f == "bns" else _ulps(got, want[f])
        assert gap <= ONE_PASS[f], (f, got, want[f])
    assert whole["lm_b"] is None, "no return, no statistic"


def test_names_match_the_realized_struct():
    """Seven fields a target and the shared ones, of the dtypes
    ``output_index`` declares; each ``lm_jump`` a Boolean; no ``coef``."""
    df = _stream(200, 8)
    for kw in [*KWS[:2], {"half_life": [10.0, 50.0]}]:
        spec = _spec(**kw)
        out = rp.ModelBank([spec]).fit_predict(df)
        fields = out.schema["v"].fields
        assert [f.name for f in fields] == rp.spec.output_fields(spec)
        flags = [f for f in fields if f.name.startswith("lm_jump")]
        assert flags and all(f.dtype == pl.Boolean for f in flags), kw
        assert not any(f.name.startswith("coef") for f in fields)


@pytest.mark.parametrize(
    ("kw", "says"),
    [
        ({"features": ["x"]}, "takes no features"),
        ({"half_life": 10.0, "window_rows": 50}, "takes no decay"),
        ({"half_life": INF, "window_rows": 0}, "window_rows must be at least 1"),
        ({"half_life": INF, "window_size": -1.0}, "window_size must be a finite span"),
        ({"half_life": INF, "window_rows": 5, "window_size": 5.0}, "two windows"),
        ({"half_life": 10.0, "closed": "both"}, "there is no clock window"),
        ({"half_life": 10.0, "lm_window": 2}, "lm_window must be at least 3"),
        ({"half_life": 10.0, "lm_beta": 1.0}, "lm_beta must be above 0 and below 1"),
        ({"half_life": 10.0, "coef_every": 5}, "coef_every"),
        ({"half_life": INF, "window_rows": 10_000_000}, "largest window_rows that fits"),
    ],
)
def test_refusals(kw, says):
    with pytest.raises((ValueError, TypeError), match=says):
        rp.ModelBank([rp.spec.realized("v", targets=["r"], **kw)])


def test_a_burst_in_a_clock_window_is_warned_of_once_and_refused_at_the_cap():
    """A window of clock units holds the rows its span has: the first row to
    take a group past half of ``max_state_bytes`` warns, once, naming the
    group and the bytes; the row that would take it past the cap stops the
    run with an error naming the group and the bytes. A group that stays
    under says nothing."""
    import rheon

    # 8 + 144 bytes of sums, 32 beside the window, 16 + 32 + 56 a row.
    cap = 8 + 144 + 32 + 104 * 150
    spec = rp.spec.realized(
        "v",
        targets=["r"],
        half_life=INF,
        window_size=1e6,
        max_state_bytes=cap,
        clock="t",
        gap_cap=1e9,
        group="g",
    )

    def burst(group: str, n: int, first: int = 0) -> pl.DataFrame:
        i = np.arange(first, first + n)
        r = np.where(i % 2 == 0, 0.001, -0.0015)
        return pl.DataFrame({"g": group, "t": i * 0.01, "r": r})

    bank = rp.ModelBank([spec])
    with pytest.warns(rheon.RheonWarning, match=r'group "a": the state holds \d+ bytes') as seen:
        bank.fit_predict(pl.concat([burst("a", 100), burst("b", 30)]))
    assert len(seen) == 1
    assert f"past half of max_state_bytes ({cap})" in str(seen[0].message)
    with pytest.raises(Exception, match=r'group "a".*to \d+ bytes, above max_state_bytes of') as e:
        bank.fit_predict(burst("a", 100, 100))
    assert str(cap) in str(e.value)
