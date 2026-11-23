"""``rheon.spec.power_variation`` against its definitions (docs/PLAN.md task
555): one pass, a decay with Kish's count, a window of rows and of the
clock, each term of each kind on simulated returns with a bid-ask bounce
and on ETCUSDT's trades; normalized terms are ``realized``'s measures;
chunking and a save change nothing; a row of weight 0 learns nothing; and
the builder's refusals. ``tests/reference_power_variation.py`` holds the
definitions, in numpy.
"""

from __future__ import annotations

import io
import math

import numpy as np
import polars as pl
import pytest

import reference_power_variation as ref
from data import public_quotes_and_trades_or_skip
from rheon import polars as rp

TIER = "essential"

INF = float("inf")

#: A term of each form: a sum of squares, normalized bipower variation,
#: unequal powers, normalized tripower quarticity, a centered Parzen sum, a
#: flat-top Bartlett sum, a Tukey-Hanning sum of two-step returns, and the
#: squares of five-step returns.
TERMS = {
    "rv": {"powers": [2]},
    "bv": {"powers": [1, 1], "normalize": True},
    "uneq": {"powers": [1, 0.5]},
    "tq": {"powers": [4 / 3, 4 / 3, 4 / 3], "normalize": True},
    "lrv": {"kernel": "parzen", "bandwidth": 5, "center": True},
    "rk": {"kernel": "bartlett", "bandwidth": 3, "flat_top": True},
    "th": {"kernel": "tukey_hanning", "bandwidth": 4, "scale": 2},
    "sub": {"powers": [2], "scale": 5},
}

#: The largest gaps the core's tests measured, in ulps of the larger of the
#: value and its scale, with room.
ONE_PASS = 16
DECAYED = 256


def _gap(got: float, want: float, scale: float) -> float:
    if math.isnan(got) and math.isnan(want):
        return 0.0
    return abs(got - want) / np.spacing(max(abs(want), abs(scale)))


def bounced(n: int, seed: int) -> np.ndarray:
    """Returns of a log price with a stochastic volatility, observed with a
    bid-ask bounce: the efficient return plus the difference of two noise
    draws, so lag 1 is negatively autocorrelated."""
    rng = np.random.default_rng(seed)
    vol = 0.0005 * np.exp(np.cumsum(0.02 * rng.standard_normal(n)))
    noise = 0.0003 * rng.standard_normal(n + 1)
    return vol * rng.standard_normal(n) + np.diff(noise)


def _frame(r, **cols) -> pl.DataFrame:
    """The returns, then one row with no return, which reads the whole."""
    return pl.DataFrame({"r": [*map(float, r), None], **cols})


def _fit(spec: dict, df: pl.DataFrame) -> pl.DataFrame:
    return rp.ModelBank([spec]).fit_predict(df).unnest("v")


def _check(row: dict, r, om, bound: float, label: str) -> None:
    n = len(r) if om is None else ref.kish(om)
    om = [1.0] * len(r) if om is None else om
    for name, spec in TERMS.items():
        want = ref.term(r, om, spec, n)
        gap = _gap(row[f"{name}_r"], want, ref.scale_of(r, om, spec))
        assert gap <= bound, f"{label}: {name} {row[f'{name}_r']} against {want}, {gap:.1f}"
    assert _gap(row["n_r"], n, 0.0) <= bound, f"{label}: n {row['n_r']} against {n}"


def test_one_pass_is_the_definition_of_every_prefix():
    """Without a decay a row's fields are the definitions on the returns
    before it, ``n`` their count."""
    r = bounced(3_000, 7)
    out = _fit(rp.spec.power_variation("v", targets=["r"], terms=TERMS, half_life=INF), _frame(r))
    for i in [*range(1, 3_000, 397), 3_000]:
        _check(out.row(i, named=True), r[:i], None, ONE_PASS, f"row {i}")
    whole = out.row(3_000, named=True)
    # The bounce: the plain sum of squares is several times the flat-top
    # kernel, which takes the noise's autocovariance out.
    assert whole["rv_r"] > 2 * whole["rk_r"] > 0


@pytest.mark.parametrize("w", [1, 3, 60])
def test_a_window_of_rows_is_the_definition_on_its_returns(w):
    """A window of ``W`` rows: each term whose returns are all in it, ``n``
    the count it holds, and its weight theirs."""
    r = bounced(800, 11)
    spec = rp.spec.power_variation("v", targets=["r"], terms=TERMS, half_life=INF, window_rows=w)
    out = _fit(spec, _frame(r))
    for i in [*range(2, 800, 97), w - 1, w, w + 1, 800]:
        if i < 1:
            continue
        row = out.row(i, named=True)
        _check(row, r[max(i - w, 0) : i], None, ONE_PASS, f"row {i}")
        assert row["weight_sum"] == min(i, w)


@pytest.mark.parametrize("h", [30.0, 300.0])
def test_a_decay_is_the_weighted_definition_with_kishs_count(h):
    """Under a decay each term is weighted by its newest row's weight and
    decayed by its age, and ``n`` is Kish's effective count: an irregular
    clock in 64ths, weights in [0.5, 1.5]."""
    rng = np.random.default_rng(3)
    n = 2_000
    r = bounced(n, 13)
    t = np.cumsum(1 + np.floor(rng.uniform(0, 256, n))) / 64
    w = rng.uniform(0.5, 1.5, n)
    df = _frame(r, t=[*t, t[-1] + 1], w=[*w, 1.0])
    spec = rp.spec.power_variation(
        "v", targets=["r"], terms=TERMS, half_life=h, clock="t", gap_cap=1e9, weight="w"
    )
    out = _fit(spec, df)
    for i in range(5, n + 1, 331):
        om = w[:i] * 2.0 ** (-(t[i - 1] - t[:i]) / h)
        _check(out.row(i, named=True), r[:i], list(om), DECAYED, f"h {h} row {i}")


@pytest.mark.parametrize("closed", ["right", "both"])
def test_a_clock_window_on_trades_holds_the_rows_of_its_span(closed):
    """ETCUSDT's trades of a day as tick returns on a temporal clock; a window of ``"5m"`` holds the
    returns stamped less than 5 minutes before the row before (or exactly 5
    under ``closed="both"``), and its fields are the definitions on them."""
    trades = public_quotes_and_trades_or_skip().filter(pl.col("kind") == "trade")
    ticks = (
        trades.group_by("ts", maintain_order=True)
        .agg(pl.col("price").last())
        .with_columns(r=pl.col("price").log().diff())
        .drop_nulls("r")
        .head(20_000)
    )
    spec = rp.spec.power_variation(
        "v",
        targets=["r"],
        terms=TERMS,
        half_life=INF,
        clock="ts",
        gap_cap="1d",
        window_size="5m",
        closed=closed,
        max_state_bytes=4 << 20,
    )
    out = _fit(spec, ticks)
    r = ticks["r"].to_numpy()
    ns = ticks["ts"].dt.epoch("ns").to_numpy()
    span = 5 * 60 * 10**9
    for i in range(500, len(r), 2_311):
        now = ns[i - 1]
        held = ns[:i] > now - span if closed == "right" else ns[:i] >= now - span
        _check(out.row(i, named=True), r[:i][held], None, ONE_PASS, f"row {i}")


def test_normalized_terms_are_realizeds_measures():
    """Normalized, ``[2]``, ``[1, 1]`` and ``[4/3]*3`` are ``realized``'s
    ``rv``, ``bv`` and ``tq``: the first two to the bit, the quarticity
    within the rounding of its powers (``realized`` takes cube roots)."""
    r = bounced(2_000, 17)
    q = 4 / 3
    terms = {
        "rv": {"powers": [2], "normalize": True},
        "bv": {"powers": [1, 1], "normalize": True},
        "tq": {"powers": [q, q, q], "normalize": True},
    }
    for half_life in (INF, 50.0):
        a = _fit(
            rp.spec.power_variation("v", targets=["r"], terms=terms, half_life=half_life),
            _frame(r),
        )
        b = _fit(rp.spec.realized("v", targets=["r"], half_life=half_life), _frame(r))
        assert a["rv_r"].to_list() == b["rv_r"].to_list()
        assert a["bv_r"].to_list() == b["bv_r"].to_list()
        gaps = [_gap(x, y, 0.0) for x, y in zip(a["tq_r"], b["tq_r"], strict=True) if y]
        assert max(gaps) <= 32, max(gaps)


def test_chunking_and_a_save_change_nothing():
    """One chunk or many, and a save at any row, give the same bits, under a
    decay and under a window, with a null return and uneven weights."""
    rng = np.random.default_rng(5)
    n = 600
    df = pl.DataFrame(
        {
            "r": [None if i % 9 == 4 else float(x) for i, x in enumerate(bounced(n, 19))],
            "w": rng.uniform(0.5, 1.5, n),
        }
    )
    for kw in (dict(half_life=40.0), dict(half_life=INF, window_rows=50)):
        spec = rp.spec.power_variation("v", targets=["r"], terms=TERMS, weight="w", **kw)
        whole = rp.ModelBank([spec]).fit_predict(df)
        for size in (1, 7, 128):
            bank = rp.ModelBank([spec])
            parts = [bank.fit_predict(df.slice(s, size)) for s in range(0, n, size)]
            assert pl.concat(parts).equals(whole), (kw, size)
        for cut in (1, 3, 251):
            bank = rp.ModelBank([spec])
            bank.fit_predict(df.head(cut))
            buf = io.BytesIO()
            bank.save(buf)
            back = rp.ModelBank.load(io.BytesIO(buf.getvalue()))
            assert back.fit_predict(df.slice(cut)).equals(whole.slice(cut)), (kw, cut)


def test_a_row_of_weight_zero_learns_nothing():
    """A row of weight 0 is in no term and no window, and counts in no
    clock: the rows after it read what they read without it, to the bit."""
    r = bounced(400, 23)
    w = [0.0 if i % 6 == 2 else 1.0 for i in range(400)]
    for kw in (dict(half_life=30.0), dict(half_life=INF, window_rows=25)):
        spec = rp.spec.power_variation("v", targets=["r"], terms=TERMS, weight="w", **kw)
        a = _fit(spec, pl.DataFrame({"r": r, "w": w})).filter(pl.Series(w) > 0)
        kept = [x for x, wi in zip(r, w, strict=True) if wi > 0]
        b = _fit(spec, pl.DataFrame({"r": kept, "w": [1.0] * len(kept)}))
        assert a.equals(b), kw


def test_the_builder_writes_the_spec_and_names_the_fields():
    spec = rp.spec.power_variation(
        "v",
        targets=["a", "b"],
        terms={"rv": {"powers": [2]}, "lrv": {"kernel": "parzen", "bandwidth": 3}},
        half_life=INF,
    )
    assert spec["model"] == {
        "type": "power_variation",
        "terms": [
            {"name": "rv", "powers": [2.0]},
            {"name": "lrv", "kernel": "parzen", "bandwidth": 3},
        ],
    }
    assert rp.ModelBank([spec]).specs[0] == spec
    fields = rp.spec.output_fields(spec)
    assert fields[:7] == ["rv_a", "lrv_a", "n_a", "rv_b", "lrv_b", "n_b", "weight_sum"]


@pytest.mark.parametrize(
    ("terms", "says"),
    [
        ({"x": {}}, "neither powers nor a kernel"),
        ({"x": {"powers": [2], "kernel": "parzen", "bandwidth": 2}}, "both powers and a kernel"),
        ({"x": {"kernel": "parzen"}}, "no bandwidth"),
        ({"x": {"powers": [2], "center": True}}, "belong to a kernel term"),
        ({"x": {"kernel": "bartlett", "bandwidth": 2, "normalize": True}}, "normalize belongs"),
        ({"n": {"powers": [2]}}, 'a term is named "n"'),
        ({"x": {"powers": [0]}}, "above 0"),
        ({"x": {"powers": [2], "scale": 2, "normalize": True}}, "normalized at a scale"),
        ({"x": {"kernel": "bartlett", "bandwidth": 0, "flat_top": True}}, "flat-top"),
        ({"x": {"powers": [2], "lags": 3}}, "which no term takes"),
    ],
)
def test_a_terms_shape_is_refused_by_name(terms, says):
    with pytest.raises(ValueError, match=says):
        rp.spec.power_variation("v", targets=["r"], terms=terms, half_life=INF)


def test_features_are_refused():
    with pytest.raises(TypeError):
        rp.spec.power_variation(
            "v", targets=["r"], features=["x"], terms={"rv": {"powers": [2]}}, half_life=INF
        )
