"""``ar`` through the bank: its coefficients against statsmodels' ``AutoReg``
on the same rows with no decay, and against weighted least squares under
decay; its predictions against ``AutoReg`` fitted on the rows before each
one; and what the bank owes every model -- chunk invariance, a save
mid-stream, zero weights and null targets, a break that empties the lags,
weights as relative evidence, its refusals and its output schema (task 486).

The oracles are statsmodels' ``AutoReg`` (``trend="c"`` or ``"n"``,
``exog=``, ``missing="drop"``) and ``WLS`` on ``lagmat``'s design with the
decay's weights written out by hand.
"""

from __future__ import annotations

import io

import numpy as np
import polars as pl
import pytest
from statsmodels.regression.linear_model import WLS
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.tsatools import lagmat

from rheon import polars as rp

TIER = "essential"


def _series(n: int, seed: int, phi=(0.55, -0.25), beta: float = 0.8) -> pl.DataFrame:
    """``y = 4 + Σ φ_i y_{t-i} + β x + e`` after 300 rows of burn-in: a
    stationary AR with an exogenous regressor, its mean far from zero."""
    rng = np.random.default_rng(seed)
    p = len(phi)
    y = np.full(n + 300, 5.0)
    x = 1.0 + rng.standard_normal(n + 300)
    e = rng.standard_normal(n + 300)
    for t in range(p, n + 300):
        y[t] = 4.0 + sum(phi[i] * y[t - 1 - i] for i in range(p)) + beta * x[t] + e[t]
    return pl.DataFrame({"t": np.arange(n, dtype=float), "x": x[300:], "y": y[300:]})


def _out(spec: dict, df: pl.DataFrame) -> pl.DataFrame:
    return rp.ModelBank([spec]).fit_predict(df).unnest("a")


def _last_coef(out: pl.DataFrame) -> np.ndarray:
    return np.asarray(out["coef"][-1][0])


def _rel(got, want) -> float:
    got, want = np.asarray(got), np.asarray(want)
    return float(np.max(np.abs(got - want)) / np.max(np.abs(want)))


@pytest.mark.parametrize(
    ("lags", "exog"), [(2, False), (2, True), ([1, 3], True), (5, False), ([1, 2, 12], True)]
)
def test_no_decay_is_autoreg_on_the_same_rows(lags, exog):
    """With no decay and no ridge the coefficients are ``AutoReg``'s:
    ``(const, lags..., exog...)`` in the same order, to 1e-9 relative over
    5,000 rows."""
    df = _series(5000, 1)
    spec = rp.spec.ar(
        "a", targets=["y"], lags=lags, features=["x"] if exog else None, half_life=float("inf")
    )
    got = _last_coef(_out(spec, df))
    want = (
        AutoReg(
            df["y"].to_numpy(),
            lags=lags,
            trend="c",
            exog=df["x"].to_numpy()[:, None] if exog else None,
        )
        .fit()
        .params
    )
    assert _rel(got, want) < 1e-9, (got, want)


def test_without_an_intercept_it_is_autoreg_trend_n():
    df = _series(3000, 2)
    spec = rp.spec.ar("a", targets=["y"], lags=2, fit_intercept=False, half_life=float("inf"))
    got = _last_coef(_out(spec, df))
    want = AutoReg(df["y"].to_numpy(), lags=2, trend="n").fit().params
    assert len(got) == 2
    assert _rel(got, want) < 1e-9


def test_decay_is_weighted_least_squares_on_the_clock():
    """Under ``half_life`` on an uneven clock, with row weights, the fit is
    ``WLS`` on the lag design with each row weighted ``w 0.5^(age /
    half_life)``, its age on the clock."""
    n, h = 3000, 400.0
    df = _series(n, 3)
    rng = np.random.default_rng(30)
    df = df.with_columns(
        pl.Series("t", np.cumsum(rng.uniform(0.2, 1.8, n))),
        pl.Series("w", rng.uniform(0.5, 1.5, n)),
    )
    spec = rp.spec.ar(
        "a", targets=["y"], lags=2, features=["x"], clock="t", weight="w", half_life=h
    )
    got = _last_coef(_out(spec, df))
    y, x, t, w = (df[c].to_numpy() for c in ("y", "x", "t", "w"))
    lagged = lagmat(y, 2, trim="both")
    design = np.column_stack([np.ones(n - 2), lagged, x[2:]])
    weights = w[2:] * 0.5 ** ((t[-1] - t[2:]) / h)
    want = WLS(y[2:], design, weights=weights).fit().params
    assert _rel(got, want) < 1e-9, (got, want)


def test_a_prediction_is_autoreg_fitted_on_the_rows_before():
    """Hard rule 2: row ``t``'s prediction is ``AutoReg`` fitted on the rows
    before ``t``, applied to ``t``'s lags and features."""
    df = _series(600, 4)
    spec = rp.spec.ar("a", targets=["y"], lags=2, features=["x"], half_life=float("inf"))
    pred = _out(spec, df)["pred_y"].to_numpy()
    y, x = df["y"].to_numpy(), df["x"].to_numpy()
    for t in (20, 150, 599):
        c = AutoReg(y[:t], lags=2, trend="c", exog=x[:t, None]).fit().params
        want = c[0] + c[1] * y[t - 1] + c[2] * y[t - 2] + c[3] * x[t]
        assert abs(pred[t] - want) <= 1e-9 * abs(want), (t, pred[t], want)
    # Nothing before the lags and a fit of four coefficients: the ring fills
    # on rows 0 and 1, and the default min_weight waits for four learned rows.
    assert np.isnan(pred[:6]).all() and np.isfinite(pred[6:]).all()


def test_a_null_target_is_not_a_lag():
    """A row whose target is null is not in the series: with no decay the
    fit is ``AutoReg(missing="drop")``'s on the series with those rows
    dropped, and the row is still predicted."""
    df = _series(2000, 5)
    holes = np.arange(2000) % 23 == 7
    df = df.with_columns(pl.when(pl.Series(holes)).then(None).otherwise(pl.col("y")).alias("y"))
    spec = rp.spec.ar("a", targets=["y"], lags=3, half_life=float("inf"))
    out = _out(spec, df)
    y = df["y"].to_numpy().astype(float)
    want = AutoReg(y, lags=3, trend="c", missing="drop").fit().params
    assert _rel(_last_coef(out), want) < 1e-9
    assert out["pred_y"].filter(pl.Series(holes)).drop_nulls().len() > 70


def test_a_zero_weight_row_counts_in_no_clock():
    """Rows of weight 0 -- wild values among them -- leave every other row's
    output, to the bit, as the stream without them gives it, under decay on
    an uneven clock: they are no lag and age nothing. They are predicted."""
    n = 1500
    df = _series(n, 6)
    rng = np.random.default_rng(60)
    zero = rng.uniform(size=n) < 0.1
    zero[:2] = True
    df = df.with_columns(
        pl.Series("t", np.cumsum(rng.uniform(0.5, 1.5, n))),
        pl.Series("w", np.where(zero, 0.0, 1.0)),
        pl.when(pl.Series(zero)).then(1e6).otherwise(pl.col("y")).alias("y"),
    )
    spec = rp.spec.ar(
        "a", targets=["y"], lags=2, features=["x"], clock="t", weight="w", half_life=60.0,
        coef_every=0,
    )
    full = _out(spec, df)
    thin = _out(spec, df.filter(~pl.Series(zero)))
    keep = full.filter(~pl.Series(zero))
    for f in ("pred_y", "weight_sum", "coef"):
        assert keep[f].equals(thin[f], null_equal=True), f
    assert full["pred_y"].filter(pl.Series(zero)).drop_nulls().len() > 100


def test_a_session_change_empties_the_lags_and_keeps_the_fit():
    """On a session change the lags empty: the next two rows of an AR(2) are
    not predicted, and the fit at the end is least squares over each
    session's own rows, its first two only lags."""
    df = _series(1200, 7).with_columns(
        pl.Series("s", np.repeat(["a", "b", "c"], 400)),
        pl.Series("t", np.arange(1200, dtype=float)),
    )
    spec = rp.spec.ar(
        "a", targets=["y"], lags=2, features=["x"], clock="t", session="s",
        half_life=float("inf"),
    )
    out = _out(spec, df)
    pred = out["pred_y"].to_numpy()
    assert np.isnan(pred[[400, 401, 800, 801]]).all()
    assert np.isfinite(pred[[402, 802, 1199]]).all()
    y, x = df["y"].to_numpy(), df["x"].to_numpy()
    rows = [
        (np.r_[1.0, y[t - 1], y[t - 2], x[t]], y[t])
        for s in (0, 400, 800)
        for t in range(s + 2, s + 400)
    ]
    design, target = np.array([r[0] for r in rows]), np.array([r[1] for r in rows])
    want = np.linalg.lstsq(design, target, rcond=None)[0]
    assert _rel(_last_coef(out), want) < 1e-9


def test_each_target_is_its_own_series():
    """Two targets in one spec are two specs of one target each, to the
    bit."""
    df = _series(800, 8).with_columns(_series(800, 9)["y"].alias("z"))
    kw = dict(lags=[1, 2], features=["x"], half_life=200.0, coef_every=0)
    both = rp.ModelBank([rp.spec.ar("a", targets=["y", "z"], **kw)]).fit_predict(df).unnest("a")
    for j, t in enumerate(("y", "z")):
        one = _out(rp.spec.ar("a", targets=[t], **kw), df)
        assert both[f"pred_{t}"].equals(one[f"pred_{t}"], null_equal=True)
        coef = both["coef"].list.get(j)
        assert coef.equals(one["coef"].list.get(0), null_equal=True)


@pytest.mark.parametrize("kw", [{}, {"half_life": 50.0, "weight": "w"}])
def test_chunk_invariance(kw):
    df = _series(700, 10).with_columns(
        pl.Series("g", np.where(np.arange(700) % 3 == 0, "a", "b")),
        pl.Series("w", np.where(np.arange(700) % 11 == 4, 0.0, 1.5)),
    )
    s = rp.spec.ar("a", targets=["y"], lags=3, features=["x"], group="g", coef_every=0, **kw)
    one = rp.ModelBank([s]).fit_predict(df)
    for size in (1, 13, 211):
        bank = rp.ModelBank([s])
        many = pl.concat([bank.fit_predict(df.slice(i, size)) for i in range(0, 700, size)])
        assert one.equals(many, null_equal=True), size


def test_a_save_goes_on_to_the_bit():
    """Saved at every depth of the lags -- none, part, full -- and later, a
    bank goes on as the one it was saved from, to the bit."""
    df = _series(500, 11).with_columns(pl.Series("g", np.where(np.arange(500) % 2 == 0, "a", "b")))
    s = rp.spec.ar("a", targets=["y"], lags=[1, 4], features=["x"], group="g", half_life=40.0)
    for cut in (1, 3, 6, 9, 250, 499):
        a = rp.ModelBank([s])
        a.fit_predict(df.slice(0, cut))
        buf = io.BytesIO()
        a.save(buf)
        b = rp.ModelBank.load_bytes(buf.getvalue(), specs=[s])
        rest = df.slice(cut)
        assert a.fit_predict(rest).equals(b.fit_predict(rest), null_equal=True), cut


def test_a_weight_scales_nothing_but_the_weight():
    """Every weight doubled leaves the predictions and the fit where they
    were, to rounding; ``weight_sum`` doubles."""
    df = _series(800, 12).with_columns(pl.Series("w", np.linspace(0.5, 2.0, 800)))
    s = rp.spec.ar("a", targets=["y"], lags=2, weight="w", half_life=100.0, min_weight=0)
    a = _out(s, df)
    b = _out(s, df.with_columns(pl.col("w") * 2.0))
    np.testing.assert_allclose(a["pred_y"].to_numpy(), b["pred_y"].to_numpy(), rtol=1e-9)
    np.testing.assert_allclose(b["weight_sum"].to_numpy(), 2 * a["weight_sum"].to_numpy())


def test_names_match_the_realized_struct():
    df = _series(200, 13).with_columns(pl.col("y").alias("z"))
    for kw in ({}, {"features": ["x"]}, {"fit_intercept": False}):
        spec = rp.spec.ar("a", targets=["y", "z"], lags=[1, 3], **kw)
        out = rp.ModelBank([spec]).fit_predict(df)
        assert [f.name for f in out.schema["a"].fields] == rp.spec.output_fields(spec)
    terms = rp.spec.coef_index(rp.spec.ar("a", targets=["y"], lags=[1, 3], features=["x"]))
    assert terms["term"].to_list() == ["intercept", "lag1", "lag3", "x"]


def test_the_state_cap_on_both_sides():
    """An AR(3) of one target and no features holds at most
    ``8·3 + 8(4·4² + 16·4 + 64)`` = 1,560 bytes: taken at that cap, refused
    a byte under it."""
    ok = rp.spec.ar("a", targets=["y"], lags=3, max_state_bytes=1560)
    rp.ModelBank([ok])
    with pytest.raises(ValueError, match="past max_state_bytes of 1559"):
        rp.ModelBank([rp.spec.ar("a", targets=["y"], lags=3, max_state_bytes=1559)])


@pytest.mark.parametrize(
    ("kw", "says"),
    [
        ({"lags": []}, "at least one lag"),
        ({"lags": [0, 1]}, "at least 1"),
        ({"lags": [2, 1]}, "strictly increasing"),
        ({"lags": [1, 2**20 + 1]}, "at most"),
        ({"lags": 2, "ridge": -1.0}, "ridge must be finite and at least 0"),
        ({"lags": 2, "max_state_bytes": 0}, "above 0"),
        ({"lags": [1, 2**20]}, "past max_state_bytes"),
        ({"lags": 2, "emit_se_coef": True}, "emit_se_coef"),
    ],
)
def test_refusals(kw, says):
    with pytest.raises(ValueError, match=says):
        rp.ModelBank([rp.spec.ar("a", targets=["y"], **kw)])
