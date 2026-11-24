"""``garch`` with inputs, GARCH-X, through the bank (task 559, part 4): the
variance equation against its definition written out in numpy, row by row,
at the parameters the bank reports; the forecasts with the inputs held at
their learned mean; a negative input under GARCH and GJR withheld as
``input_negative``; and what the bank owes every model -- chunk invariance,
a save mid-stream, zero weights, its refusals and its coefficient names.

The definition, with ``x_t`` the row's own inputs:

    garch-X   s_t = ω + α e²_{t-1} + β s_{t-1} + π'x_t
    gjr-X     s_t = ω + α e²_{t-1} + γ 1[e_{t-1} < 0] e²_{t-1} + β s_{t-1} + π'x_t
    egarch-X  ln s_t = ω + α (|z_{t-1}| - √(2/π)) + γ z_{t-1} + β ln s_{t-1} + π'x_t

``rugarch`` 1.5-6 was measured against this definition once, at fixed
parameters on 1,500 rows (docs/records/HORIZONS.md §16): it reads the input
of the row whose variance it forms, as here, and agreed to 7.2e-16. The
repository's CI has no R, so that check is a record, not a test.
``scipy``'s batch maximum-likelihood fit is the oracle for the estimates,
in ``tests/test_second_opinion_garch_x.py``.
"""

from __future__ import annotations

import io
import math

import numpy as np
import polars as pl
import pytest

from rheon import polars as rp

TIER = "essential"

SQRT_2_PI = math.sqrt(2.0 / math.pi)

TRUTH = {
    "garch": [0.02, 0.05, 0.85, 0.1, 0.04],
    "gjr": [0.02, 0.03, 0.06, 0.85, 0.1, 0.04],
    "egarch": [-0.02, 0.15, -0.08, 0.95, 0.05, -0.03],
}


def _inputs(vol: str, n: int, seed: int, q: int = 2) -> np.ndarray:
    """``q`` AR(1) inputs of unit variance, coefficient 0.9: halved squares
    (at least 0) under GARCH and GJR, as they are under EGARCH."""
    rng = np.random.default_rng(seed)
    a = np.zeros(q)
    out = np.empty((n, q))
    for t in range(n):
        a = 0.9 * a + math.sqrt(1 - 0.81) * rng.standard_normal(q)
        out[t] = a if vol == "egarch" else 0.5 * a * a
    return out


def _simulate(vol: str, theta: list[float], x: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    k = 3 if vol == "garch" else 4
    n = len(x)
    e, s, ln_s = 0.0, 1.0, 0.0
    out = np.empty(n)
    for t in range(n):
        px = float(np.dot(theta[k:], x[t]))
        if vol == "egarch":
            news = 0.0 if t == 0 else theta[1] * (abs(e / math.sqrt(s)) - SQRT_2_PI)
            news += 0.0 if t == 0 else theta[2] * e / math.sqrt(s)
            ln_s = theta[0] + news + theta[3] * ln_s + px
            s = math.exp(ln_s)
        else:
            g = theta[2] if k == 4 and e < 0 else 0.0
            s = theta[0] + (theta[1] + g) * e * e + theta[k - 1] * s + px
        e = math.sqrt(s) * rng.standard_normal()
        out[t] = e
    return out


def _frame(vol: str, n: int, seed: int) -> pl.DataFrame:
    x = _inputs(vol, n, seed)
    y = _simulate(vol, TRUTH[vol], x, seed + 1)
    rng = np.random.default_rng(seed + 2)
    return pl.DataFrame(
        {
            "g": np.where(np.arange(n) % 3 == 0, "a", "b"),
            "y": y,
            "x1": x[:, 0],
            "x2": x[:, 1],
            "w": np.where(rng.uniform(size=n) < 0.1, 0.0, 1.0 + rng.integers(0, 3, n)),
        }
    )


def _spec(vol: str, **kw) -> dict:
    return rp.spec.garch(
        "m", targets=["y"], features=["x1", "x2"], vol=vol, **{"start_rows": 500, **kw}
    )


def _out(spec: dict, df: pl.DataFrame) -> pl.DataFrame:
    return rp.ModelBank([spec]).fit_predict(df).unnest("m")


@pytest.mark.parametrize("vol", ["garch", "gjr", "egarch"])
def test_the_variance_is_the_definition_at_the_reported_parameters(vol):
    """Each row's variance is the equation from the row before and the
    row's own inputs, at the parameters the bank reported after the row
    before: to the bit's neighbourhood (`1e-13`), the terms in the bank's
    order under GARCH and GJR, EGARCH's `exp` and `ln` the platform's."""
    df = _frame(vol, 4000, 3)
    out = _out(_spec(vol, coef_every=0), df)
    y, s = df["y"].to_numpy(), out["variance_y"].to_numpy()
    x = df.select("x1", "x2").to_numpy()
    coef = out["coef"].to_list()
    k = 3 if vol == "garch" else 4
    checked = 0
    for t in range(600, 3999):
        th = coef[t]
        if th is None or not np.isfinite(s[t]):
            continue
        e = y[t]
        if vol == "egarch":
            z = e / math.sqrt(s[t])
            ln = th[0] + th[1] * (abs(z) - SQRT_2_PI) + th[2] * z + th[3] * math.log(s[t])
            want = math.exp(ln + th[4] * x[t + 1, 0] + th[5] * x[t + 1, 1])
            tol = 1e-13
        else:
            want = th[0] + th[1] * e * e
            if k == 4:
                want += th[2] * (e * e if e < 0 else 0.0)
            want += th[k - 1] * s[t]
            want += th[k] * x[t + 1, 0]
            want += th[k + 1] * x[t + 1, 1]
            # the terms in the bank's order; within two units of the last
            # place, a product's rounding in numpy's scalars
            tol = 4.5e-16
        assert abs(s[t + 1] - want) <= tol * want, (t, s[t + 1], want)
        checked += 1
    assert checked > 1500


@pytest.mark.parametrize("vol", ["garch", "gjr"])
def test_the_forecasts_hold_the_inputs_at_their_learned_mean(vol):
    """Past the row, every input is its weighted mean over the rows
    learned, ``x̄``: horizon ``h`` is ``ω + π'x̄ + P f`` run ``h - 1`` rows
    from the row's variance, which read the row's own inputs. Every row
    weighs 1 here and none is skipped, so ``x̄`` after row ``t`` is the mean of
    rows ``0 .. t``, the start's included, to a rounding of the running
    mean."""
    df = _frame(vol, 1500, 5).drop("w")
    hs = [1, 3, 50]
    out = _out(_spec(vol, horizons=hs, coef_every=0), df)
    s, coef = out["variance_y"], out["coef"].to_list()
    x = df.select("x1", "x2").to_numpy()
    k = 3 if vol == "garch" else 4
    for t in (900, 1200, 1498):
        th = coef[t]
        xbar = x[: t + 1].mean(axis=0)
        omega = th[0] + th[k] * xbar[0] + th[k + 1] * xbar[1]
        p = th[1] + 0.5 * th[2] + th[3] if k == 4 else th[1] + th[2]
        f = [s[t + 1]]
        for _ in range(max(hs) - 1):
            f.append(omega + p * f[-1])
        for h in hs:
            got = out[f"variance_h{h}_y"][t + 1]
            assert abs(got - f[h - 1]) <= 1e-12 * f[h - 1], (t, h, got, f[h - 1])
            want = math.fsum(f[:h])
            assert abs(out[f"variance_sum_h{h}_y"][t + 1] - want) <= 1e-12 * want
        assert out["variance_h1_y"][t + 1] == s[t + 1]


@pytest.mark.parametrize("vol", ["garch", "gjr", "egarch"])
def test_a_negative_input_is_withheld_under_garch_and_gjr(vol):
    """Under GARCH and GJR an input below 0 is refused: the row reports
    nothing, ``withheld_reason`` says ``input_negative``, and it is learned as
    a row of weight 0, so every other row is what the stream with that row
    at weight 0 gives. Under EGARCH an input takes any sign."""
    n = 3000
    df = _frame(vol, n, 7).drop("g").with_columns(pl.lit(1.0).alias("w"))
    # past the start, its doublings included
    bad = [2500, 2501, 2800]
    x1 = df["x1"].to_numpy().copy()
    x1[bad] = -0.25
    neg = df.with_columns(pl.Series("x1", x1))
    s = _spec(vol, weight="w", coef_every=0)
    got = _out(s, neg)
    if vol == "egarch":
        assert got["variance_y"][bad].is_not_null().all()
        assert got["withheld_reason"].drop_nulls().ne("input_negative").all()
        return
    w = np.ones(n)
    w[bad] = 0.0
    zero = _out(s, neg.with_columns(pl.Series("w", w), pl.Series("x1", np.where(w > 0, x1, 0.0))))
    assert got["withheld_reason"][bad].to_list() == ["input_negative"] * 3
    assert got["variance_y"][bad].is_null().all()
    assert got["std_resid_y"][bad].is_null().all()
    keep = pl.Series(w > 0)
    for f in ("variance_y", "std_resid_y", "coef", "weight_sum"):
        assert got[f].filter(keep).equals(zero[f].filter(keep), null_equal=True), f
    assert got["weight_sum"][bad].to_list() == zero["weight_sum"][bad].to_list()
    # a reason the start's own rows outrank: before the start, they are
    # named as the start's
    early = neg.with_columns(
        pl.when(pl.int_range(n) == 10).then(-1.0).otherwise(pl.col("x1")).alias("x1")
    )
    assert _out(s, early)["withheld_reason"][10] == "below_warm_rows"


@pytest.mark.parametrize("vol", ["garch", "egarch"])
def test_chunk_invariance(vol):
    df = _frame(vol, 1600, 9)
    s = _spec(vol, group="g", weight="w", coef_every=0, horizons=[1, 4])
    one = rp.ModelBank([s]).fit_predict(df)
    for size in (1, 37, 409):
        bank = rp.ModelBank([s])
        many = pl.concat([bank.fit_predict(df.slice(i, size)) for i in range(0, len(df), size)])
        assert one.equals(many, null_equal=True), size


@pytest.mark.parametrize("vol", ["gjr", "egarch"])
def test_a_save_goes_on_to_the_bit(vol):
    """A save mid-start, holding the start's inputs, and after it."""
    df = _frame(vol, 1600, 11)
    s = _spec(vol, group="g", weight="w", horizons=[2])
    for cut in (1, 600, 1400):
        a = rp.ModelBank([s])
        a.fit_predict(df.slice(0, cut))
        buf = io.BytesIO()
        a.save(buf)
        b = rp.ModelBank.load_bytes(buf.getvalue(), specs=[s])
        rest = df.slice(cut)
        assert a.fit_predict(rest).equals(b.fit_predict(rest), null_equal=True), cut


def test_a_rows_variance_reads_its_own_input_and_no_later_one():
    """The row's input moves its own variance (it is known before the
    row's return), and no earlier row's: out of sample in the return, in
    the input's timing as ``rugarch``'s."""
    df = _frame("garch", 900, 13).drop("g", "w")
    a = _out(_spec("garch"), df)
    moved = df.with_columns(
        pl.when(pl.int_range(900) == 800).then(5.0).otherwise(pl.col("x1")).alias("x1")
    )
    b = _out(_spec("garch"), moved)
    assert a["variance_y"][:800].equals(b["variance_y"][:800])
    assert b["variance_y"][800] > a["variance_y"][800]


def test_coef_names_each_input():
    df = _frame("gjr", 700, 15)
    bank = rp.ModelBank([_spec("gjr", dist="t")])
    bank.fit_predict(df.drop("g", "w"))
    assert rp.spec.coef_index(_spec("gjr", dist="t"))["term"].to_list() == [
        "omega",
        "alpha",
        "gamma",
        "beta",
        "pi_x1",
        "pi_x2",
        "nu",
    ]


@pytest.mark.parametrize(
    ("kw", "says"),
    [
        ({"vol": "figarch", "truncation": 30, "start_rows": 2000}, "figarch"),
        ({"variance_targeting": True}, "variance_targeting"),
    ],
)
def test_refusals(kw, says):
    with pytest.raises(ValueError, match=says):
        rp.ModelBank([rp.spec.garch("m", targets=["y"], features=["x"], **kw)])


def test_at_most_eight_inputs():
    with pytest.raises(ValueError, match="at most 8"):
        rp.ModelBank([rp.spec.garch("m", targets=["y"], features=[f"x{i}" for i in range(9)])])
