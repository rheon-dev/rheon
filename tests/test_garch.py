"""``garch`` through the bank: the variance recursion against its
definition, row by row, at the parameters the bank reports, and what the
bank owes every model -- out-of-sample reads, chunk invariance, a save
mid-stream, zero weights and nulls, weights as relative evidence, its
refusals, its output schema, and a start over a span of the clock held to
the group's state cap (task 492).

The oracle for the recursion is the definition written out in numpy:
``s_{t+1} = ω + α e_t² + γ 1[e_t < 0] e_t² + β s_t`` at the parameters the
bank reports after row ``t`` (``coef_every=0``), and FIGARCH's
``ω/(1 - β) + Σ λ_j e²_{t+1-j}`` with ``arch``'s own ``figarch_weights``.
``arch``'s batch fits are the oracle for the estimator, in
``tests/test_second_opinion_garch.py``.
"""

from __future__ import annotations

import io
import math
import warnings

import numpy as np
import polars as pl
import pytest
from arch.univariate.recursions import figarch_weights

import rheon
from rheon import polars as rp

TIER = "essential"


def _returns(n: int, seed: int, gamma: float = 0.0) -> np.ndarray:
    """A GJR series at ``(0.05, 0.08, γ, 0.9 - γ/2)``, normal innovations."""
    rng = np.random.default_rng(seed)
    w, a, b = 0.05, 0.08, 0.9 - gamma / 2
    s, e, out = 1.0, 0.0, np.empty(n)
    for t, z in enumerate(rng.standard_normal(n + 300)):
        s = w + (a + (gamma if e < 0 else 0.0)) * e * e + b * s
        e = math.sqrt(s) * z
        if t >= 300:
            out[t - 300] = e
    return out


def _frame(n: int, seed: int, gamma: float = 0.0) -> pl.DataFrame:
    rng = np.random.default_rng(seed + 1)
    y = _returns(n, seed, gamma)
    return pl.DataFrame(
        {
            "t": np.arange(n, dtype=float),
            "g": np.where(np.arange(n) % 3 == 0, "a", "b"),
            "y": y,
            "w": np.where(rng.uniform(size=n) < 0.1, 0.0, 1.0 + rng.integers(0, 3, n)),
        }
    )


def _spec(**kw) -> dict:
    return rp.spec.garch("m", targets=["y"], **{"start_rows": 100, **kw})


def _out(spec: dict, df: pl.DataFrame) -> pl.DataFrame:
    return rp.ModelBank([spec]).fit_predict(df).unnest("m")


@pytest.mark.parametrize("kw", [{}, {"vol": "gjr"}, {"vol": "gjr", "dist": "t"}])
def test_the_variance_is_the_recursion_at_the_reported_parameters(kw):
    """Each row's variance is the recursion from the row before, at the
    parameters the bank reported after that row, to rounding."""
    gamma = 0.1 if "vol" in kw else 0.0
    # a series whose start of 100 rows, doubled at most twice, tells its
    # process from a constant variance (a start that does not waits)
    df = pl.DataFrame({"y": _returns(1200, 9, gamma)})
    out = _out(_spec(coef_every=0, **kw), df)
    y, s = df["y"].to_numpy(), out["variance_y"].to_numpy()
    coef = out["coef"].to_list()
    checked = 0
    for t in range(200, 1199):
        th = coef[t]
        if th is None or not np.isfinite(s[t]):
            continue
        e2 = y[t] * y[t]
        if len(th) == 3 or "dist" in kw and len(th) == 4 and "vol" not in kw:
            want = th[0] + th[1] * e2 + th[2] * s[t]
        else:
            want = th[0] + (th[1] + (th[2] if y[t] < 0 else 0.0)) * e2 + th[3] * s[t]
        assert abs(s[t + 1] - want) <= 1e-13 * want, (t, s[t + 1], want)
        checked += 1
    assert checked > 700


def test_figarch_is_its_weights_over_the_squared_returns():
    """FIGARCH's variance at the parameters reported after the row before:
    ``arch``'s weights over the last ``M`` squared returns, the lags before
    the start's first row read as the backcast (none here: the row checked
    is past ``M`` rows)."""
    m = 50
    y = _returns(2400, 4)
    out = _out(
        _spec(vol="figarch", truncation=m, start_rows=2000, coef_every=0), pl.DataFrame({"y": y})
    )
    s, coef = out["variance_y"].to_numpy(), out["coef"].to_list()
    for t in (2100, 2250, 2398):
        om, phi, d, beta = coef[t]
        lam = figarch_weights(np.array([phi, d, beta]), 1, 1, m)
        want = om / (1 - beta) + float(np.dot(lam, y[t + 1 - np.arange(1, m + 1)] ** 2))
        assert abs(s[t + 1] - want) <= 1e-12 * want, (t, s[t + 1], want)


def test_a_rows_variance_is_read_before_the_row():
    """The variance a row reports does not read the row's return: changing
    it moves the row's standardized residual, and the variance of the next
    row, alone."""
    df = pl.DataFrame({"y": _returns(400, 9)})
    s = _spec()
    a = _out(s, df)
    b = _out(
        s,
        df.with_columns(
            pl.when(pl.int_range(400) == 300).then(9.0).otherwise(pl.col("y")).alias("y")
        ),
    )
    assert a["variance_y"][:301].equals(b["variance_y"][:301])
    assert a["std_resid_y"][300] != b["std_resid_y"][300]
    assert a["variance_y"][301] != b["variance_y"][301]
    # nothing before the start, a number on every row from it
    n0 = int(np.argmax(a["variance_y"].is_not_null().to_numpy()))
    assert 100 <= n0 <= 300, n0
    assert a["variance_y"][:n0].is_null().all() and a["variance_y"][n0:].is_not_null().all()


@pytest.mark.parametrize(
    "kw",
    [
        {},
        {"vol": "egarch", "horizons": [3, 1, 12]},
        {"vol": "gjr", "dist": "t", "horizons": {"week": 5, "now": 1}},
        {"vol": "figarch", "truncation": 30, "start_rows": 2000, "horizons": [4]},
    ],
)
def test_chunk_invariance(kw):
    n = 4400 if kw.get("vol") == "figarch" else 700
    df = _frame(n, 6)
    s = _spec(group="g", weight="w", coef_every=0, **kw)
    one = rp.ModelBank([s]).fit_predict(df)
    for size in (1, 13, 211) if n == 700 else (211, 997):
        bank = rp.ModelBank([s])
        many = pl.concat([bank.fit_predict(df.slice(i, size)) for i in range(0, n, size)])
        assert one.equals(many, null_equal=True), size


@pytest.mark.parametrize(
    "kw",
    [
        {},
        {"dist": "skewt", "horizons": [2, 9]},
        {"variance_targeting": True},
        {"vol": "egarch", "horizons": {"day": 1, "fortnight": 10}},
    ],
)
def test_a_save_goes_on_to_the_bit(kw):
    df = _frame(600, 7)
    s = _spec(group="g", weight="w", **kw)
    for cut in (1, 150, 330, 599):
        a = rp.ModelBank([s])
        a.fit_predict(df.slice(0, cut))
        buf = io.BytesIO()
        a.save(buf)
        b = rp.ModelBank.load_bytes(buf.getvalue(), specs=[s])
        rest = df.slice(cut)
        assert a.fit_predict(rest).equals(b.fit_predict(rest), null_equal=True), cut


@pytest.mark.parametrize(
    "kw",
    [
        {"horizons": [1, 4]},
        {"vol": "figarch", "truncation": 30, "start_rows": 2000, "horizons": [3]},
        {"vol": "egarch", "horizons": [2, 6]},
    ],
)
def test_a_zero_weight_row_steps_no_recursion(kw):
    """The stream with rows of weight 0, and rows whose return is null,
    gives every other row, to the bit, the output of the stream without
    them: the recursion is per observation, and such a row is none, so a
    horizon counts no such row either. They are still read, and their
    forecasts are those the next learned row reads from the same state."""
    n = 2600 if "vol" in kw else 900
    df = _frame(n, 8).drop("g")
    df = df.with_columns(
        pl.when(pl.int_range(n) % 17 == 5).then(None).otherwise(pl.col("y")).alias("y")
    )
    s = _spec(weight="w", coef_every=0, **kw)
    full = _out(s, df)
    keep = ((df["w"] > 0) & df["y"].is_not_null()).to_numpy()
    thin = _out(s, df.filter(pl.Series(keep)))
    labels = [f"h{h}" for h in kw["horizons"]]
    forecasts = [f"variance{p}_{h}_y" for h in labels for p in ("", "_sum")]
    for f in ("variance_y", "std_resid_y", "coef", *forecasts):
        assert full[f].filter(pl.Series(keep)).equals(thin[f], null_equal=True), f
    read = full.filter(pl.Series(~keep))
    assert read["variance_y"].drop_nulls().len() > 20, "such a row is still read"
    # a row that is not learned reads the forecasts of the next learned row
    for i in np.flatnonzero(~keep[:-1] & keep[1:])[:50]:
        if full["variance_y"][int(i)] is None:
            continue
        for f in forecasts:
            assert full[f][int(i)] == full[f][int(i) + 1], (f, i)


def test_a_weight_scales_nothing_but_the_weight():
    """Every weight doubled leaves the variances and the fit where they
    were, to rounding; ``weight_sum`` doubles."""
    df = _frame(800, 9).drop("g")
    s = _spec(weight="w", coef_every=0)
    a = _out(s, df)
    b = _out(s, df.with_columns(pl.col("w") * 2.0))
    np.testing.assert_allclose(a["variance_y"].to_numpy(), b["variance_y"].to_numpy(), rtol=1e-9)
    np.testing.assert_allclose(b["weight_sum"].to_numpy(), 2 * a["weight_sum"].to_numpy())


def _definition(vol: str, theta: list[float], s: float, h: int) -> list[float]:
    """The variances ``f_1 .. f_h`` from a row whose variance is ``s``, by the
    recursion written out: each later row's ``e²`` read as its variance and
    GJR's asymmetric term as half of it, ``ω + P f`` with the persistence
    ``P = α + γ/2 + β`` formed once."""
    p = theta[1] + 0.5 * theta[2] + theta[3] if vol == "gjr" else theta[1] + theta[2]
    f = [s]
    for _ in range(h - 1):
        f.append(theta[0] + p * f[-1])
    return f


@pytest.mark.parametrize("vol", ["garch", "gjr"])
def test_the_forecasts_are_the_recursion_run_ahead(vol):
    """``variance_h<h>_<t>`` is the recursion run ``h - 1`` rows from the row's
    variance at the parameters reported after the row before, to the bit,
    and ``variance_sum_h<h>_<t>`` the variances' sum to within one rounding
    (``math.fsum``); horizon 1 is the row's variance itself."""
    df = pl.DataFrame({"y": _returns(600, 10, gamma=0.08 if vol == "gjr" else 0.0)})
    hs = [6, 1, 40]
    out = _out(_spec(vol=vol, horizons=hs, coef_every=0), df)
    s, coef = out["variance_y"], out["coef"].to_list()
    for t in (450, 520, 598):
        f = _definition(vol, coef[t], s[t + 1], max(hs))
        for h in hs:
            got = out[f"variance_h{h}_y"][t + 1]
            assert got == f[h - 1], (t, h, got, f[h - 1])
            want = math.fsum(f[:h])
            assert abs(out[f"variance_sum_h{h}_y"][t + 1] - want) <= math.ulp(want), (t, h)
        assert out["variance_h1_y"][t + 1] == s[t + 1]
        assert out["variance_sum_h1_y"][t + 1] == s[t + 1]
        # reverting toward the long-run variance ω / (1 - P)
        p = coef[t][1] + coef[t][-1] + (0.5 * coef[t][2] if vol == "gjr" else 0.0)
        long_run = coef[t][0] / (1.0 - p)
        assert abs(f[-1] - long_run) < abs(s[t + 1] - long_run)


def test_a_forecast_reads_no_row_it_forecasts():
    """Out of sample by construction: changing a row's return moves no
    forecast on that row or before it, at any horizon."""
    df = pl.DataFrame({"y": _returns(500, 12)})
    spec = _spec(horizons=[1, 3, 25], vol="gjr")
    a = _out(spec, df)
    for t in (300, 420):
        moved = df.with_columns(
            pl.when(pl.int_range(500) == t).then(pl.col("y") * 7.0 + 3.0).otherwise(pl.col("y"))
        )
        b = _out(spec, moved)
        for f in a.columns:
            if f.startswith("variance"):
                assert a[f][: t + 1].equals(b[f][: t + 1], null_equal=True), (f, t)
                assert not a[f][t + 1 :].equals(b[f][t + 1 :], null_equal=True), (f, t)


def test_names_match_the_realized_struct():
    df = _frame(300, 11).with_columns(pl.col("y").alias("z"))
    for kw in (
        {},
        {"horizons": [3]},
        {"vol": "gjr", "dist": "t", "horizons": {"week": 5, "day": 1}},
    ):
        spec = rp.spec.garch("m", targets=["y", "z"], start_rows=100, **kw)
        out = rp.ModelBank([spec]).fit_predict(df)
        assert [f.name for f in out.schema["m"].fields] == rp.spec.output_fields(spec)
    terms = rp.spec.coef_index(rp.spec.garch("m", targets=["y"], vol="gjr", dist="t"))["term"]
    assert terms.to_list() == ["omega", "alpha", "gamma", "beta", "nu"]
    terms = rp.spec.coef_index(rp.spec.garch("m", targets=["y"], vol="figarch"))["term"]
    assert terms.to_list() == ["omega", "phi", "d", "beta"]


@pytest.mark.parametrize(
    ("kw", "says"),
    [
        ({"half_life": 50.0}, "half_life does not apply to garch"),
        ({"vol": "arch"}, "vol"),
        ({"dist": "ged"}, "dist"),
        ({"vol": "gjr", "dist": "skewt"}, 'does not take vol = "gjr"'),
        ({"vol": "egarch", "dist": "t"}, "takes vol"),
        ({"horizons": [0]}, "each of horizons must be at least 1"),
        ({"horizons": [3, 1, 3]}, "holds 3 twice"),
        ({"horizons": list(range(1, 66))}, "at most 64"),
        ({"horizons": {"sum5": 5}}, 'not begin with "sum"'),
        ({"horizons": {"a": 2, "b": 2}}, "holds 2 twice"),
        ({"horizons": [300_000]}, "would take 1200000 multiply-adds a target a row"),
        ({"vol": "egarch", "horizons": [7000]}, "Lower the largest horizon to 6393"),
        ({"max_horizon_work": 0}, "max_horizon_work must be above 0"),
        ({"max_horizon_work": 2**30 + 1}, "at most 1073741824"),
        ({"vol": "gjr", "variance_targeting": True}, "variance_targeting"),
        ({"truncation": 100}, "truncation is FIGARCH's"),
        ({"vol": "figarch", "start_rows": 500}, "at least 2000"),
        ({"start_rows": 10}, "at least 50"),
        ({"start_span": 50.0}, "Leave one of them out"),
        ({"max_state_bytes": 0}, "above 0"),
        ({"emit_sigma": True}, "emit_sigma"),
    ],
)
def test_refusals(kw, says):
    with pytest.raises(ValueError, match=says):
        rp.ModelBank([_spec(**kw)])


def test_a_start_span_on_a_temporal_clock_is_a_duration():
    """``start_span`` as a duration on a temporal clock: rows a minute
    apart, a start of two hours closes at its 121st row."""
    n = 400
    df = pl.DataFrame(
        {
            "ts": pl.datetime_range(
                pl.datetime(2026, 1, 1),
                pl.datetime(2026, 1, 1) + pl.duration(minutes=n - 1),
                "1m",
                eager=True,
            ),
            "y": _returns(n, 12),
        }
    )
    spec = rp.spec.garch("m", targets=["y"], clock="ts", gap_cap="1h", start_span="2h")
    out = rp.ModelBank([spec]).fit_predict(df).unnest("m")
    first = out["variance_y"].is_not_null().arg_true()[0]
    # the 121st row is the first two hours from the first, and the fit may
    # double the span: the first report is just past one of the two
    assert first in (121, 241), first
    with pytest.raises(ValueError, match="start_span"):
        rp.ModelBank(
            [rp.spec.garch("m", targets=["y"], clock="ts", gap_cap="1h", start_span=120.0)]
        )


def test_a_clock_span_start_is_warned_of_once_and_refused_at_the_cap():
    """A start over a span of the clock buffers the rows the span has, which
    the data decides, so its bytes are checked as the rows go in: the first
    row to take a group past half of ``max_state_bytes`` warns, once,
    naming the group; the row that would take it past the cap stops the run
    with an error naming the group and the bytes."""
    # the state of 300 buffered rows: 168 fixed, 16 the model's, 16 a row
    cap = 168 + 16 + 16 * 300
    s = rp.spec.garch(
        "m", targets=["y"], group="g", clock="t", gap_cap=1e9, start_span=1e9, max_state_bytes=cap
    )

    def rows(group: str, n: int, first: int = 0) -> pl.DataFrame:
        i = np.arange(first, first + n)
        return pl.DataFrame({"g": group, "t": i.astype(float), "y": _returns(n, first + 1)})

    bank = rp.ModelBank([s])
    with pytest.warns(rheon.RheonWarning, match=r'group "a": the state holds \d+ bytes') as seen:
        bank.fit_predict(pl.concat([rows("a", 200), rows("b", 30)]))
    assert len(seen) == 1
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        bank.fit_predict(rows("a", 30, 200))
    with pytest.raises(Exception, match=r'group "a".*start_span.*above max_state_bytes of') as e:
        bank.fit_predict(rows("a", 200, 230))
    assert str(cap) in str(e.value)


def test_a_realistic_series_at_the_default_start():
    """A GARCH(1,1) at ``arch``'s oracle truth, ``(0.05, 0.08, 0.9)``, at the
    default start of 2,000 rows: the start's rows are withheld as the
    model's warm-up, ``below_warm_rows``, with no coefficients; from the next
    row every variance is a number and the recursion at the parameters
    reported after the row before, and nothing is withheld."""
    df = pl.DataFrame({"y": _returns(2600, 21)})
    out = _out(rp.spec.garch("m", targets=["y"], coef_every=0), df)
    s, y, coef = out["variance_y"].to_numpy(), df["y"].to_numpy(), out["coef"].to_list()
    reason = out["withheld_reason"].cast(pl.String).to_list()
    assert np.isnan(s[:2000]).all() and np.isfinite(s[2000:]).all()
    assert reason[:2000] == ["below_warm_rows"] * 2000 and reason[2000:] == [None] * 600

    def numbers(c) -> bool:  # noqa: ANN001
        return c is not None and all(v is not None for v in c)

    assert not any(numbers(c) for c in coef[:1999]) and all(numbers(c) for c in coef[1999:])
    for t in range(1999, 2599):
        om, a, b = coef[t]
        want = om + a * y[t] ** 2 + b * s[t] if t >= 2000 else s[t + 1]
        assert abs(s[t + 1] - want) <= 1e-13 * want, t
    om, a, b = coef[-1]
    assert 0.0 < a < 0.3 and 0.6 < b < 1.0 and a + b < 1.0, coef[-1]


@pytest.mark.parametrize("kw", [{}, {"vol": "gjr", "dist": "t"}])
def test_returns_with_no_variance_process_wait_and_say_so(kw):
    """Returns with no conditional variance: no start's likelihood-ratio
    statistic against a constant variance reaches its cut, so no row
    reports and no coefficient is written. The first start's rows are the
    warm-up, ``below_warm_rows``; every row after its fit is withheld as
    ``not_identified``, the doubled and the fresh buffers alike."""
    y = np.random.default_rng(22).standard_normal(5000)
    out = _out(rp.spec.garch("m", targets=["y"], start_rows=500, **kw), pl.DataFrame({"y": y}))
    reason = out["withheld_reason"].cast(pl.String).to_list()
    assert out["variance_y"].is_null().all()
    assert out["coef"].explode().is_null().all(), "a withheld fit is no coefficient"
    assert reason[:500] == ["below_warm_rows"] * 500
    assert reason[500:] == ["not_identified"] * 4500
