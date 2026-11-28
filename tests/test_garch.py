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
from arch import arch_model
from arch.univariate.recursions import figarch_weights

import rheon
from rheon import polars as rp

TIER = "essential"


def _returns(
    n: int, seed: int, gamma: float = 0.0, alpha: float = 0.08, beta: float = 0.9
) -> np.ndarray:
    """A GJR series at ``(0.05, α, γ, β - γ/2)``, normal innovations: by
    default ``(0.05, 0.08, γ, 0.9 - γ/2)``."""
    rng = np.random.default_rng(seed)
    w, a, b = 0.05, alpha, beta - gamma / 2
    s, e, out = 1.0, 0.0, np.empty(n)
    for t, z in enumerate(rng.standard_normal(n + 300)):
        s = w + (a + (gamma if e < 0 else 0.0)) * e * e + b * s
        e = math.sqrt(s) * z
        if t >= 300:
            out[t - 300] = e
    return out


def _frame(n: int, seed: int, gamma: float = 0.0, **process) -> pl.DataFrame:
    rng = np.random.default_rng(seed + 1)
    y = _returns(n, seed, gamma, **process)
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


#: A process whose variance clusters strongly with finite fourth moment,
#: ``α = 0.2``, ``β = 0.75``: a start of 100 rows, doubled, identifies it in
#: each group of a stream of 3,000 rows (including the 1/3-share group ``a``).
CLUSTERED = {"alpha": 0.2, "beta": 0.75}


def _reports_in_every_group(out: pl.DataFrame, df: pl.DataFrame) -> None:
    """Each group's variance is a number on some row: the comparisons over
    it are of numbers, not of the nulls of a start that never came."""
    m = out.unnest("m") if "m" in out.columns else out
    counts = (
        m.select("variance_y")
        .with_columns(df["g"])
        .group_by("g")
        .agg(pl.col("variance_y").is_not_null().sum())
    )
    assert counts["variance_y"].min() > 0, counts.sort("g").rows()


def _next_variance(vol: str, th: list[float], y: float, s: float) -> float:
    """The process's variance for the row after one of return ``y`` and
    variance ``s``, at its parameters ``th`` (a t's shape after them)."""
    if vol == "garch":
        return th[0] + th[1] * y * y + th[2] * s
    if vol == "gjr":
        return th[0] + (th[1] + (th[2] if y < 0 else 0.0)) * y * y + th[3] * s
    z = y / math.sqrt(s)
    return math.exp(
        th[0] + th[1] * (abs(z) - math.sqrt(2.0 / math.pi)) + th[2] * z + th[3] * math.log(s)
    )


@pytest.mark.parametrize(
    "kw",
    [
        {},
        {"dist": "t"},
        {"dist": "skewt"},
        {"variance_targeting": True},
        {"vol": "gjr"},
        {"vol": "gjr", "dist": "t"},
        {"vol": "egarch"},
    ],
    ids=["garch", "garch_t", "garch_skewt", "targeted", "gjr", "gjr_t", "egarch"],
)
def test_the_variance_is_the_recursion_at_the_reported_parameters(kw):
    """Each row's variance is the recursion from the row before, at the
    parameters the bank reported after that row, to rounding: each process,
    each innovation it takes (a t's and a skewed t's shape follow the
    process's parameters, and move no variance), and the targeted form,
    whose reported ``ω`` is ``v̄ (1 − α − β)`` at that row."""
    vol = kw.get("vol", "garch")
    gamma = 0.1 if vol != "garch" else 0.0
    # a series whose start of 100 rows, doubled as needed, tells its
    # process from a constant variance (a start that does not waits)
    n = 2400 if kw.get("variance_targeting") else 1600
    df = pl.DataFrame({"y": _returns(n, 9, gamma)})
    out = _out(_spec(coef_every=0, **kw), df)
    y, s = df["y"].to_numpy(), out["variance_y"].to_numpy()
    coef = out["coef"].to_list()
    width = {"garch": 3, "gjr": 4, "egarch": 4}[vol] + {"t": 1, "skewt": 2}.get(kw.get("dist"), 0)
    checked = 0
    for t in range(200, n - 1):
        th = coef[t]
        if th is None or not np.isfinite(s[t]):
            continue
        assert len(th) == width, th
        want = _next_variance(vol, th, y[t], s[t])
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
    df = pl.DataFrame({"y": _returns(1000, 9)})
    s = _spec()
    a = _out(s, df)
    b = _out(
        s,
        df.with_columns(
            pl.when(pl.int_range(1000) == 900).then(9.0).otherwise(pl.col("y")).alias("y")
        ),
    )
    assert a["variance_y"][:901].equals(b["variance_y"][:901])
    assert a["std_resid_y"][900] != b["std_resid_y"][900]
    assert a["variance_y"][901] != b["variance_y"][901]
    # nothing before the start, a number on every row from it
    n0 = int(np.argmax(a["variance_y"].is_not_null().to_numpy()))
    assert 100 <= n0 <= 900, n0
    assert a["variance_y"][:n0].is_null().all() and a["variance_y"][n0:].is_not_null().all()


@pytest.mark.parametrize(
    ("kw", "seed"),
    [
        ({}, 6),
        ({"vol": "egarch", "forecast_rows": [3, 1, 12]}, 6),
        ({"vol": "gjr", "dist": "t", "forecast_rows": {"week": 5, "now": 1}}, 7),
        ({"vol": "figarch", "truncation": 30, "start_rows": 2000, "forecast_rows": [4]}, 6),
    ],
)
def test_chunk_invariance(kw, seed):
    n = 4400 if kw.get("vol") == "figarch" else 3000
    df = _frame(n, seed, 0.1 if "vol" in kw else 0.0, **CLUSTERED)
    s = _spec(group="g", weight="w", coef_every=0, **kw)
    one = rp.ModelBank([s]).fit_predict(df)
    if n == 3000:  # figarch's start of 2000 rows is past a third of 4400
        _reports_in_every_group(one, df)
    for size in (1, 13, 211) if n == 3000 else (211, 997):
        bank = rp.ModelBank([s])
        many = pl.concat([bank.fit_predict(df.slice(i, size)) for i in range(0, n, size)])
        assert one.equals(many, null_equal=True), size


@pytest.mark.parametrize(
    "kw",
    [
        {},
        {"dist": "skewt", "forecast_rows": [2, 9]},
        {"variance_targeting": True},
        {"vol": "egarch", "forecast_rows": {"day": 1, "fortnight": 10}},
    ],
)
def test_a_save_goes_on_to_the_bit(kw):
    df = _frame(3000, 6, **CLUSTERED)
    s = _spec(group="g", weight="w", **kw)
    _reports_in_every_group(rp.ModelBank([s]).fit_predict(df), df)
    # a save before either group's start, after one's, after both, and on
    # the last row
    for cut in (1, 600, 1500, 2700, 2999):
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
        {"forecast_rows": [1, 4]},
        {"vol": "figarch", "truncation": 30, "start_rows": 2000, "forecast_rows": [3]},
        {"vol": "egarch", "forecast_rows": [2, 6]},
    ],
)
def test_a_zero_weight_row_steps_no_recursion(kw):
    """The stream with rows of weight 0, and rows whose return is null,
    gives every other row, to the bit, the output of the stream without
    them: the recursion is per observation, and such a row is none, so a
    horizon counts no such row either. They are still read, and their
    forecasts are those the next learned row reads from the same state."""
    n = 2600 if "vol" in kw else 1200
    df = _frame(n, 8).drop("g")
    df = df.with_columns(
        pl.when(pl.int_range(n) % 17 == 5).then(None).otherwise(pl.col("y")).alias("y")
    )
    s = _spec(weight="w", coef_every=0, **kw)
    full = _out(s, df)
    keep = ((df["w"] > 0) & df["y"].is_not_null()).to_numpy()
    thin = _out(s, df.filter(pl.Series(keep)))
    labels = [f"h{h}" for h in kw["forecast_rows"]]
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
    df = _frame(1400, 9).drop("g")
    s = _spec(weight="w", coef_every=0)
    a = _out(s, df)
    b = _out(s, df.with_columns(pl.col("w") * 2.0))
    # the same rows report, and they are numbers: `assert_allclose` passes
    # NaN against NaN, so a start that never came would pass it too
    assert a["variance_y"].is_null().equals(b["variance_y"].is_null())
    assert a["variance_y"].is_not_null().sum() > 300
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
    out = _out(_spec(vol=vol, forecast_rows=hs, coef_every=0), df)
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
    spec = _spec(forecast_rows=[1, 3, 25], vol="gjr")
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
        {"forecast_rows": [3]},
        {"vol": "gjr", "dist": "t", "forecast_rows": {"week": 5, "day": 1}},
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
        ({"forecast_rows": [0]}, "each of forecast_rows must be at least 1"),
        ({"forecast_rows": [3, 1, 3]}, "holds 3 twice"),
        ({"forecast_rows": list(range(1, 66))}, "at most 64"),
        ({"forecast_rows": {"sum5": 5}}, 'not begin with "sum"'),
        ({"forecast_rows": {"a": 2, "b": 2}}, "holds 2 twice"),
        ({"forecast_rows": [300_000]}, "would take 1200000 multiply-adds a target a row"),
        ({"vol": "egarch", "forecast_rows": [7000]}, "Lower the largest horizon to 6393"),
        ({"max_horizon_work": 0}, "max_horizon_work must be above 0"),
        ({"max_horizon_work": 2**30 + 1}, "at most 1073741824"),
        ({"vol": "gjr", "variance_targeting": True}, "variance_targeting"),
        ({"truncation": 100}, "truncation is FIGARCH's"),
        ({"vol": "figarch", "start_rows": 0}, "at least 1"),
        ({"start_rows": 0}, "at least 1"),
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
    n = 600
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
    # the 121st row is the first two hours from the first; a try that does
    # not start waits for as many new rows as it held, 121, then doubles,
    # whatever the clock says: the first report is just past a doubling step
    assert first in (121, 242, 484), first
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
    default adaptive doubling start (from 64 rows): the initial rows are
    withheld as ``below_warm_rows`` (and ``not_identified`` while doubling),
    with no coefficients; once identified, every variance is a number and the
    recursion at the parameters reported after the row before, and nothing is
    withheld."""
    df = pl.DataFrame({"y": _returns(2600, 21)})
    out = _out(rp.spec.garch("m", targets=["y"], coef_every=0), df)
    s, y, coef = out["variance_y"].to_numpy(), df["y"].to_numpy(), out["coef"].to_list()
    reason = out["withheld_reason"].cast(pl.String).to_list()
    n0 = int(np.argmax(np.isfinite(s)))
    assert 64 <= n0 < 2000 and n0 % 64 == 0 and (n0 // 64) & (n0 // 64 - 1) == 0, n0
    assert np.isnan(s[:n0]).all() and np.isfinite(s[n0:]).all()
    assert reason[:64] == ["below_warm_rows"] * 64
    assert all(r in ("below_warm_rows", "not_identified") for r in reason[64:n0])
    assert reason[n0:] == [None] * (2600 - n0)

    def numbers(c) -> bool:  # noqa: ANN001
        return c is not None and all(v is not None for v in c)

    assert not any(numbers(c) for c in coef[: n0 - 1]) and all(numbers(c) for c in coef[n0 - 1 :])
    # The first variance reported is the start's filter run over its `n0`
    # rows at its fit, from `arch`'s backcast: `arch`'s own recursion at
    # those parameters.
    fixed = arch_model(y[: n0 + 1], mean="Zero", vol="GARCH", p=1, q=1, rescale=False)
    first = float(fixed.fix(coef[n0 - 1]).conditional_volatility[n0] ** 2)
    assert abs(s[n0] - first) <= 1e-12 * first, (s[n0], first)
    for t in range(n0, 2599):
        om, a, b = coef[t]
        want = om + a * y[t] ** 2 + b * s[t]
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


def test_figarch_short_start_waits_as_not_identified_without_row_floor():
    """FIGARCH accepts ``start_rows`` down to 1 and closes ``start_span`` on
    the span itself (no 50-row or 2,000-row floor), withholding short starts
    as ``not_identified`` while the standard error of ``d`` exceeds
    ``1 / (2 z_{0.9995})``."""
    y = _returns(300, 6, alpha=0.2, beta=0.78)
    df = pl.DataFrame({"t": np.arange(len(y), dtype=float), "y": y})
    # `start_rows` below 50 is accepted and withheld as `not_identified`
    out_rows = _out(
        rp.spec.garch("m", targets=["y"], vol="figarch", truncation=30, start_rows=10), df
    )
    reason_rows = out_rows["withheld_reason"].cast(pl.String).to_list()
    assert out_rows["variance_y"].is_null().all()
    assert reason_rows[:10] == ["below_warm_rows"] * 10
    assert "not_identified" in reason_rows[10:]
    # `start_span` closing at 21 rows tries on the span (not at 50 rows)
    out_span = _out(
        rp.spec.garch(
            "m",
            targets=["y"],
            vol="figarch",
            truncation=30,
            clock="t",
            gap_cap=1e9,
            start_span=20.0,
        ),
        df,
    )
    reason_span = out_span["withheld_reason"].cast(pl.String).to_list()
    assert out_span["variance_y"].is_null().all()
    assert reason_span[:21] == ["below_warm_rows"] * 21
    assert reason_span[21] == "not_identified"


@pytest.mark.parametrize("vol", ["garch", "gjr", "egarch"])
def test_short_start_waits_as_not_identified_without_row_floor(vol):
    """Every vol form accepts ``start_rows`` down to 1 and closes
    ``start_span`` without a 50-row floor, withholding short starts as
    ``not_identified`` under the LR and Fisher-SE gates."""
    y = _returns(60, 6, gamma=0.1 if vol != "garch" else 0.0, alpha=0.2, beta=0.78)
    df = pl.DataFrame({"t": np.arange(len(y), dtype=float), "y": y})
    out_rows = _out(rp.spec.garch("m", targets=["y"], vol=vol, start_rows=10), df)
    reason_rows = out_rows["withheld_reason"].cast(pl.String).to_list()
    assert reason_rows[:10] == ["below_warm_rows"] * 10
    assert reason_rows[10] == "not_identified"
    out_span = _out(
        rp.spec.garch("m", targets=["y"], vol=vol, clock="t", gap_cap=1e9, start_span=20.0), df
    )
    reason_span = out_span["withheld_reason"].cast(pl.String).to_list()
    assert reason_span[:21] == ["below_warm_rows"] * 21
    assert reason_span[21] == "not_identified"
