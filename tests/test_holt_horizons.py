"""``holt``'s forecasts at forecast_spans on the clock (task 559): the forecast a
span ``Δ`` past the row, ``l + b (s + Δ)``, and its integral over the span,
``Δ (l + b (s + Δ/2))``, read before the row from the level and trend the
rows before it left.

``statsmodels``' ``Holt`` is the oracle for the forecast: on a row-count
clock, once the weighted means' gains have settled to the fixed ones (the
control in ``tests/test_second_opinion.py``), its ``forecast(k)`` from the
rows before row ``t`` is the bank's horizon ``Δ = k - 1`` on row ``t``. The
integral is scipy's ``quad`` of the forecast as a function of the clock,
the line through ``statsmodels``' two first steps. The rest is what the bank
owes every output: chunk invariance, a save mid-stream, rows of weight 0,
out of sample, spans as durations on a temporal clock, the names.
"""

from __future__ import annotations

import io
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest
import statsmodels.tsa.holtwinters as holtwinters
from scipy import integrate

from rheon import polars as rp

TIER = "essential"

H_LEVEL, H_TREND = 6.0, 25.0
#: The row from which the weighted means' gains equal the fixed ones:
#: ``0.5 ** (1000 / H_TREND)`` is 1e-12 (`TestHoltAcrossAMissingObservation`).
SETTLED = 1000


def _series(n: int, seed: int = 5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 3.0 + 0.4 * np.arange(n) + np.cumsum(rng.normal(0.0, 0.5, n))


def _out(spec: dict, df: pl.DataFrame) -> pl.DataFrame:
    return rp.ModelBank([spec]).fit_predict(df).unnest("m")


def _spec(**kw) -> dict:
    return rp.spec.holt(
        "m",
        targets=["y"],
        half_life=H_LEVEL,
        trend_half_life=H_TREND,
        min_weight=0.0,
        coef_every=0,
        **kw,
    )


def test_the_forecasts_are_statsmodels_holt_and_their_integral_scipys():
    y = _series(SETTLED + 200)
    spans = [0, 1, 4, 30, 2.5]
    out = _out(_spec(forecast_spans=spans), pl.DataFrame({"y": y}))
    alpha = 1.0 - 0.5 ** (1.0 / H_LEVEL)
    beta = 1.0 - 0.5 ** (1.0 / H_TREND)
    for t in (SETTLED + 10, SETTLED + 120, SETTLED + 199):
        res = holtwinters.Holt(
            y[:t], initialization_method="known", initial_level=y[0], initial_trend=0.0
        ).fit(smoothing_level=alpha, smoothing_trend=beta, optimized=False)
        steps = np.asarray(res.forecast(31))
        # the forecast as a function of the clock past row t: the line through
        # statsmodels' steps 1 and 2, step 1 being row t itself
        a, b = float(steps[0]), float(steps[1] - steps[0])

        def line(u: float, a: float = a, b: float = b) -> float:
            return a + b * u

        for d in spans:
            label = f"h{d:g}"
            got = out[f"forecast_{label}_y"][t]
            want = line(d) if d != int(d) else steps[int(d)]
            assert got == pytest.approx(want, rel=1e-12), (t, d, got, want)
            area = integrate.quad(line, 0.0, d, epsabs=0, epsrel=1e-13)[0] if d else 0.0
            got_sum = out[f"forecast_sum_{label}_y"][t]
            assert got_sum == pytest.approx(area, rel=1e-12, abs=1e-12), (t, d, got_sum, area)
        # Δ = 0 is the row's own prediction, to the bit, and its integral 0
        assert out["forecast_h0_y"][t] == out["pred_y"][t]
        assert out["forecast_sum_h0_y"][t] == 0.0


def test_the_forecasts_are_the_definition_at_the_level_and_trend_before_the_row():
    """On an irregular clock, from the ``coef`` written after the row before:
    ``l + b (s + Δ)`` and ``Δ (l + b (s + Δ/2))`` to the bit, ``s`` the row's
    step."""
    rng = np.random.default_rng(3)
    n = 400
    t = np.cumsum(rng.exponential(1.0, n))
    df = pl.DataFrame({"t": t, "y": 1.0 + 0.3 * t + rng.normal(0, 0.4, n)})
    spans = [3.5, 0.25]
    out = _out(_spec(clock="t", gap_cap=100.0, forecast_spans=spans), df)
    coef = out["coef"].to_list()
    for r in (50, 220, 399):
        level, trend = coef[r - 1]
        s = t[r] - t[r - 1]
        for d in spans:
            assert out[f"forecast_h{d:g}_y"][r] == level + trend * (s + d)
            assert out[f"forecast_sum_h{d:g}_y"][r] == d * (level + trend * (s + 0.5 * d))


def test_durations_on_a_temporal_clock_label_the_fields_as_written():
    start = datetime(2026, 1, 1)
    n = 300
    rng = np.random.default_rng(4)
    df = pl.DataFrame(
        {
            "t": [start + timedelta(hours=i) for i in range(n)],
            "y": 2.0 + 0.1 * np.arange(n) + rng.normal(0, 0.2, n),
        }
    )
    spec = rp.spec.holt(
        "m",
        targets=["y"],
        clock="t",
        gap_cap="1d",
        half_life="6h",
        forecast_spans={"day": timedelta(days=1), "week": "7d"},
        min_weight=0.0,
        coef_every=0,
    )
    assert "forecast_day_y" in rp.spec.output_fields(spec)
    out = _out(spec, df)
    level, trend = out["coef"].to_list()[200]
    # an hour in seconds, the clock's scale, past row 201 by a day
    s, d = 3600.0, 86_400.0
    assert out["forecast_day_y"][201] == level + trend * (s + d)
    week = rp.spec.holt(
        "m", targets=["y"], clock="t", gap_cap="1d", half_life="6h", forecast_spans=["7d"]
    )
    assert "forecast_7d_y" in rp.spec.output_fields(week)


def test_chunk_invariance_and_a_save_go_on_to_the_bit():
    y = _series(700, 6)
    df = pl.DataFrame({"y": y, "g": np.where(np.arange(700) % 3 == 0, "a", "b")})
    spec = _spec(forecast_spans=[1, 7.5], group="g")
    one = rp.ModelBank([spec]).fit_predict(df)
    for size in (1, 13, 211):
        bank = rp.ModelBank([spec])
        many = pl.concat([bank.fit_predict(df.slice(i, size)) for i in range(0, 700, size)])
        assert one.equals(many, null_equal=True), size
    for cut in (1, 330, 699):
        a = rp.ModelBank([spec])
        a.fit_predict(df.slice(0, cut))
        buf = io.BytesIO()
        a.save(buf)
        b = rp.ModelBank.load_bytes(buf.getvalue(), specs=[spec])
        rest = df.slice(cut)
        assert a.fit_predict(rest).equals(b.fit_predict(rest), null_equal=True), cut


def test_a_row_of_weight_zero_is_as_if_absent_and_forecasts_from_its_own_clock():
    """A row of weight 0 or a null target learns nothing and counts in no
    clock: the learned rows' forecasts are those of the stream without it,
    its clock folded into the next row's. It is still read, at its own
    clock."""
    rng = np.random.default_rng(8)
    n = 600
    t = np.cumsum(rng.uniform(0.5, 1.5, n))
    y = 1.0 + 0.2 * t + rng.normal(0, 0.3, n)
    w = np.where(rng.uniform(size=n) < 0.15, 0.0, 1.0)
    df = pl.DataFrame({"t": t, "y": y, "w": w})
    spec = _spec(clock="t", gap_cap=100.0, weight="w", forecast_spans=[2, 0.5])
    full = _out(spec, df)
    keep = w > 0
    thin = _out(spec, df.filter(pl.Series(keep)))
    fields = [f for f in full.columns if f.startswith("forecast")]
    for f in ["pred_y", *fields]:
        assert full[f].filter(pl.Series(keep)).equals(thin[f], null_equal=True), f
    coef = full["coef"].to_list()
    for r in np.flatnonzero(~keep)[5:15]:
        r = int(r)
        last = int(np.flatnonzero(keep[:r])[-1])
        level, trend = coef[last]
        assert full["forecast_h2_y"][r] == level + trend * ((t[r] - t[last]) + 2.0)


def test_a_forecast_reads_no_row_it_forecasts():
    y = _series(500, 9)
    spec = _spec(forecast_spans=[0.5, 3, 40])
    a = _out(spec, pl.DataFrame({"y": y}))
    for r in (200, 450):
        moved = y.copy()
        moved[r] = moved[r] * 3.0 + 10.0
        b = _out(spec, pl.DataFrame({"y": moved}))
        for f in a.columns:
            if f.startswith("forecast"):
                assert a[f][: r + 1].equals(b[f][: r + 1], null_equal=True), (f, r)
                assert not a[f][r + 1 :].equals(b[f][r + 1 :], null_equal=True), (f, r)


def test_no_trend_is_a_flat_forecast():
    y = _series(300, 10)
    spec = rp.spec.holt(
        "m", targets=["y"], half_life=H_LEVEL, trend=False, forecast_spans=[5.0], min_weight=0.0
    )
    out = _out(spec, pl.DataFrame({"y": y}))
    for r in (100, 299):
        assert out["forecast_h5_y"][r] == out["pred_y"][r]
        assert out["forecast_sum_h5_y"][r] == 5.0 * out["pred_y"][r]


def test_names_and_a_half_life_grid():
    spec = rp.spec.holt(
        "m", targets=["y", "z"], half_life=[5.0, 50.0], forecast_spans={"near": 1, "far": 10}
    )
    fields = rp.spec.output_fields(spec)
    assert "forecast_far_z@h50" in fields and "forecast_sum_near_y@h5" in fields
    df = pl.DataFrame({"y": _series(200, 11), "z": _series(200, 12)})
    out = rp.ModelBank([spec]).fit_predict(df)
    assert [f.name for f in out.schema["m"].fields] == fields
    o = out.unnest("m")
    # each instance's forecasts are its own, read from its own level
    assert o["forecast_far_z@h5"][150] != o["forecast_far_z@h50"][150]
    for sfx in ("@h5", "@h50"):
        assert o[f"forecast_near_y{sfx}"][150] != o[f"forecast_far_y{sfx}"][150]


@pytest.mark.parametrize(
    ("kw", "says"),
    [
        ({"forecast_spans": [-1.0]}, "finite span of the clock of 0 or more"),
        ({"forecast_spans": [float("inf")]}, "finite"),
        ({"forecast_spans": [3, 3.0]}, 'two forecast_spans are labeled "h3"'),
        ({"forecast_spans": list(range(65))}, "at most 64"),
        ({"forecast_spans": {"sum": 1}}, 'not begin with "sum"'),
    ],
)
def test_refusals(kw, says):
    with pytest.raises(ValueError, match=says):
        rp.ModelBank([_spec(**kw)])
