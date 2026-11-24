"""``holt``'s fixed forms of seasonal exponential smoothing against
statsmodels (task 488, part 2).

statsmodels is the oracle for every form it computes, with the same fixed
parameters: ``holt``'s half-lives are read as ``α = 1 - 0.5 ** (1 / h)``,
``β = 1 - 0.5 ** (1 / trend_half_life)``, ``γ = 1 - 0.5 ** (m /
seasonal_half_life)`` and ``φ = 0.5 ** (1 / damping_half_life)`` on a row
clock.

- ``ExponentialSmoothing`` (``holtwinters``) with the known start: every one
  of the nine forms (no season, an additive or a multiplicative one, by no
  trend, an additive and a damped one). ``holt`` follows its operation order
  (``model.py``'s ``_predict``), so the two agree to the bit on Linux x86-64
  (measured: 0 ulps at every row of every form); held at 5 ulps, the room the
  platforms' ``exp2`` and ``pow`` leave in the gains.
- ``ExponentialSmoothing`` with the heuristic start, from the row after the
  rows it reads. The start is the same definition in another order (a least
  squares line by its normal equations, where statsmodels takes NumPy's
  ``pinv``; the moving average summed afresh, where pandas keeps a running
  sum), and the replay carries that rounding on: measured at most 6 ulps of
  the prediction over 8 seeds of 600 rows (Linux x86-64), held at 16.
- ``ETSModel`` with the known start, for the forms without a season and with
  an additive one, both error types, whose recursions are one (the error
  type enters the likelihood, not the states): ``smoothing_trend`` is
  ``α·β``. Its additive season is updated from the new level with
  ``γ/(1 − α)``, the same number in another order, so it agrees within a
  few ulps (measured at most 6 over the same seeds), held at 16. Its
  multiplicative season is another recursion, ``γ/(1 − α)·y/l_t`` on the
  new level, where Hyndman's and ``ExponentialSmoothing``'s divide by the
  level carried to the row; ``holt`` is the latter, so that form is held to
  ``ExponentialSmoothing`` alone.

What statsmodels cannot compute -- a clock column, rows of other weights, a
row of weight 0, an uneven clock -- is held to the definitions written out
below (``_written_out``), and chunking and a save to the bank itself.
"""

from __future__ import annotations

import io
import warnings
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from rheon import polars as rp

M = 12
H_LEVEL, H_TREND, H_SEASON, H_DAMP = 3.0, 20.0, 40.0, 30.0
ALPHA = 1.0 - 0.5 ** (1.0 / H_LEVEL)
BETA = 1.0 - 0.5 ** (1.0 / H_TREND)
GAMMA = 1.0 - 0.5 ** (M / H_SEASON)
PHI = 0.5 ** (1.0 / H_DAMP)

#: The forms: season, trend, damped.
FORMS = [
    (season, trend, damped)
    for season in (None, "add", "mul")
    for trend, damped in ((False, False), (True, False), (True, True))
]


def _series(n: int, season: str | None, seed: int) -> np.ndarray:
    """A level rising 0.2 a row with a walk on it, and a season of 12
    added (amplitude 5) or multiplied (a tenth either way)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    base = 50.0 + 0.2 * t + np.cumsum(rng.normal(0.0, 0.3, n))
    wave = np.sin(2 * np.pi * t / M)
    if season == "add":
        return base + 5.0 * wave
    if season == "mul":
        return base * (1.0 + 0.1 * wave)
    return base


def _initial(season: str | None) -> list[float]:
    one, amp = (1.0, 0.1) if season == "mul" else (0.0, 5.0)
    return [one + amp * np.sin(2 * np.pi * p / M) for p in range(M)]


def _spec(season, trend, damped, start="known", **kw):
    args: dict = dict(targets=["y"], half_life=H_LEVEL, min_weight=0.0, start=start)
    if start == "known":
        args["initial_level"] = 50.0
    if season:
        args.update(seasonal=season, seasonal_periods=M, seasonal_half_life=H_SEASON)
        if start == "known":
            args["initial_seasonal"] = _initial(season)
    if trend:
        args["trend_half_life"] = H_TREND
        if start == "known":
            args["initial_trend"] = 0.2
    else:
        args["trend"] = False
    if damped:
        args["damping_half_life"] = H_DAMP
    args.update(kw)
    return rp.spec.holt("h", **args)


def _fit(spec, frame: pl.DataFrame) -> pl.DataFrame:
    return rp.ModelBank([spec]).fit_predict(frame)["h"].struct.unnest()


def _ulps(got: np.ndarray, want: np.ndarray) -> float:
    """The largest distance between the two, in ulps of the larger."""
    scale = np.spacing(np.maximum(np.abs(got), np.abs(want)))
    return float(np.max(np.abs(got - want) / scale))


def _holtwinters(y, season, trend, damped, start="known"):
    known = start == "known"
    model = ExponentialSmoothing(
        y,
        trend="add" if trend else None,
        damped_trend=damped,
        seasonal=season,
        seasonal_periods=M if season else None,
        initialization_method=start,
        initial_level=50.0 if known else None,
        initial_trend=0.2 if known and trend else None,
        initial_seasonal=_initial(season) if known and season else None,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model.fit(
            smoothing_level=ALPHA,
            smoothing_trend=BETA if trend else None,
            smoothing_seasonal=GAMMA if season else None,
            damping_trend=PHI if damped else None,
            optimized=False,
        )


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize(("season", "trend", "damped"), FORMS)
def test_every_form_is_statsmodels_exponential_smoothing(season, trend, damped, seed):
    y = _series(600, season, seed)
    out = _fit(_spec(season, trend, damped), pl.DataFrame({"y": y}))
    res = _holtwinters(y, season, trend, damped)
    pred = out["pred_y"].to_numpy()
    assert _ulps(pred, np.asarray(res.fittedvalues)) <= 5.0
    level, slope = out["coef"][-1][0], out["coef"][-1][1]
    assert _ulps(np.array([level]), np.asarray(res.level)[-1:]) <= 5.0
    if trend:
        assert _ulps(np.array([slope]), np.asarray(res.trend)[-1:]) <= 5.0
    else:
        assert slope == 0.0


@pytest.mark.parametrize(("season", "trend", "damped"), FORMS)
def test_the_heuristic_start_is_statsmodels(season, trend, damped):
    """Withheld as the model's own warm-up while the start holds its rows
    (10 without a season, 5 seasons with one), then statsmodels' heuristic
    start learned forward."""
    y = _series(600, season, 3)
    out = _fit(_spec(season, trend, damped, start="heuristic"), pl.DataFrame({"y": y}))
    n = 5 * M if season else 10
    pred = out["pred_y"].to_numpy()
    assert np.isnan(pred[:n]).all() and not np.isnan(pred[n:]).any()
    assert out["withheld_reason"][:n].to_list() == ["below_warm_rows"] * n
    assert out["withheld_reason"][n:].is_null().all()
    res = _holtwinters(y, season, trend, damped, start="heuristic")
    assert _ulps(pred[n:], np.asarray(res.fittedvalues)[n:]) <= 16.0


@pytest.mark.parametrize("error", ["add", "mul"])
@pytest.mark.parametrize(("season", "trend", "damped"), [f for f in FORMS if f[0] != "mul"])
def test_the_error_type_does_not_enter_the_ets_recursion(season, trend, damped, error):
    y = _series(2000, season, 4)
    pred = _fit(_spec(season, trend, damped), pl.DataFrame({"y": y}))["pred_y"].to_numpy()
    model = ETSModel(
        y,
        error=error,
        trend="add" if trend else None,
        damped_trend=damped,
        seasonal=season,
        seasonal_periods=M if season else None,
        initialization_method="known",
        initial_level=50.0,
        initial_trend=0.2 if trend else None,
        initial_seasonal=_initial(season) if season else None,
    )
    params = [ALPHA] + [ALPHA * BETA] * trend + [GAMMA] * bool(season) + [PHI] * damped
    res = model.smooth(np.array(params))
    assert _ulps(pred, np.asarray(res.fittedvalues)) <= 16.0


# --- what statsmodels cannot compute: the definitions written out ------------


def _written_out(cfg: dict, clock, ys, ws, steps):
    """The model from its definitions: each update a weighted mean whose
    held weight is the one unit rows at the step saturate at, ``λ/(1 −
    λ)``; the phase from each row's own clock value, in integers; the steps
    as the stream decays by them. Returns each row's prediction."""
    m, period = cfg["m"], cfg["period"]
    seasonal, mul = cfg["season"] is not None, cfg["season"] == "mul"
    level, slope = cfg["level"], cfg["trend"]
    seas = list(cfg["seasonal"]) if seasonal else []
    at: list[float | None] = [None] * m
    since, decayed, origin = period / m if seasonal else 1.0, 0.0, None
    preds = []

    def half(s, h):
        return 0.5 ** (s / h) if np.isfinite(h) else 1.0

    def mean(old, new, lam, w):
        held = lam / (1.0 - lam)
        return (held * old + w * new) / (held + w)

    for c, y, w, d in zip(clock, ys, ws, steps, strict=True):
        if w == 0.0:
            # Weight 0: predicted at its own phase, and counted in no clock.
            s = since + d
            p = ((m * (c - origin)) // cfg["period_units"]) % m if seasonal else 0
            big = level + (s * slope if cfg["h_damp"] == np.inf else 0.0)
            preds.append(big * seas[p] if mul else big + (seas[p] if seasonal else 0.0))
            continue
        if origin is None:
            origin = c
        p = ((m * (c - origin)) // cfg["period_units"]) % m if seasonal else 0
        decayed += d
        s = since + d
        if np.isfinite(cfg["h_damp"]):
            phi = half(1.0, cfg["h_damp"])
            phi_s = phi * (1.0 - phi**s) / (1.0 - phi)
        else:
            phi_s = s
        big = level + phi_s * slope
        preds.append(big * seas[p] if mul else big + (seas[p] if seasonal else 0.0))
        if y is None or (mul and y <= 0.0):
            since = s
            continue
        obs = y / seas[p] if mul else y - (seas[p] if seasonal else 0.0)
        new_level = mean(big, obs, half(s, cfg["h_level"]), w)
        if cfg["has_trend"]:
            carried = slope * (half(s, cfg["h_damp"]) if np.isfinite(cfg["h_damp"]) else 1.0)
            slope = mean(carried, (new_level - level) / s, half(s, cfg["h_trend"]), w)
        if seasonal:
            gap = period if at[p] is None else decayed - at[p]
            seen = y / big if mul else y - big
            seas[p] = mean(seas[p], seen, half(gap, cfg["h_season"]), w)
            at[p] = decayed
        level, since = new_level, 0.0
    return np.array(preds)


@pytest.mark.parametrize("season", ["add", "mul"])
def test_a_temporal_clock_reads_its_phase_from_its_raw_values(season):
    """Hourly rows over a daily season of 24 values, with hours missing, a
    night's gap that ``gap_cap`` shortens to 2 hours, rows of weights 0.5
    and 2.5, a null target and rows of weight 0: every row's phase is its
    own hour of the day, whatever the steps between, and every prediction is
    the definitions' (a weighted mean's arithmetic, so within 1e-12)."""
    rng = np.random.default_rng(5)
    start = datetime(2026, 3, 2, 5)
    hours = [h for h in range(24 * 12) if not (100 <= h < 112) and h % 37 != 3]
    clock = [start + timedelta(hours=h) for h in hours]
    n = len(hours)
    hour = np.array([(5 + h) % 24 for h in hours])
    base = 30.0 + 0.01 * np.array(hours) + rng.normal(0, 0.2, n)
    wave = np.sin(2 * np.pi * hour / 24)
    y = base + 3 * wave if season == "add" else base * (1 + 0.1 * wave)
    ys: list[float | None] = [float(v) for v in y]
    ys[40] = None
    ws = np.ones(n)
    ws[7::11] = 0.5
    ws[9::13] = 2.5
    ws[20::29] = 0.0
    init = [(1.0 if season == "mul" else 0.0)] * 24
    spec = rp.spec.holt(
        "h",
        targets=["y"],
        clock="t",
        weight="w",
        gap_cap="2h",
        half_life="5h",
        trend_half_life="2d",
        seasonal=season,
        seasonal_periods=24,
        season_length="1d",
        seasonal_half_life="4d",
        start="known",
        initial_level=30.0,
        initial_trend=0.0,
        initial_seasonal=init,
        min_weight=0.0,
    )
    frame = pl.DataFrame({"t": clock, "y": ys, "w": ws}, schema_overrides={"y": pl.Float64})
    pred = _fit(spec, frame)["pred_y"].to_numpy()
    ns = [int((c - datetime(1970, 1, 1)).total_seconds()) * 10**9 for c in clock]
    # The steps the stream decays by: seconds since the last row of
    # positive weight, capped at 2 hours.
    steps, last = [], None
    for c, w in zip(ns, ws, strict=True):
        steps.append(0.0 if last is None else min((c - last) / 1e9, 7200.0))
        if w > 0:
            last = c
    cfg = dict(
        m=24,
        period=86400.0,
        period_units=86400 * 10**9,
        season=season,
        level=30.0,
        trend=0.0,
        seasonal=init,
        has_trend=True,
        h_level=5 * 3600.0,
        h_trend=2 * 86400.0,
        h_season=4 * 86400.0,
        h_damp=np.inf,
    )
    want = _written_out(cfg, ns, ys, ws, steps)
    np.testing.assert_allclose(pred, want, rtol=1e-12)


@pytest.mark.parametrize("clock", [False, True])
def test_a_row_of_weight_zero_is_as_if_absent(clock):
    """Every other row's output is the stream's without it, to the bit. With
    a clock column it keeps its own place in the season, and without one it
    counts in no clock, so the row after it takes its place."""
    n = 300
    y = _series(n, "add", 6)
    df = pl.DataFrame(
        {"t": np.arange(n) * 2, "y": y, "w": np.where(np.arange(n) % 17 == 9, 0.0, 1.0)}
    )
    kw = (
        dict(weight="w", clock="t", gap_cap=1000.0, season_length=2.0 * M)
        if clock
        else dict(weight="w")
    )
    spec = _spec("add", True, True, **kw)
    full = _fit(spec, df)
    keep = (df["w"] > 0).to_numpy()
    thin = _fit(spec, df.filter(pl.Series(keep)))
    for f in ("pred_y", "weight_sum", "coef"):
        assert full[f].filter(pl.Series(keep)).equals(thin[f], null_equal=True), f
    assert full["pred_y"].filter(pl.Series(~keep)).is_not_null().all(), "it is predicted"


@pytest.mark.parametrize("y_row", [0.0, -2.0, 1e-300])
def test_a_multiplicative_target_of_zero_or_below_is_predicted_and_not_learned(y_row):
    """Its prediction stands, its reason names it, and every row after is
    the stream's with its target null: not learned. A target above 0, the
    smallest a double holds near it, is learned."""
    y = _series(200, "mul", 7)
    y[120] = y_row
    spec = _spec("mul", True, False)
    out = _fit(spec, pl.DataFrame({"y": y}))
    nulled = [None if i == 120 else float(v) for i, v in enumerate(y)]
    other = _fit(spec, pl.DataFrame({"y": nulled}, schema={"y": pl.Float64}))
    assert out["pred_y"][120] is not None
    if y_row > 0:
        assert out["withheld_reason"][120] is None
        assert not out["pred_y"][121:].equals(other["pred_y"][121:])
    else:
        assert out["withheld_reason"][120] == "not_positive"
        assert out["withheld_reason"].null_count() == 199
        assert out["pred_y"][121:].equals(other["pred_y"][121:])


@pytest.mark.parametrize("start", ["known", "heuristic"])
def test_chunks_and_a_save_go_on_to_the_bit(start):
    """One chunk and many, the heuristic start's rows held across the
    chunks' edges, give the same output to the bit; so does a state saved
    while the start holds its rows, and after, and loaded."""
    n = 400
    y = _series(n, "mul", 8)
    df = pl.DataFrame({"y": y, "w": np.where(np.arange(n) % 9 == 4, 0.5, 1.0)})
    spec = _spec("mul", True, True, start=start, weight="w")
    one = rp.ModelBank([spec]).fit_predict(df)["h"].struct.unnest().drop("coef")
    for size in (1, 7, 61):
        bank = rp.ModelBank([spec])
        many = pl.concat([bank.fit_predict(df.slice(i, size)) for i in range(0, n, size)])
        assert one.equals(many["h"].struct.unnest().drop("coef"), null_equal=True), size
    for cut in (30, 200):
        a = rp.ModelBank([spec])
        a.fit_predict(df.slice(0, cut))
        buf = io.BytesIO()
        a.save(buf)
        b = rp.ModelBank.load_bytes(buf.getvalue(), specs=[spec])
        rest = df.slice(cut)
        assert a.fit_predict(rest).equals(b.fit_predict(rest), null_equal=True), cut


def test_the_state_is_held_to_max_state_bytes_on_both_sides():
    """A season of a year's hours under the heuristic start needs 1,717,105
    bytes for one target: refused at the default 256 KB with the bytes it
    needs, built at exactly them, refused a byte below."""
    kw = dict(
        seasonal="add",
        seasonal_periods=8760,
        seasonal_half_life=1e5,
        start="heuristic",
        targets=["y"],
        half_life=10.0,
    )
    need = 64 + (81 + 16 * 8760 + 36 * 5 * 8760)
    frame = pl.DataFrame({"y": [1.0, 2.0]})
    with pytest.raises(Exception, match=f"Set max_state_bytes to {need} or more"):
        rp.ModelBank([rp.spec.holt("h", **kw)]).fit_predict(frame)
    rp.ModelBank([rp.spec.holt("h", max_state_bytes=need, **kw)]).fit_predict(frame)
    with pytest.raises(Exception, match="max_state_bytes"):
        rp.ModelBank([rp.spec.holt("h", max_state_bytes=need - 1, **kw)]).fit_predict(frame)
