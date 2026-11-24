"""``rp.ops.stl`` and ``rp.ops.mstl`` through ``with_windows``, held to
statsmodels' ``STL`` and ``MSTL`` on each row's window (docs/PLAN.md task
490; the design record SEASONAL-DESIGN-489-490).

Each emitted value is statsmodels' fit of the group's last ``W`` learned
rows, read at the lag, to the bit: both sides run the authors' Fortran as
statsmodels translates it, each cube through the platform's ``pow``. The
rest is what a window operator owes: groups, nulls learning nothing, the
row's own value and every later one never read under ``closed="left"``,
any chunking and a streaming query, and the refusals by name.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from statsmodels.tsa.seasonal import MSTL, STL

from rheon import polars as rp

TIER = "essential"


def _series(n: int, seed: int, periods: tuple[int, ...] = (4,)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    y = 0.02 * t + 0.3 * rng.standard_normal(n)
    for k, p in enumerate(periods):
        y += (1.5 - 0.5 * k) * np.sin(2 * np.pi * t / p + k)
    return y


def _bits(a: list[float | None]) -> list[int | None]:
    return [None if v is None else int(np.float64(v).view(np.int64)) for v in a]


def _run(xs: np.ndarray, *exprs: pl.Expr, groups: list[str] | None = None) -> pl.DataFrame:
    frame = pl.DataFrame({"t": np.arange(len(xs), dtype=np.float64), "x": xs})
    if groups is not None:
        frame = frame.with_columns(g=pl.Series(groups))
    return rp.stream.with_windows(
        frame, *exprs, clock="t", gap_cap=1e9, group=None if groups is None else "g"
    )


STL_CASES = [
    # (settings, W, lag, closed)
    ({"period": 4}, 30, 5, "left"),
    ({"period": 7, "seasonal": 9}, 40, 0, "right"),
    ({"period": 12, "seasonal": 13, "trend": 25, "low_pass": 13}, 48, 20, "left"),
    ({"period": 5, "seasonal_deg": 0, "trend_deg": 0, "low_pass_deg": 0}, 30, 3, "right"),
    ({"period": 6, "seasonal_jump": 2, "trend_jump": 3, "low_pass_jump": 2}, 36, 6, "left"),
    ({"period": 4, "robust": True}, 32, 4, "left"),
    ({"period": 4, "robust": True, "inner_iter": 3, "outer_iter": 2}, 28, 1, "right"),
]


@pytest.mark.parametrize(("settings", "w", "lag", "closed"), STL_CASES)
def test_stl_is_statsmodels_on_each_window(settings, w, lag, closed):
    xs = _series(w + 40, seed=w + lag, periods=(settings["period"],))
    comps = ("trend", "seasonal", "resid")
    exprs = [
        rp.ops.stl("x", component=c, window=w, lag=lag, closed=closed, **settings).alias(c)
        for c in comps
    ]
    out = _run(xs, *exprs)
    kw = {k: v for k, v in settings.items() if k not in ("inner_iter", "outer_iter")}
    fit_kw = {k: settings[k] for k in ("inner_iter", "outer_iter") if k in settings}
    at = w - lag if closed == "left" else w - 1 - lag
    sm_name = {"trend": "trend", "seasonal": "seasonal", "resid": "resid"}
    for t in range(len(xs)):
        lo, hi = (t - w, t) if closed == "left" else (t + 1 - w, t + 1)
        if lo < 0:
            assert all(out[c][t] is None for c in comps), t
            continue
        fit = STL(xs[lo:hi], **kw).fit(**fit_kw)
        for c in comps:
            want = float(getattr(fit, sm_name[c])[at])
            assert _bits([out[c][t]]) == _bits([want]), (settings, t, c)


@pytest.mark.parametrize(
    ("periods", "windows", "w", "lag", "closed"),
    [
        ((4, 6), None, 30, 6, "left"),
        ((6, 4), (9, 7), 36, 0, "right"),
        ((3, 5, 8), None, 40, 10, "left"),
    ],
)
def test_mstl_is_statsmodels_on_each_window(periods, windows, w, lag, closed):
    xs = _series(w + 30, seed=w, periods=periods)
    exprs = [
        rp.ops.mstl(
            "x",
            periods=list(periods),
            windows=windows,
            component="trend",
            window=w,
            lag=lag,
            closed=closed,
        ).alias("trend"),
        rp.ops.mstl(
            "x",
            periods=list(periods),
            windows=windows,
            component="resid",
            window=w,
            lag=lag,
            closed=closed,
        ).alias("resid"),
    ]
    exprs += [
        rp.ops.mstl(
            "x",
            periods=list(periods),
            windows=windows,
            component="seasonal",
            season=p,
            window=w,
            lag=lag,
            closed=closed,
        ).alias(f"s{p}")
        for p in periods
    ]
    out = _run(xs, *exprs)
    at = w - lag if closed == "left" else w - 1 - lag
    sm_windows = None if windows is None else list(windows)
    for t in range(len(xs)):
        lo, hi = (t - w, t) if closed == "left" else (t + 1 - w, t + 1)
        if lo < 0:
            continue
        fit = MSTL(xs[lo:hi], periods=list(periods), windows=sm_windows, iterate=2).fit()
        seasonal = np.atleast_2d(np.asarray(fit.seasonal).T)
        order = sorted(periods)
        assert _bits([out["trend"][t]]) == _bits([float(fit.trend[at])]), t
        assert _bits([out["resid"][t]]) == _bits([float(fit.resid[at])]), t
        for p in periods:
            want = float(seasonal[order.index(p)][at])
            assert _bits([out[f"s{p}"][t]]) == _bits([want]), (t, p)


def test_groups_and_nulls_each_decompose_their_learned_rows():
    n = 120
    xs = _series(n, seed=3)
    xs[[5, 17, 40, 41, 77]] = np.nan
    groups = ["a" if k % 3 else "b" for k in range(n)]
    frame = pl.DataFrame({"t": np.arange(n, dtype=np.float64), "x": xs, "g": groups})
    frame = frame.with_columns(pl.col("x").fill_nan(None))
    e = rp.ops.stl("x", period=4, window=20, lag=3).alias("trend")
    out = rp.stream.with_windows(frame, e, clock="t", gap_cap=1e9, group="g")
    learned: dict[str, list[float]] = {"a": [], "b": []}
    for k in range(n):
        g, x = groups[k], frame["x"][k]
        before = learned[g][-20:]
        want = STL(np.array(before), period=4).fit().trend[17] if len(before) == 20 else None
        got = out["trend"][k]
        assert _bits([got]) == _bits([None if want is None else float(want)]), k
        if x is not None:
            learned[g].append(x)


def test_left_reads_no_row_at_or_after_its_own():
    xs = _series(90, seed=5)
    e = rp.ops.stl("x", period=4, window=24, lag=2).alias("v")
    base = _run(xs, e)["v"].to_list()
    for t in (40, 70):
        changed = xs.copy()
        changed[t:] = 1e6
        assert _run(changed, e)["v"].to_list()[: t + 1] == base[: t + 1], t


def test_chunks_and_a_streaming_query_give_the_same_bits():
    xs = _series(150, seed=9, periods=(4, 6))
    exprs = [
        rp.ops.stl("x", period=4).alias("a"),
        rp.ops.mstl("x", periods=[4, 6], component="seasonal", season=6, closed="right").alias("b"),
    ]
    frame = pl.DataFrame({"t": np.arange(150, dtype=np.float64), "x": xs})
    whole = rp.stream.with_windows(frame, *exprs, clock="t", gap_cap=1e9)
    chunked = pl.concat([frame.slice(i, 7) for i in range(0, 150, 7)], rechunk=False)
    streamed = rp.stream.with_windows(frame.lazy(), *exprs, clock="t", gap_cap=1e9).collect(
        engine="streaming"
    )
    assert whole.equals(rp.stream.with_windows(chunked, *exprs, clock="t", gap_cap=1e9))
    assert whole.equals(streamed)


def test_the_defaults_are_the_embargo_under_closed_left():
    # period 4, span 7: h = 12, W = 25, lag 13; the 26th row is the first read.
    xs = _series(60, seed=2)
    out = _run(xs, rp.ops.stl("x", period=4).alias("v"))["v"].to_list()
    assert out[:25] == [None] * 25 and out[25] is not None
    want = STL(xs[0:25], period=4).fit().trend[12]
    assert _bits([out[25]]) == _bits([float(want)])


def test_refusals_name_the_operator_and_the_setting():
    xs = _series(40, seed=1)
    with pytest.raises(ValueError, match="rp.ops.stl: closed"):
        rp.ops.stl("x", period=4, closed="both")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="rp.ops.stl: seasonal_deg"):
        rp.ops.stl("x", period=4, seasonal_deg=2)
    with pytest.raises(ValueError, match="needs season"):
        rp.ops.mstl("x", periods=[4, 6], component="seasonal")
    with pytest.raises(ValueError, match="not one of the periods"):
        rp.ops.mstl("x", periods=[4, 6], component="seasonal", season=5)
    with pytest.raises(TypeError, match="season is given"):
        rp.ops.mstl("x", periods=[4], season=4)
    with pytest.raises(ValueError, match="windows has 1 spans for 2 periods"):
        rp.ops.mstl("x", periods=[4, 6], windows=[7])
    for expr, words in [
        (rp.ops.stl("x", period=4, seasonal=6), "seasonal"),
        (rp.ops.stl("x", period=4, window=7), "fewer than two periods"),
        (rp.ops.mstl("x", periods=[4, 6], window=12), "half the window"),
        (rp.ops.stl("x", period=24, window=40000), "320024 bytes"),
    ]:
        with pytest.raises(Exception, match=words):
            _run(xs, expr.alias("v"))


def test_stl_and_mstl_are_in_ops_and_not_at_the_root():
    assert "stl" in rp.ops.__all__ and "mstl" in rp.ops.__all__
    assert not hasattr(rp, "stl") and not hasattr(rp, "mstl")
