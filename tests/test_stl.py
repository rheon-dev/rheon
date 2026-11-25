"""``rp.ops.stl`` and ``rp.ops.mstl`` through ``with_windows``, held to the
decomposition of each row's window computed exactly
(``tests/reference_stl.py``: the authors' method in 50-digit arithmetic),
within the bound of the computation's own rounding (docs/PLAN.md task 490;
the design record SEASONAL-DESIGN-489-490).

**Which side is exact.** ``reference_stl`` is: each value to 50 digits, with
``B``, the most a double evaluation of the same operations may part from it
to first order, each rounding weighted by the output's derivative in it.
This library and statsmodels are two double evaluations. They were bit-equal
on Linux x86-64, where neither fuses a multiply and an add. statsmodels'
build for macOS arm64 does: clang fuses ``b (j - abar) + 1`` and the jump's
interpolation always, and the loess's three sums in each loop's scalar
remainder after its vectorized body (four or eight terms a pass), so its last
bits there are its compiler's. A longhand of the method with those fusings
reproduces statsmodels' every value on this Mac, and without them this
library's (W14-15). So statsmodels is a second opinion, held to this
library within ``2 B``: each is within ``B`` of the exact value. ``B`` counts
two roundings where a fused operation makes one and allows a platform's
``pow`` three ulps, so it holds on every platform. Measured on macOS arm64
(2026-11-24) over the grids' 280 values: this library's error is at most
0.12 of ``B`` (MSTL 0.07) and statsmodels' 0.13 (0.08), and ``B`` is 7 to
3,900 units of rounding of the window's largest value, the robust fit's 16
runs the most.

The rest is what a window operator owes: groups, nulls learning nothing, the
row's own value and every later one never read under ``closed="left"``,
any chunking and a streaming query, and the refusals by name.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from statsmodels.tsa.seasonal import MSTL, STL

import reference_stl as ref
from rheon import polars as rp

TIER = "mixed"

_GRID = pytest.mark.extended(reason="an exact STL in 50 digits and its bound on each window")


def _series(n: int, seed: int, periods: tuple[int, ...] = (4,)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    y = 0.02 * t + 0.3 * rng.standard_normal(n)
    for k, p in enumerate(periods):
        y += (1.5 - 0.5 * k) * np.sin(2 * np.pi * t / p + k)
    return y


def _run(xs: np.ndarray, *exprs: pl.Expr, groups: list[str] | None = None) -> pl.DataFrame:
    frame = pl.DataFrame({"t": np.arange(len(xs), dtype=np.float64), "x": xs})
    if groups is not None:
        frame = frame.with_columns(g=pl.Series(groups))
    return rp.stream.with_windows(
        frame, *exprs, clock="t", gap_cap=1e9, group=None if groups is None else "g"
    )


def _held(got: float | None, exact: ref.X, second: float, what: object) -> None:
    """``got`` within ``B`` of the exact value, and statsmodels' ``second``
    within ``2 B`` of ``got``."""
    assert got is not None, what
    b = ref.bound(exact)
    assert ref.gap(got, exact) <= b, (what, got, float(exact), b)
    assert abs(got - second) <= 2 * b, (what, got, second, b)


def _rows(n: int, w: int, closed: str, every: int | None) -> list[int]:
    """The rows whose window is full, every ``every``-th from the first, and
    the last; the last alone without ``every``."""
    if every is None:
        return [n - 1]
    first = w if closed == "left" else w - 1
    return sorted({*range(first, n, every), n - 1})


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


def _check_stl(settings: dict, w: int, lag: int, closed: str, every: int | None) -> None:
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
    for t in range(len(xs)):
        lo = t - w if closed == "left" else t + 1 - w
        if lo < 0:
            assert all(out[c][t] is None for c in comps), t
    for t in _rows(len(xs), w, closed, every):
        lo, hi = (t - w, t) if closed == "left" else (t + 1 - w, t + 1)
        exact = ref.stl(xs[lo:hi], **settings)
        fit = STL(xs[lo:hi], **kw).fit(**fit_kw)
        for c, ex, sm in zip(comps, exact, (fit.trend, fit.seasonal, fit.resid), strict=True):
            _held(out[c][t], ex[at], float(sm[at]), (settings, t, c))


def test_one_window_is_the_exact_stl():
    """The grid's first case on its last window."""
    _check_stl(*STL_CASES[0], every=None)


@_GRID
@pytest.mark.parametrize(("settings", "w", "lag", "closed"), STL_CASES)
def test_stl_is_the_exact_stl_on_each_window(settings, w, lag, closed):
    _check_stl(settings, w, lag, closed, every=5)


MSTL_CASES = [
    ((4, 6), None, 30, 6, "left"),
    ((6, 4), (9, 7), 36, 0, "right"),
    ((3, 5, 8), None, 40, 10, "left"),
]


def _check_mstl(periods, windows, w, lag, closed, every: int | None) -> None:
    xs = _series(w + 30, seed=w, periods=periods)
    common = {"periods": list(periods), "windows": windows, "window": w, "lag": lag}
    exprs = [
        rp.ops.mstl("x", component="trend", closed=closed, **common).alias("trend"),
        rp.ops.mstl("x", component="resid", closed=closed, **common).alias("resid"),
    ]
    exprs += [
        rp.ops.mstl("x", component="seasonal", season=p, closed=closed, **common).alias(f"s{p}")
        for p in periods
    ]
    out = _run(xs, *exprs)
    at = w - lag if closed == "left" else w - 1 - lag
    sm_windows = None if windows is None else list(windows)
    order = sorted(periods)
    for t in _rows(len(xs), w, closed, every):
        lo, hi = (t - w, t) if closed == "left" else (t + 1 - w, t + 1)
        trend, seasons, resid = ref.mstl(xs[lo:hi], periods, windows)
        fit = MSTL(xs[lo:hi], periods=list(periods), windows=sm_windows, iterate=2).fit()
        sm_seasonal = np.atleast_2d(np.asarray(fit.seasonal).T)
        _held(out["trend"][t], trend[at], float(fit.trend[at]), (t, "trend"))
        _held(out["resid"][t], resid[at], float(fit.resid[at]), (t, "resid"))
        for p in periods:
            i = order.index(p)
            _held(out[f"s{p}"][t], seasons[i][at], float(sm_seasonal[i][at]), (t, p))


def test_one_window_is_the_exact_mstl():
    """The grid's first case on its last window."""
    _check_mstl(*MSTL_CASES[0], every=None)


@_GRID
@pytest.mark.parametrize(("periods", "windows", "w", "lag", "closed"), MSTL_CASES)
def test_mstl_is_the_exact_mstl_on_each_window(periods, windows, w, lag, closed):
    _check_mstl(periods, windows, w, lag, closed, every=5)


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
    checked = 0
    for k in range(n):
        g, x = groups[k], frame["x"][k]
        before = learned[g][-20:]
        got = out["trend"][k]
        if len(before) < 20:
            assert got is None, k
        elif k % 7 == 0:
            exact = ref.stl(before, period=4)[0][17]
            _held(got, exact, float(STL(np.array(before), period=4).fit().trend[17]), k)
            checked += 1
        else:
            assert got is not None, k
        if x is not None:
            learned[g].append(x)
    assert checked > 5


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
    exact = ref.stl(xs[0:25], period=4)[0][12]
    _held(out[25], exact, float(STL(xs[0:25], period=4).fit().trend[12]), 25)


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
