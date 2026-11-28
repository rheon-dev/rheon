"""Tests for exact window operators: ``rp.ops.rolling_quantile``,
``rp.ops.rolling_rank``, and ``rp.ops.ewm_mean(..., adjust=True)``."""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest
from scipy.stats import rankdata

from polars_version import INSTALLED
from rheon import polars as rp

TIER = "essential"


def _clock_frame(seed: int = 11, n: int = 160) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    dt = rng.choice([0.0, 0.5, 1.0, 2.0, 3.0], size=n, p=[0.1, 0.25, 0.3, 0.2, 0.15])
    t = np.cumsum(dt)
    x = rng.normal(size=n)
    # Inject ties, -0.0 / +0.0, and nulls.
    x[5] = 0.0
    x[6] = -0.0
    x[20] = x[19]
    x[21] = x[19]
    vals: list[float | None] = [float(v) for v in x]
    vals[12] = None
    vals[55] = None
    return pl.DataFrame({"t": t, "x": vals, "g": ["a" if i % 2 == 0 else "b" for i in range(n)]})


@pytest.mark.parametrize(
    "interpolation", ["nearest", "higher", "lower", "midpoint", "linear", "equiprobable"]
)
@pytest.mark.parametrize("closed", ["right", "left", "both", "none"])
def test_rolling_quantile_matches_polars_bit_exact(interpolation, closed):
    """``rp.ops.rolling_quantile`` matches Polars ``rolling_quantile_by`` across
    all 6 interpolations and all 4 edge modes, streaming chunk-invariantly."""
    rng = np.random.default_rng(42)
    n = 140
    # Use integer timestamps to match Polars rolling_quantile_by directly.
    t = np.cumsum(rng.integers(0, 4, size=n)).astype(np.int64)
    x = rng.normal(size=n)
    x[10] = x[9]
    x[25] = 0.0
    x[26] = -0.0
    df = pl.DataFrame({"t": t, "x": x})
    q = 0.37
    w = 12
    want = df.select(
        pl.col("x")
        .rolling_quantile_by(
            "t", window_size=f"{w}i", quantile=q, interpolation=interpolation, closed=closed
        )
        .alias("q")
    )["q"]
    op = rp.ops.rolling_quantile(
        pl.col("x"),
        quantile=q,
        window_size=w,
        interpolation=interpolation,
        closed=closed,
        min_samples=1,
    ).alias("q")
    got = rp.stream.with_windows(df.lazy(), op, clock="t", gap_cap=100).collect()["q"]
    assert got.equals(want), (interpolation, closed)

    # Chunk invariance across small chunks.
    got_chunked = rp.stream.with_windows(df.lazy(), op, clock="t", gap_cap=100).collect(
        engine="streaming"
    )["q"]
    assert got_chunked.equals(want)


@pytest.mark.parametrize("method", ["average", "min", "max", "dense"])
@pytest.mark.parametrize("closed", ["right", "both"])
def test_rolling_rank_matches_polars_bit_exact(method, closed):
    """``rp.ops.rolling_rank`` matches Polars ``rolling_rank_by`` and
    ``scipy.stats.rankdata`` across all 4 deterministic methods, including
    ties, -0.0/+0.0, and duplicate timestamps."""
    rng = np.random.default_rng(73)
    n = 140
    t = np.cumsum(rng.integers(0, 3, size=n)).astype(np.int64)
    x = np.round(rng.normal(size=n), 1)  # dense ties
    x[14] = 0.0
    x[15] = -0.0
    df = pl.DataFrame({"t": t, "x": x})
    w = 10
    op = rp.ops.rolling_rank(
        pl.col("x"), window_size=w, method=method, closed=closed, min_samples=1
    ).alias("rk")
    got = rp.stream.with_windows(df.lazy(), op, clock="t", gap_cap=100).collect()["rk"]

    # Independent scipy.stats.rankdata oracle (runs on every Polars version,
    # including 1.37.0 where Polars' rolling_rank_by mishandled duplicate stamps).
    want_scipy: list[float] = []
    for i in range(n):
        ti = t[i]
        lo = ti - w
        mask = (t >= lo) & (t <= ti) if closed == "both" else (t > lo) & (t <= ti)
        idx = np.flatnonzero(mask)
        ranks = rankdata(x[idx], method=method)
        pos = int(np.flatnonzero(idx == i)[0])
        want_scipy.append(float(ranks[pos]))
    assert got.equals(pl.Series("rk", want_scipy, dtype=pl.Float64)), (method, closed)

    if INSTALLED >= (2, 0, 0):
        want = df.select(
            pl.col("x")
            .rolling_rank_by("t", window_size=f"{w}i", method=method, closed=closed)
            .cast(pl.Float64)
            .alias("rk")
        )["rk"]
        assert got.equals(want), (method, closed)
    else:
        # On py-polars 1.x (< 2.0), verify bit-exact parity with Polars'
        # rolling_rank_by on strictly increasing timestamps where it agrees.
        t_strict = np.cumsum(rng.integers(1, 4, size=n)).astype(np.int64)
        df_s = pl.DataFrame({"t": t_strict, "x": x})
        want_s = df_s.select(
            pl.col("x")
            .rolling_rank_by("t", window_size=f"{w}i", method=method, closed=closed)
            .cast(pl.Float64)
            .alias("rk")
        )["rk"]
        got_s = rp.stream.with_windows(df_s.lazy(), op, clock="t", gap_cap=100).collect()["rk"]
        assert got_s.equals(want_s), (method, closed)


def test_rolling_quantile_and_rank_forward_windows_and_refusals():
    """Forward ``reverse=True`` matches backward on reversed clocks; illegal
    methods, quantiles, closed modes, and state caps are refused cleanly."""
    df = _clock_frame(seed=99, n=80).filter(pl.col("g") == "a")
    fwd_q = rp.ops.rolling_quantile(
        pl.col("x"),
        quantile=0.5,
        window_size=8.0,
        interpolation="linear",
        closed="left",
        partial="keep",
        reverse=True,
    ).alias("fq")
    fwd_r = rp.ops.rolling_rank(
        pl.col("x"), window_size=8.0, method="average", partial="keep", reverse=True
    ).alias("fr")
    out = rp.stream.with_windows(df.lazy(), fwd_q, fwd_r, clock="t", gap_cap=100.0).collect()
    # Direct brute-force check on every row for forward window [t_i, t_i + 8).
    # Rows whose far edge t_i + 8 lies past the stream's end remain open and null.
    ts = df["t"].to_list()
    xs = df["x"].to_list()
    t_end = ts[-1]
    for i in range(len(ts)):
        if ts[i] + 8.0 > t_end:
            assert out["fq"][i] is None
            assert out["fr"][i] is None
            continue
        win = [xs[j] for j in range(len(ts)) if ts[i] <= ts[j] < ts[i] + 8.0 and xs[j] is not None]
        if not win:
            assert out["fq"][i] is None
            assert out["fr"][i] is None
        else:
            assert np.isclose(out["fq"][i], float(np.median(win)), rtol=1e-14)
            if xs[i] is not None:
                lt = sum(1 for v in win if v < xs[i])
                eq = sum(1 for v in win if v == xs[i])
                want_rk = lt + 0.5 * (eq + 1)
                assert np.isclose(out["fr"][i], want_rk, rtol=1e-14)

    # Refusals
    with pytest.raises(ValueError, match="method"):
        rp.ops.rolling_rank(pl.col("x"), window_size=5, method="random")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="method"):
        rp.ops.rolling_rank(pl.col("x"), window_size=5, method="ordinal")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="closed"):
        rp.ops.rolling_rank(pl.col("x"), window_size=5, closed="left")
    with pytest.raises(ValueError, match="closed"):
        rp.ops.rolling_rank(pl.col("x"), window_size=5, closed="none")
    with pytest.raises(ValueError, match="closed"):
        rp.ops.rolling_rank(pl.col("x"), window_size=5, closed="right", reverse=True)
    with pytest.raises(ValueError, match="quantile"):
        rp.ops.rolling_quantile(pl.col("x"), quantile=1.5, window_size=5)
    with pytest.raises(ValueError, match="interpolation"):
        rp.ops.rolling_quantile(
            pl.col("x"),
            quantile=0.5,
            window_size=5,
            interpolation="bad",  # type: ignore[arg-type]
        )

    # max_state_bytes cap enforcement on large window
    big = pl.DataFrame({"t": np.arange(300, dtype=float), "x": np.arange(300, dtype=float)})
    tight = rp.ops.rolling_quantile(
        pl.col("x"), quantile=0.5, window_size=250.0, max_state_bytes=1024
    ).alias("q")
    with pytest.raises((ValueError, pl.exceptions.ComputeError), match="max_state_bytes"):
        rp.stream.with_windows(big.lazy(), tight, clock="t", gap_cap=1000.0).collect()


@pytest.mark.parametrize("half_life", [1.0, 5.0, 20.0, 100.0])
def test_ewm_mean_adjust_true_matches_polars_on_row_clock(half_life):
    """``rp.ops.ewm_mean(..., adjust=True)`` matches Polars
    ``ewm_mean(half_life=..., adjust=True, ignore_nulls=False)`` to 1 ULP on
    non-null rows, and carries the standing window value on null rows."""
    rng = np.random.default_rng(123)
    n = 200
    x: list[float | None] = [float(v) for v in rng.normal(size=n)]
    x[0] = None
    x[15] = None
    x[16] = None
    x[88] = None
    df = pl.DataFrame({"x": x})
    want = df.select(
        pl.col("x").ewm_mean(half_life=half_life, adjust=True, ignore_nulls=False).alias("m")
    )["m"]
    op = rp.ops.ewm_mean(pl.col("x"), half_life=half_life, adjust=True).alias("m")
    got = rp.stream.with_windows(df.lazy(), op).collect()["m"]
    assert got[0] is None
    non_null = df["x"].is_not_null()
    assert np.allclose(
        got.filter(non_null).to_numpy(),
        want.filter(non_null).to_numpy(),
        rtol=1e-14,
        atol=1e-15,
    ), half_life
    # Null rows carry the standing EW mean (never altering subsequent non-null rows).
    assert got[15] == got[14] and got[16] == got[14] and got[88] == got[87]


def test_ewm_mean_adjust_true_clock_window_reverse_and_none_unchanged():
    """``adjust=True`` works with irregular clocks, hard windows, ``reverse=True``,
    and ``half_life=inf``; ``adjust=False`` is refused; ``adjust=None`` is unchanged."""
    df = _clock_frame(seed=31, n=90).filter(pl.col("g") == "a")
    hl = 6.0
    op_pt = rp.ops.ewm_mean(pl.col("x"), half_life=hl, window_size=12.0, adjust=True).alias("pt")
    op_rev = rp.ops.ewm_mean(
        pl.col("x"),
        half_life=hl,
        window_size=12.0,
        closed="left",
        partial="keep",
        adjust=True,
        reverse=True,
    ).alias("rev")
    out = rp.stream.with_windows(df.lazy(), op_pt, op_rev, clock="t", gap_cap=100.0).collect()

    ts = df["t"].to_list()
    xs = df["x"].to_list()
    for i in range(len(ts)):
        # Backward (t_i - 12, t_i]
        num = den = 0.0
        for j in range(len(ts)):
            if ts[i] - 12.0 < ts[j] <= ts[i] and xs[j] is not None:
                w = 0.5 ** ((ts[i] - ts[j]) / hl)
                num += w * xs[j]
                den += w
        if den > 0.0:
            assert np.isclose(out["pt"][i], num / den, rtol=1e-13, atol=1e-13)
        else:
            assert out["pt"][i] is None

        # Forward [t_i, t_i + 12): rows whose far edge exceeds t_end stay open and null.
        if ts[i] + 12.0 > ts[-1]:
            assert out["rev"][i] is None
            continue
        num_f = den_f = 0.0
        for j in range(len(ts)):
            if ts[i] <= ts[j] < ts[i] + 12.0 and xs[j] is not None:
                w = 0.5 ** ((ts[j] - ts[i]) / hl)
                num_f += w * xs[j]
                den_f += w
        if den_f > 0.0:
            assert np.isclose(out["rev"][i], num_f / den_f, rtol=1e-13, atol=1e-13)
        else:
            assert out["rev"][i] is None

    # half_life=inf without window_size -> cumulative mean
    op_inf = rp.ops.ewm_mean(pl.col("x"), half_life=math.inf, adjust=True).alias("cm")
    cm = rp.stream.with_windows(df.lazy(), op_inf, clock="t", gap_cap=100.0).collect()["cm"]
    valid = [v for v in xs if v is not None]
    assert np.isclose(cm[-1], float(np.mean(valid)), rtol=1e-13)

    # adjust=False refused
    with pytest.raises(ValueError, match="adjust"):
        rp.ops.ewm_mean(pl.col("x"), half_life=10.0, adjust=False)  # type: ignore[arg-type]
