"""``ewm_moments``' lagged terms through the bank, against pandas and Polars.

A lagged term pairs each learned row with the row ``k`` back (on rows), or
with the latest row at or before ``t − k`` within ``lag_tolerance`` (on a
clock), and is the pairs' weighted co-moment about the pairs' own weighted
means, each pair weighing its closing row. ``tests/test_ewm_moments.py``
holds the bank to that definition written in numpy, and
``tests/test_reference_ewm_moments.py`` holds the numpy reference to
statsmodels' ``acovf``. Here the libraries build both halves of it
themselves:

- **the pairs**: a shift of ``k`` rows, or Polars' ``join_asof(strategy=
  "backward", tolerance=)``, the as-of rule the lag is defined by;
- **the moments of the pairs**: pandas' ``expanding().cov(ddof=0)`` and
  ``.corr()`` with no decay, ``ewm(halflife=).cov(bias=True)`` and
  ``.corr()`` under decay on rows, and on a clock pandas' ``ewm(halflife=,
  times=).mean()`` of the legs and their products, since it computes no
  ``cov`` with ``times``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from rheon import polars as rp

TIER = "essential"

U = 2.0**-53


def _ar(n: int, seed: int, phi: float = 0.6) -> np.ndarray:
    rng = np.random.default_rng([563, seed])
    e = rng.standard_normal(n)
    x = np.empty(n)
    x[0] = e[0]
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


def _fit(spec: dict, frame: pl.DataFrame) -> pl.DataFrame:
    return rp.ModelBank([spec]).fit_predict(frame)[spec["name"]].struct.unnest()


def _bound(n: int, ca: np.ndarray, cb: np.ndarray) -> np.ndarray:
    """Both sides fold one pair a row into a running weighted mean and
    co-moment: a few roundings each, ``c u`` relative to the pairs' spread
    ``√(var_a var_b)``, carried by every later row, so at most ``n`` times
    that after ``n`` rows. ``c = 16`` covers either update (Pébay's merge
    here, pandas' online ``ewmcov`` there) with room. Measured 2026-11-24
    (x86-64, Linux): at most 6.4e-4 of it, and the correlations at most
    2.4e-4 of theirs."""
    return 2 * 16 * n * U * np.sqrt(np.abs(ca * cb))


@pytest.mark.parametrize("k", [1, 3])
def test_a_row_lag_is_pandas_expanding_cov_of_the_shifted_pairs(k):
    """With no decay and unit weights, before row ``T`` the lag-``k``
    autocovariance is pandas' ``expanding().cov(ddof=0)`` of the pairs
    ``(x_t, x_{t−k})`` closed before it, and the autocorrelation the same
    pairs' ``expanding().corr()``; the cross term ``{x: 1, y@k: 1}`` pairs
    ``x_t`` with ``y_{t−k}`` the same way."""
    n = 500
    x, y = _ar(n, 1), _ar(n, 2, 0.3) + 0.4 * _ar(n, 1)
    frame = pl.DataFrame({"x": x, "y": y})
    lag = f"x@{k}"
    spec = rp.spec.ewm_moments(
        "m",
        terms={
            "acov": {"x": 1, lag: 1},
            "acorr": {"x": 1, lag: 1},
            "xcov": {"x": 1, f"y@{k}": 1},
        },
        kind={"acorr": "standardized"},
        bias=True,
        half_life=float("inf"),
    )
    out = _fit(spec, frame)
    a, b, by = pd.Series(x[k:]), pd.Series(x[:-k]), pd.Series(y[:-k])
    pairs = n - k
    want = {
        "acov": a.expanding().cov(b, ddof=0).to_numpy(),
        "acorr": a.expanding().corr(b).to_numpy(),
        "xcov": a.expanding().cov(by, ddof=0).to_numpy(),
    }
    va = a.expanding().var(ddof=0).to_numpy()
    vb = b.expanding().var(ddof=0).to_numpy()
    vy = by.expanding().var(ddof=0).to_numpy()
    # Row T reads the pairs closed by rows k .. T-1: pandas' entry T-1-k.
    for row in range(k + 1):
        assert out["acov"][row] is None, row
    rows = np.arange(k + 1, n)
    idx = rows - 1 - k
    got = {f: out[f].to_numpy()[rows] for f in want}
    tol = _bound(pairs, va[idx], vb[idx])
    assert np.all(np.abs(got["acov"] - want["acov"][idx]) <= tol)
    assert np.all(np.abs(got["xcov"] - want["xcov"][idx]) <= _bound(pairs, va[idx], vy[idx]))
    # The correlation, a ratio of three such sums, each off by its bound
    # relative to the spread: three times the relative bound.
    ok = idx >= 2
    rel = 3 * 2 * 16 * pairs * U
    assert np.all(np.abs(got["acorr"][ok] - want["acorr"][idx][ok]) <= rel)


@pytest.mark.parametrize("h", [25.0, 200.0])
def test_under_decay_on_rows_it_is_pandas_ewm_cov_of_the_pairs(h):
    """Under a half-life of ``h`` rows the pairs are weighed by their closing
    row's age, one pair a row: pandas' ``ewm(halflife=h, adjust=True)``
    over the pair stream, ``cov(bias=True)`` and ``corr()``."""
    n, k = 600, 2
    x = _ar(n, 3) + 5.0
    frame = pl.DataFrame({"x": x})
    spec = rp.spec.ewm_moments(
        "m",
        terms={"acov": {"x": 1, "x@2": 1}, "acorr": {"x": 1, "x@2": 1}},
        kind={"acorr": "standardized"},
        bias=True,
        half_life=h,
    )
    out = _fit(spec, frame)
    a, b = pd.Series(x[k:]), pd.Series(x[:-k])
    ew_a, ew_b = a.ewm(halflife=h), b.ewm(halflife=h)
    cov = ew_a.cov(b, bias=True).to_numpy()
    corr = ew_a.corr(b).to_numpy()
    va, vb = ew_a.var(bias=True).to_numpy(), ew_b.var(bias=True).to_numpy()
    rows = np.arange(k + 3, n)
    idx = rows - 1 - k
    got_cov = out["acov"].to_numpy()[rows]
    got_corr = out["acorr"].to_numpy()[rows]
    assert np.all(np.abs(got_cov - cov[idx]) <= _bound(n, va[idx], vb[idx]))
    assert np.all(np.abs(got_corr - corr[idx]) <= 3 * 2 * 16 * n * U)
    # The control: the undecayed pairs' covariance is another number.
    flat = a.expanding().cov(b, ddof=0).to_numpy()
    assert np.abs(got_cov[-1] - flat[idx[-1]]) > 1e3 * _bound(n, va[idx], vb[idx])[-1]


def _asof_pairs(t: np.ndarray, x: np.ndarray, k: float, tol: float) -> pl.DataFrame:
    """Polars' as-of join, the lag's definition: each row with the latest
    row at or before ``t − k``, no further back than ``tol``."""
    left = pl.DataFrame({"t": t, "at": t - k, "a": x})
    right = pl.DataFrame({"tb": t, "b": x})
    return left.join_asof(
        right, left_on="at", right_on="tb", strategy="backward", tolerance=tol
    ).with_row_index("row")


@pytest.mark.parametrize("tol", [1.0, 2.0])
def test_a_clock_lag_is_polars_asof_join_and_pandas_ewm_on_times(tol):
    """On an uneven clock of whole seconds, a lag of 3 with ``lag_tolerance``
    ``tol``: the pairs are Polars' ``join_asof`` of each row's ``t − 3`` on
    the rows' clocks, backward, within ``tol``; their decayed co-moment is
    pandas' ``ewm(halflife=, times=)`` means over the pairs at their
    closing rows' times, of each leg and of their products. Whole seconds
    put rows at exactly ``tol`` and at ``tol + 1`` past their as-of point,
    so both sides of the tolerance are met."""
    n, k, h = 700, 3.0, 40.0
    rng = np.random.default_rng(565)
    t = np.cumsum(rng.integers(1, 5, n)).astype(float)
    x = _ar(n, 4) + 2.0
    pairs = _asof_pairs(t, x, k, tol)
    reach = (pairs["at"] - pairs["tb"]).to_numpy()
    assert np.any(reach == tol) and np.any(np.isnan(reach))
    frame = pl.DataFrame({"t": t, "x": x})
    spec = rp.spec.ewm_moments(
        "m",
        terms={"acov": {"x": 1, "x@3": 1}, "acorr": {"x": 1, "x@3": 1}, "lag": {"x@3": 1}},
        kind={"acorr": "standardized", "lag": "raw"},
        bias=True,
        clock="t",
        gap_cap=100.0,
        lag_tolerance=tol,
        half_life=h,
    )
    out = _fit(spec, frame)
    closed = pairs.drop_nulls("b")
    when = pd.to_datetime((closed["t"].to_numpy() * 1e9).astype(np.int64))
    life = pd.Timedelta(seconds=h)
    # pandas computes no ``cov`` with ``times=``, only the mean: the moments
    # are its means of the pairs and their products, the pairs moved by
    # their plain means first so that ``E[ab] − E[a]E[b]`` cancels at the
    # spread's scale, not the level's (a shift moves no central moment).
    a = closed["a"].to_numpy() - closed["a"].mean()
    b = closed["b"].to_numpy() - closed["b"].mean()

    def ew(v: np.ndarray) -> np.ndarray:
        return pd.Series(v).ewm(halflife=life, times=when).mean().to_numpy()

    ma, mb, mab, maa, mbb = ew(a), ew(b), ew(a * b), ew(a * a), ew(b * b)
    cov = mab - ma * mb
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = cov / np.sqrt((maa - ma**2) * (mbb - mb**2))
    mean_b = mb + closed["b"].mean()
    # Each of these is off by its sum's bound at the raw second moments'
    # scale, then by their difference.
    va, vb = maa, mbb
    # Row T reads every pair closed by a row before it: the last such pair.
    closing = closed["row"].to_numpy()
    m = len(closing)
    checked = 0
    for row in range(n):
        j = np.searchsorted(closing, row) - 1
        if j < 2:
            continue
        bound = _bound(m, va[j], vb[j])
        assert abs(out["acov"][row] - cov[j]) <= bound, row
        assert abs(out["acorr"][row] - corr[j]) <= 3 * 2 * 16 * m * U, row
        assert abs(out["lag"][row] - mean_b[j]) <= 2 * 16 * m * U * (abs(mean_b[j]) + 1), row
        checked += 1
    assert checked > n // 2
