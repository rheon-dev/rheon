"""`expectile` against statsmodels and scipy (docs/PLAN.md task 501; the
numbers are docs/records/STUDY-501-509.md's, Part A, on the branch `vm/study-501`, not merged).

The oracles:

- ``scipy.stats.expectile(a, alpha, weights=)``, the scalar expectile, for
  a fit of one constant column: a window of rows is the expectile of their
  targets;
- statsmodels' ``RLM`` with ``MQuantileNorm(tau, LeastSquares())``, the
  expectile regression ("expectiles are M-quantiles with the LeastSquares
  as base norm"), run to ``tol=1e-14``. It takes no weights; the loss is
  homogeneous of degree 2, so the weighted fit is ``RLM`` on the rows scaled
  by the square roots of their weights;
- statsmodels' ``WLS`` at ``tau = 0.5``, where the loss is least squares.

Exact checks, each at the study's proposed tolerance (its measured largest
beside it): the refit of the first rows and every row of a window against
``RLM`` on exactly those rows, 1e-12 relative (2.3e-14 and 7.6e-15); the one
pass at ``tau = 0.5`` against the decayed ``WLS``, 1e-12 (3.3e-15); a
window of the constant alone against scipy, 4 ulps of the largest ``|y|``
(2). Statistical, the extended grid: the one pass under decay against the
exact decayed fit, the RMS of the fitted values' gap in residual scales
within the batch fit's own sampling error, ``sqrt(p / n_kish)`` (1.2 of it
at worst over 10 seeds at 500 rows, ``tau = 0.9``; the bound is 2).
"""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl
import pytest
import scipy.stats
import statsmodels.api as sm
from statsmodels.robust.norms import LeastSquares, MQuantileNorm

from rheon import polars as rp

TIER = "mixed"

EXACT = 1e-12


def design(k: int, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """An intercept and ``k - 1`` standard normal features, a target on a
    line with skewed noise (an exponential less its mean), so the levels'
    fits differ by more than rounding."""
    rng = np.random.default_rng([501, seed, k])
    X = np.empty((n, k))
    X[:, 0] = 1.0
    X[:, 1:] = rng.standard_normal((n, k - 1))
    beta = rng.normal(0.0, 1.0, k)
    y = X @ beta + 1.5 * (rng.exponential(1.0, n) - 1.0)
    return X, y


def frame(X: np.ndarray, y: np.ndarray, **cols: np.ndarray) -> pl.DataFrame:
    return pl.DataFrame({**{f"x{j}": X[:, j] for j in range(1, X.shape[1])}, "y": y, **cols})


def rlm(X: np.ndarray, y: np.ndarray, v: np.ndarray, tau: float) -> np.ndarray:
    """The weighted expectile regression by statsmodels (the module docs)."""
    keep = v > 0
    r = np.sqrt(v[keep])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = sm.RLM(r * y[keep], r[:, None] * X[keep], M=MQuantileNorm(tau, LeastSquares()))
        return np.asarray(m.fit(tol=1e-14, maxiter=500).params)


def bank_rows(spec: dict, df: pl.DataFrame) -> pl.DataFrame:
    return rp.ModelBank([spec]).fit_predict(df).select("m").unnest("m")


def coef_after(spec: dict, df: pl.DataFrame) -> np.ndarray:
    """The coefficients once ``df`` is learned, the intercept first."""
    bank = rp.ModelBank([spec])
    bank.fit(df)
    return bank.coef()["coef"].to_numpy()


def rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(a) - b)) / np.max(np.abs(b)))


def spec(k: int, tau: float, **kw: object) -> dict:
    kw = {"half_life": float("inf"), "min_weight": 0.0, "ridge": 0.0, **kw}
    return rp.spec.expectile(
        "m", targets=["y"], features=[f"x{j}" for j in range(1, k)], expectile=tau, **kw
    )


@pytest.mark.parametrize("tau", [0.1, 0.3, 0.7, 0.95])
def test_the_first_rows_refit_is_statsmodels_fit_of_them(tau):
    """On the row its buffer fills, the fit is the exact expectile
    regression of the first ``refit_rows`` rows at their decayed weights."""
    k, n, rows, h = 4, 400, 300, 150.0
    X, y = design(k, n, 1)
    for half_life in (float("inf"), h):
        s = spec(k, tau, half_life=half_life, refit_rows=rows)
        coef = coef_after(s, frame(X[:rows], y[:rows]))
        v = 0.5 ** ((rows - 1 - np.arange(rows)) / half_life)
        want = rlm(X[:rows], y[:rows], v, tau)
        assert rel(coef, want) < EXACT, (half_life, coef, want)


@pytest.mark.parametrize("tau", [0.05, 0.25, 0.75, 0.9])
def test_a_window_is_statsmodels_fit_of_its_rows(tau):
    """Each row's prediction is the fit of the window before it, the exact
    expectile regression of those rows, weights and all."""
    k, n, w = 3, 220, 60
    X, y = design(k, n, 2)
    rng = np.random.default_rng(5011)
    wt = rng.uniform(0.2, 2.0, n)
    wt[rng.random(n) < 0.1] = 0.0
    out = bank_rows(spec(k, tau, window_rows=w, weight="w"), frame(X, y, w=wt))
    pred = out["pred_y"].to_numpy()
    held = np.flatnonzero(wt > 0)
    for t in range(150, n):
        before = held[held < t][-w:]
        b = rlm(X[before], y[before], wt[before], tau)
        want = X[t] @ b
        assert abs(pred[t] - want) <= EXACT * max(1.0, abs(want)), (t, pred[t], want)


def test_a_clock_window_on_a_temporal_clock_is_statsmodels_fit_of_its_span():
    """The rows of the last 30 seconds, decided on the stamps, under both
    edges: each row's prediction is ``RLM``'s fit of exactly those rows."""
    k, n, tau = 2, 260, 0.8
    X, y = design(k, n, 3)
    rng = np.random.default_rng(5012)
    ms = 1_700_000_000_000 + np.cumsum(rng.integers(1, 9, n) * 500)
    t = pl.Series("t", ms).cast(pl.Datetime("ms"))
    for closed in ("right", "both"):
        s = spec(k, tau, window_size="30s", closed=closed, clock="t", gap_cap="1h")
        pred = bank_rows(s, frame(X, y).with_columns(t))["pred_y"].to_numpy()
        for i in range(100, n):
            age = ms[i] - ms[:i]
            inside = age < 30_000 if closed == "right" else age <= 30_000
            idx = np.flatnonzero(inside)
            want = X[i] @ rlm(X[idx], y[idx], np.ones(len(idx)), tau)
            assert abs(pred[i] - want) <= EXACT * max(1.0, abs(want)), (closed, i)


def test_at_one_half_one_pass_is_the_decayed_least_squares_fit():
    """Every side weighs 1/2 at ``tau = 0.5``: the one pass is statsmodels'
    ``WLS`` at the decayed weights, to rounding."""
    k, n, h = 5, 3000, 400.0
    X, y = design(k, n, 4)
    coef = coef_after(spec(k, 0.5, half_life=h, refit_rows=0), frame(X, y))
    v = 0.5 ** ((n - 1 - np.arange(n)) / h)
    want = np.asarray(sm.WLS(y, X, weights=v).fit().params)
    assert rel(coef, want) < EXACT, (coef, want)


@pytest.mark.parametrize("tau", [0.1, 0.5, 0.9])
def test_a_window_of_the_constant_alone_is_scipys_expectile(tau):
    """A fit of one constant column, no intercept, is the scalar expectile:
    each row's prediction is ``scipy.stats.expectile`` of the window's
    targets at their weights, within 4 ulps of their largest magnitude."""
    n, w = 300, 40
    rng = np.random.default_rng([5013, int(tau * 100)])
    y = rng.standard_t(3, n) + 2.0
    wt = rng.uniform(0.1, 3.0, n)
    df = pl.DataFrame({"one": np.ones(n), "y": y, "w": wt})
    s = rp.spec.expectile(
        "m",
        targets=["y"],
        features=["one"],
        expectile=tau,
        ridge=0.0,
        fit_intercept=False,
        half_life=float("inf"),
        window_rows=w,
        weight="w",
        min_weight=0.0,
    )
    pred = bank_rows(s, df)["pred_y"].to_numpy()
    for t in range(w, n):
        a, v = y[t - w : t], wt[t - w : t]
        want = scipy.stats.expectile(a, alpha=tau, weights=v)
        ulp = np.spacing(np.max(np.abs(a)))
        assert abs(pred[t] - want) <= 4 * ulp, (t, (pred[t] - want) / ulp)


def yardstick_gap(k: int, n: int, tau: float, half_life: float, seed: int) -> tuple[float, float]:
    """The one pass's fitted values against the exact decayed fit's, as the
    RMS of their gap over the rows in residual scales, and the yardstick
    ``sqrt(p / n_kish)`` beside it."""
    X, y = design(k, n, seed)
    b = coef_after(spec(k, tau, half_life=half_life, refit_rows=0), frame(X, y))
    v = 0.5 ** ((n - 1 - np.arange(n)) / half_life)
    exact = rlm(X, y, v, tau)
    r = y - X @ exact
    scale = np.sqrt(np.sum(v * r * r) / np.sum(v))
    gap = (X @ (b - exact)) / scale
    rms = float(np.sqrt(np.sum(v * gap * gap) / np.sum(v)))
    kish = np.sum(v) ** 2 / np.sum(v * v)
    return rms, float(np.sqrt(k / kish))


def test_the_one_pass_under_decay_stands_within_the_batch_fits_sampling_error():
    """The essentials' case of the grid below: one level, one seed."""
    rms, yard = yardstick_gap(3, 2000, 0.9, 200.0, 0)
    assert rms <= 2.0 * yard, (rms, yard)


@pytest.mark.extended(reason="10 seeds of 5,000 rows at three levels and two half-lives")
@pytest.mark.parametrize("tau", [0.1, 0.5, 0.9])
@pytest.mark.parametrize("half_life", [100.0, 1000.0])
def test_the_one_pass_grid_under_decay(tau, half_life):
    """The study's R3 under decay: over 10 seeds, the one pass's gap to the
    exact decayed fit is within twice the batch fit's sampling error at the
    worst seed (measured 1.2 at worst), and at ``tau = 0.5`` it is rounding."""
    worst = 0.0
    for seed in range(10):
        rms, yard = yardstick_gap(5, 5000, tau, half_life, seed)
        worst = max(worst, rms / yard)
    if tau == 0.5:
        assert worst < 1e-9, worst
    else:
        assert worst <= 2.0, worst


@pytest.mark.extended(reason="a window refit on every row against statsmodels at five levels")
@pytest.mark.parametrize("tau", [0.02, 0.2, 0.5, 0.8, 0.98])
def test_a_wide_window_grid(tau):
    """Twenty coefficients and a window of 300 rows, the extreme levels
    among them, where the refit halves its step most: every 10th row against
    ``RLM`` on the window's rows."""
    k, n, w = 20, 700, 300
    X, y = design(k, n, 6)
    pred = bank_rows(spec(k, tau, window_rows=w), frame(X, y))["pred_y"].to_numpy()
    for t in range(w, n, 10):
        want = X[t] @ rlm(X[t - w : t], y[t - w : t], np.ones(w), tau)
        assert abs(pred[t] - want) <= EXACT * max(1.0, abs(want)) * 10, (t, pred[t], want)


# --- the ridge, and the one pass after a refit (PLAN 578j, TEST-4) ------------
#
# Every test above sets ``ridge=0``: the default ``1e-6``, a ridge that
# binds, and the one pass that goes on from a refit under a decay had no
# oracle.


def penalized(X: np.ndarray, y: np.ndarray, v: np.ndarray, tau: float, ridge: float) -> np.ndarray:
    """The exact minimiser of ``Σ v a(r) r² + ridge V |slopes|²``, ``a(r)``
    ``tau`` above the fit and ``1 - tau`` below, ``V = Σ v``, the intercept
    (the first column) unpenalized: the definition written here, scipy's
    trust-region Newton computing it from statsmodels' unpenalized fit."""
    from scipy.optimize import minimize

    d = np.ones(X.shape[1])
    d[0] = 0.0
    big_v = v.sum()

    def f(b: np.ndarray) -> tuple[float, np.ndarray]:
        r = y - X @ b
        a = np.where(r > 0, tau, 1.0 - tau)
        g = -2.0 * X.T @ (v * a * r) + 2.0 * ridge * big_v * d * b
        return float(np.sum(v * a * r * r) + ridge * big_v * np.sum(d * b * b)), g

    def hess(b: np.ndarray) -> np.ndarray:
        a = np.where(y - X @ b > 0, tau, 1.0 - tau)
        return 2.0 * (X.T * (v * a)) @ X + 2.0 * ridge * big_v * np.diag(d)

    start = rlm(X, y, v, tau)
    res = minimize(f, start, jac=True, hess=hess, method="trust-exact", options={"gtol": 1e-13})
    return np.asarray(res.x)


@pytest.mark.parametrize("ridge", [1e-6, 0.05, 1.0])
@pytest.mark.parametrize("tau", [0.2, 0.9])
def test_a_ridged_window_is_the_penalized_minimiser_of_its_rows(tau, ridge):
    """A window's fit at ``ridge`` (the default ``1e-6`` among them) is the
    penalized expectile regression of the rows it holds, at their weights:
    the slopes shrunk by ``ridge`` times the window's weight, the intercept
    not at all (the targets sit about 25, far from 0, where a penalized
    intercept would move every prediction)."""
    k, n, w = 3, 200, 50
    X, y = design(k, n, 4)
    y = y + 25.0
    rng = np.random.default_rng(5784)
    wt = rng.uniform(0.2, 2.0, n)
    out = bank_rows(spec(k, tau, window_rows=w, weight="w", ridge=ridge), frame(X, y, w=wt))
    pred = out["pred_y"].to_numpy()
    for t in range(w, n, 7):
        rows = np.arange(t - w, t)
        want = X[t] @ penalized(X[rows], y[rows], wt[rows], tau, ridge)
        assert abs(pred[t] - want) <= 1e-9 * max(1.0, abs(want)), (t, pred[t], want)
    unpenalized = X[n - 1] @ rlm(
        X[n - 1 - w : n - 1], y[n - 1 - w : n - 1], wt[n - 1 - w : n - 1], tau
    )
    if ridge >= 0.05:
        assert abs(pred[n - 1] - unpenalized) > 1e-6, "the ridge binds"


@pytest.mark.parametrize("tau", [0.1, 0.9])
def test_the_one_pass_after_a_refit_under_decay_stands_within_the_batch_fits_sampling_error(tau):
    """The first rows fitted exactly, at the default ridge, then the one
    pass under a half-life of 200 rows: at the last row the fit's gap to the
    exact decayed fit (statsmodels) is within twice the batch fit's sampling
    error, as the one pass with no refit is."""
    k, n, h = 3, 3000, 200.0
    X, y = design(k, n, 7)
    b = coef_after(spec(k, tau, half_life=h, refit_rows=300, ridge=1e-6), frame(X, y))
    v = 0.5 ** ((n - 1 - np.arange(n)) / h)
    exact = rlm(X, y, v, tau)
    r = y - X @ exact
    scale = np.sqrt(np.sum(v * r * r) / np.sum(v))
    gap = (X @ (b - exact)) / scale
    rms = float(np.sqrt(np.sum(v * gap * gap) / np.sum(v)))
    kish = np.sum(v) ** 2 / np.sum(v * v)
    assert rms <= 2.0 * np.sqrt(k / kish), (rms, np.sqrt(k / kish))
