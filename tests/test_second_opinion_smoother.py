"""``kalman``'s fixed-lag smoother against statsmodels' ``KalmanSmoother``.

The smoothed value a row emits for the observation ``L`` before it is the
whole smoother of every row up to that row, read ``L`` back, so it is held
to statsmodels' Rauch-Tung-Striebel smoother run on that prefix: a
time-varying design (the rows' features with an intercept), ``obs_cov =
R / w``, ``state_cov = R (ln 2 / coef_half_life) ** 2`` a row on rows one
clock unit apart, the transition the identity. In units of the noise
(``obs_var`` unset, ``R = 1``) the intercept's prior is diffuse and each
slope's known at ``p0``, statsmodels' mixed exact diffuse initialization;
with ``obs_var`` every coefficient's prior is known, ``p0 * obs_var``, and
the smoothed standard errors are the smoother's covariance itself.
statsmodels computes every form here, so filterpy is not needed.

Then what holds the embargo: a smoothed value depends on the rows after the
row it describes, and nothing it reads reaches a prediction or a value
written before those rows; and the gap to the smoother of the whole stream,
over the standard deviation the filter's own covariances give it, is
standard normal at every row of data drawn from the model's own prior, the
first rows included, with no burn-in.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from scipy import stats
from statsmodels.tsa.statespace.initialization import Initialization
from statsmodels.tsa.statespace.kalman_smoother import KalmanSmoother

from rheon import polars as rp

TIER = "essential"

HL = 40.0


def _frame(x: np.ndarray, y: np.ndarray) -> pl.DataFrame:
    cols = {f"x{i}": x[:, i] for i in range(x.shape[1])}
    return pl.DataFrame({**cols, "y": y})


def _fit(frame: pl.DataFrame, kf: int, lag: int, obs_var: float | None) -> pl.DataFrame:
    spec = rp.spec.kalman(
        "k",
        targets=["y"],
        features=[f"x{i}" for i in range(kf)],
        coef_half_life=HL,
        standardize=False,
        obs_var=obs_var,
        half_life=100.0,
        min_weight=0.0,
        smooth_lag=lag,
        emit_smoothed_se=True,
        max_state_bytes=64 * 1024 * 1024,
    )
    return rp.ModelBank([spec], threads=1).fit_predict(frame).unnest("k")


def _smoother(
    x: np.ndarray, y: np.ndarray, obs_var: float | None, p0: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """statsmodels' smoothed states and their covariances on every row of
    ``y``, mapped from ``kalman`` unstandardized as the module says."""
    n, kf = x.shape
    k = kf + 1
    r = 1.0 if obs_var is None else obs_var
    ks = KalmanSmoother(k_endog=1, k_states=k)
    ks.bind(y[:, None])
    ks["design"] = np.column_stack([np.ones(n), x]).T[None, :, :]
    ks["obs_cov"] = np.full((1, 1, n), r)
    ks["transition"] = np.eye(k)
    ks["selection"] = np.eye(k)
    ks["state_cov"] = r * (np.log(2.0) / HL) ** 2 * np.eye(k)
    init = Initialization(k)
    if obs_var is None:
        init.set(0, "diffuse")
        init.set((1, k), "known", constant=np.zeros(kf), stationary_cov=p0 * np.eye(kf))
    else:
        init.set(None, "known", constant=np.zeros(k), stationary_cov=p0 * r * np.eye(k))
    ks.initialization = init
    res = ks.smooth()
    return res.smoothed_state.T, np.moveaxis(res.smoothed_state_cov, 2, 0)


def _data(n: int, kf: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, kf)) + np.linspace(0.5, -0.5, kf)
    beta = 1.0 + np.cumsum(0.03 * rng.standard_normal((n, kf + 1)), axis=0)
    y = beta[:, 0] + np.sum(x * beta[:, 1:], axis=1) + 0.5 * rng.standard_normal(n)
    return x, y


@pytest.mark.parametrize(
    ("kf", "lag"),
    [(1, 1), (1, 10), (4, 1), (4, 10), (4, 100), (19, 1), (19, 10), (19, 100)],
)
@pytest.mark.parametrize("obs_var", [None, 0.25])
def test_the_smoothed_value_is_statsmodels_smoother_of_the_prefix(kf, lag, obs_var):
    """Each emission against the smoother of its prefix, at 2, 5 and 20
    coefficients and lags of 1, 10 and 100; the standard errors where the
    covariance is absolute (``obs_var``). Within 1e-11 of each value's size
    (at least 1). Measured 2026-11-24 with statsmodels 0.15.0: 8.4e-15 at
    worst (20 coefficients, a lag of 100), the two recursions' rounding."""
    n = lag + 120
    x, y = _data(n, kf, 496 + kf + lag)
    out = _fit(_frame(x, y), kf, lag, obs_var)
    described = out["smoothed_for_y"].to_list()
    coef = out["smoothed_coef"].to_list()
    se = out["smoothed_se_coef"].to_list()
    rows = [t for t in range(n) if described[t] is not None]
    assert rows == list(range(lag, n))
    worst = 0.0
    # Every twelfth emitting row and the last: each prefix is its own run.
    for t in rows[::12] + [rows[-1]]:
        s = int(described[t])
        assert s == t - lag
        m, v = _smoother(x[: t + 1], y[: t + 1], obs_var)
        want = m[s]
        got = np.array(coef[t], dtype=float)
        worst = max(worst, float(np.max(np.abs(got - want) / np.maximum(np.abs(want), 1.0))))
        if obs_var is not None:
            want_se = np.sqrt(np.diag(v[s]))
            got_se = np.array(se[t], dtype=float)
            worst = max(worst, float(np.max(np.abs(got_se - want_se) / np.maximum(want_se, 1.0))))
    assert worst < 1e-11, worst


def test_the_gap_to_the_whole_smoother_is_standard_normal_from_the_first_row():
    """Data drawn from the model's own prior (``obs_var``, every prior
    known): the gap ``b_{s|s+L} - b_{s|N}`` over ``sqrt(P_{s|s+L} -
    P_{s|N})`` is exactly standard normal for every row ``s``, the
    prior-dominated first rows included, since both are conditional means
    of one Gaussian and their difference has that variance. One ``z`` a
    seed, the row and the coefficient taken in turn so every row from the
    first is read, makes ``sum z**2`` a chi-squared of as many degrees as
    seeds; held inside its central 99.9%. Measured 2026-11-24: 426.2 over
    400 seeds, inside (313.4, 499.7)."""
    r, p0, lag, n, kf = 0.25, 1.0, 5, 60, 2
    k = kf + 1
    q = r * (np.log(2.0) / HL) ** 2
    zs = []
    seeds = 400
    for seed in range(seeds):
        rng = np.random.default_rng(10_000 + seed)
        x = rng.standard_normal((n, kf))
        beta = np.empty((n, k))
        beta[0] = rng.normal(0.0, np.sqrt(p0 * r), k)
        for t in range(1, n):
            beta[t] = beta[t - 1] + rng.normal(0.0, np.sqrt(q), k)
        y = beta[:, 0] + np.sum(x * beta[:, 1:], axis=1) + rng.normal(0.0, np.sqrt(r), n)
        out = _fit(_frame(x, y), kf, lag, r)
        m_all, v_all = _smoother(x, y, r, p0)
        # Rows whose lag ends before the stream's last row: at the last,
        # the two smoothers read the same rows and there is no gap.
        s = seed % (n - lag - 1)
        i = (seed // (n - lag - 1)) % k
        t = s + lag
        assert out["smoothed_for_y"][t] == s
        b = out["smoothed_coef"][t][i]
        var = out["smoothed_se_coef"][t][i] ** 2
        gap_var = var - v_all[s][i, i]
        assert gap_var > 0.0, (seed, s, i, gap_var)
        zs.append((b - m_all[s][i]) / np.sqrt(gap_var))
    stat = float(np.sum(np.square(zs)))
    lo, hi = stats.chi2.ppf([0.0005, 0.9995], seeds)
    assert lo < stat < hi, (stat, lo, hi)


def test_nothing_after_a_row_reaches_its_prediction_or_a_value_written_before():
    """Targets changed after row ``t0``: every prediction and every
    smoothed value written up to ``t0`` is the same to the bit, and a value
    written later moves. And each value names a row before its own line."""
    n, kf, lag, t0 = 200, 2, 8, 120
    x, y = _data(n, kf, 4966)
    y2 = y.copy()
    y2[t0 + 1 :] += 5.0
    a = _fit(_frame(x, y), kf, lag, None)
    b = _fit(_frame(x, y2), kf, lag, None)
    assert a["pred_y"][: t0 + 2].equals(b["pred_y"][: t0 + 2])
    assert a["smoothed_coef"][: t0 + 1].equals(b["smoothed_coef"][: t0 + 1])
    assert a["smoothed_for_y"].equals(b["smoothed_for_y"])
    assert not a["smoothed_coef"][t0 + 1 :].equals(b["smoothed_coef"][t0 + 1 :])
    for row, described in enumerate(a["smoothed_for_y"].to_list()):
        assert described is None or described < row
