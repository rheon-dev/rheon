"""Checks on arma.py, with statsmodels and scipy as the oracles. Run with the
study venv: ``<venv>/bin/python -m pytest scripts/study_485``."""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import lfilter
from statsmodels.tsa.statespace.sarimax import SARIMAX

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arma  # noqa: E402


def test_residuals_are_scipys_filter():
    y = arma.simulate(3000, *arma.TRUTH, seed=21)
    ours = arma.residuals(0.55, 0.35, y)
    ref = lfilter([1.0, -0.55], [1.0, 0.35], y)
    np.testing.assert_allclose(ours, ref, rtol=1e-12, atol=1e-12)


def test_fixed_predictions_meet_the_kalman_filter():
    """The zero start differs from statsmodels' stationary start by a term that
    shrinks as theta^t; after 300 rows the one-step predictions agree to the
    Kalman filter's own rounding (3.2e-10 at most, measured on this series)."""
    y = arma.simulate(2000, *arma.TRUTH, seed=22)
    params = np.array([0.6, 0.3, 1.0])
    ref = SARIMAX(y, order=(1, 0, 1), trend="n").filter(params).forecasts[0]
    ours = y - arma.residuals(params[0], params[1], y)
    assert np.max(np.abs(ours[300:] - ref[300:])) < 1e-9


def test_prediction_is_out_of_sample():
    y = arma.simulate(800, *arma.TRUTH, seed=23)
    a = arma.run(y).pred
    y2 = y.copy()
    y2[500] += 25.0
    b = arma.run(y2).pred
    assert a[:501] == b[:501]
    assert a[501] != b[501]


@pytest.mark.parametrize("half_life", [np.inf, 300.0])
def test_scale_equivariant(half_life):
    y = arma.simulate(3000, *arma.TRUTH, seed=24)
    c = 37.0
    a, b = arma.run(y, half_life=half_life), arma.run(c * y, half_life=half_life)
    np.testing.assert_allclose(b.beta, a.beta, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(b.s2, c * c * a.s2, rtol=1e-9)


def test_parameters_stay_admissible():
    y = arma.simulate(3000, 0.95, -0.9, 1.0, seed=25)
    est = arma.OnlineArma()
    for v in y:
        est.update(float(v))
        assert np.all(np.abs(est.beta) <= 1 - est.margin)


def test_no_decay_is_the_unweighted_fit():
    y = arma.simulate(1500, *arma.TRUTH, seed=26)
    assert np.array_equal(arma.run(y).params, arma.run(y, half_life=np.inf).params)
    w = arma.weighted_css_fit(y, np.inf)
    e = lfilter([1.0, -w[0]], [1.0, w[1]], y)
    assert w[2] == pytest.approx(np.mean(e * e), rel=1e-12)


def test_robust_score_out_of_sample_admissible_scale_equivariant():
    """Request A11, remedy (c): the Huber-clipped step."""
    y = arma.simulate(3000, 0.8, 0.3, 1.0, seed=26, nu=5.0)
    a = arma.run(y, robust_c=2.0)
    assert np.all(np.abs(a.beta) <= 1 - a.margin)
    y2 = y.copy()
    y2[2000] += 50.0
    b = arma.run(y2, robust_c=2.0)
    assert a.pred[:2001] == b.pred[:2001] and a.pred[2001] != b.pred[2001]
    c = arma.run(37.0 * y, robust_c=2.0)
    np.testing.assert_allclose(c.beta, a.beta, rtol=1e-9)
    np.testing.assert_allclose(c.s2, 37.0**2 * a.s2, rtol=1e-9)


def test_robust_default_is_the_plain_estimator():
    y = arma.simulate(2000, *arma.TRUTH, seed=27)
    plain, named = arma.run(y), arma.run(y, robust_c=0.0)
    assert np.array_equal(plain.beta, named.beta) and plain.pred == named.pred


def test_batch_start_emits_nothing_then_starts_at_sarimax():
    """Request A11, remedy (a): the buffered rows emit NaN, the start is
    SARIMAX's fit of them, and the predictions stay out of sample."""
    y = arma.simulate(1500, 0.8, 0.3, 1.0, seed=28)
    est = arma.OnlineArma(batch_start=500, robust_c=2.0)
    for v in y[:500]:
        est.update(float(v))
    _, res = arma.sarimax_fit(y[:500])
    np.testing.assert_array_equal(est.beta, np.asarray(res.params)[:2])
    for v in y[500:]:
        est.update(float(v))
    assert all(np.isnan(est.pred[:500])) and not any(np.isnan(est.pred[500:]))
    y2 = y.copy()
    y2[1000] += 50.0
    other = arma.run(y2, batch_start=500, robust_c=2.0).pred
    assert est.pred[:1001] == other[:1001] and est.pred[1001] != other[1001]


def test_batch_start_default_is_the_plain_estimator():
    y = arma.simulate(2000, *arma.TRUTH, seed=29)
    plain, named = arma.run(y), arma.run(y, batch_start=0)
    assert np.array_equal(plain.beta, named.beta) and plain.pred == named.pred
