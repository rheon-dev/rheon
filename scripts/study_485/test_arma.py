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
