"""Checks on ets.py, with statsmodels as the oracle. Run with the study venv:
``<venv>/bin/python -m pytest scripts/study_485``."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ets  # noqa: E402


def test_fixed_smoother_is_statsmodels():
    """Fixed parameters and known initial states: the one-step predictions are
    ETSModel's fitted values, to rounding (the level grows with the trend)."""
    y = ets.simulate(3000, *ets.TRUTH, seed=31)
    model, _ = ets.ets_fit(y)
    ref = model.smooth(np.array([0.25, 0.05])).fittedvalues
    ours = ets.one_step(0.25, 0.05, y)
    np.testing.assert_allclose(ours, ref, rtol=1e-12, atol=1e-10)


def test_prediction_is_out_of_sample():
    y = ets.simulate(800, *ets.TRUTH, seed=32)
    a = ets.run(y).pred
    y2 = y.copy()
    y2[500] += 25.0
    b = ets.run(y2).pred
    assert a[:501] == b[:501]
    assert a[501] != b[501]


def test_scale_equivariant():
    """Target and initial states scaled by c: alpha and beta do not move,
    sigma2 scales by c^2."""
    y = ets.simulate(3000, *ets.TRUTH, seed=33)
    c = 37.0
    a = ets.run(y)
    b = ets.run(c * y, l0=c * ets.L0, b0=c * ets.B0)
    np.testing.assert_allclose(b.theta, a.theta, rtol=1e-9)
    np.testing.assert_allclose(b.s2, c * c * a.s2, rtol=1e-9)


def test_parameters_stay_admissible():
    y = ets.simulate(3000, 0.9, 0.85, 1.0, seed=34)
    est = ets.OnlineEts()
    for v in y:
        est.update(float(v))
        a, b = est.theta
        assert ets.BOUND <= a <= 1 - ets.BOUND
        assert ets.BOUND * a * (1 - 1e-12) <= b <= (1 - ets.BOUND) * a * (1 + 1e-12)
