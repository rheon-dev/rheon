"""Checks on ets.py, with statsmodels as the oracle. Run with the study venv:
``<venv>/bin/python -m pytest scripts/study_485``."""

import sys
from pathlib import Path

import numpy as np
import pytest

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


REMEDIES = [{"batch_start": 500}, {"gain_t0": 500.0}, {"gain_t0": 2000.0}]


@pytest.mark.parametrize("kw", REMEDIES, ids=str)
def test_remedies_out_of_sample_and_admissible(kw):
    """Request A11's remedies: out of sample, admissible at every row, and the
    buffered rows of the batch start emit nothing."""
    y = ets.simulate(1200, 0.6, 0.2, 1.0, seed=35)
    est = ets.OnlineEts(**kw)
    for v in y:
        est.update(float(v))
        a, b = est.theta
        assert ets.BOUND <= a <= 1 - ets.BOUND
        assert ets.BOUND * a * (1 - 1e-12) <= b <= (1 - ets.BOUND) * a * (1 + 1e-12)
    w = int(kw.get("batch_start", 0))
    assert all(np.isnan(est.pred[:w])) and not any(np.isnan(est.pred[w:]))
    y2 = y.copy()
    y2[900] += 25.0
    other = ets.run(y2, **kw).pred
    assert est.pred[:901] == other[:901] and est.pred[901] != other[901]


def test_remedy_defaults_are_part_5():
    y = ets.simulate(2000, 0.6, 0.2, 1.0, seed=36)
    plain, named = ets.run(y), ets.run(y, batch_start=0, gain_t0=0.0)
    assert np.array_equal(plain.theta, named.theta) and plain.pred == named.pred


def test_batch_start_starts_at_statsmodels_fit():
    y = ets.simulate(600, 0.6, 0.2, 1.0, seed=37)
    est = ets.OnlineEts(batch_start=500)
    for v in y[:500]:
        est.update(float(v))
    _, res = ets.ets_fit(y[:500])
    np.testing.assert_array_equal(est.theta, np.array(ets.project(*map(float, res.params[:2]))))
