"""Checks on garch.py, with ``arch`` as the oracle. Run with the study venv:
``<venv>/bin/python -m pytest scripts/study_485`` (not collected by the
repository's suite, whose environment has no ``arch``)."""

import sys
from pathlib import Path

import numpy as np
import pytest
from arch.univariate import GARCH

sys.path.insert(0, str(Path(__file__).resolve().parent))
import garch  # noqa: E402

TRUTH = (0.02, 0.08, 0.90)


def test_filter_matches_arch_compute_variance():
    """Fixed parameters, the backcast given, bounds that cannot bind: the
    row-by-row filter is arch's recursion, to 1e-13 relative."""
    r = garch.simulate(5000, *TRUTH, seed=11)
    theta = np.array([0.03, 0.1, 0.85])
    start = garch.backcast(r[: garch.BACKCAST_ROWS] ** 2)
    bounds = np.tile([0.0, np.inf], (r.shape[0], 1))
    ref = GARCH(p=1, o=0, q=1).compute_variance(theta, r, np.empty_like(r), start, bounds)
    ours = garch.filter_variance(theta, r, start)
    assert np.max(np.abs(ours / ref - 1.0)) < 1e-13


def test_backcast_is_arch_default():
    r = garch.simulate(500, *TRUTH, seed=12)
    assert garch.backcast(r[: garch.BACKCAST_ROWS] ** 2) == GARCH().backcast(r)


def test_loglik_matches_arch_fixed():
    r = garch.simulate(3000, *TRUTH, seed=13)
    am, _ = garch.arch_fit(r)
    theta = np.array([0.025, 0.07, 0.9])
    path = garch.filter_variance(theta, r, garch.backcast(r[: garch.BACKCAST_ROWS] ** 2))
    assert abs(garch.loglik(r, path) - am.fix(theta).loglikelihood) < 1e-9


def test_prediction_is_out_of_sample():
    """Changing row t's return leaves row t's prediction alone."""
    r = garch.simulate(400, *TRUTH, seed=14)
    a = garch.run(r).pred
    r2 = r.copy()
    r2[300] *= 10.0
    b = garch.run(r2).pred
    assert a[:301] == b[:301]
    assert a[301] != b[301]


def test_scale_equivariant():
    """Returns scaled by c: omega and the predictions scale by c^2, alpha and
    beta do not move (to rounding)."""
    r = garch.simulate(3000, *TRUTH, seed=15)
    c = 37.0
    a, b = garch.run(r), garch.run(c * r)
    np.testing.assert_allclose(b.theta[0], c * c * a.theta[0], rtol=1e-9)
    np.testing.assert_allclose(b.theta[1:], a.theta[1:], rtol=1e-9, atol=1e-12)
    pa, pb = np.asarray(a.pred), np.asarray(b.pred)
    np.testing.assert_allclose(
        pb[garch.BACKCAST_ROWS :], c * c * pa[garch.BACKCAST_ROWS :], rtol=1e-9
    )


def test_parameters_stay_admissible():
    r = garch.simulate(3000, *TRUTH, seed=16)
    est = garch.OnlineGarch()
    for x in r:
        est.update(float(x))
        if est.started:
            w, a, b = est.theta
            assert w > 0 and a >= 0 and b >= 0 and a + b <= 1 - est.margin + 1e-15


VARIANTS = [
    {"target": "running"},
    {"target": "warmup"},
    {"step_cap": 0.5},
    {"gain": "sqrt"},
    {"gain": "floor", "gain_floor": 1e-3},
    {"batch_start": True, "warm": 500, "t0": 500.0},
    # request A11's remedies on the targeted form
    {"target": "running", "batch_start": True, "warm": 500, "t0": 500.0},
    {"target": "running", "gain": "offset", "gain_t0": 500.0},
    {"gain": "offset", "gain_t0": 2000.0},
]


@pytest.mark.parametrize("kw", VARIANTS, ids=str)
def test_variants_out_of_sample_and_admissible(kw):
    r = garch.simulate(1200, *TRUTH, seed=17)
    est = garch.OnlineGarch(**kw)
    for x in r:
        est.update(float(x))
        if est.started:
            w, a, b = est.theta
            assert w > 0 and a >= 0 and b >= 0 and a + b <= 1 - est.margin + 1e-15
    r2 = r.copy()
    r2[900] *= 10.0
    other = garch.run(r2, **kw).pred
    assert est.pred[:901] == other[:901]
    assert est.pred[901] != other[901]


@pytest.mark.parametrize("kw", [kw for kw in VARIANTS if not kw.get("batch_start")], ids=str)
def test_variants_scale_equivariant(kw):
    r = garch.simulate(2000, *TRUTH, seed=18)
    c = 37.0
    a, b = garch.run(r, **kw), garch.run(c * r, **kw)
    np.testing.assert_allclose(b.theta[0], c * c * a.theta[0], rtol=1e-8)
    np.testing.assert_allclose(b.theta[1:], a.theta[1:], rtol=1e-8, atol=1e-12)


def test_targeting_holds_omega_to_the_running_variance():
    r = garch.simulate(1000, *TRUTH, seed=19)
    est = garch.run(r, target="running")
    w, a, b = est.theta
    assert w == pytest.approx(np.mean(r * r) * (1 - a - b), rel=1e-12)


def test_defaults_are_the_plain_estimator():
    """No option set: the arithmetic of part 2, to the bit."""
    r = garch.simulate(1000, *TRUTH, seed=20)
    plain = garch.run(r)
    named = garch.run(r, target="", step_cap=0.0, gain="inv", batch_start=False)
    assert np.array_equal(plain.theta, named.theta) and plain.pred == named.pred


def test_targeted_batch_start_starts_at_the_targeted_fit():
    """Request A11, remedy (a) on the targeted form: before any step the
    parameters are the targeted batch fit of the buffered rows, and omega is
    that fit's own target, the buffered rows' mean r^2, times 1 - alpha - beta."""
    import vt_garch

    r = garch.simulate(600, *TRUTH, seed=21)
    est = garch.OnlineGarch(target="running", batch_start=True, warm=500, t0=500.0)
    for x in r[:500]:
        est.update(float(x))
    _, res = vt_garch.vt_fit(r[:500])
    a, b = np.asarray(res.params, dtype=float)[:2]
    assert est.n_steps == 0 and est.theta[1] == a and est.theta[2] == b
    assert est.theta[0] == pytest.approx(np.mean(r[:500] ** 2) * (1 - a - b), rel=1e-15)
    assert all(np.isnan(est.pred))
