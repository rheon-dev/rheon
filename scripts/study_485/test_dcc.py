"""Checks on dcc.py. The definition is written by hand; scipy and statsmodels
compute against it. Run with the study venv:
``<venv>/bin/python -m pytest scripts/study_485``."""

import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import multivariate_normal
from statsmodels.tools.numdiff import approx_fprime

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dcc  # noqa: E402


def test_rho_path_is_the_recursion():
    """``lfilter`` on each entry equals the matrix recursion row by row."""
    rng = np.random.default_rng(41)
    z = rng.standard_normal((500, 2))
    qbar = np.array([[1.1, 0.4], [0.4, 0.9]])
    a, b = 0.07, 0.88
    q = qbar.copy()
    ref = np.empty(500)
    for t in range(500):
        if t > 0:
            q = (1 - a - b) * qbar + a * np.outer(z[t - 1], z[t - 1]) + b * q
        ref[t] = q[0, 1] / math.sqrt(q[0, 0] * q[1, 1])
    np.testing.assert_allclose(dcc.rho_path(a, b, z, qbar), ref, rtol=1e-12, atol=1e-14)


def test_loglik_is_the_bivariate_normal():
    """The second-stage row log-likelihood is the bivariate normal's log density
    with correlation rho, less that of two independent standard normals."""
    rng = np.random.default_rng(42)
    z = rng.standard_normal((50, 2))
    rho = rng.uniform(-0.9, 0.9, 50)
    ref = np.array(
        [
            multivariate_normal(cov=[[1, r], [r, 1]]).logpdf(x)
            - multivariate_normal(cov=np.eye(2)).logpdf(x)
            for x, r in zip(z, rho, strict=True)
        ]
    )
    np.testing.assert_allclose(dcc.loglik_rows(rho, z), ref, rtol=1e-10, atol=1e-12)


def test_online_score_is_the_derivative():
    """The step's score at a row, with the parameters held, is the numerical
    derivative of that row's log-likelihood along the same Q path."""
    rng = np.random.default_rng(43)
    z = rng.standard_normal((300, 2))

    def row_ll(theta: np.ndarray) -> float:
        est = dcc.OnlineDcc(a0=theta[0], b0=theta[1], warm=10**9)
        for x in z[:-1]:
            est.update(x)
        return float(dcc.loglik_rows(np.array([_next_rho(est)]), z[-1:])[0])

    est = dcc.OnlineDcc(a0=0.05, b0=0.9, warm=10**9)
    for x in z[:-1]:
        est.update(x)
    rho = _next_rho(est)
    q = _next_q(est)
    s = math.sqrt(q[0, 0] * q[1, 1])
    zz = np.outer(est.prev_z, est.prev_z)
    dqa = zz - est.qbar + 0.9 * est.dqa
    dqb = est.q - est.qbar + 0.9 * est.dqb
    grad = np.array(
        [dq[0, 1] / s - 0.5 * rho * (dq[0, 0] / q[0, 0] + dq[1, 1] / q[1, 1]) for dq in (dqa, dqb)]
    )
    z1, z2 = z[-1]
    d = 1 - rho * rho
    dl = rho / d + (z1 * z2 * d - rho * (z1 * z1 + z2 * z2 - 2 * rho * z1 * z2)) / (d * d)
    # Qbar moves with the rows but not with theta, so with theta held the carried
    # sensitivity is exact and agrees with the numerical derivative to its error.
    np.testing.assert_allclose(dl * grad, approx_fprime(np.array([0.05, 0.9]), row_ll), rtol=1e-4)


def _next_q(est: dcc.OnlineDcc) -> np.ndarray:
    a, b = est.theta
    return (1 - a - b) * est.qbar + a * np.outer(est.prev_z, est.prev_z) + b * est.q


def _next_rho(est: dcc.OnlineDcc) -> float:
    q = _next_q(est)
    return q[0, 1] / math.sqrt(q[0, 0] * q[1, 1])


def test_prediction_is_out_of_sample():
    rng = np.random.default_rng(44)
    z = rng.standard_normal((400, 2))
    a = dcc.OnlineDcc()
    for x in z:
        a.update(x)
    z2 = z.copy()
    z2[250] *= 5.0
    b = dcc.OnlineDcc()
    for x in z2:
        b.update(x)
    assert a.pred[:251] == b.pred[:251]
    assert a.pred[251] != b.pred[251]


def test_parameters_stay_admissible_and_q_positive_definite():
    r = dcc.simulate(1500, seed=45)
    est, _, z = dcc.online(r)
    chk = dcc.OnlineDcc()
    for x in z:
        chk.update(x)
        a, b = chk.theta
        assert a >= 0 and b >= 0 and a + b <= 1 - dcc.MARGIN_EPS + 1e-15
        assert np.all(np.linalg.eigvalsh(chk.q) > 0)
    assert np.array_equal(chk.theta, est.theta)


def test_scale_equivariant():
    """Returns scaled by c: the standardized residuals, and so (a, b) and every
    correlation, do not move (to rounding)."""
    r = dcc.simulate(1500, seed=46)
    a, _, _ = dcc.online(r)
    b, _, _ = dcc.online(37.0 * r)
    np.testing.assert_allclose(b.theta, a.theta, rtol=1e-8)
    np.testing.assert_allclose(b.pred, a.pred, rtol=1e-8, atol=1e-12)
