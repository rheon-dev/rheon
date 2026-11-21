"""Checks on dcc.py. The definition is written by hand; scipy and statsmodels
compute against it. Run with the study venv:
``<venv>/bin/python -m pytest scripts/study_485``."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest
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


def test_chi2_and_huber_radius_are_scipys():
    """Request A11, remedy (c): the numpy-free chi-squared CDF and the radius
    at the univariate clip's tail probability."""
    from scipy.stats import chi2, norm

    for k in (1, 2, 3, 5):
        for x in (0.3, 2.0, 8.0, 20.0):
            assert dcc.chi2_cdf(x, k) == pytest.approx(chi2.cdf(x, k), rel=1e-12)
        q = dcc.huber_radius(2.0, k)
        assert q == pytest.approx(chi2.ppf(norm.cdf(2.0) - norm.cdf(-2.0), k), rel=1e-10)
    assert dcc.huber_radius(2.0, 1) == pytest.approx(4.0, rel=1e-12)


def test_robust_second_stage_out_of_sample_and_admissible():
    r = dcc.simulate_k(1500, 41, 3, (0.05, 0.93), 0.5, nu=5.0)
    rem = dcc.REMEDIES["huber2"]
    est, _, z = dcc.online_k(r, rem["margins"], **rem["stage2"])
    a, b = est.theta
    assert a >= 0 and b >= 0 and a + b <= 1 - dcc.MARGIN_EPS + 1e-15
    z2 = z.copy()
    z2[1000] *= 30.0
    other = dcc.OnlineDccK(k=3, **rem["stage2"])
    for row in z2:
        other.update(row)
    head = zip(est.pred[:1001], other.pred[:1001], strict=True)
    assert all(np.array_equal(x, y) for x, y in head)
    assert not np.array_equal(est.pred[1001], other.pred[1001])


def test_composite_rows_are_the_pairs_sum():
    """Request A11, part 3: the composite row log-likelihood is the sum over
    pairs of the two-series one, and equals the full one at two series."""
    r = dcc.simulate_k(800, 51, 4, (0.05, 0.93), 0.5)
    z = dcc.batch_margin_z(r)
    rmat = dcc.r_path_k(0.04, 0.9, z, z.T @ z / z.shape[0])
    ref = sum(
        dcc.loglik_rows(rmat[:, i, j], z[:, [i, j]]) for i in range(4) for j in range(i + 1, 4)
    )
    np.testing.assert_allclose(dcc.loglik_rows_cl(rmat, z), ref, rtol=1e-12, atol=1e-12)
    two = rmat[:, :2, :2] / np.sqrt(
        np.einsum("tii->ti", rmat[:, :2, :2])[:, :, None]
        * np.einsum("tii->ti", rmat[:, :2, :2])[:, None, :]
    )
    np.testing.assert_allclose(
        dcc.loglik_rows_cl(two, z[:, :2]), dcc.loglik_rows_k(two, z[:, :2]), atol=1e-12
    )


def test_composite_online_score_is_the_derivative():
    """The composite step's score at a row, parameters held, is the central
    difference of that row's composite log-likelihood along the same Q path."""
    rng = np.random.default_rng(52)
    z = rng.standard_normal((300, 4))

    def run(theta: np.ndarray) -> dcc.OnlineDccCL:
        est = dcc.OnlineDccCL(k=4, a0=theta[0], b0=theta[1], warm=10**9)
        for x in z:
            est.update(x)
        return est

    def row_ll(theta: np.ndarray) -> float:
        rho = run(theta).pred[-1]
        i, j = np.triu_indices(4, 1)
        return float(
            sum(dcc.loglik_rows(np.array([rho[m]]), z[-1:, [i[m], j[m]]])[0] for m in range(len(i)))
        )

    theta = np.array([0.04, 0.9])
    num = approx_fprime(theta, row_ll, centered=True)
    np.testing.assert_allclose(run(theta).last_g, num, rtol=1e-6, atol=1e-8)


def test_composite_online_is_the_full_at_two_series():
    r = dcc.simulate_k(2000, 53, 2, (0.05, 0.93), 0.5)
    _, z = dcc.online_margin_z(r)
    a, c = dcc.OnlineDccK(k=2), dcc.OnlineDccCL(k=2)
    for x in z:
        a.update(x)
        c.update(x)
    np.testing.assert_allclose(c.theta, a.theta, rtol=1e-10)


def test_batch_start_second_stage():
    """Request A11, remedy (a) on the second stage: NaN while buffering, the
    start at the batch fit of the buffered rows, admissible after."""
    r = dcc.simulate_k(1600, 54, 3, (0.05, 0.93), 0.5, nu=5.0)
    rem = dcc.REMEDIES["batch500_huber2"]
    est, rows, z = dcc.online_k(r, rem["margins"], **rem["stage2"])
    assert rows[0] == 500
    assert all(np.all(np.isnan(x)) for x in est.pred[:500])
    assert not any(np.any(np.isnan(x)) for x in est.pred[500:])
    fit = dcc.fit_second_stage(z[:500], z[:500].T @ z[:500] / 500)
    probe = dcc.OnlineDccK(k=3, **rem["stage2"])
    for x in z[:500]:
        probe.update(x)
    np.testing.assert_array_equal(probe.theta, np.array(dcc.project(*fit["theta"])))
    a, b = est.theta
    assert a >= 0 and b >= 0 and a + b <= 1 - dcc.MARGIN_EPS + 1e-15
