"""Checks on batch_core.py, the core-held batch optimizer of request A12, with
the libraries it replaces as oracles. Run with the study venv."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arma  # noqa: E402
import batch_core  # noqa: E402
import dcc  # noqa: E402
import ets  # noqa: E402
import garch  # noqa: E402
import vt_garch  # noqa: E402


@pytest.mark.parametrize("seed", [485006, 485009])
def test_garch_vt_fit_reaches_archs_maximum(seed):
    r = garch.simulate(500, *vt_garch.truth_at(0.99), seed=seed)
    fit = batch_core.garch_vt_fit(r, garch.backcast(r[: garch.BACKCAST_ROWS] ** 2))
    _, res = vt_garch.vt_fit(r)
    assert fit.stop == "converged"
    assert fit.value >= res.loglikelihood - 1e-6
    np.testing.assert_allclose(fit.theta, np.asarray(res.params)[:2], atol=2e-4)


def test_garch_objective_gradient_is_the_derivative():
    from statsmodels.tools.numdiff import approx_fprime

    r = garch.simulate(500, *vt_garch.TRUTH, seed=61)
    f = batch_core.garch_vt_objective(r, garch.backcast(r[:75] ** 2))
    th = np.array([0.07, 0.9])
    num = approx_fprime(th, lambda x: f(x, False)[0], centered=True)
    np.testing.assert_allclose(f(th, True)[1], num, rtol=1e-6)


def test_ets_fit_is_statsmodels():
    y = ets.simulate(500, 0.6, 0.2, 1.0, seed=62)
    fit = batch_core.ets_fit(y, ets.L0, ets.B0)
    _, res = ets.ets_fit(y)
    assert fit.stop == "converged"
    np.testing.assert_allclose(fit.theta, np.asarray(res.params), atol=1e-5)


def test_arma_fit_is_conditional_least_squares():
    """The core fit's objective is conditional least squares; scipy's minimum
    of the same objective (arma.weighted_css_fit, no decay) is its oracle."""
    y = arma.simulate(500, 0.8, 0.3, 1.0, seed=63)
    fit = batch_core.arma_fit(y)
    ref = arma.weighted_css_fit(y, np.inf)[:2]
    assert fit.stop == "converged"
    assert fit.value >= batch_core.arma_objective(y)(ref, False)[0] - 1e-9
    np.testing.assert_allclose(fit.theta, ref, atol=1e-5)


def test_dcc_fit_is_slsqps():
    r = dcc.simulate_k(1200, 64, 3, (0.05, 0.93), 0.5, nu=5.0)
    _, z = dcc.online_margin_z(r)
    zb = z[:500]
    fit = batch_core.dcc_fit(zb)
    ref = dcc.fit_second_stage(zb, zb.T @ zb / 500)
    assert fit.stop == "converged"
    assert fit.value >= ref["llf"] - 1e-8
    np.testing.assert_allclose(fit.theta, ref["theta"], atol=1e-5)


def test_projection_holds_at_the_boundary():
    """A start grid at the persistence cap stays admissible through the fit."""
    r = garch.simulate(500, 0.001, 0.05, 0.949, seed=65)
    fit = batch_core.garch_vt_fit(r, garch.backcast(r[:75] ** 2))
    a, b = fit.theta
    assert a >= 0 and b >= 0 and a + b <= 1 - 1e-4 + 1e-15


def test_every_silent_row_has_a_reason():
    r = garch.simulate(1000, *vt_garch.TRUTH, seed=66)
    est = garch.run(r, **vt_garch.REMEDIES["batch500core_huber2"])
    silent = [i for i, p in enumerate(est.pred) if not np.isfinite(p)]
    assert silent == list(range(500)) and sorted(est.reasons) == silent
    assert all("core's optimizer" in est.reasons[i] for i in silent)


@pytest.mark.parametrize(("seed", "col"), [(485046, 2), (485031, 2)])
def test_garch_vt_fit_at_the_alpha_bound(seed, col):
    """Two buffers whose maximum has alpha at 0 (t(5) margins of the DCC grid):
    the reduced step reaches arch's maximum, which the full step did not."""
    x = dcc.simulate_k(10000, seed, 3, (0.05, 0.93), 0.5, nu=5.0)[:500, col]
    fit = batch_core.garch_vt_fit(x, garch.backcast(x[:75] ** 2))
    _, res = vt_garch.vt_fit(x)
    assert fit.theta[0] == 0.0 and fit.stop == "converged"
    assert fit.value >= res.loglikelihood - 1e-6


def test_digamma_is_scipys():
    from scipy.special import digamma

    for x in (0.3, 1.0, 2.5, 4.4, 6.0, 17.0, 120.0):
        assert batch_core.digamma(x) == pytest.approx(float(digamma(x)), rel=0, abs=2e-11)


def test_t_row_is_scipys_multivariate_t():
    """Request A12 part 2: the second stage's t density, unit variances."""
    from scipy.stats import multivariate_t

    rng = np.random.default_rng(71)
    rmat = np.array([[1.0, 0.4, 0.2], [0.4, 1.0, 0.3], [0.2, 0.3, 1.0]])
    for nu in (3.5, 5.0, 30.0):
        z = rng.standard_normal((20, 3))
        ref = multivariate_t(shape=rmat * (nu - 2.0) / nu, df=nu).logpdf(z)
        ours = [batch_core.dcc_t_row(x, rmat, nu)[0] for x in z]
        np.testing.assert_allclose(ours, ref, rtol=1e-12)
        np.testing.assert_allclose(dcc.loglik_rows_t(np.repeat(rmat[None], 20, 0), z, nu), ref)


def test_t_objective_gradient_is_the_derivative():
    from statsmodels.tools.numdiff import approx_fprime

    r = dcc.simulate_k(1200, 72, 3, (0.05, 0.93), 0.5, nu=5.0)
    _, z = dcc.online_margin_z(r)
    zb = z[:300]
    f = batch_core.dcc_t_objective(zb, zb.T @ zb / 300)
    th = np.array([0.04, 0.9, 6.0])
    num = approx_fprime(th, lambda x: f(x, False)[0], centered=True)
    np.testing.assert_allclose(f(th, True)[1], num, rtol=1e-6)


def test_dcc_t_fit_reaches_the_maximum():
    """On this 500-row buffer SLSQP from its fixed start stops 0.23 short of
    the maximum while reporting success (a flat ridge); the core fit reaches
    at least SLSQP's value, and SLSQP restarted at the core's point finds
    nothing higher."""
    from scipy.optimize import minimize

    r = dcc.simulate_k(1200, 73, 3, (0.05, 0.93), 0.5, nu=5.0)
    _, z = dcc.online_margin_z(r)
    zb = z[:500]
    qbar = zb.T @ zb / 500
    fit = batch_core.dcc_t_fit(zb)
    ref = dcc.fit_second_stage_t(zb, qbar)
    assert fit.stop == "converged" and fit.value >= ref["llf"] - 1e-6

    def neg(th: np.ndarray) -> float:
        a, b, nu = th
        if a < 0 or b < 0 or a + b >= 1 or nu <= 2:
            return 1e12
        return -float(np.sum(dcc.loglik_rows_t(dcc.r_path_k(a, b, zb, qbar), zb, nu)))

    polish = minimize(neg, fit.theta, method="Nelder-Mead", options={"xatol": 1e-10})
    assert -polish.fun <= fit.value + 1e-6


def test_t_stage_score_is_the_derivative():
    """The t stage's per-row score in (a, b), nu held, is the central
    difference of that row's t log density along the same Q path."""
    from statsmodels.tools.numdiff import approx_fprime

    rng = np.random.default_rng(74)
    z = rng.standard_normal((200, 3))

    def run(theta: np.ndarray) -> dcc.OnlineDccT:
        est = dcc.OnlineDccT(k=3, a0=theta[0], b0=theta[1], warm=10**9, nu=6.0)
        for x in z:
            est.update(x)
        return est

    def row_ll(theta: np.ndarray) -> float:
        est = run(theta)
        rho = est.pred[-1]
        rmat = np.eye(3)
        rmat[np.triu_indices(3, 1)] = rho
        rmat = np.triu(rmat) + np.triu(rmat, 1).T
        return batch_core.dcc_t_row(z[-1], rmat, 6.0)[0]

    th = np.array([0.04, 0.9])
    num = approx_fprime(th, row_ll, centered=True)
    np.testing.assert_allclose(run(th).last_g, num, rtol=1e-6, atol=1e-9)


def test_shared_start_silent_rows():
    """The shared start: 500 silent rows, each with the shared reason, and the
    second stage begins at the core fit of the margins' buffered rows."""
    out = dcc.compare_k(1500, 75, 3, (0.05, 0.93), 0.5, 5.0, "shared500core_t")
    assert out["silent_rows"] == 500 and out["silent_unexplained"] == 0
    (reason,) = out["silent_reasons"]
    assert "shared batch start" in reason
    assert out["first_dcc_row"] == 500.0
