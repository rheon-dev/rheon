"""``deco``'s level against scipy's Gaussian likelihood, and ``"linear"``'s
against the paper's DCC recursion on the full ``Q`` (PLAN 551).

Under ``"ew"`` the level ``rho = EW(A) / EW(V)``, ``A`` the row's mean
off-diagonal product and ``V`` its mean square, is the weighted Gaussian
maximum-likelihood estimate of the equicorrelation under the equicovariance
``sigma^2 ((1 - rho) I + rho J)``: Engle and Kelly's Lemma 2.4 with the sums
taken over the weighted window. The oracle is that likelihood, scipy's
``multivariate_normal.logpdf`` summed with the window's weights and maximised
over ``(rho, sigma^2)`` by ``scipy.optimize.minimize``; nothing of the closed
form is written here. The window is the rows the model learned from: each
standardised against the means and variances before it (the model's
``EwDiag``, replicated below), from the first row whose standardiser has
Kish's effective size above 5, each weighted by its row weight and the
decay since.

Under ``"linear"`` the level is the DCC recursion of ``Q`` (the paper's eq.
5, with correlation targeting at the window's mean of ``r r'``) averaged over
its off-diagonal and its diagonal. Averaging is linear, so the full ``n x n``
recursion run here and averaged afterwards must give the model's two
scalars to rounding; the paper's own DECO-DCC level, eq. 7 on the full
``Q``, is held beside it to the distance the pooled diagonal makes.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from scipy import optimize, stats

from rheon import polars as rp

TIER = "essential"


def stream(n_rows, k, rho, seed, scale=1.0):
    """An equicorrelated Gaussian stream with unequal means and scales,
    which the standardiser learns."""
    rng = np.random.default_rng(seed)
    cov = (1 - rho) * np.eye(k) + rho * np.ones((k, k))
    z = rng.standard_normal((n_rows, k)) @ np.linalg.cholesky(cov).T
    loc = np.linspace(-1.0, 2.0, k)
    sd = scale * np.linspace(0.5, 3.0, k)
    return loc + sd * z


def standardised(x, lam, w):
    """The rows as the model's ``EwDiag`` reads them, each against the moments
    before it, and the standardiser's Kish size before each row."""
    n_rows, k = x.shape
    r = np.full((n_rows, k), np.nan)
    kish = np.zeros(n_rows)
    m, c, big_w, w2 = np.zeros(k), np.zeros(k), 0.0, 0.0
    for t in range(n_rows):
        kish[t] = big_w * big_w / w2 if w2 > 0 else 0.0
        with np.errstate(divide="ignore", invalid="ignore"):
            r[t] = np.where(c > 0, (x[t] - m) / np.sqrt(c), np.nan)
        new = lam[t] * big_w + w[t]
        if new > 0:
            a, b = lam[t] * big_w / new, w[t] / new
            d = x[t] - m
            m = m + b * d
            c = a * c + a * b * d * d
        big_w, w2 = new, lam[t] ** 2 * w2 + w[t] ** 2
    return r, kish


def learned(r, kish, lam, w):
    """The weight each row carries in the level read after the last row: its
    own, decayed to the end, where the row has a value in every column and
    its standardiser has Kish's size above 5; 0 elsewhere."""
    n_rows = len(r)
    live = np.isfinite(r).all(1) & (kish > 5) & (w > 0)
    weight = np.zeros(n_rows)
    for t in range(n_rows):
        if live[t]:
            weight[t] = w[t] * np.prod(lam[t + 1 :])
    return weight


def ml_rho(r, weight):
    """The weighted Gaussian ML estimate of ``(rho, sigma^2)``, by scipy."""
    k = r.shape[1]
    rows = r[weight > 0]
    wt = weight[weight > 0]

    def nll(theta):
        rho, var = theta
        if not (-1.0 / (k - 1) < rho < 1.0 and var > 0):
            return 1e300
        cov = var * ((1 - rho) * np.eye(k) + rho * np.ones((k, k)))
        return -(wt @ stats.multivariate_normal(np.zeros(k), cov).logpdf(rows)) / wt.sum()

    # Nelder-Mead stops on the simplex's width alone (``fatol=inf`` drops its
    # test on the spread of the objective): the objective is scipy's
    # ``logpdf``, whose last bits follow the LAPACK and BLAS kernels under it,
    # so on another CPU path its noise (a few 1e-15 of its value) is above any
    # ``fatol`` worth setting and the simplex never reports having converged.
    best = optimize.minimize(
        nll,
        [0.1, 1.0],
        method="Nelder-Mead",
        options=dict(xatol=1e-11, fatol=np.inf, maxiter=20_000),
    )
    assert best.success, best
    # Converged, checked apart from the optimiser's own flag: the objective's
    # gradient at the optimum is small, as the Newton step it implies (the
    # gradient over the Hessian, by central differences), which is the
    # optimum's distance from the true one, taken in ``rho`` (the variance is
    # a nuisance parameter) and held inside half of ``ML_TOL``.
    x0 = best.x
    step = 1e-5
    eye = step * np.eye(2)
    grad = np.array([(nll(x0 + e) - nll(x0 - e)) / (2 * step) for e in eye])
    hess = np.empty((2, 2))
    for i in range(2):
        for j in range(2):
            hess[i, j] = (
                nll(x0 + eye[i] + eye[j])
                - nll(x0 + eye[i] - eye[j])
                - nll(x0 - eye[i] + eye[j])
                + nll(x0 - eye[i] - eye[j])
            ) / (4 * step * step)
    newton = np.linalg.solve(hess, grad)
    assert np.all(np.linalg.eigvalsh(hess) > 0), hess
    assert abs(newton[0]) < ML_TOL / 2, (grad, hess, newton)
    return best.x[0]


def last_level(spec, x, clock=None, weight=None):
    """The model's level after every row: ``rho`` on one more row, read
    before that row is learned."""
    k = x.shape[1]
    names = [f"x{i}" for i in range(k)]
    data = dict(zip(names, np.vstack([x, x[-1:]]).T, strict=True))
    if clock is not None:
        data["t"] = np.r_[clock, clock[-1] + 1.0]
    if weight is not None:
        data["w"] = np.r_[weight, 1.0]
    out = rp.ModelBank([spec]).fit_predict(pl.DataFrame(data))[spec["name"]].struct.unnest()
    return out["rho"][-1]


#: Where scipy's optimum and the closed form part: the optimiser's own
#: stopping rule. Measured at most 6.6e-9 over the cases below (scipy 1.x,
#: Nelder-Mead to `xatol` 1e-11), so the two agree to 8 places; with
#: noise of 1e-14 of its value put in the objective, as an unlucky BLAS path
#: gives, at most 2.3e-8.
ML_TOL = 1e-7


@pytest.mark.parametrize(
    ("k", "rho", "half_life", "seed"),
    [
        (2, 0.6, 60.0, 1),
        (5, 0.3, 60.0, 2),
        (20, 0.9, 60.0, 3),
        (6, -0.1, 60.0, 4),
        (10, 0.5, 15.0, 5),
    ],
)
def test_the_ew_level_is_the_weighted_gaussian_ml_estimate(k, rho, half_life, seed):
    x = stream(400, k, rho, seed)
    n_rows = len(x)
    lam = np.full(n_rows, 2.0 ** (-1.0 / half_life))
    w = np.ones(n_rows)
    spec = rp.spec.deco(
        "d", features=[f"x{i}" for i in range(k)], half_life=half_life, min_weight=0.0
    )
    got = last_level(spec, x)
    r, kish = standardised(x, lam, w)
    weight = learned(r, kish, lam, w)
    want = ml_rho(r, weight)
    assert abs(got - want) <= ML_TOL, (got, want)


def test_the_ew_level_with_weights_and_a_clock_is_the_weighted_ml_estimate():
    """Uneven row weights, rows of weight 0, and an uneven clock: the window's
    weight on a row is its own times the decay since, and a row of weight 0
    is in no clock (its clock step folds into the next row's)."""
    rng = np.random.default_rng(9)
    k, half_life = 4, 30.0
    x = stream(300, k, 0.45, 8)
    n_rows = len(x)
    clock = np.cumsum(rng.uniform(0.5, 1.5, n_rows))
    w = rng.uniform(0.25, 2.0, n_rows)
    w[5::17] = 0.0
    # The decay a row's update reads: the clock since the last row of
    # positive weight, as the bank gives it.
    lam = np.ones(n_rows)
    last = None
    for t in range(n_rows):
        if last is not None:
            lam[t] = 2.0 ** (-(clock[t] - clock[last]) / half_life)
        if w[t] > 0:
            last = t
    lam[w == 0] = 1.0
    spec = rp.spec.deco(
        "d",
        features=[f"x{i}" for i in range(k)],
        half_life=half_life,
        min_weight=0.0,
        clock="t",
        gap_cap=1e9,
        weight="w",
    )
    got = last_level(spec, x, clock=clock, weight=w)
    r, kish = standardised(x, lam, w)
    weight = learned(r, kish, lam, w)
    want = ml_rho(r, weight)
    assert abs(got - want) <= ML_TOL, (got, want)


def full_q(r, kish, lam, alpha, beta):
    """The paper's DCC recursion on the full ``Q``, eq. 5 with correlation
    targeting: ``Q' = (1 - alpha - beta) Qbar' + alpha r r' + beta Q``, Qbar the
    EW mean of ``r r'`` over the rows learned, ``Q`` started at the first such
    row's ``r r'``. Returns, before each row, the pooled level (mean off-
    diagonal over mean diagonal) and eq. 7's (mean of ``q_ij /
    sqrt(q_ii q_jj)``)."""
    n_rows, k = r.shape
    off = ~np.eye(k, dtype=bool)
    pooled, eq7 = np.full(n_rows, np.nan), np.full(n_rows, np.nan)
    qbar, wsum, q = np.zeros((k, k)), 0.0, None
    for t in range(n_rows):
        if q is not None:
            pooled[t] = q[off].mean() / np.diag(q).mean()
            d = np.sqrt(np.diag(q))
            eq7[t] = (q / np.outer(d, d))[off].mean()
        if np.isfinite(r[t]).all() and kish[t] > 5:
            rr = np.outer(r[t], r[t])
            wsum = lam[t] * wsum + 1.0
            qbar = qbar + (rr - qbar) / wsum
            q = rr.copy() if q is None else (1 - alpha - beta) * qbar + alpha * rr + beta * q
        else:
            wsum = lam[t] * wsum
    return pooled, eq7


#: How far the paper's eq. 7 on the full ``Q`` sits from the pooled level over
#: the settled rows, on these streams of unequal scales: on average and at
#: the worst row. Measured: mean 2.7e-3, 1.5e-3 and 1.1e-3, worst 2.5e-2,
#: 8.5e-3 and 6.6e-3 (3, 10 and 30 series). They part where one row moves
#: ``Q``'s diagonal apart, which ``alpha = 0.05`` lets a row do.
EQ7_MEAN_TOL, EQ7_MAX_TOL = 5e-3, 5e-2


@pytest.mark.parametrize(("k", "rho"), [(3, 0.3), (10, 0.6), (30, 0.2)])
def test_the_linear_level_is_the_full_q_recursion_averaged(k, rho):
    alpha, beta, half_life = 0.05, 0.9, 100.0
    x = stream(1500, k, rho, 20 + k)
    n_rows = len(x)
    lam = np.full(n_rows, 2.0 ** (-1.0 / half_life))
    spec = rp.spec.deco(
        "d",
        features=[f"x{i}" for i in range(k)],
        half_life=half_life,
        min_weight=0.0,
        dynamics="linear",
        alpha=alpha,
        beta=beta,
    )
    names = [f"x{i}" for i in range(k)]
    got = (
        rp.ModelBank([spec])
        .fit_predict(pl.DataFrame(dict(zip(names, x.T, strict=True))))["d"]
        .struct.field("rho")
        .to_numpy()
        .astype(float)
    )
    r, kish = standardised(x, lam, np.ones(n_rows))
    pooled, eq7 = full_q(r, kish, lam, alpha, beta)
    assert np.array_equal(np.isnan(got), np.isnan(pooled))
    live = np.isfinite(pooled)
    assert live.sum() > n_rows - 10
    assert np.max(np.abs(got[live] - pooled[live])) < 1e-11
    settled = live & (np.arange(n_rows) > 300)
    gap = np.abs(got[settled] - eq7[settled])
    assert gap.mean() < EQ7_MEAN_TOL and gap.max() < EQ7_MAX_TOL, (gap.mean(), gap.max())
