"""`glm` against statsmodels and glum (docs/PLAN.md task 500; the numbers
are docs/records/STUDY-500-GLM.md's).

Two kinds of check:

- **row by row, exact**: the recursion written longhand in numpy, its
  Fisher weight and working response taken from statsmodels' own family
  objects (``family.weights(mu)``, ``family.link.deriv(mu)``), against the
  bank's coefficients after every row;
- **converged, statistical**: the bank's coefficients against the batch
  fit of the same rows, glum's with the ridge matched (``alpha = R / n``)
  and statsmodels' ``GLM`` with ``freq_weights``, in units of the batch
  fit's standard errors (statsmodels' ``bse``; the Pearson scale for the
  Gamma and Tweedie families). The study's tolerance is 1.5 times the
  largest gap over 50 seeds, rounded up to 0.05: at 5 coefficients and
  10,000 rows (the essentials) Poisson 0.25, Gamma 0.20, Tweedie 0.25 SE;
  at 20 coefficients (the extended grid) 0.75 / 0.30, 0.50 / 0.25 and
  0.95 / 0.50 at 10,000 / 100,000 rows. Each holds at the stated
  dispersion (Gamma of shape 2; Tweedie at ``p = 1.5``, ``phi = 2``): the
  gap in SE grows as the dispersion's square root.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import statsmodels.api as sm

from rheon import polars as rp

TIER = "mixed"

P = 1.5
PHI = 2.0


def design(family: str, k: int, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """The study's design: an intercept and ``k - 1`` standard normal
    features, ``eta`` of spread about 0.5 around 0.5; Poisson counts, Gamma
    amounts of shape 2, or a compound Poisson-Gamma Tweedie amount."""
    code = {"poisson": 0, "gamma": 1, "tweedie": 2}[family]
    rng = np.random.default_rng([500, seed, k, code])
    X = np.empty((n, k))
    X[:, 0] = 1.0
    X[:, 1:] = rng.standard_normal((n, k - 1))
    beta = np.empty(k)
    beta[0] = 0.5
    beta[1:] = rng.normal(0.0, 0.5 / np.sqrt(k - 1), k - 1)
    mu = np.exp(X @ beta)
    if family == "poisson":
        y = rng.poisson(mu).astype(float)
    elif family == "gamma":
        y = rng.gamma(2.0, mu / 2.0)
    else:
        lam = mu ** (2 - P) / (PHI * (2 - P))
        shape = (2 - P) / (P - 1)
        scale = PHI * (P - 1) * mu ** (P - 1)
        N = rng.poisson(lam)
        y = np.zeros(n)
        pos = N > 0
        y[pos] = rng.gamma(N[pos] * shape, scale[pos])
    return X, y


def frame(X: np.ndarray, y: np.ndarray) -> pl.DataFrame:
    return pl.DataFrame({**{f"x{j}": X[:, j] for j in range(1, X.shape[1])}, "y": y})


def online(family: str, X: np.ndarray, y: np.ndarray, **kw: object) -> np.ndarray:
    kw = {"half_life": float("inf"), **kw}
    if family == "tweedie":
        kw["var_power"] = P
    s = rp.spec.glm(
        "m",
        targets=["y"],
        features=[f"x{j}" for j in range(1, X.shape[1])],
        family=family,
        **kw,
    )
    bank = rp.ModelBank([s])
    bank.fit(frame(X, y))
    return np.array([r["coef"] for r in bank.coef().iter_rows(named=True)])


def sm_family(family: str) -> sm.families.Family:
    log = sm.families.links.Log()
    return {
        "poisson": sm.families.Poisson(log),
        "gamma": sm.families.Gamma(log),
        "tweedie": sm.families.Tweedie(log, var_power=P),
        "binomial": sm.families.Binomial(sm.families.links.Logit()),
    }[family]


# glum runs in a process of its own. Its matrix library, tabmat, and
# scikit-learn each bundle their own copy of the OpenMP runtime, and XGBoost
# and LightGBM (tests/test_gbdt.py) load Homebrew's: with these runtimes in
# one process, a glum fit hung in a barrier on a worker thread, and the
# pytest workers segfaulted in an idle OpenMP thread. A process holding only
# glum, with one OpenMP thread, does neither.
_GLUM = """
import sys
import numpy as np
from glum import GeneralizedLinearRegressor, TweedieDistribution
d, family, ridge, power = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
z = np.load(d + "/in.npz")
X, y = z["X"], z["y"]
fam = TweedieDistribution(power) if family == "tweedie" else family
g = GeneralizedLinearRegressor(
    family=fam, link="log", alpha=ridge / len(y), l1_ratio=0.0, fit_intercept=False,
    gradient_tol=1e-10, max_iter=1000,
).fit(X, y)
np.save(d + "/out.npy", np.asarray(g.coef_))
"""


def glum_fit(family: str, X: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    """glum's ridge-penalized fit with the log link and ``alpha = R / n``,
    from a process of its own (see above)."""
    with tempfile.TemporaryDirectory() as d:
        np.savez(Path(d) / "in.npz", X=X, y=y)
        subprocess.run(
            [sys.executable, "-I", "-c", _GLUM, d, family, str(ridge), str(P)],
            check=True,
            env={**os.environ, "OMP_NUM_THREADS": "1"},
            capture_output=True,
        )
        return np.load(Path(d) / "out.npy")


def batch(family: str, X: np.ndarray, y: np.ndarray, ridge: float = 1.0):
    """glum's ridge-matched fit, and statsmodels' unpenalized one with its
    standard errors."""
    n = len(y)
    glum_b = glum_fit(family, X, y, ridge)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f = sm.GLM(y, X, family=sm_family(family), freq_weights=np.ones(n)).fit(
            method="IRLS", tol=1e-12, maxiter=200
        )
    return glum_b, np.asarray(f.params), np.asarray(f.bse)


def gaps(family: str, k: int, n: int, seed: int) -> tuple[float, float]:
    X, y = design(family, k, n, seed)
    got = online(family, X, y)
    glum_b, sm_b, bse = batch(family, X, y)
    return float(np.max(np.abs(got - glum_b) / bse)), float(np.max(np.abs(got - sm_b) / bse))


# --- row by row ------------------------------------------------------------


@pytest.mark.parametrize("family", ["poisson", "gamma", "tweedie", "binomial"])
def test_the_recursion_is_the_familys_irls_step_row_by_row(family):
    """With no refit and no cap the recursion is closed-form: ``G += W x x'``,
    ``b += (G + R I)^-1 W (z - eta) x``, its ``W = family.weights(mu)`` and
    ``z = eta + (y - mu) link.deriv(mu)`` from statsmodels' family objects.
    The longhand and the bank share no line of arithmetic; they agree to
    rounding along the whole path (measured 1e-14 relative over 300 rows),
    held at 1e-10."""
    k, n = 3, 300
    if family == "binomial":
        X, mu = design("poisson", k, n, 4)[0], None
        rng = np.random.default_rng(4)
        y = (rng.random(n) < 1 / (1 + np.exp(-(X @ [0.2, 0.8, -0.5])))).astype(float)
    else:
        X, y = design(family, k, n, 4)
    fam = sm_family(family)
    G = np.zeros((k, k))
    b = np.zeros(k)
    path = []
    for i in range(n):
        x = X[i]
        eta = float(np.clip(x @ b, -30.0, 30.0))
        mu = np.atleast_1d(fam.link.inverse(eta))
        W = float(fam.weights(mu)[0])
        dz = float((y[i] - mu[0]) * fam.link.deriv(mu)[0])
        G += W * np.outer(x, x)
        b = b + np.linalg.solve(G + np.eye(k), W * dz * x)
        path.append(b.copy())
    s = rp.spec.glm(
        "m",
        targets=["y"],
        features=["x1", "x2"],
        family=family,
        half_life=float("inf"),
        refit_rows=0,
        step_cap=float("inf"),
        coef_every=0,
        **({"var_power": P} if family == "tweedie" else {}),
    )
    out = rp.ModelBank([s]).fit_predict(frame(X, y))
    got = np.array(out["m"].struct.field("coef").to_list(), dtype=float)
    want = np.array(path)
    assert np.max(np.abs(got - want) / (1.0 + np.abs(want))) < 1e-10


def test_a_refit_of_every_row_is_glums_fit():
    """A buffer as long as the stream is fitted to convergence by the
    refit's penalized IRLS on the family's unit deviance, so the bank's fit
    is glum's ridge-matched fit of the same rows: within 1e-4 standard
    errors (glum stops at a gradient of 1e-10, the refit runs to its
    rounding; measured 1e-6)."""
    X, y = design("tweedie", 3, 2000, 5)
    got = online("tweedie", X, y, refit_rows=2000)
    glum_b, _, bse = batch("tweedie", X, y)
    assert np.max(np.abs(got - glum_b) / bse) < 1e-4


# --- converged, statistical ------------------------------------------------

ESSENTIAL_TOL = {"poisson": 0.25, "gamma": 0.20, "tweedie": 0.25}


@pytest.mark.parametrize("family", ["poisson", "gamma", "tweedie"])
@pytest.mark.parametrize("seed", [0, 1])
def test_five_coefficients_at_10k_rows(family, seed):
    """The study's k = 5 tolerance at 10,000 rows, against glum (ridge
    matched) and statsmodels (unpenalized), in batch standard errors."""
    to_glum, to_sm = gaps(family, 5, 10_000, seed)
    tol = ESSENTIAL_TOL[family]
    assert to_glum <= tol and to_sm <= tol, (to_glum, to_sm)


GRID_TOL = {
    ("poisson", 10_000): 0.75,
    ("poisson", 100_000): 0.30,
    ("gamma", 10_000): 0.50,
    ("gamma", 100_000): 0.25,
    ("tweedie", 10_000): 0.95,
    ("tweedie", 100_000): 0.50,
}


@pytest.mark.extended(reason="20 coefficients, 100,000 rows, two batch fits a case")
@pytest.mark.parametrize(("family", "n"), sorted(GRID_TOL))
@pytest.mark.parametrize("seed", range(5))
def test_twenty_coefficients(family, n, seed):
    to_glum, to_sm = gaps(family, 20, n, seed)
    tol = GRID_TOL[(family, n)]
    assert to_glum <= tol and to_sm <= tol, (to_glum, to_sm)


@pytest.mark.parametrize("family", ["poisson", "gamma"])
def test_under_decay_the_target_is_the_decayed_likelihood(family):
    """Under a half-life the ridge ages with the rows, so the fit tends to
    statsmodels' ``GLM`` with the decayed weights as ``freq_weights``,
    unpenalized. At a half-life of 1,000 rows over 10,000 rows, 5
    coefficients: measured 0.05 SE (Poisson) and 0.07 (Gamma), held at
    0.15."""
    k, n, h = 5, 10_000, 1000.0
    X, y = design(family, k, n, 9)
    got = online(family, X, y, half_life=h)
    v = 0.5 ** ((n - 1 - np.arange(n)) / h)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f = sm.GLM(y, X, family=sm_family(family), freq_weights=v).fit(tol=1e-12, maxiter=200)
    gap = float(np.max(np.abs(got - np.asarray(f.params)) / np.asarray(f.bse)))
    assert gap <= 0.15, gap


# --- the Tweedie family at other powers and at its canonical link ----------

U = 2.0**-53


def tweedie_design(power: float, link: str, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """An intercept and two standard normal features; the mean through
    ``link`` (``"log"``, or ``"power"``: ``η = μ^(1 − p)``, kept above 0);
    a Tweedie amount of power ``p`` and dispersion ``PHI``: the compound
    Poisson-Gamma below 2, with its exact zeros, and above 2 a Gamma of
    the same mean and variance ``PHI μ^p``, which is above 0."""
    rng = np.random.default_rng([500, seed, int(power * 10), link == "power"])
    X = np.column_stack([np.ones(n), rng.standard_normal((n, 2))])
    if link == "log":
        mu = np.exp(X @ np.array([0.5, 0.3, -0.2]))
    else:
        eta = X @ np.array([0.7, 0.08, -0.06])
        assert eta.min() > 0.2
        mu = eta ** (1.0 / (1.0 - power))
    if power < 2:
        lam = mu ** (2 - power) / (PHI * (2 - power))
        N = rng.poisson(lam)
        y = np.zeros(n)
        pos = N > 0
        shape = (2 - power) / (power - 1)
        y[pos] = rng.gamma(N[pos] * shape, PHI * (power - 1) * mu[pos] ** (power - 1))
    else:
        y = rng.gamma(1.0 / (PHI * mu ** (power - 2)), PHI * mu ** (power - 1))
    return X, y


def tweedie_family(power: float, link: str) -> sm.families.Family:
    lk = sm.families.links.Log() if link == "log" else sm.families.links.Power(1.0 - power)
    return sm.families.Tweedie(lk, var_power=power)


@pytest.mark.parametrize("power", [1.2, 1.8, 2.5, 3.0])
def test_the_tweedie_recursion_at_each_power_is_statsmodels_irls_step(power):
    """The row-by-row recursion of
    ``test_the_recursion_is_the_familys_irls_step_row_by_row``, at Tweedie
    powers either side of 2 (zeros among the amounts below it, none above),
    its Fisher weight and working response from statsmodels' ``Tweedie``
    family at that power.

    Without the refit or a step cap the recursion can run away from its
    first rows (``glm.rs``, *The start*), and a runaway's path is chaotic,
    so the stream is one on which no coefficient passes 10. Bound: each
    side takes ``μ^p`` and ``e^η`` from its own library, three ulps a call,
    and each row's solve amplifies a relative error by at most the largest
    condition of ``G + I`` along the path, ``κ``, read from the longhand:
    over ``n`` rows, ``64 n κ u`` relative. Measured 2026-11-24 (x86-64,
    Linux): ``κ`` at most 21, the gap at most 1.4e-4 of the bound.
    """
    k, n = 3, 300
    X, y = tweedie_design(power, "log", n, 7)
    assert (power < 2) == bool(np.any(y == 0))
    fam = tweedie_family(power, "log")
    G = np.zeros((k, k))
    b = np.zeros(k)
    path = []
    kappa = 1.0
    for i in range(n):
        x = X[i]
        eta = float(np.clip(x @ b, -30.0, 30.0))
        mu = np.atleast_1d(fam.link.inverse(eta))
        W = float(fam.weights(mu)[0])
        dz = float((y[i] - mu[0]) * fam.link.deriv(mu)[0])
        G += W * np.outer(x, x)
        kappa = max(kappa, float(np.linalg.cond(G + np.eye(k))))
        b = b + np.linalg.solve(G + np.eye(k), W * dz * x)
        path.append(b.copy())
    assert np.abs(path).max() < 10.0
    s = rp.spec.glm(
        "m", targets=["y"], features=["x1", "x2"], family="tweedie", var_power=power,
        half_life=float("inf"), refit_rows=0, step_cap=float("inf"), coef_every=0,
    )  # fmt: skip
    out = rp.ModelBank([s]).fit_predict(frame(X, y))
    got = np.array(out["m"].struct.field("coef").to_list(), dtype=float)
    want = np.array(path)
    gap = np.max(np.abs(got - want) / (1.0 + np.abs(want)))
    assert gap <= 64 * n * kappa * U, (gap, kappa)


@pytest.mark.parametrize("power", [1.5, 3.0])
@pytest.mark.parametrize("ridge", [1.0, 50.0])
def test_a_tweedie_refit_at_the_power_link_solves_statsmodels_score_equation(power, ridge):
    """At the Tweedie family's canonical link, ``η = μ^(1 − p)``, a refit of
    every row lands where the penalized likelihood is flat: statsmodels'
    ``GLM.score`` of the same rows at the fit, at unit scale (``Σ (y − μ) x
    / (g'(μ) μ^p)``), equals ``R β``.

    Bound: the refit stops once its full Fisher step is under ``1e-11 (1 +
    max|β|)`` (``glm.rs``, ``refit_fit``), so the gradient left is at most
    that step times the system's largest eigenvalue, ``‖G + R I‖``, ``G``
    statsmodels' own Fisher information at the fit; doubled, plus the
    score's own rounding, ``8 n u Σ|terms|`` (each term a few ``powf`` and
    products, summed over ``n`` rows). Measured 2026-11-24: at most 1e-5 of
    it. The penalty is held: ``R β`` is past a thousand times the bound, so
    a fit of the unpenalized likelihood, or of another ``R``, fails."""
    n = 600
    X, y = tweedie_design(power, "power", n, 7)
    s = rp.spec.glm(
        "m", targets=["y"], features=["x1", "x2"], family="tweedie", var_power=power,
        link="power", ridge=ridge, half_life=float("inf"), refit_rows=n,
    )  # fmt: skip
    bank = rp.ModelBank([s])
    bank.fit(frame(X, y))
    b = np.array(bank.coef()["coef"].to_list(), dtype=float)
    fam = tweedie_family(power, "power")
    model = sm.GLM(y, X, family=fam)
    score = model.score(b, scale=1.0)
    mu = fam.link.inverse(X @ b)
    terms = np.abs((y - mu) / (fam.link.deriv(mu) * fam.variance(mu)))[:, None] * np.abs(X)
    info = -model.hessian(b, scale=1.0, observed=False)
    top = float(np.linalg.eigvalsh(info + ridge * np.eye(3)).max())
    bound = 2 * top * 1e-11 * (1 + np.abs(b).max()) + 8 * n * U * terms.sum(axis=0).max()
    gap = float(np.abs(score - ridge * b).max())
    assert gap <= bound, (gap, bound)
    assert float(np.abs(ridge * b).max()) > 1e3 * bound
