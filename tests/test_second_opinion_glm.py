"""`glm` against statsmodels and glum (docs/PLAN.md task 500; the numbers
are docs/records/STUDY-500-GLM.md's, on the branch `vm/study-500`, not merged).

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

import ledger
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


# Two batch fits of up to 100,000 rows a seed: the oracle tier
# (`oracle_grid`), which CI shards and the ledger holds (`oracles.toml`), not
# the full tier every workflow runs (PLAN 578j, CI-1).
ledger.guard(
    "glum_twenty",
    lambda: [online(family, *design(family, 20, 2_000, 0)) for family, _ in sorted(GRID_TOL)],
    oracles=["glum", "statsmodels"],
)


@pytest.mark.oracle_grid(reason="20 coefficients, 100,000 rows, two batch fits a seed")
@pytest.mark.parametrize(("family", "n"), sorted(GRID_TOL))
def test_twenty_coefficients(family, n):
    got = np.array([gaps(family, 20, n, seed) for seed in range(5)])
    ledger.record("glum_twenty", f"{family}_{n}_to_glum", got[:, 0])
    ledger.record("glum_twenty", f"{family}_{n}_to_statsmodels", got[:, 1])
    tol = GRID_TOL[(family, n)]
    assert got.max() <= tol, got


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


# --- the binomial, the canonical links, a second power, weights (TEST-3) -----
#
# The statistical tests above read the log link of three families at unit
# weights: the binomial, Gamma's inverse link, Tweedie's power link, a power
# other than 1.5 and rows of uneven weight had no oracle (PLAN 578j, TEST-3).


def linked(case: str, n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict, object]:
    """A design whose truth is linear on ``case``'s link: the rows, the
    target, uneven weights (0.5 to 2), the spec's family keywords and
    statsmodels' family."""
    rng = np.random.default_rng([578, 3, seed, len(case)])
    X = np.column_stack([np.ones(n), rng.standard_normal((n, 2))])
    w = rng.uniform(0.5, 2.0, n)
    links = sm.families.links
    if case == "binomial":
        mu = 1.0 / (1.0 + np.exp(-(X @ [0.2, 0.8, -0.5])))
        return X, (rng.random(n) < mu).astype(float), w, {"family": "binomial"}, sm_family(case)
    if case == "gamma inverse":
        mu = 1.0 / np.clip(X @ [1.0, 0.15, -0.1], 0.2, None)
        y = rng.gamma(2.0, mu / 2.0)
        return (
            X,
            y,
            w,
            {"family": "gamma", "link": "inverse"},
            sm.families.Gamma(links.InversePower()),
        )
    p = 1.5 if case == "tweedie power" else 1.2
    if case == "tweedie power":
        mu = np.clip(X @ [0.8, 0.08, -0.08], 0.2, None) ** (1.0 / (1.0 - p))
        family = sm.families.Tweedie(links.Power(1.0 - p), var_power=p)
        kw = {"family": "tweedie", "link": "power", "var_power": p}
    else:
        mu = np.exp(X @ [0.5, 0.3, -0.2])
        family = sm.families.Tweedie(links.Log(), var_power=p)
        kw = {"family": "tweedie", "var_power": p}
    lam = mu ** (2 - p) / (PHI * (2 - p))
    count = rng.poisson(lam)
    y = np.zeros(n)
    pos = count > 0
    y[pos] = rng.gamma(count[pos] * (2 - p) / (p - 1), PHI * (p - 1) * mu[pos] ** (p - 1))
    return X, y, w, kw, family


@pytest.mark.parametrize("case", ["binomial", "gamma inverse", "tweedie power", "tweedie 1.2"])
@pytest.mark.parametrize("weighted", [False, True])
def test_each_family_and_link_is_statsmodels_fit(case, weighted):
    """At 10,000 rows, no decay and the defaults (a refit of the first
    rows, then the one pass), the coefficients are statsmodels' maximum
    likelihood fit of the same rows, weighted as the bank weighs them
    (``freq_weights``), within 0.25 of its standard errors."""
    n = 10_000
    X, y, w, kw, family = linked(case, n, 0)
    weights = w if weighted else np.ones(n)
    s = rp.spec.glm(
        "m",
        targets=["y"],
        features=["x1", "x2"],
        half_life=float("inf"),
        **kw,
        **({"weight": "w"} if weighted else {}),
    )
    bank = rp.ModelBank([s])
    bank.fit(frame(X, y).with_columns(w=w))
    got = np.array([r["coef"] for r in bank.coef().iter_rows(named=True)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f = sm.GLM(y, X, family=family, freq_weights=weights).fit(
            method="IRLS", tol=1e-12, maxiter=200
        )
    gap = float(np.max(np.abs(got - np.asarray(f.params)) / np.asarray(f.bse)))
    print(f"{case}, weighted {weighted}: {gap:.3f} SE")
    assert gap <= 0.25, (got, np.asarray(f.params), gap)
