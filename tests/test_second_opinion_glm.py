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
from scipy.optimize import brentq

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
    Under the log link the longhand takes the target guard too, its root
    from scipy's ``brentq`` (``longhand`` below). The longhand and the bank
    share no line of arithmetic; they agree to rounding along the whole
    path (measured 1e-14 relative over 300 rows), held at 1e-10."""
    k, n = 3, 300
    if family == "binomial":
        X, mu = design("poisson", k, n, 4)[0], None
        rng = np.random.default_rng(4)
        y = (rng.random(n) < 1 / (1 + np.exp(-(X @ [0.2, 0.8, -0.5])))).astype(float)
    else:
        X, y = design(family, k, n, 4)
    fam = sm_family(family)
    if family == "binomial":
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
    else:
        # The log link's target guard (below) scales a row's step that
        # would pass its own ``ln y``.
        path = list(longhand(fam, X, y, guard=True)["path"])
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


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("half_life", [float("inf"), 100.0])
def test_a_binomial_refit_of_every_row_is_statsmodels_fit(weighted, half_life):
    """A binomial refit over every row fits the decayed weighted logit
    likelihood by penalized IRLS to rounding (PLAN 578j, TEST-3):

    - At a negligible ridge (``1e-10``), the bank's coefficients match
      statsmodels' ``GLM(..., family=Binomial(), freq_weights=v).fit()``
      directly within ``1e-4`` standard errors (matching
      ``test_a_refit_of_every_row_is_glums_fit``'s tolerance style), with
      ``v_i = w_i * 2^(-(n - 1 - i) / half_life)`` under decay and ``w_i``
      without.
    - At non-trivial ridges (``1.0`` and ``25.0``), the refit lands where
      statsmodels' penalized score vanishes: ``GLM.score(b, scale=1.0) ==
      R * b`` within the refit's step tolerance and score rounding bound
      (matching
      ``test_a_tweedie_refit_at_the_power_link_solves_statsmodels_score_equation``).
    """
    n = 600
    X, y, w, kw, fam = linked("binomial", n, 3)
    base_w = w if weighted else np.ones(n)
    decay = 1.0 if np.isinf(half_life) else 0.5 ** ((n - 1 - np.arange(n)) / half_life)
    v = base_w * decay

    def fit_bank(ridge: float) -> np.ndarray:
        s = rp.spec.glm(
            "m",
            targets=["y"],
            features=["x1", "x2"],
            link="logit",
            half_life=half_life,
            ridge=ridge,
            refit_rows=n,
            **kw,
            **({"weight": "w"} if weighted else {}),
        )
        bank = rp.ModelBank([s])
        bank.fit(frame(X, y).with_columns(w=w))
        return np.array([r["coef"] for r in bank.coef().iter_rows(named=True)], dtype=float)

    model = sm.GLM(y, X, family=fam, freq_weights=v)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f = model.fit(method="IRLS", tol=1e-12, maxiter=200)
    b_unpen = fit_bank(1e-10)
    se_gap = float(np.max(np.abs(b_unpen - np.asarray(f.params)) / np.asarray(f.bse)))
    assert se_gap < 1e-4, (weighted, half_life, se_gap, b_unpen, np.asarray(f.params))

    for ridge in (1.0, 25.0):
        b = fit_bank(ridge)
        score_val = model.score(b, scale=1.0)
        mu = fam.link.inverse(X @ b)
        terms = np.abs(v * (y - mu))[:, None] * np.abs(X)
        info = -model.hessian(b, scale=1.0, observed=False)
        top = float(np.linalg.eigvalsh(info + ridge * np.eye(3)).max())
        bound = 2 * top * 1e-11 * (1 + np.abs(b).max()) + 8 * n * U * terms.sum(axis=0).max()
        gap = float(np.abs(score_val - ridge * b).max())
        assert gap <= bound, (weighted, half_life, ridge, gap, bound)
        assert float(np.abs(ridge * b).max()) > 1e3 * bound


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


# --- the log link's target guard ----------------------------------------------
#
# Under the log link an upward Fisher step moves the row's own η by
# δ = q' w μ^(1−p) (y − μ), which grows without bound as μ falls, and with
# no refit and no cap the coefficients run away. So an upward step that
# would carry η past ln y, the minimum of the row's own deviance, is
# scaled to land η on the root u in (η, ln y] of the row's proximal problem,
# h(u) = u − η + w q f'(u), f the row's half unit deviance at μ = e^u and
# q = x̃'(G + R I)⁻¹x̃ before the row (glm.rs, *The target guard*). A row it
# does not fire on is the Fisher step's.

#: The root's absolute and relative tolerance in ``brentq``: the least
#: relative one it takes, four times the doubles' epsilon, and an absolute
#: one far below the doubles' spacing at the roots here.
XTOL, RTOL = 1e-300, 8 * U


def log_family(family: str, power: float) -> sm.families.Family:
    log = sm.families.links.Log()
    if family == "poisson":
        return sm.families.Poisson(log)
    if family == "gamma":
        return sm.families.Gamma(log)
    return sm.families.Tweedie(log, var_power=power)


def log_design(family: str, power: float, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """C's stream under the log link, each family from its own generator:
    the Tweedie's ``tweedie_design``; for the Poisson and the Gamma the
    same features and mean, Poisson counts or Gamma amounts of dispersion
    ``PHI`` (shape ``1/PHI``)."""
    if family == "tweedie":
        return tweedie_design(power, "log", n, seed)
    rng = np.random.default_rng([500, seed, {"poisson": 10, "gamma": 20}[family], False])
    X = np.column_stack([np.ones(n), rng.standard_normal((n, 2))])
    mu = np.exp(X @ np.array([0.5, 0.3, -0.2]))
    if family == "poisson":
        return X, rng.poisson(mu).astype(float)
    return X, rng.gamma(1.0 / PHI, PHI * mu)


def score(fam: sm.families.Family, u: float, y: float) -> float:
    """The row's score at ``η = u`` from statsmodels' family objects, ``W
    (z − η) = W (y − μ) g'(μ)``: ``−f'(u)``, ``f`` the half unit deviance."""
    mu = np.atleast_1d(fam.link.inverse(u))
    return float(fam.weights(mu)[0] * (y - mu[0]) * fam.link.deriv(mu)[0])


def longhand(
    fam: sm.families.Family,
    X: np.ndarray,
    y: np.ndarray,
    *,
    guard: bool,
    start: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict:
    """The recursion with no cap, row by row: ``G += W x x'``, ``b += (G +
    I)⁻¹ W (z − η) x``, ``W`` and ``z`` from statsmodels' family objects,
    from every coefficient 0 or from ``start`` (``b``, ``G``). With
    ``guard``, a step whose ``η`` would pass ``ln y`` is scaled to land on
    the root of ``h(u) = u − η − q score(u)`` in ``[η, ln y]`` by scipy's
    ``brentq``, ``q = x'(G + I)⁻¹x`` before the row by ``np.linalg.solve``.
    Returns the path, the largest condition of ``G + I``, the rows fired on,
    the rows above the guard (``η + δ > ln y``), and the root's tolerance
    carried to the coefficients, ``Σ (XTOL + RTOL |u|) |d| / δ``."""
    k = X.shape[1]
    b, G = (np.zeros(k), np.zeros((k, k))) if start is None else (start[0].copy(), start[1].copy())
    eye = np.eye(k)
    path, kappa, fired, above, slack = [], 1.0, 0, 0, 0.0
    for i in range(len(y)):
        x = X[i]
        eta = float(np.clip(x @ b, -30.0, 30.0))
        mu = np.atleast_1d(fam.link.inverse(eta))
        W = float(fam.weights(mu)[0])
        dz = float((y[i] - mu[0]) * fam.link.deriv(mu)[0])
        before = G + eye
        G = G + W * np.outer(x, x)
        kappa = max(kappa, float(np.linalg.cond(G + eye)))
        d = np.linalg.solve(G + eye, W * dz * x)
        delta = float(x @ d)
        if y[i] > 0 and delta > 0 and eta + delta > np.log(y[i]):
            above += 1
            if guard:
                q = float(x @ np.linalg.solve(before, x))
                u = brentq(
                    lambda t, eta=eta, q=q, yi=y[i]: t - eta - q * score(fam, t, yi),
                    eta,
                    float(np.log(y[i])),
                    xtol=XTOL,
                    rtol=RTOL,
                )
                slack += (XTOL + RTOL * abs(u)) * float(np.abs(d).max()) / delta
                d = d * ((u - eta) / delta)
                fired += 1
        b = b + d
        path.append(b.copy())
    return {"path": np.array(path), "kappa": kappa, "fired": fired, "above": above, "slack": slack}


def rheon_path(family: str, power: float, X: np.ndarray, y: np.ndarray, **kw: object) -> np.ndarray:
    """The bank's coefficients after every row, under the log link."""
    s = rp.spec.glm(
        "m", targets=["y"], features=["x1", "x2"], family=family, half_life=float("inf"),
        coef_every=0, **({"var_power": power} if family == "tweedie" else {}), **kw,
    )  # fmt: skip
    out = rp.ModelBank([s]).fit_predict(frame(X, y))
    return np.array(out["m"].struct.field("coef").to_list(), dtype=float)


#: Per power, seeds on which the longhand's step never passes ``ln y``
#: (asserted, not assumed), and seeds on which it does on at least one row.
BELOW_GUARD = {1.2: [0, 7, 10], 1.8: [3, 4, 7], 2.5: [2, 6, 15], 3.0: [4, 44, 50]}
ABOVE_GUARD = {1.2: [1, 2, 3], 1.8: [0, 1, 2], 2.5: [0, 1, 7], 3.0: [0, 1, 7]}


@pytest.mark.parametrize("power", sorted(BELOW_GUARD))
def test_below_the_guard_the_tweedie_recursion_is_statsmodels_irls_step(power):
    """The row-by-row recursion of
    ``test_the_recursion_is_the_familys_irls_step_row_by_row``, at Tweedie
    powers either side of 2 (zeros among the amounts below it, none above),
    its Fisher weight and working response from statsmodels' ``Tweedie``
    family at that power, with no refit and no cap, on streams where no
    row's step passes its own ``ln y``, so the target guard never fires
    (the longhand checks this on every row).

    Bound: each side takes ``μ^p`` and ``e^η`` from its own library, three
    ulps a call, and each row's solve amplifies a relative error by at most
    the largest condition of ``G + I`` along the path, ``κ``, read from the
    longhand: over ``n`` rows, ``64 n κ u`` relative. Measured 2026-11-24
    (x86-64, Linux): ``κ`` at most 21, the gap at most 1.4e-4 of the bound.
    """
    n = 300
    fam = tweedie_family(power, "log")
    for seed in BELOW_GUARD[power]:
        X, y = tweedie_design(power, "log", n, seed)
        assert (power < 2) == bool(np.any(y == 0))
        lh = longhand(fam, X, y, guard=True)
        assert lh["above"] == 0, (seed, lh["above"])
        want = lh["path"]
        assert np.abs(want).max() < 10.0
        got = rheon_path("tweedie", power, X, y, refit_rows=0, step_cap=float("inf"))
        gap = np.max(np.abs(got - want) / (1.0 + np.abs(want)))
        assert gap <= 64 * n * lh["kappa"] * U, (seed, gap, lh["kappa"])


@pytest.mark.parametrize("power", sorted(ABOVE_GUARD))
def test_above_the_guard_the_tweedie_recursion_lands_on_the_rows_root(power):
    """The same recursion on streams where the guard fires (at least one
    row's step passes its own ``ln y``): on each such row the longhand
    scales its step to land ``η`` on the root of the row's proximal
    problem, ``h(u) = u − η + q f'(u)`` on ``[η, ln y]``, found by scipy's
    ``brentq``, ``f'`` from statsmodels' family (``−W (y − μ) g'(μ)``) and
    ``q`` by numpy's solve: a third-party root and solve, the definition
    alone written here.

    Bound: the bound below the guard, plus the root's tolerance carried to
    the coefficients, ``(xtol + rtol |u|) |d|/δ`` a firing row, times
    ``κ`` for the rows after it. The model's ``q`` comes by
    Sherman-Morrison from its step rather than a second solve, a relative
    error of ``κ u`` a row, inside the first term. Measured 2026-11-24
    (macOS arm64): 1 to 6 rows fired a stream, ``κ`` at most 44, the gap
    at most 3.9e-5 of the bound."""
    n = 300
    fam = tweedie_family(power, "log")
    for seed in ABOVE_GUARD[power]:
        X, y = tweedie_design(power, "log", n, seed)
        lh = longhand(fam, X, y, guard=True)
        assert lh["fired"] >= 1, seed
        want = lh["path"]
        assert np.abs(want).max() < 10.0
        got = rheon_path("tweedie", power, X, y, refit_rows=0, step_cap=float("inf"))
        gap = np.max(np.abs(got - want) / (1.0 + np.abs(want)))
        bound = 64 * n * lh["kappa"] * U + lh["kappa"] * lh["slack"]
        assert gap <= bound, (seed, gap, bound, lh["kappa"])


# --- the share over 100 seeds, and the default spec's outlier -----------------

#: The log-link families of the share, ``(family, power)``, each a case of
#: the ledger.
SHARE_FAMILIES = [
    ("poisson", 1.0),
    ("gamma", 2.0),
    ("tweedie", 1.2),
    ("tweedie", 1.5),
    ("tweedie", 1.8),
    ("tweedie", 2.5),
    ("tweedie", 3.0),
]


def case_name(family: str, power: float) -> str:
    return family if family != "tweedie" else f"tweedie_{power:.1f}".replace(".", "_")


def sm_fit(
    family: str, power: float, X: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """statsmodels' ``GLM(...).fit()`` of the rows and its standard errors.
    Above ``p = 2`` its IRLS from its own start, and from the ridge-1 fit,
    fails on some of these streams (a weight overflows on its way), so it
    starts at a converged fit: the bank's refit of every row at a ridge of
    1e-12, from which it converges on all 100 seeds a power to its own
    tolerance, 1e-12."""
    fam = log_family(family, power)
    start = None
    if family == "tweedie" and power > 2:
        start = rheon_path(family, power, X, y, refit_rows=len(y), ridge=1e-12)[-1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f = sm.GLM(y, X, family=fam).fit(start_params=start, tol=1e-12, maxiter=200)
    return np.asarray(f.params), np.asarray(f.bse)


def share_seed(family: str, power: float, seed: int) -> tuple[bool, bool, float]:
    """C's stream (300 rows) with no refit and no cap: whether the unguarded
    longhand's path passes ``|coef|`` 10, whether the bank's does, and the
    bank's final fit's gap to statsmodels', in its standard errors."""
    X, y = log_design(family, power, 300, seed)
    lh = longhand(log_family(family, power), X, y, guard=False)
    off = bool(not np.all(np.abs(lh["path"]) < 10.0))
    got = rheon_path(family, power, X, y, refit_rows=0, step_cap=float("inf"))
    on = bool(not np.all(np.abs(got) < 10.0))
    want, se = sm_fit(family, power, X, y)
    return off, on, float(np.max(np.abs(got[-1] - want) / se))


#: Of 100 seeds a family, those on which the unguarded longhand's path
#: passes ``|coef|`` 10, measured 2026-11-24: the counts the bank's guard
#: takes to 0, asserted so that the test cannot pass on streams with
#: nothing to stop.
SHARE_RUNAWAYS = {
    "poisson": 0,
    "gamma": 17,
    "tweedie_1_2": 0,
    "tweedie_1_5": 1,
    "tweedie_1_8": 7,
    "tweedie_2_5": 35,
    "tweedie_3_0": 64,
}

#: The bank's final fit against statsmodels', in statsmodels' standard
#: errors: the worst of 100 seeds at half as much again as measured,
#: rounded up to 0.05 (2026-11-24, macOS arm64, ``share_seed`` over seeds
#: 0 to 99; the measured worst and median in the comment), the ledger
#: holding the measured values. With no refit and no cap the first rows'
#: curvature stays in ``G``, so these are a floor's distances, far from the
#: refit's (0.2 to 0.3 at 100,000 rows); at ``p = 3`` the worst seeds are
#: those whose first rows the guard scaled most.
SHARE_SE = {
    "poisson": 8.3,  # 5.53 (median 0.25)
    "gamma": 5.05,  # 3.35 (0.55)
    "tweedie_1_2": 2.05,  # 1.36 (0.38)
    "tweedie_1_5": 2.45,  # 1.61 (0.45)
    "tweedie_1_8": 2.35,  # 1.54 (0.47)
    "tweedie_2_5": 7.35,  # 4.88 (0.81)
    "tweedie_3_0": 21.55,  # 14.34 (1.10)
}

ledger.guard(
    "log_link_guard",
    lambda: [
        rheon_path(f, p, *log_design(f, p, 300, s), refit_rows=0, step_cap=float("inf"))[-1]
        for f, p in SHARE_FAMILIES
        for s in (0, 1)
    ],
    oracles=["statsmodels"],
)


def test_one_seed_of_the_share():
    """The share's cheap representative: seed 0 of each family, the bank's
    path under 10 and its fit within the grid's bound."""
    for family, power in SHARE_FAMILIES:
        _, on, gap = share_seed(family, power, 0)
        name = case_name(family, power)
        assert not on, name
        assert gap <= SHARE_SE[name], (name, gap)


@pytest.mark.oracle_grid(reason="100 seeds a family, seven families, a statsmodels fit each")
@pytest.mark.parametrize(("family", "power"), SHARE_FAMILIES)
def test_with_no_refit_and_no_cap_no_seed_runs_away(family, power):
    """**With no refit and no cap no seed runs away**, of 100 a family,
    where the unguarded recursion does on ``SHARE_RUNAWAYS`` of them; and
    the bank's final fit is within ``SHARE_SE`` standard errors of
    statsmodels' ``GLM(...).fit()``."""
    name = case_name(family, power)
    off, on, gaps = zip(*(share_seed(family, power, s) for s in range(100)), strict=True)
    ledger.record("log_link_guard", name, gaps)
    ledger.record("log_link_guard", f"{name}_runaways", [int(v) for v in on], exact=True)
    assert sum(off) == SHARE_RUNAWAYS[name], (name, sum(off))
    assert sum(on) == 0, (name, sum(on))
    assert max(gaps) <= SHARE_SE[name], (name, np.median(gaps), max(gaps))


#: The default spec's outlier stream: its families, and of 100 seeds those
#: on which the recursion after the default refit, unguarded, passes
#: ``|coef|`` 10, measured 2026-11-24 (rheon before the guard gave the same
#: counts on the same streams).
DEFAULT_FAMILIES = [
    ("poisson", 1.0),
    ("gamma", 2.0),
    ("tweedie", 1.5),
    ("tweedie", 2.5),
    ("tweedie", 3.0),
]
DEFAULT_RUNAWAYS = {
    "poisson": 14,
    "gamma": 41,
    "tweedie_1_5": 17,
    "tweedie_2_5": 38,
    "tweedie_3_0": 48,
}


def default_seed(family: str, power: float, seed: int) -> tuple[bool, bool, int]:
    """C's stream with one target 1000 times its draw (at least 1) at row
    250, under the default spec (a refit of 200 rows, no cap after it):
    whether the bank's path passes ``|coef|`` 10 after the refit, whether
    the unguarded longhand does from the bank's refit (its coefficients at
    row 199, ``G`` rebuilt at them from statsmodels' weights), and the rows
    above the guard there."""
    X, y = log_design(family, power, 300, seed)
    y[250] = max(y[250], 1.0) * 1000.0
    got = rheon_path(family, power, X, y)
    refit = 200
    b = got[refit - 1]
    fam = log_family(family, power)
    mu = fam.link.inverse(X[:refit] @ b)
    G = (X[:refit] * fam.weights(mu)[:, None]).T @ X[:refit]
    lh = longhand(fam, X[refit:], y[refit:], guard=False, start=(b, G))
    return (
        bool(not np.all(np.abs(got[refit:]) < 10.0)),
        bool(not np.all(np.abs(lh["path"]) < 10.0)),
        lh["above"],
    )


#: Per family, the first seed on which the unguarded recursion runs away.
DEFAULT_ESSENTIAL_SEED = {
    "poisson": 3,
    "gamma": 0,
    "tweedie_1_5": 11,
    "tweedie_2_5": 0,
    "tweedie_3_0": 0,
}


@pytest.mark.parametrize(("family", "power"), DEFAULT_FAMILIES)
def test_the_default_spec_takes_a_far_target_after_its_refit(family, power):
    """The default spec's hazard: after its refit the cap is off, and one
    target 1000 times its draw sends the unguarded recursion past
    ``|coef|`` 10, on the seed here; the bank's path stays under 10. One
    seed a family here; 100 in the extended run."""
    on, off, above = default_seed(family, power, DEFAULT_ESSENTIAL_SEED[case_name(family, power)])
    assert off and above >= 1 and not on, (off, above, on)


@pytest.mark.extended(reason="100 seeds a family, five families, a bank run and a longhand each")
@pytest.mark.parametrize(("family", "power"), DEFAULT_FAMILIES)
def test_the_default_spec_takes_a_far_target_over_100_seeds(family, power):
    """Over 100 seeds no bank path passes ``|coef|`` 10, where the unguarded
    recursion from the same refit does on ``DEFAULT_RUNAWAYS`` of them."""
    on, off, _ = zip(*(default_seed(family, power, s) for s in range(100)), strict=True)
    assert sum(off) == DEFAULT_RUNAWAYS[case_name(family, power)], sum(off)
    assert sum(on) == 0, sum(on)
