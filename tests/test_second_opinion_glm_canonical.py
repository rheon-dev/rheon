"""`glm` under the canonical links whose mean needs ``eta > 0`` -- the Gamma's
inverse link and the Tweedie's power link -- against statsmodels' ``GLM``
on adversarial streams (docs/PLAN.md task 581).

Under these links each learned row after the refit is the exact proximal
step: the row's own penalized deviance minimised, its ``eta`` moved to the
root between the fit's and the target's (the Rust module docs of
``glm.rs``). The bank's coefficients after the stream are held to
statsmodels' unpenalized fit of the same rows, at their weights (decayed
under a half-life), in units of statsmodels' standard errors (its ``bse``,
Pearson scale), the worst coefficient a seed.

**The oracle needs a guard.** statsmodels' IRLS under ``InversePower`` and
``Power`` does not hold ``eta > 0``: from its default start it can settle on
a point with negative means, a spurious answer. Its fit is taken where every
row's ``eta`` is above 0 at its answer and the deviance's gradient there is
0 to 1e-6 of the weight; otherwise the in-domain minimum is found by a
damped Newton on the deviance (convex in ``beta`` under a canonical link,
the steps halved until they stay in the domain and do not raise it), and
statsmodels' IRLS is restarted there: it must stay within 1e-6 of its own
standard errors, so the answer and the standard errors are still
statsmodels'.

**The streams** are a simulation study's: a well-specified inverse-link
Gamma of shape 2 at three coefficients and 3,000 rows (``eta = 0.5 + 0.15
x1 - 0.1 x2``, ``x`` uniform on ``[-1, 1]``), and the same with one thing
made hard: a first target tiny or huge, a target 1000 times its mean early
or late, ``eta`` running to the domain's edge, heavy noise, 20 features at
10,000 rows, rows sorted by a feature, a rare category whose ``eta`` sits
near 0, a log-linear truth, weights with zeros, the target in other units,
the Tweedie at ``p = 1.5`` with zeros and at ``p = 3`` (inverse Gaussian
draws), and a half-life of 300 rows. The default refit of 200 rows (210 at
20 features) runs first. The essentials hold three cells at one seed; the
grid holds every case over 20 seeds, recorded in the oracle ledger.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable

import numpy as np
import polars as pl
import pytest
import statsmodels.api as sm

import ledger
from rheon import polars as rp

TIER = "mixed"

Case = tuple[float, np.ndarray, np.ndarray, np.ndarray | None, float]


def sm_family(p: float) -> sm.families.Family:
    if p == 2.0:
        return sm.families.Gamma(sm.families.links.InversePower())
    return sm.families.Tweedie(sm.families.links.Power(1.0 - p), var_power=p)


def half_deviance(
    p: float, X: np.ndarray, y: np.ndarray, v: np.ndarray, b: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    """The weighted half deviance up to terms free of ``b``, its gradient and
    its Hessian, under the canonical link ``eta = mu^(1 - p)`` (``1/mu`` for
    the Gamma): ``y eta/(p - 1) + mu^(2 - p)/(2 - p)``, ``y eta - ln eta``
    at ``p = 2``."""
    eta = X @ b
    mu = eta ** (-1.0 / (p - 1.0))
    f = y * eta - np.log(eta) if p == 2.0 else y * eta / (p - 1.0) + mu ** (2.0 - p) / (2.0 - p)
    g = X.T @ (v * (y - mu) / (p - 1.0))
    w = mu**p / (p - 1.0) ** 2
    return float(np.sum(v * f)), g, (X * (v * w)[:, None]).T @ X


def _statsmodels(
    p: float, X: np.ndarray, y: np.ndarray, v: np.ndarray, start: np.ndarray, maxiter: int
) -> tuple[np.ndarray, np.ndarray] | None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            r = sm.GLM(y, X, family=sm_family(p), freq_weights=v).fit(
                start_params=start, tol=1e-12, maxiter=maxiter
            )
        except (ValueError, np.linalg.LinAlgError):
            return None
    b, se = np.asarray(r.params), np.asarray(r.bse)
    if not (np.all(np.isfinite(b)) and (X @ b).min() > 0 and np.all(np.isfinite(se))):
        return None
    return b, se


def batch(p: float, X: np.ndarray, y: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """statsmodels' fit and standard errors, held to the link's domain (the
    module docs)."""
    keep = v > 0
    X, y, v = X[keep], y[keep], v[keep]
    start = np.zeros(X.shape[1])
    start[0] = np.average(y, weights=v) ** (1.0 - p)
    r = _statsmodels(p, X, y, v, start, 500)
    if r is not None and np.max(np.abs(half_deviance(p, X, y, v, r[0])[1])) <= 1e-6 * v.sum():
        return r
    b = start.copy()
    for _ in range(500):
        f, g, H = half_deviance(p, X, y, v, b)
        d = np.linalg.solve(H, -g)
        t = 1.0
        while t > 1e-14:
            nb = b + t * d
            if (X @ nb).min() > 0 and half_deviance(p, X, y, v, nb)[0] <= f:
                break
            t *= 0.5
        b = b + t * d
        if np.max(np.abs(t * d)) < 1e-14 * (1 + np.max(np.abs(b))):
            break
    r = _statsmodels(p, X, y, v, b, 50)
    assert r is not None, "statsmodels left the domain from the in-domain minimum"
    drift = float(np.max(np.abs(r[0] - b) / r[1]))
    assert drift < 1e-6, drift
    return r


# --- the streams -----------------------------------------------------------


def draws(
    seed: int,
    n: int = 3000,
    beta: tuple[float, ...] = (0.5, 0.15, -0.1),
    shape: float = 2.0,
    scale: float = 1.0,
    xdist: str = "uniform",
    p: float = 2.0,
    sort: bool = False,
    true_log: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.random.Generator]:
    rng = np.random.default_rng([5785, seed])
    k = len(beta)
    Z = rng.standard_normal((n, k - 1)) if xdist == "normal" else rng.uniform(-1, 1, (n, k - 1))
    if xdist == "rare":
        Z[:, 1] = (rng.random(n) < 0.02).astype(float)
    if sort:
        Z = Z[np.argsort(Z[:, 0])]
    X = np.column_stack([np.ones(n), Z])
    if true_log:
        mu = np.exp(0.7 - 0.5 * Z[:, 0] + 0.3 * Z[:, 1])
    else:
        eta = X @ np.array(beta)
        assert eta.min() > 0, eta.min()
        mu = scale * eta ** (-1.0 / (p - 1.0))
    if p == 2.0:
        y = rng.gamma(shape, mu / shape)
    else:
        lam = mu ** (2 - p) / (2 - p)
        N = rng.poisson(lam)
        y = np.zeros(n)
        pos = N > 0
        y[pos] = rng.gamma(N[pos] * (2 - p) / (p - 1), (p - 1) * mu[pos] ** (p - 1))
    return X, y, rng


def _plain(**kw: object) -> Callable[[int], Case]:
    def make(s: int) -> Case:
        X, y, _ = draws(s, **kw)  # type: ignore[arg-type]
        return float(kw.get("p", 2.0)), X, y, None, float("inf")  # type: ignore[arg-type]

    return make


def _first(value: float) -> Callable[[int], Case]:
    def make(s: int) -> Case:
        X, y, _ = draws(s)
        y[0] = value
        return 2.0, X, y, None, float("inf")

    return make


def _outlier(row: int) -> Callable[[int], Case]:
    def make(s: int) -> Case:
        X, y, _ = draws(s)
        y[row] *= 1000.0
        return 2.0, X, y, None, float("inf")

    return make


def _twenty(s: int) -> Case:
    slopes = np.random.default_rng([5786, s]).normal(0, 0.03, 20)
    X, y, _ = draws(s, n=10_000, beta=(1.0, *slopes), xdist="normal")
    return 2.0, X, y, None, float("inf")


def _weights(s: int) -> Case:
    X, y, rng = draws(s)
    return 2.0, X, y, rng.choice([0.0, 0.5, 3.0], len(y)), float("inf")


def _inverse_gaussian(s: int) -> Case:
    rng = np.random.default_rng([5787, s])
    n = 3000
    X = np.column_stack([np.ones(n), rng.uniform(-1, 1, (n, 2))])
    mu = (X @ [0.5, 0.15, -0.1]) ** -0.5
    return 3.0, X, rng.wald(mu, 4.0), None, float("inf")


def _half_life(s: int) -> Case:
    X, y, _ = draws(s)
    return 2.0, X, y, None, 300.0


CASES: dict[str, Callable[[int], Case]] = {
    "base": _plain(),
    "first_y_tiny": _first(1e-9),
    "first_y_huge": _first(1e6),
    "outlier_at_row_30": _outlier(30),
    "outlier_at_row_1500": _outlier(1500),
    "eta_to_the_edge": _plain(beta=(0.5, 0.45, 0.0)),
    "heavy_noise": _plain(shape=0.5),
    "twenty_features": _twenty,
    "covariate_order": _plain(sort=True),
    "rare_category_near_0": _plain(beta=(0.5, 0.1, -0.35), xdist="rare"),
    "log_linear_truth": _plain(true_log=True),
    "weights_with_zeros": _weights,
    "target_in_other_units": _plain(scale=1000.0),
    "tweedie_1_5_zeros": _plain(p=1.5, beta=(0.8, 0.15, -0.1)),
    "tweedie_3": _inverse_gaussian,
    "half_life_300": _half_life,
}


def weights_of(n: int, w: np.ndarray | None, half_life: float) -> np.ndarray:
    v = np.ones(n) if w is None else w.astype(float)
    if np.isfinite(half_life):
        v = v * 0.5 ** ((n - 1 - np.arange(n)) / half_life)
    return v


def online(
    p: float, X: np.ndarray, y: np.ndarray, w: np.ndarray | None, half_life: float
) -> tuple[np.ndarray, int]:
    """The bank's coefficients after the stream, its defaults otherwise, and
    its null predictions after the refit: every row up to the one that
    fills the refit's buffer (its default ``max(200, 10 k)`` learned rows)
    is null under these links, every coefficient 0 giving no mean."""
    cols = {f"x{j}": X[:, j] for j in range(1, X.shape[1])}
    kw: dict[str, object] = {"half_life": half_life}
    if w is not None:
        cols["w"] = w
        kw["weight"] = "w"
    if p == 2.0:
        kw.update(family="gamma", link="inverse")
    else:
        kw.update(family="tweedie", link="power", var_power=p)
    s = rp.spec.glm("m", targets=["y"], features=list(cols)[: X.shape[1] - 1], **kw)  # type: ignore[arg-type]
    bank = rp.ModelBank([s])
    out = bank.fit_predict(pl.DataFrame({**cols, "y": y}))
    pred = out["m"].struct.field("pred_y").to_numpy()
    finite = np.isfinite(pred)
    if not finite.any():
        nulls = len(y)
    else:
        first = int(np.argmax(finite))
        nulls = int(np.isnan(pred[first:]).sum())
    coef = np.array([r["coef"] for r in bank.coef().iter_rows(named=True)])
    return coef, nulls


def gap(name: str, seed: int) -> tuple[float, int]:
    """The worst coefficient's distance from statsmodels', in its standard
    errors, and the null predictions after the refit."""
    p, X, y, w, half_life = CASES[name](seed)
    b, se = batch(p, X, y, weights_of(len(y), w, half_life))
    got, nulls = online(p, X, y, w, half_life)
    return float(np.max(np.abs(got - b) / se)), nulls


# --- the essentials --------------------------------------------------------

#: Three cells at seed 0, each held at half as much again as measured
#: (2026-11-24, macOS arm64): 0.058, 4.36 and 0.051 standard errors, and no
#: null prediction after the refit. Before the exact step the outlier's
#: seed sat 58 standard errors off with 454 null predictions after it.
ESSENTIAL = {"base": 0.09, "outlier_at_row_1500": 6.5, "tweedie_1_5_zeros": 0.08}


@pytest.mark.parametrize("name", sorted(ESSENTIAL))
def test_one_seed_of_three_cases(name):
    se, nulls = gap(name, 0)
    assert se <= ESSENTIAL[name], se
    assert nulls == 0, nulls


# --- the grid ----------------------------------------------------------------

SEEDS = range(20)

ledger.guard(
    "canonical_links",
    lambda: [online(*CASES[name](s))[0] for name in CASES for s in (0, 1)],
    oracles=["statsmodels"],
)

#: Each case's worst seed of 20, at half as much again as measured
#: (2026-11-24, macOS arm64; the median seed's in the comment), the
#: ledger holding the measured values. The cases with a far target are the
#: widest: the row's curvature enters ``G`` at the fit around it, which a
#: target 1000 times its mean moves far from the batch fit's.
GRID = {
    "base": 0.45,  # 0.29 (median 0.07)
    "first_y_tiny": 0.45,  # 0.27 (0.07)
    "first_y_huge": 1.8,  # 1.18 (0.79)
    "outlier_at_row_30": 7.0,  # 4.57 (1.65)
    "outlier_at_row_1500": 9.5,  # 6.22 (1.04)
    "eta_to_the_edge": 1.3,  # 0.86 (0.06)
    "heavy_noise": 0.6,  # 0.38 (0.17)
    "twenty_features": 1.3,  # 0.85 (0.58)
    "covariate_order": 0.85,  # 0.54 (0.09)
    "rare_category_near_0": 0.75,  # 0.48 (0.21)
    "log_linear_truth": 0.55,  # 0.34 (0.07)
    "weights_with_zeros": 0.5,  # 0.33 (0.12)
    "target_in_other_units": 0.45,  # 0.29 (0.06)
    "tweedie_1_5_zeros": 0.55,  # 0.34 (0.09)
    "tweedie_3": 0.6,  # 0.37 (0.14)
    "half_life_300": 0.2,  # 0.11 (0.03)
}


@pytest.mark.oracle_grid(reason="16 adversarial streams over 20 seeds, a statsmodels fit each")
@pytest.mark.parametrize("name", sorted(CASES))
def test_each_stream_over_twenty_seeds(name):
    gaps, nulls = zip(*(gap(name, s) for s in SEEDS), strict=True)
    ledger.record("canonical_links", name, gaps)
    ledger.record("canonical_links", f"{name}_nulls", nulls, exact=True)
    assert max(gaps) <= GRID[name], (name, np.median(gaps), max(gaps))
