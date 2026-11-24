"""``pca`` against scikit-learn and scipy.

- **``IncrementalPCA.partial_fit``**: at no decay and unit weights the
  incremental SVD is the update scikit-learn's ``IncrementalPCA`` makes,
  one row a batch and in the mean form. Fed a first batch of ``q`` rows (the
  directions kept; ``q`` rows span ``q - 1``, so nothing is dropped yet)
  and then one row at a time, it is the same fit after every row, the
  truncation included: components up to sign, ``explained_variance_`` and
  ``explained_variance_ratio_``.
- **``PCA(svd_solver="full")``** on the decay-weighted rows: PCA takes no
  sample weights and centres on the unweighted mean, so it is given ``Y =
  sqrt(u_i / U) (x_i - m_u)`` stacked with ``-Y``, whose mean is 0 and
  whose scatter is twice the weighted covariance. With every direction kept
  the incremental SVD is that covariance's eigendecomposition.
- **``scipy.linalg.eigh``** of ``np.cov(x, aweights=u, bias=True)``, a
  second path to the same covariance.
- **``scipy.stats.chi2``**: the gate's critical value.
- **Convergence of CCIPCA and Oja's rule** to ``PCA``'s subspace by
  ``scipy.linalg.subspace_angles``: statistical, on separated spectra, the
  worst seed of a grid held to a bound measured here. No library computes
  either recursion; ``tests/test_pca.py`` holds both to their definitions.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
import scipy.linalg
import scipy.stats
from sklearn.decomposition import PCA, IncrementalPCA

import ledger
from rheon import polars as rp

TIER = "mixed"

INF = float("inf")

#: The incremental SVD against scikit-learn, in radians of the largest
#: principal angle and relative in the variances and shares: measured at
#: most 7.0e-13 rad and 5.4e-13 against ``IncrementalPCA``, 1.4e-14 rad and
#: 3.3e-13 against the weighted ``PCA`` and ``eigh``, over the cases below
#: (1.5 times the largest, up to a power of ten).
TOL_ANGLE = 1e-11
TOL_VAR = 1e-11


def stream(n: int, spectrum: list[float], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    k = len(spectrum)
    basis = np.linalg.qr(rng.standard_normal((k, k)))[0]
    return (rng.standard_normal((n, k)) * np.sqrt(spectrum)) @ basis.T + 5.0


def frame(x: np.ndarray, w: np.ndarray | None = None) -> pl.DataFrame:
    df = pl.DataFrame(x, schema=[f"x{i}" for i in range(x.shape[1])])
    return df if w is None else df.with_columns(w=pl.Series(w))


def emitted(out: pl.DataFrame, t: int, r: int, k: int):
    """Row ``t``'s components: ``(loadings r x k, variances, shares)``, or
    ``None`` where any is withheld."""
    load = np.array([[out[f"pc{j}_loading_x{i}"][t] for i in range(k)] for j in range(r)])
    var = np.array([out[f"pc{j}_var"][t] for j in range(r)])
    share = np.array([out[f"pc{j}_share"][t] for j in range(r)])
    if any(v is None for v in load.ravel()) or any(v is None for v in var):
        return None
    return load.astype(float), var.astype(float), share.astype(float)


def angle(a: np.ndarray, b: np.ndarray) -> float:
    """The largest angle between matched components, each a 1-D space."""
    return max(
        float(scipy.linalg.subspace_angles(a[j][:, None], b[j][:, None]).max())
        for j in range(a.shape[0])
    )


@pytest.mark.parametrize(("k", "r", "seed"), [(3, 1, 0), (5, 2, 1), (8, 3, 2), (8, 7, 3)])
def test_the_svd_is_incremental_pca_row_by_row(k, r, seed):
    """At no decay and unit weights, after a first batch of ``q`` rows and
    then single rows, scikit-learn's fit after rows ``0..t-1`` is what row
    ``t`` reads, truncation and all, for ``q = r + 1 < k``."""
    spectrum = list(9.0 * 0.6 ** np.arange(k))
    x = stream(1200, spectrum, seed)
    q = min(r + 1, k)
    out = (
        rp.ModelBank(
            [rp.spec.pca("p", features=[f"x{i}" for i in range(k)], n_components=r, half_life=INF)]
        )
        .fit_predict(frame(x))
        .unnest("p")
    )
    ipca = IncrementalPCA(n_components=q)
    ipca.partial_fit(x[:q])
    worst_angle = worst_var = 0.0
    compared = 0
    for t in range(q, 1200):
        got = emitted(out, t, r, k)
        if got is not None and t % 5 == 0:
            load, var, share = got
            worst_angle = max(worst_angle, angle(load, ipca.components_[:r]))
            ev = ipca.explained_variance_[:r]
            worst_var = max(worst_var, float(np.max(np.abs(var - ev) / ev)))
            ratio = ipca.explained_variance_ratio_[:r]
            worst_var = max(worst_var, float(np.max(np.abs(share - ratio) / ratio)))
            compared += 1
        ipca.partial_fit(x[t : t + 1])
    assert compared > 150
    assert worst_angle < TOL_ANGLE, worst_angle
    assert worst_var < TOL_VAR, worst_var


def weighted_pca(x: np.ndarray, u: np.ndarray) -> PCA:
    keep = u > 0
    x, u = x[keep], u[keep]
    m = (u[:, None] * x).sum(axis=0) / u.sum()
    y = np.sqrt(u / u.sum())[:, None] * (x - m)
    return PCA(svd_solver="full").fit(np.vstack([y, -y]))


@pytest.mark.parametrize("lam", [1.0, 0.995, 0.98])
@pytest.mark.parametrize("seed", [0, 1])
def test_the_full_svd_is_pca_of_the_weighted_rows(lam, seed):
    """Every direction kept (``n_components = k``), under decay and unequal
    weights with one row in ten at 0: PCA on the decay-weighted rows,
    each written component up to sign and its share; ``eigh`` of numpy's
    weighted covariance for its variance, times Bessel's factor at Kish's
    size."""
    k = 5
    rng = np.random.default_rng(100 + seed)
    x = stream(1500, [16.0, 6.0, 2.0, 0.6, 0.2], seed)
    w = rng.uniform(0.5, 1.5, 1500)
    w[rng.uniform(size=1500) < 0.1] = 0.0
    s = rp.spec.pca("p", features=[f"x{i}" for i in range(k)], n_components=k, lam=lam, weight="w")
    out = rp.ModelBank([s]).fit_predict(frame(x, w)).unnest("p")
    checked = 0
    for t in range(200, 1500, 50):
        # A row of weight 0 counts in no clock: each row's age is the rows
        # of weight after it.
        after = np.cumsum((w[:t] > 0)[::-1])[::-1] - (w[:t] > 0)
        u = w[:t] * lam**after
        pca = weighted_pca(x[:t], u)
        c = np.cov(x[:t], rowvar=False, aweights=u, bias=True)
        vals = scipy.linalg.eigh(c, eigvals_only=True)[::-1]
        n_kish = u.sum() ** 2 / (u**2).sum()
        bessel = n_kish / (n_kish - 1)
        for j in range(k):
            var = out[f"pc{j}_var"][t]
            if var is None:
                continue
            load = np.array([out[f"pc{j}_loading_x{i}"][t] for i in range(k)])
            got = float(
                scipy.linalg.subspace_angles(load[:, None], pca.components_[j][:, None]).max()
            )
            assert got < TOL_ANGLE, (t, j, got)
            share = out[f"pc{j}_share"][t]
            assert abs(share - pca.explained_variance_ratio_[j]) < TOL_VAR, (t, j)
            assert abs(var - bessel * vals[j]) / vals[j] < TOL_VAR, (t, j)
            checked += 1
    assert checked >= 40, checked


def test_the_gates_critical_value_is_chi_squared_at_one_percent():
    assert abs(scipy.stats.chi2.ppf(0.99, 2) - 9.210340371976184) < 1e-12


#: The stochastic methods' worst seed, in radians of the largest principal
#: angle between the first two components' subspace and PCA's on the same
#: rows, measured over seeds 0-9 on the separated spectrum below: no decay,
#: after 20,000 rows: CCIPCA 0.0032, Oja (its default gain, 4) 0.0045; under
#: a half-life of 2,000 rows: CCIPCA 0.0029, Oja 0.0082. Each bound is about
#: 1.5 times its measurement. A statistical bound, not rounding: no library
#: computes either recursion.
CONVERGENCE = {
    ("ccipca", INF): 0.005,
    ("oja", INF): 0.007,
    ("ccipca", 2000.0): 0.0045,
    ("oja", 2000.0): 0.0125,
}


SEEDS = range(10)
SPECTRUM = [9.0, 4.0, 1.0, 0.6, 0.4, 0.2]


def _loadings(method: str, half_life: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """The stream of ``seed`` and the first two loadings ``pca`` reads before
    its last row."""
    k = len(SPECTRUM)
    x = stream(20_000, SPECTRUM, seed)
    s = rp.spec.pca(
        "p",
        features=[f"x{i}" for i in range(k)],
        n_components=2,
        method=method,
        half_life=half_life,
    )
    out = rp.ModelBank([s]).fit_predict(frame(x)).unnest("p")
    got = emitted(out, 19_999, 2, k)
    assert got is not None, (method, seed)
    return x, got[0]


ledger.guard(
    "streampca_convergence",
    lambda: [_loadings(m, h, seed)[1] for (m, h) in CONVERGENCE for seed in SEEDS],
    oracles=["scikit-learn", "scipy"],
)


@pytest.mark.oracle_grid(reason="ten seeds of 20,000 rows for each method and half-life")
@pytest.mark.parametrize(("method", "half_life"), list(CONVERGENCE))
def test_the_stochastic_methods_converge_to_pca(method, half_life):
    angles = []
    for seed in SEEDS:
        x, load = _loadings(method, half_life, seed)
        age = np.arange(19_998, -1, -1)
        u = np.ones(19_999) if half_life == INF else 0.5 ** (age / half_life)
        pca = weighted_pca(x[:19_999], u)
        angles.append(float(scipy.linalg.subspace_angles(load.T, pca.components_[:2].T).max()))
    hl = "inf" if half_life == INF else f"{half_life:g}"
    ledger.record("streampca_convergence", f"{method}_hl{hl}_angle", angles)
    assert max(angles) < CONVERGENCE[(method, half_life)], max(angles)
