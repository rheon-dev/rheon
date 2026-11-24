"""``gmm`` against scikit-learn's ``GaussianMixture``, each covariance type.

One pass of online EM over a stream with no decay ends near batch EM's
maximum-likelihood fit of the same rows: within the sampling error of a
fit of that many rows, which is what ``GaussianMixture`` computes, to its
``tol``. Two components of different sizes, spreads and orientations,
well apart, so every type's fit is the same mixture up to the labels, which
are matched by the first coordinate of the means.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from sklearn.mixture import GaussianMixture

from rheon import polars as rp

TIER = "essential"

N = 8000
FEATURES = ["a", "b"]


def pair(n: int = N, seed: int = 4) -> np.ndarray:
    """Two components, 60 and 40 percent of the rows: one at the origin,
    correlated, the other at (6, 4), narrower and rounder."""
    rng = np.random.default_rng(seed)
    first = rng.random(n) < 0.6
    x = np.where(
        first[:, None],
        rng.multivariate_normal([0.0, 0.0], [[1.0, 0.5], [0.5, 1.0]], n),
        rng.multivariate_normal([6.0, 4.0], [[0.5, -0.1], [-0.1, 0.4]], n),
    )
    return x


def ours(x: np.ndarray, kind: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The weights, means and covariances (``k x k`` each) the bank ends
    with, one pass and no decay."""
    spec = rp.spec.gmm(
        "g", features=FEATURES, n_components=2, covariance_type=kind, half_life=float("inf")
    )
    bank = rp.ModelBank([spec])
    bank.fit(pl.DataFrame(x, schema=FEATURES))
    coef = bank.coef().pivot(on="term", index="cluster", values="coef").sort("cluster")
    weights, means, covs = [], [], []
    for r in coef.iter_rows(named=True):
        weights.append(r["weight"])
        means.append([r["a"], r["b"]])
        if kind == "diag":
            covs.append(np.diag([r["var_a"], r["var_b"]]))
        elif kind == "spherical":
            covs.append(np.eye(2) * r["var"])
        else:
            covs.append(np.array([[r["cov_a_a"], r["cov_a_b"]], [r["cov_a_b"], r["cov_b_b"]]]))
    return np.array(weights), np.array(means), np.array(covs)


def theirs(x: np.ndarray, kind: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``GaussianMixture``'s, its covariances as ``k x k`` each."""
    gm = GaussianMixture(
        n_components=2, covariance_type=kind, reg_covar=1e-6, tol=1e-8, max_iter=1000,
        random_state=0,
    ).fit(x)  # fmt: skip
    c = gm.covariances_
    covs = {
        "full": c,
        "diag": np.array([np.diag(v) for v in c]) if kind == "diag" else None,
        "spherical": np.array([np.eye(2) * v for v in c]) if kind == "spherical" else None,
        "tied": np.array([c, c]) if kind == "tied" else None,
    }[kind]
    return gm.weights_, gm.means_, covs


@pytest.mark.parametrize("kind", ["full", "diag", "spherical", "tied"])
def test_one_pass_of_online_em_ends_at_scikit_learns_fit(kind: str):
    x = pair()
    w, m, c = ours(x, kind)
    tw, tm, tc = theirs(x, kind)
    order = np.argsort(tm[:, 0])
    tw, tm, tc = tw[order], tm[order], tc[order]
    mine = np.argsort(m[:, 0])
    w, m, c = w[mine], m[mine], c[mine]
    # At 8,000 rows a weight's standard error is about 0.005, a mean's about
    # 0.015 and a variance's about 0.02: online EM's one pass lands within a
    # few of them of the maximum-likelihood fit.
    assert np.abs(w - tw).max() < 0.02, (kind, w, tw)
    assert np.abs(m - tm).max() < 0.06, (kind, m, tm)
    assert np.abs(c - tc).max() < 0.08, (kind, c, tc)
