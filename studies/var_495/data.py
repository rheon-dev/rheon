"""Seeded synthetic VAR data for the study (no data files; generated each run)."""

from __future__ import annotations

import numpy as np


def stable_coefs(k: int, p: int, rng: np.random.Generator, radius: float = 0.8) -> np.ndarray:
    """``(p, k, k)`` lag matrices whose companion matrix has spectral radius ``radius``."""
    a = rng.normal(size=(p, k, k)) / np.sqrt(k * p)
    comp = np.zeros((k * p, k * p))
    comp[:k] = np.hstack(list(a))
    comp[k:, :-k] = np.eye(k * (p - 1))
    rho = np.max(np.abs(np.linalg.eigvals(comp)))
    # Scaling A_j by s^j scales every companion eigenvalue by s.
    s = radius / rho
    return np.stack([a[j] * s ** (j + 1) for j in range(p)])


def simulate(
    t: int,
    k: int,
    p: int,
    seed: int,
    radius: float = 0.8,
    level: float = 0.0,
    zero: tuple[int, int] | None = (0, 1),
    burn: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """``t x k`` draws of a VAR(p) with correlated noise.

    ``zero=(e, j)`` zeroes variable ``j``'s lags in equation ``e`` (so ``j`` does
    not Granger-cause ``e``: the null holds for that pair); ``level`` is added to
    the constant so the series sit far from zero.
    """
    rng = np.random.default_rng(seed)
    a = stable_coefs(k, p, rng, radius)
    if zero is not None and k > 1:
        a[:, zero[0], zero[1]] = 0.0
    c = rng.normal(size=k)
    l_chol = np.linalg.cholesky(0.5 * np.eye(k) + 0.5 * np.full((k, k), 0.3) + 0.35 * np.eye(k))
    y = np.zeros((t + burn, k))
    mu = np.linalg.solve(np.eye(k) - a.sum(axis=0), c)
    y[:p] = mu
    for i in range(p, t + burn):
        y[i] = c + sum(a[j] @ y[i - 1 - j] for j in range(p)) + l_chol @ rng.normal(size=k)
    return y[burn:] + level, a
