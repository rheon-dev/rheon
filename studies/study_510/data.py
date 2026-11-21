"""Seeded synthetic streams for study 510 (no data files).

``stream`` gives ``k`` features and ``T`` targets with a mix of relations a
dependence measure should tell apart -- linear, a V (non-monotone, so Pearson
and Spearman miss it and xi and mutual information do not), a step, noise --
some columns rounded so ranks tie, and the row kinds the window must handle:
zero-weight rows (the first row among them when asked), rows with a NaN or an
inf in one column, and whole rows of NaN.
"""

from __future__ import annotations

import numpy as np


def stream(
    n: int,
    k: int = 4,
    t: int = 2,
    seed: int = 0,
    gaps: bool = True,
    ties: bool = True,
    weights: str = "unit",
    first_zero: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(x (n, k), y (n, t), w (n,))``.

    ``weights``: ``"unit"`` (ones, with zeros where gaps are asked),
    ``"float"`` (uniform on [0.2, 2], zeros likewise) or ``"ones"`` (ones only).
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, k))
    x = z.copy()
    y = np.empty((n, t))
    kinds = [lambda a: a, lambda a: np.abs(a), lambda a: (a > 0.3).astype(float), np.sin]
    for j in range(t):
        f = kinds[j % len(kinds)]
        y[:, j] = f(z[:, j % k]) + 0.5 * rng.standard_normal(n)
    if k > 2:
        x[:, 2] = rng.standard_normal(n)  # a feature unrelated to anything
    if ties:
        x[:, 1 % k] = np.round(x[:, 1 % k], 1)  # ties in a feature (about 40 values)
        y[:, t - 1] = np.round(y[:, t - 1], 1)  # ties in a target
    if weights == "float":
        w = rng.uniform(0.2, 2.0, n)
    else:
        w = np.ones(n)
    if gaps and weights != "ones":
        w[rng.random(n) < 0.05] = 0.0
        if first_zero:
            w[0] = 0.0
        cells = rng.random((n, k)) < 0.03
        x[cells] = np.nan
        x[rng.random((n, k)) < 0.01] = np.inf
        y[rng.random((n, t)) < 0.02] = -np.inf
        y[rng.random((n, t)) < 0.02] = np.nan
        whole = rng.random(n) < 0.02
        x[whole] = np.nan
        y[whole] = np.nan
    return x, y, w


def gaussian(n: int, rho: float, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """A standard bivariate normal with correlation ``rho``: ``(x (n, 1), y (n, 1))``."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(n)
    b = rho * a + np.sqrt(1.0 - rho * rho) * rng.standard_normal(n)
    return a[:, None], b[:, None]
