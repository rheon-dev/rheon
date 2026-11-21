"""Seeded synthetic streams for the study (no data files)."""

from __future__ import annotations

import numpy as np


def regression(n: int, k: int = 3, seed: int = 0, noise: str = "skew"):
    """y = 1 + x'b + (1 + 0.5 |x_1|) e: heteroscedastic, so the expectile
    slopes differ by level; e skewed (centred lognormal), t(3), or integer."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, k))
    b = np.linspace(1.0, -1.0, k)
    if noise == "skew":
        e = rng.lognormal(0.0, 0.75, n) - np.exp(0.75**2 / 2)
    elif noise == "t3":
        e = rng.standard_t(3, n)
    else:
        e = rng.normal(size=n)
    y = 1.0 + X @ b + (1.0 + 0.5 * np.abs(X[:, 0])) * e
    if noise == "int":
        y = np.round(y)
    Z = np.column_stack([np.ones(n), X])
    return Z, y


def weights(n: int, seed: int, kind: str = "ones") -> np.ndarray:
    rng = np.random.default_rng(seed + 10_000)
    if kind == "ones":
        return np.ones(n)
    if kind == "random":
        return rng.uniform(0.2, 2.0, n)
    if kind == "zeros":  # random weights, a fifth of them 0, the first one too
        w = rng.uniform(0.2, 2.0, n)
        w[rng.random(n) < 0.2] = 0.0
        w[0] = 0.0
        return w
    raise ValueError(kind)


def decayed(w: np.ndarray, lam: float) -> np.ndarray:
    """Each row's weight decayed to the last row: w_i lam^(n-1-i)."""
    n = len(w)
    return w * lam ** np.arange(n - 1, -1, -1.0)
