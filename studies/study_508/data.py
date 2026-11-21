"""Seeded synthetic streams for study 508 (no data files).

Real-valued streams (ADWIN, Page-Hinkley, KSWIN): unit-variance Gaussian noise
around a mean that is constant (``"none"``), jumps (``"step"``) or moves
linearly over ``ramp`` rows (``"gradual"``). Binary streams (DDM, EDDM, and
ADWIN on an error stream): Bernoulli bits whose rate does the same. A drift
starts at row ``at`` (0-based); rows before it are the null.
"""

from __future__ import annotations

import numpy as np


def level(kind: str, n: int, at: int, size: float, ramp: int) -> np.ndarray:
    t = np.arange(n, dtype=np.float64)
    if kind == "none":
        return np.zeros(n)
    if kind == "step":
        return np.where(t >= at, size, 0.0)
    if kind == "gradual":
        return size * np.clip((t - at + 1) / ramp, 0.0, 1.0)
    raise ValueError(kind)


def real(
    kind: str, n: int, seed: int, at: int = 0, size: float = 1.0, ramp: int = 500, sd: float = 1.0
) -> np.ndarray:
    rng = np.random.default_rng([508, seed])
    return level(kind, n, at, size, ramp) + sd * rng.standard_normal(n)


def binary(
    kind: str, n: int, seed: int, at: int = 0, p0: float = 0.1, p1: float = 0.3, ramp: int = 500
) -> np.ndarray:
    rng = np.random.default_rng([5080, seed])
    p = p0 + level(kind, n, at, p1 - p0, ramp)
    return (rng.random(n) < p).astype(np.float64)


def weights(n: int, seed: int, frac_zero: float = 0.2, first_zero: bool = True) -> np.ndarray:
    """Weights uniform on [0.2, 2] with a share of exact zeros, the first row among them."""
    rng = np.random.default_rng([5081, seed])
    w = rng.uniform(0.2, 2.0, n)
    w[rng.random(n) < frac_zero] = 0.0
    if first_zero:
        w[0] = 0.0
    return w
