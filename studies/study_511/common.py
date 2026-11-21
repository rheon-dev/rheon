"""Shared helpers: seeded streams, a table printer, the decay convention.

Decay convention (as the models': half-life in clock units, a gap `dt` since
the previous row decays everything by `lam = 0.5 ** (dt / half_life)`; the row
then enters at weight `w`). `half_life = inf` is `lam = 1`. A zero-weight row
advances the clock and learns nothing.
"""

from __future__ import annotations

import numpy as np


def lam_of(dt: float, half_life: float) -> float:
    return 1.0 if np.isinf(half_life) else 0.5 ** (dt / half_life)


def stream(seed: int, n: int, k: int, *, hetero: bool = False, offset: float = 0.0):
    """Seeded regression stream: x (n,k), y (n,), dt (n,), w (n,)."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((k, k)) * 0.5 + np.eye(k)
    x = rng.standard_normal((n, k)) @ a + offset
    beta = rng.standard_normal(k)
    sd = 1.0 + (0.8 * np.abs(x[:, 0] - offset) if hetero else 0.0)
    y = 0.7 + x @ beta + sd * rng.standard_normal(n)
    dt = rng.uniform(0.5, 1.5, n)
    w = rng.uniform(0.5, 2.0, n)
    return x, y, dt, w


def rel(a, b) -> float:
    """Largest relative gap, scaled by the larger magnitude (floor 1e-300)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.max(np.abs(a - b) / np.maximum(np.maximum(np.abs(a), np.abs(b)), 1e-300)))


def ulps(a, b) -> float:
    """Largest gap in units of the last place of the larger magnitude."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.max(np.abs(a - b) / np.spacing(np.maximum(np.abs(a), np.abs(b)) + 1e-300)))


def table(title: str, header: list[str], rows: list[list]) -> None:
    print(f"\n### {title}\n")
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join("---" for _ in header) + "|")
    for r in rows:
        print("| " + " | ".join(fmt(v) for v in r) + " |")


def fmt(v) -> str:
    if isinstance(v, float):
        if v == 0:
            return "0"
        if abs(v) < 1e-3 or abs(v) >= 1e6:
            return f"{v:.2e}"
        return f"{v:.4g}"
    return str(v)
