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


# ---------------------------------------------------------------- study 511b helpers


def dt_stream(seed: int, n: int, *, zero_first: bool = True, zero_frac: float = 0.1):
    """Seeded (dt, w) with interleaved zero-weight rows and a zero-weight first row."""
    rng = np.random.default_rng(seed + 7919)
    dt = rng.uniform(0.5, 1.5, n)
    w = rng.uniform(0.5, 2.0, n)
    w[rng.random(n) < zero_frac] = 0.0
    if zero_first:
        w[0] = 0.0
    return dt, w


def omegas(dt, w, half_life, upto):
    """Weights at row `upto` (inclusive) of rows 0..upto: `w_i * prod_{j>i} lam_j`."""
    lam = np.array([lam_of(d, half_life) for d in dt[: upto + 1]])
    out = np.empty(upto + 1)
    acc = 1.0
    for i in range(upto, -1, -1):
        out[i] = w[i] * acc
        acc *= lam[i]
    return out


class EwMoments:
    """The mean-form accumulator `ew_cov` keeps: W, Q = sum of squared weights, m, C.

    `W' = lam W + w`, `Q' = lam^2 Q + w^2`, `a = lam W / W'`, `b = w / W'`,
    `d = x - m`, `m' = m + b d`, `C' = a C + a b d d'`. `w = 0` ages and learns nothing; the
    0/0 (`W' = 0`) is guarded. `s` is the ridge scale (`s' = a s`, 1 where `a <= 0`).
    """

    def __init__(self, k: int):
        self.k = k
        self.W = 0.0
        self.Q = 0.0
        self.m = np.zeros(k)
        self.C = np.zeros((k, k))
        self.s = 1.0

    def update(self, x, lam: float, w: float) -> None:
        Wo = lam * self.W
        W2 = Wo + w
        # a negative weight retracts a row: its squared weight leaves `Q` where a row's enters
        self.Q = lam * lam * self.Q + (-w * w if w < 0 else w * w)
        self.W = W2
        if W2 == 0.0:
            return
        a = Wo / W2
        b = w / W2
        self.s = 1.0 if a <= 0.0 else self.s * a
        if w == 0.0:
            self.C = a * self.C  # a == 1 for lam == 1; decayed C keeps the shape
            return
        d = np.asarray(x, float) - self.m
        self.m = self.m + b * d
        self.C = a * self.C + a * b * np.outer(d, d)

    def unbiased_factor(self) -> float:
        w2 = self.W * self.W
        return w2 / (w2 - self.Q) if self.Q > 0 and w2 - self.Q > 0 else float("nan")

    def raw(self):
        return self.C + np.outer(self.m, self.m)


def two_pass(x, dt, w, half_life, upto):
    """Weighted moments of rows 0..upto by definition: W, Q, m, C (divisor W), raw."""
    om = omegas(dt, w, half_life, upto)
    X = np.asarray(x[: upto + 1], float)
    W = om.sum()
    Q = (om**2).sum()
    if W == 0.0:
        k = X.shape[1]
        return 0.0, 0.0, np.zeros(k), np.zeros((k, k)), np.zeros((k, k))
    m = om @ X / W
    D = X - m
    C = (D * om[:, None]).T @ D / W
    raw = (X * om[:, None]).T @ X / W
    return W, Q, m, C, raw
