"""The papers' realized measures and jump tests, written from the papers in
numpy, never from the model's code: the oracle `tests/test_realized.py`
holds ``rheon.spec.realized`` to (docs/PLAN.md task 499).

Barndorff-Nielsen and Shephard (2004, 2006) and Lee and Mykland (2008), for
``n`` returns ``r_j``, ``a_j = |r_j|`` and ``μ_p = E|Z|^p``:

.. code-block:: text

    RV = Σ r_j²
    BV = μ_1^-2 Σ_{j≥2} a_j a_{j-1}
    TV = μ_{2/3}^-3 Σ_{j≥3} (a_j a_{j-1} a_{j-2})^(2/3)
    TQ = n μ_{4/3}^-3 Σ_{j≥3} (a_j a_{j-1} a_{j-2})^(4/3)
    Z  = √n (1 - BV/RV) / √(ϑ max(1, TQ/BV²)),   ϑ = π²/4 + π - 5
    L_i = r_i / σ_i,  σ_i² = (1/(K-2)) Σ_{j=i-K+2}^{i-1} a_j a_{j-1}

Each sum is ``math.fsum`` of the terms, so a value is the paper's to the
rounding of its terms. The constants come from ``scipy.special.gamma``, a
library's, not the model's literals.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import gamma


def mu(p: float) -> float:
    """``E|Z|^p`` of a standard normal: ``2^(p/2) Γ((p+1)/2) / Γ(1/2)``."""
    return float(2 ** (p / 2) * gamma((p + 1) / 2) / gamma(0.5))


THETA = math.pi**2 / 4 + math.pi - 5


def _triples(a: np.ndarray, q: float) -> np.ndarray:
    return a[2:] ** q * a[1:-1] ** q * a[:-2] ** q


def measures(r, n: float | None = None) -> dict[str, float]:
    """RV, BV, TV, TQ and the adjusted ratio test of the returns ``r``, the
    papers' count of returns ``n`` the length of ``r`` unless given."""
    r = np.asarray(r, dtype=float)
    a = np.abs(r)
    n = float(len(r)) if n is None else n
    rv = math.fsum(r * r)
    bv = mu(1.0) ** -2 * math.fsum(a[1:] * a[:-1])
    tv = mu(2 / 3) ** -3 * math.fsum(_triples(a, 2 / 3))
    tq = n * mu(4 / 3) ** -3 * math.fsum(_triples(a, 4 / 3))
    z = math.sqrt(n) * (1 - bv / rv) / math.sqrt(THETA * max(1.0, tq / bv**2))
    return {"rv": rv, "bv": bv, "tv": tv, "tq": tq, "bns": z}


def weighted(t, r, w, half_life: float, count: float | None = None) -> dict[str, float]:
    """The measures of returns ``r`` learned at clocks ``t`` and weights
    ``w``, each term decayed by the age of its latest row from the last
    row, ``2^(-age / half_life)``, and weighted by that row's weight; ``n``
    Kish's effective count ``(Σ λw)² / Σ (λw)²``, or ``count``. Also the
    weight of the returns and ``σ²``, the weighted mean of the products."""
    t, r, w = (np.asarray(v, dtype=float) for v in (t, r, w))
    lw = np.exp2(-(t[-1] - t) / half_life) * w
    a = np.abs(r)
    nw = math.fsum(lw)
    n = nw**2 / math.fsum(lw * lw) if count is None else count
    s_bv = math.fsum(lw[1:] * a[1:] * a[:-1])
    p = math.fsum(lw[1:])
    out = {
        "rv": math.fsum(lw * r * r),
        "bv": mu(1.0) ** -2 * s_bv,
        "tv": mu(2 / 3) ** -3 * math.fsum(lw[2:] * _triples(a, 2 / 3)),
        "tq": n * mu(4 / 3) ** -3 * math.fsum(lw[2:] * _triples(a, 4 / 3)),
    }
    out["bns"] = (
        math.sqrt(n)
        * (1 - out["bv"] / out["rv"])
        / math.sqrt(THETA * max(1.0, out["tq"] / out["bv"] ** 2))
    )
    out["weight"] = nw
    out["sigma2"] = s_bv / p
    return out


def lee_mykland(r, k: int) -> np.ndarray:
    """``L_i`` for each return, its volatility from the ``K - 1`` returns
    before it (Definition 1); NaN before ``K - 1`` returns."""
    r = np.asarray(r, dtype=float)
    out = np.full(len(r), np.nan)
    a = np.abs(r)
    for i in range(k - 1, len(r)):
        before = a[i - k + 1 : i]
        out[i] = r[i] / math.sqrt(math.fsum(before[1:] * before[:-1]) / (k - 2))
    return out


def critical(n: float, beta: float) -> float:
    """The ``|L|`` past which Lee and Mykland's test at confidence ``β``
    flags a jump among ``n`` returns: ``C_n + S_n β*``, ``β* = -ln(-ln β)``."""
    c = math.sqrt(2 / math.pi)
    g = math.sqrt(2 * math.log(n))
    c_n = g / c - (math.log(math.pi) + math.log(math.log(n))) / (2 * c * g)
    return c_n + (1 / (c * g)) * -math.log(-math.log(beta))


def sv_returns(n: int, seed: int, jumps: bool) -> np.ndarray:
    """One day of ``n`` returns of a stochastic-volatility diffusion, 1% a
    day: an intraday U-shape times a log-OU factor of half-life ``n/10``
    returns; with ``jumps``, five jumps at 10, 30, 50, 70 and 90% of the day
    of +8, -10, +6, +12 and -7 local standard deviations."""
    rng = np.random.default_rng(seed)
    u = (np.arange(n) + 0.5) / n
    shape = 0.75 + 6.0 * (u - 0.5) ** 2
    kappa = math.log(2) / (n / 10)
    h = np.zeros(n)
    shocks = 0.3 * math.sqrt(2 * kappa) * rng.standard_normal(n)
    for j in range(1, n):
        h[j] = h[j - 1] * math.exp(-kappa) + shocks[j]
    r = 0.01 / math.sqrt(n) * np.sqrt(shape) * np.exp(h) * rng.standard_normal(n)
    if jumps:
        for at, size in [(0.1, 8.0), (0.3, -10.0), (0.5, 6.0), (0.7, 12.0), (0.9, -7.0)]:
            j = int(at * n)
            local = math.sqrt(np.mean(r[max(j - 20, 0) : j + 20] ** 2))
            r[j] += size * local
    return r
