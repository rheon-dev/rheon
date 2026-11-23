"""Power variation and kernel-weighted autocovariance sums, written from
their definitions in numpy, never from the model's code: the oracle
`tests/test_power_variation.py` holds ``rheon.spec.power_variation`` to
(docs/PLAN.md task 555).

For returns ``r_j`` at final weights ``ω_j`` and a scale ``K``, ``R_j =
r_{j-K+1} + ... + r_j``:

.. code-block:: text

    powers p_1..p_m:  (1/K) Σ_j ω_j Π_i |R_{j-(m-i)K}|^{p_i}
    kernel k, H:      (1/K) Σ_j ω_j [ d_j² + 2 Σ_h c_h d_j d_{j-hK} ]
                      c_h = k(h/(H+1)), or k((h-1)/H) flat-top;
                      d = R - mean(R) (ω-weighted) centered, R otherwise

Each sum is ``math.fsum`` of its terms, a ``K``-step return too; a centered
sum is taken about the mean directly. ``μ_p`` comes from
``scipy.special.gamma``, a library's.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import gamma


def mu(p: float) -> float:
    """``E|Z|^p`` of a standard normal: ``2^(p/2) Γ((p+1)/2) / Γ(1/2)``."""
    return float(2 ** (p / 2) * gamma((p + 1) / 2) / gamma(0.5))


def kernel(name: str, x: float) -> float:
    if not 0.0 <= x <= 1.0:
        return 0.0
    if name == "bartlett":
        return 1.0 - x
    if name == "parzen":
        return 1 - 6 * x**2 + 6 * x**3 if x <= 0.5 else 2 * (1 - x) ** 3
    if name == "tukey_hanning":
        return (1 + math.cos(math.pi * x)) / 2
    if name == "rectangular":
        return 1.0
    raise ValueError(name)


def weights(name: str, h_max: int, flat_top: bool = False) -> list[float]:
    """``c_h`` for ``h = 1..H``."""
    return [
        kernel(name, (h - 1) / h_max if flat_top else h / (h_max + 1)) for h in range(1, h_max + 1)
    ]


def kstep(r, k: int) -> list[float | None]:
    r = [float(x) for x in r]
    return [math.fsum(r[j + 1 - k : j + 1]) if j + 1 >= k else None for j in range(len(r))]


def multipower(r, om, powers, k: int = 1) -> float:
    big = kstep(r, k)
    m = len(powers)
    terms = []
    for j in range(m * k - 1, len(r)):
        acc = 1.0
        for i, p in enumerate(powers):
            acc *= abs(big[j - (m - 1 - i) * k]) ** p
        terms.append(om[j] * acc)
    return math.fsum(terms) / k


def kernel_sum(r, om, name: str, h_max: int, *, flat_top=False, center=False, k: int = 1):
    big = kstep(r, k)
    c = weights(name, h_max, flat_top)
    mean = 0.0
    if center:
        num = [om[j] * x for j, x in enumerate(big) if x is not None]
        den = [om[j] for j, x in enumerate(big) if x is not None]
        mean = math.fsum(num) / math.fsum(den) if den and math.fsum(den) > 0 else 0.0
    terms = []
    for j, x in enumerate(big):
        if x is None:
            continue
        d = x - mean
        terms.append(om[j] * d * d)
        for h, ch in enumerate(c, start=1):
            if j < (h + 1) * k - 1:
                break
            terms.append(om[j] * 2 * ch * d * (big[j - h * k] - mean))
    return math.fsum(terms) / k


def kish(om) -> float:
    om = np.asarray(om, dtype=float)
    v = math.fsum(om * om)
    return math.fsum(om) ** 2 / v if v > 0 else 0.0


def term(r, om, spec: dict, n: float) -> float:
    """One term of the builder's ``terms`` dict on ``r`` at weights ``om``."""
    k = spec.get("scale", 1)
    if "powers" in spec:
        s = multipower(r, om, spec["powers"], k)
        if spec.get("normalize"):
            p = spec["powers"]
            e = sum(p) / 2 - 1
            return s * n**e / math.prod(mu(x) for x in p) if n > 0 or e >= 0 else math.nan
        return s
    return kernel_sum(
        r,
        om,
        spec["kernel"],
        spec["bandwidth"],
        flat_top=spec.get("flat_top", False),
        center=spec.get("center", False),
        k=k,
    )


def scale_of(r, om, spec: dict) -> float:
    """The size a term's gap is read against: its own magnitude, or for a
    kernel sum, whose products cancel, ``(2H + 1)`` times the sum of the
    squares of its returns."""
    if "powers" in spec:
        return 0.0
    return (2 * spec["bandwidth"] + 1) * multipower(r, om, [2.0], spec.get("scale", 1))
