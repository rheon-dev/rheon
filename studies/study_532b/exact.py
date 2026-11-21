"""Exact window statistics, for measuring both Polars and the bucketed form
against the true value: the sum correctly rounded (``math.fsum``), and the
variance from exact rational arithmetic, rounded once."""

from __future__ import annotations

import math
from fractions import Fraction


def exact_sum(vals: list[float]) -> float:
    return math.fsum(vals)


def exact_mean(vals: list[float]) -> float:
    return float(sum(Fraction(v) for v in vals) / len(vals))


def exact_var(vals: list[float], ddof: int = 1) -> float | None:
    n = len(vals)
    if n <= ddof:
        return None
    fr = [Fraction(v) for v in vals]
    m = sum(fr) / n
    return float(sum((v - m) * (v - m) for v in fr) / (n - ddof))
