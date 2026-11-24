"""Reference (oracle) for the window operators: the definition, in decimal.

Each operator of ``rheon.polars`` (``ewm_mean``, ``ewm_var``, ``ewm_std``,
``ewm_sum``, ``ewm_rate``, looking back, and the mean, sum and rate looking
ahead) for one group's rows on a number clock with no clock event, computed
from its definition with ``decimal`` at 80 digits on the rows' own doubles:
the stamps, the values, the half-life and the window are read exactly, and
every power of the decay and every mass is taken at that precision.

It is ``tests/test_windows.py::loop`` with one difference, the one it is
here for. ``loop`` takes an interval's decayed mass in doubles, and the
window core takes it in doubles too; any such form has a rounding of its
own, and two forms that share one agree where both are wrong (the mass as
the difference of two masses from the reading row was 0 from 54 half-lives
in both, the review of 2026-11-23, WIN-1). At 80 digits the closed form of
the integral::

    integral_a^b lam^(T - s) ds = (h / ln 2) (lam^(T - b) - lam^(T - a))     lam = 2^(-1 / h)

loses nothing a double could show: its difference cancels ``log10(h / (b -
a))`` digits of 80, and a discount 1000 half-lives out is a decimal of
exponent -302, not a subnormal. So the reference is right wherever the
definition is, however far the window is read from its rows.

The rules of the definition are ``loop``'s: looking back a value is held
from the valued row before it (from the window's far edge for the first)
to its own row, and counts over the part of that interval inside the
window; looking ahead it is held from its own row to the next valued row,
cut at the window's far edge; a sum and a rate discount each value from its
own row; ``closed`` says which of the two edges' stamps the window holds; a
window looking ahead that no later row has shown whole is unresolved.
"""

from __future__ import annotations

import math
from decimal import Decimal, localcontext

#: The digits every quantity is taken at.
DIGITS = 80
_INF = Decimal("Infinity")


def _lam_pow(h: float, d: Decimal, ln2: Decimal) -> Decimal:
    """``lam^d`` for ``d >= 0``."""
    if math.isinf(h):
        return Decimal(1)
    if d == _INF:
        return Decimal(0)
    return (-(d / Decimal(h)) * ln2).exp()


def _between(h: float, forward: bool, t: Decimal, a: Decimal, b: Decimal, ln2: Decimal) -> Decimal:
    """``integral_a^b lam^(t - s) ds`` back, ``integral_a^b lam^(s - t) ds``
    ahead, the part of ``[a, b]`` on the far side of ``t`` none."""
    if forward:
        a = max(a, t)
    else:
        b = min(b, t)
    if b <= a:
        return Decimal(0)
    if math.isinf(h):
        return b - a
    c = Decimal(h) / ln2
    if forward:
        return c * (_lam_pow(h, a - t, ln2) - _lam_pow(h, b - t, ln2))
    return c * (_lam_pow(h, t - b, ln2) - _lam_pow(h, t - a, ln2))


def _mass(h: float, d: Decimal, ln2: Decimal) -> Decimal:
    """``integral_0^d lam^s ds``."""
    if d <= 0:
        return Decimal(0)
    if math.isinf(h):
        return d
    return Decimal(h) / ln2 * (1 - _lam_pow(h, d, ln2))


def loop_exact(
    t: list[float],
    x: list[float | None],
    op: str,
    *,
    half_life: float,
    window_size: float | None = None,
    closed: str = "right",
    min_samples: int | None = None,
    bias: bool = False,
) -> list[Decimal | None]:
    """Every row's operator from the definition, at ``DIGITS`` digits: the
    rows of one group, ``t`` ascending, ``x`` with ``None`` or NaN for a
    null. ``op`` is ``"ewm_<stat>"`` looking back and ``"rewm_<stat>"``
    looking ahead, as ``tests/test_windows.py::loop`` names them."""
    with localcontext() as ctx:
        ctx.prec = DIGITS
        return _loop(t, x, op, half_life, window_size, closed, min_samples, bias)


def _loop(
    t: list[float],
    x: list[float | None],
    op: str,
    half_life: float,
    window_size: float | None,
    closed: str,
    min_samples: int | None,
    bias: bool,
) -> list[Decimal | None]:
    ln2 = Decimal(2).ln()
    forward = op.startswith("rewm")
    stat = op.split("_")[1]
    if min_samples is None:
        min_samples = 0 if stat == "sum" else 1
    weighs_held = stat in ("mean", "var", "std")
    n = len(t)
    tau = [Decimal(tj) - Decimal(t[0]) for tj in t]
    v = [None if (xj is None or math.isnan(xj)) else Decimal(xj) for xj in x]
    near = closed in ("right", "both") if not forward else closed in ("left", "both")
    far = closed in ("left", "both") if not forward else closed in ("right", "both")
    w = None if window_size is None else Decimal(window_size)
    out: list[Decimal | None] = [None] * n
    for a in range(n):
        ta = tau[a]
        if forward:
            assert w is not None
            members = [
                b
                for b in range(n)
                if (tau[b] - ta >= 0 if near else tau[b] - ta > 0)
                and (tau[b] - ta <= w if far else tau[b] - ta < w)
            ]
            if not any((tau[b] - ta > w) if far else (tau[b] - ta >= w) for b in range(a + 1, n)):
                continue
            span = w
        else:
            members = [
                b
                for b in range(n)
                if ((b <= a or tau[b] == ta) if near else (b < a and tau[b] != ta))
                and (w is None or ((ta - tau[b] <= w) if far else (ta - tau[b] < w)))
            ]
            span = ta if w is None else min(w, ta)
        s = total = Decimal(0)
        count = 0
        held: list[tuple[Decimal, Decimal]] = []
        for b in members:
            xb = v[b]
            if xb is None:
                continue
            count += 1
            tb = tau[b]
            if weighs_held:
                if forward:
                    assert w is not None
                    nxt = next((c for c in range(b + 1, n) if v[c] is not None), None)
                    end = _INF if nxt is None else tau[nxt]
                    m = _between(half_life, True, ta, tb, min(end, ta + w), ln2)
                else:
                    prv = next((c for c in range(b - 1, -1, -1) if v[c] is not None), None)
                    # The first value is held from the window's far edge as
                    # the first row saw it, or from forever with no window.
                    first = tau[0] - w if w is not None else -_INF
                    start = first if prv is None else tau[prv]
                    edge = -_INF if w is None else ta - w
                    m = _between(half_life, False, ta, max(start, edge), tb, ln2)
                s += m * xb
                total += m
                held.append((m, xb))
            else:
                s += _lam_pow(half_life, abs(tb - ta), ln2) * xb
        if count < min_samples:
            continue
        if stat == "mean":
            out[a] = s / total if total > 0 else None
        elif stat in ("var", "std"):
            if not total > 0:
                continue
            mu = s / total
            var = sum((m * (xx - mu) ** 2 for m, xx in held), Decimal(0)) / total
            if not bias:
                # Polars' correction for unequal weights, V1^2 / (V1^2 - V2).
                rest = total * total - sum((m * m for m, _ in held), Decimal(0))
                if not rest > 0:
                    continue
                var *= total * total / rest
            out[a] = var.sqrt() if stat == "std" else var
        elif stat == "sum":
            out[a] = s
        else:
            m = _mass(half_life, span, ln2)
            out[a] = s / m if m > 0 else None
    return out
