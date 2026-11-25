"""STL and MSTL in exact arithmetic, with the most a double evaluation may
differ from it: the reference ``tests/test_stl.py`` holds ``rp.ops.stl`` and
``rp.ops.mstl`` to.

The decomposition is Cleveland, Cleveland, McRae and Terpenning's (1990),
as the authors' Fortran states it and statsmodels translates it, and MSTL is
Bandara, Hyndman and Bergmeir's (2021) composition of it:

.. code-block:: text

    loess at x_s over rows [l, r] of n, span q (positions j = 1..n):
      h   = max(x_s - l, r - x_s), plus floor((q - n) / 2) when q > n
      v_j = 1 if |j - x_s| <= 0.001 h, 0 if |j - x_s| > 0.999 h,
            else (1 - (|j - x_s| / h)^3)^3, times rho_j if robust
      v_j /= sum v_j                 (no value when the sum is 0)
      degree 1, h > 0: abar = sum v_j j, c = sum v_j (j - abar)^2,
            if sqrt(c) > 0.001 (n - 1): v_j *= (x_s - abar) / c (j - abar) + 1
      value = sum v_j y_j
    one inner pass, trend T (0 at first), period p:
      C = each cycle-subseries of Y - T smoothed (span seasonal), one value
          added before and after each: n + 2p rows
      L = loess (span low_pass) of the moving averages of p, p and 3 of C
      S = C[p .. p + n] - L;  T = loess (span trend) of Y - S
    inner passes, outer + 1 times; between them rho = B(|R| / (6 med |R|)),
      R = Y - T - S, B(u) = (1 - u^2)^2, 1 at or below 0.001, 0 above 0.999
    MSTL: seasons by ascending period, iterate times each: add S_i back, STL
      at (p_i, s_i), take its seasonal as S_i and remove it; the trend is the
      last STL's, the remainder the deseasonalised series less it

A jump ``k > 1`` fits every ``k``-th row and interpolates between them on
the line, and fits the last row if the jumps miss it.

**Exact.** Each value is carried in ``mpmath`` at 50 digits, from the
inputs' doubles and the method's own constants as the doubles its code
holds (``0.999 h`` is the double ``0.999 * h``: each evaluation computes it
alike, from an integer ``h``). Fifty digits over the ``10⁵`` operations of
a robust fit leave the exact value's own error near ``10⁻⁴⁵`` relative.
Each branch is taken on the exact value.

**The bound: the computation's own condition.** A double evaluation of the
same operations rounds each result ``r_i`` by ``δ_i``, ``|δ_i| ≤ u_i |r_i|``,
``u_i = u = 2⁻⁵³`` for ``+ − × ÷`` and ``LIBM`` for each ``x³`` through the
platform's ``pow``. To first order its output moves by ``Σ_i (∂out/∂r_i)
δ_i``, so

.. code-block:: text

    |double - exact| <= B = sum_i |d out / d r_i| u_i |r_i|

the rounding of each operation weighted by how much the output depends on
it: the condition number of the computation, read off the run by reverse
differentiation of the recorded operations (Higham, *Accuracy and
Stability of Numerical Algorithms*, 2nd ed., §3.3 and §8.3). A forward
running bound, each operation's worst case added to its operands', grew
150-fold a pass here and stood at ``10¹³ u`` after five: it cannot see that
a pass damps the errors the one before left. ``B`` sees it, since the
derivatives carry the cancellation.

``LIBM`` is six units of ``u``, three ulps: what ``crate::oracle::LIBM``
allows a platform's ``pow`` (Apple's and glibc's are within one ulp, and
``ucrtbase.dll``'s ``expm1`` was measured at three). A fused ``x · y + z``
rounds once where ``B`` counts two roundings, so ``B`` holds with or
without the fusing a compiler may do: statsmodels' build for arm64 fuses
some of these sums, and its x86-64 build and this library fuse none.
``B`` holds for any order-preserving double evaluation on any platform,
so what macOS measures against it Linux and Windows are held to alike. A
double evaluation that took another branch than the exact value takes
would part from it by far more than ``B``, so it fails rather than passes.
The terms first order drops are smaller than ``B`` by ``B``'s own relative
size, ``10⁻¹²`` here, and ``SAFETY`` covers them and ``B``'s own rounding.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import mpmath

U = 2.0**-53
#: Six units of ``u`` a platform's ``pow`` may be off (three ulps).
LIBM = 6 * U
#: The terms a first-order bound drops, and the bound's own rounding.
SAFETY = 1.001

_CTX = mpmath.mp.clone()
_CTX.dps = 50
_mpf = _CTX.mpf
_MPF = type(_mpf(0))


class _Tape:
    """The operations of one fit, in order: each result's operands with
    their partial derivatives, and its rounding's most, ``u_i |r_i|``."""

    def __init__(self) -> None:
        self.parents: list[tuple[tuple[int, float], ...]] = []
        self.rounding: list[float] = []


_TAPE = _Tape()


class X:
    """An exact value, and where the operation that made it sits on the
    tape (``-1``: a double the code holds, no rounding)."""

    __slots__ = ("v", "f", "i")

    def __init__(self, v: object, parents: tuple = (), unit: float = 0.0) -> None:
        self.v = v if isinstance(v, _MPF) else _mpf(v)
        self.f = float(self.v)
        if unit == 0.0:
            self.i = -1
            return
        self.i = len(_TAPE.rounding)
        _TAPE.parents.append(tuple((p.i, d) for p, d in parents if p.i >= 0))
        _TAPE.rounding.append(unit * abs(self.f))

    def __add__(self, o: X) -> X:
        return X(self.v + o.v, ((self, 1.0), (o, 1.0)), U)

    def __sub__(self, o: X) -> X:
        return X(self.v - o.v, ((self, 1.0), (o, -1.0)), U)

    def __mul__(self, o: X) -> X:
        return X(self.v * o.v, ((self, o.f), (o, self.f)), U)

    def __truediv__(self, o: X) -> X:
        return X(self.v / o.v, ((self, 1.0 / o.f), (o, -self.f / (o.f * o.f))), U)

    def cube(self) -> X:
        """``x³`` through the platform's ``pow``."""
        return X(self.v**3, ((self, 3.0 * self.f * self.f),), LIBM)

    def __abs__(self) -> X:
        return _exact_op(abs(self.v), self, -1.0 if self.v < 0 else 1.0)

    def __float__(self) -> float:
        return self.f


def _exact_op(v: object, parent: X, d: float) -> X:
    """An operation no double evaluation rounds (``|x|``, an order
    statistic): on the tape with no rounding, so the derivative passes."""
    x = X(v)
    if parent.i >= 0:
        x.i = len(_TAPE.rounding)
        _TAPE.parents.append(((parent.i, d),))
        _TAPE.rounding.append(0.0)
    return x


def const(x: float) -> X:
    """A double the method's code holds: exact."""
    return X(_mpf(x))


def le(a: X, b: X) -> bool:
    """``a <= b``, on the exact values."""
    return bool(a.v <= b.v)


def bound(x: X) -> float:
    """``B`` for ``x``: each rounding before it weighted by the derivative
    of ``x`` in it, by reverse differentiation of the tape."""
    if x.i < 0:
        return 0.0
    adj = [0.0] * (x.i + 1)
    adj[x.i] = 1.0
    total = 0.0
    parents, rounding = _TAPE.parents, _TAPE.rounding
    for i in range(x.i, -1, -1):
        a = adj[i]
        if a == 0.0:
            continue
        total += abs(a) * rounding[i]
        for p, d in parents[i]:
            adj[p] += a * d
    return total * SAFETY


def _fresh() -> None:
    global _TAPE
    _TAPE = _Tape()


def _est(y, n, len_, ideg, xs, nleft, nright, w, rw):
    rng = float(n - 1)
    h = float(max(xs - nleft, nright - xs))
    if len_ > n:
        h += (len_ - n) // 2
    h9, h1 = const(0.999 * h), const(0.001 * h)
    a = const(0.0)
    for j in range(nleft - 1, nright):
        w[j] = const(0.0)
        r = const(abs(j + 1 - xs))
        if le(r, h9):
            if le(r, h1):
                w[j] = const(1.0)
            else:
                w[j] = (const(1.0) - (r / const(h)).cube()).cube()
            if rw is not None:
                w[j] = w[j] * rw[j]
            a = a + w[j]
    if le(a, const(0.0)):
        return None
    for j in range(nleft - 1, nright):
        w[j] = w[j] / a
    if h > 0 and ideg > 0:
        a = const(0.0)
        for j in range(nleft - 1, nright):
            a = a + w[j] * const(j + 1)
        b = const(xs) - a
        c = const(0.0)
        for j in range(nleft - 1, nright):
            t = const(j + 1) - a
            c = c + w[j] * (t * t)
        if _CTX.sqrt(c.v) > 0.001 * rng:
            b = b / c
            for j in range(nleft - 1, nright):
                w[j] = w[j] * (b * (const(j + 1) - a) + const(1.0))
    ys = const(0.0)
    for j in range(nleft - 1, nright):
        ys = ys + w[j] * y[j]
    return ys


def _ess(y, n, len_, ideg, njump, rw, ys, res):
    if n < 2:
        ys[0] = y[0]
        return
    newnj = min(njump, n - 1)
    nleft, nright = 1, n

    def fit(i, xs):
        v = _est(y, n, len_, ideg, xs, nleft, nright, res, rw)
        ys[i] = y[i] if v is None else v

    if len_ >= n:
        for i in range(0, n, newnj):
            fit(i, i + 1)
    elif newnj == 1:
        nsh = (len_ + 2) // 2
        nright = len_
        for i in range(n):
            if i + 1 > nsh and nright != n:
                nleft += 1
                nright += 1
            fit(i, i + 1)
    else:
        nsh = (len_ + 1) // 2
        for i in range(0, n, newnj):
            if i + 1 < nsh:
                nleft, nright = 1, len_
            elif i + 1 >= n - nsh + 1:
                nleft, nright = n - len_ + 1, n
            else:
                nleft, nright = i + 1 - nsh + 1, len_ + i + 1 - nsh
            fit(i, i + 1)
    if newnj == 1:
        return
    for i in range(0, n - newnj, newnj):
        delta = (ys[i + newnj] - ys[i]) / const(newnj)
        for j in range(i, i + newnj):
            ys[j] = ys[i] + delta * const(j - i)
    k = ((n - 1) // newnj) * newnj + 1
    if k != n:
        fit(n - 1, n)
        if k != n - 1:
            delta = (ys[n - 1] - ys[k - 1]) / const(n - k)
            for j in range(k, n):
                ys[j] = ys[k - 1] + delta * const(j + 1 - k)


def _ma(x, n, len_):
    flen = const(len_)
    v = const(0.0)
    for i in range(len_):
        v = v + x[i]
    out = [v / flen]
    for j in range(1, n - len_ + 1):
        v = v + (x[j + len_ - 1] - x[j - 1])
        out.append(v / flen)
    return out


def _rwts(y, fit):
    r = [abs(a - b) for a, b in zip(y, fit, strict=True)]
    n = len(r)
    order = sorted(r, key=lambda x: x.v)
    m0, m1 = n // 2, n - n // 2 - 1
    cmad = const(3.0) * (order[m0] + order[m1])
    if cmad.v == 0:
        return [const(1.0)] * n
    c9, c1 = const(0.999) * cmad, const(0.001) * cmad
    out = []
    for x in r:
        if le(x, c1):
            out.append(const(1.0))
        elif le(x, c9):
            u = x / cmad
            v = const(1.0) - u * u
            out.append(v * v)
        else:
            out.append(const(0.0))
    return out


def _defaults(period, seasonal, trend, low_pass, robust, inner, outer):
    if trend is None:
        trend = math.ceil(1.5 * period / (1 - 1.5 / seasonal))
        trend += trend % 2 == 0
    if low_pass is None:
        low_pass = period + 1
        low_pass += low_pass % 2 == 0
    if inner is None:
        inner = 2 if robust else 5
    if outer is None:
        outer = 15 if robust else 0
    return trend, low_pass, inner, outer


def stl(
    y: Sequence[float],
    period: int,
    seasonal: int = 7,
    trend: int | None = None,
    low_pass: int | None = None,
    seasonal_deg: int = 1,
    trend_deg: int = 1,
    low_pass_deg: int = 1,
    robust: bool = False,
    seasonal_jump: int = 1,
    trend_jump: int = 1,
    low_pass_jump: int = 1,
    inner_iter: int | None = None,
    outer_iter: int | None = None,
    _ys: list[X] | None = None,
) -> tuple[list[X], list[X], list[X]]:
    """The trend, seasonal and remainder of ``y``, exact, each on this
    fit's tape for :func:`bound`. statsmodels' names and defaults."""
    trend, low_pass, inner, outer = _defaults(
        period, seasonal, trend, low_pass, robust, inner_iter, outer_iter
    )
    if _ys is None:
        _fresh()
    ys = _ys if _ys is not None else [const(v) for v in y]
    n, p = len(ys), period
    tr = [const(0.0)] * n
    se = [const(0.0)] * n
    rho = None
    for run in range(outer + 1):
        for _ in range(inner):
            d = [a - b for a, b in zip(ys, tr, strict=True)]
            c = [const(0.0)] * (n + 2 * p)
            for j in range(p):
                k = (n - (j + 1)) // p + 1
                sub = [d[i * p + j] for i in range(k)]
                sub_rw = None if rho is None else [rho[i * p + j] for i in range(k)]
                fit = [const(0.0)] * (k + 2)
                inner_fit = [const(0.0)] * k
                _ess(sub, k, seasonal, seasonal_deg, seasonal_jump, sub_rw, inner_fit, [None] * k)
                fit[1 : k + 1] = inner_fit
                lo = _est(
                    sub, k, seasonal, seasonal_deg, 0, 1, min(seasonal, k), [None] * k, sub_rw
                )
                fit[0] = fit[1] if lo is None else lo
                left = max(1, k - seasonal + 1)
                hi = _est(sub, k, seasonal, seasonal_deg, k + 1, left, k, [None] * k, sub_rw)
                fit[k + 1] = fit[k] if hi is None else hi
                for m in range(k + 2):
                    c[m * p + j] = fit[m]
            m1 = _ma(c, n + 2 * p, p)
            m2 = _ma(m1, n + p + 1, p)
            m3 = _ma(m2, n + 2, 3)
            low = [const(0.0)] * n
            _ess(m3, n, low_pass, low_pass_deg, low_pass_jump, None, low, [None] * n)
            se = [c[p + i] - low[i] for i in range(n)]
            ds = [a - b for a, b in zip(ys, se, strict=True)]
            tr = [const(0.0)] * n
            _ess(ds, n, trend, trend_deg, trend_jump, rho, tr, [None] * n)
        if run < outer:
            rho = _rwts(ys, [a + b for a, b in zip(tr, se, strict=True)])
    resid = [(a - b) - c for a, b, c in zip(ys, se, tr, strict=True)]
    return tr, se, resid


def mstl(
    y: Sequence[float],
    periods: Sequence[int],
    windows: Sequence[int] | None = None,
    iterate: int = 2,
) -> tuple[list[X], list[list[X]], list[X]]:
    """The trend, each season's seasonal (by ascending period) and the
    remainder of ``y``, exact, each on this fit's tape for :func:`bound`.
    statsmodels' defaults: seasonal spans ``7 + 4 i`` for the ``i``-th season, from 1."""
    if windows is None:
        spans = [7 + 4 * i for i in range(1, len(periods) + 1)]
        seasons = list(zip(sorted(periods), spans, strict=True))
    else:
        seasons = sorted(zip(periods, windows, strict=True))
    n = len(y)
    _fresh()
    passes = 1 if len(seasons) == 1 else iterate
    seasonal = [[const(0.0)] * n for _ in seasons]
    deseas = [const(v) for v in y]
    trend: list[X] = []
    for _ in range(passes):
        for i, (p, s) in enumerate(seasons):
            deseas = [a + b for a, b in zip(deseas, seasonal[i], strict=True)]
            trend, seasonal[i], _ = stl([], p, s, _ys=deseas)
            deseas = [a - b for a, b in zip(deseas, seasonal[i], strict=True)]
    resid = [a - b for a, b in zip(deseas, trend, strict=True)]
    return trend, seasonal, resid


def gap(got: float, want: X) -> float:
    """``|got - want|``, ``got`` a double and ``want`` exact."""
    return float(abs(_mpf(got) - want.v))
