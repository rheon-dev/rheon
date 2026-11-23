"""Reference (oracle) for ``ewm_moments``: exponentially weighted and windowed
moments, cross-moments and lagged cross-moments, in Python floats.

PLAN 554, part 2: the update the Rust state is checked against, written before
it. Not bit-exact with the Rust (it holds the mean whole, where ``EwDiag``
keeps a compensated one, and it folds a window's blocks left to right, where
``window_moments`` aggregates them by van Herk's scheme); the Rust is held to
it within a stated tolerance. It is exact against itself: the same rows give
the same bits however they are chunked, and a saved state continues to the
bit.

The definitions (every weight ``omega_t = w_t * prod_{s>t} f_s``, ``f_s`` the
decay the clock step to row ``s`` gives, ``W = sum omega``)::

    central       mu_a    = sum omega (z - m)^a / W        (a a multi-index over the legs)
    raw           mu'_a   = sum omega z^a / W
    standardized  s_a     = mu_a / prod_j mu_{2 e_j}^(a_j / 2)
    cumulant      kappa_p = mu_p - sum_{j=2}^{p-2} C(p-1, j-1) kappa_j mu_{p-j}   (one leg)

A term's legs are columns, each optionally lagged (``"x@3"``: ``x`` three
learned rows back, or three clock units back on a clock). A term is computed
over the stream of the rows at which every one of its legs has a value, about
that stream's own means: the lagged pairs' own weighted means (Pebay over
pairs), which is numpy's ``cov(x[k:], x[:-k], bias=True)`` and river's
``AutoCorr``, not ``ew_lag_cov``'s recursion (both legs against the single
stream's mean before the row) and not statsmodels' ``acovf`` (both legs
against the full sample's mean).

The update. Decay scales ``W`` alone (the normalised moments
``c_a = M_a / W`` do not move under a uniform rescaling of the weights), and
a row is a merge with the one-row summary ``(w, z, c = 0)``. Two summaries of
disjoint rows, ``A`` then ``B``, merge by Pebay's rule (Pebay, Terriberry,
Kolla and Bennett 2016, "Numerically stable, scalable formulas for parallel
and online computation of higher-order multivariate central moments with
arbitrary weights"), normalised::

    W = W_A + W_B    a = W_A / W    b = W_B / W    delta = m_B - m_A
    m = m_A + b delta
    c_a = sum_{beta <= a} C(a, beta) [ a c^A_{a-beta} (-b delta)^beta
                                       + b c^B_{a-beta} (a delta)^beta ]
    (c_0 = 1, c_{e_j} = 0; C(a, beta) = prod_j C(a_j, beta_j))
    r_k = a^k r^A_k + b^k r^B_k        r_k = sum omega^k / W^k

``r_2`` is ``1 / n``, Kish's effective size, kept always; ``r_3`` and
``r_4`` only where an unbiased third or fourth moment is asked for.

A zero-weight row, or one with a non-finite value or weight in a column the
spec reads, is no row: it advances no clock, decays nothing, enters no lag
buffer and no window. Its output is read from the state as it stands (at its
own clock, for a clock window's edge).
"""

from __future__ import annotations

import copy
import itertools
import math
from dataclasses import dataclass

KINDS = ("central", "raw", "standardized", "cumulant")

#: The highest order the unbiased forms reach: 2 (Kish) and 3, 4 (Rimoldini
#: 2014, for reliability weights). A term above it with ``bias=False`` is
#: refused by name.
MAX_UNBIASED_ORDER = 4

#: A denominator of the unbiased forms within this many epsilons of the sum
#: of its terms' sizes keeps no digit, and the term is withheld.
NO_DIGIT = 64.0 * 2.0**-52

#: The default state limit, as the models' ``max_state_bytes``.
MAX_STATE_BYTES = 256 * 1024

DOUBLE = 8


# --- multi-indices -----------------------------------------------------------


def _order(a: tuple[int, ...]) -> int:
    return sum(a)


def _closure(alphas: set[tuple[int, ...]]) -> list[tuple[int, ...]]:
    """Every multi-index below one of ``alphas`` of total order 2 or more, in
    a fixed order (by total order, then lexicographic): the moments a
    summary keeps. Order 0 is the weight and order 1 the means."""
    out: set[tuple[int, ...]] = set()
    for a in alphas:
        for b in itertools.product(*(range(k + 1) for k in a)):
            if _order(b) >= 2:
                out.add(b)
    return sorted(out, key=lambda b: (_order(b), b))


def _below(a: tuple[int, ...]):
    return itertools.product(*(range(k + 1) for k in a))


def _binom(a: tuple[int, ...], b: tuple[int, ...]) -> int:
    out = 1
    for x, y in zip(a, b, strict=True):
        out *= math.comb(x, y)
    return out


def _sub(a, b):
    return tuple(x - y for x, y in zip(a, b, strict=True))


def _add(a, b):
    return tuple(x + y for x, y in zip(a, b, strict=True))


def _unit(d: int, j: int, k: int = 1) -> tuple[int, ...]:
    return tuple(k if i == j else 0 for i in range(d))


# --- decay -------------------------------------------------------------------


@dataclass(frozen=True)
class Decay:
    """``factor = 2^(-dt / half_life)``, or ``lam^dt``, as ``clock.rs``'s
    ``Decay::factor``."""

    half_life: float | None = None
    lam: float | None = None

    def factor(self, dt: float) -> float:
        if self.lam is not None:
            if dt == 1.0:
                return self.lam
            if dt == 0.0:
                return 1.0
            return self.lam**dt
        h = math.inf if self.half_life is None else self.half_life
        if math.isinf(h):
            return 1.0
        return math.exp2(-(dt / h))


# --- the summary -------------------------------------------------------------


class Layout:
    """The moments a summary over ``d`` legs keeps, and the merge's plan:
    for each kept ``alpha``, the ``(beta, C(alpha, beta), alpha - beta)``
    whose ``c_{alpha - beta}`` is not 0 (order 1 is 0 about the own mean)."""

    def __init__(self, d: int, keep: list[tuple[int, ...]], powers: tuple[int, ...]):
        self.d = d
        self.keep = keep
        self.at = {a: i for i, a in enumerate(keep)}
        self.powers = powers  # 2, 3, 4: which r_k it keeps
        self.top = [max((a[j] for a in keep), default=0) for j in range(d)]
        self.plan = []
        for a in keep:
            row = []
            for b in _below(a):
                g = _sub(a, b)
                if _order(g) == 1:
                    continue
                row.append((b, float(_binom(a, b)), g))
            self.plan.append(row)

    def numbers(self) -> int:
        """Doubles a summary takes: ``W``, ``d`` means, the kept moments and
        the kept ``r_k``."""
        return 1 + self.d + len(self.keep) + len(self.powers)


class Summary:
    """``(W, m, c, r)`` over a layout's legs; ``W = 0`` is empty."""

    __slots__ = ("W", "m", "c", "r")

    def __init__(self, lay: Layout):
        self.W = 0.0
        self.m = [0.0] * lay.d
        self.c = [0.0] * len(lay.keep)
        self.r = [0.0] * len(lay.powers)

    @staticmethod
    def row(lay: Layout, w: float, z: list[float]) -> Summary:
        s = Summary(lay)
        s.W = w
        s.m = list(z)
        s.r = [1.0] * len(lay.powers)
        return s

    def cv(self, lay: Layout, g: tuple[int, ...]) -> float:
        n = _order(g)
        if n == 0:
            return 1.0
        if n == 1:
            return 0.0
        return self.c[lay.at[g]]

    def to_list(self) -> list[float]:
        return [self.W, *self.m, *self.c, *self.r]

    @staticmethod
    def from_list(lay: Layout, v: list[float]) -> Summary:
        s = Summary(lay)
        d, k = lay.d, len(lay.keep)
        s.W = v[0]
        s.m = list(v[1 : 1 + d])
        s.c = list(v[1 + d : 1 + d + k])
        s.r = list(v[1 + d + k :])
        return s


def _powers(x: float, top: int) -> list[float]:
    """``x^0 .. x^top`` by repeated multiplication (no ``pow``), so the Rust
    can form the same numbers."""
    out = [1.0]
    for _ in range(top):
        out.append(out[-1] * x)
    return out


def merge(lay: Layout, A: Summary, B: Summary) -> Summary:
    """Pebay's merge of two summaries of disjoint rows, ``A`` the earlier.
    An empty side gives the other unchanged: ``W = 0`` is guarded by the
    condition, so a zero-weight first row is no 0/0."""
    W = A.W + B.W
    if B.W == 0.0:
        return A
    if A.W == 0.0:
        out = Summary(lay)
        out.W, out.m, out.c, out.r = B.W, list(B.m), list(B.c), list(B.r)
        return out
    a = A.W / W
    b = B.W / W
    delta = [mb - ma for ma, mb in zip(A.m, B.m, strict=True)]
    pa = [_powers(-b * dl, t) for dl, t in zip(delta, lay.top, strict=True)]
    pb = [_powers(a * dl, t) for dl, t in zip(delta, lay.top, strict=True)]
    out = Summary(lay)
    out.W = W
    out.m = [ma + b * dl for ma, dl in zip(A.m, delta, strict=True)]
    for i, row in enumerate(lay.plan):
        s = 0.0
        for beta, coef, g in row:
            qa = 1.0
            qb = 1.0
            for j, k in enumerate(beta):
                qa *= pa[j][k]
                qb *= pb[j][k]
            s += coef * (a * A.cv(lay, g) * qa + b * B.cv(lay, g) * qb)
        out.c[i] = s
    for i, k in enumerate(lay.powers):
        out.r[i] = a**k * A.r[i] + b**k * B.r[i]
    return out


def decayed(s: Summary, f: float) -> Summary:
    """Decay scales the weight alone: ``c`` and ``r`` are ratios of sums
    that every weight's factor scales alike."""
    if f == 1.0 or s.W == 0.0:
        return s
    out = copy.copy(s)
    out.W = s.W * f
    return out


# --- the unbiased forms ------------------------------------------------------


def _e4(r2, r3, r4):
    """``E[m4] = a1 mu4 + a2 mu2^2`` and ``E[m2^2] = b1 mu4 + b2 mu2^2`` for
    the plug-in moments of i.i.d. rows under weights with ``r_k = V_k / V_1^k``
    (derived exactly by enumeration over discrete distributions and fitted in
    ``1, r2, r3, r4, r2^2`` with rational arithmetic; the test
    ``test_unbiased_forms_are_unbiased`` holds them to exact expectations)."""
    a1 = 1.0 - 4.0 * r2 + 6.0 * r3 - 3.0 * r4
    a2 = 6.0 * r2 - 6.0 * r3 + 9.0 * r4 - 9.0 * r2 * r2
    b1 = r2 - 2.0 * r3 + r4
    b2 = 1.0 - 3.0 * r2 + 2.0 * r3 - 3.0 * r4 + 3.0 * r2 * r2
    return a1, a2, b1, b2


def _kept(den: float, *terms: float) -> bool:
    return den > NO_DIGIT * sum(abs(t) for t in terms)


# --- the spec ----------------------------------------------------------------


@dataclass(frozen=True)
class Leg:
    col: str
    lag: float  # 0: the row itself; rows without a clock, clock units with one

    def spelled(self) -> str:
        if self.lag == 0:
            return self.col
        lag = int(self.lag) if float(self.lag).is_integer() else self.lag
        return f"{self.col}@{lag}"


@dataclass(frozen=True)
class Term:
    name: str
    legs: tuple[Leg, ...]
    alpha: tuple[int, ...]
    kind: str
    bias: bool


def _parse_leg(s: str, clocked: bool) -> Leg:
    if "@" not in s:
        return Leg(s, 0.0)
    col, _, lag = s.rpartition("@")
    try:
        k = float(lag)
    except ValueError:
        raise ValueError(f"the leg {s!r} has a lag {lag!r} that is not a number") from None
    if not (k > 0 and math.isfinite(k)):
        raise ValueError(f"the leg {s!r} must lag by more than 0; write {col!r} for the row itself")
    if not clocked and not k.is_integer():
        raise ValueError(f"the leg {s!r} lags by {lag} rows, which is not a whole number")
    return Leg(col, k)


def _default_bias(kind: str, order: int) -> bool:
    """Each statistic's default, as Polars': a variance or covariance is
    unbiased (``ewm_var(bias=False)``), skew and kurtosis are not
    (``skew(bias=True)``, ``kurtosis(bias=True)``), and the rest follow
    the plug-in."""
    return not (kind in ("central", "cumulant") and order == 2)


@dataclass
class Out:
    value: float | None
    reason: str | None


class EwmMoments:
    """The online moments of a stream; see the module docstring.

    ``terms`` maps an output name to its legs' powers, ``{"skew": {"x": 3},
    "xy2": {"x": 1, "y": 2}, "ac1": {"x": 1, "x@1": 1}}``. ``kind`` is one
    kind for every term or a map from a term's name to its kind (the rest
    ``"central"``). ``bias`` is ``None`` for each statistic's default.
    ``se`` adds ``se_<term>``, the delta-method standard error at Kish's
    effective size, for which the summaries keep moments to twice the
    term's order. ``window_rows`` or ``window_clock`` reads a window of the
    latest learned rows, or of the rows whose clock is in ``(t - window, t]``,
    from merged block summaries of ``block_rows`` rows.
    """

    def __init__(
        self,
        terms: dict[str, dict[str, int]],
        *,
        kind: str | dict[str, str] = "central",
        bias: bool | None = None,
        se: bool = False,
        half_life: float | None = None,
        lam: float | None = None,
        clocked: bool = False,
        window_rows: int | None = None,
        window_clock: float | None = None,
        block_rows: int = 4,
        emit: str = "after",
        max_state_bytes: int = MAX_STATE_BYTES,
    ):
        if not terms:
            raise ValueError("terms must name at least one term")
        if half_life is not None and lam is not None:
            raise ValueError("give half_life or lam, not both")
        if window_rows is not None and window_clock is not None:
            raise ValueError("give window_rows or window_clock, not both")
        if window_clock is not None and not clocked:
            raise ValueError("window_clock needs a clock")
        if emit not in ("after", "before"):
            raise ValueError(f"emit must be 'after' or 'before', not {emit!r}")
        self.decay = Decay(half_life, lam)
        self.clocked = clocked
        self.window_rows = window_rows
        self.window_clock = window_clock
        self.block_rows = block_rows
        self.emit = emit
        self.se = se
        self.max_state_bytes = max_state_bytes
        kinds = kind if isinstance(kind, dict) else {n: kind for n in terms}
        self.terms: list[Term] = []
        for name, powers in terms.items():
            k = kinds.get(name, "central")
            if k not in KINDS:
                raise ValueError(f"the term {name!r} has kind {k!r}; the kinds are {KINDS}")
            if not powers:
                raise ValueError(f"the term {name!r} names no leg")
            legs = []
            for leg, p in powers.items():
                if not (isinstance(p, int) and p >= 1):
                    raise ValueError(
                        f"the term {name!r} raises {leg!r} to {p!r}; "
                        "a power is a whole number from 1"
                    )
                legs.append((_parse_leg(leg, clocked), p))
            legs.sort(key=lambda lp: (lp[0].col, lp[0].lag))
            alpha = tuple(p for _, p in legs)
            order = sum(alpha)
            if k == "cumulant" and len(legs) > 1:
                raise ValueError(
                    f"the term {name!r} asks for a cumulant of {len(legs)} legs; "
                    "a cumulant is of one leg"
                )
            if k == "standardized" and order < 2:
                raise ValueError(
                    f"the term {name!r} is standardized at order 1, which is 0 by definition"
                )
            b = _default_bias(k, order) if bias is None else bias
            if not b and k != "raw" and order > MAX_UNBIASED_ORDER:
                raise ValueError(
                    f"the term {name!r} asks for bias=False at order {order}; "
                    "the unbiased forms stop "
                    f"at order {MAX_UNBIASED_ORDER}. Give bias=True for it."
                )
            self.terms.append(Term(name, tuple(lg for lg, _ in legs), alpha, k, b))
        self._build()
        fixed = self.fixed_state_bytes()
        if fixed > max_state_bytes:
            raise ValueError(
                f"the state takes {fixed} bytes before the first row, over max_state_bytes "
                f"({max_state_bytes}): fewer terms, lower orders, shallower lags "
                "or a shorter window"
            )
        self.reset()

    # -- layout ----------------------------------------------------------------

    def _build(self) -> None:
        groups: dict[tuple[Leg, ...], list[Term]] = {}
        for t in self.terms:
            groups.setdefault(t.legs, []).append(t)
        self.groups = list(groups)
        self.layouts: dict[tuple[Leg, ...], Layout] = {}
        for legs, ts in groups.items():
            d = len(legs)
            need: set[tuple[int, ...]] = set()
            powers: set[int] = {2}
            for t in ts:
                order = _order(t.alpha)
                supp = {t.alpha}
                if t.kind == "standardized":
                    supp |= {_unit(d, j, 2) for j in range(d)}
                if not t.bias and t.kind != "raw":
                    powers |= {k for k in (3, 4) if k <= order}
                    if order == 4:
                        supp |= {_unit(d, j, 2) for j in range(d)} | self._pairs(t.alpha)
                if self.se:
                    sup = self._if_support(t)
                    supp |= {_add(x, y) for x in sup for y in sup}
                need |= supp
            self.layouts[legs] = Layout(d, _closure(need), tuple(sorted(powers)))
        self.lags: dict[str, list[float]] = {}
        for legs in self.groups:
            for lg in legs:
                if lg.lag > 0:
                    self.lags.setdefault(lg.col, [])
                    if lg.lag not in self.lags[lg.col]:
                        self.lags[lg.col].append(lg.lag)
        self.columns = sorted({lg.col for legs in self.groups for lg in legs})

    @staticmethod
    def _pairs(alpha: tuple[int, ...]) -> set[tuple[int, ...]]:
        """The second-order indices a fourth-order ``alpha``'s polarised
        ``mu2^2`` reads."""
        d = len(alpha)
        idx = [j for j in range(d) for _ in range(alpha[j])]
        return {_add(_unit(d, i), _unit(d, j)) for i in idx for j in idx}

    def _if_support(self, t: Term) -> set[tuple[int, ...]]:
        d = len(t.legs)
        sup = {t.alpha, (0,) * d} | {_unit(d, j) for j in range(d)}
        sup |= set(_below(t.alpha))
        if t.kind == "standardized":
            sup |= {_unit(d, j, 2) for j in range(d)}
        return sup

    # -- state -----------------------------------------------------------------

    def reset(self) -> None:
        self.sums = {legs: Summary(self.layouts[legs]) for legs in self.groups}
        self.buf: dict[str, list[tuple[float, float]]] = {c: [] for c in self.lags}
        self.t_last: float | None = None
        self.n_learned = 0  # the row clock, and the window's row numbers
        self.ring: list[tuple[int, float, float, list[float | None]]] = []
        self._blocks: dict[tuple[tuple[Leg, ...], int], Summary] = {}

    def _leg_list(self) -> list[Leg]:
        return sorted({lg for legs in self.groups for lg in legs}, key=lambda lg: (lg.col, lg.lag))

    def fixed_state_bytes(self) -> int:
        """Bytes the state takes whatever the stream: the summaries, the
        row lags' buffers, a row window's ring and its block summaries. A
        clock lag's buffer and a clock window's ring hold the rows inside a
        span of the clock, whose count the data sets; ``state_bytes`` counts
        them as they stand and the run is refused when they pass the limit."""
        n = 3  # t_last, n_learned, the ring's start
        if self.window_rows is None and self.window_clock is None:
            n += sum(self.layouts[g].numbers() for g in self.groups)
        if not self.clocked:
            n += sum(int(max(ls)) for ls in self.lags.values())
        if self.window_rows is not None:
            n += self.window_rows * self._ring_width()
            nb = self.window_rows // self.block_rows + 2
            n += nb * sum(self.layouts[g].numbers() for g in self.groups)
        return DOUBLE * n

    def _ring_width(self) -> int:
        return 3 + len(self._leg_list())  # row number, clock, weight, each leg

    def state_bytes(self) -> int:
        n = 3
        if self.window_rows is None and self.window_clock is None:
            n += sum(self.layouts[g].numbers() for g in self.groups)
        n += sum((2 if self.clocked else 1) * len(b) for b in self.buf.values())
        n += len(self.ring) * self._ring_width()
        n += sum(self.layouts[g].numbers() for g, _ in self._blocks)
        return DOUBLE * n

    def state_dict(self) -> dict:
        """The state as plain numbers and lists (JSON-safe, NaN included).
        The window's block summaries are not state: they are rebuilt from
        the ring, which gives the same bits."""
        return {
            "sums": [self.sums[g].to_list() for g in self.groups],
            "buf": {c: [list(e) for e in b] for c, b in self.buf.items()},
            "t_last": self.t_last,
            "n_learned": self.n_learned,
            "ring": [[g, t, w, list(v)] for g, t, w, v in self.ring],
        }

    def load_state(self, s: dict) -> None:
        self.reset()
        self.sums = {
            g: Summary.from_list(self.layouts[g], v)
            for g, v in zip(self.groups, s["sums"], strict=True)
        }
        self.buf = {c: [tuple(e) for e in b] for c, b in s["buf"].items()}
        self.t_last = s["t_last"]
        self.n_learned = s["n_learned"]
        self.ring = [(g, t, w, list(v)) for g, t, w, v in s["ring"]]

    # -- the row ---------------------------------------------------------------

    def _partner(self, col: str, lag: float, t: float) -> float | None:
        """The value of ``col`` ``lag`` back: the ``lag``-th learned row
        back, or on a clock the latest learned row at or before ``t - lag``
        (as-of), with no pair when that row is more than ``lag`` before
        ``t - lag`` (a gap longer than the lag), or when none is kept."""
        b = self.buf[col]
        if not self.clocked:
            k = int(lag)
            return b[-k][1] if len(b) >= k else None
        edge = t - lag
        for tt, v in reversed(b):
            if tt <= edge:
                return v if edge - tt <= lag else None
        return None

    def _push(self, col: str, t: float, v: float) -> None:
        b = self.buf[col]
        b.append((t, v))
        if not self.clocked:
            k = int(max(self.lags[col]))
            del b[: max(0, len(b) - k)]
            return
        # Keep the latest row at or before t - K (a future as-of may still
        # land on it) and everything after; drop anything older than t - 2K,
        # which no lag up to K can pair with.
        K = max(self.lags[col])
        edge = t - K
        keep_from = 0
        for i in range(len(b) - 1, -1, -1):
            if b[i][0] <= edge:
                keep_from = i
                break
        while keep_from < len(b) and b[keep_from][0] < t - 2 * K:
            keep_from += 1
        del b[:keep_from]

    def _learn(self, t: float, w: float, row: dict[str, float]) -> None:
        f = 1.0 if self.t_last is None else self.decay.factor(t - self.t_last)
        vals: dict[Leg, float | None] = {}
        for lg in self._leg_list():
            vals[lg] = row[lg.col] if lg.lag == 0 else self._partner(lg.col, lg.lag, t)
        for col in self.lags:
            self._push(col, t, row[col])
        if self.window_rows is None and self.window_clock is None:
            for g in self.groups:
                lay = self.layouts[g]
                s = decayed(self.sums[g], f)
                z = [vals[lg] for lg in g]
                if all(v is not None for v in z):
                    s = merge(lay, s, Summary.row(lay, w, z))  # type: ignore[arg-type]
                self.sums[g] = s
        else:
            self.ring.append((self.n_learned, t, w, [vals[lg] for lg in self._leg_list()]))
            self._evict(t)
        self.t_last = t
        self.n_learned += 1

    def _evict(self, t: float) -> None:
        if self.window_rows is not None:
            drop = max(0, len(self.ring) - self.window_rows)
        else:
            drop = 0
            while drop < len(self.ring) and not (self.ring[drop][1] > t - self.window_clock):
                drop += 1
        del self.ring[:drop]
        if self.ring:
            first_block = self.ring[0][0] // self.block_rows
            for key in [k for k in self._blocks if k[1] < first_block]:
                del self._blocks[key]

    # -- the window ------------------------------------------------------------

    def _fold(self, g, rows) -> tuple[Summary, float | None]:
        """The summary of ``rows`` (ring entries, oldest first), decayed to
        the last one's clock; and that clock."""
        lay = self.layouts[g]
        legs = self._leg_list()
        pos = [legs.index(lg) for lg in g]
        s = Summary(lay)
        t_prev = None
        for _, t, w, v in rows:
            if t_prev is not None:
                s = decayed(s, self.decay.factor(t - t_prev))
            z = [v[p] for p in pos]
            if all(x is not None for x in z):
                s = merge(lay, s, Summary.row(lay, w, z))
            t_prev = t
        return s, t_prev

    def _window(self, g, t_read: float) -> Summary:
        """``head (+) full blocks (+) tail``, left to right, each block decayed
        to the next part's end: the window's summary as a function of its
        rows' numbers alone, so the same rows give the same bits."""
        lay = self.layouts[g]
        rows = self.ring
        if self.window_clock is not None:
            rows = [r for r in rows if r[1] > t_read - self.window_clock]
        if not rows:
            return Summary(lay)
        b = self.block_rows
        g0, g1 = rows[0][0], rows[-1][0]
        by = {r[0]: r for r in rows}
        parts: list[tuple[Summary, float | None]] = []
        n = g0
        while n <= g1:
            j = n // b
            lo, hi = j * b, min((j + 1) * b - 1, g1)
            if n == lo and hi == (j + 1) * b - 1:
                key = (g, j)
                if key not in self._blocks:
                    self._blocks[key] = self._fold(g, [by[i] for i in range(lo, hi + 1)])[0]
                parts.append((self._blocks[key], by[hi][1]))
            else:
                parts.append(self._fold(g, [by[i] for i in range(n, hi + 1)]))
            n = hi + 1
        s, t_end = parts[0]
        for p, tp in parts[1:]:
            s = merge(lay, decayed(s, self.decay.factor(tp - t_end)), p)
            t_end = tp
        return s

    # -- reading ---------------------------------------------------------------

    def _summary(self, g, t: float) -> Summary:
        if self.window_rows is None and self.window_clock is None:
            return self.sums[g]
        return self._window(g, t)

    def read(self, t: float) -> dict[str, Out]:
        out: dict[str, Out] = {}
        for term in self.terms:
            lay = self.layouts[term.legs]
            s = self._summary(term.legs, t)
            out[term.name] = self._value(lay, s, term)
            if self.se:
                out[f"se_{term.name}"] = self._se(lay, s, term)
        return out

    def _value(self, lay: Layout, s: Summary, t: Term) -> Out:
        if s.W == 0.0:
            lagged = any(lg.lag > 0 for lg in t.legs)
            return Out(None, "no lagged pair yet" if lagged else "no weight yet")
        val, why = _evaluate(lay, s, t)
        if why is not None:
            return Out(None, why)
        if not math.isfinite(val):
            return Out(None, "the moment overflowed a double")
        return Out(val, None)

    def _se(self, lay: Layout, s: Summary, t: Term) -> Out:
        if s.W == 0.0:
            return Out(None, "no weight yet")
        val, why = _evaluate(lay, s, t)
        if why is not None:
            return Out(None, why)
        poly = _influence(lay, s, t)
        var = 0.0
        items = sorted(poly.items())
        for a, ca in items:
            for b, cb in items:
                var += ca * cb * s.cv(lay, _add(a, b))
        var *= s.r[0]  # 1 / Kish's n
        if not (var > 0.0 and math.isfinite(var)):
            return Out(None, "the standard error keeps no digit")
        return Out(math.sqrt(var), None)

    # -- the stream ------------------------------------------------------------

    def feed(self, chunk: dict[str, list[float]]) -> dict[str, list[Out]]:
        """One chunk of rows: the columns, ``"w"`` (default 1), ``"t"`` (the
        clock, when clocked) and ``"brk"`` (True where a session starts:
        the lag buffers are cleared). Each row's output is read after its
        update (``emit="after"``, a window operator's convention) or before
        it (``"before"``, a model's)."""
        n = len(next(iter(chunk.values())))
        outs: dict[str, list[Out]] = {}
        for i in range(n):
            w = float(chunk["w"][i]) if "w" in chunk else 1.0
            row = {c: float(chunk[c][i]) for c in self.columns}
            if self.clocked:
                t = float(chunk["t"][i])
                if self.t_last is not None and t < self.t_last:
                    raise ValueError(f"the clock goes back, from {self.t_last} to {t}")
            if "brk" in chunk and chunk["brk"][i]:
                for b in self.buf.values():
                    b.clear()
            learn = w > 0.0 and math.isfinite(w) and all(math.isfinite(v) for v in row.values())
            if not self.clocked:
                t = float(self.n_learned)
            if self.emit == "before" or not learn:
                o = self.read(t)
            if learn:
                self._learn(t, w, row)
                if self.state_bytes() > self.max_state_bytes:
                    raise ValueError(
                        f"the state takes {self.state_bytes()} bytes, over max_state_bytes "
                        f"({self.max_state_bytes}): the clock span of a lag or window "
                        "holds more rows "
                        "than the limit allows"
                    )
                if self.emit == "after":
                    o = self.read(t)
            for k, v in o.items():
                outs.setdefault(k, []).append(v)
        return outs


# --- evaluation --------------------------------------------------------------


def _central(lay: Layout, s: Summary, a: tuple[int, ...]) -> float:
    return s.cv(lay, a)


def _polar22(lay: Layout, s: Summary, a: tuple[int, ...], cov) -> float:
    """The fourth-order ``alpha``'s polarised ``mu2^2``:
    ``(c_ij c_kl + c_ik c_jl + c_il c_jk) / 3`` over its four leg indices."""
    d = len(a)
    i, j, k, h = [x for x in range(d) for _ in range(a[x])]
    return (cov(i, j) * cov(k, h) + cov(i, k) * cov(j, h) + cov(i, h) * cov(j, k)) / 3.0


def _cov(lay: Layout, s: Summary, d: int):
    def cov(i: int, j: int) -> float:
        return s.cv(lay, _add(_unit(d, i), _unit(d, j)))

    return cov


def _unbiased(
    lay: Layout, s: Summary, a: tuple[int, ...]
) -> tuple[float | None, str | None, float]:
    """The unbiased central moment of order 2 to 4 for reliability weights
    (Kish at 2, Rimoldini 2014 at 3 and 4, polarised for cross terms), and
    the unbiased ``mu2^2`` beside a fourth order; or the reason it is not
    defined."""
    order = _order(a)
    r = dict(zip(lay.powers, s.r, strict=True))
    r2 = r[2]
    c = s.cv(lay, a)
    if order == 2:
        den = 1.0 - r2
        if not _kept(den, 1.0, r2):
            return None, "one row carries all the weight, so the unbiased form is not defined", 0.0
        return c / den, None, 0.0
    if order == 3:
        r3 = r[3]
        den = 1.0 - 3.0 * r2 + 2.0 * r3
        if not _kept(den, 1.0, 3.0 * r2, 2.0 * r3):
            return None, "the weight sits on too few rows for an unbiased third moment", 0.0
        return c / den, None, 0.0
    r3, r4 = r[3], r[4]
    a1, a2, b1, b2 = _e4(r2, r3, r4)
    det = a1 * b2 - a2 * b1
    if not _kept(det, a1 * b2, a2 * b1):
        return None, "the weight sits on too few rows for an unbiased fourth moment", 0.0
    p22 = _polar22(lay, s, a, _cov(lay, s, len(a)))
    m4 = (b2 * c - a2 * p22) / det
    u22 = (a1 * p22 - b1 * c) / det
    return m4, None, u22


def _cumulant_poly(p: int) -> dict[tuple[int, ...], float]:
    """``kappa_p`` as a polynomial in the central moments: monomial (the
    sorted orders of its factors) to coefficient."""
    k: dict[int, dict[tuple[int, ...], float]] = {2: {(2,): 1.0}, 3: {(3,): 1.0}}
    for n in range(4, p + 1):
        poly = {(n,): 1.0}
        for j in range(2, n - 1):
            c = math.comb(n - 1, j - 1)
            for mono, v in k[j].items():
                key = tuple(sorted((*mono, n - j)))
                poly[key] = poly.get(key, 0.0) - c * v
        k[n] = poly
    return k[p]


def _evaluate(lay: Layout, s: Summary, t: Term) -> tuple[float, str | None]:
    a = t.alpha
    d = len(a)
    order = _order(a)
    if t.kind == "raw":
        val = 0.0
        for b in _below(a):
            m = 1.0
            for j in range(d):
                m *= s.m[j] ** (a[j] - b[j])
            val += _binom(a, b) * m * s.cv(lay, b)
        return val, None
    if t.kind == "central":
        if order == 1:
            return 0.0, None
        if t.bias:
            return _central(lay, s, a), None
        v, why, _ = _unbiased(lay, s, a)
        return (v if v is not None else 0.0), why
    if t.kind == "standardized":
        var = [s.cv(lay, _unit(d, j, 2)) for j in range(d)]
        for j, v in enumerate(var):
            if not (v > 0.0 and math.isfinite(v)):
                return 0.0, f"the spread of {t.legs[j].spelled()} is zero"
        if t.bias or order == 2:
            num = s.cv(lay, a)
            den = 1.0
            for j in range(d):
                den *= math.sqrt(var[j]) ** a[j]
            return num / den, None
        uvar = []
        for j in range(d):
            v, why, _ = _unbiased(lay, s, _unit(d, j, 2))
            if why is not None:
                return 0.0, why
            uvar.append(v)
        num, why, u22 = _unbiased(lay, s, a)
        if why is not None:
            return 0.0, why
        if order == 4:
            # Fisher's k4 over the unbiased variances, plus the 3 of the
            # normal: G2 + 3 at equal weights, as scipy's kurtosis(bias=False).
            def ucov(i: int, j: int) -> float:
                return _cov(lay, s, d)(i, j) / (1.0 - dict(zip(lay.powers, s.r, strict=True))[2])

            num = num - 3.0 * u22 + 3.0 * _polar22(lay, s, a, ucov)
        den = 1.0
        for j in range(d):
            den *= math.sqrt(uvar[j]) ** a[j]
        return num / den, None
    # cumulant, one leg
    p = a[0]
    if p == 1:
        return s.m[0], None
    if not t.bias:
        v, why, u22 = _unbiased(lay, s, a)
        if why is not None:
            return 0.0, why
        return (v - 3.0 * u22 if p == 4 else v), None
    val = 0.0
    for mono, coef in _cumulant_poly(p).items():
        term = coef
        for q in mono:
            term *= s.cv(lay, (q,))
        val += term
    return val, None


def _poly_add(x: dict, y: dict, f: float = 1.0) -> dict:
    out = dict(x)
    for k, v in y.items():
        out[k] = out.get(k, 0.0) + f * v
    return out


def _if_central(lay: Layout, s: Summary, a: tuple[int, ...]) -> dict:
    """The influence of the central moment ``mu_a`` as a polynomial in the
    legs' deviations ``d``: ``d^a - mu_a - sum_j a_j mu_{a - e_j} d_j``."""
    d = len(a)
    poly = {a: 1.0}
    z = (0,) * d
    poly[z] = poly.get(z, 0.0) - s.cv(lay, a)
    for j in range(d):
        if a[j] > 0:
            e = _unit(d, j)
            poly[e] = poly.get(e, 0.0) - a[j] * s.cv(lay, _sub(a, e))
    return poly


def _influence(lay: Layout, s: Summary, t: Term) -> dict:
    """The term's influence function, a polynomial in the deviations, whose
    mean square over the summary is ``n`` times the delta-method variance."""
    a = t.alpha
    d = len(a)
    if t.kind == "central":
        return _if_central(lay, s, a)
    if t.kind == "raw":
        # z^a - mu'_a with z = m + d, expanded.
        poly: dict = {}
        for b in _below(a):
            m = 1.0
            for j in range(d):
                m *= s.m[j] ** (a[j] - b[j])
            poly[b] = poly.get(b, 0.0) + _binom(a, b) * m
        val, _ = _evaluate(lay, s, t)
        z = (0,) * d
        poly[z] = poly.get(z, 0.0) - val
        return poly
    if t.kind == "standardized":
        # s_a = mu_a / D, D = prod_j mu_{2e_j}^(a_j/2):
        # IF(s_a) = IF(mu_a) / D - s_a sum_j (a_j / 2) IF(mu_{2e_j}) / mu_{2e_j}.
        var = [s.cv(lay, _unit(d, j, 2)) for j in range(d)]
        den = 1.0
        for j in range(d):
            den *= math.sqrt(var[j]) ** a[j]
        val = s.cv(lay, a) / den
        poly = {k: v / den for k, v in _if_central(lay, s, a).items()}
        for j in range(d):
            if a[j] > 0:
                poly = _poly_add(
                    poly, _if_central(lay, s, _unit(d, j, 2)), -val * a[j] / (2.0 * var[j])
                )
        return poly
    p = a[0]
    if p == 1:
        return {(1,): 1.0}
    out: dict = {}
    for mono, coef in _cumulant_poly(p).items():
        for i, q in enumerate(mono):
            g = coef
            for k2, q2 in enumerate(mono):
                if k2 != i:
                    g *= s.cv(lay, (q2,))
            out = _poly_add(out, _if_central(lay, s, (q,)), g)
    return out


def run(
    spec: EwmMoments, data: dict[str, list[float]], chunks: list[int] | None = None
) -> dict[str, list[Out]]:
    """Feed ``data`` in chunks of the given sizes (one chunk by default)."""
    n = len(next(iter(data.values())))
    sizes = chunks or [n]
    assert sum(sizes) == n
    outs: dict[str, list[Out]] = {}
    i = 0
    for k in sizes:
        part = {c: v[i : i + k] for c, v in data.items()}
        for name, o in spec.feed(part).items():
            outs.setdefault(name, []).extend(o)
        i += k
    return outs
