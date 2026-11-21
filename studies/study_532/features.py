"""Resumable time-series features held in a state (PLAN task 532, the study).

Each feature is a small state machine fed one row at a time: ``step(x, t)``
returns the feature's value for that row, computed from the state as it stands
after the row has entered it, and ``to_bytes`` / ``from_bytes`` save and load
the state between any two rows. Every update mirrors, operation for operation,
the kernel Polars 2.0.0 runs for the matching expression, so the outputs agree
to the bit (the sources read are named beside each class). Python floats are
IEEE binary64 and Python does not fuse a multiply and an add, as Rust does not,
so one Python operation is one Rust operation.

Conventions, applied identically by every feature:

- ``x is None`` is a null. A NaN is a value, not a null (as in Polars).
- A zero-weight row (rule 9: it advances the clock and learns nothing) is fed
  as ``None``: it takes its slot on the row clock and its time on the time
  clock, and no value of it enters any state. The caller masks it; see
  ``runs.masked``.
- The time clock is an int64 that must rise strictly within a group; a step
  back or a repeat raises :class:`ClockError` (the reason is in the record).
"""

from __future__ import annotations

import math
import struct
from collections import deque
from dataclasses import dataclass
from typing import Callable, Iterable

Value = float | None


class ClockError(ValueError):
    """A time clock that does not rise strictly within its group."""


# ---------------------------------------------------------------------------
# Serialization helpers: little-endian, fixed widths, so a state's byte count
# is a formula in the window and the columns.


def _pack_values(vals: list[Value]) -> bytes:
    """A list of optional f64: u32 count, f64 each (0.0 for a null), then a
    validity bitmap of ceil(n/8) bytes."""
    n = len(vals)
    body = struct.pack(f"<I{n}d", n, *[0.0 if v is None else v for v in vals])
    bits = bytearray((n + 7) // 8)
    for i, v in enumerate(vals):
        if v is not None:
            bits[i >> 3] |= 1 << (i & 7)
    return body + bytes(bits)


def _unpack_values(buf: bytes, off: int) -> tuple[list[Value], int]:
    (n,) = struct.unpack_from("<I", buf, off)
    off += 4
    raw = struct.unpack_from(f"<{n}d", buf, off)
    off += 8 * n
    nb = (n + 7) // 8
    bits = buf[off : off + nb]
    off += nb
    vals: list[Value] = [raw[i] if bits[i >> 3] >> (i & 7) & 1 else None for i in range(n)]
    return vals, off


def _pack_i64s(xs: list[int]) -> bytes:
    return struct.pack(f"<I{len(xs)}q", len(xs), *xs)


def _unpack_i64s(buf: bytes, off: int) -> tuple[list[int], int]:
    (n,) = struct.unpack_from("<I", buf, off)
    off += 4
    xs = list(struct.unpack_from(f"<{n}q", buf, off))
    return xs, off + 8 * n


# ---------------------------------------------------------------------------
# Lags: shift, diff, pct_change. Polars: ``Series::shift``, ``diff`` is
# ``s - s.shift(n)``, ``pct_change`` is ``diff(s, n) / s.shift(n)``
# (polars-ops/src/series/ops/pct_change.rs). State: the last n values.


class Lag:
    """``kind`` is ``shift``, ``diff`` or ``pct_change``; ``n >= 1``."""

    def __init__(self, kind: str, n: int) -> None:
        assert kind in ("shift", "diff", "pct_change") and n >= 1
        self.kind, self.n = kind, n
        self.ring: deque[Value] = deque(maxlen=n)
        self.seen = 0  # rows seen, capped at n (whether the ring is full)

    def step(self, x: Value, t: int | None = None) -> Value:
        full = self.seen >= self.n
        back = self.ring[0] if full else None
        self.ring.append(x)
        if not full:
            self.seen += 1
        if self.kind == "shift":
            return back
        if x is None or back is None:
            return None
        d = x - back
        return d if self.kind == "diff" else d / back

    def to_bytes(self) -> bytes:
        return struct.pack("<I", self.seen) + _pack_values(list(self.ring))

    @classmethod
    def from_bytes(cls, kind: str, n: int, data: bytes) -> Lag:
        out = cls(kind, n)
        (out.seen,) = struct.unpack_from("<I", data, 0)
        vals, _ = _unpack_values(data, 4)
        out.ring.extend(vals)
        return out


# ---------------------------------------------------------------------------
# Window aggregators. Each mirrors a Polars kernel's ``update(new_start,
# new_end)`` over absolute row indices, reading values through ``get(idx)``.

Getter = Callable[[int], Value]


class KahanSum:
    """polars-compute/src/rolling/sum.rs, ``SumWindow<f64, f64>`` (nulls
    variant; the no-nulls variant is the same with no nulls). Subtractions of
    the rows leaving come first, then additions of the rows entering, each
    compensated with its own error term; a non-finite value is counted, not
    summed. Every row of the previous window leaving resets the sum."""

    __slots__ = ("s", "ea", "es", "nonfinite", "pinf", "ninf", "nulls", "start", "end")

    def __init__(self) -> None:
        self.start = self.end = 0
        self.reset()

    def reset(self) -> None:
        self.s = self.ea = self.es = 0.0
        self.nonfinite = self.pinf = self.ninf = self.nulls = 0

    def _add(self, v: float) -> None:
        if math.isfinite(v):
            y = v - self.ea
            ns = self.s + y
            self.ea = (ns - self.s) - y
            self.s = ns
        else:
            self.nonfinite += 1
            self.pinf += v > 0.0
            self.ninf += v < 0.0

    def _sub(self, v: float) -> None:
        if math.isfinite(v):
            y = (0.0 - v) - self.es
            ns = self.s + y
            self.es = (ns - self.s) - y
            self.s = ns
        else:
            self.nonfinite -= 1
            self.pinf -= v > 0.0
            self.ninf -= v < 0.0

    def update(self, get: Getter, s: int, e: int) -> None:
        if s >= self.end:
            self.reset()
            self.start = self.end = s
        for i in range(self.start, s):
            v = get(i)
            if v is None:
                self.nulls -= 1
            else:
                self._sub(v)
        for i in range(self.end, e):
            v = get(i)
            if v is None:
                self.nulls += 1
            else:
                self._add(v)
        self.start, self.end = s, e

    def total(self) -> float:
        if self.nonfinite == 0:
            return self.s
        if self.nonfinite == self.pinf:
            return math.inf
        if self.nonfinite == self.ninf:
            return -math.inf
        return math.nan

    def count(self) -> int:
        return self.end - self.start - self.nulls

    def get_sum(self) -> Value:
        return self.total()

    def get_mean(self) -> Value:
        # rolling/mean.rs: the sum over the count of non-null rows.
        n = self.count()
        return None if n == 0 else self.total() / float(n)

    # State beyond the ring: the three compensated terms. The counts are
    # recomputed from the ring at load.
    def extra_bytes(self) -> bytes:
        return struct.pack("<3d", self.s, self.ea, self.es)

    def load_extra(self, data: bytes, off: int, get: Getter) -> int:
        self.s, self.ea, self.es = struct.unpack_from("<3d", data, off)
        for i in range(self.start, self.end):
            v = get(i)
            if v is None:
                self.nulls += 1
            elif not math.isfinite(v):
                self.nonfinite += 1
                self.pinf += v > 0.0
                self.ninf += v < 0.0
        return off + 24


def _insert_one(st: tuple[float, float, float], x: float) -> tuple[float, float, float]:
    """polars-compute/src/moment.rs, ``VarState::insert_one``."""
    w, m, dp = st
    nw = w + 1.0
    d = x - m
    nm = m + d / nw
    dp = dp + (x - nm) * d
    if nw == 0.0:
        return (nw, 0.0, 0.0)
    return (nw, nm, dp)


def _combine(st: tuple[float, float, float], o: tuple[float, float, float]) -> tuple[float, float, float]:
    """``VarState::combine``."""
    ow, om, odp = o
    if ow == 0.0:
        return st
    w, m, dp = st
    nw = w + ow
    f = ow / nw
    d = om - m
    nm = m + d * f
    dp = dp + (odp + ow * (om - nm) * d)
    if nw == 0.0:
        return (nw, 0.0, 0.0)
    return (nw, nm, dp)


_ZERO = (0.0, 0.0, 0.0)


class SwagVar:
    """polars-compute/src/rolling/moment.rs, ``MomentWindow<_, VarianceMoment>``:
    a two-stack queue. ``back`` holds the values pushed since the last flip and
    their running ``VarState``; ``front`` holds, for the older values, the
    ``VarState`` of each suffix, built when ``back`` is flipped. A pop takes
    the top of ``front``. The answer combines ``agg_back`` with the top of
    ``front``. A non-finite value is pushed as 0.0 and counted."""

    __slots__ = ("ddof", "front", "back", "agg_back", "nonfinite", "nulls", "start", "end")

    def __init__(self, ddof: int = 1) -> None:
        self.ddof = ddof
        self.start = self.end = 0
        self.reset()

    def reset(self) -> None:
        self.front: list[tuple[float, float, float]] = []
        self.back: list[float] = []
        self.agg_back = _ZERO
        self.nonfinite = self.nulls = 0

    def _push(self, v: float) -> None:
        if math.isfinite(v):
            self.back.append(v)
            self.agg_back = _insert_one(self.agg_back, v)
        else:
            self.back.append(0.0)
            self.agg_back = _insert_one(self.agg_back, 0.0)
            self.nonfinite += 1

    def _flip(self) -> None:
        agg = _ZERO
        back = self.back
        while back:
            agg = _insert_one(agg, back.pop())
            self.front.append(agg)
        self.agg_back = _ZERO

    def _pop(self, v: float) -> None:
        if not self.front:
            self._flip()
        self.front.pop()
        self.nonfinite -= not math.isfinite(v)

    def update(self, get: Getter, s: int, e: int) -> None:
        if s >= self.end:
            self.reset()
            self.start = self.end = s
        for i in range(self.start, s):
            v = get(i)
            if v is None:
                self.nulls -= 1
            else:
                self._pop(v)
        for i in range(self.end, e):
            v = get(i)
            if v is None:
                self.nulls += 1
            else:
                self._push(v)
        self.start, self.end = s, e

    def count(self) -> int:
        return self.end - self.start - self.nulls

    def get_var(self) -> Value:
        st = self.agg_back
        if self.front:
            st = _combine(st, self.front[-1])
        w, _, dp = st
        if w <= float(self.ddof):
            return None
        var = dp / (w - float(self.ddof))
        if var < 0.0:
            var = 0.0
        return math.nan if self.nonfinite > 0 else var

    # State beyond the ring: how many of the window's values sit in ``front``.
    # ``front`` and ``agg_back`` are rebuilt from the ring at load, by the
    # same operations that built them (the flip and the pushes).
    def extra_bytes(self) -> bytes:
        return struct.pack("<I", len(self.front))

    def load_extra(self, data: bytes, off: int, get: Getter) -> int:
        (a,) = struct.unpack_from("<I", data, off)
        vals: list[float] = []
        for i in range(self.start, self.end):
            v = get(i)
            if v is None:
                self.nulls += 1
            else:
                if not math.isfinite(v):
                    self.nonfinite += 1
                    v = 0.0
                vals.append(v)
        self.back = vals[:a]
        self._flip()  # builds front from the oldest a values, as the flip did
        for v in vals[a:]:
            self.back.append(v)
            self.agg_back = _insert_one(self.agg_back, v)
        return off + 4


class MonoMinMax:
    """polars-compute/src/rolling/arg_min_max.rs, ``ArgMinMaxWindow`` with
    ``MinPropagateNan`` / ``MaxPropagateNan``: a deque of row indices whose
    values are strictly better than every later one; a NaN is better than any
    number, so it propagates; a tie keeps the older row."""

    __slots__ = ("is_max", "dq", "nonnulls", "start", "end")

    def __init__(self, is_max: bool) -> None:
        self.is_max = is_max
        self.dq: deque[tuple[int, float]] = deque()
        self.nonnulls = 0
        self.start = self.end = 0

    def _better(self, a: float, b: float) -> bool:
        if self.is_max:  # nan_max_lt(b, a) = !(b.is_nan() | (b >= a))
            return not (b != b or b >= a)
        return not (b != b or a >= b)  # nan_min_lt(a, b) = !(b.is_nan() | (a >= b))

    def _insert(self, i: int, v: float) -> None:
        dq = self.dq
        while dq and self._better(v, dq[-1][1]):
            dq.pop()
        dq.append((i, v))
        self.nonnulls += 1

    def update(self, get: Getter, s: int, e: int) -> None:
        dq = self.dq
        if s >= self.end:
            # every row of the window leaves: Polars pops each from the deque
            # and uncounts each, which leaves exactly this
            dq.clear()
            self.nonnulls = 0
            self.start = self.end = s
        while dq and dq[0][0] < s:
            dq.popleft()
        for i in range(self.start, min(s, self.end)):
            self.nonnulls -= get(i) is not None
        for i in range(max(s, self.end), e):
            v = get(i)
            if v is not None:
                self._insert(i, v)
        self.start, self.end = s, e

    def count(self) -> int:
        return self.nonnulls

    def get_best(self) -> Value:
        return self.dq[0][1] if self.dq else None

    # Nothing beyond the ring: the deque is the unique chain of the window's
    # values, rebuilt at load by inserting them in order.
    def extra_bytes(self) -> bytes:
        return b""

    def load_extra(self, data: bytes, off: int, get: Getter) -> int:
        for i in range(self.start, self.end):
            v = get(i)
            if v is not None:
                self._insert(i, v)
        return off


AGG_KINDS = ("sum", "mean", "var", "min", "max")


def _new_agg(kind: str, ddof: int) -> KahanSum | SwagVar | MonoMinMax:
    if kind in ("sum", "mean"):
        return KahanSum()
    if kind == "var":
        return SwagVar(ddof)
    return MonoMinMax(kind == "max")


def _read(agg: KahanSum | SwagVar | MonoMinMax, kind: str) -> Value:
    if kind == "sum":
        assert isinstance(agg, KahanSum)
        return agg.get_sum()
    if kind == "mean":
        assert isinstance(agg, KahanSum)
        return agg.get_mean()
    if kind == "var":
        assert isinstance(agg, SwagVar)
        return agg.get_var()
    assert isinstance(agg, MonoMinMax)
    return agg.get_best()


@dataclass(frozen=True)
class WindowSpec:
    """A rolling window: ``rows`` rows (Polars ``rolling_*``) or a span of
    ``span`` clock units (Polars ``rolling_*_by``, ``closed``), with
    ``min_samples`` and, for the variance, ``ddof``. ``cap`` bounds the rows a
    span may hold (a hard cap: a stream over a span has no other bound)."""

    rows: int | None = None
    span: int | None = None
    closed: str = "right"
    min_samples: int = 1
    ddof: int = 1
    cap: int = 1 << 20
    # ``asof_ties``: accept a repeated time and read the window as of the row
    # (rows up to and including it). Polars instead gives every row of a
    # repeated time the window of the last of them, which a stream knows only
    # when the time moves on; the study measures the difference.
    asof_ties: bool = False


class Rolling:
    """One or more aggregators over one column and one window, sharing one ring
    of the rows the window can still read (value, validity and, on a time
    clock, the time).

    The driver mirrors Polars: a row-counted window is rows ``[i+1-w, i+1)``
    (polars-compute/src/rolling/{no_nulls,nulls}/mod.rs); a span window is
    found as ``group_by_values_iter_lookbehind`` finds it
    (polars-time/src/windows/group_by.rs) and fed through
    ``rolling_apply_agg_window_sorted`` (polars-time .../rolling_kernels/
    shared.rs): a window holding fewer than ``min_samples`` rows, nulls
    included, is not updated at all and gives null. The no-nulls row driver
    skips the same way; the nulls row driver never skips. On a row-counted
    window the skips can only fall before the window first fills, where they
    only delay additions, so the two orders are the same operations."""

    def __init__(self, spec: WindowSpec, kinds: Iterable[str]) -> None:
        assert (spec.rows is None) != (spec.span is None)
        self.spec = spec
        self.kinds = tuple(kinds)
        assert all(k in AGG_KINDS for k in self.kinds)
        # one aggregator per distinct kernel; sum and mean share one
        self.aggs: dict[str, KahanSum | SwagVar | MonoMinMax] = {}
        for k in self.kinds:
            key = "sum" if k == "mean" else k
            if key not in self.aggs:
                self.aggs[key] = _new_agg(key, spec.ddof)
        self.n = 0  # rows seen (the row clock)
        self.base = 0  # absolute index of ring[0]
        self.vals: deque[Value] = deque()
        self.times: deque[int] = deque()
        self.scan = 0  # absolute start of the latest window
        self.last_t: int | None = None

    def _get(self, i: int) -> Value:
        return self.vals[i - self.base]

    def step(self, x: Value, t: int | None = None) -> tuple[Value, ...]:
        spec = self.spec
        i = self.n
        self.n += 1
        self.vals.append(x)
        if spec.rows is not None:
            e = i + 1
            s = max(0, e - spec.rows)
        else:
            assert t is not None
            if self.last_t is not None and (t < self.last_t or (t == self.last_t and not spec.asof_ties)):
                raise ClockError(f"time {t} after {self.last_t}")
            self.last_t = t
            self.times.append(t)
            lower = t - spec.span  # type: ignore[operator]
            left_in = spec.closed in ("left", "both")
            s = self.scan
            times = self.times
            base = self.base
            while s < i:
                ts = times[s - base]
                if (ts >= lower) if left_in else (ts > lower):
                    break
                s += 1
            self.scan = s
            # every earlier row is before t (the clock rises), so the row
            # itself decides the end
            e = i + 1 if spec.closed in ("right", "both") else i
            if e - s > spec.cap:
                raise ClockError(f"{e - s} rows inside the span, over the cap {spec.cap}")
        length = e - s
        outs: list[Value] = []
        if length < spec.min_samples:
            outs = [None] * len(self.kinds)
        else:
            for agg in self.aggs.values():
                agg.update(self._get, s, e)
            for k in self.kinds:
                agg = self.aggs["sum" if k == "mean" else k]
                outs.append(_read(agg, k) if agg.count() >= spec.min_samples else None)
        # Drop the rows no window and no aggregator can read again. The
        # aggregators move in lockstep. When the window they hold is wholly
        # before the latest window start, their next update resets them, so
        # they need none of it; mark them empty at ``s`` (the reset still
        # happens, since every later start is at or after ``s``).
        a0 = next(iter(self.aggs.values()))
        if a0.end > s:
            keep = min(s, a0.start)
        else:
            keep = s
            for a in self.aggs.values():
                a.start = a.end = s
        while self.base < keep:
            self.vals.popleft()
            if spec.span is not None:
                self.times.popleft()
            self.base += 1
        return tuple(outs)

    def ring_len(self) -> int:
        return len(self.vals)

    def to_bytes(self) -> bytes:
        head = struct.pack("<QQQ", self.n, self.base, self.scan)
        body = _pack_values(list(self.vals))
        if self.spec.span is not None:
            body += _pack_i64s(list(self.times))
        aggs = b"".join(
            struct.pack("<QQ", a.start - self.base, a.end - self.base) + a.extra_bytes()
            for a in self.aggs.values()
        )
        return head + body + aggs

    @classmethod
    def from_bytes(cls, spec: WindowSpec, kinds: Iterable[str], data: bytes) -> Rolling:
        out = cls(spec, kinds)
        out.n, out.base, out.scan = struct.unpack_from("<QQQ", data, 0)
        vals, off = _unpack_values(data, 24)
        out.vals.extend(vals)
        if spec.span is not None:
            times, off = _unpack_i64s(data, off)
            out.times.extend(times)
            out.last_t = times[-1] if times else None
        for agg in out.aggs.values():
            a0, a1 = struct.unpack_from("<QQ", data, off)
            off += 16
            agg.start, agg.end = out.base + a0, out.base + a1
            off = agg.load_extra(data, off, out._get)
        assert off == len(data)
        return out


# ---------------------------------------------------------------------------
# Exponentially weighted features. polars-compute/src/ewm/{mean,cov}.rs and
# polars-ops/src/series/ops/ewm_by.rs. Python's ``ewm_*`` maps ``half_life``
# to ``alpha = 1 - exp(-ln 2 / half_life)`` (polars/expr/expr.py,
# ``_prepare_alpha``); the same expression is used here.


def alpha_of_half_life(half_life: float) -> float:
    return 1.0 - math.exp(-math.log(2.0) / half_life)


class EwmMean:
    """``EwmMeanState``. Constant state: the mean, the weight, a count."""

    def __init__(self, alpha: float, adjust: bool = True, min_samples: int = 1, ignore_nulls: bool = False) -> None:
        self.alpha, self.adjust, self.ignore_nulls = alpha, adjust, ignore_nulls
        self.min_samples = max(min_samples, 1)
        self.mean = 0.0
        self.weight = 0.0
        self.count = 0

    def step(self, x: Value, t: int | None = None) -> Value:
        nvw = 1.0 if self.adjust else self.alpha
        if self.count == 0 and x is not None:
            self.count, self.mean, self.weight = 1, x, 1.0
        else:
            if x is not None or not self.ignore_nulls:
                self.weight *= 1.0 - self.alpha
            if x is not None:
                nw = self.weight + nvw
                self.mean = self.mean + (x - self.mean) * (nvw / nw)
                self.weight = self.weight + 1.0 if self.adjust else 1.0
                self.count += 1
        return self.mean if (x is not None and self.count >= self.min_samples) else None

    def to_bytes(self) -> bytes:
        return struct.pack("<ddQ", self.mean, self.weight, self.count)

    def load(self, data: bytes) -> None:
        self.mean, self.weight, self.count = struct.unpack("<ddQ", data)


class EwmVar:
    """``EwmCovState`` fed ``(x, x)``; ``std`` takes the square root of the
    variance, as ``ewm_std`` does. The two means of the covariance state are
    equal for ``(x, x)`` (the same operations on the same inputs), so one is
    kept. Constant state: five f64 and a count."""

    def __init__(
        self,
        alpha: float,
        adjust: bool = True,
        bias: bool = False,
        min_samples: int = 1,
        ignore_nulls: bool = False,
        std: bool = False,
    ) -> None:
        self.alpha, self.adjust, self.bias, self.ignore_nulls, self.std = alpha, adjust, bias, ignore_nulls, std
        self.min_samples = max(min_samples, 1)
        self.weight = self.mean = self.cov = self.wsum = self.wsq = 0.0
        self.count = 0

    def step(self, x: Value, t: int | None = None) -> Value:
        ow = 1.0 if self.adjust else self.alpha
        a = self.alpha
        if self.count == 0 and x is not None:
            self.count, self.mean, self.weight, self.wsum, self.wsq = 1, x, 1.0, 1.0, 1.0
        else:
            if x is not None or not self.ignore_nulls:
                self.wsum *= 1.0 - a
                self.wsq *= (1.0 - a) * (1.0 - a)
                self.weight *= 1.0 - a
            if x is not None:
                self.count += 1
                nw = self.weight + ow
                frac = ow / nw
                d = x - self.mean
                nm = self.mean + d * frac
                cov = (self.weight * (self.cov + (self.mean - nm) * (self.mean - nm)) + ow * (x - nm) * (x - nm)) / nw
                self.cov, self.weight, self.mean = cov, nw, nm
                self.wsum += ow
                self.wsq += ow * ow
                if not self.adjust:
                    self.wsum /= nw
                    self.wsq /= nw * nw
                    self.weight = 1.0
        if x is None or self.count < self.min_samples:
            return None
        if self.bias:
            v = self.cov
        else:
            num = self.wsum * self.wsum
            den = num - self.wsq
            if not den > 0.0:
                return None
            v = (num / den) * self.cov
        return math.sqrt(v) if self.std else v

    def to_bytes(self) -> bytes:
        return struct.pack("<5dQ", self.weight, self.mean, self.cov, self.wsum, self.wsq, self.count)

    def load(self, data: bytes) -> None:
        self.weight, self.mean, self.cov, self.wsum, self.wsq, self.count = struct.unpack("<5dQ", data)


class EwmMeanBy:
    """``ewm_by_core`` with ``IS_MEAN``: ``y = (1 - a) y_prev + a x`` with
    ``1 - a = 0.5 ** (dt / half_life)``, and ``y = x`` when ``x`` equals the
    previous result. A null row gives null and moves nothing, so the next
    row's ``dt`` runs from the last row with a value. Constant state: the
    previous time and result and a started flag."""

    def __init__(self, half_life: int) -> None:
        self.half_life = half_life
        self.prev_t = 0
        self.prev = 0.0
        self.started = False
        self.last_t: int | None = None

    def step(self, x: Value, t: int | None = None) -> Value:
        assert t is not None
        if self.last_t is not None and t < self.last_t:
            raise ClockError(f"time {t} after {self.last_t}")  # a repeat is fine: dt = 0
        self.last_t = t
        if x is None:
            return None
        if not self.started:
            self.prev_t, self.prev, self.started = t, x, True
            return x
        if x == self.prev:
            y = x
        else:
            oma = 0.5 ** (float(t - self.prev_t) / float(self.half_life))
            y = oma * self.prev + (1.0 - oma) * x
        self.prev_t, self.prev = t, y
        return y

    def to_bytes(self) -> bytes:
        return struct.pack("<qdq?", self.prev_t, self.prev, self.last_t if self.last_t is not None else -(1 << 63), self.started)

    def load(self, data: bytes) -> None:
        self.prev_t, self.prev, lt, self.started = struct.unpack("<qdq?", data)
        self.last_t = None if lt == -(1 << 63) else lt
