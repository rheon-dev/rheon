"""Bucketed windows: a time span cut into fixed buckets of the clock, with no
row kept.

A bucket of width ``b`` (int64 clock units) holds the rows with
``floor(t / b) = k``. Each bucket keeps, per input column, five numbers about
the values that fell in it: the count ``n``, a compensated sum ``(s, c)``, the
centred sum of squares ``m2`` (Welford's update with the bucket's mean read
from its sum), and the minimum and maximum. A window of ``B`` buckets is read
as the last ``B`` buckets of the clock, the current one included:
``[(k - B + 1) b, t]`` at a row of time ``t`` in bucket ``k``. The state is a
ring of ``B`` slots (``B + 1`` when the bucket before the window is kept for
the interpolated read) and the current bucket id. Nothing in it grows with the
stream or with the rows per bucket.

Conventions (the same as study 532's):

- ``x is None`` is a null; a zero-weight row is fed as ``None`` (the caller
  masks it): it moves the clock to its bucket and enters nothing.
- A NaN is a value. It propagates through the sum, the variance, the minimum
  and the maximum of every window that holds its bucket, as in Polars'
  ``group_by_dynamic`` and ``rolling_*_by`` (a bucket is never subtracted, so
  plain IEEE arithmetic does this; when its bucket leaves the window, it is
  gone). +inf and -inf likewise.
- The bucket clock may not go back: ``floor(t / b)`` must not fall within a
  group. A repeated time, or times out of order inside the current bucket, are
  fine: they land in the same bucket.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass

Value = float | None


class ClockError(ValueError):
    """A bucket clock that goes back within its group."""


@dataclass(frozen=True)
class BucketSpec:
    width: int  # b, in clock units
    buckets: int  # B, the buckets the window reads, the current one included
    keep_previous: bool = False  # keep bucket k - B as well (interpolated and B+1 reads)
    ddof: int = 1
    min_samples: int = 1
    sumsq: bool = False  # also keep a plain sum of squares (the cancelling form, for comparison)

    @property
    def ring(self) -> int:
        return self.buckets + (1 if self.keep_previous else 0)


_NEG = -(1 << 63)


def _nan_max(a: float, b: float) -> float:
    if a != a or b != b:
        return math.nan
    return a if a >= b else b


def _nan_min(a: float, b: float) -> float:
    if a != a or b != b:
        return math.nan
    return a if a <= b else b


class Bucketed:
    """One column's bucketed window state for one group."""

    __slots__ = ("spec", "head", "n", "s", "c", "m2", "mn", "mx", "sq")

    def __init__(self, spec: BucketSpec) -> None:
        self.spec = spec
        r = spec.ring
        self.head: int | None = None
        self.n = [0] * r
        self.s = [0.0] * r
        self.c = [0.0] * r
        self.m2 = [0.0] * r
        self.mn = [math.inf] * r
        self.mx = [-math.inf] * r
        self.sq = [0.0] * r if spec.sumsq else []

    # -- the update ---------------------------------------------------------

    def _clear(self, j: int) -> None:
        self.n[j] = 0
        self.s[j] = self.c[j] = self.m2[j] = 0.0
        self.mn[j] = math.inf
        self.mx[j] = -math.inf
        if self.spec.sumsq:
            self.sq[j] = 0.0

    def advance(self, k: int) -> None:
        """Move the clock to bucket ``k``: the buckets between leave."""
        r = self.spec.ring
        if self.head is None:
            for j in range(r):
                self._clear(j)
            self.head = k
            return
        if k < self.head:
            raise ClockError(f"bucket {k} after bucket {self.head}")
        if k > self.head:
            for kk in range(self.head + 1, min(k, self.head + r) + 1):
                self._clear(kk % r)
            self.head = k

    def insert(self, x: float) -> None:
        assert self.head is not None
        j = self.head % self.spec.ring
        n0 = self.n[j]
        n1 = n0 + 1
        s, c = self.s[j], self.c[j]
        mean0 = (s + c) / n0 if n0 else 0.0
        # Neumaier's compensated step: c gathers the low-order bits of s.
        t = s + x
        if abs(s) >= abs(x):
            c += (s - t) + x
        else:
            c += (x - t) + s
        self.s[j], self.c[j], self.n[j] = t, c, n1
        if n0 == 0:
            self.m2[j] = 0.0 if math.isfinite(x) else math.nan
        else:
            mean1 = (t + c) / n1 if math.isfinite(t) else t
            self.m2[j] += (x - mean0) * (x - mean1)
        self.mn[j] = _nan_min(self.mn[j], x)
        self.mx[j] = _nan_max(self.mx[j], x)
        if self.spec.sumsq:
            self.sq[j] += x * x

    def step(self, x: Value, t: int) -> None:
        self.advance(t // self.spec.width)
        if x is not None:
            self.insert(x)

    # -- the reads ----------------------------------------------------------

    def _ids(self, first: int, last: int) -> list[int]:
        """Ring slots of buckets ``first..last`` that are on the ring and not empty."""
        assert self.head is not None
        r = self.spec.ring
        lo = max(first, self.head - r + 1)
        return [kk % r for kk in range(lo, last + 1) if self.n[kk % r]]

    def _bucket_sum(self, j: int) -> float:
        s = self.s[j]
        return s + self.c[j] if math.isfinite(s) else s

    def read(self, lag_buckets: int = 0) -> dict[str, Value]:
        """The window of ``B`` buckets ending at bucket ``head`` (``lag_buckets``
        widens it into the past by that many buckets, which must be kept)."""
        assert self.head is not None
        sp = self.spec
        js = self._ids(self.head - sp.buckets + 1 - lag_buckets, self.head)
        return self._combine(js)

    def _combine(self, js: list[int]) -> dict[str, Value]:
        sp = self.spec
        n_tot = 0
        # The window's sum: Neumaier over the buckets' two terms.
        tot, comp = 0.0, 0.0
        # Chan's merge of (n, mean, m2), oldest bucket first.
        mean, m2 = 0.0, 0.0
        mn, mx = math.inf, -math.inf
        sq = 0.0
        for j in js:
            nb = self.n[j]
            # a bucket whose sum is not finite has a NaN compensation; leave it out
            for v in (self.s[j], self.c[j]) if math.isfinite(self.s[j]) else (self.s[j],):
                t = tot + v
                if abs(tot) >= abs(v):
                    comp += (tot - t) + v
                else:
                    comp += (v - t) + tot
                tot = t
            mb = self._bucket_sum(j) / nb
            if n_tot == 0:
                mean, m2 = mb, self.m2[j]
            else:
                nn = n_tot + nb
                d = mb - mean
                mean = mean + d * (nb / nn)
                m2 = m2 + self.m2[j] + d * d * (n_tot * nb / nn)
            n_tot += nb
            mn = _nan_min(mn, self.mn[j])
            mx = _nan_max(mx, self.mx[j])
            if sp.sumsq:
                sq += self.sq[j]
        return self._outputs(n_tot, tot, comp, m2, mn, mx, sq)

    def _outputs(
        self, n_tot: int, tot: float, comp: float, m2: float, mn: float, mx: float, sq: float
    ) -> dict[str, Value]:
        sp = self.spec
        out: dict[str, Value] = {"count": float(n_tot)}
        if n_tot == 0 or n_tot < sp.min_samples:
            out.update(
                sum=0.0 if (n_tot == 0 and sp.min_samples == 0) else None,
                mean=None,
                var=None,
                min=None,
                max=None,
            )
            if sp.sumsq:
                out["var_sumsq"] = None
            return out
        total = tot + comp if math.isfinite(tot) else tot
        out["sum"] = total
        out["mean"] = total / n_tot
        if n_tot > sp.ddof:
            v = m2 / (n_tot - sp.ddof)
            out["var"] = v if not v < 0.0 else 0.0
        else:
            out["var"] = None
        out["min"], out["max"] = mn, mx
        if sp.sumsq:
            out["var_sumsq"] = (
                (sq - total * total / n_tot) / (n_tot - sp.ddof) if n_tot > sp.ddof else None
            )
        return out

    def read_interpolated(self, t: int) -> dict[str, Value]:
        """The span ``(t - B b, t]`` estimated from the buckets: the ``B``
        buckets ``[(k - B + 1) b, t]`` plus the fraction
        ``f = (b - 1 - t mod b) / b`` of bucket ``k - B`` that lies inside the
        span, read as if its rows were spread evenly over its time (the
        "sliding window counter"). Only count, sum and mean have such a
        reading; min, max and the variance are not interpolated."""
        sp = self.spec
        assert sp.keep_previous and self.head is not None
        out = self.read()
        b = sp.width
        f = (b - 1 - (t % b)) / b
        prev = self._ids(self.head - sp.buckets, self.head - sp.buckets)
        if not prev or f == 0.0:
            return {"count": out["count"], "sum": out["sum"], "mean": out["mean"]}
        j = prev[0]
        cnt = (out["count"] or 0.0) + f * self.n[j]
        sm = (out["sum"] or 0.0) + f * self._bucket_sum(j)
        return {"count": cnt, "sum": sm, "mean": sm / cnt if cnt > 0 else None}

    # -- save and load ------------------------------------------------------

    def to_bytes(self) -> bytes:
        """``head`` (8 bytes) then each slot: n (u64), s, c, m2, min, max
        (and the sum of squares when kept): 48 or 56 bytes a slot."""
        parts = [struct.pack("<q", _NEG if self.head is None else self.head)]
        for j in range(self.spec.ring):
            parts.append(
                struct.pack(
                    "<Q5d", self.n[j], self.s[j], self.c[j], self.m2[j], self.mn[j], self.mx[j]
                )
            )
            if self.spec.sumsq:
                parts.append(struct.pack("<d", self.sq[j]))
        return b"".join(parts)

    @classmethod
    def from_bytes(cls, spec: BucketSpec, data: bytes) -> Bucketed:
        out = cls(spec)
        (h,) = struct.unpack_from("<q", data, 0)
        out.head = None if h == _NEG else h
        off = 8
        for j in range(spec.ring):
            out.n[j], out.s[j], out.c[j], out.m2[j], out.mn[j], out.mx[j] = struct.unpack_from(
                "<Q5d", data, off
            )
            off += 48
            if spec.sumsq:
                (out.sq[j],) = struct.unpack_from("<d", data, off)
                off += 8
        assert off == len(data)
        return out

    @staticmethod
    def bytes_formula(spec: BucketSpec, columns: int = 1) -> int:
        """Saved bytes for ``columns`` columns sharing one group's clock. Each
        column carries its own ``head`` here; a bank would share one."""
        per_slot = 56 if spec.sumsq else 48
        return columns * (8 + per_slot * spec.ring)
