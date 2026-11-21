"""Exponentially weighted features: constant state, no row kept.

``EwmMean``, ``EwmVar`` and ``EwmMeanBy`` mirror, operation for operation, the
kernels Polars 2.0.0 runs (``polars-compute/src/ewm/{mean,cov}.rs`` and
``polars-ops/src/series/ops/ewm_by.rs``), as study 532 established; the update
equations are in each docstring. One change from study 532: **alpha is carried
in the state**. It is computed once, when the feature is made, from Polars' own
expression ``alpha = 1 - exp(-ln 2 / half_life)``, and from then on it is read
from the saved bytes, never recomputed. ``exp`` and ``ln`` are libm results and
may differ by an ulp between operating systems; carried, they cannot, so a
state saved on one OS continues on another to the bit (nothing else in these
three updates reads libm, except ``EwmMeanBy``'s per-row ``pow``).

``PeakEnvelope`` is the decayed max (or min) envelope: the branching peak
detector with instantaneous attack (Giannoulis, Massberg and Reiss, "Digital
Dynamic Range Compressor Design -- A Tutorial and Analysis", J. Audio Eng.
Soc. 60(6), 2012, section 3.1, eq. 11 with attack coefficient 0)::

    y = x                          if x > y_prev   (attack: jump to a new peak)
    y = r y_prev + (1 - r) x       otherwise       (release toward the value)

with ``r = 1 - alpha``. The min envelope is the same with the comparison
reversed. It is translation- and scale-equivariant: shifting or scaling every
``x`` shifts or scales ``y`` alike.

``DecayedMax`` is the time-decayed maximum of Cormode, Shkapenyuk, Srivastava
and Xu, "Forward Decay: A Practical Time Decay Model for Streaming Systems",
ICDE 2009 (section V, the decayed max under exponential decay, which is the
same under forward and backward decay): ``y = max_i r^(i_now - i) x_i``,
kept as ``y = max(x, r y_prev)``. It decays toward 0, so its value depends on
where 0 is: it is a feature for non-negative quantities (a count, a volume, an
absolute return), not for a level.
"""

from __future__ import annotations

import math
import struct

Value = float | None


def alpha_of_half_life(half_life: float) -> float:
    """Polars' own mapping (py-polars ``_prepare_alpha``)."""
    return 1.0 - math.exp(-math.log(2.0) / half_life)


class EwmMean:
    """``EwmMeanState``. State: alpha (carried), the mean ``m``, the weight
    ``W``, the count. First value: ``m = x, W = 1``. Then each row:
    ``W *= 1 - alpha`` unless the row is null and ``ignore_nulls``; on a
    value, with ``u = 1`` (``adjust``) or ``alpha``:
    ``m = m + (x - m) (u / (W + u))``, then ``W = W + 1`` (``adjust``) or 1.
    32 bytes."""

    def __init__(
        self, alpha: float, adjust: bool = True, min_samples: int = 1, ignore_nulls: bool = False
    ) -> None:
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
        return struct.pack("<dddQ", self.alpha, self.mean, self.weight, self.count)

    def load(self, data: bytes) -> None:
        self.alpha, self.mean, self.weight, self.count = struct.unpack("<dddQ", data)


class EwmVar:
    """``EwmCovState`` fed ``(x, x)``; ``std`` takes the square root, as
    ``ewm_std`` does. State: alpha (carried), ``W``, the mean ``m``, the
    covariance ``c``, ``S1`` and ``S2``, the count. Each row the three weights
    decay by ``1 - alpha`` (``S2`` by its square) under the same null rule; on
    a value: ``W' = W + u; m' = m + (x - m) u / W';
    c = (W (c + (m - m')^2) + u (x - m')^2) / W'; S1 += u; S2 += u^2``, and for
    ``adjust=False`` ``S1 /= W', S2 /= W'^2, W = 1``. Output ``c`` (``bias``)
    or ``S1^2 / (S1^2 - S2) c``. 56 bytes."""

    def __init__(
        self,
        alpha: float,
        adjust: bool = True,
        bias: bool = False,
        min_samples: int = 1,
        ignore_nulls: bool = False,
        std: bool = False,
    ) -> None:
        self.alpha, self.adjust, self.bias, self.ignore_nulls, self.std = (
            alpha,
            adjust,
            bias,
            ignore_nulls,
            std,
        )
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
                cov = (
                    self.weight * (self.cov + (self.mean - nm) * (self.mean - nm))
                    + ow * (x - nm) * (x - nm)
                ) / nw
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
        return struct.pack(
            "<6dQ", self.alpha, self.weight, self.mean, self.cov, self.wsum, self.wsq, self.count
        )

    def load(self, data: bytes) -> None:
        (self.alpha, self.weight, self.mean, self.cov, self.wsum, self.wsq, self.count) = (
            struct.unpack("<6dQ", data)
        )


class EwmMeanBy:
    """``ewm_by_core`` with ``IS_MEAN``: ``y = x`` at the first value; then
    ``y = x`` when ``x`` equals the last ``y``, else
    ``y = r y_prev + (1 - r) x`` with ``r = 0.5 ** ((t - t_prev) / half_life)``.
    A null row gives null and moves nothing. State: the last time with a
    value, the last result, the last time seen, a started flag: 25 bytes
    (``half_life`` is the spec's; no libm result is carried, but ``pow`` runs
    on every row)."""

    def __init__(self, half_life: int) -> None:
        self.half_life = half_life
        self.prev_t = 0
        self.prev = 0.0
        self.started = False
        self.last_t: int | None = None

    def step(self, x: Value, t: int | None = None) -> Value:
        assert t is not None
        if self.last_t is not None and t < self.last_t:
            raise ValueError(f"time {t} after {self.last_t}")
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
        return struct.pack(
            "<qdq?",
            self.prev_t,
            self.prev,
            self.last_t if self.last_t is not None else -(1 << 63),
            self.started,
        )

    def load(self, data: bytes) -> None:
        self.prev_t, self.prev, lt, self.started = struct.unpack("<qdq?", data)
        self.last_t = None if lt == -(1 << 63) else lt


class PeakEnvelope:
    """The decayed max (``is_max``) or min envelope, instantaneous attack and
    exponential release (Giannoulis et al. 2012). State: ``r = 1 - alpha``
    (carried), the envelope, a started flag: 17 bytes. A null row gives null
    and moves nothing (there is no value to release toward); a NaN poisons the
    envelope, as it poisons an EW mean."""

    def __init__(self, alpha: float, is_max: bool = True) -> None:
        self.r = 1.0 - alpha
        self.is_max = is_max
        self.y = 0.0
        self.started = False

    def step(self, x: Value, t: int | None = None) -> Value:
        if x is None:
            return None
        if not self.started:
            self.y, self.started = x, True
            return x
        y = self.y
        if (x > y) if self.is_max else (x < y):
            self.y = x
        else:
            self.y = self.r * y + (1.0 - self.r) * x
        return self.y

    def to_bytes(self) -> bytes:
        return struct.pack("<dd?", self.r, self.y, self.started)

    def load(self, data: bytes) -> None:
        self.r, self.y, self.started = struct.unpack("<dd?", data)


class DecayedMax:
    """Cormode et al.'s decayed maximum under exponential decay, on the row
    clock: ``y = max(x, r y_prev)``. A null row decays ``y`` (time passes) and
    gives null. 17 bytes."""

    def __init__(self, alpha: float) -> None:
        self.r = 1.0 - alpha
        self.y = 0.0
        self.started = False

    def step(self, x: Value, t: int | None = None) -> Value:
        if self.started:
            self.y = self.r * self.y
        if x is None:
            return None
        if not self.started or x > self.y or x != x:
            self.y, self.started = x, True
        return self.y

    def to_bytes(self) -> bytes:
        return struct.pack("<dd?", self.r, self.y, self.started)

    def load(self, data: bytes) -> None:
        self.r, self.y, self.started = struct.unpack("<dd?", data)
