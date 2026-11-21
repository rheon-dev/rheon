"""Lags: ``shift(k)``, ``diff(k)``, ``pct_change(k)`` as a state.

A lag of ``k`` must hold the last ``k`` values: the output ``k`` rows from now
is the value of this row, which can be any f64, so any state that serves it
determines ``k`` arbitrary f64 and is at least ``8k`` bytes. This is the
smallest such state: the last ``k`` values (with their validity) and how many
rows were seen, up to ``k``. Saved: ``8 + 8k + ceil(k/8)`` bytes.

Update: read the oldest of the ring (null until ``k`` rows were seen), then
push ``x``. ``shift = x_{i-k}``; ``diff = x_i - x_{i-k}``;
``pct_change = (x_i - x_{i-k}) / x_{i-k}`` (Polars' ``diff(s, k) / s.shift(k)``).
A zero-weight row is a null: it takes its slot on the row clock.
"""

from __future__ import annotations

import math
import struct

Value = float | None


class Lag:
    def __init__(self, kind: str, k: int) -> None:
        self.kind, self.k = kind, k
        self.ring: list[Value] = [None] * k
        self.pos = 0
        self.seen = 0

    def step(self, x: Value, t: int | None = None) -> Value:
        old = self.ring[self.pos] if self.seen >= self.k else None
        self.ring[self.pos] = x
        self.pos = (self.pos + 1) % self.k
        self.seen = min(self.seen + 1, self.k)
        if self.kind == "shift":
            return old
        if old is None or x is None:
            return None
        if self.kind == "diff":
            return x - old
        return (x - old) / old

    def to_bytes(self) -> bytes:
        """seen and pos (u32 each), the k values oldest first, then the
        validity bits."""
        vals = [self.ring[(self.pos + j) % self.k] for j in range(self.k)]
        bits = bytearray(math.ceil(self.k / 8))
        for j, v in enumerate(vals):
            if v is not None:
                bits[j // 8] |= 1 << (j % 8)
        return (
            struct.pack("<II", self.seen, 0)
            + struct.pack(f"<{self.k}d", *[0.0 if v is None else v for v in vals])
            + bytes(bits)
        )

    def load(self, data: bytes) -> None:
        self.seen, _ = struct.unpack_from("<II", data, 0)
        vals = struct.unpack_from(f"<{self.k}d", data, 8)
        bits = data[8 + 8 * self.k :]
        self.ring = [vals[j] if bits[j // 8] >> (j % 8) & 1 else None for j in range(self.k)]
        self.pos = 0

    @staticmethod
    def bytes_formula(k: int) -> int:
        return 8 + 8 * k + math.ceil(k / 8)
