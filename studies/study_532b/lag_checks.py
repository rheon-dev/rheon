"""Size checks for exact lags kept in the bank.

Two lags, two checks (the user's rule: exact lags stay in the bank; a state
may not grow past the per-group cap, 256 KB by default).

* **Row lag** of ``k`` (the value ``k`` rows back): the state is ``Lag`` in
  ``lag.py``, ``8 + 8k + ceil(k/8)`` bytes a column and group, known at the
  call. Refused when a group's lag columns together would exceed the cap,
  warned when they would exceed half of it.
* **Clock lag** of ``D`` time units (the value of the latest row at or before
  ``t - D``): the rows held are those in ``(t - D, t]`` plus one anchor at or
  before ``t - D``, so how many depends on the data, and the check runs as the
  stream goes: it warns at half the cap and refuses at the cap. The state is
  ``8 + 16 m + ceil(m / 8)`` bytes for ``m`` held rows (a time and a value
  each, and a validity bit).

The refusal of the clock lag points to the bucketed approximate lag (PLAN
task 536: one value per fixed time bucket, a state that does not grow with
the rows). The text of every message describes the mechanism and cites no
plan section or task number, because rheon's error messages may not.

KB here is 1024 bytes (256 KB = 262144 bytes, half = 131072); with 1000-byte
kilobytes every threshold moves by 2.4%.
"""

from __future__ import annotations

import math
import struct
from collections import deque

from .lag import Lag

Value = float | None

CAP = 256 * 1024
WARN = CAP // 2

CLOCK_HEADER = 8  # the count (u32) and a pad (u32), as Lag's header
CLOCK_ROW = 16  # a time (i64) and a value (f64); plus one validity bit a row


class LagSizeError(Exception):
    """A lag whose state would pass the per-group cap."""


class LagSizeWarning(UserWarning):
    """A lag whose state passed half the per-group cap."""


# ---------------------------------------------------------------------------
# Row lag: known at the call.


def row_lag_bytes(k: int, columns: int = 1) -> int:
    """Saved bytes of one group's row lag of ``k`` over ``columns`` columns."""
    return columns * Lag.bytes_formula(k)


def clock_lag_bytes(m: int, columns: int = 1) -> int:
    """Saved bytes of one group's clock lag holding ``m`` rows per column."""
    return columns * (CLOCK_HEADER + CLOCK_ROW * m + math.ceil(m / 8))


def _largest(size, limit: int, columns: int) -> int:
    """The largest n with size(n, columns) <= limit (0 if none); size rises."""
    if size(0, columns) > limit:
        return -1
    lo, hi = 0, 1
    while size(hi, columns) <= limit:
        hi *= 2
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if size(mid, columns) <= limit:
            lo = mid
        else:
            hi = mid
    return lo


def max_row_k(limit: int, columns: int = 1) -> int:
    """The largest row lag whose bytes stay at or under ``limit``."""
    return _largest(row_lag_bytes, limit, columns)


def max_clock_rows(limit: int, columns: int = 1) -> int:
    """The most rows a clock lag may hold with bytes at or under ``limit``."""
    return _largest(clock_lag_bytes, limit, columns)


def row_lag_verdict(k: int, columns: int = 1, cap: int = CAP) -> tuple[str, int]:
    """``("ok" | "warn" | "refuse", bytes)`` for a row lag at the call."""
    b = row_lag_bytes(k, columns)
    if b > cap:
        return "refuse", b
    if b > cap // 2:
        return "warn", b
    return "ok", b


def check_row_lag(k: int, columns: int = 1, cap: int = CAP) -> str:
    """Raise ``LagSizeError`` for a row lag over the cap; return the message
    of the warning for one over half of it, or "" when it is under."""
    verdict, b = row_lag_verdict(k, columns, cap)
    if verdict == "refuse":
        raise LagSizeError(
            f"a lag of {k} rows over {columns} column(s) keeps the last {k} values of "
            f"every group, {b} bytes a group, over the {cap}-byte per-group cap; "
            f"use a smaller lag"
        )
    if verdict == "warn":
        return (
            f"a lag of {k} rows over {columns} column(s) keeps {b} bytes a group, "
            f"over half of the {cap}-byte per-group cap"
        )
    return ""


# ---------------------------------------------------------------------------
# Clock lag: checked as the stream goes.


def clock_refusal(duration: int, m: int, columns: int, cap: int) -> str:
    return (
        f"a lag of {duration} time units on the clock holds every row of that span, "
        f"and this group's lag state would reach {clock_lag_bytes(m, columns)} bytes "
        f"at {m} rows, over the {cap}-byte per-group cap; use a bucketed "
        f"approximation of the lag (one value per fixed time bucket, a state that "
        f"does not grow with the number of rows)"
    )


class ClockLag:
    """The value of the latest row at or before ``t - D`` (null until there
    is one); exact. Keeps the rows in ``(t - D, t]`` and one anchor at or
    before ``t - D``. With a ``cap``, checks its size on every row: refuses
    (``LagSizeError``) at the row that would take it over the cap, and notes
    the half-cap crossing once in ``warning``.

    The check is one integer comparison against two row counts worked out
    once from the byte formula (``m_warn``, ``m_cap``), so it adds no
    arithmetic per row; the time-window eviction is part of the lag itself.
    """

    def __init__(self, duration: int, columns: int = 1, cap: int | None = CAP) -> None:
        self.duration, self.columns, self.cap = duration, columns, cap
        self.rows: deque[tuple[int, Value]] = deque()
        self.warning = ""
        if cap is not None:
            self.m_cap = max_clock_rows(cap, columns)
            self.m_warn = max_clock_rows(cap // 2, columns)

    def step(self, t: int, x: Value) -> Value:
        rows, edge = self.rows, t - self.duration
        while len(rows) > 1 and rows[1][0] <= edge:
            rows.popleft()
        out = rows[0][1] if rows and rows[0][0] <= edge else None
        rows.append((t, x))
        if self.cap is not None:
            m = len(rows)
            if m > self.m_warn:
                self._size_alarm(m)
        return out

    def _size_alarm(self, m: int) -> None:
        if m > self.m_cap:
            self.rows.pop()  # the row is not kept: the state stays under the cap
            raise LagSizeError(clock_refusal(self.duration, m, self.columns, self.cap))
        if not self.warning:
            self.warning = (
                f"a lag of {self.duration} time units on the clock now holds {m} rows, "
                f"{clock_lag_bytes(m, self.columns)} bytes, over half of the "
                f"{self.cap}-byte per-group cap"
            )

    def to_bytes(self) -> bytes:
        """The count (u32) and a pad (u32), the ``m`` times (i64) oldest
        first, the ``m`` values (f64), then the validity bits."""
        m = len(self.rows)
        bits = bytearray(math.ceil(m / 8))
        for j, (_, v) in enumerate(self.rows):
            if v is not None:
                bits[j // 8] |= 1 << (j % 8)
        return (
            struct.pack("<II", m, 0)
            + struct.pack(f"<{m}q", *[t for t, _ in self.rows])
            + struct.pack(f"<{m}d", *[0.0 if v is None else v for _, v in self.rows])
            + bytes(bits)
        )

    def load(self, data: bytes) -> None:
        m, _ = struct.unpack_from("<II", data, 0)
        ts = struct.unpack_from(f"<{m}q", data, 8)
        vs = struct.unpack_from(f"<{m}d", data, 8 + 8 * m)
        bits = data[8 + 16 * m :]
        self.rows = deque(
            (ts[j], vs[j] if bits[j // 8] >> (j % 8) & 1 else None) for j in range(m)
        )
