"""Part 5: the lag size checks. Thresholds worked out and measured on real
saved states; the bytes of a held row; the clock check run on streams; the
per-row cost of the running check."""

from __future__ import annotations

import time

import numpy as np

from . import test_lag_checks as tests
from .lag import Lag
from .lag_checks import (
    CAP,
    WARN,
    ClockLag,
    LagSizeError,
    clock_lag_bytes,
    max_clock_rows,
    max_row_k,
    row_lag_bytes,
)


def _row_state_bytes(k: int) -> int:
    st = Lag("shift", k)
    for i in range(k + 3):
        st.step(float(i))
    return len(st.to_bytes())


def _clock_state_bytes(m: int) -> int:
    """Saved bytes of a clock lag holding exactly m rows, with some nulls."""
    st = ClockLag(10**12, cap=None)
    for t in range(m):
        st.step(t, None if t % 7 == 3 else float(t))
    assert len(st.rows) == m
    return len(st.to_bytes())


def thresholds() -> list[list]:
    """One column and 4: the row lag k and the clock rows at 128 KB and 256 KB,
    each with the saved bytes measured at the count and one past it."""
    rows = []
    for columns in (1, 4):
        for kind, mx, meas in (
            ("row lag, k", max_row_k, _row_state_bytes),
            ("clock lag, rows held", max_clock_rows, _clock_state_bytes),
        ):
            cells: list = [columns, kind]
            for limit in (WARN, CAP):
                n = mx(limit, columns)
                at, past = columns * meas(n), columns * meas(n + 1)
                assert at <= limit < past
                cells += [n, at, past]
            rows.append(cells)
    return rows


def held_row_bytes() -> list[list]:
    """Saved bytes of a clock lag state by rows held, and the increment a row
    adds (measured by differences of saved states, not by the formula)."""
    rows = []
    prev = None
    for m in (0, 1, 8, 9, 100, 1000, 10_000, 16_256):
        b = _clock_state_bytes(m)
        d = "" if prev is None else f"{(b - prev[1]) / (m - prev[0]):.3f}"
        rows.append([m, b, clock_lag_bytes(m), d])
        prev = (m, b)
    return rows


def day_examples(quick: bool) -> list[list]:
    """A lag of one day (86400 s) on the clock at several row rates, 1 and 4
    columns per group. The row count in a day is ``86400 / interval``; the
    verdict and the row it first appears at come from running the check on a
    stream of that rate (two days of it, or until refused)."""
    day = 86_400
    rows = []
    for interval in (60, 15, 10, 8, 5, 1):
        for columns in (1, 4):
            held = day // interval + 1
            st = ClockLag(day, columns=columns)
            warned_at = refused_at = None
            n = 0
            stop = 2 * (day // interval)
            for i in range(stop):
                n = i + 1
                try:
                    st.step(i * interval, 1.0)
                except LagSizeError:
                    refused_at = n
                    break
                if warned_at is None and st.warning:
                    warned_at = n
            verdict = "refused" if refused_at else ("warned" if warned_at else "ok")
            rows.append(
                [
                    f"1 row / {interval} s",
                    columns,
                    held,
                    clock_lag_bytes(held, columns),
                    "" if warned_at is None else warned_at,
                    "" if refused_at is None else refused_at,
                    verdict,
                ]
            )
    return rows


def example_totals() -> list[list]:
    """4 columns x 1000 groups: bytes at each threshold, a row lag and a clock
    lag (the same rows), and the state of the whole bank."""
    rows = []
    for limit_name, limit in (("128 KB", WARN), ("256 KB", CAP)):
        k = max_row_k(limit, 4)
        m = max_clock_rows(limit, 4)
        rows.append(
            [
                limit_name,
                k,
                f"{row_lag_bytes(k, 4) * 1000 / 1e6:.1f} MB",
                m,
                f"{clock_lag_bytes(m, 4) * 1000 / 1e6:.1f} MB",
            ]
        )
    return rows


# ---------------------------------------------------------------------------
# Cost of the running check.


def _stream(n: int, step: int = 5):
    rng = np.random.default_rng(7)
    t = np.cumsum(rng.integers(1, step + 1, n)).tolist()
    x = rng.standard_normal(n).tolist()
    return t, x


def _timed(fn) -> float:
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def _best(fn, reps: int = 3) -> float:
    return min(_timed(fn) for _ in range(reps))


def check_cost(n: int, reps: int = 11) -> list[list]:
    """ns a row of the clock lag's step without the check (append and eviction
    only), with it, and the difference; best of 11 over ``n`` rows of one
    group, at three held sizes. The last size is above half the cap, where the
    warning has been given and the comparison is against the cap."""
    ts, xs = _stream(n)
    mean_step = 3.0
    rows = []
    for held_target in (100, 3000, 12_000):
        d = int(held_target * mean_step)

        def run(cap: int | None, d: int = d) -> int:
            st = ClockLag(d, cap=cap)
            for t, x in zip(ts, xs, strict=True):
                st.step(t, x)
            return len(st.rows)

        held = run(None)
        # off and on alternate within each repetition, so a slow moment of the
        # machine hits both; the best of each is kept
        a = b = float("inf")
        for _ in range(reps):
            a = min(a, _timed(lambda: run(None)))
            b = min(b, _timed(lambda: run(CAP)))
        a, b = a / n * 1e9, b / n * 1e9
        rows.append([held, f"{a:.0f}", f"{b:.0f}", f"{b - a:.0f}", f"{(b - a) / a * 100:.0f}%"])
    return rows


def comparison_cost(n: int = 2_000_000) -> list[list]:
    """ns of the size comparison alone, by an empty loop and the same loop
    with ``if len(rows) > m_warn``."""
    from collections import deque

    rows_ = deque(range(5))
    m_warn = 8128

    def empty() -> None:
        for _ in range(n):
            pass

    def compare() -> None:
        for _ in range(n):
            if len(rows_) > m_warn:
                pass

    def evict() -> None:
        edge = -1
        dq: deque = deque([(0, 1.0), (1, 1.0)])
        for _ in range(n):
            while len(dq) > 1 and dq[1][0] <= edge:
                dq.popleft()

    e, c, v = _best(empty) / n * 1e9, _best(compare) / n * 1e9, _best(evict) / n * 1e9
    return [
        ["empty loop", f"{e:.1f}"],
        ["size comparison, `if len(rows) > m_warn`", f"{c - e:.1f}"],
        [
            "eviction test with nothing to evict, `while len(rows) > 1 and rows[1][0] <= edge`",
            f"{v - e:.1f}",
        ],
    ]


def test_results() -> list[list]:
    return [[name, "pass" if ok else "FAIL"] for name, ok in tests.run_all()]
