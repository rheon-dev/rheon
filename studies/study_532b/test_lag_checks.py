"""Tests of the lag size checks: plain asserts; run with
``python -m studies.study_532b.test_lag_checks`` (also collected by pytest).
"""

from __future__ import annotations

import math
import re

import numpy as np

from .lag import Lag
from .lag_checks import (
    CAP,
    WARN,
    ClockLag,
    LagSizeError,
    check_row_lag,
    clock_lag_bytes,
    max_clock_rows,
    max_row_k,
    row_lag_bytes,
    row_lag_verdict,
)


def test_row_bytes_equal_the_saved_state() -> None:
    for k in (1, 2, 7, 8, 9, 64, 1000, 8000):
        st = Lag("shift", k)
        for i in range(k + 5):
            st.step(float(i) if i % 3 else None)
        assert len(st.to_bytes()) == row_lag_bytes(k), k
        assert row_lag_bytes(k, 4) == 4 * len(st.to_bytes())


def test_row_thresholds_are_exact_boundaries() -> None:
    for columns in (1, 2, 4, 10):
        for limit in (WARN, CAP):
            k = max_row_k(limit, columns)
            assert row_lag_bytes(k, columns) <= limit < row_lag_bytes(k + 1, columns)
    # the packed state measured at the boundary, not only the formula
    k = max_row_k(CAP)
    st = Lag("shift", k)
    for _ in range(k + 1):
        st.step(1.0)
    assert len(st.to_bytes()) <= CAP
    st = Lag("shift", k + 1)
    for _ in range(k + 2):
        st.step(1.0)
    assert len(st.to_bytes()) > CAP


def test_row_verdicts() -> None:
    k_warn, k_cap = max_row_k(WARN), max_row_k(CAP)
    assert row_lag_verdict(k_warn)[0] == "ok"  # at half the cap: not above it
    assert row_lag_verdict(k_warn + 1)[0] == "warn"
    assert row_lag_verdict(k_cap)[0] == "warn"  # at the cap: not above it
    assert row_lag_verdict(k_cap + 1)[0] == "refuse"
    assert check_row_lag(k_warn) == ""
    assert check_row_lag(k_warn + 1) != ""
    try:
        check_row_lag(k_cap + 1)
    except LagSizeError as e:
        assert str(k_cap + 1) in str(e)
    else:
        raise AssertionError("not refused")


def test_a_size_exactly_at_a_threshold_passes_it() -> None:
    b = row_lag_bytes(10)
    assert row_lag_verdict(10, 1, 2 * b)[0] == "ok"  # exactly half the cap
    assert row_lag_verdict(10, 1, b)[0] == "warn"  # exactly the cap
    assert row_lag_verdict(10, 1, b - 1)[0] == "refuse"
    m = 40
    st = ClockLag(10**9, cap=clock_lag_bytes(m))  # a cap met exactly by m rows
    for t in range(1, m + 1):
        st.step(t, 1.0)
    assert len(st.rows) == m and st.warning != ""
    try:
        st.step(m + 1, 1.0)
    except LagSizeError:
        pass
    else:
        raise AssertionError("not refused")
    st = ClockLag(10**9, cap=2 * clock_lag_bytes(m))  # half the cap met exactly
    for t in range(1, m + 1):
        st.step(t, 1.0)
    assert st.warning == ""
    st.step(m + 1, 1.0)
    assert st.warning != ""


def test_more_columns_cross_sooner() -> None:
    ks = [max_row_k(CAP, c) for c in (1, 2, 4, 8)]
    assert ks == sorted(ks, reverse=True) and len(set(ks)) == 4
    ms = [max_clock_rows(CAP, c) for c in (1, 2, 4, 8)]
    assert ms == sorted(ms, reverse=True) and len(set(ms)) == 4


def _stream(n: int, seed: int = 1, step: int = 3):
    rng = np.random.default_rng(seed)
    t = np.cumsum(rng.integers(1, step + 1, n))
    x = rng.standard_normal(n)
    return t.tolist(), [
        None if r < 0.05 else float(v) for r, v in zip(rng.random(n), x, strict=True)
    ]


def _reference(ts, xs, d):
    """The value of the latest row at or before t - d, by a plain scan."""
    out = []
    for i, t in enumerate(ts):
        v = None
        for j in range(i - 1, -1, -1):
            if ts[j] <= t - d:
                v = xs[j]
                break
        out.append(v)
    return out


def test_clock_lag_is_the_latest_row_at_or_before_the_edge() -> None:
    ts, xs = _stream(600)
    for d in (1, 7, 50, 400):
        st = ClockLag(d, cap=None)
        assert [st.step(t, x) for t, x in zip(ts, xs, strict=True)] == _reference(ts, xs, d)


def test_clock_bytes_equal_the_saved_state_and_bound_the_rows() -> None:
    ts, xs = _stream(3000)
    st = ClockLag(500, cap=None)
    for t, x in zip(ts, xs, strict=True):
        st.step(t, x)
        m = len(st.rows)
        # rows in (t - D, t] plus one anchor at or before t - D
        inside = sum(1 for u, _ in st.rows if u > t - 500)
        assert m in (inside, inside + 1)
    assert len(st.to_bytes()) == clock_lag_bytes(len(st.rows))
    assert len(st.to_bytes()) == 8 + 16 * len(st.rows) + math.ceil(len(st.rows) / 8)
    other = ClockLag(500, cap=None)
    other.load(st.to_bytes())
    assert list(other.rows) == list(st.rows)
    assert len(ClockLag(500, 4, None).to_bytes()) == 8  # empty: the header


def test_clock_thresholds_are_exact_boundaries() -> None:
    for columns in (1, 4):
        for limit in (WARN, CAP):
            m = max_clock_rows(limit, columns)
            assert clock_lag_bytes(m, columns) <= limit < clock_lag_bytes(m + 1, columns)


def test_clock_check_warns_at_half_and_refuses_at_the_cap() -> None:
    # one row a tick, so a lag of d holds d + 1 rows (d in the window, one anchor)
    d = 10**6
    st = ClockLag(d)
    m_warn, m_cap = max_clock_rows(WARN), max_clock_rows(CAP)
    for t in range(1, m_warn + 1):
        st.step(t, 1.0)
    assert st.warning == "" and len(st.rows) == m_warn
    st.step(m_warn + 1, 1.0)
    assert st.warning != ""
    first = st.warning
    for t in range(m_warn + 2, m_cap + 1):
        st.step(t, 1.0)
    assert st.warning == first  # said once
    assert len(st.rows) == m_cap and len(st.to_bytes()) <= CAP
    try:
        st.step(m_cap + 1, 1.0)
    except LagSizeError as e:
        msg = str(e)
    else:
        raise AssertionError("not refused")
    assert len(st.rows) == m_cap and len(st.to_bytes()) <= CAP  # the refused row not kept
    assert "bucketed approximation of the lag" in msg


def test_a_burst_that_passes_is_not_refused() -> None:
    # many rows in one day, then the day moves on: the state shrinks again
    st = ClockLag(1000, cap=4096)
    m_cap = st.m_cap
    for i in range(m_cap):
        st.step(i // 10, 1.0)  # up to 10 rows a tick, all inside the window
    assert len(st.rows) <= m_cap
    st.step(5000, 1.0)  # a quiet day after the burst
    assert len(st.rows) <= 2


def test_messages_cite_no_plan() -> None:
    bad = re.compile(r"PLAN|task|docs/|\bTODO\b|§|section", re.I)
    msgs = [check_row_lag(max_row_k(CAP) + 0) or "x"]
    try:
        check_row_lag(max_row_k(CAP) + 1)
    except LagSizeError as e:
        msgs.append(str(e))
    st = ClockLag(10**6, cap=1024)
    try:
        for t in range(1, 10**4):
            st.step(t, 1.0)
    except LagSizeError as e:
        msgs.append(str(e))
    msgs.append(st.warning)
    assert all(m and not bad.search(m) for m in msgs), msgs
    assert "bucketed" in msgs[2]


def run_all() -> list[tuple[str, bool]]:
    out = []
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                out.append((name, True))
            except AssertionError:
                out.append((name, False))
    return out


if __name__ == "__main__":
    res = run_all()
    for name, ok in res:
        print(("pass " if ok else "FAIL ") + name)
    raise SystemExit(0 if all(ok for _, ok in res) else 1)
