"""Study 490 and 489: an online STL/MSTL behind an embargo, and Fourier seasonality terms.

Two numpy studies (the record is docs/records/STUDY-490-489.md).

**489, Fourier terms.** For a period ``P`` and harmonics ``i = 1..order`` the
features of a row with clock ``t`` are::

    sin(2 pi i (t - t0) / P),   cos(2 pi i (t - t0) / P)

statsmodels' ``Fourier(period, order)`` takes the row's *position* ``n`` for
``(t - t0) / step`` and computes ``x = 2 pi (n / period)`` then ``sin((i) x)``;
``CalendarFourier(freq, order)`` takes the fraction of the calendar period
(day, week, month, quarter, year) that has elapsed at the timestamp,
``(t - start) / (next_start - start)``. The study computes both from an integer
nanosecond clock, as rheon reads a temporal clock, and compares to the bit.

**490, STL and MSTL behind an embargo.** STL (Cleveland, Cleveland, McRae and
Terpenning 1990) splits ``y = T + S + R`` with an inner loop of loess smoothers
and an optional outer loop of robustness weights. The online form keeps a ring
of the last ``W`` rows; at row ``t`` it runs STL on the ring and emits the
components of row ``t - h`` (the embargo ``h``), so an output reads no row after
its own emission row. MSTL (Bandara, Hyndman and Bergmeir 2021) runs STL once
per season, shortest period first, and iterates. The oracle is statsmodels'
``STL`` / ``MSTL`` fitted on exactly the ring's rows, read at position
``W - 1 - h``.

The STL port here is written from the paper's definitions (loess with ``q``
nearest neighbours and tricube weights, cycle-subseries smoothing extended one
point each way, the low-pass filter of three moving averages and a loess,
bisquare robustness weights from ``6 median |R|``), with the conventions of
the authors' Fortran that statsmodels follows where the paper leaves a choice
(the record lists each). numpy only for the port; statsmodels, pandas and
mpmath are imported only in the comparison functions.

Data are generated from seeded numpy generators; nothing is read from disk.

Run (the study's venv holds numpy, scipy, pandas, statsmodels and mpmath; the
record says how it was made)::

    .venv-490/bin/python scripts/study_490_489.py part1 --out <dir>   (Fourier, 489)
    .venv-490/bin/python scripts/study_490_489.py part2 --out <dir>   (STL vs statsmodels)
    .venv-490/bin/python scripts/study_490_489.py part3 --out <dir>   (embargo, state, robust)
    .venv-490/bin/python scripts/study_490_489.py part4 --out <dir>   (MSTL)

Each part writes <dir>/<part>.json; a rerun prints the cached result unless
--force is given.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from fractions import Fraction

import numpy as np

NS = 1_000_000_000  # nanoseconds per second
TWO53 = 2**53

# --------------------------------------------------------------------------- ulps


def ulps(a, b):
    """|a - b| in units of the last place of the larger magnitude (0 where equal)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    d = np.abs(a - b)
    sp = np.spacing(np.maximum(np.abs(a), np.abs(b)))
    out = np.where(d == 0, 0.0, d / np.where(sp == 0, np.finfo(float).tiny, sp))
    return out


def gap_stats(mine, oracle, scale=None):
    """Largest absolute gap, in ulps of the value, and relative to ``scale``."""
    mine = np.asarray(mine, float)
    oracle = np.asarray(oracle, float)
    d = np.abs(mine - oracle)
    out = {
        "n": int(d.size),
        "bit_equal": int(np.sum(mine.view(np.int64) == oracle.view(np.int64))),
        "max_abs": float(d.max()) if d.size else 0.0,
        "max_ulps": float(ulps(mine, oracle).max()) if d.size else 0.0,
        # ulps of 1: the gap in units of 2^-52, the spacing at the features' scale
        "max_ulps_of_1": float(d.max() / 2.0**-52) if d.size else 0.0,
    }
    if scale is not None:
        out["max_rel_scale"] = float(d.max() / scale) if d.size else 0.0
        out["scale"] = float(scale)
    return out


# =========================================================================== 489
# Fourier terms from an integer nanosecond clock.


def fourier_terms(x, order, sin=np.sin, cos=np.cos):
    """Columns sin(i x), cos(i x), i = 1..order, from the base angle x = 2 pi u.

    The multiplication ``(i) * x`` is done after ``x = 2 pi u`` is rounded, as
    statsmodels' ``_get_terms`` does it, so the two can agree to the bit.
    """
    out = np.empty((x.shape[0], 2 * order))
    for i in range(order):
        out[:, 2 * i] = sin((i + 1) * x)
        out[:, 2 * i + 1] = cos((i + 1) * x)
    return out


def fourier_clock(t_ns, origin_ns, period_ns, order):
    """The clock form, unreduced: u = f64(t - t0) / f64(P), x = 2 pi u.

    ``t - t0`` is taken in integers and converted once; it is exact while
    ``|t - t0| < 2^53`` ns (104.2 days), and ``P`` is exact below 2^53 ns.
    """
    d = (np.asarray(t_ns, np.int64) - np.int64(origin_ns)).astype(np.float64)
    u = d / float(period_ns)
    return fourier_terms(2 * np.pi * u, order)


def fourier_reduced(t_ns, origin_ns, period_ns, order):
    """The clock form, reduced in integers first, per harmonic.

    r_i = (i (t - t0)) mod P in exact integers, u_i = f64(r_i) / f64(P),
    term = sin(2 pi u_i). The angle never exceeds 2 pi, so its rounding does
    not grow with the stream's length.
    """
    d = np.asarray(t_ns, np.int64) - np.int64(origin_ns)
    P = np.int64(period_ns)
    base = np.mod(d, P)  # floor mod: in [0, P)
    out = np.empty((d.shape[0], 2 * order))
    for i in range(order):
        r = np.mod(base * np.int64(i + 1), P)  # base < P <= 2^62 / order
        x = 2 * np.pi * (r.astype(np.float64) / float(P))
        out[:, 2 * i] = np.sin(x)
        out[:, 2 * i + 1] = np.cos(x)
    return out


# ---------------------------------------------------------------- calendar periods


def _floor_div(a, b):
    return np.floor_divide(a, b)


def calendar_bounds(t, unit, freq):
    """Start and next start of the calendar period holding each timestamp.

    ``t``: int64 counts of ``unit`` ('ns' or 'us') since the Unix epoch, UTC
    or naive wall time. Civil-date arithmetic in integers: an hour and a day
    are fixed counts; a week runs Monday to Sunday (pandas' ``W-SUN``: weeks
    *ending* Sunday); months, quarters (``QE-DEC``) and years (``YE-DEC``)
    come from numpy's calendar units, so leap years and month lengths are
    exact.
    """
    per_s = {"ns": NS, "us": 1_000_000}[unit]
    hour = 3600 * per_s
    day = 86400 * per_s
    t = np.asarray(t, np.int64)
    if freq == "h":
        s = _floor_div(t, hour) * hour
        return s, s + hour
    if freq == "D":
        s = _floor_div(t, day) * day
        return s, s + day
    if freq == "W":
        d = _floor_div(t, day)
        sd = d - np.mod(d + 3, 7)  # 1970-01-01 is a Thursday: Monday = (d + 3) % 7 == 0
        return sd * day, (sd + 7) * day
    dt = t.astype(f"datetime64[{unit}]")
    if freq in ("ME", "QE"):
        m = dt.astype("datetime64[M]").astype(np.int64)  # months since 1970-01
        k = 3 if freq == "QE" else 1
        m0 = m - np.mod(m, k)
        s = m0.astype("datetime64[M]").astype(f"datetime64[{unit}]").astype(np.int64)
        e = (m0 + k).astype("datetime64[M]").astype(f"datetime64[{unit}]").astype(np.int64)
        return s, e
    if freq == "YE":
        y = dt.astype("datetime64[Y]").astype(np.int64)
        s = y.astype("datetime64[Y]").astype(f"datetime64[{unit}]").astype(np.int64)
        e = (y + 1).astype("datetime64[Y]").astype(f"datetime64[{unit}]").astype(np.int64)
        return s, e
    raise ValueError(freq)


def calendar_fourier(t, unit, freq, order):
    """ratio = f64(t - start) / f64(next - start); then the terms at 2 pi ratio."""
    s, e = calendar_bounds(t, unit, freq)
    ratio = (t - s).astype(np.float64) / (e - s).astype(np.float64)
    return fourier_terms(2 * np.pi * ratio, order), ratio


# ------------------------------------------------------------- resolution and drift


def _phase_error_rad(u_double_angle, exact_cycles, mp):
    """|x - 2 pi u_exact| reduced modulo 2 pi, in radians, computed in mpmath."""
    x = mp.mpf(u_double_angle)
    xe = 2 * mp.pi * mp.mpf(exact_cycles.numerator) / exact_cycles.denominator
    d = (x - xe) % (2 * mp.pi)
    d = min(d, 2 * mp.pi - d)
    return float(d)


REPS = ["position_f64", "ns_since_origin_f64", "epoch_ns_f64", "epoch_seconds_f64", "int_reduced"]


def drift_table(scenarios, decades, per_decade=16, seed=4890):
    """The argument error of harmonic 1 under five clock representations: the
    largest over ``per_decade`` row counts drawn in each decade [10^k, 10^(k+1))."""
    import mpmath as mp

    mp.mp.dps = 60
    rng = np.random.default_rng(seed)
    rows = []
    for sc in scenarios:
        step, P, T0 = sc["step_ns"], sc["period_ns"], sc["t0_ns"]
        for k in decades:
            if T0 + 10 ** (k + 1) * step > 9_200_000_000_000_000_000:
                continue
            ns = [int(v) for v in rng.integers(10**k, 10 ** (k + 1), per_decade)]
            worst = dict.fromkeys(REPS, 0.0)
            for n in ns:
                for key, val in _drift_one(n, step, P, T0, mp).items():
                    worst[key] = max(worst[key], val)
            rows.append(
                {
                    "scenario": sc["name"],
                    "decade": k,
                    "span_days_max": 10 ** (k + 1) * step / (86400 * NS),
                    **worst,
                }
            )
    return rows


def _drift_one(n, step, P, T0, mp):
    two_pi = 2 * np.pi
    period_rows = P / step  # statsmodels' period in rows (exact for these scenarios)
    ex_origin = Fraction(n * step, P)  # cycles since the origin
    ex_epoch = Fraction(T0 + n * step, P)  # cycles since the epoch
    reps = {}
    # statsmodels' position form: x = 2 pi (n / period)
    u = float(n) / period_rows
    reps["position_f64"] = _phase_error_rad(two_pi * u, ex_origin, mp)
    # ns since the origin, as f64
    u = float(n * step) / float(P)
    reps["ns_since_origin_f64"] = _phase_error_rad(two_pi * u, ex_origin, mp)
    # epoch ns as f64, phase from the epoch
    u = float(T0 + n * step) / float(P)
    reps["epoch_ns_f64"] = _phase_error_rad(two_pi * u, ex_epoch, mp)
    # epoch seconds as f64 (the clock divided by 1e9 first), phase from the epoch
    s = (T0 + n * step) / NS  # Python int / int: correctly rounded
    u = s / (P / NS)
    reps["epoch_seconds_f64"] = _phase_error_rad(two_pi * u, ex_epoch, mp)
    # integer-reduced from the epoch: ((T0 + n step) mod P) / P
    r = (T0 + n * step) % P
    u = float(r) / float(P)
    reps["int_reduced"] = _phase_error_rad(two_pi * u, ex_epoch, mp)
    return reps


def part1(args):
    import pandas as pd
    from statsmodels.tsa.deterministic import CalendarFourier, Fourier

    res = {"versions": _versions()}
    # ---- (a) even clocks, Fourier(period, order) by position against the clock form
    even = []
    cases = [
        # (period in rows, order, step ns, rows)
        (7.0, 3, 86400 * NS, 20_000),
        (24.0, 6, 3600 * NS, 200_000),
        (12.5, 4, 3600 * NS, 200_000),
        (168.0, 10, 3600 * NS, 200_000),
        (365.25, 10, 86400 * NS, 40_000),
        (1440.0, 8, 60 * NS, 200_000),
        (60.0, 5, 1, 200_000),  # a 1 ns step
        (1440.0, 8, 60 * NS, 525_600),  # one year of minutes: crosses 2^53 ns at day 104.2
        (360.0, 4, 10 * NS + 1, 1_000_000),  # an odd step: t - t0 needs 54 bits past row 900,720
    ]
    for period, order, step, nrows in cases:
        P = period * step
        assert float(P).is_integer()
        P = int(P)
        t0 = 1_767_225_600 * NS  # 2026-01-01T00:00Z
        t = t0 + np.arange(nrows, dtype=np.int64) * np.int64(step)
        sm = Fourier(period, order).in_sample(pd.RangeIndex(nrows)).to_numpy()
        mine = fourier_clock(t, t0, P, order)
        red = fourier_reduced(t, t0, P, order)
        d_ns = t - t0
        exact_rows = d_ns < TWO53
        neq = np.any(mine.view(np.int64) != sm.view(np.int64), axis=1)
        first_neq = int(np.argmax(neq)) if neq.any() else None
        even.append(
            {
                "period_rows": period,
                "order": order,
                "step_ns": step,
                "rows": nrows,
                "span_days": float(d_ns[-1]) / (86400 * NS),
                "rows_below_2^53ns": int(exact_rows.sum()),
                "clock_vs_sm": gap_stats(mine, sm),
                "clock_vs_sm_rows_below_2^53ns": gap_stats(mine[exact_rows], sm[exact_rows]),
                "first_row_not_bit_equal": first_neq,
                "first_row_not_bit_equal_days": (
                    float(d_ns[first_neq]) / (86400 * NS) if first_neq is not None else None
                ),
                "reduced_vs_sm": gap_stats(red, sm),
            }
        )
    res["even"] = even

    # ---- (a2) accuracy against an exact reference (mpmath) at sampled rows
    import mpmath as mp

    mp.mp.dps = 50
    acc = []
    for period, order, step, nrows in [
        (24.0, 6, 3600 * NS, 200_000),
        (1440.0, 8, 60 * NS, 525_600),
    ]:
        P = int(period * step)
        t0 = 1_767_225_600 * NS
        idx = np.unique(np.r_[np.linspace(0, nrows - 1, 400).astype(np.int64), nrows - 1])
        t = t0 + idx * np.int64(step)
        sm = Fourier(period, order).in_sample(pd.RangeIndex(nrows)).to_numpy()[idx]
        red = fourier_reduced(t, t0, P, order)
        ref = np.empty_like(sm)
        for r, n in enumerate(idx):
            for i in range(order):
                x = 2 * mp.pi * (i + 1) * mp.mpf(int(n)) / mp.mpf(period)
                ref[r, 2 * i] = float(mp.sin(x))
                ref[r, 2 * i + 1] = float(mp.cos(x))
        acc.append(
            {
                "period_rows": period,
                "order": order,
                "rows": nrows,
                "sm_vs_exact": gap_stats(sm, ref),
                "reduced_vs_exact": gap_stats(red, ref),
                "sm_vs_exact_last_row": gap_stats(sm[-1], ref[-1]),
            }
        )
    res["accuracy"] = acc

    # ---- (b) irregular clocks: statsmodels on the even grid that holds the clock
    rng = np.random.default_rng(489)
    irr = []
    G = 1_000_000  # grid rows (1 s apart): 11.6 days
    for period, order in [(86400.0, 4), (3600.0, 6), (604800.0, 3), (1000.5, 3)]:
        pos = np.sort(rng.choice(G, size=50_000, replace=False)).astype(np.int64)
        grid = Fourier(period, order).in_sample(pd.RangeIndex(G)).to_numpy()
        sm = grid[pos]
        priv = Fourier(period, order)._get_terms(pos / period)
        t0 = 1_767_225_600 * NS + 123_456_789
        t = t0 + pos * NS
        P = int(period * NS)
        mine = fourier_clock(t, t0, P, order)
        irr.append(
            {
                "grid": "1 s",
                "period_rows": period,
                "order": order,
                "rows": int(pos.size),
                "clock_vs_sm_grid": gap_stats(mine, sm),
                "get_terms_vs_in_sample_grid": gap_stats(priv, sm),
            }
        )
        del grid
    # nanosecond-irregular clock over 100 days: the grid is 1 ns, too long to build,
    # so the oracle is statsmodels' own _get_terms at the rows' positions on that grid
    # (in_sample on the grid is np.arange(G) / period then _get_terms, elementwise;
    # the 1 s grid above checks the two agree to the bit).
    t0 = 1_767_225_600 * NS
    span = 100 * 86400 * NS
    pos = np.sort(rng.integers(0, span, size=200_000, dtype=np.int64))
    t = t0 + pos
    for period_ns, order in [(86400 * NS, 4), (3600 * NS, 6), (1_000_003, 3)]:
        sm = Fourier(float(period_ns), order)._get_terms(pos.astype(np.float64) / float(period_ns))
        mine = fourier_clock(t, t0, period_ns, order)
        red = fourier_reduced(t, t0, period_ns, order)
        irr.append(
            {
                "grid": "1 ns",
                "period_ns": period_ns,
                "order": order,
                "rows": int(pos.size),
                "clock_vs_sm_get_terms": gap_stats(mine, sm),
                "reduced_vs_sm_get_terms": gap_stats(red, sm),
            }
        )
    res["irregular"] = irr

    # ---- (c) CalendarFourier on irregular nanosecond timestamps, ns and us indexes
    lo = np.datetime64("2019-01-01", "ns").astype(np.int64)
    hi = np.datetime64("2027-01-01", "ns").astype(np.int64)
    t = np.sort(rng.integers(lo, hi, size=200_000, dtype=np.int64))
    # boundaries: every month start, one ns either side, leap day, year ends
    starts = np.arange("2019-01", "2027-01", dtype="datetime64[M]").astype("datetime64[ns]")
    edge = np.concatenate([starts.astype(np.int64) + k for k in (-1, 0, 1)])
    extra = np.array(
        ["2024-02-29T23:59:59.999999999", "2024-12-31T23:59:59.999999999", "2020-12-31T12:00"],
        dtype="datetime64[ns]",
    ).astype(np.int64)
    t = np.unique(np.concatenate([t, edge, extra]))
    cal = []
    for unit in ("ns", "us"):
        tu = t if unit == "ns" else np.unique(np.floor_divide(t, 1000))
        idx = pd.DatetimeIndex(tu.astype(f"datetime64[{unit}]"))
        assert str(idx.dtype) == f"datetime64[{unit}]"
        for freq, order in [("h", 2), ("D", 4), ("W", 3), ("ME", 3), ("QE", 2), ("YE", 6)]:
            cf = CalendarFourier(freq, order)
            sm = cf.in_sample(idx).to_numpy()
            sm_ratio = np.asarray(cf._compute_ratio(idx), float)
            mine, ratio = calendar_fourier(tu, unit, freq, order)
            cal.append(
                {
                    "unit": unit,
                    "freq": freq,
                    "pandas_freq": cf.freq,
                    "order": order,
                    "rows": int(tu.size),
                    "ratio": gap_stats(ratio, sm_ratio),
                    "terms": gap_stats(mine, sm),
                    "ratio_max": float(ratio.max()),
                }
            )
    # the same instants in ns and in us: does the index's unit move the ratio?
    t_us = np.floor_divide(t, 1000) * 1000  # instants on whole microseconds
    unit_cmp = []
    for freq in ("D", "ME", "YE"):
        _, r_ns = calendar_fourier(t_us, "ns", freq, 1)
        _, r_us = calendar_fourier(np.floor_divide(t_us, 1000), "us", freq, 1)
        unit_cmp.append({"freq": freq, "ns_vs_us_index": gap_stats(r_ns, r_us)})
    # the ratio's own rounding: f64(t - s) / f64(e - s) against the correctly rounded
    # quotient of the exact integers (they differ only when t - s or e - s needs > 53 bits)
    cr = []
    for freq in ("D", "ME", "YE"):
        s_, e_ = calendar_bounds(t, "ns", freq)
        r2 = (t - s_).astype(np.float64) / (e_ - s_).astype(np.float64)
        exact = np.array(
            [int(a) / int(b) for a, b in zip(t - s_, e_ - s_, strict=True)]
        )  # int / int: RN
        cr.append({"freq": freq, "ratio_vs_correctly_rounded": gap_stats(r2, exact)})
    res["calendar"] = cal
    res["calendar_unit"] = unit_cmp
    res["calendar_ratio_rounding"] = cr
    # a time zone: statsmodels' ratio subtracts a naive period start from the index
    tz = {}
    try:
        idx = pd.DatetimeIndex(["2024-03-30 12:00", "2024-03-31 12:00"]).tz_localize(
            "Europe/London"
        )
        CalendarFourier("D", 1).in_sample(idx)
        tz["tz_aware"] = "ran"
    except Exception as exc:  # noqa: BLE001 - recorded as the finding
        tz["tz_aware"] = f"{type(exc).__name__}: {exc}"
    # on the spring-forward day a naive wall clock reads 23:00 as 23/24 of the day
    # where 22 of the day's 23 hours have passed
    wall = pd.DatetimeIndex(["2024-03-31 23:00"])
    tz["wall_clock_ratio_2024_03_31_23h"] = float(CalendarFourier("D", 1)._compute_ratio(wall)[0])
    tz["elapsed_fraction_same_instant_london"] = 22 / 23
    res["time_zone"] = tz

    # ---- (d) numpy's sin and cos against the platform libm (Rust's f64::sin)
    x = np.concatenate(
        [
            2 * np.pi * (np.arange(200_000) / 24.0),
            2 * np.pi * rng.uniform(0, 1, 200_000),
            rng.uniform(-1e6, 1e6, 200_000),
        ]
    )
    ls = np.array([math.sin(v) for v in x])
    lc = np.array([math.cos(v) for v in x])
    res["numpy_vs_libm"] = {"sin": gap_stats(np.sin(x), ls), "cos": gap_stats(np.cos(x), lc)}

    # ---- (e) the clock's resolution and the phase drift over a long stream
    day = 86400 * NS
    T0 = 1_767_225_600 * NS
    scenarios = [
        {"name": "1 min rows, daily period", "step_ns": 60 * NS, "period_ns": day, "t0_ns": T0},
        {"name": "1 s rows, daily period", "step_ns": NS, "period_ns": day, "t0_ns": T0},
        {
            "name": "1 h rows, weekly period",
            "step_ns": 3600 * NS,
            "period_ns": 7 * day,
            "t0_ns": T0,
        },
        {
            "name": "1 day rows, 365.2425-day period",
            "step_ns": day,
            "period_ns": 31_556_952 * NS,
            "t0_ns": T0,
        },
        {"name": "1 ms rows, 1 s period", "step_ns": 1_000_000, "period_ns": NS, "t0_ns": T0},
        {"name": "1 us rows, 1 ms period", "step_ns": 1000, "period_ns": 1_000_000, "t0_ns": T0},
    ]
    res["drift"] = drift_table(scenarios, range(3, 11))
    return res


def summarise1(res):
    print("== even clocks: clock form vs statsmodels Fourier (to the bit)")
    for r in res["even"]:
        c = r["clock_vs_sm"]
        cb = r["clock_vs_sm_rows_below_2^53ns"]
        print(
            f"  period {r['period_rows']:>7} order {r['order']:>2} step {r['step_ns']:>14} ns"
            f" rows {r['rows']:>7} ({r['span_days']:.1f} d): bit-equal {c['bit_equal']}/{c['n']},"
            f" below 2^53 ns {cb['bit_equal']}/{cb['n']}; max {c['max_abs']:.2e}"
            f" ({c['max_ulps']:.0f} ulps); first mismatch day {r['first_row_not_bit_equal_days']};"
            f" reduced vs sm max {r['reduced_vs_sm']['max_abs']:.2e}"
        )
    print("== accuracy against mpmath")
    for r in res["accuracy"]:
        print(
            f"  period {r['period_rows']} rows {r['rows']}:"
            f" statsmodels max {r['sm_vs_exact']['max_abs']:.2e}"
            f" ({r['sm_vs_exact']['max_ulps_of_1']:.0f} ulps of 1); reduced max"
            f" {r['reduced_vs_exact']['max_abs']:.2e}"
            f" ({r['reduced_vs_exact']['max_ulps_of_1']:.1f} ulps of 1)"
        )
    print("== irregular clocks")
    for r in res["irregular"]:
        key = "clock_vs_sm_grid" if r["grid"] == "1 s" else "clock_vs_sm_get_terms"
        c = r[key]
        extra = ""
        if "get_terms_vs_in_sample_grid" in r:
            g = r["get_terms_vs_in_sample_grid"]
            extra = f"; _get_terms vs in_sample {g['bit_equal']}/{g['n']}"
        if "reduced_vs_sm_get_terms" in r:
            extra = f"; reduced vs sm max {r['reduced_vs_sm_get_terms']['max_abs']:.2e}"
        print(
            f"  grid {r['grid']} period"
            f" {r.get('period_rows', r.get('period_ns'))} order {r['order']}:"
            f" bit-equal {c['bit_equal']}/{c['n']} max {c['max_abs']:.2e}{extra}"
        )
    print("== CalendarFourier")
    for r in res["calendar"]:
        print(
            f"  {r['unit']} {r['freq']:>2} ({r['pandas_freq']}) rows {r['rows']}: ratio bit-equal"
            f" {r['ratio']['bit_equal']}/{r['ratio']['n']},"
            f" terms bit-equal {r['terms']['bit_equal']}/"
            f"{r['terms']['n']} max {r['terms']['max_abs']:.2e}"
        )
    for r in res["calendar_unit"]:
        c = r["ns_vs_us_index"]
        print(
            f"  same instants, ns vs us index, {r['freq']}: bit-equal"
            f" {c['bit_equal']}/{c['n']} max ulps {c['max_ulps']:.1f}"
        )
    for r in res["calendar_ratio_rounding"]:
        c = r["ratio_vs_correctly_rounded"]
        print(
            f"  ratio vs correctly rounded, ns, {r['freq']}: bit-equal"
            f" {c['bit_equal']}/{c['n']} max ulps {c['max_ulps']:.1f}"
        )
    print("  time zone:", res["time_zone"])
    print(
        "== numpy vs libm:",
        {k: (v["bit_equal"], v["n"], v["max_ulps"]) for k, v in res["numpy_vs_libm"].items()},
    )
    print("== drift: argument error of harmonic 1, radians")
    print("  " + " ".join(f"{k:>20}" for k in ["scenario/rows"] + REPS))
    for r in res["drift"]:
        print(
            f"  {r['scenario'][:14]:>14} 1e{r['decade']:<2}-1e{r['decade'] + 1:<2}"
            + " ".join(f"{r[k]:>20.2e}" for k in REPS)
        )


# =========================================================================== 490
# STL from the paper's definitions, with the Fortran's conventions.


def _cube(x):
    """x ** 3 through the C library's pow, as compiled code computes it (numpy's
    own power differs from glibc's pow in the last bit for about 5% of values;
    a square is x * x either way, since compilers turn pow(x, 2) into that)."""
    x = np.asarray(x, np.float64)
    flat = [math.pow(v, 3.0) if math.isfinite(v) else v**3 for v in x.ravel().tolist()]
    return np.array(flat, np.float64).reshape(x.shape)


class LoessPlan:
    """A loess evaluated at given positions, each over a window of rows.

    Positions are 1-based, as in the paper's x_i = i. For each output point
    ``xs`` with window ``[nleft, nright]`` (the ``q`` nearest rows, or all
    ``n`` rows when ``q >= n``)::

        h = max(xs - nleft, nright - xs)
        if q > n: h += (q - n) // 2              ("fortran", statsmodels)
                  h *= q / n                     ("paper": lambda_q = lambda_n q / n)
        v_j = T(|j - xs| / h)  (tricube (1 - u^3)^3 for u < 1, else 0)
              "fortran": 1 for |j - xs| <= 0.001 h, 0 above 0.999 h, and each
              cube through the C library's pow, as statsmodels' compiled code;
              "fortran_mul": the same with each cube as x * x * x
        v_j *= rho_j (robustness weights, when given)
        weights normalised to sum 1; for degree 1 (and h > 0) a weighted line:
        abar = sum v_j j, c = sum v_j (j - abar)^2,
        if sqrt(c) > 0.001 (n - 1): v_j *= (xs - abar) / c * (j - abar) + 1
        value = sum v_j y_j           (no value when sum v_j = 0)

    Sums run in row order (sequentially), as the Fortran's loops do.
    """

    def __init__(self, n, q, deg, xs, nleft, nright, rule="fortran"):
        self.n = n
        self.deg = deg
        xs = np.asarray(xs, np.float64)
        nleft = np.asarray(nleft, np.int64)
        nright = np.asarray(nright, np.int64)
        L = int((nright - nleft).max()) + 1 if xs.size else 0
        off = np.arange(L)
        j = nleft[:, None] + off[None, :]  # 1-based rows
        self.mask = j <= nright[:, None]
        self.idx = np.where(self.mask, j - 1, 0)
        self.jf = np.where(self.mask, j, 0).astype(np.float64)
        h = np.maximum(xs - nleft, nright - xs).astype(np.float64)
        if q > n:  # the Fortran adds floor((q - n) / 2); the paper scales by q / n
            h = h + float((q - n) // 2) if rule != "paper" else h * (q / n)
        self.h = h
        self.xs = xs
        r = np.abs(self.jf - xs[:, None])
        hh = h[:, None]
        if rule != "paper":
            h9 = 0.999 * hh
            h1 = 0.001 * hh
            with np.errstate(divide="ignore", invalid="ignore"):
                if rule == "fortran":
                    tri = _cube(1.0 - _cube(r / hh))
                else:  # "fortran_mul": the cube as x * x * x, as Rust's powi(3) gives it
                    u = r / hh
                    v = 1.0 - u * u * u
                    tri = v * v * v
            w = np.where(r <= h1, 1.0, tri)
            w = np.where(r <= h9, w, 0.0)
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                u = r / hh
                w = np.where(u < 1.0, _cube(1.0 - _cube(u)), 0.0)
            w = np.where(hh == 0, np.where(r == 0, 1.0, 0.0), w)
        self.w0 = np.where(self.mask, w, 0.0)
        self.rng = n - 1.0
        self._matrix = None

    def _weights(self, rw=None):
        w = self.w0 if rw is None else self.w0 * np.where(self.mask, rw[self.idx], 0.0)
        a = np.cumsum(w, axis=1)[:, -1]
        ok = a > 0
        with np.errstate(divide="ignore", invalid="ignore"):
            w = w / a[:, None]
        if self.deg > 0:
            lin = ok & (self.h > 0)
            abar = np.cumsum(w * self.jf, axis=1)[:, -1]
            b = self.xs - abar
            dev = np.where(self.mask, self.jf - abar[:, None], 0.0)
            c = np.cumsum(w * dev**2, axis=1)[:, -1]
            with np.errstate(divide="ignore", invalid="ignore"):
                go = lin & (np.sqrt(c) > 0.001 * self.rng)
                bc = np.where(go, b / c, 0.0)
            w = np.where(go[:, None], w * (bc[:, None] * dev + 1.0), w)
        return w, ok

    def apply(self, y, rw=None):
        """Values at the plan's points; NaN where every weight is zero."""
        if y.ndim == 2:  # a batch of columns (the operator), no robustness weights
            assert rw is None
            return self.matrix() @ y
        w, ok = self._weights(rw)
        v = np.cumsum(w * np.where(self.mask, y[self.idx], 0.0), axis=1)[:, -1]
        return np.where(ok, v, np.nan)

    def matrix(self):
        if self._matrix is None:
            w, ok = self._weights(None)
            assert ok.all()
            M = np.zeros((self.xs.size, self.n))
            rows = np.repeat(np.arange(self.xs.size), self.idx.shape[1])
            np.add.at(M, (rows, self.idx.ravel()), np.where(self.mask, w, 0.0).ravel())
            self._matrix = M
        return self._matrix


class Ess:
    """The loess of a series of n rows at every row, or every ``jump``-th row
    with linear interpolation between (the paper's computational shortcut).
    A point whose weights all vanish takes the series' own value there."""

    def __init__(self, n, q, deg, jump, rule="fortran"):
        self.n = n
        self.trivial = n < 2
        if self.trivial:
            return
        nj = min(jump, n - 1)
        self.nj = nj
        if q >= n:
            pts = np.arange(0, n, nj)
            nl = np.ones_like(pts)
            nr = np.full_like(pts, n)
        elif nj == 1:
            nsh = (q + 2) // 2
            pts = np.arange(n)
            nl = np.clip(pts + 1 - nsh + 1, 1, n - q + 1)
            nl = np.where(pts + 1 > nsh, nl, 1)
            nr = nl + q - 1
        else:
            nsh = (q + 1) // 2
            pts = np.arange(0, n, nj)
            nl = np.where(
                pts + 1 < nsh, 1, np.where(pts + 1 >= n - nsh + 1, n - q + 1, pts + 1 - nsh + 1)
            )
            nr = nl + q - 1
        self.last = None
        if nj > 1:
            k = ((n - 1) // nj) * nj + 1
            if k != n:
                # the last row is fitted with the window the loop left behind
                self.last = (k, int(nl[-1]), int(nr[-1]))
                pts_all = np.r_[pts, n - 1]
                nl = np.r_[nl, nl[-1]]
                nr = np.r_[nr, nr[-1]]
                pts = pts_all
        self.pts = pts
        self.plan = LoessPlan(n, q, deg, pts + 1, nl, nr, rule)
        self._matrix = None

    def apply(self, y, rw=None):
        n = self.n
        if self.trivial:
            return y.copy()
        if y.ndim == 2:
            return self.matrix() @ y
        v = self.plan.apply(y, rw)
        v = np.where(np.isnan(v), y[self.pts], v)
        if self.nj == 1:
            return v
        nj = self.nj
        ys = np.empty(n)
        npts = self.pts.size - (1 if self.last else 0)
        ys[self.pts[:npts]] = v[:npts]
        i = 0
        while i < n - nj:
            delta = (ys[i + nj] - ys[i]) / nj
            for j in range(i, i + nj):
                ys[j] = ys[i] + delta * ((j + 1) - (i + 1))
            i += nj
        if self.last:
            k = self.last[0]
            ys[n - 1] = v[-1]
            if k != n - 1:
                delta = (ys[n - 1] - ys[k - 1]) / (n - k)
                for j in range(k, n):
                    ys[j] = ys[k - 1] + delta * ((j + 1) - k)
        return ys

    def matrix(self):
        if self._matrix is None:
            n = self.n
            if self.nj == 1:
                self._matrix = self.plan.matrix()
            else:
                # linear in y: build column by column through the scalar path
                self._matrix = np.column_stack([self.apply(e) for e in np.eye(n)])
        return self._matrix


def moving_average(x, length):
    """The running mean of ``length`` rows: v = sum of the first ``length`` rows,
    then v += x[k] - x[m] one row at a time, each divided by ``length``.
    np.cumsum is a sequential sum, so this is the Fortran's order exactly."""
    d = x[length:] - x[:-length]
    v0 = np.cumsum(x[:length], axis=0)[-1]
    v = np.cumsum(np.concatenate([v0[None, ...], d], axis=0), axis=0)
    return v / float(length)


class STLPort:
    """STL on a fixed number of rows n (the ring), from the paper's definitions.

    Inner loop, k = 1..inner (T starts at 0, robustness weights rho at 1):
      1. detrend:            D = Y - T
      2. cycle-subseries:    each of the ``period`` subseries of D is smoothed
                             by loess (span ``seasonal``, degree
                             ``seasonal_deg``, weights rho) and extended by one
                             point before its first and after its last row,
                             giving C of n + 2 period rows
      3. low-pass of C:      moving averages of period, period and 3 rows, then
                             loess (span ``low_pass``): L, n rows
      4. seasonal:           S = C[period : period + n] - L
      5. deseasonalise:      Y - S
      6. trend:              T = loess(Y - S), span ``trend``, weights rho
    Outer loop: R = Y - T - S, h = 6 median |R|,
      rho = B(|R| / h), B(u) = (1 - u^2)^2 for u < 1, else 0
      ("fortran": 1 for |R| <= 0.001 h, 0 above 0.999 h).
    The fit runs the inner loop outer + 1 times; rho is updated between runs,
    so the reported rho is the one the last inner loop used.
    """

    def __init__(
        self,
        n,
        period,
        seasonal=7,
        trend=None,
        low_pass=None,
        seasonal_deg=1,
        trend_deg=1,
        low_pass_deg=1,
        seasonal_jump=1,
        trend_jump=1,
        low_pass_jump=1,
        rule="fortran",
    ):
        if trend is None:
            trend = int(np.ceil(1.5 * period / (1 - 1.5 / seasonal)))
            trend += (trend % 2) == 0
        if low_pass is None:
            low_pass = period + 1
            low_pass += (low_pass % 2) == 0
        self.n, self.period = n, period
        self.cfg = dict(
            period=period,
            seasonal=seasonal,
            trend=trend,
            low_pass=low_pass,
            seasonal_deg=seasonal_deg,
            trend_deg=trend_deg,
            low_pass_deg=low_pass_deg,
            seasonal_jump=seasonal_jump,
            trend_jump=trend_jump,
            low_pass_jump=low_pass_jump,
        )
        self.rule = rule
        # cycle-subseries plans, one per phase j
        self.sub = []
        for j in range(period):
            k = (n - (j + 1)) // period + 1
            ess = Ess(k, seasonal, seasonal_deg, seasonal_jump, rule)
            left = LoessPlan(k, seasonal, seasonal_deg, [0], [1], [min(seasonal, k)], rule)
            right = LoessPlan(
                k, seasonal, seasonal_deg, [k + 1], [max(1, k - seasonal + 1)], [k], rule
            )
            self.sub.append((j, k, ess, left, right))
        self.lowpass = Ess(n, low_pass, low_pass_deg, low_pass_jump, rule)
        self.trendess = Ess(n, trend, trend_deg, trend_jump, rule)

    def _subseries(self, D, rw):
        p = self.period
        n = self.n
        C = np.empty((n + 2 * p,) + D.shape[1:])
        for j, k, ess, left, right in self.sub:
            yj = D[j::p][:k]
            rj = None if rw is None else rw[j::p][:k]
            mid = ess.apply(yj, rj)
            if D.ndim == 2:
                lo = left.apply(yj)
                hi = right.apply(yj)
            else:
                lo = left.apply(yj, rj)
                hi = right.apply(yj, rj)
                lo = np.where(np.isnan(lo), mid[:1], lo)
                hi = np.where(np.isnan(hi), mid[-1:], hi)
            col = np.concatenate([lo, mid, hi], axis=0)
            C[j::p][: k + 2] = col
        return C

    def _inner(self, Y, T, rw):
        p = self.period
        n = self.n
        D = Y - T
        C = self._subseries(D, rw)
        m = moving_average(moving_average(moving_average(C, p), p), 3)
        L = self.lowpass.apply(m)
        S = C[p : p + n] - L
        T = self.trendess.apply(Y - S, rw)
        return S, T

    def fit(self, y, inner=None, outer=None, robust=False):
        if inner is None:
            inner = 2 if robust else 5
        if outer is None:
            outer = 15 if robust else 0
        Y = np.asarray(y, np.float64)
        T = np.zeros_like(Y)
        S = np.zeros_like(Y)
        rw = None
        k = 0
        while True:
            for _ in range(inner):
                S, T = self._inner(Y, T, rw)
            k += 1
            if k > outer:
                break
            assert Y.ndim == 1, "the robust fit is not linear"
            rw = robustness_weights(Y - (T + S), self.rule)
        return {
            "seasonal": S,
            "trend": T,
            "resid": Y - S - T,
            "weights": np.ones(self.n) if rw is None else rw,
        }

    def operator(self, inner=5):
        """The linear maps y -> S and y -> T (no robustness): n x n each."""
        out = self.fit(np.eye(self.n), inner=inner, outer=0)
        return out["seasonal"], out["trend"]


def robustness_weights(R, rule="fortran"):
    r = np.abs(R)
    n = r.size
    mid0 = n // 2
    mid1 = n - mid0 - 1
    part = np.partition(r, [mid0, mid1])
    h = 3.0 * (part[mid0] + part[mid1])  # 6 median |R|
    if h == 0:
        return np.ones(n)
    if rule != "paper":
        w = np.where(r <= 0.999 * h, (1.0 - (r / h) ** 2) ** 2, 0.0)
        return np.where(r <= 0.001 * h, 1.0, w)
    u = r / h
    return np.where(u < 1.0, (1.0 - u**2) ** 2, 0.0)


def mstl_port(
    y,
    periods,
    windows,
    iterate=2,
    stl_kwargs=None,
    ports=None,
    inner=None,
    outer=None,
    robust=False,
):
    """MSTL from Bandara et al. (2021): seasons in ascending period; for each
    pass and each season i, add S_i back, run STL with period_i and seasonal
    window_i, take its seasonal as the new S_i and remove it. One pass when
    there is one season. Trend is the last STL's trend; R = deseasonalised - T.
    """
    stl_kwargs = stl_kwargs or {}
    n = y.shape[0]
    order = np.argsort(periods)
    periods = [periods[i] for i in order]
    windows = [windows[i] for i in order]
    if ports is None:
        ports = [
            STLPort(n, p, seasonal=w, **stl_kwargs) for p, w in zip(periods, windows, strict=True)
        ]
    seasonal = [np.zeros_like(y, dtype=np.float64) for _ in periods]
    deseas = np.asarray(y, np.float64)
    its = 1 if len(periods) == 1 else iterate
    fit = None
    for _ in range(its):
        for i in range(len(periods)):
            deseas = deseas + seasonal[i]
            fit = ports[i].fit(deseas, inner=inner, outer=outer, robust=robust)
            seasonal[i] = fit["seasonal"]
            deseas = deseas - seasonal[i]
    return {
        "seasonal": seasonal,
        "trend": fit["trend"],
        "resid": deseas - fit["trend"],
        "weights": fit["weights"],
        "periods": periods,
        "windows": windows,
    }


class OnlineSTL:
    """A ring of the last W learned rows; at each learned row, once W rows are
    held, STL on the ring and the components of row t - h (position W - 1 - h).

    A row of weight 0 or with a null target is not pushed: it learns nothing
    and emits nothing (statsmodels' STL takes no observation weights, so a
    weight has no oracle; the ring is by position of learned rows). The state
    is the ring (W doubles), the head and the count.
    """

    def __init__(self, W, h, port, kernel=False, inner=None, outer=None, robust=False):
        assert 0 <= h < W
        self.W, self.h, self.port = W, h, port
        self.ring = np.zeros(W)
        self.head = 0
        self.count = 0
        self.inner, self.outer, self.robust = inner, outer, robust
        self.kernel = None
        if kernel:
            S, T = port.operator(inner=5 if inner is None else inner)
            p = W - 1 - h
            self.kernel = (S[p].copy(), T[p].copy())

    def window(self):
        return np.concatenate([self.ring[self.head :], self.ring[: self.head]])

    def push(self, y, w=1.0):
        if w == 0 or not math.isfinite(y):
            return None
        self.ring[self.head] = y
        self.head = (self.head + 1) % self.W
        self.count += 1
        if self.count < self.W:
            return None
        win = self.window()
        p = self.W - 1 - self.h
        if self.kernel is not None:
            s_ = float(self.kernel[0] @ win)
            t_ = float(self.kernel[1] @ win)
            return {"seasonal": s_, "trend": t_, "resid": win[p] - s_ - t_}
        fit = self.port.fit(win, inner=self.inner, outer=self.outer, robust=self.robust)
        return {k: float(fit[k][p]) for k in ("seasonal", "trend", "resid")}


STL2_CASES = [
    # name, period, W, STL settings
    ("p24 W96 defaults", 24, 96, {}),
    ("p24 W168 seasonal13", 24, 168, {"seasonal": 13}),
    ("p12 W72 degrees 0", 12, 72, {"seasonal_deg": 0, "trend_deg": 0, "low_pass_deg": 0}),
    ("p24 W120 jumps 3/5/4", 24, 120, {"seasonal_jump": 3, "trend_jump": 5, "low_pass_jump": 4}),
    ("p7 W56 seasonal11 trend15", 7, 56, {"seasonal": 11, "trend": 15}),
    ("p52 W260 seasonal7", 52, 260, {}),
]


def _sm_stl(win, period, kw, robust=False, inner=None, outer=None):
    from statsmodels.tsa.seasonal import STL

    r = STL(win, period=period, robust=robust, **kw).fit(inner_iter=inner, outer_iter=outer)
    return {
        "seasonal": np.asarray(r.seasonal),
        "trend": np.asarray(r.trend),
        "resid": np.asarray(r.resid),
        "weights": np.asarray(r.weights),
    }


def _cmp_fit(mine, oracle, comps=("seasonal", "trend", "resid")):
    return {c: gap_stats(mine[c], oracle[c]) for c in comps}


def _merge(acc, new):
    """Fold gap_stats over many windows: counts add, maxima take the max."""
    for c, st in new.items():
        if c not in acc:
            acc[c] = dict(st)
            continue
        a = acc[c]
        a["n"] += st["n"]
        a["bit_equal"] += st["bit_equal"]
        for k in ("max_abs", "max_ulps", "max_ulps_of_1", "max_rel_scale"):
            if k in st:
                a[k] = max(a[k], st[k])
    return acc


def part2(args):
    rows_after = args.stl_rows
    out = {"versions": _versions(), "cases": []}
    for ci, (name, period, W, kw) in enumerate(STL2_CASES):
        N = W + rows_after
        y = make_series(N, [period], seed=4900 + ci)
        port = STLPort(W, period, **kw)
        cfg = dict(port.cfg)
        nt = cfg["trend"]
        h = (nt - 1) // 2
        scale = float(np.std(y))
        S_op, T_op = port.operator()
        mul = STLPort(W, period, rule="fortran_mul", **kw)
        paper = STLPort(W, period, rule="paper", **kw)
        acc = {"port": {}, "kernel": {}, "kernel_at_embargo": {}, "fortran_mul": {}, "paper": {}}
        online = OnlineSTL(W, h, port)
        online_k = OnlineSTL(W, h, port, kernel=True)
        emitted_equal = 0
        emitted = 0
        for t in range(W - 1, N):
            win = y[t - W + 1 : t + 1]
            orc = _sm_stl(win, period, kw)
            for key, st in _cmp_fit(port.fit(win), orc).items():
                st["max_rel_scale"] = st["max_abs"] / scale
                _merge(acc["port"], {key: st})
            kern = {"seasonal": S_op @ win, "trend": T_op @ win}
            kern["resid"] = win - kern["seasonal"] - kern["trend"]
            g = _cmp_fit(kern, orc)
            for st in g.values():
                st["max_rel_scale"] = st["max_abs"] / scale
            _merge(acc["kernel"], g)
            if (t - W + 1) % args.variant_stride == 0:
                for key, p_ in (("fortran_mul", mul), ("paper", paper)):
                    g = _cmp_fit(p_.fit(win), orc)
                    for st in g.values():
                        st["max_rel_scale"] = st["max_abs"] / scale
                    _merge(acc[key], g)
        # the online class, row by row, against the port and the oracle at t - h
        for t in range(N):
            e = online.push(float(y[t]))
            ek = online_k.push(float(y[t]))
            if e is None:
                continue
            emitted += 1
            win = y[t - W + 1 : t + 1]
            orc = _sm_stl(win, period, kw)
            p = W - 1 - h
            same = all(e[c] == orc[c][p] for c in ("seasonal", "trend", "resid"))
            emitted_equal += same
            g = {
                c: gap_stats(np.array([ek[c]]), orc[c][p : p + 1], scale)
                for c in ("seasonal", "trend")
            }
            _merge(acc["kernel_at_embargo"], g)
        # no lookahead: rewriting every row after t leaves row t's emission unchanged
        t_chk = W + rows_after // 2
        o1 = OnlineSTL(W, h, port)
        o2 = OnlineSTL(W, h, port)
        y2 = y.copy()
        y2[t_chk + 1 :] = 1e6
        e1 = e2 = None
        for t in range(t_chk + 1):
            e1 = o1.push(float(y[t]))
            e2 = o2.push(float(y2[t]))
        out["cases"].append(
            {
                "name": name,
                "W": W,
                "rows": N,
                "windows": N - W + 1,
                "embargo_h": h,
                "cfg": cfg,
                "inner": 5,
                "outer": 0,
                "robust": False,
                "scale_sd": scale,
                "port_vs_sm": acc["port"],
                "kernel_vs_sm": acc["kernel"],
                "kernel_at_embargo_vs_sm": acc["kernel_at_embargo"],
                "fortran_mul_vs_sm": acc["fortran_mul"],
                "paper_vs_sm": acc["paper"],
                "online_emitted": emitted,
                "online_emitted_bit_equal_sm": emitted_equal,
                "no_lookahead": e1 == e2,
                "subseries_points_k": sorted({k for _, k, *_ in port.sub}),
            }
        )
    # cost per emission (one window, repeated)
    timing = []
    for name, period, W, kw in STL2_CASES:
        y = make_series(W, [period], seed=1)
        port = STLPort(W, period, **kw)
        S_op, T_op = port.operator()
        p = W - 1 - (port.cfg["trend"] - 1) // 2
        ks, kt = S_op[p].copy(), T_op[p].copy()
        timing.append(
            {
                "name": name,
                "W": W,
                "port_ms": _time(lambda port=port, y=y: port.fit(y), args.time_budget) * 1e3,
                "statsmodels_ms": _time(
                    lambda y=y, p_=period, kw=kw: _sm_stl(y, p_, kw), args.time_budget
                )
                * 1e3,
                "kernel_two_dots_us": _time(
                    lambda ks=ks, kt=kt, y=y: (ks @ y, kt @ y), args.time_budget
                )
                * 1e6,
            }
        )
    out["timing"] = timing
    return out


def _time(fn, budget):
    fn()
    n = 0
    t0 = time.perf_counter()
    while True:
        fn()
        n += 1
        el = time.perf_counter() - t0
        if el > budget:
            return el / n


def summarise2(res):
    def line(label, acc):
        parts = []
        for c in ("seasonal", "trend", "resid"):
            if c in acc:
                a = acc[c]
                parts.append(
                    f"{c[:5]} {a['bit_equal']}/{a['n']} max {a['max_abs']:.1e}"
                    f" ({a['max_ulps']:.0f} ulps, {a.get('max_rel_scale', float('nan')):.1e} sd)"
                )
        print(f"    {label:<18}" + "; ".join(parts))

    for c in res["cases"]:
        cfg = c["cfg"]
        print(
            f"== {c['name']}: W {c['W']}, {c['windows']} windows,"
            f" h {c['embargo_h']}, k {c['subseries_points_k']},"
            f" seasonal {cfg['seasonal']} trend {cfg['trend']} low_pass {cfg['low_pass']}"
            f" degs {cfg['seasonal_deg']}/{cfg['trend_deg']}/{cfg['low_pass_deg']}"
            f" jumps {cfg['seasonal_jump']}/{cfg['trend_jump']}/{cfg['low_pass_jump']}"
        )
        line("port", c["port_vs_sm"])
        line("kernel (all pos)", c["kernel_vs_sm"])
        line("kernel at t-h", c["kernel_at_embargo_vs_sm"])
        line("cube as x*x*x", c["fortran_mul_vs_sm"])
        line("paper's rule", c["paper_vs_sm"])
        print(
            f"    online: {c['online_emitted_bit_equal_sm']}/{c['online_emitted']}"
            f" emissions bit-equal;"
            f" no lookahead {c['no_lookahead']}"
        )
    for t in res["timing"]:
        print(
            f"  cost {t['name']}: port {t['port_ms']:.2f}"
            f" ms, statsmodels {t['statsmodels_ms']:.2f} ms,"
            f" kernel two dots {t['kernel_two_dots_us']:.2f} us"
        )


def _windows_matrix(y, W):
    """Column t - W + 1 holds the window ending at row t (W x (N - W + 1))."""
    from numpy.lib.stride_tricks import sliding_window_view

    return np.ascontiguousarray(sliding_window_view(y, W).T)


def part3(args):
    out = {"versions": _versions()}
    period = 24
    N = 2400
    y = make_series(N, [period], seed=4903)
    full = _sm_stl(y, period, {})
    amp = float(np.std(full["seasonal"]))
    trend_sd = float(np.std(np.diff(full["trend"])))
    nt = STLPort(10 * period, period).cfg["trend"]
    Wmax = 20 * period
    lo, hi = Wmax, N - Wmax  # rows whose full-history value is far from both ends
    emb = []
    for m in (2, 4, 7, 10, 20):
        W = m * period
        port = STLPort(W, period)
        S_op, T_op = port.operator()
        Y = _windows_matrix(y, W)  # column c: window ending at row c + W - 1
        Sw = S_op @ Y
        Tw = T_op @ Y
        ends = np.arange(W - 1, N)
        per_h = []
        for h in range(W):
            pos = W - 1 - h
            rows = ends - h
            sel = (rows >= lo) & (rows < hi)
            ds = Sw[pos, sel] - full["seasonal"][rows[sel]]
            dt = Tw[pos, sel] - full["trend"][rows[sel]]
            # revision: the same row's value from the next window (one more row seen)
            if h + 1 < W:
                nxt = Sw[pos - 1, 1:] if pos >= 1 else None
                rv = Sw[pos, :-1] - nxt
                rv = rv[sel[:-1]]
                rev = float(np.sqrt(np.mean(rv**2)))
            else:
                rev = float("nan")
            per_h.append(
                {
                    "h": h,
                    "seasonal_rms": float(np.sqrt(np.mean(ds**2))),
                    "seasonal_max": float(np.max(np.abs(ds))),
                    "trend_rms": float(np.sqrt(np.mean(dt**2))),
                    "trend_max": float(np.max(np.abs(dt))),
                    "seasonal_revision_rms": rev,
                }
            )
        best = min(per_h, key=lambda r: r["seasonal_rms"] + r["trend_rms"])
        emb.append({"m": m, "W": W, "per_h": per_h, "best_h": best["h"]})
    out["embargo"] = {
        "period": period,
        "N": N,
        "trend": nt,
        "seasonal_amp_sd": amp,
        "rows_compared": [lo, hi],
        "windows": emb,
        "trend_step_sd": trend_sd,
    }

    # influence reach of one row on another in the full fit (n = 480), central row
    n = 20 * period
    port = STLPort(n, period)
    S_op, T_op = port.operator()
    i = n // 2
    reach = {}
    for name, M in (("seasonal", S_op), ("trend", T_op)):
        row = np.abs(M[i])
        d = np.abs(np.arange(n) - i)
        tot = row.sum()
        r = {"l1": float(tot), "nonzero_max_distance": int(d[row > 0].max())}
        for tol in (1e-3, 1e-6, 1e-9, 1e-12):
            # smallest D with the mass beyond D below tol of the total
            ok = [D for D in range(n) if row[d > D].sum() <= tol * tot]
            r[f"D_{tol:g}"] = int(ok[0])
        reach[name] = r
    out["reach"] = {"n": n, "row": i, **reach}

    # robust loop and iteration counts: port against statsmodels, window by window
    rob = []
    yo = make_series(24 * 12, [24], seed=4904, outliers=0.03)
    for label, W, kw, robust, inner, outer in [
        ("robust defaults (inner 2, outer 15)", 96, {}, True, None, None),
        ("robust inner 1 outer 5, seasonal 13", 168, {"seasonal": 13}, True, 1, 5),
        (
            "robust, jumps 3/5/4",
            120,
            {"seasonal_jump": 3, "trend_jump": 5, "low_pass_jump": 4},
            True,
            None,
            None,
        ),
        ("not robust, outer 3 (weights used)", 96, {}, False, 5, 3),
        ("inner 1", 96, {}, False, 1, None),
        ("inner 2", 96, {}, False, 2, None),
        ("inner 10", 96, {}, False, 10, None),
    ]:
        port = STLPort(W, 24, **kw)
        acc = {}
        nwin = 0
        for t in range(W - 1, yo.size, args.robust_stride):
            win = yo[t - W + 1 : t + 1]
            orc = _sm_stl(win, 24, kw, robust=robust, inner=inner, outer=outer)
            mine = port.fit(win, inner=inner, outer=outer, robust=robust)
            _merge(acc, _cmp_fit(mine, orc, ("seasonal", "trend", "resid", "weights")))
            nwin += 1
        rob.append(
            {
                "label": label,
                "W": W,
                "windows": nwin,
                "vs_sm": acc,
                "port_ms": _time(
                    lambda a=(port, win, inner, outer, robust): a[0].fit(
                        a[1], inner=a[2], outer=a[3], robust=a[4]
                    ),
                    args.time_budget,
                )
                * 1e3,
                "statsmodels_ms": _time(
                    lambda a=(win, kw, robust, inner, outer): _sm_stl(a[0], 24, *a[1:]),
                    args.time_budget,
                )
                * 1e3,
            }
        )
    out["robust"] = rob
    # how far the inner loop is from converged: inner 5 against inner 50, one window
    win = y[:96]
    p5 = STLPort(96, 24).fit(win, inner=5)
    p50 = STLPort(96, 24).fit(win, inner=50)
    out["inner_convergence"] = {
        c: float(np.max(np.abs(p5[c] - p50[c]))) for c in ("seasonal", "trend")
    }
    out["inner_convergence"]["seasonal_sd"] = float(np.std(p50["seasonal"]))

    # state size
    out["state"] = state_table()
    return out


def state_table():
    """Bytes per group and target, and per spec, as functions of p and W = m p."""
    rows = []
    for p in (7, 12, 24, 52, 168, 288, 1440):
        for m in (4, 7, 10):
            W = m * p
            rows.append(
                {
                    "period": p,
                    "m": m,
                    "W": W,
                    # ring of W doubles + head + count + the last clock (a break empties the ring)
                    "ring_bytes_per_group": 8 * W + 24,
                    # kernels: seasonal and trend rows at one lag, shared by every group of a spec
                    "kernel_bytes_per_spec_one_lag": 2 * 8 * W,
                    # robust: statsmodels' scratch for one fit, freed after (not state)
                    "robust_scratch_bytes": 8 * (7 * (W + 2 * p) + 3 * W),
                }
            )
    return rows


def summarise3(res):
    e = res["embargo"]
    print(
        f"== embargo: period {e['period']}, trend"
        f" {e['trend']}, seasonal sd {e['seasonal_amp_sd']:.3f},"
        f" rows {e['rows_compared']}"
    )
    nt = e["trend"]
    for w in e["windows"]:
        W = w["W"]
        hs = sorted({0, 12, (nt - 1) // 2, nt, W // 2, W - 1, w["best_h"]} & set(range(W)))
        print(f"  W {W} ({w['m']} periods), best h {w['best_h']}:")
        for h in hs:
            r = w["per_h"][h]
            print(
                f"    h {h:>4}: seasonal rms {r['seasonal_rms']:.2e} max {r['seasonal_max']:.2e};"
                f" trend rms {r['trend_rms']:.2e} max {r['trend_max']:.2e};"
                f" revision rms {r['seasonal_revision_rms']:.2e}"
            )
    r = res["reach"]
    print(f"== reach (n {r['n']}, row {r['row']}):", {k: r[k] for k in ("seasonal", "trend")})
    for x in res["robust"]:
        parts = [
            f"{c[:5]} {a['bit_equal']}/{a['n']} max {a['max_abs']:.1e}"
            for c, a in x["vs_sm"].items()
        ]
        print(
            f"  {x['label']:<38} W {x['W']} windows {x['windows']}: "
            + "; ".join(parts)
            + f"; port {x['port_ms']:.1f} ms, sm {x['statsmodels_ms']:.2f} ms"
        )
    print("  inner 5 vs inner 50:", res["inner_convergence"])
    for s_ in res["state"]:
        if s_["m"] == 4:
            print(
                f"  p {s_['period']:>5} W {s_['W']:>5}: ring {s_['ring_bytes_per_group']}"
                f" B, kernels {s_['kernel_bytes_per_spec_one_lag']} B,"
                f" robust scratch {s_['robust_scratch_bytes']} B"
            )


def _sm_mstl(win, periods, windows, iterate=2, stl_kwargs=None, lmbda=None):
    from statsmodels.tsa.seasonal import MSTL

    r = MSTL(
        win,
        periods=periods,
        windows=windows,
        iterate=iterate,
        lmbda=lmbda,
        stl_kwargs=dict(stl_kwargs or {}),
    ).fit()
    seas = np.asarray(r.seasonal)
    if seas.ndim == 1:
        seas = seas[:, None]
    return {
        "seasonal": [seas[:, i] for i in range(seas.shape[1])],
        "trend": np.asarray(r.trend),
        "resid": np.asarray(r.resid),
        "weights": np.asarray(r.weights),
    }


def boxcox(y, lmbda):
    """(y^lambda - 1) / lambda, written as expm1(lambda log y) / lambda; log y at 0."""
    y = np.asarray(y, np.float64)
    if lmbda == 0:
        return np.log(y)
    return np.expm1(lmbda * np.log(y)) / lmbda


MSTL_CASES = [
    # name, periods, windows, W, iterate, stl kwargs (with inner/outer), lmbda, rows after, stride
    ("hourly: 24 and 168, defaults", (24, 168), None, 504, 2, {}, None, 48, 4),
    ("12 and 60, windows 7/11, iterate 3", (12, 60), (7, 11), 240, 3, {}, None, 120, 2),
    (
        "12 and 60, robust (inner 2, outer 15)",
        (12, 60),
        (7, 11),
        240,
        2,
        {"robust": True},
        None,
        60,
        4,
    ),
    ("12 and 60, Box-Cox 0.5", (12, 60), None, 240, 2, {}, 0.5, 60, 2),
    (
        "12 and 60, seasonal_deg 0, inner 3",
        (12, 60),
        (9, 13),
        240,
        2,
        {"seasonal_deg": 0, "inner_iter": 3},
        None,
        60,
        2,
    ),
]


def _mstl_port_fit(y, periods, windows, iterate, kw, lmbda, n):
    kw = dict(kw)
    inner = kw.pop("inner_iter", None)
    outer = kw.pop("outer_iter", None)
    robust = kw.pop("robust", False)
    if windows is None:
        windows = tuple(7 + 4 * i for i in range(1, len(periods) + 1))
    yy = y if lmbda is None else boxcox(y, lmbda)
    return mstl_port(
        yy,
        list(periods),
        list(windows),
        iterate=iterate,
        stl_kwargs=kw,
        inner=inner,
        outer=outer,
        robust=robust,
    )


def part4(args):
    out = {"versions": _versions(), "cases": []}
    for ci, (name, periods, windows, W, iterate, kw, lmbda, after, stride) in enumerate(MSTL_CASES):
        N = W + after
        y = make_series(
            N, list(periods), seed=4910 + ci, outliers=0.03 if kw.get("robust") else 0.0
        )
        if lmbda is not None:
            y = y - y.min() + 1.0
        scale = float(np.std(y))
        acc = {}
        kacc = {}
        nwin = 0
        kern = None
        if not kw.get("robust") and lmbda is None:
            ops = _mstl_port_fit(np.eye(W), periods, windows, iterate, kw, None, W)
            kern = ops
        t_port = t_sm = 0.0
        for t in range(W - 1, N, stride):
            win = y[t - W + 1 : t + 1]
            a = time.perf_counter()
            orc = _sm_mstl(win, periods, windows, iterate, kw, lmbda)
            b = time.perf_counter()
            mine = _mstl_port_fit(win, periods, windows, iterate, kw, lmbda, W)
            c = time.perf_counter()
            t_sm += b - a
            t_port += c - b
            g = {
                f"seasonal_{p}": gap_stats(mine["seasonal"][i], orc["seasonal"][i])
                for i, p in enumerate(sorted(periods))
            }
            g["trend"] = gap_stats(mine["trend"], orc["trend"])
            g["resid"] = gap_stats(mine["resid"], orc["resid"])
            g["weights"] = gap_stats(mine["weights"], orc["weights"])
            _merge(acc, g)
            if kern is not None:
                kg = {
                    f"seasonal_{p}": gap_stats(kern["seasonal"][i] @ win, orc["seasonal"][i], scale)
                    for i, p in enumerate(sorted(periods))
                }
                kg["trend"] = gap_stats(kern["trend"] @ win, orc["trend"], scale)
                _merge(kacc, kg)
            nwin += 1
        out["cases"].append(
            {
                "name": name,
                "periods": list(periods),
                "windows": windows,
                "W": W,
                "iterate": iterate,
                "stl_kwargs": kw,
                "lmbda": lmbda,
                "fits": nwin,
                "scale_sd": scale,
                "port_vs_sm": acc,
                "kernel_vs_sm": kacc,
                "port_ms": 1e3 * t_port / nwin,
                "statsmodels_ms": 1e3 * t_sm / nwin,
            }
        )
    # the embargo for two seasons: window values against the whole history's
    periods, windows = (12, 60), (7, 11)
    N = 2400
    y = make_series(N, list(periods), seed=4920)
    full = _sm_mstl(y, periods, windows, 2)
    lo, hi = 720, N - 720
    emb = []
    for W in (240, 480, 720, 1440):
        ops = _mstl_port_fit(np.eye(W), periods, windows, 2, {}, None, W)
        Y = _windows_matrix(y, W)
        comps = {
            "seasonal_12": ops["seasonal"][0] @ Y,
            "seasonal_60": ops["seasonal"][1] @ Y,
            "trend": ops["trend"] @ Y,
        }
        ref = {
            "seasonal_12": full["seasonal"][0],
            "seasonal_60": full["seasonal"][1],
            "trend": full["trend"],
        }
        ends = np.arange(W - 1, N)
        rows_ = []
        for h in sorted({0, 5, 29, 30, 59, 60, 300, W // 2} & set(range(W))):
            rows = ends - h
            sel = (rows >= lo) & (rows < hi)
            r = {"h": h}
            for k in comps:
                d = comps[k][W - 1 - h, sel] - ref[k][rows[sel]]
                r[k] = float(np.sqrt(np.mean(d**2)))
            rows_.append(r)
        emb.append({"W": W, "per_h": rows_})
    out["embargo"] = {
        "periods": periods,
        "windows": windows,
        "N": N,
        "sd": {
            k: float(np.std(v))
            for k, v in (("seasonal_12", full["seasonal"][0]), ("seasonal_60", full["seasonal"][1]))
        },
        "rows_compared": [lo, hi],
        "per_W": emb,
    }
    # statsmodels drops a period of at least half the window
    import warnings

    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        r = _sm_mstl(y[:300], (12, 168), None, 2)
    out["drop_long_period"] = {
        "W": 300,
        "periods": [12, 168],
        "seasonal_columns": len(r["seasonal"]),
        "warning": str(wlist[0].message) if wlist else None,
    }
    return out


def summarise4(res):
    for c in res["cases"]:
        print(
            f"== {c['name']}: W {c['W']}, fits {c['fits']},"
            f" iterate {c['iterate']}, kwargs {c['stl_kwargs']},"
            f" lmbda {c['lmbda']}; port {c['port_ms']:.0f} ms, sm {c['statsmodels_ms']:.1f} ms"
        )
        for k, a in c["port_vs_sm"].items():
            print(
                f"    port   {k:<12} bit-equal {a['bit_equal']}/{a['n']}"
                f" max {a['max_abs']:.1e} ({a['max_ulps']:.0f} ulps)"
            )
        for k, a in c["kernel_vs_sm"].items():
            print(
                f"    kernel {k:<12} bit-equal {a['bit_equal']}/{a['n']}"
                f" max {a['max_abs']:.1e} ({a['max_rel_scale']:.1e} sd)"
            )
    e = res["embargo"]
    print(f"== MSTL embargo {e['periods']} windows {e['windows']}, sd {e['sd']}")
    for w in e["per_W"]:
        for r in w["per_h"]:
            print(
                f"  W {w['W']:>5} h {r['h']:>4}: s12 {r['seasonal_12']:.2e}"
                f" s60 {r['seasonal_60']:.2e} trend {r['trend']:.2e}"
            )
    print("  drop:", res["drop_long_period"])


# ------------------------------------------------------------------------- data


def make_series(n, periods, seed, outliers=0.0, noise=0.3):
    """A smooth trend (an integrated AR(1) drift), seasons with slowly varying
    amplitude and phase, Gaussian noise, and optional gross outliers."""
    rng = np.random.default_rng(seed)
    drift = np.zeros(n)
    e = rng.normal(0, 0.02, n)
    for t in range(1, n):
        drift[t] = 0.95 * drift[t - 1] + e[t]
    trend = 10.0 + np.cumsum(drift)
    y = trend.copy()
    tt = np.arange(n)
    for k, p in enumerate(periods):
        amp = (2.0 + k) * (1.0 + 0.2 * np.sin(2 * np.pi * tt / (n / 1.7)))
        ph = 0.3 * np.sin(2 * np.pi * tt / (n / 2.3))
        y += amp * np.sin(2 * np.pi * tt / p + ph) + 0.5 * amp * np.cos(4 * np.pi * tt / p)
    y += rng.normal(0, noise, n)
    if outliers > 0:
        hit = rng.random(n) < outliers
        y[hit] += rng.choice([-1, 1], hit.sum()) * rng.uniform(5, 15, hit.sum())
    return y


def _versions():
    out = {"python": sys.version.split()[0], "numpy": np.__version__}
    for mod in ("statsmodels", "pandas", "scipy", "mpmath"):
        try:
            out[mod] = __import__(mod).__version__
        except ImportError:
            out[mod] = None
    return out


PARTS = {
    "part1": (part1, summarise1),
    "part2": (part2, summarise2),
    "part3": (part3, summarise3),
    "part4": (part4, summarise4),
}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("part", choices=sorted(PARTS))
    ap.add_argument("--out", default=os.path.join(".cache", "study_490_489"))
    ap.add_argument("--stl-rows", type=int, default=240, help="rows after the first full window")
    ap.add_argument("--variant-stride", type=int, default=8, help="windows between variant fits")
    ap.add_argument("--robust-stride", type=int, default=2, help="windows between robust fits")
    ap.add_argument("--time-budget", type=float, default=0.5, help="seconds per timed case")
    ap.add_argument("--force", action="store_true", help="recompute a part whose result exists")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"{args.part}.json")
    run, summarise = PARTS[args.part]
    if os.path.exists(path) and not args.force:
        with open(path) as fh:
            res = json.load(fh)
        print(f"(cached: {path})")
    else:
        t0 = time.perf_counter()
        res = run(args)
        res["_seconds"] = time.perf_counter() - t0
        with open(path, "w") as fh:
            json.dump(res, fh, indent=1)
    summarise(res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
