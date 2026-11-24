"""The window operator ``rp.ewm_moments``: one term a call, each row's value
weighing the decayed mass of the interval it is held over, as ``ewm_var``'s
does.

It is held to the spec kind ``rp.spec.ewm_moments`` fed those weights, to the
bit (each row's output read after the row: a row of weight 0 at the same
stamp reads the spec there), and to the weighted definitions written out in
NumPy over the window's rows. The spec kind is held to scipy, statsmodels and
Polars in ``tests/test_reference_ewm_moments.py`` and ``test_ewm_moments.py``.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest

from rheon import polars as rp

TIER = "essential"

LN2 = math.log(2.0)


def _frame(n: int = 240, seed: int = 554) -> pl.DataFrame:
    """A gamma column at a level of about 6 and an AR(1) column at 0.6, on a
    clock with uneven steps and no repeated stamp."""
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(n)
    ar = np.zeros(n)
    for i in range(1, n):
        ar[i] = 0.6 * ar[i - 1] + e[i]
    return pl.DataFrame(
        {
            "t": np.cumsum(rng.uniform(0.2, 2.0, n)),
            "g": 3.0 + rng.gamma(2.0, 1.5, n),
            "ar": ar,
        }
    )


def _mass(d: float, h: float) -> float:
    """The decayed mass of ``d`` units of held time, in the operations the
    window core takes it in: from a hundredth of a half-life up from the
    discount, below it through ``expm1``."""
    if d <= 0.0:
        return 0.0
    if math.isinf(h):
        return d
    if d / h >= 1e-2:
        return (h / LN2) * (1.0 - math.exp2(-(d / h)))
    return -(h / LN2) * math.expm1(-(d * LN2 / h))


def _held(t: np.ndarray, h: float, window: float | None) -> np.ndarray:
    """Each row's weight: the mass of the interval since the row before, the
    first held from before the stream (one window under a window), and none
    longer than the window."""
    w = np.empty(len(t))
    cap = math.inf if window is None else window
    w[0] = _mass(cap, h)
    for i in range(1, len(t)):
        w[i] = _mass(min(float(t[i] - t[i - 1]), cap), h)
    return w


def _probed(frame: pl.DataFrame, w: np.ndarray) -> pl.DataFrame:
    """Each row with its weight, then a copy of it at weight 0, which the
    spec reads at the row's stamp once the row is in."""
    real = frame.with_columns(w=pl.Series(w), probe=pl.lit(False))
    probe = frame.with_columns(w=pl.lit(0.0), probe=pl.lit(True))
    return (
        pl.concat([real.with_row_index("i"), probe.with_row_index("i")])
        .sort(["i", "probe"])
        .drop("i")
    )


def _bits(s: pl.Series) -> list[int | None]:
    return [None if v is None else int(np.float64(v).view(np.int64)) for v in s.to_list()]


CASES: list[tuple[str, dict[str, Any], dict[str, Any]]] = [
    ("skew", {"g": 3}, {"kind": "standardized"}),
    ("kurt_unbiased", {"g": 4}, {"kind": "standardized", "bias": False}),
    ("m3", {"g": 3}, {}),
    ("m4_unbiased", {"g": 4}, {"bias": False}),
    ("var", {"g": 2}, {}),
    ("raw3", {"g": 3}, {"kind": "raw"}),
    ("k4", {"g": 4}, {"kind": "cumulant"}),
    ("cross", {"g": 2, "ar": 1}, {}),
    ("corr", {"g": 1, "ar": 1}, {"kind": "standardized"}),
    ("m6", {"ar": 6}, {}),
    ("ac", {"ar": 1, "ar@1.5": 1}, {"kind": "standardized"}),
    ("lagged_cross", {"g": 1, "ar@3": 2}, {"lag_tolerance": 0.5}),
    ("window", {"g": 3}, {"kind": "standardized", "window_size": 25.0}),
    ("window_lag", {"ar": 1, "ar@2": 1}, {"window_size": 30.0, "lag_tolerance": 1.0}),
    ("even_window", {"g": 2}, {"window_size": 20.0, "half_life": math.inf}),
]


@pytest.mark.parametrize(("name", "legs", "kw"), CASES, ids=[c[0] for c in CASES])
def test_the_operator_is_the_spec_kind_fed_its_weights_to_the_bit(
    name: str, legs: dict[str, int], kw: dict[str, Any]
) -> None:
    kw = dict(kw)
    h = kw.pop("half_life", 15.0)
    frame = _frame()
    op = rp.stream.with_windows(
        frame, v=rp.ewm_moments(legs, half_life=h, **kw), clock="t", gap_cap=1e9
    )["v"]
    w = _held(frame["t"].to_numpy(), h, kw.get("window_size"))
    spec = rp.spec.ewm_moments(
        "m",
        terms={name: legs},
        kind=kw.get("kind"),
        bias=kw.get("bias"),
        window_size=kw.get("window_size"),
        lag_tolerance=kw.get("lag_tolerance"),
        half_life=h,
        clock="t",
        gap_cap=1e9,
        weight="w",
        min_weight=0.0,
    )
    stream = _probed(frame, w)
    out = rp.ModelBank([spec]).fit_predict(stream)
    read = out.filter(pl.col("probe"))["m"].struct.field(name)
    assert op.null_count() < len(op)
    assert _bits(op) == _bits(read)


def _definition(
    frame: pl.DataFrame,
    legs: list[tuple[str, int]],
    kind: str,
    h: float,
    window: float | None,
) -> list[float | None]:
    """The term at each row, from its definition over the window's rows:
    each row weighing its held mass decayed to the row read, the moments
    numpy's weighted averages of the deviations' products."""
    t = frame["t"].to_numpy()
    w = _held(t, h, window)
    z = np.column_stack([frame[c].to_numpy() for c, _ in legs])
    a = [p for _, p in legs]
    out: list[float | None] = []
    for i in range(len(t)):
        lo = 0 if window is None else int(np.searchsorted(t, t[i] - window, side="right"))
        om = w[lo : i + 1] * np.exp2(-(t[i] - t[lo : i + 1]) / h)
        zz = z[lo : i + 1]
        m = np.average(zz, axis=0, weights=om)
        dev = zz - m
        if kind == "raw":
            out.append(float(np.average(np.prod(zz**a, axis=1), weights=om)))
            continue
        mu = float(np.average(np.prod(dev**a, axis=1), weights=om))
        if kind == "central":
            out.append(mu)
            continue
        var = np.average(dev**2, axis=0, weights=om)
        if np.any(var <= 0):
            out.append(None)
            continue
        out.append(mu / float(np.prod(np.sqrt(var) ** a)))
    return out


@pytest.mark.parametrize(
    ("legs", "kind", "window"),
    [
        ({"g": 3}, "central", None),
        ({"g": 4}, "standardized", None),
        ({"g": 2}, "raw", None),
        ({"g": 2, "ar": 1}, "central", 30.0),
        ({"g": 1, "ar": 1}, "standardized", 30.0),
        ({"ar": 5}, "central", 12.0),
    ],
)
def test_the_operator_is_the_weighted_definition(
    legs: dict[str, int], kind: str, window: float | None
) -> None:
    h = 10.0
    frame = _frame(160)
    got = rp.stream.with_windows(
        frame,
        v=rp.ewm_moments(legs, half_life=h, kind=kind, bias=True, window_size=window),  # type: ignore[arg-type]
        clock="t",
        gap_cap=1e9,
    )["v"].to_list()
    want = _definition(frame, list(legs.items()), kind, h, window)
    for i, (g, wt) in enumerate(zip(got, want, strict=True)):
        if wt is None or (kind == "standardized" and i == 0):
            assert g is None, (i, g)
            continue
        assert g is not None, i
        assert abs(g - wt) <= 1e-10 * max(abs(wt), 1.0), (i, g, wt)


def test_a_lagged_term_pairs_each_row_with_its_as_of_row() -> None:
    """A lag of 1.5 on the uneven clock: each row pairs with the latest row
    at or before ``t - 1.5``, none where that row is more than the lag
    before it; each pair weighs its closing row's held mass, and the pairs
    are centred on their own weighted means."""
    h, k = 20.0, 1.5
    frame = _frame(150)
    t = frame["t"].to_numpy()
    x = frame["ar"].to_numpy()
    got = rp.stream.with_windows(
        frame,
        v=rp.ewm_moments({"ar": 1, "ar@1.5": 1}, half_life=h, bias=True),
        clock="t",
        gap_cap=1e9,
    )["v"].to_list()
    w = _held(t, h, None)
    pairs: list[tuple[int, int]] = []
    for i in range(len(t)):
        j = int(np.searchsorted(t, t[i] - k, side="right")) - 1
        if j >= 0 and t[i] - k - t[j] <= k:
            pairs.append((i, j))
        if not pairs:
            assert got[i] is None, i
            continue
        idx = np.array([p[0] for p in pairs])
        om = w[idx] * np.exp2(-(t[i] - t[idx]) / h)
        a = x[idx]
        b = x[np.array([p[1] for p in pairs])]
        da, db = a - np.average(a, weights=om), b - np.average(b, weights=om)
        want = float(np.average(da * db, weights=om))
        assert got[i] is not None and abs(got[i] - want) <= 1e-11 * max(abs(want), 1.0), i


@pytest.mark.parametrize("bias", [True, False])
def test_the_variance_term_is_ewm_var(bias: bool) -> None:
    frame = _frame()
    out = rp.stream.with_windows(
        frame,
        m=rp.ewm_moments({"g": 2}, half_life=7.0, bias=bias),
        v=rp.ewm_var("g", half_life=7.0, bias=bias),
        clock="t",
        gap_cap=1e9,
    )
    for i, (m, v) in enumerate(zip(out["m"].to_list(), out["v"].to_list(), strict=True)):
        if v is None:
            assert m is None, i
        else:
            assert m is not None and abs(m - v) <= 1e-12 * max(abs(v), 1.0), (i, m, v)


def _ops() -> dict[str, pl.Expr]:
    return {
        "s": rp.ewm_moments({"x": 3}, half_life=4.0, kind="standardized", window_size=9.0),
        "a": rp.ewm_moments({"x": 1, "x@1": 1}, half_life=6.0, kind="standardized"),
        "k": rp.ewm_moments({"x": 4}, half_life=5.0, kind="cumulant"),
        "v": rp.ewm_var("x", half_life=4.0, window_size=9.0),
    }


def _ticks(n: int, seed: int) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    step = rng.choice([0.0, 0.5, 1.0, 2.5, 30.0], n, p=[0.1, 0.4, 0.35, 0.13, 0.02])
    x = rng.standard_t(5, n)
    x[rng.uniform(size=n) < 0.05] = np.nan
    return pl.DataFrame({"t": np.cumsum(step), "g": rng.integers(0, 3, n), "x": x}).with_columns(
        pl.col("x").fill_nan(None)
    )


@pytest.mark.parametrize("rows", [1, 7, 64])
def test_chunking_changes_no_bit(rows: int) -> None:
    df = _ticks(500, 11)
    kw: dict[str, Any] = {"clock": "t", "gap_cap": 20.0, "group": "g"}
    one = rp.stream.with_windows(df, **_ops(), **kw, chunk_size=10_000)
    out = rp.stream.with_windows(df, **_ops(), **kw, chunk_size=rows)
    assert out.equals(one)
    assert one["s"].null_count() < len(one)


def test_a_saved_state_resumes_to_the_bit(tmp_path: Path) -> None:
    df = _ticks(120, 12)
    kw: dict[str, Any] = {"clock": "t", "gap_cap": 20.0, "group": "g"}
    one = rp.stream.with_windows(df, **_ops(), **kw)
    t = df["t"].to_list()
    # A state resumes on an input that starts past its last clock value.
    cuts = [at for at in (1, 2, 40, 41, 77, 78, 119) if t[at] != t[at - 1]]
    assert len(cuts) >= 4
    for at in cuts:
        state = tmp_path / f"m{at}.state"
        first = rp.stream.with_windows(df[:at], **_ops(), **kw, save_state=state)
        second = rp.stream.with_windows(df[at:], **_ops(), **kw, load_state=state)
        assert pl.concat([first, second]).equals(one), at


def test_a_repeated_stamp_and_a_missing_value_move_nothing() -> None:
    """A row at a repeated stamp holds no interval and a row with no value is
    skipped: each gives what the stream without it gives at that stamp, and
    every row at one stamp reads the same value."""
    h = 5.0
    base = pl.DataFrame({"t": [0.0, 1.0, 2.0, 4.0, 5.0, 7.0], "x": [1.0, 3.0, 2.0, 6.0, 4.0, 5.0]})
    extra = pl.DataFrame({"t": [2.0, 4.5], "x": [50.0, None]})
    both = pl.concat([base, extra]).sort("t", maintain_order=True)
    op = {"k": rp.ewm_moments({"x": 3}, half_life=h, kind="standardized")}
    alone = rp.stream.with_windows(base, **op, clock="t", gap_cap=1e9)
    got = rp.stream.with_windows(both, **op, clock="t", gap_cap=1e9)
    want = dict(zip(alone["t"].to_list(), alone["k"].to_list(), strict=True))
    for t, k in zip(got["t"].to_list(), got["k"].to_list(), strict=True):
        if t == 4.5:
            assert k == want[4.0]
        else:
            assert k == want[t], t


def test_a_clock_break_starts_the_moments_over() -> None:
    frame = _frame(80)
    later = frame.with_columns(pl.col("t") + 1e4)
    both = pl.concat([frame, later])
    op = {"k": rp.ewm_moments({"g": 4}, half_life=9.0, kind="standardized")}
    whole = rp.stream.with_windows(both, **op, clock="t", gap_cap=100.0)
    alone = rp.stream.with_windows(later, **op, clock="t", gap_cap=100.0)
    assert whole["k"][80:].equals(alone["k"])


def test_a_row_count_clock_lags_by_rows() -> None:
    """Without a clock column a lag counts rows: ``x@2`` pairs each row with
    the row two before it."""
    rng = np.random.default_rng(3)
    x = rng.standard_normal(60)
    frame = pl.DataFrame({"x": x})
    got = rp.stream.with_windows(
        frame,
        v=rp.ewm_moments({"x": 1, "x@2": 1}, half_life=math.inf, window_size=30.0, bias=True),
    )["v"].to_list()
    for i in (10, 31, 59):
        idx = np.arange(max(2, i - 29), i + 1)
        a, b = x[idx], x[idx - 2]
        want = float(np.mean((a - a.mean()) * (b - b.mean())))
        assert got[i] is not None and abs(got[i] - want) <= 1e-12, i


@pytest.mark.parametrize(
    ("legs", "kw", "error", "match"),
    [
        ({}, {}, TypeError, "legs must be a dict"),
        ([("x", 1)], {}, TypeError, "legs must be a dict"),
        ({"x": 0}, {}, ValueError, "power is a whole number from 1"),
        ({"x": 1.5}, {}, TypeError, "power is a whole number"),
        ({"x": True}, {}, TypeError, "power is a whole number"),
        ({"x@-1": 1, "x": 1}, {}, ValueError, "at least 0"),
        ({"x@1": 1, "x@1.0": 1}, {}, ValueError, "another leg names too"),
        ({"x": 1, "y": 1}, {"kind": "cumulant"}, ValueError, "a cumulant is of one leg"),
        ({"x": 1}, {"kind": "standardized"}, ValueError, "order 1 is 0"),
        ({"x": 5}, {"bias": False}, ValueError, "stops at order 4"),
        ({"x": 3}, {"kind": "skew"}, ValueError, "kind must be"),
        ({"x": 3}, {"bias": "no"}, TypeError, "bias must be"),
        ({"x": 3}, {"half_life": math.inf}, ValueError, "window_size"),
        ({"x": 3}, {"partial": "keep"}, ValueError, "partial needs window_size"),
        ({"x": 3}, {"lag_tolerance": -1.0}, ValueError, "lag_tolerance"),
    ],
)
def test_the_operator_refuses_by_name(
    legs: Any, kw: dict[str, Any], error: type[Exception], match: str
) -> None:
    kw = {"half_life": 5.0, **kw}
    with pytest.raises(error, match=match):
        rp.ewm_moments(legs, **kw)


def test_a_duration_lag_needs_a_temporal_clock() -> None:
    frame = pl.DataFrame({"t": [0.0, 1.0, 2.0], "x": [1.0, 2.0, 4.0]})
    with pytest.raises(ValueError, match="duration"):
        rp.stream.with_windows(
            frame, v=rp.ewm_moments({"x": 1, "x@5m": 1}, half_life=5.0), clock="t", gap_cap=1e9
        )


def test_a_temporal_clock_takes_duration_lags() -> None:
    frame = _frame(100)
    ts = pl.lit(0).cast(pl.Datetime("ns")) + pl.duration(
        nanoseconds=(pl.col("t") * 1e9).cast(pl.Int64)
    )
    temporal = frame.with_columns(ts=ts)
    v = rp.ewm_moments({"ar": 1, "ar@1500ms": 1}, half_life="15s", window_size="40s")
    got = rp.stream.with_windows(temporal, v=v, clock="ts", gap_cap="1h")["v"]
    num = rp.stream.with_windows(
        temporal.with_columns(pl.col("ts").dt.epoch("ns") / 1e9),
        v=rp.ewm_moments({"ar": 1, "ar@1.5": 1}, half_life=15.0, window_size=40.0),
        clock="ts",
        gap_cap=3600.0,
    )["v"]
    assert got.null_count() == num.null_count() < len(got)
    for a, b in zip(got.to_list(), num.to_list(), strict=True):
        assert (a is None and b is None) or abs(a - b) <= 1e-9 * max(abs(b), 1.0)
