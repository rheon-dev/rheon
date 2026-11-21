"""The data, the zero-weight mask, and the bit and ulp measures.

The generator and the measures follow study 532 (branch ``vm/study-532``,
``studies/study_532/oracle.py``) so the two studies read the same kind of
stream: seeded, generated, nothing read from disk.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl

Value = float | None


def make_frame(
    n: int,
    groups: int = 5,
    seed: int = 532,
    null_p: float = 0.03,
    zero_w_p: float = 0.04,
    nonfinite: bool = False,
    level: float = 1.0e3,
    max_step: int = 5,
) -> pl.DataFrame:
    """Rows interleaved across ``groups`` keys; within a key the clock rises
    strictly, by 1 to ``max_step`` units, with a gap of 200 to 2000 units about
    one row in 100. ``x`` is a random walk about ``level`` with heavy-tailed
    steps (so sums and variances cancel); about ``null_p`` of it is null; ``w``
    is 0 on about ``zero_w_p`` of the rows and positive elsewhere."""
    rng = np.random.default_rng(seed)
    g = rng.integers(0, groups, n)
    step = rng.integers(1, max_step + 1, n)
    gap = rng.random(n) < 0.01
    step = np.where(gap, rng.integers(200, 2001, n), step)
    t = np.zeros(n, dtype=np.int64)
    x = np.zeros(n)
    last_t = np.zeros(groups, dtype=np.int64)
    last_x = np.full(groups, level)
    eps = rng.standard_t(3, n)
    noise = rng.standard_normal(n)
    for i in range(n):
        k = g[i]
        last_t[k] += step[i]
        last_x[k] += eps[i]
        t[i] = last_t[k]
        x[i] = last_x[k] + 0.25 * noise[i]
    if nonfinite:
        bad = rng.random(n) < 0.004
        kinds = rng.integers(0, 3, n)
        x = np.where(bad & (kinds == 0), np.nan, x)
        x = np.where(bad & (kinds == 1), np.inf, x)
        x = np.where(bad & (kinds == 2), -np.inf, x)
    valid = rng.random(n) >= null_p
    w = np.where(rng.random(n) < zero_w_p, 0.0, rng.uniform(0.5, 2.0, n))
    return pl.DataFrame({"g": g.astype(np.int64), "t": t, "x": x, "w": w}).with_columns(
        x=pl.when(pl.Series(valid)).then(pl.col("x")),
    )


def masked(df: pl.DataFrame) -> pl.DataFrame:
    """The zero-weight rule: a zero-weight row's value is a null to every
    feature (column ``xm``). Its row and its time stay."""
    return df.with_columns(xm=pl.when(pl.col("w") != 0.0).then(pl.col("x")))


def stream_values(df: pl.DataFrame) -> tuple[list[int], list[int], list[Value]]:
    """The masked stream as Python lists: group, time, value (None for a null
    or a zero-weight row)."""
    m = masked(df)
    return m["g"].to_list(), m["t"].to_list(), m["xm"].to_list()


# ---------------------------------------------------------------------------
# Bits and ulps.


def ordered(a: np.ndarray) -> np.ndarray:
    """f64 to a signed integer whose order is the float order; +0 and -0 both
    map to 0, so adjacent floats differ by 1."""
    i = np.ascontiguousarray(a, dtype=np.float64).view(np.int64)
    return np.where(i < 0, -(i & np.int64(0x7FFFFFFFFFFFFFFF)), i)


def ulps(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """|a - b| in ulps, element-wise, for finite pairs."""
    return np.abs(ordered(a) - ordered(b))


def to_arr(vals: list[Value]) -> tuple[np.ndarray, np.ndarray]:
    valid = np.array([v is not None for v in vals], dtype=bool)
    arr = np.array([math.nan if v is None else v for v in vals], dtype=np.float64)
    return arr, valid


def compare(ours: list[Value], theirs: pl.Series | list[Value]) -> dict[str, float | int]:
    """Rows whose validity differs; rows whose bits differ (both valid; NaN
    equals NaN); the largest gap in ulps between finite values, and the
    largest relative gap."""
    o, o_valid = to_arr(ours)
    if isinstance(theirs, pl.Series):
        p_valid = theirs.is_not_null().to_numpy()
        p = theirs.fill_null(np.nan).to_numpy().astype(np.float64)
    else:
        p, p_valid = to_arr(theirs)
    both = o_valid & p_valid
    o_b, p_b = o[both], p[both]
    same = (o_b.view(np.uint64) == p_b.view(np.uint64)) | (np.isnan(o_b) & np.isnan(p_b))
    # +0 and -0 are the same value; count them equal in bits too only if both zero
    same |= (o_b == 0.0) & (p_b == 0.0)
    fin = np.isfinite(o_b) & np.isfinite(p_b)
    gap = ulps(o_b[fin], p_b[fin]) if fin.any() else np.zeros(0, np.int64)
    nonfin_mismatch = int(((~np.isfinite(o_b)) | (~np.isfinite(p_b))).sum() - (~fin & same).sum())
    rel = (
        np.abs(o_b[fin] - p_b[fin]) / np.maximum(np.abs(p_b[fin]), 1e-300)
        if fin.any()
        else np.zeros(0)
    )
    return {
        "rows": len(o),
        "valid": int(both.sum()),
        "validity_diff": int((o_valid != p_valid).sum()),
        "bit_diff": int((~same).sum()),
        "nonfinite_diff": nonfin_mismatch,
        "max_ulp": int(gap.max()) if gap.size else 0,
        "median_ulp": float(np.median(gap)) if gap.size else 0.0,
        "max_rel": float(rel.max()) if rel.size else 0.0,
    }
