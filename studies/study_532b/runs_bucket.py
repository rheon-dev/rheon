"""Part 1: bucketed windows against Polars.

Two oracles:

1. ``group_by_dynamic`` with ``every = b``, ``period = B b``,
   ``offset = -(B - 1) b``, ``closed = "left"``: the window
   ``[(k - B + 1) b, (k + 1) b)`` for each bucket ``k``. The bucketed state read
   after the last row of bucket ``k`` covers exactly these rows.
2. ``rolling_*_by(t, window_size = B b, closed = "right")`` per row: the span
   ``(t - B b, t]``. The buckets cover it exactly only when ``t mod b = b - 1``
   (the row is the last tick of its bucket); elsewhere the window's old edge
   falls inside bucket ``k - B``, and that is the bucketed form's one
   approximation, measured here.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl

from .bucket import Bucketed, BucketSpec
from .common import compare, masked, stream_values, ulps
from .exact import exact_mean, exact_sum, exact_var

STATS = ("count", "sum", "mean", "var", "min", "max")


def run(
    gs: list[int], ts: list[int], xs: list, spec: BucketSpec, extra: bool = False
) -> tuple[list[dict], list[dict] | None, list[dict] | None]:
    """Stream the rows; per row, the B-bucket read (and, with ``extra``, the
    B+1-bucket read and the interpolated read)."""
    states: dict[int, Bucketed] = {}
    out, out_b1, out_i = [], [] if extra else None, [] if extra else None
    for g, t, x in zip(gs, ts, xs, strict=True):
        st = states.get(g)
        if st is None:
            st = states[g] = Bucketed(spec)
        st.step(x, t)
        out.append(st.read())
        if extra:
            out_b1.append(st.read(lag_buckets=1))
            out_i.append(st.read_interpolated(t))
    return out, out_b1, out_i


def tick_reads(gs: list[int], ts: list[int], xs: list, spec: BucketSpec) -> dict:
    """The read at the end of every bucket ``k`` whose window holds a row:
    after the last row of a bucket with rows, and, for a bucket with none, by
    moving the clock to it with no row (a read at a clock tick; it changes
    nothing a later row sees, since moving to ``k`` and then on is the same
    as moving on directly)."""
    states: dict[int, Bucketed] = {}
    reads: dict[tuple[int, int], dict] = {}
    B = spec.buckets
    for g, t, x in zip(gs, ts, xs, strict=True):
        st = states.get(g)
        k = t // spec.width
        if st is None:
            st = states[g] = Bucketed(spec)
        elif k > st.head:
            for kk in range(st.head + 1, min(k, st.head + B)):
                st.advance(kk)
                reads[(g, kk)] = st.read()
        st.step(x, t)
        reads[(g, k)] = st.read()
    for g, st in states.items():
        for kk in range(st.head + 1, st.head + B):
            st.advance(kk)
            reads[(g, kk)] = st.read()
    return reads


def _window_values(tg: np.ndarray, xg: list, lo: int, hi: int) -> list[float]:
    """Non-null values of one group's rows with ``lo <= t < hi``."""
    a, b = np.searchsorted(tg, lo, "left"), np.searchsorted(tg, hi, "left")
    return [v for v in xg[a:b] if v is not None]


def _group_arrays(df: pl.DataFrame) -> dict[int, tuple[np.ndarray, list]]:
    m = masked(df).sort("g", "t")
    out = {}
    for (g,), sub in m.group_by("g", maintain_order=True):
        out[g] = (sub["t"].to_numpy(), sub["xm"].to_list())
    return out


def vs_group_by_dynamic(df: pl.DataFrame, b: int, B: int, ddof: int = 1) -> dict:
    """The bucketed read after the last row of each bucket against
    ``group_by_dynamic``; and both against the exact value."""
    spec = BucketSpec(width=b, buckets=B, ddof=ddof, min_samples=0)
    gs, ts, xs = stream_values(df)
    reads = tick_reads(gs, ts, xs, spec)
    keys = sorted(reads)
    pol = (
        masked(df)
        .sort("g", "t")
        .group_by_dynamic(
            "t",
            every=f"{b}i",
            period=f"{B * b}i",
            offset=f"-{(B - 1) * b}i",
            closed="left",
            label="left",
            include_boundaries=True,
            group_by="g",
        )
        .agg(
            count=pl.col("xm").count().cast(pl.Float64),
            sum=pl.col("xm").sum(),
            mean=pl.col("xm").mean(),
            var=pl.col("xm").var(ddof=ddof),
            min=pl.col("xm").min(),
            max=pl.col("xm").max(),
        )
        .with_columns(k=pl.col("_lower_boundary") // b + (B - 1))
    )
    key_df = pl.DataFrame(
        {"g": [k[0] for k in keys], "k": [k[1] for k in keys]},
        schema={"g": pl.Int64, "k": pl.Int64},
    )
    joined = key_df.join(pol, on=["g", "k"], how="left", maintain_order="left")
    n_polars_windows = pol.height
    res = {"windows": len(keys), "polars_windows": n_polars_windows}
    # Windows with no value: Polars' overlapping-window sum is its add-and-subtract
    # residue there, not 0; held apart so the ulp column reads the windows with values.
    has = [reads[k]["count"] > 0 for k in keys]
    idx = [i for i, h in enumerate(has) if h]
    empty = [i for i, h in enumerate(has) if not h]
    for s in STATS:
        res[s] = compare([reads[keys[i]][s] for i in idx], joined[s].gather(idx))
    e_sum = joined["sum"].gather(empty).to_list()
    res["empty_windows"] = len(empty)
    res["empty_polars_sum_nonzero"] = sum(1 for v in e_sum if v != 0.0)
    res["empty_polars_sum_maxabs"] = max((abs(v) for v in e_sum), default=0.0)
    res["empty_ours_sum_nonzero"] = sum(1 for i in empty if reads[keys[i]]["sum"] != 0.0)
    # against the exact value (finite data only)
    if not df["x"].is_nan().any() and df["x"].is_finite().all():
        grp = _group_arrays(df)
        ex = {"sum": [], "mean": [], "var": []}
        for g, k in keys:
            tg, xg = grp[g]
            vals = _window_values(tg, xg, (k - B + 1) * b, (k + 1) * b)
            ex["sum"].append(exact_sum(vals) if vals else 0.0)
            ex["mean"].append(exact_mean(vals) if vals else None)
            ex["var"].append(exact_var(vals, ddof) if vals else None)
        for s in ("sum", "mean", "var"):
            exs = [ex[s][i] for i in idx]
            res[s + "_ours_exact"] = compare([reads[keys[i]][s] for i in idx], exs)
            res[s + "_polars_exact"] = compare(joined[s].gather(idx).to_list(), exs)
    return res


def polars_rolling(df: pl.DataFrame, span: int, ddof: int = 1) -> pl.DataFrame:
    """Polars' per-row span window ``(t - span, t]`` over each group, on the
    masked column, in the frame's own row order."""
    ws = f"{span}i"
    x = pl.col("xm")
    kw = dict(window_size=ws, closed="right", min_samples=1)
    return masked(df).select(
        count=x.is_not_null().cast(pl.Float64).rolling_sum_by("t", **kw).over("g"),
        sum=x.rolling_sum_by("t", **kw).over("g"),
        mean=x.rolling_mean_by("t", **kw).over("g"),
        var=x.rolling_var_by("t", ddof=ddof, **kw).over("g"),
        min=x.rolling_min_by("t", **kw).over("g"),
        max=x.rolling_max_by("t", **kw).over("g"),
    )


def vs_rolling_aligned(df: pl.DataFrame, b: int, B: int, ddof: int = 1) -> dict:
    """Rows whose time is the last tick of its bucket (``t mod b = b - 1``):
    there the B buckets are exactly ``(t - B b, t]``."""
    spec = BucketSpec(width=b, buckets=B, ddof=ddof, min_samples=1)
    gs, ts, xs = stream_values(df)
    ours, _, _ = run(gs, ts, xs, spec)
    pol = polars_rolling(df, B * b, ddof)
    rows = [i for i, t in enumerate(ts) if t % b == b - 1]
    res = {"rows": len(rows)}
    for s in STATS:
        theirs = pol[s].gather(rows)
        o = [ours[i][s] for i in rows]
        res[s] = compare(o, theirs)
    return res


def edge_gap(df: pl.DataFrame, b: int, B: int, ddof: int = 1) -> dict:
    """Every row against ``rolling_*_by`` over ``(t - B b, t]``: the B-bucket
    read (covers less: the part of bucket ``k - B`` inside the span is
    missing), the B+1-bucket read (covers more: the rest of bucket ``k - B``),
    and the interpolated read (count, sum, mean)."""
    spec = BucketSpec(width=b, buckets=B, ddof=ddof, min_samples=1, keep_previous=True)
    gs, ts, xs = stream_values(df)
    ours, ours_b1, ours_i = run(gs, ts, xs, spec, extra=True)
    pol = polars_rolling(df, B * b, ddof).to_dict(as_series=False)
    sd = [math.sqrt(v) if v is not None and v > 0 else None for v in pol["var"]]
    res: dict = {}
    for name, rd in (("B", ours), ("B+1", ours_b1), ("interp", ours_i)):
        r: dict = {}
        cnt_diff = 0
        n_rows = 0
        gaps: dict[str, list[float]] = {s: [] for s in ("count", "mean", "sd", "min", "max")}
        for i in range(len(ts)):
            ex_n = pol["count"][i]
            if ex_n is None or ex_n < 3 or sd[i] is None or rd[i]["count"] in (None, 0.0):
                continue
            if rd[i].get("mean") is None:
                continue
            n_rows += 1
            if rd[i]["count"] != ex_n:
                cnt_diff += 1
            gaps["count"].append(abs(rd[i]["count"] - ex_n) / ex_n)
            gaps["mean"].append(abs(rd[i]["mean"] - pol["mean"][i]) / sd[i])
            if name != "interp":
                v = rd[i]["var"]
                if v is not None:
                    gaps["sd"].append(abs(math.sqrt(v) - sd[i]) / sd[i])
                gaps["min"].append(abs(rd[i]["min"] - pol["min"][i]) / sd[i])
                gaps["max"].append(abs(rd[i]["max"] - pol["max"][i]) / sd[i])
        r["rows"] = n_rows
        r["count_differs"] = cnt_diff
        for s, gl in gaps.items():
            if gl:
                a = np.array(gl)
                r[s] = (float(np.median(a)), float(np.quantile(a, 0.95)), float(a.max()))
        if name != "interp":
            r["min_differs"] = sum(1 for gl in [gaps["min"]] for v in gl if v != 0.0)
            r["max_differs"] = sum(1 for v in gaps["max"] if v != 0.0)
        res[name] = r
    # the bracket: B read inside the span inside the B+1 read, so
    # max_B <= max_exact <= max_{B+1} and min_B >= min_exact >= min_{B+1}
    bad = 0
    for i in range(len(ts)):
        if pol["max"][i] is None or ours[i]["max"] is None:
            continue
        if not (ours[i]["max"] <= pol["max"][i] <= ours_b1[i]["max"]):
            bad += 1
        if not (ours[i]["min"] >= pol["min"][i] >= ours_b1[i]["min"]):
            bad += 1
    res["bracket_violations"] = bad
    return res


def sumsq_vs_m2(df: pl.DataFrame, b: int, B: int) -> dict:
    """The variance from a plain sum of squares against the centred form, both
    against the exact value, on the group_by_dynamic windows."""
    spec = BucketSpec(width=b, buckets=B, min_samples=0, sumsq=True)
    gs, ts, xs = stream_values(df)
    ours, _, _ = run(gs, ts, xs, spec)
    last: dict[tuple[int, int], int] = {}
    for i, (g, t) in enumerate(zip(gs, ts, strict=True)):
        last[(g, t // b)] = i
    grp = _group_arrays(df)
    ex, o_m2, o_sq = [], [], []
    for (g, k), i in sorted(last.items()):
        tg, xg = grp[g]
        vals = _window_values(tg, xg, (k - B + 1) * b, (k + 1) * b)
        ex.append(exact_var(vals) if vals else None)
        o_m2.append(ours[i]["var"])
        o_sq.append(ours[i]["var_sumsq"])
    return {"m2": compare(o_m2, ex), "sumsq": compare(o_sq, ex)}


def max_ulp_of(a: list, b: list) -> int:
    x = np.array([v for v, w in zip(a, b, strict=True) if v is not None and w is not None])
    y = np.array([w for v, w in zip(a, b, strict=True) if v is not None and w is not None])
    return int(ulps(x, y).max()) if x.size else 0
