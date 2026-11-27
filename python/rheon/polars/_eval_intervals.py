"""Interval evaluation metrics (``interval_metrics``) and quantile pinball / D²
helpers for :mod:`rheon.polars.eval`.

Split into a private helper module to keep ``eval.py`` well under the source
size cap while sharing its validation, slot-unpacking, and ``min_samples``
conventions.
"""

from __future__ import annotations

import math
import numbers
import re
from collections.abc import Callable, Iterable, Sequence
from typing import Any

import numpy as np
import polars as pl

from rheon import spec as _spec_mod
from rheon._errors import WrongTypeError

#: Bound matching ``rheon_core::INPUT_BOUND`` / ``eval._INPUT_BOUND``.
_INPUT_BOUND = 1e100

#: Column names reserved by :func:`interval_metrics`.
INTERVAL_PRODUCED = (
    "slot",
    "target",
    "pred",
    "y",
    "nominal_coverage",
    "n",
    "coverage",
    "mean_width",
    "winkler",
    "ssc",
    "window_start",
)

_MULTI_LO_RE = re.compile(r"^lo_c([^_]+)_(.+)$")
_SINGLE_LO_RE = re.compile(r"^lo_(.+)$")


def num_label(v: float) -> str:
    """Format a quantile or coverage level like Rust's ``num_label``."""
    if v == 0.0:
        return "0"
    a = abs(v)
    if math.isfinite(v) and not (1e-6 <= a < 1e7):
        s = f"{v:e}"
        mant, exp = s.split("e")
        mant = mant.rstrip("0").rstrip(".")
        return f"{mant}e{int(exp)}"
    if a < 1e-4:
        return f"{v:.16f}".rstrip("0").rstrip(".")
    return f"{v:.15g}"


def check_bool_flag(who: str, name: str, value: object) -> bool:
    """Validate a boolean flag, refusing non-bool values cleanly."""
    if not isinstance(value, bool):
        msg = f"{who}: {name} must be a bool, but it is {value!r}. Pass True or False."
        raise WrongTypeError(msg)
    return value


def check_quantiles(who: str, quantiles: Sequence[float] | None) -> list[float]:
    """Validate an optional sequence of quantile levels strictly inside (0, 1)."""
    if quantiles is None:
        return []
    if isinstance(quantiles, (str, bytes)) or not isinstance(quantiles, Sequence):
        msg = (
            f"{who}: quantiles must be a sequence of numbers between 0 and 1, "
            f"but it is {quantiles!r}. Pass a list of levels such as [0.1, 0.5, 0.9], or None."
        )
        raise WrongTypeError(msg)
    if len(quantiles) == 0:
        msg = (
            f"{who}: quantiles cannot be empty. Pass one or more quantile levels "
            "between 0 and 1 (such as [0.1, 0.5, 0.9]), or None."
        )
        raise ValueError(msg)
    out: list[float] = []
    seen: set[float] = set()
    for q in quantiles:
        if isinstance(q, bool) or not isinstance(q, numbers.Real):
            msg = (
                f"{who}: quantiles must hold numbers between 0 and 1, but one entry is {q!r}. "
                "Pass numeric quantile levels between 0 and 1."
            )
            raise WrongTypeError(msg)
        qf = float(q)
        if not (math.isfinite(qf) and 0.0 < qf < 1.0):
            msg = (
                f"{who}: quantiles must each be above 0 and below 1, but one entry is {q!r}. "
                "Set each quantile level strictly between 0 and 1."
            )
            raise ValueError(msg)
        if qf in seen:
            msg = (
                f"{who}: quantiles lists {num_label(qf)} more than once. "
                "List each quantile level at most once."
            )
            raise ValueError(msg)
        seen.add(qf)
        out.append(qf)
    return out


def quantile_col_names(quantiles: Sequence[float]) -> list[str]:
    """Column names added to ``metrics`` / ``window_metrics`` for ``quantiles``."""
    names: list[str] = []
    for q in quantiles:
        lbl = num_label(q)
        names.append(f"pinball_q{lbl}")
        names.append(f"d2_pinball_q{lbl}")
    return names


def _weighted_percentile_avg(y: np.ndarray, w: np.ndarray, q: float) -> float:
    """Weighted ``q``-quantile matching ``sklearn.utils.stats._weighted_percentile``
    with ``average=True`` (averaged inverted CDF)."""
    order = np.argsort(y, kind="mergesort")
    ys = y[order]
    ws = w[order]
    cdf = np.cumsum(ws)
    total = float(cdf[-1])
    target = q * total
    if target == 0.0:
        target = float(np.nextafter(0.0, 1.0))
    idx = int(np.searchsorted(cdf, target, side="left"))
    max_idx = len(ys) - 1
    if idx > max_idx:
        idx = max_idx
    eps = float(np.finfo(np.float64).eps)
    if float(cdf[idx]) - target > eps:
        return float(ys[idx])
    nxt = min(idx + 1, max_idx)
    if ws[nxt] == 0.0:
        r = int(np.searchsorted(cdf, cdf[idx], side="right"))
        nxt = idx if r > max_idx else r
    return 0.5 * (float(ys[idx]) + float(ys[nxt]))


def _pinball_loss_arr(y: np.ndarray, pred: np.ndarray, w: np.ndarray, q: float) -> float:
    """Weighted mean pinball loss for quantile level ``q``."""
    diff = y - pred
    loss = np.maximum(q * diff, (q - 1.0) * diff)
    w_sum = float(np.sum(w))
    return float(np.sum(w * loss) / w_sum)


def pinball_and_d2_frame(
    df: pl.DataFrame | pl.LazyFrame,
    spec_name: str,
    slots: list[tuple[str, str, pl.Expr]],
    keep: list[str],
    keys: list[str],
    quantiles: Sequence[float],
    min_samples: int,
    *,
    weight: str | None = None,
    prepare: Callable[[pl.DataFrame], pl.DataFrame] | None = None,
    scored_fn: Callable[[pl.DataFrame], pl.DataFrame],
    weighed_fn: Callable[[pl.DataFrame, str | None], pl.DataFrame],
) -> pl.DataFrame:
    """Compute exact ``pinball_q<τ>`` and ``d2_pinball_q<τ>`` per ``keys``."""
    if isinstance(df, pl.LazyFrame):
        roots = {r for _, _, y in slots for r in y.meta.root_names()}
        needed = {spec_name, *keep, *roots}
        frame = df.select([c for c in df.collect_schema() if c in needed]).collect()
    else:
        frame = df

    col_names = quantile_col_names(quantiles)
    rows_out: list[dict[str, Any]] = []

    for slot, target, y_expr in slots:
        long = frame.select(
            *keep,
            pl.lit(slot).alias("slot"),
            pl.lit(target).alias("target"),
            pl.col(spec_name).struct.field(slot).alias("pred"),
            y_expr.alias("y"),
        )
        long = scored_fn(long)
        if prepare is not None:
            long = prepare(long)
        long = weighed_fn(long, weight)
        if long.is_empty():
            continue
        for key_vals, part in long.group_by(keys, maintain_order=True):
            n = part.height
            row: dict[str, Any] = dict(zip(keys, key_vals, strict=True))
            if n < min_samples:
                for c in col_names:
                    row[c] = None
                rows_out.append(row)
                continue
            y_arr = part["y"].to_numpy().astype(np.float64, copy=False)
            p_arr = part["pred"].to_numpy().astype(np.float64, copy=False)
            w_arr = part["__w"].to_numpy().astype(np.float64, copy=False)
            w_sum = float(np.sum(w_arr))
            for q in quantiles:
                lbl = num_label(q)
                if w_sum <= 0.0:
                    row[f"pinball_q{lbl}"] = None
                    row[f"d2_pinball_q{lbl}"] = None
                    continue
                pb = _pinball_loss_arr(y_arr, p_arr, w_arr, q)
                row[f"pinball_q{lbl}"] = pb
                if n < 2:
                    row[f"d2_pinball_q{lbl}"] = None
                else:
                    y_null = _weighted_percentile_avg(y_arr, w_arr, q)
                    denom = _pinball_loss_arr(y_arr, np.full_like(y_arr, y_null), w_arr, q)
                    row[f"d2_pinball_q{lbl}"] = None if denom <= 0.0 else 1.0 - pb / denom
            rows_out.append(row)

    if not rows_out:
        schema: dict[str, Any] = {}
        for k in keys:
            schema[k] = pl.String if k in ("slot", "target") else frame.schema.get(k, pl.Float64)
        for c in col_names:
            schema[c] = pl.Float64
        return pl.DataFrame(schema=schema)

    schema_overrides = {c: pl.Float64 for c in col_names}
    return pl.DataFrame(rows_out, schema_overrides=schema_overrides)


def _check_positive_int(who: str, name: str, value: object, floor: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        msg = f"{who}: {name} must be a whole number, but it is {value!r}. Pass an integer."
        raise WrongTypeError(msg)
    iv = int(value)
    if iv < floor:
        msg = (
            f"{who}: {name} must be at least {floor}, but it is {iv}. "
            f"Set {name} to {floor} or more."
        )
        raise ValueError(msg)
    return iv


def _parse_coverage_filter(
    who: str, coverage: float | Sequence[float] | None
) -> list[float] | None:
    if coverage is None:
        return None
    if isinstance(coverage, bool):
        msg = (
            f"{who}: coverage must be a number or sequence of numbers in (0, 1), "
            f"got {coverage!r}. Pass a coverage level such as 0.9, or leave coverage out."
        )
        raise WrongTypeError(msg)
    if isinstance(coverage, numbers.Real):
        vals = [float(coverage)]
    elif isinstance(coverage, Sequence) and not isinstance(coverage, (str, bytes)):
        vals = []
        for x in coverage:
            if isinstance(x, bool) or not isinstance(x, numbers.Real):
                msg = (
                    f"{who}: coverage entries must be numbers in (0, 1), got {x!r}. "
                    "Pass numeric coverage levels strictly between 0 and 1."
                )
                raise WrongTypeError(msg)
            vals.append(float(x))
        if not vals:
            msg = (
                f"{who}: coverage cannot be an empty sequence. Pass one or more coverage levels "
                "between 0 and 1, or None."
            )
            raise ValueError(msg)
    else:
        msg = (
            f"{who}: coverage must be a number or sequence of numbers in (0, 1), "
            f"got {coverage!r}. Pass a coverage level such as 0.9, or leave coverage out."
        )
        raise WrongTypeError(msg)
    for v in vals:
        if not (math.isfinite(v) and 0.0 < v < 1.0):
            msg = (
                f"{who}: coverage levels must be strictly between 0 and 1, got {v!r}. "
                "Set each coverage level strictly between 0 and 1."
            )
            raise ValueError(msg)
    return vals


def _conformal_slots(
    probe: pl.DataFrame,
    spec_name: str,
    written: dict[str, Any] | None,
    targets: Sequence[str] | None,
    coverage_filter: list[float] | None,
    who: str,
    slots_fn: Callable[
        [pl.DataFrame, str | dict[str, Any], Sequence[str] | None, str],
        tuple[str, list[tuple[str, str, pl.Expr]]],
    ],
) -> list[tuple[str, str, float, str, str, pl.Expr]]:
    """Resolve ``(slot, target, nominal_coverage, lo_field, hi_field, y_expr)``
    tuples from ``spec`` and the bank struct column."""
    _, pred_slots = slots_fn(probe, written or spec_name, targets, who)
    pred_map = {slot: (target, y_expr) for slot, target, y_expr in pred_slots}
    dtype = probe.schema[spec_name]
    assert isinstance(dtype, pl.Struct)
    field_set = {f.name for f in dtype.fields}

    found: list[tuple[str, str, float, str, str, pl.Expr]] = []
    if written is not None:
        idx = _spec_mod.output_index(written)
        lo_rows = idx.filter(pl.col("kind") == "lo")
        hi_rows = idx.filter(pl.col("kind") == "hi")
        if lo_rows.is_empty():
            msg = (
                f"{who}: spec {spec_name!r} has no conformal interval fields (lo_*, hi_*). "
                "Set conformal (such as 0.9 or [0.8, 0.9]) on the spec."
            )
            raise ValueError(msg)
        # Pair lo/hi rows in output_index order (they are emitted in matching
        # (instance, level, target, combo) order).
        for lo_r, hi_r in zip(
            lo_rows.iter_rows(named=True), hi_rows.iter_rows(named=True), strict=True
        ):
            lo_f = str(lo_r["field"])
            hi_f = str(hi_r["field"])
            lvl = float(lo_r["coverage"])
            # Derive corresponding pred_* slot name from lo field name.
            if (m := _MULTI_LO_RE.match(lo_f)) is not None:
                pred_name = f"pred_{m.group(2)}"
            else:
                pred_name = f"pred_{lo_f.removeprefix('lo_')}"
            if pred_name not in pred_map or lo_f not in field_set or hi_f not in field_set:
                continue
            target, y_expr = pred_map[pred_name]
            found.append((pred_name, target, lvl, lo_f, hi_f, y_expr))
    else:
        # Name-only discovery from struct fields.
        for f in dtype.fields:
            fname = f.name
            if (m := _MULTI_LO_RE.match(fname)) is not None:
                lvl_str, suffix = m.group(1), m.group(2)
                hi_f = f"hi_c{lvl_str}_{suffix}"
                pred_name = f"pred_{suffix}"
                if hi_f in field_set and pred_name in pred_map:
                    try:
                        lvl = float(lvl_str)
                    except ValueError:
                        continue
                    target, y_expr = pred_map[pred_name]
                    found.append((pred_name, target, lvl, fname, hi_f, y_expr))
            elif (m := _SINGLE_LO_RE.match(fname)) is not None:
                suffix = m.group(1)
                hi_f = f"hi_{suffix}"
                pred_name = f"pred_{suffix}"
                if hi_f in field_set and pred_name in pred_map:
                    if coverage_filter is None or len(coverage_filter) != 1:
                        msg = (
                            f"{who}: spec {spec_name!r} was passed by name with single-level "
                            "conformal fields (lo_*, hi_*), which do not encode the nominal "
                            "coverage level. Pass the spec dict as spec, or set coverage= "
                            "(such as coverage=0.9)."
                        )
                        raise ValueError(msg)
                    target, y_expr = pred_map[pred_name]
                    found.append((pred_name, target, coverage_filter[0], fname, hi_f, y_expr))

    if not found:
        msg = (
            f"{who}: output column {spec_name!r} has no conformal interval fields (lo_*, hi_*). "
            "Set conformal (such as 0.9 or [0.8, 0.9]) on the spec."
        )
        raise ValueError(msg)

    if coverage_filter is not None and written is not None:
        avail = sorted({x[2] for x in found})
        want = set(coverage_filter)
        missing = sorted(
            c
            for c in want
            if not any(math.isclose(c, a, rel_tol=0.0, abs_tol=1e-12) for a in avail)
        )
        if missing:
            msg = (
                f"{who}: requested coverage {missing} not present in spec {spec_name!r}, "
                f"which has conformal levels {avail}. Choose from {avail}, or leave coverage out."
            )
            raise ValueError(msg)
        found = [
            x for x in found if any(math.isclose(x[2], c, rel_tol=0.0, abs_tol=1e-12) for c in want)
        ]
    elif coverage_filter is not None and written is None:
        want = set(coverage_filter)
        found = [
            x for x in found if any(math.isclose(x[2], c, rel_tol=0.0, abs_tol=1e-12) for c in want)
        ]
        if not found:
            msg = (
                f"{who}: no conformal interval fields match coverage={coverage_filter}. "
                "Pass a coverage level present in the spec's output, or leave coverage out."
            )
            raise ValueError(msg)

    return found


def _usable(expr: pl.Expr) -> pl.Expr:
    return expr.is_not_null() & expr.is_finite() & (expr.abs() <= _INPUT_BOUND)


def _score_interval_arrays(
    y: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    w: np.ndarray,
    alpha: float,
    ssc_bins: int,
) -> tuple[float | None, float | None, float | None, float | None]:
    """Compute ``(coverage, mean_width, winkler, ssc)`` on un-crossed intervals."""
    w_sum = float(np.sum(w))
    if w_sum <= 0.0:
        return None, None, None, None
    l_arr = np.minimum(lo, hi)
    u_arr = np.maximum(lo, hi)
    widths = u_arr - l_arr
    hits = ((y >= l_arr) & (y <= u_arr)).astype(np.float64)
    pen_lo = np.maximum(0.0, l_arr - y)
    pen_hi = np.maximum(0.0, y - u_arr)
    winkler_arr = widths + (2.0 / alpha) * (pen_lo + pen_hi)

    cov = float(np.sum(w * hits) / w_sum)
    mean_width = float(np.sum(w * widths) / w_sum)
    winkler = float(np.sum(w * winkler_arr) / w_sum)

    # Size-stratified coverage: sort widths ascending, split into ssc_bins
    # equal-count bins (matching MAPIE's regression_ssc / regression_ssc_score).
    n = len(widths)
    if n < ssc_bins:
        ssc: float | None = None
    else:
        order = np.argsort(widths, kind="mergesort")
        bins = np.array_split(order, ssc_bins)
        bin_covs: list[float] = []
        valid = True
        for b in bins:
            if len(b) == 0:
                valid = False
                break
            bw = float(np.sum(w[b]))
            if bw <= 0.0:
                valid = False
                break
            bin_covs.append(float(np.sum(w[b] * hits[b]) / bw))
        ssc = min(bin_covs) if valid and bin_covs else None
    return cov, mean_width, winkler, ssc


def compute_interval_metrics(
    df: pl.DataFrame | pl.LazyFrame,
    spec: str | dict[str, Any],
    *,
    group: str | Iterable[str] | None,
    targets: Sequence[str] | None,
    weight: str | None,
    window_rows: int | None,
    ssc_bins: int,
    min_samples: int,
    coverage: float | Sequence[float] | None,
    spec_of_fn: Callable[[str, str | dict[str, Any]], tuple[str, dict[str, Any] | None]],
    probe_fn: Callable[[pl.DataFrame | pl.LazyFrame], pl.DataFrame],
    check_min_samples_fn: Callable[[str, int], None],
    check_columns_fn: Callable[..., None],
    group_keys_fn: Callable[[str | Iterable[str] | None], list[str]],
    kept_fn: Callable[[pl.DataFrame, Sequence[str]], list[str]],
    slots_fn: Callable[
        [pl.DataFrame, str | dict[str, Any], Sequence[str] | None, str],
        tuple[str, list[tuple[str, str, pl.Expr]]],
    ],
    weighed_fn: Callable[[pl.DataFrame, str | None], pl.DataFrame],
) -> pl.DataFrame:
    """Backend for :func:`rheon.polars.eval.interval_metrics`."""
    who = "rp.eval.interval_metrics"
    spec_name, written = spec_of_fn(who, spec)
    check_min_samples_fn(who, min_samples)
    ssc_bins_i = _check_positive_int(who, "ssc_bins", ssc_bins, floor=1)
    win_rows_i = (
        None
        if window_rows is None
        else _check_positive_int(who, "window_rows", window_rows, floor=1)
    )
    cov_filter = _parse_coverage_filter(who, coverage)
    probe = probe_fn(df)
    check_columns_fn(probe, who, group=group, weight=weight, produced=INTERVAL_PRODUCED)
    conf_slots = _conformal_slots(probe, spec_name, written, targets, cov_filter, who, slots_fn)

    gkeys = group_keys_fn(group)
    keep = kept_fn(probe, [*gkeys, *([] if weight is None else [weight])])
    if isinstance(df, pl.LazyFrame):
        roots = {r for _, _, _, _, _, y in conf_slots for r in y.meta.root_names()}
        needed = {spec_name, *keep, *roots}
        frame = df.select([c for c in df.collect_schema() if c in needed]).collect()
    else:
        frame = df

    out_keys = ["slot", "target", "nominal_coverage", *gkeys]
    if win_rows_i is not None:
        out_keys.append("window_start")

    rows_out: list[dict[str, Any]] = []
    for slot, target, level, lo_f, hi_f, y_expr in conf_slots:
        alpha = 1.0 - level
        long = frame.select(
            *keep,
            pl.col(spec_name).struct.field(lo_f).cast(pl.Float64).alias("__lo"),
            pl.col(spec_name).struct.field(hi_f).cast(pl.Float64).alias("__hi"),
            y_expr.cast(pl.Float64).alias("y"),
        )
        long = long.filter(_usable(pl.col("y")) & _usable(pl.col("__lo")) & _usable(pl.col("__hi")))
        long = weighed_fn(long, weight)
        if long.is_empty():
            continue

        if gkeys:
            groups_iter: Iterable[tuple[tuple[Any, ...], pl.DataFrame]] = long.group_by(
                gkeys, maintain_order=True
            )
        else:
            groups_iter = [((), long)]

        for gvals, part in groups_iter:
            y_all = part["y"].to_numpy().astype(np.float64, copy=False)
            lo_all = part["__lo"].to_numpy().astype(np.float64, copy=False)
            hi_all = part["__hi"].to_numpy().astype(np.float64, copy=False)
            w_all = part["__w"].to_numpy().astype(np.float64, copy=False)

            if win_rows_i is None:
                slices: list[tuple[int | None, int, int]] = [(None, 0, len(y_all))]
            else:
                slices = [
                    (start, start, min(start + win_rows_i, len(y_all)))
                    for start in range(0, len(y_all), win_rows_i)
                ]

            for w_start, s0, s1 in slices:
                n = s1 - s0
                rec: dict[str, Any] = {
                    "slot": slot,
                    "target": target,
                    "nominal_coverage": level,
                }
                for gk, gv in zip(gkeys, gvals, strict=True):
                    rec[gk] = gv
                if w_start is not None:
                    rec["window_start"] = w_start
                rec["n"] = n
                if n < min_samples:
                    rec["coverage"] = None
                    rec["mean_width"] = None
                    rec["winkler"] = None
                    rec["ssc"] = None
                else:
                    cov, mw, wink, ssc = _score_interval_arrays(
                        y_all[s0:s1], lo_all[s0:s1], hi_all[s0:s1], w_all[s0:s1], alpha, ssc_bins_i
                    )
                    rec["coverage"] = cov
                    rec["mean_width"] = mw
                    rec["winkler"] = wink
                    rec["ssc"] = ssc
                rows_out.append(rec)

    metric_cols = ("n", "coverage", "mean_width", "winkler", "ssc")
    if not rows_out:
        schema: dict[str, Any] = {
            "slot": pl.String,
            "target": pl.String,
            "nominal_coverage": pl.Float64,
        }
        for gk in gkeys:
            schema[gk] = frame.schema[gk]
        if win_rows_i is not None:
            schema["window_start"] = pl.Int64
        schema["n"] = pl.Int64
        for mc in metric_cols[1:]:
            schema[mc] = pl.Float64
        return pl.DataFrame(schema=schema)

    overrides: dict[str, Any] = {
        "slot": pl.String,
        "target": pl.String,
        "nominal_coverage": pl.Float64,
        "n": pl.Int64,
        "coverage": pl.Float64,
        "mean_width": pl.Float64,
        "winkler": pl.Float64,
        "ssc": pl.Float64,
    }
    if win_rows_i is not None:
        overrides["window_start"] = pl.Int64
    return (
        pl.DataFrame(rows_out, schema_overrides=overrides)
        .select(*out_keys, *metric_cols)
        .sort(out_keys)
    )
