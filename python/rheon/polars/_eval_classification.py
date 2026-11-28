"""Streaming binary classification metrics (``classification_metrics`` and
``reliability_bins``) for :mod:`rheon.polars.eval`.

Split into a private helper module to keep ``eval.py`` well under the source
size cap while sharing its validation, slot-unpacking, and ``min_samples``
conventions.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import Any

import polars as pl

from rheon import _rheon as _native
from rheon._errors import WrongTypeError
from rheon.polars._errors import ColumnNotFoundError

#: Output column names produced by :func:`classification_metrics`.
CLASS_METRIC_COLS = (
    "n",
    "n_pos",
    "n_neg",
    "weight_sum",
    "pos_rate",
    "roc_auc",
    "pr_auc",
    "ece",
    "mce",
    "brier",
)

#: Derived score columns nulled out when ``n < min_samples``.
_THINNED_SCORE_COLS = (
    "pos_rate",
    "roc_auc",
    "pr_auc",
    "ece",
    "mce",
    "brier",
)

#: Output column names produced by :func:`reliability_bins`.
RELIABILITY_BIN_COLS = (
    "bin",
    "bin_lo",
    "bin_hi",
    "n",
    "weight_sum",
    "mean_pred",
    "mean_actual",
    "abs_gap",
)

#: Column names reserved for ``group`` / ``weight`` validation in spec mode.
CLASS_PRODUCED = ("slot", "target", "pred", "y", *CLASS_METRIC_COLS)
BINS_PRODUCED = ("slot", "target", "pred", "y", *RELIABILITY_BIN_COLS)

_STATE_BYTES_CEILING = 64 * 1024 * 1024


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


def _check_compression(who: str, compression: object) -> float:
    if isinstance(compression, bool) or not isinstance(compression, numbers.Real):
        msg = (
            f"{who}: compression must be a finite number of at least 10, "
            f"but it is {compression!r}. Pass a finite number such as 200."
        )
        raise WrongTypeError(msg)
    cf = float(compression)
    if not (math.isfinite(cf) and cf >= 10.0):
        msg = (
            f"{who}: compression must be a finite number of at least 10, "
            f"but it is {compression!r}. Pass 200 for tight ROC-AUC and PR-AUC accuracy."
        )
        raise ValueError(msg)
    return cf


def _check_max_state_bytes(who: str, max_state_bytes: object) -> int | None:
    if max_state_bytes is None:
        return None
    if isinstance(max_state_bytes, bool) or not isinstance(max_state_bytes, numbers.Integral):
        msg = (
            f"{who}: max_state_bytes must be a whole number of bytes or None, "
            f"but it is {max_state_bytes!r}. Pass an integer byte cap, or None."
        )
        raise WrongTypeError(msg)
    iv = int(max_state_bytes)
    if iv <= 0 or iv > _STATE_BYTES_CEILING:
        msg = (
            f"{who}: max_state_bytes must be above 0 and at most {_STATE_BYTES_CEILING} "
            f"(64 MiB), but it is {iv}. Set max_state_bytes between 1 and 64 MiB, or None."
        )
        raise ValueError(msg)
    return iv


def _check_chunk_size(who: str, chunk_size: object) -> int | None:
    if chunk_size is None:
        return None
    return _check_positive_int(who, "chunk_size", chunk_size, floor=1)


def _check_plain_column(
    probe: pl.DataFrame,
    who: str,
    role: str,
    col: object,
    *,
    allow_bool: bool = False,
) -> str:
    if not isinstance(col, str):
        msg = (
            f"{who}: {role} must be a column name (str), but it is of type {type(col).__name__}. "
            f"Pass the name of a column in the frame."
        )
        raise WrongTypeError(msg)
    if col not in probe.schema:
        msg = (
            f"{who}: {role} column {col!r} is not in the frame, whose columns are "
            f"{probe.columns}. Name a column of the frame as {role}."
        )
        raise ColumnNotFoundError(msg)
    dtype = probe.schema[col]
    ok = dtype.is_numeric() or (allow_bool and dtype == pl.Boolean)
    if not ok:
        expected = "numeric or boolean" if allow_bool else "numeric"
        msg = (
            f"{who}: {role} column {col!r} must be {expected}, but it is {dtype}. "
            f"Cast {col!r} to Float64."
        )
        raise WrongTypeError(msg)
    return col


def _resolve_inputs(
    probe: pl.DataFrame,
    who: str,
    spec: str | dict[str, Any] | None,
    pred: str | None,
    target: str | None,
    targets: Sequence[str] | None,
    group: str | Iterable[str] | None,
    weight: str | None,
    produced_spec: tuple[str, ...],
    produced_plain: tuple[str, ...],
    *,
    group_keys_fn: Callable[[str | Iterable[str] | None], list[str]],
    check_columns_fn: Callable[..., None],
    slots_fn: Callable[
        [pl.DataFrame, str | dict[str, Any], Sequence[str] | None, str],
        tuple[str, list[tuple[str, str, pl.Expr]]],
    ],
) -> tuple[
    bool,
    str | None,
    list[tuple[str, str, pl.Expr, pl.Expr]],
    list[str],
    list[str],
]:
    """Validate and resolve spec mode vs plain-column mode inputs against ``probe``.

    Returns ``(is_spec_mode, spec_name, entries, gkeys, needed_cols)`` where each
    entry is ``(slot_label, target_label, pred_expr, y_expr)``.
    """
    gkeys = group_keys_fn(group)

    if spec is not None:
        if pred is not None or target is not None:
            msg = (
                f"{who}: pass either spec, or pred and target, not both. "
                "Leave out pred and target when scoring a bank spec."
            )
            raise ValueError(msg)
        check_columns_fn(probe, who, group=group, weight=weight, produced=produced_spec)
        if weight is not None and not probe.schema[weight].is_numeric():
            msg = (
                f"{who}: weight column {weight!r} must be numeric, but it is "
                f"{probe.schema[weight]}. Cast {weight!r} to Float64."
            )
            raise WrongTypeError(msg)
        spec_name, raw_slots = slots_fn(probe, spec, targets, who)
        entries = [
            (slot, tgt, pl.col(spec_name).struct.field(slot), y_expr)
            for slot, tgt, y_expr in raw_slots
        ]
        target_roots = {r for _, _, y_expr in raw_slots for r in y_expr.meta.root_names()}
        needed_set = {spec_name, *gkeys, *target_roots}
        if weight is not None:
            needed_set.add(weight)
        needed = [c for c in probe.columns if c in needed_set]
        return True, spec_name, entries, gkeys, needed

    # Plain-column mode.
    if pred is None or target is None:
        msg = (
            f"{who}: pass either spec, or both pred and target. "
            "For plain columns, pass pred='...' and target='...'."
        )
        raise ValueError(msg)
    if targets is not None:
        msg = (
            f"{who}: targets applies only when spec is given, not with plain pred/target columns. "
            "Leave out targets."
        )
        raise ValueError(msg)
    pred_col = _check_plain_column(probe, who, "pred", pred, allow_bool=False)
    target_col = _check_plain_column(probe, who, "target", target, allow_bool=True)
    # Check group columns against plain output names (no `slot`/`target` collision unless produced).
    check_columns_fn(probe, who, group=group, produced=produced_plain)
    if weight is not None:
        _check_plain_column(probe, who, "weight", weight, allow_bool=False)
    entries = [(pred_col, target_col, pl.col(pred_col), pl.col(target_col))]
    needed_set = {pred_col, target_col, *gkeys}
    if weight is not None:
        needed_set.add(weight)
    needed = [c for c in probe.columns if c in needed_set]
    return False, None, entries, gkeys, needed


def _iter_batches(
    df: pl.DataFrame | pl.LazyFrame,
    needed: list[str],
    chunk_size: int | None,
) -> Iterator[pl.DataFrame]:
    """Yield ordered chunks of ``df`` projected to ``needed`` columns."""
    if isinstance(df, pl.LazyFrame):
        eff = int(_native.default_chunk_size()) if chunk_size is None else chunk_size
        yield from df.select(needed).collect_batches(chunk_size=eff, maintain_order=True)
    else:
        sub = df.select(needed)
        if sub.is_empty():
            return
        if chunk_size is None:
            yield sub
        else:
            yield from sub.iter_slices(chunk_size)


def _stream_accumulators(
    df: pl.DataFrame | pl.LazyFrame,
    who: str,
    is_spec_mode: bool,
    entries: list[tuple[str, str, pl.Expr, pl.Expr]],
    gkeys: list[str],
    weight: str | None,
    needed: list[str],
    chunk_size: int | None,
    factory: Callable[[], _native.ClassMetrics],
) -> dict[tuple[Any, ...], _native.ClassMetrics]:
    """Stream batches of ``df`` into per-key Rust ``ClassMetrics`` accumulators.

    Feeding rows in exact arrival order within each key guarantees bit-for-bit
    chunk invariance on both ``DataFrame`` and ``LazyFrame``.
    """
    accs: dict[tuple[Any, ...], _native.ClassMetrics] = {}

    def get_acc(key: tuple[Any, ...]) -> _native.ClassMetrics:
        acc = accs.get(key)
        if acc is None:
            acc = factory()
            accs[key] = acc
        return acc

    for batch in _iter_batches(df, needed, chunk_size):
        if batch.is_empty():
            continue
        if not gkeys:
            w_series = batch[weight] if weight is not None else None
            for slot_lbl, tgt_lbl, pred_expr, y_expr in entries:
                pair = batch.select(pred_expr.alias("__p"), y_expr.alias("__y"))
                key: tuple[Any, ...] = (slot_lbl, tgt_lbl) if is_spec_mode else ()
                acc = get_acc(key)
                try:
                    acc.update_batch(pair["__p"], pair["__y"], w_series)
                except ValueError as exc:
                    raise ValueError(
                        _format_label_error(who, is_spec_mode, slot_lbl, tgt_lbl, exc)
                    ) from None
        else:
            # Project all needed expressions once per batch so group_by splits
            # aligned columns without re-evaluating struct field extractions per group.
            exprs: list[pl.Expr] = [pl.col(k) for k in gkeys]
            if weight is not None:
                exprs.append(pl.col(weight).alias("__w"))
            for idx, (_, _, pred_expr, y_expr) in enumerate(entries):
                exprs.append(pred_expr.alias(f"__p_{idx}"))
                exprs.append(y_expr.alias(f"__y_{idx}"))
            proj = batch.select(exprs)
            for key_vals, part in proj.group_by(gkeys, maintain_order=True):
                w_part = part["__w"] if weight is not None else None
                for idx, (slot_lbl, tgt_lbl, _, _) in enumerate(entries):
                    key = (slot_lbl, tgt_lbl, *key_vals) if is_spec_mode else tuple(key_vals)
                    acc = get_acc(key)
                    try:
                        acc.update_batch(part[f"__p_{idx}"], part[f"__y_{idx}"], w_part)
                    except ValueError as exc:
                        raise ValueError(
                            _format_label_error(who, is_spec_mode, slot_lbl, tgt_lbl, exc)
                        ) from None
    return accs


def _format_label_error(
    who: str,
    is_spec_mode: bool,
    slot_lbl: str,
    tgt_lbl: str,
    exc: ValueError,
) -> str:
    if is_spec_mode:
        return f"{who}: slot {slot_lbl!r} (target {tgt_lbl!r}): {exc}"
    return f"{who}: target column {tgt_lbl!r}: {exc}"


def _key_schema(
    probe: pl.DataFrame,
    is_spec_mode: bool,
    gkeys: list[str],
) -> tuple[list[str], dict[str, Any]]:
    keys = (["slot", "target", *gkeys]) if is_spec_mode else list(gkeys)
    schema: dict[str, Any] = {}
    if is_spec_mode:
        schema["slot"] = pl.String
        schema["target"] = pl.String
    for g in gkeys:
        schema[g] = probe.schema[g]
    return keys, schema


def compute_classification_metrics(
    df: pl.DataFrame | pl.LazyFrame,
    spec: str | dict[str, Any] | None = None,
    *,
    pred: str | None = None,
    target: str | None = None,
    group: str | Iterable[str] | None = None,
    targets: Sequence[str] | None = None,
    weight: str | None = None,
    min_samples: int = 30,
    bins: int = 10,
    compression: float = 200.0,
    max_state_bytes: int | None = None,
    chunk_size: int | None = None,
    probe_fn: Callable[[pl.DataFrame | pl.LazyFrame], pl.DataFrame],
    check_min_samples_fn: Callable[[str, object], None],
    check_columns_fn: Callable[..., None],
    group_keys_fn: Callable[[str | Iterable[str] | None], list[str]],
    slots_fn: Callable[
        [pl.DataFrame, str | dict[str, Any], Sequence[str] | None, str],
        tuple[str, list[tuple[str, str, pl.Expr]]],
    ],
) -> pl.DataFrame:
    """Implementation of :func:`rheon.polars.eval.classification_metrics`."""
    who = "rp.eval.classification_metrics"
    check_min_samples_fn(who, min_samples)
    nb = _check_positive_int(who, "bins", bins, floor=1)
    comp = _check_compression(who, compression)
    cap = _check_max_state_bytes(who, max_state_bytes)
    cs = _check_chunk_size(who, chunk_size)

    # Validate Rust accumulator sizing upfront before executing any LazyFrame.
    try:
        _native.ClassMetrics(bins=nb, compression=comp, max_state_bytes=cap, bins_only=False)
    except ValueError as exc:
        raise ValueError(f"{who}: {exc}") from None

    probe = probe_fn(df)
    is_spec_mode, _, entries, gkeys, needed = _resolve_inputs(
        probe,
        who,
        spec,
        pred,
        target,
        targets,
        group,
        weight,
        CLASS_PRODUCED,
        CLASS_METRIC_COLS,
        group_keys_fn=group_keys_fn,
        check_columns_fn=check_columns_fn,
        slots_fn=slots_fn,
    )

    def make_acc() -> _native.ClassMetrics:
        return _native.ClassMetrics(bins=nb, compression=comp, max_state_bytes=cap, bins_only=False)

    accs = _stream_accumulators(df, who, is_spec_mode, entries, gkeys, weight, needed, cs, make_acc)
    keys, key_schema = _key_schema(probe, is_spec_mode, gkeys)

    out_schema: dict[str, Any] = {
        **key_schema,
        "n": pl.Int64,
        "n_pos": pl.Int64,
        "n_neg": pl.Int64,
        "weight_sum": pl.Float64,
        "pos_rate": pl.Float64,
        "roc_auc": pl.Float64,
        "pr_auc": pl.Float64,
        "ece": pl.Float64,
        "mce": pl.Float64,
        "brier": pl.Float64,
    }

    rows: list[dict[str, Any]] = []
    for key, acc in accs.items():
        if acc.n == 0:
            continue
        n, n_pos, n_neg, w_sum, pos_rate, roc_auc, pr_auc, ece, mce, brier = acc.summary()
        row: dict[str, Any] = dict(zip(keys, key, strict=True))
        row["n"] = n
        row["n_pos"] = n_pos
        row["n_neg"] = n_neg
        row["weight_sum"] = w_sum
        if n < min_samples:
            for col in _THINNED_SCORE_COLS:
                row[col] = None
        else:
            row["pos_rate"] = pos_rate
            row["roc_auc"] = roc_auc
            row["pr_auc"] = pr_auc
            row["ece"] = ece
            row["mce"] = mce
            row["brier"] = brier
        rows.append(row)

    if not rows:
        return pl.DataFrame(schema=out_schema)
    out = pl.DataFrame(rows, schema=out_schema)
    return out.sort(keys) if keys else out


def compute_reliability_bins(
    df: pl.DataFrame | pl.LazyFrame,
    spec: str | dict[str, Any] | None = None,
    *,
    pred: str | None = None,
    target: str | None = None,
    group: str | Iterable[str] | None = None,
    targets: Sequence[str] | None = None,
    weight: str | None = None,
    bins: int = 10,
    chunk_size: int | None = None,
    probe_fn: Callable[[pl.DataFrame | pl.LazyFrame], pl.DataFrame],
    check_columns_fn: Callable[..., None],
    group_keys_fn: Callable[[str | Iterable[str] | None], list[str]],
    slots_fn: Callable[
        [pl.DataFrame, str | dict[str, Any], Sequence[str] | None, str],
        tuple[str, list[tuple[str, str, pl.Expr]]],
    ],
) -> pl.DataFrame:
    """Implementation of :func:`rheon.polars.eval.reliability_bins`."""
    who = "rp.eval.reliability_bins"
    nb = _check_positive_int(who, "bins", bins, floor=1)
    cs = _check_chunk_size(who, chunk_size)

    try:
        _native.ClassMetrics(bins=nb, bins_only=True)
    except ValueError as exc:
        raise ValueError(f"{who}: {exc}") from None

    probe = probe_fn(df)
    is_spec_mode, _, entries, gkeys, needed = _resolve_inputs(
        probe,
        who,
        spec,
        pred,
        target,
        targets,
        group,
        weight,
        BINS_PRODUCED,
        RELIABILITY_BIN_COLS,
        group_keys_fn=group_keys_fn,
        check_columns_fn=check_columns_fn,
        slots_fn=slots_fn,
    )

    def make_acc() -> _native.ClassMetrics:
        return _native.ClassMetrics(bins=nb, bins_only=True)

    accs = _stream_accumulators(df, who, is_spec_mode, entries, gkeys, weight, needed, cs, make_acc)
    keys, key_schema = _key_schema(probe, is_spec_mode, gkeys)

    out_schema: dict[str, Any] = {
        **key_schema,
        "bin": pl.Int64,
        "bin_lo": pl.Float64,
        "bin_hi": pl.Float64,
        "n": pl.Int64,
        "weight_sum": pl.Float64,
        "mean_pred": pl.Float64,
        "mean_actual": pl.Float64,
        "abs_gap": pl.Float64,
    }

    rows: list[dict[str, Any]] = []
    for key, acc in accs.items():
        if acc.n == 0:
            continue
        key_dict = dict(zip(keys, key, strict=True))
        for b, blo, bhi, bn, bw, mp, ma, gap in acc.bins():
            row: dict[str, Any] = dict(key_dict)
            row["bin"] = b
            row["bin_lo"] = blo
            row["bin_hi"] = bhi
            row["n"] = bn
            row["weight_sum"] = bw
            row["mean_pred"] = mp
            row["mean_actual"] = ma
            row["abs_gap"] = gap
            rows.append(row)

    if not rows:
        return pl.DataFrame(schema=out_schema)
    out = pl.DataFrame(rows, schema=out_schema)
    return out.sort([*keys, "bin"])
