"""Stateless basis-expansion operators as plain Polars expressions.

Four builders, each returning a list of ``pl.Expr`` that depend on each row's
own values alone: no state, no window, streaming and chunk-invariant in any
Polars query.

- :func:`bspline`: clamped B-spline basis of any degree at given knots,
  matching ``scipy.interpolate.BSpline.design_matrix`` and R's ``splines::bs``.
- :func:`natural_spline`: QR-projected natural cubic B-spline basis with zero
  second derivative at both boundary knots and exact linear extrapolation
  outside them, matching R's ``splines::ns``.
- :func:`hinge`: piecewise-linear hinge terms ``max(0, x - k)`` at given knots.
- :func:`interactions`: polynomial interaction products up to a given degree,
  matching scikit-learn's ``PolynomialFeatures`` beyond the degree-1 columns.

All basis construction (clamped knot vector, boundary derivatives, and the
Householder QR nullspace projection for natural splines) runs in pure Python
at expression-build time, with no NumPy dependency.
"""

from __future__ import annotations

import itertools
import math
import numbers
from collections.abc import Sequence
from typing import Literal

import polars as pl

from rheon._errors import WrongTypeError

#: The extrapolation modes :func:`bspline` accepts.
_EXTRAPOLATE_MODES = ("constant", "linear", "null")

#: Upper bound on spline degree so the Cox-de Boor expression graph stays compact.
_MAX_DEGREE = 10

#: Upper bound on interaction order to prevent accidental combinatorial blowup.
_MAX_INTERACTION_DEGREE = 6


def _resolve_input(
    col: str | pl.Expr,
    prefix: str | None,
    default_suffix: str,
    who: str,
) -> tuple[pl.Expr, str]:
    """Validate ``col`` and ``prefix`` and return ``(float64_expr, name_prefix)``."""
    if isinstance(col, str):
        if not col:
            raise ValueError(
                f"{who}: col must be a non-empty column name, but it is ''. "
                "Pass the name of a numeric column, or a Polars expression."
            )
        expr = pl.col(col).cast(pl.Float64)
        base_name = col
    elif isinstance(col, pl.Expr):
        if col.meta.has_multiple_outputs():
            raise ValueError(
                f"{who}: col must expand to a single column, but it has multiple outputs. "
                "Pass one column name or a single-column expression."
            )
        try:
            base_name = col.meta.output_name()
        except Exception as err:  # noqa: BLE001
            raise ValueError(
                f"{who}: col has no single output name ({err}). "
                "Name the expression with .alias(...) or pass prefix."
            ) from None
        expr = col.cast(pl.Float64)
    else:
        raise WrongTypeError(
            f"{who}: col must be a column name or a pl.Expr, but it is {col!r}. "
            "Pass a column name as a str, or a Polars expression."
        )

    if prefix is None:
        out_prefix = f"{base_name}_{default_suffix}"
    elif isinstance(prefix, str):
        if not prefix:
            raise ValueError(
                f"{who}: prefix must be a non-empty string, but it is ''. "
                "Pass a non-empty prefix, or leave prefix as None."
            )
        out_prefix = prefix
    else:
        raise WrongTypeError(
            f"{who}: prefix must be a str or None, but it is {prefix!r}. "
            "Pass a string prefix, or leave prefix as None."
        )
    return expr, out_prefix


def _validate_knots(
    knots: Sequence[float],
    *,
    min_count: int,
    who: str,
) -> tuple[float, ...]:
    """Validate strictly increasing finite knots and return them as floats."""
    if isinstance(knots, (str, bytes)) or not isinstance(knots, Sequence):
        raise WrongTypeError(
            f"{who}: knots must be a sequence of numbers, but it is {knots!r}. "
            "Pass a list or tuple of strictly increasing finite numbers."
        )
    if len(knots) < min_count:
        raise ValueError(
            f"{who}: knots must hold at least {min_count} values, but it has {len(knots)}. "
            f"Pass at least {min_count} strictly increasing finite knots."
        )
    vals: list[float] = []
    for i, raw in enumerate(knots):
        if isinstance(raw, bool) or not isinstance(raw, numbers.Real):
            raise WrongTypeError(
                f"{who}: knots[{i}] must be a real number, but it is {raw!r}. "
                "Pass finite numeric knot positions."
            )
        v = float(raw)
        if not math.isfinite(v):
            raise ValueError(
                f"{who}: knots[{i}] must be finite, but it is {raw!r}. "
                "Pass finite numeric knot positions."
            )
        if vals and v <= vals[-1]:
            raise ValueError(
                f"{who}: knots must be strictly increasing, but knots[{i}] is not above "
                f"knots[{i - 1}]. Sort and deduplicate the knot positions."
            )
        vals.append(v)
    return tuple(vals)


def _check_bool(value: object, name: str, who: str) -> bool:
    if not isinstance(value, bool):
        raise WrongTypeError(
            f"{who}: {name} must be True or False, but it is {value!r}. "
            f"Pass {name}=True or {name}=False."
        )
    return value


def _check_int_range(value: object, name: str, lo: int, hi: int, who: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise WrongTypeError(
            f"{who}: {name} must be an int, but it is {value!r}. Pass an integer from {lo} to {hi}."
        )
    if value < lo or value > hi:
        raise ValueError(
            f"{who}: {name} must be between {lo} and {hi}, but it is {value}. "
            f"Set {name} to an integer from {lo} to {hi}."
        )
    return value


def _clamped_knots(knots: tuple[float, ...], degree: int) -> tuple[float, ...]:
    """Full clamped knot vector with boundary multiplicity ``degree + 1``."""
    return (knots[0],) * degree + knots + (knots[-1],) * degree


def _bspline_scalar_eval(
    x_val: float,
    knots: tuple[float, ...],
    degree: int,
) -> tuple[list[float], list[float], list[float]]:
    """Evaluate clamped B-spline basis values and 1st and 2nd derivatives at ``x_val``."""
    t = _clamped_knots(knots, degree)
    m = len(knots) - 1
    n_t = len(t)

    # Degree-0 basis on the full knot vector `t`; only spans d .. d + m - 1 have positive width.
    levels: list[list[float]] = []
    b0 = [0.0] * (n_t - 1)
    for s in range(m):
        idx = degree + s
        lo, hi = t[idx], t[idx + 1]
        if (lo <= x_val < hi) or (s == m - 1 and x_val == hi):
            b0[idx] = 1.0
    levels.append(b0)

    for p in range(1, degree + 1):
        prev = levels[-1]
        cur = [0.0] * (n_t - p - 1)
        for i in range(len(cur)):
            val = 0.0
            d1 = t[i + p] - t[i]
            if d1 > 0.0 and prev[i] != 0.0:
                val += ((x_val - t[i]) / d1) * prev[i]
            d2 = t[i + p + 1] - t[i + 1]
            if d2 > 0.0 and prev[i + 1] != 0.0:
                val += ((t[i + p + 1] - x_val) / d2) * prev[i + 1]
            cur[i] = val
        levels.append(cur)

    vals = levels[degree]
    n_basis = len(vals)
    if degree == 0:
        return vals, [0.0] * n_basis, [0.0] * n_basis

    # First derivative via de Boor's derivative recurrence.
    d1_levels: dict[int, list[float]] = {}
    for p in range(1, degree + 1):
        prev = levels[p - 1]
        cur = [0.0] * (n_t - p - 1)
        for i in range(len(cur)):
            left = prev[i] / (t[i + p] - t[i]) if t[i + p] > t[i] else 0.0
            right = prev[i + 1] / (t[i + p + 1] - t[i + 1]) if t[i + p + 1] > t[i + 1] else 0.0
            cur[i] = float(p) * (left - right)
        d1_levels[p] = cur
    d1_vals = d1_levels[degree]

    if degree == 1:
        return vals, d1_vals, [0.0] * n_basis

    prev_d1 = d1_levels[degree - 1]
    d2_vals = [0.0] * n_basis
    for i in range(n_basis):
        left = prev_d1[i] / (t[i + degree] - t[i]) if t[i + degree] > t[i] else 0.0
        right = (
            prev_d1[i + 1] / (t[i + degree + 1] - t[i + 1]) if t[i + degree + 1] > t[i + 1] else 0.0
        )
        d2_vals[i] = float(degree) * (left - right)
    return vals, d1_vals, d2_vals


def _bspline_interior_exprs(
    x_clamped: pl.Expr,
    knots: tuple[float, ...],
    degree: int,
) -> list[pl.Expr]:
    """Build pure Polars Cox-de Boor expressions on ``x_clamped`` in ``[k_0, k_M]``."""
    t = _clamped_knots(knots, degree)
    m = len(knots) - 1
    n_t = len(t)

    # Active degree-0 span indicators partition [k_0, k_M] with exact exclusivity.
    basis: list[pl.Expr | None] = [None] * (n_t - 1)
    if m == 1:
        basis[degree] = pl.lit(1.0, dtype=pl.Float64)
    else:
        for s in range(m):
            idx = degree + s
            if s == 0:
                cond = x_clamped < knots[1]
            elif s == m - 1:
                cond = x_clamped >= knots[m - 1]
            else:
                cond = (x_clamped >= knots[s]) & (x_clamped < knots[s + 1])
            basis[idx] = cond.cast(pl.Float64)

    for p in range(1, degree + 1):
        nxt: list[pl.Expr | None] = [None] * (n_t - p - 1)
        for i in range(len(nxt)):
            left_expr = basis[i]
            right_expr = basis[i + 1]
            d1 = t[i + p] - t[i]
            d2 = t[i + p + 1] - t[i + 1]
            left_term = (
                ((x_clamped - t[i]) / d1) * left_expr
                if left_expr is not None and d1 > 0.0
                else None
            )
            right_term = (
                ((t[i + p + 1] - x_clamped) / d2) * right_expr
                if right_expr is not None and d2 > 0.0
                else None
            )
            if left_term is not None and right_term is not None:
                nxt[i] = left_term + right_term
            elif left_term is not None:
                nxt[i] = left_term
            elif right_term is not None:
                nxt[i] = right_term
        basis = nxt

    return [expr if expr is not None else pl.lit(0.0, dtype=pl.Float64) for expr in basis]


def _linear_arm(x: pl.Expr, edge: float, val: float, slope: float) -> pl.Expr:
    """Evaluate ``val + (x - edge) * slope`` without ``0 * inf`` when ``slope == 0``."""
    if slope == 0.0:
        return pl.lit(val, dtype=pl.Float64)
    if val == 0.0:
        return (x - edge) * slope
    return (x - edge) * slope + val


def _wrap_extrapolate(
    x: pl.Expr,
    interior: pl.Expr,
    k_lo: float,
    k_hi: float,
    extrapolate: Literal["constant", "linear", "null"],
    val_lo: float,
    slope_lo: float,
    val_hi: float,
    slope_hi: float,
) -> pl.Expr:
    """Preserve null and NaN and apply the chosen boundary extrapolation."""
    if extrapolate == "constant":
        return (
            pl.when(x.is_null())
            .then(pl.lit(None, dtype=pl.Float64))
            .when(x.is_nan())
            .then(pl.lit(float("nan"), dtype=pl.Float64))
            .otherwise(interior)
        )
    if extrapolate == "null":
        return (
            pl.when(x.is_null())
            .then(pl.lit(None, dtype=pl.Float64))
            .when(x.is_nan())
            .then(pl.lit(float("nan"), dtype=pl.Float64))
            .when((x < k_lo) | (x > k_hi))
            .then(pl.lit(None, dtype=pl.Float64))
            .otherwise(interior)
        )
    return (
        pl.when(x.is_null())
        .then(pl.lit(None, dtype=pl.Float64))
        .when(x.is_nan())
        .then(pl.lit(float("nan"), dtype=pl.Float64))
        .when(x < k_lo)
        .then(_linear_arm(x, k_lo, val_lo, slope_lo))
        .when(x > k_hi)
        .then(_linear_arm(x, k_hi, val_hi, slope_hi))
        .otherwise(interior)
    )


def _nullspace_qr_2rows(row0: list[float], row1: list[float]) -> list[list[float]]:
    """Orthonormal basis ``Z`` (shape ``n x (n - 2)``) for ``ker([row0; row1])``.

    Pure-Python Householder QR of the normalized ``n x 2`` transpose ``C^T``:
    columns ``2 .. n - 1`` of ``Q = H_0 H_1`` span ``ker(C)`` and match
    LAPACK/R's ``qr.qty`` convention without importing NumPy.
    """
    n = len(row0)
    # Row-normalize so the QR is scale-invariant across extreme knot scales.
    n0 = math.hypot(*row0)
    n1 = math.hypot(*row1)
    a0 = [v / n0 for v in row0]
    a1 = [v / n1 for v in row1]

    # First Householder reflector H_0 eliminating a0[1:].
    norm0 = math.hypot(*a0)
    sign0 = 1.0 if a0[0] >= 0.0 else -1.0
    v0 = [a0[0] + sign0 * norm0, *a0[1:]]
    v0_norm_sq = math.hypot(*v0) ** 2
    beta0 = 2.0 / v0_norm_sq

    # Apply H_0 to a1.
    dot01 = math.fsum(v0[i] * a1[i] for i in range(n))
    a1_rot = [a1[i] - beta0 * dot01 * v0[i] for i in range(n)]

    # Second Householder reflector H_1 on a1_rot[1:].
    sub1 = a1_rot[1:]
    norm1 = math.hypot(*sub1)
    sign1 = 1.0 if sub1[0] >= 0.0 else -1.0
    v1 = [sub1[0] + sign1 * norm1, *sub1[1:]]
    v1_norm_sq = math.hypot(*v1) ** 2
    beta1 = 2.0 / v1_norm_sq

    # Form Z = Q[:, 2:] = H_0 (H_1 E[:, 2:]) column by column.
    cols: list[list[float]] = []
    for col_idx in range(2, n):
        e = [0.0] * n
        e[col_idx] = 1.0
        d1 = v1[col_idx - 1]
        if d1 != 0.0:
            scale1 = beta1 * d1
            for k in range(n - 1):
                if v1[k] != 0.0:
                    e[k + 1] -= scale1 * v1[k]
        d0 = math.fsum(v0[k] * e[k] for k in range(n))
        if d0 != 0.0:
            scale0 = beta0 * d0
            for k in range(n):
                if v0[k] != 0.0:
                    e[k] -= scale0 * v0[k]
        cols.append(e)

    # Return row-major Z[i][j] for i in range(n), j in range(n - 2).
    n_out = n - 2
    return [[cols[j][i] for j in range(n_out)] for i in range(n)]


def bspline(
    col: str | pl.Expr,
    knots: Sequence[float],
    *,
    degree: int = 3,
    include_bias: bool = False,
    extrapolate: Literal["constant", "linear", "null"] = "constant",
    prefix: str | None = None,
) -> list[pl.Expr]:
    """Clamped B-spline basis functions of a column, as plain Polars expressions.

    Use it to expand a numeric feature into smooth piecewise-polynomial columns
    for any linear or generalized linear model in a bank, such as a spline
    calibrator with ``rheon.spec.sgd`` or a smooth nonlinear effect with
    ``rheon.spec.ewridge``. Each column depends only on its own row's value, so
    it needs no state, streams natively, and gives identical output under any
    chunking.

    Given ``K >= 2`` strictly increasing knots ``k_0 < k_1 < ... < k_{K-1}``
    (the boundary knots ``k_0`` and ``k_{K-1}`` plus any interior knots) and
    degree ``d``, the clamped knot vector repeats both boundary knots with
    multiplicity ``d + 1``:

    .. code-block:: text

        t = [k_0, ..., k_0, k_1, ..., k_{K-2}, k_{K-1}, ..., k_{K-1}]
             |-- d + 1 ---|                    |------ d + 1 -------|

    Cox-de Boor recursion defines ``K + d - 1`` non-negative basis functions
    that sum to 1 everywhere on ``[k_0, k_{K-1}]``, matching
    ``scipy.interpolate.BSpline.design_matrix`` and R's ``splines::bs``. Under
    the default ``include_bias=False``, the first basis function is dropped so
    the remaining ``K + d - 2`` columns do not collinearize with a model's
    intercept.

    .. rubric:: Parameters

    ``col``
        The feature to expand: a numeric column name or a single-column
        ``pl.Expr``.
    ``knots``
        Strictly increasing finite knot positions ``[k_0, ..., k_{K-1}]``
        (boundary knots and any interior knots), with at least two values.
    ``degree``
        Polynomial degree of the spline basis, an integer from 0 to 10.
        Default 3 (cubic).
    ``include_bias``
        Whether to keep all ``K + degree - 1`` basis columns (which sum to 1
        on ``[k_0, k_{K-1}]``). Default ``False`` drops the first basis column
        so the columns can sit beside a model's intercept without
        collinearity.
    ``extrapolate``
        How values outside ``[k_0, k_{K-1}]`` are evaluated:

        - ``"constant"`` (default): clamp ``x`` to ``[k_0, k_{K-1}]`` so each
          basis holds its boundary value;
        - ``"linear"``: continue each basis linearly along its boundary
          tangent ``B(k) + (x - k) B'(k)``;
        - ``"null"``: emit null outside ``[k_0, k_{K-1}]``.
    ``prefix``
        Prefix for output column names, producing ``{prefix}_0``,
        ``{prefix}_1``, and so on. ``None`` (the default) uses ``{name}_bs``,
        where ``name`` is ``col`` or its output name.

    .. rubric:: Returns

    A list of ``pl.Expr``, one per basis column, named ``{prefix}_{i}``
    (0-based). Null inputs produce null outputs and NaN inputs produce NaN
    outputs.

    .. rubric:: Raises

    ``TypeError``
        ``col`` is not a ``str`` or ``pl.Expr``; ``knots`` is not a sequence
        of real numbers; ``degree`` is not an int; ``include_bias`` is not a
        bool; ``extrapolate`` is not a ``str``; or ``prefix`` is neither a
        ``str`` nor ``None``.
    ``ValueError``
        ``knots`` has fewer than two values, holds a non-finite value, or is
        not strictly increasing; ``degree`` is outside 0 to 10; ``degree=0``
        with two knots and ``include_bias=False`` (which leaves no basis
        column); ``extrapolate`` is not one of ``"constant"``, ``"linear"`` or
        ``"null"``; or ``col`` or ``prefix`` is empty.

    .. rubric:: Examples

    .. code-block:: python

        import polars as pl
        from rheon import polars as rp

        rows = pl.DataFrame({"x": [0.0, 0.25, 0.5, 0.75, 1.0]})
        out = rows.select(rp.ops.bspline("x", [0.0, 0.5, 1.0], degree=1, include_bias=True))
        print(out.columns, out.row(2))

    .. code-block:: text

        ['x_bs_0', 'x_bs_1', 'x_bs_2'] (0.0, 1.0, 0.0)

    .. rubric:: See also

    - :func:`natural_spline`, a cubic spline basis constrained to be linear
      outside the boundary knots.
    - :func:`hinge`, piecewise-linear hinge terms at given knots.
    - :func:`interactions`, polynomial interaction products of columns.
    """
    who = "rp.ops.bspline"
    x, out_prefix = _resolve_input(col, prefix, "bs", who)
    knot_vals = _validate_knots(knots, min_count=2, who=who)
    deg = _check_int_range(degree, "degree", 0, _MAX_DEGREE, who)
    bias = _check_bool(include_bias, "include_bias", who)
    if not isinstance(extrapolate, str):
        raise WrongTypeError(
            f"{who}: extrapolate must be a str, but it is {extrapolate!r}. "
            'Set extrapolate to "constant", "linear" or "null".'
        )
    if extrapolate not in _EXTRAPOLATE_MODES:
        raise ValueError(
            f"{who}: extrapolate must be one of {', '.join(_EXTRAPOLATE_MODES)}, "
            f"but it is {extrapolate!r}. "
            'Set extrapolate to "constant", "linear" or "null".'
        )
    n_basis = len(knot_vals) + deg - 1
    start = 0 if bias else 1
    if n_basis - start < 1:
        raise ValueError(
            f"{who}: degree=0 with 2 knots and include_bias=False leaves 0 basis columns. "
            "Pass at least 3 knots, raise degree, or set include_bias=True."
        )

    k_lo, k_hi = knot_vals[0], knot_vals[-1]
    x_clamped = x.clip(lower_bound=k_lo, upper_bound=k_hi)
    interior_exprs = _bspline_interior_exprs(x_clamped, knot_vals, deg)

    vals_lo, slopes_lo, _ = _bspline_scalar_eval(k_lo, knot_vals, deg)
    vals_hi, slopes_hi, _ = _bspline_scalar_eval(k_hi, knot_vals, deg)

    out: list[pl.Expr] = []
    for out_idx, b_idx in enumerate(range(start, n_basis)):
        expr = _wrap_extrapolate(
            x,
            interior_exprs[b_idx],
            k_lo,
            k_hi,
            extrapolate,
            vals_lo[b_idx],
            slopes_lo[b_idx],
            vals_hi[b_idx],
            slopes_hi[b_idx],
        )
        out.append(expr.alias(f"{out_prefix}_{out_idx}"))
    return out


def natural_spline(
    col: str | pl.Expr,
    knots: Sequence[float],
    *,
    include_bias: bool = False,
    prefix: str | None = None,
) -> list[pl.Expr]:
    """QR-projected natural cubic B-spline basis of a column, as Polars expressions.

    Use it when a nonlinear feature must extrapolate linearly past its boundary
    knots without wild cubic tails, such as a probability calibrator or a
    yield-curve factor. Each column depends only on its own row's value, so it
    needs no state, streams natively, and gives identical output under any
    chunking.

    Starting from the clamped cubic B-spline basis ``B(x)`` on ``K >= 2``
    strictly increasing knots ``k_0 < ... < k_{K-1}`` (dropping ``B_0`` first
    when ``include_bias=False``), the boundary second-derivative constraint
    matrix ``C = [B''(k_0); B''(k_{K-1})]`` is factored by Householder QR,
    ``C^T = Q R``, and the orthonormal nullspace ``Z = Q[:, 2:]`` projects the
    basis onto the natural cubic spline space:

    .. code-block:: text

        N(x) = B(x) Z,     N''(k_0) = 0,     N''(k_{K-1}) = 0

    Outside ``[k_0, k_{K-1}]``, each basis function extrapolates linearly along
    its boundary tangent ``N(k) + (x - k) N'(k)``. This matches R's
    ``splines::ns`` and spans the exact natural cubic spline space of dimension
    ``K`` (``include_bias=True``) or ``K - 1`` (``include_bias=False``).

    .. rubric:: Parameters

    ``col``
        The feature to expand: a numeric column name or a single-column
        ``pl.Expr``.
    ``knots``
        Strictly increasing finite knot positions ``[k_0, ..., k_{K-1}]``
        (boundary knots and any interior knots), with at least two values.
    ``include_bias``
        Whether to include the intercept direction (returning ``K`` columns).
        Default ``False`` drops the first B-spline column before the QR
        projection (returning ``K - 1`` columns, each zero at ``k_0``) so the
        basis does not collinearize with a model's intercept.
    ``prefix``
        Prefix for output column names, producing ``{prefix}_0``,
        ``{prefix}_1``, and so on. ``None`` (the default) uses ``{name}_ns``,
        where ``name`` is ``col`` or its output name.

    .. rubric:: Returns

    A list of ``pl.Expr``, one per natural spline basis column, named
    ``{prefix}_{i}`` (0-based). Null inputs produce null outputs and NaN inputs
    produce NaN outputs.

    .. rubric:: Raises

    ``TypeError``
        ``col`` is not a ``str`` or ``pl.Expr``; ``knots`` is not a sequence
        of real numbers; ``include_bias`` is not a bool; or ``prefix`` is
        neither a ``str`` nor ``None``.
    ``ValueError``
        ``knots`` has fewer than two values, holds a non-finite value, or is
        not strictly increasing; or ``col`` or ``prefix`` is empty.

    .. rubric:: Examples

    .. code-block:: python

        import polars as pl
        from rheon import polars as rp

        rows = pl.DataFrame({"x": [-0.5, 0.0, 0.5, 1.0, 1.5]})
        out = rows.select(rp.ops.natural_spline("x", [0.0, 0.5, 1.0]))
        print(out.columns, len(out.columns))

    .. code-block:: text

        ['x_ns_0', 'x_ns_1'] 2

    .. rubric:: See also

    - :func:`bspline`, an unconstrained clamped B-spline basis of any degree.
    - :func:`hinge`, piecewise-linear hinge terms at given knots.
    """
    who = "rp.ops.natural_spline"
    x, out_prefix = _resolve_input(col, prefix, "ns", who)
    knot_vals = _validate_knots(knots, min_count=2, who=who)
    bias = _check_bool(include_bias, "include_bias", who)

    k_lo, k_hi = knot_vals[0], knot_vals[-1]
    deg = 3
    vals_lo, slopes_lo, d2_lo = _bspline_scalar_eval(k_lo, knot_vals, deg)
    vals_hi, slopes_hi, d2_hi = _bspline_scalar_eval(k_hi, knot_vals, deg)

    start = 0 if bias else 1
    c0 = d2_lo[start:]
    c1 = d2_hi[start:]
    z_mat = _nullspace_qr_2rows(c0, c1)  # shape: n_b x (n_b - 2)
    n_b = len(c0)
    n_out = n_b - 2

    x_clamped = x.clip(lower_bound=k_lo, upper_bound=k_hi)
    raw_interior = _bspline_interior_exprs(x_clamped, knot_vals, deg)[start:]

    v_lo_sub = vals_lo[start:]
    s_lo_sub = slopes_lo[start:]
    v_hi_sub = vals_hi[start:]
    s_hi_sub = slopes_hi[start:]

    out: list[pl.Expr] = []
    for j in range(n_out):
        terms: list[pl.Expr] = []
        for i in range(n_b):
            w = z_mat[i][j]
            if w == 0.0:
                continue
            if w == 1.0:
                terms.append(raw_interior[i])
            elif w == -1.0:
                terms.append(-raw_interior[i])
            else:
                terms.append(raw_interior[i] * w)
        if not terms:
            interior = pl.lit(0.0, dtype=pl.Float64)
        else:
            interior = terms[0]
            for term in terms[1:]:
                interior = interior + term

        nv_lo = math.fsum(v_lo_sub[i] * z_mat[i][j] for i in range(n_b))
        ns_lo = math.fsum(s_lo_sub[i] * z_mat[i][j] for i in range(n_b))
        nv_hi = math.fsum(v_hi_sub[i] * z_mat[i][j] for i in range(n_b))
        ns_hi = math.fsum(s_hi_sub[i] * z_mat[i][j] for i in range(n_b))

        expr = _wrap_extrapolate(
            x,
            interior,
            k_lo,
            k_hi,
            "linear",
            nv_lo,
            ns_lo,
            nv_hi,
            ns_hi,
        )
        out.append(expr.alias(f"{out_prefix}_{j}"))
    return out


def hinge(
    col: str | pl.Expr,
    knots: Sequence[float],
    *,
    prefix: str | None = None,
) -> list[pl.Expr]:
    """Piecewise-linear hinge terms ``max(0, x - k)`` at given knots.

    Use it to let a linear model change slope at known thresholds without
    duplicating the linear column ``col`` itself. Only the hinge terms
    ``(x - k)_+`` are emitted; keep ``col`` in the model's features if you also
    want the baseline slope below the first knot. Each term depends only on its
    own row's value, so it needs no state and streams natively.

    .. code-block:: text

        h_i(x) = max(0, x - k_i),   i = 0 .. len(knots) - 1

    .. rubric:: Parameters

    ``col``
        The feature to expand: a numeric column name or a single-column
        ``pl.Expr``.
    ``knots``
        Strictly increasing finite knot positions, with at least one value.
    ``prefix``
        Prefix for output column names, producing ``{prefix}_0``,
        ``{prefix}_1``, and so on. ``None`` (the default) uses ``{name}_hinge``,
        where ``name`` is ``col`` or its output name.

    .. rubric:: Returns

    A list of ``pl.Expr``, one per knot, named ``{prefix}_{i}`` (0-based). Null
    inputs produce null outputs and NaN inputs produce NaN outputs.

    .. rubric:: Raises

    ``TypeError``
        ``col`` is not a ``str`` or ``pl.Expr``; ``knots`` is not a sequence
        of real numbers; or ``prefix`` is neither a ``str`` nor ``None``.
    ``ValueError``
        ``knots`` is empty, holds a non-finite value, or is not strictly
        increasing; or ``col`` or ``prefix`` is empty.

    .. rubric:: Examples

    .. code-block:: python

        import polars as pl
        from rheon import polars as rp

        rows = pl.DataFrame({"x": [-1.0, 0.0, 0.5, 2.0]})
        out = rows.select(rp.ops.hinge("x", [0.0, 1.0]))
        print(out.columns, out["x_hinge_0"].to_list(), out["x_hinge_1"].to_list())

    .. code-block:: text

        ['x_hinge_0', 'x_hinge_1'] [0.0, 0.0, 0.5, 2.0] [0.0, 0.0, 0.0, 1.0]

    .. rubric:: See also

    - :func:`bspline`, smooth piecewise-polynomial B-spline basis terms.
    - :func:`natural_spline`, natural cubic spline basis with linear tails.
    """
    who = "rp.ops.hinge"
    x, out_prefix = _resolve_input(col, prefix, "hinge", who)
    knot_vals = _validate_knots(knots, min_count=1, who=who)
    return [
        (x - k).clip(lower_bound=0.0).alias(f"{out_prefix}_{i}") for i, k in enumerate(knot_vals)
    ]


def interactions(
    cols: Sequence[str | pl.Expr],
    *,
    degree: int = 2,
    include_self: bool = False,
    sep: str = "_x_",
) -> list[pl.Expr]:
    """Polynomial interaction products of columns up to a given degree.

    Use it to add pairwise or higher-order product features to a frame before
    feeding a bank. Every interaction of order ``2`` through ``degree`` is
    emitted in lexicographic order, matching the nonlinear columns of
    scikit-learn's ``PolynomialFeatures(degree=degree,
    interaction_only=not include_self, include_bias=False)``. Each product
    depends only on its own row, so it needs no state and streams natively.

    .. rubric:: Parameters

    ``cols``
        The columns to combine: a sequence of column names or single-column
        ``pl.Expr`` objects, with unique output names. At least two columns are
        required when ``include_self=False`` (and at least ``degree`` columns
        in total), or at least one column when ``include_self=True``.
    ``degree``
        Highest interaction order to include, an integer from 2 to 6. Default
        2 emits pairwise products; ``degree=3`` emits all 2-way and 3-way
        products.
    ``include_self``
        Whether a column may multiply itself (such as ``a_x_a`` for ``a^2``).
        Default ``False`` emits only cross-column products.
    ``sep``
        Separator joined between the factor names in each output column name.
        Default ``"_x_"`` (for example ``"a_x_b"``).

    .. rubric:: Returns

    A list of ``pl.Expr``, one per product term, named by joining the factor
    names with ``sep``. Null or NaN factors propagate to their products.

    .. rubric:: Raises

    ``TypeError``
        ``cols`` is a string or not a sequence, or holds an item that is
        neither a ``str`` nor a ``pl.Expr``; ``degree`` is not an int;
        ``include_self`` is not a bool; or ``sep`` is not a ``str``.
    ``ValueError``
        ``cols`` has too few items for ``degree`` and ``include_self``, has an
        empty or multi-output entry, or has duplicate factor names; ``degree``
        is outside 2 to 6; ``sep`` is empty; or two generated output names
        collide under ``sep``.

    .. rubric:: Examples

    .. code-block:: python

        import polars as pl
        from rheon import polars as rp

        rows = pl.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0], "c": [5.0, 6.0]})
        out = rows.select(rp.ops.interactions(["a", "b", "c"]))
        print(out.columns, out.row(1))

    .. code-block:: text

        ['a_x_b', 'a_x_c', 'b_x_c'] (8.0, 12.0, 24.0)

    .. rubric:: See also

    - :func:`bspline`, :func:`natural_spline`, :func:`hinge`, single-column
      basis expansions that combine with ``interactions``.
    """
    who = "rp.ops.interactions"
    if isinstance(cols, (str, bytes, pl.Expr)) or not isinstance(cols, Sequence):
        raise WrongTypeError(
            f"{who}: cols must be a sequence of column names or expressions, but it is {cols!r}. "
            "Pass a list or tuple of column names."
        )
    deg = _check_int_range(degree, "degree", 2, _MAX_INTERACTION_DEGREE, who)
    self_prod = _check_bool(include_self, "include_self", who)
    if not isinstance(sep, str):
        raise WrongTypeError(
            f"{who}: sep must be a str, but it is {sep!r}. Pass a non-empty string separator."
        )
    if not sep:
        raise ValueError(
            f"{who}: sep must be a non-empty string, but it is ''. "
            'Pass a non-empty separator such as "_x_".'
        )

    items: list[tuple[str, pl.Expr]] = []
    seen_inputs: set[str] = set()
    for i, item in enumerate(cols):
        if isinstance(item, str):
            if not item:
                raise ValueError(
                    f"{who}: cols[{i}] must be a non-empty column name, but it is ''. "
                    "Pass non-empty column names."
                )
            name = item
            expr = pl.col(item).cast(pl.Float64)
        elif isinstance(item, pl.Expr):
            if item.meta.has_multiple_outputs():
                raise ValueError(
                    f"{who}: cols[{i}] must expand to a single column, but it has multiple "
                    "outputs. Pass one column per entry."
                )
            try:
                name = item.meta.output_name()
            except Exception as err:  # noqa: BLE001
                raise ValueError(
                    f"{who}: cols[{i}] has no single output name ({err}). "
                    "Name the expression with .alias(...)."
                ) from None
            expr = item.cast(pl.Float64)
        else:
            raise WrongTypeError(
                f"{who}: cols[{i}] must be a column name or a pl.Expr, but it is {item!r}. "
                "Pass column names or single-column Polars expressions."
            )
        if name in seen_inputs:
            raise ValueError(
                f"{who}: duplicate column {name!r} in cols. Remove duplicate entries from cols."
            )
        seen_inputs.add(name)
        items.append((name, expr))

    min_cols = 1 if self_prod else deg
    if len(items) < min_cols:
        if self_prod:
            raise ValueError(
                f"{who}: cols must hold at least 1 column, but it is empty. "
                "Pass at least one column."
            )
        raise ValueError(
            f"{who}: cols holds {len(items)} column(s), fewer than degree={deg} with "
            f"include_self=False. Pass at least {deg} columns, lower degree, or set "
            "include_self=True."
        )

    combiner = itertools.combinations_with_replacement if self_prod else itertools.combinations
    out: list[pl.Expr] = []
    seen_outputs: set[str] = set()
    for order in range(2, deg + 1):
        for combo in combiner(items, order):
            out_name = sep.join(name for name, _ in combo)
            if out_name in seen_outputs:
                raise ValueError(
                    f"{who}: output column name {out_name!r} collides under sep={sep!r}. "
                    "Choose a separator that does not appear inside the column names."
                )
            seen_outputs.add(out_name)
            prod = combo[0][1]
            for _, factor in combo[1:]:
                prod = prod * factor
            out.append(prod.alias(out_name))
    return out
