"""Forecast comparison tests: loss builders for :func:`diebold_mariano`,
Giacomini-White conditional predictive ability (:func:`giacomini_white`), and
anytime-valid bounded mean sequential tests (:func:`mean_seqtest`).

Split out of ``eval.py`` so every source file stays well below the repository
size cap. Public functions are re-exported from :mod:`rheon.polars.eval`.
"""

from __future__ import annotations

import math
import numbers
import operator
from collections import deque
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, overload

import numpy as np
import polars as pl

from rheon import _rheon as _native
from rheon._errors import RefusedValueError, WrongTypeError
from rheon._renamed import renamed_keywords

LossSpec = Literal["squared", "absolute", "pinball", "linex"] | Callable[[pl.Expr], pl.Expr]

#: Per-group byte cap for Giacomini-White HAC state (256 KiB).
_GW_CAP_BYTES = 256 * 1024

_VALID_LOSSES = ("squared", "absolute", "pinball", "linex")
_VALID_REFERENCES = ("normal", "t")
_VALID_ALTERNATIVES = ("two-sided", "greater", "less")
_VALID_MEAN_METHODS = ("cm_eb", "betting")
_GROUP = {"by": "group"}


def _check_bool(who: str, param: str, val: object) -> bool:
    if not isinstance(val, bool):
        msg = f"{who}: {param} must be a bool, but it is {val!r}. Pass True or False."
        raise WrongTypeError(msg)
    return val


def _check_positive_finite(who: str, param: str, val: object) -> float:
    if isinstance(val, bool) or not isinstance(val, numbers.Real):
        msg = (
            f"{who}: {param} must be a finite number above 0, but it is {val!r}. "
            f"Pass a positive finite number for {param}."
        )
        raise WrongTypeError(msg)
    f = float(val)
    if not (math.isfinite(f) and f > 0.0):
        msg = (
            f"{who}: {param} must be finite and above 0, but it is {val!r}. "
            f"Pass a positive finite number for {param}."
        )
        raise ValueError(msg)
    return f


def _check_alpha(who: str, alpha: object) -> float:
    if isinstance(alpha, bool) or not isinstance(alpha, numbers.Real):
        msg = (
            f"{who}: alpha must be a number in (0, 1), but it is {alpha!r}. "
            "Pass a number between 0 and 1, such as 0.05."
        )
        raise WrongTypeError(msg)
    f = float(alpha)
    if not (math.isfinite(f) and 0.0 < f < 1.0):
        msg = (
            f"{who}: alpha must be strictly between 0 and 1, but it is {alpha!r}. "
            "Pass a number between 0 and 1, such as 0.05."
        )
        raise ValueError(msg)
    return f


def _check_nonneg_int(who: str, param: str, val: object) -> int:
    if isinstance(val, bool) or not isinstance(val, numbers.Integral):
        msg = (
            f"{who}: {param} must be a whole number of at least 0, but it is {val!r}. "
            f"Pass a non-negative integer for {param}."
        )
        raise WrongTypeError(msg)
    n = operator.index(val)
    if n < 0:
        msg = (
            f"{who}: {param} must be at least 0, but it is {n}. "
            f"Pass a non-negative integer for {param}."
        )
        raise ValueError(msg)
    return n


def build_loss_diffs(
    who: str,
    pairs: dict[str, tuple[pl.Expr, pl.Expr]],
    loss: LossSpec,
    quantile: float | None,
    linex_a: float | None,
) -> dict[str, pl.Expr]:
    """Validate ``loss`` / ``quantile`` / ``linex_a`` and build ``L(rb) - L(ra)``
    per target.

    When ``loss == "squared"``, returns the exact expression ``rb * rb - ra * ra``
    so default :func:`diebold_mariano` is bit-identical to prior releases.
    """
    if isinstance(loss, str):
        if loss not in _VALID_LOSSES:
            msg = (
                f"{who}: loss must be one of {_VALID_LOSSES} or a callable, but it is {loss!r}. "
                "Choose one of those names, or pass a callable mapping a residual expression to "
                "a loss expression."
            )
            raise ValueError(msg)
    elif not callable(loss):
        msg = (
            f"{who}: loss must be one of {_VALID_LOSSES} or a callable, but it is of type "
            f"{type(loss).__name__}. Choose one of those names, or pass a callable."
        )
        raise WrongTypeError(msg)

    if quantile is not None and loss != "pinball":
        msg = (
            f"{who}: quantile applies only when loss='pinball', but loss is {loss!r}. "
            "Leave out quantile, or set loss='pinball'."
        )
        raise ValueError(msg)
    if linex_a is not None and loss != "linex":
        msg = (
            f"{who}: linex_a applies only when loss='linex', but loss is {loss!r}. "
            "Leave out linex_a, or set loss='linex'."
        )
        raise ValueError(msg)

    q_val: float | None = None
    if loss == "pinball":
        if quantile is None:
            msg = (
                f"{who}: loss='pinball' requires quantile in (0, 1), but quantile is None. "
                "Pass quantile, such as quantile=0.5 or quantile=0.9."
            )
            raise ValueError(msg)
        if isinstance(quantile, bool) or not isinstance(quantile, numbers.Real):
            msg = (
                f"{who}: quantile must be a number in (0, 1), but it is {quantile!r}. "
                "Pass a number between 0 and 1, such as 0.5."
            )
            raise WrongTypeError(msg)
        q_val = float(quantile)
        if not (math.isfinite(q_val) and 0.0 < q_val < 1.0):
            msg = (
                f"{who}: quantile must be strictly between 0 and 1, but it is {quantile!r}. "
                "Pass a number between 0 and 1, such as 0.5."
            )
            raise ValueError(msg)

    a_val: float | None = None
    if loss == "linex":
        if linex_a is None:
            msg = (
                f"{who}: loss='linex' requires a non-zero finite linex_a, but linex_a is None. "
                "Pass linex_a, such as linex_a=0.5."
            )
            raise ValueError(msg)
        if isinstance(linex_a, bool) or not isinstance(linex_a, numbers.Real):
            msg = (
                f"{who}: linex_a must be a non-zero finite number, but it is {linex_a!r}. "
                "Pass a non-zero finite number, such as 0.5."
            )
            raise WrongTypeError(msg)
        a_val = float(linex_a)
        if not (math.isfinite(a_val) and a_val != 0.0):
            msg = (
                f"{who}: linex_a must be finite and non-zero, but it is {linex_a!r}. "
                "Pass a non-zero finite number, such as 0.5."
            )
            raise ValueError(msg)

    if loss == "squared":
        return {t: rb * rb - ra * ra for t, (ra, rb) in pairs.items()}

    def _apply(e: pl.Expr) -> pl.Expr:
        if loss == "absolute":
            return e.abs()
        if loss == "pinball":
            assert q_val is not None
            return pl.when(e >= 0.0).then(q_val * e).otherwise((q_val - 1.0) * e)
        if loss == "linex":
            assert a_val is not None
            ae = a_val * e
            val = ae.exp() - ae - 1.0
            return pl.when(val.is_finite()).then(val)
        assert callable(loss)
        out = loss(e)
        if not isinstance(out, pl.Expr):
            msg = (
                f"{who}: callable loss must return a Polars Expr, but it returned "
                f"{type(out).__name__}. Return a Polars expression from the loss callable."
            )
            raise WrongTypeError(msg)
        return pl.when(out.is_finite()).then(out)

    out_diffs: dict[str, pl.Expr] = {}
    for t, (ra, rb) in pairs.items():
        d = _apply(rb) - _apply(ra)
        out_diffs[t] = pl.when(d.is_finite()).then(d)
    return out_diffs


def check_dm_options(
    who: str,
    *,
    hln: object,
    reference: object,
    alternative: object,
    half_life: float | None,
) -> tuple[bool, Literal["normal", "t"], Literal["two-sided", "greater", "less"]]:
    """Validate the extended Diebold-Mariano options."""
    hln_b = _check_bool(who, "hln", hln)
    if not isinstance(reference, str) or reference not in _VALID_REFERENCES:
        msg = (
            f"{who}: reference must be one of {_VALID_REFERENCES}, but it is {reference!r}. "
            "Use 'normal' or 't'."
        )
        raise ValueError(msg)
    if not isinstance(alternative, str) or alternative not in _VALID_ALTERNATIVES:
        msg = (
            f"{who}: alternative must be one of {_VALID_ALTERNATIVES}, but it is {alternative!r}. "
            "Use 'two-sided', 'greater', or 'less'."
        )
        raise ValueError(msg)
    if half_life is not None:
        if reference != "normal":
            msg = (
                f"{who}: reference={reference!r} cannot be combined with half_life, because the "
                "per-row form emits only the statistic without a p-value. Leave reference='normal' "
                "beside half_life."
            )
            raise ValueError(msg)
        if alternative != "two-sided":
            msg = (
                f"{who}: alternative={alternative!r} cannot be combined with half_life, because "
                "the per-row form emits only the statistic without a p-value. Leave "
                "alternative='two-sided' beside half_life."
            )
            raise ValueError(msg)
    ref_lit: Literal["normal", "t"] = "normal" if reference == "normal" else "t"
    alt_lit: Literal["two-sided", "greater", "less"] = (
        "two-sided"
        if alternative == "two-sided"
        else ("greater" if alternative == "greater" else "less")
    )
    return (hln_b, ref_lit, alt_lit)


def apply_hln_and_pvalue(
    out: pl.DataFrame,
    *,
    lags: int,
    hln: bool,
    reference: Literal["normal", "t"],
    alternative: Literal["two-sided", "greater", "less"],
) -> pl.DataFrame:
    """Apply optional Harvey-Leybourne-Newbold finite-sample scaling and
    compute ``p_value`` using the requested reference distribution and tail."""
    if not hln and reference == "normal" and alternative == "two-sided":
        # Exact 2.0.0 path (0 ulp change).
        p = [
            None if z is None or math.isnan(z) else math.erfc(abs(z) / math.sqrt(2.0))
            for z in out["dm"].to_list()
        ]
        return out.with_columns(pl.Series("p_value", p, dtype=pl.Float64))

    ns = out["n"].to_list()
    dms = out["dm"].to_list()
    h = float(lags + 1)
    adj_dms: list[float | None] = []
    pvals: list[float | None] = []
    inv_sqrt2 = math.sqrt(0.5)

    for n_val, z_val in zip(ns, dms, strict=True):
        if z_val is None or math.isnan(z_val) or n_val is None or n_val <= 0:
            adj_dms.append(None)
            pvals.append(None)
            continue
        z = float(z_val)
        n = float(n_val)
        if hln:
            rad = (n + 1.0 - 2.0 * h + h * (h - 1.0) / n) / n
            if not (math.isfinite(rad) and rad > 0.0):
                adj_dms.append(None)
                pvals.append(None)
                continue
            z = z * math.sqrt(rad)
        adj_dms.append(z)
        if reference == "t":
            if n <= 1.0:
                pvals.append(None)
                continue
            df_t = n - 1.0
            if alternative == "greater":
                pval = _native.special_student_t_sf(z, df_t)
            elif alternative == "less":
                pval = _native.special_student_t_sf(-z, df_t)
            else:
                pval = min(1.0, 2.0 * _native.special_student_t_sf(abs(z), df_t))
        else:
            if alternative == "greater":
                pval = 0.5 * math.erfc(z * inv_sqrt2)
            elif alternative == "less":
                pval = 0.5 * math.erfc(-z * inv_sqrt2)
            else:
                pval = math.erfc(abs(z) * inv_sqrt2)
        pvals.append(None if math.isnan(pval) else pval)

    return out.with_columns(
        pl.Series("dm", adj_dms, dtype=pl.Float64),
        pl.Series("p_value", pvals, dtype=pl.Float64),
    )


def hln_ew_factor_expr(w1: pl.Expr, w2: pl.Expr, lags: int) -> pl.Expr:
    """Polars expression for the HLN multiplier using Kish effective sample size
    ``n = w1² / w2`` and horizon ``h = lags + 1``, null when the radicand is
    non-positive."""
    n = (w1 * w1) / w2
    h = float(lags + 1)
    rad = (n + 1.0 - 2.0 * h + h * (h - 1.0) / n) / n
    return pl.when(rad > 0.0).then(rad.sqrt())


# ---------------------------------------------------------------------------
# Giacomini-White (2006) conditional predictive ability test
# ---------------------------------------------------------------------------


def _gw_state_bytes(q: int, lags: int, instrument_lags: int) -> int:
    return 8 * ((lags + 1) * q * q + q * lags + q + instrument_lags + 2)


@dataclass
class _GwState:
    q: int
    lags: int
    instrument_lags: int
    count: int = 0
    w: float = 0.0
    q_sum: float = 0.0
    zs: np.ndarray = field(init=False)
    gam: np.ndarray = field(init=False)
    ring_d: deque[float] = field(init=False)
    zring: deque[np.ndarray] = field(init=False)

    def __post_init__(self) -> None:
        self.zs = np.zeros(self.q, dtype=np.float64)
        self.gam = np.zeros((self.lags + 1, self.q, self.q), dtype=np.float64)
        self.ring_d = deque(maxlen=self.instrument_lags)
        self.zring = deque(maxlen=self.lags)

    def step(self, d: float, exog: Sequence[float], lam: float) -> None:
        if lam != 1.0:
            self.w *= lam
            self.q_sum *= lam * lam
            self.zs *= lam
            self.gam *= lam
        if len(self.ring_d) < self.instrument_lags:
            if self.instrument_lags > 0:
                self.ring_d.append(d)
            return
        h = np.empty(self.q, dtype=np.float64)
        h[0] = 1.0
        idx = 1
        if self.instrument_lags > 0:
            for prev in reversed(self.ring_d):
                h[idx] = prev
                idx += 1
        for v in exog:
            h[idx] = v
            idx += 1
        z = h * d
        self.count += 1
        self.w += 1.0
        self.q_sum += 1.0
        self.zs += z
        self.gam[0] += np.outer(z, z)
        if self.lags > 0:
            for lag, past in enumerate(reversed(self.zring), start=1):
                self.gam[lag] += np.outer(z, past)
            self.zring.append(z)
        if self.instrument_lags > 0:
            self.ring_d.append(d)

    def stat(self, min_samples: int, few_rows: int) -> float | None:
        if (
            self.count < min_samples
            or self.count <= few_rows
            or self.count <= self.q
            or self.w <= 0.0
            or self.q_sum <= 0.0
        ):
            return None
        n_eff = (self.w * self.w) / self.q_sum
        zbar = self.zs / self.w
        om = self.gam[0] / self.w
        for lag in range(1, self.lags + 1):
            wt = 1.0 - lag / (self.lags + 1.0)
            gl = self.gam[lag] / self.w
            om = om + wt * (gl + gl.T)
        om = 0.5 * (om + om.T)
        try:
            chol = np.linalg.cholesky(om)
        except np.linalg.LinAlgError:
            return None
        y = np.linalg.solve(chol, zbar)
        val = float(n_eff * np.dot(y, y))
        if not (math.isfinite(val) and val >= 0.0):
            return None
        return val


def _validate_instruments(
    who: str, probe: pl.DataFrame, instruments: Sequence[str] | None
) -> list[str]:
    if instruments is None:
        return []
    if isinstance(instruments, str):
        msg = (
            f"{who}: instruments must be a sequence of column names, not the string "
            f"{instruments!r}. Give the names in a list, such as [{instruments!r}]."
        )
        raise WrongTypeError(msg)
    cols = list(instruments)
    seen: set[str] = set()
    for c in cols:
        if not isinstance(c, str):
            msg = (
                f"{who}: instrument column names must be strings, but got {c!r}. "
                "Give each instrument column name as a string."
            )
            raise WrongTypeError(msg)
        if c in seen:
            msg = f"{who}: instruments lists {c!r} twice. List each instrument column once."
            raise ValueError(msg)
        if c not in probe.columns:
            from rheon.polars._errors import ColumnNotFoundError

            msg = (
                f"{who}: instrument column {c!r} is not in the frame. "
                "Check the column name or add the column before calling."
            )
            raise ColumnNotFoundError(msg)
        if not probe.schema[c].is_numeric():
            msg = (
                f"{who}: instrument column {c!r} must be numeric, but it has dtype "
                f"{probe.schema[c]}. Cast the column to float or drop it from instruments."
            )
            raise WrongTypeError(msg)
        seen.add(c)
    return cols


def _gw_chunk_rows(
    chunk: pl.DataFrame,
    values: dict[str, pl.Expr],
    keys: list[str],
    inst_cols: list[str],
    lags: int,
    instrument_lags: int,
    lam: float,
    min_samples: int,
    few_rows: int,
    name: str,
    states: dict[str, dict[tuple[Any, ...], _GwState]],
) -> pl.DataFrame:
    q = 1 + instrument_lags + len(inst_cols)
    taken = {*chunk.columns, name}
    from rheon.polars import eval as _eval

    d_cols = {t: _eval._free(taken, f"gw_d_{i}") for i, t in enumerate(values)}
    work = chunk.select(
        *keys,
        *(pl.col(c).cast(pl.Float64) for c in inst_cols),
        *(expr.alias(d_cols[t]) for t, expr in values.items()),
    )
    n_rows = work.height
    key_cols = [work[k].to_list() for k in keys]
    inst_lists = [work[c].to_list() for c in inst_cols]
    out_series: list[pl.Series] = []
    for t in values:
        d_list = work[d_cols[t]].to_list()
        by_key = states[t]
        vals: list[float | None] = [None] * n_rows
        for i in range(n_rows):
            d_val = d_list[i]
            if d_val is None or not math.isfinite(d_val):
                continue
            exog: list[float] = []
            ok = True
            for lst in inst_lists:
                v = lst[i]
                if v is None or not math.isfinite(v):
                    ok = False
                    break
                exog.append(float(v))
            if not ok:
                continue
            key = tuple(kc[i] for kc in key_cols) if keys else ()
            st = by_key.get(key)
            if st is None:
                st = _GwState(q=q, lags=lags, instrument_lags=instrument_lags)
                by_key[key] = st
            st.step(float(d_val), exog, lam)
            vals[i] = st.stat(min_samples, few_rows)
        out_series.append(pl.Series(f"gw_{t}", vals, dtype=pl.Float64))
    struct_col = pl.DataFrame(out_series).select(pl.struct(pl.all()).alias(name)).to_series()
    return chunk.with_columns(struct_col)


@overload
def giacomini_white(
    df: pl.DataFrame,
    *,
    a: str,
    b: str,
    targets: Sequence[str] | None = None,
    a_suffix: str = "",
    b_suffix: str = "",
    loss: LossSpec = "squared",
    quantile: float | None = None,
    linex_a: float | None = None,
    lags: int = 0,
    instrument_lags: int = 1,
    instruments: Sequence[str] | None = None,
    group: str | Iterable[str] | None = None,
    half_life: float | None = None,
    min_samples: int = 30,
    name: str = "giacomini_white",
) -> pl.DataFrame: ...


@overload
def giacomini_white(
    df: pl.LazyFrame,
    *,
    a: str,
    b: str,
    targets: Sequence[str] | None = None,
    a_suffix: str = "",
    b_suffix: str = "",
    loss: LossSpec = "squared",
    quantile: float | None = None,
    linex_a: float | None = None,
    lags: int = 0,
    instrument_lags: int = 1,
    instruments: Sequence[str] | None = None,
    group: str | Iterable[str] | None = None,
    half_life: None = None,
    min_samples: int = 30,
    name: str = "giacomini_white",
) -> pl.DataFrame: ...


@overload
def giacomini_white(
    df: pl.LazyFrame,
    *,
    a: str,
    b: str,
    targets: Sequence[str] | None = None,
    a_suffix: str = "",
    b_suffix: str = "",
    loss: LossSpec = "squared",
    quantile: float | None = None,
    linex_a: float | None = None,
    lags: int = 0,
    instrument_lags: int = 1,
    instruments: Sequence[str] | None = None,
    group: str | Iterable[str] | None = None,
    half_life: float,
    min_samples: int = 30,
    name: str = "giacomini_white",
) -> pl.LazyFrame: ...


@renamed_keywords("rp.eval.giacomini_white", _GROUP)
def giacomini_white(
    df: pl.DataFrame | pl.LazyFrame,
    *,
    a: str,
    b: str,
    targets: Sequence[str] | None = None,
    a_suffix: str = "",
    b_suffix: str = "",
    loss: LossSpec = "squared",
    quantile: float | None = None,
    linex_a: float | None = None,
    lags: int = 0,
    instrument_lags: int = 1,
    instruments: Sequence[str] | None = None,
    group: str | Iterable[str] | None = None,
    half_life: float | None = None,
    min_samples: int = 30,
    name: str = "giacomini_white",
) -> pl.DataFrame | pl.LazyFrame:
    """Test conditional equal predictive ability between two specs.

    Applies Giacomini and White's (2006) Wald test. Unlike
    :func:`diebold_mariano`, which tests whether the unconditional mean
    loss differential is zero, this test regresses the loss differential
    ``d_t = L(resid_b) - L(resid_a)`` on a vector of instruments ``h_t``
    (a constant, ``instrument_lags`` lagged differentials, and any extra
    ``instruments`` columns) and computes the HAC Wald statistic
    ``n Z̄' Ω⁻¹ Z̄ ~ χ²(q)`` where ``Z_t = h_t d_t`` and ``q = 1 +
    instrument_lags + len(instruments)``.

    .. rubric:: Parameters

    ``df``
        A ``DataFrame`` or ``LazyFrame`` holding both specs' output columns.
    ``a``, ``b``
        The names of the two specs' output columns.
    ``targets``, ``a_suffix``, ``b_suffix``
        As in :func:`diebold_mariano`.
    ``loss``, ``quantile``, ``linex_a``
        Loss function applied to each residual: ``"squared"`` (default),
        ``"absolute"``, ``"pinball"`` (requires ``quantile`` in ``(0, 1)``),
        ``"linex"`` (requires non-zero ``linex_a``), or a callable mapping a
        residual Polars expression to a loss expression.
    ``lags``
        Newey-West Bartlett lag truncation order ``L >= 0`` (0 by default).
    ``instrument_lags``
        Number of lagged loss differentials included in the instrument vector
        beside the constant (1 by default; 0 gives the unconditional test).
    ``instruments``
        Optional sequence of numeric column names in ``df`` appended to the
        instrument vector ``h_t``.
    ``group``
        Optional group column or list of columns to test each group separately.
    ``half_life``
        ``None`` (default) returns one summary row per ``(group, target)``.
        A finite positive number of rows adds one output column ``name`` with
        the exponentially weighted per-row statistic ``gw_<target>``.
    ``min_samples``
        Minimum complete observations ``Z_t`` required before reporting a
        statistic (30 by default).
    ``name``
        Output column name when ``half_life`` is set (``"giacomini_white"``
        by default).

    .. rubric:: Returns

    Without ``half_life``, a ``DataFrame`` with columns ``[*group, "target",
    "n", "df", "gw", "p_value"]``. With ``half_life``, ``df`` (or a streaming
    ``LazyFrame``) with an output column ``name`` holding ``gw_<target>``.

    .. rubric:: Examples

    This code reads ``df`` from `Example data
    <https://github.com/rheon-dev/rheon/blob/main/docs/EXAMPLE-DATA.md>`_:

    .. code-block:: python

        a = rp.spec.ewridge("ridge", targets=["y"], features=["x0", "x1"], half_life=100.0)
        b = rp.spec.rls("rls", targets=["y"], features=["x0", "x1"], half_life=100.0)
        out = rp.ModelBank([a, b]).fit_predict(df)
        res = rp.eval.giacomini_white(out, a="ridge", b="rls", instrument_lags=1, lags=2)
    """
    from rheon.polars import eval as _eval

    who = "rp.eval.giacomini_white"
    _eval._frame(who, df, lazy=True)
    _eval._check_min_samples(who, min_samples)
    probe = _eval._probe(df)
    _eval._check_columns(probe, who, group=group)
    keys = _eval._group_keys(group)
    pairs = _eval._paired_residuals(probe, a, b, targets, a_suffix, b_suffix, who)
    values = build_loss_diffs(who, pairs, loss, quantile, linex_a)
    lags_i = _eval._lags(who, lags)
    inst_lags_i = _check_nonneg_int(who, "instrument_lags", instrument_lags)
    inst_cols = _validate_instruments(who, probe, instruments)
    q = 1 + inst_lags_i + len(inst_cols)
    state_bytes = _gw_state_bytes(q, lags_i, inst_lags_i)
    if state_bytes > _GW_CAP_BYTES:
        msg = (
            f"{who}: state per group at q={q} instruments and lags={lags_i} would hold "
            f"{state_bytes} bytes, past the {_GW_CAP_BYTES} byte cap. Lower lags, "
            "instrument_lags, or the number of instruments."
        )
        raise ValueError(msg)
    few = _eval._few_rows(lags_i)

    if half_life is not None:
        _eval._check_half_life(who, half_life)
        _eval._check_name(who, probe, name)
        lam = 0.5 ** (1.0 / half_life)
        if isinstance(df, pl.LazyFrame):
            schema = _gw_chunk_rows(
                probe,
                values,
                keys,
                inst_cols,
                lags_i,
                inst_lags_i,
                lam,
                min_samples,
                few,
                name,
                {t: {} for t in values},
            ).schema
            needed = (
                set(keys)
                | set(inst_cols)
                | set().union(*(e.meta.root_names() for e in values.values()))
            )

            def _step() -> Callable[[pl.DataFrame], pl.DataFrame]:
                st: dict[str, dict[tuple[Any, ...], _GwState]] = {t: {} for t in values}

                def _one(chunk: pl.DataFrame) -> pl.DataFrame:
                    return _gw_chunk_rows(
                        chunk,
                        values,
                        keys,
                        inst_cols,
                        lags_i,
                        inst_lags_i,
                        lam,
                        min_samples,
                        few,
                        name,
                        st,
                    )

                return _one

            return _eval._row_query(df, needed, schema, _step)
        states: dict[str, dict[tuple[Any, ...], _GwState]] = {t: {} for t in values}
        return _gw_chunk_rows(
            df, values, keys, inst_cols, lags_i, inst_lags_i, lam, min_samples, few, name, states
        )

    # Summary mode (DataFrame or streaming LazyFrame).
    needed = (
        set(keys) | set(inst_cols) | set().union(*(e.meta.root_names() for e in values.values()))
    )
    states_sum: dict[str, dict[tuple[Any, ...], _GwState]] = {t: {} for t in values}
    key_order: dict[str, list[tuple[Any, ...]]] = {t: [] for t in values}
    taken = set(probe.columns)
    d_cols = {t: _eval._free(taken, f"gw_s_{i}") for i, t in enumerate(values)}

    def _consume(chunk: pl.DataFrame) -> None:
        if chunk.is_empty():
            return
        work = chunk.select(
            *keys,
            *(pl.col(c).cast(pl.Float64) for c in inst_cols),
            *(expr.alias(d_cols[t]) for t, expr in values.items()),
        )
        n_rows = work.height
        key_cols = [work[k].to_list() for k in keys]
        inst_lists = [work[c].to_list() for c in inst_cols]
        for t in values:
            d_list = work[d_cols[t]].to_list()
            by_key = states_sum[t]
            order = key_order[t]
            for i in range(n_rows):
                d_val = d_list[i]
                if d_val is None or not math.isfinite(d_val):
                    continue
                exog: list[float] = []
                ok = True
                for lst in inst_lists:
                    v = lst[i]
                    if v is None or not math.isfinite(v):
                        ok = False
                        break
                    exog.append(float(v))
                if not ok:
                    continue
                key = tuple(kc[i] for kc in key_cols) if keys else ()
                st = by_key.get(key)
                if st is None:
                    st = _GwState(q=q, lags=lags_i, instrument_lags=inst_lags_i)
                    by_key[key] = st
                    order.append(key)
                st.step(float(d_val), exog, 1.0)

    if isinstance(df, pl.LazyFrame):
        schema_all = df.collect_schema()
        plan = df.select([c for c in schema_all if c in needed])
        for batch in plan.collect_batches(chunk_size=_eval._chunk_rows(), maintain_order=True):
            _consume(batch)
    else:
        _consume(df)

    out_schema: dict[str, Any] = {
        **{k: probe.schema[k] for k in keys},
        "target": pl.String,
        "n": pl.Int64,
        "df": pl.Int64,
        "gw": pl.Float64,
        "p_value": pl.Float64,
    }
    cols_data: dict[str, list[Any]] = {c: [] for c in out_schema}
    for t in values:
        by_key = states_sum[t]
        order = key_order[t]
        if not order and not keys:
            cols_data["target"].append(t)
            cols_data["n"].append(0)
            cols_data["df"].append(q)
            cols_data["gw"].append(None)
            cols_data["p_value"].append(None)
            continue
        for key in order:
            st = by_key[key]
            stat = st.stat(min_samples, few)
            pval = None if stat is None else _native.special_chi2_sf(stat, float(q))
            if pval is not None and math.isnan(pval):
                pval = None
            for ki, k in enumerate(keys):
                cols_data[k].append(key[ki])
            cols_data["target"].append(t)
            cols_data["n"].append(st.count)
            cols_data["df"].append(q)
            cols_data["gw"].append(stat)
            cols_data["p_value"].append(pval)
    return pl.DataFrame(cols_data, schema=out_schema)


# ---------------------------------------------------------------------------
# Anytime-valid bounded mean sequential test (CM-EB and betting e-process)
# ---------------------------------------------------------------------------


@dataclass
class _CmEbGroup:
    t: int = 0
    s: float = 0.0
    v: float = 0.0
    mu_prev: float = 0.5
    lower: float = 0.0
    upper: float = 1.0


@dataclass
class _BettingGroup:
    t: int = 0
    s: float = 0.0
    ss: float = 0.0
    log_e_pos: float = 0.0
    log_e_neg: float = 0.0


def _mean_seqtest_chunk(
    chunk: pl.DataFrame,
    *,
    who: str,
    values: dict[str, pl.Expr],
    keys: list[str],
    bound: float,
    method: Literal["cm_eb", "betting"],
    alpha: float,
    v_opt: float,
    running_intersection: bool,
    clip: bool,
    min_samples: int,
    compare: bool,
    name: str,
    states: dict[str, dict[tuple[Any, ...], Any]],
) -> pl.DataFrame:
    from rheon.polars import eval as _eval

    taken = {*chunk.columns, name}
    d_cols = {t: _eval._free(taken, f"ms_d_{i}") for i, t in enumerate(values)}
    work = chunk.select(*keys, *(expr.alias(d_cols[t]) for t, expr in values.items()))
    n_rows = work.height
    key_cols = [work[k].to_list() for k in keys]
    struct_cols: list[pl.Series] = []
    thresh = math.log(2.0 / alpha)

    for t in values:
        d_list = work[d_cols[t]].to_list()
        by_key = states[t]

        if method == "cm_eb":
            means: list[float | None] = [None] * n_rows
            lowers: list[float | None] = [None] * n_rows
            uppers: list[float | None] = [None] * n_rows
            rejects: list[bool | None] = [None] * n_rows
            counts: list[int | None] = [None] * n_rows
            # Batch per group into the compiled Rust CM-EB kernel.
            grouped_idx: dict[tuple[Any, ...], list[int]] = {}
            grouped_r: dict[tuple[Any, ...], list[float]] = {}
            for i in range(n_rows):
                d_val = d_list[i]
                if d_val is None or not math.isfinite(d_val):
                    continue
                dv = float(d_val)
                if abs(dv) > bound:
                    if clip:
                        dv = max(-bound, min(bound, dv))
                    else:
                        msg = (
                            f"{who}: value {dv:.3g} for target {t!r} exceeds bound={bound:.3g} in "
                            "magnitude. Raise bound to cover the range of the observations, or "
                            "pass clip=True."
                        )
                        raise RefusedValueError(msg)
                r = (dv / bound + 1.0) * 0.5
                key = tuple(kc[i] for kc in key_cols) if keys else ()
                grouped_idx.setdefault(key, []).append(i)
                grouped_r.setdefault(key, []).append(r)

            for key, idxs in grouped_idx.items():
                st = by_key.get(key)
                if st is None:
                    st = _CmEbGroup()
                    by_key[key] = st
                rs = grouped_r[key]
                t0, s0 = st.t, st.s
                lo_r, hi_r, t1, s1, v1, mp1, lo1, hi1 = _native.special_cm_eb_chunk(
                    rs,
                    v_opt,
                    alpha,
                    running_intersection,
                    st.t,
                    st.s,
                    st.v,
                    st.mu_prev,
                    st.lower,
                    st.upper,
                )
                st.t, st.s, st.v, st.mu_prev, st.lower, st.upper = t1, s1, v1, mp1, lo1, hi1
                cur_t = t0
                cur_s = s0
                for pos, row_i in enumerate(idxs):
                    cur_t += 1
                    cur_s += rs[pos]
                    counts[row_i] = cur_t
                    if cur_t < min_samples:
                        continue
                    mu_d = bound * (2.0 * (cur_s / cur_t) - 1.0)
                    lo_d = bound * (2.0 * lo_r[pos] - 1.0)
                    hi_d = bound * (2.0 * hi_r[pos] - 1.0)
                    means[row_i] = mu_d
                    lowers[row_i] = lo_d
                    uppers[row_i] = hi_d
                    rejects[row_i] = (lo_d > 0.0) or (hi_d < 0.0)
            struct_cols.extend(
                [
                    pl.Series(f"mean_{t}", means, dtype=pl.Float64),
                    pl.Series(f"lower_{t}", lowers, dtype=pl.Float64),
                    pl.Series(f"upper_{t}", uppers, dtype=pl.Float64),
                    pl.Series(f"reject_{t}", rejects, dtype=pl.Boolean),
                    pl.Series(f"n_{t}", counts, dtype=pl.Int64),
                ]
            )
        else:
            e_pos_name = f"log_e_a_{t}" if compare else f"log_e_pos_{t}"
            e_neg_name = f"log_e_b_{t}" if compare else f"log_e_neg_{t}"
            log_pos: list[float | None] = [None] * n_rows
            log_neg: list[float | None] = [None] * n_rows
            means_b: list[float | None] = [None] * n_rows
            rejects_b: list[bool | None] = [None] * n_rows
            counts_b: list[int | None] = [None] * n_rows
            for i in range(n_rows):
                d_val = d_list[i]
                if d_val is None or not math.isfinite(d_val):
                    continue
                dv = float(d_val)
                if abs(dv) > bound:
                    if clip:
                        dv = max(-bound, min(bound, dv))
                    else:
                        msg = (
                            f"{who}: value {dv:.3g} for target {t!r} exceeds bound={bound:.3g} in "
                            "magnitude. Raise bound to cover the range of the observations, or "
                            "pass clip=True."
                        )
                        raise RefusedValueError(msg)
                u = dv / bound
                key = tuple(kc[i] for kc in key_cols) if keys else ()
                st_b: _BettingGroup | None = by_key.get(key)
                if st_b is None:
                    st_b = _BettingGroup()
                    by_key[key] = st_b
                mu_prev = st_b.s / st_b.t if st_b.t > 0 else 0.0
                var_prev = (st_b.ss / st_b.t - mu_prev * mu_prev) if st_b.t > 1 else 0.25
                denom = var_prev + mu_prev * mu_prev + 1e-12
                lam_pos = min(max(mu_prev / denom, 0.0), 0.5)
                lam_neg = min(max(-mu_prev / denom, 0.0), 0.5)
                st_b.log_e_pos += math.log1p(lam_pos * u)
                st_b.log_e_neg += math.log1p(-lam_neg * u)
                st_b.t += 1
                st_b.s += u
                st_b.ss += u * u
                counts_b[i] = st_b.t
                if st_b.t < min_samples:
                    continue
                log_pos[i] = st_b.log_e_pos
                log_neg[i] = st_b.log_e_neg
                means_b[i] = bound * (st_b.s / st_b.t)
                rejects_b[i] = (st_b.log_e_pos >= thresh) or (st_b.log_e_neg >= thresh)
            struct_cols.extend(
                [
                    pl.Series(e_pos_name, log_pos, dtype=pl.Float64),
                    pl.Series(e_neg_name, log_neg, dtype=pl.Float64),
                    pl.Series(f"mean_{t}", means_b, dtype=pl.Float64),
                    pl.Series(f"reject_{t}", rejects_b, dtype=pl.Boolean),
                    pl.Series(f"n_{t}", counts_b, dtype=pl.Int64),
                ]
            )

    struct_col = pl.DataFrame(struct_cols).select(pl.struct(pl.all()).alias(name)).to_series()
    return chunk.with_columns(struct_col)


@overload
def mean_seqtest(
    df: pl.DataFrame,
    *,
    bound: float,
    targets: Sequence[str] | None = None,
    a: str | None = None,
    b: str | None = None,
    a_suffix: str = "",
    b_suffix: str = "",
    loss: LossSpec = "squared",
    quantile: float | None = None,
    linex_a: float | None = None,
    method: Literal["cm_eb", "betting"] = "cm_eb",
    alpha: float = 0.05,
    v_opt: float = 1.0,
    running_intersection: bool = False,
    clip: bool = False,
    group: str | Iterable[str] | None = None,
    min_samples: int = 1,
    name: str = "mean_seqtest",
) -> pl.DataFrame: ...


@overload
def mean_seqtest(
    df: pl.LazyFrame,
    *,
    bound: float,
    targets: Sequence[str] | None = None,
    a: str | None = None,
    b: str | None = None,
    a_suffix: str = "",
    b_suffix: str = "",
    loss: LossSpec = "squared",
    quantile: float | None = None,
    linex_a: float | None = None,
    method: Literal["cm_eb", "betting"] = "cm_eb",
    alpha: float = 0.05,
    v_opt: float = 1.0,
    running_intersection: bool = False,
    clip: bool = False,
    group: str | Iterable[str] | None = None,
    min_samples: int = 1,
    name: str = "mean_seqtest",
) -> pl.LazyFrame: ...


@renamed_keywords("rp.eval.mean_seqtest", _GROUP)
def mean_seqtest(
    df: pl.DataFrame | pl.LazyFrame,
    *,
    bound: float,
    targets: Sequence[str] | None = None,
    a: str | None = None,
    b: str | None = None,
    a_suffix: str = "",
    b_suffix: str = "",
    loss: LossSpec = "squared",
    quantile: float | None = None,
    linex_a: float | None = None,
    method: Literal["cm_eb", "betting"] = "cm_eb",
    alpha: float = 0.05,
    v_opt: float = 1.0,
    running_intersection: bool = False,
    clip: bool = False,
    group: str | Iterable[str] | None = None,
    min_samples: int = 1,
    name: str = "mean_seqtest",
) -> pl.DataFrame | pl.LazyFrame:
    """Anytime-valid sequential test of the mean of bounded observations or
    bounded loss differentials.

    Unlike :func:`seqtest`, which tests whether the sign leans one way, this
    tests the mean ``E[d_t]`` directly for observations bounded in
    ``[-bound, bound]``, remaining valid under optional stopping at every row.

    .. rubric:: Parameters

    ``df``
        A ``DataFrame`` or ``LazyFrame`` holding either the numeric columns in
        ``targets`` (column mode) or two specs' output columns ``a`` and ``b``
        (compare mode).
    ``bound``
        Known positive finite bound ``B`` such that ``|d_t| <= B``. Values
        outside ``[-B, B]`` raise :class:`~rheon.RefusedValueError` unless
        ``clip=True``.
    ``targets``, ``a``, ``b``, ``a_suffix``, ``b_suffix``
        As in :func:`seqtest`.
    ``loss``, ``quantile``, ``linex_a``
        Loss function for compare mode (as in :func:`diebold_mariano`); ignored
        at defaults in column mode and refused if customized in column mode.
    ``method``
        ``"cm_eb"`` (default): conjugate-mixture empirical-Bernstein confidence
        sequence (Howard et al. 2021), reporting ``mean_<t>``, ``lower_<t>``,
        ``upper_<t>``, ``reject_<t>``, and ``n_<t>``.
        ``"betting"``: predictable aGRAPA-style betting e-process, reporting
        ``log_e_a_<t>`` / ``log_e_b_<t>`` (or ``log_e_pos_<t>`` /
        ``log_e_neg_<t>`` in column mode), ``mean_<t>``, ``reject_<t>``, and
        ``n_<t>``.
    ``alpha``
        Two-sided significance level in ``(0, 1)`` (0.05 by default).
    ``v_opt``
        Intrinsic-time variance target for ``method="cm_eb"`` (1.0 by default).
    ``running_intersection``
        If ``True`` under ``method="cm_eb"``, tighten ``lower`` / ``upper``
        monotonically via running intersection (``False`` by default).
    ``clip``
        If ``True``, clip out-of-range observations to ``[-bound, bound]``;
        ``False`` (default) refuses values past ``bound``.
    ``group``
        Optional group column or list of columns to run one test per group.
    ``min_samples``
        Minimum observations folded before emitting test fields (1 by default).
    ``name``
        Name of the output column added (``"mean_seqtest"`` by default).

    .. rubric:: Returns

    ``df`` (or a streaming ``LazyFrame``) with an added output column ``name``
    containing per-target fields: ``mean_<t>``, ``lower_<t>``, ``upper_<t>``,
    ``reject_<t>``, and ``n_<t>`` under ``method="cm_eb"``; or
    ``log_e_a_<t>`` / ``log_e_b_<t>`` (``log_e_pos_<t>`` / ``log_e_neg_<t>``
    in column mode), ``mean_<t>``, ``reject_<t>``, and ``n_<t>`` under
    ``method="betting"``.

    .. rubric:: Examples

    This code reads ``df`` from `Example data
    <https://github.com/rheon-dev/rheon/blob/main/docs/EXAMPLE-DATA.md>`_:

    .. code-block:: python

        a = rp.spec.ewridge("ridge", targets=["y"], features=["x0", "x1"], half_life=100.0)
        b = rp.spec.rls("rls", targets=["y"], features=["x0", "x1"], half_life=100.0)
        out = rp.ModelBank([a, b]).fit_predict(df)
        res = rp.eval.mean_seqtest(out, a="ridge", b="rls", bound=5.0, clip=True)
    """
    from rheon.polars import eval as _eval

    who = "rp.eval.mean_seqtest"
    _eval._frame(who, df, lazy=True)
    bound_f = _check_positive_finite(who, "bound", bound)
    alpha_f = _check_alpha(who, alpha)
    v_opt_f = _check_positive_finite(who, "v_opt", v_opt)
    ri_b = _check_bool(who, "running_intersection", running_intersection)
    clip_b = _check_bool(who, "clip", clip)
    if not isinstance(method, str) or method not in _VALID_MEAN_METHODS:
        msg = (
            f"{who}: method must be one of {_VALID_MEAN_METHODS}, but it is {method!r}. "
            "Use 'cm_eb' or 'betting'."
        )
        raise ValueError(msg)
    if isinstance(targets, str):
        msg = (
            f"{who}: targets must be a list of column names, but it is the string {targets!r}. "
            f"Give the names in a list, such as [{targets!r}]."
        )
        raise WrongTypeError(msg)
    _eval._check_min_samples(who, min_samples)
    probe = _eval._probe(df)
    _eval._check_name(who, probe, name)
    _eval._check_columns(probe, who, group=group)
    keys = _eval._group_keys(group)

    if (a is None) != (b is None):
        msg = (
            f"{who}: a and b go together, but only one is given. Name both specs to compare "
            "them, or neither."
        )
        raise ValueError(msg)

    compare = a is not None and b is not None
    if compare:
        assert a is not None and b is not None
        pairs = _eval._paired_residuals(probe, a, b, targets, a_suffix, b_suffix, who)
        values = build_loss_diffs(who, pairs, loss, quantile, linex_a)
    else:
        if loss != "squared" or quantile is not None or linex_a is not None:
            msg = (
                f"{who}: loss, quantile and linex_a apply only in compare mode (with a and b). "
                "Leave them out when testing columns directly."
            )
            raise ValueError(msg)
        if targets is None or len(targets) == 0:
            msg = (
                f"{who}: targets is required without a and b, but it is empty or None. "
                "Name the columns to test in targets, or name two specs as a and b."
            )
            raise ValueError(msg)
        for t in targets:
            if t not in probe.columns:
                from rheon.polars._errors import ColumnNotFoundError

                msg = (
                    f"{who}: column {t!r} is not in the frame. "
                    "Check the column name or add the column before calling."
                )
                raise ColumnNotFoundError(msg)
            if not probe.schema[t].is_numeric():
                msg = (
                    f"{who}: column {t!r} must be numeric, but it has dtype {probe.schema[t]}. "
                    "Cast the column to float or drop it from targets."
                )
                raise WrongTypeError(msg)
        values = {t: pl.col(t).cast(pl.Float64) for t in targets}

    if isinstance(df, pl.LazyFrame):
        schema = _mean_seqtest_chunk(
            probe,
            who=who,
            values=values,
            keys=keys,
            bound=bound_f,
            method=method,
            alpha=alpha_f,
            v_opt=v_opt_f,
            running_intersection=ri_b,
            clip=clip_b,
            min_samples=min_samples,
            compare=compare,
            name=name,
            states={t: {} for t in values},
        ).schema
        needed = set(keys).union(*(e.meta.root_names() for e in values.values()))

        def _step() -> Callable[[pl.DataFrame], pl.DataFrame]:
            st: dict[str, dict[tuple[Any, ...], Any]] = {t: {} for t in values}

            def _one(chunk: pl.DataFrame) -> pl.DataFrame:
                return _mean_seqtest_chunk(
                    chunk,
                    who=who,
                    values=values,
                    keys=keys,
                    bound=bound_f,
                    method=method,
                    alpha=alpha_f,
                    v_opt=v_opt_f,
                    running_intersection=ri_b,
                    clip=clip_b,
                    min_samples=min_samples,
                    compare=compare,
                    name=name,
                    states=st,
                )

            return _one

        return _eval._row_query(df, needed, schema, _step)

    states: dict[str, dict[tuple[Any, ...], Any]] = {t: {} for t in values}
    return _mean_seqtest_chunk(
        df,
        who=who,
        values=values,
        keys=keys,
        bound=bound_f,
        method=method,
        alpha=alpha_f,
        v_opt=v_opt_f,
        running_intersection=ri_b,
        clip=clip_b,
        min_samples=min_samples,
        compare=compare,
        name=name,
        states=states,
    )
