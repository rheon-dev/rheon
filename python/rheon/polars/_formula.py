"""Formulas: a Polars expression over the window operators, kept as this
library's own compact tree.

An operator such as :func:`~rheon.polars.ops.ewm_mean` returns a genuine
``pl.Expr``: a column reference whose name carries the operator's compact
JSON after ``@po:``. So it composes with ``pl.col``, ``pl.lit``, numbers,
``when/then`` and the element-wise functions on either side of a Polars
operator. At build time :func:`to_tree` reads the expression's own tree
(``expr.meta.serialize(format="json")``, under the caller's Polars). It
walks seven node kinds -- ``Column``, ``Literal``, ``BinaryExpr``,
``Function`` (log, exp, abs, sqrt, pow, clip, fill_null, is_null,
is_not_null, negate), ``Ternary``, ``Cast`` and ``Alias`` -- into nested
lists, with the operators inlined as nodes of their own:

.. code-block:: python

    ["-", ["rewm_mean", ["col", "mid"], {"half_life": "10s", "window_size": "1m"}], ["col", "mid"]]

The serialized tree as a whole is unstable across Polars versions, which is
why only these nodes are read, here, and the Rust side rebuilds the formula
through Polars' public builders. A node outside the seven -- ``shift``,
``cum_sum``, a rolling window, ``over``, an aggregation -- would depend on
the chunking, where one chunk or a thousand must give the same numbers, and
is refused by name. :func:`from_tree`
rebuilds the expression in Python, for a round trip.
"""

from __future__ import annotations

import json
import math
import numbers
import re
from datetime import timedelta
from typing import Any

import polars as pl

from rheon._duration import _INFINITY, duration_text
from rheon._errors import RefusedValueError, WrongTypeError
from rheon._rheon import parse_duration

#: The prefix of the column an operator stands in as.
# The @po: spelling is kept: it is part of the serialized formula, so a rename
# would break formulas saved by 1.x.
PREFIX = "@po:"

#: The window operators, and the row operators, computed a row at a time
#: from the group's previous rows: the steps, then the last three.
OPERATORS = (
    "ewm_mean",
    "rewm_mean",
    "ewm_sum",
    "rewm_sum",
    "ewm_rate",
    "rewm_rate",
    "ewm_var",
    "ewm_std",
    "ewm_quantile",
    "rewm_quantile",
    "ewm_moments",
    "rolling_max",
    "rolling_min",
    "rrolling_max",
    "rrolling_min",
    "increment",
    "cross_mean",
    "cross_std",
    "cross_rank",
    "cross_quantile",
    "cross_ref",
    "ewm_profile",
    "tick_sign",
    "run_length",
    "clock_lag",
    "clock_drift",
    "clock_step",
)

#: The column the unknown of :func:`~rheon.polars.ops.invert` stands in as
#: while its function is read: an operator's prefix and the leaf's list.
UNKNOWN_COLUMN = PREFIX + '["unknown"]'

#: An invert's node: its head, its four parts and its parameters.
INVERT_ARITY = 6

#: The cross-group operators: a statistic over every group's row at one
#: stamp of the clock, with no kernel of their own.
CROSS = ("cross_mean", "cross_std", "cross_rank", "cross_quantile", "cross_ref")

#: The operators that read the row before, not a window: their input is a
#: formula of the row, and a window operator's input may read them.
STEPS = ("increment", "tick_sign")

#: The row operators: no window, each read off its group's previous rows.
ROW_OPERATORS = ("increment", "tick_sign", "run_length", "clock_lag", "clock_drift", "clock_step")

#: What a clock step reads of its time's steps.
STEP_STATS = ("mean", "var", "std", "zeros", "backs")

#: The row operators that read two times, an arrival (their input) and an
#: event (a parameter).
CLOCKS = ("clock_lag", "clock_drift")

#: The operators that take ``bias``: a variance's own (task 212).
VARIANCES = ("ewm_var", "ewm_std")

#: The operators that take ``quantile``, the level (task 272 (d)).
QUANTILES = ("ewm_quantile", "rewm_quantile")

#: The words a moments operator's ``kind`` takes (task 554).
MOMENT_KINDS = ("central", "raw", "standardized", "cumulant")

#: The highest order with an unbiased form.
MAX_UNBIASED_ORDER = 4

#: The extrema (task 561): no ``half_life``, and a ``since`` event.
EXTREMA = ("rolling_max", "rolling_min", "rrolling_max", "rrolling_min")

_BINARY = {
    "Plus": "+",
    "Minus": "-",
    "Multiply": "*",
    "TrueDivide": "/",
    "Eq": "==",
    "NotEq": "!=",
    "Lt": "<",
    "LtEq": "<=",
    "Gt": ">",
    "GtEq": ">=",
    "And": "&",
    "Or": "|",
    "LogicalAnd": "&",
    "LogicalOr": "|",
}

_UNARY = {"Negate": "neg", "Abs": "abs", "Exp": "exp"}

_DTYPES = {
    "Float64": pl.Float64,
    "Float32": pl.Float32,
    "Int64": pl.Int64,
    "Int32": pl.Int32,
    "Int16": pl.Int16,
    "Int8": pl.Int8,
    "UInt64": pl.UInt64,
    "UInt32": pl.UInt32,
    "UInt16": pl.UInt16,
    "UInt8": pl.UInt8,
    "Boolean": pl.Boolean,
    "String": pl.String,
    # A date or a time-of-day literal serializes as its integer under the
    # dtype's name, and is carried through a cast to it: read as the bare
    # integer, its text and its arithmetic were not Polars' (review
    # 2026-10-05, YB1).
    "Date": pl.Date,
    "Time": pl.Time,
}


#: What a formula reads around the operators, in the words a reader looks
#: for, each with the nodes of the compact tree it stands for. The refusal
#: message lists these words, and a test holds their nodes to the ones
#: :func:`from_tree` rebuilds and to the Rust reader's list, and their words
#: to the table in the windows guide, so the message cannot leave out a node
#: that is accepted.
ACCEPTED: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("columns", ("col",)),
    ("literals", ("lit",)),
    ("arithmetic", ("+", "-", "*", "/")),
    ("negation", ("neg",)),
    ("comparisons", ("==", "!=", "<", "<=", ">", ">=")),
    ("& and |", ("&", "|")),
    ("log", ("log",)),
    ("exp", ("exp",)),
    ("abs", ("abs",)),
    ("sqrt", ("sqrt",)),
    ("pow", ("**",)),
    ("clip", ("clip",)),
    ("fill_null", ("fill_null",)),
    ("is_null", ("is_null",)),
    ("is_not_null", ("is_not_null",)),
    ("when/then/otherwise", ("when",)),
    ("cast", ("cast",)),
    ("alias", ("alias",)),
)


def accepted_text() -> str:
    """The words of :data:`ACCEPTED` as one English list."""
    words = [word for word, _ in ACCEPTED]
    return ", ".join(words[:-1]) + " and " + words[-1]


class FormulaError(ValueError):
    """An expression a formula cannot carry: a node that is not element-wise,
    a literal with no JSON form, a cast to a dtype not read."""


def _refuse(who: str, kind: str, detail: str = "") -> FormulaError:
    """The refusal of a node a formula cannot carry, named by ``kind`` (and
    ``detail``, which follows it in the sentence)."""
    return FormulaError(
        f"{who}: a formula over window operators reads each row alone, but its expression (a "
        f'column computation, such as pl.col("x") + 1) holds {kind}{detail}, which it cannot '
        f"read. The formula is computed a chunk at a time, so a part that reads other rows, "
        f"such as a shift, a cumulative or rolling function or an aggregation, would read "
        f"only its chunk's rows. Build the formula from "
        f"{accepted_text()} around the operators, and compute anything else as a column first."
    )


#: What a refused cast is told: the types a formula casts to.
_CASTS = (
    " (a formula casts only to Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64, "
    "Float32, Float64, Boolean, String, Date or Time)"
)

#: Polars' serialized names of the binary operators a formula refuses, as
#: the reader writes them.
_REFUSED_OPERATORS = {
    "FloorDivide": "//",
    "Modulus": "%",
    "Xor": "^",
    "EqValidity": "eq_missing",
    "NotEqValidity": "ne_missing",
}

#: Polars' serialized names of the expressions that read other rows, as the
#: reader writes them.
_REFUSED_KINDS = {
    "Agg": "an aggregation",
    "Over": "the method over",
    "Window": "the method over",
    "Len": "pl.len()",
    "Not": "the operator ~",
    "Selector": "pl.all() or another selector",
    "Wildcard": "pl.all()",
    "RollingExpr": "a rolling function",
    "RollingExprBy": "a rolling function",
}

#: Function families Polars serializes under one name, the method in the
#: detail: ``.sin()`` is ``{"Trigonometry": "Sin"}``.
_FAMILIES = ("Boolean", "Pow", "Trigonometry", "StringExpr", "TemporalExpr")


def _method(name: str) -> str:
    """A serialized function's name as the method the reader called: Polars
    serializes ``.cum_sum()`` as ``CumSum``."""
    snake = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name).lower()
    # Polars' own methods spell the inverse functions as one word: arcsin.
    return re.sub(r"^arc_", "arc", snake)


def _reserved(who: str, column: str) -> FormulaError:
    """The refusal of a column whose name an operator's own column takes."""
    return FormulaError(
        f"{who}: the column {column!r} has a name starting with {PREFIX!r}, which is kept for "
        f"the window operators' own columns. Rename the column."
    )


def _literal(v: Any, who: str) -> tuple[Any, str | None]:
    """The JSON value of a serialized literal and, for a typed one
    (``pl.lit(x, dtype=...)``, or a numpy scalar), its dtype's name; or a
    refusal."""
    if v == "Null":
        return None, None
    if isinstance(v, dict) and len(v) == 1:
        ((kind, inner),) = v.items()
        if kind in ("Dyn", "Scalar") and isinstance(inner, dict) and len(inner) == 1:
            ((dtype, value),) = inner.items()
            if dtype == "Null":
                return None, None
            if value is None:
                # Polars serializes inf and nan alike, as a null value under
                # the float's type, so a formula cannot carry either; a clip
                # with one bound has a form of its own (task 159, F4).
                raise _refuse(
                    who,
                    "a literal that is inf or nan",
                    " (for a clip with one bound, leave the other out)",
                )
            if isinstance(value, (bool, int, float, str)):
                # A typed numeric literal keeps its type through a cast, so
                # its dtype and last bits are Polars' own (task 159, F3), and
                # so does a date's or a time's integer (YB1). A string or a
                # boolean is `Scalar` untyped too, and has one type to be.
                numeric = dtype in _DTYPES and dtype not in ("String", "Boolean")
                typed = dtype if kind == "Scalar" and numeric else None
                return value, typed
    raise _refuse(who, "a literal of a type a formula does not carry, such as a list or bytes")


def _walk(node: Any, who: str, unknown: bool = False) -> Any:
    if isinstance(node, str) and node in _REFUSED_KINDS:
        raise _refuse(who, _REFUSED_KINDS[node])
    if not isinstance(node, dict) or len(node) != 1:
        raise _refuse(who, "a part of the expression it does not recognize")
    ((kind, body),) = node.items()
    if kind == "Column":
        if body == UNKNOWN_COLUMN:
            if unknown:
                return ["unknown"]
            raise FormulaError(
                f"{who}: an expression holds rp.ops.unknown() outside the function f of "
                f"rp.ops.invert. The unknown stands only in the function an invert solves. Write "
                f"the target and the bracket, and every other formula, without it."
            )
        if isinstance(body, str) and body.startswith(PREFIX):
            reserved = _reserved(who, body)
            try:
                op = json.loads(body[len(PREFIX) :])
            except json.JSONDecodeError:
                raise reserved from None
            # Only an operator's own node is one: a list headed by an
            # operator, with its input and, but for a step, its
            # parameters. A short list raised a bare IndexError where the tree
            # is walked, and a number was refused as holding no operator
            # (review round 4, PD3).
            # An invert's node: its four parts and its parameters, checked
            # where rp.ops.invert wrote it and again where the run reads it.
            if isinstance(op, list) and len(op) == INVERT_ARITY and op[0] == "invert":
                return op
            if not (isinstance(op, list) and len(op) in (2, 3) and op[0] in OPERATORS):
                raise reserved
            return op
        return ["col", body]
    if kind == "Literal":
        value, typed = _literal(body, who)
        # A null literal is `["lit"]`: TOML has no null (review R2, P3).
        node = ["lit"] if value is None else ["lit", value]
        return ["cast", node, typed] if typed else node
    if kind == "BinaryExpr":
        op = body.get("op")
        if op not in _BINARY:
            raise _refuse(who, f"the operator {_REFUSED_OPERATORS.get(op, _method(str(op)))}")
        return [_BINARY[op], _walk(body["left"], who, unknown), _walk(body["right"], who, unknown)]
    if kind == "Ternary":
        return [
            "when",
            _walk(body["predicate"], who, unknown),
            _walk(body["truthy"], who, unknown),
            _walk(body["falsy"], who, unknown),
        ]
    if kind == "Cast":
        dtype = body.get("dtype")
        name = dtype.get("Literal") if isinstance(dtype, dict) else dtype
        if isinstance(name, dict) and len(name) == 1:
            # A dtype with parameters -- Datetime, Duration, Decimal, List,
            # Categorical, Enum, Struct -- is a mapping under its name, which
            # the lookup below could not hash (review 2026-10-05, YB2).
            raise _refuse(who, f"a cast to {json.dumps(next(iter(name)))}", _CASTS)
        if not isinstance(name, str) or name not in _DTYPES:
            raise _refuse(
                who,
                f"a cast to {json.dumps(name if isinstance(name, str) else dtype)[:60]}",
                _CASTS,
            )
        # Strict, Polars' default, unless the cast says otherwise; a wrapping
        # cast (`wrap_numerical=True`) has no form here.
        options = body.get("options", "Strict")
        if options == "Strict":
            return ["cast", _walk(body["expr"], who, unknown), name]
        if options == "NonStrict":
            return ["cast", _walk(body["expr"], who, unknown), name, "non_strict"]
        raise _refuse(who, f"a cast with options {json.dumps(options)}")
    if kind == "Alias":
        expr, name = body
        return ["alias", _walk(expr, who, unknown), name]
    if kind == "Function":
        fn = body.get("function")
        args = [_walk(a, who, unknown) for a in body.get("input", [])]
        if isinstance(fn, str):
            if fn in _REFUSED_KINDS:
                raise _refuse(who, _REFUSED_KINDS[fn])
            if fn in _UNARY and len(args) == 1:
                return [_UNARY[fn], args[0]]
            if fn == "Log" and len(args) == 2:
                return ["log", args[0], args[1]]
            if fn == "FillNull" and len(args) == 2:
                return ["fill_null", args[0], args[1]]
            raise _refuse(who, f"the method {_method(fn)}")
        if isinstance(fn, dict) and len(fn) == 1:
            ((name, detail),) = fn.items()
            if name == "Pow" and detail == "Generic" and len(args) == 2:
                return ["**", args[0], args[1]]
            if name == "Pow" and detail == "Sqrt" and len(args) == 1:
                return ["sqrt", args[0]]
            if name == "Boolean" and detail in ("IsNull", "IsNotNull") and len(args) == 1:
                return ["is_null" if detail == "IsNull" else "is_not_null", args[0]]
            if name == "Clip" and isinstance(detail, dict):
                rest = args[1:]
                lo = rest.pop(0) if detail.get("has_min") else ["lit"]
                hi = rest.pop(0) if detail.get("has_max") else ["lit"]
                return ["clip", args[0], lo, hi]
            for part in (name, detail):
                if isinstance(part, str) and part in _REFUSED_KINDS:
                    raise _refuse(who, _REFUSED_KINDS[part])
            shown = name
            if name in _FAMILIES and isinstance(detail, str):
                shown = detail
            elif name in _FAMILIES and isinstance(detail, dict) and detail:
                shown = str(next(iter(detail)))
            raise _refuse(who, f"the method {_method(shown)}")
        if isinstance(fn, dict) and fn:
            raise _refuse(who, f"the method {_method(str(next(iter(fn))))}")
        raise _refuse(who, "a function it does not recognize")
    raise _refuse(who, _REFUSED_KINDS.get(kind, "a part of the expression it does not recognize"))


def to_tree(expr: pl.Expr, who: str = "window formula", *, unknown: bool = False) -> list[Any]:
    """The compact tree of ``expr``, the operators inlined; under
    ``unknown``, the stand-in of an invert's unknown read as its leaf.

    Raises :class:`FormulaError` for an expression a formula cannot carry,
    naming ``who``, the call the expression was given to.
    """
    return _walk(json.loads(expr.meta.serialize(format="json")), who, unknown)


def operators(tree: Any) -> list[list[Any]]:
    """Every operator node in ``tree``, outermost first."""
    out: list[list[Any]] = []

    def walk(node: Any) -> None:
        if not isinstance(node, list) or not node:
            return
        head = node[0]
        if head in OPERATORS:
            out.append(node)
            walk(node[1])
            if len(node) > 2 and isinstance(node[2], dict):
                for key in ("weight", "where", "event"):
                    if key in node[2]:
                        walk(node[2][key])
        elif head not in ("col", "lit"):
            for arg in node[1:]:
                walk(arg)

    walk(tree)
    return out


#: The operators that look ahead: a formula with one is a target of the
#: row's future (docs/PLAN.md task 104).
FORWARD = ("rewm_mean", "rewm_sum", "rewm_rate", "rewm_quantile", "rrolling_max", "rrolling_min")


def columns(tree: Any) -> list[str]:
    """Every column ``tree`` reads, in first-seen order."""
    out: list[str] = []

    def walk(node: Any) -> None:
        if not isinstance(node, list) or not node:
            return
        head = node[0]
        if head == "col":
            if node[1] not in out:
                out.append(node[1])
        elif head == "lit":
            return
        elif head in OPERATORS:
            walk(node[1])
            if len(node) > 2 and isinstance(node[2], dict):
                for key in ("weight", "where", "since", "event"):
                    if key in node[2]:
                        walk(node[2][key])
                if node[2].get("neighbors") is not None and node[2]["neighbors"] not in out:
                    out.append(node[2]["neighbors"])
        else:
            for arg in node[1:]:
                walk(arg)

    walk(tree)
    return out


def looks_ahead(tree: Any) -> bool:
    """Whether ``tree`` holds an operator over the rows after each row."""
    return any(op[0] in FORWARD for op in operators(tree))


def compact(node: Any) -> str:
    """One operator node as the text its column carries: compact JSON with
    sorted keys, so one operator asked for twice is one column."""
    return json.dumps(node, separators=(",", ":"), sort_keys=True)


def operator_column(node: list[Any]) -> pl.Expr:
    """The column an operator stands in as."""
    return pl.col(PREFIX + compact(node))


def from_tree(tree: Any) -> pl.Expr:
    """The expression a tree stands for, rebuilt with Polars' builders; an
    operator comes back as its column."""
    if not isinstance(tree, list) or not tree or not isinstance(tree[0], str):
        raise FormulaError(
            f"from_tree: a formula node is a list headed by its kind, but it is {tree!r}. "
            f"Give it a tree that to_tree wrote."
        )
    head, args = tree[0], tree[1:]
    if head in OPERATORS or head == "invert":
        return operator_column(tree)
    if head == "unknown":
        return pl.col(UNKNOWN_COLUMN)
    if head == "col":
        return pl.col(args[0])
    if head == "lit":
        return pl.lit(args[0]) if args else pl.lit(None)
    if head == "clip":
        # A missing bound is the null literal `["lit"]` (or a bare null in
        # a tree written before TOML could not carry one).
        lo = None if args[1] in (None, ["lit"]) else from_tree(args[1])
        hi = None if args[2] in (None, ["lit"]) else from_tree(args[2])
        return from_tree(args[0]).clip(lo, hi)
    e = [from_tree(a) for a in args if isinstance(a, list)]
    match head:
        case "+":
            return e[0] + e[1]
        case "-":
            return e[0] - e[1]
        case "*":
            return e[0] * e[1]
        case "/":
            return e[0] / e[1]
        case "**":
            return e[0] ** e[1]
        case "==":
            return e[0] == e[1]
        case "!=":
            return e[0] != e[1]
        case "<":
            return e[0] < e[1]
        case "<=":
            return e[0] <= e[1]
        case ">":
            return e[0] > e[1]
        case ">=":
            return e[0] >= e[1]
        case "&":
            return e[0] & e[1]
        case "|":
            return e[0] | e[1]
        case "neg":
            return -e[0]
        case "abs":
            return e[0].abs()
        case "exp":
            return e[0].exp()
        case "sqrt":
            return e[0].sqrt()
        case "is_null":
            return e[0].is_null()
        case "is_not_null":
            return e[0].is_not_null()
        case "log":
            return e[0].log(args[1][1]) if args[1][0] == "lit" else e[0].log() / e[1].log()
        case "fill_null":
            return e[0].fill_null(e[1])
        case "when":
            return pl.when(e[0]).then(e[1]).otherwise(e[2])
        case "cast":
            return from_tree(args[0]).cast(_DTYPES[args[1]], strict=len(args) < 3)
        case "alias":
            return from_tree(args[0]).alias(args[1])
    raise FormulaError(
        f"from_tree: the formula node {head!r} is not one it reads. Give it a tree that to_tree "
        f"wrote."
    )


def _input_tree(who: str, value: Any) -> list[Any]:
    if isinstance(value, str):
        if value.startswith(PREFIX):
            raise _reserved(who, value)
        return ["col", value]
    if isinstance(value, pl.Expr):
        tree = to_tree(value, who)
        if any(op[0] not in STEPS for op in operators(tree)):
            raise ValueError(
                f"{who}: the input holds another window operator, but an operator's input may "
                f"hold only columns and Polars expressions of the same row (a column "
                f'computation, such as pl.col("x") + 1), rp.increment or rp.ops.tick_sign. To apply '
                f"an operator to another's output, add the inner one as a column with "
                f"rp.stream.with_windows first, and pass that column."
            )
        return tree
    raise TypeError(
        f"{who}: the input must be a column name or a pl.Expr, but it is of type "
        f'{type(value).__name__}. Pass a column name or an expression, such as pl.col("x").'
    )


def _span(who: str, key: str, value: Any, *, inf_ok: bool) -> float | str:
    if isinstance(value, bool) or not isinstance(value, (numbers.Real, str, timedelta, pl.Expr)):
        raise TypeError(
            f"{who}: {key} must be a number or a duration, but it is of type "
            f"{type(value).__name__}. "
            f'Give {key} a number of clock units, or a duration such as "5m".'
        )
    # A word is checked as the number it names, and a duration as its length,
    # here, as a number is: "nan", "-inf", "reset", an infinite window and a
    # duration at or below 0 passed, and were refused only when the plan was
    # built, under a serde path (review round 4, PD9).
    if isinstance(value, str):
        word = value.strip().lower()
        if word == "reset":
            # `session_gap`'s word, which the clock parameters' words admit.
            raise ValueError(
                f"{who}: {key} must be a number or a duration, but it is {value!r}. "
                f'Give {key} a number of clock units, or a duration such as "5m".'
            )
        if word == "nan" or word in _INFINITY:
            _number(who, key, math.nan if word == "nan" else _INFINITY[word], value, inf_ok=inf_ok)
            return value.strip()
    if isinstance(value, numbers.Real):
        return _number(who, key, float(value), value, inf_ok=inf_ok)
    text = duration_text(value, who, key)
    if parse_duration(text) <= 0:
        raise ValueError(
            f"{who}: {key} must be above 0, but it is {text}. Give {key} a positive duration."
        )
    return text


def _number(who: str, key: str, x: float, shown: Any, *, inf_ok: bool) -> float | str:
    """A clock parameter given as the number ``x`` (``shown`` as written):
    above 0, and finite unless ``inf_ok``; an infinity is ``"inf"``."""
    above = "above 0" if inf_ok else "finite and above 0"
    inf = ", or inf for no decay" if inf_ok else ""
    if math.isnan(x):
        raise ValueError(
            f"{who}: {key} must be {above}, but it is NaN. Give {key} a positive number of "
            f"clock units, or a duration{inf}."
        )
    if math.isinf(x):
        if not inf_ok or x < 0:
            raise ValueError(
                f"{who}: {key} must be {above}, but it is {shown}. Give {key} a positive "
                f"number of clock units, or a duration{inf}."
            )
        return "inf"
    if x <= 0:
        raise ValueError(
            f"{who}: {key} must be {above}, but it is {shown}. Give {key} a positive number of "
            f"clock units, or a duration{inf}."
        )
    return x


#: The most rows ``min_samples`` can ask for: the windows core counts them
#: in a ``u32``.
MAX_MIN_SAMPLES = 2**32 - 1

#: The most an extremum's ``max_state_bytes`` may be: 64 MiB a group.
MAX_STATE_BYTES = 64 << 20


def operator(
    name: str,
    input: Any,
    *,
    half_life: Any = None,
    window_size: Any = None,
    closed: str = "right",
    min_samples: int = 1,
    partial: str | None = None,
    bias: Any = None,
    quantile: Any = None,
    weight: Any = None,
    since: Any = None,
    max_state_bytes: Any = None,
    method: Any = "average",
    descending: Any = False,
    interpolation: Any = None,
    neighbors: Any = None,
    where: Any = None,
    who: str | None = None,
) -> pl.Expr:
    """The expression of operator ``name`` over ``input`` with these
    parameters, checked here so a wrong value names its parameter; ``bias``
    a variance's alone, ``quantile`` a quantile's. ``who`` is the call a
    message names, ``po.<name>`` unless given: a window looking ahead is
    ``ewm_*(reverse=True)`` to the caller and ``rewm_*`` in the tree."""
    who = who or f"rp.{name}"
    tree = _input_tree(who, input)
    if name in STEPS:
        return operator_column([name, tree])
    if name in CROSS:
        return _cross(
            name,
            tree,
            who,
            closed=closed,
            bias=bias,
            weight=weight,
            max_state_bytes=max_state_bytes,
            quantile=quantile,
            method=method,
            descending=descending,
            interpolation=interpolation,
            neighbors=neighbors,
            where=where,
        )
    params: dict[str, Any] = {}
    if name in EXTREMA:
        # An extremum weighs no row, so it has no half_life.
        if half_life is not None:
            raise TypeError(
                f"{who}: half_life is given, but an extremum weighs no row. Leave it out."
            )
    elif half_life is None:
        raise TypeError(
            f"{who}: half_life is required, but it is not given. Give half_life in clock units, "
            f"or inf for no decay."
        )
    else:
        params["half_life"] = _span(who, "half_life", half_life, inf_ok=True)
    if window_size is not None:
        params["window_size"] = _span(who, "window_size", window_size, inf_ok=False)
    elif name.startswith("rewm") or name.startswith("rrolling"):
        raise RefusedValueError(
            f"{who}: window_size is required when the operator looks ahead, since the rows "
            f"after a row have no end without one, but it is not given. Give window_size in "
            f"clock units."
        )
    if closed not in ("right", "left", "both", "none"):
        raise ValueError(
            f'{who}: closed must be "right", "left", "both" or "none", but it is {closed!r}. '
            f"Set closed to one of those words."
        )
    params["closed"] = closed
    # Any integer Polars' `rolling_*_by` takes, a numpy one included, up to
    # the core's ceiling, named; past it the refusal came later and said "at
    # least 1" (review round 4, PD4).
    # A sum may ask for no minimum: with 0, as in Polars' `rolling_sum_by`, a
    # window with no value sums to 0.0 where every other window gives null.
    least = 0 if name in ("ewm_sum", "rewm_sum") else 1
    wrong_type = isinstance(min_samples, bool) or not isinstance(min_samples, numbers.Integral)
    if wrong_type or not least <= min_samples <= MAX_MIN_SAMPLES:
        msg = (
            f"{who}: min_samples must be an integer of at least {least} and at most "
            f"{MAX_MIN_SAMPLES}, but it is {min_samples!r}. Set min_samples to a whole number "
            f"of rows in that range."
        )
        raise WrongTypeError(msg) if wrong_type else ValueError(msg)
    params["min_samples"] = int(min_samples)
    if partial is not None:
        if partial not in ("keep", "null", "drop"):
            raise ValueError(
                f'{who}: partial must be "keep", "null" or "drop", but it is {partial!r}. '
                f"Set partial to one of those words."
            )
        if window_size is None:
            # What a window cut short gives needs a window to cut; without
            # one it was taken and did nothing (review 2026-10-05, YB5).
            raise ValueError(
                f"{who}: partial needs window_size, but window_size is not given. partial says "
                f"what a window cut short by a clock break (a step past gap_cap, or a session "
                f"change) gives, and without window_size no window is cut short. Give "
                f"window_size, or leave out partial."
            )
        params["partial"] = partial
    if name in VARIANCES:
        # A bool, as Polars' `bias` is; written always, as `closed` is, so
        # one variance asked for twice is one column.
        if not isinstance(bias, bool):
            raise TypeError(
                f"{who}: bias must be True or False, but it is of type {type(bias).__name__}. "
                f"Set bias to True or False."
            )
        params["bias"] = bias
    elif bias is not None:
        raise TypeError(
            f"{who}: bias is given, but only a variance (ewm_var, ewm_std) takes it. "
            f"Leave out bias."
        )
    if name in QUANTILES:
        # A number in (0, 1), as Polars' `quantile` takes it; NaN and the
        # ends refused by name.
        if isinstance(quantile, bool) or not isinstance(quantile, numbers.Real):
            raise TypeError(
                f"{who}: quantile must be a number, but it is of type {type(quantile).__name__}. "
                f"Set quantile to a number above 0 and below 1."
            )
        if not 0.0 < float(quantile) < 1.0:
            raise ValueError(
                f"{who}: quantile must be above 0 and below 1, but it is {quantile!r}. "
                f"Set quantile to a number above 0 and below 1, such as 0.5 for the median."
            )
        params["quantile"] = float(quantile)
    elif quantile is not None:
        raise TypeError(
            f"{who}: quantile is given, but only ewm_quantile takes it. Leave out quantile."
        )
    if since is not None:
        if name not in EXTREMA:
            raise TypeError(
                f"{who}: since is given, but only rolling_max and rolling_min take it. "
                f"Leave out since."
            )
        if name.startswith("rrolling"):
            raise RefusedValueError(
                f"{who}: since cannot be combined with reverse=True yet. A window looking ahead "
                f"would end at the next event, and whether such a window is whole or partial "
                f"has no rule yet. Leave out since, or read the rows at or before each row."
            )
        since_tree = _input_tree(f"{who} since", since)
        if operators(since_tree):
            raise ValueError(
                f"{who}: since holds an operator, but it may hold only columns and Polars "
                f'expressions of the same row, such as pl.col("new_period"). Add the operator '
                f"as a column with rp.stream.with_windows first, and pass that column."
            )
        params["since"] = since_tree
    if max_state_bytes is not None:
        if name not in EXTREMA:
            raise TypeError(
                f"{who}: max_state_bytes is given, but only rolling_max and rolling_min take "
                f"it. Leave out max_state_bytes."
            )
        wrong = isinstance(max_state_bytes, bool) or not isinstance(
            max_state_bytes, numbers.Integral
        )
        if wrong or not 0 < max_state_bytes <= MAX_STATE_BYTES:
            msg = (
                f"{who}: max_state_bytes must be an integer above 0 and at most "
                f"{MAX_STATE_BYTES} (64 MiB), but it is {max_state_bytes!r}. Set it to the "
                f"bytes a group's window may hold."
            )
            raise WrongTypeError(msg) if wrong else ValueError(msg)
        params["max_state_bytes"] = int(max_state_bytes)
    return operator_column([name, tree, params])


#: A cross-group operator's ``max_state_bytes`` at most: 64 MiB.
CROSS_STATE_BYTES_CEILING = 64 << 20


#: A cross rank's tie rules: Polars' ``rank(method=)`` less ``"random"``.
RANK_METHODS = ("average", "min", "max", "dense", "ordinal")

#: A cross quantile's readings between two values: Polars' ``interpolation``.
INTERPOLATIONS = ("nearest", "higher", "lower", "midpoint", "linear", "equiprobable")


def _cross(
    name: str,
    tree: list[Any],
    who: str,
    *,
    closed: Any,
    bias: Any,
    weight: Any,
    max_state_bytes: Any,
    quantile: Any = None,
    method: Any = "average",
    descending: Any = False,
    interpolation: Any = None,
    neighbors: Any = None,
    where: Any = None,
) -> pl.Expr:
    """A cross-group operator's node: its stamp (``closed``), a standard
    deviation's ``bias``, a rank's ``method`` and ``descending``, a
    quantile's level and ``interpolation``, its weight's tree and its cap
    where given. The node carries what the native reader writes back, each
    default resolved, so the two spell an operator's column alike."""
    if closed not in ("right", "left"):
        raise ValueError(
            f'{who}: closed must be "right" (the row\'s own stamp) or "left" (the stamp before '
            f"it), but it is {closed!r}. Set closed to one of those words."
        )
    params: dict[str, Any] = {"closed": closed}
    if name == "cross_ref":
        if where is None:
            raise TypeError(
                f"{who}: where is required: the predicate that picks the stamp's reference row, "
                f'such as pl.col("symbol") == "SPY". Give where.'
            )
        params["where"] = _input_tree(f"{who}: where", where)
    if neighbors is not None:
        if not isinstance(neighbors, str) or not neighbors:
            raise TypeError(
                f"{who}: neighbors must name a list column, but it is {neighbors!r}. Set "
                f"neighbors to the name of a column whose rows list the group keys each row "
                f"reads."
            )
        params["neighbors"] = neighbors
    if name == "cross_std":
        if not isinstance(bias, bool):
            raise TypeError(
                f"{who}: bias must be True or False, but it is of type {type(bias).__name__}. "
                f"Set bias to True or False."
            )
        params["bias"] = bias
    if name == "cross_rank":
        if method not in RANK_METHODS:
            raise ValueError(
                f"{who}: method must be one of {', '.join(map(repr, RANK_METHODS))}, but it is "
                f"{method!r}. Set method to one of those words."
            )
        if not isinstance(descending, bool):
            raise TypeError(
                f"{who}: descending must be True or False, but it is of type "
                f"{type(descending).__name__}. Set descending to True or False."
            )
        params["method"] = method
        params["descending"] = descending
    if name == "cross_quantile":
        wrong = isinstance(quantile, bool) or not isinstance(quantile, numbers.Real)
        if wrong or not 0.0 <= float(quantile) <= 1.0:
            msg = (
                f"{who}: quantile must be a number from 0 to 1, but it is {quantile!r}. Set "
                f"quantile within that range, such as 0.5 for the median."
            )
            raise WrongTypeError(msg) if wrong else ValueError(msg)
        params["quantile"] = float(quantile)
        if interpolation is None:
            interpolation = "nearest" if weight is None else "equiprobable"
        if interpolation not in INTERPOLATIONS:
            raise ValueError(
                f"{who}: interpolation must be one of "
                f"{', '.join(map(repr, INTERPOLATIONS))}, but it is {interpolation!r}. Set "
                f"interpolation to one of those words."
            )
        if weight is not None and interpolation != "equiprobable":
            raise ValueError(
                f"{who}: interpolation {interpolation!r} has no weighted form: with weight= the "
                f"quantile is the first value whose cumulative weight share reaches the level, "
                f'"equiprobable". Set interpolation to "equiprobable" or leave it out, or '
                f"remove weight."
            )
        params["interpolation"] = interpolation
    if weight is not None:
        params["weight"] = _input_tree(f"{who}: weight", weight)
    if max_state_bytes is not None:
        wrong = isinstance(max_state_bytes, bool) or not isinstance(
            max_state_bytes, numbers.Integral
        )
        if wrong or not 1 <= max_state_bytes <= CROSS_STATE_BYTES_CEILING:
            msg = (
                f"{who}: max_state_bytes must be a whole number from 1 to "
                f"{CROSS_STATE_BYTES_CEILING} (64 MiB), but it is {max_state_bytes!r}. Set it "
                f"within that range, or leave it out for 262144 (256 KB)."
            )
            raise WrongTypeError(msg) if wrong else ValueError(msg)
        params["max_state_bytes"] = int(max_state_bytes)
    return operator_column([name, tree, params])


def _lag(who: str, key: str, text: str) -> float | str | None:
    """The lag ``text`` after a leg's last ``@`` names: a number of clock
    units at least 0, or a duration; ``None`` where it is neither, so the
    whole key is a column's name."""
    word = text.strip()
    try:
        x = float(word)
    except ValueError:
        try:
            ns = parse_duration(word)
        except ValueError:
            return None
        if ns <= 0:
            raise ValueError(
                f"{who}: the leg {key!r} lags by {word}, which is not above 0. Give a lag "
                f"above 0, or name the column alone for the row itself."
            ) from None
        return word
    if not math.isfinite(x) or x < 0:
        raise ValueError(
            f"{who}: the leg {key!r} lags by {word}, but a lag is a finite number of clock "
            f"units of at least 0. Give such a number, or a duration on a temporal clock."
        )
    return x


def moments_operator(
    legs: Any,
    *,
    half_life: Any,
    kind: Any,
    bias: Any,
    window_size: Any,
    lag_tolerance: Any,
    partial: Any,
) -> pl.Expr:
    """The expression of :func:`~rheon.polars.ops.ewm_moments`: its legs
    and their lags read from the keys of ``legs``, each checked here so a
    wrong value names its leg or its parameter."""
    who = "rp.ewm_moments"
    if not isinstance(legs, dict) or not legs:
        raise TypeError(
            f"{who}: legs must be a dict from each leg (a column, or a column and its lag "
            f'written "x@1") to its power, such as {{"x": 3}}, but it is {legs!r}. Give the '
            f"legs as such a dict."
        )
    trees: list[list[Any]] = []
    powers: list[int] = []
    lags: list[float | str] = []
    for key, power in legs.items():
        if not isinstance(key, str) or not key:
            raise TypeError(
                f"{who}: a leg must be a column's name as text, or a column and its lag written "
                f'"x@1", but it is {key!r}. Name the leg as text.'
            )
        if isinstance(power, bool) or not isinstance(power, numbers.Integral):
            raise WrongTypeError(
                f"{who}: the leg {key!r} has the power {power!r}, but a power is a whole number "
                f"from 1. Give the leg a whole number."
            )
        if power < 1:
            raise ValueError(
                f"{who}: the leg {key!r} has the power {power}, but a power is a whole number "
                f"from 1. Give the leg a power of at least 1, or leave the leg out."
            )
        column: str = key
        lag: float | str = 0.0
        if "@" in key:
            head, _, tail = key.rpartition("@")
            parsed = _lag(who, key, tail)
            if parsed is not None and head:
                column, lag = head, parsed
        if column.startswith(PREFIX):
            raise _reserved(who, column)
        tree = ["col", column]
        if any(t == tree and lg == lag for t, lg in zip(trees, lags, strict=True)):
            raise ValueError(
                f"{who}: the leg {key!r} names a column and a lag that another leg names too. "
                f"Give each leg once, with its power."
            )
        trees.append(tree)
        powers.append(int(power))
        lags.append(lag)
    order = sum(powers)
    if kind not in MOMENT_KINDS:
        raise ValueError(
            f'{who}: kind must be "central", "raw", "standardized" or "cumulant", but it is '
            f"{kind!r}. Set kind to one of those words."
        )
    if kind == "cumulant" and len(trees) > 1:
        raise ValueError(
            f"{who}: a cumulant is of one leg, but legs has {len(trees)}. Ask for a central "
            f'cross-moment (kind="central") of several legs instead.'
        )
    if kind == "standardized" and order == 1:
        raise ValueError(
            f"{who}: a standardized moment of order 1 is 0 by definition. Give the legs an "
            f"order of at least 2."
        )
    if bias is None:
        bias = not (kind in ("central", "cumulant") and order == 2)
    elif not isinstance(bias, bool):
        raise TypeError(
            f"{who}: bias must be True, False or None, but it is of type "
            f"{type(bias).__name__}. Set bias to True or False, or leave it out."
        )
    if not bias and kind != "raw" and order > MAX_UNBIASED_ORDER:
        raise ValueError(
            f"{who}: bias=False asks for the unbiased form, which stops at order "
            f"{MAX_UNBIASED_ORDER}, but the legs' order is {order}. Set bias=True for this term."
        )
    if half_life is None:
        raise TypeError(
            f"{who}: half_life is required, but it is not given. Give half_life in clock units, "
            f"or inf for no decay."
        )
    params: dict[str, Any] = {"half_life": _span(who, "half_life", half_life, inf_ok=True)}
    if window_size is not None:
        params["window_size"] = _span(who, "window_size", window_size, inf_ok=False)
    elif params["half_life"] == "inf" or (
        isinstance(params["half_life"], str) and params["half_life"].lower() in _INFINITY
    ):
        raise RefusedValueError(
            f"{who}: half_life is inf and window_size is not given, so a value held from before "
            f"the first row would have no finite weight. Give window_size, or a finite half_life."
        )
    params["closed"] = "right"
    if partial is not None:
        if partial not in ("keep", "null", "drop"):
            raise ValueError(
                f'{who}: partial must be "keep", "null" or "drop", but it is {partial!r}. '
                f"Set partial to one of those words."
            )
        if window_size is None:
            raise ValueError(
                f"{who}: partial needs window_size, but window_size is not given. partial says "
                f"what a window cut short by a clock break (a step past gap_cap, or a session "
                f"change) gives, and without window_size no window is cut short. Give "
                f"window_size, or leave out partial."
            )
        params["partial"] = partial
    params["powers"] = powers
    params["lags"] = lags
    params["kind"] = kind
    params["bias"] = bias
    if lag_tolerance is not None:
        if isinstance(lag_tolerance, numbers.Real) and not isinstance(lag_tolerance, bool):
            if not (math.isfinite(float(lag_tolerance)) and float(lag_tolerance) >= 0):
                raise ValueError(
                    f"{who}: lag_tolerance must be a finite span of at least 0, but it is "
                    f"{lag_tolerance!r}. Give how far before t - lag the row a lag pairs with "
                    f"may sit."
                )
            params["lag_tolerance"] = float(lag_tolerance)
        else:
            params["lag_tolerance"] = _span(who, "lag_tolerance", lag_tolerance, inf_ok=False)
    return operator_column(["ewm_moments", ["legs", *trees], params])


#: A profile's ``max_state_bytes`` at most: 64 MiB.
PROFILE_STATE_BYTES_CEILING = 64 << 20

#: The most harmonics a profile takes.
MAX_HARMONICS = 64

#: What a profile reads of each phase.
PROFILE_STATS = ("mean", "var", "std", "quantile")


def profile(
    input: Any,
    *,
    period: Any,
    half_life: Any,
    buckets: Any = None,
    harmonics: Any = None,
    stat: Any = "mean",
    quantile: Any = None,
    window_size: Any = None,
    closed: Any = "right",
    offset: Any = None,
    weight: Any = None,
    min_samples: Any = 1,
    bias: Any = None,
    max_state_bytes: Any = None,
    who: str = "rp.ewm_profile",
) -> pl.Expr:
    """The expression of an ``ewm_profile`` node, each parameter checked
    here by name, written with the keys the native reader writes back, so
    the two spell its column alike."""
    tree = _input_tree(who, input)
    if half_life is None:
        raise TypeError(
            f"{who}: half_life is required, but it is not given. Give half_life in clock units, "
            f"or inf for no decay."
        )
    if period is None:
        raise TypeError(
            f'{who}: period is required, but it is not given. Give period, such as "1d" for the '
            f"time of day."
        )
    params: dict[str, Any] = {
        "half_life": _span(who, "half_life", half_life, inf_ok=True),
        "period": _span(who, "period", period, inf_ok=False),
    }
    if offset is not None:
        params["offset"] = _offset(who, offset)
    if window_size is not None:
        params["window_size"] = _span(who, "window_size", window_size, inf_ok=False)
    if closed not in ("right", "left", "both", "none"):
        raise ValueError(
            f'{who}: closed must be "right", "left", "both" or "none", but it is {closed!r}. '
            f"Set closed to one of those words."
        )
    params["closed"] = closed
    if (buckets is None) == (harmonics is None):
        raise TypeError(
            f"{who}: give buckets or harmonics, one of them: how many equal phases the period "
            f"is cut into, or how many Fourier terms fit it, but "
            + ("both are given." if buckets is not None else "neither is given.")
            + " Set one, such as buckets=24 for the hours of a day."
        )
    if buckets is not None:
        wrong = isinstance(buckets, bool) or not isinstance(buckets, numbers.Integral)
        if wrong or not 1 <= buckets <= 2**32 - 1:
            msg = (
                f"{who}: buckets must be a whole number from 1 to {2**32 - 1}, but it is "
                f"{buckets!r}. Set buckets to the number of phases of the period, such as 24 "
                f"for the hours of a day."
            )
            raise WrongTypeError(msg) if wrong else ValueError(msg)
        params["buckets"] = int(buckets)
    else:
        wrong = isinstance(harmonics, bool) or not isinstance(harmonics, numbers.Integral)
        if wrong or not 1 <= harmonics <= MAX_HARMONICS:
            msg = (
                f"{who}: harmonics must be a whole number from 1 to {MAX_HARMONICS}, but it is "
                f"{harmonics!r}. Set harmonics to the Fourier terms the profile takes, such as "
                f"3 for the daily shape of an hourly series."
            )
            raise WrongTypeError(msg) if wrong else ValueError(msg)
        if stat == "quantile" or bias is not None:
            raise ValueError(
                f'{who}: harmonics takes stat "mean", "var" or "std" and no bias: a Fourier '
                f"fit's variance is the fit of x² less the square of the fit of x, and a "
                f"quantile has no one-pass fit. Leave out bias or quantile, or give buckets."
            )
        params["harmonics"] = int(harmonics)
    if stat not in PROFILE_STATS:
        raise ValueError(
            f'{who}: stat must be "mean", "var", "std" or "quantile", but it is {stat!r}. Set '
            f"stat to one of those words."
        )
    params["stat"] = stat
    if stat == "quantile":
        if isinstance(quantile, bool) or not isinstance(quantile, numbers.Real):
            raise TypeError(
                f'{who}: quantile must be a number, as stat="quantile" needs one, but it is of '
                f"type {type(quantile).__name__}. Set quantile to a number above 0 and below 1."
            )
        if not 0.0 < float(quantile) < 1.0:
            raise ValueError(
                f"{who}: quantile must be above 0 and below 1, but it is {quantile!r}. Set "
                f"quantile within that range, such as 0.5 for the median."
            )
        params["quantile"] = float(quantile)
    elif quantile is not None:
        raise TypeError(
            f'{who}: quantile is given, but only stat="quantile" takes it. Leave out quantile.'
        )
    if stat in ("var", "std") and harmonics is None:
        if bias is None:
            bias = False
        if not isinstance(bias, bool):
            raise TypeError(
                f"{who}: bias must be True or False, but it is of type {type(bias).__name__}. "
                f"Set bias to True or False."
            )
        params["bias"] = bias
    elif bias is not None:
        raise TypeError(
            f'{who}: bias is given, but only stat="var" or "std" takes it. Leave out bias.'
        )
    wrong = isinstance(min_samples, bool) or not isinstance(min_samples, numbers.Integral)
    if wrong or not 1 <= min_samples <= MAX_MIN_SAMPLES:
        msg = (
            f"{who}: min_samples must be an integer of at least 1 and at most "
            f"{MAX_MIN_SAMPLES}, but it is {min_samples!r}. Set min_samples to the rows a phase "
            f"needs before it is read."
        )
        raise WrongTypeError(msg) if wrong else ValueError(msg)
    params["min_samples"] = int(min_samples)
    if weight is not None:
        params["weight"] = _input_tree(f"{who}: weight", weight)
    if max_state_bytes is not None:
        wrong = isinstance(max_state_bytes, bool) or not isinstance(
            max_state_bytes, numbers.Integral
        )
        if wrong or not 1 <= max_state_bytes <= PROFILE_STATE_BYTES_CEILING:
            msg = (
                f"{who}: max_state_bytes must be a whole number from 1 to "
                f"{PROFILE_STATE_BYTES_CEILING} (64 MiB), but it is {max_state_bytes!r}. Set it "
                f"within that range, or leave it out for 262144 (256 KB)."
            )
            raise WrongTypeError(msg) if wrong else ValueError(msg)
        params["max_state_bytes"] = int(max_state_bytes)
    return operator_column(["ewm_profile", tree, params])


def _offset(who: str, value: Any) -> float | str:
    """A profile's offset: any finite span, 0 or negative included, where
    every other span is above 0."""
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        if not math.isfinite(value):
            raise ValueError(
                f"{who}: offset must be finite, but it is {value!r}. Set offset to where the "
                f'period starts after its default start, such as "9h30m".'
            )
        return float(value)
    if isinstance(value, (str, timedelta)) and not isinstance(value, bool):
        if isinstance(value, timedelta) and value <= timedelta(0):
            raise ValueError(
                f"{who}: offset must be above 0 as a duration, but it is {value!r}. Give it as "
                f'a positive duration within the period, such as "9h30m".'
            )
        return _span(who, "offset", value, inf_ok=False)
    raise TypeError(
        f"{who}: offset must be a number or a duration, but it is of type "
        f'{type(value).__name__}. Give it in clock units, or as a duration such as "9h30m".'
    )


def _row_input(who: str, key: str, value: Any) -> list[Any]:
    """A row operator's input: a column name, or an element-wise
    expression of the row holding no operator at all."""
    if isinstance(value, str):
        if value.startswith(PREFIX):
            raise _reserved(who, value)
        return ["col", value]
    if isinstance(value, pl.Expr):
        tree = to_tree(value, who)
        inner = operators(tree)
        if inner:
            raise ValueError(
                f"{who}: {key} holds the operator {inner[0][0]}, but it may hold only columns "
                f'and Polars expressions of the same row (a column computation, such as pl.col("x")'
                f" + 1). Add the inner operator's result as a column with one "
                f"rp.stream.with_windows call first, and pass that column to a second."
            )
        return tree
    raise TypeError(
        f"{who}: {key} must be a column name or a pl.Expr, but it is of type "
        f'{type(value).__name__}. Pass a column name or an expression, such as pl.col("x").'
    )


def row_operator(
    name: str,
    input: Any,
    *,
    abs_tol: Any = 0.0,
    event: Any = None,
    half_life: Any = None,
    stat: Any = "mean",
    bias: Any = None,
) -> pl.Expr:
    """The expression of the row operator ``name``: a run length
    over ``input`` with ``abs_tol``, a clock lag or drift of ``input``, the
    arrival time, against ``event``, or a clock step's ``stat`` of the
    steps of ``input``, decayed across ``half_life``. Checked here, so a
    wrong value names its parameter."""
    who = f"rp.{name}"
    if name == "clock_step":
        return _clock_step(who, input, half_life=half_life, stat=stat, bias=bias)
    if name == "run_length":
        tree = _row_input(who, "input", input)
        if isinstance(abs_tol, bool) or not isinstance(abs_tol, numbers.Real):
            raise TypeError(
                f"{who}: abs_tol must be a number, but it is of type {type(abs_tol).__name__}. Set abs_tol "
                f"to 0 for values exactly equal, or to the largest difference two adjacent rows "
                f"of one run may have."
            )
        t = float(abs_tol)
        if not (math.isfinite(t) and t >= 0.0):
            raise ValueError(
                f"{who}: abs_tol must be finite and at least 0, but it is {abs_tol!r}. Set abs_tol to 0 for "
                f"values exactly equal, or to the largest difference two adjacent rows of one "
                f"run may have."
            )
        # A run under the default tolerance carries no parameters, so one
        # run asked for twice is one column whichever way it was written.
        return operator_column(
            ["run_length", tree] if t == 0.0 else ["run_length", tree, {"abs_tol": t}]
        )
    assert name in CLOCKS, name
    arrival = _row_input(who, "arrival", input)
    if event is None:
        raise TypeError(
            f"{who}: event is required, but it is not given. Give event as the column of each "
            f"row's event time."
        )
    happened = _row_input(who, "event", event)
    if half_life is None:
        raise TypeError(
            f"{who}: half_life is required, but it is not given. Give half_life in clock units, "
            f"or inf for no decay."
        )
    params = {"event": happened, "half_life": _span(who, "half_life", half_life, inf_ok=True)}
    return operator_column([name, arrival, params])


def _clock_step(who: str, input: Any, *, half_life: Any, stat: Any, bias: Any) -> pl.Expr:
    """A clock step's node: its time, half-life and statistic,
    and a spread's ``bias``, resolved to ``False`` where not given, as the
    native reader writes it back."""
    tree = _row_input(who, "time", input)
    if half_life is None:
        raise TypeError(
            f"{who}: half_life is required, but it is not given. Give half_life in clock units, "
            f"or inf for no decay."
        )
    if stat not in STEP_STATS:
        raise ValueError(
            f'{who}: stat must be "mean", "var", "std", "zeros" or "backs", but it is {stat!r}. '
            f"Set stat to one of those words."
        )
    params: dict[str, Any] = {
        "half_life": _span(who, "half_life", half_life, inf_ok=True),
        "stat": stat,
    }
    if stat in ("var", "std"):
        if bias is None:
            bias = False
        if not isinstance(bias, bool):
            raise TypeError(
                f"{who}: bias must be True or False, but it is of type {type(bias).__name__}. "
                f"Set bias to True or False."
            )
        params["bias"] = bias
    elif bias is not None:
        raise TypeError(
            f'{who}: bias is given, but only stat="var" or "std" takes it. Leave out bias.'
        )
    return operator_column(["clock_step", tree, params])


def holds_invert(tree: Any) -> bool:
    """Whether ``tree`` holds an invert's node anywhere."""
    if not isinstance(tree, list) or not tree:
        return False
    if tree[0] == "invert":
        return True
    if tree[0] in ("col", "lit"):
        return False
    return any(holds_invert(arg) for arg in tree[1:])


#: The nodes the solve evaluates a row at a time, in each part of an invert:
#: the leaf of the unknown, columns, numbers, arithmetic, ``**``, negation,
#: ``abs``, ``exp``, ``sqrt``, ``log`` and a cast to Float64.
INVERT_NODES = (
    "unknown",
    "col",
    "lit",
    "+",
    "-",
    "*",
    "/",
    "**",
    "log",
    "neg",
    "abs",
    "exp",
    "sqrt",
    "cast",
)

#: The default absolute tolerance of an invert, in the unknown's units.
INVERT_XTOL = 2e-12

#: The least relative tolerance of an invert: four machine epsilons.
INVERT_MIN_RTOL = 4 * 2.0**-52


def _invert_nodes(who: str, part: str, tree: Any) -> None:
    """Refuse, naming ``part``, a node of ``tree`` that the solve does not
    evaluate a row at a time."""
    if not isinstance(tree, list) or not tree:
        raise FormulaError(f"{who}: its {part} is not an expression it can read. Rewrite it.")
    head = tree[0]
    if head in OPERATORS or head == "invert":
        what = "another invert" if head == "invert" else f"the window operator {head}"
        raise FormulaError(
            f"{who}: its {part} holds {what}, but the solve evaluates its equation a row at a "
            f"time and reads the row alone. Add it as a column with rp.stream.with_windows "
            f"first, and read that column."
        )
    if head not in INVERT_NODES:
        raise FormulaError(
            f"{who}: its {part} holds {head!r}, which the solve does not evaluate. It can hold "
            f"columns, numbers, + - * /, ** (pow), negation, abs, exp, log and sqrt, and a cast "
            f"to Float64. Compute anything else as a column first and read that column."
        )
    if head == "lit":
        value = tree[1] if len(tree) == 2 else None
        if (
            isinstance(value, bool)
            or not isinstance(value, numbers.Real)
            or not math.isfinite(value)
        ):
            raise FormulaError(
                f"{who}: its {part} holds the literal {value!r}, but the solve reads only finite "
                f"numbers. Write the literal as a number."
            )
        return
    if head == "cast":
        if len(tree) != 3 or tree[2] != "Float64":
            raise FormulaError(
                f"{who}: its {part} holds a cast to {tree[2] if len(tree) > 2 else None!r}, but "
                f"the solve reads every value as a Float64, so it casts to Float64 alone, and "
                f"strictly. Cast to pl.Float64, or leave the cast out; for a typed number, give "
                f"a plain float."
            )
        _invert_nodes(who, part, tree[1])
        return
    if head == "col":
        return
    for arg in tree[1:]:
        _invert_nodes(who, part, arg)


def _holds_unknown(tree: Any) -> bool:
    if not isinstance(tree, list) or not tree:
        return False
    if tree[0] == "unknown":
        return True
    if tree[0] in ("col", "lit"):
        return False
    return any(_holds_unknown(arg) for arg in tree[1:])


def _invert_value(who: str, key: str, value: Any) -> list[Any]:
    """``equals``, ``lo`` or ``hi`` as a tree: a column name, a finite
    number, or an expression of the row without the unknown."""
    if isinstance(value, str):
        if value.startswith(PREFIX):
            raise _reserved(who, value)
        return ["col", value]
    if isinstance(value, bool) or not isinstance(value, (numbers.Real, pl.Expr)):
        raise TypeError(
            f"{who}: {key} must be a column name, a number or a pl.Expr of the row, but it is of "
            f'type {type(value).__name__}. Give {key} such as "price", 1.5 or pl.col("price").'
        )
    if isinstance(value, numbers.Real):
        if not math.isfinite(float(value)):
            raise ValueError(
                f"{who}: {key} must be finite, but it is {value!r}. Give {key} a finite number, "
                f"or a column that holds one on each row."
            )
        return ["lit", float(value)]
    return to_tree(value, f"{who}: {key}")


def invert_node(
    f: Any,
    equals: Any,
    lo: Any,
    hi: Any,
    *,
    increasing: Any,
    xtol: Any,
    rtol: Any,
    detail: Any,
) -> pl.Expr:
    """The expression of :func:`~rheon.polars.ops.invert`: its function read
    with the unknown as a leaf, its target and bracket as trees, its
    parameters checked here so a wrong one is named, and the node written
    with every parameter, as the native reader writes it back."""
    who = "rp.ops.invert"
    if isinstance(f, pl.Expr):
        expr = f
    elif callable(f):
        expr = f(pl.col(UNKNOWN_COLUMN))
        if not isinstance(expr, pl.Expr):
            raise TypeError(
                f"{who}: f must return a pl.Expr of the unknown, but it returns a value of type "
                f"{type(expr).__name__}. Write f with Polars' operators on its argument, and "
                f"return the expression."
            )
    else:
        raise TypeError(
            f"{who}: f must be a pl.Expr in rp.ops.unknown(), or a function of the unknown that "
            f"returns one, but it is of type {type(f).__name__}. Give the equation's left side "
            f"as such an expression."
        )
    trees = {"function": to_tree(expr, f"{who}: f", unknown=True)}
    if not _holds_unknown(trees["function"]):
        raise ValueError(
            f"{who}: f does not hold the unknown, so there is nothing to solve for. Write f in "
            f"terms of rp.ops.unknown(), or of its argument where f is a function."
        )
    for key, value in (("equals", equals), ("lo", lo), ("hi", hi)):
        trees[key] = _invert_value(who, key, value)
    for part, tree in trees.items():
        _invert_nodes(who, part, tree)
    if not any(columns(tree) for tree in trees.values()):
        raise ValueError(
            f"{who}: the equation reads no column, so it has the same root on every row. Read a "
            f"column in f, equals, lo or hi, or compute the root once."
        )
    if trees["lo"][0] == "lit" and trees["hi"][0] == "lit" and not trees["lo"][1] < trees["hi"][1]:
        raise ValueError(
            f"{who}: the bracket is [{trees['lo'][1]!r}, {trees['hi'][1]!r}], but lo must be "
            f"below hi, so it holds no interval to search. Give lo below hi."
        )
    for key, value in (("increasing", increasing), ("detail", detail)):
        if not isinstance(value, bool):
            raise TypeError(
                f"{who}: {key} must be True or False, but it is {value!r}. Set {key} to True or "
                f"False."
            )
    tol: dict[str, float] = {}
    for key, value, least in (("xtol", xtol, 0.0), ("rtol", rtol, INVERT_MIN_RTOL)):
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise TypeError(
                f"{who}: {key} must be a number, but it is of type {type(value).__name__}. Set "
                f"{key} to a number."
            )
        x = float(value)
        ok = math.isfinite(x) and (x > least if key == "xtol" else x >= least)
        if not ok:
            need = "above 0" if key == "xtol" else f"at least {least!r} (four machine epsilons)"
            why = (
                "at 0 a bracket around a root at 0 would never narrow to it"
                if key == "xtol"
                else "below that a bracket around a large root cannot narrow as asked"
            )
            raise ValueError(
                f"{who}: {key} must be finite and {need}, but it is {value!r}: {why}. Set {key} "
                f"within that range."
            )
        tol[key] = x
    params = {"increasing": increasing, "xtol": tol["xtol"], "rtol": tol["rtol"], "detail": detail}
    node = [
        "invert",
        trees["function"],
        trees["equals"],
        trees["lo"],
        trees["hi"],
        params,
    ]
    return operator_column(node)
