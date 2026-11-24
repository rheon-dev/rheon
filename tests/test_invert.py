"""``rp.ops.invert``: the root of an equation of the row, through
``rp.stream.with_windows``. What a row gives (a root, or null with its
reason under ``detail``), the struct's fields, the inputs it reads as
missing, and what the call refuses when it is made or the query is built.
The agreement with scipy's ``brentq`` is ``test_second_opinion_invert.py``'s.
"""

from __future__ import annotations

import json
import math

import polars as pl
import pytest

from rheon import polars as rp
from rheon.polars import _formula

TIER = "essential"

LEVEL = {"level": rp.ewm_mean("y", half_life=2.0)}


def cubic(x: pl.Expr) -> pl.Expr:
    """``x^3 + a x``: increasing in ``x`` wherever ``a >= 0``."""
    return x**3 + pl.col("a") * x


def run(frame: pl.DataFrame, **exprs: pl.Expr) -> pl.DataFrame:
    """The frame through ``with_windows`` with ``exprs`` beside a window
    operator, which a run needs."""
    return rp.stream.with_windows(frame, **exprs, **LEVEL)


def detail(frame: pl.DataFrame, expr: pl.Expr) -> pl.DataFrame:
    return run(frame, d=expr)["d"].struct.unnest()


def test_a_root_a_row_and_null_where_it_is_withheld():
    frame = pl.DataFrame({"a": [1.0, 2.0, 0.0, 1.0], "y": [2.0, 3.0, 8.0, 500.0]})
    got = run(frame, x=rp.ops.invert(cubic, "y", -5.0, 5.0, increasing=True))
    assert got.schema["x"] == pl.Float64
    x = got["x"].to_list()
    assert x[3] is None, "500 is past f(5) = 130"
    for xi, ai, yi in zip(x[:3], frame["a"][:3], frame["y"][:3], strict=True):
        assert math.isclose(xi**3 + ai * xi, yi, rel_tol=1e-12), (xi, ai, yi)
    assert [round(v, 9) for v in x[:3]] == [1.0, 1.0, 2.0]


def test_the_detail_struct_names_its_fields_and_types():
    frame = pl.DataFrame({"a": [1.0, 1.0, 1.0], "y": [2.0, -500.0, 500.0]})
    d = run(frame, d=rp.ops.invert(cubic, "y", -5.0, 5.0, increasing=True, detail=True))
    assert d.schema["d"] == pl.Struct(
        {
            "root": pl.Float64,
            "lo": pl.Float64,
            "hi": pl.Float64,
            "evaluations": pl.UInt32,
            "bound": pl.UInt32,
            "withheld_reason": pl.String,
        }
    )
    rows = d["d"].struct.unnest()
    assert rows["withheld_reason"].to_list() == [None, "below_range", "above_range"]
    assert rows["root"][0] is not None and rows["root"][1:].to_list() == [None, None]
    assert rows["lo"][0] <= rows["root"][0] <= rows["hi"][0]
    assert rows["hi"][0] - rows["lo"][0] <= 4 * _formula.INVERT_XTOL + 1e-15
    assert (rows["evaluations"] <= rows["bound"]).all()
    # The bound follows from the bracket and the tolerance alone:
    # ceil(log2(10 / 4e-12)) + 1 steps, the two ends and two more.
    assert set(rows["bound"].to_list()) == {math.ceil(math.log2(10 / 4e-12)) + 1 + 4}


@pytest.mark.parametrize("bad", [None, math.nan, math.inf, -math.inf, 1e101])
@pytest.mark.parametrize("where", ["function", "equals", "lo", "hi"])
def test_a_null_nan_or_unusable_input_is_input_missing(bad, where):
    """A value the library cannot use -- null, NaN, infinite, past 1e100 in
    magnitude -- in a column the function reads, the target or a bracket
    end, is withheld as ``input_missing``, and the other rows are solved."""
    cols = {"a": [1.0, 1.0], "y": [2.0, 2.0], "lo": [-5.0, -5.0], "hi": [5.0, 5.0]}
    col = {"function": "a", "equals": "y", "lo": "lo", "hi": "hi"}[where]
    cols[col][1] = bad
    frame = pl.DataFrame(cols, schema=dict.fromkeys(cols, pl.Float64))
    rows = detail(frame, rp.ops.invert(cubic, "y", "lo", "hi", increasing=True, detail=True))
    assert rows["withheld_reason"].to_list() == [None, "input_missing"]
    assert rows["root"][1] is None and rows["lo"][1] is None and rows["hi"][1] is None
    assert rows["evaluations"][1] == 0
    assert round(rows["root"][0], 9) == 1.0


def test_each_reason_a_row_can_give():
    """Every reason but ``not_monotone``, which a step finds only inside the
    bracket and has its Rust tests, on rows of one frame."""
    frame = pl.DataFrame(
        {
            "a": [1.0, 1.0, 1.0, 1.0, 1.0],
            "y": [2.0, 2.0, 2.0, 2.0, 2.0],
            "lo": [-5.0, 5.0, -5.0, -5.0, -5.0],
            "hi": [5.0, -5.0, 5.0, 5.0, 5.0],
            "s": [1.0, 1.0, -1.0, 1.0, 1.0],
        }
    )
    # Row 1's bracket is upside down, and s flips the function on row 2,
    # against the declared direction.
    expr = rp.ops.invert(
        lambda x: pl.col("s") * cubic(x),
        "y",
        "lo",
        "hi",
        increasing=True,
        detail=True,
    )
    rows = detail(frame, expr)
    assert rows["withheld_reason"].to_list()[:3] == [None, "bracket_invalid", "wrong_direction"]
    domain = rp.ops.invert(
        lambda x: (x + 6.0).log() + x, "y", "lo", "hi", increasing=True, detail=True
    )
    lows = frame.with_columns(lo=pl.lit(-7.0))
    assert detail(lows, domain)["withheld_reason"][0] == "not_finite"
    # A function flat to the last bit near its root: its sign there is
    # rounding, so the root is not fixed to the tolerance.
    flat = rp.ops.invert(
        lambda x: pl.col("a") + (x - 1.0) ** 21 * 1e-300,
        "a",
        "lo",
        "hi",
        increasing=True,
        detail=True,
    )
    assert detail(frame.with_columns(lo=pl.lit(-5.0), hi=pl.lit(5.0)), flat)["withheld_reason"][0] == (
        "not_determined"
    )


def test_a_decreasing_function_is_solved_where_declared():
    frame = pl.DataFrame({"a": [1.0], "y": [-2.0]})
    x = run(frame, x=rp.ops.invert(lambda x: -cubic(x), "y", -5.0, 5.0, increasing=False))["x"]
    assert round(x[0], 9) == 1.0
    wrong = run(
        frame, d=rp.ops.invert(lambda x: -cubic(x), "y", -5.0, 5.0, increasing=True, detail=True)
    )
    assert wrong["d"].struct.field("withheld_reason")[0] == "wrong_direction"


def test_a_bracket_of_numbers_not_in_order_is_refused_when_the_call_is_made():
    for lo, hi in ((5.0, -5.0), (1.0, 1.0)):
        with pytest.raises(ValueError, match="lo must be below hi"):
            rp.ops.invert(cubic, "y", lo, hi, increasing=True)
    for bad in (math.nan, math.inf):
        with pytest.raises(ValueError, match="lo must be finite"):
            rp.ops.invert(cubic, "y", bad, 5.0, increasing=True)
    with pytest.raises(TypeError, match="lo must be a column name, a number or a pl.Expr"):
        rp.ops.invert(cubic, "y", True, 5.0, increasing=True)


def test_the_call_refuses_what_the_solve_cannot_read():
    with pytest.raises(TypeError, match="f must be a pl.Expr in rp.ops.unknown"):
        rp.ops.invert(2.0, "y", 0.0, 1.0, increasing=True)
    with pytest.raises(TypeError, match="f must return a pl.Expr"):
        rp.ops.invert(lambda x: 1.0, "y", 0.0, 1.0, increasing=True)
    for no_unknown in (pl.col("a"), lambda x: pl.col("a")):
        with pytest.raises(ValueError, match="f does not hold the unknown"):
            rp.ops.invert(no_unknown, "y", 0.0, 1.0, increasing=True)
    with pytest.raises(ValueError, match="holds the window operator ewm_mean"):
        rp.ops.invert(lambda x: x - rp.ewm_mean("a", half_life=1.0), "y", 0.0, 1.0, increasing=True)
    with pytest.raises(ValueError, match="'when', which the solve does not evaluate"):
        rp.ops.invert(
            lambda x: pl.when(x > 0).then(x).otherwise(-x), "y", 0.0, 1.0, increasing=True
        )
    with pytest.raises(ValueError, match="cast to 'Int64'"):
        rp.ops.invert(lambda x: x.cast(pl.Int64), "y", 0.0, 1.0, increasing=True)
    held: list[pl.Expr] = []

    def keep(x: pl.Expr) -> pl.Expr:
        held.append(x)
        return x

    rp.ops.invert(keep, "y", 0.0, 1.0, increasing=True)
    for unknown in (held[0], rp.ops.unknown() * 2):
        with pytest.raises(ValueError, match="outside the function f"):
            rp.ops.invert(cubic, unknown, 0.0, 1.0, increasing=True)
    frame = pl.DataFrame({"a": [1.0], "y": [2.0]})
    with pytest.raises(ValueError, match="outside the function f"):
        run(frame, z=rp.ops.unknown() + pl.col("a"))
    with pytest.raises(ValueError, match="reads no column"):
        rp.ops.invert(lambda x: x**3, 2.0, 0.0, 5.0, increasing=True)
    with pytest.raises(TypeError, match="increasing"):
        rp.ops.invert(cubic, "y", 0.0, 1.0)
    with pytest.raises(TypeError, match="increasing must be True or False"):
        rp.ops.invert(cubic, "y", 0.0, 1.0, increasing=1)
    with pytest.raises(TypeError, match="detail must be True or False"):
        rp.ops.invert(cubic, "y", 0.0, 1.0, increasing=True, detail="yes")
    for xtol in (0.0, -1.0, math.inf, math.nan):
        with pytest.raises(ValueError, match="xtol must be finite and above 0"):
            rp.ops.invert(cubic, "y", 0.0, 1.0, increasing=True, xtol=xtol)
    with pytest.raises(ValueError, match="rtol must be finite and at least"):
        rp.ops.invert(cubic, "y", 0.0, 1.0, increasing=True, rtol=1e-16)
    with pytest.raises(TypeError, match="xtol must be a number"):
        rp.ops.invert(cubic, "y", 0.0, 1.0, increasing=True, xtol="small")


def test_the_node_carries_every_parameter_and_reads_back():
    expr = rp.ops.invert(cubic, "y", -5.0, pl.col("hi") * 2, increasing=True)
    (column,) = expr.meta.root_names()
    node = json.loads(column[len(_formula.PREFIX) :])
    assert node == [
        "invert",
        ["+", ["**", ["unknown"], ["lit", 3]], ["*", ["col", "a"], ["unknown"]]],
        ["col", "y"],
        ["lit", -5.0],
        ["*", ["col", "hi"], ["lit", 2]],
        {"increasing": True, "xtol": 2e-12, "rtol": 4 * 2.0**-52, "detail": False},
    ]
    assert _formula.columns(node) == ["a", "y", "hi"]
    tree = _formula.to_tree(rp.ewm_mean(pl.col("y"), half_life=1.0) - expr)
    assert _formula.holds_invert(tree) and _formula.to_tree(_formula.from_tree(tree)) == tree


def test_a_run_of_inverts_alone_is_refused_and_a_target_does_not_take_one():
    frame = pl.DataFrame({"a": [1.0], "y": [2.0]})
    with pytest.raises(ValueError, match="no window to compute"):
        rp.stream.with_windows(frame, x=rp.ops.invert(cubic, "y", -5.0, 5.0, increasing=True))
    with pytest.raises(ValueError, match="holds no window operator"):
        rp.stream.with_windows(frame, x=pl.col("y") + 1, **LEVEL)
    ahead = rp.ewm_mean("y", half_life=1.0, window_size=2.0, reverse=True)
    target = (ahead - rp.ops.invert(cubic, "y", -5.0, 5.0, increasing=True)).alias("t")
    with pytest.raises(ValueError, match="holds rp.ops.invert, which a formula target"):
        rp.spec.ewridge(
            "m",
            features=["a"],
            targets=[target],
            clock="t",
            gap_cap=100.0,
            half_life=10.0,
            embargo=3.0,
        )


def test_the_roots_feed_an_operator_and_sit_under_arithmetic():
    frame = pl.DataFrame({"a": [1.0, 2.0, 1.0, 2.0], "y": [2.0, 3.0, 2.0, 3.0]})
    root = rp.ops.invert(cubic, "y", -5.0, 5.0, increasing=True)
    got = rp.stream.with_windows(
        frame, x=root, smooth=rp.ewm_mean(root, half_life=1.0), gap=root - pl.col("a")
    )
    want = rp.stream.with_windows(got.select("a", "y", "x"), smooth=rp.ewm_mean("x", half_life=1.0))
    assert got["smooth"].to_list() == want["smooth"].to_list()
    assert (got["gap"] - (got["x"] - got["a"])).abs().max() == 0.0


def test_the_roots_are_the_same_bits_however_the_stream_is_chunked():
    n = 500
    frame = pl.DataFrame(
        {
            "a": [0.5 + (i % 7) / 3 for i in range(n)],
            "y": [((i * 37) % 101) - 50.0 if i % 13 else None for i in range(n)],
        }
    )
    exprs = {"x": rp.ops.invert(cubic, "y", -5.0, 5.0, increasing=True)}
    one = rp.stream.with_windows(frame, **exprs, **LEVEL, chunk_size=n)
    for size in (1, 7, 333):
        other = rp.stream.with_windows(frame, **exprs, **LEVEL, chunk_size=size)
        assert one["x"].to_list() == other["x"].to_list(), size


def test_an_expression_in_the_unknown_is_the_same_node_as_a_function_of_it():
    """``f`` given as an expression in ``rp.ops.unknown()`` writes the node
    a function of the unknown writes, so the two are one column; and the
    invert is ``rp.ops``' alone, with no name at the root."""
    x = rp.ops.unknown()
    as_expr = rp.ops.invert(x**3 + pl.col("a") * x, "y", -5.0, 5.0, increasing=True)
    as_function = rp.ops.invert(cubic, "y", -5.0, 5.0, increasing=True)
    assert as_expr.meta.root_names() == as_function.meta.root_names()
    frame = pl.DataFrame({"a": [1.0, 2.0], "y": [2.0, 3.0]})
    got = run(frame, x=as_expr)
    assert [round(v, 9) for v in got["x"]] == [1.0, 1.0]
    assert "invert" in rp.ops.__all__ and "unknown" in rp.ops.__all__
    assert not hasattr(rp, "invert") and not hasattr(rp, "unknown")
    assert "invert" not in rp.__all__ and "unknown" not in rp.__all__
