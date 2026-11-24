"""``emit_target_weight``: ``weight_<target>`` is the weight that target's
``min_weight`` gate reads (hard rule 8; docs/PLAN.md task 568).

A row whose target is null teaches that target nothing but adds to the
shared ``weight_sum``, so a row can show ``weight_sum`` above ``min_weight``
together with ``below_min_weight``. The field is the number below the floor.
Held here for every regression model through the bank's Python surface;
``crates/rheon-polars/tests/target_weight.rs`` holds it against the weight's
recursion, the models that keep no weight per target, chunk invariance and a
save and load, where ``cargo mutants`` sees it.
"""

from __future__ import annotations

import math

import polars as pl
import pytest

from rheon import polars as rp
from test_semantics_all_models import BINARY, IDS, SWEEP, build, frame

TIER = "essential"

FLOOR = 20.0


def _out(model: str, extra: dict, df: pl.DataFrame, **kw: object) -> pl.DataFrame:
    if model == "ewridge":
        # Its noise gate is on by default; off, `min_weight` is the only gate.
        kw.setdefault("max_error_inflation", math.inf)
    spec = build(model, extra, min_weight=FLOOR, emit_target_weight=True, **kw)
    return rp.ModelBank([spec]).fit_predict(df).select("m").unnest("m")


def _gappy(model: str) -> pl.DataFrame:
    """200 rows with the target null on rows 10-39 and one row in five."""
    df = frame(n=200, binary=model in BINARY)
    y = [None if 10 <= i < 40 or i % 5 == 2 else v for i, v in enumerate(df["y0"].to_list())]
    return df.with_columns(y0=pl.Series(y, dtype=pl.Float64))


@pytest.mark.parametrize(("model", "extra"), SWEEP, ids=IDS)
def test_the_field_is_the_weight_the_gate_reads(model, extra):
    out = _out(model, extra, _gappy(model))
    own = out["weight_y0"].to_list()
    shared = out["weight_sum"].to_list()
    reason = out["withheld_reason"].cast(pl.String).to_list()
    pred = next(c for c in out.columns if c.startswith("pred_y0"))
    hidden = 0
    for i, (w, s) in enumerate(zip(own, shared, strict=True)):
        assert w is not None and s is not None, (model, i)
        # A target's own weight is the rows it was present on: never more
        # than the rows the model saw.
        assert w <= s * (1 + 1e-12), (model, i, w, s)
        # The gate reads it: withheld for the floor exactly where it is short.
        assert (reason[i] == "below_min_weight") == (w < FLOOR), (model, i, w, s, reason[i])
        if w < FLOOR:
            assert out[pred][i] is None, (model, i)
            hidden += s >= FLOOR
    # The case the field is for: the shared weight past the floor while the
    # target's own is short of it.
    assert hidden > 0, model


def test_the_field_is_named_and_indexed_per_target_and_instance():
    spec = rp.spec.ewridge(
        "m", targets=["y", "z"], features=["x0"], half_life=[50.0, 200.0], emit_target_weight=True
    )
    index = rp.spec.output_index(spec).filter(pl.col("kind") == "weight")
    assert index["field"].to_list() == [
        "weight_y@h50",
        "weight_z@h50",
        "weight_y@h200",
        "weight_z@h200",
    ]
    assert index["target"].to_list() == ["y", "z", "y", "z"]
    assert index["instance"].to_list() == ["@h50", "@h50", "@h200", "@h200"]
    assert set(index["dtype"].to_list()) == {"f64"}
    # Off, the spec is the dict it always was.
    plain = rp.spec.ewridge("m", targets=["y"], features=["x0"], half_life=50.0)
    assert "emit_target_weight" not in plain


def test_a_value_that_is_not_a_bool_is_refused():
    for bad in (1, 0, "yes"):
        with pytest.raises(TypeError, match="emit_target_weight must be a bool"):
            rp.spec.ewridge(
                "m", targets=["y"], features=["x0"], half_life=50.0, emit_target_weight=bad
            )


@pytest.mark.parametrize(
    "builder",
    [
        lambda: rp.spec.ew_cov("m", features=["x0", "x1"], emit_target_weight=True),
        lambda: rp.spec.marginal("m", targets=["y"], features=["x0"], emit_target_weight=True),
    ],
    ids=["ew_cov", "marginal"],
)
def test_a_model_with_no_target_refuses_it_by_name(builder):
    with pytest.raises(ValueError, match="emit_target_weight"):
        builder()
