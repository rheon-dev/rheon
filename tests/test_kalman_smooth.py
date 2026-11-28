"""``kalman``'s fixed-lag smoother on the Python surface: ``smooth_lag``,
``emit_smoothed_se`` and ``max_state_bytes`` on ``rp.spec.kalman``, the
fields they add, the refusals, and ``ModelBank.smoothed_tail``.

The exactness of the smoothed values is held in Rust, against the
whole-prefix smoother written from the filter's own pairs
(``crates/rheon-core/src/kalman/tests/smooth.rs``), and against statsmodels
in ``tests/test_second_opinion_smoother.py``. Here: what a caller sees.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from rheon import polars as rp

TIER = "essential"


def _frame(n: int = 300, seed: int = 496) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    slope = 1.0 + np.cumsum(0.02 * rng.standard_normal(n))
    y = slope * x + 0.3 * rng.standard_normal(n)
    w = np.where(rng.random(n) < 0.1, 0.0, 1.0)
    return pl.DataFrame({"t": np.arange(n, dtype=float), "x": x, "y": y, "w": w})


def _spec(**kw: object) -> dict:
    return rp.spec.kalman(
        "k",
        targets=["y"],
        features=["x"],
        clock="t",
        gap_cap=10.0,
        half_life=100.0,
        coef_half_life=50.0,
        weight="w",
        min_weight=0.0,
        **kw,  # type: ignore[arg-type]
    )


def test_a_spec_without_the_smoother_is_as_it_was():
    s = _spec()
    assert not {"smooth_lag", "emit_smoothed_se", "max_state_bytes"} & set(s["model"])
    assert rp.ModelBank([s]).specs[0] == s


def test_the_smoother_adds_its_fields_and_leaves_every_prediction():
    df = _frame()
    s = _spec(smooth_lag=4, emit_smoothed_se=True, max_state_bytes=100_000)
    assert rp.ModelBank([s]).specs[0] == s
    fields = rp.spec.output_fields(s)
    at = fields.index("coef")
    assert fields[at + 1 : at + 4] == ["smoothed_coef", "smoothed_se_coef", "smoothed_for_y"]
    out = rp.ModelBank([s]).fit_predict(df)["k"].struct.unnest()
    plain = rp.ModelBank([_spec()]).fit_predict(df)["k"].struct.unnest()
    assert out["pred_y"].equals(plain["pred_y"])
    observed = df.filter(pl.col("w") > 0)["t"].to_list()
    described = out["smoothed_for_y"].to_list()
    # Each value describes the observation four before the row's own.
    for i, row in enumerate(observed):
        got = described[int(row)]
        assert got == (observed[i - 4] if i >= 4 else None), row
    assert out["smoothed_coef"].drop_nulls().len() == len(observed) - 4


@pytest.mark.parametrize(
    ("kw", "said"),
    [
        ({"smooth_lag": 0}, "smooth_lag must be at least 1"),
        ({"emit_smoothed_se": True}, "emit_smoothed_se is set, but smooth_lag is not"),
        ({"max_state_bytes": 1000}, "max_state_bytes is set, but smooth_lag is not"),
        (
            {"smooth_lag": 3, "covariance_form": "square_root"},
            'smooth_lag needs covariance_form="covariance"',
        ),
        ({"smooth_lag": 5000, "emit_smoothed_se": True}, "past max_state_bytes of 262144"),
    ],
)
def test_the_smoothers_settings_are_refused_where_they_cannot_hold(kw, said):
    with pytest.raises(ValueError) as e:
        _spec(**kw)
    assert said in str(e.value)


def test_the_smoother_emits_list_columns_under_embargo():
    df = _frame(150)
    s = _spec(smooth_lag=4, emit_smoothed_se=True, embargo=3.0)
    idx = rp.spec.output_index(s)
    dtypes = dict(zip(idx["field"], idx["dtype"], strict=True))
    assert dtypes["smoothed_coef"] == "list[list[f64]]"
    assert dtypes["smoothed_se_coef"] == "list[list[f64]]"
    assert dtypes["smoothed_for_y"] == "list[clock]"
    out = rp.ModelBank([s]).fit_predict(df)["k"].struct.unnest()
    assert out["smoothed_coef"].dtype == pl.List(pl.List(pl.Float64))
    assert out["smoothed_se_coef"].dtype == pl.List(pl.List(pl.Float64))
    assert out["smoothed_for_y"].dtype == pl.List(pl.Float64)
    assert out["smoothed_coef"].drop_nulls().len() > 50


def test_the_bank_reads_the_last_observations_from_the_state():
    df = _frame()
    bank = rp.ModelBank([_spec(smooth_lag=4, emit_smoothed_se=True)])
    bank.fit(df)
    tail = bank.smoothed_tail("k")
    assert tail.columns == [
        "spec",
        "group",
        "instance",
        "half_life",
        "lam",
        "target",
        "smoothed_for",
        "withheld",
        "smoothed_coef",
        "smoothed_se_coef",
    ]
    observed = df.filter(pl.col("w") > 0)["t"].to_list()
    assert tail["smoothed_for"].to_list() == observed[-4:]
    assert tail["withheld"].null_count() == 4
    plain = rp.ModelBank([_spec()])
    plain.fit(df)
    with pytest.raises(ValueError, match="it has no fixed-lag smoother"):
        plain.smoothed_tail("k")
