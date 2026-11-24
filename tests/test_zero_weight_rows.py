"""Rows of weight 0: what the bank reports on them and what they leave
behind (review 2026-11-23, STRM-2 and STRM-3).

A row of weight 0 learns nothing and counts in no clock (hard rule 9): it
is scored as ``predict`` scores a row, from the state the next row of
positive weight would be scored by, and leaves nothing a later row, a
summary or ``last_row`` reads. The Rust side, every kind on every clock to
the bit, is ``crates/rheon-polars/tests/zero_weight.rs``; these hold the
bank's reports on the rows themselves.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from rheon import polars as rp

TIER = "essential"


def _session_frame(n: int = 100) -> pl.DataFrame:
    rng = np.random.default_rng(578)
    x = rng.standard_normal(n + 1)
    y = 2.0 * x + 0.1 * rng.standard_normal(n + 1)
    return pl.DataFrame(
        {
            "t": np.arange(n + 1, dtype=float),
            "s": ["A"] * n + ["B"],
            "x": x,
            "y": y,
            "w": [1.0] * n + [0.0],
        }
    )


def _rows(out: pl.DataFrame) -> list[dict]:
    """Each row's fields, NaN read as None so that two NaNs compare equal."""
    rows = out["m"].struct.unnest().drop("coef").to_dicts()
    return [
        {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in r.items()}
        for r in rows
    ]


def test_a_zero_weight_row_that_starts_a_fresh_session_reads_a_fresh_streams_diagnostics():
    """STRM-2: under ``session_gap = "reset"`` a row of weight 0 that starts
    a session is scored by a fresh model, but read the closed session's
    ``settled_frac``, ``sigma``, ``r2`` and ``learned_clock``, so its
    ``min_settled_frac`` gate opened on a model that has seen nothing. It
    reads what a fresh stream reads: the bank fed that row alone, and
    ``predict`` on it after the session before."""
    df = _session_frame()
    s = rp.spec.ewridge(
        "m",
        targets=["y"],
        features=["x"],
        clock="t",
        gap_cap=1e9,
        session="s",
        session_gap="reset",
        weight="w",
        half_life=10.0,
        min_weight=0.0,
        min_settled_frac=0.5,
        emit_sigma=True,
        emit_metrics=True,
        emit_clocks=True,
    )
    bank = rp.ModelBank([s])
    whole = _rows(bank.fit_predict(df))[-1]
    alone = _rows(rp.ModelBank([s]).fit_predict(df.tail(1)))[0]
    assert whole == alone
    before = rp.ModelBank([s])
    before.fit(df.head(df.height - 1))
    predicted = _rows(before.predict(df.tail(1)))[0]
    assert whole == predicted


def test_rows_of_weight_0_leave_the_settled_weight_where_it_was():
    """STRM-3 (2): ``summary()["weight_sum_settled"]`` projects the ceiling
    from the rows' mean weight, the summary's ``weight_sum`` over the rows
    processed, whose count takes in the rows of weight 0: a stream with one
    between each pair read a mean of half the weight its rows carry, and
    took the first row's share out of the ceiling at half its weight: 2%
    off after 20 rows at a half-life of 30. The mean is over the rows that
    carry weight."""
    rng = np.random.default_rng(5)
    n = 20
    x = rng.standard_normal(n)
    df = pl.DataFrame({"t": np.arange(n, dtype=float), "x": x, "y": 2.0 * x, "w": 1.0})
    s = rp.spec.ewridge(
        "m",
        targets=["y"],
        features=["x"],
        clock="t",
        gap_cap=1e9,
        weight="w",
        half_life=30.0,
    )
    plain = rp.ModelBank([s])
    plain.fit(df)
    # A row of weight 0 after each row, at the same clock.
    zeros = df.with_columns(pl.lit(0.0).alias("w"))
    doubled = pl.concat([df, zeros]).sort("t", maintain_order=True)
    mixed = rp.ModelBank([s])
    mixed.fit(doubled)
    a = plain.summary()["weight_sum_settled"][0]
    b = mixed.summary()["weight_sum_settled"][0]
    assert a > 0.0
    assert b == pytest.approx(a, rel=1e-12)


def test_ew_class_withholds_until_its_labelled_rows_reach_min_weight():
    """STRM-4 (1): ``ew_class`` counted a row whose label is null in the
    weight ``min_weight`` is checked against, so four unlabelled rows opened
    the gate and the first labelled row's class was predicted with
    certainty. Its own weight is the labelled rows', as each target's of a
    regression is the rows it was present on; ``weight_sum`` stays the
    shared weight."""
    x = np.arange(9.0)
    df = pl.DataFrame({"x": x, "lab": [None, None, None, None, "a", "b", "a", "b", "a"]})
    s = rp.spec.ew_class(
        "m",
        features=["x"],
        label="lab",
        classes=["a", "b"],
        precision_prior=1.0,
        half_life=1e9,
        min_weight=3.0,
    )
    out = rp.ModelBank([s]).fit_predict(df)["m"].struct.unnest()
    assert out["weight_sum"].to_list() == pytest.approx(range(9), rel=1e-8)
    # Three labelled rows before row 7, their weight a hair under 3 at this
    # half-life, as the shared weight's three rows were under it at row 3;
    # four before row 8, where the gate opens.
    assert out["withheld_reason"].to_list()[:8] == ["below_min_weight"] * 8
    assert out["withheld_reason"][8] is None
    assert out["p_a"][:8].null_count() == 8
    assert out["p_a"][8] is not None


def test_the_learned_clock_is_the_last_row_with_a_target_learned():
    """STRM-4 (2): ``learned_clock`` is the newest row the model learned
    from; a row whose every target is null is only predicted, as the
    summary counts it, and moved it."""
    df = pl.DataFrame(
        {
            "t": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "y": [0.0, 2.0, 4.0, None, None, 10.0],
        }
    )
    s = rp.spec.ewridge(
        "m",
        targets=["y"],
        features=["x"],
        clock="t",
        gap_cap=1e9,
        half_life=10.0,
        emit_clocks=True,
    )
    out = rp.ModelBank([s]).fit_predict(df)["m"].struct.field("learned_clock")
    assert out.to_list() == [None, 0.0, 1.0, 2.0, 2.0, 2.0]
