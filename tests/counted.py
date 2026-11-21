"""The rows that count in a clock (hard rule 9; docs/PLAN.md task 539).

A row of weight 0 counts in no clock: the bank scores it as ``predict``
does, steps no model on it, and the next row decays as if it were not there.
So a reference written over a stream steps over the stream with those rows
taken out, and its outputs are the bank's on the rows that count, to the
bit of the bank's own contract (``crates/rheon-polars/tests/zero_weight.rs``
holds every kind to the stream without such rows; ``a_zero_weight_row_is_
scored_as_predict_scores_it`` holds the rows themselves).
"""

from __future__ import annotations

import polars as pl


def counts(weight: str = "w") -> pl.Expr:
    """True on a row that counts: a weight that is null (a skipped row,
    whose time still passes) or other than 0."""
    return pl.col(weight).is_null() | (pl.col(weight) != 0.0)


def counted(df: pl.DataFrame, weight: str = "w") -> pl.DataFrame:
    """``df`` without its rows of weight 0."""
    return df.filter(counts(weight))
