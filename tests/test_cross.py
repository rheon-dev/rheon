"""The cross-group operators, ``rp.cross_mean`` and ``rp.cross_std``: a
statistic over every group's row at each stamp of the clock, broadcast back
to each row.

Held to Polars on one chunk -- ``pl.col(x).mean().over(clock)`` and
``.std().over(clock)``, or ``group_by(clock)`` and a join -- and, under
weights, to numpy's weighted mean and the weighted variance written out;
``closed="left"`` to the same per stamp, shifted one stamp back. And what a
window run owes: chunk invariance, a save and load at every row, zero
weights in no stamp, fewer than two weighted rows withheld, a late stamp
refused by row, and a clock break ending the stamp before.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest

from rheon import RheonWarning
from rheon import polars as rp

TIER = "essential"

#: The cross mean and standard deviation against Polars and numpy,
#: relative: Welford's form in input order against Polars' two-pass sums,
#: measured at most 2.4e-15 on the frames below.
TOL = 1e-12

CLOCK = {"clock": "t", "gap_cap": 1e9, "group": "g"}


def frame(stamps: int, groups: int, seed: int, *, holes: bool = True) -> pl.DataFrame:
    """Stamps in order, each holding some of the groups (with ``holes``) in
    a shuffled order, a value on a level of 100, and unequal weights with
    zeros among them."""
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(stamps):
        present = rng.permutation(groups)
        if holes:
            present = present[: rng.integers(1, groups + 1)]
        for g in present:
            w = 0.0 if rng.uniform() < 0.1 else rng.uniform(0.2, 3.0)
            rows.append((float(t) * 1.5, f"g{g}", 100.0 + rng.normal() * (1 + g % 3), w))
    return pl.DataFrame(rows, schema=["t", "g", "x", "w"], orient="row")


def close(a: pl.Series, b: pl.Series) -> None:
    assert a.null_count() == b.null_count(), (a.null_count(), b.null_count())
    x, y = a.drop_nulls().to_numpy(), b.drop_nulls().to_numpy()
    assert np.array_equal(a.is_null().to_numpy(), b.is_null().to_numpy())
    err = np.max(np.abs(x - y) / np.maximum(np.abs(y), 1.0)) if len(x) else 0.0
    assert err < TOL, err


def run(df: pl.DataFrame, **exprs: pl.Expr) -> pl.DataFrame:
    return rp.stream.with_windows(df.lazy(), **exprs, **CLOCK).collect()


def test_the_mean_and_std_are_polars_over_the_stamp():
    """At unit weights, every row's value is Polars' ``over(clock)`` on the
    whole frame, where its stamp holds two rows or more; null where it
    holds one."""
    df = frame(300, 12, seed=1)
    out = run(df, m=rp.cross_mean("x"), s=rp.cross_std("x"), b=rp.cross_std("x", bias=True))
    n = pl.len().over("t")
    want = df.with_columns(
        m=pl.when(n >= 2).then(pl.col("x").mean().over("t")),
        s=pl.when(n >= 2).then(pl.col("x").std().over("t")),
        b=pl.when(n >= 2).then(pl.col("x").std(ddof=0).over("t")),
    )
    for c in ("m", "s", "b"):
        close(out[c], want[c])
    # The group_by form of the oracle agrees with the window form.
    agg = df.group_by("t").agg(m2=pl.col("x").mean(), k=pl.len())
    joined = df.join(agg, on="t", how="left").with_columns(m2=pl.when(pl.col("k") >= 2).then("m2"))
    close(out["m"], joined["m2"])


def weighted(x: np.ndarray, w: np.ndarray) -> tuple[float, float, float]:
    """The weighted mean (numpy's ``average``), and the weighted standard
    deviation written out: corrected by ``V1^2 / (V1^2 - V2)``, and not."""
    keep = w > 0
    x, w = x[keep], w[keep]
    if len(x) < 2:
        return (np.nan, np.nan, np.nan)
    m = np.average(x, weights=w)
    v1, v2 = w.sum(), (w**2).sum()
    m2 = (w * (x - m) ** 2).sum()
    return m, np.sqrt(m2 / v1 * v1**2 / (v1**2 - v2)), np.sqrt(m2 / v1)


def per_stamp(df: pl.DataFrame) -> dict[float, tuple[float, float, float]]:
    return {
        t: weighted(part["x"].to_numpy(), part["w"].to_numpy())
        for (t,), part in df.group_by("t", maintain_order=True)
    }


def test_the_weighted_forms_are_the_definitions():
    """Under weights, numpy's weighted mean and the weighted variance
    written out; a zero weight contributes nothing, and a stamp with fewer
    than two rows of weight is null."""
    df = frame(300, 10, seed=2)
    out = run(
        df,
        m=rp.cross_mean("x", weight="w"),
        s=rp.cross_std("x", weight="w"),
        b=rp.cross_std("x", weight=pl.col("w"), bias=True),
    )
    stamps = per_stamp(df)
    want = np.array([stamps[t] for t in df["t"].to_list()])
    for j, c in enumerate("msb"):
        close(out[c], pl.Series(want[:, j]).fill_nan(None))
    # numpy's own weighted covariance, a second path to the spread.
    t0 = df.filter(pl.col("t") == 0.0)
    if (t0["w"] > 0).sum() >= 2:
        c = np.cov(t0["x"].to_numpy(), aweights=t0["w"].to_numpy(), ddof=0)
        assert abs(np.sqrt(c) - out.filter(pl.col("t") == 0.0)["b"][0]) < 1e-12 * np.sqrt(c)


def test_left_reads_the_stamp_before():
    """``closed="left"``: each row reads the cross-section of the stamp
    before its own, the one with a row of weight; the first stamp reads
    null. Polars: the per-stamp value, shifted one stamp, joined back."""
    df = frame(200, 8, seed=3)
    out = run(
        df, m=rp.cross_mean("x", weight="w", closed="left"), s=rp.cross_std("x", closed="left")
    )
    stamps = per_stamp(df)
    weighted_stamps = [
        t for (t,), part in df.group_by("t", maintain_order=True) if (part["w"] > 0).any()
    ]
    before = {}
    last = None
    for t in df["t"].unique(maintain_order=True).to_list():
        before[t] = last
        if t in weighted_stamps:
            last = t
    want = [np.nan if before[t] is None else stamps[before[t]][0] for t in df["t"].to_list()]
    close(out["m"], pl.Series(want).fill_nan(None))
    # The unweighted std reads the stamp before, every row of it counting.
    std = df.group_by("t", maintain_order=True).agg(s=pl.col("x").std(), n=pl.len())
    std = std.with_columns(s=pl.when(pl.col("n") >= 2).then("s").shift(1))
    close(out["s"], df.join(std, on="t", how="left")["s"])


def test_a_stamp_of_one_weighted_row_is_withheld():
    df = pl.DataFrame(
        {
            "t": [0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0],
            "g": ["a", "b", "a", "b", "a", "b", "a"],
            "x": [1.0, 3.0, 5.0, 7.0, 2.0, 4.0, 9.0],
            "w": [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        }
    )
    out = run(
        df,
        m=rp.cross_mean("x", weight="w"),
        left=rp.cross_mean("x", weight="w", closed="left"),
    )
    # Two weighted rows, then one, then none, then a single row.
    assert out["m"].to_list() == [2.0, 2.0, None, None, None, None, None]
    # Stamp 1 had a weighted row (withheld, one); stamp 2 had none, so it is
    # no stamp: stamp 3 reads stamp 1's.
    assert out["left"].to_list() == [None, None, 2.0, 2.0, None, None, None]


def test_a_zero_weight_row_reads_its_stamp_and_adds_nothing():
    """The weighted rows read the same with the zero-weight rows or without
    them, and a zero-weight row reads its stamp's value."""
    df = frame(150, 6, seed=4)
    exprs = {
        "m": rp.cross_mean("x", weight="w"),
        "s": rp.cross_std("x", weight="w"),
        "l": rp.cross_mean("x", weight="w", closed="left"),
    }
    full = run(df, **exprs)
    kept = run(df.filter(pl.col("w") > 0), **exprs)
    assert full.filter(pl.col("w") > 0).equals(kept)
    zero = full.filter(pl.col("w") == 0)
    by_stamp = dict(zip(kept["t"].to_list(), kept["m"].to_list(), strict=True))
    for t, m in zip(zero["t"].to_list(), zero["m"].to_list(), strict=True):
        if t in by_stamp:
            assert m == by_stamp[t] or (m is None and by_stamp[t] is None)


@pytest.mark.parametrize("rows", [1, 2, 7, 64, 10_000])
def test_chunking_changes_nothing(rows: int):
    df = frame(120, 9, seed=5)
    exprs = {
        "m": rp.cross_mean("x", weight="w"),
        "s": rp.cross_std("x"),
        "l": rp.cross_std("x", weight="w", closed="left"),
        "e": rp.ewm_mean("x", half_life=3.0),
    }
    one = rp.stream.with_windows(df.lazy(), **exprs, **CLOCK, chunk_size=100_000).collect()
    out = rp.stream.with_windows(df.lazy(), **exprs, **CLOCK, chunk_size=rows).collect()
    assert out.equals(one)


def test_a_save_and_load_at_every_row_is_one_run(tmp_path: Path):
    df = frame(30, 4, seed=6)
    exprs: dict[str, Any] = {
        "m": rp.cross_mean("x", weight="w"),
        "l": rp.cross_std("x", closed="left"),
    }
    one = rp.stream.with_windows(df, **exprs, **CLOCK)
    # A file resumes at a stamp's first row: one that begins inside a run of
    # tied clock values could repeat rows the state read, which is refused.
    starts = [i for i in range(1, df.height) if df["t"][i] != df["t"][i - 1]]
    for at in starts:
        state = tmp_path / f"c{at}.state"
        first = rp.stream.with_windows(df[:at], **exprs, **CLOCK, save_state=state)
        second = rp.stream.with_windows(df[at:], **exprs, **CLOCK, load_state=state)
        assert pl.concat([first, second]).equals(one), at


def test_a_stamp_out_of_order_across_groups_is_refused_by_row():
    df = pl.DataFrame({"t": [0.0, 1.0, 0.5], "g": ["a", "b", "c"], "x": [1.0, 2.0, 3.0]})
    with pytest.raises(Exception, match="row 2"):
        run(df, m=rp.cross_mean("x"))


def test_a_clock_break_ends_the_stamp_before():
    """A gap past ``gap_cap`` closes the open stamp as any later row does,
    and ends the one before: under ``"left"`` the stamp after it reads
    null."""
    df = pl.DataFrame(
        {
            "t": [0.0, 0.0, 1.0, 1.0, 50.0, 50.0, 51.0, 51.0],
            "g": ["a", "b"] * 4,
            "x": [1.0, 3.0, 2.0, 6.0, 1.0, 1.0, 5.0, 7.0],
        }
    )
    out = rp.stream.with_windows(
        df.lazy(),
        m=rp.cross_mean("x"),
        l=rp.cross_mean("x", closed="left"),
        clock="t",
        gap_cap=10.0,
        group="g",
    ).collect()
    assert out["m"].to_list() == [2.0, 2.0, 4.0, 4.0, 1.0, 1.0, 6.0, 6.0]
    assert out["l"].to_list() == [None, None, 2.0, 2.0, None, None, 1.0, 1.0]


def test_the_refusals():
    df = pl.DataFrame({"t": [0.0], "g": ["a"], "x": [1.0]})
    with pytest.raises(ValueError, match='closed must be "right"'):
        rp.cross_mean("x", closed="both")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="bias must be True or False"):
        rp.cross_std("x", bias=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="another window operator"):
        rp.cross_mean(rp.ewm_mean("x", half_life=1.0))
    with pytest.raises(Exception, match="names no clock column"):
        rp.stream.with_windows(df.lazy(), m=rp.cross_mean("x"), group="g").collect()
    # A spec learns each group apart, so no target reads a cross-section,
    # beside a window looking ahead or not.
    ahead = rp.ewm_mean("x", half_life=1.0, window_size=5.0, reverse=True)
    with pytest.raises(Exception, match="cross-group operator"):
        rp.spec.ewridge(
            "m",
            targets=[(ahead - rp.cross_mean("x")).alias("c")],
            features=["x"],
            half_life=10.0,
            embargo=5.0,
        )


def test_the_state_is_bounded_by_the_operators(tmp_path: Path):
    """The cross state is a fixed size whatever the stream's length: a run
    of 10 stamps and one of 2,000, of the same groups, save files whose
    sizes differ by the rows of the last stamp each holds and their clock,
    not by the stream."""
    sizes = []
    for stamps in (10, 2000):
        df = frame(stamps, 5, seed=7, holes=False)
        state = tmp_path / f"s{stamps}.state"
        rp.stream.with_windows(
            df,
            m=rp.cross_mean("x", closed="left"),
            s=rp.cross_std("x", closed="left"),
            **CLOCK,
            save_state=state,
        )
        sizes.append(state.stat().st_size)
    assert abs(sizes[0] - sizes[1]) < 200, sizes


def test_a_stamp_past_half_the_cap_warns_once_and_past_the_cap_is_refused():
    """``8 * (outputs + 1)`` bytes a held row: with one operator column, 16
    bytes a row of a stamp. A cap of 160 holds a stamp of 10 rows; 6 rows
    take it past half, which warns once a run; the 11th row of a stamp is
    refused by row. Under ``"left"`` nothing is held, so no cap is met."""

    def stamp(rows: int, t: float) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "t": [t] * rows,
                "g": [f"g{i}" for i in range(rows)],
                "x": [float(i) for i in range(rows)],
            }
        )

    small = pl.concat([stamp(5, 0.0), stamp(5, 1.0)])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        run(small, m=rp.cross_mean("x", max_state_bytes=160))
    big = pl.concat([stamp(6, 0.0), stamp(8, 1.0), stamp(4, 2.0)])
    with pytest.warns(RheonWarning, match="past half of its max_state_bytes of 160") as caught:
        out = run(big, m=rp.cross_mean("x", max_state_bytes=160))
    assert len([w for w in caught if issubclass(w.category, RheonWarning)]) == 1
    assert out["m"].to_list()[:6] == [2.5] * 6
    over = pl.concat([stamp(4, 0.0), stamp(11, 1.0)])
    with pytest.raises(Exception, match=r"at row 14 .*past its max_state_bytes of 160"):
        run(over, m=rp.cross_mean("x", max_state_bytes=160))
    # The causal form holds no row.
    run(over, m=rp.cross_mean("x", closed="left", max_state_bytes=160))
    with pytest.raises(ValueError, match="max_state_bytes must be a whole number from 1"):
        rp.cross_std("x", max_state_bytes=0)
