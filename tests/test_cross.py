"""The cross-group operators, ``rp.ops.cross_mean``, ``rp.ops.cross_std``,
``rp.ops.cross_rank`` and ``rp.ops.cross_quantile``: a statistic over every group's
row at each stamp of the clock, broadcast back to each row.

Held to Polars on one chunk -- ``pl.col(x).mean().over(clock)`` and
``.std().over(clock)``, or ``group_by(clock)`` and a join -- and, under
weights, to numpy's weighted mean and the weighted variance written out;
``closed="left"`` to the same per stamp, shifted one stamp back. And what a
window run owes: chunk invariance, a save and load at every row, zero
weights in no stamp, fewer than two weighted rows withheld, a late stamp
refused by row, and a clock break ending the stamp before. The rank and
the quantile are held to Polars' ``rank`` and ``quantile`` over the clock,
to SciPy's ``rankdata`` and to NumPy's ``quantile``, weighted by its
``inverted_cdf``.
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
    out = run(
        df, m=rp.ops.cross_mean("x"), s=rp.ops.cross_std("x"), b=rp.ops.cross_std("x", bias=True)
    )
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
        m=rp.ops.cross_mean("x", weight="w"),
        s=rp.ops.cross_std("x", weight="w"),
        b=rp.ops.cross_std("x", weight=pl.col("w"), bias=True),
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
        df,
        m=rp.ops.cross_mean("x", weight="w", closed="left"),
        s=rp.ops.cross_std("x", closed="left"),
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
        m=rp.ops.cross_mean("x", weight="w"),
        left=rp.ops.cross_mean("x", weight="w", closed="left"),
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
        "m": rp.ops.cross_mean("x", weight="w"),
        "s": rp.ops.cross_std("x", weight="w"),
        "l": rp.ops.cross_mean("x", weight="w", closed="left"),
        "q": rp.ops.cross_quantile("x", 0.4, weight="w"),
        "ql": rp.ops.cross_quantile("x", 0.4, weight="w", closed="left"),
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
        "m": rp.ops.cross_mean("x", weight="w"),
        "s": rp.ops.cross_std("x"),
        "l": rp.ops.cross_std("x", weight="w", closed="left"),
        "e": rp.ops.ewm_mean("x", half_life=3.0),
        "r": rp.ops.cross_rank("x", method="dense"),
        "rl": rp.ops.cross_rank("x", closed="left", descending=True),
        "q": rp.ops.cross_quantile("x", 0.3, interpolation="linear"),
        "ql": rp.ops.cross_quantile("x", 0.7, weight="w", closed="left"),
    }
    one = rp.stream.with_windows(df.lazy(), **exprs, **CLOCK, chunk_size=100_000).collect()
    out = rp.stream.with_windows(df.lazy(), **exprs, **CLOCK, chunk_size=rows).collect()
    assert out.equals(one)


def test_a_save_and_load_at_every_row_is_one_run(tmp_path: Path):
    df = frame(30, 4, seed=6)
    exprs: dict[str, Any] = {
        "m": rp.ops.cross_mean("x", weight="w"),
        "l": rp.ops.cross_std("x", closed="left"),
        "r": rp.ops.cross_rank("x", method="ordinal"),
        "rl": rp.ops.cross_rank("x", closed="left"),
        "q": rp.ops.cross_quantile("x", 0.5, weight="w"),
        "ql": rp.ops.cross_quantile("x", 0.5, closed="left"),
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
        run(df, m=rp.ops.cross_mean("x"))


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
        m=rp.ops.cross_mean("x"),
        l=rp.ops.cross_mean("x", closed="left"),
        clock="t",
        gap_cap=10.0,
        group="g",
    ).collect()
    assert out["m"].to_list() == [2.0, 2.0, 4.0, 4.0, 1.0, 1.0, 6.0, 6.0]
    assert out["l"].to_list() == [None, None, 2.0, 2.0, None, None, 1.0, 1.0]


def test_the_refusals():
    df = pl.DataFrame({"t": [0.0], "g": ["a"], "x": [1.0]})
    with pytest.raises(ValueError, match='closed must be "right"'):
        rp.ops.cross_mean("x", closed="both")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="bias must be True or False"):
        rp.ops.cross_std("x", bias=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="another window operator"):
        rp.ops.cross_mean(rp.ops.ewm_mean("x", half_life=1.0))
    with pytest.raises(Exception, match="names no clock column"):
        rp.stream.with_windows(df.lazy(), m=rp.ops.cross_mean("x"), group="g").collect()
    # A spec learns each group apart, so no target reads a cross-section,
    # beside a window looking ahead or not.
    ahead = rp.ops.ewm_mean("x", half_life=1.0, window_size=5.0, reverse=True)
    with pytest.raises(Exception, match="cross-group operator"):
        rp.spec.ewridge(
            "m",
            targets=[(ahead - rp.ops.cross_mean("x")).alias("c")],
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
            m=rp.ops.cross_mean("x", closed="left"),
            s=rp.ops.cross_std("x", closed="left"),
            r=rp.ops.cross_rank("x", closed="left"),
            q=rp.ops.cross_quantile("x", 0.5, closed="left"),
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
        run(small, m=rp.ops.cross_mean("x", max_state_bytes=160))
    big = pl.concat([stamp(6, 0.0), stamp(8, 1.0), stamp(4, 2.0)])
    with pytest.warns(RheonWarning, match="past half of its max_state_bytes of 160") as caught:
        out = run(big, m=rp.ops.cross_mean("x", max_state_bytes=160))
    assert len([w for w in caught if issubclass(w.category, RheonWarning)]) == 1
    assert out["m"].to_list()[:6] == [2.5] * 6
    over = pl.concat([stamp(4, 0.0), stamp(11, 1.0)])
    with pytest.raises(Exception, match=r"at row 14 .*past its max_state_bytes of 160"):
        run(over, m=rp.ops.cross_mean("x", max_state_bytes=160))
    # The causal form holds no row.
    run(over, m=rp.ops.cross_mean("x", closed="left", max_state_bytes=160))
    with pytest.raises(ValueError, match="max_state_bytes must be a whole number from 1"):
        rp.ops.cross_std("x", max_state_bytes=0)


# ---- rank and quantile ------------------------------------------------------

RANK_METHODS = ("average", "min", "max", "dense", "ordinal")
INTERPOLATIONS = ("nearest", "higher", "lower", "midpoint", "linear", "equiprobable")
LEVELS = (0.0, 0.1, 0.25, 0.3, 0.5, 0.73, 0.9, 1.0)


def tied(stamps: int, groups: int, seed: int) -> pl.DataFrame:
    """:func:`frame` with its values on a coarse grid, so a stamp holds
    ties, and with some values null, which are no member of any stamp."""
    rng = np.random.default_rng(seed + 1000)
    df = frame(stamps, groups, seed)
    x = np.round(df["x"].to_numpy())
    return df.with_columns(
        x=pl.when(pl.Series(rng.uniform(size=df.height) < 0.08)).then(None).otherwise(pl.Series(x))
    )


def members_over(expr: pl.Expr) -> pl.Expr:
    """``expr`` over the stamp where it holds two values or more."""
    return pl.when(pl.col("x").count().over("t") >= 2).then(expr)


def test_the_rank_is_polars_over_the_stamp():
    """Every tie rule in both directions: Polars' ``rank().over(clock)``
    on the whole frame, a null value no member and ranked null."""
    df = tied(300, 12, seed=11)
    for method in RANK_METHODS:
        for descending in (False, True):
            out = run(df, r=rp.ops.cross_rank("x", method=method, descending=descending))
            want = df.select(
                members_over(
                    pl.col("x").rank(method, descending=descending).over("t").cast(pl.Float64)
                )
            )["x"]
            assert out["r"].equals(want.rename("r")), (method, descending)


def scipy_rank(values: np.ndarray, method: str, descending: bool) -> np.ndarray:
    from scipy.stats import rankdata

    return rankdata(-values if descending else values, method=method)


def test_the_rank_is_scipys_rankdata_per_stamp():
    """A second opinion: SciPy's ``rankdata`` of each stamp's values."""
    df = tied(200, 10, seed=12)
    for method in RANK_METHODS:
        for descending in (False, True):
            out = run(df, r=rp.ops.cross_rank("x", method=method, descending=descending))
            for (t,), part in out.group_by("t", maintain_order=True):
                x = part["x"].to_numpy()
                have = ~np.isnan(x)
                got = part["r"].to_numpy()
                if have.sum() < 2:
                    assert part["r"].null_count() == part.height, t
                    continue
                want = scipy_rank(x[have], method, descending)
                assert np.array_equal(got[have], want), (method, descending, t)
                assert np.isnan(got[~have]).all()


def test_left_ranks_the_row_among_the_stamp_before():
    """Under ``"left"`` a row is ranked among the stamp before's values as
    though it were added to them last: SciPy's ``rankdata`` of those values
    and the row's, read at the row. The first stamp, and a stamp after one
    of fewer than two values, read null."""
    df = tied(150, 8, seed=13)
    stamps = [
        (t, part["x"].drop_nulls().to_numpy())
        for (t,), part in df.group_by("t", maintain_order=True)
    ]
    before: dict[float, np.ndarray | None] = {}
    last = None
    for t, values in stamps:
        before[t] = last
        if len(values):
            last = values
    for method in RANK_METHODS:
        for descending in (False, True):
            out = run(
                df, r=rp.ops.cross_rank("x", method=method, descending=descending, closed="left")
            )
            for t, x, r in zip(out["t"], out["x"], out["r"], strict=True):
                prev = before[t]
                if x is None or prev is None or len(prev) < 2:
                    assert r is None, (method, t)
                    continue
                want = scipy_rank(np.append(prev, x), method, descending)[-1]
                assert r == want, (method, descending, t, x)


def test_the_quantile_is_polars_over_the_stamp():
    """Every interpolation at levels on and between the values: Polars'
    ``quantile().over(clock)`` on the whole frame, to the bit but where a
    level falls between two values under ``"linear"``."""
    df = tied(300, 12, seed=14).with_columns(x=pl.col("x") + pl.col("w"))
    for interpolation in INTERPOLATIONS:
        for q in LEVELS:
            out = run(df, q=rp.ops.cross_quantile("x", q, interpolation=interpolation))
            want = df.select(members_over(pl.col("x").quantile(q, interpolation).over("t")))
            if interpolation == "linear":
                # Polars reads a group's level from an index offset by its
                # nulls, (n - 1) q + nulls, whose fraction can lose a last
                # bit (0.9 + 1 - 1 is 0.8999999999999999); NumPy agrees
                # with this operator there, below.
                close(out["q"], want["x"])
            else:
                assert out["q"].equals(want["x"].rename("q")), (interpolation, q)


def test_the_quantile_is_numpys_per_stamp():
    """A second opinion: NumPy's ``quantile`` of each stamp's values, by the
    method that reads a level as Polars' interpolation does. NumPy's
    ``"nearest"`` rounds a half to even where Polars rounds it away from 0,
    so it is held to Polars alone, above."""
    methods = {
        "lower": "lower",
        "higher": "higher",
        "midpoint": "midpoint",
        "linear": "linear",
        "equiprobable": "inverted_cdf",
    }
    df = tied(200, 10, seed=15).with_columns(x=pl.col("x") + pl.col("w"))
    for interpolation, method in methods.items():
        for q in LEVELS:
            out = run(df, q=rp.ops.cross_quantile("x", q, interpolation=interpolation))
            want = []
            for (_,), part in df.group_by("t", maintain_order=True):
                x = part["x"].drop_nulls().to_numpy()
                v = np.quantile(x, q, method=method) if len(x) >= 2 else np.nan
                want += [v] * part.height
            close(out["q"], pl.Series(want).fill_nan(None))


def test_the_weighted_quantile_is_numpys_inverted_cdf():
    """Under weights: NumPy's ``quantile(method="inverted_cdf",
    weights=...)`` of each stamp's rows of weight; a zero weight is no
    member, and fewer than two members are withheld."""
    df = tied(300, 10, seed=16)
    for q in LEVELS:
        out = run(
            df,
            q=rp.ops.cross_quantile("x", q, weight="w"),
            left=rp.ops.cross_quantile("x", q, weight="w", closed="left"),
        )
        per: dict[float, float] = {}
        order: list[float] = []
        for (t,), part in df.group_by("t", maintain_order=True):
            part = part.filter(pl.col("x").is_not_null() & (pl.col("w") > 0))
            order.append(t)
            if part.height >= 2:
                x, w = part["x"].to_numpy(), part["w"].to_numpy()
                per[t] = float(np.quantile(x, q, method="inverted_cdf", weights=w))
            elif part.height == 1:
                per[t] = np.nan
        want = [per.get(t, np.nan) for t in df["t"]]
        close(out["q"], pl.Series(want).fill_nan(None))
        # Under "left": the stamp before with a member.
        prev: dict[float, float] = {}
        last = np.nan
        for t in order:
            prev[t] = last
            if t in per:
                last = per[t]
        close(out["left"], pl.Series([prev[t] for t in df["t"]]).fill_nan(None))


def test_the_ordered_operators_bound_their_state():
    """A rank or a quantile keeps 16 bytes a row of the open stamp beside
    the held row: with one operator column a held row is 16 bytes, so 32 a
    row, and a cap of 160 holds a stamp of 5 rows. Under ``"left"`` a rank
    keeps 16 bytes a row of the open stamp and of each distinct value of
    the stamp before: a cap of 160 holds stamps of 5 rows of distinct
    values, and a quantile 10 rows."""

    def stamp(rows: int, t: float) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "t": [t] * rows,
                "g": [f"g{i}" for i in range(rows)],
                "x": [float(i) for i in range(rows)],
            }
        )

    five = pl.concat([stamp(5, 0.0), stamp(5, 1.0)])
    six = pl.concat([stamp(5, 0.0), stamp(6, 1.0)])
    for op in (
        rp.ops.cross_rank("x", max_state_bytes=160),
        rp.ops.cross_quantile("x", 0.5, max_state_bytes=160),
    ):
        with pytest.warns(RheonWarning):
            run(five, m=op)
        with pytest.raises(Exception, match=r"at row 10 .*past its max_state_bytes of 160"):
            run(six, m=op)
    # "left": 5 rows of the stamp before and 5 of the open one.
    with pytest.warns(RheonWarning):
        run(five, m=rp.ops.cross_rank("x", closed="left", max_state_bytes=160))
    with pytest.raises(Exception, match=r"at row 10 .*past its max_state_bytes of 160"):
        run(six, m=rp.ops.cross_rank("x", closed="left", max_state_bytes=160))
    ten = pl.concat([stamp(10, 0.0), stamp(10, 1.0)])
    with pytest.warns(RheonWarning):
        run(ten, m=rp.ops.cross_quantile("x", 0.5, closed="left", max_state_bytes=160))
    with pytest.raises(Exception, match=r"at row 10 .*past its max_state_bytes of 160"):
        run(
            pl.concat([stamp(11, 0.0)]),
            m=rp.ops.cross_quantile("x", 0.5, closed="left", max_state_bytes=160),
        )


def test_the_rank_and_quantile_refusals():
    from rheon.polars._formula import operator

    df = pl.DataFrame({"t": [0.0], "g": ["a"], "x": [1.0], "w": [1.0]})
    with pytest.raises(ValueError, match="method must be one of"):
        rp.ops.cross_rank("x", method="random")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="descending must be True or False"):
        rp.ops.cross_rank("x", descending=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="quantile must be a number from 0 to 1"):
        rp.ops.cross_quantile("x", 1.5)
    with pytest.raises(TypeError, match="quantile must be a number from 0 to 1"):
        rp.ops.cross_quantile("x", "0.5")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="interpolation must be one of"):
        rp.ops.cross_quantile("x", 0.5, interpolation="cubic")  # type: ignore[arg-type]
    with pytest.raises(
        ValueError, match="\"linear\" has no weighted form|'linear' has no weighted"
    ):
        rp.ops.cross_quantile("x", 0.5, interpolation="linear", weight="w")
    # The native reader refuses a weighted rank, which no call writes.
    with pytest.raises(Exception, match="a rank takes no weight"):
        run(df, r=operator("cross_rank", "x", weight="w"))
