"""The cross-group operators over a relation: ``rp.ops.cross_ref``, a reference
row's value read by every row of its stamp, and ``neighbors=``, a row's
statistic over the rows of the groups its own list names.

Held to Polars on one chunk: the reference read to a join on the clock and,
under ``closed="left"``, to ``join_asof`` backward without an exact match;
the neighbors to an explode of each row's list, a join on the clock and the
key, and a ``group_by`` of the row, with SciPy's ``rankdata`` for the rank.
And what a window run owes: chunk invariance, a save and load at each stamp,
a bounded state, and the refusals.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest
from scipy.stats import rankdata

from rheon import polars as rp

TIER = "essential"

#: Welford's form in input order against Polars' sums, relative.
TOL = 1e-12

CLOCK = {"clock": "t", "gap_cap": 1e9, "group": "g"}


def run(df: pl.DataFrame, **exprs: pl.Expr) -> pl.DataFrame:
    return rp.stream.with_windows(df.lazy(), **exprs, **CLOCK).collect()


def close(a: pl.Series, b: pl.Series) -> None:
    assert np.array_equal(a.is_null().to_numpy(), b.is_null().to_numpy()), (
        a.null_count(),
        b.null_count(),
    )
    x, y = a.drop_nulls().to_numpy(), b.drop_nulls().to_numpy()
    err = np.max(np.abs(x - y) / np.maximum(np.abs(y), 1.0)) if len(x) else 0.0
    assert err < TOL, err


def panel(stamps: int, groups: int, seed: int) -> pl.DataFrame:
    """Stamps in order, each holding some groups in a shuffled order, with
    values (some null) on a coarse grid so a stamp holds ties, weights (some
    0), a leader ``g0`` at most once a stamp, and each row's list of
    neighbors: some groups' keys, at times its own, a key no group has, a
    null key, a repeat, or a null list."""
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(stamps):
        present = rng.permutation(groups)[: rng.integers(1, groups + 1)]
        for g in present:
            x = None if rng.uniform() < 0.07 else float(np.round(rng.normal() * 3))
            w = 0.0 if rng.uniform() < 0.1 else float(rng.uniform(0.2, 3.0))
            if rng.uniform() < 0.05:
                peers = None
            else:
                k = rng.integers(0, 6)
                peers = [f"g{p}" for p in rng.choice(groups, size=k, replace=True)]
                if rng.uniform() < 0.1:
                    peers.append("nobody")
                if rng.uniform() < 0.05:
                    peers.append(None)
            rows.append((float(t), f"g{g}", x, w, peers))
    return pl.DataFrame(
        rows,
        schema={
            "t": pl.Float64,
            "g": pl.String,
            "x": pl.Float64,
            "w": pl.Float64,
            "peers": pl.List(pl.String),
        },
        orient="row",
    )


# ---- the reference read ----------------------------------------------------


def test_the_reference_read_is_a_join_on_the_clock():
    """Under ``"right"``: the leader's value at the row's own stamp, null
    where the stamp has no leader or its value is null. Under ``"left"``:
    ``join_asof`` on the leader's rows with a value, strictly before."""
    df = panel(400, 8, seed=21)
    leader = pl.col("g") == "g0"
    out = run(
        df,
        r=rp.ops.cross_ref("x", where=leader),
        l=rp.ops.cross_ref("x", where=leader, closed="left"),
    )
    ref = df.filter(leader).select("t", ref="x")
    want = df.join(ref, on="t", how="left", maintain_order="left")["ref"]
    assert out["r"].equals(want.rename("r"))
    valued = ref.filter(pl.col("ref").is_not_null())
    asof = df.join_asof(valued, on="t", strategy="backward", allow_exact_matches=False)["ref"]
    assert out["l"].equals(asof.rename("l"))


def test_two_reference_rows_at_one_stamp_are_refused_by_row():
    df = pl.DataFrame(
        {
            "t": [0.0, 0.0, 1.0, 1.0, 1.0],
            "g": ["a", "b", "a", "b", "c"],
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    with pytest.raises(Exception, match=r"picks row 3 .* picked row 2 at the same stamp"):
        run(df, r=rp.ops.cross_ref("x", where=pl.col("x") >= 3.0))
    # One a stamp is read.
    out = run(df, r=rp.ops.cross_ref("x", where=pl.col("g") == "b"))
    assert out["r"].to_list() == [2.0, 2.0, 4.0, 4.0, 4.0]


# ---- neighbors ------------------------------------------------------------


def exploded(df: pl.DataFrame) -> pl.DataFrame:
    """Each row's distinct keys, one a line, with the row's number."""
    return (
        df.with_row_index("i")
        .select("i", "t", "x", k=pl.col("peers").list.drop_nulls().list.unique(maintain_order=True))
        .explode("k")
        .drop_nulls("k")
    )


def members(df: pl.DataFrame) -> pl.DataFrame:
    return df.filter(pl.col("x").is_not_null() & (pl.col("w") > 0)).select(
        "t", k="g", xv="x", wv="w", seq=pl.int_range(pl.len())
    )


def previous_stamps(df: pl.DataFrame) -> dict[float, float | None]:
    """Each stamp's stamp before: the last earlier one with a member."""
    with_member = set(members(df)["t"].to_list())
    before: dict[float, float | None] = {}
    last = None
    for t in df["t"].unique(maintain_order=True).to_list():
        before[t] = last
        if t in with_member:
            last = t
    return before


def neighbors_oracle(df: pl.DataFrame, q: float, *, left: bool) -> pl.DataFrame:
    """Per row: the count of the neighbors' members, their weighted mean,
    unweighted std and nearest quantile, over the row's stamp or the stamp
    before."""
    e = exploded(df)
    if left:
        before = previous_stamps(df)
        e = e.with_columns(t=pl.col("t").replace_strict(before, return_dtype=pl.Float64))
    j = e.join(members(df), on=["t", "k"], how="inner")
    agg = j.group_by("i").agg(
        n=pl.len(),
        mean=(pl.col("xv") * pl.col("wv")).sum() / pl.col("wv").sum(),
        std=pl.col("xv").std(),
        q=pl.col("xv").quantile(q, "nearest"),
    )
    two = pl.col("n") >= 2
    return (
        df.with_row_index("i")
        .join(agg, on="i", how="left", maintain_order="left")
        .select(
            mean=pl.when(two).then("mean"),
            std=pl.when(two).then("std"),
            q=pl.when(two).then("q"),
        )
    )


@pytest.mark.parametrize("closed", ["right", "left"])
def test_the_neighbors_statistics_are_an_explode_and_a_join(closed: str):
    df = panel(300, 9, seed=22)
    out = run(
        df,
        m=rp.ops.cross_mean("x", weight="w", neighbors="peers", closed=closed),
        s=rp.ops.cross_std("x", neighbors="peers", closed=closed, weight=pl.col("w") > 0),
        q=rp.ops.cross_quantile(
            "x", 0.4, neighbors="peers", closed=closed, weight=(pl.col("w") > 0).cast(pl.Float64)
        ),
    )
    want = neighbors_oracle(df, 0.4, left=closed == "left")
    close(out["m"], want["mean"])
    close(out["s"], want["std"])
    # The weighted quantile at unit weights reads the inverted distribution
    # function; NumPy's, per row.
    e = exploded(df)
    if closed == "left":
        before = previous_stamps(df)
        e = e.with_columns(t=pl.col("t").replace_strict(before, return_dtype=pl.Float64))
    j = e.join(members(df), on=["t", "k"], how="inner")
    values = {i: part["xv"].to_numpy() for (i,), part in j.group_by("i")}
    wq = [
        float(np.quantile(values[i], 0.4, method="inverted_cdf"))
        if i in values and len(values[i]) >= 2
        else None
        for i in range(df.height)
    ]
    close(out["q"], pl.Series(wq, dtype=pl.Float64))


@pytest.mark.parametrize("closed", ["right", "left"])
@pytest.mark.parametrize("method", ["average", "min", "max", "dense", "ordinal"])
def test_the_neighbors_rank_is_scipys(closed: str, method: str):
    """The row's own value among its neighbors' members, itself among them
    under ``"right"`` where its list names its own group, and otherwise as
    though it were added last: SciPy's ``rankdata``."""
    # A rank takes no weight: every row with a value is a member.
    df = panel(200, 7, seed=23).with_columns(w=pl.lit(1.0))
    out = run(df, r=rp.ops.cross_rank("x", method=method, neighbors="peers", closed=closed))
    e = exploded(df)
    if closed == "left":
        before = previous_stamps(df)
        e = e.with_columns(t=pl.col("t").replace_strict(before, return_dtype=pl.Float64))
    m = members(df)
    j = e.join(m, on=["t", "k"], how="inner").sort("i", "seq")
    by_row = {i: part for (i,), part in j.group_by("i", maintain_order=True)}
    # Each member row's own place among the members, by input order.
    member_seq: dict[int, int] = {}
    seq = 0
    for i, (x, w) in enumerate(zip(df["x"], df["w"], strict=True)):
        if x is not None and w > 0:
            member_seq[i] = seq
            seq += 1
    got = out["r"].to_list()
    for i, x in enumerate(df["x"].to_list()):
        part = by_row.get(i)
        if x is None or part is None or part.height < 2:
            assert got[i] is None, (i, got[i])
            continue
        values = part["xv"].to_numpy()
        seqs = part["seq"].to_list()
        own = member_seq.get(i)
        if closed == "right" and own in seqs:
            want = rankdata(values, method=method)[seqs.index(own)]
        else:
            want = rankdata(np.append(values, x), method=method)[-1]
        assert got[i] == want, (i, got[i], want)


def test_an_integer_key_is_named_by_an_integer_list():
    df = pl.DataFrame(
        {
            "t": [0.0, 0.0, 0.0],
            "g": [1, 2, 3],
            "x": [1.0, 2.0, 4.0],
            "peers": [[2, 3], [1], [1, 2, 3]],
        }
    )
    out = run(df, m=rp.ops.cross_mean("x", neighbors="peers"))
    m = out["m"].to_list()
    assert m[:2] == [3.0, None]
    assert abs(m[2] - 7.0 / 3.0) < 1e-15


# ---- what a run owes ---------------------------------------------------------


def exprs() -> dict[str, Any]:
    leader = pl.col("g") == "g0"
    return {
        "r": rp.ops.cross_ref("x", where=leader),
        "rl": rp.ops.cross_ref("x", where=leader, closed="left"),
        "m": rp.ops.cross_mean("x", weight="w", neighbors="peers"),
        "ml": rp.ops.cross_std("x", neighbors="peers", closed="left"),
        "k": rp.ops.cross_rank("x", method="ordinal", neighbors="peers"),
        "kl": rp.ops.cross_rank("x", neighbors="peers", closed="left"),
        "q": rp.ops.cross_quantile("x", 0.5, neighbors="peers", closed="left"),
    }


@pytest.mark.parametrize("rows", [1, 2, 7, 64, 10_000])
def test_chunking_changes_nothing(rows: int):
    df = panel(100, 8, seed=24)
    one = rp.stream.with_windows(df.lazy(), **exprs(), **CLOCK, chunk_size=100_000).collect()
    out = rp.stream.with_windows(df.lazy(), **exprs(), **CLOCK, chunk_size=rows).collect()
    assert out.equals(one)


def test_a_save_and_load_at_every_stamp_is_one_run(tmp_path: Path):
    df = panel(25, 5, seed=25)
    one = rp.stream.with_windows(df, **exprs(), **CLOCK)
    starts = [i for i in range(1, df.height) if df["t"][i] != df["t"][i - 1]]
    for at in starts:
        state = tmp_path / f"c{at}.state"
        first = rp.stream.with_windows(df[:at], **exprs(), **CLOCK, save_state=state)
        second = rp.stream.with_windows(df[at:], **exprs(), **CLOCK, load_state=state)
        assert pl.concat([first, second]).equals(one), at


def test_the_state_is_bounded_by_the_groups(tmp_path: Path):
    """The state of a run of 2,000 stamps is the size of one of 20 of the
    same groups and lists, give or take the rows of the last stamp."""
    sizes = []
    for stamps in (20, 2000):
        df = panel(stamps, 6, seed=26)
        state = tmp_path / f"s{stamps}.state"
        rp.stream.with_windows(
            df,
            rl=rp.ops.cross_ref("x", where=pl.col("g") == "g0", closed="left"),
            ml=rp.ops.cross_mean("x", neighbors="peers", closed="left"),
            **CLOCK,
            save_state=state,
        )
        sizes.append(state.stat().st_size)
    assert abs(sizes[0] - sizes[1]) < 600, sizes


def test_a_stamp_of_long_lists_is_held_to_the_cap():
    """Under ``"right"`` each row of the open stamp is 16 bytes held (one
    operator column), 40 kept and 8 plus its text a key: a row listing the
    two keys ``"a"`` and ``"b"`` is 74 bytes, so a cap of 300 holds four
    of them and refuses the fifth."""

    def stamp(rows: int) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "t": [0.0] * rows,
                "g": [f"g{i}" for i in range(rows)],
                "x": [float(i) for i in range(rows)],
                "peers": [["a", "b"]] * rows,
            }
        )

    with pytest.warns(Warning, match="past half of its max_state_bytes of 300"):
        run(stamp(4), m=rp.ops.cross_mean("x", neighbors="peers", max_state_bytes=300))
    with pytest.raises(Exception, match=r"at row 4 .*past its max_state_bytes of 300"):
        run(stamp(5), m=rp.ops.cross_mean("x", neighbors="peers", max_state_bytes=300))


def test_the_refusals():
    df = pl.DataFrame({"t": [0.0], "g": ["a"], "x": [1.0], "peers": [["a"]], "n": [1.0]})
    with pytest.raises(TypeError, match="where is required"):
        rp.ops.cross_ref("x", where=None)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="neighbors must name a list column"):
        rp.ops.cross_mean("x", neighbors=1)  # type: ignore[arg-type]
    with pytest.raises(Exception, match="must be a list of group keys"):
        run(df, m=rp.ops.cross_mean("x", neighbors="n"))
    with pytest.raises(Exception, match="is not in the input"):
        run(df, m=rp.ops.cross_mean("x", neighbors="nope"))
    with pytest.raises(Exception, match="names no group column"):
        rp.stream.with_windows(
            df, m=rp.ops.cross_mean("x", neighbors="peers"), clock="t", gap_cap=1.0
        )
    from rheon.polars._formula import operator

    with pytest.raises(Exception, match="a reference read takes no weight"):
        run(df, r=operator("cross_ref", "x", where=pl.col("g") == "a", weight="x"))
