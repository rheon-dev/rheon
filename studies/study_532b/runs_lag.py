"""Part 3: lags. The state against Polars; the lag computed by the user with
Polars before the bank, chunk by chunk, with and without a carried tail; and
what Polars' streaming engine holds for ``shift`` and ``shift().over()``."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import polars as pl

from .common import compare, masked, stream_values
from .lag import Lag

KS = (1, 5, 20)


def _expr(kind: str, k: int) -> pl.Expr:
    x = pl.col("xm")
    if kind == "shift":
        return x.shift(k)
    if kind == "diff":
        return x.diff(k)
    return x.pct_change(k)


def vs_polars(df: pl.DataFrame) -> list[list]:
    gs, ts, xs = stream_values(df)
    rows = []
    for kind in ("shift", "diff", "pct_change"):
        for k in KS:
            pol = masked(df).select(_expr(kind, k).over("g").alias("p"))["p"]
            states: dict[int, Lag] = {}
            out = []
            for g, x in zip(gs, xs, strict=True):
                st = states.setdefault(g, Lag(kind, k))
                out.append(st.step(x))
            c = compare(out, pol)
            rows.append([f"{kind}({k})", c["valid"], c["validity_diff"], c["bit_diff"]])
    return rows


def _cuts(n: int, chunks: int, seed: int = 3) -> list[int]:
    if chunks <= 1:
        return [0, n]
    rng = np.random.default_rng(seed + chunks)
    inner = sorted(rng.choice(np.arange(1, n), chunks - 1, replace=False).tolist())
    return [0, *inner, n]


def user_shift_chunked(df: pl.DataFrame) -> list[list]:
    """The user lags with Polars on each chunk before the bank. Without a
    carry, the first k rows of each group in each chunk have no past; with the
    last k rows of each group carried into the next chunk (held by the user,
    outside the bank), the output is the one-pass output to the bit."""
    m = masked(df).with_row_index("i")
    rows = []
    for k in KS:
        one = m.select(pl.col("xm").shift(k).over("g").alias("p"))["p"]
        for chunks in (2, 17, 300):
            cuts = _cuts(m.height, chunks)
            plain, carried = [], []
            tail = m.clear()
            for a, b in zip(cuts[:-1], cuts[1:], strict=True):
                ch = m.slice(a, b - a)
                plain.append(ch.select(pl.col("xm").shift(k).over("g").alias("p"))["p"])
                both = pl.concat([tail, ch])
                lagged = both.select("i", pl.col("xm").shift(k).over("g").alias("p"))
                carried.append(lagged.filter(pl.col("i") >= a)["p"])
                # the user's carry: the last k rows of each group seen so far
                tail = both.filter(pl.int_range(pl.len()).over("g") >= pl.len().over("g") - k)
            p = pl.concat(plain)
            c = pl.concat(carried)
            cp, cc = compare(p.to_list(), one), compare(c.to_list(), one)
            rows.append(
                [k, chunks, cp["validity_diff"] + cp["bit_diff"], cc["validity_diff"] + cc["bit_diff"],
                 tail.height]
            )
    return rows


def streaming_memory(sizes: tuple[int, ...] = (10_000_000, 20_000_000, 40_000_000)) -> list[list]:
    """Peak resident memory of a streaming sink of ``scan_parquet`` through
    each lag expression, in its own process, at three sizes."""
    rows = []
    with tempfile.TemporaryDirectory() as d:
        for n in sizes:
            src = str(Path(d) / f"src{n}.parquet")
            subprocess.run(
                [sys.executable, "-m", "studies.study_532b.lag_memory", "make", src, str(n)],
                check=True,
                cwd=str(Path(__file__).resolve().parents[2]),
            )
            row: list = [f"{n / 1e6:g}M"]
            for case in ("copy", "shift(1)", "shift(20)", "shift(1).over(g)", "shift(20).over(g)"):
                r = subprocess.run(
                    [sys.executable, "-m", "studies.study_532b.lag_memory", src, case,
                     str(Path(d) / "out.parquet")],
                    check=True,
                    capture_output=True,
                    text=True,
                    cwd=str(Path(__file__).resolve().parents[2]),
                )
                _, peak, secs = r.stdout.strip().split("\t")
                row.append(f"{int(peak) / 1024:.0f} MB ({float(secs):.1f} s)")
            Path(src).unlink()
            rows.append(row)
    return rows
