"""Does Polars' streaming engine serve a lag without holding the frame?

Run as ``python -m studies.study_532b.lag_memory <parquet> <case>``: it sinks
the parquet through one expression with the streaming engine and prints the
process's peak resident memory (Linux ``ru_maxrss``, KiB). Each case runs in
its own process so the peaks do not mix. ``make`` writes the test file.
"""

from __future__ import annotations

import resource
import sys
import time

import numpy as np
import polars as pl

CASES = {
    "copy": lambda: pl.col("x"),
    "shift(1)": lambda: pl.col("x").shift(1),
    "shift(20)": lambda: pl.col("x").shift(20),
    "shift(1).over(g)": lambda: pl.col("x").shift(1).over("g"),
    "shift(20).over(g)": lambda: pl.col("x").shift(20).over("g"),
}


def make(path: str, n: int, groups: int = 1000) -> None:
    rng = np.random.default_rng(1)
    pl.DataFrame(
        {
            "g": rng.integers(0, groups, n, dtype=np.int64),
            "t": np.arange(n, dtype=np.int64),
            "x": rng.standard_normal(n),
        }
    ).write_parquet(path, row_group_size=1 << 18)


def run(path: str, case: str, out: str) -> None:
    t0 = time.perf_counter()
    pl.scan_parquet(path).select("g", "t", y=CASES[case]()).sink_parquet(out, engine="streaming")
    dt = time.perf_counter() - t0
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f"{case}\t{peak}\t{dt:.2f}")


if __name__ == "__main__":
    if sys.argv[1] == "make":
        make(sys.argv[2], int(sys.argv[3]))
    else:
        run(sys.argv[1], sys.argv[2], sys.argv[3])
