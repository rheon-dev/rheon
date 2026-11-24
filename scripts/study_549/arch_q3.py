"""PLAN 549 study (not a test), question 3: does the start's optimizer stop
short of the optimum? For each run in .cache/study549/runs.csv, the start's
final fit (the libm crate's build) on its own rows, scored by `arch`'s
likelihood (`am.fix`), against `arch`'s own fit of the same rows, from its
default start and from the start's fit; the better of the two is the
optimum. Writes .cache/study549/arch_q3.csv.
"""

from __future__ import annotations

import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))
DIR = ROOT / ".cache" / "study549"


def model(vol: str, dist: str, r: np.ndarray):  # noqa: ANN201
    from arch import arch_model
    from arch.univariate import FIGARCH, Normal, ZeroMean

    if vol == "figarch":
        return ZeroMean(
            r, volatility=FIGARCH(p=1, q=1, truncation=1000), distribution=Normal(), rescale=False
        )
    kw = {"garch": dict(vol="GARCH", p=1, o=0, q=1), "gjr": dict(vol="GARCH", p=1, o=1, q=1)}.get(
        vol, dict(vol="EGARCH", p=1, o=1, q=1)
    )
    return arch_model(r, mean="Zero", dist=dist, rescale=False, **kw)


def one(row: dict) -> dict:
    warnings.simplefilter("ignore")
    r = np.fromfile(DIR / "data" / f"{row['name']}.f64", dtype="<f8")[: row["rows_l"]]
    th = np.array([float(x) for x in row["theta_l"].split()])
    am = model(row["vol"], row["dist"], r)
    ours = float(am.fix(th).loglikelihood)
    best = -np.inf
    for sv in (None, th):
        try:
            res = am.fit(disp="off", starting_values=sv, options={"maxiter": 1000})
            best = max(best, float(res.loglikelihood))
        except Exception:  # noqa: BLE001
            pass
    return {
        **{k: row[k] for k in ("name", "vol", "dist", "start", "set", "rows_l", "path_l", "f_l")},
        "ours": ours,
        "arch": best,
    }


def main() -> None:
    df = pl.read_csv(DIR / "runs.csv", infer_schema_length=10000)
    df = df.filter(
        pl.col("set").is_in(["hi", "lo"]) | (pl.col("start") == 50), pl.col("theta_l").is_not_null()
    )
    with Pool(4) as pool:
        out = pool.map(one, df.to_dicts(), chunksize=8)
    pl.DataFrame(out).write_csv(DIR / "arch_q3.csv")


if __name__ == "__main__":
    main()
