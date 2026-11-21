"""Study 499: realised-volatility estimators against hfhd, every table.

    python scripts/study_499.py            # every table (about 6 minutes)
    python scripts/study_499.py --quick    # fewer seeds (about 2 minutes)

Any Python with numpy, scipy, pandas and hfhd (which brings numba) runs it.
The definitions, the one-pass forms and the checks live in
``studies/study_499/``; the write-up is ``docs/records/STUDY-499-REALIZED.md``.
Every tick series is simulated from a seed; nothing is read from disk.
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "studies" / "study_499"))
import compare as C  # noqa: E402
import hfhd  # noqa: E402,F401
import numba  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy  # noqa: E402
from importlib.metadata import version  # noqa: E402


def cell(c: object) -> str:
    if c is None:
        return "-"
    if isinstance(c, (bool, np.bool_)):
        return "yes" if c else "no"
    if isinstance(c, (int, np.integer)):
        return str(int(c))
    if isinstance(c, (float, np.floating)):
        c = float(c)
        if c != c:
            return "-"
        if c == 0.0:
            return "0"
        if abs(c) < 1e-3 or abs(c) >= 1e6:
            return f"{c:.2e}"
        return f"{c:.4g}"
    return str(c)


def table(title: str, cols: list[tuple[str, str]], rows: list[dict]) -> None:
    print(f"\n### {title}\n")
    print("| " + " | ".join(h for h, _ in cols) + " |")
    print("|" + "---|" * len(cols))
    for r in rows:
        print("| " + " | ".join(cell(r.get(k)) for _, k in cols) + " |")
    sys.stdout.flush()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    print("## Study 499 tables")
    print(
        f"\npython {platform.python_version()}, numpy {np.__version__}, scipy {scipy.__version__}, "
        f"pandas {pd.__version__}, numba {numba.__version__}, hfhd {version('hfhd')}; "
        f"{platform.machine()} {platform.system()}"
    )
    agree_seeds = range(2) if args.quick else range(5)
    acc_seeds = range(8) if args.quick else range(40)
    gap_seeds = range(10) if args.quick else range(40)

    table(
        f"Agreement with hfhd (largest gap over seeds {agree_seeds.start}-{agree_seeds.stop - 1}, "
        "relative to the largest entry of hfhd's matrix)",
        [("estimator", "estimator"), ("hfhd setting", "setting"), ("grid", "grid"),
         ("batch, hfhd convention", "hf"), ("one pass", "st"), ("1 = 997 chunks", "inv"),
         ("paper's form vs hfhd", "paper"), ("state bytes", "bytes")],
        C.agreement(agree_seeds),
    )
    table(
        "Bias-corrected MRC (theta 0.4): each step from hfhd's convention to the paper's, alone (seed 0)",
        [("grid", "grid"), ("k", "k"), ("change", "flag"), ("moves the estimate by", "gap")],
        C.mrc_flags(0),
    )
    table(
        f"Pre-averaged Hayashi-Yoshida: the paper's windows against hfhd's (seeds 0-{gap_seeds.stop - 1}; "
        "estimate / true covariance)",
        [("theta", "theta"), ("k", "k"), ("hfhd mean", "hfhd_mean"), ("paper mean", "paper_mean"),
         ("paper, exact psi", "paper_exact_psi_mean"), ("paper - hfhd, mean", "gap_mean"),
         ("paper - hfhd, sd", "gap_sd"), ("hfhd's own sd", "est_sd")],
        C.hy_preavg_gaps(gap_seeds),
    )
    table(
        "Bandwidth: hfhd's get_bandwidth against BNHLS's rule (seed 0, asset A's noise and variance)",
        [("grid", "grid"), ("kernel", "kernel"), ("n", "n"), ("xi^2", "xi2"), ("c* hfhd", "c_hfhd"),
         ("c* exact", "c_exact"), ("H hfhd", "H_hfhd"), ("H exact", "H_exact"), ("ceil", "H_ceil"),
         ("BNHLS estimate moves", "estimate_moves")],
        C.bandwidths(0),
    )
    table(
        "Quadratic spectral cut at L lags against every lag (seed 0)",
        [("grid", "grid"), ("form", "form"), ("H", "H"), ("L / H", "L/H"), ("L", "L"),
         ("gap", "gap"), ("state bytes, d = 2", "state_bytes")],
        C.qs_truncation(0),
    )
    table(
        f"Against the truth (seeds 0-{acc_seeds.stop - 1}): mean of estimate / true - 1, and its root mean square",
        [("grid", "grid"), ("estimator", "estimator"), ("variance A, bias", "var_bias"),
         ("variance A, rmse", "var_rmse"), ("covariance, bias", "cov_bias"), ("covariance, rmse", "cov_rmse")],
        C.accuracy(acc_seeds),
    )
    table(
        "State and cost per tick (ns; best of 3; hfhd and numpy are batch, the one pass is numba-compiled)",
        [("estimator", "estimator"), ("setting", "setting"), ("d", "d"), ("rows", "n"),
         ("state formula", "formula"), ("bytes by formula", "bytes_formula"), ("bytes counted", "bytes_counted"),
         ("hfhd", "ns_hfhd"), ("numpy", "ns_numpy"), ("one pass", "ns_one_pass")],
        C.state_and_cost((2, 10)),
    )
    print(f"\n(all tables in {time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
