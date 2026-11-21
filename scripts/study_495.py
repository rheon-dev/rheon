"""Study 495, an online VAR(p) against statsmodels: run the comparisons, print the tables.

    python scripts/study_495.py            # about 15 s
    python scripts/study_495.py --drift    # also the window's long-stream drift, minutes

Any Python with numpy, scipy and statsmodels runs it (the dev group has all three).
The model and the checks live in ``studies/var_495/``; the write-up is
``docs/records/STUDY-495-VAR.md``. Errors are ``max|a - b| / max|b|``; for the
Granger tests the worst ``|a / b - 1|`` over every test.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "studies" / "var_495"))
import compare  # noqa: E402

warnings.filterwarnings("ignore")


def table(head: list[str], rows: list[list[object]]) -> None:
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    for row in rows:
        print("| " + " | ".join(f"{c:.1e}" if isinstance(c, float) else str(c) for c in row) + " |")
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--drift", action="store_true", help="window drift over 1e4 to 1e6 rows")
    args = ap.parse_args()

    print("## Fit and Granger, no decay, T = 400\n")
    fit_cols = ["params", "sigma_u", "sigma_u_mle", "cov_params", "granger_stat"]
    fit_cols += ["granger_pvalue", "granger_tests"]
    irf_cols = ["irf", "orth_irf", "irf_cov", "fevd"]
    fits, irfs = [], []
    for t, k, p in compare.SPEC_GRID:
        res = compare.check_fit(t, k, p)
        fits.append([k, p, *(res[c] for c in fit_cols[:-1]), int(res["granger_tests"])])
        res = compare.check_irf(t, k, p)
        irfs.append([k, p, *(res[c] for c in irf_cols)])
    table(["k", "p", *fit_cols], fits)
    print("## Impulse responses (10 periods) and FEVD\n")
    table(["k", "p", *irf_cols], irfs)

    print("## Rolling window (W = 120, k = 3, p = 2)\n")
    res = compare.check_window()
    table(list(res), [[float(v) for v in res.values()]])
    gaps = compare.check_window_gaps()
    print(f"window with zero weights and NaN rows, against WLS: {gaps:.1e}\n")
    if args.drift:
        print("## Window drift (W = 200, k = 3, p = 2)\n")
        rows = []
        for level in (0.0, 1e2, 1e4):
            for t in (10_000, 100_000, 1_000_000):
                res = compare.check_window_drift(t, level)
                rows.append([f"{level:.0e}", t, *(float(v) for v in res.values())])
        table(["level", "T", "slopes", "sigma_u", "granger_stat"], rows)

    print("## The earlier checks\n")
    compare.main()


if __name__ == "__main__":
    main()
