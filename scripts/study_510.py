"""Study 510, online mutual information and windowed ranks: run every check, print the tables.

    python scripts/study_510.py              # about 75 s
    python scripts/study_510.py --no-cost    # without the timings (about 45 s)

Any Python with numpy, scipy and scikit-learn runs it (the dev group has all three).
The prototypes and the checks live in ``studies/study_510/``; the write-up is
``docs/records/STUDY-510-DEPENDENCE.md``. Gaps are absolute unless a column says
``rel``; every compared quantity is of order one or smaller.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # steadier timings for small products

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "studies" / "study_510"))
import compare  # noqa: E402
import numpy as np  # noqa: E402
import scipy  # noqa: E402
import sklearn  # noqa: E402


def cell(c: object) -> str:
    if isinstance(c, (float, np.floating)):
        c = float(c)
        if c == 0.0:
            return "0"
        return f"{c:.1e}" if abs(c) < 1e-3 or abs(c) >= 1e4 else f"{c:.4f}"
    return str(c)


def table(head: list[str], rows: list[list[object]]) -> None:
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    for row in rows:
        print("| " + " | ".join(cell(c) for c in row) + " |")
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-cost", action="store_true", help="skip the timings")
    args = ap.parse_args()
    print(
        f"numpy {np.__version__}, scipy {scipy.__version__}, scikit-learn {sklearn.__version__}, "
        f"Python {sys.version.split()[0]}\n"
    )

    print("## 1, 2. Plug-in MI against mutual_info_score; Miller-Madow against its definition\n")
    rows = []
    for rule, bx, by, window in compare.MI_GRID:
        r = compare.check_mi(rule, bx, by, window)
        rows.append(
            [
                rule,
                bx,
                by,
                window or "none",
                r["reads"],
                r["sklearn"],
                r["sklearn_rel"],
                r["mm_def"],
                r["mm_plus_sign"],
            ]
        )
    table(
        [
            "edges",
            "bx",
            "by",
            "W",
            "reads",
            "sklearn",
            "sklearn rel (MI > 1e-3)",
            "MM vs definition",
            "'+' sign vs definition",
        ],
        rows,
    )

    print("## 3. The Gaussian band: quantile bins, N = 20,000, 40 seeds\n")
    rows = []
    for rho in (0.0, 0.3, 0.6, 0.9):
        for b in (8, 16, 32):
            r = compare.check_gaussian(rho, b)
            rows.append(
                [
                    rho,
                    b,
                    r["I"],
                    r["I_bin"],
                    r["loss"],
                    r["plugin"],
                    r["bias_pred"],
                    r["mm"],
                    r["sd"],
                    r["mm_z"],
                    r["mm_worst"],
                ]
            )
    table(
        [
            "rho",
            "bins",
            "I",
            "I_bin",
            "I - I_bin",
            "plug-in mean",
            "(b-1)^2/2N",
            "MM mean",
            "MM sd",
            "MM z vs I_bin",
            "MM worst |. - I_bin|",
        ],
        rows,
    )

    print("## 4. Decay: the table and MI against a plain-Python longhand and scipy's entropy\n")
    rows = []
    for h in (5, 50, 1000):
        r = compare.check_decay(h)
        rows.append(
            [h, r["rows"], r["folds"], r["table_rel"], r["mi_longhand"], r["mi_scipy_entropy"]]
        )
    table(
        ["half-life", "rows", "scale folds", "table rel", "MI vs longhand", "MI vs scipy entropy"],
        rows,
    )

    print("## 5. The rings: zero weights (the first row too), NaN and inf, an emptied window\n")
    r = compare.check_rings()
    table(list(r), [list(r.values())])
    r = compare.check_window_float()
    print("Fractional weights, W = 500, after 100,000 rows:\n")
    table(list(r), [list(r.values())])

    print("## 6. Ranks against scipy (4 x 2 pairs, ties and gaps in the stream)\n")
    rows = []
    for window, every, n in ((50, 1, 800), (200, 7, 1500), (500, 25, 1500)):
        r = compare.check_ranks(window, every, n=n)
        rows.append(
            [
                window,
                every,
                r["reads"],
                r["spearman"],
                r["kendall"],
                r["xi_scipy"],
                r["xi_scipy_tied_x"],
                r["xi_longhand"],
                r["xi_cont"],
                r["held_ok"],
                r["chunks_ok"],
            ]
        )
    table(
        [
            "W",
            "rank_every",
            "reads",
            "spearmanr",
            "kendalltau",
            "chatterjeexi (distinct x)",
            "chatterjeexi (tied x)",
            "xi longhand",
            "xi y_continuous",
            "held",
            "chunks",
        ],
        rows,
    )

    print("## 7. Cost (k = 10 features, T = 1 target, 10 pairs; numpy, one thread)\n")
    if args.no_cost:
        print("skipped\n")
    else:
        rows = []
        for window in (1000, 10_000):
            r = compare.time_ranks(window)
            rows.append([window, r["row_us"], r["read_ms"], r["kendall_read_us"], r["fill_s"]])
        table(
            [
                "W",
                "ranks: per row (us)",
                "Spearman + xi read (ms)",
                "Kendall read (us)",
                "fill (s)",
            ],
            rows,
        )
        rows = []
        for window in (None, 1000, 10_000):
            r = compare.time_mi(window)
            rows.append([window or "none", r["bins"], r["row_us"], r["read_us"]])
        table(["W", "bins", "MI: per row (us)", "MI read, all pairs (us)"], rows)

    print(
        "## 8. State size: arrays counted against the formula "
        "(k = 10, T = 1, 16 x 16 bins, W = 1000)\n"
    )
    r = compare.check_state()
    table(list(r), [list(r.values())])

    print("## Numerical risk: MI read in doubles against np.longdouble\n")
    rows = []
    for n, b in ((200_000, 32), (2_000_000, 64)):
        r = compare.check_mi_precision(n=n, bins=b)
        rows.append(
            [n, b, r["mi"], r["abs_gap"], r["rel_gap"], r["entropy_form_gap"], r["sklearn_gap"]]
        )
    table(["N", "bins", "MI", "abs gap", "rel gap", "H+H-H form gap", "sklearn gap"], rows)


if __name__ == "__main__":
    main()
