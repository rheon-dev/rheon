"""Study 508, change detectors against river.drift: run every check, print the tables.

    python scripts/study_508.py                 # every table, 200 seeds (about 15 minutes)
    python scripts/study_508.py --seeds 20      # the alarm tables on 20 seeds (about 3 minutes)
    python scripts/study_508.py --no-cost       # without the timings

Any Python with numpy, scipy and river runs it (river is not in the dev group:
a venv of its own, ``pip install numpy scipy river``). The prototypes and the
checks live in ``studies/study_508/``; the write-up is
``docs/records/STUDY-508-DRIFT.md``. Flag comparisons are exact (rows);
continuous outputs are compared in ulps.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "studies" / "study_508"))
import compare  # noqa: E402
import numpy as np  # noqa: E402
import river  # noqa: E402
import scipy  # noqa: E402


def cell(c: object) -> str:
    if c is None:
        return "-"
    if isinstance(c, bool):
        return "yes" if c else "no"
    if isinstance(c, (float, np.floating)):
        c = float(c)
        if c != c:
            return "-"
        if c == 0.0:
            return "0"
        if c == int(c) and abs(c) < 1e7:
            return str(int(c))
        return f"{c:.2e}" if abs(c) < 1e-2 or abs(c) >= 1e5 else f"{c:.3f}"
    return str(c)


def table(head: list[str], rows: list[list[object]]) -> None:
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    for row in rows:
        print("| " + " | ".join(cell(c) for c in row) + " |")
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, default=200, help="seeds for the alarm and delay tables")
    ap.add_argument("--no-cost", action="store_true", help="skip the timings")
    args = ap.parse_args()
    t0 = time.time()
    print(f"numpy {np.__version__}, scipy {scipy.__version__}, river {river.__version__}, Python {sys.version.split()[0]}\n")

    print("## 1. Flag for flag and bit for bit against river (3 seeds x 10,000 rows a stream kind; change at row 5,000)\n")
    rows = []
    for r in compare.parity():
        rows.append([r["case"], r["kind"], r["rows"], r["ours"], r["river"], r["flag_rows_differ"], r["warn_rows_differ"],
                     r["fields_not_bit_equal"], r["max_ulps"]])
    table(["case", "stream", "rows", "flags (ours)", "flags (river)", "flag rows differing", "warning rows differing",
           "field values not bit-equal", "max ulps"], rows)

    print("## 1b. KSWIN's exact p-value: the integer sum rounded once against scipy's evaluation (every h)\n")
    rows = [[r["r"], r["max_ulps"], r["at_h"], r["h_crit(alpha=0.005)"], r["size"], r["paper_h_crit"],
             compare.ks_pvalue_equal(r["r"], r["paper_h_crit"])] for r in compare.kswin_pvalues()]
    table(["r", "max ulps", "at h", "smallest h with p <= 0.005", "its size P(D >= h/r)", "paper's smallest h",
           "its size"], rows)

    print("## 2. The paper's forms where river departs from them (5 seeds x 10,000 rows)\n")
    rows = [[r["form"], r["kind"], r["river_flags"], r["paper_flags"], r["rows_differ"], r["delay_river"], r["delay_paper"]]
            for r in compare.paper_forms()]
    table(["form", "stream", "flags, river's form", "flags, paper's form", "rows differing",
           "first flag after the change, river's (median)", "paper's (median)"], rows)

    print(f"## 3. False alarms and delay ({args.seeds} seeds x 10,000 rows; step and ramp at row 5,000)\n")
    print("Real-valued streams: N(0, 1), a step of +1 sd, or a ramp to +1 sd over 1,000 rows. Bits: error rate")
    print("0.10, a step to 0.25, or a ramp to 0.25 over 1,000 rows. Delay: rows from the change to the first")
    print("flag at or after it (the change row is 1). A false alarm is a flag before the change, or on the")
    print("null stream.\n")
    rows = []
    alarm_rows = compare.alarms(seeds=args.seeds)
    for r in alarm_rows:
        nl, st, gr = r["null"], r["step"], r["gradual"]
        per_check = nl["false_alarms"] / nl["checks"] if nl["checks"] else None
        rows.append([r["case"], nl["false_alarms"], nl["fa_per_row"], per_check,
                     st["false_alarms"], st["delay_median"], st["delay_p10"], st["delay_p90"], st["missed"],
                     gr["delay_median"], gr["delay_p90"], gr["missed"]])
    table(["detector", "null: alarms", "per row", "per check (ADWIN)", "step: alarms before",
           "step delay median", "p10", "p90", "missed", "ramp delay median", "p90", "missed"], rows)

    print("## 3a. ADWIN against its bounds\n")
    print("Per check, Theorem 3.1 bounds the chance of a false cut by delta (for the Hoeffding form, values")
    print("in [0, 1]); with no alarm the rule of three gives the 95% upper limit 3 / checks. For the step,")
    print("n1 at factor 1 is where the expected mean gap first exceeds eps_cut at the split on the change;")
    print("at factor 2 the theorem promises a cut with probability at least 1 - delta (plus the clock's wait).\n")
    params = {
        "ADWIN delta=0.002": (0.002, 32, "bernstein", 1.0, 1.0),
        "ADWIN delta=0.002, clock=1": (0.002, 1, "bernstein", 1.0, 1.0),
        "ADWIN delta=0.1": (0.1, 32, "bernstein", 1.0, 1.0),
        "ADWIN delta=0.002, bits": (0.002, 32, "bernstein", 0.15, 0.3),
        "ADWIN Hoeffding, delta=0.002, bits": (0.002, 32, "hoeffding", 0.15, 0.3),
    }
    rows = []
    for r in alarm_rows:
        if r["case"] not in params:
            continue
        delta, clock, bound, dmu, sd0 = params[r["case"]]
        nl, st = r["null"], r["step"]
        rate = nl["false_alarms"] / nl["checks"]
        upper = (3.0 / nl["checks"]) if nl["false_alarms"] == 0 else None
        n1a = compare.adwin_n1(delta, 5_000, dmu, sd0, bound, 1.0)
        n1b = compare.adwin_n1(delta, 5_000, dmu, sd0, bound, 2.0)
        within = sum(d <= n1b + clock - 1 for d in st["delays"]) / max(1, len(st["delays"]))
        rows.append([r["case"], nl["checks"], nl["false_alarms"], rate, upper, delta,
                     (upper or rate) / delta, n1a, st["delay_median"], n1b, st["delay_max"], within])
    table(["case", "null checks", "false cuts", "per check", "95% upper (none seen)", "delta", "upper / delta",
           "n1, factor 1", "step delay median", "n1, factor 2", "step delay max", "share within factor 2 + clock"], rows)

    print(f"## 3b. KSWIN's test size on the null ({args.seeds} seeds x 10,000 rows, r = 30, alpha = 0.005)\n")
    rows = []
    for dec in (None, 1, 0):
        r = compare.kswin_size(seeds=args.seeds, decimals=dec)
        se = (r["exact_size"] * (1 - r["exact_size"]) / r["tests"]) ** 0.5
        rows.append(["none" if dec is None else f"{dec} decimals", r["tests"], r["rejects"], r["rate"], r["exact_size"],
                     (r["rate"] - r["exact_size"]) / se, r["alarms"], r["alarms_per_row"]])
    table(["values rounded to", "tests (no reset)", "p <= alpha", "rate", "exact size (continuous)",
           "z (as if independent)", "alarms, river's reset", "alarms per row"], rows)

    print("## 4. State size\n")
    rows = [[r["width"], r["levels"], r["levels_bound"], r["buckets"], 5 * r["levels_bound"], r["bytes_counted"],
             r["bytes_formula"], r["rows_over_bound"]] for r in compare.adwin_state()]
    table(["ADWIN width (M = 5)", "levels", "bound 1 + floor(log2((W + M - 1)/M))", "buckets", "M x bound",
           "bytes counted", "bytes by formula", "rows over the bound"], rows)
    rows = [[r["cap"], r["max_width"], r["max_levels"], r["bound"], r["bytes_bound"]]
            for r in (compare.adwin_capped_state(1_000), compare.adwin_capped_state(10_000))]
    table(["ADWIN max_width", "largest width", "most levels", "bound", "bytes at the bound"], rows)
    rows = [[w, compare.river_adwin_pickle(w)] for w in (1_000, 10_000, 100_000)]
    table(["width", "river's pickled ADWIN (bytes)"], rows)
    rows = [[r["detector"], r["bytes"], ", ".join(f"{k} {v}" for k, v in r["arrays"].items())] for r in compare.small_states()]
    table(["detector", "bytes counted", "arrays"], rows)

    if not args.no_cost:
        print("## 5. Per-row cost (microseconds; pure Python, one thread)\n")
        rows = [[r["detector"], r["W"], r["ours_us"], r["ours_clock1_us"], r["river_us"]] for r in compare.cost()]
        table(["detector", "window", "ours", "ours, clock = 1", "river"], rows)

    print("## 6. Zero-weight rows: ours against river on the stream without them (step stream, 20,000 rows)\n")
    rows = [[r["detector"], r["rows"], r["zero_rows"], r["first_row_zero"], r["flags"], r["kept_rows_differ"],
             r["zero_rows_flagged"], r["zero_rows_moving_state"]] for r in compare.zero_weights()]
    table(["detector", "rows", "zero-weight rows", "first row zero", "flags", "kept rows differing from river",
           "zero rows flagged", "zero rows that moved the state"], rows)

    print("## 7. Chunks: one pass against 201 chunks, saved and loaded into a fresh detector at each boundary\n")
    rows = [[r["detector"], r["chunks"], r["flags"], r["rows_differ"], r["final_state_equal"]] for r in compare.chunks()]
    table(["detector", "chunks", "flags", "rows differing", "final state bit-equal"], rows)

    print("## 8. Numerical risk\n")
    rows = []
    for off, cap, n in ((0.0, 1_000, 200_000), (1e4, 1_000, 200_000), (1e8, 1_000, 200_000), (0.0, 10_000, 200_000),
                        (1e4, 10_000, 200_000), (1e8, None, 200_000)):
        r = compare.adwin_drift_of_sums(off, cap, n)
        rows.append([off, cap or "none", n, r["width"], r["total_abs"], r["total_rel"], r["var_rel"]])
    table(["ADWIN: offset", "max_width", "rows", "width", "total abs gap", "total rel gap", "sum of squares rel gap"], rows)
    rows = []
    for off in (0.0, 1e3, 1e6):
        r = compare.mean_forms(1_000_000, off)
        for k, (a, b) in r["marks"].items():
            rows.append([off, k, a, b])
    table(["running mean: offset", "rows", "ulps, (x - m)/n (river)", "ulps, sum/n"], rows)
    rows = []
    for a in (1.0, 1 - 1e-4):
        r = compare.ph_sum_error(1_000_000, alpha=a)
        rows.append([repr(a), r["rows"], r["final_m"], r["worst_abs"], r["rel_to_final"]])
    table(["Page-Hinkley alpha", "rows", "m_t at the end", "worst abs gap", "relative to m_t"], rows)
    r = compare.ddm_count_vs_recursive()
    table(["DDM rows", "max ulps, recursive p vs errors/n", "at the end"], [[r["rows"], r["max_ulps"], r["final_ulps"]]])
    r = compare.adwin_margins()
    table(["ADWIN cut tests (20 seeds x 3 streams)", "closest | |mu0 - mu1| / eps_cut - 1 |"], [[r["tests"], r["min_rel_margin"]]])
    print(f"total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
