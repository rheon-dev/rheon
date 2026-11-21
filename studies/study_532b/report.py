"""Every table of docs/records/STUDY-532-WINDOWLESS.md, printed as Markdown."""

from __future__ import annotations

import platform
import sys
import time

import numpy as np
import polars as pl
import scipy

from . import runs_bucket as rb
from . import runs_ew as rew
from . import runs_lag as rlag
from . import runs_lag_checks as rchk
from . import runs_resume as rres
from .bucket import Bucketed, BucketSpec
from .common import make_frame, stream_values
from .lag import Lag

# (b, B): a span of 60 cut 60, 12 or 4 ways, and a span of 240 cut 12, 4 or 1 ways
BUCKET_CONFIGS = ((1, 60), (5, 12), (15, 4), (20, 12), (60, 4), (240, 1))


def md(header: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        out.append("| " + " | ".join(_fmt(v) for v in r) + " |")
    return "\n".join(out)


def _fmt(v: object) -> str:
    if isinstance(v, float):
        if v == 0.0:
            return "0"
        if abs(v) >= 1e4 or abs(v) < 1e-3:
            return f"{v:.2e}"
        return f"{v:.3g}"
    return str(v)


def versions() -> None:
    print("### Versions\n")
    print(
        md(
            ["python", "numpy", "scipy", "polars", "machine"],
            [
                [
                    platform.python_version(),
                    np.__version__,
                    scipy.__version__,
                    pl.__version__,
                    f"{platform.system()} {platform.machine()}",
                ]
            ],
        )
    )
    print()


# ---------------------------------------------------------------------------
# Part 1


def table_b1(n: int) -> None:
    print(
        f"### Table B1: bucketed windows against `group_by_dynamic`, every window "
        f"({n} rows, 5 groups)\n"
    )
    print(
        "Each cell: validity differences / bit differences / largest ulp gap, over "
        "the windows that hold a value. `sum` and `mean` read the compensated sums; "
        "`var` the centred form. Second block: NaN, +inf, -inf in 0.4% of x.\n"
    )
    rows = []
    for nonfinite in (False, True):
        df = make_frame(n, nonfinite=nonfinite)
        for b, B in BUCKET_CONFIGS:
            r = rb.vs_group_by_dynamic(df, b, B)
            cells = [
                f"{r[s]['validity_diff']}/{r[s]['bit_diff']}/{r[s]['max_ulp']}" for s in rb.STATS
            ]
            rows.append(
                [
                    "yes" if nonfinite else "no",
                    b,
                    B,
                    r["windows"],
                    *cells,
                    f"{r['empty_windows']} ({r['empty_polars_sum_nonzero']}, "
                    f"{r['empty_polars_sum_maxabs']:.1e})",
                ]
            )
    print(
        md(
            [
                "non-finite",
                "b",
                "B",
                "windows",
                *rb.STATS,
                "windows with no value (Polars sum not 0, largest |sum|)",
            ],
            rows,
        )
    )
    print()


def table_b2(n: int) -> None:
    print(
        f"### Table B2: sum, mean and variance against the exact value, on the "
        f"`group_by_dynamic` windows ({n} rows, finite data)\n"
    )
    print(
        "Largest and median ulp gap to the exact value (sum: `math.fsum`; mean: the "
        "exact rational mean rounded once; var: exact rational arithmetic rounded "
        "once), the bucketed state and Polars side by side.\n"
    )
    rows = []
    df = make_frame(n)
    for b, B in BUCKET_CONFIGS:
        r = rb.vs_group_by_dynamic(df, b, B)
        row = [b, B]
        for s in ("sum", "mean", "var"):
            o, p = r[s + "_ours_exact"], r[s + "_polars_exact"]
            row += [
                f"{o['max_ulp']} ({o['median_ulp']:g})",
                f"{p['max_ulp']} ({p['median_ulp']:g})",
            ]
        rows.append(row)
    print(
        md(
            [
                "b",
                "B",
                "sum: buckets",
                "sum: Polars",
                "mean: buckets",
                "mean: Polars",
                "var: buckets",
                "var: Polars",
            ],
            rows,
        )
    )
    print()
    lv = []
    for level in (1e3, 1e6):
        r = rb.sumsq_vs_m2(make_frame(n, level=level), 20, 12)
        for form in ("m2", "sumsq"):
            c = r[form]
            lv.append(
                [
                    f"{level:g}",
                    "centred m2 (Chan merge)" if form == "m2" else "sum of squares",
                    c["max_ulp"],
                    f"{c['median_ulp']:g}",
                    c["max_rel"],
                ]
            )
    print("The variance's two bucket forms against the exact value (b = 20, B = 12):\n")
    print(md(["level", "form", "max ulp", "median ulp", "max relative"], lv))
    print()


def table_b3(n: int) -> None:
    print(
        f"### Table B3: bucketed windows against `rolling_*_by(window_size = B b, "
        f'closed = "right")` on the rows with `t mod b = b - 1` ({n} rows)\n'
    )
    print(
        "On these rows the B buckets are the span exactly. Cell: validity differences / "
        "bit differences / largest ulp gap.\n"
    )
    rows = []
    df = make_frame(n)
    for b, B in BUCKET_CONFIGS:
        r = rb.vs_rolling_aligned(df, b, B)
        rows.append(
            [
                b,
                B,
                r["rows"],
                *[
                    f"{r[s]['validity_diff']}/{r[s]['bit_diff']}/{r[s]['max_ulp']}"
                    for s in rb.STATS
                ],
            ]
        )
    print(md(["b", "B", "rows", *rb.STATS], rows))
    print()


def table_b4(n: int) -> None:
    print(
        f"### Table B4: the window edge inside a bucket, every row against "
        f"`rolling_*_by` over `(t - B b, t]` ({n} rows, rows with 3 or more values)\n"
    )
    print(
        "Each gap is median / 95th percentile / largest. Count: relative to the span's "
        "count. Mean, min, max: in units of the span's standard deviation. sd: "
        "relative. `B` reads buckets `k-B+1..k` (inside the span); `B+1` reads "
        "`k-B..k` (covers it); `interp` adds the fraction of bucket `k-B` inside the "
        "span, as if its rows were spread evenly.\n"
    )
    rows = []
    df = make_frame(n)
    for b, B in ((5, 12), (20, 3), (15, 4), (20, 12), (60, 4), (240, 1)):
        r = rb.edge_gap(df, b, B)
        for name in ("B", "B+1", "interp"):
            x = r[name]

            def trip(key: str, x: dict = x) -> str:
                if key not in x:
                    return "-"
                a, q, m = x[key]
                return f"{_fmt(a)} / {_fmt(q)} / {_fmt(m)}"

            pct = 100.0 * x["count_differs"] / x["rows"]
            mm = (
                f"{100.0 * x['min_differs'] / x['rows']:.1f}% / "
                f"{100.0 * x['max_differs'] / x['rows']:.1f}%"
                if "min_differs" in x
                else "-"
            )
            rows.append(
                [
                    b,
                    B,
                    B * b,
                    name,
                    f"{pct:.1f}%",
                    trip("count"),
                    trip("mean"),
                    trip("sd"),
                    trip("min"),
                    trip("max"),
                    mm,
                ]
            )
        rows.append(
            ["", "", "", "bracket violations", r["bracket_violations"], "", "", "", "", "", ""]
        )
    print(
        md(
            [
                "b",
                "B",
                "span",
                "read",
                "rows whose count differs",
                "count",
                "mean (sd units)",
                "sd (relative)",
                "min (sd units)",
                "max (sd units)",
                "rows whose min / max differ",
            ],
            rows,
        )
    )
    print()


def table_b5() -> None:
    print("### Table B5: bucketed state bytes, saved form, per group (no row is stored)\n")
    print(
        "Per bucket and column: `n` (u64), the sum and its compensation, the centred "
        "sum of squares, min, max: 48 bytes. Per column: the current bucket id (8 "
        "bytes; a bank shares one across the columns of a group). Measured: the saved "
        "size after streaming 20000 rows through one group (it does not move with the "
        "rows).\n"
    )
    rows = []
    _, ts, xs = stream_values(make_frame(20000, groups=1))
    for B in (1, 4, 12, 60, 240):
        for keep in (False, True):
            spec = BucketSpec(width=5, buckets=B, keep_previous=keep)
            st = Bucketed(spec)
            peak = 0
            for i, (t, x) in enumerate(zip(ts, xs, strict=True)):
                st.step(x, t)
                if i % 37 == 0:
                    peak = max(peak, len(st.to_bytes()))
            row = [B, "yes" if keep else "no", "8 + 48 (B + p)", Bucketed.bytes_formula(spec), peak]
            for k in (10,):
                row.append(8 + 48 * spec.ring * k)
            rows.append(row)
    print(
        md(
            [
                "B",
                "keeps bucket k-B (p)",
                "formula",
                "bytes, one column (formula)",
                "saved size, peak",
                "10 columns, one shared id (computed)",
            ],
            rows,
        )
    )
    print()


def part1(quick: bool) -> None:
    n = 4000 if quick else 10000
    table_b1(n)
    table_b2(n)
    table_b3(n)
    table_b4(n)
    table_b5()


# ---------------------------------------------------------------------------
# Part 2


def table_e1(n: int) -> None:
    print(
        f"### Table E1: exponentially weighted features against Polars, one pass "
        f"({n} rows, 5 groups, `over`)\n"
    )
    print(
        "Nulls about 3% of x, zero-weight rows about 4% (masked to null); the second "
        "column set adds NaN, +inf and -inf in 0.4% of x. alpha is carried in the state.\n"
    )
    a = rew.vs_polars(make_frame(n))
    b = rew.vs_polars(make_frame(n, nonfinite=True))
    rows = []
    for (name, r), (_, q) in zip(a, b, strict=True):
        rows.append(
            [
                name,
                r["valid"],
                r["validity_diff"],
                r["bit_diff"],
                r["max_ulp"],
                q["valid"],
                q["validity_diff"],
                q["bit_diff"],
                q["max_ulp"],
            ]
        )
    print(
        md(
            [
                "output",
                "valid",
                "validity diff",
                "bit diff",
                "max ulp",
                "valid (non-finite)",
                "validity diff",
                "bit diff",
                "max ulp",
            ],
            rows,
        )
    )
    print()


def table_e2() -> None:
    print(
        "### Table E2: the carried alpha, a state saved at row 10000 of 20000 and "
        "continued (one series, hl = 10)\n"
    )
    print(
        "Largest ulp gap over the second half to the uncut stream: loaded from the "
        "bytes (alpha carried), and with alpha recomputed by a libm that differs by "
        "one ulp, in alpha itself or in `exp(-ln 2 / hl)` (the libm result).\n"
    )
    print(
        md(
            [
                "feature",
                "saved bytes",
                "alpha carried",
                "alpha +1 ulp",
                "alpha -1 ulp",
                "exp +1 ulp",
                "exp -1 ulp",
            ],
            rew.carried_alpha(),
        )
    )
    print()


def table_e3() -> None:
    print(
        "### Table E3: the decayed envelopes against Polars' rolling extremes "
        "(20000 rows, after a start-up of 5 w)\n"
    )
    print(
        "Max envelope minus `rolling_max(w)`, in units of `rolling_std(w)`: mean (the "
        "bias), 5th and 95th percentiles, share of rows above; the min envelope's "
        "mean gap to `rolling_min(w)`; the correlation of the max envelope with "
        "`rolling_max(w)`.\n"
    )
    print(
        md(
            [
                "series",
                "w",
                "half-life",
                "max: mean gap",
                "p5",
                "p95",
                "rows above",
                "min: mean gap",
                "corr(max env, rolling_max)",
            ],
            rew.envelope_bias(),
        )
    )
    print()
    print(
        "Cormode et al.'s decayed max `max(x, r y)` against `rolling_max(w)` (same "
        "units); the last column is the share of rows where it is just the current "
        "value:\n"
    )
    print(
        md(
            ["series", "w", "half-life", "mean gap", "p5", "p95", "rows equal to x"],
            rew.decayed_max_rows(),
        )
    )
    print()


def part2(quick: bool) -> None:
    n = 6000 if quick else 20000
    table_e1(n)
    table_e2()
    table_e3()


# ---------------------------------------------------------------------------
# Part 3


def table_l1(n: int) -> None:
    print(f"### Table L1: the lag state against Polars, one pass ({n} rows, 5 groups, `over`)\n")
    a = rlag.vs_polars(make_frame(n))
    b = rlag.vs_polars(make_frame(n, nonfinite=True))
    rows = [[*r, *q[1:]] for r, q in zip(a, b, strict=True)]
    print(
        md(
            [
                "output",
                "valid",
                "validity diff",
                "bit diff",
                "valid (non-finite)",
                "validity diff",
                "bit diff",
            ],
            rows,
        )
    )
    print()
    print("### Table L2: lag state bytes, saved form\n")
    rows = []
    for k in rlag.KS:
        f = Lag.bytes_formula(k)
        st = Lag("shift", k)
        for x in range(1000):
            st.step(float(x))
        rows.append(
            [k, "8 + 8k + ceil(k/8)", f, len(st.to_bytes()), 8 * k, f"{f * 10 * 1000 / 1e6:.2f} MB"]
        )
    print(
        md(
            [
                "k",
                "bytes (formula)",
                "formula",
                "measured",
                "lower bound 8k",
                "10 columns x 1000 groups",
            ],
            rows,
        )
    )
    print()


def table_l3(n: int, mem: bool) -> None:
    print(
        f"### Table L3: the lag computed by the user with Polars before the bank, "
        f"chunk by chunk ({n} rows, 5 groups)\n"
    )
    print(
        'Rows that differ from one Polars pass of `shift(k).over("g")`: each chunk '
        "lagged alone, and each chunk lagged with the last k rows of each group carried "
        "from the chunks before (the user holds them, not the bank).\n"
    )
    print(
        md(
            [
                "k",
                "chunks",
                "rows wrong, no carry",
                "rows wrong, k rows carried",
                "rows the user carries (k x groups)",
            ],
            rlag.user_shift_chunked(make_frame(n)),
        )
    )
    print()
    if not mem:
        return
    print(
        "### Table L4: what Polars' streaming engine holds for a lag "
        '(`scan_parquet(...).select(g, t, y = expr).sink_parquet(engine = "streaming")`, '
        "1000 groups, peak resident memory of the process)\n"
    )
    print(
        md(
            ["rows", "copy x", "shift(1)", "shift(20)", "shift(1).over(g)", "shift(20).over(g)"],
            rlag.streaming_memory(),
        )
    )
    print()


def part3(quick: bool) -> None:
    n = 6000 if quick else 20000
    table_l1(n)
    table_l3(n, mem=not quick)


# ---------------------------------------------------------------------------
# Part 4


def table_r1(n: int) -> None:
    print(
        f"### Table R1: chunked, every state saved to bytes and loaded at every cut "
        f"({n} rows, 5 groups)\n"
    )
    print(
        "21 features: 3 bucketed windows (6 outputs each, 9 with the interpolated "
        "read), 11 EW features, 3 envelopes, 4 lags. A cut saves every group's state "
        "of every feature, drops it and loads it from the bytes; the EW states are "
        "loaded into an object made with a wrong alpha (0.5), so the carried alpha "
        "must be read back. Compared with the uncut stream (all 21, to the bit) and "
        "with one Polars pass (the 11 EW features and 4 lags on every row; the "
        "bucketed count, min and max on the rows where the buckets are the span "
        "exactly).\n"
    )
    rows = []
    for nonfinite in (False, True):
        for r in rres.invariance(n, nonfinite):
            rows.append(["NaN and infinities" if nonfinite else "nulls, zero weights", *r])
    print(
        md(
            [
                "frame",
                "chunking",
                "chunks",
                "features",
                "equal to the uncut stream",
                "equal to Polars to the bit",
                "seconds",
            ],
            rows,
        )
    )
    print()


def table_c1(quick: bool) -> None:
    n_py, n_pl = (10_000, 200_000) if quick else (50_000, 1_000_000)
    print(
        f"### Table C1: cost per row (best of 3; the Python state machine over "
        f"{n_py} rows of one group; Polars over {n_pl} rows of one group, the "
        f"matching expression, or for the envelopes the nearest, `rolling_max(20)`; "
        f"for the bucketed rows, the five `rolling_*_by` over the same span)\n"
    )
    print(md(["feature", "Python state, ns/row", "Polars, ns/row"], rres.cost(n_py, n_pl)))
    print()


def part4(quick: bool) -> None:
    table_r1(2000 if quick else 5000)
    table_c1(quick)


# ---------------------------------------------------------------------------
# Part 5


def part5(quick: bool) -> None:
    print(
        "### Table L5: where the lag size checks cross, per group "
        "(cap 256 KB = 262144 bytes, warn at 131072)\n"
    )
    print(
        "The count at the threshold is the largest the state allows at or under it; "
        "the bytes are read from `to_bytes()` of a real state at that count and one "
        "past it (columns times one column's bytes).\n"
    )
    print(
        md(
            [
                "columns",
                "lag",
                "count at 128 KB",
                "bytes there",
                "bytes one past",
                "count at 256 KB",
                "bytes there",
                "bytes one past",
            ],
            rchk.thresholds(),
        )
    )
    print()
    print("### Table L6: the clock lag's saved bytes by rows held (one column, nulls included)\n")
    print(
        md(
            [
                "rows held",
                "saved bytes",
                "8 + 16 m + ceil(m/8)",
                "bytes added per row (difference)",
            ],
            rchk.held_row_bytes(),
        )
    )
    print()
    print("### Table L7: 4 columns x 1000 groups, at each threshold\n")
    print(
        md(
            [
                "threshold",
                "row lag k",
                "bank state (1000 groups)",
                "clock lag rows",
                "bank state (1000 groups)",
            ],
            rchk.example_totals(),
        )
    )
    print()
    print(
        "### Table L8: a clock lag of one day (86400 s) on streams of one row per interval, "
        "the check run on the stream (two days, or until refused)\n"
    )
    print(
        md(
            [
                "rate",
                "columns",
                "rows held at steady state",
                "bytes",
                "first row warned",
                "first row refused",
                "verdict",
            ],
            rchk.day_examples(quick),
        )
    )
    print()
    n = 100_000 if quick else 300_000
    print(
        f"### Table L9: per-row cost of the running check (best of 11 over {n} rows of one "
        "group, off and on alternating; ns a row of `ClockLag.step`, Python 3.13)\n"
    )
    print(
        md(
            [
                "rows held",
                "step, check off (append + eviction)",
                "step, check on",
                "the check adds",
                "relative",
            ],
            rchk.check_cost(n),
        )
    )
    print()
    print(
        md(
            ["piece (ns a row, loop overhead taken off)", "ns"],
            rchk.comparison_cost(500_000 if quick else 2_000_000),
        )
    )
    print()
    print("### Table L10: the threshold tests (`studies/study_532b/test_lag_checks.py`)\n")
    print(md(["test", "result"], rchk.test_results()))
    print()


def main(argv: list[str]) -> None:
    quick = "--quick" in argv
    parts = [a for a in argv if a.startswith("--part=")]
    want = {p.split("=", 1)[1] for p in parts} or {"1", "2", "3", "4", "5"}
    t0 = time.time()
    versions()
    if "1" in want:
        part1(quick)
    if "2" in want:
        part2(quick)
    if "3" in want:
        part3(quick)
    if "4" in want:
        part4(quick)
    if "5" in want:
        part5(quick)
    print(f"(all tables: {time.time() - t0:.0f} s)", file=sys.stderr)
