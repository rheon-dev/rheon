"""Every table of docs/records/STUDY-532-FEATURES.md."""

from __future__ import annotations

import math
import platform
import time
from collections.abc import Callable

import numpy as np
import polars as pl

from . import features as F
from .oracle import compare, make_frame, masked
from .runs import KINDS, Unit, check_all, frame, polars_outputs, random_cuts, stream, units


def _table(head: list[str], rows: list[list[object]]) -> None:
    print("| " + " | ".join(head) + " |")
    print("|" + "|".join("---" for _ in head) + "|")
    for r in rows:
        print("| " + " | ".join(str(c) for c in r) + " |")
    print()


def versions() -> None:
    print("## Versions\n")
    _table(
        ["python", "numpy", "polars", "machine"],
        [
            [
                platform.python_version(),
                np.__version__,
                pl.__version__,
                f"{platform.system()} {platform.machine()}",
            ]
        ],
    )


# ---------------------------------------------------------------------------
# Table 1 and 2: equality with Polars, one pass and chunked.


def equality(n: int) -> None:
    us = units()
    print(f"## Table 1: each output against Polars, one pass, {n} rows, 5 groups (`over`)\n")
    print("Nulls about 3% of x, zero-weight rows about 4% (masked to null), and in the")
    print("second column set NaN, +inf and -inf about 0.4% of x.\n")
    clean, _ = check_all(frame(n), us, None)
    dirty, _ = check_all(frame(n, nonfinite=True), us, None)
    rows = []
    for lab in clean:
        c, d = clean[lab], dirty[lab]
        rows.append(
            [
                lab,
                c["valid"],
                c["validity_diff"],
                c["bit_diff"],
                c["max_ulp"],
                d["valid"],
                d["validity_diff"],
                d["bit_diff"],
            ]
        )
    _table(
        [
            "output",
            "valid",
            "validity diff",
            "bit diff",
            "max ulp",
            "valid (non-finite)",
            "validity diff",
            "bit diff",
        ],
        rows,
    )


def chunking(n: int) -> None:
    us = units()
    df = frame(n, nonfinite=True)
    theirs = polars_outputs(df, us)
    one, _ = stream(df, us, None)
    print(
        "## Table 2: chunked, the state saved to bytes and loaded at every cut "
        f"({n} rows, 5 groups)\n"
    )
    print("A cut saves every group's state of every feature to bytes, drops it and loads it")
    print("from the bytes. Compared with one Polars pass over the whole frame, and with the")
    print("uncut stream.\n")
    rows = []
    plans: list[tuple[str, list[int]]] = [
        ("1 chunk", []),
        ("2 chunks", [n // 2]),
        ("17 random chunks", random_cuts(n, 16, 7)),
        ("~300 random chunks", random_cuts(n, 300, 8)),
        ("every row", list(range(1, n))),
    ]
    for name, cuts in plans:
        t0 = time.perf_counter()
        ours, peak = stream(df, us, cuts)
        secs = time.perf_counter() - t0
        vs_p = [compare(ours[lab], theirs[lab]) for lab in ours]
        same_as_uncut = sum(
            1
            for lab in ours
            if all(
                (a is None and b is None)
                or (
                    a is not None
                    and b is not None
                    and (a == b or (a != a and b != b))
                    and math.copysign(1, a) == math.copysign(1, b)
                )
                for a, b in zip(ours[lab], one[lab], strict=False)
            )
        )
        rows.append(
            [
                name,
                len(cuts) + 1,
                len(ours),
                sum(1 for r in vs_p if r["validity_diff"] == 0 and r["bit_diff"] == 0),
                same_as_uncut,
                f"{secs:.1f}",
            ]
        )
    _table(
        [
            "chunking",
            "chunks",
            "outputs",
            "equal to Polars to the bit",
            "equal to the uncut stream",
            "seconds",
        ],
        rows,
    )


# ---------------------------------------------------------------------------
# Table 3: what a state that does not mirror Polars' order gives, in ulps.


def _ulp(a: float, b: float) -> int:
    def o(v: float) -> int:
        i = int(np.float64(v).view(np.int64))
        return -(i & 0x7FFFFFFFFFFFFFFF) if i < 0 else i

    return abs(o(a) - o(b))


def _gap(ours: list[float | None], ref: list[float | None]) -> tuple[int, float, str]:
    """Largest ulp gap, the median, and the largest relative gap, over rows
    where both are finite."""
    pairs = [
        (a, b)
        for a, b in zip(ours, ref, strict=False)
        if a is not None and b is not None and math.isfinite(a) and math.isfinite(b)
    ]
    if not pairs:
        return 0, 0.0, "0"
    gaps = [_ulp(a, b) for a, b in pairs]
    rel = max(abs(a - b) / abs(b) if b else (0.0 if a == b else math.inf) for a, b in pairs)
    return max(gaps), float(np.median(gaps)), f"{rel:.1e}"


def _plain_sum(xs: list[float], w: int) -> list[float | None]:
    out: list[float | None] = []
    s = 0.0
    for i, x in enumerate(xs):
        s += x
        if i >= w:
            s -= xs[i - w]
        out.append(s if i >= w - 1 else None)
    return out


def _fsum_window(xs: list[float], w: int) -> list[float | None]:
    return [math.fsum(xs[i + 1 - w : i + 1]) if i >= w - 1 else None for i in range(len(xs))]


def _welford_var(xs: list[float], w: int) -> list[float | None]:
    """Add and remove with Welford's update."""
    out: list[float | None] = []
    n, m, m2 = 0, 0.0, 0.0
    for i, x in enumerate(xs):
        n += 1
        d = x - m
        m += d / n
        m2 += d * (x - m)
        if i >= w:
            y = xs[i - w]
            n -= 1
            d = y - m
            m -= d / n
            m2 -= d * (y - m)
        out.append(m2 / (n - 1) if i >= w - 1 else None)
    return out


def _sumsq_var(xs: list[float], w: int) -> list[float | None]:
    out: list[float | None] = []
    s = q = 0.0
    for i, x in enumerate(xs):
        s += x
        q += x * x
        if i >= w:
            y = xs[i - w]
            s -= y
            q -= y * y
        out.append((q - s * s / w) / (w - 1) if i >= w - 1 else None)
    return out


def _exact_var(xs: list[float], w: int) -> list[float | None]:
    """Correctly rounded to within an ulp or two: exact sums (fsum) in two passes."""
    out: list[float | None] = []
    for i in range(len(xs)):
        if i < w - 1:
            out.append(None)
            continue
        win = xs[i + 1 - w : i + 1]
        m = math.fsum(win) / w
        out.append(math.fsum((v - m) * (v - m) for v in win) / (w - 1))
    return out


def _ewm_simple(xs: list[float], alpha: float) -> list[float | None]:
    out: list[float | None] = []
    y = xs[0]
    for i, x in enumerate(xs):
        if i:
            y = y + alpha * (x - y)
        out.append(y)
    return out


def _ewm_by_exp(xs: list[float], ts: list[int], hl: int) -> list[float | None]:
    """``0.5 ** (dt / hl)`` written as ``exp(dt * ln(0.5) / hl)``: what a
    different libm, or a different but equal formula, does to the bits."""
    out: list[float | None] = []
    prev, pt = xs[0], ts[0]
    out.append(prev)
    ln_half = math.log(0.5)
    for x, t in zip(xs[1:], ts[1:], strict=False):
        if x != prev:
            oma = math.exp(float(t - pt) * ln_half / float(hl))
            prev = oma * prev + (1.0 - oma) * x
        pt = t
        out.append(prev)
    return out


def _ewm_by_bumped(xs: list[float], ts: list[int], hl: int, seed: int = 9) -> list[float | None]:
    """The mirrored ``ewm_mean_by`` with each ``0.5 ** (dt / hl)`` moved by one
    ulp, up or down, on a random half of the rows: a libm that differs in the
    last bit."""
    rng = np.random.default_rng(seed)
    flips = rng.integers(0, 4, len(xs)).tolist()
    out: list[float | None] = []
    prev, pt = xs[0], ts[0]
    out.append(prev)
    for x, t, f in zip(xs[1:], ts[1:], flips[1:], strict=False):
        if x != prev:
            oma = 0.5 ** (float(t - pt) / float(hl))
            if f == 1:
                oma = math.nextafter(oma, 2.0)
            elif f == 2:
                oma = math.nextafter(oma, -1.0)
            prev = oma * prev + (1.0 - oma) * x
        pt = t
        out.append(prev)
    return out


def _resumed_from_ring(xs: list[float], w: int, every: int) -> list[float | None]:
    """The Kahan window sum, but at each resume the sum is rebuilt from the
    ring (the w values) and the error terms start at 0: the state a ring alone
    would carry."""
    out: list[float | None] = []
    k = F.KahanSum()
    get = xs.__getitem__
    for i in range(len(xs)):
        if i and i % every == 0 and i >= w:
            k = F.KahanSum()
            k.update(get, i - w, i)
        e = i + 1
        k.update(get, max(0, e - w), e)
        out.append(k.get_sum() if i >= w - 1 else None)
    return out


def gaps(n: int) -> None:
    print(
        "## Table 3: forms that do not keep Polars' order of operations, in ulps "
        f"({n} rows, one series, no nulls)\n"
    )
    print("Against the Polars expression (the state machines above are 0 ulps from it), and")
    print("against the exact value where one is computable (fsum). `level` is the series' mean")
    print("level; the steps are the same, so a higher level means more cancellation.\n")
    rows = []
    for level in (1.0e3, 1.0e6):
        df = make_frame(n, groups=1, null_p=0.0, zero_w_p=0.0, level=level)
        xs = df["x"].to_list()
        ts = df["t"].to_list()
        for w in (20, 500):
            ps = df.select(pl.col("x").rolling_sum(w)).to_series().to_list()
            pm = df.select(pl.col("x").rolling_mean(w)).to_series().to_list()
            pv = df.select(pl.col("x").rolling_var(w)).to_series().to_list()
            exact_s = _fsum_window(xs, w)
            exact_v = _exact_var(xs, w)
            plain = _plain_sum(xs, w)
            rows.append(
                [f"{level:g}", w, "rolling_sum: Polars vs exact (fsum)", *_gap(ps, exact_s)]
            )
            rows.append([f"{level:g}", w, "rolling_sum: plain add/sub vs Polars", *_gap(plain, ps)])
            rows.append(
                [f"{level:g}", w, "rolling_sum: plain add/sub vs exact", *_gap(plain, exact_s)]
            )
            rows.append(
                [
                    f"{level:g}",
                    w,
                    "rolling_sum: Kahan rebuilt from the ring every 97 rows vs Polars",
                    *_gap(_resumed_from_ring(xs, w, 97), ps),
                ]
            )
            rows.append(
                [
                    f"{level:g}",
                    w,
                    "rolling_mean: plain sum / w vs Polars",
                    *_gap([None if s is None else s / w for s in plain], pm),
                ]
            )
            rows.append([f"{level:g}", w, "rolling_var: Polars vs exact", *_gap(pv, exact_v)])
            rows.append(
                [
                    f"{level:g}",
                    w,
                    "rolling_var: Welford add/remove vs Polars",
                    *_gap(_welford_var(xs, w), pv),
                ]
            )
            rows.append(
                [
                    f"{level:g}",
                    w,
                    "rolling_var: Welford add/remove vs exact",
                    *_gap(_welford_var(xs, w), exact_v),
                ]
            )
            rows.append(
                [
                    f"{level:g}",
                    w,
                    "rolling_var: sum and sum of squares vs Polars",
                    *_gap(_sumsq_var(xs, w), pv),
                ]
            )
        a = F.alpha_of_half_life(10.0)
        pe = df.select(pl.col("x").ewm_mean(half_life=10.0, adjust=False)).to_series().to_list()
        rows.append(
            [
                f"{level:g}",
                "hl=10",
                "ewm_mean(adjust=False): y += a(x - y) vs Polars",
                *_gap(_ewm_simple(xs, a), pe),
            ]
        )
        pb = df.select(pl.col("x").ewm_mean_by("t", half_life="25i")).to_series().to_list()
        rows.append(
            [
                f"{level:g}",
                "hl=25i",
                "ewm_mean_by: exp(dt ln 0.5 / hl) for 0.5 ** (dt/hl) vs Polars",
                *_gap(_ewm_by_exp(xs, ts, 25), pb),
            ]
        )
    _table(["level", "window", "form", "max ulp", "median ulp", "max relative"], rows)


def ties(n: int) -> None:
    print("## Table 4: repeated times on the clock (`rolling_*_by`, closed right)\n")
    print("Polars gives each row of a repeated time the window of the last row at that time,")
    print("rows after it included. Read as of the row instead (`asof_ties`), the rows of a")
    print("repeat that are not its last differ; the last row of each repeat is equal.\n")
    rng = np.random.default_rng(11)
    step = rng.choice([0, 0, 1, 2, 3], n)
    step[0] = 0
    t = np.cumsum(step).astype(np.int64)
    x = 1000.0 + np.cumsum(rng.standard_normal(n))
    df = pl.DataFrame({"t": t, "xm": x})
    last_of_time = np.r_[t[1:] != t[:-1], True]
    spec = F.WindowSpec(span=40, closed="right", min_samples=1, asof_ties=True)
    roll = F.Rolling(spec, KINDS)
    outs = [roll.step(v, int(tt)) for v, tt in zip(x.tolist(), t.tolist(), strict=False)]
    rows = []
    for j, k in enumerate(KINDS):
        p = df.select(
            getattr(pl.col("xm"), f"rolling_{k}_by")("t", "40i", min_samples=1)
        ).to_series()
        ours = [o[j] for o in outs]
        r_all = compare(ours, p)
        r_last = compare(
            [o for o, keep in zip(ours, last_of_time, strict=True) if keep],
            p.filter(pl.Series(last_of_time)),
        )
        rows.append(
            [
                f"rolling_{k}_by",
                n,
                int((~last_of_time).sum()),
                r_all["bit_diff"],
                r_last["bit_diff"],
                f"{r_all['max_rel']:.2e}",
            ]
        )
    _table(
        [
            "output",
            "rows",
            "rows not last of their time",
            "bit diff, all rows",
            "bit diff, last rows of each time",
            "largest relative gap",
        ],
        rows,
    )


def libm(n: int) -> None:
    print("## Table 5: the features that read a libm result, and what one ulp of it moves\n")
    print("`ewm_*` read `alpha = 1 - exp(-ln 2 / half_life)`, computed once; `ewm_mean_by`")
    print("reads `0.5 ** (dt / half_life)` at every row. The same libm gives the same bits (the")
    print("tables above, on one machine). Another libm may differ in the last bit; the rows")
    print("below put one ulp into that result and measure the outputs.\n")
    df = make_frame(n, groups=1, null_p=0.0, zero_w_p=0.0)
    xs = df["x"].to_list()
    rows = []
    for hl in (3.0, 10.0, 100.0):
        e = math.exp(-math.log(2.0) / hl)  # the libm result; alpha = 1 - e
        a = 1.0 - e
        assert a == F.alpha_of_half_life(hl)
        for lab, e1 in (("+1 ulp", math.nextafter(e, 2.0)), ("-1 ulp", math.nextafter(e, -1.0))):
            for name, mk in (
                ("ewm_mean adjust=True", lambda al: F.EwmMean(al)),
                ("ewm_mean adjust=False", lambda al: F.EwmMean(al, adjust=False)),
                ("ewm_std adjust=True", lambda al: F.EwmVar(al, std=True)),
                ("ewm_var adjust=False bias", lambda al: F.EwmVar(al, adjust=False, bias=True)),
            ):
                s0, s1 = mk(a), mk(1.0 - e1)
                o0 = [s0.step(x) for x in xs]
                o1 = [s1.step(x) for x in xs]
                rows.append([name, f"hl={hl:g}", f"exp(-ln 2 / hl) {lab}", *_gap(o1, o0)])
    # ewm_mean_by: 1 - alpha by exp in place of pow, over a wide range of steps
    rng = np.random.default_rng(5)
    ts = np.cumsum(rng.integers(1, 400, n)).tolist()
    pb = (
        pl.DataFrame({"x": xs, "t": ts})
        .select(pl.col("x").ewm_mean_by("t", half_life="25i"))
        .to_series()
        .to_list()
    )
    st = F.EwmMeanBy(25)
    ours = [st.step(x, t) for x, t in zip(xs, ts, strict=False)]
    rows.append(["ewm_mean_by, mirrored (pow)", "hl=25i, dt 1..399", "none", *_gap(ours, pb)])
    rows.append(
        [
            "ewm_mean_by, exp(dt ln 0.5 / hl)",
            "hl=25i, dt 1..399",
            "the formula",
            *_gap(_ewm_by_exp(xs, ts, 25), pb),
        ]
    )
    rows.append(
        [
            "ewm_mean_by, each pow result moved 1 ulp at random",
            "hl=25i, dt 1..399",
            "+-1 ulp, p=1/2 a row",
            *_gap(_ewm_by_bumped(xs, ts, 25), pb),
        ]
    )
    omas = [(0.5 ** (d / 25), math.exp(d * math.log(0.5) / 25)) for d in range(1, 400)]
    rows.append(
        [
            "1 - alpha itself: pow vs exp form",
            "hl=25i, dt 1..399",
            "the formula",
            *_gap([o[1] for o in omas], [o[0] for o in omas]),
        ]
    )
    _table(["output", "parameters", "perturbation", "max ulp", "median ulp", "max relative"], rows)


# ---------------------------------------------------------------------------
# Table 5: the state's bytes.


def state_bytes() -> None:
    print("## Table 6: state bytes, saved form, per group and per column\n")
    print("`r` is the rows the ring holds: at most `w` for a row-counted window; for a span,")
    print("the rows inside it, at most twice that while windows short of `min_samples` are")
    print("skipped, and never above the hard cap. Measured: the largest saved size seen while")
    print("streaming 20000 rows with no nulls through one group, at a save every 37 rows.\n")
    df = masked(make_frame(20000, groups=1, null_p=0.0, zero_w_p=0.0))
    cuts = list(range(37, 20000, 37))
    rows = []
    formulas = {
        "shift": "8 + 8n + ceil(n/8)",
        "rolling": "28 + 8r + ceil(r/8) + 4 x 16 + 24 + 4",
        "rolling_by": "32 + 16r + ceil(r/8) + 4 x 16 + 24 + 4",
        "ewm_mean": "24",
        "ewm_var": "48",
        "ewm_mean_by": "25",
    }
    for u in units():
        _, peak = stream(df, [u], cuts, grouped=False)
        fam = u.name.split("[")[0].split("(")[0]
        fam = {"diff": "shift", "pct_change": "shift", "ewm_std": "ewm_var"}.get(fam, fam)
        n_or_w = ""
        pred = ""
        if fam == "shift":
            k = int(u.name.split("(")[1].rstrip(")"))
            n_or_w = f"n={k}"
            pred = str(8 + 8 * k + (k + 7) // 8)
        elif fam == "rolling":
            w = int(u.name.split("w=")[1].split(",")[0])
            n_or_w = f"w={w}"
            pred = str(28 + 8 * w + (w + 7) // 8 + 4 * 16 + 24 + 4)
        elif fam == "rolling_by":
            n_or_w = u.name.split("[")[1].split(",")[0]
            pred = "(r varies)"
        else:
            pred = formulas[fam]
        rows.append([u.name, n_or_w, formulas[fam], pred, peak[u.name]])
    _table(
        ["feature", "window", "bytes (formula)", "formula at full window", "measured peak"], rows
    )

    print("In memory the derived structures add, at most: the variance's two stacks hold")
    print("each window value once, as a (weight, mean, dp) triple of 24 bytes in the front")
    print("stack or an 8-byte value in the back one; the min and max deques hold at most `r`")
    print("(index, value) pairs each. Their worst case, a monotone series, measured:\n")
    w = 500
    roll = F.Rolling(F.WindowSpec(rows=w, min_samples=1), ("min", "max", "var"))
    peak_min = peak_max = peak_front = peak_back = 0
    for i in range(5000):
        roll.step(float(i), None)
        mn = roll.aggs["min"]
        mx = roll.aggs["max"]
        v = roll.aggs["var"]
        assert (
            isinstance(mn, F.MonoMinMax)
            and isinstance(mx, F.MonoMinMax)
            and isinstance(v, F.SwagVar)
        )
        peak_min = max(peak_min, len(mn.dq))
        peak_max = max(peak_max, len(mx.dq))
        peak_front = max(peak_front, len(v.front))
        peak_back = max(peak_back, len(v.back))
    _table(
        ["series", "w", "min deque peak", "max deque peak", "front stack peak", "back stack peak"],
        [["0, 1, 2, ... (rising)", w, peak_min, peak_max, peak_front, peak_back]],
    )


# ---------------------------------------------------------------------------
# Table 6: cost per row.


def _time(fn: Callable[[], object], reps: int = 3) -> float:
    best = math.inf
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def cost(n_py: int, n_pl: int) -> None:
    print(
        f"## Table 7: cost per row (best of 3; Python state machine over {n_py} rows, "
        f"Polars over {n_pl} rows, one group, then 5 groups with `over`)\n"
    )
    df_py = masked(make_frame(n_py, groups=1))
    df_pl = masked(make_frame(n_pl, groups=1)) if n_pl != n_py else df_py
    df_pl_g = masked(make_frame(n_pl, groups=5))
    rows = []
    for u in units():
        xs = df_py["xm"].to_list()
        ts = df_py["t"].to_list()

        def run_py(u: Unit = u, xs: list = xs, ts: list = ts) -> None:
            st = u.make()
            step = st.step  # type: ignore[attr-defined]
            for x, t in zip(xs, ts, strict=False):
                step(x, t)

        py = _time(run_py, 1) / n_py * 1e9
        exprs = [e.alias(lab) for lab, e in zip(u.labels, u.exprs, strict=False)]
        pol = _time(lambda exprs=exprs: df_pl.select(exprs)) / n_pl * 1e9
        exprs_g = [e.over("g").alias(lab) for lab, e in zip(u.labels, u.exprs, strict=False)]
        pol_g = _time(lambda exprs_g=exprs_g: df_pl_g.select(exprs_g)) / n_pl * 1e9
        rows.append([u.name, len(u.labels), f"{py:,.0f}", f"{pol:,.1f}", f"{pol_g:,.1f}"])
    _table(
        [
            "feature",
            "outputs",
            "Python state, ns/row",
            "Polars, ns/row",
            "Polars over 5 groups, ns/row",
        ],
        rows,
    )


def main(argv: list[str]) -> None:
    quick = "--quick" in argv
    versions()
    equality(4000 if quick else 20000)
    chunking(1500 if quick else 5000)
    gaps(3000 if quick else 20000)
    ties(2000 if quick else 20000)
    libm(3000 if quick else 20000)
    state_bytes()
    cost(5000 if quick else 50000, 100000 if quick else 1000000)
