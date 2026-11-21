"""A second pass over the claims of docs/records/STUDY-532-WINDOWLESS.md, each
measured a different way from the tables: strict bit equality (NaN payloads
and the sign of zero count), other seeds and levels, a brute-force reference
that shares no code with the state, the byte formulas from the saved forms, and
a fault injected into a load. ``python -m studies.study_532b.recheck`` prints
the numbers quoted under "Sonnet rerun" in the write-up (about 1 minute).
"""

from __future__ import annotations

import math
import sys
import time

import numpy as np
import polars as pl

from . import runs_bucket as rb
from . import runs_ew as rew
from . import runs_resume as rres
from .bucket import Bucketed, BucketSpec
from .common import make_frame, masked, stream_values, ulps
from .ew import DecayedMax, EwmMean, EwmMeanBy, EwmVar, PeakEnvelope, alpha_of_half_life
from .lag import Lag

CONFIGS = ((1, 60), (5, 12), (15, 4), (20, 12), (60, 4), (240, 1), (3, 7), (10, 2))


def strict(ours: list, theirs: pl.Series) -> tuple[int, int]:
    """(validity differences, bit differences by raw uint64): a NaN payload or
    the sign of a zero is a difference."""
    o_valid = np.array([v is not None for v in ours])
    o = np.array([math.nan if v is None else v for v in ours], dtype=np.float64)
    p_valid = theirs.is_not_null().to_numpy()
    p = theirs.fill_null(math.nan).to_numpy().astype(np.float64)
    both = o_valid & p_valid
    return int((o_valid != p_valid).sum()), int(
        (o[both].view(np.uint64) != p[both].view(np.uint64)).sum()
    )


def fmt2(c: tuple[int, int]) -> str:
    return f"{c[0]}/{c[1]}"


def a_group_by_dynamic() -> None:
    print("(a1) count/min/max against group_by_dynamic, strict bits, windows holding a value")
    print(
        "seed nonfinite b B | windows | windows missing either way | "
        "validity/bit diffs: count min max"
    )
    for seed in (532, 1, 2):
        for nonfinite in (False, True):
            df = make_frame(10000, seed=seed, nonfinite=nonfinite)
            gs, ts, xs = stream_values(df)
            m = masked(df).sort("g", "t")
            for b, B in CONFIGS:
                spec = BucketSpec(width=b, buckets=B, min_samples=0)
                reads = rb.tick_reads(gs, ts, xs, spec)
                keys = sorted(reads)
                pol = (
                    m.group_by_dynamic(
                        "t",
                        every=f"{b}i",
                        period=f"{B * b}i",
                        offset=f"-{(B - 1) * b}i",
                        closed="left",
                        label="left",
                        include_boundaries=True,
                        group_by="g",
                    )
                    .agg(
                        count=pl.col("xm").count().cast(pl.Float64),
                        min=pl.col("xm").min(),
                        max=pl.col("xm").max(),
                    )
                    .with_columns(k=pl.col("_lower_boundary") // b + (B - 1))
                )
                kd = pl.DataFrame(
                    {"g": [k[0] for k in keys], "k": [k[1] for k in keys]},
                    schema={"g": pl.Int64, "k": pl.Int64},
                )
                j = kd.join(pol, on=["g", "k"], how="left", maintain_order="left")
                missing_in_pol = int(j["count"].is_null().sum())
                ours_keys = set(keys)
                extra = pol.filter(
                    pl.struct("g", "k").map_elements(
                        lambda s, ks=ours_keys: (s["g"], s["k"]) not in ks, return_dtype=pl.Boolean
                    )
                )
                extra_v = int((extra["count"] > 0).sum())
                idx = [i for i, k in enumerate(keys) if reads[k]["count"] > 0]
                cells = []
                for s in ("count", "min", "max"):
                    cells.append(fmt2(strict([reads[keys[i]][s] for i in idx], j[s].gather(idx))))
                print(
                    f"{seed} {nonfinite!s:5} {b:3} {B:2} | {len(idx):6} | "
                    f"ours missing in polars {missing_in_pol}; polars not ours {extra.height} "
                    f"({extra_v}) | " + " ".join(cells)
                )


def a_b1_nan() -> None:
    print("(a4) B = 1, non-finite: which windows differ in min/max from group_by_dynamic")
    b, B = 240, 1
    for seed in (532, 1, 2):
        df = make_frame(10000, seed=seed, nonfinite=True)
        gs, ts, xs = stream_values(df)
        reads = rb.tick_reads(gs, ts, xs, BucketSpec(width=b, buckets=B, min_samples=0))
        pol = (
            masked(df)
            .sort("g", "t")
            .group_by_dynamic(
                "t",
                every=f"{b}i",
                period=f"{B * b}i",
                offset="0i",
                closed="left",
                label="left",
                include_boundaries=True,
                group_by="g",
            )
            .agg(
                mn=pl.col("xm").min(),
                mx=pl.col("xm").max(),
                has_nan=pl.col("xm").is_nan().any(),
                has_inf=pl.col("xm").is_infinite().any(),
            )
            .with_columns(k=pl.col("_lower_boundary") // b)
        )
        cnt = {"nan": [0, 0], "inf only": [0, 0], "finite": [0, 0]}
        for r in pol.iter_rows(named=True):
            o = reads[(r["g"], r["k"])]
            mism = any(
                not (o[s] == pv or (o[s] != o[s] and pv != pv))
                for s, pv in (("min", r["mn"]), ("max", r["mx"]))
                if o[s] is not None and pv is not None
            )
            kind = "nan" if r["has_nan"] else "inf only" if r["has_inf"] else "finite"
            cnt[kind][0] += 1
            cnt[kind][1] += mism
        print(
            f"seed {seed}: windows (mismatching) "
            + ", ".join(f"{k} {v[0]} ({v[1]})" for k, v in cnt.items())
        )


def a_rolling() -> None:
    print("(a2) count/min/max on aligned rows (t mod b = b-1) against rolling_*_by, strict bits")
    for seed in (532, 1):
        for nonfinite in (False, True):
            df = make_frame(10000, seed=seed, nonfinite=nonfinite)
            gs, ts, xs = stream_values(df)
            for b, B in CONFIGS:
                spec = BucketSpec(width=b, buckets=B, min_samples=1)
                ours, _, _ = rb.run(gs, ts, xs, spec)
                pol = rb.polars_rolling(df, B * b)
                rows = [i for i, t in enumerate(ts) if t % b == b - 1]
                cells = []
                for s in ("count", "min", "max"):
                    cells.append(fmt2(strict([ours[i][s] for i in rows], pol[s].gather(rows))))
                print(
                    f"{seed} {nonfinite!s:5} {b:3} {B:2} | {len(rows):5} rows | " + " ".join(cells)
                )


def a_brute() -> None:
    print("(a3) the B-bucket read against a brute-force numpy window [(k-B+1)b, t], every row")
    for nonfinite in (False, True):
        df = make_frame(6000, nonfinite=nonfinite)
        gs, ts, xs = stream_values(df)
        for b, B in ((5, 12), (20, 3), (60, 4), (240, 1)):
            spec = BucketSpec(width=b, buckets=B, min_samples=1)
            ours, _, _ = rb.run(gs, ts, xs, spec)
            by_g: dict[int, list[int]] = {}
            for i, g in enumerate(gs):
                by_g.setdefault(g, []).append(i)
            bad = 0
            maxulp_sum = 0
            for idxs in by_g.values():
                tg = np.array([ts[i] for i in idxs])
                for pos, i in enumerate(idxs):
                    k = ts[i] // b
                    lo = (k - B + 1) * b
                    vals = [
                        xs[idxs[q]]
                        for q in range(pos + 1)
                        if tg[q] >= lo and xs[idxs[q]] is not None
                    ]
                    o = ours[i]
                    if len(vals) == 0:
                        bad += o["count"] != 0.0 or o["min"] is not None
                        continue
                    a = np.array(vals)
                    mn, mx = (np.nan, np.nan) if np.isnan(a).any() else (a.min(), a.max())
                    if o["count"] != len(vals):
                        bad += 1
                    elif np.isnan(a).any():
                        bad += not (o["min"] != o["min"] and o["max"] != o["max"])
                    else:
                        bad += not (o["min"] == mn and o["max"] == mx)
                        if np.isfinite(a).all():
                            f = math.fsum(vals)
                            maxulp_sum = max(
                                maxulp_sum, int(ulps(np.array([o["sum"]]), np.array([f]))[0])
                            )
            print(
                f"nonfinite={nonfinite} b={b} B={B}: rows {len(ts)}, count/min/max mismatches "
                f"{bad}, sum max ulp vs fsum (finite windows) {maxulp_sum}"
            )


def b_sums() -> None:
    print(
        "(b,c) bucketed and Polars against the exact value on group_by_dynamic windows: "
        "max ulp / max relative"
    )
    print("seed level b B | sum ours/polars | mean ours/polars | var ours/polars")
    for seed, level in ((532, 1e3), (1, 1e3), (2, 1e3), (532, 1e6), (532, 1e9)):
        df = make_frame(10000, seed=seed, level=level)
        for b, B in ((5, 12), (20, 12), (60, 4), (240, 1)):
            r = rb.vs_group_by_dynamic(df, b, B)
            cells = []
            for s in ("sum", "mean", "var"):
                o, p = r[s + "_ours_exact"], r[s + "_polars_exact"]
                cells.append(
                    f"{o['max_ulp']}({o['max_rel']:.1e})/{p['max_ulp']}({p['max_rel']:.1e})"
                )
            print(f"{seed} {level:g} {b} {B} | " + " | ".join(cells))


def b_hard_sum() -> None:
    print("(b') bucketed sum against fsum on adversarial sums (values +-1e16 and 1, 1e-3 mixed)")
    rng = np.random.default_rng(5)
    worst = 0
    for _ in range(300):
        n = int(rng.integers(5, 60))
        vals = list(rng.choice([1e16, -1e16, 1.0, 0.001, 3.3, -2.2], n) * rng.random(n))
        st = Bucketed(BucketSpec(width=10, buckets=4))
        for i, v in enumerate(vals):
            st.step(float(v), i)
        got = st.read()["sum"]
        lo = (st.head - 3) * 10
        window = [v for i, v in enumerate(vals) if i >= lo]
        f = math.fsum(window)
        worst = max(worst, abs(got - f) / max(abs(f), 1e-300) if f else abs(got))
    print(f"worst relative error over 300 adversarial series: {worst:.2e}")


def e_ew_strict() -> None:
    print(
        "(e) EW features against Polars, strict bits (validity diffs / bit diffs, of which "
        "both NaN), 20000 rows"
    )
    for seed in (532, 3, 4):
        for nonfinite in (False, True):
            df = make_frame(20000, seed=seed, nonfinite=nonfinite)
            pol = masked(df).select([e.over("g").alias(n) for n, _, e in rew.ew_features()])
            bad_v = bad_b = both_nan = 0
            signs = set()
            for name, f, _ in rew.ew_features():
                ours = rew.stream(df, f)
                v, b = strict(ours, pol[name])
                bad_v += v
                bad_b += b
                o = np.array([math.nan if x is None else x for x in ours])
                p = pol[name].fill_null(math.nan).to_numpy().astype(np.float64)
                ok = np.array([x is not None for x in ours]) & pol[name].is_not_null().to_numpy()
                d = ok & (o.view(np.uint64) != p.view(np.uint64))
                n_nan = d & np.isnan(o) & np.isnan(p)
                both_nan += int(n_nan.sum())
                signs |= {
                    (hex(a), hex(c))
                    for a, c in zip(
                        o[n_nan].view(np.uint64)[:2], p[n_nan].view(np.uint64)[:2], strict=True
                    )
                }
            print(
                f"seed {seed} nonfinite={nonfinite}: 11 outputs, validity diffs {bad_v}, "
                f"bit diffs {bad_b}, "
                f"of them both NaN {both_nan} {sorted(signs)}"
            )


def l_lags_strict() -> None:
    print("(l) lag states against Polars, strict bits, 20000 rows")
    from . import runs_lag as rl

    for nonfinite in (False, True):
        df = make_frame(20000, nonfinite=nonfinite)
        gs, _, xs = stream_values(df)
        tot_v = tot_b = 0
        for kind in ("shift", "diff", "pct_change"):
            for k in rl.KS:
                pol = masked(df).select(rl._expr(kind, k).over("g").alias("p"))["p"]
                states: dict[int, Lag] = {}
                out = [
                    states.setdefault(g, Lag(kind, k)).step(x) for g, x in zip(gs, xs, strict=True)
                ]
                v, b = strict(out, pol)
                tot_v += v
                tot_b += b
        print(f"nonfinite={nonfinite}: 9 outputs, validity diffs {tot_v}, bit diffs {tot_b}")


def f_bytes() -> None:
    print("(f) saved sizes, measured from to_bytes()")
    a = alpha_of_half_life(10)
    print(
        "ewm_mean",
        len(EwmMean(a).to_bytes()),
        "ewm_var",
        len(EwmVar(a).to_bytes()),
        "ewm_std",
        len(EwmVar(a, std=True).to_bytes()),
        "ewm_mean_by",
        len(EwmMeanBy(10).to_bytes()),
        "envelope",
        len(PeakEnvelope(a).to_bytes()),
        "decayed max",
        len(DecayedMax(a).to_bytes()),
    )
    _, ts, xs = stream_values(make_frame(3000, groups=1))
    used = []
    for st in (EwmMean(a), EwmVar(a), EwmMeanBy(10), PeakEnvelope(a), DecayedMax(a)):
        for t, x in zip(ts, xs, strict=True):
            st.step(x, t)
        used.append(len(st.to_bytes()))
    print("after 3000 rows (ewm_mean, ewm_var, ewm_mean_by, envelope, decayed max):", used)
    for k in (1, 5, 8, 9, 20, 64):
        lg = Lag("shift", k)
        for x in xs:
            lg.step(x)
        print(f"lag k={k}: {len(lg.to_bytes())} formula {8 + 8 * k + math.ceil(k / 8)}")
    for B in (1, 4, 12, 60):
        for keep in (False, True):
            sp = BucketSpec(width=5, buckets=B, keep_previous=keep)
            st = Bucketed(sp)
            for t, x in zip(ts, xs, strict=True):
                st.step(x, t)
            print(
                f"bucketed B={B} p={int(keep)}: {len(st.to_bytes())} "
                f"formula {8 + 48 * (B + int(keep))}"
            )


def g_resume() -> None:
    print("(g) Table R1 again, and with a fault injected into EwmMean.load (alpha not read back)")
    for label, rows in (
        ("clean", rres.invariance(5000, False)),
        ("non-finite", rres.invariance(5000, True)),
    ):
        print(label, [(r[0], r[3], r[4]) for r in rows])
    orig = EwmMean.load

    def faulty(self: EwmMean, data: bytes) -> None:
        keep = self.alpha
        orig(self, data)
        self.alpha = keep

    EwmMean.load = faulty  # type: ignore[method-assign]
    try:
        for label, nf in (("clean", False), ("non-finite", True)):
            print("FAULT", label, [(r[0], r[3], r[4]) for r in rres.invariance(5000, nf)])
    finally:
        EwmMean.load = orig  # type: ignore[method-assign]


def g_resume_strict() -> None:
    print(
        "(g'') chunked and resumed outputs against the uncut stream by raw bits (NaN sign included)"
    )
    import struct

    from .runs_lag import _cuts

    def raw(v: object) -> object:
        if v is None:
            return None
        if isinstance(v, tuple):
            return tuple(raw(u) for u in v)
        return struct.pack("<d", float(v))  # type: ignore[arg-type]

    n = 5000
    for nf in (False, True):
        df = make_frame(n, nonfinite=nf)
        gs, ts, xs = stream_values(df)
        feats = rres.features()
        un = {f.name: [raw(v) for v in rres.run_feature(f, gs, ts, xs, [0, n])[0]] for f in feats}
        for chunks in (2, 17, 300, n):
            cuts = list(range(n + 1)) if chunks == n else _cuts(n, chunks)
            eq = sum(
                [raw(v) for v in rres.run_feature(f, gs, ts, xs, cuts)[0]] == un[f.name]
                for f in feats
            )
            print(f"nonfinite={nf} chunks={chunks}: {eq} of {len(feats)} equal by raw bits")


def g_nan_poison() -> None:
    print(
        "(g') EW outputs that are valid and not NaN after a group's first NaN/inf, non-finite frame"
    )
    df = make_frame(5000, nonfinite=True)
    gs, ts, xs = stream_values(df)
    first_bad: dict[int, int] = {}
    for i, (g, x) in enumerate(zip(gs, xs, strict=True)):
        if x is not None and not math.isfinite(x) and g not in first_bad:
            first_bad[g] = i
    for name, f, _ in rew.ew_features():
        out = rew.stream(df, f)
        n_after = n_finite_after = 0
        for i, g in enumerate(gs):
            if g in first_bad and i > first_bad[g] + 1 and out[i] is not None:
                n_after += 1
                n_finite_after += math.isfinite(out[i])
        print(
            f"{name}: rows valid 2+ after the first non-finite {n_after}, "
            f"of them finite {n_finite_after}"
        )
    print("rows before each group's first non-finite value:", sorted(first_bad.values()), "of 5000")


def h_repeated_times() -> None:
    print("(h') repeated times: bucketed read at aligned rows against rolling_*_by")
    df = make_frame(6000).with_columns(t=(pl.col("t") // 2) * 2).sort("g", "t", maintain_order=True)
    gs, ts, xs = stream_values(df)
    b, B = 5, 12
    spec = BucketSpec(width=b, buckets=B, min_samples=1)
    ours, _, _ = rb.run(gs, ts, xs, spec)
    pol = rb.polars_rolling(df, B * b)
    last_of_time = [
        i for i in range(len(ts)) if i + 1 == len(ts) or (gs[i + 1], ts[i + 1]) != (gs[i], ts[i])
    ]
    aligned_all = [i for i, t in enumerate(ts) if t % b == b - 1]
    aligned_last = [i for i in last_of_time if ts[i] % b == b - 1]
    for label, rows in (
        ("every aligned row", aligned_all),
        ("aligned rows that are the last of their time", aligned_last),
    ):
        cells = [
            fmt2(strict([ours[i][s] for i in rows], pol[s].gather(rows)))
            for s in ("count", "min", "max")
        ]
        print(f"{label}: {len(rows)} rows, validity/bit diffs count min max: {' '.join(cells)}")


def main(argv: list[str]) -> None:
    t0 = time.perf_counter()
    for fn in (
        a_group_by_dynamic,
        a_b1_nan,
        a_rolling,
        a_brute,
        b_sums,
        b_hard_sum,
        e_ew_strict,
        l_lags_strict,
        f_bytes,
        g_resume,
        g_resume_strict,
        g_nan_poison,
        h_repeated_times,
    ):
        if argv and fn.__name__.split("_")[0] not in argv:
            continue
        fn()
        print(flush=True)
    print(f"(recheck: {time.perf_counter() - t0:.0f} s)")


if __name__ == "__main__":
    main(sys.argv[1:])
