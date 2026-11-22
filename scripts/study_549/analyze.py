"""PLAN 549 study (not a test): summarise .cache/study549/runs.csv."""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
df = pl.read_csv(ROOT / ".cache" / "study549" / "runs.csv", infer_schema_length=10000)
df = df.with_columns(
    form=pl.col("name").str.replace(r"^(hi|lo|iid)_", "").str.replace(r"_\d+$", ""),
    path_same=(pl.col("path_p") == pl.col("path_l")) & (pl.col("refits_p") == pl.col("refits_l")),
)
pl.Config.set_tbl_rows(200)
pl.Config.set_tbl_cols(30)
pl.Config.set_tbl_width_chars(250)


def cls(c: str) -> pl.Expr:
    return (
        pl.when(pl.col(c) == 0)
        .then(pl.lit("bits"))
        .when(pl.col(c) <= 1e-12)
        .then(pl.lit("<=1e-12"))
        .when(pl.col(c) <= 1e-6)
        .then(pl.lit("<=1e-6"))
        .otherwise(pl.lit(">1e-6"))
    )


def show(t: pl.DataFrame) -> None:
    t = t.with_columns(
        pl.col(c).map_elements(lambda x: "" if x is None else f"{x:.2g}", return_dtype=pl.String)
        for c, d in t.schema.items()
        if d == pl.Float64
    )
    print(t)


what = sys.argv[1] if len(sys.argv) > 1 else "q1"
if what == "q1":
    out = (
        df.group_by("set", "form", "start")
        .agg(
            n=pl.len(),
            path_differs=(~pl.col("path_same")).sum(),
            start_bits=(pl.col("st_ulp") == 0).sum(),
            start_gt1e12=(pl.col("st_rel") > 1e-12).sum(),
            start_gt1e6=(pl.col("st_rel") > 1e-6).sum(),
            start_med_ulp=pl.col("st_ulp").median(),
            out_bits=(pl.col("all_ulp") == 0).sum(),
            w1_max=pl.col("w1_rel").max(),
            w2_max=pl.col("w2_rel").max(),
            w3_max=pl.col("w3_rel").max(),
            last_med=pl.col("last_rel").median(),
            last_max=pl.col("last_rel").max(),
            last_ulp_med=pl.col("last_ulp").median(),
            end_theta_max=pl.col("end_rel").max(),
            end_theta_ulp_med=pl.col("end_ulp").median(),
        )
        .sort("set", "form", "start")
    )
    show(out)
elif what == "growth":
    # per run: does the variance difference shrink, persist or grow?
    g = df.filter(pl.col("w1_rel") > 0).with_columns(
        trend=pl.when(pl.col("last_rel") < 0.1 * pl.col("w1_rel"))
        .then(pl.lit("shrinks"))
        .when(pl.col("last_rel") > 10 * pl.col("w1_rel"))
        .then(pl.lit("grows"))
        .otherwise(pl.lit("persists"))
    )
    print(g.group_by("set", "start", "trend").len().sort("set", "start", "trend"))
    show(
        g.group_by("set", "form")
        .agg(
            (pl.col("w1_rel").log10().median()).alias("lg_w1"),
            (pl.col("w2_rel").log10().median()).alias("lg_w2"),
            (pl.col("w3_rel").log10().median()).alias("lg_w3"),
            (pl.col("last_rel").log10().median()).alias("lg_last"),
        )
        .sort("set", "form")
    )
elif what == "dump":
    print(df.filter(pl.col("name").str.contains(sys.argv[2])).select(sys.argv[3].split(",")))
if what == "q1b":
    # realistic sets, starts >= 500: split runs whose start fit took another
    # path (start theta apart by more than 1e-6) from the rest
    real = df.filter(pl.col("set").is_in(["hi", "lo"]), pl.col("start") >= 500)
    for sub, name in (
        (real.filter(pl.col("st_rel") <= 1e-6), "start within 1e-6 (same optimum)"),
        (real.filter(pl.col("st_rel") > 1e-6), "start apart > 1e-6 (another path)"),
    ):
        print(name)
        tab = (
            sub.group_by("form")
            .agg(
                n=pl.len(),
                st_rel_med=pl.col("st_rel").median(),
                st_ulp_med=pl.col("st_ulp").median(),
                w1_med=pl.col("w1_rel").median(),
                w2_med=pl.col("w2_rel").median(),
                w3_med=pl.col("w3_rel").median(),
                last_med=pl.col("last_rel").median(),
                last_ulp_med=pl.col("last_ulp").median(),
                last_ulp_p90=pl.col("last_ulp").quantile(0.9),
                last_max=pl.col("last_rel").max(),
                all_max=pl.col("all_rel").max(),
                end_rel_med=pl.col("end_rel").median(),
                end_ulp_med=pl.col("end_ulp").median(),
                ident_frac=(pl.col("ident_rows") / pl.col("reported_rows")).median(),
            )
            .sort("form")
        )
        show(tab)
if what == "q2":
    guards = {
        "none": (pl.lit(True), pl.lit(True)),
        "sound": (pl.col("sound_p"), pl.col("sound_l")),
        "obs>=1e-3": (pl.col("obs_p").fill_nan(-1) >= 1e-3, pl.col("obs_l").fill_nan(-1) >= 1e-3),
        "obs>=1e-2": (pl.col("obs_p").fill_nan(-1) >= 1e-2, pl.col("obs_l").fill_nan(-1) >= 1e-2),
        "gn>=1e-3": (pl.col("gn_p") >= 1e-3, pl.col("gn_l") >= 1e-3),
        "gn>=1e-2": (pl.col("gn_p") >= 1e-2, pl.col("gn_l") >= 1e-2),
        "lr>=6": (pl.col("lr_p") >= 6, pl.col("lr_l") >= 6),
        "lr>=13.8": (pl.col("lr_p") >= 13.8, pl.col("lr_l") >= 13.8),
        "lr>=13.8&obs>=1e-3": (
            (pl.col("lr_p") >= 13.8) & (pl.col("obs_p").fill_nan(-1) >= 1e-3),
            (pl.col("lr_l") >= 13.8) & (pl.col("obs_l").fill_nan(-1) >= 1e-3),
        ),
    }
    rows = []
    for g, (pp, pl_) in guards.items():
        t = (
            df.with_columns(
                pp=pp,
                pl_=pl_,
                kind=pl.when(pl.col("set") == "iid").then(pl.lit("iid")).otherwise(pl.lit("real")),
            )
            .group_by("kind", "start")
            .agg(
                n=pl.len(),
                withheld=(~pl.col("pl_")).mean(),
                decision_differs=(pl.col("pp") != pl.col("pl_")).sum(),
                passed=pl.col("pl_").sum(),
                pass_bits=(pl.col("pl_") & (pl.col("all_ulp") == 0)).sum(),
                pass_other_path=(pl.col("pl_") & (pl.col("st_rel") > 1e-6)).sum(),
                pass_last_ulp_med=pl.col("last_ulp").filter(pl.col("pl_")).median(),
                pass_all_rel_max=pl.col("all_rel").filter(pl.col("pl_")).max(),
            )
            .with_columns(guard=pl.lit(g))
        )
        rows.append(t)
    show(
        pl.concat(rows)
        .sort("kind", "start", "guard")
        .select("kind", "start", "guard", pl.exclude("kind", "start", "guard"))
    )
if what == "q3":
    a = pl.read_csv(
        ROOT / ".cache" / "study549" / "arch_q3.csv", infer_schema_length=10000
    ).with_columns(
        form=pl.col("name").str.replace(r"^(hi|lo|iid)_", "").str.replace(r"_\d+$", ""),
        short=pl.col("arch") - pl.col("ours"),
        agree=(pl.col("f_l") - pl.col("ours")).abs(),
        last_stop=pl.col("path_l").str.split("|").list.last().str.replace(r"\d+$", ""),
    )
    print("our f against arch's likelihood at the same theta: max |diff|", a["agree"].max())
    real = a.filter(pl.col("set").is_in(["hi", "lo"]))
    show(
        real.group_by("set", "form", "start")
        .agg(
            n=pl.len(),
            short_gt_1e3=(pl.col("short") > 1e-3).sum(),
            short_gt_1e2=(pl.col("short") > 1e-2).sum(),
            short_gt_0_1=(pl.col("short") > 0.1).sum(),
            short_med=pl.col("short").median(),
            short_max=pl.col("short").max(),
            ours_better=(pl.col("short") < -1e-3).sum(),
        )
        .sort("set", "form", "start")
    )
    show(
        real.group_by("last_stop").agg(
            n=pl.len(),
            short_med=pl.col("short").median(),
            short_max=pl.col("short").max(),
            gt_1e2=(pl.col("short") > 1e-2).sum(),
        )
    )
    show(
        a.filter(pl.col("set") == "iid")
        .group_by("form")
        .agg(
            n=pl.len(),
            short_med=pl.col("short").median(),
            short_max=pl.col("short").max(),
            gt_1e2=(pl.col("short") > 1e-2).sum(),
            ours_better=(pl.col("short") < -1e-3).sum(),
        )
        .sort("form")
    )
