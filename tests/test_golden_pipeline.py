"""T-W7: fixed numbers out of the whole pipeline, compared on every OS.

`crates/rheon-core/tests/golden.rs` pins the Rust core's arithmetic, but
nothing pinned what comes out of the *Polars* layer -- extraction, the
per-group fan-out, the diagnostics, the struct assembly. A divergence
introduced there, or by polars' own vectorized paths on a different CPU, would
be invisible until someone compared two machines by hand.

This is that comparison, made automatic: the numbers are committed, and CI runs
this file on ubuntu, macOS and Windows. Locally the agreement is exact; the
tolerance is what "the same answer on another platform" is allowed to mean.

The constants come from the current implementation, which is held
elsewhere to an oracle it cannot share a bug with: the ten regression models
to the numpy references in `tests/reference.py` and `tests/reference_paths.py`,
and the other kinds to the oracles `docs/TESTING.md`'s table lists (their
definitions, papers and second opinions) -- the same bargain `golden.rs`
makes. Regenerate only after confirming a change is intended:

    uv run python tests/test_golden_pipeline.py
"""

from datetime import datetime, timedelta

import numpy as np
import polars as pl

from rheon import polars as rp

TIER = "essential"

#: How far two platforms may disagree, relative. Different LLVM vectorization
#: and BLAS paths can reorder floating-point operations; a genuinely divergent
#: algorithm shows up far above this.
TOL = 1e-12

#: Rows sampled from the stream. Early, mid and late, so warmup, the clock gap
#: and the converged state are all represented.
PICKS = [25, 60, 119]


def stream(n: int = 120) -> pl.DataFrame:
    """One deterministic frame exercising every input path: two groups, an
    irregular clock with a long gap, nulls in a feature and in the target,
    varying weights, and a session break."""
    rng = np.random.default_rng(20260830)
    x0 = rng.standard_normal(n)
    x1 = rng.standard_normal(n) * 3.0 + 2.0
    y = 1.5 * x0 - 0.75 * x1 + 0.25 + 0.1 * rng.standard_normal(n)
    t = np.cumsum(np.where(np.arange(n) % 17 == 0, 9.0, 1.0))
    w = 0.5 + 0.5 * (np.arange(n) % 3)

    x1 = [None if i % 31 == 7 else v for i, v in enumerate(x1)]
    y = [None if i % 29 == 11 else v for i, v in enumerate(y)]
    # A class label for `ew_class`: which side of its mean `x0` falls on, with
    # a null every 13th row (scored, not learned from).
    label = [None if i % 13 == 5 else ("hi" if v > 0.0 else "lo") for i, v in enumerate(x0)]
    return pl.DataFrame(
        {
            "t": t,
            "x0": x0,
            "x1": x1,
            "y0": y,
            # A positive amount for `glm`'s Gamma family.
            "y0_sq": [None if v is None else v * v + 0.1 for v in y],
            "label": label,
            "w": w,
            "g": ["a" if i % 2 == 0 else "b" for i in range(n)],
            "session": ["m" if i < n // 2 else "n" for i in range(n)],
        }
    )


INF = float("inf")


def specs() -> list[dict]:
    common = dict(
        targets=["y0"],
        features=["x0", "x1"],
        clock="t",
        gap_cap=6.0,
        half_life=25.0,
        weight="w",
        group="g",
        min_weight=4.0,
    )
    return [
        rp.spec.ewridge(
            "ridge",
            ridge=[1e-6, 0.5],
            standardize=True,
            max_rows_between_solves=1,
            emit_sigma=True,
            emit_zscore=True,
            conformal=0.9,
            **common,
        ),
        rp.spec.rls("rls", delta=1.0, **common),
        rp.spec.kalman("kalman", coef_half_life=80.0, **common),
        rp.spec.lasso("lasso", penalties=[0.2, 0.0], max_rows_between_solves=1, **common),
        rp.spec.huber("huber", max_rows_between_solves=1, **common),
        rp.spec.quantile("quantile", quantile=0.75, max_rows_between_solves=1, **common),
        rp.spec.sgd("sgd", learning_rate=0.02, **common),
        rp.spec.pa("pa", **common),
        # A Gamma amount: the target squared, refit on its 30th learned row.
        rp.spec.glm(
            "glm",
            family="gamma",
            refit_rows=30,
            **{**common, "targets": ["y0_sq"]},
        ),
        # An upper expectile, its first 30 learned rows refit exactly.
        rp.spec.expectile("expectile", expectile=0.85, refit_rows=30, **common),
        rp.spec.sgd("sgd_simplex", learning_rate=0.02, coef_min=0.0, coef_sum=1.0, **common),
        rp.spec.pa("pa_box", coef_min=-0.5, coef_max=0.5, **common),
        rp.spec.kalman(
            "kalman_revert", coef_half_life=80.0, revert_half_life=[INF, 30.0, 8.0], **common
        ),
        rp.spec.ftrl("ftrl", loss="squared", alpha=0.5, **common),
        rp.spec.holt(
            "holt",
            targets=["y0"],
            clock="t",
            gap_cap=6.0,
            half_life=25.0,
            weight="w",
            group="g",
            min_weight=4.0,
        ),
        # The target's power at two periods.
        rp.spec.spectral(
            "spectral",
            targets=["y0"],
            periods=[8.0, 3.0],
            clock="t",
            gap_cap=6.0,
            half_life=25.0,
            weight="w",
            group="g",
        ),
        # Two lags of the target with the features beside them.
        rp.spec.ar("ar", lags=[1, 2], ridge=0.01, **common),
        rp.spec.ew_cov(
            "moments",
            features=["x0", "x1"],
            stats=["mean", "var", "corr"],
            clock="t",
            gap_cap=6.0,
            half_life=25.0,
            weight="w",
            group="g",
            min_weight=4.0,
        ),
        rp.spec.kmeans(
            "kmeans",
            features=["x0", "x1"],
            n_clusters=2,
            warm_rows=8,
            split_merge=0.5,
            split_merge_every_rows=10,
            clock="t",
            gap_cap=6.0,
            half_life=25.0,
            weight="w",
            group="g",
            min_weight=4.0,
        ),
        rp.spec.micro(
            "micro",
            features=["x0", "x1"],
            eps=0.5,
            beta_mu=2.0,
            prune_every=10,
            clock="t",
            gap_cap=6.0,
            half_life=25.0,
            weight="w",
            group="g",
            min_weight=4.0,
        ),
        rp.spec.ew_class(
            "ew_class",
            features=["x0", "x1"],
            label="label",
            classes=["lo", "hi"],
            covariance_type="tied",
            precision_prior=0.5,
            clock="t",
            gap_cap=6.0,
            half_life=25.0,
            weight="w",
            group="g",
            min_weight=4.0,
        ),
        # Only `weight_sum` per row; the pairs are read from the state at the end.
        rp.spec.marginal("marginal", **common),
        # A constancy test over spans of 30 rows, so the 120-row stream
        # closes several per group.
        # A cap above the stream's 9-unit gaps: at `gap_cap = 6` every
        # gap is capped, `clear_lags` abandons the span (task 47), and no
        # span ever closes -- which is correct and pins nothing.
        rp.spec.corrchange(
            "corrchange",
            features=["x0", "x1"],
            window_rows=20,
            alpha=0.05,
            clock="t",
            gap_cap=100.0,
            group="g",
        ),
        # A hidden Markov model: the states are given, so the pinned
        # numbers are the filter's and not the seeding's.
        rp.spec.hmm(
            "hmm",
            features=["x0", "x1"],
            n_components=2,
            precision_prior=1e-2,
            learn=True,
            transition=[0.9, 0.1, 0.2, 0.8],
            means=[-1.0, -1.0, 1.0, 1.0],
            covs=[1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
            clock="t",
            gap_cap=6.0,
            half_life=25.0,
            weight="w",
            group="g",
        ),
        # A block's realised covariance. The stream's groups interleave, so
        # the close is on the session, not monotone; the block itself is
        # pinned below, since nothing but `weight_sum` is emitted per row.
        rp.spec.rcov(
            "rcov",
            features=["x0", "x1"],
            kind="kernel",
            bandwidth=3,
            clock="t",
            gap_cap=6.0,
            group="g",
            group_close="session",
            session="session",
        ),
        # One number for the whole correlation matrix. The stream has two
        # feature columns, so this is the unblocked form; the block path is
        # pinned by the core golden and by the numpy oracle in test_deco.py.
        # `rho` and `loglik` re-pinned for PLAN 551's ratio level.
        rp.spec.deco(
            "deco",
            features=["x0", "x1"],
            dynamics="linear",
            alpha=0.05,
            beta=0.9,
            clock="t",
            gap_cap=6.0,
            half_life=25.0,
            weight="w",
            group="g",
            min_weight=4.0,
        ),
        # A posterior over run lengths. Weighted rows scale each run's
        # statistics, and a fractional weight is legal here, so the golden
        # pins the weighted recursion and not just the plain one.
        rp.spec.bocpd(
            "bocpd",
            features=["x0", "x1"],
            hazard_rows=40.0,
            covariance_type="diag",
            covariance_prior=[1.0],
            prune_below=1e-8,
            max_run=50,
            clock="t",
            gap_cap=6.0,
            weight="w",
            group="g",
            min_weight=4.0,
        ),
        # No weight and no half-life: a test counts trials and does not forget.
        rp.spec.seqtest(
            "seqtest", targets=["y0"], clock="t", gap_cap=6.0, group="g", min_weight=4.0
        ),
        # What the columns hold: every row read, the null feature and the
        # null target among them; its counts are pinned below, since nothing
        # but `weight_sum` is emitted per row (docs/PLAN.md task 223 (b)).
        rp.spec.audit(
            "audit",
            columns=["x0", "x1", "y0"],
            pairs=True,
            clock="t",
            gap_cap=6.0,
            group="g",
        ),
        # Two nodes, each row where a column is above 0 an event of that mark,
        # two kernels, decayed; a weight only gates a row's events.
        rp.spec.hawkes(
            "hawkes",
            targets=["y0", "x0"],
            kernel_decays=[0.5, 2.0],
            clock="t",
            gap_cap=6.0,
            half_life=25.0,
            weight="w",
            group="g",
        ),
        # Two return series, GJR with a t after a start of 50 rows (a cap
        # too small to double it), a forecast three rows ahead; no decay.
        # The columns hold no conditional variance, so every start waits
        # (`below_warm_rows`, then `not_identified`; PLAN 548): a degenerate
        # case, kept; `garch`'s realistic ones are its goldens in Rust.
        rp.spec.garch(
            "garch",
            targets=["y0", "x0"],
            vol="gjr",
            dist="t",
            start_rows=50,
            max_state_bytes=2500,
            forecast_rows=[3],
            clock="t",
            gap_cap=6.0,
            weight="w",
            group="g",
        ),
        # Two series' correlation, the asymmetric model learned after a start
        # of 50 rows, on the targets as they stand (no margins). These two
        # hold one correlation, so its start does not identify a recursion
        # and waits (`not_identified`); the variant beside it gives the
        # parameters, and reports.
        rp.spec.dcc(
            "dcc",
            targets=["y0", "x0"],
            margins="none",
            asymmetric=True,
            start_rows=50,
            clock="t",
            gap_cap=6.0,
            half_life=25.0,
            weight="w",
            group="g",
        ),
        rp.spec.dcc(
            "dcc_given",
            targets=["y0", "x0"],
            margins="none",
            asymmetric=True,
            a=0.04,
            b=0.9,
            g=0.03,
            start_rows=20,
            clock="t",
            gap_cap=6.0,
            half_life=25.0,
            weight="w",
            group="g",
        ),
        # Two return series under a decay, Lee and Mykland's own window of
        # 20 returns.
        rp.spec.realized(
            "realized",
            targets=["y0", "x0"],
            lm_window=20,
            clock="t",
            gap_cap=6.0,
            half_life=25.0,
            weight="w",
            group="g",
        ),
        # Two components of the two features and the target, by the
        # incremental SVD: the target is the features' mix, so the
        # components differ and are written.
        rp.spec.pca(
            "pca",
            features=["x0", "x1", "y0"],
            n_components=2,
            clock="t",
            gap_cap=6.0,
            half_life=25.0,
            weight="w",
            group="g",
        ),
        # Two return series under a decay, a term of each kind: a sum of
        # squares, normalized bipower variation, a centered Parzen sum, a
        # flat-top Bartlett sum and squares of three-step returns.
        rp.spec.power_variation(
            "power_variation",
            targets=["y0", "x0"],
            terms={
                "rv": {"powers": [2]},
                "bv": {"powers": [1, 1], "normalize": True},
                "lrv": {"kernel": "parzen", "bandwidth": 4, "center": True},
                "rk": {"kernel": "bartlett", "bandwidth": 3, "flat_top": True},
                "sub": {"powers": [2], "scale": 3},
            },
            clock="t",
            gap_cap=6.0,
            half_life=25.0,
            weight="w",
            group="g",
        ),
        # A digest a target at a small compression, read at three levels
        # with their tail means, the tails withheld until the rows identify them; no decay.
        rp.spec.tdigest(
            "tdigest",
            targets=["y0", "x0"],
            quantiles=[0.05, 0.25, 0.95],
            compression=10.0,
            clock="t",
            gap_cap=6.0,
            weight="w",
            group="g",
        ),
        # A monotone fit on one feature, decreasing and clipped below, over a
        # clock window: a feature beyond its rows withheld.
        rp.spec.isotonic(
            "isotonic",
            targets=["y0"],
            features=["x0"],
            increasing=False,
            y_min=-4.0,
            window_size=150.0,
            clock="t",
            gap_cap=6.0,
            half_life=25.0,
            weight="w",
            group="g",
        ),
        # A sketch a target over a clock window, read at three levels with
        # their tail means, the tails withheld until the rows identify them.
        rp.spec.ddsketch(
            "ddsketch",
            targets=["y0", "x0"],
            quantiles=[0.05, 0.25, 0.95],
            alpha=0.01,
            window_size=150.0,
            clock="t",
            gap_cap=6.0,
            weight="w",
            group="g",
        ),
        # A skew, an unbiased fourth cumulant, a cross-moment and an
        # autocorrelation one clock unit back, each with its standard error.
        rp.spec.ewm_moments(
            "ewm_moments",
            terms={
                "skew": {"x0": 3},
                "k4": {"x0": 4},
                "co": {"x0": 1, "x1": 2},
                "ac1": {"x0": 1, "x0@1": 1},
            },
            kind={"skew": "standardized", "k4": "cumulant", "ac1": "standardized"},
            bias=False,
            se=True,
            clock="t",
            gap_cap=6.0,
            half_life=25.0,
            weight="w",
            group="g",
        ),
        # The two-phase bank: reads the ridge grid's `resid_y0__r0.5` and
        # kalman's `resid_y0` from the structs assembled above it.
        rp.spec.seqtest(
            "seqtest_compare",
            targets=["y0"],
            a="ridge",
            a_suffix="__r0.5",
            b="kalman",
            clock="t",
            gap_cap=6.0,
            group="g",
            min_weight=4.0,
        ),
    ]


#: An audit's column statistics the golden pins: every one that is not a
#: count, and the null count, which the extraction's null marker sets.
AUDIT_FIELDS = (
    "null_count",
    "mean",
    "std",
    "skew",
    "kurtosis",
    "median",
    "mad",
    "robust_z",
    "autocorr",
    "unit_root_t",
)


def signature() -> dict[str, float | str | None]:
    """Every non-coefficient output field, at three fixed rows; and, for a
    `marginal`, whose output is its state, every pair's derived values after
    the last row."""
    bank = rp.ModelBank(specs())
    out = bank.fit_predict(stream())
    sig: dict[str, float | str | None] = {}
    for spec in specs():
        name = spec["name"]
        for field in rp.spec.output_fields(spec):
            if field.startswith(("coef", "support_coef")):
                continue  # a list, and a reporting cadence rather than a value
            values = out[name].struct.field(field).to_list()
            for row in PICKS:
                sig[f"{name}.{field}@{row}"] = values[row]
        if spec["model"]["type"] == "marginal":
            for pair in bank.marginal(name).iter_rows(named=True):
                where = f"[{pair['group']}/{pair['feature']}]@end"
                for field in ("weight_sum", "n_kish", "corr", "beta", "t_stat"):
                    sig[f"{name}.{field}{where}"] = pair[field]
        # `corrchange` reports only where a span closes, and the picks
        # above are ordinary rows, so the statistics are pinned in the
        # order they were produced.
        if spec["model"]["type"] == "corrchange":
            stats = out[name].struct.field("stat").drop_nulls().to_list()
            for i, v in enumerate(stats):
                sig[f"{name}.stat#{i}"] = v
        # An `audit`'s value is its counts, read after the last row.
        if spec["model"]["type"] == "audit":
            for row in bank.audit(name).iter_rows(named=True):
                where = f"[{row['group']}/{row['column']}]@end"
                for field in AUDIT_FIELDS:
                    sig[f"{name}.{field}{where}"] = row[field]
            for row in bank.audit(name, table="pairs").iter_rows(named=True):
                where = f"[{row['group']}/{row['column_a']}/{row['column_b']}]@end"
                sig[f"{name}.corr{where}"] = row["corr"]
            for row in bank.audit(name, table="clock").iter_rows(named=True):
                for field in ("steps", "gaps", "step_cv"):
                    sig[f"{name}.{field}[{row['group']}]@end"] = row[field]
    # `rcov` emits nothing per row: its value is the block a group close
    # produces, so that is what is pinned.
    for row in bank.closed_groups(drop=False).iter_rows(named=True):
        where = f"[{row['spec']}/{row['group']}/{row['session']}]"
        sig[f"{row['spec']}.rcov_n{where}"] = row["rcov_n"]
        sig[f"{row['spec']}.bandwidth{where}"] = row["rcov_bandwidth_used"]
        for i, v in enumerate(row["rcov"] or []):
            sig[f"{row['spec']}.rcov{i}{where}"] = v
    return sig


def switched_stream(n: int = 120) -> pl.DataFrame:
    """`stream()` with what the second bank reads beside it: the clock as a
    `Datetime`, a minute for each unit of ``t``; a weight column with a row
    of weight 0 every seventh row; and a drifting level ``mid`` for the
    formula target."""
    df = stream(n)
    start = datetime(2024, 1, 2, 9, 30)
    ts = [start + timedelta(minutes=float(v)) for v in df["t"].to_list()]
    wz = [0.0 if i % 7 == 3 else v for i, v in enumerate(df["w"].to_list())]
    mid = 100.0 + np.cumsum(0.1 * df["x0"].to_numpy())
    return df.with_columns(
        pl.Series("ts", ts, dtype=pl.Datetime("us")), pl.Series("wz", wz), pl.Series("mid", mid)
    )


def switched_specs() -> list[dict]:
    """One spec per switch the first bank never turns (review 2026-10-06,
    TA12), each on the paths most likely to part by platform: a window, an
    embargo, a temporal clock read in integer nanoseconds with its
    parameters as durations, ``session_shrink``'s blend, pairwise gaps, no
    intercept, a coefficient prior, feature sets, every residual diagnostic,
    a formula target, rows of weight 0 (every spec reads ``wz``), and
    `bocpd`'s and `micro`'s clock parameters as durations. Clear of what task
    186 moves: no target that starts late in a solving model, no windowed
    `ew_class`, no `share_p`, no `rcov`."""
    common = dict(
        targets=["y0"],
        features=["x0", "x1"],
        clock="t",
        gap_cap=6.0,
        half_life=25.0,
        weight="wz",
        group="g",
        min_weight=4.0,
    )
    timed = dict(common, clock="ts", gap_cap="6m", half_life="25m")
    fwd = (rp.ewm_mean("mid", half_life=5.0, window_size=10.0, reverse=True) - pl.col("mid")).alias(
        "fwd"
    )
    unsupervised = {k: v for k, v in timed.items() if k not in ("targets", "min_weight")}
    return [
        # Under `closed="both"`, the edge every window had before task 196,
        # so its numbers are the ones pinned; the default `"right"` drops a
        # row exactly 20 old, moving `pred_y0@60` by 7.3e-5, and is held to
        # the definition in `test_window.py` and `test_second_opinion.py`.
        rp.spec.ewridge(
            "window", window_size=20.0, closed="both", max_rows_between_solves=1, **common
        ),
        rp.spec.kalman("embargo", coef_half_life=80.0, embargo=3.0, **common),
        rp.spec.ewridge("datetime", max_rows_between_solves=1, **timed),
        rp.spec.ewridge(
            "session_shrink",
            session="session",
            session_gap=3.0,
            session_shrink=0.5,
            long_half_life=100.0,
            max_rows_between_solves=1,
            **common,
        ),
        rp.spec.ewridge("pairwise", target_gaps="pairwise", max_rows_between_solves=1, **common),
        rp.spec.ewridge("origin", fit_intercept=False, max_rows_between_solves=1, **common),
        rp.spec.rls("prior", delta=1.0, coef_prior=[[0.2, 1.0, -0.5]], **common),
        rp.spec.ewridge(
            "feature_sets",
            feature_sets={"one": ["x0"], "both": ["x0", "x1"]},
            max_rows_between_solves=1,
            **common,
        ),
        rp.spec.ewridge(
            "diagnostics",
            ridge=[1e-6, 0.5],
            max_rows_between_solves=1,
            emit_sigma=True,
            emit_zscore=True,
            emit_drift=True,
            drift_threshold=20.0,
            resid_quantiles=[0.5, 0.9],
            emit_autocorr=True,
            emit_metrics=True,
            emit_selected=True,
            emit_averaged=True,
            emit_error_inflation=True,
            emit_clocks=True,
            **common,
        ),
        # A target resolved once its window has closed, so scored from early
        # on (no `min_weight`) to pin rows 25 and 60 as well as 119.
        rp.spec.kalman(
            "formula",
            coef_half_life=80.0,
            embargo=12.0,
            **{**common, "targets": [fwd], "min_weight": 0.0},
        ),
        rp.spec.bocpd(
            "hazard",
            hazard="40m",
            covariance_type="diag",
            covariance_prior=[1.0],
            max_run=50,
            **{k: v for k, v in unsupervised.items() if k != "half_life"},
        ),
        rp.spec.micro("prune", eps=0.5, beta_mu=2.0, prune_every="10m", **unsupervised),
    ]


def switched_signature() -> dict[str, float | str | None]:
    """Every non-coefficient output field of the second bank, at the same
    three rows."""
    bank = rp.ModelBank(switched_specs())
    out = bank.fit_predict(switched_stream())
    sig: dict[str, float | str | None] = {}
    for spec in switched_specs():
        name = spec["name"]
        for field in rp.spec.output_fields(spec):
            if field.startswith(("coef", "support_coef")):
                continue
            values = out[name].struct.field(field).to_list()
            for row in PICKS:
                value = values[row]
                # A `Datetime` clock field, pinned as its text.
                sig[f"{name}.{field}@{row}"] = str(value) if isinstance(value, datetime) else value
    return sig


#: Produced by `uv run python tests/test_golden_pipeline.py`. Re-pinned for
#: docs/PLAN.md task 195 (2026-10-07), whose defaults moved 45 of these and
#: 16 of `SWITCHED_GOLDEN`'s: `huber`'s `huber_delta` 1.345 (U3), `sgd`'s and
#: `pa`'s `standardize` on (U2) and `pa`'s tube in residual stds (U1), and
#: `bocpd`'s prior from the first rows with `nu` per emission (U4, U5). With
#: the old defaults given -- `huber_delta = 1.5`, `standardize = False`, a
#: zero mean, the identity and `nu = d + 2` -- every one of them but `pa`'s
#: and `pa_box`'s comes back to the old value within `TOL`; `pa`'s are a
#: replica of the standardized step with the sigma tube, written from the
#: docstrings, to 5.4e-16 on all 104 rows it predicts. Task 202 (the same
#: day) moved those 12 again, `pa`'s and `pa_box`'s `pred` and `resid`: the
#: tube is in units of the target's own EW std. A replica written from the
#: docstrings, the standardized PA-I step with that tube (its variance from
#: the definition) and, for `pa_box`, the box clamped in standardized
#: coordinates, gives `pa`'s to 1.2e-15 and `pa_box`'s to 5.6e-16 on all 104
#: rows each predicts; nothing else moved. Task 203 (2026-10-07) moved the
#: same 12 once more: `pa`'s `eps` defaults to 0.01, not 0.1. That replica at
#: 0.01 gives `pa`'s to 1.3e-15 and `pa_box`'s to 4.4e-16 on all 104 rows,
#: with the same nulls, and at 0.1 gives back the old 12 to 3.4e-15; nothing
#: else moved. Task 206 (2026-10-08) moved 24, the ``pred`` and ``resid`` at
#: rows 60 and 119 of ``sgd``, ``pa``, ``sgd_simplex``, ``pa_box``, ``kalman``
#: and ``kalman_revert``: past a standardizing fit's warm-up, 22 rows by
#: Kish's count of each group's weights, ``sgd``'s and ``pa``'s coefficients
#: are held in the caller's units and ``kalman``'s follow every move of the
#: moments (review round 5, G1); row 25 is inside the warm-up and kept its
#: bits. ``tests/reference_paths.py``'s ``sgd_ref`` and ``pa_ref`` and
#: ``tests/reference.py``'s ``kalman_ref`` under ``standardize``, written from
#: the docstrings with the moments by their definition, give every row the
#: six predict, 104 each, to 5.4e-16, 1.4e-15, 5.5e-16, 7.5e-16, 1.4e-14 and
#: 1.9e-15, with the same nulls; nothing else moved. Task 211 (2026-10-08)
#: moved 24: the ``pred`` and ``resid`` of ``kalman`` and ``kalman_revert``
#: at rows 25, 60 and 119 -- the process noise charged once per
#: observation for the clock since the last, each coefficient's prior
#: waiting for its feature's scale, and no warm-up (`kalman.rs`'s module
#: doc) -- and the 12 of ``seqtest_compare``, which reads ``kalman``'s
#: ``resid_y0`` as its ``b``: ``kalman``'s residual is now the smaller, and
#: the wins went to ``b``. ``tests/reference.py``'s ``kalman_ref``, rewritten
#: from the new docstring with the state re-mapped on every row, gives
#: ``kalman``'s and ``kalman_revert``'s every row, 104 each, to 8.4e-15 and
#: 1.4e-15, with the same nulls; nothing else moved. Task 214 (2026-10-08)
#: moved the 15 of ``hmm``: a row reads the transition matrix from counts its
#: own clock has decayed, ``(lam A + tau) / sum``, so that a zero-weight row is
#: clock alone (hard rule 9). A replica written from the docstrings -- each
#: group's clock capped at ``gap_cap`` with a null feature's clock carried,
#: Hamilton's filter, the states in mean form under the decaying prior --
#: gives the 15 to 4.8e-16, and under the old reading the old 15 to 2.2e-16;
#: nothing else moved. Task 214 again moved the 6 of ``kalman_revert``: a
#: reverting slot's process noise for a gap ``D`` is ``q ((1 - 2**(-D/r)) /
#: theta)**2``, bounded, where it was ``q D**2``. ``kalman_ref``, rewritten
#: from the new docstring, gives every row, 104, to 1.0e-15, with the same
#: nulls, and the old 3 ``pred`` under the old charge to 2.8e-16; nothing
#: else moved. Task 214's prior moved 26: a standardizing ``kalman`` sizes
#: each prior from the weighted median of the first three rows' squared
#: innovations over the median of a chi-squared of one degree, where it read
#: their mean over every row so far. The 6 of ``kalman`` and of
#: ``kalman_revert``: ``kalman_ref`` with the median gives every row, 104
#: each, to 1.7e-15 and 9.3e-16, with the same nulls. The 6 of
#: ``seqtest_compare``: ``seqtest_ref`` on ``|resid_kalman| -
#: |resid_ridge__r0.5|``, kalman's residual from that ``kalman_ref``, gives
#: the 106 rows both report exactly. In ``SWITCHED_GOLDEN``, the 6 of
#: ``embargo`` and 2 of ``formula``, ``kalman`` under an embargo and a
#: formula target, moved by the same prior; ``test_kalman.py`` holds the
#: native embargo to the doubled stream, whose ``kalman`` is the one above.
#: Task 223 (b) (2026-10-09) added the 75 lines of ``audit``, a new spec in
#: the same bank; nothing else moved. Its statistics are held to their
#: definitions in ``audit/tests.rs`` and to scipy and statsmodels in
#: ``tests/test_audit.py``. Task 243 (2026-10-11) moved 5, ``bocpd``'s at
#: row 119: the runs that reach ``max_run - 1 = 49`` rows fold into the
#: oldest of them, scaled back to 49 rows' weight, where they folded into
#: the youngest. Rows 25 and 60 come before either group has 49 rows (31 at
#: most), so no run can reach the cap and they kept their bits; by row 119
#: each group has 60. ``bocpd.rs``'s ``max_run_folds_the_tail`` holds the
#: scaled run to a replica of the step and the scale, and
#: ``the_fold_at_max_run_keeps_every_runs_mass`` the fold's mass and its
#: statistics to the bit; nothing else moved. Task 244 (2026-10-11) moved 6,
#: ``moments``' variances: ``ew_cov``'s default ``bias=False`` multiplies
#: them by ``W**2 / (W**2 - sum(w**2))``, 1.04 to 1.13 at these rows, which
#: ``tests/test_ew_cov_bias.py`` holds to Polars and numpy; its means and
#: correlations kept their bits, and nothing else moved. Task 278 (D-H,
#: 2026-10-26) moved 6, ``holt``'s predictions and residuals: a row takes the
#: larger of the weighted line fit's gains and the textbook's, so these rows,
#: inside the warm-up of a half-life of 25, are the line fit's, where the
#: weighted means pinned the trend to the first observation; ``holt.rs``'s
#: ``reference`` holds the step to its definition, ``tests/test_second_opinion.py``
#: to statsmodels' ``WLS`` and ``Holt``; nothing else moved. Task 278 (D-Q,
#: D10; 2026-10-26) moved 8, ``huber``'s and ``quantile``'s at rows 60 and
#: 119: ``quantile`` draws its band in the MAD, and ``min_weight`` withholds
#: their output alone, the fit learning from its own prediction from the
#: row its target holds a row per coefficient. The largest move 8.5e-3 of
#: ``1 + |v|`` (``quantile.pred_y0@60``); ``robust_ref`` holds both to every
#: row (``tests/test_oracles.py``); nothing else moved. Task 278's anchored
#: MAD sketch (2026-10-26) moved the same 8: each sketch keeps ``|e|`` in
#: units of the first, which regroups the residuals across bucket edges; the
#: largest move 9.2e-4 of ``1 + |v|`` (``quantile.pred_y0@119``);
#: ``robust_ref``'s sketch is anchored as the core's is, and holds both to
#: every row; nothing else moved. Task 278 (batch 17, 2026-10-26) moved 29,
#: ``ridge``'s ``sigma``, ``zscore`` and conformal ``lo``, ``hi`` and
#: ``coverage``: the residual spread folds each slot's own prediction on the
#: rows a readiness gate withholds (here the noise gate and ``min_weight``'s
#: first rows), so ``sigma`` starts from them and the conformal radius with
#: it: ``sigma`` by 48% at row 25 (the 1e-6 slot), 1% by row 60 and 0.2%
#: by 119; the interval more, its radius keeping its start for many rows;
#: ``tests/test_oracles_sigma.py`` holds ``sigma`` to its recursion over the
#: residuals of the same spec with no gate, and
#: ``crates/rheon-polars/tests/gate_isolation.rs`` to the gate never moving
#: it; nothing else moved. Task 323 (2026-11-05) moved the same 8,
#: ``huber``'s and ``quantile``'s at rows 60 and 119: while each refit waits
#: (500 rows here, past the stream) the MAD is the exact weighted median of
#: the buffered ``|e|``, where it was the sketch's read, within a bucket of
#: it; the largest move 1.5e-4 of ``1 + |v|`` (``quantile.pred_y0@60``);
#: ``robust_ref`` reads the same median and holds both to every row
#: (``tests/test_oracles.py``); nothing else moved. Task 368 (2026-11-09,
#: the user's decision) moved 16, ``ridge``'s conformal ``lo``, ``hi`` and
#: ``coverage`` of both slots: the tracker keeps a multiplier of ``sigma``,
#: so the radius is ``m·sigma`` for the ``sigma`` the row reads, ``m``
#: starting at ``Φ⁻¹(1 − α/2)`` on the first row with a residual, where the
#: radius was stepped by ``rate·sigma`` from ``sigma·Φ⁻¹(1 − α/2)``;
#: ``tests/test_conformal.py``'s replay, rewritten from the new docstring,
#: holds every row of every model to the bit; nothing else moved. Review C1
#: (2026-11-20) moved 24, the ``pred`` and ``resid`` of ``sgd``, ``pa``,
#: ``sgd_simplex`` and ``pa_box`` at all three rows: the scaler reads a row in
#: the stream's own unit of weight, ``Σw²/Σw`` of its warm-up's rows, where it
#: read every row at 1, and keeps that weight of history into a row, and this
#: stream's weights are 0.5, 1 and 1.5. ``tests/reference_paths.py``'s
#: ``sgd_ref`` and ``pa_ref`` with the new read give ``sgd``'s and ``pa``'s
#: three to 3.4e-16 and 8.9e-16.
#: Stabilized forgetting (2026-11-20, the user's decision; rule 5's
#: twenty-second exception) moved the 15 of ``hmm``, its posteriors and
#: ``loglik``: under ``"full"`` with ``learn = True`` each state reads
#: ``(W C + d diag(psi)) / (W + d)`` after the row's decay, where it read
#: ``C`` plus the decaying ridge. ``tests/test_second_opinion_hmm.py``
#: rebuilds that rule row by row with hmmlearn, scipy and numpy and holds
#: every row to it within its derived bound, and
#: ``tests/test_second_opinion_regimes.py`` the shrunk covariance to
#: scikit-learn's QDA; ``weight_sum``, ``settled_frac``, ``state`` and every
#: other spec kept their numbers within the tolerance.
GOLDEN: dict[str, float | str | None] = {
    "ridge.pred_y0__r0.000001@25": -4.684371132566456,
    "ridge.pred_y0__r0.000001@60": -0.25563207972202284,
    "ridge.pred_y0__r0.000001@119": -0.15976731834715135,
    "ridge.resid_y0__r0.000001@25": -0.1672346916682601,
    "ridge.resid_y0__r0.000001@60": -0.2164090490950551,
    "ridge.resid_y0__r0.000001@119": 0.10906854738892252,
    "ridge.pred_y0__r0.5@25": -3.2074053272438663,
    "ridge.pred_y0__r0.5@60": -0.1169734771432751,
    "ridge.pred_y0__r0.5@119": -0.5303952399350176,
    "ridge.resid_y0__r0.5@25": -1.64420049699085,
    "ridge.resid_y0__r0.5@60": -0.35506765167380283,
    "ridge.resid_y0__r0.5@119": 0.47969646897678875,
    "ridge.sigma_y0__r0.000001@25": 0.07641734354131406,
    "ridge.sigma_y0__r0.000001@60": 0.1483912149776585,
    "ridge.sigma_y0__r0.000001@119": 0.09350168089292761,
    "ridge.sigma_y0__r0.5@25": 2.8076126058896667,
    "ridge.sigma_y0__r0.5@60": 0.6858895190735657,
    "ridge.sigma_y0__r0.5@119": 0.8533874007852269,
    "ridge.zscore_y0__r0.000001@25": -2.188438957942143,
    "ridge.zscore_y0__r0.000001@60": -1.4583683348615835,
    "ridge.zscore_y0__r0.000001@119": 1.1664875577351506,
    "ridge.zscore_y0__r0.5@25": -0.5856222804890282,
    "ridge.zscore_y0__r0.5@60": -0.5176747009538715,
    "ridge.zscore_y0__r0.5@119": 0.5621086842100151,
    "ridge.lo_y0__r0.000001@25": -4.809053706335373,
    "ridge.lo_y0__r0.000001@60": -0.4982947539779955,
    "ridge.lo_y0__r0.000001@119": -0.31857434664426354,
    "ridge.lo_y0__r0.5@25": -7.788307379395447,
    "ridge.lo_y0__r0.5@60": -1.2389869348434963,
    "ridge.lo_y0__r0.5@119": -1.7946746031454988,
    "ridge.hi_y0__r0.000001@25": -4.559688558797539,
    "ridge.hi_y0__r0.000001@60": -0.012969405466050188,
    "ridge.hi_y0__r0.000001@119": -0.0009602900500391975,
    "ridge.hi_y0__r0.5@25": 1.3734967249077146,
    "ridge.hi_y0__r0.5@60": 1.005039980556946,
    "ridge.hi_y0__r0.5@119": 0.7338841232754636,
    "ridge.coverage_y0__r0.000001@25": 1.0,
    "ridge.coverage_y0__r0.000001@60": 0.9020291035701509,
    "ridge.coverage_y0__r0.000001@119": 0.8818947848586911,
    "ridge.coverage_y0__r0.5@25": 1.0,
    "ridge.coverage_y0__r0.5@60": 0.9207933957773561,
    "ridge.coverage_y0__r0.5@119": 0.9776214238148554,
    "ridge.weight_sum@25": 7.999488060097996,
    "ridge.weight_sum@60": 12.473100285951407,
    "ridge.weight_sum@119": 15.110060335371337,
    "ridge.settled_frac@25": 0.5136725262938573,
    "ridge.settled_frac@60": 0.8564127056253706,
    "ridge.settled_frac@119": 0.9782071302132749,
    "ridge.withheld_reason@25": None,
    "ridge.withheld_reason@60": None,
    "ridge.withheld_reason@119": None,
    "rls.pred_y0@25": -4.706391171521188,
    "rls.pred_y0@60": -0.24224285328032447,
    "rls.pred_y0@119": -0.16011065148415327,
    "rls.resid_y0@25": -0.14521465271352785,
    "rls.resid_y0@60": -0.22979827553675347,
    "rls.resid_y0@119": 0.10941188052592443,
    "rls.weight_sum@25": 7.999488060097996,
    "rls.weight_sum@60": 12.473100285951407,
    "rls.weight_sum@119": 15.110060335371337,
    "rls.settled_frac@25": 0.5136725262938573,
    "rls.settled_frac@60": 0.8564127056253706,
    "rls.settled_frac@119": 0.9782071302132749,
    "rls.withheld_reason@25": None,
    "rls.withheld_reason@60": None,
    "rls.withheld_reason@119": None,
    "kalman.pred_y0@25": -4.610536129963606,
    "kalman.pred_y0@60": -0.264290402740186,
    "kalman.pred_y0@119": -0.15393133292788796,
    "kalman.resid_y0@25": -0.24106969427111036,
    "kalman.resid_y0@60": -0.20775072607689193,
    "kalman.resid_y0@119": 0.10323256196965913,
    "kalman.weight_sum@25": 7.999488060097996,
    "kalman.weight_sum@60": 12.473100285951407,
    "kalman.weight_sum@119": 15.110060335371337,
    "kalman.settled_frac@25": 0.5136725262938573,
    "kalman.settled_frac@60": 0.8564127056253706,
    "kalman.settled_frac@119": 0.9782071302132749,
    "kalman.withheld_reason@25": None,
    "kalman.withheld_reason@60": None,
    "kalman.withheld_reason@119": None,
    "lasso.pred_y0__p0.2@25": -4.355626741152896,
    "lasso.pred_y0__p0.2@60": -0.1710572262432979,
    "lasso.pred_y0__p0.2@119": -0.26368839582410974,
    "lasso.resid_y0__p0.2@25": -0.4959790830818198,
    "lasso.resid_y0__p0.2@60": -0.30098390257378005,
    "lasso.resid_y0__p0.2@119": 0.2129896248658809,
    "lasso.pred_y0__p0@25": -4.68437561144585,
    "lasso.pred_y0__p0@60": -0.2556326067304031,
    "lasso.pred_y0__p0@119": -0.15976613727738184,
    "lasso.resid_y0__p0@25": -0.16723021278886652,
    "lasso.resid_y0__p0@60": -0.21640852208667483,
    "lasso.resid_y0__p0@119": 0.10906736631915301,
    "lasso.weight_sum@25": 7.999488060097996,
    "lasso.weight_sum@60": 12.473100285951407,
    "lasso.weight_sum@119": 15.110060335371337,
    "lasso.settled_frac@25": 0.5136725262938573,
    "lasso.settled_frac@60": 0.8564127056253706,
    "lasso.settled_frac@119": 0.9782071302132749,
    "lasso.withheld_reason@25": None,
    "lasso.withheld_reason@60": None,
    "lasso.withheld_reason@119": None,
    "lasso.penalty_selected_y0@25": 0.0,
    "lasso.penalty_selected_y0@60": 0.0,
    "lasso.penalty_selected_y0@119": 0.0,
    "huber.pred_y0@25": -4.684375830329454,
    "huber.pred_y0@60": -0.25295781642638193,
    "huber.pred_y0@119": -0.159536342178747,
    "huber.resid_y0@25": -0.16722999390526194,
    "huber.resid_y0@60": -0.219083312390696,
    "huber.resid_y0@119": 0.10883757122051815,
    "huber.weight_sum@25": 7.999488060097996,
    "huber.weight_sum@60": 12.473100285951407,
    "huber.weight_sum@119": 15.110060335371337,
    "huber.settled_frac@25": 0.5136725262938573,
    "huber.settled_frac@60": 0.8564127056253706,
    "huber.settled_frac@119": 0.9782071302132749,
    "huber.withheld_reason@25": None,
    "huber.withheld_reason@60": None,
    "huber.withheld_reason@119": None,
    "quantile.pred_y0@25": -4.684375830329454,
    "quantile.pred_y0@60": -0.22921308637891635,
    "quantile.pred_y0@119": -0.10994375923423028,
    "quantile.resid_y0@25": -0.16722999390526194,
    "quantile.resid_y0@60": -0.24282804243816158,
    "quantile.resid_y0@119": 0.05924498827600144,
    "quantile.weight_sum@25": 7.999488060097996,
    "quantile.weight_sum@60": 12.473100285951407,
    "quantile.weight_sum@119": 15.110060335371337,
    "quantile.settled_frac@25": 0.5136725262938573,
    "quantile.settled_frac@60": 0.8564127056253706,
    "quantile.settled_frac@119": 0.9782071302132749,
    "quantile.withheld_reason@25": None,
    "quantile.withheld_reason@60": None,
    "quantile.withheld_reason@119": None,
    "sgd.pred_y0@25": -0.6943372278927147,
    "sgd.pred_y0@60": -0.6151161326150822,
    "sgd.pred_y0@119": -0.5491405417436057,
    "sgd.resid_y0@25": -4.157268596342002,
    "sgd.resid_y0@60": 0.1430750037980043,
    "sgd.resid_y0@119": 0.4984417707853769,
    "sgd.weight_sum@25": 7.999488060097996,
    "sgd.weight_sum@60": 12.473100285951407,
    "sgd.weight_sum@119": 15.110060335371337,
    "sgd.settled_frac@25": 0.5136725262938573,
    "sgd.settled_frac@60": 0.8564127056253706,
    "sgd.settled_frac@119": 0.9782071302132749,
    "sgd.withheld_reason@25": None,
    "sgd.withheld_reason@60": None,
    "sgd.withheld_reason@119": None,
    "glm.pred_y0_sq@25": 16.58841226191769,
    "glm.pred_y0_sq@60": 1.0436918411103409,
    "glm.pred_y0_sq@119": 41.87477892954526,
    "glm.resid_y0_sq@25": 7.049666811830534,
    "glm.resid_y0_sq@60": -0.7208690138154397,
    "glm.resid_y0_sq@119": -41.77220856416858,
    "glm.weight_sum@25": 7.999488060097996,
    "glm.weight_sum@60": 12.473100285951407,
    "glm.weight_sum@119": 15.110060335371337,
    "glm.settled_frac@25": 0.5136725262938573,
    "glm.settled_frac@60": 0.8564127056253706,
    "glm.settled_frac@119": 0.9782071302132749,
    "glm.withheld_reason@25": None,
    "glm.withheld_reason@60": None,
    "glm.withheld_reason@119": None,
    "expectile.pred_y0@25": -4.67359086369818,
    "expectile.pred_y0@60": -0.23565082136225857,
    "expectile.pred_y0@119": -0.09977433354758529,
    "expectile.resid_y0@25": -0.17801496053653576,
    "expectile.resid_y0@60": -0.23639030745481937,
    "expectile.resid_y0@119": 0.049075562589356456,
    "expectile.weight_sum@25": 7.999488060097996,
    "expectile.weight_sum@60": 12.473100285951407,
    "expectile.weight_sum@119": 15.110060335371337,
    "expectile.settled_frac@25": 0.5136725262938573,
    "expectile.settled_frac@60": 0.8564127056253706,
    "expectile.settled_frac@119": 0.9782071302132749,
    "expectile.withheld_reason@25": None,
    "expectile.withheld_reason@60": None,
    "expectile.withheld_reason@119": None,
    "pa.pred_y0@25": -3.199990178590057,
    "pa.pred_y0@60": -0.16631546312550372,
    "pa.pred_y0@119": -0.1592482393254398,
    "pa.resid_y0@25": -1.6516156456446591,
    "pa.resid_y0@60": -0.3057256656915742,
    "pa.resid_y0@119": 0.10854946836721097,
    "pa.weight_sum@25": 7.999488060097996,
    "pa.weight_sum@60": 12.473100285951407,
    "pa.weight_sum@119": 15.110060335371337,
    "pa.settled_frac@25": 0.5136725262938573,
    "pa.settled_frac@60": 0.8564127056253706,
    "pa.settled_frac@119": 0.9782071302132749,
    "pa.withheld_reason@25": None,
    "pa.withheld_reason@60": None,
    "pa.withheld_reason@119": None,
    "sgd_simplex.pred_y0@25": 0.41185205175526035,
    "sgd_simplex.pred_y0@60": -2.7799297340528453,
    "sgd_simplex.pred_y0@119": -1.6465147095640758,
    "sgd_simplex.resid_y0@25": -5.263457875989976,
    "sgd_simplex.resid_y0@60": 2.3078886052357674,
    "sgd_simplex.resid_y0@119": 1.5958159386058468,
    "sgd_simplex.weight_sum@25": 7.999488060097996,
    "sgd_simplex.weight_sum@60": 12.473100285951407,
    "sgd_simplex.weight_sum@119": 15.110060335371337,
    "sgd_simplex.settled_frac@25": 0.5136725262938573,
    "sgd_simplex.settled_frac@60": 0.8564127056253706,
    "sgd_simplex.settled_frac@119": 0.9782071302132749,
    "sgd_simplex.withheld_reason@25": None,
    "sgd_simplex.withheld_reason@60": None,
    "sgd_simplex.withheld_reason@119": None,
    "pa_box.pred_y0@25": -2.9409092057294677,
    "pa_box.pred_y0@60": 0.8352587048416709,
    "pa_box.pred_y0@119": -0.18940573438427163,
    "pa_box.resid_y0@25": -1.9106966185052485,
    "pa_box.resid_y0@60": -1.3072998336587487,
    "pa_box.resid_y0@119": 0.1387069634260428,
    "pa_box.weight_sum@25": 7.999488060097996,
    "pa_box.weight_sum@60": 12.473100285951407,
    "pa_box.weight_sum@119": 15.110060335371337,
    "pa_box.settled_frac@25": 0.5136725262938573,
    "pa_box.settled_frac@60": 0.8564127056253706,
    "pa_box.settled_frac@119": 0.9782071302132749,
    "pa_box.withheld_reason@25": None,
    "pa_box.withheld_reason@60": None,
    "pa_box.withheld_reason@119": None,
    "kalman_revert.pred_y0@25": -0.8614617173433481,
    "kalman_revert.pred_y0@60": -1.2857093246928781,
    "kalman_revert.pred_y0@119": -1.481854845247646,
    "kalman_revert.resid_y0@25": -3.990144106891368,
    "kalman_revert.resid_y0@60": 0.8136681958758002,
    "kalman_revert.resid_y0@119": 1.4311560742894174,
    "kalman_revert.weight_sum@25": 7.999488060097996,
    "kalman_revert.weight_sum@60": 12.473100285951407,
    "kalman_revert.weight_sum@119": 15.110060335371337,
    "kalman_revert.settled_frac@25": 0.5136725262938573,
    "kalman_revert.settled_frac@60": 0.8564127056253706,
    "kalman_revert.settled_frac@119": 0.9782071302132749,
    "kalman_revert.withheld_reason@25": None,
    "kalman_revert.withheld_reason@60": None,
    "kalman_revert.withheld_reason@119": None,
    "ftrl.pred_y0@25": -4.230250400279184,
    "ftrl.pred_y0@60": -0.46261201909025584,
    "ftrl.pred_y0@119": -0.14860007206871528,
    "ftrl.resid_y0@25": -0.6213554239555323,
    "ftrl.resid_y0@60": -0.009429109726822094,
    "ftrl.resid_y0@119": 0.09790130111048645,
    "ftrl.weight_sum@25": 7.999488060097996,
    "ftrl.weight_sum@60": 12.473100285951407,
    "ftrl.weight_sum@119": 15.110060335371337,
    "ftrl.settled_frac@25": 0.5136725262938573,
    "ftrl.settled_frac@60": 0.8564127056253706,
    "ftrl.settled_frac@119": 0.9782071302132749,
    "ftrl.withheld_reason@25": None,
    "ftrl.withheld_reason@60": None,
    "ftrl.withheld_reason@119": None,
    "ar.pred_y0@25": -4.470717134738014,
    "ar.pred_y0@60": -0.30474557301931404,
    "ar.pred_y0@119": None,
    "ar.resid_y0@25": -0.38088868949670207,
    "ar.resid_y0@60": -0.1672955557977639,
    "ar.resid_y0@119": None,
    "ar.weight_sum@25": 7.999488060097996,
    "ar.weight_sum@60": 12.473100285951407,
    "ar.weight_sum@119": 15.110060335371337,
    "ar.settled_frac@25": 0.5136725262938573,
    "ar.settled_frac@60": 0.8564127056253706,
    "ar.settled_frac@119": 0.9782071302132749,
    "ar.withheld_reason@25": None,
    "ar.withheld_reason@60": None,
    "ar.withheld_reason@119": None,
    "spectral.power_y0_p0@25": 35.80199470928039,
    "spectral.power_y0_p0@60": 6.2164280661898115,
    "spectral.power_y0_p0@119": None,
    "spectral.power_y0_p1@25": 14.380914630784845,
    "spectral.power_y0_p1@60": 6.676741224271466,
    "spectral.power_y0_p1@119": None,
    "spectral.weight_sum@25": 8.573837237596514,
    "spectral.weight_sum@60": 13.244185655943454,
    "spectral.weight_sum@119": 15.09397614533663,
    "spectral.settled_frac@25": 0.5136725262938573,
    "spectral.settled_frac@60": 0.8564127056253706,
    "spectral.settled_frac@119": 0.9793826888941736,
    "spectral.withheld_reason@25": None,
    "spectral.withheld_reason@60": None,
    "spectral.withheld_reason@119": None,
    "holt.pred_y0@25": -0.6509754398513061,
    "holt.pred_y0@60": 0.8460006837790632,
    "holt.pred_y0@119": -0.9097093205227099,
    "holt.resid_y0@25": -4.20063038438341,
    "holt.resid_y0@60": -1.318041812596141,
    "holt.resid_y0@119": 0.8590105495644811,
    "holt.weight_sum@25": 8.573837237596514,
    "holt.weight_sum@60": 13.244185655943454,
    "holt.weight_sum@119": 15.09397614533663,
    "holt.settled_frac@25": 0.5136725262938573,
    "holt.settled_frac@60": 0.8564127056253706,
    "holt.settled_frac@119": 0.9793826888941736,
    "holt.withheld_reason@25": None,
    "holt.withheld_reason@60": None,
    "holt.withheld_reason@119": None,
    "moments.mean_x0@25": 0.14567573584298,
    "moments.mean_x0@60": 0.5319388857885933,
    "moments.mean_x0@119": 0.07562417774137233,
    "moments.mean_x1@25": 1.3170476393181616,
    "moments.mean_x1@60": 1.503050612553659,
    "moments.mean_x1@119": 2.055605486339294,
    "moments.var_x0@25": 1.3324937654135034,
    "moments.var_x0@60": 1.1972093809025777,
    "moments.var_x0@119": 1.0651986840630365,
    "moments.var_x1@25": 12.867570498661806,
    "moments.var_x1@60": 5.35115853967803,
    "moments.var_x1@119": 7.744213099356213,
    "moments.corr_x0_x1@25": 0.1950011675208187,
    "moments.corr_x0_x1@60": 0.3160753290419023,
    "moments.corr_x0_x1@119": 0.2439860537587861,
    "moments.weight_sum@25": 7.999488060097996,
    "moments.weight_sum@60": 12.473100285951407,
    "moments.weight_sum@119": 15.110060335371337,
    "moments.settled_frac@25": 0.5136725262938573,
    "moments.settled_frac@60": 0.8564127056253706,
    "moments.settled_frac@119": 0.9782071302132749,
    "moments.withheld_reason@25": None,
    "moments.withheld_reason@60": None,
    "moments.withheld_reason@119": None,
    "kmeans.cluster@25": 1,
    "kmeans.cluster@60": 0,
    "kmeans.cluster@119": 0,
    "kmeans.dist@25": 1.0657683037801138,
    "kmeans.dist@60": 2.0686144858219824,
    "kmeans.dist@119": 0.23320451009648485,
    "kmeans.dist_second@25": 2.9549386449346025,
    "kmeans.dist_second@60": 3.1351140254811636,
    "kmeans.dist_second@119": 1.9857367773902603,
    "kmeans.weight_sum@25": 7.999488060097996,
    "kmeans.weight_sum@60": 12.473100285951407,
    "kmeans.weight_sum@119": 15.110060335371337,
    "kmeans.settled_frac@25": 0.5136725262938573,
    "kmeans.settled_frac@60": 0.8564127056253706,
    "kmeans.settled_frac@119": 0.9782071302132749,
    "kmeans.withheld_reason@25": None,
    "kmeans.withheld_reason@60": None,
    "kmeans.withheld_reason@119": None,
    "micro.cluster@25": 2,
    "micro.cluster@60": 7,
    "micro.cluster@119": 2,
    "micro.dist@25": 1.1517210881848035,
    "micro.dist@60": 2.535806113533875,
    "micro.dist@119": 0.6444533017493204,
    "micro.micro_id@25": 2,
    "micro.micro_id@60": 13,
    "micro.micro_id@119": 7,
    "micro.outlier@25": False,
    "micro.outlier@60": True,
    "micro.outlier@119": False,
    "micro.n_clusters@25": 2,
    "micro.n_clusters@60": 1,
    "micro.n_clusters@119": 1,
    "micro.n_micro@25": 4,
    "micro.n_micro@60": 3,
    "micro.n_micro@119": 4,
    "micro.weight_sum@25": 7.999488060097996,
    "micro.weight_sum@60": 12.473100285951407,
    "micro.weight_sum@119": 15.110060335371337,
    "micro.settled_frac@25": 0.5136725262938573,
    "micro.settled_frac@60": 0.8564127056253706,
    "micro.settled_frac@119": 0.9782071302132749,
    "micro.withheld_reason@25": None,
    "micro.withheld_reason@60": None,
    "micro.withheld_reason@119": None,
    "ew_class.class@25": "hi",
    "ew_class.class@60": "lo",
    "ew_class.class@119": "lo",
    "ew_class.p_lo@25": 0.001769258464791603,
    "ew_class.p_lo@60": 0.9998050842334101,
    "ew_class.p_lo@119": 0.9706581254697835,
    "ew_class.p_hi@25": 0.9982307415352083,
    "ew_class.p_hi@60": 0.00019491576659006523,
    "ew_class.p_hi@119": 0.0293418745302166,
    "ew_class.weight_sum@25": 7.999488060097996,
    "ew_class.weight_sum@60": 12.473100285951407,
    "ew_class.weight_sum@119": 15.110060335371337,
    "ew_class.settled_frac@25": 0.5136725262938573,
    "ew_class.settled_frac@60": 0.8564127056253706,
    "ew_class.settled_frac@119": 0.9782071302132749,
    "ew_class.withheld_reason@25": None,
    "ew_class.withheld_reason@60": None,
    "ew_class.withheld_reason@119": None,
    "marginal.weight_sum@25": 7.999488060097996,
    "marginal.weight_sum@60": 12.473100285951407,
    "marginal.weight_sum@119": 15.110060335371337,
    "marginal.weight_sum[a/x0]@end": 14.643505957885951,
    "marginal.n_kish[a/x0]@end": 22.588085651395215,
    "marginal.corr[a/x0]@end": 0.38573773334037753,
    "marginal.beta[a/x0]@end": 1.2316780579628355,
    "marginal.t_stat[a/x0]@end": 1.89706698898827,
    "marginal.weight_sum[a/x1]@end": 14.643505957885951,
    "marginal.n_kish[a/x1]@end": 22.588085651395215,
    "marginal.corr[a/x1]@end": -0.8869994444789747,
    "marginal.beta[a/x1]@end": -0.7081876904377251,
    "marginal.t_stat[a/x1]@end": -8.715757847257636,
    "marginal.weight_sum[b/x0]@end": 14.25784941881905,
    "marginal.n_kish[b/x0]@end": 23.20892217040503,
    "marginal.corr[b/x0]@end": 0.41913584410223514,
    "marginal.beta[b/x0]@end": 0.9210410451370624,
    "marginal.t_stat[b/x0]@end": 2.1260076771757284,
    "marginal.weight_sum[b/x1]@end": 14.25784941881905,
    "marginal.n_kish[b/x1]@end": 23.20892217040503,
    "marginal.corr[b/x1]@end": -0.7634819469727878,
    "marginal.beta[b/x1]@end": -0.6068290466325642,
    "marginal.t_stat[b/x1]@end": -5.444279522504034,
    "corrchange.stat@25": None,
    "corrchange.stat@60": None,
    "corrchange.stat@119": None,
    "corrchange.crit@25": None,
    "corrchange.crit@60": None,
    "corrchange.crit@119": None,
    "corrchange.flag@25": None,
    "corrchange.flag@60": None,
    "corrchange.flag@119": None,
    "corrchange.since_flag@25": None,
    "corrchange.since_flag@60": None,
    "corrchange.since_flag@119": None,
    "corrchange.since_change@25": None,
    "corrchange.since_change@60": None,
    "corrchange.since_change@119": None,
    "corrchange.weight_sum@25": 11.0,
    "corrchange.weight_sum@60": 29.0,
    "corrchange.weight_sum@119": 57.0,
    "corrchange.settled_frac@25": None,
    "corrchange.settled_frac@60": None,
    "corrchange.settled_frac@119": None,
    "corrchange.withheld_reason@25": None,
    "corrchange.withheld_reason@60": None,
    "corrchange.withheld_reason@119": None,
    "corrchange.stat#0": 1.0799440280098145,
    "corrchange.stat#1": 0.8756179648489109,
    "corrchange.stat#2": 0.9035820107280972,
    "corrchange.stat#3": 0.5098154108548748,
    "hmm.filtered_state0@25": 0.0012857211925048136,
    "hmm.filtered_state0@60": 0.8112978844060631,
    "hmm.filtered_state0@119": 0.418165754278722,
    "hmm.filtered_state1@25": 0.9987142788074951,
    "hmm.filtered_state1@60": 0.188702115593937,
    "hmm.filtered_state1@119": 0.581834245721278,
    "hmm.predicted_state0@25": 0.21118422348851099,
    "hmm.predicted_state0@60": 0.6689272181155315,
    "hmm.predicted_state0@119": 0.46179870826203684,
    "hmm.predicted_state1@25": 0.788815776511489,
    "hmm.predicted_state1@60": 0.33107278188446865,
    "hmm.predicted_state1@119": 0.538201291737963,
    "hmm.state@25": 1,
    "hmm.state@60": 0,
    "hmm.state@119": 1,
    "hmm.loglik@25": -4.964233120207337,
    "hmm.loglik@60": -4.995091508981325,
    "hmm.loglik@119": -3.042105336113919,
    "hmm.weight_sum@25": 7.999488060097996,
    "hmm.weight_sum@60": 12.473100285951407,
    "hmm.weight_sum@119": 15.110060335371337,
    "hmm.settled_frac@25": 0.5136725262938573,
    "hmm.settled_frac@60": 0.8564127056253706,
    "hmm.settled_frac@119": 0.9782071302132749,
    "hmm.withheld_reason@25": None,
    "hmm.withheld_reason@60": None,
    "hmm.withheld_reason@119": None,
    "rcov.weight_sum@25": 11.0,
    "rcov.weight_sum@60": 0.0,
    "rcov.weight_sum@119": 28.0,
    "deco.u@25": 0.4167923065360934,
    "deco.u@60": 0.9917268375727508,
    "deco.u@119": 0.8126243515712438,
    "deco.rho@25": -0.08568395031476563,
    "deco.rho@60": 0.23541108428143503,
    "deco.rho@119": 0.16316105267641887,
    "deco.loglik@25": -3.9986413526629514,
    "deco.loglik@60": -4.90293781097886,
    "deco.loglik@119": -2.1930858011358536,
    "deco.weight_sum@25": 7.999488060097996,
    "deco.weight_sum@60": 12.473100285951407,
    "deco.weight_sum@119": 15.110060335371337,
    "deco.settled_frac@25": 0.5136725262938573,
    "deco.settled_frac@60": 0.8564127056253706,
    "deco.settled_frac@119": 0.9782071302132749,
    "deco.withheld_reason@25": None,
    "deco.withheld_reason@60": None,
    "deco.withheld_reason@119": None,
    "bocpd.p_change@25": 0.02628246584085018,
    "bocpd.p_change@60": 0.027730092663944658,
    "bocpd.p_change@119": 0.06807509479647987,
    "bocpd.run_mode@25": 11,
    "bocpd.run_mode@60": 29,
    "bocpd.run_mode@119": 49,
    "bocpd.run_mean@25": 10.530689528178458,
    "bocpd.run_mean@60": 23.803185886723877,
    "bocpd.run_mean@119": 45.543971636189475,
    "bocpd.pred_x0@25": 0.08622170400348289,
    "bocpd.pred_x0@60": 0.5193042747383524,
    "bocpd.pred_x0@119": -0.03850848988674341,
    "bocpd.pred_x1@25": 1.2136644279052489,
    "bocpd.pred_x1@60": 1.712608799264034,
    "bocpd.pred_x1@119": 2.09340948522182,
    "bocpd.loglik@25": -5.554086833699265,
    "bocpd.loglik@60": -7.0867959219070595,
    "bocpd.loglik@119": -3.2335170491144485,
    "bocpd.weight_sum@25": 11.0,
    "bocpd.weight_sum@60": 28.5,
    "bocpd.weight_sum@119": 57.0,
    "bocpd.settled_frac@25": None,
    "bocpd.settled_frac@60": None,
    "bocpd.settled_frac@119": None,
    "bocpd.withheld_reason@25": None,
    "bocpd.withheld_reason@60": None,
    "bocpd.withheld_reason@119": None,
    "seqtest.log_e_pos_y0@25": 0.0,
    "seqtest.log_e_pos_y0@60": 0.0,
    "seqtest.log_e_pos_y0@119": 0.0,
    "seqtest.log_e_neg_y0@25": 2.797424236204631,
    "seqtest.log_e_neg_y0@60": -1.7626699454934798,
    "seqtest.log_e_neg_y0@119": 5.481151387104117,
    "seqtest.n_pos_y0@25": 1,
    "seqtest.n_pos_y0@60": 13,
    "seqtest.n_pos_y0@119": 14,
    "seqtest.n_neg_y0@25": 10,
    "seqtest.n_neg_y0@60": 16,
    "seqtest.n_neg_y0@119": 43,
    "seqtest.weight_sum@25": 12.0,
    "seqtest.weight_sum@60": 30.0,
    "seqtest.weight_sum@119": 59.0,
    "seqtest.settled_frac@25": None,
    "seqtest.settled_frac@60": None,
    "seqtest.settled_frac@119": None,
    "seqtest.withheld_reason@25": None,
    "seqtest.withheld_reason@60": None,
    "seqtest.withheld_reason@119": None,
    "seqtest_compare.log_e_a_y0@25": 0.0,
    "seqtest_compare.log_e_a_y0@60": 0.0,
    "seqtest_compare.log_e_a_y0@119": 0.0,
    "seqtest_compare.log_e_b_y0@25": 1.4759065198095778,
    "seqtest_compare.log_e_b_y0@60": 5.22345427614284,
    "seqtest_compare.log_e_b_y0@119": 21.110825104883858,
    "seqtest_compare.wins_a_y0@25": 0,
    "seqtest_compare.wins_a_y0@60": 3,
    "seqtest_compare.wins_a_y0@119": 3,
    "seqtest_compare.wins_b_y0@25": 4,
    "seqtest_compare.wins_b_y0@60": 20,
    "seqtest_compare.wins_b_y0@119": 47,
    "seqtest_compare.weight_sum@25": 12.0,
    "seqtest_compare.weight_sum@60": 30.0,
    "seqtest_compare.weight_sum@119": 59.0,
    "seqtest_compare.settled_frac@25": None,
    "seqtest_compare.settled_frac@60": None,
    "seqtest_compare.settled_frac@119": None,
    "seqtest_compare.withheld_reason@25": None,
    "seqtest_compare.withheld_reason@60": None,
    "seqtest_compare.withheld_reason@119": None,
    "rcov.rcov_n[rcov/a/m]": 21,
    "rcov.bandwidth[rcov/a/m]": 3,
    "rcov.rcov0[rcov/a/m]": 21.95819492943922,
    "rcov.rcov1[rcov/a/m]": 62.2773177861597,
    "rcov.rcov2[rcov/a/m]": 353.6268031231857,
    "rcov.rcov_n[rcov/b/m]": 21,
    "rcov.bandwidth[rcov/b/m]": 3,
    "rcov.rcov0[rcov/b/m]": 20.168462820112076,
    "rcov.rcov1[rcov/b/m]": 17.798995434502125,
    "rcov.rcov2[rcov/b/m]": 762.0001389768797,
    "hawkes.intensity_y0@25": 0.15931455262653227,
    "hawkes.intensity_y0@60": 0.3169576385534196,
    "hawkes.intensity_y0@119": 0.25742055516793616,
    "hawkes.intensity_x0@25": 0.11467201326604769,
    "hawkes.intensity_x0@60": 0.2898184955746359,
    "hawkes.intensity_x0@119": 0.3302834393846523,
    "hawkes.weight_sum@25": 8.573837237596514,
    "hawkes.weight_sum@60": 13.244185655943454,
    "hawkes.weight_sum@119": 15.09397614533663,
    "hawkes.settled_frac@25": 0.5136725262938573,
    "hawkes.settled_frac@60": 0.8564127056253706,
    "hawkes.settled_frac@119": 0.9793826888941736,
    "hawkes.withheld_reason@25": None,
    "hawkes.withheld_reason@60": None,
    "hawkes.withheld_reason@119": None,
    "garch.variance_y0@25": None,
    "garch.variance_y0@60": None,
    "garch.variance_y0@119": None,
    "garch.std_resid_y0@25": None,
    "garch.std_resid_y0@60": None,
    "garch.std_resid_y0@119": None,
    "garch.variance_h3_y0@25": None,
    "garch.variance_h3_y0@60": None,
    "garch.variance_h3_y0@119": None,
    "garch.variance_sum_h3_y0@25": None,
    "garch.variance_sum_h3_y0@60": None,
    "garch.variance_sum_h3_y0@119": None,
    "garch.variance_x0@25": None,
    "garch.variance_x0@60": None,
    "garch.variance_x0@119": None,
    "garch.std_resid_x0@25": None,
    "garch.std_resid_x0@60": None,
    "garch.std_resid_x0@119": None,
    "garch.variance_h3_x0@25": None,
    "garch.variance_h3_x0@60": None,
    "garch.variance_h3_x0@119": None,
    "garch.variance_sum_h3_x0@25": None,
    "garch.variance_sum_h3_x0@60": None,
    "garch.variance_sum_h3_x0@119": None,
    "garch.weight_sum@25": 12.0,
    "garch.weight_sum@60": 30.0,
    "garch.weight_sum@119": 58.5,
    "garch.settled_frac@25": None,
    "garch.settled_frac@60": None,
    "garch.settled_frac@119": None,
    "garch.withheld_reason@25": "below_warm_rows",
    "garch.withheld_reason@60": "below_warm_rows",
    "garch.withheld_reason@119": "not_identified",
    "dcc.corr_y0_x0@25": None,
    "dcc.corr_y0_x0@60": None,
    "dcc.corr_y0_x0@119": None,
    "dcc.loglik@25": None,
    "dcc.loglik@60": None,
    "dcc.loglik@119": None,
    "dcc.weight_sum@25": 8.573837237596514,
    "dcc.weight_sum@60": 13.244185655943454,
    "dcc.weight_sum@119": 15.09397614533663,
    "dcc.settled_frac@25": 0.5136725262938573,
    "dcc.settled_frac@60": 0.8564127056253706,
    "dcc.settled_frac@119": 0.9793826888941736,
    "dcc.withheld_reason@25": "below_warm_rows",
    "dcc.withheld_reason@60": "below_warm_rows",
    "dcc.withheld_reason@119": "not_identified",
    "dcc_given.corr_y0_x0@25": None,
    "dcc_given.corr_y0_x0@60": 0.4201286457446873,
    "dcc_given.corr_y0_x0@119": 0.43116468016732334,
    "dcc_given.loglik@25": None,
    "dcc_given.loglik@60": -3.1753214574277164,
    "dcc_given.loglik@119": -1.8002845663538574,
    "dcc_given.weight_sum@25": 8.573837237596514,
    "dcc_given.weight_sum@60": 13.244185655943454,
    "dcc_given.weight_sum@119": 15.09397614533663,
    "dcc_given.settled_frac@25": 0.5136725262938573,
    "dcc_given.settled_frac@60": 0.8564127056253706,
    "dcc_given.settled_frac@119": 0.9793826888941736,
    "dcc_given.withheld_reason@25": "below_warm_rows",
    "dcc_given.withheld_reason@60": None,
    "dcc_given.withheld_reason@119": None,
    "tdigest.q0.05_y0@25": None,
    "tdigest.q0.05_y0@60": -4.197545663050509,
    "tdigest.q0.05_y0@119": -5.988442618363641,
    "tdigest.q0.25_y0@25": -1.9996236613730871,
    "tdigest.q0.25_y0@60": -2.0445841105377798,
    "tdigest.q0.25_y0@119": -3.295113304272044,
    "tdigest.q0.95_y0@25": None,
    "tdigest.q0.95_y0@60": 3.399413896684236,
    "tdigest.q0.95_y0@119": 3.442977541333919,
    "tdigest.es0.05_y0@25": None,
    "tdigest.es0.05_y0@60": -4.276395950314813,
    "tdigest.es0.05_y0@119": -6.144770136419573,
    "tdigest.es0.25_y0@25": -2.819533082955558,
    "tdigest.es0.25_y0@60": -3.0494582907280123,
    "tdigest.es0.25_y0@119": -4.47743700809698,
    "tdigest.es0.95_y0@25": None,
    "tdigest.es0.95_y0@60": 3.2404503114274767,
    "tdigest.es0.95_y0@119": 2.9888171225493285,
    "tdigest.q0.05_x0@25": None,
    "tdigest.q0.05_x0@60": -1.6121145373299823,
    "tdigest.q0.05_x0@119": -1.748127094276229,
    "tdigest.q0.25_x0@25": -0.9501845365622195,
    "tdigest.q0.25_x0@60": -0.3779981717058362,
    "tdigest.q0.25_x0@119": -0.8281578086090776,
    "tdigest.q0.95_x0@25": None,
    "tdigest.q0.95_x0@60": 1.8719070120626502,
    "tdigest.q0.95_x0@119": 1.5577265807226812,
    "tdigest.es0.05_x0@25": None,
    "tdigest.es0.05_x0@60": -1.7438391327061504,
    "tdigest.es0.05_x0@119": -1.7859206916930672,
    "tdigest.es0.25_x0@25": -1.2853718086686845,
    "tdigest.es0.25_x0@60": -0.9227558215280355,
    "tdigest.es0.25_x0@119": -1.244573486863645,
    "tdigest.es0.95_x0@25": None,
    "tdigest.es0.95_x0@60": 1.9591147409749547,
    "tdigest.es0.95_x0@119": 1.4605982458375633,
    "tdigest.weight_sum@25": 12.0,
    "tdigest.weight_sum@60": 30.0,
    "tdigest.weight_sum@119": 58.5,
    "tdigest.settled_frac@25": None,
    "tdigest.settled_frac@60": None,
    "tdigest.settled_frac@119": None,
    "tdigest.withheld_reason@25": "level_beyond_sample",
    "tdigest.withheld_reason@60": None,
    "tdigest.withheld_reason@119": None,
    "isotonic.pred_y0@25": -0.5069855721719836,
    "isotonic.pred_y0@60": 0.34407187441562564,
    "isotonic.pred_y0@119": -1.17378855152561,
    "isotonic.resid_y0@25": -4.344620252062732,
    "isotonic.resid_y0@60": -0.8161130032327035,
    "isotonic.resid_y0@119": 1.1230897805673812,
    "isotonic.weight_sum@25": 8.573837237596514,
    "isotonic.weight_sum@60": 13.244185655943454,
    "isotonic.weight_sum@119": 15.09397614533663,
    "isotonic.settled_frac@25": 0.5136725262938573,
    "isotonic.settled_frac@60": 0.8564127056253706,
    "isotonic.settled_frac@119": 0.9793826888941736,
    "isotonic.withheld_reason@25": None,
    "isotonic.withheld_reason@60": None,
    "isotonic.withheld_reason@119": None,
    "ddsketch.q0.05_y0@25": None,
    "ddsketch.q0.05_y0@60": -3.857406244389122,
    "ddsketch.q0.05_y0@119": -5.640697159022917,
    "ddsketch.q0.25_y0@25": -2.075027345797204,
    "ddsketch.q0.25_y0@60": -2.159713698038698,
    "ddsketch.q0.25_y0@119": -3.095618319853687,
    "ddsketch.q0.95_y0@25": None,
    "ddsketch.q0.95_y0@60": 2.5856622535218223,
    "ddsketch.q0.95_y0@119": 2.075027345797204,
    "ddsketch.es0.05_y0@25": None,
    "ddsketch.es0.05_y0@60": -4.634338774148497,
    "ddsketch.es0.05_y0@119": -6.49500614275016,
    "ddsketch.es0.25_y0@25": -2.8464824522555707,
    "ddsketch.es0.25_y0@60": -3.32093313526353,
    "ddsketch.es0.25_y0@119": -4.701412619189431,
    "ddsketch.es0.95_y0@25": None,
    "ddsketch.es0.95_y0@60": 3.4540190072227084,
    "ddsketch.es0.95_y0@119": 4.445139729130649,
    "ddsketch.q0.05_x0@25": None,
    "ddsketch.q0.05_x0@60": -1.8403767977708627,
    "ddsketch.q0.05_x0@119": -1.6988773749154278,
    "ddsketch.q0.25_x0@25": -0.9138827457526327,
    "ddsketch.q0.25_x0@60": -0.2392855495976762,
    "ddsketch.q0.25_x0@119": -0.8436178514993646,
    "ddsketch.q0.95_x0@25": None,
    "ddsketch.q0.95_x0@60": 1.7331981299642243,
    "ddsketch.q0.95_x0@119": 1.3099107407241846,
    "ddsketch.es0.05_x0@25": None,
    "ddsketch.es0.05_x0@60": -1.8403767977708627,
    "ddsketch.es0.05_x0@119": -2.0114126469164657,
    "ddsketch.es0.25_x0@25": -1.3253770502600097,
    "ddsketch.es0.25_x0@60": -0.9824531201171406,
    "ddsketch.es0.25_x0@119": -1.287893065189186,
    "ddsketch.es0.95_x0@25": None,
    "ddsketch.es0.95_x0@60": 1.9541832518843338,
    "ddsketch.es0.95_x0@119": 1.629707022408927,
    "ddsketch.weight_sum@25": 12.0,
    "ddsketch.weight_sum@60": 30.0,
    "ddsketch.weight_sum@119": 58.5,
    "ddsketch.settled_frac@25": None,
    "ddsketch.settled_frac@60": None,
    "ddsketch.settled_frac@119": None,
    "ddsketch.withheld_reason@25": "level_beyond_sample",
    "ddsketch.withheld_reason@60": None,
    "ddsketch.withheld_reason@119": None,
    "ewm_moments.skew@25": -0.4657181805853099,
    "ewm_moments.skew@60": -0.4528534031103992,
    "ewm_moments.skew@119": -0.20010655951463893,
    "ewm_moments.k4@25": -3.1464657645756278,
    "ewm_moments.k4@60": -1.3064310367476075,
    "ewm_moments.k4@119": -0.9867541739113057,
    "ewm_moments.co@25": 6.131959898451839,
    "ewm_moments.co@60": 1.2721789231882785,
    "ewm_moments.co@119": 0.9228348108713621,
    "ewm_moments.ac1@25": -0.5402338411284154,
    "ewm_moments.ac1@60": -0.2803553097445228,
    "ewm_moments.ac1@119": -0.3811263005681044,
    "ewm_moments.se_skew@25": 0.6208190887745845,
    "ewm_moments.se_skew@60": 0.3300784790940194,
    "ewm_moments.se_skew@119": 0.3492485555673649,
    "ewm_moments.se_k4@25": 1.5340938664239583,
    "ewm_moments.se_k4@60": 1.1329542644490096,
    "ewm_moments.se_k4@119": 0.5415122296147395,
    "ewm_moments.se_co@25": 4.339769479484873,
    "ewm_moments.se_co@60": 1.85837489719577,
    "ewm_moments.se_co@119": 1.7826747568899708,
    "ewm_moments.se_ac1@25": 0.31345465625452906,
    "ewm_moments.se_ac1@60": 0.21965366761883745,
    "ewm_moments.se_ac1@119": 0.15761135300748724,
    "ewm_moments.weight_sum@25": 7.999488060097996,
    "ewm_moments.weight_sum@60": 12.473100285951407,
    "ewm_moments.weight_sum@119": 15.110060335371337,
    "ewm_moments.settled_frac@25": 0.5136725262938573,
    "ewm_moments.settled_frac@60": 0.8564127056253706,
    "ewm_moments.settled_frac@119": 0.9782071302132749,
    "ewm_moments.withheld_reason@25": None,
    "ewm_moments.withheld_reason@60": None,
    "ewm_moments.withheld_reason@119": None,
    "audit.weight_sum@25": 12.0,
    "audit.weight_sum@60": 30.0,
    "audit.weight_sum@119": 59.0,
    "audit.null_count[a/x0]@end": 0,
    "audit.mean[a/x0]@end": -0.05920251622435196,
    "audit.std[a/x0]@end": 0.9988981969991507,
    "audit.skew[a/x0]@end": 0.13864597746050802,
    "audit.kurtosis[a/x0]@end": -0.734244442222745,
    "audit.median[a/x0]@end": -0.14356864736528313,
    "audit.mad[a/x0]@end": 0.6667477795461205,
    "audit.robust_z[a/x0]@end": 2.1270987873532996,
    "audit.autocorr[a/x0]@end": 0.09770051087628484,
    "audit.unit_root_t[a/x0]@end": -6.858639292134699,
    "audit.null_count[a/x1]@end": 2,
    "audit.mean[a/x1]@end": 1.6517307535456187,
    "audit.std[a/x1]@end": 2.958047090531065,
    "audit.skew[a/x1]@end": 0.25324789230021644,
    "audit.kurtosis[a/x1]@end": 0.6037690197199805,
    "audit.median[a/x1]@end": 1.1125136851972806,
    "audit.mad[a/x1]@end": 1.2737272275987532,
    "audit.robust_z[a/x1]@end": 4.311271312254894,
    "audit.autocorr[a/x1]@end": 0.04326963466135358,
    "audit.unit_root_t[a/x1]@end": -7.132496744305026,
    "audit.null_count[a/y0]@end": 2,
    "audit.mean[a/y0]@end": -0.993750609157754,
    "audit.std[a/y0]@end": 2.504101694345906,
    "audit.skew[a/y0]@end": -0.3844424563052659,
    "audit.kurtosis[a/y0]@end": 0.8468352430326291,
    "audit.median[a/y0]@end": -0.8132594552093853,
    "audit.mad[a/y0]@end": 1.7696047336914082,
    "audit.robust_z[a/y0]@end": 2.9855581334344583,
    "audit.autocorr[a/y0]@end": -0.08827084053020935,
    "audit.unit_root_t[a/y0]@end": -8.282069035756278,
    "audit.null_count[b/x0]@end": 0,
    "audit.mean[b/x0]@end": -0.08019820859367618,
    "audit.std[b/x0]@end": 1.0515509249696167,
    "audit.skew[b/x0]@end": 0.03396262324185014,
    "audit.kurtosis[b/x0]@end": -0.7350138196069129,
    "audit.median[b/x0]@end": -0.19159033498205036,
    "audit.mad[b/x0]@end": 0.7995416464065114,
    "audit.robust_z[b/x0]@end": 1.9807723637366774,
    "audit.autocorr[b/x0]@end": -0.15252340094189235,
    "audit.unit_root_t[b/x0]@end": -8.899008930439052,
    "audit.null_count[b/x1]@end": 2,
    "audit.mean[b/x1]@end": 2.201318262427008,
    "audit.std[b/x1]@end": 2.9255292136086797,
    "audit.skew[b/x1]@end": -0.0060740594311637225,
    "audit.kurtosis[b/x1]@end": -0.7910259946808611,
    "audit.median[b/x1]@end": 1.820191974869697,
    "audit.mad[b/x1]@end": 2.0908731101936437,
    "audit.robust_z[b/x1]@end": 2.0020486182498627,
    "audit.autocorr[b/x1]@end": 0.24878649113730708,
    "audit.unit_root_t[b/x1]@end": -5.978771997283295,
    "audit.null_count[b/y0]@end": 2,
    "audit.mean[b/y0]@end": -1.5358445438235244,
    "audit.std[b/y0]@end": 2.5880709572361855,
    "audit.skew[b/y0]@end": 0.18698486798685254,
    "audit.kurtosis[b/y0]@end": 0.6515582812145997,
    "audit.median[b/y0]@end": -1.2662916919638518,
    "audit.mad[b/y0]@end": 1.4750412267800674,
    "audit.robust_z[b/y0]@end": 3.1993145732429586,
    "audit.autocorr[b/y0]@end": 0.2718740592527968,
    "audit.unit_root_t[b/y0]@end": -5.462773009999813,
    "audit.corr[a/x0/x1]@end": 0.12818066217510068,
    "audit.corr[a/x0/y0]@end": 0.4545212384397137,
    "audit.corr[a/x1/y0]@end": -0.8096767137391814,
    "audit.corr[b/x0/x1]@end": 0.08501624143787098,
    "audit.corr[b/x0/y0]@end": 0.5484719648960178,
    "audit.corr[b/x1/y0]@end": -0.7982751217163684,
    "audit.steps[a]@end": 59,
    "audit.gaps[a]@end": 6,
    "audit.step_cv[a]@end": 0.0,
    "audit.steps[b]@end": 59,
    "audit.gaps[b]@end": 7,
    "audit.step_cv[b]@end": 0.0,
    "realized.rv_y0@25": 63.74614294933045,
    "realized.rv_y0@60": 45.381678675293344,
    "realized.rv_y0@119": 92.99952700716962,
    "realized.bv_y0@25": 63.83804348891109,
    "realized.bv_y0@60": 60.7944064127284,
    "realized.bv_y0@119": 106.91126348673279,
    "realized.tv_y0@25": 54.91007239387371,
    "realized.tv_y0@60": 62.93767277377058,
    "realized.tv_y0@119": 126.10379129032694,
    "realized.tq_y0@25": 2752.935139229619,
    "realized.tq_y0@60": 3514.265537580081,
    "realized.tq_y0@119": 18352.340227404962,
    "realized.bns_y0@25": -0.005366330445683928,
    "realized.bns_y0@60": -1.863613060446523,
    "realized.bns_y0@119": -0.7520039903045864,
    "realized.lm_y0@25": -2.2194606793591225,
    "realized.lm_y0@60": -0.323115718253448,
    "realized.lm_y0@119": -0.025839519040295748,
    "realized.lm_jump_y0@25": False,
    "realized.lm_jump_y0@60": False,
    "realized.lm_jump_y0@119": False,
    "realized.rv_x0@25": 9.938094595490671,
    "realized.rv_x0@60": 20.254878081364257,
    "realized.rv_x0@119": 15.476545903347631,
    "realized.bv_x0@25": 11.467088462375752,
    "realized.bv_x0@60": 24.163874225516825,
    "realized.bv_x0@119": 20.339526011602082,
    "realized.tv_x0@25": 13.329255268387866,
    "realized.tv_x0@60": 26.34291368964243,
    "realized.tv_x0@119": 21.543793635297387,
    "realized.tq_x0@25": 118.05750714979281,
    "realized.tq_x0@60": 602.63686618511,
    "realized.tq_x0@119": 320.38618082096474,
    "realized.bns_x0@25": -0.6055426823493026,
    "realized.bns_x0@60": -1.0691936629039072,
    "realized.bns_x0@119": -2.019768429793248,
    "realized.lm_x0@25": 0.6605686061601017,
    "realized.lm_x0@60": -1.60583825564811,
    "realized.lm_x0@119": -0.36639710849888896,
    "realized.lm_jump_x0@25": False,
    "realized.lm_jump_x0@60": False,
    "realized.lm_jump_x0@119": False,
    "realized.weight_sum@25": 8.573837237596514,
    "realized.weight_sum@60": 13.244185655943454,
    "realized.weight_sum@119": 15.09397614533663,
    "realized.settled_frac@25": 0.5136725262938573,
    "realized.settled_frac@60": 0.8564127056253706,
    "realized.settled_frac@119": 0.9793826888941736,
    "realized.withheld_reason@25": None,
    "realized.withheld_reason@60": None,
    "realized.withheld_reason@119": None,
    "pca.pc0_var@25": None,
    "pca.pc0_var@60": None,
    "pca.pc0_var@119": 11.351526086453823,
    "pca.pc0_share@25": None,
    "pca.pc0_share@60": None,
    "pca.pc0_share@119": 0.8166317068219135,
    "pca.pc0_loading_x0@25": None,
    "pca.pc0_loading_x0@60": None,
    "pca.pc0_loading_x0@119": -0.008065810558783434,
    "pca.pc0_loading_x1@25": None,
    "pca.pc0_loading_x1@60": None,
    "pca.pc0_loading_x1@119": 0.7960857265900269,
    "pca.pc0_loading_y0@25": None,
    "pca.pc0_loading_y0@60": None,
    "pca.pc0_loading_y0@119": -0.6051301170985119,
    "pca.pc0_score@25": None,
    "pca.pc0_score@60": None,
    "pca.pc0_score@119": -2.433613513897105,
    "pca.pc1_var@25": None,
    "pca.pc1_var@60": None,
    "pca.pc1_var@119": 2.546989753184595,
    "pca.pc1_share@25": None,
    "pca.pc1_share@60": None,
    "pca.pc1_share@119": 0.18323109805325125,
    "pca.pc1_loading_x0@25": None,
    "pca.pc1_loading_x0@60": None,
    "pca.pc1_loading_x0@119": 0.6457510711301718,
    "pca.pc1_loading_x1@25": None,
    "pca.pc1_loading_x1@60": None,
    "pca.pc1_loading_x1@119": 0.4661968591450687,
    "pca.pc1_loading_y0@25": None,
    "pca.pc1_loading_y0@60": None,
    "pca.pc1_loading_y0@119": 0.6047032682709006,
    "pca.pc1_score@25": None,
    "pca.pc1_score@60": None,
    "pca.pc1_score@119": -0.6086963042019637,
    "pca.weight_sum@25": 7.036918636925818,
    "pca.weight_sum@60": 11.929732854688375,
    "pca.weight_sum@119": 15.066926539248298,
    "pca.settled_frac@25": 0.5136725262938573,
    "pca.settled_frac@60": 0.8564127056253706,
    "pca.settled_frac@119": 0.9782071302132749,
    "pca.withheld_reason@25": "not_separated",
    "pca.withheld_reason@60": "not_separated",
    "pca.withheld_reason@119": None,
    "power_variation.rv_y0@25": 63.74614294933045,
    "power_variation.rv_y0@60": 45.381678675293344,
    "power_variation.rv_y0@119": 92.99952700716962,
    "power_variation.bv_y0@25": 63.83804348891109,
    "power_variation.bv_y0@60": 60.7944064127284,
    "power_variation.bv_y0@119": 106.91126348673279,
    "power_variation.lrv_y0@25": 51.50602412833541,
    "power_variation.lrv_y0@60": 21.564814747586745,
    "power_variation.lrv_y0@119": 87.09397873864988,
    "power_variation.rk_y0@25": 51.57976998218076,
    "power_variation.rk_y0@60": 15.924906397910242,
    "power_variation.rk_y0@119": 195.5906837797761,
    "power_variation.sub_y0@25": 59.845345055867774,
    "power_variation.sub_y0@60": 32.584390371232104,
    "power_variation.sub_y0@119": 195.94235592923977,
    "power_variation.n_y0@25": 8.437996953898091,
    "power_variation.n_y0@60": 18.336921614498657,
    "power_variation.n_y0@119": 24.711348784563615,
    "power_variation.rv_x0@25": 9.938094595490671,
    "power_variation.rv_x0@60": 20.254878081364257,
    "power_variation.rv_x0@119": 15.476545903347631,
    "power_variation.bv_x0@25": 11.467088462375752,
    "power_variation.bv_x0@60": 24.163874225516825,
    "power_variation.bv_x0@119": 20.339526011602082,
    "power_variation.lrv_x0@25": -0.4448662795265075,
    "power_variation.lrv_x0@60": 12.679631527071283,
    "power_variation.lrv_x0@119": 7.7744664104478725,
    "power_variation.rk_x0@25": -2.0007252549178083,
    "power_variation.rk_x0@60": 18.624330968606397,
    "power_variation.rk_x0@119": 7.770967410589124,
    "power_variation.sub_x0@25": 2.011976443096731,
    "power_variation.sub_x0@60": 17.03501040103659,
    "power_variation.sub_x0@119": 12.698950230138927,
    "power_variation.n_x0@25": 9.434027212891817,
    "power_variation.n_x0@60": 19.29198591107325,
    "power_variation.n_x0@119": 25.16280729672253,
    "power_variation.weight_sum@25": 8.573837237596514,
    "power_variation.weight_sum@60": 13.244185655943454,
    "power_variation.weight_sum@119": 15.09397614533663,
    "power_variation.settled_frac@25": 0.5136725262938573,
    "power_variation.settled_frac@60": 0.8564127056253706,
    "power_variation.settled_frac@119": 0.9793826888941736,
    "power_variation.withheld_reason@25": None,
    "power_variation.withheld_reason@60": None,
    "power_variation.withheld_reason@119": None,
}

#: Produced by `uv run python tests/test_golden_pipeline.py`, as `GOLDEN` is.
#: Task 206 (2026-10-08) moved 3, the two ``kalman`` specs' predictions at
#: row 119 -- ``embargo``'s ``pred`` and ``resid``, ``formula``'s ``pred_fwd``
#: -- by the change `GOLDEN`'s ``kalman`` took; rows 25 and 60 are inside the
#: warm-up, which a held row reaches late, and kept their bits. No reference
#: here replays a ``kalman`` embargo row for row: ``rp.stream.embargo``'s
#: doubled stream through ``kalman_ref`` parts from the native embargo
#: before the switch too, where nothing moved (row 25: -0.3103 against
#: -0.2970), so it is no replica of these three. Task 211 (2026-10-08) moved
#: 9, ``embargo``'s ``pred`` and ``resid`` and ``formula``'s ``pred_fwd`` at
#: the three rows, by the change `GOLDEN`'s ``kalman`` took. The doubled
#: stream now agrees with the native embargo where no gap is capped, but
#: this stream caps seven; ``kalman_ref`` fed each learned row at its
#: release with its arrival's capped clock, as the stream replays it, gives
#: ``embargo``'s rows 25 and 60 to 1e-14, and parts from row 75 on (row
#: 119 by 2e-4), which that emulation of the stream's release does not
#: reproduce. ``formula``'s window target has no replica here. Task 270
#: (review round 9, B-1) moved ``formula``'s: its row 25, a row of weight
#: past ``min_weight`` before its prior was sized, was predicted as the
#: unsized prior's 0.0 and is now withheld (``below_warm_rows``), and the
#: residual variance that 0 no longer seeds moves rows 60 and 119. Task 278
#: (batch 17, 2026-10-26) moved 17 of ``diagnostics``: ``sigma`` and
#: ``zscore`` as `GOLDEN`'s ``ridge`` did, the averaged forecast at row 25 by
#: 1.1e-12 (its weights read ``sigma``), and ``autocorr``, which weighs each
#: residual at its row's weight, as ``sigma`` does, where it gave each 1
#: (``tests/test_autocorr_weights.py`` holds it to its update equations).
SWITCHED_GOLDEN: dict[str, float | str | None] = {
    "window.pred_y0@25": -4.688774079491189,
    "window.pred_y0@60": -0.26834997352446543,
    "window.pred_y0@119": -0.15833472180192615,
    "window.resid_y0@25": -0.16283174474352702,
    "window.resid_y0@60": -0.2036911552926125,
    "window.resid_y0@119": 0.10763595084369731,
    "window.weight_sum@25": 6.16379414615054,
    "window.weight_sum@60": 6.880644727932429,
    "window.weight_sum@119": 5.527733129522695,
    "window.settled_frac@25": 0.48594308667196673,
    "window.settled_frac@60": 0.8482256394506191,
    "window.settled_frac@119": 0.9756511068856094,
    "window.withheld_reason@25": None,
    "window.withheld_reason@60": None,
    "window.withheld_reason@119": None,
    "embargo.pred_y0@25": -3.6082781334401965,
    "embargo.pred_y0@60": -0.27414774280650844,
    "embargo.pred_y0@119": -0.17198086938500046,
    "embargo.resid_y0@25": -1.2433276907945197,
    "embargo.resid_y0@60": -0.1978933860105695,
    "embargo.resid_y0@119": 0.12128209842677162,
    "embargo.weight_sum@25": 5.473081981208709,
    "embargo.weight_sum@60": 11.374162212716914,
    "embargo.weight_sum@119": 12.895186479687487,
    "embargo.settled_frac@25": 0.48594308667196673,
    "embargo.settled_frac@60": 0.8482256394506191,
    "embargo.settled_frac@119": 0.9756511068856094,
    "embargo.withheld_reason@25": None,
    "embargo.withheld_reason@60": None,
    "embargo.withheld_reason@119": None,
    "datetime.pred_y0@25": -4.660573991106876,
    "datetime.pred_y0@60": -0.27040260602938293,
    "datetime.pred_y0@119": -0.15930107822005768,
    "datetime.resid_y0@25": -0.19103183312784022,
    "datetime.resid_y0@60": -0.201638522787695,
    "datetime.resid_y0@119": 0.10860230726182885,
    "datetime.weight_sum@25": 6.677851059478573,
    "datetime.weight_sum@60": 11.760613136438161,
    "datetime.weight_sum@119": 12.895186479687487,
    "datetime.settled_frac@25": 0.48594308667196673,
    "datetime.settled_frac@60": 0.8482256394506191,
    "datetime.settled_frac@119": 0.9756511068856094,
    "datetime.withheld_reason@25": None,
    "datetime.withheld_reason@60": None,
    "datetime.withheld_reason@119": None,
    "session_shrink.pred_y0@25": -4.660573991106876,
    "session_shrink.pred_y0@60": -0.27040260602938293,
    "session_shrink.pred_y0@119": -0.15963433590947113,
    "session_shrink.resid_y0@25": -0.19103183312784022,
    "session_shrink.resid_y0@60": -0.201638522787695,
    "session_shrink.resid_y0@119": 0.10893556495124229,
    "session_shrink.weight_sum@25": 6.677851059478573,
    "session_shrink.weight_sum@60": 11.760613136438161,
    "session_shrink.weight_sum@119": 12.94330608304187,
    "session_shrink.settled_frac@25": 0.48594308667196673,
    "session_shrink.settled_frac@60": 0.8482256394506191,
    "session_shrink.settled_frac@119": 0.9749665663253243,
    "session_shrink.withheld_reason@25": None,
    "session_shrink.withheld_reason@60": None,
    "session_shrink.withheld_reason@119": None,
    "pairwise.pred_y0@25": -5.241522514093774,
    "pairwise.pred_y0@60": -0.24213173928403764,
    "pairwise.pred_y0@119": -0.15693079627154655,
    "pairwise.resid_y0@25": 0.38991668985905736,
    "pairwise.resid_y0@60": -0.2299093895330403,
    "pairwise.resid_y0@119": 0.10623202531331771,
    "pairwise.weight_sum@25": 6.677851059478573,
    "pairwise.weight_sum@60": 11.760613136438161,
    "pairwise.weight_sum@119": 12.895186479687487,
    "pairwise.settled_frac@25": 0.48594308667196673,
    "pairwise.settled_frac@60": 0.8482256394506191,
    "pairwise.settled_frac@119": 0.9756511068856094,
    "pairwise.withheld_reason@25": None,
    "pairwise.withheld_reason@60": None,
    "pairwise.withheld_reason@119": None,
    "origin.pred_y0@25": -4.448239382844861,
    "origin.pred_y0@60": -0.764935850002807,
    "origin.pred_y0@119": -0.3988501997480953,
    "origin.resid_y0@25": -0.40336644138985545,
    "origin.resid_y0@60": 0.292894721185729,
    "origin.resid_y0@119": 0.3481514287898665,
    "origin.weight_sum@25": 6.677851059478573,
    "origin.weight_sum@60": 11.760613136438161,
    "origin.weight_sum@119": 12.895186479687487,
    "origin.settled_frac@25": 0.48594308667196673,
    "origin.settled_frac@60": 0.8482256394506191,
    "origin.settled_frac@119": 0.9756511068856094,
    "origin.withheld_reason@25": None,
    "origin.withheld_reason@60": None,
    "origin.withheld_reason@119": None,
    "prior.pred_y0@25": -4.55760789016785,
    "prior.pred_y0@60": -0.2647625677781029,
    "prior.pred_y0@119": -0.15927412069927088,
    "prior.resid_y0@25": -0.2939979340668666,
    "prior.resid_y0@60": -0.20727856103897502,
    "prior.resid_y0@119": 0.10857534974104205,
    "prior.weight_sum@25": 6.677851059478573,
    "prior.weight_sum@60": 11.760613136438161,
    "prior.weight_sum@119": 12.895186479687487,
    "prior.settled_frac@25": 0.48594308667196673,
    "prior.settled_frac@60": 0.8482256394506191,
    "prior.settled_frac@119": 0.9756511068856094,
    "prior.withheld_reason@25": None,
    "prior.withheld_reason@60": None,
    "prior.withheld_reason@119": None,
    "feature_sets.pred_y0__one@25": -1.6187259495389383,
    "feature_sets.pred_y0__one@60": -2.0343969264353436,
    "feature_sets.pred_y0__one@119": -1.5951379518681144,
    "feature_sets.resid_y0__one@25": -3.2328798746957776,
    "feature_sets.resid_y0__one@60": 1.5623557976182658,
    "feature_sets.resid_y0__one@119": 1.5444391809098854,
    "feature_sets.pred_y0__both@25": -4.660573991106876,
    "feature_sets.pred_y0__both@60": -0.27040260602938293,
    "feature_sets.pred_y0__both@119": -0.15930107822005768,
    "feature_sets.resid_y0__both@25": -0.19103183312784022,
    "feature_sets.resid_y0__both@60": -0.201638522787695,
    "feature_sets.resid_y0__both@119": 0.10860230726182885,
    "feature_sets.weight_sum@25": 6.677851059478573,
    "feature_sets.weight_sum@60": 11.760613136438161,
    "feature_sets.weight_sum@119": 12.895186479687487,
    "feature_sets.settled_frac@25": 0.48594308667196673,
    "feature_sets.settled_frac@60": 0.8482256394506191,
    "feature_sets.settled_frac@119": 0.9756511068856094,
    "feature_sets.withheld_reason@25": None,
    "feature_sets.withheld_reason@60": None,
    "feature_sets.withheld_reason@119": None,
    "diagnostics.pred_y0__r0.000001@25": -4.660573991106876,
    "diagnostics.pred_y0__r0.000001@60": -0.27040260602938293,
    "diagnostics.pred_y0__r0.000001@119": -0.15930107822005768,
    "diagnostics.resid_y0__r0.000001@25": -0.19103183312784022,
    "diagnostics.resid_y0__r0.000001@60": -0.201638522787695,
    "diagnostics.resid_y0__r0.000001@119": 0.10860230726182885,
    "diagnostics.pred_y0__r0.5@25": -3.7378760260135997,
    "diagnostics.pred_y0__r0.5@60": 0.3266272825251495,
    "diagnostics.pred_y0__r0.5@119": -0.10696784719221707,
    "diagnostics.resid_y0__r0.5@25": -1.1137297982211165,
    "diagnostics.resid_y0__r0.5@60": -0.7986684113422274,
    "diagnostics.resid_y0__r0.5@119": 0.05626907623398823,
    "diagnostics.error_inflation_y0__r0.000001@25": 1.4847974607158305,
    "diagnostics.error_inflation_y0__r0.000001@60": 1.1806574838531763,
    "diagnostics.error_inflation_y0__r0.000001@119": 1.0397304322665724,
    "diagnostics.error_inflation_y0__r0.5@25": 1.408103981119847,
    "diagnostics.error_inflation_y0__r0.5@60": 1.1561563873876142,
    "diagnostics.error_inflation_y0__r0.5@119": 1.0382074672292183,
    "diagnostics.sigma_y0__r0.000001@25": 0.0802990730020251,
    "diagnostics.sigma_y0__r0.000001@60": 0.13651765429842572,
    "diagnostics.sigma_y0__r0.000001@119": 0.10015950081429413,
    "diagnostics.sigma_y0__r0.5@25": 1.160544636928746,
    "diagnostics.sigma_y0__r0.5@60": 0.7601620042326201,
    "diagnostics.sigma_y0__r0.5@119": 0.6088372396434204,
    "diagnostics.zscore_y0__r0.000001@25": -2.3790042149430852,
    "diagnostics.zscore_y0__r0.000001@60": -1.4770142647405584,
    "diagnostics.zscore_y0__r0.000001@119": 1.0842936154722709,
    "diagnostics.zscore_y0__r0.5@25": -0.9596613200233989,
    "diagnostics.zscore_y0__r0.5@60": -1.050655527236565,
    "diagnostics.zscore_y0__r0.5@119": 0.09242055605360723,
    "diagnostics.ic_y0__r0.000001@25": 1.0,
    "diagnostics.ic_y0__r0.000001@60": 0.9978630112742379,
    "diagnostics.ic_y0__r0.000001@119": 0.9990892092079929,
    "diagnostics.ic_y0__r0.5@25": -1.0,
    "diagnostics.ic_y0__r0.5@60": 0.9065594449867237,
    "diagnostics.ic_y0__r0.5@119": 0.9764724448466189,
    "diagnostics.r2_y0__r0.000001@25": 0.9988361228670762,
    "diagnostics.r2_y0__r0.000001@60": 0.9940913466634884,
    "diagnostics.r2_y0__r0.000001@119": 0.9980636266485589,
    "diagnostics.r2_y0__r0.5@25": -0.37255719720739955,
    "diagnostics.r2_y0__r0.5@60": 0.818340163175121,
    "diagnostics.r2_y0__r0.5@119": 0.9285213132201167,
    "diagnostics.hit_rate_y0__r0.000001@25": 1.0,
    "diagnostics.hit_rate_y0__r0.000001@60": 1.0,
    "diagnostics.hit_rate_y0__r0.000001@119": 1.0,
    "diagnostics.hit_rate_y0__r0.5@25": 1.0,
    "diagnostics.hit_rate_y0__r0.5@60": 0.8972624478254776,
    "diagnostics.hit_rate_y0__r0.5@119": 0.9370631137126367,
    "diagnostics.abs_resid_q0.5_y0__r0.000001@25": 0.047117875647668395,
    "diagnostics.abs_resid_q0.5_y0__r0.000001@60": 0.10400161384976526,
    "diagnostics.abs_resid_q0.5_y0__r0.000001@119": 0.08544642857142856,
    "diagnostics.abs_resid_q0.5_y0__r0.5@25": 0.21972222222222224,
    "diagnostics.abs_resid_q0.5_y0__r0.5@60": 0.6288819875776398,
    "diagnostics.abs_resid_q0.5_y0__r0.5@119": 0.39256840796019904,
    "diagnostics.abs_resid_q0.9_y0__r0.000001@25": 0.047117875647668395,
    "diagnostics.abs_resid_q0.9_y0__r0.000001@60": 0.17870560109289618,
    "diagnostics.abs_resid_q0.9_y0__r0.000001@119": 0.17284604519774013,
    "diagnostics.abs_resid_q0.9_y0__r0.5@25": 3.0467948717948716,
    "diagnostics.abs_resid_q0.9_y0__r0.5@60": 1.0077519379844961,
    "diagnostics.abs_resid_q0.9_y0__r0.5@119": 0.8632635746606334,
    "diagnostics.autocorr_y0__r0.000001@25": 0.0,
    "diagnostics.autocorr_y0__r0.000001@60": -0.1276932135427718,
    "diagnostics.autocorr_y0__r0.000001@119": -0.24219110698906277,
    "diagnostics.autocorr_y0__r0.5@25": 0.0,
    "diagnostics.autocorr_y0__r0.5@60": -0.19318031010448358,
    "diagnostics.autocorr_y0__r0.5@119": -0.11530177609956221,
    "diagnostics.drift_y0__r0.000001@25": False,
    "diagnostics.drift_y0__r0.000001@60": False,
    "diagnostics.drift_y0__r0.000001@119": False,
    "diagnostics.drift_y0__r0.5@25": False,
    "diagnostics.drift_y0__r0.5@60": False,
    "diagnostics.drift_y0__r0.5@119": False,
    "diagnostics.weight_sum@25": 6.677851059478573,
    "diagnostics.weight_sum@60": 11.760613136438161,
    "diagnostics.weight_sum@119": 12.895186479687487,
    "diagnostics.settled_frac@25": 0.48594308667196673,
    "diagnostics.settled_frac@60": 0.8482256394506191,
    "diagnostics.settled_frac@119": 0.9756511068856094,
    "diagnostics.withheld_reason@25": None,
    "diagnostics.withheld_reason@60": None,
    "diagnostics.withheld_reason@119": None,
    "diagnostics.pred_y0__selected@25": -4.660573991106876,
    "diagnostics.pred_y0__selected@60": -0.27040260602938293,
    "diagnostics.pred_y0__selected@119": -0.15930107822005768,
    "diagnostics.selected_y0@25": "r0.000001",
    "diagnostics.selected_y0@60": "r0.000001",
    "diagnostics.selected_y0@119": "r0.000001",
    "diagnostics.pred_y0__averaged@25": -4.660573991106876,
    "diagnostics.pred_y0__averaged@60": -0.27040260602932736,
    "diagnostics.pred_y0__averaged@119": -0.15930107822005768,
    "diagnostics.scored_clock@25": 42.0,
    "diagnostics.scored_clock@60": 93.0,
    "diagnostics.scored_clock@119": 184.0,
    "diagnostics.learned_clock@25": 40.0,
    "diagnostics.learned_clock@60": 91.0,
    "diagnostics.learned_clock@119": 174.0,
    "formula.pred_fwd@25": 0.06742601920651858,
    "formula.pred_fwd@60": 0.1275968547348632,
    "formula.pred_fwd@119": 0.04816194457154036,
    "formula.resid_fwd@25": None,
    "formula.resid_fwd@60": None,
    "formula.resid_fwd@119": None,
    "formula.weight_sum@25": 5.027057497905452,
    "formula.weight_sum@60": 11.735722004229697,
    "formula.weight_sum@119": 13.848982427761507,
    "formula.settled_frac@25": 0.48594308667196673,
    "formula.settled_frac@60": 0.8482256394506191,
    "formula.settled_frac@119": 0.9756511068856094,
    "formula.withheld_reason@25": None,
    "formula.withheld_reason@60": None,
    "formula.withheld_reason@119": None,
    "hazard.p_change@25": 0.008003872581226873,
    "hazard.p_change@60": 0.004758478496916383,
    "hazard.p_change@119": 0.045072499819063416,
    "hazard.run_mode@25": 9,
    "hazard.run_mode@60": 26,
    "hazard.run_mode@119": 48,
    "hazard.run_mean@25": 7.639592722701005,
    "hazard.run_mean@60": 19.158599272606548,
    "hazard.run_mean@119": 31.881415677275356,
    "hazard.pred_x0@25": 0.007200518560555411,
    "hazard.pred_x0@60": 0.47308870512176504,
    "hazard.pred_x0@119": 0.16178045816864342,
    "hazard.pred_x1@25": 2.343149935155821,
    "hazard.pred_x1@60": 1.6397593179245058,
    "hazard.pred_x1@119": 2.3890238034705362,
    "hazard.loglik@25": -5.568863436834559,
    "hazard.loglik@60": -6.948819191050172,
    "hazard.loglik@119": -3.6947022099283577,
    "hazard.weight_sum@25": 9.0,
    "hazard.weight_sum@60": 26.0,
    "hazard.weight_sum@119": 48.0,
    "hazard.settled_frac@25": None,
    "hazard.settled_frac@60": None,
    "hazard.settled_frac@119": None,
    "hazard.withheld_reason@25": None,
    "hazard.withheld_reason@60": None,
    "hazard.withheld_reason@119": None,
    "prune.cluster@25": 1,
    "prune.cluster@60": 1,
    "prune.cluster@119": 4,
    "prune.dist@25": 1.4468949251887275,
    "prune.dist@60": 3.3349827109152694,
    "prune.dist@119": 1.4157985257702088,
    "prune.micro_id@25": 1,
    "prune.micro_id@60": 9,
    "prune.micro_id@119": 17,
    "prune.outlier@25": False,
    "prune.outlier@60": True,
    "prune.outlier@119": True,
    "prune.n_clusters@25": 1,
    "prune.n_clusters@60": 1,
    "prune.n_clusters@119": 1,
    "prune.n_micro@25": 3,
    "prune.n_micro@60": 3,
    "prune.n_micro@119": 3,
    "prune.weight_sum@25": 6.677851059478573,
    "prune.weight_sum@60": 11.760613136438161,
    "prune.weight_sum@119": 12.895186479687487,
    "prune.settled_frac@25": 0.48594308667196673,
    "prune.settled_frac@60": 0.8482256394506191,
    "prune.settled_frac@119": 0.9756511068856094,
    "prune.withheld_reason@25": None,
    "prune.withheld_reason@60": None,
    "prune.withheld_reason@119": None,
}


def _same_numbers(got: dict[str, float | str | None], golden: dict[str, float | str | None]):
    assert set(got) == set(golden), (
        "the output schema changed: "
        f"added {sorted(set(got) - set(golden))}, removed {sorted(set(golden) - set(got))}"
    )
    for key, want in golden.items():
        have = got[key]
        if want is None or have is None:
            assert have == want, f"{key}: {have} vs {want} (null-ness must match)"
            continue
        if isinstance(want, str | bool):
            # A class name or a flag: exact, or it is a different answer.
            assert have == want, f"{key}: {have!r} vs {want!r}"
            continue
        assert abs(have - want) <= TOL * (1.0 + abs(want)), (
            f"{key}: {have!r} vs {want!r} (relative {abs(have - want) / (1 + abs(want)):.2e})"
        )


# An emptied table fails here as a changed schema: a skip would have turned
# the cross-platform check off without a word.
def test_the_pipeline_produces_the_same_numbers_everywhere():
    _same_numbers(signature(), GOLDEN)


def test_the_switches_produce_the_same_numbers_everywhere():
    """The second bank's numbers, compared as the first bank's are: the
    paths a platform is likeliest to move, which the first bank never ran
    (review 2026-10-06, TA12)."""
    _same_numbers(switched_signature(), SWITCHED_GOLDEN)


def test_the_stream_exercises_what_it_claims_to():
    """A guard on the fixture: if it stopped containing nulls or a clock gap,
    the golden comparison would still pass while covering less."""
    df = switched_stream()
    assert df["x1"].null_count() > 0, "no null feature"
    assert df["y0"].null_count() > 0, "no null target"
    assert df["label"].null_count() > 0, "no null label"
    assert df["label"].n_unique() == 3, "the label must hold both classes and null"
    assert df["g"].n_unique() == 2, "not two groups"
    assert df["session"].n_unique() == 2, "no session break"
    assert df["w"].n_unique() > 1, "weights are constant"
    gaps = df["t"].diff().drop_nulls()
    assert gaps.max() > 6.0, "no gap beyond gap_cap"
    assert gaps.min() > 0.0, "the clock must be strictly increasing"
    assert (df["wz"] == 0.0).sum() > 10, "no row of weight 0 for the second bank"
    assert df["ts"].dtype == pl.Datetime("us"), "no temporal clock for the second bank"


if __name__ == "__main__":
    # Regeneration, run as a script rather than a test that skips unless
    # asked: prints the table to paste over `GOLDEN` above.
    for name, sig in (("GOLDEN", signature()), ("SWITCHED_GOLDEN", switched_signature())):
        print(f"{name}: dict[str, float | str | None] = {{")
        for key, value in sig.items():
            print(f"    {key!r}: {value!r},")
        print("}")
