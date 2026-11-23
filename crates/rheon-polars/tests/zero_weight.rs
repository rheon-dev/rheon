//! Hard rule 9 at the bank, for every kind (docs/PLAN.md task 539; the user,
//! 2026-11-21: "Zero weight rows should not count in the clock the next row
//! should decay as if they were not present. This is always true on every
//! model."). A stream with rows of weight 0 scattered through it -- the
//! first row, a run of several, rows at a session change, after a gap past
//! `gap_cap`, at a refit boundary, with a step back, with a feature null --
//! gives every row of positive weight the outputs, to the bit, of the same
//! stream with those rows taken out, and leaves the same state, on a row
//! clock, a number clock and a temporal one. And every clock works the same
//! way: a number clock in seconds and a `Datetime` clock at the same
//! instants, the spans given as numbers on one and durations on the other,
//! give the same outputs to the bit.
//!
//! `audit` and `seqtest` are not here: a bank refuses a weight column for
//! either, so neither reads a row of weight 0 (`audit`'s core rule is in
//! `rheon-core`'s audit tests). Nor are the window formula kinds: they mirror Polars'
//! `ewm_mean_by` and `rolling_*_by`, which take no weights, so every row of
//! the frame is a row of their window (docs/records/ZERO-WEIGHT-AUDIT.md).

use polars::prelude::*;
use rheon_polars::{Bank, Spec};
use rmpv::Value;

/// Rows of the stream.
const N: usize = 420;
/// Rows per chunk.
const CHUNK: usize = 37;
/// Nanoseconds since the Unix epoch of 2024-01-01T00:00:00.
const T0_NS: i64 = 1_704_067_200_000_000_000;

/// A seeded generator in `[-1, 1)`.
fn lcg(seed: u64) -> impl FnMut() -> f64 {
    let mut s = seed;
    move || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }
}

/// Whether row `i` weighs 0: the first row, a scatter, a run of four, the
/// rows about the gap past every cap (260), about `glm`'s refit (its 60th
/// learned row) and at each session change (every 40 rows), and one row of
/// a step back (300).
fn zero(i: usize) -> bool {
    i == 0
        || i % 11 == 4
        || (100..104).contains(&i)
        || (259..262).contains(&i)
        || (63..67).contains(&i)
        || (i.is_multiple_of(40) && i > 0)
        || i == 300
}

/// The stream: features `x0`, `x1` (null now and then, a zero row among
/// them), a target `y`, a 0/1 target `yb`, a label `lab`, weights `w` (0 on
/// the rows [`zero`] names) and `wb` (the same, 1 elsewhere, for `rcov`),
/// a number clock `t` in quarter steps with one gap past every cap at row
/// 260, the same instants as a `Datetime` `ts`, a clock `tb` that steps back
/// on a zero row (300) and on a weighted one (340), each as a `Datetime`
/// too (`tbs`), a session `s` of 40 rows, label times `lt` and `lts` a
/// quarter step to three steps after each row, and a block key `g` for
/// `rcov`. And `tn`, a number clock whose steps are not sums a double holds
/// exactly (tenths, and a third), with `tbn` its step back and `ltn` its
/// label times: a row of weight 0 whose time were carried into the next
/// row's step as a sum would part from the stream without it in the last
/// bits there, where the gap read once from the last row that counts does
/// not.
fn frame() -> DataFrame {
    let mut r = lcg(7);
    let mut cols: [Vec<f64>; 12] = Default::default();
    let [t, tb, x0w, x1w, y, yb, w, wb, lt, tn, tbn, ltn] = &mut cols;
    let (mut x0, mut x1, mut lab, mut g, mut s) = (vec![], vec![], vec![], vec![], vec![]);
    let (mut clock, mut odd) = (0.0f64, 0.0f64);
    for i in 0..N {
        if i > 0 {
            clock += if i == 260 {
                7200.0
            } else {
                0.25 * (1.0 + (4.0 * (r() + 1.0)).floor())
            };
        }
        t.push(clock);
        if i > 0 {
            odd += if i == 260 {
                7200.1
            } else {
                // No draw: the other columns keep theirs.
                0.1 * ((i * 7919 % 7) + 1) as f64 + 1.0 / 3.0
            };
        }
        tn.push(odd);
        tbn.push(match i {
            300 => odd - 2.5,
            340 => odd - 3.0,
            _ => odd,
        });
        ltn.push(odd + 0.7);
        // Steps back: 2.5 units at the zero row 300, 3 at the weighted 340.
        tb.push(match i {
            300 => clock - 2.5,
            340 => clock - 3.0,
            _ => clock,
        });
        let (a, b) = (r(), r());
        x0w.push(a);
        x1w.push(b);
        x0.push((i % 17 != 5).then_some(a));
        x1.push((i % 23 != 9 && i != 103).then_some(b));
        let target = 2.0 * a - b + 0.1 * r();
        y.push(target);
        yb.push(if target > 0.0 { 1.0 } else { 0.0 });
        lab.push(if target > 0.0 { "a" } else { "b" });
        let z = zero(i);
        w.push(if z { 0.0 } else { 1.0 + 0.5 * r() });
        wb.push(if z { 0.0 } else { 1.0 });
        lt.push(clock + 0.25 * (1.0 + (12.0 * (r() + 1.0) / 2.0).floor()));
        g.push((i / 50) as i64);
        s.push((i / 40) as i64);
    }
    let ns = |v: &[f64]| {
        Series::new(
            "".into(),
            v.iter()
                .map(|c| T0_NS + (c * 1e9) as i64)
                .collect::<Vec<_>>(),
        )
        .cast(&DataType::Datetime(TimeUnit::Nanoseconds, None))
        .unwrap()
    };
    let (ts, tbs, lts) = (ns(t), ns(tb), ns(lt));
    df!("t" => t.clone(), "ts" => ts, "tb" => tb.clone(), "tbs" => tbs, "x0" => x0,
        "x1" => x1, "y" => y.clone(), "yb" => yb.clone(), "lab" => lab, "w" => w.clone(),
        "wb" => wb.clone(), "lt" => lt.clone(), "lts" => lts, "g" => g, "s" => s,
        "tn" => tn.clone(), "tbn" => tbn.clone(), "ltn" => ltn.clone())
    .unwrap()
}

/// A clock the specs run on.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Clock {
    /// No clock column: the row's number.
    Rows,
    /// `t`, a number of seconds.
    Number,
    /// `ts`, the same instants as a `Datetime`.
    Temporal,
    /// `tn`, a number clock whose steps no double sums exactly.
    Irregular,
}

impl Clock {
    /// The clock column's entry, the text after the spec's other fields.
    fn entry(self, number: &str, temporal: &str) -> String {
        match self {
            Clock::Rows => String::new(),
            Clock::Number => format!(r#", "clock": "{number}""#),
            Clock::Temporal => format!(r#", "clock": "{temporal}""#),
            Clock::Irregular => format!(r#", "clock": "{number}n""#),
        }
    }

    /// A span of `v` clock units: a number, or on a temporal clock the
    /// duration of `v` seconds, in whole milliseconds.
    fn span(self, v: f64) -> String {
        match self {
            Clock::Temporal => format!(r#""{}ms""#, (v * 1000.0).round() as i64),
            _ => format!("{v}"),
        }
    }
}

/// Every kind a bank takes a weight for, and the options that keep a
/// clock, a row count, a window or a diagnostic of their own: `(name,
/// model, the rest, the clocks it runs on)`. In the text, `@v` is a span of
/// `v` clock units ([`Clock::span`]), `#clock` the clock entry on `t` /
/// `ts`, and `#back` the one on the clock that steps back.
const SPECS: &[(&str, &str, &str, &[Clock])] = &[
    // `hazard_rows`, a count of rows on every clock; `hazard`, the clock
    // time between changepoints on every clock, rows without one (task 547).
    (
        "bocpd",
        r#"{"type": "bocpd", "hazard_rows": 250, "covariance_type": "diag"}"#,
        r#""targets": ["x0"], "features": ["x0", "x1"]"#,
        ALL,
    ),
    (
        "bocpd_timed",
        r#"{"type": "bocpd", "hazard": @250, "covariance_type": "diag"}"#,
        r#""targets": ["x0"], "features": ["x0", "x1"]"#,
        ALL,
    ),
    (
        "corrchange",
        r#"{"type": "corrchange", "kind": "monitor", "window_rows": 20, "alpha": 0.05, "alpha_adjust": "bonferroni", "scalar": false, "norm": "l1", "change_action": "flag"}"#,
        r#""targets": ["x0"], "features": ["x0", "x1"]"#,
        ALL,
    ),
    (
        "deco",
        r#"{"type": "deco", "dynamics": "ew"}"#,
        r#""targets": ["x0"], "features": ["x0", "x1"], "half_life": @50"#,
        ALL,
    ),
    (
        "ew_class",
        r#"{"type": "ew_class", "classes": ["a", "b"], "precision_prior": 1.0}"#,
        r#""targets": ["lab"], "features": ["x0"], "half_life": @50"#,
        ALL,
    ),
    (
        "ew_class_window",
        r#"{"type": "ew_class", "classes": ["a", "b"], "precision_prior": 1.0, "window_size": @30}"#,
        r#""targets": ["lab"], "features": ["x0"], "half_life": @50"#,
        ALL,
    ),
    (
        "ew_cov",
        r#"{"type": "ew_cov"}"#,
        r#""targets": ["x0"], "features": ["x0", "x1"], "half_life": @50"#,
        ALL,
    ),
    (
        "ew_cov_window",
        r#"{"type": "ew_cov", "window_size": @30, "pca": 1}"#,
        r#""targets": ["x0"], "features": ["x0", "x1"], "half_life": @20, "gap_cap": @50"#,
        ALL,
    ),
    (
        "ewridge",
        r#"{"type": "ewridge", "standardize": false, "ridge_scale": "mean", "target_gaps": "own_rows"}"#,
        r#""targets": ["y"], "features": ["x0"], "half_life": @50"#,
        ALL,
    ),
    (
        "ewridge_window",
        r#"{"type": "ewridge", "window_size": @30}"#,
        r#""targets": ["y"], "features": ["x0", "x1"], "half_life": @50, "gap_cap": @20"#,
        ALL,
    ),
    // Every diagnostic, the coefficients on a cadence, a gap past the cap
    // and drift resets: each keeps a memory of its own.
    (
        "ewridge_diagnosed",
        r#"{"type": "ewridge"}"#,
        r#""targets": ["y"], "features": ["x0", "x1"], "half_life": @40, "gap_cap": @20,
           "emit_sigma": true, "emit_zscore": true, "emit_metrics": true,
           "conformal": 0.9, "resid_quantiles": [0.5, 0.9], "emit_autocorr": true,
           "emit_drift": true, "drift_action": "reset", "drift_threshold": @8,
           "emit_calibration": true, "emit_breaks": true, "emit_robust_se": true,
           "emit_specification": true, "emit_tails": true, "emit_influence": true,
           "emit_feature_health": true, "emit_se_coef": true, "emit_error_inflation": true,
           "emit_clocks": true, "coef_every": @3, "min_weight": 3.0"#,
        ALL,
    ),
    // Each other drift detector, resetting the model on a flag: each
    // counts rows of positive weight, ADWIN its scan cadence and KSWIN its
    // draw among them (PLAN 508). The settings make each flag on this
    // stream (`every_drift_kind_flags_here`); ADWIN watches a slow `sgd`,
    // whose errors fall as it learns, so its window has a change to cut.
    (
        "drift_cusum",
        r#"{"type": "ewridge"}"#,
        r#""targets": ["y"], "features": ["x0", "x1"], "half_life": @40, "emit_sigma": true,
           "emit_drift": true, "drift_kind": "cusum", "drift_delta": 0.25,
           "drift_threshold": @2, "drift_action": "reset""#,
        ALL,
    ),
    (
        "drift_adwin",
        r#"{"type": "sgd", "loss": "squared", "learning_rate": 0.005, "schedule": "constant", "standardize": false}"#,
        r#""targets": ["y"], "features": ["x0", "x1"], "half_life": @40,
           "emit_drift": true, "drift_kind": "adwin", "drift_action": "reset",
           "drift_options": {"delta": 0.1, "clock": 3, "grace_period": 5, "max_width": 200}"#,
        ALL,
    ),
    (
        "drift_adwin_kept",
        r#"{"type": "sgd", "loss": "squared", "learning_rate": 0.005, "schedule": "constant", "standardize": false}"#,
        r#""targets": ["y"], "features": ["x0", "x1"], "half_life": @40,
           "emit_drift": true, "drift_kind": "adwin",
           "drift_options": {"delta": 0.1, "clock": 2, "restart": false, "max_buckets": 2}"#,
        ALL,
    ),
    (
        "drift_ddm",
        r#"{"type": "ewridge"}"#,
        r#""targets": ["y"], "features": ["x0", "x1"], "half_life": @40,
           "emit_drift": true, "drift_kind": "ddm", "drift_action": "reset",
           "drift_options": {"error_above": 0.04, "warm_start": 10, "drift_threshold": 1.5}"#,
        ALL,
    ),
    (
        "drift_eddm",
        r#"{"type": "ewridge"}"#,
        r#""targets": ["y"], "features": ["x0", "x1"], "half_life": @40,
           "emit_drift": true, "drift_kind": "eddm", "drift_action": "reset",
           "drift_options": {"error_above": 0.04, "warm_start": 5, "beta": 0.95}"#,
        ALL,
    ),
    (
        "drift_kswin",
        r#"{"type": "ewridge"}"#,
        r#""targets": ["y"], "features": ["x0", "x1"], "half_life": @40,
           "emit_drift": true, "drift_kind": "kswin", "drift_action": "reset",
           "drift_options": {"alpha": 0.3, "window_size": 24, "stat_size": 8, "seed": 3}"#,
        ALL,
    ),
    (
        "drift_kswin_kept",
        r#"{"type": "ewridge"}"#,
        r#""targets": ["y"], "features": ["x0", "x1"], "half_life": @40,
           "emit_drift": true, "drift_kind": "kswin",
           "drift_options": {"alpha": 0.3, "window_size": 24, "stat_size": 8, "restart": false}"#,
        ALL,
    ),
    (
        "ewridge_grid",
        r#"{"type": "ewridge"}"#,
        r#""targets": ["y"], "features": ["x0", "x1"], "half_life": [@10, @80],
           "emit_averaged": true, "emit_selected": true, "emit_sigma": true"#,
        ALL,
    ),
    (
        "ewridge_session",
        r#"{"type": "ewridge"}"#,
        r#""targets": ["y"], "features": ["x0", "x1"], "half_life": @40, "session": "s",
           "session_gap": @2"#,
        ALL,
    ),
    (
        "ewridge_session_reset",
        r#"{"type": "ewridge"}"#,
        r#""targets": ["y"], "features": ["x0", "x1"], "half_life": @40, "session": "s",
           "session_gap": "reset""#,
        ALL,
    ),
    (
        "ewridge_embargo",
        r#"{"type": "ewridge"}"#,
        r#""targets": ["y"], "features": ["x0", "x1"], "half_life": @40, "gap_cap": @20,
           "embargo": @3, "emit_sigma": true, "conformal": 0.8"#,
        ALL,
    ),
    (
        "ewridge_label_clock",
        r#"{"type": "ewridge"}"#,
        r#""targets": ["y"], "features": ["x0", "x1"], "half_life": @40, "gap_cap": @20,
           "emit_sigma": true #label"#,
        CLOCKED,
    ),
    (
        "ewridge_step_back",
        r#"{"type": "ewridge"}"#,
        r#""targets": ["y"], "features": ["x0", "x1"], "half_life": @40,
           "restart_after_step_back": @1 #back"#,
        CLOCKED,
    ),
    (
        "expectile",
        r#"{"type": "expectile", "expectile": 0.8, "refit_rows": 60}"#,
        r#""targets": ["y"], "features": ["x0", "x1"], "half_life": @50"#,
        ALL,
    ),
    (
        "expectile_rows",
        r#"{"type": "expectile", "expectile": 0.8, "window_rows": 40}"#,
        r#""targets": ["y"], "features": ["x0", "x1"], "half_life": "inf""#,
        UNTIMED,
    ),
    (
        "expectile_clock",
        r#"{"type": "expectile", "expectile": 0.3, "window_size": @20, "closed": "both"}"#,
        r#""targets": ["y"], "features": ["x0", "x1"], "half_life": "inf""#,
        ALL,
    ),
    (
        "ftrl",
        r#"{"type": "ftrl", "strict_binary": false, "loss": "logistic"}"#,
        r#""targets": ["yb"], "features": ["x0"], "half_life": @50"#,
        ALL,
    ),
    (
        "garch",
        r#"{"type": "garch", "start_rows": 50, "horizon": 2}"#,
        r#""targets": ["y"], "features": []"#,
        ALL,
    ),
    (
        "garch_span",
        r#"{"type": "garch", "start_span": @20}"#,
        r#""targets": ["y"], "features": []"#,
        ALL,
    ),
    (
        "dcc",
        r#"{"type": "dcc", "asymmetric": true, "margins": "none", "start_rows": 50}"#,
        r#""targets": ["y", "x0"], "features": [], "half_life": @50"#,
        ALL,
    ),
    (
        "dcc_margins",
        r#"{"type": "dcc", "margin_start_rows": 50, "start_rows": 50}"#,
        r#""targets": ["y", "x0"], "features": [], "half_life": "inf""#,
        ALL,
    ),
    (
        "dcc_span",
        r#"{"type": "dcc", "margins": "none", "start_span": @20}"#,
        r#""targets": ["y", "x0"], "features": [], "half_life": @80"#,
        ALL,
    ),
    (
        "pca",
        r#"{"type": "pca", "n_components": 1}"#,
        r#""targets": ["x0"], "features": ["x0", "x1"], "half_life": @50"#,
        ALL,
    ),
    (
        "pca_oja",
        r#"{"type": "pca", "method": "oja", "n_components": 2}"#,
        r#""targets": ["x0"], "features": ["x0", "x1"], "half_life": @50"#,
        ALL,
    ),
    (
        "glm",
        r#"{"type": "glm", "refit_rows": 60}"#,
        r#""targets": ["yb"], "features": ["x0", "x1"], "half_life": @50"#,
        ALL,
    ),
    (
        "hawkes_rows",
        r#"{"type": "hawkes", "kernel_decays": [0.5, 2.0], "window_events": 25}"#,
        r#""targets": ["y", "x0"], "features": [], "half_life": "inf""#,
        UNTIMED,
    ),
    (
        "hawkes_clock",
        r#"{"type": "hawkes", "kernel_half_lives": [@1.5, @0.25], "window_size": @20}"#,
        r#""targets": ["y", "x0"], "features": [], "half_life": "inf""#,
        ALL,
    ),
    (
        "hawkes_clock_both",
        r#"{"type": "hawkes", "kernel_half_lives": [@1.5, @0.25], "window_size": @20, "closed": "both"}"#,
        r#""targets": ["y", "x0"], "features": [], "half_life": "inf""#,
        ALL,
    ),
    (
        "hawkes",
        r#"{"type": "hawkes", "kernel_half_lives": [@1.5, @0.25]}"#,
        r#""targets": ["y", "x0"], "features": [], "half_life": @50"#,
        ALL,
    ),
    (
        "hmm",
        r#"{"type": "hmm", "n_components": 2, "covariance_type": "full", "precision_prior": 0.1, "learn": true}"#,
        r#""targets": ["x0"], "features": ["x0", "x1"], "half_life": @50"#,
        ALL,
    ),
    (
        "realized",
        r#"{"type": "realized", "lm_window": 30}"#,
        r#""targets": ["y", "x0"], "features": [], "half_life": @50"#,
        ALL,
    ),
    (
        "realized_rows",
        r#"{"type": "realized", "window_rows": 40}"#,
        r#""targets": ["y", "x0"], "features": [], "half_life": "inf""#,
        UNTIMED,
    ),
    (
        "realized_clock",
        r#"{"type": "realized", "window_size": @20, "lm_window": 12}"#,
        r#""targets": ["y", "x0"], "features": [], "half_life": "inf""#,
        ALL,
    ),
    (
        "realized_clock_both",
        r#"{"type": "realized", "window_size": @20, "closed": "both"}"#,
        r#""targets": ["y", "x0"], "features": [], "half_life": "inf""#,
        ALL,
    ),
    (
        "holt",
        r#"{"type": "holt"}"#,
        r#""targets": ["y"], "features": [], "half_life": @50"#,
        ALL,
    ),
    (
        "huber",
        r#"{"type": "huber", "standardize": false}"#,
        r#""targets": ["y"], "features": ["x0"], "half_life": @50"#,
        ALL,
    ),
    (
        "kalman",
        r#"{"type": "kalman", "coef_half_life": @50, "share_p": false, "standardize": true}"#,
        r#""targets": ["y"], "features": ["x0"], "half_life": @50"#,
        ALL,
    ),
    (
        "kmeans",
        r#"{"type": "kmeans", "n_clusters": 2}"#,
        r#""targets": ["x0"], "features": ["x0", "x1"], "half_life": @50"#,
        ALL,
    ),
    (
        "lasso",
        r#"{"type": "lasso", "penalties": [0.1, 0.0], "target_gaps": "own_rows"}"#,
        r#""targets": ["y"], "features": ["x0"], "half_life": @50"#,
        ALL,
    ),
    (
        "marginal",
        r#"{"type": "marginal"}"#,
        r#""targets": ["y"], "features": ["x0"], "half_life": @50"#,
        ALL,
    ),
    (
        "micro",
        r#"{"type": "micro", "eps": 0.3}"#,
        r#""targets": ["x0"], "features": ["x0", "x1"], "half_life": @50"#,
        ALL,
    ),
    (
        "pa",
        r#"{"type": "pa", "mode": "pa1"}"#,
        r#""targets": ["y"], "features": ["x0"], "half_life": @50"#,
        ALL,
    ),
    (
        "quantile",
        r#"{"type": "quantile", "quantile": 0.5, "standardize": false}"#,
        r#""targets": ["y"], "features": ["x0"], "half_life": @50"#,
        ALL,
    ),
    (
        "rcov",
        r#"{"type": "rcov", "kind": "kernel", "kernel": "parzen", "jitter": 2, "psd": true, "block_rows": 100}"#,
        r#""targets": ["x0"], "features": ["x0", "x1"], "group": "g", "group_close": "monotone", "weight": "wb""#,
        ALL,
    ),
    (
        "rls",
        r#"{"type": "rls"}"#,
        r#""targets": ["y"], "features": ["x0"], "half_life": @50"#,
        ALL,
    ),
    (
        "sgd",
        r#"{"type": "sgd", "loss": "squared", "learning_rate": 0.01, "schedule": "constant", "standardize": false}"#,
        r#""targets": ["y"], "features": ["x0"], "half_life": @50"#,
        ALL,
    ),
    (
        "sgd_logistic",
        r#"{"type": "sgd", "loss": "logistic", "learning_rate": 0.01, "schedule": "constant", "standardize": false}"#,
        r#""targets": ["yb"], "features": ["x0"], "half_life": @50"#,
        ALL,
    ),
];

const ALL: &[Clock] = &[Clock::Rows, Clock::Number, Clock::Temporal];
const CLOCKED: &[Clock] = &[Clock::Number, Clock::Temporal];
/// `kernel_decays` is a rate per clock unit, which a temporal clock refuses
/// for its duration form, `kernel_half_lives` (`hawkes_clock`).
const UNTIMED: &[Clock] = &[Clock::Rows, Clock::Number];

/// The spec's text on `clock`, weighted by `w` unless it names a weight.
fn spec_text(name: &str, model: &str, rest: &str, clock: Clock) -> String {
    let mut text = format!(r#"{{"name": "{name}", "model": {model}, {rest}"#);
    // A clock needs a `gap_cap`, and the row clock takes none.
    match (clock, text.find(r#", "gap_cap": @"#)) {
        (Clock::Rows, Some(at)) => {
            let end = text[at + 14..]
                .find(|c: char| !(c.is_ascii_digit() || c == '.'))
                .map_or(text.len(), |e| at + 14 + e);
            text.replace_range(at..end, "");
        }
        (Clock::Number | Clock::Temporal | Clock::Irregular, None) => {
            text.push_str(r#", "gap_cap": @100"#)
        }
        _ => {}
    }
    // `@v`: a span of `v` clock units.
    while let Some(at) = text.find('@') {
        let end = text[at + 1..]
            .find(|c: char| !(c.is_ascii_digit() || c == '.'))
            .map_or(text.len(), |e| at + 1 + e);
        let v: f64 = text[at + 1..end].parse().unwrap();
        text.replace_range(at..end, &clock.span(v));
    }
    let label = match clock {
        Clock::Rows => String::new(),
        Clock::Number => r#", "label_clock": "lt""#.into(),
        Clock::Irregular => r#", "label_clock": "ltn""#.into(),
        Clock::Temporal => r#", "label_clock": "lts""#.into(),
    };
    text = text.replace(" #label", &label);
    let back = clock.entry("tb", "tbs");
    let uses_back = text.contains(" #back");
    text = text.replace(" #back", &back);
    if !uses_back {
        text.push_str(&clock.entry("t", "ts"));
    }
    if !text.contains(r#""weight""#) {
        text.push_str(r#", "weight": "w""#);
    }
    text.push('}');
    text
}

fn spec_of(text: &str) -> Spec {
    serde_json::from_str(text).unwrap_or_else(|e| panic!("{text}: {e}"))
}

/// The bank's output over `df` fed in chunks of [`CHUNK`], and its state,
/// decoded.
fn run(spec: &Spec, df: &DataFrame) -> (DataFrame, Value) {
    let lens = vec![CHUNK; df.height().div_ceil(CHUNK)];
    try_run(spec, df, &lens).unwrap_or_else(|e| panic!("{e}"))
}

/// [`run`], in chunks of `lens` rows (the last cut to the frame), or the
/// bank's refusal.
fn try_run(spec: &Spec, df: &DataFrame, lens: &[usize]) -> Result<(DataFrame, Value), String> {
    let mut bank = Bank::new(vec![spec.clone()]).map_err(|e| e.to_string())?;
    let mut parts: Vec<DataFrame> = Vec::new();
    let mut at = 0;
    for &len in lens {
        let len = len.min(df.height() - at);
        if len == 0 {
            continue;
        }
        let cols = bank
            .fit_predict_from(&df.slice(at as i64, len), at)
            .map_err(|e| format!("{}: {e}", spec.name))?;
        parts.push(DataFrame::new(len, cols).unwrap());
        at += len;
    }
    let mut out = parts.remove(0);
    for p in &parts {
        out.vstack_mut(p).unwrap();
    }
    // The spec's struct, one column a field.
    let out = out.unnest([spec.name.as_str()], None).unwrap();
    let bytes = bank.save_bytes().unwrap();
    let state = rmpv::decode::read_value(&mut &bytes[..]).unwrap();
    Ok((out, state))
}

/// Every value of a column, as bits where it is a number and as text
/// otherwise, so that two columns compare to the bit, nulls included.
fn cells(c: &Column) -> Vec<String> {
    let s = c.as_materialized_series();
    match s.dtype() {
        DataType::Float64 => s
            .f64()
            .unwrap()
            .iter()
            .map(|v| v.map_or("null".into(), |v| format!("{:016x}", v.to_bits())))
            .collect(),
        DataType::List(inner) if **inner == DataType::Float64 => (0..s.len())
            .map(|i| s.list().unwrap().get_as_series(i))
            .map(|v| {
                v.map_or("null".into(), |v| {
                    v.f64()
                        .unwrap()
                        .iter()
                        .map(|x| x.map_or("null".into(), |x| format!("{:016x}", x.to_bits())))
                        .collect::<Vec<String>>()
                        .join(",")
                })
            })
            .collect(),
        _ => (0..s.len())
            .map(|i| format!("{:?}", s.get(i).unwrap()))
            .collect(),
    }
}

/// The parts of a bank's state that describe the rows fed, which a row of
/// weight 0 is: what `rows_processed`, `rows_zero_weight` and the data
/// summary report (a closed group's among them, with the range of its clock
/// column), and the checksum over them. No model reads them.
const COUNTS: &[&str] = &[
    "rows_seen",
    "rows_fed",
    "summary",
    "clock_min",
    "clock_max",
    "checksum",
];

/// The state with the row counts taken out, wherever they sit.
fn without_counts(v: &Value) -> Value {
    match v {
        Value::Map(m) => Value::Map(
            m.iter()
                .filter(|(k, _)| !k.as_str().is_some_and(|k| COUNTS.contains(&k)))
                .map(|(k, v)| (k.clone(), without_counts(v)))
                .collect(),
        ),
        Value::Array(a) => Value::Array(a.iter().map(without_counts).collect()),
        other => other.clone(),
    }
}

/// The first place two states part, as a path; a double compared by its
/// bits.
fn first_difference(a: &Value, b: &Value, path: &str) -> Option<String> {
    match (a, b) {
        (Value::Map(x), Value::Map(y)) if x.len() == y.len() => {
            x.iter().zip(y).find_map(|((ka, va), (kb, vb))| {
                if ka != kb {
                    return Some(format!("{path}: key {ka} against {kb}"));
                }
                first_difference(va, vb, &format!("{path}.{ka}"))
            })
        }
        (Value::Array(x), Value::Array(y)) if x.len() == y.len() => x
            .iter()
            .zip(y)
            .enumerate()
            .find_map(|(i, (v, w))| first_difference(v, w, &format!("{path}[{i}]"))),
        (Value::F64(x), Value::F64(y)) if x.to_bits() == y.to_bits() => None,
        (Value::F32(x), Value::F32(y)) if x.to_bits() == y.to_bits() => None,
        (Value::F64(_), _) | (Value::F32(_), _) => Some(format!("{path}: {a} against {b}")),
        _ if a == b => None,
        _ => Some(format!(
            "{path}: {} against {}",
            short(a.to_string()),
            short(b.to_string())
        )),
    }
}

fn short(s: String) -> String {
    if s.len() > 120 {
        format!("{}...", &s[..120])
    } else {
        s
    }
}

/// What parts between the two streams, by field: the rows of positive
/// weight that differ, and the first of them.
fn parted(with: &DataFrame, weights: &[f64], without: &DataFrame) -> Vec<String> {
    let kept: Vec<usize> = (0..weights.len()).filter(|&i| weights[i] != 0.0).collect();
    let mut found = Vec::new();
    for c in without.columns() {
        let name = c.name().as_str();
        let Ok(other) = with.column(name) else {
            found.push(format!("{name}: missing"));
            continue;
        };
        let (a, b) = (cells(other), cells(c));
        let differ: Vec<usize> = kept
            .iter()
            .enumerate()
            .filter(|(k, i)| a[**i] != b[*k])
            .map(|(k, _)| k)
            .collect();
        if let Some(&k) = differ.first() {
            // The largest move in ulps, where both are numbers.
            let ulps = differ
                .iter()
                .filter_map(|&k| ulps(&a[kept[k]], &b[k]))
                .max()
                .map_or(String::new(), |u| format!(", at most {u} ulps"));
            found.push(format!(
                "{name}: {} rows{ulps}, the first at weighted row {k} (input row {}): {} \
                 against {}",
                differ.len(),
                kept[k],
                a[kept[k]],
                b[k]
            ));
        }
    }
    found
}

/// The doubles between two cells that each hold one number's bits.
fn ulps(a: &str, b: &str) -> Option<u64> {
    let bits = |s: &str| u64::from_str_radix(s, 16).ok().filter(|_| s.len() == 16);
    let order = |b: u64| {
        if b >> 63 == 1 {
            -((b & !(1 << 63)) as i128)
        } else {
            b as i128
        }
    };
    let (x, y) = (bits(a)?, bits(b)?);
    Some(order(x).abs_diff(order(y)) as u64)
}

/// The stream with its zero-weight rows, and without them (`rcov`'s 0/1
/// weight is 0 on the same rows).
fn streams() -> (DataFrame, DataFrame, Vec<f64>) {
    let df = frame();
    let w: Vec<f64> = df
        .column("w")
        .unwrap()
        .f64()
        .unwrap()
        .into_no_null_iter()
        .collect();
    let mask = BooleanChunked::from_iter_values("m".into(), w.iter().map(|&v| v != 0.0));
    let without = df.filter(&mask).unwrap();
    (df, without, w)
}

/// Every kind a bank weights is here: a new kind fails until it joins.
#[test]
fn every_kind_but_the_two_that_take_no_weight_is_here() {
    let mut have: Vec<String> = SPECS
        .iter()
        .map(|(name, model, rest, clocks)| {
            spec_of(&spec_text(name, model, rest, clocks[0]))
                .model
                .kind_name()
                .to_string()
        })
        .chain(["audit", "seqtest"].map(String::from))
        .collect();
    have.sort();
    have.dedup();
    let mut want: Vec<String> = rheon_polars::ModelKind::KINDS
        .iter()
        .map(|k| k.to_string())
        .collect();
    want.sort();
    assert_eq!(have, want);
}

/// The drift detectors' cases above flag on this stream, on every clock,
/// so the comparison reads their resets and restarts, not only their quiet
/// rows.
#[test]
fn every_drift_kind_flags_here() {
    let (with_df, _, _) = streams();
    for (name, model, rest, clocks) in SPECS.iter().filter(|s| s.0.starts_with("drift_")) {
        for clock in clocks.iter().copied() {
            let (out, _) = run(&spec_of(&spec_text(name, model, rest, clock)), &with_df);
            let flags = out
                .column("drift_y")
                .unwrap()
                .bool()
                .unwrap()
                .sum()
                .unwrap_or(0);
            assert!(flags > 0, "{name} on {clock:?}: no flag");
        }
    }
}

/// Every kind, every clock: the weighted rows' outputs and the final state
/// of the stream with zero-weight rows are those of the stream without them.
/// `RHEON_ZERO_WEIGHT_AUDIT=1` reports every case that parts instead of
/// failing at the first, which is how docs/records/ZERO-WEIGHT-AUDIT.md
/// was written.
#[test]
fn a_zero_weight_row_counts_in_no_clock_on_every_kind() {
    let (with_df, without_df, weights) = streams();
    assert!(weights.iter().filter(|w| **w == 0.0).count() > 40);
    let audit = std::env::var_os("RHEON_ZERO_WEIGHT_AUDIT").is_some();
    let mut failures = Vec::new();
    let mut cases = 0;
    for (name, model, rest, clocks) in SPECS {
        let irregular = clocks.contains(&Clock::Number).then_some(Clock::Irregular);
        for clock in clocks.iter().copied().chain(irregular) {
            let text = spec_text(name, model, rest, clock);
            let spec = spec_of(&text);
            cases += 1;
            // The same chunks: those of the stream without the zero-weight
            // rows end at the same rows of positive weight, so the last such
            // row of each, which carries `coef`, is the same row.
            let ends: Vec<usize> = (0..with_df.height()).step_by(CHUNK).collect();
            let with_lens = vec![CHUNK; ends.len()];
            let without_lens: Vec<usize> = ends
                .iter()
                .map(|&a| {
                    let b = (a + CHUNK).min(weights.len());
                    weights[a..b].iter().filter(|w| **w != 0.0).count()
                })
                .collect();
            let (with, without) = (
                try_run(&spec, &with_df, &with_lens),
                try_run(&spec, &without_df, &without_lens),
            );
            let (Ok((with, state_with)), Ok((without, state_without))) = (with, without) else {
                let line = format!(
                    "{name} on {clock:?}: refused: {:?}",
                    try_run(&spec, &with_df, &with_lens).err()
                );
                assert!(audit, "{line}");
                failures.push(line);
                continue;
            };
            let mut found = parted(&with, &weights, &without);
            if let Some(d) = first_difference(
                &without_counts(&state_with),
                &without_counts(&state_without),
                "state",
            ) {
                found.push(d);
            }
            if !found.is_empty() {
                let line = format!("{name} on {clock:?}:\n  {}", found.join("\n  "));
                if !audit {
                    panic!("{line}");
                }
                failures.push(line);
            }
        }
    }
    if audit {
        eprintln!(
            "{} of {cases} cases part:\n{}",
            failures.len(),
            failures.join("\n")
        );
    }
    assert!(failures.is_empty() || audit);
    assert!(cases > 130, "{cases} cases");
}

/// Every clock works the same way (the user, 2026-11-21: "All clocks should
/// work the same way, if the clock is a time then 5m should work"): each
/// kind on the number clock `t`, in seconds, and on `ts`, the same instants
/// as a `Datetime`, with every span a number on the first and a duration
/// on the second, gives the same outputs to the bit, zero-weight rows and
/// all. The steps are quarter seconds, so both clocks hold them exactly.
#[test]
fn a_number_clock_and_a_temporal_one_agree_on_every_kind() {
    let (df, _, _) = streams();
    let mut failures = Vec::new();
    let mut cases = 0;
    for (name, model, rest, clocks) in SPECS {
        if !(clocks.contains(&Clock::Temporal) && clocks.contains(&Clock::Number)) {
            continue;
        }
        let number = spec_of(&spec_text(name, model, rest, Clock::Number));
        let temporal = spec_of(&spec_text(name, model, rest, Clock::Temporal));
        let (a, _) = run(&number, &df);
        let (b, _) = run(&temporal, &df);
        cases += 1;
        let all: Vec<f64> = vec![1.0; df.height()];
        // The clocks a row shows are the clock column's own values; a grid's
        // fields are named by the span as written, so they are read by
        // place.
        let a = a.drop_many(clock_fields(&a));
        let mut b = b.drop_many(clock_fields(&b));
        b.set_column_names(&a.get_column_names_owned()).unwrap();
        let found = parted(&a, &all, &b);
        if !found.is_empty() {
            failures.push(format!("{name}:\n  {}", found.join("\n  ")));
        }
    }
    assert!(cases > 30, "{cases} cases");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// The fields that show a row's clock, which a temporal clock shows as a
/// `Datetime`.
fn clock_fields(df: &DataFrame) -> Vec<String> {
    df.get_column_names()
        .iter()
        .filter(|n| n.ends_with("_clock") || n.ends_with("learned_clock"))
        .map(|n| n.to_string())
        .collect()
}

/// A row of weight 0 is scored as `predict` scores the row against the
/// state the rows before it left: on each clock, at every zero row of the
/// stream, the prediction the bank wrote is the one a bank fed the rows
/// before it gives the row by `predict`.
#[test]
fn a_zero_weight_row_is_scored_as_predict_scores_it() {
    let (df, _, weights) = streams();
    for clock in ALL {
        let spec = spec_of(&spec_text(
            "m",
            r#"{"type": "ewridge"}"#,
            r#""targets": ["y"], "features": ["x0", "x1"], "half_life": @40, "gap_cap": @20,
               "emit_sigma": true"#,
            *clock,
        ));
        let (out, _) = run(&spec, &df);
        let pred = cells(out.column("pred_y").unwrap());
        let sigma = cells(out.column("sigma_y").unwrap());
        let mut bank = Bank::new(vec![spec.clone()]).unwrap();
        let mut fed = 0;
        let mut checked = 0;
        for i in (0..df.height()).filter(|&i| weights[i] == 0.0) {
            if i > fed {
                bank.fit_predict_from(&df.slice(fed as i64, i - fed), fed)
                    .unwrap();
                fed = i;
            }
            let row = df.slice(i as i64, 1);
            let scored = DataFrame::new(1, bank.predict(&row).unwrap())
                .unwrap()
                .unnest(["m"], None)
                .unwrap();
            assert_eq!(
                cells(scored.column("pred_y").unwrap())[0],
                pred[i],
                "{clock:?}: row {i}"
            );
            assert_eq!(
                cells(scored.column("sigma_y").unwrap())[0],
                sigma[i],
                "{clock:?}: row {i}: sigma"
            );
            checked += 1;
        }
        assert!(checked > 40);
    }
}

// --- what a chunk's refusal and a session close read (W9) ---
//
// The property test drew a stream whose rows of weight 0 sat between a step
// back and the row it is measured from. These hold the outcome to every
// cutting of a short stream: the same refusal at the same input row, or the
// same outputs and state.

/// Every cutting of `n` rows worth a look: chunks of each length, and each
/// single cut.
fn cuttings(n: usize) -> Vec<Vec<usize>> {
    let mut all: Vec<Vec<usize>> = (1..=n).map(|k| vec![k; n.div_ceil(k)]).collect();
    all.extend((1..n).map(|cut| vec![cut, n - cut]));
    all
}

/// The outcome of `spec` over `df` under every cutting: they all agree, and
/// this is it.
fn under_every_cutting(spec: &Spec, df: &DataFrame) -> Result<(DataFrame, Value), String> {
    let whole = try_run(spec, df, &[df.height()]);
    for lens in cuttings(df.height()) {
        let cut = try_run(spec, df, &lens);
        match (&whole, &cut) {
            (Err(a), Err(b)) => assert_eq!(a, b, "{lens:?}"),
            (Ok((a, sa)), Ok((b, sb))) => {
                let (a, b) = (
                    a.drop_many(["coef", "support_coef"]),
                    b.drop_many(["coef", "support_coef"]),
                );
                for c in a.columns() {
                    let (x, y) = (cells(c), cells(b.column(c.name()).unwrap()));
                    assert_eq!(x, y, "{lens:?}: {}", c.name());
                }
                assert_eq!(a.width(), b.width(), "{lens:?}");
                assert_eq!(first_difference(sa, sb, ""), None, "{lens:?}");
            }
            _ => panic!("{lens:?}: {:?} against {:?}", whole.is_ok(), cut.is_ok()),
        }
    }
    whole
}

fn small_spec(rest: &str) -> Spec {
    spec_of(&format!(
        r#"{{"name": "m", "model": {{"type": "ewridge"}}, "targets": ["y"], "features": ["x"],
            "clock": "t", "gap_cap": 100.0, "half_life": 5.0, "weight": "w" {rest}}}"#
    ))
}

/// The falsifying stream, cut down: the clock steps 0, 1, 2 and back to -5,
/// the rows at 1 and 2 of weight 0. The step back is 5 from row 1, the last
/// that counts, not 7 from the row before: no more than the minimum, so
/// every cutting refuses at row 4. A null weight skips its row, and a
/// skipped row still moves the clock, so row 1 counts and row 4 steps.
#[test]
fn a_step_back_past_zero_weight_rows_is_measured_from_the_last_row_that_counts() {
    let df = df!(
        "t" => [0.0, 0.0, 1.0, 2.0, -5.0, -5.0],
        "x" => [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
        "y" => [1.0, 2.0, 1.0, 3.0, 2.0, 1.0],
        "w" => [Some(1.0), None, Some(0.0), Some(0.0), None, Some(1.0)],
    )
    .unwrap();
    let spec = small_spec(r#", "restart_after_step_back": 5.0"#);
    let err = under_every_cutting(&spec, &df).unwrap_err();
    assert!(err.contains("steps back by 5 at row 4 "), "{err}");
    // One unit further back the step is past the minimum, and every
    // cutting restarts the stream there alike.
    let df = df
        .lazy()
        .with_column(
            when(col("t").lt(lit(0.0)))
                .then(lit(-6.0))
                .otherwise(col("t"))
                .alias("t"),
        )
        .collect()
        .unwrap();
    under_every_cutting(&spec, &df).unwrap();
}

/// A step that is not a number of clock units is read from the last row
/// that counts, too: a row of weight 0 at 0 between -1e308 and 1e308 does
/// not halve it. Every cutting refuses at row 2 as a step too large for a
/// float, where one chunk let it through to the step-back check, which
/// named a step of infinity a step back. A row of weight 0 is still scored
/// over its own step, so that step is refused on it (row 1 of the second
/// stream).
#[test]
fn a_step_too_large_for_a_float_is_read_past_zero_weight_rows() {
    let spec = small_spec("");
    for (t, w, row) in [
        ([-1e308, 0.0, 1e308], [1.0, 0.0, 1.0], 2),
        ([-1e308, 1e308, -1e308], [1.0, 0.0, 1.0], 1),
    ] {
        let df = df!("t" => t, "x" => [1.0, 2.0, 3.0], "y" => [1.0, 2.0, 1.0], "w" => w).unwrap();
        let err = under_every_cutting(&spec, &df).unwrap_err();
        assert!(err.contains(&format!("at row {row} ")), "{err}");
        assert!(err.contains("a difference too large for a float"), "{err}");
    }
}

/// Two groups that each step back: the chunk is refused at the first such
/// row in the input, whichever group's check ran first. The longer group,
/// `b`, steps back at row 7, the shorter, `a`, at row 4 behind its row of
/// weight 0 at row 2, and a cut anywhere before row 7 names row 4, so one
/// chunk does too.
#[test]
fn a_chunk_is_refused_at_its_first_refused_row() {
    let df = df!(
        "g" => ["a", "b", "a", "b", "a", "b", "b", "b"],
        "t" => [0.0, 0.0, 3.0, 1.0, -2.0, 2.0, 3.0, 1.0],
        "x" => [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0],
        "y" => [1.0, 2.0, 1.0, 3.0, 2.0, 1.0, 2.0, 1.0],
        "w" => [1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    )
    .unwrap();
    let spec = small_spec(r#", "group": "g", "restart_after_step_back": 5.0"#);
    let err = under_every_cutting(&spec, &df).unwrap_err();
    assert!(err.contains("steps back by 2 at row 4 "), "{err}");
}

/// Under `label_clock`, a label time before its row is refused at its row
/// though the clock steps back later in the chunk: the label check reads
/// the rows before the refused step, where it once read none.
#[test]
fn a_label_refused_before_a_step_back_is_the_one_named() {
    let df = df!(
        "t" => [0.0, 1.0, 2.0, 3.0, 1.0],
        "lt" => [0.5, 0.5, 2.5, 3.5, 1.5],
        "x" => [1.0, 2.0, 3.0, 1.0, 2.0],
        "y" => [1.0, 2.0, 1.0, 3.0, 2.0],
        "w" => [1.0, 1.0, 0.0, 1.0, 1.0],
    )
    .unwrap();
    let spec = small_spec(r#", "label_clock": "lt""#);
    let err = under_every_cutting(&spec, &df).unwrap_err();
    assert!(err.contains("before clock column \"t\" at row 1 "), "{err}");
}

/// Under `group_close = "session"`, a row of weight 0 of another session is
/// no session boundary (hard rule 9: it counts in no session rule): the
/// span goes on past it, and the row after it, of the span's session, is
/// learned on the span's state. It once closed the span twice in one chunk
/// and once where a chunk ended after it, so the closed groups and the
/// group's row count parted with the cutting. A row of weight 0 opens no
/// span either: alone in a new session at a chunk's end, it closes nothing.
#[test]
fn a_zero_weight_row_of_another_session_closes_no_span() {
    let df = df!(
        "g" => [0i64; 7],
        "s" => [0i64, 0, 1, 0, 0, 2, 2],
        "t" => [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "x" => [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0],
        "y" => [1.0, 2.0, 1.0, 3.0, 2.0, 1.0, 2.0],
        "w" => [1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0],
    )
    .unwrap();
    let spec = small_spec(r#", "group": "g", "session": "s", "group_close": "session""#);
    let (out, _) = under_every_cutting(&spec, &df).unwrap();
    // Row 3 is learned on the span rows 0 and 1 began: its weight before
    // the row is theirs, decayed, and not a fresh stream's 0.
    let w = out.column("weight_sum").unwrap().f64().unwrap();
    assert!(w.get(3).unwrap() > 1.0, "{out}");
    // Row 5 opens session 2, a fresh span; row 6, of weight 0, ends it.
    assert_eq!(w.get(5), Some(0.0), "{out}");
    let closed = |lens: &[usize]| {
        let mut bank = Bank::new(vec![spec.clone()]).unwrap();
        let mut at = 0;
        for &len in lens {
            let len = len.min(df.height() - at);
            bank.fit_predict_from(&df.slice(at as i64, len), at)
                .unwrap();
            at += len;
        }
        bank.closed_groups(None, true).unwrap()
    };
    let one = closed(&[df.height()]);
    // One span closed, session 0's, at row 5.
    assert_eq!(one.height(), 1, "{one}");
    for lens in cuttings(df.height()) {
        assert!(closed(&lens).equals_missing(&one), "{lens:?}");
    }
}
