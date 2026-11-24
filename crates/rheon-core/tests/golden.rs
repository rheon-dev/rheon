//! Golden-value regression tests: one fixed stream per model, with the exact
//! expected outputs embedded.
//!
//! Why these exist. `rheon-core` is meant to be exhaustively unit-tested
//! (CLAUDE.md), but a `cargo mutants` pass showed 517 of 1645 mutations
//! surviving — the arithmetic inside the recursions is largely pinned by the
//! *Python* oracle suite (`tests/reference.py`, agreement to ~1e-13), which
//! `cargo test` cannot see. A single golden run per model closes that gap
//! cheaply: any change to a coefficient, a decay factor, an accumulator update
//! or a solve moves these numbers.
//!
//! The constants are not arbitrary. They are the current implementation's
//! output, and that implementation is independently verified elsewhere: the
//! numpy references in `tests/reference.py` for `ewridge`, `rls`, `kalman`,
//! `huber`/`quantile` and `ftrl`; the lasso's KKT conditions in
//! `tests/test_oracles.py`; and, for `sgd`, `pa`, `holt` and `ew_cov`, the
//! recursion written out longhand in each module's own unit tests. This file
//! locks in numbers that have already been checked against something else.
//!
//! Every model has a signature here, and
//! `tests/test_model_registry.py::test_the_core_golden_file_pins_every_model`
//! fails when one is missing.
//!
//! Regenerate with `PRINT_GOLDEN=1 cargo test -p rheon-core --test golden --
//! --nocapture`, and only after confirming a change is intended.

use rheon_core::*;

#[path = "garch_series/mod.rs"]
mod garch_series;
use garch_series::Innovations;

/// Deterministic pseudo-random stream, no external rng dependency.
fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

/// One row of the fixed stream: features, targets, clock delta, row weight.
type Row = ([f64; 2], [Option<f64>; 1], f64, f64);

/// 60 rows: two features, one null target and one clock gap, so the null
/// policy and the decay both participate.
fn stream() -> Vec<Row> {
    let mut s = 20240830u64;
    (0..60)
        .map(|i| {
            let x = [lcg(&mut s), lcg(&mut s)];
            let y = 1.5 * x[0] - 0.75 * x[1] + 0.25 + 0.1 * lcg(&mut s);
            let target = if i == 31 { None } else { Some(y) };
            let d = if i == 0 {
                0.0
            } else if i == 40 {
                25.0
            } else {
                1.0
            };
            (x, [target], d, 0.5 + 0.5 * (i % 3) as f64)
        })
        .collect()
}

/// Predictions at rows 20, 45 and 59 for one output slot.
fn signature<M: OnlineModel>(model: &mut M, pick: usize) -> Vec<f64> {
    signature_of(model, pick, true)
}

/// The same three rows, with the features withheld for the model that
/// takes none (`holt`).
fn signature_of<M: OnlineModel>(model: &mut M, pick: usize, features: bool) -> Vec<f64> {
    let mut out = Vec::new();
    for (i, (x, y, d, w)) in stream().into_iter().enumerate() {
        let x: &[f64] = if features { &x } else { &[] };
        let step = model.step(x, &y, d, w);
        if matches!(i, 20 | 45 | 59) {
            out.push(step.pred[pick]);
        }
    }
    out
}

fn check(name: &str, got: &[f64], want: &[f64]) {
    // `1` and nothing else: any value, the empty string included, turned
    // every golden off with only stdout to say so (review 2026-10-06, CF9).
    if std::env::var("PRINT_GOLDEN").is_ok_and(|v| v == "1") {
        // `{v:?}` prints the shortest representation that round-trips, which
        // avoids clippy's excessive-precision lint on the embedded constants.
        let vals: Vec<String> = got
            .iter()
            .map(|v| {
                if v.is_nan() {
                    "f64::NAN".to_owned()
                } else {
                    format!("{v:?}")
                }
            })
            .collect();
        println!("GOLDEN {name}: &[{}];", vals.join(", "));
        // A printing run compared nothing: it fails, so it never stands for
        // a test run (review S20).
        panic!("PRINT_GOLDEN=1 printed {name} and compared nothing; run again without it");
    }
    assert_eq!(got.len(), want.len(), "{name}: wrong signature length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        // a number withheld is pinned as withheld
        assert!(
            (g.is_nan() && w.is_nan()) || (g - w).abs() <= 1e-12 * (1.0 + w.abs()),
            "{name}[{i}]: got {g:.17e}, want {w:.17e}"
        );
    }
}

/// [`check`] to the bit: for a model that calls no platform math library
/// (`garch` reads `ln`, `exp` and `ln_1p` from the `libm` crate), whose
/// numbers are therefore the same bits on every platform, so a platform
/// whose library crept back into it fails here on that platform's CI.
fn check_bits(name: &str, got: &[f64], want: &[f64]) {
    check(name, got, want);
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert_eq!(
            g.to_bits(),
            w.to_bits(),
            "{name}[{i}]: got {g:.17e}, want {w:.17e}, the same bits on every platform"
        );
    }
}

fn ewridge_cfg(standardize: bool, ridge: f64) -> EwRidgeCfg {
    EwRidgeCfg {
        n_features: 2,
        n_targets: 1,
        fit_intercept: true,
        decay: Decay::Halflife(20.0),
        ridge: vec![ridge],
        feature_sets: vec![],
        standardize,
        ridge_scale: false,
        session_shrink: None,
        long_half_life: None,
        coef_prior: None,
        min_weight: 3.0,
        solve_every: 0.0,
        max_rows_between_solves: 1,
        solve_share: None,
        gram_block_rows: 0,
        target_gaps: rheon_core::TargetGaps::OwnRows,
        window: None,
        window_every: None,
        max_rows_between_snapshots: None,
    }
}

fn robust_cfg(loss: RobustLoss, standardize: bool) -> RobustCfg {
    RobustCfg {
        n_features: 2,
        n_targets: 1,
        fit_intercept: true,
        decay: Decay::Halflife(20.0),
        loss,
        ridge: 1e-4,
        standardize,
        min_weight: 3.0,
        solve_every: 0.0,
        max_rows_between_solves: 1,
        solve_share: None,
        quantile_eps: 1e-3,
        huber_scale: rheon_core::HuberScale::Rms,
        huber_scale_decay: None,
        feature_sets: Vec::new(),
        ridge_scale: false,
        emit_scale: false,
        refit_rows: 0,
        mad_anchor: false,
        min_weight_holds_learning: false,
        refit_clock: None,
        max_state_bytes: None,
    }
}

#[test]
fn ewridge_golden() {
    let mut m = EwRidge::new(ewridge_cfg(false, 1e-4)).unwrap();
    check("ewridge", &signature(&mut m, 0), GOLDEN_EW_RIDGE);
}

#[test]
fn ewridge_standardized_golden() {
    let mut m = EwRidge::new(ewridge_cfg(true, 0.01)).unwrap();
    check("ewridge_std", &signature(&mut m, 0), GOLDEN_EW_RIDGE_STD);
}

#[test]
fn ewridge_windowed_golden() {
    // A hard window with `window_every = 3`: the fit reads only the rows
    // inside the window, and a snapshot is kept every third row. This pins it
    // so P1's move of the snapshot build into the `offer` closure -- built
    // only when it is kept, not on every row -- cannot move a number (review
    // 2026-09-18, P1).
    let mut c = ewridge_cfg(false, 1e-4);
    c.window = Some(30.0);
    c.max_rows_between_snapshots = Some(3);
    let mut m = EwRidge::new(c).unwrap();
    // Under `closed = "both"`, as `lasso_windowed_golden` says why (this
    // stream's cadence puts no snapshot exactly one window old, so the two
    // edges agree here).
    m.set_window_closed(WindowClosed::Both);
    check(
        "ewridge_windowed",
        &signature(&mut m, 0),
        GOLDEN_EW_RIDGE_WINDOWED,
    );
}

/// The through-origin fits move no number above (every signature there has
/// an intercept), so the branches that only they take -- `ewridge`'s
/// standardized solve through the origin, with the warm prior it dropped
/// until the review of 2026-09-18 (B2, T1), `robust`'s raw-scaled one, and
/// `kalman` without the standardizer -- are pinned here.
#[test]
fn ewridge_origin_standardized_golden() {
    let mut c = ewridge_cfg(true, 0.05);
    c.fit_intercept = false;
    c.coef_prior = Some(vec![vec![1.0, -0.5]]);
    let mut m = EwRidge::new(c).unwrap();
    check(
        "ewridge_origin_std",
        &signature(&mut m, 0),
        GOLDEN_EW_RIDGE_ORIGIN_STD,
    );
}

#[test]
fn huber_origin_standardized_golden() {
    let mut c = robust_cfg(RobustLoss::Huber { delta: 1.5 }, true);
    c.fit_intercept = false;
    let mut m = Robust::new(c).unwrap();
    check(
        "huber_origin_std",
        &signature(&mut m, 0),
        GOLDEN_HUBER_ORIGIN_STD,
    );
}

#[test]
fn kalman_plain_golden() {
    let mut m = Kalman::new(KalmanCfg {
        horizons: Vec::new(),
        // The form a state saved before schema 64 goes on in: `P` absolute
        // and `min_weight` holding learning back, as `tests/reference.py`'s
        // `kalman_ref` has it (task 278, D10).
        min_weight_holds_learning: true,
        noise_units: false,
        n_features: 2,
        n_targets: 1,
        fit_intercept: true,
        decay: Decay::Halflife(50.0),
        half_life: vec![f64::INFINITY, 30.0, 100.0],
        q: None,
        obs_var: None,
        p0: 1.0,
        share_p: false,
        min_weight: 3.0,
        revert_half_life: vec![f64::INFINITY],
        standardize: false,
    })
    .unwrap();
    check("kalman_plain", &signature(&mut m, 0), GOLDEN_KALMAN_PLAIN);
}

#[test]
fn rls_golden() {
    let mut m = Rls::new(RlsCfg {
        n_features: 2,
        n_targets: 1,
        fit_intercept: true,
        decay: Decay::Halflife(20.0),
        delta: 0.5,
        coef_prior: None,
        min_weight: 3.0,
    })
    .unwrap();
    check("rls", &signature(&mut m, 0), GOLDEN_RLS);
}

#[test]
fn kalman_golden() {
    let mut m = Kalman::new(KalmanCfg {
        horizons: Vec::new(),
        // The form a state saved before schema 64 goes on in: `P` absolute
        // and `min_weight` holding learning back, as `tests/reference.py`'s
        // `kalman_ref` has it (task 278, D10).
        min_weight_holds_learning: true,
        noise_units: false,
        n_features: 2,
        n_targets: 1,
        fit_intercept: true,
        decay: Decay::Halflife(50.0),
        half_life: vec![f64::INFINITY, 30.0, 100.0],
        q: None,
        obs_var: None,
        p0: 1.0,
        share_p: false,
        min_weight: 3.0,
        revert_half_life: vec![f64::INFINITY],
        standardize: true,
    })
    .unwrap();
    check("kalman", &signature(&mut m, 0), GOLDEN_KALMAN);
}

/// E41: the intercept a random walk, both slopes reverting, one of them
/// fast, on the same stream as `kalman_golden`.
#[test]
fn kalman_revert_golden() {
    let mut m = Kalman::new(KalmanCfg {
        horizons: Vec::new(),
        // The form a state saved before schema 64 goes on in: `P` absolute
        // and `min_weight` holding learning back, as `tests/reference.py`'s
        // `kalman_ref` has it (task 278, D10).
        min_weight_holds_learning: true,
        noise_units: false,
        n_features: 2,
        n_targets: 1,
        fit_intercept: true,
        decay: Decay::Halflife(50.0),
        half_life: vec![f64::INFINITY, 30.0, 100.0],
        q: None,
        obs_var: None,
        p0: 1.0,
        share_p: false,
        min_weight: 3.0,
        revert_half_life: vec![f64::INFINITY, 40.0, 8.0],
        standardize: true,
    })
    .unwrap();
    check("kalman_revert", &signature(&mut m, 0), GOLDEN_KALMAN_REVERT);
}

#[test]
fn lasso_golden() {
    let mut m = Lasso::new(LassoCfg {
        min_weight_holds_learning: false,
        n_features: 2,
        n_targets: 1,
        fit_intercept: true,
        decay: Decay::Halflife(20.0),
        lasso_path: vec![0.2, 0.02, 0.0],
        l1_ratio: 1.0,
        select_half_life: None,
        min_weight: 3.0,
        target_min_weight: Vec::new(),
        solve_every: 0.0,
        max_rows_between_solves: 1,
        solve_share: None,
        window: None,
        window_every: None,
        max_rows_between_snapshots: None,
        max_iter: 200,
        tol: 1e-12,
        target_gaps: rheon_core::TargetGaps::OwnRows,
    })
    .unwrap();
    check("lasso", &signature(&mut m, 1), GOLDEN_LASSO);
}

#[test]
fn huber_golden() {
    let mut m = Robust::new(robust_cfg(RobustLoss::Huber { delta: 1.5 }, false)).unwrap();
    check("huber", &signature(&mut m, 0), GOLDEN_HUBER);
}

#[test]
fn quantile_golden() {
    let mut m = Robust::new(robust_cfg(RobustLoss::Quantile { tau: 0.7 }, true)).unwrap();
    check("quantile", &signature(&mut m, 0), GOLDEN_QUANTILE);
}

#[test]
fn ftrl_golden() {
    let mut m = Ftrl::new(FtrlCfg {
        n_features: 2,
        n_targets: 1,
        fit_intercept: true,
        decay: Decay::Halflife(40.0),
        alpha: 0.1,
        beta: 1.0,
        l1: 0.05,
        l2: 1.0,
        min_weight: 3.0,
        strict_binary: false,
        loss: FtrlLoss::Logistic,
        standardize: false,
    })
    .unwrap();
    check("ftrl", &signature(&mut m, 0), GOLDEN_FTRL);
}

#[test]
fn ftrl_squared_golden() {
    let mut m = Ftrl::new(FtrlCfg {
        n_features: 2,
        n_targets: 1,
        fit_intercept: true,
        decay: Decay::Halflife(40.0),
        alpha: 0.5,
        beta: 1.0,
        l1: 0.05,
        l2: 0.01,
        min_weight: 3.0,
        strict_binary: false,
        loss: FtrlLoss::Squared,
        standardize: false,
    })
    .unwrap();
    check("ftrl_squared", &signature(&mut m, 0), GOLDEN_FTRL_SQUARED);
}

/// The busier path: a robust loss, an annealed rate, a penalty and the
/// running standardization all take part, so each has a number to move.
#[test]
fn sgd_golden() {
    let mut m = Sgd::new(SgdCfg {
        n_features: 2,
        n_targets: 1,
        fit_intercept: true,
        decay: Decay::Halflife(20.0),
        loss: SgdLoss::Huber { delta: 0.5 },
        learning_rate: 0.05,
        schedule: LearningRate::InvScaling { power: 0.25 },
        l2: 0.01,
        clip_gradient: 1e3,
        constraint: None,
        standardize: true,
        strict_binary: false,
        min_weight: 3.0,
        huber_scale: rheon_core::HuberScale::Rms,
        huber_scale_decay: None,
        refit_rows: 0,
        min_weight_holds_learning: false,
        target_spread: false,
        mad_anchor: false,
        refit_clock: None,
        max_state_bytes: None,
    })
    .unwrap();
    check("sgd", &signature(&mut m, 0), GOLDEN_SGD);
}

/// The plain path: squared loss, constant rate, no penalty, raw features.
#[test]
fn sgd_squared_golden() {
    let mut m = Sgd::new(SgdCfg {
        n_features: 2,
        n_targets: 1,
        fit_intercept: true,
        decay: Decay::Halflife(20.0),
        loss: SgdLoss::Squared,
        learning_rate: 0.05,
        schedule: LearningRate::Constant,
        l2: 0.0,
        clip_gradient: 1e3,
        constraint: None,
        standardize: false,
        strict_binary: false,
        min_weight: 3.0,
        huber_scale: rheon_core::HuberScale::Rms,
        huber_scale_decay: None,
        refit_rows: 0,
        min_weight_holds_learning: false,
        target_spread: false,
        mad_anchor: false,
        refit_clock: None,
        max_state_bytes: None,
    })
    .unwrap();
    check("sgd_squared", &signature(&mut m, 0), GOLDEN_SGD_SQUARED);
}

#[test]
fn pa_golden() {
    let mut m = Pa::new(PaCfg {
        n_features: 2,
        n_targets: 1,
        fit_intercept: true,
        decay: Decay::Halflife(20.0),
        mode: PaMode::Pa2,
        c: 0.5,
        eps: 0.05,
        min_weight: 3.0,
        constraint: None,
        standardize: false,
    })
    .unwrap();
    check("pa", &signature(&mut m, 0), GOLDEN_PA);
}

/// The positive target `glm` reads: the stream's `y` squared plus 0.1, so
/// a Gamma or Tweedie amount, with the same null row.
fn glm_signature(m: &mut Glm) -> Vec<f64> {
    let mut out = Vec::new();
    for (i, (x, y, d, w)) in stream().into_iter().enumerate() {
        let y = [y[0].map(|v| v * v + 0.1)];
        let step = m.step(&x, &y, d, w);
        if matches!(i, 20 | 45 | 59) {
            out.push(step.pred[0]);
        }
    }
    out
}

/// A Gamma `glm` under decay whose first 15 learned rows are refit, so the
/// signature's rows read the refit's fit and the recursion after it.
#[test]
fn glm_golden() {
    let mut m = Glm::new(GlmCfg {
        n_features: 2,
        n_targets: 1,
        fit_intercept: true,
        decay: Decay::Halflife(20.0),
        family: GlmFamily::Gamma,
        link: GlmLink::Log,
        var_power: 1.5,
        ridge: 1.0,
        min_weight: 3.0,
        refit_rows: 15,
        refit_clock: None,
        step_cap: f64::INFINITY,
        max_state_bytes: None,
    })
    .unwrap();
    check("glm", &glm_signature(&mut m), GOLDEN_GLM);
}

/// A Tweedie `glm` at `p = 1.5` with no refit, every row held to the step
/// cap.
#[test]
fn glm_cap_golden() {
    let mut m = Glm::new(GlmCfg {
        n_features: 2,
        n_targets: 1,
        fit_intercept: true,
        decay: Decay::Halflife(20.0),
        family: GlmFamily::Tweedie,
        link: GlmLink::Log,
        var_power: 1.5,
        ridge: 1.0,
        min_weight: 3.0,
        refit_rows: 0,
        refit_clock: None,
        step_cap: 0.5,
        max_state_bytes: None,
    })
    .unwrap();
    check("glm_cap", &glm_signature(&mut m), GOLDEN_GLM_CAP);
}

/// The constrained path (ENHANCEMENTS E40): slopes on the simplex, so the
/// projection with a sum and a wall both take part; the truth (1.5, -0.75)
/// lies outside the set.
#[test]
fn sgd_simplex_golden() {
    let mut m = Sgd::new(SgdCfg {
        n_features: 2,
        n_targets: 1,
        fit_intercept: true,
        decay: Decay::Halflife(20.0),
        loss: SgdLoss::Squared,
        learning_rate: 0.05,
        schedule: LearningRate::Constant,
        l2: 0.0,
        clip_gradient: 1e3,
        constraint: Some(Constraint {
            lo: vec![0.0, 0.0],
            hi: vec![f64::INFINITY, f64::INFINITY],
            sum: Some(1.0),
        }),
        standardize: false,
        strict_binary: false,
        min_weight: 3.0,
        huber_scale: rheon_core::HuberScale::Rms,
        huber_scale_decay: None,
        refit_rows: 0,
        min_weight_holds_learning: false,
        target_spread: false,
        mad_anchor: false,
        refit_clock: None,
        max_state_bytes: None,
    })
    .unwrap();
    check("sgd_simplex", &signature(&mut m, 0), GOLDEN_SGD_SIMPLEX);
}

/// A box with one wall the truth crosses (`x1`'s slope capped at 0) and a
/// sum below what the free fit would give.
#[test]
fn pa_box_golden() {
    let mut m = Pa::new(PaCfg {
        n_features: 2,
        n_targets: 1,
        fit_intercept: true,
        decay: Decay::Halflife(20.0),
        mode: PaMode::Pa1,
        c: 0.5,
        eps: 0.05,
        min_weight: 3.0,
        constraint: Some(Constraint {
            lo: vec![-1.0, -1.0],
            hi: vec![1.0, 0.0],
            sum: Some(0.5),
        }),
        standardize: false,
    })
    .unwrap();
    check("pa_box", &signature(&mut m, 0), GOLDEN_PA_BOX);
}

#[test]
fn holt_golden() {
    let mut m = Holt::new(HoltCfg {
        horizons: Vec::new(),
        n_targets: 1,
        level_half_life: 10.0,
        trend_half_life: 40.0,
        min_weight: 3.0,
        trend: true,
        ..Default::default()
    })
    .unwrap();
    check("holt", &signature_of(&mut m, 0, false), GOLDEN_HOLT);
}

#[test]
fn seqtest_golden() {
    // Slot 0 is the log e-value for "the target is positive", slot 1 the one
    // for "negative": together they pin the stake, the wealth update and the
    // counts (a wrong count moves the next stake).
    let mut m = SeqTest::new(SeqTestCfg {
        n_targets: 1,
        min_weight: 0.0,
    })
    .unwrap();
    check("seqtest", &signature_of(&mut m, 0, false), GOLDEN_SEQTEST);
    let mut m = SeqTest::new(SeqTestCfg {
        n_targets: 1,
        min_weight: 0.0,
    })
    .unwrap();
    check(
        "seqtest_neg",
        &signature_of(&mut m, 1, false),
        GOLDEN_SEQTEST_NEG,
    );
}

/// Slot 0 is `p_change`, which reads the whole run-length posterior and
/// every run's predictive at once.
#[test]
fn bocpd_golden() {
    let mut m = Bocpd::new(BocpdCfg {
        n_features: 2,
        hazard: 30.0,
        hazard_from_row: false,
        emission: BocpdEmission::Diag,
        prior_mean: Some(vec![0.0, 0.0]),
        prior_kappa: 1.0,
        prior_nu: Some(2.0),
        prior_scale: Some(vec![1.0]),
        robust_beta: 0.0,
        prune_below: 1e-8,
        max_run: 100,
        min_weight: 0.0,
        warm_rows: None,
        hazard_on_clock: false,
        warm_clock: None,
    })
    .unwrap();
    check("bocpd", &signature(&mut m, 0), GOLDEN_BOCPD);
}

/// `bocpd_golden` with the hazard on the clock, `τ = 30` clock units
/// between changepoints (docs/PLAN.md task 179): the stream's steps of 1
/// carry a chance of 0.033 each and its gap of 25 one of 0.565, applied
/// before the row each leads into. Checked against the definition by the
/// enumeration in `tests/test_bocpd.py`, and against the per-row hazard on
/// regular steps by `bocpd.rs`'s tests.
#[test]
fn bocpd_on_the_clock_golden() {
    let mut m = Bocpd::new(BocpdCfg {
        n_features: 2,
        hazard: 30.0,
        hazard_from_row: false,
        emission: BocpdEmission::Diag,
        prior_mean: Some(vec![0.0, 0.0]),
        prior_kappa: 1.0,
        prior_nu: Some(2.0),
        prior_scale: Some(vec![1.0]),
        robust_beta: 0.0,
        prune_below: 1e-8,
        max_run: 100,
        min_weight: 0.0,
        warm_rows: None,
        hazard_on_clock: true,
        warm_clock: None,
    })
    .unwrap();
    check(
        "bocpd_on_the_clock",
        &signature(&mut m, 0),
        GOLDEN_BOCPD_ON_THE_CLOCK,
    );
}

/// `corrchange` reports only where a span closes, so its signature is the
/// statistic at the three span ends a 60-row stream at `horizon = 20` has.
#[test]
fn corrchange_golden() {
    let mut m = CorrChange::new(CorrChangeCfg {
        n_features: 2,
        kind: CorrChangeKind::Monitor,
        span_rows: 20,
        alpha: 0.05,
        alpha_adjust: "bonferroni".into(),
        bandwidth: None,
        scalar: false,
        decay: Decay::Halflife(20.0),
        crit: None,
        n_perm: 20,
        permute_every_rows: 10,
        perm_block: 1,
        norm: ChangeNorm::L1,
        seed: 5,
        reset: false,
        monitor_rows: 0,
        boundary_gamma: 0.0,
        permute_every: None,
        span_clock: None,
        monitor_clock: None,
        max_state_bytes: None,
    })
    .unwrap();
    let mut out = Vec::new();
    for (x, y, d, w) in stream() {
        let step = m.step(&x, &y, d, w);
        if step.pred[0].is_finite() {
            out.push(step.pred[0]);
        }
    }
    check("corrchange", &out, GOLDEN_CORRCHANGE);
}

/// A correlated pair whose correlation moves from 0.3 to 0.85 at row 350 of
/// 600, each column an Irwin-Hall draw (twelve uniforms less 6), so the
/// stream is the same bits on every platform: the realistic case (PLAN 550)
/// beside `stream()`'s independent columns.
fn correlated_pair() -> Vec<[f64; 2]> {
    let mut s = 553u64;
    let mut normal = || (0..12).map(|_| lcg(&mut s) / 2.0 + 0.5).sum::<f64>() - 6.0;
    (0..600)
        .map(|i| {
            let rho: f64 = if i < 350 { 0.3 } else { 0.85 };
            let (a, b) = (normal(), normal());
            [a, rho * a + (1.0 - rho * rho).sqrt() * b]
        })
        .collect()
}

/// `corrchange`'s monitor and sequential detector on [`correlated_pair`]:
/// every report's statistic, critical value and flag, and a flag's date
/// (PLAN 553). The monitor's spans of 100 rows flag the one the change
/// falls in; the sequential detector's history of 100 rows flags inside the
/// period the change falls in.
#[test]
fn corrchange_realistic_golden() {
    let base = |kind, span_rows, monitor_rows| CorrChangeCfg {
        n_features: 2,
        kind,
        span_rows,
        alpha: 0.05,
        alpha_adjust: "bonferroni".into(),
        bandwidth: None,
        scalar: false,
        decay: Decay::Halflife(f64::INFINITY),
        crit: None,
        n_perm: 20,
        permute_every_rows: 10,
        perm_block: 1,
        norm: ChangeNorm::L1,
        seed: 5,
        reset: false,
        monitor_rows,
        boundary_gamma: 0.25,
        permute_every: None,
        span_clock: None,
        monitor_clock: None,
        max_state_bytes: None,
    };
    for (name, cfg, want) in [
        (
            "corrchange_monitor_realistic",
            base(CorrChangeKind::Monitor, 100, 0),
            GOLDEN_CORRCHANGE_MONITOR_REALISTIC,
        ),
        (
            "corrchange_sequential_realistic",
            base(CorrChangeKind::Sequential, 100, 150),
            GOLDEN_CORRCHANGE_SEQUENTIAL_REALISTIC,
        ),
    ] {
        let mut m = CorrChange::new(cfg).unwrap();
        let mut out = Vec::new();
        let mut flags = 0;
        for x in correlated_pair() {
            let step = m.step(&x, &[], 1.0, 1.0);
            if step.pred[0].is_finite() {
                out.extend([step.pred[0], step.pred[1], step.pred[2]]);
            }
            if step.pred[2] == 1.0 {
                out.push(step.pred[4]);
                flags += 1;
            }
        }
        assert!(flags >= 1, "{name}: the change is found");
        check(name, &out, want);
    }
}

/// Slot 5 is `loglik`, which reads the transition matrix, both densities
/// and the softmax at once -- the whole filter in one number.
#[test]
fn hmm_golden() {
    let mut m = Hmm::new(HmmCfg {
        n_features: 2,
        k: 2,
        decay: Decay::Halflife(20.0),
        covariance: Covariance::Full,
        precision_prior: 1e-2,
        min_weight: 3.0,
        learn: true,
        transition_prior: 1.0,
        transition: Some(vec![0.9, 0.1, 0.2, 0.8]),
        means: Some(vec![-0.5, -0.5, 0.5, 0.5]),
        covs: Some(vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]),
        warm_rows: 10,
        seed_rule: SeedRule::First,
        seed: 1,
        tvtp: None,
        prior_rows: None,
        warm_clock: None,
    })
    .unwrap();
    check("hmm", &signature(&mut m, 5), GOLDEN_HMM);
}

/// `rcov` reports nothing per row, so its signature is the block: the
/// kernel estimate's three distinct entries after the whole stream.
#[test]
fn rcov_golden() {
    let mut m = Rcov::new(RcovCfg {
        n_features: 2,
        kind: RcovKind::Kernel,
        kernel: "parzen".into(),
        bandwidth: Some(3),
        jitter: 2,
        theta: 1.0,
        psd: false,
        block_rows: Some(60),
        max_bandwidth: None,
        preavg_rows: None,
        noise_stride: 1,
        iv_stride: 20,
    })
    .unwrap();
    for (x, y, d, w) in stream() {
        m.step(&x, &y, d, w);
    }
    let cov = m.estimate().rcov.expect("a block");
    check("rcov", &[cov[0], cov[1], cov[3]], GOLDEN_RCOV);
}

/// `audit` on the same stream, its first feature null on one row and
/// infinite on another, past a cap of 16 counters: every statistic of the
/// first column that is not a count, the pair's correlation and the clock's
/// spread of regular steps (docs/PLAN.md task 223 (b)).
#[test]
fn audit_golden() {
    let mut m = Audit::new(AuditCfg {
        n_columns: 2,
        pairs: true,
        distinct_cap: 16,
        // Past the stream's one long step, so it is a regular step.
        gap_cap: Some(30.0),
        has_clock: true,
    })
    .unwrap();
    for (i, (mut x, y, d, w)) in stream().into_iter().enumerate() {
        match i {
            5 => x[0] = NULL,
            6 => x[0] = f64::INFINITY,
            // A run of three, for the frozen counts.
            11..=13 => x[0] = 0.25,
            _ => {}
        }
        m.step(&x, &y, d, w);
    }
    let c = m.column(0);
    let k = m.clock().unwrap();
    let got = [
        c.mean,
        c.std,
        c.skew,
        c.kurtosis,
        c.median,
        c.mad,
        c.robust_z,
        c.autocorr,
        c.unit_root_t,
        c.equal_by_chance,
        m.pair(0, 1).unwrap().corr,
        k.step_cv,
        (c.null + 10 * c.pos_inf + 100 * c.longest_run + 1000 * c.count_error) as f64,
    ];
    check("audit", &got, GOLDEN_AUDIT);
}

/// `hawkes` on the same stream: its one target an event of that mark on
/// every row where it is above 0, two kernels, decayed and in a window of
/// eight rows. The three rows' intensities, then the fit at the end: the
/// recursion, the read of `G` from `b`, the window's removal and its
/// rebuild all move these.
#[test]
fn hawkes_golden() {
    let cfg = HawkesCfg {
        n_targets: 1,
        kernel_decays: vec![0.8, 3.0],
        decay: Decay::Halflife(15.0),
        min_weight: 0.0,
        window_events: None,
        window_clock: None,
        closed: WindowClosed::Right,
        max_state_bytes: None,
    };
    let mut m = Hawkes::new(cfg.clone()).unwrap();
    let mut got = signature_of(&mut m, 0, false);
    got.extend(&m.coefficients()[0]);
    check("hawkes", &got, GOLDEN_HAWKES);
    let mut m = Hawkes::new(HawkesCfg {
        decay: Decay::Halflife(f64::INFINITY),
        window_events: Some(8),
        ..cfg
    })
    .unwrap();
    let mut got = signature_of(&mut m, 0, false);
    got.extend(&m.coefficients()[0]);
    check("hawkes_window", &got, GOLDEN_HAWKES_WINDOW);
}

/// `realized` on the same stream, its one target read as a return series:
/// every output (`rv`, `bv`, `tv`, `tq`, `bns`, `lm`, `lm_jump`) at the
/// three rows, under a decay with Lee and Mykland's own window of 10, then
/// in a window of 12 rows; the null target and the gap move both. The window
/// calls no platform math library (its cube roots and logarithms are the
/// `libm` crate's, and nothing decays), so it is pinned to the bit.
#[test]
fn realized_golden() {
    let all = |m: &mut Realized| {
        let mut out = Vec::new();
        for (i, (_, y, d, w)) in stream().into_iter().enumerate() {
            let step = m.step(&[], &y, d, w);
            if matches!(i, 20 | 45 | 59) {
                out.extend(&step.pred);
            }
        }
        out
    };
    let cfg = RealizedCfg {
        n_targets: 1,
        decay: Decay::Halflife(15.0),
        min_weight: 0.0,
        window_rows: None,
        window_clock: None,
        closed: WindowClosed::Right,
        lm_window: Some(10),
        lm_beta: 0.99,
        max_state_bytes: None,
    };
    let got = all(&mut Realized::new(cfg.clone()).unwrap());
    check("realized", &got, GOLDEN_REALIZED);
    let got = all(&mut Realized::new(RealizedCfg {
        decay: Decay::Halflife(f64::INFINITY),
        window_rows: Some(12),
        lm_window: None,
        ..cfg
    })
    .unwrap());
    check_bits("realized_window", &got, GOLDEN_REALIZED_WINDOW);
}

/// `power_variation` on the same stream: a term of each form (a sum of
/// squares, normalized bipower and tripower quarticity, unequal powers, a
/// centered Parzen sum, a flat-top Bartlett sum, a Tukey-Hanning sum and the
/// squares of three-step returns) and `n`, at the three rows, under a
/// decay; the null target and the gap move them.
#[test]
fn power_variation_golden() {
    let got = power_variation_rows(Decay::Halflife(15.0), None);
    check("power_variation", &got, GOLDEN_POWER_VARIATION);
}

/// The same in a window of 12 rows. Nothing decays and its powers, cosines
/// and gammas are the `libm` crate's, so it is pinned to the bit.
#[test]
fn power_variation_window_golden() {
    let got = power_variation_rows(Decay::Halflife(f64::INFINITY), Some(12));
    check_bits(
        "power_variation_window",
        &got,
        GOLDEN_POWER_VARIATION_WINDOW,
    );
}

fn power_variation_rows(decay: Decay, window_rows: Option<usize>) -> Vec<f64> {
    let mp = |powers: &[f64], normalize: bool, scale: usize| PvTerm {
        kind: PvTermKind::Multipower {
            powers: powers.to_vec(),
            normalize,
        },
        scale,
    };
    let ker = |kernel, bandwidth, flat_top, center| PvTerm {
        kind: PvTermKind::Kernel {
            kernel,
            bandwidth,
            flat_top,
            center,
        },
        scale: 1,
    };
    let q = 4.0 / 3.0;
    let mut m = PowerVariation::new(PowerVariationCfg {
        n_targets: 1,
        terms: vec![
            mp(&[2.0], false, 1),
            mp(&[1.0, 1.0], true, 1),
            mp(&[q, q, q], true, 1),
            mp(&[1.0, 0.5], false, 1),
            ker(PvKernel::Parzen, 4, false, true),
            ker(PvKernel::Bartlett, 3, true, false),
            ker(PvKernel::TukeyHanning, 2, false, false),
            mp(&[2.0], false, 3),
        ],
        decay,
        min_weight: 0.0,
        window_rows,
        window_clock: None,
        closed: WindowClosed::Right,
        max_state_bytes: None,
    })
    .unwrap();
    let mut out = Vec::new();
    for (i, (_, y, d, w)) in stream().into_iter().enumerate() {
        let step = m.step(&[], &y, d, w);
        if matches!(i, 20 | 45 | 59) {
            out.extend(&step.pred);
        }
    }
    out
}

/// `ar` on the same stream: an AR(2) under decay with the two features as
/// exogenous regressors, then its coefficients at the end. The ring (its
/// null target and its gap), the design read from it and the fit all move
/// these.
#[test]
fn ar_golden() {
    let mut m = Ar::new(ArCfg {
        n_targets: 1,
        n_features: 2,
        lags: vec![1, 2],
        fit_intercept: true,
        decay: Decay::Halflife(15.0),
        ridge: 1e-6,
        min_weight: 2.0,
        max_state_bytes: None,
        select: None,
        window: None,
        window_every: None,
        max_rows_between_snapshots: None,
    })
    .unwrap();
    let mut got = signature(&mut m, 0);
    got.extend(&m.coefficients().unwrap()[0]);
    check("ar", &got, GOLDEN_AR);
}

/// `spectral` on the same stream's target: its power at three frequencies
/// under decay, its null target and its gap among the rows.
#[test]
fn spectral_golden() {
    let mut m = Spectral::new(SpectralCfg {
        n_targets: 1,
        frequencies: vec![0.0, 0.1, 0.5],
        spacing: 1.0,
        decay: Decay::Halflife(15.0),
        scaling: SpectralScaling::Density,
        detrend: true,
        min_weight: 0.0,
        max_state_bytes: None,
        window: None,
        window_every: None,
        max_rows_between_snapshots: None,
    })
    .unwrap();
    let mut got = signature_of(&mut m, 1, false);
    got.extend(&m.powers()[0]);
    // Its trigonometry is the `libm` crate's: the same bits everywhere.
    check_bits("spectral", &got, GOLDEN_SPECTRAL);
}

/// `expectile` on the same stream: an upper expectile under decay whose
/// first 15 learned rows are refit, then the fit at the end.
#[test]
fn expectile_golden() {
    let cfg = ExpectileCfg {
        n_features: 2,
        n_targets: 1,
        fit_intercept: true,
        decay: Decay::Halflife(20.0),
        expectile: 0.85,
        ridge: 1e-6,
        min_weight: 3.0,
        refit_rows: 15,
        refit_clock: None,
        window_rows: None,
        window_clock: None,
        closed: WindowClosed::Right,
        max_state_bytes: None,
    };
    let mut m = Expectile::new(cfg).unwrap();
    let mut got = signature(&mut m, 0);
    got.extend(&m.coefficients()[0]);
    check("expectile", &got, GOLDEN_EXPECTILE);
}

/// A window of twelve rows at a lower level, refit exactly on every row.
#[test]
fn expectile_window_golden() {
    let mut m = Expectile::new(ExpectileCfg {
        n_features: 2,
        n_targets: 1,
        fit_intercept: true,
        min_weight: 3.0,
        window_clock: None,
        closed: WindowClosed::Right,
        max_state_bytes: None,
        decay: Decay::Halflife(f64::INFINITY),
        expectile: 0.2,
        ridge: 1e-6,
        refit_rows: 0,
        refit_clock: None,
        window_rows: Some(12),
    })
    .unwrap();
    let mut got = signature(&mut m, 0);
    got.extend(&m.coefficients()[0]);
    check("expectile_window", &got, GOLDEN_EXPECTILE_WINDOW);
}

/// `garch` on the stream run over and over (its start buffers 50 rows,
/// FIGARCH's 2,000, so the 60 alone would leave it a few): the variance,
/// the standardized residual and the forecast on three rows after the
/// start, then the fit at the end. Each with a cap too small to double its
/// start. A degenerate case, kept as an edge test (PLAN 550): the stream is
/// a linear target with no conditional variance, so the start's
/// likelihood-ratio statistic against constant variance is under its cut,
/// every start is withheld and buffered afresh, and every number is
/// withheld (PLAN 548). The realistic cases below pin the start's
/// optimizer, the step and its projection.
fn garch_signature(vol: GarchVol, dist: GarchDist, targeting: bool) -> Vec<f64> {
    let fig = vol == GarchVol::Figarch;
    let mut m = Garch::new(GarchCfg {
        n_inputs: 0,
        n_targets: 1,
        vol,
        dist,
        truncation: fig.then_some(20),
        variance_targeting: targeting,
        start_rows: if fig { 2000 } else { 50 },
        start_span: None,
        horizons: if vol == GarchVol::Egarch {
            Vec::new()
        } else {
            vec![4]
        },
        max_horizon_work: None,
        min_weight: 0.0,
        max_state_bytes: Some(if fig { 40_000 } else { 1700 }),
    })
    .unwrap();
    let rows = stream();
    let n = if fig { 2700 } else { 180 };
    let mut got = Vec::new();
    for i in 0..n {
        let (_, y, d, w) = rows[i % 60];
        let step = m.step(&[], &y, d, w);
        if [n - 110, n - 50, n - 1].contains(&i) {
            got.extend(&step.pred);
        }
    }
    got.extend(&m.coefficients()[0]);
    got
}

/// `tdigest` on the stream run five times at the smallest compression,
/// whose buffer merges every 80 rows: the three levels and their tail means
/// on three rows, each past a merge, one with a full buffer's merge just
/// made. The merge's limit, its centroid means, the read's interpolation and
/// the tail's partial mass move every one;
/// no libm call is in it, so it is held to the bit.
#[test]
fn tdigest_golden() {
    let mut m = TDigestModel::new(TDigestModelCfg {
        n_targets: 1,
        compression: 10.0,
        levels: vec![0.05, 0.25, 0.95],
        max_state_bytes: None,
    })
    .unwrap();
    let rows = stream();
    let mut got = Vec::new();
    for i in 0..300 {
        let (_, y, d, w) = rows[i % 60];
        let step = m.step(&[], &y, d, w);
        if [150, 170, 299].contains(&i) {
            got.extend(&step.pred);
        }
    }
    check_bits("tdigest", &got, GOLDEN_TDIGEST);
}

/// `ddsketch` on the stream run five times: plain, under a row window of
/// 40 and under a clock window of 80 units, the three levels and their tail
/// means on three rows each. The key's logarithm, the representative's
/// power and the partial-mass walk move every one; no libm call is in it,
/// so it is held to the bit.
#[test]
fn ddsketch_golden() {
    let mut got = Vec::new();
    for (rows, clock) in [(None, None), (Some(40), None), (None, Some(80.0))] {
        let mut m = DdSketchModel::new(DdSketchModelCfg {
            n_targets: 1,
            alpha: 0.01,
            levels: vec![0.05, 0.25, 0.95],
            window_rows: rows,
            window_clock: clock,
            closed: WindowClosed::Right,
            decay: Decay::Halflife(f64::INFINITY),
            max_state_bytes: None,
        })
        .unwrap();
        let stream = stream();
        for i in 0..300 {
            let (_, y, d, w) = stream[i % 60];
            let step = m.step(&[], &y, d, w);
            if [150, 170, 299].contains(&i) {
                got.extend(&step.pred);
            }
        }
    }
    check_bits("ddsketch", &got, GOLDEN_DDSKETCH);
}

/// `isotonic` of the stream's target on its first feature, over the 60
/// rows five times (so every `x` is tied five times by the end), pooled,
/// under a row window and under a clock window, increasing and
/// decreasing: no decay, so no `libm`, and the same bits everywhere.
#[test]
fn isotonic_golden() {
    let mut got = Vec::new();
    for (rows, clock, increasing) in [
        (None, None, true),
        (Some(40), None, false),
        (None, Some(80.0), true),
    ] {
        let mut m = Isotonic::new(IsotonicCfg {
            n_targets: 1,
            increasing,
            y_min: None,
            y_max: Some(2.0),
            out_of_bounds: IsotonicOutOfBounds::Clip,
            min_weight: 0.0,
            window_rows: rows,
            window_clock: clock,
            closed: WindowClosed::Right,
            decay: Decay::Halflife(f64::INFINITY),
            max_state_bytes: None,
        })
        .unwrap();
        let stream = stream();
        for i in 0..300 {
            let (x, y, d, w) = &stream[i % 60];
            let step = m.step(&x[..1], y, *d, *w);
            if [20, 150, 170, 299].contains(&i) {
                got.extend(&step.pred);
            }
        }
    }
    check_bits("isotonic", &got, GOLDEN_ISOTONIC);
}

/// `isotonic` under a decay of 20 rows: the weights are the platform's
/// `exp2`, so it is held to the golden files' tolerance, not to the bit.
#[test]
fn isotonic_decayed_golden() {
    let mut m = Isotonic::new(IsotonicCfg {
        n_targets: 1,
        increasing: true,
        y_min: None,
        y_max: None,
        out_of_bounds: IsotonicOutOfBounds::Nan,
        min_weight: 0.0,
        window_rows: None,
        window_clock: None,
        closed: WindowClosed::Right,
        decay: Decay::Halflife(20.0),
        max_state_bytes: None,
    })
    .unwrap();
    let mut got = Vec::new();
    let stream = stream();
    for i in 0..300 {
        let (x, y, d, w) = &stream[i % 60];
        let step = m.step(&x[..1], y, *d, *w);
        if [20, 150, 170, 299].contains(&i) {
            got.extend(&step.pred);
            got.push(step.n_eff);
        }
    }
    check("isotonic_decayed", &got, GOLDEN_ISOTONIC_DECAYED);
}

/// `ddsketch`'s decayed form, a half-life of 20 rows: the factor is the
/// platform's `exp2`, so it is held to the golden files' tolerance, not to
/// the bit.
#[test]
fn ddsketch_decayed_golden() {
    let mut m = DdSketchModel::new(DdSketchModelCfg {
        n_targets: 1,
        alpha: 0.01,
        levels: vec![0.05, 0.25, 0.95],
        window_rows: None,
        window_clock: None,
        closed: WindowClosed::Right,
        decay: Decay::Halflife(20.0),
        max_state_bytes: None,
    })
    .unwrap();
    let stream = stream();
    let mut got = Vec::new();
    for i in 0..300 {
        let (_, y, d, w) = stream[i % 60];
        let step = m.step(&[], &y, d, w);
        if [150, 170, 299].contains(&i) {
            got.extend(&step.pred);
            got.push(step.n_eff);
        }
    }
    check("ddsketch_decayed", &got, GOLDEN_DDSKETCH_DECAYED);
}

/// The free GARCH(1,1).
#[test]
fn garch_golden() {
    let got = garch_signature(GarchVol::Garch, GarchDist::Normal, false);
    check_bits("garch", &got, GOLDEN_GARCH);
}

/// The variance-targeted GARCH(1,1) and its Huber score.
#[test]
fn garch_targeted_golden() {
    let got = garch_signature(GarchVol::Garch, GarchDist::Normal, true);
    check_bits("garch_targeted", &got, GOLDEN_GARCH_TARGETED);
}

/// A GJR with a Student t: the shape's step and the Schur metric.
#[test]
fn garch_gjr_t_golden() {
    let got = garch_signature(GarchVol::Gjr, GarchDist::T, false);
    check_bits("garch_gjr_t", &got, GOLDEN_GARCH_GJR_T);
}

/// A GARCH with a skewed t.
#[test]
fn garch_skewt_golden() {
    let got = garch_signature(GarchVol::Garch, GarchDist::Skewt, false);
    check_bits("garch_skewt", &got, GOLDEN_GARCH_SKEWT);
}

/// An EGARCH: its bounds and its invertibility guard.
#[test]
fn garch_egarch_golden() {
    let got = garch_signature(GarchVol::Egarch, GarchDist::Normal, false);
    check_bits("garch_egarch", &got, GOLDEN_GARCH_EGARCH);
}

/// A FIGARCH of 20 lags after a start of 2,000 rows.
#[test]
fn garch_figarch_golden() {
    let got = garch_signature(GarchVol::Figarch, GarchDist::Normal, false);
    check_bits("garch_figarch", &got, GOLDEN_GARCH_FIGARCH);
}

/// `garch` on a series of the process it models, at the truth `arch` is
/// held to (`tests/test_second_opinion_garch.py`): 2,500 rows, the default
/// start of 2,000 rows, unit weights (PLAN 550's realistic case beside the
/// degenerate one above, whose start has no conditional variance to find
/// and now waits). The variance, the standardized residual and the forecast
/// (four rows past, but EGARCH's one) on rows 2,100, 2,300 and 2,499, then
/// the parameters at the end. The series calls no platform math library
/// (`garch_series`), so these are the same bits on every platform.
fn garch_realistic(vol: GarchVol, dist: GarchDist, targeting: bool, ys: &[f64]) -> Vec<f64> {
    let mut m = Garch::new(GarchCfg {
        n_inputs: 0,
        n_targets: 1,
        vol,
        dist,
        truncation: None,
        variance_targeting: targeting,
        start_rows: 2000,
        start_span: None,
        horizons: vec![5, 1],
        max_horizon_work: None,
        min_weight: 0.0,
        max_state_bytes: None,
    })
    .unwrap();
    let mut got = Vec::new();
    for (i, y) in ys.iter().enumerate() {
        let step = m.step(&[], &[Some(*y)], 1.0, 1.0);
        if [2100, 2300, 2499].contains(&i) {
            got.extend(&step.pred);
        }
    }
    assert!(m.started()[0], "the start's rows identify the process");
    got.extend(&m.coefficients()[0]);
    got
}

const GARCH_TRUTH: [f64; 4] = [0.05, 0.08, 0.0, 0.90];
const GJR_TRUTH: [f64; 4] = [0.03, 0.05, 0.10, 0.87];

#[test]
fn garch_realistic_golden() {
    let ys = garch_series::gjr(GARCH_TRUTH, Innovations::Normal, 2500, 1);
    let got = garch_realistic(GarchVol::Garch, GarchDist::Normal, false, &ys);
    check_bits("garch_realistic", &got, GOLDEN_GARCH_REALISTIC);
}

#[test]
fn garch_targeted_realistic_golden() {
    let ys = garch_series::gjr([0.02, 0.08, 0.0, 0.90], Innovations::Normal, 2500, 2);
    let got = garch_realistic(GarchVol::Garch, GarchDist::Normal, true, &ys);
    check_bits(
        "garch_targeted_realistic",
        &got,
        GOLDEN_GARCH_TARGETED_REALISTIC,
    );
}

#[test]
fn garch_gjr_realistic_golden() {
    let ys = garch_series::gjr(GJR_TRUTH, Innovations::Normal, 2500, 3);
    let got = garch_realistic(GarchVol::Gjr, GarchDist::Normal, false, &ys);
    check_bits("garch_gjr_realistic", &got, GOLDEN_GARCH_GJR_REALISTIC);
}

#[test]
fn garch_egarch_realistic_golden() {
    let ys = garch_series::egarch([0.0, 0.15, -0.08, 0.97], 2500, 4);
    let got = garch_realistic(GarchVol::Egarch, GarchDist::Normal, false, &ys);
    check_bits(
        "garch_egarch_realistic",
        &got,
        GOLDEN_GARCH_EGARCH_REALISTIC,
    );
}

#[test]
fn garch_figarch_realistic_golden() {
    let ys = garch_series::figarch([0.03, 0.2, 0.4, 0.5], 1000, 2500, 5);
    let got = garch_realistic(GarchVol::Figarch, GarchDist::Normal, false, &ys);
    check_bits(
        "garch_figarch_realistic",
        &got,
        GOLDEN_GARCH_FIGARCH_REALISTIC,
    );
}

#[test]
fn garch_t_realistic_golden() {
    let ys = garch_series::gjr(GARCH_TRUTH, Innovations::T6, 2500, 6);
    let got = garch_realistic(GarchVol::Garch, GarchDist::T, false, &ys);
    check_bits("garch_t_realistic", &got, GOLDEN_GARCH_T_REALISTIC);
}

/// The t at persistence 0.90, `(0.1, 0.1, 0.8)`.
#[test]
fn garch_t_low_persistence_golden() {
    let ys = garch_series::gjr([0.1, 0.1, 0.0, 0.8], Innovations::T6, 2500, 7);
    let got = garch_realistic(GarchVol::Garch, GarchDist::T, false, &ys);
    check_bits(
        "garch_t_low_persistence",
        &got,
        GOLDEN_GARCH_T_LOW_PERSISTENCE,
    );
}

#[test]
fn garch_skewt_realistic_golden() {
    let ys = garch_series::gjr(GARCH_TRUTH, Innovations::SkewT6(-0.2), 2500, 8);
    let got = garch_realistic(GarchVol::Garch, GarchDist::Skewt, false, &ys);
    check_bits("garch_skewt_realistic", &got, GOLDEN_GARCH_SKEWT_REALISTIC);
}

#[test]
fn garch_gjr_t_realistic_golden() {
    let ys = garch_series::gjr(GJR_TRUTH, Innovations::T6, 2500, 9);
    let got = garch_realistic(GarchVol::Gjr, GarchDist::T, false, &ys);
    check_bits("garch_gjr_t_realistic", &got, GOLDEN_GARCH_GJR_T_REALISTIC);
}

/// A realistic asymmetric DCC stream for `dcc` (`(a, b, g) = (0.04, 0.92,
/// 0.03)`, every target correlation 0.5), each series scaled by its own
/// GARCH(1,1) volatility where asked: arithmetic and `sqrt` alone,
/// innovations by Irwin and Hall's sum of twelve uniforms, so the same bits
/// on every platform. Every 97th row has its second series absent, every
/// 89th weighs 0.
fn dcc_stream(k: usize, n: usize, garch: bool) -> Vec<(Vec<Option<f64>>, f64)> {
    let mut s = 20261122u64;
    let kk = k * k;
    let mut qbar = vec![0.5_f64; kk];
    for i in 0..k {
        qbar[i * k + i] = 1.0;
    }
    let (a, b, g) = (0.04, 0.92, 0.03);
    let mut q = qbar.clone();
    let mut v = vec![1.0_f64; k];
    let mut e = vec![0.0_f64; k];
    let mut out = Vec::with_capacity(n);
    for t in 0..n + 200 {
        let mut l = vec![0.0_f64; kk];
        for i in 0..k {
            for j in 0..=i {
                let r: f64 = q[i * k + j] / (q[i * k + i] * q[j * k + j]).sqrt();
                let mut acc: f64 = if i == j { 1.0 } else { r };
                for m in 0..j {
                    acc -= l[i * k + m] * l[j * k + m];
                }
                l[i * k + j] = if i == j {
                    acc.sqrt()
                } else {
                    acc / l[j * k + j]
                };
            }
        }
        let x: Vec<f64> = (0..k)
            .map(|_| (0..12).map(|_| 0.5 * (lcg(&mut s) + 1.0)).sum::<f64>() - 6.0)
            .collect();
        let z: Vec<f64> = (0..k)
            .map(|i| (0..=i).map(|j| l[i * k + j] * x[j]).sum())
            .collect();
        for i in 0..k {
            for j in 0..k {
                let ij = i * k + j;
                q[ij] = (1.0 - a - b) * qbar[ij] - g * 0.5 * qbar[ij]
                    + a * z[i] * z[j]
                    + g * z[i].min(0.0) * z[j].min(0.0)
                    + b * q[ij];
            }
        }
        for i in 0..k {
            v[i] = 0.05 + 0.08 * e[i] * e[i] + 0.9 * v[i];
            e[i] = if garch { v[i].sqrt() * z[i] } else { z[i] };
        }
        if t >= 200 {
            let row = t - 200;
            let y = (0..k)
                .map(|i| (!(i == 1 && row % 97 == 50)).then_some(e[i]))
                .collect();
            out.push((y, if row % 89 == 40 { 0.0 } else { 1.0 }));
        }
    }
    out
}

/// `dcc` on [`dcc_stream`]: every slot on three rows after the start, and
/// the parameters at the end. It calls no platform math library (`ln` is
/// the `libm` crate's, and a decay by `lam` at unit steps takes no power),
/// so its numbers are the same bits everywhere.
fn dcc_signature(
    asymmetric: bool,
    params: Option<Vec<f64>>,
    margins: bool,
    decay: Decay,
) -> Vec<f64> {
    let k = 3;
    let mut m = Dcc::new(DccCfg {
        n_targets: k,
        asymmetric,
        margins: margins.then_some(GarchCfg {
            n_targets: k,
            vol: GarchVol::Garch,
            dist: GarchDist::Normal,
            truncation: None,
            variance_targeting: false,
            start_rows: 300,
            start_span: None,
            horizons: Vec::new(),
            max_horizon_work: None,
            min_weight: 0.0,
            max_state_bytes: None,
            n_inputs: 0,
        }),
        params,
        decay,
        start_rows: 200,
        start_span: None,
        min_weight: 0.0,
        max_state_bytes: None,
    })
    .unwrap();
    let n = 1500;
    let mut got = Vec::new();
    for (i, (y, w)) in dcc_stream(k, n, margins).iter().enumerate() {
        let step = m.step(&[], y, 1.0, *w);
        if [n - 700, n - 300, n - 1].contains(&i) {
            got.extend(&step.pred);
        }
    }
    got.extend(m.coefficients());
    got
}

/// DCC(1,1), estimated, on standardized residuals.
#[test]
fn dcc_golden() {
    let got = dcc_signature(false, None, false, Decay::Halflife(f64::INFINITY));
    check_bits("dcc", &got, GOLDEN_DCC);
}

/// ADCC, estimated: `g`, `N̄` and `δ`'s projection.
#[test]
fn dcc_adcc_golden() {
    let got = dcc_signature(true, None, false, Decay::Halflife(f64::INFINITY));
    check_bits("dcc_adcc", &got, GOLDEN_DCC_ADCC);
}

/// ADCC with its parameters given.
#[test]
fn dcc_given_golden() {
    let got = dcc_signature(
        true,
        Some(vec![0.04, 0.92, 0.03]),
        false,
        Decay::Halflife(f64::INFINITY),
    );
    check_bits("dcc_given", &got, GOLDEN_DCC_GIVEN);
}

/// DCC on returns, through GARCH margins.
#[test]
fn dcc_margins_golden() {
    let got = dcc_signature(false, None, true, Decay::Halflife(f64::INFINITY));
    check_bits("dcc_margins", &got, GOLDEN_DCC_MARGINS);
}

/// ADCC under a decay: the targets', the weight's and the gain's.
#[test]
fn dcc_decayed_golden() {
    let got = dcc_signature(true, None, false, Decay::Lam(0.999));
    check_bits("dcc_decayed", &got, GOLDEN_DCC_DECAYED);
}

/// The pre-averaged estimate on the same stream: the other arithmetic.
#[test]
fn rcov_preavg_golden() {
    let mut m = Rcov::new(RcovCfg {
        n_features: 2,
        kind: RcovKind::Preavg,
        kernel: "parzen".into(),
        bandwidth: None,
        jitter: 2,
        theta: 1.0,
        psd: false,
        block_rows: Some(60),
        max_bandwidth: None,
        preavg_rows: Some(6),
        noise_stride: 1,
        iv_stride: 20,
    })
    .unwrap();
    for (x, y, d, w) in stream() {
        m.step(&x, &y, d, w);
    }
    let cov = m.estimate().rcov.expect("a block");
    check("rcov_preavg", &[cov[0], cov[1], cov[3]], GOLDEN_RCOV_PREAVG);
}

#[test]
fn deco_golden() {
    // Slot 1 is `rho`, the level the whole model is for: it reads the
    // standardiser, the row estimate and the recursion at once. The
    // `"linear"` dynamics is the busier path -- it runs the `"ew"` recursion
    // alongside as its target -- so that is the one pinned.
    let mut m = Deco::new(DecoCfg {
        n_features: 2,
        decay: Decay::Halflife(20.0),
        dynamics: DecoDynamics::Linear,
        alpha: Some(0.05),
        beta: Some(0.9),
        blocks: Vec::new(),
        min_weight: 3.0,
    })
    .unwrap();
    check("deco", &signature(&mut m, 1), GOLDEN_DECO);
}

#[test]
fn deco_loglik_golden() {
    // Slot 2 is `loglik`, which is the only consumer of the block algebra.
    let mut m = Deco::new(DecoCfg {
        n_features: 2,
        decay: Decay::Halflife(20.0),
        dynamics: DecoDynamics::Ew,
        alpha: None,
        beta: None,
        blocks: Vec::new(),
        min_weight: 3.0,
    })
    .unwrap();
    check("deco_loglik", &signature(&mut m, 2), GOLDEN_DECO_LOGLIK);
}

#[test]
fn ew_cov_golden() {
    // Slots in emission order: mean x0, mean x1, var x0, var x1, corr x0x1.
    // The correlation is the one that reads every accumulator at once.
    let mut m = EwCovModel::new(EwCovCfg {
        bias: true,
        n_features: 2,
        decay: Decay::Halflife(20.0),
        stats: vec![EwCovStat::Mean, EwCovStat::Var, EwCovStat::Corr],
        min_weight: 3.0,
        precision_prior: None,
        mahal_quantiles: Vec::new(),
        pca: 0,
        pca_every: 0.0,
        max_rows_between_pca: u32::MAX,
        lags: Vec::new(),
        window: None,
        window_every: None,
        max_rows_between_snapshots: None,
    })
    .unwrap();
    check("ew_cov", &signature(&mut m, 4), GOLDEN_EW_COV);
}

/// `marginal` predicts nothing per row, so its signature is read from the
/// state: the pair `(x1, y)`'s correlation, slope and Kish size after rows
/// 20, 45 and 59 -- together they touch every accumulator (both means, both
/// variances, the covariance, `W_t` and `Q_t`).
fn marginal_signature(model: &mut Marginal) -> Vec<f64> {
    let mut out = Vec::new();
    for (i, (x, y, d, w)) in stream().into_iter().enumerate() {
        let step = model.step(&x, &y, d, w);
        assert!(step.pred.is_empty());
        if matches!(i, 20 | 45 | 59) {
            let p = model.pair(0, 1);
            out.extend([p.corr, p.beta, p.n_kish]);
        }
    }
    out
}

#[test]
fn marginal_golden() {
    let mut m = Marginal::new(MarginalCfg {
        n_features: 2,
        n_targets: 1,
        decay: Decay::Halflife(20.0),
        min_weight: vec![3.0],
        lags: Vec::new(),
        serial_rule: None,
        cross_lags: None,
        bins: None,
        feature_moments: rheon_core::FeatureMomentLayout::PerTarget,
        window: None,
        window_every: None,
        max_rows_between_snapshots: None,
    })
    .unwrap();
    check("marginal", &marginal_signature(&mut m), GOLDEN_MARGINAL);
}

fn kmeans_cfg(rule: SeedRule) -> KMeansCfg {
    KMeansCfg {
        n_features: 2,
        k: 3,
        decay: Decay::Halflife(20.0),
        min_weight: 3.0,
        warm_rows: 12,
        seed_rule: rule,
        seed: 0,
        update_every_rows: 1,
        split_merge: 0.5,
        split_merge_every_rows: 10,
        dead_frac: 0.05,
        standardize: true,
        scale_floor: 0.0,
        warm_clock: None,
        update_every: None,
        split_merge_every: None,
    }
}

#[test]
fn kmeans_golden() {
    // Slot 1 is the distance to the assigned centre under the standardized
    // metric: it reads the centres, the feature moments and the assignment
    // at once. Slot 0 pins the assignment itself, seeded by the generator.
    let mut m = KMeans::new(kmeans_cfg(SeedRule::Lloyd)).unwrap();
    check("kmeans", &signature(&mut m, 1), GOLDEN_KMEANS);
    let mut m = KMeans::new(kmeans_cfg(SeedRule::Lloyd)).unwrap();
    check(
        "kmeans_cluster",
        &signature(&mut m, 0),
        GOLDEN_KMEANS_CLUSTER,
    );
    // The `first` rule with a checkpoint every seven rows: the batch path.
    let mut m = KMeans::new(KMeansCfg {
        update_every_rows: 7,
        ..kmeans_cfg(SeedRule::First)
    })
    .unwrap();
    check("kmeans_first", &signature(&mut m, 2), GOLDEN_KMEANS_FIRST);
}

fn micro_cfg() -> MicroCfg {
    MicroCfg {
        n_features: 2,
        decay: Decay::Halflife(20.0),
        min_weight: 3.0,
        eps: 0.35,
        beta_mu: 2.0,
        max_clusters: 8,
        prune_every: f64::INFINITY,
        max_rows_between_prunes: 7,
        macro_link: None,
        standardize: true,
        scale_floor: 0.0,
    }
}

#[test]
fn micro_golden() {
    // Slot 1 is the distance to the nearest potential summary under the
    // standardized metric: it reads the centres, the moments and the
    // promotion rule at once. Slot 0 pins the labels the linkage assigns
    // with the derived threshold; slot 2 the ids, which read the cap and
    // the pruning.
    let mut m = Micro::new(micro_cfg()).unwrap();
    check("micro", &signature(&mut m, 1), GOLDEN_MICRO);
    let mut m = Micro::new(micro_cfg()).unwrap();
    check("micro_cluster", &signature(&mut m, 0), GOLDEN_MICRO_CLUSTER);
    let mut m = Micro::new(MicroCfg {
        macro_link: Some(0.0),
        standardize: false,
        ..micro_cfg()
    })
    .unwrap();
    check("micro_id", &signature(&mut m, 2), GOLDEN_MICRO_ID);
}

fn ew_class_cfg(covariance: Covariance) -> EwClassCfg {
    EwClassCfg {
        n_features: 2,
        n_classes: 2,
        decay: Decay::Halflife(20.0),
        min_weight: 3.0,
        covariance,
        precision_prior: 0.1,
        window: None,
        window_every: None,
        max_rows_between_snapshots: None,
    }
}

/// The three rows' outputs with the stream's target turned into a label:
/// class 1 where `y > 0.25`, else class 0; the null stays null.
fn labelled_signature<M: OnlineModel>(model: &mut M, pick: usize) -> Vec<f64> {
    let mut out = Vec::new();
    for (i, (x, y, d, w)) in stream().into_iter().enumerate() {
        let label = y[0].map(|v| if v > 0.25 { 1.0 } else { 0.0 });
        let step = model.step(&x, &[label], d, w);
        if matches!(i, 20 | 45 | 59) {
            out.push(step.pred[pick]);
        }
    }
    out
}

#[test]
fn ew_class_golden() {
    // Slot 1 is the posterior of class 0: it reads both classes' means,
    // co-moments, decaying ridges and priors at once, through the solve.
    // Slot 0 pins the assignment.
    let mut m = EwClass::new(ew_class_cfg(Covariance::Full)).unwrap();
    check("ew_class", &labelled_signature(&mut m, 1), GOLDEN_EW_CLASS);
    let mut m = EwClass::new(ew_class_cfg(Covariance::Full)).unwrap();
    check(
        "ew_class_class",
        &labelled_signature(&mut m, 0),
        GOLDEN_EW_CLASS_CLASS,
    );
    let mut m = EwClass::new(ew_class_cfg(Covariance::Shared)).unwrap();
    check(
        "ew_class_shared",
        &labelled_signature(&mut m, 2),
        GOLDEN_EW_CLASS_SHARED,
    );
    let mut m = EwClass::new(ew_class_cfg(Covariance::Diagonal)).unwrap();
    check(
        "ew_class_diagonal",
        &labelled_signature(&mut m, 1),
        GOLDEN_EW_CLASS_DIAGONAL,
    );
}

// --- second signatures: the settings the first never runs ---
//
// One per model, each over what its golden above leaves at a default, so a
// mutant in those branches moves a number `cargo mutants` sees: the Python
// golden pipeline pins most of them, which `cargo test` cannot see (review
// 2026-10-06, CF10). Clear of what task 186 moves: no target that starts
// late in a solving model, no windowed `ew_class` coefficient (its
// posteriors only), no `share_p`, no `rcov` across a break.

/// The fixed stream with `x1` held still from row 30 on: a feature gone
/// quiet, where a clusterer's `scale_floor` binds.
fn quiet_stream() -> Vec<Row> {
    stream()
        .into_iter()
        .enumerate()
        .map(|(i, (x, y, d, w))| (if i >= 30 { [x[0], 0.3] } else { x }, y, d, w))
        .collect()
}

/// [`signature`] on `rows`.
fn signature_on<M: OnlineModel>(model: &mut M, pick: usize, rows: Vec<Row>) -> Vec<f64> {
    let mut out = Vec::new();
    for (i, (x, y, d, w)) in rows.into_iter().enumerate() {
        let step = model.step(&x, &y, d, w);
        if matches!(i, 20 | 45 | 59) {
            out.push(step.pred[pick]);
        }
    }
    out
}

/// [`signature`] with a call between rows: `between(model, i)` runs before
/// row `i`.
fn signature_between<M: OnlineModel>(
    model: &mut M,
    pick: usize,
    mut between: impl FnMut(&mut M, usize),
) -> Vec<f64> {
    let mut out = Vec::new();
    for (i, (x, y, d, w)) in stream().into_iter().enumerate() {
        between(model, i);
        let step = model.step(&x, &y, d, w);
        if matches!(i, 20 | 45 | 59) {
            out.push(step.pred[pick]);
        }
    }
    out
}

/// `kmeans` seeded by farthest-first and by k-means++, on the raw metric,
/// and on the standardized one with a scale floor, where a feature has gone
/// quiet (`quiet_stream`): the floor moves the distances there.
#[test]
fn kmeans_seeding_and_metric_golden() {
    let mut m = KMeans::new(kmeans_cfg(SeedRule::Farthest)).unwrap();
    check(
        "kmeans_farthest",
        &signature(&mut m, 1),
        GOLDEN_KMEANS_FARTHEST,
    );
    let mut m = KMeans::new(kmeans_cfg(SeedRule::Kmeanspp)).unwrap();
    check(
        "kmeans_kmeanspp",
        &signature(&mut m, 1),
        GOLDEN_KMEANS_KMEANSPP,
    );
    let mut m = KMeans::new(KMeansCfg {
        standardize: false,
        ..kmeans_cfg(SeedRule::Lloyd)
    })
    .unwrap();
    check("kmeans_raw", &signature(&mut m, 1), GOLDEN_KMEANS_RAW);
    let floored = |scale_floor| {
        let mut m = KMeans::new(KMeansCfg {
            scale_floor,
            ..kmeans_cfg(SeedRule::Lloyd)
        })
        .unwrap();
        signature_on(&mut m, 1, quiet_stream())
    };
    let got = floored(0.5);
    assert_ne!(got, floored(0.0), "the floor binds on the quiet feature");
    check("kmeans_floor", &got, GOLDEN_KMEANS_FLOOR);
}

/// `micro` pruned on the clock, every 5 units (the gap of 25 is one prune),
/// and with a scale floor where a feature has gone quiet.
#[test]
fn micro_on_the_clock_golden() {
    let mut m = Micro::new(MicroCfg {
        prune_every: 5.0,
        max_rows_between_prunes: u32::MAX,
        ..micro_cfg()
    })
    .unwrap();
    check("micro_clock", &signature(&mut m, 1), GOLDEN_MICRO_CLOCK);
    let floored = |scale_floor| {
        let mut m = Micro::new(MicroCfg {
            scale_floor,
            ..micro_cfg()
        })
        .unwrap();
        signature_on(&mut m, 1, quiet_stream())
    };
    let got = floored(0.5);
    assert_ne!(got, floored(0.0), "the floor binds on the quiet feature");
    check("micro_floor", &got, GOLDEN_MICRO_FLOOR);
}

/// `ewridge` over feature sets, under `session_shrink` with a session
/// boundary at row 30, with pairwise gaps, with the Gram blocked, and on
/// the solve schedules: the clock, and the default share of the weight.
#[test]
fn ewridge_paths_golden() {
    let mut c = ewridge_cfg(false, 1e-4);
    c.feature_sets = vec![("one".into(), vec![0]), ("both".into(), vec![0, 1])];
    let mut m = EwRidge::new(c.clone()).unwrap();
    check(
        "ewridge_set_one",
        &signature(&mut m, 0),
        GOLDEN_EW_RIDGE_SET_ONE,
    );
    let mut m = EwRidge::new(c).unwrap();
    check(
        "ewridge_set_both",
        &signature(&mut m, 1),
        GOLDEN_EW_RIDGE_SET_BOTH,
    );

    let mut c = ewridge_cfg(false, 1e-4);
    c.session_shrink = Some(0.5);
    c.long_half_life = Some(80.0);
    let mut m = EwRidge::new(c).unwrap();
    let got = signature_between(&mut m, 0, |m, i| {
        if i == 30 {
            m.blend_toward_long_run();
        }
    });
    check(
        "ewridge_session_shrink",
        &got,
        GOLDEN_EW_RIDGE_SESSION_SHRINK,
    );

    let mut c = ewridge_cfg(false, 1e-4);
    c.target_gaps = rheon_core::TargetGaps::Pairwise;
    let mut m = EwRidge::new(c).unwrap();
    check(
        "ewridge_pairwise",
        &signature(&mut m, 0),
        GOLDEN_EW_RIDGE_PAIRWISE,
    );

    let mut c = ewridge_cfg(false, 1e-4);
    c.gram_block_rows = 5;
    c.solve_every = 4.0;
    c.max_rows_between_solves = 8;
    let mut m = EwRidge::new(c).unwrap();
    check(
        "ewridge_blocked",
        &signature(&mut m, 0),
        GOLDEN_EW_RIDGE_BLOCKED,
    );

    let mut c = ewridge_cfg(false, 1e-4);
    c.solve_every = 3.0;
    c.max_rows_between_solves = u32::MAX;
    let mut m = EwRidge::new(c).unwrap();
    check(
        "ewridge_on_the_clock",
        &signature(&mut m, 0),
        GOLDEN_EW_RIDGE_ON_THE_CLOCK,
    );

    // A fifth of the fit's weight between solves: the default share, ln 2 /
    // 50, solves after every row on a stream this short.
    let mut c = ewridge_cfg(false, 1e-4);
    c.max_rows_between_solves = u32::MAX;
    c.solve_share = Some(0.2);
    let mut m = EwRidge::new(c).unwrap();
    check(
        "ewridge_by_weight",
        &signature(&mut m, 0),
        GOLDEN_EW_RIDGE_BY_WEIGHT,
    );
}

/// The lasso under a window, and selecting on its own half-life.
#[test]
fn lasso_windowed_golden() {
    let cfg = || LassoCfg {
        min_weight_holds_learning: false,
        n_features: 2,
        n_targets: 1,
        fit_intercept: true,
        decay: Decay::Halflife(20.0),
        lasso_path: vec![0.2, 0.02, 0.0],
        l1_ratio: 1.0,
        select_half_life: None,
        min_weight: 3.0,
        target_min_weight: Vec::new(),
        solve_every: 0.0,
        max_rows_between_solves: 1,
        solve_share: None,
        window: None,
        window_every: None,
        max_rows_between_snapshots: None,
        max_iter: 200,
        tol: 1e-12,
        target_gaps: rheon_core::TargetGaps::OwnRows,
    };
    let mut m = Lasso::new(LassoCfg {
        window: Some(30.0),
        ..cfg()
    })
    .unwrap();
    // Pinned under `closed = "both"`, the edge every window had before task
    // 196, so the numbers are the old ones to the bit; `"right"`, the
    // default, is held to the definition in each model's unit tests.
    m.set_window_closed(WindowClosed::Both);
    check(
        "lasso_windowed",
        &signature(&mut m, 1),
        GOLDEN_LASSO_WINDOWED,
    );
    // The selection, read every fifth row from row 10: the penalty whose EW
    // mean squared error, at `select_half_life`, is the least.
    let selected = |select_half_life| {
        let mut m = Lasso::new(LassoCfg {
            select_half_life,
            ..cfg()
        })
        .unwrap();
        let mut out = Vec::new();
        for (i, (x, y, d, w)) in stream().into_iter().enumerate() {
            m.step(&x, &y, d, w);
            if i >= 10 && i % 5 == 0 {
                out.push(m.lam_selected()[0]);
            }
        }
        out
    };
    let got = selected(Some(3.0));
    assert_ne!(
        got,
        selected(None),
        "the selection's own half-life moves it"
    );
    check("lasso_select", &got, GOLDEN_LASSO_SELECT);
}

/// `kalman` with an explicit process noise, and with a fixed observation
/// noise and its prior.
#[test]
fn kalman_noise_golden() {
    let cfg = || KalmanCfg {
        horizons: Vec::new(),
        // The form a state saved before schema 64 goes on in: `P` absolute
        // and `min_weight` holding learning back, as `tests/reference.py`'s
        // `kalman_ref` has it (task 278, D10).
        min_weight_holds_learning: true,
        noise_units: false,
        n_features: 2,
        n_targets: 1,
        fit_intercept: true,
        decay: Decay::Halflife(50.0),
        half_life: vec![f64::INFINITY, 30.0, 100.0],
        q: None,
        obs_var: None,
        p0: 1.0,
        share_p: false,
        min_weight: 3.0,
        revert_half_life: vec![f64::INFINITY],
        standardize: false,
    };
    let mut m = Kalman::new(KalmanCfg {
        q: Some(vec![0.0, 0.01, 0.02]),
        ..cfg()
    })
    .unwrap();
    check("kalman_q", &signature(&mut m, 0), GOLDEN_KALMAN_Q);
    let mut m = Kalman::new(KalmanCfg {
        obs_var: Some(0.25),
        p0: 4.0,
        ..cfg()
    })
    .unwrap();
    check(
        "kalman_obs_var",
        &signature(&mut m, 0),
        GOLDEN_KALMAN_OBS_VAR,
    );
}

/// `hmm` with a transition that moves with an exogenous column, the
/// stream's target in the target slot.
#[test]
fn hmm_tvtp_golden() {
    let mut m = Hmm::new(HmmCfg {
        n_features: 2,
        k: 2,
        decay: Decay::Halflife(20.0),
        covariance: Covariance::Full,
        precision_prior: 1e-2,
        min_weight: 3.0,
        learn: true,
        transition_prior: 1.0,
        transition: None,
        means: Some(vec![-0.5, -0.5, 0.5, 0.5]),
        covs: Some(vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]),
        warm_rows: 10,
        seed_rule: SeedRule::First,
        seed: 1,
        tvtp: Some((vec![2.0, 0.0, 0.0, 2.0], vec![0.0, 1.5, -1.0, 0.0])),
        prior_rows: None,
        warm_clock: None,
    })
    .unwrap();
    check("hmm_tvtp", &signature(&mut m, 5), GOLDEN_HMM_TVTP);
}

/// `bocpd` with the full Gaussian emission, and the robust one at a
/// `robust_beta` above 0.
#[test]
fn bocpd_emissions_golden() {
    let cfg = |emission, robust_beta| BocpdCfg {
        n_features: 2,
        hazard: 30.0,
        hazard_from_row: false,
        emission,
        prior_mean: Some(vec![0.0, 0.0]),
        prior_kappa: 1.0,
        prior_nu: Some(4.0),
        prior_scale: Some(vec![1.0]),
        robust_beta,
        prune_below: 1e-8,
        max_run: 100,
        min_weight: 0.0,
        warm_rows: None,
        hazard_on_clock: false,
        warm_clock: None,
    };
    let mut m = Bocpd::new(cfg(BocpdEmission::Gaussian, 0.0)).unwrap();
    check(
        "bocpd_gaussian",
        &signature(&mut m, 0),
        GOLDEN_BOCPD_GAUSSIAN,
    );
    let mut m = Bocpd::new(cfg(BocpdEmission::Robust, 0.2)).unwrap();
    check("bocpd_robust", &signature(&mut m, 0), GOLDEN_BOCPD_ROBUST);
}

/// `corrchange`'s sequential monitor: a history of 20 rows, then 20
/// monitored against W&G's boundary at `γ = 0.25`. Its signature is the
/// statistic on every row it reports.
#[test]
fn corrchange_sequential_golden() {
    let mut m = CorrChange::new(CorrChangeCfg {
        n_features: 2,
        kind: CorrChangeKind::Sequential,
        span_rows: 20,
        alpha: 0.05,
        alpha_adjust: "bonferroni".into(),
        bandwidth: None,
        scalar: false,
        decay: Decay::Halflife(20.0),
        crit: None,
        n_perm: 20,
        permute_every_rows: 10,
        perm_block: 1,
        norm: ChangeNorm::L1,
        seed: 5,
        reset: false,
        monitor_rows: 20,
        boundary_gamma: 0.25,
        permute_every: None,
        span_clock: None,
        monitor_clock: None,
        max_state_bytes: None,
    })
    .unwrap();
    let mut out = Vec::new();
    for (x, y, d, w) in stream() {
        let step = m.step(&x, &y, d, w);
        if step.pred[0].is_finite() {
            out.push(step.pred[0]);
        }
    }
    assert!(out.len() >= 10, "the monitor reports: {}", out.len());
    check("corrchange_sequential", &out, GOLDEN_CORRCHANGE_SEQUENTIAL);
}

/// `rcov`'s pre-averaged estimate in its PSD form: the longer window, and
/// no bias term. (The kernel's PSD form clips a negative eigenvalue, which
/// this stream's estimate does not have: it repeats `GOLDEN_RCOV`. The clip
/// is held in `tests/test_rcov.py`.)
#[test]
fn rcov_psd_golden() {
    let cfg = |kind, bandwidth, preavg_rows| RcovCfg {
        n_features: 2,
        kind,
        kernel: "parzen".into(),
        bandwidth,
        jitter: 2,
        theta: 1.0,
        psd: true,
        block_rows: Some(60),
        max_bandwidth: None,
        preavg_rows,
        noise_stride: 1,
        iv_stride: 20,
    };
    for (name, c, want) in [(
        "rcov_preavg_psd",
        cfg(RcovKind::Preavg, None, None),
        GOLDEN_RCOV_PREAVG_PSD,
    )] {
        let mut m = Rcov::new(c).unwrap();
        for (x, y, d, w) in stream() {
            m.step(&x, &y, d, w);
        }
        let cov = m.estimate().rcov.expect("a block");
        check(name, &[cov[0], cov[1], cov[3]], want);
    }
}

/// `ew_cov` with lagged moments, with a principal component refreshed on
/// every row, and under a window.
#[test]
fn ew_cov_paths_golden() {
    let cfg = || EwCovCfg {
        bias: true,
        n_features: 2,
        decay: Decay::Halflife(20.0),
        stats: vec![EwCovStat::Corr],
        min_weight: 3.0,
        precision_prior: None,
        mahal_quantiles: Vec::new(),
        pca: 0,
        pca_every: 0.0,
        max_rows_between_pca: u32::MAX,
        lags: Vec::new(),
        window: None,
        window_every: None,
        max_rows_between_snapshots: None,
    };
    // Slots: corr, then lag_corr per lag and ordered pair; slot 3 is lag 1's
    // (x0, x1).
    let mut m = EwCovModel::new(EwCovCfg {
        stats: vec![EwCovStat::Corr, EwCovStat::LagCorr],
        lags: vec![1, 2],
        ..cfg()
    })
    .unwrap();
    check("ew_cov_lags", &signature(&mut m, 2), GOLDEN_EW_COV_LAGS);
    // Slots: corr, then pc0's variance, share, two loadings and score; the
    // score reads the eigenvectors and the means at once.
    let mut m = EwCovModel::new(EwCovCfg { pca: 1, ..cfg() }).unwrap();
    check("ew_cov_pca", &signature(&mut m, 5), GOLDEN_EW_COV_PCA);
    let mut m = EwCovModel::new(EwCovCfg {
        window: Some(30.0),
        ..cfg()
    })
    .unwrap();
    // Under `closed = "both"`, as `lasso_windowed_golden` says why.
    m.set_window_closed(WindowClosed::Both);
    check(
        "ew_cov_windowed",
        &signature(&mut m, 0),
        GOLDEN_EW_COV_WINDOWED,
    );
}

/// `marginal` with lags and the serial count they feed, with bins and the
/// best split, and under a window.
#[test]
fn marginal_paths_golden() {
    let cfg = || MarginalCfg {
        n_features: 2,
        n_targets: 1,
        decay: Decay::Halflife(20.0),
        min_weight: vec![3.0],
        lags: Vec::new(),
        serial_rule: None,
        cross_lags: None,
        bins: None,
        feature_moments: rheon_core::FeatureMomentLayout::PerTarget,
        window: None,
        window_every: None,
        max_rows_between_snapshots: None,
    };
    let read = |m: &mut Marginal, f: &dyn Fn(&MarginalPair) -> [f64; 3]| {
        let mut out = Vec::new();
        for (i, (x, y, d, w)) in stream().into_iter().enumerate() {
            m.step(&x, &y, d, w);
            if matches!(i, 20 | 45 | 59) {
                out.extend(f(&m.pair(0, 1)));
            }
        }
        out
    };
    let mut m = Marginal::new(MarginalCfg {
        lags: vec![1, 2],
        serial_rule: Some(rheon_core::SerialRule::Truncated),
        ..cfg()
    })
    .unwrap();
    let got = read(&mut m, &|p| [p.n_serial, p.t_serial, p.lag_corr_xy[0]]);
    check("marginal_lags", &got, GOLDEN_MARGINAL_LAGS);
    let mut m = Marginal::new(MarginalCfg {
        bins: Some(Box::new(rheon_core::BinCfg {
            n_bins: 4,
            edges: None,
            rule: rheon_core::BinRule::Quantile,
            warm_rows: 10,
            stat: rheon_core::BinStat::Mean,
            refit_rows: 0,
            budget_mib: None,
            mad_anchor: false,
        })),
        ..cfg()
    })
    .unwrap();
    let got = read(&mut m, &|p| [p.split_gain, p.split_at, p.bin_mean_y[1]]);
    check("marginal_bins", &got, GOLDEN_MARGINAL_BINS);
    let mut m = Marginal::new(MarginalCfg {
        window: Some(30.0),
        ..cfg()
    })
    .unwrap();
    // Under `closed = "both"`, as `lasso_windowed_golden` says why.
    m.set_window_closed(WindowClosed::Both);
    let got = read(&mut m, &|p| [p.corr, p.beta, p.n_kish]);
    check("marginal_windowed", &got, GOLDEN_MARGINAL_WINDOWED);
}

/// `ew_class` under a window: the posterior, which reads the window's
/// class moments (its `coef` is task 186's).
#[test]
fn ew_class_windowed_golden() {
    let mut m = EwClass::new(EwClassCfg {
        window: Some(30.0),
        ..ew_class_cfg(Covariance::Full)
    })
    .unwrap();
    // Under `closed = "both"`, as `lasso_windowed_golden` says why.
    m.set_window_closed(WindowClosed::Both);
    check(
        "ew_class_windowed",
        &labelled_signature(&mut m, 1),
        GOLDEN_EW_CLASS_WINDOWED,
    );
}

/// `pca` on the golden stream's two columns mixed, `(3 x0 + x1, x0 - 0.5
/// x1)`, so its components differ (the columns alone are of one spread, a
/// tie `pca` withholds): component 0's variance, first loading and score at
/// rows 20, 45 and 59.
fn pca_signature(method: PcaMethod) -> Vec<f64> {
    let mut m = StreamPca::new(StreamPcaCfg {
        n_features: 2,
        n_components: 1,
        method,
        decay: Decay::Halflife(20.0),
        min_weight: 0.0,
        learning_rate: None,
        max_state_bytes: None,
    })
    .unwrap();
    let mut out = Vec::new();
    for (i, (x, _, d, w)) in stream().into_iter().enumerate() {
        let x = [3.0 * x[0] + x[1], x[0] - 0.5 * x[1]];
        let step = m.step(&x, &[], d, w);
        if matches!(i, 20 | 45 | 59) {
            out.extend([step.pred[0], step.pred[2], step.pred[4]]);
        }
    }
    out
}

#[test]
fn pca_golden() {
    check("pca", &pca_signature(PcaMethod::Svd), GOLDEN_PCA);
}

#[test]
fn pca_ccipca_golden() {
    check(
        "pca_ccipca",
        &pca_signature(PcaMethod::Ccipca),
        GOLDEN_PCA_CCIPCA,
    );
}

#[test]
fn pca_oja_golden() {
    check("pca_oja", &pca_signature(PcaMethod::Oja), GOLDEN_PCA_OJA);
}

/// A realistic stream for `ewm_moments`: a skewed column at a level of
/// about 6 (a shifted exponential of mean 3), an AR(1) column at 0.6, a
/// clock of uneven steps with a gap of 25, uneven weights with two rows of
/// weight 0 (the first among them). 400 rows.
fn moments_stream() -> Vec<([f64; 2], f64, f64)> {
    let mut s = 554_u64;
    let mut ar = 0.0;
    (0..400)
        .map(|i| {
            let u = 0.5 * (lcg(&mut s) + 1.0);
            let g = 3.0 - 3.0 * (1.0 - u).max(1e-12).ln();
            ar = 0.6 * ar + lcg(&mut s);
            let d = match i {
                0 => 0.0,
                200 => 25.0,
                _ => 0.6 + 0.4 * lcg(&mut s),
            };
            let w = if i == 0 || i == 57 {
                0.0
            } else {
                0.75 + 0.5 * lcg(&mut s)
            };
            ([g, ar], d, w)
        })
        .collect()
}

/// `ewm_moments`: a skew, an unbiased kurtosis, a fourth cumulant, a
/// co-moment of order (1, 2), a raw second moment and an autocorrelation at
/// a clock lag of 1.5, each with its standard error, read before rows 150,
/// 260 and 399; then the same terms over a window of 40 clock units. The
/// structure is held (two outputs a term, every one defined by row 150, a
/// positive skew of the skewed column, a positive autocorrelation of the
/// AR(1) one) and the numbers pinned.
#[test]
fn ewm_moments_golden() {
    let col = |c, p| (MomentLeg::of(c), p);
    let term = |name: &str, powers, kind, bias| MomentTerm {
        name: name.into(),
        powers,
        kind,
        bias,
    };
    let terms = vec![
        term("skew", vec![col(0, 3)], MomentKind::Standardized, true),
        term("kurt", vec![col(0, 4)], MomentKind::Standardized, false),
        term("k4", vec![col(0, 4)], MomentKind::Cumulant, true),
        term("co", vec![col(0, 1), col(1, 2)], MomentKind::Central, true),
        term("raw2", vec![col(1, 2)], MomentKind::Raw, true),
        term(
            "ac",
            vec![col(1, 1), (MomentLeg::lagged(1, 1.5), 1)],
            MomentKind::Standardized,
            true,
        ),
    ];
    let mut cfg = EwmMomentsCfg::new(Decay::Halflife(80.0), terms);
    cfg.clocked = true;
    cfg.se = true;
    for (name, window, want) in [
        ("ewm_moments", None, GOLDEN_EWM_MOMENTS),
        ("ewm_moments_window", Some(40.0), GOLDEN_EWM_MOMENTS_WINDOW),
    ] {
        let mut m = EwmMoments::new(EwmMomentsCfg {
            window_clock: window,
            ..cfg.clone()
        })
        .unwrap();
        assert_eq!(
            m.n_outputs(),
            12,
            "{name}: a term and its standard error each"
        );
        let mut got = Vec::new();
        for (i, (x, d, w)) in moments_stream().into_iter().enumerate() {
            let step = m.step(&x, &[], d, w);
            if [150, 260, 399].contains(&i) {
                assert!(step.pred.iter().all(|v| v.is_finite()), "{name}: row {i}");
                assert!(step.pred[0] > 0.0 && step.pred[5] > 0.0, "{name}: row {i}");
                got.extend(&step.pred);
                got.push(step.n_eff);
            }
        }
        check(name, &got, want);
    }
}

// --- generated; see the module docs ---
const GOLDEN_ISOTONIC: &[f64] = &[
    0.058108255205977066,
    -1.1322602444563972,
    0.7026314179645919,
    -0.14824593943839737,
    0.5234052159619503,
    0.2924647563908718,
    0.23662083677230278,
    0.098572970303041,
    0.058108255205977066,
    -1.2452224788841555,
    1.2824141151347344,
    -0.3156758345391851,
];
const GOLDEN_ISOTONIC_DECAYED: &[f64] = &[
    0.060139151477943076,
    14.342256954981341,
    -0.9887640434617117,
    26.867539807603332,
    0.7998676929930724,
    16.92073916656056,
    -0.18039268640178485,
    20.212303545600623,
];
const GOLDEN_REALIZED: &[f64] = &[
    20.301763088150093,
    14.524981482223351,
    15.210767597581253,
    152.5427122837151,
    1.4617729663448074,
    0.26175973848626405,
    0.0,
    10.459155766645733,
    10.642877740344339,
    11.845985890132823,
    97.82724355929129,
    -0.08494201272143055,
    2.354413313024124,
    0.0,
    15.904128496397625,
    18.06313225979898,
    19.541166883965733,
    266.2749803565534,
    -0.8091849003222343,
    -0.18028949568691144,
    0.0,
];
const GOLDEN_REALIZED_WINDOW: &[f64] = &[
    23.20165621168584,
    13.363161652229811,
    14.77767677736757,
    146.95773760815013,
    1.7426913336195444,
    0.2768024645607887,
    0.0,
    12.833685109912663,
    13.473997818648956,
    13.546309083923113,
    188.87998695368736,
    -0.20102730581128775,
    2.6349174735544527,
    0.0,
    11.7484856053016,
    13.10735007891979,
    12.579384454213699,
    131.7993766981005,
    -0.47534076903068045,
    -0.1662952885840179,
    0.0,
];

const GOLDEN_PCA: &[f64] = &[
    3.111731224759684,
    0.9381509032059694,
    0.3805052170124993,
    3.5709325086051242,
    0.9585947766729436,
    2.0614940695355655,
    3.993684243361783,
    0.9648635656349285,
    -0.6711049564076937,
];
const GOLDEN_PCA_CCIPCA: &[f64] = &[
    3.0947140108359883,
    0.9392450889119539,
    0.381798733276914,
    3.5578435679969886,
    0.9579038842355805,
    2.063278804381221,
    3.984871387572133,
    0.9643292688869347,
    -0.6710734268145182,
];
const GOLDEN_PCA_OJA: &[f64] = &[
    3.0407400810474576,
    0.9162070213751425,
    0.35588819104155783,
    3.519191553423375,
    0.9638677378816289,
    2.0469372985484084,
    3.954591836341751,
    0.9755719093513584,
    -0.6711043772119846,
];
const GOLDEN_EWM_MOMENTS: &[f64] = &[
    2.8329964637774814,
    16.02475808583336,
    1342.4260623665018,
    0.19749780915523601,
    0.45361175689062017,
    0.08653074498767489,
    0.6227033929036353,
    4.424843520682192,
    1088.4487593316915,
    0.15013223770551765,
    0.0528104410605394,
    0.1039648761186183,
    75.16866030031775,
    2.3011944298525524,
    11.644114135119917,
    941.9942065251553,
    0.08280406423595192,
    0.5782095311811001,
    0.20838520778454261,
    0.46604866892042085,
    2.9573294573236737,
    643.2681080958494,
    0.13376235812053172,
    0.054581904025623545,
    0.07441600775973262,
    86.00883395894138,
    1.839216837975061,
    8.407160392465022,
    397.6953053029218,
    0.33907848144993424,
    0.7133314202793511,
    0.393455614799461,
    0.5006575162868554,
    3.7577927563274716,
    378.69849162898095,
    0.18538010107369973,
    0.05306375077557374,
    0.05573958304140239,
    120.41184145309974,
];
const GOLDEN_EWM_MOMENTS_WINDOW: &[f64] = &[
    0.9302446287947042,
    3.4345051986135253,
    9.71287769945026,
    0.16988343024479924,
    0.5582423507347112,
    0.13058435201476315,
    0.19183122137139663,
    0.6066857489249887,
    16.83390768274169,
    0.17041953387867434,
    0.08487785071546758,
    0.14374730188285947,
    42.84779336144839,
    1.9567292633956257,
    9.476294398981619,
    865.7713801823895,
    -0.12223676107434481,
    0.8094345047851008,
    0.23519325716714704,
    0.5372767949758197,
    2.0912773835351843,
    786.3941580315961,
    0.3167673565461328,
    0.1270772046877625,
    0.12535619396484723,
    35.608016459273706,
    1.4778260868020379,
    4.575808230696152,
    98.92582037975185,
    0.3087690477889771,
    0.8932579053788221,
    0.4522764808837566,
    0.27975225011452476,
    1.1735985002288558,
    51.75962935810329,
    0.39093203611902083,
    0.12480108077550034,
    0.10613107986933701,
    46.13394192468252,
];
const GOLDEN_GLM: &[f64] = &[0.8190287945803094, 1.5909174696403168, 1.3226756022889994];
const GOLDEN_GLM_CAP: &[f64] = &[0.8059966005129862, 2.2639186345702202, 1.0964813502655293];
const GOLDEN_EXPECTILE: &[f64] = &[
    0.28722805885299973,
    2.2382733779673214,
    -0.02537400516394961,
    0.29742742033317526,
    1.5009295142639878,
    -0.7627708885306641,
];
const GOLDEN_EXPECTILE_WINDOW: &[f64] = &[
    0.17140744111304002,
    2.1597812960535903,
    -0.09429068171192226,
    0.214942275621624,
    1.5275050476441234,
    -0.7809933429467285,
];
const GOLDEN_HAWKES: &[f64] = &[
    0.5698035355306798,
    0.3189638413584075,
    0.4443771726681645,
    0.4067008349412284,
    0.17274161174942407,
    -0.25623622092507065,
];
const GOLDEN_HAWKES_WINDOW: &[f64] = &[
    0.6869138969117743,
    0.28538342803512085,
    0.4593991977177034,
    0.5920678320322507,
    -0.30485849466826326,
    -0.18213030070959965,
];
// Frozen 2026-10-09 (docs/PLAN.md task 223 (b)); the moments, the lag
// pairs and the median are held to their definitions in `audit/tests.rs`.
const GOLDEN_AUDIT: &[f64] = &[
    0.007351359111987893,
    0.6172135907442536,
    -0.21888805190556487,
    -1.297830470011374,
    0.10337402765025028,
    0.5182677744840106,
    1.4146505816120811,
    -0.10763359428819168,
    -8.230048359338621,
    0.002080856123662307,
    -0.11940372649057979,
    2.202151259526913,
    3311.0,
];
const GOLDEN_BOCPD: &[f64] = &[
    0.04617202802926437,
    0.05977980779859338,
    0.04295639028471272,
];
// Frozen 2026-10-06 (task 179), after a longhand of the clock form in plain
// probabilities -- weighted runs two-pass, the step's chance moved to the
// empty run before each row, pruned at 1e-8 -- gave the same three numbers
// to 4e-14.
const GOLDEN_BOCPD_ON_THE_CLOCK: &[f64] = &[
    0.01304148203335747,
    0.026769829077477554,
    0.009878006501232724,
];
// Re-frozen 2026-09-19 for the review's S3: the delta-method gradient of `ρ`
// carried the wrong powers of `σ_x` and `σ_y`, so `D̂` and every `Q` moved
// where the two columns' variances differ, as this stream's do. The
// finite-difference check in `corrchange.rs` is what the new numbers rest on.
// Regenerated 2026-09-28 (docs/PLAN.md task 114): the long-run variance's
// kernel is WKD's `1 − l/γ`, where it was Newey–West's `1 − l/(γ+1)`
// (0.8418929529846794, 0.7933802263147072, 0.787701333637405).
const GOLDEN_CORRCHANGE: &[f64] = &[0.6469993250907099, 0.6637003325160533, 0.5067551764605898];
// Re-frozen 2026-09-06: `hmm`'s `min_weight` used to withhold a row from
// the *update* as well as from the report, so the first rows of this stream
// (`min_weight = 3`) never reached the filter. It now gates the report
// alone, as it does in every other model (docs/REVIEW-E54-E64.md H1).
// Re-pinned 2026-10-08 (docs/PLAN.md task 214): a row reads the transition
// matrix from counts its clock has aged, `(λA + τ)/Σ`, where it read `(A +
// τ)/Σ`, so a zero-weight row is clock alone (hard rule 9). A replica written
// from the module docs gives these to 2.5e-16, and the old ones, under the old
// reading, to 1.2e-16 (-1.1511065244639265, -2.6218190023816343,
// -0.875789289803333).
// Re-pinned 2026-11-20 (schema 70, rule 5's twenty-second exception): the
// states' covariances forget toward a stabilized prior worth `d` rows, `Σ =
// (W·M + d·diag(ψ))/(W + d)`, where they kept a decaying ridge; `hmm`'s
// `tests_stabilized` holds that recursion against one carried in 512 bits,
// and `tests/test_second_opinion.py` against scikit-learn's QDA at no decay.
// Before: -1.1509510476558056, -2.6217067703865613, -0.8756632595908331.
const GOLDEN_HMM: &[f64] = &[
    -1.1938166867425974,
    -2.6918303565613972,
    -0.8942115051164505,
];
const GOLDEN_RCOV: &[f64] = &[15.118271471980519, -2.2219191583655915, 22.721761773534745];
// Re-pinned 2026-10-05 (docs/PLAN.md task 158): the stream runs `preavg_rows
// = 6` at the default `theta`, and the bias term now reads θ from that window,
// `k_n/√n`, as CKP's Eq. 7 defines it, where it read the configured `theta`.
// Re-pinned 2026-10-05 (task 159, D1): the pre-averaged estimate now forms
// CKP's first term, `Ȳ₀`, so a block has `n − k_n + 2` terms and the scale
// counts the terms summed; the entries moved by 4e-4 to 3e-3.
// Re-pinned 2026-10-06 (review CE1): the estimate is rescaled by `1/(1 −
// ψ₁/(2ψ₂k_n²))`, CKP's footnote 1, which at `k_n = 6` is exactly 19/16
// (`ψ₁ = 1`, `ψ₂ = 19/216`); each entry is the last pin times 19/16, to an ulp
// (7.978660961271573, -1.6354544255766184, 20.275069787152503).
const GOLDEN_RCOV_PREAVG: &[f64] = &[9.474659891509994, -1.9421021303722343, 24.0766453722436];
// `deco` and `deco_loglik` re-pinned 2026-11-22 (PLAN 551): `rho` is the
// ratio of the EW means of the row's numerator and denominator, and the
// linear form their averaged DCC recursion, learning once the standardiser's
// Kish size passes 5, where it was the EW mean of `u` and the recursion on
// it; `loglik` moves with the level it is read at. The pins were
// (-0.05328065158114557, -0.10464551302436545, -0.036211528292942816) and
// (-2.2831901919538913, -3.7586606055778677, -1.896095909765855).
// `deco.rs`'s `the_ratio_level_is_its_definition` holds the recursion to
// its longhand sums.
const GOLDEN_DECO: &[f64] = &[
    0.03481957260969185,
    -0.18599722774742422,
    -0.05901327009183074,
];
const GOLDEN_DECO_LOGLIK: &[f64] = &[-2.2843768890798684, -3.654894485613948, -1.897177036587538];
// `ewridge`, `ewridge_std` and `lasso` re-frozen 2026-09-14 (docs/PLAN.md
// task 81): the target is null on row 31, and under `target_gaps =
// "own_rows"` its Gram no longer learns that row's features, so rows 45 and
// 59 moved; row 20, before it, moved in the last bits only (`lasso` keeps
// its cross-moments centred now, N2).
const GOLDEN_EW_RIDGE: &[f64] = &[0.23958810892448573, 2.20363868480897, -0.06755913936964057];
const GOLDEN_EW_RIDGE_WINDOWED: &[f64] =
    &[0.23958810892448573, 2.192008777608849, -0.07602329017942382];
const GOLDEN_EW_RIDGE_STD: &[f64] = &[
    0.24074332641726603,
    2.1866449169182474,
    -0.06533729995816817,
];
// The three through-origin signatures, frozen 2026-09-19 (the review's T1)
// on the build with B2 fixed: `ewridge_origin_std` reads the prior.
const GOLDEN_EW_RIDGE_ORIGIN_STD: &[f64] = &[
    -0.06496193224777927,
    1.900894045241888,
    -0.31464368574465446,
];
const GOLDEN_HUBER_ORIGIN_STD: &[f64] = &[
    -0.06606160896249036,
    1.9278207732957797,
    -0.3195028239242011,
];
// The three `kalman` signatures, regenerated 2026-10-06 (review 2026-10-05,
// CC4): before a target's first residual its noise is the row's innovation
// squared, where it was the literal 1, and its prior variance is `p0` times
// that first noise, where it was `p0` in the target's units. Each matches
// `tests/reference.py`'s `kalman_ref`, with the same rules, on this stream to
// 1.3e-14; the old values matched the old `kalman_ref` to 6.3e-16.
//
// Every `kalman` signature again for task 211 (2026-10-08): the process noise
// is charged once on each row that observes the target, for the whole clock
// since the last, `Q D²`, so the null target on row 31 takes its clock into
// row 32's charge (`Q 2²`, where two rows charged `Q 1²` each); and a
// standardizing filter sizes each coefficient's prior once its feature's
// scale is usable, from the mean squared innovation over three rows, and
// follows the moments from its first row, with no warm-up. Rows 45 and 59
// moved for all five; row 20, before the null, for the standardizing two
// alone. Before: `kalman_plain` 2.226426987212936 and -0.05944106082324331,
// `kalman_q` 2.1635957406955013 and -0.08050613330500465, `kalman_obs_var`
// 2.2123547676208344 and -0.06081942698886093 (rows 45 and 59). The new
// values match `tests/reference.py`'s `kalman_ref`, rewritten from the module
// doc with the state re-mapped on every row, to 1.8e-15 (the scratchpad's
// `replica_rust_pins.py`).
const GOLDEN_KALMAN_PLAIN: &[f64] = &[0.2534302289666238, 2.2264285943696005, -0.0594422280502202];
const GOLDEN_RLS: &[f64] = &[
    0.24355619170018697,
    2.1844587322364037,
    -0.06708586579330882,
];
// `kalman` and `kalman_revert` again for task 206 (2026-10-08): past the
// standardizer's warm-up, 22 rows by Kish's count of the weights, `b` and `P`
// follow every move of the moments (review round 5, G1). Row 20 is inside the
// warm-up and kept its bits; rows 45 and 59 were 1.9517466851899055 and
// 0.026207278510925018 (`kalman`), 1.153918160782199 and 0.028098479936384025
// (`kalman_revert`). `tests/reference.py`'s `kalman_ref`, with the warm-up and
// the change of coordinates by matrix products, gives the new ones to 1.4e-14.
// Task 211 (above): `kalman` was 0.08051708612823805, 1.9812568625118254 and
// -0.06615560201475972, `kalman_revert` 0.3098186122738434, 1.1519113095475324
// and 0.009601088217273672. `kalman_revert` again for task 214 (2026-10-08):
// a reverting slot's process noise for a gap `D` is `q ((1 − 2^(−D/r)) /
// θ)²`, bounded, where it was `q D²`; row 40's gap of 25 is three of slope
// 2's half-lives of 8. `kalman_ref`, rewritten from the module doc, gives
// the new ones to 8.4e-16, and the old ones, 0.4810076582323497,
// 1.2730031757113542 and 0.13848333209268554, under the old charge to 1.8e-15.
// Both again for task 214's prior: `R` is the weighted median of the first
// three rows' squared innovations over the median of a χ²₁, where it was
// the mean over the rows so far. `kalman_ref` with the median gives the new
// ones to 1.0e-16; they were 0.22526048935963278, 2.2330040098824835 and
// -0.06612869061558382 (`kalman`), and 0.4810352559516703,
// 1.1575926629367428 and 0.16244017811259087 (`kalman_revert`).
const GOLDEN_KALMAN: &[f64] = &[
    0.22287716866303497,
    2.2273943734190746,
    -0.06712508541180909,
];
const GOLDEN_KALMAN_REVERT: &[f64] =
    &[0.46012636674356183, 1.1589140098081587, 0.11903305981915821];
const GOLDEN_LASSO: &[f64] = &[
    0.25359037757905656,
    2.1511829817060866,
    -0.06380197762074491,
];
const GOLDEN_HUBER: &[f64] = &[0.2431909931355622, 2.2046451423295514, -0.06491242691901175];
// Regenerated 2026-09-26 (review, G2): the quantile nudge is bounded by the
// row's leverage, which binds on this stream's early rows; the three values
// moved by 1.4e-3, 1.3e-3 and 4e-4 of themselves, and the QuantReg oracles
// in `robust.rs` hold. Regenerated again 2026-10-05 (review, TC1b): the
// leverage is the row's full one against the band system, where it was the
// Gram's diagonal; the early rows' band Gram is not diagonal, so the bound
// they bind moved, and the three values by 5.1e-4, 4.6e-4 and 1.5e-3 of
// themselves. The QuantReg oracles hold.
const GOLDEN_QUANTILE: &[f64] = &[0.25697145681782263, 2.226414143382377, -0.0210007825524009];
// Regenerated for the code review's C24 (2026-09-15), this and the next:
// under a half-life the proximal term is a decayed sum of its own, where `n`
// was decayed inside its square root and every coefficient shrank.
// Both moved on 2026-09-29 (docs/PLAN.md task 115 (d)): under a half-life the
// penalties now age with the sums, so the fit no longer shrinks between rows.
// Both moved again on 2026-10-08 (task 215), at rows 45 and 59, after the
// null target at row 31: the penalties keep their size, and the first row to
// teach after it meets the sums aged by two rows (-0.093% and +0.032% for the
// logistic loss, -0.17% and +0.18% for the squared). `ftrl_ref`, rewritten
// from the module docs, gives them to 1.2e-15 (and the old ones, under the
// old scale, to 1.6e-15).
const GOLDEN_FTRL_SQUARED: &[f64] = &[
    0.31755217793601365,
    1.8418686598361362,
    -0.059195743763157324,
];
const GOLDEN_FTRL: &[f64] = &[0.4937166166955374, 0.5893822512193234, 0.45266254275109397];
// Regenerated for docs/PLAN.md task 74 (2026-09-08): `standardize`
// standardises against the moments with the row admitted, so every
// prediction of this scaled fit moved. Again for task 195 (2026-10-07):
// `huber_delta` is in units of the residual's EW std (U1), so the cut of 0.5
// is 0.5·s where it was 0.5 of the target; a replica of the docstring's
// recursion -- the scaler with the row admitted, `s²` the EW mean of the
// squared out-of-sample residuals -- gave those three to 1e-15. Again for
// task 206 (2026-10-08): past the scaler's warm-up, 22 rows by Kish's count of
// the weights, the coefficients are held in the caller's units and each step
// mapped into them by its own row's scaler (review round 5, G1). Row 20 is
// inside the warm-up and kept its bits; rows 45 and 59 were
// 0.8740905120546256 and -0.026418737010923815. `tests/reference_paths.py`'s
// `sgd_ref` under `standardize`, the moments by their definition, gives the
// new ones to 1.3e-17. Again for task 270 (h)-B (2026-10-24): `inv_scaling`'s
// rate reads the weight decayed to the row, `lam·W`, where it read `W`, so a
// zero-weight row in between moves nothing; the three were
// -9.140859901712872e-5, 0.9626546015542387 and -0.09404195296589056, and
// `sgd_ref` with the new rate gives the new ones to 1.2e-16. Again for review
// C1 (2026-11-20): the scaler reads a row in the stream's own unit of weight,
// `Σw²/Σw` of the warm-up's rows (here 1.147 at the switch), while it warms
// up and `min` of it and the row's weight once the fit is held, where it read
// every row at 1, and keeps that weight of history into a row; this stream's
// weights are 0.5, 1 and 1.5, so every row from the second moved. The three
// were -0.038104258384859244, 0.9538283159665557 and -0.1118003974730884;
// `sgd_ref` with the new read gives the new ones to 2.3e-16.
const GOLDEN_SGD: &[f64] = &[
    -0.042084949526490584,
    0.9624471778471987,
    -0.11215033114181669,
];
const GOLDEN_SGD_SQUARED: &[f64] = &[
    0.31727038792368356,
    1.429415537069254,
    -0.007974524370910653,
];
// Regenerated for task 195 (2026-10-07): `eps` is in units of the residual's
// EW std (U1), the tube 0.05·σ where it was 0.05 of the target. Again for
// task 202 (the same day): `eps` is in units of the target's own EW std, the
// tube 0.05·σ_y. `tests/reference_paths.py::pa_ref`, written from the
// docstring with `σ_y²` the EW variance of `y` from its definition, gives
// these to the bit on this stream, and the same recursion with the box and
// the sum projected gives `GOLDEN_PA_BOX`'s to 8e-17.
const GOLDEN_PA: &[f64] = &[0.3512828551618081, 2.136865004465535, -0.062000906653408974];
const GOLDEN_SGD_SIMPLEX: &[f64] = &[0.5169094734826561, 1.109247996359838, -0.020146022626397198];
const GOLDEN_PA_BOX: &[f64] = &[0.5262123538733023, 1.6796740252199436, 0.1566262229378625];
// Re-frozen 2026-09-13: a null target is transparent to `holt` now -- the
// next observed row forecasts over the clock since the last one -- where the
// level stood still across it (review 2026-09-12, C22). Row 20 comes before
// the stream's null at row 31 and did not move.
// Regenerated for the code review's S29/S30 (2026-09-15): level and trend
// are weighted means, whose gains start at 1 where the textbook's were fixed.
// Regenerated for task 278 (D-H, 2026-10-26): a row takes the larger of the
// weighted line fit's gains and the textbook's, so the warm-up is the line
// fit, where the weighted means pinned the trend to the first observation.
const GOLDEN_AR: &[f64] = &[
    0.20651737786601188,
    2.183823230103924,
    -0.061890170712155385,
    0.2507806385801581,
    0.0023940696647469463,
    0.006207809809043063,
    1.5217126158114949,
    -0.7726238396515506,
];
const GOLDEN_SPECTRAL: &[f64] = &[
    0.6649015871575261,
    0.09516153180377941,
    0.9749530137841262,
    0.0,
    0.9737255863944811,
    2.9174867193629694,
];
const GOLDEN_HOLT: &[f64] = &[1.510543784944197, 0.05346222911849938, 0.1063893470704486];
const GOLDEN_SEQTEST: &[f64] = &[-0.6443570163905132, -1.086393303225433, -0.9577414997686987];
const GOLDEN_SEQTEST_NEG: &[f64] = &[
    -0.9917118216489557,
    -0.9917118216489557,
    -0.9917118216489557,
];
const GOLDEN_EW_COV: &[f64] = &[
    -0.3469363807058677,
    -0.1331528573613194,
    -0.013968574548668105,
];
const GOLDEN_MARGINAL: &[f64] = &[
    -0.658828097450141,
    -1.2763707481988686,
    16.974118022314695,
    -0.575404661163831,
    -1.0280783998000982,
    21.444816431035854,
    -0.4441310076916231,
    -0.7754079187908046,
    26.59144641744387,
];
const GOLDEN_KMEANS: &[f64] = &[0.5834101526098997, 0.8292779994915372, 0.923506490724854];
const GOLDEN_KMEANS_CLUSTER: &[f64] = &[2.0, 0.0, 2.0];
const GOLDEN_KMEANS_FIRST: &[f64] = &[1.8013837727258295, 2.935845007414875, 1.351429916256865];
const GOLDEN_MICRO: &[f64] = &[0.3004267420269594, 0.8769688913202295, 1.4153400286799591];
const GOLDEN_MICRO_CLUSTER: &[f64] = &[1.0, 6.0, 6.0];
const GOLDEN_MICRO_ID: &[f64] = &[0.0, 3.0, 0.0];
const GOLDEN_EW_CLASS: &[f64] = &[
    0.7905217777831283,
    2.3962695799139553e-7,
    0.8870452595331249,
];
const GOLDEN_EW_CLASS_CLASS: &[f64] = &[0.0, 1.0, 0.0];
const GOLDEN_EW_CLASS_SHARED: &[f64] =
    &[0.10145341892546339, 0.9999995909585125, 0.22525421810089402];
const GOLDEN_EW_CLASS_DIAGONAL: &[f64] =
    &[0.9347724076192444, 0.004084196343630897, 0.8672014742687666];
const GOLDEN_BOCPD_GAUSSIAN: &[f64] = &[
    0.056219033365587236,
    0.05468567337937875,
    0.050622861481265845,
];
const GOLDEN_BOCPD_ROBUST: &[f64] = &[0.05647515197223256, 0.05942243315748, 0.05285901703185968];
const GOLDEN_CORRCHANGE_SEQUENTIAL: &[f64] = &[
    0.4521205301189889,
    0.8383825456783482,
    0.9650674372722438,
    0.68881947719374,
    0.7687384238767587,
    0.8665169652644557,
    0.2804127989270904,
    0.06733939091874662,
    0.18234031021031075,
    0.26903161489029487,
    0.4527772374571688,
    0.47343163795422794,
    0.4672185819282127,
    0.724875551702304,
    0.7923589035682767,
    1.072890953651047,
    0.7050611338478604,
];
const GOLDEN_EW_CLASS_WINDOWED: &[f64] =
    &[0.7905217777831282, 0.999999973946462, 0.8824779940832221];
const GOLDEN_EW_COV_LAGS: &[f64] = &[
    0.42675674146655357,
    -0.2941126260468523,
    -0.23312910489114877,
];
const GOLDEN_EW_COV_PCA: &[f64] = &[
    -0.31399435832348843,
    1.2149616979919209,
    -0.17449357703751237,
];
const GOLDEN_EW_COV_WINDOWED: &[f64] =
    &[-0.3469363807058677, 0.06020754802385295, 0.0899414616598929];
const GOLDEN_EW_RIDGE_BLOCKED: &[f64] =
    &[0.23958810892448573, 2.20363868480897, -0.07344251215567098];
const GOLDEN_EW_RIDGE_BY_WEIGHT: &[f64] =
    &[0.23728591098765225, 2.20363868480897, -0.06946714940555972];
const GOLDEN_EW_RIDGE_ON_THE_CLOCK: &[f64] =
    &[0.2424004251925553, 2.2241093236471907, -0.06755913936964053];
const GOLDEN_EW_RIDGE_PAIRWISE: &[f64] =
    &[0.23958810892448573, 2.1830039922943563, -0.0678454894530192];
const GOLDEN_EW_RIDGE_SESSION_SHRINK: &[f64] = &[
    0.23958810892448573,
    2.2036034970611253,
    -0.06773477106376563,
];
const GOLDEN_EW_RIDGE_SET_BOTH: &[f64] =
    &[0.23958810892448573, 2.20363868480897, -0.06755913936964053];
const GOLDEN_EW_RIDGE_SET_ONE: &[f64] =
    &[0.6226036377649254, 1.6401144565737151, -0.12394041898406416];
// Re-pinned 2026-11-20 as `GOLDEN_HMM` was (schema 70). Before:
// -1.0643298399245595, -1.8091334719187875, -1.0356264682715979.
const GOLDEN_HMM_TVTP: &[f64] = &[
    -1.1218784704483222,
    -1.9440536711070382,
    -1.0190379841570159,
];
// Both again for task 211 (`GOLDEN_KALMAN_PLAIN`'s note).
const GOLDEN_KALMAN_OBS_VAR: &[f64] =
    &[0.24438391241807816, 2.212385812063775, -0.06082198940989911];
const GOLDEN_KALMAN_Q: &[f64] = &[0.2624940862221372, 2.16361003592116, -0.08052213955730514];
const GOLDEN_KMEANS_FARTHEST: &[f64] =
    &[0.5834101526098997, 0.8292779994915372, 1.0040104663998375];
const GOLDEN_KMEANS_FLOOR: &[f64] = &[0.5834101526098997, 1.1754072514037952, 0.7869506375655098];
const GOLDEN_KMEANS_KMEANSPP: &[f64] =
    &[0.5834101526098997, 0.8292779994915372, 1.0040104663998377];
const GOLDEN_KMEANS_RAW: &[f64] = &[0.3137395740537139, 0.48742765492297646, 0.5466642991217668];
const GOLDEN_LASSO_SELECT: &[f64] = &[0.0, 0.0, 0.0, 0.0, 0.0, 0.02, 0.0, 0.0, 0.0, 0.0];
const GOLDEN_LASSO_WINDOWED: &[f64] = &[
    0.25359037757905656,
    2.0887627806236106,
    -0.07423130166085738,
];
const GOLDEN_MARGINAL_BINS: &[f64] = &[
    0.2835419144684436,
    -0.14863035120403767,
    0.8850728014130631,
    0.27125897159512596,
    -0.5516946583900888,
    0.37497034342253555,
    0.2335678702944847,
    -0.5516946583900888,
    0.21387162829564357,
];
const GOLDEN_MARGINAL_LAGS: &[f64] = &[
    16.268396176259806,
    -3.308052338372048,
    -0.043937176942903604,
    19.5919219948599,
    -2.9508446537440984,
    0.10989778139661917,
    26.18686633083888,
    -2.437874678623021,
    -0.012455077577307155,
];
const GOLDEN_MARGINAL_WINDOWED: &[f64] = &[
    -0.658828097450141,
    -1.2763707481988686,
    16.974118022314695,
    -0.4279581919553364,
    -0.6817858147667967,
    5.45319450047077,
    -0.3564749745323196,
    -0.6156300333085452,
    16.500104235927783,
];
const GOLDEN_MICRO_CLOCK: &[f64] = &[0.3004267420269594, 0.8769688913202294, 1.4507685900691734];
const GOLDEN_MICRO_FLOOR: &[f64] = &[0.3004267420269594, 1.161118351411367, 0.8126582237424175];
const GOLDEN_RCOV_PREAVG_PSD: &[f64] =
    &[11.307541539663788, -0.7389071190718174, 18.691806669414845];
const GOLDEN_GARCH: &[f64] = &[
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
];
const GOLDEN_GARCH_TARGETED: &[f64] = &[
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
];
const GOLDEN_GARCH_GJR_T: &[f64] = &[
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
];
const GOLDEN_GARCH_SKEWT: &[f64] = &[
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
];
const GOLDEN_GARCH_EGARCH: &[f64] = &[
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
];
const GOLDEN_GARCH_FIGARCH: &[f64] = &[
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
];
const GOLDEN_GARCH_REALISTIC: &[f64] = &[
    2.2399589474984163,
    -0.727664650290237,
    2.2219621790100286,
    11.154139420457081,
    2.2399589474984163,
    2.2399589474984163,
    1.6963910313546728,
    0.03311181645164273,
    1.7450322055127563,
    8.60531059604579,
    1.6963910313546728,
    1.6963910313546728,
    1.6821436824777916,
    1.3101133530662716,
    1.7283180776263207,
    8.527892739666806,
    1.6821436824777916,
    1.6821436824777916,
    0.06204666892870737,
    0.06647116055943113,
    0.9038518417965252,
];
const GOLDEN_GARCH_TARGETED_REALISTIC: &[f64] = &[
    0.9949557097392965,
    0.6697869730267316,
    0.9869391845608312,
    4.954533901106475,
    0.9949557097392965,
    0.9949557097392965,
    0.5845801290178484,
    1.4824140245744857,
    0.6076356166942833,
    2.981118394500516,
    0.5845801290178484,
    0.5845801290178484,
    1.5989122167611232,
    0.1674656286367991,
    1.546284265907806,
    7.861694389456653,
    1.5989122167611232,
    1.5989122167611232,
    0.017697936123760687,
    0.07901947063002264,
    0.9014151625116107,
];
const GOLDEN_GARCH_GJR_REALISTIC: &[f64] = &[
    0.6957581299499292,
    -0.7765016170142898,
    0.7440189323922475,
    3.6015736076780795,
    0.6957581299499292,
    0.6957581299499292,
    1.6647155976298975,
    -0.34741537159647373,
    1.5891310258171312,
    8.13135232341462,
    1.6647155976298975,
    1.6647155976298975,
    1.329196214346552,
    0.5278220056374964,
    1.2914196310249186,
    6.549780223959964,
    1.329196214346552,
    1.329196214346552,
    0.03867863132859658,
    0.07031250265564645,
    0.09628848609412849,
    0.8448664387362972,
];
const GOLDEN_GARCH_EGARCH_REALISTIC: &[f64] = &[
    0.5335048315756751,
    -0.0945140560276713,
    0.5793074546162525,
    2.7833688567644908,
    0.5335048315756751,
    0.5335048315756751,
    1.3526435148288876,
    -1.4297205226217766,
    1.3139557657490408,
    6.666480665062554,
    1.3526435148288876,
    1.3526435148288876,
    2.6581900669492162,
    0.44762947637609185,
    2.392526310737723,
    12.6109069037114,
    2.6581900669492162,
    2.6581900669492162,
    -0.005263731925161575,
    0.14292607726452722,
    -0.09394030945738703,
    0.9685660216164435,
];
const GOLDEN_GARCH_FIGARCH_REALISTIC: &[f64] = &[
    0.39857348364694856,
    -1.026908584366157,
    0.3999136067672373,
    2.01803673349308,
    0.39857348364694856,
    0.39857348364694856,
    0.428903421965956,
    -0.5086366140567313,
    0.44168157307532596,
    2.1772076422068976,
    0.428903421965956,
    0.428903421965956,
    0.24858950048201273,
    0.06590265675494203,
    0.29325100332608156,
    1.36228522242274,
    0.24858950048201273,
    0.24858950048201273,
    0.0323399409875803,
    0.2393473448668678,
    0.3265372342419699,
    0.4587686140715305,
];
const GOLDEN_GARCH_T_REALISTIC: &[f64] = &[
    2.2885496942968775,
    0.1603749979725027,
    2.297274104176434,
    11.464866689975453,
    2.2885496942968775,
    2.2885496942968775,
    1.0120613568512764,
    0.19068359152910785,
    1.156773710662873,
    5.427604918985226,
    1.0120613568512764,
    1.0120613568512764,
    2.6129661337355947,
    0.4844816887283076,
    2.56087583115259,
    12.932513414717125,
    2.6129661337355947,
    2.6129661337355947,
    0.06900346251738707,
    0.088405421257835,
    0.8798915769497312,
    7.3123446077751915,
];
const GOLDEN_GARCH_T_LOW_PERSISTENCE: &[f64] = &[
    0.7759270295334134,
    -0.4709311869580696,
    0.8559123344287154,
    4.089434673550422,
    0.7759270295334134,
    0.7759270295334134,
    0.6851240521286311,
    -0.5758610993529932,
    0.782619621277854,
    3.681110872448417,
    0.6851240521286311,
    0.6851240521286311,
    1.0028983182918356,
    -0.23741988745156573,
    0.9828288037937298,
    4.961950124338716,
    1.0028983182918356,
    1.0028983182918356,
    0.08466421653019822,
    0.10798964854289916,
    0.8016825284821371,
    6.774316512544334,
];
const GOLDEN_GARCH_SKEWT_REALISTIC: &[f64] = &[
    2.8167569753019492,
    -0.6190502808173738,
    2.6825102611870455,
    13.742825525638576,
    2.8167569753019492,
    2.8167569753019492,
    1.2468102828207452,
    -0.3600732663796168,
    1.2983459614253783,
    6.364812235775348,
    1.2468102828207452,
    1.2468102828207452,
    1.3185241083003951,
    -1.79375181690488,
    1.3637097264676832,
    6.706994262873854,
    1.3185241083003951,
    1.3185241083003951,
    0.04402310635217727,
    0.06254476813741976,
    0.9129694209409008,
    6.926672507282499,
    -0.20847350554841632,
];
const GOLDEN_GARCH_GJR_T_REALISTIC: &[f64] = &[
    1.2758366452078276,
    -0.9848155860088434,
    1.2192427614221337,
    6.2355746555305185,
    1.2758366452078276,
    1.2758366452078276,
    0.7389351386485398,
    0.9455510641880728,
    0.7401976431214408,
    3.697884285971264,
    0.7389351386485398,
    0.7389351386485398,
    0.3534741623503383,
    -3.315999459195701,
    0.4015018694337203,
    1.8894553358710924,
    0.3534741623503383,
    0.3534741623503383,
    0.024661551390733224,
    0.035612763820360764,
    0.12241777500356756,
    0.8699205611467933,
    7.114105652321648,
];

const GOLDEN_DCC: &[f64] = &[
    0.4235966473744679,
    0.6031760321276809,
    0.3027091519596326,
    -3.794013674519368,
    0.26708833514569535,
    0.486841844378243,
    0.38410018784932287,
    -2.8396861315938375,
    0.55330438795163,
    0.6788582582778027,
    0.627134753655765,
    -4.418521207829423,
    0.05184281030720079,
    0.9209945595510746,
];
const GOLDEN_DCC_ADCC: &[f64] = &[
    0.48458418860922786,
    0.5858299148208974,
    0.3271127311070238,
    -3.8573373987811443,
    0.25507303762307754,
    0.4323424899860381,
    0.3294783259697756,
    -2.8978143108253307,
    0.5952965930206311,
    0.6965144111810854,
    0.6582222454076307,
    -4.455481429051513,
    0.03117901673842428,
    0.9306037795258475,
    0.043167478090169166,
];
const GOLDEN_DCC_GIVEN: &[f64] = &[
    0.45993520774557745,
    0.6023139356915606,
    0.2741706822578032,
    -3.8370129133266566,
    0.27395386337185373,
    0.46173822016843463,
    0.3474341578277121,
    -2.8728544091338266,
    0.5769915893060064,
    0.6921182452910445,
    0.6510327833777574,
    -4.424814565121372,
    0.04,
    0.92,
    0.03,
];
const GOLDEN_DCC_MARGINS: &[f64] = &[
    1.632449472378928,
    1.6130389388363788,
    3.119290393504543,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    f64::NAN,
    1.1868143509543494,
    1.642863259435476,
    0.9385254085000607,
    0.28248638780969604,
    0.5110106953981232,
    0.32446372320473277,
    -2.847062701683519,
    2.2035151422767525,
    1.9748154475166262,
    4.908019459058437,
    0.5687575535034975,
    0.6990426621259763,
    0.6317942624649461,
    -4.373001758543537,
    0.06374974794547447,
    0.90384097205027,
];
const GOLDEN_DCC_DECAYED: &[f64] = &[
    0.46951936270545763,
    0.594562666839342,
    0.3230074798425693,
    -3.8361085230644445,
    0.26318076111417427,
    0.44977193183375097,
    0.3211419292378255,
    -2.8939674916306273,
    0.5966014976041538,
    0.6977714869710139,
    0.6585497263261488,
    -4.457327754020744,
    0.037419982718833344,
    0.9226257409287546,
    0.038106455446553146,
];
const GOLDEN_CORRCHANGE_MONITOR_REALISTIC: &[f64] = &[
    0.5175388423443728,
    1.299838923528649,
    0.0,
    0.927267384651194,
    1.299838923528649,
    0.0,
    1.06354656443545,
    1.299838923528649,
    0.0,
    1.5776864934719674,
    1.299838923528649,
    1.0,
    52.0,
    0.6103982837661497,
    1.299838923528649,
    0.0,
    0.6761447573653405,
    1.299838923528649,
    0.0,
];
const GOLDEN_CORRCHANGE_SEQUENTIAL_REALISTIC: &[f64] = &[
    0.37631039097771973,
    2.0995546966895327,
    0.0,
    0.4782010402389974,
    2.0995546966895327,
    0.0,
    0.4491729726949962,
    2.0995546966895327,
    0.0,
    0.40169088520549484,
    2.0995546966895327,
    0.0,
    0.473972288387234,
    2.0995546966895327,
    0.0,
    0.97477991128773,
    2.0995546966895327,
    0.0,
    0.7348085921815097,
    2.0995546966895327,
    0.0,
    0.5365627330553566,
    2.0995546966895327,
    0.0,
    0.5482585076980296,
    2.0995546966895327,
    0.0,
    0.367721907450489,
    2.0995546966895327,
    0.0,
    0.4538435090711439,
    2.0995546966895327,
    0.0,
    0.5465345297053715,
    2.0995546966895327,
    0.0,
    0.7894119394409368,
    2.0995546966895327,
    0.0,
    0.7252163569141724,
    2.0995546966895327,
    0.0,
    0.6832125290826009,
    2.0995546966895327,
    0.0,
    0.41953128255309907,
    2.0995546966895327,
    0.0,
    0.5009295389620827,
    2.0995546966895327,
    0.0,
    0.4649433565572973,
    2.0995546966895327,
    0.0,
    0.4696218632474659,
    2.0995546966895327,
    0.0,
    0.47944492640782804,
    2.0995546966895327,
    0.0,
    0.44802327061472064,
    2.0995546966895327,
    0.0,
    0.4706166839446203,
    2.0995546966895327,
    0.0,
    0.4731826338416649,
    2.0995546966895327,
    0.0,
    0.5395472514891563,
    2.0995546966895327,
    0.0,
    0.5710188538052846,
    2.0995546966895327,
    0.0,
    0.5751961799553456,
    2.0995546966895327,
    0.0,
    0.7259658893574313,
    2.0995546966895327,
    0.0,
    0.7689979661502525,
    2.0995546966895327,
    0.0,
    0.7977296039145975,
    2.0995546966895327,
    0.0,
    0.800133750323979,
    2.0995546966895327,
    0.0,
    0.8509229825430418,
    2.0995546966895327,
    0.0,
    0.9184200714691402,
    2.0995546966895327,
    0.0,
    0.937465387568265,
    2.0995546966895327,
    0.0,
    0.956457485164983,
    2.0995546966895327,
    0.0,
    0.6080449396876201,
    2.0995546966895327,
    0.0,
    0.9587720827565909,
    2.0995546966895327,
    0.0,
    0.9991292999057336,
    2.0995546966895327,
    0.0,
    1.069286855411994,
    2.0995546966895327,
    0.0,
    1.0736097943039378,
    2.0995546966895327,
    0.0,
    1.083363925783781,
    2.0995546966895327,
    0.0,
    1.0802696450399234,
    2.0995546966895327,
    0.0,
    1.109967396481821,
    2.0995546966895327,
    0.0,
    0.7424179099794448,
    2.0995546966895327,
    0.0,
    0.794525240428222,
    2.0995546966895327,
    0.0,
    0.809790628510859,
    2.0995546966895327,
    0.0,
    0.8370007317728826,
    2.0995546966895327,
    0.0,
    0.7690802356128759,
    2.0995546966895327,
    0.0,
    0.6915890438618476,
    2.0995546966895327,
    0.0,
    0.7046928585978578,
    2.0995546966895327,
    0.0,
    0.6600623147719836,
    2.0995546966895327,
    0.0,
    0.6474831301685352,
    2.0995546966895327,
    0.0,
    0.5339542969454338,
    2.0995546966895327,
    0.0,
    0.5889332813627571,
    2.0995546966895327,
    0.0,
    0.6098465297162793,
    2.0995546966895327,
    0.0,
    0.7386210246517276,
    2.0995546966895327,
    0.0,
    0.7154636106460116,
    2.0995546966895327,
    0.0,
    0.7326985542937365,
    2.0995546966895327,
    0.0,
    0.6709952486152306,
    2.0995546966895327,
    0.0,
    0.684121006428644,
    2.0995546966895327,
    0.0,
    0.6373723535648865,
    2.0995546966895327,
    0.0,
    0.6736185146833589,
    2.0995546966895327,
    0.0,
    0.5573186846611754,
    2.0995546966895327,
    0.0,
    0.5492049324664255,
    2.0995546966895327,
    0.0,
    0.47579892301476406,
    2.0995546966895327,
    0.0,
    0.45339776910979906,
    2.0995546966895327,
    0.0,
    0.48394859736528756,
    2.0995546966895327,
    0.0,
    0.4939408803299518,
    2.0995546966895327,
    0.0,
    0.4614387637360561,
    2.0995546966895327,
    0.0,
    0.46900248924473,
    2.0995546966895327,
    0.0,
    0.48567710135601255,
    2.0995546966895327,
    0.0,
    0.4293781154943662,
    2.0995546966895327,
    0.0,
    0.6971339507035714,
    2.0995546966895327,
    0.0,
    0.7447836257307305,
    2.0995546966895327,
    0.0,
    0.8238238814492621,
    2.0995546966895327,
    0.0,
    0.8630956781954591,
    2.0995546966895327,
    0.0,
    0.9263294137704036,
    2.0995546966895327,
    0.0,
    0.839470557157698,
    2.0995546966895327,
    0.0,
    0.8132490007757339,
    2.0995546966895327,
    0.0,
    0.7987907216009417,
    2.0995546966895327,
    0.0,
    0.869902941965475,
    2.0995546966895327,
    0.0,
    0.8478397457098449,
    2.0995546966895327,
    0.0,
    0.7859631004847999,
    2.0995546966895327,
    0.0,
    0.8601756489014052,
    2.0995546966895327,
    0.0,
    0.8932371589493124,
    2.0995546966895327,
    0.0,
    0.9845695955623129,
    2.0995546966895327,
    0.0,
    1.011847431091807,
    2.0995546966895327,
    0.0,
    0.9293749668620732,
    2.0995546966895327,
    0.0,
    0.9593549600706618,
    2.0995546966895327,
    0.0,
    0.9573941769390072,
    2.0995546966895327,
    0.0,
    0.9262394711866417,
    2.0995546966895327,
    0.0,
    0.9527291961829675,
    2.0995546966895327,
    0.0,
    1.1939922534178478,
    2.0995546966895327,
    0.0,
    1.2895132774286842,
    2.0995546966895327,
    0.0,
    1.295831532816404,
    2.0995546966895327,
    0.0,
    1.2622619405248714,
    2.0995546966895327,
    0.0,
    1.3338082720383955,
    2.0995546966895327,
    0.0,
    1.3115764949174988,
    2.0995546966895327,
    0.0,
    1.2600926140374287,
    2.0995546966895327,
    0.0,
    1.2196523269608133,
    2.0995546966895327,
    0.0,
    1.0436744577900252,
    2.0995546966895327,
    0.0,
    1.1710972065379772,
    2.0995546966895327,
    0.0,
    1.185144111601391,
    2.0995546966895327,
    0.0,
    1.129884877977235,
    2.0995546966895327,
    0.0,
    1.0187320486090576,
    2.0995546966895327,
    0.0,
    0.9901721807078712,
    2.0995546966895327,
    0.0,
    0.9889062320581938,
    2.0995546966895327,
    0.0,
    1.10566273306427,
    2.0995546966895327,
    0.0,
    1.0912102956000003,
    2.0995546966895327,
    0.0,
    1.0813725786269217,
    2.0995546966895327,
    0.0,
    1.071095932431618,
    2.0995546966895327,
    0.0,
    1.1097910414859815,
    2.0995546966895327,
    0.0,
    1.112373000552525,
    2.0995546966895327,
    0.0,
    1.1018435937574975,
    2.0995546966895327,
    0.0,
    1.0996626823057374,
    2.0995546966895327,
    0.0,
    1.0285658675930787,
    2.0995546966895327,
    0.0,
    1.031326233534007,
    2.0995546966895327,
    0.0,
    1.0339352450339785,
    2.0995546966895327,
    0.0,
    1.0483064304946665,
    2.0995546966895327,
    0.0,
    1.0341415719495706,
    2.0995546966895327,
    0.0,
    0.9395644748016205,
    2.0995546966895327,
    0.0,
    0.9099678126305076,
    2.0995546966895327,
    0.0,
    0.9108825921560586,
    2.0995546966895327,
    0.0,
    0.9099395776520443,
    2.0995546966895327,
    0.0,
    0.9130063551707321,
    2.0995546966895327,
    0.0,
    0.9182592887143443,
    2.0995546966895327,
    0.0,
    0.8621885458850154,
    2.0995546966895327,
    0.0,
    0.9127422624661345,
    2.0995546966895327,
    0.0,
    0.9043604250433781,
    2.0995546966895327,
    0.0,
    0.8459465690732716,
    2.0995546966895327,
    0.0,
    0.8003760282852169,
    2.0995546966895327,
    0.0,
    0.8284646058555889,
    2.0995546966895327,
    0.0,
    0.855807243495259,
    2.0995546966895327,
    0.0,
    0.7514494526665831,
    2.0995546966895327,
    0.0,
    0.9329651131604942,
    2.0995546966895327,
    0.0,
    0.9350262993726116,
    2.0995546966895327,
    0.0,
    0.9373334375806442,
    2.0995546966895327,
    0.0,
    0.9238128225698414,
    2.0995546966895327,
    0.0,
    0.8803050051984886,
    2.0995546966895327,
    0.0,
    0.904539919683849,
    2.0995546966895327,
    0.0,
    0.9276932245972966,
    2.0995546966895327,
    0.0,
    0.9384139430342365,
    2.0995546966895327,
    0.0,
    0.7644920230729826,
    2.0995546966895327,
    0.0,
    0.7378409982445477,
    2.0995546966895327,
    0.0,
    0.7361921978689427,
    2.0995546966895327,
    0.0,
    0.7548400820025435,
    2.0995546966895327,
    0.0,
    0.7531698029740062,
    2.0995546966895327,
    0.0,
    0.7400006566400981,
    2.0995546966895327,
    0.0,
    0.31463847210500806,
    2.0995546966895327,
    0.0,
    0.5384064611014164,
    2.0995546966895327,
    0.0,
    0.6574309116248024,
    2.0995546966895327,
    0.0,
    0.8152577552348628,
    2.0995546966895327,
    0.0,
    0.5450657696539288,
    2.0995546966895327,
    0.0,
    0.6268093910106167,
    2.0995546966895327,
    0.0,
    0.775286608296337,
    2.0995546966895327,
    0.0,
    0.4124232520845501,
    2.0995546966895327,
    0.0,
    0.6986316343837219,
    2.0995546966895327,
    0.0,
    0.6309553398203781,
    2.0995546966895327,
    0.0,
    1.1109705143288073,
    2.0995546966895327,
    0.0,
    1.0633829669353443,
    2.0995546966895327,
    0.0,
    1.6199132497072055,
    2.0995546966895327,
    0.0,
    1.7643181995898702,
    2.0995546966895327,
    0.0,
    1.8541173694902753,
    2.0995546966895327,
    0.0,
    2.1125045052432205,
    2.0995546966895327,
    1.0,
    8.0,
    0.3267303528567436,
    2.0995546966895327,
    0.0,
    0.5954261482970906,
    2.0995546966895327,
    0.0,
    0.8307228365501749,
    2.0995546966895327,
    0.0,
    0.8220761946172391,
    2.0995546966895327,
    0.0,
    0.9615154739127793,
    2.0995546966895327,
    0.0,
    0.815625533086969,
    2.0995546966895327,
    0.0,
    0.8966541218717585,
    2.0995546966895327,
    0.0,
    0.6346179982231477,
    2.0995546966895327,
    0.0,
    0.6129221404792131,
    2.0995546966895327,
    0.0,
    0.7112889999346801,
    2.0995546966895327,
    0.0,
    0.6164786674229163,
    2.0995546966895327,
    0.0,
    0.6616099950232152,
    2.0995546966895327,
    0.0,
    0.5790925941828617,
    2.0995546966895327,
    0.0,
    0.6829065558212262,
    2.0995546966895327,
    0.0,
    0.1759818712707995,
    2.0995546966895327,
    0.0,
    0.14816933569462443,
    2.0995546966895327,
    0.0,
    0.16419517757642646,
    2.0995546966895327,
    0.0,
    0.41893407756984274,
    2.0995546966895327,
    0.0,
    0.33813490017224,
    2.0995546966895327,
    0.0,
    0.17881181469591526,
    2.0995546966895327,
    0.0,
    0.19423257054551643,
    2.0995546966895327,
    0.0,
    0.14201287106097876,
    2.0995546966895327,
    0.0,
    0.21222960419803785,
    2.0995546966895327,
    0.0,
    0.23373817615087528,
    2.0995546966895327,
    0.0,
    0.23201669506161762,
    2.0995546966895327,
    0.0,
    0.23618741641986465,
    2.0995546966895327,
    0.0,
    0.25845995468448646,
    2.0995546966895327,
    0.0,
    0.24255745420078836,
    2.0995546966895327,
    0.0,
    0.23208236911922595,
    2.0995546966895327,
    0.0,
    0.27488440821596705,
    2.0995546966895327,
    0.0,
    0.35730545513016654,
    2.0995546966895327,
    0.0,
    0.3598139193440761,
    2.0995546966895327,
    0.0,
    0.21226627503289844,
    2.0995546966895327,
    0.0,
    0.19879730382407615,
    2.0995546966895327,
    0.0,
    0.010341861049069504,
    2.0995546966895327,
    0.0,
    0.028657795814336567,
    2.0995546966895327,
    0.0,
    0.039662353955954355,
    2.0995546966895327,
    0.0,
    0.04953674609006061,
    2.0995546966895327,
    0.0,
    0.30391227555543954,
    2.0995546966895327,
    0.0,
    0.34151305424290485,
    2.0995546966895327,
    0.0,
    0.3474161576173229,
    2.0995546966895327,
    0.0,
    0.066811920550578,
    2.0995546966895327,
    0.0,
    0.09071569002508945,
    2.0995546966895327,
    0.0,
    0.15191284645594094,
    2.0995546966895327,
    0.0,
    0.11676253719235884,
    2.0995546966895327,
    0.0,
    0.14343965863717456,
    2.0995546966895327,
    0.0,
    0.07982109270711707,
    2.0995546966895327,
    0.0,
    0.04184846682579438,
    2.0995546966895327,
    0.0,
    0.16337486471049645,
    2.0995546966895327,
    0.0,
    0.1441566019987237,
    2.0995546966895327,
    0.0,
    0.14611177656927704,
    2.0995546966895327,
    0.0,
    0.15663811668718333,
    2.0995546966895327,
    0.0,
    0.3084411597751212,
    2.0995546966895327,
    0.0,
    0.20017544670610818,
    2.0995546966895327,
    0.0,
    0.29549012655902446,
    2.0995546966895327,
    0.0,
    0.4367249122864925,
    2.0995546966895327,
    0.0,
    0.2201316996361567,
    2.0995546966895327,
    0.0,
    0.20876560733865812,
    2.0995546966895327,
    0.0,
    0.29704293880331195,
    2.0995546966895327,
    0.0,
    0.3198452475926289,
    2.0995546966895327,
    0.0,
    0.300653090698907,
    2.0995546966895327,
    0.0,
    0.15005118074812054,
    2.0995546966895327,
    0.0,
    0.138228605357463,
    2.0995546966895327,
    0.0,
    0.01885542977162675,
    2.0995546966895327,
    0.0,
    0.046013610159281744,
    2.0995546966895327,
    0.0,
    0.02729663850565639,
    2.0995546966895327,
    0.0,
    0.04896989822127926,
    2.0995546966895327,
    0.0,
    0.058076514417387856,
    2.0995546966895327,
    0.0,
    0.3399896293419202,
    2.0995546966895327,
    0.0,
    0.3794735985845303,
    2.0995546966895327,
    0.0,
    0.3525585595712578,
    2.0995546966895327,
    0.0,
    0.19798737412961817,
    2.0995546966895327,
    0.0,
    0.1418508904064168,
    2.0995546966895327,
    0.0,
    0.14497924806066723,
    2.0995546966895327,
    0.0,
    0.12827775992300497,
    2.0995546966895327,
    0.0,
    0.019011120649123515,
    2.0995546966895327,
    0.0,
    0.030796229490342733,
    2.0995546966895327,
    0.0,
    0.13205109976981297,
    2.0995546966895327,
    0.0,
    0.09932947565361314,
    2.0995546966895327,
    0.0,
    0.09799815973371669,
    2.0995546966895327,
    0.0,
    0.1665230249058771,
    2.0995546966895327,
    0.0,
    0.09836387082167049,
    2.0995546966895327,
    0.0,
    0.10681279893381958,
    2.0995546966895327,
    0.0,
    0.05644153857158041,
    2.0995546966895327,
    0.0,
    0.30686997093624935,
    2.0995546966895327,
    0.0,
    0.19916066686317815,
    2.0995546966895327,
    0.0,
    0.20729783862368487,
    2.0995546966895327,
    0.0,
    0.29357824431765706,
    2.0995546966895327,
    0.0,
    0.2879864809410163,
    2.0995546966895327,
    0.0,
    0.2588865237777807,
    2.0995546966895327,
    0.0,
    0.20344181549764145,
    2.0995546966895327,
    0.0,
    0.2535083449713034,
    2.0995546966895327,
    0.0,
    0.25635180755234144,
    2.0995546966895327,
    0.0,
    0.3270110459142569,
    2.0995546966895327,
    0.0,
    0.348364719833074,
    2.0995546966895327,
    0.0,
    0.4337925744660372,
    2.0995546966895327,
    0.0,
    0.31216624429535983,
    2.0995546966895327,
    0.0,
    0.3210425947621258,
    2.0995546966895327,
    0.0,
    0.3657513689269985,
    2.0995546966895327,
    0.0,
    0.4751201073344292,
    2.0995546966895327,
    0.0,
    0.3974820346658131,
    2.0995546966895327,
    0.0,
    0.4240823996363771,
    2.0995546966895327,
    0.0,
    0.40116224076756296,
    2.0995546966895327,
    0.0,
    0.45597854502564034,
    2.0995546966895327,
    0.0,
    0.42321018010138534,
    2.0995546966895327,
    0.0,
    0.44743527661761134,
    2.0995546966895327,
    0.0,
    0.42414861825029837,
    2.0995546966895327,
    0.0,
    0.12896366985044055,
    2.0995546966895327,
    0.0,
    0.11076936900010477,
    2.0995546966895327,
    0.0,
    0.18791603785303002,
    2.0995546966895327,
    0.0,
    0.16928171411948167,
    2.0995546966895327,
    0.0,
    0.16581875026611434,
    2.0995546966895327,
    0.0,
    0.15930353959707605,
    2.0995546966895327,
    0.0,
    0.16341835362372686,
    2.0995546966895327,
    0.0,
    0.17096649570012057,
    2.0995546966895327,
    0.0,
    0.13202668439620868,
    2.0995546966895327,
    0.0,
    0.13736010831257037,
    2.0995546966895327,
    0.0,
    0.15215788087341356,
    2.0995546966895327,
    0.0,
    0.1490209149920598,
    2.0995546966895327,
    0.0,
    0.18243672711346387,
    2.0995546966895327,
    0.0,
    0.2558606195111961,
    2.0995546966895327,
    0.0,
    0.25318083227503024,
    2.0995546966895327,
    0.0,
    0.205573988971439,
    2.0995546966895327,
    0.0,
    0.1287735015252253,
    2.0995546966895327,
    0.0,
    0.14659216927747448,
    2.0995546966895327,
    0.0,
    0.21314344948015862,
    2.0995546966895327,
    0.0,
    0.0058195813945725984,
    2.0995546966895327,
    0.0,
    0.004094959914973394,
    2.0995546966895327,
    0.0,
];

const GOLDEN_POWER_VARIATION: &[f64] = &[
    20.301763088150093,
    14.524981482223351,
    152.54271228371493,
    9.302374921594874,
    11.002419458319203,
    29.464292253703377,
    20.847172608307673,
    14.276477863287328,
    16.071940011962074,
    10.459155766645733,
    10.642877740344339,
    97.82724355929126,
    7.025526342384912,
    2.778344739613954,
    -0.0030414029344230124,
    4.580167185467167,
    5.553668494365291,
    14.240623039921394,
    15.904128496397625,
    18.06313225979898,
    266.2749803565534,
    12.028522824449905,
    5.811766800865598,
    1.317485386565981,
    9.772659335199275,
    10.58332018400874,
    21.63821001394927,
];

const GOLDEN_POWER_VARIATION_WINDOW: &[f64] = &[
    23.20165621168584,
    13.363161652229811,
    146.95773760814996,
    8.459800999631693,
    8.726110417548874,
    36.34337751793883,
    25.735361329750464,
    14.509426922427343,
    10.285714285714286,
    12.833685109912663,
    13.473997818648956,
    188.8799869536873,
    8.860895137388496,
    2.047731521551962,
    -2.7628090527922424,
    4.898316695965254,
    6.419299085112919,
    10.285714285714286,
    11.7484856053016,
    13.10735007891979,
    131.79937669810045,
    8.662419548382553,
    5.47306623704143,
    1.7445983458686738,
    7.5394428256191475,
    5.381814193807589,
    10.285714285714286,
];

const GOLDEN_TDIGEST: &[f64] = &[
    -1.4506754991295818,
    -0.6153938389536291,
    1.9669305121855334,
    -1.4626009005582017,
    -0.9992829396489874,
    1.9874652286997394,
    -1.4889104278521395,
    -0.63596345124358,
    1.948147518472067,
    -1.5045266365995258,
    -1.0229911855131937,
    1.9616845856185046,
    -1.5012510796384302,
    -0.6234457757321644,
    1.9185853898662055,
    -1.5199855742701565,
    -1.0211641212789853,
    1.9367196420068518,
];

const GOLDEN_DDSKETCH: &[f64] = &[
    -1.476926144063773,
    -0.624961225664581,
    1.7331981299642243,
    -1.6254534993405247,
    -1.071237747864269,
    2.290251881689183,
    -1.537202693153204,
    -0.624961225664581,
    1.7331981299642243,
    -1.6592146905265275,
    -1.0996546789342858,
    2.27294307536278,
    -1.537202693153204,
    -0.600455344842521,
    1.7331981299642243,
    -1.6525632698860737,
    -1.0992880578758688,
    2.2435191601110698,
    -1.3099107407241846,
    -0.5220083628371914,
    1.6652362387784885,
    -1.4633920513902652,
    -0.9858309127376075,
    2.288711348987963,
    -1.6322612637531722,
    -0.7334027039711133,
    2.2932675214577865,
    -1.6840237592460194,
    -1.18135595786429,
    2.4401953309769513,
    -1.6322612637531722,
    -0.7334027039711133,
    1.4476800818050848,
    -1.6840237592460194,
    -1.274658337302112,
    1.8153649851088371,
    -1.537202693153204,
    -0.600455344842521,
    1.7331981299642243,
    -1.6570551861482388,
    -1.0879553281795669,
    2.2633900674796195,
    -1.537202693153204,
    -0.624961225664581,
    1.7331981299642243,
    -1.6614534427719094,
    -1.1197345743729152,
    2.28284665234257,
    -1.537202693153204,
    -0.7334027039711133,
    1.6652362387784885,
    -1.6626039126757866,
    -1.136873377715784,
    2.2829017763508332,
];

const GOLDEN_DDSKETCH_DECAYED: &[f64] = &[
    -1.3099107407241846,
    -0.7633344539546297,
    1.6652362387784885,
    -1.5388589569648967,
    -1.0710360328115065,
    2.315423616777208,
    26.867539807603332,
    -1.7331981299642243,
    -0.7334027039711133,
    1.7331981299642243,
    -1.7331981299642243,
    -1.2098232667812512,
    2.260432223073342,
    16.92073916656056,
    -1.6322612637531722,
    -0.600455344842521,
    1.4476800818050848,
    -1.7104392927357428,
    -1.1778803166775635,
    1.988277768204018,
    20.212303545600623,
];
