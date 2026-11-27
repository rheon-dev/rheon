//! Core unit & invariant tests for the regression variants:
//! - `ewridge` empirical-Bayes evidence maximization (`EvidenceCfg` + `pred_std`)
//! - `pa` and `sgd` decay-weighted Polyak-Ruppert averaging (`average`)
//! - `lasso` non-negative coefficients (`positive`, both `k < 40` and `k >= 40`)
//! - `huber` redescending norms (`tukey`, `andrews`, `hampel`, `ramsay`)
//! - `sgd` elastic-net (`l1`), Bottou's `LearningRate::Optimal`, and
//!   Karampatziakis-Langford `invariant` updates.

use rheon_core::*;

fn lcg(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

type TestRow = (Vec<f64>, Vec<Option<f64>>, f64, f64);

fn save_restore_identical<M: OnlineModel + Clone>(
    mut a: M,
    rows: &[TestRow],
    split: usize,
    label: &str,
) {
    let mut b = a.clone();
    for (x, y, d, w) in &rows[..split] {
        a.step(x, y, *d, *w);
        b.step(x, y, *d, *w);
    }
    let bytes = rmp_serde::to_vec_named(&b.state()).unwrap();
    let state: State = rmp_serde::from_slice(&bytes).unwrap();
    let mut c = M::restore(&state).unwrap();
    for (i, (x, y, d, w)) in rows[split..].iter().enumerate() {
        let sa = a.step(x, y, *d, *w);
        let sc = c.step(x, y, *d, *w);
        assert_eq!(
            sa.pred.len(),
            sc.pred.len(),
            "{label} row {i}: pred len differs"
        );
        for (pa, pc) in sa.pred.iter().zip(&sc.pred) {
            if pa.is_nan() {
                assert!(pc.is_nan(), "{label} row {i}: expected NaN, got {pc}");
            } else {
                assert_eq!(
                    pa.to_bits(),
                    pc.to_bits(),
                    "{label} row {i}: pred bit mismatch {pa} vs {pc}"
                );
            }
        }
        assert_eq!(
            sa.n_eff.to_bits(),
            sc.n_eff.to_bits(),
            "{label} row {i}: n_eff"
        );
        assert_eq!(sa.extra, sc.extra, "{label} row {i}: extra");
    }
}

#[test]
fn ewridge_evidence_recovers_truth_emits_pred_std_and_survives_restore() {
    let k = 4;
    let true_b = [1.5, -0.8, 0.0, 2.2];
    for (ev, std_x, intercept) in [
        (EvidenceCfg::default(), false, true),
        (EvidenceCfg::uniform(0.0), false, true),
        (EvidenceCfg::default(), true, true),
        (EvidenceCfg::default(), false, false),
    ] {
        let mut seed = 101u64;
        let rows: Vec<_> = (0..180)
            .map(|i| {
                let x: Vec<f64> = (0..k).map(|_| lcg(&mut seed)).collect();
                let bias = if intercept { 0.7 } else { 0.0 };
                let y = bias
                    + true_b.iter().zip(&x).map(|(b, xi)| b * xi).sum::<f64>()
                    + 0.2 * lcg(&mut seed);
                (x, vec![Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0)
            })
            .collect();
        let cfg = EwRidgeCfg {
            n_features: k,
            n_targets: 1,
            fit_intercept: intercept,
            decay: Decay::Halflife(f64::INFINITY),
            ridge: vec![0.0],
            feature_sets: vec![],
            standardize: std_x,
            ridge_scale: false,
            coef_prior: None,
            session_shrink: None,
            long_half_life: None,
            min_weight: 10.0,
            solve_every: 0.0,
            max_rows_between_solves: 1,
            solve_share: None,
            gram_block_rows: 0,
            target_gaps: TargetGaps::OwnRows,
            window: None,
            window_every: None,
            max_rows_between_snapshots: None,
            evidence: Some(ev),
        };
        let mut m = EwRidge::new(cfg.clone()).unwrap();
        let mut last_std = f64::NAN;
        for (i, (x, y, d, w)) in rows.iter().enumerate() {
            let s = m.step(x, y, *d, *w);
            match s.extra {
                Some(Extra::EwRidge { pred_std }) => {
                    assert_eq!(pred_std.len(), 1);
                    if i >= 20 {
                        assert!(pred_std[0].is_finite() && pred_std[0] > 0.0);
                        last_std = pred_std[0];
                    }
                }
                other => panic!("expected Extra::EwRidge, got {other:?}"),
            }
        }
        assert!(last_std > 0.05 && last_std < 0.5, "pred_std={last_std}");
        let coef = &m.coefficients().unwrap()[0];
        let off = usize::from(intercept);
        for j in 0..k {
            assert!(
                (coef[off + j] - true_b[j]).abs() < 0.2,
                "j={j}: got {}, want {}",
                coef[off + j],
                true_b[j]
            );
        }
        assert!(m.evidence_alpha().is_some());
        assert!(m.evidence_lambda().is_some());
        save_restore_identical(EwRidge::new(cfg).unwrap(), &rows, 90, "ewridge_evidence");
    }
}

#[test]
fn ewridge_evidence_zero_weight_leaves_state_untouched() {
    let mut seed = 202u64;
    let cfg = EwRidgeCfg {
        n_features: 2,
        n_targets: 1,
        fit_intercept: true,
        decay: Decay::Halflife(40.0),
        ridge: vec![0.0],
        feature_sets: vec![],
        standardize: false,
        ridge_scale: false,
        coef_prior: None,
        session_shrink: None,
        long_half_life: None,
        min_weight: 5.0,
        solve_every: 0.0,
        max_rows_between_solves: 1,
        solve_share: None,
        gram_block_rows: 0,
        target_gaps: TargetGaps::OwnRows,
        window: None,
        window_every: None,
        max_rows_between_snapshots: None,
        evidence: Some(EvidenceCfg::default()),
    };
    let mut a = EwRidge::new(cfg.clone()).unwrap();
    let mut b = EwRidge::new(cfg).unwrap();
    for i in 0..40 {
        let x = [lcg(&mut seed), lcg(&mut seed)];
        let y = 0.5 + 1.4 * x[0] - 0.9 * x[1] + 0.1 * lcg(&mut seed);
        let d = if i == 0 { 0.0 } else { 1.0 };
        a.step(&x, &[Some(y)], d, 1.0);
        b.step(&x, &[Some(y)], d, 1.0);
        if i == 20 {
            // Insert a zero-weight row in b and fold its clock delta into the next row.
            b.step(&[99.0, -99.0], &[Some(1e5)], 0.0, 0.0);
        }
    }
    assert_eq!(a, b);
}

#[test]
fn pa_and_sgd_averaging_zero_weight_and_save_restore() {
    let mut seed = 303u64;
    let rows: Vec<_> = (0..120)
        .map(|i| {
            let x = vec![lcg(&mut seed), lcg(&mut seed)];
            let y = 0.4 + 1.5 * x[0] - 0.7 * x[1] + 0.15 * lcg(&mut seed);
            (x, vec![Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0)
        })
        .collect();

    // PA average=true
    let pa_cfg = PaCfg {
        n_features: 2,
        n_targets: 1,
        fit_intercept: true,
        decay: Decay::Halflife(50.0),
        mode: PaMode::Pa1,
        c: 0.5,
        eps: 0.01,
        min_weight: 5.0,
        constraint: None,
        standardize: true,
        average: true,
    };
    save_restore_identical(Pa::new(pa_cfg.clone()).unwrap(), &rows, 60, "pa_average");
    let mut pa_a = Pa::new(pa_cfg.clone()).unwrap();
    let mut pa_b = Pa::new(pa_cfg).unwrap();
    for (i, (x, y, d, w)) in rows.iter().enumerate() {
        pa_a.step(x, y, *d, *w);
        pa_b.step(x, y, *d, *w);
        if i == 40 {
            pa_b.step(&[100.0, -100.0], &[Some(1e4)], 0.0, 0.0);
        }
    }
    assert_eq!(pa_a, pa_b);

    // SGD elastic-net + Optimal + average + invariant
    for (schedule, l1, l2, average, invariant) in [
        (LearningRate::Constant, 0.02, 0.01, true, false),
        (LearningRate::Optimal, 0.0, 0.05, true, false),
        (LearningRate::Optimal, 0.01, 0.05, false, false),
        (LearningRate::Constant, 0.0, 0.02, false, true),
        (LearningRate::Optimal, 0.01, 0.05, true, true),
    ] {
        let sgd_cfg = SgdCfg {
            n_features: 2,
            n_targets: 1,
            fit_intercept: true,
            decay: Decay::Halflife(50.0),
            loss: SgdLoss::Squared,
            learning_rate: 0.05,
            schedule,
            l2,
            min_weight: 5.0,
            clip_gradient: 1e3,
            constraint: None,
            standardize: true,
            strict_binary: false,
            huber_scale: HuberScale::Rms,
            huber_scale_decay: None,
            refit_rows: 0,
            min_weight_holds_learning: false,
            target_spread: true,
            mad_anchor: true,
            refit_clock: None,
            max_state_bytes: None,
            l1,
            average,
            invariant,
        };
        save_restore_identical(
            Sgd::new(sgd_cfg.clone()).unwrap(),
            &rows,
            60,
            "sgd_variants",
        );
        let mut s_a = Sgd::new(sgd_cfg.clone()).unwrap();
        let mut s_b = Sgd::new(sgd_cfg).unwrap();
        for (i, (x, y, d, w)) in rows.iter().enumerate() {
            s_a.step(x, y, *d, *w);
            s_b.step(x, y, *d, *w);
            if i == 40 {
                s_b.step(&[100.0, -100.0], &[Some(1e4)], 0.0, 0.0);
            }
        }
        assert_eq!(s_a, s_b);
    }
}

#[test]
fn lasso_positive_enforces_non_negative_slopes_on_small_and_split_k() {
    for k in [4usize, 44usize] {
        let mut seed = 404u64 + k as u64;
        let rows: Vec<_> = (0..150)
            .map(|i| {
                let x: Vec<f64> = (0..k).map(|_| lcg(&mut seed)).collect();
                // Feature 0 positive (+2), feature 1 strongly negative (-2), intercept negative (-1.5)
                let y = -1.5 + 2.0 * x[0] - 2.0 * x[1] + 0.05 * lcg(&mut seed);
                (x, vec![Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0)
            })
            .collect();
        let cfg = LassoCfg {
            n_features: k,
            n_targets: 1,
            fit_intercept: true,
            decay: Decay::Halflife(f64::INFINITY),
            lasso_path: vec![0.2, 0.05, 0.01],
            l1_ratio: 0.8,
            select_half_life: None,
            min_weight: 5.0,
            target_min_weight: Vec::new(),
            min_weight_holds_learning: false,
            solve_every: 0.0,
            max_rows_between_solves: 1,
            solve_share: None,
            window: None,
            window_every: None,
            max_rows_between_snapshots: None,
            max_iter: 400,
            tol: 1e-12,
            target_gaps: TargetGaps::OwnRows,
            positive: true,
        };
        let mut m = Lasso::new(cfg.clone()).unwrap();
        for (x, y, d, w) in &rows {
            m.step(x, y, *d, *w);
        }
        for slot in &m.coefficients().unwrap()[0] {
            // Intercept may be negative!
            assert!(
                slot[0] < -0.5,
                "k={k}: intercept should stay unconstrained, got {}",
                slot[0]
            );
            for (j, &c) in slot[1..].iter().enumerate() {
                assert!(
                    c >= 0.0,
                    "k={k}: slope {j} was negative ({c}) under positive=true"
                );
            }
            // Feature 0 should be positive, feature 1 clamped to 0.
            assert!(slot[1] > 1.0, "k={k}: positive slope was {}", slot[1]);
            assert_eq!(
                slot[2], 0.0,
                "k={k}: negative truth slope should clamp to 0"
            );
        }
        save_restore_identical(
            Lasso::new(cfg).unwrap(),
            &rows,
            80,
            &format!("lasso_pos_k{k}"),
        );
    }
}

#[test]
fn huber_redescending_norms_reject_outliers_and_keep_scale_alive() {
    let mut seed = 505u64;
    let rows: Vec<_> = (0..180)
        .map(|i| {
            let x = vec![lcg(&mut seed), lcg(&mut seed)];
            let mut y = 1.0 + 2.0 * x[0] - 1.5 * x[1] + 0.1 * lcg(&mut seed);
            if i % 10 == 7 {
                y += 100.0; // massive outlier — Tukey/Andrews/Hampel weight = 0!
            }
            (x, vec![Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0)
        })
        .collect();

    for norm in [
        HuberNorm::Tukey,
        HuberNorm::Andrews,
        HuberNorm::Hampel,
        HuberNorm::Ramsay,
    ] {
        let cfg = RobustCfg {
            n_features: 2,
            n_targets: 1,
            fit_intercept: true,
            decay: Decay::Halflife(f64::INFINITY),
            loss: RobustLoss::Huber {
                delta: norm.default_delta(),
            },
            ridge: 1e-4,
            standardize: false,
            min_weight: 5.0,
            solve_every: 0.0,
            max_rows_between_solves: 1,
            solve_share: None,
            quantile_eps: 1e-3,
            huber_scale: HuberScale::Mad,
            huber_scale_decay: None,
            feature_sets: vec![],
            ridge_scale: false,
            emit_scale: true,
            refit_rows: 40,
            mad_anchor: true,
            min_weight_holds_learning: false,
            refit_clock: None,
            max_state_bytes: None,
            huber_norm: norm,
        };
        let mut m = Robust::new(cfg.clone()).unwrap();
        for (x, y, d, w) in &rows {
            m.step(x, y, *d, *w);
        }
        let coef = &m.coefficients().unwrap()[0];
        assert!(
            (coef[0] - 1.0).abs() < 0.15
                && (coef[1] - 2.0).abs() < 0.15
                && (coef[2] + 1.5).abs() < 0.15,
            "norm={norm:?}: coef={coef:?} failed to reject 100-sigma outliers"
        );
        save_restore_identical(
            Robust::new(cfg).unwrap(),
            &rows,
            90,
            &format!("huber_{norm:?}"),
        );
    }
}
