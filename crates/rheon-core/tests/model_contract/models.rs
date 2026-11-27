fn decay() -> Decay {
    Decay::Halflife(HALFLIFE)
}

fn ewridge_cfg() -> EwRidgeCfg {
    EwRidgeCfg {
        n_features: K,
        n_targets: 2,
        fit_intercept: true,
        decay: decay(),
        ridge: vec![1e-6, 0.1],
        feature_sets: vec![],
        standardize: false,
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
        constraints: None,
    }
}

#[test]
fn ewridge() {
    let cfg = ewridge_cfg();
    let r = probe_with(EwRidge::new(cfg).unwrap(), 2, Some(&EwRidge::n_eff));
    check(&r, "ewridge", 2, 2);
}

fn rls_cfg() -> RlsCfg {
    RlsCfg {
        n_features: K,
        n_targets: 2,
        fit_intercept: true,
        decay: decay(),
        delta: 1.0,
        coef_prior: None,
        min_weight: 3.0,
        constraints: None,
    }
}

#[test]
fn rls() {
    let cfg = rls_cfg();
    let r = probe_with(Rls::new(cfg).unwrap(), 2, Some(&Rls::n_eff));
    check(&r, "rls", 2, 1);
}

fn lasso_cfg() -> LassoCfg {
    LassoCfg {
        min_weight_holds_learning: false,
        n_features: K,
        n_targets: 1,
        fit_intercept: true,
        decay: decay(),
        lasso_path: vec![0.1, 0.0],
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
        max_iter: 100,
        tol: 1e-10,
        target_gaps: rheon_core::TargetGaps::OwnRows,
    }
}

#[test]
fn lasso() {
    let cfg = lasso_cfg();
    let r = probe_with(Lasso::new(cfg).unwrap(), 1, Some(&Lasso::n_eff));
    check(&r, "lasso", 1, 2);
}

/// The form the bank builds, `P` in units of the noise (task 278); the
/// form a state saved before it goes on in is held below too.
fn kalman_cfg() -> KalmanCfg {
    KalmanCfg {
        horizons: Vec::new(),
        min_weight_holds_learning: false,
        noise_units: true,
        n_features: K,
        n_targets: 2,
        fit_intercept: true,
        decay: decay(),
        half_life: vec![100.0],
        q: None,
        obs_var: None,
        p0: 1.0,
        share_p: false,
        min_weight: 3.0,
        revert_half_life: vec![f64::INFINITY],
        standardize: true,
        obs: Default::default(),
    }
}

#[test]
fn kalman() {
    let cfg = kalman_cfg();
    let r = probe_with(Kalman::new(cfg).unwrap(), 2, Some(&Kalman::n_eff));
    check(&r, "kalman", 2, 1);
}

/// A state saved before task 278 has no `noise_units` and goes on with `P`
/// absolute and the noise estimate in the gain: held to the same contract.
#[test]
fn kalman_with_the_noise_in_the_gain() {
    let cfg = KalmanCfg {
        noise_units: false,
        ..kalman_cfg()
    };
    let r = probe_with(Kalman::new(cfg).unwrap(), 2, Some(&Kalman::n_eff));
    check(&r, "kalman", 2, 1);
}

/// E41: a reverting filter is held to the same contract as the random walk.
fn kalman_revert_cfg() -> KalmanCfg {
    KalmanCfg {
        revert_half_life: vec![f64::INFINITY, 25.0, 6.0],
        ..kalman_cfg()
    }
}

#[test]
fn kalman_reverting() {
    let r = probe_with(
        Kalman::new(kalman_revert_cfg()).unwrap(),
        2,
        Some(&Kalman::n_eff),
    );
    check(&r, "kalman", 2, 1);
}

#[test]
fn kalman_ekf_and_ukf() {
    use rheon_core::{KalmanFamily, KalmanFilter, KalmanLink, KalmanObs, UkfNoise, UkfRule};
    let cases = [
        (
            KalmanObs {
                family: KalmanFamily::Poisson,
                link: KalmanLink::Log,
                filter: KalmanFilter::Ekf,
                ..KalmanObs::default()
            },
            None,
        ),
        (
            KalmanObs {
                family: KalmanFamily::Poisson,
                link: KalmanLink::Log,
                filter: KalmanFilter::Ukf,
                ukf_rule: UkfRule::ScalarIndex,
                ukf_noise: UkfNoise::Expected,
                ..KalmanObs::default()
            },
            None,
        ),
        (
            KalmanObs {
                family: KalmanFamily::Gaussian,
                link: KalmanLink::Log,
                filter: KalmanFilter::Ukf,
                ukf_rule: UkfRule::Merwe,
                ukf_noise: UkfNoise::Plugin,
                ..KalmanObs::default()
            },
            Some(0.5),
        ),
    ];
    for (obs, obs_var) in cases {
        let cfg = KalmanCfg {
            half_life: Vec::new(),
            q: Some(vec![1e-4; K + 1]),
            obs_var,
            obs,
            ..kalman_cfg()
        };
        let r = probe_with(Kalman::new(cfg.clone()).unwrap(), 2, Some(&Kalman::n_eff));
        check(&r, "kalman", 2, 1);
        predict_is_the_step_without_the_step(|| Kalman::new(cfg.clone()).unwrap(), 2, false);
    }
}

fn robust_cfg(loss: RobustLoss) -> RobustCfg {
    RobustCfg {
        n_features: K,
        n_targets: 2,
        fit_intercept: true,
        decay: decay(),
        loss,
        ridge: 1e-6,
        standardize: false,
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

const ROBUST_LOSSES: [RobustLoss; 2] = [
    RobustLoss::Huber { delta: 1.5 },
    RobustLoss::Quantile { tau: 0.5 },
];

/// Huber's cut in the MAD scale, its sketch of `|e|` aged on a memory of
/// its own (docs/PLAN.md task 271): held to the contract as the RMS is.
fn huber_mad_cfg() -> RobustCfg {
    RobustCfg {
        huber_scale: rheon_core::HuberScale::Mad,
        huber_scale_decay: Some(Decay::Halflife(HALFLIFE / 5.0)),
        ..robust_cfg(ROBUST_LOSSES[0])
    }
}

#[test]
fn robust() {
    for cfg in ROBUST_LOSSES
        .map(robust_cfg)
        .into_iter()
        .chain([huber_mad_cfg()])
    {
        let r = probe_with(Robust::new(cfg).unwrap(), 2, Some(&Robust::n_eff));
        check(&r, "robust", 2, 1);
    }
}

fn ftrl_cfg() -> FtrlCfg {
    FtrlCfg {
        n_features: K,
        n_targets: 2,
        fit_intercept: true,
        decay: decay(),
        alpha: 0.1,
        beta: 1.0,
        l1: 0.0,
        l2: 1.0,
        min_weight: 3.0,
        strict_binary: false,
        loss: FtrlLoss::Squared,
        standardize: false,
    }
}

#[test]
fn ftrl() {
    let cfg = ftrl_cfg();
    let r = probe_with(Ftrl::new(cfg).unwrap(), 2, Some(&Ftrl::n_eff));
    check(&r, "ftrl", 2, 1);
    let cfg = FtrlCfg {
        standardize: true,
        ..ftrl_cfg()
    };
    let r = probe_with(Ftrl::new(cfg).unwrap(), 2, Some(&Ftrl::n_eff));
    check(&r, "ftrl", 2, 1);
}

fn sgd_cfg() -> SgdCfg {
    SgdCfg {
        n_features: K,
        n_targets: 2,
        fit_intercept: true,
        decay: decay(),
        loss: SgdLoss::Squared,
        learning_rate: 0.01,
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
    }
}

/// `sgd` under the Huber loss, its cut in each scale (docs/PLAN.md task
/// 271): the RMS on the model's clock, the MAD on a memory of its own.
fn sgd_huber_cfgs() -> [SgdCfg; 2] {
    let huber = SgdCfg {
        loss: SgdLoss::Huber { delta: 1.345 },
        ..sgd_cfg()
    };
    [
        huber.clone(),
        SgdCfg {
            huber_scale: rheon_core::HuberScale::Mad,
            huber_scale_decay: Some(Decay::Halflife(HALFLIFE / 5.0)),
            ..huber
        },
    ]
}

#[test]
fn sgd() {
    for cfg in [sgd_cfg()].into_iter().chain(sgd_huber_cfgs()) {
        let r = probe_with(Sgd::new(cfg).unwrap(), 2, Some(&Sgd::n_eff));
        check(&r, "sgd", 2, 1);
    }
}

fn pa_cfg() -> PaCfg {
    PaCfg {
        n_features: K,
        n_targets: 2,
        fit_intercept: true,
        decay: decay(),
        mode: PaMode::Pa1,
        c: 1.0,
        eps: 0.1,
        min_weight: 3.0,
        constraint: None,
        standardize: false,
    }
}

#[test]
fn pa() {
    let cfg = pa_cfg();
    let r = probe_with(Pa::new(cfg).unwrap(), 2, Some(&Pa::n_eff));
    check(&r, "pa", 2, 1);
}

fn holt_cfg() -> HoltCfg {
    HoltCfg {
        horizons: Vec::new(),
        n_targets: 2,
        level_half_life: HALFLIFE,
        trend_half_life: 4.0 * HALFLIFE,
        min_weight: 3.0,
        trend: true,
        ..Default::default()
    }
}

#[test]
fn holt() {
    let cfg = holt_cfg();
    // Holt reads no features, so it is the one model whose `n_features` is 0.
    let r = probe_with(Holt::new(cfg).unwrap(), 2, Some(&Holt::n_eff));
    assert_eq!(r.n_features, 0, "holt takes no features");
    assert_eq!(r.kind, "holt");
    assert_eq!(r.n_targets, 2);
    assert_eq!(r.n_outputs, 2);
    assert_eq!(r.pred_len, 2);
    assert_eq!(r.n_eff[0], 0.0);
    assert_eq!(r.n_eff[1], 1.0);
    assert!((r.n_eff[2] - (0.5f64.powf(1.0 / HALFLIFE) + 1.0)).abs() < 1e-12);
    assert!((r.after_gap - (r.before_gap * 0.5f64.powi(10) + 1.0)).abs() < 1e-9);
    assert!(r.roundtrips);
}

/// The contract on a noisy line (`probe_noisy`): the regression models above
/// on a target whose residuals are not 0, so a weight, a gap, a round trip or
/// a state that holds only where every residual vanishes (a scale of 0, a
/// zero gradient, a gain that never has to move) fails here and not only in
/// the field.
#[test]
fn a_noisy_line_meets_the_same_contract() {
    let r = probe_noisy(
        EwRidge::new(ewridge_cfg()).unwrap(),
        2,
        Some(&EwRidge::n_eff),
    );
    check(&r, "ewridge", 2, 2);
    let r = probe_noisy(Rls::new(rls_cfg()).unwrap(), 2, Some(&Rls::n_eff));
    check(&r, "rls", 2, 1);
    let r = probe_noisy(Lasso::new(lasso_cfg()).unwrap(), 1, Some(&Lasso::n_eff));
    check(&r, "lasso", 1, 2);
    for cfg in [kalman_cfg(), kalman_revert_cfg()] {
        let r = probe_noisy(Kalman::new(cfg).unwrap(), 2, Some(&Kalman::n_eff));
        check(&r, "kalman", 2, 1);
    }
    for cfg in ROBUST_LOSSES
        .map(robust_cfg)
        .into_iter()
        .chain([huber_mad_cfg()])
    {
        let r = probe_noisy(Robust::new(cfg).unwrap(), 2, Some(&Robust::n_eff));
        check(&r, "robust", 2, 1);
    }
    let r = probe_noisy(Ftrl::new(ftrl_cfg()).unwrap(), 2, Some(&Ftrl::n_eff));
    check(&r, "ftrl", 2, 1);
    for cfg in [sgd_cfg()].into_iter().chain(sgd_huber_cfgs()) {
        let r = probe_noisy(Sgd::new(cfg).unwrap(), 2, Some(&Sgd::n_eff));
        check(&r, "sgd", 2, 1);
    }
    let r = probe_noisy(Pa::new(pa_cfg()).unwrap(), 2, Some(&Pa::n_eff));
    check(&r, "pa", 2, 1);
    let r = probe_noisy(Holt::new(holt_cfg()).unwrap(), 2, Some(&Holt::n_eff));
    assert!(r.roundtrips && r.n_eff[..3] == [0.0, 1.0, 0.5f64.powf(1.0 / HALFLIFE) + 1.0]);
    assert!((r.after_gap - (r.before_gap * 0.5f64.powi(10) + 1.0)).abs() < 1e-9);
}

fn ew_cov_model_cfg() -> EwCovCfg {
    EwCovCfg {
        bias: false,
        n_features: K,
        decay: decay(),
        stats: vec![
            EwCovStat::Mean,
            EwCovStat::Var,
            EwCovStat::Corr,
            EwCovStat::LagCorr,
        ],
        min_weight: 3.0,
        precision_prior: None,
        mahal_quantiles: Vec::new(),
        pca: 0,
        pca_every: 0.0,
        max_rows_between_pca: u32::MAX,
        // The probe carries lags so that the save/restore and `clear_lags`
        // arms above see a model with a ring in them (L1).
        lags: vec![1, 3],
        window: None,
        window_every: None,
        max_rows_between_snapshots: None,
    }
}

#[test]
fn ew_cov_model() {
    let cfg = ew_cov_model_cfg();
    // No targets: it emits statistics, one slot per (stat x column-or-pair).
    let m = EwCovModel::new(cfg).unwrap();
    assert_eq!(m.n_targets(), 0, "ew_cov has no targets");
    assert_eq!(m.n_features(), K);
    assert_eq!(
        m.n_outputs(),
        2 + 2 + 1 + 2 * K * K,
        "mean and var per column, one pair, and a k x k block per lag"
    );
    let r = probe_with(m, 0, Some(&EwCovModel::n_eff));
    assert_eq!(r.kind, "ew_cov");
    assert_eq!(r.pred_len, r.n_outputs);
    assert_eq!(r.n_eff[0], 0.0);
    assert_eq!(r.n_eff[1], 1.0);
    // The decayed row and the gap, as the sibling probes hold them (review
    // 2026-10-05, CF1: this probe read rows 0 and 1 alone).
    assert!((r.n_eff[2] - (0.5f64.powf(1.0 / HALFLIFE) + 1.0)).abs() < 1e-12);
    assert!((r.after_gap - (r.before_gap * 0.5f64.powi(10) + 1.0)).abs() < 1e-9);
    assert!(r.roundtrips);
}

fn kmeans_cfg() -> KMeansCfg {
    KMeansCfg {
        n_features: K,
        k: 3,
        decay: decay(),
        min_weight: 3.0,
        warm_rows: 10,
        seed_rule: SeedRule::Lloyd,
        seed: 0,
        update_every_rows: 1,
        split_merge: 0.5,
        split_merge_every_rows: 50,
        dead_frac: 0.05,
        standardize: true,
        scale_floor: 0.1,
        warm_clock: None,
        update_every: None,
        split_merge_every: None,
    }
}

#[test]
fn kmeans() {
    let cfg = kmeans_cfg();
    // No targets: it emits an assignment and two distances.
    let m = KMeans::new(cfg).unwrap();
    assert_eq!(m.n_targets(), 0, "kmeans has no targets");
    assert_eq!(m.n_features(), K);
    assert_eq!(m.n_outputs(), 3, "cluster, dist, dist_second");
    let r = probe_with(m, 0, Some(&KMeans::n_eff));
    assert_eq!(r.kind, "kmeans");
    assert_eq!(r.pred_len, r.n_outputs);
    assert_eq!(r.n_eff[0], 0.0);
    assert_eq!(r.n_eff[1], 1.0);
    assert!((r.n_eff[2] - (0.5f64.powf(1.0 / HALFLIFE) + 1.0)).abs() < 1e-12);
    assert!((r.after_gap - (r.before_gap * 0.5f64.powi(10) + 1.0)).abs() < 1e-9);
    assert!(r.roundtrips);
}

/// Task 281 (mutants): outside the crate's own tests the `lloyd` rule's runs
/// are held to their full cap, 300 passes, which three blobs reach their
/// fixed point well within: a seeding counts no run stopped at the cap.
/// (A cap of one pass stops every run short: its first pass assigns every
/// row.)
#[test]
fn a_lloyd_seeding_of_three_blobs_runs_each_start_to_its_fixed_point() {
    let mut m = KMeans::new(KMeansCfg {
        warm_rows: 60,
        ..kmeans_cfg()
    })
    .unwrap();
    let mut s = 7u64;
    for i in 0..80 {
        let c = [[0.0, 0.0], [5.0, 1.0], [1.0, 6.0]][i % 3];
        let x = [c[0] + lcg(&mut s), c[1] + lcg(&mut s)];
        m.step(&x, &[], 1.0, 1.0);
    }
    assert!(m.seeded());
    assert_eq!(m.solve_failures, 0);
}

fn micro_cfg() -> MicroCfg {
    // eps and beta_mu sized for a 20-half-life window over the harness's
    // uniform rows: a summary must reach beta_mu before it decays back.
    MicroCfg {
        n_features: K,
        decay: decay(),
        min_weight: 3.0,
        eps: 0.6,
        beta_mu: 2.0,
        max_clusters: 50,
        prune_every: f64::INFINITY,
        max_rows_between_prunes: 10,
        macro_link: None,
        standardize: true,
        scale_floor: 0.1,
    }
}

#[test]
fn micro() {
    let cfg = micro_cfg();
    // No targets: a label, a distance, a micro-cluster id, an outlier flag
    // and two counts.
    let m = Micro::new(cfg).unwrap();
    assert_eq!(m.n_targets(), 0, "micro has no targets");
    assert_eq!(m.n_features(), K);
    assert_eq!(
        m.n_outputs(),
        6,
        "cluster, dist, micro_id, outlier, n_clusters, n_micro"
    );
    let r = probe_with(m, 0, Some(&Micro::n_eff));
    assert_eq!(r.kind, "micro");
    assert_eq!(r.pred_len, r.n_outputs);
    assert_eq!(r.n_eff[0], 0.0);
    assert_eq!(r.n_eff[1], 1.0);
    assert!((r.n_eff[2] - (0.5f64.powf(1.0 / HALFLIFE) + 1.0)).abs() < 1e-12);
    assert!((r.after_gap - (r.before_gap * 0.5f64.powi(10) + 1.0)).abs() < 1e-9);
    assert!(r.roundtrips);
}
