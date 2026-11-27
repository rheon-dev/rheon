fn ew_class_cfg() -> EwClassCfg {
    EwClassCfg {
        n_features: K,
        n_classes: 2,
        decay: decay(),
        min_weight: 3.0,
        covariance: Covariance::Full,
        precision_prior: 0.1,
        window: None,
        window_every: None,
        max_rows_between_snapshots: None,
    }
}

/// 60 rows a clock unit apart, each model's `n_eff` accessor read before
/// every row against the `n_eff` that row's step reports; the last row's.
fn accessor_is_the_step<M: OnlineModel>(
    mut m: M,
    targets: usize,
    n_eff_of: &dyn Fn(&M) -> f64,
) -> f64 {
    let kind = m.state().model.kind();
    let mut s = 20261121u64;
    let mut last = 0.0;
    for i in 0..60 {
        let x: Vec<f64> = (0..K).map(|_| lcg(&mut s)).collect();
        let y: Vec<Option<f64>> = (0..targets)
            .map(|j| Some(j as f64 + x[0] - 0.5 * x[1]))
            .collect();
        let before = n_eff_of(&m);
        let step = m.step(&x, &y, if i == 0 { 0.0 } else { 1.0 }, 1.0);
        assert!(
            (before - step.n_eff).abs() < 1e-12,
            "{kind}, row {i}: the accessor said {before}, the step reported {}",
            step.n_eff
        );
        last = step.n_eff;
    }
    last
}

/// Rule 8 under a window, for every model that keeps one: `n_eff` is the
/// weight inside the window in the accessor as in the field the step
/// emits (review S10: `Lasso::n_eff` reading the whole decayed weight
/// passed every test, the contract running `lasso` unwindowed only). Each
/// windowed run's weight is under the unwindowed run's, so the window acts.
#[test]
fn the_n_eff_accessor_is_the_steps_under_a_window() {
    const WINDOW: f64 = 10.0;
    let pair = |name: &str, windowed: f64, whole: f64| {
        assert!(
            windowed < whole - 1.0,
            "{name}: the window held {windowed}, the whole stream {whole}: no window acted"
        );
    };
    let ewridge = |window| {
        EwRidge::new(EwRidgeCfg {
            window,
            ..ewridge_cfg()
        })
        .unwrap()
    };
    pair(
        "ewridge",
        accessor_is_the_step(ewridge(Some(WINDOW)), 2, &EwRidge::n_eff),
        accessor_is_the_step(ewridge(None), 2, &EwRidge::n_eff),
    );
    let lasso = |window| {
        Lasso::new(LassoCfg {
            window,
            ..lasso_cfg()
        })
        .unwrap()
    };
    pair(
        "lasso",
        accessor_is_the_step(lasso(Some(WINDOW)), 1, &Lasso::n_eff),
        accessor_is_the_step(lasso(None), 1, &Lasso::n_eff),
    );
    let ew_class = |window| {
        EwClass::new(EwClassCfg {
            window,
            ..ew_class_cfg()
        })
        .unwrap()
    };
    pair(
        "ew_class",
        accessor_is_the_step(ew_class(Some(WINDOW)), 0, &EwClass::n_eff),
        accessor_is_the_step(ew_class(None), 0, &EwClass::n_eff),
    );
    let ew_cov = |window| {
        // A hard window cuts no lagged co-moments, so it runs without lags.
        EwCovModel::new(EwCovCfg {
            window,
            stats: vec![EwCovStat::Mean, EwCovStat::Var, EwCovStat::Corr],
            lags: vec![],
            ..ew_cov_model_cfg()
        })
        .unwrap()
    };
    pair(
        "ew_cov",
        accessor_is_the_step(ew_cov(Some(WINDOW)), 0, &EwCovModel::n_eff),
        accessor_is_the_step(ew_cov(None), 0, &EwCovModel::n_eff),
    );
    let marginal = |window| {
        Marginal::new(MarginalCfg {
            window,
            ..marginal_cfg()
        })
        .unwrap()
    };
    pair(
        "marginal",
        accessor_is_the_step(marginal(Some(WINDOW)), 2, &Marginal::n_eff),
        accessor_is_the_step(marginal(None), 2, &Marginal::n_eff),
    );
    // `hawkes` keeps two windows, of rows with an event and of clock units;
    // each against its decayed run.
    let whole = accessor_is_the_step(Hawkes::new(hawkes_cfg()).unwrap(), 2, &Hawkes::n_eff);
    pair(
        "hawkes, window_events",
        accessor_is_the_step(
            Hawkes::new(hawkes_windowed_cfg()).unwrap(),
            2,
            &Hawkes::n_eff,
        ),
        whole,
    );
    pair(
        "hawkes, window_size",
        accessor_is_the_step(Hawkes::new(hawkes_clock_cfg()).unwrap(), 2, &Hawkes::n_eff),
        whole,
    );
    // `realized` keeps a window of rows and one of clock units.
    let whole = accessor_is_the_step(Realized::new(realized_cfg()).unwrap(), 2, &Realized::n_eff);
    pair(
        "realized, window_rows",
        accessor_is_the_step(
            Realized::new(realized_rows_cfg()).unwrap(),
            2,
            &Realized::n_eff,
        ),
        whole,
    );
    pair(
        "realized, window_size",
        accessor_is_the_step(
            Realized::new(realized_clock_cfg()).unwrap(),
            2,
            &Realized::n_eff,
        ),
        whole,
    );
    // `power_variation` keeps a window of rows and one of clock units.
    let whole = accessor_is_the_step(
        PowerVariation::new(power_variation_cfg()).unwrap(),
        2,
        &PowerVariation::n_eff,
    );
    pair(
        "power_variation, window_rows",
        accessor_is_the_step(
            PowerVariation::new(power_variation_rows_cfg()).unwrap(),
            2,
            &PowerVariation::n_eff,
        ),
        whole,
    );
    pair(
        "power_variation, window_size",
        accessor_is_the_step(
            PowerVariation::new(power_variation_clock_cfg()).unwrap(),
            2,
            &PowerVariation::n_eff,
        ),
        whole,
    );
    // `expectile` keeps a window of rows and one of clock units.
    let whole = accessor_is_the_step(
        Expectile::new(expectile_cfg()).unwrap(),
        2,
        &Expectile::n_eff,
    );
    pair(
        "expectile, window_rows",
        accessor_is_the_step(
            Expectile::new(expectile_rows_cfg()).unwrap(),
            2,
            &Expectile::n_eff,
        ),
        whole,
    );
    pair(
        "expectile, window_size",
        accessor_is_the_step(
            Expectile::new(expectile_clock_cfg()).unwrap(),
            2,
            &Expectile::n_eff,
        ),
        whole,
    );
    // `isotonic` keeps a window of rows and one of clock units, each with
    // or without a decay.
    let whole = accessor_is_the_step(
        Isotonic::new(isotonic_cfg()).unwrap(),
        2,
        &Isotonic::n_eff,
    );
    for (what, c) in isotonic_windows() {
        pair(
            what,
            accessor_is_the_step(Isotonic::new(c).unwrap(), 2, &Isotonic::n_eff),
            whole,
        );
    }
    let whole = accessor_is_the_step(Krls::new(krls_cfg()).unwrap(), 2, &Krls::n_eff);
    pair(
        "krls, window_rows",
        accessor_is_the_step(Krls::new(krls_window_rows_cfg()).unwrap(), 2, &Krls::n_eff),
        whole,
    );
    // `dependence` keeps row windows over MI and ranks.
    let whole = accessor_is_the_step(
        Dependence::new(dependence_cfg()).unwrap(),
        2,
        &Dependence::n_eff,
    );
    for (what, c) in dependence_windows() {
        pair(
            what,
            accessor_is_the_step(Dependence::new(c).unwrap(), 2, &Dependence::n_eff),
            whole,
        );
    }
}

/// 40 rows, the second target absent on every third row: each target's
/// weight from `target_n_eff_into` after each row, against its recursion
/// over the rows it learned, `w_j <- lam * w_j + [learned]`. A model that
/// fits its targets on one shared solve (`joint`, `rls`) learns a row only
/// where every target is present, so both targets' weights skip those
/// rows; any other learns each target on the rows that carry it. Either
/// way a weight with gaps ends under the shared weight of every row, so a
/// model that reported the shared weight fails. Whether the model keeps a
/// weight per target at all.
fn each_target_weight_is_its_own<M: OnlineModel>(mut m: M, joint: bool) -> bool {
    let kind = m.state().model.kind();
    let lam = 0.5f64.powf(1.0 / HALFLIFE);
    let mut s = 20261122u64;
    let (mut want, mut every) = ([0.0f64; 2], 0.0f64);
    let mut got = Vec::new();
    for i in 0..40 {
        let x: Vec<f64> = (0..K).map(|_| lcg(&mut s)).collect();
        let y0 = 0.5 + x[0] - 0.5 * x[1];
        let y = [Some(y0), (i % 3 != 2).then_some(1.0 + x[1])];
        let d = if i == 0 { 0.0 } else { 1.0 };
        m.step(&x, &y, d, 1.0);
        let lam = if i == 0 { 1.0 } else { lam };
        let all = y.iter().all(Option::is_some);
        for (w, y) in want.iter_mut().zip(&y) {
            let learned = if joint { all } else { y.is_some() };
            *w = lam * *w + f64::from(u8::from(learned));
        }
        every = lam * every + 1.0;
        if !m.target_n_eff_into(&mut got) {
            return false;
        }
        assert_eq!(got.len(), 2, "{kind}: one weight a target");
        for j in 0..2 {
            assert!(
                (got[j] - want[j]).abs() <= 1e-12 * want[j].max(1.0),
                "{kind}, row {i}, target {j}: weight {} where the rows it learned give {}",
                got[j],
                want[j]
            );
        }
    }
    assert!(
        want[1] < every - 1.0,
        "{kind}: the absent rows took nothing off"
    );
    true
}

/// Every model that keeps a weight per target reports each target's own
/// (review S11: `Rls::target_n_eff_into` returning the shared weight for
/// every target passed every test). The list is held to the models that
/// say they keep one, so a new one is not left out.
#[test]
fn every_per_target_weight_is_the_rows_that_carried_that_target() {
    let kept: Vec<&str> = [
        (
            "ewridge",
            each_target_weight_is_its_own(EwRidge::new(ewridge_cfg()).unwrap(), false),
        ),
        (
            "rls",
            each_target_weight_is_its_own(Rls::new(rls_cfg()).unwrap(), true),
        ),
        (
            "lasso",
            each_target_weight_is_its_own(
                Lasso::new(LassoCfg {
                    n_targets: 2,
                    ..lasso_cfg()
                })
                .unwrap(),
                false,
            ),
        ),
        (
            "kalman",
            each_target_weight_is_its_own(Kalman::new(kalman_cfg()).unwrap(), false),
        ),
        (
            "robust",
            each_target_weight_is_its_own(Robust::new(huber_mad_cfg()).unwrap(), false),
        ),
        (
            "ftrl",
            each_target_weight_is_its_own(Ftrl::new(ftrl_cfg()).unwrap(), false),
        ),
        (
            "sgd",
            each_target_weight_is_its_own(Sgd::new(sgd_cfg()).unwrap(), false),
        ),
        (
            "pa",
            each_target_weight_is_its_own(Pa::new(pa_cfg()).unwrap(), false),
        ),
        (
            "holt",
            each_target_weight_is_its_own(Holt::new(holt_cfg()).unwrap(), false),
        ),
        (
            "isotonic",
            each_target_weight_is_its_own(Isotonic::new(isotonic_cfg()).unwrap(), false),
        ),
        (
            "krls",
            each_target_weight_is_its_own(Krls::new(krls_cfg()).unwrap(), true),
        ),
        (
            "dependence",
            each_target_weight_is_its_own(Dependence::new(dependence_cfg()).unwrap(), false),
        ),
    ]
    .into_iter()
    .filter_map(|(k, kept)| kept.then_some(k))
    .collect();
    assert_eq!(
        kept,
        [
            "ewridge", "rls", "lasso", "kalman", "robust", "ftrl", "sgd", "pa", "holt",
            "isotonic", "krls", "dependence",
        ],
        "the models that keep a weight per target"
    );
}

#[test]
fn ew_class() {
    for covariance in [Covariance::Full, Covariance::Shared, Covariance::Diagonal] {
        let cfg = EwClassCfg {
            covariance,
            ..ew_class_cfg()
        };
        // The label is learned from, not predicted as a number: no targets,
        // and the outputs are the class and one posterior per class.
        let m = EwClass::new(cfg).unwrap();
        assert_eq!(m.n_targets(), 0, "ew_class has no targets");
        assert_eq!(m.n_features(), K);
        assert_eq!(m.n_outputs(), 3, "class, p_0, p_1");
        // Probed without labels: n_eff counts every accepted row, so
        // `min_weight` means the same number of rows as everywhere else.
        let r = probe_with(m, 0, Some(&EwClass::n_eff));
        assert_eq!(r.kind, "ew_class");
        assert_eq!(r.pred_len, r.n_outputs);
        assert_eq!(r.n_eff[0], 0.0);
        assert_eq!(r.n_eff[1], 1.0);
        assert!((r.n_eff[2] - (0.5f64.powf(1.0 / HALFLIFE) + 1.0)).abs() < 1e-12);
        assert!((r.after_gap - (r.before_gap * 0.5f64.powi(10) + 1.0)).abs() < 1e-9);
        assert!(r.roundtrips);
    }
}

fn seqtest_cfg() -> SeqTestCfg {
    SeqTestCfg {
        n_targets: 2,
        min_weight: 3.0,
    }
}

#[test]
fn seqtest() {
    // A test of the sign of each target: no features, four slots per target
    // (the two log e-values and the two counts), and no decay at all -- an
    // e-process that forgot would not be one -- so `n_eff` is the plain
    // weight sum and a clock gap changes nothing.
    let m = SeqTest::new(seqtest_cfg()).unwrap();
    assert_eq!(m.n_targets(), 2);
    assert_eq!(m.n_features(), 0, "seqtest reads no features");
    assert_eq!(m.n_outputs(), 2 * SEQTEST_SLOTS);
    let r = probe_with(m, 2, Some(&SeqTest::n_eff));
    assert_eq!(r.kind, "seqtest");
    assert_eq!(r.pred_len, 2 * SEQTEST_SLOTS);
    assert_eq!(r.n_eff, vec![0.0, 1.0, 2.0, 40.0]);
    assert_eq!(r.after_gap, r.before_gap + 1.0, "no decay over a gap");
    assert!(r.roundtrips);
}

fn marginal_cfg() -> MarginalCfg {
    MarginalCfg {
        n_features: K,
        n_targets: 2,
        decay: decay(),
        min_weight: vec![3.0; 2],
        // A lag, so the contract exercises marginal's `clear_lags` arm (it
        // keeps a lag ring; review 2026-09-18, T2) -- which is why marginal is
        // in KEEPS_LAGS above.
        lags: vec![1],
        serial_rule: None,
        cross_lags: None,
        bins: None,
        feature_moments: rheon_core::FeatureMomentLayout::PerTarget,
        window: None,
        window_every: None,
        max_rows_between_snapshots: None,
    }
}

#[test]
fn marginal() {
    // Per-pair moments read from the state, nothing per row: targets and
    // features both counted, zero output slots, and `n_eff` -- every learned
    // row's weight -- on the same recursion as every other model.
    let m = Marginal::new(marginal_cfg()).unwrap();
    assert_eq!(m.n_targets(), 2);
    assert_eq!(m.n_features(), K);
    assert_eq!(m.n_outputs(), 0, "marginal predicts nothing per row");
    let r = probe_with(m, 2, Some(&Marginal::n_eff));
    check(&r, "marginal", 2, 0);
    assert_eq!(r.pred_len, 0);
}

fn deco_cfg() -> DecoCfg {
    DecoCfg {
        n_features: K,
        decay: decay(),
        dynamics: DecoDynamics::Ew,
        alpha: None,
        beta: None,
        blocks: Vec::new(),
        min_weight: 3.0,
    }
}

#[test]
fn deco() {
    // One number for the whole correlation matrix: no targets, three slots
    // (`u`, `rho`, `loglik`), and `n_eff` -- the standardiser's accumulated
    // weight -- on the same recursion as every other model.
    let m = Deco::new(deco_cfg()).unwrap();
    assert_eq!(m.n_targets(), 0, "deco has no targets");
    assert_eq!(m.n_features(), K);
    assert_eq!(m.n_outputs(), 3, "u, rho, loglik");
    let r = probe_with(m, 0, Some(&Deco::n_eff));
    assert_eq!(r.kind, "deco");
    assert_eq!(r.pred_len, r.n_outputs);
    assert_eq!(r.n_eff[0], 0.0);
    assert_eq!(r.n_eff[1], 1.0);
    assert!((r.n_eff[2] - (0.5f64.powf(1.0 / HALFLIFE) + 1.0)).abs() < 1e-12);
    assert!((r.after_gap - (r.before_gap * 0.5f64.powi(10) + 1.0)).abs() < 1e-9);
    assert!(r.roundtrips);
}

fn rcov_cfg() -> RcovCfg {
    RcovCfg {
        n_features: K,
        kind: RcovKind::Kernel,
        kernel: "parzen".into(),
        bandwidth: Some(3),
        jitter: 2,
        theta: 1.0,
        psd: false,
        block_rows: Some(100),
        max_bandwidth: None,
        preavg_rows: None,
        noise_stride: 1,
        iv_stride: 20,
    }
}

#[test]
fn rcov() {
    // A block estimator: no targets, no output slots, and no decay -- the
    // block is the value, read at the group's close. `n_eff` is the plain
    // count of returns, so a clock gap changes nothing.
    let m = Rcov::new(rcov_cfg()).unwrap();
    assert_eq!(m.n_targets(), 0);
    assert_eq!(m.n_features(), K);
    assert_eq!(m.n_outputs(), 0, "rcov reports nothing per row");
    let r = probe_with(m, 0, Some(&Rcov::n_eff));
    assert_eq!(r.kind, "rcov");
    assert_eq!(r.pred_len, 0);
    assert_eq!(r.n_eff, vec![0.0, 1.0, 2.0, 40.0]);
    assert_eq!(r.after_gap, r.before_gap + 1.0, "no decay over a gap");
    assert!(r.roundtrips);
}

fn hmm_cfg() -> HmmCfg {
    HmmCfg {
        n_features: K,
        k: 2,
        decay: decay(),
        covariance: rheon_core::HmmCovariance::Full,
        precision_prior: 1e-2,
        min_weight: 0.0,
        learn: true,
        transition_prior: 1.0,
        transition: None,
        means: Some(vec![-1.0, -1.0, 1.0, 1.0]),
        covs: Some(vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]),
        warm_rows: 10,
        seed_rule: SeedRule::First,
        seed: 1,
        tvtp: None,
        prior_rows: None,
        warm_clock: None,
        extras: None,
    }
}

#[test]
fn hmm() {
    // A hidden state, not a label: no targets, `2K + 2` slots (the filtered
    // and predicted posteriors, the state and the row's log-likelihood),
    // and `n_eff` on the shared recursion -- the responsibilities sum to
    // the row's weight, so it is unchanged.
    let m = Hmm::new(hmm_cfg()).unwrap();
    assert_eq!(m.n_targets(), 0, "hmm has no targets");
    assert_eq!(m.n_features(), K);
    assert_eq!(m.n_outputs(), 6, "p_0, p_1, p1_0, p1_1, state, loglik");
    let r = probe_with(m, 0, Some(&Hmm::n_eff));
    assert_eq!(r.kind, "hmm");
    assert_eq!(r.pred_len, r.n_outputs);
    assert_eq!(r.n_eff[0], 0.0);
    assert_eq!(r.n_eff[1], 1.0);
    assert!((r.n_eff[2] - (0.5f64.powf(1.0 / HALFLIFE) + 1.0)).abs() < 1e-12);
    assert!((r.after_gap - (r.before_gap * 0.5f64.powi(10) + 1.0)).abs() < 1e-9);
    assert!(r.roundtrips);
}

fn corrchange_cfg() -> CorrChangeCfg {
    CorrChangeCfg {
        n_features: K,
        kind: CorrChangeKind::Monitor,
        span_rows: 20,
        alpha: 0.05,
        alpha_adjust: "bonferroni".into(),
        bandwidth: None,
        scalar: false,
        decay: decay(),
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
    }
}

#[test]
fn corrchange() {
    // A test, not a model of the data: no targets, five slots (the
    // statistic, its critical value, the flag, the rows since the last one
    // and since the change it dates), and `n_eff` on the shared recursion.
    // The sequential kind (Wied & Galeano) keeps the same contract.
    for kind in [CorrChangeKind::Monitor, CorrChangeKind::Sequential] {
        let m = CorrChange::new(CorrChangeCfg {
            kind,
            monitor_rows: 15,
            ..corrchange_cfg()
        })
        .unwrap();
        assert_eq!(m.n_targets(), 0);
        assert_eq!(m.n_features(), K);
        assert_eq!(
            m.n_outputs(),
            5,
            "stat, crit, flag, since_flag, since_change"
        );
        let r = probe_with(m, 0, Some(&CorrChange::n_eff));
        assert_eq!(r.kind, "corrchange");
        assert_eq!(r.pred_len, r.n_outputs);
        assert_eq!(r.n_eff[0], 0.0);
        assert_eq!(r.n_eff[1], 1.0);
        assert!((r.n_eff[2] - (0.5f64.powf(1.0 / HALFLIFE) + 1.0)).abs() < 1e-12);
        assert!((r.after_gap - (r.before_gap * 0.5f64.powi(10) + 1.0)).abs() < 1e-9);
        assert!(r.roundtrips, "{kind:?}");
    }
}

fn bocpd_cfg() -> BocpdCfg {
    BocpdCfg {
        n_features: K,
        hazard: 50.0,
        hazard_from_row: false,
        emission: BocpdEmission::Diag,
        prior_mean: Some(vec![0.0; K]),
        prior_kappa: 1.0,
        prior_nu: Some(2.0),
        prior_scale: Some(vec![1.0]),
        robust_beta: 0.0,
        prune_below: 1e-6,
        max_run: 200,
        min_weight: 0.0,
        warm_rows: None,
        hazard_on_clock: false,
        warm_clock: None,
        extras: None,
    }
}

/// [`bocpd_cfg`] with its hazard on the clock: `τ = 20` clock units between
/// changepoints, each step's chance of a break applied before the row it
/// leads into (docs/PLAN.md task 179).
fn bocpd_on_the_clock_cfg() -> BocpdCfg {
    BocpdCfg {
        hazard: 20.0,
        hazard_on_clock: true,
        ..bocpd_cfg()
    }
}

#[test]
fn bocpd() {
    // A posterior over run lengths, not a fit: no targets, `4 + d` slots,
    // and `n_eff` the plain weight sum -- the recursion does not decay, so
    // a clock gap changes nothing.
    let m = Bocpd::new(bocpd_cfg()).unwrap();
    assert_eq!(m.n_targets(), 0);
    assert_eq!(m.n_features(), K);
    assert_eq!(m.n_outputs(), 4 + K);
    let r = probe_with(m, 0, Some(&Bocpd::n_eff));
    assert_eq!(r.kind, "bocpd");
    assert_eq!(r.pred_len, r.n_outputs);
    assert_eq!(r.n_eff, vec![0.0, 1.0, 2.0, 40.0]);
    assert_eq!(r.after_gap, r.before_gap + 1.0, "no decay over a gap");
    assert!(r.roundtrips);
}
