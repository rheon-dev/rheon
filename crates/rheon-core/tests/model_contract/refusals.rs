/// Every model with a decay of its own refuses, through its own `new`, a
/// decay it cannot run on -- a half-life of 0, below 0 or NaN, a factor of
/// 0, below 0, above 1 or NaN -- naming itself and the parameter, and runs
/// on no decay (`Halflife(inf)`, `Lam(1)`) and an ordinary one. Only the
/// bank's spec checked (`Spec::decays`), so the Rust API built each on any
/// of them, and `Halflife(0)` gave NaN from the first row (review
/// 2026-10-05, CF5; docs/EXTENDING.md: every parameter check belongs in
/// `new`). `holt`, `seqtest`, `rcov` and `bocpd` take no `Decay`.
#[test]
fn every_model_refuses_a_decay_it_cannot_run_on() {
    type Build<'a> = (&'a str, &'a dyn Fn(Decay) -> Result<(), String>);
    let builds: [Build; 21] = [
        ("ewridge", &|decay| {
            EwRidge::new(EwRidgeCfg {
                decay,
                ..ewridge_cfg()
            })
            .map(drop)
        }),
        ("rls", &|decay| {
            Rls::new(RlsCfg { decay, ..rls_cfg() }).map(drop)
        }),
        ("lasso", &|decay| {
            Lasso::new(LassoCfg {
                decay,
                ..lasso_cfg()
            })
            .map(drop)
        }),
        ("kalman", &|decay| {
            Kalman::new(KalmanCfg {
                decay,
                ..kalman_cfg()
            })
            .map(drop)
        }),
        ("huber", &|decay| {
            Robust::new(RobustCfg {
                decay,
                ..robust_cfg(ROBUST_LOSSES[0])
            })
            .map(drop)
        }),
        ("quantile", &|decay| {
            Robust::new(RobustCfg {
                decay,
                ..robust_cfg(ROBUST_LOSSES[1])
            })
            .map(drop)
        }),
        ("ftrl", &|decay| {
            Ftrl::new(FtrlCfg {
                decay,
                ..ftrl_cfg()
            })
            .map(drop)
        }),
        ("sgd", &|decay| {
            Sgd::new(SgdCfg { decay, ..sgd_cfg() }).map(drop)
        }),
        ("pa", &|decay| {
            Pa::new(PaCfg { decay, ..pa_cfg() }).map(drop)
        }),
        ("ew_cov", &|decay| {
            EwCovModel::new(EwCovCfg {
                bias: false,
                decay,
                ..ew_cov_model_cfg()
            })
            .map(drop)
        }),
        ("kmeans", &|decay| {
            KMeans::new(KMeansCfg {
                decay,
                ..kmeans_cfg()
            })
            .map(drop)
        }),
        ("micro", &|decay| {
            Micro::new(MicroCfg {
                decay,
                ..micro_cfg()
            })
            .map(drop)
        }),
        ("ew_class", &|decay| {
            EwClass::new(EwClassCfg {
                decay,
                ..ew_class_cfg()
            })
            .map(drop)
        }),
        ("marginal", &|decay| {
            Marginal::new(MarginalCfg {
                decay,
                ..marginal_cfg()
            })
            .map(drop)
        }),
        ("deco", &|decay| {
            Deco::new(DecoCfg {
                decay,
                ..deco_cfg()
            })
            .map(drop)
        }),
        ("hmm", &|decay| {
            Hmm::new(HmmCfg { decay, ..hmm_cfg() }).map(drop)
        }),
        ("corrchange", &|decay| {
            CorrChange::new(CorrChangeCfg {
                decay,
                ..corrchange_cfg()
            })
            .map(drop)
        }),
        ("hawkes", &|decay| {
            Hawkes::new(HawkesCfg {
                decay,
                ..hawkes_cfg()
            })
            .map(drop)
        }),
        ("realized", &|decay| {
            Realized::new(RealizedCfg {
                decay,
                ..realized_cfg()
            })
            .map(drop)
        }),
        ("power_variation", &|decay| {
            PowerVariation::new(PowerVariationCfg {
                decay,
                ..power_variation_cfg()
            })
            .map(drop)
        }),
        ("expectile", &|decay| {
            Expectile::new(ExpectileCfg {
                decay,
                ..expectile_cfg()
            })
            .map(drop)
        }),
    ];
    let bad = [
        Decay::Halflife(0.0),
        Decay::Halflife(-1.0),
        Decay::Halflife(f64::NAN),
        Decay::Lam(1.5),
        Decay::Lam(0.0),
        Decay::Lam(-0.5),
        Decay::Lam(f64::NAN),
    ];
    for (name, build) in builds {
        for decay in bad {
            let e = build(decay).expect_err(&format!("{name}: {decay:?} was accepted"));
            let says = match decay {
                Decay::Halflife(_) => "half_life must be above 0",
                Decay::Lam(_) => "lam must be above 0 and at most 1",
            };
            assert!(
                e.starts_with(&format!("{name}: {says}")),
                "{name}, {decay:?}: {e}"
            );
        }
        for decay in [
            Decay::Halflife(f64::INFINITY),
            Decay::Halflife(HALFLIFE),
            Decay::Lam(1.0),
            Decay::Lam(0.97),
        ] {
            build(decay).unwrap_or_else(|e| panic!("{name}, {decay:?}: {e}"));
        }
    }
}

/// `model.<variant>.cfg.<key>` of a state's msgpack, set to `value`.
fn set_cfg(v: &mut rmpv::Value, key: &str, value: rmpv::Value) {
    fn entry<'a>(v: &'a mut rmpv::Value, key: &str) -> &'a mut rmpv::Value {
        let rmpv::Value::Map(entries) = v else {
            panic!("a map holds {key}")
        };
        &mut entries
            .iter_mut()
            .find(|(k, _)| k.as_str() == Some(key))
            .unwrap_or_else(|| panic!("no {key}"))
            .1
    }
    let rmpv::Value::Map(variant) = entry(v, "model") else {
        panic!("a model")
    };
    *entry(entry(&mut variant[0].1, "cfg"), key) = value;
}

/// Every model's `restore` holds the configuration its state carries to
/// what its `new` holds a fresh one to: no `restore` ran `validate`, so a
/// state whose cfg `new` refuses -- `kmeans` with `k = 0`, `micro` with
/// `max_clusters = 0`, a half-life of 0 -- loaded and failed at a later
/// row, as a panic or a NaN for good (review 2026-10-06, CF4, CA6, CD14).
/// Each model's state with one cfg field set, in the named msgpack a bank
/// writes, to a value its `validate` refuses: a half-life of 0 for the
/// seventeen with a `Decay`, a field of its own for the five without.
#[test]
fn every_model_refuses_at_restore_a_configuration_its_new_refuses() {
    fn refused<M: OnlineModel>(m: M, key: &str, value: rmpv::Value, says: &str) -> String {
        let state = m.state();
        let kind = state.model.kind();
        let damaged = edit_state(&state, |v| set_cfg(v, key, value));
        match M::restore(&damaged) {
            Err(StateError::Invalid(e)) => {
                // `robust`'s messages name the spec's model, `huber` or
                // `quantile` by its loss (docs/PLAN.md task 196, N1).
                let named = |k: &str| {
                    e.starts_with(&format!(
                        "{k}: it is damaged, since it saved a setting {k} refuses: "
                    )) && e.ends_with(rheon_core::DAMAGED)
                };
                let head =
                    named(kind) || (kind == "robust" && (named("huber") || named("quantile")));
                assert!(head && e.contains(says), "{kind}: {e}");
            }
            Err(e) => panic!("{kind}: {e}"),
            Ok(_) => panic!("{kind}: a {key} its `new` refuses was restored"),
        }
        assert!(M::restore(&state).is_ok(), "{kind}: the state as saved");
        kind.to_string()
    }
    let zero = || rmpv::Value::Map(vec![("Halflife".into(), 0.0.into())]);
    let h = "half_life must be above 0";
    let kinds = [
        refused(EwRidge::new(ewridge_cfg()).unwrap(), "decay", zero(), h),
        refused(Rls::new(rls_cfg()).unwrap(), "decay", zero(), h),
        refused(Lasso::new(lasso_cfg()).unwrap(), "decay", zero(), h),
        refused(Kalman::new(kalman_cfg()).unwrap(), "decay", zero(), h),
        refused(
            Robust::new(robust_cfg(ROBUST_LOSSES[0])).unwrap(),
            "decay",
            zero(),
            h,
        ),
        refused(
            Robust::new(robust_cfg(ROBUST_LOSSES[1])).unwrap(),
            "decay",
            zero(),
            h,
        ),
        refused(Ftrl::new(ftrl_cfg()).unwrap(), "decay", zero(), h),
        refused(Sgd::new(sgd_cfg()).unwrap(), "decay", zero(), h),
        refused(Pa::new(pa_cfg()).unwrap(), "decay", zero(), h),
        refused(
            Holt::new(holt_cfg()).unwrap(),
            "level_half_life",
            0.0.into(),
            "half_life, the level's half-life, must be above 0",
        ),
        refused(
            EwCovModel::new(ew_cov_model_cfg()).unwrap(),
            "decay",
            zero(),
            h,
        ),
        refused(KMeans::new(kmeans_cfg()).unwrap(), "decay", zero(), h),
        refused(Micro::new(micro_cfg()).unwrap(), "decay", zero(), h),
        refused(EwClass::new(ew_class_cfg()).unwrap(), "decay", zero(), h),
        refused(
            SeqTest::new(seqtest_cfg()).unwrap(),
            "min_weight",
            f64::NAN.into(),
            "min_weight must be at least 0",
        ),
        refused(Marginal::new(marginal_cfg()).unwrap(), "decay", zero(), h),
        refused(Deco::new(deco_cfg()).unwrap(), "decay", zero(), h),
        refused(
            Rcov::new(rcov_cfg()).unwrap(),
            "theta",
            0.0.into(),
            "theta must be finite and above 0",
        ),
        refused(Hmm::new(hmm_cfg()).unwrap(), "decay", zero(), h),
        refused(
            CorrChange::new(corrchange_cfg()).unwrap(),
            "decay",
            zero(),
            h,
        ),
        refused(
            Bocpd::new(bocpd_cfg()).unwrap(),
            "hazard",
            0.5.into(),
            "must be finite and above 1",
        ),
        refused(
            Audit::new(audit_cfg()).unwrap(),
            "distinct_cap",
            0.into(),
            "distinct_cap must be from 1 to",
        ),
        refused(Glm::new(glm_cfg()).unwrap(), "decay", zero(), h),
        refused(Hawkes::new(hawkes_cfg()).unwrap(), "decay", zero(), h),
        refused(Expectile::new(expectile_cfg()).unwrap(), "decay", zero(), h),
        refused(Realized::new(realized_cfg()).unwrap(), "decay", zero(), h),
        refused(
            PowerVariation::new(power_variation_cfg()).unwrap(),
            "decay",
            zero(),
            h,
        ),
        refused(
            Garch::new(garch_cfg()).unwrap(),
            "horizons",
            rmpv::Value::Array(vec![2.into(), 0.into()]),
            "each of horizons must be at least 1",
        ),
        refused(
            Dcc::new(dcc_cfg()).unwrap(),
            "start_rows",
            0.into(),
            "start_rows must be at least 1",
        ),
        refused(StreamPca::new(pca_cfg()).unwrap(), "decay", zero(), h),
        refused(
            TDigestModel::new(tdigest_cfg()).unwrap(),
            "compression",
            9.5.into(),
            "compression must be at least 10",
        ),
        refused(
            DdSketchModel::new(ddsketch_cfg()).unwrap(),
            "alpha",
            1.5.into(),
            "alpha must be at least",
        ),
        refused(
            EwmMoments::new(ewm_moments_cfg()).unwrap(),
            "decay",
            zero(),
            h,
        ),
        refused(Ar::new(ar_cfg()).unwrap(), "decay", zero(), h),
        refused(Spectral::new(spectral_cfg()).unwrap(), "decay", zero(), h),
    ];
    // Every model's `restore`: the twenty-one kinds, `robust` under both
    // losses.
    let mut seen: Vec<&str> = kinds.iter().map(String::as_str).collect();
    seen.dedup();
    let models = PROBED.len() - 1;
    assert_eq!(seen.len(), models, "{seen:?}");
}

/// Each core `validate` refuses what the spec layer refuses (`spec.rs`),
/// naming the model, the parameter and the value, so the Rust API and a
/// state file are held to the rule a spec is. Each of these built a model
/// that learned nothing, predicted 0 or poisoned its state with a NaN that
/// never washes out, with no error: a NaN or negative `ridge`, a `tol` that
/// is not a positive number, a `select_half_life` of 0 (the selection
/// weight `2^(-0/0)`), a path point that is not a number, an infinite `rls`
/// ridge, a NaN `min_weight`, a `quantile_eps` of NaN or infinity, an
/// infinite `p0`, `obs_var` or `q` (review 2026-10-06, CA6 and CC7). The
/// legal neighbours -- `inf` where it means something -- still build.
#[test]
fn every_core_validate_refuses_what_the_spec_refuses() {
    let (nan, inf) = (f64::NAN, f64::INFINITY);
    let ewridge = |c: EwRidgeCfg| EwRidge::new(c).map(drop);
    let lasso = |c: LassoCfg| Lasso::new(c).map(drop);
    let rls = |c: RlsCfg| Rls::new(c).map(drop);
    let robust = |c: RobustCfg| Robust::new(c).map(drop);
    let kalman = |c: KalmanCfg| Kalman::new(c).map(drop);
    let huber = || robust_cfg(ROBUST_LOSSES[0]);
    let quantile = || robust_cfg(ROBUST_LOSSES[1]);
    let refused: Vec<(Result<(), String>, &str)> = vec![
        (
            ewridge(EwRidgeCfg {
                ridge: vec![1e-6, nan],
                ..ewridge_cfg()
            }),
            "ewridge: ridge must be finite and at least 0, but one of its values is NaN.",
        ),
        (
            ewridge(EwRidgeCfg {
                ridge: vec![-1.0],
                ..ewridge_cfg()
            }),
            "ewridge: ridge must be finite and at least 0, but it is -1.",
        ),
        (
            ewridge(EwRidgeCfg {
                ridge: vec![inf],
                ..ewridge_cfg()
            }),
            "ewridge: ridge must be finite and at least 0, but it is inf.",
        ),
        (
            ewridge(EwRidgeCfg {
                min_weight: nan,
                ..ewridge_cfg()
            }),
            "ewridge: min_weight must be at least 0, but it is NaN.",
        ),
        (
            lasso(LassoCfg {
                tol: nan,
                ..lasso_cfg()
            }),
            "lasso: tol must be finite and above 0, but it is NaN.",
        ),
        (
            lasso(LassoCfg {
                tol: 0.0,
                ..lasso_cfg()
            }),
            "lasso: tol must be finite and above 0, but it is 0.",
        ),
        (
            lasso(LassoCfg {
                tol: inf,
                ..lasso_cfg()
            }),
            "lasso: tol must be finite and above 0, but it is inf.",
        ),
        (
            lasso(LassoCfg {
                select_half_life: Some(0.0),
                ..lasso_cfg()
            }),
            "lasso: select_half_life must be above 0, but it is 0.",
        ),
        (
            lasso(LassoCfg {
                select_half_life: Some(nan),
                ..lasso_cfg()
            }),
            "lasso: select_half_life must be above 0, but it is NaN.",
        ),
        (
            lasso(LassoCfg {
                lasso_path: vec![nan],
                ..lasso_cfg()
            }),
            "lasso: penalties must each be finite and at least 0, but one of them is NaN.",
        ),
        (
            lasso(LassoCfg {
                lasso_path: vec![inf, 0.1],
                ..lasso_cfg()
            }),
            "lasso: penalties must each be finite and at least 0, but one of them is inf.",
        ),
        (
            lasso(LassoCfg {
                min_weight: nan,
                ..lasso_cfg()
            }),
            "lasso: min_weight must be at least 0, but it is NaN.",
        ),
        (
            rls(RlsCfg {
                delta: inf,
                ..rls_cfg()
            }),
            "rls: delta must be finite and above 0, but it is inf.",
        ),
        (
            rls(RlsCfg {
                delta: nan,
                ..rls_cfg()
            }),
            "rls: delta must be finite and above 0, but it is NaN.",
        ),
        (
            rls(RlsCfg {
                min_weight: nan,
                ..rls_cfg()
            }),
            "rls: min_weight must be at least 0, but it is NaN.",
        ),
        // Review round 5 (C6): a NaN `min_weight` loaded through the Rust
        // API for the three models that did not check it, and a NaN
        // `solve_every` for the three that solve on a schedule.
        (
            Sgd::new(SgdCfg {
                min_weight: nan,
                ..sgd_cfg()
            })
            .map(drop),
            "sgd: min_weight must be at least 0, but it is NaN.",
        ),
        (
            Sgd::new(SgdCfg {
                min_weight: -1.0,
                ..sgd_cfg()
            })
            .map(drop),
            "sgd: min_weight must be at least 0, but it is -1.",
        ),
        (
            Holt::new(HoltCfg {
                min_weight: nan,
                ..holt_cfg()
            })
            .map(drop),
            "holt: min_weight must be at least 0, but it is NaN.",
        ),
        (
            EwCovModel::new(EwCovCfg {
                bias: false,
                min_weight: nan,
                ..ew_cov_model_cfg()
            })
            .map(drop),
            "ew_cov: min_weight must be at least 0, but it is NaN.",
        ),
        (
            ewridge(EwRidgeCfg {
                solve_every: nan,
                ..ewridge_cfg()
            }),
            "ewridge: solve_every must be a number, but it is NaN.",
        ),
        (
            lasso(LassoCfg {
                solve_every: nan,
                ..lasso_cfg()
            }),
            "lasso: solve_every must be a number, but it is NaN.",
        ),
        (
            robust(RobustCfg {
                solve_every: nan,
                ..quantile()
            }),
            "quantile: solve_every must be a number, but it is NaN.",
        ),
        (
            robust(RobustCfg {
                ridge: nan,
                ..huber()
            }),
            "huber: ridge must be finite and at least 0, but it is NaN.",
        ),
        (
            robust(RobustCfg {
                ridge: inf,
                ..quantile()
            }),
            "quantile: ridge must be finite and at least 0, but it is inf.",
        ),
        (
            robust(RobustCfg {
                quantile_eps: nan,
                ..quantile()
            }),
            "quantile: quantile_eps must be finite and above 0, but it is NaN.",
        ),
        (
            robust(RobustCfg {
                quantile_eps: inf,
                ..quantile()
            }),
            "quantile: quantile_eps must be finite and above 0, but it is inf.",
        ),
        (
            robust(RobustCfg {
                min_weight: nan,
                ..huber()
            }),
            "huber: min_weight must be at least 0, but it is NaN.",
        ),
        (
            robust(RobustCfg {
                loss: RobustLoss::Huber { delta: nan },
                ..huber()
            }),
            "huber: huber_delta must be above 0, but it is NaN.",
        ),
        (
            robust(RobustCfg {
                loss: RobustLoss::Quantile { tau: 1.5 },
                ..quantile()
            }),
            "quantile: quantile must be above 0 and below 1, but it is 1.5.",
        ),
        (
            kalman(KalmanCfg {
                p0: inf,
                ..kalman_cfg()
            }),
            "kalman: p0 must be finite and above 0, but it is inf.",
        ),
        (
            kalman(KalmanCfg {
                obs_var: Some(inf),
                ..kalman_cfg()
            }),
            "kalman: obs_var must be finite and above 0, but it is inf.",
        ),
        (
            kalman(KalmanCfg {
                q: Some(vec![0.0, inf, 0.0]),
                ..kalman_cfg()
            }),
            "kalman: q must be finite and at least 0, but one of its values is inf.",
        ),
        (
            kalman(KalmanCfg {
                min_weight: nan,
                ..kalman_cfg()
            }),
            "kalman: min_weight must be at least 0, but it is NaN.",
        ),
        (
            kalman(KalmanCfg {
                half_life: vec![nan],
                ..kalman_cfg()
            }),
            "kalman: coef_half_life must be above 0, but it is NaN.",
        ),
        (
            kalman(KalmanCfg {
                revert_half_life: vec![0.0],
                ..kalman_cfg()
            }),
            "kalman: revert_half_life must be above 0, but it is 0.",
        ),
    ];
    let wrong: Vec<String> = refused
        .into_iter()
        .enumerate()
        .filter_map(|(i, (got, msg))| match got {
            Ok(()) => Some(format!("case {i} accepted, wanted: {msg}")),
            Err(e) if !e.contains(msg) => Some(format!("case {i} refused as: {e}")),
            Err(_) => None,
        })
        .collect();
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
    let accepted: Vec<Result<(), String>> = vec![
        ewridge(EwRidgeCfg {
            ridge: vec![0.0],
            min_weight: inf,
            ..ewridge_cfg()
        }),
        lasso(LassoCfg {
            select_half_life: Some(inf),
            lasso_path: vec![0.0],
            min_weight: inf,
            ..lasso_cfg()
        }),
        rls(RlsCfg {
            delta: 1e300,
            min_weight: inf,
            ..rls_cfg()
        }),
        robust(RobustCfg {
            ridge: 0.0,
            loss: RobustLoss::Huber { delta: inf },
            min_weight: inf,
            ..huber()
        }),
        robust(RobustCfg {
            quantile_eps: 1e-9,
            ..quantile()
        }),
        kalman(KalmanCfg {
            q: Some(vec![0.0; 3]),
            obs_var: Some(1e300),
            half_life: vec![inf],
            min_weight: inf,
            ..kalman_cfg()
        }),
        // An infinite `min_weight` never predicts, and an infinite
        // `solve_every` leaves the cadence to the row cap: both legal, as
        // the models above take them (C6).
        Sgd::new(SgdCfg {
            min_weight: inf,
            ..sgd_cfg()
        })
        .map(drop),
        Holt::new(HoltCfg {
            min_weight: inf,
            ..holt_cfg()
        })
        .map(drop),
        EwCovModel::new(EwCovCfg {
            bias: false,
            min_weight: inf,
            ..ew_cov_model_cfg()
        })
        .map(drop),
        ewridge(EwRidgeCfg {
            solve_every: inf,
            ..ewridge_cfg()
        }),
        lasso(LassoCfg {
            solve_every: inf,
            ..lasso_cfg()
        }),
        robust(RobustCfg {
            solve_every: inf,
            ..huber()
        }),
    ];
    for (i, got) in accepted.into_iter().enumerate() {
        got.unwrap_or_else(|e| panic!("legal neighbour {i}: {e}"));
    }
}

/// Exactly the models that solve on a schedule report a solve share --
/// `ewridge`, `lasso`, and `robust` under both losses -- each the share it
/// is given; every other model keeps the trait's `None` whatever it is set
/// to (docs/EXTENDING.md's table of hooks; review 2026-10-05, CF7). The
/// stream sets the spec's share on every model it builds or restores, so
/// these are the models whose cadence it reaches.
#[test]
fn exactly_the_scheduled_solvers_report_a_solve_share() {
    fn share<M: OnlineModel>(mut m: M) -> Option<f64> {
        m.set_solve_share(Some(0.3));
        m.solve_share()
    }
    let shares = [
        ("ewridge", share(EwRidge::new(ewridge_cfg()).unwrap())),
        ("rls", share(Rls::new(rls_cfg()).unwrap())),
        ("lasso", share(Lasso::new(lasso_cfg()).unwrap())),
        ("kalman", share(Kalman::new(kalman_cfg()).unwrap())),
        (
            "huber",
            share(Robust::new(robust_cfg(ROBUST_LOSSES[0])).unwrap()),
        ),
        (
            "quantile",
            share(Robust::new(robust_cfg(ROBUST_LOSSES[1])).unwrap()),
        ),
        ("ftrl", share(Ftrl::new(ftrl_cfg()).unwrap())),
        ("sgd", share(Sgd::new(sgd_cfg()).unwrap())),
        ("pa", share(Pa::new(pa_cfg()).unwrap())),
        ("holt", share(Holt::new(holt_cfg()).unwrap())),
        (
            "ew_cov",
            share(EwCovModel::new(ew_cov_model_cfg()).unwrap()),
        ),
        ("kmeans", share(KMeans::new(kmeans_cfg()).unwrap())),
        ("micro", share(Micro::new(micro_cfg()).unwrap())),
        ("ew_class", share(EwClass::new(ew_class_cfg()).unwrap())),
        ("seqtest", share(SeqTest::new(seqtest_cfg()).unwrap())),
        ("marginal", share(Marginal::new(marginal_cfg()).unwrap())),
        ("deco", share(Deco::new(deco_cfg()).unwrap())),
        ("rcov", share(Rcov::new(rcov_cfg()).unwrap())),
        ("hmm", share(Hmm::new(hmm_cfg()).unwrap())),
        (
            "corrchange",
            share(CorrChange::new(corrchange_cfg()).unwrap()),
        ),
        ("bocpd", share(Bocpd::new(bocpd_cfg()).unwrap())),
        ("audit", share(Audit::new(audit_cfg()).unwrap())),
        ("glm", share(Glm::new(glm_cfg()).unwrap())),
        ("hawkes", share(Hawkes::new(hawkes_cfg()).unwrap())),
        ("expectile", share(Expectile::new(expectile_cfg()).unwrap())),
        ("garch", share(Garch::new(garch_cfg()).unwrap())),
        ("dcc", share(Dcc::new(dcc_cfg()).unwrap())),
        ("realized", share(Realized::new(realized_cfg()).unwrap())),
        ("pca", share(StreamPca::new(pca_cfg()).unwrap())),
        (
            "power_variation",
            share(PowerVariation::new(power_variation_cfg()).unwrap()),
        ),
        ("tdigest", share(TDigestModel::new(tdigest_cfg()).unwrap())),
        (
            "ddsketch",
            share(DdSketchModel::new(ddsketch_cfg()).unwrap()),
        ),
        ("ar", share(Ar::new(ar_cfg()).unwrap())),
        ("spectral", share(Spectral::new(spectral_cfg()).unwrap())),
    ];
    let reporting: Vec<&str> = shares
        .iter()
        .filter(|(_, s)| s.is_some())
        .map(|(name, _)| *name)
        .collect();
    assert_eq!(reporting, ["ewridge", "lasso", "huber", "quantile"]);
    for (name, s) in shares {
        assert!(s.is_none() || s == Some(0.3), "{name}: {s:?}");
    }
}

/// The variants of `ModelState` this file probes, each beside the kind its
/// states name (`ModelState::kind`), which names its probe: `fn
/// <kind>_predict_is_the_step()` above. A model added to the enum and not to
/// this list fails here, and so does an entry without its probe, which is
/// the reminder to write its `*_cfg()` and probe (docs/EXTENDING.md): the
/// list was held to the enum alone, so an entry with no probe passed
/// (review 2026-10-06, CF12). The bare accumulator is no model and has no
/// probe; `every_state_kind_is_distinct_and_named` reads its kind, and
/// every model's, off a real state and holds each pair here to it.
const PROBED: &[(&str, &str)] = &[
    ("EwCov", "ew_cov_accumulator"),
    ("EwRidge", "ewridge"),
    ("Rls", "rls"),
    ("Lasso", "lasso"),
    ("Kalman", "kalman"),
    ("Robust", "robust"),
    ("Ftrl", "ftrl"),
    ("EwCovModel", "ew_cov"),
    ("Sgd", "sgd"),
    ("Pa", "pa"),
    ("Holt", "holt"),
    ("KMeans", "kmeans"),
    ("Micro", "micro"),
    ("EwClass", "ew_class"),
    ("SeqTest", "seqtest"),
    ("Marginal", "marginal"),
    ("Deco", "deco"),
    ("Rcov", "rcov"),
    ("Hmm", "hmm"),
    ("CorrChange", "corrchange"),
    ("Bocpd", "bocpd"),
    ("Audit", "audit"),
    ("Glm", "glm"),
    ("Hawkes", "hawkes"),
    ("Expectile", "expectile"),
    ("Garch", "garch"),
    ("Dcc", "dcc"),
    ("Realized", "realized"),
    ("StreamPca", "pca"),
    ("PowerVariation", "power_variation"),
    ("TDigest", "tdigest"),
    ("DdSketch", "ddsketch"),
    ("EwmMoments", "ewm_moments"),
    ("Ar", "ar"),
    ("Spectral", "spectral"),
];

#[test]
fn every_model_state_variant_is_probed_here() {
    // serde's unknown-variant error names every variant the enum has -- the
    // one place that list exists outside the enum itself.
    let err = serde_json::from_str::<ModelState>(r#"{"Nope": null}"#)
        .unwrap_err()
        .to_string();
    let quoted: Vec<&str> = err.split('`').skip(1).step_by(2).collect();
    assert_eq!(quoted[0], "Nope", "{err}");
    let variants: Vec<&str> = PROBED.iter().map(|(variant, _)| *variant).collect();
    assert_eq!(
        &quoted[1..],
        variants,
        "a ModelState variant has no contract probe"
    );
    // Each model's probes are in this file, found by its kind: the step
    // against predict, and the contract probe, which runs `probe_with` and
    // so rule 8's recursion (review S6: the roll call required only the
    // first, so a model added without the second was never held to rule 8).
    let source = contract_source();
    let src = source.as_str();
    for (variant, kind) in PROBED {
        if *kind == "ew_cov_accumulator" {
            continue;
        }
        let probe = format!("fn {kind}_predict_is_the_step()");
        assert!(
            src.contains(&probe),
            "{variant}: add `{probe}` to this file"
        );
        let contract = contract_probe_of(kind);
        let body = fn_body(src, &contract)
            .unwrap_or_else(|| panic!("{variant}: add `#[test] {contract}` to this file"));
        assert!(
            body.contains("probe_with("),
            "{variant}: `{contract}` must run `probe_with`, which holds rule 8"
        );
    }
}

/// The contract probe of a kind: `fn <kind>()`, `ew_cov`'s named apart from
/// the accumulator's.
fn contract_probe_of(kind: &str) -> String {
    match kind {
        "ew_cov" => "fn ew_cov_model()".to_owned(),
        _ => format!("fn {kind}()"),
    }
}

/// The body of the test function `signature` in `src`, from its line to the
/// closing brace at the start of a line; `None` unless it is a `#[test]`.
fn fn_body<'a>(src: &'a str, signature: &str) -> Option<&'a str> {
    let at = src.find(&format!("#[test]\n{signature} {{"))?;
    let end = src[at..].find("\n}\n")?;
    Some(&src[at..at + end])
}

#[test]
fn restoring_the_wrong_model_is_an_error_that_names_both() {
    let holt = Holt::new(HoltCfg {
        horizons: Vec::new(),
        n_targets: 1,
        level_half_life: 10.0,
        trend_half_life: 40.0,
        min_weight: 0.0,
        trend: true,
    })
    .unwrap();
    let s = holt.state();
    match Rls::restore(&s) {
        Err(StateError::WrongModel { expected, found }) => {
            assert_eq!(expected, "rls");
            assert_eq!(found, "holt");
        }
        other => panic!("expected a WrongModel error, got {other:?}"),
    }
}
