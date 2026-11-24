#[test]
fn every_state_kind_is_distinct_and_named() {
    // `ModelState::kind` names the model in every state error; a mutation that
    // returns a constant would make "expected X, found Y" meaningless.
    let kinds = [
        "ewridge",
        "rls",
        "lasso",
        "kalman",
        "robust",
        "ftrl",
        "sgd",
        "pa",
        "holt",
        "ew_cov",
        "kmeans",
        "micro",
        "ew_class",
        "seqtest",
        "marginal",
        "deco",
        "rcov",
        "hmm",
        "corrchange",
        "bocpd",
        "audit",
        "glm",
        "hawkes",
        "expectile",
        "garch",
        "dcc",
        "realized",
        "pca",
        "power_variation",
        "tdigest",
        "ddsketch",
        "ar",
        "spectral",
        "sarima",
    ];
    let mut seen = std::collections::HashSet::new();
    for k in kinds {
        assert!(!k.is_empty());
        assert!(seen.insert(k), "duplicate kind {k}");
    }
    // And the names really come from the states, not this list: every one
    // read off a real model's state, where four of them were (review
    // 2026-09-18, T2; docs/PLAN.md task 112). `robust` covers both of its
    // losses, and the two `kalman` configurations are one kind.
    let states = [
        EwRidge::new(ewridge_cfg()).unwrap().state(),
        Rls::new(rls_cfg()).unwrap().state(),
        Lasso::new(lasso_cfg()).unwrap().state(),
        Kalman::new(kalman_cfg()).unwrap().state(),
        Robust::new(robust_cfg(ROBUST_LOSSES[0])).unwrap().state(),
        Ftrl::new(ftrl_cfg()).unwrap().state(),
        Sgd::new(sgd_cfg()).unwrap().state(),
        Pa::new(pa_cfg()).unwrap().state(),
        Holt::new(holt_cfg()).unwrap().state(),
        EwCovModel::new(ew_cov_model_cfg()).unwrap().state(),
        KMeans::new(kmeans_cfg()).unwrap().state(),
        Micro::new(micro_cfg()).unwrap().state(),
        EwClass::new(ew_class_cfg()).unwrap().state(),
        SeqTest::new(seqtest_cfg()).unwrap().state(),
        Marginal::new(marginal_cfg()).unwrap().state(),
        Deco::new(deco_cfg()).unwrap().state(),
        Rcov::new(rcov_cfg()).unwrap().state(),
        Hmm::new(hmm_cfg()).unwrap().state(),
        CorrChange::new(corrchange_cfg()).unwrap().state(),
        Bocpd::new(bocpd_cfg()).unwrap().state(),
        Audit::new(audit_cfg()).unwrap().state(),
        Glm::new(glm_cfg()).unwrap().state(),
        Hawkes::new(hawkes_cfg()).unwrap().state(),
        Expectile::new(expectile_cfg()).unwrap().state(),
        Garch::new(garch_cfg()).unwrap().state(),
        Dcc::new(dcc_cfg()).unwrap().state(),
        Realized::new(realized_cfg()).unwrap().state(),
        StreamPca::new(pca_cfg()).unwrap().state(),
        PowerVariation::new(power_variation_cfg()).unwrap().state(),
        TDigestModel::new(tdigest_cfg()).unwrap().state(),
        DdSketchModel::new(ddsketch_cfg()).unwrap().state(),
        Ar::new(ar_cfg()).unwrap().state(),
        Spectral::new(spectral_cfg()).unwrap().state(),
        Sarima::new(sarima_cfg()).unwrap().state(),
    ];
    let from_states: Vec<&str> = states.iter().map(|s| s.model.kind()).collect();
    assert_eq!(from_states, kinds);
    // Each state's variant beside its kind is a pair `PROBED` lists, so the
    // probe found by the kind is the variant's (review 2026-10-06, CF12).
    let bare = State::new(ModelState::EwCov(Box::new(EwCov::new(1))));
    for s in states.iter().chain([&bare]) {
        let debug = format!("{:?}", s.model);
        let variant = debug.split('(').next().unwrap();
        assert!(
            PROBED.contains(&(variant, s.model.kind())),
            "{variant} names {} and PROBED pairs it otherwise",
            s.model.kind()
        );
    }
    assert_eq!(
        Robust::new(robust_cfg(ROBUST_LOSSES[1]))
            .unwrap()
            .state()
            .model
            .kind(),
        "robust"
    );
    assert_eq!(
        Kalman::new(kalman_revert_cfg())
            .unwrap()
            .state()
            .model
            .kind(),
        "kalman"
    );
    // The accumulator an `ew_cov` state nests has a kind of its own.
    assert_eq!(
        State::new(ModelState::EwCov(Box::new(EwCov::new(1))))
            .model
            .kind(),
        "ew_cov_accumulator"
    );
}

/// One state of every kind a bank runs, and the bare accumulator.
fn one_state_of_each_kind() -> Vec<State> {
    vec![
        EwRidge::new(ewridge_cfg()).unwrap().state(),
        Rls::new(rls_cfg()).unwrap().state(),
        Lasso::new(lasso_cfg()).unwrap().state(),
        Kalman::new(kalman_cfg()).unwrap().state(),
        Robust::new(robust_cfg(ROBUST_LOSSES[0])).unwrap().state(),
        Ftrl::new(ftrl_cfg()).unwrap().state(),
        Sgd::new(sgd_cfg()).unwrap().state(),
        Pa::new(pa_cfg()).unwrap().state(),
        Holt::new(holt_cfg()).unwrap().state(),
        EwCovModel::new(ew_cov_model_cfg()).unwrap().state(),
        KMeans::new(kmeans_cfg()).unwrap().state(),
        Micro::new(micro_cfg()).unwrap().state(),
        EwClass::new(ew_class_cfg()).unwrap().state(),
        SeqTest::new(seqtest_cfg()).unwrap().state(),
        Marginal::new(marginal_cfg()).unwrap().state(),
        Deco::new(deco_cfg()).unwrap().state(),
        Rcov::new(rcov_cfg()).unwrap().state(),
        Hmm::new(hmm_cfg()).unwrap().state(),
        CorrChange::new(corrchange_cfg()).unwrap().state(),
        Bocpd::new(bocpd_cfg()).unwrap().state(),
        Audit::new(audit_cfg()).unwrap().state(),
        Glm::new(glm_cfg()).unwrap().state(),
        Hawkes::new(hawkes_cfg()).unwrap().state(),
        Expectile::new(expectile_cfg()).unwrap().state(),
        Garch::new(garch_cfg()).unwrap().state(),
        Dcc::new(dcc_cfg()).unwrap().state(),
        Realized::new(realized_cfg()).unwrap().state(),
        StreamPca::new(pca_cfg()).unwrap().state(),
        PowerVariation::new(power_variation_cfg()).unwrap().state(),
        TDigestModel::new(tdigest_cfg()).unwrap().state(),
        DdSketchModel::new(ddsketch_cfg()).unwrap().state(),
        Ar::new(ar_cfg()).unwrap().state(),
        Spectral::new(spectral_cfg()).unwrap().state(),
        Sarima::new(sarima_cfg()).unwrap().state(),
        State::new(ModelState::EwCov(Box::new(EwCov::new(1)))),
    ]
}

/// `M::restore` takes its own kind's state and refuses every other with a
/// `WrongModel` naming the kind it wanted and the kind it was handed.
fn refuses_every_other_kind<M: OnlineModel>(own: &str, states: &[State]) {
    let mut took_own = false;
    for s in states {
        let found = s.model.kind();
        match M::restore(s) {
            Ok(_) => {
                assert_eq!(found, own, "{own} accepted a {found} state");
                took_own = true;
            }
            Err(StateError::WrongModel {
                expected,
                found: named,
            }) => {
                assert_ne!(found, own, "{own} refused its own state");
                assert_eq!((expected, named), (own, found), "{own} on a {found} state");
            }
            Err(e) => panic!("{own} on a {found} state: {e}"),
        }
    }
    assert!(took_own, "{own}: no state of its own kind in the list");
}

/// Task 252: each model's `restore` refuses every other model's state by
/// name -- the arm no test reached in most of them, so a mutation that
/// accepted a foreign state, or named the wrong kind, went unseen.
#[test]
fn every_model_refuses_every_other_models_state_by_name() {
    let states = one_state_of_each_kind();
    refuses_every_other_kind::<EwRidge>("ewridge", &states);
    refuses_every_other_kind::<Rls>("rls", &states);
    refuses_every_other_kind::<Lasso>("lasso", &states);
    refuses_every_other_kind::<Kalman>("kalman", &states);
    refuses_every_other_kind::<Robust>("robust", &states);
    refuses_every_other_kind::<Ftrl>("ftrl", &states);
    refuses_every_other_kind::<Sgd>("sgd", &states);
    refuses_every_other_kind::<Pa>("pa", &states);
    refuses_every_other_kind::<Holt>("holt", &states);
    refuses_every_other_kind::<EwCovModel>("ew_cov", &states);
    refuses_every_other_kind::<KMeans>("kmeans", &states);
    refuses_every_other_kind::<Micro>("micro", &states);
    refuses_every_other_kind::<EwClass>("ew_class", &states);
    refuses_every_other_kind::<SeqTest>("seqtest", &states);
    refuses_every_other_kind::<Marginal>("marginal", &states);
    refuses_every_other_kind::<Deco>("deco", &states);
    refuses_every_other_kind::<Rcov>("rcov", &states);
    refuses_every_other_kind::<Hmm>("hmm", &states);
    refuses_every_other_kind::<CorrChange>("corrchange", &states);
    refuses_every_other_kind::<Bocpd>("bocpd", &states);
    refuses_every_other_kind::<Audit>("audit", &states);
    refuses_every_other_kind::<Glm>("glm", &states);
    refuses_every_other_kind::<Expectile>("expectile", &states);
    refuses_every_other_kind::<TDigestModel>("tdigest", &states);
    refuses_every_other_kind::<DdSketchModel>("ddsketch", &states);
}

/// Task 252: each model hands back the configuration it was built from,
/// and a restored model the one its state carried.
#[test]
fn every_model_hands_back_its_configuration() {
    fn same<M: OnlineModel, C: PartialEq + std::fmt::Debug>(m: M, cfg: &C, of: impl Fn(&M) -> &C) {
        assert_eq!(of(&m), cfg);
        let back = M::restore(&m.state()).unwrap();
        assert_eq!(of(&back), cfg);
    }
    let c = ewridge_cfg();
    same(EwRidge::new(c.clone()).unwrap(), &c, EwRidge::cfg);
    let c = rls_cfg();
    same(Rls::new(c.clone()).unwrap(), &c, Rls::cfg);
    let c = lasso_cfg();
    same(Lasso::new(c.clone()).unwrap(), &c, Lasso::cfg);
    let c = kalman_cfg();
    same(Kalman::new(c.clone()).unwrap(), &c, Kalman::cfg);
    for loss in ROBUST_LOSSES {
        let c = robust_cfg(loss);
        same(Robust::new(c.clone()).unwrap(), &c, Robust::cfg);
    }
    let c = ftrl_cfg();
    same(Ftrl::new(c.clone()).unwrap(), &c, Ftrl::cfg);
    let c = sgd_cfg();
    same(Sgd::new(c.clone()).unwrap(), &c, Sgd::cfg);
    let c = pa_cfg();
    same(Pa::new(c.clone()).unwrap(), &c, Pa::cfg);
    let c = holt_cfg();
    same(Holt::new(c.clone()).unwrap(), &c, Holt::cfg);
    let c = ew_cov_model_cfg();
    same(EwCovModel::new(c.clone()).unwrap(), &c, EwCovModel::cfg);
    let c = kmeans_cfg();
    same(KMeans::new(c.clone()).unwrap(), &c, KMeans::cfg);
    let c = micro_cfg();
    same(Micro::new(c.clone()).unwrap(), &c, Micro::cfg);
    let c = ew_class_cfg();
    same(EwClass::new(c.clone()).unwrap(), &c, EwClass::cfg);
    let c = seqtest_cfg();
    same(SeqTest::new(c.clone()).unwrap(), &c, SeqTest::cfg);
    let c = marginal_cfg();
    same(Marginal::new(c.clone()).unwrap(), &c, Marginal::cfg);
    let c = deco_cfg();
    same(Deco::new(c.clone()).unwrap(), &c, Deco::cfg);
    let c = rcov_cfg();
    same(Rcov::new(c.clone()).unwrap(), &c, Rcov::cfg);
    let c = hmm_cfg();
    same(Hmm::new(c.clone()).unwrap(), &c, Hmm::cfg);
    let c = corrchange_cfg();
    same(CorrChange::new(c.clone()).unwrap(), &c, CorrChange::cfg);
    let c = bocpd_cfg();
    same(Bocpd::new(c.clone()).unwrap(), &c, Bocpd::cfg);
    let c = audit_cfg();
    same(Audit::new(c.clone()).unwrap(), &c, Audit::cfg);
    let c = glm_cfg();
    same(Glm::new(c.clone()).unwrap(), &c, Glm::cfg);
    let c = hawkes_cfg();
    same(Hawkes::new(c.clone()).unwrap(), &c, Hawkes::cfg);
    let c = expectile_cfg();
    same(Expectile::new(c.clone()).unwrap(), &c, Expectile::cfg);
    let c = garch_cfg();
    same(Garch::new(c.clone()).unwrap(), &c, Garch::cfg);
    let c = dcc_cfg();
    same(Dcc::new(c.clone()).unwrap(), &c, Dcc::cfg);
    let c = realized_cfg();
    same(Realized::new(c.clone()).unwrap(), &c, Realized::cfg);
    let c = pca_cfg();
    same(StreamPca::new(c.clone()).unwrap(), &c, StreamPca::cfg);
    let c = power_variation_cfg();
    same(
        PowerVariation::new(c.clone()).unwrap(),
        &c,
        PowerVariation::cfg,
    );
    let c = tdigest_cfg();
    same(TDigestModel::new(c.clone()).unwrap(), &c, TDigestModel::cfg);
    let c = ddsketch_cfg();
    same(
        DdSketchModel::new(c.clone()).unwrap(),
        &c,
        DdSketchModel::cfg,
    );
    let c = ar_cfg();
    same(Ar::new(c.clone()).unwrap(), &c, Ar::cfg);
    let c = spectral_cfg();
    same(Spectral::new(c.clone()).unwrap(), &c, Spectral::cfg);
    let c = sarima_cfg();
    same(Sarima::new(c.clone()).unwrap(), &c, Sarima::cfg);
}
