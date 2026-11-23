// ---------------------------------------------------------------------------
// The prediction at a lower floor is the lower floor's twin's (task 278, batch 17).
// ---------------------------------------------------------------------------

/// `predict_with_floor(.., floor)` on a model built with a higher
/// `min_weight` floor (and, where the model keeps one, each target's own) is,
/// on every row of the stream `predict_is_the_step_without_the_step` reads --
/// missing targets, zero-weight and uneven rows, clock gaps -- the prediction
/// of its twin built with `min_weight = floor` and fed the same rows, to the
/// bit: `min_weight` withholds the output alone, so below the higher floor
/// the two are one model. The read leaves the gated model's state as it
/// was, byte for byte. And it is evidence only where the higher floor
/// withheld something: the gated model's own `predict_with` must be NaN
/// where the twin's is a number on some rows, and on every one of those
/// this read is that number.
fn the_read_at_a_lower_floor_is_its_twin<M: OnlineModel>(
    gated: impl Fn() -> M,
    lower: impl Fn(f64) -> M,
    targets: usize,
    binary: bool,
) {
    for floor in [0.0, 4.0] {
        the_read_at_a_lower_floor_is_its_twin_at(&gated, &lower, floor, targets, binary);
    }
}

fn the_read_at_a_lower_floor_is_its_twin_at<M: OnlineModel>(
    gated: &impl Fn() -> M,
    lower: &impl Fn(f64) -> M,
    floor: f64,
    targets: usize,
    binary: bool,
) {
    let (mut g, mut f) = (gated(), lower(floor));
    let kind = g.state().model.kind();
    let mut s = 20261026u64;
    let (mut withheld, mut read) = (0usize, 0usize);
    for i in 0..400 {
        let x: Vec<f64> = (0..K).map(|_| lcg(&mut s) * 3.0).collect();
        let y: Vec<Option<f64>> = (0..targets)
            .map(|j| {
                if lcg(&mut s) > 0.8 {
                    return None;
                }
                let lin = 0.5 * (j as f64 + 1.0) + x[0] - 0.5 * x[1] + 0.1 * lcg(&mut s);
                Some(if binary { f64::from(lin > 0.5) } else { lin })
            })
            .collect();
        let u = lcg(&mut s);
        let w = if u < -0.9 {
            0.0
        } else if u < -0.5 {
            0.5
        } else if u > 0.7 {
            2.0
        } else {
            1.0
        };
        // A gap of 25 every so often takes the weight back under the floor.
        let d = match (i, lcg(&mut s)) {
            (0, _) => 0.0,
            (_, v) if v > 0.97 => 25.0,
            (_, v) if v > 0.8 => 3.0,
            _ => 1.0,
        };
        let before = state_bytes(&g);
        let past = g.predict_with_floor(&x, &y, d, floor);
        assert!(
            state_bytes(&g) == before,
            "{kind} at {floor}: row {i}: the read at a lower floor moved the state"
        );
        let twin = f.predict_with(&x, &y, d);
        assert!(
            same_bits(&past, &twin),
            "{kind} at {floor}: row {i}: read {past:?}, the twin {twin:?}"
        );
        let output = g.predict_with(&x, &y, d);
        for (slot, (o, t)) in output.pred.iter().zip(&twin.pred).enumerate() {
            if o.is_nan() && t.is_finite() {
                withheld += 1;
                read += usize::from(past.pred[slot].is_finite());
            }
        }
        g.step(&x, &y, d, w);
        f.step(&x, &y, d, w);
    }
    assert!(
        withheld > 10 && read == withheld,
        "{kind} at {floor}: the floor withheld {withheld} slots the twin reports, {read} of them read"
    );
}

/// The floor the gated model is built with: past the cfgs' 3, so it
/// withholds rows the model's own warm-up does not, and comes back after
/// each long gap; the reads are at 0 and at 4, between.
const FLOOR: f64 = 12.0;

#[test]
fn the_read_at_a_lower_floor_is_its_twin_for_every_model_with_a_residual() {
    let at = |floor: f64| {
        let mut c = ewridge_cfg();
        c.min_weight = floor;
        c
    };
    the_read_at_a_lower_floor_is_its_twin(
        || EwRidge::new(at(FLOOR)).unwrap(),
        |f| EwRidge::new(at(f)).unwrap(),
        2,
        false,
    );
    // Each target's own floor beside the shared one.
    the_read_at_a_lower_floor_is_its_twin(
        || {
            let mut m = EwRidge::new(at(FLOOR)).unwrap();
            m.set_target_min_weight(vec![FLOOR, 2.0 * FLOOR]).unwrap();
            m
        },
        |f| EwRidge::new(at(f)).unwrap(),
        2,
        false,
    );
    let at = |floor: f64| {
        let mut c = rls_cfg();
        c.min_weight = floor;
        c
    };
    the_read_at_a_lower_floor_is_its_twin(
        || Rls::new(at(FLOOR)).unwrap(),
        |f| Rls::new(at(f)).unwrap(),
        2,
        false,
    );
    let at = |floor: f64, own: Vec<f64>| {
        let mut c = lasso_cfg();
        c.min_weight = floor;
        c.target_min_weight = own;
        c
    };
    the_read_at_a_lower_floor_is_its_twin(
        || Lasso::new(at(FLOOR, Vec::new())).unwrap(),
        |f| Lasso::new(at(f, Vec::new())).unwrap(),
        1,
        false,
    );
    the_read_at_a_lower_floor_is_its_twin(
        || Lasso::new(at(FLOOR, vec![2.0 * FLOOR])).unwrap(),
        |f| Lasso::new(at(f, Vec::new())).unwrap(),
        1,
        false,
    );
    let at = |floor: f64| {
        let mut c = kalman_cfg();
        c.min_weight = floor;
        c
    };
    the_read_at_a_lower_floor_is_its_twin(
        || Kalman::new(at(FLOOR)).unwrap(),
        |f| Kalman::new(at(f)).unwrap(),
        2,
        false,
    );
    for loss in ROBUST_LOSSES {
        let at = |floor: f64| {
            let mut c = robust_cfg(loss);
            c.min_weight = floor;
            c
        };
        the_read_at_a_lower_floor_is_its_twin(
            || Robust::new(at(FLOOR)).unwrap(),
            |f| Robust::new(at(f)).unwrap(),
            2,
            false,
        );
        the_read_at_a_lower_floor_is_its_twin(
            || {
                let mut m = Robust::new(at(FLOOR)).unwrap();
                m.set_target_min_weight(vec![2.0 * FLOOR, FLOOR]).unwrap();
                m
            },
            |f| Robust::new(at(f)).unwrap(),
            2,
            false,
        );
    }
    let at = |floor: f64| {
        let mut c = ftrl_cfg();
        c.min_weight = floor;
        c
    };
    the_read_at_a_lower_floor_is_its_twin(
        || Ftrl::new(at(FLOOR)).unwrap(),
        |f| Ftrl::new(at(f)).unwrap(),
        2,
        false,
    );
    for logistic in [false, true] {
        let at = |floor: f64| {
            let mut c = sgd_cfg();
            c.min_weight = floor;
            if logistic {
                c.loss = SgdLoss::Logistic;
            }
            c
        };
        the_read_at_a_lower_floor_is_its_twin(
            || Sgd::new(at(FLOOR)).unwrap(),
            |f| Sgd::new(at(f)).unwrap(),
            2,
            logistic,
        );
    }
    let at = |floor: f64| {
        let mut c = pa_cfg();
        c.min_weight = floor;
        c
    };
    the_read_at_a_lower_floor_is_its_twin(
        || Pa::new(at(FLOOR)).unwrap(),
        |f| Pa::new(at(f)).unwrap(),
        2,
        false,
    );
    let at = |floor: f64| {
        let mut c = holt_cfg();
        c.min_weight = floor;
        c
    };
    the_read_at_a_lower_floor_is_its_twin(
        || Holt::new(at(FLOOR)).unwrap(),
        |f| Holt::new(at(f)).unwrap(),
        2,
        false,
    );
}

/// Task 281 (mutants): a model with no floor of its own keeps the default
/// `predict_with_floor`, which is its `predict_with` whatever the floor: the
/// same answer to the bit, read on rows where it predicts something.
fn the_default_read_at_a_floor_is_the_plain_read<M: OnlineModel>(mut m: M, targets: usize) {
    let kind = m.state().model.kind();
    let mut s = 20261027u64;
    let mut read = 0usize;
    for i in 0..200 {
        let x: Vec<f64> = (0..K).map(|_| lcg(&mut s) * 3.0).collect();
        let y: Vec<Option<f64>> = (0..targets)
            .map(|j| Some(0.5 * (j as f64 + 1.0) + x[0] - 0.5 * x[1] + 0.1 * lcg(&mut s)))
            .collect();
        let d = if i == 0 { 0.0 } else { 1.0 };
        for floor in [0.0, 4.0] {
            let at = m.predict_with_floor(&x, &y, d, floor);
            let plain = m.predict_with(&x, &y, d);
            assert!(
                same_bits(&at, &plain),
                "{kind}: row {i}, floor {floor}: {at:?} against {plain:?}"
            );
            read += usize::from(at.pred.iter().any(|v| v.is_finite()));
        }
        m.step(&x, &y, d, 1.0);
    }
    assert!(read > 100, "{kind}: {read} reads predicted something");
}

#[test]
fn a_model_with_no_floor_reads_at_a_floor_as_it_reads() {
    the_default_read_at_a_floor_is_the_plain_read(SeqTest::new(seqtest_cfg()).unwrap(), 2);
}

#[test]
fn seqtest_predict_is_the_step() {
    predict_is_the_step_without_the_step(|| SeqTest::new(seqtest_cfg()).unwrap(), 2, false);
}

#[test]
fn kmeans_predict_is_the_step() {
    predict_is_the_step_without_the_step(|| KMeans::new(kmeans_cfg()).unwrap(), 0, false);
    let cfg = KMeansCfg {
        update_every_rows: 7,
        seed_rule: SeedRule::Farthest,
        standardize: false,
        ..kmeans_cfg()
    };
    predict_is_the_step_without_the_step(move || KMeans::new(cfg.clone()).unwrap(), 0, false);
}

#[test]
fn micro_predict_is_the_step() {
    predict_is_the_step_without_the_step(|| Micro::new(micro_cfg()).unwrap(), 0, false);
    let cfg = MicroCfg {
        macro_link: Some(2.5),
        standardize: false,
        max_clusters: 6,
        ..micro_cfg()
    };
    predict_is_the_step_without_the_step(move || Micro::new(cfg.clone()).unwrap(), 0, false);
}

#[test]
fn ew_class_predict_is_the_step() {
    for covariance in [Covariance::Full, Covariance::Shared, Covariance::Diagonal] {
        let cfg = EwClassCfg {
            covariance,
            ..ew_class_cfg()
        };
        predict_is_the_step_without_the_step(move || EwClass::new(cfg.clone()).unwrap(), 1, true);
    }
}

#[test]
fn ew_cov_predict_is_the_step() {
    predict_is_the_step_without_the_step(|| EwCovModel::new(ew_cov_model_cfg()).unwrap(), 0, false);
}

#[test]
fn marginal_predict_is_the_step() {
    // No slots to compare, so this holds `n_eff` and `extra` alone -- and
    // that `predict` did not move the state.
    predict_is_the_step_without_the_step(|| Marginal::new(marginal_cfg()).unwrap(), 2, false);
}

#[test]
fn bocpd_predict_is_the_step() {
    predict_is_the_step_without_the_step(|| Bocpd::new(bocpd_cfg()).unwrap(), 0, true);
}

/// [`bocpd_cfg`] with its prior left to the first rows (docs/PLAN.md task
/// 195, U4): the warm-up holds `d + 2` learned rows, reporting nothing, then
/// reads them as rows of the prior they set.
fn bocpd_from_the_data_cfg() -> BocpdCfg {
    BocpdCfg {
        prior_mean: None,
        prior_scale: None,
        prior_nu: None,
        ..bocpd_cfg()
    }
}

/// Inside the warm-up `predict` reports nothing, as the step does; after
/// it, the step's number.
#[test]
fn bocpd_from_the_data_predict_is_the_step() {
    predict_is_the_step_without_the_step(
        || Bocpd::new(bocpd_from_the_data_cfg()).unwrap(),
        0,
        true,
    );
}

/// On the clock `predict` applies the step's chance before it reads the
/// row, as the step does; and the zero-weight contracts this runs hold a
/// row of weight 0 to the stream without it, its step carried into the
/// next row, which on the clock is a transition that composes.
#[test]
fn bocpd_on_the_clock_predict_is_the_step() {
    predict_is_the_step_without_the_step(|| Bocpd::new(bocpd_on_the_clock_cfg()).unwrap(), 0, true);
}

#[test]
#[ignore = "extended: a second or more: 30,000 rows of recovery (9.5 s); the clause stays for the other models"]
fn bocpd_recovers_from_bounded_extremes() {
    // A 1e100 row is a changepoint by any reading, and the posterior is
    // path-dependent, so a clean twin is not the criterion. What must hold
    // is a finite state, a run posterior that stays a distribution, and a
    // model that goes on reporting.
    let rows = bounded_script(0);
    let mut m = Bocpd::new(bocpd_cfg()).unwrap();
    for r in &rows {
        let step = m.step(&r.x, &r.y, 1.0, r.w);
        assert!(step.n_eff.is_finite());
        let p = m.run_posterior();
        assert!(p.iter().all(|v| v.is_finite() && (0.0..=1.0).contains(v)));
        assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-9);
    }
    assert!(m.run_posterior().len() <= 200, "max_run holds");
}

#[test]
fn corrchange_predict_is_the_step() {
    predict_is_the_step_without_the_step(|| CorrChange::new(corrchange_cfg()).unwrap(), 0, true);
    // The scalar mode ages its standardiser on a zero-weight row where the
    // pairwise mode has none to age (review S12: that line was in no
    // configuration the contract ran).
    let scalar = || {
        CorrChange::new(CorrChangeCfg {
            scalar: true,
            ..corrchange_cfg()
        })
        .unwrap()
    };
    predict_is_the_step_without_the_step(scalar, 0, true);
}

#[test]
fn corrchange_recovers_from_bounded_extremes() {
    // A test of the data, not a model of it: what must hold is that the
    // state stays finite and it goes on testing. A statistic computed over
    // a span containing a 1e100 row is whatever it is; the twin, which
    // never saw that row, is testing a different span.
    let rows = bounded_script(0);
    let mut m = CorrChange::new(corrchange_cfg()).unwrap();
    let mut reported = 0;
    for r in &rows {
        let step = m.step(&r.x, &r.y, 1.0, r.w);
        assert!(step.n_eff.is_finite());
        reported += usize::from(step.pred[0].is_finite());
    }
    assert!(reported > 100, "spans go on closing");
    assert!(m.n_eff().is_finite());
}

#[test]
fn hmm_predict_is_the_step() {
    predict_is_the_step_without_the_step(|| Hmm::new(hmm_cfg()).unwrap(), 0, true);
}

/// `bocpd`'s hazard and `hmm`'s exogenous value ride in the **targets**
/// slot, which `predict` does not see: it answered from the configured
/// default instead and disagreed with the step on every row where the
/// column differed from it. `predict_with` is the one the plumbing calls
/// (docs/REVIEW-E54-E64.md C1, B3, H2).
#[test]
fn a_value_in_the_targets_slot_reaches_predict() {
    let mut s = 20260906u64;
    let mut bo = Bocpd::new(BocpdCfg {
        hazard_from_row: true,
        ..bocpd_cfg()
    })
    .unwrap();
    let a = vec![2.0, -2.0, -2.0, 2.0];
    let b = vec![1.5, -1.5, -1.5, 1.5];
    let mut hm = Hmm::new(HmmCfg {
        tvtp: Some((a, b)),
        ..hmm_cfg()
    })
    .unwrap();
    // The row's value has to *matter*, or the parity below is vacuous:
    // count the rows where ignoring it gives a different answer.
    let (mut bo_moved, mut hm_moved) = (0usize, 0usize);
    // A hazard between 2 and 1000, never the configured 50; an exogenous
    // value swinging either side of the default 0.
    let (mut hazards, mut exogenous) = (Vec::new(), Vec::new());
    for i in 0..300 {
        let x: Vec<f64> = (0..K).map(|_| lcg(&mut s) * 3.0).collect();
        let hz = vec![Some(2.0 + 499.0 * (1.0 + lcg(&mut s)))];
        let z = vec![Some(3.0 * lcg(&mut s))];
        let d = if i == 0 { 0.0 } else { 1.0 };
        hazards.push((x.clone(), hz, d));
        exogenous.push((x, z, d));
    }
    for i in 0..300 {
        let ((x, hz, d), (_, z, _)) = (&hazards[i], &exogenous[i]);
        let d = *d;
        // A value there that is not usable is an absent one, as a target
        // that is not usable is (task 183): the configured hazard, the
        // default exogenous value. At the head of the stream, and warm.
        if matches!(i, 0 | 150) {
            let next = i + 1..i + 13;
            refuses_unusable_values(&bo, &hazards[i], &hazards[next.clone()], "bocpd", None);
            refuses_unusable_values(&hm, &exogenous[i], &exogenous[next], "hmm", None);
        }

        let p = bo.predict_with(x, hz, d);
        bo_moved += usize::from(!same_pred(&p, &bo.predict(x, d)));
        same_step("bocpd", i, &p, &bo.step(x, hz, d, 1.0));

        let p = hm.predict_with(x, z, d);
        hm_moved += usize::from(!same_pred(&p, &hm.predict(x, d)));
        same_step("hmm", i, &p, &hm.step(x, z, d, 1.0));
    }
    assert!(
        bo_moved > 200,
        "bocpd: the hazard moved only {bo_moved} rows"
    );
    assert!(
        hm_moved > 200,
        "hmm: the exogenous value moved only {hm_moved} rows"
    );
}

/// Whether two steps report the same slots; `same_step` asserts it.
fn same_pred(a: &Step, b: &Step) -> bool {
    a.pred.len() == b.pred.len()
        && a.pred
            .iter()
            .zip(&b.pred)
            .all(|(x, y)| x == y || (x.is_nan() && y.is_nan()))
}

#[test]
fn hmm_recovers_from_bounded_extremes() {
    // Agreement with a clean twin is not a property a *filter* has: `p` is
    // path-dependent by construction, and one extra row -- even one the
    // filter rejects, because both densities underflowed -- leaves the two
    // copies on different paths through a sticky chain. What the contract
    // asks of every model is that the state stays finite and it goes on
    // learning, and what is a property here is the **fit**: the state means
    // return to the twin's once the extreme row has decayed away.
    let rows = bounded_script(0);
    let mut m = Hmm::new(hmm_cfg()).unwrap();
    for r in &rows {
        let step = m.step(&r.x, &r.y, 1.0, r.w);
        assert!(step.n_eff.is_finite());
        for v in &step.pred {
            assert!(!v.is_nan() || step.pred.iter().all(|p| p.is_nan()));
        }
        // `p` is a distribution on every row it is reported.
        let p = &step.pred[..2];
        if p.iter().all(|v| v.is_finite()) {
            assert!(p.iter().all(|v| (0.0..=1.0).contains(v)));
            assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        }
    }
    // A clean twin was stepped through this whole script and then discarded:
    // agreement is not a property a filter has (the comment above), so the
    // twin proved nothing and ran a full-covariance two-state filter for
    // ~30 500 rows on every `cargo test` for nothing (review 2026-09-18, T2).
    // Every state's moments are finite, and the model goes on filtering.
    for s in 0..2 {
        assert!(m.state_means(s).iter().all(|v| v.is_finite()));
        assert!(
            m.state_cov(s)
                .unwrap()
                .comoments()
                .to_full()
                .iter()
                .all(|v| v.is_finite())
        );
    }
    // **A known limitation, and the reason this is not a twin test.** A
    // bounded-extreme row is captured by whichever state wins it, and that
    // state's mean moves to ~1e98. In mean form a state with zero
    // responsibility keeps its moments -- `EwCov::update` at weight 0 moves
    // nothing -- so a state that stops winning never forgets, and the
    // mixture is left with one live state. `docs/PLAN.md` §11a records it
    // with the mitigations. What must still hold is that the survivor
    // tracks the data: fed a clean two-blob stream, the filter's `state`
    // output follows the blob it is in.
    let mut right = 0;
    let rows = 2000;
    let mut seed = 991u64;
    for i in 0..rows {
        let g = (i / 50) % 2;
        let c = if g == 0 { -6.0 } else { 6.0 };
        let x: Vec<f64> = (0..K).map(|_| c + lcg(&mut seed)).collect();
        let step = m.step(&x, &[], 1.0, 1.0);
        let state = step.pred[2 * 2];
        if state.is_finite() && i > rows / 2 {
            // Either labelling of the two blobs is correct; count the one
            // that is consistent.
            right += usize::from((state as usize == g) == (m.filtered()[g] >= 0.5));
        }
    }
    assert!(right > 0, "the filter reports a state on a clean stream");
    assert!(
        m.state_means(0).iter().all(|v| v.is_finite())
            && m.state_means(1).iter().all(|v| v.is_finite())
    );
}

/// A Poisson `glm` whose refit fills on its tenth learned row, inside the
/// probes' streams, so the contract holds the buffer, the refit and the
/// recursion after it.
fn glm_cfg() -> GlmCfg {
    GlmCfg {
        n_features: K,
        n_targets: 2,
        fit_intercept: true,
        decay: decay(),
        family: GlmFamily::Poisson,
        link: GlmLink::Log,
        var_power: 1.5,
        ridge: 1.0,
        min_weight: 3.0,
        refit_rows: 10,
        refit_clock: None,
        step_cap: f64::INFINITY,
        max_state_bytes: None,
    }
}

fn hawkes_cfg() -> HawkesCfg {
    HawkesCfg {
        n_targets: 2,
        kernel_decays: vec![1.0, 4.0],
        decay: decay(),
        min_weight: 10.0,
        window_events: None,
        window_clock: None,
        closed: WindowClosed::Right,
        max_state_bytes: None,
    }
}

/// A window of a few rows, so the probes run through its turnovers; a
/// window takes no decay, and its `n_eff` is the weight inside it, at most
/// the six rows', so a `min_weight` of 3.
fn hawkes_windowed_cfg() -> HawkesCfg {
    HawkesCfg {
        decay: Decay::Halflife(f64::INFINITY),
        window_events: Some(6),
        min_weight: 3.0,
        ..hawkes_cfg()
    }
}

/// The same over a span of the clock: the probes' rows are a unit apart, so
/// a window of 12.5 units holds thirteen rows.
fn hawkes_clock_cfg() -> HawkesCfg {
    HawkesCfg {
        decay: Decay::Halflife(f64::INFINITY),
        window_clock: Some(12.5),
        min_weight: 3.0,
        ..hawkes_cfg()
    }
}

#[test]
fn glm() {
    let r = probe_with(Glm::new(glm_cfg()).unwrap(), 2, Some(&Glm::n_eff));
    check(&r, "glm", 2, 1);
    // The cap's path, with no refit.
    let c = GlmCfg {
        refit_rows: 0,
        step_cap: 1.0,
        family: GlmFamily::Gamma,
        ..glm_cfg()
    };
    let r = probe_with(Glm::new(c).unwrap(), 2, Some(&Glm::n_eff));
    check(&r, "glm", 2, 1);
}

#[test]
fn glm_predict_is_the_step() {
    predict_is_the_step_without_the_step(|| Glm::new(glm_cfg()).unwrap(), 2, false);
}

/// An upper expectile whose first rows' refit fills on its tenth learned
/// row, inside the probes' streams, so the contract holds the buffer, the
/// refit and the one pass after it.
fn expectile_cfg() -> ExpectileCfg {
    ExpectileCfg {
        n_features: K,
        n_targets: 2,
        fit_intercept: true,
        decay: decay(),
        expectile: 0.8,
        ridge: 1e-6,
        min_weight: 3.0,
        refit_rows: 10,
        refit_clock: None,
        window_rows: None,
        window_clock: None,
        closed: WindowClosed::Right,
        max_state_bytes: None,
    }
}

/// A window of a dozen rows, so the probes run through its turnovers; a
/// window takes no decay.
fn expectile_rows_cfg() -> ExpectileCfg {
    ExpectileCfg {
        decay: Decay::Halflife(f64::INFINITY),
        refit_rows: 0,
        window_rows: Some(12),
        ..expectile_cfg()
    }
}

/// The same over a span of the clock: the probes' rows are a unit apart.
fn expectile_clock_cfg() -> ExpectileCfg {
    ExpectileCfg {
        decay: Decay::Halflife(f64::INFINITY),
        refit_rows: 0,
        window_clock: Some(12.5),
        ..expectile_cfg()
    }
}

#[test]
fn expectile() {
    let r = probe_with(
        Expectile::new(expectile_cfg()).unwrap(),
        2,
        Some(&Expectile::n_eff),
    );
    check(&r, "expectile", 2, 1);
    // The one pass with no refit, at the lower tail.
    let c = ExpectileCfg {
        refit_rows: 0,
        expectile: 0.1,
        ..expectile_cfg()
    };
    let r = probe_with(Expectile::new(c).unwrap(), 2, Some(&Expectile::n_eff));
    check(&r, "expectile", 2, 1);
}

#[test]
fn expectile_predict_is_the_step() {
    predict_is_the_step_without_the_step(|| Expectile::new(expectile_cfg()).unwrap(), 2, false);
    predict_is_the_step_without_the_step(
        || Expectile::new(expectile_rows_cfg()).unwrap(),
        2,
        false,
    );
    predict_is_the_step_without_the_step(
        || Expectile::new(expectile_clock_cfg()).unwrap(),
        2,
        false,
    );
}

#[test]
fn hawkes() {
    // A point process: one node per target, no features, one intensity a
    // node, and `n_eff` every row's weight on the shared recursion.
    let m = Hawkes::new(hawkes_cfg()).unwrap();
    assert_eq!(m.n_features(), 0, "hawkes reads no features");
    let r = probe_with(m, 2, Some(&Hawkes::n_eff));
    assert_eq!(r.n_features, 0);
    assert_eq!(r.kind, "hawkes");
    assert_eq!(r.n_targets, 2);
    assert_eq!(r.n_outputs, 2);
    assert_eq!(r.pred_len, 2);
    assert_eq!(r.n_eff[0], 0.0);
    assert_eq!(r.n_eff[1], 1.0);
    assert!((r.n_eff[2] - (0.5f64.powf(1.0 / HALFLIFE) + 1.0)).abs() < 1e-12);
    assert!((r.after_gap - (r.before_gap * 0.5f64.powi(10) + 1.0)).abs() < 1e-9);
    assert!(r.roundtrips);
    // Under a window `n_eff` is the weight inside it (hard rule 8): the six
    // rows' of a window of six, the thirteen of a span of 12.5 units a step
    // apart (the clocks 28 to 40); and a gap neither decays it nor adds a row the window has not
    // room for.
    let r = probe_windowed(
        Hawkes::new(hawkes_windowed_cfg()).unwrap(),
        2,
        Some(&Hawkes::n_eff),
    );
    assert_eq!(r.n_eff, vec![0.0, 1.0, 2.0, 6.0]);
    assert_eq!(r.after_gap, r.before_gap, "a gap does not move the window");
    assert!(r.roundtrips);
    // A gap past the clock window's span takes every row out of it, at the
    // row the gap ends on; only that row is in the weight the next reports.
    let r = probe_windowed(
        Hawkes::new(hawkes_clock_cfg()).unwrap(),
        2,
        Some(&Hawkes::n_eff),
    );
    assert_eq!(r.n_eff, vec![0.0, 1.0, 2.0, 13.0]);
    assert_eq!(r.before_gap, 13.0);
    assert_eq!(r.after_gap, 1.0, "the gap emptied the window");
    assert!(r.roundtrips);
}

/// One series after a start of 50 rows, with a forecast three rows ahead;
/// a cap too small to double the start, so the stream reports from its 51st
/// learned row.
fn garch_cfg() -> GarchCfg {
    GarchCfg {
        n_targets: 1,
        vol: GarchVol::Garch,
        dist: GarchDist::Normal,
        truncation: None,
        variance_targeting: false,
        start_rows: 50,
        start_span: None,
        horizon: 3,
        min_weight: 0.0,
        max_state_bytes: Some(1700),
    }
}

#[test]
fn garch() {
    // A variance per target, no features: the variance, the standardized
    // residual and the forecast a target, and `n_eff` every learned row's
    // weight, summed (no decay: the recursion is per observation).
    let m = Garch::new(garch_cfg()).unwrap();
    assert_eq!(m.n_features(), 0, "garch reads no features");
    let r = probe_with(m, 1, Some(&Garch::n_eff));
    assert_eq!(r.kind, "garch");
    assert_eq!(r.n_targets, 1);
    assert_eq!(r.n_outputs, 3);
    assert_eq!(r.pred_len, 3);
    assert!(r.roundtrips);
    for (vol, dist) in [
        (GarchVol::Gjr, GarchDist::T),
        (GarchVol::Garch, GarchDist::Skewt),
        (GarchVol::Egarch, GarchDist::Normal),
        (GarchVol::Figarch, GarchDist::Normal),
    ] {
        let c = GarchCfg {
            vol,
            dist,
            horizon: if vol == GarchVol::Egarch { 1 } else { 3 },
            truncation: (vol == GarchVol::Figarch).then_some(30),
            start_rows: if vol == GarchVol::Figarch { 2000 } else { 50 },
            n_targets: 2,
            max_state_bytes: None,
            ..garch_cfg()
        };
        let r = probe_with(Garch::new(c).unwrap(), 2, Some(&Garch::n_eff));
        assert!(r.roundtrips, "{vol:?} {dist:?}");
    }
}

/// Two targets read at three levels, the tails withheld until the rows
/// identify them (`level_identified`).
fn tdigest_cfg() -> TDigestModelCfg {
    TDigestModelCfg {
        n_targets: 2,
        compression: 20.0,
        levels: vec![0.05, 0.25, 0.95],
        max_state_bytes: None,
    }
}

#[test]
fn tdigest() {
    // A digest per target, no features: per target one quantile a level,
    // then one tail mean a level, and `n_eff` every learned row's weight, summed (no decay: a
    // merging digest cannot take a row out).
    let m = TDigestModel::new(tdigest_cfg()).unwrap();
    assert_eq!(m.n_features(), 0, "tdigest reads no features");
    let r = probe_with(m, 2, Some(&TDigestModel::n_eff));
    assert_eq!(r.kind, "tdigest");
    assert_eq!(r.n_targets, 2);
    assert_eq!(r.n_outputs, 12);
    assert_eq!(r.pred_len, 12);
    assert!(r.roundtrips);
}

#[test]
fn tdigest_predict_is_the_step() {
    predict_is_the_step_without_the_step(|| TDigestModel::new(tdigest_cfg()).unwrap(), 2, false);
}

#[test]
fn garch_predict_is_the_step() {
    predict_is_the_step_without_the_step(|| Garch::new(garch_cfg()).unwrap(), 1, false);
    predict_is_the_step_without_the_step(
        || {
            Garch::new(GarchCfg {
                vol: GarchVol::Gjr,
                dist: GarchDist::T,
                ..garch_cfg()
            })
            .unwrap()
        },
        1,
        false,
    );
    predict_is_the_step_without_the_step(
        || {
            Garch::new(GarchCfg {
                vol: GarchVol::Egarch,
                horizon: 1,
                ..garch_cfg()
            })
            .unwrap()
        },
        1,
        false,
    );
    predict_is_the_step_without_the_step(
        || {
            Garch::new(GarchCfg {
                variance_targeting: true,
                ..garch_cfg()
            })
            .unwrap()
        },
        1,
        false,
    );
}

/// Two series, the asymmetric model with its parameters given after a start
/// of 50 rows, decayed at the contract's half-life, the targets taken as
/// standardized residuals (no margins). The contract's stream holds one
/// correlation, which identifies no recursion, so a learned model's start
/// withholds its fit there: [`dcc`] probes that one too.
fn dcc_cfg() -> DccCfg {
    DccCfg {
        n_targets: 2,
        asymmetric: true,
        margins: None,
        params: Some(vec![0.03, 0.9, 0.04]),
        decay: Decay::Halflife(HALFLIFE),
        start_rows: 50,
        start_span: None,
        min_weight: 0.0,
        max_state_bytes: None,
    }
}

#[test]
fn dcc() {
    // A correlation per pair, no features: the pair's correlation and the
    // row's log-density, and `n_eff` the decayed weight (rule 8).
    let m = Dcc::new(dcc_cfg()).unwrap();
    assert_eq!(m.n_features(), 0, "dcc reads no features");
    let r = probe_with(m, 2, Some(&Dcc::n_eff));
    assert_eq!(r.kind, "dcc");
    assert_eq!(r.n_targets, 2);
    assert_eq!(r.n_outputs, 2);
    assert_eq!(r.pred_len, 2);
    assert!(r.roundtrips);
    // learned: its start withheld on this stream, its contract kept
    let c = DccCfg {
        params: None,
        ..dcc_cfg()
    };
    let r = probe_with(Dcc::new(c).unwrap(), 2, Some(&Dcc::n_eff));
    assert!(r.roundtrips);
    // with GARCH margins and given parameters, three series
    let c = DccCfg {
        n_targets: 3,
        asymmetric: false,
        params: Some(vec![0.05, 0.9]),
        margins: Some(GarchCfg {
            start_rows: 50,
            horizon: 1,
            n_targets: 3,
            max_state_bytes: None,
            ..garch_cfg()
        }),
        ..dcc_cfg()
    };
    let r = probe_with(Dcc::new(c).unwrap(), 3, Some(&Dcc::n_eff));
    assert_eq!(r.n_outputs, 7);
    assert!(r.roundtrips);
}

#[test]
fn dcc_predict_is_the_step() {
    // With margins the start is the margins' and the stage's, too long for
    // the parity's 300 ready rows: `dcc/tests.rs` holds that form.
    predict_is_the_step_without_the_step(|| Dcc::new(dcc_cfg()).unwrap(), 2, false);
}

/// `ewm_moments` of the first feature: a slot a term, every kind and the
/// unbiased forms among them, and a floor on the weight.
fn ewm_moments_cfg() -> EwmMomentsCfg {
    let term = MomentTerm::of_column;
    EwmMomentsCfg {
        min_weight: 3.0,
        ..EwmMomentsCfg::new(
            decay(),
            vec![
                term("var", 2, MomentKind::Central, false),
                term("skew", 3, MomentKind::Standardized, true),
                term("kurt_u", 4, MomentKind::Standardized, false),
                term("k5", 5, MomentKind::Cumulant, true),
                term("raw2", 2, MomentKind::Raw, true),
            ],
        )
    }
}

#[test]
fn ewm_moments() {
    // One column, the first feature; no targets; a slot a term; `n_eff`
    // rule 8's decayed recursion.
    let m = EwmMoments::new(ewm_moments_cfg()).unwrap();
    assert_eq!(m.n_features(), 1, "ewm_moments reads one column");
    assert_eq!(m.n_targets(), 0);
    let r = probe_with(m, 0, Some(&EwmMoments::n_eff));
    assert_eq!(r.kind, "ewm_moments");
    assert_eq!(r.n_outputs, 5);
    assert_eq!(r.pred_len, 5);
    assert!(r.roundtrips);
}

#[test]
fn ewm_moments_predict_is_the_step() {
    predict_is_the_step_without_the_step(|| EwmMoments::new(ewm_moments_cfg()).unwrap(), 0, false);
}

#[test]
fn hawkes_predict_is_the_step() {
    predict_is_the_step_without_the_step(|| Hawkes::new(hawkes_cfg()).unwrap(), 2, false);
    predict_is_the_step_without_the_step(|| Hawkes::new(hawkes_windowed_cfg()).unwrap(), 2, false);
    predict_is_the_step_without_the_step(|| Hawkes::new(hawkes_clock_cfg()).unwrap(), 2, false);
}

#[test]
#[ignore = "extended: a second or more: 30,000 rows of recovery; the clause stays for the other models"]
fn hawkes_recovers_from_bounded_extremes() {
    // An event of mark 1e100 is one the decays forget: `R` at its kernels'
    // rates, `b` and `Q` at the half-life.
    recovers_from_bounded_extremes(
        || Hawkes::new(hawkes_cfg()).unwrap(),
        2,
        Recovery::Twin(1e-9),
    );
}

#[test]
fn audit_predict_is_the_step() {
    // No slots, so this holds `n_eff` and that `predict` moved nothing; the
    // zero-weight rows are held to add no weight, which is all a weight does
    // here (the values on them are counted, by design).
    predict_is_the_step_without_the_step(|| Audit::new(audit_cfg()).unwrap(), 0, false);
}

fn audit_cfg() -> AuditCfg {
    AuditCfg {
        n_columns: K,
        pairs: true,
        distinct_cap: 16,
        gap_cap: Some(5.0),
        has_clock: true,
    }
}

#[test]
fn audit() {
    // A record of the rows: no targets, no output slots, no decay. `n_eff`
    // is every row's weight summed, so a clock gap changes nothing, and the
    // values that are not usable are counted, not refused.
    let m = Audit::new(audit_cfg()).unwrap();
    assert_eq!(m.n_targets(), 0);
    assert_eq!(m.n_features(), K);
    assert_eq!(m.n_outputs(), 0, "audit reports nothing per row");
    let r = probe_with(m, 0, Some(&Audit::n_eff));
    assert_eq!(r.kind, "audit");
    assert_eq!(r.pred_len, 0);
    assert_eq!(r.n_eff, vec![0.0, 1.0, 2.0, 40.0]);
    assert_eq!(r.after_gap, r.before_gap + 1.0, "no decay over a gap");
    assert!(r.roundtrips);
}

#[test]
fn rcov_predict_is_the_step() {
    // No slots, so this holds `n_eff` and that `predict` moved nothing.
    predict_is_the_step_without_the_step(|| Rcov::new(rcov_cfg()).unwrap(), 0, false);
}

#[test]
fn rcov_recovers_from_bounded_extremes() {
    // A 1e100 return dominates the sums for good -- there is no decay to
    // wash it out -- so the twin comparison is not the contract here; what
    // is, is that the state stays finite and the model goes on accepting
    // rows.
    let rows = bounded_script(0);
    let mut m = Rcov::new(rcov_cfg()).unwrap();
    for r in &rows {
        let step = m.step(&r.x, &r.y, 1.0, r.w);
        assert!(step.n_eff.is_finite());
    }
    let e = m.estimate();
    assert!(e.rcov.expect("a block").iter().all(|v| !v.is_nan()));
}

#[test]
fn deco_predict_is_the_step() {
    predict_is_the_step_without_the_step(|| Deco::new(deco_cfg()).unwrap(), 0, true);
}

#[test]
fn every_model_with_a_recovery_test_has_a_predict_parity_test() {
    // The recovery tests above are the per-model roll call of this file; each
    // `<model>_recovers_from_bounded_extremes` must have its
    // `<model>_predict_is_the_step` twin.
    let source = contract_source();
    let src = source.as_str();
    let models: Vec<&str> = src
        .lines()
        .filter_map(|l| {
            l.strip_prefix("fn ")?
                .strip_suffix("_recovers_from_bounded_extremes() {")
        })
        .collect();
    assert!(models.len() >= 10, "found only {models:?}");
    for model in models {
        let name = format!("fn {model}_predict_is_the_step()");
        assert!(
            src.contains(&name),
            "{model}: add `{name}` to the predict parity tests"
        );
    }
}
