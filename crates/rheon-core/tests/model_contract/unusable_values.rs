// ---------------------------------------------------------------------------
// A value that is not usable, refused by every model by one rule (the
// `OnlineModel` docs; docs/PLAN.md task 183).
// ---------------------------------------------------------------------------

/// The values a model reads as missing (`rheon_core::usable`): not a
/// number, either infinity, and either sign past the input bound.
const UNUSABLE: [f64; 5] = [
    f64::NAN,
    f64::INFINITY,
    f64::NEG_INFINITY,
    2.0 * INPUT_BOUND,
    -2.0 * INPUT_BOUND,
];

/// A row: the features, the targets and the clock since the row before.
type Row3 = (Vec<f64>, Vec<Option<f64>>, f64);

fn state_bytes<M: OnlineModel>(m: &M) -> Vec<u8> {
    rmp_serde::to_vec(&m.state()).unwrap()
}

/// The same numbers to the bit, NaN for NaN.
fn same_number(u: f64, v: f64) -> bool {
    u.to_bits() == v.to_bits() || (u.is_nan() && v.is_nan())
}

/// Two steps that report the same thing to the bit, NaN for NaN: every
/// slot, `n_eff` and `extra`.
fn same_bits(a: &Step, b: &Step) -> bool {
    let same = |u: &[f64], v: &[f64]| {
        u.len() == v.len() && u.iter().zip(v).all(|(p, q)| same_number(*p, *q))
    };
    let extra = match (&a.extra, &b.extra) {
        (Some(Extra::Lasso { lam_selected: u }), Some(Extra::Lasso { lam_selected: v })) => {
            same(u, v)
        }
        (Some(Extra::Huber { scale: u }), Some(Extra::Huber { scale: v })) => same(u, v),
        (Some(Extra::VectorAr { values: u }), Some(Extra::VectorAr { values: v })) => same(u, v),
        (u, v) => u == v,
    };
    a.pred.len() == b.pred.len()
        && a.pred.iter().zip(&b.pred).all(|(u, v)| same_number(*u, *v))
        && same_number(a.n_eff, b.n_eff)
        && extra
}

/// `step_into` on one kept buffer, reset to another row's answer every
/// third row, against `step` on a twin, over 400 rows of the stream
/// [`predict_is_the_step_without_the_step`] reads (missing targets,
/// zero-weight and uneven rows, clock gaps): the same answer to the bit and
/// the same state, byte for byte, on every row, of which at least a hundred
/// report every slot.
fn step_into_is_step_on_a_stream<M: OnlineModel>(
    build: impl Fn() -> M,
    targets: usize,
    binary: bool,
) {
    let (mut m, mut twin) = (build(), build());
    let kind = m.state().model.kind();
    let mut kept = Step::default();
    let mut s = 20260902u64;
    let mut ready = 0usize;
    // `garch` models a variance, and its start waits on a target with none
    // (PLAN 548): its targets are each a GARCH(1,1)'s returns at (0.05,
    // 0.3, 0.65), each row's variance and return kept here.
    let mut returns = vec![(1.0f64, 0.0f64); targets];
    for i in 0..400 {
        let x: Vec<f64> = (0..K).map(|_| lcg(&mut s) * 3.0).collect();
        let y: Vec<Option<f64>> = (0..targets)
            .map(|j| {
                if lcg(&mut s) > 0.8 {
                    return None;
                }
                if kind == "garch" {
                    let (v, e) = &mut returns[j];
                    *v = 0.05 + 0.3 * *e * *e + 0.65 * *v;
                    let z = (0..12).map(|_| 0.5 * lcg(&mut s)).sum::<f64>();
                    *e = v.sqrt() * z;
                    return Some(*e);
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
        let d = match (i, lcg(&mut s)) {
            (0, _) => 0.0,
            (_, v) if v > 0.95 => 25.0,
            (_, v) if v > 0.8 => 3.0,
            _ => 1.0,
        };
        let step = m.step(&x, &y, d, w);
        if i % 3 == 0 {
            kept = stale_step(i / 3);
        }
        twin.step_into(&x, &y, d, w, &mut kept);
        assert!(
            same_bits(&kept, &step),
            "{kind}: row {i}: step_into wrote {kept:?}, step returned {step:?}"
        );
        assert!(
            state_bytes(&twin) == state_bytes(&m),
            "{kind}: row {i}: step_into left another state than step"
        );
        ready += usize::from(step.pred.iter().all(|v| v.is_finite()));
    }
    assert!(
        ready >= 100,
        "{kind}: only {ready} rows reported every slot"
    );
}

/// A buffer holding another row's answer, of the wrong length and with an
/// `extra` of the wrong length, for [`OnlineModel::step_into`] to overwrite
/// whole: `lasso`'s variant for an even `which`, `huber`'s for an odd one,
/// so a model that reuses its own variant's vector meets it stale as well
/// as the other's (task 285).
fn stale_step(which: usize) -> Step {
    let extra = if which.is_multiple_of(2) {
        Extra::Lasso {
            lam_selected: vec![0.5; 3],
        }
    } else {
        Extra::Huber {
            scale: vec![-1.5; 5],
        }
    };
    Step {
        pred: vec![7.25; 13],
        n_eff: -3.5,
        extra: Some(extra),
    }
}

/// [`OnlineModel::step_into`] on a [`stale_step`] against `step` on a copy:
/// the same answer to the bit, and the same state, byte for byte.
fn step_into_is_step<M: OnlineModel + Clone>(
    m: &M,
    (x, y, d, w): (&[f64], &[Option<f64>], f64, f64),
    kind: &str,
    what: &str,
) {
    let (mut a, mut b) = (m.clone(), m.clone());
    let want = a.step(x, y, d, w);
    let mut got = stale_step(0);
    b.step_into(x, y, d, w, &mut got);
    assert!(
        same_bits(&got, &want),
        "{kind}: {what}: step_into wrote {got:?}, step returned {want:?}"
    );
    assert!(
        state_bytes(&a) == state_bytes(&b),
        "{kind}: {what}: step_into left another state than step"
    );
}

/// Nothing reported: NaN in every slot, and in every number of `extra`.
fn reports_nothing(s: &Step) -> bool {
    s.pred.iter().all(|v| v.is_nan())
        && match &s.extra {
            None => true,
            Some(Extra::Lasso { lam_selected }) => lam_selected.iter().all(|v| v.is_nan()),
            Some(Extra::VectorAr { values }) => values.iter().all(|v| v.is_nan()),
            Some(_) => false,
        }
}

/// The rule for a value that is not usable, checked on `m` as it stands
/// before `row`, with `after` the rows that follow it:
///
/// - **A feature**, in each position, as each of [`UNUSABLE`]: `step`,
///   `predict` and `predict_with` report nothing, and `n_eff` as on any row
///   (the accessor's, where `n_eff_of` is given); the state is the one the
///   same row leaves at weight 0 with that feature as it was -- byte for
///   byte, and so the same whatever the value and wherever it stands -- and
///   the rows after it report, to the bit, what they report after that row.
/// - **A target** is an absent one: `Some(bad)` and `None` leave the same
///   state and report the same step, `predict_with`'s included.
/// - **A weight**: the row reports what `predict_with` does, and leaves the
///   state a row with a feature that is not usable leaves.
///
/// One model is excepted, by design: `audit` counts such a value
/// ([`COUNTS_UNUSABLE`]). No other model's zero-weight row takes anything
/// from the row's features into its state -- no ring slot, warm-up row or likelihood
/// -- so the refused row, which takes none of it, leaves that row's state.
/// A blocked `ewridge` Gram, which holds a zero-weight row in its block, is
/// the one place one does, and is tested in `ewridge`'s own file.
fn refuses_unusable_values<M: OnlineModel + Clone>(
    m: &M,
    (x, y, d): &Row3,
    after: &[Row3],
    kind: &str,
    n_eff_of: Option<&dyn Fn(&M) -> f64>,
) {
    if COUNTS_UNUSABLE.contains(&kind) {
        return;
    }
    let d = *d;
    // The same row at weight 0, and the rows after it from there.
    let mut reference = m.clone();
    let zero = reference.step(x, y, d, 0.0);
    let zero_state = state_bytes(&reference);
    let goes_on_as_the_reference = |mut a: M, what: &str| {
        let mut b = reference.clone();
        for (i, (x, y, d)) in after.iter().enumerate() {
            let (sa, sb) = (a.step(x, y, *d, 1.0), b.step(x, y, *d, 1.0));
            assert!(
                same_bits(&sa, &sb),
                "{kind}: {what}: row {i} after it reports {sa:?}, and {sb:?} after the row at \
                 weight 0"
            );
        }
    };

    let mut refused_state: Option<Vec<u8>> = None;
    for pos in 0..m.n_features() {
        for bad in UNUSABLE {
            let what = format!("feature {pos} = {bad:e}");
            let mut xb = x.clone();
            xb[pos] = bad;
            let mut c = m.clone();
            let s = c.step(&xb, y, d, 1.0);
            assert!(
                reports_nothing(&s),
                "{kind}: {what}: the step reported {s:?}"
            );
            step_into_is_step(m, (&xb, y, d, 1.0), kind, &what);
            for (how, p) in [
                ("predict", m.predict(&xb, d)),
                ("predict_with", m.predict_with(&xb, y, d)),
            ] {
                assert!(
                    same_bits(&p, &s),
                    "{kind}: {what}: {how} reported {p:?}, the step {s:?}"
                );
            }
            assert!(
                same_number(s.n_eff, zero.n_eff),
                "{kind}: {what}: n_eff {}, and {} on any row",
                s.n_eff,
                zero.n_eff
            );
            if let Some(n_eff) = n_eff_of {
                assert_eq!(
                    s.n_eff,
                    n_eff(m),
                    "{kind}: {what}: n_eff against the accessor"
                );
            }
            let state = state_bytes(&c);
            assert!(
                state == zero_state,
                "{kind}: {what}: the state is not the one the row leaves at weight 0"
            );
            match &refused_state {
                Some(first) => assert!(
                    *first == state,
                    "{kind}: {what}: the state depends on the value or where it stands"
                ),
                None => refused_state = Some(state),
            }
            goes_on_as_the_reference(c, &what);
        }
    }

    for j in 0..y.len() {
        for bad in UNUSABLE {
            let what = format!("target {j} = Some({bad:e})");
            let (mut yb, mut yn) = (y.clone(), y.clone());
            yb[j] = Some(bad);
            yn[j] = None;
            let (pb, pn) = (m.predict_with(x, &yb, d), m.predict_with(x, &yn, d));
            assert!(
                same_bits(&pb, &pn),
                "{kind}: {what}: predict_with reported {pb:?}, and {pn:?} for None"
            );
            step_into_is_step(m, (x, &yb, d, 1.0), kind, &what);
            let (mut b, mut n) = (m.clone(), m.clone());
            let (sb, sn) = (b.step(x, &yb, d, 1.0), n.step(x, &yn, d, 1.0));
            assert!(
                same_bits(&sb, &sn),
                "{kind}: {what}: the step reported {sb:?}, and {sn:?} for None"
            );
            assert!(
                state_bytes(&b) == state_bytes(&n),
                "{kind}: {what}: the state is not the one None leaves"
            );
        }
    }

    // A model that reads no features is held to the row at weight 0.
    let refused_state = refused_state.unwrap_or(zero_state);
    for bad in UNUSABLE {
        let what = format!("weight {bad:e}");
        let p = m.predict_with(x, y, d);
        step_into_is_step(m, (x, y, d, bad), kind, &what);
        let mut c = m.clone();
        let s = c.step(x, y, d, bad);
        assert!(
            same_bits(&p, &s),
            "{kind}: {what}: the step reported {s:?}, predict_with {p:?}"
        );
        assert!(
            state_bytes(&c) == refused_state,
            "{kind}: {what}: the state is not the one a row with a feature that is not \
             usable leaves"
        );
        goes_on_as_the_reference(c, &what);
    }
}

/// [`refuses_unusable_values`] at the head of a stream, at its fourth row
/// (inside every warm-up the configurations here keep) and on a warm model
/// at row 60: the stream of [`zero_weight_rows_leave_no_nan`],
/// with twelve rows after each checked one.
fn unusable_values_are_refused<M: OnlineModel + Clone>(
    build: &impl Fn() -> M,
    targets: usize,
    binary: bool,
    kind: &'static str,
) {
    let mut s = 20261006u64;
    let rows: Vec<Row3> = (0..80usize)
        .map(|i| {
            let x: Vec<f64> = (0..K).map(|_| lcg(&mut s) * 3.0).collect();
            let y = (0..targets)
                .map(|j| {
                    let lin = 0.5 * (j as f64 + 1.0) + x[0] - 0.5 * x[1];
                    Some(if binary { f64::from(lin > 0.5) } else { lin })
                })
                .collect();
            let d = match i {
                0 => 0.0,
                _ if i.is_multiple_of(5) => 3.0,
                _ => 1.0,
            };
            (x, y, d)
        })
        .collect();
    let mut m = build();
    for (i, (x, y, d)) in rows.iter().enumerate() {
        if matches!(i, 0 | 3 | 60) {
            refuses_unusable_values(&m, &rows[i], &rows[i + 1..i + 13], kind, None);
        }
        m.step(x, y, *d, 1.0);
    }
}

#[test]
fn ewridge_predict_is_the_step() {
    predict_is_the_step_without_the_step(|| EwRidge::new(ewridge_cfg()).unwrap(), 2, false);
    let mut cfg = ewridge_cfg();
    cfg.standardize = true;
    cfg.session_shrink = Some(0.5);
    cfg.long_half_life = Some(4.0 * HALFLIFE);
    predict_is_the_step_without_the_step(move || EwRidge::new(cfg.clone()).unwrap(), 2, false);
    // A lazily refreshed solve: both read the cached coefficients.
    let mut cfg = ewridge_cfg();
    cfg.solve_every = 5.0;
    cfg.max_rows_between_solves = 10_000;
    predict_is_the_step_without_the_step(move || EwRidge::new(cfg.clone()).unwrap(), 2, false);
}

#[test]
fn rls_predict_is_the_step() {
    predict_is_the_step_without_the_step(|| Rls::new(rls_cfg()).unwrap(), 2, false);
}

#[test]
fn lasso_predict_is_the_step() {
    predict_is_the_step_without_the_step(|| Lasso::new(lasso_cfg()).unwrap(), 1, false);
    // With a selection half-life `lam_selected` moves; `extra` must match too.
    let mut cfg = lasso_cfg();
    cfg.lasso_path = vec![1.0, 0.1, 0.01, 0.0];
    cfg.select_half_life = Some(HALFLIFE);
    predict_is_the_step_without_the_step(move || Lasso::new(cfg.clone()).unwrap(), 1, false);
    // A solve on the clock, which a zero-weight row must not fire (hard rule
    // 9; docs/PLAN.md task 214: the fit moved 4.2e-2).
    let mut cfg = lasso_cfg();
    cfg.solve_every = 5.0;
    cfg.max_rows_between_solves = 10_000;
    predict_is_the_step_without_the_step(move || Lasso::new(cfg.clone()).unwrap(), 1, false);
}

#[test]
fn kalman_predict_is_the_step() {
    for standardize in [true, false] {
        for reverting in [false, true] {
            // Under reversion the prediction depends on `d`: `predict(x, d)`
            // has to propagate the mean by the same `phi_i(d)` the step does.
            let mut cfg = if reverting {
                kalman_revert_cfg()
            } else {
                kalman_cfg()
            };
            cfg.standardize = standardize;
            predict_is_the_step_without_the_step(
                move || Kalman::new(cfg.clone()).unwrap(),
                2,
                false,
            );
        }
    }
}

/// The same where a reversion gap takes the covariance to exactly 0 (review
/// fable2, C6): reversion half-lives of 0.02 to 0.04 against the stream's
/// gaps of 25 are 625 to 1,250 half-lives, past the underflow of `P·phi²`
/// (about 540) and of the factor's `C·phi` (about 1,075); its gaps of 3
/// and 1 fall short of it. The step withholds on such a row, the
/// covariance unsized, where `predict` read it before the transition and
/// gave a number. In both forms, one half-life for every slot and one each,
/// which couples the intercept to the slopes.
#[test]
fn kalman_predict_is_the_step_where_reversion_underflows() {
    for form in [CovarianceForm::Covariance, CovarianceForm::SquareRoot] {
        for revert in [vec![0.04], vec![0.04, 0.03, 0.02], vec![0.02, 0.04, 0.03]] {
            let mut cfg = kalman_cfg();
            cfg.revert_half_life = revert;
            predict_is_the_step_without_the_step(
                move || {
                    Kalman::new(cfg.clone())
                        .unwrap()
                        .with_covariance_form(form)
                        .unwrap()
                },
                2,
                false,
            );
        }
    }
}

#[test]
fn robust_predict_is_the_step() {
    for loss in ROBUST_LOSSES {
        for standardize in [false, true] {
            let mut cfg = robust_cfg(loss);
            cfg.standardize = standardize;
            predict_is_the_step_without_the_step(
                move || Robust::new(cfg.clone()).unwrap(),
                2,
                false,
            );
        }
        // A solve on the clock, which a zero-weight row must not fire (hard
        // rule 9; docs/PLAN.md task 214: the fit moved 8.4e-8 under huber,
        // 1.0e-7 under quantile).
        let mut cfg = robust_cfg(loss);
        cfg.solve_every = 5.0;
        cfg.max_rows_between_solves = 10_000;
        predict_is_the_step_without_the_step(move || Robust::new(cfg.clone()).unwrap(), 2, false);
    }
    // The MAD's sketch, aged and fed beside the RMS's variance: a
    // zero-weight row ages it and adds nothing (hard rules 8 and 9).
    for standardize in [false, true] {
        let mut cfg = huber_mad_cfg();
        cfg.standardize = standardize;
        predict_is_the_step_without_the_step(move || Robust::new(cfg.clone()).unwrap(), 2, false);
    }
}

/// `huber` with its scale emitted, in both scales, with and without a
/// scaler: the step into a kept buffer reuses the `Extra::Huber` vector it
/// left there, and meets it stale and of the wrong length, and still
/// reports `step`'s bits and leaves its state (task 285).
#[test]
fn robust_step_into_is_step_with_the_emitted_scale() {
    for base in [robust_cfg(ROBUST_LOSSES[0]), huber_mad_cfg()] {
        for standardize in [false, true] {
            let cfg = RobustCfg {
                emit_scale: true,
                standardize,
                ..base.clone()
            };
            step_into_is_step_on_a_stream(move || Robust::new(cfg.clone()).unwrap(), 2, false);
        }
    }
}

#[test]
fn ftrl_predict_is_the_step() {
    // `l1 > 0` so some proximal weights sit at exactly zero.
    let mut cfg = ftrl_cfg();
    cfg.l1 = 0.05;
    predict_is_the_step_without_the_step(move || Ftrl::new(cfg.clone()).unwrap(), 2, false);
    let mut cfg = ftrl_cfg();
    cfg.loss = FtrlLoss::Logistic;
    predict_is_the_step_without_the_step(move || Ftrl::new(cfg.clone()).unwrap(), 2, true);
    // Under `standardize` (PLAN 569) the prediction reads the moments with
    // the row admitted, as the step does.
    let mut cfg = ftrl_cfg();
    cfg.l1 = 0.05;
    cfg.standardize = true;
    predict_is_the_step_without_the_step(move || Ftrl::new(cfg.clone()).unwrap(), 2, false);
}

#[test]
fn sgd_predict_is_the_step() {
    for standardize in [false, true] {
        let mut cfg = sgd_cfg();
        cfg.standardize = standardize;
        predict_is_the_step_without_the_step(move || Sgd::new(cfg.clone()).unwrap(), 2, false);
    }
    let mut cfg = sgd_cfg();
    cfg.loss = SgdLoss::Logistic;
    predict_is_the_step_without_the_step(move || Sgd::new(cfg.clone()).unwrap(), 2, true);
    for cfg in sgd_huber_cfgs() {
        predict_is_the_step_without_the_step(move || Sgd::new(cfg.clone()).unwrap(), 2, false);
    }
}

/// `sgd`'s own `step_into` (task 270 (h)) under every loss (Huber in both
/// scales), every schedule, with and without a constraint, a scaler and an
/// intercept: [`step_into_is_step_on_a_stream`] holds a twin stepped into a
/// kept buffer to `step`'s bits and state, row by row. The parity alone:
/// `sgd_predict_is_the_step` runs the whole contract on the settings it
/// holds; `sgd_and_pa_zero_weight_rows_under_every_setting` runs its
/// zero-weight rows on this grid.
#[test]
fn sgd_step_into_is_step_under_every_setting() {
    for (cfg, binary) in sgd_settings() {
        step_into_is_step_on_a_stream(move || Sgd::new(cfg.clone()).unwrap(), 2, binary);
    }
    for standardize in [false, true] {
        let cfg = SgdCfg {
            fit_intercept: false,
            standardize,
            ..sgd_cfg()
        };
        step_into_is_step_on_a_stream(move || Sgd::new(cfg.clone()).unwrap(), 2, false);
    }
}

/// `sgd` under every loss (Huber in both scales), every schedule, with and
/// without a simplex and a scaler, `l2` 0.01; and whether its labels are
/// binary.
fn sgd_settings() -> Vec<(SgdCfg, bool)> {
    let [huber_rms, huber_mad] = sgd_huber_cfgs();
    let losses = [
        (sgd_cfg(), false),
        (huber_rms, false),
        (huber_mad, false),
        (
            SgdCfg {
                loss: SgdLoss::Quantile { tau: 0.3 },
                ..sgd_cfg()
            },
            false,
        ),
        (
            SgdCfg {
                loss: SgdLoss::EpsilonInsensitive { eps: 0.1 },
                ..sgd_cfg()
            },
            false,
        ),
        (
            SgdCfg {
                loss: SgdLoss::Poisson,
                ..sgd_cfg()
            },
            false,
        ),
        (
            SgdCfg {
                loss: SgdLoss::Logistic,
                ..sgd_cfg()
            },
            true,
        ),
    ];
    let schedules = [
        LearningRate::Constant,
        LearningRate::InvScaling { power: 0.25 },
        LearningRate::AdaGrad,
    ];
    let mut out = Vec::new();
    for (base, binary) in losses {
        for schedule in schedules {
            for constraint in [None, Some(simplex())] {
                for standardize in [false, true] {
                    let cfg = SgdCfg {
                        schedule,
                        constraint: constraint.clone(),
                        standardize,
                        l2: 0.01,
                        ..base.clone()
                    };
                    out.push((cfg, binary));
                }
            }
        }
    }
    out
}

/// The simplex over the `K` features: each weight at least 0, summing to 1.
fn simplex() -> Constraint {
    Constraint {
        lo: vec![0.0; K],
        hi: vec![f64::INFINITY; K],
        sum: Some(1.0),
    }
}

/// `pa` with and without a simplex and a scaler.
fn pa_settings() -> Vec<PaCfg> {
    let mut out = Vec::new();
    for constraint in [None, Some(simplex())] {
        for standardize in [false, true] {
            out.push(PaCfg {
                constraint: constraint.clone(),
                standardize,
                ..pa_cfg()
            });
        }
    }
    out
}

/// Hard rule 9 over every setting of `sgd` and `pa` (task 270 (h)), which
/// the contract ran on a few of them: a zero-weight row advances the clock
/// and teaches nothing, and one past the underflow forgets as the decay
/// does. Two defects hid outside the few. Under `inv_scaling` the rate read
/// the weight before the row's own decay, so a zero-weight row in between
/// moved the next row's rate (up to 1.6e-2 relative at half-life 1). And
/// under a scaler with a constraint or AdaGrad, a zero-weight row one
/// half-life short of the underflow left the scaler a history of
/// subnormal weight, whose spread the next row's step divided by: every
/// coefficient NaN for good.
#[test]
fn sgd_and_pa_zero_weight_rows_under_every_setting() {
    for (cfg, binary) in sgd_settings() {
        // `poisson` at a level of 1,000 chases targets near 500 through an
        // exponential link with its gradient clipped, and amplifies
        // rounding past this test's 1e-9: one ulp more in one feature on
        // one row moved its fit by up to 3.6e-7 relative 66 rows later,
        // the same order as the zero-weight rows (up to 6.0e-7, under a
        // constant rate, a simplex and a scaler), so there the comparison
        // measures the path's conditioning rather than the rule. It is
        // compared at a level of 0.
        let levels: &[f64] = if cfg.loss == SgdLoss::Poisson {
            &[0.0]
        } else {
            &[0.0, 1e3, -1e3]
        };
        let build = move || Sgd::new(cfg.clone()).unwrap();
        for &level in levels {
            zero_weight_rows_leave_no_nan(&build, 2, binary, "sgd", level);
        }
        a_zero_weight_row_past_the_underflow_forgets(&build, 2, binary, "sgd");
    }
    for cfg in pa_settings() {
        let build = move || Pa::new(cfg.clone()).unwrap();
        for level in [0.0, 1e3, -1e3] {
            zero_weight_rows_leave_no_nan(&build, 2, false, "pa", level);
        }
        a_zero_weight_row_past_the_underflow_forgets(&build, 2, false, "pa");
    }
}

/// A scaler's history aged below the normal range of the row it meets is
/// forgotten, as one the decay takes to exactly 0 is (task 270 (h)-A, the
/// rule `lasso`'s Grams keep, `gaps::decay_into`). A zero-weight row
/// `g` half-lives on, for every `g` from 1022 to 1075, under every setting
/// of `sgd` and `pa` and at half-lives 1 and 20: no prediction or
/// coefficient after the gap is NaN, and wherever the history the scaler
/// carries into the first row after the gap is in the subnormal range of
/// that row, every prediction, coefficient and `n_eff` is the bits of the
/// stream whose gap was 1075 half-lives, where the decay itself is 0 (and a
/// prediction that stream makes, every gap makes). Before
/// the fix a gap of 1074 half-lives left every coefficient NaN for good
/// under a scaler with a constraint or AdaGrad.
#[test]
fn a_scaler_history_aged_below_the_normal_range_is_forgotten() {
    type Run = Vec<(Vec<f64>, Vec<Vec<f64>>, f64)>;
    fn run<M: OnlineModel>(
        build: &impl Fn(f64) -> M,
        coefs: &impl Fn(&M) -> Vec<Vec<f64>>,
        half_life: f64,
        binary: bool,
        half_lives: f64,
    ) -> (Run, f64) {
        let mut m = build(half_life);
        let mut s = 2701u64;
        let row = |s: &mut u64| {
            let x: Vec<f64> = (0..K).map(|_| 3.0 * lcg(s)).collect();
            let y: Vec<Option<f64>> = (0..2)
                .map(|j| {
                    let lin = 0.5 * (j as f64 + 1.0) + x[0] - 0.5 * x[1] + 0.1 * lcg(s);
                    Some(if binary { f64::from(lin > 0.5) } else { lin })
                })
                .collect();
            (x, y)
        };
        for i in 0..40 {
            let (x, y) = row(&mut s);
            m.step(&x, &y, if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        let (x, y) = row(&mut s);
        m.step(&x, &y, half_life * half_lives, 0.0);
        // The weight the scaler carries into the next row, which is the
        // model's own: both age `lam·W + w` from 0 on the same rows.
        let carried = Decay::Halflife(half_life).factor(1.0) * m.predict(&[0.0; K], 0.0).n_eff;
        let out = (0..40)
            .map(|_| {
                let (x, y) = row(&mut s);
                let st = m.step(&x, &y, 1.0, 1.0);
                (st.pred, coefs(&m), st.n_eff)
            })
            .collect();
        (out, carried)
    }
    fn check<M: OnlineModel>(
        what: &str,
        build: impl Fn(f64) -> M,
        coefs: impl Fn(&M) -> Vec<Vec<f64>>,
        binary: bool,
    ) -> usize {
        let mut forgotten = 0;
        for half_life in [1.0, 20.0] {
            let (reference, _) = run(&build, &coefs, half_life, binary, 1075.0);
            for g in 1022..=1075 {
                let (got, carried) = run(&build, &coefs, half_life, binary, f64::from(g));
                for (i, ((pred, coef, _), (want, _, _))) in got.iter().zip(&reference).enumerate() {
                    // A prediction the stream that forgot makes, this one
                    // makes too: `poisson` refuses a negative count, so its
                    // own weight may not reach `min_weight` on every row.
                    assert!(
                        coef.iter().flatten().all(|c| c.is_finite())
                            && pred
                                .iter()
                                .zip(want)
                                .all(|(p, q)| p.is_finite() || !q.is_finite()),
                        "{what}, half-life {half_life}, a gap of {g} half-lives: row {i} \
                         after it, predictions {pred:?}, coefficients {coef:?}"
                    );
                }
                if carried <= f64::MIN_POSITIVE {
                    forgotten += 1;
                    let bits = |r: &Run| -> Vec<u64> {
                        r.iter()
                            .skip(1)
                            .flat_map(|(p, c, n)| {
                                p.iter()
                                    .chain(c.iter().flatten())
                                    .chain([n])
                                    .map(|v| v.to_bits())
                            })
                            .collect::<Vec<_>>()
                    };
                    let (u, v) = (bits(&got), bits(&reference));
                    let first = u.iter().zip(&v).position(|(a, b)| a != b);
                    assert!(
                        u.len() == v.len() && first.is_none(),
                        "{what}, half-life {half_life}: a gap of {g} half-lives carries {carried:e} \
                         into the next row and is not the stream whose decay was 0 (number \
                         {first:?} of {} differs)",
                        v.len()
                    );
                    // The first row after the gap reports the weight before
                    // it, which the decay to 0 left at 0 and the other at
                    // the subnormal it carries; its numbers are the same.
                    let (p0, c0, _) = &got[0];
                    let (q0, d0, _) = &reference[0];
                    let same = |u: &f64, v: &f64| u.to_bits() == v.to_bits();
                    assert!(
                        p0.iter().zip(q0).all(|(u, v)| same(u, v))
                            && c0
                                .iter()
                                .flatten()
                                .zip(d0.iter().flatten())
                                .all(|(u, v)| same(u, v)),
                        "{what}, half-life {half_life}, a gap of {g} half-lives: the first row"
                    );
                }
            }
        }
        forgotten
    }
    for (cfg, binary) in sgd_settings() {
        let what = format!("{cfg:?}");
        let build = |h: f64| {
            Sgd::new(SgdCfg {
                decay: Decay::Halflife(h),
                min_weight: 0.5,
                ..cfg.clone()
            })
            .unwrap()
        };
        let n = check(&what, build, Sgd::coefficients, binary);
        assert!(n > 2 * 40, "{what}: {n} gaps forgotten");
    }
    for cfg in pa_settings() {
        let what = format!("{cfg:?}");
        let build = |h: f64| {
            Pa::new(PaCfg {
                decay: Decay::Halflife(h),
                min_weight: 0.5,
                ..cfg.clone()
            })
            .unwrap()
        };
        let n = check(&what, build, Pa::coefficients, false);
        assert!(n > 2 * 40, "{what}: {n} gaps forgotten");
    }
}

#[test]
fn pa_predict_is_the_step() {
    // Standardized too, the default since task 195: before the scaler's
    // warm-up ends and after it, when the fit is held in the caller's units
    // (docs/PLAN.md task 206).
    for standardize in [false, true] {
        let mut cfg = pa_cfg();
        cfg.standardize = standardize;
        predict_is_the_step_without_the_step(move || Pa::new(cfg.clone()).unwrap(), 2, false);
    }
}

#[test]
fn holt_predict_is_the_step() {
    predict_is_the_step_without_the_step(|| Holt::new(holt_cfg()).unwrap(), 2, false);
}
