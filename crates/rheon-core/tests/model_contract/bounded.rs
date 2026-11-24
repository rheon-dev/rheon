// ---------------------------------------------------------------------------
// Bounded input (docs/IMPROVEMENTS.md C2, T4)
// ---------------------------------------------------------------------------

/// The largest magnitude the stream layer lets through for a feature, target
/// or weight; beyond it a value is treated as missing.
const BOUND: f64 = rheon_core::INPUT_BOUND;

struct Row {
    x: Vec<f64>,
    y: Vec<Option<f64>>,
    w: f64,
    /// Rows the clean twin never sees.
    extreme: bool,
}

/// One row of the well-behaved process, at `scale`.
fn nice(s: &mut u64, targets: usize, scale: f64) -> Row {
    let x: Vec<f64> = (0..K).map(|_| lcg(s) * scale).collect();
    let y = (0..targets)
        .map(|j| Some(0.5 * (j as f64 + 1.0) * scale + x[0] - 0.5 * x[1]))
        .collect();
    Row {
        x,
        y,
        w: 1.0,
        extreme: false,
    }
}

fn extreme(rows: &mut Vec<Row>, s: &mut u64, targets: usize, edit: impl FnOnce(&mut Row)) {
    let mut r = nice(s, targets, 1.0);
    edit(&mut r);
    r.extreme = true;
    rows.push(r);
}

/// A warm start, then the bound in every position and sign, a run at a tiny
/// scale followed by the bound again (a standardized regressor is then
/// `(1e100 - mean) / 1e-100`), and a long well-behaved tail.
fn bounded_script(targets: usize) -> Vec<Row> {
    let mut s = 20260901u64;
    let mut rows = Vec::new();
    // At the bound before anything else: the first observation is the `a = 0`
    // branch of every accumulator (CLAUDE.md rule 9).
    extreme(&mut rows, &mut s, targets, |r| r.x[0] = BOUND);
    extreme(&mut rows, &mut s, targets, |r| {
        r.y.iter_mut().for_each(|y| *y = Some(BOUND))
    });
    for _ in 0..300 {
        rows.push(nice(&mut s, targets, 1.0));
    }
    for sign in [1.0, -1.0] {
        let b = sign * BOUND;
        extreme(&mut rows, &mut s, targets, |r| r.x[0] = b);
        extreme(&mut rows, &mut s, targets, |r| {
            r.x.iter_mut().for_each(|x| *x = b)
        });
        extreme(&mut rows, &mut s, targets, |r| {
            r.y.iter_mut().for_each(|y| *y = Some(b))
        });
        extreme(&mut rows, &mut s, targets, |r| r.w = BOUND);
        extreme(&mut rows, &mut s, targets, |r| {
            r.x[0] = b;
            r.w = BOUND;
        });
        extreme(&mut rows, &mut s, targets, |r| {
            r.y.iter_mut().for_each(|y| *y = Some(b));
            r.w = BOUND;
        });
        extreme(&mut rows, &mut s, targets, |r| {
            r.x.iter_mut().for_each(|x| *x = b);
            r.y.iter_mut().for_each(|y| *y = Some(b));
            r.w = BOUND;
        });
        extreme(&mut rows, &mut s, targets, |r| r.w = 1.0 / BOUND);
    }
    for _ in 0..200 {
        let mut r = nice(&mut s, targets, 1.0 / BOUND);
        r.extreme = true;
        rows.push(r);
    }
    extreme(&mut rows, &mut s, targets, |r| r.x[0] = BOUND);
    extreme(&mut rows, &mut s, targets, |r| {
        r.y.iter_mut().for_each(|y| *y = Some(BOUND))
    });
    // 1500 half-lives. A row at the bound with weight at the bound leaves a
    // moment of 1e300 on the sum scale, which needs 1000 half-lives to fall
    // below 1e-6; the rest is margin (measured: ewridge agrees with its twin
    // to 3e-5 after 1000 half-lives and to rounding after 1100).
    for _ in 0..30_000 {
        rows.push(nice(&mut s, targets, 1.0));
    }
    rows
}

/// What "recovered" means for a model over the tail of the script; both are
/// relative errors, `|a - b| / (1 + |b|)`.
#[derive(Clone, Copy)]
enum Recovery {
    /// Every prediction agrees with the clean twin's to this tolerance. The
    /// right criterion for a model that converges to one answer on clean data.
    Twin(f64),
    /// Every prediction of the model *and* of its twin is within this distance
    /// of the target. For models that stop learning inside a tolerance band
    /// (`pa` inside its epsilon tube, the quantile loss at its residual floor),
    /// two copies with different histories legitimately settle at different
    /// points of the band, so agreement with the twin is not a property they
    /// have; being as accurate as the twin is.
    Tube(f64),
    /// Every output is finite, and the mean squared distance to the assigned
    /// centre (slot 1, squared) over the tail is within this relative
    /// tolerance of the twin's. For a clustering model: two histories settle
    /// on two labelings of the same data, so neither the label nor the
    /// distance of one row is a property two copies share, but the fit is.
    Fit(f64),
    /// `Twin`, where a slot both withhold (NaN) agrees as well: a model
    /// whose fit can decline to predict on clean data too.
    TwinWithheld(f64),
}

/// The stream layer accepts any finite value with `|v| <= BOUND`, so a model
/// must keep a finite state -- and go on learning -- after any such row. Two
/// copies of the model: one sees the whole script, the twin only its
/// well-behaved rows. Over the last thousand rows every prediction of the
/// first must be finite and satisfy `how`: the extreme rows perturbed the
/// model as its equations say they should, and then washed out, rather than
/// leaving an `inf` or NaN that never decays.
fn recovers_from_bounded_extremes<M: OnlineModel>(
    build: impl Fn() -> M,
    targets: usize,
    how: Recovery,
) {
    recovers_over(&bounded_script(targets), build, how);
}

/// The same, over a script the caller has prepared (a classifier wants the
/// targets turned into labels).
fn recovers_over<M: OnlineModel>(rows: &[Row], build: impl Fn() -> M, how: Recovery) {
    let mut model = build();
    let mut twin = build();
    let kind = model.state().model.kind();
    let n = rows.len();
    let mut seen = [false, false];
    let mut worst = 0.0f64;
    let mut fit = [0.0f64, 0.0];
    for (i, r) in rows.iter().enumerate() {
        let d = if seen[0] { 1.0 } else { 0.0 };
        seen[0] = true;
        let a = model.step(&r.x, &r.y, d, r.w);
        assert!(
            a.n_eff.is_finite(),
            "{kind}: n_eff is {} at row {i}",
            a.n_eff
        );
        if r.extreme {
            continue;
        }
        let d = if seen[1] { 1.0 } else { 0.0 };
        seen[1] = true;
        let b = twin.step(&r.x, &r.y, d, r.w);
        if i + 1000 < n {
            continue;
        }
        for (slot, (pa, pb)) in a.pred.iter().zip(&b.pred).enumerate() {
            if matches!(how, Recovery::TwinWithheld(_)) && pa.is_nan() && pb.is_nan() {
                continue;
            }
            assert!(
                pa.is_finite(),
                "{kind}: slot {slot} predicts {pa} at row {i} (twin: {pb})"
            );
            let rel = |a: f64, b: f64| (a - b).abs() / (1.0 + b.abs());
            match how {
                Recovery::Twin(tol) | Recovery::TwinWithheld(tol) => {
                    let err = rel(*pa, *pb);
                    assert!(
                        err <= tol,
                        "{kind}: slot {slot} at row {i}: {pa} vs the twin's {pb} (tol {tol})"
                    );
                    worst = worst.max(err);
                }
                Recovery::Tube(tol) => {
                    let y = r.y[slot].unwrap();
                    for (who, p) in [("model", *pa), ("twin", *pb)] {
                        let err = rel(p, y);
                        assert!(
                            err <= tol,
                            "{kind}: slot {slot} at row {i}: the {who} predicts {p} for {y} (tol {tol})"
                        );
                        worst = worst.max(err);
                    }
                }
                Recovery::Fit(_) => {
                    assert!(
                        pb.is_finite(),
                        "{kind}: the twin's slot {slot} is {pb} at row {i}"
                    );
                    if slot == 1 {
                        fit[0] += pa * pa;
                        fit[1] += pb * pb;
                    }
                }
            }
        }
    }
    match how {
        Recovery::Twin(_) | Recovery::TwinWithheld(_) => {
            eprintln!("{kind}: worst relative disagreement with the twin {worst:.2e}")
        }
        Recovery::Tube(_) => eprintln!("{kind}: worst relative error of model or twin {worst:.2e}"),
        Recovery::Fit(tol) => {
            let err = (fit[0] - fit[1]).abs() / fit[1];
            eprintln!(
                "{kind}: tail mean squared distance {} vs the twin's {} ({err:.2e})",
                fit[0] / 1000.0,
                fit[1] / 1000.0
            );
            assert!(
                err <= tol,
                "{kind}: tail fit {} vs the twin's {} (tol {tol})",
                fit[0],
                fit[1]
            );
        }
    }
}

#[test]
#[ignore = "extended: a second or more: two 30,000-row recoveries (3.5 s); the clause stays for the other models"]
fn ewridge_recovers_from_bounded_extremes() {
    recovers_from_bounded_extremes(
        || EwRidge::new(ewridge_cfg()).unwrap(),
        2,
        Recovery::Twin(1e-9),
    );
    let mut cfg = ewridge_cfg();
    cfg.standardize = true;
    recovers_from_bounded_extremes(
        move || EwRidge::new(cfg.clone()).unwrap(),
        2,
        Recovery::Twin(1e-9),
    );
}

#[test]
fn rls_recovers_from_bounded_extremes() {
    recovers_from_bounded_extremes(|| Rls::new(rls_cfg()).unwrap(), 2, Recovery::Twin(1e-9));
}

#[test]
fn lasso_recovers_from_bounded_extremes() {
    recovers_from_bounded_extremes(|| Lasso::new(lasso_cfg()).unwrap(), 1, Recovery::Twin(1e-9));
}

#[test]
fn kalman_recovers_from_bounded_extremes() {
    // A target at the bound moves the coefficients by its innovation, as any
    // observation does, and the drift forgets it at `coef_half_life`: 1e100
    // needs about 330 of them. So the drift's half-life is the script's,
    // whose tail is 1500 of it. With the noise estimate in the gain, the
    // estimate the bound inflated inflated the derived process noise with
    // it, and `P` forgot everything within a few rows: the filter recovered
    // at a coefficient half-life of 100 (task 278).
    let drift = |c: KalmanCfg| KalmanCfg {
        half_life: vec![HALFLIFE],
        ..c
    };
    let cfg = drift(kalman_cfg());
    recovers_from_bounded_extremes(
        move || Kalman::new(cfg.clone()).unwrap(),
        2,
        Recovery::Twin(1e-9),
    );
    let mut cfg = drift(kalman_cfg());
    cfg.standardize = false;
    recovers_from_bounded_extremes(
        move || Kalman::new(cfg.clone()).unwrap(),
        2,
        Recovery::Twin(1e-9),
    );
    let cfg = drift(kalman_revert_cfg());
    recovers_from_bounded_extremes(
        move || Kalman::new(cfg.clone()).unwrap(),
        2,
        Recovery::Twin(1e-9),
    );
    // The form a state saved before task 278 goes on in, at the half-life
    // it always recovered at.
    for standardize in [true, false] {
        let cfg = KalmanCfg {
            noise_units: false,
            standardize,
            ..kalman_cfg()
        };
        recovers_from_bounded_extremes(
            move || Kalman::new(cfg.clone()).unwrap(),
            2,
            Recovery::Twin(1e-9),
        );
    }
}

#[test]
#[ignore = "extended: a second or more: four 30,000-row recoveries (6.4 s); the clause stays for the other models"]
fn robust_recovers_from_bounded_extremes() {
    for loss in ROBUST_LOSSES {
        // Huber is least squares near the solution and converges; the
        // quantile loss reweights by `s / |r|`, which never settles closer than
        // its residual floor, so two histories agree only to about that.
        let how = match loss {
            RobustLoss::Huber { .. } => Recovery::Twin(1e-9),
            RobustLoss::Quantile { .. } => Recovery::Tube(1e-3),
        };
        for standardize in [false, true] {
            let mut cfg = robust_cfg(loss);
            cfg.standardize = standardize;
            recovers_from_bounded_extremes(move || Robust::new(cfg.clone()).unwrap(), 2, how);
        }
    }
}

#[test]
fn ftrl_recovers_from_bounded_extremes() {
    recovers_from_bounded_extremes(|| Ftrl::new(ftrl_cfg()).unwrap(), 2, Recovery::Twin(1e-9));
    let mut cfg = ftrl_cfg();
    cfg.loss = FtrlLoss::Logistic;
    recovers_from_bounded_extremes(
        move || Ftrl::new(cfg.clone()).unwrap(),
        2,
        Recovery::Twin(1e-9),
    );
}

#[test]
fn sgd_recovers_from_bounded_extremes() {
    for standardize in [false, true] {
        let mut cfg = sgd_cfg();
        cfg.standardize = standardize;
        recovers_from_bounded_extremes(
            move || Sgd::new(cfg.clone()).unwrap(),
            2,
            Recovery::Twin(1e-9),
        );
    }
}

#[test]
fn pa_recovers_from_bounded_extremes() {
    // PA-I with `epsilon = 0.1` stops updating inside its tube, so it is
    // accurate to the tube, not to the twin.
    recovers_from_bounded_extremes(|| Pa::new(pa_cfg()).unwrap(), 2, Recovery::Tube(1e-1));
}

#[test]
fn holt_recovers_from_bounded_extremes() {
    // The tail is 1500 half-lives of HALFLIFE: what a row at the bound with
    // weight at the bound needs to wash out of a mean-form accumulator. Since
    // the code review's S29 `holt`'s level and trend are weighted means, so a
    // row's weight counts, and the trend forgets such a row on its own
    // half-life -- four times the level's by default, which would want a tail
    // four times as long. So the probe runs both at HALFLIFE. (The textbook
    // form it replaced ignored the weight, and recovered on the level's rate.)
    let cfg = HoltCfg {
        trend_half_life: HALFLIFE,
        ..holt_cfg()
    };
    recovers_from_bounded_extremes(|| Holt::new(cfg.clone()).unwrap(), 2, Recovery::Twin(1e-9));
}

#[test]
fn seqtest_is_indifferent_to_bounded_extremes() {
    // The e-process reads only the sign of each target, so a row at the
    // bound is one more trial like any other: no accumulator holds the
    // magnitude, and the state after the script equals a twin's that was
    // fed the signs alone. (The twin criteria above do not apply: a test
    // never forgets, so a copy that saw more rows is a different test.)
    let rows = bounded_script(2);
    let mut model = SeqTest::new(seqtest_cfg()).unwrap();
    let mut twin = SeqTest::new(seqtest_cfg()).unwrap();
    for (i, r) in rows.iter().enumerate() {
        let d = if i == 0 { 0.0 } else { 1.0 };
        let a = model.step(&r.x, &r.y, d, r.w);
        let signs: Vec<Option<f64>> = r.y.iter().map(|y| y.map(f64::signum)).collect();
        let b = twin.step(&[], &signs, d, r.w);
        assert!(a.n_eff.is_finite(), "n_eff is {} at row {i}", a.n_eff);
        assert!(
            a.pred.iter().all(|p| p.is_finite() || a.n_eff < 3.0),
            "row {i}: {:?}",
            a.pred
        );
        same_step("seqtest", i, &a, &b);
    }
    assert_eq!(model, twin);
    // A weight at the bound counts itself, once; the rows after it are
    // lost in its rounding but the counts and the wealth go on moving.
    assert!(model.n_eff() >= BOUND);
    assert!(model.n_pos()[0] + model.n_neg()[0] > 30_000.0);
}

#[test]
fn kmeans_recovers_from_bounded_extremes() {
    // The extreme rows drag a centre to the bound and blow the metric up;
    // the split–merge check re-places the emptied centre and the moments
    // decay back, so the tail is fitted as well as the twin fits it
    // (measured: the two agree to 1e-14; the tolerance is the margin).
    for standardize in [true, false] {
        for rule in [SeedRule::First, SeedRule::Lloyd] {
            let cfg = KMeansCfg {
                standardize,
                seed_rule: rule,
                ..kmeans_cfg()
            };
            recovers_from_bounded_extremes(
                move || KMeans::new(cfg.clone()).unwrap(),
                0,
                Recovery::Fit(1e-6),
            );
        }
    }
}

#[test]
#[ignore = "extended: a second or more: three 30,000-row recoveries (1.8 s); the clause stays for the other models"]
fn ew_class_recovers_from_bounded_extremes() {
    // The script's target becomes the label: class 1 where y > 0. A row at
    // the bound with weight at the bound leaves one class's mean at 1e100
    // and its co-moments at 1e200; both decay back through the mean-form
    // update as the weight does, and the ridge scale, once driven to zero
    // by that row, never matters again (measured: the two states are
    // bitwise equal over the tail; the tolerance is the margin).
    let mut rows = bounded_script(1);
    for r in &mut rows {
        r.y = vec![Some(f64::from(r.y[0].unwrap() > 0.0))];
    }
    for covariance in [Covariance::Full, Covariance::Shared, Covariance::Diagonal] {
        let cfg = EwClassCfg {
            covariance,
            ..ew_class_cfg()
        };
        recovers_over(
            &rows,
            move || EwClass::new(cfg.clone()).unwrap(),
            Recovery::Twin(1e-9),
        );
    }
}

#[test]
fn micro_recovers_from_bounded_extremes() {
    // A row of weight 1e100 makes a summary nothing moves until it has
    // decayed below `beta_mu` (332 half-lives) and is pruned; a row at the
    // bound opens a summary there that the xi rule prunes at the next
    // checkpoint. By the tail both histories tile the unit square afresh,
    // and two tilings agree on the mean squared distance to the nearest
    // potential summary only loosely (measured: to about a tenth).
    for standardize in [true, false] {
        for macro_link in [None, Some(0.0)] {
            let cfg = MicroCfg {
                standardize,
                macro_link,
                ..micro_cfg()
            };
            recovers_from_bounded_extremes(
                move || Micro::new(cfg.clone()).unwrap(),
                0,
                Recovery::Fit(0.5),
            );
        }
    }
}

#[test]
fn deco_recovers_from_bounded_extremes() {
    // A 1e100 row makes every standardised value huge, so `u` is at its
    // bounds for a while and `loglik` is very negative; the state must stay
    // finite and the outputs must return to a clean twin's once the row has
    // decayed away.
    recovers_from_bounded_extremes(|| Deco::new(deco_cfg()).unwrap(), 0, Recovery::Twin(1e-9));
}

#[test]
fn ew_cov_recovers_from_bounded_extremes() {
    recovers_from_bounded_extremes(
        || EwCovModel::new(ew_cov_model_cfg()).unwrap(),
        0,
        Recovery::Twin(1e-9),
    );
    let mut cfg = ew_cov_model_cfg();
    cfg.stats.push(EwCovStat::PartialCorr);
    cfg.precision_prior = Some(1e-4);
    recovers_from_bounded_extremes(
        move || EwCovModel::new(cfg.clone()).unwrap(),
        0,
        Recovery::Twin(1e-9),
    );
}

#[test]
fn marginal_recovers_from_bounded_extremes() {
    // `marginal` predicts nothing, so `recovers_over` would have nothing to
    // compare: read the pairs from the state instead. Over the last thousand
    // rows every statistic of every pair must be finite and agree with the
    // twin's to rounding -- the extreme rows moved the moments as the
    // equations say and then washed out.
    let rows = bounded_script(2);
    let mut model = Marginal::new(marginal_cfg()).unwrap();
    let mut twin = Marginal::new(marginal_cfg()).unwrap();
    let n = rows.len();
    let mut seen = [false, false];
    let rel = |a: f64, b: f64| (a - b).abs() / (1.0 + b.abs());
    for (i, r) in rows.iter().enumerate() {
        let d = if seen[0] { 1.0 } else { 0.0 };
        seen[0] = true;
        let a = model.step(&r.x, &r.y, d, r.w);
        assert!(
            a.n_eff.is_finite(),
            "marginal: n_eff is {} at row {i}",
            a.n_eff
        );
        assert!(a.pred.is_empty());
        if r.extreme {
            continue;
        }
        let d = if seen[1] { 1.0 } else { 0.0 };
        seen[1] = true;
        twin.step(&r.x, &r.y, d, r.w);
        if i + 1000 < n {
            continue;
        }
        for t in 0..2 {
            for j in 0..K {
                let (pa, pb) = (model.pair(t, j), twin.pair(t, j));
                for (what, va, vb) in [
                    ("weight_sum", pa.n_eff, pb.n_eff),
                    ("n_kish", pa.n_kish, pb.n_kish),
                    ("mean_x", pa.mean_x, pb.mean_x),
                    ("var_x", pa.var_x, pb.var_x),
                    ("mean_y", pa.mean_y, pb.mean_y),
                    ("var_y", pa.var_y, pb.var_y),
                    ("cov", pa.cov, pb.cov),
                    ("corr", pa.corr, pb.corr),
                    ("beta", pa.beta, pb.beta),
                    ("t", pa.t, pb.t),
                ] {
                    assert!(
                        va.is_finite(),
                        "marginal: {what}[{t},{j}] is {va} at row {i} (twin: {vb})"
                    );
                    assert!(
                        rel(va, vb) <= 1e-9,
                        "marginal: {what}[{t},{j}] at row {i}: {va} vs the twin's {vb}"
                    );
                }
            }
        }
    }
}
