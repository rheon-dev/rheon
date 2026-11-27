use super::*;
use crate::oracle::CanonBits;

/// The dispatched `standardize_row` (AVX2 lanes where the CPU has them)
/// against the scalar loop, to the bit: random scalers and rows at every
/// width up to 37 (each remainder of a vector), with and without an
/// intercept, features held constant (no spread), and wide magnitudes.
/// On an x86-64 CPU with AVX2 the lanes run from `LANES_FROM` features;
/// elsewhere -- without AVX2, or on another architecture, such as an
/// arm64 Mac -- the dispatch takes the loop, and the test holds the loop
/// to itself.
#[test]
fn standardize_row_lanes_match_the_scalar_loop_to_the_bit() {
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    let mut next = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    let same = |a: f64, b: f64| a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan());
    let mut checked = 0usize;
    for k in 1..=37usize {
        for off in [0usize, 1] {
            for trial in 0..6 {
                let slots = k + off;
                let mut sc = EwDiag::new(slots);
                let scale_mag = [1e-6, 1.0, 1e6][trial % 3];
                let rows = 3 + (next() * 40.0) as usize;
                for r in 0..rows {
                    let x: Vec<f64> = (0..slots)
                        .map(|j| {
                            if j % 5 == 3 && trial >= 3 {
                                2.5
                            } else {
                                scale_mag * (next() * 2.0 - 1.0) + j as f64
                            }
                        })
                        .collect();
                    let w = if r % 7 == 6 { 0.0 } else { 0.5 + next() };
                    sc.update(&x, 0.9 + 0.1 * next(), w);
                }
                let x: Vec<f64> = (0..k)
                    .map(|j| {
                        if j % 5 == 3 && trial >= 3 {
                            2.5
                        } else {
                            scale_mag * (next() * 2.0 - 1.0)
                        }
                    })
                    .collect();
                let lam = 0.95;
                let (mut z1, mut m1, mut s1) = (vec![0.0; k], vec![0.0; k], vec![0.0; k]);
                standardize_row(&sc, off, &x, lam, 1.0, 1.0, &mut z1, &mut m1, &mut s1);
                let inc = sc.including(scaler_decay(&sc, lam, 1.0, 1.0));
                let (hi, lo, c) = sc.parts();
                let (mut z2, mut m2, mut s2) = (vec![0.0; k], vec![0.0; k], vec![0.0; k]);
                standardize_scalar(
                    &inc,
                    off,
                    &x,
                    &hi[off..off + k],
                    &lo[off..off + k],
                    &c[off..off + k],
                    &mut z2,
                    &mut m2,
                    &mut s2,
                );
                for i in 0..k {
                    assert!(
                        same(z1[i], z2[i]) && same(m1[i], m2[i]) && same(s1[i], s2[i]),
                        "k={k} off={off} trial={trial} i={i}: z {:?}/{:?} m {:?}/{:?} s {:?}/{:?}",
                        z1[i],
                        z2[i],
                        m1[i],
                        m2[i],
                        s1[i],
                        s2[i]
                    );
                    checked += 1;
                }
            }
        }
    }
    assert!(checked > 4000, "{checked}");
}

/// A row standardized against the moments that include it, after a
/// clock gap of up to 1,000 half-lives, is what exact arithmetic gives
/// (review fable2, C1): `z = (x − m)/sqrt(v)` from the stream's
/// definitions at 2,048 bits (`oracle::exact::standardized`), to 1e-12
/// of itself. After a gap the history's share `a = λW/(λW + 1)` is
/// tiny, and the deviation from the mean that includes the row is
/// `a·d`, `d` its deviation from the mean before it. Formed as `x − (m +
/// b·d)`, it carried the rounding of `b·d`, `2⁻⁵³·|d|`, `2⁻⁵³/a` of the
/// deviation: on a5c4dcef `z` was 6.4e-2 off at 55 half-lives, and past
/// `a < 2⁻⁵³` exactly 0 or rounding noise over a scale of `sqrt(a)`,
/// and the step, divided by that scale, left slopes of up to 1e151 for
/// good. The ordinary rows (a gap of one row) are held to the same
/// bound, and the gaps each side of `SMALL_SHARE`.
#[test]
fn a_row_after_a_long_gap_is_standardized_as_exact_arithmetic_does() {
    let hl = 20.0f64;
    let mut worst = 0.0f64;
    let mut checked = 0usize;
    // Each side of `SMALL_SHARE` (13 and 15 half-lives here), and on
    // to 1,000 half-lives.
    for gap in [1.0, 20.0, 260.0, 300.0, 1100.0, 2000.0, 10_000.0, 20_000.0] {
        for seed in 0..40u64 {
            let mut s = seed.wrapping_mul(0x9e37_79b9) + 80;
            let (k, off) = (2usize, 1usize);
            let mut sc = EwDiag::new(k + off);
            let mut rows: Vec<Vec<(f64, f64, f64)>> = vec![Vec::new(); k];
            for i in 0..300 {
                let x = [1.0, 0.5 + lcg(&mut s), -3.0 + 2.0 * lcg(&mut s)];
                let lam = if i == 0 { 1.0 } else { (-1.0 / hl).exp2() };
                let w = if i % 11 == 5 { 0.25 } else { 1.0 };
                sc.update(&x, lam, w);
                for (f, r) in rows.iter_mut().enumerate() {
                    r.push((x[off + f], lam, w));
                }
            }
            let lam = (-gap / hl).exp2();
            assert_eq!(scaler_decay(&sc, lam, 1.0, 0.0), lam, "the history is kept");
            let x = [0.5 + lcg(&mut s), -3.0 + 2.0 * lcg(&mut s)];
            let (mut z, mut m, mut sd) = (vec![0.0; k], vec![0.0; k], vec![0.0; k]);
            // The warm-up's read, which keeps `lam` however long the gap.
            standardize_row(&sc, off, &x, lam, 1.0, 0.0, &mut z, &mut m, &mut sd);
            let lazy: Vec<f64> = standardized(&sc, off, &x, lam, 1.0, 0.0).collect();
            for f in 0..k {
                assert_eq!(z[f].to_bits(), lazy[f].to_bits(), "the two paths agree");
                let want =
                    crate::oracle::exact::standardized(&rows[f], lam, x[f]).expect("a spread");
                let err = (z[f] - want).abs() / want.abs();
                worst = worst.max(err);
                assert!(
                    err <= 1e-12,
                    "gap {gap}, seed {seed}, feature {f}: z {:e} against {want:e}, \
                     {err:e} of it",
                    z[f]
                );
                checked += 1;
            }
        }
    }
    assert_eq!(checked, 8 * 40 * 2);
    println!("worst relative error of z: {worst:e}");
}

/// One clock gap of 500 half-lives, then 3,000 ordinary rows: the fit
/// comes back (review fable2, C1, its reproduction). Before the fix the
/// worst error over the last 500 rows was 6.1e75 at a constant rate,
/// the slopes near 1e75; each later step moves a slope by at most
/// `lr·clip`, so they never came back. Under `inv_scaling` too.
#[test]
fn one_long_gap_leaves_a_standardized_fit_that_recovers() {
    for schedule in [
        LearningRate::Constant,
        LearningRate::InvScaling { power: 0.25 },
    ] {
        for gap in [1_300.0, 2_000.0, 10_000.0, 20_000.0] {
            let mut worst = 0.0f64;
            for seed in [80u64, 81, 82, 83, 84, 85, 86, 87] {
                let mut m = Sgd::new(SgdCfg {
                    n_targets: 2,
                    decay: Decay::Halflife(20.0),
                    learning_rate: 0.01,
                    schedule,
                    clip_gradient: 1e3,
                    standardize: true,
                    min_weight: 3.0,
                    ..cfg(2, SgdLoss::Squared)
                })
                .unwrap();
                let mut s = seed;
                let row = |s: &mut u64| {
                    let x = [lcg(s), lcg(s)];
                    let y: Vec<Option<f64>> = (0..2)
                        .map(|j| Some(0.5 * (j as f64 + 1.0) + x[0] - 0.5 * x[1] + 0.05 * lcg(s)))
                        .collect();
                    (x, y)
                };
                for i in 0..300 {
                    let (x, y) = row(&mut s);
                    m.step(&x, &y, if i == 0 { 0.0 } else { 1.0 }, 1.0);
                }
                for i in 0..3000 {
                    let (x, y) = row(&mut s);
                    let st = m.step(&x, &y, if i == 0 { gap } else { 1.0 }, 1.0);
                    if i >= 2500 {
                        worst = worst.max((st.pred[0] - y[0].unwrap()).abs());
                    }
                }
            }
            assert!(
                worst < 1.0,
                "{schedule:?}, gap {gap}: |error| {worst:e} 2,500 rows after one gap"
            );
        }
    }
}

fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

fn cfg(k: usize, loss: SgdLoss) -> SgdCfg {
    SgdCfg {
        n_features: k,
        n_targets: 1,
        fit_intercept: true,
        decay: Decay::Halflife(f64::INFINITY),
        loss,
        learning_rate: 0.05,
        schedule: LearningRate::Constant,
        l2: 0.0,
        min_weight: 5.0,
        clip_gradient: 1e12,
        constraint: None,
        standardize: false,
        strict_binary: false,
        huber_scale: crate::HuberScale::Rms,
        huber_scale_decay: None,
        refit_rows: 0,
        min_weight_holds_learning: false,
        target_spread: false,
        mad_anchor: false,
        refit_clock: None,
        max_state_bytes: None,
        l1: 0.0,
        average: false,
        invariant: false,
    }
}

#[test]
fn cfg_validation_rejects_each_bad_field() {
    let bad = |f: &dyn Fn(&mut SgdCfg), want: &str| {
        let mut c = cfg(2, SgdLoss::Squared);
        f(&mut c);
        match c.validate() {
            Err(e) => assert!(e.contains(want), "wanted {want:?}, got {e:?}"),
            Ok(()) => panic!("expected rejection mentioning {want:?}"),
        }
    };
    bad(
        &|c| c.n_features = 0,
        "needs at least one feature and one target",
    );
    bad(
        &|c| c.n_targets = 0,
        "needs at least one feature and one target",
    );
    bad(&|c| c.learning_rate = 0.0, "learning_rate must be above 0");
    bad(
        &|c| c.learning_rate = f64::NAN,
        "learning_rate must be above 0",
    );
    bad(&|c| c.l2 = -1e-9, "l2 must be at least 0");
    // clip_gradient is a magnitude bound, so inf disables it rather than
    // being an error, but zero would clip everything to nothing.
    bad(&|c| c.clip_gradient = 0.0, "clip_gradient must be above 0");
    // NaN passed every `<= 0.0` / `< 0.0` test here: a NaN clip then
    // panicked in `f64::clamp` on the first learned row, a NaN `l2`
    // poisoned every coefficient, a NaN `eps` emptied the tube and a NaN
    // `power` made every rate NaN (review 2026-09-18, B4).
    bad(
        &|c| c.clip_gradient = f64::NAN,
        "clip_gradient must be above 0",
    );
    bad(&|c| c.l2 = f64::NAN, "l2 must be at least 0");
    bad(
        &|c| c.loss = SgdLoss::EpsilonInsensitive { eps: f64::NAN },
        "eps must be at least 0",
    );
    bad(
        &|c| c.schedule = LearningRate::InvScaling { power: f64::NAN },
        "power must be at least 0",
    );
    let mut ok = cfg(2, SgdLoss::Squared);
    ok.clip_gradient = f64::INFINITY;
    ok.validate().unwrap();

    bad(
        &|c| c.loss = SgdLoss::Huber { delta: 0.0 },
        "huber_delta must be above 0",
    );
    for t in [0.0, 1.0, 1.5] {
        bad(
            &|c| c.loss = SgdLoss::Quantile { tau: t },
            "quantile must be above 0 and below 1",
        );
    }
    bad(
        &|c| c.loss = SgdLoss::EpsilonInsensitive { eps: -1.0 },
        "eps must be at least 0",
    );
    // eps = 0 is a legal (if pointless) epsilon-insensitive loss.
    cfg(2, SgdLoss::EpsilonInsensitive { eps: 0.0 })
        .validate()
        .unwrap();

    bad(
        &|c| c.schedule = LearningRate::InvScaling { power: -0.1 },
        "power must be at least 0",
    );
    let mut ok = cfg(2, SgdLoss::Squared);
    ok.schedule = LearningRate::InvScaling { power: 0.0 };
    ok.validate().unwrap();

    // Every problem at once, each a line of its own.
    let mut two = cfg(2, SgdLoss::Squared);
    two.l2 = -1.0;
    two.clip_gradient = 0.0;
    let e = two.validate().unwrap_err();
    let lines: Vec<&str> = e.lines().collect();
    assert_eq!(lines.len(), 2, "{e}");
    assert!(
        lines[0].starts_with("sgd: l2 must be at least 0, but it is -1."),
        "{e}"
    );
    assert!(
        lines[1].starts_with("sgd: clip_gradient must be above 0, but it is 0."),
        "{e}"
    );
}

#[test]
fn each_loss_has_the_gradient_it_claims() {
    // `dloss` is dL/d(eta) and `link` maps eta to the prediction. Both are
    // small and entirely arithmetic, and every model output depends on
    // them, so they are checked directly rather than through a fit.
    let m = |loss| Sgd::new(cfg(1, loss)).unwrap();

    // Squared: the plain residual, in both directions.
    let one = Some(1.0);
    assert_eq!(m(SgdLoss::Squared).dloss(3.0, 1.0, one), 2.0);
    assert_eq!(m(SgdLoss::Squared).dloss(1.0, 3.0, one), -2.0);

    // Huber: the residual, clipped symmetrically at delta residual
    // stds, and not at all before the target has a scale.
    let h = m(SgdLoss::Huber { delta: 1.5 });
    assert_eq!(h.dloss(1.0, 0.0, one), 1.0, "inside the band: squared");
    assert_eq!(h.dloss(9.0, 0.0, one), 1.5, "outside: clipped");
    assert_eq!(h.dloss(-9.0, 0.0, one), -1.5);
    assert_eq!(h.dloss(9.0, 0.0, Some(2.0)), 3.0, "delta stds of 2");
    assert_eq!(h.dloss(9.0, 0.0, None), 9.0, "no scale: no cut");

    // Quantile: a constant gradient whose sign depends on which side of
    // the prediction the target fell, asymmetric except at tau = 0.5.
    let q = m(SgdLoss::Quantile { tau: 0.9 });
    assert_eq!(q.dloss(5.0, 1.0, one), 1.0 - 0.9, "over-predicted");
    assert_eq!(q.dloss(1.0, 5.0, one), -0.9, "under-predicted");
    let med = m(SgdLoss::Quantile { tau: 0.5 });
    assert_eq!(
        med.dloss(5.0, 1.0, one),
        -med.dloss(1.0, 5.0, one),
        "symmetric at 0.5"
    );

    // Epsilon-insensitive: exactly zero inside the tube of eps of the
    // target's std, and a tube of no width before the target has one.
    let e = m(SgdLoss::EpsilonInsensitive { eps: 1.0 });
    assert_eq!(e.dloss(0.5, 0.0, one), 0.0);
    assert_eq!(e.dloss(1.0, 0.0, one), 0.0, "the boundary is inside");
    assert!(e.dloss(3.0, 0.0, one) > 0.0);
    assert!(e.dloss(-3.0, 0.0, one) < 0.0);
    assert_eq!(e.dloss(3.0, 0.0, Some(4.0)), 0.0, "a tube of 4");
    assert_eq!(e.dloss(0.5, 0.0, None), 1.0, "no scale: no tube");
    assert_eq!(e.dloss(0.0, 0.0, None), 0.0, "on the target");

    // Links. Squared and the robust losses are the identity; logistic and
    // Poisson are not, and both must stay finite at extreme inputs.
    assert_eq!(m(SgdLoss::Squared).link(-7.0), -7.0);
    let lg = m(SgdLoss::Logistic);
    assert_eq!(lg.link(0.0), 0.5);
    for eta in [900.0, -900.0] {
        let p = lg.link(eta);
        assert!(
            p.is_finite() && (0.0..=1.0).contains(&p),
            "link({eta}) = {p}"
        );
    }
    let po = m(SgdLoss::Poisson);
    assert_eq!(po.link(0.0), 1.0);
    assert!(po.link(1e9).is_finite(), "the exponent is clamped");
    assert!(po.link(-1e9) > 0.0, "a rate is positive");
    // Where the module docs put the clamp: `exp(clamp(eta, ±30))`, so a
    // linear predictor past 30 either way predicts as 30 does (review
    // 2026-10-05, CC6: the table said `exp(eta)`).
    assert_eq!(po.link(31.0), po.link(30.0));
    assert_eq!(po.link(-31.0), po.link(-30.0));
    assert_eq!(po.link(30.0), 30.0f64.exp());
    assert!(po.link(29.0) < po.link(30.0), "inside the clamp it is exp");
}

/// The case scaling exists for: one feature in thousands, one in
/// thousandths. A single learning rate cannot suit both.
#[test]
fn scaling_rescues_badly_scaled_features() {
    let run = |scale: bool| {
        let mut c = cfg(2, SgdLoss::Squared);
        c.standardize = scale;
        c.learning_rate = 0.01;
        c.min_weight = 0.0;
        let mut m = Sgd::new(c).unwrap();
        let mut s = 61u64;
        for i in 0..20000 {
            let x = [1000.0 * lcg(&mut s), 0.001 * lcg(&mut s)];
            let y = 0.002 * x[0] + 900.0 * x[1];
            m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        m.coefficients()[0].clone()
    };
    let plain = run(false);
    let scaled = run(true);
    let err = |b: &[f64]| (b[1] - 0.002).abs() / 0.002 + (b[2] - 900.0).abs() / 900.0;
    assert!(
        err(&scaled) < err(&plain),
        "scaled {scaled:?} should beat unscaled {plain:?} (truth [_, 0.002, 900])"
    );
    assert!(err(&scaled) < 0.2, "scaled fit still poor: {scaled:?}");
}

#[test]
fn scaling_reports_coefficients_in_the_callers_units() {
    let mut c = cfg(1, SgdLoss::Squared);
    c.standardize = true;
    c.learning_rate = 0.1;
    c.min_weight = 0.0;
    let mut m = Sgd::new(c).unwrap();
    let mut s = 63u64;
    for i in 0..20000 {
        let x = [500.0 + 100.0 * lcg(&mut s)];
        let y = 0.05 * x[0] + 3.0;
        m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
    }
    let b = &m.coefficients()[0];
    assert!(
        (b[1] - 0.05).abs() < 5e-3,
        "slope in original units: {}",
        b[1]
    );
    assert!(
        (b[0] - 3.0).abs() < 1.0,
        "intercept absorbs the shift: {}",
        b[0]
    );
}

#[test]
fn scaling_admits_the_row_it_scales() {
    // The row is standardized against moments that include it (docs/PLAN.md
    // task 74). The first row of a stream is then the whole history: its
    // feature standardizes to 0, so only the intercept learns from it, and
    // an enormous first row cannot throw the slope.
    let mut c = cfg(1, SgdLoss::Squared);
    c.standardize = true;
    c.min_weight = 0.0;
    let mut m = Sgd::new(c.clone()).unwrap();
    let step = m.step(&[1e6], &[Some(1.0)], 0.0, 1.0);
    assert_eq!(step.pred[0], 0.0, "an empty fit predicts 0");
    let coef = &m.coefficients()[0];
    assert_eq!(
        coef[1], 0.0,
        "the first row is z = 0: the slope cannot move"
    );
    assert_eq!(coef[0], 0.05, "the intercept learned from it: lr * (1 - 0)");

    // With the row inside the moments a standardized value is bounded by
    // sqrt(lam * W): z = a d / sqrt(a c + a b d^2) <= sqrt(a / b), with
    // equality when the history has no spread (c = 0). So a row a million
    // deviations out predicts within |b0| + |b1| sqrt(W) of zero, where
    // against the moments from *before* the row -- the order this
    // replaced -- the same row predicted about 1.7e6.
    let mut m = Sgd::new(c).unwrap();
    for i in 0..10 {
        let x = if i % 2 == 0 { 1.0 } else { -1.0 };
        m.step(&[x], &[Some(0.5 * x)], f64::from(u8::from(i > 0)), 1.0);
    }
    let b = m.beta[0].clone(); // in standardized coordinates
    let w = m.scaler.as_ref().unwrap().n_eff();
    assert_eq!(w, 10.0);
    let step = m.step(&[1e6], &[Some(0.0)], 1.0, 1.0);
    let bound = b[0].abs() + b[1].abs() * w.sqrt();
    assert!(
        step.pred[0].abs() <= bound,
        "pred {} exceeds the bound {bound} of a standardized row",
        step.pred[0]
    );
    assert!(
        step.pred[0].abs() > 0.5 * bound,
        "the row is at the bound's edge"
    );
}

/// Runs a stream and returns the final coefficients.
fn fit(cfg: SgdCfg, n: usize, seed: u64, mut f: impl FnMut(&[f64], &mut u64) -> f64) -> Vec<f64> {
    let k = cfg.n_features;
    let mut m = Sgd::new(cfg).unwrap();
    let mut s = seed;
    for i in 0..n {
        let x: Vec<f64> = (0..k).map(|_| lcg(&mut s)).collect();
        let y = f(&x, &mut s);
        m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
    }
    m.coefficients()[0].clone()
}

#[test]
fn squared_loss_recovers_the_coefficients() {
    let b = fit(cfg(2, SgdLoss::Squared), 20000, 1, |x, _| {
        1.5 * x[0] - 0.5 * x[1] + 0.25
    });
    assert!((b[0] - 0.25).abs() < 0.05, "intercept {}", b[0]);
    assert!((b[1] - 1.5).abs() < 0.05, "slope0 {}", b[1]);
    assert!((b[2] + 0.5).abs() < 0.05, "slope1 {}", b[2]);
}

#[test]
fn poisson_loss_recovers_a_log_rate() {
    // y ~ counts with log-rate 0.4 + 0.8 x0. The canonical link means the
    // coefficients live on the log scale.
    let mut c = cfg(1, SgdLoss::Poisson);
    c.learning_rate = 0.02;
    let mut m = Sgd::new(c).unwrap();
    let mut s = 11u64;
    for i in 0..40000 {
        let x = [lcg(&mut s)];
        let rate = (0.4 + 0.8 * x[0]).exp();
        // Poisson draw by inversion (rate is small enough for this to be cheap)
        let mut kdraw = 0.0;
        let mut prod = (lcg(&mut s) + 1.0) / 2.0;
        let limit = (-rate).exp();
        while prod > limit && kdraw < 50.0 {
            prod *= (lcg(&mut s) + 1.0) / 2.0;
            kdraw += 1.0;
        }
        m.step(&x, &[Some(kdraw)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
    }
    let b = &m.coefficients()[0];
    assert!((b[0] - 0.4).abs() < 0.15, "log-intercept {}", b[0]);
    assert!((b[1] - 0.8).abs() < 0.15, "log-slope {}", b[1]);
}

#[test]
fn poisson_predictions_are_non_negative() {
    let mut c = cfg(1, SgdLoss::Poisson);
    c.min_weight = 0.0;
    let mut m = Sgd::new(c).unwrap();
    let mut s = 12u64;
    for i in 0..2000 {
        let x = [lcg(&mut s) * 5.0];
        let st = m.step(&x, &[Some(3.0)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        if st.pred[0].is_finite() {
            assert!(st.pred[0] >= 0.0, "poisson predicted {}", st.pred[0]);
        }
    }
}

#[test]
fn quantile_loss_tracks_the_level() {
    let lo = fit(cfg(1, SgdLoss::Quantile { tau: 0.1 }), 20000, 13, |_, s| {
        1.0 + 2.0 * lcg(s)
    });
    let hi = fit(cfg(1, SgdLoss::Quantile { tau: 0.9 }), 20000, 13, |_, s| {
        1.0 + 2.0 * lcg(s)
    });
    assert!(
        lo[0] < hi[0],
        "tau=0.1 intercept {} !< tau=0.9 {}",
        lo[0],
        hi[0]
    );
}

/// `y = 2x`, with every 20th row replaced by a gross outlier.
fn contaminated() -> impl FnMut(&[f64], &mut u64) -> f64 {
    let mut row = 0u64;
    move |x: &[f64], s: &mut u64| {
        row += 1;
        if row.is_multiple_of(20) {
            500.0 * lcg(s)
        } else {
            2.0 * x[0]
        }
    }
}

#[test]
fn huber_resists_outliers() {
    let hub = fit(
        cfg(1, SgdLoss::Huber { delta: 1.0 }),
        20000,
        17,
        contaminated(),
    );
    let sq = fit(cfg(1, SgdLoss::Squared), 20000, 17, contaminated());
    assert!(
        (hub[1] - 2.0).abs() < (sq[1] - 2.0).abs(),
        "huber {} should beat squared {} (truth 2.0)",
        hub[1],
        sq[1]
    );
}

#[test]
fn epsilon_insensitive_needs_an_annealed_rate_to_settle() {
    // Its subgradient is sign-valued (+/-1), so the step size does not
    // shrink near the optimum: with a constant rate the coefficient
    // oscillates in a band, and only an annealed schedule settles. Both
    // halves are asserted, because the constant-rate behaviour is a real
    // property of this loss rather than a bug. The band is the slope's
    // range over the last 2000 of 30000 rows: a final value alone is one
    // draw from it. The tube, `eps = 0.02` of the target's std of about
    // 1.15, is narrower than the noise of up to 0.1, so most rows step;
    // at `eps = 0.2` it held every residual once the slope reached 1.87
    // under the constant rate and 1.89 under the annealed one, and both
    // fits stopped there (docs/PLAN.md task 202).
    let band = |c: SgdCfg| {
        let mut m = Sgd::new(c).unwrap();
        let mut s = 19u64;
        let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
        for i in 0..30000 {
            let x = [lcg(&mut s)];
            let y = 2.0 * x[0] + 0.1 * lcg(&mut s);
            m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
            if i >= 28000 {
                let b = m.coefficients()[0][1];
                (lo, hi) = (lo.min(b), hi.max(b));
            }
        }
        (lo, hi)
    };
    let mut constant = cfg(1, SgdLoss::EpsilonInsensitive { eps: 0.02 });
    constant.learning_rate = 0.01;
    let (c_lo, c_hi) = band(constant);

    let mut annealed = cfg(1, SgdLoss::EpsilonInsensitive { eps: 0.02 });
    annealed.learning_rate = 0.5;
    annealed.schedule = LearningRate::InvScaling { power: 0.5 };
    let (a_lo, a_hi) = band(annealed);

    assert!(
        a_hi - a_lo < 0.75 * (c_hi - c_lo),
        "annealed band [{a_lo}, {a_hi}] should be narrower than constant [{c_lo}, {c_hi}]"
    );
    assert!(
        (a_lo - 2.0).abs() < 0.05 && (a_hi - 2.0).abs() < 0.05,
        "annealed band [{a_lo}, {a_hi}] (truth 2.0)"
    );
}

/// The tube does not shrink as the fit improves, so a fit stops wherever
/// every residual is inside it (docs/PLAN.md task 203). `y = 2x` plus
/// noise of up to 0.01, `x` in `[-1, 1]`, so the target's spread is
/// about `2/√3 = 1.155` and the noise's standard deviation `0.01/√3 =
/// 0.0058` -- a tube of 0.1 is 20 noise standard deviations wide, one
/// of 0.01 two (task 207): a fit with every residual inside `eps` of it
/// is off by `|b0| + |b1 − 2| ≤ 1.155·eps − 0.01` at most. At 0.01 that
/// is 0.0016, and an annealed rate ends within 0.005 (the scaler still
/// moves the coefficients a little); at 0.1 it is 0.105, and the fit
/// stopped more than 0.01 off.
#[test]
fn a_tube_two_noise_stds_wide_reaches_the_slope() {
    let off = |eps: f64| {
        let mut cf = cfg(1, SgdLoss::EpsilonInsensitive { eps });
        cf.learning_rate = 0.5;
        cf.schedule = LearningRate::InvScaling { power: 0.5 };
        cf.standardize = true;
        cf.clip_gradient = 1e3;
        cf.min_weight = 2.0;
        let mut m = Sgd::new(cf).unwrap();
        let mut s = 19u64;
        for i in 0..20_000 {
            let x = [lcg(&mut s)];
            let y = 2.0 * x[0] + 0.01 * lcg(&mut s);
            m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        let b = &m.coefficients()[0];
        b[0].abs() + (b[1] - 2.0).abs()
    };
    let (inside, outside) = (off(0.01), off(0.1));
    assert!(inside < 0.005, "eps 0.01: {inside}");
    assert!(outside > 0.01, "eps 0.1: {outside}");
}

/// A residual inside the tube, `eps` of the target's own std wide,
/// leaves the fit alone, and one outside it steps: the "insensitive" of
/// the name. The first two rows, 1 and 3, have no spread to draw a tube
/// in, so each steps whatever its residual; they give the target a
/// spread of 1, the unit the next rows are judged in (docs/PLAN.md task
/// 202). The Huber scale is not kept under this loss.
#[test]
fn epsilon_insensitive_ignores_residuals_inside_the_tube() {
    let mut c = cfg(1, SgdLoss::EpsilonInsensitive { eps: 0.8 });
    c.learning_rate = 0.01;
    c.min_weight = 0.0;
    let mut m = Sgd::new(c).unwrap();
    m.step(&[1.0], &[Some(1.0)], 0.0, 1.0);
    assert_eq!(m.coefficients()[0], vec![0.01, 0.01], "no spread: a step");
    assert_eq!(m.target_variance(), vec![0.0]);
    m.step(&[1.0], &[Some(3.0)], 1.0, 1.0);
    let second = m.coefficients()[0].clone();
    assert_eq!(second, vec![0.02, 0.02], "one row has no spread: a step");
    assert_eq!(m.target_variance(), vec![1.0]);
    assert!(m.sigma2().is_empty(), "no residual scale under this loss");
    // The prediction is now 0.04: a residual of 0.5 is inside 0.8 · 1.
    m.step(&[1.0], &[Some(0.54)], 1.0, 1.0);
    assert_eq!(m.coefficients()[0], second, "inside the tube");
    // The third row joined the spread; a residual just past the tube
    // it draws now steps.
    let tube = 0.8 * m.target_variance()[0].sqrt();
    m.step(&[1.0], &[Some(0.04 + tube + 0.01)], 1.0, 1.0);
    assert_ne!(m.coefficients()[0], second, "outside the tube");
}

/// Each scale a loss draws in is kept exactly under that loss, one per
/// target: the residual's under Huber, the target's own under
/// epsilon-insensitive, neither under the rest. A state carrying one
/// its loss does not read, or missing one it does, is refused as it is
/// read (docs/PLAN.md task 202).
#[test]
fn a_state_is_refused_when_its_scales_disagree_with_its_loss() {
    let json = |loss: SgdLoss| {
        let mut c = cfg(2, loss);
        c.decay = Decay::Halflife(50.0); // JSON has no `inf`
        c.n_targets = 2;
        serde_json::to_value(Sgd::new(c).unwrap()).unwrap()
    };
    let loads = |v: &serde_json::Value| serde_json::from_value::<Sgd>(v.clone());
    let huber = json(SgdLoss::Huber { delta: 1.345 });
    let tube = json(SgdLoss::EpsilonInsensitive { eps: 0.1 });
    let squared = json(SgdLoss::Squared);
    for v in [&huber, &tube, &squared] {
        assert!(loads(v).is_ok(), "the control");
    }
    assert_eq!(huber["sig2"].as_array().unwrap().len(), 2);
    assert_eq!(tube["spread"].as_array().unwrap().len(), 2);
    for field in ["sig2", "wsig", "spread"] {
        assert!(squared[field].as_array().unwrap().is_empty(), "{field}");
    }
    assert!(huber["spread"].as_array().unwrap().is_empty());
    assert!(tube["sig2"].as_array().unwrap().is_empty());

    let mut lost = huber.clone();
    lost["sig2"] = serde_json::json!([]);
    let err = loads(&lost).unwrap_err().to_string();
    assert!(err.contains("residual variances do not match"), "{err}");
    let mut stray = squared.clone();
    stray["wsig"] = huber["wsig"].clone();
    let err = loads(&stray).unwrap_err().to_string();
    assert!(err.contains("residual variances do not match"), "{err}");

    let mut lost = tube.clone();
    lost["spread"].as_array_mut().unwrap().pop();
    let err = loads(&lost).unwrap_err().to_string();
    assert!(err.contains("target spreads do not match"), "{err}");
    let mut stray = huber.clone();
    stray["spread"] = tube["spread"].clone();
    let err = loads(&stray).unwrap_err().to_string();
    assert!(err.contains("target spreads do not match"), "{err}");
}

#[test]
fn adagrad_and_inv_scaling_also_converge() {
    for schedule in [
        LearningRate::AdaGrad,
        LearningRate::InvScaling { power: 0.25 },
    ] {
        let mut c = cfg(2, SgdLoss::Squared);
        c.schedule = schedule;
        c.learning_rate = if matches!(schedule, LearningRate::AdaGrad) {
            0.5
        } else {
            0.1
        };
        let b = fit(c, 30000, 23, |x, _| 1.5 * x[0] - 0.5 * x[1]);
        assert!((b[1] - 1.5).abs() < 0.15, "{schedule:?}: slope0 {}", b[1]);
        assert!((b[2] + 0.5).abs() < 0.15, "{schedule:?}: slope1 {}", b[2]);
    }
}

#[test]
fn l2_shrinks_toward_zero() {
    let plain = fit(cfg(1, SgdLoss::Squared), 5000, 29, |x, _| 2.0 * x[0]);
    let mut c = cfg(1, SgdLoss::Squared);
    c.l2 = 1.0;
    let shrunk = fit(c, 5000, 29, |x, _| 2.0 * x[0]);
    assert!(shrunk[1].abs() < plain[1].abs());
}

/// A log-link loss diverges without the cap. Deterministic rather than
/// stochastic: one large count is enough to push `eta` into the exp clamp,
/// after which the gradient is ~1e13 and a single step throws the
/// coefficients to ~1e11. (The same thing happens on real Poisson data,
/// where the heavy tail supplies the large count — measured on a 30k-row
/// stream, the intercept ran to -4e10.)
#[test]
fn poisson_diverges_without_gradient_clipping() {
    let run = |clip: f64| {
        let mut c = cfg(1, SgdLoss::Poisson);
        c.learning_rate = 0.02;
        c.clip_gradient = clip;
        c.min_weight = 0.0;
        let mut m = Sgd::new(c).unwrap();
        for i in 0..5 {
            m.step(&[1.0], &[Some(1e6)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        m.coefficients()[0].clone()
    };
    let unclipped = run(f64::INFINITY);
    assert!(
        unclipped.iter().any(|b| b.abs() > 1e9),
        "expected the unclipped fit to blow up, got {unclipped:?}"
    );
    let clipped = run(1e3);
    assert!(
        clipped.iter().all(|b| b.abs() < 1e3),
        "the cap should bound the coefficients, got {clipped:?}"
    );
}

/// A row any of whose gradients is not a finite number -- or, under
/// AdaGrad, one whose square is not, which `g2` never decays away --
/// teaches nothing, before `beta` moves, as `pa`, `ftrl` and `kalman`
/// skip theirs: with `clip_gradient = inf` ("disables it"), two rows
/// inside the input bound made a coefficient infinite and every later
/// prediction null (review round 4, CC2), and under AdaGrad one froze
/// the slope where it started, for good. The stream after them learns
/// the line.
#[test]
fn a_row_whose_gradient_overflows_teaches_nothing() {
    for schedule in [LearningRate::Constant, LearningRate::AdaGrad] {
        let mut c = cfg(1, SgdLoss::Squared);
        c.clip_gradient = f64::INFINITY;
        c.min_weight = 0.0;
        c.schedule = schedule;
        c.learning_rate = 0.2;
        let mut m = Sgd::new(c).unwrap();
        m.step(&[1e100], &[Some(1e100)], 0.0, 1.0);
        let before = m.coefficients()[0].clone();
        m.step(&[1e100], &[Some(0.0)], 1.0, 1.0);
        assert_eq!(
            m.coefficients()[0],
            before,
            "{schedule:?}: a row whose gradient overflows moved the fit"
        );
        let mut s = 37u64;
        let mut last = f64::NAN;
        for i in 0..20_000 {
            let x = [lcg(&mut s)];
            let p = m.step(&x, &[Some(1.0 + 0.5 * x[0])], 1.0, 1.0).pred[0];
            assert!(p.is_finite(), "{schedule:?}, row {i}: {p}");
            last = p - (1.0 + 0.5 * x[0]);
        }
        let beta = m.coefficients()[0].clone();
        assert!(
            (beta[0] - 1.0).abs() < 0.01 && (beta[1] - 0.5).abs() < 0.01,
            "{schedule:?}: {beta:?}, last error {last}"
        );
    }
}

/// The folded check (task 250) gives a short-circuiting `any`'s
/// verdict on every input: a NaN, an infinity or (under AdaGrad) a
/// square that overflows in any one slot is refused, and a finite
/// gradient never is, with and without a clip.
#[test]
fn the_folded_gradient_check_is_the_short_circuit_verdict() {
    let any_form = |beta: &[f64], z: &[f64], d: f64, w: f64, l2: f64, clip: f64, sq: bool| {
        !beta.iter().zip(z).any(|(b, &zi)| {
            let g = (d * zi * w + l2 * *b).clamp(-clip, clip);
            !if sq {
                (g * g).is_finite()
            } else {
                g.is_finite()
            }
        })
    };
    let k = 5;
    let beta = vec![0.25; k];
    let zeros = [0.0; 5];
    let mut refusals = 0;
    for squared in [false, true] {
        for clip in [f64::INFINITY, 1e300] {
            let z = vec![1.0; k];
            assert!(slopes_usable(
                &beta,
                &z,
                2.0,
                1.0,
                0.1,
                clip,
                squared.then_some(&zeros)
            ));
            assert!(any_form(&beta, &z, 2.0, 1.0, 0.1, clip, squared));
            for pos in 0..k {
                for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 1e200] {
                    let mut z = vec![1.0; k];
                    z[pos] = bad;
                    let want = if bad.is_nan() || (bad.is_infinite() && clip.is_infinite()) {
                        false
                    } else {
                        // A clipped infinity or 2e200: finite, its square not.
                        !squared
                    };
                    let got =
                        slopes_usable(&beta, &z, 2.0, 1.0, 0.1, clip, squared.then_some(&zeros));
                    assert_eq!(
                        got, want,
                        "squared {squared}, clip {clip}, slot {pos}: {bad}"
                    );
                    assert_eq!(got, any_form(&beta, &z, 2.0, 1.0, 0.1, clip, squared));
                    refusals += usize::from(!got);
                }
            }
        }
    }
    // Per slot: NaN under all four settings; each infinity unclipped
    // under both rates and clipped under AdaGrad; 2e200 under AdaGrad.
    assert_eq!(refusals, 5 * (4 + 3 + 3 + 2));
}

/// The one-pass check and step at a shared rate (task 314) gives the
/// two-pass form's verdict and, slot by slot, the bits its update
/// loop leaves in `β`, on finite rows, on rows refused for a NaN or an
/// unclipped infinity in one slot, under a clip and an `l2`.
#[test]
fn the_one_pass_step_is_the_two_pass_check_and_update() {
    let mut s = 11u64;
    let k = 7;
    let mut cases = 0;
    for clip in [f64::INFINITY, 1e3, 0.05] {
        for l2 in [0.0, 1e-3] {
            for bad in [None, Some(f64::NAN), Some(f64::INFINITY), Some(1e300)] {
                for pos in 0..k {
                    let beta: Vec<f64> = (0..k).map(|_| 3.0 * lcg(&mut s)).collect();
                    let mut z: Vec<f64> = (0..k).map(|_| 2.0 * lcg(&mut s)).collect();
                    if let Some(v) = bad {
                        z[pos] = v;
                    }
                    let (d, w, lr) = (1.5 * lcg(&mut s), 0.5 + lcg(&mut s).abs(), 0.01);
                    let want_ok = slopes_usable(&beta, &z, d, w, l2, clip, None);
                    let mut want = beta.clone();
                    for (b, &zi) in want.iter_mut().zip(&z) {
                        let g = (d * zi * w + l2 * *b).clamp(-clip, clip);
                        *b -= lr * g;
                    }
                    let mut next = vec![0.0; k];
                    let ok = slopes_stepped(&beta, &z, d, w, (l2, clip, lr), &mut next);
                    assert_eq!(ok, want_ok, "clip {clip}, l2 {l2}, {bad:?} at {pos}");
                    if ok {
                        let bits = |v: &[f64]| v.iter().map(|x| x.canon_bits()).collect::<Vec<_>>();
                        assert_eq!(bits(&next), bits(&want), "clip {clip}, l2 {l2}");
                    }
                    cases += usize::from(!ok);
                }
            }
        }
    }
    // A NaN refuses under every setting, an infinity only unclipped,
    // and 1e300 times a finite `d·w` never.
    assert_eq!(cases, k * 2 * (3 + 1));
}

/// The check inside `step`, slot by slot. A row whose gradient is
/// infinite in one slot only -- there through `l2·β`, the slot's
/// coefficient at 5e298 after a first row -- leaves every coefficient
/// where it was. A row whose gradient is 1e200 in one slot moves the fit
/// under a constant rate and, its square infinite, not under AdaGrad.
#[test]
fn a_step_refuses_a_row_by_any_one_slot() {
    let k = 4;
    let fresh = |schedule: LearningRate, l2: f64| {
        let mut c = cfg(k, SgdLoss::Squared);
        c.clip_gradient = f64::INFINITY;
        c.min_weight = 0.0;
        c.schedule = schedule;
        c.l2 = l2;
        Sgd::new(c).unwrap()
    };
    for pos in 0..k {
        let mut x = vec![1.0; k];
        x[pos] = 1e100;

        let mut m = fresh(LearningRate::Constant, 1e10);
        m.step(&x, &[Some(1e100)], 0.0, 1e100);
        let before = m.coefficients()[0].clone();
        assert!(before[1 + pos] > 1e298, "slot {pos}: {before:?}");
        m.step(&[1.0; 4], &[Some(0.0)], 1.0, 1.0);
        assert_eq!(
            m.coefficients()[0],
            before,
            "slot {pos}: an infinite gradient moved the fit"
        );

        for (schedule, refused) in [
            (LearningRate::Constant, false),
            (LearningRate::AdaGrad, true),
        ] {
            let mut m = fresh(schedule, 0.0);
            let before = m.coefficients()[0].clone();
            m.step(&x, &[Some(1e100)], 0.0, 1.0);
            let after = m.coefficients()[0].clone();
            assert_eq!(
                after == before,
                refused,
                "slot {pos}, {schedule:?}: {after:?}"
            );
            assert!(after.iter().all(|b| b.is_finite()), "{after:?}");
        }
    }
}

/// The folded check at the edges where the exact expression decides:
/// each case's verdict hangs on one product or sum overflowing, so an
/// arithmetic change to any term of `d·z_i·w + l2·β_i` flips it. Each
/// is held to the short-circuit form the check replaced (`c65ef07`).
#[test]
fn the_folded_check_decides_at_the_overflow_edge_of_each_term() {
    let any_form = |b: f64, z: f64, d: f64, w: f64, l2: f64, clip: f64, sq: bool| {
        let g = (d * z * w + l2 * b).clamp(-clip, clip);
        if sq {
            (g * g).is_finite()
        } else {
            g.is_finite()
        }
    };
    let inf = f64::INFINITY;
    // (β, z, d, w, l2, clip, squared, verdict)
    let cases = [
        // `d·z·w` overflows; `d·z + w` and `d·z / w` do not.
        (0.0, 1.0, 1e300, 1e10, 0.0, inf, false, false),
        // `d·z·w + l2·β` overflows; their difference is 0.
        (1e308, 1.0, 1e308, 1.0, 1.0, inf, false, false),
        // The same sum, inside the clip: finite.
        (1e308, 1.0, 1e308, 1.0, 1.0, 1e300, false, true),
        // Under AdaGrad: `(d·z·w)²` overflows at 1e160; `(d·z + w)²`
        // and `(d·z / w)²` do not.
        (0.0, 1.0, 1e150, 1e10, 0.0, inf, true, false),
        // `(d·z·w + l2·β)²` overflows at 2e154; the difference's
        // square is 0.
        (1e154, 1.0, 1e154, 1.0, 1.0, inf, true, false),
        // `l2·β` at 1e154 tips the square over; `l2 + β`, 2e77, does
        // not.
        (1e77, 1.0, 1e154, 1.0, 1e77, inf, true, false),
        // Each alone squares to a finite number.
        (0.0, 1.0, 1e154, 1.0, 0.0, inf, true, true),
        (1e77, 1.0, 0.0, 1.0, 1e77, inf, true, true),
        // A clip at 1e150 keeps the square finite.
        (1e154, 1.0, 1e154, 1.0, 1.0, 1e150, true, true),
    ];
    for (i, &(b, z, d, w, l2, clip, sq, want)) in cases.iter().enumerate() {
        let got = slopes_usable(
            &[0.5, b],
            &[1.0, z],
            d,
            w,
            l2,
            clip,
            sq.then_some(&[0.0; 2]),
        );
        assert_eq!(got, want, "case {i}");
        assert_eq!(got, any_form(b, z, d, w, l2, clip, sq), "case {i}");
    }
}

/// The intercept's check inside `step`, at its own overflow edge,
/// with every input inside the bound. A first row at 1e100 sets the
/// slope to 5e298; a second, its feature at 1e-50, predicts 5e248, so
/// the intercept's gradient `d·w` at a weight of 1e100 is infinite
/// where the slope's, `d·z·w`, is 5e298 (and `d + w`, `d / w` are
/// finite). The row is refused, as the short-circuit form refused it,
/// and the fit does not move. Without an intercept no slot's gradient
/// is `d·w`, and the same row moves the slope, to the bit.
#[test]
fn a_step_refuses_a_row_by_the_intercept_alone() {
    for fit_intercept in [true, false] {
        let mut c = cfg(1, SgdLoss::Squared);
        c.clip_gradient = f64::INFINITY;
        c.min_weight = 0.0;
        c.fit_intercept = fit_intercept;
        let mut m = Sgd::new(c).unwrap();
        m.step(&[1e100], &[Some(1e100)], 0.0, 1e100);
        let before = m.coefficients()[0].clone();
        let b1 = *before.last().unwrap();
        assert!(b1 > 1e298, "{before:?}");
        m.step(&[1e-50], &[Some(0.0)], 0.0, 1e100);
        let after = m.coefficients()[0].clone();
        if fit_intercept {
            assert_eq!(after, before);
        } else {
            let d = b1 * 1e-50;
            assert!((d * 1e100).is_infinite());
            let g = d * 1e-50 * 1e100 + 0.0 * b1;
            assert_eq!(after, vec![b1 - 0.05 * g]);
            assert!(after[0] < b1);
        }
    }
}

#[test]
fn clip_gradient_bounds_the_step() {
    let mut c = cfg(1, SgdLoss::Squared);
    c.clip_gradient = 1.0;
    c.min_weight = 0.0;
    let mut m = Sgd::new(c).unwrap();
    // one absurd row: with lr 0.05 and the clip at 1, |db| <= 0.05
    m.step(&[1.0], &[Some(1e9)], 0.0, 1.0);
    assert!(m.coefficients()[0][1].abs() <= 0.05 + 1e-12);
}

#[test]
fn null_target_is_predict_only() {
    let mut c = cfg(1, SgdLoss::Squared);
    c.min_weight = 0.0;
    let mut m = Sgd::new(c).unwrap();
    let mut s = 31u64;
    for i in 0..50 {
        let x = [lcg(&mut s)];
        m.step(&x, &[Some(2.0 * x[0])], if i == 0 { 0.0 } else { 1.0 }, 1.0);
    }
    let before = m.coefficients()[0].clone();
    let st = m.step(&[0.5], &[None], 1.0, 1.0);
    assert!(st.pred[0].is_finite());
    assert_eq!(m.coefficients()[0], before);
}

fn constrained(k: usize, lo: f64, hi: f64, sum: Option<f64>) -> SgdCfg {
    let mut c = cfg(k, SgdLoss::Squared);
    c.constraint = Some(Constraint {
        lo: vec![lo; k],
        hi: vec![hi; k],
        sum,
    });
    c
}

#[test]
fn a_constraint_is_validated_by_name() {
    let mut c = constrained(2, 0.0, 1.0, Some(3.0));
    assert_eq!(
        c.validate().unwrap_err(),
        "sgd: coef_sum must be one the bounds allow, from 0 to 2, but it is 3. Set \
         coef_sum inside that range, or widen coef_min and coef_max."
    );
    c.constraint.as_mut().unwrap().lo = vec![0.0];
    assert!(
        c.validate()
            .unwrap_err()
            .contains("coef_min must give one bound for each of the 2 features")
    );
}

#[test]
fn a_simplex_starts_uniform_and_stays_on_it() {
    // Truth on the simplex: the fit lands on it, and every intermediate
    // coefficient vector is on it too (sum exact to rounding, slopes >= 0).
    let mut m = Sgd::new(constrained(3, 0.0, f64::INFINITY, Some(1.0))).unwrap();
    let start = &m.coefficients()[0];
    assert_eq!(start[0], 0.0);
    for b in &start[1..] {
        assert!((b - 1.0 / 3.0).abs() <= 1e-15);
    }
    let mut s = 11u64;
    for i in 0..20000 {
        let x = [lcg(&mut s), lcg(&mut s), lcg(&mut s)];
        let y = 0.2 * x[0] + 0.5 * x[1] + 0.3 * x[2] + 0.05 * lcg(&mut s);
        m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        let b = &m.coefficients()[0][1..];
        assert!(b.iter().all(|v| *v >= 0.0), "row {i}: {b:?}");
        assert!(
            (b.iter().sum::<f64>() - 1.0).abs() <= 1e-12,
            "row {i}: {b:?}"
        );
    }
    let b = &m.coefficients()[0];
    for (got, want) in b[1..].iter().zip([0.2, 0.5, 0.3]) {
        assert!((got - want).abs() < 0.03, "{b:?}");
    }
}

#[test]
fn a_box_holds_where_the_truth_lies_outside_it() {
    // Uncorrelated unit-scale features: the constrained least-squares
    // optimum is the truth clamped into the box.
    let mut c = constrained(2, 0.0, 0.5, None);
    c.learning_rate = 0.02;
    let b = fit(c, 20000, 3, |x, s| -0.4 * x[0] + 0.9 * x[1] + 0.1 * lcg(s));
    assert_eq!(b[1], 0.0, "{b:?}");
    assert_eq!(b[2], 0.5, "{b:?}");
    assert!(b[0].abs() < 0.05, "{b:?}");
}

#[test]
fn a_pinned_slope_never_moves_and_the_rest_learn() {
    let mut c = cfg(2, SgdLoss::Squared);
    c.constraint = Some(Constraint {
        lo: vec![0.7, f64::NEG_INFINITY],
        hi: vec![0.7, f64::INFINITY],
        sum: None,
    });
    let b = fit(c, 20000, 5, |x, _| 1.5 * x[0] - 0.5 * x[1] + 0.25);
    assert_eq!(b[1], 0.7);
    assert!((b[2] + 0.5).abs() < 0.05, "{b:?}");
}

#[test]
fn the_constraint_holds_in_the_callers_units_under_scaling() {
    // The bound is on c_i = b_i / scale_i; the projection is taken with
    // the scales the coefficients are reported with, after every row.
    let mut c = constrained(2, 0.0, 0.01, Some(0.01));
    c.standardize = true;
    c.min_weight = 0.0;
    let mut m = Sgd::new(c).unwrap();
    let mut s = 17u64;
    for i in 0..5000 {
        let x = [1000.0 * lcg(&mut s), 0.001 * lcg(&mut s)];
        let y = 0.002 * x[0] + 900.0 * x[1];
        m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        let b = &m.coefficients()[0][1..];
        assert!(
            b.iter().all(|v| *v >= -1e-15 && *v <= 0.01 + 1e-15),
            "row {i}: {b:?}"
        );
        assert!(
            (b.iter().sum::<f64>() - 0.01).abs() <= 1e-12,
            "row {i}: {b:?}"
        );
    }
    // The truth (0.002, 900) has one slope inside the box and one far
    // outside. The fit keeps the first at its truth and gives the second
    // whatever the sum leaves: in the standardized metric that is the
    // nearest feasible point, since a unit of c_1 is 10^6 times cheaper
    // than a unit of c_0.
    let b = &m.coefficients()[0];
    assert!((b[1] - 0.002).abs() <= 2e-4, "{b:?}");
    assert!((b[2] - 0.008).abs() <= 2e-4, "{b:?}");
}

#[test]
fn a_zero_weight_row_leaves_a_constrained_fit_alone() {
    let mut m = Sgd::new(constrained(2, 0.0, f64::INFINITY, Some(1.0))).unwrap();
    let mut s = 23u64;
    for i in 0..50 {
        let x = [lcg(&mut s), lcg(&mut s)];
        m.step(&x, &[Some(x[0])], if i == 0 { 0.0 } else { 1.0 }, 1.0);
    }
    let before = m.coefficients().to_vec();
    m.step(&[0.3, -0.2], &[Some(5.0)], 1.0, 0.0);
    m.step(&[0.3, -0.2], &[None], 1.0, 1.0);
    assert_eq!(m.coefficients(), &before[..]);
}

#[test]
fn a_constrained_state_roundtrips() {
    let mut m1 = Sgd::new(constrained(2, 0.0, f64::INFINITY, Some(1.0))).unwrap();
    let mut s = 29u64;
    for i in 0..60 {
        let x = [lcg(&mut s), lcg(&mut s)];
        m1.step(&x, &[Some(x[0])], if i == 0 { 0.0 } else { 1.0 }, 1.0);
    }
    let bytes = rmp_serde::to_vec(&m1.state()).unwrap();
    let mut m2 = Sgd::restore(&rmp_serde::from_slice(&bytes).unwrap()).unwrap();
    assert_eq!(m2.cfg().constraint, m1.cfg().constraint);
    for _ in 0..60 {
        let x = [lcg(&mut s), lcg(&mut s)];
        assert_eq!(
            m1.step(&x, &[Some(x[0])], 1.0, 1.0).pred,
            m2.step(&x, &[Some(x[0])], 1.0, 1.0).pred
        );
        assert_eq!(m1.coefficients(), m2.coefficients());
    }
}

#[test]
fn state_roundtrip() {
    let mut c = cfg(2, SgdLoss::Squared);
    c.schedule = LearningRate::AdaGrad;
    let mut m1 = Sgd::new(c).unwrap();
    let mut s = 37u64;
    let rows: Vec<([f64; 2], f64)> = (0..120)
        .map(|_| {
            let x = [lcg(&mut s), lcg(&mut s)];
            (x, x[0] - 0.5 * x[1])
        })
        .collect();
    for (i, (x, y)) in rows[..60].iter().enumerate() {
        m1.step(x, &[Some(*y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
    }
    let bytes = rmp_serde::to_vec(&m1.state()).unwrap();
    let mut m2 = Sgd::restore(&rmp_serde::from_slice(&bytes).unwrap()).unwrap();
    for (x, y) in &rows[60..] {
        assert_eq!(
            m1.step(x, &[Some(*y)], 1.0, 1.0).pred,
            m2.step(x, &[Some(*y)], 1.0, 1.0).pred
        );
    }
}

#[test]
fn rejects_bad_config() {
    let mut c = cfg(1, SgdLoss::Squared);
    c.learning_rate = 0.0;
    assert!(Sgd::new(c).is_err());
    assert!(Sgd::new(cfg(1, SgdLoss::Quantile { tau: 0.0 })).is_err());
    assert!(Sgd::new(cfg(1, SgdLoss::Huber { delta: -1.0 })).is_err());
}

/// A target's `min_weight` counts only the rows that carried it, and
/// `n_eff` stays every row's (hard rule 8, docs/PLAN.md task 115 (d)):
/// ten rows with no target met `min_weight = 3` with every coefficient
/// at zero. A zero-weight row with the target (row 12) only ages it; the
/// scaler, which a null target does not stop, is on.
#[test]
fn min_periods_counts_only_the_rows_that_carried_the_target() {
    let mut c = cfg(2, SgdLoss::Squared);
    c.min_weight = 3.0;
    c.standardize = true;
    let mut m = Sgd::new(c).unwrap();
    let mut s = 7u64;
    for i in 0..16usize {
        let x = [lcg(&mut s), lcg(&mut s)];
        let y = (i >= 10).then(|| x[0] - x[1]);
        let out = m.step(&x, &[y], 1.0, if i == 12 { 0.0 } else { 1.0 });
        assert_eq!(out.n_eff, (i - usize::from(i > 12)) as f64, "row {i}");
        assert_eq!(out.pred[0].is_nan(), i < 14, "row {i}");
        assert_eq!(m.predict(&x, 1.0).pred[0].is_nan(), i < 13, "row {i}");
    }
    assert_eq!(m.target_weights(), &[5.0]);
}

/// A state without target weights is refused, where it loaded with each
/// target at the shared weight -- the layout of schema 19, which this
/// build refuses by its version, and the repair a damaged file reached
/// (docs/PLAN.md task 198).
#[test]
fn a_state_without_target_weights_is_refused() {
    let mut c = cfg(2, SgdLoss::Squared);
    c.decay = Decay::Halflife(50.0); // JSON has no `inf`
    let mut m = Sgd::new(c).unwrap();
    let mut s = 7u64;
    for i in 0..8 {
        let x = [lcg(&mut s), lcg(&mut s)];
        m.step(&x, &[(i >= 5).then(|| x[0])], 1.0, 1.0);
    }
    assert!(m.target_weights()[0] < m.n_eff());
    let mut v = serde_json::to_value(&m).unwrap();
    assert!(v.as_object_mut().unwrap().remove("w_target").is_some());
    let err = serde_json::from_value::<Sgd>(v).unwrap_err().to_string();
    assert!(err.contains("missing field `w_target`"), "{err}");
    let mut short = serde_json::to_value(&m).unwrap();
    short["w_target"] = serde_json::json!([]);
    let err = serde_json::from_value::<Sgd>(short)
        .unwrap_err()
        .to_string();
    assert!(err.contains("target weights have the wrong shape"), "{err}");
}

/// Without `standardize` there is no scaler in either schema, and a
/// file written before the field existed (no `scaler` key at all) still
/// loads as one without.
#[test]
fn a_state_without_a_scaler_loads_in_every_layout() {
    let mut c = cfg(2, SgdLoss::Squared);
    c.decay = Decay::Halflife(50.0); // JSON has no `inf`
    let m = Sgd::new(c).unwrap();
    let mut v = serde_json::to_value(&m).unwrap();
    assert!(v["scaler"].is_null());
    let back: Sgd = serde_json::from_value(v.clone()).unwrap();
    assert!(back.scaler.is_none());
    v.as_object_mut().unwrap().remove("scaler");
    let back: Sgd = serde_json::from_value(v).unwrap();
    assert!(back.scaler.is_none());
}

/// A state whose cfg says `standardize` carries its scaler, and one
/// on AdaGrad its sums, one per target -- and neither carries what its
/// cfg does not ask for. A file that lost the scaler loaded as a model
/// reading raw inputs with coefficients learned on standardized ones;
/// one that lost its sums panicked on its first AdaGrad step
/// (review 2026-09-12, S16).
#[test]
fn a_state_is_refused_when_its_scaler_or_sums_disagree_with_its_cfg() {
    let json = |c: &SgdCfg| serde_json::to_value(Sgd::new(c.clone()).unwrap()).unwrap();
    let loads = |v: &serde_json::Value| serde_json::from_value::<Sgd>(v.clone()).is_ok();
    let mut plain = cfg(2, SgdLoss::Squared);
    plain.decay = Decay::Halflife(50.0); // JSON has no `inf`

    let mut scaled = plain.clone();
    scaled.standardize = true;
    let with = json(&scaled);
    assert!(loads(&with), "the control");
    let mut lost = with.clone();
    lost.as_object_mut().unwrap().remove("scaler");
    assert!(!loads(&lost), "standardize without its scaler");
    let mut stray = json(&plain);
    stray["scaler"] = with["scaler"].clone();
    assert!(!loads(&stray), "a scaler the cfg does not ask for");

    let mut ada = plain.clone();
    ada.schedule = LearningRate::AdaGrad;
    let with = json(&ada);
    assert!(loads(&with), "the control");
    let mut lost = with.clone();
    lost["g2"] = serde_json::json!([]);
    assert!(!loads(&lost), "AdaGrad without its sums");
    let mut stray = json(&plain);
    stray["g2"] = with["g2"].clone();
    assert!(!loads(&stray), "sums a constant rate does not keep");
}

/// The named msgpack of `value` with `edit` applied, read back: the path
/// a saved state takes into the model, whose check is in its `TryFrom`.
fn reread<T: serde::Serialize + serde::de::DeserializeOwned>(
    value: &T,
    edit: impl FnOnce(&mut rmpv::Value),
) -> Result<T, String> {
    let bytes = rmp_serde::to_vec_named(value).unwrap();
    let mut v = rmpv::decode::read_value(&mut bytes.as_slice()).unwrap();
    edit(&mut v);
    let mut out = Vec::new();
    rmpv::encode::write_value(&mut out, &v).unwrap();
    rmp_serde::from_slice(&out).map_err(|e| e.to_string())
}

/// The first coefficient vector of the state, one entry short.
fn shorten_first(v: &mut rmpv::Value, field: &str) {
    let rmpv::Value::Map(entries) = v else {
        panic!("a map")
    };
    let (_, x) = entries
        .iter_mut()
        .find(|(k, _)| k.as_str() == Some(field))
        .unwrap_or_else(|| panic!("no {field}"));
    let rmpv::Value::Array(rows) = x else {
        panic!("{field} is a list")
    };
    let rmpv::Value::Array(first) = &mut rows[0] else {
        panic!("{field}[0] is a list")
    };
    first.pop();
}

/// A state whose coefficients are not its cfg's shape is refused as it is
/// read, which is how a saved state reaches `restore`, where it loaded and
/// panicked on the first step (review 2026-09-18, B3; docs/PLAN.md task
/// 111: every other model had this test).
#[test]
fn a_state_of_the_wrong_shape_is_refused() {
    let m = Sgd::new(cfg(2, SgdLoss::Squared)).unwrap();
    assert!(reread(&m, |_| {}).is_ok(), "the control");
    let err = reread(&m, |v| shorten_first(v, "beta")).unwrap_err();
    assert!(err.contains("wrong shape"), "{err}");
}

/// A state whose AdaGrad sums do not match its schedule is refused in
/// the words of the schedule it has: sums missing under AdaGrad, sums
/// present under another (task 358).
#[test]
fn adagrad_sums_that_do_not_match_the_schedule_are_refused_by_it() {
    let set_g2 = |v: &mut rmpv::Value, to: rmpv::Value| {
        let rmpv::Value::Map(entries) = v else {
            panic!("a map")
        };
        entries
            .iter_mut()
            .find(|(k, _)| k.as_str() == Some("g2"))
            .expect("g2")
            .1 = to;
    };
    let mut c = cfg(2, SgdLoss::Squared);
    c.schedule = LearningRate::AdaGrad;
    let adagrad = Sgd::new(c).unwrap();
    let err = reread(&adagrad, |v| set_g2(v, rmpv::Value::Array(vec![]))).unwrap_err();
    assert!(
        err.contains("do not match its setting schedule=\"adagrad\". Load"),
        "{err}"
    );
    let constant = Sgd::new(cfg(2, SgdLoss::Squared)).unwrap();
    let sums = rmpv::Value::Array(vec![rmpv::Value::Array(vec![rmpv::Value::F64(0.0); 3])]);
    let err = reread(&constant, |v| set_g2(v, sums)).unwrap_err();
    assert!(
        err.contains("do not match its setting schedule, which is not \"adagrad\". Load"),
        "{err}"
    );
}

/// Each shape check refuses on its own: a state one target short of its
/// cfg, and one whose scaler is another width than the row, are each
/// refused with every other part of the state in shape (task 158).
#[test]
fn a_state_missing_a_target_or_with_a_scaler_of_another_width_is_refused() {
    let json = |c: &SgdCfg| serde_json::to_value(Sgd::new(c.clone()).unwrap()).unwrap();
    let loads = |v: &serde_json::Value| serde_json::from_value::<Sgd>(v.clone());
    let mut two = cfg(2, SgdLoss::Squared);
    two.decay = Decay::Halflife(50.0); // JSON has no `inf`
    two.n_targets = 2;
    let mut v = json(&two);
    assert!(loads(&v).is_ok(), "the control");
    v["beta"].as_array_mut().unwrap().pop();
    let err = loads(&v).unwrap_err().to_string();
    assert!(
        err.contains("sgd: its saved state has the wrong shape"),
        "{err}"
    );

    let mut scaled = two.clone();
    scaled.standardize = true;
    let mut wider = scaled.clone();
    wider.n_features = 3;
    let mut v = json(&scaled);
    assert!(loads(&v).is_ok(), "the control");
    v["scaler"] = json(&wider)["scaler"].clone();
    let err = loads(&v).unwrap_err().to_string();
    assert!(
        err.contains("sgd: its saved state has the wrong shape"),
        "{err}"
    );
}

/// `dot` is `beta · [1, z]` whatever the lanes it sums in: 3, 8, 11 and
/// 19 features, every slot nonzero, against the plain sum (task 158).
#[test]
fn dot_is_the_plain_sum_in_every_lane() {
    let mut s = 71u64;
    for k in [3usize, 5, 7, 8, 11, 19] {
        for off in [0usize, 1] {
            let beta: Vec<f64> = (0..k + off).map(|_| 0.5 + lcg(&mut s)).collect();
            let z: Vec<f64> = (0..k).map(|_| 1.5 + lcg(&mut s)).collect();
            let plain: f64 = (0..k).map(|i| beta[off + i] * z[i]).sum::<f64>()
                + if off == 1 { beta[0] } else { 0.0 };
            let got = dot(&beta, off, &z);
            assert!(
                (got - plain).abs() <= 1e-13 * plain.abs().max(1.0),
                "k {k}, off {off}: {got} against {plain}"
            );
        }
    }
}

/// The logistic link is `1 / (1 + e^-eta)` on both sides of 0, and the
/// quantile loss's gradient at `y = p` is `-tau` (`1{y < p} − tau`,
/// the module table) (task 158).
#[test]
fn the_logistic_link_is_the_sigmoid_and_a_quantile_tie_is_minus_tau() {
    let lg = Sgd::new(cfg(1, SgdLoss::Logistic)).unwrap();
    for eta in [-3.0, -0.5, 0.25, 2.0] {
        let want = 1.0 / (1.0 + f64::exp(-eta));
        let got = lg.link(eta);
        assert!(
            (got - want).abs() <= 1e-15,
            "eta {eta}: {got} against {want}"
        );
    }
    let q = Sgd::new(cfg(1, SgdLoss::Quantile { tau: 0.9 })).unwrap();
    assert_eq!(q.dloss(1.0, 1.0, Some(1.0)), -0.9);
}

/// Without an intercept, `standardize` divides each feature by the root
/// of its raw second moment, the row admitted at one row's weight of the
/// stream, `ŵ = Σw²/Σw` over the warm-up's rows, while the scaler warms
/// up and at `min(ŵ, w)` once the fit is held, the history kept at no
/// less than `ŵ·n/(n + 1)` and then `ŵ` (`scaler_decay`; review C1), and
/// leaves a feature with none (here one always 0) at a scale of 1. While the
/// scaler warms up the betas are in those coordinates and the
/// coefficients are the betas over the root of the raw moment as it
/// stands; on the row Kish's count of the weights reaches 22 they are
/// read out so once and held in the caller's units, the prediction `xᵀb`
/// and the step `Δβ_f = −lr (d z_f w + l2 s_f b_f)`, taken in the row's
/// coordinates, mapped back by `Δb_f = Δβ_f / s_f` (the module docs).
/// Held to SGD written from them over the raw sums `Σ w`, `Σ w x²`
/// (task 158; docs/PLAN.md task 206).
#[test]
fn standardizing_without_an_intercept_is_the_raw_moment_scaling() {
    let mut c = cfg(3, SgdLoss::Squared);
    c.fit_intercept = false;
    c.standardize = true;
    c.min_weight = 0.0;
    c.learning_rate = 0.1;
    c.l2 = 0.01;
    c.decay = Decay::Halflife(10.0);
    let mut m = Sgd::new(c).unwrap();
    let (mut w_sum, mut s2, mut b) = (0.0f64, [0.0f64; 3], [0.0f64; 3]);
    let (mut kish_w, mut kish_w2, mut held) = (0.0f64, 0.0f64, false);
    let mut s = 73u64;
    for i in 0..300 {
        let x = [3.0 + 2.0 * lcg(&mut s), 0.01 * lcg(&mut s), 0.0];
        let y = 0.4 * x[0] - 50.0 * x[1] + 0.1 * lcg(&mut s);
        let w = 0.5 + 0.5 * (lcg(&mut s) + 1.0);
        let d_clock = if i == 0 { 0.0 } else { 1.0 };
        let lam = 0.5f64.powf(d_clock / 10.0);
        let one = if kish_w > 0.0 { kish_w2 / kish_w } else { 1.0 };
        let n = if kish_w2 > 0.0 {
            kish_w * kish_w / kish_w2
        } else {
            0.0
        };
        let (admit, floor) = if held {
            (one.min(w), one)
        } else {
            (one, one * (n / (n + 1.0)))
        };
        let lam_s = if floor > 0.0 && w_sum > 0.0 {
            lam.max((floor / w_sum).min(1.0))
        } else {
            lam
        };
        let scale: Vec<f64> = (0..3)
            .map(|f| {
                let raw = (lam_s * s2[f] + admit * x[f] * x[f]) / (lam_s * w_sum + admit);
                if raw > 0.0 { raw.sqrt() } else { 1.0 }
            })
            .collect();
        let z: Vec<f64> = (0..3).map(|f| x[f] / scale[f]).collect();
        let want: f64 = if held {
            (0..3).map(|f| b[f] * x[f]).sum()
        } else {
            (0..3).map(|f| b[f] * z[f]).sum()
        };
        let got = m.step(&x, &[Some(y)], d_clock, w).pred[0];
        assert!(
            (got - want).abs() <= 1e-10 * want.abs().max(1.0),
            "row {i}: {got} against {want}"
        );
        for f in 0..3 {
            if held {
                let step = -0.1 * ((want - y) * z[f] * w + 0.01 * scale[f] * b[f]);
                b[f] += step / scale[f];
            } else {
                b[f] -= 0.1 * ((want - y) * z[f] * w + 0.01 * b[f]);
            }
            s2[f] = lam_s * s2[f] + w * x[f] * x[f];
        }
        w_sum = lam_s * w_sum + w;
        if !held {
            kish_w += w;
            kish_w2 += w * w;
            if kish_w * kish_w / kish_w2 >= crate::WARMUP_ROWS {
                held = true;
                for f in 0..3 {
                    let raw = s2[f] / w_sum;
                    if raw > 0.0 {
                        b[f] /= raw.sqrt();
                    }
                }
            }
        }
        assert_eq!(m.warmup().switched(), held, "row {i}");
        assert_eq!(m.scaler_warming(), !held, "row {i}");
    }
    let mut raw_cfg = cfg(3, SgdLoss::Squared);
    raw_cfg.standardize = false;
    let mut raw = Sgd::new(raw_cfg).unwrap();
    raw.step(&[1.0, 2.0, 3.0], &[Some(1.0)], 0.0, 1.0);
    assert!(!raw.scaler_warming(), "no scaler, no warm-up");
    let got = &m.coefficients()[0];
    for f in 0..3 {
        assert!(
            (got[f] - b[f]).abs() <= 1e-10 * b[f].abs().max(1e-3),
            "slot {f}: {} against {}",
            got[f],
            b[f]
        );
    }
    assert!(
        (got[0] - 0.4).abs() < 0.05 && (got[1] + 50.0).abs() < 5.0,
        "{got:?}"
    );
    assert_eq!(got[2], 0.0, "the feature with no moment");
}

/// AdaGrad, from the module docs: each slot's sum of squared gradients
/// decays on the clock and grows by `g²`, and the slot steps by
/// `lr / (√G + 1e-8) · g`, `g = d z w (+ l2 b off the intercept)`. With
/// and without an intercept, at weights other than 1 and through a gap
/// of 40 half-lives that re-opens the rate (task 158).
#[test]
fn adagrad_is_its_recursion() {
    for fit_intercept in [true, false] {
        let mut c = cfg(3, SgdLoss::Squared);
        c.fit_intercept = fit_intercept;
        c.schedule = LearningRate::AdaGrad;
        c.min_weight = 0.0;
        c.learning_rate = 0.3;
        c.l2 = 0.05;
        c.decay = Decay::Halflife(5.0);
        let mut m = Sgd::new(c).unwrap();
        let off = usize::from(fit_intercept);
        let k = 3 + off;
        let (mut b, mut g2) = (vec![0.0f64; k], vec![0.0f64; k]);
        let mut s = 79u64;
        for i in 0..200 {
            let x = [lcg(&mut s), 2.0 * lcg(&mut s), 0.5 * lcg(&mut s)];
            let y = 1.0 + 0.7 * x[0] - 0.3 * x[1] + 0.05 * lcg(&mut s);
            let w = 0.25 + 1.5 * (lcg(&mut s) + 1.0) / 2.0;
            let d_clock = match i {
                0 => 0.0,
                100 => 200.0,
                _ => 1.0,
            };
            let lam = 0.5f64.powf(d_clock / 5.0);
            g2.iter_mut().for_each(|g| *g *= lam);
            let zrow: Vec<f64> = (0..off).map(|_| 1.0).chain(x).collect();
            let want: f64 = (0..k).map(|f| b[f] * zrow[f]).sum();
            let got = m.step(&x, &[Some(y)], d_clock, w).pred[0];
            assert!(
                (got - want).abs() <= 1e-12 * want.abs().max(1.0),
                "intercept {fit_intercept}, row {i}: {got} against {want}"
            );
            for f in 0..k {
                let pen = if f < off { 0.0 } else { 0.05 * b[f] };
                let g = (want - y) * zrow[f] * w + pen;
                g2[f] += g * g;
                b[f] -= 0.3 / (g2[f].sqrt() + 1e-8) * g;
            }
        }
    }
}

/// The scaler learns the row as `[1, x]`: its intercept slot holds the
/// constant 1 (task 158).
#[test]
fn the_scaler_learns_the_intercepts_constant() {
    let mut c = cfg(2, SgdLoss::Squared);
    c.standardize = true;
    let mut m = Sgd::new(c).unwrap();
    let mut s = 83u64;
    for i in 0..20 {
        let x = [lcg(&mut s), lcg(&mut s)];
        m.step(&x, &[Some(x[0])], if i == 0 { 0.0 } else { 1.0 }, 1.0);
    }
    let sc = m.scaler.as_ref().unwrap();
    assert_eq!((sc.mean(0), sc.var(0)), (1.0, 0.0));
}

/// Under `standardize` a row that moves the scales re-projects every
/// target, also one that did not learn from it: target 1 carries a value
/// on one row in three, and its coefficients stay in the box on every
/// row, where the scales shrinking would carry them past it. A row of
/// weight 0 moves no scale and leaves every coefficient where it was, to
/// the bit (task 158).
#[test]
fn every_target_is_reprojected_when_the_scales_move() {
    let mut c = constrained(2, 0.0, 0.01, None);
    c.n_targets = 2;
    c.standardize = true;
    c.min_weight = 0.0;
    c.decay = Decay::Halflife(20.0);
    let mut m = Sgd::new(c).unwrap();
    let mut s = 89u64;
    let mut moved = 0;
    for i in 0..400 {
        let x = [10.0 * lcg(&mut s), 0.1 * lcg(&mut s)];
        let y = 0.5 * x[0] + 5.0 * x[1];
        let y1 = (i % 3 == 0).then_some(y);
        let before = m.beta[1].clone();
        m.step(&x, &[Some(y), y1], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        moved += usize::from(y1.is_none() && m.beta[1] != before);
        for (j, b) in m.coefficients().iter().enumerate() {
            assert!(
                b[1..].iter().all(|v| *v >= -1e-15 && *v <= 0.01 + 1e-15),
                "row {i}, target {j}: {b:?}"
            );
        }
    }
    assert!(
        moved > 0,
        "target 1 was re-projected on rows it did not learn"
    );

    let mut c = constrained(2, f64::NEG_INFINITY, f64::INFINITY, Some(0.3));
    c.n_targets = 2;
    c.standardize = true;
    c.decay = Decay::Halflife(20.0);
    let mut m = Sgd::new(c).unwrap();
    for i in 0..100 {
        let x = [10.0 * lcg(&mut s), 0.1 * lcg(&mut s)];
        let y = 0.5 * x[0] + 5.0 * x[1];
        m.step(
            &x,
            &[Some(y), Some(-y)],
            if i == 0 { 0.0 } else { 1.0 },
            1.0,
        );
    }
    let before = m.beta.clone();
    m.step(&[3.0, -0.05], &[Some(1.0), Some(1.0)], 1.0, 0.0);
    assert_eq!(m.beta, before);
}

/// The bank's per-target `min_weight` gate reads each target's own
/// weight through the trait: `true`, and one entry a target, the decayed
/// weight of the rows that carried it -- a sparse second target its own
/// rows only (task 158, E14; the mutation run of 2026-10-05 left the
/// body replaced by `true` or `false` alive).
#[test]
fn the_trait_reports_each_targets_own_weight() {
    let mut c = cfg(2, SgdLoss::Squared);
    c.n_targets = 2;
    c.decay = Decay::Halflife(10.0);
    let mut m = Sgd::new(c).unwrap();
    let lam = Decay::Halflife(10.0).factor(1.0);
    let (mut want0, mut want1) = (0.0f64, 0.0f64);
    let mut s = 3u64;
    for i in 0..30 {
        let x = [lcg(&mut s), lcg(&mut s)];
        let w = 0.5 + lcg(&mut s).abs();
        let y1 = (i % 4 == 1).then_some(x[1]);
        m.step(&x, &[Some(x[0]), y1], if i == 0 { 0.0 } else { 1.0 }, w);
        let f = if i == 0 { 1.0 } else { lam };
        want0 = f * want0 + w;
        want1 = f * want1 + if y1.is_some() { w } else { 0.0 };
    }
    let mut out = vec![7.0; 5];
    assert!(m.target_n_eff_into(&mut out));
    assert_eq!(out.len(), 2, "{out:?}");
    assert!(
        (out[0] - want0).abs() <= 1e-12 * want0,
        "{out:?} against {want0}"
    );
    assert!(
        (out[1] - want1).abs() <= 1e-12 * want1,
        "{out:?} against {want1}"
    );
    assert!(
        out[1] < 0.5 * out[0],
        "the sparse target's own weight: {out:?}"
    );
}

/// Bits of a prediction, every NaN one value: a withheld row is NaN
/// whatever its sign or payload.
fn bits_of(v: f64) -> u64 {
    if v.is_nan() {
        f64::NAN.to_bits()
    } else {
        v.to_bits()
    }
}

/// `huber_delta` is in units of the target's EW residual standard
/// deviation σ, as `huber`'s is (`robust.rs`): a target scaled by a power
/// of two fits as the unscaled one does, scaled by it, to the bit, since
/// every number the step forms -- the residual, σ, the cut, the gradient
/// -- scales with it exactly. In the target's units the cut bound on the
/// outliers at scale 1, never at 2^-10 and on every row at 2^10 (review
/// round 4, CC4; docs/PLAN.md task 195, U1). The squared loss, which has
/// no cut, is the control.
#[test]
fn the_huber_cut_is_in_units_of_the_residual_std() {
    for loss in [SgdLoss::Huber { delta: 1.345 }, SgdLoss::Squared] {
        let run = |c: f64| -> Vec<u64> {
            let mut cf = cfg(2, loss);
            cf.standardize = true;
            cf.clip_gradient = f64::INFINITY;
            let mut m = Sgd::new(cf).unwrap();
            let mut s = 5u64;
            (0..400)
                .map(|i| {
                    let x = [lcg(&mut s), lcg(&mut s)];
                    let noise = lcg(&mut s);
                    let outlier = if i % 37 == 5 { 20.0 } else { 0.0 };
                    let y = c * (0.5 + 2.0 * x[0] - x[1] + 0.3 * noise + outlier);
                    let d = if i == 0 { 0.0 } else { 1.0 };
                    bits_of(m.step(&x, &[Some(y)], d, 1.0).pred[0] / c)
                })
                .collect()
        };
        let one = run(1.0);
        for c in [2f64.powi(-10), 2f64.powi(10)] {
            let scaled = run(c);
            let differ = scaled.iter().zip(&one).filter(|(a, b)| a != b).count();
            assert_eq!(
                differ, 0,
                "{loss:?} at scale {c}: {differ} of 400 rows differ"
            );
        }
    }
}

/// Under `epsilon_insensitive` a target scaled by a power of two, with
/// the learning rate scaled by it -- the rate of a sign-valued gradient
/// is in the target's units (the module docs) -- fits as the unscaled
/// one does, scaled by it, to the bit: the band is in units of the
/// target's own spread, which scales with it exactly. The stream sits at
/// a level of 1,000 in a spread of 2, under a decay and irregular
/// weights (docs/PLAN.md task 202).
#[test]
fn the_band_scales_with_the_target() {
    let run = |c: f64| -> Vec<u64> {
        let mut cf = cfg(2, SgdLoss::EpsilonInsensitive { eps: 0.1 });
        cf.standardize = true;
        cf.clip_gradient = f64::INFINITY;
        cf.decay = Decay::Halflife(50.0);
        cf.learning_rate = 5.0 * c;
        cf.schedule = LearningRate::InvScaling { power: 0.5 };
        let mut m = Sgd::new(cf).unwrap();
        let mut s = 5u64;
        (0..600)
            .map(|i| {
                let x = [lcg(&mut s), lcg(&mut s)];
                let y = c * (1000.5 + 2.0 * x[0] - x[1] + 0.3 * lcg(&mut s));
                let d = if i == 0 { 0.0 } else { 1.0 };
                let w = 0.5 + lcg(&mut s).abs();
                bits_of(m.step(&x, &[Some(y)], d, w).pred[0] / c)
            })
            .collect()
    };
    let one = run(1.0);
    for c in [2f64.powi(-10), 2f64.powi(10)] {
        let scaled = run(c);
        let differ = scaled.iter().zip(&one).filter(|(a, b)| a != b).count();
        assert_eq!(differ, 0, "at scale {c}: {differ} of 600 rows differ");
    }
}

/// A target held at one value has no spread, so under
/// `epsilon_insensitive` it has no band: every row teaches, and an
/// annealed rate takes the fit to the value from zero, at a level of
/// 1,000, with or without decay. A band in the residual's spread was
/// drawn from the first residuals, the whole level, and without decay
/// the fit stopped short of the value by most of it for good
/// (docs/PLAN.md task 202).
#[test]
fn a_target_held_constant_is_learned_under_epsilon_insensitive() {
    for half_life in [f64::INFINITY, 500.0] {
        let mut cf = cfg(1, SgdLoss::EpsilonInsensitive { eps: 0.1 });
        cf.standardize = true;
        cf.decay = Decay::Halflife(half_life);
        cf.learning_rate = 10.0;
        cf.schedule = LearningRate::InvScaling { power: 0.5 };
        cf.min_weight = 2.0;
        let mut m = Sgd::new(cf).unwrap();
        let mut s = 3u64;
        let mut last = f64::NAN;
        for i in 0..20_000 {
            let x = [lcg(&mut s)];
            let d = if i == 0 { 0.0 } else { 1.0 };
            last = m.step(&x, &[Some(1000.0)], d, 1.0).pred[0];
        }
        assert!(
            (last - 1000.0).abs() < 1.0,
            "half-life {half_life}: the last prediction {last}"
        );
    }
}

/// The Huber loss and the squared loss are where they were before the
/// band moved to the target's spread, to the bit: every prediction of a
/// 400-row stream, under a decay, irregular weights, null targets and
/// outliers the Huber cut binds on, digested, and three of them pinned
/// as they print. `huber_delta` stays in units of the residual's
/// spread: outside its cut the gradient is clipped, not zero, so a fit
/// from zero still learns from rows the cut holds (docs/PLAN.md task
/// 202). The digests were BASE's, task 195's build.
///
/// No call into the platform's libm is on the stream's path, so the bits
/// are every platform's: the decay is a literal factor taken once per
/// clock unit (`lam^1` is `lam`, `lam^0` is 1, exactly), and `sqrt` is
/// correctly rounded everywhere. Under `Halflife(80)` with irregular
/// steps each row's factor was an `exp2`, whose last bit glibc and
/// Apple's libm round differently: the digests pinned on macOS failed
/// on Linux (CI, 2026-10-08), and were re-pinned here. Re-pinned again
/// for task 206 (2026-10-08): past the scaler's warm-up the coefficients
/// are held in the caller's units, each step mapped by its own row's
/// scaler, where they were read through the moments as they stood
/// (review round 5, G1). The picks were `[2.5454166364986803,
/// 0.22878767638248743, 4.291195395028382]` (Huber) and
/// `[3.8900811448866905, 0.6435315190771895, 4.854641183478482]`
/// (squared), the digests `0x95a9_bba1_2435_1e24` and
/// `0x5948_2068_29a3_3663`; `tests/reference_paths.py`'s `sgd_ref` under
/// `standardize`, on this stream, gives the new picks to 6e-16. Under
/// the gate that held the scale's learning back, as every fit did then
/// and a state written before schema 64 goes on (task 278, D10).
/// Re-pinned for review C1 (2026-11-20, rule 5's twenty-third
/// exception): the scaler reads each row in the stream's own unit of
/// weight, `Σw²/Σw` of the warm-up's rows, while it warms up and the
/// smaller of that and the row's weight once the fit is held, where it
/// read every row at 1, so this stream's weights from 0.5 to 1.5 moved
/// every prediction from the second row. The picks were
/// `[2.496277502087633, 0.14614547077981044, 4.2455196022404635]` and
/// `[3.8184571260400335, 0.6039531382865846, 4.8134477635370345]`, the
/// digests `0xe5e2_e8b6_26de_7159` and `0x5c6b_9580_7ef9_2eeb`;
/// `sgd_ref` with the new read gives the squared loss's new picks to
/// 2.7e-15 (it has no `min_weight_holds_learning`, which the Huber
/// loss's scale reads here, and is 1.2e-3 off that loss's picks, as it
/// was off the old ones).
#[test]
fn the_huber_and_squared_losses_did_not_move() {
    let run = |loss: SgdLoss| {
        let mut cf = cfg(2, loss);
        cf.standardize = true;
        cf.decay = Decay::Lam(0.9914);
        cf.learning_rate = 0.05;
        cf.min_weight = 3.0;
        cf.min_weight_holds_learning = true;
        let mut m = Sgd::new(cf).unwrap();
        let mut s = 17u64;
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        let mut got = Vec::new();
        for i in 0..400 {
            let x = [lcg(&mut s), 3.0 + lcg(&mut s)];
            let outlier = if i % 41 == 7 { 25.0 } else { 0.0 };
            let y = (i % 17 != 3).then(|| 5.0 + 2.0 * x[0] - x[1] + 0.3 * lcg(&mut s) + outlier);
            let d = if i == 0 { 0.0 } else { 1.0 };
            let w = if i % 23 == 9 {
                0.0
            } else {
                0.5 + lcg(&mut s).abs()
            };
            let p = m.step(&x, &[y], d, w).pred[0];
            for byte in bits_of(p).to_le_bytes() {
                h = (h ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3);
            }
            if matches!(i, 50 | 200 | 399) {
                got.push(p);
            }
        }
        (h, got)
    };
    let huber = run(SgdLoss::Huber { delta: 1.345 });
    let squared = run(SgdLoss::Squared);
    println!("huber {huber:x?}\nsquared {squared:x?}");
    assert_ne!(huber.0, squared.0, "the cut binds on the outliers");
    for ((h, got), (digest, picks)) in [huber, squared].into_iter().zip([
        (
            0x64d4_76ea_964d_a496_u64,
            [2.494501641711098, 0.14615718195207794, 4.245156316660929],
        ),
        (
            0xf556_bcfe_7a79_a474_u64,
            [3.811314295083695, 0.6033274744442707, 4.811773607294475],
        ),
    ]) {
        assert_eq!(h, digest, "picks {got:?}");
        for (g, w) in got.iter().zip(picks) {
            assert_eq!(g.to_bits(), f64::to_bits(w), "{got:?}");
        }
    }
}

/// `eps` is in units of the target's spread: a target whose spread is
/// under 0.1 of its own units -- a return in decimal -- is learned. In
/// the target's units every residual sat inside the tube, so the model
/// was passive for ever and every coefficient stayed 0 (review round 4,
/// CC4).
#[test]
fn a_target_smaller_than_the_tube_in_its_own_units_is_learned() {
    let mut cf = cfg(1, SgdLoss::EpsilonInsensitive { eps: 0.1 });
    cf.standardize = true;
    cf.learning_rate = 1e-4;
    let mut m = Sgd::new(cf).unwrap();
    let mut s = 9u64;
    for i in 0..2000 {
        let x = [lcg(&mut s)];
        let y = 1e-3 * (0.5 + 2.0 * x[0] + 0.3 * lcg(&mut s));
        m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
    }
    let c = &m.coefficients()[0];
    assert!(
        (c[1] - 2e-3).abs() < 1e-3,
        "the slope is 2e-3 in the target's units: {c:?}"
    );
}

/// A logistic label outside {0, 1} is clamped into [0, 1], as `ftrl`
/// clamps it: a label of 5 teaches what a label of 1 does and one of -2
/// what 0 does, to the bit. `p - y` with `p` in (0, 1) and `y = 5` pushed
/// the linear predictor up on every row for ever (review round 4, CC9;
/// docs/PLAN.md task 195, S4).
#[test]
fn a_logistic_label_outside_zero_and_one_is_clamped() {
    let fit = |hi: f64, lo: f64| {
        let mut m = Sgd::new(cfg(1, SgdLoss::Logistic)).unwrap();
        let mut s = 21u64;
        for i in 0..300 {
            let x = [lcg(&mut s)];
            let y = if x[0] + 0.5 * lcg(&mut s) > 0.0 {
                hi
            } else {
                lo
            };
            m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        m.coefficients()[0]
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>()
    };
    assert_eq!(fit(5.0, -2.0), fit(1.0, 0.0));
    assert_eq!(fit(1.0, -0.5), fit(1.0, 0.0));
}

/// Under `strict_binary` a label that is not 0 or 1 teaches nothing and
/// counts nothing toward its target's weight, as in `ftrl`; the option
/// is the logistic loss's alone (docs/PLAN.md task 195, S4).
#[test]
fn strict_binary_skips_a_label_that_is_not_zero_or_one() {
    let mut c = cfg(1, SgdLoss::Logistic);
    c.strict_binary = true;
    c.min_weight = 0.0;
    let mut m = Sgd::new(c.clone()).unwrap();
    m.step(&[0.5], &[Some(1.0)], 0.0, 1.0);
    let (before, w) = (m.coefficients(), m.target_weights()[0]);
    m.step(&[0.5], &[Some(0.7)], 1.0, 1.0);
    assert_eq!(m.coefficients(), before, "a label of 0.7 taught nothing");
    assert_eq!(m.target_weights()[0], w, "and counted nothing");
    m.step(&[0.5], &[Some(0.0)], 1.0, 1.0);
    assert_ne!(m.coefficients(), before, "a label of 0 teaches");
    let mut bad = c;
    bad.loss = SgdLoss::Squared;
    let err = bad.validate().unwrap_err();
    assert!(err.contains("strict_binary"), "{err}");
}

/// Under the Poisson loss a negative count teaches nothing and counts
/// nothing toward its target's weight, as a label `strict_binary`
/// refuses; `-0.0` is the count 0, and another loss learns from a
/// negative target (docs/PLAN.md task 209 (e)). Before, `p − y` drove
/// the prediction to the link's clamp, `e^−30`.
#[test]
fn a_negative_count_teaches_a_poisson_fit_nothing() {
    let mut c = cfg(1, SgdLoss::Poisson);
    c.min_weight = 0.0;
    let mut m = Sgd::new(c.clone()).unwrap();
    m.step(&[0.5], &[Some(2.0)], 0.0, 1.0);
    let (before, w) = (m.coefficients(), m.target_weights()[0]);
    for y in [-1.0, -1e-300, -crate::INPUT_BOUND] {
        m.step(&[0.5], &[Some(y)], 0.0, 1.0);
        assert_eq!(m.coefficients(), before, "a count of {y} taught nothing");
        assert_eq!(m.target_weights()[0], w, "and counted nothing");
    }
    let fit = |y: f64| {
        let mut m = Sgd::new(c.clone()).unwrap();
        m.step(&[0.5], &[Some(2.0)], 0.0, 1.0);
        m.step(&[0.5], &[Some(y)], 1.0, 1.0);
        let bits = |v: &[f64]| v.iter().map(|b| b.to_bits()).collect::<Vec<_>>();
        (bits(&m.coefficients()[0]), m.target_weights()[0])
    };
    assert_eq!(fit(-0.0), fit(0.0), "-0.0 is the count 0");
    assert_ne!(fit(0.0).0, fit(2.0).0, "a count of 0 teaches");
    let mut squared = Sgd::new(cfg(1, SgdLoss::Squared)).unwrap();
    squared.step(&[0.5], &[Some(2.0)], 0.0, 1.0);
    let before = squared.coefficients();
    squared.step(&[0.5], &[Some(-1.0)], 0.0, 1.0);
    assert_ne!(
        squared.coefficients(),
        before,
        "the squared loss takes any target"
    );
}

/// `s²` is the EW mean of the squared out-of-sample residuals of the
/// rows that carried the target, a weight above 0 and a prediction
/// (the module docs): held to that mean from its definition, every
/// qualifying row's weight aged by the clock since it, over irregular
/// steps and weights, null targets and zero weights. The rows before
/// `min_weight` add their residuals from the prediction the fit made,
/// which a twin at `min_weight = 0` reports (task 278, D10); under the
/// gate that held learning back (a state written before schema 64) they
/// add nothing.
#[test]
fn the_residual_variance_is_the_ew_mean_of_the_squared_residuals() {
    for holds in [false, true] {
        let mut c = cfg(2, SgdLoss::Huber { delta: 1.345 });
        c.decay = Decay::Halflife(20.0);
        c.min_weight = 3.0;
        c.min_weight_holds_learning = holds;
        let mut m = Sgd::new(c.clone()).unwrap();
        let mut twin = Sgd::new(SgdCfg {
            min_weight: 0.0,
            ..c.clone()
        })
        .unwrap();
        let mut seen: Vec<(f64, f64)> = Vec::new();
        let mut withheld = 0;
        let mut s = 7u64;
        for i in 0..200 {
            let x = [lcg(&mut s), lcg(&mut s)];
            let y = (i % 9 != 4).then(|| 1.0 + x[0] - 2.0 * x[1] + 0.2 * lcg(&mut s));
            let w = if i % 13 == 6 {
                0.0
            } else {
                0.5 + lcg(&mut s).abs()
            };
            let d = if i == 0 { 0.0 } else { 1.0 + lcg(&mut s).abs() };
            let lam = c.decay.factor(d);
            let out = m.step(&x, &[y], d, w).pred[0];
            let own = twin.step(&x, &[y], d, w).pred[0];
            withheld += usize::from(out.is_nan() && own.is_finite());
            let p = if holds { out } else { own };
            for (_, wi) in seen.iter_mut() {
                *wi *= lam;
            }
            if let Some(yv) = y
                && w > 0.0
                && p.is_finite()
            {
                seen.push(((yv - p) * (yv - p), w));
            }
            let total: f64 = seen.iter().map(|(_, wi)| wi).sum();
            let want = if total > 0.0 {
                seen.iter().map(|(r2, wi)| r2 * wi).sum::<f64>() / total
            } else {
                0.0
            };
            let got = m.sigma2()[0];
            assert!(
                (got - want).abs() <= 1e-12 * want,
                "holds {holds}, row {i}: s² {got} against {want}"
            );
        }
        assert!(seen.len() > 150, "{} rows taught it", seen.len());
        assert!(withheld >= 2, "the gate withheld rows the fit predicted");
    }
}

/// Under `epsilon_insensitive` the tube's unit is its definition:
/// the target's EW variance around its EW mean, `Σ ωᵢ (yᵢ − m)² / Σ ωᵢ`
/// with `m = Σ ωᵢ yᵢ / Σ ωᵢ`, over the rows that carried the target
/// with a weight above 0, the rows before `min_weight` included, `ωᵢ`
/// aged by the model's decay over every row since, rebuilt from the
/// history at every row in two passes, not by the recursion. Irregular
/// steps and weights, null targets and rows of weight 0, at a level of
/// 1,000 (docs/PLAN.md task 202).
#[test]
fn the_target_variance_is_the_ew_variance_of_the_target() {
    let mut c = cfg(2, SgdLoss::EpsilonInsensitive { eps: 0.1 });
    c.decay = Decay::Halflife(20.0);
    c.min_weight = 3.0;
    c.standardize = true;
    let mut m = Sgd::new(c.clone()).unwrap();
    let mut seen: Vec<(f64, f64)> = Vec::new();
    let mut s = 7u64;
    for i in 0..200 {
        let x = [lcg(&mut s), lcg(&mut s)];
        let y = (i % 9 != 4).then(|| 1000.0 + x[0] - 2.0 * x[1] + 0.2 * lcg(&mut s));
        let w = if i % 13 == 6 {
            0.0
        } else {
            0.5 + lcg(&mut s).abs()
        };
        let d = if i == 0 { 0.0 } else { 1.0 + lcg(&mut s).abs() };
        let lam = c.decay.factor(d);
        m.step(&x, &[y], d, w);
        for (_, wi) in seen.iter_mut() {
            *wi *= lam;
        }
        if let Some(v) = y
            && w > 0.0
        {
            seen.push((v, w));
        }
        let total: f64 = seen.iter().map(|(_, wi)| wi).sum();
        let want = if seen.len() < 2 {
            0.0
        } else {
            let mean = seen.iter().map(|(v, wi)| v * wi).sum::<f64>() / total;
            seen.iter()
                .map(|(v, wi)| wi * (v - mean) * (v - mean))
                .sum::<f64>()
                / total
        };
        let got = m.target_variance()[0];
        assert!(
            (got - want).abs() <= 1e-9 * want,
            "row {i}: s_y² {got} against {want}"
        );
        let unit = m.scale(0);
        assert_eq!(unit, (got > 0.0).then(|| got.sqrt()), "row {i}");
    }
    assert!(seen.len() > 150, "{} rows taught it", seen.len());
    assert!(m.sigma2().is_empty(), "no residual scale under this loss");
}

/// A step held in the caller's units is formed whole before anything
/// moves (`Sgd::step_mapped`; docs/PLAN.md task 220): one that would
/// make any coefficient infinite -- a slope's `Δβ_i / s_i`, or the
/// intercept's -- teaches nothing, leaves every coefficient's bits and
/// says so; one that leaves each finite is taken, however large; and
/// under AdaGrad a gradient whose square is not finite teaches nothing
/// and leaves the sums, which a decay never empties, where they were.
/// The row's map is written in by hand: `z`, the means (0 here) and the
/// scales (1); the rate is 1, the clip off, the row at weight 1.
#[test]
fn a_mapped_step_that_would_make_a_coefficient_infinite_teaches_nothing() {
    let make = |schedule: LearningRate| {
        let mut c = cfg(1, SgdLoss::Squared);
        c.standardize = true;
        c.learning_rate = 1.0;
        c.clip_gradient = f64::INFINITY;
        c.schedule = schedule;
        let mut m = Sgd::new(c).unwrap();
        m.ensure_buffers();
        m.mbuf.fill(0.0);
        m.sbuf.fill(1.0);
        m
    };
    let mut m = make(LearningRate::Constant);
    let big = 1.5e308;
    // (what, b, z, d, taken, b after): `Δb = −d z` and `Δb_0 = −d`.
    type Case = (&'static str, [f64; 2], [f64; 2], f64, bool, [f64; 2]);
    let cases: [Case; 4] = [
        (
            "a slope past the range",
            [0.0, big],
            [1.0, 1.0],
            -1e308,
            false,
            [0.0, big],
        ),
        (
            "a slope down past it",
            [0.0, -big],
            [1.0, 1.0],
            1e308,
            false,
            [0.0, -big],
        ),
        (
            "the intercept past it",
            [big, 1.0],
            [1.0, 0.0],
            -1e308,
            false,
            [big, 1.0],
        ),
        (
            "large and finite",
            [1e200, 1e200],
            [1.0, 1.0],
            -1e200,
            true,
            [2e200, 2e200],
        ),
    ];
    for (what, b, z, d, taken, after) in cases {
        m.beta[0] = b.to_vec();
        m.zbuf.copy_from_slice(&z);
        assert_eq!(
            m.step_mapped(0, d, 1.0, Some(1.0), (m.cfg.clip_gradient, 1e-8), None),
            taken,
            "{what}"
        );
        assert_eq!(m.beta[0], after, "{what}");
    }
    let mut m = make(LearningRate::AdaGrad);
    m.g2[0] = vec![4.0, 9.0];
    m.zbuf.copy_from_slice(&[1.0, 1.0]);
    // A gradient of 1e160, whose square is past the range and whose
    // double is not.
    assert!(
        !m.step_mapped(0, 1e160, 1.0, None, (m.cfg.clip_gradient, 1e-8), None),
        "an infinite square"
    );
    assert_eq!((&m.beta[0], &m.g2[0]), (&vec![0.0, 0.0], &vec![4.0, 9.0]));
    assert!(
        m.step_mapped(0, 2.0, 1.0, None, (m.cfg.clip_gradient, 1e-8), None),
        "a usable one"
    );
    assert_eq!(m.g2[0], vec![8.0, 13.0]);
    // Task 252: the feature's gradient usable (`1e160 · 1e-200`) and the
    // intercept's, `d` itself, with a square past the range: the row
    // is not taken, and neither the coefficients nor the sums move.
    let (b, g) = (m.beta[0].clone(), m.g2[0].clone());
    m.zbuf.copy_from_slice(&[1.0, 1e-200]);
    assert!(
        !m.step_mapped(0, 1e160, 1.0, None, (m.cfg.clip_gradient, 1e-8), None),
        "the intercept's square"
    );
    assert_eq!((&m.beta[0], &m.g2[0]), (&b, &g));
}

/// Task 252: a state without a scaler that carries a scaler's warm-up is
/// refused as it is read, naming the mismatch: without a scaler a
/// warm-up counts nothing.
#[test]
fn a_state_without_a_scaler_carrying_a_warm_up_is_refused() {
    let mut plain = cfg(2, SgdLoss::Squared);
    plain.decay = Decay::Halflife(50.0); // JSON has no `inf`
    let mut scaled = plain.clone();
    scaled.standardize = true;
    let mut warm = Sgd::new(scaled).unwrap();
    warm.step(&[0.5, -1.0], &[Some(1.0)], 0.0, 1.0);
    assert_ne!(warm.warmup, Warmup::default(), "the scaler counted the row");
    let mut v = serde_json::to_value(Sgd::new(plain).unwrap()).unwrap();
    assert!(
        serde_json::from_value::<Sgd>(v.clone()).is_ok(),
        "the control"
    );
    v["warmup"] = serde_json::to_value(&warm.warmup).unwrap();
    let e = serde_json::from_value::<Sgd>(v).unwrap_err().to_string();
    assert!(
        e.contains("sgd: its saved warm-up does not match its standardize setting"),
        "{e}"
    );
}

/// Under `huber_scale = "mad"` each target has its own sketch of `|e|`
/// ([`Sgd::mad_sketch`]), and its median is the EW median of that
/// target's out-of-sample residuals within the sketch's accuracy: held
/// on two targets of different spreads, under a decay, with uneven and
/// zero weights, to the exact weighted median of the residuals kept
/// with their decayed weights (the smallest `|e|` whose cumulative
/// weight reaches half the total). The residuals come from the
/// predictions the model reported.
#[test]
fn each_targets_mad_sketch_holds_the_ew_median_of_its_residuals() {
    let mut c = cfg(1, SgdLoss::Huber { delta: 1.345 });
    c.n_targets = 2;
    c.decay = Decay::Lam(0.98);
    c.huber_scale = crate::HuberScale::Mad;
    // Every prediction the scale learns from is reported.
    c.min_weight = 0.0;
    let mut m = Sgd::new(c).unwrap();
    let mut kept: [Vec<(f64, f64)>; 2] = [Vec::new(), Vec::new()];
    let mut s = 41u64;
    let mut checked = 0;
    for i in 0..600 {
        let x = lcg(&mut s);
        let y = [
            1.0 + 2.0 * x + 0.3 * lcg(&mut s),
            -4.0 + 0.5 * x + 3.0 * lcg(&mut s),
        ];
        let w = if i % 7 == 3 {
            0.0
        } else {
            0.5 + lcg(&mut s).abs()
        };
        let d = if i == 0 { 0.0 } else { 1.0 };
        let pred = m.step(&[x], &[Some(y[0]), Some(y[1])], d, w).pred;
        assert_eq!(m.mad_sketch().len(), 2, "row {i}");
        for j in 0..2 {
            kept[j].iter_mut().for_each(|k| k.1 *= 0.98);
            if pred[j].is_finite() && w > 0.0 {
                kept[j].push(((y[j] - pred[j]).abs(), w));
            }
            let Some(got) = m.mad_sketch()[j].get(0) else {
                assert!(kept[j].is_empty(), "row {i}, target {j}");
                continue;
            };
            let total: f64 = kept[j].iter().map(|k| k.1).sum();
            let mut sorted = kept[j].clone();
            sorted.sort_by(|a, b| a.0.total_cmp(&b.0));
            let mut acc = 0.0;
            let want = sorted
                .iter()
                .find(|(_, w)| {
                    acc += w;
                    acc >= 0.5 * total
                })
                .unwrap()
                .0;
            assert!(
                (got - want).abs() <= crate::EW_QUANTILE_ALPHA * want * (1.0 + 1e-9),
                "row {i}, target {j}: {got} against {want}"
            );
            checked += 1;
        }
    }
    assert!(checked > 1100, "{checked}");
}

/// Task 252: under `huber`, a row whose weighted squared residual is
/// past the range (`1e300 · (1e99)²`) leaves the residual scale `s²` and
/// its weight where they were: an infinite scale would make every cut
/// infinite for good.
#[test]
fn a_squared_residual_past_the_range_leaves_the_scale() {
    let mut c = cfg(1, SgdLoss::Huber { delta: 1.345 });
    c.min_weight = 0.0;
    let mut m = Sgd::new(c).unwrap();
    for i in 0..5 {
        let x = 0.1 * f64::from(i);
        m.step(&[x], &[Some(0.5 + x)], f64::from(u8::from(i > 0)), 1.0);
    }
    let (s, w) = (m.sig2[0], m.wsig[0]);
    assert!(s.is_finite() && s > 0.0 && w > 0.0, "{s} {w}");
    let out = m.step(&[0.1], &[Some(1e99)], 1.0, 1e300);
    assert!(out.pred[0].is_finite(), "the row was scored");
    assert_eq!(
        (m.sig2[0].to_bits(), m.wsig[0].to_bits()),
        (s.to_bits(), w.to_bits())
    );
}
