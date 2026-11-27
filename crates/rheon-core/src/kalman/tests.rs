use super::*;
use crate::oracle::CanonBits;

fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

/// `take_the_row` and `matvec` (task 420) are the loops they replaced, to
/// the bit: the upper triangle updated and mirrored, and each row's
/// products summed from column 0, at every width up to 40 (the blocks of
/// four rows and their remainders), on values of mixed scale with zeros
/// and signs.
#[test]
fn the_covariance_passes_are_the_serial_forms_to_the_bit() {
    let mut st = 420u64;
    for k in 1..=40usize {
        for round in 0..20 {
            let mut vals = |n: usize| -> Vec<f64> {
                (0..n)
                    .map(|_| {
                        let u = lcg(&mut st);
                        match (u * 1e3) as i64 % 7 {
                            0 => 0.0,
                            1 => u * 1e-30,
                            2 => u * 1e30,
                            _ => u,
                        }
                    })
                    .collect()
            };
            // A bit-symmetric `P`, as the filter keeps it, and one that is not.
            let mut p = vals(k * k);
            if round % 2 == 0 {
                for i in 0..k {
                    for j in 0..i {
                        p[i * k + j] = p[j * k + i];
                    }
                }
            }
            let (gain, pz, z) = (vals(k), vals(k), vals(k));

            let mut old = p.clone();
            for i in 0..k {
                for jj in i..k {
                    let v = gain[i] * pz[jj];
                    old[i * k + jj] -= v;
                    if jj != i {
                        old[jj * k + i] -= v;
                    }
                }
            }
            let mut new = p.clone();
            take_the_row(&mut new, &gain, &pz);
            let bits = |v: &[f64]| v.iter().map(|x| x.canon_bits()).collect::<Vec<_>>();
            // Mirrored writes make the old form's lower triangle `P_ji - v`,
            // which is the new form's on any `P`, symmetric or not.
            assert_eq!(bits(&old), bits(&new), "take_the_row, k = {k}");

            let mut old_pz = vec![0.0; k];
            for i in 0..k {
                let mut acc = 0.0;
                for jj in 0..k {
                    acc += p[i * k + jj] * z[jj];
                }
                old_pz[i] = acc;
            }
            let mut new_pz = vec![f64::NAN; k];
            matvec(&p, &z, &mut new_pz);
            assert_eq!(bits(&old_pz), bits(&new_pz), "matvec, k = {k}");
        }
    }
}

/// The noise basis's median, sorted on the stack (task 250), is the
/// one the vector it replaced gave, to the bit: every order of three
/// rows, with ties in the squares and weights that put the half on a
/// row's edge, and none before the third row.
#[test]
fn the_noise_median_on_the_stack_is_the_vectors() {
    let vector_form = |b: &NoiseBasis| -> Option<f64> {
        if b.e2.len() < PRIOR_ROWS {
            return None;
        }
        let mut rows: Vec<(f64, f64)> = b.e2.iter().copied().zip(b.w.iter().copied()).collect();
        rows.sort_by(|a, b| a.0.total_cmp(&b.0));
        let half = 0.5 * b.w.iter().sum::<f64>();
        let mut seen = 0.0;
        let median = rows
            .iter()
            .find(|(_, w)| {
                seen += w;
                seen >= half
            })
            .map_or(rows[rows.len() - 1].0, |(v, _)| *v);
        let v = median / CHI2_1_MEDIAN;
        (v > 0.0 && v.is_finite()).then_some(v)
    };
    let squares = [
        [1.0, 1.0, 2.0],
        [1.0, 2.0, 2.0],
        [3.0, 3.0, 3.0],
        [0.5, 2.0, 7.0],
        [1e-300, 1e300, 1.0],
    ];
    let weights = [
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 2.0],
        [2.0, 1.0, 1.0],
        [0.25, 0.5, 0.25],
        [1e-9, 1.0, 1e9],
    ];
    let orders = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ];
    let mut read = 0;
    for e2 in squares {
        for w in weights {
            for o in orders {
                let mut b = NoiseBasis::default();
                for (n, &i) in o.iter().enumerate() {
                    assert_eq!(b.noise(), None, "{n} rows");
                    b.add(e2[i] * 2.0, 2.0, w[i]);
                }
                let got = b.noise().expect("three rows");
                assert_eq!(
                    got.to_bits(),
                    vector_form(&b).expect("three rows").to_bits(),
                    "{e2:?} {w:?} {o:?}"
                );
                read += 1;
            }
        }
    }
    assert_eq!(read, 5 * 5 * 6);
}

fn cfg(k: usize, m: usize, hl: Vec<f64>) -> KalmanCfg {
    KalmanCfg {
        horizons: Vec::new(),
        min_weight_holds_learning: false,
        noise_units: false,
        n_features: k,
        n_targets: m,
        fit_intercept: true,
        decay: Decay::Halflife(200.0),
        half_life: hl,
        q: None,
        obs_var: None,
        p0: 1.0,
        share_p: false,
        min_weight: 10.0,
        revert_half_life: vec![f64::INFINITY],
        standardize: true,
    }
}

/// Feed a deterministic stream, returning the fitted filter.
fn fit(cfg: KalmanCfg, n: usize, seed: u64) -> Kalman {
    let m = cfg.n_targets;
    let mut model = Kalman::new(cfg).unwrap();
    let mut s = seed;
    for i in 0..n {
        let x = [lcg(&mut s), 0.5 + lcg(&mut s)];
        let ys: Vec<Option<f64>> = (0..m)
            .map(|j| Some((j as f64 + 1.0) * (2.0 * x[0] - x[1]) + 0.1 * lcg(&mut s)))
            .collect();
        model.step(&x, &ys, if i == 0 { 0.0 } else { 1.0 }, 1.0);
    }
    model
}

#[test]
fn cfg_validation_rejects_each_bad_field() {
    let bad = |f: &dyn Fn(&mut KalmanCfg), want: &str| {
        let mut c = cfg(2, 1, vec![100.0]);
        f(&mut c);
        match c.validate() {
            Err(e) => assert!(e.contains(want), "wanted {want:?}, got {e:?}"),
            Ok(()) => panic!("expected rejection mentioning {want:?}"),
        }
    };
    let good = |f: &dyn Fn(&mut KalmanCfg)| {
        let mut c = cfg(2, 1, vec![100.0]);
        f(&mut c);
        c.validate().expect("should be accepted");
    };

    bad(
        &|c| c.n_features = 0,
        "needs at least one feature and one target",
    );
    bad(
        &|c| c.n_targets = 0,
        "needs at least one feature and one target",
    );

    // `q` is the process noise per slot: one entry per coefficient,
    // including the intercept, and zero means "pinned".
    bad(&|c| c.q = Some(vec![0.0; 2]), "each of the 3 coefficients");
    bad(
        &|c| c.q = Some(vec![0.0, 0.0, -1e-9]),
        "q must be finite and at least 0",
    );
    good(&|c| c.q = Some(vec![0.0; 3]));

    // Without `q`, the half-lives are broadcast: one value, or one per slot.
    bad(
        &|c| c.half_life = vec![1.0, 2.0],
        "or one for each of the 3 coefficients",
    );
    bad(&|c| c.half_life = vec![0.0], "must be above 0");
    bad(&|c| c.half_life = vec![-1.0], "must be above 0");
    good(&|c| c.half_life = vec![f64::INFINITY]);
    good(&|c| c.half_life = vec![1.0, 2.0, 3.0]);

    // p0 is the prior variance and obs_var the measurement noise; both
    // divide, so neither may be zero. obs_var may be absent (inferred).
    bad(&|c| c.p0 = 0.0, "p0 must be finite and above 0");
    bad(&|c| c.p0 = -1.0, "p0 must be finite and above 0");
    bad(
        &|c| c.obs_var = Some(0.0),
        "obs_var must be finite and above 0",
    );
    bad(
        &|c| c.obs_var = Some(-1.0),
        "obs_var must be finite and above 0",
    );
    good(&|c| c.obs_var = None);
    good(&|c| c.obs_var = Some(1e-9));
    // NaN passed every `<= 0.0` / `< 0.0` test here. A NaN `obs_var` was
    // the silent one: `s_inn` is NaN on every row, the update's guard is
    // never met, and the filter predicts its prior for the life of the
    // stream with no error and no counted failure (review 2026-09-18, B4).
    bad(
        &|c| c.obs_var = Some(f64::NAN),
        "obs_var must be finite and above 0",
    );
    bad(&|c| c.p0 = f64::NAN, "p0 must be finite and above 0");
    bad(&|c| c.half_life = vec![f64::NAN], "must be above 0");
    bad(
        &|c| c.q = Some(vec![0.0, f64::NAN, 0.0]),
        "q must be finite and at least 0",
    );

    cfg(2, 1, vec![100.0]).validate().unwrap();

    // Every problem at once, each a line of its own.
    let mut two = cfg(2, 1, vec![100.0]);
    two.p0 = 0.0;
    two.obs_var = Some(0.0);
    let e = two.validate().unwrap_err();
    let lines: Vec<&str> = e.lines().collect();
    assert_eq!(lines.len(), 2, "{e}");
    assert!(
        lines[0].starts_with("kalman: p0 must be finite and above 0"),
        "{e}"
    );
    assert!(
        lines[1].starts_with("kalman: obs_var must be finite and above 0"),
        "{e}"
    );
}

#[test]
fn standardize_defaults_to_on_when_a_state_file_omits_it() {
    // `#[serde(default = "default_true")]`: a state written before the
    // field existed must load with standardization on, which is the
    // behaviour that state was produced under. Defaulting to `false`
    // instead would silently change every restored model's numbers.
    let json = r#"{
        "n_features": 2, "n_targets": 1, "fit_intercept": true,
        "decay": {"Halflife": 200.0}, "half_life": [100.0], "q": null,
        "obs_var": null, "p0": 1.0, "share_p": false, "min_weight": 10.0
    }"#;
    let cfg: KalmanCfg = serde_json::from_str(json).expect("should load without the field");
    assert!(cfg.standardize, "the omitted field must default to true");
}

#[test]
fn coefficients_are_reported_in_the_callers_units() {
    // The filter works on standardized, centered features; `coefficients`
    // has to undo both -- divide by the scale, then unshift the intercept
    // by the feature means -- or the numbers a caller reads are not the
    // ones their data is in.
    // A coefficient half-life rather than a pinned one, so the filter keeps
    // re-learning as the standardization stats settle. With `q = 0` and a
    // near-zero observation noise it would instead converge in a handful of
    // rows, locking its betas into the standardized space of the first few
    // rows while `coefficients` unscales with the current stats -- which is
    // why the Bayesian-regression correspondence test turns standardization
    // off rather than working around it.
    let mut c = cfg(2, 1, vec![500.0]);
    c.min_weight = 3.0;
    let mut m = Kalman::new(c).unwrap();
    let mut s = 149u64;
    // Features on very different scales and far from zero, so a missing
    // unscale or a missing unshift is unmistakable.
    for i in 0..20_000 {
        let x = [500.0 + 10.0 * lcg(&mut s), 0.01 * lcg(&mut s)];
        let y = 12.0 + 0.25 * x[0] - 800.0 * x[1];
        m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
    }
    let b = &m.coefficients()[0];
    assert!((b[1] - 0.25).abs() < 0.02, "slope 0: {}", b[1]);
    assert!((b[2] + 800.0).abs() < 10.0, "slope 1: {}", b[2]);
    // The intercept carries the accumulated slope error times mean(x0), so
    // it is the loosest of the three -- but it must be near 12, not near
    // the ~137 that dropping the unshift would give.
    assert!((b[0] - 12.0).abs() < 12.0, "intercept: {}", b[0]);
}

/// `pred_var` read the regressor of the last row stepped, a scratch a save
/// does not keep, so a loaded filter answered for another regressor than
/// the saved one would (review 2026-09-12, V7).
#[test]
fn pred_var_after_a_load_is_the_saved_filters() {
    let mut c = cfg(2, 1, vec![100.0]);
    c.obs_var = Some(0.25);
    c.min_weight = 3.0;
    let m = fit(c, 80, 61);
    let x = [0.3, -0.2];
    let before = m.pred_var(&x);
    // Through the bytes a save writes: `state()` alone is an in-memory
    // clone, which keeps the scratch a file does not.
    let bytes = rmp_serde::to_vec(&m.state()).unwrap();
    let state: State = rmp_serde::from_slice(&bytes).unwrap();
    let back = Kalman::restore(&state).unwrap();
    assert_eq!(back.pred_var(&x), before);
}

#[test]
fn pred_var_is_the_quadratic_form_plus_observation_noise() {
    // No layer above this crate reads `pred_var`, so its arithmetic is
    // tested here. It is z' P z + R, and both halves are checked: the
    // quadratic form against a longhand loop over the stored covariance,
    // and R against the configured or inferred observation noise.
    let mut c = cfg(2, 1, vec![100.0]);
    c.obs_var = Some(0.25);
    c.min_weight = 3.0;
    let m = fit(c, 80, 61);
    let k = m.cfg.k_total();
    let x = [0.4, -0.7];
    // `P` is held in the anchor's coordinates, and so is the row read
    // against it (the module doc).
    let z = m.anchored(&x);

    let mut want = 0.0;
    for i in 0..k {
        for j in 0..k {
            want += z[i] * m.p[0][i * k + j] * z[j];
        }
    }
    want += 0.25;
    let got = m.pred_var(&x)[0];
    assert!((got - want).abs() < 1e-12, "{got} vs {want}");
    assert!(got > 0.25, "must exceed the observation noise: {got}");

    // Without a configured obs_var it falls back to the tracked residual
    // variance of that target.
    let mut c = cfg(2, 1, vec![100.0]);
    c.min_weight = 3.0;
    let m = fit(c, 80, 61);
    let got = m.pred_var(&x)[0];
    assert!(got > m.sigma2()[0], "{got} vs {}", m.sigma2()[0]);
}

#[test]
fn share_p_shares_one_covariance_and_averages_the_noise() {
    // With `share_p` the process step runs once rather than once per
    // target, one covariance is kept, and the inferred observation noise
    // is the mean across targets rather than each target's own.
    let mut shared = cfg(2, 2, vec![100.0]);
    shared.share_p = true;
    shared.min_weight = 3.0;
    let ms = fit(shared, 200, 71);
    assert_eq!(ms.p.len(), 1, "one covariance for all targets");
    // Both targets read the same P, which the mean noise drives; each
    // target's own covariance is P times its noise over the mean, so its
    // pred_var is `σ²_j (zᵀ P z / σ̄² + 1)` (the module doc, task 211):
    // the same over `σ²_j` for both, and apart by their noises, which
    // here differ by construction (target 1 is 2x target 0).
    let pv = ms.pred_var(&[0.3, -0.2]);
    let s2 = ms.sigma2();
    assert!(s2[1] > 2.0 * s2[0], "targets differ: {s2:?}");
    assert!(
        (pv[0] / s2[0] - pv[1] / s2[1]).abs() < 1e-12,
        "{pv:?} {s2:?}"
    );
    let mean = s2.iter().sum::<f64>() / 2.0;
    let z = ms.anchored(&[0.3, -0.2]);
    let k = 3;
    let quad: f64 = (0..k)
        .map(|a| {
            (0..k)
                .map(|b| z[a] * ms.p[0][a * k + b] * z[b])
                .sum::<f64>()
        })
        .sum();
    for j in 0..2 {
        let want = s2[j] * (quad / mean + 1.0);
        assert!(
            (pv[j] - want).abs() < 1e-12 * want,
            "target {j}: {} against {want}",
            pv[j]
        );
    }

    let mut separate = cfg(2, 2, vec![100.0]);
    separate.min_weight = 3.0;
    let msep = fit(separate, 200, 71);
    assert_eq!(msep.p.len(), 2, "one covariance per target");
    let pv2 = msep.pred_var(&[0.3, -0.2]);
    assert!(
        (pv2[0] - pv2[1]).abs() > 1e-6,
        "unshared targets should differ: {pv2:?}"
    );
}

/// Task 204: `P`'s recursion never reads `y`, so under `share_p` a
/// target beside an exact copy of itself predicts as it would alone, and
/// as it would without `share_p`, to the bit (the mean of two equal
/// noises is exact). Updating the shared `P` once per target, as it did,
/// moved such a target's predictions by up to 0.92 on a spread of 1.1.
/// Decay, irregular steps, weights other than 1, zero weights and null
/// targets.
#[test]
fn share_p_a_target_beside_its_own_copy_predicts_as_it_would_alone() {
    let run = |m: usize, share: bool| -> Vec<u64> {
        let mut c = cfg(2, m, vec![50.0]);
        c.share_p = share;
        c.min_weight = 2.0;
        c.decay = Decay::Halflife(40.0);
        let mut model = Kalman::new(c).unwrap();
        let mut s = 211u64;
        let mut out = Vec::new();
        for i in 0..400 {
            let x = [lcg(&mut s), 0.5 + lcg(&mut s)];
            let e = lcg(&mut s);
            let y = (i % 7 != 3).then(|| 1.0 - 2.0 * x[0] + x[1] + 0.3 * e);
            let w = if i % 11 == 5 {
                0.0
            } else {
                0.5 + lcg(&mut s).abs()
            };
            let d = if i == 0 { 0.0 } else { 0.5 + (i % 3) as f64 };
            let pred = model.step(&x, &vec![y; m], d, w).pred;
            for p in &pred {
                assert_eq!(p.to_bits(), pred[0].to_bits(), "row {i}: the copies part");
            }
            out.push(pred[0].to_bits());
        }
        out
    };
    let alone = run(1, false);
    let scored = alone
        .iter()
        .filter(|b| f64::from_bits(**b).is_finite())
        .count();
    assert!(scored > 300, "only {scored} rows predicted");
    assert_eq!(run(1, true), alone, "share_p, alone");
    assert_eq!(run(2, true), alone, "share_p, beside a copy");
}

/// Task 204: under `share_p` the order of `targets` moves no bit of any
/// target's prediction. Three targets of different noise, each missing
/// on rows of its own, so the shared noise sums three terms, in ascending
/// order. Before, swapping two targets moved predictions by up to 0.36.
#[test]
fn share_p_the_order_of_the_targets_moves_no_bit() {
    let run = |order: [usize; 3]| -> Vec<[u64; 3]> {
        let mut c = cfg(2, 3, vec![50.0]);
        c.share_p = true;
        c.min_weight = 2.0;
        let mut model = Kalman::new(c).unwrap();
        let mut s = 223u64;
        let mut out = Vec::new();
        for i in 0..400 {
            let x = [lcg(&mut s), 0.5 + lcg(&mut s)];
            let e = [lcg(&mut s), lcg(&mut s), lcg(&mut s)];
            let ys = [
                (i % 5 != 1).then(|| 2.0 * x[0] - x[1] + 0.1 * e[0]),
                (i % 7 != 2).then(|| -x[0] + 3.0 * x[1] + e[1]),
                (i % 3 != 0).then(|| 0.5 + x[0] + 3.0 * e[2]),
            ];
            let w = 0.5 + lcg(&mut s).abs();
            let given: Vec<Option<f64>> = order.iter().map(|&t| ys[t]).collect();
            let pred = model
                .step(&x, &given, if i == 0 { 0.0 } else { 1.0 }, w)
                .pred;
            let mut by_target = [0u64; 3];
            for (slot, &t) in order.iter().enumerate() {
                by_target[t] = pred[slot].to_bits();
            }
            out.push(by_target);
        }
        out
    };
    let base = run([0, 1, 2]);
    let scored = base
        .iter()
        .filter(|r| f64::from_bits(r[2]).is_finite())
        .count();
    assert!(scored > 300, "only {scored} rows predicted");
    for order in [[2, 0, 1], [1, 2, 0], [0, 2, 1]] {
        assert_eq!(run(order), base, "order {order:?}");
    }
}

#[test]
fn a_null_target_decays_its_weights_and_leaves_the_filter_alone() {
    let mut c = cfg(2, 1, vec![100.0]);
    c.decay = Decay::Halflife(10.0);
    c.min_weight = 3.0;
    let mut m = Kalman::new(c).unwrap();
    let mut s = 73u64;
    for i in 0..60 {
        let x = [lcg(&mut s), lcg(&mut s)];
        m.step(
            &x,
            &[Some(x[0] - x[1])],
            if i == 0 { 0.0 } else { 1.0 },
            1.0,
        );
    }
    // `b` follows the moments to their new coordinates (the module
    // doc), so a row with no target leaves the coefficients in the
    // caller's units where they were, to rounding.
    let coef = m.coefficients();
    let (wj, wsig, sig2) = (m.wj[0], m.wsig[0], m.sig2[0]);

    let lam = 0.5f64.powf(4.0 / 10.0);
    m.step(&[0.3, -0.2], &[None], 4.0, 1.0);
    for (a, b) in m.coefficients()[0].iter().zip(&coef[0]) {
        assert!(
            (a - b).abs() <= 1e-14 * (1.0 + b.abs()),
            "no target, no correction: {a} against {b}"
        );
    }
    assert_eq!(m.sig2[0], sig2);
    assert!((m.wj[0] - wj * lam).abs() < 1e-12);
    assert!((m.wsig[0] - wsig * lam).abs() < 1e-12);
}

#[test]
fn a_zero_weight_row_does_not_correct_the_filter() {
    let mut c = cfg(2, 1, vec![100.0]);
    c.min_weight = 3.0;
    let mut m = Kalman::new(c).unwrap();
    let mut s = 79u64;
    for i in 0..60 {
        let x = [lcg(&mut s), lcg(&mut s)];
        m.step(
            &x,
            &[Some(x[0] - x[1])],
            if i == 0 { 0.0 } else { 1.0 },
            1.0,
        );
    }
    let beta = m.beta[0].clone();
    m.step(&[0.3, -0.2], &[Some(-500.0)], 1.0, 0.0);
    assert_eq!(m.beta[0], beta, "weight 0 must not move the coefficients");
}

/// `a_null_target_decays_its_weights_and_leaves_the_filter_alone` with
/// the target present at weight zero. To the filter the two rows are the
/// same -- a prediction step, no update -- and so they are to the
/// weights: `wj` and `wsig` decay by the row's `lam` in both. The zero
/// weight skipped the decay, so `σ²`, which sets `R` and `Q`, forgot less
/// across such a row than across a null (review 2026-09-12, S9).
#[test]
fn a_zero_weight_row_decays_its_weights_as_a_null_does() {
    let mut c = cfg(2, 1, vec![100.0]);
    c.decay = Decay::Halflife(10.0);
    c.min_weight = 3.0;
    let mut m = Kalman::new(c).unwrap();
    let mut s = 73u64;
    for i in 0..60 {
        let x = [lcg(&mut s), lcg(&mut s)];
        m.step(
            &x,
            &[Some(x[0] - x[1])],
            if i == 0 { 0.0 } else { 1.0 },
            1.0,
        );
    }
    let beta = m.beta[0].clone();
    let (wj, wsig, sig2) = (m.wj[0], m.wsig[0], m.sig2[0]);

    let lam = 0.5f64.powf(4.0 / 10.0);
    m.step(&[0.3, -0.2], &[Some(-500.0)], 4.0, 0.0);
    assert_eq!(m.beta[0], beta, "weight 0, no correction");
    assert_eq!(m.sig2[0], sig2);
    assert!(
        (m.wj[0] - wj * lam).abs() < 1e-12,
        "{} vs {}",
        m.wj[0],
        wj * lam
    );
    assert!(
        (m.wsig[0] - wsig * lam).abs() < 1e-12,
        "{} vs {}",
        m.wsig[0],
        wsig * lam
    );
}

/// The prediction it learns from is the filter's own, made on every row
/// whatever `min_weight` says -- a twin at `min_weight` 0 reports it --
/// and, with the gate holding learning back (a state written before
/// schema 64; task 278, D10), the one the gate lets out.
#[test]
fn residual_variance_is_the_ew_mean_of_squared_out_of_sample_errors() {
    for holds in [false, true] {
        let mut c = cfg(1, 1, vec![f64::INFINITY]);
        c.decay = Decay::Halflife(25.0);
        c.min_weight = 3.0;
        c.standardize = false;
        c.obs_var = Some(0.5);
        c.min_weight_holds_learning = holds;
        let mut m = Kalman::new(c.clone()).unwrap();
        let mut twin = Kalman::new(KalmanCfg {
            min_weight: 0.0,
            ..c
        })
        .unwrap();

        let (mut want, mut wsig, mut withheld) = (0.0, 0.0, 0);
        let mut s = 83u64;
        for i in 0..120 {
            let x = [lcg(&mut s)];
            let y = 2.0 * x[0] + 0.3 * lcg(&mut s);
            let d = if i == 0 { 0.0 } else { 1.0 };
            let lam = 0.5f64.powf(d / 25.0);
            let shown = m.step(&x, &[Some(y)], d, 1.0).pred[0];
            let own = twin.step(&x, &[Some(y)], d, 1.0).pred[0];
            if own.is_finite() && !shown.is_finite() {
                withheld += 1;
            }
            let p = if holds { shown } else { own };
            if p.is_finite() {
                let resid = y - p;
                let ws_new = lam * wsig + 1.0;
                want = (lam * wsig * want + resid * resid) / ws_new;
                wsig = ws_new;
            }
            assert!(
                (m.sigma2()[0] - want).abs() < 1e-12,
                "holds {holds}, row {i}: {} vs {want}",
                m.sigma2()[0]
            );
        }
        assert!(wsig > 30.0 && want > 0.0);
        assert!(withheld >= 1, "the gate withheld a prediction");
    }
}

/// The same mean across rows with no prediction: after a clock gap
/// takes `n_eff` under `min_weight`, the next rows have a target and
/// no prediction. They add nothing, and age `σ²`'s weight as every row
/// does; they aged nothing, so `σ²` -- which sets `R` and `Q` -- forgot
/// less across them than the clock says (N6, found beside review
/// 2026-09-12 S13). Rows `min_weight` withholds are no such rows unless
/// the gate holds learning back, as in a state written before schema
/// 64 (task 278, D10): the filter predicts on them for its own use.
#[test]
fn the_residual_variance_ages_across_rows_with_no_prediction() {
    let hl = 25.0;
    let mut c = cfg(1, 1, vec![f64::INFINITY]);
    c.min_weight_holds_learning = true;
    c.decay = Decay::Halflife(hl);
    c.min_weight = 3.0;
    c.standardize = false;
    c.obs_var = Some(0.5);
    let mut m = Kalman::new(c).unwrap();
    let (mut want, mut wsig, mut unpredicted) = (0.0f64, 0.0f64, 0);
    let mut s = 83u64;
    for i in 0..160 {
        let x = [lcg(&mut s)];
        let y = 2.0 * x[0] + 0.3 * lcg(&mut s);
        let d = match i {
            0 => 0.0,
            80 => 400.0,
            _ => 1.0,
        };
        let p = m.step(&x, &[Some(y)], d, 1.0).pred[0];
        wsig *= 0.5f64.powf(d / hl);
        if p.is_finite() {
            let r = y - p;
            let ws_new = wsig + 1.0;
            want = (wsig * want + r * r) / ws_new;
            wsig = ws_new;
        } else if wsig > 0.0 {
            unpredicted += 1;
        }
        let got = m.sigma2()[0];
        assert!(
            (got - want).abs() <= 1e-12 * want,
            "row {i}: sigma2 {got}, the EW mean of the squared errors {want}"
        );
    }
    assert!(
        unpredicted >= 2,
        "the gap must leave rows with no prediction"
    );
}

/// With `standardize = false`, `q = 0` and a fixed `obs_var`, the filter is
/// exactly a Bayesian linear regression: coefficients converge to the ridge
/// solution with penalty `obs_var / P_0 = 1 / p0`, the prior being
/// `P_0 = p0 obs_var I` (CC4).
#[test]
fn unstandardized_with_no_process_noise_is_bayesian_regression() {
    let (p0, obs_var) = (10.0, 0.25);
    let mut m = Kalman::new(KalmanCfg {
        horizons: Vec::new(),
        min_weight_holds_learning: false,
        noise_units: false,
        n_features: 2,
        n_targets: 1,
        fit_intercept: false,
        decay: Decay::Halflife(f64::INFINITY),
        half_life: vec![f64::INFINITY],
        q: Some(vec![0.0, 0.0]),
        obs_var: Some(obs_var),
        p0,
        share_p: false,
        min_weight: 0.0,
        revert_half_life: vec![f64::INFINITY],
        standardize: false,
    })
    .unwrap();
    // Accumulate the normal equations alongside, then compare with the
    // closed-form ridge solution (1 / p0 is the implied penalty).
    //
    // Task 274: within a bound on both sides' rounding, where it was
    // `1e-9`. The filter's: its covariance form `P' = P − g(Pz)ᵀ`, `g =
    // Pz/(z'Pz + R)`, `β' = β + g·e`, carries an error `ΔP` to `(I −
    // gzᵀ)ΔP(I − zgᵀ)` and `Δβ` to `(I − gzᵀ)Δβ` (first order, exactly,
    // `ΔP`'s reaching `β` as `(I − gzᵀ)ΔP z/s` times the innovation), and
    // `I − gzᵀ = A'⁻¹A` for the information `A = P⁻¹ = I/(p₀R) + Σzzᵀ/R`,
    // so an error made at row `t` reaches row `N` through `A_N⁻¹A_t`.
    // Each row's own: `Pz`, `z'Pz` and the gain (`γ₂` of each sum and
    // `u` of each quotient), `u` of each product and difference of the
    // two updates, and `ΔP`'s reach into `β`. The magnitudes are the
    // exact recursion's, from the information sums. The definition's:
    // its sums, `u` of each term and of each partial sum (Higham §3.3),
    // through the ridge system's forward error, and its LU's own.
    // Measured on aarch64 (macOS): at most 0.021 of the bound.
    use crate::oracle::{self, U, abs_times, gamma};
    let mut s = 55u64;
    let (mut xtx, mut xty) = ([[0.0f64; 2]; 2], [0.0f64; 2]);
    let (mut exx, mut exy) = ([[0.0f64; 2]; 2], [0.0f64; 2]);
    let r = obs_var;
    let info = |xtx: &[[f64; 2]; 2]| -> Vec<f64> {
        let a0 = 1.0 / (p0 * r);
        vec![
            a0 + xtx[0][0] / r,
            xtx[0][1] / r,
            xtx[1][0] / r,
            a0 + xtx[1][1] / r,
        ]
    };
    // `A_t` after each row, each row's error in `β` and in `P`.
    let mut infos: Vec<Vec<f64>> = vec![info(&xtx)];
    let (mut loc_b, mut loc_p): (Vec<[f64; 2]>, Vec<Vec<f64>>) = (Vec::new(), Vec::new());
    // `A_to⁻¹·A_from`, each inverse taken once.
    let mut inverses: Vec<Vec<f64>> = vec![oracle::inverse(&infos[0])];
    let reach = |inverses: &[Vec<f64>], infos: &[Vec<f64>], to: usize, from: usize| -> Vec<f64> {
        crate::sandwich::rounding::mul(&inverses[to], &infos[from], 2)
    };
    for i in 0..400 {
        let x = [lcg(&mut s), lcg(&mut s)];
        let y = 1.25 * x[0] - 0.5 * x[1];
        m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        // The filter's row, from the exact recursion's magnitudes.
        let t = infos.len() - 1;
        let pm = inverses[t].clone();
        let beta: Vec<f64> = (0..2)
            .map(|a| (0..2).map(|b| pm[a * 2 + b] * xty[b] / r).sum())
            .collect();
        let e = y - x[0] * beta[0] - x[1] * beta[1];
        let pz = [pm[0] * x[0] + pm[1] * x[1], pm[2] * x[0] + pm[3] * x[1]];
        let sn = x[0] * pz[0] + x[1] * pz[1] + r;
        let g = [pz[0] / sn, pz[1] / sn];
        // `ΔP` as the row finds it: every earlier row's, carried.
        let mut dp = vec![0.0; 4];
        for (u0, lp) in loc_p.iter().enumerate() {
            let mm = reach(&inverses, &infos, t, u0 + 1);
            let mt: Vec<f64> = (0..4).map(|ij| mm[(ij % 2) * 2 + ij / 2]).collect();
            let c = crate::sandwich::rounding::abs_mul(
                &crate::sandwich::rounding::abs_mul(&mm, lp, 2),
                &mt,
                2,
            );
            dp.iter_mut().zip(c).for_each(|(a, b)| *a += b);
        }
        let az = [x[0].abs(), x[1].abs()];
        let abs_p: Vec<f64> = pm.iter().map(|v| v.abs()).collect();
        let dpz = abs_times(&abs_p, &az)
            .iter()
            .map(|v| gamma(2) * v)
            .collect::<Vec<_>>();
        let dzpz = gamma(2) * (az[0] * pz[0].abs() + az[1] * pz[1].abs())
            + az[0] * dpz[0]
            + az[1] * dpz[1];
        let dg: Vec<f64> = (0..2)
            .map(|a| (dpz[a] + g[a].abs() * (dzpz + U * sn)) / sn + U * g[a].abs())
            .collect();
        let igz = [
            1.0 - g[0] * x[0],
            -g[0] * x[1],
            -g[1] * x[0],
            1.0 - g[1] * x[1],
        ];
        let reach_p = abs_times(&igz, &abs_times(&dp, &az));
        let bn = [beta[0] + g[0] * e, beta[1] + g[1] * e];
        loc_b.push(std::array::from_fn(|a| {
            reach_p[a] / sn * e.abs() + dg[a] * e.abs() + U * (g[a] * e).abs() + U * bn[a].abs()
        }));
        loc_p.push(
            (0..4)
                .map(|ij| {
                    let (a, b) = (ij / 2, ij % 2);
                    let pn = pm[ij] - g[a] * pz[b];
                    U * ((g[a] * pz[b]).abs() + pn.abs())
                        + dg[a] * pz[b].abs()
                        + g[a].abs() * dpz[b]
                })
                .collect(),
        );
        for a in 0..2 {
            let t = x[a] * y;
            xty[a] += t;
            exy[a] += U * t.abs() + U * xty[a].abs();
            for b in 0..2 {
                let t = x[a] * x[b];
                xtx[a][b] += t;
                exx[a][b] += U * t.abs() + U * xtx[a][b].abs();
            }
        }
        infos.push(info(&xtx));
        inverses.push(oracle::inverse(infos.last().unwrap()));
    }
    // The ridge normal equations solved by faer's LU (`crate::oracle`),
    // which the filter shares no arithmetic with, where Cramer's rule by
    // hand stood (review 2026-10-06, CF8).
    let lam = 1.0 / p0;
    let gram = [xtx[0][0] + lam, xtx[0][1], xtx[1][0], xtx[1][1] + lam];
    let want = crate::oracle::solve(&gram, &xty);
    let got = &m.coefficients()[0];
    let n = infos.len() - 1;
    let mut lib = [0.0f64; 2];
    for (t, lb) in loc_b.iter().enumerate() {
        let mm = reach(&inverses, &infos, n, t + 1);
        let c = abs_times(&mm, lb);
        lib[0] += c[0];
        lib[1] += c[1];
    }
    let eg = [
        exx[0][0] + U * gram[0].abs(),
        exx[0][1],
        exx[1][0],
        exx[1][1] + U * gram[3].abs(),
    ];
    let def = oracle::forward_error(&gram, &want, &eg, &exy);
    let own = oracle::solve_error(&gram, &xty, &want);
    for i in 0..2 {
        let bound = lib[i] + U * want[i].abs() + def[i] + own[i];
        oracle::assert_within(got[i], want[i], bound, format_args!("coef {i}"));
    }
}

#[test]
fn tracks_a_static_beta() {
    let mut m = Kalman::new(cfg(2, 1, vec![500.0])).unwrap();
    let mut s = 3u64;
    for i in 0..2000 {
        let x = [lcg(&mut s), lcg(&mut s)];
        let y = 2.0 * x[0] - 1.0 * x[1] + 0.5 + 0.05 * lcg(&mut s);
        m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
    }
    let c = &m.coefficients()[0];
    assert!((c[1] - 2.0).abs() < 0.1, "slope0 {}", c[1]);
    assert!((c[2] + 1.0).abs() < 0.1, "slope1 {}", c[2]);
    assert!((c[0] - 0.5).abs() < 0.1, "intercept {}", c[0]);
}

#[test]
fn tracks_a_drifting_beta_better_than_a_pinned_one() {
    // Same data through a responsive filter and a pinned one: the responsive
    // filter must have lower out-of-sample error.
    let mut fast = Kalman::new(cfg(1, 1, vec![50.0])).unwrap();
    let mut pinned = Kalman::new(cfg(1, 1, vec![f64::INFINITY])).unwrap();
    let mut s = 4u64;
    let (mut e_fast, mut e_pin) = (0.0f64, 0.0f64);
    let mut beta = 1.0f64;
    for i in 0..3000 {
        beta += 0.01 * lcg(&mut s); // random walk
        let x = [lcg(&mut s)];
        let y = beta * x[0] + 0.05 * lcg(&mut s);
        let d = if i == 0 { 0.0 } else { 1.0 };
        let a = fast.step(&x, &[Some(y)], d, 1.0);
        let b = pinned.step(&x, &[Some(y)], d, 1.0);
        if i > 500 {
            if a.pred[0].is_finite() {
                e_fast += (y - a.pred[0]).powi(2);
            }
            if b.pred[0].is_finite() {
                e_pin += (y - b.pred[0]).powi(2);
            }
        }
    }
    assert!(e_fast < e_pin, "fast {e_fast} should beat pinned {e_pin}");
}

#[test]
fn infinite_halflife_pins_the_coefficient() {
    // Per-factor: slot 1 (x0) pinned, slot 2 (x1) free.
    let mut m = Kalman::new(cfg(2, 1, vec![1e9, f64::INFINITY, 30.0])).unwrap();
    let mut s = 5u64;
    for i in 0..300 {
        let x = [lcg(&mut s), lcg(&mut s)];
        let y = x[0] + x[1];
        m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
    }
    let q = m.q_vec(1.0);
    assert_eq!(q[1], 0.0, "pinned factor must have zero process noise");
    assert!(q[2] > 0.0);
}

#[test]
fn explicit_q_overrides_halflife() {
    let mut c = cfg(1, 1, vec![10.0]);
    c.q = Some(vec![0.0, 0.25]);
    let m = Kalman::new(c).unwrap();
    assert_eq!(m.q_vec(99.0), vec![0.0, 0.25]);
}

#[test]
fn share_p_keeps_one_covariance() {
    let mut c = cfg(2, 3, vec![100.0]);
    c.share_p = true;
    let mut m = Kalman::new(c).unwrap();
    assert_eq!(m.p.len(), 1);
    let mut s = 6u64;
    for i in 0..200 {
        let x = [lcg(&mut s), lcg(&mut s)];
        let y0 = x[0];
        let y1 = x[1];
        let y2 = x[0] + x[1];
        m.step(
            &x,
            &[Some(y0), Some(y1), Some(y2)],
            if i == 0 { 0.0 } else { 1.0 },
            1.0,
        );
    }
    assert!(
        m.coefficients()
            .iter()
            .all(|c| c.iter().all(|v| v.is_finite()))
    );
}

/// Predictive variance must start wide and narrow with evidence — that is
/// the whole reason to report it alongside `sigma`.
#[test]
fn predictive_variance_narrows_with_evidence() {
    let mut m = Kalman::new(cfg(2, 1, vec![f64::INFINITY])).unwrap();
    let mut s = 44u64;
    let mut early = 0.0;
    let mut late = 0.0;
    for i in 0..3000 {
        let x = [lcg(&mut s), lcg(&mut s)];
        let y = 2.0 * x[0] - x[1] + 0.1 * lcg(&mut s);
        m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        if i == 20 {
            early = m.pred_var(&x)[0];
        }
        if i == 2999 {
            late = m.pred_var(&x)[0];
        }
    }
    assert!(early.is_finite() && late.is_finite());
    assert!(
        late < early,
        "predictive variance should narrow: {early} -> {late}"
    );
}

#[test]
fn predictive_variance_exceeds_the_observation_noise() {
    // It is parameter uncertainty PLUS noise, so it can never be smaller
    // than the noise alone.
    let mut c = cfg(2, 1, vec![f64::INFINITY]);
    c.obs_var = Some(0.25);
    let mut m = Kalman::new(c).unwrap();
    let mut s = 46u64;
    for i in 0..500 {
        let x = [lcg(&mut s), lcg(&mut s)];
        m.step(&x, &[Some(x[0])], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        assert!(m.pred_var(&x)[0] >= 0.25 - 1e-12);
    }
}

#[test]
fn state_roundtrip() {
    let mut m1 = Kalman::new(cfg(2, 1, vec![100.0])).unwrap();
    let mut s = 7u64;
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
    let mut m2 = Kalman::restore(&rmp_serde::from_slice(&bytes).unwrap()).unwrap();
    for (x, y) in &rows[60..] {
        assert_eq!(
            m1.step(x, &[Some(*y)], 1.0, 1.0).pred,
            m2.step(x, &[Some(*y)], 1.0, 1.0).pred
        );
    }
}

#[test]
fn null_target_is_predict_only() {
    let mut m = Kalman::new(cfg(1, 2, vec![100.0])).unwrap();
    let mut s = 8u64;
    for i in 0..60 {
        let x = [lcg(&mut s)];
        m.step(
            &x,
            &[Some(x[0]), Some(-x[0])],
            if i == 0 { 0.0 } else { 1.0 },
            1.0,
        );
    }
    // The coefficients in the caller's units: past the warm-up `b`
    // follows the moments, so the null target's are where they were, to
    // rounding, and the observed target's moved.
    let before = m.coefficients();
    let st = m.step(&[0.5], &[Some(1.0), None], 1.0, 1.0);
    assert!(st.pred[1].is_finite());
    let after = m.coefficients();
    for (a, b) in after[1].iter().zip(&before[1]) {
        assert!((a - b).abs() <= 1e-14 * (1.0 + b.abs()), "{a} against {b}");
    }
    assert_ne!(after[0], before[0], "the target that was there corrected");
}

/// Each slot takes its prior on the first row that observes the target
/// on which its scale is usable (the module doc, task 211): the
/// intercept and a moving feature on the row where the noise first rests
/// on three rows' innovations, a feature that has not moved yet on the
/// first observed row after it has. Until then its variance and its
/// coefficient are 0. With `obs_var` the intercept is sized from the
/// start; unstandardized, the whole `P` on the first row with a noise.
#[test]
fn each_slot_takes_its_prior_once_its_scale_is_usable() {
    let mut c = cfg(2, 1, vec![50.0]);
    c.decay = Decay::Lam(0.9);
    c.min_weight = 0.0;
    let mut m = Kalman::new(c.clone()).unwrap();
    let mut s = 5u64;
    let k = 3;
    for i in 0..12usize {
        // The second feature holds 0.5 for the first 6 rows; the target
        // is null on row 1.
        let x = [lcg(&mut s), if i < 6 { 0.5 } else { lcg(&mut s) }];
        let y = (i != 1).then(|| 1.0 + x[0] - x[1] + 0.1 * lcg(&mut s));
        m.step(&x, &[y], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        let diag: Vec<f64> = (0..k).map(|a| m.p[0][a * k + a]).collect();
        // Rows 0, 2 and 3 observe the target: the third innovation is
        // row 3's, and the first feature's scale is usable from row 2.
        let sized = [i >= 3, i >= 3, i >= 7];
        for a in 0..k {
            assert_eq!(diag[a] != 0.0, sized[a], "row {i}, slot {a}: {diag:?}");
            if !sized[a] {
                assert_eq!(m.beta[0][a], 0.0, "row {i}, slot {a}");
            }
        }
    }
    // With `obs_var` the intercept is sized from the start, and each
    // feature waits for its scale.
    let mut known = c.clone();
    known.obs_var = Some(0.5);
    let m = Kalman::new(known).unwrap();
    assert_eq!(m.p[0][0], 0.5);
    assert_eq!(m.p[0][4], 0.0);
    // Unstandardized there is no scale to wait for and no anchor: the
    // first row with a noise sizes every slot.
    c.standardize = false;
    let mut raw = Kalman::new(c).unwrap();
    raw.step(&[0.3, 0.2], &[Some(1.0)], 0.0, 1.0);
    assert!((0..k).all(|a| raw.p[0][a * k + a] > 0.0));
    assert!(raw.anchor_scale.is_empty());
}

/// A move of the moments re-maps `b` and `P` once they drift past
/// `ANCHOR_DRIFT` from the anchor (the module doc), so a row with no
/// target that moves them -- a feature well off its mean -- leaves the
/// prediction for any row where it was, to rounding, while `b` itself
/// moves. A move past [`REMAP_LIMIT`] -- here a feature at the input
/// bound after a run at a scale of 1e-100, a factor of about 1e200 -- is
/// refused whole, as a move whose numbers would not be finite is: `b`
/// and `P` keep their bits, finite, and the filter goes on predicting.
#[test]
fn a_move_of_the_moments_is_followed_and_one_past_the_limit_is_refused() {
    let mut c = cfg(1, 1, vec![30.0]);
    c.decay = Decay::Lam(0.5);
    c.min_weight = 0.0;
    let mut m = Kalman::new(c).unwrap();
    let mut s = 9u64;
    for _ in 0..40 {
        let x = [lcg(&mut s)];
        m.step(&x, &[Some(1.0 + x[0] + 0.1 * lcg(&mut s))], 1.0, 1.0);
    }
    let at = [0.4];
    let (pred, beta) = (m.predict(&at, 1.0).pred[0], m.beta[0].clone());
    m.step(&[6.0], &[None], 1.0, 1.0);
    assert_ne!(m.beta[0], beta, "the moments moved and b followed");
    assert_eq!(m.anchor_hi[1], m.stats.mean_pair(1).0, "the anchor moved");
    let after = m.predict(&at, 1.0).pred[0];
    assert!(
        (after - pred).abs() <= 1e-14 * (1.0 + pred.abs()),
        "{after} against {pred}"
    );
    // A long run at a scale of 1e-100, the target at its own scale, so
    // that `P` is not tiny with it; then the bound, with no target.
    for _ in 0..1500 {
        let x = [1e-100 * lcg(&mut s)];
        m.step(&x, &[Some(1.0 + 0.1 * lcg(&mut s))], 1.0, 1.0);
    }
    assert!(m.scales()[1] < 1e-90, "the scale shrank: {:?}", m.scales());
    let (beta, p) = (m.beta[0].clone(), m.p[0].clone());
    m.step(&[1e100], &[None], 1.0, 1.0);
    assert_eq!(m.beta[0], beta, "the re-map was refused");
    // `P` took nothing: a row with no target adds no process noise.
    assert_eq!(m.p[0], p);
    assert!(m.predict(&[0.5], 1.0).pred[0].is_finite());
    for _ in 0..50 {
        let x = [lcg(&mut s)];
        let got = m.step(&x, &[Some(1.0 + x[0])], 1.0, 1.0);
        assert!(got.n_eff.is_finite());
    }
    assert!(m.beta[0].iter().all(|v| v.is_finite()));
}

/// The drift is two-sided in a scale: a feature whose spread shrinks by
/// a factor of 4e4 is re-mapped as it passes each halving, so the
/// anchor's scale follows the feature's down, never more than a factor
/// of 2 from it. `|a − 1|` alone never passes 1 for a scale that
/// shrinks, and leaves `a` tiny with arrowhead terms of `q / a²` that
/// cancel in every prediction (the module doc): measured on this stream
/// at 1,500 rows, the anchored replica with that drift parted from the
/// per-row re-map by 8e9 noise standard deviations at `coef_half_life`
/// 20, and this filter by 2.7e-14.
#[test]
fn a_shrinking_scale_is_re_mapped_as_it_halves() {
    let mut c = cfg(1, 1, vec![20.0]);
    c.decay = Decay::Lam(0.8);
    c.min_weight = 0.0;
    let mut m = Kalman::new(c).unwrap();
    let mut s = 31u64;
    let mut spread = 1.0;
    for i in 0..400 {
        if i >= 50 {
            spread *= 0.97;
        }
        let x = [3.0 + spread * lcg(&mut s)];
        let y = 1.0 + 2.0 * (x[0] - 3.0) / spread + 0.1 * lcg(&mut s);
        assert!(m.step(&x, &[Some(y)], 1.0, 1.0).pred[0].is_finite() || i < 3);
        let a = m.scales()[1] / m.anchor_scale[1];
        assert!(
            i < 10 || (0.5..=2.0).contains(&a),
            "row {i}: the scale is {a} of the anchor's"
        );
    }
    assert!(m.anchor_scale[1] < 1e-4, "{}", m.anchor_scale[1]);
}

// ---- reversion (ENHANCEMENTS E41) ----

fn revert_cfg(r: Vec<f64>) -> KalmanCfg {
    KalmanCfg {
        revert_half_life: r,
        ..cfg(2, 1, vec![100.0])
    }
}

#[test]
fn revert_halflife_defaults_to_the_random_walk_when_a_state_file_omits_it() {
    let json = r#"{
        "n_features": 2, "n_targets": 1, "fit_intercept": true,
        "decay": {"Halflife": 200.0}, "half_life": [100.0], "q": null,
        "obs_var": null, "p0": 1.0, "share_p": false, "min_weight": 10.0
    }"#;
    let cfg: KalmanCfg = serde_json::from_str(json).expect("should load without the field");
    assert_eq!(cfg.revert_half_life, vec![f64::INFINITY]);
    assert!(!cfg.reverts());
}

#[test]
fn revert_halflife_is_validated() {
    for (r, msg) in [
        (vec![10.0, 10.0], "or one for each of the 3 coefficients"),
        (vec![0.0], "must be above 0"),
        (vec![-5.0], "must be above 0"),
        (vec![f64::NAN], "must be above 0"),
        (
            vec![f64::INFINITY, 10.0, f64::NEG_INFINITY],
            "must be above 0",
        ),
    ] {
        let err = Kalman::new(revert_cfg(r.clone())).unwrap_err();
        assert!(err.contains(msg), "{r:?}: {err}");
    }
    for r in [
        vec![f64::INFINITY],
        vec![10.0],
        vec![f64::INFINITY, 5.0, 1e300],
    ] {
        Kalman::new(revert_cfg(r)).unwrap();
    }
}

#[test]
fn an_infinite_revert_halflife_is_bit_identical_to_the_default() {
    // Spelled as a scalar or per slot, `inf` must not touch a number:
    // the transition is skipped, not multiplied by 1.
    let a = fit(cfg(2, 1, vec![100.0]), 200, 5);
    let b = fit(revert_cfg(vec![f64::INFINITY]), 200, 5);
    let c = fit(revert_cfg(vec![f64::INFINITY; 3]), 200, 5);
    assert_eq!(a.beta, b.beta);
    assert_eq!(a.p, b.p);
    assert_eq!(a.beta, c.beta);
    assert_eq!(a.p, c.p);
    assert_eq!(
        a.predict(&[0.3, 0.7], 2.5).pred,
        c.predict(&[0.3, 0.7], 2.5).pred
    );
}

#[test]
fn reversion_is_exact_between_observations() {
    // With nothing to learn from (null targets), the mean shrinks by
    // exactly `2^(-d/r_i)` per slot over the elapsed clock and the
    // covariance by `phi_i phi_j` (with `q = 0` so nothing is added
    // back), whatever the clock's spacing. On the unstandardized
    // filter so the reported coefficients are the state itself.
    let r = vec![f64::INFINITY, 20.0, 5.0];
    let mut m = Kalman::new(KalmanCfg {
        q: Some(vec![0.0; 3]),
        standardize: false,
        ..revert_cfg(r.clone())
    })
    .unwrap();
    let mut s = 11u64;
    for i in 0..80 {
        let x = [lcg(&mut s), lcg(&mut s)];
        let y = 0.4 + 1.5 * x[0] - 2.0 * x[1] + 0.05 * lcg(&mut s);
        m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
    }
    let b0 = m.beta[0].clone();
    let p0 = m.p[0].clone();
    let gaps = [0.5, 3.0, 1.0, 0.0, 7.25, 2.0];
    for d in gaps {
        let x = [lcg(&mut s), lcg(&mut s)];
        m.step(&x, &[None], d, 1.0);
    }
    let total: f64 = gaps.iter().sum();
    let phi: Vec<f64> = r.iter().map(|h| (-(total / h)).exp2()).collect();
    assert_eq!(phi[0], 1.0);
    for i in 0..3 {
        let want = b0[i] * phi[i];
        assert!(
            (m.beta[0][i] - want).abs() <= 1e-12 * want.abs().max(1e-300),
            "slot {i}: {} vs {want}",
            m.beta[0][i]
        );
        for j in 0..3 {
            let want = p0[i * 3 + j] * phi[i] * phi[j];
            assert!(
                (m.p[0][i * 3 + j] - want).abs() <= 1e-12 * want.abs().max(1e-300),
                "P[{i},{j}]: {} vs {want}",
                m.p[0][i * 3 + j]
            );
        }
    }
    // The intercept slot, `inf`, was left alone to the bit.
    assert_eq!(m.beta[0][0], b0[0]);
    assert_eq!(m.p[0][0], p0[0]);
    // And the state is what `coefficients` reports: unstandardized.
    assert_eq!(m.coefficients()[0], m.beta[0]);
}

/// The filter is sizing ([`Kalman::sizing`], the stream's `seeding`)
/// exactly while a target it has weight on waits for its prior: never
/// before a row, nor for a target no row has carried; standardized
/// without `obs_var`, from a target's first row until three of its rows
/// have sized it, as the predictions show (`an_unsized_prior_predicts_nothing`):
/// a target that arrives later makes the filter size again; shared,
/// the one prior serves the later target at once. Unstandardized, the
/// first observed row sizes `P`, so the filter is never seen sizing
/// between rows. Each row's state is checked against what the next
/// row's prediction shows: a target with weight past `min_weight` and
/// no prediction.
#[test]
fn the_filter_is_sizing_while_a_weighted_target_waits_for_its_prior() {
    for (standardize, share_p) in [(true, false), (true, true), (false, false)] {
        let mut c = cfg(1, 2, vec![100.0]);
        c.standardize = standardize;
        c.share_p = share_p;
        c.min_weight = 0.5;
        let mut m = Kalman::new(c).unwrap();
        assert!(!m.sizing(), "before a row");
        let mut s = 20261024u64;
        let mut seen = Vec::new();
        for i in 0..16 {
            let x = [lcg(&mut s)];
            let y0 = 10.0 + 2.0 * x[0] + 0.5 * lcg(&mut s);
            let y1 = (i >= 8).then(|| -5.0 + x[0] + 0.5 * lcg(&mut s));
            let d = if i == 0 { 0.0 } else { 1.0 };
            m.step(&x, &[Some(y0), y1], d, 1.0);
            seen.push(m.sizing());
            // What the next row would show: a target with weight and
            // no prediction.
            let next = m.predict(&[0.3], 1.0).pred;
            let waiting = (0..2).any(|j| m.wj[j] > 0.0 && next[j].is_nan());
            assert_eq!(
                m.sizing(),
                waiting,
                "standardize {standardize}, share {share_p}, row {i}"
            );
        }
        let want: Vec<bool> = (0..16)
            .map(|i| standardize && (i < 2 || (!share_p && (8..10).contains(&i))))
            .collect();
        assert_eq!(seen, want, "standardize {standardize}, share {share_p}");
    }
}

/// A filter whose prior is not sized predicts nothing (review round 9,
/// B-1; docs/PLAN.md task 270). Standardized without `obs_var`, a slot
/// is sized once three observed rows give the noise a median, and on a
/// row past `min_weight` before then -- here every row carries weight
/// 3 and the floor is 1 -- the prediction was `z·b = 0.0` whatever the
/// target's level, and that 0 seeded the residual variance with the
/// level squared. Unstandardized, `P` is sized on the first observed
/// row, after its prediction. Each row's prediction is withheld exactly
/// while the target's covariance is unsized, `predict` agrees, no
/// prediction is 0.0 on a target at 100, and the residual variance is
/// 0 until the first prediction and then that prediction's squared
/// residual: it starts at the sized filter's first prediction.
#[test]
fn an_unsized_prior_predicts_nothing() {
    for standardize in [true, false] {
        let mut c = cfg(1, 1, vec![100.0]);
        c.standardize = standardize;
        c.min_weight = 1.0;
        let mut m = Kalman::new(c).unwrap();
        let mut s = 20261023u64;
        let mut first = None;
        for i in 0..200 {
            let x = [lcg(&mut s)];
            let yv = 100.0 + 2.0 * x[0] + 0.5 * lcg(&mut s);
            let d = if i == 0 { 0.0 } else { 1.0 };
            let waiting = m.is_unsized(0);
            let served = m.predict(&x, d).pred[0];
            let got = m.step(&x, &[Some(yv)], d, 3.0).pred[0];
            let what = format!("standardize {standardize}, row {i}");
            assert_eq!(served.to_bits(), got.to_bits(), "{what}");
            assert_eq!(got.is_nan(), waiting || i == 0, "{what}: {got}");
            assert_ne!(got, 0.0, "{what}");
            if got.is_nan() && first.is_none() {
                assert_eq!(m.sig2[0], 0.0, "{what}: no prediction, no residual");
            } else if first.is_none() {
                first = Some(i);
                assert_eq!(m.sig2[0], (yv - got) * (yv - got), "{what}");
            }
        }
        let first = first.unwrap();
        assert!(
            if standardize { first == 3 } else { first == 1 },
            "standardize {standardize}: first prediction at row {first}"
        );
    }
}

/// A target at the input bound among the first rows (docs/PLAN.md task
/// 214, found by task 211, older than it): its squared innovation,
/// `1e200`, made a third of the mean the priors were sized from, `P`
/// started near `3e199`, the update left a slot's variance negative,
/// and the filter never learned again -- its slopes sat at `1.8e-101`
/// and `1.20` after 30,000 rows, against a truth of 1 and -0.5. Here a
/// feature at the bound on row 0 and both targets at it on row 1 lead a
/// stream of 20,000 ordinary rows: the filter learns the truth, as the
/// stream without the two rows does, every variance it reports is at
/// least 0 (none before there is a noise), and every prediction is the one `predict` gave
/// before the row, which never reads the row's target (hard rule 2).
#[test]
fn a_target_at_the_input_bound_among_the_first_rows_is_survived() {
    let mut c = cfg(2, 2, vec![100.0]);
    c.decay = Decay::Halflife(20.0);
    c.min_weight = 3.0;
    let mut m = Kalman::new(c).unwrap();
    let mut s = 20260901u64;
    let row = |s: &mut u64| -> (Vec<f64>, Vec<Option<f64>>) {
        let x: Vec<f64> = (0..2).map(|_| lcg(s)).collect();
        let y = (0..2)
            .map(|j| Some(0.5 * (j as f64 + 1.0) + x[0] - 0.5 * x[1]))
            .collect();
        (x, y)
    };
    let (mut x0, y0) = row(&mut s);
    x0[0] = crate::INPUT_BOUND;
    let (x1, mut y1) = row(&mut s);
    y1.iter_mut().for_each(|v| *v = Some(crate::INPUT_BOUND));
    let mut rows = vec![(x0, y0), (x1, y1)];
    rows.extend((0..20_000).map(|_| row(&mut s)));
    for (i, (x, y)) in rows.iter().enumerate() {
        let d = if i == 0 { 0.0 } else { 1.0 };
        let served = m.predict(x, d).pred;
        let got = m.step(x, y, d, 1.0).pred;
        assert!(
            served
                .iter()
                .zip(&got)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "row {i}: {served:?} before the row, {got:?} from it"
        );
        if i % 97 == 0
            && let Some(crate::CoefVariance::Absolute(v)) = m.coef_variance()
        {
            assert!(
                v.iter().flatten().all(|v| v.is_nan() || *v >= 0.0),
                "row {i}: variances {v:?}"
            );
        }
    }
    for (j, b) in m.coefficients().iter().enumerate() {
        let truth = [0.5 * (j as f64 + 1.0), 1.0, -0.5];
        assert!(
            b.iter().zip(&truth).all(|(u, v)| (u - v).abs() < 0.02),
            "target {j}: {b:?} against {truth:?}"
        );
    }
}

/// A covariance that has lost its positive-definiteness, a variance
/// below 0 on its diagonal, is sized again from the noise as it stands,
/// as at the start (docs/PLAN.md task 214): the row it shows on is
/// predicted as `predict` predicts it, before the row, and after it
/// every variance is a number of at least 0 -- unstandardized, `p0`
/// times the residual variance on the whole diagonal -- and the filter
/// learns the truth again. Damaged here by hand, a slope's variance at
/// `-1e6` against its neighbours'; the median prior keeps the stream
/// that did it in task 211 from getting here.
#[test]
fn a_covariance_that_loses_its_positive_diagonal_is_sized_again() {
    // In units of the noise too (task 278), where it is sized at `p0`.
    for (standardize, units) in [(true, false), (false, false), (true, true), (false, true)] {
        let mut c = cfg(2, 1, vec![100.0]);
        c.decay = Decay::Halflife(50.0);
        c.min_weight = 3.0;
        c.standardize = standardize;
        c.noise_units = units;
        let mut m = Kalman::new(c).unwrap();
        let mut s = 5u64;
        let mut row = || {
            let x = [2.0 + lcg(&mut s), lcg(&mut s)];
            let y = 0.3 + x[0] - 0.5 * x[1] + 0.05 * lcg(&mut s);
            (x, [Some(y)])
        };
        for i in 0..300 {
            let (x, y) = row();
            m.step(&x, &y, if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        let k = 3;
        m.p[0][k + 1] = -1e6;
        let variances = |m: &Kalman| match m.coef_variance() {
            Some(crate::CoefVariance::Absolute(v)) => v[0].clone(),
            other => panic!("{other:?}"),
        };
        assert!(variances(&m)[1] < 0.0, "the damage shows");
        for i in 0..3000 {
            let (x, y) = row();
            let served = m.predict(&x, 1.0).pred;
            let got = m.step(&x, &y, 1.0, 1.0).pred;
            assert_eq!(served[0].to_bits(), got[0].to_bits(), "row {i}");
            let v = variances(&m);
            assert!(
                v.iter().all(|v| v.is_finite() && *v >= 0.0),
                "standardize {standardize}, row {i}: {v:?}"
            );
            if i == 0 && !standardize {
                let want = if units {
                    m.cfg.p0
                } else {
                    m.cfg.p0 * m.readiness_noise(0)
                };
                assert!(
                    (0..k).all(|a| m.p[0][a * k + a] == want),
                    "{:?} against {want}",
                    m.p[0]
                );
            }
        }
        let b = &m.coefficients()[0];
        assert!(
            b.iter()
                .zip([0.3, 1.0, -0.5])
                .all(|(u, v)| (u - v).abs() < 0.02),
            "standardize {standardize}: {b:?}"
        );
    }
}

/// A reverting slot's process noise for a gap `D` is `q g(D)²`, `g(D) =
/// (1 − 2^(−D/r)) / θ`, `θ = ln 2 / r` (the module doc, docs/PLAN.md task
/// 214), read through `P` as it stands after a row that observes
/// nothing, which carries the noise the next observation will charge:
/// `q D²` for a short gap (the random walk's, to 1e-3 at `D = r/1000`),
/// and `q / θ²` past many half-lives, however long the gap -- the same
/// after 1,000 rows of null targets as after 10,000. Charged `q D²`, as
/// task 211 did, it was 769 and then 7.7e4 times that bound there. The
/// walk beside it keeps `q D²`. The longhand: the variance after the
/// gap less the transition's `phi² P`.
#[test]
fn a_reverting_slots_gap_noise_is_bounded() {
    let (q, r) = (1e-3, 25.0);
    let theta = std::f64::consts::LN_2 / r;
    let mk = |revert: f64| {
        let mut c = plain(1.0, f64::INFINITY, Some(0.04));
        c.decay = Decay::Halflife(f64::INFINITY);
        c.q = Some(vec![q]);
        c.revert_half_life = vec![revert];
        let mut m = Kalman::new(c).unwrap();
        let mut s = 7u64;
        for i in 0..200 {
            let x = [1.0 + 0.3 * lcg(&mut s)];
            let y = 0.5 * x[0] + 0.2 * lcg(&mut s);
            m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        m
    };
    let var = |m: &Kalman| match m.coef_variance() {
        Some(crate::CoefVariance::Absolute(v)) => v[0][0],
        other => panic!("{other:?}"),
    };
    let added = |revert: f64, gap: f64, rows: usize| {
        let mut m = mk(revert);
        let before = var(&m);
        for _ in 0..rows {
            m.step(&[1.0], &[None], gap / rows as f64, 1.0);
        }
        let phi = Decay::Halflife(revert).factor(gap);
        var(&m) - phi * phi * before
    };
    let short = r / 1000.0;
    let got = added(r, short, 1) / (q * short * short);
    assert!((got - 1.0).abs() < 1e-3, "a short gap: {got} of q D²");
    let bound = q / (theta * theta);
    for rows in [1_000, 10_000] {
        let gap = rows as f64;
        let got = added(r, gap, rows);
        assert!(
            (got - bound).abs() <= 1e-9 * bound,
            "{rows} rows of null targets: {got} against the bound {bound}"
        );
        let walk = added(f64::INFINITY, gap, rows);
        assert!(
            (walk - q * gap * gap).abs() <= 1e-9 * q * gap * gap,
            "the walk: {walk}"
        );
    }
}

#[test]
fn a_reverting_slot_forgets_a_stale_effect_and_a_random_walk_keeps_it() {
    // 300 rows identify a slope of 2 on `x1`, then `x1` goes flat at
    // zero for 300 rows: no row says anything about that slope any more.
    // The random walk carries the 2 for ever; the reverting filter lets
    // it go at `2^(-d/r)`. Predictions agree either way (`x1 = 0`), so
    // this is only visible in the coefficients -- the point of E41 is
    // what the filter believes when the evidence dries up.
    let run = |r: Vec<f64>| {
        let mut m = Kalman::new(KalmanCfg {
            standardize: false,
            ..revert_cfg(r)
        })
        .unwrap();
        let mut s = 12u64;
        for i in 0..600 {
            let x1 = if i < 300 { lcg(&mut s) } else { 0.0 };
            let x = [lcg(&mut s), x1];
            let y = 0.5 * x[0] + 2.0 * x[1] + 0.05 * lcg(&mut s);
            m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        m.coefficients()[0].clone()
    };
    let walk = run(vec![f64::INFINITY]);
    let revert = run(vec![f64::INFINITY, f64::INFINITY, 30.0]);
    assert!(
        (walk[2] - 2.0).abs() < 0.1,
        "random walk keeps the slope: {walk:?}"
    );
    // 300 rows at half-life 30 is 2^-10 of the slope.
    assert!(
        revert[2].abs() < 2.0 * 2f64.powi(-9),
        "reverting slot forgets it: {revert:?}"
    );
    // The slope still in evidence is learned equally well by both.
    assert!((walk[1] - 0.5).abs() < 0.05, "{walk:?}");
    assert!((revert[1] - 0.5).abs() < 0.05, "{revert:?}");
}

#[test]
fn reversion_is_applied_once_per_row_under_share_p() {
    // One shared `P`, two targets: the transition runs once, not once
    // per target, or the shared covariance would shrink twice.
    let r = vec![f64::INFINITY, 10.0, 10.0];
    let mut m = Kalman::new(KalmanCfg {
        q: Some(vec![0.0; 3]),
        share_p: true,
        standardize: false,
        ..revert_cfg(r)
    })
    .unwrap();
    m.cfg.n_targets = 2;
    let mut m = Kalman::new(m.cfg.clone()).unwrap();
    let mut s = 13u64;
    for i in 0..50 {
        let x = [lcg(&mut s), lcg(&mut s)];
        m.step(
            &x,
            &[Some(x[0]), Some(-x[1])],
            if i == 0 { 0.0 } else { 1.0 },
            1.0,
        );
    }
    let p0 = m.p[0].clone();
    let b0 = m.beta.clone();
    m.step(&[0.1, 0.2], &[None, None], 10.0, 1.0);
    // phi = 2^-1 for both slopes over d = 10 at half-life 10.
    assert!((m.p[0][4] - p0[4] * 0.25).abs() <= 1e-12 * p0[4].abs());
    assert!((m.p[0][1] - p0[1] * 0.5).abs() <= 1e-12 * p0[1].abs());
    for (after, before) in m.beta.iter().zip(&b0) {
        assert!((after[1] - before[1] * 0.5).abs() <= 1e-12 * before[1].abs());
    }
}

#[test]
fn a_zero_weight_row_still_advances_the_transition() {
    // Weight 0 means "advance the clock, learn nothing": the reversion
    // is clock, so it applies; the measurement update does not.
    let mut m = Kalman::new(KalmanCfg {
        q: Some(vec![0.0; 3]),
        standardize: false,
        ..revert_cfg(vec![4.0])
    })
    .unwrap();
    let mut s = 14u64;
    for i in 0..40 {
        let x = [lcg(&mut s), lcg(&mut s)];
        m.step(
            &x,
            &[Some(x[0] + x[1])],
            if i == 0 { 0.0 } else { 1.0 },
            1.0,
        );
    }
    let b0 = m.beta[0].clone();
    m.step(&[0.3, 0.3], &[Some(100.0)], 4.0, 0.0);
    for (i, (after, before)) in m.beta[0].iter().zip(&b0).enumerate() {
        assert!((after - 0.5 * before).abs() <= 1e-12 * before.abs(), "{i}");
    }
}

/// A map without `stats` is refused rather than defaulted, and so is the
/// schema-2 layout's `cov` in its place.
#[test]
fn a_state_without_the_standardizer_is_refused() {
    let mut c = cfg(2, 1, vec![100.0]);
    c.revert_half_life = vec![500.0]; // JSON has no `inf`
    let m = fit(c, 30, 5);
    let v = serde_json::to_value(&m).unwrap();
    // The control: the same value loads with the field present.
    let back: Kalman = serde_json::from_value(v.clone()).unwrap();
    assert_eq!(back.stats, m.stats);
    let mut v = v;
    v.as_object_mut().unwrap().remove("stats");
    assert!(serde_json::from_value::<Kalman>(v).is_err());
    // Nor does the schema-2 layout it used to be told apart from: a
    // `cov` in place of `stats` is now simply a missing field.
    let mut v = serde_json::to_value(&m).unwrap();
    let obj = v.as_object_mut().unwrap();
    obj.remove("stats");
    obj.insert("cov".into(), serde_json::Value::Null);
    assert!(serde_json::from_value::<Kalman>(v).is_err());
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

/// A state whose coefficients or their covariance are not its cfg's shape
/// is refused as it is read (review 2026-09-18, B3; docs/PLAN.md task
/// 111: every other model had this test).
#[test]
fn a_state_of_the_wrong_shape_is_refused() {
    let m = Kalman::new(cfg(2, 1, vec![50.0])).unwrap();
    assert!(reread(&m, |_| {}).is_ok(), "the control");
    for field in ["beta", "p"] {
        let err = reread(&m, |v| shorten_first(v, field)).unwrap_err();
        assert!(err.contains("wrong shape"), "{field}: {err}");
    }
}

/// The list `field` of the state, one entry short.
fn shorten(v: &mut rmpv::Value, field: &str) {
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
    rows.pop();
}

/// Each per-target list is checked on its own: a state one target short
/// in its coefficients, its residual weights or its target weights alone
/// is refused (task 158).
#[test]
fn a_state_one_target_short_in_any_list_is_refused() {
    let m = Kalman::new(cfg(2, 2, vec![50.0])).unwrap();
    assert!(reread(&m, |_| {}).is_ok(), "the control");
    for field in ["beta", "wsig", "wj"] {
        let err = reread(&m, |v| shorten(v, field)).unwrap_err();
        assert!(
            err.contains("kalman: its saved state has the wrong shape"),
            "{field}: {err}"
        );
    }
}

/// Before any residual there is no observation variance to report, and
/// `pred_var` says so with NaN rather than a 0 (task 158).
#[test]
fn pred_var_is_nan_before_any_residual() {
    let m = Kalman::new(cfg(2, 1, vec![50.0])).unwrap();
    assert!(m.pred_var(&[0.3, -0.2])[0].is_nan());
}

/// Each target's weight is the EW sum of the weights of the rows that
/// carried it, null or weight-0 rows only ageing it, and the model
/// reports it (task 158).
#[test]
fn the_target_weights_are_the_rows_that_carried_each_target() {
    let mut c = cfg(2, 2, vec![50.0]);
    c.decay = Decay::Halflife(7.0);
    let mut m = Kalman::new(c).unwrap();
    let mut want = [0.0f64; 2];
    let mut s = 97u64;
    for i in 0..40 {
        let x = [lcg(&mut s), lcg(&mut s)];
        let w = if i % 5 == 4 { 0.0 } else { 1.0 + lcg(&mut s) };
        let ys = [Some(x[0]), (i % 3 == 0).then_some(x[1])];
        let d = if i == 0 { 0.0 } else { 1.0 + f64::from(i % 2) };
        m.step(&x, &ys, d, w);
        let lam = 0.5f64.powf(d / 7.0);
        for (j, wj) in want.iter_mut().enumerate() {
            *wj = lam * *wj + if ys[j].is_some() { w } else { 0.0 };
        }
    }
    let mut out = vec![-1.0; 5];
    assert!(m.target_n_eff_into(&mut out));
    assert_eq!(out.len(), 2);
    for j in 0..2 {
        assert!(
            (out[j] - want[j]).abs() <= 1e-12 * want[j],
            "{out:?} against {want:?}"
        );
    }
    assert!(want[1] < want[0], "the targets' weights differ");
}

/// Without an intercept, `standardize` scales each feature by the root of
/// its raw second moment, so the filter cannot tell a feature's units:
/// the same stream with one feature in thousands and one in thousandths
/// predicts the same, and a feature that is always 0 is left at 0 rather
/// than divided by a zero scale. `predict` is the next `step`'s number
/// to the bit (task 158).
#[test]
fn standardizing_without_an_intercept_does_not_see_the_units() {
    let run = |units: [f64; 2]| {
        let mut c = cfg(3, 1, vec![30.0]);
        c.fit_intercept = false;
        c.min_weight = 1.5;
        c.decay = Decay::Halflife(40.0);
        let mut m = Kalman::new(c).unwrap();
        let mut s = 101u64;
        let mut preds = Vec::new();
        for i in 0..200 {
            let x = [2.0 + lcg(&mut s), lcg(&mut s)];
            let y = 0.8 * x[0] - 1.5 * x[1] + 0.1 * lcg(&mut s);
            let xs = [units[0] * x[0], units[1] * x[1], 0.0];
            let d = if i == 0 { 0.0 } else { 1.0 };
            let ahead = m.predict(&xs, d).pred[0];
            // The first row meets no scale yet and is read raw, so it
            // carries no target: it only sets the scales.
            let got = m.step(&xs, &[(i > 0).then_some(y)], d, 1.0).pred[0];
            assert_eq!(
                ahead.to_bits(),
                got.to_bits(),
                "row {i}: predict is step's number"
            );
            // Withheld until the prior is sized, from the first three
            // rows that carry the target, 1 to 3 (task 270, B-1).
            assert_eq!(got.is_nan(), i < 4, "row {i}: {got}");
            preds.push(got);
        }
        preds
    };
    let (a, b) = (run([1.0, 1.0]), run([1000.0, 0.001]));
    for (i, (pa, pb)) in a.iter().zip(&b).enumerate().skip(4) {
        assert!(
            (pa - pb).abs() <= 1e-9 * pa.abs().max(1.0),
            "row {i}: {pa} against {pb}"
        );
    }
}

/// A row of weight 0 learns nothing, the residual variance included:
/// across every such row `σ²` keeps its bits, not `(a σ²) / a` (task
/// 158).
#[test]
fn a_zero_weight_row_keeps_the_residual_variance_to_the_bit() {
    let mut c = cfg(2, 1, vec![50.0]);
    c.decay = Decay::Halflife(9.0);
    c.min_weight = 0.0;
    let mut m = Kalman::new(c).unwrap();
    let mut s = 107u64;
    let mut checked = 0;
    for i in 0..300 {
        let x = [lcg(&mut s), lcg(&mut s)];
        let y = x[0] - x[1] + 0.3 * lcg(&mut s);
        let w = if i % 3 == 2 { 0.0 } else { 1.0 };
        let before = m.sigma2()[0];
        m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 0.7 }, w);
        if w == 0.0 && before > 0.0 {
            assert_eq!(m.sigma2()[0].to_bits(), before.to_bits(), "row {i}");
            checked += 1;
        }
    }
    assert!(checked > 90);
}

/// A row whose innovation variance is 0 -- a zero regressor and an
/// observation variance that underflows at the row's weight -- is
/// skipped, not divided by; and so is a target that is not a number
/// (task 158).
#[test]
fn an_innovation_of_zero_or_a_target_not_a_number_corrects_nothing() {
    let mut m = Kalman::new(KalmanCfg {
        horizons: Vec::new(),
        min_weight_holds_learning: false,
        noise_units: false,
        n_features: 1,
        n_targets: 1,
        fit_intercept: false,
        decay: Decay::Halflife(f64::INFINITY),
        half_life: vec![f64::INFINITY],
        q: None,
        obs_var: Some(f64::from_bits(1)),
        p0: 1.0,
        share_p: false,
        min_weight: 0.0,
        revert_half_life: vec![f64::INFINITY],
        standardize: false,
    })
    .unwrap();
    assert_eq!(f64::from_bits(1) / 4.0, 0.0, "R / w underflows");
    m.step(&[0.0], &[Some(1.0)], 0.0, 4.0);
    assert_eq!(m.beta[0], vec![0.0], "s = 0: skipped");
    m.step(&[1.0], &[Some(f64::NAN)], 1.0, 1.0);
    assert_eq!(m.beta[0], vec![0.0], "a NaN target: skipped");
    assert!(m.step(&[1.0], &[Some(2.0)], 1.0, 1.0).pred[0].is_finite());
}

/// Before a target has a residual variance, a row that gives its noise
/// no scale leaves the filter as it was, to the bit, though the clock
/// moves: an innovation of exactly 0, a row of weight 0, a null target.
/// Taken as `R = 0`, an exact row would collapse `P` along `z`, so the
/// filter would hold its first fit with no doubt at all; the other two
/// have no innovation to read. None of them corrects anything or adds
/// process noise from `coef_half_life`, and the next row that has an
/// innovation corrects the filter (CC4). With a shared `P`, the same when
/// no target the row observes has an innovation other than 0.
#[test]
fn rows_that_size_no_noise_leave_the_filter_as_it_was() {
    for share in [false, true] {
        let mut c = cfg(2, 2, vec![40.0]);
        c.standardize = false;
        c.min_weight = 0.0;
        c.share_p = share;
        let fresh = Kalman::new(c).unwrap();
        let mut m = fresh.clone();
        // `b = 0`, so a target of 0 is an innovation of 0; clock steps of
        // 2, where a process noise would show.
        m.step(&[0.3, -0.2], &[Some(0.0), Some(0.0)], 2.0, 1.0);
        m.step(&[-0.5, 0.4], &[Some(0.0), None], 2.0, 1.0);
        m.step(&[0.7, 0.1], &[Some(3.0), Some(-2.0)], 2.0, 0.0);
        m.step(&[0.2, 0.2], &[None, None], 2.0, 1.0);
        assert_eq!(m.beta, fresh.beta, "share {share}");
        assert_eq!(m.p, fresh.p, "share {share}: no collapse, no process noise");
        assert_eq!(
            m.sig2,
            vec![0.0, 0.0],
            "share {share}: still no residual variance"
        );
        // Innovations of 1 and -1: the prior is sized at `p0 e² = 1`, then
        // narrowed by the correction.
        m.step(&[0.1, 0.6], &[Some(1.0), Some(-1.0)], 1.0, 1.0);
        assert!(
            m.beta.iter().all(|b| b.iter().any(|v| *v != 0.0)),
            "share {share}"
        );
        let p00 = m.p[0][0];
        assert!(
            p00 > 0.0 && p00 < 1.0,
            "share {share}: P sized and narrowed, {p00}"
        );
    }
}

/// `p0` is a ratio: the prior variance is `p0` times the first noise
/// estimate, placed on the row that first sizes the noise, before its
/// gain, with no process noise on that row (CC4). Until then `P` is
/// unsized, all zero, and nothing adds to it, an explicit `q` included:
/// not an exact observation, a null target at a gap, nor a row of weight
/// 0. So the first correction, `P_0 z e / (zᵀ P_0 z + e² / w)` with
/// `P_0 = p0 e² I`, is `p0 z e / (p0 zᵀz + 1 / w)`, and does not see the
/// target's units. With `obs_var` given the noise is known from the
/// start, and `P_0 = p0 obs_var I` from construction.
#[test]
fn the_prior_is_p0_times_the_first_noise() {
    for share in [false, true] {
        let mut c = cfg(2, 1, vec![40.0]);
        c.standardize = false;
        c.min_weight = 0.0;
        c.share_p = share;
        c.p0 = 0.5;
        c.q = Some(vec![0.1, 0.2, 0.3]);
        let mut m = Kalman::new(c).unwrap();
        let no_prior = |m: &Kalman| m.p[0].iter().all(|v| *v == 0.0);
        assert!(no_prior(&m), "share {share}: unsized from the start");
        m.step(&[0.3, -0.2], &[Some(0.0)], 0.0, 1.0);
        m.step(&[0.1, 0.4], &[None], 3.0, 1.0);
        m.step(&[0.7, 0.1], &[Some(5.0)], 3.0, 0.0);
        assert!(no_prior(&m), "share {share}: still unsized");
        let (x, y, w) = ([0.2, -0.6], 3.0, 2.0);
        m.step(&x, &[Some(y)], 3.0, w);
        let z = [1.0, x[0], x[1]];
        let zz: f64 = z.iter().map(|v| v * v).sum();
        let p0e2 = 0.5 * y * y;
        for i in 0..3 {
            let want = 0.5 * z[i] * y / (0.5 * zz + 1.0 / w);
            let got = m.beta[0][i];
            assert!(
                (got - want).abs() <= 1e-14 * want.abs(),
                "share {share}, b[{i}]: {got} against {want}"
            );
            for jj in 0..3 {
                let eye = if i == jj { 1.0 } else { 0.0 };
                let want = p0e2 * eye - p0e2 * p0e2 * z[i] * z[jj] / (p0e2 * zz + y * y / w);
                let got = m.p[0][i * 3 + jj];
                assert!(
                    (got - want).abs() <= 1e-14 * p0e2,
                    "share {share}, P[{i},{jj}]: {got} against {want}"
                );
            }
        }
    }
    // Unstandardized every slot; standardized the intercept, each
    // feature's prior waiting for its scale (task 211).
    for standardize in [false, true] {
        let mut c = cfg(2, 1, vec![40.0]);
        c.obs_var = Some(0.25);
        c.p0 = 3.0;
        c.standardize = standardize;
        let m = Kalman::new(c).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                let sized = i == j && (i == 0 || !standardize);
                assert_eq!(m.p[0][i * 3 + j], if sized { 0.75 } else { 0.0 });
            }
        }
    }
}

/// A target that is not a number has no innovation, so before the first
/// residual it sizes no noise: its own `P` is left unsized, with no
/// process noise, where its square would have put a NaN on the diagonal.
/// Under `share_p` the shared noise is then the other targets', and they
/// learn on the row with it (CC4). Here that is `R = 4`, the finite
/// target's innovation of 2, squared, against the prior it sizes on the
/// row, `P = p0 R I = 4 I`.
#[test]
fn a_target_not_a_number_sizes_no_noise() {
    for share in [false, true] {
        let mut c = cfg(2, 2, vec![40.0]);
        c.standardize = false;
        c.min_weight = 0.0;
        c.share_p = share;
        let fresh = Kalman::new(c).unwrap();
        let mut m = fresh.clone();
        m.step(&[0.3, -0.2], &[Some(f64::NAN), Some(2.0)], 1.0, 1.0);
        assert_eq!(m.beta[0], fresh.beta[0], "share {share}");
        if !share {
            assert_eq!(m.p[0], fresh.p[0], "the NaN target's own P");
        }
        let z = [1.0, 0.3, -0.2];
        let zz: f64 = z.iter().map(|v| v * v).sum();
        for (i, b) in m.beta[1].iter().enumerate() {
            let want = 4.0 * z[i] * 2.0 / (4.0 * zz + 4.0);
            assert!(
                (b - want).abs() <= 1e-15,
                "share {share}, slot {i}: {b} against {want}"
            );
        }
    }
}

/// Scaling every target by `c` scales every prediction by `c`, and `σ²`
/// by `c²`, at any `p0`, once an `obs_var` or `q` given in the target's
/// units is scaled by `c²` with it: the filter has no unit of its own.
/// Before a target's first residual its noise was the literal 1.0, in the
/// target's units, so the warm-up's gains, and every prediction after
/// them, moved with the units: 100% apart between scales of 1e-6 and 1e6
/// (review 2026-10-05, CC4). The prior variance was `p0` in the target's
/// units too, and is now `p0` times the first noise estimate. Powers of
/// two keep every operation exact, so the comparison is to the bit. The
/// stream opens on an exact observation (an innovation of 0), a row of
/// weight 0 and a gap of five clock units fall before any residual, and
/// the second target joins on that gap's row; per-target and shared `P`,
/// standardized and not, reverting and not, `p0` other than 1, and a
/// fixed `obs_var`.
#[test]
fn scaling_the_targets_scales_every_prediction_to_the_bit() {
    type Row = ([f64; 2], [Option<f64>; 2], f64, f64);
    let mut s = 109u64;
    let rows: Vec<Row> = (0..160)
        .map(|i| {
            let x = [lcg(&mut s), 1.0 + lcg(&mut s)];
            let y0 = 0.5 + x[0] - 2.0 * x[1] + 0.2 * lcg(&mut s);
            let y1 = -x[0] + 0.7 * x[1] + 0.3 * lcg(&mut s);
            let ys = [
                match i {
                    0 => Some(0.0),
                    _ if i % 11 == 7 => None,
                    _ => Some(y0),
                },
                (i >= 3 && i % 5 != 2).then_some(y1),
            ];
            let d = match i {
                0 => 0.0,
                3 => 5.0,
                _ => 1.0,
            };
            let w = if i == 2 { 0.0 } else { 1.5 + lcg(&mut s) };
            (x, ys, d, w)
        })
        .collect();
    // Predictions, the filter after the stream, and how many target-rows
    // the filter learned from before that target had a residual variance.
    let run = |base: &KalmanCfg, c: f64| {
        let mut cfg = base.clone();
        cfg.obs_var = cfg.obs_var.map(|v| v * c * c);
        cfg.q = cfg.q.map(|q| q.iter().map(|v| v * c * c).collect());
        let mut m = Kalman::new(cfg).unwrap();
        let mut before_sigma = 0;
        let mut preds = Vec::new();
        for (x, ys, d, w) in &rows {
            let ys: Vec<Option<f64>> = ys.iter().map(|y| y.map(|v| v * c)).collect();
            for (j, y) in ys.iter().enumerate() {
                let s2 = if m.cfg.share_p {
                    m.sig2.iter().sum::<f64>()
                } else {
                    m.sig2[j]
                };
                if y.is_some() && *w > 0.0 && s2 == 0.0 {
                    before_sigma += 1;
                }
            }
            preds.push(m.step(x, &ys, *d, *w).pred);
        }
        (preds, m, before_sigma)
    };
    let mut cases = Vec::new();
    for share in [false, true] {
        for standardize in [true, false] {
            let mut c = cfg(2, 2, vec![40.0]);
            c.decay = Decay::Halflife(30.0);
            c.min_weight = 4.0;
            c.share_p = share;
            c.standardize = standardize;
            cases.push((format!("share {share}, standardize {standardize}"), c));
        }
    }
    let mut c = cfg(2, 2, vec![40.0]);
    c.min_weight = 4.0;
    c.revert_half_life = vec![f64::INFINITY, 20.0, 6.0];
    cases.push(("reverting".into(), c.clone()));
    c.q = Some(vec![0.0, 1e-3, 2e-3]);
    cases.push(("reverting, q given".into(), c));
    let mut c = cfg(2, 2, vec![40.0]);
    c.min_weight = 4.0;
    c.p0 = 0.25;
    cases.push(("p0 = 1/4".into(), c.clone()));
    c.obs_var = Some(0.25);
    cases.push(("p0 = 1/4, obs_var given".into(), c));
    // And each in units of the noise, the form the bank builds (task 278).
    let units: Vec<(String, KalmanCfg)> = cases
        .iter()
        .map(|(name, c)| {
            let mut c = c.clone();
            c.noise_units = true;
            (format!("{name}, in units of the noise"), c)
        })
        .collect();
    cases.extend(units);
    // And each with `min_weight` holding learning back, as a state
    // written before schema 64 does (task 278, D10): the noise waits
    // for the first prediction the gate lets out, so the warm-up's
    // noise, the row's own innovation, runs for longer.
    let held: Vec<(String, KalmanCfg)> = cases
        .iter()
        .map(|(name, c)| {
            let mut c = c.clone();
            c.min_weight_holds_learning = true;
            (format!("{name}, the gate holding learning back"), c)
        })
        .collect();
    cases.extend(held);

    for (name, base) in &cases {
        let (want, m1, learned) = run(base, 1.0);
        // The rows before a residual variance: the filter's own sizing
        // rows, and under the old gate the rows it withheld.
        let warm = if base.min_weight_holds_learning { 4 } else { 1 };
        assert!(learned >= warm, "{name}: the warm-up was learned from");
        let first = (0..want.len()).find(|&i| want[i][0].is_finite()).unwrap();
        assert!(first > 3, "{name}: the first prediction follows the gap");
        for c in [2f64.powi(20), 2f64.powi(-20)] {
            let (got, mc, _) = run(base, c);
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                for j in 0..2 {
                    let scaled = w[j] * c;
                    assert!(
                        g[j].to_bits() == scaled.to_bits() || (g[j].is_nan() && w[j].is_nan()),
                        "{name}, c = {c:e}, row {i}, target {j}: {} against {} \
                         ({:.2e} apart, relative)",
                        g[j],
                        scaled,
                        ((g[j] - scaled) / scaled).abs()
                    );
                }
            }
            for j in 0..2 {
                assert_eq!(
                    mc.sig2[j].to_bits(),
                    (m1.sig2[j] * c * c).to_bits(),
                    "{name}"
                );
                for (a, b) in mc.coefficients()[j].iter().zip(&m1.coefficients()[j]) {
                    assert_eq!(a.to_bits(), (b * c).to_bits(), "{name}");
                }
            }
        }
    }
}

/// One feature through the origin, unstandardized, at `min_weight` 0:
/// the filter the tests below start from.
fn plain(p0: f64, half_life: f64, obs_var: Option<f64>) -> KalmanCfg {
    KalmanCfg {
        horizons: Vec::new(),
        min_weight_holds_learning: false,
        noise_units: false,
        n_features: 1,
        n_targets: 1,
        fit_intercept: false,
        decay: Decay::Halflife(50.0),
        half_life: vec![half_life],
        q: None,
        obs_var,
        p0,
        share_p: false,
        min_weight: 0.0,
        revert_half_life: vec![f64::INFINITY],
        standardize: false,
    }
}

/// A covariance is unsized only when its whole diagonal is 0 (the module
/// doc, CC4). A slot the reversion takes to exactly 0 across a step,
/// beside one that still holds its variance, leaves `P` sized: the row
/// adds its process noise to it (none here), as to any sized `P`, and
/// the variance the other slot has learned carries on. Sized afresh at
/// `p0` times the noise, `P` would forget it. Slot 0 reverts at a
/// half-life of `1e-4`, so a step of 1 takes it and its covariances to
/// exactly 0 (`2^-10000`); slot 1 is a random walk with no process
/// noise. From the second row on slot 1 alone learns, and its variance
/// is a one-slot filter's, the oracle in information form: `1 / P11` is
/// the first row's marginal, `(1 + z0² + z1²) / (1 + z0²)` at
/// `P_0 = I`, plus `x1²` a row at `R = 1`.
#[test]
fn a_slot_reverted_to_zero_leaves_the_covariance_sized() {
    let mut c = plain(1.0, f64::INFINITY, Some(1.0));
    c.n_features = 2;
    c.q = Some(vec![0.0, 0.0]);
    c.revert_half_life = vec![1e-4, f64::INFINITY];
    let mut m = Kalman::new(c).unwrap();
    let z = [1.0, 2.0];
    m.step(&z, &[Some(0.5)], 0.0, 1.0);
    let mut info = (1.0 + z[0] * z[0] + z[1] * z[1]) / (1.0 + z[0] * z[0]);
    let mut s = 113u64;
    for i in 1..40 {
        let x = [lcg(&mut s), 0.5 + lcg(&mut s)];
        m.step(&x, &[Some(x[0] - 2.0 * x[1])], 1.0, 1.0);
        info += x[1] * x[1];
        let p = &m.p[0];
        assert_eq!(p[0], 0.0, "row {i}: slot 0 holds nothing");
        assert!(
            (p[3] * info - 1.0).abs() <= 1e-12,
            "row {i}: P11 = {} against {}",
            p[3],
            1.0 / info
        );
    }
}

/// A first noise so large that `p0` times it overflows sizes no prior:
/// `P` stays unsized, all zero and finite, where an infinite diagonal
/// would make every later innovation variance infinite, refuse every
/// correction and end the filter's learning for good. The next row
/// whose noise sizes a finite prior sizes it, and the filter learns
/// from it. A target at the input bound is within the model's contract,
/// as is any `p0 > 0`: here `p0 e² = 1e200 · 1e200`.
#[test]
fn a_prior_that_would_overflow_is_not_sized() {
    let mut m = Kalman::new(plain(1e200, f64::INFINITY, None)).unwrap();
    m.step(&[1.0], &[Some(crate::INPUT_BOUND)], 0.0, 1.0);
    assert_eq!(m.p[0], vec![0.0], "unsized, and finite");
    assert_eq!(m.beta[0], vec![0.0]);
    // An innovation of 3 sizes a finite prior, `p0 · 9`, and corrects.
    m.step(&[1.0], &[Some(3.0)], 1.0, 1.0);
    assert!(m.p[0].iter().all(|v| v.is_finite()), "{:?}", m.p[0]);
    assert_eq!(m.beta[0], vec![3.0], "learned");
}

/// Once `P` is sized, a row whose innovation is exactly 0 before the
/// target has a residual variance still sizes no noise, and corrects
/// nothing: at `R = 0` the gain would be `P z / zᵀ P z`, taking the row
/// as exact and collapsing `P` along `z`, here to 0, so the filter would
/// hold its fit with no doubt left (CC4, the module doc). `P` and `b`
/// keep their bits. The row before sized `P` at `p0 e² = 1` and
/// narrowed it to 1/2; the process noise derived from `σ² = 0` adds
/// nothing.
#[test]
fn an_exact_row_after_the_prior_is_sized_corrects_nothing() {
    let mut m = Kalman::new(plain(1.0, 40.0, None)).unwrap();
    m.step(&[1.0], &[Some(1.0)], 0.0, 1.0);
    assert_eq!((m.p[0][0], m.beta[0][0]), (0.5, 0.5), "sized, narrowed");
    assert_eq!(m.sig2, vec![0.0], "no residual variance yet");
    let pred = m.step(&[1.0], &[Some(0.5)], 1.0, 1.0).pred[0];
    assert_eq!(pred, 0.5, "the row is its prediction exactly");
    assert_eq!((m.p[0][0], m.beta[0][0]), (0.5, 0.5));
    assert_eq!(m.sig2, vec![0.0], "every residual so far exactly 0");
}

// ---- readiness (docs/PLAN.md task 116) ----

/// A standard normal, near enough: twelve uniforms, centred.
fn normal(s: &mut u64) -> f64 {
    (0..12).map(|_| 0.5 * (lcg(s) + 1.0)).sum::<f64>() - 6.0
}

/// `error_inflation` is `sqrt(1 + z' P⁻ z / R)`, the prior predictive
/// variance over the noise, with `P⁻` written here from the module
/// doc's recursion -- `b <- Phi b`, `P <- Phi P Phi` on every row, `P <-
/// P + Q D²` with `D` the clock since the last observation on a row that
/// observes the target, the gain `P z / (z' P z + R / w)`, `P <- P − g
/// z' P` -- in its own loops, on a
/// reverting slot, irregular steps, a zero-weight row and a null
/// target, with a fixed `obs_var` so `R` is known. The gate reads the
/// same per-row value.
#[test]
fn the_row_statistic_is_the_prior_predictive_variance_over_the_noise() {
    let (r, coef_hl) = (0.3, 40.0);
    let mut c = plain(2.0, coef_hl, Some(r));
    c.n_features = 2;
    c.fit_intercept = true;
    c.revert_half_life = vec![f64::INFINITY, 25.0, 60.0];
    let k = 3;
    let mut m = Kalman::new(c).unwrap();
    let mut b = vec![0.0; k];
    let mut p = vec![0.0; k * k];
    for i in 0..k {
        p[i * k + i] = 2.0 * r;
    }
    let q = r * (std::f64::consts::LN_2 / coef_hl).powi(2);
    let revert = [f64::INFINITY, 25.0, 60.0];
    let mut s = 17u64;
    let mut checked = 0;
    // The clock since the last observation (task 211).
    let mut elapsed = 0.0;
    for i in 0..300 {
        let x = [3.0 * lcg(&mut s), 1.0 + lcg(&mut s)];
        let d = match i {
            0 => 0.0,
            _ if i % 7 == 3 => 4.0,
            _ => 1.0,
        };
        let w = if i % 11 == 5 { 0.0 } else { 1.0 };
        let y = (i % 13 != 8).then(|| 0.5 + x[0] - 2.0 * x[1] + 0.4 * lcg(&mut s));
        let z = [1.0, x[0], x[1]];
        let phi: Vec<f64> = revert
            .iter()
            .map(|h| Decay::Halflife(*h).factor(d))
            .collect();
        for (bi, ph) in b.iter_mut().zip(&phi) {
            *bi *= ph;
        }
        for a in 0..k {
            for e in 0..k {
                p[a * k + e] *= phi[a] * phi[e];
            }
        }
        // The prior the row's update would start from: `Q (D + d)²` for
        // the clock since the last observation, this row's included,
        // charged to `P` only if the row observes the target; on a
        // reverting slot `(D + d)²` is `((1 − 2^(−(D + d)/r)) / θ)²`,
        // `θ = ln 2 / r` (task 214).
        elapsed += d;
        let mut prior = p.clone();
        for a in 0..k {
            let gap2 = if revert[a].is_infinite() {
                elapsed * elapsed
            } else {
                let theta = std::f64::consts::LN_2 / revert[a];
                ((1.0 - Decay::Halflife(revert[a]).factor(elapsed)) / theta).powi(2)
            };
            prior[a * k + a] += q * gap2;
        }
        let pz: Vec<f64> = (0..k)
            .map(|a| (0..k).map(|e| prior[a * k + e] * z[e]).sum())
            .collect();
        let zpz: f64 = z.iter().zip(&pz).map(|(a, e)| a * e).sum();
        let want = (1.0 + zpz / r).sqrt();
        let (mut row, mut gate) = (Vec::new(), Vec::new());
        assert!(m.row_error_inflation_into(&x, d, &mut row));
        assert!(m.error_inflation_gate_into(&x, d, &mut gate, 1.1));
        assert!(
            (row[0] - want).abs() <= 1e-12 * want,
            "row {i}: {} vs {want}",
            row[0]
        );
        assert_eq!(row[0].to_bits(), gate[0].to_bits(), "row {i}");
        checked += 1;
        m.step(&x, &[y], d, w);
        if let (Some(y), true) = (y, w > 0.0) {
            p = prior;
            elapsed = 0.0;
            let s_inn = zpz + r / w;
            let err = y - z.iter().zip(&b).map(|(a, e)| a * e).sum::<f64>();
            let g: Vec<f64> = pz.iter().map(|v| v / s_inn).collect();
            for a in 0..k {
                b[a] += g[a] * err;
                for e in 0..k {
                    p[a * k + e] -= g[a] * pz[e];
                }
            }
        }
    }
    assert_eq!(checked, 300);
}

/// The noise is the state's, never the row's own innovation, which
/// reads its target (hard rule 2; the bank's test perturbs the target):
/// without `obs_var` the ratio is infinite while `P` is unsized and
/// until the target has a residual variance.
#[test]
fn the_row_statistic_is_infinite_until_there_is_a_noise() {
    let mut m = Kalman::new(cfg(2, 1, vec![50.0])).unwrap();
    let mut out = Vec::new();
    let mut s = 5u64;
    let mut finite_from = None;
    for i in 0..40 {
        let x = [lcg(&mut s), lcg(&mut s)];
        let y = 1.0 + x[0] + 0.2 * lcg(&mut s);
        m.row_error_inflation_into(&x, 1.0, &mut out);
        let unknown = m.is_unsized(0) || m.sig2[0] <= 0.0;
        assert_eq!(out[0].is_infinite(), unknown, "row {i}");
        if !unknown {
            finite_from.get_or_insert(i);
        }
        m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
    }
    // The noise waits for the first residual, which the filter's own
    // prediction gives once its covariance is sized: before `min_weight`
    // 10 lets a prediction out, which withholds the output alone (task
    // 278, D10).
    assert!(
        finite_from.is_some_and(|i| (3..10).contains(&i)),
        "{finite_from:?}"
    );
}

/// Under `share_p` every target reads the one `P` and the mean noise:
/// one value for all of them.
#[test]
fn share_p_reads_one_statistic_for_every_target() {
    let mut c = cfg(2, 3, vec![50.0]);
    c.share_p = true;
    c.min_weight = 0.0;
    let m = fit(c, 120, 9);
    let mut out = Vec::new();
    assert!(m.row_error_inflation_into(&[0.3, 0.1], 1.0, &mut out));
    assert_eq!(out.len(), 3);
    assert!(out[0].is_finite() && out[0] > 1.0, "{out:?}");
    assert!(
        out.iter().all(|v| v.to_bits() == out[0].to_bits()),
        "{out:?}"
    );
}

/// `se_coef`'s variances are `T P Tᵀ`'s diagonal: through `coef`'s own
/// read-out, a prediction's variance at a row is the quadratic form of
/// `P` in its standardized coordinates, so at the features' origin it
/// is the intercept's variance, and a slope's is `P_ii / s_i²`.
/// Unstandardized, `T` is the identity and they are `P`'s diagonal.
#[test]
fn the_coefficient_variances_are_the_posterior_in_coefs_units() {
    let mut c = cfg(2, 1, vec![50.0]);
    c.min_weight = 0.0;
    let m = fit(c.clone(), 200, 3);
    let Some(crate::CoefVariance::Absolute(v)) = m.coef_variance() else {
        panic!("kalman's variances are absolute");
    };
    let k = 3;
    // The intercept is the prediction at `x = 0`, read in the anchor's
    // coordinates `P` is held in; a slope is `b_i / s^a_i`.
    let z0 = m.anchored(&[0.0, 0.0]);
    let quad: f64 = (0..k)
        .map(|a| {
            (0..k)
                .map(|e| z0[a] * m.p[0][a * k + e] * z0[e])
                .sum::<f64>()
        })
        .sum();
    assert!(
        (v[0][0] - quad).abs() <= 1e-12 * quad,
        "{} vs {quad}",
        v[0][0]
    );
    let s = &m.anchor_scale;
    for i in 1..k {
        let want = m.p[0][i * k + i] / (s[i] * s[i]);
        assert!((v[0][i] - want).abs() <= 1e-12 * want, "{i}");
    }
    c.standardize = false;
    let m = fit(c, 200, 3);
    let Some(crate::CoefVariance::Absolute(v)) = m.coef_variance() else {
        panic!()
    };
    let diag: Vec<f64> = (0..k).map(|i| m.p[0][i * k + i]).collect();
    assert_eq!(v[0], diag);
    // Unsized, nothing is estimated.
    let fresh = Kalman::new(cfg(2, 1, vec![50.0])).unwrap();
    let Some(crate::CoefVariance::Absolute(v)) = fresh.coef_variance() else {
        panic!()
    };
    assert!(v[0].iter().all(|x| x.is_nan()));
}

/// The mean of the per-row `h` over rows 5,000 to 20,000, the mean of
/// the summary's mean field over the same rows, and the doc's `p`, for
/// `n_feat` standardized, uncorrelated features and a constant truth
/// at `coef_half_life`, rows one clock unit apart.
fn settled_readiness(n_feat: usize, coef_hl: f64) -> (f64, f64, f64) {
    let c = KalmanCfg {
        horizons: Vec::new(),
        min_weight_holds_learning: false,
        noise_units: false,
        n_features: n_feat,
        n_targets: 1,
        fit_intercept: true,
        decay: Decay::Halflife(200.0),
        half_life: vec![coef_hl],
        q: None,
        obs_var: None,
        p0: 1.0,
        share_p: false,
        min_weight: 0.0,
        revert_half_life: vec![f64::INFINITY],
        standardize: true,
    };
    let mut m = Kalman::new(c).unwrap();
    let mut s = 29u64;
    let (mut sum_h, mut sum_field, mut rows) = (0.0, 0.0, 0.0);
    let mut out = Vec::new();
    for i in 0..20_000 {
        let x: Vec<f64> = (0..n_feat).map(|_| normal(&mut s)).collect();
        let truth: f64 = x
            .iter()
            .enumerate()
            .map(|(j, v)| (j as f64 - 4.0) * v)
            .sum();
        let y = 1.0 + truth + normal(&mut s);
        let d = if i == 0 { 0.0 } else { 1.0 };
        if i >= 5_000 {
            m.row_error_inflation_into(&x, d, &mut out);
            sum_h += out[0] * out[0] - 1.0;
            m.error_inflation_into(&mut out);
            sum_field += out[0] * out[0] - 1.0;
            rows += 1.0;
        }
        m.step(&x, &[Some(y)], d, 1.0);
    }
    let c = std::f64::consts::LN_2 / coef_hl;
    let p = (c * c + (c.powi(4) + 4.0 * c * c).sqrt()) / 2.0;
    (sum_h / rows, sum_field / rows, p)
}

/// The floor the module doc derives: under process noise `P⁻` never
/// reaches 0, and per direction `p = P⁻ / R` settles where
/// `p² / (p + 1) = c²`, `c = ln 2 · d / coef_half_life`, so the average
/// row reads about `sqrt(1 + k p)` -- 1.0675 at `k = 10`, `d = 1`,
/// `coef_half_life = 50`. Measured (task 116, 2026-10-07) on
/// standardized, uncorrelated features: the mean of `h` is 6.0% above
/// `k p` there (`sqrt(1 + h)` 1.0715), and the summary's mean field
/// 5.3% above `k p / (p + 1)`, its value at the posterior. The mean
/// field is a slow-drift reading: at `coef_half_life = 5` the mean of
/// `h` is 1.60 times `k p` (`sqrt(1 + h)` 1.84 against the formula's
/// 1.58), each row's rank-one update narrowing one direction where the
/// mean field spreads it over all of them.
#[test]
fn the_average_row_settles_near_the_floor_the_doc_derives() {
    let k = 10.0;
    let (mean_h, field, p) = settled_readiness(9, 50.0);
    let floor = k * p;
    assert!(((1.0 + floor).sqrt() - 1.0675).abs() < 1e-4, "{floor}");
    assert!(
        mean_h > floor && mean_h < 1.08 * floor,
        "mean h {mean_h} vs k p {floor}"
    );
    let posterior = floor / (p + 1.0);
    assert!(
        field > posterior && field < 1.08 * posterior,
        "mean field {field} vs k p / (p + 1) {posterior}"
    );
    let (fast, _, p5) = settled_readiness(9, 5.0);
    let ratio = fast / (k * p5);
    assert!(ratio > 1.5 && ratio < 1.7, "{ratio}");
}

/// Review round 5, A1: under `share_p` the noise is the mean `σ²` over
/// the targets that have a residual variance, so a target that has none
/// yet is no noise estimate rather than a noise of 0. A target beside a
/// copy of itself that is null for its first 150 rows predicts as it
/// would alone, to the bit, through the row after the copy gains a
/// residual variance of its own (the noise is read before the row's
/// updates), and `pred_var` with it; from there the mean takes the
/// copy's variance in. The mean over all the targets halved the noise
/// while the copy was null: `pred_var` parted at row 4, 0.181 against
/// 0.208 alone, and the probe through the bank moved the target's
/// predictions by up to 0.45.
#[test]
fn share_p_a_copy_null_for_its_first_rows_moves_nothing_until_it_has_a_noise() {
    let mk = |m: usize| {
        let mut c = cfg(2, m, vec![50.0]);
        c.share_p = true;
        c.min_weight = 3.0;
        Kalman::new(c).unwrap()
    };
    let (mut alone, mut pair) = (mk(1), mk(2));
    let mut s = 7u64;
    let mut noised: Option<usize> = None;
    let mut parted = false;
    for i in 0..400usize {
        let x = [lcg(&mut s), lcg(&mut s)];
        let y = 1.0 - 2.0 * x[0] + x[1] + 0.3 * lcg(&mut s);
        let d = if i == 0 { 0.0 } else { 1.0 };
        let probe = [0.2, -0.4];
        let a = alone.step(&x, &[Some(y)], d, 1.0).pred[0];
        let b = pair
            .step(&x, &[Some(y), (i >= 150).then_some(y)], d, 1.0)
            .pred[0];
        if noised.is_none_or(|r| i <= r + 1) {
            assert_eq!(a.to_bits(), b.to_bits(), "row {i}: {a} against {b}");
        } else if a != b {
            parted = true;
        }
        // `pred_var` reads the state after the row: equal while the
        // copy has no residual variance in it.
        if pair.sigma2()[1] == 0.0 {
            let (va, vb) = (alone.pred_var(&probe)[0], pair.pred_var(&probe)[0]);
            assert_eq!(
                va.to_bits(),
                vb.to_bits(),
                "row {i}: pred_var {va} against {vb}"
            );
        }
        if noised.is_none() && pair.sigma2()[1] > 0.0 {
            noised = Some(i);
        }
    }
    assert_eq!(noised, Some(151), "the copy's first residual");
    assert!(parted, "the copy's own variance enters the mean");
}

// Task 218's tests, every readout against the recursion, in a file of
// their own: `kalman.rs` stands near the source-size cap
// (`tests/test_repo_hygiene.py`).
mod readouts;
// Task 250's: the per-row paths that run faster, each against the form
// it replaced, to the bit, and shown to run.
mod per_row;
// The filter against its recursion written by hand, in both forms.
mod recursion;
// Task 278's: `P` in units of the noise, each against the form before.
mod noise_units;
// Task 281 (3)'s: the covariance's update forms at a huge prior.
mod cov_forms;
// Review C1's follow-up: the moments keep the row's weight of history
// after a gap of more than a half-life.
mod gap_moments;
mod horizons;
// The 2026-11-23 review's findings (PLAN 578f).
mod review;
// Task 496's fixed-lag smoother against the whole-prefix smoother.
mod smooth;
