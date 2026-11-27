use super::*;
use crate::Extra;

fn base_cfg(k: usize, m: usize) -> KalmanCfg {
    KalmanCfg {
        n_features: k,
        n_targets: m,
        fit_intercept: true,
        decay: Decay::Halflife(50.0),
        half_life: vec![f64::INFINITY],
        q: Some(vec![0.0; k + 1]),
        obs_var: Some(0.25),
        p0: 2.0,
        share_p: false,
        min_weight: 0.0,
        revert_half_life: vec![f64::INFINITY],
        standardize: false,
        min_weight_holds_learning: false,
        noise_units: true,
        horizons: Vec::new(),
        obs: KalmanObs::default(),
        coef_prior: None,
        emit_loglik: false,
        emit_mahal: false,
    }
}

fn lcg_val(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

#[test]
fn cfg_validation_checks_coef_prior_and_nonlinear_innovation_refusals() {
    // Wrong target count, wrong feature count, and non-finite entries.
    let mut bad_m = base_cfg(2, 2);
    bad_m.coef_prior = Some(vec![vec![0.0, 1.0, 2.0]]);
    assert!(bad_m.validate().unwrap_err().contains("coef_prior"));

    let mut bad_k = base_cfg(2, 1);
    bad_k.coef_prior = Some(vec![vec![1.0, 2.0]]);
    assert!(bad_k.validate().unwrap_err().contains("coef_prior"));

    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut c = base_cfg(2, 1);
        c.coef_prior = Some(vec![vec![0.0, bad, 1.0]]);
        let e = c.validate().unwrap_err();
        assert!(e.contains("coef_prior") && e.contains("finite"), "{e}");
    }

    // Valid coef_prior on both linear and nonlinear (EKF / UKF).
    let mut lin = base_cfg(2, 1);
    lin.coef_prior = Some(vec![vec![1.5, -2.0, 0.5]]);
    lin.emit_loglik = true;
    lin.emit_mahal = true;
    lin.validate().unwrap();

    for filter in [KalmanFilter::Ekf, KalmanFilter::Ukf] {
        let mut nl = base_cfg(2, 1);
        nl.obs = KalmanObs {
            family: KalmanFamily::Poisson,
            link: KalmanLink::Log,
            filter,
            ..KalmanObs::default()
        };
        nl.obs_var = None;
        nl.half_life = Vec::new();
        nl.q = Some(vec![1e-4; 3]);
        nl.coef_prior = Some(vec![vec![0.2, -0.3, 0.4]]);
        nl.validate().unwrap();

        let mut bad_ll = nl.clone();
        bad_ll.emit_loglik = true;
        assert!(bad_ll.validate().unwrap_err().contains("emit_loglik"));

        let mut bad_mh = nl.clone();
        bad_mh.emit_mahal = true;
        assert!(bad_mh.validate().unwrap_err().contains("emit_mahal"));
    }

    // Also refused on Gaussian with identity link when filter != Linear.
    let mut ekf_lin = base_cfg(2, 1);
    ekf_lin.obs.filter = KalmanFilter::Ekf;
    ekf_lin.emit_loglik = true;
    ekf_lin.emit_mahal = true;
    let err = ekf_lin.validate().unwrap_err();
    assert!(
        err.contains("emit_loglik") && err.contains("emit_mahal"),
        "{err}"
    );
}

/// With `q = 0`, fixed `obs_var`, `standardize = false`, and `coef_prior = b0`,
/// the Kalman filter posterior is exact ridge regression toward `b0` with
/// penalty `1 / p0`: `(Z' W Z + (1/p0) I) b = Z' W y + (1/p0) b0`.
#[test]
fn coef_prior_matches_closed_form_ridge_oracle_with_obs_var() {
    let prior = vec![vec![1.25, -2.5, 0.75], vec![-0.5, 3.0, -1.25]];
    let p0 = 1.5;
    let mut c = base_cfg(2, 2);
    c.p0 = p0;
    c.coef_prior = Some(prior.clone());
    let mut m = Kalman::new(c).unwrap();

    let mut s = 42u64;
    let k = 3;
    let mut gram = vec![0.0; k * k];
    let mut rhs = vec![vec![0.0; k]; 2];
    for i in 0..k {
        gram[i * k + i] = 1.0 / p0;
        for j in 0..2 {
            rhs[j][i] = prior[j][i] / p0;
        }
    }
    for row in 0..40 {
        let x = [lcg_val(&mut s) * 2.0, lcg_val(&mut s) * 1.5];
        let w = 0.5 + lcg_val(&mut s).abs();
        let y0 = 0.3 + 1.1 * x[0] - 0.8 * x[1] + 0.2 * lcg_val(&mut s);
        let y1 = -1.0 + 0.4 * x[0] + 2.2 * x[1] + 0.3 * lcg_val(&mut s);
        let z = [1.0, x[0], x[1]];
        for a in 0..k {
            for b in 0..k {
                gram[a * k + b] += w * z[a] * z[b];
            }
            rhs[0][a] += w * z[a] * y0;
            rhs[1][a] += w * z[a] * y1;
        }
        m.step(
            &x,
            &[Some(y0), Some(y1)],
            if row == 0 { 0.0 } else { 1.0 },
            w,
        );
    }

    let got = m.coefficients();
    for j in 0..2 {
        let want = crate::oracle::solve(&gram, &rhs[j]);
        for i in 0..k {
            assert!(
                (got[j][i] - want[i]).abs() < 1e-12,
                "target {j} slot {i}: got {}, want {want:?}",
                got[j][i]
            );
        }
    }
}

/// Without `obs_var` (`noise_units = true`), the intercept prior is diffuse
/// (the intercept entry of `coef_prior` is not read) and each slope has ridge
/// `1 / p0` toward `coef_prior[j][i]`.
#[test]
fn coef_prior_with_diffuse_intercept_ignores_intercept_entry_and_matches_ridge_slopes() {
    let mut c1 = base_cfg(2, 1);
    c1.obs_var = None;
    c1.coef_prior = Some(vec![vec![0.0, 2.0, -1.5]]);

    let mut c2 = c1.clone();
    c2.coef_prior = Some(vec![vec![1e5, 2.0, -1.5]]);

    let mut m1 = Kalman::new(c1).unwrap();
    let mut m2 = Kalman::new(c2).unwrap();

    let k = 3;
    let p0 = m1.cfg().p0;
    let mut gram = vec![0.0; k * k];
    let mut rhs = vec![0.0; k];
    // Ridge on slopes only, no penalty on intercept.
    gram[k + 1] = 1.0 / p0;
    gram[2 * k + 2] = 1.0 / p0;
    rhs[1] = 2.0 / p0;
    rhs[2] = -1.5 / p0;

    let mut s = 99u64;
    for row in 0..35 {
        let x = [lcg_val(&mut s), 0.5 + lcg_val(&mut s)];
        let y = 1000.0 + 0.5 * x[0] + 0.25 * x[1] + 0.1 * lcg_val(&mut s);
        let z = [1.0, x[0], x[1]];
        for a in 0..k {
            for b in 0..k {
                gram[a * k + b] += z[a] * z[b];
            }
            rhs[a] += z[a] * y;
        }
        let s1 = m1.step(&x, &[Some(y)], if row == 0 { 0.0 } else { 1.0 }, 1.0);
        let s2 = m2.step(&x, &[Some(y)], if row == 0 { 0.0 } else { 1.0 }, 1.0);
        assert_eq!(s1.pred[0].to_bits(), s2.pred[0].to_bits(), "row {row}");
    }
    let want = crate::oracle::solve(&gram, &rhs);
    let got = &m1.coefficients()[0];
    for i in 0..k {
        assert!(
            (got[i] - want[i]).abs() < 1e-10 * (1.0 + want[i].abs()),
            "slot {i}: got {}, want {}",
            got[i],
            want[i]
        );
    }
}

/// Under `standardize = true` with diffuse intercept, shifting features by
/// `1e8` preserves predictions and slopes to rounding when `coef_prior` is set.
#[test]
fn coef_prior_under_standardize_is_shift_invariant_and_seeds_original_units() {
    let run = |shift: f64| -> (Vec<f64>, Vec<f64>) {
        let mut c = base_cfg(2, 1);
        c.obs_var = None;
        c.standardize = true;
        c.p0 = 1e-4; // tight prior so coef_prior visibly anchors slopes
        c.coef_prior = Some(vec![vec![0.0, 2.5, -1.25]]);
        let mut m = Kalman::new(c).unwrap();
        let mut s = 777u64;
        let mut preds = Vec::new();
        for row in 0..30 {
            let x0 = lcg_val(&mut s) * 2.0 + shift;
            let x1 = lcg_val(&mut s) * 0.5 - shift;
            let y = 3.0 + 2.5 * (x0 - shift) - 1.25 * (x1 + shift) + 0.05 * lcg_val(&mut s);
            let st = m.step(&[x0, x1], &[Some(y)], if row == 0 { 0.0 } else { 1.0 }, 1.0);
            preds.push(st.pred[0]);
        }
        (preds, m.coefficients()[0].clone())
    };
    let (p0, b0) = run(0.0);
    let (p8, b8) = run(1e8);
    // Tight prior keeps slopes near [2.5, -1.25] in original units.
    assert!((b0[1] - 2.5).abs() < 0.05, "slope0: {}", b0[1]);
    assert!((b0[2] + 1.25).abs() < 0.05, "slope1: {}", b0[2]);
    assert!(
        (b8[1] - b0[1]).abs() < 1e-6,
        "shift slope0: {} vs {}",
        b8[1],
        b0[1]
    );
    assert!(
        (b8[2] - b0[2]).abs() < 1e-6,
        "shift slope1: {} vs {}",
        b8[2],
        b0[2]
    );
    for i in 3..30 {
        assert!(
            (p8[i] - p0[i]).abs() < 1e-5,
            "row {i}: shifted {} vs unshifted {}",
            p8[i],
            p0[i]
        );
    }
}

/// `revert_half_life` pulls `beta` toward 0 (not `coef_prior`), matching
/// `b <- Phi b` from `b_0 = coef_prior`.
#[test]
fn revert_half_life_pulls_toward_zero_from_coef_prior() {
    let mut c = base_cfg(1, 1);
    c.fit_intercept = false;
    c.q = Some(vec![0.0]);
    c.revert_half_life = vec![10.0];
    c.coef_prior = Some(vec![vec![4.0]]);
    let mut m = Kalman::new(c).unwrap();
    // Advance 10 clock units with no target observed: beta should halve to 2.0.
    m.step(&[1.0], &[None], 10.0, 1.0);
    let b = m.coefficients()[0][0];
    assert!((b - 2.0).abs() < 1e-14, "expected 2.0, got {b}");
}

/// Innovation `loglik` and `mahal` match hand-computed Gaussian formulas from
/// row 0 under `obs_var`, agree between `Covariance` and `SquareRoot` forms,
/// are withheld on zero-weight/missing/diffuse/pre-noise rows, and
/// `predict_with_weight` matches `step` bit-for-bit across varied weights.
#[test]
fn emit_loglik_and_mahal_match_exact_formula_and_predict_with_weight() {
    for form in [CovarianceForm::Covariance, CovarianceForm::SquareRoot] {
        for standardize in [false, true] {
            for share_p in [false, true] {
                for obs_var in [Some(0.36), None] {
                    let mut c = base_cfg(2, 2);
                    c.obs_var = obs_var;
                    c.standardize = standardize;
                    c.share_p = share_p;
                    c.q = None;
                    c.half_life = vec![30.0];
                    c.revert_half_life = vec![f64::INFINITY, 80.0, 40.0];
                    c.coef_prior = Some(vec![vec![0.5, 1.0, -0.5], vec![-0.25, 0.3, 0.8]]);
                    c.emit_loglik = true;
                    c.emit_mahal = true;
                    let mut m = Kalman::new(c).unwrap().with_covariance_form(form).unwrap();
                    let mut twin = m.clone();
                    let mut reused = Step::default();
                    let mut s = 12345u64;
                    let mut finite_count = 0usize;

                    for row in 0..45 {
                        let x = [lcg_val(&mut s) * 2.0, lcg_val(&mut s) * 1.5];
                        let y = [
                            (row % 7 != 2)
                                .then(|| 0.5 + 1.2 * x[0] - 0.7 * x[1] + 0.3 * lcg_val(&mut s)),
                            (row % 5 != 3)
                                .then(|| -0.3 + 0.5 * x[0] + 0.9 * x[1] + 0.4 * lcg_val(&mut s)),
                        ];
                        let w = match row % 9 {
                            4 => 0.0,
                            2 => 0.5,
                            6 => 2.5,
                            _ => 1.0,
                        };
                        let d = if row == 0 {
                            0.0
                        } else {
                            1.0 + (row % 3) as f64
                        };
                        let p = m.predict_with_weight(&x, &y, d, w);
                        let st = m.step(&x, &y, d, w);
                        twin.step_into(&x, &y, d, w, &mut reused);

                        let (
                            Some(Extra::Kalman { values: pv }),
                            Some(Extra::Kalman { values: sv }),
                        ) = (&p.extra, &st.extra)
                        else {
                            panic!("expected Extra::Kalman");
                        };
                        let Some(Extra::Kalman { values: rv }) = &reused.extra else {
                            panic!("expected reused Extra::Kalman");
                        };
                        assert_eq!(sv.len(), 4);
                        for idx in 0..4 {
                            let same = |a: f64, b: f64| {
                                a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
                            };
                            assert!(
                                same(pv[idx], sv[idx]),
                                "{form:?} std={standardize} share={share_p} ov={obs_var:?} row={row} idx={idx}: predict {} vs step {}",
                                pv[idx],
                                sv[idx]
                            );
                            assert!(same(rv[idx], sv[idx]));
                        }
                        if w == 0.0 {
                            assert!(
                                sv.iter().all(|v| v.is_nan()),
                                "row {row}: zero-weight must be NaN"
                            );
                        }
                        for j in 0..2 {
                            if y[j].is_none() {
                                assert!(sv[j].is_nan() && sv[2 + j].is_nan());
                            }
                            if sv[j].is_finite() {
                                finite_count += 1;
                                assert!(sv[2 + j].is_finite() && sv[2 + j] >= 0.0);
                            }
                        }
                    }
                    assert!(
                        finite_count >= 50,
                        "too few finite diagnostics: {finite_count}"
                    );
                }
            }
        }
    }
}

/// On row 0 with `obs_var = Some(r)`, `standardize = false`, `coef_prior = b0`,
/// `loglik` and `mahal` equal `-0.5*(ln(2*pi) + ln(S0) + nu0^2/S0)` and
/// `|nu0|/sqrt(S0)` with `nu0 = y0 - z0' b0` and `S0 = p0*r*||z0||^2 + r/w0`.
#[test]
fn row_0_innovation_matches_closed_form_gaussian() {
    let r = 0.5;
    let p0 = 2.0;
    let w0 = 1.5;
    let b0 = [1.0, 2.0, -3.0];
    let mut c = base_cfg(2, 1);
    c.obs_var = Some(r);
    c.p0 = p0;
    c.coef_prior = Some(vec![b0.to_vec()]);
    c.emit_loglik = true;
    c.emit_mahal = true;

    let mut m_cov = Kalman::new(c.clone()).unwrap();
    let mut m_sr = Kalman::new(c)
        .unwrap()
        .with_covariance_form(CovarianceForm::SquareRoot)
        .unwrap();

    let x = [0.7, -1.2];
    let y = 4.5;
    let z = [1.0, x[0], x[1]];
    let nu = y - (z[0] * b0[0] + z[1] * b0[1] + z[2] * b0[2]);
    let s0 = p0 * r * (z[0] * z[0] + z[1] * z[1] + z[2] * z[2]) + r / w0;
    let want_ll = -0.5 * ((2.0 * std::f64::consts::PI).ln() + libm::log(s0) + nu * nu / s0);
    let want_mh = nu.abs() / s0.sqrt();

    for m in [&mut m_cov, &mut m_sr] {
        let st = m.step(&x, &[Some(y)], 0.0, w0);
        let Some(Extra::Kalman { values }) = st.extra else {
            panic!("missing Extra::Kalman");
        };
        assert!(
            (values[0] - want_ll).abs() < 1e-14,
            "{} vs {want_ll}",
            values[0]
        );
        assert!(
            (values[1] - want_mh).abs() < 1e-14,
            "{} vs {want_mh}",
            values[1]
        );
    }
}

/// Nonlinear EKF and UKF accept `coef_prior` (including overriding the default
/// `1.0` intercept start for inverse links) and preserve it across save/load.
#[test]
fn nonlinear_ekf_and_ukf_apply_coef_prior_and_round_trip() {
    for (filter, family, link, prior) in [
        (
            KalmanFilter::Ekf,
            KalmanFamily::Poisson,
            KalmanLink::Log,
            vec![vec![0.4, 0.7, -0.3]],
        ),
        (
            KalmanFilter::Ukf,
            KalmanFamily::Gamma,
            KalmanLink::Inverse,
            vec![vec![2.5, 0.1, -0.1]],
        ),
    ] {
        let mut c = base_cfg(2, 1);
        c.obs = KalmanObs {
            family,
            link,
            filter,
            ..KalmanObs::default()
        };
        c.obs_var = None;
        c.half_life = Vec::new();
        c.q = Some(vec![1e-4; 3]);
        c.p0 = 0.05;
        c.coef_prior = Some(prior.clone());

        let mut m = Kalman::new(c).unwrap();
        assert_eq!(m.coefficients(), prior);

        for i in 0..10 {
            let x = [0.2 * (i as f64 - 4.0), -0.1 * i as f64];
            let y = match family {
                KalmanFamily::Poisson => (i % 5) as f64,
                _ => 0.4 + 0.05 * i as f64,
            };
            m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        let bytes = rmp_serde::to_vec(&m.state()).unwrap();
        let back_state: State = rmp_serde::from_slice(&bytes).unwrap();
        let mut back = Kalman::restore(&back_state).unwrap();
        assert_eq!(back.cfg().coef_prior, Some(prior));
        let st_a = m.step(&[0.3, -0.2], &[Some(1.0)], 1.0, 1.0);
        let st_b = back.step(&[0.3, -0.2], &[Some(1.0)], 1.0, 1.0);
        assert_eq!(st_a.pred[0].to_bits(), st_b.pred[0].to_bits());
    }
}
