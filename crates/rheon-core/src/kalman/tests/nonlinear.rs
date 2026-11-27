//! Unit and contract tests for EKF and UKF nonlinear observation filters on
//! `Kalman`.

use super::*;
use crate::model::OnlineModel;

fn nl_cfg(k: usize, obs: KalmanObs, q: Vec<f64>) -> KalmanCfg {
    KalmanCfg {
        n_features: k,
        n_targets: 1,
        fit_intercept: true,
        decay: Decay::Halflife(200.0),
        half_life: Vec::new(),
        q: Some(q),
        obs_var: None,
        p0: 1.0,
        share_p: false,
        min_weight: 0.0,
        revert_half_life: vec![f64::INFINITY],
        standardize: false,
        min_weight_holds_learning: false,
        noise_units: true,
        horizons: Vec::new(),
        obs,
    }
}

fn obs_ekf(family: KalmanFamily, link: KalmanLink) -> KalmanObs {
    KalmanObs {
        family,
        link,
        filter: KalmanFilter::Ekf,
        ..KalmanObs::default()
    }
}

fn obs_ukf(
    family: KalmanFamily,
    link: KalmanLink,
    ukf_rule: UkfRule,
    ukf_noise: UkfNoise,
) -> KalmanObs {
    KalmanObs {
        family,
        link,
        filter: KalmanFilter::Ukf,
        ukf_rule,
        ukf_noise,
        ..KalmanObs::default()
    }
}

fn rel_err(a: &[f64], b: &[f64]) -> f64 {
    let num: f64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt();
    let den: f64 = b.iter().map(|y| y * y).sum::<f64>().sqrt().max(1.0);
    num / den
}

/// On `gaussian + identity`, EKF and both UKF rules (`scalar_index` and `merwe`,
/// `expected` and `plugin`) match the linear `Kalman` filter to ~1e-13 across
/// fixed `obs_var`, learned `noise_units`, standardized/unstandardized,
/// reverting, and `share_p = true`.
#[test]
fn gaussian_identity_ekf_and_ukf_match_linear() {
    let filters = [
        obs_ekf(KalmanFamily::Gaussian, KalmanLink::Identity),
        obs_ukf(
            KalmanFamily::Gaussian,
            KalmanLink::Identity,
            UkfRule::ScalarIndex,
            UkfNoise::Expected,
        ),
        obs_ukf(
            KalmanFamily::Gaussian,
            KalmanLink::Identity,
            UkfRule::ScalarIndex,
            UkfNoise::Plugin,
        ),
        obs_ukf(
            KalmanFamily::Gaussian,
            KalmanLink::Identity,
            UkfRule::Merwe,
            UkfNoise::Expected,
        ),
        obs_ukf(
            KalmanFamily::Gaussian,
            KalmanLink::Identity,
            UkfRule::Merwe,
            UkfNoise::Plugin,
        ),
    ];

    for (case_idx, (standardize, obs_var, revert, share_p, m_targets)) in [
        (false, Some(0.5), f64::INFINITY, false, 1usize),
        (true, Some(0.5), 40.0, false, 1),
        (true, None, 60.0, false, 1),
        (true, None, f64::INFINITY, true, 2),
    ]
    .into_iter()
    .enumerate()
    {
        let base_cfg = KalmanCfg {
            n_features: 3,
            n_targets: m_targets,
            fit_intercept: true,
            decay: Decay::Halflife(100.0),
            half_life: if obs_var.is_some() {
                Vec::new()
            } else {
                vec![50.0]
            },
            q: obs_var.map(|_| vec![1e-4; 4]),
            obs_var,
            p0: 1.5,
            share_p,
            min_weight: 0.0,
            revert_half_life: vec![revert],
            standardize,
            min_weight_holds_learning: false,
            noise_units: true,
            horizons: Vec::new(),
            obs: KalmanObs::default(),
        };

        for obs in &filters {
            let mut lin = Kalman::new(base_cfg.clone()).unwrap();
            let mut nl = Kalman::new(KalmanCfg {
                obs: obs.clone(),
                ..base_cfg.clone()
            })
            .unwrap();
            let mut s = 100u64 + case_idx as u64;
            for t in 0..120 {
                let x = [lcg(&mut s), lcg(&mut s) * 2.0, lcg(&mut s) - 0.5];
                let d = if t == 0 {
                    0.0
                } else {
                    1.0 + (t % 3) as f64 * 0.5
                };
                let w = if t % 17 == 5 {
                    0.0
                } else {
                    0.7 + 0.3 * (t % 4) as f64
                };
                let ys: Vec<Option<f64>> = (0..m_targets)
                    .map(|j| {
                        if t % 19 == 7 + j {
                            None
                        } else {
                            Some(
                                (0.5 + j as f64) + 1.2 * x[0] - 0.7 * x[1]
                                    + 0.4 * x[2]
                                    + 0.3 * lcg(&mut s),
                            )
                        }
                    })
                    .collect();

                // Predict parity before stepping.
                let pr_nl = nl.predict(&x, d);
                let st_lin = lin.step(&x, &ys, d, w);
                let st_nl = nl.step(&x, &ys, d, w);
                for j in 0..m_targets {
                    if pr_nl.pred[j].is_nan() {
                        assert!(
                            st_nl.pred[j].is_nan(),
                            "case {case_idx} t={t} j={j}: predict NaN vs step {}",
                            st_nl.pred[j]
                        );
                    } else {
                        assert_eq!(
                            pr_nl.pred[j].to_bits(),
                            st_nl.pred[j].to_bits(),
                            "case {case_idx} t={t} j={j} {:?}: predict != step",
                            obs.filter
                        );
                    }
                    if st_lin.pred[j].is_nan() {
                        assert!(st_nl.pred[j].is_nan());
                    } else {
                        let diff = (st_lin.pred[j] - st_nl.pred[j]).abs();
                        assert!(
                            diff < 1e-12 * (1.0 + st_lin.pred[j].abs()),
                            "case {case_idx} t={t} j={j} {:?}/{:?}: pred diff {diff}",
                            obs.filter,
                            obs.ukf_rule
                        );
                    }
                }
            }
            for j in 0..m_targets {
                assert!(
                    rel_err(&nl.beta[j], &lin.beta[j]) < 1e-12,
                    "case {case_idx} j={j} {:?}: beta mismatch",
                    obs.filter
                );
            }
            for pi in 0..lin.p.len() {
                assert!(
                    rel_err(&nl.p[pi], &lin.p[pi]) < 1e-12,
                    "case {case_idx} pi={pi} {:?}: P mismatch",
                    obs.filter
                );
            }
        }
    }
}

/// Every offered `(family, link)` pair runs cleanly under EKF, UKF `scalar_index`
/// and UKF `merwe` (both `expected` and `plugin`), keeping `predict == step`
/// to the bit, finite state, and `error_inflation_into == false`.
#[test]
fn all_nonlinear_pairs_predict_matches_step_and_stays_finite() {
    let pairs = [
        (KalmanFamily::Poisson, KalmanLink::Log, None, None, 1.0),
        (KalmanFamily::Binomial, KalmanLink::Logit, None, None, 1.0),
        (KalmanFamily::Gamma, KalmanLink::Log, None, None, 0.5),
        (KalmanFamily::Gamma, KalmanLink::Inverse, None, None, 0.5),
        (KalmanFamily::Tweedie, KalmanLink::Log, Some(1.5), None, 0.8),
        (
            KalmanFamily::Tweedie,
            KalmanLink::Power,
            Some(1.5),
            None,
            0.8,
        ),
        (
            KalmanFamily::Tweedie,
            KalmanLink::Power,
            Some(2.5),
            None,
            0.8,
        ),
        (
            KalmanFamily::Gaussian,
            KalmanLink::Log,
            None,
            Some(0.25),
            1.0,
        ),
        (
            KalmanFamily::Gaussian,
            KalmanLink::Logit,
            None,
            Some(0.04),
            1.0,
        ),
    ];

    for (family, link, var_power, obs_var, disp) in pairs {
        let variants = [
            KalmanObs {
                family,
                link,
                var_power,
                dispersion: disp,
                filter: KalmanFilter::Ekf,
                ..KalmanObs::default()
            },
            KalmanObs {
                family,
                link,
                var_power,
                dispersion: disp,
                filter: KalmanFilter::Ukf,
                ukf_rule: UkfRule::ScalarIndex,
                ukf_noise: UkfNoise::Expected,
                ..KalmanObs::default()
            },
            KalmanObs {
                family,
                link,
                var_power,
                dispersion: disp,
                filter: KalmanFilter::Ukf,
                ukf_rule: UkfRule::Merwe,
                ukf_noise: UkfNoise::Expected,
                ..KalmanObs::default()
            },
            KalmanObs {
                family,
                link,
                var_power,
                dispersion: disp,
                filter: KalmanFilter::Ukf,
                ukf_rule: UkfRule::ScalarIndex,
                ukf_noise: UkfNoise::Plugin,
                ..KalmanObs::default()
            },
        ];

        for standardize in [false, true] {
            for obs in &variants {
                let mut cfg = nl_cfg(2, obs.clone(), vec![1e-4; 3]);
                cfg.obs_var = obs_var;
                cfg.standardize = standardize;
                cfg.p0 = if link.defined_at_zero() { 0.5 } else { 0.05 };
                cfg.revert_half_life = if standardize {
                    vec![80.0]
                } else {
                    vec![f64::INFINITY]
                };
                let mut model = Kalman::new(cfg).unwrap();

                // Error inflation is refused/false on non-Gaussian-identity.
                let mut ei = Vec::new();
                assert!(!model.error_inflation_into(&mut ei));
                assert!(!model.row_error_inflation_into(&[0.1, -0.2], 1.0, &mut ei));
                assert!(!model.error_inflation_gate_into(&[0.1, -0.2], 1.0, &mut ei, 2.0));

                let mut s = 42u64;
                let mut predicted = 0usize;
                for t in 0..80 {
                    let x = [0.4 * lcg(&mut s), 0.4 * lcg(&mut s)];
                    let d = if t == 0 {
                        0.0
                    } else {
                        1.0 + 0.5 * (t % 2) as f64
                    };
                    let w = if t % 13 == 4 { 0.0 } else { 1.0 };
                    let y = if t % 17 == 3 {
                        None
                    } else if t % 23 == 5 && family != KalmanFamily::Gaussian {
                        // Out-of-support target must act as non-observing.
                        Some(-1.0)
                    } else {
                        let u = 0.5 * (lcg(&mut s) + 1.0);
                        Some(match family {
                            KalmanFamily::Poisson => (2.0 + 3.0 * u).floor(),
                            KalmanFamily::Binomial => {
                                if u > 0.4 {
                                    1.0
                                } else {
                                    0.0
                                }
                            }
                            KalmanFamily::Gamma => 0.4 + 1.5 * u,
                            KalmanFamily::Tweedie => {
                                if var_power.unwrap() < 2.0 && u < 0.25 {
                                    0.0
                                } else {
                                    0.5 + 1.5 * u
                                }
                            }
                            KalmanFamily::Gaussian => match link {
                                KalmanLink::Log => 0.8 + 1.0 * u,
                                KalmanLink::Logit => 0.2 + 0.6 * u,
                                _ => u,
                            },
                        })
                    };
                    let pr = model.predict(&x, d);
                    let st = model.step(&x, &[y], d, w);
                    if pr.pred[0].is_nan() {
                        assert!(
                            st.pred[0].is_nan(),
                            "{family:?}/{link:?}/{:?}/std={standardize} t={t}: predict NaN vs step {}",
                            obs.filter,
                            st.pred[0]
                        );
                    } else {
                        assert_eq!(
                            pr.pred[0].to_bits(),
                            st.pred[0].to_bits(),
                            "{family:?}/{link:?}/{:?}/std={standardize} t={t}: predict != step",
                            obs.filter
                        );
                        predicted += 1;
                    }
                    assert!(model.beta[0].iter().all(|v| v.is_finite()));
                    assert!(model.p[0].iter().all(|v| v.is_finite()));
                }
                assert!(
                    predicted > 50,
                    "{family:?}/{link:?}/{:?}/std={standardize}: only {predicted} predictions",
                    obs.filter
                );
                let pv = model.pred_var(&[0.1, -0.1]);
                assert!(pv[0].is_finite() && pv[0] > 0.0);
                let cv = model.coef_variance().unwrap();
                let crate::CoefVariance::Absolute(vars) = cv else {
                    panic!("expected Absolute coef_variance");
                };
                assert!(vars[0].iter().all(|v| v.is_finite() && *v > 0.0));
            }
        }
    }
}

/// Hard Rule 2 (out of sample): changing row `t`'s target and weight cannot
/// move any prediction up to and including row `t` by a single bit.
#[test]
fn nonlinear_filters_are_strictly_out_of_sample() {
    let configs = [
        obs_ekf(KalmanFamily::Poisson, KalmanLink::Log),
        obs_ukf(
            KalmanFamily::Poisson,
            KalmanLink::Log,
            UkfRule::ScalarIndex,
            UkfNoise::Expected,
        ),
        obs_ukf(
            KalmanFamily::Gamma,
            KalmanLink::Log,
            UkfRule::Merwe,
            UkfNoise::Expected,
        ),
    ];
    for obs in configs {
        for standardize in [false, true] {
            let mut cfg = nl_cfg(2, obs.clone(), vec![1e-3; 3]);
            cfg.standardize = standardize;
            let mut a = Kalman::new(cfg.clone()).unwrap();
            let mut b = Kalman::new(cfg).unwrap();
            let mut s = 77u64;
            for t in 0..40 {
                let x = [lcg(&mut s), lcg(&mut s)];
                let y = 1.0 + (lcg(&mut s) + 1.0) * 2.0;
                let (yb, wb) = if t == 25 { (999.0, 7.5) } else { (y, 1.0) };
                let pa = a
                    .step(&x, &[Some(y)], if t == 0 { 0.0 } else { 1.0 }, 1.0)
                    .pred[0];
                let pb = b
                    .step(&x, &[Some(yb)], if t == 0 { 0.0 } else { 1.0 }, wb)
                    .pred[0];
                if t <= 25 {
                    assert_eq!(pa.to_bits(), pb.to_bits(), "row {t} leaked target/weight");
                }
            }
        }
    }
}

/// Hard Rule 9 (zero-weight rows): inserting zero-weight rows inside a gap
/// leaves `b`, `P`, `n_eff`, and all subsequent predictions bit-identical to
/// the stream without those rows.
#[test]
fn zero_weight_rows_fold_into_clock_to_the_bit() {
    let variants = [
        obs_ekf(KalmanFamily::Poisson, KalmanLink::Log),
        obs_ukf(
            KalmanFamily::Binomial,
            KalmanLink::Logit,
            UkfRule::ScalarIndex,
            UkfNoise::Expected,
        ),
        obs_ukf(
            KalmanFamily::Gamma,
            KalmanLink::Log,
            UkfRule::Merwe,
            UkfNoise::Plugin,
        ),
    ];
    for obs in variants {
        for (standardize, revert) in [(false, f64::INFINITY), (true, f64::INFINITY), (true, 25.0)] {
            let mut cfg = nl_cfg(2, obs.clone(), vec![2e-4; 3]);
            cfg.standardize = standardize;
            cfg.revert_half_life = vec![revert];
            let mut clean = Kalman::new(cfg.clone()).unwrap();
            let mut split = Kalman::new(cfg).unwrap();
            let bit_exact = !standardize && revert.is_infinite();
            let mut s = 91u64;
            for t in 0..60 {
                let x = [lcg(&mut s), lcg(&mut s)];
                let u = 0.5 * (lcg(&mut s) + 1.0);
                let y = match obs.family {
                    KalmanFamily::Poisson => (4.0 * u).floor(),
                    KalmanFamily::Binomial => {
                        if u > 0.5 {
                            1.0
                        } else {
                            0.0
                        }
                    }
                    _ => 0.5 + 2.0 * u,
                };
                if t == 20 {
                    // Split a total gap of 4.0 into 1.0 + 1.5 + 1.5 with zero-weight rows.
                    split.step(&[10.0, -5.0], &[Some(y)], 1.0, 0.0);
                    split.step(&[-3.0, 8.0], &[Some(y)], 1.5, 0.0);
                    let ps = split.step(&x, &[Some(y)], 1.5, 1.0);
                    let pc = clean.step(&x, &[Some(y)], 4.0, 1.0);
                    if pc.pred[0].is_nan() {
                        assert!(ps.pred[0].is_nan());
                    } else if bit_exact {
                        assert_eq!(ps.pred[0].to_bits(), pc.pred[0].to_bits());
                    } else {
                        assert!((ps.pred[0] - pc.pred[0]).abs() < 1e-13 * (1.0 + pc.pred[0].abs()));
                    }
                } else {
                    let d = if t == 0 { 0.0 } else { 1.0 };
                    let ps = split.step(&x, &[Some(y)], d, 1.0);
                    let pc = clean.step(&x, &[Some(y)], d, 1.0);
                    assert!(
                        (ps.n_eff - pc.n_eff).abs() <= 1e-13 * (1.0 + pc.n_eff),
                        "row {t}: n_eff {} vs {}",
                        ps.n_eff,
                        pc.n_eff
                    );
                    if pc.pred[0].is_nan() {
                        assert!(ps.pred[0].is_nan());
                    } else if bit_exact {
                        assert_eq!(ps.pred[0].to_bits(), pc.pred[0].to_bits());
                    } else {
                        assert!((ps.pred[0] - pc.pred[0]).abs() < 1e-13 * (1.0 + pc.pred[0].abs()));
                    }
                }
            }
            if bit_exact {
                for (a, b) in split.beta[0].iter().zip(&clean.beta[0]) {
                    assert_eq!(a.to_bits(), b.to_bits());
                }
                for (a, b) in split.p[0].iter().zip(&clean.p[0]) {
                    assert_eq!(a.to_bits(), b.to_bits());
                }
            } else {
                assert!(rel_err(&split.beta[0], &clean.beta[0]) < 1e-13);
                assert!(rel_err(&split.p[0], &clean.p[0]) < 1e-13);
            }
        }
    }
}

/// Compact positional msgpack, named msgpack, and JSON round-trips continue
/// bit-for-bit, and linear `KalmanCfg` omits `obs` completely.
#[test]
fn state_roundtrip_compact_named_and_json_continues_to_the_bit() {
    // Verify linear KalmanCfg writes zero `obs` key in named msgpack & JSON.
    let lin_cfg = plain(1.0, 50.0, Some(0.5));
    let json = serde_json::to_string(&lin_cfg).unwrap();
    assert!(
        !json.contains("\"obs\""),
        "linear KalmanCfg wrote obs: {json}"
    );
    let named = rmp_serde::to_vec_named(&lin_cfg).unwrap();
    let val = rmpv::decode::read_value(&mut &named[..]).unwrap();
    if let rmpv::Value::Map(entries) = val {
        assert!(
            !entries.iter().any(|(k, _)| k.as_str() == Some("obs")),
            "linear KalmanCfg wrote obs in msgpack"
        );
    }

    let variants = [
        obs_ekf(KalmanFamily::Poisson, KalmanLink::Log),
        obs_ukf(
            KalmanFamily::Gamma,
            KalmanLink::Inverse,
            UkfRule::ScalarIndex,
            UkfNoise::Expected,
        ),
        obs_ukf(
            KalmanFamily::Binomial,
            KalmanLink::Logit,
            UkfRule::Merwe,
            UkfNoise::Plugin,
        ),
    ];
    for obs in variants {
        let mut cfg = nl_cfg(2, obs.clone(), vec![1e-4; 3]);
        cfg.p0 = 0.05;
        cfg.standardize = true;
        cfg.revert_half_life = vec![50.0];
        let mut whole = Kalman::new(cfg).unwrap();
        let mut s = 33u64;
        let mut rows = Vec::new();
        for t in 0..50 {
            let x = [0.3 * lcg(&mut s), 0.3 * lcg(&mut s)];
            let u = 0.5 * (lcg(&mut s) + 1.0);
            let y = match obs.family {
                KalmanFamily::Poisson => (3.0 * u).floor(),
                KalmanFamily::Binomial => {
                    if u > 0.5 {
                        1.0
                    } else {
                        0.0
                    }
                }
                _ => 0.5 + u,
            };
            rows.push((x, Some(y), if t == 0 { 0.0 } else { 1.0 }, 1.0));
        }
        for &(x, y, d, w) in &rows[..25] {
            whole.step(&x, &[y], d, w);
        }
        let st = whole.state();
        let compact = rmp_serde::to_vec(&st).unwrap();
        let named = rmp_serde::to_vec_named(&st).unwrap();
        let json_st = serde_json::to_string(&st).unwrap();

        let mut from_compact = Kalman::restore(&rmp_serde::from_slice(&compact).unwrap()).unwrap();
        let mut from_named = Kalman::restore(&rmp_serde::from_slice(&named).unwrap()).unwrap();
        let mut from_json = Kalman::restore(&serde_json::from_str(&json_st).unwrap()).unwrap();

        for &(x, y, d, w) in &rows[25..] {
            let pw = whole.step(&x, &[y], d, w).pred[0];
            let pc = from_compact.step(&x, &[y], d, w).pred[0];
            let pn = from_named.step(&x, &[y], d, w).pred[0];
            let pj = from_json.step(&x, &[y], d, w).pred[0];
            assert_eq!(pw.to_bits(), pc.to_bits());
            assert_eq!(pw.to_bits(), pn.to_bits());
            assert_eq!(pw.to_bits(), pj.to_bits());
        }
    }
}

/// When Cholesky fails on `P⁻` (non-positive-definite in rounding), Merwe UKF
/// falls back to the EKF step on the exact same state with no jitter.
#[test]
fn merwe_cholesky_failure_falls_back_to_ekf_on_same_state() {
    let mut cfg_ukf = nl_cfg(
        1,
        obs_ukf(
            KalmanFamily::Poisson,
            KalmanLink::Log,
            UkfRule::Merwe,
            UkfNoise::Expected,
        ),
        vec![0.0; 2],
    );
    cfg_ukf.p0 = 1.0;
    let mut cfg_ekf = cfg_ukf.clone();
    cfg_ekf.obs.filter = KalmanFilter::Ekf;

    let mut ukf = Kalman::new(cfg_ukf).unwrap();
    let mut ekf = Kalman::new(cfg_ekf).unwrap();

    // Seed both filters identically with a non-PD `P` (positive diagonal, det < 0).
    let bad_p = vec![1.0, 1.01, 1.01, 1.0];
    ukf.p[0] = bad_p.clone();
    ekf.p[0] = bad_p;
    ukf.beta[0] = vec![0.3, -0.2];
    ekf.beta[0] = vec![0.3, -0.2];
    ukf.wj[0] = 5.0;
    ekf.wj[0] = 5.0;

    let x = [0.7];
    let pr_u = ukf.predict(&x, 1.0);
    let pr_e = ekf.predict(&x, 1.0);
    assert_eq!(pr_u.pred[0].to_bits(), pr_e.pred[0].to_bits());

    let st_u = ukf.step(&x, &[Some(2.0)], 1.0, 1.0);
    let st_e = ekf.step(&x, &[Some(2.0)], 1.0, 1.0);
    assert_eq!(st_u.pred[0].to_bits(), st_e.pred[0].to_bits());
    assert_eq!(
        ukf.beta[0].iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        ekf.beta[0].iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
    assert_eq!(
        ukf.p[0].iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        ekf.p[0].iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
}

/// Inverse/power link guards:
/// 1. When `η_0 ≤ 0`, `pred` is withheld (NaN), `b` and `P` do not move, and `elapsed` accumulates.
/// 2. When `η_0 > 0` is valid but a UKF sigma point crosses `η ≤ 0`, UKF falls back to EKF on the same state.
#[test]
fn inverse_link_guards_withhold_at_nonpositive_eta_and_fallback_ukf_sigma() {
    // Case 1: η_0 <= 0 withholds and learns nothing.
    let cfg = nl_cfg(
        1,
        obs_ekf(KalmanFamily::Gamma, KalmanLink::Inverse),
        vec![1e-3; 2],
    );
    let mut m = Kalman::new(cfg).unwrap();
    // Initial intercept is 1.0 (defined at zero-slope).
    assert_eq!(m.beta[0], vec![1.0, 0.0]);
    m.step(&[0.5], &[Some(1.0)], 0.0, 1.0);
    // Force `beta` so `eta0 = -0.5 <= 0` at `x = [1.0]`.
    m.beta[0] = vec![0.5, -1.0];
    let b_before = m.beta[0].clone();
    let p_before = m.p[0].clone();
    let el_before = m.elapsed[0];
    let pr = m.predict(&[1.0], 2.0);
    assert!(pr.pred[0].is_nan());
    let st = m.step(&[1.0], &[Some(1.5)], 2.0, 1.0);
    assert!(st.pred[0].is_nan());
    assert_eq!(m.beta[0], b_before);
    assert_eq!(m.p[0], p_before);
    assert_eq!(m.elapsed[0], el_before + 2.0);

    // Case 2: η_0 = 1.0 > 0, but large `p0 = 1.0` sends `η_0 - sqrt(3) * s <= 0`.
    let cfg_ukf = nl_cfg(
        1,
        obs_ukf(
            KalmanFamily::Gamma,
            KalmanLink::Inverse,
            UkfRule::ScalarIndex,
            UkfNoise::Expected,
        ),
        vec![0.0; 2],
    );
    let mut cfg_ekf = cfg_ukf.clone();
    cfg_ekf.obs.filter = KalmanFilter::Ekf;
    let mut ukf = Kalman::new(cfg_ukf).unwrap();
    let mut ekf = Kalman::new(cfg_ekf).unwrap();
    ukf.wj[0] = 1.0;
    ekf.wj[0] = 1.0;
    let x = [0.5];
    assert_eq!(
        ukf.predict(&x, 1.0).pred[0].to_bits(),
        ekf.predict(&x, 1.0).pred[0].to_bits()
    );
    let su = ukf.step(&x, &[Some(1.2)], 1.0, 1.0);
    let se = ekf.step(&x, &[Some(1.2)], 1.0, 1.0);
    assert_eq!(su.pred[0].to_bits(), se.pred[0].to_bits());
    assert_eq!(ukf.beta[0], ekf.beta[0]);
    assert_eq!(ukf.p[0], ekf.p[0]);
}

/// Clipped `η = ±30` still linearizes at the bound (`dμ/dη != 0`) so EKF does
/// not freeze.
#[test]
fn clipped_eta_linearizes_at_bound_without_freezing() {
    let cfg = nl_cfg(
        1,
        obs_ekf(KalmanFamily::Poisson, KalmanLink::Log),
        vec![0.0; 2],
    );
    let mut m = Kalman::new(cfg).unwrap();
    m.beta[0] = vec![40.0, 0.0]; // well past ETA_CLIP = 30
    m.wj[0] = 1.0;
    let b0_before = m.beta[0][0];
    let st = m.step(&[0.0], &[Some(1.0)], 1.0, 1.0);
    assert!(st.pred[0].is_finite());
    assert!(
        m.beta[0][0] < b0_before,
        "EKF froze at clipped eta: before {b0_before}, after {}",
        m.beta[0][0]
    );
}

/// `ScalarIndex` UKF is invariant to feature ordering to machine epsilon, even
/// at `k = 10` on Poisson, Binomial and Gamma.
#[test]
fn scalar_index_ukf_is_feature_order_invariant() {
    for (family, link) in [
        (KalmanFamily::Poisson, KalmanLink::Log),
        (KalmanFamily::Binomial, KalmanLink::Logit),
        (KalmanFamily::Gamma, KalmanLink::Log),
    ] {
        let k = 10;
        let obs = obs_ukf(family, link, UkfRule::ScalarIndex, UkfNoise::Expected);
        let cfg = nl_cfg(k, obs, vec![1e-4; k + 1]);
        let mut fwd = Kalman::new(cfg.clone()).unwrap();
        let mut rev = Kalman::new(cfg).unwrap();
        let mut s = 555u64;
        let mut max_diff = 0.0f64;
        for t in 0..80 {
            let x: Vec<f64> = (0..k).map(|_| 0.25 * lcg(&mut s)).collect();
            let xr: Vec<f64> = x.iter().copied().rev().collect();
            let u = 0.5 * (lcg(&mut s) + 1.0);
            let y = match family {
                KalmanFamily::Poisson => (3.0 * u).floor(),
                KalmanFamily::Binomial => {
                    if u > 0.5 {
                        1.0
                    } else {
                        0.0
                    }
                }
                _ => 0.5 + 2.0 * u,
            };
            let d = if t == 0 { 0.0 } else { 1.0 };
            let pf = fwd.step(&x, &[Some(y)], d, 1.0).pred[0];
            let pr = rev.step(&xr, &[Some(y)], d, 1.0).pred[0];
            if pf.is_finite() {
                max_diff = max_diff.max((pf - pr).abs() / pf.abs().max(1.0));
            }
        }
        assert!(
            max_diff < 1e-13,
            "{family:?} scalar_index feature-order shift {max_diff:e}"
        );
    }
}

/// `smooth_lag` composes with EKF and UKF (both rules), matching an explicit
/// backward RTS recursion over the logged filtered/prior `(b, P)` steps and
/// skipping non-observing rows (out-of-support `y`, zero weight, null `y`).
#[test]
fn smoother_composes_with_ekf_and_ukf_matching_explicit_rts() {
    for obs in [
        obs_ekf(KalmanFamily::Poisson, KalmanLink::Log),
        obs_ukf(
            KalmanFamily::Gamma,
            KalmanLink::Log,
            UkfRule::ScalarIndex,
            UkfNoise::Expected,
        ),
        obs_ukf(
            KalmanFamily::Binomial,
            KalmanLink::Logit,
            UkfRule::Merwe,
            UkfNoise::Expected,
        ),
    ] {
        let k = 3usize; // intercept + 2 features
        let lag = 4usize;
        let mut cfg = nl_cfg(2, obs.clone(), vec![2e-3; k]);
        cfg.revert_half_life = vec![30.0];
        let mut model = Kalman::new(cfg).unwrap().with_smoother(lag, true).unwrap();

        let mut s = 123u64;
        let mut emitted_rows = Vec::new();
        for t in 0..40 {
            let x = [0.5 * lcg(&mut s), 0.5 * lcg(&mut s)];
            let d = if t == 0 { 0.0 } else { 1.0 };
            let w = if t == 12 { 0.0 } else { 1.0 };
            let u = 0.5 * (lcg(&mut s) + 1.0);
            let y = if t == 8 {
                None
            } else if t == 15 {
                Some(-5.0) // out of support: must NOT enter smoother ring
            } else {
                Some(match obs.family {
                    KalmanFamily::Poisson => (1.0 + 4.0 * u).floor(),
                    KalmanFamily::Binomial => {
                        if u > 0.5 {
                            1.0
                        } else {
                            0.0
                        }
                    }
                    _ => 0.4 + 1.8 * u,
                })
            };
            model.label_next(crate::ClockValue::I64(t as i64));
            model.step(&x, &[y], d, w);
            let sm = model.smoothed().unwrap();
            if sm.emitted[0] {
                emitted_rows.push((t, sm.label[0].unwrap(), sm.coef.clone(), sm.var.clone()));
            }
        }

        // Verify the logged sequence: only valid observing rows were logged.
        // Expected observing rows: 40 total minus rows 8 (null), 12 (w=0), 15 (out of support) = 37.
        // Each observing row logs prior + filtered (2 entries) => 37 * 2 = 74 entries.
        let log = model.smooth_log();
        assert_eq!(
            log.len(),
            74,
            "{:?}: unexpected smooth_log length",
            obs.filter
        );
        assert_eq!(emitted_rows.len(), 37 - lag);

        // Extract filtered pairs (b_n, P_n) and next-step priors (b⁻_{n+1}, P⁻_{n+1}) and run explicit RTS.
        let mut filt_b = Vec::new();
        let mut filt_p = Vec::new();
        let mut filt_label = Vec::new();
        let mut prior_b = Vec::new();
        let mut prior_p = Vec::new();
        for entry in log {
            if entry.prior {
                prior_b.push(entry.b.clone());
                prior_p.push(entry.p.clone());
            } else {
                filt_b.push(entry.b.clone());
                filt_p.push(entry.p.clone());
                filt_label.push(entry.label);
            }
        }
        assert_eq!(filt_b.len(), 37);
        assert_eq!(prior_b.len(), 37);
        // `prior_b[n + 1]` is the prior before observation `n + 1`.
        let next_prior_b = &prior_b[1..];
        let next_prior_p = &prior_p[1..];

        // Check every emitted smoothed row (at observation `idx + lag` describing `idx`).
        for (idx, (_, label, coef, var)) in emitted_rows.iter().enumerate() {
            assert_eq!(*label, filt_label[idx]);
            let end = idx + lag;
            let mut m_vec = filt_b[end].clone();
            let mut v_mat = filt_p[end].clone();
            for n in (idx..end).rev() {
                // F_n = diag(phi) for elapsed clock between observation n and n+1.
                let (crate::ClockValue::I64(t0), crate::ClockValue::I64(t1)) =
                    (filt_label[n], filt_label[n + 1])
                else {
                    unreachable!()
                };
                let dt = (t1 - t0) as f64;
                let ph = Decay::Halflife(30.0).portable_factor(dt);
                // Solve P⁻_{n+1} X = (P_n * ph) => G_n = (P_n * ph) * (P⁻_{n+1})⁻¹.
                let mut pp = next_prior_p[n].clone();
                // Simple Cholesky inverse via Gauss-Jordan for 3x3.
                let inv = invert_small(&mut pp, k);
                let mut g = vec![0.0; k * k];
                for i in 0..k {
                    for j in 0..k {
                        g[i * k + j] = (0..k)
                            .map(|l| filt_p[n][i * k + l] * ph * inv[l * k + j])
                            .sum();
                    }
                }
                let mut m_next = vec![0.0; k];
                for i in 0..k {
                    let diff_dot: f64 = (0..k)
                        .map(|j| g[i * k + j] * (m_vec[j] - next_prior_b[n][j]))
                        .sum();
                    m_next[i] = filt_b[n][i] + diff_dot;
                }
                let mut dv = vec![0.0; k * k];
                for a in 0..k * k {
                    dv[a] = v_mat[a] - next_prior_p[n][a];
                }
                let mut gdv = vec![0.0; k * k];
                for i in 0..k {
                    for j in 0..k {
                        gdv[i * k + j] = (0..k).map(|l| g[i * k + l] * dv[l * k + j]).sum();
                    }
                }
                let mut v_next = filt_p[n].clone();
                for i in 0..k {
                    for j in 0..k {
                        v_next[i * k + j] +=
                            (0..k).map(|l| gdv[i * k + l] * g[j * k + l]).sum::<f64>();
                    }
                }
                m_vec = m_next;
                v_mat = v_next;
            }
            assert!(
                rel_err(coef, &m_vec) < 1e-12,
                "{:?} obs {idx}: smoothed coef diff",
                obs.filter
            );
            let diag: Vec<f64> = (0..k).map(|i| v_mat[i * k + i]).collect();
            assert!(
                rel_err(var, &diag) < 1e-12,
                "{:?} obs {idx}: smoothed var diff",
                obs.filter
            );
        }
    }
}

fn invert_small(a: &mut [f64], n: usize) -> Vec<f64> {
    let mut aug = vec![0.0; n * 2 * n];
    for i in 0..n {
        for j in 0..n {
            aug[i * 2 * n + j] = a[i * n + j];
        }
        aug[i * 2 * n + n + i] = 1.0;
    }
    for col in 0..n {
        let piv = aug[col * 2 * n + col];
        for j in 0..2 * n {
            aug[col * 2 * n + j] /= piv;
        }
        for row in 0..n {
            if row == col {
                continue;
            }
            let f = aug[row * 2 * n + col];
            for j in 0..2 * n {
                aug[row * 2 * n + j] -= f * aug[col * 2 * n + j];
            }
        }
    }
    let mut inv = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            inv[i * n + j] = aug[i * 2 * n + n + j];
        }
    }
    inv
}

/// Every invalid `KalmanObs` / `KalmanCfg` combination is refused at construction
/// and on damaged-state restore (`STATE-3`).
#[test]
fn nonlinear_config_and_damaged_state_refusals() {
    let base_pois = nl_cfg(
        2,
        obs_ekf(KalmanFamily::Poisson, KalmanLink::Log),
        vec![1e-4; 3],
    );

    let bad_cases: Vec<(KalmanCfg, &str)> = vec![
        // Linear filter with non-Gaussian family
        (
            KalmanCfg {
                obs: KalmanObs {
                    family: KalmanFamily::Poisson,
                    link: KalmanLink::Log,
                    filter: KalmanFilter::Linear,
                    ..KalmanObs::default()
                },
                ..base_pois.clone()
            },
            "filter=\"linear\" applies only",
        ),
        // Invalid family/link pair (binomial + log)
        (
            KalmanCfg {
                obs: obs_ekf(KalmanFamily::Binomial, KalmanLink::Log),
                ..base_pois.clone()
            },
            "the binomial family takes the link logit",
        ),
        // Missing var_power on Tweedie
        (
            KalmanCfg {
                obs: obs_ekf(KalmanFamily::Tweedie, KalmanLink::Log),
                ..base_pois.clone()
            },
            "Tweedie family needs var_power",
        ),
        // Bad var_power = 2.0 on Tweedie
        (
            KalmanCfg {
                obs: KalmanObs {
                    var_power: Some(2.0),
                    ..obs_ekf(KalmanFamily::Tweedie, KalmanLink::Log)
                },
                ..base_pois.clone()
            },
            "var_power must be above 1, finite and not 2",
        ),
        // var_power on non-Tweedie
        (
            KalmanCfg {
                obs: KalmanObs {
                    var_power: Some(1.5),
                    ..obs_ekf(KalmanFamily::Poisson, KalmanLink::Log)
                },
                ..base_pois.clone()
            },
            "var_power is the Tweedie family's power",
        ),
        // Non-positive dispersion
        (
            KalmanCfg {
                obs: KalmanObs {
                    dispersion: 0.0,
                    ..obs_ekf(KalmanFamily::Poisson, KalmanLink::Log)
                },
                ..base_pois.clone()
            },
            "dispersion must be finite and above 0",
        ),
        // dispersion != 1 on Gaussian
        (
            KalmanCfg {
                obs: KalmanObs {
                    dispersion: 2.0,
                    ..obs_ekf(KalmanFamily::Gaussian, KalmanLink::Log)
                },
                obs_var: Some(1.0),
                ..base_pois.clone()
            },
            "dispersion is for the non-Gaussian families",
        ),
        // Missing obs_var on Gaussian + Log
        (
            KalmanCfg {
                obs: obs_ekf(KalmanFamily::Gaussian, KalmanLink::Log),
                obs_var: None,
                ..base_pois.clone()
            },
            "needs a fixed obs_var",
        ),
        // obs_var on Poisson
        (
            KalmanCfg {
                obs_var: Some(1.0),
                ..base_pois.clone()
            },
            "obs_var cannot be set with family=",
        ),
        // Missing q / present coef_half_life on Poisson
        (
            KalmanCfg {
                q: None,
                half_life: vec![50.0],
                ..base_pois.clone()
            },
            "needs an explicit q and cannot use coef_half_life",
        ),
        // fit_intercept = false on Inverse link
        (
            KalmanCfg {
                fit_intercept: false,
                q: Some(vec![1e-4; 2]),
                obs: obs_ekf(KalmanFamily::Gamma, KalmanLink::Inverse),
                ..base_pois.clone()
            },
            "the inverse link gives a mean only where",
        ),
        // share_p on Poisson
        (
            KalmanCfg {
                share_p: true,
                ..base_pois.clone()
            },
            "share_p cannot be used with a nonlinear observation",
        ),
        // forecast_spans on Poisson
        (
            KalmanCfg {
                horizons: vec![5.0],
                ..base_pois.clone()
            },
            "forecast_spans is only supported for family=\"gaussian\" with link=\"identity\"",
        ),
        // alpha <= 0
        (
            KalmanCfg {
                obs: KalmanObs {
                    alpha: 0.0,
                    ..obs_ukf(
                        KalmanFamily::Poisson,
                        KalmanLink::Log,
                        UkfRule::Merwe,
                        UkfNoise::Expected,
                    )
                },
                ..base_pois.clone()
            },
            "alpha must be finite and above 0",
        ),
        // kappa <= -k on Merwe
        (
            KalmanCfg {
                obs: KalmanObs {
                    kappa: -3.0,
                    ..obs_ukf(
                        KalmanFamily::Poisson,
                        KalmanLink::Log,
                        UkfRule::Merwe,
                        UkfNoise::Expected,
                    )
                },
                ..base_pois.clone()
            },
            "Merwe sigma-point scale must be finite and above 0",
        ),
    ];

    for (i, (c, want)) in bad_cases.into_iter().enumerate() {
        let err = Kalman::new(c).expect_err(&format!("case {i} accepted, expected {want:?}"));
        assert!(
            err.contains(want),
            "case {i} wrong message: got {err:?}, expected substring {want:?}"
        );
    }

    // SquareRoot form refused with EKF/UKF on construction...
    let err = Kalman::new(base_pois.clone())
        .unwrap()
        .with_covariance_form(CovarianceForm::SquareRoot)
        .unwrap_err();
    assert!(
        err.contains("covariance_form=\"square_root\" applies only"),
        "{err}"
    );

    // ...and refused when a saved state is tampered to hold SquareRoot + EKF!
    let mut m = Kalman::new(base_pois).unwrap();
    m.form = CovarianceForm::SquareRoot;
    let bytes = rmp_serde::to_vec_named(&m.state()).unwrap();
    let res: Result<State, _> = rmp_serde::from_slice(&bytes);
    assert!(
        res.is_err(),
        "tampered SquareRoot + EKF state must be refused on load"
    );
}
