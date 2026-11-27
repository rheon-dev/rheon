//! Unit tests for `hmm` extensions:
//! - Fixed-lag smoothing (`smooth_lag`, `max_state_bytes`, `smoothed_for`)
//! - Spherical covariance (`HmmCovariance::Spherical`)
//! - Conjugate prior weights (`means_weight`, `covars_prior`, `covars_weight`)
//! - Partial learning (`params` subsets of `"tmc"`)

use super::*;
use crate::{ClockValue, OnlineModel};

fn given_cfg(d: usize, k: usize, cov: HmmCovariance, hl: f64) -> HmmCfg {
    let mut means = Vec::with_capacity(k * d);
    let mut covs = Vec::new();
    for s in 0..k {
        let m = if s == 0 { -2.0 } else { 2.0 + s as f64 };
        for _ in 0..d {
            means.push(m);
        }
        if matches!(cov, HmmCovariance::Spherical) {
            covs.push(1.0 + 0.5 * s as f64);
        } else {
            for i in 0..d {
                for j in 0..d {
                    covs.push(if i == j { 1.0 + 0.5 * s as f64 } else { 0.0 });
                }
            }
        }
    }
    HmmCfg {
        n_features: d,
        k,
        decay: Decay::Halflife(hl),
        covariance: cov,
        precision_prior: 1e-2,
        min_weight: 0.0,
        learn: true,
        transition_prior: 1.0,
        transition: Some(vec![0.9, 0.1, 0.1, 0.9]),
        means: Some(means),
        covs: Some(covs),
        warm_rows: 10,
        seed_rule: SeedRule::First,
        seed: 7,
        tvtp: None,
        prior_rows: None,
        warm_clock: None,
        extras: None,
    }
}

/// Batch forward-backward smoother oracle over a fixed-parameter HMM on
/// observations `xs[0..=t]`, returning `γ_{target | t}`.
fn batch_fb_smoothed(
    xs: &[Vec<f64>],
    pi: &[f64],
    means: &[Vec<f64>],
    vars: &[f64],
    target: usize,
) -> Vec<f64> {
    let k = means.len();
    let d = means[0].len();
    let n = xs.len();
    let log_emit = |x: &[f64], s: usize| -> f64 {
        let v = vars[s];
        let sq: f64 = x
            .iter()
            .zip(&means[s])
            .map(|(a, b)| (a - b) * (a - b))
            .sum();
        -0.5 * (d as f64 * (std::f64::consts::TAU * v).ln() + sq / v)
    };
    let mut alpha = vec![vec![0.0; k]; n];
    let mut emit = vec![vec![0.0; k]; n];
    for t in 0..n {
        let mut lf = vec![0.0; k];
        for (s, slot) in lf.iter_mut().enumerate() {
            *slot = log_emit(&xs[t], s);
        }
        let top = lf.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        for s in 0..k {
            emit[t][s] = (lf[s] - top).exp();
        }
        let mut pred = vec![0.0; k];
        if t == 0 {
            pred.fill(1.0 / k as f64);
        } else {
            for j in 0..k {
                for l in 0..k {
                    pred[l] += alpha[t - 1][j] * pi[j * k + l];
                }
            }
        }
        let mut z = 0.0;
        for l in 0..k {
            alpha[t][l] = pred[l] * emit[t][l];
            z += alpha[t][l];
        }
        for v in &mut alpha[t] {
            *v /= z;
        }
    }
    // Backward pass from n - 1 down to target.
    let mut beta = vec![1.0; k];
    for t in (target + 1..n).rev() {
        let mut next = vec![0.0; k];
        for i in 0..k {
            let mut acc = 0.0;
            for j in 0..k {
                acc += pi[i * k + j] * emit[t][j] * beta[j];
            }
            next[i] = acc;
        }
        let z: f64 = next.iter().sum();
        for v in &mut next {
            *v /= z;
        }
        beta = next;
    }
    let mut gamma: Vec<f64> = (0..k).map(|i| alpha[target][i] * beta[i]).collect();
    let z: f64 = gamma.iter().sum();
    for v in &mut gamma {
        *v /= z;
    }
    gamma
}

#[test]
fn fixed_lag_smoother_matches_exact_forward_backward_for_every_lag() {
    let pi = vec![0.9, 0.1, 0.1, 0.9];
    let means = vec![vec![-1.5, -1.5], vec![1.5, 1.5]];
    let vars = vec![1.0, 1.5];
    // Synthetic stream with a switch and an ambiguous point right at the edge.
    let xs: Vec<Vec<f64>> = vec![
        vec![-1.8, -1.2],
        vec![-1.3, -1.6],
        vec![-0.1, 0.1], // ambiguous
        vec![-1.6, -1.4],
        vec![0.2, -0.1], // ambiguous before shift
        vec![1.7, 1.3],
        vec![1.9, 1.6],
        vec![1.4, 1.8],
        vec![0.1, 0.0], // ambiguous inside regime 1
        vec![1.6, 1.5],
        vec![-1.2, -1.9],
        vec![-1.7, -1.5],
    ];
    for lag in [1usize, 3, 6] {
        let mut cfg = given_cfg(2, 2, HmmCovariance::Spherical, f64::INFINITY);
        cfg.learn = false;
        cfg.means = Some(vec![-1.5, -1.5, 1.5, 1.5]);
        cfg.covs = Some(vec![1.0, 1.5]);
        cfg.precision_prior = 1e-12;
        cfg.extras = Some(Box::new(HmmCfgExtras {
            smooth_lag: Some(lag),
            ..Default::default()
        }));
        let mut m = Hmm::new(cfg).unwrap();
        assert_eq!(m.n_outputs(), 2 * 2 + 2 + 2);
        let mut sm_buf = crate::Smoothed::default();
        for (t, x) in xs.iter().enumerate() {
            m.label_next(ClockValue::F64(t as f64 * 10.0));
            let step = m.step(x, &[], 1.0, 1.0);
            let sm = &step.pred[6..8];
            if t < lag {
                assert!(sm.iter().all(|v| v.is_nan()), "lag={lag} t={t}");
                assert!(m.smoothed_into(&mut sm_buf) && sm_buf.emitted == vec![false]);
            } else {
                let target = t - lag;
                let want = batch_fb_smoothed(&xs[..=t], &pi, &means, &vars, target);
                for s in 0..2 {
                    assert!(
                        (sm[s] - want[s]).abs() < 1e-11,
                        "lag={lag} t={t} s={s}: got {} want {}",
                        sm[s],
                        want[s]
                    );
                }
                assert!(m.smoothed_into(&mut sm_buf));
                assert_eq!(sm_buf.emitted, vec![true]);
                assert_eq!(
                    sm_buf.label,
                    vec![Some(ClockValue::F64(target as f64 * 10.0))]
                );
            }
        }
    }
}

#[test]
fn fixed_lag_smoother_respects_zero_weight_and_clear_lags() {
    let mut cfg = given_cfg(2, 2, HmmCovariance::Full, 50.0);
    cfg.learn = false;
    cfg.extras = Some(Box::new(HmmCfgExtras {
        smooth_lag: Some(2),
        ..Default::default()
    }));
    let mut clean = Hmm::new(cfg.clone()).unwrap();
    let mut with_zero = Hmm::new(cfg).unwrap();
    let xs = [
        [-2.0, -2.0],
        [-1.8, -2.1],
        [0.0, 0.0],
        [2.0, 2.0],
        [2.2, 1.9],
        [2.1, 2.3],
    ];
    for (i, x) in xs.iter().enumerate() {
        clean.label_next(ClockValue::F64(i as f64));
        let sc = clean.step(x, &[], 1.0, 1.0);
        if i == 2 {
            // Insert a zero-weight row with wild features: ring must not move,
            // and `step` agrees with `predict_with` on the zero-weight row.
            with_zero.label_next(ClockValue::F64(999.0));
            let pred_z = with_zero.predict_with(&[50.0, -50.0], &[], 1.0);
            let sz = with_zero.step(&[50.0, -50.0], &[], 1.0, 0.0);
            assert_eq!(sz.pred, pred_z.pred);
        }
        with_zero.label_next(ClockValue::F64(i as f64));
        let sw = with_zero.step(x, &[], 1.0, 1.0);
        for (a, b) in sc.pred.iter().zip(&sw.pred) {
            assert_eq!(a.to_bits(), b.to_bits(), "row {i}");
        }
    }
    // `clear_lags` empties the smoother ring so the next `lag` rows emit NaN.
    with_zero.clear_lags();
    for i in 0..2 {
        with_zero.label_next(ClockValue::F64(100.0 + i as f64));
        let s = with_zero.step(&[2.0, 2.0], &[], 1.0, 1.0);
        assert!(
            s.pred[6..8].iter().all(|v| v.is_nan()),
            "post-clear row {i}"
        );
    }
    with_zero.label_next(ClockValue::F64(102.0));
    let s = with_zero.step(&[2.0, 2.0], &[], 1.0, 1.0);
    assert!(s.pred[6..8].iter().all(|v| v.is_finite()));
    let mut buf = crate::Smoothed::default();
    assert!(with_zero.smoothed_into(&mut buf));
    assert_eq!(buf.label, vec![Some(ClockValue::F64(100.0))]);
}

#[test]
fn spherical_covariance_learns_and_works_with_stabilized_prior() {
    let mut cfg = given_cfg(3, 2, HmmCovariance::Spherical, 40.0);
    cfg.means = None;
    cfg.covs = None;
    cfg.warm_rows = 20;
    cfg.prior_rows = Some(3.0);
    let mut m = Hmm::new(cfg).unwrap();
    let mut s = 42u64;
    let mut lcg = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    };
    let mut right = 0usize;
    let mut total = 0usize;
    for i in 0..200 {
        let reg = (i / 25) % 2;
        let center = if reg == 0 { -3.0 } else { 3.0 };
        let scale = if reg == 0 { 0.5 } else { 1.5 };
        let x = [
            center + scale * lcg(),
            center + scale * lcg(),
            center + scale * lcg(),
        ];
        let step = m.step(&x, &[], 1.0, 1.0);
        if i >= 30 {
            total += 1;
            if step.pred[4] as usize == reg {
                right += 1;
            }
        }
    }
    let acc = (right.max(total - right)) as f64 / total as f64;
    assert!(acc > 0.9, "spherical accuracy {acc}");
    // Spherical stores no full or diag states.
    assert!(m.state_cov(0).is_none());
    assert!(m.state_diag(0).is_none());
    assert!(m.state_factor(0).is_none());
}

#[test]
fn partial_learning_freezes_omitted_components_to_the_bit() {
    let xs = [
        [-2.2, -1.7],
        [-1.6, -2.3],
        [1.9, 2.4],
        [2.5, 1.8],
        [-1.9, -2.0],
        [2.1, 2.1],
    ];
    for params in ["", "t", "m", "c", "tm", "tc", "mc", "tmc"] {
        for cov in [
            HmmCovariance::Full,
            HmmCovariance::Shared,
            HmmCovariance::Diagonal,
            HmmCovariance::Spherical,
        ] {
            let mut cfg = given_cfg(2, 2, cov, 30.0);
            cfg.learn = !params.is_empty();
            cfg.extras = Some(Box::new(HmmCfgExtras {
                params: Some(params.to_string()),
                ..Default::default()
            }));
            let mut m = Hmm::new(cfg).unwrap();
            let init_means: Vec<Vec<f64>> = (0..2).map(|s| m.state_means(s).to_vec()).collect();
            let init_a = m.transition_at(None, 1.0);
            for x in &xs {
                let st = m.step(x, &[], 1.0, 1.0);
                assert!(st.pred.iter().all(|v| v.is_finite()), "{params} {cov:?}");
            }
            let after_means: Vec<Vec<f64>> = (0..2).map(|s| m.state_means(s).to_vec()).collect();
            let after_a = m.transition_at(None, 1.0);
            if !params.contains('m') {
                assert_eq!(
                    after_means, init_means,
                    "means moved under params={params:?} {cov:?}"
                );
            } else {
                assert_ne!(
                    after_means, init_means,
                    "means did not move under params={params:?} {cov:?}"
                );
            }
            if !params.contains('t') {
                assert_eq!(
                    after_a, init_a,
                    "transition moved under params={params:?} {cov:?}"
                );
            } else {
                assert_ne!(
                    after_a, init_a,
                    "transition did not move under params={params:?} {cov:?}"
                );
            }
        }
    }
    // Wide full HMM with params="m" or "c" disables EwFactor (needs both m and c).
    for params in ["m", "c", "tm", "tc"] {
        let mut cfg = given_cfg(14, 2, HmmCovariance::Full, 30.0);
        cfg.extras = Some(Box::new(HmmCfgExtras {
            params: Some(params.to_string()),
            ..Default::default()
        }));
        let mut m = Hmm::new(cfg).unwrap();
        m.step(&[0.5; 14], &[], 1.0, 1.0);
        assert!(
            m.state_factor(0).is_none(),
            "factor active under params={params:?}"
        );
    }
}

#[test]
fn conjugate_prior_weights_match_closed_form_mstep_at_infinite_half_life() {
    let mw = 3.0;
    let cw = 4.0;
    let cp = 0.5;
    let mu0 = [vec![-1.0, -1.0], vec![1.0, 1.0]];
    let mut cfg = given_cfg(2, 2, HmmCovariance::Diagonal, f64::INFINITY);
    cfg.means = Some(vec![-1.0, -1.0, 1.0, 1.0]);
    cfg.covs = Some(vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]);
    cfg.extras = Some(Box::new(HmmCfgExtras {
        means_weight: Some(mw),
        covars_prior: Some(vec![cp]),
        covars_weight: Some(cw),
        ..Default::default()
    }));
    let mut m = Hmm::new(cfg).unwrap();
    // Feed two rows and track responsibilities `gamma` manually to check the
    // closed-form M-step mean: (mw * mu0 + sum gamma * x) / (mw + sum gamma).
    let xs = [[-2.0, -0.5], [1.5, 2.5], [-1.5, -1.2]];
    let mut post_sum = [0.0; 2];
    let mut wx = [[0.0; 2]; 2];
    for x in &xs {
        let st = m.step(x, &[], 1.0, 1.0);
        // After step, `m.p` holds the filtered posterior used to update state.
        for s in 0..2 {
            let g = m.p[s];
            post_sum[s] += g;
            for i in 0..2 {
                wx[s][i] += g * x[i];
            }
        }
        assert!(st.pred.iter().all(|v| v.is_finite()));
    }
    for s in 0..2 {
        for i in 0..2 {
            let want = (mw * mu0[s][i] + wx[s][i]) / (mw + post_sum[s]);
            let got = m.state_means(s)[i];
            assert!(
                (got - want).abs() < 1e-13,
                "s={s} i={i}: got {got} want {want}"
            );
        }
    }
}

#[test]
fn hmm_522_refusals_are_clear() {
    let check = |cfg: HmmCfg, needle: &str| {
        let e = Hmm::new(cfg).unwrap_err();
        assert!(e.contains(needle), "expected {needle:?} in {e:?}");
    };
    // 's' refused in params.
    let mut c = given_cfg(2, 2, HmmCovariance::Full, 20.0);
    c.extras = Some(Box::new(HmmCfgExtras {
        params: Some("stmc".into()),
        ..Default::default()
    }));
    check(c, "cannot include 's'");

    // Conflicting learn + params.
    let mut c = given_cfg(2, 2, HmmCovariance::Full, 20.0);
    c.learn = false;
    c.extras = Some(Box::new(HmmCfgExtras {
        params: Some("m".into()),
        ..Default::default()
    }));
    check(c, "conflicts with params");

    let mut c = given_cfg(2, 2, HmmCovariance::Full, 20.0);
    c.learn = true;
    c.extras = Some(Box::new(HmmCfgExtras {
        params: Some(String::new()),
        ..Default::default()
    }));
    check(c, "conflicts with params=\"\"");

    // 't' with tvtp refused.
    let mut c = given_cfg(2, 2, HmmCovariance::Full, 20.0);
    c.transition = None;
    c.tvtp = Some((vec![0.0; 4], vec![0.0; 4]));
    c.extras = Some(Box::new(HmmCfgExtras {
        params: Some("tm".into()),
        ..Default::default()
    }));
    check(c, "tvtp_coef");

    // covars_prior beside prior_rows refused.
    let mut c = given_cfg(2, 2, HmmCovariance::Full, 20.0);
    c.prior_rows = Some(2.0);
    c.extras = Some(Box::new(HmmCfgExtras {
        covars_prior: Some(vec![0.1]),
        ..Default::default()
    }));
    check(c, "prior_rows");

    // smooth_lag == 0 refused, and cap exceeded refused.
    let mut c = given_cfg(2, 2, HmmCovariance::Full, 20.0);
    c.extras = Some(Box::new(HmmCfgExtras {
        smooth_lag: Some(0),
        ..Default::default()
    }));
    check(c, "smooth_lag must be at least 1");

    let mut c = given_cfg(2, 2, HmmCovariance::Full, 20.0);
    c.extras = Some(Box::new(HmmCfgExtras {
        smooth_lag: Some(500),
        max_state_bytes: Some(128),
        ..Default::default()
    }));
    check(c, "past max_state_bytes");
}
