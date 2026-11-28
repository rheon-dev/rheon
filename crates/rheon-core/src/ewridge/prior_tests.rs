//! Unit tests for [`EwRidge`] hierarchical dynamic priors (`prior_weight` and
//! `prior_include_intercept`), verified against dense closed-form linear
//! algebra oracles.

use crate::constraint::Constraint;
use crate::lincon::LinearConstraints;
use crate::model::{ModelState, OnlineModel, State};
use crate::{Decay, EwRidge, EwRidgeCfg, TargetGaps};

fn base_cfg(n_features: usize, n_targets: usize) -> EwRidgeCfg {
    EwRidgeCfg {
        n_features,
        n_targets,
        fit_intercept: true,
        decay: Decay::Halflife(f64::INFINITY),
        ridge: vec![0.25],
        feature_sets: vec![],
        standardize: false,
        ridge_scale: false,
        coef_prior: None,
        session_shrink: None,
        long_half_life: None,
        min_weight: 0.0,
        solve_every: 0.0,
        max_rows_between_solves: 1,
        solve_share: None,
        gram_block_rows: 0,
        target_gaps: TargetGaps::OwnRows,
        window: None,
        window_every: None,
        max_rows_between_snapshots: None,
        constraints: None,
        evidence: None,
        constraint: None,
        prior_weight: Some(5.0),
        prior_include_intercept: true,
    }
}

/// Dense Gaussian elimination with partial pivoting (`A x = b`, `n x n`).
fn dense_solve(mut a: Vec<f64>, mut b: Vec<f64>, n: usize) -> Vec<f64> {
    for i in 0..n {
        let mut piv = i;
        let mut best = a[i * n + i].abs();
        for r in (i + 1)..n {
            let v = a[r * n + i].abs();
            if v > best {
                best = v;
                piv = r;
            }
        }
        assert!(best > 1e-15, "singular oracle system at col {i}");
        if piv != i {
            for c in i..n {
                a.swap(i * n + c, piv * n + c);
            }
            b.swap(i, piv);
        }
        let d = a[i * n + i];
        for r in (i + 1)..n {
            let f = a[r * n + i] / d;
            for c in i..n {
                a[r * n + c] -= f * a[i * n + c];
            }
            b[r] -= f * b[i];
        }
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = b[i];
        for c in (i + 1)..n {
            s -= a[i * n + c] * x[c];
        }
        x[i] = s / a[i * n + i];
    }
    x
}

fn dense_inv(a: &[f64], n: usize) -> Vec<f64> {
    let mut inv = vec![0.0; n * n];
    for c in 0..n {
        let mut e = vec![0.0; n];
        e[c] = 1.0;
        let col = dense_solve(a.to_vec(), e, n);
        for r in 0..n {
            inv[r * n + c] = col[r];
        }
    }
    inv
}

fn synthetic_rows(n: usize, k: usize, seed: u64, shift: f64) -> Vec<(Vec<f64>, f64, f64)> {
    let mut s = seed;
    let mut next = || -> f64 {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u = ((s >> 11) as f64) / ((1u64 << 53) as f64);
        u * 2.0 - 1.0
    };
    let mut rows = Vec::with_capacity(n);
    for _ in 0..n {
        let x: Vec<f64> = (0..k).map(|i| shift + (i as f64 + 1.0) * next()).collect();
        let y = 1.5
            + x.iter()
                .enumerate()
                .map(|(i, &xi)| (0.4 * (i as f64 + 1.0)) * (xi - shift))
                .sum::<f64>()
            + 0.3 * next();
        let w = 0.5 + 1.5 * next().abs();
        rows.push((x, y, w));
    }
    rows
}

/// Exact sum-scale unstandardized normal equation oracle on decayed rows.
#[allow(clippy::too_many_arguments)]
fn unstd_oracle(
    rows: &[(Vec<f64>, f64, f64)],
    decay: Decay,
    fit_intercept: bool,
    ridge: f64,
    ridge_scale: bool,
    c0: Option<&[f64]>,
    bp: &[f64],
    tau: f64,
    inc_int: bool,
) -> (Vec<f64>, Vec<f64>, f64, f64) {
    let kf = rows[0].0.len();
    let kc = kf + usize::from(fit_intercept);
    let mut w_tot = 0.0;
    let mut w2_tot = 0.0;
    let mut ps = 1.0;
    let mut s_sum = vec![0.0; kc * kc];
    let mut r_sum = vec![0.0; kc];
    for (x, y, w) in rows {
        let lam = decay.factor(1.0);
        for v in &mut s_sum {
            *v *= lam;
        }
        for v in &mut r_sum {
            *v *= lam;
        }
        w_tot = lam * w_tot + w;
        w2_tot = lam * lam * w2_tot + w * w;
        ps *= lam;
        let mut z = Vec::with_capacity(kc);
        if fit_intercept {
            z.push(1.0);
        }
        z.extend_from_slice(x);
        for i in 0..kc {
            for j in 0..kc {
                s_sum[i * kc + j] += w * z[i] * z[j];
            }
            r_sum[i] += w * z[i] * y;
        }
    }
    let mut a = s_sum.clone();
    let mut b = r_sum.clone();
    for i in 0..kc {
        let r_pen = if ridge_scale {
            ps * ridge
        } else if fit_intercept && i == 0 {
            0.0
        } else {
            w_tot * ridge
        };
        let p_pen = if fit_intercept && !inc_int && i == 0 {
            0.0
        } else {
            tau
        };
        a[i * kc + i] += r_pen + p_pen;
        if let Some(c0) = c0 {
            b[i] += r_pen * c0[i];
        }
        b[i] += p_pen * bp[i];
    }
    let beta = dense_solve(a.clone(), b, kc);
    let a_inv = dense_inv(&a, kc);
    let n_kish = (w_tot * w_tot) / w2_tot;
    (beta, a_inv, w_tot, n_kish)
}

#[test]
fn unstandardized_matches_closed_form_oracle_all_modes() {
    let rows = synthetic_rows(25, 3, 42, 0.0);
    let bp_int = vec![0.7, -0.3, 1.2, 0.5];
    let c0_int = vec![0.1, 0.2, -0.4, 0.3];
    let bp_no = vec![-0.3, 1.2, 0.5];
    let c0_no = vec![0.2, -0.4, 0.3];

    for &decay in &[Decay::Halflife(f64::INFINITY), Decay::Halflife(10.0)] {
        for &(fit_intercept, ridge_scale, inc_int) in &[
            (true, false, true),
            (true, false, false),
            (false, false, true),
            (true, true, true),
            (true, true, false),
            (false, true, true),
        ] {
            let bp = if fit_intercept { &bp_int } else { &bp_no };
            let c0 = if fit_intercept { &c0_int } else { &c0_no };
            let mut cfg = base_cfg(3, 1);
            cfg.decay = decay;
            cfg.fit_intercept = fit_intercept;
            cfg.ridge_scale = ridge_scale;
            cfg.prior_include_intercept = inc_int;
            cfg.coef_prior = Some(vec![c0.clone()]);
            cfg.prior_weight = Some(4.5);
            let mut m = EwRidge::new(cfg).unwrap();
            m.set_keep_factor(true);
            m.set_dynamic_prior(Some(vec![bp.clone()])).unwrap();

            for (x, y, w) in &rows {
                m.step(x, &[Some(*y)], 1.0, *w);
            }
            let got = &m.coefficients().unwrap()[0];
            let (exp_beta, a_inv, w_tot, n_kish) = unstd_oracle(
                &rows,
                decay,
                fit_intercept,
                0.25,
                ridge_scale,
                Some(c0),
                bp,
                4.5,
                inc_int,
            );
            let kc = exp_beta.len();
            for i in 0..kc {
                assert!(
                    (got[i] - exp_beta[i]).abs() < 1e-11,
                    "beta mismatch mode=({fit_intercept},{ridge_scale},{inc_int}) i={i}: got={} exp={}",
                    got[i],
                    exp_beta[i]
                );
            }

            // Verify exact coef_variance against (w_tot / n_kish) * diag(A_full^-1).
            let crate::CoefVariance::PerNoise(cv) = m.coef_variance().unwrap() else {
                panic!("expected PerNoise");
            };
            for i in 0..kc {
                let exp_v = (w_tot / n_kish) * a_inv[i * kc + i];
                assert!(
                    (cv[0][i] - exp_v).abs() < 1e-11,
                    "coef_variance mismatch mode=({fit_intercept},{ridge_scale},{inc_int}) i={i}: got={} exp={exp_v}",
                    cv[0][i]
                );
            }

            // Verify exact row_error_inflation against sqrt(1 + (w_tot / n_kish) * z^T A_full^-1 z).
            let x_probe = vec![0.8, -1.2, 0.4];
            let mut row_infl = Vec::new();
            assert!(m.row_error_inflation_into(&x_probe, 1.0, &mut row_infl));
            let mut z = Vec::new();
            if fit_intercept {
                z.push(1.0);
            }
            z.extend_from_slice(&x_probe);
            let mut quad = 0.0;
            for i in 0..kc {
                for j in 0..kc {
                    quad += z[i] * a_inv[i * kc + j] * z[j];
                }
            }
            let exp_infl = (1.0 + (w_tot / n_kish) * quad).sqrt();
            assert!(
                (row_infl[0] - exp_infl).abs() < 1e-11,
                "row_error_inflation mismatch mode=({fit_intercept},{ridge_scale},{inc_int}): got={} exp={exp_infl}",
                row_infl[0]
            );

            // Verify exact edf = tr(W S A_full^-1) via error_inflation_into.
            let mut infl = Vec::new();
            assert!(m.error_inflation_into(&mut infl));
            let mut pen_tr = 0.0;
            let ps = if matches!(decay, Decay::Halflife(h) if h.is_finite()) {
                decay.factor(1.0).powi(rows.len() as i32)
            } else {
                1.0
            };
            for i in 0..kc {
                let r_pen = if ridge_scale {
                    ps * 0.25
                } else if fit_intercept && i == 0 {
                    0.0
                } else {
                    w_tot * 0.25
                };
                let p_pen = if fit_intercept && !inc_int && i == 0 {
                    0.0
                } else {
                    4.5
                };
                pen_tr += (r_pen + p_pen) * a_inv[i * kc + i];
            }
            let exp_edf = (kc as f64) - pen_tr;
            let exp_avg_infl = (1.0 + exp_edf / n_kish).sqrt();
            assert!(
                (infl[0] - exp_avg_infl).abs() < 1e-11,
                "edf/error_inflation mismatch mode=({fit_intercept},{ridge_scale},{inc_int}): got={} exp={exp_avg_infl}",
                infl[0]
            );
        }
    }
}

#[test]
fn standardized_shift_and_scale_invariance_and_oracle() {
    let rows0 = synthetic_rows(30, 3, 99, 0.0);
    let shift = 1e7;
    let rows_shifted: Vec<(Vec<f64>, f64, f64)> = rows0
        .iter()
        .map(|(x, y, w)| (x.iter().map(|&v| v + shift).collect(), *y, *w))
        .collect();
    let bp0 = vec![1.2, 0.5, -0.8, 0.3];
    // Shifted equivalent prior: b0' = b0 - sum(shift * b_s).
    let bp_shifted = vec![
        bp0[0] - shift * (bp0[1] + bp0[2] + bp0[3]),
        bp0[1],
        bp0[2],
        bp0[3],
    ];

    for &inc_int in &[true, false] {
        let mut cfg = base_cfg(3, 1);
        cfg.standardize = true;
        cfg.prior_weight = Some(6.0);
        cfg.prior_include_intercept = inc_int;
        let mut m0 = EwRidge::new(cfg.clone()).unwrap();
        let mut ms = EwRidge::new(cfg).unwrap();
        m0.set_keep_factor(true);
        ms.set_keep_factor(true);
        m0.set_dynamic_prior(Some(vec![bp0.clone()])).unwrap();
        ms.set_dynamic_prior(Some(vec![bp_shifted.clone()]))
            .unwrap();
        for (r0, rs) in rows0.iter().zip(&rows_shifted) {
            m0.step(&r0.0, &[Some(r0.1)], 1.0, r0.2);
            ms.step(&rs.0, &[Some(rs.1)], 1.0, rs.2);
        }
        let b0 = &m0.coefficients().unwrap()[0];
        let bs = &ms.coefficients().unwrap()[0];
        // Slopes invariant under huge 1e7 level shift!
        for i in 1..=3 {
            assert!(
                (b0[i] - bs[i]).abs() < 1e-7,
                "standardized slope not shift-invariant at 1e7: inc_int={inc_int} i={i} {} vs {}",
                b0[i],
                bs[i]
            );
        }
        // Predictions at corresponding probe points agree tightly!
        let x0 = vec![0.4, -0.7, 1.1];
        let xs: Vec<f64> = x0.iter().map(|&v| v + shift).collect();
        let p0 = m0.predict(&x0, 1.0).pred[0];
        let ps = ms.predict(&xs, 1.0).pred[0];
        assert!(
            (p0 - ps).abs() < 1e-6,
            "standardized pred shift mismatch inc_int={inc_int}: {p0} vs {ps}"
        );
    }
}

#[test]
fn linear_constraints_and_qp_bounds_compose_with_prior() {
    let rows = synthetic_rows(20, 3, 7, 0.0);
    let bp = vec![0.5, 0.4, 0.4, 0.2];

    // Linear equality constraint: sum of slopes = 1.0.
    let lc = LinearConstraints {
        matrix: vec![vec![1.0, 1.0, 1.0]],
        rhs: vec![1.0],
    };
    for &standardize in &[false, true] {
        let mut cfg = base_cfg(3, 1);
        cfg.standardize = standardize;
        cfg.constraints = Some(lc.clone());
        cfg.prior_weight = Some(8.0);
        let mut m = EwRidge::new(cfg).unwrap();
        m.set_keep_factor(true);
        m.set_dynamic_prior(Some(vec![bp.clone()])).unwrap();
        for (x, y, w) in &rows {
            m.step(x, &[Some(*y)], 1.0, *w);
        }
        let b = &m.coefficients().unwrap()[0];
        assert!(
            (b[1] + b[2] + b[3] - 1.0).abs() < 1e-11,
            "linear constraint violated (std={standardize}): sum={}",
            b[1] + b[2] + b[3]
        );
    }

    // QP bounds + sum: 0 <= beta_s <= 0.6, sum = 1.0.
    let con = Constraint {
        lo: vec![0.0, 0.0, 0.0],
        hi: vec![0.6, 0.6, 0.6],
        sum: Some(1.0),
    };
    for &(fit_intercept, standardize, ridge_scale, inc_int) in &[
        (true, false, false, true),
        (true, false, false, false),
        (true, true, false, true),
        (false, false, false, true),
        (false, true, false, true),
        (true, false, true, true),
    ] {
        let mut cfg = base_cfg(3, 1);
        cfg.fit_intercept = fit_intercept;
        cfg.standardize = standardize;
        cfg.ridge_scale = ridge_scale;
        cfg.prior_include_intercept = inc_int;
        cfg.constraint = Some(con.clone());
        cfg.prior_weight = Some(5.0);
        let mut m = EwRidge::new(cfg).unwrap();
        let p = if fit_intercept {
            bp.clone()
        } else {
            bp[1..].to_vec()
        };
        m.set_dynamic_prior(Some(vec![p])).unwrap();
        for (x, y, w) in &rows {
            m.step(x, &[Some(*y)], 1.0, *w);
        }
        let b = &m.coefficients().unwrap()[0];
        let off = usize::from(fit_intercept);
        let sum: f64 = b[off..].iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "QP sum violated: {sum}");
        for &v in &b[off..] {
            assert!((-1e-9..=0.6 + 1e-9).contains(&v), "QP bound violated: {v}");
        }
    }
}

#[test]
fn cold_start_limits_and_zero_weight_immutability() {
    let bp = vec![2.0, 0.5, -1.0];
    let mut cfg = base_cfg(2, 1);
    cfg.ridge = vec![0.0];
    cfg.prior_weight = Some(10.0);
    let mut m = EwRidge::new(cfg).unwrap();
    m.set_keep_factor(true);
    m.set_step_priors_flat(Some(&bp), Some(&bp));

    // At W = 0: cold-start predicts dot(bp, [1, x]) and inflation is 1.0.
    let x0 = vec![1.0, 2.0];
    let mut infl = Vec::new();
    assert!(m.error_inflation_into(&mut infl));
    assert_eq!(infl[0], 1.0);
    let mut row_infl = Vec::new();
    assert!(m.row_error_inflation_into(&x0, 1.0, &mut row_infl));
    assert_eq!(row_infl[0], 1.0);

    // Zero-weight row: predicts from cold-start prior, does NOT mutate
    // moments, dynamic_prior, or beta (Rule 9: clock/decay only).
    let s0 = m.step(&x0, &[Some(999.0)], 1.0, 0.0);
    assert!((s0.pred[0] - (2.0 + 0.5 * 1.0 - 1.0 * 2.0)).abs() < 1e-12);
    assert_eq!(m.n_eff(), 0.0);
    assert!(m.dynamic_prior().is_none());
    assert!(m.coefficients().is_none());
    // Following positive-weight row gives exact same result as without the zero-weight row.
    let mut cfg_c = base_cfg(2, 1);
    cfg_c.ridge = vec![0.0];
    cfg_c.prior_weight = Some(10.0);
    let mut m_clean = EwRidge::new(cfg_c).unwrap();
    m_clean.set_keep_factor(true);
    m.set_step_priors_flat(Some(&bp), Some(&bp));
    m_clean.set_step_priors_flat(Some(&bp), Some(&bp));
    m.step(&[0.5, -0.5], &[Some(1.0)], 1.0, 1.0);
    m_clean.step(&[0.5, -0.5], &[Some(1.0)], 1.0, 1.0);
    assert_eq!(
        m.coefficients().unwrap()[0],
        m_clean.coefficients().unwrap()[0]
    );

    // Shrinkage limit: huge prior_weight -> beta -> bp; tiny prior_weight -> OLS.
    let rows = synthetic_rows(30, 2, 123, 0.0);
    let mut cfg_hi = base_cfg(2, 1);
    cfg_hi.ridge = vec![0.0];
    cfg_hi.prior_weight = Some(1e9);
    let mut m_hi = EwRidge::new(cfg_hi).unwrap();
    m_hi.set_dynamic_prior(Some(vec![bp.clone()])).unwrap();
    for (x, y, w) in &rows {
        m_hi.step(x, &[Some(*y)], 1.0, *w);
    }
    let b_hi = &m_hi.coefficients().unwrap()[0];
    for i in 0..3 {
        assert!(
            (b_hi[i] - bp[i]).abs() < 1e-5,
            "high tau failed to match prior"
        );
    }
}

#[test]
fn state_roundtrip_and_byte_identity_without_prior() {
    // 1. Without prior_weight, compact & named msgpack round-trip cleanly.
    let mut cfg_plain = base_cfg(2, 1);
    cfg_plain.prior_weight = None;
    let mut plain = EwRidge::new(cfg_plain).unwrap();
    plain.step(&[1.0, 2.0], &[Some(3.0)], 1.0, 1.0);
    let bytes_named = rmp_serde::to_vec_named(&plain.state()).unwrap();
    let s_named: State = rmp_serde::from_slice(&bytes_named).unwrap();
    let restored_plain = EwRidge::restore(&s_named).unwrap();
    assert_eq!(
        restored_plain.predict(&[2.0, 3.0], 1.0).pred[0].to_bits(),
        plain.predict(&[2.0, 3.0], 1.0).pred[0].to_bits()
    );
    assert_eq!(restored_plain.coefficients(), plain.coefficients());

    // 2. With prior_weight + dynamic_prior + window, BOTH compact & named round-trip!
    let mut cfg_p = base_cfg(2, 1);
    cfg_p.prior_weight = Some(3.5);
    cfg_p.prior_include_intercept = false;
    cfg_p.window = Some(10.0);
    let mut mp = EwRidge::new(cfg_p).unwrap();
    mp.set_keep_factor(true);
    mp.set_dynamic_prior(Some(vec![vec![0.1, 0.2, 0.3]]))
        .unwrap();
    for i in 0..5 {
        mp.step(
            &[i as f64, (i * i) as f64 + 0.5],
            &[Some(1.0 + i as f64)],
            1.0,
            1.0,
        );
    }
    for bytes in [
        rmp_serde::to_vec(&mp.state()).unwrap(),
        rmp_serde::to_vec_named(&mp.state()).unwrap(),
    ] {
        let s: State = rmp_serde::from_slice(&bytes).unwrap();
        let mut restored = EwRidge::restore(&s).unwrap();
        assert_eq!(restored.coefficients(), mp.coefficients());
        assert_eq!(restored.dynamic_prior(), mp.dynamic_prior());
        assert_eq!(
            restored.predict(&[2.0, 3.0], 1.0).pred[0].to_bits(),
            mp.predict(&[2.0, 3.0], 1.0).pred[0].to_bits()
        );
        // Stepping both identically continues bit-for-bit identically.
        let mut mp_clone = mp.clone();
        let a = restored.step(&[1.5, -0.5], &[Some(4.0)], 1.0, 1.0);
        let b = mp_clone.step(&[1.5, -0.5], &[Some(4.0)], 1.0, 1.0);
        assert_eq!(a.pred[0].to_bits(), b.pred[0].to_bits());
        assert_eq!(restored.coefficients(), mp_clone.coefficients());
    }

    // 3. Damaged dynamic_prior shape is refused on restore.
    let mut damaged = mp.state();
    if let ModelState::EwRidge(ref mut m) = damaged.model {
        m.dynamic_prior = Some(crate::model::Fit(vec![vec![1.0]]));
    }
    assert!(EwRidge::restore(&damaged).is_err());
}

#[test]
fn cfg_validation_refuses_invalid_prior_options() {
    for &bad in &[0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut cfg = base_cfg(2, 1);
        cfg.prior_weight = Some(bad);
        let err = EwRidge::new(cfg).unwrap_err();
        assert!(err.contains("prior_weight"), "err={err}");
    }
    let mut cfg = base_cfg(2, 1);
    cfg.prior_weight = None;
    cfg.prior_include_intercept = false;
    assert!(EwRidge::new(cfg).is_err());

    let mut cfg = base_cfg(2, 1);
    cfg.fit_intercept = false;
    cfg.prior_include_intercept = false;
    assert!(EwRidge::new(cfg).is_err());

    let mut cfg = base_cfg(2, 1);
    cfg.feature_sets = vec![("a".into(), vec![0])];
    assert!(EwRidge::new(cfg).is_err());
}
