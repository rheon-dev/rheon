//! Unit tests for bounded and fixed-sum [`EwRidge`] (`EwRidgeCfg::constraint`).

use crate::constraint::Constraint;
use crate::gramfit::{Comoments, GramArrays, qp_solve};
use crate::qp::kkt_residuals;
use crate::{
    Decay, EwRidge, EwRidgeCfg, LinearConstraints, ModelState, OnlineModel, State, TargetGaps,
};

fn lcg(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    ((*s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

fn base_cfg(k: usize, m: usize, constraint: Constraint) -> EwRidgeCfg {
    EwRidgeCfg {
        n_features: k,
        n_targets: m,
        fit_intercept: true,
        decay: Decay::Halflife(40.0),
        ridge: vec![1e-3],
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
        constraint: Some(constraint),
    }
}

#[test]
fn simplex_nonneg_box_and_box_sum_satisfy_kkt_and_disable_ls_uncertainty() {
    let k = 4;
    let constraints = [
        Constraint {
            lo: vec![0.0; k],
            hi: vec![f64::INFINITY; k],
            sum: Some(1.0),
        },
        Constraint {
            lo: vec![0.0; k],
            hi: vec![f64::INFINITY; k],
            sum: None,
        },
        Constraint {
            lo: vec![-0.4; k],
            hi: vec![0.4; k],
            sum: None,
        },
        Constraint {
            lo: vec![0.0; k],
            hi: vec![0.45; k],
            sum: Some(1.0),
        },
    ];

    for (ci, con) in constraints.iter().enumerate() {
        for &(fit_intercept, standardize, ridge_scale) in &[
            (true, false, false),
            (true, true, false),
            (false, false, false),
            (false, true, false),
            (true, false, true),
            (false, false, true),
        ] {
            let mut cfg = base_cfg(k, 1, con.clone());
            cfg.fit_intercept = fit_intercept;
            cfg.standardize = standardize;
            cfg.ridge_scale = ridge_scale;
            cfg.ridge = vec![0.05];

            let mut m = EwRidge::new(cfg).unwrap();
            let mut seed = 1001u64 + ci as u64 * 17;
            for t in 0..80 {
                let x = [
                    0.5 + 0.8 * lcg(&mut seed),
                    -0.3 + 1.5 * lcg(&mut seed),
                    0.2 + 0.4 * lcg(&mut seed),
                    -0.1 + 2.0 * lcg(&mut seed),
                ];
                let y =
                    1.2 + 1.4 * x[0] - 0.8 * x[1] + 0.5 * x[2] - 0.3 * x[3] + 0.1 * lcg(&mut seed);
                m.step(&x, &[Some(y)], t as f64, 1.0);
            }

            let coef = &m.coefficients().unwrap()[0];
            let off = usize::from(fit_intercept);
            let slopes = &coef[off..];

            // Bounds and fixed sum must hold to numerical tolerance.
            for (i, &s_i) in slopes.iter().enumerate() {
                assert!(
                    s_i >= con.lo[i] - 1e-11 && s_i <= con.hi[i] + 1e-11,
                    "case {ci} ({fit_intercept},{standardize},{ridge_scale}) slope[{i}]={s_i}"
                );
            }
            if let Some(s) = con.sum {
                let got: f64 = slopes.iter().sum();
                assert!(
                    (got - s).abs() < 1e-11,
                    "case {ci} ({fit_intercept},{standardize},{ridge_scale}) sum={got} want={s}"
                );
            }

            // Linear least-squares uncertainty / leverage / support_coef are disabled.
            let mut buf = vec![0.0; 1];
            assert!(!m.error_inflation_into(&mut buf));
            assert!(!m.error_inflation_gate_into(&[0.1, 0.2, 0.3, 0.4], 1.0, &mut buf, 2.0));
            assert!(!m.row_error_inflation_into(&[0.1, 0.2, 0.3, 0.4], 1.0, &mut buf));
            let mut sup = Vec::new();
            assert!(!m.support_coef_into(&mut sup));
            assert!(m.coef_variance().is_none());
            assert!(m.support_coef().is_none());

            // Verify exact KKT optimality directly against the model's live Gram.
            let cov = m.gram(0);
            let mut h = vec![0.0; k * k];
            let mut c = vec![0.0; k];
            if fit_intercept && !ridge_scale {
                let raw_c = &m.acc.cross.c[0];
                for i in 0..k {
                    for j in 0..k {
                        h[i * k + j] = cov.cov(i + 1, j + 1);
                    }
                    let diag_pen = if standardize {
                        0.05 * cov.cov(i + 1, i + 1)
                    } else {
                        0.05
                    };
                    h[i * k + i] += diag_pen;
                    c[i] = raw_c[i + 1];
                }
                let a_eq = vec![1.0; k];
                let eq = con.sum.map(|s| (a_eq.as_slice(), s));
                let kkt = kkt_residuals(&h, &c, k, &con.lo, &con.hi, eq, slopes);
                assert!(
                    kkt.max_err() < 1e-10,
                    "case {ci} ({fit_intercept},{standardize},{ridge_scale}) kkt={kkt:?}"
                );
            }
        }
    }
}

#[test]
fn coef_prior_window_multi_target_and_zero_weight_rows() {
    let k = 3;
    let con = Constraint {
        lo: vec![0.0, 0.0, 0.0],
        hi: vec![0.7, 0.7, 0.7],
        sum: Some(1.0),
    };
    let mut cfg = base_cfg(k, 2, con.clone());
    cfg.coef_prior = Some(vec![vec![0.0, 0.5, 0.3, 0.2], vec![0.0, 0.2, 0.3, 0.5]]);
    cfg.ridge = vec![0.1, 1.0];
    cfg.window = Some(30.0);
    cfg.standardize = true;

    let mut m_clean = EwRidge::new(cfg.clone()).unwrap();
    let mut m_zero = EwRidge::new(cfg).unwrap();
    let mut seed = 77u64;
    for t in 0..60 {
        let x = [lcg(&mut seed), lcg(&mut seed), lcg(&mut seed)];
        let y0 = 0.3 + 0.9 * x[0] - 0.4 * x[1] + 0.2 * x[2] + 0.05 * lcg(&mut seed);
        let y1 = -0.2 - 0.5 * x[0] + 0.8 * x[1] + 0.4 * x[2] + 0.05 * lcg(&mut seed);
        let o0 = m_clean.step(&x, &[Some(y0), Some(y1)], 1.0, 1.0);
        let o1 = m_zero.step(&x, &[Some(y0), Some(y1)], 1.0, 1.0);
        for (a, b) in o0.pred.iter().zip(o1.pred.iter()) {
            assert_eq!(a.to_bits(), b.to_bits(), "row {t} pred");
        }
        // Insert a zero-weight row at the same timestamp with extreme values (Rule 9).
        if t % 5 == 2 {
            let _ = m_zero.step(&[1e6, -1e6, 1e6], &[Some(1e9), Some(-1e9)], 0.0, 0.0);
        }
    }
    let c0 = m_clean.coefficients().unwrap();
    let c1 = m_zero.coefficients().unwrap();
    assert_eq!(c0.len(), 4);
    for (slot, (v0, v1)) in c0.iter().zip(c1.iter()).enumerate() {
        assert_eq!(v0, v1, "slot {slot} altered by zero-weight row");
        let slopes = &v0[1..];
        let s: f64 = slopes.iter().sum();
        assert!((s - 1.0).abs() < 1e-11);
        for &b in slopes {
            assert!((-1e-11..=0.7 + 1e-11).contains(&b));
        }
    }
}

#[test]
fn collinear_and_constant_features_at_zero_ridge_solve_cleanly() {
    let k = 4;
    let con = Constraint {
        lo: vec![0.0; k],
        hi: vec![f64::INFINITY; k],
        sum: Some(1.0),
    };
    for standardize in [false, true] {
        let mut cfg = base_cfg(k, 1, con.clone());
        cfg.ridge = vec![0.0];
        cfg.standardize = standardize;
        cfg.decay = Decay::Halflife(f64::INFINITY);
        let mut m = EwRidge::new(cfg).unwrap();
        let mut seed = 31337u64;
        for t in 0..50 {
            let x0 = lcg(&mut seed);
            let x1 = lcg(&mut seed);
            let x = [x0, x1, x0, 5.0]; // x2 == x0 (collinear), x3 == 5 (constant)
            let y = 1.0 + 0.7 * x0 + 0.3 * x1 + 0.02 * lcg(&mut seed);
            m.step(&x, &[Some(y)], t as f64, 1.0);
        }
        let coef = &m.coefficients().unwrap()[0];
        assert!(
            coef.iter().all(|v| v.is_finite()),
            "standardize={standardize}: {coef:?}"
        );
        let slopes = &coef[1..];
        assert!((slopes.iter().sum::<f64>() - 1.0).abs() < 1e-10);
        assert!(slopes.iter().all(|&v| v >= -1e-10));
        assert_eq!(m.solve_failures, 0);
    }
}

#[test]
fn validation_refusals_and_corrupted_state_restore_rejections() {
    let simplex = Constraint {
        lo: vec![0.0, 0.0],
        hi: vec![f64::INFINITY, f64::INFINITY],
        sum: Some(1.0),
    };
    // Refuses combining `constraints` with `constraint`.
    let mut c1 = base_cfg(2, 1, simplex.clone());
    c1.constraints = Some(LinearConstraints {
        matrix: vec![vec![1.0, 1.0]],
        rhs: vec![1.0],
    });
    assert!(EwRidge::new(c1).is_err());

    // Refuses combining `feature_sets` with `constraint`.
    let mut c2 = base_cfg(2, 1, simplex.clone());
    c2.feature_sets = vec![("a".into(), vec![0])];
    assert!(EwRidge::new(c2).is_err());

    // Refuses infeasible sum.
    let c3 = base_cfg(
        2,
        1,
        Constraint {
            lo: vec![0.0, 0.0],
            hi: vec![0.4, 0.4],
            sum: Some(1.0),
        },
    );
    assert!(EwRidge::new(c3).is_err());

    // STATE-3: restore rejects corrupted bounded states.
    let mut ok = EwRidge::new(base_cfg(2, 1, simplex)).unwrap();
    for t in 0..10 {
        ok.step(
            &[t as f64, (t * t) as f64],
            &[Some(t as f64)],
            t as f64,
            1.0,
        );
    }
    let bad_cases: [fn(&mut EwRidge); 4] = [
        |m| m.cfg.constraint.as_mut().unwrap().lo = vec![0.0],
        |m| {
            let c = m.cfg.constraint.as_mut().unwrap();
            c.lo = vec![0.8, 0.0];
            c.hi = vec![0.2, 1.0];
        },
        |m| {
            let c = m.cfg.constraint.as_mut().unwrap();
            c.hi = vec![0.3, 0.3];
            c.sum = Some(1.0);
        },
        |m| {
            m.cfg.constraints = Some(LinearConstraints {
                matrix: vec![vec![1.0, 1.0]],
                rhs: vec![1.0],
            });
        },
    ];
    for (i, corrupt) in bad_cases.into_iter().enumerate() {
        let mut bad = ok.clone();
        corrupt(&mut bad);
        let st = State::new(ModelState::EwRidge(Box::new(bad)));
        assert!(
            EwRidge::restore(&st).is_err(),
            "corrupt case {i} must fail restore"
        );
    }
}

#[test]
fn gram_qp_solve_matches_ewridge_to_the_bit() {
    let k = 3;
    let con = Constraint {
        lo: vec![0.0, -0.2, 0.0],
        hi: vec![0.8, 0.6, 0.9],
        sum: Some(1.0),
    };
    for &(fit_intercept, standardize) in
        &[(true, false), (true, true), (false, false), (false, true)]
    {
        let mut cfg = base_cfg(k, 1, con.clone());
        cfg.fit_intercept = fit_intercept;
        cfg.standardize = standardize;
        cfg.ridge = vec![1e-4, 0.2];
        let mut m = EwRidge::new(cfg).unwrap();
        let mut seed = 42u64;
        for t in 0..60 {
            let x = [
                1.0 + 0.7 * lcg(&mut seed),
                -2.0 + 3.0 * lcg(&mut seed),
                0.5 + 0.2 * lcg(&mut seed),
            ];
            let y = 0.7 + 1.2 * x[0] - 0.5 * x[1] + 0.6 * x[2] + 0.1 * lcg(&mut seed);
            m.step(&x, &[Some(y)], t as f64, 1.0);
        }

        let (parts, tm) = m.gram_parts();
        let p = &parts[0];
        let tm = tm.unwrap();
        let k_tot = m.cfg().k_total();
        let means: Vec<f64> = (0..k_tot).map(|i| p.cov.mean(i)).collect();
        let mut comoments = vec![0.0; k_tot * k_tot];
        for i in 0..k_tot {
            for j in 0..k_tot {
                comoments[i * k_tot + j] = p.cov.cov(i, j);
            }
        }
        let cross_flat: Vec<f64> = p.cross_moments.iter().flatten().copied().collect();
        let means_t_flat: Vec<f64> = p.means_by_target.iter().flatten().copied().collect();
        let cross_c_flat: Vec<f64> = p.cross_centred.iter().flatten().copied().collect();
        let t_means = tm.means();
        let t_vars = tm.vars();
        let t_kish: Vec<f64> = tm
            .n_kish(&p.target_weights)
            .into_iter()
            .map(|v| v.unwrap_or(f64::NAN))
            .collect();
        let ga = GramArrays {
            k: k_tot,
            means: &means,
            comoments: Comoments::Full(&comoments),
            cross_moments: &cross_flat,
            means_by_target: &means_t_flat,
            cross_centred: &cross_c_flat,
            weight_sum: p.cov.n_eff(),
            target_weights: &p.target_weights,
            target_means: t_means,
            target_vars: t_vars,
            target_n_kish: &t_kish,
            within: None,
        };
        let icept = fit_intercept.then_some(0);
        let slots: Vec<usize> = if fit_intercept {
            (1..k_tot).collect()
        } else {
            (0..k_tot).collect()
        };
        let res = qp_solve(&ga, 0, &slots, icept, &[1e-4, 0.2], standardize, &con);
        assert!(res.hit_max_iter.iter().all(|&h| !h));

        let coefs = m.coefficients().unwrap();
        for (ri, want) in coefs.iter().enumerate() {
            let got = &res.coefs[ri * k_tot..(ri + 1) * k_tot];
            for c in 0..k_tot {
                assert_eq!(
                    got[c].to_bits(),
                    want[c].to_bits(),
                    "({fit_intercept},{standardize}) ridge[{ri}] col {c}: got={} want={}",
                    got[c],
                    want[c]
                );
            }
        }
    }
}
