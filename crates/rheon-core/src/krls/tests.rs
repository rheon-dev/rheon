use super::*;
use crate::oracle;

fn lcg(s: &mut u64) -> f64 {
    *s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    let u = ((*s >> 11) as f64) / ((1u64 << 53) as f64);
    2.0 * u - 1.0
}

fn base_cfg() -> KrlsCfg {
    KrlsCfg {
        n_features: 3,
        n_targets: 2,
        fit_intercept: true,
        kernel: KernelKind::Rbf,
        gamma: None,
        degree: None,
        coef0: None,
        ridge: 0.25,
        max_atoms: Some(25),
        ald_threshold: Some(1e-3),
        window_rows: None,
        window_clock: None,
        closed: WindowClosed::Right,
        decay: Decay::Halflife(f64::INFINITY),
        min_weight: 0.0,
        emit_n_atoms: false,
        max_state_bytes: None,
    }
}

/// Independent batch kernel ridge oracle using `crate::oracle::solve_columns`
/// (faer's partial-pivot LU — shares zero arithmetic with `krls`'s Cholesky).
fn batch_kernel_ridge(
    kern: KernelEval,
    x_rows: &[Vec<f64>],
    y_rows: &[Vec<f64>],
    weights: &[f64],
    ridge: f64,
    fit_intercept: bool,
    x_test: &[f64],
) -> Vec<f64> {
    let n = x_rows.len();
    let t_cnt = y_rows[0].len();
    let mut a = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            a[i * n + j] = kern.eval(&x_rows[i], &x_rows[j]);
        }
        a[i * n + i] += ridge / weights[i];
    }
    let mut y_flat = Vec::with_capacity(n * t_cnt);
    for row in y_rows {
        y_flat.extend_from_slice(row);
    }
    let vy = oracle::solve_columns(&a, &y_flat, t_cnt);
    let mut kv = Vec::with_capacity(n);
    for row in x_rows {
        kv.push(kern.eval(row, x_test));
    }
    if !fit_intercept {
        (0..t_cnt)
            .map(|t| (0..n).map(|i| kv[i] * vy[i * t_cnt + t]).sum())
            .collect()
    } else {
        let ones = vec![1.0; n];
        let v1 = oracle::solve(&a, &ones);
        let denom: f64 = v1.iter().copied().sum();
        (0..t_cnt)
            .map(|t| {
                let num: f64 = (0..n).map(|i| vy[i * t_cnt + t]).sum();
                let b = num / denom;
                let dot: f64 = (0..n)
                    .map(|i| kv[i] * (vy[i * t_cnt + t] - v1[i] * b))
                    .sum();
                b + dot
            })
            .collect()
    }
}

/// Independent subset-of-regressors batch oracle on dictionary `D`, solving
/// `(K_{XD}^T U K_{XD} + r K_D) alpha = K_{XD}^T U y` (with optional unpenalized
/// intercept via weighted centering) through `crate::oracle::solve_columns`.
#[allow(clippy::too_many_arguments)]
fn batch_sor(
    kern: KernelEval,
    atoms: &[Vec<f64>],
    x_rows: &[Vec<f64>],
    y_rows: &[Vec<f64>],
    weights: &[f64],
    ridge: f64,
    fit_intercept: bool,
    x_test: &[f64],
) -> Vec<f64> {
    let m = atoms.len();
    let n = x_rows.len();
    let t_cnt = y_rows[0].len();
    let w_total: f64 = weights.iter().copied().sum();
    let mut k_xd = vec![0.0; n * m];
    for i in 0..n {
        for j in 0..m {
            k_xd[i * m + j] = kern.eval(&x_rows[i], &atoms[j]);
        }
    }
    let mut k_bar = vec![0.0; m];
    let mut y_bar = vec![0.0; t_cnt];
    if fit_intercept {
        for i in 0..n {
            let wi = weights[i] / w_total;
            for j in 0..m {
                k_bar[j] += wi * k_xd[i * m + j];
            }
            for t in 0..t_cnt {
                y_bar[t] += wi * y_rows[i][t];
            }
        }
    }
    let mut h = vec![0.0; m * m];
    for a in 0..m {
        for b in 0..m {
            h[a * m + b] = ridge * kern.eval(&atoms[a], &atoms[b]);
        }
    }
    let mut rhs = vec![0.0; m * t_cnt];
    for i in 0..n {
        let wi = weights[i];
        for a in 0..m {
            let da = k_xd[i * m + a] - k_bar[a];
            for b in 0..m {
                let db = k_xd[i * m + b] - k_bar[b];
                h[a * m + b] += wi * da * db;
            }
            for t in 0..t_cnt {
                rhs[a * t_cnt + t] += wi * da * (y_rows[i][t] - y_bar[t]);
            }
        }
    }
    let alpha = oracle::solve_columns(&h, &rhs, t_cnt);
    let kv: Vec<f64> = atoms.iter().map(|d| kern.eval(d, x_test)).collect();
    (0..t_cnt)
        .map(|t| {
            let mut dot = 0.0;
            for a in 0..m {
                dot += (kv[a] - k_bar[a]) * alpha[a * t_cnt + t];
            }
            y_bar[t] + dot
        })
        .collect()
}

#[test]
fn ald_matches_lu_oracle_across_kernels_decay_and_intercept() {
    let mut s = 20261127u64;
    let n = 80;
    let k = 3;
    let xs: Vec<Vec<f64>> = (0..n)
        .map(|_| (0..k).map(|_| lcg(&mut s) * 1.5).collect())
        .collect();
    let ys: Vec<Vec<f64>> = xs
        .iter()
        .map(|x| {
            vec![
                (1.2 * x[0]).sin() + 0.3 * x[1] * x[1] + 2.5 + 0.05 * lcg(&mut s),
                (0.8 * x[2]).cos() - 0.4 * x[0] * x[1] - 1.5 + 0.05 * lcg(&mut s),
            ]
        })
        .collect();
    let ws: Vec<f64> = (0..n).map(|_| 0.5 + 0.4 * (lcg(&mut s) + 1.0)).collect();
    let x_test = vec![0.4, -0.7, 0.25];

    for kernel in [
        KernelKind::Rbf,
        KernelKind::Laplacian,
        KernelKind::Poly,
        KernelKind::Linear,
    ] {
        // Choose cap = 3 for Linear (rank 3) and 10 for others so every atom
        // is admitted on the first `cap` rows (no pre-admission projection ->
        // exact algebraic match to batch subset-of-regressors).
        let cap = if kernel == KernelKind::Linear { 3 } else { 10 };
        for decay in [Decay::Halflife(f64::INFINITY), Decay::Halflife(25.0)] {
            for fit_intercept in [false, true] {
                let cfg = KrlsCfg {
                    kernel,
                    fit_intercept,
                    decay,
                    max_atoms: Some(cap),
                    ald_threshold: Some(0.0),
                    ..base_cfg()
                };
                let kern = cfg.kernel_eval();
                let mut m = Krls::new(cfg).unwrap();
                for i in 0..n {
                    let y = [Some(ys[i][0]), Some(ys[i][1])];
                    m.step(&xs[i], &y, if i == 0 { 0.0 } else { 1.0 }, ws[i]);
                }
                assert_eq!(m.n_atoms(), cap);
                let pred = m.predict(&x_test, 1.0).pred;
                let lam = decay.factor(1.0);
                let u: Vec<f64> = (0..n)
                    .map(|i| ws[i] * lam.powi((n - 1 - i) as i32))
                    .collect();
                let want = batch_sor(kern, &xs[..cap], &xs, &ys, &u, 0.25, fit_intercept, &x_test);
                for t in 0..2 {
                    let gap = (pred[t] - want[t]).abs();
                    assert!(
                        gap < 1e-10 * (1.0 + want[t].abs()),
                        "kernel={kernel:?} decay={decay:?} int={fit_intercept} t={t}: got {} want {} gap {gap:e}",
                        pred[t],
                        want[t]
                    );
                }
            }
        }
    }
}

#[test]
fn window_rows_matches_lu_oracle_across_all_kernels_and_intercepts() {
    let mut s = 999111u64;
    let n = 70;
    let w_rows = 18;
    let k = 3;
    let xs: Vec<Vec<f64>> = (0..n)
        .map(|_| (0..k).map(|_| lcg(&mut s) * 1.4).collect())
        .collect();
    let ys: Vec<Vec<f64>> = xs
        .iter()
        .map(|x| {
            vec![
                (1.1 * x[0]).sin() + 0.4 * x[2] * x[2] + 3.0,
                -0.6 * x[1] + 0.5 * x[0] * x[2] - 2.0,
            ]
        })
        .collect();
    let ws: Vec<f64> = (0..n).map(|_| 0.4 + 0.6 * (lcg(&mut s) + 1.0)).collect();
    let x_test = vec![-0.3, 0.8, -0.5];

    for kernel in [
        KernelKind::Rbf,
        KernelKind::Laplacian,
        KernelKind::Poly,
        KernelKind::Linear,
    ] {
        for fit_intercept in [false, true] {
            let cfg = KrlsCfg {
                kernel,
                fit_intercept,
                max_atoms: None,
                ald_threshold: None,
                window_rows: Some(w_rows),
                emit_n_atoms: true,
                ..base_cfg()
            };
            let kern = cfg.kernel_eval();
            let mut m = Krls::new(cfg).unwrap();
            for i in 0..n {
                let y = [Some(ys[i][0]), Some(ys[i][1])];
                let step = m.step(&xs[i], &y, if i == 0 { 0.0 } else { 1.0 }, ws[i]);
                let expected_atoms = i.min(w_rows);
                assert_eq!(
                    step.extra,
                    Some(Extra::Krls {
                        n_atoms: expected_atoms as f64
                    })
                );
            }
            assert_eq!(m.n_atoms(), w_rows);
            let pred = m.predict(&x_test, 1.0).pred;
            let lo = n - w_rows;
            let want = batch_kernel_ridge(
                kern,
                &xs[lo..],
                &ys[lo..],
                &ws[lo..],
                0.25,
                fit_intercept,
                &x_test,
            );
            for t in 0..2 {
                let gap = (pred[t] - want[t]).abs();
                assert!(
                    gap < 1e-10 * (1.0 + want[t].abs()),
                    "win kernel={kernel:?} int={fit_intercept} t={t}: got {} want {} gap {gap:e}",
                    pred[t],
                    want[t]
                );
            }
        }
    }
}

#[test]
fn window_clock_respects_closed_edges_and_predict_at() {
    let mut s = 777222u64;
    let k = 2;
    let xs: Vec<Vec<f64>> = (0..12)
        .map(|_| (0..k).map(|_| lcg(&mut s)).collect())
        .collect();
    let ys: Vec<Vec<f64>> = xs.iter().map(|x| vec![2.0 * x[0] - x[1] + 1.0]).collect();

    for closed in [WindowClosed::Right, WindowClosed::Both] {
        let cfg = KrlsCfg {
            n_features: k,
            n_targets: 1,
            fit_intercept: false,
            kernel: KernelKind::Rbf,
            max_atoms: None,
            ald_threshold: None,
            window_clock: Some(5.0),
            closed,
            emit_n_atoms: true,
            ..base_cfg()
        };
        let kern = cfg.kernel_eval();
        let mut m = Krls::new(cfg).unwrap();
        // Stamps: 0, 1, 2, ..., 9
        for i in 0..10 {
            m.stamp_next(Stamp::from(i as f64));
            m.step(
                &xs[i],
                &[Some(ys[i][0])],
                if i == 0 { 0.0 } else { 1.0 },
                1.0,
            );
        }
        // Probe at stamp 10.0:
        // Right keeps stamps strictly > 5.0 -> 6,7,8,9 (4 rows)
        // Both keeps stamps >= 5.0 -> 5,6,7,8,9 (5 rows)
        let st = m.predict_at(&xs[10], 1.0, Stamp::from(10.0));
        let lo = if closed == WindowClosed::Right { 6 } else { 5 };
        assert_eq!(st.n_eff, (10 - lo) as f64);
        assert_eq!(
            st.extra,
            Some(Extra::Krls {
                n_atoms: (10 - lo) as f64
            })
        );
        let want = batch_kernel_ridge(
            kern,
            &xs[lo..10],
            &ys[lo..10],
            &vec![1.0; 10 - lo],
            0.25,
            false,
            &xs[10],
        );
        assert!((st.pred[0] - want[0]).abs() < 1e-12);

        // predict_with_floor_at agrees with predict_at at floor 0 and withholds at high floor.
        let open = m.predict_with_floor_at(&xs[10], &[None], 1.0, 0.0, Stamp::from(10.0));
        assert_eq!(open.pred[0].to_bits(), st.pred[0].to_bits());
        let shut = m.predict_with_floor_at(&xs[10], &[None], 1.0, 100.0, Stamp::from(10.0));
        assert!(shut.pred[0].is_nan());
    }
}

#[test]
fn clock_window_held_cap_warns_once_and_refuses_atomically() {
    let cfg = KrlsCfg {
        n_features: 2,
        n_targets: 1,
        max_atoms: None,
        ald_threshold: None,
        window_clock: Some(1000.0),
        max_state_bytes: Some(2_048),
        ..base_cfg()
    };
    let mut m = Krls::new(cfg).unwrap();
    let mut warned_count = 0usize;
    let mut hit_row = None;
    let mut state_before_refusal = None;

    for i in 0..100 {
        let x = [i as f64 * 0.1, (i as f64 * 0.2).sin()];
        let y = [Some(x[0] + x[1])];
        let snap = m.clone();
        m.stamp_next(Stamp::from(i as f64));
        m.step(&x, &y, if i == 0 { 0.0 } else { 1.0 }, 1.0);
        if m.take_state_warning().is_some() {
            warned_count += 1;
        }
        if m.state_over_cap().is_some() {
            hit_row = Some(i);
            state_before_refusal = Some(snap);
            break;
        }
    }
    let _hit = hit_row.expect("burst must cross 2 KB cap");
    assert_eq!(warned_count, 1, "half-cap warning fires exactly once");
    let before = state_before_refusal.unwrap();
    // Atomic refusal: ring, clock, weights, and atoms were NOT mutated by the refusing row!
    assert_eq!(m.n_atoms(), before.n_atoms());
    assert_eq!(m.n_eff().to_bits(), before.n_eff().to_bits());
    assert_eq!(m.clock, before.clock);
    assert_eq!(m.mode, before.mode);
    assert!(m.bytes() <= 2_048);

    // Subsequent rows after cap refusal learn nothing either.
    let over_before = m.state_over_cap();
    m.stamp_next(Stamp::from(200.0));
    m.step(&[1.0, 2.0], &[Some(3.0)], 1.0, 1.0);
    assert_eq!(m.state_over_cap(), over_before);
    assert_eq!(m.n_atoms(), before.n_atoms());
}

#[test]
fn save_restore_is_bit_exact_at_every_depth_and_mode() {
    let mut s = 555666u64;
    let xs: Vec<Vec<f64>> = (0..30)
        .map(|_| (0..3).map(|_| lcg(&mut s)).collect())
        .collect();
    let ys: Vec<Vec<f64>> = xs
        .iter()
        .map(|x| vec![x[0] + 0.5 * x[1], -x[2] + 0.3])
        .collect();

    let cfgs = [
        KrlsCfg {
            max_atoms: Some(8),
            emit_n_atoms: true,
            ..base_cfg()
        },
        KrlsCfg {
            decay: Decay::Halflife(15.0),
            max_atoms: Some(8),
            fit_intercept: false,
            ..base_cfg()
        },
        KrlsCfg {
            max_atoms: None,
            ald_threshold: None,
            window_rows: Some(6),
            ..base_cfg()
        },
        KrlsCfg {
            max_atoms: None,
            ald_threshold: None,
            window_clock: Some(6.5),
            ..base_cfg()
        },
    ];

    for cfg in cfgs {
        for cut in [0, 1, 3, 6, 12] {
            let mut a = Krls::new(cfg.clone()).unwrap();
            for i in 0..cut {
                a.step(
                    &xs[i],
                    &[Some(ys[i][0]), Some(ys[i][1])],
                    if i == 0 { 0.0 } else { 1.0 },
                    1.0,
                );
            }
            let bytes = rmp_serde::to_vec(&a.state()).unwrap();
            let decoded: State = rmp_serde::from_slice(&bytes).unwrap();
            let mut b = Krls::restore(&decoded).unwrap();
            for i in cut..30 {
                let sa = a.step(
                    &xs[i],
                    &[Some(ys[i][0]), Some(ys[i][1])],
                    if i == 0 { 0.0 } else { 1.0 },
                    1.0,
                );
                let sb = b.step(
                    &xs[i],
                    &[Some(ys[i][0]), Some(ys[i][1])],
                    if i == 0 { 0.0 } else { 1.0 },
                    1.0,
                );
                assert_eq!(sa.n_eff.to_bits(), sb.n_eff.to_bits());
                assert_eq!(sa.extra, sb.extra);
                for (pa, pb) in sa.pred.iter().zip(&sb.pred) {
                    assert_eq!(pa.to_bits(), pb.to_bits());
                }
            }
        }
    }
}

#[test]
fn a_state_of_the_wrong_shape_is_refused() {
    // ALD state corruptions.
    let mut m = Krls::new(base_cfg()).unwrap();
    for i in 0..10 {
        let f = i as f64 * 0.3;
        m.step(&[f, -f, f * 0.5], &[Some(f), Some(-f)], 1.0, 1.0);
    }
    assert!(m.n_atoms() >= 3);

    let corrupt = |mutate: &dyn Fn(&mut Krls)| {
        let mut bad = m.clone();
        mutate(&mut bad);
        let st = bad.state();
        assert!(
            Krls::restore(&st).is_err(),
            "corrupted ALD state must be refused"
        );
    };

    corrupt(&|b| b.w_sum = -1.0);
    corrupt(&|b| b.w_sum = f64::NAN);
    corrupt(&|b| {
        b.w_target.pop();
    });
    corrupt(&|b| b.w_target[0] = f64::INFINITY);
    corrupt(&|b| b.clock.own = f64::NAN);
    corrupt(&|b| {
        if let Mode::Ald(a) = &mut b.mode {
            a.atoms.pop();
        }
    });
    corrupt(&|b| {
        if let Mode::Ald(a) = &mut b.mode {
            a.l_d.pop();
        }
    });
    corrupt(&|b| {
        if let Mode::Ald(a) = &mut b.mode {
            a.l_d[0] = 0.0; // non-positive Cholesky diagonal
        }
    });
    corrupt(&|b| {
        if let Mode::Ald(a) = &mut b.mode {
            a.mat[0] = -1.0; // non-positive Cholesky diagonal in undecayed L_s
        }
    });
    corrupt(&|b| {
        if let Mode::Ald(a) = &mut b.mode {
            a.rhs.pop();
        }
    });
    corrupt(&|b| {
        if let Mode::Ald(a) = &mut b.mode {
            a.psi_bar_lo.pop();
        }
    });
    corrupt(&|b| {
        if let Mode::Ald(a) = &mut b.mode {
            a.y_bar_lo.pop();
        }
    });
    corrupt(&|b| {
        if let Mode::Ald(a) = &mut b.mode {
            a.atoms[0] = f64::NAN;
        }
    });
    corrupt(&|b| b.mode = Mode::Window(WindowState::new(2))); // mode/cfg mismatch

    // Window state corruptions.
    let mut wm = Krls::new(KrlsCfg {
        max_atoms: None,
        ald_threshold: None,
        window_clock: Some(10.0),
        ..base_cfg()
    })
    .unwrap();
    for i in 0..5 {
        let f = i as f64 * 0.2;
        wm.stamp_next(Stamp::from(i as f64));
        wm.step(&[f, -f, f * 0.5], &[Some(f), Some(-f)], 1.0, 1.0);
    }
    let corrupt_w = |mutate: &dyn Fn(&mut Krls)| {
        let mut bad = wm.clone();
        mutate(&mut bad);
        let st = bad.state();
        assert!(
            Krls::restore(&st).is_err(),
            "corrupted Window state must be refused"
        );
    };
    corrupt_w(&|b| {
        if let Mode::Window(w) = &mut b.mode {
            w.stamp.pop_back();
        }
    });
    corrupt_w(&|b| {
        if let Mode::Window(w) = &mut b.mode {
            w.w[0] = 0.0;
        }
    });
    corrupt_w(&|b| {
        if let Mode::Window(w) = &mut b.mode {
            w.l_w[0] = 0.0;
        }
    });
    corrupt_w(&|b| {
        if let Mode::Window(w) = &mut b.mode {
            w.x.pop_back();
        }
    });
    corrupt_w(&|b| b.mode = Mode::Ald(AldState::new(&b.cfg)));
}

#[test]
fn validation_refuses_invalid_and_incompatible_settings() {
    let check_err = |cfg: KrlsCfg, needle: &str| {
        let err = Krls::new(cfg).unwrap_err();
        assert!(err.contains(needle), "expected {needle:?} in: {err}");
    };
    check_err(
        KrlsCfg {
            ridge: 0.0,
            ..base_cfg()
        },
        "ridge",
    );
    check_err(
        KrlsCfg {
            ridge: -1.0,
            ..base_cfg()
        },
        "ridge",
    );
    check_err(
        KrlsCfg {
            ridge: f64::INFINITY,
            ..base_cfg()
        },
        "ridge",
    );
    check_err(
        KrlsCfg {
            gamma: Some(0.0),
            ..base_cfg()
        },
        "gamma",
    );
    check_err(
        KrlsCfg {
            degree: Some(2),
            ..base_cfg()
        },
        "degree",
    );
    check_err(
        KrlsCfg {
            coef0: Some(0.5),
            ..base_cfg()
        },
        "coef0",
    );
    check_err(
        KrlsCfg {
            kernel: KernelKind::Linear,
            gamma: Some(0.5),
            ..base_cfg()
        },
        "gamma",
    );
    check_err(
        KrlsCfg {
            kernel: KernelKind::Poly,
            degree: Some(0),
            ..base_cfg()
        },
        "degree",
    );
    check_err(
        KrlsCfg {
            kernel: KernelKind::Poly,
            coef0: Some(-0.1),
            ..base_cfg()
        },
        "coef0",
    );
    check_err(
        KrlsCfg {
            max_atoms: Some(0),
            ..base_cfg()
        },
        "max_atoms",
    );
    check_err(
        KrlsCfg {
            ald_threshold: Some(-0.1),
            ..base_cfg()
        },
        "ald_threshold",
    );
    check_err(
        KrlsCfg {
            ald_threshold: Some(1.0),
            ..base_cfg()
        },
        "ald_threshold",
    );
    check_err(
        KrlsCfg {
            max_atoms: None,
            ald_threshold: None,
            window_rows: Some(0),
            ..base_cfg()
        },
        "window_rows",
    );
    check_err(
        KrlsCfg {
            max_atoms: None,
            ald_threshold: None,
            window_clock: Some(0.0),
            ..base_cfg()
        },
        "window_size",
    );
    check_err(
        KrlsCfg {
            max_atoms: None,
            ald_threshold: None,
            window_rows: Some(10),
            window_clock: Some(10.0),
            ..base_cfg()
        },
        "not both",
    );
    check_err(
        KrlsCfg {
            closed: WindowClosed::Both,
            ..base_cfg()
        },
        "closed",
    );
    check_err(
        KrlsCfg {
            max_atoms: None,
            ald_threshold: None,
            window_rows: Some(10),
            decay: Decay::Halflife(20.0),
            ..base_cfg()
        },
        "no decay",
    );
    check_err(
        KrlsCfg {
            window_rows: Some(10),
            ..base_cfg()
        },
        "max_atoms",
    );
    check_err(
        KrlsCfg {
            max_atoms: None,
            window_rows: Some(10),
            ..base_cfg()
        },
        "ald_threshold",
    );
    check_err(
        KrlsCfg {
            max_atoms: Some(300),
            ..base_cfg()
        },
        "max_state_bytes",
    );
    check_err(
        KrlsCfg {
            max_state_bytes: Some(0),
            ..base_cfg()
        },
        "max_state_bytes",
    );
}

#[test]
fn zero_weight_and_input_bound_stay_finite_for_all_kernels() {
    for kernel in [
        KernelKind::Rbf,
        KernelKind::Laplacian,
        KernelKind::Poly,
        KernelKind::Linear,
    ] {
        let mut m = Krls::new(KrlsCfg {
            kernel,
            decay: Decay::Halflife(10.0),
            ..base_cfg()
        })
        .unwrap();
        // Zero-weight first row: no NaN in state, zero atoms.
        let s0 = m.step(&[1.0, 2.0, 3.0], &[Some(1.0), Some(2.0)], 0.0, 0.0);
        assert_eq!(s0.n_eff, 0.0);
        assert_eq!(m.n_atoms(), 0);
        // Normal rows, then INPUT_BOUND row, then normal row.
        for i in 0..5 {
            let f = (i + 1) as f64 * 0.5;
            m.step(&[f, -f, 0.2 * f], &[Some(f), Some(-f)], 1.0, 1.0);
        }
        let sb = m.step(
            &[crate::INPUT_BOUND, 0.0, 0.0],
            &[Some(1.0), Some(1.0)],
            1.0,
            1.0,
        );
        assert!(sb.n_eff.is_finite());
        assert!(sb.pred.iter().all(|p| !p.is_infinite()));
        let back = Krls::restore(&m.state()).expect("state after INPUT_BOUND restores cleanly");
        assert_eq!(back.state(), m.state());
    }
}

#[test]
fn no_platform_math_library_is_called() {
    let files = [
        ("krls.rs", include_str!("../krls.rs")),
        ("kernel.rs", include_str!("kernel.rs")),
        ("ald.rs", include_str!("ald.rs")),
        ("window.rs", include_str!("window.rs")),
    ];
    let platform = [
        ".ln(", ".ln_1p(", ".log(", ".log2(", ".log10(", ".exp(", ".exp2(", ".exp_m1(", ".powf(",
        ".powi(", ".sin(", ".cos(", ".tan(", ".sinh(", ".cosh(", ".tanh(", ".asin(", ".acos(",
        ".atan(", ".atan2(", ".cbrt(", ".hypot(", "f64::ln", "f64::exp", "f64::pow",
    ];
    for (name, src) in files {
        let code = src.split("#[cfg(test)]").next().unwrap();
        for call in platform {
            assert!(
                !code.contains(call),
                "{name} calls `{call}`: route it through `libm`"
            );
        }
    }
}
