use super::*;

fn lcg(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

fn default_cfg(k: usize, m: usize) -> HoeffdingTreeRegressorCfg {
    HoeffdingTreeRegressorCfg {
        n_features: k,
        n_targets: m,
        ..Default::default()
    }
}

/// Pure-Rust `libm` only: no platform `f64::{ln, log2, exp, ...}` calls in
/// the regressor's library code, so every split and state fixture is
/// bit-identical across macOS, Linux and Windows.
#[test]
fn no_platform_math_library_is_called() {
    let files = [
        (
            "hoeffding_tree_regressor.rs",
            include_str!("../hoeffding_tree_regressor.rs"),
        ),
        ("qo.rs", include_str!("qo.rs")),
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

#[test]
fn validation_refuses_bad_settings() {
    assert!(HoeffdingTreeRegressor::new(default_cfg(0, 1)).is_err());
    assert!(HoeffdingTreeRegressor::new(default_cfg(2, 0)).is_err());
    assert!(
        HoeffdingTreeRegressor::new(HoeffdingTreeRegressorCfg {
            grace_period: 0.0,
            ..default_cfg(2, 1)
        })
        .is_err()
    );
    assert!(
        HoeffdingTreeRegressor::new(HoeffdingTreeRegressorCfg {
            delta: 0.0,
            ..default_cfg(2, 1)
        })
        .is_err()
    );
    assert!(
        HoeffdingTreeRegressor::new(HoeffdingTreeRegressorCfg {
            delta: 1.0,
            ..default_cfg(2, 1)
        })
        .is_err()
    );
    assert!(
        HoeffdingTreeRegressor::new(HoeffdingTreeRegressorCfg {
            tau: -0.1,
            ..default_cfg(2, 1)
        })
        .is_err()
    );
    assert!(
        HoeffdingTreeRegressor::new(HoeffdingTreeRegressorCfg {
            min_child_weight: 0.5,
            ..default_cfg(2, 1)
        })
        .is_err()
    );
    assert!(
        HoeffdingTreeRegressor::new(HoeffdingTreeRegressorCfg {
            radius: 0.0,
            ..default_cfg(2, 1)
        })
        .is_err()
    );
    assert!(
        HoeffdingTreeRegressor::new(HoeffdingTreeRegressorCfg {
            max_slots: 1,
            ..default_cfg(2, 1)
        })
        .is_err()
    );
    assert!(
        HoeffdingTreeRegressor::new(HoeffdingTreeRegressorCfg {
            max_state_bytes: Some(0),
            ..default_cfg(2, 1)
        })
        .is_err()
    );
    assert!(
        HoeffdingTreeRegressor::new(HoeffdingTreeRegressorCfg {
            max_state_bytes: Some(crate::clock_span::STATE_BYTES_CEILING + 1),
            ..default_cfg(2, 1)
        })
        .is_err()
    );
    let min_b = default_cfg(4, 2).min_state_bytes();
    assert!(
        HoeffdingTreeRegressor::new(HoeffdingTreeRegressorCfg {
            max_state_bytes: Some(min_b - 1),
            ..default_cfg(4, 2)
        })
        .is_err()
    );
    assert!(
        HoeffdingTreeRegressor::new(HoeffdingTreeRegressorCfg {
            max_state_bytes: Some(min_b),
            ..default_cfg(4, 2)
        })
        .is_ok()
    );
}

#[test]
fn qo_fixes_partition_and_left_child_stats() {
    // Feed a clean step function y = -5 for x < 0 and y = +5 for x > 0.
    // After the first split, the left child must start with mean ~ -5 (not 0)
    // and weight equal to the observations sent left (not the full parent).
    let mut m = HoeffdingTreeRegressor::new(HoeffdingTreeRegressorCfg {
        n_features: 2,
        n_targets: 1,
        grace_period: 100.0,
        radius: 0.25,
        max_slots: 32,
        ..Default::default()
    })
    .unwrap();
    let mut seed = 2026u64;
    let mut split_at = None;
    for i in 0..400 {
        let x0 = lcg(&mut seed);
        let x1 = lcg(&mut seed);
        let y = if x0 <= 0.0 { -5.0 } else { 5.0 };
        m.step(&[x0, x1], &[Some(y)], 1.0, 1.0);
        if m.n_leaves(0) > 1 {
            split_at = Some(i);
            break;
        }
    }
    assert!(
        split_at.is_some(),
        "tree should have split on step function"
    );
    let tree = &m.trees[0];
    let left_mean = tree.leaves[0].stats.mean;
    let right_mean = tree.leaves[1].stats.mean;
    let total_w = tree.leaves[0].stats.n + tree.leaves[1].stats.n;
    assert!(
        (total_w - tree.target_weight).abs() < 1e-9,
        "children weights must sum to parent weight: {total_w} vs {}",
        tree.target_weight
    );
    assert!(
        left_mean < -3.0,
        "left child mean should be near -5, got {left_mean}"
    );
    assert!(
        right_mean > 3.0,
        "right child mean should be near +5, got {right_mean}"
    );
}

#[test]
fn learns_multi_target_with_sparse_nulls() {
    let mut m = HoeffdingTreeRegressor::new(HoeffdingTreeRegressorCfg {
        n_features: 3,
        n_targets: 2,
        grace_period: 50.0,
        ..Default::default()
    })
    .unwrap();
    let mut seed = 314u64;
    let mut mse0 = 0.0;
    let mut mse1 = 0.0;
    let mut n1 = 0usize;
    for i in 0..2500 {
        let x = [lcg(&mut seed), lcg(&mut seed), lcg(&mut seed)];
        let y0 = if x[0] > 0.0 { 2.0 } else { -2.0 } + 0.1 * lcg(&mut seed);
        let y1 = if x[1] > 0.0 { 3.0 } else { -3.0 } + 0.1 * lcg(&mut seed);
        let y1_opt = (i % 2 == 0).then_some(y1);
        let s = m.step(&x, &[Some(y0), y1_opt], 1.0, 1.0);
        if i >= 1500 {
            mse0 += (s.pred[0] - y0).powi(2);
            if y1_opt.is_some() {
                mse1 += (s.pred[1] - y1).powi(2);
                n1 += 1;
            }
        }
    }
    assert!(m.n_leaves(0) >= 2 && m.n_leaves(1) >= 2);
    assert!(mse0 / 1000.0 < 0.25, "mse0={}", mse0 / 1000.0);
    assert!(mse1 / (n1 as f64) < 0.35, "mse1={}", mse1 / (n1 as f64));
    let mut tw = Vec::new();
    assert!(m.target_n_eff_into(&mut tw));
    assert_eq!(tw, vec![2500.0, 1250.0]);
}

#[test]
fn zero_weight_rows_leave_state_identical_to_the_byte() {
    let cfg = HoeffdingTreeRegressorCfg {
        n_features: 4,
        n_targets: 2,
        grace_period: 40.0,
        ..Default::default()
    };
    let mut plain = HoeffdingTreeRegressor::new(cfg.clone()).unwrap();
    let mut spliced = HoeffdingTreeRegressor::new(cfg).unwrap();
    let mut seed = 888u64;
    // Zero-weight first row with wild features/targets.
    let s0 = spliced.step(
        &[1e5, -1e5, 42.0, -42.0],
        &[Some(1e4), Some(-1e4)],
        0.0,
        0.0,
    );
    assert!(s0.pred.iter().all(|v| v.is_nan()));
    assert_eq!(s0.n_eff, 0.0);
    for i in 0..1000 {
        if i % 5 == 0 {
            let wild = [
                lcg(&mut seed) * 100.0,
                lcg(&mut seed) * 100.0,
                lcg(&mut seed) * 100.0,
                lcg(&mut seed) * 100.0,
            ];
            let p_before = spliced.predict(&wild, 1.0);
            let s_zero = spliced.step(&wild, &[Some(999.0), Some(-999.0)], 1.0, 0.0);
            assert_eq!(
                p_before
                    .pred
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                s_zero.pred.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
        }
        let x = [
            lcg(&mut seed),
            lcg(&mut seed),
            lcg(&mut seed),
            lcg(&mut seed),
        ];
        let y0 = 2.0 * x[0] - x[1];
        let y1 = x[2] * x[2] - x[3];
        let a = plain.step(&x, &[Some(y0), Some(y1)], 1.0, 1.0);
        let b = spliced.step(&x, &[Some(y0), Some(y1)], 1.0, 1.0);
        assert_eq!(
            a.pred.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            b.pred.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        assert_eq!(a.n_eff.to_bits(), b.n_eff.to_bits());
    }
    assert_eq!(
        rmp_serde::to_vec(&plain.state()).unwrap(),
        rmp_serde::to_vec(&spliced.state()).unwrap()
    );
}

#[test]
fn memory_cap_never_exceeded_and_resumes_to_the_byte() {
    let cap = 8_000u64; // fits ~2 active leaves at k=4, max_slots=16
    let cfg = HoeffdingTreeRegressorCfg {
        n_features: 4,
        n_targets: 1,
        grace_period: 30.0,
        tau: 0.1,
        max_slots: 16,
        max_state_bytes: Some(cap),
        ..Default::default()
    };
    let mut whole = HoeffdingTreeRegressor::new(cfg.clone()).unwrap();
    let mut resumed = HoeffdingTreeRegressor::new(cfg).unwrap();
    let mut seed = 1234u64;
    let mut saw_capped = false;
    for i in 0..3500 {
        let x: Vec<f64> = (0..4).map(|_| lcg(&mut seed) * 2.0).collect();
        let y = x[0] * x[0] + 2.0 * x[1] - x[2] * x[3];
        let a = whole.step(&x, &[Some(y)], 1.0, 1.0);
        let b = resumed.step(&x, &[Some(y)], 1.0, 1.0);
        assert_eq!(
            a.pred.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            b.pred.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        assert!(
            whole.state_bytes() <= cap && whole.reserved_bytes() <= cap,
            "row {i}: bytes={} reserved={} cap={cap}",
            whole.state_bytes(),
            whole.reserved_bytes()
        );
        if whole.trees[0].leaves.iter().any(|l| l.capped) {
            saw_capped = true;
        }
        if i % 333 == 0 {
            let bytes = rmp_serde::to_vec(&resumed.state()).unwrap();
            let decoded: State = rmp_serde::from_slice(&bytes).unwrap();
            resumed = HoeffdingTreeRegressor::restore(&decoded).unwrap();
        }
    }
    assert!(saw_capped, "cap should have deactivated some leaves");
    assert_eq!(
        rmp_serde::to_vec(&whole.state()).unwrap(),
        rmp_serde::to_vec(&resumed.state()).unwrap()
    );
}

#[test]
fn max_depth_drops_observers_at_cap_depth() {
    let mut m = HoeffdingTreeRegressor::new(HoeffdingTreeRegressorCfg {
        n_features: 3,
        n_targets: 1,
        grace_period: 25.0,
        tau: 0.1,
        max_depth: Some(2),
        ..Default::default()
    })
    .unwrap();
    let mut seed = 77u64;
    for _ in 0..2500 {
        let x = [lcg(&mut seed), lcg(&mut seed), lcg(&mut seed)];
        let y = x[0] * 3.0 + x[1] * 2.0 - x[2];
        m.step(&x, &[Some(y)], 1.0, 1.0);
    }
    assert!(m.n_leaves(0) >= 2);
    for lf in &m.trees[0].leaves {
        assert!(lf.depth <= 2);
        if lf.depth == 2 && lf.stats.n > 0.0 && !lf.active {
            assert!(lf.obs.is_none());
        }
    }
}

#[test]
fn a_state_of_the_wrong_shape_is_refused() {
    let mut m = HoeffdingTreeRegressor::new(HoeffdingTreeRegressorCfg {
        n_features: 3,
        n_targets: 1,
        grace_period: 25.0,
        max_slots: 8,
        ..Default::default()
    })
    .unwrap();
    let mut seed = 19u64;
    for _ in 0..400 {
        let x = [lcg(&mut seed), lcg(&mut seed), lcg(&mut seed)];
        let y = 2.0 * x[0] - x[1];
        m.step(&x, &[Some(y)], 1.0, 1.0);
    }
    assert!(m.n_leaves(0) >= 2);
    let check_refused = |bad: HoeffdingTreeRegressor| {
        let err = HoeffdingTreeRegressor::restore(&bad.state()).unwrap_err();
        assert!(matches!(err, StateError::Invalid(_)), "{err:?}");
    };

    // Wrong target tree count.
    let mut bad = m.clone();
    bad.trees.pop();
    check_refused(bad);

    // Out-of-bounds feature index.
    let mut bad = m.clone();
    bad.trees[0].branches[0].feature = 99;
    check_refused(bad);

    // Cycle in tree topology.
    let mut bad = m.clone();
    bad.trees[0].branches[0].right = NodeRef::Branch(0);
    check_refused(bad);

    // Unsorted QO slots.
    let mut bad = m.clone();
    if let Some(lf) = bad.trees[0].leaves.iter_mut().find(|l| l.obs.is_some()) {
        let slots = &mut lf.obs.as_mut().unwrap()[0].slots;
        if slots.len() >= 2 {
            slots.swap(0, 1);
            check_refused(bad);
        }
    }

    // Slot count exceeding max_slots.
    let mut bad = m.clone();
    if let Some(lf) = bad.trees[0].leaves.iter_mut().find(|l| l.obs.is_some()) {
        let slots = &mut lf.obs.as_mut().unwrap()[0].slots;
        let mut next_idx = slots.last().map_or(0, |s| s.idx) + 1;
        while slots.len() <= 8 {
            slots.push(qo::QoSlot {
                idx: next_idx,
                x_mean: 0.0,
                y_var: TargetVar {
                    n: 1.0,
                    mean: 0.0,
                    s: 0.0,
                },
            });
            next_idx += 1;
        }
        check_refused(bad);
    }

    // Active and capped simultaneously.
    let mut bad = m.clone();
    bad.trees[0].leaves[0].active = true;
    bad.trees[0].leaves[0].capped = true;
    check_refused(bad);

    // Negative weight.
    let mut bad = m.clone();
    bad.w_sum = -1.0;
    check_refused(bad);

    // Reserved bytes past cap.
    let mut bad = m.clone();
    bad.cfg.max_state_bytes = Some(bad.cfg.min_state_bytes());
    for lf in &mut bad.trees[0].leaves {
        lf.active = true;
        lf.capped = false;
    }
    check_refused(bad);
}

#[test]
fn adaptive_recheck_skips_when_uncapped_and_scales_when_capped() {
    let tight = HoeffdingTreeRegressorCfg {
        n_features: 2,
        n_targets: 1,
        grace_period: 15.2,
        delta: 0.05,
        max_slots: 8,
        max_state_bytes: Some(reg_struct_bytes(16, 15) + 2 * reg_obs_reserve(2, 8)),
        ..Default::default()
    };
    let mut whole = HoeffdingTreeRegressor::new(tight.clone()).unwrap();
    let mut split_m = HoeffdingTreeRegressor::new(tight).unwrap();
    let mut saw_capped = false;
    for i in 0..800 {
        let f0 = ((i * 37) % 100) as f64 / 50.0 - 1.0;
        let f1 = ((i * 53) % 100) as f64 / 50.0 - 1.0;
        let y = 3.0 * f0 - 2.0 * f1 + (if f0 > 0.0 { 4.0 } else { -4.0 });
        whole.step(&[f0, f1], &[Some(y)], 1.0, 1.0);
        if i == 377 {
            let bytes = rmp_serde::to_vec_named(&split_m.state()).unwrap();
            let st: State = rmp_serde::from_slice(&bytes).unwrap();
            split_m = HoeffdingTreeRegressor::restore(&st).unwrap();
        }
        split_m.step(&[f0, f1], &[Some(y)], 1.0, 1.0);
        if whole.trees[0].leaves.iter().any(|l| l.capped) {
            saw_capped = true;
            let active = whole.trees[0]
                .leaves
                .iter()
                .filter(|l| l.active)
                .count()
                .max(1) as u64;
            assert_eq!(whole.trees[0].recheck_interval(&whole.cfg), 16 * active);
        }
    }
    assert!(saw_capped, "expected at least one capped leaf");
    assert_eq!(
        whole, split_m,
        "save/load across adaptive recheck must stay bit-identical"
    );
}
