use super::*;

fn lcg(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

fn default_cfg(k: usize, c: usize) -> HoeffdingTreeClassifierCfg {
    HoeffdingTreeClassifierCfg {
        n_features: k,
        n_classes: c,
        ..Default::default()
    }
}

/// Pure-Rust `libm` only: no platform `f64::{ln, log2, exp, ...}` calls in
/// the classifier's library code, so every split and state fixture is
/// bit-identical across macOS, Linux and Windows.
#[test]
fn no_platform_math_library_is_called() {
    let files = [
        (
            "hoeffding_tree_classifier.rs",
            include_str!("../hoeffding_tree_classifier.rs"),
        ),
        ("observer.rs", include_str!("observer.rs")),
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
    assert!(HoeffdingTreeClassifier::new(default_cfg(0, 2)).is_err());
    assert!(HoeffdingTreeClassifier::new(default_cfg(2, 1)).is_err());
    assert!(
        HoeffdingTreeClassifier::new(HoeffdingTreeClassifierCfg {
            split_criterion: SplitCriterion::Hellinger,
            n_classes: 3,
            ..default_cfg(2, 3)
        })
        .is_err()
    );
    assert!(
        HoeffdingTreeClassifier::new(HoeffdingTreeClassifierCfg {
            grace_period: 0.0,
            ..default_cfg(2, 2)
        })
        .is_err()
    );
    assert!(
        HoeffdingTreeClassifier::new(HoeffdingTreeClassifierCfg {
            delta: 1.0,
            ..default_cfg(2, 2)
        })
        .is_err()
    );
    assert!(
        HoeffdingTreeClassifier::new(HoeffdingTreeClassifierCfg {
            delta: 0.0,
            ..default_cfg(2, 2)
        })
        .is_err()
    );
    assert!(
        HoeffdingTreeClassifier::new(HoeffdingTreeClassifierCfg {
            tau: -0.01,
            ..default_cfg(2, 2)
        })
        .is_err()
    );
    assert!(
        HoeffdingTreeClassifier::new(HoeffdingTreeClassifierCfg {
            nb_threshold: -1.0,
            ..default_cfg(2, 2)
        })
        .is_err()
    );
    assert!(
        HoeffdingTreeClassifier::new(HoeffdingTreeClassifierCfg {
            n_splits: 0,
            ..default_cfg(2, 2)
        })
        .is_err()
    );
    assert!(
        HoeffdingTreeClassifier::new(HoeffdingTreeClassifierCfg {
            min_branch_fraction: 0.5,
            ..default_cfg(2, 2)
        })
        .is_err()
    );
    assert!(
        HoeffdingTreeClassifier::new(HoeffdingTreeClassifierCfg {
            max_share_to_split: 0.0,
            ..default_cfg(2, 2)
        })
        .is_err()
    );
    assert!(
        HoeffdingTreeClassifier::new(HoeffdingTreeClassifierCfg {
            max_state_bytes: Some(0),
            ..default_cfg(2, 2)
        })
        .is_err()
    );
    assert!(
        HoeffdingTreeClassifier::new(HoeffdingTreeClassifierCfg {
            max_state_bytes: Some(crate::clock_span::STATE_BYTES_CEILING + 1),
            ..default_cfg(2, 2)
        })
        .is_err()
    );
    // Below one active leaf's packed bytes: refused.
    let min_b = default_cfg(2, 2).min_state_bytes();
    assert!(
        HoeffdingTreeClassifier::new(HoeffdingTreeClassifierCfg {
            max_state_bytes: Some(min_b - 1),
            ..default_cfg(2, 2)
        })
        .is_err()
    );
    assert!(
        HoeffdingTreeClassifier::new(HoeffdingTreeClassifierCfg {
            max_state_bytes: Some(min_b),
            ..default_cfg(2, 2)
        })
        .is_ok()
    );
}

#[test]
fn learns_axis_aligned_split_for_every_criterion_and_predictor() {
    for criterion in [
        SplitCriterion::InfoGain,
        SplitCriterion::Gini,
        SplitCriterion::Hellinger,
    ] {
        for pred_kind in [
            LeafPrediction::MajorityClass,
            LeafPrediction::NaiveBayes,
            LeafPrediction::NaiveBayesAdaptive,
        ] {
            let mut m = HoeffdingTreeClassifier::new(HoeffdingTreeClassifierCfg {
                n_features: 3,
                n_classes: 2,
                grace_period: 50.0,
                split_criterion: criterion,
                leaf_prediction: pred_kind,
                ..Default::default()
            })
            .unwrap();
            let mut seed = 101u64;
            let mut correct = 0usize;
            let mut total = 0usize;
            for i in 0..2000 {
                let x = [lcg(&mut seed), lcg(&mut seed), lcg(&mut seed)];
                let y = if x[0] > 0.1 { 1.0 } else { 0.0 };
                let s = m.step(&x, &[Some(y)], 1.0, 1.0);
                if i == 0 {
                    assert!(s.pred.iter().all(|v| v.is_nan()));
                } else {
                    assert!(s.pred.iter().all(|v| v.is_finite()));
                    let sum_p = s.pred[1] + s.pred[2];
                    assert!((sum_p - 1.0).abs() < 1e-12, "sum_p={sum_p}");
                    if i >= 1000 {
                        total += 1;
                        if s.pred[0] == y {
                            correct += 1;
                        }
                    }
                }
            }
            assert!(
                m.n_leaves() >= 2,
                "{criterion:?}/{pred_kind:?} did not split"
            );
            let acc = correct as f64 / total as f64;
            assert!(
                acc > 0.90,
                "{criterion:?}/{pred_kind:?} accuracy too low: {acc}"
            );
        }
    }
}

#[test]
fn null_split_merit_zero_prevents_spurious_single_feature_noise_splits() {
    // Study 503 §3.4: on 1 feature with 45% label noise, null_merit = 0.0
    // avoids river's 60 spurious splits.
    let mut m = HoeffdingTreeClassifier::new(HoeffdingTreeClassifierCfg {
        n_features: 1,
        n_classes: 2,
        grace_period: 50.0,
        merit_preprune: true,
        ..Default::default()
    })
    .unwrap();
    let mut seed = 503u64;
    for _ in 0..4000 {
        let x = [lcg(&mut seed)];
        let pure = lcg(&mut seed) > 0.0;
        let y = if pure { 1.0 } else { 0.0 };
        m.step(&x, &[Some(y)], 1.0, 1.0);
    }
    assert!(
        m.n_leaves() <= 5,
        "pure noise with 1 feature should barely split under null_merit=0.0, got {} leaves",
        m.n_leaves()
    );
}

#[test]
fn zero_weight_rows_leave_state_identical_to_the_byte() {
    let cfg = HoeffdingTreeClassifierCfg {
        n_features: 4,
        n_classes: 3,
        grace_period: 40.0,
        ..Default::default()
    };
    let mut plain = HoeffdingTreeClassifier::new(cfg.clone()).unwrap();
    let mut spliced = HoeffdingTreeClassifier::new(cfg).unwrap();
    let mut seed = 777u64;
    // Zero-weight first row with wild features.
    let s0 = spliced.step(&[1e5, -1e5, 50.0, -50.0], &[Some(2.0)], 0.0, 0.0);
    assert!(s0.pred.iter().all(|v| v.is_nan()));
    assert_eq!(s0.n_eff, 0.0);
    for i in 0..1000 {
        if i % 7 == 0 {
            let wild = [
                lcg(&mut seed) * 100.0,
                lcg(&mut seed) * 100.0,
                lcg(&mut seed) * 100.0,
                lcg(&mut seed) * 100.0,
            ];
            let p_before = spliced.predict(&wild, 1.0);
            let s_zero = spliced.step(&wild, &[Some((i % 3) as f64)], 1.0, 0.0);
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
        let c = if x[0] > 0.3 {
            2.0
        } else if x[1] < -0.2 {
            1.0
        } else {
            0.0
        };
        let a = plain.step(&x, &[Some(c)], 1.0, 1.0);
        let b = spliced.step(&x, &[Some(c)], 1.0, 1.0);
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
    let cap = 1_500u64; // tiny cap: forces leaf deactivation and reactivation
    let cfg = HoeffdingTreeClassifierCfg {
        n_features: 5,
        n_classes: 2,
        grace_period: 30.0,
        tau: 0.1,
        max_state_bytes: Some(cap),
        ..Default::default()
    };
    let mut whole = HoeffdingTreeClassifier::new(cfg.clone()).unwrap();
    let mut resumed = HoeffdingTreeClassifier::new(cfg).unwrap();
    let mut seed = 909u64;
    let mut saw_capped = false;
    for i in 0..3500 {
        let x: Vec<f64> = (0..5).map(|_| lcg(&mut seed)).collect();
        let y = if (x[0] > 0.0) ^ (x[1] > 0.0) ^ (x[2] > 0.2) {
            1.0
        } else {
            0.0
        };
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
        if whole.leaves.iter().any(|l| l.capped) {
            saw_capped = true;
        }
        if i % 333 == 0 {
            let bytes = rmp_serde::to_vec(&resumed.state()).unwrap();
            let decoded: State = rmp_serde::from_slice(&bytes).unwrap();
            resumed = HoeffdingTreeClassifier::restore(&decoded).unwrap();
        }
    }
    assert!(saw_capped, "cap should have deactivated some leaves");
    assert_eq!(
        rmp_serde::to_vec(&whole.state()).unwrap(),
        rmp_serde::to_vec(&resumed.state()).unwrap()
    );
}

#[test]
fn max_depth_is_honored() {
    let mut m = HoeffdingTreeClassifier::new(HoeffdingTreeClassifierCfg {
        n_features: 4,
        n_classes: 2,
        grace_period: 25.0,
        tau: 0.1,
        max_depth: Some(2),
        ..Default::default()
    })
    .unwrap();
    let mut seed = 42u64;
    for _ in 0..3000 {
        let x: Vec<f64> = (0..4).map(|_| lcg(&mut seed)).collect();
        let y = if (x[0] > 0.0) ^ (x[1] > 0.0) {
            1.0
        } else {
            0.0
        };
        m.step(&x, &[Some(y)], 1.0, 1.0);
    }
    assert!(m.n_leaves() >= 2);
    assert!(m.leaves.iter().all(|l| l.depth <= 2));
}

#[test]
fn a_state_of_the_wrong_shape_is_refused() {
    let mut m = HoeffdingTreeClassifier::new(HoeffdingTreeClassifierCfg {
        n_features: 3,
        n_classes: 2,
        grace_period: 25.0,
        ..Default::default()
    })
    .unwrap();
    let mut seed = 11u64;
    for _ in 0..400 {
        let x = [lcg(&mut seed), lcg(&mut seed), lcg(&mut seed)];
        let y = if x[0] > 0.0 { 1.0 } else { 0.0 };
        m.step(&x, &[Some(y)], 1.0, 1.0);
    }
    assert!(m.n_leaves() >= 2);
    let check_refused = |bad: HoeffdingTreeClassifier| {
        let err = HoeffdingTreeClassifier::restore(&bad.state()).unwrap_err();
        assert!(matches!(err, StateError::Invalid(_)), "{err:?}");
    };

    // Wrong counts length.
    let mut bad = m.clone();
    bad.leaves[0].counts.pop();
    check_refused(bad);

    // Out-of-bounds branch feature.
    let mut bad = m.clone();
    bad.branches[0].feature = 99;
    check_refused(bad);

    // Cycle in tree topology.
    let mut bad = m.clone();
    bad.branches[0].left = NodeRef::Branch(0);
    check_refused(bad);

    // Active and capped simultaneously.
    let mut bad = m.clone();
    bad.leaves[0].active = true;
    bad.leaves[0].capped = true;
    check_refused(bad);

    // Observer present on inactive leaf.
    let mut bad = m.clone();
    bad.leaves[0].active = false;
    bad.leaves[0].obs = Some(vec![GaussianObs::default(); 6]);
    check_refused(bad);

    // Wrong observer vector length.
    let mut bad = m.clone();
    bad.leaves[0].active = true;
    bad.leaves[0].capped = false;
    bad.leaves[0].obs = Some(vec![GaussianObs::default(); 2]);
    check_refused(bad);

    // Negative weight in state.
    let mut bad = m.clone();
    bad.w_sum = -1.0;
    check_refused(bad);

    // Reserved bytes past cap.
    let mut bad = m.clone();
    bad.cfg.max_state_bytes = Some(bad.cfg.min_state_bytes());
    // Force two active leaves while cap fits only one.
    for lf in &mut bad.leaves {
        lf.active = true;
        lf.capped = false;
    }
    check_refused(bad);
}
