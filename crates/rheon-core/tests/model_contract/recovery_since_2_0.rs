// ---------------------------------------------------------------------------
// Bounded input, the kinds and forms added since 2.0 (PLAN 578j, TEST-10)
// ---------------------------------------------------------------------------
//
// The roll call held every model with a recovery test to a predict parity
// test, and not every model to a recovery test: thirteen kinds and four
// forms had none. Each is here, against the clean twin where the model
// forgets, and held to a finite state that goes on learning where it keeps
// every row (a digest, a count).

/// The script with `d` features: the two it has, and `d - 2` more drawn
/// from them, a bound where the row's first feature is one.
fn widened(rows: Vec<Row>, d: usize) -> Vec<Row> {
    rows.into_iter()
        .map(|mut r| {
            let (a, b) = (r.x[0], r.x[1]);
            r.x = (0..d)
                .map(|j| match j {
                    0 => a,
                    1 => b,
                    _ if a.abs() >= BOUND => a,
                    _ => a * (j as f64).sin() + b * (j as f64).cos() + 0.01 * j as f64,
                })
                .collect();
            r
        })
        .collect()
}

/// A model that keeps every row, or a count, where a row at the bound, or
/// of weight at the bound, stays in what it holds for good (a digest's
/// weight, a count, a start's buffer): through the script its `n_eff` stays
/// finite, no prediction is infinite, a withheld one is NaN, and its state
/// restores as itself at the end, the soundness a load holds it to.
fn stays_finite_over<M: OnlineModel>(rows: &[Row], build: impl Fn() -> M) {
    let mut model = build();
    let kind = model.state().model.kind();
    for (i, r) in rows.iter().enumerate() {
        let d = if i == 0 { 0.0 } else { 1.0 };
        let a = model.step(&r.x, &r.y, d, r.w);
        assert!(a.n_eff.is_finite(), "{kind}: n_eff is {} at row {i}", a.n_eff);
        for (slot, p) in a.pred.iter().enumerate() {
            assert!(!p.is_infinite(), "{kind}: slot {slot} predicts {p} at row {i}");
        }
    }
    // Compared as written, so a NaN a state holds by design (a statistic
    // not yet computed) compares as itself.
    let state = model.state();
    let back = M::restore(&state).unwrap_or_else(|e| panic!("{kind}: its state does not restore: {e}"));
    assert_eq!(
        format!("{:?}", back.state()),
        format!("{state:?}"),
        "{kind}: a restore moved the state"
    );
}

#[test]
fn expectile_recovers_from_bounded_extremes() {
    for cfg in [expectile_cfg(), expectile_rows_cfg(), expectile_clock_cfg()] {
        recovers_from_bounded_extremes(
            move || Expectile::new(cfg.clone()).unwrap(),
            2,
            Recovery::Twin(1e-6),
        );
    }
}

#[test]
fn garch_recovers_from_bounded_extremes() {
    stays_finite_over(&bounded_script(1), || Garch::new(garch_cfg()).unwrap());
}

#[test]
fn dcc_recovers_from_bounded_extremes() {
    stays_finite_over(&bounded_script(2), || Dcc::new(dcc_cfg()).unwrap());
}

#[test]
fn pca_recovers_from_bounded_extremes() {
    recovers_from_bounded_extremes(
        || StreamPca::new(pca_cfg()).unwrap(),
        0,
        Recovery::TwinWithheld(1e-6),
    );
}

#[test]
fn power_variation_recovers_from_bounded_extremes() {
    recovers_from_bounded_extremes(
        || PowerVariation::new(power_variation_cfg()).unwrap(),
        2,
        Recovery::Twin(1e-6),
    );
}

#[test]
fn tdigest_recovers_from_bounded_extremes() {
    stays_finite_over(&bounded_script(2), || TDigestModel::new(tdigest_cfg()).unwrap());
}

#[test]
fn ddsketch_recovers_from_bounded_extremes() {
    stays_finite_over(&bounded_script(2), || DdSketchModel::new(ddsketch_cfg()).unwrap());
}

#[test]
fn ar_recovers_from_bounded_extremes() {
    recovers_from_bounded_extremes(|| Ar::new(ar_cfg()).unwrap(), 2, Recovery::Twin(1e-6));
}

#[test]
fn spectral_recovers_from_bounded_extremes() {
    recovers_from_bounded_extremes(|| Spectral::new(spectral_cfg()).unwrap(), 2, Recovery::Twin(1e-6));
}

#[test]
fn vector_ar_recovers_from_bounded_extremes() {
    recovers_from_bounded_extremes(
        || VectorAr::new(vector_ar_cfg()).unwrap(),
        2,
        Recovery::Twin(1e-6),
    );
}

/// `isotonic` keeps every distinct feature value, with its weight and its
/// weighted sum: a row at the bound stays among them (a point at `1e100`
/// beside the ordinary ones), so it is held to a finite state that goes on
/// learning, as a digest is.
#[test]
fn isotonic_recovers_from_bounded_extremes() {
    stays_finite_over(&bounded_script(2), || Isotonic::new(isotonic_cfg()).unwrap());
}

#[test]
fn audit_recovers_from_bounded_extremes() {
    stays_finite_over(&bounded_script(0), || Audit::new(audit_cfg()).unwrap());
}

#[test]
fn kalman_recovers_from_bounded_extremes_in_square_root_form() {
    let cfg = KalmanCfg {
        half_life: vec![HALFLIFE],
        ..kalman_cfg()
    };
    recovers_from_bounded_extremes(
        move || {
            Kalman::new(cfg.clone())
                .unwrap()
                .with_covariance_form(CovarianceForm::SquareRoot)
                .unwrap()
        },
        2,
        Recovery::Twin(1e-9),
    );
}

/// `hmm`'s factor form, from 13 features: as `hmm_recovers_from_bounded_extremes`
/// holds the two-feature form, the state stays finite through the script,
/// `p` is a distribution wherever it is reported, and the filter goes on to
/// follow a clean two-blob stream (a filter's `p` is path-dependent, so no
/// twin agrees with it).
#[test]
fn hmm_recovers_from_bounded_extremes_in_factor_form() {
    let d = 13;
    let mut covs = vec![0.0; 2 * d * d];
    for k in 0..2 {
        for j in 0..d {
            covs[k * d * d + j * d + j] = 1.0;
        }
    }
    let cfg = HmmCfg {
        n_features: d,
        means: Some([vec![-1.0; d], vec![1.0; d]].concat()),
        covs: Some(covs),
        ..hmm_cfg()
    };
    let mut m = Hmm::new(cfg).unwrap();
    for (i, r) in widened(bounded_script(0), d).iter().enumerate() {
        let step = m.step(&r.x, &r.y, 1.0, r.w);
        assert!(step.n_eff.is_finite(), "row {i}");
        let p = &step.pred[..2];
        if p.iter().all(|v| v.is_finite()) {
            assert!(p.iter().all(|v| (0.0..=1.0).contains(v)), "row {i}: {p:?}");
            assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-12, "row {i}: {p:?}");
        } else {
            assert!(step.pred.iter().all(|v| v.is_nan()), "row {i}");
        }
    }
    for s in 0..2 {
        assert!(m.state_means(s).iter().all(|v| v.is_finite()), "state {s}'s mean");
    }
    let mut seed = 991u64;
    let mut reported = 0;
    for i in 0..2000 {
        let c = if (i / 50) % 2 == 0 { -6.0 } else { 6.0 };
        let x: Vec<f64> = (0..d).map(|_| c + lcg(&mut seed)).collect();
        let step = m.step(&x, &[], 1.0, 1.0);
        reported += usize::from(i > 1000 && step.pred[2 * 2].is_finite());
    }
    assert!(reported > 0, "the filter reports a state on a clean stream");
}

#[test]
fn hawkes_recovers_from_bounded_extremes_in_a_window() {
    for cfg in [hawkes_windowed_cfg(), hawkes_clock_cfg()] {
        recovers_from_bounded_extremes(move || Hawkes::new(cfg.clone()).unwrap(), 2, Recovery::Twin(1e-6));
    }
}

/// Its moments are kept exactly past the largest double, at a scale of a
/// power of two, so a moment of order 5 of a row at the bound of weight at
/// the bound (`1e600` on the sum scale) washes out as the stream's own
/// does: over the 2,012 half-lives the order asks for, where its square
/// (`1e300`) needs [`bounded_script`]'s 1,000 (PLAN 578j, TEST-10).
#[test]
fn ewm_moments_recovers_from_bounded_extremes() {
    recovers_over(
        &bounded_script_for_order(0, 5),
        || EwmMoments::new(ewm_moments_cfg()).unwrap(),
        Recovery::Twin(1e-6),
    );
}

/// In a window the row at the bound leaves: the window's summaries are
/// merged at the larger of their scales, and once it has left every
/// output is the twin's, a window of the same rows.
#[test]
fn ewm_moments_recovers_from_bounded_extremes_in_a_window() {
    let rows = EwmMomentsCfg {
        decay: Decay::Halflife(f64::INFINITY),
        window_rows: Some(12),
        ..ewm_moments_cfg()
    };
    let clock = EwmMomentsCfg {
        clocked: true,
        window_rows: None,
        window_clock: Some(12.5),
        ..rows.clone()
    };
    for cfg in [rows, clock] {
        recovers_from_bounded_extremes(
            move || EwmMoments::new(cfg.clone()).unwrap(),
            0,
            Recovery::Twin(1e-6),
        );
    }
}

#[test]
fn realized_recovers_from_bounded_extremes() {
    recovers_from_bounded_extremes(|| Realized::new(realized_cfg()).unwrap(), 2, Recovery::Twin(1e-6));
}

/// The windows take each term back as its row leaves: a quarticity past the
/// largest double, held at it, and a return near the bound leaving a window
/// of ordinary ones, whose compensated sums kept only rounding of what was
/// left (PLAN 578j, found beside TEST-10's `realized`).
#[test]
fn realized_recovers_from_bounded_extremes_in_a_window() {
    for cfg in [realized_rows_cfg(), realized_clock_cfg()] {
        recovers_from_bounded_extremes(move || Realized::new(cfg.clone()).unwrap(), 2, Recovery::Twin(1e-6));
    }
}

/// The kinds with no recovery test here, each with why. Every other kind
/// the contract probes (`PROBED`) has `<kind>_recovers_from_bounded_extremes`
/// or, where a bound row is no evidence at all, `<kind>_is_indifferent_to_bounded_extremes`.
/// Each kind below fails its test on this build; the test is the owning
/// fix's to land (PLAN 578j, TEST-10).
const RECOVERY_PENDING: &[(&str, &str)] = &[
    (
        "glm",
        "a target at the bound freezes a coefficient for good (the review's GLM-5, PLAN 578b): \
         30,000 rows on its slot 0 predicts 1.1e13 where the twin predicts 0.29",
    ),
];

#[test]
fn every_model_has_a_recovery_test_or_a_reason() {
    // The roll call that held a recovery test to its predict parity held no
    // model to a recovery test: thirteen kinds had none (TEST-10).
    let source = contract_source();
    let src = source.as_str();
    let mut missing = Vec::new();
    for (_, kind) in PROBED.iter().filter(|(_, k)| *k != "ew_cov_accumulator") {
        let tested = src.contains(&format!("fn {kind}_recovers_from_bounded_extremes()"))
            || src.contains(&format!("fn {kind}_is_indifferent_to_bounded_extremes()"));
        let pending = RECOVERY_PENDING.iter().any(|(k, _)| k == kind);
        assert!(!(tested && pending), "{kind}: tested, so out of RECOVERY_PENDING");
        if !tested && !pending {
            missing.push(*kind);
        }
    }
    assert!(missing.is_empty(), "no recovery test: {missing:?}");
}

/// Fourth powers, a tripower quarticity and a kernel over returns of twenty
/// steps, under a decay and in both windows: `|r|⁴` of a return near the
/// bound, and the square of a twenty-step return of them, pass the largest
/// double. The probes' powers are squares, which never do, so the sums'
/// overflow to `inf`, which no decay washes out, went unseen (PLAN 578j,
/// found beside TEST-10's `ewm_moments` and `realized`).
#[test]
fn power_variation_recovers_from_bounded_extremes_at_higher_powers() {
    let mp = |powers: &[f64], normalize: bool| PvTerm {
        kind: PvTermKind::Multipower {
            powers: powers.to_vec(),
            normalize,
        },
        scale: 1,
    };
    let wide = PvTerm {
        kind: PvTermKind::Kernel {
            kernel: PvKernel::Parzen,
            bandwidth: 2,
            flat_top: false,
            center: true,
        },
        scale: 20,
    };
    let quartic = |base: PowerVariationCfg| PowerVariationCfg {
        terms: vec![
            mp(&[4.0], false),
            mp(&[2.0], false),
            mp(&[4.0 / 3.0; 3], true),
            mp(&[2.0, 2.0], true),
            wide.clone(),
        ],
        ..base
    };
    for cfg in [
        quartic(power_variation_cfg()),
        quartic(power_variation_rows_cfg()),
        quartic(power_variation_clock_cfg()),
    ] {
        recovers_from_bounded_extremes(
            move || PowerVariation::new(cfg.clone()).unwrap(),
            2,
            Recovery::Twin(1e-6),
        );
    }
}

