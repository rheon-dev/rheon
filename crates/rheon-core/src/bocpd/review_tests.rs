//! The 2026-11-23 review's `bocpd` findings (PLAN 578f), each test first:
//! it failed on the build before the fix for the review's reason, or,
//! marked `should_panic`, holds a confirmed defect whose fix is the user's
//! call.

use super::tests::cfg;
use super::*;
use crate::OnlineModel;

fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// Rows each of which some run reads at a finite density, at a prior
/// scale of `1e-150`: their runs' spread grows to `1e79`.
const SPREAD: [f64; 5] = [0.0, 1e79, -1e79, 1e79, -1e79];

/// BOCPD-1 (1): a row of a weight so far below the stream's mean that its
/// share `rel` is 0 learns nothing; where a run cannot have produced it at
/// all (its log predictive `−∞`), the second pass's `0·(−∞)` was NaN, and
/// it went into the log joint for good. Its runs now pass on as they are,
/// and the joint stays a number.
#[test]
fn a_row_of_no_share_leaves_the_joint_a_number() {
    for emission in [BocpdEmission::Diag, BocpdEmission::Gaussian] {
        let mut m = Bocpd::new(BocpdCfg {
            emission,
            prior_scale: Some(vec![1e-150]),
            prior_nu: None,
            ..cfg(1)
        })
        .unwrap();
        // Runs that saw a spread of `1e79` read `1e100` at a finite
        // density; the empty run, at the prior's `1e-150`, has `z² = ∞`.
        for x in SPREAD {
            m.step(&[x], &[], 1.0, 1e100);
        }
        let before = m.solve_failures;
        assert_eq!(before, 0);
        m.step(&[1e100], &[], 1.0, 1e-300);
        assert!(
            m.logjoint.iter().all(|l| !l.is_nan()),
            "{emission:?}: {:?}",
            m.logjoint
        );
        assert_eq!(m.solve_failures, before, "{emission:?}");
        let out = m.step(&[0.0], &[], 1.0, 1e100);
        assert!(
            out.pred.iter().all(|v| v.is_finite()),
            "{emission:?}: {:?}",
            out.pred
        );
    }
}

/// BOCPD-1 (2): under `robust` a run that cannot have produced the row has
/// weight `exp(β (−∞ − mode)) = 0`, and its tempered likelihood `π⁰` is 1:
/// the run passes on with its mass, where `0·(−∞)` made the row's joint NaN
/// and the row a failure. The row is learned by the runs that can read it.
#[test]
fn a_robust_run_of_weight_0_passes_on_as_it_is() {
    let mut m = Bocpd::new(BocpdCfg {
        emission: BocpdEmission::Robust,
        robust_beta: 0.5,
        prior_scale: Some(vec![1e-150]),
        ..cfg(1)
    })
    .unwrap();
    // Every row after the first is past the empty run's reach.
    for x in SPREAD.into_iter().chain([1e100]) {
        let out = m.step(&[x], &[], 1.0, 1.0);
        assert!(
            out.pred.iter().all(|v| v.is_finite()),
            "{x}: {:?}",
            out.pred
        );
    }
    assert_eq!(m.solve_failures, 0, "rows were failures");
    assert!(m.logjoint.iter().all(|l| !l.is_nan()));
}

/// BOCPD-2: `Σ ln vᵢ` by blocks of 8 took the log of each block's product
/// where the product was a normal number, but a product can pass through
/// the subnormals on the way there: `1e-160 · 7.4e-164` kept a bit of its
/// 53, and `· 1e160 · 1e164` came back to the range 0.4 off in its log.
/// The sum is the logs' one at a time, to rounding, whatever the order.
#[test]
fn a_sum_of_logs_holds_through_a_product_that_passes_the_subnormals() {
    let v: [f64; 4] = [1e-160, 7.4e-164, 1e160, 1e164];
    let want: f64 = v.iter().map(|x| x.ln()).sum();
    for order in [[0, 1, 2, 3], [2, 0, 3, 1], [3, 2, 1, 0]] {
        let got = ln_sum_prod(&order.map(|i| v[i]));
        assert!(
            (got - want).abs() < 1e-12,
            "{order:?}: {got} against {want}"
        );
    }
    // A block in the range is read as its product, as before, to the bit.
    let plain: [f64; 5] = [2.0, 3.5, 0.25, 7.0, 1.5];
    assert_eq!(
        ln_sum_prod(&plain).to_bits(),
        plain.iter().product::<f64>().ln().to_bits()
    );
}

/// BOCPD-3: on the clock a step `d` under `τ` breaks with chance `1 −
/// e^(−d/τ)`; a step so long that `d/τ` passes the doubles (`1e100` under
/// `1e-250`) is a break for certain, `ln H = 0` and `ln(1 − H) = −∞`, where
/// the row was refused and counted as a failure. It reads as a step whose
/// `d/τ` is merely huge (`1e290`), to the bit.
#[test]
fn a_clock_step_past_the_doubles_is_a_certain_break() {
    let on = |tau: f64| BocpdCfg {
        hazard: tau,
        hazard_on_clock: true,
        ..cfg(1)
    };
    let run = |d: f64| {
        let mut m = Bocpd::new(on(1e-250)).unwrap();
        let mut out = Vec::new();
        for (i, x) in [0.1, -0.3, 0.2, 0.5, -0.1].iter().enumerate() {
            let step = m.step(&[*x], &[], if i == 0 { 0.0 } else { d }, 1.0);
            out.push(bits(&step.pred));
        }
        (out, m.solve_failures)
    };
    let (past, failures) = run(1e100);
    assert_eq!(failures, 0);
    assert_eq!(past, run(1e40).0);
}

/// BOCPD-5: a run's predictive variance `ψₙ(κₙ + 1)/(νₙκₙ)` overflowed in
/// `ψₙ(κₙ + 1)` at `ψ₀ = 1e308`, where the variance itself is half that,
/// and an infinite variance passed the `≤ 0` check: every row read as
/// impossible under the runs. The run posterior is the same at every scale
/// of two of `ψ₀` (the data, all 0 at the prior's mean, move with it), so
/// `ψ₀ = 1e308` reads as `1e308 · 2⁻¹⁰⁰⁰` does in each emission and
/// width, to the rounding of a log.
#[test]
fn a_large_prior_scale_reads_as_the_same_prior_at_a_smaller_scale() {
    for (emission, d) in [
        (BocpdEmission::Diag, 1),
        (BocpdEmission::Diag, 2),
        (BocpdEmission::Gaussian, 2),
        (BocpdEmission::Robust, 1),
    ] {
        let run = |scale: f64| {
            let mut m = Bocpd::new(BocpdCfg {
                emission,
                robust_beta: if emission == BocpdEmission::Robust {
                    0.5
                } else {
                    0.0
                },
                prior_scale: Some(vec![scale]),
                prior_nu: None,
                ..cfg(d)
            })
            .unwrap();
            let mut out = Vec::new();
            for _ in 0..12 {
                let step = m.step(&vec![0.0; d], &[], 1.0, 1.0);
                // `p_change`, `run_mode` and `run_mean`: the log score moves
                // with the scale, by `−d/2 ln s²`.
                out.push(step.pred[..3].to_vec());
            }
            (out, m.solve_failures)
        };
        let (big, failures) = run(1e308);
        assert_eq!(failures, 0, "{emission:?} {d}");
        // The variance's log is taken in two parts at `1e308`, so the
        // two scales agree to the rounding of a log, and the most likely
        // run exactly.
        for (i, (a, b)) in big.iter().zip(run(1e308 * 2f64.powi(-1000)).0).enumerate() {
            assert_eq!(a[1], b[1], "{emission:?} {d}, row {i}: run_mode");
            for (x, y) in a.iter().zip(&b) {
                assert!(
                    (x - y).abs() <= 1e-12 * y.abs().max(1e-300),
                    "{emission:?} {d}, row {i}: {a:?} against {b:?}"
                );
            }
        }
    }
}

/// BOCPD-5's second part: the prior's pull on the mean, `κ₀n/κₙ (x̄ −
/// μ₀)²`, overflowed in `κ₀n` at `κ₀ = 1e307` and made `∞·0 = NaN` on data
/// at the prior's mean; it is `(κ₀/κₙ) n (x̄ − μ₀)²`, 0 there.
#[test]
fn a_heavy_prior_on_data_at_its_mean_reads_a_number() {
    for emission in [BocpdEmission::Diag, BocpdEmission::Gaussian] {
        for d in [1, 2] {
            let mut m = Bocpd::new(BocpdCfg {
                emission,
                prior_kappa: 1e307,
                prior_nu: None,
                ..cfg(d)
            })
            .unwrap();
            for i in 0..40 {
                let out = m.step(&vec![0.0; d], &[], 1.0, 1.0);
                assert!(
                    out.pred.iter().all(|v| v.is_finite()),
                    "{emission:?} {d}, row {i}: {:?}",
                    out.pred
                );
            }
            assert_eq!(m.solve_failures, 0, "{emission:?} {d}");
        }
    }
}

/// BOCPD-4: the compact (positional) msgpack encoding does not round-trip
/// a state whose skipped field is not its last: no failures, and then a
/// warm-up's place. Every file the library writes is named msgpack, which
/// does, so this is the user's call (writing every field always changes the
/// bytes of every saved state).
#[test]
#[should_panic(expected = "compact")]
fn a_compact_state_with_a_skipped_middle_field_round_trips() {
    let mut m = Bocpd::new(BocpdCfg {
        prior_scale: None,
        warm_clock: Some(5.0),
        ..cfg(1)
    })
    .unwrap();
    m.stamp_next(crate::Stamp::from(0.0));
    m.step(&[0.5], &[], 1.0, 1.0);
    assert!(m.solve_failures == 0 && m.place.is_some());
    let bytes = rmp_serde::to_vec_named(&m).unwrap();
    let compact = rmp_serde::from_slice::<Bocpd>(&rmp_serde::to_vec(&m).unwrap());
    let back = compact
        .as_ref()
        .map(|c| rmp_serde::to_vec_named(c).unwrap());
    assert!(
        back.as_ref().is_ok_and(|b| *b == bytes),
        "the compact encoding read back {:?}",
        compact.err()
    );
}
