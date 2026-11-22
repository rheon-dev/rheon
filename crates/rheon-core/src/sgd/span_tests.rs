//! `sgd(loss = "huber")`'s refit of the first rows over a span of the clock
//! (`refit_clock`, docs/PLAN.md task 538): `refit_rows = D + 1` on a clock
//! of one unit a row, both sides of the span's end, the buffers' cap as
//! they grow, a state saved inside the span, the refusals.

use super::*;
use crate::{OnlineModel, Stamp};

const S: i128 = 1_000_000_000;

fn cfg(standardize: bool) -> SgdCfg {
    SgdCfg {
        n_features: 2,
        n_targets: 2,
        fit_intercept: true,
        decay: Decay::Halflife(400.0),
        loss: SgdLoss::Huber { delta: 1.345 },
        learning_rate: 0.05,
        schedule: LearningRate::AdaGrad,
        l2: 0.0,
        min_weight: 3.0,
        clip_gradient: 1e3,
        constraint: None,
        standardize,
        strict_binary: false,
        huber_scale: crate::HuberScale::Mad,
        huber_scale_decay: None,
        refit_rows: 0,
        min_weight_holds_learning: false,
        target_spread: false,
        mad_anchor: false,
        refit_clock: None,
    }
}

/// Both targets on every row, a share of the first wild.
fn rows(n: usize, seed: u64) -> Vec<([f64; 2], [Option<f64>; 2])> {
    let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    let mut u = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    };
    (0..n)
        .map(|_| {
            let x = [u(), 3.0 + u()];
            let wild = if x[0] > 0.0 && u() > 0.6 { 50.0 } else { 0.0 };
            (
                x,
                [
                    Some(100.0 + 2.0 * x[0] - x[1] + 0.5 * u() + wild),
                    Some(-3.0 + x[1] + 0.5 * u()),
                ],
            )
        })
        .collect()
}

fn bits(m: &mut Sgd, x: &[f64], y: &[Option<f64>]) -> Vec<u64> {
    let out = m.step(x, y, 1.0, 1.0);
    let mut b: Vec<u64> = out.pred.iter().map(|v| v.to_bits()).collect();
    b.push(out.n_eff.to_bits());
    b
}

#[test]
fn a_span_of_d_is_d_plus_one_rows_on_a_regular_clock() {
    for standardize in [false, true] {
        for d in [6.0, 40.0] {
            let count = SgdCfg {
                refit_rows: d as u32 + 1,
                ..cfg(standardize)
            };
            let span = SgdCfg {
                refit_clock: Some(d),
                ..cfg(standardize)
            };
            for temporal in [false, true] {
                let (mut a, mut b) = (
                    Sgd::new(count.clone()).unwrap(),
                    Sgd::new(span.clone()).unwrap(),
                );
                for (i, (x, y)) in rows(150, 3).iter().enumerate() {
                    let t = i as i128 + 1;
                    if temporal {
                        b.stamp_next(Stamp::Ns(t * S));
                    } else {
                        b.stamp_next_on_rows(Stamp::from(t as f64));
                    }
                    assert_eq!(
                        bits(&mut a, x, y),
                        bits(&mut b, x, y),
                        "{standardize} {d} {temporal} row {i}"
                    );
                }
                assert!(b.refit_status().iter().all(|s| s.done));
            }
        }
    }
}

#[test]
fn the_span_closes_at_its_length_and_not_a_nanosecond_before() {
    let c = SgdCfg {
        refit_clock: Some(10.0),
        ..cfg(false)
    };
    let data = rows(8, 4);
    for (last, done) in [(10 * S - 1, false), (10 * S, true)] {
        let mut m = Sgd::new(c.clone()).unwrap();
        for (i, (x, y)) in data[..5].iter().enumerate() {
            m.stamp_next(Stamp::Ns(if i == 4 { last } else { i as i128 * S }));
            m.step(x, y, 1.0, 1.0);
        }
        assert!(
            m.refit_status().iter().all(|s| s.done == done),
            "last at {last}"
        );
    }
    let mut few = Sgd::new(c).unwrap();
    for (i, (x, y)) in data[..3].iter().enumerate() {
        few.stamp_next(Stamp::Ns([0, 20, 21][i] * S));
        few.step(x, y, 1.0, 1.0);
        assert!(
            few.refit_status().iter().all(|s| s.done == (i == 2)),
            "row {i}"
        );
    }
}

#[test]
fn the_spans_buffers_are_warned_of_at_half_their_cap_and_stopped_at_it() {
    let c = SgdCfg {
        refit_clock: Some(1e12),
        ..cfg(false)
    };
    let mut m = Sgd::new(c).unwrap();
    let row = 8 * (2 + 4) + 48;
    let cap = 100 * row;
    m.set_cap_for_test(cap);
    let mut warned = 0;
    for (i, (x, y)) in rows(150, 5).iter().enumerate() {
        m.stamp_next(Stamp::Ns(i as i128 * S));
        m.step(x, y, 1.0, 1.0);
        if let Some((_, got)) = m.take_state_warning() {
            warned += 1;
            assert_eq!(got, cap);
        }
    }
    assert_eq!(warned, 1);
    let (bytes, got) = m.state_over_cap().expect("stopped at the cap");
    assert!(bytes > cap && got == cap);
    let held: u64 = m.refit.iter().map(RefitBuf::held_bytes).sum();
    assert!(held <= cap && held + row > cap, "{held}");
}

#[test]
fn a_state_saved_inside_the_span_goes_on_to_the_bit() {
    let c = SgdCfg {
        refit_clock: Some(40.0),
        ..cfg(true)
    };
    let data = rows(160, 6);
    let mut a = Sgd::new(c).unwrap();
    for (x, y) in &data[..25] {
        a.step(x, y, 0.9, 1.0);
    }
    let bytes = rmp_serde::to_vec_named(&a.state()).unwrap();
    let mut b = Sgd::restore(&rmp_serde::from_slice(&bytes).unwrap()).unwrap();
    for (i, (x, y)) in data[25..].iter().enumerate() {
        let (p, q) = (a.step(x, y, 0.9, 1.0), b.step(x, y, 0.9, 1.0));
        let bits = |s: &Step| s.pred.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&p), bits(&q), "row {i}");
    }
    assert!(b.refit_status().iter().all(|s| s.done));
    let plain = rmp_serde::to_vec_named(
        &Sgd::new(SgdCfg {
            refit_rows: 40,
            ..cfg(true)
        })
        .unwrap()
        .state(),
    )
    .unwrap();
    for f in ["refit_clock", "place", "first", "over", "warned"] {
        let mut key = vec![0xa0 | f.len() as u8];
        key.extend_from_slice(f.as_bytes());
        assert!(!plain.windows(key.len()).any(|w| w == key), "{f}");
    }
}

#[test]
fn the_span_is_checked() {
    let refuse = |c: SgdCfg, words: &str| {
        let e = Sgd::new(c).unwrap_err();
        assert!(e.contains(words), "{e}");
    };
    for d in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        refuse(
            SgdCfg {
                refit_clock: Some(d),
                ..cfg(false)
            },
            "huber_refit_span must be a finite number",
        );
    }
    refuse(
        SgdCfg {
            refit_clock: Some(5.0),
            refit_rows: 20,
            ..cfg(false)
        },
        "two refits of the first rows",
    );
    refuse(
        SgdCfg {
            refit_clock: Some(5.0),
            loss: SgdLoss::Squared,
            ..cfg(false)
        },
        "huber_refit_span",
    );
}
