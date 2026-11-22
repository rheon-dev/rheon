//! `huber`'s and `quantile`'s refit of the first rows over a span of the
//! clock (`refit_clock`, docs/PLAN.md task 538): `refit_rows = D + 1` on a
//! clock of one unit a row under each scale, both sides of the span's end,
//! the wait for its least rows, the buffers' cap as they grow, a state
//! saved inside the span, the refusals.

use super::*;
use crate::{OnlineModel, Stamp};

const S: i128 = 1_000_000_000;

fn base(loss: RobustLoss, scale: HuberScale) -> RobustCfg {
    RobustCfg {
        n_features: 2,
        n_targets: 2,
        fit_intercept: true,
        decay: Decay::Halflife(500.0),
        loss,
        ridge: 1e-6,
        standardize: false,
        min_weight: 3.0,
        solve_every: 0.0,
        max_rows_between_solves: u32::MAX,
        solve_share: None,
        quantile_eps: 1e-3,
        huber_scale: scale,
        huber_scale_decay: None,
        feature_sets: Vec::new(),
        ridge_scale: false,
        emit_scale: false,
        refit_rows: 0,
        mad_anchor: false,
        min_weight_holds_learning: false,
        refit_clock: None,
    }
}

const HUBER: RobustLoss = RobustLoss::Huber { delta: 1.345 };
const QUANTILE: RobustLoss = RobustLoss::Quantile { tau: 0.7 };

/// Both targets on every row, a share of them wild.
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
            let x = [u(), u()];
            let wild = if x[0] > 0.0 && u() > 0.5 { 40.0 } else { 0.0 };
            let y0 = 10.0 + 2.0 * x[0] - x[1] + 0.5 * u() + wild;
            let y1 = -3.0 + x[1] + 0.5 * u();
            (x, [Some(y0), Some(y1)])
        })
        .collect()
}

fn bits(m: &mut Robust, x: &[f64], y: &[Option<f64>]) -> Vec<u64> {
    let out = m.step(x, y, 1.0, 1.0);
    let mut b: Vec<u64> = out.pred.iter().map(|v| v.to_bits()).collect();
    b.push(out.n_eff.to_bits());
    b
}

#[test]
fn a_span_of_d_is_d_plus_one_rows_on_a_regular_clock() {
    for (loss, d) in [(HUBER, 30.0), (QUANTILE, 30.0), (HUBER, 6.0)] {
        for scale in [HuberScale::Mad, HuberScale::Rms] {
            let count = RobustCfg {
                refit_rows: d as u32 + 1,
                ..base(loss, scale)
            };
            let span = RobustCfg {
                refit_clock: Some(d),
                ..base(loss, scale)
            };
            for temporal in [false, true] {
                let (mut a, mut b) = (
                    Robust::new(count.clone()).unwrap(),
                    Robust::new(span.clone()).unwrap(),
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
                        "{loss:?} {scale:?} {temporal} row {i}"
                    );
                    let (sa, sb) = (a.refit_status(), b.refit_status());
                    for (p, q) in sa.iter().zip(&sb) {
                        assert_eq!((p.done, p.learned), (q.done, q.learned), "row {i}");
                    }
                }
                assert!(b.refit_status().iter().all(|s| s.done));
            }
        }
    }
}

#[test]
fn the_span_closes_at_its_length_and_not_a_nanosecond_before() {
    let c = RobustCfg {
        refit_clock: Some(10.0),
        ..base(HUBER, HuberScale::Mad)
    };
    let data = rows(8, 4);
    for (last, done) in [(10 * S - 1, false), (10 * S, true)] {
        let mut m = Robust::new(c.clone()).unwrap();
        for (i, (x, y)) in data[..5].iter().enumerate() {
            m.stamp_next(Stamp::Ns(if i == 4 { last } else { i as i128 * S }));
            m.step(x, y, 1.0, 1.0);
        }
        assert!(
            m.refit_status().iter().all(|s| s.done == done),
            "last at {last}"
        );
    }
    // Past its span with two rows, three coefficients wait for a third.
    let mut few = Robust::new(c).unwrap();
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
    let c = RobustCfg {
        refit_clock: Some(1e12),
        ..base(HUBER, HuberScale::Mad)
    };
    let mut m = Robust::new(c).unwrap();
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
    for loss in [HUBER, QUANTILE] {
        let c = RobustCfg {
            refit_clock: Some(40.0),
            ..base(loss, HuberScale::Mad)
        };
        let data = rows(160, 6);
        let mut a = Robust::new(c).unwrap();
        for (x, y) in &data[..25] {
            a.step(x, y, 0.9, 1.0);
        }
        let bytes = rmp_serde::to_vec_named(&a.state()).unwrap();
        let mut b = Robust::restore(&rmp_serde::from_slice(&bytes).unwrap()).unwrap();
        for (i, (x, y)) in data[25..].iter().enumerate() {
            let (p, q) = (a.step(x, y, 0.9, 1.0), b.step(x, y, 0.9, 1.0));
            let bits = |s: &Step| s.pred.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&p), bits(&q), "{loss:?} row {i}");
        }
        assert!(b.refit_status().iter().all(|s| s.done));
    }
    let plain = rmp_serde::to_vec_named(
        &Robust::new(RobustCfg {
            refit_rows: 40,
            ..base(HUBER, HuberScale::Mad)
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
    let refuse = |c: RobustCfg, words: &str| {
        let e = Robust::new(c).unwrap_err();
        assert!(e.contains(words), "{e}");
    };
    for d in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        refuse(
            RobustCfg {
                refit_clock: Some(d),
                ..base(HUBER, HuberScale::Mad)
            },
            "huber_refit_span must be a finite number",
        );
    }
    refuse(
        RobustCfg {
            refit_clock: Some(5.0),
            refit_rows: 20,
            ..base(HUBER, HuberScale::Mad)
        },
        "two refits of the first rows",
    );
    refuse(
        RobustCfg {
            refit_clock: Some(5.0),
            ridge_scale: true,
            ..base(HUBER, HuberScale::Mad)
        },
        "ridge_scale",
    );
}
