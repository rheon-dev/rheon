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
        max_state_bytes: None,
        norm: crate::HuberNorm::Huber,
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
    assert!(held <= cap && held + 2 * row > cap, "{held}");
    assert_eq!(m.refit_status()[0].learned, m.refit_status()[1].learned);
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

// ---- mutants-w11: what a span's buffers are charged, and when ----

fn held_of(m: &Robust) -> u64 {
    m.refit.iter().map(RefitBuf::held_bytes).sum()
}

/// A row is charged `8 (n + 4)` bytes per target, and 48 more under the MAD:
/// the row that would pass the cap is refused atomically across both targets
/// and recorded as `held + 2 * row`, exactly, under each scale.
#[test]
fn a_row_is_charged_its_width_and_the_mads_extra_to_the_byte() {
    for (scale, extra) in [(HuberScale::Mad, 48u64), (HuberScale::Rms, 0)] {
        let c = RobustCfg {
            refit_clock: Some(1e12),
            ..base(HUBER, scale)
        };
        let mut m = Robust::new(c).unwrap();
        let row = 8 * (2 + 4) + extra;
        let cap = 100 * row;
        m.set_cap_for_test(cap);
        for (i, (x, y)) in rows(150, 5).iter().enumerate() {
            m.stamp_next(Stamp::Ns(i as i128 * S));
            m.step(x, y, 1.0, 1.0);
        }
        assert_eq!(
            m.state_over_cap(),
            Some((held_of(&m) + 2 * row, cap)),
            "{scale:?}"
        );
    }
}

/// A buffer that has run asks the cap for nothing: its rows went to the fit.
#[test]
fn a_buffer_that_has_run_is_charged_nothing() {
    let data = rows(12, 8);
    let mut m = Robust::new(RobustCfg {
        refit_clock: Some(3.0),
        ..base(HUBER, HuberScale::Mad)
    })
    .unwrap();
    for (i, (x, y)) in data[..8].iter().enumerate() {
        m.stamp_next(Stamp::Ns(i as i128 * S));
        m.step(x, y, 1.0, 1.0);
    }
    assert!(m.refit_status().iter().all(|s| s.done));
    m.set_cap_for_test(1);
    m.stamp_next(Stamp::Ns(9 * S));
    m.step(&data[8].0, &data[8].1, 1.0, 1.0);
    assert_eq!(m.state_over_cap(), None);
}

/// The quantile loss's refit waits for three rows a coefficient, the rows
/// its warm-up takes as least squares, however long its span has run.
#[test]
fn a_quantile_refit_waits_for_three_rows_a_coefficient() {
    let c = RobustCfg {
        refit_clock: Some(10.0),
        ..base(QUANTILE, HuberScale::Mad)
    };
    let data = rows(12, 4);
    let mut m = Robust::new(c).unwrap();
    for (i, (x, y)) in data[..10].iter().enumerate() {
        let t = if i == 0 { 0 } else { 20 * S + i as i128 * S };
        m.stamp_next(Stamp::Ns(t));
        m.step(x, y, 1.0, 1.0);
        let done = m.refit_status().iter().all(|s| s.done);
        assert_eq!(done, i >= 8, "row {i}");
    }
}

/// The warning's flag is state: saved after the warning, it restores raised.
#[test]
fn a_warned_flag_survives_a_save() {
    let mut m = Robust::new(RobustCfg {
        refit_clock: Some(1e12),
        ..base(HUBER, HuberScale::Mad)
    })
    .unwrap();
    m.set_cap_for_test(100 * (8 * 6 + 48));
    for (i, (x, y)) in rows(30, 5).iter().enumerate() {
        m.stamp_next(Stamp::Ns(i as i128 * S));
        m.step(x, y, 1.0, 1.0);
    }
    assert!(m.warned && m.over.is_none());
    let bytes = rmp_serde::to_vec_named(&m.state()).unwrap();
    let key = b"\xa6warned";
    assert!(bytes.windows(key.len()).any(|w| w == key));
    let back = Robust::restore(&rmp_serde::from_slice(&bytes).unwrap()).unwrap();
    assert!(back.warned);
}

/// Review 2026-11-23, REG-2: a row whose target is null ages the open
/// buffer, a run of 32 bytes wherever its decay is not the last run's, and
/// that was checked against no cap: on an irregular clock a target that
/// went quiet grew its buffer by a run a row, without a warning or a
/// refusal, up to a bound of 512 GiB. An ageing row is held to the cap as
/// a learned row is.
#[test]
fn an_ageing_row_is_held_to_the_spans_cap() {
    for loss in [HUBER, QUANTILE] {
        let c = RobustCfg {
            refit_clock: Some(1e12),
            ..base(loss, HuberScale::Mad)
        };
        let mut m = Robust::new(c).unwrap();
        let cap = 4096;
        m.set_cap_for_test(cap);
        let data = rows(2000, 9);
        for (i, (x, y)) in data.iter().enumerate() {
            m.stamp_next(Stamp::Ns(i as i128 * S));
            // Both targets quiet after the first rows, on a clock whose every
            // step differs.
            let y = if i < 4 { *y } else { [None, None] };
            m.step(x, &y, 1.0 + i as f64 * 1e-6, 1.0);
            assert!(held_of(&m) <= cap, "{loss:?} row {i}: {}", held_of(&m));
        }
        assert!(m.state_over_cap().is_some(), "{loss:?}");
    }
}

/// A state whose place on the clock does not match its refit's span is
/// refused (review 2026-11-23, REG-16), as `sgd`'s is: a span's state
/// without its place never closed its span, and held rows until the byte
/// cap stopped them; a place without a span is a state no configuration
/// writes.
#[test]
fn a_state_whose_place_does_not_match_its_span_is_refused() {
    let span = Robust::new(RobustCfg {
        refit_clock: Some(40.0),
        ..base(HUBER, HuberScale::Mad)
    })
    .unwrap();
    let rows_ = Robust::new(RobustCfg {
        refit_rows: 40,
        ..base(HUBER, HuberScale::Mad)
    })
    .unwrap();
    for (m, place) in [
        (span, None),
        (rows_, Some(crate::clock_span::Place::default())),
    ] {
        let mut s = m.state();
        assert!(Robust::restore(&s).is_ok());
        let crate::ModelState::Robust(inner) = &mut s.model else {
            unreachable!()
        };
        inner.place = place;
        match Robust::restore(&s) {
            Err(crate::StateError::Invalid(e)) => assert!(e.contains("refit"), "{e}"),
            other => panic!("{other:?}"),
        }
    }
}

/// STATE-3: a `huber` or `quantile` span refit whose buffered rows exceed
/// `max_state_bytes` (or a non-span state recording `over`) is refused on
/// load, while an empty span refit with a 1-byte cap still round-trips.
#[test]
fn state_3_a_span_past_its_cap_is_refused_on_load() {
    for loss in [HUBER, QUANTILE] {
        let fresh = Robust::new(RobustCfg {
            refit_clock: Some(10.0),
            max_state_bytes: Some(1),
            ..base(loss, HuberScale::Mad)
        })
        .unwrap();
        assert!(Robust::restore(&fresh.state()).is_ok(), "{loss:?}");

        let mut m = Robust::new(RobustCfg {
            refit_clock: Some(1e12),
            ..base(loss, HuberScale::Mad)
        })
        .unwrap();
        for (i, (x, y)) in rows(6, 12).iter().enumerate() {
            m.stamp_next(Stamp::Ns(i as i128 * S));
            m.step(x, y, 1.0, 1.0);
        }
        let held = held_of(&m);
        m.cfg.max_state_bytes = Some(held);
        assert!(Robust::restore(&m.state()).is_ok(), "{loss:?}");

        let mut over = m.clone();
        over.cfg.max_state_bytes = Some(held - 1);
        match Robust::restore(&over.state()) {
            Err(crate::StateError::Invalid(e)) => {
                assert!(e.contains("max_state_bytes"), "{loss:?}: {e}");
            }
            other => panic!("{loss:?} over cap: {other:?}"),
        }

        let mut non_span = Robust::new(RobustCfg {
            refit_rows: 20,
            ..base(loss, HuberScale::Mad)
        })
        .unwrap();
        non_span.over = Some(1);
        assert!(Robust::restore(&non_span.state()).is_err(), "{loss:?}");
    }
}

/// Direct tests for `Robust::scale_after_age`, `Robust::span_row_delta`, and
/// `step_into`'s span-cap guards (`refit_clock.is_some() && open`, `delta > 0`).
#[test]
fn scale_after_age_span_row_delta_and_step_into_guards() {
    let data = rows(60, 77);

    // 1. `scale_after_age`:
    // (a) `Mad`, refit done (`!reads_resid`): returns `Some(s)` (`s != 1.0`) when `lam_scale != 0.0`,
    //     and `None` when `lam_scale == 0.0`.
    let mut m_mad = Robust::new(RobustCfg {
        refit_rows: 10,
        ..base(HUBER, HuberScale::Mad)
    })
    .unwrap();
    for (x, y) in &data[..30] {
        m_mad.step(x, y, 1.0, 1.0);
    }
    assert!(m_mad.refit_status()[0].done);
    let s_mad = m_mad.residual_scale(0).expect("settled Mad scale");
    assert!((s_mad - 1.0).abs() > 1e-6, "need scale != 1.0, got {s_mad}");
    assert_eq!(m_mad.scale_after_age(0, 1.0), Some(s_mad));
    assert_eq!(m_mad.scale_after_age(0, 0.0), None);

    // (b) `Rms` (`mad` empty): returns `Some(s)` even when `lam_scale == 0.0`.
    let mut m_rms = Robust::new(RobustCfg {
        refit_rows: 10,
        ..base(HUBER, HuberScale::Rms)
    })
    .unwrap();
    for (x, y) in &data[..30] {
        m_rms.step(x, y, 1.0, 1.0);
    }
    let s_rms = m_rms.residual_scale(0).expect("settled Rms scale");
    assert_eq!(m_rms.scale_after_age(0, 0.0), Some(s_rms));

    // (c) `Mad` with open refit (`reads_resid == true`): returns `Some(s)` even at `lam_scale == 0.0`.
    let mut m_open = Robust::new(RobustCfg {
        refit_clock: Some(1e9),
        ..base(HUBER, HuberScale::Mad)
    })
    .unwrap();
    for (i, (x, y)) in data[..6].iter().enumerate() {
        m_open.stamp_next(Stamp::Ns(i as i128 * S));
        m_open.step(x, y, 1.0, 1.0);
    }
    assert!(m_open.refit[0].reads_resid());
    let s_open = m_open.residual_scale(0).expect("warm refit Mad scale");
    assert_eq!(m_open.scale_after_age(0, 0.0), Some(s_open));

    // 2. `span_row_delta`: `counts = weight > 0.0 && weight.is_finite()`,
    //    `delta += b.age_bytes(...)` across 2 targets, and `pred[j * nc + weighing]`.
    let per = 8 * (2 + 4) as u64 + 48; // 96 bytes per target
    let pred_ok = [0.0; 4];
    assert_eq!(
        m_open.span_row_delta(
            &[1.0, 1.0],
            &[Some(1.0), Some(1.0)],
            &pred_ok,
            1,
            0,
            1.0,
            0.9,
            0.9,
        ),
        2 * per
    );
    // Zero, negative, and infinite weight each use `age_bytes` (32 per open target => 64 total), not `per` (96).
    for bad_w in [0.0, -1.0, f64::INFINITY] {
        assert_eq!(
            m_open.span_row_delta(
                &[1.0, 1.0],
                &[Some(1.0), Some(1.0)],
                &pred_ok,
                1,
                0,
                bad_w,
                0.9,
                0.9,
            ),
            64,
            "bad_w={bad_w}"
        );
    }
    // Missing targets (`None`) also sum `age_bytes` across both open targets.
    assert_eq!(
        m_open.span_row_delta(&[1.0, 1.0], &[None, None], &pred_ok, 1, 0, 1.0, 0.9, 0.9),
        64
    );
    // `pred[j * nc + weighing]` with `j = 1, nc = 2, weighing = 1` (`1 * 2 + 1 = 3`):
    // Under Hampel with `delta = 0.6e308`, `c = 4 * delta = +inf` while `b = 1.2e308` is finite,
    // so `u = 1.5e308` yields `inf / inf = NaN` in `row_update` (`buffers = false` => `age_bytes = 32`).
    // Put `1.5e308` ONLY at index 3 (`pred[0..3] = 0.0`): `* -> /`, `+ -> *`, and `+ -> -` read
    // index 1 or 2 (`0.0` => `w = 1.0` => `per_rms = 48`), failing the exact `32` check.
    let mut m_idx = Robust::new(RobustCfg {
        n_targets: 2,
        loss: RobustLoss::Huber { delta: 0.6e308 },
        norm: crate::robust::HuberNorm::Hampel,
        refit_clock: Some(100.0),
        ..base(HUBER, HuberScale::Rms)
    })
    .unwrap();
    for (i, (x, y)) in data[..2].iter().enumerate() {
        m_idx.stamp_next(Stamp::Raw(i as f64, 0.0));
        m_idx.step(x, &[y[0], y[0]], 1.0, 1.0);
    }
    let _ = m_idx.refit[0].take();
    m_idx.sig2[1] = 1.0;
    m_idx.wsig[1] = 1.0;
    let pred_hampel3 = [0.0, 0.0, 0.0, 1.5e308];
    assert_eq!(
        m_idx.span_row_delta(
            &[1.0, 1.0],
            &[Some(0.0), Some(0.0)],
            &pred_hampel3,
            2,
            1,
            1.0,
            0.9,
            0.9,
        ),
        32
    );

    // 3. `step_into` guards:
    // (a) `refit_clock.is_some() && open` (`&&` not `||`): a `refit_rows` model (no `refit_clock`)
    //     with an open refit buffer and tiny `max_state_bytes` must NOT enforce span cap!
    let mut m_rows = Robust::new(RobustCfg {
        refit_rows: 10,
        ..base(HUBER, HuberScale::Mad)
    })
    .unwrap();
    m_rows.set_cap_for_test(1);
    m_rows.step(&data[0].0, &data[0].1, 1.0, 1.0);
    assert_eq!(m_rows.state_over_cap(), None);

    // (b) `if delta > 0` (`>` not `>=`): when `delta == 0` (joining an existing age run),
    //     `guard.note` must NOT fire even if `warned` was cleared while `2 * held > cap`.
    let mut m_zero_delta = m_open.clone();
    // First zero-weight row adds an age run at `(lam, lam_scale)` for `d_clock = 1.0`.
    m_zero_delta.step(&[1.0, 1.0], &[None, None], 1.0, 0.0);
    let held = held_of(&m_zero_delta);
    m_zero_delta.set_cap_for_test(held); // `2 * held > held`
    m_zero_delta.warned = false;
    m_zero_delta.warning = None;
    // Second identical `None` row at `d_clock = 1.0` joins the last age run => `delta == 0`!
    m_zero_delta.step(&[1.0, 1.0], &[None, None], 1.0, 0.0);
    assert_eq!(
        m_zero_delta.take_state_warning(),
        None,
        "delta == 0 row must not enter the cap/note block"
    );
    assert_eq!(m_zero_delta.state_over_cap(), None);
}
