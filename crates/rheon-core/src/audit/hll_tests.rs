//! Unit tests for `audit`'s HyperLogLog++ and sliding-window HyperLogLog++
//! distinct-count sketches.

use super::hll::{Hll, MAX_DISTINCT_PRECISION, MIN_DISTINCT_PRECISION, SlidingHll, splitmix64};
use super::*;
use crate::clock_span::{DEFAULT_STATE_BYTES, STATE_BYTES_CEILING};

fn hll_cfg(
    k: usize,
    precision: Option<u8>,
    window_rows: Option<usize>,
    window_clock: Option<f64>,
    closed: WindowClosed,
    max_state_bytes: Option<u64>,
) -> AuditCfg {
    AuditCfg {
        n_columns: k,
        pairs: false,
        distinct_cap: DISTINCT_CAP,
        gap_cap: None,
        has_clock: window_clock.is_some(),
        distinct: Some(Box::new(AuditDistinctCfg {
            precision,
            window_rows,
            window_clock,
            closed,
            max_state_bytes,
        })),
    }
}

/// Deterministic pseudo-random sequence of usable f64 values.
fn values(seed: u64, n: usize) -> Vec<f64> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = splitmix64(s);
            (s as f64) * 1e-12
        })
        .collect()
}

#[test]
fn splitmix64_matches_pinned_vectors_and_key_hasher() {
    // SplitMix64's first output from state 0.
    assert_eq!(splitmix64(0), 0xe220_a839_7b1d_cdaf);
    // Matches `audit::KeyHasher::write_u64` for every tested key.
    for k in [0u64, 1, 7, 42, u64::MAX, 0x3ff0_0000_0000_0000] {
        let mut h = KeyHasher::default();
        h.write_u64(k);
        assert_eq!(h.finish(), splitmix64(k));
    }
}

#[test]
fn plain_hll_counts_usable_values_out_of_sample_and_merges_exactly() {
    let mut m = Audit::new(hll_cfg(2, Some(12), None, None, WindowClosed::Right, None)).unwrap();
    assert_eq!(m.n_outputs(), 2);
    // Empty sketch reads 0.0.
    assert_eq!(m.column(0).approx_distinct, Some(0.0));
    assert_eq!(m.predict(&[0.0, 0.0], 1.0).pred, vec![0.0, 0.0]);

    // -0.0 and +0.0 map to the same key; unusable values reach no register.
    let s0 = m.step(&[0.0, NULL], &[], 1.0, 1.0);
    assert_eq!(s0.pred, vec![0.0, 0.0], "out-of-sample before row 0");
    let s1 = m.step(&[-0.0, f64::NAN], &[], 1.0, 1.0);
    assert!(s1.pred[0] > 0.5 && s1.pred[0] < 1.5);
    assert_eq!(s1.pred[1], 0.0, "null did not enter HLL");
    let s2 = m.step(&[f64::INFINITY, -f64::INFINITY], &[], 1.0, 1.0);
    assert_eq!(s2.pred[0], s1.pred[0], "-0.0 matched +0.0");
    assert_eq!(s2.pred[1], 0.0, "NaN did not enter HLL");
    let s3 = m.step(&[1e200, 1.0], &[], 1.0, 1.0);
    assert_eq!(s3.pred[0], s1.pred[0], "+inf did not enter HLL");
    assert_eq!(s3.pred[1], 0.0, "-inf did not enter HLL");

    // Large distinct count past `distinct_cap` (256): exact count is None, HLL++
    // is within 3 SE (`3 * 1.04 / sqrt(4096) ≈ 4.9%`).
    let xs = values(12345, 8_000);
    let mut a = Audit::new(hll_cfg(1, Some(12), None, None, WindowClosed::Right, None)).unwrap();
    let mut left = Audit::new(hll_cfg(1, Some(12), None, None, WindowClosed::Right, None)).unwrap();
    let mut right =
        Audit::new(hll_cfg(1, Some(12), None, None, WindowClosed::Right, None)).unwrap();
    for (i, &v) in xs.iter().enumerate() {
        a.step(&[v], &[], 1.0, 1.0);
        if i < 3_500 {
            left.step(&[v], &[], 1.0, 1.0);
        } else {
            right.step(&[v], &[], 1.0, 1.0);
        }
    }
    let rep = a.column(0);
    assert_eq!(rep.distinct, None, "past distinct_cap");
    let est = rep.approx_distinct.unwrap();
    let rel_err = (est - 8_000.0).abs() / 8_000.0;
    assert!(
        rel_err < 3.0 * 1.04 / (4096.0f64).sqrt(),
        "rel_err={rel_err}"
    );

    // Merge equals single-stream sketch register-for-register and count-for-count.
    let mut merged_lr = left.clone();
    merged_lr.merge(&right).unwrap();
    let mut merged_rl = right.clone();
    merged_rl.merge(&left).unwrap();
    assert_eq!(merged_lr.hll_registers(0), a.hll_registers(0));
    assert_eq!(merged_rl.hll_registers(0), a.hll_registers(0));
    assert_eq!(
        merged_lr.column(0).approx_distinct.unwrap().to_bits(),
        est.to_bits()
    );
    // Idempotent merge.
    let before = merged_lr.hll_registers(0).unwrap().to_vec();
    merged_lr.merge(&left).unwrap();
    assert_eq!(merged_lr.hll_registers(0).unwrap(), before.as_slice());
}

#[test]
fn plain_hll_hits_linear_bias_and_raw_regimes_across_all_precisions() {
    for p in MIN_DISTINCT_PRECISION..=MAX_DISTINCT_PRECISION {
        let m = 1usize << p;
        let mut h = Hll::new(p);
        assert_eq!(h.count(p), 0.0);
        // Linear counting regime (`n = 5`).
        for i in 1..=5u64 {
            h.add_key(key(i as f64), p);
        }
        let est_small = h.count(p);
        assert!(
            (est_small - 5.0).abs() < 1.5,
            "p={p}, est_small={est_small}"
        );
        // Bias-corrected regime (`n ≈ 2 * m` for `p <= 12`).
        if p <= 12 {
            for i in 6..=(2 * m as u64) {
                h.add_key(key(i as f64), p);
            }
            let est_mid = h.count(p);
            let target = (2 * m) as f64;
            let se = 1.04 / (m as f64).sqrt();
            assert!(
                (est_mid - target).abs() / target < 4.0 * se,
                "p={p}, est_mid={est_mid}, target={target}"
            );
            // Raw regime (`E > 5 * m`, `n = 8 * m`).
            for i in (2 * m as u64 + 1)..=(8 * m as u64) {
                h.add_key(key(i as f64), p);
            }
            let est_hi = h.count(p);
            let target_hi = (8 * m) as f64;
            assert!(
                (est_hi - target_hi).abs() / target_hi < 4.0 * se,
                "p={p}, est_hi={est_hi}, target_hi={target_hi}"
            );
        }
    }
}

#[test]
fn sliding_row_window_matches_fresh_hll_register_for_register() {
    let p = 8u8;
    let w = 25usize;
    let mut xs = values(99, 120);
    // Sprinkle repeats, unusable values, and a constant run.
    for i in (10..40).step_by(3) {
        xs[i] = 42.0;
    }
    xs[15] = NULL;
    xs[30] = f64::NAN;
    xs[60] = f64::INFINITY;
    xs[85] = 1e200;

    let mut audit = Audit::new(hll_cfg(
        1,
        Some(p),
        Some(w),
        None,
        WindowClosed::Right,
        None,
    ))
    .unwrap();
    let mut direct: SlidingHll<u64> = SlidingHll::new(p);

    for (i, &v) in xs.iter().enumerate() {
        let step = audit.step(&[v], &[], 1.0, 1.0);
        // Fresh HLL over rows `[i.saturating_sub(w) .. i]` (out-of-sample).
        let start_before = i.saturating_sub(w);
        let mut ref_before = Hll::new(p);
        for &u in &xs[start_before..i] {
            if crate::usable(u) {
                ref_before.add_key(key(u), p);
            }
        }
        assert_eq!(
            step.pred[0].to_bits(),
            ref_before.count(p).to_bits(),
            "out-of-sample mismatch at row {i}"
        );

        // Also check direct SlidingHll register-for-register after row i.
        let seq = (i + 1) as u64;
        direct.trim(|s| s <= seq.saturating_sub(w as u64));
        if crate::usable(v) {
            direct.add_key(key(v), seq);
        }
        let start_after = (i + 1).saturating_sub(w);
        let mut ref_after = Hll::new(p);
        for &u in &xs[start_after..=i] {
            if crate::usable(u) {
                ref_after.add_key(key(u), p);
            }
        }
        assert_eq!(
            direct.active_registers(),
            ref_after.registers(),
            "register mismatch after row {i}"
        );
        assert_eq!(
            audit.column(0).approx_distinct.unwrap().to_bits(),
            ref_after.count(p).to_bits(),
            "column report mismatch after row {i}"
        );
    }
}

#[test]
fn sliding_hll_heap_stays_bounded_on_constant_stream() {
    // Regression check: a constant stream replaces the same register's head on
    // every row; heap compaction keeps internal heap bounded.
    let mut audit = Audit::new(hll_cfg(
        1,
        Some(8),
        Some(5_000),
        None,
        WindowClosed::Right,
        None,
    ))
    .unwrap();
    for _ in 0..3_000 {
        audit.step(&[7.0], &[], 1.0, 1.0);
    }
    let est = audit.column(0).approx_distinct.unwrap();
    let mut ref_hll = Hll::new(8);
    ref_hll.add_key(key(7.0), 8);
    assert_eq!(est.to_bits(), ref_hll.count(8).to_bits());
}

#[test]
fn sliding_clock_window_matches_fresh_hll_across_stamps_and_closed_modes() {
    let p = 8u8;
    let span = 10.0;
    let xs = values(2026, 90);
    // Stamps with exact boundary hits (deltas summing to 10.0), duplicates, and jumps.
    let raw_ts: Vec<f64> = {
        let mut out = Vec::with_capacity(xs.len());
        let mut t = 0.0;
        for i in 0..xs.len() {
            if i == 0 {
                t = 0.0;
            } else if i % 7 == 0 {
                // duplicate timestamp!
            } else if i % 13 == 0 {
                t += 10.0; // exact window boundary jump
            } else if i % 19 == 0 {
                t += 15.0; // clears window
            } else {
                t += 2.0; // hits exact multiples of 10.0
            }
            out.push(t);
        }
        out
    };

    for closed in [WindowClosed::Right, WindowClosed::Both] {
        // Test three stamp representations: Raw, Ns (nanoseconds), and Int (epoch ns near 1.8e18).
        for mode in 0..3 {
            let stamps: Vec<Stamp> = raw_ts
                .iter()
                .map(|&t| match mode {
                    0 => Stamp::Raw(t, 0.0),
                    1 => Stamp::Ns((t * 1e9) as i128),
                    _ => Stamp::Int(1_800_000_000_000_000_000i128 + t as i128, 0.0),
                })
                .collect();
            let mut audit =
                Audit::new(hll_cfg(1, Some(p), None, Some(span), closed, None)).unwrap();

            for i in 0..xs.len() {
                let d_clock = if i == 0 {
                    0.0
                } else {
                    raw_ts[i] - raw_ts[i - 1]
                };
                audit.stamp_next(stamps[i]);
                let step = audit.step(&[xs[i]], &[], d_clock, 1.0);

                // Fresh HLL over all j < i inside window at stamps[i].
                let mut ref_before = Hll::new(p);
                for j in 0..i {
                    if !closed.has_left(stamps[i], stamps[j], span) && crate::usable(xs[j]) {
                        ref_before.add_key(key(xs[j]), p);
                    }
                }
                assert_eq!(
                    step.pred[0].to_bits(),
                    ref_before.count(p).to_bits(),
                    "closed={closed:?}, mode={mode}, row={i}"
                );

                // Check predict_at at future stamp (stamps[i] + 10.0) without mutating state.
                let future = match stamps[i] {
                    Stamp::Raw(t, r) => Stamp::Raw(t + 10.0, r),
                    Stamp::Ns(ns) => Stamp::Ns(ns + 10_000_000_000),
                    Stamp::Int(w, f) => Stamp::Int(w + 10, f),
                };
                let pred_fut = audit.predict_at(&[0.0], 10.0, future);
                let mut ref_fut = Hll::new(p);
                for j in 0..=i {
                    if !closed.has_left(future, stamps[j], span) && crate::usable(xs[j]) {
                        ref_fut.add_key(key(xs[j]), p);
                    }
                }
                assert_eq!(
                    pred_fut.pred[0].to_bits(),
                    ref_fut.count(p).to_bits(),
                    "predict_at mismatch closed={closed:?}, mode={mode}, row={i}"
                );

                // Column report equals window at stamps[i] including row i.
                let mut ref_after = Hll::new(p);
                for j in 0..=i {
                    if !closed.has_left(stamps[i], stamps[j], span) && crate::usable(xs[j]) {
                        ref_after.add_key(key(xs[j]), p);
                    }
                }
                assert_eq!(
                    audit.column(0).approx_distinct.unwrap().to_bits(),
                    ref_after.count(p).to_bits(),
                    "column report closed={closed:?}, mode={mode}, row={i}"
                );
            }
        }
    }
}

#[test]
fn zero_weight_and_nonpositive_weights_reach_no_hll_register() {
    let xs = values(55, 60);
    // Test plain, sliding rows, and sliding clock.
    let configs = [
        hll_cfg(1, Some(8), None, None, WindowClosed::Right, None),
        hll_cfg(1, Some(8), Some(15), None, WindowClosed::Right, None),
        hll_cfg(1, Some(8), None, Some(10.0), WindowClosed::Right, None),
    ];
    for c in configs {
        let mut with_zero = Audit::new(c.clone()).unwrap();
        let mut without_zero = Audit::new(c.clone()).unwrap();
        let mut t = 0.0;
        for (i, &v) in xs.iter().enumerate() {
            t += 1.0;
            let stamp = Stamp::Raw(t, 0.0);
            if i % 3 == 0 {
                // Zero-weight row: must still emit valid out-of-sample prediction,
                // reach no register, and leave state bit-identical.
                with_zero.stamp_next(stamp);
                let pred_before = with_zero.predict_at(&[v], 1.0, stamp);
                with_zero.stamp_next(stamp);
                let stepped = with_zero.step(&[v], &[], 1.0, 0.0);
                assert_eq!(stepped.pred[0].to_bits(), pred_before.pred[0].to_bits());
            } else {
                with_zero.stamp_next(stamp);
                let a = with_zero.step(&[v], &[], 1.0, 1.0);
                without_zero.stamp_next(stamp);
                let b = without_zero.step(&[v], &[], 1.0, 1.0);
                assert_eq!(a.pred[0].to_bits(), b.pred[0].to_bits());
                assert_eq!(a.n_eff.to_bits(), b.n_eff.to_bits());
            }
        }
        assert_eq!(
            with_zero.column(0).approx_distinct.unwrap().to_bits(),
            without_zero.column(0).approx_distinct.unwrap().to_bits()
        );
        assert_eq!(with_zero, without_zero);
    }

    // Negative, NaN, +inf, and >INPUT_BOUND weights reach no HLL register either.
    let mut m = Audit::new(hll_cfg(1, Some(8), None, None, WindowClosed::Right, None)).unwrap();
    for &bad_w in &[-1.0, f64::NAN, f64::INFINITY, 1e200] {
        m.step(&[123.0 + bad_w.abs().min(1.0)], &[], 1.0, bad_w);
    }
    assert_eq!(m.column(0).approx_distinct, Some(0.0));
}

#[test]
fn clock_window_merge_matches_union_and_row_window_merge_is_refused() {
    let p = 8u8;
    let span = 12.0;
    let c = hll_cfg(1, Some(p), None, Some(span), WindowClosed::Right, None);
    let mut a = Audit::new(c.clone()).unwrap();
    let mut b = Audit::new(c.clone()).unwrap();
    let mut union = Audit::new(c).unwrap();

    let rows_a = [(0.0, 1.0), (4.0, 2.0), (10.0, 3.0), (16.0, 4.0)];
    let rows_b = [(2.0, 5.0), (8.0, 2.0), (14.0, 6.0), (20.0, 7.0)];
    for &(t, v) in &rows_a {
        a.stamp_next(Stamp::Raw(t, 0.0));
        a.step(&[v], &[], 1.0, 1.0);
    }
    for &(t, v) in &rows_b {
        b.stamp_next(Stamp::Raw(t, 0.0));
        b.step(&[v], &[], 1.0, 1.0);
    }
    let mut all: Vec<(f64, f64)> = rows_a.iter().chain(&rows_b).copied().collect();
    all.sort_by(|x, y| x.0.total_cmp(&y.0));
    for &(t, v) in &all {
        union.stamp_next(Stamp::Raw(t, 0.0));
        union.step(&[v], &[], 1.0, 1.0);
    }
    let mut ab = a.clone();
    ab.merge(&b).unwrap();
    let mut ba = b.clone();
    ba.merge(&a).unwrap();
    assert_eq!(
        ab.column(0).approx_distinct.unwrap().to_bits(),
        union.column(0).approx_distinct.unwrap().to_bits()
    );
    assert_eq!(
        ba.column(0).approx_distinct.unwrap().to_bits(),
        union.column(0).approx_distinct.unwrap().to_bits()
    );

    // Row-window merge is refused with a clear message.
    let rc = hll_cfg(1, Some(8), Some(10), None, WindowClosed::Right, None);
    let mut r1 = Audit::new(rc.clone()).unwrap();
    let r2 = Audit::new(rc).unwrap();
    let err = r1.merge(&r2).unwrap_err();
    assert!(err.contains("distinct_window_rows"), "{err}");

    // Mismatched precision is refused.
    let mut p8 = Audit::new(hll_cfg(1, Some(8), None, None, WindowClosed::Right, None)).unwrap();
    let p9 = Audit::new(hll_cfg(1, Some(9), None, None, WindowClosed::Right, None)).unwrap();
    assert!(p8.merge(&p9).is_err());
}

#[test]
fn restart_keeps_plain_and_row_window_but_clears_clock_window() {
    let mut plain = Audit::new(hll_cfg(1, Some(8), None, None, WindowClosed::Right, None)).unwrap();
    let mut rows = Audit::new(hll_cfg(
        1,
        Some(8),
        Some(10),
        None,
        WindowClosed::Right,
        None,
    ))
    .unwrap();
    let mut clock = Audit::new(hll_cfg(
        1,
        Some(8),
        None,
        Some(10.0),
        WindowClosed::Right,
        None,
    ))
    .unwrap();
    for i in 0..5 {
        let v = i as f64;
        plain.step(&[v], &[], 1.0, 1.0);
        rows.step(&[v], &[], 1.0, 1.0);
        clock.stamp_next(Stamp::Raw(v, 0.0));
        clock.step(&[v], &[], 1.0, 1.0);
    }
    let p_before = plain.column(0).approx_distinct;
    let r_before = rows.column(0).approx_distinct;
    plain.restart();
    rows.restart();
    clock.restart();
    assert_eq!(plain.column(0).approx_distinct, p_before);
    assert_eq!(rows.column(0).approx_distinct, r_before);
    assert_eq!(clock.column(0).approx_distinct, Some(0.0));
    // Clock window can now take fresh epoch stamps starting back at 0.
    clock.stamp_next(Stamp::Raw(0.0, 0.0));
    clock.step(&[99.0], &[], 0.0, 1.0);
    assert!(clock.column(0).approx_distinct.unwrap() > 0.5);
}

#[test]
fn save_and_restore_continue_bit_identically_in_named_and_compact_msgpack() {
    let xs = values(777, 80);
    let configs = [
        hll_cfg(2, Some(8), None, None, WindowClosed::Right, None),
        hll_cfg(2, Some(8), Some(20), None, WindowClosed::Right, None),
        hll_cfg(2, Some(8), None, Some(15.0), WindowClosed::Both, None),
    ];
    for c in configs {
        let mut whole = Audit::new(c.clone()).unwrap();
        let mut half = Audit::new(c).unwrap();
        for (i, &v) in xs.iter().enumerate() {
            let row = [v, (i % 7) as f64];
            let s = Stamp::Ns((i as i128) * 1_000_000_000);
            whole.stamp_next(s);
            whole.step(&row, &[], 1.0, 1.0);
            if i < 40 {
                half.stamp_next(s);
                half.step(&row, &[], 1.0, 1.0);
            }
        }
        // Both named and compact msgpack encodings round-trip and continue identically.
        for compact in [false, true] {
            let bytes = if compact {
                rmp_serde::to_vec(&half.state()).unwrap()
            } else {
                rmp_serde::to_vec_named(&half.state()).unwrap()
            };
            let state: State = rmp_serde::from_slice(&bytes).unwrap();
            let mut resumed = Audit::restore(&state).unwrap();
            assert_eq!(resumed, half);
            for (i, &v) in xs.iter().enumerate().skip(40) {
                let row = [v, (i % 7) as f64];
                let s = Stamp::Ns((i as i128) * 1_000_000_000);
                resumed.stamp_next(s);
                resumed.step(&row, &[], 1.0, 1.0);
            }
            assert_eq!(resumed, whole);
            for col in 0..2 {
                assert_eq!(
                    resumed.column(col).approx_distinct.unwrap().to_bits(),
                    whole.column(col).approx_distinct.unwrap().to_bits()
                );
            }
        }
    }
}

#[test]
fn cfg_validation_refuses_bad_hll_options_and_over_cap_fixed_state() {
    // Precision out of range.
    for bad_p in [0u8, 4, 7, 17, 255] {
        let err = Audit::new(hll_cfg(
            1,
            Some(bad_p),
            None,
            None,
            WindowClosed::Right,
            None,
        ))
        .unwrap_err();
        assert!(err.contains("distinct_precision"), "{err}");
    }
    // Window without distinct_precision.
    let err = Audit::new(hll_cfg(1, None, Some(10), None, WindowClosed::Right, None)).unwrap_err();
    assert!(err.contains("distinct_window_rows"), "{err}");
    let err = Audit::new(hll_cfg(
        1,
        None,
        None,
        Some(10.0),
        WindowClosed::Right,
        None,
    ))
    .unwrap_err();
    assert!(err.contains("distinct_window_size"), "{err}");
    // Window rows == 0.
    let err = Audit::new(hll_cfg(
        1,
        Some(8),
        Some(0),
        None,
        WindowClosed::Right,
        None,
    ))
    .unwrap_err();
    assert!(err.contains("distinct_window_rows"), "{err}");
    // Non-positive or non-finite window_clock.
    for bad_w in [0.0, -5.0, f64::NAN, f64::INFINITY] {
        let err = Audit::new(hll_cfg(
            1,
            Some(8),
            None,
            Some(bad_w),
            WindowClosed::Right,
            None,
        ))
        .unwrap_err();
        assert!(err.contains("distinct_window_size"), "{err}");
    }
    // Both windows set.
    let err = Audit::new(hll_cfg(
        1,
        Some(8),
        Some(10),
        Some(10.0),
        WindowClosed::Right,
        None,
    ))
    .unwrap_err();
    assert!(
        err.contains("distinct_window_rows") && err.contains("distinct_window_size"),
        "{err}"
    );
    // closed = Both without distinct_window_size.
    let err = Audit::new(hll_cfg(
        1,
        Some(8),
        Some(10),
        None,
        WindowClosed::Both,
        None,
    ))
    .unwrap_err();
    assert!(err.contains("closed"), "{err}");
    // max_state_bytes bounds.
    let err = Audit::new(hll_cfg(
        1,
        Some(8),
        None,
        None,
        WindowClosed::Right,
        Some(0),
    ))
    .unwrap_err();
    assert!(err.contains("max_state_bytes"), "{err}");
    let err = Audit::new(hll_cfg(
        1,
        Some(8),
        None,
        None,
        WindowClosed::Right,
        Some(STATE_BYTES_CEILING + 1),
    ))
    .unwrap_err();
    assert!(err.contains("max_state_bytes"), "{err}");
    // Fixed bytes exceed cap (1 column at p=12 plain takes 13,696 B > 10,000 B).
    let err = Audit::new(hll_cfg(
        1,
        Some(12),
        None,
        None,
        WindowClosed::Right,
        Some(10_000),
    ))
    .unwrap_err();
    assert!(err.contains("max_state_bytes"), "{err}");
    // Explicitly setting max_state_bytes without precision enforces cap on base audit state.
    let err = Audit::new(hll_cfg(
        1,
        None,
        None,
        None,
        WindowClosed::Right,
        Some(1_000),
    ))
    .unwrap_err();
    assert!(err.contains("max_state_bytes"), "{err}");
    // Without distinct_precision AND without max_state_bytes, large distinct_cap still works.
    let big = AuditCfg {
        n_columns: 1,
        pairs: false,
        distinct_cap: MAX_DISTINCT_CAP,
        gap_cap: None,
        has_clock: false,
        distinct: None,
    };
    assert!(big.fixed_state_bytes() > DEFAULT_STATE_BYTES);
    assert!(Audit::new(big).is_ok());
}

#[test]
fn runtime_held_cap_warns_at_half_and_refuses_atomically() {
    // 1 column, distinct_cap=16 (base=5,760 B), p=8 sliding clock (fixed=7,808 B).
    // Set cap = 8,208 B -> allows at most 10 live entries (10 * 40 = 400 B).
    let cfg = AuditCfg {
        n_columns: 1,
        pairs: false,
        distinct_cap: 16,
        gap_cap: None,
        has_clock: true,
        distinct: Some(Box::new(AuditDistinctCfg {
            precision: Some(8),
            window_rows: None,
            window_clock: Some(1_000.0),
            closed: WindowClosed::Right,
            max_state_bytes: Some(8_208),
        })),
    };
    assert_eq!(cfg.fixed_state_bytes(), 7_808);
    let mut m = Audit::new(cfg).unwrap();
    let mut warned = false;
    let mut before_breach = None;
    let mut breached_at = None;
    for i in 0..500 {
        let snap = m.clone();
        m.stamp_next(Stamp::Raw(i as f64, 0.0));
        m.step(&[1000.0 + i as f64], &[], 1.0, 1.0);
        if m.take_state_warning().is_some() {
            warned = true;
        }
        if let Some((bytes, cap)) = m.state_over_cap() {
            assert!(bytes > cap && cap == 8_208);
            before_breach = Some(snap);
            breached_at = Some(i);
            break;
        }
    }
    assert!(warned, "expected half-cap warning");
    let snap = before_breach.expect("expected cap refusal");
    let i_breach = breached_at.unwrap();
    // Atomicity: base counts, moments, clock, n_eff, and HLL estimate were NOT mutated
    // by the breaching row!
    assert_eq!(m.n_eff(), snap.n_eff());
    assert_eq!(m.column(0).rows, snap.column(0).rows);
    assert_eq!(m.column(0).count, snap.column(0).count);
    assert_eq!(
        m.column(0).approx_distinct.unwrap().to_bits(),
        snap.column(0).approx_distinct.unwrap().to_bits()
    );
    // Subsequent rows are also rejected without mutating state.
    m.stamp_next(Stamp::Raw((i_breach + 1) as f64, 0.0));
    m.step(&[9999.0], &[], 1.0, 1.0);
    assert_eq!(m.n_eff(), snap.n_eff());
    assert_eq!(m.column(0).rows, snap.column(0).rows);
}

#[test]
fn corrupt_hll_states_are_refused_on_restore() {
    // 1. Plain HLL: wrong register count and out-of-range rank.
    let mut plain = Audit::new(hll_cfg(1, Some(8), None, None, WindowClosed::Right, None)).unwrap();
    plain.step(&[1.0], &[], 1.0, 1.0);
    let mut bad_len = plain.clone();
    if let Some(AuditHllState {
        sketches: HllSketches::Plain(cols),
        ..
    }) = bad_len.hll.as_deref_mut()
    {
        cols[0] = Hll::from(vec![0u8; 10]);
    }
    assert!(Audit::restore(&bad_len.state()).is_err());

    let mut bad_rank = plain.clone();
    if let Some(AuditHllState {
        sketches: HllSketches::Plain(cols),
        ..
    }) = bad_rank.hll.as_deref_mut()
    {
        let mut regs = vec![0u8; 256];
        regs[0] = 65 - 8 + 1; // 58 > max rank 57
        cols[0] = Hll::from(regs);
    }
    assert!(Audit::restore(&bad_rank.state()).is_err());

    // 2. Missing hll when precision is set.
    let mut missing = plain.clone();
    missing.hll = None;
    assert!(Audit::restore(&missing.state()).is_err());

    // 3. Sliding clock HLL: corrupt serialized entries (non-decreasing rank, expired entry,
    //    non-finite stamp, future stamp).
    let mut clock = Audit::new(hll_cfg(
        1,
        Some(8),
        None,
        Some(10.0),
        WindowClosed::Right,
        None,
    ))
    .unwrap();
    clock.stamp_next(Stamp::Raw(20.0, 0.0));
    clock.step(&[1.0], &[], 1.0, 1.0);
    let bytes = rmp_serde::to_vec_named(&clock.state()).unwrap();
    let val: rmpv::Value = rmpv::decode::read_value(&mut &bytes[..]).unwrap();

    // Helper to replace `entries` inside `SlidingClock.cols[0]` and verify restore refuses.
    let check_bad_entries = |entries: rmpv::Value| {
        let mut v = val.clone();
        fn patch(v: &mut rmpv::Value, entries: &rmpv::Value) {
            if let rmpv::Value::Map(m) = v {
                for (k, val) in m.iter_mut() {
                    if k.as_str() == Some("entries") {
                        *val = entries.clone();
                    } else {
                        patch(val, entries);
                    }
                }
            } else if let rmpv::Value::Array(a) = v {
                for item in a.iter_mut() {
                    patch(item, entries);
                }
            }
        }
        patch(&mut v, &entries);
        let mut buf = Vec::new();
        rmpv::encode::write_value(&mut buf, &v).unwrap();
        let st: State = rmp_serde::from_slice(&buf).unwrap();
        assert!(Audit::restore(&st).is_err(), "accepted corrupt entries");
    };

    let raw_stamp = |t: f64| {
        rmpv::Value::Map(vec![(
            rmpv::Value::from("Raw"),
            rmpv::Value::Array(vec![rmpv::Value::from(t), rmpv::Value::from(0.0)]),
        )])
    };
    // Rank 0 (invalid).
    check_bad_entries(rmpv::Value::Array(vec![rmpv::Value::Array(vec![
        rmpv::Value::from(0u32),
        rmpv::Value::from(0u8),
        raw_stamp(20.0),
    ])]));
    // Non-decreasing rank in same register.
    check_bad_entries(rmpv::Value::Array(vec![
        rmpv::Value::Array(vec![
            rmpv::Value::from(0u32),
            rmpv::Value::from(3u8),
            raw_stamp(15.0),
        ]),
        rmpv::Value::Array(vec![
            rmpv::Value::from(0u32),
            rmpv::Value::from(4u8),
            raw_stamp(18.0),
        ]),
    ]));
    // Expired entry (stamp 5.0 vs last_stamp 20.0 with span 10.0).
    check_bad_entries(rmpv::Value::Array(vec![rmpv::Value::Array(vec![
        rmpv::Value::from(0u32),
        rmpv::Value::from(3u8),
        raw_stamp(5.0),
    ])]));
    // Future entry (stamp 25.0 > last_stamp 20.0).
    check_bad_entries(rmpv::Value::Array(vec![rmpv::Value::Array(vec![
        rmpv::Value::from(0u32),
        rmpv::Value::from(3u8),
        raw_stamp(25.0),
    ])]));
}

#[test]
fn hll_source_calls_no_platform_math_library() {
    // Same guarantee as `garch::tests::no_platform_math_library_is_called`:
    // HLL++ uses `libm::log`, never `f64::ln`, `exp`, `powf`, or `powi`.
    let src = include_str!("hll.rs");
    for bad in [".ln(", ".exp(", ".powf(", ".powi(", ".log("] {
        assert!(!src.contains(bad), "hll.rs calls platform math {bad}");
    }
}
