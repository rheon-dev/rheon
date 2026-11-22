//! Tests that hold the scaler's floor and the lanes' choices at their edges
//! (mutation testing of `sgd.rs`): the floor against a history, the small
//! share's choice of the deviation's form, and the scale's choice at zero.

use super::*;

fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

/// The floor raises the scaler's decay only where the history's decayed
/// weight is strictly below it and there is a history: at the tie, where
/// `floor / W` can round a bit above `lam`, and with no history at all,
/// where `floor / 0` is infinite, the decay is `lam`, to the bit.
#[test]
fn the_floor_binds_only_below_it_and_only_over_a_history() {
    // No history, a floor above `lam · 0`: the decay is `lam`.
    let empty = EwDiag::new(1);
    for lam in [0.9, 0.5, 1.0] {
        assert_eq!(scaler_decay(&empty, lam, 1.0, 1.0).to_bits(), lam.to_bits());
    }
    // The tie `lam · W == floor`, found where dividing back rounds up.
    let mut found = 0;
    for h in [1.5, 2.0, 3.0, 5.0, 7.0, 11.0, 13.0, 17.0, 19.0, 23.0] {
        let mut sc = EwDiag::new(1);
        sc.update(&[1.0], 1.0, h);
        assert_eq!(sc.n_eff(), h);
        for i in 1..200 {
            let lam = 0.5 + f64::from(i) * 0.0025;
            let floor = lam * h;
            if floor / h > lam {
                found += 1;
                assert_eq!(
                    scaler_decay(&sc, lam, 1.0, floor).to_bits(),
                    lam.to_bits(),
                    "lam {lam} h {h}"
                );
            }
        }
    }
    assert!(found > 0, "no tie where the division rounds up");
    // Above the tie it binds: the history keeps its share, `floor / W`.
    let mut sc = EwDiag::new(1);
    sc.update(&[1.0], 1.0, 4.0);
    assert_eq!(scaler_decay(&sc, 0.25, 1.0, 2.0), 0.5);
    assert_eq!(scaler_decay(&sc, 0.25, 1.0, 100.0), 1.0);
    assert_eq!(scaler_decay(&sc, 0.25, 1.0, 1.0), 0.25);
}

/// The lanes choose the deviation's form as `Including::deviation_of`
/// does, and where the choice is made: `a·d` for a history's share `a`
/// in `(0, 2⁻¹⁰)` -- strictly -- and the pair's otherwise, including at
/// `a = 2⁻¹⁰` itself and at `a = 0`, an empty history, where `a·d` would
/// give `-0` for a negative `d`. Widths from `LANES_FROM` with every
/// remainder, with and without the intercept's slot, each held to the
/// loop's bits.
#[test]
fn the_small_share_is_chosen_alike_by_the_lanes_and_the_loop() {
    let same = |a: f64, b: f64| a.to_bits() == b.to_bits();
    let mut checked = 0usize;
    // Per share: the cases in which the two forms of the deviation part
    // in a bit of `z`, at a lane and at a feature past the last one.
    let mut told_apart = std::collections::BTreeMap::<u64, (usize, usize)>::new();
    // (history weight, row weight admitted): the share is
    // `lam·W / (lam·W + w)`, `lam` 1: 0, 1e-6, 1/1024 exactly, a hair
    // above it, and ordinary.
    for (hist, admit) in [
        (0.0, 1.0),
        (1.0, 1e6),
        (1.0, 1023.0),
        (1.0, 1022.0),
        (1.0, 1.0),
    ] {
        for k in [12usize, 13, 15, 16, 19, 24] {
            for off in [0usize, 1] {
                let slots = k + off;
                let mut sc = EwDiag::new(slots);
                let mut st = 11u64 + k as u64;
                if hist > 0.0 {
                    let seed_row: Vec<f64> = (0..slots)
                        .map(|j| 0.3 + 0.7 * lcg(&mut st) + j as f64)
                        .collect();
                    sc.update(&seed_row, 1.0, hist);
                }
                // Negative values among them, for the sign of a zero.
                let x: Vec<f64> = (0..k)
                    .map(|j| {
                        let v = (0.2 + lcg(&mut st).abs()) * 3.0;
                        if j % 2 == 0 { -v } else { v }
                    })
                    .collect();
                let lam = 1.0;
                let (mut z1, mut m1, mut s1) = (vec![0.0; k], vec![0.0; k], vec![0.0; k]);
                standardize_row(&sc, off, &x, lam, admit, 0.0, &mut z1, &mut m1, &mut s1);
                let inc = sc.including_at(scaler_decay(&sc, lam, admit, 0.0), admit);
                let (hi, lo, c) = sc.parts();
                let (mut z2, mut m2, mut s2) = (vec![0.0; k], vec![0.0; k], vec![0.0; k]);
                standardize_scalar(
                    &inc,
                    off,
                    &x,
                    &hi[off..off + k],
                    &lo[off..off + k],
                    &c[off..off + k],
                    &mut z2,
                    &mut m2,
                    &mut s2,
                );
                let a = hist / (hist + admit);
                for i in 0..k {
                    assert!(
                        same(z1[i], z2[i]) && same(m1[i], m2[i]) && same(s1[i], s2[i]),
                        "hist {hist} admit {admit} k {k} off {off} i {i}: z {:?}/{:?}",
                        z1[i],
                        z2[i]
                    );
                    checked += 1;
                    // Where the two forms of the deviation differ in a
                    // bit, which the lanes could mistake for each other.
                    if off == 1 {
                        let d = crate::comp::dev(x[i], hi[off + i], lo[off + i]);
                        let b = admit / (hist + admit);
                        let (mut h2, mut l2) = (hi[off + i], lo[off + i]);
                        crate::comp::add(&mut h2, &mut l2, b * d);
                        let pair = crate::comp::dev(x[i], h2, l2);
                        if !same(a * d, pair) {
                            let e = told_apart.entry(a.to_bits()).or_default();
                            if i >= k - k % 4 {
                                e.1 += 1;
                            } else {
                                e.0 += 1;
                            }
                        }
                    }
                }
            }
        }
    }
    assert_eq!(checked, 5 * 2 * (12 + 13 + 15 + 16 + 19 + 24));
    // Each share that decides the form is one the forms part at, in a lane
    // and past the last, so that the lanes' choice is held at each.
    for share in [0.0f64, 1.0 / 1024.0, 1.0 / (1.0 + 1e6)] {
        let (lane, tail) = told_apart
            .get(&share.to_bits())
            .copied()
            .unwrap_or_default();
        assert!(
            lane > 0 && tail > 0,
            "share {share:e}: {lane} lanes, {tail} tails"
        );
    }
}

/// The scale's choice at its edges, held on every platform through
/// `standardize_row` (the lanes, and the features past the last full
/// vector, on an x86-64 CPU with AVX2 from `LANES_FROM` features; the
/// loop elsewhere) and through the loop itself. Four kinds of feature,
/// each at every position mod 4 so that each is once a lane and once
/// past the last: constant 0 (raw second moment exactly 0 through the
/// origin, variance exactly 0 with it), constant 2, constant 1e-150 (a
/// raw moment of 1e-300, positive and small), and one with spread.
/// Through the origin `raw > 0` decides the root against 1; with the
/// intercept `v > 0` does, and `v` of exactly 0 gives scale 1 and a
/// `z` of 0, where a scale of 0 would give 0/0.
#[test]
fn the_scale_choice_holds_at_zero_and_just_above_it_in_lanes_and_tail() {
    let hist = |kind: usize, r: usize| match kind {
        0 => 0.0,
        1 => 2.0,
        2 => 1e-150,
        _ => {
            if r.is_multiple_of(2) {
                1.0
            } else {
                3.0
            }
        }
    };
    let mut checked = 0usize;
    for k in 12..=19usize {
        for shift in 0..4usize {
            for off in [0usize, 1] {
                let slots = k + off;
                let mut sc = EwDiag::new(slots);
                for r in 0..6 {
                    // Slot 0 is the intercept's column when `off` is 1.
                    let x: Vec<f64> = (0..slots)
                        .map(|j| {
                            if j < off {
                                1.0
                            } else {
                                hist((j - off + shift) % 4, r)
                            }
                        })
                        .collect();
                    sc.update(&x, 0.9, 1.0);
                }
                // The row is each constant feature's own value, so that
                // it adds no deviation; the feature with spread gets 2.
                let x: Vec<f64> = (0..k)
                    .map(|j| match (j + shift) % 4 {
                        0 => 0.0,
                        1 => 2.0,
                        2 => 1e-150,
                        _ => 2.0,
                    })
                    .collect();
                let (mut z1, mut m1, mut s1) = (vec![9.0; k], vec![9.0; k], vec![9.0; k]);
                standardize_row(&sc, off, &x, 0.95, 1.0, 1.0, &mut z1, &mut m1, &mut s1);
                let inc = sc.including(scaler_decay(&sc, 0.95, 1.0, 1.0));
                let (hi, lo, c) = sc.parts();
                let (mut z2, mut m2, mut s2) = (vec![9.0; k], vec![9.0; k], vec![9.0; k]);
                standardize_scalar(
                    &inc,
                    off,
                    &x,
                    &hi[off..off + k],
                    &lo[off..off + k],
                    &c[off..off + k],
                    &mut z2,
                    &mut m2,
                    &mut s2,
                );
                for (z, m, s) in [(&z1, &m1, &s1), (&z2, &m2, &s2)] {
                    for i in 0..k {
                        let kind = (i + shift) % 4;
                        let ctx = format!("k={k} shift={shift} off={off} i={i} kind={kind}");
                        let near = |a: f64, b: f64| (a - b).abs() <= 1e-12 * b.abs();
                        match (off, kind) {
                            // Raw second moment exactly 0: scale 1, z the
                            // row's own 0, not 0/0.
                            (0, 0) => assert_eq!((z[i], m[i], s[i]), (0.0, 0.0, 1.0), "{ctx}"),
                            (0, 1) => assert_eq!((z[i], m[i], s[i]), (1.0, 0.0, 2.0), "{ctx}"),
                            // 1e-300 is positive: the root 1e-150 is the
                            // scale, not 1, and z is about 1.
                            (0, 2) => {
                                assert!(near(s[i], 1e-150) && near(z[i], 1.0), "{ctx}");
                                assert_eq!(m[i], 0.0, "{ctx}");
                            }
                            (0, _) => assert!(s[i] > 1.0 && z[i].is_finite(), "{ctx}"),
                            // No spread: scale 1, dev 0, mean the constant.
                            (_, 0) => assert_eq!((z[i], m[i], s[i]), (0.0, 0.0, 1.0), "{ctx}"),
                            (_, 1) => assert_eq!((z[i], m[i], s[i]), (0.0, 2.0, 1.0), "{ctx}"),
                            (_, 2) => {
                                assert_eq!((z[i], m[i], s[i]), (0.0, 1e-150, 1.0), "{ctx}")
                            }
                            // Spread: the sd, below 2 here and not 1.
                            (_, _) => assert!(
                                s[i] > 0.0 && s[i] != 1.0 && s[i] < 2.0 && z[i].is_finite(),
                                "{ctx}: s {}",
                                s[i]
                            ),
                        }
                        checked += 1;
                    }
                }
            }
        }
    }
    assert_eq!(checked, 2 * 2 * 4 * (12 + 13 + 14 + 15 + 16 + 17 + 18 + 19));
}
