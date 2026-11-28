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

#[allow(clippy::too_many_arguments)]
fn span_sgd_cfg(
    n_features: usize,
    fit_intercept: bool,
    standardize: bool,
    loss: SgdLoss,
    schedule: LearningRate,
    learning_rate: f64,
    l2: f64,
    clip_gradient: f64,
    target_spread: bool,
) -> SgdCfg {
    SgdCfg {
        n_features,
        n_targets: 1,
        fit_intercept,
        decay: Decay::Halflife(f64::INFINITY),
        loss,
        learning_rate,
        schedule,
        l2,
        min_weight: 0.0,
        clip_gradient,
        constraint: None,
        standardize,
        strict_binary: false,
        huber_scale: crate::HuberScale::Rms,
        huber_scale_decay: None,
        refit_rows: 0,
        min_weight_holds_learning: false,
        target_spread,
        mad_anchor: false,
        refit_clock: None,
        max_state_bytes: None,
        average: false,
        invariant: false,
        l1: 0.0,
    }
}

/// Build an `Sgd` with an open span refit buffer on target 0 (`refit[0].open()`)
/// and optional switched `held` scaler with controlled mean/sd.
fn open_span_sgd(cfg: SgdCfg, held: bool, scaler_mean: &[f64], scaler_sd: &[f64]) -> Sgd {
    let mut m = Sgd::new(cfg).unwrap();
    // Open a span refit buffer directly so `span_row_delta` checks target 0
    // for any loss/schedule/standardize combination.
    m.refit = vec![RefitBuf::new()];
    if m.scaler.is_some() {
        let off = usize::from(m.cfg.fit_intercept);
        let n = m.cfg.n_features;
        let mut sc = EwDiag::new(off + n);
        // Feed 24 balanced pairs `(mean - sd, mean + sd)` with equal weight so
        // mean == scaler_mean and sd == scaler_sd (exact variance `sd^2`).
        let mut lo = vec![1.0; off + n];
        let mut hi = vec![1.0; off + n];
        for i in 0..n {
            lo[off + i] = scaler_mean[i] - scaler_sd[i];
            hi[off + i] = scaler_mean[i] + scaler_sd[i];
        }
        let pairs = if held { 20 } else { 2 };
        for _ in 0..pairs {
            sc.update(&lo, 1.0, 1.0);
            sc.update(&hi, 1.0, 1.0);
            m.warmup.learn(1.0);
            m.warmup.learn(1.0);
        }
        assert_eq!(m.warmup.switched(), held);
        m.scaler = Some(sc);
    }
    m
}

/// Helper to set target 0's `TargetSpread` to exact standard deviation `unit`.
fn set_unit(m: &mut Sgd, unit: f64) {
    m.spread = crate::spread::TargetSpread::new(1);
    if unit > 0.0 {
        m.spread.update(0, Some(-unit), 1.0, 1.0);
        m.spread.update(0, Some(unit), 1.0, 1.0);
    }
    assert!((m.unit(0) - unit).abs() <= 1e-12 * unit.max(1.0));
}

#[test]
fn span_row_delta_scaler_resize_reads_z_inv_scaling_decay_clip_and_ada_eps() {
    let per = 100u64;

    // 1. `mbuf.resize(off + n, 0.0)` and `sbuf.resize(off + n, 1.0)` under `held && off == 1 && n == 2`.
    let m2 = open_span_sgd(
        span_sgd_cfg(
            2,
            true,
            true,
            SgdLoss::Squared,
            LearningRate::Constant,
            0.1,
            0.0,
            1e6,
            false,
        ),
        true,
        &[0.0, 0.0],
        &[1.0, 1.0],
    );
    assert_eq!(
        m2.span_row_delta(&[1.0, 2.0], &[Some(1.0)], 1.0, 1.0, 1.0, per),
        per
    );

    // 2. `reads_z = self.scaler.is_some() && !held`:
    //    - `held = true`: `p` must read raw `x`, not standardized `z`.
    //      With `x = [1e160]` (sd = 1e160 so `z ≈ 1`), `beta = [0, 1e160]`, `clip = inf`:
    //      `dot(beta, x) = +inf` => `d = +inf` => returns 0; reading `z` stays finite => `per`.
    let mut m_held = open_span_sgd(
        span_sgd_cfg(
            1,
            true,
            true,
            SgdLoss::Squared,
            LearningRate::Constant,
            1e-300,
            0.0,
            f64::INFINITY,
            false,
        ),
        true,
        &[0.0],
        &[1e154],
    );
    m_held.beta[0] = vec![0.0, 1e160];
    assert_eq!(
        m_held.span_row_delta(&[1e160], &[Some(0.0)], 1.0, 1.0, 1.0, per),
        0
    );

    //    - `held = false`: `p` must read standardized `z`, not raw `x`.
    let mut m_unheld = open_span_sgd(
        span_sgd_cfg(
            1,
            true,
            true,
            SgdLoss::Squared,
            LearningRate::Constant,
            1e-300,
            0.0,
            f64::INFINITY,
            false,
        ),
        false,
        &[0.0],
        &[1e100],
    );
    m_unheld.beta[0] = vec![0.0, 1e160];
    assert_eq!(
        m_unheld.span_row_delta(&[1e150], &[Some(0.0)], 1.0, 1.0, 1.0, per),
        per
    );

    // 3. `LearningRate::InvScaling`: `lr0 / (1.0 + lam * w_sum).powf(power)`.
    //    `lr0 = 2.0, lam = 0.5, w_sum = 6.0, power = 1.0` => exact `lr = 0.5`.
    //    With `off = 0, held = true, scaler_sd = [1.0]`, a row at `x = [1.0]` has `z ≈ 1.0, sbuf ≈ 1.0`.
    let mut m_inv = open_span_sgd(
        span_sgd_cfg(
            1,
            false,
            true,
            SgdLoss::Squared,
            LearningRate::InvScaling { power: 1.0 },
            2.0,
            0.0,
            f64::INFINITY,
            false,
        ),
        true,
        &[0.0],
        &[1.0],
    );
    m_inv.w_sum = 6.0;
    // Choose `x = [1.0]`: `zbuf[0] = x / sqrt(raw) = 1.0`, `sbuf[0] = 1.0`.
    // Call A: `b[0] = 1.0e308`, `p = 1.0e308`, `y = 2.5e308`? Wait keep `y` finite:
    // use `x = [-1.0]` (`zbuf = -1.0, p = -1.0e308`) and `y = 0.5e308` => `d = -1.5e308`,
    // `g = d * zbuf * weight = +1.5e308`? We want `g = -1.5e308` so `step = +0.75e308`:
    // `b[0] = 1.0e308, x = [1.0], p = 1.0e308`, wait `d = p - y`: pick `y = 1.75e308` =>
    // `d = -0.75e308`, `weight = 2.0` => `g = -1.5e308`, `step = +0.75e308`, `b + step = 1.75e308` finite!
    // Mutants `%` (lr=2), `*` (lr=8), `+ -> *` (lr=2/3), `* -> /` (lr=1.846) all give `b + step >= 2.0e308 = +inf`!
    m_inv.beta[0] = vec![1.0e308];
    assert_eq!(
        m_inv.span_row_delta(&[1.0], &[Some(1.75e308)], 2.0, 0.5, 1.0, per),
        per
    );
    // Call B: `b[0] = -1.1e308`, `x = [1.0]`, `p = -1.1e308`, `y = -1.75e308` => `d = +0.65e308`?
    // Want `d = +0.75e308` => `y = -1.85e308` overflows! Use `b[0] = -1.1e308, x = [-1.0]` (`z = -1, p = +1.1e308`),
    // `y = +0.35e308` => `d = +0.75e308`, `g = d * (-1) * 2 = -1.5e308`, `step = +0.75e308` (opposite sign).
    // Flip `y`: `b[0] = -1.1e308, x = [-1.0], p = +1.1e308`, `y = +1.7e308` => `d = -0.6e308`, `g = +1.2e308`?
    // With `weight = 2.5`: `g = (-0.6e308) * (-1.0) * 2.5 = +1.5e308`, `step = -0.75e308`, `b + step = -1.85e308 = -inf`!
    // Meanwhile `* -> +` (`lr = 2/7.5 = 0.2667`) gives `step = -0.4e308`, `b + step = -1.5e308` finite;
    // `+ -> -` (`lr = -1.0`) gives `step = +1.5e308`, `b + step = +0.4e308` finite!
    m_inv.beta[0] = vec![-1.1e308];
    assert_eq!(
        m_inv.span_row_delta(&[-1.0], &[Some(1.7e308)], 2.5, 0.5, 1.0, per),
        0
    );

    // 4. `dec = |s| if lam != 1.0 { lam * s } else { s }`, `weight <= 0.0 || !yj.is_finite()`.
    let mut m_dec = open_span_sgd(
        span_sgd_cfg(
            1,
            false,
            false,
            SgdLoss::Squared,
            LearningRate::AdaGrad,
            0.1,
            0.0,
            f64::INFINITY,
            false,
        ),
        false,
        &[],
        &[],
    );
    // `weight <= 0.0 || !yj.is_finite()` each alone returns 0.
    assert_eq!(
        m_dec.span_row_delta(&[1.0], &[Some(1.0)], 0.0, 1.0, 1.0, per),
        0
    );
    assert_eq!(
        m_dec.span_row_delta(&[1.0], &[Some(f64::NAN)], 1.0, 1.0, 1.0, per),
        0
    );
    // `sums[0] = 1.2e308`, `lam = 0.5`: `dec(sums) = 0.6e308`.
    m_dec.g2[0] = vec![1.2e308];
    // `g = 1.0e154` => `g^2 = 1.0e308` => `0.6e308 + 1.0e308 = 1.6e308` finite (`!= -> ==` and `* -> +` overflow).
    assert_eq!(
        m_dec.span_row_delta(&[1.0], &[Some(1.0e154)], 1.0, 0.5, 1.0, per),
        per
    );
    // `g = 1.15e154` => `g^2 = 1.3225e308` => `0.6e308 + 1.3225e308 = +inf` (`* -> /` stays finite).
    assert_eq!(
        m_dec.span_row_delta(&[1.0], &[Some(1.15e154)], 1.0, 0.5, 1.0, per),
        0
    );

    // 5. `clip = if unit > 0.0 { clip0 * unit } else { f64::INFINITY }`.
    let mut m_clip = open_span_sgd(
        span_sgd_cfg(
            1,
            false,
            false,
            SgdLoss::Squared,
            LearningRate::AdaGrad,
            0.1,
            0.0,
            0.5e154,
            true,
        ),
        false,
        &[],
        &[],
    );
    // `unit == 0.0` => `clip = inf`: raw `d * w = +inf` => 0 (mutants `==` / `>=` clamp to 0 => `per`).
    set_unit(&mut m_clip, 0.0);
    assert_eq!(
        m_clip.span_row_delta(&[1.0], &[Some(1e200)], 1e200, 1.0, 1.0, per),
        0
    );
    // `unit = 2.0, clip0 = 0.5e154` => `clip = 1.0e154` => `g^2 = 1.0e308` finite (`<` gives `clip = inf` => 0).
    set_unit(&mut m_clip, 2.0);
    assert_eq!(
        m_clip.span_row_delta(&[1.0], &[Some(1e200)], 1.0, 1.0, 1.0, per),
        per
    );
    // `unit = 2.0, clip0 = 1.0e154` => `clip = 2.0e154` => `g^2 = 4.0e308 = +inf` (`* -> +` gives 1e154, `* -> /` gives 0.5e154, both finite!).
    m_clip.cfg.clip_gradient = 1.0e154;
    assert_eq!(
        m_clip.span_row_delta(&[1.0], &[Some(1e200)], 1.0, 1.0, 1.0, per),
        0
    );

    // 6. `ada_eps = 1e-8 * unit`: under `held = true`, AdaGrad, `sums = 0`, `g = 1e-160` (`g^2 = 0`, `s = 0`).
    let mut m_eps = open_span_sgd(
        span_sgd_cfg(
            1,
            false,
            true,
            SgdLoss::Squared,
            LearningRate::AdaGrad,
            1e305,
            0.0,
            f64::INFINITY,
            true,
        ),
        true,
        &[0.0],
        &[1.0],
    );
    set_unit(&mut m_eps, 100.0); // `ada_eps = 1e-6`
    // `lr0 = 1e305`: `1e305 / 1e-6 = +inf` => 0 (`* -> +` gives `ada_eps ≈ 100` => finite `per`).
    assert_eq!(
        m_eps.span_row_delta(&[1.0], &[Some(1e-160)], 1.0, 1.0, 1.0, per),
        0
    );
    // `lr0 = 1e300`: `1e300 / 1e-6 = 1e306` finite => `per` (`* -> /` gives `ada_eps = 1e-10` => `1e310 = +inf` => 0).
    m_eps.cfg.learning_rate = 1e300;
    assert_eq!(
        m_eps.span_row_delta(&[1.0], &[Some(1e-160)], 1.0, 1.0, 1.0, per),
        per
    );
}

#[test]
fn span_row_delta_level_warmup_branches() {
    let per = 100u64;

    // 1. `first = lv.w == 0.0` returns `(beta[0] + step).is_finite()` early before slopes.
    let mut m = open_span_sgd(
        span_sgd_cfg(
            1,
            true,
            true,
            SgdLoss::Squared,
            LearningRate::Constant,
            1.0,
            0.0,
            f64::INFINITY,
            true,
        ),
        true,
        &[0.0],
        &[1.0],
    );
    // Overflowing slope `b[1] = 1e308, x = [1e10]` would fail slopes if `first` fell through.
    m.level = vec![Level::default()];
    m.beta[0] = vec![0.0, 1e308];
    assert_eq!(
        m.span_row_delta(&[0.0], &[Some(1.0)], 1.0, 1.0, 1.0, per),
        per
    );
    // With `x = [1e10]`, `p = +inf`, so instead make `x = [1.0], b[1] = 0.0, lr = 1e308` so slopes overflow:
    m.cfg.learning_rate = 1e308;
    m.beta[0] = vec![0.0, 0.0];
    assert_eq!(
        m.span_row_delta(&[1.0], &[Some(10.0)], 1.0, 1.0, 1.0, per),
        per
    );
    // `beta[0] + step`: `b[0] = 1.2e308, y = 0.0 => d = 1.2e308, step = -1.2e308`:
    // `+` = 0.0 (finite => `per`); `*` and `-` overflow => 0.
    m.cfg.learning_rate = 1.0;
    m.beta[0] = vec![1.2e308, 0.0];
    assert_eq!(
        m.span_row_delta(&[0.0], &[Some(0.0)], 1.0, 1.0, 1.0, per),
        per
    );

    // 2. `gain = weight / lv.w` (`%` and `*` mutants): `lv.w = 2.0, weight = 2.0 => gain = 0.5`.
    //    `b[0] = 1.0e308, y = 1.7e308 => d = -0.7e308`, `lr = 2.0` (fallback overflows).
    //    Original: `level_step = +0.35e308`, `b[0] + level_step = 1.35e308` finite (`per`).
    //    `%`: `gain = 2.0 => +1.4e308 => 2.4e308 = +inf` (0).
    //    `*`: `gain = 8.0 => step = +inf => fallback sgd_step = +inf` (0).
    let mut lv2 = Level::default();
    lv2.learn(2.0);
    m.level = vec![lv2];
    m.cfg.learning_rate = 2.0;
    m.beta[0] = vec![1.0e308, 0.0];
    assert_eq!(
        m.span_row_delta(&[0.0], &[Some(1.7e308)], 2.0, 1.0, 1.0, per),
        per
    );

    // 3. `above = lr_row.is_some_and(|lr| gain > lr)`, `above || lv.count() <= WARMUP_ROWS`,
    //    `if warm && step.is_finite()`:
    //    Use uniform `weight = 40.0` rows so Kish's count `lv.count()` equals the row count
    //    exactly (`n_rows + 1`) and `gain = 1 / (n_rows + 1)`.
    //    `x = [0.0], b[0] = 1.75e308, y = 1.79e308 => d = -0.04e308, g = d * 40 = -1.6e308`:
    //    `level_step = +0.04e308 / (n + 1)` (finite `b[0] + level_step ≈ 1.7515e308`),
    //    while `sgd_step = +lr * 1.6e308` overflows `b[0] + sgd_step` for `lr >= 0.99 / 26`!
    let make_lv = |n_rows: usize, w: f64| {
        let mut lv = Level::default();
        for _ in 0..n_rows {
            lv.learn(w);
        }
        lv
    };
    m.beta[0] = vec![1.75e308, 0.0];
    // (a) Past warmup (`count = 26 > 22`), `lr = gain = 1/26`: `above = false, warm = false` => 0.
    //     Kills `gain > lr` -> `==`, `>=`; `<= WARMUP_ROWS` -> `>`; `warm && step.is_finite()` -> `||`.
    let gain26 = 1.0 / 26.0;
    m.level = vec![make_lv(25, 40.0)];
    m.cfg.learning_rate = gain26;
    assert_eq!(
        m.span_row_delta(&[0.0], &[Some(1.79e308)], 40.0, 1.0, 1.0, per),
        0
    );
    // (b) Past warmup (`count = 26 > 22`), `lr = 0.99 * gain < gain`: `above = true, warm = true` => `per`.
    //     Kills `gain > lr` -> `<`; `above || ...` -> `&&`.
    m.cfg.learning_rate = 0.99 * gain26;
    assert_eq!(
        m.span_row_delta(&[0.0], &[Some(1.79e308)], 40.0, 1.0, 1.0, per),
        per
    );
    // (c) At warmup boundary (`count = 22 <= WARMUP_ROWS`, `gain = 1/22`), `lr = 2 * gain > gain`:
    //     `above = false, warm = true` => `per`. Kills `<= WARMUP_ROWS` -> `<` and `above || ...` -> `&&`.
    m.level = vec![make_lv(21, 40.0)];
    m.cfg.learning_rate = 2.0 / 22.0;
    assert_eq!(
        m.span_row_delta(&[0.0], &[Some(1.79e308)], 40.0, 1.0, 1.0, per),
        per
    );

    // 4. `-gain * d` (`delete -`, `* -> +`, `* -> /`):
    //    `gain = 0.5`, `b[0] = 1.5e308, b[1] = -1.5e308, x = [1.0]` (`mbuf[1] = 0, shift = 0, p = 0`),
    //    `y = 0.8e308 => d = -0.8e308`, `lr = 1e-300` (so `db` on slope is tiny).
    //    Original: `step = +0.4e308`, `b[0] + step = 1.9e308 = +inf` => 0.
    //    `delete -`: `-0.4e308 => 1.1e308` finite (`per`);
    //    `* -> +`: `-0.5 - 0.8e308 => 0.7e308` finite (`per`);
    //    `* -> /`: `~0 => 1.5e308` finite (`per`).
    let mut lv1 = Level::default();
    lv1.learn(1.0);
    m.level = vec![lv1];
    m.cfg.learning_rate = 1e-300;
    m.beta[0] = vec![1.5e308, -1.5e308];
    assert_eq!(
        m.span_row_delta(&[1.0], &[Some(0.8e308)], 1.0, 1.0, 1.0, per),
        0
    );

    // 5. Non-Squared/Huber `else` branch `(gain * (yj - p), lv.count() <= WARMUP_ROWS)`:
    //    Use `SgdLoss::Quantile { tau: 0.5 }` (identity link, bounded `d = ±0.5`).
    let mut mq = open_span_sgd(
        span_sgd_cfg(
            1,
            true,
            true,
            SgdLoss::Quantile { tau: 0.5 },
            LearningRate::Constant,
            1e-300,
            0.0,
            f64::INFINITY,
            false,
        ),
        true,
        &[0.0],
        &[1.0],
    );
    mq.level = vec![lv1]; // `lv.w = 1.0`, with `weight = 1.0 => gain = 0.5, count = 2 <= 22`.
    // Call A: `b[0] = 1.5e308, b[1] = -1.7e308, x = [1.0] => p = -0.2e308`, `y = 0.6e308 => y - p = 0.8e308`:
    // Original: `step = +0.4e308`, `b[0] + step = 1.9e308 = +inf` => 0.
    // `* -> /` (~0), `- -> +` (`0.5 * 0.4e308 = 0.2e308 => 1.7e308`), `- -> /` (`-1.5`) all finite (`per`)!
    mq.beta[0] = vec![1.5e308, -1.7e308];
    assert_eq!(
        mq.span_row_delta(&[1.0], &[Some(0.6e308)], 1.0, 1.0, 1.0, per),
        0
    );
    // Call B: `b[0] = 1.5e308, b[1] = -1.6e308, x = [1.0] => p = -0.1e308`, `y = 0.3e308 => y - p = 0.4e308`:
    // Original: `0.5 * 0.4e308 = 0.2e308 => 1.7e308` finite (`per`);
    // `* -> +`: `0.5 + 0.4e308 = 0.4e308 => 1.9e308 = +inf` (0)!
    mq.beta[0] = vec![1.5e308, -1.6e308];
    assert_eq!(
        mq.span_row_delta(&[1.0], &[Some(0.3e308)], 1.0, 1.0, 1.0, per),
        per
    );
    // Call C: past warmup (`count > 22`), `sgd_step` overflows (`lr = 1e308, d = -0.5 => sgd_step = +0.5e308`,
    // `b[0] = 1.5e308 => +inf`), while `level_step` is tiny (`0.05e308`):
    // Original `warm = false` => 0; mutant `<= -> >` sets `warm = true` => `per`!
    mq.level = vec![make_lv(25, 25.0)];
    mq.cfg.learning_rate = 1e308;
    mq.beta[0] = vec![1.5e308, 0.0];
    assert_eq!(
        mq.span_row_delta(&[0.0], &[Some(1.6e308)], 1.0, 1.0, 1.0, per),
        0
    );
}

#[test]
fn span_row_delta_held_path_all_slope_shift_and_intercept_operators() {
    let per = 100u64;

    // A. `held` slope gradient `d * zbuf[i] * weight + l2 * (sbuf[i] * b[i])` and `clamp(-clip, clip)`.
    //    `off = 0, n = 1, scaler_sd = [1.0]`, `x = [2.0] => zbuf[0] ≈ 2.0, sbuf[0] ≈ 1.0` (exact via scaler).
    let mut m = open_span_sgd(
        span_sgd_cfg(
            1,
            false,
            true,
            SgdLoss::Squared,
            LearningRate::Constant,
            1.0,
            0.0,
            f64::INFINITY,
            false,
        ),
        true,
        &[0.0],
        &[1.0],
    );
    // Seed scaler with large weight so one row `x = [2.0]` leaves `sbuf ≈ 1.0, zbuf ≈ 2.0`.
    {
        let sc = m.scaler.as_mut().unwrap();
        sc.update(&[-1.0], 1.0, 1e6);
        sc.update(&[1.0], 1.0, 1e6);
    }
    // 1) `d * zbuf * weight`: `b[0] = 1.0e308`, `x = [2.0]` (`p = 2.0e308` overflows!).
    //    Avoid `p` overflow: use `x = [2.0], b[0] = 0.0, y = 0.15e308` => `d = -0.15e308`,
    //    `zbuf = 2.0`, `weight = 3.0` => `g = -0.9e308`, `lr = 2.5` => `-(lr * g) = +2.25e308 = +inf` => 0!
    //    Meanwhile `d + z`, `d / z`, `+ w`, `/ w` all have `|g| <= 0.45e308 => |lr*g| <= 1.125e308` finite (`per`)!
    m.cfg.learning_rate = 2.5;
    m.beta[0] = vec![0.0];
    assert_eq!(
        m.span_row_delta(&[2.0], &[Some(0.15e308)], 3.0, 1.0, 1.0, per),
        0
    );

    // 2) `... + l2 * (sbuf * b)`: `sbuf = 2.0`, `zbuf = 1.0` (`x = [2.0]`), `b[0] = 0.1e308`,
    //    `p = 0.2e308`, `y = -0.3e308 => d = +0.5e308`, `weight = 1.0`, `l2 = 2.0`:
    //    `d*z*w = 0.5e308`, `l2*(sbuf*b) = 2*(2*0.1e308) = 0.4e308` => `g = 0.9e308`, `lr = 2.5` => `-2.25e308 = -inf` => 0!
    //    `+ -> -` (0.1e308), `l2 * -> /` (0.5e308), `sbuf * b -> +` (0.7e308), `sbuf * b -> /` (0.5e308) all finite (`per`)!
    let mut m_l2 = open_span_sgd(
        span_sgd_cfg(
            1,
            false,
            true,
            SgdLoss::Squared,
            LearningRate::Constant,
            2.5,
            2.0,
            f64::INFINITY,
            false,
        ),
        true,
        &[0.0],
        &[2.0],
    );
    {
        let sc = m_l2.scaler.as_mut().unwrap();
        sc.update(&[-2.0], 1.0, 1e6);
        sc.update(&[2.0], 1.0, 1e6);
    }
    m_l2.beta[0] = vec![0.1e308];
    assert_eq!(
        m_l2.span_row_delta(&[2.0], &[Some(-0.3e308)], 1.0, 1.0, 1.0, per),
        0
    );
    //    `+ -> *`: `d*z*w = 1e160`, `l2*sbuf*b = 1e160` => `+` is finite (`per`), `*` is `+inf` (0)!
    m_l2.cfg.learning_rate = 1.0;
    m_l2.cfg.l2 = 1.0;
    m_l2.beta[0] = vec![0.5e160];
    assert_eq!(
        m_l2.span_row_delta(&[2.0], &[Some(0.0)], 1.0, 1.0, 1.0, per),
        per
    );

    // 3) `.clamp(-clip, clip)` `delete -` and `-(lr * g)` (`delete -`, `* -> +`, `* -> /`):
    //    With `sbuf = 1.0, zbuf = 1.0` (`x = [1.0]`), `b[0] = 1.5e308`, `p = 1.5e308`,
    //    use `y = 1.7e308 => d = -0.2e308`, `weight = 1.0`, `lr = 2.0`:
    //    `g = -0.2e308`, `step = +0.4e308`, `b + db = +1.9e308 = +inf` => 0.
    //    `delete -` (`-0.4e308 => 1.1e308`), `* -> +` (`+0.2e308 => 1.7e308`), `* -> /` (`~0 => 1.5e308`) all finite!
    m.cfg.learning_rate = 2.0;
    m.beta[0] = vec![1.5e308];
    assert_eq!(
        m.span_row_delta(&[1.0], &[Some(1.7e308)], 1.0, 1.0, 1.0, per),
        0
    );
    //    `clamp(-clip, clip)` `delete -`: negative raw gradient clamped to `-1e100`:
    //    `b[0] = 1.5e308`, `x = [1.0]`, `y = 1.5e308 + 1e200`? Use `weight = 1e100, d = -0.1e308` ->
    //    `raw = -inf`, clamped to `-1e100`; `lr = 0.5e208` => `step = +0.5e308 => b + db = +inf` (0),
    //    while `delete -` clamps to `+1e100 => step = -0.5e308 => b + db = 1.0e308` finite (`per`)!
    m.cfg.clip_gradient = 1e100;
    m.cfg.learning_rate = 0.5e208;
    m.beta[0] = vec![1.5e308];
    assert_eq!(
        m.span_row_delta(&[1.0], &[Some(1.7e308)], 1e100, 1.0, 1.0, per),
        0
    );

    // 4) `db = step / sbuf[i]` (`/ -> %, *`) and `!(b[i] + db).is_finite()` (`delete !`, `+ -> *, -`):
    //    `sbuf = 0.5, zbuf = 1.0` (`x = [0.5]`), `b[0] = 1.2e308, p = 0.6e308, y = 1.0e308 => d = -0.4e308`,
    //    `lr = 1.0, weight = 1.0 => step = +0.4e308, db = 0.4e308 / 0.5 = 0.8e308 => b + db = 2.0e308 = +inf` (0).
    //    `/ -> %` (`db < 0.5`), `/ -> *` (`db = 0.2e308`), `+ -> -` (`0.4e308`) all finite (`per`)!
    let mut m_s = open_span_sgd(
        span_sgd_cfg(
            1,
            false,
            true,
            SgdLoss::Squared,
            LearningRate::Constant,
            1.0,
            0.0,
            f64::INFINITY,
            false,
        ),
        true,
        &[0.0],
        &[0.5],
    );
    {
        let sc = m_s.scaler.as_mut().unwrap();
        sc.update(&[-0.5], 1.0, 1e6);
        sc.update(&[0.5], 1.0, 1e6);
    }
    m_s.beta[0] = vec![1.2e308];
    assert_eq!(
        m_s.span_row_delta(&[0.5], &[Some(1.0e308)], 1.0, 1.0, 1.0, per),
        0
    );
    //    Finite call with `b[0] = 1e200, db = 1e200`: `+` finite (`per`), `delete !` and `+ -> *` return 0!
    m_s.beta[0] = vec![1e200];
    assert_eq!(
        m_s.span_row_delta(&[0.5], &[Some(1.0e200)], 1.0, 1.0, 1.0, per),
        per
    );

    // 5) AdaGrad slopes in `held`: `s = dec(sums[i]) + g * g`, `!s.is_finite()`, `-adagrad_step`:
    let mut m_ada = open_span_sgd(
        span_sgd_cfg(
            1,
            false,
            true,
            SgdLoss::Squared,
            LearningRate::AdaGrad,
            1.0,
            0.0,
            f64::INFINITY,
            false,
        ),
        true,
        &[0.0],
        &[1.0],
    );
    {
        let sc = m_ada.scaler.as_mut().unwrap();
        sc.update(&[-1.0], 1.0, 1e6);
        sc.update(&[1.0], 1.0, 1e6);
    }
    // Finite `s = 1e300 + 1e200`: kills `!s.is_finite()` `delete !` and `+ -> *`.
    m_ada.g2[0] = vec![1e300];
    assert_eq!(
        m_ada.span_row_delta(&[1.0], &[Some(1e100)], 1.0, 1.0, 1.0, per),
        per
    );
    // Overflow `s = 1.2e308 + 1.0e308`: kills `+ -> -`, `* -> +`, `* -> /`.
    m_ada.g2[0] = vec![1.2e308];
    assert_eq!(
        m_ada.span_row_delta(&[1.0], &[Some(1e154)], 1.0, 1.0, 1.0, per),
        0
    );
    // `-adagrad_step` `delete -` on slope: `off = 1, b = [-1.5e308, 1.5e308], x = [1.0]`
    // gives exact `p = 0.0, mbuf = 0, shift = 0`, so `y = 1.0 => d = -1.0, g_slope = -1.0`,
    // `-adagrad_step = +0.5e308 => b[1] + db = +2.0e308 = +inf` (0), while `delete -`
    // gives `-0.5e308 => b[1] + db = 1.0e308` finite (`per`)!
    let mut m_ada_sign = open_span_sgd(
        span_sgd_cfg(
            1,
            true,
            true,
            SgdLoss::Squared,
            LearningRate::AdaGrad,
            0.5e308,
            0.0,
            f64::INFINITY,
            false,
        ),
        true,
        &[0.0],
        &[1.0],
    );
    {
        let sc = m_ada_sign.scaler.as_mut().unwrap();
        sc.update(&[1.0, -1.0], 1.0, 1e6);
        sc.update(&[1.0, 1.0], 1.0, 1e6);
    }
    m_ada_sign.beta[0] = vec![-1.5e308, 1.5e308]; // `p = 0.0` at `x = [1.0]`, `mbuf = 0, shift = 0`
    m_ada_sign.g2[0] = vec![0.0, 0.0];
    // `y = 1.0 => d = -1.0, g_slope = -1.0 => -adagrad_step = +0.5e308 => b[1] + db = +2.0e308 = +inf` (0)!
    // Meanwhile `delete -` on slope gives `-0.5e308 => b[1] + db = 1.0e308` finite (`per`)!
    // (And `b[0] = -1.5e308` + intercept step `+0.5e308 = -1.0e308` stays finite!)
    assert_eq!(
        m_ada_sign.span_row_delta(&[1.0], &[Some(1.0)], 1.0, 1.0, 1.0, per),
        0
    );

    // 6) `shift += mbuf[i] * db` (`+= -> *=, -=`, `* -> +, /`):
    //    `n = 2, off = 1, mbuf = [2.0, 2.0], sbuf = [1.0, 1.0]`, `x = [3.0, 3.0]` (`zbuf = [1.0, 1.0]`),
    //    `b = [0.8e308, 0.0, 0.0]` (`p = 0.8e308`), `y = 0.5e308 => d = +0.3e308`, `lr = 1.0, w = 1.0`:
    //    `db1 = db2 = -0.3e308`, `shift = 2*(-0.3e308) + 2*(-0.3e308) = -1.2e308`,
    //    intercept `step0 = -0.3e308`, `b[0] + step0 - shift = 0.5e308 - (-1.2e308) = 1.7e308` (close!).
    //    Bump `b[0] = 1.0e308, y = 0.7e308 => d = +0.3e308`:
    //    `b[0] + step0 - shift = 0.7e308 - (-1.2e308) = 1.9e308 = +inf` => 0!
    //    Check all 4 mutants:
    //    `+= -> *=`: `shift = 0 => 0.7e308` finite (`per`)!
    //    `+= -> -=`: `shift = +1.2e308 => -0.5e308` finite (`per`)!
    //    `* -> +`: `shift = 2*(2 - 0.3e308) = -0.6e308 => 1.3e308` finite (`per`)!
    //    `* -> /`: `shift ≈ 0 => 0.7e308` finite (`per`)!
    let mut m_shift = open_span_sgd(
        span_sgd_cfg(
            2,
            true,
            true,
            SgdLoss::Squared,
            LearningRate::Constant,
            1.0,
            0.0,
            f64::INFINITY,
            false,
        ),
        true,
        &[2.0, 2.0],
        &[1.0, 1.0],
    );
    {
        let sc = m_shift.scaler.as_mut().unwrap();
        sc.update(&[1.0, 1.0, 1.0], 1.0, 1e6);
        sc.update(&[1.0, 3.0, 3.0], 1.0, 1e6);
    }
    m_shift.beta[0] = vec![1.0e308, 0.0, 0.0];
    assert_eq!(
        m_shift.span_row_delta(&[3.0, 3.0], &[Some(0.7e308)], 1.0, 1.0, 1.0, per),
        0
    );

    // 7) `if off == 1 { ... }` inside `held`: intercept `g`, `step` (Constant & AdaGrad),
    //    `b[0] + level_step.unwrap_or(step) - shift`.
    let mut m_int = open_span_sgd(
        span_sgd_cfg(
            1,
            true,
            true,
            SgdLoss::Squared,
            LearningRate::Constant,
            1.0,
            0.0,
            f64::INFINITY,
            false,
        ),
        true,
        &[0.0],
        &[1.0],
    );
    {
        let sc = m_int.scaler.as_mut().unwrap();
        sc.update(&[1.0, -1.0], 1.0, 1e6);
        sc.update(&[1.0, 1.0], 1.0, 1e6);
    }
    // `x = [0.0]` => `zbuf[1] = 0.0`, so slope `g = 0, db = 0, shift = 0`.
    // (a) `g = (d * weight).clamp(-clip, clip)` (`* -> +, /`) and `off == 1` -> `!=`:
    //     `b[0] = 1.0e308, y = 1.5e308 => d = -0.5e308, weight = 2.0 => g = -1.0e308`,
    //     `step = +1.0e308`, `b[0] + step = 2.0e308 = +inf` => 0!
    //     `* -> +` (`g = -0.5e308 => 1.5e308`), `* -> /` (`g = -0.25e308 => 1.25e308`), `off != 1` (skips!) all `per`!
    m_int.beta[0] = vec![1.0e308, 0.0];
    assert_eq!(
        m_int.span_row_delta(&[0.0], &[Some(1.5e308)], 2.0, 1.0, 1.0, per),
        0
    );
    // (b) `clamp(-clip, clip)` `delete -` on intercept:
    //     `clip = 1e100, lr = 0.5e208, b[0] = 1.5e308, d = -0.2e308, weight = 1e100`:
    //     Original: `g = -1e100, step = +0.5e308 => +inf` (0); `delete -`: `g = +1e100, step = -0.5e308 => 1.0e308` (`per`)!
    m_int.cfg.clip_gradient = 1e100;
    m_int.cfg.learning_rate = 0.5e208;
    m_int.beta[0] = vec![1.5e308, 0.0];
    assert_eq!(
        m_int.span_row_delta(&[0.0], &[Some(1.7e308)], 1e100, 1.0, 1.0, per),
        0
    );
    // (c) `-(lr * g)` on intercept (`delete -`, `* -> +, /`):
    //     `clip = inf, lr = 2.0, b[0] = 1.5e308, y = 1.7e308 => g = -0.2e308, step = +0.4e308 => +inf` (0).
    //     `delete -` (`1.1e308`), `* -> +` (`1.7e308`), `* -> /` (`1.5e308`) all `per`!
    m_int.cfg.clip_gradient = f64::INFINITY;
    m_int.cfg.learning_rate = 2.0;
    m_int.beta[0] = vec![1.5e308, 0.0];
    assert_eq!(
        m_int.span_row_delta(&[0.0], &[Some(1.7e308)], 1.0, 1.0, 1.0, per),
        0
    );
    // (d) `b[0] + step - shift` (`delete !`, `+ -> *, -`, `- -> +, /`):
    //     - Finite `b[0] = 1e200, step = 1e200, shift = 0.0`: kills `delete !`, `+ -> *`, `- -> /`!
    m_int.cfg.learning_rate = 1.0;
    m_int.beta[0] = vec![1e200, 0.0];
    assert_eq!(
        m_int.span_row_delta(&[0.0], &[Some(2e200)], 1.0, 1.0, 1.0, per),
        per
    );
    //     - Overflow on `+`: `b[0] = 1.2e308, step = +0.8e308, shift = 0.0`: kills `+ -> -` (`0.4e308` finite)!
    m_int.beta[0] = vec![1.2e308, 0.0];
    assert_eq!(
        m_int.span_row_delta(&[0.0], &[Some(1.6e308)], 2.0, 1.0, 1.0, per),
        0
    );
    //     - Overflow on `- shift`: `b[0] = 1.2e308, step = 0.0, shift = -0.8e308` (use `m_shift`!):
    //       With `m_shift` (`mbuf = [2, 2], x = [3, 3] => z = [1, 1]`): `b = [1.2e308, 0, 0]`,
    //       `d = +0.2e308`, `lr = 1.0` => `step0 = -0.2e308`, `b[0] + step0 = 1.0e308`,
    //       `shift = 2*(-0.2e308) + 2*(-0.2e308) = -0.8e308`:
    //       Wait `1.0e308 - (-0.8e308) = 1.8e308 = +inf` (0), while `- -> +` gives `1.0 - 0.8 = 0.2e308` finite (`per`)!
    m_shift.beta[0] = vec![1.2e308, 0.0, 0.0];
    assert_eq!(
        m_shift.span_row_delta(&[3.0, 3.0], &[Some(1.0e308)], 1.0, 1.0, 1.0, per),
        0
    );

    // (e) AdaGrad intercept in `held`: `s = dec(sums[0]) + g * g` (`+ -> *, -`, `* -> +, /`)
    //     and `-adagrad_step` (`delete -`).
    let mut m_int_ada = open_span_sgd(
        span_sgd_cfg(
            1,
            true,
            true,
            SgdLoss::Squared,
            LearningRate::AdaGrad,
            1.0,
            0.0,
            f64::INFINITY,
            false,
        ),
        true,
        &[0.0],
        &[1.0],
    );
    {
        let sc = m_int_ada.scaler.as_mut().unwrap();
        sc.update(&[1.0, -1.0], 1.0, 1e6);
        sc.update(&[1.0, 1.0], 1.0, 1e6);
    }
    // `x = [0.0]` => slope `z = 0, g = 0`.
    // Finite `s = 1e300 + 1e200`: kills `+ -> *`.
    m_int_ada.g2[0] = vec![1e300, 0.0];
    assert_eq!(
        m_int_ada.span_row_delta(&[0.0], &[Some(1e100)], 1.0, 1.0, 1.0, per),
        per
    );
    // Overflow `s = 1.2e308 + 1.0e308`: kills `+ -> -`, `* -> +`, `* -> /`.
    m_int_ada.g2[0] = vec![1.2e308, 0.0];
    assert_eq!(
        m_int_ada.span_row_delta(&[0.0], &[Some(1e154)], 1.0, 1.0, 1.0, per),
        0
    );
    // `-adagrad_step` `delete -` on intercept:
    // `b = [1.5e308, -1.5e308], x = [1.0]` (`p = 0.0`), `y = 1.0 => d = -1.0`,
    // `sums = [0.0, 1e300]` (so slope `db` is tiny ~`1e-150`, intercept `-adagrad_step = +0.5e308`):
    // Original: `b[0] + 0.5e308 = +inf` (0); `delete -`: `b[0] - 0.5e308 = 1.0e308` finite (`per`)!
    m_int_ada.cfg.learning_rate = 0.5e308;
    m_int_ada.beta[0] = vec![1.5e308, -1.5e308];
    m_int_ada.g2[0] = vec![0.0, 1e300];
    assert_eq!(
        m_int_ada.span_row_delta(&[1.0], &[Some(1.0)], 1.0, 1.0, 1.0, per),
        0
    );
}

#[test]
fn span_row_delta_non_held_path_and_step_into_cap_addition() {
    let per = 100u64;

    // 1. Non-held intercept (`off == 1` -> `!=`, `d * weight` `* -> +, /`, `.clamp(-clip, clip)` `delete -`,
    //    `(dec(sums[0]) + g * g).is_finite()` `+ -> *, -`).
    let m_const = open_span_sgd(
        span_sgd_cfg(
            1,
            true,
            false,
            SgdLoss::Squared,
            LearningRate::Constant,
            0.1,
            0.0,
            f64::INFINITY,
            false,
        ),
        false,
        &[],
        &[],
    );
    // `d = -1e200, weight = 1e200, x = [1e-200]` => intercept `g = -inf` (0), slope `d*z*w = -1e200` finite!
    // Kills `off == 1` -> `!=`, `d * weight` -> `+, /`!
    assert_eq!(
        m_const.span_row_delta(&[1e-200], &[Some(1e200)], 1e200, 1.0, 1.0, per),
        0
    );
    // Same inputs with `off = 0`: skips intercept => `per` (kills `off == 1` -> `!=` in other direction).
    let m_no_int = open_span_sgd(
        span_sgd_cfg(
            1,
            false,
            false,
            SgdLoss::Squared,
            LearningRate::Constant,
            0.1,
            0.0,
            f64::INFINITY,
            false,
        ),
        false,
        &[],
        &[],
    );
    assert_eq!(
        m_no_int.span_row_delta(&[1e-200], &[Some(1e200)], 1e200, 1.0, 1.0, per),
        per
    );

    // Non-held AdaGrad intercept: `clamp(-clip, clip)` `delete -` and `sums[0] + g*g` (`+ -> *, -`).
    let mut m_ada = open_span_sgd(
        span_sgd_cfg(
            1,
            true,
            false,
            SgdLoss::Squared,
            LearningRate::AdaGrad,
            0.1,
            0.0,
            1e155,
            false,
        ),
        false,
        &[],
        &[],
    );
    // `d = 0.0, l2 = 0.0`, `clip = 1e155`: original clamps 0.0 to 0.0 (`per`),
    // while `delete -` clamps 0.0 to `+1e155` => `g*g = +inf` (0) — kills BOTH intercept & slope `delete -`!
    assert_eq!(
        m_ada.span_row_delta(&[0.0], &[Some(0.0)], 1.0, 1.0, 1.0, per),
        per
    );
    // `sums[0] + g*g` (`+ -> *, -`):
    m_ada.cfg.clip_gradient = f64::INFINITY;
    m_ada.g2[0] = vec![1e200, 0.0];
    assert_eq!(
        m_ada.span_row_delta(&[0.0], &[Some(1e100)], 1.0, 1.0, 1.0, per),
        per
    );
    m_ada.g2[0] = vec![1.2e308, 0.0];
    assert_eq!(
        m_ada.span_row_delta(&[0.0], &[Some(1e154)], 1.0, 1.0, 1.0, per),
        0
    );

    // 2. Non-held AdaGrad slopes (`off = 0`):
    //    `(d * zi * weight + l2 * *bi).clamp(-clip, clip)` and `(dec(s) + g * g).is_finite()`.
    let mut ms = open_span_sgd(
        span_sgd_cfg(
            1,
            false,
            false,
            SgdLoss::Squared,
            LearningRate::AdaGrad,
            0.1,
            0.0,
            f64::INFINITY,
            false,
        ),
        false,
        &[],
        &[],
    );
    // (a) `d * zi * weight` (`* -> +, /` on both `*`):
    //     `d = -0.5e154, zi = 2.0, weight = 2.0 => g = -2.0e154 => g^2 = 4e308 = +inf` (0).
    //     All 4 mutants give `|g| <= 1.0e154 => g^2 <= 1.0e308` finite (`per`)!
    assert_eq!(
        ms.span_row_delta(&[2.0], &[Some(0.5e154)], 2.0, 1.0, 1.0, per),
        0
    );
    // (b) `... + l2 * *bi` (`+ -> *, -`; `l2 * -> +, /`):
    //     `bi = 0.35e154, l2 = 2.0 => l2*bi = 0.7e154`, `zi = 1.0, p = 0.35e154`,
    //     `y = -0.45e154 => d = +0.8e154, w = 1.0 => g = 1.5e154 => g^2 = 2.25e308 = +inf` (0).
    //     `+ -> -` (`0.1e154`), `l2 * -> +` (`1.15e154`), `l2 * -> /` (`0.8e154`) all have `g^2 < 1.33e308` finite (`per`)!
    ms.cfg.l2 = 2.0;
    ms.beta[0] = vec![0.35e154];
    assert_eq!(
        ms.span_row_delta(&[1.0], &[Some(-0.45e154)], 1.0, 1.0, 1.0, per),
        0
    );
    //     `+ -> *`: `d*zi*w = 1e100, l2*bi = 1e100` => `+` gives `2e100` (`g^2 = 4e200` finite => `per`),
    //     `*` gives `1e200` (`g^2 = +inf` => 0)!
    ms.cfg.l2 = 1.0;
    ms.beta[0] = vec![1e100];
    assert_eq!(
        ms.span_row_delta(&[1.0], &[Some(0.0)], 1.0, 1.0, 1.0, per),
        per
    );
    // (c) `clamp(-clip, clip)` `delete -` on slope (off = 0):
    ms.cfg.l2 = 0.0;
    ms.beta[0] = vec![0.0];
    ms.cfg.clip_gradient = 1e155;
    assert_eq!(
        ms.span_row_delta(&[0.0], &[Some(0.0)], 1.0, 1.0, 1.0, per),
        per
    );
    // (d) `(dec(s) + g * g).is_finite()` (`+ -> *, -`, `* -> +, /`):
    ms.cfg.clip_gradient = f64::INFINITY;
    ms.g2[0] = vec![1e200];
    assert_eq!(
        ms.span_row_delta(&[1.0], &[Some(1e100)], 1.0, 1.0, 1.0, per),
        per
    );
    ms.g2[0] = vec![1.2e308];
    assert_eq!(
        ms.span_row_delta(&[1.0], &[Some(1e154)], 1.0, 1.0, 1.0, per),
        0
    );

    // 3. `step_into`: exact first-row cap breach (`held = 32` stamp + `one_row = 40` => `72 > 71`).
    let mut m_cap = Sgd::new(SgdCfg {
        refit_clock: Some(10.0),
        ..span_sgd_cfg(
            1,
            true,
            false,
            SgdLoss::Huber { delta: 1.345 },
            LearningRate::AdaGrad,
            0.05,
            0.0,
            1e3,
            false,
        )
    })
    .unwrap();
    let held0 = m_cap.refit[0].held_bytes();
    let one_row = 8 * (1 + 4) as u64; // Rms scale (`mad` empty) => 40 bytes
    let cap = held0 + one_row - 1;
    m_cap.set_cap_for_test(cap);
    m_cap.step(&[1.0], &[Some(2.0)], 1.0, 1.0);
    assert_eq!(m_cap.state_over_cap(), Some((held0 + one_row, cap)));
}
