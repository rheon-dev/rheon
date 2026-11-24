//! `ftrl` under `standardize` (docs/PLAN.md task 569): FTRL run on the row
//! standardized against the features' EW moments with the row admitted, and
//! its coefficients read out in the caller's units with the moments as they
//! stand.
//!
//! The oracle is the definition: the standardized fit is plain FTRL fed the
//! rows standardized longhand -- the EW mean and variance of every row up to
//! and including this one, written out as weighted sums over the history in
//! two passes, which shares no arithmetic with `EwDiag`'s one-pass
//! recursion -- and its coefficients are that fit's, mapped back through
//! the last moments, `b_i = β_i / s_i` and `b_0 = β_0 − Σ m_i b_i`.

use super::*;

fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

fn cfg(k: usize, decay: Decay, standardize: bool) -> FtrlCfg {
    FtrlCfg {
        n_features: k,
        n_targets: 1,
        fit_intercept: true,
        decay,
        alpha: 0.1,
        beta: 1.0,
        l1: 0.0,
        l2: 1.0,
        min_weight: 0.0,
        strict_binary: false,
        loss: FtrlLoss::Squared,
        standardize,
    }
}

/// Features at very different scales and levels: a signal measured in
/// thousandths around 5, and noise measured in thousands around -2,000.
/// The target is `u0 + 0.1·e`.
fn scaled_rows(n: usize, seed: u64) -> Vec<([f64; 2], f64)> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            let (u0, u1, e) = (lcg(&mut s), lcg(&mut s), lcg(&mut s));
            ([5.0 + 1e-3 * u0, -2000.0 + 1e3 * u1], u0 + 0.1 * e)
        })
        .collect()
}

/// The EW mean and (biased, the scaler's) variance of feature `f` over
/// `rows[..=t]`, each row weighted `lam^(t − s)`: two passes, longhand.
fn moments(rows: &[([f64; 2], f64)], t: usize, f: usize, lam: f64) -> (f64, f64) {
    let w = |s: usize| lam.powi((t - s) as i32);
    let total: f64 = (0..=t).map(w).sum();
    let mean = (0..=t).map(|s| w(s) * rows[s].0[f]).sum::<f64>() / total;
    let var = (0..=t)
        .map(|s| w(s) * (rows[s].0[f] - mean).powi(2))
        .sum::<f64>()
        / total;
    (mean, var)
}

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol * (1.0 + a.abs().max(b.abs()))
}

/// The standardized fit is plain FTRL on the longhand-standardized rows,
/// row for row, and its coefficients are that fit's mapped back through
/// the last moments: without decay and under one, with an `l1` that zeroes
/// some coordinates on the way.
#[test]
fn the_standardized_fit_is_ftrl_on_rows_standardized_longhand() {
    for (decay, lam) in [
        (Decay::Halflife(f64::INFINITY), 1.0),
        (Decay::Lam(0.97), 0.97),
    ] {
        let rows = scaled_rows(300, 11);
        let mut c = cfg(2, decay, true);
        c.l1 = 0.5;
        let mut fit = Ftrl::new(c.clone()).unwrap();
        let mut oracle = Ftrl::new(FtrlCfg {
            standardize: false,
            ..c
        })
        .unwrap();
        let mut zeroed = 0;
        for (t, (x, y)) in rows.iter().enumerate() {
            let d = if t == 0 { 0.0 } else { 1.0 };
            let mut z = [0.0; 2];
            for (f, zf) in z.iter_mut().enumerate() {
                let (m, v) = moments(&rows, t, f, lam);
                *zf = if v > 0.0 { (x[f] - m) / v.sqrt() } else { 0.0 };
            }
            let p = fit.step(x, &[Some(*y)], d, 1.0).pred[0];
            let q = oracle.step(&z, &[Some(*y)], d, 1.0).pred[0];
            assert!(close(p, q, 1e-9), "lam {lam}, row {t}: {p} against {q}");
            let beta = &oracle.coefficients()[0];
            zeroed += beta.iter().filter(|b| **b == 0.0).count();
            let mut want = beta.clone();
            let mut b0 = beta[0];
            for f in 0..2 {
                let (m, v) = moments(&rows, t, f, lam);
                let s = if v > 0.0 { v.sqrt() } else { 1.0 };
                want[f + 1] = beta[f + 1] / s;
                b0 -= want[f + 1] * m;
            }
            want[0] = b0;
            let got = &fit.coefficients()[0];
            for i in 0..3 {
                // A slope `l1` zeroes is 0 in the caller's units too; the
                // intercept absorbs the means.
                if i > 0 {
                    assert_eq!(got[i] == 0.0, beta[i] == 0.0, "lam {lam}, row {t}, {i}");
                }
                assert!(
                    close(got[i], want[i], 1e-9),
                    "lam {lam}, row {t}, {i}: {} against {}",
                    got[i],
                    want[i]
                );
            }
        }
        assert!(zeroed > 0, "lam {lam}: l1 zeroed no coordinate on the way");
    }
}

/// The reason for the option: `l1` zeroes a coefficient in the units it is
/// learned in. On the features as they stand, the signal measured in
/// thousandths has gradients a thousandth the size and never clears `l1`,
/// while the noise measured in thousands does; standardized, each is read
/// in units of its own spread, and the signal survives where the noise is
/// zeroed.
#[test]
fn l1_selects_by_the_features_spread_under_standardize() {
    // Centred, so that a level does not carry the signal's gradient.
    let rows: Vec<([f64; 2], f64)> = scaled_rows(2000, 5)
        .into_iter()
        .map(|(x, y)| ([x[0] - 5.0, x[1] + 2000.0], y))
        .collect();
    let fit = |standardize: bool| {
        let mut c = cfg(2, Decay::Halflife(f64::INFINITY), standardize);
        c.l1 = 50.0;
        let mut m = Ftrl::new(c).unwrap();
        for (t, (x, y)) in rows.iter().enumerate() {
            m.step(x, &[Some(*y)], if t == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        m.coefficients()[0].clone()
    };
    let raw = fit(false);
    assert_eq!(raw[1], 0.0, "as they stand the signal is zeroed: {raw:?}");
    assert_ne!(raw[2], 0.0, "as they stand the noise survives: {raw:?}");
    let std = fit(true);
    // The truth is 1 per unit of `u0`, a thousand per unit of `x0`.
    assert!(std[1] > 500.0, "standardized the signal survives: {std:?}");
    assert_eq!(std[2], 0.0, "standardized the noise is zeroed: {std:?}");
}

/// Off, the option is today's model: the same predictions and coefficients
/// to the bit as a cfg that never names it, and a state without a scaler,
/// whose encoding carries none of the new names.
#[test]
fn off_it_is_the_model_without_it() {
    let rows = scaled_rows(200, 3);
    let mut m = Ftrl::new(cfg(2, Decay::Lam(0.98), false)).unwrap();
    for (t, (x, y)) in rows.iter().enumerate() {
        m.step(x, &[Some(*y)], if t == 0 { 0.0 } else { 1.0 }, 1.0);
    }
    assert!(m.scaler.is_none() && no_warmup(&m.warmup));
    let v = serde_json::to_value(&m).unwrap();
    for key in ["scaler", "warmup"] {
        assert!(v.get(key).is_none(), "{key} written: {v}");
    }
    assert!(v["cfg"].get("standardize").is_none(), "{v}");
    // A cfg read without the key is off.
    let mut cfg_v = v["cfg"].clone();
    cfg_v.as_object_mut().unwrap().remove("standardize");
    let back: FtrlCfg = serde_json::from_value(cfg_v).unwrap();
    assert!(!back.standardize);
}

/// `predict` gives the step's number, and a saved and restored model goes
/// on to the bit, on both sides of the warm-up count's 22 (where the
/// count, and with it the row's unit of weight and the scaler's floor,
/// stop moving), across uneven weights, a row of weight 0 and a gap of
/// sixty half-lives, which the scaler's floor holds.
#[test]
fn predict_and_a_restored_state_follow_the_step_to_the_bit() {
    let rows = scaled_rows(120, 17);
    let weight = |t: usize| match t % 7 {
        0 => 0.0,
        3 => 2.5,
        5 => 0.25,
        _ => 1.0,
    };
    let clock = |t: usize| match t {
        0 => 0.0,
        60 => 600.0,
        _ => 1.0,
    };
    for save_at in [5usize, 21, 30, 59, 61] {
        let mut c = cfg(2, Decay::Halflife(10.0), true);
        c.l1 = 0.05;
        let mut m = Ftrl::new(c).unwrap();
        let mut restored: Option<Ftrl> = None;
        let mut crossed = (false, false);
        for (t, (x, y)) in rows.iter().enumerate() {
            let served = m.predict(x, clock(t)).pred;
            let warm = m.warmup.switched();
            let out = m.step(x, &[Some(*y)], clock(t), weight(t)).pred;
            assert_eq!(served[0].to_bits(), out[0].to_bits(), "row {t}");
            if let Some(r) = &mut restored {
                let again = r.step(x, &[Some(*y)], clock(t), weight(t)).pred;
                assert_eq!(
                    again[0].to_bits(),
                    out[0].to_bits(),
                    "save {save_at}, row {t}"
                );
            }
            if t == save_at {
                let bytes = rmp_serde::to_vec(&m.state()).unwrap();
                restored = Some(Ftrl::restore(&rmp_serde::from_slice(&bytes).unwrap()).unwrap());
            }
            crossed.0 |= !warm;
            crossed.1 |= warm;
        }
        assert!(crossed.0 && crossed.1, "the warm-up count did not cross 22");
        let p = m.coefficients()[0].clone();
        assert!(p.iter().all(|v| v.is_finite()), "{p:?}");
        assert_eq!(restored.unwrap().coefficients()[0], p);
    }
}

/// A state whose scaler does not match its `standardize`, or is the wrong
/// width, or a warm-up without a scaler, is refused as damaged.
#[test]
fn a_scaler_that_does_not_match_the_setting_is_refused() {
    let rows = scaled_rows(30, 9);
    let mut on = Ftrl::new(cfg(2, Decay::Lam(0.99), true)).unwrap();
    let mut off = Ftrl::new(cfg(2, Decay::Lam(0.99), false)).unwrap();
    for (x, y) in &rows {
        on.step(x, &[Some(*y)], 1.0, 1.0);
        off.step(x, &[Some(*y)], 1.0, 1.0);
    }
    let refused = |m: Ftrl| {
        let e = Ftrl::restore(&State::new(ModelState::Ftrl(Box::new(m)))).unwrap_err();
        assert!(e.to_string().contains("scaler"), "{e}");
    };
    let mut a = on.clone();
    a.scaler = None;
    refused(a);
    let mut b = off.clone();
    b.scaler = on.scaler.clone();
    refused(b);
    let mut c = off.clone();
    c.warmup = on.warmup.clone();
    refused(c);
    let mut d = on.clone();
    d.scaler = Some(EwDiag::new(2));
    refused(d);
    assert!(Ftrl::restore(&on.state()).is_ok());
    assert!(Ftrl::restore(&off.state()).is_ok());
}
