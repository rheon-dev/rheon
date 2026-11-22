//! `bocpd`'s caches of what the predictive takes of a run's length and of
//! its chances of a break: each held to what it computes, keyed on every
//! bit, and no part of the state. Moved out of `bocpd.rs`'s tests as they
//! were (task 538, part 2), keeping that file under the repository's size
//! cap for a source file.

use super::tests::{Normals, cfg, from_the_data, step_bits};
use super::*;
use crate::OnlineModel;

/// A stream of 200 runs, which a direct-mapped table of 256 entries
/// answered 30% of the time: every length but the first row's is held,
/// whatever the keys' bits.
#[test]
fn the_length_terms_miss_each_length_once() {
    let mut t = LengthTerms::default();
    let nus: Vec<f64> = (0..200).map(|r| 3.0 + r as f64).collect();
    for row in 0..50 {
        t.begin(nus.len(), 1.0, false);
        for &nu in &nus {
            assert_eq!(t.gamma(nu, 1.0).to_bits(), gamma_diff(nu, 1.0).to_bits());
        }
        assert_eq!(t.misses, 200, "row {row}");
        assert_eq!(t.hits, 200 * row, "row {row}");
    }
    // Halves too, as a `gaussian` run's `ν − d + 1` is, and fractions.
    let mut t = LengthTerms::default();
    let nus: Vec<f64> = (0..300).map(|r| 0.5 + r as f64 * 0.25).collect();
    for _ in 0..3 {
        t.begin(nus.len(), 2.0, true);
        for &nu in &nus {
            assert_eq!(
                t.gamma_and_ln_pi(nu, 2.0).0.to_bits(),
                gamma_diff(nu, 2.0).to_bits()
            );
        }
    }
    assert_eq!((t.misses, t.hits), (300, 600));
}

/// An entry is keyed on every bit of `ν`: `0.0` and `-0.0` are two keys,
/// another `a`, or the other form, empties the table, a NaN is computed
/// and never stored, two keys the hash puts at one place each read
/// their own value, and no number of distinct keys grows the table past
/// its room or leaves it without an empty entry.
#[test]
fn the_length_terms_never_share_an_entry_between_two_keys() {
    let mut c = LengthTerms::default();
    assert_eq!(
        c.gamma(f64::NAN, 1.0).to_bits(),
        gamma_diff(f64::NAN, 1.0).to_bits()
    );
    assert!(
        c.slots.is_empty() && c.hits + c.misses == 0,
        "a NaN is not stored"
    );
    let p: f64 = 3.5;
    c.begin(0, 1.0, false);
    // Every `ν` the hash sends where `p` goes.
    let home = |nu: f64| {
        let b = nu.to_bits();
        ((b ^ (b >> 32)).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 32) as usize & (c.slots.len() - 1)
    };
    let q = (1..)
        .map(|k| f64::from_bits(p.to_bits() + k))
        .find(|q| home(*q) == home(p))
        .unwrap();
    for (nu, hits, misses) in [(p, 0, 1), (p, 1, 1), (q, 1, 2), (p, 2, 2), (q, 3, 2)] {
        assert_eq!(c.gamma(nu, 1.0).to_bits(), gamma_diff(nu, 1.0).to_bits());
        assert_eq!((c.hits, c.misses), (hits, misses), "{nu}");
    }
    c.gamma(0.0, 1.0);
    let misses = c.misses;
    c.gamma(-0.0, 1.0);
    assert_eq!(c.misses, misses + 1, "-0.0 is not 0.0");
    // Another `a` asks for another function: nothing from the first is
    // reused.
    let misses = c.misses;
    c.begin(0, 3.0, false);
    assert_eq!(c.gamma(q, 3.0).to_bits(), gamma_diff(q, 3.0).to_bits());
    assert_eq!(c.misses, misses + 1, "a new a starts empty");
    assert_eq!(c.gamma(q, 3.0).to_bits(), gamma_diff(q, 3.0).to_bits());
    assert_eq!(c.misses, misses + 1, "and then fills");
    // So does the other form, which carries `ln(ν·π)` as well.
    c.begin(0, 3.0, true);
    let (g, l) = c.gamma_and_ln_pi(q, 3.0);
    assert_eq!(c.misses, misses + 2, "the other form starts empty");
    assert_eq!(g.to_bits(), gamma_diff(q, 3.0).to_bits());
    assert_eq!(l.to_bits(), (q * std::f64::consts::PI).ln().to_bits());
    let room = c.slots.len();
    for k in 0..10_000u64 {
        let nu = 1.0 + k as f64 * 0.37;
        assert_eq!(
            c.gamma_and_ln_pi(nu, 3.0).0.to_bits(),
            gamma_diff(nu, 3.0).to_bits()
        );
        assert!(2 * c.filled <= c.slots.len());
    }
    assert_eq!(c.slots.len(), room);
    assert!(c.slots.iter().any(|s| s.key == LengthTerms::EMPTY));
    // A row of more runs makes room, and starts again.
    c.begin(1_000, 3.0, true);
    assert!(c.slots.len() >= 4_000 && c.filled == 0);
    c.begin(10, 3.0, true);
    assert!(c.slots.len() >= 4_000, "the room is not given back");
}

/// The chance of a break is the one `Branches` makes, to the bit, held
/// while the same bits come and made again when they do not; and a hazard
/// that is no hazard is refused and holds nothing.
#[test]
fn the_held_branches_are_the_ones_made_and_follow_the_bits() {
    let bits = |b: Branches| (b.ln_break.to_bits(), b.ln_grow.to_bits());
    let mut m = Bocpd::new(BocpdCfg {
        hazard_from_row: true,
        ..cfg(1)
    })
    .unwrap();
    let mut held = HeldBranches::default();
    for h in [50.0, 50.0, 7.5, 50.0, 7.5, 7.5, -0.0, 3.0] {
        let (_, got) = m.before_row_held(&[Some(h)], 1.0, &mut held);
        let (_, want) = m.before_row(&[Some(h)], 1.0);
        assert_eq!(got.map(bits), want.map(bits), "{h}");
    }
    // A hazard of 1 or below is refused, and holds nothing of its own.
    let before = held.0.map(|(k, b)| (k, bits(b)));
    let (_, got) = m.before_row_held(&[Some(1.0)], 1.0, &mut held);
    assert!(got.is_none());
    assert_eq!(held.0.map(|(k, b)| (k, bits(b))), before);
    // On the clock, by the step.
    m.cfg.hazard_from_row = false;
    m.cfg.hazard_on_clock = true;
    let mut held = HeldBranches::default();
    for dc in [1.0, 1.0, 2.5, 0.0, 1.0, 0.0, 0.0, 2.5] {
        let (a, got) = m.before_row_held(&[], dc, &mut held);
        let (b, want) = m.before_row(&[], dc);
        assert_eq!(got.map(bits), want.map(bits), "{dc}");
        let eq = |a: Option<Vec<f64>>, b: Option<Vec<f64>>| {
            a.map(|v| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>())
                == b.map(|v| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>())
        };
        assert!(eq(a, b), "{dc}");
    }
    for bad in [-1.0, f64::NAN, f64::INFINITY] {
        assert!(m.before_row_held(&[], bad, &mut held).1.is_none());
    }
}

/// The cache is not state: a state written with a warm cache is the
/// bytes of one written by a model that never had one, a restored model
/// starts empty and equal to the one it came from, and it goes on as
/// that model, to the bit, rebuilding its cache as it goes.
#[test]
fn a_restored_model_starts_with_no_cache_and_goes_on_to_the_bit() {
    let mut m = Bocpd::new(BocpdCfg {
        max_run: 40,
        prune_below: 1e-5,
        ..from_the_data(2, BocpdEmission::Diag)
    })
    .unwrap();
    let mut n = Normals::new(31);
    for _ in 0..600 {
        m.step(&[n.normal(), n.normal()], &[], 1.0, 1.0);
    }
    assert!(m.gammas.hits > 0 && !m.gammas.slots.is_empty());
    let mut cold = m.clone();
    cold.gammas = LengthTerms::default();
    assert_eq!(
        rmp_serde::to_vec(&m).unwrap(),
        rmp_serde::to_vec(&cold).unwrap()
    );
    assert_eq!(
        rmp_serde::to_vec_named(&m).unwrap(),
        rmp_serde::to_vec_named(&cold).unwrap()
    );
    // A state read back from its file, as the bank loads one; and the
    // model's own bytes.
    let saved = rmp_serde::to_vec(&m.state()).unwrap();
    let state: crate::State = rmp_serde::from_slice(&saved).unwrap();
    let mut back = Bocpd::restore(&state).unwrap();
    let bytes = rmp_serde::to_vec(&m).unwrap();
    let mut loaded: Bocpd = rmp_serde::from_slice(&bytes).unwrap();
    for r in [&back, &loaded] {
        assert!(r.gammas.slots.is_empty() && r.gammas.hits + r.gammas.misses == 0);
        assert_eq!(*r, m, "equality ignores the cache");
    }
    for t in 0..400 {
        let x = [n.normal() + if t > 200 { 3.0 } else { 0.0 }, n.normal()];
        let want = step_bits(&m.step(&x, &[], 1.0, 1.0));
        assert_eq!(step_bits(&back.step(&x, &[], 1.0, 1.0)), want, "row {t}");
        assert_eq!(step_bits(&loaded.step(&x, &[], 1.0, 1.0)), want, "row {t}");
    }
    assert_eq!(
        rmp_serde::to_vec(&back).unwrap(),
        rmp_serde::to_vec(&m).unwrap()
    );
    assert!(back.gammas.misses > 0 && back.gammas.hits > back.gammas.misses);
}
