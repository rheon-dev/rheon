//! Terms and sums past the largest double and the window's sums summed
//! again (PLAN 578j, found beside TEST-10's `realized`), each on both sides
//! of its threshold.

use super::*;
use crate::power_variation::PvKernel;

fn p(k: i32) -> f64 {
    libm::scalbn(1.0, k)
}

#[test]
fn a_sum_is_held_at_the_largest_double_and_not_before() {
    let (mut hi, mut lo) = (p(1023), 0.0);
    add(&mut hi, &mut lo, 0.75 * p(1023));
    assert_eq!((hi, lo), (1.75 * p(1023), 0.0));
    add(&mut hi, &mut lo, 0.25 * p(1023));
    assert_eq!((hi, lo), (f64::MAX, 0.0));
    let (mut hi, mut lo) = (-p(1023), 0.0);
    add(&mut hi, &mut lo, -p(1023));
    assert_eq!((hi, lo), (-f64::MAX, 0.0));
    let (mut hi, mut lo) = (1.0, 0.0);
    add(&mut hi, &mut lo, p(-60));
    assert_eq!((hi, lo), (1.0, p(-60)));
}

#[test]
fn a_term_past_the_largest_double_is_held_at_it() {
    assert_eq!(capped(f64::INFINITY), f64::MAX);
    assert_eq!(capped(f64::NEG_INFINITY), -f64::MAX);
    assert_eq!(capped(-3.5), -3.5);
    assert_eq!(capped(f64::MAX), f64::MAX);
}

#[test]
fn a_sum_left_with_less_than_its_rounding_is_lost() {
    assert!(!lost(1.0, f64::EPSILON));
    assert!(lost(1.0, 0.5 * f64::EPSILON));
    assert!(lost(-1.0, -0.5 * f64::EPSILON));
    assert!(!lost(0.0, 0.0));
}

/// A product of powers formed from fractions and powers of two: the
/// plain product where it fits, to the rounding of a power that is not
/// whole times an exponent (`ln 2 · ε · |e p|`, below `1e-13` for any
/// exponent a double has); 0 with a factor of 0; held at the largest
/// double past it.
#[test]
fn a_product_past_the_range_is_formed_from_its_parts() {
    let cases: [(f64, &[(f64, f64)]); 4] = [
        (0.7, &[(3.0, 2.0), (0.25, 1.5)]),
        (
            2.5,
            &[(1e30, 4.0 / 3.0), (1e-20, 4.0 / 3.0), (7.0, 4.0 / 3.0)],
        ),
        (1e-3, &[(1e-150, 2.0)]),
        (1.0, &[(5.0, 0.5)]),
    ];
    for (w, factors) in cases {
        let plain = factors.iter().fold(w, |acc, &(a, q)| acc * libm::pow(a, q));
        let got = product_past_range(w, factors);
        assert!(
            (got / plain - 1.0).abs() < 1e-13,
            "{w} {factors:?}: {got} vs {plain}"
        );
    }
    // 1e-100 × (1e100)⁴ is 1e300, where (1e100)⁴ alone is past a double.
    let got = product_past_range(1e-100, &[(1e100, 4.0)]);
    assert!((got / 1e300 - 1.0).abs() < 1e-13, "{got}");
    assert_eq!(product_past_range(1.0, &[(1e100, 4.0)]), f64::MAX);
    assert_eq!(
        product_past_range(1e100, &[(1e100, 2.0), (1e100, 2.0)]),
        f64::MAX
    );
    // 0 × inf: the term of a factor of 0 is 0.
    assert_eq!(product_past_range(1.0, &[(1e100, 4.0), (0.0, 4.0)]), 0.0);
    assert_eq!(product_past_range(1.0, &[(0.0, 4.0), (1e100, 4.0)]), 0.0);
    // Far below the smallest double, 0.
    assert_eq!(product_past_range(1e-300, &[(1e-100, 30.0)]), 0.0);
}

fn plan() -> Plan {
    Plan::of(&[
        PvTerm {
            kind: PvTermKind::Multipower {
                powers: vec![1.0, 1.0],
                normalize: false,
            },
            scale: 1,
        },
        PvTerm {
            kind: PvTermKind::Kernel {
                kernel: PvKernel::Bartlett,
                bandwidth: 2,
                flat_top: false,
                center: true,
            },
            scale: 1,
        },
    ])
}

/// Every sum summed again from the window's credits is the sum the window
/// kept as its rows went in and out, to its rounding.
#[test]
fn a_sum_summed_again_is_the_windows_sum() {
    let plan = plan();
    let mut s = Series::new(&plan);
    let mut g = 5u64;
    for seq in 0..40 {
        g = g
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let r = (g >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
        s.learn(&plan, r, 1.0 + r * r, Some(seq));
    }
    for seq in 0..5 {
        s.drop_front(seq);
    }
    let kept = s.clone();
    let stride = 2 * (plan.sums - BASE);
    for k in 0..plan.sums {
        s.rebuild(k, stride);
        let (got, want) = (s.sum(k), kept.sum(k));
        assert!(
            (got - want).abs() <= 8.0 * f64::EPSILON * want.abs().max(1e-300),
            "sum {k}: {got} summed again, {want} kept"
        );
    }
}

/// A credit taken back from a sum held at the largest double: one of a
/// unit in its last place or more sums it again, a smaller one leaves it.
#[test]
fn a_held_sum_is_summed_again_where_a_credit_leaving_can_move_it() {
    let plan = plan();
    let mut s = Series::new(&plan);
    for seq in 0..4 {
        s.learn(&plan, 1.0 + seq as f64, 1.0, Some(seq));
    }
    let ulp = f64::MAX * f64::EPSILON;
    let k = BASE;
    s.hi[k] = f64::MAX;
    s.lo[k] = 0.0;
    let mut again = vec![false; plan.sums];
    s.take(k, 0.5 * ulp, 0.0, &mut again);
    assert!(!again[k]);
    assert_eq!(s.hi[k], f64::MAX);
    s.take(k, ulp, 0.0, &mut again);
    assert!(again[k]);
    let mut again = vec![false; plan.sums];
    let before = s.sum(NW);
    s.take(NW, 1.0, 0.0, &mut again);
    assert_eq!(s.sum(NW), before - 1.0);
    assert!(!again[NW]);
    s.take(NW, 3.0, 0.0, &mut again);
    assert!(again[NW], "what is left, 0, is less than the rounding of 3");
}

/// A higher-order normalized term whose `W^e * s` overflows `f64::MAX` at the
/// input bound withholds `NaN` rather than emitting `+inf`.
#[test]
fn normalized_overflow_at_the_input_bound_is_withheld_rather_than_infinite() {
    let terms = vec![
        PvTerm {
            kind: PvTermKind::Multipower {
                powers: vec![4.0 / 3.0; 3],
                normalize: true,
            },
            scale: 1,
        },
        PvTerm {
            kind: PvTermKind::Kernel {
                kernel: PvKernel::Parzen,
                bandwidth: 2,
                flat_top: false,
                center: true,
            },
            scale: 1,
        },
    ];
    let plan = Plan::of(&terms);
    let mut s = Series::new(&plan);
    for (r, w) in [(-2.0, 1.0), (1e100, 1.0), (0.5, 1e100)] {
        s.learn(&plan, r, w, None);
    }
    let mut out = Vec::new();
    s.read(&plan, 3.0, &mut out);
    assert!(
        out.iter().all(|v| !v.is_infinite()),
        "no output may be infinite: {out:?}"
    );
}
