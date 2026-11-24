//! The sums past the largest double and the window's sums summed again
//! (PLAN 578j, TEST-10's `realized`), each on both sides of its threshold.

use super::*;

#[test]
fn a_sum_is_held_at_the_largest_double_and_not_before() {
    let p = |k| libm::scalbn(1.0, k);
    let (mut hi, mut lo) = (p(1022), 0.0);
    add(&mut hi, &mut lo, p(1022));
    assert_eq!((hi, lo), (p(1023), 0.0));
    add(&mut hi, &mut lo, 0.75 * p(1023));
    assert_eq!((hi, lo), (1.75 * p(1023), 0.0), "below it, the sum");
    add(&mut hi, &mut lo, 0.25 * p(1023));
    assert_eq!((hi, lo), (f64::MAX, 0.0), "2^1024 is past it");
    // Held, it stays held, and nothing goes into `lo`.
    add(&mut hi, &mut lo, 1.0);
    assert_eq!((hi, lo), (f64::MAX, 0.0));
    let (mut hi, mut lo) = (-0.75 * f64::MAX, 0.0);
    add(&mut hi, &mut lo, -0.5 * f64::MAX);
    assert_eq!((hi, lo), (-f64::MAX, 0.0));
    // Below it, TwoSum as it was: the rounding of 1 + 2^-60 kept in `lo`.
    let (mut hi, mut lo) = (1.0, 0.0);
    add(&mut hi, &mut lo, libm::scalbn(1.0, -60));
    assert_eq!((hi, lo), (1.0, libm::scalbn(1.0, -60)));
}

#[test]
fn a_sum_left_with_less_than_its_rounding_is_lost() {
    assert!(!lost(1.0, f64::EPSILON));
    assert!(lost(1.0, 0.5 * f64::EPSILON));
    assert!(lost(-1.0, -0.5 * f64::EPSILON));
    assert!(!lost(0.0, 0.0));
    assert!(!lost(1e300, 1e285));
    assert!(lost(1e300, 1e283));
}

/// A series in a window of `n` returns, the window's rows numbered from 0.
fn windowed(returns: &[(f64, f64)]) -> Series {
    let mut s = Series::new(None);
    for (seq, &(r, w)) in returns.iter().enumerate() {
        s.learn(r, w, Some(seq as u64), None);
    }
    s
}

/// Every sum summed again from the window's terms is the sum the window
/// kept as its rows went in and out, to its rounding: a return's own terms
/// from each, a product from the second on, a triple from the third on.
#[test]
fn a_sum_summed_again_is_the_windows_sum() {
    let mut s = 9u64;
    let mut u = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    };
    let rows: Vec<(f64, f64)> = (0..40).map(|_| (u(), 1.0 + u())).collect();
    let mut series = windowed(&rows);
    // Five rows leave, so the front's product and triple have left too.
    for seq in 0..5 {
        series.drop_front(seq);
    }
    let kept = series.clone();
    for k in 0..SUMS {
        series.rebuild(k);
        let (got, want) = (series.sum(k), kept.sum(k));
        assert!(
            (got - want).abs() <= 4.0 * f64::EPSILON * want.abs(),
            "sum {k}: {got} summed again, {want} kept"
        );
    }
    // Each sum by hand, over the 35 rows left.
    let left = &rows[5..];
    let a: Vec<f64> = left.iter().map(|&(r, _)| r.abs()).collect();
    let w: Vec<f64> = left.iter().map(|&(_, w)| w).collect();
    let bv: f64 = (1..35).map(|i| w[i] * (a[i] * a[i - 1])).sum();
    let pw: f64 = w[1..].iter().sum();
    let near = |got: f64, want: f64| (got - want).abs() <= 1e-13 * want.abs();
    assert!(near(series.sum(BV), bv));
    assert!(near(series.sum(PW), pw));
    assert!(near(series.sum(NW), w.iter().sum()));
}

/// A term taken back from a sum held at the largest double: one of a unit
/// in its last place or more sums it again, a smaller one leaves it.
#[test]
fn a_held_sum_is_summed_again_where_a_term_leaving_can_move_it() {
    let ulp = f64::MAX * f64::EPSILON;
    let mut s = windowed(&[(1.0, 1.0), (2.0, 1.0), (3.0, 1.0)]);
    s.hi[RV] = f64::MAX;
    s.lo[RV] = 0.0;
    let mut again = [false; SUMS];
    s.take(RV, 0.5 * ulp, &mut again);
    assert!(!again[RV]);
    assert_eq!(s.hi[RV], f64::MAX, "nothing taken from a held sum");
    s.take(RV, ulp, &mut again);
    assert!(again[RV]);
    // Below the largest double a term is taken, and marked only where it
    // leaves too little.
    let mut again = [false; SUMS];
    let before = s.sum(NW);
    s.take(NW, 1.0, &mut again);
    assert_eq!(s.sum(NW), before - 1.0);
    assert!(!again[NW]);
    s.take(NW, 2.0, &mut again);
    assert!(again[NW], "what is left, 0, is less than the rounding of 2");
}

/// A quarticity term past the largest double is held at it; one whose
/// weight brings it back is kept, the weight going in first.
#[test]
fn a_quarticity_past_the_largest_double_is_held_at_it() {
    let mut s = Series::new(None);
    for r in [1e100, 1e100, 1e100] {
        s.learn(r, 1.0, None, None);
    }
    assert_eq!(s.sum(TQ), f64::MAX);
    assert!(s.sum(TV).is_finite() && s.sum(TV) > 0.0);
    let mut s = Series::new(None);
    for r in [1e100, 1e100, 1e100] {
        s.learn(r, 1e-100, None, None);
    }
    // t = 1e100: w t⁴ = 1e300, where t⁴ alone is past a double.
    let tq = s.sum(TQ);
    assert!((tq / 1e300 - 1.0).abs() < 1e-13, "{tq}");
}

/// Lee and Mykland's window sums its products again where one leaving
/// leaves less than its rounding, or leaves a held sum.
#[test]
fn lee_and_myklands_window_is_summed_again_where_it_must_be() {
    let mut ring = LmRing::new();
    for (wp, w) in [(1e300, 1e100), (1e-200, 1.0), (2e-200, 1.0)] {
        ring.push(wp, w, 5);
    }
    ring.push(3e-200, 1.0, 5);
    // The product of 1e300 has left: what is left is the three small ones.
    assert_eq!(ring.products.len(), 3);
    assert!(
        (ring.variance() - 2e-200).abs() < 1e-214,
        "{}",
        ring.variance()
    );
    assert_eq!(ring.weight(), 3.0);
    let mut ring = LmRing::new();
    ring.p_hi = f64::MAX;
    ring.products.extend([(1.0, 1.0), (2.0, 1.0), (3.0, 1.0)]);
    ring.w_hi = 3.0;
    ring.push(4.0, 1.0, 5);
    assert_eq!(ring.p_hi + ring.p_lo, 9.0, "a held sum is summed again");
    // Ordinary products leave as they always did.
    let mut ring = LmRing::new();
    for wp in [1.0, 2.0, 3.0, 4.0, 5.0] {
        ring.push(wp, 1.0, 5);
    }
    assert_eq!((ring.p_hi, ring.p_lo), (12.0, 0.0));
}
