//! The scale a summary holds its moments at (PLAN 578j, TEST-10's
//! `ewm_moments`): the bound that keeps the arithmetic as written, the rise
//! past it and the fall back, each on both sides of its threshold; and the
//! update and the merge at a scale, the update itself to the bit.

use super::*;

/// One leg, the moments of orders 2 to `p`.
fn one_leg(p: u32) -> Group {
    Group::new(vec![Leg::of(0)], &[vec![p]], vec![2])
}

fn pow2(k: i32) -> f64 {
    libm::scalbn(1.0, k)
}

#[test]
fn the_bound_holds_the_arithmetic_as_written_to_its_edge() {
    // Orders 2 to 5: 2·5 + 2 + c + 5 (d + 1) ≤ 1020.
    let g = one_leg(5);
    let at = |c: f64| vec![1.0, c, 0.5, -c];
    assert!(g.fits(&at(pow2(1003)), 0));
    assert!(!g.fits(&at(pow2(1004)), 0));
    assert!(!g.fits(&at(-pow2(1004)), 0));
    assert!(g.fits(&at(1.0), 200));
    assert!(!g.fits(&at(1.0), 201));
    // A deviation below 2 counts as 2, whatever its exponent below.
    assert_eq!(g.fits(&at(pow2(1003)), -40), g.fits(&at(pow2(1003)), 0));
    // A deviation's exponent grows the bound by the order: at order 2,
    // 2·2 + 2 + 0 + 2 (d + 1) ≤ 1020 to 506, where order 5 stops at 200.
    assert!(one_leg(2).fits(&[1.0], 506));
    assert!(!one_leg(2).fits(&[1.0], 507));
}

#[test]
fn a_rise_takes_the_deviations_below_one_and_the_moments_into_room() {
    let g = one_leg(5);
    // The room at order 5: 1020 − 15 − 2 = 1003.
    let small = [1.0, 1.0, 1.0, 1.0];
    assert_eq!(g.rise(&[&small], 333), 334);
    assert_eq!(g.rise(&[&small], 0), 1);
    assert_eq!(g.rise(&[&small], -7), 1);
    // c_2 at 2^1010 is 7 over: ⌈7/2⌉ = 4; c_5 at 2^1010, ⌈7/5⌉ = 2.
    assert_eq!(g.rise(&[&[pow2(1010), 1.0, 1.0, 1.0]], 0), 4);
    assert_eq!(g.rise(&[&[1.0, 1.0, 1.0, pow2(1010)]], 0), 2);
    assert_eq!(g.rise(&[&[1.0, 1.0, 1.0, pow2(1008)]], 0), 1);
    assert_eq!(g.rise(&[&[1.0, 1.0, 1.0, pow2(1003)]], 0), 1);
    // Each summary of a merge counts, the larger rise winning.
    let other = [1.0, 1.0, pow2(1015), 1.0];
    assert_eq!(g.rise(&[&small, &other], 0), 3);
    assert_eq!(g.rise(&[&other, &small], 0), 3);
    // Past the orders the bound leaves room for, the moments go below 2.
    let wide = one_leg(400);
    let mut c = vec![1.0; wide.keep.len()];
    c[0] = pow2(9);
    assert_eq!(wide.rise(&[&c], 0), 5);
}

#[test]
fn the_scale_falls_back_as_far_as_the_moments_allow() {
    let g = one_leg(5);
    // The room for deviations below 16 at order 5: 1020 − 30 − 2 = 988.
    assert_eq!(g.fall(&[1.0, 1.0, 1.0, 1.0], 100), 0);
    assert_eq!(g.fall(&[0.0; 4], 100), 0);
    // c_5 at 2^900: e' with 900 + (100 − e') 5 ≤ 988, the lowest 83.
    assert_eq!(g.fall(&[1.0, 1.0, 1.0, pow2(900)], 100), 83);
    assert_eq!(g.fall(&[1.0, 1.0, 1.0, -pow2(900)], 100), 83);
    // At its edge, 900 + 17·5 = 985 and 900 + 18·5 = 990: one more is over.
    assert_eq!(g.fall(&[1.0, 1.0, 1.0, pow2(903)], 100), 83);
    assert_eq!(g.fall(&[1.0, 1.0, 1.0, pow2(904)], 100), 84);
    // Never above the scale it falls from.
    assert_eq!(g.fall(&[1.0, 1.0, 1.0, pow2(1000)], 10), 10);
}

#[test]
fn a_scale_holds_the_orders_to_its_limit() {
    let at = one_leg(MAX_HELD_ORDER);
    assert_eq!(at.held(), at.keep.len());
    assert_eq!(at.top_order(), MAX_HELD_ORDER as i32);
    let past = one_leg(MAX_HELD_ORDER + 1);
    assert_eq!(past.held(), past.keep.len() - 1);
    assert_eq!(past.top_order(), MAX_HELD_ORDER as i32);
    // A moment past the limit is not asked to be finite.
    let mut c = vec![1.0; past.keep.len()];
    *c.last_mut().unwrap() = f64::NAN;
    assert!(past.finite(&c));
    c[0] = f64::INFINITY;
    assert!(!past.finite(&c));
}

#[test]
fn exponents_are_read_from_the_bits() {
    assert_eq!(exponent(1.0), 0);
    assert_eq!(exponent(-3.0), 1);
    assert_eq!(exponent(0.75), -1);
    assert_eq!(exponent(f64::MAX), 1023);
    assert_eq!(top_exponent(&[0.5, -1.5, 0.0]), 0);
    assert_eq!(top_exponent(&[0.5, -1e100, 4.0]), 332);
    assert_eq!(top_exponent(&[]), 0);
}

/// The update and the merge are homogeneous: at a scale of `2^k` they are
/// the update and the merge of the moments as they are, to the bit, where
/// nothing underflows.
#[test]
fn the_update_at_a_scale_is_the_update_to_the_bit() {
    let g = Group::new(
        vec![Leg::of(0), Leg::of(1)],
        &[vec![4, 0], vec![2, 2], vec![1, 3]],
        vec![2],
    );
    let mut s = 3u64;
    let mut u = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    };
    let c: Vec<f64> = g.keep.iter().map(|_| 3.0 * u()).collect();
    let other: Vec<f64> = g.keep.iter().map(|_| 2.0 * u()).collect();
    let d = [7.0 * u(), -5.0 * u()];
    // Order 4 at 2^200 is 2^-800 of the moment: no underflow.
    for k in [1, 40, 200] {
        let mut plain = c.clone();
        row_moments(&mut plain, &g, 0.7, 0.3, &d);
        let mut at = c.clone();
        rescale(&mut at, &g, k);
        row_moments(&mut at, &g, 0.7, 0.3, &scaled(&d, k));
        rescale(&mut at, &g, -k);
        assert_eq!(
            plain.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            at.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "the update at 2^{k}"
        );
        let plain = merged_moments(&g, &c, &other, 0.6, 0.4, &d);
        let (mut a, mut b) = (c.clone(), other.clone());
        rescale(&mut a, &g, k);
        rescale(&mut b, &g, k);
        let mut at = merged_moments(&g, &a, &b, 0.6, 0.4, &scaled(&d, k));
        rescale(&mut at, &g, -k);
        assert_eq!(
            plain.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            at.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "the merge at 2^{k}"
        );
    }
}

/// A row past the bound raises the scale; the moments it holds are the
/// moments, read in units of the scale; and once they fit again the scale
/// falls to 0, the summary then the one the arithmetic as written gives.
#[test]
fn a_summary_rises_past_the_range_and_falls_back() {
    let g = one_leg(5);
    let mut s = Summary::empty(&g);
    for x in [0.5, -1.0, 2.0] {
        s.learn(&g, 0.9, 1.0, &[x]);
    }
    let before = s.clone();
    s.learn(&g, 0.9, 1.0, &[1e100]);
    assert!(s.e > 0, "the bound row raises the scale");
    assert!(s.c.iter().all(|v| v.is_finite()));
    // c_2 is of degree 2: read in the deviations' units it fits a double,
    // and is the arithmetic's own (a square of 1e100 fits).
    let mut plain = before.clone();
    let d = [crate::comp::dev(1e100, plain.m[0], plain.m_lo[0])];
    let w_new = 0.9 * plain.w + 1.0;
    let (a, b) = (0.9 * plain.w / w_new, 1.0 / w_new);
    row_moments(&mut plain.c, &g, a, b, &d);
    let c2 = s.unscale(s.c[0], 2);
    assert!(
        (c2 - plain.c[0]).abs() <= 4.0 * f64::EPSILON * plain.c[0].abs(),
        "{c2} vs {}",
        plain.c[0]
    );
    // The arithmetic as written overflows the fourth and fifth moments.
    assert!(!plain.c[2].is_finite() || !plain.c[3].is_finite());
    assert_eq!(s.unscale(s.c[3], 5), f64::INFINITY.copysign(s.c[3]));
    // Ordinary rows: the deviation from a mean near 1e100 keeps the scale up
    // a while, then it falls to 0 and stays there.
    let mut fell = None;
    for i in 0..40_000 {
        s.learn(&g, 0.9, 1.0, &[(i % 7) as f64 - 3.0]);
        if s.e == 0 && fell.is_none() {
            fell = Some(i);
        }
        if fell.is_some() {
            assert_eq!(s.e, 0, "row {i}: the scale stays down");
        }
    }
    assert!(fell.is_some(), "the scale falls back once the moments fit");
}
