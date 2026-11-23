//! The expansion against `crate::oracle::sum` (Shewchuk's partials rounded
//! as `math.fsum` rounds them), which shares no code with `exact.rs`.

use super::*;
use crate::oracle;
use proptest::prelude::*;

/// The canonical form: finite nonzero parts in increasing magnitude, each
/// the rounded sum of itself and every part below it, so the top is the
/// whole sum rounded.
fn assert_canonical(e: &Expansion) {
    let p = e.parts();
    assert!(p.len() <= MAX_PARTS, "{} parts", p.len());
    for (i, &x) in p.iter().enumerate() {
        assert!(x.is_finite() && x != 0.0, "part {i} is {x}");
        assert_eq!(oracle::sum(p[..=i].iter().copied()), x, "part {i} of {p:?}");
        if i > 0 {
            assert!(p[i - 1].abs() < x.abs(), "parts out of order: {p:?}");
        }
    }
}

/// `2^e` exactly, for `e` in `-1074..=1023`, subnormals included.
fn p2(e: i32) -> f64 {
    if e >= -1022 {
        f64::from_bits(((e + 1023) as u64) << 52)
    } else {
        f64::from_bits(1u64 << (e + 1074))
    }
}

/// A double from a sign, an exponent and a mantissa: any finite double in
/// the exponent range given.
fn double(lo: i32, hi: i32) -> impl Strategy<Value = f64> {
    (any::<bool>(), lo..=hi, 0u64..(1 << 52)).prop_map(|(neg, e, m)| {
        let v = f64::from_bits((1023u64 << 52) | m) * p2(e);
        if neg { -v } else { v }
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(400))]

    #[test]
    fn the_sum_is_exact_and_canonical(xs in prop::collection::vec(double(-1074, 330), 0..60)) {
        let mut e = Expansion::new();
        for (i, &x) in xs.iter().enumerate() {
            e.add(x);
            assert_canonical(&e);
            prop_assert_eq!(e.value(), oracle::sum(xs[..=i].iter().copied()));
        }
        prop_assert!(e.parts().len() <= parts_below(334));
    }

    #[test]
    fn a_comparison_is_exact(
        xs in prop::collection::vec(double(-60, 60), 1..20),
        t in double(-60, 60),
        pick in any::<prop::sample::Index>(),
    ) {
        let mut e = Expansion::new();
        for &x in &xs {
            e.add(x);
        }
        // A threshold the sum lands on exactly, half the time.
        let t = if pick.index(2) == 0 { e.value() } else { t };
        let diff = oracle::sum(xs.iter().copied().chain([-t]));
        prop_assert_eq!(e.cmp_f64(t), diff.partial_cmp(&0.0).unwrap());
        prop_assert_eq!(e.signum(), oracle::sum(xs.iter().copied()).partial_cmp(&0.0).unwrap());
    }

    #[test]
    fn floor_div_leaves_the_remainder(
        xs in prop::collection::vec(double(-30, 30), 1..20),
        theta in double(-10, 20),
    ) {
        let theta = theta.abs();
        let mut e = Expansion::new();
        for &x in &xs {
            e.add(x.abs());
        }
        let before = e.clone();
        let m = e.floor_div(theta).unwrap();
        assert_canonical(&e);
        prop_assert_ne!(e.signum(), Ordering::Less);
        prop_assert_eq!(e.cmp_f64(theta), Ordering::Less);
        // `V − m·θ − R = 0`, exactly.
        let mut parts: Vec<f64> = before.parts().to_vec();
        for j in 0..53 {
            if m >> j & 1 == 1 {
                parts.push(-(theta * p2(j)));
            }
        }
        parts.extend(e.parts().iter().map(|p| -p));
        prop_assert_eq!(oracle::sum(parts), 0.0);
    }

    #[test]
    fn an_integer_enters_exactly(vs in prop::collection::vec(any::<i64>(), 1..20), us in prop::collection::vec(any::<u64>(), 0..5)) {
        let mut e = Expansion::new();
        let mut exact: i128 = 0;
        for &v in &vs {
            e.add_int(i128::from(v));
            exact += i128::from(v);
            assert_canonical(&e);
            // `as f64` on an `i128` rounds to nearest, ties to even.
            prop_assert_eq!(e.value(), exact as f64);
        }
        for &u in &us {
            e.add_int(i128::from(u));
            exact += i128::from(u);
            prop_assert_eq!(e.value(), exact as f64);
        }
        // What is left after the rounded value is the exact remainder.
        let mut rest = e.clone();
        rest.add(-e.value());
        prop_assert_eq!(rest.value(), (exact - e.value() as i128) as f64);
    }

    #[test]
    fn a_saved_list_loads_as_itself(xs in prop::collection::vec(double(-1074, 330), 0..30)) {
        let mut e = Expansion::new();
        for &x in &xs {
            e.add(x);
        }
        prop_assert_eq!(Expansion::from_parts(e.parts().to_vec()).unwrap(), e);
    }
}

/// Near the worst case of the bound: one part every 54 binades, from
/// `2^330` down to the smallest double, each a quarter of a unit of the part
/// above, so no rounding merges two.
#[test]
fn the_bound_holds_at_its_worst() {
    let mut e = Expansion::new();
    let mut k = 0;
    while 330 - 54 * k >= -1074 {
        e.add(p2(330 - 54 * k));
        k += 1;
    }
    assert_canonical(&e);
    assert_eq!(e.parts().len(), 27);
    assert!(e.parts().len() <= parts_below(334));
    assert_eq!(parts_below(334), 27);
    assert_eq!(parts_below(1023), MAX_PARTS);
    // The smallest subnormal and the largest double in one sum.
    let mut e = Expansion::new();
    e.add(f64::from_bits(1));
    e.add(f64::MAX / 4.0);
    assert_eq!(e.parts(), &[f64::from_bits(1), f64::MAX / 4.0]);
}

#[test]
fn a_tie_rounds_to_even_as_fsum_does() {
    // 1 + 2^-53 is half way between 1 and its successor: to even, 1.
    let mut e = Expansion::new();
    e.add(1.0);
    e.add(p2(-53));
    assert_eq!(e.value(), 1.0);
    assert_eq!(e.parts(), &[p2(-53), 1.0]);
    // A part below it leaning up takes it past half way.
    e.add(p2(-120));
    assert_eq!(e.value(), 1.0 + f64::EPSILON);
    assert_canonical(&e);
    assert_eq!(e.cmp_f64(1.0 + f64::EPSILON), Ordering::Less);
    assert_eq!(e.cmp_f64(1.0), Ordering::Greater);
}

#[test]
fn zero_and_its_comparisons() {
    let mut e = Expansion::new();
    assert!(e.is_zero());
    assert_eq!(e.cmp_f64(0.0), Ordering::Equal);
    assert_eq!(e.cmp_f64(-1.0), Ordering::Greater);
    e.add(0.1);
    e.add(-0.1);
    assert!(e.is_zero());
    e.add(0.0);
    assert!(e.is_zero());
    e.add(3.0);
    e.negate();
    assert_eq!(e.value(), -3.0);
    assert_eq!(e.signum(), Ordering::Less);
    e.clear();
    assert!(e.is_zero());
}

#[test]
fn floor_div_refuses_a_count_past_2_to_the_53() {
    let mut e = Expansion::new();
    e.add(1.0);
    let before = e.clone();
    assert_eq!(e.floor_div(p2(-60)), Err(FloorDiv::TooMany));
    assert_eq!(e, before);
    // Just below the limit, it is counted exactly.
    let mut e = Expansion::new();
    e.add(p2(52));
    assert_eq!(e.floor_div(1.0), Ok(1 << 52));
    assert!(e.is_zero());
}

#[test]
fn a_damaged_list_is_refused() {
    assert!(Expansion::from_parts(vec![0.0]).is_err());
    assert!(Expansion::from_parts(vec![f64::NAN]).is_err());
    assert!(Expansion::from_parts(vec![1.0, 2.0]).is_err());
    assert!(Expansion::from_parts(vec![1.0; MAX_PARTS + 1]).is_err());
    assert!(Expansion::from_parts(vec![p2(-60), 1.0]).is_ok());
}

#[test]
fn add_checked_stops_short_of_overflow() {
    let mut e = Expansion::new();
    let big = p2(1019);
    assert!(e.add_checked(big));
    assert!(!e.add_checked(big));
    assert_eq!(e.value(), big);
    assert!(e.add_checked(-1.0));
}
