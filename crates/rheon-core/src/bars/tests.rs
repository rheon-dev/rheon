//! The crossing rule against its definitions, written out over integers
//! (exact) and, for general doubles, against `crate::oracle::sum`.

use super::*;
use crate::oracle;
use proptest::prelude::*;

/// The definition of the remainder carried: row `t` closes a bar when
/// `⌊S_t/θ⌋ > ⌊S_{t−1}/θ⌋`, crossing the difference.
fn carried(q: &[i64], theta: i64) -> Vec<(usize, u64)> {
    let mut s: i128 = 0;
    let mut out = Vec::new();
    for (t, &x) in q.iter().enumerate() {
        let before = s.div_euclid(i128::from(theta));
        s += i128::from(x);
        let after = s.div_euclid(i128::from(theta));
        if after > before {
            out.push((t, (after - before) as u64));
        }
    }
    out
}

/// The definition of the reset: `|R| >= θ` closes, crossing `⌊|R|/θ⌋`,
/// and the next bar starts from 0.
fn reset(q: &[i64], theta: i64) -> Vec<(usize, u64)> {
    let mut r: i128 = 0;
    let mut out = Vec::new();
    for (t, &x) in q.iter().enumerate() {
        r += i128::from(x);
        if r.abs() >= i128::from(theta) {
            out.push((t, (r.abs() / i128::from(theta)) as u64));
            r = 0;
        }
    }
    out
}

fn run(kind: BarKind, q: &[f64], theta: f64) -> Vec<(usize, u64)> {
    let mut c = Crossing::new(kind);
    q.iter()
        .enumerate()
        .filter_map(|(t, &x)| c.step(&[x], theta).unwrap().map(|m| (t, m)))
        .collect()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(300))]

    #[test]
    fn sum_is_the_multiples_of_the_threshold(q in prop::collection::vec(0i64..5000, 0..300), theta in 1i64..3000) {
        let qf: Vec<f64> = q.iter().map(|&x| x as f64).collect();
        prop_assert_eq!(run(BarKind::Sum, &qf, theta as f64), carried(&q, theta));
        // On a grid of 1/1024, scaled: the same bars.
        let qs: Vec<f64> = q.iter().map(|&x| x as f64 / 1024.0).collect();
        prop_assert_eq!(run(BarKind::Sum, &qs, theta as f64 / 1024.0), carried(&q, theta));
    }

    #[test]
    fn imbalance_resets_at_each_bar(q in prop::collection::vec(-5000i64..5000, 0..300), theta in 1i64..3000) {
        let qf: Vec<f64> = q.iter().map(|&x| x as f64).collect();
        prop_assert_eq!(run(BarKind::Imbalance, &qf, theta as f64), reset(&q, theta));
    }

    #[test]
    fn count_closes_every_theta_rows(n in 0usize..500, theta in 1u32..40) {
        let q = vec![0.0; n];
        let want: Vec<(usize, u64)> = (0..n).filter(|t| (t + 1) % theta as usize == 0).map(|t| (t, 1)).collect();
        prop_assert_eq!(run(BarKind::Count, &q, f64::from(theta)), want);
    }

    /// General doubles, against the exact sign of `S_t − j·θ`, with `θ` a
    /// power of two so that its multiples are exact doubles: `k`, the
    /// multiples crossed through row `t`, is the largest `j` with that sign
    /// not negative, found from the rounded total and stepped exactly.
    #[test]
    fn sum_of_general_doubles_is_exact(
        q in prop::collection::vec(0.0f64..1e6, 0..200),
        j in -20i32..20,
    ) {
        let theta = 2f64.powi(j);
        let got = run(BarKind::Sum, &q, theta);
        let past = |t: usize, j: u64| oracle::sum(q[..=t].iter().copied().chain([-(j as f64 * theta)])) >= 0.0;
        let mut k: u64 = 0;
        let mut want = Vec::new();
        for t in 0..q.len() {
            let mut n = (oracle::sum(q[..=t].iter().copied()) / theta).floor() as u64;
            while n > 0 && !past(t, n) {
                n -= 1;
            }
            while past(t, n + 1) {
                n += 1;
            }
            if n > k {
                want.push((t, n - k));
                k = n;
            }
        }
        prop_assert_eq!(got, want);
    }

    /// Integers past 2^53 enter as two doubles, exactly.
    #[test]
    fn wide_integers_are_exact(q in prop::collection::vec(0i64..i64::MAX / 512, 1..100), theta in (1i64 << 20)..(1i64 << 50)) {
        let mut c = Crossing::new(BarKind::Sum);
        let got: Vec<(usize, u64)> = q
            .iter()
            .enumerate()
            .filter_map(|(t, &x)| {
                let hi = x as f64;
                let lo = (i128::from(x) - hi as i128) as f64;
                c.step(&[hi, lo], theta as f64).unwrap().map(|m| (t, m))
            })
            .collect();
        prop_assert_eq!(got, carried(&q, theta));
    }
}

#[test]
fn the_threshold_is_reached_with_ge() {
    // 4 + 6 = 10 closes a bar of 10; 3 + 6 = 9 does not.
    assert_eq!(run(BarKind::Sum, &[4.0, 6.0, 3.0, 6.0], 10.0), vec![(1, 1)]);
    assert_eq!(run(BarKind::Imbalance, &[-4.0, -6.0], 10.0), vec![(1, 1)]);
    assert_eq!(run(BarKind::Imbalance, &[-4.0, -5.0], 10.0), vec![]);
    // A threshold just above the total: no bar.
    assert_eq!(run(BarKind::Sum, &[4.0, 6.0], 10.000000000000002), vec![]);
    // 0.1 + 0.2 is 0.30000000000000004 exactly as doubles sum: a double
    // total of 0.1 + 0.2 reaches 0.30000000000000004, and the exact one
    // is 0.3000000000000000166..., just below it.
    assert_eq!(run(BarKind::Sum, &[0.1, 0.2], 0.30000000000000004), vec![]);
    assert_eq!(run(BarKind::Sum, &[0.1, 0.2], 0.3), vec![(1, 1)]);
}

#[test]
fn a_row_past_the_threshold_stays_whole_and_counts_its_thresholds() {
    let mut c = Crossing::new(BarKind::Sum);
    assert_eq!(c.step(&[7.0], 10.0), Ok(None));
    // 7 + 35 = 42: four thresholds, 2 carried.
    assert_eq!(c.step(&[35.0], 10.0), Ok(Some(4)));
    assert_eq!(c.carry(), 2.0);
    // Imbalance: 35 alone crosses 3, and the next bar starts from 0.
    let mut c = Crossing::new(BarKind::Imbalance);
    assert_eq!(c.step(&[-35.0], 10.0), Ok(Some(3)));
    assert_eq!(c.carry(), 0.0);
}

#[test]
fn a_negative_quantity_is_refused_under_sum_and_nothing_changes() {
    let mut c = Crossing::new(BarKind::Sum);
    c.step(&[3.0], 10.0).unwrap();
    let before = c.clone();
    assert_eq!(c.step(&[-1.0], 10.0), Err(Refused::Negative));
    assert_eq!(c, before);
    // A missing quantity adds 0, and -0.0 is 0.
    assert_eq!(c.step(&[], 10.0), Ok(None));
    assert_eq!(c.step(&[-0.0], 10.0), Ok(None));
    assert_eq!(c, before);
}

#[test]
fn too_many_thresholds_are_refused_and_nothing_changes() {
    for kind in [BarKind::Sum, BarKind::Imbalance] {
        let mut c = Crossing::new(kind);
        c.step(&[0.25], 1.0).unwrap();
        let before = c.clone();
        assert_eq!(c.step(&[1e300], 1e-300), Err(Refused::TooMany));
        assert_eq!(c, before);
        assert_eq!(
            c.step(&[-1e300], 1e-300),
            Err(if kind == BarKind::Sum {
                Refused::Negative
            } else {
                Refused::TooMany
            })
        );
        assert_eq!(c, before);
    }
}

#[test]
fn a_threshold_column_is_read_at_each_row() {
    // Each row's crossing reads its own threshold; a close subtracts
    // multiples of it.
    let mut c = Crossing::new(BarKind::Sum);
    assert_eq!(c.step(&[6.0], 10.0), Ok(None));
    assert_eq!(c.step(&[0.0], 5.0), Ok(Some(1)));
    assert_eq!(c.carry(), 1.0);
    let mut c = Crossing::new(BarKind::Count);
    assert_eq!(c.step(&[], 3.0), Ok(None));
    assert_eq!(c.step(&[], 3.0), Ok(None));
    assert_eq!(c.step(&[], 1.0), Ok(Some(3)));
}

#[test]
fn reset_and_from_parts() {
    let mut c = Crossing::new(BarKind::Sum);
    c.step(&[3.0], 10.0).unwrap();
    c.reset();
    assert_eq!(c, Crossing::new(BarKind::Sum));
    let mut neg = Expansion::new();
    neg.add(-1.0);
    assert!(Crossing::from_parts(BarKind::Sum, Running::Exact(neg.clone())).is_err());
    assert!(Crossing::from_parts(BarKind::Imbalance, Running::Exact(neg)).is_ok());
    assert!(Crossing::from_parts(BarKind::Count, Running::Exact(Expansion::new())).is_err());
    assert!(Crossing::from_parts(BarKind::Sum, Running::Count(0)).is_err());
}
