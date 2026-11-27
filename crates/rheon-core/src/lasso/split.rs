//! A sweep of the `lasso` model's coordinate descent with no dependent
//! chain of subtractions, run from [`SPLIT_MIN_K`] features (task 479).
//!
//! [`super::cd_sweep`] takes `rho_i`'s terms off `d_i` in the order
//! `j = 0, 1, ..., i-1, i+1, ..., k-1`: the coefficients this sweep has set,
//! then the sweep's starting ones. The last run, `k - i - 1` terms, is one
//! chain of subtractions that waits for `b_{i-1}`, and the sweep is bound
//! by that chain's latency (73% of a row at `k = 500`, task 479's profile).
//!
//! Here the two runs swap: the starting coefficients' terms go first, all
//! of them before the sweep begins, and the coefficients the sweep sets go
//! after, each one's column as soon as it is set:
//!
//! ```text
//! a_m = d_m − Σ_{j > m} c_mj b_j          (j ascending, b the sweep's start)
//! for i = 0, 1, ..., k−1:
//!     b_i = soft(a_i, l1) / (c_ii + l2)
//!     a_m −= c_mi b_i   for every m > i
//! ```
//!
//! `rho_i = a_i` holds the same terms as the chained sweep, in another
//! order, so the iteration is the same cyclic descent (Gauss–Seidel) and
//! only the rounding of each `rho_i` differs. Every step is an update of a
//! vector by a column, whose elements are independent, so it runs as
//! vectors with nothing between `b_{i-1}` and `b_i` but `a_i`'s own last
//! subtraction. Columns are taken four at a time ([`take_off`]), the terms
//! of each element in the same ascending order, so a partial sum stays in
//! a register across four terms. A zero coefficient's column subtracts
//! exactly 0 and is left out where every `c` is finite (`finite`); with a
//! NaN or an infinity in `c` it is taken off, as the chained sweep's
//! `c * 0` is. Whether `c` is finite is read once a correlation matrix,
//! and only when a sweep first meets a zero coefficient: a fit with none
//! never pays the pass over its `k²` entries ([`all_finite`]).

/// The fewest features the split sweep runs at, on every platform and for
/// any number of penalties. Measured (task 479, ABBA in both function
/// layouts, ns a row split / chained): on arm64 (Apple M4 Pro) 1.07 to
/// 1.15 at 8 features and 1.00 to 1.03 at 20, then against the chained
/// sweep (the sparse one from 64) 0.85 to 0.91 at 32, 0.78 to 0.86 at 40,
/// 0.61 to 0.81 from 48 to 128, 0.41 to 0.63 at 500, one penalty or
/// three. On x86-64 (AMD EPYC 9V74) 1.00 to 1.03 at 32, slower with three
/// penalties in both layouts, so the floor is 40 on every platform: 0.95 to
/// 0.98 at 40, 0.76 to 0.96 from 48 to 128, 0.54 to 0.67 at 500. Below it
/// the chained sweep runs, and its bits are the ones before the split
/// sweep.
pub(super) const SPLIT_MIN_K: usize = 40;

/// One sweep of the descent with the two runs swapped (the module's
/// account); the largest move. `acc` is the partial sums' buffer; `finite`
/// says whether every entry of `c` is finite, `None` until a sweep over
/// this `c` first needed to know, when it is read ([`all_finite`]) and kept.
#[allow(clippy::too_many_arguments)]
pub(super) fn cd_sweep_split<const POSITIVE: bool>(
    c: &[f64],
    k: usize,
    d: &[f64],
    s: &[f64],
    l1: f64,
    l2: f64,
    b: &mut [f64],
    acc: &mut Vec<f64>,
    finite: &mut Option<bool>,
) -> f64 {
    debug_assert_eq!(c.len(), k * k);
    acc.clear();
    acc.extend_from_slice(&d[..k]);
    let mut skip = |v: f64| v == 0.0 && *finite.get_or_insert_with(|| all_finite(c));
    // The starting coefficients' terms: `a_m −= c_mj b_j` for `j > m`, `j`
    // ascending, read from row `j`'s first `j` entries (`c` is symmetric).
    let mut j0 = 0;
    while j0 < k {
        let j1 = (j0 + 4).min(k);
        let mut rows: [&[f64]; 4] = [&[]; 4];
        let mut bs = [0.0; 4];
        let mut n = 0;
        // The prefix every coordinate of the block reaches: `a_m`, `m < j0`.
        for j in j0..j1 {
            if !skip(b[j]) {
                rows[n] = &c[j * k..j * k + j0];
                bs[n] = b[j];
                n += 1;
            }
        }
        take_off(&mut acc[..j0], &rows[..n], &bs[..n]);
        // Inside the block: `a_m`, `j0 ≤ m < j1`, takes the block's `j > m`.
        for m in j0..j1 {
            let mut a = acc[m];
            for j in m + 1..j1 {
                if !skip(b[j]) {
                    a -= c[j * k + m] * b[j];
                }
            }
            acc[m] = a;
        }
        j0 = j1;
    }
    // The sweep: each coordinate from its partial sum, then its column off
    // the coordinates after it, four coordinates' columns at a time past
    // the block.
    let mut max_delta: f64 = 0.0;
    let mut i0 = 0;
    while i0 < k {
        let i1 = (i0 + 4).min(k);
        for i in i0..i1 {
            let newb = if s[i] <= 0.0 {
                0.0
            } else {
                super::soft_over::<POSITIVE>(acc[i], l1, c[i * k + i] + l2)
            };
            if s[i] > 0.0 {
                max_delta = max_delta.max((newb - b[i]).abs());
            }
            b[i] = newb;
            if !skip(newb) {
                for m in i + 1..i1 {
                    acc[m] -= c[i * k + m] * newb;
                }
            }
        }
        let mut rows: [&[f64]; 4] = [&[]; 4];
        let mut bs = [0.0; 4];
        let mut n = 0;
        for i in i0..i1 {
            if !skip(b[i]) {
                rows[n] = &c[i * k + i1..(i + 1) * k];
                bs[n] = b[i];
                n += 1;
            }
        }
        take_off(&mut acc[i1..], &rows[..n], &bs[..n]);
        i0 = i1;
    }
    max_delta
}

/// Whether every entry of `c` is finite: so a term `c_ij * 0` is exactly 0
/// and leaving it out changes no bit of a sum. Every entry is read, no
/// early exit, so the pass runs as vectors (the early-exit `all` ran one
/// entry at a time).
pub(super) fn all_finite(c: &[f64]) -> bool {
    c.iter().fold(true, |ok, v| ok & v.is_finite())
}

/// `a_m −= rows[0][m] bs[0]`, then `rows[1][m] bs[1]`, ..., in that order
/// for each `m`: up to four columns, each element's partial sum held in a
/// register across them.
#[inline(always)]
fn take_off(a: &mut [f64], rows: &[&[f64]], bs: &[f64]) {
    match (rows, bs) {
        ([], _) => {}
        ([r0], [b0]) => {
            for (a, x0) in a.iter_mut().zip(*r0) {
                *a -= x0 * b0;
            }
        }
        ([r0, r1], [b0, b1]) => {
            for ((a, x0), x1) in a.iter_mut().zip(*r0).zip(*r1) {
                *a = *a - x0 * b0 - x1 * b1;
            }
        }
        ([r0, r1, r2], [b0, b1, b2]) => {
            for (((a, x0), x1), x2) in a.iter_mut().zip(*r0).zip(*r1).zip(*r2) {
                *a = *a - x0 * b0 - x1 * b1 - x2 * b2;
            }
        }
        ([r0, r1, r2, r3], [b0, b1, b2, b3]) => {
            for ((((a, x0), x1), x2), x3) in a.iter_mut().zip(*r0).zip(*r1).zip(*r2).zip(*r3) {
                *a = *a - x0 * b0 - x1 * b1 - x2 * b2 - x3 * b3;
            }
        }
        _ => unreachable!("at most four columns, one coefficient each"),
    }
}

/// The split sweep one term at a time, in the module's order, every zero
/// column taken off: the definition the blocked one must equal to the bit
/// (and `descend`'s tests in `lasso.rs` hold it to).
#[cfg(test)]
pub(super) fn cd_sweep_split_plain(
    c: &[f64],
    k: usize,
    d: &[f64],
    s: &[f64],
    (l1, l2): (f64, f64),
    b: &mut [f64],
) -> f64 {
    let mut a = d.to_vec();
    for j in 0..k {
        for m in 0..j {
            a[m] -= c[j * k + m] * b[j];
        }
    }
    let mut delta: f64 = 0.0;
    for i in 0..k {
        let newb = if s[i] <= 0.0 {
            0.0
        } else {
            super::soft_over::<false>(a[i], l1, c[i * k + i] + l2)
        };
        if s[i] > 0.0 {
            delta = delta.max((newb - b[i]).abs());
        }
        b[i] = newb;
        for m in i + 1..k {
            a[m] -= c[i * k + m] * newb;
        }
    }
    delta
}

#[cfg(test)]
mod tests {
    use super::cd_sweep_split;
    use super::cd_sweep_split_plain as plain;

    fn lcg(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (*state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    }

    /// A symmetric `c` with unit diagonal, `d`, and a start with zeros in it.
    fn problem(k: usize, seed: u64) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut st = seed;
        let mut c = vec![0.0; k * k];
        for i in 0..k {
            c[i * k + i] = 1.0;
            for j in 0..i {
                let v = 0.6 * lcg(&mut st);
                c[i * k + j] = v;
                c[j * k + i] = v;
            }
        }
        let d: Vec<f64> = (0..k).map(|_| lcg(&mut st)).collect();
        let s: Vec<f64> = (0..k).map(|i| if i % 7 == 3 { 0.0 } else { 1.0 }).collect();
        let b: Vec<f64> = (0..k)
            .map(|i| if i % 3 == 0 { 0.0 } else { lcg(&mut st) })
            .collect();
        (c, d, s, b)
    }

    /// Blocked by four, zero columns left out: the same bits as the plain
    /// definition, at every width around the block's edges and with every
    /// kind of penalty, sweep after sweep; with `c` known finite, and with
    /// it not yet read, when the first zero coefficient reads it.
    #[test]
    fn the_blocked_sweep_is_the_plain_one_to_the_bit() {
        for k in [0, 1, 2, 3, 4, 5, 7, 8, 9, 13, 64, 67] {
            for (l1, l2) in [(0.0, 0.0), (0.05, 0.0), (0.3, 0.1)] {
                for known in [Some(true), None] {
                    let (c, d, s, b0) = problem(k, 11 + k as u64);
                    let (mut b, mut want) = (b0.clone(), b0);
                    let mut acc = Vec::new();
                    let mut finite = known;
                    for sweep in 0..4 {
                        let got = cd_sweep_split::<false>(
                            &c,
                            k,
                            &d,
                            &s,
                            l1,
                            l2,
                            &mut b,
                            &mut acc,
                            &mut finite,
                        );
                        let exp = plain(&c, k, &d, &s, (l1, l2), &mut want);
                        assert_eq!(got.to_bits(), exp.to_bits(), "k={k} sweep {sweep}");
                        for (x, y) in b.iter().zip(&want) {
                            assert_eq!(x.to_bits(), y.to_bits(), "k={k} sweep {sweep}");
                        }
                    }
                    // Every start but k = 0's has a zero coefficient.
                    let read = if k == 0 { known } else { Some(true) };
                    assert_eq!(finite, read, "k={k} {known:?}");
                }
            }
        }
    }

    /// Where no coefficient is zero, before or after the sweep, `c` is not
    /// read for finiteness at all (`finite` stays unknown), and the sweep
    /// is the plain definition's bits; the first zero reads it, once.
    #[test]
    fn finiteness_is_read_only_when_a_zero_coefficient_needs_it() {
        let k = 40;
        let (c, d, _, _) = problem(k, 7);
        let s = vec![1.0; k];
        let b0: Vec<f64> = (0..k).map(|i| 0.5 + i as f64).collect();
        let (mut b, mut want) = (b0.clone(), b0);
        let (mut acc, mut finite) = (Vec::new(), None);
        // No penalty: no coefficient thresholds to zero.
        let got = cd_sweep_split::<false>(&c, k, &d, &s, 0.0, 0.0, &mut b, &mut acc, &mut finite);
        let exp = plain(&c, k, &d, &s, (0.0, 0.0), &mut want);
        assert_eq!(got.to_bits(), exp.to_bits());
        assert!(b.iter().all(|v| *v != 0.0), "the fit is dense");
        assert_eq!(finite, None, "read with no zero coefficient");
        // A penalty past every |rho|: the first coefficient set is zero.
        cd_sweep_split::<false>(&c, k, &d, &s, 1e9, 0.0, &mut b, &mut acc, &mut finite);
        assert_eq!(finite, Some(true));
        // Kept for the next sweep: a `c` with a NaN is not read again.
        let mut nan = c.clone();
        nan[3] = f64::NAN;
        cd_sweep_split::<false>(&nan, k, &d, &s, 1e9, 0.0, &mut b, &mut acc, &mut finite);
        assert_eq!(finite, Some(true));
    }

    /// `all_finite` over every position of a vector's length and its tail:
    /// one NaN, one infinity of either sign, anywhere, makes it false.
    #[test]
    fn all_finite_finds_one_non_finite_entry_anywhere() {
        for n in [0, 1, 3, 4, 5, 8, 17, 33] {
            let c: Vec<f64> = (0..n).map(|i| i as f64 - 2.5).collect();
            assert!(super::all_finite(&c), "n={n}");
            for at in 0..n {
                for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
                    let mut x = c.clone();
                    x[at] = bad;
                    assert!(!super::all_finite(&x), "n={n} at {at} {bad}");
                }
            }
            let big: Vec<f64> = (0..n).map(|_| f64::MAX).collect();
            assert!(super::all_finite(&big), "the largest finite, n={n}");
        }
    }

    /// With an infinity in `c`, a zero coefficient's column is taken off
    /// (`inf * 0` is NaN), as the chained sweep's `c * 0` is: the NaN
    /// reaches the partial sums, and the coefficients are the plain
    /// definition's.
    #[test]
    fn a_zero_column_is_taken_off_where_c_is_not_finite() {
        let k = 6;
        let (mut c, d, s, _) = problem(k, 5);
        c[4 * k + 1] = f64::INFINITY;
        c[k + 4] = f64::INFINITY;
        let b0 = vec![0.0; k];
        // Known not finite, and not yet read: the zero start reads it.
        for known in [Some(false), None] {
            let (mut b, mut want) = (b0.clone(), b0.clone());
            let (mut acc, mut finite) = (Vec::new(), known);
            cd_sweep_split::<false>(&c, k, &d, &s, 0.01, 0.0, &mut b, &mut acc, &mut finite);
            assert_eq!(finite, Some(false));
            plain(&c, k, &d, &s, (0.01, 0.0), &mut want);
            for (x, y) in b.iter().zip(&want) {
                assert_eq!(x.is_nan(), y.is_nan());
                if !x.is_nan() {
                    assert_eq!(x.to_bits(), y.to_bits());
                }
            }
            // `soft_over` reads a NaN `rho` as inside the threshold (0), as
            // the chained sweep does: the NaN is in the partial sum, not
            // the move.
            assert!(
                acc.iter().any(|v| v.is_nan()),
                "the infinity reached nothing"
            );
        }
    }

    /// Signed zeros in `c`, `d` and the start, with zero columns left out
    /// (`finite`): the coefficients and the move are the plain
    /// definition's bits. Leaving out `c * 0` can only change the sign of
    /// a partial sum that is itself zero, and a zero `rho` soft-thresholds
    /// to `0.0` either way.
    #[test]
    fn signed_zeros_do_not_separate_the_skip_from_the_plain_sweep() {
        for k in [9, 40, 80] {
            let (mut c, mut d, s, _) = problem(k, 21 + k as u64);
            for i in 0..k {
                for j in 0..i {
                    if (i + 2 * j) % 5 == 0 {
                        let v = if (i + j) % 2 == 0 { -0.0 } else { 0.0 };
                        c[i * k + j] = v;
                        c[j * k + i] = v;
                    }
                }
                if i % 3 == 0 {
                    d[i] = -0.0;
                }
            }
            let start: Vec<f64> = (0..k)
                .map(|i| match i % 9 {
                    0 => 0.3,
                    1 | 5 => -0.0,
                    _ => 0.0,
                })
                .collect();
            for (l1, l2) in [(0.2, 0.0), (0.0, 0.0), (0.05, 0.02)] {
                let (mut b, mut want) = (start.clone(), start.clone());
                let mut acc = Vec::new();
                for sweep in 0..12 {
                    let got = cd_sweep_split::<false>(
                        &c,
                        k,
                        &d,
                        &s,
                        l1,
                        l2,
                        &mut b,
                        &mut acc,
                        &mut Some(true),
                    );
                    let exp = plain(&c, k, &d, &s, (l1, l2), &mut want);
                    assert_eq!(got.to_bits(), exp.to_bits(), "k={k} sweep {sweep}");
                    for (x, y) in b.iter().zip(&want) {
                        assert_eq!(x.to_bits(), y.to_bits(), "k={k} sweep {sweep}");
                    }
                }
            }
        }
    }

    /// The split sweep is the chained one but for the order of each
    /// `rho_i`'s terms: the same moves to within rounding, sweep by sweep.
    #[test]
    fn the_split_sweep_is_the_chained_one_to_rounding() {
        for k in [3, 20, 70] {
            let (c, d, s, b0) = problem(k, 3 + k as u64);
            let (mut b, mut chained) = (b0.clone(), b0);
            let (mut acc, mut acc2) = (Vec::new(), Vec::new());
            for _ in 0..3 {
                cd_sweep_split::<false>(
                    &c,
                    k,
                    &d,
                    &s,
                    0.02,
                    0.0,
                    &mut b,
                    &mut acc,
                    &mut Some(true),
                );
                super::super::cd_sweep::<false>(&c, k, &d, &s, 0.02, 0.0, &mut chained, &mut acc2);
                for (x, y) in b.iter().zip(&chained) {
                    assert!((x - y).abs() <= 1e-12 * (1.0 + y.abs()), "{x} vs {y}");
                }
            }
        }
    }
}
