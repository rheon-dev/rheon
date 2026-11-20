//! Task 479 (lasso split prototype): a sweep of the `lasso` model's
//! coordinate descent with no dependent chain of subtractions, behind the
//! `lasso-split` build feature, for the user to decide on. Not the default.
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
//! `c * 0` is.

/// One sweep of the descent with the two runs swapped (the module's
/// account); the largest move. `acc` is the partial sums' buffer; `finite`
/// says every entry of `c` is finite.
#[allow(clippy::too_many_arguments)]
pub(super) fn cd_sweep_split(
    c: &[f64],
    k: usize,
    d: &[f64],
    s: &[f64],
    l1: f64,
    l2: f64,
    b: &mut [f64],
    acc: &mut Vec<f64>,
    finite: bool,
) -> f64 {
    debug_assert_eq!(c.len(), k * k);
    acc.clear();
    acc.extend_from_slice(&d[..k]);
    let skip = |v: f64| finite && v == 0.0;
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
                super::soft_over(acc[i], l1, c[i * k + i] + l2)
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

#[cfg(test)]
mod tests {
    use super::cd_sweep_split;

    /// The split sweep one term at a time, in the module's order: the
    /// definition the blocked one must equal to the bit.
    fn plain(
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
                super::super::soft_over(a[i], l1, c[i * k + i] + l2)
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
    /// kind of penalty, sweep after sweep.
    #[test]
    fn the_blocked_sweep_is_the_plain_one_to_the_bit() {
        for k in [0, 1, 2, 3, 4, 5, 7, 8, 9, 13, 64, 67] {
            for (l1, l2) in [(0.0, 0.0), (0.05, 0.0), (0.3, 0.1)] {
                let (c, d, s, b0) = problem(k, 11 + k as u64);
                let (mut b, mut want) = (b0.clone(), b0);
                let mut acc = Vec::new();
                for sweep in 0..4 {
                    let got = cd_sweep_split(&c, k, &d, &s, l1, l2, &mut b, &mut acc, true);
                    let exp = plain(&c, k, &d, &s, (l1, l2), &mut want);
                    assert_eq!(got.to_bits(), exp.to_bits(), "k={k} sweep {sweep}");
                    for (x, y) in b.iter().zip(&want) {
                        assert_eq!(x.to_bits(), y.to_bits(), "k={k} sweep {sweep}");
                    }
                }
            }
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
        let (mut b, mut want) = (b0.clone(), b0);
        let mut acc = Vec::new();
        cd_sweep_split(&c, k, &d, &s, 0.01, 0.0, &mut b, &mut acc, false);
        plain(&c, k, &d, &s, (0.01, 0.0), &mut want);
        for (x, y) in b.iter().zip(&want) {
            assert_eq!(x.is_nan(), y.is_nan());
            if !x.is_nan() {
                assert_eq!(x.to_bits(), y.to_bits());
            }
        }
        // `soft_over` reads a NaN `rho` as inside the threshold (0), as the
        // chained sweep does: the NaN is in the partial sum, not the move.
        assert!(
            acc.iter().any(|v| v.is_nan()),
            "the infinity reached nothing"
        );
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
                cd_sweep_split(&c, k, &d, &s, 0.02, 0.0, &mut b, &mut acc, true);
                super::super::cd_sweep(&c, k, &d, &s, 0.02, 0.0, &mut chained, &mut acc2);
                for (x, y) in b.iter().zip(&chained) {
                    assert!((x - y).abs() <= 1e-12 * (1.0 + y.abs()), "{x} vs {y}");
                }
            }
        }
    }
}
