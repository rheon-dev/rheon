//! The small dense algebra the correlation stage runs: row-major `k × k`
//! matrices, a Cholesky factor and what it solves, the largest eigenvalue
//! of a symmetric matrix by cyclic Jacobi, and a pivoted solve of the
//! `p ≤ 4` systems of the estimator. Every operation is IEEE arithmetic in
//! a fixed order, and `ln` is the `libm` crate's, so the same inputs give
//! the same bits on every platform.

/// `L` with `A = L Lᵀ`, lower triangle row-major into `l` (the upper left
/// at 0); false where `A` is not positive definite or not finite.
pub(super) fn cholesky(a: &[f64], k: usize, l: &mut [f64]) -> bool {
    for i in 0..k {
        for j in 0..=i {
            let mut s = a[i * k + j];
            for m in 0..j {
                s -= l[i * k + m] * l[j * k + m];
            }
            if i == j {
                if !(s > 0.0 && s.is_finite()) {
                    return false;
                }
                l[i * k + i] = s.sqrt();
            } else {
                l[i * k + j] = s / l[j * k + j];
            }
        }
        for j in (i + 1)..k {
            l[i * k + j] = 0.0;
        }
    }
    true
}

/// `b ← A⁻¹ b` from `A`'s factor `l`.
pub(super) fn chol_solve(l: &[f64], k: usize, b: &mut [f64]) {
    for i in 0..k {
        let mut s = b[i];
        for m in 0..i {
            s -= l[i * k + m] * b[m];
        }
        b[i] = s / l[i * k + i];
    }
    for i in (0..k).rev() {
        let mut s = b[i];
        for m in (i + 1)..k {
            s -= l[m * k + i] * b[m];
        }
        b[i] = s / l[i * k + i];
    }
}

/// `A⁻¹` from `A`'s factor `l`, symmetric to the bit (the lower triangle
/// solved, the upper copied); `col` is `k` floats of scratch.
pub(super) fn chol_inverse(l: &[f64], k: usize, inv: &mut [f64], col: &mut [f64]) {
    for j in 0..k {
        col.iter_mut().for_each(|c| *c = 0.0);
        col[j] = 1.0;
        chol_solve(l, k, col);
        for i in j..k {
            inv[i * k + j] = col[i];
        }
    }
    for i in 0..k {
        for j in (i + 1)..k {
            inv[i * k + j] = inv[j * k + i];
        }
    }
}

/// `ln det A` from `A`'s factor: `2 Σ ln L_ii`.
pub(super) fn chol_log_det(l: &[f64], k: usize) -> f64 {
    let mut s = 0.0;
    for i in 0..k {
        s += libm::log(l[i * k + i]);
    }
    2.0 * s
}

/// The largest eigenvalue of the symmetric `a` (row-major, overwritten) by
/// cyclic Jacobi rotations, swept until every off-diagonal entry is below
/// `10⁻¹⁶` of the matrix's scale, at most 100 sweeps.
pub(super) fn max_eigenvalue(a: &mut [f64], k: usize) -> f64 {
    let scale = a.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    if scale == 0.0 || !scale.is_finite() {
        return if scale == 0.0 { 0.0 } else { f64::NAN };
    }
    for _ in 0..100 {
        let mut off = 0.0_f64;
        for p in 0..k {
            for q in (p + 1)..k {
                off = off.max(a[p * k + q].abs());
            }
        }
        if off <= 1e-16 * scale {
            break;
        }
        for p in 0..k {
            for q in (p + 1)..k {
                let apq = a[p * k + q];
                if apq == 0.0 {
                    continue;
                }
                let theta = (a[q * k + q] - a[p * k + p]) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let t = if theta == 0.0 { 1.0 } else { t };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for r in 0..k {
                    let (arp, arq) = (a[r * k + p], a[r * k + q]);
                    a[r * k + p] = c * arp - s * arq;
                    a[r * k + q] = s * arp + c * arq;
                }
                for r in 0..k {
                    let (apr, aqr) = (a[p * k + r], a[q * k + r]);
                    a[p * k + r] = c * apr - s * aqr;
                    a[q * k + r] = s * apr + c * aqr;
                }
            }
        }
    }
    (0..k)
        .map(|i| a[i * k + i])
        .fold(f64::NEG_INFINITY, f64::max)
}

/// `x` with `A x = b` for a small square `A` (row-major `p × p`), by
/// Gaussian elimination with partial pivoting; `None` where a pivot is 0
/// or not finite.
pub(super) fn solve_small(a: &[f64], p: usize, b: &[f64]) -> Option<Vec<f64>> {
    let mut x = vec![0.0; p];
    solve_into(a, p, b, &mut x).then_some(x)
}

/// [`solve_small`] into `x`, for `p ≤ 4`, allocating nothing (a learned
/// row's step); false where it has no solution.
pub(super) fn solve_into(a: &[f64], p: usize, b: &[f64], out: &mut [f64]) -> bool {
    let mut store = [0.0; 16];
    let mut xs = [0.0; 4];
    let (m, x) = if p <= 4 {
        store[..p * p].copy_from_slice(&a[..p * p]);
        xs[..p].copy_from_slice(&b[..p]);
        (&mut store[..p * p], &mut xs[..p])
    } else {
        return match solve_large(a, p, b) {
            Some(v) => {
                out[..p].copy_from_slice(&v);
                true
            }
            None => false,
        };
    };
    if !eliminate(m, x, p) {
        return false;
    }
    out[..p].copy_from_slice(x);
    out[..p].iter().all(|v| v.is_finite())
}

fn solve_large(a: &[f64], p: usize, b: &[f64]) -> Option<Vec<f64>> {
    let (mut m, mut x) = (a.to_vec(), b.to_vec());
    (eliminate(&mut m, &mut x, p) && x.iter().all(|v| v.is_finite())).then_some(x)
}

/// Gaussian elimination with partial pivoting of `m x = x` in place.
fn eliminate(m: &mut [f64], x: &mut [f64], p: usize) -> bool {
    for c in 0..p {
        let Some(piv) = (c..p).max_by(|&i, &j| m[i * p + c].abs().total_cmp(&m[j * p + c].abs()))
        else {
            return false;
        };
        if !(m[piv * p + c] != 0.0 && m[piv * p + c].is_finite()) {
            return false;
        }
        if piv != c {
            for j in 0..p {
                m.swap(c * p + j, piv * p + j);
            }
            x.swap(c, piv);
        }
        for r in (c + 1)..p {
            let f = m[r * p + c] / m[c * p + c];
            for j in c..p {
                m[r * p + j] -= f * m[c * p + j];
            }
            x[r] -= f * x[c];
        }
    }
    for c in (0..p).rev() {
        let mut s = x[c];
        for j in (c + 1)..p {
            s -= m[c * p + j] * x[j];
        }
        x[c] = s / m[c * p + c];
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::oracle;

    fn spd(k: usize, seed: u64) -> Vec<f64> {
        let mut s = seed;
        let mut next = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 11) as f64) / ((1u64 << 53) as f64) - 0.5
        };
        let b: Vec<f64> = (0..k * k).map(|_| next()).collect();
        let mut a = vec![0.0; k * k];
        for i in 0..k {
            for j in 0..k {
                a[i * k + j] = (0..k).map(|m| b[i * k + m] * b[j * k + m]).sum::<f64>()
                    + if i == j { 0.5 } else { 0.0 };
            }
        }
        a
    }

    /// The factor's solve, inverse and log-determinant against faer's LU
    /// and eigenvalues (`crate::oracle`), which share no arithmetic with it.
    #[test]
    fn the_factor_solves_inverts_and_takes_the_log_determinant() {
        for k in [1, 2, 3, 5, 8] {
            let a = spd(k, k as u64);
            let mut l = vec![0.0; k * k];
            assert!(cholesky(&a, k, &mut l));
            let b: Vec<f64> = (0..k).map(|i| i as f64 - 1.5).collect();
            let mut x = b.clone();
            chol_solve(&l, k, &mut x);
            for (got, want) in x.iter().zip(oracle::solve(&a, &b)) {
                assert!((got - want).abs() < 1e-12 * (1.0 + want.abs()));
            }
            let mut inv = vec![0.0; k * k];
            let mut col = vec![0.0; k];
            chol_inverse(&l, k, &mut inv, &mut col);
            for (got, want) in inv.iter().zip(oracle::inverse(&a)) {
                assert!((got - want).abs() < 1e-11 * (1.0 + want.abs()));
            }
            assert!((chol_log_det(&l, k) - oracle::log_det(&a)).abs() < 1e-12);
        }
        // not positive definite, either side of 0
        let mut l = vec![0.0; 4];
        assert!(!cholesky(&[1.0, 1.0, 1.0, 1.0], 2, &mut l));
        assert!(cholesky(&[1.0, 0.999, 0.999, 1.0], 2, &mut l));
        assert!(!cholesky(&[1.0, f64::NAN, f64::NAN, 1.0], 2, &mut l));
    }

    /// Jacobi's largest eigenvalue against faer's eigensolver.
    #[test]
    fn the_largest_eigenvalue_is_faers() {
        for k in [1, 2, 3, 4, 7] {
            let a = spd(k, 40 + k as u64);
            let want = *oracle::sym_eigen(&a).0.last().unwrap();
            let mut m = a.clone();
            let got = max_eigenvalue(&mut m, k);
            assert!((got - want).abs() < 1e-13 * want.abs(), "{k}: {got} {want}");
        }
        assert_eq!(max_eigenvalue(&mut [0.0; 4], 2), 0.0);
        // an indefinite matrix: its largest, not its largest in magnitude
        let got = max_eigenvalue(&mut [1.0, 2.0, 2.0, 1.0], 2);
        assert!((got - 3.0).abs() < 1e-15);
        let got = max_eigenvalue(&mut [-5.0, 0.0, 0.0, 1.0], 2);
        assert_eq!(got, 1.0);
    }

    /// The pivoted solve against faer's LU, and a singular system refused.
    #[test]
    fn the_small_solve_is_faers() {
        let a = [0.0, 2.0, 1.0, 3.0, 1.0, 0.5, 1.0, 1.0, 4.0];
        let b = [1.0, 2.0, 3.0];
        let x = solve_small(&a, 3, &b).unwrap();
        for (got, want) in x.iter().zip(oracle::solve(&a, &b)) {
            assert!((got - want).abs() < 1e-14);
        }
        assert!(solve_small(&[1.0, 2.0, 2.0, 4.0], 2, &[1.0, 1.0]).is_none());
    }
}
