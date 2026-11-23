//! The small dense algebra `garch` needs, on at most six parameters: a
//! packed symmetric matrix, Cholesky and LU solves, the relative pivot
//! rule, and the projections onto
//! a set of half-spaces `A x ≥ b`, Euclidean and in a metric (Dykstra's
//! algorithm). Nothing here allocates on a model's row but the projections,
//! which run only on a step that leaves the admissible set.

// Index loops over a matrix's rows and columns read plainest as indices.
#![allow(clippy::needless_range_loop)]

/// The index of `(i, j)` in a packed lower triangle.
pub(super) fn tri(i: usize, j: usize) -> usize {
    let (i, j) = if i >= j { (i, j) } else { (j, i) };
    i * (i + 1) / 2 + j
}

/// A packed symmetric `n × n` matrix unpacked row-major.
pub(super) fn unpack(h: &[f64], n: usize) -> Vec<f64> {
    let mut a = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            a[i * n + j] = h[tri(i, j)];
        }
    }
    a
}

/// The lower Cholesky factor of a row-major `n × n` matrix, `None` where a
/// pivot is not a positive finite number (the matrix is not positive
/// definite, or holds a value that is not a number).
pub(super) fn cholesky(a: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut l = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[i * n + j];
            for k in 0..j {
                s -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                if !(s > 0.0 && s.is_finite()) {
                    return None;
                }
                l[i * n + i] = s.sqrt();
            } else {
                l[i * n + j] = s / l[j * n + j];
            }
        }
    }
    Some(l)
}

/// `L z = b` by forward substitution.
pub(super) fn forward(l: &[f64], n: usize, b: &[f64]) -> Vec<f64> {
    let mut z = b.to_vec();
    for i in 0..n {
        for k in 0..i {
            z[i] -= l[i * n + k] * z[k];
        }
        z[i] /= l[i * n + i];
    }
    z
}

/// `Lᵀ x = z` by back substitution.
pub(super) fn backward(l: &[f64], n: usize, z: &[f64]) -> Vec<f64> {
    let mut x = z.to_vec();
    for i in (0..n).rev() {
        for k in i + 1..n {
            x[i] -= l[k * n + i] * x[k];
        }
        x[i] /= l[i * n + i];
    }
    x
}

/// `A x = b` for a row-major `A` that is positive definite, by Cholesky;
/// `None` where it is not.
pub(super) fn chol_solve(a: &[f64], n: usize, b: &[f64]) -> Option<Vec<f64>> {
    let l = cholesky(a, n)?;
    Some(backward(&l, n, &forward(&l, n, b)))
}

/// `A x = b` by Gaussian elimination with partial pivoting, `None` for a
/// singular `A` or an answer that is not finite.
pub(super) fn lu_solve(a: &[f64], n: usize, b: &[f64]) -> Option<Vec<f64>> {
    let mut m = a.to_vec();
    let mut x = b.to_vec();
    for c in 0..n {
        let p = (c..n).max_by(|&i, &j| m[i * n + c].abs().total_cmp(&m[j * n + c].abs()))?;
        if m[p * n + c] == 0.0 || !m[p * n + c].is_finite() {
            return None;
        }
        if p != c {
            for k in 0..n {
                m.swap(c * n + k, p * n + k);
            }
            x.swap(c, p);
        }
        for i in c + 1..n {
            let f = m[i * n + c] / m[c * n + c];
            for k in c..n {
                m[i * n + k] -= f * m[c * n + k];
            }
            x[i] -= f * x[c];
        }
    }
    for i in (0..n).rev() {
        for k in i + 1..n {
            x[i] -= m[i * n + k] * x[k];
        }
        x[i] /= m[i * n + i];
    }
    x.iter().all(|v| v.is_finite()).then_some(x)
}

/// A matrix with each diagonal entry raised by `rel` of its own size: a
/// ridge that scales with each parameter's units, so scaling a series
/// leaves the step it gives unchanged.
pub(super) fn ridged(a: &[f64], n: usize, rel: f64) -> Vec<f64> {
    let mut r = a.to_vec();
    for i in 0..n {
        r[i * n + i] += rel * r[i * n + i].abs();
    }
    r
}

/// The relative pivot rule: whether every pivot of the Cholesky
/// factorization of the symmetric `n × n` matrix `a` (row-major), `L_ii²`,
/// is a finite number above `floor` times `a`'s largest diagonal entry, as
/// the library's other definite solves read a system (`expectile`'s `G`,
/// and `hawkes`'s through `solve::solve_definite_in`, whose own diagonal is
/// the largest on a unit diagonal). A pivot is at least the matrix's least
/// eigenvalue, so a pivot under the floor means a condition number over
/// `1 / floor`: a term the others nearly reproduce. In plain scalar
/// arithmetic in a fixed order, so the answer is the same on every
/// platform. False where the largest diagonal entry is not a positive
/// finite number.
pub(super) fn pivots_clear(a: &[f64], n: usize, floor: f64) -> bool {
    let top = (0..n).map(|i| a[i * n + i]).fold(0.0, f64::max);
    if !(top > 0.0 && top.is_finite()) {
        return false;
    }
    let cut = floor * top;
    let mut l = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[i * n + j];
            for m in 0..j {
                s -= l[i * n + m] * l[j * n + m];
            }
            if i == j {
                if !(s > cut && s.is_finite()) {
                    return false;
                }
                l[i * n + i] = s.sqrt();
            } else {
                l[i * n + j] = s / l[j * n + j];
            }
        }
    }
    true
}

/// Half-spaces `A x ≥ b`, `A` row-major with `n` columns.
#[derive(Debug, Clone)]
pub(super) struct HalfSpaces {
    pub a: Vec<f64>,
    pub b: Vec<f64>,
    pub n: usize,
}

impl HalfSpaces {
    fn rows(&self) -> usize {
        self.b.len()
    }

    fn row(&self, i: usize) -> &[f64] {
        &self.a[i * self.n..(i + 1) * self.n]
    }

    fn slack(&self, i: usize, x: &[f64]) -> f64 {
        self.row(i).iter().zip(x).map(|(a, x)| a * x).sum::<f64>() - self.b[i]
    }

    /// Whether `x` is in the set, to `tol` of each constraint's own size
    /// at `x` (`Σ |a_j x_j| + |b|`), so a scaled point is held alike.
    pub fn holds(&self, x: &[f64], tol: f64) -> bool {
        (0..self.rows()).all(|i| {
            let size: f64 = self
                .row(i)
                .iter()
                .zip(x)
                .map(|(a, x)| (a * x).abs())
                .sum::<f64>()
                + self.b[i].abs();
            self.slack(i, x) >= -tol * size
        })
    }

    /// The Euclidean projection of `x` onto the set by Dykstra's algorithm:
    /// at most 2,000 sweeps, stopped when a sweep moves no coordinate by
    /// `1e-14`, then each constraint still broken by rounding pushed back
    /// across its plane. `x` itself where it is in the set, to `1e-15`.
    pub fn project(&self, x: &[f64]) -> Vec<f64> {
        if self.holds(x, 1e-15) {
            return x.to_vec();
        }
        let (r, n) = (self.rows(), self.n);
        let nrm: Vec<f64> = (0..r)
            .map(|i| self.row(i).iter().map(|a| a * a).sum())
            .collect();
        let mut y = x.to_vec();
        let mut inc = vec![0.0; r * n];
        let mut w = vec![0.0; n];
        for _ in 0..2000 {
            let prev = y.clone();
            for i in 0..r {
                for k in 0..n {
                    w[k] = y[k] + inc[i * n + k];
                }
                let v = self.slack(i, &w);
                for k in 0..n {
                    let z = if v < 0.0 {
                        w[k] - (v / nrm[i]) * self.a[i * n + k]
                    } else {
                        w[k]
                    };
                    inc[i * n + k] = w[k] - z;
                    y[k] = z;
                }
            }
            let moved = y
                .iter()
                .zip(&prev)
                .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
            if moved < 1e-14 {
                break;
            }
        }
        for i in 0..r {
            let v = self.slack(i, &y);
            if v < 0.0 {
                for k in 0..n {
                    y[k] -= (v / nrm[i]) * self.a[i * n + k] * (1.0 + 1e-12);
                }
            }
        }
        y
    }

    /// `argmin (y − x)ᵀ M (y − x)` over the set, `M` row-major positive
    /// definite: with `M = L Lᵀ` and `u = Lᵀ y`, the Euclidean projection of
    /// `Lᵀ x` onto `{A L⁻ᵀ u ≥ b}`. A Newton step projected in the
    /// Euclidean metric is no ascent direction of the likelihood when the
    /// constraints bind on parameters it barely tells apart; in the step's
    /// own metric it is. `None` where `M` is not positive definite.
    pub fn project_metric(&self, x: &[f64], m: &[f64]) -> Option<Vec<f64>> {
        if self.holds(x, 1e-15) {
            return Some(x.to_vec());
        }
        let n = self.n;
        let l = cholesky(m, n)?;
        let mut u = vec![0.0; n];
        for i in 0..n {
            for k in i..n {
                u[i] += l[k * n + i] * x[k];
            }
        }
        let mut a = Vec::with_capacity(self.a.len());
        for i in 0..self.rows() {
            a.extend(forward(&l, n, self.row(i)));
        }
        let moved = HalfSpaces {
            a,
            b: self.b.clone(),
            n,
        }
        .project(&u);
        let y = backward(&l, n, &moved);
        y.iter().all(|v| v.is_finite()).then_some(y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_solves_are_faers() {
        let a = [4.0, 1.0, 0.5, 1.0, 3.0, 0.2, 0.5, 0.2, 2.0];
        let b = [1.0, -2.0, 0.5];
        let want = crate::oracle::solve(&a, &b);
        for got in [chol_solve(&a, 3, &b).unwrap(), lu_solve(&a, 3, &b).unwrap()] {
            for (g, w) in got.iter().zip(&want) {
                assert!((g - w).abs() < 1e-14, "{got:?} {want:?}");
            }
        }
        // a matrix that is not positive definite has no Cholesky factor,
        // and a singular one no LU solve
        assert!(cholesky(&[1.0, 2.0, 2.0, 1.0], 2).is_none());
        assert!(lu_solve(&[1.0, 2.0, 2.0, 4.0], 2, &[1.0, 1.0]).is_none());
    }

    #[test]
    fn a_pivot_is_read_against_the_largest_diagonal_entry() {
        // [[1, r], [r, 1]]: the second pivot is 1 − r², at any scale
        for scale in [1.0, 1e-8, 1e8] {
            let a = |r: f64| [scale, scale * r, scale * r, scale];
            let r = |p: f64| (1.0f64 - p).sqrt();
            assert!(pivots_clear(&a(r(2e-10)), 2, 1e-10), "{scale}");
            assert!(!pivots_clear(&a(r(0.5e-10)), 2, 1e-10), "{scale}");
            // exactly collinear: the pivot is 0
            assert!(!pivots_clear(&a(1.0), 2, 1e-10), "{scale}");
        }
        // the floor is a share of the largest diagonal entry, not of the
        // pivot's own: a well-conditioned small entry beside a large one
        let a = [1e6, 0.0, 0.0, 1e-5];
        assert!(!pivots_clear(&a, 2, 1e-10));
        assert!(pivots_clear(&a, 2, 1e-12));
        // the last pivot is 1 / (A⁻¹)_nn, read from faer's LU
        let a = [
            4.0, 1.0, 0.5, 0.2, //
            1.0, 3.0, 0.2, 0.1, //
            0.5, 0.2, 2.0, 1.4, //
            0.2, 0.1, 1.4, 1.0,
        ];
        let last = 1.0 / crate::oracle::solve(&a, &[0.0, 0.0, 0.0, 1.0])[3];
        assert!(last < 0.1, "{last}");
        assert!(pivots_clear(&a, 4, last / 4.0 * (1.0 - 1e-9)));
        assert!(!pivots_clear(&a, 4, last / 4.0 * (1.0 + 1e-9)));
        // not a number, or no positive diagonal: no fit
        assert!(!pivots_clear(&[f64::NAN, 0.0, 0.0, 1.0], 2, 1e-10));
        assert!(!pivots_clear(&[0.0, 0.0, 0.0, 0.0], 2, 1e-10));
    }

    #[test]
    fn a_point_inside_is_its_own_projection_and_one_outside_lands_on_the_nearest_face() {
        // alpha, beta >= 0, alpha + beta <= 0.9999
        let hs = HalfSpaces {
            a: vec![1.0, 0.0, 0.0, 1.0, -1.0, -1.0],
            b: vec![0.0, 0.0, -0.9999],
            n: 2,
        };
        assert_eq!(hs.project(&[0.1, 0.8]), vec![0.1, 0.8]);
        let y = hs.project(&[0.3, 0.9]);
        // the nearest point of the face alpha + beta = 0.9999
        assert!(
            (y[0] - 0.19995).abs() < 1e-12 && (y[1] - 0.79995).abs() < 1e-12,
            "{y:?}"
        );
        let y = hs.project(&[-0.5, 2.0]);
        assert!(y[0].abs() < 1e-12 && (y[1] - 0.9999).abs() < 1e-12, "{y:?}");
        // in the metric of the identity the projection is the Euclidean one
        let m = hs
            .project_metric(&[0.3, 0.9], &[1.0, 0.0, 0.0, 1.0])
            .unwrap();
        assert!((m[0] - 0.19995).abs() < 1e-12 && (m[1] - 0.79995).abs() < 1e-12);
        // a metric that makes beta dear moves alpha instead
        let m = hs
            .project_metric(&[0.3, 0.9], &[1.0, 0.0, 0.0, 1e6])
            .unwrap();
        assert!(m[1] > 0.8999 && hs.holds(&m, 1e-12), "{m:?}");
    }
}
