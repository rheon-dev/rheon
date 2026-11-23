//! The admissible region of `(a, b)` or `(a, b, g)` and the projection
//! onto it.
//!
//! The intercept of the recursion is `C = (1 − a − b) Q̄ − g N̄`. Written
//! `Q̄^{1/2} ((1 − a − b) I − g Q̄^{−1/2} N̄ Q̄^{−1/2}) Q̄^{1/2}`, it is
//! positive semi-definite exactly when `a + b + δ g ≤ 1`, `δ` the largest
//! eigenvalue of `Q̄^{−1/2} N̄ Q̄^{−1/2}`; then every `Q_t`, a sum of `C`,
//! the positive semi-definite `a z zᵀ` and `g n nᵀ` and `b` times the last
//! `Q`, is positive definite. Outside it `Q_t` can leave the cone, and the
//! correlation is then not defined. The set is
//!
//! ```text
//! a, b, g ≥ 0,     a + b + δ g ≤ 1 − 10⁻⁴
//! ```
//!
//! (`g` and `δ` absent for the symmetric model), `10⁻⁴` the margin the
//! GARCH margins keep on their own persistence. `δ` is a function of the
//! targets, so it moves as they do.

use super::mat::{cholesky, max_eigenvalue, solve_into};

/// The persistence's bound: `a + b + δ g ≤ 1 − 10⁻⁴`.
pub const PERSISTENCE_CAP: f64 = 1.0 - 1e-4;

/// `δ = λ_max(L⁻¹ N̄ L⁻ᵀ)`, `L` the Cholesky factor of `Q̄` (the same
/// eigenvalue as `Q̄^{−1/2} N̄ Q̄^{−1/2}`'s); `None` where `Q̄` is not
/// positive definite. `work` holds `3 k²` floats.
pub(super) fn delta(qbar: &[f64], nbar: &[f64], k: usize, work: &mut Vec<f64>) -> Option<f64> {
    work.clear();
    work.resize(3 * k * k, 0.0);
    let (l, rest) = work.split_at_mut(k * k);
    let (x, m) = rest.split_at_mut(k * k);
    if !cholesky(qbar, k, l) {
        return None;
    }
    // X = L⁻¹ N̄ (column by column, forward substitution)
    for c in 0..k {
        for i in 0..k {
            let mut s = nbar[i * k + c];
            for j in 0..i {
                s -= l[i * k + j] * x[j * k + c];
            }
            x[i * k + c] = s / l[i * k + i];
        }
    }
    // M = X L⁻ᵀ = (L⁻¹ Xᵀ)ᵀ: row r of M solves L mᵣ = (row r of X)
    for r in 0..k {
        for i in 0..k {
            let mut s = x[r * k + i];
            for j in 0..i {
                s -= l[i * k + j] * m[r * k + j];
            }
            m[r * k + i] = s / l[i * k + i];
        }
    }
    for i in 0..k {
        for j in (i + 1)..k {
            let s = 0.5 * (m[i * k + j] + m[j * k + i]);
            m[i * k + j] = s;
            m[j * k + i] = s;
        }
    }
    let d = max_eigenvalue(m, k);
    d.is_finite().then_some(d.max(0.0))
}

/// Whether `θ` is in the set with the targets `Q̄`, `N̄`: no `δ` formed
/// when `g` is 0 or absent; otherwise `(cap − a − b) Q̄ − g N̄` must factor
/// (the matrix form of `a + b + δ g ≤ cap`).
pub(super) fn inside(
    theta: &[f64],
    qbar: &[f64],
    nbar: &[f64],
    k: usize,
    work: &mut Vec<f64>,
) -> bool {
    if theta.iter().any(|t| t.is_nan() || *t < 0.0) {
        return false;
    }
    let ab = theta[0] + theta[1];
    if ab.is_nan() || ab > PERSISTENCE_CAP {
        return false;
    }
    let g = theta.get(2).copied().unwrap_or(0.0);
    if g == 0.0 {
        return true;
    }
    let free = PERSISTENCE_CAP - ab;
    work.clear();
    work.resize(2 * k * k, 0.0);
    let (c, l) = work.split_at_mut(k * k);
    for i in 0..k * k {
        c[i] = free * qbar[i] - g * nbar[i];
    }
    cholesky(c, k, l)
}

/// The point of `{θ ≥ 0, cᵀθ ≤ cap}` nearest `x` in the metric of the
/// positive definite `h` (`argmin (y − x)ᵀ h (y − x)`), `c = (1, 1)` or
/// `(1, 1, δ)`. Every face of the set is tried -- each choice of the
/// constraints held as equalities, `y = x + h⁻¹Aᵀμ` with `A h⁻¹ Aᵀ μ = b −
/// A x` -- and the feasible point of least distance kept, which for a
/// convex quadratic over a polyhedron is the minimum (`p ≤ 3`, so at most
/// 16 faces). Rounding that leaves a coordinate a hair below 0 or the sum
/// a hair over the cap is taken back onto the face.
pub(super) fn project(x: &[f64], h: &[f64], c: &[f64], cap: f64) -> Vec<f64> {
    let mut y = vec![0.0; x.len()];
    project_into(x, h, c, cap, &mut y);
    y
}

/// [`project`] into `out`, allocating nothing: every buffer is on the stack
/// (`p ≤ 3`), so a row the targets carry outside the set costs no
/// allocation.
pub(super) fn project_into(x: &[f64], h: &[f64], c: &[f64], cap: f64, out: &mut [f64]) {
    let p = x.len();
    debug_assert!(p <= 3);
    let mut hinv = [0.0; 9];
    for j in 0..p {
        let mut e = [0.0; 3];
        e[j] = 1.0;
        let mut col = [0.0; 3];
        if !solve_into(h, p, &e[..p], &mut col[..p]) {
            // a metric that does not solve: the Euclidean one
            let eye = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
            let mut id = [0.0; 9];
            for a in 0..p {
                for b in 0..p {
                    id[a * p + b] = eye[a * 3 + b];
                }
            }
            return project_into(x, &id[..p * p], c, cap, out);
        }
        for i in 0..p {
            hinv[i * p + j] = col[i];
        }
    }
    let tol = 1e-12;
    let mut best: Option<(f64, [f64; 3])> = None;
    // constraint r < p is θ_r ≥ 0 (as θ_r = 0 when held); r = p is cᵀθ ≤ cap
    for mask in 0u32..(1 << (p + 1)) {
        let mut held = [0usize; 4];
        let mut m = 0;
        for r in 0..=p {
            if mask & (1 << r) != 0 {
                if m < 4 {
                    held[m] = r;
                }
                m += 1;
            }
        }
        if m > p {
            continue;
        }
        let mut rows = [[0.0; 3]; 3];
        let mut rhs = [0.0; 3];
        for (u, &r) in held[..m].iter().enumerate() {
            if r < p {
                for (i, v) in rows[u][..p].iter_mut().enumerate() {
                    *v = f64::from(u8::from(i == r));
                }
            } else {
                rows[u][..p].copy_from_slice(&c[..p]);
            }
            let target = if r < p { 0.0 } else { cap };
            rhs[u] = target
                - rows[u][..p]
                    .iter()
                    .zip(x)
                    .map(|(ai, xi)| ai * xi)
                    .sum::<f64>();
        }
        let mut y = [0.0; 3];
        y[..p].copy_from_slice(x);
        if m > 0 {
            // S = A h⁻¹ Aᵀ
            let mut hat = [[0.0; 3]; 3];
            for u in 0..m {
                for i in 0..p {
                    hat[u][i] = (0..p).map(|j| hinv[i * p + j] * rows[u][j]).sum::<f64>();
                }
            }
            let mut s = [0.0; 9];
            for u in 0..m {
                for v in 0..m {
                    s[u * m + v] = rows[u][..p]
                        .iter()
                        .zip(&hat[v][..p])
                        .map(|(ai, hi)| ai * hi)
                        .sum();
                }
            }
            let mut mu = [0.0; 3];
            if !solve_into(&s[..m * m], m, &rhs[..m], &mut mu[..m]) {
                continue;
            }
            for u in 0..m {
                for i in 0..p {
                    y[i] += hat[u][i] * mu[u];
                }
            }
        }
        let sum: f64 = c.iter().zip(&y[..p]).map(|(ci, yi)| ci * yi).sum();
        if y[..p].iter().any(|v| *v < -tol) || sum > cap + tol || !sum.is_finite() {
            continue;
        }
        let mut d = [0.0; 3];
        for i in 0..p {
            d[i] = y[i] - x[i];
        }
        let dist: f64 = (0..p)
            .map(|i| d[i] * (0..p).map(|j| h[i * p + j] * d[j]).sum::<f64>())
            .sum();
        if best.as_ref().is_none_or(|(b, _)| dist < *b) {
            best = Some((dist, y));
        }
    }
    let y = &mut out[..p];
    match best {
        Some((_, b)) => y.copy_from_slice(&b[..p]),
        None => y.iter_mut().for_each(|v| *v = 0.0),
    }
    for v in y.iter_mut() {
        if *v < 0.0 {
            *v = 0.0;
        }
    }
    for _ in 0..4 {
        let sum: f64 = c.iter().zip(y.iter()).map(|(ci, yi)| ci * yi).sum();
        if sum <= cap {
            break;
        }
        // a hair under the ratio, so the rescaled sum's rounding cannot
        // land it back over the cap
        let f = cap / sum * (1.0 - f64::EPSILON);
        for v in y.iter_mut() {
            *v *= f;
        }
    }
}

pub(super) fn identity(p: usize) -> Vec<f64> {
    let mut h = vec![0.0; p * p];
    for i in 0..p {
        h[i * p + i] = 1.0;
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::oracle;

    /// `δ` against faer's eigenvalues of `Q̄^{−1/2} N̄ Q̄^{−1/2}` formed from
    /// its eigendecomposition, which shares nothing with the Cholesky and
    /// Jacobi route; and the matrix form of the set agreeing with `δ` on
    /// both sides of the cap.
    #[test]
    fn delta_is_the_generalized_eigenvalue_and_the_set_agrees_with_it() {
        let qbar = [1.0, 0.5, 0.2, 0.5, 1.2, 0.3, 0.2, 0.3, 0.9];
        let nbar = [0.5, 0.3, 0.1, 0.3, 0.6, 0.2, 0.1, 0.2, 0.45];
        let (vals, vecs) = oracle::sym_eigen(&qbar);
        // Q̄^{-1/2} = V diag(λ^{-1/2}) Vᵀ
        let mut half = [0.0; 9];
        for i in 0..3 {
            for j in 0..3 {
                half[i * 3 + j] = (0..3)
                    .map(|m| vecs[m][i] * vecs[m][j] / vals[m].sqrt())
                    .sum();
            }
        }
        let mut m = [0.0; 9];
        for i in 0..3 {
            for j in 0..3 {
                m[i * 3 + j] = (0..3)
                    .flat_map(|a| (0..3).map(move |b| (a, b)))
                    .map(|(a, b)| half[i * 3 + a] * nbar[a * 3 + b] * half[b * 3 + j])
                    .sum();
            }
        }
        let want = *oracle::sym_eigen(&m).0.last().unwrap();
        let mut work = Vec::new();
        let got = delta(&qbar, &nbar, 3, &mut work).unwrap();
        assert!((got - want).abs() < 1e-13, "{got} {want}");
        // the set's matrix form, either side of a + b + δ g = cap
        let (a, b) = (0.03, 0.9);
        let g_edge = (PERSISTENCE_CAP - a - b) / got;
        assert!(inside(
            &[a, b, g_edge * (1.0 - 1e-9)],
            &qbar,
            &nbar,
            3,
            &mut work
        ));
        assert!(!inside(
            &[a, b, g_edge * (1.0 + 1e-9)],
            &qbar,
            &nbar,
            3,
            &mut work
        ));
        // a + b alone, at the cap and past it
        assert!(inside(
            &[0.5, PERSISTENCE_CAP - 0.5],
            &qbar,
            &nbar,
            3,
            &mut work
        ));
        assert!(!inside(
            &[0.5, PERSISTENCE_CAP - 0.5 + 1e-12],
            &qbar,
            &nbar,
            3,
            &mut work
        ));
        assert!(!inside(&[-1e-300, 0.5], &qbar, &nbar, 3, &mut work));
        // a Q̄ that is not positive definite has no δ
        assert!(delta(&[1.0, 1.0, 1.0, 1.0], &[0.5; 4], 2, &mut work).is_none());
    }

    /// The projection is the nearest point of the set in `h`'s metric: it
    /// lies in the set, a point inside is its own, and no point of a fine
    /// grid over the set is nearer.
    #[test]
    fn the_projection_is_the_nearest_point_of_the_set() {
        let h = [2.0, 0.8, 0.1, 0.8, 1.0, 0.2, 0.1, 0.2, 0.5];
        let c = [1.0, 1.0, 0.6];
        let dist = |y: &[f64], x: &[f64]| -> f64 {
            let d: Vec<f64> = y.iter().zip(x).map(|(a, b)| a - b).collect();
            (0..3)
                .map(|i| d[i] * (0..3).map(|j| h[i * 3 + j] * d[j]).sum::<f64>())
                .sum()
        };
        let inside = [0.05, 0.9, 0.02];
        assert_eq!(project(&inside, &h, &c, PERSISTENCE_CAP), inside.to_vec());
        for x in [
            [0.2, 0.95, 0.1],
            [-0.1, 0.99, 0.05],
            [0.05, -0.2, -0.3],
            [0.5, 0.6, 0.4],
            [-0.5, -0.5, 2.0],
        ] {
            let y = project(&x, &h, &c, PERSISTENCE_CAP);
            assert!(y.iter().all(|v| *v >= 0.0));
            let s: f64 = c.iter().zip(&y).map(|(a, b)| a * b).sum();
            assert!(s <= PERSISTENCE_CAP);
            let dy = dist(&y, &x);
            let n = 40;
            for i in 0..=n {
                for j in 0..=n {
                    for l in 0..=n {
                        let z = [
                            i as f64 / n as f64,
                            j as f64 / n as f64,
                            1.5 * l as f64 / n as f64,
                        ];
                        let s: f64 = c.iter().zip(&z).map(|(a, b)| a * b).sum();
                        if s <= PERSISTENCE_CAP {
                            assert!(dist(&z, &x) >= dy - 1e-12, "{x:?} -> {y:?}, {z:?} nearer");
                        }
                    }
                }
            }
        }
        // two parameters: onto a + b = cap in the Euclidean metric
        let y = project(&[0.6, 0.6], &identity(2), &[1.0, 1.0], PERSISTENCE_CAP);
        assert!((y[0] - PERSISTENCE_CAP / 2.0).abs() < 1e-15 && y[0] == y[1]);
    }
}
