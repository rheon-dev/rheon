//! Linear equality constraints `R β = r` on least-squares slope fits
//! (`EwRidge` and `Rls`).
//!
//! Given an unconstrained symmetric positive definite normal system
//! `H x = g` (`H = L Lᵀ` or `H = R_cholᵀ R_chol`) with unconstrained solution
//! `x_u = H⁻¹ g` and linear constraints `C x = r` (`q × d`, full row rank):
//!
//! ```text
//! x_c   = x_u − H⁻¹ Cᵀ (C H⁻¹ Cᵀ)⁻¹ (C x_u − r)
//! H_c⁻¹ = H⁻¹ − H⁻¹ Cᵀ (C H⁻¹ Cᵀ)⁻¹ C H⁻¹
//! ```
//!
//! Working in `L`-whitened coordinates (`L V = Cᵀ`), equilibrating the
//! columns of `V` by `d_a = ‖V[:, a]‖₂` gives a unit-diagonal correlation
//! matrix `M̂ = V̂ᵀ V̂ = L_m L_mᵀ` (strict Cholesky, zero jitter). Then
//! `W = V̂ L_m⁻ᵀ` (`d × q`) has orthonormal columns (`Wᵀ W = I_q`),
//! `U_w = L⁻ᵀ W` satisfies `H_c⁻¹ = H⁻¹ − U_w U_wᵀ = L⁻ᵀ (I − W Wᵀ) L⁻¹`,
//! and for any `a = L⁻¹ v`, the projected quadratic form
//! `vᵀ H_c⁻¹ v = ‖a − W (Wᵀ a)‖²` is guaranteed non-negative without
//! catastrophic cancellation.

use serde::{Deserialize, Serialize};

use crate::solve::SpdFactor;

/// Linear equality constraints `R β = r` on the slope coefficients.
///
/// `matrix` holds `q` rows each of length `n_features` (slopes only, in
/// feature order; any intercept is unconstrained), and `rhs` holds the `q`
/// target values. `R` must have full row rank (`1 ≤ q ≤ n_features`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LinearConstraints {
    /// Constraint matrix `R`, `q` rows of length `n_features`.
    pub matrix: Vec<Vec<f64>>,
    /// Right-hand side `r`, length `q`.
    pub rhs: Vec<f64>,
}

impl LinearConstraints {
    /// Number of equality constraints `q`.
    pub fn n_constraints(&self) -> usize {
        self.matrix.len()
    }

    /// Validate dimensions, finiteness, and full row rank against `n_features`.
    pub fn validate(&self, kind: &str, n_features: usize) -> Result<(), String> {
        let mut no = crate::Refusals::new();
        let q = self.matrix.len();
        if q == 0 {
            no.refuse(format!(
                "{kind}: constraints matrix must have at least 1 row, but it is empty. \
                 Pass at least one linear constraint row, or leave constraints out."
            ));
        }
        if self.rhs.len() != q {
            no.refuse(format!(
                "{kind}: constraints rhs must have one entry per row of matrix ({q}), but \
                 it has {}. Provide one right-hand side value per row of matrix.",
                self.rhs.len()
            ));
        }
        if q > n_features {
            no.refuse(format!(
                "{kind}: constraints has {q} rows for {n_features} features, but at most \
                 {n_features} linearly independent equality constraints can hold on the \
                 slopes. Remove redundant constraints so there are at most {n_features}."
            ));
        }
        let mut shape_ok = q > 0 && q <= n_features && self.rhs.len() == q;
        for (i, row) in self.matrix.iter().enumerate() {
            if row.len() != n_features {
                shape_ok = false;
                no.refuse(format!(
                    "{kind}: constraints matrix row {i} must have {n_features} entries (one per \
                     feature), but it has {}. Pass one weight per feature in feature order.",
                    row.len()
                ));
            }
            if row.iter().any(|v| !v.is_finite()) {
                shape_ok = false;
                no.refuse(format!(
                    "{kind}: constraints matrix row {i} contains a non-finite value. Replace \
                     every non-finite constraint weight with a finite number."
                ));
            }
        }
        if self.rhs.iter().any(|v| !v.is_finite()) {
            shape_ok = false;
            no.refuse(format!(
                "{kind}: constraints rhs contains a non-finite value. Replace every non-finite \
                 right-hand side value with a finite number."
            ));
        }
        if !shape_ok {
            return no.finish();
        }
        // Check row norms and full row rank on the row-normalized matrix so
        // differently scaled orthogonal rows (e.g. `[1, 1]` and `[1e-8, -1e-8]`)
        // are judged by angle rather than row scale.
        let mut unit_rows: Vec<Vec<f64>> = Vec::with_capacity(q);
        for (i, row) in self.matrix.iter().enumerate() {
            let norm = row.iter().map(|v| v * v).sum::<f64>().sqrt();
            if !(norm > 0.0 && norm.is_finite()) {
                no.refuse(format!(
                    "{kind}: constraints matrix row {i} is all zeros (or overflows its norm). \
                     Replace that row with at least one non-zero finite entry."
                ));
            } else {
                unit_rows.push(row.iter().map(|v| v / norm).collect());
            }
        }
        if unit_rows.len() == q && q > 1 {
            let rank = row_rank_unit(&mut unit_rows, n_features);
            if rank < q {
                no.refuse(format!(
                    "{kind}: constraints matrix rows are linearly dependent (rank {rank} for \
                     {q} rows). Remove redundant or conflicting rows so the constraint matrix \
                     has full row rank."
                ));
            }
        }
        no.finish()
    }
}

/// Row rank of a `q × k` matrix whose rows already have unit Euclidean norm,
/// via column-pivoted Modified Gram-Schmidt with reorthogonalization.
fn row_rank_unit(rows: &mut [Vec<f64>], k: usize) -> usize {
    let q = rows.len();
    let tol = (q.max(k) as f64) * f64::EPSILON;
    let mut rank = 0usize;
    for t in 0..q {
        // Pick the remaining row with largest residual norm.
        let mut best = t;
        let mut best_norm = 0.0f64;
        for (i, row) in rows.iter().enumerate().skip(t) {
            let n = row.iter().map(|v| v * v).sum::<f64>().sqrt();
            if n > best_norm {
                best_norm = n;
                best = i;
            }
        }
        if best_norm <= tol || !best_norm.is_finite() {
            break;
        }
        rows.swap(t, best);
        let inv = best_norm.recip();
        for v in &mut rows[t] {
            *v *= inv;
        }
        rank += 1;
        // Two passes of Gram-Schmidt against `rows[0..=t]` (Kahan–Parlett)
        // for full machine-precision orthogonality.
        for _ in 0..2 {
            let (done, rest) = rows.split_at_mut(t + 1);
            for rem in rest.iter_mut() {
                for basis in done.iter() {
                    let dot: f64 = basis.iter().zip(rem.iter()).map(|(a, b)| a * b).sum();
                    for (r, &b) in rem.iter_mut().zip(basis.iter()) {
                        *r -= dot * b;
                    }
                }
            }
        }
    }
    rank
}

/// Precomputed projection in whitened coordinates for equality constraints
/// `C x = r` over an SPD system of order `d`.
#[derive(Debug, Clone)]
pub(crate) struct ConstraintProj {
    d: usize,
    q: usize,
    /// `C`, row-major `q × d`.
    c_mat: Vec<f64>,
    /// `r`, length `q`.
    rhs: Vec<f64>,
    /// `d_scale[a] = ‖V[:, a]‖₂`, length `q`.
    d_scale: Vec<f64>,
    /// Strict lower-triangular Cholesky factor `L_m` of equilibrated `M̂`,
    /// row-major `q × q`.
    l_m: Vec<f64>,
    /// Orthonormal whitened basis `W = V̂ L_m⁻ᵀ`, column-major `d × q`.
    w: Vec<f64>,
    /// Unwhitened basis `U_w = L⁻ᵀ W`, column-major `d × q`.
    u_w: Vec<f64>,
}

impl ConstraintProj {
    /// Build from an `SpdFactor` `L` (`H = L Lᵀ`) and constraint matrix `c_mat`
    /// (row-major `q × d`) with right-hand side `rhs` (length `q`).
    /// Returns `None` if `C H⁻¹ Cᵀ` is singular or ill-conditioned.
    pub(crate) fn from_spd_factor(
        factor: &SpdFactor,
        c_mat: Vec<f64>,
        rhs: Vec<f64>,
        d: usize,
        q: usize,
    ) -> Option<Self> {
        if q == 0 || d < q || c_mat.len() != q * d || rhs.len() != q {
            return None;
        }
        // `Cᵀ` column-major `d × q`: column `a` is row `a` of `C`.
        let mut ct = vec![0.0; d * q];
        for a in 0..q {
            ct[a * d..(a + 1) * d].copy_from_slice(&c_mat[a * d..(a + 1) * d]);
        }
        let v = factor.solve_lower_cols(&ct, d, q);
        let (d_scale, l_m, w) = orthonormalize_whitened(&v, d, q)?;
        let u_w = factor.solve_upper_transpose_cols(&w, d, q);
        if !u_w.iter().all(|x| x.is_finite()) {
            return None;
        }
        Some(Self {
            d,
            q,
            c_mat,
            rhs,
            d_scale,
            l_m,
            w,
            u_w,
        })
    }

    /// Build from a packed upper-triangular Cholesky factor `R_chol`
    /// (`H = R_cholᵀ R_chol`, length `d(d+1)/2`, row-major packed as in `Rls`).
    pub(crate) fn from_upper_packed(
        r_packed: &[f64],
        d: usize,
        c_mat: Vec<f64>,
        rhs: Vec<f64>,
        q: usize,
    ) -> Option<Self> {
        if q == 0 || d < q || c_mat.len() != q * d || rhs.len() != q {
            return None;
        }
        // Solve `Rᵀ V = Cᵀ`, one column `a` at a time.
        let mut v = vec![0.0; d * q];
        for a in 0..q {
            let src = &c_mat[a * d..(a + 1) * d];
            let dst = &mut v[a * d..(a + 1) * d];
            if !solve_upper_t_packed(r_packed, d, src, dst) {
                return None;
            }
        }
        let (d_scale, l_m, w) = orthonormalize_whitened(&v, d, q)?;
        // Solve `R U_w = W`, one column `a` at a time.
        let mut u_w = vec![0.0; d * q];
        for a in 0..q {
            let src = &w[a * d..(a + 1) * d];
            let dst = &mut u_w[a * d..(a + 1) * d];
            if !solve_upper_packed(r_packed, d, src, dst) {
                return None;
            }
        }
        if !u_w.iter().all(|x| x.is_finite()) {
            return None;
        }
        Some(Self {
            d,
            q,
            c_mat,
            rhs,
            d_scale,
            l_m,
            w,
            u_w,
        })
    }

    pub(crate) fn n_constraints(&self) -> usize {
        self.q
    }

    /// Project unconstrained solution `x` (length `d`) onto `C x = r` in place:
    /// `x ← x − H⁻¹ Cᵀ (C H⁻¹ Cᵀ)⁻¹ (C x − r)`.
    pub(crate) fn project_solution(&self, x: &mut [f64]) {
        debug_assert_eq!(x.len(), self.d);
        let q = self.q;
        let d = self.d;
        // `res[a] = (C[a]·x − r[a]) / d_scale[a]`, then forward-solve `L_m z = res`.
        let mut z = vec![0.0; q];
        for a in 0..q {
            let dot: f64 = self.c_mat[a * d..(a + 1) * d]
                .iter()
                .zip(x.iter())
                .map(|(c, v)| c * v)
                .sum();
            let mut acc = (dot - self.rhs[a]) / self.d_scale[a];
            for (b, &zb) in z.iter().enumerate().take(a) {
                acc -= self.l_m[a * q + b] * zb;
            }
            z[a] = acc / self.l_m[a * q + a];
        }
        for (a, &za) in z.iter().enumerate() {
            let col = &self.u_w[a * d..(a + 1) * d];
            for (xi, &ui) in x.iter_mut().zip(col.iter()) {
                *xi -= ui * za;
            }
        }
    }

    /// Projected quadratic form `vᵀ H_c⁻¹ v = ‖(I − W Wᵀ) a‖²` from the
    /// whitened vector `a = L⁻¹ v` (or `R_chol⁻ᵀ v`), length `d`.
    pub(crate) fn projected_quad_from_whitened(&self, a: &[f64]) -> f64 {
        debug_assert_eq!(a.len(), self.d);
        let d = self.d;
        let q = self.q;
        let mut t = vec![0.0; q];
        for (c, tc) in t.iter_mut().enumerate() {
            let col = &self.w[c * d..(c + 1) * d];
            *tc = col.iter().zip(a.iter()).map(|(w, x)| w * x).sum();
        }
        let mut sum = 0.0;
        for (i, &ai) in a.iter().enumerate() {
            let mut p = ai;
            for (c, &tc) in t.iter().enumerate() {
                p -= self.w[c * d + i] * tc;
            }
            sum += p * p;
        }
        sum
    }

    /// Projected quadratic form `vᵀ H_c⁻¹ v` directly from `v` and an
    /// `SpdFactor` `L`.
    pub(crate) fn projected_quad_spd(&self, factor: &SpdFactor, v: &[f64]) -> f64 {
        let a = factor.solve_lower_cols(v, self.d, 1);
        self.projected_quad_from_whitened(&a)
    }

    /// Exact diagonal of `H_c⁻¹ = L⁻ᵀ (I − W Wᵀ) L⁻¹` from an `SpdFactor` `L`.
    pub(crate) fn projected_inverse_diagonal_spd(&self, factor: &SpdFactor) -> Vec<f64> {
        let d = self.d;
        let mut eye = vec![0.0; d * d];
        for i in 0..d {
            eye[i * d + i] = 1.0;
        }
        let linv = factor.solve_lower_cols(&eye, d, d);
        (0..d)
            .map(|i| self.projected_quad_from_whitened(&linv[i * d..(i + 1) * d]))
            .collect()
    }

    /// Exact diagonal of `H_c⁻¹` from a packed upper-triangular factor `R_chol`.
    pub(crate) fn projected_inverse_diagonal_upper(&self, r_packed: &[f64]) -> Vec<f64> {
        let d = self.d;
        let mut e = vec![0.0; d];
        let mut a = vec![0.0; d];
        (0..d)
            .map(|i| {
                e.fill(0.0);
                e[i] = 1.0;
                if solve_upper_t_packed(r_packed, d, &e, &mut a) {
                    self.projected_quad_from_whitened(&a)
                } else {
                    f64::INFINITY
                }
            })
            .collect()
    }

    /// Diagonal of the oblique constraint projector `Cᵀ M⁻¹ C H⁻¹` (trace `q`),
    /// used to compute each coefficient's constrained data share
    /// `support_coef_i = (1 − (Cᵀ M⁻¹ C H⁻¹)_ii − λ (H_c⁻¹)_ii).clamp(0, 1)`.
    pub(crate) fn oblique_proj_diag(&self) -> Vec<f64> {
        let d = self.d;
        let q = self.q;
        // Solve `L_m C̃ = D⁻¹ C` (`q × d`), column by column.
        let mut c_tilde = vec![0.0; q * d];
        for i in 0..d {
            for a in 0..q {
                let mut acc = self.c_mat[a * d + i] / self.d_scale[a];
                for b in 0..a {
                    acc -= self.l_m[a * q + b] * c_tilde[b * d + i];
                }
                c_tilde[a * d + i] = acc / self.l_m[a * q + a];
            }
        }
        // `(C̃ᵀ U_wᵀ)_ii = Σ_a C̃[a, i] * U_w[i, a]`.
        (0..d)
            .map(|i| {
                (0..q)
                    .map(|a| c_tilde[a * d + i] * self.u_w[a * d + i])
                    .sum()
            })
            .collect()
    }
}

/// Equilibrate `V` (`d × q`, column-major), strict-Cholesky-factorize
/// `M̂ = V̂ᵀ V̂ = L_m L_mᵀ`, check conditioning, and return `(d_scale, L_m, W)`
/// where `W = V̂ L_m⁻ᵀ` (`d × q`, column-major, orthonormal columns).
fn orthonormalize_whitened(
    v: &[f64],
    d: usize,
    q: usize,
) -> Option<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    let mut d_scale = vec![0.0; q];
    let mut v_hat = vec![0.0; d * q];
    for a in 0..q {
        let col = &v[a * d..(a + 1) * d];
        if !col.iter().all(|x| x.is_finite()) {
            return None;
        }
        let norm = col.iter().map(|x| x * x).sum::<f64>().sqrt();
        if !(norm > 0.0 && norm.is_finite()) {
            return None;
        }
        d_scale[a] = norm;
        let inv = norm.recip();
        for (dst, &src) in v_hat[a * d..(a + 1) * d].iter_mut().zip(col.iter()) {
            *dst = src * inv;
        }
    }
    let mut m_hat = vec![0.0; q * q];
    for a in 0..q {
        m_hat[a * q + a] = 1.0;
        let ca = &v_hat[a * d..(a + 1) * d];
        for b in 0..a {
            let cb = &v_hat[b * d..(b + 1) * d];
            let dot: f64 = ca.iter().zip(cb.iter()).map(|(x, y)| x * y).sum();
            m_hat[a * q + b] = dot;
            m_hat[b * q + a] = dot;
        }
    }
    // Strict lower-triangular Cholesky of unit-diagonal `M̂`, zero jitter.
    let mut l_m = vec![0.0; q * q];
    for j in 0..q {
        let mut s = m_hat[j * q + j];
        for t in 0..j {
            let ljt = l_m[j * q + t];
            s -= ljt * ljt;
        }
        // Unit-diagonal correlation matrix: a pivot `s = L_jj²` below `1e-12`
        // implies condition > 1e12 (near-singular `R H⁻¹ Rᵀ`).
        if !(s > 1e-12 && s.is_finite()) {
            return None;
        }
        let ljj = s.sqrt();
        l_m[j * q + j] = ljj;
        let inv = ljj.recip();
        for i in (j + 1)..q {
            let mut sum = m_hat[i * q + j];
            for t in 0..j {
                sum -= l_m[i * q + t] * l_m[j * q + t];
            }
            l_m[i * q + j] = sum * inv;
        }
    }
    // Check condition of `M̂` via `SpdFactor` as well for larger `q`.
    if q > 1 {
        let f = SpdFactor::of(&m_hat, q)?;
        if f.attempts() != 0 || f.carries_no_digit(&m_hat) {
            return None;
        }
    }
    // `W = V̂ L_m⁻ᵀ` <=> `L_m Wᵀ = V̂ᵀ` (forward sub on each row `i` of `V̂`).
    let mut w = vec![0.0; d * q];
    for i in 0..d {
        for a in 0..q {
            let mut acc = v_hat[a * d + i];
            for b in 0..a {
                acc -= l_m[a * q + b] * w[b * d + i];
            }
            w[a * d + i] = acc / l_m[a * q + a];
        }
    }
    if !w.iter().all(|x| x.is_finite()) {
        return None;
    }
    Some((d_scale, l_m, w))
}

/// Packed upper-triangular row start (same layout as `rls.rs::base`).
#[inline(always)]
const fn packed_base(k: usize, i: usize) -> usize {
    i * (2 * k - i - 1) / 2
}

/// Forward substitution `Rᵀ x = b` against packed upper-triangular `R`.
pub(crate) fn solve_upper_t_packed(r: &[f64], k: usize, b: &[f64], x: &mut [f64]) -> bool {
    for i in 0..k {
        let mut acc = b[i];
        let mut at = i;
        for (m, &xm) in x.iter().enumerate().take(i) {
            acc -= r[at] * xm;
            at += k - m - 1;
        }
        let d = r[at];
        if d > 0.0 {
            let v = acc / d;
            if !v.is_finite() {
                return false;
            }
            x[i] = v;
        } else if acc != 0.0 {
            return false;
        } else {
            x[i] = 0.0;
        }
    }
    true
}

/// Back substitution `R x = b` against packed upper-triangular `R`.
fn solve_upper_packed(r: &[f64], k: usize, b: &[f64], x: &mut [f64]) -> bool {
    for i in (0..k).rev() {
        let row_start = packed_base(k, i);
        let mut acc = b[i];
        for (j, &xj) in x.iter().enumerate().skip(i + 1) {
            acc -= r[row_start + j] * xj;
        }
        let d = r[row_start + i];
        if d > 0.0 {
            let v = acc / d;
            if !v.is_finite() {
                return false;
            }
            x[i] = v;
        } else if acc != 0.0 {
            return false;
        } else {
            x[i] = 0.0;
        }
    }
    true
}

/// Build the constraint matrix `C` (row-major `q × d`) over active system
/// columns `z` (accumulator indices) with column scales `s` and intercept
/// offset `off = usize::from(fit_intercept)` (so feature `f` lives at
/// accumulator slot `f + off`). Returns `None` if a constraint references no
/// active column or any scale is non-positive.
pub(crate) fn system_constraint_matrix(
    lc: &LinearConstraints,
    z: &[usize],
    s: &[f64],
    off: usize,
) -> Option<Vec<f64>> {
    let q = lc.matrix.len();
    let d = z.len();
    let mut c_mat = vec![0.0; q * d];
    for (a, row) in lc.matrix.iter().enumerate() {
        for (i2, (&zi, &si)) in z.iter().zip(s.iter()).enumerate() {
            if zi >= off {
                let f = zi - off;
                if !(si > 0.0 && si.is_finite()) {
                    return None;
                }
                c_mat[a * d + i2] = row[f] / si;
            }
        }
    }
    Some(c_mat)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Decay, EwRidge, EwRidgeCfg, OnlineModel, Rls, RlsCfg, TargetGaps};

    fn lcg(s: &mut u64) -> f64 {
        *s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((*s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }

    #[test]
    fn linear_constraints_validation_catches_every_bad_input() {
        let bad_cases = [
            LinearConstraints {
                matrix: vec![],
                rhs: vec![],
            },
            LinearConstraints {
                matrix: vec![vec![1.0, 1.0]],
                rhs: vec![1.0, 2.0],
            },
            LinearConstraints {
                matrix: vec![vec![1.0]],
                rhs: vec![1.0],
            },
            LinearConstraints {
                matrix: vec![vec![0.0, 0.0]],
                rhs: vec![0.0],
            },
            LinearConstraints {
                matrix: vec![vec![1.0, f64::NAN]],
                rhs: vec![1.0],
            },
            LinearConstraints {
                matrix: vec![vec![1.0, 0.0]],
                rhs: vec![f64::INFINITY],
            },
            LinearConstraints {
                matrix: vec![vec![1.0, 2.0], vec![2.0, 4.0]],
                rhs: vec![1.0, 2.0],
            },
            LinearConstraints {
                matrix: vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]],
                rhs: vec![1.0, 1.0, 2.0],
            },
        ];
        for lc in &bad_cases {
            assert!(
                lc.validate("ewridge", 2).is_err(),
                "expected err for {lc:?}"
            );
        }
        let ok = LinearConstraints {
            matrix: vec![vec![1.0, 1.0, 0.0], vec![0.0, 1.0, -1.0]],
            rhs: vec![1.0, 0.25],
        };
        assert!(ok.validate("ewridge", 3).is_ok());
    }

    #[test]
    fn constrained_rls_matches_ewridge_sum_scale_to_1e_12() {
        let lc = LinearConstraints {
            matrix: vec![vec![1.0, 1.0, 1.0], vec![1.0, -2.0, 0.0]],
            rhs: vec![1.0, 0.2],
        };
        for fit_intercept in [true, false] {
            let mut ew = EwRidge::new(EwRidgeCfg {
                n_features: 3,
                n_targets: 2,
                fit_intercept,
                decay: Decay::Halflife(35.0),
                ridge: vec![0.5],
                feature_sets: vec![],
                standardize: false,
                ridge_scale: true,
                coef_prior: None,
                session_shrink: None,
                long_half_life: None,
                min_weight: 0.0,
                solve_every: 0.0,
                max_rows_between_solves: 1,
                solve_share: None,
                gram_block_rows: 0,
                target_gaps: TargetGaps::OwnRows,
                window: None,
                window_every: None,
                max_rows_between_snapshots: None,
                constraints: Some(lc.clone()),
                constraint: None,
            })
            .unwrap();
            ew.set_keep_factor(true);

            let mut rls = Rls::new(RlsCfg {
                n_features: 3,
                n_targets: 2,
                fit_intercept,
                decay: Decay::Halflife(35.0),
                delta: 0.5,
                coef_prior: None,
                min_weight: 0.0,
                constraints: Some(lc.clone()),
            })
            .unwrap();

            let mut s = 123u64;
            for i in 0..80 {
                let x = [
                    lcg(&mut s) * 2.0 + 1.0,
                    lcg(&mut s) * 1.5,
                    lcg(&mut s) - 0.5,
                ];
                let y0 = 0.4 + 0.5 * x[0] + 0.15 * x[1] + 0.35 * x[2] + 0.1 * lcg(&mut s);
                let y1 = -0.2 + 0.3 * x[0] + 0.05 * x[1] + 0.65 * x[2] + 0.1 * lcg(&mut s);
                let d = if i == 0 {
                    0.0
                } else {
                    0.5 + (i % 3) as f64 * 0.5
                };
                let w = 0.5 + (lcg(&mut s) + 1.0);
                let pe = ew.step(&x, &[Some(y0), Some(y1)], d, w);
                let pr = rls.step(&x, &[Some(y0), Some(y1)], d, w);
                if i >= 10 {
                    for (a, b) in pe.pred.iter().zip(&pr.pred) {
                        assert!((a - b).abs() < 1e-12, "row {i}: {a} vs {b}");
                    }
                    // Verify constraint satisfaction R β_slopes = r on both models.
                    let off = usize::from(fit_intercept);
                    for (ce, cr) in ew.coefficients().unwrap().iter().zip(rls.coefficients()) {
                        for (a, b) in ce.iter().zip(cr) {
                            assert!((a - b).abs() < 1e-12, "coef row {i}: {a} vs {b}");
                        }
                        let b = &ce[off..];
                        assert!((b[0] + b[1] + b[2] - 1.0).abs() < 1e-12);
                        assert!((b[0] - 2.0 * b[1] - 0.2).abs() < 1e-12);
                    }
                    // Verify row error inflation and coef_variance parity.
                    let (mut ie, mut ir) = (Vec::new(), Vec::new());
                    ew.row_error_inflation_into(&x, 1.0, &mut ie);
                    rls.row_error_inflation_into(&x, 1.0, &mut ir);
                    for (a, b) in ie.iter().zip(&ir) {
                        assert!((a - b).abs() < 1e-12, "infl row {i}: {a} vs {b}");
                    }
                    let (
                        Some(crate::CoefVariance::PerNoise(ve)),
                        Some(crate::CoefVariance::PerNoise(vr)),
                    ) = (ew.coef_variance(), rls.coef_variance())
                    else {
                        panic!("expected PerNoise coef_variance");
                    };
                    for (se, sr) in ve.iter().zip(&vr) {
                        for (a, b) in se.iter().zip(sr) {
                            assert!((a - b).abs() < 1e-12, "var row {i}: {a} vs {b}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn constrained_ewridge_handles_standardize_and_singular_constraint() {
        let lc = LinearConstraints {
            matrix: vec![vec![2.0, -1.0]],
            rhs: vec![0.5],
        };
        for (fit_intercept, standardize) in
            [(true, false), (true, true), (false, false), (false, true)]
        {
            let mut ew = EwRidge::new(EwRidgeCfg {
                n_features: 2,
                n_targets: 1,
                fit_intercept,
                decay: Decay::Halflife(f64::INFINITY),
                ridge: vec![1e-6],
                feature_sets: vec![],
                standardize,
                ridge_scale: false,
                coef_prior: None,
                session_shrink: None,
                long_half_life: None,
                min_weight: 0.0,
                solve_every: 0.0,
                max_rows_between_solves: 1,
                solve_share: None,
                gram_block_rows: 0,
                target_gaps: TargetGaps::OwnRows,
                window: None,
                window_every: None,
                max_rows_between_snapshots: None,
                constraints: Some(lc.clone()),
                constraint: None,
            })
            .unwrap();
            ew.set_keep_factor(true);
            let mut s = 77u64;
            for _ in 0..60 {
                let x = [lcg(&mut s) * 10.0 + 3.0, lcg(&mut s) * 0.2 - 1.0];
                let y = 1.5 + 0.8 * x[0] + 1.1 * x[1] + 0.05 * lcg(&mut s);
                ew.step(&x, &[Some(y)], 1.0, 1.0);
            }
            let off = usize::from(fit_intercept);
            let b = &ew.coefficients().unwrap()[0][off..];
            assert!(
                (2.0 * b[0] - b[1] - 0.5).abs() < 1e-11,
                "intercept={fit_intercept} std={standardize}: {b:?}"
            );
            // Projected variance along the constrained direction must be ~0 (v = [2, -1]).
            // Check every entry is finite and non-negative.
            let Some(crate::CoefVariance::PerNoise(v)) = ew.coef_variance() else {
                panic!("missing coef_variance");
            };
            assert!(v[0].iter().all(|x| x.is_finite() && *x >= 0.0));
        }

        // Constant feature under standardize=true drops column 0 -> constraint on column 0
        // makes R A R' singular -> fit withheld cleanly.
        let mut ew_sing = EwRidge::new(EwRidgeCfg {
            n_features: 2,
            n_targets: 1,
            fit_intercept: true,
            decay: Decay::Halflife(f64::INFINITY),
            ridge: vec![1e-6],
            feature_sets: vec![],
            standardize: true,
            ridge_scale: false,
            coef_prior: None,
            session_shrink: None,
            long_half_life: None,
            min_weight: 0.0,
            solve_every: 0.0,
            max_rows_between_solves: 1,
            solve_share: None,
            gram_block_rows: 0,
            target_gaps: TargetGaps::OwnRows,
            window: None,
            window_every: None,
            max_rows_between_snapshots: None,
            constraints: Some(LinearConstraints {
                matrix: vec![vec![1.0, 0.0]],
                rhs: vec![1.0],
            }),
            constraint: None,
        })
        .unwrap();
        let mut s = 99u64;
        for _ in 0..30 {
            let x = [5.0, lcg(&mut s)];
            let y = 2.0 * x[1] + 0.1 * lcg(&mut s);
            ew_sing.step(&x, &[Some(y)], 1.0, 1.0);
        }
        assert!(
            ew_sing
                .coefficients()
                .is_none_or(|cs| cs[0].iter().all(|c| c.is_nan()))
        );
        assert!(ew_sing.solve_failures > 0);
    }
}
