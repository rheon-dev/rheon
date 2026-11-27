//! Packed row-major lower-triangular linear algebra and ALD-KRLS state.
//!
//! In ALD mode, the dictionary `D` holds `m <= max_atoms` atoms and its kernel
//! matrix is kept as its exact lower-triangular Cholesky factor `L_D`
//! (`K_D = L_D L_D^T`). For every row `x`, `z = L_D^{-1} k(D, x)` is both the
//! ALD projection and the whitened feature vector `psi(x)`:
//!
//! - `delta = k(x, x) - ||z||^2`: admit when `m < max_atoms`, `k(x, x) > 0`,
//!   and `delta > max(ald_threshold, ALD_PIVOT_FLOOR) * k(x, x)`.
//! - Extending `L_D` appends `[z^T, sqrt(delta)]` in `O(m)` with no `1/delta`
//!   division.
//! - Under Engel's projection approximation for past rows, their new coordinate
//!   is identically zero (`psi'_s = [psi_s; 0]`), so the whitened Gram `S` (or
//!   its Cholesky factor `L_s`) simply appends a zero row/col (plus `sqrt(r)`
//!   on the diagonal of `L_s`), with no `1/(r * delta)` division.
//! - Without decay (`!decays()`), `mat` keeps the lower-triangular Cholesky
//!   factor `L_s` of `S + r I` and `rhs` keeps `u = L_s^{-1} c`, updated in
//!   `O(m^2)` per row via orthogonal Givens rotations for both `fit_intercept =
//!   false` and `fit_intercept = true` (using the exact weighted Welford
//!   positive rank-1 factor `W_old * w / W_new`).
//! - With decay (`decays()`), `mat` keeps the whitened Gram `S` (sum scale)
//!   and `rhs` keeps `c`, decayed by `lam` each row and solved by packed
//!   Cholesky of `S + r I` (`O(m^3 / 3)`).

use serde::{Deserialize, Serialize};

use crate::comp;

use super::KrlsCfg;
use super::kernel::KernelEval;

/// Numerical floor on the relative ALD residual `delta / k(x, x)` so even
/// `ald_threshold = 0.0` never admits rank-deficient rounding residuals.
pub const ALD_PIVOT_FLOOR: f64 = 1e-12;

/// Number of entries in a packed lower-triangular `m x m` matrix.
#[inline]
pub(super) const fn tri_len(m: usize) -> usize {
    m * (m + 1) / 2
}

/// Starting index of row `i` in a packed row-major lower-triangular matrix.
#[inline]
pub(super) const fn tri_row(i: usize) -> usize {
    i * (i + 1) / 2
}

/// Forward substitution `L z = b` for a packed row-major lower-triangular
/// `m x m` matrix `l`, writing `z` into `out`.
pub(super) fn forward_solve_lower(l: &[f64], m: usize, b: &[f64], out: &mut Vec<f64>) {
    out.clear();
    out.resize(m, 0.0);
    for i in 0..m {
        let row = &l[tri_row(i)..tri_row(i) + i + 1];
        let mut s = b[i];
        for (lij, zj) in row[..i].iter().zip(out[..i].iter()) {
            s -= lij * zj;
        }
        out[i] = s / row[i];
    }
}

/// Back substitution `L^T x = b` for a packed row-major lower-triangular
/// `m x m` matrix `l`, where `b` and `x` are stored in place in `x`.
pub(super) fn back_solve_lower_t_in_place(l: &[f64], m: usize, x: &mut [f64]) {
    for i in (0..m).rev() {
        let row = &l[tri_row(i)..tri_row(i) + i + 1];
        let xi = x[i] / row[i];
        x[i] = xi;
        for (xj, &lij) in x[..i].iter_mut().zip(row[..i].iter()) {
            *xj -= lij * xi;
        }
    }
}

/// Factor `S + ridge * I` (where `s` is packed row-major lower-triangular)
/// into its lower-triangular Cholesky factor `out_l`. Returns `false` if any
/// pivot is non-positive or non-finite.
pub(super) fn cholesky_lower_ridge(s: &[f64], ridge: f64, m: usize, out_l: &mut Vec<f64>) -> bool {
    out_l.clear();
    out_l.resize(tri_len(m), 0.0);
    for i in 0..m {
        let row_i = tri_row(i);
        for j in 0..i {
            let row_j = tri_row(j);
            let mut sum = s[row_i + j];
            for k in 0..j {
                sum -= out_l[row_i + k] * out_l[row_j + k];
            }
            out_l[row_i + j] = sum / out_l[row_j + j];
        }
        let mut diag = s[row_i + i] + ridge;
        for k in 0..i {
            let v = out_l[row_i + k];
            diag -= v * v;
        }
        if !(diag > 0.0 && diag.is_finite()) {
            return false;
        }
        out_l[row_i + i] = diag.sqrt();
    }
    true
}

/// Rank-1 Cholesky update of packed row-major lower-triangular `L` and
/// right-hand sides `u` (`T` targets, stored as `t * m + i`):
///
/// ```text
/// L' L'^T = L L^T + v v^T
/// L' u'_t = L u_t + v y_t
/// ```
///
/// Mutates `v` and `yv` in place as scratch. Uses orthogonal Givens rotations
/// (unconditionally stable, no subtraction from pivots).
pub(super) fn givens_rank1_update(
    l: &mut [f64],
    m: usize,
    v: &mut [f64],
    u: &mut [f64],
    yv: &mut [f64],
    n_targets: usize,
) {
    for i in 0..m {
        let vi = v[i];
        if vi == 0.0 {
            continue;
        }
        let diag_idx = tri_row(i) + i;
        let lii = l[diag_idx];
        let sq = lii * lii + vi * vi;
        let mut rho = sq.sqrt();
        if !(sq >= f64::MIN_POSITIVE && sq.is_finite()) {
            rho = libm::hypot(lii, vi);
            if !(rho > 0.0 && rho.is_finite()) {
                continue;
            }
        }
        let c = lii / rho;
        let s = vi / rho;
        l[diag_idx] = rho;
        v[i] = 0.0;
        for (j, vj) in v.iter_mut().enumerate().take(m).skip(i + 1) {
            let idx = tri_row(j) + i;
            let a = l[idx];
            let b = *vj;
            l[idx] = c * a + s * b;
            *vj = c * b - s * a;
        }
        for (t, yvt) in yv.iter_mut().enumerate().take(n_targets) {
            let ui = u[t * m + i];
            let yb = *yvt;
            u[t * m + i] = c * ui + s * yb;
            *yvt = c * yb - s * ui;
        }
    }
}

/// Scratch buffers reused across rows in ALD mode so `step_into` allocates
/// nothing once the dictionary is full.
#[derive(Debug, Clone, Default)]
pub(super) struct AldScratch {
    pub kv: Vec<f64>,
    pub z: Vec<f64>,
    pub v: Vec<f64>,
    pub yv: Vec<f64>,
    pub chol: Vec<f64>,
    pub tmp: Vec<f64>,
}

fn default_one() -> f64 {
    1.0
}

fn is_one(v: &f64) -> bool {
    *v == 1.0
}

/// Serialized state of ALD-KRLS.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(super) struct AldState {
    /// Atoms `D`, row-major `m x k`.
    pub atoms: Vec<f64>,
    /// Packed lower-triangular Cholesky factor `L_D` of `K_D`, length `m(m+1)/2`.
    pub l_d: Vec<f64>,
    /// Without decay: packed lower-triangular Cholesky `L_s` of `S + r I`.
    /// With decay: packed lower-triangular whitened Gram `S` (sum scale).
    pub mat: Vec<f64>,
    /// Without decay: `u = L_s^{-1} c`, flat `T x m` (target-major).
    /// With decay: whitened RHS `c` (sum scale), flat `T x m`.
    pub rhs: Vec<f64>,
    /// Compensated EW mean of `psi` (`hi` and `lo`, each length `m` when
    /// `fit_intercept`, empty otherwise).
    pub psi_bar: Vec<f64>,
    pub psi_bar_lo: Vec<f64>,
    /// Compensated EW mean of `y` (`hi` and `lo`, each length `T` when
    /// `fit_intercept`, empty otherwise).
    pub y_bar: Vec<f64>,
    pub y_bar_lo: Vec<f64>,
    /// Deferred decay factor accumulated across non-learning rows so `mat` and
    /// `rhs` remain consistent with the cached solution across save/restore
    /// until the next learned row arrives (`1.0` otherwise).
    #[serde(default = "default_one", skip_serializing_if = "is_one")]
    pub pending_decay: f64,
    /// Cached whitened solution `theta`, flat `T x m` (target-major). Never state.
    #[serde(skip)]
    pub theta: Vec<f64>,
    /// Cached intercept `b`, length `T` (zeros when `!fit_intercept`). Never state.
    #[serde(skip)]
    pub intercept: Vec<f64>,
}

impl AldState {
    pub fn new(cfg: &KrlsCfg) -> Self {
        let t = cfg.n_targets;
        let (y_bar, y_bar_lo) = if cfg.fit_intercept {
            (vec![0.0; t], vec![0.0; t])
        } else {
            (Vec::new(), Vec::new())
        };
        Self {
            atoms: Vec::new(),
            l_d: Vec::new(),
            mat: Vec::new(),
            rhs: Vec::new(),
            psi_bar: Vec::new(),
            psi_bar_lo: Vec::new(),
            y_bar,
            y_bar_lo,
            pending_decay: 1.0,
            theta: Vec::new(),
            intercept: vec![0.0; t],
        }
    }

    #[inline]
    pub fn n_atoms(&self, k: usize) -> usize {
        self.atoms.len() / k
    }

    /// Predict all targets into `pred` (length `T`) given `z = L_D^{-1} k(D, x)`.
    pub fn predict_from_z(
        &self,
        z: &[f64],
        w_target: &[f64],
        min_weight: f64,
        n_targets: usize,
        pred: &mut Vec<f64>,
    ) {
        let m = z.len();
        pred.clear();
        pred.resize(n_targets, f64::NAN);
        if m == 0 || !z.iter().all(|v| v.is_finite()) {
            return;
        }
        for t in 0..n_targets {
            if w_target[t] >= min_weight {
                let th = &self.theta[t * m..(t + 1) * m];
                let mut dot = 0.0;
                for (&zi, &ti) in z.iter().zip(th) {
                    dot += zi * ti;
                }
                let val = self.intercept[t] + dot;
                if val.is_finite() {
                    pred[t] = val;
                }
            }
        }
    }

    /// Recompute `theta` and `intercept` from `(mat, rhs, psi_bar, y_bar)`.
    pub fn solve(&mut self, cfg: &KrlsCfg, chol_buf: &mut Vec<f64>, tmp_buf: &mut Vec<f64>) {
        let m = self.n_atoms(cfg.n_features);
        let t_cnt = cfg.n_targets;
        self.theta.clear();
        self.theta.resize(t_cnt * m, 0.0);
        self.intercept.clear();
        self.intercept.resize(t_cnt, 0.0);
        if m == 0 {
            return;
        }
        if !cfg.decays() {
            // `mat` is `L_s`, `rhs` is `u = L_s^{-1} c`: `L_s^T theta = u`.
            self.theta.copy_from_slice(&self.rhs);
            for t in 0..t_cnt {
                back_solve_lower_t_in_place(&self.mat, m, &mut self.theta[t * m..(t + 1) * m]);
            }
        } else if cholesky_lower_ridge(&self.mat, cfg.ridge, m, chol_buf) {
            // `mat` is `S`, `rhs` is `c`: solve `L L^T theta = c`.
            for t in 0..t_cnt {
                forward_solve_lower(chol_buf, m, &self.rhs[t * m..(t + 1) * m], tmp_buf);
                back_solve_lower_t_in_place(chol_buf, m, tmp_buf);
                self.theta[t * m..(t + 1) * m].copy_from_slice(tmp_buf);
            }
        }
        if cfg.fit_intercept {
            for t in 0..t_cnt {
                let th = &self.theta[t * m..(t + 1) * m];
                let mut dot = 0.0;
                for ((&hi, &lo), &ti) in self
                    .psi_bar
                    .iter()
                    .zip(self.psi_bar_lo.iter())
                    .zip(th.iter())
                {
                    dot += (hi + lo) * ti;
                }
                self.intercept[t] = (self.y_bar[t] + self.y_bar_lo[t]) - dot;
            }
        }
    }

    /// Defer decay across a null-target, zero-weight, or overflow-guarded row
    /// when `decays()` is true, leaving `(mat, rhs, theta, intercept)` unchanged
    /// until the next learned row arrives so save/restore reproduces the cached
    /// prediction to the bit. Never divides by `lam`; cleanly handles `lam == 0`
    /// and subnormals.
    pub fn decay_only(&mut self, cfg: &KrlsCfg, lam: f64) {
        if !cfg.decays() || lam == 1.0 {
            return;
        }
        self.pending_decay *= lam;
    }

    /// Learn one row `(x, y)` of positive weight `w` with decay `lam` (and
    /// precomputed `z = L_D^{-1} k(D, x)`, `knn = k(x, x)` in `sc`).
    #[allow(clippy::too_many_arguments)]
    pub fn learn(
        &mut self,
        cfg: &KrlsCfg,
        kern: KernelEval,
        x: &[f64],
        y: &[Option<f64>],
        w: f64,
        lam: f64,
        w_old_learned: f64,
        sc: &mut AldScratch,
    ) -> bool {
        let k = cfg.n_features;
        let t_cnt = cfg.n_targets;
        let mut m = self.n_atoms(k);
        let knn = kern.eval_self(x);
        if !(knn.is_finite() && knn >= 0.0) {
            return false;
        }
        if !sc.z.iter().all(|v| v.is_finite()) {
            return false;
        }
        let z_norm2: f64 = sc.z.iter().map(|v| v * v).sum();
        if !z_norm2.is_finite() {
            return false;
        }
        let delta = knn - z_norm2;
        let thresh = cfg.ald_threshold_or_default().max(ALD_PIVOT_FLOOR);
        let admit = m < cfg.max_atoms_or_default() && knn > 0.0 && delta > thresh * knn;

        // Check finiteness of the whitened vector and update factor before
        // mutating any state (protects against overflow at INPUT_BOUND under
        // high-degree polynomial kernels).
        let sqrt_delta = if admit { delta.sqrt() } else { 0.0 };
        if admit && !(sqrt_delta > 0.0 && sqrt_delta.is_finite()) {
            return false;
        }

        // Build `psi` in `sc.v` (length `m_new`).
        let m_new = m + usize::from(admit);
        if m_new == 0 {
            // Cannot learn before any atom exists (e.g. `linear` kernel at x = 0).
            return false;
        }
        sc.v.clear();
        sc.v.extend_from_slice(&sc.z);
        if admit {
            sc.v.push(sqrt_delta);
        }

        // Verify that the update terms stay finite before mutating state.
        let w_decayed = lam * w_old_learned;
        let w_new = w_decayed + w;
        if !(w_new > 0.0 && w_new.is_finite()) {
            return false;
        }
        let max_psi = sc.v.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
        let max_y = y
            .iter()
            .filter_map(|o| *o)
            .fold(0.0_f64, |acc, v| acc.max(v.abs()));
        if !(w * max_psi * max_psi).is_finite() || !(w * max_psi * max_y).is_finite() {
            return false;
        }

        // Commit admission to `D`, `L_D`, `mat`, `rhs`, `psi_bar`.
        if admit {
            self.atoms.extend_from_slice(x);
            self.l_d.extend_from_slice(&sc.z);
            self.l_d.push(sqrt_delta);

            // Pad `mat` with one new row of length `m + 1`:
            // undecayed `L_s` gets `[0, ..., 0, sqrt(ridge)]`;
            // decayed `S` gets `[0, ..., 0, 0]`.
            self.mat.resize(tri_len(m_new), 0.0);
            if !cfg.decays() {
                self.mat[tri_row(m) + m] = cfg.ridge.sqrt();
            }

            // Pad `rhs` (stored target-major `T x m`) from `T x m` to `T x (m + 1)`.
            let mut new_rhs = vec![0.0; t_cnt * m_new];
            for t in 0..t_cnt {
                new_rhs[t * m_new..t * m_new + m].copy_from_slice(&self.rhs[t * m..(t + 1) * m]);
            }
            self.rhs = new_rhs;

            if cfg.fit_intercept {
                self.psi_bar.push(0.0);
                self.psi_bar_lo.push(0.0);
            }
            m = m_new;
        }

        // Apply the update to `(mat, rhs)` and (if `fit_intercept`) compensated means.
        let eff_lam = if cfg.decays() {
            let d = lam * self.pending_decay;
            self.pending_decay = 1.0;
            d
        } else {
            1.0
        };
        if cfg.fit_intercept {
            let a = w_decayed / w_new;
            let b = w / w_new;
            let alpha_w = a * w;

            // Replace `sc.v` with `dpsi = psi - psi_bar`, and build `dy = y - y_bar` in `sc.yv`.
            sc.yv.clear();
            for (t, yj) in y.iter().enumerate().take(t_cnt) {
                let yv = yj.expect("all targets present");
                sc.yv.push(comp::dev(yv, self.y_bar[t], self.y_bar_lo[t]));
            }
            for i in 0..m {
                let pi = sc.v[i];
                let d = comp::dev(pi, self.psi_bar[i], self.psi_bar_lo[i]);
                comp::step_toward(&mut self.psi_bar[i], &mut self.psi_bar_lo[i], pi, d, a, b);
                sc.v[i] = d;
            }
            for (t, yj) in y.iter().enumerate().take(t_cnt) {
                let yv = yj.expect("all targets present");
                let dy = sc.yv[t];
                comp::step_toward(&mut self.y_bar[t], &mut self.y_bar_lo[t], yv, dy, a, b);
            }

            if !cfg.decays() {
                // Positive rank-1 Givens update with `sqrt(alpha_w) * [dpsi, dy]`.
                if alpha_w > 0.0 {
                    let s_w = alpha_w.sqrt();
                    for v in &mut sc.v {
                        *v *= s_w;
                    }
                    for v in &mut sc.yv {
                        *v *= s_w;
                    }
                    givens_rank1_update(
                        &mut self.mat,
                        m,
                        &mut sc.v,
                        &mut self.rhs,
                        &mut sc.yv,
                        t_cnt,
                    );
                }
            } else {
                // Decayed sum-scale centered Gram: `S = eff_lam * S + alpha_w * dpsi dpsi^T`,
                // `c = eff_lam * c + alpha_w * dpsi dy^T`.
                for i in 0..m {
                    let row = &mut self.mat[tri_row(i)..tri_row(i) + i + 1];
                    let a_di = alpha_w * sc.v[i];
                    for (entry, &dj) in row.iter_mut().zip(&sc.v[..=i]) {
                        *entry = eff_lam * *entry + a_di * dj;
                    }
                }
                for (t, &dy) in sc.yv.iter().enumerate().take(t_cnt) {
                    let a_dy = alpha_w * dy;
                    let c_t = &mut self.rhs[t * m..(t + 1) * m];
                    for (entry, &di) in c_t.iter_mut().zip(&sc.v) {
                        *entry = eff_lam * *entry + a_dy * di;
                    }
                }
            }
        } else if !cfg.decays() {
            // Undecayed, no intercept: Givens rank-1 update with `sqrt(w) * [psi, y]`.
            let sw = w.sqrt();
            for v in &mut sc.v {
                *v *= sw;
            }
            sc.yv.clear();
            for yj in y.iter().take(t_cnt) {
                sc.yv.push(sw * yj.expect("all targets present"));
            }
            givens_rank1_update(
                &mut self.mat,
                m,
                &mut sc.v,
                &mut self.rhs,
                &mut sc.yv,
                t_cnt,
            );
        } else {
            // Decayed, no intercept: `S = eff_lam * S + w * psi psi^T`, `c = eff_lam * c + w * psi y^T`.
            for i in 0..m {
                let row = &mut self.mat[tri_row(i)..tri_row(i) + i + 1];
                let w_pi = w * sc.v[i];
                for (entry, &pj) in row.iter_mut().zip(&sc.v[..=i]) {
                    *entry = eff_lam * *entry + w_pi * pj;
                }
            }
            for (t, yj) in y.iter().enumerate().take(t_cnt) {
                let w_y = w * yj.expect("all targets present");
                let c_t = &mut self.rhs[t * m..(t + 1) * m];
                for (entry, &pi) in c_t.iter_mut().zip(&sc.v) {
                    *entry = eff_lam * *entry + w_y * pi;
                }
            }
        }

        self.solve(cfg, &mut sc.chol, &mut sc.tmp);
        true
    }

    /// Validate deserialized state shape and numerical invariants (`STATE-3`).
    pub fn has_shape(&self, cfg: &KrlsCfg) -> bool {
        let k = cfg.n_features;
        let t = cfg.n_targets;
        if !self.atoms.len().is_multiple_of(k) {
            return false;
        }
        let m = self.atoms.len() / k;
        if m > cfg.max_atoms_or_default() {
            return false;
        }
        let tri = tri_len(m);
        if self.l_d.len() != tri || self.mat.len() != tri || self.rhs.len() != t * m {
            return false;
        }
        if cfg.fit_intercept {
            if self.psi_bar.len() != m
                || self.psi_bar_lo.len() != m
                || self.y_bar.len() != t
                || self.y_bar_lo.len() != t
            {
                return false;
            }
        } else if !self.psi_bar.is_empty()
            || !self.psi_bar_lo.is_empty()
            || !self.y_bar.is_empty()
            || !self.y_bar_lo.is_empty()
        {
            return false;
        }
        let all_finite = self.atoms.iter().all(|v| v.is_finite())
            && self.l_d.iter().all(|v| v.is_finite())
            && self.mat.iter().all(|v| v.is_finite())
            && self.rhs.iter().all(|v| v.is_finite())
            && self.psi_bar.iter().all(|v| v.is_finite())
            && self.psi_bar_lo.iter().all(|v| v.is_finite())
            && self.y_bar.iter().all(|v| v.is_finite())
            && self.y_bar_lo.iter().all(|v| v.is_finite())
            && self.pending_decay.is_finite()
            && (0.0..=1.0).contains(&self.pending_decay)
            && (cfg.decays() || self.pending_decay == 1.0);
        if !all_finite {
            return false;
        }
        // Strictly positive Cholesky diagonals in `L_D` (and in `L_s` when undecayed).
        for i in 0..m {
            let d = tri_row(i) + i;
            if self.l_d[d] <= 0.0 {
                return false;
            }
            if !cfg.decays() && self.mat[d] <= 0.0 {
                return false;
            }
        }
        true
    }
}
