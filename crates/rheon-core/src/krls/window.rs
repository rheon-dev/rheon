//! Sliding-window KRLS state with Givens-updated Cholesky factorization.
//!
//! Every learned row inside the window is an atom, and the system
//! `A = K_W + ridge * diag(1 / w)` is held as its packed row-major
//! lower-triangular Cholesky factor `L_W` (`A = L_W L_W^T`):
//!
//! - Dropping the oldest row removes row/col 0 via `A_11 = L_11 L_11^T +
//!   l_10 l_10^T`, a *positive* rank-1 Cholesky update computed by `W - 1`
//!   orthogonal Givens rotations (`O(W^2)`, unconditionally positive definite
//!   with zero subtraction drift).
//! - Adding a new row `(x, y, w)` solves `L_W z = k(X_W, x)`, computes
//!   `s = k(x, x) + ridge / w - ||z||^2 >= ridge / w > 0`, and appends
//!   `[z^T, sqrt(s)]` (`O(W^2 / 2)`).
//! - Solving `L_W L_W^T v_y = y_W` gives `alpha = v_y` (without intercept),
//!   matches `sklearn.kernel_ridge.KernelRidge(alpha=ridge, sample_weight=w)`.
//!   With `fit_intercept = true`, solving `L_W L_W^T v_1 = 1` gives
//!   `b = (1^T v_y) / (1^T v_1)` and `alpha = v_y - v_1 b^T`, matching
//!   `Nystroem + Ridge(alpha=ridge, fit_intercept=True, sample_weight=w)`.

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

use crate::Stamp;

use super::KrlsCfg;
use super::ald::{back_solve_lower_t_in_place, forward_solve_lower, tri_len, tri_row};
use super::kernel::KernelEval;

/// Scratch buffers reused across rows in sliding-window mode.
#[derive(Debug, Clone, Default)]
pub(super) struct WinScratch {
    pub kv: Vec<f64>,
    pub z: Vec<f64>,
    pub rhs: Vec<f64>,
    pub v1: Vec<f64>,
    pub drop_v: Vec<f64>,
}

/// Serialized state of sliding-window KRLS.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(super) struct WindowState {
    /// Window feature rows, oldest first, flat `len * k`.
    pub x: VecDeque<f64>,
    /// Window target rows, oldest first, flat `len * T`.
    pub y: VecDeque<f64>,
    /// Window positive row weights, oldest first, length `len`.
    pub w: VecDeque<f64>,
    /// Row stamps under `window_clock`, oldest first, length `len` (empty under
    /// `window_rows`).
    pub stamp: VecDeque<Stamp>,
    /// Packed row-major lower-triangular Cholesky factor `L_W` of
    /// `K_W + ridge * diag(1 / w)`, length `len * (len + 1) / 2`.
    pub l_w: Vec<f64>,
    /// Cached dual coefficients `alpha`, flat `T x len` (target-major). Never state.
    #[serde(skip)]
    pub alpha: Vec<f64>,
    /// Cached intercept `b`, length `T` (zeros when `!fit_intercept`). Never state.
    #[serde(skip)]
    pub intercept: Vec<f64>,
}

impl WindowState {
    pub fn new(n_targets: usize) -> Self {
        Self {
            x: VecDeque::new(),
            y: VecDeque::new(),
            w: VecDeque::new(),
            stamp: VecDeque::new(),
            l_w: Vec::new(),
            alpha: Vec::new(),
            intercept: vec![0.0; n_targets],
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.w.len()
    }

    /// Exact sum of weights currently inside the window.
    #[inline]
    pub fn weight_sum(&self) -> f64 {
        self.w.iter().copied().sum()
    }

    /// Count how many leading rows have left the clock window at `now`
    /// (read-only, used to check `HeldCap` atomically before any mutation).
    pub fn expired_count(&self, cfg: &KrlsCfg, now: Stamp) -> usize {
        let Some(span) = cfg.window_clock else {
            return 0;
        };
        self.stamp
            .iter()
            .take_while(|&&s| cfg.closed.has_left(now, s, span))
            .count()
    }

    /// Predict all targets into `pred` (length `T`) at feature vector `x`.
    #[allow(clippy::too_many_arguments)]
    pub fn predict_into(
        &self,
        kern: KernelEval,
        x: &[f64],
        w_target: &[f64],
        min_weight: f64,
        k: usize,
        n_targets: usize,
        kv_buf: &mut Vec<f64>,
        pred: &mut Vec<f64>,
    ) {
        let len = self.len();
        pred.clear();
        pred.resize(n_targets, f64::NAN);
        if len == 0 {
            return;
        }
        kern.eval_deque(&self.x, len, k, x, kv_buf);
        if !kv_buf.iter().all(|v| v.is_finite()) {
            return;
        }
        for t in 0..n_targets {
            if w_target[t] >= min_weight {
                let al = &self.alpha[t * len..(t + 1) * len];
                let mut dot = 0.0;
                for (&kvi, &ai) in kv_buf.iter().zip(al) {
                    dot += kvi * ai;
                }
                let val = self.intercept[t] + dot;
                if val.is_finite() {
                    pred[t] = val;
                }
            }
        }
    }

    /// Drop the oldest row from the ring and update `L_W` by orthogonal Givens
    /// rotations (`L_11 L_11^T + l_10 l_10^T`).
    pub fn drop_oldest(&mut self, k: usize, n_targets: usize, drop_v: &mut Vec<f64>) {
        let m = self.len();
        if m == 0 {
            return;
        }
        self.x.drain(..k);
        self.y.drain(..n_targets);
        self.w.pop_front();
        if !self.stamp.is_empty() {
            self.stamp.pop_front();
        }
        let nm = m - 1;
        if nm == 0 {
            self.l_w.clear();
            return;
        }
        // Extract column 0 of rows `1..m` into `drop_v`, and compact `L[1..m, 1..m]`
        // in place into the leading `tri_len(nm)` entries of `self.l_w`.
        drop_v.clear();
        drop_v.reserve(nm);
        for i in 0..nm {
            let old_row = tri_row(i + 1);
            drop_v.push(self.l_w[old_row]);
            let dst = tri_row(i);
            self.l_w.copy_within(old_row + 1..old_row + 1 + i + 1, dst);
        }
        self.l_w.truncate(tri_len(nm));

        // Positive rank-1 Cholesky update: `L <- chol(L L^T + drop_v drop_v^T)`.
        for i in 0..nm {
            let vi = drop_v[i];
            if vi == 0.0 {
                continue;
            }
            let diag_idx = tri_row(i) + i;
            let lii = self.l_w[diag_idx];
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
            self.l_w[diag_idx] = rho;
            for (j, vj) in drop_v.iter_mut().enumerate().take(nm).skip(i + 1) {
                let idx = tri_row(j) + i;
                let a = self.l_w[idx];
                let b = *vj;
                self.l_w[idx] = c * a + s * b;
                *vj = c * b - s * a;
            }
        }
    }

    /// Add one learned row `(x, y, w)` (and optional `stamp`) to the window.
    /// Returns `false` if the kernel evaluation or Schur complement overflows.
    #[allow(clippy::too_many_arguments)]
    pub fn push_row(
        &mut self,
        cfg: &KrlsCfg,
        kern: KernelEval,
        x: &[f64],
        y: &[Option<f64>],
        w: f64,
        stamp: Option<Stamp>,
        sc: &mut WinScratch,
    ) -> bool {
        let k = cfg.n_features;
        let t_cnt = cfg.n_targets;
        let m = self.len();
        let knn = kern.eval_self(x);
        let reg = cfg.ridge / w;
        if !(knn.is_finite() && reg.is_finite()) {
            return false;
        }
        kern.eval_deque(&self.x, m, k, x, &mut sc.kv);
        if !sc.kv.iter().all(|v| v.is_finite()) {
            return false;
        }
        forward_solve_lower(&self.l_w, m, &sc.kv, &mut sc.z);
        if !sc.z.iter().all(|v| v.is_finite()) {
            return false;
        }
        let z_norm2: f64 = sc.z.iter().map(|v| v * v).sum();
        let s = (knn - z_norm2) + reg;
        if !(s > 0.0 && s.is_finite()) {
            return false;
        }
        let diag = s.sqrt();
        if !(diag > 0.0 && diag.is_finite()) {
            return false;
        }

        // Commit row to `L_W` and ring buffers.
        self.l_w.extend_from_slice(&sc.z);
        self.l_w.push(diag);
        self.x.extend(x.iter().copied());
        for yj in y.iter().take(t_cnt) {
            self.y.push_back(yj.expect("all targets present"));
        }
        self.w.push_back(w);
        if let Some(st) = stamp {
            self.stamp.push_back(st);
        }
        true
    }

    /// Recompute cached dual weights `alpha` and `intercept` from `(L_W, y)`.
    pub fn solve(&mut self, cfg: &KrlsCfg, sc: &mut WinScratch) {
        let m = self.len();
        let t_cnt = cfg.n_targets;
        self.alpha.clear();
        self.alpha.resize(t_cnt * m, 0.0);
        self.intercept.clear();
        self.intercept.resize(t_cnt, 0.0);
        if m == 0 {
            return;
        }
        for t in 0..t_cnt {
            sc.rhs.clear();
            sc.rhs.reserve(m);
            for i in 0..m {
                sc.rhs.push(self.y[i * t_cnt + t]);
            }
            forward_solve_lower(&self.l_w, m, &sc.rhs, &mut sc.z);
            back_solve_lower_t_in_place(&self.l_w, m, &mut sc.z);
            self.alpha[t * m..(t + 1) * m].copy_from_slice(&sc.z);
        }
        if cfg.fit_intercept {
            sc.rhs.clear();
            sc.rhs.resize(m, 1.0);
            forward_solve_lower(&self.l_w, m, &sc.rhs, &mut sc.v1);
            back_solve_lower_t_in_place(&self.l_w, m, &mut sc.v1);
            let denom: f64 = sc.v1.iter().copied().sum();
            if denom > 0.0 && denom.is_finite() {
                for t in 0..t_cnt {
                    let al = &mut self.alpha[t * m..(t + 1) * m];
                    let num: f64 = al.iter().copied().sum();
                    let b = num / denom;
                    self.intercept[t] = b;
                    for (ai, &v1i) in al.iter_mut().zip(&sc.v1) {
                        *ai -= v1i * b;
                    }
                }
            }
        }
    }

    /// Validate deserialized state shape and numerical invariants (`STATE-3`).
    pub fn has_shape(&self, cfg: &KrlsCfg) -> bool {
        let k = cfg.n_features;
        let t = cfg.n_targets;
        let m = self.w.len();
        if cfg.window_rows.is_some_and(|cap| m > cap) {
            return false;
        }
        let expected_stamps = if cfg.window_clock.is_some() { m } else { 0 };
        if self.x.len() != m * k
            || self.y.len() != m * t
            || self.stamp.len() != expected_stamps
            || self.l_w.len() != tri_len(m)
        {
            return false;
        }
        if !self.x.iter().all(|v| v.is_finite())
            || !self.y.iter().all(|v| v.is_finite())
            || !self.w.iter().all(|v| v.is_finite() && *v > 0.0)
            || !self.l_w.iter().all(|v| v.is_finite())
        {
            return false;
        }
        for i in 0..m {
            if self.l_w[tri_row(i) + i] <= 0.0 {
                return false;
            }
        }
        true
    }
}
