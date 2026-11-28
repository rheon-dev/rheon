//! Primal active-set quadratic program (QP) solver for box bounds and an
//! optional single linear equality constraint:
//!
//! ```text
//! min_{x in R^d}  0.5 * x^T H x - c^T x
//! subject to      lo_i <= x_i <= hi_i   (i = 0..d)
//!                 [ a^T x = rhs ]       (optional)
//! ```
//!
//! Used by [`crate::EwRidge`] and `rheon.gram.solve` when slope bounds
//! (`coef_min`, `coef_max`) and/or a fixed slope sum (`coef_sum`) are active.
//!
//! ## Algorithm
//!
//! 1. **Diagonal equilibration**: scale coordinate `i` by `e_i = sqrt(H_ii)`
//!    when `H_ii > 0` (else `1.0`) so every non-zero diagonal entry of the
//!    working Hessian is `1.0`, making tolerances and pivoting independent of
//!    feature units. Normalize any equality vector `a` to unit Euclidean norm.
//! 2. **Feasible warm start**: project the caller's previous fit (or zeros)
//!    onto the constraint set via [`crate::Constraint::project`], initializing
//!    tight bounds into the working set.
//! 3. **Active-set iterations** (Nocedal & Wright §16.5):
//!    - On the free set `F`, attempt strict Cholesky factorization of `H_FF`
//!      (no jitter). When `H_FF` is SPD (the common case), solve directly for
//!      the equality-constrained subspace minimizer `y_F^*` via Schur
//!      complement; if no bound blocks the step (`alpha = 1`), the iterate
//!      takes `y_F^*` directly so the final bits depend only on the optimal
//!      working set and not on the warm-start trajectory.
//!    - When `H_FF` is rank-deficient / positive semi-definite (e.g.,
//!      `ridge = 0` with collinear or constant features), eliminate any
//!      equality constraint on `F` by a Householder reflection `Z` and solve
//!      the reduced subproblem `Z^T H_FF Z` by strict Cholesky if SPD, or by
//!      cyclic Jacobi symmetric eigendecomposition (pure arithmetic + `sqrt`,
//!      no platform `libm`), taking either a zero-curvature descent ray or the
//!      minimum-norm pseudoinverse step.
//!    - When all equality-coupled coordinates are pinned at bounds, check the
//!      feasible multiplier interval `[mu_lo, mu_hi]` and drop a violating
//!      pair `(i_lo, i_hi)` simultaneously so the equality stays feasible.
//!    - Switch from Dantzig's most-negative multiplier rule to Bland's
//!      smallest-index rule after consecutive zero-length steps to guarantee
//!      finite termination under degeneracy.

use crate::constraint::{Constraint, Scratch as ProjScratch};

/// Active-set status of a single variable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BoundState {
    Free,
    Lower,
    Upper,
    /// `lo_i == hi_i`: permanently fixed, never released.
    Fixed,
}

/// Reusable scratch buffers for [`solve_qp`] and [`solve_ridge_qp`], avoiding
/// per-solve heap allocations on the streaming path.
#[derive(Debug, Clone, Default)]
pub struct QpWork {
    e: Vec<f64>,
    h: Vec<f64>,
    c: Vec<f64>,
    lo: Vec<f64>,
    hi: Vec<f64>,
    a: Vec<f64>,
    y: Vec<f64>,
    g: Vec<f64>,
    p: Vec<f64>,
    y_target: Vec<f64>,
    state: Vec<BoundState>,
    free: Vec<usize>,
    h_ff: Vec<f64>,
    l_ff: Vec<f64>,
    rhs_f: Vec<f64>,
    v_f: Vec<f64>,
    w_f: Vec<f64>,
    a_f: Vec<f64>,
    g_f: Vec<f64>,
    /// Working matrices for the ridge-QP wrapper (standardized / unstandardized).
    mat: Vec<f64>,
    lin: Vec<f64>,
    scales: Vec<f64>,
    lo_s: Vec<f64>,
    hi_s: Vec<f64>,
    a_s: Vec<f64>,
    init: Vec<f64>,
    proj: ProjScratch,
}

impl QpWork {
    /// Drop buffers that grew past `max_entries` (for thread-local reuse).
    pub(crate) fn trim(&mut self, max_entries: usize) {
        for v in [
            &mut self.e,
            &mut self.h,
            &mut self.c,
            &mut self.lo,
            &mut self.hi,
            &mut self.a,
            &mut self.y,
            &mut self.g,
            &mut self.p,
            &mut self.y_target,
            &mut self.h_ff,
            &mut self.l_ff,
            &mut self.rhs_f,
            &mut self.v_f,
            &mut self.w_f,
            &mut self.a_f,
            &mut self.g_f,
            &mut self.mat,
            &mut self.lin,
            &mut self.scales,
            &mut self.lo_s,
            &mut self.hi_s,
            &mut self.a_s,
            &mut self.init,
        ] {
            if v.capacity() > max_entries {
                *v = Vec::new();
            }
        }
        if self.state.capacity() > max_entries {
            self.state = Vec::new();
        }
        if self.free.capacity() > max_entries {
            self.free = Vec::new();
        }
    }
}

/// Termination status of [`solve_qp`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QpStatus {
    /// Satisfied KKT optimality conditions within tolerance.
    Converged,
    /// Reached `max_iter` before full dual optimality; `sol` holds the best
    /// primal-feasible iterate found.
    MaxIter,
    /// Objective is unbounded below on the feasible set, or inputs contained
    /// non-finite / strongly indefinite values.
    Failed,
}

/// Output of [`solve_qp`] / [`solve_ridge_qp`].
#[derive(Debug, Clone, PartialEq)]
pub struct QpOutcome {
    pub status: QpStatus,
    pub iters: usize,
}

/// Measured KKT optimality residuals for a candidate solution `x`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KktResiduals {
    /// Maximum bound violation `max(0, lo_i - x_i, x_i - hi_i)`.
    pub bound_err: f64,
    /// Absolute equality violation `|a^T x - rhs|` (`0.0` if no equality).
    pub eq_err: f64,
    /// Maximum stationarity violation on strictly interior variables
    /// `|g_i - mu * a_i|`.
    pub stationarity_err: f64,
    /// Maximum dual-sign violation on active bounds.
    pub dual_err: f64,
}

impl KktResiduals {
    /// Largest of all four KKT residual components.
    pub fn max_err(&self) -> f64 {
        self.bound_err
            .max(self.eq_err)
            .max(self.stationarity_err)
            .max(self.dual_err)
    }
}

/// Compute the exact KKT optimality residuals of `x` for
/// `min 0.5 x^T H x - c^T x` s.t. `lo <= x <= hi`, optional `a^T x = rhs`.
pub fn kkt_residuals(
    h: &[f64],
    c: &[f64],
    d: usize,
    lo: &[f64],
    hi: &[f64],
    eq: Option<(&[f64], f64)>,
    x: &[f64],
) -> KktResiduals {
    debug_assert_eq!(h.len(), d * d);
    debug_assert_eq!(c.len(), d);
    debug_assert_eq!(lo.len(), d);
    debug_assert_eq!(hi.len(), d);
    debug_assert_eq!(x.len(), d);
    if d == 0 {
        return KktResiduals {
            bound_err: 0.0,
            eq_err: 0.0,
            stationarity_err: 0.0,
            dual_err: 0.0,
        };
    }
    let mut g = vec![0.0; d];
    let mut bound_err = 0.0f64;
    for i in 0..d {
        let mut s = -c[i];
        for j in 0..d {
            let hij = if i >= j { h[i * d + j] } else { h[j * d + i] };
            s += hij * x[j];
        }
        g[i] = s;
        if lo[i].is_finite() {
            bound_err = bound_err.max(lo[i] - x[i]);
        }
        if hi[i].is_finite() {
            bound_err = bound_err.max(x[i] - hi[i]);
        }
    }
    let (eq_err, mu) = match eq {
        None => (0.0, 0.0),
        Some((a, rhs)) => {
            let dot: f64 = a.iter().zip(x.iter()).map(|(ai, xi)| ai * xi).sum();
            let err = (dot - rhs).abs();
            // Estimate `mu` from interior variables with `a_i != 0` by least
            // squares `min_mu sum_{i in free} (g_i - mu a_i)^2`, or from the
            // midpoint of the feasible interval `[mu_lo, mu_hi]` when all
            // equality-coupled variables sit at bounds.
            let mut num = 0.0;
            let mut den = 0.0;
            for i in 0..d {
                let tol_b = 1e-8 * (1.0 + x[i].abs());
                let at_lo = lo[i].is_finite() && x[i] <= lo[i] + tol_b;
                let at_hi = hi[i].is_finite() && x[i] >= hi[i] - tol_b;
                if !at_lo && !at_hi && a[i] != 0.0 {
                    num += a[i] * g[i];
                    den += a[i] * a[i];
                }
            }
            let mu = if den > 0.0 {
                num / den
            } else {
                let mut mu_lo = f64::NEG_INFINITY;
                let mut mu_hi = f64::INFINITY;
                for i in 0..d {
                    if a[i] == 0.0 || lo[i] == hi[i] {
                        continue;
                    }
                    let tol_b = 1e-8 * (1.0 + x[i].abs());
                    let at_lo = lo[i].is_finite() && x[i] <= lo[i] + tol_b;
                    let at_hi = hi[i].is_finite() && x[i] >= hi[i] - tol_b;
                    let q = g[i] / a[i];
                    if (at_lo && a[i] > 0.0) || (at_hi && a[i] < 0.0) {
                        mu_hi = mu_hi.min(q);
                    }
                    if (at_hi && a[i] > 0.0) || (at_lo && a[i] < 0.0) {
                        mu_lo = mu_lo.max(q);
                    }
                }
                if mu_lo.is_finite() && mu_hi.is_finite() {
                    0.5 * (mu_lo + mu_hi)
                } else if mu_lo.is_finite() {
                    mu_lo
                } else if mu_hi.is_finite() {
                    mu_hi
                } else {
                    0.0
                }
            };
            (err, mu)
        }
    };
    let mut stationarity_err = 0.0f64;
    let mut dual_err = 0.0f64;
    for i in 0..d {
        if lo[i] == hi[i] {
            continue;
        }
        let ai = eq.map_or(0.0, |(a, _)| a[i]);
        let r = g[i] - mu * ai;
        let tol_b = 1e-8 * (1.0 + x[i].abs());
        let at_lo = lo[i].is_finite() && x[i] <= lo[i] + tol_b;
        let at_hi = hi[i].is_finite() && x[i] >= hi[i] - tol_b;
        if at_lo {
            dual_err = dual_err.max(-r);
        } else if at_hi {
            dual_err = dual_err.max(r);
        } else {
            stationarity_err = stationarity_err.max(r.abs());
        }
    }
    KktResiduals {
        bound_err,
        eq_err,
        stationarity_err,
        dual_err,
    }
}

/// Solve a ridge-regression QP over `k` slope features (shared by
/// [`crate::EwRidge`] and `rheon.gram.solve` for bit-identical results).
///
/// `a_mat` is row-major `k × k` (only the lower triangle `i >= j` is read),
/// `rhs` has length `k` (either centered cross-moments `c_j` or raw `E[x y]`).
/// When `standardize` is `true`, each feature with usable variance is scaled by
/// `s_i = sqrt(A_ii)` and `ridge` is added on the correlation diagonal;
/// `raw_diag` supplies `E[x_i^2]` for [`crate::variance_is_usable`] in the
/// centered solve (`None` checks `A_ii > 0`, matching the origin solve).
///
/// `out` is overwritten with the `k` slope coefficients in the features'
/// original units.
#[allow(clippy::too_many_arguments)]
pub fn solve_ridge_qp(
    a_mat: &[f64],
    rhs: &[f64],
    k: usize,
    ridge: f64,
    standardize: bool,
    raw_diag: Option<&[f64]>,
    coef_prior: Option<&[f64]>,
    constraint: &Constraint,
    warm_start: Option<&[f64]>,
    work: &mut QpWork,
    out: &mut [f64],
) -> QpOutcome {
    debug_assert_eq!(a_mat.len(), k * k);
    debug_assert_eq!(rhs.len(), k);
    debug_assert_eq!(out.len(), k);
    if k == 0 {
        return QpOutcome {
            status: QpStatus::Converged,
            iters: 0,
        };
    }

    // Feasible starting point in original units via exact Euclidean projection.
    work.init.clear();
    match warm_start {
        Some(w) if w.len() == k && w.iter().all(|v| v.is_finite()) => {
            work.init.extend_from_slice(w);
        }
        _ => work.init.resize(k, 0.0),
    }
    constraint.project(&mut work.init, None, &mut work.proj);

    if !standardize {
        work.mat.clear();
        work.mat.resize(k * k, 0.0);
        for i in 0..k {
            for j in 0..=i {
                let v = a_mat[i * k + j];
                work.mat[i * k + j] = v;
                work.mat[j * k + i] = v;
            }
            work.mat[i * k + i] += ridge;
        }
        work.lin.clear();
        work.lin.resize(k, 0.0);
        for i in 0..k {
            let mut b = rhs[i];
            if let Some(c0) = coef_prior {
                b += ridge * c0[i];
            }
            work.lin[i] = b;
        }
        let has_sum = constraint.sum.is_some();
        if has_sum {
            work.a_s.clear();
            work.a_s.resize(k, 1.0);
        }
        // Take the prepared buffers out cleanly so `solve_qp` can borrow `work`.
        let mat = std::mem::take(&mut work.mat);
        let lin = std::mem::take(&mut work.lin);
        let init = std::mem::take(&mut work.init);
        let a_s = std::mem::take(&mut work.a_s);
        let eq = constraint.sum.map(|s| (a_s.as_slice(), s));
        let outcome = solve_qp(
            &mat,
            &lin,
            k,
            &constraint.lo,
            &constraint.hi,
            eq,
            &init,
            work,
            out,
        );
        work.mat = mat;
        work.lin = lin;
        work.init = init;
        work.a_s = a_s;
        return outcome;
    }

    // Standardized solve: work in `u_i = d_i * beta_i` where `d_i = sqrt(A_ii)`
    // for usable columns and `d_i = 1.0` (with zero row/col in H and zero in c)
    // for constant / zero-variance columns.
    work.scales.clear();
    work.scales.resize(k, 0.0);
    for i in 0..k {
        let var = a_mat[i * k + i];
        let usable = match raw_diag {
            Some(rd) => crate::variance_is_usable(var, rd[i]),
            None => var > 0.0 && var.is_finite(),
        };
        work.scales[i] = if usable { var.max(0.0).sqrt() } else { 0.0 };
    }

    work.mat.clear();
    work.mat.resize(k * k, 0.0);
    for i in 0..k {
        let si = work.scales[i];
        if si == 0.0 {
            continue;
        }
        for j in 0..=i {
            let sj = work.scales[j];
            if sj == 0.0 {
                continue;
            }
            let v = a_mat[i * k + j] / (si * sj);
            work.mat[i * k + j] = v;
            work.mat[j * k + i] = v;
        }
        work.mat[i * k + i] += ridge;
    }
    work.lin.clear();
    work.lin.resize(k, 0.0);
    for i in 0..k {
        let si = work.scales[i];
        if si > 0.0 {
            let mut b = rhs[i] / si;
            if let Some(c0) = coef_prior {
                b += ridge * c0[i] * si;
            }
            work.lin[i] = b;
        }
    }
    work.lo_s.clear();
    work.hi_s.clear();
    work.a_s.clear();
    for i in 0..k {
        let di = if work.scales[i] > 0.0 {
            work.scales[i]
        } else {
            1.0
        };
        let lo = constraint.lo[i];
        let hi = constraint.hi[i];
        work.lo_s.push(if lo.is_finite() { lo * di } else { lo });
        work.hi_s.push(if hi.is_finite() { hi * di } else { hi });
        work.init[i] *= di;
        if constraint.sum.is_some() {
            work.a_s.push(1.0 / di);
        }
    }

    // Guard against overflow/underflow in `lo * di` or `1 / di` at extreme scales.
    let scales_safe = work.lo_s.iter().all(|v| !v.is_nan())
        && work.hi_s.iter().all(|v| !v.is_nan())
        && work.init.iter().all(|v| v.is_finite())
        && work.a_s.iter().all(|v| v.is_finite() && *v > 0.0);
    if !scales_safe {
        out.fill(f64::NAN);
        return QpOutcome {
            status: QpStatus::Failed,
            iters: 0,
        };
    }

    let mat = std::mem::take(&mut work.mat);
    let lin = std::mem::take(&mut work.lin);
    let lo_s = std::mem::take(&mut work.lo_s);
    let hi_s = std::mem::take(&mut work.hi_s);
    let a_s = std::mem::take(&mut work.a_s);
    let init = std::mem::take(&mut work.init);
    let scales = std::mem::take(&mut work.scales);

    let eq = constraint.sum.map(|s| (a_s.as_slice(), s));
    let outcome = solve_qp(&mat, &lin, k, &lo_s, &hi_s, eq, &init, work, out);

    if outcome.status != QpStatus::Failed {
        for i in 0..k {
            let di = if scales[i] > 0.0 { scales[i] } else { 1.0 };
            out[i] = (out[i] / di).clamp(constraint.lo[i], constraint.hi[i]);
        }
    }

    work.mat = mat;
    work.lin = lin;
    work.lo_s = lo_s;
    work.hi_s = hi_s;
    work.a_s = a_s;
    work.init = init;
    work.scales = scales;
    outcome
}

/// Solve `min_{x in R^d} 0.5 x^T H x - c^T x` subject to `lo <= x <= hi`
/// and optional `a^T x = rhs`, starting from a primal-feasible `x_init`.
///
/// Only the lower triangle (`i >= j`) of `h` needs to be populated; symmetry is
/// enforced internally. `out` receives the solution vector of length `d`
/// (`NaN` on [`QpStatus::Failed`]).
#[allow(clippy::too_many_arguments)]
pub fn solve_qp(
    h_in: &[f64],
    c_in: &[f64],
    d: usize,
    lo_in: &[f64],
    hi_in: &[f64],
    eq_in: Option<(&[f64], f64)>,
    x_init: &[f64],
    work: &mut QpWork,
    out: &mut [f64],
) -> QpOutcome {
    debug_assert_eq!(h_in.len(), d * d);
    debug_assert_eq!(c_in.len(), d);
    debug_assert_eq!(lo_in.len(), d);
    debug_assert_eq!(hi_in.len(), d);
    debug_assert_eq!(x_init.len(), d);
    debug_assert_eq!(out.len(), d);

    if d == 0 {
        return QpOutcome {
            status: QpStatus::Converged,
            iters: 0,
        };
    }

    // Validate finite inputs (infinite bounds are allowed: -inf on lo, +inf on hi).
    for i in 0..d {
        if !c_in[i].is_finite() || !x_init[i].is_finite() || lo_in[i].is_nan() || hi_in[i].is_nan()
        {
            out.fill(f64::NAN);
            return QpOutcome {
                status: QpStatus::Failed,
                iters: 0,
            };
        }
        for j in 0..=i {
            if !h_in[i * d + j].is_finite() {
                out.fill(f64::NAN);
                return QpOutcome {
                    status: QpStatus::Failed,
                    iters: 0,
                };
            }
        }
    }
    if let Some((a, rhs)) = eq_in
        && (!rhs.is_finite() || a.len() != d || !a.iter().all(|v| v.is_finite()))
    {
        out.fill(f64::NAN);
        return QpOutcome {
            status: QpStatus::Failed,
            iters: 0,
        };
    }

    // 1. Diagonal equilibration: `y_i = e_i * x_i`, `H_hat = E^-1 H E^-1`.
    work.e.clear();
    work.e.resize(d, 1.0);
    let mut use_equil = true;
    for i in 0..d {
        let diag = h_in[i * d + i];
        if diag > 0.0 {
            let s = diag.sqrt();
            if s.is_finite() && s > 0.0 {
                work.e[i] = s;
            }
        }
    }
    // Check that equilibration won't overflow finite bounds, `c`, `x_init`, or `a`.
    for i in 0..d {
        let ei = work.e[i];
        let ok = (c_in[i] / ei).is_finite()
            && (x_init[i] * ei).is_finite()
            && (!lo_in[i].is_finite() || (lo_in[i] * ei).is_finite())
            && (!hi_in[i].is_finite() || (hi_in[i] * ei).is_finite())
            && eq_in.is_none_or(|(a, _)| (a[i] / ei).is_finite());
        if !ok {
            use_equil = false;
            break;
        }
    }
    if !use_equil {
        work.e.fill(1.0);
    }

    work.h.clear();
    work.h.resize(d * d, 0.0);
    for i in 0..d {
        let ei = work.e[i];
        for j in 0..=i {
            let ej = work.e[j];
            let v = if i == j && h_in[i * d + i] > 0.0 && use_equil {
                1.0
            } else {
                h_in[i * d + j] / (ei * ej)
            };
            if !v.is_finite() {
                out.fill(f64::NAN);
                return QpOutcome {
                    status: QpStatus::Failed,
                    iters: 0,
                };
            }
            work.h[i * d + j] = v;
            work.h[j * d + i] = v;
        }
    }
    work.c.clear();
    work.lo.clear();
    work.hi.clear();
    work.y.clear();
    for i in 0..d {
        let ei = work.e[i];
        work.c.push(c_in[i] / ei);
        let lo = if lo_in[i].is_finite() {
            lo_in[i] * ei
        } else {
            lo_in[i]
        };
        let hi = if hi_in[i].is_finite() {
            hi_in[i] * ei
        } else {
            hi_in[i]
        };
        work.lo.push(lo);
        work.hi.push(hi);
        work.y.push((x_init[i] * ei).clamp(lo, hi));
    }

    // Normalize optional equality constraint `a_hat^T y = rhs_hat` to `||a_hat||_2 = 1`.
    work.a.clear();
    let (has_eq, rhs_eq) = match eq_in {
        None => (false, 0.0),
        Some((a_orig, rhs_orig)) => {
            work.a.reserve(d);
            let mut max_abs = 0.0f64;
            for (&ai, &ei) in a_orig.iter().zip(work.e.iter()) {
                let v = ai / ei;
                max_abs = max_abs.max(v.abs());
                work.a.push(v);
            }
            if max_abs == 0.0 {
                // Degenerate zero equality row.
                if rhs_orig.abs() > 1e-12 {
                    out.fill(f64::NAN);
                    return QpOutcome {
                        status: QpStatus::Failed,
                        iters: 0,
                    };
                }
                (false, 0.0)
            } else {
                // Scaled norm avoids overflow/underflow when squaring.
                let mut sum_sq = 0.0;
                for v in &work.a {
                    let q = *v / max_abs;
                    sum_sq += q * q;
                }
                let norm = max_abs * sum_sq.sqrt();
                for v in &mut work.a {
                    *v /= norm;
                }
                (true, rhs_orig / norm)
            }
        }
    };

    // 2. Initialize working set from feasible `y`.
    work.state.clear();
    work.state.resize(d, BoundState::Free);
    for i in 0..d {
        let lo = work.lo[i];
        let hi = work.hi[i];
        if lo == hi {
            work.y[i] = lo;
            work.state[i] = BoundState::Fixed;
        } else {
            let tol_b = 1e-12 * (1.0 + work.y[i].abs());
            if lo.is_finite() && work.y[i] <= lo + tol_b {
                work.y[i] = lo;
                work.state[i] = BoundState::Lower;
            } else if hi.is_finite() && work.y[i] >= hi - tol_b {
                work.y[i] = hi;
                work.state[i] = BoundState::Upper;
            }
        }
    }

    work.g.resize(d, 0.0);
    work.p.resize(d, 0.0);
    work.y_target.resize(d, 0.0);

    let max_iter = (10 * d).max(200);
    let mut zero_steps = 0usize;
    let mut last_dropped: Option<usize> = None;
    let mut status = QpStatus::MaxIter;
    let mut iters = 0usize;

    for iter in 0..max_iter {
        iters = iter + 1;
        let bland = zero_steps >= 3 || iter >= 4 * d;

        // Gradient `g = H y - c` and scale for dimensionless tolerances.
        let mut g_scale = 1.0f64;
        for i in 0..d {
            let mut acc = -work.c[i];
            let row = &work.h[i * d..(i + 1) * d];
            for (&hij, &yj) in row.iter().zip(work.y.iter()) {
                acc += hij * yj;
            }
            work.g[i] = acc;
            g_scale = g_scale.max(work.c[i].abs()).max(acc.abs());
        }
        let tol_dual = 1e-11 * g_scale;

        // Collect free indices.
        work.free.clear();
        let mut n_eq_free = 0usize;
        for i in 0..d {
            if work.state[i] == BoundState::Free {
                work.free.push(i);
                if has_eq && work.a[i] != 0.0 {
                    n_eq_free += 1;
                }
            }
        }
        let nf = work.free.len();

        // Special case: equality constraint is present, and NO free variable
        // carries a non-zero `a_i` (e.g., vertex where every slope is pinned,
        // possibly with a free unconstrained intercept `a_0 = 0`).
        if has_eq && n_eq_free == 0 {
            // If free variables exist with `a_i == 0` (e.g., unconstrained
            // intercept under `ridge_scale`), optimize them first.
            if nf > 0 {
                let sub = solve_subproblem(work, d, false, 0.0, g_scale);
                match sub {
                    SubResult::Failed => {
                        out.fill(f64::NAN);
                        return QpOutcome {
                            status: QpStatus::Failed,
                            iters,
                        };
                    }
                    SubResult::Step {
                        direct_target,
                        is_ray,
                        ..
                    } => {
                        let p_norm = work.p.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
                        let y_norm = work.y.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
                        if p_norm > 1e-12 * (1.0 + y_norm) {
                            let step_ok = apply_step(
                                work,
                                d,
                                direct_target,
                                is_ray,
                                bland,
                                &mut zero_steps,
                                &mut last_dropped,
                            );
                            if !step_ok {
                                out.fill(f64::NAN);
                                return QpOutcome {
                                    status: QpStatus::Failed,
                                    iters,
                                };
                            }
                            continue;
                        }
                    }
                }
            }
            // Check dual feasibility of the pinned equality-coupled bounds:
            // we need a scalar `mu` with `mu_lo <= mu <= mu_hi`.
            let mut mu_lo = f64::NEG_INFINITY;
            let mut idx_lo = None;
            let mut mu_hi = f64::INFINITY;
            let mut idx_hi = None;
            for i in 0..d {
                if work.a[i] == 0.0
                    || work.state[i] == BoundState::Fixed
                    || work.state[i] == BoundState::Free
                {
                    continue;
                }
                let q = work.g[i] / work.a[i];
                let pushes_hi = (work.state[i] == BoundState::Lower && work.a[i] > 0.0)
                    || (work.state[i] == BoundState::Upper && work.a[i] < 0.0);
                if pushes_hi {
                    if q < mu_hi {
                        mu_hi = q;
                        idx_hi = Some(i);
                    }
                } else if q > mu_lo {
                    mu_lo = q;
                    idx_lo = Some(i);
                }
            }
            if mu_lo <= mu_hi + tol_dual {
                // Also check any pinned non-equality bounds (`a_i == 0`), if any.
                if let Some(drop_i) = pick_drop_non_eq(work, d, tol_dual, bland) {
                    work.state[drop_i] = BoundState::Free;
                    last_dropped = Some(drop_i);
                    continue;
                }
                status = QpStatus::Converged;
                break;
            }
            // Drop both `(i_lo, i_hi)` so the equality subspace gains 1 DoF.
            if let (Some(il), Some(ih)) = (idx_lo, idx_hi) {
                work.state[il] = BoundState::Free;
                work.state[ih] = BoundState::Free;
                last_dropped = None;
                continue;
            }
            // No valid pair can be released (all variables fixed/impossible).
            status = QpStatus::Converged;
            break;
        }

        // General subproblem on `F` (with `has_eq && n_eq_free >= 1`, or `!has_eq`).
        let sub = if nf == 0 {
            SubResult::Step {
                mu: 0.0,
                direct_target: false,
                is_ray: false,
            }
        } else {
            solve_subproblem(work, d, has_eq, rhs_eq, g_scale)
        };

        let (mu, direct_target, is_ray) = match sub {
            SubResult::Failed => {
                out.fill(f64::NAN);
                return QpOutcome {
                    status: QpStatus::Failed,
                    iters,
                };
            }
            SubResult::Step {
                mu,
                direct_target,
                is_ray,
            } => (mu, direct_target, is_ray),
        };

        let p_norm = work
            .free
            .iter()
            .map(|&i| work.p[i].abs())
            .fold(0.0f64, f64::max);
        let y_norm = work.y.iter().map(|v| v.abs()).fold(0.0f64, f64::max);

        if p_norm <= 1e-12 * (1.0 + y_norm) && !is_ray {
            // Take the tiny polish step if it stays feasible.
            if direct_target {
                let feasible = work.free.iter().all(|&i| {
                    let v = work.y_target[i];
                    v >= work.lo[i] - 1e-12 * (1.0 + work.lo[i].abs().min(1e100))
                        && v <= work.hi[i] + 1e-12 * (1.0 + work.hi[i].abs().min(1e100))
                });
                if feasible {
                    for &i in &work.free {
                        work.y[i] = work.y_target[i].clamp(work.lo[i], work.hi[i]);
                    }
                    recompute_grad(work, d);
                }
            }
            // Check dual multipliers on active bounds.
            if let Some(drop_i) = pick_drop(work, d, has_eq, mu, tol_dual, bland) {
                work.state[drop_i] = BoundState::Free;
                last_dropped = Some(drop_i);
                continue;
            }
            status = QpStatus::Converged;
            break;
        }

        // Non-zero step: line search to blocking bound (or full step `alpha = 1`).
        let (alpha, blocker) = line_search(work, is_ray, bland);
        if is_ray && blocker.is_none() {
            // Unbounded below along a zero-curvature descent ray.
            out.fill(f64::NAN);
            return QpOutcome {
                status: QpStatus::Failed,
                iters,
            };
        }

        if let Some((b_idx, b_side)) = blocker {
            // Guard against numerical re-blocking of the exact variable just
            // dropped at `alpha == 0` with a tiny step.
            if alpha == 0.0
                && last_dropped == Some(b_idx)
                && work.p[b_idx].abs() <= 1e-10 * (1.0 + y_norm)
            {
                work.state[b_idx] = b_side;
                work.y[b_idx] = match b_side {
                    BoundState::Lower => work.lo[b_idx],
                    BoundState::Upper => work.hi[b_idx],
                    _ => work.y[b_idx],
                };
                status = QpStatus::Converged;
                break;
            }
            for &i in &work.free {
                work.y[i] = (work.y[i] + alpha * work.p[i]).clamp(work.lo[i], work.hi[i]);
            }
            work.state[b_idx] = b_side;
            work.y[b_idx] = match b_side {
                BoundState::Lower => work.lo[b_idx],
                BoundState::Upper => work.hi[b_idx],
                _ => work.y[b_idx],
            };
            if alpha == 0.0 {
                zero_steps += 1;
            } else {
                zero_steps = 0;
            }
            last_dropped = None;
        } else {
            // Full step `alpha = 1`: no bound blocked!
            if direct_target {
                for &i in &work.free {
                    work.y[i] = work.y_target[i].clamp(work.lo[i], work.hi[i]);
                }
            } else {
                for &i in &work.free {
                    work.y[i] = (work.y[i] + work.p[i]).clamp(work.lo[i], work.hi[i]);
                }
            }
            zero_steps = 0;
            last_dropped = None;

            // Fast optimality check right after a full SPD Newton step: avoids
            // a redundant second Cholesky factorization on the same working set.
            if direct_target {
                recompute_grad(work, d);
                let mut g_scale2 = 1.0f64;
                for i in 0..d {
                    g_scale2 = g_scale2.max(work.c[i].abs()).max(work.g[i].abs());
                }
                let stat_ok = work.free.iter().all(|&i| {
                    let ai = if has_eq { work.a[i] } else { 0.0 };
                    (work.g[i] - mu * ai).abs() <= 1e-10 * g_scale2
                });
                if stat_ok {
                    let tol_dual2 = 1e-11 * g_scale2;
                    if let Some(drop_i) = pick_drop(work, d, has_eq, mu, tol_dual2, bland) {
                        work.state[drop_i] = BoundState::Free;
                        last_dropped = Some(drop_i);
                        continue;
                    }
                    status = QpStatus::Converged;
                    break;
                }
            }
        }
    }

    // Unscale back to original coordinates and clamp tiny rounding excursions.
    for i in 0..d {
        let v = work.y[i] / work.e[i];
        if !v.is_finite() {
            out.fill(f64::NAN);
            return QpOutcome {
                status: QpStatus::Failed,
                iters,
            };
        }
        out[i] = v.clamp(lo_in[i], hi_in[i]);
    }
    QpOutcome { status, iters }
}

fn recompute_grad(work: &mut QpWork, d: usize) {
    for i in 0..d {
        let mut acc = -work.c[i];
        let row = &work.h[i * d..(i + 1) * d];
        for (&hij, &yj) in row.iter().zip(work.y.iter()) {
            acc += hij * yj;
        }
        work.g[i] = acc;
    }
}

fn apply_step(
    work: &mut QpWork,
    _d: usize,
    direct_target: bool,
    is_ray: bool,
    bland: bool,
    zero_steps: &mut usize,
    last_dropped: &mut Option<usize>,
) -> bool {
    let (alpha, blocker) = line_search(work, is_ray, bland);
    if is_ray && blocker.is_none() {
        return false;
    }
    if let Some((b_idx, b_side)) = blocker {
        for &i in &work.free {
            work.y[i] = (work.y[i] + alpha * work.p[i]).clamp(work.lo[i], work.hi[i]);
        }
        work.state[b_idx] = b_side;
        work.y[b_idx] = match b_side {
            BoundState::Lower => work.lo[b_idx],
            BoundState::Upper => work.hi[b_idx],
            _ => work.y[b_idx],
        };
        if alpha == 0.0 {
            *zero_steps += 1;
        } else {
            *zero_steps = 0;
        }
        *last_dropped = None;
    } else {
        if direct_target {
            for &i in &work.free {
                work.y[i] = work.y_target[i].clamp(work.lo[i], work.hi[i]);
            }
        } else {
            for &i in &work.free {
                work.y[i] = (work.y[i] + work.p[i]).clamp(work.lo[i], work.hi[i]);
            }
        }
        *zero_steps = 0;
        *last_dropped = None;
    }
    true
}

/// Ratio test along `work.p` on `work.free`. Returns `(alpha, blocker)`.
fn line_search(work: &QpWork, is_ray: bool, bland: bool) -> (f64, Option<(usize, BoundState)>) {
    let p_max = work
        .free
        .iter()
        .map(|&i| work.p[i].abs())
        .fold(0.0f64, f64::max);
    let dir_tol = 1e-14 * p_max;
    let mut best_alpha = if is_ray { f64::INFINITY } else { 1.0 };
    let mut blocker: Option<(usize, BoundState)> = None;
    for &i in &work.free {
        let pi = work.p[i];
        if pi < -dir_tol && work.lo[i].is_finite() {
            let a = ((work.lo[i] - work.y[i]) / pi).max(0.0);
            if a < best_alpha || (blocker.is_none() && !is_ray && a <= 1.0) {
                let take = match blocker {
                    None => is_ray || a <= 1.0,
                    Some((prev_i, _)) => {
                        a < best_alpha - 1e-15 * best_alpha
                            || ((a - best_alpha).abs() <= 1e-15 * best_alpha
                                && (bland && i < prev_i))
                    }
                };
                if take {
                    best_alpha = a;
                    blocker = Some((i, BoundState::Lower));
                }
            }
        } else if pi > dir_tol && work.hi[i].is_finite() {
            let a = ((work.hi[i] - work.y[i]) / pi).max(0.0);
            if a < best_alpha || (blocker.is_none() && !is_ray && a <= 1.0) {
                let take = match blocker {
                    None => is_ray || a <= 1.0,
                    Some((prev_i, _)) => {
                        a < best_alpha - 1e-15 * best_alpha
                            || ((a - best_alpha).abs() <= 1e-15 * best_alpha
                                && (bland && i < prev_i))
                    }
                };
                if take {
                    best_alpha = a;
                    blocker = Some((i, BoundState::Upper));
                }
            }
        }
    }
    (best_alpha, blocker)
}

/// Choose an active bound to drop from the working set (Dantzig's most-negative
/// multiplier, or Bland's smallest-index rule when `bland` is `true`).
fn pick_drop(
    work: &QpWork,
    d: usize,
    has_eq: bool,
    mu: f64,
    tol_dual: f64,
    bland: bool,
) -> Option<usize> {
    let mut best_idx = None;
    let mut best_lam = -tol_dual;
    for i in 0..d {
        let ai = if has_eq { work.a[i] } else { 0.0 };
        let lam = match work.state[i] {
            BoundState::Lower => work.g[i] - mu * ai,
            BoundState::Upper => mu * ai - work.g[i],
            BoundState::Free | BoundState::Fixed => continue,
        };
        if lam < -tol_dual {
            if bland {
                return Some(i);
            }
            if lam < best_lam {
                best_lam = lam;
                best_idx = Some(i);
            }
        }
    }
    best_idx
}

/// Choose a violating active bound among coordinates with `a_i == 0`.
fn pick_drop_non_eq(work: &QpWork, d: usize, tol_dual: f64, bland: bool) -> Option<usize> {
    let mut best_idx = None;
    let mut best_lam = -tol_dual;
    for i in 0..d {
        if work.a[i] != 0.0 {
            continue;
        }
        let lam = match work.state[i] {
            BoundState::Lower => work.g[i],
            BoundState::Upper => -work.g[i],
            BoundState::Free | BoundState::Fixed => continue,
        };
        if lam < -tol_dual {
            if bland {
                return Some(i);
            }
            if lam < best_lam {
                best_lam = lam;
                best_idx = Some(i);
            }
        }
    }
    best_idx
}

enum SubResult {
    Step {
        mu: f64,
        direct_target: bool,
        is_ray: bool,
    },
    Failed,
}

/// Solve the equality-constrained quadratic subproblem on `work.free`.
fn solve_subproblem(
    work: &mut QpWork,
    d: usize,
    has_eq: bool,
    rhs_eq: f64,
    g_scale: f64,
) -> SubResult {
    let nf = work.free.len();
    work.p.fill(0.0);
    if nf == 0 {
        return SubResult::Step {
            mu: 0.0,
            direct_target: false,
            is_ray: false,
        };
    }

    // Gather `H_FF` (lower triangle + full) and `rhs_F = c_F - H_FA y_A`.
    work.h_ff.clear();
    work.h_ff.resize(nf * nf, 0.0);
    work.rhs_f.clear();
    work.rhs_f.resize(nf, 0.0);
    work.g_f.clear();
    work.g_f.resize(nf, 0.0);
    for (r, &i) in work.free.iter().enumerate() {
        let row = &work.h[i * d..(i + 1) * d];
        for (c, &j) in work.free.iter().enumerate() {
            work.h_ff[r * nf + c] = row[j];
        }
        let mut rhs_i = work.c[i];
        for ((&hij, &st), &yj) in row.iter().zip(work.state.iter()).zip(work.y.iter()) {
            if st != BoundState::Free {
                rhs_i -= hij * yj;
            }
        }
        work.rhs_f[r] = rhs_i;
        work.g_f[r] = work.g[i];
    }

    let mut eq_rhs_f = 0.0;
    let mut eq_norm2 = 0.0;
    if has_eq {
        work.a_f.clear();
        work.a_f.resize(nf, 0.0);
        eq_rhs_f = rhs_eq;
        for j in 0..d {
            if work.state[j] != BoundState::Free {
                eq_rhs_f -= work.a[j] * work.y[j];
            }
        }
        for (r, &i) in work.free.iter().enumerate() {
            let ai = work.a[i];
            work.a_f[r] = ai;
            eq_norm2 += ai * ai;
        }
    }

    // Fast path: strict Cholesky on `H_FF` (no jitter).
    if cholesky_strict(&work.h_ff, nf, &mut work.l_ff) {
        work.v_f.resize(nf, 0.0);
        chol_solve(&work.l_ff, nf, &work.rhs_f, &mut work.v_f);
        let mu = if has_eq && eq_norm2 > 0.0 {
            work.w_f.resize(nf, 0.0);
            chol_solve(&work.l_ff, nf, &work.a_f, &mut work.w_f);
            let aw: f64 = work
                .a_f
                .iter()
                .zip(work.w_f.iter())
                .map(|(a, w)| a * w)
                .sum();
            let av: f64 = work
                .a_f
                .iter()
                .zip(work.v_f.iter())
                .map(|(a, v)| a * v)
                .sum();
            if !(aw > 0.0 && aw.is_finite()) {
                return SubResult::Failed;
            }
            let mu = (eq_rhs_f - av) / aw;
            if !mu.is_finite() {
                return SubResult::Failed;
            }
            for r in 0..nf {
                work.v_f[r] += mu * work.w_f[r];
            }
            mu
        } else {
            0.0
        };
        if !work.v_f.iter().all(|v| v.is_finite()) {
            return SubResult::Failed;
        }
        for (r, &i) in work.free.iter().enumerate() {
            work.y_target[i] = work.v_f[r];
            work.p[i] = work.v_f[r] - work.y[i];
        }
        return SubResult::Step {
            mu,
            direct_target: true,
            is_ray: false,
        };
    }

    // PSD / singular fallback: solve in step form `p_F` via Householder
    // null-space elimination (if equality is active) + strict Cholesky or
    // cyclic Jacobi symmetric eigendecomposition.
    solve_subproblem_psd(work, has_eq && eq_norm2 > 0.0, rhs_eq, eq_norm2, g_scale)
}

/// PSD / rank-deficient subproblem solver for `min 0.5 p^T H_FF p + g_F^T p`
/// subject to `[a_F^T p = r_eq]`.
fn solve_subproblem_psd(
    work: &mut QpWork,
    has_eq: bool,
    rhs_eq: f64,
    eq_norm2: f64,
    g_scale: f64,
) -> SubResult {
    let nf = work.free.len();
    let r_eq = if has_eq {
        let dot: f64 = work
            .a
            .iter()
            .zip(work.y.iter())
            .map(|(ai, yi)| ai * yi)
            .sum();
        rhs_eq - dot
    } else {
        0.0
    };

    if has_eq && nf == 1 {
        // Only 1 free equality-coupled variable: `p_0 = r_eq / a_0`, 0 DoF.
        let i0 = work.free[0];
        let a0 = work.a_f[0];
        let p0 = r_eq / a0;
        if !p0.is_finite() {
            return SubResult::Failed;
        }
        work.p[i0] = p0;
        let g_new = work.g_f[0] + work.h_ff[0] * p0;
        let mu = g_new / a0;
        return if mu.is_finite() {
            SubResult::Step {
                mu,
                direct_target: false,
                is_ray: false,
            }
        } else {
            SubResult::Failed
        };
    }

    // Build reduced problem `(H_Z, g_Z)` of order `nz`:
    // - without equality: `nz = nf`, `H_Z = H_FF`, `g_Z = g_F`, `p_0 = 0`;
    // - with equality: Householder `Q` maps `a_F / ||a_F||` to `+- e_0`,
    //   `Z = Q[:, 1..nf]`, `nz = nf - 1`, `p_0 = (r_eq / ||a_F||^2) a_F`.
    let (nz, p0, hh_v, hh_beta, h_z, g_z) = if has_eq {
        let norm = eq_norm2.sqrt();
        let mut u: Vec<f64> = work.a_f.iter().map(|&x| x / norm).collect();
        let sigma = if u[0] >= 0.0 { 1.0 } else { -1.0 };
        u[0] += sigma;
        let v_norm2: f64 = u.iter().map(|x| x * x).sum();
        let beta = 2.0 / v_norm2;

        // `p_0 = (r_eq / eq_norm2) * a_F`
        let scale0 = r_eq / eq_norm2;
        let p0: Vec<f64> = work.a_f.iter().map(|&ai| scale0 * ai).collect();
        // `g_shifted = g_F + H_FF p_0`
        let mut g_sh = vec![0.0; nf];
        for (r, out_r) in g_sh.iter_mut().enumerate() {
            let mut s = work.g_f[r];
            for (&hrc, &p0c) in work.h_ff[r * nf..(r + 1) * nf].iter().zip(p0.iter()) {
                s += hrc * p0c;
            }
            *out_r = s;
        }
        // `Q g_shifted`, then slice `[1..nf]`.
        apply_householder_vec(&u, beta, &mut g_sh);
        let g_z = g_sh[1..].to_vec();

        // `M = Q H_FF Q`, then take bottom-right `(nf - 1) x (nf - 1)` block.
        let mut m = work.h_ff.clone();
        // Left-multiply every column `c` by `Q`.
        let mut col = vec![0.0; nf];
        for c in 0..nf {
            for r in 0..nf {
                col[r] = m[r * nf + c];
            }
            apply_householder_vec(&u, beta, &mut col);
            for r in 0..nf {
                m[r * nf + c] = col[r];
            }
        }
        // Right-multiply every row `r` by `Q`.
        for r in 0..nf {
            apply_householder_vec(&u, beta, &mut m[r * nf..(r + 1) * nf]);
        }
        let nz = nf - 1;
        let mut h_z = vec![0.0; nz * nz];
        for r in 0..nz {
            for c in 0..=r {
                let sym = 0.5 * (m[(r + 1) * nf + (c + 1)] + m[(c + 1) * nf + (r + 1)]);
                h_z[r * nz + c] = sym;
                h_z[c * nz + r] = sym;
            }
        }
        (nz, p0, u, beta, h_z, g_z)
    } else {
        (
            nf,
            vec![0.0; nf],
            Vec::new(),
            0.0,
            work.h_ff.clone(),
            work.g_f.clone(),
        )
    };

    // Try strict Cholesky on `H_Z` first (handles e.g. constant feature + sum
    // constraint where `H_FF` is singular but `H_Z` is SPD!).
    let mut l_z = Vec::new();
    let (q, is_ray) = if cholesky_strict(&h_z, nz, &mut l_z) {
        let neg_g: Vec<f64> = g_z.iter().map(|v| -v).collect();
        let mut q = vec![0.0; nz];
        chol_solve(&l_z, nz, &neg_g, &mut q);
        (q, false)
    } else {
        // Cyclic Jacobi eigendecomposition `H_Z = V diag(lam) V^T`.
        let (evals, evecs) = jacobi_eigh(&h_z, nz);
        let max_eig = evals.iter().copied().fold(0.0f64, f64::max);
        let min_eig = evals.iter().copied().fold(f64::INFINITY, f64::min);
        if !max_eig.is_finite() || !min_eig.is_finite() || min_eig < -1e-6 * max_eig.max(1.0) {
            return SubResult::Failed;
        }
        let eig_tol = (1e-12 * max_eig).max(1e-14);

        // Project `g_Z` onto eigenvectors.
        let mut proj = vec![0.0; nz];
        for j in 0..nz {
            let mut dot = 0.0;
            for r in 0..nz {
                dot += evecs[r * nz + j] * g_z[r];
            }
            proj[j] = dot;
        }

        // Check null-space component of `-g_Z`.
        let mut d_null = vec![0.0; nz];
        let mut null_norm2 = 0.0;
        for j in 0..nz {
            if evals[j] <= eig_tol {
                let coeff = -proj[j];
                null_norm2 += coeff * coeff;
                for r in 0..nz {
                    d_null[r] += coeff * evecs[r * nz + j];
                }
            }
        }
        if null_norm2.sqrt() > 1e-10 * g_scale {
            (d_null, true)
        } else {
            // Minimum-norm pseudoinverse step in range of `H_Z`.
            let mut q = vec![0.0; nz];
            for j in 0..nz {
                if evals[j] > eig_tol {
                    let coeff = -proj[j] / evals[j];
                    for r in 0..nz {
                        q[r] += coeff * evecs[r * nz + j];
                    }
                }
            }
            (q, false)
        }
    };

    // Map `q` back to `p_F = p_0 + Z q` (or `Z q` for a zero-curvature ray).
    let p_f = if has_eq {
        let mut zq = vec![0.0; nf];
        zq[1..].copy_from_slice(&q);
        apply_householder_vec(&hh_v, hh_beta, &mut zq);
        if is_ray {
            zq
        } else {
            (0..nf).map(|r| p0[r] + zq[r]).collect()
        }
    } else {
        q
    };

    if !p_f.iter().all(|v| v.is_finite()) {
        return SubResult::Failed;
    }
    for (r, &i) in work.free.iter().enumerate() {
        work.p[i] = p_f[r];
    }

    // Equality multiplier `mu = a_F^T (g_F + H_FF p_F) / ||a_F||^2`.
    let mu = if has_eq {
        let mut num = 0.0;
        for r in 0..nf {
            let mut gn = work.g_f[r];
            for (&hrc, &pfc) in work.h_ff[r * nf..(r + 1) * nf].iter().zip(p_f.iter()) {
                gn += hrc * pfc;
            }
            num += work.a_f[r] * gn;
        }
        let m = num / eq_norm2;
        if !m.is_finite() {
            return SubResult::Failed;
        }
        m
    } else {
        0.0
    };

    SubResult::Step {
        mu,
        direct_target: false,
        is_ray,
    }
}

/// Apply Householder reflection `x <- (I - beta v v^T) x` in place.
#[inline]
fn apply_householder_vec(v: &[f64], beta: f64, x: &mut [f64]) {
    let dot: f64 = v.iter().zip(x.iter()).map(|(vi, xi)| vi * xi).sum();
    let s = beta * dot;
    for (xi, &vi) in x.iter_mut().zip(v.iter()) {
        *xi -= s * vi;
    }
}

/// Strict lower-triangular Cholesky `L L^T = A` (row-major `n × n`), with no
/// jitter. Returns `false` if any pivot drops below `1e-13 * max(1, A_jj)`.
fn cholesky_strict(a: &[f64], n: usize, l: &mut Vec<f64>) -> bool {
    l.clear();
    l.resize(n * n, 0.0);
    for j in 0..n {
        let diag = a[j * n + j];
        let mut s = diag;
        let row_j = j * n;
        for t in 0..j {
            let ljt = l[row_j + t];
            s -= ljt * ljt;
        }
        let tol = 1e-13 * diag.abs().max(1.0);
        if !(s > tol && s.is_finite()) {
            return false;
        }
        let ljj = s.sqrt();
        l[row_j + j] = ljj;
        let inv = 1.0 / ljj;
        for i in (j + 1)..n {
            let row_i = i * n;
            let mut sum = a[row_i + j];
            for t in 0..j {
                sum -= l[row_i + t] * l[row_j + t];
            }
            l[row_i + j] = sum * inv;
        }
    }
    true
}

/// Solve `L L^T x = b` given lower-triangular `L` (row-major `n × n`).
fn chol_solve(l: &[f64], n: usize, b: &[f64], x: &mut [f64]) {
    // Forward substitution `L y = b`.
    for i in 0..n {
        let row_i = i * n;
        let mut s = b[i];
        for j in 0..i {
            s -= l[row_i + j] * x[j];
        }
        x[i] = s / l[row_i + i];
    }
    // Back substitution `L^T x = y`.
    for i in (0..n).rev() {
        let mut s = x[i];
        for j in (i + 1)..n {
            s -= l[j * n + i] * x[j];
        }
        x[i] = s / l[i * n + i];
    }
}

/// Cyclic Jacobi symmetric eigendecomposition `A = V diag(evals) V^T` using
/// pure IEEE-754 arithmetic + `sqrt` (zero `libm` calls).
///
/// Returns `(evals, evecs)` where column `j` of row-major `evecs` (`n × n`)
/// is the unit eigenvector for `evals[j]`.
fn jacobi_eigh(a_in: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut a = a_in.to_vec();
    let mut v = vec![0.0; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    if n <= 1 {
        let evals = if n == 1 { vec![a[0]] } else { Vec::new() };
        return (evals, v);
    }

    let frob_scale = a.iter().map(|x| x.abs()).fold(0.0f64, f64::max).max(1e-300);

    for _sweep in 0..64 {
        let mut max_off = 0.0f64;
        for p in 0..n {
            for q in (p + 1)..n {
                max_off = max_off.max(a[p * n + q].abs());
            }
        }
        if max_off <= 1e-15 * frob_scale {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = a[p * n + q];
                if apq.abs() <= 1e-16 * frob_scale {
                    continue;
                }
                let app = a[p * n + p];
                let aqq = a[q * n + q];
                let tau = (aqq - app) / (2.0 * apq);
                let t = if tau.abs() > 1e150 {
                    0.5 / tau
                } else if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;

                a[p * n + p] = app - t * apq;
                a[q * n + q] = aqq + t * apq;
                a[p * n + q] = 0.0;
                a[q * n + p] = 0.0;

                for r in 0..n {
                    if r == p || r == q {
                        continue;
                    }
                    let arp = a[r * n + p];
                    let arq = a[r * n + q];
                    let np = c * arp - s * arq;
                    let nq = s * arp + c * arq;
                    a[r * n + p] = np;
                    a[p * n + r] = np;
                    a[r * n + q] = nq;
                    a[q * n + r] = nq;
                }
                for r in 0..n {
                    let vrp = v[r * n + p];
                    let vrq = v[r * n + q];
                    v[r * n + p] = c * vrp - s * vrq;
                    v[r * n + q] = s * vrp + c * vrq;
                }
            }
        }
    }

    let evals = (0..n).map(|i| a[i * n + i]).collect();
    (evals, v)
}

#[cfg(test)]
#[path = "qp_tests.rs"]
mod tests;
