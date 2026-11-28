//! Bounded and fixed-sum ridge solves read off an exported Gram, using the
//! exact same primal active-set QP solver as [`crate::EwRidge`].

use crate::constraint::Constraint;
use crate::qp::{QpStatus, QpWork, solve_ridge_qp};

use super::{Design, GramArrays, Response};

/// Result of solving a bounded/fixed-sum ridge QP over a grid of penalties on
/// one target of a Gram.
#[derive(Clone, Debug, PartialEq)]
pub struct GramQpResult {
    /// Flattened `ridges.len() × g.k` row-major coefficients in `g`'s column
    /// order (intercept recovered when `icept` is `Some` and `g.within` is
    /// `None`; `NaN` for a row whose QP failed).
    pub coefs: Vec<f64>,
    /// Per-ridge indicator that the active-set solver reached its iteration
    /// limit before full dual optimality.
    pub hit_max_iter: Vec<bool>,
}

/// Solve bounded / fixed-sum ridge regression on target `target` over `slots`
/// (with optional intercept column `icept`) for every penalty in `ridges`.
///
/// The objective, standardization, and intercept recovery match
/// [`crate::EwRidge`] under `ridge_scale = "mean"`.
pub fn qp_solve(
    g: &GramArrays<'_>,
    target: usize,
    slots: &[usize],
    icept: Option<usize>,
    ridges: &[f64],
    standardize: bool,
    constraint: &Constraint,
) -> GramQpResult {
    let k = g.k;
    let n = slots.len();
    let design = Design::of(g, slots, icept);
    let resp = Response::of(g, target, slots, icept);
    let raw_diag: Option<Vec<f64>> = (icept.is_some() && standardize).then(|| {
        slots
            .iter()
            .enumerate()
            .map(|(p, &col)| design.a[p * n + p] + g.means[col] * g.means[col])
            .collect()
    });

    let mut coefs = vec![0.0; ridges.len() * k];
    let mut hit_max_iter = vec![false; ridges.len()];
    let mut work = QpWork::default();
    let mut slopes = vec![0.0; n];

    for (ri, &ridge) in ridges.iter().enumerate() {
        let row = &mut coefs[ri * k..(ri + 1) * k];
        let outcome = solve_ridge_qp(
            &design.a,
            &resp.rhs,
            n,
            ridge,
            standardize,
            raw_diag.as_deref(),
            None,
            constraint,
            None,
            &mut work,
            &mut slopes,
        );
        match outcome.status {
            QpStatus::Converged | QpStatus::MaxIter => {
                if outcome.status == QpStatus::MaxIter {
                    hit_max_iter[ri] = true;
                }
                for (p, &col) in slots.iter().enumerate() {
                    row[col] = slopes[p];
                }
                if let Some(ci) = icept {
                    if g.within.is_some() {
                        row[ci] = f64::NAN;
                    } else {
                        let mut b0 = resp.ybar;
                        for (&mean, &slope) in resp.means.iter().zip(slopes.iter()) {
                            b0 -= mean * slope;
                        }
                        row[ci] = b0;
                    }
                }
            }
            QpStatus::Failed => {
                row.fill(f64::NAN);
            }
        }
    }

    GramQpResult {
        coefs,
        hit_max_iter,
    }
}
