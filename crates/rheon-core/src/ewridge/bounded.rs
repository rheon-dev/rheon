//! Bounded / fixed-sum active-set QP solve path for [`EwRidge`].
//!
//! Kept in its own submodule so `ewridge.rs` stays well below the repository
//! file-size cap and the unconstrained path pays zero extra code in its hot
//! functions.

use crate::constraint::{Constraint, Scratch as ProjScratch};
use crate::gaps::Cross;
use crate::qp::{QpStatus, QpWork, solve_qp, solve_ridge_qp};
use crate::{EwCov, TargetGaps};

use super::{EwRidge, Parts};

impl EwRidge {
    /// Solve one `(gram, combo)` system under slope box bounds and/or a fixed
    /// slope sum (`self.cfg.constraint`), warm-started from `self.beta`.
    ///
    /// Writes each reader's coefficients into `beta[j * nc + ci]` and counts
    /// any cap hit or failure into `failures`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn solve_bounded(
        &self,
        cov: &EwCov,
        cross: &Cross,
        readers: &[usize],
        r_raw: &[Vec<f64>],
        ci: usize,
        nc: usize,
        ridge: f64,
        con: &Constraint,
        centred: bool,
        failures: &mut u64,
        parts: &mut Parts,
        qp: &mut QpWork,
        beta: &mut [Vec<f64>],
    ) {
        let kf = self.cfg.n_features;
        let kc = self.cfg.k_total();
        let prev: Option<&[Vec<f64>]> = self.beta.as_deref().map(Vec::as_slice);

        if centred {
            // Centered system on slopes 1..=kf, intercept recovered via
            // `beta_0 = ybar_j - m_j . beta_slopes`.
            parts.a.clear();
            parts.a.resize(kf * kf, 0.0);
            for i in 0..kf {
                let zi = i + 1;
                for j in 0..=i {
                    parts.a[i * kf + j] = cov.cov(zi, j + 1);
                }
            }
            if self.cfg.standardize {
                parts.scale.clear();
                parts.scale.extend((0..kf).map(|i| cov.raw(i + 1, i + 1)));
            }
            let raw_diag = self.cfg.standardize.then_some(parts.scale.as_slice());
            let pairwise = self.cfg.target_gaps == TargetGaps::Pairwise;

            for &j in readers {
                let slot = j * nc + ci;
                let rhs = &cross.c[j][1..=kf];
                let prior = self.cfg.coef_prior.as_ref().map(|c0| &c0[j][1..=kf]);
                let warm = prev.and_then(|pb| pb.get(slot)).map(|v| &v[1..=kf]);
                let out = solve_ridge_qp(
                    &parts.a,
                    rhs,
                    kf,
                    ridge,
                    self.cfg.standardize,
                    raw_diag,
                    prior,
                    con,
                    warm,
                    qp,
                    &mut beta[slot][1..=kf],
                );
                match out.status {
                    QpStatus::Converged | QpStatus::MaxIter => {
                        if out.status == QpStatus::MaxIter {
                            *failures += 1;
                        }
                        let mut b0 = cross.my[j];
                        for i in 0..kf {
                            let z = i + 1;
                            let m_j = if pairwise {
                                cross.mj[j][z]
                            } else {
                                cov.mean(z)
                            };
                            b0 -= m_j * beta[slot][z];
                        }
                        beta[slot][0] = b0;
                    }
                    QpStatus::Failed => {
                        *failures += 1;
                        fallback_slot(prev, slot, &mut beta[slot]);
                    }
                }
            }
            return;
        }

        if !self.cfg.ridge_scale {
            // Through the origin (`fit_intercept = false`), standardized or raw.
            parts.a.clear();
            parts.a.resize(kf * kf, 0.0);
            for i in 0..kf {
                for j in 0..=i {
                    parts.a[i * kf + j] = cov.raw(i, j);
                }
            }
            for (jj, &j) in readers.iter().enumerate() {
                let slot = j * nc + ci;
                let rhs = &r_raw[jj][..kf];
                let prior = self.cfg.coef_prior.as_ref().map(|c0| c0[j].as_slice());
                let warm = prev.and_then(|pb| pb.get(slot)).map(Vec::as_slice);
                let out = solve_ridge_qp(
                    &parts.a,
                    rhs,
                    kf,
                    ridge,
                    self.cfg.standardize,
                    None,
                    prior,
                    con,
                    warm,
                    qp,
                    &mut beta[slot][..kf],
                );
                match out.status {
                    QpStatus::Converged => {}
                    QpStatus::MaxIter => *failures += 1,
                    QpStatus::Failed => {
                        *failures += 1;
                        fallback_slot(prev, slot, &mut beta[slot]);
                    }
                }
            }
            return;
        }

        // `ridge_scale = true`: `(W S + ps * ridge * I) beta = W r + ps * ridge * beta_0`.
        // `standardize` is always false here. If `fit_intercept = true`, slot 0
        // is the penalized but unconstrained intercept.
        let w = cov.n_eff();
        let ps = cov.prior_scale();
        parts.a.clear();
        parts.a.resize(kc * kc, 0.0);
        for i in 0..kc {
            for j in 0..=i {
                parts.a[i * kc + j] = w * cov.raw(i, j);
            }
            parts.a[i * kc + i] += ps * ridge;
        }

        let off = usize::from(self.cfg.fit_intercept);
        let mut lo = Vec::with_capacity(kc);
        let mut hi = Vec::with_capacity(kc);
        let mut eq_a = Vec::new();
        if self.cfg.fit_intercept {
            lo.push(f64::NEG_INFINITY);
            hi.push(f64::INFINITY);
        }
        lo.extend_from_slice(&con.lo);
        hi.extend_from_slice(&con.hi);
        if con.sum.is_some() {
            if self.cfg.fit_intercept {
                eq_a.push(0.0);
            }
            eq_a.resize(kc, 1.0);
        }
        let eq = con.sum.map(|s| (eq_a.as_slice(), s));

        let mut rhs = vec![0.0; kc];
        let mut init = vec![0.0; kc];
        let mut proj_scratch = ProjScratch::default();

        for (jj, &j) in readers.iter().enumerate() {
            let slot = j * nc + ci;
            for i in 0..kc {
                let mut v = w * r_raw[jj][i];
                if let Some(c0) = &self.cfg.coef_prior {
                    v += ps * ridge * c0[j][i];
                }
                rhs[i] = v;
            }
            match prev.and_then(|pb| pb.get(slot)) {
                Some(p) if p.len() == kc && p.iter().all(|v| v.is_finite()) => {
                    init.copy_from_slice(p);
                }
                _ => init.fill(0.0),
            }
            con.project(&mut init[off..], None, &mut proj_scratch);

            let out = solve_qp(&parts.a, &rhs, kc, &lo, &hi, eq, &init, qp, &mut beta[slot]);
            match out.status {
                QpStatus::Converged => {}
                QpStatus::MaxIter => *failures += 1,
                QpStatus::Failed => {
                    *failures += 1;
                    fallback_slot(prev, slot, &mut beta[slot]);
                }
            }
        }
    }
}

fn fallback_slot(prev: Option<&[Vec<f64>]>, slot: usize, dst: &mut [f64]) {
    if let Some(p) = prev.and_then(|pb| pb.get(slot)) {
        dst.copy_from_slice(p);
    } else {
        dst.fill(f64::NAN);
    }
}
