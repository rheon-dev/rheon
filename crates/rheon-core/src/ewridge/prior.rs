//! Hierarchical dynamic-prior solve path for [`EwRidge`].
//!
//! Kept in its own submodule so `ewridge.rs` stays well below the repository
//! file-size cap and models without `prior_weight` pay zero extra code in
//! their hot solve loops.
//!
//! On the sum scale, a dynamic prior `beta_p` with strength `tau = prior_weight`
//! solves:
//!
//! ```text
//! (W_j S_j + ridge_eff D_ridge + tau D_prior) beta
//!     = W_j eta_j + ridge_eff D_ridge beta_0_static + tau D_prior beta_p
//! ```
//!
//! where `D_prior = I` when `prior_include_intercept` is true (or
//! `fit_intercept` is false) and `D_prior = diag(0, I_k)` when
//! `fit_intercept` is true and `prior_include_intercept` is false.
//!
//! Under `fit_intercept && !ridge_scale && !standardize && prior_include_intercept`,
//! the intercept is eliminated by exact Schur complement so feature levels `L`
//! never incur `L^2 * eps` cancellation while matching the full `(k+1)x(k+1)`
//! system to machine precision:
//!
//! ```text
//! alpha = W_j * tau / (W_j + tau),   q = (W_j / (W_j + tau)) * m_j
//! (W_j C + alpha m_j m_j^T + (W_j ridge + tau) I) beta_s
//!     = W_j c_j + alpha (ybar_j - beta_{p,0}) m_j + W_j ridge beta_{0,s} + tau beta_{p,s}
//! beta_0 = (W_j (ybar_j - m_j . beta_s) + tau beta_{p,0}) / (W_j + tau)
//! ```
//!
//! Under `standardize`, the prior enters in standardized centered coordinates
//! `z_tilde = [1, (x - m)/s]`, where the data Gram is block-diagonal
//! `diag(1, R)` and the prior preserves both shift and scale invariance.

use crate::constraint::{Constraint, Scratch as ProjScratch};
use crate::gaps::Cross;
use crate::lincon::{ConstraintProj, LinearConstraints};
use crate::qp::{QpStatus, QpWork, solve_qp};
use crate::solve::{FactorWork, SpdFactor};
use crate::{EwCov, TargetGaps};

use super::{EwRidge, Parts, Solved};

/// Whether `prior` has a finite `k_total`-length vector for target `j`.
pub(super) fn valid_prior_of(
    prior: Option<&[Vec<f64>]>,
    j: usize,
    k_total: usize,
) -> Option<&[f64]> {
    let v = prior?.get(j)?;
    (v.len() == k_total && v.iter().all(|x| x.is_finite())).then_some(v.as_slice())
}

/// Whether a flat `n_targets * k_total` slice has a finite slice for target `j`.
pub(super) fn valid_flat_prior_of(
    flat: Option<&[f64]>,
    j: usize,
    k_total: usize,
) -> Option<&[f64]> {
    let slice = flat?;
    let start = j.checked_mul(k_total)?;
    let end = start.checked_add(k_total)?;
    let v = slice.get(start..end)?;
    v.iter().all(|x| x.is_finite()).then_some(v)
}

impl EwRidge {
    /// Whether cold-start (`W_j == 0`) can predict directly from a parent prior.
    #[inline]
    pub(super) fn cold_start_identified(&self) -> bool {
        !self.cfg.fit_intercept || self.cfg.prior_include_intercept
    }

    /// Cold-start posterior at `W_j == 0`: `bp` itself, projected onto
    /// `constraints` or `constraint` if configured. Returns `false` if the
    /// intercept is unidentified or projection fails.
    pub(super) fn cold_start_beta(&self, bp: &[f64], out: &mut [f64]) -> bool {
        let kc = self.cfg.k_total();
        if !self.cold_start_identified()
            || bp.len() != kc
            || !bp.iter().all(|v| v.is_finite())
            || out.len() != kc
        {
            return false;
        }
        out.copy_from_slice(bp);
        let off = usize::from(self.cfg.fit_intercept);
        let kf = self.cfg.n_features;
        if let Some(lc) = &self.cfg.constraints {
            return project_slopes_linear(lc, kf, &mut out[off..]);
        }
        if let Some(con) = &self.cfg.constraint {
            let mut scratch = ProjScratch::default();
            con.project(&mut out[off..], None, &mut scratch);
        }
        true
    }

    /// Solve one group of targets sharing the same Gram, weight `wj`, and
    /// feature means, all having valid finite dynamic priors.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn solve_with_prior(
        &self,
        cov: &EwCov,
        cross: &Cross,
        readers: &[usize],
        wj: f64,
        r_raw: &[Vec<f64>],
        priors: &[Vec<f64>],
        failures: &mut u64,
        work: &mut FactorWork,
        parts: &mut Parts,
        zidx: &[usize],
        ridge: f64,
        centred: bool,
    ) -> Option<Solved> {
        let tau = self.cfg.prior_weight?;
        let inc_int = self.cfg.prior_include_intercept;
        let kc = zidx.len();
        let mr = readers.len();
        debug_assert_eq!(kc, self.cfg.k_total());
        debug_assert!(!readers.is_empty());

        // Zero child weight under mean-scale ridge: exact limit `W_j -> 0+`
        // is `beta = bp` (projected onto constraints if present) when the
        // intercept is pooled (or absent); unidentified otherwise.
        if !self.cfg.ridge_scale && wj <= 0.0 {
            if !self.cold_start_identified() {
                return None;
            }
            let mut sol = vec![0.0; kc * mr];
            let mut ill = false;
            for (jj, &j) in readers.iter().enumerate() {
                if !self.cold_start_beta(&priors[j], &mut sol[jj * kc..(jj + 1) * kc]) {
                    *failures += 1;
                    ill = true;
                    sol.fill(f64::NAN);
                    break;
                }
            }
            return Some(Solved {
                sol,
                factor: None,
                proj: None,
                a: Vec::new(),
                keep: Vec::new(),
                s: Vec::new(),
                mean: Vec::new(),
                centred,
                shift: tau,
                scale: 0.0,
                ill,
                intercept_weight: (centred && inc_int).then_some(0.0),
                edf0: Some(0.0),
                lam_intercept: None,
            });
        }

        if centred {
            if self.cfg.standardize {
                self.solve_centred_std_prior(
                    cov, cross, readers, wj, priors, failures, work, parts, zidx, ridge, tau,
                    inc_int,
                )
            } else {
                self.solve_centred_unstd_prior(
                    cov, cross, readers, wj, priors, failures, work, parts, zidx, ridge, tau,
                    inc_int,
                )
            }
        } else if self.cfg.ridge_scale {
            self.solve_ridge_scale_prior(
                cov, readers, wj, r_raw, priors, failures, work, parts, zidx, ridge, tau, inc_int,
            )
        } else if !self.cfg.standardize {
            self.solve_origin_unstd_prior(
                cov, readers, wj, r_raw, priors, failures, work, parts, zidx, ridge, tau,
            )
        } else {
            self.solve_origin_std_prior(
                cov, readers, wj, r_raw, priors, failures, work, parts, zidx, ridge, tau,
            )
        }
    }

    /// Centred unstandardized solve with dynamic prior (Schur complement when
    /// `prior_include_intercept` is true; plain centred slopes when false).
    #[allow(clippy::too_many_arguments)]
    fn solve_centred_unstd_prior(
        &self,
        cov: &EwCov,
        cross: &Cross,
        readers: &[usize],
        wj: f64,
        priors: &[Vec<f64>],
        failures: &mut u64,
        work: &mut FactorWork,
        parts: &mut Parts,
        zidx: &[usize],
        ridge: f64,
        tau: f64,
        inc_int: bool,
    ) -> Option<Solved> {
        let kc = zidx.len();
        let kf = kc - 1;
        let mr = readers.len();
        let pairwise = self.cfg.target_gaps == TargetGaps::Pairwise;
        let j0 = readers[0];
        let mut mj = Vec::with_capacity(kf);
        for i in 0..kf {
            let z = zidx[i + 1];
            mj.push(if pairwise {
                cross.mj[j0][z]
            } else {
                cov.mean(z)
            });
        }
        let d0 = wj + tau;
        let w_ratio = wj / d0;
        let alpha = if inc_int { w_ratio * tau } else { 0.0 };
        let shift = wj * ridge + tau;

        let Parts {
            a: asub,
            b: bsub,
            sol,
            out,
            pos,
            kept_s,
            mean,
            quads,
            ..
        } = &mut *parts;

        asub.clear();
        asub.resize(kf * kf, 0.0);
        for i in 0..kf {
            let zi = zidx[i + 1];
            for l in 0..=i {
                let mut v = wj * cov.cov(zi, zidx[l + 1]);
                if inc_int {
                    v += alpha * mj[i] * mj[l];
                }
                asub[i * kf + l] = v;
            }
            asub[i * kf + i] += shift;
        }

        bsub.clear();
        bsub.resize(kf * mr, 0.0);
        for (jj, &j) in readers.iter().enumerate() {
            let bp = &priors[j];
            let bp0 = bp[0];
            for i in 0..kf {
                let z = zidx[i + 1];
                let mut v = wj * cross.c[j][z] + tau * bp[z];
                if inc_int {
                    v += alpha * (cross.my[j] - bp0) * mj[i];
                }
                if let Some(c0) = &self.cfg.coef_prior {
                    v += wj * ridge * c0[j][z];
                }
                bsub[jj * kf + i] = v;
            }
        }

        let (f, mut ill) = Self::run_solve(failures, work, quads, asub, bsub, kf, mr, None, sol)?;
        let mut proj = None;
        if !ill && let Some(lc) = &self.cfg.constraints {
            let q = lc.n_constraints();
            let mut c_mat = vec![0.0; q * kf];
            for (row_i, row) in lc.matrix.iter().enumerate() {
                c_mat[row_i * kf..(row_i + 1) * kf].copy_from_slice(row);
            }
            match ConstraintProj::from_spd_factor(&f, c_mat, lc.rhs.clone(), kf, q) {
                Some(p) => {
                    for jj in 0..mr {
                        p.project_solution(&mut sol[jj * kf..(jj + 1) * kf]);
                    }
                    proj = Some(p);
                }
                None => {
                    *failures += 1;
                    ill = true;
                }
            }
        }

        out.clear();
        out.resize(kc * mr, if ill { f64::NAN } else { 0.0 });
        if !ill {
            for (jj, &j) in readers.iter().enumerate() {
                let mut dot_mb = 0.0;
                for i in 0..kf {
                    let b_i = sol[jj * kf + i];
                    out[jj * kc + i + 1] = b_i;
                    dot_mb += mj[i] * b_i;
                }
                out[jj * kc] = if inc_int {
                    (wj * (cross.my[j] - dot_mb) + tau * priors[j][0]) / d0
                } else {
                    cross.my[j] - dot_mb
                };
            }
        }

        mean.clear();
        if inc_int {
            mean.extend(mj.iter().map(|&m_i| w_ratio * m_i));
        } else {
            mean.extend_from_slice(&mj);
        }
        let edf0 = if ill {
            f64::NAN
        } else if inc_int {
            let q_form = match &proj {
                Some(p) => p.projected_quad_spd(&f, mean),
                None => f.quad_forms(mean, kf, 1)[0],
            };
            (w_ratio - tau * q_form).clamp(0.0, 1.0)
        } else {
            1.0
        };

        kept_s.clear();
        kept_s.resize(kf, 1.0);
        pos.clear();
        pos.extend(1..=kf);

        Some(Solved {
            sol: std::mem::take(out),
            factor: Some(f),
            proj,
            a: std::mem::take(asub),
            keep: std::mem::take(pos),
            s: std::mem::take(kept_s),
            mean: std::mem::take(mean),
            centred: true,
            shift,
            scale: wj,
            ill,
            intercept_weight: inc_int.then_some(w_ratio),
            edf0: Some(edf0),
            lam_intercept: None,
        })
    }

    /// Centred standardized solve with dynamic prior in standardized centered
    /// coordinates `z_tilde = [1, (x - m)/s]`.
    #[allow(clippy::too_many_arguments)]
    fn solve_centred_std_prior(
        &self,
        cov: &EwCov,
        cross: &Cross,
        readers: &[usize],
        wj: f64,
        priors: &[Vec<f64>],
        failures: &mut u64,
        work: &mut FactorWork,
        parts: &mut Parts,
        zidx: &[usize],
        ridge: f64,
        tau: f64,
        inc_int: bool,
    ) -> Option<Solved> {
        let kc = zidx.len();
        let kf = kc - 1;
        let mr = readers.len();
        let pairwise = self.cfg.target_gaps == TargetGaps::Pairwise;
        let j0 = readers[0];
        let mut mj = Vec::with_capacity(kf);
        for i in 0..kf {
            let z = zidx[i + 1];
            mj.push(if pairwise {
                cross.mj[j0][z]
            } else {
                cov.mean(z)
            });
        }
        let d0 = wj + tau;
        let w_ratio = wj / d0;
        let shift = wj * ridge + tau;

        let Parts {
            scale: s,
            cols: keep,
            a: asub,
            b: bsub,
            sol,
            out,
            pos,
            kept_s,
            mean,
            quads,
            ..
        } = &mut *parts;

        let var = |i: usize| cov.cov(zidx[i + 1], zidx[i + 1]);
        s.clear();
        s.extend((0..kf).map(|i| var(i).max(0.0).sqrt()));
        keep.clear();
        keep.extend(
            (0..kf)
                .filter(|&i| crate::variance_is_usable(var(i), cov.raw(zidx[i + 1], zidx[i + 1]))),
        );
        let kk = keep.len();
        let mut is_kept = vec![false; kf];
        for &i in keep.iter() {
            is_kept[i] = true;
        }

        out.clear();
        out.resize(kc * mr, 0.0);
        // Dropped (zero-variance) slopes inherit the parent prior directly.
        for (jj, &j) in readers.iter().enumerate() {
            let bp = &priors[j];
            for (i, &kept) in is_kept.iter().enumerate() {
                if !kept {
                    out[jj * kc + i + 1] = bp[zidx[i + 1]];
                }
            }
        }

        let mut factor = None;
        let mut proj = None;
        let mut ill = false;

        if kk == 0 {
            if let Some(lc) = &self.cfg.constraints {
                for jj in 0..mr {
                    if !project_slopes_linear(lc, kf, &mut out[jj * kc + 1..(jj + 1) * kc]) {
                        *failures += 1;
                        ill = true;
                        out.fill(f64::NAN);
                        break;
                    }
                }
            }
        } else {
            asub.clear();
            asub.resize(kk * kk, 0.0);
            for (i2, &i) in keep.iter().enumerate() {
                let zi = zidx[i + 1];
                for (j2, &l) in keep[..=i2].iter().enumerate() {
                    asub[i2 * kk + j2] = wj * cov.cov(zi, zidx[l + 1]) / (s[i] * s[l]);
                }
            }
            for i2 in 0..kk {
                asub[i2 * kk + i2] += shift;
            }
            bsub.clear();
            bsub.resize(kk * mr, 0.0);
            for (jj, &j) in readers.iter().enumerate() {
                let bp = &priors[j];
                for (i2, &i) in keep.iter().enumerate() {
                    let z = zidx[i + 1];
                    let mut v = wj * cross.c[j][z] / s[i] + tau * bp[z] * s[i];
                    if let Some(c0) = &self.cfg.coef_prior {
                        v += wj * ridge * c0[j][z] * s[i];
                    }
                    bsub[jj * kk + i2] = v;
                }
            }
            let (f, past) = Self::run_solve(failures, work, quads, asub, bsub, kk, mr, None, sol)?;
            ill = past;
            if !ill && let Some(lc) = &self.cfg.constraints {
                let q = lc.n_constraints();
                let mut c_mat = vec![0.0; q * kk];
                for (row_i, row) in lc.matrix.iter().enumerate() {
                    for (i2, &i) in keep.iter().enumerate() {
                        c_mat[row_i * kk + i2] = row[i] / s[i];
                    }
                }
                if kk == kf {
                    match ConstraintProj::from_spd_factor(&f, c_mat, lc.rhs.clone(), kk, q) {
                        Some(p) => {
                            for jj in 0..mr {
                                p.project_solution(&mut sol[jj * kk..(jj + 1) * kk]);
                            }
                            proj = Some(p);
                        }
                        None => {
                            *failures += 1;
                            ill = true;
                            out.fill(f64::NAN);
                        }
                    }
                } else {
                    // Some columns were dropped: adjust RHS per reader for the
                    // fixed dropped slopes (`bp[d+1]`).
                    for (jj, &j) in readers.iter().enumerate() {
                        let bp = &priors[j];
                        let mut rhs_adj = lc.rhs.clone();
                        for (row_i, row) in lc.matrix.iter().enumerate() {
                            for (i, &kept) in is_kept.iter().enumerate() {
                                if !kept {
                                    rhs_adj[row_i] -= row[i] * bp[zidx[i + 1]];
                                }
                            }
                        }
                        match ConstraintProj::from_spd_factor(&f, c_mat.clone(), rhs_adj, kk, q) {
                            Some(p) => {
                                p.project_solution(&mut sol[jj * kk..(jj + 1) * kk]);
                                if jj == 0 {
                                    proj = Some(p);
                                }
                            }
                            None => {
                                *failures += 1;
                                ill = true;
                                out.fill(f64::NAN);
                                break;
                            }
                        }
                    }
                }
            }
            if !ill {
                for jj in 0..mr {
                    for (i2, &i) in keep.iter().enumerate() {
                        out[jj * kc + i + 1] = sol[jj * kk + i2] / s[i];
                    }
                }
            }
            factor = Some(f);
        }

        if !ill {
            for (jj, &j) in readers.iter().enumerate() {
                let bp = &priors[j];
                let mut dot_mb = 0.0;
                let mut dot_mbp = 0.0;
                for i in 0..kf {
                    dot_mb += mj[i] * out[jj * kc + i + 1];
                    dot_mbp += mj[i] * bp[zidx[i + 1]];
                }
                out[jj * kc] = if inc_int {
                    let b0_tilde = (wj * cross.my[j] + tau * (bp[0] + dot_mbp)) / d0;
                    b0_tilde - dot_mb
                } else {
                    cross.my[j] - dot_mb
                };
            }
        }

        let a = if kk > 0 {
            std::mem::take(asub)
        } else {
            Vec::new()
        };
        mean.clear();
        mean.extend(keep.iter().map(|&i| mj[i]));
        kept_s.clear();
        kept_s.extend(keep.iter().map(|&i| s[i]));
        pos.clear();
        pos.extend(keep.iter().map(|&i| i + 1));

        Some(Solved {
            sol: std::mem::take(out),
            factor,
            proj,
            a,
            keep: std::mem::take(pos),
            s: std::mem::take(kept_s),
            mean: std::mem::take(mean),
            centred: true,
            shift,
            scale: wj,
            ill,
            intercept_weight: inc_int.then_some(w_ratio),
            edf0: Some(if inc_int { w_ratio } else { 1.0 }),
            lam_intercept: None,
        })
    }

    /// Sum-scale (`ridge_scale = true`) solve with dynamic prior.
    #[allow(clippy::too_many_arguments)]
    fn solve_ridge_scale_prior(
        &self,
        cov: &EwCov,
        readers: &[usize],
        wj: f64,
        r_raw: &[Vec<f64>],
        priors: &[Vec<f64>],
        failures: &mut u64,
        work: &mut FactorWork,
        parts: &mut Parts,
        zidx: &[usize],
        ridge: f64,
        tau: f64,
        inc_int: bool,
    ) -> Option<Solved> {
        let kc = zidx.len();
        let mr = readers.len();
        let ps = cov.prior_scale();
        let pool_slot = |ai: usize| !self.cfg.fit_intercept || inc_int || ai > 0;

        parts.a.clear();
        parts.a.resize(kc * kc, 0.0);
        for (ai, &zi) in zidx.iter().enumerate() {
            for (aj, &zj) in zidx.iter().enumerate() {
                parts.a[ai * kc + aj] = wj * cov.raw(zi, zj);
            }
            parts.a[ai * kc + ai] += ps * ridge + if pool_slot(ai) { tau } else { 0.0 };
        }
        parts.b.clear();
        parts.b.resize(kc * mr, 0.0);
        for (jj, &j) in readers.iter().enumerate() {
            let bp = &priors[j];
            for (ai, &zi) in zidx.iter().enumerate() {
                let mut v = wj * r_raw[jj][zi];
                if let Some(c0) = &self.cfg.coef_prior {
                    v += ps * ridge * c0[j][zi];
                }
                if pool_slot(ai) {
                    v += tau * bp[zi];
                }
                parts.b[jj * kc + ai] = v;
            }
        }
        let mut solved = Self::solve_raw(
            failures,
            work,
            parts,
            kc,
            mr,
            ps * ridge + tau,
            wj,
            self.cfg.constraints.as_ref(),
            self.cfg.fit_intercept,
        )?;
        solved.lam_intercept = (self.cfg.fit_intercept && !inc_int).then_some(ps * ridge);
        solved.edf0 = Some(0.0);
        Some(solved)
    }

    /// Through-the-origin unstandardized solve with dynamic prior.
    #[allow(clippy::too_many_arguments)]
    fn solve_origin_unstd_prior(
        &self,
        cov: &EwCov,
        readers: &[usize],
        wj: f64,
        r_raw: &[Vec<f64>],
        priors: &[Vec<f64>],
        failures: &mut u64,
        work: &mut FactorWork,
        parts: &mut Parts,
        zidx: &[usize],
        ridge: f64,
        tau: f64,
    ) -> Option<Solved> {
        let kc = zidx.len();
        let mr = readers.len();
        let shift = wj * ridge + tau;
        parts.a.clear();
        parts.a.resize(kc * kc, 0.0);
        for (ai, &zi) in zidx.iter().enumerate() {
            for (aj, &zj) in zidx.iter().enumerate() {
                parts.a[ai * kc + aj] = wj * cov.raw(zi, zj);
            }
            parts.a[ai * kc + ai] += shift;
        }
        parts.b.clear();
        parts.b.resize(kc * mr, 0.0);
        for (jj, &j) in readers.iter().enumerate() {
            let bp = &priors[j];
            for (ai, &zi) in zidx.iter().enumerate() {
                let mut v = wj * r_raw[jj][zi] + tau * bp[zi];
                if let Some(c0) = &self.cfg.coef_prior {
                    v += wj * ridge * c0[j][zi];
                }
                parts.b[jj * kc + ai] = v;
            }
        }
        let mut solved = Self::solve_raw(
            failures,
            work,
            parts,
            kc,
            mr,
            shift,
            wj,
            self.cfg.constraints.as_ref(),
            false,
        )?;
        solved.edf0 = Some(0.0);
        Some(solved)
    }

    /// Through-the-origin standardized solve with dynamic prior.
    #[allow(clippy::too_many_arguments)]
    fn solve_origin_std_prior(
        &self,
        cov: &EwCov,
        readers: &[usize],
        wj: f64,
        r_raw: &[Vec<f64>],
        priors: &[Vec<f64>],
        failures: &mut u64,
        work: &mut FactorWork,
        parts: &mut Parts,
        zidx: &[usize],
        ridge: f64,
        tau: f64,
    ) -> Option<Solved> {
        let kc = zidx.len();
        let mr = readers.len();
        let shift = wj * ridge + tau;
        let Parts {
            scale,
            cols,
            a,
            bs,
            sol,
            out,
            pos,
            kept_s,
            mean,
            quads,
            ..
        } = &mut *parts;

        scale.clear();
        scale.extend((0..kc).map(|i| cov.raw(zidx[i], zidx[i]).max(0.0).sqrt()));
        cols.clear();
        cols.extend((0..kc).filter(|&i| scale[i] > 0.0));
        let kk = cols.len();
        let mut is_kept = vec![false; kc];
        for &i in cols.iter() {
            is_kept[i] = true;
        }

        out.clear();
        out.resize(kc * mr, 0.0);
        for (jj, &j) in readers.iter().enumerate() {
            let bp = &priors[j];
            for (i, &kept) in is_kept.iter().enumerate() {
                if !kept {
                    out[jj * kc + i] = bp[zidx[i]];
                }
            }
        }

        if kk == 0 {
            let mut ill = false;
            if let Some(lc) = &self.cfg.constraints {
                for jj in 0..mr {
                    if !project_slopes_linear(lc, kc, &mut out[jj * kc..(jj + 1) * kc]) {
                        *failures += 1;
                        ill = true;
                        out.fill(f64::NAN);
                        break;
                    }
                }
            }
            return Some(Solved {
                sol: std::mem::take(out),
                factor: None,
                proj: None,
                a: Vec::new(),
                keep: Vec::new(),
                s: Vec::new(),
                mean: Vec::new(),
                centred: false,
                shift,
                scale: wj,
                ill,
                intercept_weight: None,
                edf0: Some(0.0),
                lam_intercept: None,
            });
        }

        a.clear();
        a.resize(kk * kk, 0.0);
        for (i2, &i) in cols.iter().enumerate() {
            for (j2, &l) in cols[..=i2].iter().enumerate() {
                a[i2 * kk + j2] = wj * cov.raw(zidx[i], zidx[l]) / (scale[i] * scale[l]);
            }
        }
        for i2 in 0..kk {
            a[i2 * kk + i2] += shift;
        }
        bs.clear();
        bs.resize(kk * mr, 0.0);
        for (jj, &j) in readers.iter().enumerate() {
            let bp = &priors[j];
            for (i2, &i) in cols.iter().enumerate() {
                let zi = zidx[i];
                let mut v = wj * r_raw[jj][zi] / scale[i] + tau * bp[zi] * scale[i];
                if let Some(c0) = &self.cfg.coef_prior {
                    v += wj * ridge * c0[j][zi] * scale[i];
                }
                bs[jj * kk + i2] = v;
            }
        }

        let (factor, mut ill) = Self::run_solve(failures, work, quads, a, bs, kk, mr, None, sol)?;
        let mut proj = None;
        if !ill && let Some(lc) = &self.cfg.constraints {
            let q = lc.n_constraints();
            let mut c_mat = vec![0.0; q * kk];
            for (row_i, row) in lc.matrix.iter().enumerate() {
                for (i2, &i) in cols.iter().enumerate() {
                    c_mat[row_i * kk + i2] = row[i] / scale[i];
                }
            }
            if kk == kc {
                match ConstraintProj::from_spd_factor(&factor, c_mat, lc.rhs.clone(), kk, q) {
                    Some(p) => {
                        for jj in 0..mr {
                            p.project_solution(&mut sol[jj * kk..(jj + 1) * kk]);
                        }
                        proj = Some(p);
                    }
                    None => {
                        *failures += 1;
                        ill = true;
                        out.fill(f64::NAN);
                    }
                }
            } else {
                for (jj, &j) in readers.iter().enumerate() {
                    let bp = &priors[j];
                    let mut rhs_adj = lc.rhs.clone();
                    for (row_i, row) in lc.matrix.iter().enumerate() {
                        for (i, &kept) in is_kept.iter().enumerate() {
                            if !kept {
                                rhs_adj[row_i] -= row[i] * bp[zidx[i]];
                            }
                        }
                    }
                    match ConstraintProj::from_spd_factor(&factor, c_mat.clone(), rhs_adj, kk, q) {
                        Some(p) => {
                            p.project_solution(&mut sol[jj * kk..(jj + 1) * kk]);
                            if jj == 0 {
                                proj = Some(p);
                            }
                        }
                        None => {
                            *failures += 1;
                            ill = true;
                            out.fill(f64::NAN);
                            break;
                        }
                    }
                }
            }
        }
        if !ill {
            for jj in 0..mr {
                for (i2, &i) in cols.iter().enumerate() {
                    out[jj * kc + i] = sol[jj * kk + i2] / scale[i];
                }
            }
        }
        kept_s.clear();
        kept_s.extend(cols.iter().map(|&i| scale[i]));
        mean.clear();
        mean.resize(kk, 0.0);
        pos.clear();
        pos.extend(cols.iter().copied());

        Some(Solved {
            sol: std::mem::take(out),
            factor: Some(factor),
            proj,
            a: std::mem::take(a),
            keep: std::mem::take(pos),
            s: std::mem::take(kept_s),
            mean: std::mem::take(mean),
            centred: false,
            shift,
            scale: wj,
            ill,
            intercept_weight: None,
            edf0: Some(0.0),
            lam_intercept: None,
        })
    }

    /// Bounded / fixed-sum QP solve with dynamic prior for a single target `j`
    /// that has a valid finite `bp`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn solve_bounded_with_prior(
        &self,
        cov: &EwCov,
        cross: &Cross,
        j: usize,
        wj: f64,
        r_raw_j: &[f64],
        bp: &[f64],
        slot: usize,
        ridge: f64,
        con: &Constraint,
        centred: bool,
        failures: &mut u64,
        parts: &mut Parts,
        qp: &mut QpWork,
        beta_slot: &mut [f64],
    ) {
        let tau = self.cfg.prior_weight.expect("prior_weight active");
        let inc_int = self.cfg.prior_include_intercept;
        let kf = self.cfg.n_features;
        let kc = self.cfg.k_total();
        let off = usize::from(self.cfg.fit_intercept);
        let prev: Option<&[f64]> = self
            .beta
            .as_deref()
            .and_then(|pb| pb.get(slot))
            .map(Vec::as_slice);

        // Zero child weight under mean-scale ridge -> Euclidean projection of bp.
        if !self.cfg.ridge_scale && wj <= 0.0 {
            if !self.cold_start_beta(bp, beta_slot) {
                beta_slot.fill(f64::NAN);
            }
            return;
        }

        let mut proj_scratch = ProjScratch::default();

        if centred {
            let pairwise = self.cfg.target_gaps == TargetGaps::Pairwise;
            let mut mj = Vec::with_capacity(kf);
            for i in 0..kf {
                mj.push(if pairwise {
                    cross.mj[j][i + 1]
                } else {
                    cov.mean(i + 1)
                });
            }
            let d0 = wj + tau;
            let alpha = if inc_int { (wj * tau) / d0 } else { 0.0 };
            let shift = wj * ridge + tau;

            if !self.cfg.standardize {
                parts.a.clear();
                parts.a.resize(kf * kf, 0.0);
                for i in 0..kf {
                    for l in 0..=i {
                        let mut v = wj * cov.cov(i + 1, l + 1);
                        if inc_int {
                            v += alpha * mj[i] * mj[l];
                        }
                        parts.a[i * kf + l] = v;
                    }
                    parts.a[i * kf + i] += shift;
                }
                let mut rhs = vec![0.0; kf];
                for i in 0..kf {
                    let mut v = wj * cross.c[j][i + 1] + tau * bp[i + 1];
                    if inc_int {
                        v += alpha * (cross.my[j] - bp[0]) * mj[i];
                    }
                    if let Some(c0) = &self.cfg.coef_prior {
                        v += wj * ridge * c0[j][i + 1];
                    }
                    rhs[i] = v;
                }
                let mut init = init_slopes(prev, Some(bp), off, kf);
                con.project(&mut init, None, &mut proj_scratch);
                let eq_a = vec![1.0; kf];
                let eq = con.sum.map(|s| (eq_a.as_slice(), s));
                let out = solve_qp(
                    &parts.a,
                    &rhs,
                    kf,
                    &con.lo,
                    &con.hi,
                    eq,
                    &init,
                    qp,
                    &mut beta_slot[1..=kf],
                );
                finish_centred_qp(
                    out.status,
                    prev,
                    beta_slot,
                    &mj,
                    cross.my[j],
                    wj,
                    tau,
                    bp[0],
                    inc_int,
                    failures,
                );
            } else {
                // Standardized centered QP: transform slopes via `tilde_i = d_i * beta_i`,
                // where `d_i = s_i` for usable-variance features and `d_i = 1.0` for
                // zero-variance features (prior-only term `tau * beta_i = tau * bp_i`).
                let mut d = vec![1.0; kf];
                let mut usable = vec![false; kf];
                for i in 0..kf {
                    let v = cov.cov(i + 1, i + 1);
                    if crate::variance_is_usable(v, cov.raw(i + 1, i + 1)) {
                        d[i] = v.max(0.0).sqrt();
                        usable[i] = true;
                    }
                }
                parts.a.clear();
                parts.a.resize(kf * kf, 0.0);
                let mut rhs = vec![0.0; kf];
                for i in 0..kf {
                    if usable[i] {
                        for l in 0..=i {
                            if usable[l] {
                                parts.a[i * kf + l] = wj * cov.cov(i + 1, l + 1) / (d[i] * d[l]);
                            }
                        }
                        parts.a[i * kf + i] += shift;
                        let mut v = wj * cross.c[j][i + 1] / d[i] + tau * bp[i + 1] * d[i];
                        if let Some(c0) = &self.cfg.coef_prior {
                            v += wj * ridge * c0[j][i + 1] * d[i];
                        }
                        rhs[i] = v;
                    } else {
                        parts.a[i * kf + i] = tau;
                        rhs[i] = tau * bp[i + 1];
                    }
                }
                let lo: Vec<f64> = con.lo.iter().zip(&d).map(|(&l, &di)| l * di).collect();
                let hi: Vec<f64> = con.hi.iter().zip(&d).map(|(&h, &di)| h * di).collect();
                let eq_a: Vec<f64> = d.iter().map(|&di| 1.0 / di).collect();
                let eq = con.sum.map(|s| (eq_a.as_slice(), s));
                let mut init = init_slopes(prev, Some(bp), off, kf);
                con.project(&mut init, None, &mut proj_scratch);
                for (v, &di) in init.iter_mut().zip(&d) {
                    *v *= di;
                }
                let mut sol_s = vec![0.0; kf];
                let out = solve_qp(&parts.a, &rhs, kf, &lo, &hi, eq, &init, qp, &mut sol_s);
                match out.status {
                    QpStatus::Converged | QpStatus::MaxIter => {
                        if out.status == QpStatus::MaxIter {
                            *failures += 1;
                        }
                        let mut dot_mb = 0.0;
                        let mut dot_mbp = 0.0;
                        for i in 0..kf {
                            let b_i = sol_s[i] / d[i];
                            beta_slot[i + 1] = b_i;
                            dot_mb += mj[i] * b_i;
                            dot_mbp += mj[i] * bp[i + 1];
                        }
                        beta_slot[0] = if inc_int {
                            let b0_tilde = (wj * cross.my[j] + tau * (bp[0] + dot_mbp)) / d0;
                            b0_tilde - dot_mb
                        } else {
                            cross.my[j] - dot_mb
                        };
                    }
                    QpStatus::Failed => {
                        *failures += 1;
                        fallback(prev, beta_slot);
                    }
                }
            }
            return;
        }

        if !self.cfg.ridge_scale {
            // Through the origin, unstandardized or standardized.
            let shift = wj * ridge + tau;
            if !self.cfg.standardize {
                parts.a.clear();
                parts.a.resize(kf * kf, 0.0);
                for i in 0..kf {
                    for l in 0..=i {
                        parts.a[i * kf + l] = wj * cov.raw(i, l);
                    }
                    parts.a[i * kf + i] += shift;
                }
                let mut rhs = vec![0.0; kf];
                for i in 0..kf {
                    let mut v = wj * r_raw_j[i] + tau * bp[i];
                    if let Some(c0) = &self.cfg.coef_prior {
                        v += wj * ridge * c0[j][i];
                    }
                    rhs[i] = v;
                }
                let mut init = init_slopes(prev, Some(bp), 0, kf);
                con.project(&mut init, None, &mut proj_scratch);
                let eq_a = vec![1.0; kf];
                let eq = con.sum.map(|s| (eq_a.as_slice(), s));
                let out = solve_qp(
                    &parts.a, &rhs, kf, &con.lo, &con.hi, eq, &init, qp, beta_slot,
                );
                finish_origin_qp(out.status, prev, beta_slot, failures);
            } else {
                let mut d = vec![1.0; kf];
                let mut usable = vec![false; kf];
                for i in 0..kf {
                    let s = cov.raw(i, i).max(0.0).sqrt();
                    if s > 0.0 {
                        d[i] = s;
                        usable[i] = true;
                    }
                }
                parts.a.clear();
                parts.a.resize(kf * kf, 0.0);
                let mut rhs = vec![0.0; kf];
                for i in 0..kf {
                    if usable[i] {
                        for l in 0..=i {
                            if usable[l] {
                                parts.a[i * kf + l] = wj * cov.raw(i, l) / (d[i] * d[l]);
                            }
                        }
                        parts.a[i * kf + i] += shift;
                        let mut v = wj * r_raw_j[i] / d[i] + tau * bp[i] * d[i];
                        if let Some(c0) = &self.cfg.coef_prior {
                            v += wj * ridge * c0[j][i] * d[i];
                        }
                        rhs[i] = v;
                    } else {
                        parts.a[i * kf + i] = tau;
                        rhs[i] = tau * bp[i];
                    }
                }
                let lo: Vec<f64> = con.lo.iter().zip(&d).map(|(&l, &di)| l * di).collect();
                let hi: Vec<f64> = con.hi.iter().zip(&d).map(|(&h, &di)| h * di).collect();
                let eq_a: Vec<f64> = d.iter().map(|&di| 1.0 / di).collect();
                let eq = con.sum.map(|s| (eq_a.as_slice(), s));
                let mut init = init_slopes(prev, Some(bp), 0, kf);
                con.project(&mut init, None, &mut proj_scratch);
                for (v, &di) in init.iter_mut().zip(&d) {
                    *v *= di;
                }
                let mut sol_s = vec![0.0; kf];
                let out = solve_qp(&parts.a, &rhs, kf, &lo, &hi, eq, &init, qp, &mut sol_s);
                match out.status {
                    QpStatus::Converged | QpStatus::MaxIter => {
                        if out.status == QpStatus::MaxIter {
                            *failures += 1;
                        }
                        for i in 0..kf {
                            beta_slot[i] = sol_s[i] / d[i];
                        }
                    }
                    QpStatus::Failed => {
                        *failures += 1;
                        fallback(prev, beta_slot);
                    }
                }
            }
            return;
        }

        // `ridge_scale = true`: full `kc x kc` QP with dynamic prior.
        let ps = cov.prior_scale();
        let pool_slot = |i: usize| !self.cfg.fit_intercept || inc_int || i > 0;
        parts.a.clear();
        parts.a.resize(kc * kc, 0.0);
        for i in 0..kc {
            for l in 0..=i {
                parts.a[i * kc + l] = wj * cov.raw(i, l);
            }
            parts.a[i * kc + i] += ps * ridge + if pool_slot(i) { tau } else { 0.0 };
        }
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
        for i in 0..kc {
            let mut v = wj * r_raw_j[i];
            if let Some(c0) = &self.cfg.coef_prior {
                v += ps * ridge * c0[j][i];
            }
            if pool_slot(i) {
                v += tau * bp[i];
            }
            rhs[i] = v;
        }
        let mut init = match prev {
            Some(p) if p.len() == kc && p.iter().all(|v| v.is_finite()) => p.to_vec(),
            _ => bp.to_vec(),
        };
        con.project(&mut init[off..], None, &mut proj_scratch);
        let out = solve_qp(&parts.a, &rhs, kc, &lo, &hi, eq, &init, qp, beta_slot);
        finish_origin_qp(out.status, prev, beta_slot, failures);
    }
}

/// Euclidean projection of `slopes` (length `kf`) onto linear equality
/// constraints `R beta_s = r`.
fn project_slopes_linear(lc: &LinearConstraints, kf: usize, slopes: &mut [f64]) -> bool {
    let mut ident = vec![0.0; kf * kf];
    for i in 0..kf {
        ident[i * kf + i] = 1.0;
    }
    let Some(factor) = SpdFactor::of(&ident, kf) else {
        return false;
    };
    let q = lc.n_constraints();
    let mut c_mat = vec![0.0; q * kf];
    for (row_i, row) in lc.matrix.iter().enumerate() {
        c_mat[row_i * kf..(row_i + 1) * kf].copy_from_slice(row);
    }
    let Some(proj) = ConstraintProj::from_spd_factor(&factor, c_mat, lc.rhs.clone(), kf, q) else {
        return false;
    };
    proj.project_solution(slopes);
    slopes.iter().all(|v| v.is_finite())
}

fn init_slopes(prev: Option<&[f64]>, bp: Option<&[f64]>, off: usize, kf: usize) -> Vec<f64> {
    if let Some(p) = prev
        && p.len() >= off + kf
        && p[off..off + kf].iter().all(|v| v.is_finite())
    {
        return p[off..off + kf].to_vec();
    }
    if let Some(p) = bp
        && p.len() >= off + kf
        && p[off..off + kf].iter().all(|v| v.is_finite())
    {
        return p[off..off + kf].to_vec();
    }
    vec![0.0; kf]
}

#[allow(clippy::too_many_arguments)]
fn finish_centred_qp(
    status: QpStatus,
    prev: Option<&[f64]>,
    beta_slot: &mut [f64],
    mj: &[f64],
    my: f64,
    wj: f64,
    tau: f64,
    bp0: f64,
    inc_int: bool,
    failures: &mut u64,
) {
    match status {
        QpStatus::Converged | QpStatus::MaxIter => {
            if status == QpStatus::MaxIter {
                *failures += 1;
            }
            let dot_mb: f64 = mj
                .iter()
                .zip(beta_slot[1..].iter())
                .map(|(m, b)| m * b)
                .sum();
            beta_slot[0] = if inc_int {
                (wj * (my - dot_mb) + tau * bp0) / (wj + tau)
            } else {
                my - dot_mb
            };
        }
        QpStatus::Failed => {
            *failures += 1;
            fallback(prev, beta_slot);
        }
    }
}

fn finish_origin_qp(
    status: QpStatus,
    prev: Option<&[f64]>,
    beta_slot: &mut [f64],
    failures: &mut u64,
) {
    match status {
        QpStatus::Converged => {}
        QpStatus::MaxIter => *failures += 1,
        QpStatus::Failed => {
            *failures += 1;
            fallback(prev, beta_slot);
        }
    }
}

fn fallback(prev: Option<&[f64]>, dst: &mut [f64]) {
    if let Some(p) = prev {
        dst.copy_from_slice(p);
    } else {
        dst.fill(f64::NAN);
    }
}
