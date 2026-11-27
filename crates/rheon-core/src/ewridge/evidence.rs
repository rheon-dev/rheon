//! Empirical-Bayes (MacKay / Tipping) evidence maximization at solve time for
//! [`super::EwRidge`], with exact posterior predictive standard deviation
//! `pred_std` (matching scikit-learn's `BayesianRidge`).
//!
//! At each solve, for each `(Gram, feature_set)`:
//!
//! 1. Build the normalized Gram `C` (`kk × kk`) once and eigendecompose
//!    `C = V diag(e_norm) Vᵀ` using [`crate::solve::self_adjoint_eigen`].
//! 2. For each reader target `j` with Kish effective sample size `n > 0`:
//!    - Scale eigenvalues `e_m = max(n · e_norm[m], 0)`, cross-vector
//!      `X'y = n · c_xy` (or raw `n · r_xy` through the origin), and
//!      `y'y = n · var(y)` (or `n · (var(y) + mean(y)²)` through the origin).
//!    - Project `vtx = Vᵀ X'y`.
//!    - Cold-start `alpha = 1 / (var(y) + eps)`, `lambda = 1` and run MacKay's
//!      fixed-point iteration in the eigenbasis:
//!      - `c_m = vtx_m / (e_m + lambda / alpha)`
//!      - `coef = V c`
//!      - `sse = max(y'y - sum_m c_m (2 vtx_m - e_m c_m), 0)`
//!      - `gamma = sum_m (alpha e_m) / (lambda + alpha e_m)`
//!      - `lambda = (gamma + 2 lambda_1) / (||c||² + 2 lambda_2)`
//!      - `alpha = (n - gamma + 2 alpha_1) / (sse + 2 alpha_2)`
//!
//!      stopping when `sum_i |coef_i - coef_old_i| < tol` (after iteration 0).
//!    - Recompute final `c`, `coef`, and posterior covariance
//!      `Sigma = V diag(1 / (alpha e_m + lambda)) Vᵀ`.
//! 3. At prediction time, for a feature row `x`, form `u_i = (z_i - mean_i) / s_i`
//!    and report `pred_std = sqrt(1 / alpha + uᵀ Sigma u)`.

use faer::Mat;
use serde::{Deserialize, Serialize};

use crate::gaps::Cross;
use crate::{EwCov, LowerTriangle, TargetMoments};

/// Maximum MacKay fixed-point iterations per solve.
const MAX_EVIDENCE_ITER: usize = 500;

/// L1 coefficient convergence tolerance (matching `BayesianRidge(tol=1e-12)`).
const EVIDENCE_TOL: f64 = 1e-12;

/// Default Gamma hyperprior parameter matching `sklearn.linear_model.BayesianRidge`.
pub const DEFAULT_EVIDENCE_PRIOR: f64 = 1e-6;

/// Gamma hyperpriors `(alpha_1, alpha_2, lambda_1, lambda_2)` for empirical-Bayes
/// evidence maximization on [`super::EwRidge`].
///
/// Defaults to `1e-6` for all four (matching `sklearn.linear_model.BayesianRidge`).
/// Setting all four to `0.0` gives pure ML-II evidence maximization.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct EvidenceCfg {
    pub alpha_1: f64,
    pub alpha_2: f64,
    pub lambda_1: f64,
    pub lambda_2: f64,
}

impl Default for EvidenceCfg {
    fn default() -> Self {
        Self::uniform(DEFAULT_EVIDENCE_PRIOR)
    }
}

impl EvidenceCfg {
    /// Set all four Gamma hyperprior parameters to the same non-negative value `p`.
    pub const fn uniform(p: f64) -> Self {
        Self {
            alpha_1: p,
            alpha_2: p,
            lambda_1: p,
            lambda_2: p,
        }
    }

    /// Pure ML-II evidence maximization (all four Gamma hyperpriors set to `0.0`).
    pub const fn ml2() -> Self {
        Self::uniform(0.0)
    }

    pub fn validate(&self) -> Result<(), String> {
        let mut no = crate::Refusals::new();
        for (name, val) in [
            ("alpha_1", self.alpha_1),
            ("alpha_2", self.alpha_2),
            ("lambda_1", self.lambda_1),
            ("lambda_2", self.lambda_2),
        ] {
            if !(val.is_finite() && val >= 0.0) {
                no.refuse(format!(
                    "ewridge: evidence_prior {name} must be finite and at least 0, but it is \
                     {val}. Set each entry of evidence_prior to 0 for pure ML-II evidence \
                     maximization, or to a finite non-negative Gamma hyperprior such as 1e-6."
                ));
            }
        }
        no.finish()
    }
}

/// Posterior summary kept per output slot from the last evidence solve so
/// `pred_std`, `coef_variance`, and `row_error_inflation` are bit-identical
/// across save/load and between periodic solves.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(super) struct EvidenceSlot {
    /// Noise precision `alpha` (`1 / alpha` is the posterior noise variance).
    pub(super) alpha: f64,
    /// Weight precision `lambda` (effective ridge penalty `lambda / alpha`).
    pub(super) lambda: f64,
    /// Posterior covariance `Sigma = (alpha G + lambda I)⁻¹` in the kept
    /// system's coordinates (`kk × kk`, symmetric, stored as lower triangle).
    pub(super) sigma: LowerTriangle,
    /// Accumulator indices (`z` slots) of kept columns.
    pub(super) z: Vec<usize>,
    /// Per-kept-column scale (`1.0` when unstandardized).
    pub(super) s: Vec<f64>,
    /// Per-kept-column centering mean (`0.0` when uncentred).
    pub(super) mean: Vec<f64>,
    /// Whether the intercept was eliminated by centering.
    pub(super) centred: bool,
}

impl EvidenceSlot {
    pub(super) fn has_shape(&self, k_total: usize) -> bool {
        let kk = self.z.len();
        self.alpha.is_finite()
            && self.alpha > 0.0
            && self.lambda.is_finite()
            && self.lambda >= 0.0
            && self.sigma.len() == kk * kk
            && self.s.len() == kk
            && self.mean.len() == kk
            && self.z.iter().all(|&zi| zi < k_total)
            && self.s.iter().all(|&s| s.is_finite() && s > 0.0)
            && self.mean.iter().all(|m| m.is_finite())
    }

    /// Quadratic form `uᵀ Sigma u` where `u_i = (x_at(z_i) - mean_i) / s_i`.
    pub(super) fn quad_form(&self, x_at: impl Fn(usize) -> f64) -> f64 {
        let kk = self.z.len();
        if kk == 0 {
            return 0.0;
        }
        let mut u = Vec::with_capacity(kk);
        for ((&zi, &s), &mu) in self.z.iter().zip(&self.s).zip(&self.mean) {
            u.push((x_at(zi) - mu) / s);
        }
        self.quad_vec(&u)
    }

    /// Quadratic form `uᵀ Sigma u` for a precomputed coordinate vector `u`.
    pub(super) fn quad_vec(&self, u: &[f64]) -> f64 {
        let kk = self.z.len();
        debug_assert_eq!(u.len(), kk);
        let packed = self.sigma.packed();
        let mut q = 0.0;
        let mut at = 0;
        for i in 0..kk {
            let row = &packed[at..at + i + 1];
            let mut off = 0.0;
            for j in 0..i {
                off += row[j] * u[j];
            }
            q += u[i] * (2.0 * off + row[i] * u[i]);
            at += i + 1;
        }
        if q.is_finite() { q.max(0.0) } else { q }
    }

    /// Posterior predictive standard deviation `sqrt(1 / alpha + uᵀ Sigma u)`.
    pub(super) fn pred_std(&self, x_at: impl Fn(usize) -> f64) -> f64 {
        if !(self.alpha.is_finite() && self.alpha > 0.0) {
            return f64::NAN;
        }
        let q = self.quad_form(x_at);
        if !q.is_finite() {
            return f64::NAN;
        }
        (1.0 / self.alpha + q).sqrt()
    }
}

/// Per-slot evidence state carried by [`super::EwRidge`] when
/// `cfg.evidence.is_some()`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub(super) struct EvidenceState {
    pub(super) slots: Vec<Option<EvidenceSlot>>,
}

/// Output of solving one `(Gram, feature_set)` for one target under evidence
/// maximization.
pub(super) struct EvidenceSolvedTarget {
    pub(super) beta: Vec<f64>,
    pub(super) edf: f64,
    pub(super) support: Vec<f64>,
    pub(super) slot: EvidenceSlot,
}

/// Result of MacKay's fixed-point evidence maximization in the eigenbasis.
struct MacKayResult {
    alpha: f64,
    lambda: f64,
    coef: Vec<f64>,
    gamma: f64,
}

/// Run MacKay's fixed-point evidence maximization in the eigenbasis for one
/// target. Returns `(alpha, lambda, coef, gamma)` where `coef = V c`.
fn mackay_fixed_point(
    e: &[f64],
    v: &Mat<f64>,
    vtx: &[f64],
    yy: f64,
    cyy: f64,
    n: f64,
    ev_cfg: EvidenceCfg,
) -> Option<MacKayResult> {
    let kk = e.len();
    let mut alpha = 1.0 / (cyy.max(0.0) + f64::EPSILON);
    let mut lambda = 1.0;
    if !alpha.is_finite() || alpha <= 0.0 {
        return None;
    }

    let mut c = vec![0.0; kk];
    let mut coef = vec![0.0; kk];
    let mut old_coef = vec![0.0; kk];

    for it in 0..MAX_EVIDENCE_ITER {
        let ratio = lambda / alpha;
        if !ratio.is_finite() || ratio < 0.0 {
            return None;
        }
        for m in 0..kk {
            let denom = e[m] + ratio;
            c[m] = if denom > 0.0 { vtx[m] / denom } else { 0.0 };
        }
        for a in 0..kk {
            let mut sum = 0.0;
            for m in 0..kk {
                sum += v[(a, m)] * c[m];
            }
            coef[a] = sum;
        }
        let coef_norm2: f64 = c.iter().map(|ci| ci * ci).sum();
        let quad: f64 = (0..kk).map(|m| c[m] * (2.0 * vtx[m] - e[m] * c[m])).sum();
        let sse = (yy - quad).max(0.0);
        let gamma: f64 = e
            .iter()
            .map(|&em| {
                let num = alpha * em;
                let den = lambda + num;
                if den > 0.0 { num / den } else { 0.0 }
            })
            .sum();

        let lam_num = gamma + 2.0 * ev_cfg.lambda_1;
        let lam_den = coef_norm2 + 2.0 * ev_cfg.lambda_2;
        let next_lambda = if lam_den > 0.0 && lam_num.is_finite() && lam_num >= 0.0 {
            lam_num / lam_den
        } else if lam_num == 0.0 {
            lambda
        } else {
            // Pure ML-II (`lambda_2 = 0`) with zero coefficients: infinite
            // weight precision shrinks slopes to zero; cap safely.
            1e12_f64.max(lambda)
        };

        let alp_num = (n - gamma) + 2.0 * ev_cfg.alpha_1;
        let alp_den = sse + 2.0 * ev_cfg.alpha_2;
        let next_alpha = if alp_num > 0.0 && alp_den > 0.0 {
            alp_num / alp_den
        } else if alp_num <= 0.0 {
            // Under pairwise gaps or n <= gamma with alpha_1 == 0, keep
            // alpha positive and stop cleanly.
            alpha
        } else {
            // Exact noiseless fit with alpha_2 == 0: cap noise precision
            // finitely so 1/alpha stays finite and near zero.
            (alp_num / (f64::EPSILON * (1.0 + yy))).min(1e30)
        };

        if !next_alpha.is_finite()
            || next_alpha <= 0.0
            || !next_lambda.is_finite()
            || next_lambda < 0.0
        {
            break;
        }
        alpha = next_alpha;
        lambda = next_lambda;

        if it != 0 {
            let diff: f64 = coef.iter().zip(&old_coef).map(|(a, b)| (a - b).abs()).sum();
            if diff < EVIDENCE_TOL {
                break;
            }
        }
        old_coef.copy_from_slice(&coef);
    }

    // Recompute final `c`, `coef`, and `gamma` at the converged `(alpha, lambda)`
    // so coefficients, covariance, and degrees of freedom are mutually exact.
    let ratio = lambda / alpha;
    if !ratio.is_finite() || ratio < 0.0 {
        return None;
    }
    for m in 0..kk {
        let denom = e[m] + ratio;
        c[m] = if denom > 0.0 { vtx[m] / denom } else { 0.0 };
    }
    for a in 0..kk {
        let mut sum = 0.0;
        for m in 0..kk {
            sum += v[(a, m)] * c[m];
        }
        coef[a] = sum;
    }
    let gamma: f64 = e
        .iter()
        .map(|&em| {
            let num = alpha * em;
            let den = lambda + num;
            if den > 0.0 { num / den } else { 0.0 }
        })
        .sum();
    if coef.iter().any(|x| !x.is_finite()) {
        return None;
    }
    Some(MacKayResult {
        alpha,
        lambda,
        coef,
        gamma,
    })
}

/// Solve one `(Gram, feature_set)` for every reader target `j` using MacKay
/// evidence maximization. Returns one `Option<EvidenceSolvedTarget>` per reader
/// in `readers`.
#[allow(clippy::too_many_arguments)]
pub(super) fn solve_evidence_combo(
    cov: &EwCov,
    cross: &Cross,
    tm: Option<&TargetMoments>,
    wj: &[f64],
    readers: &[usize],
    zidx: &[usize],
    fit_intercept: bool,
    standardize: bool,
    pairwise: bool,
    ev_cfg: EvidenceCfg,
    k_total: usize,
) -> Vec<Option<EvidenceSolvedTarget>> {
    let mr = readers.len();
    let mut results: Vec<Option<EvidenceSolvedTarget>> = (0..mr).map(|_| None).collect();
    if cov.n_eff() <= 0.0 {
        return results;
    }

    // Identify kept feature columns and their scales.
    let mut kept_z = Vec::new();
    let mut kept_s = Vec::new();
    if fit_intercept {
        for &zi in &zidx[1..] {
            if standardize {
                let v = cov.var(zi);
                if crate::variance_is_usable(v, cov.raw(zi, zi)) {
                    kept_z.push(zi);
                    kept_s.push(v.sqrt());
                }
            } else {
                kept_z.push(zi);
                kept_s.push(1.0);
            }
        }
    } else {
        for &zi in zidx {
            if standardize {
                let r = cov.raw(zi, zi);
                if r > 0.0 && r.is_finite() {
                    kept_z.push(zi);
                    kept_s.push(r.sqrt());
                }
            } else {
                kept_z.push(zi);
                kept_s.push(1.0);
            }
        }
    }
    let kk = kept_z.len();

    // Eigendecompose the normalized Gram once for all readers sharing this Gram.
    let eigen = if kk > 0 {
        let mut mat = Mat::<f64>::zeros(kk, kk);
        for a in 0..kk {
            let za = kept_z[a];
            let sa = kept_s[a];
            for b in 0..=a {
                let zb = kept_z[b];
                let sb = kept_s[b];
                let val = if fit_intercept {
                    cov.cov(za, zb) / (sa * sb)
                } else {
                    cov.raw(za, zb) / (sa * sb)
                };
                if !val.is_finite() {
                    return results;
                }
                mat[(a, b)] = val;
            }
        }
        let Some((e_norm, v)) = crate::solve::self_adjoint_eigen(mat.as_ref()) else {
            return results;
        };
        Some((e_norm, v))
    } else {
        None
    };

    let own_kish = if pairwise {
        tm.map(|t| t.n_kish(wj))
    } else {
        None
    };

    let mut e_scaled = vec![0.0; kk];
    let mut xy = vec![0.0; kk];
    let mut vtx = vec![0.0; kk];

    for (jj, &j) in readers.iter().enumerate() {
        let n = if pairwise {
            own_kish.as_ref().and_then(|v| v[j])
        } else {
            cov.n_kish()
        };
        let Some(n) = n.filter(|&n| n.is_finite() && n > 0.0) else {
            continue;
        };
        let cyy = tm.map_or(0.0, |t| t.vars()[j].max(0.0));
        let my = cross.my[j];
        let yy = if fit_intercept {
            n * cyy
        } else {
            n * (cyy + my * my)
        };
        if !yy.is_finite() {
            continue;
        }

        let mut beta = vec![0.0; k_total];
        let mut support = vec![f64::NAN; k_total];
        let slope_zidx = if fit_intercept { &zidx[1..] } else { zidx };
        for &zi in slope_zidx {
            support[zi] = 0.0;
        }

        if kk == 0 {
            let alp_num = n + 2.0 * ev_cfg.alpha_1;
            let alp_den = yy + 2.0 * ev_cfg.alpha_2;
            let alpha = if alp_den > 0.0 {
                alp_num / alp_den
            } else {
                (alp_num / (f64::EPSILON * (1.0 + yy))).min(1e30)
            };
            if !(alpha.is_finite() && alpha > 0.0) {
                continue;
            }
            if fit_intercept {
                beta[0] = my;
            }
            let edf = if fit_intercept { 1.0 } else { 0.0 };
            results[jj] = Some(EvidenceSolvedTarget {
                beta,
                edf,
                support,
                slot: EvidenceSlot {
                    alpha,
                    lambda: 1.0,
                    sigma: LowerTriangle::zeros(0),
                    z: Vec::new(),
                    s: Vec::new(),
                    mean: Vec::new(),
                    centred: fit_intercept,
                },
            });
            continue;
        }

        let (e_norm, v) = eigen.as_ref().expect("kk > 0");
        for m in 0..kk {
            e_scaled[m] = (n * e_norm[m]).max(0.0);
        }
        let mut xy_ok = true;
        for a in 0..kk {
            let za = kept_z[a];
            let sa = kept_s[a];
            let c_xy = if fit_intercept {
                cross.c[j][za]
            } else {
                cross.c[j][za] + cross.mj[j][za] * my
            };
            let val = n * c_xy / sa;
            if !val.is_finite() {
                xy_ok = false;
                break;
            }
            xy[a] = val;
        }
        if !xy_ok {
            continue;
        }
        for m in 0..kk {
            let mut sum = 0.0;
            for a in 0..kk {
                sum += v[(a, m)] * xy[a];
            }
            vtx[m] = sum;
        }

        let Some(MacKayResult {
            alpha,
            lambda,
            coef,
            gamma,
        }) = mackay_fixed_point(&e_scaled, v, &vtx, yy, cyy, n, ev_cfg)
        else {
            continue;
        };

        // Unscale slopes and recover intercept.
        for a in 0..kk {
            let za = kept_z[a];
            beta[za] = if standardize {
                coef[a] / kept_s[a]
            } else {
                coef[a]
            };
        }
        let mut kept_means = Vec::with_capacity(kk);
        if fit_intercept {
            let mut b0 = my;
            for &za in kept_z.iter().take(kk) {
                let mj = if pairwise {
                    cross.mj[j][za]
                } else {
                    cov.mean(za)
                };
                kept_means.push(mj);
                b0 -= mj * beta[za];
            }
            beta[0] = b0;
        } else {
            kept_means.resize(kk, 0.0);
        }
        if beta.iter().any(|x| !x.is_finite()) {
            continue;
        }

        // Build posterior covariance `Sigma = V diag(1 / (alpha * e_m + lambda)) Vᵀ`.
        let mut inv_d = vec![0.0; kk];
        let mut shrink_d = vec![0.0; kk];
        for m in 0..kk {
            let den = alpha * e_scaled[m] + lambda;
            inv_d[m] = if den > 0.0 { 1.0 / den } else { 0.0 };
            shrink_d[m] = if den > 0.0 {
                (alpha * e_scaled[m]) / den
            } else {
                0.0
            };
        }
        let mut sigma = LowerTriangle::zeros(kk);
        let packed = sigma.packed_mut();
        let mut at = 0;
        for a in 0..kk {
            for b in 0..=a {
                let mut sum = 0.0;
                for m in 0..kk {
                    sum += v[(a, m)] * v[(b, m)] * inv_d[m];
                }
                packed[at + b] = sum;
            }
            at += a + 1;
        }
        if packed.iter().any(|x| !x.is_finite()) {
            continue;
        }

        // Per-coefficient data shares `diag(V diag(alpha*e / (alpha*e + lambda)) Vᵀ)`.
        for a in 0..kk {
            let mut share = 0.0;
            for m in 0..kk {
                let vam = v[(a, m)];
                share += vam * vam * shrink_d[m];
            }
            support[kept_z[a]] = share.clamp(0.0, 1.0);
        }
        let edf = gamma + if fit_intercept { 1.0 } else { 0.0 };

        results[jj] = Some(EvidenceSolvedTarget {
            beta,
            edf,
            support,
            slot: EvidenceSlot {
                alpha,
                lambda,
                sigma,
                z: kept_z.clone(),
                s: kept_s.clone(),
                mean: kept_means,
                centred: fit_intercept,
            },
        });
    }

    results
}
