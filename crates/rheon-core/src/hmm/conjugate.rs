//! Conjugate covariance-prior density evaluation and spherical stabilized
//! density evaluation for [`super::Hmm`].

use super::options::HmmCovariance;
use super::spherical::EwCovSpherical;
use super::stabilized::Stabilized;
use super::{Hmm, RowWork};

impl Hmm {
    /// Whether this model uses conjugate covariance priors (`covariance_prior` /
    /// `covariance_weight`) instead of `Stabilized` or the decaying ridge.
    #[inline]
    pub(super) fn uses_conjugate_covars(&self) -> bool {
        self.cfg.has_covariance_prior() && self.cfg.effective_params().c
    }

    /// The per-state spherical accumulators (empty unless `Spherical`).
    #[inline]
    pub(super) fn spherical_states(&self) -> &[EwCovSpherical] {
        self.extras
            .as_deref()
            .map_or(&[], |e| e.spherical_states.as_slice())
    }

    /// Mutable access to spherical accumulators.
    #[inline]
    pub(super) fn spherical_states_mut(&mut self) -> &mut [EwCovSpherical] {
        self.extras
            .as_deref_mut()
            .map_or(&mut [], |e| e.spherical_states.as_mut_slice())
    }

    /// The per-state conjugate data weights `G_l` (empty unless conjugate mode).
    #[inline]
    pub(super) fn data_w(&self) -> &[f64] {
        self.extras.as_deref().map_or(&[], |e| e.data_w.as_slice())
    }

    /// Mutable access to `data_w`.
    #[inline]
    pub(super) fn data_w_mut(&mut self) -> &mut [f64] {
        self.extras
            .as_deref_mut()
            .map_or(&mut [], |e| e.data_w.as_mut_slice())
    }

    /// Log densities for `spherical` under stabilized forgetting:
    /// `v_s = (W_s · s_s + P₀ · ψ₀ · g_s) / (W_s + P₀)`.
    #[inline(never)]
    pub(super) fn spherical_stabilized(
        st: &Stabilized,
        states: &[EwCovSpherical],
        x: &[f64],
        base: f64,
        out: &mut [f64],
    ) -> bool {
        let df = x.len() as f64;
        let p0 = st.prior_rows;
        let psi0 = st.psi()[0];
        let shape = st.shape();
        for (s, (o, sp)) in out.iter_mut().zip(states).enumerate() {
            let g = shape.map_or(1.0, |sh| sh.diag(s)[0]);
            let t = sp.n_eff() + p0;
            let v = (sp.n_eff() / t) * sp.var() + (p0 / t) * (psi0 * g);
            let sq = sp.sq_dist(x);
            *o = base - 0.5 * df * v.ln() - 0.5 * sq / v;
        }
        out.iter().all(|v| v.is_finite())
    }

    /// Log densities under conjugate covariance priors (`covariance_prior` /
    /// `covariance_weight`), matching `hmmlearn._do_mstep` at `half_life = inf`
    /// and aging `W_l` and `G_l` by `lam_age` before reading so zero-weight
    /// rows obey hard rule 9.
    #[inline(never)]
    pub(super) fn conjugate_log_densities_into(
        &self,
        x: &[f64],
        lam_age: f64,
        work: &mut RowWork,
    ) -> bool {
        let (k, d) = (self.cfg.k, self.cfg.n_features);
        let df = d as f64;
        let base = -0.5 * df * std::f64::consts::TAU.ln();
        let cvd0 = self.cfg.covariance_weight_offset();
        let data_w = self.data_w();
        let pre_first = self.n_eff == 0.0;
        let given_covs = self.cfg.covs.as_deref();
        let RowWork {
            logf: out,
            delta,
            quad,
            shared: m,
            deltas,
            forms,
            shared_work,
            ..
        } = work;
        out.clear();
        out.resize(k, f64::NEG_INFINITY);
        match self.cfg.covariance {
            HmmCovariance::Spherical => {
                let states = self.spherical_states();
                for (s, (o, sp)) in out.iter_mut().zip(states).enumerate() {
                    let v = if let Some(c) = given_covs.filter(|_| pre_first) {
                        c[s]
                    } else {
                        let w_l = lam_age * sp.n_eff();
                        let g_l = lam_age * data_w[s];
                        let prior = self.cfg.covariance_prior_spherical(s);
                        let num = prior + w_l * sp.var();
                        let denom = (cvd0 + g_l).max(1e-5);
                        if num > 0.0 {
                            num / denom
                        } else {
                            given_covs.map_or(self.cfg.precision_prior, |c| c[s])
                        }
                    };
                    let sq = sp.sq_dist(x);
                    *o = base - 0.5 * df * v.ln() - 0.5 * sq / v;
                }
            }
            HmmCovariance::Diagonal => {
                for (s, (o, cov)) in out.iter_mut().zip(&self.diag_states).enumerate() {
                    let w_l = lam_age * cov.n_eff();
                    let g_l = lam_age * data_w[s];
                    let denom = (cvd0 + g_l).max(1e-5);
                    let (mut log_det, mut q) = (0.0, 0.0);
                    for (i, &xi) in x.iter().enumerate() {
                        let v = if let Some(c) = given_covs.filter(|_| pre_first) {
                            c[s * d * d + i * d + i]
                        } else {
                            let prior = self.cfg.covariance_prior_diag(s, i);
                            let num = prior + w_l * cov.var(i);
                            if num > 0.0 {
                                num / denom
                            } else {
                                given_covs
                                    .map_or(self.cfg.precision_prior, |c| c[s * d * d + i * d + i])
                            }
                        };
                        let dv = cov.deviation(i, xi);
                        log_det += v.ln();
                        q += dv * dv / v;
                    }
                    *o = base - 0.5 * log_det - 0.5 * q;
                }
            }
            HmmCovariance::Full => {
                delta.clear();
                delta.resize(d, 0.0);
                for (s, (o, cov)) in out.iter_mut().zip(&self.states).enumerate() {
                    m.clear();
                    m.resize(d * d, 0.0);
                    if let Some(gc) = given_covs.filter(|_| pre_first) {
                        let c = &gc[s * d * d..(s + 1) * d * d];
                        for i in 0..d {
                            for j in 0..=i {
                                m[i * d + j] = c[i * d + j];
                            }
                        }
                    } else {
                        let w_l = lam_age * cov.n_eff();
                        let g_l = lam_age * data_w[s];
                        let denom = (cvd0 + g_l).max(1e-5);
                        let packed = cov.comoments().packed();
                        let mut any_pos = false;
                        for i in 0..d {
                            let at = crate::ewcov::row_at(i);
                            for j in 0..=i {
                                let v = (self.cfg.covariance_prior_matrix(s, i, j)
                                    + w_l * packed[at + j])
                                    / denom;
                                m[i * d + j] = v;
                                if i == j && v > 0.0 {
                                    any_pos = true;
                                }
                            }
                        }
                        if !any_pos {
                            for i in 0..d {
                                m[i * d + i] = given_covs
                                    .map_or(self.cfg.precision_prior, |c| c[s * d * d + i * d + i]);
                            }
                        }
                    }
                    for (i, (dv, &xi)) in delta.iter_mut().zip(x).enumerate() {
                        *dv = cov.deviation(i, xi);
                    }
                    let Some(f) = shared_work.factor(m, d) else {
                        return false;
                    };
                    *o = base - 0.5 * f.log_det() - 0.5 * f.quad_form(delta, quad);
                    shared_work.give(f);
                }
            }
            HmmCovariance::Shared => {
                m.clear();
                m.resize(d * d, 0.0);
                if let Some(c) = given_covs.filter(|_| pre_first) {
                    let inv_k = 1.0 / k as f64;
                    for s in 0..k {
                        let block = &c[s * d * d..(s + 1) * d * d];
                        for i in 0..d {
                            for j in 0..=i {
                                m[i * d + j] += inv_k * block[i * d + j];
                            }
                        }
                    }
                } else {
                    let total_g: f64 = data_w.iter().map(|&g| lam_age * g).sum();
                    let denom = (cvd0 + total_g).max(1e-5);
                    for i in 0..d {
                        for j in 0..=i {
                            m[i * d + j] = self.cfg.covariance_prior_matrix(0, i, j);
                        }
                    }
                    for cov in &self.states {
                        let w_l = lam_age * cov.n_eff();
                        let packed = cov.comoments().packed();
                        for i in 0..d {
                            let at = crate::ewcov::row_at(i);
                            for j in 0..=i {
                                m[i * d + j] += w_l * packed[at + j];
                            }
                        }
                    }
                    for i in 0..d {
                        for j in 0..=i {
                            m[i * d + j] /= denom;
                        }
                    }
                }
                deltas.clear();
                deltas.resize(d * k, 0.0);
                for (s, state) in self.states.iter().enumerate() {
                    for (i, &xi) in x.iter().enumerate() {
                        deltas[s * d + i] = state.deviation(i, xi);
                    }
                }
                let Some(f) = shared_work.factor(m, d) else {
                    return false;
                };
                f.quad_forms_into(deltas, d, k, quad, forms);
                let log_det = f.log_det();
                shared_work.give(f);
                for (s, o) in out.iter_mut().enumerate() {
                    *o = base - 0.5 * log_det - 0.5 * forms[s];
                }
            }
        }
        out.iter().all(|v| v.is_finite())
    }
}
