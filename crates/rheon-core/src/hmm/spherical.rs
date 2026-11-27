//! [`EwCovSpherical`]: an isotropic covariance accumulator for `hmm` under
//! `covariance_type = "spherical"`.
//!
//! Each state keeps a mean vector of length `k` (compensated pairs, as
//! [`crate::EwCovDiag`] keeps) and one scalar variance `var` — the mean of the
//! per-feature centred second moments, `8` bytes of covariance state per state
//! instead of `8 k` (`diag`) or `4 k (k + 1)` (`full`).
//!
//! ```text
//! W'    = lam·W + w
//! a     = lam·W / W'        b = w / W'
//! dᵢ    = (xᵢ − mᵢ) − loᵢ
//! var'  = a·var + a·b·(Σᵢ dᵢ² / k)       (learning both mean and variance)
//! var'  = a·var + b·(Σᵢ dᵢ² / k)         (frozen mean, learning variance)
//! mᵢ,loᵢ += b·dᵢ                         (when learning mean, b > 0)
//! ```

use serde::{Deserialize, Serialize};

/// One hidden state's spherical (isotropic) sufficient statistics.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EwCovSpherical {
    k: usize,
    w_sum: f64,
    m: Vec<f64>,
    m_lo: Vec<f64>,
    var: f64,
    precision_prior: f64,
    precision_scale: f64,
    rows_learned: u64,
}

impl EwCovSpherical {
    /// An empty spherical accumulator over `k` features with a ridge of `prior`.
    pub fn with_precision_prior(k: usize, prior: f64) -> Result<Self, String> {
        if prior <= 0.0 || !prior.is_finite() {
            return Err(format!(
                "EwCovSpherical::with_precision_prior: prior must be finite and above 0, but it \
                 is {prior}. Pass a finite positive prior."
            ));
        }
        Ok(Self {
            k,
            w_sum: 0.0,
            m: vec![0.0; k],
            m_lo: vec![0.0; k],
            var: 0.0,
            precision_prior: prior,
            precision_scale: 1.0,
            rows_learned: 0,
        })
    }

    /// An accumulator standing at the given weight, means, and isotropic
    /// variance.
    pub fn from_moments(
        k: usize,
        w_sum: f64,
        means: &[f64],
        var: f64,
        prior: f64,
    ) -> Result<Self, String> {
        if means.len() != k {
            return Err(format!(
                "{k} columns need {k} means, but there are {}. Give one mean for each column.",
                means.len()
            ));
        }
        if w_sum < 0.0 || !w_sum.is_finite() {
            return Err(format!(
                "the weight of the rows must be finite and at least 0, but it is {w_sum}. Give a \
                 finite weight of at least 0."
            ));
        }
        if var < 0.0 || !var.is_finite() {
            return Err(format!(
                "the spherical variance must be finite and at least 0, but it is {var}. Give a \
                 finite non-negative variance."
            ));
        }
        let mut ew = Self::with_precision_prior(k, prior)?;
        ew.w_sum = w_sum;
        ew.m.copy_from_slice(means);
        ew.var = var;
        Ok(ew)
    }

    #[inline]
    pub fn k(&self) -> usize {
        self.k
    }

    pub fn has_shape(&self, k: usize) -> bool {
        self.k == k && self.m.len() == k && self.m_lo.len() == k
    }

    pub(crate) fn has_valid_weights(&self) -> bool {
        let non_neg = |v: f64| v.is_finite() && v >= 0.0;
        non_neg(self.w_sum)
            && non_neg(self.var)
            && non_neg(self.precision_prior)
            && non_neg(self.precision_scale)
    }

    #[inline]
    pub fn n_eff(&self) -> f64 {
        self.w_sum
    }

    #[inline]
    pub fn precision_scale(&self) -> f64 {
        self.precision_scale
    }

    #[inline]
    pub fn means(&self) -> &[f64] {
        &self.m
    }

    #[inline]
    pub fn deviation(&self, i: usize, x: f64) -> f64 {
        crate::comp::dev(x, self.m[i], self.m_lo[i])
    }

    /// Centred isotropic variance, clamped against negative rounding.
    #[inline]
    pub fn var(&self) -> f64 {
        crate::solve::clamp_rounding(self.var)
    }

    /// Sum of squared deviations of `x` from this state's mean.
    #[inline]
    pub fn sq_dist(&self, x: &[f64]) -> f64 {
        let mut sq = 0.0;
        for ((&mi, &lo), &xi) in self.m.iter().zip(self.m_lo.iter()).zip(x) {
            let d = crate::comp::dev(xi, mi, lo);
            sq += d * d;
        }
        sq
    }

    /// One observation with decay `lam` and weight `w`, updating both mean and
    /// isotropic variance.
    pub fn update(&mut self, x: &[f64], lam: f64, w: f64) {
        debug_assert_eq!(x.len(), self.k);
        if w > 0.0 {
            self.rows_learned += 1;
        }
        if w == 0.0 && lam * self.w_sum == 0.0 {
            self.decay(lam);
            return;
        }
        if w < 0.0 {
            return;
        }
        let w_new = lam * self.w_sum + w;
        if w_new <= 0.0 {
            return;
        }
        let a = lam * self.w_sum / w_new;
        let b = w / w_new;
        let mut sq = 0.0;
        for ((mi, lo), &xi) in self.m.iter_mut().zip(self.m_lo.iter_mut()).zip(x) {
            let d = crate::comp::dev(xi, *mi, *lo);
            sq += d * d;
            if b > 0.0 {
                crate::comp::step_toward(mi, lo, xi, d, a, b);
            }
        }
        self.var = a * self.var + (a * b) * (sq / self.k as f64);
        self.precision_scale = if a <= 0.0 {
            1.0
        } else {
            self.precision_scale * a
        };
        self.w_sum = w_new;
    }

    /// Update only the mean (and weight); leave the variance and prior scale
    /// bit-equal.
    pub fn update_mean_only(&mut self, x: &[f64], lam: f64, w: f64) {
        debug_assert_eq!(x.len(), self.k);
        if w > 0.0 {
            self.rows_learned += 1;
        }
        if w == 0.0 && lam * self.w_sum == 0.0 {
            self.w_sum *= lam;
            return;
        }
        if w < 0.0 {
            return;
        }
        let w_new = lam * self.w_sum + w;
        if w_new <= 0.0 {
            return;
        }
        let a = lam * self.w_sum / w_new;
        let b = w / w_new;
        if b > 0.0 {
            for ((mi, lo), &xi) in self.m.iter_mut().zip(self.m_lo.iter_mut()).zip(x) {
                let d = crate::comp::dev(xi, *mi, *lo);
                crate::comp::step_toward(mi, lo, xi, d, a, b);
            }
        }
        self.w_sum = w_new;
    }

    /// Update only the isotropic variance about the frozen mean; leave the mean
    /// bit-equal.
    pub fn update_cov_only(&mut self, x: &[f64], lam: f64, w: f64) {
        debug_assert_eq!(x.len(), self.k);
        if w > 0.0 {
            self.rows_learned += 1;
        }
        if w == 0.0 && lam * self.w_sum == 0.0 {
            self.decay(lam);
            return;
        }
        if w < 0.0 {
            return;
        }
        let w_new = lam * self.w_sum + w;
        if w_new <= 0.0 {
            return;
        }
        let a = lam * self.w_sum / w_new;
        let b = w / w_new;
        let sq = self.sq_dist(x);
        self.var = a * self.var + b * (sq / self.k as f64);
        self.precision_scale = if a <= 0.0 {
            1.0
        } else {
            self.precision_scale * a
        };
        self.w_sum = w_new;
    }

    /// Age the weight without adding data.
    pub fn decay(&mut self, lam: f64) {
        self.w_sum *= lam;
    }
}
