//! Extended and unscented Kalman filters for generalized linear observations
//! on [`Kalman`] (docs/PLAN.md task 497; design record:
//! `docs/records/NONLINEAR-FILTERS-DESIGN.md`).
//!
//! State-space model (linear transition, scalar single-index observation):
//!
//! ```text
//! transition   β_n = Φ(D_n) β_{n-1} + u_n,   u_n ~ N(0, Q(D_n))
//!              Φ(D) = diag(2^(-D/r_i)),  Q(D) = diag(q_i g_i(D)²)
//! observation  η_n = z_n' β_n,   μ_n = g⁻¹(η_n),
//!              y_n | η_n ~ family(μ_n, φ V(μ_n) / w_n)
//! ```
//!
//! Every transcendental (`exp`, `ln`, `pow`, `exp2`, `expm1`) in this module
//! calls the `libm` crate compiled from source in this build
//! ([`crate::LibmMath`], [`crate::Decay::portable_factor`]), never the
//! platform's math library: clipped EKF streams amplify a one-ulp difference
//! to `1e-10`, and using `libm` keeps every output and saved state bit-exact
//! across macOS, Linux and Windows.

use serde::{Deserialize, Serialize};

use super::{
    Kalman, KalmanCfg, a_number_or_none, first_noise, forms, matvec, shared_noise, sum_ascending,
    transit,
};
use crate::family::{ETA_CLIP, Family, GlmFamily, GlmLink, LibmMath};
use crate::{Decay, Step};

/// `sqrt(3)` for the 3-point Gauss-Hermite scalar-index UKF rule.
const SQRT_3: f64 = 1.732_050_807_568_877_2;

/// Observation distribution of a [`Kalman`] filter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KalmanFamily {
    /// Constant-variance Gaussian (`V(μ) = 1`).
    #[default]
    Gaussian,
    /// Counts (`V(μ) = μ`, `y ≥ 0`).
    Poisson,
    /// Proportions or 0/1 outcomes (`V(μ) = μ(1 − μ)`, `0 ≤ y ≤ 1`).
    Binomial,
    /// Positive amounts (`V(μ) = μ²`, `y > 0`).
    Gamma,
    /// Power-variance family (`V(μ) = μᵖ`, `p > 1`, `p ≠ 2`).
    Tweedie,
}

/// Link function `η = g(μ)` relating the observation mean to `z'β`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KalmanLink {
    /// `μ = η` (Gaussian only).
    #[default]
    Identity,
    /// `μ = e^η`, `η` clipped to `±30`.
    Log,
    /// `μ = 1 / (1 + e^(−η))`, `η` clipped to `±30`.
    Logit,
    /// `μ = 1 / η`, defined for `η > 0` (Gamma only).
    Inverse,
    /// `μ = η^(1/(1 − p))`, defined for `η > 0` (Tweedie only).
    Power,
}

/// Filtering recursion for a [`Kalman`] model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KalmanFilter {
    /// Exact linear Kalman filter (`gaussian` with `identity` link only).
    #[default]
    Linear,
    /// Extended Kalman filter (closed-form Jacobian at the prior mean).
    Ekf,
    /// Unscented Kalman filter (sigma-point update).
    Ukf,
}

/// Sigma-point quadrature rule for [`KalmanFilter::Ukf`].
///
/// Default is [`UkfRule::ScalarIndex`]: measured against the exact filtering
/// posterior (the `k = 2` grid filter across 9 settings, the `k = 5` particle
/// filter across 3 families, and statsmodels' static MAP at `k = 2, 5, 20`),
/// the 3-point Gauss-Hermite rule along `η = z'β` is closer on 8 of 9 grid
/// settings (up to 2.4x closer on binomial), 7 of 9 static settings and the
/// Poisson particle filter, is invariant to feature ordering (`1.7e-15` vs
/// `7.6%` for `merwe` at `k = 20`), and runs in `O(k²)` with 3 link calls
/// instead of `O(k³)` with `2k + 1`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UkfRule {
    /// Three-point Gauss-Hermite rule along `η = z'β` (`O(k²)`).
    #[default]
    ScalarIndex,
    /// Full Merwe scaled sigma points (`2k + 1` points via Cholesky, `O(k³)`).
    Merwe,
}

/// How [`KalmanFilter::Ukf`] forms the observation variance `R`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UkfNoise {
    /// Expected observation variance over the sigma points: `φ Σ Wm_i V(ζ_i) / w`.
    #[default]
    Expected,
    /// Plug-in variance at the prior mean: `φ V(g⁻¹(η̃)) / w`, as in the EKF.
    Plugin,
}

impl KalmanFamily {
    pub fn name(self) -> &'static str {
        match self {
            KalmanFamily::Gaussian => "gaussian",
            KalmanFamily::Poisson => "poisson",
            KalmanFamily::Binomial => "binomial",
            KalmanFamily::Gamma => "gamma",
            KalmanFamily::Tweedie => "tweedie",
        }
    }

    pub fn default_link(self) -> KalmanLink {
        match self {
            KalmanFamily::Gaussian => KalmanLink::Identity,
            KalmanFamily::Binomial => KalmanLink::Logit,
            KalmanFamily::Poisson | KalmanFamily::Gamma | KalmanFamily::Tweedie => KalmanLink::Log,
        }
    }

    pub fn links(self) -> &'static [KalmanLink] {
        match self {
            KalmanFamily::Gaussian => &[KalmanLink::Identity, KalmanLink::Log, KalmanLink::Logit],
            KalmanFamily::Poisson => &[KalmanLink::Log],
            KalmanFamily::Binomial => &[KalmanLink::Logit],
            KalmanFamily::Gamma => &[KalmanLink::Log, KalmanLink::Inverse],
            KalmanFamily::Tweedie => &[KalmanLink::Log, KalmanLink::Power],
        }
    }
}

impl KalmanLink {
    pub fn name(self) -> &'static str {
        match self {
            KalmanLink::Identity => "identity",
            KalmanLink::Log => "log",
            KalmanLink::Logit => "logit",
            KalmanLink::Inverse => "inverse",
            KalmanLink::Power => "power",
        }
    }

    /// Whether the link gives a mean at `η = 0`.
    pub fn defined_at_zero(self) -> bool {
        matches!(
            self,
            KalmanLink::Identity | KalmanLink::Log | KalmanLink::Logit
        )
    }
}

impl KalmanFilter {
    pub fn name(self) -> &'static str {
        match self {
            KalmanFilter::Linear => "linear",
            KalmanFilter::Ekf => "ekf",
            KalmanFilter::Ukf => "ukf",
        }
    }
}

impl UkfRule {
    pub fn name(self) -> &'static str {
        match self {
            UkfRule::ScalarIndex => "scalar_index",
            UkfRule::Merwe => "merwe",
        }
    }
}

impl UkfNoise {
    pub fn name(self) -> &'static str {
        match self {
            UkfNoise::Expected => "expected",
            UkfNoise::Plugin => "plugin",
        }
    }
}

/// Nonlinear observation and filter settings for [`KalmanCfg`].
///
/// The default is the linear Gaussian identity filter (`filter = Linear`,
/// `family = Gaussian`, `link = Identity`). [`KalmanCfg`]'s serializer skips
/// this field at its default so every linear `kalman` state file keeps its
/// exact bytes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KalmanObs {
    pub family: KalmanFamily,
    pub link: KalmanLink,
    /// Tweedie variance power `p` (`> 1`, finite, `≠ 2`); `None` for other families.
    pub var_power: Option<f64>,
    /// Fixed dispersion `φ > 0` for non-Gaussian families (default `1.0`).
    pub dispersion: f64,
    pub filter: KalmanFilter,
    pub ukf_rule: UkfRule,
    /// Merwe scaled sigma-point spread `α > 0` (default `1.0`).
    pub alpha: f64,
    /// Merwe distribution parameter `β` (default `2.0`).
    pub beta: f64,
    /// Merwe secondary scaling parameter `κ` (default `0.0`).
    pub kappa: f64,
    pub ukf_noise: UkfNoise,
}

impl Default for KalmanObs {
    fn default() -> Self {
        Self {
            family: KalmanFamily::Gaussian,
            link: KalmanLink::Identity,
            var_power: None,
            dispersion: 1.0,
            filter: KalmanFilter::Linear,
            ukf_rule: UkfRule::ScalarIndex,
            alpha: 1.0,
            beta: 2.0,
            kappa: 0.0,
            ukf_noise: UkfNoise::Expected,
        }
    }
}

impl KalmanObs {
    /// True when every field is at the linear Gaussian default (skipped on write).
    pub fn is_default(&self) -> bool {
        *self == Self::default()
    }

    /// True when the observation is Gaussian with the identity link.
    #[inline]
    pub fn is_gaussian_identity(&self) -> bool {
        self.family == KalmanFamily::Gaussian && self.link == KalmanLink::Identity
    }

    /// True when the filter is the exact linear Kalman filter.
    #[inline]
    pub fn is_linear(&self) -> bool {
        self.filter == KalmanFilter::Linear
    }

    /// The underlying [`Family`] for the four non-Gaussian GLM families, or
    /// `None` for `Gaussian`.
    pub fn glm_family(&self) -> Option<Family> {
        let family = match self.family {
            KalmanFamily::Gaussian => return None,
            KalmanFamily::Poisson => GlmFamily::Poisson,
            KalmanFamily::Binomial => GlmFamily::Binomial,
            KalmanFamily::Gamma => GlmFamily::Gamma,
            KalmanFamily::Tweedie => GlmFamily::Tweedie,
        };
        let link = match self.link {
            KalmanLink::Log => GlmLink::Log,
            KalmanLink::Logit => GlmLink::Logit,
            KalmanLink::Inverse => GlmLink::Inverse,
            KalmanLink::Power => GlmLink::Power,
            KalmanLink::Identity => return None,
        };
        Some(Family {
            family,
            link,
            power: self.var_power.unwrap_or(0.0),
        })
    }

    /// Whether `y` lies in this family's support (`-0.0` is 0).
    #[inline]
    pub fn takes(&self, y: f64) -> bool {
        if !y.is_finite() {
            return false;
        }
        match self.glm_family() {
            Some(f) => f.takes(y),
            None => true,
        }
    }

    /// `μ = g⁻¹(η)` and clipped `η` through `libm`, or `None` when undefined
    /// (`η` non-finite, or `η ≤ 0` under inverse/power links).
    #[inline]
    pub fn mean_libm(&self, eta: f64) -> Option<(f64, f64)> {
        if !eta.is_finite() {
            return None;
        }
        match self.link {
            KalmanLink::Identity => Some((eta, eta)),
            KalmanLink::Log => {
                let e = eta.clamp(-ETA_CLIP, ETA_CLIP);
                let mu = libm::exp(e);
                (mu.is_finite() && mu > 0.0).then_some((e, mu))
            }
            KalmanLink::Logit => {
                let e = eta.clamp(-ETA_CLIP, ETA_CLIP);
                let mu = 1.0 / (1.0 + libm::exp(-e));
                (mu.is_finite() && mu > 0.0).then_some((e, mu))
            }
            KalmanLink::Inverse | KalmanLink::Power => {
                self.glm_family()?.mean_with::<LibmMath>(eta)
            }
        }
    }

    /// `dμ/dη = 1 / g'(μ)` at `μ = g⁻¹(η̃)` through `libm`.
    #[inline]
    pub fn inv_link_deriv_libm(&self, mu: f64) -> f64 {
        match self.link {
            KalmanLink::Identity => 1.0,
            KalmanLink::Log => mu,
            KalmanLink::Logit => mu * (1.0 - mu),
            KalmanLink::Inverse => -(mu * mu),
            KalmanLink::Power => self
                .glm_family()
                .map_or(f64::NAN, |f| f.inv_link_deriv_with::<LibmMath>(mu)),
        }
    }

    /// `V(μ)` through `libm`.
    #[inline]
    pub fn variance_libm(&self, mu: f64) -> f64 {
        match self.glm_family() {
            Some(f) => f.variance_with::<LibmMath>(mu),
            None => 1.0,
        }
    }

    /// Observation scale `φ`: `obs_var` (or `sigma2`) for Gaussian, `dispersion`
    /// for the non-Gaussian families.
    #[inline]
    fn obs_scale(&self, cfg: &KalmanCfg, sigma2: f64) -> f64 {
        match self.family {
            KalmanFamily::Gaussian => cfg.obs_var.unwrap_or(sigma2),
            _ => self.dispersion,
        }
    }

    /// Validate `self` and its interactions with the rest of `cfg`.
    pub(super) fn validate(&self, cfg: &KalmanCfg, no: &mut crate::Refusals) {
        if !self.family.links().contains(&self.link) {
            let takes: Vec<&str> = self.family.links().iter().map(|l| l.name()).collect();
            let lead = format!(
                "kalman: the {} family takes the link {}, but it was given {}.",
                self.family.name(),
                takes.join(" or "),
                self.link.name(),
            );
            let rest = format!(
                "Set link to one of those, or leave it out for the family's default, {}.",
                self.family.default_link().name(),
            );
            no.refuse(match (self.family, self.link) {
                (KalmanFamily::Binomial, KalmanLink::Log) => format!(
                    "{lead} The log link's mean e^η must stay below 1, and nothing in the fit \
                     keeps it there. {rest}"
                ),
                _ => format!("{lead} {rest}"),
            });
        }

        if self.family == KalmanFamily::Tweedie {
            match self.var_power {
                Some(p) if p > 1.0 && p.is_finite() && p != 2.0 => {}
                Some(p) => no.refuse(format!(
                    "kalman: the Tweedie family's var_power must be above 1, finite and not 2, \
                     but it is {p}. Set var_power between 1 and 2 for a count of positive \
                     amounts with exact zeros, such as 1.5, or above 2; for a power of 1 or 2 \
                     use the poisson or gamma family."
                )),
                None => no.refuse(
                    "kalman: the Tweedie family needs var_power, but it is not set. Set \
                     var_power to a finite number above 1 and not 2, such as 1.5.",
                ),
            }
        } else if self.var_power.is_some() {
            no.refuse(format!(
                "kalman: var_power is the Tweedie family's power, but family is {:?}, which \
                 reads none. Leave var_power out, or set family=\"tweedie\".",
                self.family.name()
            ));
        }

        if !(self.dispersion.is_finite() && self.dispersion > 0.0) {
            no.refuse(format!(
                "kalman: dispersion must be finite and above 0, but it is {}. Set dispersion to \
                 a finite positive scale parameter, such as 1.",
                self.dispersion
            ));
        } else if self.family == KalmanFamily::Gaussian && self.dispersion != 1.0 {
            no.refuse(
                "kalman: dispersion is for the non-Gaussian families, but family is \"gaussian\". \
                 Set obs_var for a Gaussian observation variance, or leave dispersion out.",
            );
        }

        if self.filter == KalmanFilter::Linear && !self.is_gaussian_identity() {
            no.refuse(format!(
                "kalman: filter=\"linear\" applies only to family=\"gaussian\" with \
                 link=\"identity\", not family={:?} with link={:?}. Set filter=\"ekf\" or \
                 filter=\"ukf\".",
                self.family.name(),
                self.link.name()
            ));
        }

        if !self.is_gaussian_identity() {
            if self.family == KalmanFamily::Gaussian {
                if cfg.obs_var.is_none() {
                    no.refuse(format!(
                        "kalman: family=\"gaussian\" with link={:?} needs a fixed obs_var, \
                         because the nonlinear filter reads P on an absolute scale. Set obs_var \
                         to a finite positive variance.",
                        self.link.name()
                    ));
                }
            } else if cfg.obs_var.is_some() {
                no.refuse(format!(
                    "kalman: obs_var cannot be set with family={:?}: the observation variance is \
                     the family's variance scaled by dispersion. Leave obs_var out and set \
                     dispersion instead.",
                    self.family.name()
                ));
            }

            if cfg.q.is_none() || !cfg.half_life.is_empty() {
                no.refuse(
                    "kalman: a nonlinear observation family needs an explicit q and cannot use \
                     coef_half_life: coef_half_life's gain matching assumes a constant Gaussian \
                     information per row. Give q (one finite variance per coefficient) and leave \
                     out coef_half_life.",
                );
            }

            if !self.link.defined_at_zero() && !cfg.fit_intercept {
                no.refuse(format!(
                    "kalman: the {} link gives a mean only where η > 0, and without an intercept \
                     every coefficient starts at 0. Set fit_intercept=True, or use the log link.",
                    self.link.name()
                ));
            }

            if cfg.share_p {
                no.refuse(
                    "kalman: share_p cannot be used with a nonlinear observation: each target's \
                     Jacobian and observation variance depend on its own coefficients, so targets \
                     cannot share one covariance. Leave share_p=False.",
                );
            }

            if !cfg.horizons.is_empty() {
                no.refuse(
                    "kalman: forecast_spans is only supported for family=\"gaussian\" with \
                     link=\"identity\", whose mean and integral are linear in the coefficients. \
                     Leave out forecast_spans.",
                );
            }
        }

        if !self.is_linear() || !self.is_gaussian_identity() {
            let reason = if !self.is_linear() {
                format!("not filter={:?}", self.filter.name())
            } else {
                format!(
                    "not family={:?} with link={:?}",
                    self.family.name(),
                    self.link.name()
                )
            };
            if cfg.emit_loglik {
                no.refuse(format!(
                    "kalman: emit_loglik applies only to filter=\"linear\" (with \
                     family=\"gaussian\" and link=\"identity\"), {reason}: the Gaussian \
                     innovation log-likelihood assumes a linear Gaussian observation. Leave \
                     emit_loglik=False, or set filter=\"linear\"."
                ));
            }
            if cfg.emit_mahal {
                no.refuse(format!(
                    "kalman: emit_mahal applies only to filter=\"linear\" (with \
                     family=\"gaussian\" and link=\"identity\"), {reason}: the Gaussian \
                     innovation Mahalanobis distance assumes a linear Gaussian observation. Leave \
                     emit_mahal=False, or set filter=\"linear\"."
                ));
            }
        }

        if !(self.alpha.is_finite() && self.alpha > 0.0) {
            no.refuse(format!(
                "kalman: alpha must be finite and above 0, but it is {}. Set alpha to a finite \
                 positive sigma-point spread, such as the default 1.",
                self.alpha
            ));
        }
        if !self.beta.is_finite() {
            no.refuse(format!(
                "kalman: beta must be finite, but it is {}. Set beta to a finite number, such as \
                 the default 2.",
                self.beta
            ));
        }
        if !self.kappa.is_finite() {
            no.refuse(format!(
                "kalman: kappa must be finite, but it is {}. Set kappa to a finite number, such \
                 as the default 0.",
                self.kappa
            ));
        } else if self.filter == KalmanFilter::Ukf && self.ukf_rule == UkfRule::Merwe {
            let k = cfg.k_total() as f64;
            let c = self.alpha * self.alpha * (k + self.kappa);
            if !(k + self.kappa > 0.0 && c > 0.0 && c.is_finite()) {
                no.refuse(format!(
                    "kalman: the Merwe sigma-point scale must be finite and above 0, but with \
                     k = {}, alpha = {} and kappa = {} it is {c}. Raise kappa above -{} or adjust \
                     alpha.",
                    cfg.k_total(),
                    self.alpha,
                    self.kappa,
                    cfg.k_total()
                ));
            }
        }
    }
}

/// Helper to serialize `Vec<f64>` through [`crate::humanfloat::vec_f64_or_tag`].
struct VecF64OrTag<'a>(&'a [f64]);

impl Serialize for VecF64OrTag<'_> {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        crate::humanfloat::vec_f64_or_tag::serialize(self.0, s)
    }
}

/// Custom [`Serialize`] for [`KalmanCfg`]: skips trailing optional fields
/// (`noise_units`, `forecast_spans`, `obs`, `coef_prior`, `emit_loglik`,
/// `emit_mahal`) when at their defaults so every existing `kalman` state file
/// keeps its exact bytes, while always writing every earlier optional field
/// whenever a later one is written so compact positional msgpack
/// (`rmp_serde::to_vec`) round-trips cleanly.
impl Serialize for KalmanCfg {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let has_mahal = self.emit_mahal;
        let has_loglik = self.emit_loglik || has_mahal;
        let has_coef_prior = self.coef_prior.is_some() || has_loglik;
        let has_obs = !self.obs.is_default() || has_coef_prior;
        let has_horizons = !self.horizons.is_empty() || has_obs;
        let has_noise_units = self.noise_units || has_horizons;
        let n = 13
            + usize::from(has_noise_units)
            + usize::from(has_horizons)
            + usize::from(has_obs)
            + usize::from(has_coef_prior)
            + usize::from(has_loglik)
            + usize::from(has_mahal);
        let mut st = s.serialize_struct("KalmanCfg", n)?;
        st.serialize_field("n_features", &self.n_features)?;
        st.serialize_field("n_targets", &self.n_targets)?;
        st.serialize_field("fit_intercept", &self.fit_intercept)?;
        st.serialize_field("decay", &self.decay)?;
        st.serialize_field("half_life", &VecF64OrTag(&self.half_life))?;
        st.serialize_field("q", &self.q)?;
        st.serialize_field("obs_var", &self.obs_var)?;
        st.serialize_field("p0", &self.p0)?;
        st.serialize_field("share_p", &self.share_p)?;
        st.serialize_field("min_weight", &self.min_weight)?;
        st.serialize_field("revert_half_life", &VecF64OrTag(&self.revert_half_life))?;
        st.serialize_field("standardize", &self.standardize)?;
        st.serialize_field("min_weight_holds_learning", &self.min_weight_holds_learning)?;
        if has_noise_units {
            st.serialize_field("noise_units", &self.noise_units)?;
        } else {
            st.skip_field("noise_units")?;
        }
        if has_horizons {
            st.serialize_field("forecast_spans", &self.horizons)?;
        } else {
            st.skip_field("forecast_spans")?;
        }
        if has_obs {
            st.serialize_field("obs", &self.obs)?;
        } else {
            st.skip_field("obs")?;
        }
        if has_coef_prior {
            st.serialize_field("coef_prior", &self.coef_prior)?;
        } else {
            st.skip_field("coef_prior")?;
        }
        if has_loglik {
            st.serialize_field("emit_loglik", &self.emit_loglik)?;
        } else {
            st.skip_field("emit_loglik")?;
        }
        if has_mahal {
            st.serialize_field("emit_mahal", &self.emit_mahal)?;
        } else {
            st.skip_field("emit_mahal")?;
        }
        st.end()
    }
}

/// Preallocated scratch buffers for UKF sigma points (split from `p_prior`
/// so `eval_ukf` can borrow `&work.p_prior` and `&mut work.ukf` simultaneously).
#[derive(Debug, Clone, Default)]
struct UkfScratch {
    chol: Vec<f64>,
    idx: Vec<usize>,
    mu_p: Vec<f64>,
    mu_m: Vec<f64>,
    cross: Vec<f64>,
}

/// Preallocated scratch buffers for EKF/UKF steps (never serialized).
#[derive(Debug, Clone, Default)]
pub(super) struct NonlinearWork {
    p_prior: Vec<f64>,
    ukf: UkfScratch,
    y_ok: Vec<Option<f64>>,
    informs: Vec<bool>,
}

impl PartialEq for NonlinearWork {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

impl NonlinearWork {
    fn ensure(&mut self, k: usize, m: usize, n_p: usize) {
        if self.p_prior.len() != k * k {
            self.p_prior.resize(k * k, 0.0);
            self.ukf.chol.resize(k * k, 0.0);
            self.ukf.idx.reserve(k);
            self.ukf.mu_p.resize(k, 0.0);
            self.ukf.mu_m.resize(k, 0.0);
            self.ukf.cross.resize(k, 0.0);
        }
        if self.y_ok.len() != m {
            self.y_ok.resize(m, None);
        }
        if self.informs.len() != n_p {
            self.informs.resize(n_p, false);
        }
    }
}

/// Result of evaluating one UKF quadrature on `(eta0, P⁻, z)`.
struct UkfEval {
    /// Predicted mean `ζ̄` (or EKF's `μ̃` on fallback).
    pred: f64,
    /// Centre mean `μ_0 = g⁻¹(η̃_0)`.
    mu0: f64,
    /// `true` if Cholesky failed or a sigma point left `η > 0` under
    /// inverse/power: the row falls back to the EKF step on the same state.
    fallback_ekf: bool,
    /// Scalar-index values (used when `rule == ScalarIndex` and `!fallback_ekf`).
    sc_delta: f64,
    sc_mu_p: f64,
    sc_mu_m: f64,
    /// Active dimension `n` for Merwe (points stored in `work.chol`, `work.idx`,
    /// `work.mu_p`, `work.mu_m`).
    merwe_n: usize,
}

/// Lower-triangular Cholesky `A = L Lᵀ` in place on the lower triangle of `a`
/// (`n x n` row-major). Returns `false` if any pivot is not finite and `> 0`.
fn cholesky(a: &mut [f64], n: usize) -> bool {
    for j in 0..n {
        let mut d = a[j * n + j];
        for l in 0..j {
            d -= a[j * n + l] * a[j * n + l];
        }
        if !(d > 0.0 && d.is_finite()) {
            return false;
        }
        let d = d.sqrt();
        a[j * n + j] = d;
        for i in j + 1..n {
            let mut s = a[i * n + j];
            for l in 0..j {
                s -= a[i * n + l] * a[j * n + l];
            }
            a[i * n + j] = s / d;
        }
    }
    true
}

/// Symmetric rank-1 downdate `P <- P - s K K'`, mirroring the upper triangle
/// to the lower triangle so `P` stays bit-symmetric across every update.
fn take_rank1_sym(p: &mut [f64], gain: &[f64], s: f64) {
    let k = gain.len();
    for i in 0..k {
        let sgi = s * gain[i];
        for j in i..k {
            let v = p[i * k + j] - sgi * gain[j];
            p[i * k + j] = v;
            if j != i {
                p[j * k + i] = v;
            }
        }
    }
}

/// Portable gap noise `g_i(d)²` using `libm::expm1`.
pub(super) fn gap_noise_portable(cfg: &KalmanCfg, i: usize, d: f64) -> f64 {
    let r = cfg.revert_of(i);
    if r.is_infinite() {
        return d * d;
    }
    let theta = std::f64::consts::LN_2 / r;
    let g = -libm::expm1(-(theta * d)) / theta;
    g * g
}

/// Portable transition factor `2^(-d / r_i)` using `Decay::portable_factor`.
fn phi_portable(cfg: &KalmanCfg, i: usize, d_clock: f64) -> f64 {
    Decay::Halflife(cfg.revert_of(i)).portable_factor(d_clock)
}

/// Add process noise `Q g(d)²` in anchor coordinates to a dense `k x k` matrix `p`.
fn add_noise_to(
    cfg: &KalmanCfg,
    q: &[f64],
    d: f64,
    open: bool,
    map_a: &[f64],
    map_c: &[f64],
    p: &mut [f64],
) {
    if d == 0.0 {
        return;
    }
    let k = cfg.k_total();
    let off = usize::from(cfg.fit_intercept);
    let single_revert = cfg.revert_half_life.len() == 1;
    let shared_gap = if single_revert {
        gap_noise_portable(cfg, 0, d)
    } else {
        0.0
    };
    let mut corner = 0.0;
    for (i, &qi) in q.iter().enumerate().take(k) {
        let gap = if single_revert {
            shared_gap
        } else {
            gap_noise_portable(cfg, i, d)
        };
        let qd = qi * gap;
        if !cfg.standardize {
            if !(open && i < off) {
                p[i * k + i] += qd;
            }
            continue;
        }
        if p[i * k + i] == 0.0 || qd == 0.0 {
            continue;
        }
        if i < off {
            p[i * k + i] += qd;
            continue;
        }
        let a = map_a[i];
        if off == 1 {
            let v = map_c[i] / a;
            corner += v * v * qd;
            let o = -v * qd / a;
            p[i] += o;
            p[i * k] += o;
        }
        p[i * k + i] += qd / (a * a);
    }
    if corner != 0.0 {
        p[0] += corner;
    }
}

/// Evaluate UKF quadrature (`ScalarIndex` or `Merwe`) at `(eta0, p_prior, zs)`.
/// Returns `None` if `obs.mean_libm(eta0)` is undefined (`η_0 ≤ 0` under
/// inverse/power links, or non-finite).
fn eval_ukf(
    obs: &KalmanObs,
    eta0: f64,
    p_prior: &[f64],
    zs: &[f64],
    pz: &mut [f64],
    work: &mut UkfScratch,
) -> Option<UkfEval> {
    let (_, mu0) = obs.mean_libm(eta0)?;
    let k = zs.len();
    match obs.ukf_rule {
        UkfRule::ScalarIndex => {
            matvec(p_prior, zs, pz);
            let zpz: f64 = zs.iter().zip(pz.iter()).map(|(z, p)| z * p).sum();
            let s2 = crate::solve::clamp_rounding(zpz);
            if !(s2 >= 0.0 && s2.is_finite()) {
                return Some(UkfEval {
                    pred: a_number_or_none(mu0),
                    mu0,
                    fallback_ekf: true,
                    sc_delta: 0.0,
                    sc_mu_p: mu0,
                    sc_mu_m: mu0,
                    merwe_n: 0,
                });
            }
            if s2 == 0.0 {
                return Some(UkfEval {
                    pred: a_number_or_none(mu0),
                    mu0,
                    fallback_ekf: false,
                    sc_delta: 0.0,
                    sc_mu_p: mu0,
                    sc_mu_m: mu0,
                    merwe_n: 0,
                });
            }
            let delta = SQRT_3 * s2.sqrt();
            let (Some((_, mu_p)), Some((_, mu_m))) =
                (obs.mean_libm(eta0 + delta), obs.mean_libm(eta0 - delta))
            else {
                return Some(UkfEval {
                    pred: a_number_or_none(mu0),
                    mu0,
                    fallback_ekf: true,
                    sc_delta: 0.0,
                    sc_mu_p: mu0,
                    sc_mu_m: mu0,
                    merwe_n: 0,
                });
            };
            let zbar = (2.0 / 3.0) * mu0 + (1.0 / 6.0) * (mu_p + mu_m);
            Some(UkfEval {
                pred: a_number_or_none(zbar),
                mu0,
                fallback_ekf: false,
                sc_delta: delta,
                sc_mu_p: mu_p,
                sc_mu_m: mu_m,
                merwe_n: 0,
            })
        }
        UkfRule::Merwe => {
            work.idx.clear();
            for i in 0..k {
                if p_prior[i * k + i] > 0.0 {
                    work.idx.push(i);
                }
            }
            let n = work.idx.len();
            if n == 0 {
                return Some(UkfEval {
                    pred: a_number_or_none(mu0),
                    mu0,
                    fallback_ekf: false,
                    sc_delta: 0.0,
                    sc_mu_p: mu0,
                    sc_mu_m: mu0,
                    merwe_n: 0,
                });
            }
            let kf = k as f64;
            let c = obs.alpha * obs.alpha * (kf + obs.kappa);
            let lam_u = c - kf;
            work.chol.truncate(n * n);
            if work.chol.len() < n * n {
                work.chol.resize(n * n, 0.0);
            }
            for r in 0..n {
                let ir = work.idx[r];
                for col in 0..=r {
                    let ic = work.idx[col];
                    work.chol[r * n + col] = c * p_prior[ir * k + ic];
                }
            }
            if !cholesky(&mut work.chol, n) {
                return Some(UkfEval {
                    pred: a_number_or_none(mu0),
                    mu0,
                    fallback_ekf: true,
                    sc_delta: 0.0,
                    sc_mu_p: mu0,
                    sc_mu_m: mu0,
                    merwe_n: 0,
                });
            }
            for col in 0..n {
                let mut d_col = 0.0;
                for r in col..n {
                    d_col += zs[work.idx[r]] * work.chol[r * n + col];
                }
                let (Some((_, mp)), Some((_, mm))) =
                    (obs.mean_libm(eta0 + d_col), obs.mean_libm(eta0 - d_col))
                else {
                    return Some(UkfEval {
                        pred: a_number_or_none(mu0),
                        mu0,
                        fallback_ekf: true,
                        sc_delta: 0.0,
                        sc_mu_p: mu0,
                        sc_mu_m: mu0,
                        merwe_n: 0,
                    });
                };
                work.mu_p[col] = mp;
                work.mu_m[col] = mm;
            }
            let wi = 0.5 / c;
            let wm0_eff = lam_u / c + 2.0 * (k - n) as f64 * wi;
            let mut sum_pairs = 0.0;
            for col in 0..n {
                sum_pairs += work.mu_p[col] + work.mu_m[col];
            }
            let zbar = wm0_eff * mu0 + wi * sum_pairs;
            Some(UkfEval {
                pred: a_number_or_none(zbar),
                mu0,
                fallback_ekf: false,
                sc_delta: 0.0,
                sc_mu_p: 0.0,
                sc_mu_m: 0.0,
                merwe_n: n,
            })
        }
    }
}

/// Compute UKF innovation variance `S_z + R_eff` and gain `K` into `gain`,
/// given a completed [`UkfEval`]. Returns `Some((s_inn, r_eff, zbar))` if the
/// update is well-defined and finite; `None` if skipped.
#[allow(clippy::too_many_arguments)]
fn ukf_gain(
    obs: &KalmanObs,
    ev: &UkfEval,
    r_scale: f64,
    weight: f64,
    pz: &[f64],
    work: &mut UkfScratch,
    gain: &mut [f64],
) -> Option<(f64, f64, f64)> {
    let k = gain.len();
    let zbar = ev.pred;
    if !zbar.is_finite() {
        return None;
    }
    match obs.ukf_rule {
        UkfRule::ScalarIndex => {
            let (w0, w1) = (2.0 / 3.0, 1.0 / 6.0);
            let v_term = match obs.ukf_noise {
                UkfNoise::Plugin => obs.variance_libm(ev.mu0),
                UkfNoise::Expected => {
                    w0 * obs.variance_libm(ev.mu0)
                        + w1 * (obs.variance_libm(ev.sc_mu_p) + obs.variance_libm(ev.sc_mu_m))
                }
            };
            let r_eff = (r_scale * v_term) / weight;
            let (d0, dp, dm) = (ev.mu0 - zbar, ev.sc_mu_p - zbar, ev.sc_mu_m - zbar);
            let s_inn = w0 * d0 * d0 + w1 * (dp * dp + dm * dm) + r_eff;
            let slope = if ev.sc_delta > 0.0 {
                (ev.sc_mu_p - ev.sc_mu_m) / (2.0 * ev.sc_delta)
            } else {
                0.0
            };
            if !(r_eff > 0.0
                && r_eff.is_finite()
                && s_inn > 0.0
                && s_inn.is_finite()
                && slope.is_finite())
            {
                return None;
            }
            let scale = slope / s_inn;
            for i in 0..k {
                gain[i] = scale * pz[i];
            }
            gain.iter()
                .all(|g| g.is_finite())
                .then_some((s_inn, r_eff, zbar))
        }
        UkfRule::Merwe => {
            let n = ev.merwe_n;
            let kf = k as f64;
            let c = obs.alpha * obs.alpha * (kf + obs.kappa);
            let lam_u = c - kf;
            let wi = 0.5 / c;
            let wm0 = lam_u / c;
            let wc0 = wm0 + (1.0 - obs.alpha * obs.alpha + obs.beta);
            let wm0_eff = wm0 + 2.0 * (k - n) as f64 * wi;
            let wc0_eff = wc0 + 2.0 * (k - n) as f64 * wi;
            let v_term = match obs.ukf_noise {
                UkfNoise::Plugin => obs.variance_libm(ev.mu0),
                UkfNoise::Expected => {
                    let mut sv = 0.0;
                    for col in 0..n {
                        sv += obs.variance_libm(work.mu_p[col]) + obs.variance_libm(work.mu_m[col]);
                    }
                    wm0_eff * obs.variance_libm(ev.mu0) + wi * sv
                }
            };
            let r_eff = (r_scale * v_term) / weight;
            let d0 = ev.mu0 - zbar;
            let mut sz_pairs = 0.0;
            for col in 0..n {
                let dp = work.mu_p[col] - zbar;
                let dm = work.mu_m[col] - zbar;
                sz_pairs += dp * dp + dm * dm;
            }
            let s_inn = wc0_eff * d0 * d0 + wi * sz_pairs + r_eff;
            if !(r_eff > 0.0 && r_eff.is_finite() && s_inn > 0.0 && s_inn.is_finite()) {
                return None;
            }
            work.cross[..k].fill(0.0);
            for col in 0..n {
                let wdiff = wi * (work.mu_p[col] - work.mu_m[col]);
                for r in col..n {
                    work.cross[work.idx[r]] += wdiff * work.chol[r * n + col];
                }
            }
            for (g, &cr) in gain.iter_mut().zip(&work.cross[..k]) {
                *g = cr / s_inn;
            }
            gain.iter()
                .all(|g| g.is_finite())
                .then_some((s_inn, r_eff, zbar))
        }
    }
}

/// Compute EKF gain `K` into `gain` and return `Some((s_inn, mu))` if finite
/// and positive; `None` if skipped.
#[allow(clippy::too_many_arguments)]
fn ekf_gain(
    obs: &KalmanObs,
    eta0: f64,
    r_scale: f64,
    weight: f64,
    p_prior: &[f64],
    zs: &[f64],
    pz: &mut [f64],
    gain: &mut [f64],
) -> Option<(f64, f64)> {
    let (_, mu) = obs.mean_libm(eta0)?;
    let m_jac = obs.inv_link_deriv_libm(mu);
    let v_mu = obs.variance_libm(mu);
    let r_eff = (r_scale * v_mu) / weight;
    matvec(p_prior, zs, pz);
    let zpz: f64 = zs.iter().zip(pz.iter()).map(|(z, p)| z * p).sum();
    let s_inn = m_jac * m_jac * zpz + r_eff;
    if !(r_eff > 0.0 && r_eff.is_finite() && s_inn > 0.0 && s_inn.is_finite() && m_jac.is_finite())
    {
        return None;
    }
    let factor = m_jac / s_inn;
    for i in 0..gain.len() {
        gain[i] = factor * pz[i];
    }
    gain.iter().all(|g| g.is_finite()).then_some((s_inn, mu))
}

impl Kalman {
    /// Ensure nonlinear scratch buffers are sized.
    fn ensure_nl_work(&mut self) {
        let k = self.cfg.k_total();
        let m = self.cfg.n_targets;
        let n_p = self.p.len();
        self.nl_work
            .get_or_insert_with(|| Box::new(NonlinearWork::default()))
            .ensure(k, m, n_p);
    }

    /// Transition using portable `libm` exponentials.
    fn transition_portable(&mut self, d_clock: f64) {
        if !self.cfg.reverts() {
            return;
        }
        if self.cfg.revert_half_life.len() == 1 {
            self.phi.fill(phi_portable(&self.cfg, 0, d_clock));
        } else {
            for (i, ph) in self.phi.iter_mut().enumerate() {
                *ph = phi_portable(&self.cfg, i, d_clock);
            }
        }
        let mut g = std::mem::take(&mut self.gbuf);
        let coupled = self.coupling(&self.phi, &self.map_c, &self.live, &mut g);
        for b in &mut self.beta {
            transit(b, &self.phi, coupled.then_some(&g[..]));
        }
        for p in &mut self.p {
            forms::transit_cov::<false>(p, &self.phi, coupled.then_some(&g[..]), &mut self.vbuf);
        }
        if let Some(s) = self.smoother.as_deref_mut() {
            s.transit(&self.phi, coupled.then_some(&g[..]));
        }
        self.gbuf = g;
    }

    /// Size newly-usable slots of covariance `pi` and report `(added, seeded)`
    /// — whether any slot was newly sized and whether any slope was seeded from
    /// `coef_prior`.
    fn size_slots_counting(&mut self, pi: usize, noise: f64, intercept: bool) -> (bool, bool) {
        let v = self.cfg.p0 * noise;
        if !(v > 0.0 && v.is_finite()) {
            return (false, false);
        }
        let k = self.cfg.k_total();
        let off = usize::from(self.cfg.fit_intercept);
        let mut added = false;
        let mut seeded = false;
        for i in 0..k {
            if self.sized::<false>(pi, i) || !self.current_scale(i).1 {
                continue;
            }
            if i < off {
                if intercept {
                    self.add_slot_as::<false>(pi, 0, v, 1.0, None);
                    self.live[0] = true;
                    added = true;
                }
                continue;
            }
            if !self.live[i] {
                self.anchor_slot_here(i);
            }
            let a = self.map_a[i];
            let open = self.open::<false>(pi);
            let cc = (off == 1 && !open).then(|| self.map_c[i] / a);
            self.add_slot_as::<false>(pi, i, v, a, cc);
            self.live[i] = true;
            added = true;
            seeded |= self.seed_standardized_slot_prior(pi, i, off, open);
        }
        if seeded {
            self.refresh_zb_for_cov(pi);
        }
        (added, seeded)
    }

    /// Predictive variance `pred_var(x)` for non-Gaussian-identity observations.
    pub(super) fn pred_var_nonlinear(&self, x: &[f64]) -> Vec<f64> {
        let k = self.cfg.k_total();
        let zs = self.anchored(x);
        let obs = &self.cfg.obs;
        let r_scale = obs.obs_scale(&self.cfg, 1.0);
        let mut pz = vec![0.0; k];
        let mut work = NonlinearWork::default();
        work.ensure(k, self.cfg.n_targets, self.p.len());
        (0..self.cfg.n_targets)
            .map(|j| {
                let pi = if self.cfg.share_p { 0 } else { j };
                if self.is_unsized_as::<false>(pi) {
                    return f64::NAN;
                }
                let p = self.covariance_now(pi);
                let eta0: f64 = zs.iter().zip(&self.beta[j]).map(|(z, b)| z * b).sum();
                match obs.filter {
                    KalmanFilter::Ekf => {
                        let Some((_, mu)) = obs.mean_libm(eta0) else {
                            return f64::NAN;
                        };
                        let m = obs.inv_link_deriv_libm(mu);
                        let v = obs.variance_libm(mu);
                        matvec(&p, &zs, &mut pz);
                        let zpz: f64 = zs.iter().zip(&pz).map(|(z, pv)| z * pv).sum();
                        a_number_or_none(m * m * zpz + r_scale * v)
                    }
                    KalmanFilter::Ukf => {
                        let Some(ev) = eval_ukf(obs, eta0, &p, &zs, &mut pz, &mut work.ukf) else {
                            return f64::NAN;
                        };
                        if ev.fallback_ekf {
                            let m = obs.inv_link_deriv_libm(ev.mu0);
                            let v = obs.variance_libm(ev.mu0);
                            matvec(&p, &zs, &mut pz);
                            let zpz: f64 = zs.iter().zip(&pz).map(|(z, pv)| z * pv).sum();
                            return a_number_or_none(m * m * zpz + r_scale * v);
                        }
                        let mut gain = vec![0.0; k];
                        ukf_gain(obs, &ev, r_scale, 1.0, &pz, &mut work.ukf, &mut gain)
                            .map_or(f64::NAN, |(s, _, _)| a_number_or_none(s))
                    }
                    KalmanFilter::Linear => f64::NAN,
                }
            })
            .collect()
    }

    /// `OnlineModel::predict` for EKF and UKF (bit-identical to `step_nonlinear`).
    #[inline(never)]
    pub(super) fn predict_nonlinear(&self, x: &[f64], d_clock: f64) -> Step {
        if let Some(refused) = crate::model::refused_predict(self, x, d_clock) {
            return refused;
        }
        let m = self.cfg.n_targets;
        let k = self.cfg.k_total();
        let n_eff = self.stats.n_eff();
        let mut pred = vec![f64::NAN; m];
        if n_eff >= self.cfg.min_weight {
            let zs = self.anchored(x);
            let reverts = self.cfg.reverts();
            let live: Vec<bool> = (0..k).map(|i| self.is_live_as::<false>(i)).collect();
            let (mut map_a, mut map_c) = (vec![1.0; k], vec![0.0; k]);
            if self.cfg.standardize {
                self.map_into(&live, &mut map_a, &mut map_c);
            }
            let (phi, g, coupled) = if reverts {
                let phi: Vec<f64> = (0..k)
                    .map(|i| phi_portable(&self.cfg, i, d_clock))
                    .collect();
                let mut g = vec![0.0; k];
                let coupled = self.coupling(&phi, &map_c, &live, &mut g);
                (phi, g, coupled)
            } else {
                (vec![], vec![], false)
            };
            let g_opt = coupled.then_some(&g[..]);
            let mut vbuf = vec![0.0; k];
            let mut pz = vec![0.0; k];
            let mut qbuf = vec![0.0; k];
            let mut work = NonlinearWork::default();
            work.ensure(k, m, self.p.len());
            let obs = &self.cfg.obs;

            for (j, slot) in pred.iter_mut().enumerate() {
                let pi = if self.cfg.share_p { 0 } else { j };
                if self.wj[j] <= 0.0 {
                    continue;
                }
                let mut p_now = self.p[pi].clone();
                if reverts {
                    forms::transit_cov::<false>(&mut p_now, &phi, g_opt, &mut vbuf);
                }
                if forms::unsized_of::<false>(&p_now, k) {
                    continue;
                }
                let eta0 = if reverts {
                    let mut b = self.beta[j].clone();
                    transit(&mut b, &phi, g_opt);
                    zs.iter().zip(&b).map(|(z, bv)| z * bv).sum()
                } else {
                    zs.iter().zip(&self.beta[j]).map(|(z, bv)| z * bv).sum()
                };
                match obs.filter {
                    KalmanFilter::Ekf => {
                        if let Some((_, mu)) = obs.mean_libm(eta0) {
                            *slot = a_number_or_none(mu);
                        }
                    }
                    KalmanFilter::Ukf => {
                        let gap = self.elapsed[pi] + d_clock;
                        let open = self.open::<false>(pi);
                        let noise = if !obs.is_gaussian_identity()
                            || (self.units() && self.cfg.q.is_none())
                        {
                            1.0
                        } else {
                            self.readiness_noise(j)
                        };
                        if gap != 0.0 && !noise.is_nan() {
                            self.q_held_into(noise, &mut qbuf);
                            add_noise_to(&self.cfg, &qbuf, gap, open, &map_a, &map_c, &mut p_now);
                        }
                        if let Some(ev) = eval_ukf(obs, eta0, &p_now, &zs, &mut pz, &mut work.ukf) {
                            *slot = ev.pred;
                        }
                    }
                    KalmanFilter::Linear => {}
                }
            }
            if !self.cfg.horizons.is_empty() {
                let ahead = self.ahead_of_the_read(&pred, &zs, reverts, &phi, g_opt, d_clock);
                pred.extend_from_slice(&ahead);
            }
        }
        let per = 3 * self.cfg.horizons.len();
        pred.resize(m * (1 + per), f64::NAN);
        Step {
            pred,
            n_eff,
            extra: None,
        }
    }

    /// `OnlineModel::step_into` for EKF and UKF (all transcendentals via `libm`).
    #[inline(never)]
    pub(super) fn step_nonlinear(
        &mut self,
        x: &[f64],
        y: &[Option<f64>],
        d_clock: f64,
        weight: f64,
        out: &mut Step,
    ) {
        if let Some(refused) = crate::model::refused_step(self, x, y, d_clock, weight) {
            *out = refused;
            return;
        }
        if self.smoother.is_some() {
            self.smooth_begin(d_clock);
        }
        self.ensure_buffers();
        self.ensure_nl_work();

        let k = self.cfg.k_total();
        let m = self.cfg.n_targets;
        let n_p = self.p.len();
        let lam = self.cfg.decay.portable_factor(d_clock);

        if self.cfg.fit_intercept {
            self.zbuf[0] = 1.0;
            self.zbuf[1..].copy_from_slice(x);
        } else {
            self.zbuf.copy_from_slice(x);
        }

        for e in &mut self.elapsed {
            *e += d_clock;
        }
        self.moments_gap += d_clock;
        let gap_moments = self.moments_gap;

        if self.cfg.standardize {
            self.mark_live::<false>();
            if !self.map_fresh {
                self.form_map();
            }
        }

        self.transition_portable(d_clock);

        let mut zs = std::mem::take(&mut self.zs);
        self.anchored_into(&self.zbuf, &self.live, &mut zs);
        self.zs = zs;

        for (zb, b) in self.zb.iter_mut().zip(&self.beta) {
            *zb = self.zs.iter().zip(b).map(|(z, bv)| z * bv).sum();
        }

        let n_eff = self.stats.n_eff();
        let ready = n_eff >= self.cfg.min_weight;
        let pred = &mut out.pred;
        pred.clear();
        pred.resize(m, f64::NAN);

        let is_gauss_id = self.cfg.obs.is_gaussian_identity();
        let units = self.units();
        let diffuse = units && self.cfg.fit_intercept;
        let mut work = self.nl_work.take().expect("ensured above");

        // Determine valid observations per target (`weight > 0`, `takes(y)`,
        // and for inverse/power links a defined mean at `zb[j]`).
        for (j, (slot, &yj_opt)) in work.y_ok[..m].iter_mut().zip(y).enumerate() {
            *slot = yj_opt.filter(|&yj| {
                weight > 0.0
                    && self.cfg.obs.takes(yj)
                    && (self.cfg.obs.link.defined_at_zero()
                        || self.cfg.obs.mean_libm(self.zb[j]).is_some())
            });
        }
        for pi in 0..n_p {
            work.informs[pi] = if self.cfg.share_p {
                work.y_ok[..m].iter().any(Option::is_some)
            } else {
                work.y_ok[pi].is_some()
            };
        }

        // Gaussian-identity noise bookkeeping (unused by non-Gaussian-identity).
        if is_gauss_id {
            let shared_s2 = if self.cfg.share_p {
                let mut buf = std::mem::take(&mut self.sumbuf);
                let s2 = shared_noise(&mut buf, &self.sig2);
                self.sumbuf = buf;
                s2
            } else {
                f64::NAN
            };
            let mut shared_first: Option<f64> = None;
            for j in 0..m {
                let sigma2 = match self.cfg.obs_var {
                    Some(v) => v,
                    None => {
                        let s2 = if self.cfg.share_p {
                            shared_s2
                        } else {
                            self.sig2[j]
                        };
                        if s2 > 0.0 {
                            s2
                        } else if self.cfg.share_p {
                            *shared_first.get_or_insert_with(|| self.shared_first_noise(y, weight))
                        } else {
                            work.y_ok[j].map_or(0.0, |yj| {
                                let e = yj - self.zb[j];
                                first_noise(e * e)
                            })
                        }
                    }
                };
                self.noise[j] = sigma2;
            }
            if self.cfg.standardize && self.cfg.obs_var.is_none() && !self.cfg.noise_units {
                let square = |j: usize| {
                    let e = work.y_ok[j]? - self.zb[j];
                    Some(e * e).filter(|e2| *e2 > 0.0 && e2.is_finite())
                };
                if self.cfg.share_p {
                    let mut buf = std::mem::take(&mut self.sumbuf);
                    let sum = sum_ascending(&mut buf, (0..m).filter_map(square));
                    let n = buf.len() as f64;
                    self.sumbuf = buf;
                    if n > 0.0 {
                        self.basis[0].add(sum, n, weight);
                    }
                } else {
                    for (j, basis) in self.basis.iter_mut().enumerate().take(m) {
                        if let Some(e2) = square(j) {
                            basis.add(e2, 1.0, weight);
                        }
                    }
                }
            }
        } else {
            self.noise.fill(1.0);
        }

        // ---- Per-target path (`share_p == false`, always true for nonlinear obs) ----
        if !self.cfg.share_p {
            // Horizons are allowed only under Gaussian identity (`g⁻¹(η) = η`),
            // where `ahead_of_the_step` reads the pre-update `(beta, P)` and
            // needs only which targets have a finite prior prediction.
            let ahead = if !self.cfg.horizons.is_empty() {
                let mut pre = vec![f64::NAN; m];
                for (j, slot) in pre.iter_mut().enumerate() {
                    if (ready || !self.cfg.min_weight_holds_learning)
                        && self.predicts_as::<false>(j)
                    {
                        *slot = a_number_or_none(self.zb[j]);
                    }
                }
                self.ahead_of_the_step(&pre)
            } else {
                Vec::new()
            };
            for (j, pred_slot) in pred.iter_mut().enumerate() {
                let wants_pred =
                    (ready || !self.cfg.min_weight_holds_learning) && self.predicts_as::<false>(j);
                let informs = work.informs[j];
                let open = diffuse && !self.sized::<false>(j, 0);
                let obs = self.cfg.obs.clone();

                // Check whether `pred`'s process-noise scale matches the
                // observing update's scale so UKF can reuse one evaluation.
                let pred_noise = if !is_gauss_id || (units && self.cfg.q.is_none()) {
                    1.0
                } else {
                    self.readiness_noise(j)
                };
                let upd_noise = self.noise[j];
                let same_noise = !is_gauss_id
                    || self.cfg.obs_var.is_some()
                    || (units && self.cfg.q.is_none())
                    || (!pred_noise.is_nan() && pred_noise.to_bits() == upd_noise.to_bits());

                let mut cached_ukf: Option<UkfEval> = None;

                if informs {
                    let gap = self.elapsed[j];
                    self.elapsed[j] = 0.0;

                    if obs.filter == KalmanFilter::Ukf && wants_pred && !same_noise {
                        // Compute `pred` from `readiness_noise` before `upd_noise` touches `P`.
                        work.p_prior.copy_from_slice(&self.p[j]);
                        if gap != 0.0 && !pred_noise.is_nan() {
                            let mut q = std::mem::take(&mut self.qbuf);
                            self.q_held_into(pred_noise, &mut q);
                            add_noise_to(
                                &self.cfg,
                                &q,
                                gap,
                                open,
                                &self.map_a,
                                &self.map_c,
                                &mut work.p_prior,
                            );
                            self.qbuf = q;
                        }
                        if let Some(ev) = eval_ukf(
                            &obs,
                            self.zb[j],
                            &work.p_prior,
                            &self.zs,
                            &mut self.pz,
                            &mut work.ukf,
                        ) {
                            *pred_slot = ev.pred;
                        }
                    }

                    // Add process noise to `self.p[j]` (unless unstandardized unsized legacy).
                    let legacy_unsized = is_gauss_id
                        && !self.cfg.standardize
                        && !units
                        && self.is_unsized_as::<false>(j);
                    if !legacy_unsized && gap != 0.0 {
                        let mut q = std::mem::take(&mut self.qbuf);
                        self.q_held_into(upd_noise, &mut q);
                        add_noise_to(
                            &self.cfg,
                            &q,
                            gap,
                            open,
                            &self.map_a,
                            &self.map_c,
                            &mut self.p[j],
                        );
                        self.qbuf = q;
                    }

                    // Compute `pred` BEFORE sizing any new slots (Rule 2 + predict parity).
                    if obs.filter == KalmanFilter::Ekf {
                        if wants_pred && let Some((_, mu)) = obs.mean_libm(self.zb[j]) {
                            *pred_slot = a_number_or_none(mu);
                        }
                    } else if same_noise && wants_pred {
                        let ev = eval_ukf(
                            &obs,
                            self.zb[j],
                            &self.p[j],
                            &self.zs,
                            &mut self.pz,
                            &mut work.ukf,
                        );
                        if let Some(ref e) = ev {
                            *pred_slot = e.pred;
                        }
                        cached_ukf = ev;
                    }

                    // Size newly-usable slots (if any).
                    let newly_sized = if !is_gauss_id {
                        if self.cfg.standardize {
                            self.size_slots_counting(j, 1.0, true).0
                        } else if self.is_unsized_as::<false>(j) {
                            self.size_whole(j, self.cfg.p0);
                            true
                        } else {
                            false
                        }
                    } else if self.cfg.standardize {
                        if units {
                            self.size_slots_counting(j, 1.0, false).0
                        } else if let Some(r) = self.cfg.obs_var.or_else(|| self.basis[j].noise()) {
                            self.size_slots_counting(j, r, true).0
                        } else {
                            false
                        }
                    } else if legacy_unsized {
                        self.size_whole(j, self.cfg.p0 * upd_noise);
                        true
                    } else {
                        false
                    };
                    if newly_sized {
                        cached_ukf = None;
                    }

                    // Smoother links pending `(b_n, P_n)` to `(b⁻_{n+1}, P⁻_{n+1})`.
                    if self.smoother.is_some() {
                        self.smooth_link_one(j);
                    }

                    // Observation update.
                    let yj = work.y_ok[j].expect("informs implies y_ok");
                    let r_scale = if units {
                        1.0
                    } else {
                        obs.obs_scale(&self.cfg, upd_noise)
                    };

                    if is_gauss_id && diffuse && (open || self.wj[j] == 0.0) {
                        // Exact diffuse intercept step on first observation (Gaussian identity).
                        let zpz = self.read_pz::<false>(j);
                        let s_inn = zpz + 1.0 / weight;
                        let err = yj - self.zb[j];
                        if s_inn > 0.0 && s_inn.is_finite() && err.is_finite() {
                            self.beta[j][0] += err;
                            if open {
                                super::diffuse_step(&mut self.p[j], &self.pz, s_inn);
                            } else {
                                for i in 0..k {
                                    self.gain[i] = self.pz[i] / s_inn;
                                }
                                super::take_the_row(&mut self.p[j], &self.gain, &self.pz);
                            }
                        }
                    } else {
                        let use_ekf = match obs.filter {
                            KalmanFilter::Ekf => true,
                            KalmanFilter::Ukf => {
                                let ev = cached_ukf.or_else(|| {
                                    eval_ukf(
                                        &obs,
                                        self.zb[j],
                                        &self.p[j],
                                        &self.zs,
                                        &mut self.pz,
                                        &mut work.ukf,
                                    )
                                });
                                if let Some(ev) = ev {
                                    if ev.fallback_ekf {
                                        true
                                    } else {
                                        if let Some((s_inn, _, zbar)) = ukf_gain(
                                            &obs,
                                            &ev,
                                            r_scale,
                                            weight,
                                            &self.pz,
                                            &mut work.ukf,
                                            &mut self.gain,
                                        ) {
                                            let err = yj - zbar;
                                            if err.is_finite() {
                                                for (b, g) in
                                                    self.beta[j].iter_mut().zip(&self.gain)
                                                {
                                                    *b += g * err;
                                                }
                                                take_rank1_sym(&mut self.p[j], &self.gain, s_inn);
                                            }
                                        }
                                        false
                                    }
                                } else {
                                    false
                                }
                            }
                            KalmanFilter::Linear => false,
                        };
                        if use_ekf
                            && let Some((s_inn, mu)) = ekf_gain(
                                &obs,
                                self.zb[j],
                                r_scale,
                                weight,
                                &self.p[j],
                                &self.zs,
                                &mut self.pz,
                                &mut self.gain,
                            )
                        {
                            let err = yj - mu;
                            if err.is_finite() {
                                for (b, g) in self.beta[j].iter_mut().zip(&self.gain) {
                                    *b += g * err;
                                }
                                take_rank1_sym(&mut self.p[j], &self.gain, s_inn);
                            }
                        }
                    }

                    // EW residual variance & target weight.
                    let aged = lam * self.wsig[j];
                    self.wsig[j] = aged;
                    if pred_slot.is_finite() {
                        let resid = yj - *pred_slot;
                        let ws_new = aged + weight;
                        let s2 = (aged * self.sig2[j] + weight * resid * resid) / ws_new;
                        if s2.is_finite() {
                            self.sig2[j] = s2;
                            self.wsig[j] = ws_new;
                        }
                    }
                    self.wj[j] = lam * self.wj[j] + weight;

                    if self.smoother.is_some() {
                        self.smooth_observe_one(j, ready);
                    }
                } else {
                    // Row does not observe target `j`: emit `pred` without mutating `P` or `elapsed`.
                    if wants_pred {
                        match obs.filter {
                            KalmanFilter::Ekf => {
                                if let Some((_, mu)) = obs.mean_libm(self.zb[j]) {
                                    *pred_slot = a_number_or_none(mu);
                                }
                            }
                            KalmanFilter::Ukf => {
                                work.p_prior.copy_from_slice(&self.p[j]);
                                let gap = self.elapsed[j];
                                if gap != 0.0 && !pred_noise.is_nan() {
                                    let mut q = std::mem::take(&mut self.qbuf);
                                    self.q_held_into(pred_noise, &mut q);
                                    add_noise_to(
                                        &self.cfg,
                                        &q,
                                        gap,
                                        open,
                                        &self.map_a,
                                        &self.map_c,
                                        &mut work.p_prior,
                                    );
                                    self.qbuf = q;
                                }
                                if let Some(ev) = eval_ukf(
                                    &obs,
                                    self.zb[j],
                                    &work.p_prior,
                                    &self.zs,
                                    &mut self.pz,
                                    &mut work.ukf,
                                ) {
                                    *pred_slot = ev.pred;
                                }
                            }
                            KalmanFilter::Linear => {}
                        }
                    }
                    self.wj[j] *= lam;
                    self.wsig[j] *= lam;
                }
            }
            self.nl_work = Some(work);
            self.finish_nonlinear_row(lam, gap_moments, weight, ready, n_eff, &ahead, out);
            return;
        }

        // ---- Shared `P` path (`share_p == true`, legal only for Gaussian + identity) ----
        // `g⁻¹(η) = η`, `m = 1`, `V = 1`: all targets share `self.p[0]`.
        let obs = self.cfg.obs.clone();
        let open_shared = diffuse && !self.sized::<false>(0, 0);
        let informs = work.informs[0];
        let pred_noise = if units && self.cfg.q.is_none() {
            1.0
        } else {
            self.readiness_noise(0)
        };
        let upd_noise = self.noise[0];
        let gap = self.elapsed[0];

        // Compute `pred` for every ready target before mutating `self.p[0]`.
        for (j, pred_slot) in pred.iter_mut().enumerate() {
            if !((ready || !self.cfg.min_weight_holds_learning) && self.predicts_as::<false>(j)) {
                continue;
            }
            match obs.filter {
                KalmanFilter::Ekf => {
                    if let Some((_, mu)) = obs.mean_libm(self.zb[j]) {
                        *pred_slot = a_number_or_none(mu);
                    }
                }
                KalmanFilter::Ukf => {
                    work.p_prior.copy_from_slice(&self.p[0]);
                    if gap != 0.0 && !pred_noise.is_nan() {
                        let mut q = std::mem::take(&mut self.qbuf);
                        self.q_held_into(pred_noise, &mut q);
                        add_noise_to(
                            &self.cfg,
                            &q,
                            gap,
                            open_shared,
                            &self.map_a,
                            &self.map_c,
                            &mut work.p_prior,
                        );
                        self.qbuf = q;
                    }
                    if let Some(ev) = eval_ukf(
                        &obs,
                        self.zb[j],
                        &work.p_prior,
                        &self.zs,
                        &mut self.pz,
                        &mut work.ukf,
                    ) {
                        *pred_slot = ev.pred;
                    }
                }
                KalmanFilter::Linear => {}
            }
        }
        let ahead = if !self.cfg.horizons.is_empty() {
            self.ahead_of_the_step(pred)
        } else {
            Vec::new()
        };

        if informs {
            self.elapsed[0] = 0.0;
            let legacy_unsized = !self.cfg.standardize && !units && self.is_unsized_as::<false>(0);
            if !legacy_unsized && gap != 0.0 {
                let mut q = std::mem::take(&mut self.qbuf);
                self.q_held_into(upd_noise, &mut q);
                add_noise_to(
                    &self.cfg,
                    &q,
                    gap,
                    open_shared,
                    &self.map_a,
                    &self.map_c,
                    &mut self.p[0],
                );
                self.qbuf = q;
            }
            if self.cfg.standardize {
                if units {
                    let _ = self.size_slots_counting(0, 1.0, false);
                } else if let Some(r) = self.cfg.obs_var.or_else(|| self.basis[0].noise()) {
                    let _ = self.size_slots_counting(0, r, true);
                }
            } else if legacy_unsized {
                self.size_whole(0, self.cfg.p0 * upd_noise);
            }
            if self.smoother.is_some() {
                self.smooth_link_one(0);
            }
        }

        let mut shared_update = false;
        let mut shared_s = f64::NAN;
        let r_scale = if units {
            1.0
        } else {
            obs.obs_scale(&self.cfg, upd_noise)
        };
        let zpz = if informs {
            self.read_pz::<false>(0)
        } else {
            0.0
        };

        for (j, &pred_j) in pred.iter().enumerate() {
            let Some(yj) = work.y_ok[j] else {
                self.wj[j] *= lam;
                self.wsig[j] *= lam;
                continue;
            };
            if diffuse && (open_shared || self.wj[j] == 0.0) {
                let s_inn = zpz + 1.0 / weight;
                let err = yj - self.zb[j];
                if s_inn > 0.0 && s_inn.is_finite() && err.is_finite() {
                    self.beta[j][0] += err;
                    for i in 0..k {
                        self.gain[i] = self.pz[i] / s_inn;
                    }
                    shared_update = true;
                    shared_s = s_inn;
                }
            } else {
                let updated = match obs.filter {
                    KalmanFilter::Ekf => ekf_gain(
                        &obs,
                        self.zb[j],
                        r_scale,
                        weight,
                        &self.p[0],
                        &self.zs,
                        &mut self.pz,
                        &mut self.gain,
                    ),
                    KalmanFilter::Ukf => {
                        if let Some(ev) = eval_ukf(
                            &obs,
                            self.zb[j],
                            &self.p[0],
                            &self.zs,
                            &mut self.pz,
                            &mut work.ukf,
                        ) {
                            if ev.fallback_ekf {
                                ekf_gain(
                                    &obs,
                                    self.zb[j],
                                    r_scale,
                                    weight,
                                    &self.p[0],
                                    &self.zs,
                                    &mut self.pz,
                                    &mut self.gain,
                                )
                            } else {
                                ukf_gain(
                                    &obs,
                                    &ev,
                                    r_scale,
                                    weight,
                                    &self.pz,
                                    &mut work.ukf,
                                    &mut self.gain,
                                )
                                .map(|(s, _, zbar)| (s, zbar))
                            }
                        } else {
                            None
                        }
                    }
                    KalmanFilter::Linear => None,
                };
                if let Some((s_inn, mu)) = updated {
                    let err = yj - mu;
                    if err.is_finite() {
                        for (b, g) in self.beta[j].iter_mut().zip(&self.gain) {
                            *b += g * err;
                        }
                        shared_update = true;
                        shared_s = s_inn;
                    }
                }
            }
            let aged = lam * self.wsig[j];
            self.wsig[j] = aged;
            if pred_j.is_finite() {
                let resid = yj - pred_j;
                let ws_new = aged + weight;
                let s2 = (aged * self.sig2[j] + weight * resid * resid) / ws_new;
                if s2.is_finite() {
                    self.sig2[j] = s2;
                    self.wsig[j] = ws_new;
                }
            }
            self.wj[j] = lam * self.wj[j] + weight;
        }
        if shared_update {
            if open_shared {
                super::diffuse_step(&mut self.p[0], &self.pz, shared_s);
            } else {
                take_rank1_sym(&mut self.p[0], &self.gain, shared_s);
            }
        }
        if informs && self.smoother.is_some() {
            self.smooth_observe_one(0, ready);
        }
        self.nl_work = Some(work);
        self.finish_nonlinear_row(lam, gap_moments, weight, ready, n_eff, &ahead, out);
    }

    /// Finalize a nonlinear row: stats update, re-map, definiteness check,
    /// horizons and `min_weight` gate.
    #[allow(clippy::too_many_arguments)]
    fn finish_nonlinear_row(
        &mut self,
        lam: f64,
        gap_moments: f64,
        weight: f64,
        ready: bool,
        n_eff: f64,
        ahead: &[f64],
        out: &mut Step,
    ) {
        if self.keeps_history(gap_moments) {
            let had = lam * self.stats.n_eff() / self.cfg.decay.portable_factor(gap_moments);
            self.stats
                .update_keeping(&self.zbuf, lam, weight, weight, had);
        } else {
            self.stats.update(&self.zbuf, lam, weight);
        }
        if weight > 0.0 {
            self.moments_gap = 0.0;
        }
        if self.cfg.standardize && weight > 0.0 {
            self.follow_the_moments();
        }
        for pi in 0..self.p.len() {
            if self.lost_definiteness::<false>(pi) {
                self.size_again(pi);
            }
        }
        out.pred.extend_from_slice(ahead);
        if !ready {
            out.pred.fill(f64::NAN);
        }
        out.n_eff = n_eff;
        out.extra = None;
    }
}

#[cfg(test)]
mod tests {
    /// Every transcendental in `kalman/nonlinear.rs` calls the `libm` crate,
    /// never the platform's math library (matching `garch::tests::no_platform_math_library_is_called`).
    #[test]
    fn no_platform_math_library_is_called() {
        let src = include_str!("nonlinear.rs");
        let code: String = src
            .split("#[cfg(test)]")
            .next()
            .unwrap()
            .lines()
            .filter(|l| !l.trim_start().starts_with("//"))
            .collect::<Vec<_>>()
            .join("\n");
        let platform = [
            ".ln(", ".ln_1p(", ".log(", ".log2(", ".log10(", ".exp(", ".exp2(", ".exp_m1(",
            ".powf(", ".powi(", ".sin(", ".cos(", ".tan(", ".sinh(", ".cosh(", ".tanh(", ".asin(",
            ".acos(", ".atan(", ".atan2(", ".cbrt(", ".hypot(", "f64::ln", "f64::exp", "f64::pow",
        ];
        for call in platform {
            assert!(
                !code.contains(call),
                "kalman/nonlinear.rs calls `{call}`: route it through `libm`"
            );
        }
    }
}
