//! Nonlinear observation and EKF/UKF filter settings for `kalman` specs:
//! parsing into [`rheon_core::KalmanObs`], word-list refusals, and validation.

use rheon_core::{
    Family, KalmanFamily, KalmanFilter, KalmanLink, KalmanObs, Refusals, UkfNoise, UkfRule,
};

use super::checks::positive;
use super::{ModelKind, Spec, num_label, spec_words_refusal};

pub(crate) const KALMAN_FAMILIES: [&str; 5] =
    ["gaussian", "poisson", "binomial", "gamma", "tweedie"];
pub(crate) const KALMAN_LINKS: [&str; 5] = ["identity", "log", "logit", "inverse", "power"];
pub(crate) const KALMAN_FILTERS: [&str; 3] = ["linear", "ekf", "ukf"];
pub(crate) const KALMAN_UKF_RULES: [&str; 2] = ["scalar_index", "merwe"];
pub(crate) const KALMAN_UKF_NOISE: [&str; 2] = ["expected", "plugin"];

fn parse_family(name: &str, s: Option<&str>) -> Result<KalmanFamily, String> {
    match s.unwrap_or("gaussian") {
        "gaussian" => Ok(KalmanFamily::Gaussian),
        "poisson" => Ok(KalmanFamily::Poisson),
        "binomial" => Ok(KalmanFamily::Binomial),
        "gamma" => Ok(KalmanFamily::Gamma),
        "tweedie" => Ok(KalmanFamily::Tweedie),
        other => Err(spec_words_refusal(name, "family", &KALMAN_FAMILIES, other)),
    }
}

fn parse_link(name: &str, family: KalmanFamily, s: Option<&str>) -> Result<KalmanLink, String> {
    match s {
        None => Ok(family.default_link()),
        Some("identity") => Ok(KalmanLink::Identity),
        Some("log") => Ok(KalmanLink::Log),
        Some("logit") => Ok(KalmanLink::Logit),
        Some("inverse") => Ok(KalmanLink::Inverse),
        Some("power") => Ok(KalmanLink::Power),
        Some(other) => Err(spec_words_refusal(name, "link", &KALMAN_LINKS, other)),
    }
}

fn parse_filter(name: &str, s: Option<&str>) -> Result<KalmanFilter, String> {
    match s.unwrap_or("linear") {
        "linear" => Ok(KalmanFilter::Linear),
        "ekf" => Ok(KalmanFilter::Ekf),
        "ukf" => Ok(KalmanFilter::Ukf),
        other => Err(spec_words_refusal(name, "filter", &KALMAN_FILTERS, other)),
    }
}

fn parse_ukf_rule(name: &str, s: Option<&str>) -> Result<UkfRule, String> {
    match s.unwrap_or("scalar_index") {
        "scalar_index" => Ok(UkfRule::ScalarIndex),
        "merwe" => Ok(UkfRule::Merwe),
        other => Err(spec_words_refusal(
            name,
            "ukf_rule",
            &KALMAN_UKF_RULES,
            other,
        )),
    }
}

fn parse_ukf_noise(name: &str, s: Option<&str>) -> Result<UkfNoise, String> {
    match s.unwrap_or("expected") {
        "expected" => Ok(UkfNoise::Expected),
        "plugin" => Ok(UkfNoise::Plugin),
        other => Err(spec_words_refusal(
            name,
            "ukf_noise",
            &KALMAN_UKF_NOISE,
            other,
        )),
    }
}

/// Resolve a `kalman` spec's observation and filter settings into
/// [`KalmanObs`].
pub(crate) fn kalman_obs(spec: &Spec) -> Result<KalmanObs, String> {
    let ModelKind::Kalman {
        family,
        link,
        var_power,
        dispersion,
        filter,
        ukf_rule,
        alpha,
        beta,
        kappa,
        ukf_noise,
        ..
    } = &spec.model
    else {
        return Ok(KalmanObs::default());
    };
    let family = parse_family(&spec.name, family.as_deref())?;
    let link = parse_link(&spec.name, family, link.as_deref())?;
    let filter = parse_filter(&spec.name, filter.as_deref())?;
    let ukf_rule = parse_ukf_rule(&spec.name, ukf_rule.as_deref())?;
    let ukf_noise = parse_ukf_noise(&spec.name, ukf_noise.as_deref())?;
    let var_power = if family == KalmanFamily::Tweedie {
        Some(var_power.unwrap_or(1.5))
    } else {
        *var_power
    };
    Ok(KalmanObs {
        family,
        link,
        var_power,
        dispersion: dispersion.unwrap_or(1.0),
        filter,
        ukf_rule,
        alpha: alpha.unwrap_or(1.0),
        beta: beta.unwrap_or(2.0),
        kappa: kappa.unwrap_or(0.0),
        ukf_noise,
    })
}

impl Spec {
    /// True when a `kalman` spec uses the Gaussian family with the identity
    /// link (or its defaults). Returns `false` for other models.
    pub(crate) fn kalman_is_gaussian_identity(&self) -> bool {
        let ModelKind::Kalman { family, link, .. } = &self.model else {
            return false;
        };
        matches!(family.as_deref(), None | Some("gaussian"))
            && matches!(link.as_deref(), None | Some("identity"))
    }

    /// True when a `kalman` spec uses one of the four non-Gaussian GLM families.
    pub(crate) fn kalman_is_non_gaussian(&self) -> bool {
        let ModelKind::Kalman { family, .. } = &self.model else {
            return false;
        };
        matches!(
            family.as_deref(),
            Some("poisson" | "binomial" | "gamma" | "tweedie")
        )
    }

    /// The underlying [`Family`] for a non-Gaussian `kalman` spec, used to
    /// hold targets to the family's support.
    pub(crate) fn kalman_glm_family(&self) -> Option<Family> {
        if !self.kalman_is_non_gaussian() {
            return None;
        }
        kalman_obs(self).ok()?.glm_family()
    }

    /// Validate `kalman`'s observation family, link, filter and UKF settings.
    pub(super) fn check_kalman_obs(&self, no: &mut Refusals) {
        let ModelKind::Kalman {
            coef_half_life,
            q,
            obs_var,
            share_p,
            covariance_form,
            horizons,
            family,
            link,
            var_power,
            dispersion,
            filter,
            ukf_rule,
            alpha,
            beta,
            kappa,
            ukf_noise,
            coef_prior,
            emit_loglik,
            emit_mahal,
            ..
        } = &self.model
        else {
            return;
        };

        let name = &self.name;

        let fam_ok = match parse_family(name, family.as_deref()) {
            Ok(f) => Some(f),
            Err(e) => {
                no.refuse(e);
                None
            }
        };

        let link_ok = match (fam_ok, link.as_deref()) {
            (Some(f), s) => match parse_link(name, f, s) {
                Ok(l) => Some(l),
                Err(e) => {
                    no.refuse(e);
                    None
                }
            },
            (None, Some(s)) => {
                if !KALMAN_LINKS.contains(&s) {
                    no.refuse(spec_words_refusal(name, "link", &KALMAN_LINKS, s));
                }
                None
            }
            (None, None) => None,
        };

        let filter_ok = match parse_filter(name, filter.as_deref()) {
            Ok(f) => Some(f),
            Err(e) => {
                no.refuse(e);
                None
            }
        };

        let rule_ok = match parse_ukf_rule(name, ukf_rule.as_deref()) {
            Ok(r) => Some(r),
            Err(e) => {
                no.refuse(e);
                None
            }
        };

        if let Err(e) = parse_ukf_noise(name, ukf_noise.as_deref()) {
            no.refuse(e);
        }

        // Refuse UKF-only options when `filter != "ukf"`, and Merwe-only options
        // when `ukf_rule != "merwe"`.
        if let Some(flt) = filter_ok {
            if flt != KalmanFilter::Ukf {
                let chosen = flt.name();
                for (key, given) in [
                    ("ukf_rule", ukf_rule.is_some()),
                    ("ukf_noise", ukf_noise.is_some()),
                    ("alpha", alpha.is_some()),
                    ("beta", beta.is_some()),
                    ("kappa", kappa.is_some()),
                ] {
                    if given {
                        no.refuse(format!(
                            "spec {name:?}: {key} applies only under filter=\"ukf\", but filter \
                             is {chosen:?}, which does not use it. Leave {key} out, or set \
                             filter=\"ukf\"."
                        ));
                    }
                }
            } else if let Some(rule) = rule_ok
                && rule != UkfRule::Merwe
            {
                let chosen = rule.name();
                for (key, given) in [
                    ("alpha", alpha.is_some()),
                    ("beta", beta.is_some()),
                    ("kappa", kappa.is_some()),
                ] {
                    if given {
                        no.refuse(format!(
                            "spec {name:?}: {key} applies only under ukf_rule=\"merwe\", but \
                             ukf_rule is {chosen:?}, which does not use it. Leave {key} out, or \
                             set ukf_rule=\"merwe\"."
                        ));
                    }
                }
            }

            if flt != KalmanFilter::Linear && !covariance_form.is_covariance() {
                no.refuse(format!(
                    "spec {name:?}: covariance_form=\"square_root\" applies only to \
                     filter=\"linear\", not filter={:?}: nonlinear EKF and UKF updates operate on \
                     the full covariance matrix. Leave covariance_form at its default.",
                    flt.name()
                ));
            }
            if flt != KalmanFilter::Linear {
                if *emit_loglik {
                    no.refuse(format!(
                        "spec {name:?}: emit_loglik applies only to the linear Gaussian Kalman \
                         filter (filter=\"linear\", family=\"gaussian\", link=\"identity\"), not \
                         filter={:?}. Leave emit_loglik=False.",
                        flt.name()
                    ));
                }
                if *emit_mahal {
                    no.refuse(format!(
                        "spec {name:?}: emit_mahal applies only to the linear Gaussian Kalman \
                         filter (filter=\"linear\", family=\"gaussian\", link=\"identity\"), not \
                         filter={:?}. Leave emit_mahal=False.",
                        flt.name()
                    ));
                }
            }
        }
        if let Some(prior) = coef_prior {
            let mut bad: Vec<String> = prior
                .iter()
                .flatten()
                .filter(|v| !v.is_finite())
                .map(|v| num_label(*v))
                .collect();
            bad.sort();
            bad.dedup();
            if !bad.is_empty() {
                no.refuse(format!(
                    "spec {name:?}: coef_prior must be finite, but it holds {}. Set each \
                     coefficient of coef_prior to a finite number.",
                    super::checks::listed(&bad)
                ));
            }
        }

        // Validate numeric UKF parameters when given.
        if let Some(a) = alpha.filter(|a| !(a.is_finite() && *a > 0.0)) {
            no.refuse(format!(
                "spec {name:?}: alpha must be finite and above 0, but it is {}. Set alpha to a \
                 finite positive sigma-point spread, such as the default 1.",
                num_label(a)
            ));
        }
        if let Some(b) = beta.filter(|b| !b.is_finite()) {
            no.refuse(format!(
                "spec {name:?}: beta must be finite, but it is {}. Set beta to a finite number, \
                 such as the default 2.",
                num_label(b)
            ));
        }
        if let Some(kap) = kappa.filter(|kap| !kap.is_finite()) {
            no.refuse(format!(
                "spec {name:?}: kappa must be finite, but it is {}. Set kappa to a finite number, \
                 such as the default 0.",
                num_label(kap)
            ));
        } else if filter_ok == Some(KalmanFilter::Ukf) && rule_ok == Some(UkfRule::Merwe) {
            let k = (self.k() + usize::from(self.fit_intercept)) as f64;
            let a = alpha.unwrap_or(1.0);
            let kap = kappa.unwrap_or(0.0);
            if a.is_finite() && a > 0.0 {
                let c = a * a * (k + kap);
                if !(k + kap > 0.0 && c > 0.0 && c.is_finite()) {
                    no.refuse(format!(
                        "spec {name:?}: the Merwe sigma-point scale must be finite and above 0, \
                         but with k = {}, alpha = {} and kappa = {} it is {}. Raise kappa above \
                         -{} or adjust alpha.",
                        self.k() + usize::from(self.fit_intercept),
                        num_label(a),
                        num_label(kap),
                        num_label(c),
                        self.k() + usize::from(self.fit_intercept),
                    ));
                }
            }
        }

        // Family/link pairing, var_power, dispersion, and nonlinear constraints.
        if let Some(fam) = fam_ok {
            if let Some(lnk) = link_ok
                && !fam.links().contains(&lnk)
            {
                let takes: Vec<&str> = fam.links().iter().map(|l| l.name()).collect();
                let lead = format!(
                    "spec {name:?}: the {} family takes the link {}, but it was given {}.",
                    fam.name(),
                    takes.join(" or "),
                    lnk.name()
                );
                let rest = format!(
                    "Set link to one of those, or leave it out for the family's default, {}.",
                    fam.default_link().name()
                );
                no.refuse(match (fam, lnk) {
                    (KalmanFamily::Binomial, KalmanLink::Log) => format!(
                        "{lead} The log link's mean e^η must stay below 1, and nothing in the fit \
                         keeps it there. {rest}"
                    ),
                    _ => format!("{lead} {rest}"),
                });
            }

            if fam == KalmanFamily::Tweedie {
                if let Some(p) = var_power.filter(|p| !(p.is_finite() && *p > 1.0 && *p != 2.0)) {
                    no.refuse(format!(
                        "spec {name:?}: the Tweedie family's var_power must be above 1, finite \
                         and not 2, but it is {}. Set var_power between 1 and 2 for a count of \
                         positive amounts with exact zeros, such as 1.5, or above 2; for a power \
                         of 1 or 2 use the poisson or gamma family.",
                        num_label(p)
                    ));
                }
            } else if var_power.is_some() {
                no.refuse(format!(
                    "spec {name:?}: var_power is the Tweedie family's power, but family is {:?}, \
                     which reads none. Leave var_power out, or set family=\"tweedie\".",
                    fam.name()
                ));
            }

            if fam == KalmanFamily::Gaussian {
                if dispersion.is_some() {
                    no.refuse(format!(
                        "spec {name:?}: dispersion is for the non-Gaussian families, but family \
                         is \"gaussian\". Set obs_var for a Gaussian observation variance, or \
                         leave dispersion out."
                    ));
                }
            } else if let Some(d) = dispersion.filter(|d| !(d.is_finite() && positive(*d))) {
                no.refuse(format!(
                    "spec {name:?}: dispersion must be finite and above 0, but it is {}. Set \
                     dispersion to a finite positive scale parameter, such as 1.",
                    num_label(d)
                ));
            }

            if let Some(lnk) = link_ok {
                let is_gauss_id = fam == KalmanFamily::Gaussian && lnk == KalmanLink::Identity;
                if !is_gauss_id {
                    if filter_ok == Some(KalmanFilter::Linear) {
                        no.refuse(format!(
                            "spec {name:?}: filter=\"linear\" applies only to family=\"gaussian\" \
                             with link=\"identity\", not family={:?} with link={:?}. Set \
                             filter=\"ekf\" or filter=\"ukf\".",
                            fam.name(),
                            lnk.name()
                        ));
                    }
                    if fam == KalmanFamily::Gaussian {
                        if obs_var.is_none() {
                            no.refuse(format!(
                                "spec {name:?}: family=\"gaussian\" with link={:?} needs a fixed \
                                 obs_var, because the nonlinear filter reads P on an absolute \
                                 scale. Set obs_var to a finite positive variance.",
                                lnk.name()
                            ));
                        }
                    } else if obs_var.is_some() {
                        no.refuse(format!(
                            "spec {name:?}: obs_var cannot be set with family={:?}: the \
                             observation variance is the family's variance scaled by dispersion. \
                             Leave obs_var out and set dispersion instead.",
                            fam.name()
                        ));
                    }
                    if coef_half_life.is_some() {
                        no.refuse(format!(
                            "spec {name:?}: coef_half_life cannot be used with family={:?} and \
                             link={:?}: coef_half_life's gain matching assumes a constant Gaussian \
                             information per row. Give q (one finite variance per coefficient) \
                             and leave out coef_half_life.",
                            fam.name(),
                            lnk.name()
                        ));
                    } else if q.is_none() {
                        no.refuse(format!(
                            "spec {name:?}: family={:?} with link={:?} needs an explicit q, but q \
                             is not set. Give q (one finite variance per coefficient, intercept \
                             first where fit_intercept=True).",
                            fam.name(),
                            lnk.name()
                        ));
                    }
                    if !lnk.defined_at_zero() && !self.fit_intercept {
                        no.refuse(format!(
                            "spec {name:?}: the {} link gives a mean only where η > 0, and \
                             without an intercept every coefficient starts at 0. Set \
                             fit_intercept=True, or use the log link.",
                            lnk.name()
                        ));
                    }
                    if *share_p {
                        no.refuse(format!(
                            "spec {name:?}: share_p cannot be used with a nonlinear observation: \
                             each target's Jacobian and observation variance depend on its own \
                             coefficients, so targets cannot share one covariance. Leave \
                             share_p=False."
                        ));
                    }
                    if horizons.as_ref().is_some_and(|h| !h.is_empty()) {
                        no.refuse(format!(
                            "spec {name:?}: forecast_spans is only supported for \
                             family=\"gaussian\" with link=\"identity\", whose mean and integral \
                             are linear in the coefficients. Leave out forecast_spans."
                        ));
                    }
                    if filter_ok == Some(KalmanFilter::Linear) {
                        if *emit_loglik {
                            no.refuse(format!(
                                "spec {name:?}: emit_loglik applies only to the linear Gaussian \
                                 Kalman filter (filter=\"linear\", family=\"gaussian\", \
                                 link=\"identity\"), not family={:?} with link={:?}. Leave \
                                 emit_loglik=False.",
                                fam.name(),
                                lnk.name()
                            ));
                        }
                        if *emit_mahal {
                            no.refuse(format!(
                                "spec {name:?}: emit_mahal applies only to the linear Gaussian \
                                 Kalman filter (filter=\"linear\", family=\"gaussian\", \
                                 link=\"identity\"), not family={:?} with link={:?}. Leave \
                                 emit_mahal=False.",
                                fam.name(),
                                lnk.name()
                            ));
                        }
                    }
                }
            }
        }
    }
}
