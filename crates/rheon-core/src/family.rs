//! Exponential-family distributions and their links, for the generalized
//! linear models (`glm`; docs/PLAN.md task 500), written once so that a
//! later model that needs a family's variance, link or deviance -- a
//! dynamic GLM in a state-space filter, a point process's likelihood step
//! -- reads the same functions.
//!
//! A family is a variance function `V(μ)` and a unit deviance `d(y, μ)`;
//! a link is `η = g(μ)` with its inverse and derivative. Together they give
//! the two numbers iteratively reweighted least squares (IRLS, Fisher
//! scoring) reads at a linear predictor `η`:
//!
//! ```text
//! μ = g⁻¹(η)
//! W = 1 / (g'(μ)² V(μ))          the Fisher weight
//! z = η + (y − μ) g'(μ)           the working response
//! ```
//!
//! | family | `V(μ)` | `y` | links (default first) |
//! |---|---|---|---|
//! | Poisson | `μ` | `y ≥ 0` | log (canonical) |
//! | Gamma | `μ²` | `y > 0` | log; inverse `1/μ` (canonical) |
//! | binomial | `μ(1 − μ)` | `0 ≤ y ≤ 1` | logit (canonical) |
//! | Tweedie, power `p` | `μᵖ` | `y ≥ 0` for `1 < p < 2`, `y > 0` for `p > 2` | log; power `μ^(1−p)` (canonical) |
//!
//! The unit deviances, `0·ln 0 = 0`:
//!
//! ```text
//! Poisson    d = 2 (y ln(y/μ) − (y − μ))
//! Gamma      d = 2 (−ln(y/μ) + (y − μ)/μ)
//! binomial   d = 2 (y ln(y/μ) + (1 − y) ln((1 − y)/(1 − μ)))
//! Tweedie    d = 2 (y^(2−p)/((1−p)(2−p)) − y μ^(1−p)/(1−p) + μ^(2−p)/(2−p))
//! ```
//!
//! These are statsmodels' `Poisson`, `Gamma`, `Binomial` and
//! `Tweedie(var_power=p)` with their `Log`, `InversePower`, `Logit` and
//! `Power(1 − p)` links; `tests/test_second_opinion_glm.py` holds `W`, `z`
//! and `d` to statsmodels' family objects.
//!
//! **The log and logit links clip `η` to `±30`** ([`ETA_CLIP`]) before
//! `μ` is formed, so `μ` stays finite and, for the logit, inside `(0, 1)`,
//! where the weight is positive; the working response is formed at the
//! clipped `η`. **The inverse and power links need `η > 0`**: `μ` is
//! positive only there. At any other `η` the link gives no mean
//! ([`Family::working`] returns `None`), and a model withholds the
//! prediction and learns nothing from the row rather than clamp it.
//!
//! The binomial's log link (the relative-risk model) is not offered: its
//! mean `e^η` must stay below 1 and nothing in IRLS keeps it there. Its
//! own start `η = 0` is `μ = 1`, where the weight `μ/(1 − μ)` is infinite,
//! and statsmodels and glum both fail on a well-specified problem of it.

use serde::{Deserialize, Serialize};

/// The most `|η|` the log and logit links take before forming `μ`: `e^30`
/// is about `1.1e13`, and a logit's `μ` at `±30` is `1e-13` from its bound.
pub const ETA_CLIP: f64 = 30.0;

/// A family: the variance function and the unit deviance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GlmFamily {
    /// Counts, `V(μ) = μ`.
    Poisson,
    /// Positive amounts, `V(μ) = μ²`.
    Gamma,
    /// Proportions or 0/1 outcomes, `V(μ) = μ(1 − μ)`.
    Binomial,
    /// `V(μ) = μᵖ`, `p` the family's power: for `1 < p < 2` a Poisson sum
    /// of Gamma amounts, with an exact zero.
    Tweedie,
}

/// A link, `η = g(μ)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GlmLink {
    /// `η = ln μ`.
    Log,
    /// `η = ln(μ/(1 − μ))`.
    Logit,
    /// `η = 1/μ`, the Gamma family's canonical link.
    Inverse,
    /// `η = μ^(1−p)`, the Tweedie family's canonical link.
    Power,
}

impl GlmFamily {
    /// The family's name as a spec writes it.
    pub fn name(self) -> &'static str {
        match self {
            GlmFamily::Poisson => "poisson",
            GlmFamily::Gamma => "gamma",
            GlmFamily::Binomial => "binomial",
            GlmFamily::Tweedie => "tweedie",
        }
    }

    /// The family's default link: the log for the three positive families,
    /// the logit for the binomial.
    pub fn default_link(self) -> GlmLink {
        match self {
            GlmFamily::Binomial => GlmLink::Logit,
            _ => GlmLink::Log,
        }
    }

    /// The links the family takes, its default first.
    pub fn links(self) -> &'static [GlmLink] {
        match self {
            GlmFamily::Poisson => &[GlmLink::Log],
            GlmFamily::Gamma => &[GlmLink::Log, GlmLink::Inverse],
            GlmFamily::Binomial => &[GlmLink::Logit],
            GlmFamily::Tweedie => &[GlmLink::Log, GlmLink::Power],
        }
    }
}

impl GlmLink {
    /// The link's name as a spec writes it.
    pub fn name(self) -> &'static str {
        match self {
            GlmLink::Log => "log",
            GlmLink::Logit => "logit",
            GlmLink::Inverse => "inverse",
            GlmLink::Power => "power",
        }
    }

    /// Whether the link's mean is defined at `η = 0`: the log's and the
    /// logit's are, the inverse's and the power's are not.
    pub fn defined_at_zero(self) -> bool {
        matches!(self, GlmLink::Log | GlmLink::Logit)
    }
}

/// The two numbers IRLS reads at a linear predictor, and the mean.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Working {
    /// `μ = g⁻¹(η)`, `η` clipped under the log and logit links.
    pub mu: f64,
    /// The Fisher weight `W = 1/(g'(μ)² V(μ))`, above 0.
    pub w: f64,
    /// The working response `z = η + (y − μ) g'(μ)`.
    pub z: f64,
    /// `η` as `μ` was formed from it: clipped under the log and logit links.
    pub eta: f64,
}

/// A family with its link and, for the Tweedie family, its power.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Family {
    pub family: GlmFamily,
    pub link: GlmLink,
    /// The Tweedie power `p`; read by no other family.
    pub power: f64,
}

impl Family {
    /// A family with its link, or why not: a link the family does not take,
    /// or a Tweedie power that is not above 1, or is exactly 2. `who` leads
    /// the message.
    pub fn new(who: &str, family: GlmFamily, link: GlmLink, power: f64) -> Result<Self, String> {
        let f = Self {
            family,
            link,
            power,
        };
        f.check(who)?;
        Ok(f)
    }

    /// [`Self::new`]'s check of a family already built (a saved state's).
    pub fn check(&self, who: &str) -> Result<(), String> {
        let mut no = crate::Refusals::new();
        if !self.family.links().contains(&self.link) {
            let takes: Vec<&str> = self.family.links().iter().map(|l| l.name()).collect();
            let lead = format!(
                "{who}: the {} family takes the link {}, but it was given {}.",
                self.family.name(),
                takes.join(" or "),
                self.link.name(),
            );
            let rest = format!(
                "Set link to one of those, or leave it out for the family's default, {}.",
                self.family.default_link().name(),
            );
            no.refuse(match (self.family, self.link) {
                (GlmFamily::Binomial, GlmLink::Log) => format!(
                    "{lead} The log link's mean e^η must stay below 1, and nothing in the fit \
                     keeps it there. {rest}"
                ),
                _ => format!("{lead} {rest}"),
            });
        }
        if self.family == GlmFamily::Tweedie {
            let p = self.power;
            if !(p > 1.0 && p.is_finite()) || p == 2.0 {
                no.refuse(format!(
                    "{who}: the Tweedie family's var_power must be above 1, finite and not 2, \
                     but it is {p}. Set var_power between 1 and 2 for a count of positive \
                     amounts with exact zeros, such as 1.5, or above 2; for a power of 1 or 2 \
                     use the poisson or gamma family."
                ));
            }
        }
        no.finish()
    }

    /// `V(μ)`.
    pub fn variance(&self, mu: f64) -> f64 {
        match self.family {
            GlmFamily::Poisson => mu,
            GlmFamily::Gamma => mu * mu,
            GlmFamily::Binomial => mu * (1.0 - mu),
            GlmFamily::Tweedie => mu.powf(self.power),
        }
    }

    /// Whether `y` is in the family's support: a count is never negative,
    /// a Gamma amount (and a Tweedie one at `p > 2`) is above 0, and a
    /// binomial proportion is in `[0, 1]`. `-0.0` is 0.
    pub fn takes(&self, y: f64) -> bool {
        match self.family {
            GlmFamily::Poisson => y >= 0.0,
            GlmFamily::Gamma => y > 0.0,
            GlmFamily::Binomial => (0.0..=1.0).contains(&y),
            GlmFamily::Tweedie if self.power > 2.0 => y > 0.0,
            GlmFamily::Tweedie => y >= 0.0,
        }
    }

    /// What [`Self::takes`] refuses, in words for a message.
    pub fn support(&self) -> &'static str {
        match self.family {
            GlmFamily::Poisson => "a count, which is never negative",
            GlmFamily::Gamma => "a positive amount, above 0",
            GlmFamily::Binomial => "a proportion, from 0 to 1",
            GlmFamily::Tweedie if self.power > 2.0 => {
                "a positive amount, above 0, at a var_power above 2"
            }
            GlmFamily::Tweedie => "an amount, which is never negative",
        }
    }

    /// `g(μ)`.
    pub fn link(&self, mu: f64) -> f64 {
        match self.link {
            GlmLink::Log => mu.ln(),
            GlmLink::Logit => (mu / (1.0 - mu)).ln(),
            GlmLink::Inverse => 1.0 / mu,
            GlmLink::Power => mu.powf(1.0 - self.power),
        }
    }

    /// `g'(μ)`.
    pub fn link_deriv(&self, mu: f64) -> f64 {
        match self.link {
            GlmLink::Log => 1.0 / mu,
            GlmLink::Logit => 1.0 / (mu * (1.0 - mu)),
            GlmLink::Inverse => -1.0 / (mu * mu),
            GlmLink::Power => {
                let q = 1.0 - self.power;
                q * mu.powf(q - 1.0)
            }
        }
    }

    /// `μ = g⁻¹(η)` and `η` as it was read, or `None` where the link gives
    /// no positive, finite mean: `η` not finite, or `η ≤ 0` under the
    /// inverse and power links. The log and logit clip `η` to
    /// [`ETA_CLIP`] first.
    pub fn mean(&self, eta: f64) -> Option<(f64, f64)> {
        if !eta.is_finite() {
            return None;
        }
        let (eta, mu) = match self.link {
            GlmLink::Log => {
                let e = eta.clamp(-ETA_CLIP, ETA_CLIP);
                (e, e.exp())
            }
            GlmLink::Logit => {
                let e = eta.clamp(-ETA_CLIP, ETA_CLIP);
                (e, 1.0 / (1.0 + (-e).exp()))
            }
            GlmLink::Inverse if eta > 0.0 => (eta, 1.0 / eta),
            GlmLink::Power if eta > 0.0 => (eta, eta.powf(1.0 / (1.0 - self.power))),
            GlmLink::Inverse | GlmLink::Power => return None,
        };
        (mu.is_finite() && mu > 0.0).then_some((eta, mu))
    }

    /// The working weight and response at `η` for a target `y`: `None`
    /// where the link gives no mean ([`Self::mean`]) or the weight is not
    /// a positive, finite number. Each pair's closed form:
    ///
    /// ```text
    /// Poisson, log     W = μ          z = η + (y − μ)/μ
    /// Gamma, log       W = 1          z = η + (y − μ)/μ
    /// Gamma, inverse   W = μ²         z = η − (y − μ)/μ²
    /// binomial, logit  W = μ(1 − μ)   z = η + (y − μ)/(μ(1 − μ))
    /// Tweedie, log     W = μ^(2−p)    z = η + (y − μ)/μ
    /// Tweedie, power   W = μᵖ/q²      z = η + (y − μ) q μ^(q−1),  q = 1 − p
    /// ```
    pub fn working(&self, eta: f64, y: f64) -> Option<Working> {
        let (eta, mu) = self.mean(eta)?;
        let (w, z) = match (self.family, self.link) {
            (GlmFamily::Poisson, GlmLink::Log) => (mu, eta + (y - mu) / mu),
            (GlmFamily::Gamma, GlmLink::Log) => (1.0, eta + (y - mu) / mu),
            (GlmFamily::Gamma, GlmLink::Inverse) => (mu * mu, eta - (y - mu) / (mu * mu)),
            (GlmFamily::Binomial, GlmLink::Logit) => {
                let v = mu * (1.0 - mu);
                (v, eta + (y - mu) / v)
            }
            (GlmFamily::Tweedie, GlmLink::Log) => (mu.powf(2.0 - self.power), eta + (y - mu) / mu),
            (GlmFamily::Tweedie, GlmLink::Power) => {
                let q = 1.0 - self.power;
                (
                    mu.powf(self.power) / (q * q),
                    eta + (y - mu) * q * mu.powf(q - 1.0),
                )
            }
            // A pair `check` refuses: the general form.
            _ => {
                let d = self.link_deriv(mu);
                (1.0 / (d * d * self.variance(mu)), eta + (y - mu) * d)
            }
        };
        (w.is_finite() && w > 0.0 && z.is_finite()).then_some(Working { mu, w, z, eta })
    }

    /// The unit deviance `d(y, μ)` (the module docs), `0 · ln 0 = 0`.
    pub fn unit_deviance(&self, y: f64, mu: f64) -> f64 {
        // `y ln(y/m)`, 0 at `y = 0`.
        let xlogy = |y: f64, m: f64| if y == 0.0 { 0.0 } else { y * (y / m).ln() };
        match self.family {
            GlmFamily::Poisson => 2.0 * (xlogy(y, mu) - (y - mu)),
            GlmFamily::Gamma => 2.0 * (-(y / mu).ln() + (y - mu) / mu),
            GlmFamily::Binomial => 2.0 * (xlogy(y, mu) + xlogy(1.0 - y, 1.0 - mu)),
            GlmFamily::Tweedie => {
                let p = self.power;
                let a = if y > 0.0 {
                    y.powf(2.0 - p) / ((1.0 - p) * (2.0 - p))
                } else {
                    0.0
                };
                2.0 * (a - y * mu.powf(1.0 - p) / (1.0 - p) + mu.powf(2.0 - p) / (2.0 - p))
            }
        }
    }

    /// The start of a batch fit's intercept: `g` at the weighted mean of
    /// the targets, held inside the family's open range where the mean is
    /// at its edge (all zeros, all ones), so `g` is finite.
    pub fn start_eta(&self, ybar: f64) -> f64 {
        let m = match self.family {
            GlmFamily::Binomial => ybar.clamp(1e-12, 1.0 - 1e-12),
            _ => ybar.max(1e-12),
        };
        self.link(m)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fam(family: GlmFamily, link: GlmLink, p: f64) -> Family {
        Family::new("glm", family, link, p).unwrap()
    }

    fn every() -> Vec<Family> {
        vec![
            fam(GlmFamily::Poisson, GlmLink::Log, 0.0),
            fam(GlmFamily::Gamma, GlmLink::Log, 0.0),
            fam(GlmFamily::Gamma, GlmLink::Inverse, 0.0),
            fam(GlmFamily::Binomial, GlmLink::Logit, 0.0),
            fam(GlmFamily::Tweedie, GlmLink::Log, 1.5),
            fam(GlmFamily::Tweedie, GlmLink::Power, 1.5),
            fam(GlmFamily::Tweedie, GlmLink::Log, 2.5),
            fam(GlmFamily::Tweedie, GlmLink::Power, 3.0),
        ]
    }

    /// Each closed form of `working` is the general `W = 1/(g'² V)`, with
    /// `z = η` plus `(y − μ) g'`, from the family's own variance and link
    /// derivative, and `g(μ)` gives back the `η` the mean was formed at. The two sides share
    /// the libm calls only through `μ`, so they agree to a few ulps of the
    /// terms: 1e-13 relative covers the powf and the divisions.
    #[test]
    fn the_closed_forms_are_the_general_ones() {
        for f in every() {
            for &eta in &[0.05, 0.3, 1.2, 2.5] {
                let eta = if f.link.defined_at_zero() {
                    eta - 0.6
                } else {
                    eta
                };
                let y = if f.family == GlmFamily::Binomial {
                    0.3
                } else {
                    1.7
                };
                let wk = f.working(eta, y).unwrap();
                let d = f.link_deriv(wk.mu);
                let w = 1.0 / (d * d * f.variance(wk.mu));
                let z = eta + (y - wk.mu) * d;
                let close = |a: f64, b: f64| (a - b).abs() <= 1e-13 * (1.0 + b.abs());
                assert!(close(wk.w, w), "{f:?} at {eta}: W {} vs {w}", wk.w);
                assert!(close(wk.z, z), "{f:?} at {eta}: z {} vs {z}", wk.z);
                assert!(close(f.link(wk.mu), eta), "{f:?}: g(μ) {}", f.link(wk.mu));
            }
        }
    }

    /// The unit deviance is 0 at `μ = y` and its derivative in `μ` is
    /// `−2 (y − μ)/V(μ)`, the definition of a family's deviance: checked by
    /// a central difference, whose error at a step of 1e-6 is below 1e-8 of
    /// the derivative's size for these smooth functions.
    #[test]
    fn the_deviance_is_the_familys() {
        for f in every() {
            let y = if f.family == GlmFamily::Binomial {
                0.3
            } else {
                1.7
            };
            assert!(f.unit_deviance(y, y).abs() < 1e-14, "{f:?}");
            for &mu in &[0.2, 0.45, 0.8] {
                let mu = if f.family == GlmFamily::Binomial {
                    mu
                } else {
                    mu * 4.0
                };
                let h = 1e-6;
                let num = (f.unit_deviance(y, mu + h) - f.unit_deviance(y, mu - h)) / (2.0 * h);
                let want = -2.0 * (y - mu) / f.variance(mu);
                assert!(
                    (num - want).abs() < 1e-7 * (1.0 + want.abs()),
                    "{f:?} at μ {mu}: {num} vs {want}"
                );
            }
        }
    }

    #[test]
    fn a_link_the_family_does_not_take_is_refused_by_name() {
        let e = Family::new("glm", GlmFamily::Binomial, GlmLink::Log, 0.0).unwrap_err();
        assert!(
            e.contains("takes the link logit") && e.contains("below 1"),
            "{e}"
        );
        let e = Family::new("glm", GlmFamily::Poisson, GlmLink::Inverse, 0.0).unwrap_err();
        assert!(e.contains("poisson family takes the link log,"), "{e}");
        for p in [1.0, 2.0, 0.5, f64::NAN, f64::INFINITY] {
            let e = Family::new("glm", GlmFamily::Tweedie, GlmLink::Log, p).unwrap_err();
            assert!(e.contains("var_power must be above 1"), "{p}: {e}");
        }
        // The power is the Tweedie family's alone.
        Family::new("glm", GlmFamily::Gamma, GlmLink::Log, f64::NAN).unwrap();
    }

    #[test]
    fn the_support_is_the_familys() {
        let p = fam(GlmFamily::Poisson, GlmLink::Log, 0.0);
        assert!(p.takes(0.0) && p.takes(-0.0) && p.takes(3.0) && !p.takes(-1e-300));
        let g = fam(GlmFamily::Gamma, GlmLink::Log, 0.0);
        assert!(!g.takes(0.0) && g.takes(1e-300));
        let b = fam(GlmFamily::Binomial, GlmLink::Logit, 0.0);
        assert!(b.takes(0.0) && b.takes(1.0) && b.takes(0.5) && !b.takes(1.0 + 1e-15));
        let t = fam(GlmFamily::Tweedie, GlmLink::Log, 1.5);
        assert!(t.takes(0.0) && !t.takes(-1.0));
        let t = fam(GlmFamily::Tweedie, GlmLink::Log, 2.5);
        assert!(!t.takes(0.0) && t.takes(2.0));
    }

    /// The log and logit clip `η`; the inverse and power links give no
    /// mean at `η ≤ 0`, and none of the four at an `η` that is not finite.
    #[test]
    fn the_mean_is_clipped_or_withheld() {
        let p = fam(GlmFamily::Poisson, GlmLink::Log, 0.0);
        assert_eq!(p.mean(50.0), Some((ETA_CLIP, ETA_CLIP.exp())));
        assert_eq!(p.mean(-50.0), Some((-ETA_CLIP, (-ETA_CLIP).exp())));
        let b = fam(GlmFamily::Binomial, GlmLink::Logit, 0.0);
        let (_, mu) = b.mean(1e3).unwrap();
        assert!(mu < 1.0 && b.working(1e3, 1.0).unwrap().w > 0.0);
        for f in [
            fam(GlmFamily::Gamma, GlmLink::Inverse, 0.0),
            fam(GlmFamily::Tweedie, GlmLink::Power, 1.5),
        ] {
            assert!(f.mean(0.0).is_none() && f.mean(-1.0).is_none(), "{f:?}");
            assert!(f.working(-1.0, 1.0).is_none(), "{f:?}");
            assert!(f.mean(0.5).is_some(), "{f:?}");
        }
        for f in every() {
            assert!(f.mean(f64::NAN).is_none() && f.mean(f64::INFINITY).is_none());
        }
    }

    #[test]
    fn the_start_is_the_links_value_at_the_mean() {
        let b = fam(GlmFamily::Binomial, GlmLink::Logit, 0.0);
        assert!(b.start_eta(0.0).is_finite() && b.start_eta(1.0).is_finite());
        assert_eq!(b.start_eta(0.5), 0.0);
        let p = fam(GlmFamily::Poisson, GlmLink::Log, 0.0);
        assert!(p.start_eta(0.0).is_finite());
        assert_eq!(p.start_eta(1.0), 0.0);
        let g = fam(GlmFamily::Gamma, GlmLink::Inverse, 0.0);
        assert_eq!(g.start_eta(4.0), 0.25);
    }
}
