//! `dist_fit`: an online fit of a distribution to each target, exponentially
//! weighted on the clock. The family is the normal, in scipy's `loc`/`scale`
//! parameterization (the scale is the standard deviation).
//!
//! **The estimand.** Per target, the rows `i` learned so far with weights
//! `w_i` and `λ_i` the product of the decay factors since row `i`:
//!
//! ```text
//! ω_i = w_i λ_i,      W = Σ ω_i
//! loc   = Σ ω_i y_i / W                      (or the fixed `floc`)
//! scale = √( Σ ω_i (y_i − loc)² / W )
//! ```
//!
//! the weighted maximum-likelihood fit of the normal (the variance with no
//! degrees-of-freedom correction, as `scipy.stats.norm.fit` and
//! statsmodels' `DescrStatsW(ddof=0)` give it). It is exact, one pass and
//! of bounded state: the weighted Welford recursion of [`crate::EwDiag`],
//! one slot a target,
//!
//! ```text
//! W' = λ W + w,   a = λ W / W',   b = w / W',   d = y − m
//! m' = m + b d,   c' = a c + a b d²
//! ```
//!
//! with `λ` the decay over the row's clock delta. With a fixed location the
//! recursion runs on `y − floc`, and the scale is the root of its raw second
//! moment `c + m²`.
//!
//! **What a row reports**, per target, all read from the fit before the row
//! is learned (out of sample): `loc`, `scale`; the quantile `loc + scale
//! Φ⁻¹(p)` at each configured level `p`; the lower-tail expected shortfall
//! `E[y | y ≤ q_α] = loc − scale φ(Φ⁻¹(α)) / α` at each configured `α`; the
//! probability integral transform `pit = Φ((y − loc)/scale)` and the log
//! score `log f(y) = −ln scale − ½ ln 2π − ½ z²` of the row's own target. A
//! target with no learned weight, under its `min_weight`, or whose variance
//! is exactly 0 (a constant stream) reports not a number in every slot, and
//! [`DistFit::withheld`] says which: a degenerate fit is withheld, never
//! divided by.
//!
//! **Weights and the clock.** `weight_sum` is the weight learned before the
//! row and before its own decay; each target's `min_weight` is read against
//! its own weight, the rows it was present on. A row of weight 0 handed to
//! the model ages every weight by its decay and learns nothing; a target
//! absent from a row ages its own weight the same way. Scaling every weight
//! changes nothing but the weights.

use serde::{Deserialize, Serialize};

use crate::model::{OnlineModel, State, StateError, Step, check_schema};
use crate::{Decay, EwDiag, ModelState};

/// The family a [`DistFit`] fits. Only the normal so far.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum DistFamily {
    /// The normal, `loc` the mean and `scale` the standard deviation.
    Normal,
}

/// Why a target reports nothing on a row ([`DistFit::withheld`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistWithheld {
    /// The target has learned no weight, or the decay has aged it to
    /// nothing (below the smallest normal double).
    NoWeight,
    /// The target's weight is under `min_weight`.
    MinWeight,
    /// The fitted variance is 0: every learned value was the same (or, with
    /// a fixed location, equal to it).
    DegenerateVariance,
}

impl DistWithheld {
    /// The reason's name, as an output column carries it.
    pub fn name(self) -> &'static str {
        match self {
            DistWithheld::NoWeight => "no_weight",
            DistWithheld::MinWeight => "min_weight",
            DistWithheld::DegenerateVariance => "degenerate_variance",
        }
    }
}

/// The settings of a [`DistFit`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DistFitCfg {
    pub n_targets: usize,
    pub family: DistFamily,
    /// The decay per clock unit.
    pub decay: Decay,
    /// The weight a target must have learned before it reports a fit.
    pub min_weight: f64,
    /// The levels of the reported quantiles, each in `(0, 1)`.
    pub quantiles: Vec<f64>,
    /// The levels of the lower-tail expected shortfalls, each in `(0, 1)`.
    pub es_levels: Vec<f64>,
    /// A fixed location (scipy's `floc`); `None` fits it.
    pub floc: Option<f64>,
}

impl DistFitCfg {
    /// The output slots per target: `loc`, `scale`, the quantiles, the
    /// shortfalls, `pit`, `log_score`.
    pub fn slots(&self) -> usize {
        4 + self.quantiles.len() + self.es_levels.len()
    }

    pub fn validate(&self) -> Result<(), String> {
        let mut no = crate::Refusals::new();
        if self.n_targets == 0 {
            no.refuse("dist_fit: it needs at least one target. Name a target column.");
        }
        if let Err(e) = self.decay.check() {
            no.refuse(format!("dist_fit: {e}"));
        }
        if let Err(e) = crate::model::check_min_weight("dist_fit", self.min_weight) {
            no.refuse(e);
        }
        for (what, levels) in [
            ("quantiles", &self.quantiles),
            ("es_levels", &self.es_levels),
        ] {
            let bad: Vec<String> = levels
                .iter()
                .filter(|p| !(**p > 0.0 && **p < 1.0))
                .map(f64::to_string)
                .collect();
            if !bad.is_empty() {
                no.refuse(format!(
                    "dist_fit: each of {what} must be a level strictly between 0 and 1, but \
                     these are not: {}. Give levels such as 0.05 or 0.95.",
                    bad.join(", ")
                ));
            }
        }
        if let Some(f) = self.floc
            && !crate::model::usable(f)
        {
            no.refuse(format!(
                "dist_fit: floc must be a finite number, but it is {f}. Give the fixed \
                 location, or leave floc out to fit it."
            ));
        }
        no.finish()
    }
}

/// An online normal fit to each target. See the module docs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DistFit {
    cfg: DistFitCfg,
    /// The weight learned by every row, before the next row's decay.
    w_sum: f64,
    /// Each target's weighted Welford moments, of `y − floc` with a fixed
    /// location and of `y` without.
    fit: Vec<EwDiag>,
}

const HALF_LN_TAU: f64 = 0.918_938_533_204_672_8; // ½ ln 2π
const INV_SQRT_TAU: f64 = 0.398_942_280_401_432_7; // 1/√(2π)

impl DistFit {
    pub fn new(cfg: DistFitCfg) -> Result<Self, String> {
        cfg.validate()?;
        Ok(Self {
            fit: (0..cfg.n_targets).map(|_| EwDiag::new(1)).collect(),
            w_sum: 0.0,
            cfg,
        })
    }

    pub fn cfg(&self) -> &DistFitCfg {
        &self.cfg
    }

    /// The weight learned before the next row, every row's.
    pub fn n_eff(&self) -> f64 {
        self.w_sum
    }

    /// Target `j`'s fit as it stands, `(loc, scale)`, whether or not it
    /// would be withheld.
    pub fn params(&self, j: usize) -> (f64, f64) {
        let f = &self.fit[j];
        let (hi, lo) = f.mean_pair(0);
        match self.cfg.floc {
            Some(floc) => (floc, crate::solve::clamp_rounding(f.raw(0)).sqrt()),
            None => (hi + lo, f.var(0).sqrt()),
        }
    }

    /// Why target `j` would report nothing on the next row, or `None` when
    /// it reports its fit.
    pub fn withheld(&self, j: usize) -> Option<DistWithheld> {
        // A weight in the subnormal range is a history the decay has aged
        // to nothing: forgotten, as one that underflows to 0 is.
        let w = self.fit[j].n_eff();
        if w.is_nan() || w < f64::MIN_POSITIVE {
            return Some(DistWithheld::NoWeight);
        }
        if w < self.cfg.min_weight {
            return Some(DistWithheld::MinWeight);
        }
        let (_, scale) = self.params(j);
        if scale.is_nan() || scale <= 0.0 {
            return Some(DistWithheld::DegenerateVariance);
        }
        None
    }

    /// The row's report, before it is learned, into `out.pred`.
    fn read_into(&self, y: &[Option<f64>], out: &mut Step) {
        let per = self.cfg.slots();
        let pred = &mut out.pred;
        pred.clear();
        pred.resize(self.cfg.n_targets * per, f64::NAN);
        for (j, slots) in pred.chunks_exact_mut(per).enumerate() {
            if self.withheld(j).is_some() {
                continue;
            }
            let (loc, scale) = self.params(j);
            slots[0] = loc;
            slots[1] = scale;
            let nq = self.cfg.quantiles.len();
            for (s, &p) in slots[2..2 + nq].iter_mut().zip(&self.cfg.quantiles) {
                *s = loc + scale * crate::special::ndtri(p);
            }
            for (s, &a) in slots[2 + nq..per - 2].iter_mut().zip(&self.cfg.es_levels) {
                let z = crate::special::ndtri(a);
                *s = loc - scale * INV_SQRT_TAU * libm::exp(-0.5 * z * z) / a;
            }
            if let Some(v) = y
                .get(j)
                .copied()
                .flatten()
                .filter(|v| crate::model::usable(*v))
            {
                let z = (v - loc) / scale;
                slots[per - 2] = crate::special::norm_cdf(z);
                slots[per - 1] = -libm::log(scale) - HALF_LN_TAU - 0.5 * z * z;
            }
        }
        out.n_eff = self.w_sum;
        out.extra = None;
    }
}

impl OnlineModel for DistFit {
    fn target_n_eff_into(&self, out: &mut Vec<f64>) -> bool {
        out.clear();
        out.extend(self.fit.iter().map(EwDiag::n_eff));
        true
    }

    fn step_into(
        &mut self,
        _x: &[f64],
        y: &[Option<f64>],
        d_clock: f64,
        weight: f64,
        out: &mut Step,
    ) {
        if let Some(refused) = crate::model::refused_step(self, &[], y, d_clock, weight) {
            *out = refused;
            return;
        }
        self.read_into(y, out);
        // A weight below the smallest normal double is a weight of 0: what
        // it would teach is below a double's resolution, and its reciprocal
        // is not finite.
        let weight = if weight < f64::MIN_POSITIVE {
            0.0
        } else {
            weight
        };
        let lam = self.cfg.decay.factor(d_clock);
        self.w_sum = lam * self.w_sum + weight;
        let shift = self.cfg.floc.unwrap_or(0.0);
        for (j, f) in self.fit.iter_mut().enumerate() {
            // A history the decay ages into the subnormal range is
            // forgotten, as one it ages to exactly 0 is: what is left of it
            // is below a double's resolution beside any row it meets.
            if f.n_eff() != 0.0 && lam * f.n_eff() < f64::MIN_POSITIVE {
                *f = EwDiag::new(1);
            }
            match y.get(j).copied().flatten() {
                Some(v) if weight > 0.0 => f.update_one(v - shift, lam, weight),
                _ => f.update_one(0.0, lam, 0.0),
            }
        }
    }

    fn predict(&self, _x: &[f64], _d_clock: f64) -> Step {
        self.predict_with(&[], &[], _d_clock)
    }

    /// The row's own target is read for `pit` and `log_score`, both scored
    /// against the fit before the row.
    fn predict_with(&self, _x: &[f64], y: &[Option<f64>], _d_clock: f64) -> Step {
        let mut out = Step::default();
        self.read_into(y, &mut out);
        out
    }

    fn state(&self) -> State {
        State::new(ModelState::DistFit(Box::new(self.clone())))
    }

    fn restore(s: &State) -> Result<Self, StateError> {
        check_schema(s)?;
        match &s.model {
            ModelState::DistFit(m) => {
                let m = (**m).clone();
                crate::model::check_cfg("dist_fit", m.cfg.validate())?;
                if m.fit.len() != m.cfg.n_targets || m.fit.iter().any(|f| f.k() != 1) {
                    return Err(crate::model::damaged(
                        "dist_fit",
                        "its moments have the wrong shape",
                    ));
                }
                Ok(m)
            }
            other => Err(StateError::WrongModel {
                expected: "dist_fit",
                found: other.kind(),
            }),
        }
    }

    fn n_targets(&self) -> usize {
        self.cfg.n_targets
    }

    /// The model reads no features.
    fn n_features(&self) -> usize {
        0
    }

    fn n_outputs(&self) -> usize {
        self.cfg.n_targets * self.cfg.slots()
    }
}

#[cfg(test)]
mod tests;
