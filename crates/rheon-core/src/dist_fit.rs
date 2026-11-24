//! `dist_fit`: an online fit of a distribution to each target, exponentially
//! weighted on the clock. The family is the normal (below), Student's t
//! (`t_fit`), each in scipy's `loc`/`scale` parameterization (the normal's
//! scale is the standard deviation, the t's scipy's `scale`, with `df` its
//! degrees of freedom), or Hansen's skewed t (`skewt_fit`), whose `loc` is
//! the mean and `scale` the standard deviation, with `df` and the skew
//! `lambda`.
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
//! **Under the t family** a row reports `df` after `scale`, and its
//! quantiles `loc + scale t_ν⁻¹(p)`, shortfalls `loc + scale E[Z | Z ≤
//! t_α]`, `pit` and `log_score` are the t's (`student_t`). Two more reasons
//! withhold (`t_fit` has the tests): `not_converged`, every slot, while the
//! fit has not started or is further from its likelihood's maximum than
//! noise; and `shape_unidentified`, every slot but `loc` and `scale`, while
//! the tail cannot be told from the normal's.
//!
//! **Under the skewed t** a row reports `df` and `lambda` after `scale`, and
//! its quantiles, shortfalls, `pit` and `log_score` are Hansen's (`skew_t`).
//! The same two reasons withhold, `shape_unidentified` while the tail and
//! the skew together cannot be told from the normal's.
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

use skewt_fit::SkewTFit;
use t_fit::{TFit, TStatus};

/// The family a [`DistFit`] fits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum DistFamily {
    /// The normal, `loc` the mean and `scale` the standard deviation.
    Normal,
    /// Student's t, scipy's `loc`, `scale` and `df`, fitted by weighted
    /// maximum likelihood one row at a time (`t_fit`).
    StudentT,
    /// Hansen's skewed t, `loc` the mean, `scale` the standard deviation,
    /// `df` and the skew `lambda` as `arch`'s `SkewStudent` has them, fitted
    /// as the t is (`skewt_fit`).
    SkewT,
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
    /// The t's or the skewed t's fit has not started, or is further from
    /// the maximum of its weighted likelihood than sampling noise alone would
    /// put it: nothing reports.
    NotConverged,
    /// The t's tail, or the skewed t's tail and skew together, cannot yet be
    /// told from the normal's: `loc` and `scale` report, and every value that
    /// reads the shape does not.
    ShapeUnidentified,
}

impl DistWithheld {
    /// The reason's name, as an output column carries it.
    pub fn name(self) -> &'static str {
        match self {
            DistWithheld::NoWeight => "no_weight",
            DistWithheld::MinWeight => "min_weight",
            DistWithheld::DegenerateVariance => "degenerate_variance",
            DistWithheld::NotConverged => "not_converged",
            DistWithheld::ShapeUnidentified => "shape_unidentified",
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
    /// The output slots per target: `loc`, `scale`, the t's `df` (and the
    /// skewed t's `lambda`), the quantiles, the shortfalls, `pit`,
    /// `log_score`.
    pub fn slots(&self) -> usize {
        4 + self.shape_slots() + self.quantiles.len() + self.es_levels.len()
    }

    /// The family's shape slots after `scale`: the t's `df`, the skewed t's
    /// `df` and `lambda`, none for the normal.
    pub fn shape_slots(&self) -> usize {
        match self.family {
            DistFamily::Normal => 0,
            DistFamily::StudentT => 1,
            DistFamily::SkewT => 2,
        }
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

/// An online distribution fit to each target. See the module docs.
///
/// Its fields are written in this order, `t` and `skewt` only where a
/// family needs them (the `Serialize` below): `t` under the t, and `t`
/// empty then `skewt` under the skewed t, so a file written field by field
/// in place of by name (msgpack's compact form) still reads each into its
/// own field.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct DistFit {
    cfg: DistFitCfg,
    /// The weight learned by every row, before the next row's decay.
    w_sum: f64,
    /// Each target's weighted Welford moments, of `y − floc` with a fixed
    /// location and of `y` without: the normal's fit, and the t's start and
    /// weight.
    fit: Vec<EwDiag>,
    /// Each target's t fit under the t family; empty under the others, and
    /// not written under the normal.
    #[serde(default)]
    t: Vec<TFit>,
    /// Each target's skewed-t fit under the skewed t; empty, and not
    /// written, under the others.
    #[serde(default)]
    skewt: Vec<SkewTFit>,
}

impl Serialize for DistFit {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let with_skewt = !self.skewt.is_empty();
        let with_t = with_skewt || !self.t.is_empty();
        let n = 3 + usize::from(with_t) + usize::from(with_skewt);
        let mut st = s.serialize_struct("DistFit", n)?;
        st.serialize_field("cfg", &self.cfg)?;
        st.serialize_field("w_sum", &self.w_sum)?;
        st.serialize_field("fit", &self.fit)?;
        if with_t {
            st.serialize_field("t", &self.t)?;
        } else {
            st.skip_field("t")?;
        }
        if with_skewt {
            st.serialize_field("skewt", &self.skewt)?;
        } else {
            st.skip_field("skewt")?;
        }
        st.end()
    }
}

const HALF_LN_TAU: f64 = 0.918_938_533_204_672_8; // ½ ln 2π
const INV_SQRT_TAU: f64 = 0.398_942_280_401_432_7; // 1/√(2π)

impl DistFit {
    pub fn new(cfg: DistFitCfg) -> Result<Self, String> {
        cfg.validate()?;
        let t = match cfg.family {
            DistFamily::StudentT => (0..cfg.n_targets).map(|_| TFit::new()).collect(),
            _ => Vec::new(),
        };
        let skewt = match cfg.family {
            DistFamily::SkewT => (0..cfg.n_targets).map(|_| SkewTFit::new()).collect(),
            _ => Vec::new(),
        };
        Ok(Self {
            fit: (0..cfg.n_targets).map(|_| EwDiag::new(1)).collect(),
            w_sum: 0.0,
            t,
            skewt,
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
    /// would be withheld; the t's or the skewed t's before its start is the
    /// normal's.
    pub fn params(&self, j: usize) -> (f64, f64) {
        if let Some((loc, scale, _)) = self.t.get(j).and_then(TFit::params) {
            return (loc, scale);
        }
        if let Some((loc, scale, _, _)) = self.skewt.get(j).and_then(SkewTFit::params) {
            return (loc, scale);
        }
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
        let (_, scale) = self.normal_params(j);
        if scale.is_nan() || scale <= 0.0 {
            return Some(DistWithheld::DegenerateVariance);
        }
        let fixed = self.cfg.floc.is_some();
        let status = match self.t.get(j) {
            Some(t) => Some(t.status(w, fixed)),
            None => self.skewt.get(j).map(|s| s.status(w, fixed)),
        };
        match status {
            None | Some(TStatus::Identified) => None,
            Some(TStatus::NotConverged) => Some(DistWithheld::NotConverged),
            Some(TStatus::ShapeUnidentified) => Some(DistWithheld::ShapeUnidentified),
        }
    }

    /// Target `j`'s t or skewed-t tail `df = 1/η` as it stands, once its
    /// fit has started; `None` under the normal.
    pub fn df(&self, j: usize) -> Option<f64> {
        if let Some(t) = self.t.get(j) {
            return t.params().map(|(_, _, eta)| 1.0 / eta);
        }
        self.skewt
            .get(j)
            .and_then(SkewTFit::params)
            .map(|(_, _, eta, _)| 1.0 / eta)
    }

    /// Target `j`'s skewed-t skew `λ` as it stands, once its fit has
    /// started; `None` under the other families.
    pub fn lambda(&self, j: usize) -> Option<f64> {
        self.skewt
            .get(j)
            .and_then(SkewTFit::params)
            .map(|(_, _, _, lam)| lam)
    }

    /// The normal's `(loc, scale)` from the moments.
    fn normal_params(&self, j: usize) -> (f64, f64) {
        let f = &self.fit[j];
        let (hi, lo) = f.mean_pair(0);
        match self.cfg.floc {
            Some(floc) => (floc, crate::solve::clamp_rounding(f.raw(0)).sqrt()),
            None => (hi + lo, f.var(0).sqrt()),
        }
    }

    /// The row's report, before it is learned, into `out.pred`.
    fn read_into(&self, y: &[Option<f64>], out: &mut Step) {
        let per = self.cfg.slots();
        let pred = &mut out.pred;
        pred.clear();
        pred.resize(self.cfg.n_targets * per, f64::NAN);
        let first = 2 + self.cfg.shape_slots();
        let nq = self.cfg.quantiles.len();
        for (j, slots) in pred.chunks_exact_mut(per).enumerate() {
            match self.withheld(j) {
                None => {}
                Some(DistWithheld::ShapeUnidentified) => {
                    let (loc, scale) = self.params(j);
                    slots[0] = loc;
                    slots[1] = scale;
                    continue;
                }
                Some(_) => continue,
            }
            let (loc, scale) = self.params(j);
            slots[0] = loc;
            slots[1] = scale;
            let v = y
                .get(j)
                .copied()
                .flatten()
                .filter(|v| crate::model::usable(*v));
            let (qs, rest) = slots[first..].split_at_mut(nq);
            let es = &mut rest[..self.cfg.es_levels.len()];
            if let Some((_, _, eta, lam)) = self.skewt.get(j).and_then(SkewTFit::params) {
                let sh = skew_t::Shape::new(eta, lam);
                for (s, &p) in qs.iter_mut().zip(&self.cfg.quantiles) {
                    *s = loc + scale * sh.ppf(p);
                }
                for (s, &a) in es.iter_mut().zip(&self.cfg.es_levels) {
                    *s = loc + scale * sh.lower_es(a);
                }
                slots[2] = 1.0 / eta;
                slots[3] = lam;
                if let Some(v) = v {
                    let z = (v - loc) / scale;
                    slots[per - 2] = sh.cdf(z);
                    slots[per - 1] = sh.log_density(z) - libm::log(scale);
                }
                continue;
            }
            match self.t.get(j).and_then(TFit::params) {
                None => {
                    for (s, &p) in qs.iter_mut().zip(&self.cfg.quantiles) {
                        *s = loc + scale * crate::special::ndtri(p);
                    }
                    for (s, &a) in es.iter_mut().zip(&self.cfg.es_levels) {
                        let z = crate::special::ndtri(a);
                        *s = loc - scale * INV_SQRT_TAU * libm::exp(-0.5 * z * z) / a;
                    }
                    if let Some(v) = v {
                        let z = (v - loc) / scale;
                        slots[per - 2] = crate::special::norm_cdf(z);
                        slots[per - 1] = -libm::log(scale) - HALF_LN_TAU - 0.5 * z * z;
                    }
                }
                Some((_, _, eta)) => {
                    for (s, &p) in qs.iter_mut().zip(&self.cfg.quantiles) {
                        *s = loc + scale * student_t::ppf(p, eta);
                    }
                    for (s, &a) in es.iter_mut().zip(&self.cfg.es_levels) {
                        *s = loc + scale * student_t::lower_es(a, eta);
                    }
                    slots[2] = 1.0 / eta;
                    if let Some(v) = v {
                        let z = (v - loc) / scale;
                        slots[per - 2] = student_t::cdf(z, eta);
                        slots[per - 1] = student_t::log_density(z, eta) - libm::log(scale);
                    }
                }
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
        let fixed_loc = self.cfg.floc.is_some();
        for (j, f) in self.fit.iter_mut().enumerate() {
            let mut t = self.t.get_mut(j);
            let mut sk = self.skewt.get_mut(j);
            // A history the decay ages into the subnormal range is
            // forgotten, as one it ages to exactly 0 is: what is left of it
            // is below a double's resolution beside any row it meets.
            if f.n_eff() != 0.0 && lam * f.n_eff() < f64::MIN_POSITIVE {
                *f = EwDiag::new(1);
                if let Some(t) = t.as_deref_mut() {
                    *t = TFit::new();
                }
                if let Some(s) = sk.as_deref_mut() {
                    *s = SkewTFit::new();
                }
            }
            if let Some(t) = t.as_deref_mut() {
                t.decay(lam);
            }
            if let Some(s) = sk.as_deref_mut() {
                s.decay(lam);
            }
            match y.get(j).copied().flatten() {
                Some(v) if weight > 0.0 => {
                    let w_before = lam * f.n_eff();
                    if let Some(t) = t.as_deref_mut() {
                        t.learn_moments(f, v - shift, lam, weight);
                    }
                    if let Some(s) = sk.as_deref_mut() {
                        s.learn_moments(f, v - shift, lam, weight);
                    }
                    f.update_one(v - shift, lam, weight);
                    if let Some(t) = t {
                        if t.started() {
                            t.learn(v, weight, w_before, fixed_loc);
                        } else {
                            t.try_start(f, self.cfg.floc);
                        }
                    }
                    if let Some(s) = sk {
                        if s.started() {
                            s.learn(v, weight, w_before, fixed_loc);
                        } else {
                            s.try_start(f, self.cfg.floc);
                        }
                    }
                }
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
                let n = m.cfg.n_targets;
                let (n_t, n_skewt) = match m.cfg.family {
                    DistFamily::Normal => (0, 0),
                    DistFamily::StudentT => (n, 0),
                    DistFamily::SkewT => (0, n),
                };
                if m.fit.len() != m.cfg.n_targets
                    || m.fit.iter().any(|f| f.k() != 1)
                    || m.t.len() != n_t
                    || !m.t.iter().all(TFit::sound)
                    || m.skewt.len() != n_skewt
                    || !m.skewt.iter().all(SkewTFit::sound)
                {
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

mod skew_t;
mod skewt_fit;
mod student_t;
mod t_fit;
#[cfg(test)]
mod tests;
