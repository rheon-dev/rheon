//! `Ar`: an autoregression, AR(p), fit by exponentially weighted least
//! squares on lags of each target that the model keeps itself.
//!
//! # The model
//!
//! For each target `y`, with lags `ℓ_1 < … < ℓ_q` (by default `1..=p`) and
//! the row's features `x` as exogenous regressors:
//!
//! ```text
//! y_t = c + Σ_i φ_i y_{t−ℓ_i} + β' x_t + e_t
//! ```
//!
//! `c` is present with `fit_intercept` (statsmodels' `trend="c"`) and absent
//! without it (`trend="n"`).
//!
//! # The recursion
//!
//! Each target keeps a ring of its last `max(lags)` **learned** values and
//! an [`EwRidge`] over the design `z_t = (y_{t−ℓ_1}, …, y_{t−ℓ_q}, x_t)`
//! read from that ring. Per row, decay `lam = decay.factor(d_clock)`,
//! weight `w`, with `W` the weight of every row and `W_j` target `j`'s own:
//!
//! ```text
//! pred_t = [1, z_t] · β_{t−1}                 out of sample: before the update
//! W'     = lam W + w                          (n_eff, before the row: W)
//! W_j'   = lam W_j + w                        (a learned row of target j only)
//! S_j'   = (lam W_j S_j + w z z')  / W_j'     (EW mean of z z', kept centered)
//! r_j'   = (lam W_j r_j + w z y_j) / W_j'     (EW mean of z y_j, kept centered)
//! β_j    = (S_j + ridge D)⁻¹ r_j              solved after every learned row
//! ```
//!
//! `D` is the identity without the intercept's slot; `ridge = 0`, the
//! default, is ordinary least squares, which with no decay is statsmodels'
//! `AutoReg` on the same rows, to rounding. The moments are kept centred
//! (the weighted Welford step), so a series far from zero loses nothing to
//! its level.
//!
//! **A learned row** of target `j` is one with a positive weight and the
//! target present. It is pushed onto the target's ring; it is learned into
//! the fit only once the ring already holds `max(lags)` values, since before
//! that its lags do not exist (statsmodels' `hold_back = max(lags)`). Every
//! other row -- a missing target, or one of the first `max(lags)` learned
//! rows -- only ages the fit by the clock. A row of weight 0 learns nothing
//! and moves no ring: the next row's lags are the ones before it, as if it
//! were not there (statsmodels' `missing="drop"`).
//!
//! **Lags count learned rows, not clock units.** "The value `ℓ` rows back"
//! has no meaning on a clock, so the lags are positions in the series of
//! learned values, and the half-life alone is on the clock. A clock break
//! (a session change, a gap past `gap_cap`) empties every ring
//! ([`OnlineModel::clear_lags`]): the rows on either side are no longer
//! adjacent, and the series starts again, its first `max(lags)` values
//! rebuilding the ring. The fit is kept across the break.
//!
//! # The state
//!
//! Per target, the ring (`8 max(lags)` bytes) and its fit, whose size is set
//! by `k = q + n_features + 1`; never by the stream's length.
//! [`Ar::state_bytes`] bounds it, and `max_state_bytes` caps that bound at
//! construction.

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

use crate::model::{ModelState, OnlineModel, State, StateError, Step, check_schema};
use crate::{Decay, EwRidge, EwRidgeCfg, TargetGaps};

/// The lags an [`Ar`] takes: at least one, each at least 1, strictly
/// increasing (their order is the coefficients' order), none past
/// [`crate::MAX_LAG`].
pub fn check_lags(lags: &[usize]) -> Result<(), String> {
    let mut no = crate::Refusals::new();
    if lags.is_empty() {
        no.refuse(
            "ar: lags must hold at least one lag, but it is empty. Give the order p, or a \
             list of lags such as [1, 2].",
        );
    }
    if lags.contains(&0) {
        no.refuse(
            "ar: lags must each be at least 1, but one of them is 0, which would regress a \
             value on itself. Remove the 0 from lags.",
        );
    }
    if lags.windows(2).any(|w| w[0] >= w[1]) {
        no.refuse(format!(
            "ar: lags must be strictly increasing, since their order is the coefficients' \
             order, but it is {lags:?}. List the lags in increasing order, each once."
        ));
    }
    if let Some(&max) = lags.iter().max() {
        no.take(crate::check_lag_ceiling("lags", max).map_err(|e| format!("ar: {e}")));
    }
    no.finish()
}

/// The settings of an [`Ar`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ArCfg {
    /// The series, one per target, each with its own lags and fit.
    pub n_targets: usize,
    /// Exogenous regressors, the row's features, shared by every target.
    pub n_features: usize,
    /// The lags, in learned rows: strictly increasing, each at least 1.
    /// `[1, …, p]` is AR(p).
    pub lags: Vec<usize>,
    /// The constant `c` (statsmodels' `trend="c"`); without it, `"n"`.
    pub fit_intercept: bool,
    pub decay: Decay,
    /// The ridge on the lag and feature coefficients (never the constant);
    /// 0 is least squares.
    pub ridge: f64,
    /// A target's prediction is withheld until the weight before the row
    /// reaches this.
    pub min_weight: f64,
    /// The bytes a group's state may hold; `None` is
    /// [`crate::DEFAULT_STATE_BYTES`]. The state's size is fixed by the
    /// settings, so it is checked once, when the model is built.
    #[serde(default)]
    pub max_state_bytes: Option<u64>,
}

impl ArCfg {
    /// The deepest lag: the ring's depth.
    pub fn max_lag(&self) -> usize {
        self.lags.iter().copied().max().unwrap_or(0)
    }

    /// Regressors per target, the constant excluded: the lags, then the
    /// features.
    pub fn n_regressors(&self) -> usize {
        self.lags.len() + self.n_features
    }

    /// Coefficients per target: the constant (with `fit_intercept`), the
    /// lags in order, then the features.
    pub fn k_total(&self) -> usize {
        self.n_regressors() + usize::from(self.fit_intercept)
    }

    /// The cap on the state's bytes.
    pub fn cap(&self) -> u64 {
        self.max_state_bytes
            .unwrap_or(crate::clock_span::DEFAULT_STATE_BYTES)
    }

    pub fn validate(&self) -> Result<(), String> {
        let mut no = crate::Refusals::new();
        no.take(self.decay.check().map_err(|e| format!("ar: {e}")));
        // Features are optional: a pure autoregression has none.
        if self.n_targets == 0 {
            no.refuse(
                "ar: the model needs at least one target, the series it regresses on its own \
                 lags, but it has none. Name at least one column in targets.",
            );
        }
        no.take(check_lags(&self.lags));
        if !(self.ridge.is_finite() && self.ridge >= 0.0) {
            no.refuse(format!(
                "ar: ridge must be finite and at least 0, but it is {}. Set ridge to 0 for \
                 least squares, or to a finite positive value.",
                self.ridge
            ));
        }
        no.take(crate::model::check_min_weight("ar", self.min_weight));
        no.take(crate::clock_span::check_max_state_bytes(
            "ar",
            self.max_state_bytes,
        ));
        no.stop_here()?;
        let bytes = Ar::state_bytes(self);
        if bytes > self.cap() {
            no.refuse(format!(
                "ar: the state of {} lags and {} features over {} targets would hold {bytes} \
                 bytes, past max_state_bytes of {}. Use fewer or shorter lags, or raise \
                 max_state_bytes.",
                self.lags.len(),
                self.n_features,
                self.n_targets,
                self.cap()
            ));
        }
        no.finish()
    }

    /// One target's fit: an [`EwRidge`] with a single target, solved after
    /// every learned row, its Gram over the target's own learned rows.
    fn ridge_cfg(&self) -> EwRidgeCfg {
        EwRidgeCfg {
            n_features: self.n_regressors(),
            n_targets: 1,
            fit_intercept: self.fit_intercept,
            decay: self.decay,
            ridge: vec![self.ridge],
            feature_sets: vec![],
            standardize: false,
            ridge_scale: false,
            coef_prior: None,
            session_shrink: None,
            long_half_life: None,
            // The gate is `Ar`'s own (`Ar::gate`): a fit that withheld its
            // predictions would age its residual variance on them alone, and
            // a floor, which withholds output only, would move the state.
            min_weight: 0.0,
            solve_every: 0.0,
            max_rows_between_solves: 1,
            solve_share: None,
            gram_block_rows: 0,
            target_gaps: TargetGaps::OwnRows,
            window: None,
            window_every: None,
            max_rows_between_snapshots: None,
        }
    }
}

/// An AR(p) per target, with exogenous features; see the module docs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Ar {
    cfg: ArCfg,
    /// One fit per target.
    fits: Vec<EwRidge>,
    /// Per target, its last `max(lags)` learned values, oldest first.
    rings: Vec<VecDeque<f64>>,
    /// Per target, whether its fit has been defined: its Gram positive
    /// definite (`Ar::definite`). Its predictions are withheld until then.
    /// Once set it stays: decay scales a Gram and a row adds to it, and
    /// neither takes a definite one back to singular.
    defined: Vec<bool>,
    /// The row's design, kept so a row allocates nothing: not state.
    #[serde(skip)]
    z: Vec<f64>,
    /// One fit's step, likewise.
    #[serde(skip)]
    one: Step,
}

impl Ar {
    pub fn new(cfg: ArCfg) -> Result<Self, String> {
        cfg.validate()?;
        let ridge = EwRidge::new(cfg.ridge_cfg())?;
        let depth = cfg.max_lag();
        Ok(Self {
            fits: vec![ridge; cfg.n_targets],
            rings: (0..cfg.n_targets)
                .map(|_| VecDeque::with_capacity(depth))
                .collect(),
            z: Vec::with_capacity(cfg.n_regressors()),
            one: Step::default(),
            defined: vec![false; cfg.n_targets],
            cfg,
        })
    }

    /// The smallest share of a regressor's spread that the regressors
    /// before it may leave unexplained for the Gram to count as definite.
    /// An exactly collinear design leaves rounding, about `1e-16` to
    /// `1e-14` of it; `1e-10` is past that and below any design whose
    /// coefficients carry a digit.
    pub const DEFINITE: f64 = 1e-10;

    /// Whether the system `fit` solves is positive definite: its regressors'
    /// second moments, centred with an intercept and raw without one, plus
    /// the ridge on the diagonal, factor by Cholesky with every pivot above
    /// [`Self::DEFINITE`] of its diagonal. The guard on the fit's early
    /// rows is this condition, not a count of rows: until the lags and the
    /// features have spread in every direction, the coefficients are not
    /// determined, and the fit reports none.
    fn definite(fit: &EwRidge, intercept: bool, n: usize, ridge: f64) -> bool {
        let (parts, _) = fit.gram_parts();
        let Some(part) = parts.first() else {
            return false;
        };
        let c = &part.cov;
        let off = c.k() - n;
        // The system solved: the moments plus the ridge on the diagonal.
        let at = |i: usize, j: usize| {
            let m = if intercept {
                c.cov(off + i, off + j)
            } else {
                c.raw(off + i, off + j)
            };
            if i == j { m + ridge } else { m }
        };
        let mut l = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..=i {
                let s: f64 = (0..j).map(|k| l[i * n + k] * l[j * n + k]).sum();
                let a = at(i, j) - s;
                if i == j {
                    let d = at(i, i);
                    if !(d > 0.0 && a > Self::DEFINITE * d) {
                        return false;
                    }
                    l[i * n + i] = a.sqrt();
                } else {
                    l[i * n + j] = a / l[j * n + j];
                }
            }
        }
        true
    }

    /// Keep each fit's factor, which the coefficients' standard errors are
    /// read from ([`OnlineModel::coef_variance`]), as `ewridge` keeps its
    /// own under `emit_se_coef`.
    pub fn set_keep_factor(&mut self, on: bool) {
        for fit in &mut self.fits {
            fit.set_keep_factor(on);
        }
    }

    pub fn cfg(&self) -> &ArCfg {
        &self.cfg
    }

    /// The bytes the state holds at most, by its settings: per target the
    /// ring, `8 max(lags)`, and the fit's arrays, at most `8 (4k² + 16k +
    /// 64)` for `k = k_total` (its centred Gram and the factor it solves
    /// with, the means, cross-moments and coefficients, and its scalars).
    /// `tests` holds this above the bytes a state writes, fresh and after a
    /// stream.
    pub fn state_bytes(cfg: &ArCfg) -> u64 {
        let k = cfg.k_total() as u64;
        let per = 8 * cfg.max_lag() as u64 + 8 * (4 * k * k + 16 * k + 64);
        per.saturating_mul(cfg.n_targets as u64)
    }

    /// Each target's coefficients, `k_total` each: the constant (with
    /// `fit_intercept`), the lags in the order given, then the features.
    /// `None` before the first fit.
    pub fn coefficients(&self) -> Option<Vec<Vec<f64>>> {
        let k = self.cfg.k_total();
        let mut out = Vec::with_capacity(self.fits.len());
        let mut any = false;
        for (fit, &defined) in self.fits.iter().zip(&self.defined) {
            // A fit not yet defined reports none.
            match fit.coefficients().filter(|_| defined) {
                Some(c) => {
                    any = true;
                    out.push(c[0].clone());
                }
                None => out.push(vec![f64::NAN; k]),
            }
        }
        any.then_some(out)
    }

    /// The weight of every row before the next (hard rule 8).
    pub fn n_eff(&self) -> f64 {
        self.fits[0].n_eff()
    }

    /// Target `j`'s ring, oldest first.
    pub fn ring(&self, j: usize) -> &VecDeque<f64> {
        &self.rings[j]
    }

    /// Whether target `j`'s ring holds every lag.
    fn full(&self, j: usize) -> bool {
        self.rings[j].len() == self.cfg.max_lag()
    }

    /// Target `j`'s design for the next row into `z`: its lags from the
    /// ring, then the features. Zeros in the lags' place while the ring is
    /// short: such a row is not learned, and its prediction is not read.
    fn design_into(
        lags: &[usize],
        ring: &VecDeque<f64>,
        depth: usize,
        x: &[f64],
        z: &mut Vec<f64>,
    ) {
        z.clear();
        if ring.len() == depth {
            z.extend(lags.iter().map(|&l| ring[depth - l]));
        } else {
            z.resize(lags.len(), 0.0);
        }
        z.extend_from_slice(x);
    }

    /// `pred` withheld, every target's, while the weight before the row is
    /// below `floor` (`min_weight`), as `ewridge` gates on its shared weight.
    fn gate(pred: &mut [f64], n_eff: f64, floor: f64) {
        if n_eff < floor {
            pred.fill(f64::NAN);
        }
    }

    /// The row's report, the state untouched: target `j`'s prediction from
    /// its fit, or NaN while its ring is short, gated at `floor`.
    fn read(&self, x: &[f64], d_clock: f64, floor: f64, out: &mut Step) {
        let depth = self.cfg.max_lag();
        let mut z = Vec::with_capacity(self.cfg.n_regressors());
        out.pred.clear();
        out.pred.resize(self.cfg.n_targets, f64::NAN);
        out.extra = None;
        out.n_eff = self.n_eff();
        for j in 0..self.cfg.n_targets {
            if !self.full(j) || !self.defined[j] {
                continue;
            }
            Self::design_into(&self.cfg.lags, &self.rings[j], depth, x, &mut z);
            out.pred[j] = self.fits[j].predict(&z, d_clock).pred[0];
        }
        Self::gate(&mut out.pred, out.n_eff, floor);
    }
}

impl OnlineModel for Ar {
    fn step_into(
        &mut self,
        x: &[f64],
        y: &[Option<f64>],
        d_clock: f64,
        weight: f64,
        out: &mut Step,
    ) {
        debug_assert_eq!(x.len(), self.cfg.n_features);
        debug_assert_eq!(y.len(), self.cfg.n_targets);
        if let Some(refused) = crate::model::refused_step(self, x, y, d_clock, weight) {
            *out = refused;
            return;
        }
        let m = self.cfg.n_targets;
        let depth = self.cfg.max_lag();
        out.pred.clear();
        out.pred.resize(m, f64::NAN);
        out.extra = None;
        let learns = weight > 0.0;
        for (j, target) in y.iter().enumerate() {
            let full = self.rings[j].len() == depth;
            Self::design_into(&self.cfg.lags, &self.rings[j], depth, x, &mut self.z);
            let yj = target.filter(|_| learns);
            // Every fit takes every row, so its weight and its clock are the
            // stream's; only a row with its lags and its target teaches it.
            let taught = if full { yj } else { None };
            self.fits[j].step_into(&self.z, &[taught], d_clock, weight, &mut self.one);
            if j == 0 {
                out.n_eff = self.one.n_eff;
            }
            if full && self.defined[j] {
                out.pred[j] = self.one.pred[0];
            }
            // The fit a learned row taught, checked until it is defined.
            if taught.is_some() && !self.defined[j] {
                self.defined[j] = Self::definite(
                    &self.fits[j],
                    self.cfg.fit_intercept,
                    self.cfg.n_regressors(),
                    self.cfg.ridge,
                );
            }
            if let Some(v) = yj {
                let ring = &mut self.rings[j];
                if ring.len() == depth {
                    ring.pop_front();
                }
                ring.push_back(v);
            }
        }
        Self::gate(&mut out.pred, out.n_eff, self.cfg.min_weight);
    }

    fn predict(&self, x: &[f64], d_clock: f64) -> Step {
        if let Some(refused) = crate::model::refused_predict(self, x, d_clock) {
            return refused;
        }
        let mut out = Step::default();
        self.read(x, d_clock, self.cfg.min_weight, &mut out);
        out
    }

    fn predict_with_floor(
        &mut self,
        x: &[f64],
        y: &[Option<f64>],
        d_clock: f64,
        floor: f64,
    ) -> Step {
        let _ = y;
        if let Some(refused) = crate::model::refused_predict(self, x, d_clock) {
            return refused;
        }
        let mut out = Step::default();
        self.read(x, d_clock, floor, &mut out);
        out
    }

    /// The rows behind a break are no longer the series' last values: every
    /// ring empties, and each target's fit waits for `max(lags)` learned
    /// rows before it learns again. The fits are kept.
    fn clear_lags(&mut self) {
        for ring in &mut self.rings {
            ring.clear();
        }
    }

    /// Each target's coefficients' variances over its noise, its fit's as
    /// `ewridge` reports them, target after target; NaN for a target whose
    /// fit is not defined yet.
    fn coef_variance(&self) -> Option<crate::CoefVariance> {
        let k = self.cfg.k_total();
        let mut out = Vec::with_capacity(self.fits.len());
        let mut any = false;
        for (fit, &defined) in self.fits.iter().zip(&self.defined) {
            match fit.coef_variance() {
                Some(crate::CoefVariance::PerNoise(v)) if defined => {
                    any = true;
                    out.push(v.into_iter().next().unwrap_or_else(|| vec![f64::NAN; k]));
                }
                _ => out.push(vec![f64::NAN; k]),
            }
        }
        any.then_some(crate::CoefVariance::PerNoise(out))
    }

    fn target_n_eff_into(&self, out: &mut Vec<f64>) -> bool {
        out.clear();
        out.extend(self.fits.iter().map(|f| f.target_weights()[0]));
        true
    }

    fn state(&self) -> State {
        State::new(ModelState::Ar(Box::new(self.clone())))
    }

    fn restore(s: &State) -> Result<Self, StateError> {
        check_schema(s)?;
        match &s.model {
            ModelState::Ar(m) => {
                let mut m = (**m).clone();
                crate::model::check_cfg("ar", m.cfg.validate())?;
                let depth = m.cfg.max_lag();
                let want = m.cfg.ridge_cfg();
                let shaped = m.fits.len() == m.cfg.n_targets
                    && m.defined.len() == m.cfg.n_targets
                    && m.rings.len() == m.cfg.n_targets
                    && m.rings.iter().all(|r| r.len() <= depth)
                    && m.fits.iter().all(|f| *f.cfg() == want);
                if !shaped {
                    return Err(crate::model::damaged(
                        "ar",
                        "its lags or its fits have the wrong shape",
                    ));
                }
                for ring in &mut m.rings {
                    ring.reserve_exact(depth - ring.len());
                }
                // Each fit through its own loader, which checks its shape and
                // rebuilds what it keeps out of the file.
                let fits = std::mem::take(&mut m.fits);
                for fit in fits {
                    let state = State::new(ModelState::EwRidge(Box::new(fit)));
                    let fit = EwRidge::restore_owned(state)
                        .map_err(|_| crate::model::damaged("ar", "a target's fit is damaged"))?;
                    m.fits.push(fit);
                }
                m.z = Vec::with_capacity(m.cfg.n_regressors());
                Ok(m)
            }
            other => Err(StateError::WrongModel {
                expected: "ar",
                found: other.kind(),
            }),
        }
    }

    fn n_targets(&self) -> usize {
        self.cfg.n_targets
    }

    fn n_features(&self) -> usize {
        self.cfg.n_features
    }
}

#[cfg(test)]
#[path = "ar_tests.rs"]
mod tests;
