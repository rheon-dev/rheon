//! Holt's linear trend method (docs/ENHANCEMENTS.md E25).
//!
//! The forecasting baseline every other model here should have to beat: no
//! features at all, just a level and a slope extrapolated forward. If a
//! regression cannot outperform "the series is going up at about this rate",
//! the features are not earning their place.
//!
//! Per row and target, with `s` the clock since the target was last observed
//! -- this row's delta included -- `w` the row's weight, `e = y − pred` the
//! forecast error, and `W`, `V` the weight the level and the trend have
//! gathered, each decayed on its own half-life, `λ_l = 0.5^(s/level_half_life)`
//! and `λ_b = 0.5^(s/trend_half_life)`:
//!
//! ```text
//! pred  = l + b·s                                  (extrapolate s clock units ahead)
//! textbook:  g_t = w/(λ_l·W + w)                   W' = λ_l·W + w
//!            h_t = w/(λ_b·V + w) · g/s             V' = λ_b·V + w    (s > 0)
//! line fit:  S0' = S0 + w     ā' = S0·(ā + s)/S0'    Q' = Q + S0·w·(ā + s)²/S0'
//!            g_f = w·(1/S0' + ā'²/Q')              h_f = w·ā'/Q'
//! l' = pred + g·e     g = max(g_f, g_t)
//! b' = b + h·e        h = max(h_f, h_t)
//! ```
//!
//! **The textbook gains** are weighted means, of what the row observes and
//! what the state forecast, as every accumulator here is: `l'` is then
//! `(λ_l·W·pred + w·y)/(λ_l·W + w)`, and `b'` the weighted mean of `b` and
//! the slope the level moved at, `(l' − l)/s`. A row at weight `w` counts `w`
//! times. At `w = 1` with `W` saturated this is the textbook recursion, `α =
//! 1 − λ_l`, `β = 1 − λ_b`. The textbook form read the weight only as "learn
//! or not" -- a row at weight 0.5 moved the level as far as one at 1 -- and at
//! an infinite half-life its rate was 0, so the level froze at the first row
//! and a trend never learned (review 2026-09-12, S29/S30).
//!
//! **The line fit's gains** are those of the weighted least-squares line
//! through every observation so far, on the clock, read at the last: `S0` its
//! weight, `ā` the observations' weighted mean age and `Q = Σ w·(age − ā)²`
//! the spread of the ages, none decayed. In the coordinates (level now,
//! slope) the fit's information is `[[S0, −S0·ā], [−S0·ā, Q + S0·ā²]]`, and
//! `(g_f, h_f)` is `w` times the first column of its inverse. With every
//! observation at one clock (`Q = 0`) no slope is determined: `g_f = w/S0'`,
//! `h_f = 0`. The weighted means alone gained `1/n` on both the level and the
//! trend, which telescoped the trend to `(l − y₁)/(t − t₁)`, pinned to the
//! first observation, and the level, which adds the trend every row, carried
//! that observation's error times `ln n` (docs/PLAN.md task 278, D-H: 10.96
//! off a flat series at 40,000 rows without decay, the worst of 50 seeds at a
//! noise of 0.5; 6.33 at 10,000 rows at `half_life = 10,000`).
//!
//! **A row takes the larger gain**, on the level and on the slope. An infinite
//! half-life has no textbook gain to give (its fixed gain `1 − λ` is 0), so
//! without decay the model *is* the weighted line fit, to rounding, whatever
//! the weights and the clock. Under a half-life the line fit's gains fall as
//! `1/n` and `1/n²` and the textbook's settle at `1 − λ`, so after a warm-up of
//! a few half-lives every row takes the textbook's for good, and statsmodels'
//! `Holt` agrees with the model from there; before, the warm-up is the line
//! fit, never slower than the textbook. `trend_half_life = inf` is the whole
//! history's drift: the line fit's gain on the slope, and the weighted
//! least-squares slope exactly only when the level does not decay either --
//! beside a finite level half-life the slope learns from errors against a
//! level that forgets, and lands near the line fit's, 2.2e-3 to 1.0e-2 of it
//! off at a level half-life of 5 (review round 10, S4-2). A row at the last row's clock (`s =
//! 0`) is a second observation the level takes in; the textbook trend holds,
//! since a move over no clock has no slope, and the line fit's moves by its
//! own gain, as a least-squares slope does.
//!
//! **A state saved before the line fit** (schema 63 and earlier) keeps none
//! and goes on with the textbook gains alone, the recursion it was fitted
//! under, to the bit.
//!
//! **Without a trend** (`trend = false`) the trend is held at zero and the
//! forecast is flat, `pred = l`: simple exponential smoothing. The level is
//! then the weighted mean of the observations, each at its weight times
//! `λ_l` per clock unit of its age, `Σ w·λ_l^age·y / Σ w·λ_l^age`, which
//! pandas' `ewm(adjust=True)` computes for unit weights (docs/PLAN.md task
//! 115, S30). Since S30 an infinite `trend_half_life` is the whole history's
//! drift, so this is the only way to ask for no trend.
//!
//! On a row that observes every target, `s` is the row's own delta `d`.
//! Deriving the decays from half-lives keeps the parameter meaning the same
//! as everywhere else in this library: a half-life is in clock units, so an
//! irregular clock is handled correctly instead of every row counting the
//! same.
//!
//! **Forecasts at horizons** (`forecast_spans`, spans of the clock `Δ ≥ 0` past
//! the row read): from the level and trend as they stand before the row,
//!
//! ```text
//! f(Δ) = l + b·(s + Δ)                         f(0) = pred
//! F(Δ) = ∫_0^Δ f(u) du = Δ·(l + b·(s + Δ/2))
//! ```
//!
//! the forecast `Δ` past the row and its integral over the span. Nothing
//! decays across the span: the half-lives weigh the evidence behind `l`
//! and `b`, not the extrapolation. A horizon is a span of the clock, never a
//! count of rows: the clock of rows to come is not known. Each target's
//! slots follow the predictions, two a horizon.
//!
//! **The fixed forms** (`start = known | heuristic`, task 488;
//! `holt/ets.rs`): a season `S` of `m` values over `P` clock units, added or
//! a factor, and a trend damped by `φ` a clock unit. Every gain is the fixed
//! one, `g = w·a/(λ + w·a)`, `a = 1 − λ` over the row's step (`a` itself at
//! `w = 1`), and, with `p` the row's phase `⌊m·(c − c₀)/P⌋ mod m` of its raw
//! clock and `L = l + φ_s·b`, `φ_s = φ(1 − φ^s)/(1 − φ)`:
//!
//! ```text
//! pred  = L + S[p]  |  L·S[p]  |  L
//! l'    = g·y − g·S[p] + (1 − g)·L  |  g·y/S[p] + (1 − g)·L  |  g·y + (1 − g)·L
//! b'    = h·(l' − l)/s + (1 − h)·φ^s·b
//! S'[p] = γ·y − γ·L + (1 − γ)·S[p]  |  γ·y/L + (1 − γ)·S[p]
//! ```
//!
//! `γ` the fixed gain over the clock since phase `p` was last learned.
//!
//! **A row the model cannot learn from is transparent.** A null target, or a
//! zero weight -- and a weight below the smallest normal double, `2⁻¹⁰²²`,
//! which is read as 0 -- leaves the level and slope where the last observation put
//! them and adds the row's delta to that target's `s`, so the next observed
//! row forecasts, and decays, over the whole gap: the same numbers as if the
//! row were absent and its clock folded into the next one, which is what
//! `clock.rs` does with a row it skips. The level used to stand still across
//! such a row, so the row after it forecast one trend step short (review
//! 2026-09-12, C22).

use serde::{Deserialize, Serialize};

use crate::model::{ModelState, OnlineModel, State, StateError, Step, check_schema};

mod ets;
#[cfg(test)]
mod ets_tests;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HoltCfg {
    pub n_targets: usize,
    /// Half-life of the level, in clock units. `inf` forgets nothing: the
    /// level is the whole history's.
    #[serde(with = "crate::humanfloat::f64_or_tag")]
    pub level_half_life: f64,
    /// Half-life of the trend. `inf` forgets no slope: the trend is the whole
    /// history's drift (it pinned the trend at zero before review S30).
    #[serde(with = "crate::humanfloat::f64_or_tag")]
    pub trend_half_life: f64,
    pub min_weight: f64,
    /// `false` holds the trend at zero: the level alone, with a flat
    /// forecast (see the module docs). A config written before the switch
    /// has none and loads with the trend on, what it was produced under.
    #[serde(default = "default_true")]
    pub trend: bool,
    // The fixed forms of seasonal exponential smoothing (task 488), each
    // appended with a default that is the form above, so a config written
    // before them loads as what it was produced under.
    /// The seasonal component: none, added to the level, or a factor on it.
    #[serde(default, rename = "seasonal")]
    pub season: HoltSeason,
    /// `m`, the values a season is held in: one per bucket of
    /// `period / season_periods` clock units. 0 without a season.
    #[serde(default, rename = "seasonal_periods")]
    pub season_periods: usize,
    /// `P`, the season's span in clock units (seconds on a temporal clock).
    /// 0 reads as `season_periods`: one bucket per clock unit, as on the
    /// row clock.
    #[serde(default)]
    pub period: f64,
    /// The half-life a seasonal value forgets at, in clock units: on an
    /// even clock it is learned once a season, at `γ = 1 − 0.5^(P/h)`.
    #[serde(default = "infinite", with = "crate::humanfloat::f64_or_tag")]
    pub seasonal_half_life: f64,
    /// The trend's damping as a half-life in clock units, `φ = 0.5^(1/h)`
    /// a clock unit; `inf` does not damp.
    #[serde(default = "infinite", with = "crate::humanfloat::f64_or_tag")]
    pub damping_half_life: f64,
    /// Where the fit starts: the weighted means' warm start, initial states
    /// given, or statsmodels' heuristic on the first seasons.
    #[serde(default)]
    pub start: HoltStart,
    /// Under the known start, per target: the level, the trend (empty
    /// without one) and the `m` seasonal values, phase 0 first (empty
    /// without a season).
    #[serde(default)]
    pub initial_level: Vec<f64>,
    #[serde(default)]
    pub initial_trend: Vec<f64>,
    #[serde(default)]
    pub initial_seasonal: Vec<Vec<f64>>,
    /// The bytes the state may hold; `None` is [`HOLT_MAX_STATE_BYTES`].
    #[serde(default)]
    pub max_state_bytes: Option<u64>,
    /// The forecasts' horizons, each a span of the clock `Δ ≥ 0` past the
    /// row read, in its units: the forecast `l + b·(s + Δ)` and its integral
    /// over the span (see the module docs). Empty reports none; skipped
    /// then, so a configuration of before it is written as it was, and,
    /// being the last field, a state written as an array keeps its places.
    #[serde(
        default,
        rename = "forecast_spans",
        skip_serializing_if = "Vec::is_empty"
    )]
    pub horizons: Vec<f64>,
}

/// The most horizons a `holt` spec may ask for.
pub const HOLT_MAX_HORIZONS: usize = 64;

fn default_true() -> bool {
    true
}

fn infinite() -> f64 {
    f64::INFINITY
}

/// One target, no decay, no `min_weight`, the trend on, and none of the
/// fixed forms: the base a literal names the fields it sets on.
impl Default for HoltCfg {
    fn default() -> Self {
        Self {
            n_targets: 1,
            level_half_life: f64::INFINITY,
            trend_half_life: f64::INFINITY,
            min_weight: 0.0,
            trend: true,
            season: HoltSeason::None,
            season_periods: 0,
            period: 0.0,
            seasonal_half_life: f64::INFINITY,
            damping_half_life: f64::INFINITY,
            start: HoltStart::Warm,
            initial_level: Vec::new(),
            initial_trend: Vec::new(),
            initial_seasonal: Vec::new(),
            max_state_bytes: None,
            horizons: Vec::new(),
        }
    }
}

/// The bytes a `holt` state may hold by default: 256 KB, a group's.
pub const HOLT_MAX_STATE_BYTES: u64 = 256 * 1024;
/// The most `max_state_bytes` may be set to: 64 MiB.
pub const HOLT_STATE_BYTES_CEILING: u64 = 64 * 1024 * 1024;

/// The seasonal component of a `holt` fit.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum HoltSeason {
    #[default]
    None,
    /// `ŷ = L + S[p]`.
    Add,
    /// `ŷ = L·S[p]`: every component must stay above 0.
    Mul,
}

/// Where a `holt` fit starts.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum HoltStart {
    /// The weighted means from no weight: the first observation is the
    /// level, and the gains fall to the fixed ones as the weight gathers.
    #[default]
    Warm,
    /// Initial states given, and every gain the fixed one from row 1.
    Known,
    /// statsmodels' heuristic on the first rows, then those rows replayed.
    Heuristic,
}

impl HoltCfg {
    pub fn validate(&self) -> Result<(), String> {
        let mut no = crate::Refusals::new();
        if self.n_targets == 0 {
            no.refuse(
                "holt: the model needs at least one target, but it has none. Name at least one \
                 column in targets.",
            );
        }
        if self.level_half_life <= 0.0 || self.level_half_life.is_nan() {
            no.refuse(format!(
                "holt: half_life, the level's half-life, must be above 0, but it is {}. Set \
                 half_life to a positive number of clock units, or to inf for no decay.",
                self.level_half_life
            ));
        }
        if self.trend_half_life <= 0.0 || self.trend_half_life.is_nan() {
            no.refuse(format!(
                "holt: trend_half_life must be above 0, but it is {}. Set trend_half_life to a \
                 positive number of clock units, or to inf for a slope over the whole history.",
                self.trend_half_life
            ));
        }
        // A NaN `min_weight` is never reached, and a negative one is no
        // floor; `inf` never predicts, which is legal (review round 5, C6).
        no.take(crate::model::check_min_weight("holt", self.min_weight));
        if self.horizons.len() > HOLT_MAX_HORIZONS {
            no.refuse(format!(
                "holt: forecast_spans may hold at most {HOLT_MAX_HORIZONS} spans, but it holds {}. Each \
                 adds two fields a target to every row. Ask for {HOLT_MAX_HORIZONS} or fewer.",
                self.horizons.len()
            ));
        }
        if let Some(h) = self
            .horizons
            .iter()
            .find(|h| !(h.is_finite() && **h >= 0.0))
        {
            no.refuse(format!(
                "holt: each of forecast_spans must be a finite span of the clock of 0 or more, but one \
                 is {h}. A horizon is the clock past the row read: 0 is the row's own forecast. \
                 Give a span of 0 or more."
            ));
        }
        for (i, h) in self.horizons.iter().enumerate() {
            if self.horizons[..i].contains(h) {
                no.refuse(format!(
                    "holt: forecast_spans holds {h} twice. Each horizon's fields are named by it, so it \
                     may be asked for once. Leave out the second."
                ));
                break;
            }
        }
        if !self.horizons.is_empty() && self.fixed() {
            no.refuse(
                "holt: forecast_spans are forecasts of the plain level and trend, and a season, a \
                 damped trend or a given or heuristic start has no such forecast to integrate. \
                 Leave out forecast_spans, or leave out the season, the damping and the start."
                    .to_owned(),
            );
        }
        self.validate_forms(&mut no);
        no.finish()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Holt {
    cfg: HoltCfg,
    level: Vec<f64>,
    trend: Vec<f64>,
    seen: Vec<bool>,
    w_sum: f64,
    /// Per target, the clock since its last observation, which a row it could
    /// not learn from adds to and the next observed row extrapolates over
    /// (see the module docs).
    since: Vec<f64>,
    /// Per target, the weight the level has gathered, decayed on the level's
    /// half-life over the clock since the target was last observed: `W` in
    /// the module docs.
    #[serde(default)]
    w_level: Vec<f64>,
    /// Per target, the weight the trend has gathered, on the trend's
    /// half-life: `V` in the module docs.
    #[serde(default)]
    w_trend: Vec<f64>,
    /// Per target, the line fit to every observation so far, undecayed
    /// (docs/PLAN.md task 278, D-H): its weight `S0`, its weighted mean age
    /// `ā` on the clock, and the spread of the ages about it `Q =
    /// Σ w·(age − ā)²`. They give the exact weighted least-squares gains,
    /// which a row takes where they are the larger (the module docs). Empty
    /// in a state saved before them (schema 63 and earlier), which goes on
    /// with the textbook gains alone, as it was fitted.
    #[serde(default)]
    fit_weight: Vec<f64>,
    #[serde(default)]
    fit_age: Vec<f64>,
    #[serde(default)]
    fit_spread: Vec<f64>,
    // The fixed forms (task 488), appended: empty or 0 in a state saved
    // before them, which has no season, no damping and the warm start.
    /// Per target, the `m` seasonal values, phase 0 first.
    #[serde(default)]
    season: Vec<Vec<f64>>,
    /// Per target and phase, the decayed clock its value was last learned
    /// at; NaN for one not learned since the start.
    #[serde(default)]
    season_at: Vec<Vec<f64>>,
    /// The decayed clock, the sum of every step's `d_clock`: what a phase's
    /// clock since it was last learned is read on.
    #[serde(default)]
    clock: f64,
    /// The phase origin `c₀`: the raw clock of the first row of positive
    /// weight, which is phase 0. `None` before it.
    #[serde(default)]
    origin: Option<crate::ClockValue>,
    /// Per target, the rows the heuristic start holds until it has the
    /// seasons its definition reads; empty once the start is fitted.
    #[serde(default)]
    held: Vec<ets::Held>,
    /// The raw clock of the row the next step or prediction is for, handed
    /// by [`OnlineModel::clock_next`]; `None` reads the model's own clock.
    #[serde(skip)]
    next_clock: Option<crate::ClockValue>,
    /// Whether the last step left a target unlearned because a
    /// multiplicative component would not stay above 0.
    #[serde(skip)]
    not_positive: bool,
}

impl Holt {
    pub fn new(cfg: HoltCfg) -> Result<Self, String> {
        cfg.validate()?;
        let m = cfg.n_targets;
        Ok(Self {
            level: vec![0.0; m],
            trend: vec![0.0; m],
            seen: vec![false; m],
            w_sum: 0.0,
            since: vec![0.0; m],
            w_level: vec![0.0; m],
            w_trend: vec![0.0; m],
            fit_weight: vec![0.0; m],
            fit_age: vec![0.0; m],
            fit_spread: vec![0.0; m],
            season: Vec::new(),
            season_at: Vec::new(),
            clock: 0.0,
            origin: None,
            held: Vec::new(),
            next_clock: None,
            not_positive: false,
            cfg,
        }
        .started())
    }

    /// Whether the state keeps the line fit: every state made by this
    /// build, and none saved before it (`fit_weight`).
    fn has_line_fit(&self) -> bool {
        !self.fit_weight.is_empty()
    }

    /// Fold target `j`'s observation at weight `w`, `s` clock units after
    /// the last, into its line fit, and return the fit's gains `(g, h)`:
    /// on the level, and on the slope per unit of the forecast error. In
    /// the coordinates (level now, slope) the fit's information is
    /// `[[S0, −S0·ā], [−S0·ā, Q + S0·ā²]]`, so the gain `w·A⁻¹·(1, 0)ᵀ` is
    /// `g = w·(1/S0 + ā²/Q)` and `h = w·ā/Q`, read after the row. With every
    /// observation at one clock (`Q = 0`) the slope is not determined: the
    /// level is their weighted mean and the slope does not move.
    fn line_fit_gains(&mut self, j: usize, s: f64, w: f64) -> (f64, f64) {
        let s0 = self.fit_weight[j];
        let aged = self.fit_age[j] + s;
        let s0_new = s0 + w;
        // Each denominator is divided into once and its reciprocal
        // multiplied by after: these values feed the next row's, so every
        // division here sits on the chain one row waits on the last for,
        // and a division takes several times a multiplication's latency.
        // A row that learns has a normal weight (`step_into` reads a
        // subnormal one as 0), so `s0_new` is normal and `r0` finite. `rs`
        // is infinite where the spread is 0, every observation at one
        // clock, and where it is below `2⁻¹⁰²⁴`, a spread the doubles keep
        // only a few bits of, reached by small weights a small clock step
        // apart: there the slope is read as not determined, as at 0. Its
        // gain `w·ā·∞` made the trend NaN for good (review fable2, C7).
        let r0 = 1.0 / s0_new;
        let age = s0 * aged * r0;
        // The spread's term `S0·w·ā²/S0'`. Where `S0·w·ā²` is not a normal
        // double -- weights near `2^{-512}`, or near `2^{512}` with a long
        // clock step -- the term is formed from `S0/S0'`, in [0, 1], first:
        // `S0·w` underflowed to 0 and lost the slope's gain, or overflowed
        // and left it 0 for good, where the term itself is in range (review
        // 2026-11-23, REG-5). A term formed in range keeps its bits.
        let swa = s0 * w * aged * aged;
        let term = swa * r0;
        let term = if swa.is_normal() && term.is_normal() {
            term
        } else {
            s0 * r0 * w * aged * aged
        };
        let spread = self.fit_spread[j] + term;
        self.fit_weight[j] = s0_new;
        self.fit_age[j] = age;
        self.fit_spread[j] = spread;
        let rs = 1.0 / spread;
        if rs.is_finite() {
            (w * (r0 + age * age * rs), w * age * rs)
        } else {
            (w * r0, 0.0)
        }
    }

    /// `(λ_l, λ_b)`, the level's and the trend's decay over `s` clock units
    /// since the last observation; an infinite half-life forgets nothing.
    fn decays(&self, s: f64) -> (f64, f64) {
        let f = |h: f64| {
            if h.is_infinite() {
                1.0
            } else {
                (-(s / h)).exp2()
            }
        };
        (f(self.cfg.level_half_life), f(self.cfg.trend_half_life))
    }

    pub fn cfg(&self) -> &HoltCfg {
        &self.cfg
    }

    /// Target `j`'s forecasts at the horizons, from its level and trend as
    /// they stand and the clock `s` since its last observation, this row's
    /// included, into `out` (two slots a horizon): the forecast `Δ` past the
    /// row, `l + b·(s + Δ)`, and its integral over the span, `Δ·(l + b·(s +
    /// Δ/2))`. At `Δ = 0` the first is the row's own prediction, to the bit.
    fn forecasts_into(&self, j: usize, s: f64, out: &mut [f64]) {
        let (l, b) = (self.level[j], self.trend[j]);
        for (k, &dt) in self.cfg.horizons.iter().enumerate() {
            out[2 * k] = l + b * (s + dt);
            out[2 * k + 1] = dt * (l + b * (s + 0.5 * dt));
        }
    }

    pub fn level(&self) -> &[f64] {
        &self.level
    }

    pub fn trend(&self) -> &[f64] {
        &self.trend
    }

    pub fn n_eff(&self) -> f64 {
        self.w_sum
    }

    /// Reported as `coef`: `[level, trend]` per target, which is the whole
    /// state and the only thing worth inspecting. NaN, which the stream
    /// writes as null, for a target not yet observed: it has no level to
    /// report, and the `[0, 0]` it reported was a level no row had given
    /// (docs/OUTPUTS.md; PLAN task 97).
    pub fn coefficients(&self) -> Vec<Vec<f64>> {
        (0..self.cfg.n_targets)
            .map(|j| {
                if self.seen[j] {
                    vec![self.level[j], self.trend[j]]
                } else {
                    vec![f64::NAN, f64::NAN]
                }
            })
            .collect()
    }
}

impl OnlineModel for Holt {
    /// Each target's weight is its level's, decayed over the clock since the
    /// target was last observed: the rows it was present on (S2).
    fn target_n_eff_into(&self, out: &mut Vec<f64>) -> bool {
        out.clear();
        out.extend(
            self.w_level
                .iter()
                .zip(&self.since)
                .map(|(&w, &s)| w * self.decays(s).0),
        );
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
        // A target or a weight that is not usable, by the rule every model
        // keeps (`OnlineModel`); Holt reads no features.
        if let Some(refused) = crate::model::refused_step(self, &[], y, d_clock, weight) {
            *out = refused;
            return;
        }
        let m = self.cfg.n_targets;
        // A weight below the smallest normal double is a weight of 0: the
        // row advances the clock and learns nothing (hard rule 9). Each
        // gain is a weight times a reciprocal kept to multiply by, and the
        // reciprocal of a sum that small is infinite: a first row of weight
        // `1e-310` left the line fit's `1/Q` infinite and the level and trend
        // NaN for good (review fable2, C7). What such a row would teach is
        // below a double's resolution beside any normal weight.
        let weight = if weight < f64::MIN_POSITIVE {
            0.0
        } else {
            weight
        };
        if self.cfg.fixed() {
            let pred = &mut out.pred;
            pred.clear();
            pred.resize(m, f64::NAN);
            let n_eff = self.w_sum;
            self.fixed_step(y, d_clock, weight, pred);
            out.n_eff = n_eff;
            out.extra = None;
            return;
        }
        // The decays for a target observed on the previous row, where `s` is
        // this row's delta: every target of a stream with no gaps.
        let row_decays = self.decays(d_clock);

        let n_eff = self.w_sum;
        let ready = n_eff >= self.cfg.min_weight;
        let per = 2 * self.cfg.horizons.len();
        out.pred.clear();
        out.pred.resize(m * (1 + per), f64::NAN);
        for (j, yv) in y.iter().enumerate().take(m) {
            let yj = yv.filter(|v| v.is_finite() && weight > 0.0);
            if !self.seen[j] {
                // Nothing to extrapolate from yet: the first observation is
                // the level, at its own weight.
                if let Some(yj) = yj {
                    self.level[j] = yj;
                    self.w_level[j] = weight;
                    self.seen[j] = true;
                    if self.has_line_fit() {
                        self.fit_weight[j] = weight;
                    }
                }
                continue;
            }
            // Extrapolate over the clock since the target was last observed,
            // so an irregular clock, or a gap in the target, forecasts the
            // right distance ahead.
            let s = self.since[j] + d_clock;
            let p = self.level[j] + self.trend[j] * s;
            if ready {
                out.pred[j] = p;
                if per > 0 {
                    let at = m + j * per;
                    self.forecasts_into(j, s, &mut out.pred[at..at + per]);
                }
            }
            let Some(yj) = yj else {
                // Nothing to learn from: carry the clock to the next
                // observation (C22).
                self.since[j] = s;
                continue;
            };
            let (lam_l, lam_b) = if self.since[j] == 0.0 {
                row_decays
            } else {
                self.decays(s)
            };
            // The line fit's gains, where the state keeps one and fits a
            // trend; without a trend the level's weighted mean below is
            // already the larger gain (`λ·W ≤ S0`).
            let line =
                (self.cfg.trend && self.has_line_fit()).then(|| self.line_fit_gains(j, s, weight));
            // The textbook gain: the level as a weighted mean of what the
            // row observes and what the state held; `weight > 0` here, so
            // no denominator is 0 (hard rule 9). An infinite half-life has
            // no textbook gain to give: its fixed gain `1 − λ` is 0, and the
            // line fit is the whole history's.
            // `rh` is the reciprocal of the weighted mean's denominator,
            // kept to multiply by, as in `line_fit_gains`: one division on
            // the row-to-row chain in place of two.
            let held = lam_l * self.w_level[j];
            let rh = 1.0 / (held + weight);
            let gain = weight * rh;
            let prev_level = self.level[j];
            let error = yj - p;
            let level_gain = match line {
                Some((g, _)) if g > gain || self.cfg.level_half_life.is_infinite() => {
                    self.level[j] = p + g * error;
                    g
                }
                _ => {
                    let mean = (held * p + weight * yj) * rh;
                    // `held·p` or `w·y` past the largest double, where the
                    // mean is in range: the mean of shares, `held/(held +
                    // w)` and `w/(held + w)` each in [0, 1] (REG-4).
                    self.level[j] = if mean.is_finite() {
                        mean
                    } else {
                        (held * rh) * p + gain * yj
                    };
                    gain
                }
            };
            self.w_level[j] = held + weight;
            if self.cfg.trend {
                // The textbook trend: a weighted mean of the slopes the
                // level moved at, whose gain on the error is `w/(λ_b·V + w)
                // · g/s`. A move over no clock has no slope, and an infinite
                // trend half-life none to give where the line fit is kept:
                // the line fit's slope is then the whole history's drift.
                // The reciprocals of the trend's denominator and of the
                // clock step, each divided into once. `ri = 1/s` is infinite
                // at `s = 0`, and at a step below `2⁻¹⁰²⁴`, which is read as
                // no clock (review fable2, C7): `textbook` is false there,
                // so neither the comparison nor `by_textbook` below reads it.
                let ri = 1.0 / s;
                let clocked = ri.is_finite();
                let textbook = clocked && (line.is_none() || self.cfg.trend_half_life.is_finite());
                let held_b = lam_b * self.w_trend[j];
                let rb = 1.0 / (held_b + weight);
                let trend = self.trend[j];
                let level = self.level[j];
                let by_textbook = || {
                    let slope = (level - prev_level) * ri;
                    (held_b * trend + weight * slope) * rb
                };
                let next = match (line, textbook) {
                    // Both gains are on offer: both trends are formed and
                    // the larger gain's is kept by a select, not a branch.
                    // Which gain is the larger turns on the row's clock
                    // step and on how young the line fit is, so while the
                    // fit is young it changes from row to row about as
                    // often as a coin flip, and a branch on it is
                    // mispredicted about half the time. Each miss stalls
                    // the chain the next row waits on; the select costs
                    // the textbook arm's few operations on every row
                    // instead, which overlap the rest of the row.
                    (Some((_, h)), true) => std::hint::select_unpredictable(
                        h > weight * rb * level_gain * ri,
                        trend + h * error,
                        by_textbook(),
                    ),
                    (Some((_, h)), false) => trend + h * error,
                    (None, true) => by_textbook(),
                    (None, false) => trend,
                };
                // A trend past the largest double: `w·slope` alone past it,
                // which the mean of shares forms in range; or the slope
                // itself, a move over a clock step so small that it is not
                // a double (`y = 0, 10` a step of `1e-308` apart), read as
                // a step of no clock, as one whose reciprocal is infinite
                // is, so the trend holds. An infinite trend made the level
                // NaN on the next row, for good (review 2026-11-23, REG-4).
                // A finite trend takes neither branch: its bits are as
                // before.
                let (next, clocked) = if next.is_finite() {
                    (next, clocked)
                } else {
                    let by_textbook = match (line, textbook) {
                        (Some((_, h)), true) => {
                            h.partial_cmp(&(weight * rb * level_gain * ri))
                                != Some(std::cmp::Ordering::Greater)
                        }
                        (None, true) => true,
                        _ => false,
                    };
                    let slope = (level - prev_level) * ri;
                    let shares = (held_b * rb) * trend + (weight * rb) * slope;
                    if by_textbook && shares.is_finite() {
                        (shares, clocked)
                    } else {
                        (trend, false)
                    }
                };
                self.trend[j] = next;
                if clocked {
                    self.w_trend[j] = held_b + weight;
                }
            }
            self.since[j] = 0.0;
        }
        self.w_sum = self.w_sum * row_decays.0 + weight;
        out.n_eff = n_eff;
        out.extra = None;
    }

    /// The prediction at another `min_weight` floor, set for the read and
    /// put back ([`OnlineModel::predict_with_floor`]).
    fn predict_with_floor(
        &mut self,
        x: &[f64],
        y: &[Option<f64>],
        d_clock: f64,
        floor: f64,
    ) -> Step {
        let floor = std::mem::replace(&mut self.cfg.min_weight, floor);
        let out = OnlineModel::predict_with(self, x, y, d_clock);
        self.cfg.min_weight = floor;
        out
    }

    fn predict(&self, _x: &[f64], d_clock: f64) -> Step {
        let n_eff = self.w_sum;
        let m = self.cfg.n_targets;
        let per = 2 * self.cfg.horizons.len();
        let mut pred = vec![f64::NAN; m * (1 + per)];
        if n_eff >= self.cfg.min_weight && self.cfg.fixed() {
            self.fixed_predict(d_clock, None, &mut pred);
        } else if n_eff >= self.cfg.min_weight {
            for j in 0..m {
                if self.seen[j] {
                    let s = self.since[j] + d_clock;
                    pred[j] = self.level[j] + self.trend[j] * s;
                    if per > 0 {
                        let at = m + j * per;
                        self.forecasts_into(j, s, &mut pred[at..at + per]);
                    }
                }
            }
        }
        Step {
            pred,
            n_eff,
            extra: None,
        }
    }

    /// The row's raw clock, which a season's phase is read from.
    fn clock_next(&mut self, clock: crate::ClockValue) {
        if self.cfg.season != HoltSeason::None {
            self.next_clock = Some(clock);
        }
    }

    fn predict_at_into(
        &self,
        x: &[f64],
        y: &[Option<f64>],
        d_clock: f64,
        clock: Option<crate::ClockValue>,
        out: &mut Step,
    ) {
        if !(self.cfg.fixed() && clock.is_some()) {
            self.predict_with_into(x, y, d_clock, out);
            return;
        }
        out.pred.clear();
        out.pred.resize(self.cfg.n_targets, f64::NAN);
        out.n_eff = self.w_sum;
        out.extra = None;
        if self.w_sum >= self.cfg.min_weight {
            self.fixed_predict(d_clock, clock, &mut out.pred);
        }
    }

    fn state(&self) -> State {
        State::new(ModelState::Holt(Box::new(self.clone())))
    }

    fn restore(s: &State) -> Result<Self, StateError> {
        check_schema(s)?;
        match &s.model {
            ModelState::Holt(m) => {
                let m = (**m).clone();
                crate::model::check_cfg("holt", m.cfg.validate())?;
                let n = m.cfg.n_targets;
                // Every vector is checked, not repaired: a level or a weight
                // vector of the wrong length loaded and panicked on the
                // first `step` (review 2026-09-18, B3), and a clock since an
                // observation of the wrong length was set to zero, the
                // repair for a schema-6 state (docs/PLAN.md task 198).
                // The line fit is all three vectors or none: none is a state
                // saved before it, which goes on with the textbook gains.
                let fit = [&m.fit_weight, &m.fit_age, &m.fit_spread];
                let fit_len = m.fit_weight.len();
                if m.seen.len() != n
                    || [&m.level, &m.trend, &m.w_level, &m.w_trend, &m.since]
                        .iter()
                        .any(|v| v.len() != n)
                    || (fit_len != 0 && fit_len != n)
                    || fit.iter().any(|v| v.len() != fit_len)
                    || !m.fixed_shape_ok()
                {
                    return Err(crate::model::damaged(
                        "holt",
                        "its saved values have the wrong shape",
                    ));
                }
                Ok(m)
            }
            other => Err(StateError::WrongModel {
                expected: "holt",
                found: other.kind(),
            }),
        }
    }

    fn n_targets(&self) -> usize {
        self.cfg.n_targets
    }

    /// Holt uses no features.
    fn n_features(&self) -> usize {
        0
    }

    /// A prediction a target, then each target's forecasts at the horizons,
    /// two a horizon.
    fn n_outputs(&self) -> usize {
        self.cfg.n_targets * (1 + 2 * self.cfg.horizons.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A state whose vectors are not the cfg's is refused, where it loaded
    /// and panicked on the first `step` (review 2026-09-18, B3).
    #[test]
    fn a_state_of_the_wrong_shape_is_refused() {
        use crate::{ModelState, OnlineModel, StateError};
        let m = Holt::new(cfg(10.0, 20.0)).unwrap();
        let mut s = m.state();
        let ModelState::Holt(inner) = &mut s.model else {
            unreachable!()
        };
        inner.level.pop();
        match Holt::restore(&s) {
            Err(StateError::Invalid(e)) => assert!(e.contains("wrong shape"), "{e}"),
            other => panic!("{other:?}"),
        }
    }

    fn lcg(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }

    fn cfg(level: f64, trend: f64) -> HoltCfg {
        HoltCfg {
            horizons: Vec::new(),
            n_targets: 1,
            level_half_life: level,
            trend_half_life: trend,
            min_weight: 0.0,
            trend: true,
            ..Default::default()
        }
    }

    /// PLAN task 97. Before a target's first observation there is no level
    /// and no trend to report, and `coef` is null there, as docs/OUTPUTS.md
    /// says it is "before the model has anything to report" (NaN here, which
    /// the stream writes as null). It was `[0, 0]`, a level of zero that no
    /// row had given. A null value or a row of weight 0 observes nothing,
    /// and the first observation reports its own value as the level.
    #[test]
    fn a_target_not_yet_observed_reports_no_coefficients() {
        let mut m = Holt::new(HoltCfg {
            n_targets: 2,
            ..cfg(10.0, 10.0)
        })
        .unwrap();
        let unseen = |c: &[f64]| c.iter().all(|v| v.is_nan());
        let coef = m.coefficients();
        assert!(
            unseen(&coef[0]) && unseen(&coef[1]),
            "nothing yet: {coef:?}"
        );
        // Target 0 observed; target 1 null, then observed at weight 0.
        m.step(&[], &[Some(4.0), None], 0.0, 1.0);
        m.step(&[], &[Some(5.0), Some(9.0)], 1.0, 0.0);
        let coef = m.coefficients();
        assert_eq!(coef[0][0], 4.0, "the first observation is the level");
        assert_eq!(coef[0][1], 0.0, "and no slope yet: one point has none");
        assert!(
            unseen(&coef[1]),
            "a null and a weight of 0 observe nothing: {coef:?}"
        );
        m.step(&[], &[Some(6.0), Some(-2.0)], 1.0, 1.0);
        let coef = m.coefficients();
        assert_eq!(coef[1], vec![-2.0, 0.0], "its first observation");
        assert!(coef[0].iter().all(|v| v.is_finite()));
    }

    fn run(cfg: HoltCfg, ys: &[f64], d: f64) -> (Vec<f64>, Holt) {
        let mut m = Holt::new(cfg).unwrap();
        let mut preds = Vec::new();
        for (i, y) in ys.iter().enumerate() {
            let step = m.step(&[], &[Some(*y)], if i == 0 { 0.0 } else { d }, 1.0);
            preds.push(step.pred[0]);
        }
        (preds, m)
    }

    /// The textbook recursion alone, written out longhand: level and trend
    /// are each a weighted mean of what the row observes and what the state
    /// held, `(λ·W·old + w·new)/(λ·W + w)`, with `W` the weight each has
    /// gathered and `λ` its decay over the clock since the target was last
    /// observed (review 2026-09-12, S29/S30). The whole model until the line
    /// fit (task 278, D-H), and still a state's saved before it.
    fn textbook(
        cfg: &HoltCfg,
        ys: &[Option<f64>],
        ds: &[f64],
        ws: &[f64],
    ) -> (Vec<f64>, f64, f64, f64) {
        let decay = |s: f64, h: f64| {
            if h.is_infinite() {
                1.0
            } else {
                0.5f64.powf(s / h)
            }
        };
        let (mut level, mut trend, mut w_level, mut w_trend) = (0.0, 0.0, 0.0, 0.0);
        let (mut w_sum, mut since, mut seen) = (0.0, 0.0, false);
        let mut preds = Vec::new();
        for ((&y, &d), &w) in ys.iter().zip(ds).zip(ws) {
            let y = y.filter(|v| v.is_finite() && w > 0.0);
            if !seen {
                preds.push(f64::NAN);
                if let Some(y) = y {
                    (level, w_level, seen) = (y, w, true);
                }
            } else {
                let s = since + d;
                let p = level + trend * s;
                preds.push(if w_sum >= cfg.min_weight { p } else { f64::NAN });
                if let Some(y) = y {
                    let held = decay(s, cfg.level_half_life) * w_level;
                    let prev = level;
                    level = (held * p + w * y) / (held + w);
                    w_level = held + w;
                    if s > 0.0 {
                        let held = decay(s, cfg.trend_half_life) * w_trend;
                        trend = (held * trend + w * (level - prev) / s) / (held + w);
                        w_trend = held + w;
                    }
                    since = 0.0;
                } else {
                    since = s;
                }
            }
            w_sum = w_sum * decay(d, cfg.level_half_life) + w;
        }
        (preds, level, trend, w_sum)
    }

    /// The model written out from its definitions, against an irregular
    /// clock, so the test cannot share a mistake with the implementation:
    /// the line fit's weight, mean age and spread summed afresh over every
    /// observation so far (`crate::oracle::sum`), its gains `g = w·(1/S0 +
    /// ā²/Q)` and `h = w·ā/Q`, and the textbook's weighted means; a row
    /// takes the larger gain, on the level and on the slope, and an infinite
    /// half-life has no textbook gain (task 278, D-H).
    fn reference(
        cfg: &HoltCfg,
        ys: &[Option<f64>],
        ds: &[f64],
        ws: &[f64],
    ) -> (Vec<f64>, f64, f64, f64) {
        use crate::oracle::sum;
        let decay = |s: f64, h: f64| {
            if h.is_infinite() {
                1.0
            } else {
                0.5f64.powf(s / h)
            }
        };
        let (mut level, mut trend, mut w_level, mut w_trend) = (0.0, 0.0, 0.0, 0.0);
        let (mut w_sum, mut since, mut seen) = (0.0, 0.0, false);
        let (mut clock, mut obs) = (0.0, Vec::<(f64, f64)>::new());
        let mut preds = Vec::new();
        for ((&y, &d), &w) in ys.iter().zip(ds).zip(ws) {
            clock += d;
            let y = y.filter(|v| v.is_finite() && w > 0.0);
            if !seen {
                preds.push(f64::NAN);
                if let Some(y) = y {
                    (level, w_level, seen) = (y, w, true);
                    obs.push((clock, w));
                }
            } else {
                let s = since + d;
                let p = level + trend * s;
                preds.push(if w_sum >= cfg.min_weight { p } else { f64::NAN });
                if let Some(y) = y {
                    obs.push((clock, w));
                    let s0 = sum(obs.iter().map(|o| o.1));
                    let age = sum(obs.iter().map(|o| o.1 * (clock - o.0))) / s0;
                    let q = sum(obs.iter().map(|o| o.1 * (clock - o.0 - age).powi(2)));
                    let (g_fit, h_fit) = if q > 0.0 {
                        (w / s0 + w * age * age / q, w * age / q)
                    } else {
                        (w / s0, 0.0)
                    };
                    let e = y - p;
                    let held = decay(s, cfg.level_half_life) * w_level;
                    let g_book = if cfg.level_half_life.is_infinite() {
                        0.0
                    } else {
                        w / (held + w)
                    };
                    let g = g_fit.max(g_book);
                    level = p + g * e;
                    w_level = held + w;
                    if s > 0.0 {
                        let held = decay(s, cfg.trend_half_life) * w_trend;
                        let h_book = if cfg.trend_half_life.is_infinite() {
                            0.0
                        } else {
                            w / (held + w) * g / s
                        };
                        trend += h_fit.max(h_book) * e;
                        w_trend = held + w;
                    } else {
                        trend += h_fit * e;
                    }
                    since = 0.0;
                } else {
                    since = s;
                }
            }
            w_sum = w_sum * decay(d, cfg.level_half_life) + w;
        }
        (preds, level, trend, w_sum)
    }

    /// A model whose state keeps no line fit: a state saved before it.
    fn without_line_fit(c: HoltCfg) -> Holt {
        let mut m = Holt::new(c).unwrap();
        m.fit_weight.clear();
        m.fit_age.clear();
        m.fit_spread.clear();
        Holt::restore(&m.state()).expect("a state without the line fit loads")
    }

    #[test]
    fn every_step_matches_the_recursion_written_out() {
        // Pins each arithmetic step -- the extrapolation distance, both
        // weighted means, the per-clock-unit slope and the decayed weight --
        // on a clock whose gaps vary, rows whose weights vary (one of them
        // 0), and a null, so no factor can cancel out.
        let ds = [0.0, 1.0, 0.25, 7.0, 1.0, 1.0, 0.5, 13.0, 2.0, 1.0, 1.0, 3.0];
        let ws = [1.0, 0.5, 2.0, 1.0, 0.25, 1.0, 3.0, 1.0, 0.0, 1.0, 1.5, 1.0];
        let mut s = 5u64;
        let mut ys: Vec<Option<f64>> = (0..ds.len())
            .map(|i| Some(3.0 + 0.8 * i as f64 + lcg(&mut s)))
            .collect();
        ys[5] = None;
        let c = HoltCfg {
            horizons: Vec::new(),
            n_targets: 1,
            level_half_life: 4.0,
            trend_half_life: 9.0,
            min_weight: 2.5,
            trend: true,
            ..Default::default()
        };
        let (want_pred, want_level, want_trend, want_w) = reference(&c, &ys, &ds, &ws);

        let mut m = Holt::new(c).unwrap();
        for (i, ((&y, &d), &w)) in ys.iter().zip(&ds).zip(&ws).enumerate() {
            let step = m.step(&[], &[y], d, w);
            match (step.pred[0].is_nan(), want_pred[i].is_nan()) {
                (true, true) => {}
                (false, false) => assert!(
                    (step.pred[0] - want_pred[i]).abs() < 1e-12,
                    "row {i}: {} vs {}",
                    step.pred[0],
                    want_pred[i]
                ),
                _ => panic!("row {i}: {} vs {}", step.pred[0], want_pred[i]),
            }
        }
        assert!((m.level()[0] - want_level).abs() < 1e-12);
        assert!((m.trend()[0] - want_trend).abs() < 1e-12);
        assert!((m.n_eff() - want_w).abs() < 1e-12);
    }

    #[test]
    fn n_eff_decays_on_the_clock_and_gates_output() {
        let c = HoltCfg {
            horizons: Vec::new(),
            n_targets: 1,
            level_half_life: 10.0,
            trend_half_life: 40.0,
            min_weight: 3.0,
            trend: true,
            ..Default::default()
        };
        let mut m = Holt::new(c).unwrap();
        assert_eq!(m.n_eff(), 0.0, "nothing seen yet");

        // n_eff is the weight before the row, so it lags the row count by one.
        let mut want = 0.0;
        for i in 0..6 {
            let d = if i == 0 { 0.0 } else { 1.0 };
            let step = m.step(&[], &[Some(i as f64)], d, 1.0);
            assert!((step.n_eff - want).abs() < 1e-12, "row {i}");
            assert_eq!(
                step.pred[0].is_nan(),
                want < 3.0,
                "row {i}: gated at n_eff = {want}"
            );
            want = want * 0.5f64.powf(d / 10.0) + 1.0;
        }

        // A long gap decays it rather than resetting it.
        let before = m.n_eff();
        m.step(&[], &[Some(99.0)], 100.0, 1.0);
        let after = m.n_eff();
        assert!(
            (after - (before * 0.5f64.powf(10.0) + 1.0)).abs() < 1e-12,
            "{before} -> {after}"
        );
    }

    #[test]
    fn the_level_is_seeded_only_by_a_usable_first_row() {
        // Two independent reasons a row cannot seed the level: the target is
        // not a number, or the row carries no weight. Either alone must leave
        // the model unseeded, still predicting nothing.
        for (y, w) in [
            (Some(f64::NAN), 1.0),
            (Some(f64::INFINITY), 1.0),
            (None, 1.0),
            (Some(5.0), 0.0),
            (Some(5.0), -1.0),
        ] {
            let mut m = Holt::new(cfg(10.0, 40.0)).unwrap();
            let step = m.step(&[], &[y], 0.0, w);
            assert!(step.pred[0].is_nan(), "({y:?}, {w}) must not predict");
            assert!(!m.seen[0], "({y:?}, {w}) must not seed the level");
            assert_eq!(m.level()[0], 0.0);

            // A usable row afterwards still seeds it, at its own value.
            m.step(&[], &[Some(9.0)], 1.0, 1.0);
            assert!(m.seen[0]);
            assert_eq!(m.level()[0], 9.0, "the first usable row seeds the level");
        }
    }

    /// A repeated timestamp is a second observation at the same time: the
    /// level takes it in, as a weighted mean takes any row. The textbook
    /// trend holds, since a move over no clock has no slope; the line fit's
    /// does not, since a second observation at one time still moves a
    /// least-squares slope, by the fit's own gain (task 278, D-H). The
    /// textbook form's rate was 0 at `d = 0`, so such a row once changed
    /// nothing (S29).
    #[test]
    fn a_zero_gap_folds_the_row_into_the_level_and_the_line_fits_slope() {
        let c = cfg(5.0, 20.0);
        let mut ys: Vec<Option<f64>> = (0..30).map(|i| Some(2.0 * i as f64)).collect();
        let mut ds: Vec<f64> = (0..30).map(|i| if i == 0 { 0.0 } else { 1.0 }).collect();
        let mut m = Holt::new(c.clone()).unwrap();
        let mut old = without_line_fit(c.clone());
        for (y, d) in ys.iter().zip(&ds) {
            m.step(&[], &[*y], *d, 1.0);
            old.step(&[], &[*y], *d, 1.0);
        }
        let (l, t) = (m.level()[0], m.trend()[0]);
        assert!(t > 0.5, "there should be a trend: {t}");
        let t_old = old.trend()[0];
        m.step(&[], &[Some(-1000.0)], 0.0, 1.0);
        old.step(&[], &[Some(-1000.0)], 0.0, 1.0);
        assert_eq!(
            old.trend()[0],
            t_old,
            "the textbook trend holds at a zero gap"
        );
        assert!(m.trend()[0] < t, "the line fit's slope takes the row in");
        assert!(
            m.level()[0] < l,
            "the row is taken in: {} vs {l}",
            m.level()[0]
        );
        ys.push(Some(-1000.0));
        ds.push(0.0);
        let ws = vec![1.0; ys.len()];
        let (_, want, want_t, _) = reference(&c, &ys, &ds, &ws);
        assert!(
            (m.level()[0] - want).abs() < 1e-9,
            "{} vs {want}",
            m.level()[0]
        );
        assert!(
            (m.trend()[0] - want_t).abs() < 1e-12,
            "{} vs {want_t}",
            m.trend()[0]
        );
    }

    #[test]
    fn targets_are_independent() {
        // `n_targets` and the per-target loop: two targets with unrelated
        // series must not contaminate each other, and both must be reported.
        let c = HoltCfg {
            horizons: Vec::new(),
            n_targets: 2,
            level_half_life: 8.0,
            trend_half_life: 30.0,
            min_weight: 0.0,
            trend: true,
            ..Default::default()
        };
        let mut m = Holt::new(c).unwrap();
        for i in 0..600 {
            let d = if i == 0 { 0.0 } else { 1.0 };
            let (a, b) = (10.0 + 3.0 * i as f64, 500.0 - 1.0 * i as f64);
            m.step(&[], &[Some(a), Some(b)], d, 1.0);
        }
        let coef = m.coefficients();
        assert_eq!(coef.len(), 2, "one [level, trend] pair per target");
        assert_eq!(coef[0].len(), 2);
        assert!(
            (coef[0][1] - 3.0).abs() < 0.05,
            "target 0 trend {}",
            coef[0][1]
        );
        assert!(
            (coef[1][1] + 1.0).abs() < 0.05,
            "target 1 trend {}",
            coef[1][1]
        );
        assert!(coef[0][0] > 1000.0, "levels are far apart: {}", coef[0][0]);
        assert!(
            coef[1][0] < 0.0,
            "target 1 has fallen below zero: {}",
            coef[1][0]
        );
    }

    #[test]
    fn a_null_target_predicts_without_learning() {
        // Prediction still happens; the level and slope stay where the last
        // observation put them, and the row's clock is carried to the next
        // one (`a_row_it_cannot_learn_from_is_as_if_absent`).
        let mut m = Holt::new(cfg(6.0, 25.0)).unwrap();
        for i in 0..40 {
            m.step(
                &[],
                &[Some(1.0 + 2.0 * i as f64)],
                if i == 0 { 0.0 } else { 1.0 },
                1.0,
            );
        }
        let (l, t) = (m.level()[0], m.trend()[0]);
        let step = m.step(&[], &[None], 1.0, 1.0);
        assert!((step.pred[0] - (l + t)).abs() < 1e-12, "still extrapolates");
        assert_eq!(m.level()[0], l);
        assert_eq!(m.trend()[0], t);
        assert_eq!(m.since[0], 1.0, "the clock is carried");
    }

    /// Weight 0 folds nothing in: the level and slope stay, the clock is
    /// carried as across a null, and `n_eff` decays.
    #[test]
    fn a_zero_weight_row_is_pure_decay() {
        let mut m = Holt::new(cfg(6.0, 25.0)).unwrap();
        for i in 0..40 {
            m.step(
                &[],
                &[Some(1.0 + 2.0 * i as f64)],
                if i == 0 { 0.0 } else { 1.0 },
                1.0,
            );
        }
        let (l, t, w) = (m.level()[0], m.trend()[0], m.n_eff());
        m.step(&[], &[Some(-500.0)], 1.0, 0.0);
        assert_eq!(m.level()[0], l, "weight 0 must not fold the row in");
        assert_eq!(m.trend()[0], t);
        assert_eq!(m.since[0], 1.0, "the clock is carried");
        assert!((m.n_eff() - w * 0.5f64.powf(1.0 / 6.0)).abs() < 1e-12);
    }

    #[test]
    fn tracks_a_constant_level() {
        let ys: Vec<f64> = vec![7.0; 200];
        let (preds, m) = run(cfg(10.0, 20.0), &ys, 1.0);
        assert!((preds[199] - 7.0).abs() < 1e-9);
        assert!(m.trend()[0].abs() < 1e-9, "no trend in a flat series");
    }

    #[test]
    fn extrapolates_a_linear_trend() {
        // y = 3 + 2t. A level-only model would always lag; with a trend the
        // one-step-ahead prediction should be right.
        let ys: Vec<f64> = (0..500).map(|i| 3.0 + 2.0 * i as f64).collect();
        let (preds, m) = run(cfg(5.0, 5.0), &ys, 1.0);
        assert!(
            (m.trend()[0] - 2.0).abs() < 0.05,
            "trend should be ~2, got {}",
            m.trend()[0]
        );
        assert!(
            (preds[499] - ys[499]).abs() < 0.5,
            "should predict the next value closely: {} vs {}",
            preds[499],
            ys[499]
        );
    }

    /// An infinite trend half-life forgets no slope: the trend is the mean of
    /// every one observed, which is what `inf` means for every other model
    /// here. It pinned the trend at zero once -- the textbook form's rate at
    /// an infinite half-life is 0, a trend that never learns (S30).
    #[test]
    fn an_infinite_trend_halflife_is_the_whole_history_drift() {
        let ys: Vec<f64> = (0..500).map(|i| 3.0 + 2.0 * i as f64).collect();
        let (preds, m) = run(cfg(5.0, f64::INFINITY), &ys, 1.0);
        assert!((m.trend()[0] - 2.0).abs() < 0.05, "trend {}", m.trend()[0]);
        assert!(
            (preds[499] - ys[499]).abs() < 1.0,
            "{} vs {}",
            preds[499],
            ys[499]
        );
    }

    #[test]
    fn an_irregular_clock_extrapolates_the_right_distance() {
        // Same series sampled every 5 clock units: the trend is per clock unit,
        // so the prediction must step 5 units ahead, not 1.
        let mut m = Holt::new(cfg(20.0, 20.0)).unwrap();
        let mut last = f64::NAN;
        for i in 0..400 {
            let t = i as f64 * 5.0;
            let y = 3.0 + 2.0 * t;
            let step = m.step(&[], &[Some(y)], if i == 0 { 0.0 } else { 5.0 }, 1.0);
            last = step.pred[0];
        }
        let truth = 3.0 + 2.0 * (399.0 * 5.0);
        assert!(
            (last - truth).abs() / truth < 0.01,
            "expected ~{truth}, got {last}"
        );
        assert!(
            (m.trend()[0] - 2.0).abs() < 0.1,
            "trend per clock unit: {}",
            m.trend()[0]
        );
    }

    #[test]
    fn first_value_seeds_the_level() {
        let (preds, m) = run(cfg(10.0, 10.0), &[42.0, 42.0], 1.0);
        assert!(
            preds[0].is_nan(),
            "nothing to predict from on the first row"
        );
        assert!((m.level()[0] - 42.0).abs() < 1e-9);
    }

    #[test]
    fn null_target_is_predict_only() {
        let mut m = Holt::new(cfg(10.0, 10.0)).unwrap();
        for i in 0..20 {
            m.step(&[], &[Some(i as f64)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        let before = (m.level()[0], m.trend()[0]);
        let step = m.step(&[], &[None], 1.0, 1.0);
        assert!(step.pred[0].is_finite());
        assert_eq!((m.level()[0], m.trend()[0]), before);
    }

    #[test]
    fn state_roundtrip() {
        let ys: Vec<f64> = (0..100)
            .map(|i| (i as f64 * 0.3).sin() + 0.1 * i as f64)
            .collect();
        let (_, m1) = run(cfg(10.0, 30.0), &ys[..50], 1.0);
        let bytes = rmp_serde::to_vec(&m1.state()).unwrap();
        let mut m2 = Holt::restore(&rmp_serde::from_slice(&bytes).unwrap()).unwrap();
        let mut m1 = m1;
        for y in &ys[50..] {
            assert_eq!(
                m1.step(&[], &[Some(*y)], 1.0, 1.0).pred,
                m2.step(&[], &[Some(*y)], 1.0, 1.0).pred
            );
        }
    }

    #[test]
    fn rejects_bad_config() {
        assert!(Holt::new(cfg(0.0, 10.0)).is_err());
        assert!(Holt::new(cfg(10.0, 0.0)).is_err());
    }

    /// A row the model cannot learn from -- a null target, or a present one
    /// at weight zero -- is transparent: the same numbers as if it were
    /// absent and its clock folded into the next row's, which is what
    /// `clock.rs` does with a row it skips (review 2026-09-12, C22). The
    /// textbook form's rate is not additive in the clock, `α(d₁ + d₂) ≠
    /// α(d₂)`, so advancing the level with the row's own rate would not do:
    /// each target keeps its clock since its last observation and the next
    /// observed row extrapolates, and forms its rates, over all of it.
    #[test]
    fn a_row_it_cannot_learn_from_is_as_if_absent() {
        let c = cfg(6.0, 25.0);
        let mut s = 11u64;
        let ys: Vec<f64> = (0..91)
            .map(|i| 3.0 + 2.0 * i as f64 + lcg(&mut s))
            .collect();
        for zero_weight in [false, true] {
            let mut full = Holt::new(c.clone()).unwrap();
            let mut kept = Holt::new(c.clone()).unwrap();
            let mut folded = 0.0;
            for (i, &y) in ys.iter().enumerate() {
                let d = if i == 0 { 0.0 } else { 1.0 };
                if i % 2 == 1 {
                    // The value is one no fit could absorb, so a row that
                    // leaked into the state would show.
                    let (yv, w) = if zero_weight {
                        (Some(-500.0), 0.0)
                    } else {
                        (None, 1.0)
                    };
                    full.step(&[], &[yv], d, w);
                    folded += d;
                    continue;
                }
                let a = full.step(&[], &[Some(y)], d, 1.0).pred[0];
                let b = kept.step(&[], &[Some(y)], d + folded, 1.0).pred[0];
                folded = 0.0;
                assert!(
                    (a.is_nan() && b.is_nan()) || (a - b).abs() <= 1e-12 * (1.0 + b.abs()),
                    "zero weight {zero_weight}, row {i}: {a} vs {b}"
                );
            }
            assert_eq!(full.coefficients(), kept.coefficients());
        }
    }

    /// `y = 3 + 2t` with every other target null. Before C22 the observed
    /// rows were forecast from a level that had stood still across the null,
    /// one trend step short -- about 2 -- on every one of them.
    #[test]
    fn a_trend_is_extrapolated_across_a_missing_row() {
        let mut m = Holt::new(cfg(5.0, 5.0)).unwrap();
        let mut miss = f64::NAN;
        for i in 0..500 {
            let y = 3.0 + 2.0 * i as f64;
            let yv = (i % 2 == 0).then_some(y);
            let step = m.step(&[], &[yv], if i == 0 { 0.0 } else { 1.0 }, 1.0);
            // The null row is forecast one step out, the next two.
            miss = step.pred[0] - y;
        }
        assert!(miss.abs() < 0.5, "the last forecast missed by {miss}");
    }

    /// `predict` extrapolates over the clock since the target was last
    /// observed plus its own delta, as the next `step` would.
    #[test]
    fn predict_extrapolates_over_the_clock_since_the_last_observation() {
        let mut m = Holt::new(cfg(6.0, 25.0)).unwrap();
        for i in 0..40 {
            m.step(
                &[],
                &[Some(1.0 + 2.0 * i as f64)],
                if i == 0 { 0.0 } else { 1.0 },
                1.0,
            );
        }
        let (l, t) = (m.level()[0], m.trend()[0]);
        let on_the_null = m.step(&[], &[None], 3.0, 1.0).pred[0];
        assert!((on_the_null - (l + 3.0 * t)).abs() < 1e-12);
        let p = m.predict(&[], 2.0).pred[0];
        assert!((p - (l + 5.0 * t)).abs() < 1e-12, "{p} vs {}", l + 5.0 * t);
    }

    /// A row's weight is how much it counts: the level after a row at weight
    /// 2 is the level after the same row twice at weight 1, the second at no
    /// clock -- the review's exact check for S29. It passed on the textbook
    /// form too, where neither row's weight counted and a row at no clock
    /// changed nothing; `a_lighter_row_moves_the_level_less` is the one that
    /// failed. The trend reads a move over the clock, which the second row
    /// does not have, so the identity is the level's and `n_eff`'s.
    #[test]
    fn a_row_at_weight_two_is_the_row_given_twice() {
        let feed = |m: &mut Holt| {
            for i in 0..25 {
                let d = if i == 0 { 0.0 } else { 1.0 };
                m.step(&[], &[Some(1.0 + 0.5 * i as f64)], d, 1.0);
            }
        };
        let mut once = Holt::new(cfg(6.0, 25.0)).unwrap();
        let mut twice = Holt::new(cfg(6.0, 25.0)).unwrap();
        feed(&mut once);
        feed(&mut twice);
        once.step(&[], &[Some(40.0)], 1.0, 2.0);
        twice.step(&[], &[Some(40.0)], 1.0, 1.0);
        twice.step(&[], &[Some(40.0)], 0.0, 1.0);
        let (a, b) = (once.level()[0], twice.level()[0]);
        assert!((a - b).abs() <= 1e-12 * b.abs(), "{a} vs {b}");
        assert!((once.n_eff() - twice.n_eff()).abs() < 1e-12);
    }

    /// The review's test for S29: the same stream with its last 40 rows at
    /// weight 0.5 or at 1. A lighter row counts for less against the history
    /// before it, so the two levels differ, each what the weighted means
    /// give; the textbook form read the weight only as "learn or not", and
    /// the two were the same to the last digit.
    #[test]
    fn a_lighter_row_moves_the_level_less() {
        let c = cfg(6.0, 25.0);
        let mut s = 17u64;
        let ys: Vec<Option<f64>> = (0..80)
            .map(|i| Some(1.0 + 2.0 * i as f64 + 3.0 * lcg(&mut s)))
            .collect();
        let ds: Vec<f64> = (0..80).map(|i| if i == 0 { 0.0 } else { 1.0 }).collect();
        let mut levels = Vec::new();
        for late in [0.5, 1.0] {
            let ws: Vec<f64> = (0..80).map(|i| if i < 40 { 1.0 } else { late }).collect();
            let mut m = Holt::new(c.clone()).unwrap();
            for ((y, d), w) in ys.iter().zip(&ds).zip(&ws) {
                m.step(&[], &[*y], *d, *w);
            }
            let (_, want, _, _) = reference(&c, &ys, &ds, &ws);
            let got = m.level()[0];
            assert!((got - want).abs() < 1e-9, "weight {late}: {got} vs {want}");
            levels.push(got);
        }
        assert_ne!(levels[0], levels[1], "the weight must count");
    }

    /// Weights are relative: every row at 0.5 is every row at 1, to the bit,
    /// as for any weighted mean. (Every weight is scaled by the same power of
    /// two, so the arithmetic is exact.)
    #[test]
    fn a_constant_weight_cancels() {
        let mut s = 23u64;
        let ys: Vec<f64> = (0..60)
            .map(|i| 5.0 - 0.3 * i as f64 + lcg(&mut s))
            .collect();
        let fit = |w: f64| {
            let mut m = Holt::new(cfg(6.0, 25.0)).unwrap();
            let preds: Vec<f64> = ys
                .iter()
                .enumerate()
                .map(|(i, y)| {
                    let d = if i == 0 { 0.0 } else { 1.0 };
                    m.step(&[], &[Some(*y)], d, w).pred[0]
                })
                .collect();
            (preds, m.coefficients())
        };
        let (half, whole) = (fit(0.5), fit(1.0));
        assert_eq!(half.1, whole.1);
        for (a, b) in half.0.iter().zip(&whole.0) {
            assert!(
                a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()),
                "{a} vs {b}"
            );
        }
    }

    /// An infinite half-life forgets nothing, so the fit is the whole
    /// history's: the weighted least-squares line through every observation
    /// on the clock, read at the last one (task 278, D-H). Held against
    /// faer's LU solve of the normal equations (`crate::oracle`), on an
    /// irregular clock with uneven weights, nulls and zero weights. The
    /// weighted means before it pinned the trend to the first observation,
    /// `(l − y₁)/(t − t₁)`, and the level's error grew like that
    /// observation's noise times `ln n`; the textbook form's rate is 0
    /// there, so it forecast the first row for ever (review 2026-09-12,
    /// S30).
    #[test]
    fn an_infinite_halflife_is_the_weighted_line_fit() {
        let c = cfg(f64::INFINITY, f64::INFINITY);
        let mut st = 29u64;
        let n = 3000;
        let mut clock = 0.0;
        let (mut ys, mut ds, mut ws, mut rows) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for i in 0..n {
            let d = if i == 0 {
                0.0
            } else {
                [0.0, 0.5, 1.0, 3.0][i % 4]
            };
            let w = [1.0, 0.5, 2.0, 0.0, 1.5][i % 5];
            clock += d;
            let y = 2.0 + 0.3 * clock + 0.7 * lcg(&mut st);
            let y = (i % 7 != 3).then_some(y);
            if let Some(y) = y.filter(|_| w > 0.0) {
                rows.push((clock, w, y));
            }
            ds.push(d);
            ws.push(w);
            ys.push(y);
        }
        let mut m = Holt::new(c).unwrap();
        for i in 0..n {
            m.step(&[], &[ys[i]], ds[i], ws[i]);
        }
        // In clock units from the last observation, so the oracle's
        // intercept is the level.
        use crate::oracle::{solve, sum};
        let t_last = rows.last().unwrap().0;
        let a = |f: &dyn Fn(f64) -> f64| sum(rows.iter().map(|r| r.1 * f(r.0 - t_last)));
        let b = |f: &dyn Fn(f64) -> f64| sum(rows.iter().map(|r| r.1 * r.2 * f(r.0 - t_last)));
        let (s0, s1, s2) = (a(&|_| 1.0), a(&|t| t), a(&|t| t * t));
        let fit = solve(&[s0, s1, s1, s2], &[b(&|_| 1.0), b(&|t| t)]);
        let (level, trend) = (m.level()[0], m.trend()[0]);
        assert!(
            (level - fit[0]).abs() <= 1e-10 * fit[0].abs(),
            "level {level} vs {}",
            fit[0]
        );
        assert!(
            (trend - fit[1]).abs() <= 1e-10 * fit[1].abs(),
            "trend {trend} vs {}",
            fit[1]
        );
    }

    /// `trend_half_life = inf` beside a finite level half-life takes the
    /// line fit's gain on the slope, but the slope it reaches is the whole
    /// history's weighted least-squares slope only approximately: the level
    /// it corrects against forgets at its own half-life, so the forecast
    /// errors the slope learns from are not the line fit's residuals. At a
    /// level half-life of 5, over 3,000 rows of a line in noise, the trend
    /// was 2.2e-3 to 1.0e-2 of the slope off that line fit's on three seeds
    /// (the reviewer read 9.6e-3 on a stream of its own), not the 1e-10 it
    /// is within when the level does not decay either (`an_infinite_halflife_is_the_weighted_line_fit`; review round
    /// 10, S4-2).
    #[test]
    fn an_infinite_trend_half_life_beside_a_finite_level_is_near_the_line_fit() {
        use crate::oracle::{solve, sum};
        let mut gaps = Vec::new();
        for seed in [29u64, 30, 31] {
            let mut m = Holt::new(cfg(5.0, f64::INFINITY)).unwrap();
            let mut st = seed;
            let mut rows = Vec::new();
            for i in 0..3000 {
                let t = i as f64;
                let y = 2.0 + 0.3 * t + 0.7 * lcg(&mut st);
                m.step(&[], &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
                rows.push((t, y));
            }
            let t_last = rows.last().unwrap().0;
            let a = |f: &dyn Fn(f64) -> f64| sum(rows.iter().map(|r| f(r.0 - t_last)));
            let b = |f: &dyn Fn(f64) -> f64| sum(rows.iter().map(|r| r.1 * f(r.0 - t_last)));
            let (s0, s1, s2) = (a(&|_| 1.0), a(&|t| t), a(&|t| t * t));
            let fit = solve(&[s0, s1, s1, s2], &[b(&|_| 1.0), b(&|t| t)]);
            gaps.push(((m.trend()[0] - fit[1]) / fit[1]).abs());
        }
        println!("relative gaps to the line fit's slope: {gaps:?}");
        let worst = gaps.iter().fold(0.0f64, |m, g| m.max(*g));
        assert!(worst > 1e-6, "near, not exact: {gaps:?}");
        assert!(worst < 3e-2, "about 1e-2: {gaps:?}");
    }

    /// The defect itself (task 278, D-H): a flat series whose first
    /// observation is 10 off. The weighted means carried that 10 into the
    /// trend, `(l − y₁)/(t − t₁)`, so the level's error grew like `−10·ln
    /// n` without decay and through every warm-up under one. The line fit's
    /// first error falls as `1/n`.
    #[test]
    fn the_first_observations_error_does_not_grow() {
        for (h, rows, bound) in [(f64::INFINITY, 10_000, 0.02), (1000.0, 1000, 0.1)] {
            let mut m = Holt::new(cfg(h, 4.0 * h)).unwrap();
            let mut st = 3u64;
            for i in 0..rows {
                let y = 3.0 + 0.5 * lcg(&mut st) + if i == 0 { 10.0 } else { 0.0 };
                m.step(&[], &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
            }
            let off = m.level()[0] - 3.0;
            assert!(
                off.abs() < bound,
                "half-life {h}: the level is {off} off after {rows} rows"
            );
        }
    }

    /// Once the line fit's gains fall below the textbook's, every row takes
    /// the textbook's, so the model is the textbook recursion from there:
    /// beside a state that never kept a line fit, on the same rows, the
    /// predictions agree to 1e-12 once the warm-ups' differences have
    /// decayed at the trend's half-life (`tests/test_second_opinion.py`
    /// holds the same against statsmodels' `Holt`).
    #[test]
    fn settled_it_is_the_textbook_recursion() {
        let c = cfg(6.0, 25.0);
        let mut m = Holt::new(c.clone()).unwrap();
        let mut old = without_line_fit(c);
        let mut st = 41u64;
        let mut level = 3.0;
        for i in 0..2000 {
            level += 0.4 + 0.5 * lcg(&mut st);
            let d = if i == 0 { 0.0 } else { 1.0 };
            let a = m.step(&[], &[Some(level)], d, 1.0).pred[0];
            let b = old.step(&[], &[Some(level)], d, 1.0).pred[0];
            if i >= 1000 {
                assert!((a - b).abs() <= 1e-12 * b.abs(), "row {i}: {a} vs {b}");
            }
        }
    }

    /// A state saved before the line fit (schema 63 and earlier) has none,
    /// and goes on with the textbook gains alone, the recursion it was
    /// fitted under (hard rule 5; the kept fixtures hold it within 1e-14,
    /// its rounding moved by the reciprocals the recursion keeps): a row
    /// the line fit would have taken does not reach it.
    #[test]
    fn a_state_saved_before_the_line_fit_goes_on_with_the_textbook_gains() {
        let ds = [0.0, 1.0, 0.25, 7.0, 1.0, 1.0, 0.5, 13.0, 2.0, 1.0, 1.0, 3.0];
        let ws = [1.0, 0.5, 2.0, 1.0, 0.25, 1.0, 3.0, 1.0, 0.0, 1.0, 1.5, 1.0];
        let mut s = 5u64;
        let mut ys: Vec<Option<f64>> = (0..ds.len())
            .map(|i| Some(3.0 + 0.8 * i as f64 + lcg(&mut s)))
            .collect();
        ys[5] = None;
        for c in [cfg(4.0, 9.0), cfg(f64::INFINITY, f64::INFINITY)] {
            let (want, level, trend, _) = textbook(&c, &ys, &ds, &ws);
            let mut m = without_line_fit(c.clone());
            for (i, ((&y, &d), &w)) in ys.iter().zip(&ds).zip(&ws).enumerate() {
                let p = m.step(&[], &[y], d, w).pred[0];
                // The decays here run through `powf`, the model's through
                // `exp2`; the kept fixtures hold the rest (state_fixtures).
                assert!(
                    (p - want[i]).abs() <= 1e-13 * (1.0 + want[i].abs())
                        || (p.is_nan() && want[i].is_nan()),
                    "row {i}: {p} vs {}",
                    want[i]
                );
            }
            assert!((m.level()[0] - level).abs() <= 1e-13 * (1.0 + level.abs()));
            assert!((m.trend()[0] - trend).abs() <= 1e-13 * (1.0 + trend.abs()));
            assert!(m.fit_weight.is_empty(), "and it saves none");
        }
        // The line fit is all three vectors or none.
        let mut m = Holt::new(cfg(4.0, 9.0)).unwrap();
        m.fit_age.clear();
        assert!(
            Holt::restore(&m.state()).is_err(),
            "a part of the line fit is refused"
        );
    }

    /// `trend = false` holds the trend at zero (docs/PLAN.md task 115, S30):
    /// the forecast is flat and the level is the weighted mean of the
    /// observations, each at its weight times `0.5^(age/level_half_life)`,
    /// written here from that definition. The clock is irregular, and a null
    /// and a zero weight are among the rows, which observe nothing.
    #[test]
    fn without_a_trend_the_level_is_the_weighted_mean_of_the_observations() {
        let ds = [0.0, 1.0, 0.25, 7.0, 1.0, 1.0, 0.5, 13.0, 2.0, 1.0, 1.0, 3.0];
        let ws = [1.0, 0.5, 2.0, 1.0, 0.25, 1.0, 3.0, 1.0, 0.0, 1.0, 1.5, 1.0];
        let mut s = 5u64;
        let mut ys: Vec<Option<f64>> = (0..ds.len())
            .map(|i| Some(3.0 + 0.8 * i as f64 + lcg(&mut s)))
            .collect();
        ys[5] = None;
        let mut m = Holt::new(HoltCfg {
            trend: false,
            ..cfg(4.0, 9.0)
        })
        .unwrap();
        let (mut clock, mut seen) = (0.0, Vec::<(f64, f64, f64)>::new());
        for (i, ((&y, &d), &w)) in ys.iter().zip(&ds).zip(&ws).enumerate() {
            clock += d;
            let pred = m.step(&[], &[y], d, w).pred[0];
            let at = |&(t, w, _): &(f64, f64, f64)| w * 0.5f64.powf((clock - t) / 4.0);
            let num: f64 = seen.iter().map(|o| at(o) * o.2).sum();
            let den: f64 = seen.iter().map(at).sum();
            if seen.is_empty() {
                assert!(pred.is_nan(), "row {i}: nothing observed yet");
            } else {
                let want = num / den;
                assert!(
                    (pred - want).abs() < 1e-12 * want.abs(),
                    "row {i}: {pred} vs {want}"
                );
            }
            if let Some(y) = y.filter(|_| w > 0.0) {
                seen.push((clock, w, y));
            }
            assert_eq!(m.trend()[0], 0.0, "row {i}: the trend never moves");
            assert_eq!(m.coefficients()[0][1], 0.0, "row {i}: and reports 0");
        }
        assert_eq!(seen.len(), 10, "a null and a zero weight observe nothing");
    }

    /// A config written before the switch has no `trend`, and loads with the
    /// trend on: what that state was produced under.
    #[test]
    fn a_config_written_before_the_trend_switch_keeps_its_trend() {
        let json = r#"{"n_targets": 1, "level_half_life": 10.0, "trend_half_life": 40.0, "min_weight": 0.0}"#;
        let c: HoltCfg = serde_json::from_str(json).expect("loads without the field");
        assert!(c.trend, "the omitted field is the trend on");
    }

    /// Each target's weight is its level's, decayed over the clock since the
    /// target was last seen: `Σᵢ wᵢ·2^(−(T − tᵢ)/h)` over the rows that
    /// carried it, `T` the clock now and `h` the level's half-life. The
    /// second target is absent on the last rows, so its weight has aged
    /// since; a zero-weight row carries nothing.
    #[test]
    fn a_targets_weight_is_its_rows_decayed_to_now() {
        let h = 4.0;
        let mut m = Holt::new(HoltCfg {
            n_targets: 2,
            ..cfg(h, 8.0)
        })
        .unwrap();
        let mut clock = 0.0;
        let mut carried: [Vec<(f64, f64)>; 2] = [Vec::new(), Vec::new()];
        for i in 0..12usize {
            let d = 0.5 + 0.75 * (i % 3) as f64;
            clock += d;
            let w = if i == 5 {
                0.0
            } else {
                0.5 + 0.25 * (i % 4) as f64
            };
            let y = [Some(i as f64), (i % 4 != 3 && i < 9).then(|| -(i as f64))];
            m.step(&[], &y, d, w);
            for t in 0..2 {
                if y[t].is_some() && w > 0.0 {
                    carried[t].push((clock, w));
                }
            }
        }
        let mut out = Vec::new();
        assert!(m.target_n_eff_into(&mut out));
        assert_eq!(out.len(), 2);
        for t in 0..2 {
            let want: f64 = carried[t]
                .iter()
                .map(|(at, w)| w * (-(clock - at) / h).exp2())
                .sum();
            assert!(
                (out[t] - want).abs() <= 1e-12 * want,
                "target {t}: {} vs {want}",
                out[t]
            );
        }
        assert!(m.since[1] > 0.0, "the second target has aged since");
    }

    /// Task 252: a model with no targets is refused by name.
    #[test]
    fn a_model_with_no_targets_is_refused_by_name() {
        let e = Holt::new(HoltCfg {
            n_targets: 0,
            ..cfg(6.0, 25.0)
        })
        .unwrap_err();
        assert_eq!(
            e,
            "holt: the model needs at least one target, but it has none. Name at least one \
             column in targets."
        );
        // Every problem at once, each a line of its own.
        let e = HoltCfg {
            n_targets: 0,
            trend_half_life: 0.0,
            ..cfg(6.0, 25.0)
        }
        .validate()
        .unwrap_err();
        let lines: Vec<&str> = e.lines().collect();
        assert_eq!(lines.len(), 2, "{e}");
        assert!(
            lines[1].starts_with("holt: trend_half_life must be above 0"),
            "{e}"
        );
    }

    /// Task 281 (mutants): with every observation at one clock the line
    /// fit's spread `Q` is 0 and its slope not determined: the level is
    /// the observations' weighted mean and the trend does not move, nor
    /// does the trend's weight, which counts only moves over some clock.
    /// Under a finite level half-life the line fit's gain `w/S0` ties the
    /// textbook's `w/(λW + w)` here (λ = 1, `W = S0`), and at a tie the
    /// textbook's weighted mean stands, `(W·p + w·y)·(1/(W + w))` to the
    /// bit; under an infinite one there is no textbook gain, and the level
    /// is the line fit's, `p + w·(1/S0)·(y − p)`. Each denominator is
    /// divided into once and its reciprocal multiplied by, as the model
    /// forms them.
    #[test]
    fn observations_at_one_clock_give_their_weighted_mean_and_no_slope() {
        for level in [10.0, f64::INFINITY] {
            let mut m = Holt::new(cfg(level, 20.0)).unwrap();
            let mut s = 21u64;
            let (mut lvl, mut w_sum) = (0.0f64, 0.0f64);
            let mut parted = 0;
            for i in 0..12 {
                let y = 5.0 + 3.0 * lcg(&mut s);
                let w = 0.5 + lcg(&mut s).abs();
                m.step(&[], &[Some(y)], 0.0, w);
                if i == 0 {
                    lvl = y;
                } else {
                    let mean = (w_sum * lvl + w * y) * (1.0 / (w_sum + w));
                    let line = lvl + w * (1.0 / (w_sum + w)) * (y - lvl);
                    parted += usize::from(mean.to_bits() != line.to_bits());
                    lvl = if level.is_finite() { mean } else { line };
                }
                w_sum += w;
                assert_eq!(m.level()[0].to_bits(), lvl.to_bits(), "{level}, row {i}");
                assert_eq!(m.trend()[0], 0.0, "{level}, row {i}");
                assert_eq!(m.w_trend[0], 0.0, "{level}, row {i}");
            }
            assert!(parted > 0, "{level}: the two forms never parted");
        }
    }

    /// Task 281 (mutants): on a target's second observation, `s` after its
    /// first and at the same weight, the line fit's slope gain `h = 1/s`
    /// ties the textbook trend's `w/(0 + w) · g/s` with `g = 1`, and at a
    /// tie the textbook's slope stands: `(level − previous)·(1/s)`, to the
    /// bit, where the line's step `h·(y − p)` rounds otherwise on these
    /// values. The trend's weight is then the row's. (Task 291a: with the
    /// reciprocals kept, `h = (s/2)·(1/(s²/2))` and the textbook's
    /// `1·(1/s)` are the same double at `s = 3`, so the forms part only
    /// where `(y0 + (y1 − y0)) − y0` rounds away from `y1 − y0`: these
    /// values, one pair in 160,000 of a grid of two- and three-place ones.)
    #[test]
    fn at_a_tie_of_the_slope_gains_the_textbook_slope_stands() {
        let (y0, y1, s) = (0.09, 0.34, 3.0);
        let mut m = Holt::new(cfg(10.0, 20.0)).unwrap();
        m.step(&[], &[Some(y0)], 0.0, 1.0);
        m.step(&[], &[Some(y1)], s, 1.0);
        let level = m.level()[0];
        let textbook = (level - y0) * (1.0 / s);
        let h = (s * 0.5) * (1.0 / (s * s * 0.5));
        assert_eq!(h.to_bits(), (1.0 / s).to_bits(), "the gains tie");
        let line = h * (y1 - y0);
        assert_ne!(
            textbook.to_bits(),
            line.to_bits(),
            "the two forms part here"
        );
        assert_eq!(m.trend()[0].to_bits(), textbook.to_bits());
        assert_eq!(m.w_trend[0], 1.0);
    }

    /// Every prediction, level and trend of a 400-row stream digested
    /// (FNV-1a over the bits, NaN folded to one pattern), and four picks,
    /// held to the bit. The fixtures and the oracle tests above hold the
    /// model to its definition within rounding, or on rows whose last bits
    /// happen to agree, so a change of the arithmetic's form -- a division
    /// written as a reciprocal kept and multiplied by -- passes them. This
    /// test catches it.
    ///
    /// No library's rounding is on the path, so the bits are every
    /// platform's: each clock step is a whole number of each half-life, so
    /// every decay `2^(-s/h)` is an exact power of two, which any `exp2`
    /// returns exactly, where at other arguments glibc's and Apple's differ
    /// in the last bit. Three models: finite half-lives, an infinite level
    /// half-life (the line fit's level, the trend choosing between its
    /// gains), and an infinite trend half-life (the line fit's slope on
    /// every row). Under decays of a half or less a row, the line fit's
    /// slope gain is the larger only while a target's fit is young, or on a
    /// light row after a heavy one, whose textbook gain the heavy row's held
    /// weight shrinks: so targets start every 40 rows and the weights
    /// alternate heavy and light (the select took the line fit's trend on
    /// 60 rows of the two models that choose, the textbook's on 3,088). A
    /// zero weight every 23 rows, rows at one clock, and nulls, across which
    /// a target's clock runs on.
    #[test]
    fn the_arithmetic_is_pinned_to_the_bit() {
        const N: usize = 8;
        let run = |level: f64, trend: f64| {
            let mut m = Holt::new(HoltCfg {
                horizons: Vec::new(),
                n_targets: N,
                level_half_life: level,
                trend_half_life: trend,
                min_weight: 0.25,
                trend: true,
                ..Default::default()
            })
            .unwrap();
            let mut s = 29u64;
            let mut h = 0xcbf2_9ce4_8422_2325u64;
            let mut fold = |v: f64| {
                let b = if v.is_nan() {
                    0x7ff8_0000_0000_0000
                } else {
                    v.to_bits()
                };
                for byte in b.to_le_bytes() {
                    h = (h ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3);
                }
            };
            let mut got = Vec::new();
            for i in 0..400usize {
                let d = match i {
                    0 => 0.0,
                    _ if i % 11 == 5 => 0.0,
                    _ => [2.0, 4.0, 2.0, 6.0][i % 4],
                };
                let t = i as f64;
                let y: Vec<Option<f64>> = (0..N)
                    .map(|j| {
                        let v = 3.0 * j as f64 + (0.05 + 0.1 * j as f64) * t + lcg(&mut s);
                        (i >= 40 * j && (i + j) % 7 != 2).then_some(v)
                    })
                    .collect();
                let w = if i % 23 == 9 {
                    0.0
                } else if i % 2 == 0 {
                    40.0 + lcg(&mut s).abs()
                } else {
                    0.5 + lcg(&mut s).abs()
                };
                let out = m.step(&[], &y, d, w);
                for v in out.pred.iter().chain(m.level()).chain(m.trend()) {
                    fold(*v);
                }
                fold(out.n_eff);
                if i == 399 {
                    got.extend(out.pred[..4].iter().copied());
                }
            }
            (h, got)
        };
        let finite = run(1.0, 2.0);
        let line_level = run(f64::INFINITY, 2.0);
        let line_trend = run(1.0, f64::INFINITY);
        println!("finite {finite:x?}\nline_level {line_level:x?}\nline_trend {line_trend:x?}");
        let bits = |v: &[f64]| v.iter().map(|p| p.to_bits()).collect::<Vec<_>>();
        let want = [
            (
                0xd5e4_f9e3_229f_e96f_u64,
                [
                    0x4034_6a48_2656_6dda_u64,
                    0x404e_b926_46ed_cfd4,
                    0x405a_c556_48da_f6d5,
                    0x4061_984f_589c_ae42,
                ],
            ),
            (
                0x1016_03b8_5ea3_3ea8,
                [
                    0x4034_397f_9f7f_1d5a,
                    0x4050_01f1_612c_849b,
                    0x405b_391f_b64f_a4c8,
                    0x4062_b4c0_b017_d1c6,
                ],
            ),
            (
                0xba56_7b60_4d4a_fee1,
                [
                    0x4033_ff43_ab3f_7f16,
                    0x404f_0691_7a50_e8bb,
                    0x405a_a84c_586f_bc41,
                    0x4062_a869_923a_1638,
                ],
            ),
        ];
        for ((name, got), (h, picks)) in [
            ("finite", finite),
            ("line_level", line_level),
            ("line_trend", line_trend),
        ]
        .into_iter()
        .zip(want)
        {
            assert_eq!((got.0, bits(&got.1)), (h, picks.to_vec()), "{name}");
        }
    }

    /// A row of subnormal weight learns nothing, as a row of weight 0 learns
    /// nothing (hard rule 9; review fable2, C7): the stream with it is the
    /// stream with that row at weight 0, to the bit, and so first or later,
    /// on a trend or without. The line fit kept `1/S0`, `1/Q` and `1/(λW +
    /// w)` to multiply by, and the reciprocal of a number below `2⁻¹⁰²⁴` is
    /// infinite: a first row of weight `1e-310` made the second row's slope
    /// gain `0·∞`, and the level and trend NaN for good. A weight of `1e-300`,
    /// normal, is still learned.
    #[test]
    fn a_row_of_subnormal_weight_is_a_row_of_weight_zero() {
        let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        for trend in [true, false] {
            for at in [0usize, 1, 7, 200] {
                for tiny in [5e-324, 1e-320, 1e-310, 2e-308] {
                    let run = |w_at: f64| {
                        let mut m = Holt::new(HoltCfg {
                            min_weight: 3.0,
                            trend,
                            ..cfg(20.0, 80.0)
                        })
                        .unwrap();
                        let mut s = 7u64;
                        let mut out = Vec::new();
                        for i in 0..400 {
                            let w = if i == at { w_at } else { 1.0 };
                            let y = 1.0 + 0.01 * i as f64 + 0.05 * lcg(&mut s);
                            let st = m.step(&[], &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, w);
                            out.push((st.pred[0].to_bits(), st.n_eff.to_bits()));
                        }
                        (out, bits(&m.coefficients()[0]))
                    };
                    let (got, coef) = run(tiny);
                    let (want, want_coef) = run(0.0);
                    assert!(
                        got == want && coef == want_coef,
                        "trend {trend}, weight {tiny:e} at row {at}"
                    );
                    // On the line, or with no trend a half-life's lag behind it.
                    let last = f64::from_bits(got[399].0);
                    let lag = if trend {
                        0.0
                    } else {
                        0.01 * 20.0 / std::f64::consts::LN_2
                    };
                    assert!((last - (4.99 - lag)).abs() < 0.1, "trend {trend}: {last}");
                }
            }
        }
        // A normal weight, however small, is learned: the first row is the level.
        let mut m = Holt::new(cfg(20.0, 80.0)).unwrap();
        m.step(&[], &[Some(3.0)], 0.0, 1e-300);
        assert_eq!(m.level(), &[3.0]);
        // The smallest normal weight is the first that is learned.
        let mut m = Holt::new(cfg(20.0, 80.0)).unwrap();
        m.step(&[], &[Some(3.0)], 0.0, f64::MIN_POSITIVE);
        assert_eq!(m.level(), &[3.0]);
        assert_eq!(m.n_eff(), f64::MIN_POSITIVE);
    }

    /// The line fit's spread of ages `Q` below `2⁻¹⁰²⁴`, where its
    /// reciprocal is infinite, reached with normal weights: a first row at
    /// `1e-300` and a second `1e-5` of a clock unit later leave `Q = 1e-310`,
    /// and the slope's gain `w·ā/Q` was `1e-305·∞`, the trend NaN for good
    /// (review fable2, C7). The slope is read as not determined there, as at
    /// `Q = 0`: the level's weighted mean, the slope unmoved. A clock step
    /// below the smallest normal, whose reciprocal is infinite too, is a
    /// step of no clock: the level moves and the slope does not. Each stream
    /// then follows its line.
    #[test]
    fn a_spread_or_a_clock_step_too_small_to_invert_leaves_the_fit_finite() {
        for (w0, d1) in [(1e-300, 1e-5), (1.0, 1e-310), (1e-300, 1e-160)] {
            let mut m = Holt::new(cfg(20.0, 80.0)).unwrap();
            m.step(&[], &[Some(1.0)], 0.0, w0);
            m.step(&[], &[Some(1.0)], d1, 1.0);
            let mut last = f64::NAN;
            for i in 2..400 {
                last = m.step(&[], &[Some(1.0 + 0.01 * i as f64)], 1.0, 1.0).pred[0];
                assert!(
                    m.level()[0].is_finite() && m.trend()[0].is_finite(),
                    "{w0:e}, {d1:e}: row {i}, {:?}",
                    m.coefficients()
                );
            }
            assert!((last - 4.99).abs() < 1e-2, "{w0:e}, {d1:e}: {last}");
        }
    }

    /// A clock step small enough that the slope it measures is past the
    /// largest double leaves the fit finite (review 2026-11-23, REG-4):
    /// at `y = 0, 10, 20` and a step of `1e-308`, `1/s` is finite, so the
    /// step was read as clock, and the slope `5·10^308` made the trend
    /// infinite and the level NaN for good. Such a step is read as one of
    /// no clock, as one whose reciprocal is infinite is. And a slope
    /// within range at a weight past it (`s = 1e-200`, a rise of `1e50`,
    /// `w = 1e100`) is formed without `w · slope` overflowing.
    #[test]
    fn a_slope_past_the_largest_double_leaves_the_fit_finite() {
        for (d1, rise, w) in [
            (1e-308, 10.0, 1.0),
            (1e-200, 1e50, 1e100),
            (1e-300, 1e10, 1.0),
        ] {
            for hl in [10.0, f64::INFINITY] {
                let mut m = Holt::new(cfg(hl, hl)).unwrap();
                m.step(&[], &[Some(0.0)], 0.0, w);
                m.step(&[], &[Some(rise)], d1, w);
                for i in 2..50 {
                    m.step(&[], &[Some(rise * i as f64)], 1.0, w);
                    assert!(
                        m.level()[0].is_finite() && m.trend()[0].is_finite(),
                        "{d1:e}, {rise:e}, {w:e}, {hl}: row {i}, {:?}",
                        m.coefficients()
                    );
                }
            }
        }
    }

    /// The forecasts at horizons, task 559: each target's two a horizon
    /// after the predictions, `l + b·(s + Δ)` and `Δ·(l + b·(s + Δ/2))` from
    /// the level and trend before the row, to the bit; `Δ = 0` is the row's
    /// prediction; `predict` reads what `step` writes.
    #[test]
    fn the_forecasts_are_the_line_run_ahead_and_its_integral() {
        use crate::OnlineModel;
        let spans = vec![0.0, 2.5, 40.0];
        let mut m = Holt::new(HoltCfg {
            horizons: spans.clone(),
            n_targets: 2,
            ..cfg(5.0, 20.0)
        })
        .unwrap();
        assert_eq!(m.n_outputs(), 2 * (1 + 6));
        let mut state = 9u64;
        for i in 0..300 {
            let d = 0.5 + (lcg(&mut state) + 1.0) * 0.5;
            let y0 = 1.0 + 0.3 * i as f64 + lcg(&mut state);
            let y1 = -2.0 + 0.1 * i as f64 + lcg(&mut state);
            let (l, b) = (m.level.clone(), m.trend.clone());
            let since = m.since.clone();
            let read = m.predict(&[], d).pred;
            let got = m.step(&[], &[Some(y0), Some(y1)], d, 1.0).pred;
            assert_eq!(got.len(), 14);
            assert_eq!(
                got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                read.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "row {i}: predict and step"
            );
            if i == 0 {
                assert!(got.iter().all(|v| v.is_nan()), "nothing to extrapolate");
                continue;
            }
            for j in 0..2 {
                let s = since[j] + d;
                for (k, &dt) in spans.iter().enumerate() {
                    let at = 2 + j * 6 + 2 * k;
                    assert_eq!(got[at].to_bits(), (l[j] + b[j] * (s + dt)).to_bits());
                    let area = dt * (l[j] + b[j] * (s + 0.5 * dt));
                    assert_eq!(got[at + 1].to_bits(), area.to_bits());
                }
                // Δ = 0 is the prediction, and its integral 0
                assert_eq!(got[2 + j * 6].to_bits(), got[j].to_bits());
                assert_eq!(got[3 + j * 6], 0.0);
            }
        }
    }

    /// Scaling every weight by a power of two moves no prediction past
    /// rounding while the line fit's spread is inside the doubles (review
    /// 2026-11-23, REG-5): it was formed as `S0 · w · ā² / S0'`, whose
    /// `S0 · w` underflowed at `w = 2^{-660}` (the slope's gain lost, the
    /// forecast of a straight line 41% short) and overflowed at `w =
    /// 2^{330}` with a clock step of `10^60` (the slope's gain 0 for good).
    /// Out of range it is formed as `(S0 / S0') · w · ā²`, each factor in
    /// range, which rounds otherwise than the plain form.
    #[test]
    fn scaling_every_weight_by_a_power_of_two_moves_no_prediction() {
        for (scale, step) in [
            (2f64.powi(-660), 1.0),
            (2f64.powi(330), 1e60),
            (2f64.powi(-900), 1e30),
        ] {
            for hl in [5.0 * step, f64::INFINITY] {
                let run = |w: f64| {
                    let mut m = Holt::new(cfg(hl, 4.0 * hl)).unwrap();
                    let mut s = 3u64;
                    (0..60)
                        .map(|i| {
                            let y = 10.0 * i as f64 + lcg(&mut s);
                            let d = if i == 0 {
                                0.0
                            } else {
                                step * (1.0 + 0.5 * lcg(&mut s))
                            };
                            m.step(&[], &[Some(y)], d, w * (1.5 + lcg(&mut s))).pred[0]
                        })
                        .collect::<Vec<_>>()
                };
                let (got, want) = (run(scale), run(1.0));
                for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                    assert!(
                        g.to_bits() == w.to_bits() || (g - w).abs() <= 1e-12 * w.abs(),
                        "{scale:e}, {step:e}, {hl}: row {i}, {g} against {w}"
                    );
                }
            }
        }
    }

    /// A horizon below 0, not finite, given twice, or too many, is refused.
    #[test]
    fn horizons_are_refused_with_a_reason() {
        let bad = |h: Vec<f64>, says: &str| {
            let e = Holt::new(HoltCfg {
                horizons: h,
                ..cfg(5.0, 20.0)
            })
            .unwrap_err();
            assert!(e.contains(says), "{says}: {e}");
        };
        bad(vec![-1.0], "0 or more");
        bad(vec![f64::NAN], "0 or more");
        bad(vec![f64::INFINITY], "finite");
        bad(vec![2.0, 1.0, 2.0], "holds 2 twice");
        bad((0..65).map(f64::from).collect(), "at most 64");
    }

    /// Horizons are the plain level and trend's forecast: a damped trend, a
    /// season or a given start has none to integrate, and is refused beside
    /// them rather than reported without.
    #[test]
    fn horizons_are_refused_beside_the_fixed_forms() {
        let e = Holt::new(HoltCfg {
            horizons: vec![2.0],
            damping_half_life: 9.0,
            start: HoltStart::Heuristic,
            ..cfg(5.0, 20.0)
        })
        .unwrap_err();
        assert!(
            e.contains("forecast_spans are forecasts of the plain level"),
            "{e}"
        );
    }
}
