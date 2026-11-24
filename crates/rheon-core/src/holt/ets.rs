//! `holt`'s fixed forms of seasonal exponential smoothing (task 488): an
//! additive or multiplicative season, a damped additive trend, and the
//! known and heuristic starts. The module docs of `holt` state the update.
//!
//! Each form's arithmetic follows statsmodels' `ExponentialSmoothing`
//! (`holtwinters/model.py`, `_predict`), operation for operation, at a
//! step of one clock unit and a weight of 1, where every gain below is the
//! fixed one to the bit: `w·a/(λ + w·a)` with `a = 1 − λ` is `a` exactly,
//! since `λ + fl(1 − λ)` rounds to 1.

use serde::{Deserialize, Serialize};

use super::{HOLT_MAX_STATE_BYTES, HOLT_STATE_BYTES_CEILING, Holt, HoltCfg, HoltSeason, HoltStart};
use crate::ClockValue;

/// The rows the heuristic start holds for a target, until it has the
/// seasons its definition reads: each row's value, weight, step since the
/// row before (the start's first step for the first), decayed clock and
/// phase.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Held {
    pub(crate) y: Vec<f64>,
    pub(crate) w: Vec<f64>,
    pub(crate) step: Vec<f64>,
    pub(crate) at: Vec<f64>,
    pub(crate) phase: Vec<u32>,
}

/// The bytes one held row takes: four doubles and a phase.
const HELD_ROW_BYTES: u64 = 4 * 8 + 4;

/// `0.5^(s/h)`, the decay over `s` clock units at half-life `h`; an
/// infinite half-life forgets nothing. The helper `holt` has always used.
fn decay(s: f64, h: f64) -> f64 {
    if h.is_infinite() {
        1.0
    } else {
        (-(s / h)).exp2()
    }
}

/// The gain of a weighted mean whose held weight is the one a stream of
/// unit rows at this step saturates at, `1/(1 − λ)`: `w·a/(λ + w·a)`,
/// `a = 1 − λ`. At `w = 1` it is `a` to the bit.
fn fixed_gain(lam: f64, w: f64) -> f64 {
    let a = 1.0 - lam;
    let wa = w * a;
    wa / (lam + wa)
}

impl HoltCfg {
    /// Whether this config is one of the fixed forms: a start other than
    /// the warm one.
    pub(crate) fn fixed(&self) -> bool {
        self.start != HoltStart::Warm
    }

    /// `P`, the season's span in clock units.
    pub(crate) fn period(&self) -> f64 {
        if self.season_length > 0.0 {
            self.season_length
        } else {
            self.season_periods as f64
        }
    }

    /// The step the initial states stand before the first row: a season's
    /// bucket, `P/m`, or one clock unit without a season.
    pub(crate) fn first_step(&self) -> f64 {
        if self.season == HoltSeason::None {
            1.0
        } else {
            self.period() / self.season_periods as f64
        }
    }

    /// The rows the heuristic start reads: statsmodels' `k` seasons, the
    /// fewer of 5 and the rows', raised to leave 10 seasonally adjusted
    /// values (`10 + 2⌊m/2⌋` rows), which is 6 at `m = 2`; 10 rows without
    /// a season.
    pub fn start_rows(&self) -> usize {
        if self.season == HoltSeason::None {
            return 10;
        }
        let m = self.season_periods.max(1);
        let min_obs = 10 + 2 * (m / 2);
        5.max(min_obs.div_ceil(m)) * m
    }

    /// The bytes the state holds at most, the heuristic start's rows
    /// included: per target, the level's, the trend's and the line fit's
    /// ten doubles and a flag, the `m` seasonal values and their clocks, and
    /// the held rows.
    pub fn state_bytes(&self) -> u64 {
        let m = if self.season == HoltSeason::None {
            0
        } else {
            self.season_periods as u64
        };
        let held = if self.start == HoltStart::Heuristic {
            self.start_rows() as u64 * HELD_ROW_BYTES
        } else {
            0
        };
        let per_target = 10 * 8 + 1 + 16 * m + held;
        64 + (self.n_targets as u64).saturating_mul(per_target)
    }

    fn cap(&self) -> u64 {
        self.max_state_bytes.unwrap_or(HOLT_MAX_STATE_BYTES)
    }

    /// The fixed forms' refusals: a form refused beside a start that has no
    /// definition for it, a parameter beside a form that does not read it,
    /// initial states of the wrong shape, and a state past its cap.
    pub(crate) fn validate_forms(&self, no: &mut crate::Refusals) {
        let seasonal = self.season != HoltSeason::None;
        let damped = self.damping_half_life.is_finite();
        if self.start == HoltStart::Warm && (seasonal || damped) {
            no.refuse(format!(
                "holt: a {} has no warm start: the weighted means' first rows confound the \
                 level with {}. Set start to \"known\" with the initial states, or to \
                 \"heuristic\".",
                if seasonal { "season" } else { "damped trend" },
                if seasonal {
                    "the seasonal values"
                } else {
                    "a trend the damping has not yet shrunk"
                },
            ));
        }
        if seasonal {
            if self.season_periods < 2 {
                no.refuse(format!(
                    "holt: seasonal_periods must be at least 2, but it is {}. Give the number \
                     of values a season is held in.",
                    self.season_periods
                ));
            }
            if !(self.season_length >= 0.0 && self.season_length.is_finite()) {
                no.refuse(format!(
                    "holt: season_length must be a positive number of clock units, but it is \
                     {}. Give the season's span, or leave it out for seasonal_periods clock \
                     units.",
                    self.season_length
                ));
            }
            if self.seasonal_half_life <= 0.0 || self.seasonal_half_life.is_nan() {
                no.refuse(format!(
                    "holt: seasonal_half_life must be above 0, but it is {}. Set it to a \
                     positive number of clock units, or to inf to hold the seasonal values \
                     where they start.",
                    self.seasonal_half_life
                ));
            }
        } else {
            if self.season_periods != 0 || self.season_length != 0.0 {
                no.refuse(
                    "holt: seasonal_periods and season_length size a season, but seasonal is \
                     not set. Set seasonal to \"add\" or \"mul\", or leave them out.",
                );
            }
            if !self.seasonal_half_life.is_infinite() || self.seasonal_half_life < 0.0 {
                no.refuse(
                    "holt: seasonal_half_life sets how fast the seasonal values forget, but \
                     seasonal is not set. Set seasonal to \"add\" or \"mul\", or leave \
                     seasonal_half_life out.",
                );
            }
        }
        if self.damping_half_life <= 0.0 || self.damping_half_life.is_nan() {
            no.refuse(format!(
                "holt: damping_half_life must be above 0, but it is {}. Set it to a positive \
                 number of clock units, or to inf for an undamped trend.",
                self.damping_half_life
            ));
        } else if damped && !self.trend {
            no.refuse(
                "holt: damping_half_life damps the trend, but trend=False holds the trend at 0. \
                 Leave damping_half_life out, or set trend to True.",
            );
        }
        self.validate_initial(no);
        if let Some(cap) = self.max_state_bytes {
            if cap == 0 {
                no.refuse(
                    "holt: max_state_bytes must be above 0, but it is 0. Give the bytes the \
                     state may hold, or leave it out for 256 KB.",
                );
                return;
            }
            if cap > HOLT_STATE_BYTES_CEILING {
                no.refuse(format!(
                    "holt: max_state_bytes may be at most {HOLT_STATE_BYTES_CEILING} (64 MiB), \
                     but it is {cap}. A group's state is held in memory for every group of the \
                     bank at once. Set max_state_bytes to 64 MiB or less."
                ));
                return;
            }
        }
        let bytes = self.state_bytes();
        if bytes > self.cap() {
            no.refuse(format!(
                "holt: its state would hold {bytes} bytes, past max_state_bytes of {}, with {} \
                 targets, seasonal_periods {}{}. Set max_state_bytes to {bytes} or more, or \
                 lower seasonal_periods.",
                self.cap(),
                self.n_targets,
                self.season_periods,
                if self.start == HoltStart::Heuristic {
                    format!(" and the heuristic start's {} held rows", self.start_rows())
                } else {
                    String::new()
                },
            ));
        }
    }

    fn validate_initial(&self, no: &mut crate::Refusals) {
        let given = !self.initial_level.is_empty()
            || !self.initial_trend.is_empty()
            || !self.initial_seasonal.is_empty();
        if self.start != HoltStart::Known {
            if given {
                no.refuse(
                    "holt: initial_level, initial_trend and initial_seasonal are the known \
                     start's states, but start is not \"known\". Set start to \"known\", or \
                     leave them out.",
                );
            }
            return;
        }
        let n = self.n_targets;
        if self.initial_level.len() != n {
            no.refuse(format!(
                "holt: the known start needs initial_level for each of the {n} targets, but it \
                 has {}. Give one level a target.",
                self.initial_level.len()
            ));
        }
        let want_trend = if self.trend { n } else { 0 };
        if self.initial_trend.len() != want_trend {
            no.refuse(if self.trend {
                format!(
                    "holt: the known start needs initial_trend for each of the {n} targets, but \
                     it has {}. Give one trend a target.",
                    self.initial_trend.len()
                )
            } else {
                "holt: initial_trend is the trend's start, but trend=False holds the trend at \
                 0. Leave initial_trend out."
                    .to_string()
            });
        }
        let m = if self.season == HoltSeason::None {
            0
        } else {
            self.season_periods
        };
        let want = if m == 0 { 0 } else { n };
        if self.initial_seasonal.len() != want || self.initial_seasonal.iter().any(|s| s.len() != m)
        {
            no.refuse(if m == 0 {
                "holt: initial_seasonal is the season's start, but seasonal is not set. Leave \
                 initial_seasonal out."
                    .to_string()
            } else {
                format!(
                    "holt: the known start needs initial_seasonal for each of the {n} targets, \
                     {m} values each (seasonal_periods), phase 0 first. Give one list of {m} a \
                     target."
                )
            });
        }
        let all = self
            .initial_level
            .iter()
            .chain(&self.initial_trend)
            .chain(self.initial_seasonal.iter().flatten());
        if all.clone().any(|v| !v.is_finite()) {
            no.refuse(
                "holt: the known start's states must be numbers, but one is not finite. Give \
                 finite initial values.",
            );
        }
        if self.season == HoltSeason::Mul
            && (self.initial_level.iter().any(|&v| v <= 0.0)
                || self.initial_seasonal.iter().flatten().any(|&v| v <= 0.0))
        {
            no.refuse(
                "holt: a multiplicative season needs its level and every seasonal factor above \
                 0, but an initial one is 0 or below. Give positive initial values, or set \
                 seasonal to \"add\".",
            );
        }
    }
}

impl Holt {
    /// The state a fixed form starts in: the known start's states, set and
    /// seen, or the heuristic start's empty holds. The warm start is the
    /// state as built.
    pub(super) fn started(mut self) -> Self {
        if !self.cfg.fixed() {
            return self;
        }
        let n = self.cfg.n_targets;
        let m = if self.cfg.season == HoltSeason::None {
            0
        } else {
            self.cfg.season_periods
        };
        let (one, seasonal) = match self.cfg.season {
            HoltSeason::Mul => (1.0, true),
            HoltSeason::Add => (0.0, true),
            HoltSeason::None => (0.0, false),
        };
        self.season = if seasonal {
            vec![vec![one; m]; n]
        } else {
            Vec::new()
        };
        self.season_at = if seasonal {
            vec![vec![f64::NAN; m]; n]
        } else {
            Vec::new()
        };
        // The initial states stand one step before the first row.
        self.since = vec![self.cfg.first_step(); n];
        // The line fit is the warm start's; a fixed form keeps none.
        self.fit_weight.clear();
        self.fit_age.clear();
        self.fit_spread.clear();
        match self.cfg.start {
            HoltStart::Known => {
                self.level.clone_from(&self.cfg.initial_level);
                if self.cfg.trend {
                    self.trend.clone_from(&self.cfg.initial_trend);
                }
                if seasonal {
                    self.season.clone_from(&self.cfg.initial_seasonal);
                }
                self.seen = vec![true; n];
            }
            HoltStart::Heuristic => self.held = vec![Held::default(); n],
            HoltStart::Warm => {}
        }
        self
    }

    /// Whether the fixed forms' vectors have the config's shape: none at
    /// all under the warm start, else a season's `n × m` values and clocks
    /// and the heuristic start's `n` holds (empty once every start is fit).
    pub(super) fn fixed_shape_ok(&self) -> bool {
        let n = self.cfg.n_targets;
        if !self.cfg.fixed() {
            return self.season.is_empty() && self.season_at.is_empty() && self.held.is_empty();
        }
        let m = self.cfg.season_periods;
        let grid = |v: &Vec<Vec<f64>>| {
            if self.cfg.season == HoltSeason::None {
                v.is_empty()
            } else {
                v.len() == n && v.iter().all(|r| r.len() == m)
            }
        };
        let held_ok = self.held.iter().all(|h| {
            let k = h.y.len();
            [h.w.len(), h.step.len(), h.at.len(), h.phase.len()]
                .iter()
                .all(|&l| l == k)
                && k < self.cfg.start_rows()
                && h.phase.iter().all(|&p| (p as usize) < m.max(1))
        });
        grid(&self.season)
            && grid(&self.season_at)
            && (self.held.is_empty() || self.held.len() == n)
            && held_ok
    }

    /// Whether a target's heuristic start still holds rows, and so reports
    /// nothing: the model's own warm-up.
    pub fn starting(&self) -> bool {
        self.cfg.start == HoltStart::Heuristic && self.seen.iter().any(|s| !s)
    }

    /// Whether the start is the heuristic one, which holds each target's
    /// first rows ([`Self::starting`]).
    pub fn heuristic_start(&self) -> bool {
        self.cfg.start == HoltStart::Heuristic
    }

    /// Whether the last step left a row unlearned because a multiplicative
    /// form's target, or the level or seasonal factor its update would
    /// make, was 0 or below.
    pub fn not_positive(&self) -> bool {
        self.not_positive
    }

    /// Target `j`'s seasonal values, phase 0 first; empty without a season.
    pub fn seasonal(&self, j: usize) -> &[f64] {
        self.season.get(j).map_or(&[], Vec::as_slice)
    }

    /// The row's raw clock: the one handed, else the model's own clock
    /// `d` units on.
    fn row_clock(&self, d: f64) -> ClockValue {
        self.next_clock.unwrap_or(ClockValue::F64(self.clock + d))
    }

    /// The season's phase of a row at raw clock `c`: `⌊m·(c − c₀)/P⌋ mod m`,
    /// in integers on a temporal or an integer clock whose `P` is a whole
    /// number of its units, else in doubles. 0 before the origin is set
    /// (the first row, which sets it) and without a season.
    pub(crate) fn phase(&self, c: ClockValue) -> usize {
        let m = self.cfg.season_periods;
        if self.cfg.season == HoltSeason::None || m == 0 {
            return 0;
        }
        let Some(o) = self.origin else {
            return 0;
        };
        let p = self.cfg.period();
        let int = |c: i64, o: i64, per: f64| -> Option<usize> {
            // A whole period, in integers: exact at any clock's size.
            (per.fract() == 0.0 && per >= 1.0 && per < 2f64.powi(62)).then(|| {
                let num = (m as i128) * (i128::from(c) - i128::from(o));
                num.div_euclid(per as i128).rem_euclid(m as i128) as usize
            })
        };
        let float = |x: f64| -> usize {
            let k = (x * m as f64 / p).floor();
            let r = k.rem_euclid(m as f64);
            if r.is_finite() { r as usize % m } else { 0 }
        };
        match (c, o) {
            (ClockValue::Ns(c), ClockValue::Ns(o)) => int(c, o, p * 1e9)
                .unwrap_or_else(|| float((i128::from(c) - i128::from(o)) as f64 * 1e-9)),
            (ClockValue::I64(c), ClockValue::I64(o)) => {
                int(c, o, p).unwrap_or_else(|| float((i128::from(c) - i128::from(o)) as f64))
            }
            (c, o) => float(c.seconds() - o.seconds()),
        }
    }

    /// `φ_s = Σ_{i=1..s} φ^i = φ·(1 − φ^s)/(1 − φ)`, the steps of trend a
    /// damped trend adds over `s` clock units; `s` undamped. At `s = 1` it
    /// is `φ` to the bit: both factors are the same call.
    fn damped_steps(&self, s: f64) -> f64 {
        let h = self.cfg.damping_half_life;
        if h.is_infinite() {
            return s;
        }
        let phi = decay(1.0, h);
        phi * ((1.0 - decay(s, h)) / (1.0 - phi))
    }

    /// Target `j`'s level carried `s` clock units: `L = l + φ_s·b`.
    fn carried(&self, j: usize, s: f64) -> f64 {
        if self.cfg.trend {
            self.level[j] + self.trend[j] * self.damped_steps(s)
        } else {
            self.level[j]
        }
    }

    /// The forecast of target `j`, `s` clock units after its last
    /// observation, at phase `p`: `L ⊙ S[p]`.
    fn forecast(&self, j: usize, s: f64, p: usize) -> f64 {
        let l = self.carried(j, s);
        match self.cfg.season {
            HoltSeason::None => l,
            HoltSeason::Add => l + self.season[j][p],
            HoltSeason::Mul => l * self.season[j][p],
        }
    }

    /// The fixed forms' prediction at step `d`, each seen target's forecast.
    pub(super) fn fixed_predict(&self, d: f64, clock: Option<ClockValue>, pred: &mut [f64]) {
        let p = self.phase(clock.unwrap_or_else(|| self.row_clock(d)));
        for (j, out) in pred.iter_mut().enumerate() {
            if self.seen[j] {
                *out = self.forecast(j, self.since[j] + d, p);
            }
        }
    }

    /// The fixed forms' step, after the weight has been read as `holt`
    /// reads it (0 below the smallest normal double).
    pub(super) fn fixed_step(&mut self, y: &[Option<f64>], d: f64, weight: f64, pred: &mut [f64]) {
        let c = self.row_clock(d);
        self.next_clock = None;
        self.not_positive = false;
        let ready = self.w_sum >= self.cfg.min_weight;
        // The phase origin: the first row of positive weight is phase 0.
        if self.origin.is_none() && weight > 0.0 {
            self.origin = Some(c);
        }
        let p = self.phase(c);
        self.clock += d;
        let at = self.clock;
        let mul = self.cfg.season == HoltSeason::Mul;
        for (j, yj) in y.iter().enumerate() {
            let s = self.since[j] + d;
            if self.seen[j] && ready {
                pred[j] = self.forecast(j, s, p);
            }
            let yj = yj.filter(|v| v.is_finite() && weight > 0.0);
            let Some(yj) = yj else {
                self.since[j] = s;
                continue;
            };
            // A multiplicative season divides by the target and by the
            // level: a target of 0 or below is predicted and not learned.
            if mul && yj <= 0.0 {
                self.not_positive = true;
                self.since[j] = s;
                continue;
            }
            if !self.seen[j] {
                self.hold(j, yj, weight, s, at, p);
                continue;
            }
            if self.learn(j, yj, weight, s, at, p) {
                self.since[j] = 0.0;
            } else {
                self.not_positive = true;
                self.since[j] = s;
            }
        }
        self.w_sum = self.w_sum * decay(d, self.cfg.level_half_life) + weight;
    }

    /// One observation of target `j` learned: `y` at weight `w`, `s` clock
    /// units after its last, at decayed clock `at` and phase `p`. Returns
    /// false, changing nothing, where a multiplicative season's level or
    /// factor would not stay above 0.
    fn learn(&mut self, j: usize, y: f64, w: f64, s: f64, at: f64, p: usize) -> bool {
        let lam_l = decay(s, self.cfg.level_half_life);
        let g = fixed_gain(lam_l, w);
        let gc = 1.0 - g;
        let l_old = self.level[j];
        let big_l = self.carried(j, s);
        // statsmodels' order: `α·y` first, then the season's term, then
        // `(1 − α)·L`.
        let level = match self.cfg.season {
            HoltSeason::None => g * y + gc * big_l,
            HoltSeason::Add => g * y - g * self.season[j][p] + gc * big_l,
            HoltSeason::Mul => g * y / self.season[j][p] + gc * big_l,
        };
        let factor = match self.cfg.season {
            HoltSeason::None => f64::NAN,
            season => {
                let since = self.season_at[j][p];
                let gap = if since.is_nan() {
                    self.cfg.period()
                } else {
                    at - since
                };
                let gs = fixed_gain(decay(gap, self.cfg.seasonal_half_life), w);
                let gsc = 1.0 - gs;
                let sp = self.season[j][p];
                if season == HoltSeason::Add {
                    gs * y - gs * big_l + gsc * sp
                } else {
                    gs * y / big_l + gsc * sp
                }
            }
        };
        // Under a multiplicative season the level and the factor divide the
        // next rows: each must stay a positive number. A carried level of 0
        // or below makes the factor infinite or negative, so it is caught
        // here too.
        if self.cfg.season == HoltSeason::Mul
            && !(level > 0.0 && factor > 0.0 && level.is_finite() && factor.is_finite())
        {
            return false;
        }
        if self.cfg.trend {
            // The trend: a weighted mean of the trend carried over the step,
            // `φ^s·b`, and the slope the level moved at, `(l' − l)/s`. A move
            // over no clock has no slope: the trend is carried.
            let ri = 1.0 / s;
            let b = self.trend[j];
            let carried_b = if self.cfg.damping_half_life.is_infinite() {
                b
            } else {
                b * decay(s, self.cfg.damping_half_life)
            };
            self.trend[j] = if ri.is_finite() {
                let lam_b = decay(s, self.cfg.trend_half_life);
                let h = fixed_gain(lam_b, w);
                let slope = (level - l_old) * ri;
                let held_b = lam_b * self.w_trend[j];
                self.w_trend[j] = held_b + w;
                h * slope + (1.0 - h) * carried_b
            } else {
                carried_b
            };
        }
        self.level[j] = level;
        self.w_level[j] = lam_l * self.w_level[j] + w;
        if self.cfg.season != HoltSeason::None {
            self.season[j][p] = factor;
            self.season_at[j][p] = at;
        }
        true
    }

    /// Hold target `j`'s row for the heuristic start, and fit the start
    /// once it holds the rows the definition reads.
    fn hold(&mut self, j: usize, y: f64, w: f64, s: f64, at: f64, p: usize) {
        let held = &mut self.held[j];
        held.y.push(y);
        held.w.push(w);
        held.step.push(s);
        held.at.push(at);
        held.phase.push(p as u32);
        self.since[j] = 0.0;
        if held.y.len() >= self.cfg.start_rows() {
            self.fit_start(j);
        }
    }

    /// statsmodels' `_initialization_heuristic` on target `j`'s held rows,
    /// then those rows replayed through the recursion from the states it
    /// gives. A held row's seasonal mean is its phase's.
    fn fit_start(&mut self, j: usize) {
        let held = std::mem::take(&mut self.held[j]);
        let (level, trend, season) = heuristic(
            &held.y,
            &held.phase,
            self.cfg.season,
            self.cfg.season_periods,
        );
        // A multiplicative start the rows could not size (a moving average
        // of 0 or below cannot be, since every held value is above 0, but
        // a factor's mean of 0 can only come of an overflow) holds on.
        if !level.is_finite()
            || !trend.is_finite()
            || season.iter().any(|v| !v.is_finite())
            || (self.cfg.season == HoltSeason::Mul
                && (level <= 0.0 || season.iter().any(|&v| v <= 0.0)))
        {
            // Bounded: the oldest row goes, and the next row held fits
            // again on the rows after it.
            let mut held = held;
            held.y.remove(0);
            held.w.remove(0);
            held.step.remove(0);
            held.at.remove(0);
            held.phase.remove(0);
            self.held[j] = held;
            return;
        }
        self.level[j] = level;
        if self.cfg.trend {
            self.trend[j] = trend;
        }
        if self.cfg.season != HoltSeason::None {
            self.season[j] = season;
        }
        self.seen[j] = true;
        for i in 0..held.y.len() {
            // A replayed row the guard refuses is not learned, as it would
            // not have been had the start been known.
            if self.learn(
                j,
                held.y[i],
                held.w[i],
                held.step[i] + self.since[j],
                held.at[i],
                held.phase[i] as usize,
            ) {
                self.since[j] = 0.0;
            } else {
                self.since[j] += held.step[i];
            }
        }
    }
}

/// statsmodels' heuristic start (Hyndman and Athanasopoulos, *Forecasting:
/// principles and practice*, §2.6) on `y`: with a season, the centred
/// moving average over a season (`2×m` when `m` is even), the rows
/// detrended by it, their mean per phase normalized to a mean of 0 (add)
/// or 1 (mul), and the average's values in place of the rows; then the
/// least-squares line through the first 10 against `1..10`, its intercept
/// the level and its slope the trend. `(level, trend, seasonal)`.
pub(crate) fn heuristic(
    y: &[f64],
    phase: &[u32],
    season: HoltSeason,
    m: usize,
) -> (f64, f64, Vec<f64>) {
    let mut seasonal = Vec::new();
    let mut base: Vec<f64> = y.to_vec();
    if season != HoltSeason::None {
        let n = y.len();
        // `rolling(m, center=True).mean()`: the window at `i` reaches
        // `⌊m/2⌋` rows back; for an even `m`, `shift(-1).rolling(2)`
        // averages it with the next.
        let back = m / 2;
        let ma = |i: usize| -> f64 {
            if i < back || i + (m - 1 - back) >= n {
                return f64::NAN;
            }
            let lo = i - back;
            y[lo..lo + m].iter().sum::<f64>() / m as f64
        };
        let trend: Vec<f64> = (0..n)
            .map(|i| {
                if m.is_multiple_of(2) {
                    if i == 0 {
                        f64::NAN
                    } else {
                        (ma(i) + ma(i + 1)) / 2.0
                    }
                } else {
                    ma(i)
                }
            })
            .collect();
        let mut sums = vec![0.0; m];
        let mut counts = vec![0usize; m];
        for i in 0..n {
            if trend[i].is_nan() {
                continue;
            }
            let v = if season == HoltSeason::Add {
                y[i] - trend[i]
            } else {
                y[i] / trend[i]
            };
            let p = phase[i] as usize;
            sums[p] += v;
            counts[p] += 1;
        }
        seasonal = sums
            .iter()
            .zip(&counts)
            .map(|(s, &c)| s / c as f64)
            .collect();
        let mean = seasonal.iter().sum::<f64>() / m as f64;
        for v in &mut seasonal {
            if season == HoltSeason::Add {
                *v -= mean;
            } else {
                *v /= mean;
            }
        }
        base = trend.into_iter().filter(|v| !v.is_nan()).collect();
    }
    let ten = &base[..10.min(base.len())];
    let k = ten.len() as f64;
    let xbar = (k + 1.0) / 2.0;
    let ybar = ten.iter().sum::<f64>() / k;
    let sxx: f64 = (1..=ten.len()).map(|x| (x as f64 - xbar).powi(2)).sum();
    let sxy: f64 = ten
        .iter()
        .enumerate()
        .map(|(i, v)| (i as f64 + 1.0 - xbar) * v)
        .sum();
    let slope = sxy / sxx;
    (ybar - slope * xbar, slope, seasonal)
}
