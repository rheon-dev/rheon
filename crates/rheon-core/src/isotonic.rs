//! `Isotonic`: isotonic regression of each target on one feature, kept
//! online. The bank's `isotonic` kind: the monotone calibrator.
//!
//! For target `t`, the rows learned so far `(x_i, y_i, w_i)` (the window's,
//! under a window), each weight times its decay `d_i` (1 without a decay):
//!
//! ```text
//! f_t = argmin_f  sum_i w_i d_i (y_i - f(x_i))^2   subject to  f non-decreasing
//!                                                  (non-increasing if not increasing)
//! f_t <- clip(f_t, y_min, y_max)
//! pred_t(x) = f_t interpolated linearly between the distinct x it was fitted at,
//!             read before the row is learned (out of sample)
//! ```
//!
//! The fit is scikit-learn's `IsotonicRegression` with `sample_weight = w_i
//! d_i` on the same rows, computed as it computes it ([`fit`]): ties in `x`
//! pooled (`W = sum w`, value `(sum w y) / W`), pool-adjacent-violators
//! over the distinct `x` (scipy's), the clip, and `numpy.interp` between
//! the two neighbours of the row's `x`. Outside the range of the learned
//! `x` the prediction is NaN under [`OutOfBounds::Nan`] and the end value
//! under [`OutOfBounds::Clip`]; one distinct `x` predicts its value
//! everywhere; before any row, NaN. A row of weight 0 is no row.
//!
//! **The state.** Without a window, a target keeps its distinct `x`, each
//! with its pooled weight `W` and sum `S = sum w y`, ties added as they
//! arrive. Under a window (`window_rows`, the last rows of positive weight,
//! or `window_clock`, the rows of the last span of the clock under
//! `closed`) each row is held, sorted by `(x, y)` with the older first
//! among equals, and a row leaving takes itself out; the distinct `x` are
//! pooled at each fit, in scikit-learn's order. Each learned row refits
//! the targets it touched in full, `O(m)` for `m` distinct `x`.
//!
//! **A decay** (`Halflife` or `Lam`, with or without a window) weighs a row
//! learned at clock `T_i` by `d_i = decay.factor(T - T_i)` at clock `T`. The
//! fit does not move when every weight is scaled alike, so a row is kept
//! at `w_i / decay.factor(T_i - T0)`, from an origin `T0` that moves to the
//! row's clock (every kept weight scaled down with it) once that factor
//! passes `2^128`; a kept weight that falls below the smallest normal
//! double, a thousand half-lives on, is no weight. The factor is the
//! platform's `exp2` or `powf`, so the weights, and the fit, agree across
//! platforms within their last bits.
//!
//! **`n_eff`** (hard rule 8) is the weight before the row's update and its
//! decay, `w_sum <- f w_sum + w`, the window's under a window; each target
//! keeps its own over the rows it was present on.
//!
//! **The bound.** The distinct `x` without a window, and the rows inside a
//! `window_clock`, are the data's to number, so the bytes are checked as
//! the rows go in: the first row to take the state past half of
//! `max_state_bytes` is said, and a row that would take it past the cap is
//! not learned, nor any after it. A `window_rows` window's bytes are fixed
//! by the settings and checked at the call.

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

use crate::Stamp;
use crate::clock_span::{DEFAULT_STATE_BYTES, HeldCap, check_max_state_bytes};
use crate::model::{ModelState, OnlineModel, State, StateError, Step, check_schema};
use crate::{Decay, WindowClosed};

pub(crate) mod fit;

/// What the fit reads at an `x` beyond the range of the learned `x`:
/// scikit-learn's `out_of_bounds`, less `"raise"`, which a stream cannot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutOfBounds {
    /// NaN, scikit-learn's default: the rows do not reach the row's `x`.
    #[default]
    Nan,
    /// The value at the nearer end of the range.
    Clip,
}

/// The kept weight's scale past which the origin moves: `2^128`.
const REBASE_AT: f64 = 3.402_823_669_209_385e38;

/// The bytes a distinct `x` takes a target without a window: its `x`, its
/// weight and its sum, and its place in the fit (`x` and value).
pub const ISOTONIC_POINT_BYTES: u64 = 40;

/// The bytes a held row takes a target present under a window: its `x`,
/// `y` and weight, and its place in the fit.
pub const ISOTONIC_ROW_BYTES: u64 = 40;

/// The settings of an [`Isotonic`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IsotonicCfg {
    /// One fit per target column, each on the one feature.
    pub n_targets: usize,
    /// Non-decreasing (`true`, scikit-learn's default) or non-increasing.
    pub increasing: bool,
    /// The fit is clipped below at this value.
    pub y_min: Option<f64>,
    /// The fit is clipped above at this value.
    pub y_max: Option<f64>,
    /// What the fit reads beyond the learned `x`.
    pub out_of_bounds: OutOfBounds,
    /// A target's prediction is withheld while its weight is below this.
    pub min_weight: f64,
    /// Keep the last this many rows of positive weight alone.
    pub window_rows: Option<usize>,
    /// Keep the rows of the last this many clock units alone.
    pub window_clock: Option<f64>,
    /// Which edge of `window_clock` holds a row exactly that old.
    pub closed: WindowClosed,
    /// The decay on the model's clock: `Halflife(inf)` (none) by default.
    pub decay: Decay,
    /// The bytes a group's state may hold; `None` is 256 KB.
    pub max_state_bytes: Option<u64>,
}

impl IsotonicCfg {
    pub fn cap(&self) -> u64 {
        self.max_state_bytes.unwrap_or(DEFAULT_STATE_BYTES)
    }

    fn windowed(&self) -> bool {
        self.window_rows.is_some() || self.window_clock.is_some()
    }

    /// Whether the decay forgets at all: a finite half-life, or `lam`
    /// below 1.
    pub fn decays(&self) -> bool {
        match self.decay {
            Decay::Halflife(h) => h.is_finite(),
            Decay::Lam(l) => l < 1.0,
        }
    }

    /// The lower and upper clip, `-inf` and `inf` where none is given.
    fn clip(&self) -> (f64, f64) {
        (
            self.y_min.unwrap_or(f64::NEG_INFINITY),
            self.y_max.unwrap_or(f64::INFINITY),
        )
    }

    /// The bytes a ring row takes: its `x`, its weight, a `y` a target, and
    /// a stamp under a clock window.
    fn ring_row_bytes(&self) -> u64 {
        let stamp = if self.window_clock.is_some() {
            std::mem::size_of::<Stamp>() as u64
        } else {
            0
        };
        16 + 8 * self.n_targets as u64 + stamp
    }

    /// The bytes a held row takes at most: its ring row and an entry in
    /// every target.
    fn held_row_bytes(&self) -> u64 {
        self.ring_row_bytes() + self.n_targets as u64 * ISOTONIC_ROW_BYTES
    }

    /// The bytes the settings fix: the header, a target's own, and a row
    /// window at its fullest.
    pub fn fixed_bytes(&self) -> u64 {
        std::mem::size_of::<Isotonic>() as u64
            + self.n_targets as u64 * std::mem::size_of::<Target>() as u64
            + self
                .window_rows
                .map_or(0, |n| n as u64 * self.held_row_bytes())
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.n_targets == 0 {
            return Err(
                "isotonic: it needs at least one target, a column it fits as a monotone \
                 function of the feature. Name the column in targets."
                    .into(),
            );
        }
        for (name, v) in [("y_min", self.y_min), ("y_max", self.y_max)] {
            if let Some(v) = v
                && !crate::model::usable(v)
            {
                return Err(format!(
                    "isotonic: {name} must be a finite number, but it is {v}. Give the bound \
                     the fit is clipped to, or leave it out for none."
                ));
            }
        }
        if let (Some(lo), Some(hi)) = (self.y_min, self.y_max)
            && lo > hi
        {
            return Err(format!(
                "isotonic: y_min ({lo}) is above y_max ({hi}), so no value lies between them. \
                 Give y_min at most y_max."
            ));
        }
        crate::model::check_min_weight("isotonic", self.min_weight)?;
        self.decay.check().map_err(|e| format!("isotonic: {e}"))?;
        if self.window_rows.is_some() && self.window_clock.is_some() {
            return Err(
                "isotonic: window_rows and window_size are each a window, and only one can \
                 decide which rows are fitted. Give window_rows or window_size, not both."
                    .into(),
            );
        }
        if self.closed == WindowClosed::Both && self.window_clock.is_none() {
            return Err(
                "isotonic: closed = \"both\" names the edge of window_size, which is not \
                 given. Give window_size, or leave closed out."
                    .into(),
            );
        }
        if self.window_rows == Some(0) {
            return Err(
                "isotonic: window_rows must be at least 1, but it is 0. Give the number of \
                 rows the window keeps."
                    .into(),
            );
        }
        if let Some(d) = self.window_clock
            && !(d > 0.0 && d.is_finite())
        {
            return Err(format!(
                "isotonic: window_size must be above 0 and finite, but it is {d}. Give the \
                 span of the clock the window keeps."
            ));
        }
        check_max_state_bytes("isotonic", self.max_state_bytes)?;
        let (most, cap) = (self.fixed_bytes(), self.cap());
        if most > cap {
            return Err(format!(
                "isotonic: a group's {} fits and their window may hold {most} bytes, past \
                 max_state_bytes {cap}. Shorten window_rows, fit fewer targets in this spec, \
                 or raise max_state_bytes.",
                self.n_targets
            ));
        }
        Ok(())
    }
}

/// One target's rows and its fit.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
struct Target {
    /// Without a window, the distinct `x` in increasing order; under a
    /// window, each held row's `x`, the rows sorted by `(x, y)`, the older
    /// first among equals.
    x: Vec<f64>,
    /// Each one's kept weight (scaled from the origin under a decay).
    w: Vec<f64>,
    /// Without a window, each distinct `x`'s sum `S = sum w y` of kept
    /// weights; under a window, each row's `y`.
    v: Vec<f64>,
    /// The target's weight, `n_eff` over the rows it was present on.
    weight: f64,
    /// The fit: its distinct `x` and its values there. Never state: it is
    /// formed from the rows on every learned row and on a restore.
    #[serde(skip)]
    fx: Vec<f64>,
    #[serde(skip)]
    fy: Vec<f64>,
    /// Without a window, each distinct `x`'s value `S / W`, kept beside
    /// its sums so a refit divides only where a row landed; empty when it
    /// must be formed again (a restore, a move of the origin). Never state.
    #[serde(skip)]
    uy: Vec<f64>,
}

/// A window's rows, oldest first: each row's `x`, its kept weight, a `y` a
/// target (NaN where the target was absent), and under a clock window its
/// stamp.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
struct Ring {
    x: VecDeque<f64>,
    w: VecDeque<f64>,
    y: VecDeque<f64>,
    stamp: VecDeque<Stamp>,
}

/// Where the clock stands: the decay's clock and origin, and a clock
/// window's keys.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
struct Clock {
    /// The clock summed from every step.
    own: f64,
    /// The decay's origin on `own`: a row is kept at its weight over
    /// `decay.factor(own - origin)`.
    origin: f64,
    /// The rows that counted: the key on the row clock.
    counted: u64,
    /// The stamp of the last row stepped, whatever its weight.
    read: Option<Stamp>,
    /// The last stamp handed was the row clock's.
    on_rows: bool,
}

/// Scratch for a fit: the weights, scipy's block bounds, and which
/// targets a row touched.
#[derive(Debug, Clone, Default)]
struct Scratch {
    w: Vec<f64>,
    r: Vec<usize>,
    touched: Vec<bool>,
}

impl PartialEq for Scratch {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

/// Isotonic regression of each target on one feature (the module's
/// equations).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Isotonic {
    cfg: IsotonicCfg,
    targets: Vec<Target>,
    /// Every learned row's weight (the window's under a window): `n_eff`.
    w_sum: f64,
    ring: Option<Ring>,
    clock: Clock,
    /// The bytes a row would have taken the state to, past the cap: from
    /// then on no row is learned.
    over: Option<u64>,
    /// The half-of-the-cap warning has been raised (once a group).
    warned: bool,
    /// The next row's stamp, and whether it is the row clock's; a warning
    /// raised and not yet taken; scratch. Never state.
    #[serde(skip)]
    next: Option<(Stamp, bool)>,
    #[serde(skip)]
    warning: Option<u64>,
    #[serde(skip)]
    scratch: Scratch,
}

impl Isotonic {
    pub fn new(cfg: IsotonicCfg) -> Result<Self, String> {
        cfg.validate()?;
        Ok(Self {
            targets: vec![Target::default(); cfg.n_targets],
            w_sum: 0.0,
            ring: cfg.windowed().then(Ring::default),
            clock: Clock::default(),
            over: None,
            warned: false,
            next: None,
            warning: None,
            scratch: Scratch::default(),
            cfg,
        })
    }

    pub fn cfg(&self) -> &IsotonicCfg {
        &self.cfg
    }

    /// Every learned row's weight (the window's under a window): `n_eff`.
    pub fn n_eff(&self) -> f64 {
        self.w_sum
    }

    /// Target `j`'s weight over the rows it was present on.
    pub fn target_weight(&self, j: usize) -> f64 {
        self.targets[j].weight
    }

    /// Target `j`'s fit: its distinct `x` and the fitted values there.
    pub fn fit(&self, j: usize) -> (&[f64], &[f64]) {
        (&self.targets[j].fx, &self.targets[j].fy)
    }

    /// The distinct `x` (no window) or the rows (a window) target `j`
    /// holds.
    pub fn held(&self, j: usize) -> usize {
        self.targets[j].x.len()
    }

    /// The rows a window holds now; 0 without one.
    pub fn window_len(&self) -> usize {
        self.ring.as_ref().map_or(0, |r| r.x.len())
    }

    /// Whether the next row at `x` reads some target as beyond the rows
    /// that identify its fit: no row yet, or `x` outside the learned range
    /// under [`OutOfBounds::Nan`].
    pub fn beyond_sample(&self, x: f64) -> bool {
        self.targets.iter().any(|t| match t.fx.len() {
            0 => true,
            1 => false,
            n => self.cfg.out_of_bounds == OutOfBounds::Nan && (x < t.fx[0] || x > t.fx[n - 1]),
        })
    }

    /// The bytes the state holds now.
    pub fn bytes(&self) -> u64 {
        self.cfg.fixed_bytes() + self.held_bytes()
    }

    /// The bytes the data sizes: the distinct `x` without a window, the
    /// rows of a clock window. A row window's are fixed.
    fn held_bytes(&self) -> u64 {
        if self.cfg.window_rows.is_some() {
            return 0;
        }
        let entries: u64 = self.targets.iter().map(|t| t.x.len() as u64).sum();
        match &self.ring {
            Some(r) => r.x.len() as u64 * self.cfg.ring_row_bytes() + entries * ISOTONIC_ROW_BYTES,
            None => entries * ISOTONIC_POINT_BYTES,
        }
    }

    /// The bytes the state would hold with a row at `x` of targets `y`
    /// added.
    fn bytes_with(&self, x: f64, y: &[Option<f64>]) -> u64 {
        let present = || self.targets.iter().zip(y).filter(|(_, v)| v.is_some());
        let grow = match (&self.ring, self.cfg.window_rows) {
            (_, Some(_)) => 0,
            (Some(_), None) => {
                self.cfg.ring_row_bytes() + present().count() as u64 * ISOTONIC_ROW_BYTES
            }
            (None, None) => {
                present().filter(|(t, _)| !holds(&t.x, x)).count() as u64 * ISOTONIC_POINT_BYTES
            }
        };
        self.bytes() + grow
    }

    fn held_cap(&mut self) -> HeldCap<'_> {
        HeldCap {
            over: &mut self.over,
            warned: &mut self.warned,
            warning: &mut self.warning,
        }
    }

    /// Each target's fit read at `x`, withheld where its weight is short of
    /// `min_weight`.
    fn read_into(&self, x: f64, out: &mut Step) {
        self.read_into_at(x, self.cfg.min_weight, out);
    }

    /// Each target's fit read at `x`, withheld where its weight is short of
    /// `floor`.
    fn read_into_at(&self, x: f64, floor: f64, out: &mut Step) {
        out.pred.clear();
        out.pred.extend(self.targets.iter().map(|t| {
            if t.weight >= floor {
                fit::interp(&t.fx, &t.fy, x, self.cfg.out_of_bounds)
            } else {
                f64::NAN
            }
        }));
        out.n_eff = self.w_sum;
        out.extra = None;
    }

    /// The window's key for a row: on the row clock the count of the rows
    /// that counted (this one too, where `counting`), else the stamp
    /// handed, else the clock summed from the steps.
    fn window_key(&self, now: Option<(Stamp, bool)>, d_clock: f64, counting: bool) -> Stamp {
        match now {
            Some((_, true)) => Stamp::from((self.clock.counted + u64::from(counting)) as f64),
            Some((s, false)) => s,
            None => Stamp::from(self.clock.own + d_clock),
        }
    }

    /// A read ahead's key, for a row `d_clock` after the last stepped.
    fn key_ahead(&self, d_clock: f64) -> Stamp {
        match self.clock.read {
            _ if self.clock.on_rows => Stamp::from(self.clock.counted as f64 + 1.0),
            Some(last) => crate::since::back_by(last, -d_clock),
            None => Stamp::from(self.clock.own + d_clock),
        }
    }

    /// Where a row left a clock window's keys: nothing without one, so a
    /// row of weight 0 changes nothing there.
    fn note_clock(&mut self, handed: Option<(Stamp, bool)>, counting: bool) {
        if self.cfg.window_clock.is_none() {
            return;
        }
        self.clock.counted += u64::from(counting);
        if let Some((s, rows)) = handed {
            self.clock.read = Some(s);
            self.clock.on_rows = rows;
        }
    }

    /// The row's own step of `d_clock`: the weights decay by its factor and
    /// the clock moves on. Nothing but the clock without a decay.
    fn advance(&mut self, d_clock: f64) {
        self.clock.own += d_clock;
        if !self.cfg.decays() {
            return;
        }
        let f = self.cfg.decay.factor(d_clock);
        self.w_sum *= f;
        for t in &mut self.targets {
            t.weight *= f;
        }
    }

    /// The scale a row learned now is kept at: `1 / decay.factor(own -
    /// origin)`, after moving the origin to now where it would pass
    /// [`REBASE_AT`]; 1 without a decay.
    fn scale_now(&mut self) -> f64 {
        if !self.cfg.decays() {
            return 1.0;
        }
        let scale = self.cfg.decay.factor(self.clock.origin - self.clock.own);
        if scale <= REBASE_AT {
            return scale;
        }
        let down = self.cfg.decay.factor(self.clock.own - self.clock.origin);
        self.clock.origin = self.clock.own;
        let windowed = self.ring.is_some();
        for t in &mut self.targets {
            for w in &mut t.w {
                *w *= down;
            }
            if !windowed {
                t.uy.clear();
                for s in &mut t.v {
                    *s *= down;
                }
                // A weight past the smallest normal double is no weight.
                let mut i = 0;
                while i < t.x.len() {
                    if t.w[i] < f64::MIN_POSITIVE {
                        t.x.remove(i);
                        t.w.remove(i);
                        t.v.remove(i);
                    } else {
                        i += 1;
                    }
                }
            }
        }
        if let Some(r) = self.ring.as_mut() {
            for w in &mut r.w {
                *w *= down;
            }
        }
        1.0
    }

    /// Whether the oldest row of a clock window has left at `now`.
    fn oldest_has_left(&self, now: Stamp) -> bool {
        let Some(span) = self.cfg.window_clock else {
            return false;
        };
        self.ring
            .as_ref()
            .and_then(|r| r.stamp.front())
            .is_some_and(|&s| self.cfg.closed.has_left(now, s, span))
    }

    /// Take the oldest row out of the window and its targets, the weights
    /// read at the clock as it stands. Marks the targets it touched.
    fn pop_oldest(&mut self, touched: &mut [bool]) {
        let m = self.cfg.n_targets;
        let ring = self.ring.as_mut().expect("windowed");
        let Some(x) = ring.x.pop_front() else {
            return;
        };
        let kept = ring.w.pop_front().expect("a weight a row");
        ring.stamp.pop_front();
        let empty = ring.x.is_empty();
        let w = if self.cfg.decays() {
            kept * self.cfg.decay.factor(self.clock.own - self.clock.origin)
        } else {
            kept
        };
        for ((t, mark), &y) in self.targets.iter_mut().zip(touched.iter_mut()).zip(&ring.y) {
            if y.is_nan() {
                continue;
            }
            // The oldest of the rows equal to it in `(x, y)` is the first.
            let i = lower_bound(&t.x, &t.v, x, y);
            assert!(
                t.x.get(i) == Some(&x) && t.v.get(i) == Some(&y),
                "a held row is in its target's rows"
            );
            t.x.remove(i);
            t.w.remove(i);
            t.v.remove(i);
            t.weight = if t.x.is_empty() {
                0.0
            } else {
                (t.weight - w).max(0.0)
            };
            *mark = true;
        }
        ring.y.drain(..m);
        self.w_sum = if empty {
            0.0
        } else {
            (self.w_sum - w).max(0.0)
        };
    }

    /// Drop a clock window's rows that have left at `now`.
    fn trim(&mut self, now: Stamp, touched: &mut [bool]) {
        while self.oldest_has_left(now) {
            self.pop_oldest(touched);
        }
    }

    /// Fit the targets marked.
    fn refit(&mut self, touched: &[bool]) {
        let (lo, hi) = self.cfg.clip();
        let windowed = self.ring.is_some();
        for (t, _) in self.targets.iter_mut().zip(touched).filter(|(_, t)| **t) {
            t.fx.clear();
            t.fy.clear();
            let sw = &mut self.scratch.w;
            sw.clear();
            if windowed {
                fit::pool_rows(&t.x, &t.v, &t.w, &mut t.fx, &mut t.fy, sw);
            } else {
                if t.uy.len() != t.x.len() {
                    t.uy.clear();
                    t.uy.extend(t.v.iter().zip(&t.w).map(|(s, w)| s / w));
                }
                t.fx.extend_from_slice(&t.x);
                t.fy.extend_from_slice(&t.uy);
                sw.extend_from_slice(&t.w);
            }
            fit::isotonic(
                &mut t.fy,
                sw,
                &mut self.scratch.r,
                self.cfg.increasing,
                lo,
                hi,
            );
        }
    }

    /// The read for a row at `key` that learns nothing: from the window as
    /// it would stand at the row, gated on `floor`.
    fn read_ahead(&self, key: Stamp, x: f64, floor: f64, out: &mut Step) {
        if self.oldest_has_left(key) {
            let mut at = self.clone();
            let mut touched = vec![false; self.cfg.n_targets];
            at.trim(key, &mut touched);
            at.refit(&touched);
            at.read_into_at(x, floor, out);
            return;
        }
        self.read_into_at(x, floor, out);
    }

    fn key_at(&self, d_clock: f64, stamp: Stamp) -> Stamp {
        if self.cfg.window_clock.is_some() && !self.clock.on_rows {
            stamp
        } else {
            self.key_ahead(d_clock)
        }
    }

    /// [`OnlineModel::predict`] for a row whose place on the clock the
    /// caller knows, `stamp`, as [`OnlineModel::stamp_next`] hands a
    /// step's. Without a clock window (or on the row clock, whose keys
    /// count accepted rows) the stamp is not read.
    pub fn predict_at(&self, x: &[f64], d_clock: f64, stamp: Stamp) -> Step {
        self.predict_with_floor_at(x, d_clock, stamp, self.cfg.min_weight)
    }

    /// [`OnlineModel::predict_with_floor`] for a row whose place on the
    /// clock the caller knows, `stamp`.
    pub fn predict_with_floor_at(&self, x: &[f64], d_clock: f64, stamp: Stamp, floor: f64) -> Step {
        if let Some(refused) = crate::model::refused_predict(self, x, d_clock) {
            return refused;
        }
        let mut out = Step::default();
        let key = self.key_at(d_clock, stamp);
        self.read_ahead(key, x[0], floor, &mut out);
        out
    }

    /// Whether trimming a clock window at `stamp` (or `d_clock` ahead on
    /// the row clock) can take any target's weight below `top`.
    pub fn trims_below(&self, d_clock: f64, stamp: Option<Stamp>, top: f64) -> bool {
        let (Some(span), Some(ring)) = (self.cfg.window_clock, &self.ring) else {
            return false;
        };
        let key = match stamp {
            Some(s) => self.key_at(d_clock, s),
            None => self.key_ahead(d_clock),
        };
        let scale = if self.cfg.decays() {
            self.cfg.decay.factor(self.clock.own - self.clock.origin)
        } else {
            1.0
        };
        let mut drop_w = 0.0;
        for (&s, &kept) in ring.stamp.iter().zip(&ring.w) {
            if !self.cfg.closed.has_left(key, s, span) {
                break;
            }
            drop_w += kept * scale;
        }
        drop_w > 0.0 && self.targets.iter().any(|t| t.weight - drop_w < top)
    }

    /// Whether a restored state has the shape [`Self::new`] gives it.
    fn has_shape(&self) -> bool {
        let m = self.cfg.n_targets;
        let windowed = self.cfg.windowed();
        let finite_pos = |w: &f64| w.is_finite() && *w >= 0.0;
        let targets = self.targets.len() == m
            && self.targets.iter().all(|t| {
                let n = t.x.len();
                t.w.len() == n
                    && t.v.len() == n
                    && finite_pos(&t.weight)
                    && t.w.iter().all(finite_pos)
                    && t.x.iter().chain(&t.v).all(|v| v.is_finite())
                    && if windowed {
                        (1..n).all(|i| (t.x[i - 1], t.v[i - 1]) <= (t.x[i], t.v[i]))
                    } else {
                        (1..n).all(|i| t.x[i - 1] < t.x[i])
                    }
            });
        let ring = match &self.ring {
            None => !windowed,
            Some(r) => {
                let n = r.x.len();
                let held = |j: usize| (0..n).filter(|&i| !r.y[i * m + j].is_nan()).count();
                windowed
                    && self.targets.len() == m
                    && r.w.len() == n
                    && r.y.len() == n * m
                    && r.stamp.len()
                        == if self.cfg.window_clock.is_some() {
                            n
                        } else {
                            0
                        }
                    && self.cfg.window_rows.is_none_or(|cap| n <= cap)
                    && r.x.iter().all(|v| v.is_finite())
                    && r.w.iter().all(finite_pos)
                    && r.y.iter().all(|v| v.is_nan() || v.is_finite())
                    && (0..m).all(|j| held(j) == self.targets[j].x.len())
                    && (0..m).all(|j| {
                        // Each target's rows are the ring's, sorted by `(x,
                        // y)` with the older first among equals.
                        let mut rows: Vec<(f64, f64, f64)> = (0..n)
                            .filter(|&i| !r.y[i * m + j].is_nan())
                            .map(|i| (r.x[i], r.y[i * m + j], r.w[i]))
                            .collect();
                        rows.sort_by(|a, b| {
                            (a.0, a.1)
                                .partial_cmp(&(b.0, b.1))
                                .unwrap_or(std::cmp::Ordering::Equal)
                        });
                        let t = &self.targets[j];
                        rows.iter().enumerate().all(|(i, &(x, y, w))| {
                            t.x[i].to_bits() == x.to_bits()
                                && t.v[i].to_bits() == y.to_bits()
                                && t.w[i].to_bits() == w.to_bits()
                        })
                    })
            }
        };
        targets
            && ring
            && finite_pos(&self.w_sum)
            && self.clock.own.is_finite()
            && self.clock.origin.is_finite()
    }
}

/// Whether the distinct `x` (increasing) hold one equal to `x`.
fn holds(xs: &[f64], x: f64) -> bool {
    xs.get(xs.partition_point(|p| *p < x)) == Some(&x)
}

/// The first place in rows sorted by `(x, y)` that is not below `(x, y)`.
fn lower_bound(xs: &[f64], ys: &[f64], x: f64, y: f64) -> usize {
    let (mut lo, mut hi) = (0, xs.len());
    while lo < hi {
        let mid = (lo + hi) / 2;
        if (xs[mid], ys[mid]) < (x, y) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

/// The first place in rows sorted by `(x, y)` that is above `(x, y)`: a
/// new row goes after the rows equal to it, so the older stays first.
fn upper_bound(xs: &[f64], ys: &[f64], x: f64, y: f64) -> usize {
    let (mut lo, mut hi) = (0, xs.len());
    while lo < hi {
        let mid = (lo + hi) / 2;
        if (xs[mid], ys[mid]) <= (x, y) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

impl OnlineModel for Isotonic {
    fn step_into(
        &mut self,
        x: &[f64],
        y: &[Option<f64>],
        d_clock: f64,
        weight: f64,
        out: &mut Step,
    ) {
        // A value or a weight that is not usable, by the rule every model
        // keeps: such a target is absent, and such a feature or weight
        // learns nothing.
        if let Some(refused) = crate::model::refused_step(self, x, y, d_clock, weight) {
            *out = refused;
            return;
        }
        let x0 = x[0];
        let handed = self.next.take();
        if self.over.is_some() {
            let key = self.window_key(handed, d_clock, weight >= f64::MIN_POSITIVE);
            self.read_ahead(key, x0, out);
            return;
        }
        // A weight below the smallest normal double is a weight of 0: what
        // it would teach is below a double's resolution beside any normal
        // weight, and its `y w` would round in the subnormal range.
        if weight < f64::MIN_POSITIVE {
            // Counts in no clock (hard rule 9): read from the window as it
            // would stand at the row, and nothing kept but where it stood.
            // Handed to the model directly, under a decay it decays the
            // weights, which advances its clock and learns nothing.
            let key = self.window_key(handed, d_clock, false);
            self.read_ahead(key, x0, self.cfg.min_weight, out);
            self.note_clock(handed, false);
            self.advance(d_clock);
            return;
        }
        let m = self.cfg.n_targets;
        let key = self.window_key(handed, d_clock, true);
        let cap = self.cfg.cap();
        if self.cfg.window_rows.is_none() {
            let drop_bytes = match (&self.ring, self.cfg.window_clock) {
                (Some(r), Some(span)) => {
                    let drop_rows = r
                        .stamp
                        .iter()
                        .take_while(|&&s| self.cfg.closed.has_left(key, s, span))
                        .count();
                    let drop_entries =
                        r.y.iter()
                            .take(drop_rows * m)
                            .filter(|v| !v.is_nan())
                            .count();
                    drop_rows as u64 * self.cfg.ring_row_bytes()
                        + drop_entries as u64 * ISOTONIC_ROW_BYTES
                }
                _ => 0,
            };
            let bytes = self.bytes_with(x0, y) - drop_bytes;
            if !self.held_cap().admit(bytes, cap) {
                self.read_ahead(key, x0, out);
                return;
            }
        }
        let mut touched = std::mem::take(&mut self.scratch.touched);
        touched.clear();
        touched.resize(m, false);
        self.note_clock(handed, true);
        // The rows out of a clock window by now leave before the row is
        // read, and `n_eff` is the weight of the rows it reads (hard rule 8).
        self.trim(key, &mut touched);
        if touched.iter().any(|t| *t) {
            self.refit(&touched);
            touched.fill(false);
        }
        self.read_into(x0, out);
        // The row's own decay, after its read (hard rule 8: `n_eff` before
        // it), then the row.
        self.advance(d_clock);
        let scale = self.scale_now();
        let kept = weight * scale;
        let windowed = self.ring.is_some();
        for (j, yj) in y.iter().enumerate().take(m) {
            let Some(yj) = *yj else { continue };
            let t = &mut self.targets[j];
            if windowed {
                let i = upper_bound(&t.x, &t.v, x0, yj);
                t.x.insert(i, x0);
                t.w.insert(i, kept);
                t.v.insert(i, yj);
            } else {
                let i = t.x.partition_point(|p| *p < x0);
                match t.x.get(i) {
                    Some(&p) if p == x0 => {
                        t.w[i] += kept;
                        t.v[i] += yj * kept;
                        if let Some(u) = t.uy.get_mut(i) {
                            *u = t.v[i] / t.w[i];
                        }
                    }
                    _ => {
                        t.x.insert(i, x0);
                        t.w.insert(i, 0.0 + kept);
                        t.v.insert(i, 0.0 + yj * kept);
                        if t.uy.len() + 1 == t.x.len() {
                            t.uy.insert(i, t.v[i] / t.w[i]);
                        }
                    }
                }
            }
            t.weight += weight;
            touched[j] = true;
        }
        self.w_sum += weight;
        if let Some(ring) = self.ring.as_mut() {
            ring.x.push_back(x0);
            ring.w.push_back(kept);
            ring.y
                .extend((0..m).map(|j| y.get(j).copied().flatten().unwrap_or(f64::NAN)));
            if self.cfg.window_clock.is_some() {
                ring.stamp.push_back(key);
            }
            if self.cfg.window_rows.is_some_and(|cap| ring.x.len() > cap) {
                self.pop_oldest(&mut touched);
            }
        }
        self.refit(&touched);
        self.scratch.touched = touched;
        if self.cfg.window_rows.is_none() {
            let bytes = self.bytes();
            self.held_cap().note(bytes, cap);
        }
    }

    fn predict(&self, x: &[f64], d_clock: f64) -> Step {
        if let Some(refused) = crate::model::refused_predict(self, x, d_clock) {
            return refused;
        }
        let mut out = Step::default();
        let key = self.key_ahead(d_clock);
        self.read_ahead(key, x[0], self.cfg.min_weight, &mut out);
        out
    }

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

    fn stamp_next(&mut self, stamp: Stamp) {
        self.next = Some((stamp, false));
    }

    fn stamp_next_on_rows(&mut self, stamp: Stamp) {
        self.next = Some((stamp, true));
    }

    fn take_state_warning(&mut self) -> Option<(u64, u64)> {
        self.warning.take().map(|bytes| (bytes, self.cfg.cap()))
    }

    fn state_over_cap(&self) -> Option<(u64, u64)> {
        self.over.map(|bytes| (bytes, self.cfg.cap()))
    }

    fn target_n_eff_into(&self, out: &mut Vec<f64>) -> bool {
        out.clear();
        out.extend(self.targets.iter().map(|t| t.weight));
        true
    }

    fn state(&self) -> State {
        State::new(ModelState::Isotonic(Box::new(self.clone())))
    }

    fn restore(s: &State) -> Result<Self, StateError> {
        check_schema(s)?;
        match &s.model {
            ModelState::Isotonic(m) => {
                let mut m = (**m).clone();
                crate::model::check_cfg("isotonic", m.cfg.validate())?;
                if !m.has_shape() {
                    return Err(crate::model::damaged(
                        "isotonic",
                        "its rows have the wrong shape",
                    ));
                }
                let all = vec![true; m.cfg.n_targets];
                m.refit(&all);
                Ok(m)
            }
            other => Err(StateError::WrongModel {
                expected: "isotonic",
                found: other.kind(),
            }),
        }
    }

    fn n_targets(&self) -> usize {
        self.cfg.n_targets
    }

    /// The one feature each target is fitted on.
    fn n_features(&self) -> usize {
        1
    }
}

#[cfg(test)]
#[path = "isotonic/tests.rs"]
mod tests;
