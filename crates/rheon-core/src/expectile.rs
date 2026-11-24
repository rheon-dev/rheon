//! Expectile regression by recursive reweighted least squares (docs/PLAN.md
//! task 501; the study: docs/records/STUDY-501-509.md, Part A).
//!
//! **The quantity.** At level `τ` in `(0, 1)` the expectile regression
//! minimises the asymmetric squared loss
//!
//! ```text
//! L(β) = Σ_i v_i a(r_i) r_i²,    r_i = y_i − x̃_i'β,
//! a(r) = τ      if r > 0
//!      = 1 − τ  if r ≤ 0
//! ```
//!
//! (Newey and Powell 1987), `x̃ = [1, x]` (`x` alone without an intercept),
//! `v_i` the row's weight decayed to now. `L` is convex, piecewise
//! quadratic and C1, so for a fixed assignment of sides its minimiser is a
//! weighted least-squares fit, and the minimiser of `L` is the WLS fit whose
//! sides agree with its own residuals: `Σ v a(r) r x̃ = 0`. At `τ = 1/2` it
//! is the weighted least-squares fit; the loss's curvature lies between
//! `min(τ, 1 − τ)` and `max(τ, 1 − τ)` times least squares' everywhere, so
//! no smoothing band and no residual scale are needed, as `quantile` needs.
//!
//! **One pass** (the default). Each target keeps a mean-form Gram `G`, a
//! cross-moment `c`, the asymmetric weight `S` and its coefficients `β`.
//! Row `t`, of weight `w` and clock step `Δ`, `λ = 2^(−Δ/half_life)`:
//!
//! ```text
//! pred = x̃'β                    r = y − pred       (out of sample: before the update)
//! a    = a(r), or 1/2 before the target's first fit
//! S'   = λ S + w a
//! f    = w a / S'               (no update where w a = 0)
//! G'   = G + f (x̃x̃' − G)
//! c'   = c + f (x̃ y − c)
//! W'   = λ W + w                (the raw weight of the target's rows; n_eff is every row's)
//! β'   = (G' + r (W'/S') D)⁻¹ c'
//! ```
//!
//! `D` is the slopes' diagonal (the intercept unpenalized) and `r` the
//! `ridge`, so `β` minimises `Σ v a(r) r² + r W |β_s|²` at the sides judged:
//! a penalty on the mean-form Gram, in the features' squared units, which
//! keeps every solve defined from the first row, a constant or duplicated
//! feature included. With `r = 0` the fit is the exact loss's, and waits for
//! `p` rows of positive weight and a pivot of `G` above 1e-12 of its largest
//! diagonal. The refits minimise the same penalized loss, `W` the weight of
//! the rows they hold.
//!
//! A row's own update cannot move its residual across zero: `β' − β = w a
//! (S'G')⁻¹ x̃ r`, so `r_after = r (1 − w a h)`, `h = x̃'(S'G')⁻¹x̃`, and `0 ≤ w
//! a h ≤ 1` because `S'G'` holds `w a x̃x̃'`. Judging the side after the row
//! picks the same side, so one solve a row is the whole step (the study's
//! table R2: 0 of 29,960 rows differ). What one pass cannot do is re-judge
//! an earlier row whose side the fit has since crossed: at `τ = 1/2` it is
//! the decayed WLS to rounding, and elsewhere a statistical approximation
//! whose fitted values stand within the batch fit's own sampling error of
//! the exact decayed fit (the study's R3: with decay the mean gap is 0.02 to
//! 0.4 of `√(p / n_kish)` residual scales, the largest 1.2).
//!
//! **The first rows refit** (`refit_rows`). Without decay the rows judged
//! on the wrong side early stay at full weight. Each target's first
//! `refit_rows` learned rows are kept, and on the row that fills the
//! buffer, after its update, they are fitted exactly (below) at their
//! decayed weights, warm-started at the one-pass fit; the sums are rebuilt
//! with each row's side judged at the converged fit, `S = Σ v a`, `G = Σ v a
//! x̃x̃' / S`, `c = Σ v a x̃ y / S`, and the buffer freed. The study measured
//! a fivefold cut in the gap at 500 rows without decay (0.102 to 0.020 of
//! the yardstick at `τ = 0.9`). `refit_clock = D` is the same refit over a
//! span of the clock (docs/PLAN.md task 538): a target's rows from its first
//! learned row through the first at least `D` clock units after it, once
//! they hold a row per coefficient, `D` read off the rows' exact stamps
//! ([`crate::clock_span::span_reached`]); on the row clock, `D + 1` rows.
//! How many rows that is the data decides, so they are checked against
//! `max_state_bytes` as they come, as a clock window's are.
//!
//! **The exact refit**, of a buffer or a window: start from the sides at
//! the fit before (warm) or from least squares (cold); a pass is one WLS
//! solve at the current sides, `(Σ v a x̃x̃' + r V D) β = Σ v a x̃ y`, then every side is judged again. A step that
//! does not lower `L` is halved (the loss is convex, so a halving always
//! helps). It stops when no side changes after a full step, which is the
//! exact minimiser; when the fitted values moved by at most `max(1e-12 s, 8
//! ε max|fitted|)`, `s` the weighted residual RMS, since rows at rounding
//! level trade sides with no effect on the fit and an exact fit has `s =
//! 0`; when a side pattern comes back with no lower loss than it had, the
//! lowest loss met kept; or after [`EXPECTILE_REFIT_PASSES`] passes, which
//! counts in `solve_failures`, the lowest-loss fit kept. At `τ = 1/2` one
//! solve is the answer.
//!
//! **A window** (`window_rows`, or `window_clock`): the fit is the exact
//! expectile regression of the rows in it alone, refit on every row
//! warm-started at the fit before, so the prediction for a row is the fit of
//! the window before it. A window forgets by dropping, so it takes no
//! decay. `window_rows = N` keeps the last `N` rows of positive weight;
//! `window_clock = D` the rows of the last `D` clock units, decided on the
//! rows' exact stamps ([`crate::Stamp::cmp_span`]) under `closed`: `Right`,
//! the default, drops a row exactly `D` old, `Both` keeps it. A row leaves
//! at the first row at or after the one it ages out on, before that row's
//! prediction, and the fit of what is left is refit then. Under a window,
//! `n_eff` is the weight of the rows in it (hard rule 8), read before the
//! row's own clock step, as weight a decay takes at the row is still
//! undecayed in it.
//!
//! **A row of weight 0** learns nothing and counts in no clock (hard rule
//! 9): its prediction is read as any row's, and the next row that counts
//! steps as if it were not there, its gap taken once from the last row that
//! counted -- the two rows' stamps where the caller hands them
//! ([`crate::OnlineModel::stamp_next`]), the step itself on the row clock,
//! where a row of weight 0 is no step, and the steps summed only for a
//! caller that hands no stamps. It is in no window and none of `n_eff`.
//!
//! **The state.** Per target, `p(p+1)/2 + 2p + 4` numbers (`G` packed, `c`,
//! `β`, `S`, the target's weight, its row count and whether it has a fit),
//! and the shared weight and clock; under a window only `β` and the flags,
//! the fit being the window's. The parts the data could size are bounded at
//! the call: a refit buffer of `refit_rows (k + 2)` numbers a target and a
//! row window of `window_rows (k + m + 1)` numbers (`k` features, `m`
//! targets) beside one flag a target a row, checked against
//! `max_state_bytes` as the model is built. A clock window holds the rows
//! its span has, which the data decides: it is checked as it runs, a warning
//! at half of `max_state_bytes` and the row that would pass it not learned,
//! nor any after it ([`OnlineModel::state_over_cap`]).

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

use crate::model::{ModelState, OnlineModel, State, StateError, Step, check_schema};
use crate::since::back_by;
use crate::solve::dot_aug;
use crate::{Decay, RefitStatus, Stamp, WindowClosed};

#[cfg(test)]
mod mutant_tests;
#[cfg(test)]
mod refit_clock_tests;
#[cfg(test)]
mod review_tests;
#[cfg(test)]
mod tests;

/// The most passes an exact refit takes; one that takes them all counts in
/// `solve_failures`. The study's refits took at most 10 on awkward
/// problems (duplicated rows) and 5 on ordinary ones.
pub const EXPECTILE_REFIT_PASSES: u32 = 50;

/// The least of a spec's default `refit_rows` ([`default_expectile_refit_rows`]):
/// the count the study measured.
pub const DEFAULT_EXPECTILE_REFIT_ROWS: u32 = 200;

/// The bytes a group's state may hold by default, a spec's
/// `max_state_bytes`: 256 KB.
pub const EXPECTILE_MAX_STATE_BYTES: u64 = 256 * 1024;

/// The most `max_state_bytes` may be set to: 64 MiB a group.
pub const EXPECTILE_STATE_BYTES_CEILING: u64 = 64 * 1024 * 1024;

/// A pivot of `G` below this share of its largest diagonal is no fit.
const PIVOT_FLOOR: f64 = 1e-12;

/// The refit's relative stop on the fitted values, in residual scales.
const REFIT_TOL: f64 = 1e-12;

/// A spec's default `refit_rows`: `max(200, 10 k)`, `k` the coefficients,
/// held to what `cap` bytes hold beside the sums; 0 -- off -- past
/// [`crate::DEFAULT_REFIT_MAX_FEATURES`] features, under a window, or where
/// the cap holds fewer than `2 k` rows.
pub fn default_expectile_refit_rows(
    n_features: usize,
    n_targets: usize,
    fit_intercept: bool,
    windowed: bool,
    cap: u64,
) -> u32 {
    if windowed || n_features > crate::DEFAULT_REFIT_MAX_FEATURES {
        return 0;
    }
    let k = (n_features + usize::from(fit_intercept)) as u64;
    let fixed = Expectile::sums_bytes(n_features, n_targets, fit_intercept, false);
    let row = 8 * n_targets.max(1) as u64 * (n_features as u64 + 2);
    let fits = cap.saturating_sub(fixed) / row;
    let rows = (u64::from(DEFAULT_EXPECTILE_REFIT_ROWS).max(10 * k)).min(fits);
    if rows < 2 * k.max(1) {
        0
    } else {
        u32::try_from(rows).unwrap_or(u32::MAX)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExpectileCfg {
    pub n_features: usize,
    pub n_targets: usize,
    pub fit_intercept: bool,
    pub decay: Decay,
    /// `τ`, the level, in `(0, 1)`.
    pub expectile: f64,
    /// `r`, the penalty `r V |β_s|²` on the slopes, `V` the rows' weight
    /// (on the mean-form Gram, in the features' squared units): at least 0.
    pub ridge: f64,
    #[serde(with = "crate::humanfloat::f64_or_tag")]
    pub min_weight: f64,
    /// Each target's first learned rows fitted exactly, once; 0 is off.
    pub refit_rows: u32,
    /// Fit the last this many rows of positive weight alone.
    #[serde(default)]
    pub window_rows: Option<usize>,
    /// Fit the rows of the last this many clock units alone (seconds on a
    /// temporal clock).
    #[serde(default)]
    pub window_clock: Option<f64>,
    /// Which edge of a clock window keeps the row exactly its span old.
    #[serde(default)]
    pub closed: WindowClosed,
    /// The bytes a group's state may hold; the default is
    /// [`EXPECTILE_MAX_STATE_BYTES`].
    #[serde(default)]
    pub max_state_bytes: Option<u64>,
    /// The refit over a span of the clock in place of a count of rows:
    /// each target's rows from its first learned row through the first at
    /// least this many clock units after it (seconds on a temporal clock),
    /// once they hold a row per coefficient. `refit_rows` is 0 beside it.
    /// Last, and written only when set, so a state without it keeps its
    /// bytes and the array form its order.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub refit_clock: Option<f64>,
}

impl ExpectileCfg {
    pub fn k_total(&self) -> usize {
        self.n_features + usize::from(self.fit_intercept)
    }

    /// The bytes a group's state may hold.
    pub fn cap(&self) -> u64 {
        self.max_state_bytes.unwrap_or(EXPECTILE_MAX_STATE_BYTES)
    }

    fn windowed(&self) -> bool {
        self.window_rows.is_some() || self.window_clock.is_some()
    }

    /// Whether the first rows are refit, after a count or a span.
    fn refits(&self) -> bool {
        self.refit_rows > 0 || self.refit_clock.is_some()
    }

    /// The bytes the settings fix: the sums, a refit buffer at its full
    /// size and a row window full; a clock window's rows are the data's
    /// ([`Expectile::clock_state_bytes`]).
    pub fn fixed_bytes(&self) -> u64 {
        let (k, m) = (self.n_features, self.n_targets);
        Expectile::sums_bytes(k, m, self.fit_intercept, self.windowed())
            + 8 * m as u64 * u64::from(self.refit_rows) * (k as u64 + 2)
            + self
                .window_rows
                .map_or(0, |n| n as u64 * Expectile::window_row_bytes(k, m, false))
    }

    pub fn validate(&self) -> Result<(), String> {
        let mut no = crate::Refusals::new();
        no.take(self.decay.check().map_err(|e| format!("expectile: {e}")));
        no.take(crate::model::check_dims(
            "expectile",
            self.n_features,
            self.n_targets,
        ));
        no.take(crate::model::check_min_weight("expectile", self.min_weight));
        if !(self.expectile > 0.0 && self.expectile < 1.0) {
            no.refuse(format!(
                "expectile: expectile, the level tau, must be above 0 and below 1, but it is \
                 {}. Give the level, such as 0.9 for an upper expectile or 0.5 for the \
                 mean.",
                self.expectile
            ));
        }
        let p = self.k_total();
        if !(self.ridge >= 0.0 && self.ridge.is_finite()) {
            no.refuse(format!(
                "expectile: ridge must be finite and at least 0, but it is {}. Set ridge to a \
                 small number, such as the default 1e-6, in the features' squared units, or \
                 to 0 for the exact fit.",
                self.ridge
            ));
        }
        if let Some(n) = self.window_rows
            && n < p.max(1)
        {
            no.refuse(format!(
                "expectile: window_rows must hold at least one row per coefficient, {p}, but \
                 it is {n}: a window of fewer rows never determines the fit. Raise \
                 window_rows, or leave it out for no window."
            ));
        }
        if let Some(d) = self.refit_clock {
            if !(d.is_finite() && d > 0.0) {
                no.refuse(format!(
                    "expectile: refit_span must be a finite number of clock units above 0, but \
                     it is {d}. Give the span of clock whose rows the refit fits, or leave it \
                     out for refit_rows."
                ));
            }
            if self.refit_rows > 0 {
                no.refuse(format!(
                    "expectile: refit_span and refit_rows = {} are two refits of the first \
                     rows, and a target is refit once. Leave one of them out.",
                    self.refit_rows
                ));
            }
        }
        if let Some(d) = self.window_clock {
            if !(d.is_finite() && d > 0.0) {
                no.refuse(format!(
                    "expectile: the clock window must be a finite number of clock units above \
                     0, but it is {d}. Give the span of clock the fit keeps rows of, or leave \
                     it out for no window."
                ));
            }
            if self.window_rows.is_some() {
                no.refuse(
                    "expectile: a window of clock units and window_rows are two windows, and \
                     the fit holds one. Leave one of them out.",
                );
            }
        } else if self.closed != WindowClosed::Right {
            no.refuse(
                "expectile: closed = \"both\" says which edge of a clock window keeps the row \
                 exactly its span old, but there is no clock window. Give the window's span, \
                 or leave closed out.",
            );
        }
        if self.windowed() {
            if self.decay.check().is_ok() && self.decay.factor(1.0) != 1.0 {
                no.refuse(
                    "expectile: a window forgets by dropping its oldest rows, so it takes no \
                     decay, but half_life (or lam) decays too. Set half_life to inf, or leave \
                     out the window.",
                );
            }
            if self.refit_rows > 0 {
                no.refuse(format!(
                    "expectile: a window refits its rows exactly on every row, so the first \
                     rows' refit has nothing to add, but refit_rows is {}. Set refit_rows to \
                     0, or leave out the window.",
                    self.refit_rows
                ));
            }
            if self.refit_clock.is_some() {
                no.refuse(
                    "expectile: a window refits its rows exactly on every row, so the first \
                     rows' refit has nothing to add, but refit_span is set. Leave out \
                     refit_span, or the window.",
                );
            }
        }
        match self.max_state_bytes {
            Some(0) => no.refuse(
                "expectile: max_state_bytes must be above 0, but it is 0. Give the bytes a \
                 group's state may hold, or leave it out for 256 KB.",
            ),
            Some(c) if c > EXPECTILE_STATE_BYTES_CEILING => no.refuse(format!(
                "expectile: max_state_bytes may be at most {EXPECTILE_STATE_BYTES_CEILING} (64 \
                 MiB) a group, but it is {c}. Lower it, or keep fewer rows: a group's state is \
                 held in memory beside every other group's."
            )),
            _ => {
                let (bytes, cap) = (self.fixed_bytes(), self.cap());
                if self.n_targets > 0 && bytes > cap {
                    no.refuse(format!(
                        "expectile: its state would hold {bytes} bytes a group, past \
                         max_state_bytes of {cap}: {}. Lower {}, or raise max_state_bytes.",
                        self.bytes_said(),
                        if self.window_rows.is_some() {
                            "window_rows"
                        } else {
                            "refit_rows"
                        }
                    ));
                }
            }
        }
        no.finish()
    }

    /// Where the fixed bytes go, for a refusal.
    fn bytes_said(&self) -> String {
        let (k, m) = (self.n_features, self.n_targets);
        let mut s = format!(
            "{} bytes of sums for {}",
            Expectile::sums_bytes(k, m, self.fit_intercept, self.windowed()),
            crate::model::count(m, "target")
        );
        if self.refit_rows > 0 {
            s.push_str(&format!(
                ", and a refit buffer of {} rows of {} numbers for each",
                self.refit_rows,
                k + 2
            ));
        }
        if let Some(n) = self.window_rows {
            s.push_str(&format!(
                ", and a window of {n} rows of {} bytes each",
                Expectile::window_row_bytes(k, m, false)
            ));
        }
        s
    }
}

/// One target's refit buffer: its learned rows, `n_features + 2` numbers
/// each -- the features, the target and the row's weight decayed to now --
/// until the refit runs or is given up, and how it went.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Refit {
    rows: Vec<f64>,
    learned: u32,
    done: bool,
    passes: u32,
    failed: bool,
    over_budget: bool,
    /// Under a refit over a span of the clock, the stamp of the target's
    /// first learned row, from which the span runs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    first: Option<Stamp>,
}

impl Refit {
    fn new() -> Self {
        Self {
            rows: Vec::new(),
            learned: 0,
            done: false,
            passes: 0,
            failed: false,
            over_budget: false,
            first: None,
        }
    }

    fn status(&self) -> RefitStatus {
        RefitStatus {
            learned: self.learned,
            done: self.done,
            passes: self.passes,
            abandoned: false,
            failed: self.failed,
            over_budget: self.over_budget,
        }
    }

    /// Whether the buffer's shape is its count's: `learned` rows of `width`
    /// numbers until the refit runs, and fewer rows than a count of them
    /// (a span of the clock holds the rows its span has).
    fn fits(&self, width: usize, refit_rows: u32, clocked: bool) -> bool {
        let n = if self.done { 0 } else { self.learned as usize };
        n.checked_mul(width) == Some(self.rows.len())
            && (self.done || clocked || self.learned < refit_rows.max(1))
            && (clocked || self.first.is_none())
            // A span runs from the first row it holds.
            && (!clocked || self.done || self.learned == 0 || self.first.is_some())
    }
}

/// The rows a window holds, oldest first: each row's features, its
/// targets (a target absent on the row has its flag off and holds 0), its
/// weight, and under a clock window its stamp.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct RowWindow {
    x: VecDeque<f64>,
    y: VecDeque<f64>,
    present: VecDeque<bool>,
    w: VecDeque<f64>,
    #[serde(default)]
    stamp: VecDeque<Stamp>,
}

impl RowWindow {
    fn new() -> Self {
        Self {
            x: VecDeque::new(),
            y: VecDeque::new(),
            present: VecDeque::new(),
            w: VecDeque::new(),
            stamp: VecDeque::new(),
        }
    }

    fn rows(&self) -> usize {
        self.w.len()
    }

    fn has_shape(&self, k: usize, m: usize, clocked: bool) -> bool {
        let n = self.w.len();
        self.x.len() == n * k
            && self.y.len() == n * m
            && self.present.len() == n * m
            && self.stamp.len() == if clocked { n } else { 0 }
    }

    fn pop_front(&mut self, k: usize, m: usize) {
        self.x.drain(..k);
        self.y.drain(..m);
        self.present.drain(..m);
        self.w.pop_front();
        self.stamp.pop_front();
    }

    /// The weight in the window: every row's, or target `j`'s rows'.
    fn weight(&self, m: usize, j: Option<usize>) -> f64 {
        let mut s = 0.0;
        for (r, w) in self.w.iter().enumerate() {
            if j.is_none_or(|j| self.present[r * m + j]) {
                s += w;
            }
        }
        s
    }
}

/// What the rows so far have been: none, only rows of weight 0, or at least
/// one that counted.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
enum Seen {
    #[default]
    Nothing,
    OnlyZero,
    Counted,
}

/// The model's clock: where the last row that counted stood, and what the
/// rows of weight 0 since have stepped over.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
struct Clock {
    seen: Seen,
    /// Rows of weight 0 have come since the last row that counted.
    skipped: bool,
    /// Their steps, summed: read only by a caller that hands no stamps.
    carry: f64,
    /// The stamp of the last row that counted, where one was handed.
    last: Option<Stamp>,
    /// The stamp of the last row stepped, whatever its weight: where a read
    /// ahead is taken from.
    read: Option<Stamp>,
    /// The clock summed from every step, which keys a clock window's rows
    /// for a caller that hands no stamps.
    own: f64,
    /// The rows that counted: a clock window's key on the row clock.
    counted: u64,
    /// The last stamp handed was the row clock's.
    on_rows: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Expectile {
    cfg: ExpectileCfg,
    w_sum: f64,
    /// Per target, the weight of the rows that carried it, decayed.
    w_target: Vec<f64>,
    /// Per target, the asymmetric weight `S`; empty under a window.
    s: Vec<f64>,
    /// Per target, `G`'s lower triangle packed row by row, mean form; empty
    /// under a window.
    gram: Vec<Vec<f64>>,
    /// Per target, `c`; empty under a window.
    cross: Vec<Vec<f64>>,
    beta: Vec<Vec<f64>>,
    fitted: Vec<bool>,
    /// Per target, the rows of positive weight learned, one each.
    rows: Vec<u64>,
    /// Per target with the refit on; empty with it off.
    refit: Vec<Refit>,
    window: Option<RowWindow>,
    clock: Clock,
    /// The bytes a clock window's next row would have taken the state to,
    /// past the cap: from then on no row is learned.
    over: Option<u64>,
    /// The half-of-the-cap warning has been raised (once a group).
    warned: bool,
    /// Refits that ran to their pass cap or found no fit.
    pub(crate) solve_failures: u64,
    /// The next row's stamp, and whether it is the row clock's; and a
    /// warning raised and not yet taken. Never state.
    #[serde(skip)]
    next: Option<(Stamp, bool)>,
    #[serde(skip)]
    warning: Option<u64>,
    #[serde(skip)]
    work: Work,
}

/// The scratch a row writes: not state.
#[derive(Debug, Clone, Default)]
struct Work {
    x: Vec<f64>,
    /// The system a one-pass solve factorizes: `G` and its ridge.
    a: Vec<f64>,
    l: Vec<f64>,
    d: Vec<f64>,
    fit: Fit,
    /// The targets' weights a windowed row is gated on.
    weights: Vec<f64>,
}

impl PartialEq for Work {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

/// The buffers an exact refit reads and writes: the rows as `x̃` (row-major,
/// `n × p`), their targets and weights, and the solve's scratch.
#[derive(Debug, Clone, Default)]
struct Fit {
    /// The penalty `r V |β_s|²` and whether `β`'s first entry is the
    /// intercept, which it spares.
    ridge: f64,
    icpt: bool,
    z: Vec<f64>,
    y: Vec<f64>,
    v: Vec<f64>,
    g: Vec<f64>,
    c: Vec<f64>,
    l: Vec<f64>,
    b: Vec<f64>,
    trial: Vec<f64>,
    step: Vec<f64>,
}

/// Index of `(i, j)`, `j ≤ i`, in a lower triangle packed row by row.
#[inline]
fn at(i: usize, j: usize) -> usize {
    i * (i + 1) / 2 + j
}

/// The asymmetric weight of a residual: `τ` above 0, `1 − τ` at or below.
#[inline]
fn side_weight(r: f64, tau: f64) -> f64 {
    if r > 0.0 { tau } else { 1.0 - tau }
}

/// Solve `A x = b` for `A` symmetric, its lower triangle packed in `a`, by a
/// Cholesky factor written into `l`: `false` where a pivot is below
/// [`PIVOT_FLOOR`] of `A`'s largest diagonal, or not finite -- no fit.
fn chol_solve(a: &[f64], b: &[f64], k: usize, l: &mut Vec<f64>, x: &mut Vec<f64>) -> bool {
    chol_solve_floor::<false>(a, b, k, l, x)
}

/// [`chol_solve`] with the floor read against each pivot's own diagonal
/// where `OWN`: the pivot over its diagonal is one less the share of the
/// column the columns before it explain, whatever the columns' scales.
/// The centred solves read it so: a feature that is always 0 beside one
/// whose spread is 1e7 has the ridge alone as its diagonal and its pivot,
/// the whole of it, where against the largest diagonal it was under the
/// floor and the fit was refused. A raw `[1, x]` system keeps the largest
/// diagonal's floor, which also refuses a feature nearly constant against
/// the intercept: the centred solve then takes it, about the means.
fn chol_solve_floor<const OWN: bool>(
    a: &[f64],
    b: &[f64],
    k: usize,
    l: &mut Vec<f64>,
    x: &mut Vec<f64>,
) -> bool {
    let top = (0..k).fold(0.0_f64, |m, i| m.max(a[at(i, i)]));
    if !(top > 0.0 && top.is_finite()) {
        return false;
    }
    l.clear();
    l.resize(a.len(), 0.0);
    for i in 0..k {
        for j in 0..=i {
            let mut s = a[at(i, j)];
            for m in 0..j {
                s -= l[at(i, m)] * l[at(j, m)];
            }
            if i == j {
                let floor = if OWN { a[at(i, i)] } else { top };
                if !(s > 0.0 && s >= PIVOT_FLOOR * floor && s.is_finite()) {
                    return false;
                }
                l[at(i, i)] = s.sqrt();
            } else {
                l[at(i, j)] = s / l[at(j, j)];
            }
        }
    }
    x.clear();
    x.extend_from_slice(b);
    for i in 0..k {
        let mut s = x[i];
        for m in 0..i {
            s -= l[at(i, m)] * x[m];
        }
        x[i] = s / l[at(i, i)];
    }
    for i in (0..k).rev() {
        let mut s = x[i];
        for m in i + 1..k {
            s -= l[at(m, i)] * x[m];
        }
        x[i] = s / l[at(i, i)];
    }
    x.iter().all(|v| v.is_finite())
}

/// The ridged system solved about the features' means, where the raw one
/// failed its pivot floor: `G` is the mean-form Gram over `[1, x]`, so its
/// first row holds the means `m` and `G_xx − m m'` is the features'
/// covariance; the slopes solve `(G_xx − m m' + pen I) β_s = c_x − m c_0`
/// and the intercept is `c_0 − m'β_s`, the same minimiser of the same
/// penalized loss, the intercept being unpenalized. The floor is then read
/// against the covariance's largest diagonal, the features' spread, where
/// the raw system's was their squared level: a feature constant at -1000
/// left a pivot of the ridge alone, 1e-6, below 1e-12 of a diagonal of 1e6,
/// and the model never fitted (docs/PLAN.md task 538, the property stream
/// `test_expectile_collinear.py` pins). Reached only where the raw solve
/// fails, so every stream it fitted keeps its bits. Its floor is read
/// against each pivot's own diagonal ([`chol_solve_floor`]).
fn centred_solve(g: &[f64], c: &[f64], k: usize, pen: f64) -> Option<Vec<f64>> {
    let p = k.checked_sub(1).filter(|&p| p > 0)?;
    let m: Vec<f64> = (0..p).map(|i| g[at(i + 1, 0)]).collect();
    let mut a = vec![0.0; p * (p + 1) / 2];
    for i in 0..p {
        for t in 0..=i {
            a[at(i, t)] = g[at(i + 1, t + 1)] - m[i] * m[t];
        }
        a[at(i, i)] += pen;
    }
    let b: Vec<f64> = (0..p).map(|i| c[i + 1] - m[i] * c[0]).collect();
    let (mut l, mut x) = (Vec::new(), Vec::new());
    if !chol_solve_floor::<true>(&a, &b, p, &mut l, &mut x) {
        return None;
    }
    let intercept = c[0] - m.iter().zip(&x).map(|(mi, xi)| mi * xi).sum::<f64>();
    let mut beta = Vec::with_capacity(k);
    beta.push(intercept);
    beta.extend_from_slice(&x);
    beta.iter().all(|v| v.is_finite()).then_some(beta)
}

/// What an exact refit found: the fit, the passes it took, and whether it
/// stopped by a rule rather than at the cap.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Refitted {
    pub beta: Vec<f64>,
    pub passes: u32,
    pub converged: bool,
}

impl Fit {
    fn n(&self) -> usize {
        self.y.len()
    }

    fn clear(&mut self) {
        self.z.clear();
        self.y.clear();
        self.v.clear();
    }

    /// The asymmetric loss and its penalty, `Σ v a(r) r² + r V |β_s|²`, at
    /// `b`.
    fn loss(&self, b: &[f64], p: usize, tau: f64) -> f64 {
        let mut f = 0.0;
        for i in 0..self.n() {
            let r = self.y[i] - dot(&self.z[i * p..(i + 1) * p], b);
            f += self.v[i] * side_weight(r, tau) * r * r;
        }
        if self.ridge > 0.0 {
            let slopes = &b[usize::from(self.icpt)..];
            let v: f64 = self.v.iter().sum();
            f += self.ridge * v * slopes.iter().map(|x| x * x).sum::<f64>();
        }
        f
    }

    /// The WLS fit at the sides `b` gives (`None` for least squares, every
    /// side at 1/2), into `out`: `false` where it has none.
    fn wls(&mut self, sides_at: Option<&[f64]>, p: usize, tau: f64, out: &mut Vec<f64>) -> bool {
        self.g.clear();
        self.g.resize(p * (p + 1) / 2, 0.0);
        self.c.clear();
        self.c.resize(p, 0.0);
        let mut counted = 0usize;
        for i in 0..self.n() {
            let z = &self.z[i * p..(i + 1) * p];
            let a = match sides_at {
                Some(b) => side_weight(self.y[i] - dot(z, b), tau),
                None => 0.5,
            };
            let c = self.v[i] * a;
            if c == 0.0 {
                continue;
            }
            counted += 1;
            for r in 0..p {
                let cz = c * z[r];
                for (s, zs) in z.iter().enumerate().take(r + 1) {
                    self.g[at(r, s)] += cz * zs;
                }
                self.c[r] += cz * self.y[i];
            }
        }
        if self.ridge > 0.0 {
            let v: f64 = self.v.iter().sum();
            for i in usize::from(self.icpt)..p {
                self.g[at(i, i)] += self.ridge * v;
            }
        }
        let enough = if self.ridge > 0.0 { 1 } else { p };
        counted >= enough
            && (chol_solve(&self.g, &self.c, p, &mut self.l, out)
                || (self.icpt && p > 1 && self.centred_wls(sides_at, p, tau, out)))
    }

    /// [`Self::wls`] about the rows' own weighted means, where the raw
    /// system failed its pivot floor: the same penalized minimiser, the
    /// intercept being unpenalized, `β_s = (Σ c (z − m)(z − m)' + r V I)⁻¹
    /// Σ c (z − m)(y − ȳ)` and the intercept `ȳ − m'β_s`, `m` and `ȳ` the
    /// means at the weights `c = v a`. Formed from the rows themselves, it
    /// is exact where a feature is constant (the raw pivot is the ridge
    /// alone) or far from 0 against its spread (the raw pivot is lost to
    /// the level squared), where the one pass's fallback, read from the
    /// mean-form sums, can only be as good as their cancellation. Its floor
    /// is read against each pivot's own diagonal ([`chol_solve_floor`]).
    #[cold]
    #[inline(never)]
    fn centred_wls(
        &mut self,
        sides_at: Option<&[f64]>,
        p: usize,
        tau: f64,
        out: &mut Vec<f64>,
    ) -> bool {
        let q = p - 1;
        let n = self.n();
        let mut cw = Vec::with_capacity(n);
        let (mut s, mut sy) = (0.0, 0.0);
        let mut m = vec![0.0; q];
        for i in 0..n {
            let z = &self.z[i * p..(i + 1) * p];
            let a = match sides_at {
                Some(b) => side_weight(self.y[i] - dot(z, b), tau),
                None => 0.5,
            };
            let c = self.v[i] * a;
            cw.push(c);
            s += c;
            sy += c * self.y[i];
            for (mr, zr) in m.iter_mut().zip(&z[1..]) {
                *mr += c * zr;
            }
        }
        if !(s > 0.0 && s.is_finite()) {
            return false;
        }
        let ybar = sy / s;
        m.iter_mut().for_each(|v| *v /= s);
        let mut a = vec![0.0; q * (q + 1) / 2];
        let mut b = vec![0.0; q];
        let mut d = vec![0.0; q];
        for (i, &c) in cw.iter().enumerate() {
            if c == 0.0 {
                continue;
            }
            let z = &self.z[i * p + 1..(i + 1) * p];
            for (dr, (zr, mr)) in d.iter_mut().zip(z.iter().zip(&m)) {
                *dr = zr - mr;
            }
            let r = self.y[i] - ybar;
            for t in 0..q {
                let cd = c * d[t];
                for (u, du) in d.iter().enumerate().take(t + 1) {
                    a[at(t, u)] += cd * du;
                }
                b[t] += cd * r;
            }
        }
        if self.ridge > 0.0 {
            let v: f64 = self.v.iter().sum();
            for t in 0..q {
                a[at(t, t)] += self.ridge * v;
            }
        }
        let (mut l, mut x) = (Vec::new(), Vec::new());
        if !chol_solve_floor::<true>(&a, &b, q, &mut l, &mut x) {
            return false;
        }
        out.clear();
        out.push(ybar - m.iter().zip(&x).map(|(mi, xi)| mi * xi).sum::<f64>());
        out.extend_from_slice(&x);
        out.iter().all(|v| v.is_finite())
    }

    /// The exact expectile regression of the rows held, warm-started at
    /// `warm` (the module docs): `None` where the rows determine no fit.
    fn refit(&mut self, p: usize, tau: f64, warm: Option<&[f64]>) -> Option<Refitted> {
        if tau == 0.5 {
            let mut b = std::mem::take(&mut self.b);
            let ok = self.wls(None, p, tau, &mut b);
            self.b = b;
            return ok.then(|| Refitted {
                beta: self.b.clone(),
                passes: 1,
                converged: true,
            });
        }
        let mut beta = Vec::with_capacity(p);
        match warm {
            Some(b) => beta.extend_from_slice(b),
            None => {
                let mut b = std::mem::take(&mut self.b);
                let ok = self.wls(None, p, tau, &mut b);
                self.b = b;
                if !ok {
                    return None;
                }
                beta.extend_from_slice(&self.b);
            }
        }
        let n = self.n();
        let eps = f64::EPSILON;
        let sides_of = |fit: &Fit, b: &[f64]| -> Vec<bool> {
            (0..n)
                .map(|i| fit.y[i] - dot(&fit.z[i * p..(i + 1) * p], b) > 0.0)
                .collect()
        };
        let mut sides = sides_of(self, &beta);
        let mut f_old = self.loss(&beta, p, tau);
        let f_floor = 8.0
            * eps
            * (0..n)
                .map(|i| self.v[i] * self.y[i] * self.y[i])
                .sum::<f64>();
        let wsum: f64 = self.v.iter().sum();
        let mut seen: Vec<(Vec<bool>, f64, Vec<f64>)> = vec![(sides.clone(), f_old, beta.clone())];
        let mut best = (f_old, beta.clone());
        for pass in 1..=EXPECTILE_REFIT_PASSES {
            let mut new = std::mem::take(&mut self.trial);
            let ok = self.wls(Some(&beta), p, tau, &mut new);
            if !ok {
                self.trial = new;
                // A warm start is a fit of other rows: where these rows
                // give no first pass from it, they start cold, from least
                // squares, or determine no fit.
                if pass == 1 && warm.is_some() {
                    return self.refit(p, tau, None);
                }
                return Some(Refitted {
                    beta: best.1,
                    passes: pass,
                    converged: false,
                });
            }
            self.step.clear();
            self.step.extend(new.iter().zip(&beta).map(|(a, b)| a - b));
            let mut f_new = self.loss(&new, p, tau);
            let mut h = 1.0;
            while f_new > f_old + f_old.max(f_floor) * 8.0 * eps && h > 1e-6 {
                h *= 0.5;
                for ((t, b), s) in new.iter_mut().zip(&beta).zip(&self.step) {
                    *t = b + h * s;
                }
                f_new = self.loss(&new, p, tau);
            }
            let (mut moved, mut rounding, mut rss) = (0.0_f64, 0.0_f64, 0.0);
            let mut new_sides = Vec::with_capacity(n);
            for i in 0..n {
                let z = &self.z[i * p..(i + 1) * p];
                let fitted = dot(z, &new);
                let r = self.y[i] - fitted;
                moved = moved.max((fitted - dot(z, &beta)).abs());
                rounding = rounding.max(fitted.abs());
                rss += self.v[i] * r * r;
                new_sides.push(r > 0.0);
            }
            let scale = (rss / wsum.max(1e-300)).sqrt();
            rounding *= 8.0 * eps;
            beta.clear();
            beta.extend_from_slice(&new);
            self.trial = new;
            f_old = f_new;
            if f_new < best.0 {
                best = (f_new, beta.clone());
            }
            let done = |b: Vec<f64>| {
                Some(Refitted {
                    beta: b,
                    passes: pass,
                    converged: true,
                })
            };
            if h == 1.0 && new_sides == sides {
                return done(beta);
            }
            // Read only at a finite scale: residuals past the doubles'
            // range give no yardstick for "moved".
            if scale.is_finite() && moved <= (REFIT_TOL * scale).max(rounding) {
                return done(beta);
            }
            // A side pattern met before with no lower loss since: rows at
            // rounding level trading sides, which no further pass resolves.
            if let Some(prev) = seen.iter().find(|(s, _, _)| *s == new_sides)
                && f_new >= prev.1 - f_new.max(f_floor) * 8.0 * eps
            {
                return done(best.1);
            }
            seen.push((new_sides.clone(), f_new, beta.clone()));
            sides = new_sides;
        }
        Some(Refitted {
            beta: best.1,
            passes: EXPECTILE_REFIT_PASSES,
            converged: false,
        })
    }
}

#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    let mut s = 0.0;
    for (x, y) in a.iter().zip(b) {
        s += x * y;
    }
    s
}

impl Expectile {
    pub fn new(cfg: ExpectileCfg) -> Result<Self, String> {
        cfg.validate()?;
        let k = cfg.k_total();
        let m = cfg.n_targets;
        let windowed = cfg.windowed();
        let sums = if windowed { 0 } else { m };
        Ok(Self {
            w_sum: 0.0,
            w_target: vec![0.0; m],
            s: vec![0.0; sums],
            gram: vec![vec![0.0; k * (k + 1) / 2]; sums],
            cross: vec![vec![0.0; k]; sums],
            beta: vec![vec![0.0; k]; m],
            fitted: vec![false; m],
            rows: vec![0; m],
            refit: if cfg.refits() {
                vec![Refit::new(); m]
            } else {
                Vec::new()
            },
            window: windowed.then(RowWindow::new),
            clock: Clock::default(),
            over: None,
            warned: false,
            solve_failures: 0,
            next: None,
            warning: None,
            work: Work::default(),
            cfg,
        })
    }

    pub fn cfg(&self) -> &ExpectileCfg {
        &self.cfg
    }

    /// Per target, `[intercept, slopes…]`; `NaN` before its first fit.
    pub fn coefficients(&self) -> Vec<Vec<f64>> {
        self.beta
            .iter()
            .zip(&self.fitted)
            .map(|(b, &f)| {
                if f {
                    b.clone()
                } else {
                    vec![f64::NAN; b.len()]
                }
            })
            .collect()
    }

    /// Whether any target has a fit: before one, there are no coefficients
    /// to report.
    pub fn has_fit(&self) -> bool {
        self.fitted.iter().any(|&f| f)
    }

    /// The weight the fit is made of (hard rule 8): every row's, decayed,
    /// and under a window the weight of the rows it holds.
    pub fn n_eff(&self) -> f64 {
        match &self.window {
            Some(w) => w.weight(self.cfg.n_targets, None),
            None => self.w_sum,
        }
    }

    /// Per target, the weight of the rows it learned from (under a window,
    /// of the rows in it).
    pub fn target_weights(&self) -> Vec<f64> {
        match &self.window {
            Some(w) => (0..self.cfg.n_targets)
                .map(|j| w.weight(self.cfg.n_targets, Some(j)))
                .collect(),
            None => self.w_target.clone(),
        }
    }

    pub fn solve_failures(&self) -> u64 {
        self.solve_failures
    }

    /// The stamp last handed to a step, `None` before one: where a caller
    /// that stamps its rows resumes from a restored state.
    pub fn last_stamp(&self) -> Option<Stamp> {
        self.clock.read
    }

    /// The rows a window holds; 0 without a window.
    pub fn window_rows(&self) -> usize {
        self.window.as_ref().map_or(0, RowWindow::rows)
    }

    /// How each target's refit stands; empty with the refit off.
    pub fn refit_status(&self) -> Vec<RefitStatus> {
        self.refit.iter().map(Refit::status).collect()
    }

    /// The bytes this model's refit buffers can hold at most: under a
    /// refit over a span of the clock, what the cap leaves beside the sums,
    /// since past that the run is refused.
    pub fn refit_reservation(&self) -> usize {
        if self.refit.is_empty() {
            return 0;
        }
        if self.cfg.refit_clock.is_some() {
            let (k, m) = (self.cfg.n_features, self.cfg.n_targets);
            let sums = Self::sums_bytes(k, m, self.cfg.fit_intercept, false);
            return usize::try_from(self.cfg.cap().saturating_sub(sums)).unwrap_or(usize::MAX);
        }
        self.cfg.n_targets * 8 * self.cfg.refit_rows as usize * (self.cfg.n_features + 2)
    }

    /// The bytes the state holds under a refit over a span of the clock,
    /// with `extra` more rows in its buffers: the sums, each buffered row's
    /// `k + 2` numbers and each target's first stamp.
    fn refit_clock_bytes(&self, extra: usize) -> u64 {
        let (k, m) = (self.cfg.n_features, self.cfg.n_targets);
        let held: usize = self.refit.iter().map(|r| r.rows.len()).sum();
        let stamps = (m * std::mem::size_of::<Option<Stamp>>()) as u64;
        Self::sums_bytes(k, m, self.cfg.fit_intercept, false)
            + 8 * (held + extra * (k + 2)) as u64
            + stamps
    }

    /// The run-time check of a state the data sizes, over this model's
    /// fields ([`crate::clock_span::HeldCap`]).
    fn held_cap(&mut self) -> crate::clock_span::HeldCap<'_> {
        crate::clock_span::HeldCap {
            over: &mut self.over,
            warned: &mut self.warned,
            warning: &mut self.warning,
        }
    }

    /// Give every target's refit up before it begins: its rows are freed
    /// and it keeps the one-pass fit.
    pub fn refit_give_up(&mut self) {
        for r in self.refit.iter_mut().filter(|r| !r.done) {
            r.rows = Vec::new();
            r.done = true;
            r.over_budget = true;
        }
    }

    /// The bytes of the sums: per target `8 (p(p+1)/2 + 2p + 4)` -- under a
    /// window `8 (p + 3)`, `β`, its weight, its row count and its flag --
    /// and the shared weight and clock, `8 · 6` and two stamps
    /// ([`std::mem::size_of`] a [`Stamp`], an `Option` of one each).
    pub fn sums_bytes(
        n_features: usize,
        n_targets: usize,
        fit_intercept: bool,
        windowed: bool,
    ) -> u64 {
        let p = (n_features + usize::from(fit_intercept)) as u64;
        let per = if windowed {
            p + 3
        } else {
            p * (p + 1) / 2 + 2 * p + 4
        };
        let stamp = std::mem::size_of::<Option<Stamp>>() as u64;
        8 * (n_targets as u64 * per + 6) + 2 * stamp
    }

    /// The bytes of one row a window holds: its features, its targets and
    /// its weight, 8 bytes each, a flag a target, and under a clock window
    /// its stamp.
    pub fn window_row_bytes(n_features: usize, n_targets: usize, clocked: bool) -> u64 {
        8 * (n_features + n_targets + 1) as u64
            + n_targets as u64
            + if clocked {
                std::mem::size_of::<Stamp>() as u64
            } else {
                0
            }
    }

    /// The bytes a clock window's state holds with `rows` rows in it.
    pub fn clock_state_bytes(
        n_features: usize,
        n_targets: usize,
        fit_intercept: bool,
        rows: usize,
    ) -> u64 {
        Self::sums_bytes(n_features, n_targets, fit_intercept, true)
            + rows as u64 * Self::window_row_bytes(n_features, n_targets, true)
    }

    fn clock_bytes_with(&self, extra: usize) -> u64 {
        Self::clock_state_bytes(
            self.cfg.n_features,
            self.cfg.n_targets,
            self.cfg.fit_intercept,
            self.window_rows() + extra,
        )
    }

    fn ensure_work(&mut self) {
        let k = self.cfg.k_total();
        if self.work.x.len() != k {
            self.work.x = vec![0.0; k];
            if self.cfg.fit_intercept {
                self.work.x[0] = 1.0;
            }
        }
    }

    /// Target `j`'s prediction at `x`, gated on `floor`.
    fn read(&self, j: usize, x: &[f64], weight: f64, floor: f64) -> f64 {
        if self.fitted[j] && weight >= floor {
            dot_aug(&self.beta[j], x, self.cfg.fit_intercept)
        } else {
            f64::NAN
        }
    }

    /// The gap a row that counts decays by: its own step where it follows a
    /// row that counted; else taken once from the last row that counted --
    /// the two stamps, or on the row clock the step alone, a row of weight
    /// 0 being no step there -- and only for a caller that hands no stamps
    /// the steps summed.
    fn gap(&self, d_clock: f64, now: Option<(Stamp, bool)>) -> f64 {
        if !self.clock.skipped {
            return d_clock;
        }
        match (now, self.clock.last) {
            (Some((_, true)), _) => d_clock,
            (Some((now, false)), Some(last)) => now.since(last),
            _ => self.clock.carry + d_clock,
        }
    }

    /// A clock window's key for a row: on the row clock the count of the
    /// rows that counted (this one too, where `counting`), else the stamp
    /// handed, else the clock summed from the steps.
    fn window_key(&self, now: Option<(Stamp, bool)>, d_clock: f64, counting: bool) -> Stamp {
        match now {
            Some((_, true)) if counting => Stamp::from((self.clock.counted + 1) as f64),
            // A row of weight 0 counts in no clock, and is read as a row
            // `d_clock` on would be: as `predict` reads it.
            Some((_, true)) => Stamp::from(self.clock.counted as f64 + d_clock),
            Some((s, false)) => s,
            None => Stamp::from(self.clock.own + d_clock),
        }
    }

    /// A read ahead's key, for a row `d_clock` after the last stepped that
    /// would count: on the row clock `d_clock` rows on, as a clock of
    /// numbers reads `d_clock` units on (a row scored under `embargo` sits
    /// after the rows held, its step theirs and its own).
    fn key_ahead(&self, d_clock: f64) -> Stamp {
        match self.clock.read {
            _ if self.clock.on_rows => Stamp::from(self.clock.counted as f64 + d_clock),
            Some(last) => back_by(last, -d_clock),
            None => Stamp::from(self.clock.own + d_clock),
        }
    }

    /// Whether the oldest row of a clock window has left at `now`.
    fn oldest_has_left(&self, now: Stamp) -> bool {
        let Some(span) = self.cfg.window_clock else {
            return false;
        };
        self.window
            .as_ref()
            .and_then(|w| w.stamp.front())
            .is_some_and(|&s| self.cfg.closed.has_left(now, s, span))
    }

    /// Drop a clock window's rows that have left at `now`: true if any did.
    fn trim(&mut self, now: Stamp) -> bool {
        let (k, m) = (self.cfg.n_features, self.cfg.n_targets);
        let mut dropped = false;
        while self.oldest_has_left(now) {
            self.window.as_mut().expect("windowed").pop_front(k, m);
            dropped = true;
        }
        dropped
    }

    /// Every target refit on the window's rows, warm-started at its fit.
    fn refit_window(&mut self) {
        let (k, m, p) = (self.cfg.n_features, self.cfg.n_targets, self.cfg.k_total());
        let tau = self.cfg.expectile;
        let icpt = self.cfg.fit_intercept;
        let Some(w) = self.window.as_ref() else {
            return;
        };
        let fit = &mut self.work.fit;
        fit.ridge = self.cfg.ridge;
        fit.icpt = icpt;
        for j in 0..m {
            fit.clear();
            for r in 0..w.rows() {
                if !w.present[r * m + j] {
                    continue;
                }
                if icpt {
                    fit.z.push(1.0);
                }
                fit.z.extend((0..k).map(|f| w.x[r * k + f]));
                fit.y.push(w.y[r * m + j]);
                fit.v.push(w.w[r]);
            }
            let warm = self.fitted[j].then_some(self.beta[j].as_slice());
            match fit.refit(p, tau, warm) {
                Some(r) => {
                    if !r.converged {
                        self.solve_failures += 1;
                    }
                    self.beta[j] = r.beta;
                    self.fitted[j] = true;
                }
                // A window whose rows no longer determine the fit it had
                // (emptied, or too few rows carry the target) found no fit,
                // which counts; a target never fitted lost nothing.
                None => {
                    if self.fitted[j] {
                        self.solve_failures += 1;
                    }
                    self.fitted[j] = false;
                }
            }
        }
    }

    /// Target `j`'s one-pass update at the asymmetric weight `c = w a`.
    fn learn(&mut self, j: usize, y: f64, c: f64, lam: f64) {
        let k = self.cfg.k_total();
        let s = lam * self.s[j] + c;
        self.s[j] = s;
        if c == 0.0 || s <= 0.0 {
            return;
        }
        let f = c / s;
        let x = &self.work.x;
        let g = &mut self.gram[j];
        for i in 0..k {
            for (m, xm) in x.iter().enumerate().take(i + 1) {
                let at_im = at(i, m);
                g[at_im] += f * (x[i] * xm - g[at_im]);
            }
        }
        for (ci, xi) in self.cross[j].iter_mut().zip(x) {
            *ci += f * (xi * y - *ci);
        }
        self.rows[j] += 1;
        self.solve(j);
    }

    /// `β = (G + r (W/S) D)⁻¹ c` for target `j`, `W` its rows' raw weight
    /// and `D` the slopes' diagonal, from its first row under a ridge and
    /// once it has a row per coefficient without one; where the system is
    /// not positive definite -- a history forgotten to its last row -- the
    /// fit before stands.
    fn solve(&mut self, j: usize) {
        let k = self.cfg.k_total();
        let ridge = self.cfg.ridge;
        if ridge == 0.0 && self.rows[j] < k as u64 {
            return;
        }
        let Work { l, d, a, .. } = &mut self.work;
        a.clear();
        a.extend_from_slice(&self.gram[j]);
        if ridge > 0.0 && self.s[j] > 0.0 {
            let pen = ridge * self.w_target[j] / self.s[j];
            for i in usize::from(self.cfg.fit_intercept)..k {
                a[at(i, i)] += pen;
            }
        }
        if chol_solve(a, &self.cross[j], k, l, d) {
            self.beta[j].copy_from_slice(d);
            self.fitted[j] = true;
        } else if ridge > 0.0
            && self.cfg.fit_intercept
            && self.s[j] > 0.0
            && let Some(beta) = centred_solve(
                &self.gram[j],
                &self.cross[j],
                k,
                ridge * self.w_target[j] / self.s[j],
            )
        {
            self.beta[j].copy_from_slice(&beta);
            self.fitted[j] = true;
        }
    }

    /// Target `j`'s first rows fitted exactly, and its sums rebuilt at the
    /// converged sides (the module docs).
    fn run_refit(&mut self, j: usize) {
        let (nf, p) = (self.cfg.n_features, self.cfg.k_total());
        let tau = self.cfg.expectile;
        let icpt = self.cfg.fit_intercept;
        let rows = std::mem::take(&mut self.refit[j].rows);
        let fit = &mut self.work.fit;
        fit.ridge = self.cfg.ridge;
        fit.icpt = icpt;
        fit.clear();
        for row in rows.chunks_exact(nf + 2) {
            if icpt {
                fit.z.push(1.0);
            }
            fit.z.extend_from_slice(&row[..nf]);
            fit.y.push(row[nf]);
            fit.v.push(row[nf + 1]);
        }
        let warm = self.fitted[j].then_some(self.beta[j].as_slice());
        let got = fit.refit(p, tau, warm);
        let r = &mut self.refit[j];
        r.done = true;
        let Some(got) = got else {
            r.failed = true;
            self.solve_failures += 1;
            return;
        };
        r.passes = got.passes;
        if !got.converged {
            self.solve_failures += 1;
        }
        // The sums rebuilt at the converged sides, in sums form, then as
        // means.
        let (mut s, g, c) = (0.0, &mut self.gram[j], &mut self.cross[j]);
        g.fill(0.0);
        c.fill(0.0);
        let fit = &self.work.fit;
        for i in 0..fit.n() {
            let z = &fit.z[i * p..(i + 1) * p];
            let a = fit.v[i] * side_weight(fit.y[i] - dot(z, &got.beta), tau);
            s += a;
            for q in 0..p {
                for (t, zt) in z.iter().enumerate().take(q + 1) {
                    g[at(q, t)] += a * z[q] * zt;
                }
                c[q] += a * z[q] * fit.y[i];
            }
        }
        if s > 0.0 {
            g.iter_mut().for_each(|v| *v /= s);
            c.iter_mut().for_each(|v| *v /= s);
        }
        self.s[j] = s;
        self.beta[j] = got.beta;
        self.fitted[j] = true;
    }

    /// Each target's weight, the window's rows' under a window, into
    /// `out`: what `min_weight` is checked against.
    fn weights_into(&self, out: &mut Vec<f64>) {
        let m = self.cfg.n_targets;
        out.clear();
        match &self.window {
            Some(w) => out.extend((0..m).map(|j| w.weight(m, Some(j)))),
            None => out.extend_from_slice(&self.w_target),
        }
    }

    /// The row's reading: each target's prediction at the fit as it stands,
    /// gated on `weights`, the targets' weights before the row's own clock
    /// step (a clock window's before the rows that leave at the row, as
    /// `n_eff` is: hard rule 8), and `n_eff`.
    fn read_into(&self, x: &[f64], floor: f64, n_eff: f64, weights: &[f64], out: &mut Step) {
        out.pred.clear();
        for (j, &weight) in weights.iter().enumerate() {
            out.pred.push(self.read(j, x, weight, floor));
        }
        out.n_eff = n_eff;
        out.extra = None;
    }

    /// What a row `d_clock` after the last stepped reads: under a clock
    /// window whose oldest row has left by then, the fit of the rows left,
    /// on a copy; else the fit as it stands.
    fn read_ahead(&self, x: &[f64], d_clock: f64, key: Stamp, floor: f64, out: &mut Step) {
        let n_eff = self.n_eff();
        let mut weights = Vec::new();
        self.weights_into(&mut weights);
        if self.oldest_has_left(key) {
            let mut at = self.clone();
            at.ensure_work();
            at.trim(key);
            at.refit_window();
            at.read_into(x, floor, n_eff, &weights, out);
            return;
        }
        let _ = d_clock;
        self.read_into(x, floor, n_eff, &weights, out);
    }

    fn step_row(
        &mut self,
        x: &[f64],
        y: &[Option<f64>],
        d_clock: f64,
        weight: f64,
        out: &mut Step,
    ) {
        let handed = self.next.take();
        let floor = self.cfg.min_weight;
        // A row of weight 0 counts in no clock (hard rule 9): read as any
        // row is, from the last row that counted, and nothing kept of it but
        // the clock it stepped over.
        if weight <= 0.0 {
            let key = self.window_key(handed, d_clock, false);
            self.read_ahead(x, d_clock, key, floor, out);
            match self.clock.seen {
                Seen::Counted => {
                    self.clock.skipped = true;
                    self.clock.carry += d_clock;
                }
                _ => self.clock.seen = Seen::OnlyZero,
            }
            self.clock.own += d_clock;
            if let Some((s, rows)) = handed {
                self.clock.read = Some(s);
                self.clock.on_rows = rows;
            }
            return;
        }
        let gap = match self.clock.seen {
            Seen::Counted => self.gap(d_clock, handed),
            Seen::OnlyZero => 0.0,
            Seen::Nothing => d_clock,
        };
        let key = self.window_key(handed, d_clock, true);
        self.clock.seen = Seen::Counted;
        self.clock.skipped = false;
        self.clock.carry = 0.0;
        self.clock.own += d_clock;
        self.clock.counted += 1;
        if let Some((s, rows)) = handed {
            self.clock.read = Some(s);
            self.clock.last = (!rows).then_some(s);
            self.clock.on_rows = rows;
        }
        let lam = self.cfg.decay.factor(gap);
        let n_eff = self.n_eff();
        let (m, nf) = (self.cfg.n_targets, self.cfg.n_features);
        let off = usize::from(self.cfg.fit_intercept);
        self.ensure_work();
        self.work.x[off..].copy_from_slice(x);
        if self.window.is_some() {
            // The rows out of a clock window by now leave before the row is
            // read, and the fit of what is left is the one it reads, gated
            // on the weight before they left, as `n_eff` reports it.
            let mut weights = std::mem::take(&mut self.work.weights);
            self.weights_into(&mut weights);
            if self.cfg.window_clock.is_some() && self.trim(key) {
                self.refit_window();
            }
            self.read_into(x, floor, n_eff, &weights, out);
            self.work.weights = weights;
            if self.over.is_some() {
                return;
            }
            if self.cfg.window_clock.is_some() {
                let (bytes, cap) = (self.clock_bytes_with(1), self.cfg.cap());
                if !self.held_cap().admit(bytes, cap) {
                    return;
                }
            }
            let w = self.window.as_mut().expect("windowed");
            w.x.extend(x.iter().copied());
            for t in y.iter().take(m) {
                w.y.push_back(t.unwrap_or(0.0));
                w.present.push_back(t.is_some());
            }
            w.w.push_back(weight);
            if self.cfg.window_clock.is_some() {
                w.stamp.push_back(key);
            } else if let Some(cap) = self.cfg.window_rows
                && w.rows() > cap
            {
                w.pop_front(nf, m);
            }
            self.refit_window();
            if self.cfg.window_clock.is_some() {
                let (bytes, cap) = (self.clock_bytes_with(0), self.cfg.cap());
                self.held_cap().note(bytes, cap);
            }
            return;
        }
        self.read_into(x, floor, n_eff, &self.w_target, out);
        // A refit over a span of the clock holds the rows the span has: a
        // row that would take the state past the cap is not learned, nor
        // any after it.
        if self.cfg.refit_clock.is_some() {
            let joining = self
                .refit
                .iter()
                .zip(y)
                .filter(|(r, t)| !r.done && t.is_some())
                .count();
            let (bytes, cap) = (self.refit_clock_bytes(joining), self.cfg.cap());
            if !self.held_cap().admit(bytes, cap) {
                return;
            }
        }
        let tau = self.cfg.expectile;
        let p = self.cfg.k_total();
        // The targets' weights first: a solve reads its rows' weight `W`
        // with this row's in it, as `S` has it.
        crate::model::age_target_weights(&mut self.w_target, |j| y[j].is_some(), lam, weight);
        for (j, &yj) in y.iter().enumerate().take(m) {
            let Some(t) = yj else {
                // The sums age with the clock, whether or not the row
                // carries the target.
                self.s[j] *= lam;
                if let Some(r) = self.refit.get_mut(j).filter(|r| !r.done) {
                    for row in r.rows.chunks_exact_mut(nf + 2) {
                        row[nf + 1] *= lam;
                    }
                }
                continue;
            };
            // The side at the fit before the row, unmasked by `min_weight`.
            let a = if self.fitted[j] {
                side_weight(t - dot_aug(&self.beta[j], x, self.cfg.fit_intercept), tau)
            } else {
                0.5
            };
            self.learn(j, t, weight * a, lam);
            if let Some(r) = self.refit.get_mut(j).filter(|r| !r.done) {
                for row in r.rows.chunks_exact_mut(nf + 2) {
                    row[nf + 1] *= lam;
                }
                r.rows.extend_from_slice(x);
                r.rows.push(t);
                r.rows.push(weight);
                r.learned += 1;
                let due = match self.cfg.refit_clock {
                    Some(span) => {
                        let first = *r.first.get_or_insert(key);
                        r.learned as usize >= p.max(1)
                            && crate::clock_span::span_reached(key, first, span)
                    }
                    None => r.learned >= self.cfg.refit_rows,
                };
                if due {
                    self.run_refit(j);
                }
            }
        }
        self.w_sum = lam * self.w_sum + weight;
        if self.cfg.refit_clock.is_some() {
            let (bytes, cap) = (self.refit_clock_bytes(0), self.cfg.cap());
            self.held_cap().note(bytes, cap);
        }
    }
}

impl OnlineModel for Expectile {
    fn target_n_eff_into(&self, out: &mut Vec<f64>) -> bool {
        out.clear();
        out.extend(self.target_weights());
        true
    }

    fn step_into(
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
        self.step_row(x, y, d_clock, weight, out);
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

    fn predict(&self, x: &[f64], d_clock: f64) -> Step {
        if let Some(refused) = crate::model::refused_predict(self, x, d_clock) {
            return refused;
        }
        let mut out = Step::default();
        let key = self.key_ahead(d_clock);
        self.read_ahead(x, d_clock, key, self.cfg.min_weight, &mut out);
        out
    }

    /// A row's stamp decides the gap over rows of weight 0, and a clock
    /// window's edge.
    fn stamp_next(&mut self, stamp: Stamp) {
        self.next = Some((stamp, false));
    }

    /// On the row clock a row of weight 0 is no step, and a clock window
    /// counts the rows that counted.
    fn stamp_next_on_rows(&mut self, stamp: Stamp) {
        self.next = Some((stamp, true));
    }

    fn take_state_warning(&mut self) -> Option<(u64, u64)> {
        self.warning.take().map(|bytes| (bytes, self.cfg.cap()))
    }

    fn state_over_cap(&self) -> Option<(u64, u64)> {
        self.over.map(|bytes| (bytes, self.cfg.cap()))
    }

    fn state(&self) -> State {
        State::new(ModelState::Expectile(Box::new(self.clone())))
    }

    fn restore(s: &State) -> Result<Self, StateError> {
        check_schema(s)?;
        match &s.model {
            ModelState::Expectile(m) => {
                let m = (**m).clone();
                crate::model::check_cfg("expectile", m.cfg.validate())?;
                let (n, k) = (m.cfg.n_targets, m.cfg.k_total());
                let sums = if m.cfg.windowed() { 0 } else { n };
                let width = m.cfg.n_features + 2;
                let shaped = m.w_target.len() == n
                    && m.s.len() == sums
                    && m.gram.len() == sums
                    && m.gram.iter().all(|g| g.len() == k * (k + 1) / 2)
                    && m.cross.len() == sums
                    && m.cross.iter().all(|c| c.len() == k)
                    && m.beta.len() == n
                    && m.beta.iter().all(|b| b.len() == k)
                    && m.fitted.len() == n
                    && m.rows.len() == n
                    && m.refit.len() == if m.cfg.refits() { n } else { 0 }
                    && m.refit
                        .iter()
                        .all(|r| r.fits(width, m.cfg.refit_rows, m.cfg.refit_clock.is_some()))
                    // Only a span's buffers or a clock window can be stopped
                    // at the cap.
                    && (m.over.is_none()
                        || m.cfg.refit_clock.is_some()
                        || m.cfg.window_clock.is_some())
                    && match (&m.window, m.cfg.windowed()) {
                        (None, false) => true,
                        (Some(w), true) => {
                            w.has_shape(m.cfg.n_features, n, m.cfg.window_clock.is_some())
                                && m.cfg.window_rows.is_none_or(|c| w.rows() <= c)
                        }
                        _ => false,
                    };
                if !shaped {
                    return Err(crate::model::damaged(
                        "expectile",
                        "its running sums have the wrong shape",
                    ));
                }
                Ok(m)
            }
            other => Err(StateError::WrongModel {
                expected: "expectile",
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
