//! Generalized linear models by recursive IRLS: Poisson, Gamma, binomial
//! and Tweedie regressions fitted one row at a time (docs/PLAN.md task 500;
//! the study: docs/records/STUDY-500-GLM.md).
//!
//! **The recursion.** Each target `j` keeps a Gram `G` and its
//! coefficients `β`, `x̃ = [1, x]` (`x` without an intercept). Row `t`, of
//! weight `w` and clock step `Δ`, `λ = 2^(−Δ/half_life)`:
//!
//! ```text
//! η = x̃'β_{t−1}                      the fit before the row
//! μ, W, z at η                        the family's mean, Fisher weight and working response
//! pred = μ                            emitted before the update (out of sample)
//! G_t = λ G_{t−1} + w W x̃x̃'
//! β_t = β_{t−1} + (G_t + R I)⁻¹ w W (z − η) x̃
//! ```
//!
//! with `W = 1/(g'(μ)² V(μ))` and `z = η + (y − μ) g'(μ)`
//! ([`crate::Family::working`]); `w W (z − η) x̃` is the row's score. This
//! is one Fisher-scoring step a row on the objective
//!
//! ```text
//! F_t(β) = Σ_i w_i λ^(age_i) d(y_i, μ_i(β))/2 + λ^(age_0) R/2 |β|²
//! ```
//!
//! (`d` the family's unit deviance, `age_0` the stream's whole age), each
//! past row's curvature frozen at the fit that scored it. Without decay
//! that is the penalized batch fit, the sums form `β_t = (G_t + R I)⁻¹ h_t`,
//! `h_t = h_{t−1} + w W z x̃`, step for step: statsmodels' `GLM` with
//! `freq_weights` (a weight counts its row that many times) penalized by
//! `R/2 |β|²`, or glum's with `alpha = R / Σ w`. Under decay the ridge
//! ages as a row at the stream's start does, so a row of weight 0 decays
//! `G` and moves no coefficient, and the fit's target tends to the decayed
//! likelihood's own maximum, `GLM(freq_weights = λ^age)`. `R` keeps every
//! solve's `G + R I` positive definite, the first row's included. The
//! intercept is penalized with the slopes.
//!
//! **The start.** Before about `k` rows the fit is underdetermined, and one
//! row can move `η` by tens. Measured over 50 seeds at 20 coefficients and
//! `R = 1`, with no remedy the recursion then either runs away -- a Gamma
//! row scored at a very negative `η` has a score of order `y e^|η|`, the
//! next step is enormous and the fit never returns (34 of 50 seeds) -- or
//! sticks: a Poisson row scored at a too-large `μ` enters `G` with the
//! weight `W = μ` of that fit, and without decay that direction is starved
//! for good (23 of 50). Two remedies hold every seed:
//!
//! - **the refit** (`refit_rows`, the default): each target's first
//!   `refit_rows` learned rows are kept, and on the row that fills the
//!   buffer, after its prediction and update, they are fitted to
//!   convergence by penalized IRLS with step halving from `β_0 = (g(ȳ),
//!   0, …)`, at their decayed weights under the penalty `R/2 |β|²`, and `G`
//!   rebuilt at that fit: `G = Σ v_i W_i(β*) x̃_i x̃_i'`, `β = β*`. The buffer is
//!   then freed. Its largest gap to the batch fit at 100,000 rows was 0.20,
//!   0.16 and 0.31 standard errors (Poisson, Gamma, Tweedie at `p = 1.5`).
//!   While the buffer fills, the recursion runs held to a step cap of 1
//!   ([`WAITING_CAP`]), so the predictions before the refit are the capped
//!   recursion's rather than a runaway's; the refit then replaces all it
//!   left. `refit_clock = D` is the same refit over a span of the clock
//!   (docs/PLAN.md task 538): a target's rows from its first learned row
//!   through the first at least `D` clock units after it, once they hold a
//!   row per coefficient, `D` read off the rows' exact stamps
//!   ([`crate::clock_span::span_reached`]), `D + 1` rows on the row clock.
//!   How many rows that is the data decides, so the buffers are checked
//!   against [`GLM_GROUP_REFIT_CAP`] as they fill: a warning past half of
//!   it, once, and the row that would pass it learned by no target, nor any
//!   row after it ([`OnlineModel::state_over_cap`]);
//! - **the step cap** (`step_cap`): a learned row whose own `η` would move
//!   by more than `c`, `|x̃'(β_t − β_{t−1})| > c`, has its step scaled to
//!   `c`. No buffer; without decay its gap was 2.9
//!   (Poisson) and 1.8 (Gamma) times the refit's in the worst seed, since
//!   the capped rows' weights stay in `G`, and with decay the same as the
//!   refit's.
//!
//! **Rows that teach nothing.** A row of weight 0 decays `G` and learns
//! nothing; a target
//! that is absent or outside the family's support ([`crate::Family::takes`])
//! does the same for that target, and the bank refuses the chunk naming
//! such a value rather than hand it in. A row whose `η` gives no mean under
//! the inverse or power link (`η ≤ 0`) is not predicted and teaches the
//! recursion nothing; it still joins the refit's buffer.
//!
//! **The state** is `k(k+1)/2 + k + 1` numbers a target (`G` packed, `β`,
//! the target's weight) and the shared weight, and until each target's
//! refit runs its buffer of `refit_rows (n_features + 2)` numbers: 35 KB a
//! target at 20 features and 200 rows. Bounded by the model's shape, never
//! by the stream's length. There is no window: the window's sums would be
//! each row's contribution at the fit that scored it, so removing a row
//! exactly needs the window's rows, `(k + 4)` numbers each (384 KB at 20
//! coefficients and 2,000 rows), and no library computes the windowed fit
//! to check it against.

use serde::{Deserialize, Serialize};

use crate::family::{Family, GlmFamily, GlmLink};
use crate::model::{ModelState, OnlineModel, State, StateError, Step, check_schema};
use crate::solve::dot_aug;
use crate::{Decay, RefitStatus, Stamp};

#[cfg(test)]
mod mutant_tests;
#[cfg(test)]
mod refit_clock_tests;
#[cfg(test)]
mod tests;

/// The most IRLS passes a refit takes; one that takes them all is counted
/// in the model's `solve_failures`. Run to its rounding ([`refit_fit`]),
/// a refit of the default 210 rows at 20 features took at most 7 passes
/// for the Poisson, 33 for the Tweedie at `p = 1.5` and 75 for the Gamma
/// under the log link, whose Fisher scoring converges only linearly (50
/// seeds each, 2026-11-22; 9, 24 and 41 while it stopped at a step of
/// 1e-11).
pub const GLM_REFIT_PASSES: u32 = 100;

/// The step cap a target's rows are held to while its refit waits, below
/// `step_cap` (the module docs): the cap the study found to hold every seed.
pub const WAITING_CAP: f64 = 1.0;

/// The least of a spec's default `refit_rows` ([`default_glm_refit_rows`]):
/// the count the study measured at 5 and 20 coefficients.
pub const DEFAULT_GLM_REFIT_ROWS: u32 = 200;

/// The most bytes one group's refit buffers may hold, all its targets
/// together: 256 KB. The buffers are the one part of the state whose size
/// is not the model's shape alone, and their size is known from the
/// settings, `8 · targets · refit_rows · (features + 2)` bytes
/// ([`GlmCfg::refit_bytes`]), so a configuration past it is refused as it
/// is built, never found as the rows run.
pub const GLM_GROUP_REFIT_CAP: usize = 256 * 1024;

/// A spec's default `refit_rows` for `n_features` features and
/// `n_targets` targets: `max(200, 10 k)`, `k` the coefficients with the
/// intercept, held to the rows [`GLM_GROUP_REFIT_CAP`] holds; and 0 --
/// off, the step cap standing in -- past
/// [`crate::DEFAULT_REFIT_MAX_FEATURES`] features, where a refit of `10 k`
/// rows costs seconds a target, or where the cap holds fewer than `2 k`
/// rows, too few to determine the fit.
pub fn default_glm_refit_rows(n_features: usize, n_targets: usize) -> u32 {
    default_glm_refit_rows_within(n_features, n_targets, GLM_GROUP_REFIT_CAP as u64)
}

/// [`default_glm_refit_rows`] held to a group's `max_state_bytes` of `cap`
/// in place of [`GLM_GROUP_REFIT_CAP`] (docs/PLAN.md task 538).
pub fn default_glm_refit_rows_within(n_features: usize, n_targets: usize, cap: u64) -> u32 {
    if n_features > crate::DEFAULT_REFIT_MAX_FEATURES {
        return 0;
    }
    let k = n_features as u32 + 1;
    let fits =
        usize::try_from(cap).unwrap_or(usize::MAX) / (8 * n_targets.max(1) * (n_features + 2));
    let rows = DEFAULT_GLM_REFIT_ROWS
        .max(10 * k)
        .min(u32::try_from(fits).unwrap_or(u32::MAX));
    if rows < 2 * k { 0 } else { rows }
}

/// The step cap a spec without one gets: off where the refit runs, 1 where
/// it does not (the study's cap 1 held all 50 seeds; cap 2 left one Poisson
/// seed 4.2 standard errors off).
pub fn default_glm_step_cap(refit_rows: u32) -> f64 {
    if refit_rows > 0 { f64::INFINITY } else { 1.0 }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GlmCfg {
    pub n_features: usize,
    pub n_targets: usize,
    pub fit_intercept: bool,
    pub decay: Decay,
    pub family: GlmFamily,
    pub link: GlmLink,
    /// The Tweedie family's power `p`; read by no other family.
    #[serde(with = "crate::humanfloat::f64_or_tag")]
    pub var_power: f64,
    /// `R`, the penalty `R/2 |β|²` on the sum scale (the module docs):
    /// above 0.
    pub ridge: f64,
    #[serde(with = "crate::humanfloat::f64_or_tag")]
    pub min_weight: f64,
    /// Each target's first learned rows fitted to convergence, once; 0 is
    /// off.
    pub refit_rows: u32,
    /// The most a learned row may move its own `η`; `inf` is no cap.
    #[serde(with = "crate::humanfloat::f64_or_tag")]
    pub step_cap: f64,
    /// The refit over a span of the clock in place of a count of rows: each
    /// target's rows from its first learned row through the first at least
    /// this many clock units after it (seconds on a temporal clock), once
    /// they hold a row per coefficient. `refit_rows` is 0 beside it.
    /// Last, and written only when set, so a state without it keeps its
    /// bytes and the array form its order.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub refit_clock: Option<f64>,
    /// The bytes a group's refit buffers may hold, a count's checked as the
    /// model is built and a span's as they fill: `None` is
    /// [`GLM_GROUP_REFIT_CAP`], at most 64 MiB (docs/PLAN.md task 538). Last,
    /// and written only when set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_state_bytes: Option<u64>,
}

impl GlmCfg {
    pub fn k_total(&self) -> usize {
        self.n_features + usize::from(self.fit_intercept)
    }

    /// Whether the first rows are refit, after a count or a span.
    pub fn refits(&self) -> bool {
        self.refit_rows > 0 || self.refit_clock.is_some()
    }

    /// The bytes a group's refit buffers may hold: `max_state_bytes`, or
    /// [`GLM_GROUP_REFIT_CAP`], 256 KB.
    pub fn cap(&self) -> u64 {
        self.max_state_bytes.unwrap_or(GLM_GROUP_REFIT_CAP as u64)
    }

    /// The most bytes one group's refit buffers hold: `refit_rows` rows of
    /// `n_features + 2` numbers for each target.
    pub fn refit_bytes(&self) -> usize {
        8usize
            .saturating_mul(self.n_targets)
            .saturating_mul(self.refit_rows as usize)
            .saturating_mul(self.n_features + 2)
    }

    /// Whether one group's refit buffers fit [`GLM_GROUP_REFIT_CAP`]: the
    /// refusal names the bytes and the most `refit_rows` that fits.
    pub fn check_refit_cap(&self) -> Result<(), String> {
        let cap = usize::try_from(self.cap()).unwrap_or(usize::MAX);
        if self.refit_bytes() <= cap {
            return Ok(());
        }
        let fits = cap / (8 * self.n_targets.max(1) * (self.n_features + 2));
        Err(format!(
            "glm: refit_rows = {} holds {} rows of {} numbers for each of {}, {} bytes \
                 a group, past the cap of {cap} bytes a group's refit may hold \
                 (max_state_bytes). Set refit_rows to at most {fits}, to 0 for the step cap \
                 alone, or raise max_state_bytes.",
            self.refit_rows,
            self.refit_rows,
            self.n_features + 2,
            crate::model::count(self.n_targets, "target"),
            self.refit_bytes(),
        ))
    }

    pub fn family(&self) -> Family {
        Family {
            family: self.family,
            link: self.link,
            power: self.var_power,
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        let mut no = crate::Refusals::new();
        no.take(self.decay.check().map_err(|e| format!("glm: {e}")));
        no.take(crate::model::check_dims(
            "glm",
            self.n_features,
            self.n_targets,
        ));
        no.take(self.family().check("glm"));
        no.take(crate::model::check_min_weight("glm", self.min_weight));
        if !(self.ridge > 0.0 && self.ridge.is_finite()) {
            no.refuse(format!(
                "glm: ridge must be above 0 and finite, but it is {}. Set ridge to a positive \
                 number, such as the default 1: it keeps every solve defined before the \
                 coefficients are determined.",
                self.ridge
            ));
        }
        if self.step_cap.is_nan() || self.step_cap <= 0.0 {
            no.refuse(format!(
                "glm: step_cap must be above 0, but it is {}. Set it to how far one row may \
                 move its own η, such as 1, or to inf for no cap.",
                self.step_cap
            ));
        }
        no.take(crate::clock_span::check_max_state_bytes(
            "glm",
            self.max_state_bytes,
        ));
        no.take(self.check_refit_cap());
        if let Some(d) = self.refit_clock {
            if !(d.is_finite() && d > 0.0) {
                no.refuse(format!(
                    "glm: refit_span must be a finite number of clock units above 0, but it is \
                     {d}. Give the span of clock whose rows the refit fits, or leave it out \
                     for refit_rows."
                ));
            }
            if self.refit_rows > 0 {
                no.refuse(format!(
                    "glm: refit_span and refit_rows = {} are two refits of the first rows, \
                     and a target is refit once. Leave one of them out.",
                    self.refit_rows
                ));
            }
        }
        if !self.link.defined_at_zero() && (!self.refits() || !self.fit_intercept) {
            no.refuse(format!(
                "glm: the {} link needs refit_rows above 0 and fit_intercept, but {}. Its \
                 mean is positive only where η is above 0, so the \
                 recursion's own start, every coefficient 0, gives none; the refit starts \
                 from the intercept at the link of the targets' mean. Keep the refit and \
                 the intercept, or use the log link.",
                self.link.name(),
                if !self.refits() {
                    "refit_rows is 0"
                } else {
                    "fit_intercept is off"
                }
            ));
        }
        no.finish()
    }
}

/// One target's refit buffer: its learned rows, `n_features + 2` numbers
/// each -- the features, the target and the row's weight decayed to now --
/// until the refit runs or is given up, and how it went.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct GlmRefit {
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

impl GlmRefit {
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
    /// numbers while it waits, fewer than a count of them, none once it is
    /// done (a span of the clock holds the rows its span has).
    fn fits(&self, width: usize, refit_rows: u32, clocked: bool) -> bool {
        let n = if self.done { 0 } else { self.learned as usize };
        self.rows.len() == n * width
            && (self.done || clocked || self.learned < refit_rows.max(1))
            && (clocked || self.first.is_none())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Glm {
    cfg: GlmCfg,
    w_sum: f64,
    /// Per target, the weight of the rows that carried it in the family's
    /// support, decayed: what its `min_weight` is checked against.
    w_target: Vec<f64>,
    /// Per target, `G`'s lower triangle packed row by row.
    gram: Vec<Vec<f64>>,
    beta: Vec<Vec<f64>>,
    /// Per target with the refit on; empty with it off.
    refit: Vec<GlmRefit>,
    /// Solves that failed (`G + R I` not positive definite in floating
    /// point), and refits that failed or ran to their pass cap.
    pub(crate) solve_failures: u64,
    /// Under a refit over a span of the clock: the clock summed from the
    /// rows' steps, a row's place where no stamp is handed.
    #[serde(default, skip_serializing_if = "is_zero")]
    own: f64,
    /// The bytes a row would have taken the refit buffers to, past
    /// [`GLM_GROUP_REFIT_CAP`]: from then on no row is learned.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    over: Option<u64>,
    /// The half-of-the-cap warning has been raised (once a group).
    #[serde(default, skip_serializing_if = "is_false")]
    warned: bool,
    /// The next row's stamp, and a warning raised and not yet taken. Never
    /// state.
    #[serde(skip)]
    next: Option<Stamp>,
    #[serde(skip)]
    warning: Option<u64>,
    #[serde(skip)]
    work: Work,
}

fn is_zero(v: &f64) -> bool {
    *v == 0.0
}

fn is_false(v: &bool) -> bool {
    !*v
}

/// The scratch a row writes: not state.
#[derive(Debug, Clone, Default)]
struct Work {
    /// `x̃`.
    x: Vec<f64>,
    /// A Cholesky factor, packed as `G`.
    l: Vec<f64>,
    /// The new `G` of a learned row, before it is known finite.
    g: Vec<f64>,
    /// The row's score, then its step.
    s: Vec<f64>,
    d: Vec<f64>,
}

impl PartialEq for Work {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

/// Index of `(i, j)`, `j ≤ i`, in a lower triangle packed row by row.
#[inline]
fn at(i: usize, j: usize) -> usize {
    i * (i + 1) / 2 + j
}

/// Solve `(A + r I) x = b` for `A` symmetric, its lower triangle packed in
/// `a`, by a Cholesky factor written into `l`; `false` where a pivot is not
/// a positive, finite number.
fn solve_ridged(
    a: &[f64],
    r: f64,
    b: &[f64],
    k: usize,
    l: &mut Vec<f64>,
    x: &mut Vec<f64>,
) -> bool {
    l.clear();
    l.resize(a.len(), 0.0);
    for i in 0..k {
        for j in 0..=i {
            let mut s = a[at(i, j)];
            if i == j {
                s += r;
            }
            for m in 0..j {
                s -= l[at(i, m)] * l[at(j, m)];
            }
            if i == j {
                if !(s > 0.0 && s.is_finite()) {
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

impl Glm {
    pub fn new(cfg: GlmCfg) -> Result<Self, String> {
        cfg.validate()?;
        let k = cfg.k_total();
        let m = cfg.n_targets;
        Ok(Self {
            w_sum: 0.0,
            w_target: vec![0.0; m],
            gram: vec![vec![0.0; k * (k + 1) / 2]; m],
            beta: vec![vec![0.0; k]; m],
            refit: if cfg.refits() {
                vec![GlmRefit::new(); m]
            } else {
                Vec::new()
            },
            solve_failures: 0,
            own: 0.0,
            over: None,
            warned: false,
            next: None,
            warning: None,
            work: Work::default(),
            cfg,
        })
    }

    pub fn cfg(&self) -> &GlmCfg {
        &self.cfg
    }

    /// Per target, `[intercept, slopes…]` on the link's scale.
    pub fn coefficients(&self) -> Vec<Vec<f64>> {
        self.beta.clone()
    }

    pub fn n_eff(&self) -> f64 {
        self.w_sum
    }

    /// Per target, the weight of the rows it learned from.
    pub fn target_weights(&self) -> &[f64] {
        &self.w_target
    }

    /// Solves and refits that failed (the field's doc).
    pub fn solve_failures(&self) -> u64 {
        self.solve_failures
    }

    /// How each target's refit stands; empty with the refit off.
    pub fn refit_status(&self) -> Vec<RefitStatus> {
        self.refit.iter().map(GlmRefit::status).collect()
    }

    /// The bytes this model's refit buffers can hold at most: `refit_rows`
    /// rows of `n_features + 2` numbers a target, under a span of the clock
    /// [`GLM_GROUP_REFIT_CAP`]; 0 with the refit off.
    pub fn refit_reservation(&self) -> usize {
        if self.refit.is_empty() {
            return 0;
        }
        if self.cfg.refit_clock.is_some() {
            return usize::try_from(self.cfg.cap()).unwrap_or(usize::MAX);
        }
        self.cfg.n_targets * 8 * self.cfg.refit_rows as usize * (self.cfg.n_features + 2)
    }

    /// The bytes the refit buffers hold, with `extra` more rows: each row's
    /// `n_features + 2` numbers, and under a span of the clock each
    /// target's first stamp.
    pub fn refit_held_bytes(&self, extra: usize) -> u64 {
        let held: usize = self.refit.iter().map(|r| r.rows.len()).sum();
        let stamps = self.refit.len() * std::mem::size_of::<Option<Stamp>>();
        (8 * (held + extra * (self.cfg.n_features + 2)) + stamps) as u64
    }

    /// The run-time check of a refit over a span of the clock, over this
    /// model's fields ([`crate::clock_span::HeldCap`]).
    fn held_cap(&mut self) -> crate::clock_span::HeldCap<'_> {
        crate::clock_span::HeldCap {
            over: &mut self.over,
            warned: &mut self.warned,
            warning: &mut self.warning,
        }
    }

    /// Give every target's refit up before it begins, past a spec's refit
    /// budget: its rows are freed and it keeps the one-pass fit.
    pub fn refit_give_up(&mut self) {
        for r in self.refit.iter_mut().filter(|r| !r.done) {
            r.rows = Vec::new();
            r.done = true;
            r.over_budget = true;
        }
    }

    /// Target `j`'s mean at the fit as it stands, or NaN where the link
    /// gives none.
    fn mean_of(&self, j: usize, x: &[f64]) -> f64 {
        let eta = dot_aug(&self.beta[j], x, self.cfg.fit_intercept);
        self.cfg.family().mean(eta).map_or(f64::NAN, |(_, mu)| mu)
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

    /// Target `j`'s learned row: `G += c x̃x̃'` if the result is finite
    /// (else the row teaches nothing and `false`), then the step `(G + R
    /// I)⁻¹ s x̃`, `s = c (z − η)`, held to the cap. A solve that fails is
    /// counted and moves no coefficient.
    fn learn(&mut self, j: usize, c: f64, s: f64) -> bool {
        let k = self.cfg.k_total();
        let Self {
            gram,
            beta,
            work,
            cfg,
            refit,
            solve_failures,
            ..
        } = self;
        work.g.clear();
        work.g.extend_from_slice(&gram[j]);
        let x = &work.x;
        for i in 0..k {
            let cx = c * x[i];
            for (m, xm) in x.iter().enumerate().take(i + 1) {
                work.g[at(i, m)] += cx * xm;
            }
        }
        if !work.g.iter().all(|v| v.is_finite()) {
            return false;
        }
        std::mem::swap(&mut gram[j], &mut work.g);
        work.s.clear();
        work.s.extend(x.iter().map(|t| s * t));
        if !solve_ridged(&gram[j], cfg.ridge, &work.s, k, &mut work.l, &mut work.d) {
            *solve_failures += 1;
            return true;
        }
        let moved: f64 = x.iter().zip(&work.d).map(|(t, d)| t * d).sum();
        // While the target's refit waits, its rows are held to the start's
        // cap as well: the refit replaces what they leave, and the
        // predictions made before it are then the capped recursion's.
        let waiting = refit.get(j).is_some_and(|r| !r.done);
        let cap = if waiting {
            cfg.step_cap.min(WAITING_CAP)
        } else {
            cfg.step_cap
        };
        let scale = if moved.abs() > cap {
            cap / moved.abs()
        } else {
            1.0
        };
        for (b, d) in beta[j].iter_mut().zip(&work.d) {
            *b += scale * d;
        }
        true
    }

    /// Target `j`'s refit, on the row that filled its buffer (the module
    /// docs).
    fn run_refit(&mut self, j: usize) {
        let nf = self.cfg.n_features;
        let k = self.cfg.k_total();
        let rows = std::mem::take(&mut self.refit[j].rows);
        let fit = refit_fit(&self.cfg, &rows, nf);
        let r = &mut self.refit[j];
        r.done = true;
        match fit {
            Some((b, passes, converged)) => {
                r.passes = passes;
                if !converged {
                    self.solve_failures += 1;
                }
                let fam = self.cfg.family();
                let off = usize::from(self.cfg.fit_intercept);
                let g = &mut self.gram[j];
                g.fill(0.0);
                let x = &mut self.work.x;
                for row in rows.chunks_exact(nf + 2) {
                    x[off..].copy_from_slice(&row[..nf]);
                    let eta = dot_aug(&b, &row[..nf], self.cfg.fit_intercept);
                    let Some(wk) = fam.working(eta, row[nf]) else {
                        continue;
                    };
                    let c = row[nf + 1] * wk.w;
                    for i in 0..k {
                        for m in 0..=i {
                            g[at(i, m)] += c * x[i] * x[m];
                        }
                    }
                }
                self.beta[j] = b;
            }
            None => {
                r.failed = true;
                self.solve_failures += 1;
            }
        }
    }
}

/// The penalized objective `Σ v d(y, μ)/2 + R/2 |b|²` over a buffer's rows,
/// `inf` where a row's `η` gives no mean.
fn objective(cfg: &GlmCfg, fam: &Family, rows: &[f64], nf: usize, b: &[f64]) -> f64 {
    let mut f = 0.0;
    for row in rows.chunks_exact(nf + 2) {
        let eta = dot_aug(b, &row[..nf], cfg.fit_intercept);
        let Some((_, mu)) = fam.mean(eta) else {
            return f64::INFINITY;
        };
        f += 0.5 * row[nf + 1] * fam.unit_deviance(row[nf], mu);
    }
    f + 0.5 * cfg.ridge * b.iter().map(|v| v * v).sum::<f64>()
}

/// `score(b)·d`, the score `Σ v W (z − η) x̃ − R b` at `b` read along
/// `d`: the objective's slope along `−d`. `None` where a row's `η` gives
/// no mean.
fn score_along(
    cfg: &GlmCfg,
    fam: &Family,
    rows: &[f64],
    nf: usize,
    b: &[f64],
    d: &[f64],
) -> Option<f64> {
    let mut s = -cfg.ridge * b.iter().zip(d).map(|(u, v)| u * v).sum::<f64>();
    for row in rows.chunks_exact(nf + 2) {
        let eta = dot_aug(b, &row[..nf], cfg.fit_intercept);
        let wk = fam.working(eta, row[nf])?;
        s += row[nf + 1] * wk.w * (wk.z - wk.eta) * dot_aug(d, &row[..nf], cfg.fit_intercept);
    }
    Some(s)
}

/// The halvings of a step the line search tries: from 1 to below `1e-12`,
/// `2⁻⁴⁰`. A count rather than a test of the step's size, so that a line
/// search that stopped halving would end, wrongly, rather than never.
const HALVINGS: u32 = 40;

/// How small a change of the objective, relative to `1 + |F|`, the line
/// search reads from the slopes rather than from `F` itself (the
/// [`refit_fit`] docs).
const UNSEEN: f64 = 1e-10;

/// A buffer's rows fitted by penalized IRLS with step halving, from the
/// intercept at the link of the targets' weighted mean: the fit, the
/// passes it took, and whether it converged. `None` where the start or a
/// pass is not finite.
///
/// Each pass solves `(G + R I) d = s` for the Fisher step `d` (`s` the
/// score) and halves `t` from 1 until `b + t d` does not raise the
/// objective `F`. Two things keep the answer the same on every platform,
/// whose `exp`, `ln` and `powf` differ in their last bits:
///
/// - **a change too small for `F` to show is read from the slopes.** Where
///   the step's predicted fall `t s'd (1 − t/2)` is under `1e-10 (1 + |F|)`,
///   `F`'s own rounding (its terms are far larger than their sum near the
///   fit) can hide or invert it, and a comparison of `F` would decide the
///   step by its last bits. There the step stands when the trapezoid rule
///   on the slopes at its two ends, exact for a quadratic, says `F` did not
///   rise: `s(b)'d + s(b + t d)'d ≥ 0`.
/// - **the fit runs to its rounding.** Fisher scoring converges only
///   linearly where the link is not the family's canonical one (the Gamma's
///   log link, 0.6 a pass in a fixture here), so a stop at a fixed step size
///   leaves the fit that step's multiple from its fixed point, and which
///   pass first passes the test is a last-bit matter: on 2026-11-21 such a
///   stop moved a Gamma fit's prediction by 2e-8 between macOS and x86-64.
///   It converges once the full step is at the fit's rounding, `4ε (1 +
///   max|b|)`; or under `1e-11 (1 + max|b|)` and no smaller than the pass
///   before's, which a contracting iteration reaches only at its rounding;
///   or once no step does not raise `F`.
fn refit_fit(cfg: &GlmCfg, rows: &[f64], nf: usize) -> Option<(Vec<f64>, u32, bool)> {
    let fam = cfg.family();
    let k = cfg.k_total();
    let off = usize::from(cfg.fit_intercept);
    let (mut sv, mut svy) = (0.0, 0.0);
    for row in rows.chunks_exact(nf + 2) {
        sv += row[nf + 1];
        svy += row[nf + 1] * row[nf];
    }
    let mut b = vec![0.0; k];
    if off == 1 && sv > 0.0 {
        b[0] = fam.start_eta(svy / sv);
    }
    let mut f = objective(cfg, &fam, rows, nf, &b);
    if !f.is_finite() {
        return None;
    }
    let (mut hess, mut grad, mut l, mut d) = (
        vec![0.0; k * (k + 1) / 2],
        vec![0.0; k],
        Vec::new(),
        Vec::new(),
    );
    let mut x = vec![1.0; k];
    let mut trial = vec![0.0; k];
    let mut last = f64::INFINITY;
    for pass in 1..=GLM_REFIT_PASSES {
        hess.fill(0.0);
        for (g, bi) in grad.iter_mut().zip(&b) {
            *g = -cfg.ridge * bi;
        }
        for row in rows.chunks_exact(nf + 2) {
            x[off..].copy_from_slice(&row[..nf]);
            let eta = dot_aug(&b, &row[..nf], cfg.fit_intercept);
            let wk = fam.working(eta, row[nf])?;
            let c = row[nf + 1] * wk.w;
            let s = c * (wk.z - wk.eta);
            for i in 0..k {
                for m in 0..=i {
                    hess[at(i, m)] += c * x[i] * x[m];
                }
                grad[i] += s * x[i];
            }
        }
        if !solve_ridged(&hess, cfg.ridge, &grad, k, &mut l, &mut d) {
            return None;
        }
        let size = b.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
        let full = d.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
        let floor = full <= 4.0 * f64::EPSILON * (1.0 + size);
        if floor || (full <= 1e-11 * (1.0 + size) && full >= last) {
            return Some((b, pass, true));
        }
        last = full;
        // `s'd = d'(G + R I)d`: the objective's fall along `d` at `t = 0`.
        let slope: f64 = grad.iter().zip(&d).map(|(g, v)| g * v).sum();
        let unseen = UNSEEN * (1.0 + f.abs());
        let mut t = 1.0;
        let mut fresh = f;
        let mut fell = false;
        for _ in 0..HALVINGS {
            for i in 0..k {
                trial[i] = b[i] + t * d[i];
            }
            fresh = objective(cfg, &fam, rows, nf, &trial);
            fell = if t * slope * (1.0 - 0.5 * t) > unseen {
                fresh <= f
            } else {
                fresh.is_finite()
                    && score_along(cfg, &fam, rows, nf, &trial, &d)
                        .is_some_and(|at_t| slope + at_t >= 0.0)
            };
            if fell {
                break;
            }
            t *= 0.5;
        }
        if !fell {
            return Some((b, pass, true));
        }
        b.copy_from_slice(&trial);
        f = fresh;
    }
    Some((b, GLM_REFIT_PASSES, false))
}

impl OnlineModel for Glm {
    fn target_n_eff_into(&self, out: &mut Vec<f64>) -> bool {
        out.clear();
        out.extend_from_slice(&self.w_target);
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
        self.ensure_work();
        let m = self.cfg.n_targets;
        let nf = self.cfg.n_features;
        let lam = self.cfg.decay.factor(d_clock);
        let off = usize::from(self.cfg.fit_intercept);
        let fam = self.cfg.family();
        self.work.x[off..].copy_from_slice(x);
        let n_eff = self.w_sum;
        out.pred.clear();
        out.pred.resize(m, f64::NAN);
        // A refit over a span of the clock: the row's place on it, the
        // stamp handed or the clock summed from the steps; and the rows it
        // would add to the buffers, checked against the cap before any is
        // learned.
        let handed = self.next.take();
        let now = self.cfg.refit_clock.map(|_| {
            self.own += d_clock;
            handed.unwrap_or(Stamp::from(self.own))
        });
        if now.is_some() && weight > 0.0 {
            let joining = self
                .refit
                .iter()
                .zip(y)
                .filter(|(r, t)| !r.done && t.is_some_and(|t| fam.takes(t)))
                .count();
            let with = self.refit_held_bytes(joining);
            let cap = self.cfg.cap();
            if !self.held_cap().admit(with, cap) {
                for (j, p) in out.pred.iter_mut().enumerate() {
                    if self.w_target[j] >= self.cfg.min_weight {
                        *p = self.mean_of(j, x);
                    }
                }
                out.n_eff = n_eff;
                out.extra = None;
                return;
            }
        }
        let p = self.cfg.k_total();
        for (j, &yj) in y.iter().enumerate().take(m) {
            let eta = dot_aug(&self.beta[j], x, self.cfg.fit_intercept);
            if self.w_target[j] >= self.cfg.min_weight {
                out.pred[j] = fam.mean(eta).map_or(f64::NAN, |(_, mu)| mu);
            }
            // The clock first: `G` decays on every row.
            let decayed = lam != 1.0;
            if decayed {
                self.gram[j].iter_mut().for_each(|v| *v *= lam);
            }
            let teaching = yj.filter(|&t| weight > 0.0 && fam.takes(t));
            if let Some(wk) = teaching.and_then(|t| fam.working(eta, t)) {
                let c = weight * wk.w;
                self.learn(j, c, c * (wk.z - wk.eta));
            }
            // The refit's buffer ages with the sums, and keeps every row
            // that carried the target in the family's support, whether or
            // not the fit before it gave a mean.
            if let Some(r) = self.refit.get_mut(j).filter(|r| !r.done) {
                if decayed {
                    for row in r.rows.chunks_exact_mut(nf + 2) {
                        row[nf + 1] *= lam;
                    }
                }
                if let Some(t) = teaching {
                    r.rows.extend_from_slice(x);
                    r.rows.push(t);
                    r.rows.push(weight);
                    r.learned += 1;
                    let due = match (self.cfg.refit_clock, now) {
                        (Some(span), Some(now)) => {
                            let first = *r.first.get_or_insert(now);
                            r.learned as usize >= p.max(1)
                                && crate::clock_span::span_reached(now, first, span)
                        }
                        _ => r.learned >= self.cfg.refit_rows,
                    };
                    if due {
                        self.run_refit(j);
                    }
                }
            }
        }
        self.w_sum = lam * self.w_sum + weight;
        crate::model::age_target_weights(
            &mut self.w_target,
            |j| y[j].is_some_and(|t| fam.takes(t)),
            lam,
            weight,
        );
        out.n_eff = n_eff;
        out.extra = None;
        if now.is_some() {
            let with = self.refit_held_bytes(0);
            let cap = self.cfg.cap();
            self.held_cap().note(with, cap);
        }
    }

    /// A refit over a span of the clock learns its rows' stamps: they
    /// decide where it closes.
    fn stamp_next(&mut self, stamp: Stamp) {
        if self.cfg.refit_clock.is_some() {
            self.next = Some(stamp);
        }
    }

    fn take_state_warning(&mut self) -> Option<(u64, u64)> {
        self.warning.take().map(|bytes| (bytes, self.cfg.cap()))
    }

    fn state_over_cap(&self) -> Option<(u64, u64)> {
        self.over.map(|bytes| (bytes, self.cfg.cap()))
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
        let pred = (0..self.cfg.n_targets)
            .map(|j| {
                if self.w_target[j] >= self.cfg.min_weight {
                    self.mean_of(j, x)
                } else {
                    f64::NAN
                }
            })
            .collect();
        Step {
            pred,
            n_eff: self.w_sum,
            extra: None,
        }
    }

    fn state(&self) -> State {
        State::new(ModelState::Glm(Box::new(self.clone())))
    }

    fn restore(s: &State) -> Result<Self, StateError> {
        check_schema(s)?;
        match &s.model {
            ModelState::Glm(m) => {
                let m = (**m).clone();
                crate::model::check_cfg("glm", m.cfg.validate())?;
                let (n, k) = (m.cfg.n_targets, m.cfg.k_total());
                let width = m.cfg.n_features + 2;
                let shaped = m.w_target.len() == n
                    && m.gram.len() == n
                    && m.gram.iter().all(|g| g.len() == k * (k + 1) / 2)
                    && m.beta.len() == n
                    && m.beta.iter().all(|b| b.len() == k)
                    && m.refit.len() == if m.cfg.refits() { n } else { 0 }
                    && m.refit
                        .iter()
                        .all(|r| r.fits(width, m.cfg.refit_rows, m.cfg.refit_clock.is_some()));
                if !shaped {
                    return Err(crate::model::damaged(
                        "glm",
                        "its running sums have the wrong shape",
                    ));
                }
                Ok(m)
            }
            other => Err(StateError::WrongModel {
                expected: "glm",
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
