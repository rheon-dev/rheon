//! Stabilized forgetting for an `hmm` state's covariance (review fable2,
//! C4; the study `.claude/coord/review-fable2/C4-STUDY.md`). Not yet an
//! option: [`super::Hmm::with_stabilized`] builds a model under it, for the
//! study and the tests, until the user decides whether it is the default.
//!
//! Exponential forgetting flattens a state's posterior toward nothing, so a
//! state that wins no rows for a while keeps a matrix whose ridge decays
//! with its weight, and on its first row back collapses onto that row.
//! Stabilized forgetting (Kulhavý and Zarrop, "On a general concept of
//! forgetting", Int. J. Control 58(4), 1993) flattens it toward a prior
//! instead: `p̄(θ) ∝ p(θ)^λ · p₀(θ)^(1−λ)`. A normal-inverse-Wishart's
//! statistics are linear, so the forgetting step is `ν̄ = λν + (1−λ)ν₀`,
//! `S̄ = λS + (1−λ)S₀`, and a prior worth `P₀` rows from the start stays
//! worth `P₀` rows: `P' = λP₀ + (1−λ)P₀ = P₀`. With `Ψ = diag(ψ)` the
//! prior's covariance and `W`, `M` the state's weight and mean-form
//! co-moments, the state reads
//!
//! ```text
//! Σ = (W·M + P₀·Ψ) / (W + P₀)
//! ```
//!
//! after the row's decay (`W ← λW`), so a zero-weight row and the gap it
//! sits in read alike. With no data (`W → 0`) `Σ → Ψ`, never 0, so a
//! returning state reads a broad density; at no decay it is scikit-learn
//! QDA's `reg_param` shrinkage, `reg_param = P₀/(n + P₀)`, toward `Ψ`. The
//! means take no prior.
//!
//! **The prior's scale** `ψ` is the pooled within-state variance, tracked
//! at ten times the model's half-life from the row's squared deviations
//! from the states' means before it, weighted by the posterior:
//!
//! ```text
//! eᵢ  = Σₛ pₛ·(xᵢ − μₛᵢ)²
//! V'  = λ_slow·V + w                 λ_slow the decay at 10 half-lives
//! ŝᵢ' = ŝᵢ + (w/V')·(eᵢ − ŝᵢ)
//! ψᵢ  = max(ŝᵢ, 2⁻¹⁰⁰·maxⱼ ŝⱼ)        precision_prior where every ŝ is 0
//! ```
//!
//! seeded from the seeded states' variances, pooled by weight. The floor
//! keeps a feature that never varies from a zero variance (review C2): its
//! deviations are then 0 and read nothing, and one that moves again reads a
//! large but finite form, so the row is learned and `ŝ` follows it.
//!
//! **The factor form** (`full` at [`super::Hmm::FACTOR_FROM`] features and
//! more) keeps `Σ = σ·L Lᵀ` and moves it by rotations, `O(d²)` a row. The
//! data's step is one rank-one step and the decay a scalar, but the prior's
//! share of `Σ` changes with `W`, and adding a diagonal to a factor costs a
//! rank-one step a coordinate, `O(d³)`. So the factor holds `W·M + F` over
//! `W + P₀`, with `F = diag(f)` the prior *as factored*, which decays with
//! the data, `f ← λf`, and is topped up to `P₀ψ` one coordinate at a time
//! (`F += qⱼ·eⱼeⱼᵀ`, a rotation from column `j` on, about `(d − j)²/2`):
//!
//! - after each learned row, coordinate `j = cursor` where it is short of
//!   its target, the cursor moving on;
//! - before each row's read, every coordinate whose `f` is more than `τ`
//!   ([`TAU`]) from `P₀ψ`: below it, so a gap's read stands on its whole
//!   prior, topped up to `1 + τ/2` of it so it decays through the target
//!   before it is due again; or above it, where `ψ` has fallen faster than
//!   `f` decays (after the seeding, or a gap long enough to empty `V`),
//!   taken back to it by a downdate (`F −= qⱼ·eⱼeⱼᵀ`, LINPACK's `dchdd`).
//!
//! On a row of weight 0 the moves before the read serve that read alone and
//! are put back after it, so the row leaves the factors as the gap it sits
//! in would, to rounding (hard rule 9). `f` is the same for every state
//! (they share `λ` and the moves), so it is kept once. The read is off the
//! exact one by `diag(P₀ψ − f)`: within `τ` of a coordinate's prior, and
//! within `1 − λ^d` of it where the cursor comes round first; the prior's
//! share of `Σ` is `P₀/(W + P₀)`. The moves cost `(d − j)²/2` rotations
//! each; at a half-life of `h` rows the band is crossed every `1.5τ·h/ln 2`
//! rows, so about `d·ln 2/(1.5τ·h)` of them a row, beside the cursor's one.
//! The state is `ŝ`, `V`, `f` and the cursor: `2d + 2` numbers a model.

use crate::Decay;

/// A coordinate's factored prior is moved to its target before a read once
/// it is more than `TAU` of it away.
pub(crate) const TAU: f64 = 1.0 / 16.0;

/// The floor of `ψ`, relative to the largest `ŝ`.
const FLOOR: f64 = 7.888609052210118e-31; // 2⁻¹⁰⁰

#[cfg(test)]
thread_local! {
    /// A test's switch: every coordinate topped up before every read, so
    /// the factor form holds the exact recursion, to rounding.
    pub(crate) static EAGER: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// The model's part of stabilized forgetting; see the module docs.
#[derive(Debug, Clone, PartialEq)]
pub struct Stabilized {
    /// `P₀`, the prior's weight in rows.
    pub(crate) prior_rows: f64,
    /// `ŝ`, the tracked within-state variance.
    scale: Vec<f64>,
    /// `V`, its weight.
    scale_w: f64,
    /// `ψ`, `ŝ` floored (derived).
    psi: Vec<f64>,
    /// The factor form's `f`, in the units of `W·M + F`; empty elsewhere.
    pub(crate) held: Vec<f64>,
    /// The next coordinate a learned row tops up.
    cursor: usize,
    /// The decay at ten half-lives.
    slow: Decay,
    /// `(d_clock, λ_slow)` of the last row (derived).
    slow_at: (f64, f64),
    /// The ridge where every `ŝ` is 0.
    fallback: f64,
    /// Working space: the row's deviations, and the coordinates due.
    pub(crate) e: Vec<f64>,
    due: Vec<usize>,
    add: Vec<f64>,
    /// Read through co-moments at every width (the exact recursion).
    pub(crate) comoments: bool,
    /// A study's mean prior: rows of weight at the state's current mean
    /// that each learned row's mean step also divides by; 0 is none.
    pub(crate) mean_rows: f64,
    /// [`TAU`], or a study's own.
    pub(crate) tau: f64,
}

impl Stabilized {
    pub(crate) fn new(prior_rows: f64, d: usize, decay: Decay, fallback: f64) -> Self {
        let slow = match decay {
            Decay::Halflife(h) => Decay::Halflife(10.0 * h),
            Decay::Lam(l) => Decay::Lam(l.powf(0.1)),
        };
        Self {
            prior_rows,
            scale: vec![0.0; d],
            scale_w: 0.0,
            psi: vec![fallback; d],
            held: Vec::new(),
            cursor: 0,
            slow,
            slow_at: (f64::NAN, 1.0),
            fallback,
            e: Vec::with_capacity(d),
            due: Vec::with_capacity(d),
            add: Vec::with_capacity(d),
            comoments: false,
            mean_rows: 0.0,
            tau: TAU,
        }
    }

    /// `ψ`.
    pub(crate) fn psi(&self) -> &[f64] {
        &self.psi
    }

    /// `P₀ψⱼ`, the prior a factored state holds at coordinate `j` when it is
    /// topped up.
    #[inline]
    pub(crate) fn target(&self, j: usize) -> f64 {
        self.prior_rows * self.psi[j]
    }

    /// `P₀ψ`.
    pub(crate) fn targets(&self) -> Vec<f64> {
        self.psi.iter().map(|p| self.prior_rows * p).collect()
    }

    /// A state of weight `w` reads `m·C + diag(add)`: `(m, add)`, with
    /// `m = w/(w + P₀)` and `addᵢ = P₀ψᵢ/(w + P₀)`.
    pub(crate) fn shrink(&mut self, w: f64) -> (f64, &[f64]) {
        let t = w + self.prior_rows;
        self.add.clear();
        let p0 = self.prior_rows;
        self.add.extend(self.psi.iter().map(|p| p0 * p / t));
        (w / t, &self.add)
    }

    /// [`Self::shrink`] without its working space: `addᵢ` as a function.
    pub(crate) fn shrink_of(&self, w: f64) -> (f64, impl Fn(usize) -> f64 + '_) {
        let t = w + self.prior_rows;
        (w / t, move |i| self.prior_rows * self.psi[i] / t)
    }

    /// The due coordinates [`Self::age`] found.
    pub(crate) fn due(&self) -> &[usize] {
        &self.due
    }

    /// `ŝ` and `V`.
    pub fn scale(&self) -> (&[f64], f64) {
        (&self.scale, self.scale_w)
    }

    fn refresh_psi(&mut self) {
        let top = self.scale.iter().cloned().fold(0.0, f64::max);
        if top > 0.0 && top.is_finite() {
            let floor = FLOOR * top;
            for (p, &s) in self.psi.iter_mut().zip(&self.scale) {
                *p = s.max(floor);
            }
        } else {
            self.psi.iter_mut().for_each(|p| *p = self.fallback);
        }
    }

    /// `ŝ` from the seeded states' variances `pooled`, at weight `w`.
    pub(crate) fn seed(&mut self, pooled: &[f64], w: f64) {
        if w > 0.0 {
            self.scale.copy_from_slice(pooled);
            self.scale_w = w;
        }
        self.refresh_psi();
    }

    /// The factor form's start: every coordinate topped up.
    pub(crate) fn hold_all(&mut self) {
        self.held = self.targets();
        self.cursor = 0;
    }

    /// The row's deviations `e` at weight `w` into `ŝ` (none at `w = 0`,
    /// where `V` decays alone), after a clock step of `d_clock`.
    pub(crate) fn observe(&mut self, d_clock: f64, w: f64, e: &[f64]) {
        if d_clock.to_bits() != self.slow_at.0.to_bits() {
            self.slow_at = (d_clock, self.slow.factor(d_clock));
        }
        let v = self.slow_at.1 * self.scale_w + w;
        if v > 0.0 && w > 0.0 {
            let b = w / v;
            for (s, &ei) in self.scale.iter_mut().zip(e) {
                *s += b * (ei - *s);
            }
            self.refresh_psi();
        }
        self.scale_w = v;
    }

    /// The factored prior's decay, `f ← λf`, and the coordinates then due a
    /// move before the read: those more than `τ` from their target, or
    /// every one where `all`, which is handed back ([`Self::settle`]).
    pub(crate) fn age(&mut self, lam: f64, all: bool) -> bool {
        #[cfg(test)]
        let all = all || EAGER.with(std::cell::Cell::get);
        self.due.clear();
        for (j, h) in self.held.iter_mut().enumerate() {
            *h *= lam;
            let t = self.prior_rows * self.psi[j];
            if all || *h < (1.0 - self.tau) * t || *h > (1.0 + self.tau) * t {
                self.due.push(j);
            }
        }
        all
    }

    /// The coordinate a learned row tops up, and the cursor moved on.
    pub(crate) fn next(&mut self) -> usize {
        let j = self.cursor;
        self.cursor = (self.cursor + 1) % self.held.len().max(1);
        j
    }

    /// [`Self::top_up`] where `fⱼ` is short of its target; 0, and `fⱼ` as
    /// it is, where it is at or above it.
    pub(crate) fn top_up_short(&mut self, j: usize) -> f64 {
        if self.held[j] < self.target(j) {
            self.top_up(j)
        } else {
            0.0
        }
    }

    /// Coordinate `j` moved, before a read: to its target where `all` or
    /// where it holds more, else to `1 + τ/2` of it, so it decays through
    /// its target before it is due again (the band `[1 − τ, 1 + τ/2]` is
    /// crossed in `1.5τ/(1 − λ)` rows where `[1 − τ, 1]` took `τ/(1 − λ)`).
    /// What it moved by.
    pub(crate) fn settle(&mut self, j: usize, all: bool) -> f64 {
        let t = self.target(j);
        let h = self.held[j];
        let goal = if all || h > t {
            t
        } else {
            t * (1.0 + 0.5 * self.tau)
        };
        self.held[j] = goal;
        goal - h
    }

    /// What coordinate `j` is short of its target, `P₀ψⱼ − fⱼ` (negative
    /// where `ψ` has fallen below what it holds), and `fⱼ` set to it.
    pub(crate) fn top_up(&mut self, j: usize) -> f64 {
        let t = self.target(j);
        let q = t - self.held[j];
        self.held[j] = t;
        q
    }
}
