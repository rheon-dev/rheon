//! Stabilized forgetting for an `hmm` state's covariance, under
//! `covariance_type` `"full"` and `"diag"` in a model that learns.
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
//! means take no prior. `P₀` is the spec's `prior_rows`, `d` (the number of
//! features) by default: the degrees of freedom scikit-learn's
//! `BayesianGaussianMixture` gives its Wishart prior.
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
//! seeded from the seeded states' variances, pooled by weight (`V` their
//! total weight). The floor keeps a feature that never varies from a zero
//! variance: its deviations are then 0 and read nothing, and one that moves
//! again reads a large but finite form, so the row is learned and `ŝ`
//! follows it.
//!
//! **The factor form** (`full` at [`super::Hmm::FACTOR_FROM`] features and
//! more) keeps `Σ = σ·L Lᵀ` and moves it by rotations, `O(d²)` a row. The
//! data's step is one rank-one step and the decay a scalar, but the prior's
//! share of `Σ` changes with `W`: after a row's decay the matrix is short of
//! `(1 − λ)·P₀Ψ`, a diagonal of full rank, and no `O(d²)` step adds one to a
//! factor. So the factor holds `W·M + F` over `W + P₀`, with `F = diag(f)`
//! the prior *as factored*, which decays with the data, `f ← λf`, and is
//! moved back to `P₀ψ` only when it has drifted: before a row's read,
//! whenever any coordinate's `f` is more than `τ` ([`TAU`]) from `P₀ψ`,
//! every coordinate is moved at once, to `1 + τ/2` of its target (so it
//! decays through the target before it is due again):
//!
//! ```text
//! A   = L Lᵀ + diag(q)/((W + P₀)·σ)     qⱼ = (1 + τ/2)·P₀ψⱼ − fⱼ, either sign
//! L'  = chol(A)                          f ← (1 + τ/2)·P₀ψ
//! ```
//!
//! `L Lᵀ` is laid out in `d³/6` products and factorized in `d³/6` more, both
//! in runs over contiguous columns
//! ([`crate::solve::FactorWork::refactor_shifted`]). Every coordinate decays
//! by the same `λ`, so all fall due together: at a half-life of `h` rows the
//! band `[1 − τ, 1 + τ/2]` is crossed every `1.5τ·h/ln 2` rows (6.8 rows at
//! `h = 50`, 68 at `h = 500`), and that is how often a state is factorized.
//! A coordinate is also due where it holds more than `1 + τ` of its target:
//! `ψ` has fallen faster than `f` decays (after the seeding, or a gap long
//! enough to empty `V`).
//!
//! On a row of weight 0 the move before the read serves that read alone and
//! is put back after it, so the row leaves the factors as the gap it sits
//! in would, to rounding (hard rule 9). `f` is the same for every state
//! (they share `λ` and the moves), so it is kept once. The read is off the
//! exact one by `diag(P₀ψ − f)`, within `τ` of each coordinate's prior; the
//! prior's share of `Σ` is `P₀/(W + P₀)`, so a quadratic form is within a
//! factor `1/(1 − τ)` of the exact recursion's and `ln det` within
//! `d·ln(1 − τ)`, a bound reached only by a state with no data.
//!
//! The state is `ŝ`, `V` and, in the factor form, `f`: `2d + 1` numbers a
//! model there, `d + 1` elsewhere. `ψ`, the slow decay and `P₀` are derived
//! from `ŝ` and the configuration when a state is loaded.

use serde::{Deserialize, Serialize};

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

/// The model's part of stabilized forgetting; see the module docs. Written
/// as `ŝ`, `V` and `f` alone (`Saved`); the rest is derived on load from
/// the configuration (`rebuild`).
#[derive(Clone, Serialize, Deserialize)]
#[serde(into = "Saved", from = "Saved")]
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
    /// The decay at ten half-lives (derived).
    slow: Decay,
    /// `(d_clock, λ_slow)` of the last row, `None` before one (derived).
    slow_at: Option<(f64, f64)>,
    /// The ridge where every `ŝ` is 0 (derived: `precision_prior`).
    fallback: f64,
    /// Working space: the row's deviations, and every coordinate's move.
    pub(crate) e: Vec<f64>,
    add: Vec<f64>,
    moves: Vec<f64>,
    /// A test's exact recursion: read through co-moments at every width.
    #[cfg(test)]
    pub(crate) comoments: bool,
}

/// How [`Stabilized`] is written: `ŝ`, `V` and `f`.
#[derive(Serialize, Deserialize)]
struct Saved {
    scale: Vec<f64>,
    scale_weight: f64,
    factored_prior: Vec<f64>,
}

impl From<Stabilized> for Saved {
    fn from(st: Stabilized) -> Self {
        Self {
            scale: st.scale,
            scale_weight: st.scale_w,
            factored_prior: st.held,
        }
    }
}

/// Read with nothing derived yet: [`Stabilized::rebuild`] derives it from
/// the configuration, which the owner's `restore` holds.
impl From<Saved> for Stabilized {
    fn from(s: Saved) -> Self {
        Self {
            prior_rows: f64::NAN,
            scale: s.scale,
            scale_w: s.scale_weight,
            psi: Vec::new(),
            held: s.factored_prior,
            slow: Decay::Lam(1.0),
            slow_at: None,
            fallback: f64::NAN,
            e: Vec::new(),
            add: Vec::new(),
            moves: Vec::new(),
            #[cfg(test)]
            comoments: false,
        }
    }
}

/// What a state holds: `P₀`, `ŝ`, `V` and `f`. Everything else is derived
/// from them and the configuration, or is working space.
impl PartialEq for Stabilized {
    fn eq(&self, other: &Self) -> bool {
        self.prior_rows.to_bits() == other.prior_rows.to_bits()
            && self.scale == other.scale
            && self.scale_w.to_bits() == other.scale_w.to_bits()
            && self.held == other.held
    }
}

/// As [`PartialEq`] reads it: no working space and nothing derived.
impl std::fmt::Debug for Stabilized {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Stabilized")
            .field("prior_rows", &self.prior_rows)
            .field("scale", &self.scale)
            .field("scale_w", &self.scale_w)
            .field("held", &self.held)
            .finish()
    }
}

/// The slow decay, at ten times the model's half-life.
fn slow_of(decay: Decay) -> Decay {
    match decay {
        Decay::Halflife(h) => Decay::Halflife(10.0 * h),
        Decay::Lam(l) => Decay::Lam(l.powf(0.1)),
    }
}

impl Stabilized {
    pub(crate) fn new(prior_rows: f64, d: usize, decay: Decay, fallback: f64) -> Self {
        Self {
            prior_rows,
            scale: vec![0.0; d],
            scale_w: 0.0,
            psi: vec![fallback; d],
            held: Vec::new(),
            slow: slow_of(decay),
            slow_at: None,
            fallback,
            e: Vec::with_capacity(d),
            add: Vec::with_capacity(d),
            moves: Vec::with_capacity(d),
            #[cfg(test)]
            comoments: false,
        }
    }

    /// A loaded state's derived parts, from its configuration: `P₀`, the
    /// slow decay, the fallback and `ψ`. `false` where what was read
    /// cannot be this model's: `ŝ` not `d` numbers, any of `ŝ`, `V` and `f`
    /// not finite or below 0, or `f` neither empty nor `d` long.
    pub(crate) fn rebuild(
        &mut self,
        prior_rows: f64,
        d: usize,
        decay: Decay,
        fallback: f64,
    ) -> bool {
        let fine = |v: &f64| v.is_finite() && *v >= 0.0;
        if self.scale.len() != d
            || !(self.held.is_empty() || self.held.len() == d)
            || !self.scale.iter().all(fine)
            || !self.held.iter().all(fine)
            || !fine(&self.scale_w)
        {
            return false;
        }
        self.prior_rows = prior_rows;
        self.slow = slow_of(decay);
        self.slow_at = None;
        self.fallback = fallback;
        self.psi = vec![fallback; d];
        self.refresh_psi();
        true
    }

    /// `ψ`.
    pub(crate) fn psi(&self) -> &[f64] {
        &self.psi
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

    /// `ŝ` and `V`.
    pub fn scale(&self) -> (&[f64], f64) {
        (&self.scale, self.scale_w)
    }

    /// `P₀`.
    pub fn prior_rows(&self) -> f64 {
        self.prior_rows
    }

    /// `f`, the prior the factor form holds as factored, in the units of
    /// `W·M + F`; empty in the other forms.
    pub fn factored_prior(&self) -> &[f64] {
        &self.held
    }

    /// Whether the states are read through their co-moments at every width:
    /// a test's exact recursion.
    pub(crate) fn comoments(&self) -> bool {
        #[cfg(test)]
        return self.comoments;
        #[cfg(not(test))]
        false
    }

    fn refresh_psi(&mut self) {
        let top = self.scale.iter().cloned().fold(0.0, f64::max);
        self.floor_psi(top);
    }

    /// `ψ` from `ŝ`, whose largest is `top`.
    fn floor_psi(&mut self, top: f64) {
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

    /// The factor form's start: every coordinate at its target.
    pub(crate) fn hold_all(&mut self) {
        self.held = self.targets();
    }

    /// The row's deviations `e` at weight `w` into `ŝ` (none at `w = 0`,
    /// where `V` decays alone), after a clock step of `d_clock`.
    pub(crate) fn observe(&mut self, d_clock: f64, w: f64, e: &[f64]) {
        let lam = match self.slow_at {
            Some((at, lam)) if at.to_bits() == d_clock.to_bits() => lam,
            _ => {
                let lam = self.slow.factor(d_clock);
                self.slow_at = Some((d_clock, lam));
                lam
            }
        };
        let v = lam * self.scale_w + w;
        if v > 0.0 && w > 0.0 {
            let b = w / v;
            // `ŝ`'s largest in the pass that moves it, for the floor.
            let mut top = 0.0;
            for (s, &ei) in self.scale.iter_mut().zip(e) {
                *s += b * (ei - *s);
                top = f64::max(top, *s);
            }
            self.floor_psi(top);
        }
        self.scale_w = v;
    }

    /// The factored prior's decay, `f ← λf`, and whether a move is due
    /// before the read: some coordinate below `1 − τ` or above `1 + τ` of its
    /// target, or `all` (handed back too).
    pub(crate) fn age(&mut self, lam: f64, all: bool) -> (bool, bool) {
        #[cfg(test)]
        let all = all || EAGER.with(std::cell::Cell::get);
        let mut due = false;
        for (h, p) in self.held.iter_mut().zip(&self.psi) {
            *h *= lam;
            let t = self.prior_rows * p;
            due |= *h < (1.0 - TAU) * t || *h > (1.0 + TAU) * t;
        }
        (all, due || all)
    }

    /// Every coordinate moved at once, before a read: to its target where
    /// `all`, else to `1 + τ/2` of it, so it decays through its target
    /// before it is due again (the band `[1 − τ, 1 + τ/2]` is crossed in
    /// `1.5τ/(1 − λ)` rows where `[1 − τ, 1]` took `τ/(1 − λ)`). What each
    /// moved by, of either sign, is [`Self::moves`].
    pub(crate) fn settle_all(&mut self, all: bool) {
        let lift = if all { 1.0 } else { 1.0 + 0.5 * TAU };
        self.moves.clear();
        for (h, p) in self.held.iter_mut().zip(&self.psi) {
            let goal = self.prior_rows * p * lift;
            self.moves.push(goal - *h);
            *h = goal;
        }
    }

    /// What [`Self::settle_all`] moved each coordinate by.
    pub(crate) fn moves(&self) -> &[f64] {
        &self.moves
    }
}
