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
//! **Given states** (`means` and `covs`) put the prior on what was given.
//! With `G_s` the covariance given for state `s`, its prior is that matrix,
//! each feature rescaled by how far the rows have moved from it:
//!
//! ```text
//! Ψ_s = diag(√ψ)·G_s·diag(√ψ)
//! eᵢ  = Σₛ pₛ·(xᵢ − μₛᵢ)²/G_s,ii        the row's squared deviations, each in
//!                                        units of its state's given variance
//! ```
//!
//! with `ŝ` tracked from `e` as above and started at 1, so `Ψ_s = G_s` on
//! the first row and for as long as the rows have the given variances,
//! whatever share of the rows each state holds. A state with no data reads
//! the covariance it was given, at the scale the rows have now, where the
//! diagonal prior read one matrix for every state: regimes that differ in
//! correlation or in variance alone would be told apart by nothing but
//! their data, and a state absent for a while was read as the others are.
//! Given covariances that are the same multiple of the identity for every
//! state are the diagonal prior exactly (`ψᵢ·g` is the pooled variance, and
//! the floor reads every feature alike), so no shape is kept for them
//! ([`Shape::of`]). The same diagonal with variances that differ keeps one:
//! as variances, `ψ`'s floor would lift a variance more than `2¹⁰⁰` below
//! the largest, where in units of the given ones every feature reads 1 on
//! the first row. In the factor form the factor holds
//! `W·M + diag(√f)·G_s·diag(√f)`, and a move adds the difference of two
//! such matrices, `d²/2` sums a state beside the factorization's `d³/3`
//! products; the prior as factored is the target with each feature's scale
//! within `√(1 ± τ)` of it.
//!
//! The state is `ŝ`, `V` and, in the factor form, `f`: `2d + 1` numbers a
//! model there, `d + 1` elsewhere, and under a shape a flag that says so.
//! `ψ`, the slow decay, `P₀` and the shape are derived from `ŝ` and the
//! configuration when a state is loaded. A state without the flag, written
//! by a build that kept no shape for the same diagonal in every state,
//! loads without one there and keeps its variances.

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

/// The shape of each state's prior where the states were given: the given
/// covariances (the module docs, "Given states"). Derived from the
/// configuration, never written.
#[derive(Clone, Debug)]
pub(crate) struct Shape {
    /// Each state's matrix, its lower triangle packed row by row
    /// ([`crate::LowerTriangle::packed`]'s order); empty under `diag`, which
    /// reads the diagonals alone.
    tri: Vec<f64>,
    /// Each state's diagonal, `K·d`.
    diag: Vec<f64>,
    d: usize,
    /// Whether every state's matrix is the same diagonal one, which a build
    /// before kept no shape for: a state it saved holds `ŝ` as variances,
    /// and loads without the shape ([`Stabilized::rebuild`]).
    same_diagonal: bool,
}

impl Shape {
    /// The shape `covs` gives (`K` matrices, each `d·d` row-major, positive
    /// definite: the configuration's check), read as its diagonals alone
    /// where `diagonal`. `None` where every state's matrix is the same
    /// multiple of the identity: that is the diagonal prior exactly, which
    /// needs none. The same diagonal with variances that differ keeps one:
    /// without it `ψ` is the variances themselves, and its floor, `2⁻¹⁰⁰`
    /// of the largest, lifts one more than `2¹⁰⁰` below another (review
    /// 2026-11-23, HMM-3: `diag(1e-20, 1e20)` read `7.9e-11` on the first
    /// row).
    pub(crate) fn of(covs: &[f64], k: usize, d: usize, diagonal: bool) -> Option<Self> {
        debug_assert_eq!(covs.len(), k * d * d);
        let at = |s: usize, i: usize, j: usize| covs[s * d * d + i * d + j];
        let same_diagonal = (0..k).all(|s| {
            (0..d).all(|i| {
                at(s, i, i) == at(0, i, i) && (diagonal || (0..i).all(|j| at(s, i, j) == 0.0))
            })
        });
        if same_diagonal && (0..d).all(|i| at(0, i, i) == at(0, 0, 0)) {
            return None;
        }
        let mut diag = Vec::with_capacity(k * d);
        let mut tri = Vec::new();
        for s in 0..k {
            for i in 0..d {
                diag.push(at(s, i, i));
                if !diagonal {
                    tri.extend((0..=i).map(|j| at(s, i, j)));
                }
            }
        }
        Some(Self {
            tri,
            diag,
            d,
            same_diagonal,
        })
    }

    /// Shape of given spherical variances (`k` positive scalars). `None` when
    /// every state starts at the same scalar variance (the unshaped scalar
    /// prior).
    pub(crate) fn of_spherical(covs: &[f64], k: usize) -> Option<Self> {
        debug_assert_eq!(covs.len(), k);
        if covs.iter().all(|&v| v == covs[0]) {
            return None;
        }
        Some(Self {
            tri: Vec::new(),
            diag: covs.to_vec(),
            d: 1,
            same_diagonal: false,
        })
    }

    /// State `s`'s matrix, its lower triangle packed row by row; empty
    /// under `diag`, which keeps none.
    pub(crate) fn tri(&self, s: usize) -> &[f64] {
        let n = crate::ewcov::row_at(self.d);
        self.tri.get(s * n..(s + 1) * n).unwrap_or(&[])
    }

    /// State `s`'s diagonal.
    pub(crate) fn diag(&self, s: usize) -> &[f64] {
        &self.diag[s * self.d..(s + 1) * self.d]
    }

    /// Whether a build before this one kept this shape too: every shape but
    /// the same diagonal in every state, which it read as the diagonal
    /// prior.
    pub(crate) fn kept_before(&self) -> bool {
        !self.same_diagonal
    }
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
    /// Each state's prior's shape, where the states were given (derived).
    shape: Option<Shape>,
    /// Whether `ŝ` is in units of each state's given covariance: a shape
    /// is kept. Written as such; read back, it says which shape
    /// [`Self::rebuild`] takes.
    shaped: bool,
    /// Working space: the row's deviations, and every coordinate's move.
    pub(crate) e: Vec<f64>,
    add: Vec<f64>,
    moves: Vec<f64>,
    /// Working space under a shape: `√f` before a move and after it.
    roots: (Vec<f64>, Vec<f64>),
    /// A test's exact recursion: read through co-moments at every width.
    #[cfg(test)]
    pub(crate) comoments: bool,
}

/// How [`Stabilized`] is written: `ŝ`, `V`, `f` and, where a shape is
/// kept, `shaped`.
#[derive(Serialize, Deserialize)]
struct Saved {
    scale: Vec<f64>,
    scale_weight: f64,
    factored_prior: Vec<f64>,
    /// Written, and last, only where a shape is kept, so a state without
    /// one keeps its bytes. A state without it was written either with no
    /// shape or by a build that kept none for the same diagonal in every
    /// state, and loads as that build read it (review 2026-11-23, HMM-3).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    shaped: bool,
}

impl From<Stabilized> for Saved {
    fn from(st: Stabilized) -> Self {
        Self {
            scale: st.scale,
            scale_weight: st.scale_w,
            factored_prior: st.held,
            // The flag as held, not the shape: a state read and not yet
            // rebuilt holds no shape, and writes back what it read.
            shaped: st.shaped,
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
            shape: None,
            shaped: s.shaped,
            e: Vec::new(),
            add: Vec::new(),
            moves: Vec::new(),
            roots: (Vec::new(), Vec::new()),
            #[cfg(test)]
            comoments: false,
        }
    }
}

/// What a state holds: `P₀`, `ŝ`, `V`, `f` and whether a shape is kept.
/// Everything else is derived from them and the configuration, or is
/// working space.
impl PartialEq for Stabilized {
    fn eq(&self, other: &Self) -> bool {
        self.prior_rows.to_bits() == other.prior_rows.to_bits()
            && self.scale == other.scale
            && self.scale_w.to_bits() == other.scale_w.to_bits()
            && self.held == other.held
            && self.shaped == other.shaped
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
            .field("shaped", &self.shaped)
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
    pub(crate) fn new(
        prior_rows: f64,
        d: usize,
        decay: Decay,
        fallback: f64,
        shape: Option<Shape>,
    ) -> Self {
        let fallback = Self::fallback_of(fallback, shape.as_ref());
        Self {
            prior_rows,
            scale: vec![0.0; d],
            scale_w: 0.0,
            psi: vec![fallback; d],
            held: Vec::new(),
            slow: slow_of(decay),
            slow_at: None,
            fallback,
            shaped: shape.is_some(),
            shape,
            e: Vec::with_capacity(d),
            add: Vec::with_capacity(d),
            moves: Vec::with_capacity(d),
            roots: (Vec::new(), Vec::new()),
            #[cfg(test)]
            comoments: false,
        }
    }

    /// A loaded state's derived parts, from its configuration: `P₀`, the
    /// slow decay, the fallback, the shape and `ψ`. `false` where what was read
    /// cannot be this model's: `ŝ` not `d` numbers, any of `ŝ`, `V` and `f`
    /// not finite or below 0, `f` neither empty nor `d` long, or written
    /// under a shape the configuration does not give.
    ///
    /// `shape` is the configuration's ([`Shape::of`]). A state written
    /// under a shape takes it; one written without takes it only where a
    /// build before this one kept it too ([`Shape::kept_before`]), so a
    /// state that build saved for the same diagonal in every state, whose
    /// `ŝ` is a variance, reads on as it did.
    pub(crate) fn rebuild(
        &mut self,
        prior_rows: f64,
        d: usize,
        decay: Decay,
        fallback: f64,
        shape: Option<Shape>,
    ) -> bool {
        let fine = |v: &f64| v.is_finite() && *v >= 0.0;
        if self.scale.len() != d
            || !(self.held.is_empty() || self.held.len() == d)
            || !self.scale.iter().all(fine)
            || !self.held.iter().all(fine)
            || !fine(&self.scale_w)
            || (self.shaped && shape.is_none())
        {
            return false;
        }
        let shape = shape.filter(|g| self.shaped || g.kept_before());
        self.shaped = shape.is_some();
        self.prior_rows = prior_rows;
        self.slow = slow_of(decay);
        self.slow_at = None;
        self.fallback = Self::fallback_of(fallback, shape.as_ref());
        self.shape = shape;
        self.psi = vec![fallback; d];
        self.refresh_psi();
        true
    }

    /// `ψ` where every `ŝ` is 0: the ridge `precision_prior`, a variance,
    /// where `ψ` is one; and 1 under a shape, where `ψ` is a multiple of
    /// each state's given covariance, so the states read the covariances
    /// they were given (review 2026-11-23, HMM-2: a ridge of `1e-3` read
    /// them a thousand times too narrow).
    fn fallback_of(precision_prior: f64, shape: Option<&Shape>) -> f64 {
        if shape.is_some() {
            1.0
        } else {
            precision_prior
        }
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

    /// The shape of each state's prior, where the states were given.
    pub(crate) fn shape(&self) -> Option<&Shape> {
        self.shape.as_ref()
    }

    /// [`Self::shrink`] under a shape: state `s`, of weight `w`, reads
    /// `m·C + R·G·R`, `R = diag(r)`: `(m, r, G)`, with `m = w/(w + P₀)`,
    /// `rᵢ = √(P₀ψᵢ/(w + P₀))` and `G` the state's shape, its lower triangle
    /// packed row by row (empty with no shape).
    pub(crate) fn shrink_roots(&mut self, w: f64, s: usize) -> (f64, &[f64], &[f64]) {
        let t = w + self.prior_rows;
        self.add.clear();
        let p0 = self.prior_rows;
        self.add
            .extend(self.psi.iter().map(|p| (p0 * p / t).sqrt()));
        let g = self.shape.as_ref().map_or(&[][..], |g| g.tri(s));
        (w / t, &self.add, g)
    }

    /// `ŝ` and `V`. Where the states were given, `ŝ` is in units of each
    /// state's given variance (the module docs, "Given states").
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
        if w > 0.0 && pooled.iter().all(|p| p.is_finite()) {
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
        // A row whose squared deviations leave the doubles (`1e80` from a
        // state given a variance of `1e-150`) would leave `ŝ` infinite, and
        // the next row NaN, a state no load takes: it ages `V` as a row of
        // no weight does (review 2026-11-23, HMM-1).
        let w = if e.iter().all(|v| v.is_finite()) {
            w
        } else {
            0.0
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
        if self.shape.is_some() {
            self.roots.0.clear();
            self.roots.0.extend(self.held.iter().map(|h| h.sqrt()));
        }
        self.moves.clear();
        for (h, p) in self.held.iter_mut().zip(&self.psi) {
            let goal = self.prior_rows * p * lift;
            self.moves.push(goal - *h);
            *h = goal;
        }
        if self.shape.is_some() {
            self.roots.1.clear();
            self.roots.1.extend(self.held.iter().map(|h| h.sqrt()));
        }
    }

    /// Under a shape, `√f` as [`Self::settle_all`] found it and as it left
    /// it: a state's factored prior moved from `diag(old)·G·diag(old)` to
    /// `diag(new)·G·diag(new)`.
    pub(crate) fn roots(&self) -> (&[f64], &[f64]) {
        (&self.roots.0, &self.roots.1)
    }

    /// What [`Self::settle_all`] moved each coordinate by.
    pub(crate) fn moves(&self) -> &[f64] {
        &self.moves
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fresh() -> Stabilized {
        let mut st = Stabilized::new(2.0, 3, Decay::Halflife(5.0), 1e-3, None);
        st.seed(&[1.0, 2.0, 3.0], 4.0);
        st
    }

    /// What a state holds is `P₀`, `ŝ`, `V` and `f`, and it is equal to
    /// another only where all four are: one that differs in any one alone is
    /// not equal.
    #[test]
    fn a_state_is_equal_to_another_in_all_four_of_its_parts() {
        let a = fresh();
        assert_eq!(a, a.clone());
        let mut p0 = a.clone();
        p0.prior_rows = 2.5;
        let mut scale = a.clone();
        scale.scale[1] = 2.5;
        let mut weight = a.clone();
        weight.scale_w = 5.0;
        let mut held = a.clone();
        held.hold_all();
        for (what, b) in [("P₀", &p0), ("ŝ", &scale), ("V", &weight), ("f", &held)] {
            assert_ne!(&a, b, "differs in {what} alone");
            assert_ne!(b, &a, "differs in {what} alone");
        }
        // What is derived or working space is no part of it.
        let mut other = a.clone();
        other.e = vec![1.0];
        other.fallback = 7.0;
        assert_eq!(a, other);
    }

    /// The debug form names what a state holds.
    #[test]
    fn a_state_shows_what_it_holds() {
        let mut st = fresh();
        st.hold_all();
        let text = format!("{st:?}");
        assert!(text.starts_with("Stabilized {"), "{text}");
        for part in [
            "prior_rows: 2.0",
            "scale: [1.0, 2.0, 3.0]",
            "scale_w: 4.0",
            "held: [",
        ] {
            assert!(text.contains(part), "{text}");
        }
    }

    /// A state read from a file is refused where a number of its is not
    /// finite or is below 0, or its sizes are not this model's: each part
    /// alone is enough.
    #[test]
    fn a_loaded_state_is_refused_where_any_one_part_is_damaged() {
        let built = |scale: &[f64], w: f64, held: &[f64]| {
            let mut st = Stabilized::from(Saved {
                scale: scale.to_vec(),
                scale_weight: w,
                factored_prior: held.to_vec(),
                shaped: false,
            });
            st.rebuild(2.0, 3, Decay::Halflife(5.0), 1e-3, None)
        };
        let ok = [1.0, 2.0, 3.0];
        assert!(built(&ok, 4.0, &[]));
        assert!(built(&ok, 0.0, &[1.0, 2.0, 3.0]));
        assert!(built(&[0.0; 3], 0.0, &[0.0; 3]));
        // The sizes.
        assert!(!built(&[1.0, 2.0], 4.0, &[]));
        assert!(!built(&[1.0, 2.0, 3.0, 4.0], 4.0, &[]));
        assert!(!built(&ok, 4.0, &[1.0, 2.0]));
        assert!(!built(&ok, 4.0, &[1.0, 2.0, 3.0, 4.0]));
        // Each number, alone, in each part.
        for bad in [-1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1e-300] {
            assert!(!built(&[1.0, bad, 3.0], 4.0, &[]), "ŝ {bad}");
            assert!(!built(&ok, bad, &[]), "V {bad}");
            assert!(!built(&ok, 4.0, &[1.0, bad, 3.0]), "f {bad}");
        }
    }

    /// `shrink_of` is `shrink` as a function: the same weight on the data and
    /// the same prior beside it, `P₀ψᵢ/(w + P₀)`.
    #[test]
    fn shrink_of_is_shrink() {
        let mut st = fresh();
        for w in [0.0, 0.5, 4.0, 100.0] {
            let (m, add) = st.shrink(w);
            let add = add.to_vec();
            let (m2, f) = st.shrink_of(w);
            assert_eq!(m.to_bits(), m2.to_bits());
            assert_eq!(m, w / (w + 2.0));
            for (i, a) in add.iter().enumerate() {
                assert_eq!(f(i).to_bits(), a.to_bits(), "w {w}, coordinate {i}");
                let want = 2.0 * st.psi()[i] / (w + 2.0);
                assert!((a - want).abs() <= 1e-15 * want, "w {w}, coordinate {i}");
            }
        }
    }
}
