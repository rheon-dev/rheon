//! Study C4 (review fable2, rust-core.md C4): a prototype, compiled only with
//! the `study-c4` feature, of two alternatives to the ridge that decays with
//! a state's (`hmm`) or a class's (`ew_class`) own data. The default build
//! does not contain this module, and no model reads it unless a variant is
//! set on it ([`crate::Hmm::set_study`], [`crate::EwClass::set_study`]).
//!
//! Notation, per state (or class) `s`: `W` its data weight (`n_eff`), `M`
//! its mean-form co-moments, `r_A = precision_prior · π` today's ridge,
//! where `π` decays by the update's `a` (variant A). Per model and feature
//! `i`: `ŝ_i`, a slowly tracked within-state variance (below).
//!
//! **A (today).** `Σ_s = M + r_A·I`.
//!
//! **B, stabilized forgetting** (Kulhavý and Zarrop, "On a general concept
//! of forgetting", Int. J. Control 58(4), 1993). Forgetting flattens the
//! posterior toward an alternative (prior) density rather than toward
//! nothing: `p̄(θ) ∝ p(θ)^λ · p₀(θ)^(1−λ)`. For a Gaussian with a
//! normal-inverse-Wishart prior the statistics are linear, so the
//! forgetting step is `ν̄ = λ·ν + (1−λ)·ν₀`, `S̄ = λ·S + (1−λ)·S₀`: the
//! prior's share is replenished every step and never decays away. Here the
//! prior is `P₀` pseudo-rows (`prior_rows`) at covariance `Ψ = diag(ŝ)`, and
//! the data keep today's recursion untouched, so in the mean form
//!
//! ```text
//! P' = λ·P + (1 − λ)·P₀              every row, whatever the state's weight
//! θ  = P / (W + P)
//! Σ_s = (1 − θ)·M + θ·diag(ŝ)        (W·M + P·Ψ) / (W + P)
//! ```
//!
//! With no data (`W → 0`) the state's covariance tends to `Ψ`, not to 0;
//! in steady state at a data rate `w̄` a row the prior's share is
//! `P₀(1−λ) / (P₀(1−λ) + w̄)`; with no decay (`λ = 1`) `P = P₀` and `Σ_s`
//! is scikit-learn QDA's `reg_param` shrinkage at `reg_param = P₀/(n+P₀)`
//! toward `diag(ŝ)` in place of `I`. A composes over a gap exactly:
//! `λ^Δ` in place of `λ`.
//!
//! **C, a fixed relative floor** (hmmlearn's `min_covar`, added to every
//! covariance; here relative to the data scale): `Σ_s = M + c·diag(ŝ)`.
//!
//! **The data scale `ŝ`.** A per-feature EW mean of the row's squared
//! deviation from the state means before the row, weighted by the
//! responsibilities (`hmm`: `e_i = Σ_s p_s·(x_i − μ_si)²`; `ew_class`: the
//! labelled class's, once it has weight), at a slow decay `λ^(1/10)`
//! (ten times the model's half-life):
//!
//! ```text
//! V' = λ_slow·V + w        ŝ_i' = ŝ_i + (w / V')·(e_i − ŝ_i)
//! ```
//!
//! seeded in `hmm` from the seeded states' pooled variances. Until `ŝ` is
//! known (`V = 0`) B and C read as A; once it is, they drop `r_A`, so both
//! are scale-free. State: `P` per state (B), `ŝ` and `V` per model (B, C):
//! `K + d + 1` doubles.

use crate::ewcov::row_at;

/// Which ridge the study's model reads.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum Variant {
    /// Today's ridge, decaying with the data.
    #[default]
    A,
    /// Stabilized forgetting toward `prior_rows` pseudo-rows at `diag(ŝ)`.
    B { prior_rows: f64 },
    /// A floor `floor·diag(ŝ)` added to every covariance.
    C { floor: f64 },
    /// Beyond the brief: stabilized forgetting toward the state's own
    /// current moments, `W' = λW + w + (1 − λ)·P₀` (`prior_rows` = `P₀`):
    /// a state's weight never decays below about `P₀`, so the update's
    /// `a = λW/(λW + w)` stays near `P₀/(P₀ + w)` after an absence, and a
    /// row of tiny responsibility moves its mean by about `w/P₀`.
    D { prior_rows: f64 },
}

/// The slow tracker's exponent: `λ_slow = λ^SLOW`, ten half-lives.
const SLOW: f64 = 0.1;

/// The study's state beside a model's; see the module docs.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Study {
    pub variant: Variant,
    prior_w: Vec<f64>,
    scale: Vec<f64>,
    scale_w: f64,
    /// `(λ, λ^SLOW)` of the last row: a clock of even steps takes one `powf`.
    slow_at: (f64, f64),
    /// The row's squared deviations, kept between rows.
    pub(crate) e: Vec<f64>,
}

impl Study {
    pub fn new(variant: Variant, n_states: usize, d: usize) -> Self {
        let p0 = match variant {
            Variant::B { prior_rows } => prior_rows,
            _ => 0.0,
        };
        Self {
            variant,
            prior_w: vec![p0; n_states],
            scale: vec![0.0; d],
            scale_w: 0.0,
            slow_at: (1.0, 1.0),
            e: Vec::with_capacity(d),
        }
    }

    /// Whether B or C is in force: a model reads the study's matrices.
    pub fn active(&self) -> bool {
        matches!(self.variant, Variant::B { .. } | Variant::C { .. }) && self.scale_w > 0.0
    }

    /// Variant D's weight for each state after a row of decay `lam`.
    pub fn lift(&self, lam: f64) -> Option<f64> {
        match self.variant {
            Variant::D { prior_rows } => Some((1.0 - lam) * prior_rows),
            _ => None,
        }
    }

    /// Whether a variant other than A is set, known scale or not.
    pub fn set(&self) -> bool {
        !matches!(self.variant, Variant::A)
    }

    /// The forgetting step on the prior weights, `P' = λP + (1−λ)P₀`.
    pub fn age(&mut self, lam: f64) {
        if let Variant::B { prior_rows } = self.variant {
            for p in &mut self.prior_w {
                *p = lam * *p + (1.0 - lam) * prior_rows;
            }
        }
    }

    /// A row's per-feature squared deviations `e` at weight `w` into `ŝ`.
    pub fn observe(&mut self, lam: f64, w: f64, e: &[f64]) {
        if !self.set() {
            return;
        }
        if lam != self.slow_at.0 {
            self.slow_at = (lam, lam.powf(SLOW));
        }
        let v = self.slow_at.1 * self.scale_w + w;
        if v > 0.0 && w > 0.0 {
            let b = w / v;
            for (s, &ei) in self.scale.iter_mut().zip(e) {
                *s += b * (ei - *s);
            }
        }
        self.scale_w = v;
    }

    /// `ŝ` from pooled variances at weight `w` (the `hmm` seeding).
    pub fn seed_scale(&mut self, scale: &[f64], w: f64) {
        if !self.set() || w <= 0.0 {
            return;
        }
        self.scale.copy_from_slice(scale);
        self.scale_w = w;
    }

    /// `(m, add_i)`: state `s`'s covariance is `m·M + diag(add)`.
    fn shape(&self, s: usize, w_state: f64) -> (f64, f64, &[f64]) {
        match self.variant {
            Variant::B { .. } => {
                let p = self.prior_w[s];
                let theta = p / (w_state + p);
                (1.0 - theta, theta, &self.scale)
            }
            Variant::C { floor } => (1.0, floor, &self.scale),
            Variant::A | Variant::D { .. } => unreachable!("read only when active"),
        }
    }

    /// One diagonal entry of state `s`'s covariance from its variance.
    pub fn diag(&self, s: usize, w_state: f64, i: usize, var: f64) -> f64 {
        let (m, t, sc) = self.shape(s, w_state);
        m * var + t * sc[i]
    }

    /// State `s`'s covariance, `d²` row-major (its lower triangle meant),
    /// from its packed co-moments: what `SpdFactor::of` reads.
    pub fn matrix(&self, s: usize, w_state: f64, packed: &[f64], d: usize, out: &mut Vec<f64>) {
        let (m, t, sc) = self.shape(s, w_state);
        out.clear();
        out.resize(d * d, 0.0);
        for i in 0..d {
            let at = row_at(i);
            for j in 0..=i {
                out[i * d + j] = m * packed[at + j];
            }
            out[i * d + i] += t * sc[i];
        }
    }

    /// The doubles this study adds to a model's state.
    pub fn state_len(&self) -> usize {
        match self.variant {
            Variant::A | Variant::D { .. } => 0,
            Variant::B { .. } => self.prior_w.len() + self.scale.len() + 1,
            Variant::C { .. } => self.scale.len() + 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// B's prior weight is an affine map per row, so two rows of decay
    /// `λ₁`, `λ₂` are one of `λ₁λ₂` (a gap composes), and it tends to `P₀`.
    #[test]
    fn the_prior_weight_composes_over_a_gap_and_tends_to_its_rows() {
        let v = Variant::B { prior_rows: 3.0 };
        let (mut two, mut one) = (Study::new(v, 1, 1), Study::new(v, 1, 1));
        two.prior_w[0] = 0.25;
        one.prior_w[0] = 0.25;
        two.age(0.9);
        two.age(0.7);
        one.age(0.9 * 0.7);
        assert!((two.prior_w[0] - one.prior_w[0]).abs() < 1e-15);
        for _ in 0..2000 {
            one.age(0.99);
        }
        assert!((one.prior_w[0] - 3.0).abs() < 1e-8);
    }

    /// The matrix B reads is `(W·M + P·diag(ŝ)) / (W + P)`; C's is
    /// `M + c·diag(ŝ)`.
    #[test]
    fn the_matrices_are_the_stated_mixtures() {
        let packed = [2.0, 0.5, 1.0];
        let (w, p, s) = (3.0, 1.5, [4.0, 0.25]);
        let mut b = Study::new(Variant::B { prior_rows: p }, 1, 2);
        b.seed_scale(&s, 1.0);
        let mut m = Vec::new();
        b.matrix(0, w, &packed, 2, &mut m);
        let want = [
            (w * 2.0 + p * s[0]) / (w + p),
            0.0,
            w * 0.5 / (w + p),
            (w * 1.0 + p * s[1]) / (w + p),
        ];
        for (got, want) in m.iter().zip(want) {
            assert!((got - want).abs() < 1e-15, "{m:?}");
        }
        assert!((b.diag(0, w, 1, 1.0) - want[3]).abs() < 1e-15);
        let mut c = Study::new(Variant::C { floor: 0.1 }, 1, 2);
        c.seed_scale(&s, 1.0);
        c.matrix(0, w, &packed, 2, &mut m);
        assert_eq!(m, vec![2.0 + 0.4, 0.0, 0.5, 1.0 + 0.025]);
    }
}
