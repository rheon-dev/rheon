//! The factor form of an `hmm` state under `covariance_type = "full"`, at
//! [`super::Hmm::FACTOR_FROM`] features and more (task 479).
//!
//! A state's density reads `M = C + s·r·I`, its centred co-moments `C` and
//! its decaying ridge, through a Cholesky factor of `M`. Kept as co-moments,
//! `M` is factorized afresh on every row after a learned one, `O(d³/3)`. The
//! row moves `M` by a scale and a rank-one step, `M' = a·M + a·b·δδᵀ` (the
//! accumulator's mean-form step, `δ` the row's deviation from the means
//! before it, `a = λW/(λW + w)`, `b = w/(λW + w)`; the ridge's scale decays
//! by the same `a`), so this form keeps the factor itself as the state and
//! moves it with the row, `O(d²)`:
//!
//! ```text
//! M  = σ·L Lᵀ                         σ in [1/4, 1]
//! σ' = a·σ
//! v  = √(a·b / σ')·δ                  = √(b/σ)·δ
//! L' L'ᵀ = L Lᵀ + v vᵀ               plane rotations, column by column
//! ```
//!
//! so `σ'·L' L'ᵀ = a·σ·L Lᵀ + a·b·δδᵀ = M'`. The rotations are LINPACK's
//! `dchud` ([`SpdFactor::add_outer_in_place`]): each replaces a pivot `d`
//! by `√(d² + p²)` and turns the rest of its column and of `v` by
//! `cos = d/r`, `sin = p/r`, both at most 1, so the step is backward
//! stable. The decay goes into the scalar `σ`, one rounding a row, where
//! scaling `L` by `√a` would round every entry. Whenever `σ` falls below
//! `1/4`, `L` is multiplied by a power of two `2⁻ᵉ` and `σ` by `4ᵉ`, which
//! moves no bit of `M`, so `σ` stays in `[1/4, 1]` and a density's
//! `ln det M = ln det L Lᵀ + d·ln σ` adds nothing large to cancel (at a
//! `σ` near `2⁻¹²⁸` it lost about `d` units in the last place of 88).
//!
//! A density reads, from the row's deviation `δ` from the means,
//!
//! ```text
//! ln det M = 2·Σ ln Lⱼⱼ + d·ln σ
//! δᵀ M⁻¹ δ = |L⁻¹ δ|² / σ
//! ```
//!
//! The first row with weight after a decay that took the whole history
//! (`a = 0`) starts `M` again at the ridge, `r·I` (`L = √r·I`, `σ = 1`), as
//! the co-moments start again at 0 with the ridge's scale at 1. A row of
//! weight 0 moves no `M`, no mean and no ridge, and ages the weight by `λ`.
//! A rotation that leaves no factor (a pivot that is not a positive finite
//! number) starts `M` again at the ridge it has reached, and is counted as a
//! failure by the owner.
//!
//! The means are the accumulator's pair `hi + lo` ([`crate::comp`]), stepped
//! as [`crate::EwCov`] steps them, so a state's means and weight are its
//! co-moment form's to the bit; only `M` is held otherwise. The state is
//! `d(d + 1)/2 + 2d + 3` numbers, where the co-moment form keeps
//! `d(d + 1)/2 + 2d + 5` and more (its runs, its learned rows, the sum of
//! squared weights), none of which `hmm` reads.

use serde::{Deserialize, Serialize};

use crate::EwCov;
use crate::solve::{QuadWork, SpdFactor};

/// The least `σ` a state keeps: below it, `L` is halved as many times as
/// it takes and `σ` multiplied by 4 as many.
const SIGMA_FLOOR: f64 = 0.25;

/// One `hmm` state's weight, means and `M = σ·L Lᵀ` (the module docs).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(into = "EwFactorState", try_from = "EwFactorState")]
pub struct EwFactor {
    /// The EW sum of the state's weights, its `n_eff`.
    w_sum: f64,
    /// The means, each the pair `m[i] + m_lo[i]` ([`crate::comp`]).
    m: Vec<f64>,
    m_lo: Vec<f64>,
    /// The ridge's decaying scale `s`: `M` holds `s·r·I` of ridge. Read
    /// only where `M` starts again at the ridge it has reached.
    precision_scale: f64,
    /// `σ`, in `[1/4, 1]`.
    sigma: f64,
    /// `L`, lower triangular with a positive diagonal.
    l: SpdFactor,
}

/// How a state writes [`EwFactor`]: `L` as its lower triangle, column by
/// column from the diagonal down.
#[derive(Serialize, Deserialize)]
struct EwFactorState {
    w_sum: f64,
    m: Vec<f64>,
    m_lo: Vec<f64>,
    precision_scale: f64,
    sigma: f64,
    l: Vec<f64>,
}

impl From<EwFactor> for EwFactorState {
    fn from(f: EwFactor) -> Self {
        Self {
            l: f.l.lower_columns(),
            w_sum: f.w_sum,
            m: f.m,
            m_lo: f.m_lo,
            precision_scale: f.precision_scale,
            sigma: f.sigma,
        }
    }
}

/// A factor read from a state is refused unless this module could have
/// left it: one mean and one low part a feature, all finite; a weight and
/// a ridge scale finite and at least 0; `σ` in `[1/4, 1]`; and `L` a
/// triangle of the means' order with every pivot a positive finite number
/// and every entry finite.
impl TryFrom<EwFactorState> for EwFactor {
    type Error = String;

    fn try_from(st: EwFactorState) -> Result<Self, String> {
        let damaged = || {
            "a saved hidden state's covariance factor holds values no state could have. Load an \
             undamaged copy of the state file, or rebuild it from its input."
                .to_string()
        };
        let d = st.m.len();
        let finite = |v: &[f64]| v.iter().all(|x| x.is_finite());
        if d == 0
            || st.m_lo.len() != d
            || !finite(&st.m)
            || !finite(&st.m_lo)
            || !(st.w_sum >= 0.0 && st.w_sum.is_finite())
            || !(st.precision_scale >= 0.0 && st.precision_scale.is_finite())
            || !(SIGMA_FLOOR..=1.0).contains(&st.sigma)
        {
            return Err(damaged());
        }
        let l = SpdFactor::of_lower_columns(d, &st.l).ok_or_else(damaged)?;
        Ok(Self {
            w_sum: st.w_sum,
            m: st.m,
            m_lo: st.m_lo,
            precision_scale: st.precision_scale,
            sigma: st.sigma,
            l,
        })
    }
}

impl EwFactor {
    /// The factor form of `cov`, whose `M = C + s·r·I` factors as `l`: the
    /// same weight, means and ridge scale, and `M = 1·L Lᵀ`.
    pub(crate) fn take_up(cov: &EwCov, l: SpdFactor) -> Self {
        debug_assert!(!cov.has_pending(), "an hmm state is never blocked");
        debug_assert_eq!(l.order(), cov.k());
        Self {
            w_sum: cov.n_eff(),
            m: cov.means().to_vec(),
            m_lo: cov.means_lo().to_vec(),
            precision_scale: cov.precision_scale(),
            sigma: 1.0,
            l: l.into_state(),
        }
    }

    /// The number of features.
    pub fn order(&self) -> usize {
        self.m.len()
    }

    /// The state's weight, as [`EwCov::n_eff`] reads it.
    pub fn n_eff(&self) -> f64 {
        self.w_sum
    }

    /// The state's means.
    pub fn means(&self) -> &[f64] {
        &self.m
    }

    /// `x`'s deviation from mean `i`, the pair's ([`crate::comp::dev`]).
    #[inline]
    pub fn deviation(&self, i: usize, x: f64) -> f64 {
        crate::comp::dev(x, self.m[i], self.m_lo[i])
    }

    /// The ridge's decaying scale `s`, as [`EwCov::precision_scale`].
    pub fn precision_scale(&self) -> f64 {
        self.precision_scale
    }

    /// `σ`, the scale `M = σ·L Lᵀ` holds `L Lᵀ` at.
    pub fn sigma(&self) -> f64 {
        self.sigma
    }

    /// `L`.
    pub fn factor(&self) -> &SpdFactor {
        &self.l
    }

    /// `M = σ·L Lᵀ`, row-major `d*d`, its ridge included: what the state's
    /// density reads.
    pub fn matrix(&self) -> Vec<f64> {
        let d = self.order();
        let l = self.l.lower_columns();
        // Column `j` of the packed triangle starts at `j·d − j(j−1)/2`.
        let at = |i: usize, j: usize| l[j * d - j * j.saturating_sub(1) / 2 + (i - j)];
        let mut out = vec![0.0; d * d];
        for i in 0..d {
            for j in 0..=i {
                let s: f64 = (0..=j).map(|t| at(i, t) * at(j, t)).sum();
                out[i * d + j] = self.sigma * s;
                out[j * d + i] = self.sigma * s;
            }
        }
        out
    }

    /// `(ln det M, δᵀ M⁻¹ δ)` for the row's deviation `delta` from the
    /// means: `L`'s own at `σ = 1`, to the bit of a fresh factor of the
    /// same matrix.
    #[inline]
    pub(crate) fn read(&self, delta: &[f64], quad: &mut QuadWork) -> (f64, f64) {
        let (log_det, q) = (self.l.log_det(), self.l.quad_form(delta, quad));
        if self.sigma == 1.0 {
            (log_det, q)
        } else {
            (
                log_det + delta.len() as f64 * self.sigma.ln(),
                q / self.sigma,
            )
        }
    }

    /// The row `x` learned at weight `w` under the decay `lam`, as
    /// [`EwCov::update`] learns it (the module docs): `false` where the
    /// rotation left no factor and `M` started again at its ridge, which the
    /// owner counts as a failure. `prior` is the ridge `r` at scale 1; `v` is
    /// working space.
    pub(crate) fn update(
        &mut self,
        x: &[f64],
        lam: f64,
        w: f64,
        prior: f64,
        v: &mut Vec<f64>,
    ) -> bool {
        debug_assert_eq!(x.len(), self.order());
        debug_assert!(w >= 0.0, "a weight of {w}");
        // No weight on the row and none to come out of it: the decay alone,
        // [`EwCov::decay`]'s, which leaves the ridge's scale.
        if w == 0.0 && lam * self.w_sum == 0.0 {
            self.w_sum *= lam;
            return true;
        }
        // [`EwCov`]'s `step_factors`, to the bit.
        if w < 0.0 {
            return true;
        }
        let w_new = lam * self.w_sum + w;
        if w_new <= 0.0 {
            return true;
        }
        let (a, b) = (lam * self.w_sum / w_new, w / w_new);
        v.clear();
        v.extend(
            x.iter()
                .zip(&self.m)
                .zip(&self.m_lo)
                .map(|((&xi, &mi), &li)| crate::comp::dev(xi, mi, li)),
        );
        // The means' step, the accumulator's: none on a row of weight 0.
        if b > 0.0 {
            for ((mi, lo), &di) in self.m.iter_mut().zip(self.m_lo.iter_mut()).zip(v.iter()) {
                crate::comp::add(mi, lo, b * di);
            }
        }
        self.precision_scale = if a <= 0.0 {
            1.0
        } else {
            self.precision_scale * a
        };
        self.w_sum = w_new;
        let kept = if a <= 0.0 {
            // The first row with weight: `C' = 0` with the ridge's scale
            // at 1, so `M' = r·I`.
            self.restart(prior);
            true
        } else if a == 1.0 && a * b == 0.0 {
            // `M' = M`: a row too light to move it.
            true
        } else {
            let mut sigma = self.sigma * a;
            if sigma.is_nan() || sigma < f64::MIN_POSITIVE {
                // `a` past the normal range (a gap of over a thousand
                // half-lives): what it keeps of `M` is under `2⁻¹⁰²⁰` of it,
                // and so is all of `M'`, which no double holds to 53 bits
                // (the co-moments' `a·C` is rounded to multiples of
                // `2⁻¹⁰⁷⁴`). `M` starts at the ridge it has reached, `σ = 1`,
                // and the row is added to that; the next row whose `a` is
                // normal outweighs what either form kept by `2¹⁰²⁰`.
                self.restart_at_ridge(prior);
                sigma = 1.0;
            }
            let root = (a * b / sigma).sqrt();
            v.iter_mut().for_each(|x| *x *= root);
            if self.l.add_outer_in_place(v) {
                self.sigma = sigma;
                self.lift_sigma();
                true
            } else {
                self.restart_at_ridge(prior);
                false
            }
        };
        debug_assert!((SIGMA_FLOOR..=1.0).contains(&self.sigma));
        kept
    }

    /// A row of weight 0, as [`Self::update`] takes it: the weight ages by
    /// `lam` and nothing else moves.
    pub(crate) fn age(&mut self, lam: f64) {
        // `λW + 0` is `λW`, and `a = λW / λW` is 1: the step moves neither
        // `M` nor the ridge's scale, and the decay alone ages `W` alike.
        self.w_sum *= lam;
    }

    /// `M = ridge·I`, `σ = 1`: whether `ridge` is a positive finite number,
    /// without which `M` is not moved.
    fn restart(&mut self, ridge: f64) -> bool {
        match SpdFactor::scaled_identity(self.order(), ridge) {
            Some(l) => {
                self.l = l;
                self.sigma = 1.0;
                true
            }
            None => false,
        }
    }

    /// `M` started again at the ridge it has reached, `s·r·I`; at the ridge
    /// at scale 1 where that scale has decayed to 0, its scale with it.
    fn restart_at_ridge(&mut self, prior: f64) {
        if !self.restart(prior * self.precision_scale) {
            self.precision_scale = 1.0;
            self.restart(prior);
        }
    }

    /// `σ` back into `[1/4, 1]` by powers of two: `L·2⁻ᵉ` and `σ·4ᵉ`, the
    /// same `M`. `σ`'s exponent picks `e`, so no logarithm decides a bit.
    fn lift_sigma(&mut self) {
        while self.sigma < SIGMA_FLOOR {
            let biased = (self.sigma.to_bits() >> 52) & 0x7ff;
            // `σ` in `[2ᵖ, 2ᵖ⁺¹)` with `p <= -3`; a subnormal `σ` reads as
            // `p = -1023` and is lifted in more than one pass.
            let p = biased as i64 - 1023;
            let e = ((-p - 1) / 2).clamp(1, 511) as u32;
            self.l.scale_by_power_of_two(e);
            self.sigma *= f64::from_bits(u64::from(1023 + 2 * e) << 52);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }

    /// `C + s·r·I` of an accumulator, row-major `d*d`.
    fn ridged(cov: &EwCov, prior: f64) -> Vec<f64> {
        let d = cov.k();
        let mut m = cov.comoments().to_full();
        for i in 0..d {
            m[i * d + i] += prior * cov.precision_scale();
        }
        m
    }

    /// `|A − B|`'s largest entry over `B`'s largest diagonal entry.
    fn apart(a: &[f64], b: &[f64]) -> f64 {
        let d = (b.len() as f64).sqrt() as usize;
        let scale = (0..d).map(|i| b[i * d + i]).fold(0.0, f64::max);
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f64::max)
            / scale
    }

    /// The factor form taken up from an accumulator and stepped beside it
    /// by the same rows -- weights of 0, light and heavy ones, gaps of
    /// several half-lives, one past the normal range of `a` and one that
    /// takes the whole history -- keeps the accumulator's weight, means and
    /// ridge scale to the bit, `σ` in `[1/4, 1]`, and `σ·L Lᵀ` within
    /// rounding of `C + s·r·I`.
    #[test]
    fn the_factor_steps_as_the_accumulator_does() {
        for d in [1, 3, 13] {
            let prior = 0.5;
            let mut cov = EwCov::with_precision_prior(d, prior).unwrap();
            let mut seed = 11 + d as u64;
            for _ in 0..5 {
                let x: Vec<f64> = (0..d).map(|_| lcg(&mut seed)).collect();
                cov.update(&x, 1.0, 1.0);
            }
            let l = SpdFactor::of(&ridged(&cov, prior), d).unwrap();
            let mut f = EwFactor::take_up(&cov, l);
            let mut v = Vec::new();
            let (mut lifted, mut restarted, mut subnormals) = (0, 0, 0);
            for i in 0..3000 {
                let x: Vec<f64> = (0..d)
                    .map(|j| {
                        lcg(&mut seed) * (1.0 + j as f64) + if i % 50 < 25 { 3.0 } else { 0.0 }
                    })
                    .collect();
                // `2⁻¹⁰⁶⁰` written as its bits: `2f64.powi(-1060)` is folded
                // to them by an optimized build, but a dev build calls
                // compiler-rt's `__powidf2`, which forms `1/2¹⁰⁶⁰ = 1/∞ =
                // 0`, so the case past the normal range ran in release alone
                // (and failed there). Once only: at row 533 the ridge the
                // state has reached, `r·s·a`, underflows to 0, where the
                // accumulator keeps a ridge scale of 0 and the factor, which
                // cannot start at a zero matrix, takes the ridge at scale 1
                // (`restart_at_ridge`), so the two forms part.
                let lam = match i % 400 {
                    7 => 1e-3,
                    133 if i == 133 => f64::from_bits(1 << 14),
                    271 => 0.0,
                    _ => 0.99,
                };
                let w = match i % 7 {
                    3 => 0.0,
                    5 => 1e-9,
                    6 => 30.0,
                    _ => 1.0,
                };
                let w_new = lam * cov.n_eff() + w;
                let a = lam * cov.n_eff() / w_new;
                let subnormal = a > 0.0 && a < f64::MIN_POSITIVE;
                let before = f.sigma;
                assert!(f.update(&x, lam, w, prior, &mut v));
                cov.update(&x, lam, w);
                lifted += usize::from(f.sigma > before && f.sigma < 1.0);
                restarted += usize::from(lam == 0.0);
                assert!(
                    (SIGMA_FLOOR..=1.0).contains(&f.sigma),
                    "{d}, row {i}: σ {}",
                    f.sigma
                );
                assert_eq!(f.w_sum.to_bits(), cov.n_eff().to_bits(), "{d}, row {i}");
                assert_eq!(f.m, cov.means(), "{d}, row {i}");
                assert_eq!(f.m_lo, cov.means_lo(), "{d}, row {i}");
                assert_eq!(
                    f.precision_scale.to_bits(),
                    cov.precision_scale().to_bits(),
                    "{d}, row {i}"
                );
                // Past the normal range of `a` the whole matrix is under
                // `2⁻¹⁰²⁰`: the accumulator's `a·C` is rounded to multiples of
                // `2⁻¹⁰⁷⁴`, a part in `10⁷` of it, and the factor starts again
                // at the ridge it has reached with the row, dropping `a·C`.
                // Neither holds 53 bits of a subnormal matrix, so that row is
                // not compared; the next one, whose `a` is normal, outweighs
                // what either kept by `2¹⁰²⁰` and agrees again.
                subnormals += usize::from(subnormal);
                if !subnormal {
                    let off = apart(&f.matrix(), &ridged(&cov, prior));
                    assert!(off < 1e-13, "{d}, row {i}: σ L Lᵀ {off:e} of its size off");
                }
            }
            assert!(
                lifted > 10 && restarted > 0 && subnormals > 0,
                "{d}: {lifted} lifts, {restarted} restarts, {subnormals} subnormal steps"
            );
        }
    }

    /// A row of weight 0 moves no `M`, mean or ridge scale and ages the
    /// weight by `λ`, as [`EwFactor::age`] does; the first row with weight
    /// after a decay that took the whole history starts `M` at the ridge.
    #[test]
    fn a_row_of_no_weight_ages_the_weight_alone_and_a_total_decay_restarts() {
        let (d, prior) = (4, 0.25);
        let mut cov = EwCov::with_precision_prior(d, prior).unwrap();
        cov.update(&[1.0, 2.0, -1.0, 0.5], 1.0, 1.0);
        cov.update(&[0.0, 1.0, 1.0, -0.5], 1.0, 1.0);
        let f0 = EwFactor::take_up(&cov, SpdFactor::of(&ridged(&cov, prior), d).unwrap());
        let mut v = Vec::new();
        for lam in [1.0, 0.5, 0.0] {
            let (mut a, mut b) = (f0.clone(), f0.clone());
            a.update(&[9.0, 9.0, 9.0, 9.0], lam, 0.0, prior, &mut v);
            b.age(lam);
            assert_eq!(a, b, "λ {lam}");
            assert_eq!(a.w_sum, f0.w_sum * lam);
            assert_eq!((&a.m, &a.l, a.sigma), (&f0.m, &f0.l, f0.sigma));
        }
        let mut f = f0.clone();
        f.age(0.0);
        f.update(&[1.0, 1.0, 1.0, 1.0], 0.5, 2.0, prior, &mut v);
        assert_eq!(f.l, SpdFactor::scaled_identity(d, prior).unwrap());
        assert_eq!((f.sigma, f.precision_scale, f.w_sum), (1.0, 1.0, 2.0));
        assert_eq!(f.m, [1.0; 4]);
    }

    /// A rotation that leaves no factor starts `M` again at the ridge the
    /// state has reached, `σ = 1`, and says so for the owner to count. No
    /// usable row reaches it (each rotation's factors are at most 1, and a
    /// usable value is under 1e100), so a step that is not a number stands
    /// in for one: the rotations refuse it before anything moves.
    #[test]
    fn a_rotation_that_leaves_no_factor_restarts_at_the_ridge() {
        let (d, prior) = (2, 1.0);
        let mut cov = EwCov::with_precision_prior(d, prior).unwrap();
        cov.update(&[0.0, 0.0], 1.0, 1.0);
        cov.update(&[1.0, -1.0], 1.0, 1.0);
        let l = SpdFactor::of(&ridged(&cov, prior), d).unwrap();
        let mut f = EwFactor::take_up(&cov, l);
        let mut v = Vec::new();
        assert!(!f.update(&[f64::INFINITY, 0.0], 1.0, 1.0, prior, &mut v));
        let r = prior * f.precision_scale;
        assert!(r > 0.0 && r < 1.0);
        assert_eq!(f.l, SpdFactor::scaled_identity(d, r).unwrap());
        assert_eq!(f.sigma, 1.0);
        let mut l = SpdFactor::scaled_identity(d, 1.0).unwrap();
        assert!(!l.add_outer_in_place(&mut [f64::NAN, 0.0]));
        assert!(!l.add_outer_in_place(&mut [1.0]));
        assert_eq!(
            l,
            SpdFactor::scaled_identity(d, 1.0).unwrap(),
            "refused before moving"
        );
    }

    /// `σ` lifted into `[1/4, 1]` from anywhere in the normal range by
    /// powers of two moves no bit of `M`.
    #[test]
    fn lifting_sigma_moves_no_bit_of_the_matrix() {
        let (d, prior) = (3, 1.0);
        let mut cov = EwCov::with_precision_prior(d, prior).unwrap();
        cov.update(&[0.3, -1.0, 2.0], 1.0, 1.0);
        cov.update(&[1.3, 0.5, -2.0], 1.0, 1.0);
        let f0 = EwFactor::take_up(&cov, SpdFactor::of(&ridged(&cov, prior), d).unwrap());
        for sigma in [0.2499, 0.125, 0.1, 1e-3, 3e-200, f64::MIN_POSITIVE] {
            let mut f = f0.clone();
            f.sigma = sigma;
            let before: Vec<u64> = f.matrix().iter().map(|x| x.to_bits()).collect();
            f.lift_sigma();
            assert!(
                (SIGMA_FLOOR..=1.0).contains(&f.sigma),
                "{sigma:e}: {}",
                f.sigma
            );
            let after: Vec<u64> = f.matrix().iter().map(|x| x.to_bits()).collect();
            assert_eq!(before, after, "σ {sigma:e}");
        }
    }

    /// A factor read from a state is refused where no state could hold it.
    #[test]
    fn a_saved_factor_no_state_could_hold_is_refused() {
        let good = || EwFactorState {
            w_sum: 2.0,
            m: vec![0.5, -0.5],
            m_lo: vec![0.0, 1e-17],
            precision_scale: 0.75,
            sigma: 0.5,
            l: vec![1.0, 0.25, 2.0],
        };
        assert!(EwFactor::try_from(good()).is_ok());
        let edits: [&dyn Fn(&mut EwFactorState); 10] = [
            &|s| s.sigma = 0.2,
            &|s| s.sigma = 1.5,
            &|s| s.sigma = f64::NAN,
            &|s| s.l.pop().map(drop).unwrap_or(()),
            &|s| s.l[0] = -1.0,
            &|s| s.l[1] = f64::INFINITY,
            &|s| s.m_lo.pop().map(drop).unwrap_or(()),
            &|s| s.m[0] = f64::NAN,
            &|s| s.w_sum = -1.0,
            &|s| s.precision_scale = f64::INFINITY,
        ];
        for (i, edit) in edits.iter().enumerate() {
            let mut s = good();
            edit(&mut s);
            let e = EwFactor::try_from(s).unwrap_err();
            assert!(e.contains("no state could have"), "{i}: {e}");
        }
        let empty = EwFactorState {
            m: vec![],
            m_lo: vec![],
            l: vec![],
            ..good()
        };
        assert!(EwFactor::try_from(empty).is_err());
    }

    /// The state round-trips through both encodings to the same factor,
    /// `L` written column by column.
    #[test]
    fn a_factor_round_trips_through_its_state() {
        let f = EwFactor::try_from(EwFactorState {
            w_sum: 2.0,
            m: vec![0.5, -0.5],
            m_lo: vec![0.0, 1e-17],
            precision_scale: 0.75,
            sigma: 0.5,
            l: vec![1.0, 0.25, 2.0],
        })
        .unwrap();
        let named = rmp_serde::to_vec_named(&f).unwrap();
        assert_eq!(rmp_serde::from_slice::<EwFactor>(&named).unwrap(), f);
        let compact = rmp_serde::to_vec(&f).unwrap();
        assert_eq!(rmp_serde::from_slice::<EwFactor>(&compact).unwrap(), f);
        assert_eq!(f.factor().lower_columns(), [1.0, 0.25, 2.0]);
        assert_eq!(f.matrix(), [0.5, 0.125, 0.125, 0.5 * (0.0625 + 4.0)]);
    }
}
