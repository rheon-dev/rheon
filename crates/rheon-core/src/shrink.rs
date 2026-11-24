//! Shrinkage of a covariance matrix toward a target (docs/PLAN.md task 511):
//! the intensity of Ledoit and Wolf (2004), the oracle-approximating
//! intensity of Chen, Wiesel, Eldar and Hero (2010) in scikit-learn's form,
//! and a given one, for `ew_cov`'s second moments and `ew_class`'s class
//! covariances.
//!
//! The shrunk matrix is
//!
//! ```text
//! Σ̂ = (1 − s) S + s T,   T = μ I,   μ = tr S / p,   0 ≤ s ≤ 1
//! ```
//!
//! with `S` the weighted (biased) covariance the accumulator keeps, its
//! divisor the weight `W`, and `p` the features. A given intensity may take
//! a given target `T₀` in place of `μ I`.
//!
//! - **Ledoit-Wolf** (J. Multivariate Analysis 88, 2004, §3.3):
//!   `δ = (‖S‖²_F − p μ²) / p`, `β = (F − ‖S‖²_F) / (p n)`, `β ← min(β, δ)`,
//!   `s = β / δ`, 0 where `β = 0`. `F` is the weighted mean of `|x − m|⁴`
//!   over the rows, which the model keeps by the recursion in [`Fourth`].
//! - **OAS**, as scikit-learn computes it: `α = ‖S‖²_F / p²`,
//!   `s = min((α + μ²) / ((n + 1)(α − μ² / p)), 1)`, 1 where the denominator
//!   is 0. The paper's Eq. 23 has `2/p` terms besides; scikit-learn drops
//!   them, which moves `s` by `O(1/p)`.
//!
//! At `p = 1` both intensities are 0, as scikit-learn's: a single variance
//! is its own target.
//!
//! `n` is Kish's effective size `W² / Q`, `Q` the sum of the squared
//! weights, not the weight `W`: both intensities read `n` through the
//! variance of `S` as a weighted mean of the rows' `z zᵀ`, and the variance
//! of a weighted mean of independent terms is the terms' variance over
//! `W² / Q`. That number is free of the weights' units, where `W` is not,
//! and it is the row count at unit weights without decay, scikit-learn's `n`.

use serde::{Deserialize, Serialize};

use crate::ewcov::row_at;

/// How an `ew_cov`'s second moments, or an `ew_class`'s class covariances,
/// are shrunk toward a target.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Shrinkage {
    /// Ledoit and Wolf's intensity, toward `μ I`.
    LedoitWolf,
    /// The oracle-approximating intensity, in scikit-learn's form, toward
    /// `μ I`.
    Oas,
    /// A given intensity, in `[0, 1]`, toward `μ I`, or toward `target`: a
    /// symmetric positive semidefinite matrix given as its lower triangle's
    /// rows, packed (entry `(i, j)`, `j ≤ i`, at `i(i + 1)/2 + j`), in the
    /// features' units.
    Fixed {
        intensity: f64,
        #[serde(default)]
        target: Option<Vec<f64>>,
    },
}

impl Shrinkage {
    /// Whether the intensity reads the rows' fourth moments ([`Fourth`]).
    pub fn needs_fourth(&self) -> bool {
        matches!(self, Self::LedoitWolf)
    }

    /// The name a message gives it.
    pub fn name(&self) -> &'static str {
        match self {
            Self::LedoitWolf => "ledoit_wolf",
            Self::Oas => "oas",
            Self::Fixed { .. } => "fixed",
        }
    }

    /// Refuse an intensity outside `[0, 1]`, and a target not of `k`
    /// features, not finite, or not positive semidefinite. `model` names the
    /// model in the message.
    pub fn validate(&self, model: &str, k: usize) -> Result<(), String> {
        let Self::Fixed { intensity, target } = self else {
            return Ok(());
        };
        if !(intensity.is_finite() && (0.0..=1.0).contains(intensity)) {
            return Err(format!(
                "{model}: the shrinkage intensity must be at least 0 and at most 1, but it is \
                 {intensity}. Set it to the share of the target in the shrunk matrix, such as 0.1."
            ));
        }
        let Some(t) = target else {
            return Ok(());
        };
        if t.len() != row_at(k) {
            return Err(format!(
                "{model}: the shrinkage target must hold the lower triangle of a {k} by {k} \
                 matrix, {} numbers, but it holds {}. Give row i's entries up to and including \
                 the diagonal, row after row.",
                row_at(k),
                t.len()
            ));
        }
        if t.iter().any(|v| !v.is_finite()) {
            return Err(format!(
                "{model}: the shrinkage target must be finite, but it holds a NaN or an \
                 infinity. Give a finite matrix."
            ));
        }
        if !is_psd(t, k) {
            return Err(format!(
                "{model}: the shrinkage target must be positive semidefinite, but it has a \
                 negative eigenvalue: a shrunk covariance would not be one. Give a covariance \
                 matrix as the target."
            ));
        }
        Ok(())
    }
}

/// Whether the symmetric matrix of packed lower rows `t` has no eigenvalue
/// below `-64 ε` times its largest magnitude: one decomposition, at the
/// build, never on a row.
fn is_psd(t: &[f64], k: usize) -> bool {
    use faer::prelude::*;
    let get = |i: usize, j: usize| t[row_at(i.max(j)) + i.min(j)];
    let mat = Mat::from_fn(k, k, get);
    let Some((s, _)) = crate::solve::self_adjoint_eigen(mat.as_ref()) else {
        return false;
    };
    let top = s.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    s.iter().all(|&v| v >= -64.0 * f64::EPSILON * top)
}

/// `tr S` and `‖S‖²_F` of the symmetric matrix whose packed lower rows are
/// `c`: each off-diagonal entry counted twice.
pub(crate) fn trace_and_frobenius(c: &[f64], k: usize) -> (f64, f64) {
    // Every packed entry's square in one pass over the rows, four sums
    // apart so the pass vectorises, and the diagonal's beside it.
    let mut acc = [0.0; 4];
    let (chunks, rest) = c.as_chunks::<4>();
    for ch in chunks {
        for (a, v) in acc.iter_mut().zip(ch) {
            *a += v * v;
        }
    }
    let all = (acc[0] + acc[1]) + (acc[2] + acc[3]) + rest.iter().map(|v| v * v).sum::<f64>();
    let mut tr = 0.0;
    let mut diag = 0.0;
    for i in 0..k {
        let d = c[row_at(i) + i];
        tr += d;
        diag += d * d;
    }
    (tr, 2.0 * all - diag)
}

/// Ledoit and Wolf's intensity from `tr S`, `‖S‖²_F`, the rows' weighted
/// mean of `|x − m|⁴` and Kish's size `n`: `min(β, δ) / δ`, with `β` held
/// at 0 below (it is `E‖zzᵀ‖² − ‖E zzᵀ‖²`, never negative but in its
/// rounding). 0 at `p = 1` and where `δ` is not above 0 (`S` a multiple of
/// the identity, the target already); NaN where `n` is not known.
pub fn ledoit_wolf(p: usize, tr: f64, frob2: f64, fourth: f64, n: Option<f64>) -> f64 {
    if p <= 1 {
        return 0.0;
    }
    let Some(n) = n.filter(|n| *n > 0.0 && n.is_finite()) else {
        return f64::NAN;
    };
    let pf = p as f64;
    let mu = tr / pf;
    let delta = (frob2 - pf * mu * mu) / pf;
    // Not above 0, a NaN included.
    if delta.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
        return 0.0;
    }
    let beta = ((fourth - frob2) / (pf * n)).max(0.0).min(delta);
    if beta == 0.0 { 0.0 } else { beta / delta }
}

/// The oracle-approximating intensity, scikit-learn's form, from `tr S`,
/// `‖S‖²_F` and Kish's size `n`: `min((α + μ²) / ((n + 1)(α − μ²/p)), 1)`,
/// `α = ‖S‖²_F / p²`; 1 where the denominator is not above 0 (`S` a
/// multiple of the identity: shrinking moves nothing); 0 at `p = 1`; NaN
/// where `n` is not known.
pub fn oas(p: usize, tr: f64, frob2: f64, n: Option<f64>) -> f64 {
    if p <= 1 {
        return 0.0;
    }
    let Some(n) = n.filter(|n| *n > 0.0 && n.is_finite()) else {
        return f64::NAN;
    };
    let pf = p as f64;
    let alpha = frob2 / (pf * pf);
    let mu = tr / pf;
    let mu2 = mu * mu;
    let num = alpha + mu2;
    let den = (n + 1.0) * (alpha - mu2 / pf);
    if den.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
        1.0
    } else {
        (num / den).min(1.0)
    }
}

/// `(1 − s) c + s T` into `dst`, both packed lower rows of `k × k`: `T` the
/// given target, or `μ I` with `μ = tr c / k`.
pub(crate) fn shrink_into(c: &[f64], k: usize, s: f64, target: Option<&[f64]>, dst: &mut [f64]) {
    debug_assert_eq!(c.len(), row_at(k));
    debug_assert_eq!(dst.len(), row_at(k));
    let keep = 1.0 - s;
    match target {
        Some(t) => {
            for ((d, &v), &tv) in dst.iter_mut().zip(c).zip(t) {
                *d = keep * v + s * tv;
            }
        }
        None => {
            let tr: f64 = (0..k).map(|i| c[row_at(i) + i]).sum();
            let toward = s * (tr / k as f64);
            for (d, &v) in dst.iter_mut().zip(c) {
                *d = keep * v;
            }
            for i in 0..k {
                let at = row_at(i) + i;
                dst[at] = keep * c[at] + toward;
            }
        }
    }
}

/// The rows' third and fourth central moments that Ledoit and Wolf's
/// intensity reads, kept in the mean form beside the co-moments: `t =
/// E_w[|z|² z]` (`p` numbers) and `f = E_w[|z|⁴]`, `z = x − m`, about the
/// same mean `m`. 8 `(p + 1)` bytes, whatever the stream's length.
///
/// A row of weight `w` moves the mean by `u = b d`, `d = x − m`, `a = λW /
/// W'`, `b = w / W'`, and every row's deviation by `−u`; expanding `|z − u|²
/// (z − u)` and `|z − u|⁴` over rows whose weighted mean of `z` is 0:
///
/// ```text
/// t' = a (t − tr(C) u − 2 C u − |u|² u)                    + b (1 − b)³ |d|² d
/// f' = a (f + 4 uᵀ C u + |u|⁴ − 4 t·u + 2 |u|² tr C)       + b (1 − b)⁴ |d|⁴
/// ```
///
/// with `C` the co-moments before the row. A row of weight 0 (`a = 1`,
/// `b = 0`) leaves both to the bit.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Fourth {
    pub(crate) t: Vec<f64>,
    pub(crate) f: f64,
}

impl Fourth {
    pub fn new(k: usize) -> Self {
        Self {
            t: vec![0.0; k],
            f: 0.0,
        }
    }

    /// Whether it is of `k` features.
    pub fn has_shape(&self, k: usize) -> bool {
        self.t.len() == k
    }

    /// The weighted mean of `|x − m|⁴`.
    pub fn f(&self) -> f64 {
        self.f
    }

    /// The weighted mean of `|x − m|² (x − m)`.
    pub fn t(&self) -> &[f64] {
        &self.t
    }

    /// The row's step, `c` the co-moments' packed lower rows before it, `d`
    /// its deviation from the mean before it, `a` and `b` its mixing
    /// factors. `cu` is scratch, `2k` long.
    pub(crate) fn learn(&mut self, c: &[f64], d: &[f64], a: f64, b: f64, cu: &mut Vec<f64>) {
        if b == 0.0 && a == 1.0 {
            return;
        }
        let k = d.len();
        // `u = b d` in the first half of `cu`, and `C u` in the second,
        // from the packed lower rows, each entry read once: row `i`'s
        // entries below the diagonal are a dot product with `u` for
        // `(C u)_i` and an update of `(C u)_j` by `c_ij u_i`.
        cu.clear();
        cu.extend(d.iter().map(|di| b * di));
        cu.resize(2 * k, 0.0);
        let (u, cu) = cu.split_at_mut(k);
        let mut tr = 0.0;
        for i in 0..k {
            let at = row_at(i);
            let row = &c[at..at + i];
            let ui = u[i];
            let (head, rest) = cu.split_at_mut(i);
            let mut acc = 0.0;
            for ((h, &cij), &uj) in head.iter_mut().zip(row).zip(&u[..i]) {
                acc += cij * uj;
                *h += cij * ui;
            }
            let cii = c[at + i];
            tr += cii;
            rest[0] += acc + cii * ui;
        }
        let mut uu = 0.0;
        let mut ucu = 0.0;
        let mut tu = 0.0;
        let mut dd = 0.0;
        for i in 0..k {
            let ui = u[i];
            uu += ui * ui;
            ucu += ui * cu[i];
            tu += self.t[i] * ui;
            dd += d[i] * d[i];
        }
        let nb = 1.0 - b;
        let nb2 = nb * nb;
        let row3 = b * nb2 * nb * dd;
        for i in 0..k {
            let ui = u[i];
            self.t[i] = a * (self.t[i] - tr * ui - 2.0 * cu[i] - uu * ui) + row3 * d[i];
        }
        self.f =
            a * (self.f + 4.0 * ucu + uu * uu - 4.0 * tu + 2.0 * uu * tr) + b * nb2 * nb2 * dd * dd;
    }
}

/// What a shrunk accumulator keeps beside its moments: the fourth moments
/// Ledoit and Wolf's intensity reads (absent for the others), and Kish's
/// size as of the last row that moved the moments.
///
/// Kish's size is kept, not read from `W` and `Q` each row, because a row
/// that only ages the moments -- one of weight 0, or a gap -- multiplies
/// `W` by `λ` and `Q` by `λ²`, and `(λW)² / (λ²Q)` is not `W² / Q` to the
/// last bit: read afresh it would move the intensity, and the shrunk
/// matrix, on a row that learned nothing (hard rule 9). 8 bytes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ShrinkState {
    pub(crate) fourth: Option<Fourth>,
    pub(crate) kish: Option<f64>,
}

impl ShrinkState {
    pub fn new(shrinkage: &Shrinkage, k: usize) -> Self {
        Self {
            fourth: shrinkage.needs_fourth().then(|| Fourth::new(k)),
            kish: None,
        }
    }

    /// Whether it holds what `shrinkage` reads, of `k` features.
    pub fn fits(&self, shrinkage: &Shrinkage, k: usize) -> bool {
        match &self.fourth {
            Some(f) => shrinkage.needs_fourth() && f.has_shape(k),
            None => !shrinkage.needs_fourth(),
        }
    }

    /// Kish's size as of the last learned row.
    pub fn kish(&self) -> Option<f64> {
        self.kish
    }

    /// The fourth moments, under Ledoit and Wolf's intensity.
    pub fn fourth(&self) -> Option<&Fourth> {
        self.fourth.as_ref()
    }
}

/// The intensity `shrinkage` gives the co-moments `c` (packed lower rows,
/// `k × k`), with Kish's size `n` and the rows' fourth moments `fourth`
/// (Ledoit and Wolf's alone reads them).
pub(crate) fn intensity(
    shrinkage: &Shrinkage,
    c: &[f64],
    k: usize,
    n: Option<f64>,
    fourth: Option<f64>,
) -> f64 {
    match shrinkage {
        Shrinkage::Fixed { intensity, .. } => *intensity,
        Shrinkage::Oas => {
            let (tr, frob2) = trace_and_frobenius(c, k);
            oas(k, tr, frob2, n)
        }
        Shrinkage::LedoitWolf => {
            let (tr, frob2) = trace_and_frobenius(c, k);
            match fourth {
                Some(f) => ledoit_wolf(k, tr, frob2, f, n),
                None => f64::NAN,
            }
        }
    }
}

#[cfg(test)]
mod tests;
