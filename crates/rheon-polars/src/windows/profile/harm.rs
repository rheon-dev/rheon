//! The harmonic form of a profile: a Fourier series in the phase, fitted
//! by exponentially weighted least squares on the group's rows.
//!
//! With the phase `φ ∈ [0, 1)` and `K` harmonics, a row's basis is
//! `z(φ) = (1, cos 2πφ, sin 2πφ, …, cos 2πKφ, sin 2πKφ)`, `p = 2K + 1`
//! terms. The fit keeps the weighted Gram `G = Σ w d z zᵀ` and the moments
//! `r₁ = Σ w d z x` and `r₂ = Σ w d z x²`, `d` each row's decay to the row
//! read. The mean profile at `φ` is `z(φ)ᵀ G⁻¹ r₁`; the variance is the
//! fit of `x²` less the square of the fit of `x`, `z(φ)ᵀ G⁻¹ r₂ − (z(φ)ᵀ
//! G⁻¹ r₁)²`, withheld where it is negative. A decay scales `G` and both
//! moments alike, so it drops out of the coefficients.
//!
//! **Withheld** where the rows read are fewer than `min_samples`, or where
//! `G` is not positive definite as it stands (fewer than `p` distinct
//! phases, or as good as): its Cholesky factor meets a pivot at or below
//! `PIVOT` times `G`'s largest diagonal entry, where rounding alone would
//! pass a rank-deficient `G` as definite. No jitter is added: a fit on a
//! jitter is no fit of the rows. The fit's
//! weight at `φ`, `n_eff(φ) = 1 / (z(φ)ᵀ G⁻¹ z(φ))`, the reciprocal of its
//! variance factor there, must be finite and above 0 too: for the
//! indicator basis of a bucket it is the bucket's weight exactly.
//!
//! **Under a window** the rows are kept in two stacks, as a bucket's are:
//! each front row keeps the Gram and moments of itself and every front row
//! newer than it, so a row leaves by dropping its own, never by a
//! subtraction (a rank-one downdate of `G`), whose roundings would build up
//! over a long stream.

use std::collections::VecDeque;

use rheon_core::ClockValue;
use serde::{Deserialize, Serialize};

use super::super::KernelDef;

/// A Cholesky pivot at or below this share of the Gram's largest diagonal
/// entry is a Gram that is not positive definite as it stands: a
/// rank-deficient Gram's pivots are rounding, about `1e-16` of it.
const PIVOT: f64 = 1e-12;

/// Solve `A X = B` for the `p × p` symmetric `a` (row-major) and the
/// columns of `b` (column-major, `p` a column), by Cholesky with no jitter;
/// `None` where `A` is not positive definite as [`PIVOT`] reads it.
fn solve(a: &[f64], b: &[f64], p: usize) -> Option<Vec<f64>> {
    let scale = (0..p).map(|i| a[i * p + i]).fold(0.0, f64::max);
    if !(scale > 0.0 && scale.is_finite()) {
        return None;
    }
    let mut l = vec![0.0; p * p];
    for j in 0..p {
        let mut d = a[j * p + j];
        for k in 0..j {
            d -= l[j * p + k] * l[j * p + k];
        }
        if d.is_nan() || d <= PIVOT * scale {
            return None;
        }
        let d = d.sqrt();
        l[j * p + j] = d;
        for i in j + 1..p {
            let mut v = a[i * p + j];
            for k in 0..j {
                v -= l[i * p + k] * l[j * p + k];
            }
            l[i * p + j] = v / d;
        }
    }
    let mut x = b.to_vec();
    for col in x.chunks_mut(p) {
        for i in 0..p {
            let mut v = col[i];
            for k in 0..i {
                v -= l[i * p + k] * col[k];
            }
            col[i] = v / l[i * p + i];
        }
        for i in (0..p).rev() {
            let mut v = col[i];
            for k in i + 1..p {
                v -= l[k * p + i] * col[k];
            }
            col[i] = v / l[i * p + i];
        }
    }
    Some(x)
}
use super::{ProfileDef, ProfileStat, outside};

/// The basis at phase `phi`, `p = 2K + 1` terms, into `z`.
pub(super) fn basis(k: u32, phi: f64, z: &mut Vec<f64>) {
    z.clear();
    z.push(1.0);
    for h in 1..=k {
        // The `libm` crate's, not the platform's: the Grams are state, and
        // every platform then holds the same bits.
        let a = std::f64::consts::TAU * f64::from(h) * phi;
        z.push(libm::cos(a));
        z.push(libm::sin(a));
    }
}

/// The weighted Gram of `n` rows and their moments, seen from `tau`: `g`
/// the upper triangle of `G` by rows, `r1` and `r2` the moments of `x` and
/// `x²`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(super) struct Gram {
    n: f64,
    g: Vec<f64>,
    r1: Vec<f64>,
    r2: Vec<f64>,
    tau: f64,
}

impl Gram {
    pub(super) fn empty(p: usize) -> Self {
        Self {
            n: 0.0,
            g: vec![0.0; p * (p + 1) / 2],
            r1: vec![0.0; p],
            r2: vec![0.0; p],
            tau: f64::NAN,
        }
    }

    fn p(&self) -> usize {
        self.r1.len()
    }

    /// Seen from `tau`, at or after its own.
    fn at(&self, k: &KernelDef, tau: f64) -> Gram {
        if self.n == 0.0 {
            return Gram {
                tau,
                ..Gram::empty(self.p())
            };
        }
        let d = k.discount(tau - self.tau);
        Gram {
            n: self.n,
            g: self.g.iter().map(|v| v * d).collect(),
            r1: self.r1.iter().map(|v| v * d).collect(),
            r2: self.r2.iter().map(|v| v * d).collect(),
            tau,
        }
    }

    /// Add a row of basis `z`, value `x` and weight `w`, seen from `self.tau`.
    fn add(&mut self, z: &[f64], x: f64, w: f64) {
        let p = self.p();
        self.n += 1.0;
        let mut at = 0;
        for i in 0..p {
            let wz = w * z[i];
            for zj in &z[i..p] {
                self.g[at] += wz * zj;
                at += 1;
            }
            self.r1[i] += wz * x;
            self.r2[i] += wz * x * x;
        }
    }

    /// The sum of two seen from one time.
    fn merge(mut self, o: &Gram) -> Gram {
        if o.n == 0.0 {
            return self;
        }
        if self.n == 0.0 {
            return Gram {
                tau: self.tau,
                ..o.clone()
            };
        }
        self.n += o.n;
        for (a, b) in self.g.iter_mut().zip(&o.g) {
            *a += b;
        }
        for (a, b) in self.r1.iter_mut().zip(&o.r1) {
            *a += b;
        }
        for (a, b) in self.r2.iter_mut().zip(&o.r2) {
            *a += b;
        }
        self
    }

    /// The fit at basis `z`: the mean, or the variance, per `def`; NaN where
    /// it is withheld.
    fn read(&self, def: &ProfileDef, z: &[f64]) -> f64 {
        if self.n < f64::from(def.min_samples.max(1)) {
            return f64::NAN;
        }
        let p = self.p();
        let mut a = vec![0.0; p * p];
        let mut at = 0;
        for i in 0..p {
            for j in i..p {
                a[i * p + j] = self.g[at];
                a[j * p + i] = self.g[at];
                at += 1;
            }
        }
        let var = def.stat != ProfileStat::Mean;
        // The right-hand sides: `r₁`, `r₂` where a variance reads it, and
        // `z` for the weight at `φ`.
        let mut b = self.r1.clone();
        if var {
            b.extend_from_slice(&self.r2);
        }
        b.extend_from_slice(z);
        let m = b.len() / p;
        let Some(beta) = solve(&a, &b, p) else {
            return f64::NAN;
        };
        let dot = |c: &[f64]| c.iter().zip(z).map(|(c, z)| c * z).sum::<f64>();
        let n_eff = 1.0 / dot(&beta[(m - 1) * p..]);
        if !(n_eff.is_finite() && n_eff > 0.0) {
            return f64::NAN;
        }
        let mean = dot(&beta[..p]);
        if !var {
            return mean;
        }
        let v = dot(&beta[p..]) - mean * mean;
        if v.is_nan() || v < 0.0 {
            return f64::NAN;
        }
        if def.stat == ProfileStat::Std {
            v.sqrt()
        } else {
            v
        }
    }

    /// The fit's weight at basis `z`, `1 / (zᵀ G⁻¹ z)`; NaN where `G` is
    /// not positive definite.
    #[cfg(test)]
    fn n_eff(&self, z: &[f64]) -> f64 {
        let p = self.p();
        let mut a = vec![0.0; p * p];
        let mut at = 0;
        for i in 0..p {
            for j in i..p {
                a[i * p + j] = self.g[at];
                a[j * p + i] = self.g[at];
                at += 1;
            }
        }
        match solve(&a, z, p) {
            Some(x) => 1.0 / x.iter().zip(z).map(|(x, z)| x * z).sum::<f64>(),
            None => f64::NAN,
        }
    }

    fn fits(&self, p: usize) -> bool {
        self.g.len() == p * (p + 1) / 2 && self.r1.len() == p && self.r2.len() == p && self.n >= 0.0
    }
}

/// A row of the harmonic window: its clock, time, phase, value and weight.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
struct HRow {
    at: ClockValue,
    tau: f64,
    phi: f64,
    x: f64,
    w: f64,
}

/// One group's harmonic fit, in two stacks under a window.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(super) struct Harm {
    rows: VecDeque<HRow>,
    front: usize,
    suffix: VecDeque<Gram>,
    back: Gram,
}

impl Harm {
    pub(super) fn new(def: &ProfileDef) -> Self {
        Self {
            rows: VecDeque::new(),
            front: 0,
            suffix: VecDeque::new(),
            back: Gram::empty(def.terms()),
        }
    }

    /// The bytes it is counted at beyond the fit's own.
    pub(super) fn bytes(&self, def: &ProfileDef) -> u64 {
        self.rows.len() as u64 * def.row_bytes()
    }

    pub(super) fn fits(&self, def: &ProfileDef) -> bool {
        let p = def.terms();
        self.front <= self.rows.len()
            && self.suffix.len() == self.front
            && (def.window.is_some() || self.rows.is_empty())
            && self.back.fits(p)
            && self.suffix.iter().all(|g| g.fits(p))
            && self
                .rows
                .iter()
                .all(|r| r.w > 0.0 && r.x.is_finite() && (0.0..1.0).contains(&r.phi))
    }

    /// The rows the window would drop at `now`.
    pub(super) fn leaving(&self, def: &ProfileDef, now: ClockValue) -> u64 {
        def.window.map_or(0, |w| {
            self.rows
                .iter()
                .take_while(|r| outside(w, def.closed, r.at, now))
                .count() as u64
        })
    }

    pub(super) fn evict(&mut self, def: &ProfileDef, k: &KernelDef, now: ClockValue) {
        let Some(w) = def.window else {
            return;
        };
        while self
            .rows
            .front()
            .is_some_and(|r| outside(w, def.closed, r.at, now))
        {
            if self.front == 0 {
                self.turn_over(def, k);
            }
            self.rows.pop_front();
            self.suffix.pop_front();
            self.front -= 1;
        }
    }

    fn turn_over(&mut self, def: &ProfileDef, k: &KernelDef) {
        let n = self.rows.len();
        self.front = n;
        self.suffix.clear();
        let p = def.terms();
        if let Some(newest) = self.rows.back() {
            let mut acc = Gram {
                tau: newest.tau,
                ..Gram::empty(p)
            };
            let mut z = Vec::with_capacity(p);
            let mut out = Vec::with_capacity(n);
            for i in (0..n).rev() {
                let r = self.rows[i];
                basis(def.harmonics, r.phi, &mut z);
                acc.add(&z, r.x, r.w * k.discount(newest.tau - r.tau));
                out.push(acc.clone());
            }
            out.reverse();
            self.suffix = out.into();
        }
        self.back = Gram::empty(p);
    }

    /// A row joins the back stack: its clock, time and phase in `at`,
    /// `tau` and `phi`, its value `x` and weight `w`.
    pub(super) fn push(
        &mut self,
        def: &ProfileDef,
        k: &KernelDef,
        row: (ClockValue, f64, f64),
        x: f64,
        w: f64,
    ) {
        let (at, tau, phi) = row;
        let mut z = Vec::with_capacity(def.terms());
        basis(def.harmonics, phi, &mut z);
        let mut b = self.back.at(k, tau);
        b.add(&z, x, w);
        self.back = b;
        if def.window.is_some() {
            self.rows.push_back(HRow { at, tau, phi, x, w });
        }
    }

    /// The fit read at phase `phi` and time `tau`.
    pub(super) fn read(&self, def: &ProfileDef, k: &KernelDef, phi: f64, tau: f64) -> f64 {
        let front = self.suffix.front().map_or_else(
            || Gram {
                tau,
                ..Gram::empty(def.terms())
            },
            |g| g.at(k, tau),
        );
        let all = front.merge(&self.back.at(k, tau));
        let mut z = Vec::with_capacity(def.terms());
        basis(def.harmonics, phi, &mut z);
        all.read(def, &z)
    }
}

/// The bytes of one Gram and its moments: `p(p+1)/2 + 2p + 2` numbers.
pub(super) fn gram_bytes(p: usize) -> u64 {
    8 * (p * (p + 1) / 2 + 2 * p + 2) as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    /// On `N` evenly spaced phases at equal weights `w`, `ZᵀWZ` is
    /// `w · diag(N, N/2, …, N/2)` (the discrete orthogonality of the
    /// Fourier basis, `N > 2K`), so `zᵀ (ZᵀWZ)⁻¹ z = (1 + 2K) / (N w)` at
    /// every phase and `n_eff = N w / (2K + 1)`: the closed form the fit's
    /// weight is held to, at phases on and off the grid.
    #[test]
    fn n_eff_is_the_reciprocal_of_the_variance_factor() {
        for (k, n, w) in [(1u32, 7usize, 1.0), (2, 12, 0.5), (3, 29, 2.0)] {
            let p = 2 * k as usize + 1;
            let mut g = Gram {
                tau: 0.0,
                ..Gram::empty(p)
            };
            let mut z = Vec::new();
            for i in 0..n {
                basis(k, i as f64 / n as f64, &mut z);
                g.add(&z, 1.0, w);
            }
            for phi in [0.0, 0.13, 0.5, 0.91] {
                basis(k, phi, &mut z);
                let want = n as f64 * w / p as f64;
                let got = g.n_eff(&z);
                assert!(
                    (got - want).abs() < 1e-9 * want,
                    "K={k} φ={phi}: {got} {want}"
                );
            }
        }
        // Fewer distinct phases than terms: no fit, and no weight.
        let mut g = Gram {
            tau: 0.0,
            ..Gram::empty(5)
        };
        let mut z = Vec::new();
        for phi in [0.1, 0.4, 0.7] {
            basis(2, phi, &mut z);
            g.add(&z, 1.0, 1.0);
        }
        basis(2, 0.2, &mut z);
        assert!(g.n_eff(&z).is_nan());
    }
}
