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
/// the upper triangle of `G` by rows, and `r1` and `r2` the moments of `x −
/// c` and `(x − c)²`, `c` a shift near the rows' weighted mean.
///
/// The moments are taken about `c` because the variance is a difference:
/// the fit of `x²` less the square of the fit of `x`. Of a level near 100
/// that moves by `1e-3` about its curve, the two are `1e4` and agree to ten
/// digits, so the moments of `x` itself kept six of the variance's digits,
/// and one ulp of a decay factor (another platform's `exp2`) moved it by
/// `5e-9` of itself. Of `x − c` they are of the spread's size. The fit is
/// the same: the basis holds the constant, so the fit of `x − c` is the fit
/// of `x` less `c`, and the fit of `(x − c)²` less the square of the fit
/// of `x − c` is the variance, whatever `c` is. Each row moves `c` to the
/// rows' weighted mean, a step of Welford's, and the moments with it
/// ([`Gram::shift`]); a merge takes the other's moments to this one's `c`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(super) struct Gram {
    n: f64,
    g: Vec<f64>,
    r1: Vec<f64>,
    r2: Vec<f64>,
    tau: f64,
    /// The shift the moments are taken about; 0 takes them of `x` itself.
    #[serde(default)]
    c: f64,
}

impl Gram {
    pub(super) fn empty(p: usize) -> Self {
        Self {
            n: 0.0,
            g: vec![0.0; p * (p + 1) / 2],
            r1: vec![0.0; p],
            r2: vec![0.0; p],
            tau: f64::NAN,
            c: 0.0,
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
            c: self.c,
        }
    }

    /// The moments taken about `m` where they were about `c`: with `δ = c −
    /// m`, `Σ w d zᵢ (x − m) = r1ᵢ + δ G₀ᵢ` and `Σ w d zᵢ (x − m)² = r2ᵢ +
    /// 2δ r1ᵢ + δ² G₀ᵢ`, `G₀ᵢ = Σ w d zᵢ` the first row of `G` (the basis's
    /// first term is 1).
    fn shift(&mut self, m: f64) {
        let delta = self.c - m;
        for i in 0..self.p() {
            let g0 = self.g[i];
            self.r2[i] += delta * (2.0 * self.r1[i] + delta * g0);
            self.r1[i] += delta * g0;
        }
        self.c = m;
    }

    /// Add a row of basis `z`, value `x` and weight `w`, seen from `self.tau`.
    fn add(&mut self, z: &[f64], x: f64, w: f64) {
        let p = self.p();
        self.n += 1.0;
        // The shift moves to the weighted mean with the row in it, `c +
        // (r1₀ + w (x − c)) / W'`, `r1₀ = Σ w d (x − c)` and `W' = Σ w d +
        // w`, above 0: a row is added only at a weight above 0. On the
        // first row it is `x`, to the rounding of `w x / w`.
        self.shift(self.c + (self.r1[0] + w * (x - self.c)) / (self.g[0] + w));
        let e = x - self.c;
        let mut at = 0;
        for i in 0..p {
            let wz = w * z[i];
            for zj in &z[i..p] {
                self.g[at] += wz * zj;
                at += 1;
            }
            self.r1[i] += wz * e;
            self.r2[i] += wz * e * e;
        }
    }

    /// The sum of two seen from one time, the other's moments taken to this
    /// one's shift.
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
        let mut o = o.clone();
        o.shift(self.c);
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
        // The right-hand sides: the moments of `x − c`, of `(x − c)²` where
        // a variance reads it, and `z` for the weight at `φ`.
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
        // The fit of `x − c` at `φ`: the mean less `c`.
        let dev = dot(&beta[..p]);
        if !var {
            return self.c + dev;
        }
        let v = dot(&beta[p..2 * p]) - dev * dev;
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

/// The bytes of one Gram and its moments: `p(p+1)/2 + 2p + 3` numbers,
/// the Gram's upper triangle, the two moments, and the row count, time and
/// shift.
pub(super) fn gram_bytes(p: usize) -> u64 {
    8 * (p * (p + 1) / 2 + 2 * p + 3) as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::windows::profile::{PROFILE_STATE_BYTES, PhaseSpan};
    use crate::windows::{Closed, Direction};

    fn var_def() -> ProfileDef {
        ProfileDef {
            stat: ProfileStat::Var,
            buckets: 0,
            period: PhaseSpan::Units(6.0),
            offset: PhaseSpan::Units(0.0),
            zone: None,
            half_life: 12.0,
            closed: Closed::Right,
            input: 0,
            weight: None,
            min_samples: 1,
            bias: false,
            cap: PROFILE_STATE_BYTES,
            window: None,
            quantile: None,
            harmonics: 1,
        }
    }

    /// A level of 100 with a cycle and noise of `1e-3` about it, as a
    /// price's: `x²` is about `1e10` times the variance about the curve.
    /// With the moments taken of `x` itself, the fit of `x²` less the square
    /// of the fit of `x` kept five of the variance's digits (`2.6e-5`
    /// relative here, measured on the build before), and one ulp of a decay
    /// factor (another platform's `exp2`) moved the variance by `5e-9` of
    /// itself in the state fixture's stream. The variance does not move
    /// under a shift of `x` by a constant, and each `y = x − 100` is exact
    /// here (`y` on a grid of `2⁻³⁰`, `|y| < 2⁻⁸`), so the fit of `y` is the
    /// reference: the same fit with no cancellation worth the name, `y`'s
    /// mean being near 0 against its spread. The gap is read against the
    /// larger of the variance and the rows' mean square about their mean,
    /// the two terms the reference's variance is the difference of. Rows
    /// join under a decay and are read through merges, as a window's two
    /// stacks are, so Grams of different shifts are merged; where the fit
    /// of the squares falls below the square of the fit both withhold
    /// (122 of the 400 reads). Measured on aarch64 (macOS): at most 1.6e-15.
    /// The bound, `1e-12`, is what the state fixtures hold a continuation
    /// to across platforms.
    #[test]
    fn the_variance_keeps_its_digits_at_a_large_level() {
        let def = var_def();
        let k = KernelDef {
            direction: Direction::Backward,
            half_life: 12.0,
            window_size: None,
            window_ns: None,
            closed: Closed::Right,
            since: None,
        };
        let mut s = 198u64;
        let mut r = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        };
        let grid = 2f64.powi(30);
        let mut tau = 0.0f64;
        let (mut front, mut back) = (Gram::empty(3), Gram::empty(3));
        let (mut front0, mut back0) = (Gram::empty(3), Gram::empty(3));
        let mut z = Vec::new();
        let (mut worst, mut compared) = (0.0f64, 0);
        for i in 0..400 {
            tau += 0.25 * (1.0 + (4.0 * (r() + 1.0)).floor());
            let w = 1.0 + 0.5 * r();
            let phi = (tau / 6.0).fract();
            basis(1, phi, &mut z);
            let y = ((1e-3 * z[1] + 1e-3 * r()) * grid).round() / grid;
            let x = 100.0 + y;
            assert_eq!(x - 100.0, y);
            // Every 50 rows the back stack is merged into the front, as a
            // window's turn makes a front, so reads merge Grams whose means
            // differ.
            if i % 50 == 49 {
                front = back.at(&k, tau).merge(&front.at(&k, tau));
                front0 = back0.at(&k, tau).merge(&front0.at(&k, tau));
                (back, back0) = (Gram::empty(3), Gram::empty(3));
            }
            let all = |f: &Gram, b: &Gram| f.at(&k, tau).merge(&b.at(&k, tau));
            let (mine, reference) = (all(&front, &back), all(&front0, &back0));
            let (got, want) = (mine.read(&def, &z), reference.read(&def, &z));
            // Where a fit is withheld (fewer phases than terms, or a fit
            // of the squares below the square of the fit) both are; else the
            // gap is read against the larger of the variance and the
            // rows' mean square about their mean, the size of the two terms
            // the variance is the difference of in the reference.
            assert_eq!(got.is_nan(), want.is_nan(), "row {i}: {got} {want}");
            if !want.is_nan() {
                let scale = want.max(reference.r2[0] / reference.g[0]);
                worst = worst.max((got - want).abs() / scale);
                compared += 1;
            }
            back = back.at(&k, tau);
            back.add(&z, x, w);
            back0 = back0.at(&k, tau);
            back0.add(&z, y, w);
        }
        eprintln!("worst relative gap {worst:e} over {compared} reads");
        assert!(compared > 250, "{compared}");
        assert!(worst < 1e-12, "{worst:e}");
    }

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
