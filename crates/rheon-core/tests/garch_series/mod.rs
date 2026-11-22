//! GARCH-family series at the processes `arch` is the oracle of in
//! `tests/test_second_opinion_garch.py`, for the goldens and the state
//! fixtures that need a process the start can identify (PLAN 550): a
//! realistic case beside the degenerate ones, whose start has no
//! conditional variance to find.
//!
//! Every number is the same bits on every platform: the innovations are
//! drawn with no math library, a normal as the sum of twelve uniforms less
//! six, a t as a normal over the square root of a mean of squared normals
//! (`sqrt` is correctly rounded), and Hansen's skewed t from that t by its
//! two half-scales; EGARCH's `exp` is the `libm` crate's.

#![allow(dead_code)]

/// Uniform draws from a 64-bit linear congruential generator.
pub struct Draws(u64);

impl Draws {
    pub fn new(seed: u64) -> Self {
        Self(seed ^ 0x9e37_79b9_7f4a_7c15)
    }

    /// A uniform in `(0, 1)`.
    pub fn uniform(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    /// Mean 0, variance 1: twelve uniforms less six.
    pub fn normal(&mut self) -> f64 {
        (0..12).map(|_| self.uniform()).sum::<f64>() - 6.0
    }

    /// Student's t of 6 degrees of freedom, standardized to variance 1:
    /// `N / √(χ²₆/6) · √(4/6)`, `arch`'s `StudentsT`.
    pub fn t6(&mut self) -> f64 {
        let n = self.normal();
        let chi: f64 = (0..6)
            .map(|_| {
                let x = self.normal();
                x * x
            })
            .sum();
        n / (chi / 6.0).sqrt() * (4.0f64 / 6.0).sqrt()
    }

    /// Hansen's skewed t of 6 degrees of freedom and skew `lam`, `arch`'s
    /// `SkewStudent`: `x` the standardized t, folded to the left with
    /// chance `(1 − λ)/2` and scaled by `1 − λ`, to the right by `1 + λ`,
    /// then `(y − a)/b`. At `ν = 6` Hansen's `c` is `Γ(7/2)/(√(4π) Γ(3)) =
    /// 15/32`, so `a = 4λc(ν − 2)/(ν − 1) = 1.5 λ` and `b² = 1 + 3λ² − a²`.
    pub fn skewt6(&mut self, lam: f64) -> f64 {
        let x = self.t6().abs();
        let y = if self.uniform() < 0.5 * (1.0 - lam) {
            -(1.0 - lam) * x
        } else {
            (1.0 + lam) * x
        };
        let a = 1.5 * lam;
        let b = (1.0 + 3.0 * lam * lam - a * a).sqrt();
        (y - a) / b
    }
}

/// The innovations' distribution.
#[derive(Clone, Copy, Debug)]
pub enum Innovations {
    Normal,
    T6,
    SkewT6(f64),
}

impl Innovations {
    fn draw(self, d: &mut Draws) -> f64 {
        match self {
            Innovations::Normal => d.normal(),
            Innovations::T6 => d.t6(),
            Innovations::SkewT6(lam) => d.skewt6(lam),
        }
    }
}

/// `n` rows of a GJR at `(ω, α, γ, β)` (`γ = 0` a GARCH) after 1,000
/// discarded, started at its unconditional variance.
pub fn gjr(theta: [f64; 4], z: Innovations, n: usize, seed: u64) -> Vec<f64> {
    let mut d = Draws::new(seed);
    let [w, a, g, b] = theta;
    let mut s = w / (1.0 - a - 0.5 * g - b);
    let mut e: f64 = 0.0;
    let mut out = Vec::with_capacity(n);
    for t in 0..n + 1000 {
        s = w + (a + if e < 0.0 { g } else { 0.0 }) * e * e + b * s;
        e = s.sqrt() * z.draw(&mut d);
        if t >= 1000 {
            out.push(e);
        }
    }
    out
}

/// `n` rows of an EGARCH at `(ω, α, γ, β)`, normal innovations, after
/// 1,000 discarded.
pub fn egarch(theta: [f64; 4], n: usize, seed: u64) -> Vec<f64> {
    let mut d = Draws::new(seed);
    let [w, a, g, b] = theta;
    // √(2/π), E|z| of a normal
    let abs_z = 0.797_884_560_802_865_4;
    let (mut lv, mut zp) = (w / (1.0 - b), 0.0f64);
    let mut out = Vec::with_capacity(n);
    for t in 0..n + 1000 {
        lv = w + a * (zp.abs() - abs_z) + g * zp + b * lv;
        let z = d.normal();
        if t >= 1000 {
            out.push(libm::exp(0.5 * lv) * z);
        }
        zp = z;
    }
    out
}

/// `n` rows of a FIGARCH(1, d, 1) at `(ω, φ, d, β)` cut at `m` lags, normal
/// innovations, after 3,000 discarded; the lags before the first row its
/// long-run variance. The weights by their recursion: `λ₁ = φ − β + d`,
/// `λⱼ = β λⱼ₋₁ + δⱼ − φ δⱼ₋₁`, `δ₁ = d`, `δⱼ = δⱼ₋₁ (j − 1 − d)/j`.
pub fn figarch(theta: [f64; 4], m: usize, n: usize, seed: u64) -> Vec<f64> {
    let mut d = Draws::new(seed);
    let [w, phi, fd, b] = theta;
    let mut lam = vec![0.0; m + 1];
    let mut delta = fd;
    lam[1] = phi - b + fd;
    for j in 2..=m {
        let prev = delta;
        delta = prev * (j as f64 - 1.0 - fd) / j as f64;
        lam[j] = b * lam[j - 1] + delta - phi * prev;
    }
    let level = w / (1.0 - b);
    let sum: f64 = lam[1..].iter().sum();
    // the lags, newest first: e²ₜ₋₁ at 0
    let mut lags = std::collections::VecDeque::from(vec![level / (1.0 - sum); m]);
    let mut out = Vec::with_capacity(n);
    for t in 0..n + 3000 {
        let s = level + (1..=m).map(|j| lam[j] * lags[j - 1]).sum::<f64>();
        let e = s.sqrt() * d.normal();
        lags.pop_back();
        lags.push_front(e * e);
        if t >= 3000 {
            out.push(e);
        }
    }
    out
}
