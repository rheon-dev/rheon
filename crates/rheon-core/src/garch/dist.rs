//! The standardized innovation densities (mean 0, variance 1) and their
//! scores: Student's t, and Hansen's (1994) skewed t, as `arch` writes them
//! (`StudentsT`, `SkewStudent`).
//!
//! ```text
//! c        = Γ((ν+1)/2) / (√(π (ν−2)) Γ(ν/2))
//! a        = 4 λ c (ν−2)/(ν−1),     b = √(1 + 3λ² − a²)
//! sg       = sign(z + a/b),         q = 1 + sg λ,     m = (b z + a)/q
//! log f(z) = ln b + ln c − (ν+1)/2 ln(1 + m²/(ν−2))
//! ```
//!
//! `λ = 0` is Student's t (`a = 0`, `b = 1`, `m = z`). The score of the
//! variance parameters runs through `z = e/√s` only: `∂ l/∂θ = ½ w ∂ ln s/∂θ`
//! with `w = (ν+1) m b z / (q (ν−2+m²)) − 1`, `(ν+1) z²/(ν−2+z²) − 1` for the
//! t. The shape's score is closed too, with `D = ν−2+m²` and
//! `(ln c)_ν = ½ [ψ((ν+1)/2) − ψ(ν/2) − 1/(ν−2)]`:
//!
//! ```text
//! a_ν = a (ln c)_ν + 4 λ c/(ν−1)²     a_λ = 4 c (ν−2)/(ν−1)
//! b_ν = −a a_ν / b                    b_λ = (3λ − a a_λ)/b
//! m_ν = (b_ν z + a_ν)/q               m_λ = (b_λ z + a_λ)/q − sg m/q
//! ∂/∂ν = b_ν/b + (ln c)_ν − ½ ln(1 + m²/(ν−2)) − (ν+1)(2 m m_ν (ν−2) − m²)/(2 (ν−2) D)
//! ∂/∂λ = b_λ/b − (ν+1) m m_λ / D
//! ```
//!
//! The branch point `z = −a/b` moves with the shape, but `m = 0` there on
//! both branches, so `log f` is continuous in it and no term comes from the
//! moving kink. At `λ = 0` the `ν` score is the t's, term for term.
//!
//! `ln Γ` and `ψ` are written here (std has neither): each is shifted by its
//! recurrence to an argument of at least 20 and read from its asymptotic
//! series there, whose first term left out is below `1e-17` there.

use super::math;

/// `ν`'s bounds, `arch`'s: 2.05 to 300.
pub(super) const NU_LO: f64 = 2.05;
pub(super) const NU_HI: f64 = 300.0;
/// `|λ| ≤ 0.99`.
pub(super) const LAM_LIM: f64 = 0.99;

const LN_SQRT_2PI: f64 = 0.918_938_533_204_672_7;

/// `ln Γ(x)` for `x > 0`.
pub(super) fn ln_gamma(mut x: f64) -> f64 {
    let mut shift = 0.0;
    while x < 20.0 {
        shift += math::ln(x);
        x += 1.0;
    }
    let r = 1.0 / x;
    let r2 = r * r;
    let series = r
        * (1.0 / 12.0
            + r2 * (-1.0 / 360.0
                + r2 * (1.0 / 1260.0 + r2 * (-1.0 / 1680.0 + r2 * (1.0 / 1188.0)))));
    (x - 0.5) * math::ln(x) - x + LN_SQRT_2PI + series - shift
}

/// The digamma function `ψ(x) = d ln Γ(x)/dx` for `x > 0`.
pub(super) fn digamma(mut x: f64) -> f64 {
    let mut shift = 0.0;
    while x < 20.0 {
        shift += 1.0 / x;
        x += 1.0;
    }
    let r = 1.0 / x;
    let r2 = r * r;
    let series = r2
        * (1.0 / 12.0
            - r2 * (1.0 / 120.0 - r2 * (1.0 / 252.0 - r2 * (1.0 / 240.0 - r2 * (1.0 / 132.0)))));
    math::ln(x) - 0.5 * r - series - shift
}

/// `(a, b, c)` of the density at `(ν, λ)`.
fn consts(nu: f64, lam: f64) -> (f64, f64, f64) {
    let c = math::exp(ln_gamma((nu + 1.0) / 2.0) - ln_gamma(nu / 2.0))
        / (std::f64::consts::PI * (nu - 2.0)).sqrt();
    let a = 4.0 * lam * c * (nu - 2.0) / (nu - 1.0);
    let b = (1.0 + 3.0 * lam * lam - a * a).sqrt();
    (a, b, c)
}

/// `log f(z)` of the skewed t at `(ν, λ)`; `λ = 0` is Student's t.
pub(super) fn log_density(z: f64, nu: f64, lam: f64) -> f64 {
    let (a, b, c) = consts(nu, lam);
    let sg = if z > -a / b { 1.0 } else { -1.0 };
    let m = (b * z + a) / (1.0 + sg * lam);
    math::ln(b) + math::ln(c) - 0.5 * (nu + 1.0) * math::ln_1p(m * m / (nu - 2.0))
}

/// The scores of one row's `z`: `w`, the factor of the variance
/// parameters' score `½ w ∂ ln s/∂θ`, and the shape's `(∂/∂ν, ∂/∂λ)`
/// (`∂/∂λ` is 0 for the t, `skew` false).
pub(super) fn scores(z: f64, nu: f64, lam: f64, skew: bool) -> (f64, f64, f64) {
    let n2 = nu - 2.0;
    let dlnc = 0.5 * (digamma((nu + 1.0) / 2.0) - digamma(nu / 2.0) - 1.0 / n2);
    if !skew {
        let q = z * z;
        let w = (nu + 1.0) * q / (n2 + q) - 1.0;
        let s_nu = dlnc - 0.5 * math::ln_1p(q / n2) + (nu + 1.0) * q / (2.0 * n2 * (n2 + q));
        return (w, s_nu, 0.0);
    }
    let (a, b, c) = consts(nu, lam);
    let sg = if z > -a / b { 1.0 } else { -1.0 };
    let q = 1.0 + sg * lam;
    let m = (b * z + a) / q;
    let dd = n2 + m * m;
    let w = (nu + 1.0) * m * b * z / (q * dd) - 1.0;
    let a_nu = a * dlnc + 4.0 * lam * c / ((nu - 1.0) * (nu - 1.0));
    let a_lam = 4.0 * c * n2 / (nu - 1.0);
    let b_nu = -a * a_nu / b;
    let b_lam = (3.0 * lam - a * a_lam) / b;
    let m_nu = (b_nu * z + a_nu) / q;
    let m_lam = (b_lam * z + a_lam) / q - sg * m / q;
    let s_nu = b_nu / b + dlnc
        - 0.5 * math::ln_1p(m * m / n2)
        - (nu + 1.0) * (2.0 * m * m_nu * n2 - m * m) / (2.0 * n2 * dd);
    let s_lam = b_lam / b - (nu + 1.0) * m * m_lam / dd;
    (w, s_nu, s_lam)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ln_gamma_and_digamma_meet_their_closed_values() {
        // Γ(n) = (n − 1)!, Γ(1/2) = √π, ψ(1) = −γ_E, ψ(1/2) = −γ_E − 2 ln 2
        let euler = std::f64::consts::EULER_GAMMA;
        let mut fact: f64 = 1.0;
        for n in 1..40 {
            let got = ln_gamma(n as f64);
            assert!(
                (got - fact.ln()).abs() <= 1e-13 * fact.ln().abs().max(1.0),
                "{n}"
            );
            fact *= n as f64;
        }
        assert!((ln_gamma(0.5) - std::f64::consts::PI.sqrt().ln()).abs() < 1e-14);
        assert!((digamma(1.0) + euler).abs() < 1e-14);
        assert!((digamma(0.5) + euler + 2.0 * std::f64::consts::LN_2).abs() < 1e-14);
        // ψ(x + 1) = ψ(x) + 1/x across the shift point
        for x in [1.3, 7.9, 9.5, 12.25, 150.0] {
            assert!(
                (digamma(x + 1.0) - digamma(x) - 1.0 / x).abs() < 1e-13,
                "{x}"
            );
        }
        // ψ is the derivative of ln Γ
        for x in [1.025, 3.5, 10.0, 150.5] {
            let h = 1e-5 * x;
            let num = (ln_gamma(x + h) - ln_gamma(x - h)) / (2.0 * h);
            assert!((num - digamma(x)).abs() < 1e-8, "{x}");
        }
    }

    /// The density integrates to 1 with mean 0 and variance 1 (midpoint
    /// rule over [-60, 60]), at a heavy and a light tail, either skew.
    #[test]
    fn the_density_is_standardized() {
        for (nu, lam) in [(6.0, 0.0), (6.0, -0.2), (3.5, 0.5), (30.0, 0.9)] {
            let (mut m0, mut m1, mut m2) = (0.0, 0.0, 0.0);
            let h = 1e-3;
            let mut z = -60.0 + 0.5 * h;
            while z < 60.0 {
                let f = log_density(z, nu, lam).exp() * h;
                m0 += f;
                m1 += f * z;
                m2 += f * z * z;
                z += h;
            }
            assert!((m0 - 1.0).abs() < 2e-4, "{nu} {lam}: {m0}");
            assert!(m1.abs() < 2e-3, "{nu} {lam}: {m1}");
            assert!((m2 - 1.0).abs() < 2e-2, "{nu} {lam}: {m2}");
        }
    }

    /// Each score against a central difference of the log density.
    #[test]
    fn the_scores_are_the_log_densitys_derivatives() {
        for (nu, lam) in [
            (6.0, -0.2),
            (6.0, 0.0),
            (3.0, 0.5),
            (4.5, -0.7),
            (30.0, 0.3),
        ] {
            for z in [-4.0, -1.3, -0.2, 0.1, 0.9, 2.5, 7.0] {
                let skew = lam != 0.0;
                let (a, b, _) = consts(nu, lam);
                if (z + a / b).abs() < 1e-2 {
                    continue;
                }
                let (w, s_nu, s_lam) = scores(z, nu, lam, skew);
                // w: d log f/d ln s = −½ z ∂ log f/∂z − ½, so w = −z f'/f − 1
                let h = 1e-6;
                let dz = (log_density(z + h, nu, lam) - log_density(z - h, nu, lam)) / (2.0 * h);
                assert!((w - (-z * dz - 1.0)).abs() < 1e-6, "w {nu} {lam} {z}");
                let hn = 1e-6 * nu;
                let num =
                    (log_density(z, nu + hn, lam) - log_density(z, nu - hn, lam)) / (2.0 * hn);
                assert!((s_nu - num).abs() < 1e-6, "nu {nu} {lam} {z}: {s_nu} {num}");
                if skew {
                    let num =
                        (log_density(z, nu, lam + h) - log_density(z, nu, lam - h)) / (2.0 * h);
                    assert!((s_lam - num).abs() < 1e-6, "lam {nu} {lam} {z}");
                }
                // the skewed form at lambda 0 is the t's
                if !skew {
                    let (w2, s2, _) = scores(z, nu, 0.0, true);
                    assert!((w2 - w).abs() < 1e-12 && (s2 - s_nu).abs() < 1e-12);
                }
            }
        }
    }
}
