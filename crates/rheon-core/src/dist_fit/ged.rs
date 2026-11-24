// The fit that calls these functions lands in the next part of the GED family;
// until then they are reached by the tests only.
#![allow(dead_code)]

//! The generalized error distribution (GED, scipy's `gennorm`) for
//! `dist_fit`, in the form that makes its shape `ν` a tail exponent: with
//! `s` the scipy scale and `z = (y − μ)/s`, the density is
//!
//! ```text
//! f(y) = ν / (2 s Γ(1/ν)) · exp(−|z|^ν)
//! ```
//!
//! `ν = 2` is the normal, `ν = 1` the Laplace. The standard deviation is
//! `σ = s √(Γ(3/ν)/Γ(1/ν))`, and the fit is stated in `(μ, ln s, ν)`, with
//! `σ` reported. Let `g = 1/ν` and `T = |z|^ν`. Then `T ~ Gamma(g, 1)`,
//! which gives every moment below in closed form.
//!
//! **Log density** (per unit weight), and its scores in `(μ, ln s, ν)`:
//!
//! ```text
//! ln f = ln ν − ln 2 − ln s − ln Γ(g) − T
//! ∂/∂μ      = ν z |z|^(ν−2) / s
//! ∂/∂ln s   = ν T − 1
//! ∂/∂ν      = g + g² ψ(g) − T ln|z|
//! ```
//!
//! **Expected (Fisher) information**, from the moments of `T` and `T ln T`
//! (the integrals `E[T^a] = Γ(g+a)/Γ(g)`, differentiated in `a`):
//!
//! ```text
//! I_μμ = ν² Γ(2 − g) / (Γ(g) s²),   I_μ,· = 0 (the μ score is odd in z)
//! I_lns,lns = ν,                     I_lns,ν = −g (1 + ψ(g+1))
//! I_νν = g² [ g(g+1)(ψ(g+2)² + ψ′(g+2)) − g² ψ(g+1)² ]
//! ```
//!
//! At `ν = 2` the information in `ln s` is `2` and `I_μμ = 2/s²`, the
//! normal's. **CDF and quantile**: `P(Z ≤ z) = ½ + ½ sign(z) P(g, |z|^ν)`
//! with `P` the regularized lower incomplete gamma, so the quantile is the
//! gamma quantile raised to `g`. **Lower expected shortfall** at level `α`
//! (standardized, `s = 1`): with `t_c` the `Q(g, t_c) = 2α` point,
//! `E[Z | Z ≤ −c] = −Γ(2g) Q(2g, t_c) / (Γ(g) Q(g, t_c))`.
//!
//! Standardized moments are in units of `s = 1`; the callers scale them.

use crate::special::{digamma, gamma_q, gamma_q_inv, ln_gamma, trigamma};

/// `ln(s/σ) = ½ (ln Γ(g) − ln Γ(3g))`: the scipy scale per unit of standard
/// deviation, with `g = 1/ν`.
pub(crate) fn ln_scale_per_sigma(nu: f64) -> f64 {
    let g = 1.0 / nu;
    0.5 * (ln_gamma(g) - ln_gamma(3.0 * g))
}

/// The log density of the standardized GED (`s = 1`) at `z`.
pub(crate) fn log_density(z: f64, nu: f64) -> f64 {
    let g = 1.0 / nu;
    nu.ln() - std::f64::consts::LN_2 - ln_gamma(g) - power(z, nu)
}

/// `|z|^p`, with `0^p = 0` for `p > 0` (and the power of `0` is never used
/// where `p ≤ 0`: the callers take none there).
fn power(z: f64, p: f64) -> f64 {
    let a = z.abs();
    if a == 0.0 { 0.0 } else { a.powf(p) }
}

/// The scores of the standardized GED at `z` (`s = 1`), per unit weight, in
/// `(μ, ln s, ν)`. The `μ` score is `ν sign(z) |z|^(ν−1)`, zero at `z = 0`
/// where `ν > 1`.
pub(crate) fn scores(z: f64, nu: f64) -> [f64; 3] {
    let g = 1.0 / nu;
    let t = power(z, nu);
    let d_mu = if z == 0.0 {
        0.0
    } else {
        nu * z.signum() * power(z, nu - 1.0)
    };
    let d_nu = if t == 0.0 {
        g + g * g * digamma(g)
    } else {
        g + g * g * digamma(g) - t * z.abs().ln()
    };
    [d_mu, nu * t - 1.0, d_nu]
}

/// The expected information of the standardized GED (`s = 1`), upper
/// triangle in `(μ, ln s, ν)`: `[I_μμ, I_μs, I_μν, I_ss, I_sν, I_νν]`. The
/// `μ` row is orthogonal to the others by symmetry.
pub(crate) fn expected_info(nu: f64) -> [f64; 6] {
    let g = 1.0 / nu;
    let i_mm = nu * nu * (ln_gamma(2.0 - g) - ln_gamma(g)).exp();
    let i_ss = nu;
    let i_sn = -g * (1.0 + digamma(g + 1.0));
    let i_nn = g
        * g
        * (g * (g + 1.0) * (digamma(g + 2.0).powi(2) + trigamma(g + 2.0))
            - g * g * digamma(g + 1.0).powi(2));
    [i_mm, 0.0, 0.0, i_ss, i_sn, i_nn]
}

/// The standardized GED's distribution function at `z`.
pub(crate) fn cdf(z: f64, nu: f64) -> f64 {
    let g = 1.0 / nu;
    // The lower tail is ½Q directly: ½ − ½P would cancel to nothing there.
    let t = power(z, nu);
    if z < 0.0 {
        0.5 * gamma_q(g, t)
    } else {
        1.0 - 0.5 * gamma_q(g, t)
    }
}

/// The standardized GED's quantile at `p`: `−(Q⁻¹(g, 2p))^g` below ½, and
/// `+(Q⁻¹(g, 2(1−p)))^g` above, from the gamma quantile. `±∞` at `0`, `1`.
pub(crate) fn ppf(p: f64, nu: f64) -> f64 {
    let g = 1.0 / nu;
    let lower = p < 0.5;
    let tail = if lower { p } else { 1.0 - p };
    let x = gamma_q_inv(g, 2.0 * tail).powf(g);
    if lower { -x } else { x }
}

/// The lower expected shortfall of the standardized GED at level `α`:
/// `E[Z | Z ≤ ppf(α)]`, negative. See the module's formula.
pub(crate) fn lower_es(alpha: f64, nu: f64) -> f64 {
    let g = 1.0 / nu;
    // The point t_c with Q(g, t_c) = 2α, where P(Z ≤ −c) = α.
    let t_c = gamma_q_inv(g, 2.0 * alpha);
    let ratio = (ln_gamma(2.0 * g) - ln_gamma(g)).exp() * gamma_q(2.0 * g, t_c) / gamma_q(g, t_c);
    -ratio
}

#[cfg(test)]
mod tests;
