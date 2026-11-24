//! Student's t for `dist_fit`, in scipy's `loc`/`scale`/`df` form, with
//! the tail stepped on `η = 1/ν`. With `z = (y − loc)/scale` and `u = η z²`:
//!
//! ```text
//! log f(y) = −ln B(ν/2, ½) − ½ ln ν − ln scale − ½ (1+η) z² · ln(1+u)/u
//! ```
//!
//! `η = 0` is the normal: the density, its scores and its information are
//! continuous there, which is what lets a test of the tail against the
//! normal be regular (the point is in the parameter set, at its boundary).
//! The scale is scipy's, not the standard deviation: the standard deviation
//! is `scale √(ν/(ν−2))` for `ν > 2`. The `garch` innovations are the same
//! family standardized to variance 1 (`z_std = z √((ν−2)/ν)`) and step the
//! tail on `ln(ν − 2)`, which needs `ν > 2`; here every `ν > 0` is the
//! family, and `1/ν` is the coordinate.
//!
//! **Scores** in `(loc, ln scale, η)`, per unit weight, with `r = 1/(1+u)`:
//!
//! ```text
//! ∂/∂loc     = (1+η) z r / scale
//! ∂/∂ln scale = (1+η) z² r − 1
//! ∂/∂η       = A(η) + ½ z⁴ G(u) − ½ z² r
//! A(η) = −ν² [½ ψ((ν+1)/2) − ½ ψ(ν/2) − 1/(2ν)],   G(u) = (ln(1+u) − u r)/u²
//! ```
//!
//! **Expected (Fisher) information**, closed (Lange, Little and Taylor
//! 1989, mapped to these coordinates; loc is orthogonal to the rest):
//!
//! ```text
//! I_loc,loc = (1+η)/((1+3η) scale²)
//! I_lnσ,lnσ = 2/(1+3η),    I_lnσ,η = 2/((1+η)(1+3η))
//! I_η,η     = ν⁴ [¼ ψ′(ν/2) − ¼ ψ′((ν+1)/2) − (ν+5)/(2ν(ν+1)(ν+3))]
//! ```
//!
//! At `η = 0`: `(1, 2, 2, 7/2)`, positive definite. `A`, `I_η,η` and the
//! density's constant are differences of terms of order `1/ν` whose result
//! is of order 1 after the `ν²` or `ν⁴` factor, so below `η = 1/32` they are
//! read from their power series in `η` (from Stirling's series for `ln Γ`
//! and the asymptotic series of `ψ` and `ψ′`, with Bernoulli numbers,
//! coefficients in exact rational arithmetic); `G` below `u = 1/100` from
//! `Σ_{k≥2} (−1)ᵏ (k−1)/k u^(k−2)`.
//!
//! **CDF and quantile**: `P(Z ≤ −|t|) = ½ I_x(ν/2, ½)` with `x = ν/(ν+t²)`,
//! by the regularized incomplete beta and its inverse. The continued
//! fraction takes `O(√ν)` terms, so below `η = 10⁻⁵` the quantile is
//! Fisher's expansion about the normal quantile `x` (Abramowitz and Stegun
//! 26.7.5),
//!
//! ```text
//! t = x + η (x³+x)/4 + η² (5x⁵+16x³+3x)/96 + η³ (3x⁷+19x⁵+17x³−15x)/384
//! ```
//!
//! whose error is of order `η⁴`, below `10⁻¹⁷` there; the CDF inverts that
//! expansion by Newton steps on `x`.
//!
//! **Expected shortfall** (lower tail, `ν > 1`): `E[Z | Z ≤ t_α] =
//! −(ν + t_α²)/(ν − 1) f(t_α)/α = −(1 + η t_α²)/(1 − η) f(t_α)/α`.

use crate::special::{
    beta_inc, beta_inc_c, beta_inc_inv, digamma, ln_beta, ndtri, norm_cdf, trigamma,
};

/// Below this `η` (`ν > 32`) the series in `η` are read, above it the
/// closed forms.
const ETA_SERIES: f64 = 1.0 / 32.0;
/// Below this `η` the CDF and quantile are Fisher's expansion.
const ETA_EXPAND: f64 = 1e-5;
/// Below this `u` the series of `G(u)`.
const U_SERIES: f64 = 1e-2;
const HALF_LN_TAU: f64 = 0.918_938_533_204_672_8; // ½ ln 2π

/// `A(η)`'s coefficients of `η⁰, η¹, …`.
const A_SERIES: [f64; 20] = [
    -0.25,
    0.0,
    0.125,
    0.0,
    -0.25,
    0.0,
    1.0625,
    0.0,
    -7.75,
    0.0,
    86.375,
    0.0,
    -1365.25,
    0.0,
    29049.03125,
    0.0,
    -800572.75,
    0.0,
    27741322.625,
    0.0,
];
/// `I_η,η(η)`'s coefficients.
const IEE_SERIES: [f64; 20] = [
    3.5,
    -13.0,
    39.5,
    -119.0,
    363.5,
    -1101.0,
    3279.5,
    -9763.0,
    29523.5,
    -89609.0,
    265719.5,
    -778047.0,
    2391483.5,
    -7639237.0,
    21523359.5,
    -50159771.0,
    193710243.5,
    -1135957185.0,
    1743392199.5,
    20741464265.0,
];
/// The coefficients of `ln Γ((ν+1)/2) − ln Γ(ν/2) − ½ ln ν + ½ ln 2`.
const C_SERIES: [f64; 20] = [
    0.0,
    -0.25,
    0.0,
    0.041666666666666664,
    0.0,
    -0.05,
    0.0,
    0.15178571428571427,
    0.0,
    -0.8611111111111112,
    0.0,
    7.8522727272727275,
    0.0,
    -105.01923076923077,
    0.0,
    1936.6020833333334,
    0.0,
    -47092.51470588235,
    0.0,
    1460069.6118421052,
];

fn horner(c: &[f64], x: f64) -> f64 {
    c.iter().rev().fold(0.0, |acc, &k| acc * x + k)
}

/// The density's constant, `log f` at `z = 0` with scale 1:
/// `−ln B(ν/2, ½) − ½ ln ν`.
pub(crate) fn log_norm(eta: f64) -> f64 {
    if eta < ETA_SERIES {
        // −ln B(ν/2, ½) = ln Γ((ν+1)/2) − ln Γ(ν/2) − ½ ln π
        horner(&C_SERIES, eta) - HALF_LN_TAU
    } else {
        let nu = 1.0 / eta;
        -ln_beta(0.5 * nu, 0.5) - 0.5 * libm::log(nu)
    }
}

/// `ln(1+u)/u`, 1 at `u = 0`.
fn ln1p_over(u: f64) -> f64 {
    if u == 0.0 { 1.0 } else { libm::log1p(u) / u }
}

/// `log f(z)` at scale 1 (subtract `ln scale` for another).
pub(crate) fn log_density(z: f64, eta: f64) -> f64 {
    let q = z * z;
    log_norm(eta) - 0.5 * (1.0 + eta) * q * ln1p_over(eta * q)
}

/// `A(η)`, the part of the `η` score that does not depend on `z`.
pub(crate) fn score_const(eta: f64) -> f64 {
    if eta < ETA_SERIES {
        horner(&A_SERIES, eta)
    } else {
        let nu = 1.0 / eta;
        let d = 0.5 * (digamma(0.5 * (nu + 1.0)) - digamma(0.5 * nu)) - 0.5 * eta;
        -nu * nu * d
    }
}

/// The scores `(∂/∂loc · scale, ∂/∂ln scale, ∂/∂η)` of one row's `z`.
pub(crate) fn scores(z: f64, eta: f64) -> [f64; 3] {
    let q = z * z;
    let u = eta * q;
    let r = 1.0 / (1.0 + u);
    let s_loc = (1.0 + eta) * z * r;
    let s_ln = (1.0 + eta) * q * r - 1.0;
    // ½ z⁴ G(u) = ½ ν² (ln(1+u) − u r): the series near 0, and the closed
    // form divided by η twice (z⁴ overflows before ν² does not).
    let quart = if u < U_SERIES {
        let mut g = 0.0;
        for k in (2..16).rev() {
            let kf = k as f64;
            let c = (kf - 1.0) / kf;
            g = g * u + if k % 2 == 0 { c } else { -c };
        }
        0.5 * q * q * g
    } else {
        0.5 * (libm::log1p(u) - u * r) / eta / eta
    };
    let s_eta = score_const(eta) + quart - 0.5 * q * r;
    [s_loc, s_ln, s_eta]
}

/// `A′(η)`, the derivative of [`score_const`]: with `Δψ = ψ((ν+1)/2) −
/// ψ(ν/2)`, `ν³ Δψ + ¼ ν⁴ (ψ′((ν+1)/2) − ψ′(ν/2)) − ½ ν²`, or the
/// derivative of `A`'s series below the switch.
pub(crate) fn score_const_deriv(eta: f64) -> f64 {
    if eta < ETA_SERIES {
        let mut acc = 0.0;
        for (k, c) in A_SERIES.iter().enumerate().skip(1).rev() {
            acc = acc * eta + k as f64 * c;
        }
        acc
    } else {
        let nu = 1.0 / eta;
        let nu2 = nu * nu;
        let dpsi = digamma(0.5 * (nu + 1.0)) - digamma(0.5 * nu);
        let dtri = trigamma(0.5 * (nu + 1.0)) - trigamma(0.5 * nu);
        nu2 * (nu * dpsi + 0.25 * nu2 * dtri - 0.5)
    }
}

/// The observed information `−∂² log f` of one row's `z` per unit weight at
/// scale 1, packed `[loc·loc, loc·ln s, loc·η, ln s·ln s, ln s·η, η·η]`
/// (the `loc` entries times `scale²`, `scale`, `scale` at another scale).
/// With `q = z²`, `u = η q`, `r = 1/(1+u)`:
///
/// ```text
/// J_loc,loc = (1+η) r² (1 − u)        J_loc,ln s = 2 (1+η) z r²
/// J_loc,η   = z r² (q − 1)            J_ln s,ln s = 2 (1+η) q r²
/// J_ln s,η  = q r² (q − 1)
/// J_η,η     = −A′(η) − q³ H(u) − ½ q² r²,   H(u) = (½ u² r² + u r − ln(1+u))/u³
/// ```
///
/// whose expectation is [`expected_info`]. `q³ H(u)` below `u = 1/100` is
/// read from `H`'s series `Σ_{k≥3} (−1)ᵏ (k−1)(k−2)/(2k) u^(k−3)`.
pub(crate) fn observed_info(z: f64, eta: f64) -> [f64; 6] {
    let q = z * z;
    let u = eta * q;
    let r = 1.0 / (1.0 + u);
    let r2 = r * r;
    let e1 = 1.0 + eta;
    let cube = if u < U_SERIES {
        let mut h = 0.0;
        for k in (3..19).rev() {
            let kf = k as f64;
            let c = (kf - 1.0) * (kf - 2.0) / (2.0 * kf);
            h = h * u + if k % 2 == 0 { c } else { -c };
        }
        q * q * q * h
    } else {
        (0.5 * u * u * r2 + u * r - libm::log1p(u)) / (eta * eta * eta)
    };
    [
        e1 * r2 * (1.0 - u),
        2.0 * e1 * z * r2,
        z * r2 * (q - 1.0),
        2.0 * e1 * q * r2,
        q * r2 * (q - 1.0),
        -score_const_deriv(eta) - cube - 0.5 * q * q * r2,
    ]
}

/// The expected information per unit weight at scale 1, `[I_loc,loc,
/// I_lnσ,lnσ, I_lnσ,η, I_η,η]` (`I_loc,loc` divides by `scale²` at
/// another scale; loc's cross terms are 0).
pub(crate) fn expected_info(eta: f64) -> [f64; 4] {
    let a = 1.0 + 3.0 * eta;
    let iee = if eta < ETA_SERIES {
        horner(&IEE_SERIES, eta)
    } else {
        let nu = 1.0 / eta;
        let nu2 = nu * nu;
        nu2 * nu2
            * (0.25 * (trigamma(0.5 * nu) - trigamma(0.5 * (nu + 1.0)))
                - (nu + 5.0) / (2.0 * nu * (nu + 1.0) * (nu + 3.0)))
    };
    [(1.0 + eta) / a, 2.0 / a, 2.0 / ((1.0 + eta) * a), iee]
}

/// Fisher's expansion of the quantile about the normal's `x`.
fn expand(x: f64, eta: f64) -> f64 {
    let x2 = x * x;
    let g1 = (x2 + 1.0) * x / 4.0;
    let g2 = ((5.0 * x2 + 16.0) * x2 + 3.0) * x / 96.0;
    let g3 = (((3.0 * x2 + 19.0) * x2 + 17.0) * x2 - 15.0) * x / 384.0;
    x + eta * (g1 + eta * (g2 + eta * g3))
}

/// `P(Z ≤ z)`.
pub(crate) fn cdf(z: f64, eta: f64) -> f64 {
    if z.is_nan() {
        return f64::NAN;
    }
    if eta == 0.0 {
        return norm_cdf(z);
    }
    if eta < ETA_EXPAND {
        if !z.is_finite() {
            return if z > 0.0 { 1.0 } else { 0.0 };
        }
        // the x with expand(x) = z, by Newton from z
        let mut x = z;
        for _ in 0..4 {
            let x2 = x * x;
            let d1 = (3.0 * x2 + 1.0) / 4.0;
            let d2 = ((25.0 * x2 + 48.0) * x2 + 3.0) / 96.0;
            let d3 = (((21.0 * x2 + 95.0) * x2 + 51.0) * x2 - 15.0) / 384.0;
            let d = 1.0 + eta * (d1 + eta * (d2 + eta * d3));
            x -= (expand(x, eta) - z) / d;
        }
        return norm_cdf(x);
    }
    let u = eta * z * z;
    // P(Z ≤ −|z|) = ½ I_{1/(1+u)}(ν/2, ½) = ½ (1 − I_{u/(1+u)}(½, ν/2)),
    // with whichever argument is the smaller formed directly
    let tail = if u > 1.0 {
        0.5 * beta_inc(0.5 / eta, 0.5, 1.0 / (1.0 + u))
    } else {
        0.5 * beta_inc_c(0.5, 0.5 / eta, u / (1.0 + u))
    };
    if z < 0.0 { tail } else { 1.0 - tail }
}

/// The quantile `t` with `P(Z ≤ t) = p`, `p` in `(0, 1)`.
pub(crate) fn ppf(p: f64, eta: f64) -> f64 {
    if eta == 0.0 {
        return ndtri(p);
    }
    if eta < ETA_EXPAND {
        return expand(ndtri(p), eta);
    }
    if p.is_nan() || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if p == 0.5 {
        return 0.0;
    }
    let lo = p.min(1.0 - p);
    let nu = 1.0 / eta;
    // 2 lo = I_x(ν/2, ½), x = ν/(ν+t²); read whichever of x and 1 − x is
    // the small one directly.
    let t2 = if lo < 0.25 {
        let x = beta_inc_inv(0.5 * nu, 0.5, 2.0 * lo);
        nu * ((1.0 - x) / x)
    } else {
        let v = beta_inc_inv(0.5, 0.5 * nu, 1.0 - 2.0 * lo);
        nu * (v / (1.0 - v))
    };
    let t = t2.sqrt();
    if p < 0.5 { -t } else { t }
}

/// The lower tail's expected shortfall `E[Z | Z ≤ t_α]`; `−∞` for `ν ≤ 1`,
/// where the mean does not exist.
pub(crate) fn lower_es(alpha: f64, eta: f64) -> f64 {
    if eta >= 1.0 {
        return f64::NEG_INFINITY;
    }
    let t = ppf(alpha, eta);
    if eta == 0.0 {
        return -libm::exp(-0.5 * t * t - HALF_LN_TAU) / alpha;
    }
    -(1.0 + eta * t * t) / (1.0 - eta) * libm::exp(log_density(t, eta)) / alpha
}

#[cfg(test)]
mod tests;
