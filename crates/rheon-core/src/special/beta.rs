//! `ln B(a, b)` and the regularized incomplete beta function
//! `I_x(a, b) = B(x; a, b)/B(a, b)` with its inverse in `x`.
//!
//! ```text
//! I_x(a, b) = xᵃ (1−x)ᵇ / (a B(a, b)) · 1/(1+ d₁/(1+ d₂/(1+ …)))     x < (a+1)/(a+b+2)
//! d₂ₘ₊₁ = −(a+m)(a+b+m) x / ((a+2m)(a+2m+1)),   d₂ₘ = m(b−m) x / ((a+2m−1)(a+2m))
//! I_x(a, b) = 1 − I_{1−x}(b, a)                                       otherwise
//! ```
//!
//! the continued fraction by the modified Lentz method, which converges in
//! `O(√max(a, b))` terms on the side of its switch point `(a+1)/(a+b+2)`
//! it is used on, and slowly, with its rounding growing over the terms, on
//! the other. Both `I` and `1 − I` are returned, the other as `1 −` the one
//! the fraction forms; near the switch point both are of order ½.
//!
//! `ln B` is formed so that the terms of size `a ln a` that cancel are
//! never formed: from 20 up in both arguments by Stirling's series as
//! `(a−½) ln(a/(a+b)) + (b−½) ln(b/(a+b)) − ½ ln(a+b) + ln √(2π) + S(a) +
//! S(b) − S(a+b)`, with `ln(a/(a+b)) = −ln(1 + b/a)`; with one argument
//! below 20 as `ln Γ(b) − (a−½) ln(1 + b/a) − b ln(a+b) + b + S(a) −
//! S(a+b)`; with both below 20 as the log of the ratio of the `libm`
//! crate's `Γ` values.
//!
//! The inverse starts from the normal-approximation guess (both arguments
//! at least 1) or the two tails' power laws, and takes Halley steps on `I`,
//! each with the density `x^(a−1) (1−x)^(b−1)/B(a, b)`.

use super::LN_SQRT_2PI;
use super::gamma::{bisect, ln_gamma, ln_gamma_1p, stirling_tail};

#[inline]
fn ln(x: f64) -> f64 {
    libm::log(x)
}
#[inline]
fn exp(x: f64) -> f64 {
    libm::exp(x)
}

/// `ln Γ(x)`, `x > 0`, to full relative accuracy near its zeros at 1 and 2,
/// and below 20 from the `libm` crate's `Γ` rather than [`ln_gamma`]'s
/// recurrence, which sums up to twenty logarithms.
fn ln_gamma_acc(x: f64) -> f64 {
    if x < 1.0 {
        ln_gamma_1p(x) - ln(x)
    } else if x <= 2.0 {
        ln_gamma_1p(x - 1.0)
    } else if x < 20.0 {
        ln(libm::tgamma(x))
    } else {
        ln_gamma(x)
    }
}

/// `ln B(a, b) = ln Γ(a) + ln Γ(b) − ln Γ(a + b)` for `a, b > 0`; NaN
/// outside.
pub fn ln_beta(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() || a <= 0.0 || b <= 0.0 {
        return f64::NAN;
    }
    let (lo, hi) = if a < b { (a, b) } else { (b, a) };
    if hi == f64::INFINITY {
        return f64::NEG_INFINITY;
    }
    let s = hi + lo;
    if lo >= 20.0 {
        let corr = stirling_tail(lo) + stirling_tail(hi) - stirling_tail(s);
        -(lo - 0.5) * libm::log1p(hi / lo) - (hi - 0.5) * libm::log1p(lo / hi) - 0.5 * ln(s)
            + LN_SQRT_2PI
            + corr
    } else if hi >= 20.0 {
        let corr = stirling_tail(hi) - stirling_tail(s);
        ln_gamma_acc(lo) + (-(hi - 0.5) * libm::log1p(lo / hi) - lo * ln(s) + lo + corr)
    } else {
        // Both below 20: the `libm` crate's Γ, within a few ulps, whose
        // ratio is far from overflow here; `ln Γ`'s recurrence would sum
        // twenty logarithms to each term.
        let r = libm::tgamma(lo) / libm::tgamma(s) * libm::tgamma(hi);
        if r.is_normal() {
            ln(r)
        } else {
            ln_gamma_acc(lo) + ln_gamma_acc(hi) - ln_gamma_acc(s)
        }
    }
}

const EPS: f64 = f64::EPSILON;
const TINY: f64 = 1e-300;
const MAX_TERMS: usize = 100_000;

/// `ln(xᵃ yᵇ / B(a, b))`, `y = 1 − x` given exactly by the caller.
fn ln_front(a: f64, b: f64, x: f64, y: f64) -> f64 {
    let lx = if x > 0.5 { libm::log1p(-y) } else { ln(x) };
    let ly = if y > 0.5 { libm::log1p(-x) } else { ln(y) };
    a * lx + b * ly - ln_beta(a, b)
}

/// `xᵃ yᵇ / B(a, b)`. Where the product is a normal number, as the
/// product: each power within an ulp or two (`(1 − x)ᵇ` as
/// `exp(b ln(1 − x))` below `x = ½`, where `1 − x` is not exact). The log
/// form's sum carries an absolute rounding of `a |ln x| · ε`, which `exp`
/// turns into a relative error of that many ulps.
fn front(a: f64, b: f64, x: f64, y: f64) -> f64 {
    let px = libm::pow(x, a);
    let py = if x >= 0.5 {
        libm::pow(y, b)
    } else {
        exp(b * libm::log1p(-x))
    };
    // 1/B from the `libm` crate's Γ for small arguments, else from
    // [`ln_beta`], which is accurate to an ε or two there while the ratio
    // of two large Γ values is not.
    let mut inv_b = if a + b <= 20.0 {
        libm::tgamma(a + b) / libm::tgamma(a) / libm::tgamma(b)
    } else {
        0.0
    };
    if !inv_b.is_normal() {
        inv_b = exp(-ln_beta(a, b));
    }
    let v = px * py * inv_b;
    if px.is_normal() && py.is_normal() && v.is_normal() {
        return v;
    }
    exp(ln_front(a, b, x, y))
}

/// The continued fraction of `I_x(a, b)`, used where
/// `x < (a+1)/(a+b+2)`.
fn beta_cf(a: f64, b: f64, x: f64) -> f64 {
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < TINY {
        d = TINY;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..MAX_TERMS {
        let m = m as f64;
        let m2 = 2.0 * m;
        let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < TINY {
            d = TINY;
        }
        c = 1.0 + aa / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        h *= d * c;
        let aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < TINY {
            d = TINY;
        }
        c = 1.0 + aa / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() <= EPS * 0.5 {
            break;
        }
    }
    h
}

/// `(I_x(a, b), 1 − I_x(a, b))`, with `y = 1 − x`.
fn beta_pair(a: f64, b: f64, x: f64, y: f64) -> (f64, f64) {
    if a.is_nan() || b.is_nan() || x.is_nan() || a < 0.0 || b < 0.0 || !(0.0..=1.0).contains(&x) {
        return (f64::NAN, f64::NAN);
    }
    if a == 0.0 && b == 0.0 {
        return (f64::NAN, f64::NAN);
    }
    if a == f64::INFINITY || b == f64::INFINITY {
        return (f64::NAN, f64::NAN);
    }
    if a == 0.0 {
        // All the mass at 0.
        return (1.0, 0.0);
    }
    if b == 0.0 {
        // All the mass at 1.
        return if x < 1.0 { (0.0, 1.0) } else { (1.0, 0.0) };
    }
    if x == 0.0 {
        return (0.0, 1.0);
    }
    if y == 0.0 {
        return (1.0, 0.0);
    }
    let front = front(a, b, x, y);
    if x * (a + b + 2.0) < a + 1.0 {
        let i = front * beta_cf(a, b, x) / a;
        (i, 1.0 - i)
    } else {
        let ic = front * beta_cf(b, a, y) / b;
        (1.0 - ic, ic)
    }
}

/// The regularized incomplete beta function
/// `I_x(a, b) = (1/B(a, b)) ∫₀ˣ t^(a−1) (1−t)^(b−1) dt`, `a, b ≥ 0` (not
/// both 0), `x` in `[0, 1]`: the beta distribution's CDF. `a = 0` puts all
/// the mass at 0 and `b = 0` at 1; NaN outside the domain.
pub fn beta_inc(a: f64, b: f64, x: f64) -> f64 {
    beta_pair(a, b, x, 1.0 - x).0
}

/// `1 − I_x(a, b)`, to full relative accuracy where it is small. For an
/// `x` near 1, `I_{1−x}(b, a)` read with `1 − x` exact is the better call.
pub fn beta_inc_c(a: f64, b: f64, x: f64) -> f64 {
    beta_pair(a, b, x, 1.0 - x).1
}

/// The inverse of [`beta_inc`] in `x`: the `x` in `[0, 1]` with
/// `I_x(a, b) = p`, `a, b > 0`, the beta distribution's quantile. `0` at
/// `p = 0`, `1` at `p = 1`, NaN outside the domain. The upper tail's
/// quantile, with its distance from 1 to full accuracy, is
/// `1 − x = beta_inc_inv(b, a, 1 − p)`.
pub fn beta_inc_inv(a: f64, b: f64, p: f64) -> f64 {
    if a.is_nan() || b.is_nan() || p.is_nan() || a <= 0.0 || b <= 0.0 {
        return f64::NAN;
    }
    if !(0.0..=1.0).contains(&p) || a == f64::INFINITY || b == f64::INFINITY {
        return f64::NAN;
    }
    if p == 0.0 {
        return 0.0;
    }
    if p == 1.0 {
        return 1.0;
    }
    let q = 1.0 - p;
    let lnb = ln_beta(a, b);
    let mut x = if a >= 1.0 && b >= 1.0 {
        let pp = p.min(q);
        let t = (-2.0 * ln(pp)).sqrt();
        let mut z = (2.30753 + t * 0.27061) / (1.0 + t * (0.99229 + t * 0.04481)) - t;
        if p < 0.5 {
            z = -z;
        }
        let al = (z * z - 3.0) / 6.0;
        let h = 2.0 / (1.0 / (2.0 * a - 1.0) + 1.0 / (2.0 * b - 1.0));
        let w = z * (al + h).sqrt() / h
            - (1.0 / (2.0 * b - 1.0) - 1.0 / (2.0 * a - 1.0)) * (al + 5.0 / 6.0 - 2.0 / (3.0 * h));
        a / (a + b * exp(2.0 * w))
    } else {
        // I ≈ xᵃ/(a B) near 0 and 1 − (1−x)ᵇ/(b B) near 1; `t` and `u`
        // are the two power laws' masses at the mean `a/(a+b)`.
        let lna = ln(a / (a + b));
        let lnb_ = ln(b / (a + b));
        let t = exp(a * lna) / a;
        let u = exp(b * lnb_) / b;
        let w = t + u;
        if p < t / w {
            libm::pow(a * w * p, 1.0 / a)
        } else {
            1.0 - libm::pow(b * w * q, 1.0 / b)
        }
    };
    if x.is_nan() || x <= 0.0 {
        x = f64::MIN_POSITIVE;
    }
    if x >= 1.0 {
        x = 1.0 - EPS / 2.0;
    }
    // Halley steps, safeguarded as the incomplete gamma's inverse is: a
    // bracket on the root, and a bisection for a step that leaves it.
    let (mut lo, mut hi) = (0.0f64, 1.0f64);
    for _ in 0..300 {
        let y = 1.0 - x;
        let (i, ic) = beta_pair(a, b, x, y);
        let err = if p < 0.5 { i - p } else { q - ic };
        if err == 0.0 {
            break;
        }
        if err > 0.0 {
            hi = x;
        } else {
            lo = x;
        }
        let dens = exp((a - 1.0) * ln(x) + (b - 1.0) * libm::log1p(-x) - lnb);
        let mut next = f64::NAN;
        if dens.is_normal() {
            next = if p < 0.1 {
                // Newton on ln I in the lower tail, ln(1 − I) in the upper
                x - (ln(i) - ln(p)) * i / dens
            } else if q < 0.1 {
                x + (ln(ic) - ln(q)) * ic / dens
            } else {
                let u = err / dens;
                // Halley: f''/f' = (a − 1)/x − (b − 1)/(1 − x)
                x - u / (1.0 - 0.5 * (u * ((a - 1.0) / x - (b - 1.0) / y)).min(1.0))
            };
        }
        if (next - x).abs() <= 2.0 * EPS * x {
            // converged: the step is below the rounding of `x`
            x = next;
            break;
        }
        if !(next > lo && next < hi) {
            next = bisect(lo, hi, x);
        }
        x = next;
        if hi - lo <= 2.0 * EPS * lo {
            break;
        }
    }
    x
}
