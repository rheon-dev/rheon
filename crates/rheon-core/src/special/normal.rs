//! The standard normal distribution: its CDF `Φ(z) = ½ erfc(−z/√2)` and
//! two inverses. [`norm_ppf`] is Acklam's rational approximation with
//! basic arithmetic alone (its tails' logarithm [`ln_basic`]), the same
//! bits on every platform with no math library at all, good to `1.2e-9`;
//! [`ndtri`] takes one Halley step from it on [`norm_cdf`] (the `libm`
//! crate's `erfc` and `exp`), which brings it to the last few bits.
//!
//! ```text
//! e = Φ(x) − p,   u = e √(2π) e^{x²/2},   x ← x − u / (1 + x u / 2)
//! ```
//!
//! taken in the lower tail (`p ≤ ½`, negated for the upper one, `1 − p`
//! being exact there), where `Φ(x)` has its digits.

/// `ln x` for a finite `x > 0` in basic arithmetic, the same bits on every
/// platform: `x = 2^k · f` with `f` in `[√½, √2)` read from the bits (a
/// subnormal scaled up by `2^54` first), and `ln f = 2·atanh(s)`, `s = (f −
/// 1)/(f + 1)`, `|s| ≤ 0.1716`, by its odd series to the term below the
/// last bit; `k·ln 2` added in two parts (Cody and Waite), as `conformal::exp_neg`
/// subtracts it. Within a few ulps of the true value; NaN for a negative
/// or NaN `x`, `−∞` at 0 and `∞` at `∞`.
pub fn ln_basic(x: f64) -> f64 {
    if x.is_nan() || x < 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return f64::NEG_INFINITY;
    }
    if x.is_infinite() {
        return f64::INFINITY;
    }
    const LN2_HI: f64 = 0.693_147_180_369_123_8;
    const LN2_LO: f64 = 1.908_214_929_270_587_7e-10;
    // A subnormal has no implicit bit: scale it into the normals exactly.
    let (x, k0) = if x < f64::MIN_POSITIVE {
        (x * f64::from_bits((1023 + 54) << 52), -54i64)
    } else {
        (x, 0)
    };
    let bits = x.to_bits();
    let mut k = ((bits >> 52) & 0x7ff) as i64 - 1023 + k0;
    // The mantissa with the exponent of 1: `f` in [1, 2), exactly.
    let mut f = f64::from_bits((bits & ((1u64 << 52) - 1)) | (1023u64 << 52));
    if f > std::f64::consts::SQRT_2 {
        f *= 0.5;
        k += 1;
    }
    let s = (f - 1.0) / (f + 1.0);
    let s2 = s * s;
    // 2·(s + s³/3 + s⁵/5 + …): s² ≤ 0.0295, so the 15th term is below
    // 2^-53 of the first.
    let mut term = s;
    let mut sum = s;
    for i in 1..=15 {
        term *= s2;
        sum += term / f64::from(2 * i + 1);
    }
    let kf = k as f64;
    (kf * LN2_HI + 2.0 * sum) + kf * LN2_LO
}

/// Inverse of the standard normal CDF, `Φ⁻¹(p)` for `p` in `(0, 1)`.
///
/// Acklam's rational approximation (relative error below `1.2e-9` over the
/// whole range), evaluated in a fixed order, its tails' logarithm in basic
/// arithmetic ([`ln_basic`]), so the same input gives the same bits on
/// every platform, and the multiplier a tracker starts at is the same
/// wherever its state is loaded. Good to every purpose here -- it seeds a
/// multiplier that is then tracked -- and a test pins it against known
/// quantiles.
pub fn norm_ppf(p: f64) -> f64 {
    const A: [f64; 6] = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.38357751867269e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    const D: [f64; 4] = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];
    const P_LOW: f64 = 0.02425;
    if !(p > 0.0 && p < 1.0) {
        return f64::NAN;
    }
    let tail = |q: f64| {
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    };
    if p < P_LOW {
        tail((-2.0 * ln_basic(p)).sqrt())
    } else if p <= 1.0 - P_LOW {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        -tail((-2.0 * ln_basic(1.0 - p)).sqrt())
    }
}

/// The standard normal CDF, `Φ(z) = ½ erfc(−z/√2)`, to full relative
/// accuracy in both tails.
pub fn norm_cdf(z: f64) -> f64 {
    0.5 * libm::erfc(-z * std::f64::consts::FRAC_1_SQRT_2)
}

/// `Φ⁻¹(p)`, `p` in `[0, 1]`, to the last few bits: [`norm_ppf`] refined
/// by a Halley step on [`norm_cdf`]. `−∞` at 0, `+∞` at 1, NaN outside.
pub fn ndtri(p: f64) -> f64 {
    if p.is_nan() || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if p == 0.0 {
        return f64::NEG_INFINITY;
    }
    if p == 1.0 {
        return f64::INFINITY;
    }
    if p > 0.5 {
        // 1 − p is exact for p ≥ ½.
        return -ndtri_lower(1.0 - p);
    }
    ndtri_lower(p)
}

/// [`ndtri`] for `0 < p ≤ ½`.
fn ndtri_lower(p: f64) -> f64 {
    // √(2π)
    const SQRT_2PI: f64 = 2.506_628_274_631_000_5;
    let mut x = norm_ppf(p);
    // Two steps: Acklam's 1.2e-9 needs one Halley step for the last bits;
    // the second removes the rounding left by the first.
    for _ in 0..2 {
        let e = norm_cdf(x) - p;
        let u = e * SQRT_2PI * libm::exp(0.5 * x * x);
        x -= u / (1.0 + 0.5 * x * u);
    }
    x
}
