//! `ln Γ` by the Lanczos approximation, the one `bocpd`'s Student-t
//! predictive density has always used. It reads the platform's `ln` and
//! `sin`, not the `libm` crate's: switching would move `bocpd`'s outputs in
//! their last bits on some platform, and its saved states continue to the
//! bit on the platform that wrote them. New callers use
//! [`super::ln_gamma`], which is the `libm` crate's on every platform.

/// `ln Γ(x)`, by the Lanczos approximation: the Student-t density needs it
/// and `f64::ln_gamma` is unstable.
pub(crate) fn ln_gamma_lanczos(x: f64) -> f64 {
    // Lanczos g = 7, n = 9; accurate to ~1e-13 over the range used here.
    #[allow(clippy::excessive_precision, clippy::inconsistent_digit_grouping)]
    const C: [f64; 9] = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];
    if x < 0.5 {
        // Reflection, for completeness; the callers stay above 0.5.
        return (std::f64::consts::PI / (std::f64::consts::PI * x).sin()).ln()
            - ln_gamma_lanczos(1.0 - x);
    }
    let x = x - 1.0;
    let mut a = C[0];
    let t = x + 7.5;
    for (i, c) in C.iter().enumerate().skip(1) {
        a += c / (x + i as f64);
    }
    0.5 * (std::f64::consts::TAU).ln() + (x + 0.5) * t.ln() - t + a.ln()
}
