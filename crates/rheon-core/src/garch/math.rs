//! `ln`, `exp` and `ln_1p` for `garch`, from the `libm` crate: Rust
//! compiled from source in this build, so the same bits on every
//! platform, where the platforms' own libraries (Apple's, glibc, Windows'
//! `ucrtbase.dll`) differ in their last bits.
//!
//! Elsewhere such a difference stays a last-bit difference; here it does
//! not. The start's optimizer compares likelihoods in its line search and
//! step sizes in its stop, and the start's doubling reads whether the fit
//! converged off its bounds. Where the likelihood is flat along a ridge (a
//! short start with no GARCH effect, or a t's shape the rows barely tell),
//! those comparisons sit at the rounding and the last bits choose the
//! path: a 50-row skewed-t start's `ω` was 28.3 on macOS, 29.6 on x86-64
//! Linux and 20.5 on Windows (2026-11-22), and with the `libm` crate's
//! functions in place of Apple's, 3 to 6 of 50 series of 10,000 rows with
//! t innovations, started on 2,000, took another path. No ordering of the
//! comparisons fixes that. The step each row takes after the start then
//! carries a difference through `H⁻¹`, nearly singular in a flat
//! direction, and the projection's iterations in its metric: with only the
//! start's fit made the same everywhere, a skewed t's variance still moved
//! by 5e-12 within 20 rows of its start, and an EGARCH's coefficients by
//! 4e-11 over the golden stream. With the same bits in, every platform
//! takes the same path.
//!
//! The cost, on arm64: the `libm` crate's `ln` is 1.4 times Apple's and its
//! `exp` 1.8 times (2.4 and 3.2 ns a call). A row's step calls neither for
//! `garch` and `gjr` under the normal, nor for `figarch`, and too few for
//! the t to show; the skewed t's step is 4% to 13% slower and `egarch`'s
//! 14%, whose guard reads `ln` on each of its 64 bins at every point it
//! tries. The start's fit, one-off per series, calls them on every row of
//! every likelihood it evaluates.

/// `ln x`.
#[inline]
pub(super) fn ln(x: f64) -> f64 {
    libm::log(x)
}

/// `eˣ`.
#[inline]
pub(super) fn exp(x: f64) -> f64 {
    libm::exp(x)
}

/// `ln(1 + x)`.
#[inline]
pub(super) fn ln_1p(x: f64) -> f64 {
    libm::log1p(x)
}
