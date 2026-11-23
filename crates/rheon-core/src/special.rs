//! Special functions, on the `libm` crate: Rust compiled from source in
//! this build, so each function here is the same bits on every platform,
//! where the platforms' own math libraries (Apple's, glibc, Windows'
//! `ucrtbase.dll`) differ in their last bits. A model that compares or
//! accumulates these values in its state then continues to the bit across
//! operating systems.
//!
//! - [`ln_gamma`], [`ln_gamma_1p`], [`digamma`], [`trigamma`]: the log
//!   gamma function and its first two derivatives;
//! - [`gamma_p`], [`gamma_q`] and their inverses [`gamma_p_inv`],
//!   [`gamma_q_inv`]: the regularized incomplete gamma functions, the gamma
//!   distribution's CDF and quantile;
//! - [`ln_beta`], [`beta_inc`], [`beta_inc_c`] and [`beta_inc_inv`]: the
//!   regularized incomplete beta function, the beta distribution's CDF and
//!   quantile, from which Student's t's are read;
//! - [`norm_cdf`], [`ndtri`], [`norm_ppf`]: the standard normal CDF and its
//!   inverse; [`norm_ppf`] and its logarithm [`ln_basic`] are basic
//!   arithmetic alone, with no math library.
//!
//! Each function's accuracy is measured against `scipy.special` and stated,
//! in units in the last place, beside that test.

mod beta;
mod gamma;
mod lanczos;
mod normal;
#[cfg(test)]
mod tests;

pub use beta::{beta_inc, beta_inc_c, beta_inc_inv, ln_beta};
pub use gamma::{
    digamma, gamma_p, gamma_p_inv, gamma_q, gamma_q_inv, ln_gamma, ln_gamma_1p, trigamma,
};
pub(crate) use lanczos::ln_gamma_lanczos;
pub use normal::{ln_basic, ndtri, norm_cdf, norm_ppf};

/// `ln √(2π)`.
const LN_SQRT_2PI: f64 = 0.918_938_533_204_672_7;
