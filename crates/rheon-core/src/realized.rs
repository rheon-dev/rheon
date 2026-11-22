//! Realized-volatility estimators and the jump tests built on them, for a
//! stream of returns, one pass.
//!
//! **The definitions** (Barndorff-Nielsen and Shephard 2004, 2006; Lee and
//! Mykland 2008), for `n` returns `r_j` of one stretch of the clock, `a_j =
//! |r_j|`, and `μ_p = E|Z|^p = 2^(p/2) Γ((p + 1)/2) / Γ(1/2)` for a standard
//! normal `Z`:
//!
//! ```text
//! RV = Σ_j r_j²                                        realized variance
//! BV = μ_1^-2 Σ_{j≥2} a_j a_{j−1}                      bipower variation, μ_1^-2 = π/2
//! TV = μ_{2/3}^-3 Σ_{j≥3} (a_j a_{j−1} a_{j−2})^(2/3)  tripower variation
//! TQ = n μ_{4/3}^-3 Σ_{j≥3} (a_j a_{j−1} a_{j−2})^(4/3) tripower quarticity
//! Z  = √n (1 − BV/RV) / √(ϑ max(1, TQ/BV²))           the adjusted ratio test, ϑ = π²/4 + π − 5
//! L_i = r_i / σ_i,  σ_i² = mean of a_j a_{j−1} over the products before row i
//! ```
//!
//! `RV`, `BV` and `TV` estimate the integrated variance, `BV` and `TV` with
//! the jumps left out; `TQ` the integrated quarticity. `Z` is N(0, 1) without
//! jumps, and large and positive with them. `L_i` is Lee and Mykland's
//! statistic of the row's own return against the volatility of the returns
//! before it: under no jump `c L_i` is N(0, 1), `c = √(2/π)`. None carries a
//! small-sample factor such as `n/(n − 1)`: the papers' estimators have none.
//! A window of `W` returns holds the `W − 1` products of its consecutive
//! returns, and the `W − 2` triples, as the papers' sums over `W` returns do.

/// `μ_1^-2 = π/2`, the scale of bipower variation.
pub const BV_SCALE: f64 = std::f64::consts::FRAC_PI_2;

/// `μ_{2/3}^-3`, the scale of tripower variation, `μ_{2/3} = 2^(1/3) Γ(5/6) /
/// Γ(1/2)`; correctly rounded from 40 digits.
pub const TV_SCALE: f64 = 1.935_792_404_880_348_3;

/// `μ_{4/3}^-3`, the scale of tripower quarticity, `μ_{4/3} = 2^(2/3) Γ(7/6) /
/// Γ(1/2)`; correctly rounded from 40 digits.
pub const TQ_SCALE: f64 = 1.743_472_074_531_984_5;

/// `ϑ = π²/4 + π − 5`, the asymptotic variance of the ratio statistic
/// `√n (1 − BV/RV)` per unit of `TQ/BV²`; correctly rounded from 40 digits.
pub const BNS_THETA: f64 = 0.608_993_753_862_133;

#[cfg(test)]
mod paper;
#[cfg(test)]
mod tests;
