//! Influence weight functions for `huber` ([`crate::Robust`]): Huber's
//! monotone cut and four redescending M-estimators matching
//! `statsmodels.robust.norms`.
//!
//! Each weight is a function of the non-negative scaled residual
//! `u = |r| / s` and the tuning parameter `delta > 0`:
//!
//! | norm | default `delta` | `w(u, delta)` |
//! |---|---|---|
//! | `Huber` | `1.345` | `1` for `u <= c`, else `c / u` |
//! | `Tukey` | `4.685` | `(1 - (u / c)^2)^2` for `u < c`, else `0` |
//! | `Andrews` | `1.339` | `sin(u / c) / (u / c)` for `u <= pi * c` (`1` at `0`), else `0` |
//! | `Hampel` | `2.0` | `1` on `[0, a]`, `a / u` on `(a, b]`, `a(c - u) / (u(c - b))` on `(b, c]`, else `0`, with `(a, b, c) = (d, 2d, 4d)` |
//! | `Ramsay` | `0.3` | `exp(-c * u)` |
//!
//! Transcendentals (`sin`, `exp`) come from `libm` so every platform writes
//! the same bits.

use serde::{Deserialize, Serialize};

/// The M-estimator weight function used by `huber` ([`crate::Robust`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HuberNorm {
    /// Huber's monotone cut (`1` inside `delta`, `delta / u` beyond).
    #[default]
    Huber,
    /// Tukey's biweight: `(1 - (u / c)^2)^2` for `u < c`, `0` beyond.
    Tukey,
    /// Andrews' sine wave: `sin(u / a) / (u / a)` for `u <= pi * a`, `0` beyond.
    Andrews,
    /// Hampel's three-part redescender with breakpoints `(d, 2d, 4d)`.
    Hampel,
    /// Ramsay's `E_a` exponential weight: `exp(-a * u)`.
    Ramsay,
}

impl HuberNorm {
    /// The word a spec spells this norm with.
    pub fn word(self) -> &'static str {
        match self {
            Self::Huber => "huber",
            Self::Tukey => "tukey",
            Self::Andrews => "andrews",
            Self::Hampel => "hampel",
            Self::Ramsay => "ramsay",
        }
    }

    /// Parse a spec word into a [`HuberNorm`].
    pub fn from_word(s: &str) -> Option<Self> {
        match s {
            "huber" => Some(Self::Huber),
            "tukey" => Some(Self::Tukey),
            "andrews" => Some(Self::Andrews),
            "hampel" => Some(Self::Hampel),
            "ramsay" => Some(Self::Ramsay),
            _ => None,
        }
    }

    /// The standard 95%-efficiency Gaussian default for `huber_delta` under
    /// each norm (matching `statsmodels.robust.norms`).
    pub fn default_delta(self) -> f64 {
        match self {
            Self::Huber => 1.345,
            Self::Tukey => 4.685,
            Self::Andrews => 1.339,
            Self::Hampel => 2.0,
            Self::Ramsay => 0.3,
        }
    }

    /// Whether this is the default monotone Huber norm.
    pub fn is_huber(self) -> bool {
        matches!(self, Self::Huber)
    }

    /// IRLS weight `w(u, delta)` for non-negative scaled residual `u = |r| / s`
    /// and positive tuning parameter `delta`.
    ///
    /// For `Huber`, the caller uses the exact existing `cut = delta * s` path
    /// so no bit of the default Huber model moves; this method gives the same
    /// mathematical weight when called directly.
    #[inline]
    pub fn weight(self, u: f64, delta: f64) -> f64 {
        if !u.is_finite() {
            return match self {
                Self::Huber if delta.is_infinite() => 1.0,
                Self::Tukey | Self::Andrews | Self::Hampel | Self::Ramsay if u.is_infinite() => {
                    if delta.is_infinite() && !matches!(self, Self::Ramsay) {
                        1.0
                    } else {
                        0.0
                    }
                }
                _ => f64::NAN,
            };
        }
        let u = u.abs();
        match self {
            Self::Huber => {
                if u <= delta {
                    1.0
                } else {
                    delta / u
                }
            }
            Self::Tukey => {
                if delta.is_infinite() {
                    return 1.0;
                }
                if u < delta {
                    let t = u / delta;
                    let d = 1.0 - t * t;
                    d * d
                } else {
                    0.0
                }
            }
            Self::Andrews => {
                if delta.is_infinite() {
                    return 1.0;
                }
                let z = u / delta;
                if z == 0.0 {
                    1.0
                } else if z <= std::f64::consts::PI {
                    libm::sin(z) / z
                } else {
                    0.0
                }
            }
            Self::Hampel => {
                if delta.is_infinite() {
                    return 1.0;
                }
                let a = delta;
                let b = 2.0 * delta;
                let c = 4.0 * delta;
                if u <= a {
                    1.0
                } else if u <= b {
                    a / u
                } else if u < c {
                    a * (c - u) / (u * (c - b))
                } else {
                    0.0
                }
            }
            Self::Ramsay => {
                if u == 0.0 {
                    1.0
                } else {
                    libm::exp(-delta * u)
                }
            }
        }
    }

    /// Fill `out` with IRLS weights for raw absolute residuals `absr` at scale
    /// `scale` and tuning parameter `delta`. When `scale` is `None` (no scale
    /// yet), every weight is `1.0`.
    pub(crate) fn fill_weights(
        self,
        absr: &[f64],
        scale: Option<f64>,
        delta: f64,
        out: &mut [f64],
    ) {
        let Some(s) = scale.filter(|s| *s > 0.0 && s.is_finite()) else {
            out.fill(1.0);
            return;
        };
        if matches!(self, Self::Huber) {
            crate::huber_refit::huber_weights(absr, Some(s), delta, out);
            return;
        }
        for (o, &a) in out.iter_mut().zip(absr) {
            *o = self.weight(a / s, delta);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::HuberNorm;

    #[test]
    fn words_and_defaults_roundtrip() {
        let all = [
            HuberNorm::Huber,
            HuberNorm::Tukey,
            HuberNorm::Andrews,
            HuberNorm::Hampel,
            HuberNorm::Ramsay,
        ];
        for n in all {
            assert_eq!(HuberNorm::from_word(n.word()), Some(n));
            assert!(n.default_delta() > 0.0 && n.default_delta().is_finite());
        }
        assert_eq!(HuberNorm::from_word("unknown"), None);
    }

    #[test]
    fn exact_weight_values_and_cutoffs() {
        // Every norm gives 1.0 at u = 0.
        for n in [
            HuberNorm::Huber,
            HuberNorm::Tukey,
            HuberNorm::Andrews,
            HuberNorm::Hampel,
            HuberNorm::Ramsay,
        ] {
            assert_eq!(n.weight(0.0, n.default_delta()), 1.0);
        }
        // Tukey cuts off cleanly at c = 4.685.
        assert!((HuberNorm::Tukey.weight(2.3425, 4.685) - 0.5625).abs() < 1e-15);
        assert_eq!(HuberNorm::Tukey.weight(4.685, 4.685), 0.0);
        assert_eq!(HuberNorm::Tukey.weight(10.0, 4.685), 0.0);
        // Andrews cuts off at a * pi and equals sin(1) at u = a.
        let a = 1.339;
        assert!((HuberNorm::Andrews.weight(a, a) - libm::sin(1.0)).abs() < 1e-15);
        assert_eq!(
            HuberNorm::Andrews.weight(a * std::f64::consts::PI + 1e-9, a),
            0.0
        );
        // Hampel breakpoints (2, 4, 8): 1 on [0, 2], 2/u on (2, 4], 2*(8-u)/(u*4) on (4, 8), 0 at 8.
        assert_eq!(HuberNorm::Hampel.weight(2.0, 2.0), 1.0);
        assert!((HuberNorm::Hampel.weight(3.0, 2.0) - 2.0 / 3.0).abs() < 1e-15);
        assert!((HuberNorm::Hampel.weight(6.0, 2.0) - 1.0 / 6.0).abs() < 1e-15);
        assert_eq!(HuberNorm::Hampel.weight(8.0, 2.0), 0.0);
        // Ramsay: exp(-0.3 * u).
        assert!((HuberNorm::Ramsay.weight(2.0, 0.3) - libm::exp(-0.6)).abs() < 1e-15);
    }
}
