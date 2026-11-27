//! Gaussian per-feature-per-class observer and split criteria for
//! [`super::HoeffdingTreeClassifier`].
//!
//! Every transcendental (`erf`, `log`, `log2`, `exp`) goes through the pure-Rust
//! `libm` crate so that split decisions, child statistics, probabilities and
//! saved states are bit-identical across macOS, Linux and Windows.

use serde::{Deserialize, Serialize};

use super::SplitCriterion;

/// `ln(2 * pi) / 2`, matching `river`'s Gaussian log-density constant.
const HALF_LOG_TAU: f64 = 0.918_938_533_204_672_7;

/// `sqrt(2)`, matching `river`'s Gaussian CDF scaling.
const SQRT_2: f64 = std::f64::consts::SQRT_2;

/// Packed byte size of one [`GaussianObs`]: five `f64` values (`n`, `mean`,
/// `s`, `min`, `max`).
pub(super) const GAUSSIAN_OBS_BYTES: u64 = 40;

/// Per-feature, per-class Gaussian sufficient statistics at an active leaf.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub(super) struct GaussianObs {
    /// Total positive weight seen by this observer.
    pub n: f64,
    /// Weighted mean of the feature values.
    pub mean: f64,
    /// Weighted sum of squared deviations (Welford; variance `s / (n - 1)`).
    pub s: f64,
    /// Minimum feature value observed (0.0 when `n == 0.0`).
    pub min: f64,
    /// Maximum feature value observed (0.0 when `n == 0.0`).
    pub max: f64,
}

impl GaussianObs {
    /// Update with one observation `(x, w)` of positive normal weight `w`.
    #[inline]
    pub fn update(&mut self, x: f64, w: f64) {
        if self.n == 0.0 {
            self.min = x;
            self.max = x;
        } else {
            if x < self.min {
                self.min = x;
            }
            if x > self.max {
                self.max = x;
            }
        }
        let m_old = self.mean;
        let n = self.n + w;
        self.n = n;
        let ratio = w / n;
        let upd = m_old + ratio * (x - m_old);
        self.mean = if upd.is_finite() {
            upd
        } else {
            m_old * (1.0 - ratio) + x * ratio
        };
        let c = w * (x - m_old) * (x - self.mean);
        if c.is_finite() {
            if c > 0.0 {
                let ns = self.s + c;
                self.s = if ns.is_finite() { ns } else { f64::MAX };
            }
        } else if x != m_old {
            self.s = f64::MAX;
        }
    }

    /// Sample variance with `ddof = 1` on the weight (`s / (n - 1)` when `n > 1`).
    #[inline]
    pub fn var(&self) -> f64 {
        if self.n > 1.0 {
            let v = self.s / (self.n - 1.0);
            if v.is_finite() && v > 0.0 { v } else { 0.0 }
        } else {
            0.0
        }
    }

    /// Log Gaussian PDF at `x`, or `-inf` when variance is zero.
    #[inline]
    pub fn log_pdf(&self, x: f64) -> f64 {
        let v = self.var();
        if v > 0.0 {
            let d = x - self.mean;
            let q = d * d / (-2.0 * v);
            if q.is_finite() {
                q - HALF_LOG_TAU - 0.5 * libm::log(v)
            } else {
                f64::NEG_INFINITY
            }
        } else {
            f64::NEG_INFINITY
        }
    }

    /// Gaussian CDF at `x` via pure-Rust `libm::erf`, or `0.0` when standard
    /// deviation is zero.
    #[inline]
    pub fn cdf(&self, x: f64) -> f64 {
        let sd = self.var().sqrt();
        if sd > 0.0 {
            0.5 * (1.0 + libm::erf((x - self.mean) / sd / SQRT_2))
        } else {
            0.0
        }
    }

    /// Structural validity check for `STATE-3` restore.
    pub fn is_valid(&self) -> bool {
        self.n.is_finite()
            && self.n >= 0.0
            && self.mean.is_finite()
            && self.s.is_finite()
            && self.s >= 0.0
            && self.min.is_finite()
            && self.max.is_finite()
            && (self.n == 0.0 || self.min <= self.max)
    }
}

/// Shannon entropy in base-2 bits, matching `river`'s Rust order of operations:
/// `(-sum_c n_c log2(n_c) + N log2(N)) / N`.
#[inline]
pub(super) fn entropy(counts: &[f64]) -> f64 {
    let mut e = 0.0;
    let mut t = 0.0;
    for &c in counts {
        if c > 0.0 {
            e -= c * libm::log2(c);
            t += c;
        }
    }
    if t == 0.0 {
        0.0
    } else {
        (e + t * libm::log2(t)) / t
    }
}

/// Gini impurity `1 - sum_c (n_c / N)^2`, matching `river`'s Rust backend.
#[inline]
fn gini_impurity(counts: &[f64]) -> f64 {
    let mut t = 0.0;
    for &c in counts {
        t += c;
    }
    if t == 0.0 {
        return 1.0;
    }
    let mut sq = 0.0;
    for &c in counts {
        let p = c / t;
        sq += p * p;
    }
    1.0 - sq
}

/// Merit of a binary split `(left, right)` from parent `counts`.
#[inline]
pub(super) fn split_merit(
    counts: &[f64],
    left: &[f64],
    right: &[f64],
    lw: f64,
    rw: f64,
    criterion: SplitCriterion,
) -> f64 {
    let tot = lw + rw;
    if tot <= 0.0 {
        return 0.0;
    }
    match criterion {
        SplitCriterion::InfoGain => {
            entropy(counts) - (lw * entropy(left) + rw * entropy(right)) / tot
        }
        SplitCriterion::Gini => 1.0 - (lw * gini_impurity(left) + rw * gini_impurity(right)) / tot,
        SplitCriterion::Hellinger => {
            let tn = left[0] + right[0];
            let tp = left[1] + right[1];
            if tn <= 0.0 || tp <= 0.0 {
                return 0.0;
            }
            let a = (left[0] / tn).sqrt() - (left[1] / tp).sqrt();
            let b = (right[0] / tn).sqrt() - (right[1] / tp).sqrt();
            (a * a + b * b).sqrt()
        }
    }
}
