//! DDM, the drift detection method (Gama, Medas, Castillo and Rodrigues
//! 2004), as river's `drift.binary.DDM` computes it.
//!
//! Each row is an error (`x = 1`) or not (`x = 0`). With `p_t` the error rate
//! of the rows since the detector last started, kept as river keeps it, and
//! `s_t` its binomial standard deviation,
//!
//! ```text
//! p_t = p_{t−1} + (1/t)(x_t − p_{t−1})
//! s_t = sqrt(p_t (1 − p_t) / t)
//! ```
//!
//! once `t > warm_start`: the minimum moves to `(p_t, s_t)` where `p_t + s_t ≤
//! p_min + s_min`, and a drift is flagged where
//!
//! ```text
//! p_t + s_t > p_min + drift_threshold · s_min
//! ```
//!
//! after which the detector starts again from no rows. river's warning
//! level (`p_min + 2 s_min`) is not kept: nothing reads it.

use serde::{Deserialize, Serialize};

/// DDM on one stream of errors: the module doc has the equations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Ddm {
    /// Rows before the minimum is tracked: river's 30.
    warm_start: u64,
    /// The `s_min`s past `p_min + s_min` that flag: river's 3.
    drift_threshold: f64,
    /// Rows since the start.
    n: f64,
    /// The error rate.
    p: f64,
    p_min: f64,
    s_min: f64,
    ps_min: f64,
}

impl Ddm {
    pub fn new(warm_start: u64, drift_threshold: f64) -> Self {
        Self {
            warm_start,
            drift_threshold,
            n: 0.0,
            p: 0.0,
            p_min: f64::INFINITY,
            s_min: f64::INFINITY,
            ps_min: f64::INFINITY,
        }
    }

    /// Rows since the detector last started.
    pub fn n(&self) -> f64 {
        self.n
    }

    /// The error rate of those rows.
    pub fn p(&self) -> f64 {
        self.p
    }

    /// `(p_min, s_min)`, infinite before the first is recorded.
    pub fn min(&self) -> (f64, f64) {
        (self.p_min, self.s_min)
    }

    /// Whether a restored state keeps the shape every row relies on: a
    /// `drift_threshold` finite and above 0, as a spec takes it, a whole
    /// count of rows, a rate in `[0, 1]`, and a minimum that is either none
    /// yet (every part infinite) or a rate in `[0, 1]` with a deviation at
    /// least 0.
    pub fn sound(&self) -> bool {
        let none = self.p_min == f64::INFINITY
            && self.s_min == f64::INFINITY
            && self.ps_min == f64::INFINITY;
        let some = (0.0..=1.0).contains(&self.p_min)
            && self.s_min.is_finite()
            && self.s_min >= 0.0
            && self.ps_min.is_finite();
        self.drift_threshold.is_finite()
            && self.drift_threshold > 0.0
            && self.n.is_finite()
            && self.n >= 0.0
            && self.n.fract() == 0.0
            && (0.0..=1.0).contains(&self.p)
            && (none || some)
    }

    pub fn reset(&mut self) {
        *self = Self::new(self.warm_start, self.drift_threshold);
    }

    /// Feed one row, an error or not. Returns whether it flags a drift, and
    /// then starts again.
    pub fn update(&mut self, error: bool) -> bool {
        let x = if error { 1.0 } else { 0.0 };
        self.n += 1.0;
        self.p += (1.0 / self.n) * (x - self.p);
        let (p, n) = (self.p, self.n);
        let s = (p * (1.0 - p) / n).sqrt();
        if n > self.warm_start as f64 {
            if p + s <= self.ps_min {
                self.p_min = p;
                self.s_min = s;
                self.ps_min = self.p_min + self.s_min;
            }
            if p + s > self.p_min + self.drift_threshold * self.s_min {
                self.reset();
                return true;
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*state >> 11) as f64 / (1u64 << 53) as f64
    }

    /// The recursion written out: the rate as a count over the rows, the
    /// binomial deviation, the minimum of `p + s` past the warm start, and the
    /// flag where `p + s` passes `p_min + 3 s_min`.
    #[test]
    fn the_rate_and_the_minimum_are_the_definitions() {
        let mut d = Ddm::new(30, 1e9);
        let (mut errors, mut ps_min) = (0.0f64, f64::INFINITY);
        let mut s = 3u64;
        for t in 1..=2000u64 {
            let e = lcg(&mut s) < 0.2;
            assert!(!d.update(e));
            errors += if e { 1.0 } else { 0.0 };
            let p = errors / t as f64;
            assert!((d.p() - p).abs() < 1e-12, "row {t}");
            let sd = (p * (1.0 - p) / t as f64).sqrt();
            if t > 30 && p + sd <= ps_min + 1e-12 {
                ps_min = p + sd;
            }
            if t > 30 {
                let (pm, sm) = d.min();
                assert!((pm + sm - ps_min).abs() < 1e-9, "row {t}");
            } else {
                assert_eq!(d.min().0, f64::INFINITY, "row {t}: inside the warm start");
            }
        }
    }

    #[test]
    fn a_rise_in_the_error_rate_is_flagged_and_restarts() {
        // DDM's minimum is read from the rows so far, so a steady rate is
        // flagged now and then by chance: a few times in 3,000 rows at most.
        let mut d = Ddm::new(30, 3.0);
        let mut s = 5u64;
        let alarms = (0..3000).filter(|_| d.update(lcg(&mut s) < 0.2)).count();
        assert!(alarms <= 2, "{alarms} on a steady rate");
        d.reset();
        for _ in 0..1000 {
            d.update(lcg(&mut s) < 0.2);
        }
        let at = (0..3000)
            .position(|_| d.update(lcg(&mut s) < 0.6))
            .expect("a rate of 0.6 after 0.2 is flagged");
        assert!(at < 200, "{at} rows");
        assert_eq!(d.n(), 0.0, "started again");
        assert_eq!(d.min().0, f64::INFINITY);
    }

    /// The warm start, both sides: `t = warm_start` tracks nothing, `t =
    /// warm_start + 1` records the minimum.
    #[test]
    fn the_warm_start_is_strict() {
        let mut d = Ddm::new(10, 3.0);
        for t in 1..=10 {
            d.update(t % 3 == 0);
        }
        assert_eq!(d.min().0, f64::INFINITY);
        d.update(false);
        assert!(d.min().0.is_finite());
    }

    /// The flag is `>` the level: at the level exactly it is not.
    #[test]
    fn the_drift_level_itself_is_not_a_drift() {
        // warm_start 0: a first error has p = 1, s = 0, so p + s = 1 is the
        // minimum and the level; a second error keeps p + s = 1, at the level.
        let mut d = Ddm::new(0, 3.0);
        assert!(!d.update(true));
        assert!(!d.update(true), "at the level");
        let mut d = Ddm::new(0, 0.0);
        assert!(!d.update(false));
        assert!(d.update(true), "past it: p + s above p_min + 0");
    }

    #[test]
    fn a_saved_detector_goes_on_to_the_bit() {
        let mut a = Ddm::new(30, 3.0);
        let mut s = 9u64;
        for _ in 0..500 {
            a.update(lcg(&mut s) < 0.1);
        }
        let mut b: Ddm = rmp_serde::from_slice(&rmp_serde::to_vec_named(&a).unwrap()).unwrap();
        for _ in 0..2000 {
            let e = lcg(&mut s) < 0.3;
            assert_eq!(a.update(e), b.update(e));
        }
        assert_eq!(a, b);
    }

    /// The row count is the rows since the start, and the flag adds the
    /// deviation to the rate: a window whose minimum is a rate of 0.1 over
    /// 99 rows flags at 0.109 + 0.0312 = 0.140 past its level of 0.13, where
    /// the rate less its deviation (0.078) would not.
    #[test]
    fn the_flag_adds_the_deviation_to_the_rate() {
        let mut d = Ddm::new(0, 3.0);
        assert_eq!(d.n(), 0.0);
        d.update(false);
        d.update(false);
        assert_eq!(d.n(), 2.0);
        let mut d = Ddm::new(0, 3.0);
        (d.n, d.p, d.p_min, d.s_min, d.ps_min) = (99.0, 0.1, 0.1, 0.01, 0.11);
        assert!(d.update(true));
        assert_eq!(d.n(), 0.0, "started again");
        // The same state a row short of the level: no flag.
        let mut d = Ddm::new(0, 3.0);
        (d.n, d.p, d.p_min, d.s_min, d.ps_min) = (99.0, 0.1, 0.1, 0.01, 0.11);
        assert!(!d.update(false));
        assert_eq!(d.n(), 100.0);
    }
}
