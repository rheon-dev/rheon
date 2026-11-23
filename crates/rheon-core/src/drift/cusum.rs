//! The tabular two-sided CUSUM (Page 1954), on the clock.
//!
//! On a stream of standardized errors `z_t` (a residual over its scale, so 0
//! when the model is right on average), each arriving `d_t` clock units after
//! the one before:
//!
//! ```text
//! C⁺_t = max(0, C⁺_{t−1} + d_t (z_t − k))      an upward shift
//! C⁻_t = max(0, C⁻_{t−1} + d_t (−z_t − k))     a downward one
//! flag when  C⁺_t > h  or  C⁻_t > h
//! ```
//!
//! and both sums start again from 0 after a flag. `k`, Page's reference
//! value, is the shift tolerated, in units of the scale (half the shift to be
//! found, 0.5 by custom); `h`, the decision interval, how much accumulated
//! excess counts. At rows one clock unit apart this is Page's scheme exactly,
//! whose average run length on a standard normal stream at `k = 0.5` is about
//! 168 rows at `h = 4` and 465 at `h = 5`. Integrating over the clock, as the
//! library's Page-Hinkley detector does, makes the same shift held for the
//! same time count the same however many rows fall in it. The target mean is
//! 0: the errors of a model that is right on average.

use serde::{Deserialize, Serialize};

/// The two-sided CUSUM on one stream: the module doc has the equations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Cusum {
    /// The reference value `k`, in units of the scale.
    k: f64,
    /// The decision interval `h`, in units of the scale times the clock.
    h: f64,
    /// `C⁺`.
    up: f64,
    /// `C⁻`.
    down: f64,
}

impl Cusum {
    pub fn new(k: f64, h: f64) -> Self {
        Self {
            k,
            h,
            up: 0.0,
            down: 0.0,
        }
    }

    /// `(C⁺, C⁻)`.
    pub fn sums(&self) -> (f64, f64) {
        (self.up, self.down)
    }

    pub fn reset(&mut self) {
        self.up = 0.0;
        self.down = 0.0;
    }

    /// Feed one standardized error `z`, `d` clock units after the one before.
    /// Returns whether either sum passed `h`, and then starts both again.
    pub fn update(&mut self, z: f64, d: f64) -> bool {
        self.up = (self.up + d * (z - self.k)).max(0.0);
        self.down = (self.down + d * (-z - self.k)).max(0.0);
        if self.up > self.h || self.down > self.h {
            self.reset();
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SplitMix64;

    /// A standard normal by Box-Muller from a seeded `SplitMix64`.
    fn normal(g: &mut SplitMix64) -> f64 {
        let u = 1.0 - g.uniform();
        let v = g.uniform();
        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
    }

    /// Page's sums written out, each from its definition, on an irregular
    /// clock and a stream that shifts both ways.
    #[test]
    fn the_sums_are_pages_definitions() {
        let (k, h) = (0.5, 1e9);
        let mut c = Cusum::new(k, h);
        let (mut up, mut down) = (0.0f64, 0.0f64);
        let mut g = SplitMix64::new(3);
        for i in 0..2000 {
            let shift = match i / 500 {
                1 => 1.0,
                3 => -1.0,
                _ => 0.0,
            };
            let z = normal(&mut g) + shift;
            let d = 0.25 + 2.0 * g.uniform();
            assert!(!c.update(z, d));
            up = (up + d * (z - k)).max(0.0);
            down = (down + d * (-z - k)).max(0.0);
            let (u, w) = c.sums();
            assert_eq!(u.to_bits(), up.to_bits(), "row {i}");
            assert_eq!(w.to_bits(), down.to_bits(), "row {i}");
        }
    }

    /// The decision interval, both sides: a sum at `h` is not a flag, past it
    /// is, upward and downward, and a flag starts both sums again.
    #[test]
    fn h_itself_is_not_a_flag() {
        let mut c = Cusum::new(0.5, 2.0);
        assert!(!c.update(2.5, 1.0), "C+ = 2 = h");
        assert!(c.update(0.5 + 1e-9, 1.0), "C+ past h");
        assert_eq!(c.sums(), (0.0, 0.0));
        assert!(!c.update(-2.5, 1.0), "C- = 2 = h");
        assert!(c.update(-0.5 - 1e-9, 1.0), "C- past h");
        assert_eq!(c.sums(), (0.0, 0.0));
        // The clock weighs the excess: one unit of z - k over two units of
        // clock is two.
        let mut c = Cusum::new(0.5, 1.9);
        assert!(c.update(1.5, 2.0));
    }

    /// Page's average run length at `k = 0.5`, two-sided, on a standard
    /// normal stream: about 168 rows at `h = 4` and 465 at `h = 5`
    /// (Montgomery, *Introduction to Statistical Quality Control*, table of
    /// the tabular CUSUM's ARL). Each run is a geometric-like count whose
    /// standard deviation is about its mean, so 4,000 runs hold the mean to
    /// within 1.6% (one standard error); the bound is four.
    #[test]
    fn the_average_run_length_is_pages() {
        for (h, arl) in [(4.0, 168.0), (5.0, 465.0)] {
            let mut g = SplitMix64::new(h as u64 * 7919);
            let runs = 4000;
            let mut total = 0u64;
            let mut c = Cusum::new(0.5, h);
            for _ in 0..runs {
                let mut n = 0u64;
                loop {
                    n += 1;
                    // A run is geometric-like: 100,000 rows is hundreds of
                    // times its mean, so a detector that never flags fails
                    // here and does not hang.
                    assert!(n <= 100_000, "h = {h}: no flag in {n} rows");
                    if c.update(normal(&mut g), 1.0) {
                        break;
                    }
                }
                total += n;
            }
            let mean = total as f64 / runs as f64;
            let se = arl / (runs as f64).sqrt();
            assert!(
                (mean - arl).abs() < 4.0 * se,
                "h = {h}: {mean} rows against {arl}"
            );
        }
    }

    /// A shift of one standard deviation is found in about 10 rows at `h = 5`
    /// (the same table: 10.4).
    #[test]
    fn a_shift_of_one_scale_is_found_quickly() {
        let mut g = SplitMix64::new(5);
        let mut total = 0u64;
        let runs = 2000;
        for _ in 0..runs {
            let mut c = Cusum::new(0.5, 5.0);
            let mut n = 0u64;
            while !c.update(normal(&mut g) + 1.0, 1.0) {
                n += 1;
                assert!(n <= 100_000, "no flag in {n} rows of a shift");
            }
            total += n + 1;
        }
        let mean = total as f64 / runs as f64;
        assert!((mean - 10.4).abs() < 0.6, "{mean}");
    }

    #[test]
    fn a_saved_detector_goes_on_to_the_bit() {
        let mut a = Cusum::new(0.5, 5.0);
        let mut g = SplitMix64::new(13);
        for _ in 0..300 {
            a.update(normal(&mut g), 1.0);
        }
        let mut b: Cusum = rmp_serde::from_slice(&rmp_serde::to_vec_named(&a).unwrap()).unwrap();
        for _ in 0..3000 {
            let z = normal(&mut g) + 0.3;
            assert_eq!(a.update(z, 0.5), b.update(z, 0.5));
        }
        assert_eq!(a, b);
    }
}
