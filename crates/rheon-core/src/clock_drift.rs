//! The drift between two clocks (PLAN 564, part 2): where each row carries
//! the time an event happened, `e_t`, and the time it arrived, `a_t`, the
//! lag `d_t = a_t − e_t`, its decayed mean, and the decayed slope of the lag
//! on the arrival time: how fast the lag grows, in seconds of lag a second
//! of arrival when both are times.
//!
//! **Definition.** Row `i` of the rows so far weighs
//! `ω_i = w_i · 2^(−(τ_t − τ_i)/h)`: its own weight `w_i`, decayed by the
//! policy clock `τ` across the half-life `h` to the row read, `t`. Then
//!
//! ```text
//! W = Σ ω_i,   ā = Σ ω_i a_i / W,   d̄ = Σ ω_i d_i / W,
//! C_aa = Σ ω_i (a_i − ā)²,   C_ad = Σ ω_i (a_i − ā)(d_i − d̄),
//! lag = d̄,   slope = C_ad / C_aa,
//! ```
//!
//! the weighted least-squares fit of `d` on `a` with weights `ω`, which is
//! its oracle (statsmodels' WLS). The lag is withheld while `W = 0`, and the
//! slope while `C_aa = 0`: every row of weight arrived at one time, so no
//! slope is defined, and none is guessed.
//!
//! **The update.** The two sums are kept about their own means, never as
//! a mean square less a squared mean, which cancels where the arrivals are
//! far from 0 next to their spread. A step of the policy clock `Δτ` scales
//! the weight and the sums about the means by `f = 2^(−Δτ/h)`, which moves
//! no mean; then a row of weight `w` joins as in West's weighted update
//! (1979):
//!
//! ```text
//! W' = f W + w,   δ = a − ā,   ā' = ā + (w/W') δ,   d̄' = d̄ + (w/W') (d − d̄),
//! C_aa' = f C_aa + w δ (a − ā'),   C_ad' = f C_ad + w δ (d − d̄').
//! ```
//!
//! So `C_aa` is 0 exactly while every arrival of weight is one value
//! (`δ = 0`), and the slope is withheld on the data's own condition, never
//! after a fixed count of rows. The caller centres the arrivals (subtracts
//! a fixed origin, such as the stream's first arrival, in integers where
//! the clock is one), which moves no slope and keeps a timestamp's
//! sixteen digits for its spread.
//!
//! **Weight.** A row of weight 0 learns nothing (rule 9): the clock still
//! decays the sums, and no mean moves; the first row of a stream may be
//! one, where `W' = 0` would divide 0 by 0, so no division is made.
//! [`ClockDrift::weight`] is the weight before the row (rule 8).
//!
//! **State.** The half-life and five doubles: 56 bytes, bounded by nothing
//! the stream does.

use serde::{Deserialize, Serialize};

use crate::clock::Decay;

/// The lag between two clocks and its slope on the arrival time ([the
/// module docs](self)).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClockDrift {
    decay: Decay,
    /// `W`, the decayed weight.
    w: f64,
    /// `ā` and `d̄`, the weighted means of the centred arrival and the lag.
    mean_a: f64,
    mean_d: f64,
    /// `C_aa` and `C_ad`, the sums about the means.
    c_aa: f64,
    c_ad: f64,
}

impl ClockDrift {
    /// A drift that decays across a half-life of `half_life` units of the
    /// policy clock; `inf` weighs every row the same.
    ///
    /// # Errors
    ///
    /// A half-life that is not above 0.
    pub fn new(half_life: f64) -> Result<Self, String> {
        let decay = Decay::Halflife(half_life);
        decay.check()?;
        Ok(Self {
            decay,
            w: 0.0,
            mean_a: 0.0,
            mean_d: 0.0,
            c_aa: 0.0,
            c_ad: 0.0,
        })
    }

    /// Take one row: `d_clock` units of the policy clock since the last
    /// row, its centred arrival `a`, its lag `d` and its weight `w`. The
    /// caller passes only finite `a` and `d`, and a weight that is finite
    /// and at least 0; a row missing either value is not passed at all, and
    /// its clock step is carried by the next row's `d_clock`.
    pub fn step(&mut self, d_clock: f64, a: f64, d: f64, w: f64) {
        debug_assert!(
            a.is_finite() && d.is_finite(),
            "a missing value is the caller's"
        );
        debug_assert!(
            w.is_finite() && w >= 0.0,
            "a weight is finite and at least 0"
        );
        let f = self.decay.factor(d_clock);
        let total = f * self.w + w;
        self.c_aa *= f;
        self.c_ad *= f;
        self.w = total;
        // A row of weight 0 moves no mean, and before the first row of
        // weight `total` is 0, where `w / total` is 0/0 (rule 9).
        if w > 0.0 {
            let share = w / total;
            let da = a - self.mean_a;
            self.mean_a += share * da;
            self.mean_d += share * (d - self.mean_d);
            self.c_aa += w * da * (a - self.mean_a);
            self.c_ad += w * da * (d - self.mean_d);
        }
    }

    /// The weight of the rows taken, decayed to the last of them: what the
    /// next row's `n_eff` is before its own decay (rule 8).
    pub fn weight(&self) -> f64 {
        self.w
    }

    /// The decayed mean lag, `None` while no row of weight was taken.
    pub fn lag(&self) -> Option<f64> {
        (self.w > 0.0).then_some(self.mean_d)
    }

    /// The decayed slope of the lag on the arrival time, `None` while the
    /// arrivals of weight have no spread (`C_aa = 0`): one row, or rows
    /// that all arrived at one time.
    pub fn slope(&self) -> Option<f64> {
        (self.c_aa > 0.0).then(|| self.c_ad / self.c_aa)
    }

    /// Forget every row: a new session, a reset of the clock, or a gap past
    /// `gap_cap`, after which the next row starts the sums again.
    pub fn restart(&mut self) {
        let decay = self.decay;
        *self = Self {
            decay,
            w: 0.0,
            mean_a: 0.0,
            mean_d: 0.0,
            c_aa: 0.0,
            c_ad: 0.0,
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The weighted least-squares fit from its definition, the sums taken
    /// about the weighted means in two passes: the brute force the update
    /// is held to here (the Python suite holds it to statsmodels' WLS).
    fn brute(rows: &[(f64, f64, f64, f64)], h: f64) -> (Option<f64>, Option<f64>) {
        let t_end = rows.last().map_or(0.0, |r| r.0);
        let om: Vec<f64> = rows
            .iter()
            .map(|&(t, _, _, w)| {
                w * if h.is_infinite() {
                    1.0
                } else {
                    (-(t_end - t) / h).exp2()
                }
            })
            .collect();
        let w: f64 = om.iter().sum();
        if w == 0.0 {
            return (None, None);
        }
        let ma = rows.iter().zip(&om).map(|(r, o)| o * r.1).sum::<f64>() / w;
        let md = rows.iter().zip(&om).map(|(r, o)| o * r.2).sum::<f64>() / w;
        let caa: f64 = rows
            .iter()
            .zip(&om)
            .map(|(r, o)| o * (r.1 - ma).powi(2))
            .sum();
        let cad: f64 = rows
            .iter()
            .zip(&om)
            .map(|(r, o)| o * (r.1 - ma) * (r.2 - md))
            .sum();
        (Some(md), (caa > 0.0).then(|| cad / caa))
    }

    /// Rows of `(τ, a, d, w)` through the update, each step `τ_i − τ_{i−1}`.
    fn fed(rows: &[(f64, f64, f64, f64)], h: f64) -> ClockDrift {
        let mut c = ClockDrift::new(h).unwrap();
        let mut last = rows.first().map_or(0.0, |r| r.0);
        for &(t, a, d, w) in rows {
            c.step(t - last, a, d, w);
            last = t;
        }
        c
    }

    fn close(a: Option<f64>, b: Option<f64>, tol: f64) -> bool {
        match (a, b) {
            (Some(x), Some(y)) => (x - y).abs() <= tol * (1.0 + y.abs()),
            (None, None) => true,
            _ => false,
        }
    }

    /// A lag that grows 2 ms a second with a wobble, on an uneven clock,
    /// under a finite and an infinite half-life: the update is the fit from
    /// its definition at every row.
    #[test]
    fn the_update_is_the_weighted_fit_at_every_row() {
        let mut rows = Vec::new();
        let mut t = 0.0;
        for i in 0..300 {
            t += 0.5 + f64::from(i % 7) * 0.3;
            let d = 0.1 + 0.002 * t + 0.01 * (f64::from(i) * 1.3).sin();
            let w = if i % 11 == 0 {
                0.0
            } else {
                1.0 + f64::from(i % 3)
            };
            rows.push((t, t, d, w));
        }
        for h in [5.0, 60.0, f64::INFINITY] {
            for n in 1..=rows.len() {
                let c = fed(&rows[..n], h);
                let (lag, slope) = brute(&rows[..n], h);
                assert!(
                    close(c.lag(), lag, 1e-12),
                    "h {h}, n {n}: {:?} {lag:?}",
                    c.lag()
                );
                assert!(
                    close(c.slope(), slope, 1e-9),
                    "h {h}, n {n}: {:?} {slope:?}",
                    c.slope()
                );
            }
        }
        let c = fed(&rows, f64::INFINITY);
        assert!((c.slope().unwrap() - 0.002).abs() < 1e-4, "{:?}", c.slope());
    }

    /// Both sides of the slope's condition: one row, and any number of rows
    /// at one arrival, have no spread and no slope; the next row at another
    /// arrival has both. The lag is there from the first row of weight.
    #[test]
    fn the_slope_is_withheld_while_the_arrivals_have_no_spread() {
        let mut c = ClockDrift::new(10.0).unwrap();
        assert_eq!((c.lag(), c.slope()), (None, None));
        c.step(0.0, 5.0, 0.2, 1.0);
        assert_eq!((c.lag(), c.slope()), (Some(0.2), None));
        for d in [0.3, 0.7, 0.1] {
            c.step(1.0, 5.0, d, 1.0);
            assert_eq!(c.slope(), None, "every arrival is 5: C_aa is 0 exactly");
        }
        assert!(c.lag().is_some());
        c.step(1.0, 5.0f64.next_up(), 0.4, 1.0);
        assert!(
            c.slope().is_some_and(f64::is_finite),
            "one ulp of spread is a spread"
        );
    }

    /// Rule 9: zero-weight rows, the first ones included, learn nothing and
    /// divide nothing; the clock still decays what came before. Rule 8: the
    /// weight read is the weight before the next row's own decay.
    #[test]
    fn a_row_of_weight_zero_learns_nothing() {
        let mut c = ClockDrift::new(1.0).unwrap();
        c.step(0.0, 1.0, 9.0, 0.0);
        c.step(3.0, 2.0, 9.0, 0.0);
        assert_eq!((c.weight(), c.lag(), c.slope()), (0.0, None, None));
        assert!(!c.mean_a.is_nan() && !c.mean_d.is_nan(), "no 0/0");
        c.step(1.0, 0.0, 1.0, 1.0);
        c.step(1.0, 1.0, 2.0, 1.0);
        let before = c.clone();
        c.step(1.0, 50.0, -100.0, 0.0);
        assert_eq!(c.lag(), before.lag());
        assert_eq!(
            c.slope(),
            before.slope(),
            "a scale of both sums moves no slope"
        );
        assert_eq!(c.weight(), before.weight() * 0.5);
        assert_eq!(
            before.weight(),
            1.5,
            "1·½ + 1: the weight before the next row's decay"
        );
    }

    /// No decay across a step of 0, and the half-life's decay across one.
    #[test]
    fn the_clock_decays_the_weight_by_its_half_life() {
        let mut c = ClockDrift::new(2.0).unwrap();
        c.step(0.0, 0.0, 0.0, 4.0);
        c.step(0.0, 0.0, 0.0, 0.0);
        assert_eq!(c.weight(), 4.0);
        c.step(2.0, 0.0, 0.0, 0.0);
        assert_eq!(c.weight(), 2.0);
        c.step(4.0, 0.0, 0.0, 1.0);
        assert_eq!(c.weight(), 1.5);
    }

    /// A restart forgets every row and keeps the half-life.
    #[test]
    fn a_restart_forgets_every_row() {
        let mut c = fed(&[(0.0, 0.0, 1.0, 1.0), (1.0, 1.0, 3.0, 1.0)], 4.0);
        assert!(c.slope().is_some());
        c.restart();
        assert_eq!(c, ClockDrift::new(4.0).unwrap());
    }

    /// Centring is the caller's: arrivals near 1.7e9 seconds, their spread
    /// of seconds, lose a slope that the same arrivals less the first one
    /// keep to 1e-12.
    #[test]
    fn centred_arrivals_keep_the_slope() {
        let origin = 1.7e9;
        let rows: Vec<_> = (0..50)
            .map(|i| {
                let a = f64::from(i) * 0.25;
                (a, a, 0.05 + 1e-3 * a, 1.0)
            })
            .collect();
        let c = fed(&rows, 30.0);
        assert!((c.slope().unwrap() - 1e-3).abs() < 1e-12, "{:?}", c.slope());
        let far: Vec<_> = rows.iter().map(|r| (r.0, r.1 + origin, r.2, r.3)).collect();
        let f = fed(&far, 30.0);
        assert!(
            (f.slope().unwrap() - 1e-3).abs() < 1e-6,
            "about the means, far is still close: {:?}",
            f.slope()
        );
    }

    /// A state saved at any row continues to the bit.
    #[test]
    fn a_saved_drift_continues_to_the_bit() {
        let rows: Vec<_> = (0..40)
            .map(|i| {
                let t = f64::from(i) * 0.7;
                (t, t, (t * 0.37).cos(), f64::from(i % 4))
            })
            .collect();
        let whole = fed(&rows, 3.0);
        for cut in 1..rows.len() {
            let a = fed(&rows[..cut], 3.0);
            let bytes = rmp_serde::to_vec_named(&a).unwrap();
            let mut b: ClockDrift = rmp_serde::from_slice(&bytes).unwrap();
            let mut last = rows[cut - 1].0;
            for &(t, x, d, w) in &rows[cut..] {
                b.step(t - last, x, d, w);
                last = t;
            }
            assert_eq!(b, whole, "cut at {cut}");
        }
    }

    /// The state's size, as the module docs state it.
    #[test]
    fn the_state_is_56_bytes() {
        assert_eq!(std::mem::size_of::<ClockDrift>(), 56);
    }

    #[test]
    fn the_half_life_must_be_above_zero() {
        assert!(ClockDrift::new(f64::INFINITY).is_ok());
        for bad in [0.0, -1.0, f64::NAN] {
            assert!(
                ClockDrift::new(bad)
                    .unwrap_err()
                    .starts_with("half_life must be above 0")
            );
        }
    }
}
