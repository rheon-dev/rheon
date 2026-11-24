//! Clock-step statistics (PLAN 564, part 3): for one stream of times, the
//! step `s_i = t_i − t_{i−1}` between each time and the one before it, and
//! its decayed mean, variance and standard deviation, or the decayed count
//! of its zero steps or of its steps back, on the half-life rules every
//! window operator decays by.
//!
//! **Definition.** Step `i` of the steps so far weighs
//! `ω_i = w_i · 2^(−(τ_t − τ_i)/h)`: its own weight `w_i`, decayed by the
//! policy clock `τ` across the half-life `h` to the row read, `t`. With
//! `W = Σ ω_i`:
//!
//! ```text
//! mean    m  = Σ ω_i s_i / W
//! var     v  = Σ ω_i (s_i − m)² / W                    (bias = true)
//!         v  = Σ ω_i (s_i − m)² / W / (1 − Σ ω_i² / W²) (bias = false, Kish)
//! std     √v
//! zeros   Σ ω_i [s_i = 0]
//! backs   Σ ω_i [s_i < 0]
//! ```
//!
//! The mean and the variances are Polars' `ewm_mean`, and `ewm_var` under
//! `bias` (its default `bias=False` is Kish's correction above), on the
//! steps; a count is the decayed weight of the steps it counts.
//!
//! **The update.** Each statistic is [`EwmMoments`] over the steps (554's
//! moments, Pébay's merge, the order-2 update `ew_cov`'s to the bit): the
//! mean its first cumulant, the variance its central moment of order 2.
//! A count is the weight `W` of an [`EwmMoments`] that learns only the
//! steps it counts, every other row one of weight 0, so the count's
//! recursion is the moments' own `W' = λ W + w`. No second running sum is
//! kept.
//!
//! **Weight.** A row of weight 0, or a row with no step (the first time
//! of a stream, or a time after a missing one), learns nothing (rule 9):
//! the clock decays the statistics, and no mean moves; where every row so
//! far is one, `W` is 0 and the moments make no division. [`ClockSteps::weight`]
//! is the weight before the row (rule 8).
//!
//! **Withholding.** The mean and the variances are withheld while `W` is 0,
//! and an unbiased variance while its denominator keeps no digit (one step
//! carrying all the weight): the moments' own conditions, never a count of
//! rows. A count is never withheld: before any step it is 0.
//!
//! **State.** The statistic and its form, and one [`EwmMoments`] of one
//! term over one column: its settings, the weight and one summary.

use serde::{Deserialize, Deserializer, Serialize};

use crate::clock::Decay;
use crate::ewm_moments::{EwmMoments, EwmMomentsCfg, MomentKind, MomentTerm};

/// What [`ClockSteps`] reads of the steps.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StepStat {
    /// The decayed mean step.
    Mean,
    /// The decayed variance of the steps.
    Var,
    /// The square root of [`StepStat::Var`].
    Std,
    /// The decayed count of steps of exactly 0: rows that repeat the time
    /// before them.
    Zeros,
    /// The decayed count of steps below 0: rows before the time before
    /// them.
    Backs,
}

impl StepStat {
    /// Every statistic with its name, the name first.
    pub const NAMES: [(&'static str, StepStat); 5] = [
        ("mean", StepStat::Mean),
        ("var", StepStat::Var),
        ("std", StepStat::Std),
        ("zeros", StepStat::Zeros),
        ("backs", StepStat::Backs),
    ];

    /// The statistic `name` names, as the formula writes it.
    pub fn parse(name: &str) -> Option<Self> {
        Self::NAMES
            .iter()
            .find(|(n, _)| *n == name)
            .map(|&(_, s)| s)
    }

    /// The statistic's name, as the formula writes it.
    pub fn name(self) -> &'static str {
        Self::NAMES
            .iter()
            .find(|(_, s)| *s == self)
            .map_or("mean", |(n, _)| n)
    }

    /// Whether it is a spread, which takes `bias`.
    pub fn is_spread(self) -> bool {
        matches!(self, StepStat::Var | StepStat::Std)
    }

    /// Whether it is a count of steps of one sign.
    fn is_count(self) -> bool {
        matches!(self, StepStat::Zeros | StepStat::Backs)
    }
}

/// The decayed statistics of a stream's clock steps ([the module
/// docs](self)).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClockSteps {
    stat: StepStat,
    #[serde(deserialize_with = "reloaded")]
    moments: Box<EwmMoments>,
}

/// A saved model, its derived parts formed again as a state file's are.
fn reloaded<'de, D: Deserializer<'de>>(d: D) -> Result<Box<EwmMoments>, D::Error> {
    let m = EwmMoments::deserialize(d)?;
    m.reloaded().map(Box::new).map_err(serde::de::Error::custom)
}

impl ClockSteps {
    /// The statistic `stat` of the steps, decayed across a half-life of
    /// `half_life` units of the policy clock (`inf` weighs every step the
    /// same); a spread unbiased by Kish's correction unless `bias`.
    ///
    /// # Errors
    ///
    /// A half-life that is not above 0.
    pub fn new(half_life: f64, stat: StepStat, bias: bool) -> Result<Self, String> {
        let decay = Decay::Halflife(half_life);
        decay.check()?;
        let term = if stat.is_spread() {
            MomentTerm::of_column("step", 2, MomentKind::Central, bias)
        } else {
            MomentTerm::of_column("step", 1, MomentKind::Cumulant, true)
        };
        let moments = EwmMoments::new(EwmMomentsCfg::new(decay, vec![term]))?;
        Ok(Self {
            stat,
            moments: Box::new(moments),
        })
    }

    /// What it reads of the steps.
    pub fn stat(&self) -> StepStat {
        self.stat
    }

    /// Take one row: `d_clock` units of the policy clock since the last
    /// row, its step `s` (`None` where the row has none: the first time,
    /// or one after a missing time), and its weight `w`, finite and at
    /// least 0. A step the caller passes is finite.
    pub fn step(&mut self, d_clock: f64, s: Option<f64>, w: f64) {
        debug_assert!(
            s.is_none_or(f64::is_finite),
            "a missing step is the caller's"
        );
        debug_assert!(
            w.is_finite() && w >= 0.0,
            "a weight is finite and at least 0"
        );
        let lam = self.moments.cfg().decay.factor(d_clock);
        let (x, w) = match (s, self.stat) {
            (None, _) => (0.0, 0.0),
            (Some(s), StepStat::Zeros) => (0.0, if s == 0.0 { w } else { 0.0 }),
            (Some(s), StepStat::Backs) => (0.0, if s < 0.0 { w } else { 0.0 }),
            (Some(s), _) => (s, w),
        };
        self.moments.update(x, lam, w);
    }

    /// The weight of the steps taken, decayed to the last row: what the
    /// next row's `n_eff` is before its own decay (rule 8). For a count it
    /// is the count.
    pub fn weight(&self) -> f64 {
        self.moments.n_eff()
    }

    /// The statistic as it stands; `None` where it is withheld (a mean or
    /// a spread with no step of weight, or an unbiased spread whose
    /// correction keeps no digit).
    pub fn value(&self) -> Option<f64> {
        if self.stat.is_count() {
            return Some(self.moments.n_eff());
        }
        let v = self.moments.value(0).ok()?;
        Some(match self.stat {
            StepStat::Std => v.max(0.0).sqrt(),
            _ => v,
        })
    }

    /// Forget every step: a new session, a reset of the clock, or a gap
    /// past `gap_cap`, after which the next step starts the statistics
    /// again.
    pub fn restart(&mut self) {
        let cfg = self.moments.cfg().clone();
        *self.moments = EwmMoments::new(cfg).expect("its settings were checked");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Steps `(τ, s, w)`, a missing step `None`, through the update, each
    /// clock step `τ_i − τ_{i−1}`.
    fn fed(rows: &[(f64, Option<f64>, f64)], h: f64, stat: StepStat, bias: bool) -> ClockSteps {
        let mut c = ClockSteps::new(h, stat, bias).unwrap();
        let mut last = rows.first().map_or(0.0, |r| r.0);
        for &(t, s, w) in rows {
            c.step(t - last, s, w);
            last = t;
        }
        c
    }

    /// The statistic from its definition, in two passes over the decayed
    /// weights: the brute force the update is held to here (the Python
    /// suite holds the operator to Polars' `ewm_*` and statsmodels).
    fn brute(rows: &[(f64, Option<f64>, f64)], h: f64, stat: StepStat, bias: bool) -> Option<f64> {
        let t_end = rows.last().map_or(0.0, |r| r.0);
        let kept: Vec<(f64, f64)> = rows
            .iter()
            .filter_map(|&(t, s, w)| {
                let f = if h.is_infinite() {
                    1.0
                } else {
                    (-(t_end - t) / h).exp2()
                };
                s.map(|s| (s, w * f))
            })
            .collect();
        let count =
            |keep: fn(f64) -> bool| kept.iter().filter(|(s, _)| keep(*s)).map(|p| p.1).sum();
        match stat {
            StepStat::Zeros => return Some(count(|s| s == 0.0)),
            StepStat::Backs => return Some(count(|s| s < 0.0)),
            _ => {}
        }
        let w: f64 = kept.iter().map(|p| p.1).sum();
        if w == 0.0 {
            return None;
        }
        let m = kept.iter().map(|(s, o)| o * s).sum::<f64>() / w;
        if stat == StepStat::Mean {
            return Some(m);
        }
        let mut v = kept.iter().map(|(s, o)| o * (s - m).powi(2)).sum::<f64>() / w;
        if !bias {
            let r2 = kept.iter().map(|p| p.1 * p.1).sum::<f64>() / (w * w);
            if 1.0 - r2 <= 64.0 * f64::EPSILON {
                return None;
            }
            v /= 1.0 - r2;
        }
        Some(if stat == StepStat::Std { v.sqrt() } else { v })
    }

    fn close(a: Option<f64>, b: Option<f64>) -> bool {
        match (a, b) {
            (Some(x), Some(y)) => (x - y).abs() <= 1e-12 * (1.0 + x.abs().max(y.abs())),
            (None, None) => true,
            _ => false,
        }
    }

    fn rows() -> Vec<(f64, Option<f64>, f64)> {
        // An uneven clock: steps of every sign, a repeat, a missing step,
        // a zero weight among them.
        vec![
            (0.0, None, 1.0),
            (1.0, Some(1.0), 1.0),
            (1.5, Some(0.5), 1.0),
            (1.5, Some(0.0), 1.0),
            (3.0, Some(1.5), 1.0),
            (3.2, Some(-0.2), 1.0),
            (4.0, None, 1.0),
            (6.0, Some(2.0), 0.0),
            (7.0, Some(1.0), 2.0),
            (7.0, Some(0.0), 1.0),
            (9.5, Some(2.5), 0.5),
        ]
    }

    #[test]
    fn each_statistic_is_its_definition_at_every_row() {
        let all = rows();
        for (_, stat) in StepStat::NAMES {
            for bias in [true, false] {
                for h in [0.7, 3.0, f64::INFINITY] {
                    for end in 1..=all.len() {
                        let got = fed(&all[..end], h, stat, bias).value();
                        let want = brute(&all[..end], h, stat, bias);
                        assert!(
                            close(got, want),
                            "{stat:?} {bias} {h} {end}: {got:?} {want:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn a_count_counts_its_sign_and_no_other() {
        // Both sides of each threshold: a step of exactly 0 and the
        // smallest either side of it.
        let tiny = f64::MIN_POSITIVE;
        let rows = [
            (0.0, Some(0.0), 1.0),
            (0.0, Some(-0.0), 1.0),
            (0.0, Some(tiny), 1.0),
            (0.0, Some(-tiny), 1.0),
        ];
        let zeros = fed(&rows, f64::INFINITY, StepStat::Zeros, true);
        let backs = fed(&rows, f64::INFINITY, StepStat::Backs, true);
        assert_eq!(zeros.value(), Some(2.0));
        assert_eq!(backs.value(), Some(1.0));
    }

    #[test]
    fn a_mean_is_withheld_until_a_step_of_weight_and_a_count_is_zero() {
        // Rule 9: rows with no step or of weight 0, the first included,
        // leave no NaN and move nothing.
        let rows = [(0.0, None, 1.0), (1.0, Some(3.0), 0.0), (2.0, None, 1.0)];
        for (_, stat) in StepStat::NAMES {
            let c = fed(&rows, 2.0, stat, false);
            assert_eq!(c.weight(), 0.0);
            let want = stat.is_count().then_some(0.0);
            assert_eq!(c.value(), want, "{stat:?}");
        }
        // One step of weight: a mean, a biased spread of 0, and an
        // unbiased spread withheld, since one step carries all the weight.
        let one = [(0.0, None, 1.0), (1.0, Some(3.0), 1.0)];
        assert_eq!(fed(&one, 2.0, StepStat::Mean, false).value(), Some(3.0));
        assert_eq!(fed(&one, 2.0, StepStat::Var, true).value(), Some(0.0));
        assert_eq!(fed(&one, 2.0, StepStat::Std, false).value(), None);
        let two = [(0.0, Some(1.0), 1.0), (1.0, Some(3.0), 1.0)];
        assert!(fed(&two, 2.0, StepStat::Std, false).value().is_some());
    }

    #[test]
    fn the_weight_is_the_weight_before_the_row() {
        // Rule 8: the weight read before a row is the steps' weight decayed
        // to the row before, and the row's own decay is not yet in it.
        let mut c = ClockSteps::new(1.0, StepStat::Mean, true).unwrap();
        c.step(0.0, Some(1.0), 1.0);
        assert_eq!(c.weight(), 1.0);
        c.step(1.0, Some(1.0), 1.0);
        assert_eq!(c.weight(), 1.5);
    }

    #[test]
    fn a_restart_forgets_every_step() {
        let mut c = fed(&rows(), 3.0, StepStat::Var, true);
        c.restart();
        assert_eq!(c, ClockSteps::new(3.0, StepStat::Var, true).unwrap());
    }

    #[test]
    fn a_saved_state_loads_and_continues_to_the_bit() {
        let all = rows();
        for (_, stat) in StepStat::NAMES {
            let whole = fed(&all, 3.0, stat, false);
            for cut in 0..all.len() {
                let first = fed(&all[..=cut], 3.0, stat, false);
                let bytes = rmp_serde::to_vec_named(&first).unwrap();
                let mut back: ClockSteps = rmp_serde::from_slice(&bytes).unwrap();
                assert_eq!(back, first);
                let mut last = all[cut].0;
                for &(t, s, w) in &all[cut + 1..] {
                    back.step(t - last, s, w);
                    last = t;
                }
                assert_eq!(
                    back.value().map(f64::to_bits),
                    whole.value().map(f64::to_bits)
                );
                assert_eq!(back, whole);
            }
        }
    }

    /// The saved state is the same size after one step and after ten
    /// thousand, for every statistic: nothing grows with the stream. The
    /// size, under 256 bytes of msgpack (the moments' settings and term
    /// name, `W` and one summary of one leg), is the state's whole cost.
    #[test]
    fn the_state_does_not_grow_with_the_stream() {
        for (_, stat) in StepStat::NAMES {
            let mut c = ClockSteps::new(30.0, stat, false).unwrap();
            c.step(0.0, Some(1.5), 1.0);
            c.step(1.0, Some(0.5), 1.0);
            let early = rmp_serde::to_vec_named(&c).unwrap().len();
            for i in 0..10_000 {
                c.step(0.5, Some(f64::from(i % 7) - 1.0), 1.0);
            }
            let late = rmp_serde::to_vec_named(&c).unwrap().len();
            assert_eq!(early, late, "{stat:?}");
            assert!(late < 256, "{stat:?}: {late} bytes");
        }
    }

    #[test]
    fn a_half_life_not_above_zero_is_refused() {
        for h in [0.0, -1.0, f64::NAN] {
            assert!(ClockSteps::new(h, StepStat::Mean, true).is_err());
        }
    }

    #[test]
    fn every_name_reads_back() {
        for (n, s) in StepStat::NAMES {
            assert_eq!(StepStat::parse(n), Some(s));
            assert_eq!(s.name(), n);
        }
        assert_eq!(StepStat::parse("median"), None);
        assert!(StepStat::Var.is_spread() && StepStat::Std.is_spread());
        assert!(!StepStat::Mean.is_spread() && !StepStat::Zeros.is_spread());
    }
}
