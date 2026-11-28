//! The window core (docs/PLAN.md tasks 78, 143 and 144): exponentially
//! weighted operators over a stream, looking back and ahead, one pass,
//! O(window) memory, exact.
//!
//! Everything stateful is a *kernel*: a direction, a half-life, a window and
//! a `closed` rule on the policy clock. Any number of *operators* share a
//! kernel's queue -- a time-weighted mean, a decayed sum, a rate -- each
//! reading one input of the row. The formula around the operators is
//! Polars' (`formula.rs`), and an `increment` is the frame runner's; this
//! core sees neither.
//!
//! **Definitions**, each held to a brute-force loop in the tests:
//!
//! - `ewm_mean` / `rewm_mean`: the time-weighted mean, Polars' `ewm_mean_by`.
//!   Looking back, row `i`'s value is held over `(t_{i-1}, t_i]`, the first
//!   value of a stretch from before it (`-inf`, as `ewm_mean_by`'s `a_1 = 1`;
//!   from `t_1 - w` under a window of `w`); the mean over `(T - w, T]` weighs
//!   each held value by `∫ λ^(T - s) ds` over its interval inside the
//!   window. Looking ahead, value `j` is held over `[t_j, t_{j+1})`, until
//!   the next row, and weighed by `∫ λ^(s - t) ds` inside `(t, t + w]`. A
//!   row at a repeated stamp holds no interval, so it moves no mean -- as in
//!   Polars.
//! - `ewm_sum` / `rewm_sum`: `Σ λ^|t_j - t| x_j` over the rows of the window,
//!   each counted once at its own time (`ewm_sum_by` on distinct stamps; at
//!   a repeated stamp every row holds the stamp's whole window, so each
//!   carries the stamp's total where `ewm_sum_by`'s is a running sum).
//! - `ewm_rate` / `rewm_rate`: the sum over the decayed time the window
//!   covers, `∫_0^T λ^s ds = h/ln2 · (1 - 2^(-T/h))`, `T` the window's span
//!   inside the stretch.
//! - `ewm_var` / `ewm_std` (task 212), looking back only: the variance about
//!   `ewm_mean`, each value weighed by the mean's weight `m_i`, its held
//!   interval's decayed mass inside the window: `Σ m_i (x_i - μ)² / V1`
//!   with `μ = Σ m_i x_i / V1`, `V1 = Σ m_i`; unless `bias`, times
//!   `V1² / (V1² - V2)`, `V2 = Σ m_i²`, the correction for unequal weights
//!   Polars' `ewm_var` applies to its own, null where one row carries all
//!   the weight (`V1² = V2`). `ewm_std`
//!   is its root. On a clock that steps by 1 a row these are Polars'
//!   `ewm_var` and `ewm_std` under `adjust = False`, the form `ewm_mean_by`
//!   is: a stretch's first value is held from before it, so it weighs
//!   `(1 - α)^(n-1)` as `adjust = False`'s first row does. A variance's
//!   sums are merged about their means -- `(V1, μ, M2)` by Chan, Golub and
//!   LeVeque's pairwise update, `M2 = M2_a + M2_b + δ² V1_a V1_b / V1` --
//!   never as a mean square less a squared mean, so a window holding one
//!   value is 0 exactly, and the mean is a compensated pair
//!   (`rheon_core::comp`), so a variance at a level decays with its
//!   history rather than settling on the square of a stalled mean's gap.
//!
//! **Which rows a window holds** follows Polars' `rolling_*_by`: a window is
//! a set of timestamps. `closed = "right"`, the default, is `(T - w, T]`
//! looking back and `(t, t + w]` looking ahead; `"left"`, `"both"` and
//! `"none"` move the ends. Every row at one stamp gets the same backward
//! window, later rows at that stamp included, so under `"right"` and
//! `"both"` a backward output waits for the next distinct stamp; under
//! `"left"` and `"none"` the stamp's own rows are outside, and the output
//! is known as the stamp arrives. A forward window counts a row exactly `w`
//! later under `"right"` and `"both"`.
//!
//! **Clock events** are task 78's: a gap past `gap_cap` or a session change
//! ends every window open across it (partial, under each operator's
//! `partial`); a reset discards them (null, never dropped); a group that
//! falls silent is cut once the stream's clock is `gap_cap` past its last
//! row. A forward `"right"` or `"both"` window whose far edge is the last
//! row before such a cut or reset holds every timestamp it can, and is
//! whole, as its backward mirror is (tasks 160 and 173). Only a clock column
//! cuts a silent group, so the frame runner and the specs refuse a forward
//! window under groups without one (`spec.rs`, `clock_cfg_of`; task 173):
//! on a row-count clock a silent group's clock never moves, and its open
//! windows would hold every later row. Rows leave in input order, each once
//! every window over it has resolved.

use std::cmp::Ordering;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use rheon_core::{ClockAdvance, ClockCfg, ClockState, ClockValue, Stamp, seconds_of_ns};
use serde::{Deserialize, Serialize};

use crate::stream::usable;
use crate::window_quantile::QSide;

mod exact_order;
use exact_order::OrdSide;
mod extrema;
use extrema::XSide;

/// Which way a window looks from its row.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Direction {
    /// The rows at or before the row, less than `window_size` older.
    Backward,
    /// The rows after the row, at most `window_size` later.
    Forward,
}

/// Which ends of a window are in it, as Polars' `closed` says it. The
/// windowed models take it too, `Right` or `Both` (`crate::spec`,
/// docs/PLAN.md task 196).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Closed {
    /// `(T - w, T]` back, `(t, t + w]` ahead: Polars' default.
    #[default]
    Right,
    /// `[T - w, T)` back, `[t, t + w)` ahead.
    Left,
    /// `[T - w, T]` back, `[t, t + w]` ahead.
    Both,
    /// `(T - w, T)` back, `(t, t + w)` ahead.
    #[serde(rename = "none")]
    Neither,
}

impl Closed {
    pub fn name(self) -> &'static str {
        match self {
            Closed::Right => "right",
            Closed::Left => "left",
            Closed::Both => "both",
            Closed::Neither => "none",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "right" => Some(Closed::Right),
            "left" => Some(Closed::Left),
            "both" => Some(Closed::Both),
            "none" => Some(Closed::Neither),
            _ => None,
        }
    }

    /// Whether the end at the row's own stamp is in the window: `T` of a
    /// backward window, `t` of a forward one.
    fn near(self, direction: Direction) -> bool {
        match direction {
            Direction::Backward => matches!(self, Closed::Right | Closed::Both),
            Direction::Forward => matches!(self, Closed::Left | Closed::Both),
        }
    }

    /// Whether the end a window away is in it: `T - w` of a backward window,
    /// `t + w` of a forward one.
    fn far(self, direction: Direction) -> bool {
        match direction {
            Direction::Backward => matches!(self, Closed::Left | Closed::Both),
            Direction::Forward => matches!(self, Closed::Right | Closed::Both),
        }
    }
}

/// What a window cut short before its span passed gives.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Partial {
    /// The value over what the window saw.
    Keep,
    /// Null.
    Null,
    /// The row leaves the output.
    Drop,
}

impl Partial {
    pub fn name(self) -> &'static str {
        match self {
            Partial::Keep => "keep",
            Partial::Null => "null",
            Partial::Drop => "drop",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "keep" => Some(Partial::Keep),
            "null" => Some(Partial::Null),
            "drop" => Some(Partial::Drop),
            _ => None,
        }
    }
}

/// What an operator computes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Stat {
    Mean,
    Sum,
    Rate,
    /// The variance about the mean, under the mean's weights (task 212).
    Var,
    /// The variance's root.
    Std,
    /// The quantile at an operator's level of the values in the window, each
    /// weighed as a sum weighs it (task 272 (d), `window_quantile.rs`).
    Quantile,
    /// A term of moments (task 554, `moments.rs`).
    Moments,
    /// The largest value in the window (task 561, `windows/extrema.rs`).
    Max,
    /// The smallest.
    Min,
    /// Point-mass EW mean (`ewm_mean(..., adjust=True)`): each observation
    /// weighs `lam^|T - t_j|`, matching Polars' `ewm_mean(adjust=True)`.
    PointMean,
    /// Exact windowed quantile (`rolling_quantile`, `windows/exact_order.rs`).
    ExactQuantile,
    /// Exact windowed rank (`rolling_rank`, `windows/exact_order.rs`).
    ExactRank,
}

impl Stat {
    /// Whether a value weighs its held interval's mass, as a mean's does.
    #[inline]
    pub fn weighs_held(self) -> bool {
        matches!(self, Stat::Mean | Stat::Var | Stat::Std)
    }

    /// Whether its sums are a variance's: merged about a compensated mean.
    #[inline]
    pub fn is_var(self) -> bool {
        matches!(self, Stat::Var | Stat::Std)
    }

    /// Whether it is a maximum or a minimum: read from the queue's extrema.
    #[inline]
    pub fn is_extremum(self) -> bool {
        matches!(self, Stat::Max | Stat::Min)
    }

    /// Whether it is an exact windowed quantile or rank (order-statistic tree).
    #[inline]
    pub fn is_exact_order(self) -> bool {
        matches!(self, Stat::ExactQuantile | Stat::ExactRank)
    }

    /// Whether it carries a `max_state_bytes` queue cap.
    #[inline]
    pub fn is_capped(self) -> bool {
        self.is_extremum() || self.is_exact_order()
    }

    /// Whether a queue row keeps the row's own value, not the held one: a
    /// quantile's, extremum's, exact-order's or point-mass mean's.
    #[inline]
    fn keeps_own(self) -> bool {
        matches!(
            self,
            Stat::Quantile
                | Stat::Max
                | Stat::Min
                | Stat::PointMean
                | Stat::ExactQuantile
                | Stat::ExactRank
        )
    }
}

/// An operator, as the formulas name it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OpKind {
    EwmMean,
    RewmMean,
    EwmSum,
    RewmSum,
    EwmRate,
    RewmRate,
    EwmVar,
    EwmStd,
    EwmQuantile,
    RewmQuantile,
    EwmMoments,
    RollingMax,
    RollingMin,
    RrollingMax,
    RrollingMin,
    RollingQuantile,
    RrollingQuantile,
    RollingRank,
    RrollingRank,
    Increment,
    /// A cross-group operator ([`CrossStat::Mean`]): no kernel, the
    /// stream's stamps.
    CrossMean,
    CrossStd,
    CrossRank,
    CrossQuantile,
    CrossRef,
    /// A periodic profile ([`ProfileDef`]): no kernel, a group's buckets.
    EwmProfile,
    TickSign,
    /// The run length of unchanged values (task 564): a row operator.
    RunLength,
    /// The decayed mean lag between two clocks (task 564): a row operator.
    ClockLag,
    /// The decayed slope of that lag on the arrival time (task 564): a row
    /// operator.
    ClockDrift,
    /// A decayed statistic of a time's steps (task 564): a row operator.
    ClockStep,
    /// STL and MSTL behind an embargo ([`StlDef`]): no kernel, a group's
    /// ring of learned rows.
    Stl,
    Mstl,
}

impl OpKind {
    pub fn name(self) -> &'static str {
        match self {
            OpKind::EwmMean => "ewm_mean",
            OpKind::RewmMean => "rewm_mean",
            OpKind::EwmSum => "ewm_sum",
            OpKind::RewmSum => "rewm_sum",
            OpKind::EwmRate => "ewm_rate",
            OpKind::RewmRate => "rewm_rate",
            OpKind::EwmVar => "ewm_var",
            OpKind::EwmStd => "ewm_std",
            OpKind::EwmQuantile => "ewm_quantile",
            OpKind::RewmQuantile => "rewm_quantile",
            OpKind::EwmMoments => "ewm_moments",
            OpKind::RollingMax => "rolling_max",
            OpKind::RollingMin => "rolling_min",
            OpKind::RrollingMax => "rrolling_max",
            OpKind::RrollingMin => "rrolling_min",
            OpKind::RollingQuantile => "rolling_quantile",
            OpKind::RrollingQuantile => "rrolling_quantile",
            OpKind::RollingRank => "rolling_rank",
            OpKind::RrollingRank => "rrolling_rank",
            OpKind::Increment => "increment",
            OpKind::CrossMean => "cross_mean",
            OpKind::CrossStd => "cross_std",
            OpKind::CrossRank => "cross_rank",
            OpKind::CrossQuantile => "cross_quantile",
            OpKind::CrossRef => "cross_ref",
            OpKind::EwmProfile => "ewm_profile",
            OpKind::TickSign => "tick_sign",
            OpKind::RunLength => "run_length",
            OpKind::ClockLag => "clock_lag",
            OpKind::ClockDrift => "clock_drift",
            OpKind::ClockStep => "clock_step",
            OpKind::Stl => "stl",
            OpKind::Mstl => "mstl",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "ewm_mean" => Some(OpKind::EwmMean),
            "rewm_mean" => Some(OpKind::RewmMean),
            "ewm_sum" => Some(OpKind::EwmSum),
            "rewm_sum" => Some(OpKind::RewmSum),
            "ewm_rate" => Some(OpKind::EwmRate),
            "rewm_rate" => Some(OpKind::RewmRate),
            "ewm_var" => Some(OpKind::EwmVar),
            "ewm_std" => Some(OpKind::EwmStd),
            "ewm_quantile" => Some(OpKind::EwmQuantile),
            "rewm_quantile" => Some(OpKind::RewmQuantile),
            "ewm_moments" => Some(OpKind::EwmMoments),
            "rolling_max" => Some(OpKind::RollingMax),
            "rolling_min" => Some(OpKind::RollingMin),
            "rrolling_max" => Some(OpKind::RrollingMax),
            "rrolling_min" => Some(OpKind::RrollingMin),
            "rolling_quantile" => Some(OpKind::RollingQuantile),
            "rrolling_quantile" => Some(OpKind::RrollingQuantile),
            "rolling_rank" => Some(OpKind::RollingRank),
            "rrolling_rank" => Some(OpKind::RrollingRank),
            "increment" => Some(OpKind::Increment),
            "cross_mean" => Some(OpKind::CrossMean),
            "cross_std" => Some(OpKind::CrossStd),
            "cross_rank" => Some(OpKind::CrossRank),
            "cross_quantile" => Some(OpKind::CrossQuantile),
            "cross_ref" => Some(OpKind::CrossRef),
            "ewm_profile" => Some(OpKind::EwmProfile),
            "tick_sign" => Some(OpKind::TickSign),
            "run_length" => Some(OpKind::RunLength),
            "clock_lag" => Some(OpKind::ClockLag),
            "clock_drift" => Some(OpKind::ClockDrift),
            "clock_step" => Some(OpKind::ClockStep),
            "stl" => Some(OpKind::Stl),
            "mstl" => Some(OpKind::Mstl),
            _ => None,
        }
    }

    pub fn direction(self) -> Direction {
        match self {
            OpKind::RewmMean
            | OpKind::RewmSum
            | OpKind::RewmRate
            | OpKind::RewmQuantile
            | OpKind::RrollingMax
            | OpKind::RrollingMin
            | OpKind::RrollingQuantile
            | OpKind::RrollingRank => Direction::Forward,
            _ => Direction::Backward,
        }
    }

    /// A step (`increment`, `tick_sign`): the row before, not a window.
    pub fn is_step(self) -> bool {
        matches!(self, OpKind::Increment | OpKind::TickSign)
    }

    /// Whether it is a row operator: computed one row at a time from the
    /// group's own previous rows, before the window core, which never sees
    /// it -- `increment`, `tick_sign`, `run_length`, `clock_lag`, `clock_drift`
    /// and `clock_step`.
    pub fn is_row(self) -> bool {
        matches!(
            self,
            OpKind::Increment
                | OpKind::TickSign
                | OpKind::RunLength
                | OpKind::ClockLag
                | OpKind::ClockDrift
                | OpKind::ClockStep
        )
    }

    /// The statistic of a window operator; `None` for a row operator.
    pub fn stat(self) -> Option<Stat> {
        match self {
            OpKind::EwmMean | OpKind::RewmMean => Some(Stat::Mean),
            OpKind::EwmSum | OpKind::RewmSum => Some(Stat::Sum),
            OpKind::EwmRate | OpKind::RewmRate => Some(Stat::Rate),
            OpKind::EwmVar => Some(Stat::Var),
            OpKind::EwmStd => Some(Stat::Std),
            OpKind::EwmQuantile | OpKind::RewmQuantile => Some(Stat::Quantile),
            OpKind::EwmMoments => Some(Stat::Moments),
            OpKind::RollingMax | OpKind::RrollingMax => Some(Stat::Max),
            OpKind::RollingMin | OpKind::RrollingMin => Some(Stat::Min),
            OpKind::RollingQuantile | OpKind::RrollingQuantile => Some(Stat::ExactQuantile),
            OpKind::RollingRank | OpKind::RrollingRank => Some(Stat::ExactRank),
            OpKind::Increment
            | OpKind::TickSign
            | OpKind::RunLength
            | OpKind::ClockLag
            | OpKind::ClockDrift
            | OpKind::ClockStep
            | OpKind::CrossMean
            | OpKind::CrossStd
            | OpKind::CrossRank
            | OpKind::CrossQuantile
            | OpKind::CrossRef
            | OpKind::EwmProfile
            | OpKind::Stl
            | OpKind::Mstl => None,
        }
    }

    /// The cross-group statistic, for a cross operator; `None` for every
    /// window operator and `increment`.
    pub fn cross(self) -> Option<CrossStat> {
        match self {
            OpKind::CrossMean => Some(CrossStat::Mean),
            OpKind::CrossStd => Some(CrossStat::Std),
            OpKind::CrossRank => Some(CrossStat::Rank),
            OpKind::CrossQuantile => Some(CrossStat::Quantile),
            OpKind::CrossRef => Some(CrossStat::Ref),
            _ => None,
        }
    }
}

/// A kernel: the queue every operator with these settings shares.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KernelDef {
    pub direction: Direction,
    /// Policy-clock units, above 0; infinite weighs the window evenly.
    pub half_life: f64,
    /// Policy-clock units, finite and above 0; `None` is no cutoff, which
    /// only a backward kernel may have.
    pub window_size: Option<f64>,
    /// The window in integer nanoseconds where it was given as a duration:
    /// on a temporal clock an edge is then decided between integers
    /// (review R2, W1), with no conversion to seconds on the way.
    pub window_ns: Option<i64>,
    pub closed: Closed,
    /// An extremum's `since` event (task 561): the index of the row input
    /// whose value 1.0 (true) starts the kernel's windows over at the row's
    /// stamp. `None`, and then not written, for every other kernel.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub since: Option<usize>,
}

impl KernelDef {
    /// # Errors
    ///
    /// A half-life that is not above 0, a window that is not finite and
    /// above 0 or that a forward kernel lacks.
    pub fn check(&self) -> Result<(), String> {
        if self.half_life.is_nan() || self.half_life <= 0.0 {
            return Err(format!(
                "half_life must be above 0, but it is {}. Set half_life to a positive number of \
                 clock units or a duration, or to inf for no decay.",
                self.half_life
            ));
        }
        if self.half_life < f64::MIN_POSITIVE {
            // A subnormal half-life: `mass·x` underflows to 0 and a rate
            // divides by that mass, so the mean came out 0 and the rate
            // infinite (task 159, W5).
            return Err(format!(
                "half_life must be at least {:.2e}, but it is {:.2e}. Set half_life to at least \
                 {:.2e}.",
                f64::MIN_POSITIVE,
                self.half_life,
                f64::MIN_POSITIVE
            ));
        }
        match (self.direction, self.window_size) {
            (Direction::Forward, None) => {
                return Err(
                    "window_size is not set, but an operator that reads the rows after \
                            each row needs one, since those rows have no end without it. Set \
                            window_size to a finite positive number of clock units or a \
                            duration."
                        .into(),
                );
            }
            (_, Some(w)) if !w.is_finite() || w <= 0.0 => {
                return Err(format!(
                    "window_size must be finite and above 0, but it is {w}. Set window_size to a \
                     finite positive number of clock units or a duration."
                ));
            }
            _ => {}
        }
        if self.since.is_some() && self.direction == Direction::Forward {
            return Err(SINCE_AHEAD.into());
        }
        // The two forms of one window agree (review R4, A4): membership is
        // decided on the integer, the far edge and `complete` on the number.
        if let (Some(w), Some(ns)) = (self.window_size, self.window_ns)
            && seconds_of_ns(i128::from(ns)) != w
        {
            return Err(format!(
                "window_ns {ns} is not window_size {w} in nanoseconds"
            ));
        }
        Ok(())
    }

    /// Whether a backward kernel's rows wait for the next distinct stamp:
    /// under `"right"` and `"both"`, whose stamp's rows are in its windows,
    /// and under a `since` event, which any row of a stamp may carry.
    #[inline]
    pub(crate) fn waits(&self) -> bool {
        self.direction == Direction::Backward
            && (self.closed.near(Direction::Backward) || self.since.is_some())
    }

    /// `λ^d`, the discount across `d` units of policy time.
    #[inline]
    pub(crate) fn discount(&self, d: f64) -> f64 {
        if self.half_life.is_infinite() || d == 0.0 {
            // No distance discounts nothing: `exp2(-0)` is 1 exactly, for
            // the half-life above 0 every kernel has (`check`), and every
            // read of a backward window at its newest row is one.
            1.0
        } else {
            // `exp2(-x)`, as `Decay::factor` spells it, for the same reason.
            (-(d / self.half_life)).exp2()
        }
    }

    /// [`Self::discount`] through `memo`, which keeps the last distance it
    /// was asked for and its discount: a clock with a regular step asks the
    /// same distance row after row, and the operators of a kernel ask the same
    /// one on a row. The same function of the same bits, so the same bits.
    #[inline]
    pub(crate) fn discount_in(&self, d: f64, memo: &mut DecayMemo) -> f64 {
        if self.half_life.is_infinite() || d == 0.0 {
            return 1.0;
        }
        let (bits, v) = memo.discount;
        if bits == d.to_bits() {
            return v;
        }
        let v = (-(d / self.half_life)).exp2();
        memo.discount = (d.to_bits(), v);
        v
    }

    /// [`Self::mass`] through `memo`, as [`Self::discount_in`].
    #[inline]
    fn mass_in(&self, t: f64, memo: &mut DecayMemo) -> f64 {
        if t <= 0.0 {
            return 0.0;
        }
        let bits = t.to_bits();
        if memo.mass.0 == bits {
            return memo.mass.1;
        }
        let v = if self.half_life >= held_mass::MASS_SCALE_FROM {
            if self.half_life.is_infinite() {
                return t;
            }
            self.mass_huge(t)
        } else {
            self.mass_normal(t)
        };
        memo.mass = (bits, v);
        v
    }

    /// [`Self::mass`] of the distance from a row to the window's far edge,
    /// through the four entries of `memo.cut`, each the last distance whose
    /// low two bits it is of: the same function of the same bits, so the same
    /// bits.
    #[inline]
    fn mass_cut_in(&self, t: f64, memo: &mut DecayMemo) -> f64 {
        if t <= 0.0 {
            return 0.0;
        }
        let bits = t.to_bits();
        let slot = &mut memo.cut.get_or_insert_with(Default::default)[(bits & 3) as usize];
        if slot.0 == bits {
            return slot.1;
        }
        let v = if self.half_life >= held_mass::MASS_SCALE_FROM {
            if self.half_life.is_infinite() {
                return t;
            }
            self.mass_huge(t)
        } else {
            self.mass_normal(t)
        };
        *slot = (bits, v);
        v
    }

    /// [`Self::mass_in`] on the memo's second slot, for a second distance
    /// asked beside the first.
    #[inline]
    fn mass_far_in(&self, t: f64, memo: &mut DecayMemo) -> f64 {
        if t <= 0.0 {
            return 0.0;
        }
        let bits = t.to_bits();
        if memo.mass_far.0 == bits {
            return memo.mass_far.1;
        }
        let v = if self.half_life >= held_mass::MASS_SCALE_FROM {
            if self.half_life.is_infinite() {
                return t;
            }
            self.mass_huge(t)
        } else {
            self.mass_normal(t)
        };
        memo.mass_far = (bits, v);
        v
    }

    /// [`Self::discount`] through one slot of a [`DecayMemo`].
    #[inline]
    fn discount_slot(&self, d: f64, slot: &mut (u64, f64)) -> f64 {
        if self.half_life.is_infinite() || d == 0.0 {
            return 1.0;
        }
        if slot.0 == d.to_bits() {
            return slot.1;
        }
        let v = (-(d / self.half_life)).exp2();
        *slot = (d.to_bits(), v);
        v
    }

    /// [`Self::mass`] through one slot of a [`DecayMemo`].
    #[inline]
    fn mass_slot(&self, t: f64, slot: &mut (u64, f64)) -> f64 {
        if t <= 0.0 {
            return 0.0;
        }
        let bits = t.to_bits();
        if slot.0 == bits {
            return slot.1;
        }
        let v = if self.half_life >= held_mass::MASS_SCALE_FROM {
            if self.half_life.is_infinite() {
                return t;
            }
            self.mass_huge(t)
        } else {
            self.mass_normal(t)
        };
        *slot = (bits, v);
        v
    }

    /// `∫_0^T λ^s ds`: the decayed mass of `T` units of time.
    #[inline]
    fn mass(&self, t: f64) -> f64 {
        if t <= 0.0 {
            0.0
        } else if self.half_life >= held_mass::MASS_SCALE_FROM {
            self.mass_extreme(t)
        } else {
            self.mass_normal(t)
        }
    }

    /// [`Self::mass`] of a distance above 0 under a finite half-life below `2^511`.
    #[inline]
    fn mass_normal(&self, t: f64) -> f64 {
        // `1 − 2^(−t/h)` cancels where `t ≪ h`: 6e-5 of the mass at
        // `h/t = 1e12`, all of it from 1e17, every mean null (task 159,
        // W2). `exp_m1` keeps the small difference whole; `discount`
        // itself has nothing to lose.
        let h = self.half_life;
        let arg = t * std::f64::consts::LN_2 / h;
        if arg >= held_mass::MASS_LINEAR_BELOW {
            -(h / std::f64::consts::LN_2) * (-arg).exp_m1()
        } else {
            t
        }
    }
}

/// Why `since` is refused looking ahead (task 561).
pub const SINCE_AHEAD: &str = "since cannot be combined with reverse=True yet: a window \
     looking ahead would end at the next event, and whether such a window is whole or partial \
     has no rule yet. Leave out since, or read the rows at or before each row";

/// A queue's last discount and last mass, each with the bits of the distance
/// it is of. Never written to a state and never compared; a queue read back
/// starts with none (0 bits is a distance no call reaches the memo with: a
/// distance of 0 has its own answer first).
#[derive(Debug, Clone, Default)]
pub(crate) struct DecayMemo {
    discount: (u64, f64),
    mass: (u64, f64),
    /// The mass of the interval a row holds its value over, which takes the
    /// discount's own `exp2` from `HELD_MASS_FROM` and so is not the cut's
    /// mass of the same distance (`mass`).
    held: (u64, f64),
    /// The mass at the far end of an interval read in the window's cut.
    mass_far: (u64, f64),
    /// The discount from the near end of the interval a window's cut reads
    /// to the reading row: a slot of its own, since a read asks for it
    /// between the discounts across its queue's anchors (`discount`), once
    /// for each operator of the kernel, and one slot would hold neither.
    disc_cut: (u64, f64),
    /// A forward kernel's close, which a backward kernel's queue never needs.
    fwd: Option<Box<FwdMemo>>,
    /// A windowed backward kernel's cut: the mass of the distance from a
    /// row to the window's far edge, which is the window up to the rounding
    /// of `tau - w` against `tau` (a few values, never the one before), made
    /// where a queue first has it.
    cut: Option<Box<[(u64, f64); 4]>>,
}

/// A forward kernel's close: the mass from the row to its open row, and the
/// discounts of the window's sums and of its open row, each to the row the
/// window is of, and a rate's mass of the window's span. A close asks each
/// operator of a kernel for the same distances, and a regular clock asks them
/// again at the next row.
#[derive(Debug, Clone, Default)]
pub(crate) struct FwdMemo {
    mass_open: (u64, f64),
    disc_seg: (u64, f64),
    disc_open: (u64, f64),
    /// A mean's open row, discounted to its sums' anchor, not to the row as
    /// a sum's is (`disc_open`): a slot of its own, so a mean and a sum of
    /// one kernel each find their last distance.
    disc_mean: (u64, f64),
    mass_span: (u64, f64),
}

impl DecayMemo {
    #[inline]
    fn fwd(&mut self) -> &mut FwdMemo {
        self.fwd.get_or_insert_with(Default::default)
    }
}

impl PartialEq for DecayMemo {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

/// One window operator: its kernel, statistic and input.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OpDef {
    /// Index into the kernels.
    pub kernel: usize,
    pub stat: Stat,
    /// Index into a row's values.
    pub input: usize,
    /// Null with fewer rows carrying a value in the window.
    pub min_samples: u32,
    pub partial: Partial,
    /// A variance's: the variance of the weights as they are, uncorrected
    /// for their count (Polars' `bias`); read by no other statistic.
    pub bias: bool,
    /// A quantile's level, in (0, 1); `None` for every other statistic, and
    /// then not written, so a state without a quantile keeps its bytes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quantile: Option<f64>,
    /// An extremum's or exact-order operator's bound on its queue's bytes a
    /// group; `None`, and then not written, for every other operator.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_state_bytes: Option<u64>,
    /// An exact quantile's interpolation (`rolling_quantile`); `None`, and then
    /// not written, for every other operator.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub interpolation: Option<Interp>,
    /// An exact rank's tie rule (`rolling_rank`); `None`, and then not written,
    /// for every other operator.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rank_method: Option<RankMethod>,
}

/// One row as the core reads it: its group, clock and session, every
/// operator input, and whether a `like=` spec would learn from it.
#[derive(Debug)]
pub struct RowIn<'a> {
    /// From [`Windows::group`].
    pub group: usize,
    /// `None` on a row-count clock.
    pub clock: Option<ClockValue>,
    pub session: Option<u64>,
    /// Per operator input; not [`usable`] is missing.
    pub values: &'a [f64],
    /// Whether the spec behind `like=` would learn from this row. Its
    /// forward windows are null when it would not: a model never learns the
    /// target of a row it skips (docs/PLAN.md task 104). The row still
    /// steps the clocks and still counts in other rows' windows.
    pub accept: bool,
    /// Per list a cross operator reads its neighbors from: the row's keys,
    /// as text (`None` for a null list). Empty with no such operator.
    pub lists: &'a [Option<&'a [Option<String>]>],
}

/// What the clocks make of a row before it is taken ([`Windows::peek`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Peek {
    /// A new start by the policy's word: a step back past
    /// `restart_after_step_back`, or a new session, on the stream's clock
    /// or the group's.
    NewStart,
    /// A step forward on the stream's clock, or its first row.
    Forward,
    /// The same stamp as the last row the stream read.
    SameStamp,
    /// A step back that the policy skips as stale (`on_step_back = "skip"`).
    StepBack,
}

/// Which kind of operator a state cap or warning is about: its `op` is an
/// index among the operators of that kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StateOf {
    /// A cross-group operator, holding the rows of its open stamp.
    Cross,
    /// A profile operator, holding a group's buckets.
    Profile,
}

/// Why a row was refused. The caller names the row in the input.
#[derive(Debug, Clone, PartialEq)]
pub enum Refusal {
    /// The clock stepped back `back` clock units, from `prev` to `now`,
    /// where the policy refuses it: any step with no `restart_after_step_back`,
    /// and one no larger than it (carried here) with one. The two values let
    /// a caller state a temporal step exactly, in nanoseconds.
    Backwards {
        seq: u64,
        back: f64,
        prev: Option<ClockValue>,
        now: Option<ClockValue>,
        min_backwards_jump: Option<f64>,
    },
    /// The cross-group operator or profile operator `op` (its index among
    /// those of its kind) would hold `bytes` for its open stamp or its group
    /// with this row, past its `cap`; the row is not taken.
    StateCap {
        seq: u64,
        of: StateOf,
        op: usize,
        bytes: u64,
        cap: u64,
    },
    /// The reference read `op` (its index among the cross operators) picks
    /// this row at a stamp where it picked the row `first` already.
    CrossRefTwice { seq: u64, op: usize, first: u64 },
    /// A row took a window's queue past its extrema's `max_state_bytes` a
    /// group (task 561): the row is taken, the run is refused, and the core
    /// is not to be fed again. `rows` held at `row_bytes` each.
    OverCap {
        seq: u64,
        rows: usize,
        row_bytes: usize,
        cap: u64,
    },
    /// A number clock's distance from its stretch's first row, `origin`, to
    /// this row, `now`, is past the largest double, each step being finite:
    /// the policy time a window's edges and masses are read from has no
    /// value (review 2026-11-23, WIN-9). As [`Refusal::OverCap`], the row is
    /// taken, the run is refused, and the core is not to be fed again.
    Span {
        seq: u64,
        origin: ClockValue,
        now: ClockValue,
    },
}

/// A segment's sums and how two are joined, in a module of their own.
mod acc;
use acc::{Acc, then, then_by, then_by_point, then_point};

/// Per row, per operator, in a stack's `vals`: where the held interval
/// starts, the value held (NaN for none), and the row's own sums.
const ROW: usize = 5;
/// Per row, per operator, in a stack's `sums`: the partial sums.
const SUM: usize = 3;
/// [`ROW`] and [`SUM`] in a queue holding a variance: its sums are six wide
/// ([`Acc`]), and every operator of the kernel takes the one width.
const ROW_VAR: usize = 8;
const SUM_VAR: usize = 6;

/// A row's width per operator in a queue's `vals`: [`ROW_VAR`] in a queue
/// holding a variance (`W`), else [`ROW`]. The queue code is compiled once
/// for each width, so a queue without a variance runs the code it ran
/// before there was one.
#[inline(always)]
const fn row_w<const W: bool>() -> usize {
    if W { ROW_VAR } else { ROW }
}

/// [`row_w`] for the sums: [`SUM_VAR`] or [`SUM`].
#[inline(always)]
const fn sum_w<const W: bool>() -> usize {
    if W { SUM_VAR } else { SUM }
}

/// Rows with their values and partial sums in flat arenas, so a row costs
/// no allocation: one value and one accumulator per operator, side by side.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
struct Stack {
    n: usize,
    /// Whether the kernel holds a variance: its arenas' strides are
    /// [`ROW_VAR`] and [`SUM_VAR`], not [`ROW`] and [`SUM`].
    wide: bool,
    seq: Vec<u64>,
    tau: Vec<f64>,
    /// The row's raw clock in integer nanoseconds, 0 without a temporal
    /// clock: what a window's edge is decided from (review R2, W1).
    off: Vec<i64>,
    end: Vec<f64>,
    /// Stride `row_w() * n`.
    vals: Vec<f64>,
    /// Stride `sum_w() * n`, anchored at `at`.
    sums: Vec<f64>,
    at: Vec<f64>,
}

/// `v.extend_from_slice(s)` with the lengths a row's values and sums have for
/// one to four operators written out (`ROW`, `SUM` and their wide forms, times
/// the operators): a copy of a length the compiler knows is a few moves in
/// place, where the general copy is a call to `memmove`, and a row of a window
/// core is two of them.
#[inline(always)]
fn append(v: &mut Vec<f64>, s: &[f64]) {
    macro_rules! fixed {
        ($($n:literal),*) => {
            match s.len() {
                $($n => v.extend_from_slice(&s[..$n]),)*
                _ => v.extend_from_slice(s),
            }
        };
    }
    fixed!(3, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24, 32)
}

impl Stack {
    fn new(n: usize, wide: bool) -> Self {
        Stack {
            n,
            wide,
            ..Default::default()
        }
    }

    #[inline]
    fn row_w(&self) -> usize {
        if self.wide { ROW_VAR } else { ROW }
    }

    #[inline]
    fn sum_w(&self) -> usize {
        if self.wide { SUM_VAR } else { SUM }
    }

    #[inline]
    fn len(&self) -> usize {
        self.seq.len()
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.seq.is_empty()
    }

    fn clear(&mut self) {
        self.seq.clear();
        self.tau.clear();
        self.off.clear();
        self.end.clear();
        self.vals.clear();
        self.sums.clear();
        self.at.clear();
    }

    #[allow(clippy::too_many_arguments)]
    fn push(&mut self, seq: u64, tau: f64, off: i64, end: f64, row: &[f64], sums: &[f64], at: f64) {
        self.seq.push(seq);
        self.tau.push(tau);
        self.off.push(off);
        self.end.push(end);
        append(&mut self.vals, row);
        append(&mut self.sums, sums);
        self.at.push(at);
    }

    /// Room for `more` rows, in every arena.
    fn reserve(&mut self, more: usize) {
        self.seq.reserve(more);
        self.tau.reserve(more);
        self.off.reserve(more);
        self.end.reserve(more);
        self.vals.reserve(more * self.row_w() * self.n);
        self.sums.reserve(more * self.sum_w() * self.n);
        self.at.reserve(more);
    }

    fn pop(&mut self) {
        let i = self.len() - 1;
        self.seq.truncate(i);
        self.tau.truncate(i);
        self.off.truncate(i);
        self.end.truncate(i);
        self.vals.truncate(i * self.row_w() * self.n);
        self.sums.truncate(i * self.sum_w() * self.n);
        self.at.truncate(i);
    }

    #[inline]
    fn start<const W: bool>(&self, i: usize, j: usize) -> f64 {
        self.vals[(i * self.n + j) * row_w::<W>()]
    }

    fn x_of(&self, wide: bool, i: usize, j: usize) -> f64 {
        if wide {
            self.x::<true>(i, j)
        } else {
            self.x::<false>(i, j)
        }
    }

    #[inline]
    fn x<const W: bool>(&self, i: usize, j: usize) -> f64 {
        self.vals[(i * self.n + j) * row_w::<W>() + 1]
    }

    #[inline]
    fn own<const W: bool>(&self, i: usize, j: usize) -> Acc {
        let b = (i * self.n + j) * row_w::<W>() + 2;
        Acc::read(&self.vals[b..], W)
    }

    #[inline]
    fn sum<const W: bool>(&self, i: usize, j: usize) -> Acc {
        let b = (i * self.n + j) * sum_w::<W>();
        Acc::read(&self.sums[b..], W)
    }

    /// Whether every arena is as long as the rows say, at `n` operators of
    /// this width: what the accessors above index by ([`Windows::check`]).
    fn has_width(&self, n: usize, wide: bool) -> bool {
        let len = self.seq.len();
        self.n == n
            && self.wide == wide
            && [
                self.tau.len(),
                self.off.len(),
                self.end.len(),
                self.at.len(),
            ]
            .iter()
            .all(|&l| l == len)
            && len.checked_mul(self.row_w() * n) == Some(self.vals.len())
            && len.checked_mul(self.sum_w() * n) == Some(self.sums.len())
    }
}

/// Write `a` into `out`'s slot for operator `j`, at a queue's width.
#[inline]
fn put<const W: bool>(out: &mut [f64], j: usize, a: Acc) {
    a.write(&mut out[j * sum_w::<W>()..], W);
}

/// The fold of `back`'s rows `from..=upto`, oldest first, as a mean's cut
/// sums them ([`Queue::sum_after_op`]), with the rows' `seq`s at both ends:
/// `back` only grows between its turnovers, and a row's `seq` is never
/// reused, so the two say the fold is still of the rows it was made from.
#[derive(Debug, Clone, Copy)]
struct Tail {
    from: usize,
    from_seq: u64,
    upto: usize,
    upto_seq: u64,
    acc: Acc,
    at: f64,
}

/// What a queue remembers between reads to keep a read O(1) amortized
/// (docs/PLAN.md task 270, M-2): per operator, the fold of `back`'s rows
/// past the operator's cut, extended by the rows pushed since rather than
/// summed again, and a `seq` below which no row the queue holds carries the
/// operator's value, so the search for its oldest valued row starts there.
/// Neither changes a bit: the fold is the same `then`s in the same order,
/// and the search finds the same row. Never written to a state and never
/// compared: a queue read back from a file starts without one and rebuilds
/// it on its first read, as a queue does on its first window.
#[derive(Debug, Clone, Default)]
struct Memo {
    tails: Vec<Option<Tail>>,
    /// Per operator: every row the queue holds with a lower `seq` carries
    /// no value of it. Rows leave oldest first and join with higher `seq`s,
    /// so it stays true until it is moved.
    valued_from: Vec<u64>,
}

impl PartialEq for Memo {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

/// The two-stack queue. `front` holds the older rows, the oldest on top,
/// each beside the sums from itself through the newest row in `front`;
/// `back` holds the newer rows in order, each beside the sums from the
/// oldest row in `back` through itself. The window's sums are `front`'s
/// top then `back`'s last. Reaching the oldest row with `front` empty first
/// turns `back` over into it, so each row is moved once: O(1) a row on
/// average, and exact. The partial sums also give the window less its
/// oldest rows without a subtraction, which is what lets a held value be
/// cut at the window's edge without cancellation.
///
/// A backward kernel with no window evicts nothing, so it keeps only
/// `total`, the running sums.
///
/// A quantile (task 272 (d)) keeps sketches beside the stacks, in `quant`:
/// its rows' sums give only its count of values, for `min_samples`
/// (`window_quantile.rs` has the rest).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
struct Queue {
    n: usize,
    /// Per operator: whether its sums are a variance's, merged rather than
    /// added ([`merge`]). A queue holding one is wide.
    var: Vec<bool>,
    front: Stack,
    back: Stack,
    /// Stride `sum_w() * n`; empty until the first row.
    total: Vec<f64>,
    total_at: f64,
    /// Per operator, a quantile's sketches, `None` for any other; empty for
    /// a kernel with no quantile, and then not written, so its state keeps
    /// the bytes it had before there was one.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    quant: Vec<Option<QSide>>,
    /// Per operator, an extremum's part (`windows/extrema.rs`), `None` for
    /// any other; empty, and then not written, for a kernel with none.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    ext: Vec<Option<XSide>>,
    /// A kernel with a `since` event: the raw clock of its last event's
    /// stamp since the stretch began, for whether a window is whole.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    since_off: Option<i64>,
    /// Per operator: whether its sums are a point-mass mean's (`adjust=True`);
    /// empty, and then not written, for a kernel with none.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    point: Vec<bool>,
    /// Per operator, an exact-order operator's multiset (`windows/exact_order.rs`),
    /// `None` for any other; empty, and then not written, for a kernel with none.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    ord: Vec<Option<OrdSide>>,
    #[serde(skip)]
    memo: Memo,
    #[serde(skip)]
    decay: DecayMemo,
}

impl Queue {
    fn new(
        var: Vec<bool>,
        point: &[bool],
        quant: &[bool],
        ext: &[Option<bool>],
        later: bool,
        ord: &[bool],
    ) -> Self {
        let (n, wide) = (var.len(), var.contains(&true));
        Queue {
            n,
            var,
            front: Stack::new(n, wide),
            back: Stack::new(n, wide),
            total: Vec::new(),
            total_at: 0.0,
            quant: if quant.contains(&true) {
                quant.iter().map(|&q| q.then(QSide::default)).collect()
            } else {
                Vec::new()
            },
            ext: if ext.iter().any(Option::is_some) {
                ext.iter()
                    .map(|m| m.map(|max| XSide::new(max, later)))
                    .collect()
            } else {
                Vec::new()
            },
            since_off: None,
            point: if point.contains(&true) {
                point.to_vec()
            } else {
                Vec::new()
            },
            ord: if ord.contains(&true) {
                ord.iter().map(|&o| o.then(OrdSide::default)).collect()
            } else {
                Vec::new()
            },
            memo: Memo::default(),
            decay: DecayMemo::default(),
        }
    }

    #[inline]
    fn is_point(&self, j: usize) -> bool {
        !self.point.is_empty() && self.point[j]
    }

    #[inline]
    fn has_ord(&self) -> bool {
        !self.ord.is_empty()
    }

    #[inline]
    fn ord(&self, j: usize) -> &OrdSide {
        self.ord[j]
            .as_ref()
            .expect("an exact-order operator keeps its tree")
    }

    /// The `k_from_newest`-th newest row's input value for operator `j`
    /// (`0` = newest row in the queue).
    fn newest_x<const W: bool>(&self, k_from_newest: usize, j: usize) -> f64 {
        if k_from_newest < self.back.len() {
            self.back.x::<W>(self.back.len() - 1 - k_from_newest, j)
        } else {
            self.front.x::<W>(k_from_newest - self.back.len(), j)
        }
    }

    /// Look up the input value of row `seq` for operator `j` in `front` or `back`.
    fn x_by_seq<const W: bool>(&self, seq: u64, j: usize) -> f64 {
        if let Ok(i) = self.front.seq.binary_search_by(|s| seq.cmp(s)) {
            return self.front.x::<W>(i, j);
        }
        if let Ok(i) = self.back.seq.binary_search(&seq) {
            return self.back.x::<W>(i, j);
        }
        f64::NAN
    }

    /// Operator `j`'s extremum part; every extremum has one (`has_width`).
    #[inline]
    fn ext(&self, j: usize) -> &XSide {
        self.ext[j].as_ref().expect("an extremum keeps its part")
    }

    /// The quantile at `p` of operator `j`'s window, with `open`, a forward
    /// kernel's open row as its time and value, counted too; `None` with no
    /// weight.
    fn quantile(&self, k: &KernelDef, j: usize, p: f64, open: Option<(f64, f64)>) -> Option<f64> {
        self.quant.get(j)?.as_ref()?.read(k, p, open)
    }

    /// The buckets every quantile's sketches hold.
    fn quantile_buckets(&self) -> usize {
        self.quant.iter().flatten().map(QSide::buckets).sum()
    }

    #[inline]
    fn wide(&self) -> bool {
        self.front.wide
    }

    /// A row's width per operator in this queue's arenas.
    #[inline]
    fn row_w(&self) -> usize {
        self.front.row_w()
    }

    #[inline]
    fn sum_w(&self) -> usize {
        self.front.sum_w()
    }

    /// [`Queue::push`] compiled for this queue's width.
    #[allow(clippy::too_many_arguments)]
    fn push_row(
        &mut self,
        k: &KernelDef,
        seq: u64,
        tau: f64,
        off: i64,
        end: f64,
        row: &[f64],
        scratch: &mut [f64],
    ) {
        if self.wide() {
            self.push::<true>(k, seq, tau, off, end, row, scratch);
        } else {
            self.push::<false>(k, seq, tau, off, end, row, scratch);
        }
    }

    /// Push a row: `row` is its `row_w() * n` values, `scratch` a buffer of
    /// `sum_w() * n` for the sums.
    #[allow(clippy::too_many_arguments)]
    fn push<const W: bool>(
        &mut self,
        k: &KernelDef,
        seq: u64,
        tau: f64,
        off: i64,
        end: f64,
        row: &[f64],
        scratch: &mut [f64],
    ) {
        let (n, rw) = (self.n, row_w::<W>());
        // The buffers are sized for the widest kernel; this one's part.
        let scratch = &mut scratch[..sum_w::<W>() * n];
        let own = |j: usize| Acc::read(&row[j * rw + 2..], W);
        if k.window_size.is_some() {
            let mut at = tau;
            match self.back.len().checked_sub(1) {
                None => {
                    for j in 0..n {
                        put::<W>(scratch, j, own(j));
                    }
                }
                Some(last) => {
                    let last_at = self.back.at[last];
                    let f = k.discount_in(tau - last_at, &mut self.decay);
                    for j in 0..n {
                        let (a, a_at) = if self.is_point(j) {
                            then_by_point(k, f, self.back.sum::<W>(last, j), last_at, own(j), tau)
                        } else {
                            then_by(
                                k,
                                f,
                                self.back.sum::<W>(last, j),
                                last_at,
                                own(j),
                                tau,
                                W && self.var[j],
                            )
                        };
                        put::<W>(scratch, j, a);
                        at = a_at;
                    }
                }
            }
            self.back.push(seq, tau, off, end, row, scratch, at);
        } else if self.total.is_empty() {
            self.total.resize(sum_w::<W>() * n, 0.0);
            for j in 0..n {
                put::<W>(&mut self.total, j, own(j));
            }
            self.total_at = tau;
        } else {
            let mut at = tau;
            let sw = sum_w::<W>();
            let f = k.discount_in(tau - self.total_at, &mut self.decay);
            for j in 0..n {
                let t = Acc::read(&self.total[j * sw..], W);
                let (a, a_at) = if self.is_point(j) {
                    then_by_point(k, f, t, self.total_at, own(j), tau)
                } else {
                    then_by(k, f, t, self.total_at, own(j), tau, W && self.var[j])
                };
                put::<W>(&mut self.total, j, a);
                at = a_at;
            }
            self.total_at = at;
        }
        // An extremum's value joins its part: the row's `x`, its own value,
        // as a quantile's; under a `since` event the current stamp's apart.
        let (windowed, split) = (k.window_size.is_some(), k.since.is_some());
        for (j, x) in self.ext.iter_mut().enumerate() {
            if let Some(x) = x {
                x.push(windowed, split, row[j * rw + 1]);
            }
        }
        for (j, o) in self.ord.iter_mut().enumerate() {
            if let Some(o) = o {
                o.insert(row[j * rw + 1]);
            }
        }
        // A quantile's value joins its sketch: the row's `x`, its own value
        // in either direction (`fill_forward`).
        for (j, q) in self.quant.iter_mut().enumerate() {
            if let Some(q) = q {
                let x = row[j * rw + 1];
                if !x.is_nan() {
                    q.push(k, tau, x);
                }
            }
        }
    }

    /// Turn `back` over into `front` when `front` is empty, so the oldest
    /// row is `front`'s top.
    fn settle<const W: bool>(&mut self, k: &KernelDef, scratch: &mut [f64]) {
        if !self.front.is_empty() {
            return;
        }
        if !self.back.is_empty() {
            let back = &self.back;
            for (j, q) in self.quant.iter_mut().enumerate() {
                if let Some(q) = q {
                    q.turn_over(k, &back.tau, |i| back.x::<W>(i, j));
                }
            }
            for (j, x) in self.ext.iter_mut().enumerate() {
                if let Some(x) = x {
                    x.turn_over(back.len(), |i| back.x::<W>(i, j));
                }
            }
        }
        let (n, rw) = (self.n, row_w::<W>());
        let scratch = &mut scratch[..sum_w::<W>() * n];
        // Every row of `back` moves, the newest first: `front` takes room for
        // them at once and `back` is emptied once, after the last.
        let moved = self.back.len();
        self.front.reserve(moved);
        for i in (0..moved).rev() {
            let (seq, tau, off, end) = (
                self.back.seq[i],
                self.back.tau[i],
                self.back.off[i],
                self.back.end[i],
            );
            let row_base = i * rw * n;
            let mut at = tau;
            match self.front.len().checked_sub(1) {
                None => {
                    for j in 0..n {
                        put::<W>(scratch, j, self.back.own::<W>(i, j));
                    }
                }
                Some(newer) => {
                    let newer_at = self.front.at[newer];
                    let f = k.discount_in(newer_at - tau, &mut self.decay);
                    for j in 0..n {
                        let (a, a_at) = if self.is_point(j) {
                            then_by_point(
                                k,
                                f,
                                self.back.own::<W>(i, j),
                                tau,
                                self.front.sum::<W>(newer, j),
                                newer_at,
                            )
                        } else {
                            then_by(
                                k,
                                f,
                                self.back.own::<W>(i, j),
                                tau,
                                self.front.sum::<W>(newer, j),
                                newer_at,
                                W && self.var[j],
                            )
                        };
                        put::<W>(scratch, j, a);
                        at = a_at;
                    }
                }
            }
            let (front, back) = (&mut self.front, &self.back);
            front.push(
                seq,
                tau,
                off,
                end,
                &back.vals[row_base..row_base + rw * n],
                scratch,
                at,
            );
        }
        self.back.clear();
    }

    /// Pop the rows `stale` says have left the window, oldest first.
    fn evict<const W: bool>(
        &mut self,
        k: &KernelDef,
        scratch: &mut [f64],
        stale: impl Fn(u64, f64, i64) -> bool,
    ) {
        loop {
            self.settle::<W>(k, scratch);
            let Some(top) = self.front.len().checked_sub(1) else {
                return;
            };
            if !stale(
                self.front.seq[top],
                self.front.tau[top],
                self.front.off[top],
            ) {
                return;
            }
            for (j, q) in self.quant.iter_mut().enumerate() {
                if let Some(q) = q {
                    let x = self.front.x::<W>(top, j);
                    q.pop(top, x);
                    // Looking ahead, the rows left are seen from the one that
                    // went: made again from the oldest of them once that is
                    // too far back to read them (WIN-2). Rows with no value
                    // weigh nothing either way, and are not made again.
                    let front = &self.front;
                    if !x.is_nan()
                        && k.direction == Direction::Forward
                        && q.faded()
                        && (0..top).any(|i| !front.x::<W>(i, j).is_nan())
                    {
                        let tau: Vec<f64> = (0..top).rev().map(|i| front.tau[i]).collect();
                        q.rebuild_front(k, &tau, |i| front.x::<W>(top - 1 - i, j));
                    }
                }
            }
            self.ext.iter_mut().flatten().for_each(XSide::pop);
            for (j, o) in self.ord.iter_mut().enumerate() {
                if let Some(o) = o {
                    o.remove(self.front.x::<W>(top, j));
                }
            }
            self.front.pop();
        }
    }

    /// The window's sums for operator `j`, with their anchor.
    fn sum_op<const W: bool>(&mut self, k: &KernelDef, j: usize) -> Option<(Acc, f64)> {
        if k.window_size.is_none() {
            if self.total.is_empty() {
                return None;
            }
            return Some((Acc::read(&self.total[j * sum_w::<W>()..], W), self.total_at));
        }
        let f = self
            .front
            .len()
            .checked_sub(1)
            .map(|t| (self.front.sum::<W>(t, j), self.front.at[t]));
        let b = self
            .back
            .len()
            .checked_sub(1)
            .map(|l| (self.back.sum::<W>(l, j), self.back.at[l]));
        match (f, b) {
            (Some((a, a_at)), Some((b, b_at))) => Some(if self.is_point(j) {
                then_point(k, &mut self.decay, a, a_at, b, b_at)
            } else {
                then(k, &mut self.decay, a, a_at, b, b_at, W && self.var[j])
            }),
            (Some(x), None) | (None, Some(x)) => Some(x),
            (None, None) => None,
        }
    }

    /// The window's sums for operator `j` less its `p + 1` oldest rows.
    /// Where those reach into `back`, whose sums are prefixes anchored at
    /// its oldest row, the rest is `back`'s rows past them folded oldest
    /// first. That is every row of a stream's first window, before anything
    /// has left it and turned `back` over: a mean cuts its first row's held
    /// interval at the window's edge on each of them. The fold is kept
    /// between reads and extended by the rows pushed since ([`Memo`]), so a
    /// read is O(1) amortized, not O(rows in `back`); its `then`s are the
    /// ones summing the rows afresh takes, in the same order, so its bits
    /// are too (docs/PLAN.md task 270, M-2).
    fn sum_after_op<const W: bool>(
        &mut self,
        k: &KernelDef,
        p: usize,
        j: usize,
    ) -> Option<(Acc, f64)> {
        let var = W && self.var[j];
        let n = self.front.len();
        if p + 1 < n {
            let i = n - 2 - p;
            let a = (self.front.sum::<W>(i, j), self.front.at[i]);
            return Some(match self.back.len().checked_sub(1) {
                Some(l) => then(
                    k,
                    &mut self.decay,
                    a.0,
                    a.1,
                    self.back.sum::<W>(l, j),
                    self.back.at[l],
                    var,
                ),
                None => a,
            });
        }
        let from = p + 1 - n;
        let len = self.back.len();
        if from >= len {
            return None;
        }
        let back = &self.back;
        if self.memo.tails.len() != self.n {
            self.memo.tails = vec![None; self.n];
        }
        let kept = self.memo.tails[j].filter(|t| {
            t.from == from
                && t.upto < len
                && back.seq[from] == t.from_seq
                && back.seq[t.upto] == t.upto_seq
        });
        let (mut acc, mut at, next) = match kept {
            Some(t) => (t.acc, t.at, t.upto + 1),
            None => (back.own::<W>(from, j), back.tau[from], from + 1),
        };
        for i in next..len {
            (acc, at) = then(
                k,
                &mut self.decay,
                acc,
                at,
                back.own::<W>(i, j),
                back.tau[i],
                var,
            );
        }
        self.memo.tails[j] = Some(Tail {
            from,
            from_seq: back.seq[from],
            upto: len - 1,
            upto_seq: back.seq[len - 1],
            acc,
            at,
        });
        Some((acc, at))
    }

    /// The first row from the oldest that `pred` takes for operator `j`, as
    /// its position from the oldest, whether it is in `front`, and its index
    /// there. A queue is read by one predicate, its kernel's, so the search
    /// starts from the oldest row it has not already passed over ([`Memo`]):
    /// O(1) amortized where an operator's input is null for a long run,
    /// which a search from the oldest row each read paid for with every row
    /// of the run (docs/PLAN.md task 270, M-2).
    fn find_oldest(
        &mut self,
        j: usize,
        pred: impl Fn(&Stack, usize) -> bool,
    ) -> Option<(usize, bool, usize)> {
        if self.memo.valued_from.len() != self.n {
            self.memo.valued_from = vec![0; self.n];
        }
        let from = self.memo.valued_from[j];
        let n = self.front.len();
        // Positions from the oldest: `front`'s top down, then `back` up, in
        // ascending `seq`; the first at or past `from`.
        let start = if n > 0 && self.front.seq[n - 1] >= from
            || n == 0 && self.back.seq.first().is_some_and(|&s| s >= from)
        {
            0
        } else {
            let in_front = self.front.seq.partition_point(|&s| s >= from);
            if in_front > 0 {
                n - in_front
            } else {
                n + self.back.seq.partition_point(|&s| s < from)
            }
        };
        let total = n + self.back.len();
        for p in start..total {
            let found = if p < n {
                let i = n - 1 - p;
                pred(&self.front, i).then_some((p, true, i))
            } else {
                pred(&self.back, p - n).then_some((p, false, p - n))
            };
            if let Some((_, in_front, i)) = found {
                let st = if in_front { &self.front } else { &self.back };
                self.memo.valued_from[j] = st.seq[i];
                return found;
            }
        }
        // Nothing valued: the next search starts past the newest row.
        let newest = self
            .back
            .seq
            .last()
            .or_else(|| self.front.seq.first())
            .copied();
        if let Some(s) = newest {
            self.memo.valued_from[j] = s + 1;
        }
        None
    }

    /// The stack [`Queue::find_oldest`]'s `in_front` names.
    fn stack(&self, in_front: bool) -> &Stack {
        if in_front { &self.front } else { &self.back }
    }

    fn clear(&mut self) {
        self.front.clear();
        self.back.clear();
        self.total.clear();
        self.quant.iter_mut().flatten().for_each(QSide::clear);
        self.ext.iter_mut().flatten().for_each(XSide::clear);
        self.ord.iter_mut().flatten().for_each(OrdSide::clear);
        self.since_off = None;
    }

    fn len(&self) -> usize {
        self.front.len() + self.back.len()
    }

    /// Whether both stacks and the running sums are at these operators:
    /// their count, which are variances, and the width that gives; and a
    /// sketch for each quantile, none for any other, each at `front`'s rows.
    fn has_width(
        &self,
        var: &[bool],
        point: &[bool],
        quant: &[bool],
        ext: &[Option<bool>],
        later: bool,
        ord: &[bool],
    ) -> bool {
        let (n, wide) = (var.len(), var.contains(&true));
        let rows = self.front.len();
        let back_rows = self.back.len();
        let ext_fits = if ext.iter().any(Option::is_some) {
            self.ext.len() == n
                && self.ext.iter().zip(ext).all(|(s, e)| match (s, e) {
                    (Some(s), Some(max)) => s.has_shape(*max, later, rows),
                    (None, None) => true,
                    _ => false,
                })
        } else {
            self.ext.is_empty()
        };
        let point_fits = if point.contains(&true) {
            self.point == point
        } else {
            self.point.is_empty()
        };
        let ord_fits = if ord.contains(&true) {
            self.ord.len() == n
                && (self.ord.iter().zip(ord).enumerate()).all(|(j, (s, &o))| match s {
                    Some(s) => {
                        o && s.has_shape(
                            (0..rows)
                                .map(|i| self.front.x_of(wide, i, j))
                                .chain((0..back_rows).map(|i| self.back.x_of(wide, i, j))),
                        )
                    }
                    None => !o,
                })
        } else {
            self.ord.is_empty()
        };
        self.n == n
            && self.var == var
            && point_fits
            && self.front.has_width(n, wide)
            && self.back.has_width(n, wide)
            && (self.total.is_empty() || self.total.len() == self.sum_w() * n)
            && if quant.contains(&true) {
                self.quant.len() == n
                    && (self.quant.iter().zip(quant).enumerate()).all(|(j, (s, &q))| match s {
                        Some(s) => q && s.has_shape((0..rows).map(|i| self.front.x_of(wide, i, j))),
                        None => !q,
                    })
            } else {
                self.quant.is_empty()
            }
            && ext_fits
            && ord_fits
    }
}

/// A row waiting for its group's forward kernels to close.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
struct Wait {
    seq: u64,
    tau: f64,
    /// The row's raw clock in integer nanoseconds on a temporal clock; the
    /// bits of its value on a number clock, and of the policy time without
    /// a clock column (task 159, W3).
    off: i64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Group {
    clock: ClockState,
    /// Policy time of the group's last row, from its last restart.
    tau: f64,
    /// The raw clock the policy time is measured from: the first row of
    /// the stretch (review R1, C2).
    origin: Option<ClockValue>,
    /// The last row's raw clock in integer nanoseconds (a temporal clock),
    /// as its integer (an integer clock, docs/PLAN.md task 200), else the
    /// bits of the clock's value (of the policy time without a clock
    /// column), `form` saying which: a window's edge between two rows is
    /// decided from the difference of their raw clocks, never from two
    /// rounded policy times (review R2, W1; task 159, W3).
    off: i64,
    form: OffForm,
    /// The next row starts the windows over: the group has no row yet, or
    /// an event since its last row ended them.
    restart: bool,
    /// Per kernel.
    queues: Vec<Queue>,
    /// Per forward kernel: the last row, whose interval ends at the next
    /// row, as its place and time; its values are in `open_x` and the
    /// values held over its interval in `open_held`, `n` each.
    open: Vec<Option<(u64, f64, i64)>>,
    open_x: Vec<Vec<f64>>,
    open_held: Vec<Vec<f64>>,
    /// Rows whose forward windows are not all closed, oldest first.
    waiting: VecDeque<Wait>,
    /// Per kernel: how many of `waiting`, from the front, it has closed.
    /// Always 0 for a backward kernel.
    closed: Vec<usize>,
    /// Per operator of a backward kernel: the policy time of its last row
    /// with a value, NaN for none since the restart. A value is held from
    /// there, as `ewm_mean_by` skips a null.
    last_valued: Vec<f64>,
    /// Per operator of a forward kernel: the last value, NaN for none since
    /// the restart, held until the next.
    held: Vec<f64>,
    /// Rows at the current stamp whose backward windows under `"right"` or
    /// `"both"` wait for the next distinct stamp.
    pending: Vec<u64>,
    /// The raw clock of the group's last row and of `pending`.
    stamp: Option<ClockValue>,
    /// Per backward kernel under `"left"` or `"none"`, the values of the
    /// current stamp's window, per operator of the kernel, read before the
    /// stamp's rows joined.
    stamp_values: Vec<Vec<f64>>,
    /// Raw clock of the group's last row, while it is in the silent list.
    last_raw: Option<ClockValue>,
    /// Neighbors in the silent list, the oldest `last_raw` at the head.
    prev_link: Option<usize>,
    next_link: Option<usize>,
    linked: bool,
    /// Per moments operator, its model (`moments.rs`).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    moments: Vec<MomentsRun>,
    /// Per profile operator, its state since the group's last restart;
    /// empty before the group's first row after one.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    profile: Vec<profile::ProfState>,
    /// Per STL or MSTL operator, the group's ring since its last restart;
    /// empty before the group's first row after one.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    seasonal: Vec<seasonal::SeasState>,
    /// Stale rows skipped on this group under `on_step_back = "skip"`.
    #[serde(default, skip_serializing_if = "step_back::is_zero_u64")]
    skipped_step_back: u64,
}

#[cfg(test)]
thread_local! {
    /// [`Windows::flush_group`]'s calls on this thread, and those of them
    /// that found nothing pending: what a row's walk over the groups costs,
    /// counted rather than timed (task 325).
    static FLUSHES: std::cell::Cell<(u64, u64)> = const { std::cell::Cell::new((0, 0)) };
}

/// How a window over a row ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum End {
    /// Its span passed.
    Complete,
    /// A capped gap or a session change cut it short: partial.
    Cut,
    /// A reset discarded it: null, and never dropped.
    Discard,
    /// The end of the input: a window whose far edge is the last row is
    /// whole, as at a cut; every other forward window is unresolved (null,
    /// never dropped), since rows past the input would have joined it.
    Finish,
}

/// Per row not yet emitted.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
struct Meta {
    /// Kernels not yet resolved over the row.
    open: u32,
    accept: bool,
    drop: bool,
}

/// Rows out, in input order: `rows` rows from `first_seq` on.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Emitted {
    pub first_seq: u64,
    pub rows: usize,
    /// Per operator, per row: the value, NaN where null.
    pub values: Vec<Vec<f64>>,
    /// Per row: whether a `"drop"` operator was partial on it.
    pub drop: Vec<bool>,
    /// Per row: whether the row was skipped as stale under `on_step_back = "skip"`.
    pub stale: Vec<bool>,
}

/// The window core: every group's kernels over one stream, fed rows in
/// input order.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Windows {
    /// The operators' definitions: the call's, and the same for every group
    /// of a spec, so behind an `Arc` they are shared (see [`Windows::share_with`]).
    kernels: Arc<Vec<KernelDef>>,
    ops: Arc<Vec<OpDef>>,
    #[serde(with = "clock_form")]
    clock_cfg: ClockCfg,
    /// Per kernel: its operators, in output order.
    members: Arc<Vec<Vec<usize>>>,
    n_outputs: usize,
    n_forward: u32,
    groups: Vec<Group>,
    /// Group keys, as text, to their index in `groups`; the null key apart.
    #[serde(with = "sorted_map")]
    index: KeyMap,
    null_group: Option<usize>,
    /// Integer group keys to their groups, beside `index`: what a chunk of
    /// integer keys is looked up in, never saved.
    #[serde(skip)]
    int_groups: IntGroups,
    /// The stream's own clock, in input order across groups.
    shared: ClockState,
    /// Whether the rows carry a group column: then a session is each
    /// group's and the stream's clock takes none (review R2, W3); without
    /// one the stream is the one group, and its session is the stream's.
    grouped: bool,
    /// Groups with rows waiting, in the order of their last row: the oldest
    /// at the head.
    head: Option<usize>,
    tail: Option<usize>,
    /// The rows fed and not yet emitted, from `held_first` on: per row,
    /// `n_outputs` values (NaN null) and one meta.
    held_first: u64,
    held_values: VecDeque<f64>,
    held_meta: VecDeque<Meta>,
    /// How many held rows, from the first, every kernel has resolved, and
    /// how many of those no operator drops: what [`Windows::drain`] emits.
    ready: usize,
    ready_kept: usize,
    /// The `seq` the next row fed takes.
    next_seq: u64,
    /// Buffers a row's work reuses, sized for the widest kernel.
    #[serde(skip)]
    scratch_row: Vec<f64>,
    #[serde(skip)]
    scratch_sum: Vec<f64>,
    /// What the kernels and operators say once, for the plain loop and the
    /// row's wait count; derived by [`Windows::scratch`], never saved.
    #[serde(skip)]
    derived: Derived,
    /// The cross-group operators, after the window operators in output
    /// order, and where their stamps stand; absent, and not written, where
    /// the call has none, so a state without them keeps its bytes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    cross: Option<Box<Cross>>,
    /// The moments operators' terms, in operator order (`moments.rs`).
    #[serde(default, skip_serializing_if = "moments::none")]
    moments: Arc<Vec<MomentsDef>>,
    /// The profile operators (`ewm_profile`), after the window operators in
    /// output order.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    profiles: Arc<Vec<ProfileDef>>,
    /// The STL and MSTL operators (`stl`, `mstl`), after the profiles in
    /// output order.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    seasonals: Arc<Vec<StlDef>>,
    /// Whether a row has taken a profile's state past half of its cap:
    /// said once a run, and kept with the state so a resumed run does not
    /// say it again; and that warning, `(op, bytes, cap)`, not yet taken.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    state_warned: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    state_warning: Option<(usize, u64, u64)>,
    /// Held rows skipped as stale under `on_step_back = "skip"`, ascending by `seq`.
    #[serde(default, skip_serializing_if = "VecDeque::is_empty")]
    stale_seqs: VecDeque<u64>,
    #[serde(skip)]
    skipped_run: step_back::TransientSkips,
}

/// A map as a sorted list, so the state's bytes do not depend on the
/// hasher.
mod sorted_map;

/// `ClockCfg` with its session gap tagged. The core type's own derive
/// writes `Some(SessionGap::Reset)` as a bare unit, which msgpack reads back
/// as `None`: a saved `session_gap = "reset"` would resume as no session
/// gap at all. The bank never saves a `ClockCfg` -- it rebuilds one from
/// its specs -- so this state is the first to need the form.
mod clock_form;

mod gap;
use gap::*;
// The row operators read a step of the clock as the window core does
// (task 564).
pub(crate) use gap::{elapsed, past_cap};

/// Whether operator `o` can run on its kernel: a mean or a variance with
/// no decay needs a window, and a variance looks back only (task 212).
fn check_op(o: usize, op: &OpDef, k: &KernelDef) -> Result<(), String> {
    if op.stat.weighs_held() && k.half_life.is_infinite() && k.window_size.is_none() {
        return Err(format!(
            "operator {o}: a mean or a variance under half_life = inf needs a window_size, \
             since a value held from before the first row would have no finite weight"
        ));
    }
    if op.stat.is_var() && k.direction == Direction::Forward {
        return Err(format!(
            "operator {o}: a variance reads only the rows up to each row, but this one reads \
             the rows after it"
        ));
    }
    moments::check_op(o, op, k)?;
    extrema::check_op(o, op, k)?;
    exact_order::check_op(o, op, k)?;
    match (op.stat, op.quantile) {
        (Stat::Quantile, Some(p)) if p > 0.0 && p < 1.0 => {}
        (Stat::Quantile, p) => {
            return Err(format!(
                "operator {o}: quantile must be above 0 and below 1, but it is {}",
                p.map_or_else(|| "not set".to_string(), |p| p.to_string())
            ));
        }
        (Stat::ExactQuantile, _) => {}
        (_, Some(_)) => {
            return Err(format!(
                "operator {o}: quantile is given, but only a quantile operator takes it"
            ));
        }
        _ => {}
    }
    Ok(())
}

/// `$f::<true>(..)` for a queue holding a variance, `$f::<false>(..)`
/// for any other: the queue code compiled for its width ([`row_w`]).
macro_rules! by_width {
    ($wide:expr, $f:ident ( $($arg:expr),* $(,)? )) => {
        if $wide { $f::<true>($($arg),*) } else { $f::<false>($($arg),*) }
    };
}

// After `by_width!`, which it uses.
mod check;
mod cross;
#[cfg(test)]
mod empty_sum_tests;
#[cfg(test)]
mod extrema_tests;
#[cfg(test)]
mod far_tests;
mod held_mass;
#[cfg(test)]
mod int_clock_tests;
mod keys;
#[cfg(test)]
mod memo_tests;
mod moments;
mod plain;
mod profile;
mod seasonal;
mod step_back;

pub use cross::{
    CROSS_STATE_BYTES, CROSS_STATE_BYTES_CEILING, Cross, CrossClosed, CrossDef, CrossStat, Interp,
    RankMethod,
};
use keys::IntGroups;
pub(crate) use keys::{ChunkKeys, session_hashes};
pub use moments::MomentsDef;
use moments::{MomentsRun, check_moments, learn_moments, read_moments};
pub(crate) use plain::PlainRows;
use plain::{Derived, Parts};
pub use profile::{
    BUCKET_BYTES as PROFILE_BUCKET_BYTES, PROFILE_STATE_BYTES, PROFILE_STATE_BYTES_CEILING,
    PhaseSpan, ProfileDef, ProfileStat, canonical_zone,
};
pub use seasonal::{StlComponent, StlDef};

/// Group keys, as text, to their groups: hashed with a fixed, fast hasher,
/// which the saved state never sees ([`sorted_map`]).
type KeyMap = HashMap<String, usize, polars_utils::aliases::PlFixedStateQuality>;

/// The rows not yet emitted, borrowed apart from the groups.
struct Held<'a> {
    first: u64,
    n_outputs: usize,
    values: &'a mut VecDeque<f64>,
    meta: &'a mut VecDeque<Meta>,
}

/// A refusal no caller's input can reach, said as the defect it is: every
/// caller checks the kernels and operators it hands [`Windows::new`] first
/// (`formula.rs`, `windows_frame.rs`), so a refusal there is this library's.
fn defect(what: &str) -> String {
    format!(
        "internal: {what}, which is a defect in rheon. Open an issue at \
         https://github.com/rheon-dev/rheon/issues with this message."
    )
}

impl Windows {
    /// # Errors
    ///
    /// A kernel with no operator, or a kernel this core cannot run: a defect
    /// of the caller's, which checks both first ([`defect`]). Empty `kernels`
    /// and `ops` builds a clock-only core with zero outputs, used when a run
    /// holds only row operators.
    pub fn new(
        kernels: Vec<KernelDef>,
        ops: Vec<OpDef>,
        clock_cfg: ClockCfg,
    ) -> Result<Self, String> {
        Self::with_moments(kernels, ops, Vec::new(), clock_cfg)
    }

    /// [`Self::new`] with cross-group operators beside the window
    /// operators, after them in output order.
    ///
    /// # Errors
    ///
    /// As [`Self::new`].
    pub fn with_cross(
        kernels: Vec<KernelDef>,
        ops: Vec<OpDef>,
        cross: Vec<CrossDef>,
        clock_cfg: ClockCfg,
    ) -> Result<Self, String> {
        Self::with_cross_moments(kernels, ops, cross, Vec::new(), clock_cfg)
    }

    /// [`Self::new`] with profile operators after the window operators and
    /// the cross-group ones.
    ///
    /// # Errors
    ///
    /// As [`Self::new`].
    pub fn with_profiles(
        kernels: Vec<KernelDef>,
        ops: Vec<OpDef>,
        profiles: Vec<ProfileDef>,
        clock_cfg: ClockCfg,
    ) -> Result<Self, String> {
        Self::with_all(kernels, ops, Vec::new(), Vec::new(), profiles, clock_cfg)
    }

    fn build(
        kernels: Vec<KernelDef>,
        ops: Vec<OpDef>,
        cross: Vec<CrossDef>,
        profiles: Vec<ProfileDef>,
        seasonals: Vec<StlDef>,
        clock_cfg: ClockCfg,
    ) -> Result<Self, String> {
        for p in &profiles {
            p.check()?;
        }
        for s in &seasonals {
            s.check()?;
        }
        for k in &kernels {
            k.check()?;
        }
        let mut members = vec![Vec::new(); kernels.len()];
        for (o, op) in ops.iter().enumerate() {
            if op.kernel >= kernels.len() {
                return Err(format!(
                    "operator {o} names kernel {}, which there is not",
                    op.kernel
                ));
            }
            if op.min_samples == 0 && op.stat != Stat::Sum {
                return Err("min_samples must be at least 1 (only a sum may take 0)".into());
            }
            let k = &kernels[op.kernel];
            check_op(o, op, k)?;
            members[op.kernel].push(o);
        }
        if let Some(k) = members.iter().position(Vec::is_empty) {
            return Err(format!("kernel {k} has no operator"));
        }
        let widest = members.iter().map(Vec::len).max().unwrap_or(0);
        let n_forward = kernels
            .iter()
            .filter(|k| k.direction == Direction::Forward)
            .count();
        let n_cross = cross.len();
        let first_out = ops.len();
        Ok(Self {
            cross: (n_cross > 0).then(|| Box::new(Cross::new(cross, first_out))),
            n_outputs: ops.len() + n_cross + profiles.len() + seasonals.len(),
            profiles: Arc::new(profiles),
            seasonals: Arc::new(seasonals),
            state_warned: false,
            state_warning: None,
            stale_seqs: VecDeque::new(),
            skipped_run: step_back::TransientSkips::default(),
            n_forward: u32::try_from(n_forward).map_err(|_| "too many kernels")?,
            kernels: Arc::new(kernels),
            ops: Arc::new(ops),
            clock_cfg,
            members: Arc::new(members),
            groups: Vec::new(),
            index: KeyMap::default(),
            null_group: None,
            int_groups: IntGroups::default(),
            shared: ClockState::new(),
            grouped: false,
            head: None,
            tail: None,
            held_first: 0,
            held_values: VecDeque::new(),
            held_meta: VecDeque::new(),
            ready: 0,
            ready_kept: 0,
            next_seq: 0,
            scratch_row: vec![0.0; ROW_VAR * widest],
            scratch_sum: vec![0.0; SUM_VAR * widest],
            derived: Derived::default(),
            moments: Arc::new(Vec::new()),
        })
    }

    /// The profile operators, after [`Self::ops`] in output order.
    pub fn profiles(&self) -> &[ProfileDef] {
        &self.profiles
    }

    /// The STL and MSTL operators, after [`Self::profiles`] in output order.
    pub fn seasonals(&self) -> &[StlDef] {
        &self.seasonals
    }

    /// Per kernel, which of its operators are variances: what its queue is
    /// built with, and checked against.
    fn var_of(&self, ki: usize) -> Vec<bool> {
        self.members[ki]
            .iter()
            .map(|&o| self.ops[o].stat.is_var())
            .collect()
    }

    fn point_of(&self, ki: usize) -> Vec<bool> {
        self.members[ki]
            .iter()
            .map(|&o| self.ops[o].stat == Stat::PointMean)
            .collect()
    }

    fn ord_of(&self, ki: usize) -> Vec<bool> {
        self.members[ki]
            .iter()
            .map(|&o| self.ops[o].stat.is_exact_order())
            .collect()
    }

    /// Per kernel, which of its operators are quantiles: which keep
    /// sketches in its queue.
    fn quant_of(&self, ki: usize) -> Vec<bool> {
        self.members[ki]
            .iter()
            .map(|&o| self.ops[o].stat == Stat::Quantile)
            .collect()
    }

    /// Per kernel, which of its operators are extrema (`Some(true)` a
    /// maximum), and whether ties go to the later row (looking ahead).
    fn ext_of(&self, ki: usize) -> (Vec<Option<bool>>, bool) {
        let ext = self.members[ki]
            .iter()
            .map(|&o| match self.ops[o].stat {
                Stat::Max => Some(true),
                Stat::Min => Some(false),
                _ => None,
            })
            .collect();
        (ext, self.kernels[ki].direction == Direction::Forward)
    }

    /// Say whether the rows carry a group column (review R2, W3): with
    /// one, a session is each group's and the stream's clock takes none;
    /// without, the stream is the one group, and its session change ends
    /// the windows as the one group's clock sees it.
    pub fn set_grouped(&mut self, grouped: bool) {
        self.grouped = grouped;
    }

    /// The scratch buffers, which a loaded state lacks.
    #[inline]
    fn scratch(&mut self) {
        if !self.derived.ready {
            // The plain loop has no moments (`moments.rs`), and a profile reads
            // a group's own state, which it knows none of.
            // An STL operator's ring takes every row in order, as a profile's
            // buckets do: the plain loop has neither.
            let off = self.derived.off
                || !self.moments.is_empty()
                || !self.profiles.is_empty()
                || !self.seasonals.is_empty();
            self.derived = Derived::of(&self.kernels, &self.ops, &self.members, off);
            self.derived.n_wait |= u32::from(self.profiles.iter().any(ProfileDef::waits));
        }
        if self.scratch_sum.is_empty() && !self.members.is_empty() {
            let widest = self.members.iter().map(Vec::len).max().unwrap_or(0);
            self.scratch_row = vec![0.0; ROW_VAR * widest];
            self.scratch_sum = vec![0.0; SUM_VAR * widest];
        }
    }

    /// Whether this core and `other` hold one copy of the operator
    /// definitions.
    #[cfg(test)]
    pub fn shares_definitions_with(&self, other: &Windows) -> bool {
        Arc::ptr_eq(&self.kernels, &other.kernels)
            && Arc::ptr_eq(&self.ops, &other.ops)
            && Arc::ptr_eq(&self.members, &other.members)
    }

    /// Share `other`'s operator definitions where they are the same as this
    /// core's, so that the cores of one spec's groups hold one copy of them.
    pub fn share_with(&mut self, other: &Windows) {
        if self.kernels != other.kernels {
            return;
        }
        if self.ops == other.ops && self.members == other.members {
            self.kernels = Arc::clone(&other.kernels);
            self.ops = Arc::clone(&other.ops);
            self.members = Arc::clone(&other.members);
        }
        if self.moments == other.moments {
            self.moments = Arc::clone(&other.moments);
        }
    }

    pub fn kernels(&self) -> &[KernelDef] {
        &self.kernels
    }

    pub fn ops(&self) -> &[OpDef] {
        &self.ops
    }

    /// The cross-group operators, after [`Self::ops`] in output order.
    pub fn cross_defs(&self) -> &[CrossDef] {
        self.cross.as_ref().map_or(&[], |c| c.defs())
    }

    /// The first row to take a cross-group operator's, or a profile's,
    /// state past half of its cap, once a run, as `(of, op, bytes, cap)`,
    /// taken by this call; `op` is the operator's index among those of its
    /// kind.
    pub fn take_state_warning(&mut self) -> Option<(StateOf, usize, u64, u64)> {
        if let Some((op, bytes, cap)) = self.cross.as_mut().and_then(|c| c.take_warning()) {
            return Some((StateOf::Cross, op, bytes, cap));
        }
        let (op, bytes, cap) = self.state_warning.take()?;
        Some((StateOf::Profile, op, bytes, cap))
    }

    pub fn clock_cfg(&self) -> &ClockCfg {
        &self.clock_cfg
    }

    pub fn n_outputs(&self) -> usize {
        self.n_outputs
    }

    /// The stream's own clock: what the last row left it at.
    pub fn shared_clock(&self) -> &ClockState {
        &self.shared
    }

    /// Rows fed and not yet emitted.
    pub fn held(&self) -> usize {
        self.held_meta.len()
    }

    /// Contributing rows every queue holds, summed: what the windows cost.
    pub fn queued(&self) -> usize {
        self.groups
            .iter()
            .flat_map(|g| g.queues.iter())
            .map(Queue::len)
            .sum()
    }

    /// Buckets every quantile's sketches hold, summed: what the quantiles
    /// cost beside the rows queued.
    pub fn quantile_buckets(&self) -> usize {
        self.groups
            .iter()
            .flat_map(|g| g.queues.iter())
            .map(Queue::quantile_buckets)
            .sum()
    }

    /// The `seq` the next row fed takes: the rows fed so far, across a save
    /// and a load.
    pub fn next_seq(&self) -> u64 {
        self.next_seq
    }

    /// The index of the group with this key, as text; a new key makes a new
    /// group. Only a group's first row allocates.
    pub fn group(&mut self, key: Option<&str>) -> usize {
        let found = match key {
            Some(k) => self.index.get(k).copied(),
            None => self.null_group,
        };
        if let Some(gi) = found {
            return gi;
        }
        let gi = self.groups.len();
        let queues = (0..self.kernels.len())
            .map(|ki| {
                let (ext, later) = self.ext_of(ki);
                Queue::new(
                    self.var_of(ki),
                    &self.point_of(ki),
                    &self.quant_of(ki),
                    &ext,
                    later,
                    &self.ord_of(ki),
                )
            })
            .collect();
        self.groups.push(Group {
            clock: ClockState::new(),
            tau: 0.0,
            origin: None,
            off: 0,
            form: OffForm::Bits,
            restart: true,
            queues,
            open: vec![None; self.kernels.len()],
            open_x: self
                .members
                .iter()
                .map(|m| vec![f64::NAN; m.len()])
                .collect(),
            open_held: self
                .members
                .iter()
                .map(|m| vec![f64::NAN; m.len()])
                .collect(),
            waiting: VecDeque::new(),
            closed: vec![0; self.kernels.len()],
            last_valued: vec![f64::NAN; self.ops.len()],
            held: vec![f64::NAN; self.ops.len()],
            pending: Vec::new(),
            stamp: None,
            stamp_values: self
                .members
                .iter()
                .map(|m| vec![f64::NAN; m.len()])
                .collect(),
            last_raw: None,
            prev_link: None,
            next_link: None,
            linked: false,
            moments: self.moments.iter().map(MomentsRun::new).collect(),
            profile: Vec::new(),
            seasonal: Vec::new(),
            skipped_step_back: 0,
        });
        match key {
            Some(k) => {
                self.index.insert(k.to_string(), gi);
            }
            None => self.null_group = Some(gi),
        }
        gi
    }

    /// Both clocks stepped on copies by a row's clock and session, for the
    /// caller to keep once neither refuses it. With groups a session is
    /// each group's: the stream's clock keeps the stream's order and its
    /// gaps and takes no session (review R2, W3: groups with sessions of
    /// their own restarted every group at every row); without, the stream
    /// is the one group. `gi` is `None` for a group with no row yet.
    #[inline(always)]
    fn step_clocks(
        &self,
        gi: Option<usize>,
        clock: Option<ClockValue>,
        session: Option<u64>,
    ) -> Result<(ClockState, ClockAdvance, ClockState, ClockAdvance), Refusal> {
        let seq = self.next_seq;
        let stream_session = if self.grouped { None } else { session };
        let mut shared = self.shared.clone();
        let adv = shared.advance(&self.clock_cfg, clock, stream_session, true);
        refuse_backwards(&adv, seq, self.shared.last_clock(), clock)?;
        // A restart on the stream's clock starts every group over, its clock
        // included: this row is the first on a fresh clock of its group, not
        // a step back on a stale one (task 159, W1: a group's next row within
        // `restart_after_step_back` of its last was refused as a late row
        // after the stream had restarted, where its windows had started over).
        let mut own_clock = if adv.reset {
            ClockState::new()
        } else {
            gi.map_or_else(ClockState::new, |g| self.groups[g].clock.clone())
        };
        let prev = own_clock.last_clock();
        let own = own_clock.advance(&self.clock_cfg, clock, session, true);
        refuse_backwards(&own, seq, prev, clock)?;
        Ok((shared, adv, own_clock, own))
    }

    /// What the clocks would make of a row with this group key, clock and
    /// session, without taking it: refused as a step back the policy
    /// refuses, or taken as a new start, a step forward or the same stamp.
    /// What a run resumed on an input that starts elsewhere asks of the
    /// input's first row (review R6, D3 and D4): a new start or a step
    /// forward is the next file.
    pub fn peek(
        &self,
        key: Option<&str>,
        clock: Option<ClockValue>,
        session: Option<u64>,
    ) -> Result<Peek, Refusal> {
        let gi = match key {
            Some(k) => self.index.get(k).copied(),
            None => self.null_group,
        };
        let (_, adv, _, own) = self.step_clocks(gi, clock, session)?;
        if adv.skipped_step_back || own.skipped_step_back {
            return Ok(Peek::StepBack);
        }
        if adv.reset || adv.session_changed || own.reset || own.session_changed {
            return Ok(Peek::NewStart);
        }
        match (clock, self.shared.last_clock()) {
            (Some(now), Some(last)) if !now.is_before(last) && !last.is_before(now) => {
                Ok(Peek::SameStamp)
            }
            _ => Ok(Peek::Forward),
        }
    }

    /// Feed one row. Earlier rows may be resolved by it; take them with
    /// [`Self::drain`].
    ///
    /// # Errors
    ///
    /// A step back the clock policy refuses, on the stream's clock or on
    /// the group's. A refused row changes nothing.
    pub fn push(&mut self, row: &RowIn<'_>) -> Result<(), Refusal> {
        self.scratch();
        let seq = self.next_seq;
        let gi = row.group;
        let stream_session = if self.grouped { None } else { row.session };
        let plain = match self.shared.plain_forward(row.clock, stream_session, true) {
            Some(a) => self.groups[gi]
                .clock
                .plain_forward_after(&self.shared, a, row.clock, row.session, true)
                .map(|b| (a, b)),
            None => None,
        };
        let pre_stepped = if plain.is_none() && self.clock_cfg.on_clock_reset.skips() {
            let res = self.step_clocks(Some(gi), row.clock, row.session)?;
            if res.1.skipped_step_back || res.3.skipped_step_back {
                self.push_stale(gi, seq);
                return Ok(());
            }
            Some(res)
        } else {
            None
        };
        // The cross-group operators' state with this row, before anything
        // moves: a row that would take an operator past its cap is refused.
        let cross_held = match &self.cross {
            Some(c) => Some(c.check_row(row, seq, self.n_outputs)?),
            None => None,
        };
        // A profile's state with this row, before anything moves: a row
        // that would take one past its cap is refused.
        let said = if self.profiles.is_empty() {
            None
        } else {
            profile::check_row(&self.profiles, &self.groups[gi], row.clock, row.values, seq)?
        };
        let (adv, clock, own) = if let Some((a, b)) = plain {
            let adv = self
                .shared
                .take_plain_forward(&self.clock_cfg, row.clock, stream_session, a);
            let own = self.groups[gi].clock.take_plain_forward(
                &self.clock_cfg,
                row.clock,
                row.session,
                b,
            );
            (adv, None, own)
        } else {
            let (shared, adv, clock, own) = match pre_stepped {
                Some(res) => res,
                None => self.step_clocks(Some(gi), row.clock, row.session)?,
            };
            self.shared = shared;
            (adv, Some(clock), own)
        };
        self.next_seq += 1;
        // An event on the stream's clock reaches every group now; each
        // group's own clock would show it at that group's next row.
        if adv.reset {
            self.end_all(End::Discard);
            // Every group's clock too (task 159, W1): the row's own group
            // takes the fresh one stepped above.
            for g in &mut self.groups {
                g.clock = ClockState::new();
            }
        } else if adv.session_changed || adv.capped {
            self.end_all(End::Cut);
        }
        if let Some(now) = row.clock {
            self.end_silent(now);
        }
        if own.reset {
            self.end_group(gi, End::Discard);
        } else if own.session_changed || own.capped {
            self.end_group(gi, End::Cut);
        }

        // A new stamp resolves the stamps before it, in every group: the
        // stream is in clock order across groups.
        if let Some(now) = row.clock {
            self.flush_stamps_before(now);
        }

        // The cross-group operators' stamp: a later one closes the open
        // stamp, whose rows are the ones before this; a reset discards it.
        if let Some(c) = self.cross.as_mut() {
            let index = &self.index;
            let find = |k: &str| index.get(k).copied();
            let mut rows = cross::Rows {
                first: self.held_first,
                n_outputs: self.n_outputs,
                values: &mut self.held_values,
                meta: &mut self.held_meta,
                find: &find,
            };
            c.before_row(
                row.clock,
                adv.reset,
                adv.session_changed || adv.capped,
                &mut rows,
                seq,
                &self.stale_seqs,
            );
        }

        // One push each: `extend(repeat_n(..))` was a call to a fill routine
        // for the one or two values a row holds.
        for _ in 0..self.n_outputs {
            self.held_values.push_back(f64::NAN);
        }
        let has_clock = row.clock.is_some();
        // Under "right" and "both" a backward window waits for the next
        // distinct stamp; a row-count clock repeats no stamp, so the row's
        // own windows are read at the end of this push (review R1, C1: they
        // were never read without a clock, and every such output was null).
        let n_wait = self.derived.n_wait;
        let cross_wait = self.cross.as_ref().map_or(0, |c| u32::from(c.waits()));
        self.held_meta.push_back(Meta {
            open: self.n_forward + n_wait + cross_wait,
            accept: row.accept,
            drop: false,
        });
        if let Some(c) = self.cross.as_mut() {
            let index = &self.index;
            let find = |k: &str| index.get(k).copied();
            let mut rows = cross::Rows {
                first: self.held_first,
                n_outputs: self.n_outputs,
                values: &mut self.held_values,
                meta: &mut self.held_meta,
                find: &find,
            };
            c.row(row, seq, &mut rows);
            if let Some(held) = cross_held {
                c.note_held(&held);
            }
        }

        let g = &mut self.groups[gi];
        if let Some(clock) = clock {
            g.clock = clock;
        }
        let restarted = own.reset || own.session_changed || own.capped || g.restart;
        if restarted {
            // Every window of the group was ended above or before: nothing
            // refers to the old policy time, so it starts over at 0.
            restart_in(g, row.clock);
        } else if let (Some(now), Some(origin)) = (row.clock, g.origin) {
            // From the stretch's origin in one subtraction (exact in
            // integer nanoseconds), not a sum of rounded steps: summed, a
            // row exactly one window later landed on either side of the
            // edge by the sum's noise (review R1, C2).
            g.tau = elapsed(now, origin);
            if !g.tau.is_finite() {
                return Err(Refusal::Span { seq, origin, now });
            }
        } else {
            g.tau += own.d_clock;
            if g.origin.is_none() {
                g.origin = row.clock;
            }
        }
        // The raw clock a window's edge is decided from (review R2, W1): the
        // nanoseconds of a temporal clock, exact; the bits of a number
        // clock, so the edge is the difference of the two rows' clocks, as
        // Polars takes it, not of two origin-subtracted policy times, which
        // put a row exactly one window back outside the edge, `0.4 − 0.1`
        // not being `0.3` (task 159, W3); the policy time's bits without a
        // clock column, where the two are one.
        (g.off, g.form) = match row.clock {
            Some(ClockValue::Ns(c)) => (c, OffForm::Ns),
            Some(ClockValue::I64(c)) => (c, OffForm::Int),
            Some(ClockValue::F64(c)) => (c.to_bits() as i64, OffForm::Bits),
            None => (g.tau.to_bits() as i64, OffForm::Bits),
        };
        let (tau, off, form) = (g.tau, g.off, g.form);
        let new_stamp = g.stamp != row.clock || !has_clock;
        // A queue past its extrema's cap: the row is taken and refused after.
        let mut over: Option<(usize, usize)> = None;

        // Forward kernels: the windows this row reaches past close first --
        // it is not in them -- with the kernel's open row, the last one,
        // counted up to here; then that row joins the queue, its interval
        // ended by this one, and this row is the open one.
        let mut held = Held {
            first: self.held_first,
            n_outputs: self.n_outputs,
            values: &mut self.held_values,
            meta: &mut self.held_meta,
        };
        for &ki in &self.derived.forward {
            let ki = ki as usize;
            let k = &self.kernels[ki];
            forward_step(
                k,
                &self.ops,
                &self.members[ki],
                ki,
                g,
                &mut self.scratch_row,
                &mut self.scratch_sum,
                FwdRow {
                    seq,
                    tau,
                    off,
                    form,
                },
                |input| row.values[input],
                &mut held,
            );
            if let Some(limit) = self.derived.cap_rows[ki]
                && g.queues[ki].len() > limit
            {
                over = Some((ki, g.queues[ki].len()));
            }
        }
        if !self.derived.forward.is_empty() {
            retire(&self.kernels, g);
        }

        // Backward kernels: under "left" and "none" the stamp's window is
        // read before its rows join; then the row joins; under "right" and
        // "both" the stamp's rows wait for the next stamp.
        for &ki in &self.derived.backward {
            let ki = ki as usize;
            let k = &self.kernels[ki];
            let m = &self.members[ki];
            if k.since.is_some() && new_stamp {
                g.queues[ki]
                    .ext
                    .iter_mut()
                    .flatten()
                    .for_each(XSide::new_stamp);
            }
            if !k.closed.near(Direction::Backward) && new_stamp {
                by_width!(
                    g.queues[ki].wide(),
                    read_backward(
                        k,
                        &self.ops,
                        m,
                        &mut g.queues[ki],
                        &mut self.scratch_sum,
                        tau,
                        off,
                        form,
                        &mut g.stamp_values[ki],
                    )
                );
            }
            if k.since.is_some_and(|i| row.values[i] == 1.0) {
                extrema::event(k, &self.ops, m, g, ki, &mut self.scratch_sum, off, form);
            }
            // A value is held from the operator's last valued row; the first
            // of a stretch from before it: from `-inf`, or from a window
            // back where one bounds the mass.
            let (rw, wide) = (g.queues[ki].row_w(), g.queues[ki].wide());
            for (j, &o) in m.iter().enumerate() {
                let v = row.values[self.ops[o].input];
                let v = if usable(v) && self.ops[o].stat != Stat::Moments {
                    v
                } else {
                    f64::NAN
                };
                let start = if g.last_valued[o].is_nan() {
                    k.window_size.map_or(f64::NEG_INFINITY, |w| tau - w)
                } else {
                    g.last_valued[o]
                };
                let stat = self.ops[o].stat;
                let acc = if v.is_nan() {
                    Acc::default()
                } else if stat.weighs_held() {
                    Acc::held(
                        v,
                        mass_held_to(k, tau, start, &mut g.queues[ki].decay),
                        stat.is_var(),
                    )
                } else if stat == Stat::PointMean {
                    Acc {
                        s: v,
                        w: 1.0,
                        n: 1.0,
                        ..Acc::default()
                    }
                } else {
                    Acc {
                        s: v,
                        n: 1.0,
                        ..Acc::default()
                    }
                };
                let b = j * rw;
                self.scratch_row[b] = start;
                self.scratch_row[b + 1] = v;
                acc.write(&mut self.scratch_row[b + 2..], wide);
                if !v.is_nan() {
                    g.last_valued[o] = tau;
                }
            }
            g.queues[ki].push_row(
                k,
                seq,
                tau,
                off,
                tau,
                &self.scratch_row[..rw * m.len()],
                &mut self.scratch_sum,
            );
            if !self.moments.is_empty() {
                let now = row.clock.unwrap_or(ClockValue::F64(tau));
                learn_moments(k, ki, &self.ops, &self.moments, g, row.values, now);
            }
            if let Some(limit) = self.derived.cap_rows[ki]
                && g.queues[ki].len() > limit
            {
                over = Some((ki, g.queues[ki].len()));
            }
            if !k.waits() {
                let complete = k.window_size.is_none_or(|w| tau >= w);
                let held = Held {
                    first: self.held_first,
                    n_outputs: self.n_outputs,
                    values: &mut self.held_values,
                    meta: &mut self.held_meta,
                };
                write_backward(&self.ops, m, seq, &g.stamp_values[ki], complete, held);
            }
        }
        if !self.profiles.is_empty() {
            let out = profile::Out {
                defs: &self.profiles,
                first: self.ops.len() + self.cross.as_ref().map_or(0, |c| c.defs().len()),
            };
            let mut held = Held {
                first: self.held_first,
                n_outputs: self.n_outputs,
                values: &mut self.held_values,
                meta: &mut self.held_meta,
            };
            profile::row(&out, g, row.clock, row.values, new_stamp, seq, &mut held);
        }
        if !self.seasonals.is_empty() {
            let out = seasonal::Out {
                defs: &self.seasonals,
                first: self.ops.len()
                    + self.cross.as_ref().map_or(0, |c| c.defs().len())
                    + self.profiles.len(),
            };
            let mut held = Held {
                first: self.held_first,
                n_outputs: self.n_outputs,
                values: &mut self.held_values,
                meta: &mut self.held_meta,
            };
            let clock = row.clock.map_or(f64::NAN, ClockValue::seconds);
            seasonal::row(&out, g, clock, row.values, seq, &mut held);
        }
        if n_wait > 0 {
            g.pending.push(seq);
        }
        g.stamp = row.clock;
        if let Some(w) = said.filter(|_| !self.state_warned) {
            self.state_warned = true;
            self.state_warning = Some(w);
        }

        if self.n_forward > 0 {
            g.waiting.push_back(Wait { seq, tau, off });
        }
        if let Some(now) = row.clock {
            g.last_raw = Some(now);
            self.link_back(gi);
        }
        if !has_clock {
            // A row-count clock repeats no stamp: the row's own backward
            // windows are known now.
            self.flush_group(gi);
        }
        self.advance_ready();
        match over {
            Some((ki, rows)) => {
                let q = &self.groups[gi].queues[ki];
                let m = &self.members[ki];
                Err(Refusal::OverCap {
                    seq,
                    rows,
                    row_bytes: extrema::queue_bytes(q) / rows.max(1),
                    cap: extrema::cap_of(&self.kernels[ki], &self.ops, m).unwrap_or(0),
                })
            }
            None => Ok(()),
        }
    }

    /// Resolve the backward windows of every group's stamp before `now`.
    ///
    /// The silent list holds every group with a clock, in the order of its
    /// last row, and the stream is in clock order across groups: a step
    /// back on the stream's clock is refused or starts every group over,
    /// and a session change on it ends every group's windows, either way
    /// emptying the list. So the groups with rows pending are those whose
    /// last row is at the stream's latest stamp -- every row of a group
    /// joins its pending rows, and only a later stamp, which flushes them,
    /// or the group's end, which unlinks it, empties them -- and they are
    /// the list's tail. Walking from the head, as this did, stepped over
    /// every group ever seen and already flushed, on every row (task 325:
    /// 80 µs a row at 10,000 groups). From the tail it steps over only the
    /// groups it flushes, plus one: nothing under `"left"` or `"none"`,
    /// which pend no row, and nothing on a row at the same stamp. They are
    /// flushed in the list's order, the head walk's.
    fn flush_stamps_before(&mut self, now: ClockValue) {
        let passed = |g: &Group| !g.pending.is_empty() && g.stamp.is_some_and(|s| is_after(now, s));
        let mut first = None;
        let mut at = self.tail;
        while let Some(i) = at.filter(|&i| passed(&self.groups[i])) {
            first = Some(i);
            at = self.groups[i].prev_link;
        }
        let mut gi = first;
        while let Some(i) = gi {
            self.flush_group(i);
            gi = self.groups[i].next_link;
        }
    }

    /// The kernels, the held rows and the buffers a group's windows are read
    /// and written with, borrowed apart from the groups.
    #[inline(always)]
    fn split(&mut self) -> (Parts<'_>, &mut Vec<Group>) {
        let Windows {
            kernels,
            ops,
            members,
            groups,
            held_values,
            held_meta,
            held_first,
            n_outputs,
            scratch_row,
            scratch_sum,
            n_forward,
            derived,
            moments,
            profiles,
            ..
        } = self;
        (
            Parts {
                kernels: kernels.as_slice(),
                ops: ops.as_slice(),
                members: members.as_slice(),
                held_values,
                held_meta,
                held_first: *held_first,
                n_out: *n_outputs,
                scratch_row,
                scratch_sum,
                n_wait: derived.n_wait,
                n_forward: *n_forward,
                moments: moments.as_slice(),
                profiles: profiles.as_slice(),
                fast: derived.fast.as_slice(),
            },
            groups,
        )
    }

    /// Resolve the pending backward windows of group `gi` at its stamp.
    fn flush_group(&mut self, gi: usize) {
        #[cfg(test)]
        FLUSHES.with(|c| {
            let (calls, idle) = c.get();
            let idle = idle + u64::from(self.groups[gi].pending.is_empty());
            c.set((calls + 1, idle));
        });
        if self.groups[gi].pending.is_empty() {
            return;
        }
        let (mut p, groups) = self.split();
        flush_in(&mut p, &mut groups[gi]);
    }

    /// End every group's open windows `how`; every group starts its windows
    /// over at its next row.
    fn end_all(&mut self, how: End) {
        for gi in 0..self.groups.len() {
            self.end_group(gi, how);
        }
    }

    /// Cut the open windows of every group whose last row is more than
    /// `gap_cap` behind `now`: the stream's clock has shown that its next
    /// row opens with a longer gap. Only a clock column has a cap; on a
    /// row-count clock nothing cuts a silent group, which is why a forward
    /// window under groups needs a clock column (task 173, PC1).
    fn end_silent(&mut self, now: ClockValue) {
        let cap = self.clock_cfg.gap_cap;
        if !cap.is_finite() {
            return;
        }
        while let Some(gi) = self.head {
            let last = self.groups[gi]
                .last_raw
                .expect("a listed group has a clock");
            if !past_cap(now, last, cap) {
                break;
            }
            self.end_group(gi, End::Cut);
        }
    }

    /// End a group's open windows `how`: its pending backward windows are
    /// read as they stand, its forward windows closed at the last row the
    /// group saw -- "the value over what it saw", with nothing held past
    /// that row (review R1, C4: held `gap_cap` past it, a kept rate's span
    /// grew with the cap) -- and its queues emptied; its next row starts
    /// every window over.
    fn end_group(&mut self, gi: usize, how: End) {
        self.flush_group(gi);
        let (mut p, groups) = self.split();
        end_in(&mut p, &mut groups[gi], how);
        self.unlink(gi);
    }

    fn unlink(&mut self, gi: usize) {
        let g = &mut self.groups[gi];
        if !g.linked {
            return;
        }
        let (prev, next) = (g.prev_link.take(), g.next_link.take());
        g.linked = false;
        match prev {
            Some(p) => self.groups[p].next_link = next,
            None => self.head = next,
        }
        match next {
            Some(n) => self.groups[n].prev_link = prev,
            None => self.tail = prev,
        }
    }

    /// Move a group to the tail of the silent list: its last row is the
    /// stream's latest.
    fn link_back(&mut self, gi: usize) {
        // Already the tail: unlinking it and linking it back would leave
        // every link as it is.
        if self.tail == Some(gi) && self.groups[gi].linked {
            return;
        }
        self.unlink(gi);
        let tail = self.tail;
        let g = &mut self.groups[gi];
        g.prev_link = tail;
        g.linked = true;
        match tail {
            Some(t) => self.groups[t].next_link = Some(gi),
            None => self.head = Some(gi),
        }
        self.tail = Some(gi);
    }

    fn advance_ready(&mut self) {
        while let Some(m) = self.held_meta.get(self.ready).filter(|m| m.open == 0) {
            self.ready_kept += usize::from(!m.drop);
            self.ready += 1;
        }
    }

    /// Resolved rows no operator drops, ready to go out: how far a
    /// [`Self::drain_kept`] could go now.
    pub fn ready_kept(&self) -> usize {
        self.ready_kept
    }

    /// The held rows every kernel has resolved, dropped ones included: what
    /// [`Self::drain`] emits.
    pub fn ready(&self) -> usize {
        self.ready
    }

    /// The rows every kernel has resolved, in input order.
    pub fn drain(&mut self) -> Emitted {
        self.emit(self.ready)
    }

    /// [`Self::drain`], stopping at the `limit`-th row no operator drops:
    /// the resolved rows after it stay held, for a later drain or a saved
    /// state.
    pub fn drain_kept(&mut self, limit: usize) -> Emitted {
        let mut kept = 0;
        let mut rows = 0;
        while rows < self.ready && kept < limit {
            kept += usize::from(!self.held_meta[rows].drop);
            rows += 1;
        }
        self.emit(rows)
    }

    /// The end of an input whose state is saved to go on: every backward
    /// window still waiting on the last stamp is read as it stands, and the
    /// rows every kernel has now resolved are ready for [`Self::drain`]. No
    /// forward window is touched, and the core takes rows as before: a row
    /// at the stamp read would have joined those windows, which is why a
    /// run resumed on such a row is refused, and one after it reads the same
    /// windows as one that went on without this call.
    pub fn settle(&mut self) {
        self.scratch();
        for gi in 0..self.groups.len() {
            self.flush_group(gi);
        }
        self.advance_ready();
    }

    /// The end of the input: every backward window still waiting is read
    /// as it stands, a forward window under "right" or "both" whose far edge
    /// is its group's last row is whole (no later row could join it), every
    /// other forward window still open is unresolved -- null whatever
    /// `partial` says, never dropped -- and every row goes out. The core
    /// takes no more rows after this.
    pub fn finish(&mut self) -> Emitted {
        self.scratch();
        for gi in 0..self.groups.len() {
            self.flush_group(gi);
            let (mut p, groups) = self.split();
            // The state a finished core saves is as it always was: the
            // closing leaves the group's restart flag where it was.
            let restart = groups[gi].restart;
            end_in(&mut p, &mut groups[gi], End::Finish);
            groups[gi].restart = restart;
            let g = &mut self.groups[gi];
            g.waiting.clear();
            g.closed.iter_mut().for_each(|c| *c = 0);
            for (k, q) in self.kernels.iter().zip(g.queues.iter_mut()) {
                if k.direction == Direction::Forward {
                    q.clear();
                }
            }
            g.open.iter_mut().for_each(|o| *o = None);
            self.unlink(gi);
        }
        if let Some(c) = self.cross.as_mut() {
            let index = &self.index;
            let find = |k: &str| index.get(k).copied();
            let mut rows = cross::Rows {
                first: self.held_first,
                n_outputs: self.n_outputs,
                values: &mut self.held_values,
                meta: &mut self.held_meta,
                find: &find,
            };
            c.finish(&mut rows, self.next_seq, &self.stale_seqs);
        }
        self.ready = self.held_meta.len();
        self.ready_kept = self.held_meta.iter().filter(|m| !m.drop).count();
        self.emit(self.ready)
    }

    fn emit(&mut self, rows: usize) -> Emitted {
        let n_out = self.n_outputs;
        let mut stale = vec![false; rows];
        let limit = self.held_first + rows as u64;
        while let Some(&s) = self.stale_seqs.front() {
            if s >= limit {
                break;
            }
            self.stale_seqs.pop_front();
            stale[(s - self.held_first) as usize] = true;
        }
        let mut out = Emitted {
            first_seq: self.held_first,
            rows,
            values: vec![Vec::with_capacity(rows); n_out],
            drop: Vec::with_capacity(rows),
            stale,
        };
        // The rows' values are one run of `n_out` a row; each operator's are
        // every `n_out`-th of it.
        if n_out > 0 {
            let flat = &self.held_values.make_contiguous()[..rows * n_out];
            for (o, col) in out.values.iter_mut().enumerate() {
                col.extend(flat.chunks_exact(n_out).map(|row| row[o]));
            }
        }
        for meta in self.held_meta.drain(..rows) {
            out.drop.push(meta.drop);
            self.ready_kept -= usize::from(!meta.drop);
        }
        self.ready -= rows;
        self.held_values.drain(..rows * n_out);
        self.held_first += rows as u64;
        out
    }
}

/// A forward row filled into the queue, a backward window read, and the
/// values written: in a module of their own.
mod read;
use read::{
    empty_value, fill_forward, rate_of, read_backward, value_of, write_backward,
    write_backward_ranks,
};

/// Close forward kernel `k` over waiting row `t`: the rows its queue holds
/// after `t` are its window, every one of them held until a row inside it,
/// plus the kernel's open row, held from its time until `open_end` and
/// counted up to the window's far edge.
#[allow(clippy::too_many_arguments)]
fn close<const W: bool>(
    k: &KernelDef,
    ops: &[OpDef],
    members: &[usize],
    q: &mut Queue,
    scratch: &mut [f64],
    open: Option<(u64, f64, i64)>,
    open_x: &[f64],
    open_held: &[f64],
    open_end: f64,
    t: Wait,
    how: End,
    end_tau: f64,
    form: OffForm,
    held: Held<'_>,
) {
    let w = k.window_size.expect("a forward kernel has a window");
    // A window is a set of timestamps: under "left" and "both" every row
    // at the row's own stamp is in it, the row itself included, as every
    // row at a stamp is in a backward "right" window (review R1, C3);
    // under "right" and "none" none at the stamp is. Each edge is the exact
    // span from the row (review R2, W1).
    let near_in = k.closed.near(Direction::Forward);
    q.evict::<W>(k, scratch, |_, _, j_off| {
        let ahead = gap(form, t.off, j_off).cmp_zero();
        if near_in {
            ahead == Ordering::Less
        } else {
            ahead != Ordering::Greater
        }
    });
    let r = usize::try_from(t.seq - held.first).expect("a waiting row is held");
    let meta = &mut held.meta[r];
    meta.open -= 1;
    let far = (t.tau + w).min(end_tau);
    let len = far - t.tau;
    // The open row is in the window when it is past the row (or at its
    // stamp, under "left" and "both") and not past the far edge.
    // The open row may be the row itself, when the next row closed the
    // window before the row joined the queue: in under "left" and "both".
    // An open row exactly a window ahead is in a "right" or "both" window and
    // is held from the far edge on: no mass inside the window, however
    // `t.tau + w` rounds against its `tau`.
    let mut open_on_edge = false;
    let open = open.filter(|&(_, _, o_off)| {
        let ahead = gap(form, t.off, o_off);
        let (near, far) = (ahead.cmp_zero(), ahead.cmp_window(k));
        open_on_edge = far == Ordering::Equal;
        (if near_in {
            near != Ordering::Less
        } else {
            near == Ordering::Greater
        }) && (if k.closed.far(Direction::Forward) {
            far != Ordering::Greater
        } else {
            far == Ordering::Less
        })
    });
    for (j, &o) in members.iter().enumerate() {
        let op = &ops[o];
        if how == End::Cut && op.partial == Partial::Drop {
            meta.drop = true;
        }
        let keep = match how {
            End::Complete => true,
            End::Cut => op.partial == Partial::Keep,
            End::Discard | End::Finish => false,
        };
        // A variance has no forward kernel (`check_op`).
        if !(keep && meta.accept) || op.stat.is_var() {
            continue;
        }
        // The sums are anchored at the oldest row in the window; seen from
        // the row. A held value counts only where the row holding it is in
        // the window: a mean starts at the first row in the window with a
        // value of its own, since the rows before it hold a value from the
        // row itself or from one its stamp excludes.
        let seg = q.sum_op::<W>(k, j);
        let scaled = |memo: &mut DecayMemo, (a, at): (Acc, f64)| {
            a.scaled(k.discount_slot(at - t.tau, &mut memo.fwd().disc_seg), false)
        };
        let mut a = Acc::default();
        let mut any = seg.is_some();
        let mut first_own: Option<usize> = None;
        // A mean is a ratio of its sums, the same seen from any time, so it
        // is read where they are anchored: at the first row in the window
        // with a value of its own, whose mass is discounted by nothing.
        // Discounted to the row, a value more than 1074 half-lives ahead
        // weighed 0 and the mean was null (review 2026-11-23, WIN-2 (3)).
        let mut mean_at: Option<f64> = None;
        match op.stat {
            Stat::Mean => {
                first_own = q
                    .find_oldest(j, |st, i| st.own::<W>(i, j).n > 0.0)
                    .map(|(p, _, _)| p);
                let base = match first_own {
                    None => None,
                    Some(0) => seg,
                    Some(p) => q.sum_after_op::<W>(k, p - 1, j),
                };
                if let Some((b, at)) = base {
                    a = b;
                    mean_at = Some(at);
                }
            }
            Stat::PointMean => {
                if let Some((b, at)) = seg {
                    a = b;
                    mean_at = Some(at);
                }
            }
            Stat::Sum
            | Stat::Rate
            | Stat::Quantile
            | Stat::Max
            | Stat::Min
            | Stat::ExactQuantile
            | Stat::ExactRank => {
                if let Some(s) = seg {
                    a = scaled(&mut q.decay, s);
                }
            }
            Stat::Var | Stat::Std | Stat::Moments => {}
        }
        let mut open_value = None;
        if let Some((_, otau, _)) = open {
            let (x, h) = (open_x[j], open_held[j]);
            let own = !x.is_nan();
            let n = f64::from(u8::from(own));
            match op.stat {
                Stat::Mean if !h.is_nan() && (own || first_own.is_some()) => {
                    // Seen from the sums' anchor; with no sums, from the
                    // open row itself.
                    let m = if open_on_edge {
                        0.0
                    } else {
                        let from = mean_at.unwrap_or(otau);
                        mass_between_close(k, &mut q.decay, t.tau, from, otau, open_end.min(far))
                    };
                    a = a.plus(
                        Acc {
                            s: m * h,
                            w: m,
                            n,
                            ..Acc::default()
                        },
                        false,
                    );
                    any = true;
                }
                Stat::PointMean if own => {
                    let from = mean_at.unwrap_or(otau);
                    let f = k.discount_slot(otau - from, &mut q.decay.fwd().disc_mean);
                    a = a.plus_point(Acc {
                        s: x,
                        w: f,
                        n: 1.0,
                        ..Acc::default()
                    });
                    any = true;
                }
                Stat::Sum
                | Stat::Rate
                | Stat::Quantile
                | Stat::Max
                | Stat::Min
                | Stat::ExactQuantile
                | Stat::ExactRank
                    if own =>
                {
                    a = a.plus(
                        Acc {
                            s: k.discount_slot(otau - t.tau, &mut q.decay.fwd().disc_open) * x,
                            w: 0.0,
                            n,
                            ..Acc::default()
                        },
                        false,
                    );
                    open_value = Some((otau, x));
                    any = true;
                }
                _ => {}
            }
        }
        if !any {
            // A closed window with no row in it.
            held.values[r * held.n_outputs + o] = empty_value(op);
            continue;
        }
        held.values[r * held.n_outputs + o] = if op.stat.is_extremum() {
            if a.n < f64::from(op.min_samples) {
                f64::NAN
            } else {
                q.ext(j).windowed(open_value.map(|(_, x)| x))
            }
        } else if op.stat == Stat::ExactQuantile {
            if a.n < f64::from(op.min_samples) {
                f64::NAN
            } else {
                let p = op.quantile.expect("an exact quantile has a level");
                let interp = op.interpolation.unwrap_or(Interp::Nearest);
                q.ord(j).quantile(p, interp, open_value.map(|(_, x)| x))
            }
        } else if op.stat == Stat::ExactRank {
            if a.n < f64::from(op.min_samples) {
                f64::NAN
            } else {
                let my_x = if open.is_some_and(|(oseq, ..)| oseq == t.seq) {
                    open_x[j]
                } else {
                    q.x_by_seq::<W>(t.seq, j)
                };
                let method = op.rank_method.unwrap_or(RankMethod::Average);
                q.ord(j).rank(my_x, method, open_value.map(|(_, x)| x))
            }
        } else if op.stat == Stat::Quantile {
            if a.n < f64::from(op.min_samples) {
                f64::NAN
            } else {
                let p = op.quantile.expect("a quantile has a level (`check_op`)");
                q.quantile(k, j, p, open_value).unwrap_or(f64::NAN)
            }
        } else if op.stat == Stat::Rate {
            // `value_of`'s rate, its span's mass on a slot: every window of
            // a kernel but a cut one has the span of the window.
            if a.n < f64::from(op.min_samples) {
                f64::NAN
            } else {
                rate_of(a, k.clock_mass_slot(len, &mut q.decay.fwd().mass_span))
            }
        } else {
            value_of(k, op, a, len)
        };
    }
}

/// A row as a forward kernel meets it: its place, its policy time and its raw
/// clock.
#[derive(Clone, Copy)]
struct FwdRow {
    seq: u64,
    tau: f64,
    off: i64,
    form: OffForm,
}

/// One forward kernel at one row of its group (the forward arm of
/// [`Windows::push`], and of the plain loop, which takes a kernel over the
/// rows of a block): the windows the row reaches past close, with the kernel's
/// open row counted up to here; then the open row joins the queue, its
/// interval ended by this row, and this row is the open one. `value` is the
/// row's value of an operator input.
#[allow(clippy::too_many_arguments)]
#[inline]
fn forward_step(
    k: &KernelDef,
    ops: &[OpDef],
    members: &[usize],
    ki: usize,
    g: &mut Group,
    scratch_row: &mut [f64],
    scratch_sum: &mut [f64],
    row: FwdRow,
    value: impl Fn(usize) -> f64,
    held: &mut Held<'_>,
) {
    let FwdRow {
        seq,
        tau,
        off,
        form,
    } = row;
    let w = k.window_size.expect("a forward kernel has a window");
    // A row exactly a window later is in it under "right" and
    // "both", so the window closes only past it.
    while g.closed[ki] < g.waiting.len() && {
        let t = g.waiting[g.closed[ki]];
        let d = gap(form, t.off, off).cmp_window(k);
        if k.closed.far(Direction::Forward) {
            d == Ordering::Greater
        } else {
            d != Ordering::Less
        }
    } {
        let t = g.waiting[g.closed[ki]];
        g.closed[ki] += 1;
        let h = Held {
            first: held.first,
            n_outputs: held.n_outputs,
            values: &mut *held.values,
            meta: &mut *held.meta,
        };
        by_width!(
            g.queues[ki].wide(),
            close(
                k,
                ops,
                members,
                &mut g.queues[ki],
                scratch_sum,
                g.open[ki],
                &g.open_x[ki],
                &g.open_held[ki],
                tau,
                t,
                End::Complete,
                t.tau + w,
                form,
                h,
            )
        );
    }
    if let Some((oseq, otau, ooff)) = g.open[ki].take() {
        let (rw, wide) = (g.queues[ki].row_w(), g.queues[ki].wide());
        by_width!(
            wide,
            fill_forward(
                k,
                ops,
                members,
                otau,
                tau,
                &g.open_x[ki],
                &g.open_held[ki],
                scratch_row,
                &mut g.queues[ki].decay,
            )
        );
        g.queues[ki].push_row(
            k,
            oseq,
            otau,
            ooff,
            tau,
            &scratch_row[..rw * members.len()],
            scratch_sum,
        );
    }
    // A value is held until the next: a row with none holds the last.
    for (j, &o) in members.iter().enumerate() {
        let v = value(ops[o].input);
        let v = if usable(v) { v } else { f64::NAN };
        g.open_x[ki][j] = v;
        if !v.is_nan() {
            g.held[o] = v;
        }
        g.open_held[ki][j] = g.held[o];
    }
    g.open[ki] = Some((seq, tau, off));
}

/// [`extrema::complete`] where kernel `k` can have a `since` event, and its
/// window test alone where it cannot: only an extremum takes `since`
/// (`extrema::check_op`), and only a `since` event sets a queue's
/// `since_off`, so without one the two answer alike. A frame with no
/// extremum pays nothing of the extrema's test a kernel a flush.
#[inline(always)]
fn complete_at(k: &KernelDef, q: &Queue, tau: f64, off: i64, form: OffForm) -> bool {
    if k.since.is_some() {
        extrema::complete(k, q, tau, off, form)
    } else {
        k.window_size.is_none_or(|w| tau >= w)
    }
}

/// Resolve the pending backward windows of group `g` at its stamp: every
/// kernel that waits for the next stamp is read as the group stands, and the
/// pending rows are written.
#[inline(always)]
fn flush_in(p: &mut Parts<'_>, g: &mut Group) {
    if g.pending.is_empty() {
        return;
    }
    let (tau, off, form) = (g.tau, g.off, g.form);
    for (ki, k) in p.kernels.iter().enumerate() {
        if !k.waits() {
            continue;
        }
        let m = &p.members[ki];
        // Under "left" and "none" the stamp's values were read as it began
        // (a `since` kernel's, which waits for an event at any of its rows).
        if k.closed.near(Direction::Backward) {
            by_width!(
                g.queues[ki].wide(),
                read_backward(
                    k,
                    p.ops,
                    m,
                    &mut g.queues[ki],
                    p.scratch_sum,
                    tau,
                    off,
                    form,
                    &mut g.stamp_values[ki],
                )
            );
        }
        // A frame with no moments operator has nothing to read here, and
        // pays no call a kernel a flush for it.
        if !p.moments.is_empty() {
            read_moments(ki, p.ops, m, p.moments, g, tau);
        }
        let complete = complete_at(k, &g.queues[ki], tau, off, form);
        for &seq in &g.pending {
            let held = Held {
                first: p.held_first,
                n_outputs: p.n_out,
                values: &mut *p.held_values,
                meta: &mut *p.held_meta,
            };
            write_backward(p.ops, m, seq, &g.stamp_values[ki], complete, held);
        }
        if g.queues[ki].has_ord() {
            let held = Held {
                first: p.held_first,
                n_outputs: p.n_out,
                values: &mut *p.held_values,
                meta: &mut *p.held_meta,
            };
            by_width!(
                g.queues[ki].wide(),
                write_backward_ranks(p.ops, m, &g.queues[ki], &g.pending, complete, held)
            );
        }
    }
    if !p.profiles.is_empty() {
        let out = profile::Out {
            defs: p.profiles,
            first: p.ops.len(),
        };
        let mut held = Held {
            first: p.held_first,
            n_outputs: p.n_out,
            values: &mut *p.held_values,
            meta: &mut *p.held_meta,
        };
        let pending = std::mem::take(&mut g.pending);
        profile::flush(&out, g, &pending, &mut held);
        g.pending = pending;
    }
    for &seq in &g.pending {
        let r = usize::try_from(seq - p.held_first).expect("a pending row is held");
        p.held_meta[r].open -= 1;
    }
    // Emptied, not taken: a stamp's rows wait here on every row of a clock,
    // and the next row's push would make the list again.
    g.pending.clear();
}

/// End group `g`'s open windows `how` (the silent list is the caller's): its
/// pending backward windows are read as they stand, its forward windows
/// closed at the last row the group saw, and its queues emptied; its next row
/// starts every window over.
#[inline(always)]
fn end_in(p: &mut Parts<'_>, g: &mut Group, how: End) {
    flush_in(p, g);
    g.restart = true;
    let end_tau = g.tau;
    for (ki, k) in p.kernels.iter().enumerate() {
        if k.direction != Direction::Forward {
            continue;
        }
        let w = k.window_size.expect("a forward kernel has a window");
        while g.closed[ki] < g.waiting.len() {
            let t = g.waiting[g.closed[ki]];
            g.closed[ki] += 1;
            // Under "right" and "both" a window whose far edge is the
            // stretch's last row holds every timestamp it can, and a cut
            // or a reset leaves none of them to come: it is whole, as its
            // backward mirror is, where a cut cut it short -- null, or
            // dropped (task 160, PC2) -- and a reset discarded it (task
            // 173, PC2). Under "left" and "none" a row on the edge closed
            // it already. A reset discards every other window.
            let whole = matches!(how, End::Cut | End::Discard | End::Finish)
                && k.closed.far(Direction::Forward)
                && gap(g.form, t.off, g.off).cmp_window(k) == Ordering::Equal;
            // At the end of the input every other window stays unresolved.
            if how == End::Finish && !whole {
                continue;
            }
            let (how, far) = if whole {
                (End::Complete, t.tau + w)
            } else {
                (how, end_tau)
            };
            let held = Held {
                first: p.held_first,
                n_outputs: p.n_out,
                values: &mut *p.held_values,
                meta: &mut *p.held_meta,
            };
            by_width!(
                g.queues[ki].wide(),
                close(
                    k,
                    p.ops,
                    &p.members[ki],
                    &mut g.queues[ki],
                    p.scratch_sum,
                    g.open[ki],
                    &g.open_x[ki],
                    &g.open_held[ki],
                    end_tau,
                    t,
                    how,
                    far,
                    g.form,
                    held,
                )
            );
        }
        g.open[ki] = None;
        g.queues[ki].clear();
    }
    retire(p.kernels, g);
}

/// A group's row that starts its windows over -- the first after a cut or a
/// reset -- finds nothing of the old stretch: every window ended already or
/// before, so the policy time starts again at 0 from this row's clock.
#[inline(always)]
fn restart_in(g: &mut Group, origin: Option<ClockValue>) {
    for q in &mut g.queues {
        q.clear();
    }
    g.open.iter_mut().for_each(|o| *o = None);
    g.last_valued.iter_mut().for_each(|v| *v = f64::NAN);
    g.held.iter_mut().for_each(|v| *v = f64::NAN);
    g.moments.iter_mut().for_each(MomentsRun::restart);
    g.tau = 0.0;
    g.origin = origin;
    g.restart = false;
    g.stamp = None;
    g.pending.clear();
    g.profile.clear();
    g.seasonal.clear();
}

/// Drop the waiting rows every forward kernel has closed.
fn retire(kernels: &[KernelDef], g: &mut Group) {
    let done = kernels
        .iter()
        .zip(&g.closed)
        .filter(|(k, _)| k.direction == Direction::Forward)
        .map(|(_, &c)| c)
        .min()
        .unwrap_or(0);
    if done == 0 {
        return;
    }
    g.waiting.drain(..done);
    for (k, c) in kernels.iter().zip(g.closed.iter_mut()) {
        if k.direction == Direction::Forward {
            *c -= done;
        }
    }
}

#[cfg(test)]
mod tests;
