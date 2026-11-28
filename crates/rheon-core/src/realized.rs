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
//!
//! **The online form.** A row is a clock step, one return per target (a
//! target absent on a row has no return there) and a weight `w ≥ 0`. Per
//! target, with `f = e^(−ln 2 · d / half_life)` the row's decay, every sum
//! decays and then takes the row's terms, each term weighted by the weight
//! of the latest row in it:
//!
//! ```text
//! W ← f W + w        V ← f² V + w²  the returns' weight and squared weights
//! s_rv ← f s_rv + w r_j²
//! s_bv ← f s_bv + w a_j a_{j−1}     p ← f p + w      (from the second return on)
//! s_tv ← f s_tv + w (c_j c_{j−1} c_{j−2})²            c = a^(1/3)
//! s_tq ← f s_tq + w (c_j c_{j−1} c_{j−2})⁴            (from the third return on)
//! ```
//!
//! and a row reads `RV = s_rv`, `BV = (π/2) s_bv`, `TV = μ_{2/3}^-3 s_tv`,
//! `TQ = W μ_{4/3}^-3 s_tq`, `Z` with `n` for the papers' count, and `σ_i² =
//! s_bv / p`, the weighted mean of the products before the row. `n`, the
//! count of returns the statistics are of, is Kish's effective count `W² /
//! V` of the returns' weights, decayed or in a window: the count of returns
//! at equal weights without a decay, and otherwise the count the weights
//! are worth, `Z` and the flag reading the same `n`. The two places the
//! papers' `n` stands for different things once the weights are unequal,
//! and each takes its own: at a constant volatility `σ²` a step `δ`,
//! `Var(RV − BV) ≈ ϑ σ⁴ δ² V` and `RV ≈ σ² δ W`, so `1 − BV/RV` has the
//! variance `ϑ V/W² = ϑ/n` and `√n` in front is Kish's count; `TQ/BV²`,
//! which is 1 there (the `max(1, ·)` is Jensen's bound for a varying one),
//! needs `TQ`'s factor to be the weight `W`, the period its sum is of. With
//! Kish's count in `TQ` too, `max(1, TQ/BV²)` read `W/V`, about `1 + λ` under
//! any decay, and `Z` was N(0, 1/(1 + λ)). That variance holds for weights
//! that vary slowly from one return to the next, as a decay's do: `ϑ`
//! counts each product's covariance with its neighbour, which shares a
//! return, at the products' own weight. With no decay (`half_life = inf`)
//! and unit weights these are the papers' values on the returns so far, `n
//! − 1` products and `n − 2` triples. A weight counts returns: the
//! estimators are sums, so they scale with it, and `Z` and the flag do not
//! move with the weights' scale.
//!
//! **Out of sample** (hard rule 2): a row reports from the state before its
//! own return and before its own decay, so the variations are those of the
//! returns before it, and `L_i` is the row's return against their
//! volatility. `weight_sum` is the weight of every row, and each target's
//! own weight `W` that of its returns, both before the row's update and
//! before its decay (hard rule 8); a target reports `NaN` while its `W` is
//! under `min_weight`, the test where `RV` or `BV` is 0, and `L_i` where
//! `σ_i` is 0 or the row has no return for it.
//!
//! **A row of weight 0** learns nothing (hard rule 9): no sum takes a term
//! and the last returns are kept, so the next product reaches across it to
//! the last return learned. The stream hands no such row to the model, so
//! it counts in no clock; handed one directly, the model ages its sums over
//! the row's step, as every model does.
//!
//! **The sums are compensated**: each is a pair (TwoSum), its high
//! part and what that leaves out, so a sum over a day of 23,400 returns
//! keeps the rounding of its terms rather than a random walk of 23,400
//! roundings (40 to 130 units in the last place, measured, against 0 to 5).
//! The cube roots are the `libm` crate's, the same bits on every platform.
//!
//! **Past the largest double.** A term of returns near the input bound can
//! pass the largest double: `w t⁴` of three returns of `1e100` is
//! `1e400` (`t` the product of their cube roots). Such a term, and a sum it
//! takes past the largest double, is held at it, the largest a double
//! holds: a bound on the term, where `inf` would never decay (`f · inf` is
//! `inf`) and a window's next take-back of it, `inf − inf`, would be NaN.
//! The weight goes into such a term first, so one that fits is kept. Each
//! output reads its own sum, so a sum held at the bound moves no other:
//! while its sum is held, `tq` is withheld (`NaN`) if the scaled product
//! passes the largest double or reads a lower bound on itself, and `Z`,
//! which divides by the raw product, reads near 0, as the quarticity's true
//! size would make it. Below the largest double nothing moves, to the bit.
//!
//! A window takes a term back from a sum held at the bound by summing the
//! terms it still holds again, and does the same where a term leaving is
//! more than `1/ε` times what it leaves (a return near the bound leaving a
//! window of ordinary ones): the compensated pair keeps a sum to about
//! `ε²` of the largest term it has held, so what it leaves is otherwise
//! the rounding of that term, which no row would ever take away. Lee and
//! Mykland's window of products does the same.
//!
//! **Lee and Mykland's flag.** With `c = √(2/π)`, `c L_i` is N(0, 1) under
//! no jump, so `c · lm` is read as a z-score. The paper tests the largest
//! `|L_i|` of `n` returns against the Gumbel law of its maximum,
//!
//! ```text
//! C_n = √(2 ln n)/c − (ln π + ln ln n) / (2 c √(2 ln n))     S_n = 1 / (c √(2 ln n))
//! jump at i  ⇔  (|L_i| − C_n) / S_n > β*,   β* = −ln(−ln β)
//! ```
//!
//! `β` the test's confidence, `P(ξ ≤ β*) = β` for the standard Gumbel `ξ`
//! (`lm_confidence`, 0.99 by default, a test at 1%: `β* = 4.6001`). `n` is the
//! target's own count of returns, the `n` of the other estimators (above),
//! so the threshold is that of the stretch they are measured over; the flag
//! is `NaN` where `L_i` is, or where `n ≤ 1`, where `C_n` has no value (`ln
//! ln n`).
//!
//! **A window** (`window_rows = W`, or `window_clock = D`) in place of a
//! decay: the estimators of the returns of the last `W` rows of positive
//! weight, or of the rows of the last `D` clock units, alone. A product or a
//! triple is in the window when every return in it is: `W` returns hold `W −
//! 1` products and `W − 2` triples, as the papers' sums over `W` returns do.
//! Each row's terms are held in a ring, a target's apart, and taken back as
//! its oldest return leaves, with the products and triples that reached it;
//! the sums are compensated, and set to exactly 0 when a target's ring
//! empties, so their rounding is of the window, not of every row the stream
//! dropped. `D` is read on the rows' exact stamps ([`crate::Stamp`]) under
//! `closed`: `Right`, the default, drops a row exactly `D` old, `Both` keeps
//! it. As under a decay, a row reports from the window as it stood after
//! the row before; the rows that have left by its own clock leave after it
//! reports. A window forgets by dropping, so it takes no decay. `weight_sum`
//! and each target's own weight are then the weights inside the window
//! (hard rule 8), and its `n` Kish's count of them.
//!
//! **`lm_window_rows = K`**, Lee and Mykland's window, takes `σ_i` from the `K −
//! 1` returns of the target before row `i` alone, their `K − 2` products,
//! whatever the estimators' window or decay. It counts returns, as the
//! paper's `K` does, so it has no form in clock units. Without it `σ_i` is
//! read from the estimators' own sums.
//!
//! **The state.** Per target, `18` numbers (seven compensated sums, the last
//! return's size, the last two cube roots and how many returns are
//! remembered) and the shared weight: `8 + 144 m` bytes for `m` targets
//! under a decay, whatever the stream's length. `lm_window_rows` adds `16 (K − 2)
//! + 32` bytes a target. A window adds `32` bytes, and a row it holds `16`
//! (its sequence and weight; under `window_clock` a stamp of 32 more) and `56`
//! a target with a return there (its sequence, weight, four terms and how
//! many of its products were taken). `window_rows` is checked against
//! `max_state_bytes` when the model is built, and refused naming the
//! largest window that fits; `window_clock` holds as many rows as its span
//! has, which the data decides, so it is checked as the rows go in: the
//! first row to take the state past half of `max_state_bytes` raises a
//! warning, once, and a row that would take it past the cap is not learned,
//! nor any after it ([`OnlineModel::state_over_cap`]).

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

use crate::clock_span::Place;
use crate::model::{ModelState, OnlineModel, State, StateError, Step, check_schema};
use crate::{Decay, Stamp, WindowClosed};

mod series;

#[cfg(test)]
use series::SUMS;
use series::{Entry, LmRing, Series};

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

/// The outputs a target reports, in order: `rv`, `bv`, `tv`, `tq`, `bns`,
/// `lm`, `lm_jump`.
pub const OUTPUTS_PER_TARGET: usize = 7;

/// Lee and Mykland's confidence `β` where none is set.
pub const DEFAULT_LM_CONFIDENCE: f64 = 0.99;

fn default_lm_confidence() -> f64 {
    DEFAULT_LM_CONFIDENCE
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RealizedCfg {
    /// The return series, one per target.
    pub n_targets: usize,
    /// The forgetting, per clock unit; `Halflife(inf)` keeps every return.
    pub decay: Decay,
    /// The weight of a target's returns before it reports.
    pub min_weight: f64,
    /// The estimators of the last this many rows of positive weight alone,
    /// in place of a decay.
    #[serde(default)]
    pub window_rows: Option<usize>,
    /// The estimators of the rows of the last this many clock units alone
    /// (seconds on a temporal clock), in place of a decay. How many rows
    /// that is, the data decides.
    #[serde(default)]
    pub window_clock: Option<f64>,
    /// Which edge of a clock window holds the row exactly `window_clock`
    /// old: `Right`, the default, drops it, `Both` keeps it.
    #[serde(default)]
    pub closed: WindowClosed,
    /// Lee and Mykland's `K`: `σ_i` from the target's `K − 1` returns
    /// before the row alone. At least 3 by the statistic's definition, the
    /// mean of `K − 2` products; before `K − 1` returns, the mean of those
    /// there are. Without it, from the estimators' sums.
    #[serde(default)]
    pub lm_window_rows: Option<usize>,
    /// The confidence `β` of Lee and Mykland's flag, above 0 and below 1.
    #[serde(default = "default_lm_confidence")]
    pub lm_confidence: f64,
    /// The bytes a group's state may hold, a window's rows included; the
    /// default is [`crate::clock_span::DEFAULT_STATE_BYTES`].
    #[serde(default)]
    pub max_state_bytes: Option<u64>,
}

impl RealizedCfg {
    pub fn validate(&self) -> Result<(), String> {
        let mut no = crate::Refusals::new();
        if self.n_targets == 0 {
            no.refuse(
                "realized: the model needs at least one target, a series of returns, but it \
                 has none. Name at least one column of returns in targets.",
            );
        }
        if let Err(e) = self.decay.check() {
            no.refuse(format!("realized: {e}"));
        }
        no.take(crate::model::check_min_weight("realized", self.min_weight));
        if self.window_rows == Some(0) {
            no.refuse(
                "realized: window_rows must be at least 1, but it is 0. Give the number of \
                 rows the estimators are measured over, or leave it out for no window.",
            );
        }
        if let Some(d) = self.window_clock {
            if !(d.is_finite() && d > 0.0) {
                no.refuse(format!(
                    "realized: the clock window must be a finite number of clock units above \
                     0, but it is {d}. Give the span of clock the estimators are measured \
                     over, or leave it out for no window."
                ));
            }
            if self.window_rows.is_some() {
                no.refuse(
                    "realized: a window of clock units and window_rows are two windows, and \
                     the estimators hold one. Leave one of them out.",
                );
            }
        } else if self.closed != WindowClosed::Right {
            no.refuse(
                "realized: closed = \"both\" says which edge of a clock window keeps the row \
                 exactly its span old, but there is no clock window. Give the window's span, \
                 or leave closed out.",
            );
        }
        if self.windowed() && self.decay.check().is_ok() && self.decay.factor(1.0) != 1.0 {
            no.refuse(
                "realized: a window forgets by dropping the oldest rows, so it takes no decay, \
                 but half_life (or lam) decays too. Set half_life to inf, or leave out the \
                 window.",
            );
        }
        if let Some(k) = self.lm_window_rows
            && k < 3
        {
            no.refuse(format!(
                "realized: lm_window_rows must be at least 3, but it is {k}: Lee and Mykland's \
                 volatility is the mean of the K - 2 products of the K - 1 returns before a \
                 row. Give K of 3 or more, or leave it out to read it from the estimators."
            ));
        }
        if !(self.lm_confidence > 0.0 && self.lm_confidence < 1.0) {
            no.refuse(format!(
                "realized: lm_confidence must be above 0 and below 1, but it is {}. Give the \
                 confidence of Lee and Mykland's test, such as 0.99, or leave it out for \
                 0.99.",
                self.lm_confidence
            ));
        }
        match crate::clock_span::check_max_state_bytes("realized", self.max_state_bytes) {
            Err(e) => no.refuse(e),
            Ok(()) if self.n_targets > 0 => {
                let (bytes, cap) = (self.fixed_bytes(), self.cap());
                if bytes > cap {
                    no.refuse(self.past_the_cap(bytes, cap));
                }
            }
            Ok(()) => {}
        }
        no.finish()
    }

    /// The refusal of settings whose fixed state passes the cap, naming the
    /// largest window that fits where a window is what passes it.
    fn past_the_cap(&self, bytes: u64, cap: u64) -> String {
        let m = self.n_targets as u64;
        let base = Realized::state_bytes(self.n_targets, self.lm_window_rows, Some(0), false);
        let row = Realized::window_row_bytes(self.n_targets, false);
        match self.window_rows {
            Some(w) if base <= cap => format!(
                "realized: its state would hold {bytes} bytes a group, past max_state_bytes of \
                 {cap}: a window of {w} rows keeps {row} bytes a row for {}. The largest \
                 window_rows that fits is {}. Lower window_rows, or raise max_state_bytes.",
                crate::model::count(self.n_targets, "target"),
                (cap - base) / row,
            ),
            _ => format!(
                "realized: its state would hold {bytes} bytes a group, past max_state_bytes of \
                 {cap}: {} bytes for each of {}{}. Give fewer targets a spec, {}or raise \
                 max_state_bytes.",
                bytes / m.max(1),
                crate::model::count(self.n_targets, "target"),
                if self.lm_window_rows.is_some() {
                    ", lm_window_rows's products included"
                } else {
                    ""
                },
                if self.lm_window_rows.is_some() {
                    "lower lm_window_rows, "
                } else {
                    ""
                },
            ),
        }
    }

    fn windowed(&self) -> bool {
        self.window_rows.is_some() || self.window_clock.is_some()
    }

    /// The bytes the settings fix: the state with a full `window_rows`
    /// window, or with an empty clock window.
    fn fixed_bytes(&self) -> u64 {
        Realized::state_bytes(
            self.n_targets,
            self.lm_window_rows,
            self.window_rows.or(self.window_clock.map(|_| 0)),
            self.window_clock.is_some(),
        )
    }

    /// The bytes a group's state may hold.
    fn cap(&self) -> u64 {
        self.max_state_bytes
            .unwrap_or(crate::clock_span::DEFAULT_STATE_BYTES)
    }
}

/// A row a window holds: its place in the stream, its weight and, in a
/// window of clock units, its stamp.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Held {
    seq: u64,
    w: f64,
    #[serde(default)]
    stamp: Option<Stamp>,
}

/// The rows a window holds, oldest first, the weight inside it, the next
/// row's sequence and, in a window of clock units, the clock.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Window {
    rows: VecDeque<Held>,
    w_hi: f64,
    w_lo: f64,
    next_seq: u64,
    #[serde(default)]
    place: Option<Place>,
}

impl Window {
    fn new(clocked: bool) -> Self {
        Self {
            rows: VecDeque::new(),
            w_hi: 0.0,
            w_lo: 0.0,
            next_seq: 0,
            place: clocked.then(Place::default),
        }
    }

    fn weight(&self) -> f64 {
        self.w_hi + self.w_lo
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Realized {
    cfg: RealizedCfg,
    series: Vec<Series>,
    /// Every row's weight, decayed: the emitted `weight_sum` under a decay.
    w_sum: f64,
    window: Option<Window>,
    /// The bytes a clock window's next row would have taken the state to,
    /// past the cap: from then on no row is learned.
    #[serde(default)]
    over: Option<u64>,
    /// The half-of-the-cap warning has been raised (once a group).
    #[serde(default)]
    warned: bool,
    /// The warning raised and not yet taken. Never state.
    #[serde(skip)]
    warning: Option<u64>,
}

impl Realized {
    pub fn new(cfg: RealizedCfg) -> Result<Self, String> {
        cfg.validate()?;
        Ok(Self {
            series: vec![Series::new(cfg.lm_window_rows); cfg.n_targets],
            w_sum: 0.0,
            window: cfg
                .windowed()
                .then(|| Window::new(cfg.window_clock.is_some())),
            over: None,
            warned: false,
            warning: None,
            cfg,
        })
    }

    pub fn cfg(&self) -> &RealizedCfg {
        &self.cfg
    }

    /// The bytes a window keeps a row it holds: its sequence and weight, a
    /// stamp under a clock window, and an [`Entry`] for each target.
    pub fn window_row_bytes(n_targets: usize, clocked: bool) -> u64 {
        let stamp = if clocked {
            std::mem::size_of::<Stamp>() as u64
        } else {
            0
        };
        16 + stamp + Entry::BYTES * n_targets as u64
    }

    /// The bytes of the numbers a model of `n_targets` series holds: `8 +
    /// 144 m` under a decay; with `lm_window_rows = K`, `16 (K − 2) + 32` more a
    /// target; with a window holding `rows` rows, its fixed `32` and
    /// [`Self::window_row_bytes`] a row, every target's return there.
    pub fn state_bytes(
        n_targets: usize,
        lm_window_rows: Option<usize>,
        rows: Option<usize>,
        clocked: bool,
    ) -> u64 {
        let m = n_targets as u64;
        let lm = lm_window_rows.map_or(0, LmRing::bytes);
        8 + (Series::BYTES + lm) * m
            + rows.map_or(0, |r| {
                32 + r as u64 * Self::window_row_bytes(n_targets, clocked)
            })
    }

    /// The weight the estimators are of (hard rule 8): every row's, decayed,
    /// or the weight inside the window.
    pub fn n_eff(&self) -> f64 {
        self.window.as_ref().map_or(self.w_sum, Window::weight)
    }

    /// The stamp of the newest row a clock window holds, `None` before one
    /// and without a clock window: where a caller that stamps its rows
    /// resumes from a restored state.
    pub fn last_stamp(&self) -> Option<Stamp> {
        self.window.as_ref()?.rows.back()?.stamp
    }

    /// The rows a window holds; 0 without one.
    pub fn window_len(&self) -> usize {
        self.window.as_ref().map_or(0, |w| w.rows.len())
    }

    /// The bytes the state holds now, a window's rows as they are.
    fn bytes_with(&self, extra: usize) -> u64 {
        Self::state_bytes(
            self.cfg.n_targets,
            self.cfg.lm_window_rows,
            self.window.as_ref().map(|w| w.rows.len() + extra),
            self.cfg.window_clock.is_some(),
        )
    }

    /// Each target's outputs into `pred`, from the state as it stands, for
    /// a row whose returns are `y`.
    fn read_into(&self, y: &[Option<f64>], pred: &mut Vec<f64>) {
        pred.clear();
        for (j, s) in self.series.iter().enumerate() {
            s.read(
                y.get(j).copied().flatten(),
                self.cfg.min_weight,
                self.cfg.lm_confidence,
                pred,
            );
        }
    }

    /// The run-time check of a clock window's rows, over this model's
    /// fields ([`crate::clock_span::HeldCap`]).
    fn held_cap(&mut self) -> crate::clock_span::HeldCap<'_> {
        crate::clock_span::HeldCap {
            over: &mut self.over,
            warned: &mut self.warned,
            warning: &mut self.warning,
        }
    }

    /// The oldest row out of the window, and each target's terms that
    /// reached it.
    fn drop_oldest(&mut self) {
        let Some(w) = self.window.as_mut() else {
            return;
        };
        let Some(old) = w.rows.pop_front() else {
            return;
        };
        series::add(&mut w.w_hi, &mut w.w_lo, -old.w);
        if w.rows.is_empty() {
            (w.w_hi, w.w_lo) = (0.0, 0.0);
        }
        for s in &mut self.series {
            s.drop_front(old.seq);
        }
    }

    /// The rows of a clock window that have left at `now`, oldest first.
    fn trim(&mut self, now: Stamp) {
        let (Some(span), closed) = (self.cfg.window_clock, self.cfg.closed) else {
            return;
        };
        while self
            .window
            .as_ref()
            .and_then(|w| w.rows.front())
            .and_then(|h| h.stamp)
            .is_some_and(|s| closed.has_left(now, s, span))
        {
            self.drop_oldest();
        }
    }

    #[cold]
    #[inline(never)]
    fn admit_clock_window(&mut self, d_clock: f64) -> bool {
        let Some(now) = self
            .window
            .as_ref()
            .and_then(|w| w.place.as_ref())
            .map(|p| p.peek(d_clock))
        else {
            return true;
        };
        let dropped = match (self.cfg.window_clock, &self.window) {
            (Some(span), Some(w)) => w
                .rows
                .iter()
                .take_while(|h| {
                    h.stamp
                        .is_some_and(|s| self.cfg.closed.has_left(now, s, span))
                })
                .count(),
            _ => 0,
        };
        let rows = self.window.as_ref().map(|w| w.rows.len() - dropped + 1);
        let bytes = Self::state_bytes(self.cfg.n_targets, self.cfg.lm_window_rows, rows, true);
        let cap = self.cfg.cap();
        if !self.held_cap().admit(bytes, cap) {
            if let Some(p) = self.window.as_mut().and_then(|w| w.place.as_mut()) {
                p.discard_next();
            }
            return false;
        }
        true
    }
}

impl OnlineModel for Realized {
    fn step_into(
        &mut self,
        _x: &[f64],
        y: &[Option<f64>],
        d_clock: f64,
        weight: f64,
        out: &mut Step,
    ) {
        // A return or a weight that is not usable, by the rule every model
        // keeps (`OnlineModel`): such a return is absent, and such a weight
        // learns nothing. The model reads no features.
        if let Some(refused) = crate::model::refused_step(self, &[], y, d_clock, weight) {
            *out = refused;
            return;
        }
        // Read before the row's return and before its decay, or before the
        // rows that leave a window at this row have left (hard rules 2 and
        // 8).
        self.read_into(y, &mut out.pred);
        out.n_eff = self.n_eff();
        out.extra = None;
        if self.over.is_some() {
            if let Some(p) = self.window.as_mut().and_then(|w| w.place.as_mut()) {
                p.discard_next();
            }
            return;
        }
        if weight > 0.0 && self.window.is_some() && !self.admit_clock_window(d_clock) {
            return;
        }
        let f = self.cfg.decay.factor(d_clock);
        self.w_sum = f * self.w_sum + weight;
        for s in &mut self.series {
            s.decay(f);
        }
        let now = self
            .window
            .as_mut()
            .and_then(|w| w.place.as_mut())
            .map(|p| p.step(d_clock));
        if let Some(now) = now {
            self.trim(now);
        }
        // A row of weight 0 learns nothing, and keeps the last returns as
        // they are (hard rule 9).
        if weight <= 0.0 {
            return;
        }
        let seq = match self.window.as_mut() {
            Some(w) => {
                let seq = w.next_seq;
                w.next_seq += 1;
                w.rows.push_back(Held {
                    seq,
                    w: weight,
                    stamp: now,
                });
                series::add(&mut w.w_hi, &mut w.w_lo, weight);
                Some(seq)
            }
            None => None,
        };
        let lm_window_rows = self.cfg.lm_window_rows;
        for (j, s) in self.series.iter_mut().enumerate() {
            if let Some(r) = y.get(j).copied().flatten() {
                s.learn(r, weight, seq, lm_window_rows);
            }
        }
        if let Some(cap) = self.cfg.window_rows {
            while self.window_len() > cap {
                self.drop_oldest();
            }
        }
        if now.is_some() {
            let (bytes, cap) = (self.bytes_with(0), self.cfg.cap());
            self.held_cap().note(bytes, cap);
        }
    }

    fn predict(&self, x: &[f64], d_clock: f64) -> Step {
        self.predict_with(x, &[], d_clock)
    }

    /// The row's returns are read for `lm` and its flag, the row's own
    /// statistic; every other output is of the returns before the row.
    fn predict_with(&self, _x: &[f64], y: &[Option<f64>], d_clock: f64) -> Step {
        if let Some(refused) = crate::model::refused_predict_with(self, &[], y, d_clock) {
            return refused;
        }
        let mut pred = Vec::with_capacity(self.n_outputs());
        self.read_into(y, &mut pred);
        Step {
            pred,
            n_eff: self.n_eff(),
            extra: None,
        }
    }

    /// A clock window learns its rows' stamps: they decide when each leaves.
    fn stamp_next(&mut self, stamp: Stamp) {
        if let Some(p) = self.window.as_mut().and_then(|w| w.place.as_mut()) {
            p.hand(stamp);
        }
    }

    fn take_state_warning(&mut self) -> Option<(u64, u64)> {
        self.warning.take().map(|bytes| (bytes, self.cfg.cap()))
    }

    fn state_over_cap(&self) -> Option<(u64, u64)> {
        self.over.map(|bytes| (bytes, self.cfg.cap()))
    }

    fn target_n_eff_into(&self, out: &mut Vec<f64>) -> bool {
        out.clear();
        out.extend(self.series.iter().map(Series::weight));
        true
    }

    fn state(&self) -> State {
        State::new(ModelState::Realized(Box::new(self.clone())))
    }

    fn restore(s: &State) -> Result<Self, StateError> {
        check_schema(s)?;
        match &s.model {
            ModelState::Realized(m) => {
                let m = (**m).clone();
                crate::model::check_cfg("realized", m.cfg.validate())?;
                let rows = m.window.as_ref().map(|w| &w.rows);
                let shaped = m.series.len() == m.cfg.n_targets
                    && m.series
                        .iter()
                        .all(|s| s.has_shape(m.cfg.lm_window_rows, rows))
                    && match (&m.window, m.cfg.windowed()) {
                        (None, false) => true,
                        (Some(w), true) => {
                            w.place.is_some() == m.cfg.window_clock.is_some()
                                && w.rows.iter().all(|h| {
                                    h.stamp.is_some() == m.cfg.window_clock.is_some()
                                        && h.seq < w.next_seq
                                })
                                && m.cfg.window_rows.is_none_or(|c| w.rows.len() <= c)
                        }
                        _ => false,
                    };
                if !shaped {
                    return Err(crate::model::damaged(
                        "realized",
                        "its running sums have the wrong shape",
                    ));
                }
                Ok(m)
            }
            other => Err(StateError::WrongModel {
                expected: "realized",
                found: other.kind(),
            }),
        }
    }

    fn n_targets(&self) -> usize {
        self.cfg.n_targets
    }

    /// The series are the targets; the model reads no features.
    fn n_features(&self) -> usize {
        0
    }

    fn n_outputs(&self) -> usize {
        OUTPUTS_PER_TARGET * self.cfg.n_targets
    }
}

#[cfg(test)]
mod paper;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod window_tests;
