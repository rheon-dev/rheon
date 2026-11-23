//! `ewm_moments`: exponentially weighted (or windowed) central moments and
//! cross-moments of columns, each leg optionally lagged, to any order, and
//! the statistics read from them -- raw moments, standardized moments
//! (skew, kurtosis, correlation), cumulants, the unbiased forms of orders 2
//! to 4 and each term's standard error -- from one bounded state, in one
//! pass.
//!
//! # The definitions
//!
//! A term names its legs, each a column and a lag, and a power of each:
//! `{x: 3}` is the third moment of `x`, `{x: 1, y: 2}` a cross-moment and
//! `{x: 1, x@1: 1}` the first autocovariance. A term is computed over the
//! stream of the rows at which every one of its legs has a value, `z_t`.
//! After row `T`, row `t` holds the weight `ω_t = w_t · Π_{s>t} λ_s`, its
//! own weight times every later row's decay, and with `W = Σ ω_t`:
//!
//! ```text
//! mean           m       = Σ ω_t z_t / W
//! central        μ_a     = Σ ω_t (z_t − m)^a / W           (a multi-index over the legs)
//! raw            μ'_a    = Σ ω_t z_t^a / W = Σ_{β≤a} C(a, β) m^(a−β) μ_β
//! standardized   s_a     = μ_a / Π_j μ_{2e_j}^(a_j/2)
//! cumulant       κ_p     = μ_p − Σ_{j=2}^{p−2} C(p−1, j−1) κ_j μ_{p−j}   (one leg)
//! ```
//!
//! These are frequency-weighted plug-in moments: with every weight 1 and no
//! decay they are scipy's `moment`, `skew(bias=True)` and
//! `kurtosis(bias=True)`, and a lagged cross-moment of order (1, 1) is
//! numpy's `cov(x[k:], y[:-k], bias=True)`.
//!
//! # The update
//!
//! Each group of terms over the same legs keeps one summary: `W`, each
//! leg's mean as a pair `m + m_lo` ([`crate::comp`]), the normalised
//! central co-moments `c_a = μ_a` for every `a` of order 2 or more below a
//! term's index, and Kish's ratio `r_2 = Σ ω_t² / W²` (`1 / r_2` is the
//! effective number of rows), with `r_3 = Σ ω³ / W³` and `r_4 = Σ ω⁴ / W⁴`
//! where an unbiased third or fourth moment is asked for. A row of weight
//! `w` after a decay `λ` is Pébay's merge (Pébay, Terriberry, Kolla and
//! Bennett 2016) of the decayed summary with the one-row summary `(w, z)`:
//!
//! ```text
//! W'   = λ W + w        a = λ W / W'        b = w / W'        (a + b = 1)
//! d    = z − m                                       (against the old means)
//! c'_a = Σ_{β≤a, |a−β|≠1} C(a, β) a c_{a−β} (−b d)^β  +  b (a d)^a     (c_0 = 1)
//! c'_a = a c_a + a b d_i d_j                         (order 2, `EwDiag`'s expression)
//! r'_k = a^k r_k + b^k
//! m'   = m + b d
//! ```
//!
//! Two summaries of disjoint rows, `A` before `B`, merge by the same rule
//! with `δ = m_B − m_A` in place of `d` and `B`'s moments beside `A`'s:
//! `c_a = Σ_β C(a, β) [a c^A_{a−β} (−b δ)^β + b c^B_{a−β} (a δ)^β]`,
//! `r_k = a^k r^A_k + b^k r^B_k`.
//!
//! Every term is a product of quantities at the scale of the spread -- the
//! deviation `d`, the shares `a` and `b` in `[0, 1]`, the past central
//! moments -- so nothing carries the level, and a column at `1e6` with a
//! spread of 1 keeps its fourth moment to about `1e-10`, where the raw
//! power sums keep no digit of it. Decay multiplies `W` alone: every `c_a`
//! and `r_k` is a ratio that a uniform rescaling of the weights leaves
//! where it was, so a long gap moves no moment and underflows none.
//!
//! Order 2 is [`crate::EwDiag`]'s update, operation for operation, so a
//! variance here is `ew_cov`'s to the bit; the higher orders are the merge
//! as written, each power formed by repeated multiplication.
//!
//! **A row of weight 0** decays `W` and changes nothing else: `b` is 0, so
//! no moment moves and no mean takes a step; it enters no lag buffer and no
//! window, and counts in no row lag. Where `W'` is 0 -- a zero-weight first
//! row, or every row so far of weight 0 -- `a` and `b` are both 0/0, so the
//! update stops at that condition and the state stays empty: no NaN enters
//! it. A row of weight 0 decays by the clock step it is handed, as
//! `ew_cov`'s does: on a clock its time passes.
//!
//! # Lags
//!
//! A lagged leg `x@k` reads `x` at `k` learned rows back, or, on a clock,
//! at the latest learned row at or before `t − k` (as-of, the rule of
//! Polars' `join_asof` with `strategy="backward"`), and is no value where
//! that row is more than `lag_tolerance` (by default `k` itself) before
//! `t − k`: after a gap longer than the lag a row closes no pair. Each
//! lagged column keeps one buffer of its last learned values, shared by
//! every lag and term that reads it: `K` values for a deepest row lag `K`,
//! and on a clock the rows inside the deepest lag's reach. A row whose
//! lagged leg has no value adds nothing to that term's summary (which
//! still decays) and counts for every other term. A session change or a
//! capped clock gap clears the buffers ([`crate::OnlineModel::clear_lags`]);
//! the moments go on.
//!
//! The lagged pairs are centred on **their own** weighted means, Pébay's
//! merge over the pair stream: the autocovariance at lag `k` is numpy's
//! `cov(x[k:], x[:-k], bias=True)` under equal weights, and the
//! autocorrelation (a standardized term of order (1, 1)) is the pairs'
//! Pearson correlation, river's `AutoCorr`, over both legs' variances on
//! the same pairs. `ew_lag_cov` centres both legs on the single stream's
//! mean before the row and reads the variance of the whole stream, which
//! is a different statistic: on an AR(1) stream the two autocorrelations
//! part by about 4e-2 at lag 2, and statsmodels' `acf` (the full sample's
//! mean, over `acovf_0`) is a third. A pair is weighed by the weight of
//! the row that closes it, as `ew_lag_cov` weighs one.
//!
//! # Windows
//!
//! Under `window_rows` the moments are of the latest learned rows; under
//! `window_clock` of the rows whose clock is in `(t − window, t]`, the edge
//! decided on the raw clocks as Polars' `rolling_*_by` decides it (one
//! subtraction, [`crate::Stamp::cmp_span`]). Rows are kept in a ring and
//! the window's summary is read through two stacks of merged summaries,
//! Chan, Golub and LeVeque's rule at every order (Pébay's merge above); a
//! row leaving the window is never subtracted, so nothing drifts. The ring
//! is cut into blocks of `b` rows, and a boundary `F` splits it: the back
//! is the running fold of the rows from `F` on, one row merged as each
//! arrives; the front holds, for each block before `F`, the merge of the
//! rows from its start to `F`, formed right to left once the front is
//! needed; and the head holds the same from each row of the block the
//! window starts in. A read is one merge, the head's summary from the
//! window's first row then the back. When the window's first row reaches
//! `F` and a block has filled since, `F` moves to the last block boundary
//! and the stacks begin again: each row is folded into the back once, into
//! its block once and into the head once, so a read costs a few merges
//! whatever the window's length (`b` sets only the memory, `w / b + b`
//! parts, least at `b = √w`). Measured at windows of 1,000 and 10,000
//! rows, on rows and on a clock, 4.2 to 4.3 µs a row (skew, kurtosis and
//! a correlation, release build), where merging the window's blocks on
//! each read took 54 and 178 µs on rows and 86 and 907 µs on a clock.
//!
//! Each part is decayed to the next part's end, so decay still weighs the
//! window's rows. Every summary the stacks hold is a function of the rows'
//! numbers and of `F`, which the state saves: the stacks are a cache,
//! rebuilt from the ring on loading, and the same rows give the same bits
//! however they were fed and wherever the state was saved.
//!
//! # The unbiased forms
//!
//! For rows drawn independently under weights with ratios `r_k`, the
//! plug-in moments' expectations are
//!
//! ```text
//! E[c_2]   = (1 − r_2) μ_2
//! E[c_3]   = (1 − 3 r_2 + 2 r_3) μ_3
//! E[c_4]   = (1 − 4 r_2 + 6 r_3 − 3 r_4) μ_4 + (6 r_2 − 6 r_3 + 9 r_4 − 9 r_2²) μ_2²
//! E[c_2²]  = (r_2 − 2 r_3 + r_4) μ_4 + (1 − 3 r_2 + 2 r_3 − 3 r_4 + 3 r_2²) μ_2²
//! ```
//!
//! and inverting them gives the unbiased `M_2` (Kish), `M_3` and `M_4`
//! (Rimoldini 2014, "Weighted skewness and kurtosis unbiased by sample
//! size and Gaussian uncertainties", for reliability weights) and an
//! unbiased `μ_2²`; the unbiased fourth cumulant is `M_4 − 3 (μ_2²)`. A
//! cross term of order 3 or 4 takes the same coefficients, polarised
//! (`μ_2²` becomes `(c_ij c_kl + c_ik c_jl + c_il c_jk) / 3` over its four
//! leg indices). With every weight 1 these are scipy's `kstat`,
//! `skew(bias=False)` and `kurtosis(bias=False)`. Above order 4 no unbiased
//! form is offered, and a term asking for one is refused by name.
//!
//! # The standard error
//!
//! Under `se`, each term `θ` has `se_θ`, its delta-method standard error at
//! Kish's effective size: `sqrt(r_2 · E[IF²])`, `IF` the term's influence
//! function as a polynomial in the legs' deviations `d`. For a central
//! moment `IF = d^a − μ_a − Σ_j a_j μ_{a−e_j} d_j`; a standardized moment's
//! and a cumulant's follow by the chain rule, a raw moment's is
//! `(m + d)^a − μ'_a` expanded. `E[IF²]` reads the moments to twice the
//! term's order, so the summaries keep them. At order 2 of one leg under
//! equal weights it is `sqrt((μ_4 − μ_2²) / n)`, the textbook form.
//!
//! # Withholding
//!
//! A term is withheld, with its reason ([`Withheld`]), where it is not
//! defined, and never by a count of rows: no weight yet; no lagged pair
//! yet; a weight below `min_weight`; a standardized moment of a leg with no
//! spread; an unbiased form whose denominator keeps no digit of the terms
//! it is formed from (one row carrying all the weight, or the weight on too
//! few rows, read from the weights themselves); a moment past a double's
//! range; a standard error that keeps no digit.
//!
//! # The state's size
//!
//! In doubles, `L` the distinct legs, `K_c` column `c`'s deepest row lag:
//!
//! ```text
//! the weight           1
//! a summary            1 + 2d + |kept| + |r_k|
//! the summaries        their sum            (none under a window)
//! a lag buffer         1 + K_c              (rows)
//!                      1 + 3 per row held   (clock: a stamp is two)
//! the clock's place    1                    (on a clock)
//! a window's ring      3 + (2 + L) a row    (+ 2 a row for its stamp on a clock)
//! its two stacks       (rows / b + b + 2)(Σ summaries + 2), not saved: a cache
//! ```
//!
//! Nothing grows with the stream: the summaries with the terms, a row
//! lag's buffer with the lag, a row window's ring with the window. Those
//! are checked against `max_state_bytes` before the first row. A clock
//! lag's buffer and a clock window's ring hold the rows inside a span of
//! the clock, whose number the data decides: they are checked as each row
//! goes in, a warning raised once past half of the limit and the run
//! refused at it ([`crate::OnlineModel::state_over_cap`]).

mod layout;

use std::cmp::Ordering;
use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

use crate::clock_span::{HeldCap, Place};
use crate::{Decay, Stamp};
use layout::{Group, Idx, Summary, add as idx_add, below, binom, order, sub as idx_sub, unit};

/// The default limit on the state, as the other models' `max_state_bytes`:
/// 256 KB.
pub const EWM_MOMENTS_MAX_STATE_BYTES: u64 = 256 * 1024;

/// The highest order an unbiased form reaches: 2 (Kish), 3 and 4
/// (Rimoldini). A term asking for one above it is refused by name.
pub const MAX_UNBIASED_MOMENT_ORDER: u32 = 4;

/// Rows a block of a clock window's ring covers: its two stacks hold at
/// most `rows / 16 + 18` parts.
pub const EWM_MOMENTS_CLOCK_BLOCK_ROWS: u64 = 16;

/// A denominator of an unbiased form within this many epsilons of the sum
/// of its terms' sizes keeps no digit of them, and the term is withheld.
const NO_DIGIT: f64 = 64.0 * f64::EPSILON;

/// Which statistic a term reads from the moments.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MomentKind {
    /// `μ_a`, about the weighted means.
    Central,
    /// `μ'_a`, about 0.
    Raw,
    /// `μ_a / Π_j μ_{2e_j}^(a_j/2)`: skew at 3, kurtosis (not excess) at
    /// 4, a correlation at (1, 1).
    Standardized,
    /// `κ_p` of one leg: the mean at 1, the variance at 2, excess kurtosis
    /// times `μ_2²` at 4.
    Cumulant,
}

/// A leg of a term: a column, read at the row itself or lagged.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Leg {
    /// The column's place among the model's features.
    pub col: usize,
    /// How far back: 0 for the row itself; otherwise learned rows (a whole
    /// number), or clock units under [`EwmMomentsCfg::clocked`].
    pub lag: f64,
}

impl Leg {
    /// The column at the row itself.
    pub fn of(col: usize) -> Self {
        Self { col, lag: 0.0 }
    }

    /// The column `lag` back.
    pub fn lagged(col: usize, lag: f64) -> Self {
        Self { col, lag }
    }

    fn key_cmp(&self, other: &Self) -> Ordering {
        self.col
            .cmp(&other.col)
            .then(self.lag.total_cmp(&other.lag))
    }
}

/// One output: a statistic of the term's legs at their powers, plug-in
/// (`bias = true`) or unbiased.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MomentTerm {
    /// The output's name, which every refusal of the term names.
    pub name: String,
    /// Each leg once, with its power (a whole number from 1).
    pub powers: Vec<(Leg, u32)>,
    pub kind: MomentKind,
    /// `true` for the plug-in moment, `false` for the unbiased form (orders
    /// 2 to 4; a raw moment is unbiased either way).
    pub bias: bool,
}

impl MomentTerm {
    /// A term of column 0 at order `order`.
    pub fn of_column(name: &str, order: u32, kind: MomentKind, bias: bool) -> Self {
        Self {
            name: name.to_owned(),
            powers: vec![(Leg::of(0), order)],
            kind,
            bias,
        }
    }

    /// The term's order: the sum of its powers.
    pub fn order(&self) -> u32 {
        self.powers.iter().map(|&(_, p)| p).sum()
    }

    /// Each statistic's default `bias`, as Polars': a variance (a central
    /// moment or cumulant of order 2) is unbiased, as `ewm_var`'s default;
    /// skew, kurtosis and every other order are the plug-in, as `skew` and
    /// `kurtosis` default.
    pub fn default_bias(kind: MomentKind, order: u32) -> bool {
        !(matches!(kind, MomentKind::Central | MomentKind::Cumulant) && order == 2)
    }

    /// The legs sorted by column and lag, with their powers.
    fn sorted(&self) -> (Vec<Leg>, Idx) {
        let mut p = self.powers.clone();
        p.sort_by(|a, b| a.0.key_cmp(&b.0));
        (
            p.iter().map(|&(l, _)| l).collect(),
            p.iter().map(|&(_, k)| k).collect(),
        )
    }
}

/// The settings of [`EwmMoments`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EwmMomentsCfg {
    pub decay: Decay,
    /// The outputs, in slot order.
    pub terms: Vec<MomentTerm>,
    /// Every term is withheld while its weight is below it.
    pub min_weight: f64,
    /// The limit on the state's bytes; `None` for
    /// [`EWM_MOMENTS_MAX_STATE_BYTES`].
    pub max_state_bytes: Option<u64>,
    /// The stream has a clock: lags and `window_clock` are in its units, a
    /// lag pairs a row with the latest at or before `t − lag`. Without one,
    /// lags are learned rows.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub clocked: bool,
    /// How far before `t − lag` the as-of row may be and still pair, in
    /// clock units; `None` for each lag's own length.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lag_tolerance: Option<f64>,
    /// Emit each term's standard error, `se_<term>`, after the terms.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub se: bool,
    /// The moments of the latest this many learned rows.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub window_rows: Option<usize>,
    /// The moments of the rows whose clock is in `(t − window, t]`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub window_clock: Option<f64>,
}

/// What the settings give the state, formed from them and so no part of
/// it: every two are equal.
#[derive(Debug, Clone, Default)]
struct Derived {
    /// One a distinct set of legs, in the order the terms first name them.
    groups: Vec<Group>,
    /// The weight's own group (no legs): a window's `W`.
    weight: Group,
    /// Each term's group and index in that group's leg order.
    terms: Vec<(usize, Idx)>,
    /// Each term's cumulant polynomial (empty unless a plug-in cumulant of
    /// order 2 or more).
    polys: Vec<Poly>,
    /// Every distinct leg, sorted: the ring's columns and the row's values.
    legs: Vec<Leg>,
    /// Each group's legs' places in `legs`.
    pos: Vec<Vec<usize>>,
    /// Each lagged column: the column, its deepest lag, and its reach (the
    /// longest a lag plus its tolerance).
    lagged: Vec<(usize, f64, f64)>,
    /// Each leg's buffer, for a lagged one.
    leg_buf: Vec<Option<usize>>,
}

impl PartialEq for Derived {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

impl EwmMomentsCfg {
    /// `terms` under `decay`, every other setting its default: no floor on
    /// the weight, the default limit, no clock, no standard error, no
    /// window.
    pub fn new(decay: Decay, terms: Vec<MomentTerm>) -> Self {
        Self {
            decay,
            terms,
            min_weight: 0.0,
            max_state_bytes: None,
            clocked: false,
            lag_tolerance: None,
            se: false,
            window_rows: None,
            window_clock: None,
        }
    }

    fn cap(&self) -> u64 {
        self.max_state_bytes.unwrap_or(EWM_MOMENTS_MAX_STATE_BYTES)
    }

    fn tolerance(&self, lag: f64) -> f64 {
        self.lag_tolerance.unwrap_or(lag)
    }

    fn windowed(&self) -> bool {
        self.window_rows.is_some() || self.window_clock.is_some()
    }

    /// Rows a block of the window's ring covers: `⌊√w⌋` on rows, which
    /// keeps the stacks' `w / b + b` parts least, and
    /// [`EWM_MOMENTS_CLOCK_BLOCK_ROWS`] on a clock. A read costs the same
    /// at any `b`.
    fn block_rows(&self) -> u64 {
        match self.window_rows {
            Some(w) => (w.isqrt() as u64).max(1),
            None => EWM_MOMENTS_CLOCK_BLOCK_ROWS,
        }
    }

    /// The layout the settings give: the groups, each term's place, the
    /// lagged columns.
    fn derive(&self) -> Derived {
        let mut d = Derived::default();
        let mut needs: Vec<(Vec<Idx>, Vec<u32>)> = Vec::new();
        let mut keys: Vec<Vec<Leg>> = Vec::new();
        for t in &self.terms {
            let (legs, alpha) = t.sorted();
            let gi = match keys.iter().position(|k| *k == legs) {
                Some(i) => i,
                None => {
                    keys.push(legs.clone());
                    needs.push((Vec::new(), vec![2]));
                    keys.len() - 1
                }
            };
            let n = legs.len();
            let ord = order(&alpha);
            let (need, powers) = &mut needs[gi];
            need.push(alpha.clone());
            if t.kind == MomentKind::Standardized {
                need.extend((0..n).map(|j| unit(n, j, 2)));
            }
            if !t.bias && t.kind != MomentKind::Raw {
                for k in [3, 4] {
                    if k <= ord && !powers.contains(&k) {
                        powers.push(k);
                    }
                }
                if ord == 4 {
                    need.extend((0..n).map(|j| unit(n, j, 2)));
                    need.extend(pairs(&alpha));
                }
            }
            if self.se {
                let sup = if_support(t.kind, &alpha);
                for x in &sup {
                    for y in &sup {
                        need.push(idx_add(x, y));
                    }
                }
            }
            d.terms.push((gi, alpha));
            d.polys
                .push(if t.kind == MomentKind::Cumulant && t.bias && ord >= 2 {
                    cumulant_poly(ord)
                } else {
                    Vec::new()
                });
        }
        for (legs, (need, mut powers)) in keys.into_iter().zip(needs) {
            powers.sort_unstable();
            d.groups.push(Group::new(legs, &need, powers));
        }
        d.weight = Group::new(Vec::new(), &[], Vec::new());
        let mut legs: Vec<Leg> = Vec::new();
        for g in &d.groups {
            for l in &g.legs {
                if !legs.contains(l) {
                    legs.push(*l);
                }
            }
        }
        legs.sort_by(Leg::key_cmp);
        d.pos = d
            .groups
            .iter()
            .map(|g| {
                g.legs
                    .iter()
                    .map(|l| legs.iter().position(|m| m == l).unwrap_or(0))
                    .collect()
            })
            .collect();
        for l in legs.iter().filter(|l| l.lag > 0.0) {
            let reach = l.lag + self.tolerance(l.lag);
            match d.lagged.iter_mut().find(|e| e.0 == l.col) {
                Some(e) => {
                    e.1 = e.1.max(l.lag);
                    e.2 = e.2.max(reach);
                }
                None => d.lagged.push((l.col, l.lag, reach)),
            }
        }
        d.leg_buf = legs
            .iter()
            .map(|l| {
                (l.lag > 0.0)
                    .then(|| d.lagged.iter().position(|e| e.0 == l.col))
                    .flatten()
            })
            .collect();
        d.legs = legs;
        d
    }

    /// The doubles of the state the settings fix, and the working doubles
    /// of a window's two stacks.
    fn fixed_numbers(&self, d: &Derived) -> (u64, u64) {
        let sums: u64 = d.groups.iter().map(Group::numbers).sum();
        let mut n = 1;
        if !self.windowed() {
            n += sums;
        }
        for &(_, k, _) in &d.lagged {
            n += 1;
            if !self.clocked {
                n += k as u64;
            }
        }
        if self.clocked {
            n += 1;
        }
        let mut working = 0;
        if self.windowed() {
            n += 3;
        }
        if let Some(w) = self.window_rows {
            let w = w as u64;
            n += w * (2 + d.legs.len() as u64);
            working = self.working(sums, w);
        } else if self.window_clock.is_some() {
            // An empty ring's: its rows are the data's, counted as they go in.
            working = self.working(sums, 0);
        }
        (n, working)
    }

    /// The doubles of a window's two stacks over `rows` rows, each part
    /// `sums` doubles and the weight's two: at most `rows / b` block
    /// suffixes in the front, `b` row suffixes in the head, the back and
    /// the part a read forms.
    fn working(&self, sums: u64, rows: u64) -> u64 {
        let b = self.block_rows();
        (rows / b + b + 2) * (sums + 2)
    }

    /// The state's bytes before the first row: what the settings fix.
    pub fn state_bytes(&self) -> u64 {
        let d = self.derive();
        let (n, working) = self.fixed_numbers(&d);
        8 * (n + working)
    }

    /// The column count the terms read: one past the highest column.
    pub fn n_columns(&self) -> usize {
        self.terms
            .iter()
            .flat_map(|t| t.powers.iter().map(|(l, _)| l.col + 1))
            .max()
            .unwrap_or(1)
    }

    pub fn validate(&self) -> Result<(), String> {
        let mut no = crate::Refusals::new();
        no.take(self.decay.check().map_err(|e| format!("ewm_moments: {e}")));
        if self.terms.is_empty() {
            no.refuse(
                "ewm_moments: terms names no term. Name at least one, such as a third central \
                 moment of order 3.",
            );
        }
        no.take(crate::model::check_min_weight(
            "ewm_moments",
            self.min_weight,
        ));
        if self.max_state_bytes == Some(0) {
            no.refuse(
                "ewm_moments: max_state_bytes must be above 0, but it is 0. Give a limit in \
                 bytes, or leave it unset for 256 KB.",
            );
        }
        self.validate_window(&mut no);
        for (i, t) in self.terms.iter().enumerate() {
            self.validate_term(i, t, &mut no);
        }
        no.stop_here()?;
        let d = self.derive();
        let (n, working) = self.fixed_numbers(&d);
        let (bytes, cap) = (8 * (n + working), self.cap());
        if bytes > cap {
            let top = d
                .groups
                .iter()
                .flat_map(|g| g.keep.iter().map(|a| order(a)))
                .max()
                .unwrap_or(1);
            no.refuse(format!(
                "ewm_moments: the state takes {bytes} bytes, over max_state_bytes ({cap}): \
                 its moments to order {top} in {} summaries, {} lag buffers and {} window \
                 rows are {} doubles. Ask for fewer terms, lower orders, shallower lags or a \
                 shorter window, or raise max_state_bytes.",
                d.groups.len(),
                d.lagged.len(),
                self.window_rows.unwrap_or(0),
                n + working,
            ));
        }
        no.finish()
    }

    fn validate_window(&self, no: &mut crate::Refusals) {
        if self.window_rows.is_some() && self.window_clock.is_some() {
            no.refuse(
                "ewm_moments: window_rows and window_clock are both given. Give one: a window of \
                 rows or of the clock.",
            );
        }
        if self.window_rows == Some(0) {
            no.refuse(
                "ewm_moments: window_rows must be at least 1, but it is 0. Give a window of at \
                 least one row, or leave window_rows out for no window.",
            );
        }
        if let Some(w) = self.window_clock {
            if !self.clocked {
                no.refuse(
                    "ewm_moments: window_clock needs a clock. Give a clock column, or a window \
                     of rows (window_rows).",
                );
            }
            if !(w > 0.0 && w.is_finite()) {
                no.refuse(format!(
                    "ewm_moments: window_clock must be a finite span above 0, but it is {w}. \
                     Give a finite span above 0, or leave it out for no window."
                ));
            }
        }
        if let Some(tol) = self.lag_tolerance {
            if !self.clocked {
                no.refuse(
                    "ewm_moments: lag_tolerance is the as-of tolerance of a lag on a clock, and \
                     this stream has none: a lag of rows pairs with exactly that row. Leave it \
                     out.",
                );
            }
            if !(tol >= 0.0 && tol.is_finite()) {
                no.refuse(format!(
                    "ewm_moments: lag_tolerance must be a finite span of at least 0, but it is \
                     {tol}. Give a finite span of at least 0, or leave it out for each lag's own \
                     length."
                ));
            }
        }
    }

    fn validate_term(&self, i: usize, t: &MomentTerm, no: &mut crate::Refusals) {
        let name = &t.name;
        if self.terms[..i].iter().any(|u| u.name == *name) {
            no.refuse(format!(
                "ewm_moments: the term {name:?} is named twice. Give each term its own name."
            ));
        }
        if t.powers.is_empty() {
            no.refuse(format!(
                "ewm_moments: the term {name:?} names no leg. Give it a column and a power."
            ));
            return;
        }
        for (j, &(leg, p)) in t.powers.iter().enumerate() {
            if p == 0 {
                no.refuse(format!(
                    "ewm_moments: the term {name:?} gives a leg the power 0, but a power is a \
                     whole number from 1. Give each leg a power of at least 1."
                ));
            }
            if t.powers[..j].iter().any(|(l, _)| l == &leg) {
                no.refuse(format!(
                    "ewm_moments: the term {name:?} names a leg twice. Give each leg once, \
                     with its power."
                ));
            }
            if !(leg.lag >= 0.0 && leg.lag.is_finite()) {
                no.refuse(format!(
                    "ewm_moments: the term {name:?} lags a leg by {}, but a lag is finite and at \
                     least 0. Give a lag of at least 0, or none for the row itself.",
                    leg.lag
                ));
            } else if !self.clocked && leg.lag.fract() != 0.0 {
                no.refuse(format!(
                    "ewm_moments: the term {name:?} lags a leg by {} rows, which is not a \
                     whole number. Give a whole number of rows, or a clock.",
                    leg.lag
                ));
            } else if !self.clocked && leg.lag > crate::MAX_LAG as f64 {
                no.refuse(format!(
                    "ewm_moments: the term {name:?} lags a leg by {} rows, past the most a lag \
                     of rows may be, {}. Give a shorter lag, or a clock and a lag of its units.",
                    leg.lag,
                    crate::MAX_LAG
                ));
            }
        }
        let ord = t.order();
        if ord == 0 {
            no.refuse(format!(
                "ewm_moments: the term {name:?} has order 0, but an order is a whole number \
                 from 1. Give its legs powers of at least 1."
            ));
        }
        if t.kind == MomentKind::Cumulant && t.powers.len() > 1 {
            no.refuse(format!(
                "ewm_moments: the term {name:?} asks for a joint cumulant of {} legs; a \
                 cumulant is of one leg. Ask for a central cross-moment instead.",
                t.powers.len()
            ));
        }
        if t.kind == MomentKind::Standardized && ord == 1 {
            no.refuse(format!(
                "ewm_moments: the term {name:?} is standardized at order 1, which is 0 by \
                 definition. Give it an order from 2."
            ));
        }
        if !t.bias && t.kind != MomentKind::Raw && ord > MAX_UNBIASED_MOMENT_ORDER {
            no.refuse(format!(
                "ewm_moments: the term {name:?} asks for bias=False at order {ord}; the \
                 unbiased forms stop at order {MAX_UNBIASED_MOMENT_ORDER}. Give bias=True for it."
            ));
        }
    }
}

/// The second-order indices a fourth-order `a`'s polarised `μ_2²` reads.
fn pairs(a: &[u32]) -> Vec<Idx> {
    let n = a.len();
    let idx: Vec<usize> = (0..n)
        .flat_map(|j| std::iter::repeat_n(j, a[j] as usize))
        .collect();
    let mut out = Vec::new();
    for &i in &idx {
        for &j in &idx {
            out.push(idx_add(&unit(n, i, 1), &unit(n, j, 1)));
        }
    }
    out
}

/// The multi-indices a term's influence function reaches.
fn if_support(kind: MomentKind, a: &[u32]) -> Vec<Idx> {
    let n = a.len();
    let mut sup: Vec<Idx> = below(a);
    if kind == MomentKind::Standardized {
        sup.extend((0..n).map(|j| unit(n, j, 2)));
    }
    sup
}

/// Why a term reports no value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Withheld {
    /// No row of weight has been learned.
    NoWeight,
    /// A lagged term with no pair learned yet.
    NoLaggedPair,
    /// The weight is below `min_weight`.
    BelowMinWeight,
    /// A standardized moment of a leg with no spread.
    ZeroSpread,
    /// An unbiased form when one row carries all the weight (`r_2 = 1`).
    OneRow,
    /// An unbiased third moment with the weight on too few rows.
    TooFewRowsThird,
    /// An unbiased fourth moment with the weight on too few rows.
    TooFewRowsFourth,
    /// The moment is past a double's range.
    Overflow,
    /// A standard error whose variance keeps no digit.
    NoStandardError,
}

impl Withheld {
    /// The reason, in the words an output gives it.
    pub fn reason(self) -> &'static str {
        match self {
            Self::NoWeight => "no weight yet",
            Self::NoLaggedPair => "no lagged pair yet",
            Self::BelowMinWeight => "weight_sum is below min_weight",
            Self::ZeroSpread => "the spread of the column is zero",
            Self::OneRow => "one row carries all the weight, so the unbiased form is not defined",
            Self::TooFewRowsThird => "the weight sits on too few rows for an unbiased third moment",
            Self::TooFewRowsFourth => {
                "the weight sits on too few rows for an unbiased fourth moment"
            }
            Self::Overflow => "the moment overflowed a double",
            Self::NoStandardError => "the standard error keeps no digit",
        }
    }
}

/// `κ_p` as a polynomial in the central moments: each monomial (the orders
/// of its factors, sorted) with its coefficient, in the order the recursion
/// first meets it.
type Poly = Vec<(Vec<u32>, f64)>;

/// `C(n, k)` for `k = 0..=n` in turn: exact while the integers fit, then
/// by the floating-point recurrence.
struct Binomials {
    n: u32,
    k: u32,
    exact: Option<u128>,
    approx: f64,
}

impl Binomials {
    fn new(n: u32) -> Self {
        Self {
            n,
            k: 0,
            exact: Some(1),
            approx: 1.0,
        }
    }

    /// `C(n, k)` at the current `k`.
    fn get(&self) -> f64 {
        match self.exact {
            Some(v) => v as f64,
            None => self.approx,
        }
    }

    /// Steps to `k + 1`.
    fn step(&mut self) {
        let (num, den) = (self.n - self.k, self.k + 1);
        self.approx = self.get() * f64::from(num) / f64::from(den);
        self.exact = self
            .exact
            .and_then(|v| v.checked_mul(u128::from(num)))
            .map(|v| v / u128::from(den));
        self.k += 1;
    }
}

fn binomial(n: u32, k: u32) -> f64 {
    let mut b = Binomials::new(n);
    for _ in 0..k {
        b.step();
    }
    b.get()
}

/// `κ_p` (`p >= 2`) by the recursion of the module docs, expanded into
/// central moments.
fn cumulant_poly(p: u32) -> Poly {
    let mut k: Vec<Poly> = vec![
        Vec::new(),
        Vec::new(),
        vec![(vec![2], 1.0)],
        vec![(vec![3], 1.0)],
    ];
    for n in 4..=p {
        let mut poly: Poly = vec![(vec![n], 1.0)];
        for j in 2..n - 1 {
            let c = binomial(n - 1, j - 1);
            for (mono, v) in &k[j as usize] {
                let mut key = mono.clone();
                key.push(n - j);
                key.sort_unstable();
                match poly.iter_mut().find(|(m, _)| *m == key) {
                    Some((_, e)) => *e -= c * v,
                    None => poly.push((key, 0.0 - c * v)),
                }
            }
        }
        k.push(poly);
    }
    k.swap_remove(p as usize)
}

/// A lagged column's last learned values (and their stamps, on a clock).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct LagBuf {
    col: usize,
    vals: VecDeque<f64>,
    #[serde(default, skip_serializing_if = "VecDeque::is_empty")]
    stamps: VecDeque<Stamp>,
}

/// A window's rows: each row's decay from the row before it, its weight
/// and every leg's value (NaN where a lagged leg has none), and on a clock
/// its stamp. `first` numbers the oldest row among every learned row;
/// `pending` is the decay rows of weight 0 have passed since the last.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Ring {
    first: u64,
    pending: f64,
    /// `F`, where the window's back stack starts: a block boundary, the
    /// rows before it read through the front stack's suffixes and the rows
    /// from it through the back's running fold. Saved, since the window's
    /// summary is a function of the rows and of `F`.
    flip: u64,
    vals: VecDeque<f64>,
    #[serde(default, skip_serializing_if = "VecDeque::is_empty")]
    stamps: VecDeque<Stamp>,
}

/// A run of a window's rows: every group's summary (the weight's last) and
/// the product of the rows' decays.
#[derive(Debug, Clone)]
struct Part {
    sums: Vec<Summary>,
    lp: f64,
}

impl Part {
    fn empty(d: &Derived) -> Self {
        let mut sums: Vec<Summary> = d.groups.iter().map(Summary::empty).collect();
        sums.push(Summary::empty(&d.weight));
        Self { sums, lp: 1.0 }
    }

    /// The rows of `self` followed by those of `later`: each summary
    /// decayed by the later rows' decays, then Pébay's merge.
    fn then(&mut self, d: &Derived, later: &Part) {
        for (i, (s, p)) in self.sums.iter_mut().zip(&later.sums).enumerate() {
            let g = d.groups.get(i).unwrap_or(&d.weight);
            s.decay(later.lp);
            s.merge(g, p);
        }
        self.lp *= later.lp;
    }
}

/// A window's two stacks (the cache the window is read through, rebuilt
/// from the ring and [`Ring::flip`], so neither saved nor compared). With
/// `F` the flip and `b` the block's rows: the front holds, for each full
/// block `j` before `F`, the summary of the rows from `j b` to `F`; the
/// head holds, for the block the window starts in, the summary of the rows
/// from each of its rows to `F`; the back is the running fold of the rows
/// from `F` on. A read is one merge, the head's row then the back.
#[derive(Debug, Clone, Default)]
struct Stacks {
    /// `front[k]`: the rows from block `front_lo + k` to `F`.
    front: VecDeque<Part>,
    front_lo: u64,
    /// The block the head is of, and `head[k]`: the rows from `head_lo + k`
    /// to `F`.
    head_block: Option<u64>,
    head: VecDeque<Part>,
    head_lo: u64,
    /// The rows from `F` to the ring's end; `None` once rows before `F`
    /// have left the ring (it is not read again before the next flip).
    back: Option<Part>,
}

impl PartialEq for Stacks {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

/// Exponentially weighted or windowed moments of columns and their lags,
/// to any order. See the module docs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EwmMoments {
    cfg: EwmMomentsCfg,
    /// `W`, the decayed sum of every learned row's weight (the `n_eff`
    /// count; a window reads its own).
    w_sum: f64,
    /// One summary a group (none under a window).
    sums: Vec<Summary>,
    /// One buffer a lagged column.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    bufs: Vec<LagBuf>,
    /// The window's rows.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    ring: Option<Ring>,
    /// The place on the clock, on a clock.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    place: Option<Place>,
    /// The bytes a row would have taken a clock span's state to, past the
    /// cap ([`crate::clock_span::HeldCap`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    over: Option<u64>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    warned: bool,
    #[serde(skip)]
    warning: Option<u64>,
    #[serde(skip)]
    derived: Derived,
    #[serde(skip)]
    stacks: Stacks,
}

impl EwmMoments {
    pub fn new(cfg: EwmMomentsCfg) -> Result<Self, String> {
        cfg.validate()?;
        let derived = cfg.derive();
        let sums = if cfg.windowed() {
            Vec::new()
        } else {
            derived.groups.iter().map(Summary::empty).collect()
        };
        let bufs = derived
            .lagged
            .iter()
            .map(|&(col, _, _)| LagBuf {
                col,
                vals: VecDeque::new(),
                stamps: VecDeque::new(),
            })
            .collect();
        let ring = cfg.windowed().then(|| Ring {
            first: 0,
            pending: 1.0,
            flip: 0,
            vals: VecDeque::new(),
            stamps: VecDeque::new(),
        });
        let place = cfg.clocked.then(Place::default);
        Ok(Self {
            cfg,
            w_sum: 0.0,
            sums,
            bufs,
            ring,
            place,
            over: None,
            warned: false,
            warning: None,
            derived,
            stacks: Stacks::default(),
        })
    }

    pub fn cfg(&self) -> &EwmMomentsCfg {
        &self.cfg
    }

    /// `W`: the weight before the next row and its decay (hard rule 8);
    /// under a window, the window's.
    #[inline]
    pub fn n_eff(&self) -> f64 {
        if self.cfg.windowed() {
            self.window_at(None).1.w
        } else {
            self.w_sum
        }
    }

    /// The first group's summary, as it stands.
    fn first(&self) -> Summary {
        match self.cfg.windowed() {
            true => self.window_at(None).0.swap_remove(0),
            false => self.sums[0].clone(),
        }
    }

    /// The first group's first leg's mean, the pair's high part.
    pub fn mean(&self) -> f64 {
        self.first().m[0]
    }

    /// The first group's plug-in central moment of order `p` of its first
    /// leg (0 at order 1; NaN where the state keeps none of that order).
    pub fn central(&self, p: u32) -> f64 {
        let g = &self.derived.groups[0];
        let a = unit(g.legs.len(), 0, p);
        self.first().cv(g, &a)
    }

    /// The first group's Kish ratio `r_2 = Σ ω² / W²`, `1 / n_eff_kish`.
    pub fn kish_ratio(&self) -> f64 {
        self.first().r[0]
    }

    /// The state's bytes as it stands: what the settings fix, the rows a
    /// clock span holds, and a window's two stacks.
    pub fn state_bytes(&self) -> u64 {
        let (lags, ring) = self.held_rows();
        self.bytes_for(&lags, ring)
    }

    /// The rows each clock lag buffer and a clock window's ring hold.
    fn held_rows(&self) -> (Vec<usize>, usize) {
        let lags = self.bufs.iter().map(|b| b.vals.len()).collect();
        let ring = self.ring.as_ref().map_or(0, |r| r.stamps.len());
        (lags, ring)
    }

    /// The bytes the state takes with `lags` rows in each clock lag buffer
    /// and `ring` rows in a clock window's ring.
    fn bytes_for(&self, lags: &[usize], ring: usize) -> u64 {
        let (mut n, mut working) = self.cfg.fixed_numbers(&self.derived);
        if self.cfg.clocked {
            n += 3 * lags.iter().sum::<usize>() as u64;
        }
        if self.cfg.window_clock.is_some() {
            let sums: u64 = self.derived.groups.iter().map(Group::numbers).sum();
            n += ring as u64 * (4 + self.derived.legs.len() as u64);
            working = self.cfg.working(sums, ring as u64);
        }
        8 * (n + working)
    }

    /// The bytes the state's encoding holds as numbers: [`Self::state_bytes`]
    /// less the window's two stacks.
    #[cfg(test)]
    fn saved_bytes(&self) -> u64 {
        let sums: u64 = self.derived.groups.iter().map(Group::numbers).sum();
        let working = match (self.cfg.window_rows, &self.ring) {
            (Some(w), _) => self.cfg.working(sums, w as u64),
            (None, Some(r)) => self.cfg.working(sums, r.stamps.len() as u64),
            _ => 0,
        };
        self.state_bytes() - 8 * working
    }

    /// Whether a clock span holds rows the data numbers: a lag buffer or a
    /// window ring on a clock.
    fn data_sized(&self) -> bool {
        self.cfg.clocked && (!self.bufs.is_empty() || self.cfg.window_clock.is_some())
    }

    /// Whether the state's vectors are at the widths its settings give them.
    pub(crate) fn has_shape(&self) -> bool {
        let d = &self.derived;
        let sums_ok = if self.cfg.windowed() {
            self.sums.is_empty()
        } else {
            self.sums.len() == d.groups.len()
                && self.sums.iter().zip(&d.groups).all(|(s, g)| s.has_shape(g))
        };
        let bufs_ok = self.bufs.len() == d.lagged.len()
            && self.bufs.iter().zip(&d.lagged).all(|(b, &(col, k, _))| {
                b.col == col
                    && if self.cfg.clocked {
                        b.stamps.len() == b.vals.len()
                    } else {
                        b.stamps.is_empty() && b.vals.len() <= k as usize
                    }
            });
        let ring_ok = match (&self.ring, self.cfg.windowed()) {
            (None, false) => true,
            (Some(r), true) => {
                let stride = 2 + d.legs.len();
                let rows = r.vals.len() / stride;
                r.vals.len() % stride == 0
                    && r.flip <= r.first + rows as u64
                    && r.flip % self.cfg.block_rows() == 0
                    && if self.cfg.window_clock.is_some() {
                        r.stamps.len() == rows
                    } else {
                        r.stamps.is_empty() && rows <= self.cfg.window_rows.unwrap_or(0)
                    }
            }
            _ => false,
        };
        sums_ok && bufs_ok && ring_ok && self.place.is_some() == self.cfg.clocked
    }

    /// One row of a single column of weight `w` after a decay of `lam`
    /// (from [`Decay::factor`]), on a stream without a clock. A negative
    /// weight is a caller's bug (a debug assert, and no change in release);
    /// a row that leaves the weight at 0 changes nothing else (hard rule
    /// 9).
    pub fn update(&mut self, x: f64, lam: f64, w: f64) {
        self.learn(&[x], None, lam, w);
    }

    /// One row of every column, `d_clock` after the last, of weight `w`.
    pub fn update_row(&mut self, x: &[f64], d_clock: f64, w: f64) {
        let now = self.place.as_mut().map(|p| p.step(d_clock));
        let lam = self.cfg.decay.factor(d_clock);
        self.learn(x, now, lam, w);
    }

    fn learn(&mut self, x: &[f64], now: Option<Stamp>, lam: f64, w: f64) {
        debug_assert!(
            w >= 0.0,
            "EwmMoments::update needs a weight of at least 0, got {w}"
        );
        if w < 0.0 || self.over.is_some() {
            return;
        }
        if w == 0.0 {
            self.zero_row(lam);
            return;
        }
        if self.data_sized() {
            let now = now.unwrap_or(Stamp::from(0.0));
            let (lags, ring) = self.rows_with(now);
            let with = self.bytes_for(&lags, ring);
            let cap = self.cfg.cap();
            if !self.held_cap().admit(with, cap) {
                return;
            }
        }
        let d = &self.derived;
        let z: Vec<f64> = d
            .legs
            .iter()
            .zip(&d.leg_buf)
            .map(|(l, b)| match b {
                None => x.get(l.col).copied().unwrap_or(f64::NAN),
                Some(i) => self.partner(*i, l.lag, now),
            })
            .collect();
        for i in 0..self.bufs.len() {
            let col = self.bufs[i].col;
            let v = x.get(col).copied().unwrap_or(f64::NAN);
            self.push_lag(i, v, now);
        }
        if self.ring.is_some() {
            self.push_ring(lam, w, &z, now);
        } else {
            let d = &self.derived;
            let mut zg = Vec::new();
            for ((s, g), pos) in self.sums.iter_mut().zip(&d.groups).zip(&d.pos) {
                zg.clear();
                zg.extend(pos.iter().map(|&p| z[p]));
                let w_g = if zg.iter().all(|v| v.is_finite()) {
                    w
                } else {
                    0.0
                };
                s.learn(g, lam, w_g, &zg);
            }
        }
        self.w_sum = lam * self.w_sum + w;
        if self.data_sized() {
            let (bytes, cap) = (self.state_bytes(), self.cfg.cap());
            self.held_cap().note(bytes, cap);
        }
    }

    /// A row of weight 0: the decay alone.
    fn zero_row(&mut self, lam: f64) {
        match &mut self.ring {
            Some(r) => r.pending *= lam,
            None => {
                for (s, g) in self.sums.iter_mut().zip(&self.derived.groups) {
                    s.learn(g, lam, 0.0, &[]);
                }
            }
        }
        self.w_sum *= lam;
    }

    fn held_cap(&mut self) -> HeldCap<'_> {
        HeldCap {
            over: &mut self.over,
            warned: &mut self.warned,
            warning: &mut self.warning,
        }
    }

    /// The value of buffer `i`'s column `lag` back: the `lag`-th learned
    /// row back, or on a clock the latest row at or before `now − lag`,
    /// none where that row is more than the tolerance before it.
    fn partner(&self, i: usize, lag: f64, now: Option<Stamp>) -> f64 {
        let b = &self.bufs[i];
        match now {
            None => {
                let k = lag as usize;
                if b.vals.len() >= k {
                    b.vals[b.vals.len() - k]
                } else {
                    f64::NAN
                }
            }
            Some(now) => {
                let reach = lag + self.cfg.tolerance(lag);
                for (t, v) in b.stamps.iter().zip(&b.vals).rev() {
                    if now.cmp_span(*t, lag) != Ordering::Less {
                        return if now.cmp_span(*t, reach) == Ordering::Greater {
                            f64::NAN
                        } else {
                            *v
                        };
                    }
                }
                f64::NAN
            }
        }
    }

    /// How many of buffer `i`'s oldest rows no later row can pair with,
    /// once a row at `now` is in it: on a clock, every row before the
    /// latest at or before `now − K` (a later as-of point is never before
    /// it), and any more than the column's reach before `now`. The row at
    /// `now` is neither (every lag is above 0), so the count is the same
    /// before it is pushed and after.
    fn lag_drops(&self, i: usize, now: Stamp) -> usize {
        let b = &self.bufs[i];
        let (_, k, reach) = self.derived.lagged[i];
        let mut keep_from = b
            .stamps
            .iter()
            .rposition(|t| now.cmp_span(*t, k) != Ordering::Less)
            .unwrap_or(0);
        while keep_from < b.stamps.len()
            && now.cmp_span(b.stamps[keep_from], reach) == Ordering::Greater
        {
            keep_from += 1;
        }
        keep_from
    }

    fn push_lag(&mut self, i: usize, v: f64, now: Option<Stamp>) {
        match now {
            None => {
                let k = self.derived.lagged[i].1 as usize;
                let b = &mut self.bufs[i];
                b.vals.push_back(v);
                while b.vals.len() > k {
                    b.vals.pop_front();
                }
            }
            Some(now) => {
                self.bufs[i].vals.push_back(v);
                self.bufs[i].stamps.push_back(now);
                let drop = self.lag_drops(i, now);
                let b = &mut self.bufs[i];
                b.vals.drain(..drop);
                b.stamps.drain(..drop);
            }
        }
    }

    /// The rows each lag buffer and a clock window's ring would hold with
    /// a row at `now` taken.
    fn rows_with(&self, now: Stamp) -> (Vec<usize>, usize) {
        let lags = (0..self.bufs.len())
            .map(|i| self.bufs[i].stamps.len() + 1 - self.lag_drops(i, now))
            .collect();
        let ring = match (&self.ring, self.cfg.window_clock) {
            (Some(r), Some(span)) => {
                let gone = r
                    .stamps
                    .iter()
                    .take_while(|t| now.cmp_span(**t, span) != Ordering::Less)
                    .count();
                r.stamps.len() + 1 - gone
            }
            _ => 0,
        };
        (lags, ring)
    }

    fn stride(&self) -> usize {
        2 + self.derived.legs.len()
    }

    fn push_ring(&mut self, lam: f64, w: f64, z: &[f64], now: Option<Stamp>) {
        let stride = self.stride();
        let span = self.cfg.window_clock;
        let rows = self.cfg.window_rows;
        let r = self.ring.as_mut().expect("a window keeps a ring");
        r.vals.push_back(r.pending * lam);
        r.vals.push_back(w);
        r.vals.extend(z.iter().copied());
        r.pending = 1.0;
        if let (Some(span), Some(now)) = (span, now) {
            r.stamps.push_back(now);
            while let Some(t) = r.stamps.front() {
                if now.cmp_span(*t, span) == Ordering::Less {
                    break;
                }
                r.stamps.pop_front();
                r.vals.drain(..stride);
                r.first += 1;
            }
        }
        if let Some(n) = rows {
            while r.vals.len() / stride > n {
                r.vals.drain(..stride);
                r.first += 1;
            }
        }
        let (first, end) = (r.first, r.first + (r.vals.len() / stride) as u64);
        let flip = r.flip;
        if flip < first {
            self.stacks.back = None;
        }
        if let Some(mut back) = self.stacks.back.take() {
            self.learn_row(&mut back, end - 1);
            self.stacks.back = Some(back);
        }
    }

    /// Row `n` (numbered among every learned row) into `part`: each group's
    /// summary after the row's decay, the product of the decays.
    fn learn_row(&self, part: &mut Part, n: u64) {
        let d = &self.derived;
        let r = self.ring.as_ref().expect("a window keeps a ring");
        let stride = self.stride();
        let at = (n - r.first) as usize * stride;
        let (lam, w) = (r.vals[at], r.vals[at + 1]);
        part.lp *= lam;
        let mut zg = Vec::new();
        for ((s, g), pos) in part.sums.iter_mut().zip(&d.groups).zip(&d.pos) {
            zg.clear();
            zg.extend(pos.iter().map(|&p| r.vals[at + 2 + p]));
            let w_g = if zg.iter().all(|v| v.is_finite()) {
                w
            } else {
                0.0
            };
            s.learn(g, lam, w_g, &zg);
        }
        if let Some(s) = part.sums.last_mut() {
            s.learn(&d.weight, lam, w, &[]);
        }
    }

    /// Every group's summary (the weight's last) of the rows `lo..hi`
    /// (numbered among every learned row) folded in order, and the product
    /// of their decays.
    fn fold(&self, lo: u64, hi: u64) -> Part {
        let mut part = Part::empty(&self.derived);
        for n in lo..hi {
            self.learn_row(&mut part, n);
        }
        part
    }

    /// Brings the stacks to a read of the rows `lo..hi`: where the front is
    /// spent (`lo` at or past `F`) and a block has filled since, `F` moves
    /// to the last block boundary and the stacks are begun again; then the
    /// head is filled down to `lo`. Every summary the stacks hold is a
    /// function of the rows and `F` alone.
    fn prepare(&self, st: &mut Stacks, flip: &mut u64, lo: u64, hi: u64) {
        let d = &self.derived;
        let b = self.cfg.block_rows();
        if lo >= *flip {
            // A boundary after `lo`; none is a window inside one block,
            // read from its rows.
            let f = hi / b * b;
            if f > lo {
                *flip = f;
                st.front.clear();
                st.front_lo = f / b;
                st.head.clear();
                st.head_block = None;
                st.back = Some(self.fold(f, hi));
            }
        }
        if lo >= *flip {
            return;
        }
        let jh = lo / b;
        if st.head_block != Some(jh) {
            st.head.clear();
            st.head_block = Some(jh);
            st.head_lo = (jh + 1) * b;
        }
        if st.head_lo > lo {
            if st.head.is_empty() {
                // The front from the block after the head's to `F` (none
                // where that block is `F`'s).
                if st.front.is_empty() {
                    st.front_lo = *flip / b;
                }
                while st.front_lo < jh + 1 && !st.front.is_empty() {
                    st.front.pop_front();
                    st.front_lo += 1;
                }
                while st.front_lo > jh + 1 {
                    let k = st.front_lo - 1;
                    let mut p = self.fold(k * b, (k + 1) * b);
                    if let Some(next) = st.front.front() {
                        p.then(d, next);
                    }
                    st.front.push_front(p);
                    st.front_lo = k;
                }
            }
            while st.head_lo > lo {
                let n = st.head_lo - 1;
                let mut p = self.fold(n, n + 1);
                if let Some(next) = st.head.front().or(st.front.front()) {
                    p.then(d, next);
                }
                st.head.push_front(p);
                st.head_lo = n;
            }
            // The head holds the block after it: the front's next is the
            // one after that.
            while st.front_lo < jh + 2 && !st.front.is_empty() {
                st.front.pop_front();
                st.front_lo += 1;
            }
        }
        while st.head_lo < lo && st.head.len() > 1 {
            st.head.pop_front();
            st.head_lo += 1;
        }
    }

    /// The window's rows `lo..hi` read through prepared stacks.
    fn read_stacks(&self, st: &Stacks, flip: u64, lo: u64, hi: u64) -> Part {
        if lo >= flip {
            return self.fold(lo, hi);
        }
        let mut p = st.head[(lo - st.head_lo) as usize].clone();
        match &st.back {
            Some(back) => p.then(&self.derived, back),
            None => p.then(&self.derived, &self.fold(flip, hi)),
        }
        p
    }

    /// Whether the stacks are ready for a read from `lo` as they stand.
    fn stacks_ready(&self, flip: u64, lo: u64) -> bool {
        let st = &self.stacks;
        let b = self.cfg.block_rows();
        lo < flip && st.head_block == Some(lo / b) && st.head_lo <= lo
    }

    /// The window's rows `lo..hi` through the stacks, as `prepare` then a
    /// read give them, without moving the stacks.
    fn window_stacks(&self, lo: u64, hi: u64) -> Part {
        let flip = self.ring.as_ref().map_or(0, |r| r.flip);
        if self.stacks_ready(flip, lo) {
            return self.read_stacks(&self.stacks, flip, lo, hi);
        }
        let (mut st, mut f) = (self.stacks.clone(), flip);
        self.prepare(&mut st, &mut f, lo, hi);
        self.read_stacks(&st, f, lo, hi)
    }

    /// `prepare` on the model's own stacks, before a step's read.
    fn slide(&mut self, now: Option<Stamp>) {
        let (lo, hi) = self.window_span(now);
        let Some(mut flip) = self.ring.as_ref().map(|r| r.flip) else {
            return;
        };
        let mut st = std::mem::take(&mut self.stacks);
        self.prepare(&mut st, &mut flip, lo, hi);
        self.stacks = st;
        if let Some(r) = self.ring.as_mut() {
            r.flip = flip;
        }
    }

    /// The back stack from the ring, as a state is loaded.
    fn rebuild_back(&mut self) {
        let Some(r) = &self.ring else { return };
        let end = r.first + (r.vals.len() / self.stride()) as u64;
        let back = (r.flip >= r.first && r.flip <= end).then(|| self.fold(r.flip, end));
        self.stacks = Stacks {
            back,
            ..Stacks::default()
        };
    }

    /// The window's rows at `now`: the rows numbered `lo..hi`.
    fn window_span(&self, now: Option<Stamp>) -> (u64, u64) {
        let r = self.ring.as_ref().expect("a window keeps a ring");
        let end = r.first + (r.vals.len() / self.stride()) as u64;
        let mut lo = r.first;
        if let (Some(span), Some(now)) = (self.cfg.window_clock, now) {
            lo += r
                .stamps
                .iter()
                .take_while(|t| now.cmp_span(**t, span) != Ordering::Less)
                .count() as u64;
        }
        (lo, end)
    }

    /// Every group's window summary at `now` (`None`: the ring as it
    /// stands), and the weight's.
    fn window_at(&self, now: Option<Stamp>) -> (Vec<Summary>, Summary) {
        let (lo, hi) = self.window_span(now);
        let mut all = self.window_stacks(lo, hi).sums;
        let weight = all.pop().expect("the weight's summary");
        (all, weight)
    }

    /// Every output, a term then (under `se`) each standard error after
    /// the terms, read from `sums`; and the weight.
    fn read_sums(&self, sums: &[Summary]) -> Vec<Result<f64, Withheld>> {
        let n = self.cfg.terms.len();
        let mut out: Vec<Result<f64, Withheld>> = (0..n).map(|i| self.value_in(i, sums)).collect();
        if self.cfg.se {
            out.extend((0..n).map(|i| self.se_in(i, sums)));
        }
        out
    }

    /// Every output's value, or why it is withheld, read from the state as
    /// it stands: the terms, then under `se` their standard errors.
    pub fn read(&self) -> Vec<Result<f64, Withheld>> {
        if self.cfg.windowed() {
            let (sums, _) = self.window_at(None);
            self.read_sums(&sums)
        } else {
            self.read_sums(&self.sums)
        }
    }

    /// Every output at the row at `now` into `out`, a withheld one as NaN;
    /// and the weight.
    fn read_into(&self, now: Option<Stamp>, out: &mut Vec<f64>) -> f64 {
        out.clear();
        if self.cfg.windowed() {
            let (sums, weight) = self.window_at(now);
            out.extend(
                self.read_sums(&sums)
                    .into_iter()
                    .map(|v| v.unwrap_or(f64::NAN)),
            );
            weight.w
        } else {
            out.extend(
                self.read_sums(&self.sums)
                    .into_iter()
                    .map(|v| v.unwrap_or(f64::NAN)),
            );
            self.w_sum
        }
    }

    /// [`crate::OnlineModel::predict`] for a row whose place on the clock
    /// the caller knows, `stamp`, as [`crate::OnlineModel::stamp_next`]
    /// hands a step's: a window of the clock is read with its edge there.
    /// `predict` reads it at the clock summed from the steps, which begins
    /// at the model's first row where a caller's stamps begin at the
    /// clock's, so a caller that stamps its steps reads its rows of weight
    /// 0, and the rows it only scores, here. Without a clock the stamp is
    /// not read.
    pub fn predict_at(&self, x: &[f64], d_clock: f64, stamp: Stamp) -> crate::Step {
        if let Some(refused) = crate::model::refused_predict(self, x, d_clock) {
            return refused;
        }
        let now = self.place.as_ref().map(|_| stamp);
        let mut out = crate::Step::default();
        out.n_eff = self.read_into(now, &mut out.pred);
        out
    }

    /// One row learned at `stamp`, `d_clock` after the last row learned, of
    /// weight `w`, with nothing read: the learning half of
    /// [`crate::OnlineModel::step`], for a caller that reads the moments
    /// after the row (a window operator, closed on the right) and not before
    /// it. The row goes in as `step` takes it, the window slid to `stamp`
    /// first, so the state is the one `step` leaves, to the bit.
    pub fn learn_at(&mut self, x: &[f64], d_clock: f64, w: f64, stamp: Stamp) {
        if let Some(p) = &mut self.place {
            p.hand(stamp);
        }
        let now = self.place.as_mut().map(|p| p.step(d_clock));
        if self.ring.is_some() {
            self.slide(now);
        }
        let lam = self.cfg.decay.factor(d_clock);
        self.learn(x, now, lam, w);
    }

    /// Every output read at `stamp`, a window of the clock with its edge
    /// there: [`Self::predict_at`]'s values, with no row to score.
    pub fn read_at(&self, stamp: Stamp) -> Vec<Result<f64, Withheld>> {
        if self.cfg.windowed() {
            let now = self.place.as_ref().map(|_| stamp);
            let (sums, _) = self.window_at(now);
            self.read_sums(&sums)
        } else {
            self.read_sums(&self.sums)
        }
    }

    /// A state read back from its serialized form, as a state file's is:
    /// the settings checked and what they give the state formed again.
    ///
    /// # Errors
    ///
    /// Settings `new` refuses, or vectors at the wrong widths.
    pub fn reloaded(self) -> Result<Self, String> {
        self.checked().map_err(|e| e.to_string())
    }

    /// Term `i`'s value, or why it is withheld.
    pub fn value(&self, i: usize) -> Result<f64, Withheld> {
        if self.cfg.windowed() {
            let (sums, _) = self.window_at(None);
            self.value_in(i, &sums)
        } else {
            self.value_in(i, &self.sums)
        }
    }

    /// The gate every output passes: a weight, and `min_weight`.
    fn gate(&self, i: usize, s: &Summary) -> Result<(), Withheld> {
        if s.w == 0.0 {
            let lagged = self.derived.groups[self.derived.terms[i].0].lagged();
            return Err(if lagged {
                Withheld::NoLaggedPair
            } else {
                Withheld::NoWeight
            });
        }
        if s.w < self.cfg.min_weight {
            return Err(Withheld::BelowMinWeight);
        }
        Ok(())
    }

    fn value_in(&self, i: usize, sums: &[Summary]) -> Result<f64, Withheld> {
        let gi = self.derived.terms[i].0;
        let s = &sums[gi];
        self.gate(i, s)?;
        let v = self.evaluate(i, s)?;
        if v.is_finite() {
            Ok(v)
        } else {
            Err(Withheld::Overflow)
        }
    }

    fn evaluate(&self, i: usize, s: &Summary) -> Result<f64, Withheld> {
        let t = &self.cfg.terms[i];
        let (gi, a) = &self.derived.terms[i];
        let g = &self.derived.groups[*gi];
        let p = order(a);
        match t.kind {
            MomentKind::Raw => Ok(raw(g, s, a)),
            MomentKind::Central if p == 1 => Ok(0.0),
            MomentKind::Central if t.bias => Ok(s.cv(g, a)),
            MomentKind::Central => Ok(unbiased(g, s, a)?.0),
            MomentKind::Standardized => standardized(g, s, a, t.bias),
            MomentKind::Cumulant if p == 1 => Ok(s.m[0]),
            MomentKind::Cumulant if !t.bias => {
                let (v, u22) = unbiased(g, s, a)?;
                Ok(if p == 4 { v - 3.0 * u22 } else { v })
            }
            MomentKind::Cumulant => {
                let mut val = 0.0;
                for (mono, coef) in &self.derived.polys[i] {
                    let mut term = *coef;
                    for &q in mono {
                        term *= s.cv(g, &[q]);
                    }
                    val += term;
                }
                Ok(val)
            }
        }
    }

    /// Term `i`'s standard error, or why it has none.
    fn se_in(&self, i: usize, sums: &[Summary]) -> Result<f64, Withheld> {
        let (gi, _) = &self.derived.terms[i];
        let g = &self.derived.groups[*gi];
        let s = &sums[*gi];
        if s.w == 0.0 {
            return Err(Withheld::NoWeight);
        }
        if s.w < self.cfg.min_weight {
            return Err(Withheld::BelowMinWeight);
        }
        self.evaluate(i, s)?;
        let poly = self.influence(i, s);
        let (mut var, mut size) = (0.0, 0.0);
        for (a, ca) in &poly {
            for (b, cb) in &poly {
                let v = ca * cb * s.cv(g, &idx_add(a, b));
                var += v;
                size += v.abs();
            }
        }
        // The mean square cancels to the variance: where it keeps no digit
        // of its terms' sizes (two rows, whose correlation is exactly 1),
        // what is left is rounding, and the error is withheld.
        if !(kept(var, &[size]) && var.is_finite()) {
            return Err(Withheld::NoStandardError);
        }
        Ok((var * s.r[0]).sqrt())
    }

    /// Term `i`'s influence function, a polynomial in the deviations whose
    /// mean square over the summary is `n` times the delta-method variance.
    fn influence(&self, i: usize, s: &Summary) -> std::collections::BTreeMap<Idx, f64> {
        let t = &self.cfg.terms[i];
        let (gi, a) = &self.derived.terms[i];
        let g = &self.derived.groups[*gi];
        let n = a.len();
        match t.kind {
            MomentKind::Central => if_central(g, s, a),
            MomentKind::Raw => {
                let mut poly = std::collections::BTreeMap::new();
                for b in below(a) {
                    let mut m = 1.0;
                    for j in 0..n {
                        m *= s.m[j].powf(f64::from(a[j] - b[j]));
                    }
                    *poly.entry(b.clone()).or_insert(0.0) += binom(a, &b) * m;
                }
                let val = raw(g, s, a);
                *poly.entry(vec![0; n]).or_insert(0.0) -= val;
                poly
            }
            MomentKind::Standardized => {
                let var: Vec<f64> = (0..n).map(|j| s.cv(g, &unit(n, j, 2))).collect();
                let mut den = 1.0;
                for j in 0..n {
                    den *= var[j].sqrt().powf(f64::from(a[j]));
                }
                let val = s.cv(g, a) / den;
                let mut poly: std::collections::BTreeMap<Idx, f64> = if_central(g, s, a)
                    .into_iter()
                    .map(|(k, v)| (k, v / den))
                    .collect();
                for j in 0..n {
                    if a[j] > 0 {
                        let f = -val * f64::from(a[j]) / (2.0 * var[j]);
                        poly_add(&mut poly, &if_central(g, s, &unit(n, j, 2)), f);
                    }
                }
                poly
            }
            MomentKind::Cumulant => {
                let p = a[0];
                if p == 1 {
                    return std::iter::once((vec![1], 1.0)).collect();
                }
                let mut out = std::collections::BTreeMap::new();
                let poly = if t.bias {
                    self.derived.polys[i].clone()
                } else {
                    cumulant_poly(p)
                };
                for (mono, coef) in &poly {
                    for (k, &q) in mono.iter().enumerate() {
                        let mut f = *coef;
                        for (k2, &q2) in mono.iter().enumerate() {
                            if k2 != k {
                                f *= s.cv(g, &[q2]);
                            }
                        }
                        poly_add(&mut out, &if_central(g, s, &[q]), f);
                    }
                }
                out
            }
        }
    }

    /// The settings and the state checked as a state file is loaded: the
    /// settings `new` takes, and the vectors at their widths.
    pub(crate) fn checked(mut self) -> Result<Self, crate::StateError> {
        crate::model::check_cfg("ewm_moments", self.cfg.validate())?;
        self.derived = self.cfg.derive();
        if !self.has_shape() {
            return Err(crate::model::damaged(
                "ewm_moments",
                "its running sums have the wrong shape",
            ));
        }
        self.rebuild_back();
        Ok(self)
    }
}

/// `μ'_a = Σ_{β≤a} C(a, β) m^(a−β) μ_β`.
fn raw(g: &Group, s: &Summary, a: &[u32]) -> f64 {
    let mut val = 0.0;
    for b in below(a) {
        let mut m = 1.0;
        for j in 0..a.len() {
            m *= s.m[j].powf(f64::from(a[j] - b[j]));
        }
        val += binom(a, &b) * m * s.cv(g, &b);
    }
    val
}

/// The influence of `μ_a`: `d^a − μ_a − Σ_j a_j μ_{a−e_j} d_j`.
fn if_central(g: &Group, s: &Summary, a: &[u32]) -> std::collections::BTreeMap<Idx, f64> {
    let n = a.len();
    let mut poly = std::collections::BTreeMap::new();
    poly.insert(a.to_vec(), 1.0);
    *poly.entry(vec![0; n]).or_insert(0.0) -= s.cv(g, a);
    for j in 0..n {
        if a[j] > 0 {
            let e = unit(n, j, 1);
            let c = f64::from(a[j]) * s.cv(g, &idx_sub(a, &e));
            *poly.entry(e).or_insert(0.0) -= c;
        }
    }
    poly
}

fn poly_add(
    out: &mut std::collections::BTreeMap<Idx, f64>,
    y: &std::collections::BTreeMap<Idx, f64>,
    f: f64,
) {
    for (k, v) in y {
        *out.entry(k.clone()).or_insert(0.0) += f * v;
    }
}

/// The polarised `μ_2²` of a fourth-order `a`: `(c_ij c_kl + c_ik c_jl +
/// c_il c_jk) / 3` over its four leg indices, `cov` reading `c_ij`.
fn polar22(a: &[u32], cov: impl Fn(usize, usize) -> f64) -> f64 {
    let idx: Vec<usize> = (0..a.len())
        .flat_map(|j| std::iter::repeat_n(j, a[j] as usize))
        .collect();
    let (i, j, k, h) = (idx[0], idx[1], idx[2], idx[3]);
    (cov(i, j) * cov(k, h) + cov(i, k) * cov(j, h) + cov(i, h) * cov(j, k)) / 3.0
}

fn cov_of<'a>(g: &'a Group, s: &'a Summary) -> impl Fn(usize, usize) -> f64 + 'a {
    let n = g.legs.len();
    move |i, j| s.cv(g, &idx_add(&unit(n, i, 1), &unit(n, j, 1)))
}

fn standardized(g: &Group, s: &Summary, a: &[u32], bias: bool) -> Result<f64, Withheld> {
    let n = a.len();
    let var: Vec<f64> = (0..n).map(|j| s.cv(g, &unit(n, j, 2))).collect();
    if var.iter().any(|v| !(*v > 0.0 && v.is_finite())) {
        return Err(Withheld::ZeroSpread);
    }
    let p = order(a);
    if bias || p == 2 {
        let mut den = 1.0;
        for j in 0..n {
            den *= var[j].sqrt().powf(f64::from(a[j]));
        }
        return Ok(s.cv(g, a) / den);
    }
    let mut uvar = Vec::with_capacity(n);
    for j in 0..n {
        uvar.push(unbiased(g, s, &unit(n, j, 2))?.0);
    }
    let (mut num, u22) = unbiased(g, s, a)?;
    if p == 4 {
        // Fisher's k4 over the unbiased variances, plus the normal's 3:
        // `G2 + 3` at equal weights, as scipy's `kurtosis(bias=False,
        // fisher=False)`.
        let r2 = s.r[0];
        let cov = cov_of(g, s);
        num = num - 3.0 * u22 + 3.0 * polar22(a, |i, j| cov(i, j) / (1.0 - r2));
    }
    let mut den = 1.0;
    for j in 0..n {
        den *= uvar[j].sqrt().powf(f64::from(a[j]));
    }
    Ok(num / den)
}

/// The unbiased central moment of order 2 to 4 and, at 4, the unbiased
/// `μ_2²`; or why it is not defined.
fn unbiased(g: &Group, s: &Summary, a: &[u32]) -> Result<(f64, f64), Withheld> {
    let r2 = s.r[0];
    let c = s.cv(g, a);
    match order(a) {
        2 => {
            let den = 1.0 - r2;
            if !kept(den, &[1.0, r2]) {
                return Err(Withheld::OneRow);
            }
            Ok((c / den, 0.0))
        }
        3 => {
            let r3 = s.r(g, 3);
            let den = 1.0 - 3.0 * r2 + 2.0 * r3;
            if !kept(den, &[1.0, 3.0 * r2, 2.0 * r3]) {
                return Err(Withheld::TooFewRowsThird);
            }
            Ok((c / den, 0.0))
        }
        _ => {
            let (r3, r4) = (s.r(g, 3), s.r(g, 4));
            let a1 = 1.0 - 4.0 * r2 + 6.0 * r3 - 3.0 * r4;
            let a2 = 6.0 * r2 - 6.0 * r3 + 9.0 * r4 - 9.0 * r2 * r2;
            let b1 = r2 - 2.0 * r3 + r4;
            let b2 = 1.0 - 3.0 * r2 + 2.0 * r3 - 3.0 * r4 + 3.0 * r2 * r2;
            let det = a1 * b2 - a2 * b1;
            if !kept(det, &[a1 * b2, a2 * b1]) {
                return Err(Withheld::TooFewRowsFourth);
            }
            let p22 = polar22(a, cov_of(g, s));
            Ok(((b2 * c - a2 * p22) / det, (a1 * p22 - b1 * c) / det))
        }
    }
}

/// The columns are the features; there are no targets, and a slot a term
/// (then a slot a standard error under `se`). As every model's, each row's
/// output is read from the state before the row is learned.
impl crate::OnlineModel for EwmMoments {
    fn step_into(
        &mut self,
        x: &[f64],
        y: &[Option<f64>],
        d_clock: f64,
        weight: f64,
        out: &mut crate::Step,
    ) {
        if let Some(refused) = crate::model::refused_step(self, x, y, d_clock, weight) {
            *out = refused;
            return;
        }
        let now = self.place.as_mut().map(|p| p.step(d_clock));
        if self.ring.is_some() {
            self.slide(now);
        }
        out.n_eff = self.read_into(now, &mut out.pred);
        out.extra = None;
        let lam = self.cfg.decay.factor(d_clock);
        self.learn(x, now, lam, weight);
    }

    fn predict(&self, x: &[f64], d_clock: f64) -> crate::Step {
        if let Some(refused) = crate::model::refused_predict(self, x, d_clock) {
            return refused;
        }
        let now = self.place.as_ref().map(|p| p.peek(d_clock));
        let mut out = crate::Step::default();
        out.n_eff = self.read_into(now, &mut out.pred);
        out
    }

    fn clear_lags(&mut self) {
        for b in &mut self.bufs {
            b.vals.clear();
            b.stamps.clear();
        }
    }

    fn stamp_next(&mut self, stamp: Stamp) {
        if let Some(p) = &mut self.place {
            p.hand(stamp);
        }
    }

    fn take_state_warning(&mut self) -> Option<(u64, u64)> {
        self.warning.take().map(|bytes| (bytes, self.cfg.cap()))
    }

    fn state_over_cap(&self) -> Option<(u64, u64)> {
        self.over.map(|bytes| (bytes, self.cfg.cap()))
    }

    fn state(&self) -> crate::State {
        crate::State::new(crate::ModelState::EwmMoments(Box::new(self.clone())))
    }

    fn restore(s: &crate::State) -> Result<Self, crate::StateError> {
        Self::restore_owned(s.clone())
    }

    fn restore_owned(s: crate::State) -> Result<Self, crate::StateError> {
        crate::check_schema(&s)?;
        match s.model {
            crate::ModelState::EwmMoments(m) => m.checked(),
            other => Err(crate::StateError::WrongModel {
                expected: "ewm_moments",
                found: other.kind(),
            }),
        }
    }

    fn n_targets(&self) -> usize {
        0
    }

    fn n_features(&self) -> usize {
        self.cfg.n_columns()
    }

    fn n_outputs(&self) -> usize {
        self.cfg.terms.len() * if self.cfg.se { 2 } else { 1 }
    }
}

/// Whether a denominator keeps a digit of the terms it is formed from.
fn kept(den: f64, terms: &[f64]) -> bool {
    den > NO_DIGIT * terms.iter().fold(0.0, |s, t| s + t.abs())
}

#[cfg(test)]
mod tests;
