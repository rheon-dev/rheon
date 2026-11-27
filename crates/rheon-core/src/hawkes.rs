//! A self-exciting point process (Hawkes) fitted online by least squares,
//! exactly, one row at a time.
//!
//! Each target is a node `i = 1..D`, and a row is an instant on the clock: a
//! target's value `m > 0` on the row is an event of that node, of mark `m`
//! (1 for an event, a count for that many at once, or a size); a value of 0
//! or below, or none, is no event. The intensity of node `i` is
//!
//! ```text
//! λ_i(t) = μ_i + Σ_j Σ_u α_iju β_u Σ_{events t_k of j before t} m_k e^(−β_u (t − t_k))
//! ```
//!
//! with the kernel decays `β_u`, `u = 1..U`, given (`kernel_decays`), so the
//! kernel `α β e^(−β t)` integrates to `α`. With the excitation per source
//! node and decay, `R_ju(t) = Σ_{t_k of j < t} β_u m_k e^(−β_u (t − t_k))`, and
//! `φ(t) = (1, R_11, .., R_DU)` (`p = 1 + k` terms, `k = D U`), the intensity is
//! `λ_i = θ_iᵀ φ` with `θ_i = (μ_i, α_i11, .., α_iDU)`: linear in the
//! parameters. The fit minimises the least-squares contrast, each instant
//! weighted `w(t) = e^(−γ (T − t))` (`γ` the forgetting rate per clock unit,
//! `ln 2 / half_life`; 0 forgets nothing):
//!
//! ```text
//! L(θ) = Σ_i [ ∫ w λ_i² dt − 2 ∫ w λ_i dN_i ] = Σ_i [ θ_iᵀ G θ_i − 2 θ_iᵀ b_i ]
//! G   = ∫ w φ φᵀ dt          b_i = Σ_{events of i} w(t_k) m_k φ(t_k−)
//! θ_i = G⁻¹ b_i
//! ```
//!
//! `φ(t_k−)` is the left limit: an event is not in its own intensity, nor
//! in that of an event on the same row. An event on an earlier row is, at
//! the same clock or not.
//!
//! **The recursion.** On a row `d` clock units after the last, with `f =
//! e^(−γ d)`:
//!
//! ```text
//! R_ju ← e^(−β_u d) R_ju        b ← f b        Q ← f Q
//! G₀₀ ← f G₀₀ + (1 − e^(−γ d)) / γ   (d where γ = 0)
//! ```
//!
//! then, for each node `j` with an event of mark `m_j`, from `R` as it
//! stands before any of the row's events:
//!
//! ```text
//! b_j ← b_j + m_j φ(t−)          Q_jl ← Q_jl + m_j m_l          R_ju ← R_ju + m_j β_u
//! ```
//!
//! `G` is not kept. Between events `R_a R_b` decays at `β_a + β_b`, and at an
//! row with events `m` it jumps by `m_j(a) β_a R_b + m_j(b) β_b R_a + m_j(a)
//! m_j(b) β_a β_b`; summed over the rows, `Σ m_j R_b(t−)` is `b_j`'s entry
//! for `b`, and `Σ m_j m_l` is `Q_jl` (`l = j` but where two nodes have an
//! event on one row). Integrating `w d(R_a R_b)` and
//! `w dR_a` by parts then gives `G` from the state, at any read:
//!
//! ```text
//! G₀ₐ = (β_a b_{j(a)}[0] − R_a) / (β_a − γ)
//! Gₐᵦ = (β_a b_{j(a)}[b] + β_b b_{j(b)}[a] + β_a β_b Q_{j(a) j(b)} − R_a R_b)
//!       / (β_a + β_b − γ)
//! ```
//!
//! So the state is `R` (`k`), `b` (`D p`), `Q` (symmetric, `D (D + 1) / 2`),
//! `G₀₀` and the weight: `8 (k + D (1 + k) + D (D + 1) / 2 + 2)` bytes,
//! whatever the stream's length, and a row costs `U` exponentials and `O(D k)` additions to learn. A read assembles `G`
//! in `O(k²)` and solves it by Cholesky in `O(p³)`, once a row, for the
//! intensities the row reports. `γ` within 0.1% of a decay `β_u` or of a sum
//! `β_u + β_v` is refused: those divisions lose about `log10(β / |β − γ|)`
//! digits, and the exact formula at equality is another one.
//!
//! **What a row reports.** Each node's intensity `λ_i(t−)` before the row's
//! events, from the fit on everything before them (hard rule 2): `G` read to
//! the row's clock, which uses only that no event came since the last row,
//! and `b` without the row's events. `NaN` while `weight_sum` is under
//! `min_weight`, or while `G` is not positive definite (before every node
//! has had an event, or before any clock has passed), or is so near
//! singular that the events cannot tell two of its terms apart: a pivot of
//! `G` scaled to a unit diagonal below `1e-10`, a condition number above
//! `1e10`, as when two nodes' events fall on the same rows. Under a window,
//! also while the window holds no event of a node before the read's clock
//! (none at all, or only on rows at that clock): the node's excitation is
//! 0 over the whole observation, its column of `G` is 0 by its definition,
//! and its kernels' parameters are not identified. That is decided from
//! each node's count of the rows it holds, exactly, not from `G`, whose
//! column the subtractions that took the node's events back leave as
//! rounding, which the scaling to a unit diagonal would hide from the pivot
//! test; and as a node's last event leaves, its sums are set to 0. Every
//! such report but `min_weight`'s is one the events do not identify
//! ([`Hawkes::not_identified`]). `coefficients` reads `θ_i` the same way.
//! The fit is unconstrained: a parameter, and an intensity, can be negative
//! where the data say so.
//!
//! **The weight.** A row's weight decides only whether its events are
//! learned: 0 learns nothing and counts in no clock (hard rule 9: the
//! stream reads such a row as `predict` does and hands it to no model, so
//! the next row steps as if it were not there, from the last row that
//! counted, and it is in no window and none of `n_eff`; a row of weight 0
//! handed to the model directly ages the state over its step, as in every
//! model), and any weight above 0 learns the row's events at their marks. An event's
//! excitation is a fact of the process, not evidence to scale, and `G`'s
//! event part is read from `b` only because the same marks build both.
//! `weight_sum` is the decayed sum of the rows' weights, every row's, as in
//! every model (hard rule 8).
//!
//! **The window** (`window_events = W`, or `window_clock = D`). The fit is
//! that of the last `W` rows that carried an event, or of the events of the
//! last `D` clock units, alone: excitation by an event before the window is
//! dropped too. Dropping the oldest row `e` takes back exactly what it added:
//!
//! ```text
//! b_j[0] −= m_e,j      Q_jl −= m_e,j m_e,l      R_ju −= m_e,j β_u e^(−β_u (T − t_e))
//! b_i[(j, u)] −= β_u m_e,j F_e[i, u],    F_e[i, u] = Σ_{later rows l} m_l,i e^(−β_u (t_l − t_e))
//! ```
//!
//! and `G₀₀` is the clock observed: `T − t_e` for a window of `W` rows, from
//! the clock of the row dropped last (from the first row until one is), and
//! `min(D, the clock since the first row)` for a window of `D` units, the
//! interval `(T − D, T]` a fit of its events alone observes.
//!
//! **The window of `D` clock units** keeps each event row until it is `D`
//! old on the decayed clock. A row's age is decided on the exact stamps of
//! its clock and the reading row's ([`Stamp::cmp_span`]: integer nanoseconds
//! on a temporal clock, a single subtraction of raw values on a number
//! clock), never on a clock summed in doubles, so a row exactly `D` old is
//! always on one side of the edge. It is the window kinds' edge
//! ([`WindowClosed`], Polars' `rolling_*_by(closed=)`): under the default
//! `Right`, a row exactly `D` old has left and the window is `(T − D, T]`;
//! under `Both` it is kept and the window is `[T − D, T]`. A row leaves at
//! the first row at or after the one it ages out on, before that row's
//! intensity is read, so each row reports from the events of the last `D`
//! before it (events on earlier rows at its own clock included). How many
//! events that holds is the data's, so the bytes it takes are checked as it
//! runs ([`Hawkes::state_bytes`]): the first row to take the state past half
//! of `max_state_bytes` raises a warning, once, and one that would take it
//! past the cap is not learned and is reported ([`OnlineModel::state_over_cap`]).
//!
//! **`F_e` cannot be kept as one running sum**: rescaled as `t_e` moves, it would
//! multiply its rounding by up to `e^(β W)`. The window keeps its rows in two
//! blocks. The older holds, for each row, `F` over the later rows of its own
//! block, formed once by the backward recursion `F_r = e^(−β (t_{r+1} − t_r))
//! (m_{r+1} + F_{r+1})`; the newer keeps `B[i, u] = Σ m_l,i e^(−β_u (t_l −
//! t_m))` from its own start `t_m`, so `F_e = F_e^block + e^(−β (t_m − t_e)) B`,
//! every factor at most 1. When the older block runs out, the newer becomes
//! it, and `R`, `b` and `Q` are formed afresh from its rows, so their
//! rounding is that of one window, not of every row the stream ever dropped.
//! A row costs `O(D k)` amortised; the window holds `8 (2 + D + k)` bytes a
//! row in either block (a newer row's `k` sums are 0 until its block becomes
//! the older), and, under `D` clock units, a stamp of 32 bytes beside each.
//! A window forgets by dropping, so it takes no decay.
//!
//! **The window's clock.** No time is held as the clock's sum. An older
//! row holds `t_m − t_r`; the newer block's first row `t_r − t_m`; each
//! later newer row its gap to the row before it, read from the clock since
//! the newest row, which the window keeps beside the clock since `t_m`.
//! When the newer block becomes the older, each row's clock to now is
//! summed from the gaps, newest first. A gap after a long quiet stretch so
//! keeps its digits: held from `t_m`, a gap of 0.25 after a stretch of
//! `1e16` would be in `1e16 + 0.25`, which a double rounds to `1e16`, and
//! the block formed from it would read the events after the stretch as
//! all at one clock.
//!
//! **The weight under a window** (hard rule 8) is the weight inside it: the
//! rows that carried an event and are held, each with its weight (8 bytes a
//! row), added as the row is learned and taken back as it leaves. A row with
//! no event is no part of the fit, so the model keeps no record of it: its
//! weight is not in `n_eff`, which a window's `min_weight` therefore reads as
//! the weight of the events the fit is made of. As every `n_eff` is, it is
//! read before the row's own clock step: a row that a window of clock units
//! drops at this row is still in the weight the row reports, as weight a
//! decay takes at this row is still undecayed in it.

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

use crate::model::{ModelState, OnlineModel, State, StateError, Step, check_schema};
use crate::since::back_by;
use crate::solve::{KEPT_ORDER, Solves, solve_definite_in};
use crate::{Decay, Stamp, WindowClosed};

/// The least pivot of `G` scaled to a unit diagonal that a fit is read
/// at: below it `G`'s condition number is above `1e10`, and the fit has
/// lost all but a few of its digits to two terms the events cannot tell
/// apart -- two nodes whose events fall on the same rows, or a decay the
/// data cannot separate from another.
const PIVOT_FLOOR: f64 = 1e-10;

/// The bytes a group's state may hold by default, a spec's
/// `max_state_bytes`: 256 KB.
pub const MAX_STATE_BYTES: u64 = 256 * 1024;

/// The most `max_state_bytes` may be set to: 64 MiB a group, as every
/// model's.
pub const STATE_BYTES_CEILING: u64 = crate::clock_span::STATE_BYTES_CEILING;

/// How near the forgetting rate may come to a kernel decay, or to a sum of
/// two, relative to it: the read divides by `β − γ` and `β_u + β_v − γ`.
const RATE_MARGIN: f64 = 1e-3;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HawkesCfg {
    /// The nodes, one per target.
    pub n_targets: usize,
    /// The kernels' decays `β_u`, per clock unit: each node excites every
    /// node through one kernel `α β e^(−β t)` per decay.
    pub kernel_decays: Vec<f64>,
    /// The forgetting, per clock unit.
    pub decay: Decay,
    /// `weight_sum` before a row reports.
    pub min_weight: f64,
    /// Fit the last this many rows that carried an event, alone.
    pub window_events: Option<usize>,
    /// Fit the events of the last this many clock units, alone (seconds on a
    /// temporal clock). How many that is, the data decides.
    #[serde(default)]
    pub window_clock: Option<f64>,
    /// Which edge of a clock window holds the row exactly `window_clock`
    /// old: `Right`, the default, keeps rows less than that old, `Both`
    /// keeps that row too.
    #[serde(default)]
    pub closed: WindowClosed,
    /// The bytes a group's state may hold, a window's rows included; the
    /// default is [`MAX_STATE_BYTES`].
    #[serde(default)]
    pub max_state_bytes: Option<u64>,
}

/// `γ`, the forgetting rate per clock unit that `decay` gives.
fn rate(decay: Decay) -> f64 {
    match decay {
        Decay::Halflife(h) if h.is_infinite() => 0.0,
        Decay::Halflife(h) => std::f64::consts::LN_2 / h,
        Decay::Lam(1.0) => 0.0,
        Decay::Lam(l) => -l.ln(),
    }
}

impl HawkesCfg {
    pub fn validate(&self) -> Result<(), String> {
        let mut no = crate::Refusals::new();
        if self.n_targets == 0 {
            no.refuse(
                "hawkes: the model needs at least one target, one per node, but it has none. \
                 Name at least one column in targets.",
            );
        }
        let betas = &self.kernel_decays;
        if betas.is_empty() {
            no.refuse(
                "hawkes: kernel_decays must list at least one decay, but it lists none. Give \
                 the decay of each exponential kernel, per clock unit.",
            );
        }
        let bad: Vec<String> = betas
            .iter()
            .filter(|b| !(b.is_finite() && **b > 0.0))
            .map(f64::to_string)
            .collect();
        if !bad.is_empty() {
            no.refuse(format!(
                "hawkes: each of kernel_decays must be a finite number above 0, but {} {}. \
                 Give each kernel's decay as a positive number per clock unit.",
                if bad.len() == 1 {
                    "one is"
                } else {
                    "these are not:"
                },
                crate::model::listed(&bad),
            ));
        }
        let mut sorted: Vec<f64> = betas.clone();
        sorted.sort_by(f64::total_cmp);
        if sorted.windows(2).any(|w| w[0] == w[1]) {
            no.refuse(format!(
                "hawkes: kernel_decays must be distinct, but {} lists one twice. Two equal \
                 decays are one kernel, and their parameters cannot be told apart. List each \
                 decay once.",
                crate::model::listed(&betas.iter().map(f64::to_string).collect::<Vec<_>>()),
            ));
        }
        // `G`'s entries divide by `β_u + β_v − γ` and hold `β_u β_v Q`: a
        // pair of decays whose sum or product leaves the doubles has no
        // finite system (review 2026-11-23, HAWKES-4).
        if bad.is_empty() {
            let past: Vec<String> = betas
                .iter()
                .enumerate()
                .flat_map(|(u, &bu)| betas[u..].iter().map(move |&bv| (bu, bv)))
                .filter(|(bu, bv)| !((bu + bv).is_finite() && (bu * bv).is_finite()))
                .map(|(bu, bv)| format!("{bu} and {bv}"))
                .collect();
            if !past.is_empty() {
                no.refuse(format!(
                    "hawkes: kernel_decays are too large: the fit adds and multiplies each \
                     pair of decays, and {} leave the range of a double. Use a smaller unit \
                     for the clock, so that the decays are smaller numbers.",
                    crate::model::listed(&past),
                ));
            }
        }
        if let Err(e) = self.decay.check() {
            no.refuse(format!("hawkes: {e}"));
        } else if !rate(self.decay).is_finite() {
            // Only a half-life reaches here: `−ln λ` of a `λ` above 0 is
            // at most 745.
            let h = match self.decay {
                Decay::Halflife(h) => h,
                Decay::Lam(l) => l,
            };
            no.refuse(format!(
                "hawkes: half_life is {h}, so small that its forgetting rate, ln 2 / \
                 half_life, is past the range of a double. Set half_life to a larger number \
                 of clock units.",
            ));
        } else if bad.is_empty() {
            let gamma = rate(self.decay);
            let near = |s: f64| gamma > 0.0 && (s - gamma).abs() <= RATE_MARGIN * s;
            let mut close: Vec<String> = Vec::new();
            for (u, &bu) in betas.iter().enumerate() {
                if near(bu) {
                    close.push(format!("{bu}"));
                }
                for &bv in &betas[u..] {
                    if near(bu + bv) {
                        close.push(format!("{bu} + {bv}"));
                    }
                }
            }
            if !close.is_empty() {
                no.refuse(format!(
                    "hawkes: the forgetting rate, {gamma} per clock unit (ln 2 / half_life), is \
                     within 0.1% of {}, where the fit divides by the difference and loses its \
                     digits. Set half_life well above the kernels' memory (1 / the smallest \
                     decay), or to inf to forget nothing.",
                    crate::model::listed(&close),
                ));
            }
        }
        no.take(crate::model::check_min_weight("hawkes", self.min_weight));
        if self.window_events == Some(0) {
            no.refuse(
                "hawkes: window_events must be at least 1, but it is 0. Give the number \
                 of rows with an event the fit keeps, or leave it out for no window.",
            );
        }
        if let Some(d) = self.window_clock {
            if !(d.is_finite() && d > 0.0) {
                no.refuse(format!(
                    "hawkes: the clock window must be a finite number of clock units above 0, \
                     but it is {d}. Give the span of clock the fit keeps events of, or leave \
                     it out for no window.",
                ));
            }
            if self.window_events.is_some() {
                no.refuse(
                    "hawkes: a window of clock units and window_events are two windows, and \
                     the fit holds one. Leave one of them out.",
                );
            }
        } else if self.closed != WindowClosed::Right {
            no.refuse(
                "hawkes: closed = \"both\" says which edge of a clock window keeps the row \
                 exactly its span old, but there is no clock window. Give the window's span, \
                 or leave closed out.",
            );
        }
        if (self.window_events.is_some() || self.window_clock.is_some())
            && self.decay.check().is_ok()
            && rate(self.decay) > 0.0
        {
            no.refuse(
                "hawkes: a window forgets by dropping the oldest events, so it takes no decay, \
                 but half_life (or lam) decays too. Set half_life to inf, or leave out the \
                 window.",
            );
        }
        // Held to every model's ceiling, and the default cap checked as a
        // given one is (review 2026-11-23, HAWKES-3).
        let given = crate::clock_span::check_max_state_bytes("hawkes", self.max_state_bytes);
        if let Err(e) = given {
            no.refuse(e);
        } else if self.n_targets > 0 && !betas.is_empty() {
            let (n, u, cap) = (self.n_targets, betas.len(), self.cap());
            let bytes = match (self.window_events, self.window_clock) {
                (Some(w), _) => Hawkes::state_bytes(n, u, Some(w)),
                (None, Some(_)) => Hawkes::clock_state_bytes(n, u, 0),
                (None, None) => Hawkes::state_bytes(n, u, None),
            };
            if bytes > cap {
                no.refuse(format!(
                    "hawkes: its state would hold {bytes} bytes, past max_state_bytes of \
                     {cap}. Lower window_events, or raise max_state_bytes.",
                ));
            }
        }
        no.finish()
    }

    /// The bytes a group's state may hold.
    fn cap(&self) -> u64 {
        self.max_state_bytes.unwrap_or(MAX_STATE_BYTES)
    }

    fn dims(&self) -> (usize, usize, usize) {
        let k = self.n_targets * self.kernel_decays.len();
        (self.n_targets, self.kernel_decays.len(), k)
    }
}

/// The rows a window keeps, oldest first, in two blocks (see the module
/// docs): the first `old` the older block, the rest the newer. Every row
/// holds the same numbers -- its time, its weight, its `D` marks and its `k`
/// forward sums, which a newer row holds at 0 until its block becomes the
/// older -- so the window's size is its rows' count times one figure,
/// whatever the blocks' split. Times are never held as the clock's sum: an
/// older row's is `t_m − t_r`, from the newer block's start `t_m`; the newer
/// block's first row's is `t_r − t_m`, and each later newer row's is its gap
/// to the row before it. So a gap after a long quiet stretch keeps its
/// digits: counted from `t_m`, a gap of 0.25 after a stretch of `1e16` would
/// be held in `1e16 + 0.25`, which rounds to `1e16`. A window of clock units
/// holds each row's stamp too, which decides when it leaves.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct EventWindow {
    /// The clock since `t_m`, now.
    since: f64,
    /// The clock since the newer block's newest row, now, from which the
    /// next row's gap is read; while the newer block is empty, the same
    /// clock as `since`. Not a number in a state saved before the window
    /// kept its gaps, whose newer rows' times are `t_r − t_m`: such a state
    /// is moved to the gaps as it loads ([`EventWindow::rebase`]).
    #[serde(default = "unset")]
    last: f64,
    /// The rows of the older block, at the front.
    old: usize,
    time: VecDeque<f64>,
    /// Each row's weight, and their sum: the weight inside the window.
    weight: VecDeque<f64>,
    w_in: f64,
    marks: VecDeque<f64>,
    fwd: VecDeque<f64>,
    /// `B[i, u] = Σ_newer m_l,i e^(−β_u (t_l − t_m))`, `k`.
    new_sum: Vec<f64>,
    /// A window of clock units: each row's place on the decayed clock, held
    /// exactly, and where the clock stands.
    #[serde(default)]
    stamp: VecDeque<Stamp>,
    #[serde(default)]
    track: Option<ClockTrack>,
    /// Each node's rows with an event the window holds, read from the marks;
    /// never saved, and counted again as a state loads.
    #[serde(skip)]
    held: Vec<u32>,
}

/// A clock a state saved before it had the field holds: not a number.
fn unset() -> f64 {
    f64::NAN
}

/// Where a clock window reads the clock from: the steps summed, which times
/// a row handed no stamp, and the last stamp handed, which a read ahead
/// (`predict`) moves forward by its step.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
struct ClockTrack {
    own: f64,
    #[serde(default)]
    last: Option<Stamp>,
}

impl EventWindow {
    fn new(n: usize, k: usize, clocked: bool) -> Self {
        Self {
            since: 0.0,
            last: 0.0,
            old: 0,
            time: VecDeque::new(),
            weight: VecDeque::new(),
            w_in: 0.0,
            marks: VecDeque::new(),
            fwd: VecDeque::new(),
            new_sum: vec![0.0; k],
            stamp: VecDeque::new(),
            track: clocked.then(ClockTrack::default),
            held: vec![0; n],
        }
    }

    /// The clock from row `r − 1` to row `r`, `r ≥ 1`, from the times as the
    /// blocks hold them.
    fn gap_before(&self, r: usize) -> f64 {
        if r > self.old {
            self.time[r]
        } else if r == self.old {
            self.time[r] + self.time[r - 1]
        } else {
            self.time[r - 1] - self.time[r]
        }
    }

    /// The clock since the newest row, `ahead` past where the window stands.
    fn newest_age(&self, ahead: f64) -> f64 {
        let rows = self.rows();
        if rows > self.old {
            self.last + ahead
        } else {
            self.since + ahead + self.time[rows - 1]
        }
    }

    /// Whether some node's excitation is 0 over the whole observation read
    /// `ahead` past where the window stands: the window holds no event of
    /// it, or holds them only on the rows at the read's own clock. Its
    /// column of `G` is then 0 by its definition, and its kernels'
    /// parameters are not identified.
    fn has_unexcited(&self, n: usize, ahead: f64) -> bool {
        if self.held.contains(&0) {
            return true;
        }
        if self.newest_age(ahead) > 0.0 {
            return false;
        }
        // The rows at the read's own clock, newest first: `from..rows`.
        let rows = self.rows();
        let mut from = rows - 1;
        while from > 0 && self.gap_before(from) == 0.0 {
            from -= 1;
        }
        (0..n).any(|j| {
            let at = (from..rows)
                .filter(|&r| self.marks[r * n + j] > 0.0)
                .count();
            at == self.held[j] as usize
        })
    }

    /// Each node's rows with an event, counted from the marks.
    fn recount(&mut self, n: usize) {
        self.held.clear();
        self.held.resize(n, 0);
        for (i, m) in self.marks.iter().enumerate() {
            self.held[i % n] += u32::from(*m > 0.0);
        }
    }

    /// A state saved before the window kept its gaps, whose newer rows'
    /// times are `t_r − t_m`, to the gaps: each newer row after the block's
    /// first takes its gap to the row before it, and `last` the clock since
    /// the newest. Exact where the clock's sums were (a clock of whole
    /// units, say), so such a state goes on to the bit and saves the bytes
    /// the same rows give now; elsewhere it goes on within the rounding of
    /// the times it held. A state with `last` is left as it is.
    fn rebase(&mut self) {
        if !self.last.is_nan() {
            return;
        }
        let rows = self.rows();
        self.last = if rows > self.old {
            self.since - self.time[rows - 1]
        } else {
            self.since
        };
        for r in (self.old + 1..rows).rev() {
            self.time[r] -= self.time[r - 1];
        }
    }

    fn rows(&self) -> usize {
        self.time.len()
    }

    fn has_shape(&self, n: usize, k: usize, cap: Option<usize>, clocked: bool) -> bool {
        let rows = self.time.len();
        self.marks.len() == rows * n
            && self.weight.len() == rows
            && self.fwd.len() == rows * k
            && self.new_sum.len() == k
            && self.old <= rows
            && cap.is_none_or(|c| rows <= c)
            && self.track.is_some() == clocked
            && self.stamp.len() == if clocked { rows } else { 0 }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Hawkes {
    cfg: HawkesCfg,
    /// `R`, node-major: `a = j U + u`.
    r: Vec<f64>,
    /// `b`, one row of `p = 1 + k` per node.
    b: Vec<f64>,
    /// `Q_jl = Σ w m_j m_l` over the rows, its lower triangle packed by
    /// rows ([`tri`]).
    q: Vec<f64>,
    /// `G₀₀ = ∫ w dt`.
    g0: f64,
    w_sum: f64,
    window: Option<EventWindow>,
    /// The bytes a clock window's next row would have taken the state to,
    /// past the cap: from then on no row's events are learned.
    #[serde(default)]
    over: Option<u64>,
    /// The half-of-the-cap warning has been raised (once a group).
    #[serde(default)]
    warned: bool,
    /// The next row's stamp, which a step takes; and the warning raised and
    /// not yet taken. Never state: neither is held between two calls.
    #[serde(skip)]
    next: Option<Stamp>,
    #[serde(skip)]
    warning: Option<u64>,
}

/// The buffers a read assembles and solves its system in, one set a thread,
/// kept between reads and written over; nothing in it is read between calls.
#[derive(Default)]
struct ReadScratch {
    solves: Solves,
    g: Vec<f64>,
    x: Vec<f64>,
    r: Vec<f64>,
    b: Vec<f64>,
    q: Vec<f64>,
}

thread_local! {
    /// Lent to a read and set back after it, never borrowed: see
    /// [`crate::scratch`].
    static READ: std::cell::Cell<Option<Box<ReadScratch>>> = const { std::cell::Cell::new(None) };
}

/// `f` in this thread's [`ReadScratch`], which a read of order `p` past
/// [`KEPT_ORDER`] leaves empty.
fn with_scratch<T>(p: usize, f: impl FnOnce(&mut ReadScratch) -> T) -> T {
    let mut sc = READ.take().unwrap_or_default();
    let out = f(&mut sc);
    if p > KEPT_ORDER {
        *sc = ReadScratch::default();
    }
    READ.set(Some(sc));
    out
}

/// The place of `Q_jl` in its packed lower triangle.
#[inline]
fn tri(j: usize, l: usize) -> usize {
    let (hi, lo) = if j >= l { (j, l) } else { (l, j) };
    hi * (hi + 1) / 2 + lo
}

/// The system's `G`, row-major `p × p`, from a state (see the module docs).
#[allow(clippy::too_many_arguments)]
fn gram_into(
    betas: &[f64],
    n: usize,
    gamma: f64,
    r: &[f64],
    b: &[f64],
    q: &[f64],
    g0: f64,
    g: &mut Vec<f64>,
) {
    let u_n = betas.len();
    let k = n * u_n;
    let p = 1 + k;
    g.clear();
    g.resize(p * p, 0.0);
    g[0] = g0;
    for a in 0..k {
        let (ja, ba) = (a / u_n, betas[a % u_n]);
        let v = (ba * b[ja * p] - r[a]) / (ba - gamma);
        g[a + 1] = v;
        g[(a + 1) * p] = v;
        for c in 0..=a {
            let (jc, bc) = (c / u_n, betas[c % u_n]);
            let mut jump = ba * b[ja * p + 1 + c] + bc * b[jc * p + 1 + a];
            jump += ba * bc * q[tri(ja, jc)];
            let v = (jump - r[a] * r[c]) / (ba + bc - gamma);
            g[(a + 1) * p + c + 1] = v;
            g[(c + 1) * p + a + 1] = v;
        }
    }
}

/// `θ` for every node, column-major `p × D` in `x`, from a state: false
/// where `G` is not positive definite as it stands (a factorization that
/// needed a jitter is refused, so a singular `G` reports nothing rather than
/// a fit of the jitter), or is so near singular that the fit is not the
/// data's ([`PIVOT_FLOOR`]).
#[allow(clippy::too_many_arguments)]
fn solve_into(
    betas: &[f64],
    n: usize,
    gamma: f64,
    r: &[f64],
    b: &[f64],
    q: &[f64],
    g0: f64,
    solves: &mut Solves,
    g: &mut Vec<f64>,
    x: &mut Vec<f64>,
) -> bool {
    let p = 1 + n * betas.len();
    gram_into(betas, n, gamma, r, b, q, g0, g);
    solve_definite_in(solves, g, b, p, n, PIVOT_FLOOR, x)
}

impl Hawkes {
    pub fn new(cfg: HawkesCfg) -> Result<Self, String> {
        cfg.validate()?;
        let (n, _, k) = cfg.dims();
        Ok(Self {
            r: vec![0.0; k],
            b: vec![0.0; n * (1 + k)],
            q: vec![0.0; n * (n + 1) / 2],
            g0: 0.0,
            w_sum: 0.0,
            window: (cfg.window_events.is_some() || cfg.window_clock.is_some())
                .then(|| EventWindow::new(n, k, cfg.window_clock.is_some())),
            over: None,
            warned: false,
            next: None,
            warning: None,
            cfg,
        })
    }

    pub fn cfg(&self) -> &HawkesCfg {
        &self.cfg
    }

    /// The weight the fit is made of (hard rule 8): every row's, decayed,
    /// and under a window the weight of the event rows it holds.
    pub fn n_eff(&self) -> f64 {
        self.window.as_ref().map_or(self.w_sum, |w| w.w_in)
    }

    /// The numbers of the running sums, in bytes: `8 (k + D (1 + k) + D (D +
    /// 1) / 2 + 2)`.
    fn sums_bytes(n: u64, k: u64) -> u64 {
        let pairs = n.saturating_mul(n.saturating_add(1)) / 2;
        k.saturating_add(n.saturating_mul(k.saturating_add(1)))
            .saturating_add(pairs)
            .saturating_add(2)
            .saturating_mul(8)
    }

    /// The bytes of a window's rows: `8 (2 + D + k)` each, with the newer
    /// block's sum `B`, the clock since its start, the clock since its
    /// newest row and the weight inside, `8 (k + 3)`.
    fn window_bytes(n: u64, k: u64, rows: u64) -> u64 {
        rows.saturating_mul(n.saturating_add(k).saturating_add(2))
            .saturating_add(k)
            .saturating_add(3)
            .saturating_mul(8)
    }

    /// The bytes of the numbers a model of this shape holds at most: its
    /// running sums, `8 (k + D (1 + k) + D (D + 1) / 2 + 2)`, and a full
    /// window of `W` rows with an event, `8 W (2 + D + k)`, with its sum `B`,
    /// its two clocks and its weight, `8 (k + 3)`. The one part whose size the
    /// data could set is the window's rows, and `W` bounds it, so this is
    /// known from the settings alone. A window of clock units is not
    /// bounded by its settings: see [`Self::clock_state_bytes`]. Past what a
    /// `u64` counts it reads `u64::MAX`, which no cap takes.
    pub fn state_bytes(n_targets: usize, n_decays: usize, window_events: Option<usize>) -> u64 {
        let n = n_targets as u64;
        let k = n.saturating_mul(n_decays as u64);
        Self::sums_bytes(n, k)
            .saturating_add(window_events.map_or(0, |w| Self::window_bytes(n, k, w as u64)))
    }

    /// The bytes a window of clock units holds with `rows` event rows in it:
    /// [`Self::state_bytes`] for that many rows, and a stamp
    /// ([`std::mem::size_of`] a [`Stamp`]) with each, the one the clock
    /// tracks, and the clock summed (8). It grows with the events the span
    /// holds, so it is read as the rows go in.
    pub fn clock_state_bytes(n_targets: usize, n_decays: usize, rows: usize) -> u64 {
        let stamp = std::mem::size_of::<Stamp>() as u64;
        Self::state_bytes(n_targets, n_decays, Some(rows))
            .saturating_add((rows as u64).saturating_add(1).saturating_mul(stamp))
            .saturating_add(8)
    }

    /// The stamp last handed to a step of a clock window ([`Stamp`]), `None`
    /// before one, and without a clock window: where a caller that stamps
    /// its rows resumes from a restored state.
    pub fn last_stamp(&self) -> Option<Stamp> {
        self.window.as_ref()?.track.as_ref()?.last
    }

    /// Whether a step's or a read's report was withheld because the events
    /// do not identify the fit: its intensities are not numbers though its
    /// weight is at `min_weight`, so the system it was read from was
    /// singular, or so near it that its terms cannot be told apart (see the
    /// module docs).
    pub fn not_identified(&self, out: &Step) -> bool {
        out.n_eff >= self.cfg.min_weight
            && !out.pred.is_empty()
            && out.pred.iter().all(|v| v.is_nan())
    }

    /// Whether every node's excitation is above 0 somewhere in the
    /// observation as read `ahead` past where the state stands: under a
    /// window, from the rows it holds ([`EventWindow::has_unexcited`]);
    /// without one, the solve decides, since a node's sums are 0 exactly
    /// until its first event.
    fn excited(&self, ahead: f64) -> bool {
        self.window
            .as_ref()
            .is_none_or(|w| !w.has_unexcited(self.cfg.n_targets, ahead))
    }

    /// The rows with an event the window holds; 0 without a window.
    pub fn window_rows(&self) -> usize {
        self.window.as_ref().map_or(0, EventWindow::rows)
    }

    fn gamma(&self) -> f64 {
        rate(self.cfg.decay)
    }

    /// The least-squares system as the state stands: `G` (row-major `p ×
    /// p`) and the right-hand sides `b` (one row of `p` per node), whose
    /// solutions are the fit.
    pub fn system(&self) -> (Vec<f64>, Vec<f64>) {
        let mut g = Vec::new();
        gram_into(
            &self.cfg.kernel_decays,
            self.cfg.n_targets,
            self.gamma(),
            &self.r,
            &self.b,
            &self.q,
            self.g0,
            &mut g,
        );
        (g, self.b.clone())
    }

    /// The fitted parameters as the state stands, one list per node: `μ_i`,
    /// then `α_iju` for each source node `j` and decay `u` (`j` outer). `NaN`
    /// throughout where the system is not positive definite.
    pub fn coefficients(&self) -> Vec<Vec<f64>> {
        let (n, _, k) = self.cfg.dims();
        let p = 1 + k;
        let excited = self.excited(0.0);
        with_scratch(p, |sc| {
            let ok = excited
                && solve_into(
                    &self.cfg.kernel_decays,
                    n,
                    self.gamma(),
                    &self.r,
                    &self.b,
                    &self.q,
                    self.g0,
                    &mut sc.solves,
                    &mut sc.g,
                    &mut sc.x,
                );
            (0..n)
                .map(|i| {
                    if ok {
                        sc.x[i * p..(i + 1) * p].to_vec()
                    } else {
                        vec![f64::NAN; p]
                    }
                })
                .collect()
        })
    }

    /// The intensities from a state advanced to the row, into `pred`:
    /// `NaN` for each node where `n_eff` is under `min_weight` or the system
    /// is not solvable.
    #[allow(clippy::too_many_arguments)]
    fn intensities_into(
        cfg: &HawkesCfg,
        n_eff: f64,
        excited: bool,
        r: &[f64],
        b: &[f64],
        q: &[f64],
        g0: f64,
        sc: &mut ReadScratch,
        pred: &mut Vec<f64>,
    ) {
        let (n, _, k) = cfg.dims();
        let p = 1 + k;
        pred.clear();
        let ReadScratch { solves, g, x, .. } = sc;
        if !(n_eff >= cfg.min_weight
            && excited
            && solve_into(
                &cfg.kernel_decays,
                n,
                rate(cfg.decay),
                r,
                b,
                q,
                g0,
                solves,
                g,
                x,
            ))
        {
            pred.resize(n, f64::NAN);
            return;
        }
        for i in 0..n {
            let theta = &x[i * p..(i + 1) * p];
            let mut lam = theta[0];
            for (t, ra) in theta[1..].iter().zip(r) {
                lam += t * ra;
            }
            pred.push(lam);
        }
    }

    /// `G₀₀`'s step over `d`: `(1 − e^(−γ d)) / γ`, `d` where `γ = 0`.
    fn g0_step(gamma: f64, d: f64) -> f64 {
        if gamma == 0.0 {
            d
        } else {
            -(-gamma * d).exp_m1() / gamma
        }
    }

    /// Learn one row's events into `r`, `b`, `q`: each node's mark `m > 0`
    /// adds `m φ(t−)` to its `b` and `m m_l` to its `Q`, from `R` before any of
    /// the row's events, then raises its `R` by `m β`.
    fn learn_marks(betas: &[f64], marks: &[f64], r: &mut [f64], b: &mut [f64], q: &mut [f64]) {
        let u_n = betas.len();
        let k = r.len();
        let p = 1 + k;
        for (j, &m) in marks.iter().enumerate() {
            if m > 0.0 {
                let row = &mut b[j * p..(j + 1) * p];
                row[0] += m;
                for (bj, ra) in row[1..].iter_mut().zip(r.iter()) {
                    *bj += m * ra;
                }
                for (l, &ml) in marks.iter().enumerate().take(j + 1) {
                    if ml > 0.0 {
                        q[tri(j, l)] += m * ml;
                    }
                }
            }
        }
        for (j, &m) in marks.iter().enumerate() {
            if m > 0.0 {
                for (ra, beta) in r[j * u_n..(j + 1) * u_n].iter_mut().zip(betas) {
                    *ra += m * beta;
                }
            }
        }
    }

    /// The row's marks: a target's value above 0, else 0.
    fn mark(y: Option<f64>) -> f64 {
        match y {
            Some(v) if v > 0.0 => v,
            _ => 0.0,
        }
    }

    /// Add the row's events to the newer block, and drop the oldest row
    /// while the window holds more than its rows.
    fn window_learn(&mut self, y: &[Option<f64>], weight: f64, now: Option<Stamp>) {
        let (n, u_n, k) = self.cfg.dims();
        let betas = &self.cfg.kernel_decays;
        let Some(w) = self.window.as_mut() else {
            return;
        };
        // The newer block's first row is held from its start; each later
        // one by its gap to the row before it.
        w.time
            .push_back(if w.rows() == w.old { w.since } else { w.last });
        w.last = 0.0;
        w.weight.push_back(weight);
        w.w_in += weight;
        w.stamp.extend(now);
        for j in 0..n {
            let m = Self::mark(y.get(j).copied().flatten());
            w.held[j] += u32::from(m > 0.0);
            w.marks.push_back(m);
        }
        w.fwd.extend(std::iter::repeat_n(0.0, k));
        let at = w.marks.len() - n;
        for (u, beta) in betas.iter().enumerate() {
            let e = (-beta * w.since).exp();
            for i in 0..n {
                let m = w.marks[at + i];
                if m > 0.0 {
                    w.new_sum[i * u_n + u] += m * e;
                }
            }
        }
        let cap = self.cfg.window_events.unwrap_or(usize::MAX);
        if w.rows() > cap {
            if w.old == 0 {
                self.form_old_block();
            } else {
                self.drop_oldest();
            }
        }
    }

    /// The oldest row out of the older block, taking back what it added.
    fn drop_oldest(&mut self) {
        let (n, u_n, k) = self.cfg.dims();
        let p = 1 + k;
        let betas = &self.cfg.kernel_decays;
        let w = self.window.as_mut().expect("windowed");
        let off = w.time[0];
        for (u, &beta) in betas.iter().enumerate() {
            let to_block = (-beta * off).exp();
            let to_now = (-beta * (w.since + off)).exp();
            for j in 0..n {
                let m = w.marks[j];
                if m <= 0.0 {
                    continue;
                }
                let a = j * u_n + u;
                self.r[a] -= m * beta * to_now;
                for i in 0..n {
                    let f = w.fwd[i * u_n + u] + to_block * w.new_sum[i * u_n + u];
                    self.b[i * p + 1 + a] -= beta * m * f;
                }
            }
        }
        for j in 0..n {
            let m = w.marks[j];
            if m <= 0.0 {
                continue;
            }
            self.b[j * p] -= m;
            for l in 0..=j {
                let ml = w.marks[l];
                if ml > 0.0 {
                    self.q[tri(j, l)] -= m * ml;
                }
            }
        }
        self.g0 = w.since + off;
        w.time.pop_front();
        w.w_in = (w.w_in - w.weight.pop_front().unwrap_or(0.0)).max(0.0);
        w.stamp.pop_front();
        w.fwd.drain(..k);
        w.old -= 1;
        for j in 0..n {
            let m = w.marks.pop_front().unwrap_or(0.0);
            if m <= 0.0 {
                continue;
            }
            w.held[j] -= 1;
            if w.held[j] == 0 {
                // The node's last event has left: what it added is taken
                // back to the rounding of the subtractions, and the
                // window's sums for it are 0 exactly.
                Self::clear_node(&self.cfg, j, &mut self.r, &mut self.b, &mut self.q);
            }
        }
        // A window that every row has left holds nothing: its sums are 0
        // exactly, not what the subtractions rounded to (marks of `1e8` and
        // 1 left `Q = −1`, which the next burst fitted from), and its clock
        // starts again at the next row (review 2026-11-23, HAWKES-2).
        if w.time.is_empty() {
            w.w_in = 0.0;
            w.since = 0.0;
            w.last = 0.0;
            w.new_sum.fill(0.0);
            self.r.fill(0.0);
            self.b.fill(0.0);
            self.q.fill(0.0);
        }
    }

    /// Node `j`'s part of `R`, `b` and `Q` set to 0: its excitation, every
    /// node's sums of it, its own events' sums, and its products of marks.
    fn clear_node(cfg: &HawkesCfg, j: usize, r: &mut [f64], b: &mut [f64], q: &mut [f64]) {
        let (n, u_n, k) = cfg.dims();
        let p = 1 + k;
        r[j * u_n..(j + 1) * u_n].fill(0.0);
        for i in 0..n {
            b[i * p + 1 + j * u_n..i * p + 1 + (j + 1) * u_n].fill(0.0);
        }
        b[j * p..(j + 1) * p].fill(0.0);
        for l in 0..n {
            q[tri(j, l)] = 0.0;
        }
    }

    /// The older block has run out: the newer becomes it, less its oldest
    /// row, which is the one to drop. Its forward sums by the backward
    /// recursion, and `R`, `b` and `Q` formed afresh from its rows; every
    /// buffer is written in place.
    fn form_old_block(&mut self) {
        let (n, u_n, k) = self.cfg.dims();
        let betas = &self.cfg.kernel_decays;
        let w = self.window.as_mut().expect("windowed");
        let rows = w.time.len();
        // Every row is the newer block's: row `r ≥ 1` holds its gap to row
        // `r − 1`, and row 0, dropped here, needs none.
        // In place: a deque held in one piece reads as a slice.
        let marks: &[f64] = w.marks.make_contiguous();
        // Row `rows - 1` has no later row: its sums stay the 0 it was
        // pushed with. Row 0 is dropped, so it needs none.
        for r in (1..rows - 1).rev() {
            let gap = w.time[r + 1];
            for (u, beta) in betas.iter().enumerate() {
                let e = (-beta * gap).exp();
                for i in 0..n {
                    let a = i * u_n + u;
                    w.fwd[r * k + a] = e * (marks[(r + 1) * n + i] + w.fwd[(r + 1) * k + a]);
                }
            }
        }
        self.r.fill(0.0);
        self.b.fill(0.0);
        self.q.fill(0.0);
        for r in 1..rows {
            if r > 1 {
                let gap = w.time[r];
                for (u, beta) in betas.iter().enumerate() {
                    let e = (-beta * gap).exp();
                    for j in 0..n {
                        self.r[j * u_n + u] *= e;
                    }
                }
            }
            Self::learn_marks(
                betas,
                &marks[r * n..(r + 1) * n],
                &mut self.r,
                &mut self.b,
                &mut self.q,
            );
        }
        let gap = w.last;
        for (u, beta) in betas.iter().enumerate() {
            let e = (-beta * gap).exp();
            for j in 0..n {
                self.r[j * u_n + u] *= e;
            }
        }
        // Each row's clock to now, the newest's first, summed from the
        // gaps: the older block's times.
        let mut age = w.last;
        for t in w.time.iter_mut().rev() {
            let gap = *t;
            *t = age;
            age += gap;
        }
        self.g0 = w.time[0];
        w.time.pop_front();
        w.weight.pop_front();
        w.w_in = w.weight.iter().sum();
        w.stamp.pop_front();
        for (h, m) in w.held.iter_mut().zip(w.marks.drain(..n)) {
            *h -= u32::from(m > 0.0);
        }
        w.fwd.drain(..k);
        w.old = rows - 1;
        w.new_sum.fill(0.0);
        w.since = 0.0;
        w.last = 0.0;
    }
}

impl Hawkes {
    /// The state to the row's clock, `d_clock` after the last: what `predict`
    /// reads on a copy. `R` decays by the kernels, `b` and `Q` by the row's
    /// decay, `G₀₀` by its step; a window of clock units then drops the rows
    /// that are out of it at `now`, and observes no more than its span.
    fn advance(&mut self, d_clock: f64, now: Option<Stamp>) {
        let (n, u_n, _) = self.cfg.dims();
        let gamma = self.gamma();
        let f = self.cfg.decay.factor(d_clock);
        for (u, beta) in self.cfg.kernel_decays.iter().enumerate() {
            let e = (-beta * d_clock).exp();
            for j in 0..n {
                self.r[j * u_n + u] *= e;
            }
        }
        for v in self.b.iter_mut().chain(self.q.iter_mut()) {
            *v *= f;
        }
        self.g0 = f * self.g0 + Self::g0_step(gamma, d_clock);
        // The clock since the newer block's start runs only while the
        // window holds a row: an empty one's next row starts it at 0, where
        // a quiet stretch of `1e16` before it rounded its gaps away.
        if let Some(w) = self.window.as_mut().filter(|w| !w.time.is_empty()) {
            w.since += d_clock;
            w.last += d_clock;
        }
        if let (Some(now), Some(span)) = (now, self.cfg.window_clock) {
            // A row that left means the clock is at least a span old; and
            // the observation is no longer than the span in any case.
            let dropped = self.trim(now);
            if dropped || self.g0 > span {
                self.g0 = span;
            }
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

    /// Whether the oldest row of a clock window has left at `now`.
    fn oldest_has_left(&self, now: Stamp) -> bool {
        let Some(span) = self.cfg.window_clock else {
            return false;
        };
        self.window
            .as_ref()
            .and_then(|w| w.stamp.front())
            .is_some_and(|&s| self.cfg.closed.has_left(now, s, span))
    }

    /// Drop the rows of a clock window that have left at `now`, oldest
    /// first, each by the older block's exact subtraction (the block is formed
    /// from the newer when it has run out). True if any left. Each pass
    /// takes one row out, so the window's rows bound the passes.
    fn trim(&mut self, now: Stamp) -> bool {
        let mut dropped = false;
        for _ in 0..self.window_rows() {
            if !self.oldest_has_left(now) {
                break;
            }
            if self.window.as_ref().is_some_and(|w| w.old == 0) {
                self.form_old_block();
            } else {
                self.drop_oldest();
            }
            dropped = true;
        }
        dropped
    }

    /// A row's stamp: the one handed, or the clock summed from the steps
    /// where none was, for a window of clock units; and the clock moved on.
    fn stamp_row(&mut self, d_clock: f64, handed: Option<Stamp>) -> Option<Stamp> {
        let track = self.window.as_mut()?.track.as_mut()?;
        track.own += d_clock;
        track.last = handed;
        Some(handed.unwrap_or(Stamp::from(track.own)))
    }

    /// The stamp of a row `d_clock` after the last stepped, for a read ahead.
    fn stamp_ahead(&self, d_clock: f64) -> Option<Stamp> {
        let track = self.window.as_ref()?.track.as_ref()?;
        Some(match track.last {
            Some(last) => back_by(last, -d_clock),
            None => Stamp::from(track.own + d_clock),
        })
    }

    /// The intensities of the state as it stands, gated on `n_eff`, for a
    /// copy that [`Self::advance`] has brought to a row.
    fn read_here(&self, n_eff: f64) -> Step {
        let p = self.b.len() / self.cfg.n_targets;
        let mut pred = Vec::new();
        with_scratch(p, |sc| {
            Self::intensities_into(
                &self.cfg,
                n_eff,
                self.excited(0.0),
                &self.r,
                &self.b,
                &self.q,
                self.g0,
                sc,
                &mut pred,
            )
        });
        Step {
            pred,
            n_eff,
            extra: None,
        }
    }

    /// What a row `d_clock` after the last row stepped reads, at the
    /// stamp `now` (none without a clock window): the state advanced on a
    /// copy, the state untouched.
    fn read_ahead(&self, d_clock: f64, now: Option<Stamp>) -> Step {
        // A clock window drops, at the row, the events out of it by then:
        // on a copy, by the step's own routine.
        if let Some(now) = now.filter(|&s| self.oldest_has_left(s)) {
            let mut at = self.clone();
            at.advance(d_clock, Some(now));
            return at.read_here(self.n_eff());
        }
        let (n, u_n, _) = self.cfg.dims();
        let gamma = self.gamma();
        let f = self.cfg.decay.factor(d_clock);
        let mut g0 = f * self.g0 + Self::g0_step(gamma, d_clock);
        if let Some(span) = self.cfg.window_clock.filter(|&span| g0 > span) {
            g0 = span;
        }
        let mut pred = Vec::new();
        let p = self.b.len() / n;
        with_scratch(p, |sc| {
            let mut r = std::mem::take(&mut sc.r);
            let mut b = std::mem::take(&mut sc.b);
            let mut q = std::mem::take(&mut sc.q);
            r.clear();
            r.extend_from_slice(&self.r);
            for (u, beta) in self.cfg.kernel_decays.iter().enumerate() {
                let e = (-beta * d_clock).exp();
                for j in 0..n {
                    r[j * u_n + u] *= e;
                }
            }
            b.clear();
            b.extend(self.b.iter().map(|v| v * f));
            q.clear();
            q.extend(self.q.iter().map(|v| v * f));
            Self::intensities_into(
                &self.cfg,
                self.n_eff(),
                self.excited(d_clock),
                &r,
                &b,
                &q,
                g0,
                sc,
                &mut pred,
            );
            sc.r = r;
            sc.b = b;
            sc.q = q;
        });
        Step {
            pred,
            n_eff: self.n_eff(),
            extra: None,
        }
    }

    /// [`OnlineModel::predict`] for a row whose place on the clock the
    /// caller knows, `stamp`, as [`OnlineModel::stamp_next`] hands a
    /// step's: a window of clock units drops the rows that have left at
    /// that stamp, exactly as the row's step would. `predict` reads the
    /// window at the last stamp handed moved on by `d_clock`, a sum of two
    /// doubles, which puts a row exactly a span old on the other side of
    /// the edge on a number clock in tenths (`0.2 + 0.7` is not `0.9`), so
    /// a caller that stamps its steps reads its rows of weight 0 here.
    /// Without a clock window the stamp is not read.
    pub fn predict_at(&self, _x: &[f64], d_clock: f64, stamp: Stamp) -> Step {
        let now = self.cfg.window_clock.map(|_| stamp);
        self.read_ahead(d_clock, now)
    }

    /// [`OnlineModel::predict_with_floor`] for a row whose place on the
    /// clock the caller knows, `stamp`, as [`OnlineModel::stamp_next`]
    /// hands a step's.
    pub fn predict_with_floor_at(
        &mut self,
        x: &[f64],
        d_clock: f64,
        stamp: Stamp,
        floor: f64,
    ) -> Step {
        let prev = std::mem::replace(&mut self.cfg.min_weight, floor);
        let out = self.predict_at(x, d_clock, stamp);
        self.cfg.min_weight = prev;
        out
    }

    /// A row's stamp read without moving the clock.
    fn peek_stamp(&self, d_clock: f64, handed: Option<Stamp>) -> Option<Stamp> {
        let track = self.window.as_ref()?.track.as_ref()?;
        Some(handed.unwrap_or(Stamp::from(track.own + d_clock)))
    }

    /// Bytes a clock window would hold after trimming rows out at `now` and
    /// adding `extra` more.
    fn clock_bytes_at(&self, now: Stamp, extra: usize) -> u64 {
        let dropped = match (&self.window, self.cfg.window_clock) {
            (Some(w), Some(span)) => w
                .stamp
                .iter()
                .take_while(|&&s| self.cfg.closed.has_left(now, s, span))
                .count(),
            _ => 0,
        };
        let (n, u, _) = self.cfg.dims();
        Self::clock_state_bytes(n, u, self.window_rows() - dropped + extra)
    }

    /// Bytes a clock window holds with its newest row in, and whether the
    /// row that would add one more passes the cap.
    fn clock_bytes_with(&self, extra: usize) -> u64 {
        let (n, u, _) = self.cfg.dims();
        Self::clock_state_bytes(n, u, self.window_rows() + extra)
    }
}

impl OnlineModel for Hawkes {
    fn step_into(
        &mut self,
        _x: &[f64],
        y: &[Option<f64>],
        d_clock: f64,
        weight: f64,
        out: &mut Step,
    ) {
        // A target or a weight that is not usable, by the rule every model
        // keeps (`OnlineModel`): such a target is no event, and such a
        // weight learns nothing. The model reads no features.
        if let Some(refused) = crate::model::refused_step(self, &[], y, d_clock, weight) {
            *out = refused;
            return;
        }
        let n = self.cfg.n_targets;
        let handed = self.next.take();
        let peek_now = self.peek_stamp(d_clock, handed);
        let has_mark = weight > 0.0 && y.iter().take(n).any(|v| Self::mark(*v) > 0.0);
        // Check the clock window cap atomically before advancing clocks,
        // decaying state, or trimming expired rows.
        if self.over.is_some() {
            *out = self.read_ahead(d_clock, peek_now);
            return;
        }
        if has_mark && let Some(now) = peek_now {
            let (bytes, cap) = (self.clock_bytes_at(now, 1), self.cfg.cap());
            if !self.held_cap().admit(bytes, cap) {
                *out = self.read_ahead(d_clock, peek_now);
                return;
            }
        }
        let now = self.stamp_row(d_clock, handed);
        // Read before the row's own clock step, as every `n_eff` is (hard
        // rule 8): a row a clock window drops at this row still counts, as
        // weight not yet decayed does.
        let n_eff = self.n_eff();
        // The state to the row's clock, what `predict` reads on copies.
        self.advance(d_clock, now);
        let f = self.cfg.decay.factor(d_clock);
        let p = self.b.len() / n;
        let excited = self.excited(0.0);
        with_scratch(p, |sc| {
            Self::intensities_into(
                &self.cfg,
                n_eff,
                excited,
                &self.r,
                &self.b,
                &self.q,
                self.g0,
                sc,
                &mut out.pred,
            )
        });
        out.n_eff = n_eff;
        out.extra = None;
        if self.window.is_none() {
            self.w_sum = f * self.w_sum + weight;
        }
        // A row of weight 0 learns none of its events: it ages the state
        // over its step, as every model's does (`OnlineModel`). The stream
        // never hands one over, since such a row counts in no clock (hard
        // rule 9): it scores it with `predict` and steps the next row from
        // the last that counted.
        if !has_mark {
            return;
        }
        // Each event from `R` before the row's events (the left limit).
        let k = self.r.len();
        let mut marks = [0.0f64; 8];
        if n <= marks.len() {
            for (j, m) in marks.iter_mut().enumerate().take(n) {
                *m = Self::mark(y.get(j).copied().flatten());
            }
            Self::learn_marks(
                &self.cfg.kernel_decays,
                &marks[..n],
                &mut self.r,
                &mut self.b,
                &mut self.q,
            );
        } else {
            let marks: Vec<f64> = (0..n)
                .map(|j| Self::mark(y.get(j).copied().flatten()))
                .collect();
            Self::learn_marks(
                &self.cfg.kernel_decays,
                &marks,
                &mut self.r,
                &mut self.b,
                &mut self.q,
            );
        }
        debug_assert_eq!(self.r.len(), k);
        if self.window.is_some() {
            self.window_learn(y, weight, now);
            if now.is_some() {
                let (bytes, cap) = (self.clock_bytes_with(0), self.cfg.cap());
                self.held_cap().note(bytes, cap);
            }
        }
    }

    fn predict(&self, _x: &[f64], d_clock: f64) -> Step {
        self.read_ahead(d_clock, self.stamp_ahead(d_clock))
    }

    fn predict_with_floor(
        &mut self,
        x: &[f64],
        y: &[Option<f64>],
        d_clock: f64,
        floor: f64,
    ) -> Step {
        let prev = std::mem::replace(&mut self.cfg.min_weight, floor);
        let out = OnlineModel::predict_with(self, x, y, d_clock);
        self.cfg.min_weight = prev;
        out
    }

    /// A clock window learns its rows' stamps: they decide when each leaves.
    fn stamp_next(&mut self, stamp: Stamp) {
        if self.cfg.window_clock.is_some() {
            self.next = Some(stamp);
        }
    }

    fn take_state_warning(&mut self) -> Option<(u64, u64)> {
        self.warning.take().map(|bytes| (bytes, self.cfg.cap()))
    }

    fn state_over_cap(&self) -> Option<(u64, u64)> {
        self.over.map(|bytes| (bytes, self.cfg.cap()))
    }

    fn state(&self) -> State {
        State::new(ModelState::Hawkes(Box::new(self.clone())))
    }

    fn restore(s: &State) -> Result<Self, StateError> {
        check_schema(s)?;
        match &s.model {
            ModelState::Hawkes(m) => {
                let mut m = (**m).clone();
                crate::model::check_cfg("hawkes", m.cfg.validate())?;
                let (n, _, k) = m.cfg.dims();
                let shaped = m.r.len() == k
                    && m.b.len() == n * (1 + k)
                    && m.q.len() == n * (n + 1) / 2
                    && match (&m.window, m.cfg.window_events, m.cfg.window_clock) {
                        (None, None, None) => true,
                        (Some(w), cap, clock) if cap.is_some() || clock.is_some() => {
                            w.has_shape(n, k, cap, clock.is_some())
                        }
                        _ => false,
                    };
                if !shaped {
                    return Err(crate::model::damaged(
                        "hawkes",
                        "its running sums have the wrong shape",
                    ));
                }
                if let Some(w) = m.window.as_mut() {
                    w.rebase();
                    w.recount(n);
                }
                // A window of clock units never holds more than its cap
                // takes: a row that would pass it is not learned (review
                // 2026-11-23, HAWKES-3).
                if m.cfg.window_clock.is_some() && m.clock_bytes_with(0) > m.cfg.cap() {
                    return Err(crate::model::damaged(
                        "hawkes",
                        "its window holds more bytes than its max_state_bytes takes",
                    ));
                }
                Ok(m)
            }
            other => Err(StateError::WrongModel {
                expected: "hawkes",
                found: other.kind(),
            }),
        }
    }

    fn n_targets(&self) -> usize {
        self.cfg.n_targets
    }

    /// The nodes are the targets; the model reads no features.
    fn n_features(&self) -> usize {
        0
    }

    fn n_outputs(&self) -> usize {
        self.cfg.n_targets
    }
}

#[cfg(test)]
mod mutant_tests;
#[cfg(test)]
mod review_tests;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod window_tests;
