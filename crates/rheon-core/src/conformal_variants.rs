//! Conformal variants: asymmetric (signed) adaptive conformal trackers,
//! multi-level conformal slots with non-crossing bound reads, and the
//! standalone per-row conformal p-value ring.
//!
//! # Asymmetric signed conformal ([`SignedConformal`])
//!
//! Two independent quantile trackers on the signed scaled residual
//! `s_t = r_t / σ_t`, each targeting tail probability `α / 2` (matching
//! MAPIE's `AbsoluteConformityScore(sym=False)` with `β = α / 2`):
//!
//! ```text
//! e_hi  = 1{  r_t > q_hi }     m_hi' = m_hi + η (w / w̄) (e_hi − α / 2)
//! e_lo  = 1{ −r_t > q_lo }     m_lo' = m_lo + η (w / w̄) (e_lo − α / 2)
//! ```
//!
//! with shown interval `[pred − m_lo · σ, pred + m_hi · σ]` and **no** clamp
//! at zero (a skewed residual distribution can place one side's quantile on
//! the opposite side of `pred`). Both multipliers start at `Φ⁻¹(1 − α / 2)`
//! (portable [`crate::conformal::upper_z`]).
//!
//! # Multi-level conformal ([`ConformalSlot`])
//!
//! Independent trackers per nominal coverage level `c_0 < c_1 < … < c_{L-1}`.
//! At read time, [`sort_offsets_non_crossing`] sorts lower and upper offsets
//! non-decreasingly in level so lower bounds are non-increasing and upper
//! bounds are non-decreasing (nested intervals never cross), with zero extra
//! state.
//!
//! # Standalone per-row conformal p-value ([`ConformalPRing`])
//!
//! Unsmoothed conformal p-value (matching `crepes`' `predict_p_online` with
//! `smoothing=False`), read **before** the row enters the ring:
//!
//! ```text
//! p_t = (Σ_i ω_i 1{a_i ≥ a_t} + 1) / (Σ_i ω_i + 1),   a = |resid| / σ
//! ```
//!
//! where `ω_i = w_i · decay.factor(age_i)` over the active window of past
//! positive-weight scores. Zero-weight rows never enter the ring (hard rule 9).

use std::cmp::Ordering;
use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

use crate::clock::{Decay, Stamp};
use crate::clock_span::{DEFAULT_STATE_BYTES, HeldCap, Place};
use crate::conformal::{Conformal, HELD_GAIN_BOUND, gauss_gain, upper_z};

/// Default row window for `emit_conformal_p` when neither
/// `conformal_p_window_rows` nor `conformal_p_window_size` is given.
pub const DEFAULT_CONFORMAL_P_WINDOW_ROWS: usize = 1000;

/// Fixed per-ring accounting bytes for [`ConformalPRing`].
pub const CONFORMAL_P_FIXED_BYTES: u64 = 64;

/// Platform-independent accounting bytes per stored score in [`ConformalPRing`]
/// (score + clock stamp + weight: 24 bytes).
pub const CONFORMAL_P_ENTRY_BYTES: u64 = 24;

fn is_zero(v: &f64) -> bool {
    *v == 0.0
}

/// Asymmetric (two-sided signed) adaptive conformal tracker for one slot and
/// one nominal coverage level `1 − α`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(from = "SavedSigned")]
pub struct SignedConformal {
    alpha: f64,
    rate: f64,
    /// Lower-side multiplier (`lo = pred - m_lo * sigma`), `NaN` until started.
    /// May become negative (no 0-clamp).
    m_lo: f64,
    /// Upper-side multiplier (`hi = pred + m_hi * sigma`), `NaN` until started.
    /// May become negative (no 0-clamp).
    m_hi: f64,
    /// EW realized coverage of the un-crossed emitted interval.
    cov: f64,
    cov_w: f64,
    #[serde(default)]
    w_mean: f64,
    #[serde(default)]
    w_rows: f64,
    #[serde(default, skip_serializing_if = "is_zero")]
    held_mean: f64,
    #[serde(default, skip_serializing_if = "is_zero")]
    released: f64,
    /// One-sided Gaussian tail density `φ(Φ⁻¹(1 − α/2)) = ½ g(α)` for the
    /// delay-aware step (`conformal_rate = "auto"`). Derived, never written.
    #[serde(skip_serializing)]
    gain: f64,
    /// Starting multiplier `Φ⁻¹(1 − α/2)`. Derived, never written.
    #[serde(skip_serializing)]
    z: f64,
}

#[derive(Deserialize)]
struct SavedSigned {
    alpha: f64,
    rate: f64,
    m_lo: f64,
    m_hi: f64,
    cov: f64,
    cov_w: f64,
    #[serde(default)]
    w_mean: f64,
    #[serde(default)]
    w_rows: f64,
    #[serde(default)]
    held_mean: f64,
    #[serde(default)]
    released: f64,
}

impl From<SavedSigned> for SignedConformal {
    fn from(s: SavedSigned) -> Self {
        Self {
            alpha: s.alpha,
            rate: s.rate,
            m_lo: s.m_lo,
            m_hi: s.m_hi,
            cov: s.cov,
            cov_w: s.cov_w,
            w_mean: s.w_mean,
            w_rows: s.w_rows,
            held_mean: s.held_mean,
            released: s.released,
            gain: 0.5 * gauss_gain(s.alpha),
            z: upper_z(s.alpha),
        }
    }
}

impl PartialEq for SignedConformal {
    fn eq(&self, o: &Self) -> bool {
        let bits = |c: &Self| {
            [
                c.alpha,
                c.rate,
                c.m_lo,
                c.m_hi,
                c.cov,
                c.cov_w,
                c.w_mean,
                c.w_rows,
                c.held_mean,
                c.released,
                c.gain,
                c.z,
            ]
            .map(f64::to_bits)
        };
        bits(self) == bits(o)
    }
}

impl SignedConformal {
    /// Construct a fresh signed tracker at target `coverage = 1 − α ∈ (0, 1)`
    /// and step `rate > 0`.
    pub fn new(coverage: f64, rate: f64) -> Result<Self, String> {
        let mut no = crate::Refusals::new();
        if !(coverage > 0.0 && coverage < 1.0) {
            no.refuse(format!(
                "conformal must be above 0 and below 1, but it is {coverage}. Set conformal to \
                 the share of rows whose target the interval should cover, such as 0.9."
            ));
        }
        if !(rate > 0.0 && rate.is_finite()) {
            no.refuse(format!(
                "conformal_rate must be finite and above 0, but it is {rate}. Set \
                 conformal_rate to how fast the interval's width adapts, such as the default \
                 0.05."
            ));
        }
        no.finish()?;
        let alpha = 1.0 - coverage;
        Ok(Self {
            alpha,
            rate,
            m_lo: f64::NAN,
            m_hi: f64::NAN,
            cov: 0.0,
            cov_w: 0.0,
            w_mean: 0.0,
            w_rows: 0.0,
            held_mean: 0.0,
            released: 0.0,
            gain: 0.5 * gauss_gain(alpha),
            z: upper_z(alpha),
        })
    }

    /// Check deserialized invariants. Unlike symmetric [`Conformal`], `m_lo`
    /// and `m_hi` may be negative (no 0-clamp), and start/unset together.
    pub fn has_shape(&self) -> bool {
        let unset_or_finite = |v: f64| v.is_nan() || v.is_finite();
        let held = |v: f64| v.is_finite() && v >= 0.0;
        self.alpha > 0.0
            && self.alpha < 1.0
            && self.rate.is_finite()
            && self.rate > 0.0
            && unset_or_finite(self.m_lo)
            && unset_or_finite(self.m_hi)
            && self.m_lo.is_nan() == self.m_hi.is_nan()
            && (0.0..=1.0).contains(&self.cov)
            && held(self.cov_w)
            && held(self.w_mean)
            && held(self.w_rows)
            && held(self.held_mean)
            && held(self.released)
    }

    /// Raw lower and upper offsets `(q_lo, q_hi) = (m_lo · σ, m_hi · σ)` such
    /// that the interval is `[pred − q_lo, pred + q_hi]`.
    pub fn offsets(&self, sigma: f64) -> Option<(f64, f64)> {
        if self.m_lo.is_finite() && self.m_hi.is_finite() && sigma.is_finite() && sigma > 0.0 {
            Some((self.m_lo * sigma, self.m_hi * sigma))
        } else {
            None
        }
    }

    /// Current signed multipliers `(m_lo, m_hi)` once started.
    pub fn multipliers(&self) -> Option<(f64, f64)> {
        (self.m_lo.is_finite() && self.m_hi.is_finite()).then_some((self.m_lo, self.m_hi))
    }

    /// Realized EW coverage of the shown (un-crossed) interval.
    pub fn coverage(&self) -> Option<f64> {
        (self.cov_w > 0.0).then_some(self.cov)
    }

    /// Nominal miscoverage `α = 1 − coverage`.
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Fold one row in without delay (`shown = None, emitted = None, held = 0`).
    pub fn update(&mut self, resid: f64, sigma: f64, lam: f64, w: f64) {
        self.update_against(resid, sigma, lam, w, None, None, 0);
    }

    /// Fold one row in, optionally against delayed shown offsets.
    ///
    /// - `shown`: the tracker's own `(q_lo, q_hi)` at score time (used for the
    ///   independent tail recursion so each level preserves its telescoping
    ///   guarantee). `None` reads `(m_lo * sigma, m_hi * sigma)` now.
    /// - `emitted`: the emitted (sorted across levels, if multi-level) `(q_lo,
    ///   q_hi)` shown to the caller (un-crossed with `min`/`max` when scoring
    ///   realized `coverage`). `None` uses `shown` / current offsets.
    #[allow(clippy::too_many_arguments)]
    pub fn update_against(
        &mut self,
        resid: f64,
        sigma: f64,
        lam: f64,
        w: f64,
        shown: Option<(f64, f64)>,
        emitted: Option<(f64, f64)>,
        held: usize,
    ) {
        if !(resid.is_finite() && w > 0.0 && w.is_finite()) {
            self.cov_w *= lam;
            self.w_rows *= lam;
            return;
        }
        self.w_rows *= lam;
        self.w_mean = crate::cluster::mean_weight(&mut self.w_rows, self.w_mean, w);
        if held > 0 || self.held_mean != 0.0 {
            self.held_mean += (held as f64 - self.held_mean) / self.w_rows;
        }
        if !self.m_lo.is_finite() {
            self.released += 1.0;
            if self.released >= 1.0 + self.held_mean {
                self.m_lo = self.z;
                self.m_hi = self.z;
                self.released = 0.0;
            }
            self.cov_w *= lam;
            return;
        }
        let (q_lo, q_hi) = match shown {
            Some(pair) => pair,
            None if sigma.is_finite() && sigma > 0.0 => (self.m_lo * sigma, self.m_hi * sigma),
            None => (f64::NAN, f64::NAN),
        };
        if !(q_lo.is_finite() && q_hi.is_finite()) {
            self.cov_w *= lam;
            return;
        }
        // Realized coverage uses un-crossed bounds of the emitted interval.
        let (eq_lo, eq_hi) = emitted.unwrap_or((q_lo, q_hi));
        if eq_lo.is_finite() && eq_hi.is_finite() {
            let l_off = (-eq_lo).min(eq_hi);
            let u_off = (-eq_lo).max(eq_hi);
            let hit = resid >= l_off && resid <= u_off;
            let cw = lam * self.cov_w + w;
            self.cov = (lam * self.cov_w * self.cov + w * f64::from(hit)) / cw;
            self.cov_w = cw;
        } else {
            self.cov_w *= lam;
        }
        // Independent signed tail updates (no 0-clamp).
        let miss_hi = resid > q_hi;
        let miss_lo = -resid > q_lo;
        let mut eta = self.rate;
        if self.held_mean > 0.0 {
            eta /= 1.0 + self.rate * self.gain * self.held_mean / HELD_GAIN_BOUND;
        }
        let rel = if self.w_mean > 0.0 {
            w / self.w_mean
        } else {
            1.0
        };
        let half_alpha = 0.5 * self.alpha;
        self.m_hi += eta * rel * (f64::from(miss_hi) - half_alpha);
        self.m_lo += eta * rel * (f64::from(miss_lo) - half_alpha);
        if !(self.m_lo.is_finite() && self.m_hi.is_finite()) {
            self.m_lo = self.z;
            self.m_hi = self.z;
        }
    }
}

/// Sort lower offsets `q_lo` and upper offsets `q_hi` in ascending order across
/// levels (which are sorted ascending in nominal coverage `c_0 < c_1 < …`).
///
/// Since `lo[l] = pred - q_lo[l]` and `hi[l] = pred + q_hi[l]`, ascending
/// `q_lo` makes `lo` non-increasing and ascending `q_hi` makes `hi`
/// non-decreasing, so nested intervals never cross. If any level is non-finite,
/// the slices are left unchanged.
pub fn sort_offsets_non_crossing(q_lo: &mut [f64], q_hi: &mut [f64]) {
    if q_lo.len() <= 1 {
        return;
    }
    if q_lo.iter().all(|v| v.is_finite()) {
        q_lo.sort_by(f64::total_cmp);
    }
    if q_hi.iter().all(|v| v.is_finite()) {
        q_hi.sort_by(f64::total_cmp);
    }
}

/// Multi-level or asymmetric conformal state for one output slot.
///
/// Legacy single-level symmetric conformal stays in `Persisted::conformal`
/// (`Vec<Vec<Conformal>>`) untouched; this enum backs multi-level symmetric
/// conformal and single- or multi-level asymmetric conformal.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ConformalSlot {
    /// One symmetric [`Conformal`] tracker per sorted nominal coverage level.
    Symmetric(Vec<Conformal>),
    /// One asymmetric [`SignedConformal`] tracker per sorted nominal coverage level.
    Asymmetric(Vec<SignedConformal>),
}

impl ConformalSlot {
    /// Construct a fresh slot for sorted coverage `levels`, step `rate`, and
    /// `symmetric` flag.
    pub fn new(levels: &[f64], rate: f64, symmetric: bool) -> Result<Self, String> {
        if symmetric {
            let v = levels
                .iter()
                .map(|&c| Conformal::new(c, rate))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(Self::Symmetric(v))
        } else {
            let v = levels
                .iter()
                .map(|&c| SignedConformal::new(c, rate))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(Self::Asymmetric(v))
        }
    }

    /// Number of conformal levels.
    pub fn len(&self) -> usize {
        match self {
            Self::Symmetric(v) => v.len(),
            Self::Asymmetric(v) => v.len(),
        }
    }

    /// True if empty (never happens for validated specs).
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Check deserialized shape against a freshly constructed slot.
    pub fn matches_shape(&self, expected: &Self) -> bool {
        match (self, expected) {
            (Self::Symmetric(a), Self::Symmetric(b)) => {
                a.len() == b.len() && a.iter().all(Conformal::has_shape)
            }
            (Self::Asymmetric(a), Self::Asymmetric(b)) => {
                a.len() == b.len() && a.iter().all(SignedConformal::has_shape)
            }
            _ => false,
        }
    }

    /// Raw (pre-sort) `(q_lo, q_hi)` offsets per level into `q_lo` and `q_hi`.
    pub fn raw_offsets(&self, sigma: f64, q_lo: &mut Vec<f64>, q_hi: &mut Vec<f64>) {
        q_lo.clear();
        q_hi.clear();
        match self {
            Self::Symmetric(v) => {
                for c in v {
                    let q = c.radius(sigma).unwrap_or(f64::NAN);
                    q_lo.push(q);
                    q_hi.push(q);
                }
            }
            Self::Asymmetric(v) => {
                for c in v {
                    let (lo, hi) = c.offsets(sigma).unwrap_or((f64::NAN, f64::NAN));
                    q_lo.push(lo);
                    q_hi.push(hi);
                }
            }
        }
    }

    /// Realized coverage at level index `li`.
    pub fn coverage(&self, li: usize) -> Option<f64> {
        match self {
            Self::Symmetric(v) => v.get(li).and_then(Conformal::coverage),
            Self::Asymmetric(v) => v.get(li).and_then(SignedConformal::coverage),
        }
    }

    /// Update every level's tracker.
    ///
    /// `shown` (if `Some`) carries `2 * L` floats `[q_lo_0, q_hi_0, q_lo_1,
    /// q_hi_1, ...]` from score time (pre-sort); `emitted_lo`/`emitted_hi`
    /// carry the sorted offsets shown to the caller.
    #[allow(clippy::too_many_arguments)]
    pub fn update_against(
        &mut self,
        resid: f64,
        sigma: f64,
        lam: f64,
        w: f64,
        shown: Option<&[f64]>,
        emitted: Option<(&[f64], &[f64])>,
        held: usize,
    ) {
        match self {
            Self::Symmetric(v) => {
                for (li, c) in v.iter_mut().enumerate() {
                    let s_rad = shown.map(|s| s[2 * li]);
                    let e_pair = emitted.map(|(lo, hi)| (lo[li], hi[li]));
                    c.update_against_emitted(resid, sigma, lam, w, s_rad, e_pair, held);
                }
            }
            Self::Asymmetric(v) => {
                for (li, c) in v.iter_mut().enumerate() {
                    let s_pair = shown.map(|s| (s[2 * li], s[2 * li + 1]));
                    let e_pair = emitted.map(|(lo, hi)| (lo[li], hi[li]));
                    c.update_against(resid, sigma, lam, w, s_pair, e_pair, held);
                }
            }
        }
    }
}

/// Window policy for [`ConformalPRing`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ConformalPWindow {
    /// Keep at most the last `n` positive-weight calibration scores.
    Rows(usize),
    /// Keep scores whose clock age is strictly less than `span` (`closed = "right"`).
    /// `span_ns` holds exact integer nanoseconds for temporal durations.
    Clock { span: f64, span_ns: Option<i128> },
}

/// One stored calibration score in [`ConformalPRing`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ConformalPEntry {
    pub score: f64,
    pub stamp: Stamp,
    pub weight: f64,
}

/// Standalone per-row conformal p-value ring for one output slot.
///
/// Computes the unsmoothed out-of-sample conformal p-value
/// `(Σ ω_i 1{a_i ≥ a_t} + 1) / (Σ ω_i + 1)` over the active window of past
/// scaled residuals `a = |resid| / σ`, with `ω_i = w_i · decay.factor(age_i)`.
///
/// Bounded by `max_state_bytes` through atomic [`HeldCap`]: warns once past
/// half the cap and refuses atomically before exceeding it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConformalPRing {
    window: ConformalPWindow,
    cap_bytes: u64,
    entries: VecDeque<ConformalPEntry>,
    place: Place,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    over: Option<u64>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    warned: bool,
    #[serde(skip)]
    warning: Option<u64>,
}

impl ConformalPRing {
    /// Create a fresh ring with `window` and per-ring byte budget `cap_bytes`.
    pub fn new(window: ConformalPWindow, cap_bytes: u64) -> Self {
        Self {
            window,
            cap_bytes: cap_bytes.max(CONFORMAL_P_FIXED_BYTES),
            entries: VecDeque::new(),
            place: Place::default(),
            over: None,
            warned: false,
            warning: None,
        }
    }

    /// Accounting bytes for a ring holding `n` entries.
    pub fn bytes_for(n: usize) -> u64 {
        CONFORMAL_P_FIXED_BYTES.saturating_add((n as u64).saturating_mul(CONFORMAL_P_ENTRY_BYTES))
    }

    /// Current accounted bytes.
    pub fn state_bytes(&self) -> u64 {
        Self::bytes_for(self.entries.len())
    }

    /// Configured cap bytes for this ring (default [`DEFAULT_STATE_BYTES`]).
    pub fn cap_bytes(&self) -> u64 {
        self.cap_bytes
    }

    /// Reset all stored scores and clock state, keeping window and cap.
    pub fn reset(&mut self) {
        self.entries.clear();
        self.place = Place::default();
        self.over = None;
        self.warned = false;
        self.warning = None;
    }

    /// Number of stored scores currently in the ring.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether no score is stored.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Check deserialized invariants against a freshly constructed ring.
    pub fn matches_shape(&self, expected: &Self) -> bool {
        let valid_window = match self.window {
            ConformalPWindow::Rows(n) => n >= 1 && self.entries.len() <= n,
            ConformalPWindow::Clock { span, .. } => span.is_finite() && span > 0.0,
        };
        valid_window
            && self.window == expected.window
            && self.cap_bytes == expected.cap_bytes
            && self.entries.iter().all(|e| {
                e.score.is_finite() && e.score >= 0.0 && e.weight.is_finite() && e.weight > 0.0
            })
    }

    /// If a row would have taken this ring past its byte cap, returns
    /// `(attempted_bytes, cap_bytes)`.
    pub fn state_over_cap(&self) -> Option<(u64, u64)> {
        self.over.map(|b| (b, self.cap_bytes))
    }

    /// Drain the once-per-group warning raised when state crosses half of its cap.
    pub fn take_state_warning(&mut self) -> Option<(u64, u64)> {
        self.warning.take().map(|b| (b, self.cap_bytes))
    }

    /// Whether entry `older` has expired at `now` under a clock window
    /// (`closed = "right"`: age `>= span` is outside).
    #[inline]
    fn expired(&self, now: Stamp, older: Stamp) -> bool {
        match self.window {
            ConformalPWindow::Rows(_) => false,
            ConformalPWindow::Clock { span, span_ns } => {
                now.cmp_span_ns(older, span, span_ns) != Ordering::Less
            }
        }
    }

    /// Evaluate the out-of-sample unsmoothed conformal p-value for scaled score
    /// `a_t = |resid_t| / σ_t` read at `now` (from `stamp` if provided, else
    /// `place.peek(d_clock)`), without mutating the ring.
    ///
    /// Returns `None` if `a_t` is not finite/non-negative or if no active
    /// calibration entry remains in the window.
    pub fn p_value_at(
        &self,
        a_t: f64,
        d_clock: f64,
        stamp: Option<Stamp>,
        decay: Decay,
    ) -> Option<f64> {
        if !(a_t.is_finite() && a_t >= 0.0) || self.entries.is_empty() {
            return None;
        }
        let now = stamp.unwrap_or_else(|| self.place.peek(d_clock));
        let mut ge = 0.0;
        let mut total = 0.0;
        let mut active = 0usize;
        for e in &self.entries {
            if self.expired(now, e.stamp) {
                continue;
            }
            active += 1;
            let age = now.since(e.stamp).max(0.0);
            let om = e.weight * decay.factor(age);
            if om > 0.0 && om.is_finite() {
                total += om;
                if e.score >= a_t {
                    ge += om;
                }
            }
        }
        (active > 0).then_some((ge + 1.0) / (total + 1.0))
    }

    /// Advance the ring's clock by `(d_clock, stamp)` and push a positive-weight
    /// score `a_t = |resid| / σ` if finite.
    ///
    /// - Zero-weight rows (`w <= 0.0`) do nothing and leave the clock where it
    ///   was (hard rule 9).
    /// - Non-finite scores (warmup / null target) advance the clock and expire
    ///   aged entries without storing a score.
    /// - Under `ConformalPWindow::Clock`, expiry and the [`HeldCap`] check run
    ///   atomically: if the new entry would exceed `cap_bytes`, `self.entries`
    ///   and `self.place` are NOT mutated and `self.over` records the attempted
    ///   size.
    pub fn push(&mut self, a_t: f64, d_clock: f64, stamp: Option<Stamp>, w: f64) {
        if !(w > 0.0 && w.is_finite()) {
            return;
        }
        if self.over.is_some() {
            return;
        }
        let now = stamp.unwrap_or_else(|| self.place.peek(d_clock));
        let drop_count = match self.window {
            ConformalPWindow::Rows(max_n) => {
                if a_t.is_finite() && a_t >= 0.0 && self.entries.len() >= max_n {
                    self.entries.len() - max_n + 1
                } else {
                    0
                }
            }
            ConformalPWindow::Clock { .. } => self
                .entries
                .iter()
                .take_while(|e| self.expired(now, e.stamp))
                .count(),
        };
        let add_one = usize::from(a_t.is_finite() && a_t >= 0.0);
        let projected_len = self.entries.len() - drop_count + add_one;
        let with_bytes = Self::bytes_for(projected_len);
        let cap_bytes = self.cap_bytes;
        let mut hc = HeldCap {
            over: &mut self.over,
            warned: &mut self.warned,
            warning: &mut self.warning,
        };
        if add_one > 0 && !hc.admit(with_bytes, cap_bytes) {
            return;
        }
        // Admitted: commit expiry, clock advance, push, and warning check.
        for _ in 0..drop_count {
            self.entries.pop_front();
        }
        if let Some(s) = stamp {
            self.place.hand(s);
        }
        let stepped = self.place.step(d_clock);
        if add_one > 0 {
            self.entries.push_back(ConformalPEntry {
                score: a_t,
                stamp: stepped,
                weight: w,
            });
            let bytes = Self::bytes_for(self.entries.len());
            let mut hc = HeldCap {
                over: &mut self.over,
                warned: &mut self.warned,
                warning: &mut self.warning,
            };
            hc.note(bytes, cap_bytes);
        }
    }

    /// Default ring budget for a single-slot spec.
    pub fn default_cap_bytes() -> u64 {
        DEFAULT_STATE_BYTES
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signed_conformal_telescoping_identity_and_negative_multiplier() {
        let mut c = SignedConformal::new(0.9, 0.05).unwrap();
        assert!(c.has_shape());
        assert_eq!(c.offsets(1.0), None);
        assert_eq!(c.coverage(), None);

        // First row starts the tracker at z = Phi^-1(0.95).
        c.update(0.1, 1.0, 1.0, 1.0);
        let (m0_lo, m0_hi) = c.multipliers().unwrap();
        assert_eq!(m0_lo, m0_hi);
        assert!((m0_lo - 1.6448536269514722).abs() < 1e-8);

        // Feed strictly positive residuals (skewed right): m_lo drops below 0
        // (no 0-clamp!) while m_hi rises.
        let mut sum_lo = 0.0;
        let mut sum_hi = 0.0;
        for i in 0..800 {
            let r = 2.0 + (i % 5) as f64 * 0.5;
            let (q_lo, q_hi) = c.offsets(1.0).unwrap();
            let miss_lo = f64::from(-r > q_lo);
            let miss_hi = f64::from(r > q_hi);
            sum_lo += miss_lo - 0.05;
            sum_hi += miss_hi - 0.05;
            c.update(r, 1.0, 1.0, 1.0);
        }
        let (m_lo, m_hi) = c.multipliers().unwrap();
        assert!(m_lo < 0.0, "signed tracker must allow m_lo < 0: {m_lo}");
        assert!(m_hi > 2.0, "m_hi tracks upper tail: {m_hi}");
        assert!(
            ((m_lo - m0_lo) - 0.05 * sum_lo).abs() < 1e-11,
            "telescoping m_lo"
        );
        assert!(
            ((m_hi - m0_hi) - 0.05 * sum_hi).abs() < 1e-11,
            "telescoping m_hi"
        );
        assert!(c.has_shape());
    }

    #[test]
    fn sort_offsets_prevents_nested_crossing() {
        let mut q_lo = [2.1, 1.8, 2.5];
        let mut q_hi = [1.9, 2.4, 2.0];
        sort_offsets_non_crossing(&mut q_lo, &mut q_hi);
        assert_eq!(q_lo, [1.8, 2.1, 2.5]);
        assert_eq!(q_hi, [1.9, 2.0, 2.4]);
        // Lower bounds pred - q_lo are non-increasing; upper bounds pred + q_hi non-decreasing.
        let pred = 10.0;
        let los: Vec<f64> = q_lo.iter().map(|&q| pred - q).collect();
        let his: Vec<f64> = q_hi.iter().map(|&q| pred + q).collect();
        assert!(los.windows(2).all(|w| w[0] >= w[1]));
        assert!(his.windows(2).all(|w| w[0] <= w[1]));
    }

    #[test]
    fn conformal_p_ring_exact_rank_decay_and_rule9() {
        let mut ring = ConformalPRing::new(ConformalPWindow::Rows(100), DEFAULT_STATE_BYTES);
        let inf = Decay::Halflife(f64::INFINITY);
        assert_eq!(ring.p_value_at(1.0, 0.0, None, inf), None);

        // Zero-weight first row leaves ring empty (Rule 9).
        ring.push(0.5, 0.0, None, 0.0);
        assert!(ring.is_empty());

        let scores = [0.2, 0.8, 1.5, 0.4, 2.0];
        for (i, &a) in scores.iter().enumerate() {
            ring.push(a, if i == 0 { 0.0 } else { 1.0 }, None, 1.0);
        }
        // Score 0.8: entries >= 0.8 are {0.8, 1.5, 2.0} (3 of 5) -> (3 + 1)/(5 + 1) = 4/6.
        let p = ring.p_value_at(0.8, 1.0, None, inf).unwrap();
        assert!((p - 4.0 / 6.0).abs() < 1e-15);

        // Zero-weight row does not enter ring or advance clock.
        ring.push(99.0, 1.0, None, 0.0);
        assert_eq!(ring.len(), 5);
        assert!((ring.p_value_at(0.8, 1.0, None, inf).unwrap() - 4.0 / 6.0).abs() < 1e-15);

        // Decayed weighted check against explicit sum.
        let hl = 5.0;
        let dec = Decay::Halflife(hl);
        let mut wring = ConformalPRing::new(ConformalPWindow::Rows(100), DEFAULT_STATE_BYTES);
        let ws = [1.0, 2.0, 0.5, 1.5, 0.8];
        for i in 0..5 {
            wring.push(scores[i], if i == 0 { 0.0 } else { 1.0 }, None, ws[i]);
        }
        let a_t = 0.8;
        let got = wring.p_value_at(a_t, 1.0, None, dec).unwrap();
        let mut ge = 0.0;
        let mut tot = 0.0;
        for i in 0..5 {
            let age = (5 - i) as f64;
            let om = ws[i] * 0.5_f64.powf(age / hl);
            tot += om;
            if scores[i] >= a_t {
                ge += om;
            }
        }
        let want = (ge + 1.0) / (tot + 1.0);
        assert!((got - want).abs() < 1e-15, "{got} vs {want}");
    }

    #[test]
    fn conformal_p_ring_clock_window_and_atomic_held_cap() {
        // Cap allows 4 entries: 64 + 4*24 = 160 bytes; half is 80 bytes (entry 1 = 88 > 80 warns!).
        let cap = ConformalPRing::bytes_for(4);
        let mut ring = ConformalPRing::new(
            ConformalPWindow::Clock {
                span: 10.0,
                span_ns: None,
            },
            cap,
        );
        let inf = Decay::Halflife(f64::INFINITY);
        for i in 0..4 {
            ring.push(i as f64, 1.0, Some(Stamp::from(i as f64)), 1.0);
        }
        assert!(ring.take_state_warning().is_some());
        assert_eq!(ring.state_over_cap(), None);
        assert_eq!(ring.len(), 4);

        // 5th entry within span 10 would exceed cap: refused atomically!
        let snapshot = ring.entries.clone();
        ring.push(4.0, 1.0, Some(Stamp::from(4.0)), 1.0);
        assert_eq!(
            ring.state_over_cap(),
            Some((ConformalPRing::bytes_for(5), cap))
        );
        assert_eq!(
            ring.entries, snapshot,
            "atomic refusal leaves entries unchanged"
        );

        // Fresh ring: stamp-aware read expires entries >= span old without mutating.
        let mut ring2 = ConformalPRing::new(
            ConformalPWindow::Clock {
                span: 10.0,
                span_ns: None,
            },
            DEFAULT_STATE_BYTES,
        );
        ring2.push(0.1, 0.0, Some(Stamp::from(0.0)), 1.0);
        ring2.push(2.0, 5.0, Some(Stamp::from(5.0)), 1.0);
        // At t = 10.0, t=0.0 is age 10.0 >= 10.0 (expired under right-closed), only t=5.0 (2.0) active.
        let p = ring2
            .p_value_at(1.0, 5.0, Some(Stamp::from(10.0)), inf)
            .unwrap();
        assert!((p - (1.0 + 1.0) / (1.0 + 1.0)).abs() < 1e-15);
        assert_eq!(ring2.len(), 2, "read did not mutate ring");
        // At t = 15.0, both entries expired -> None.
        assert_eq!(
            ring2.p_value_at(1.0, 10.0, Some(Stamp::from(15.0)), inf),
            None
        );
    }
}
