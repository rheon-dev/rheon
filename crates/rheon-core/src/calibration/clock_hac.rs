//! Clock-duration Newey-West Bartlett HAC accumulators for [`Calibration`]
//! and [`Coverage`] under a clocked `embargo` with no explicit `horizon_rows`.
//!
//! With bandwidth `B = 2 * embargo` (in clock units; seconds on a temporal
//! clock, held as exact integer nanoseconds `bandwidth_ns` when `embargo` is a
//! duration), each pair of observations `(t, s)` with `0 <= tau_t - tau_s < B`
//! enters the long-run variance with Bartlett weight
//!
//! ```text
//! w(dt) = 1 - dt / B,    0 <= dt < B.
//! ```
//!
//! On a unit-step numeric clock `t = 0, 1, 2, ...` with integer `embargo = E`,
//! `B = 2 E`, so lags `l = 1, ..., 2E - 1` receive weights
//! `1 - l / (2E) = 1 - l / ((2E - 1) + 1)`, matching discrete Newey-West HAC
//! at `maxlags = 2E - 1` exactly. On irregular or temporal clocks, the kernel
//! is strictly positive definite (the autocorrelation of a boxcar of width `B`)
//! and scale-invariant in clock units.
//!
//! State size is bounded by `max_state_bytes` through atomic [`HeldCap`]:
//! warns once past half the cap and refuses atomically before exceeding it.
//! Zero-weight rows neither enter the ring nor advance clock stamps (hard
//! rule 9).

use std::cmp::Ordering;
use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

use crate::Stamp;
use crate::clock_span::{HeldCap, Place};

/// Fixed per-slot accounting bytes for [`ClockHac`].
pub const CLOCK_HAC_FIXED_BYTES: u64 = 160;

/// Per-ring-entry accounting bytes for [`ClockHac`] (`Stamp` + `[f64; 2]`).
pub const CLOCK_HAC_ENTRY_BYTES: u64 = 48;

/// Fixed per-slot accounting bytes for [`ClockCoverageHac`].
pub const CLOCK_COVERAGE_HAC_FIXED_BYTES: u64 = 128;

/// Per-ring-entry accounting bytes for [`ClockCoverageHac`] (`Stamp` + `f64`).
pub const CLOCK_COVERAGE_HAC_ENTRY_BYTES: u64 = 40;

/// Whether `stamp` holds only finite floating-point components.
fn stamp_is_finite(stamp: Stamp) -> bool {
    match stamp {
        Stamp::Ns(_) => true,
        Stamp::Raw(a, r) => a.is_finite() && r.is_finite(),
        Stamp::Int(_, f) => f.is_finite() && f.abs() <= 1.0,
    }
}

/// Exact `dt / bandwidth` in `[0, 1)`, using integer nanoseconds when both
/// stamps are `Stamp::Ns` and `bandwidth_ns` is available.
#[inline]
fn dt_ratio(now: Stamp, older: Stamp, bandwidth: f64, bandwidth_ns: Option<i128>) -> f64 {
    if let (Stamp::Ns(a), Stamp::Ns(b), Some(bw_ns)) = (now, older, bandwidth_ns)
        && bw_ns > 0
    {
        return (a.saturating_sub(b).max(0) as f64) / (bw_ns as f64);
    }
    now.since(older).max(0.0) / bandwidth
}

/// Strict 2×2 SPD Cholesky factorization `M = L L^T` with no jitter.
/// Returns `(l00, l10, l11)` or `None` if `M` is not strictly positive definite.
#[inline]
fn chol2(m: &[f64; 4]) -> Option<(f64, f64, f64)> {
    let m00 = m[0];
    let m01 = 0.5 * (m[1] + m[2]);
    let m11 = m[3];
    if !(m00 > 0.0 && m00.is_finite() && m01.is_finite() && m11.is_finite()) {
        return None;
    }
    let l00 = m00.sqrt();
    let l10 = m01 / l00;
    let d11 = m11 - l10 * l10;
    if !(d11 > 0.0 && d11.is_finite()) {
        return None;
    }
    let l11 = d11.sqrt();
    (l00 > 0.0 && l11 > 0.0 && l00.is_finite() && l10.is_finite() && l11.is_finite())
        .then_some((l00, l10, l11))
}

/// Solve `L L^T x = b` for a 2×2 lower triangular Cholesky factor `(l00, l10, l11)`.
#[inline]
fn chol2_solve(l: (f64, f64, f64), b: [f64; 2]) -> [f64; 2] {
    let (l00, l10, l11) = l;
    let y0 = b[0] / l00;
    let y1 = (b[1] - l10 * y0) / l11;
    let x1 = y1 / l11;
    let x0 = (y0 - l10 * x1) / l00;
    [x0, x1]
}

/// One scored row held inside [`ClockHac`]'s bandwidth ring.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ClockHacEntry {
    pub stamp: Stamp,
    pub u: [f64; 2],
}

/// Clock-duration Newey-West Bartlett HAC sandwich for [`crate::Calibration`]'s
/// Mincer-Zarnowitz Wald test (`(1, pred - origin)` with null residual
/// `y - pred`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClockHac {
    bandwidth: f64,
    bandwidth_ns: Option<i128>,
    cap_bytes: u64,
    origin: Option<f64>,
    bread: [f64; 4],
    meat: [f64; 4],
    ring: VecDeque<ClockHacEntry>,
    place: Place,
    over: Option<u64>,
    warned: bool,
    #[serde(skip)]
    warning: Option<u64>,
}

impl ClockHac {
    /// Fresh clock-duration Bartlett HAC sandwich with bandwidth `bandwidth`
    /// (`2 * embargo`), optional exact nanosecond bandwidth `bandwidth_ns`,
    /// and per-slot byte cap `cap_bytes`.
    pub fn new(bandwidth: f64, bandwidth_ns: Option<i128>, cap_bytes: u64) -> Self {
        Self {
            bandwidth,
            bandwidth_ns,
            cap_bytes: cap_bytes.max(CLOCK_HAC_FIXED_BYTES),
            origin: None,
            bread: [0.0; 4],
            meat: [0.0; 4],
            ring: VecDeque::new(),
            place: Place::default(),
            over: None,
            warned: false,
            warning: None,
        }
    }

    /// Accounted bytes for a ring holding `n` entries.
    pub fn bytes_for(n: usize) -> u64 {
        CLOCK_HAC_FIXED_BYTES.saturating_add((n as u64).saturating_mul(CLOCK_HAC_ENTRY_BYTES))
    }

    /// Current accounted bytes.
    pub fn state_bytes(&self) -> u64 {
        Self::bytes_for(self.ring.len())
    }

    /// Per-slot configured byte cap.
    pub fn cap_bytes(&self) -> u64 {
        self.cap_bytes
    }

    /// Bandwidth `B = 2 * embargo` in clock units.
    pub fn bandwidth(&self) -> f64 {
        self.bandwidth
    }

    /// Number of active score entries currently held in the ring.
    pub fn ring_len(&self) -> usize {
        self.ring.len()
    }

    /// If a row would have taken this accumulator past its byte cap, returns
    /// `(attempted_bytes, cap_bytes)`.
    pub fn state_over_cap(&self) -> Option<(u64, u64)> {
        self.over.map(|b| (b, self.cap_bytes))
    }

    /// Drain the once-per-slot half-cap warning `(bytes, cap_bytes)`.
    pub fn take_state_warning(&mut self) -> Option<(u64, u64)> {
        self.warning.take().map(|b| (b, self.cap_bytes))
    }

    /// Validate deserialized invariants against the spec's parameters.
    pub fn has_shape(&self, bandwidth: f64, bandwidth_ns: Option<i128>, cap_bytes: u64) -> bool {
        let cap = cap_bytes.max(CLOCK_HAC_FIXED_BYTES);
        self.bandwidth.is_finite()
            && self.bandwidth > 0.0
            && self.bandwidth == bandwidth
            && self.bandwidth_ns == bandwidth_ns
            && self.cap_bytes == cap
            && self.origin.is_none_or(f64::is_finite)
            && self.bread.iter().chain(&self.meat).all(|v| v.is_finite())
            && self.bread[0] >= 0.0
            && self.bread[3] >= 0.0
            && self.meat[0] >= 0.0
            && self.meat[3] >= 0.0
            && Self::bytes_for(self.ring.len()) <= self.cap_bytes
            && self.over.is_none_or(|b| b > self.cap_bytes)
            && self
                .ring
                .iter()
                .all(|e| stamp_is_finite(e.stamp) && e.u[0].is_finite() && e.u[1].is_finite())
            && self
                .ring
                .iter()
                .zip(self.ring.iter().skip(1))
                .all(|(a, b)| b.stamp.since(a.stamp) >= 0.0)
    }

    #[inline]
    fn expired(&self, now: Stamp, older: Stamp) -> bool {
        now.cmp_span_ns(older, self.bandwidth, self.bandwidth_ns) != Ordering::Less
    }

    /// Drop lagged scores across a capped gap or session boundary.
    pub fn clear_lags(&mut self) {
        self.ring.clear();
    }

    /// Decay existing sums and ring scores without advancing clock stamps.
    pub fn age(&mut self, lam: f64) {
        let lam2 = lam * lam;
        self.bread.iter_mut().for_each(|v| *v *= lam);
        self.meat.iter_mut().for_each(|v| *v *= lam2);
        for e in &mut self.ring {
            e.u[0] *= lam;
            e.u[1] *= lam;
        }
    }

    /// Decay existing sums and ring scores, advance clock place, and evict
    /// expired entries (used on positive-weight rows with missing `pred`/`y`).
    pub fn age_and_step(&mut self, lam: f64, d_clock: f64, stamp: Option<Stamp>) {
        if self.over.is_some() {
            return;
        }
        self.age(lam);
        if let Some(s) = stamp {
            self.place.hand(s);
        }
        let now = self.place.step(d_clock);
        while self
            .ring
            .front()
            .is_some_and(|front| self.expired(now, front.stamp))
        {
            self.ring.pop_front();
        }
    }

    /// Atomically check [`HeldCap`] and fold one finite positive-weight
    /// `(pred, y)` row. Returns `true` if admitted (so the caller should also
    /// update `joint`), or `false` if refused past `cap_bytes` (leaving all
    /// fields untouched except `over`).
    pub fn try_update(
        &mut self,
        pred: f64,
        y: f64,
        lam: f64,
        w: f64,
        d_clock: f64,
        stamp: Option<Stamp>,
    ) -> bool {
        if self.over.is_some() {
            return false;
        }
        let now = stamp.unwrap_or_else(|| self.place.peek(d_clock));
        let drop_count = self
            .ring
            .iter()
            .take_while(|e| self.expired(now, e.stamp))
            .count();
        let projected_len = self.ring.len() - drop_count + 1;
        let with_bytes = Self::bytes_for(projected_len);
        let cap_bytes = self.cap_bytes;
        let mut hc = HeldCap {
            over: &mut self.over,
            warned: &mut self.warned,
            warning: &mut self.warning,
        };
        if !hc.admit(with_bytes, cap_bytes) {
            return false;
        }

        // Check overflow safety before mutating sums.
        let origin = self.origin.unwrap_or(pred);
        let z = [1.0, pred - origin];
        let we = w * (y - pred);
        let u = [we * z[0], we * z[1]];
        let lam2 = lam * lam;
        let mut v_preview = [0.0, 0.0];
        for entry in self.ring.iter().skip(drop_count) {
            let kw = (1.0 - dt_ratio(now, entry.stamp, self.bandwidth, self.bandwidth_ns)).max(0.0);
            v_preview[0] += kw * (entry.u[0] * lam);
            v_preview[1] += kw * (entry.u[1] * lam);
        }
        let fits = (0..2).all(|i| {
            (self.bread[i * 2 + i] * lam + w * z[i] * z[i]).is_finite()
                && (self.meat[i * 2 + i] * lam2 + u[i] * u[i] + 2.0 * u[i] * v_preview[i])
                    .is_finite()
        });
        if !fits {
            self.age_and_step(lam, d_clock, stamp);
            return true;
        }

        // Admitted: commit expiry, clock step, Bartlett meat update, push, warning.
        for _ in 0..drop_count {
            self.ring.pop_front();
        }
        if let Some(s) = stamp {
            self.place.hand(s);
        }
        let stepped = self.place.step(d_clock);
        if self.origin.is_none() {
            self.origin = Some(origin);
        }

        let mut v = [0.0, 0.0];
        for entry in &mut self.ring {
            entry.u[0] *= lam;
            entry.u[1] *= lam;
            let kw =
                (1.0 - dt_ratio(stepped, entry.stamp, self.bandwidth, self.bandwidth_ns)).max(0.0);
            v[0] += kw * entry.u[0];
            v[1] += kw * entry.u[1];
        }
        for i in 0..2 {
            for j in 0..2 {
                self.bread[i * 2 + j] = self.bread[i * 2 + j] * lam + w * z[i] * z[j];
                self.meat[i * 2 + j] =
                    self.meat[i * 2 + j] * lam2 + u[i] * u[j] + u[i] * v[j] + v[i] * u[j];
            }
        }
        self.ring.push_back(ClockHacEntry { stamp: stepped, u });
        let bytes = Self::bytes_for(self.ring.len());
        let mut hc = HeldCap {
            over: &mut self.over,
            warned: &mut self.warned,
            warning: &mut self.warning,
        };
        hc.note(bytes, cap_bytes);
        true
    }

    /// Wald statistic `d' V⁻¹ d` in centred coordinates `z = (1, pred - origin)`
    /// where `V_c = B⁻¹ M B⁻¹` and `d_c = [a + origin * (b - 1), b - 1]`.
    pub fn wald(&self, a: f64, b_minus_1: f64) -> Option<f64> {
        let c = self.origin?;
        let dc = [a + c * b_minus_1, b_minus_1];
        if !(dc[0].is_finite() && dc[1].is_finite()) {
            return None;
        }
        let b_fac = chol2(&self.bread)?;
        // `Left = B^{-1} M` (columns of symmetric M).
        let col0 = chol2_solve(b_fac, [self.meat[0], self.meat[2]]);
        let col1 = chol2_solve(b_fac, [self.meat[1], self.meat[3]]);
        // `V_c = B^{-1} Left^T = B^{-1} M B^{-1}`.
        let v_col0 = chol2_solve(b_fac, [col0[0], col1[0]]);
        let v_col1 = chol2_solve(b_fac, [col0[1], col1[1]]);
        let off = 0.5 * (v_col0[1] + v_col1[0]);
        let vc = [v_col0[0], off, off, v_col1[1]];
        let (l00, l10, l11) = chol2(&vc)?;
        let y0 = dc[0] / l00;
        let y1 = (dc[1] - l10 * y0) / l11;
        let q = y0 * y0 + y1 * y1;
        (q.is_finite() && q >= 0.0).then_some(q)
    }
}

/// One scored quantile indicator entry held inside [`ClockCoverageHac`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ClockCoverageEntry {
    pub stamp: Stamp,
    pub u: f64,
}

/// Clock-duration Newey-West Bartlett HAC long-run variance accumulator for
/// [`crate::Coverage`] (quantile calibration under a clocked `embargo`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClockCoverageHac {
    bandwidth: f64,
    bandwidth_ns: Option<i128>,
    cap_bytes: u64,
    /// Weighted sum of lagged cross-products `Σ_{s < t} (1 - dt / B) u_t u_s`.
    lagged: f64,
    /// Empirical lag-0 sum `Σ u_t²` (positive fallback when null lag-0 + 2*lagged <= 0).
    g: f64,
    ring: VecDeque<ClockCoverageEntry>,
    place: Place,
    over: Option<u64>,
    warned: bool,
    #[serde(skip)]
    warning: Option<u64>,
}

impl ClockCoverageHac {
    pub fn new(bandwidth: f64, bandwidth_ns: Option<i128>, cap_bytes: u64) -> Self {
        Self {
            bandwidth,
            bandwidth_ns,
            cap_bytes: cap_bytes.max(CLOCK_COVERAGE_HAC_FIXED_BYTES),
            lagged: 0.0,
            g: 0.0,
            ring: VecDeque::new(),
            place: Place::default(),
            over: None,
            warned: false,
            warning: None,
        }
    }

    pub fn bytes_for(n: usize) -> u64 {
        CLOCK_COVERAGE_HAC_FIXED_BYTES
            .saturating_add((n as u64).saturating_mul(CLOCK_COVERAGE_HAC_ENTRY_BYTES))
    }

    pub fn state_bytes(&self) -> u64 {
        Self::bytes_for(self.ring.len())
    }

    pub fn cap_bytes(&self) -> u64 {
        self.cap_bytes
    }

    pub fn ring_len(&self) -> usize {
        self.ring.len()
    }

    pub fn state_over_cap(&self) -> Option<(u64, u64)> {
        self.over.map(|b| (b, self.cap_bytes))
    }

    pub fn take_state_warning(&mut self) -> Option<(u64, u64)> {
        self.warning.take().map(|b| (b, self.cap_bytes))
    }

    pub fn lagged(&self) -> f64 {
        self.lagged
    }

    pub fn empirical_lag0(&self) -> f64 {
        self.g
    }

    pub fn has_shape(&self, bandwidth: f64, bandwidth_ns: Option<i128>, cap_bytes: u64) -> bool {
        let cap = cap_bytes.max(CLOCK_COVERAGE_HAC_FIXED_BYTES);
        self.bandwidth.is_finite()
            && self.bandwidth > 0.0
            && self.bandwidth == bandwidth
            && self.bandwidth_ns == bandwidth_ns
            && self.cap_bytes == cap
            && self.lagged.is_finite()
            && self.g.is_finite()
            && self.g >= 0.0
            && Self::bytes_for(self.ring.len()) <= self.cap_bytes
            && self.over.is_none_or(|b| b > self.cap_bytes)
            && self
                .ring
                .iter()
                .all(|e| stamp_is_finite(e.stamp) && e.u.is_finite())
            && self
                .ring
                .iter()
                .zip(self.ring.iter().skip(1))
                .all(|(a, b)| b.stamp.since(a.stamp) >= 0.0)
    }

    #[inline]
    fn expired(&self, now: Stamp, older: Stamp) -> bool {
        now.cmp_span_ns(older, self.bandwidth, self.bandwidth_ns) != Ordering::Less
    }

    pub fn clear_lags(&mut self) {
        self.ring.clear();
    }

    pub fn age(&mut self, lam: f64) {
        let lam2 = lam * lam;
        self.lagged *= lam2;
        self.g *= lam2;
        for e in &mut self.ring {
            e.u *= lam;
        }
    }

    pub fn age_and_step(&mut self, lam: f64, d_clock: f64, stamp: Option<Stamp>) {
        if self.over.is_some() {
            return;
        }
        self.age(lam);
        if let Some(s) = stamp {
            self.place.hand(s);
        }
        let now = self.place.step(d_clock);
        while self
            .ring
            .front()
            .is_some_and(|front| self.expired(now, front.stamp))
        {
            self.ring.pop_front();
        }
    }

    /// Atomically admit and fold one quantile indicator score `u = w * (1{y < pred} - q)`.
    pub fn try_update(&mut self, u: f64, lam: f64, d_clock: f64, stamp: Option<Stamp>) -> bool {
        if self.over.is_some() {
            return false;
        }
        let now = stamp.unwrap_or_else(|| self.place.peek(d_clock));
        let drop_count = self
            .ring
            .iter()
            .take_while(|e| self.expired(now, e.stamp))
            .count();
        let projected_len = self.ring.len() - drop_count + 1;
        let with_bytes = Self::bytes_for(projected_len);
        let cap_bytes = self.cap_bytes;
        let mut hc = HeldCap {
            over: &mut self.over,
            warned: &mut self.warned,
            warning: &mut self.warning,
        };
        if !hc.admit(with_bytes, cap_bytes) {
            return false;
        }
        for _ in 0..drop_count {
            self.ring.pop_front();
        }
        if let Some(s) = stamp {
            self.place.hand(s);
        }
        let stepped = self.place.step(d_clock);
        let lam2 = lam * lam;
        self.lagged *= lam2;
        self.g = self.g * lam2 + u * u;
        let mut cross = 0.0;
        for entry in &mut self.ring {
            entry.u *= lam;
            let kw =
                (1.0 - dt_ratio(stepped, entry.stamp, self.bandwidth, self.bandwidth_ns)).max(0.0);
            cross += kw * entry.u;
        }
        self.lagged += u * cross;
        self.ring
            .push_back(ClockCoverageEntry { stamp: stepped, u });
        let bytes = Self::bytes_for(self.ring.len());
        let mut hc = HeldCap {
            over: &mut self.over,
            warned: &mut self.warned,
            warning: &mut self.warning,
        };
        hc.note(bytes, cap_bytes);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Calibration, Coverage, DEFAULT_STATE_BYTES};

    fn mul2(a: &[f64], b: &[f64]) -> Vec<f64> {
        vec![
            a[0] * b[0] + a[1] * b[2],
            a[0] * b[1] + a[1] * b[3],
            a[2] * b[0] + a[3] * b[2],
            a[2] * b[1] + a[3] * b[3],
        ]
    }

    fn lcg(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    }

    /// On a unit-step numeric clock `t = 0, 1, 2, ...`, bandwidth `B = 2 * E`
    /// with `w(dt) = 1 - dt / B` for `0 <= dt < B` equals discrete Newey-West
    /// Bartlett HAC at `maxlags = 2 * E - 1` (`1 - l / (L + 1)`) to rounding!
    #[test]
    fn unit_step_clock_matches_discrete_newey_west_at_2e_minus_1_lags() {
        let embargo = 3usize;
        let bw = (2 * embargo) as f64;
        let lags = 2 * embargo - 1;
        let lam = 0.99;

        let mut clock_cal = Calibration::with_clock_bandwidth(bw, None, DEFAULT_STATE_BYTES);
        let mut disc_cal = Calibration::with_lags(lags);
        let mut clock_cov = Coverage::with_clock_bandwidth(0.8, bw, None, DEFAULT_STATE_BYTES);
        let mut disc_cov = Coverage::new(0.8, lags);

        let mut s = 17u64;
        let mut prev = 0.0;
        for i in 0..250 {
            let p = 2.0 * lcg(&mut s) + 5.0;
            let shock = 0.65 * prev + lcg(&mut s);
            prev = shock;
            let y = 0.15 + 0.9 * p + shock;
            let w = 0.5 + (lcg(&mut s) + 0.5);
            let stamp = Stamp::Raw(i as f64, 0.0);
            clock_cal.update_at(p, y, lam, w, 1.0, Some(stamp));
            disc_cal.update(p, y, lam, w);
            clock_cov.update_at(p, y, lam, w, 1.0, Some(stamp));
            disc_cov.update(p, y, lam, w);
            if i >= 20 && i % 25 == 0 {
                let cw = clock_cal.wald().unwrap();
                let dw = disc_cal.wald().unwrap();
                assert!(
                    (cw - dw).abs() <= 1e-11 * dw.max(1.0),
                    "cal row {i}: clock={cw} vs disc={dw}"
                );
                let qcw = clock_cov.wald().unwrap();
                let qdw = disc_cov.wald().unwrap();
                assert!(
                    (qcw - qdw).abs() <= 1e-11 * qdw.max(1.0),
                    "cov row {i}: clock={qcw} vs disc={qdw}"
                );
            }
        }
        assert!(clock_cal.has_shape_clock(bw, None, DEFAULT_STATE_BYTES));
        assert!(!clock_cal.has_shape(lags));
        assert!(clock_cov.has_shape_clock(0.8, bw, None, DEFAULT_STATE_BYTES));
        assert!(!clock_cov.has_shape(0.8, lags));
    }

    /// Irregular numeric & temporal clocks match the exact double-sum Bartlett
    /// kernel definition with decay, uneven weights, ties (`dt = 0`), and
    /// Rule 9 zero-weight rows.
    #[test]
    fn irregular_and_temporal_clocks_match_exact_bartlett_double_sum() {
        let bw = 6.0;
        let bw_ns = Some(6_000_000_000_i128);
        let mut cal = Calibration::with_clock_bandwidth(bw, bw_ns, DEFAULT_STATE_BYTES);
        let mut s = 73u64;
        let mut t_ns: i128 = 0;
        let half_life = 40.0;
        let mut rows: Vec<(i128, f64, f64, f64)> = Vec::new(); // (t_ns, p, y, omega)
        let mut prev_shock = 0.0;

        for i in 0..180 {
            let dt_ms: i128 = if i == 0 || i % 9 == 0 {
                // Initial row or simultaneous timestamp (dt == 0 -> Bartlett weight 1.0)
                0
            } else {
                200 + ((lcg(&mut s) + 0.5) * 1800.0) as i128
            };
            let p = 3.0 * lcg(&mut s) + 10.0;
            let shock = 0.7 * prev_shock + lcg(&mut s);
            prev_shock = shock;
            let y = -0.25 + 0.85 * p + shock;

            // Rule 9: zero-weight row must leave clock and ring completely unchanged!
            if i % 15 == 7 {
                let before = cal.clone();
                cal.update_at(p, y, 1.0, 0.0, 0.0, Some(Stamp::Ns(t_ns + 500_000_000)));
                assert_eq!(cal, before, "Rule 9 violated on zero-weight row {i}");
                continue;
            }

            t_ns += dt_ms * 1_000_000;
            let dt_s = (dt_ms as f64) / 1000.0;
            let lam = 0.5_f64.powf(dt_s / half_life);
            let w = 0.5 + (lcg(&mut s) + 0.5);
            cal.update_at(p, y, lam, w, dt_s, Some(Stamp::Ns(t_ns)));
            for r in &mut rows {
                r.3 *= lam;
            }
            rows.push((t_ns, p, y, w));

            if rows.len() >= 25 && i % 30 == 0 {
                // Exact double-sum sandwich in raw (1, pred) coordinates via crate::oracle.
                let mut b = [0.0; 4];
                let mut m = [0.0; 4];
                for &(ta, pa, ya, wa) in &rows {
                    let za = [1.0, pa];
                    let ua = [wa * (ya - pa) * za[0], wa * (ya - pa) * za[1]];
                    for r in 0..2 {
                        for c in 0..2 {
                            b[r * 2 + c] += wa * za[r] * za[c];
                        }
                    }
                    for &(tb, pb, yb, wb) in &rows {
                        let diff_ns = (ta - tb).abs();
                        if diff_ns < 6_000_000_000 {
                            let kw = 1.0 - (diff_ns as f64) / 6_000_000_000.0;
                            let ub = [wb * (yb - pb), wb * (yb - pb) * pb];
                            for r in 0..2 {
                                for c in 0..2 {
                                    m[r * 2 + c] += kw * ua[r] * ub[c];
                                }
                            }
                        }
                    }
                }
                let bi = crate::oracle::inverse(&b);
                let left = mul2(&bi, &m);
                let v = mul2(&left, &bi);
                let d = [cal.intercept().unwrap(), cal.slope().unwrap() - 1.0];
                let want = crate::oracle::quad_form(&v, &d);
                let got = cal.wald().unwrap();
                assert!(
                    (got - want).abs() <= 1e-9 * want.max(1.0),
                    "row {i}: got={got} vs want={want}"
                );
            }
        }
    }

    /// Atomic `HeldCap`: warns once past half the cap, refuses past the cap
    /// leaving `joint`, `bread`, `meat`, `ring`, and `place` untouched, and
    /// refuses all subsequent rows.
    #[test]
    fn held_cap_warns_at_half_and_refuses_atomically() {
        // Cap fits exactly 5 entries: 160 + 5 * 48 = 400 bytes.
        let cap = ClockHac::bytes_for(5);
        let mut cal = Calibration::with_clock_bandwidth(100.0, None, cap);
        let mut warned_at = None;
        for i in 0..5 {
            cal.update_at(
                i as f64,
                i as f64 + 0.1,
                1.0,
                1.0,
                1.0,
                Some(Stamp::Raw(i as f64, 0.0)),
            );
            assert!(cal.state_over_cap().is_none());
            if let Some((b, c)) = cal.take_state_warning() {
                assert!(warned_at.is_none(), "warning must fire at most once");
                assert_eq!(c, cap);
                assert!(2 * b > c);
                warned_at = Some((i, b));
            }
        }
        assert!(warned_at.is_some(), "half-cap warning must have fired");

        // Snapshot before the 6th row (which would exceed `cap`).
        let before_joint = cal.joint.clone();
        let before_ch = cal.clock_hac.clone().unwrap();
        cal.update_at(5.0, 5.5, 1.0, 1.0, 1.0, Some(Stamp::Raw(5.0, 0.0)));
        assert_eq!(cal.state_over_cap(), Some((ClockHac::bytes_for(6), cap)));
        // Atomicity: joint, bread, meat, ring, place are all untouched!
        assert_eq!(cal.joint, before_joint);
        let after_ch = cal.clock_hac.as_ref().unwrap();
        assert_eq!(after_ch.bread, before_ch.bread);
        assert_eq!(after_ch.meat, before_ch.meat);
        assert_eq!(after_ch.ring, before_ch.ring);
        assert_eq!(after_ch.place, before_ch.place);
        assert!(cal.has_shape_clock(100.0, None, cap));
    }

    /// Existing `Calibration::new()` and `Calibration::with_lags(2)` serialize
    /// identically without any `clock_hac` field, and `with_clock_bandwidth`
    /// round-trips bit-identically through named and compact msgpack.
    #[test]
    fn serde_round_trip_and_no_extra_bytes_when_clock_hac_absent() {
        let mut cal = Calibration::with_clock_bandwidth(6.0, None, DEFAULT_STATE_BYTES);
        for i in 0..10 {
            cal.update_at(
                i as f64 * 0.5,
                i as f64 * 0.5 + 0.2,
                0.98,
                1.0,
                1.0,
                Some(Stamp::Raw(i as f64, 0.0)),
            );
        }
        let named = rmp_serde::to_vec_named(&cal).unwrap();
        let back: Calibration = rmp_serde::from_slice(&named).unwrap();
        assert_eq!(back, cal);

        let compact = rmp_serde::to_vec(&cal).unwrap();
        let back_c: Calibration = rmp_serde::from_slice(&compact).unwrap();
        assert_eq!(back_c, cal);

        let plain = Calibration::with_lags(2);
        let val =
            rmpv::decode::read_value(&mut rmp_serde::to_vec_named(&plain).unwrap().as_slice())
                .unwrap();
        let map = val.as_map().unwrap();
        assert!(
            map.iter().all(|(k, _)| k.as_str() != Some("clock_hac")),
            "clock_hac must be omitted when None"
        );
    }
}
