//! Windowed rank correlations for [`crate::Dependence`]:
//! - incremental Kendall's tau-b (`O(W)` per row per pair, `O(1)` read);
//! - Spearman's rho and Chatterjee's xi (`O(W log W)` per pair at cadence,
//!   sorting `x` and `y` only once per pair when both are active).

use std::cmp::Ordering;
use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

use crate::Stamp;
use crate::model::SameBits;

/// Ring of window rows `(x[0..k], y[0..T], w, stamp?)` in flat `VecDeque`s so
/// eviction and insertion allocate nothing after warmup and slices stay contiguous.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct ValueRing {
    pub(crate) len: usize,
    pub(crate) xs: VecDeque<f64>,
    /// `f64::NAN` marks an absent target.
    pub(crate) ys: VecDeque<f64>,
    pub(crate) ws: VecDeque<f64>,
    pub(crate) stamps: VecDeque<Stamp>,
}

impl PartialEq for ValueRing {
    fn eq(&self, other: &Self) -> bool {
        self.len == other.len
            && self.xs.len() == other.xs.len()
            && self
                .xs
                .iter()
                .zip(&other.xs)
                .all(|(a, b)| a.to_bits() == b.to_bits())
            && self.ys.len() == other.ys.len()
            && self
                .ys
                .iter()
                .zip(&other.ys)
                .all(|(a, b)| a.to_bits() == b.to_bits())
            && self.ws.len() == other.ws.len()
            && self
                .ws
                .iter()
                .zip(&other.ws)
                .all(|(a, b)| a.to_bits() == b.to_bits())
            && self.stamps == other.stamps
    }
}

impl ValueRing {
    pub(crate) fn new(k: usize, n_targets: usize, cap_rows: usize, clock: bool) -> Self {
        Self {
            len: 0,
            xs: VecDeque::with_capacity(cap_rows.saturating_mul(k)),
            ys: VecDeque::with_capacity(cap_rows.saturating_mul(n_targets)),
            ws: VecDeque::with_capacity(cap_rows),
            stamps: VecDeque::with_capacity(if clock { cap_rows } else { 0 }),
        }
    }

    pub(crate) fn bytes(&self, k: usize, n_targets: usize, clock: bool) -> u64 {
        Self::bytes_for(self.len, k, n_targets, clock)
    }

    pub(crate) fn bytes_for(len: usize, k: usize, n_targets: usize, clock: bool) -> u64 {
        let per_row = 8 * (k as u64 + n_targets as u64 + 1)
            + if clock {
                std::mem::size_of::<Stamp>() as u64
            } else {
                0
            };
        (len as u64).saturating_mul(per_row)
    }

    pub(crate) fn push(&mut self, x: &[f64], y: &[f64], w: f64, stamp: Option<Stamp>) {
        self.xs.extend(x.iter().copied());
        self.ys.extend(y.iter().copied());
        self.ws.push_back(w);
        if let Some(s) = stamp {
            self.stamps.push_back(s);
        }
        self.len += 1;
    }

    /// Pop the oldest row into `out_x` and `out_y`, returning its weight.
    pub(crate) fn pop_front(
        &mut self,
        k: usize,
        n_targets: usize,
        out_x: &mut Vec<f64>,
        out_y: &mut Vec<f64>,
    ) -> Option<f64> {
        let w = self.ws.pop_front()?;
        out_x.clear();
        out_x.extend(self.xs.drain(..k));
        out_y.clear();
        out_y.extend(self.ys.drain(..n_targets));
        if !self.stamps.is_empty() {
            self.stamps.pop_front();
        }
        self.len -= 1;
        Some(w)
    }

    pub(crate) fn expired_count(
        &self,
        now: Stamp,
        span: f64,
        closed: crate::WindowClosed,
    ) -> usize {
        self.stamps
            .iter()
            .take_while(|&&s| closed.has_left(now, s, span))
            .count()
    }

    pub(crate) fn has_shape(
        &self,
        k: usize,
        n_targets: usize,
        clock: bool,
        window_rows: Option<usize>,
    ) -> bool {
        if self.ws.len() != self.len
            || self.xs.len() != self.len.saturating_mul(k)
            || self.yb_len_bad(n_targets)
            || self.stamps.len() != if clock { self.len } else { 0 }
        {
            return false;
        }
        if let Some(wr) = window_rows
            && self.len > wr
        {
            return false;
        }
        self.ws.iter().all(|w| w.is_finite() && *w > 0.0)
            && self.xs.iter().all(|v| v.is_finite())
            && self.ys.iter().all(|v| v.is_finite() || v.is_nan())
    }

    fn yb_len_bad(&self, n_targets: usize) -> bool {
        self.ys.len() != self.len.saturating_mul(n_targets)
    }
}

/// Incremental Kendall tau-b integer counters, stored target-outer (`t * k + j`)
/// so each target's feature sweep is unit-stride contiguous.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct KendallCounters {
    /// `n_c - n_d`
    pub(crate) s: Vec<i64>,
    /// Total pair count `n * (n - 1) / 2`
    pub(crate) n0: Vec<i64>,
    /// Pairs tied in `x`
    pub(crate) n1: Vec<i64>,
    /// Pairs tied in `y`
    pub(crate) n2: Vec<i64>,
}

impl KendallCounters {
    pub(crate) fn new(k: usize, n_targets: usize) -> Self {
        let len = k * n_targets;
        Self {
            s: vec![0; len],
            n0: vec![0; len],
            n1: vec![0; len],
            n2: vec![0; len],
        }
    }

    pub(crate) fn bytes(k: usize, n_targets: usize) -> u64 {
        (32usize).saturating_mul(k.saturating_mul(n_targets)) as u64
    }

    pub(crate) fn has_shape(&self, k: usize, n_targets: usize) -> bool {
        let len = k * n_targets;
        if self.s.len() != len
            || self.n0.len() != len
            || self.n1.len() != len
            || self.n2.len() != len
        {
            return false;
        }
        for i in 0..len {
            let n0 = self.n0[i];
            let n1 = self.n1[i];
            let n2 = self.n2[i];
            let s = self.s[i];
            if n0 < 0 || n1 < 0 || n2 < 0 || n1 > n0 || n2 > n0 || s.saturating_abs() > n0 {
                return false;
            }
        }
        true
    }

    /// Add (`sign == 1`) or remove (`sign == -1`) pairs formed by row `(x, y)`
    /// against every row currently in `ring`.
    pub(crate) fn update_against_ring(
        &mut self,
        ring: &ValueRing,
        x: &[f64],
        y: &[f64],
        sign: i64,
    ) {
        let k = x.len();
        let nt = y.len();
        if ring.len == 0 || k == 0 || nt == 0 {
            return;
        }
        if nt == 1 {
            let y0 = y[0];
            if !y0.is_finite() {
                return;
            }
            let (xs_a, xs_b) = ring.xs.as_slices();
            let (ys_a, ys_b) = ring.ys.as_slices();
            // Process rows using direct indexing when slices don't align at row boundaries,
            // or contiguous row chunks when they do (which holds whenever head is row-aligned,
            // i.e., always for a flat row-pushed/row-popped VecDeque whose capacity is a multiple
            // of k — and we handle the general split cleanly too).
            if xs_a.len().is_multiple_of(k) && ys_a.len() * k == xs_a.len() {
                self.update_t1_slice(xs_a, ys_a, x, y0, sign);
                self.update_t1_slice(xs_b, ys_b, x, y0, sign);
            } else {
                for r in 0..ring.len {
                    let ry = ring.ys[r];
                    if !ry.is_finite() {
                        continue;
                    }
                    let sy = sign_cmp(y0, ry);
                    let ty = i64::from(sy == 0) * sign;
                    let sy_s = sy * sign;
                    let base_x = r * k;
                    for (j, &xj) in x.iter().enumerate().take(k) {
                        let rx = ring.xs[base_x + j];
                        let sx = sign_cmp(xj, rx);
                        self.s[j] += sx * sy_s;
                        self.n0[j] += sign;
                        self.n1[j] += i64::from(sx == 0) * sign;
                        self.n2[j] += ty;
                    }
                }
            }
            return;
        }
        // General T > 1: compare each ring row once per feature and accumulate across finite targets.
        for r in 0..ring.len {
            let base_y = r * nt;
            let base_x = r * k;
            for (t, &yt) in y.iter().enumerate().take(nt) {
                let ryt = ring.ys[base_y + t];
                if !yt.is_finite() || !ryt.is_finite() {
                    continue;
                }
                let sy = sign_cmp(yt, ryt);
                let sy_s = sy * sign;
                let ty = i64::from(sy == 0) * sign;
                let off = t * k;
                for (j, &xj) in x.iter().enumerate().take(k) {
                    let sx = sign_cmp(xj, ring.xs[base_x + j]);
                    let p = off + j;
                    self.s[p] += sx * sy_s;
                    self.n0[p] += sign;
                    self.n1[p] += i64::from(sx == 0) * sign;
                    self.n2[p] += ty;
                }
            }
        }
    }

    #[inline]
    fn update_t1_slice(&mut self, xs: &[f64], ys: &[f64], x: &[f64], y0: f64, sign: i64) {
        let k = x.len();
        for (ry, rx) in ys.iter().zip(xs.chunks_exact(k)) {
            if !ry.is_finite() {
                continue;
            }
            let sy = sign_cmp(y0, *ry);
            let sy_s = sy * sign;
            let ty = i64::from(sy == 0) * sign;
            for j in 0..k {
                let sx = sign_cmp(x[j], rx[j]);
                self.s[j] += sx * sy_s;
                self.n0[j] += sign;
                self.n1[j] += i64::from(sx == 0) * sign;
                self.n2[j] += ty;
            }
        }
    }

    #[inline]
    pub(crate) fn tau_b(&self, j: usize, t: usize, k: usize) -> f64 {
        let p = t * k + j;
        tau_b_of(self.s[p], self.n0[p], self.n1[p], self.n2[p])
    }
}

#[inline]
fn sign_cmp(a: f64, b: f64) -> i64 {
    if a > b {
        1
    } else if a < b {
        -1
    } else {
        0
    }
}

#[inline]
fn tau_b_of(s: i64, n0: i64, n1: i64, n2: i64) -> f64 {
    let d1 = (n0 - n1) as f64;
    let d2 = (n0 - n2) as f64;
    let den = (d1 * d2).sqrt();
    if den > 0.0 {
        (s as f64 / den).clamp(-1.0, 1.0)
    } else {
        f64::NAN
    }
}

/// Brute-force Kendall tau-b over a surviving slice `ring[skip..]` for pair `(j, t)`
/// (used only by non-mutating `predict_at` when clock-window rows have expired).
pub(crate) fn kendall_slice(
    ring: &ValueRing,
    skip: usize,
    j: usize,
    t: usize,
    k: usize,
    nt: usize,
    rs: &mut RankScratch,
) -> f64 {
    gather_pair(ring, skip, j, t, k, nt, rs);
    let n = rs.xs.len();
    if n < 2 {
        return f64::NAN;
    }
    let (mut s, mut n0, mut n1, mut n2) = (0i64, 0i64, 0i64, 0i64);
    for a in 0..n {
        let (xa, ya) = (rs.xs[a], rs.ys[a]);
        for b in (a + 1)..n {
            let sx = sign_cmp(xa, rs.xs[b]);
            let sy = sign_cmp(ya, rs.ys[b]);
            s += sx * sy;
            n0 += 1;
            n1 += i64::from(sx == 0);
            n2 += i64::from(sy == 0);
        }
    }
    tau_b_of(s, n0, n1, n2)
}

/// Reusable scratch buffers for Spearman and Chatterjee xi reads so zero heap
/// allocations happen after warmup.
#[derive(Debug, Clone, Default)]
pub(crate) struct RankScratch {
    pub(crate) xs: Vec<f64>,
    pub(crate) ys: Vec<f64>,
    pub(crate) ord_x: Vec<usize>,
    pub(crate) ord_y: Vec<usize>,
    pub(crate) ax: Vec<f64>,
    pub(crate) ay: Vec<f64>,
    pub(crate) r_of: Vec<i64>,
    pub(crate) l_of: Vec<i64>,
    pub(crate) evict_x: Vec<f64>,
    pub(crate) evict_y: Vec<f64>,
}

/// Gather finite `(x, y)` pairs for `(j, t)` from `ring[skip..]` in oldest-first arrival order.
pub(crate) fn gather_pair(
    ring: &ValueRing,
    skip: usize,
    j: usize,
    t: usize,
    k: usize,
    nt: usize,
    rs: &mut RankScratch,
) {
    rs.xs.clear();
    rs.ys.clear();
    for r in skip..ring.len {
        let xv = ring.xs[r * k + j];
        let yv = ring.ys[r * nt + t];
        if xv.is_finite() && yv.is_finite() {
            rs.xs.push(xv);
            rs.ys.push(yv);
        }
    }
}

/// Compute Spearman's rho and/or Chatterjee's xi from finite samples in `rs.xs, rs.ys`
/// (in arrival order, oldest first), sorting `xs` and `ys` at most once each.
///
/// Ties in `x` are broken by arrival order via stable sort (deterministic, chunk
/// invariant, matching `STUDY-510-DEPENDENCE.md`). Ties in `y` use Chatterjee
/// (2021) eq. 1.2, which reduces algebraically to eq. 1.1 when `y` has no ties.
pub(crate) fn spearman_and_xi_from_sample(
    want_spearman: bool,
    want_xi: bool,
    rs: &mut RankScratch,
) -> (f64, f64) {
    let xs = &rs.xs;
    let ys = &rs.ys;
    let ord_x = &mut rs.ord_x;
    let ord_y = &mut rs.ord_y;
    let ax = &mut rs.ax;
    let ay = &mut rs.ay;
    let r_of = &mut rs.r_of;
    let l_of = &mut rs.l_of;
    let n = xs.len();
    if n < 2 || (!want_spearman && !want_xi) {
        return (f64::NAN, f64::NAN);
    }
    ord_x.clear();
    ord_x.extend(0..n);
    // Stable sort with `partial_cmp` treats `-0.0 == +0.0` as Equal and breaks ties by arrival.
    ord_x.sort_by(|&a, &b| xs[a].partial_cmp(&xs[b]).unwrap_or(Ordering::Equal));

    ord_y.clear();
    ord_y.extend(0..n);
    ord_y.sort_by(|&a, &b| ys[a].partial_cmp(&ys[b]).unwrap_or(Ordering::Equal));

    let spearman = if want_spearman {
        ax.clear();
        ax.resize(n, 0.0);
        ay.clear();
        ay.resize(n, 0.0);
        centered_avg_ranks(xs, ord_x, ax);
        centered_avg_ranks(ys, ord_y, ay);
        let (mut ab, mut aa, mut bb) = (0.0, 0.0, 0.0);
        for i in 0..n {
            let a = ax[i];
            let b = ay[i];
            ab += a * b;
            aa += a * a;
            bb += b * b;
        }
        let den = (aa * bb).sqrt();
        if den > 0.0 {
            (ab / den).clamp(-1.0, 1.0)
        } else {
            f64::NAN
        }
    } else {
        f64::NAN
    };

    let xi = if want_xi {
        r_of.clear();
        r_of.resize(n, 0);
        l_of.clear();
        l_of.resize(n, 0);
        let mut lo = 0usize;
        while lo < n {
            let v = ys[ord_y[lo]];
            let mut hi = lo + 1;
            while hi < n && ys[ord_y[hi]] == v {
                hi += 1;
            }
            let r_val = hi as i64;
            let l_val = (n - lo) as i64;
            for &idx in &ord_y[lo..hi] {
                r_of[idx] = r_val;
                l_of[idx] = l_val;
            }
            lo = hi;
        }
        let mut num: i128 = 0;
        let mut prev = r_of[ord_x[0]];
        for &idx in &ord_x[1..] {
            let cur = r_of[idx];
            num += (cur - prev).abs() as i128;
            prev = cur;
        }
        let mut den: i128 = 0;
        let n128 = n as i128;
        for &l in l_of.iter() {
            let l128 = l as i128;
            den += l128 * (n128 - l128);
        }
        den *= 2;
        if den > 0 {
            1.0 - ((n128 * num) as f64) / (den as f64)
        } else {
            f64::NAN
        }
    } else {
        f64::NAN
    };

    (spearman, xi)
}

/// Centered average ranks `(lo + hi - n) / 2` for each position, exact half-integers.
fn centered_avg_ranks(v: &[f64], ord: &[usize], out: &mut [f64]) {
    let n = v.len();
    let mut lo = 0usize;
    while lo < n {
        let val = v[ord[lo]];
        let mut hi = lo + 1;
        while hi < n && v[ord[hi]] == val {
            hi += 1;
        }
        let rank = ((lo + hi) as f64 - n as f64) * 0.5;
        for &idx in &ord[lo..hi] {
            out[idx] = rank;
        }
        lo = hi;
    }
}

/// Bit-exact `PartialEq` wrapper for held rank vectors that legitimately contain `NaN`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub(crate) struct HeldFloats(pub(crate) Vec<f64>);

impl PartialEq for HeldFloats {
    fn eq(&self, other: &Self) -> bool {
        self.0.same_bits(&other.0)
    }
}
