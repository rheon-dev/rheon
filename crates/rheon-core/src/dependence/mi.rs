//! Joint-histogram mutual information and Miller-Madow bias correction for
//! [`crate::Dependence`].
//!
//! Each `(feature j, target t)` pair keeps a ragged joint table of weights
//! `n_ab` over feature bins `0..bx_j` and target bins `0..by_t`. A value `v`
//! maps to `edges.partition_point(|e| *e <= v)` (a value equal to an edge goes
//! to the bin above; non-finite values go to [`NO_BIN`]), matching
//! [`crate::margbins::MarginalBins::bin_of`].
//!
//! At read time only (never in the state), plug-in MI is read in ratio form
//! via the `libm` crate so no platform `libm` is called and decay scales cancel
//! without subtracting large logarithms:
//!
//! ```text
//! MI    = sum_{n_ab > 0} (n_ab / N) * libm::log((n_ab / n_a) * (N / n_b))   (nats)
//! MI_MM = MI - (B_xy - B_x - B_y + 1) / (2 * N)                             (unit weights only)
//! ```
//!
//! Under decay, stored cell weights stay undecayed against a single group
//! `scale` that folds into every cell whenever `scale * lam < RENORM_AT`.
//! Under a window, each cell also keeps a `u32` row count so emptied cells zero
//! cleanly under fractional weights, and large-cancellation evictions
//! (`w_out > RESUM_RATIO * remaining`) resum that pair's table from the ring.

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

use crate::Stamp;
use crate::model::SameBits;

/// Sentinel bin index for a missing or non-finite value.
pub const NO_BIN: u16 = u16::MAX;

/// Maximum bins per column so every valid bin index fits in `u16` below [`NO_BIN`].
pub const MAX_BINS: usize = (u16::MAX as usize) - 1;

/// Scale threshold below which decay folds into stored cell weights (matches `margbins.rs`).
const RENORM_AT: f64 = 1e-50;

/// Eviction ratio past which a pair's windowed table is resummed from the ring
/// to prevent catastrophic cancellation (including `1e100` bound-row recovery).
const RESUM_RATIO: f64 = 1e8;

/// Find the bin of `v` against strictly increasing interior `edges`.
#[inline]
pub(crate) fn bin_of(edges: &[f64], v: f64) -> u16 {
    if !v.is_finite() {
        return NO_BIN;
    }
    edges.partition_point(|e| *e <= v) as u16
}

/// Validate a list of explicit interior edges for `n_cols` columns.
pub(crate) fn check_edges(
    role: &str,
    param: &str,
    n_cols: usize,
    edges: &[Vec<f64>],
) -> Result<(), String> {
    if edges.len() != n_cols {
        return Err(format!(
            "dependence: {param} must hold one list of edges for each of the {n_cols} {role}s, \
             but it has {}. Give one list for each {role}, in order.",
            edges.len()
        ));
    }
    let bad: Vec<String> = edges
        .iter()
        .enumerate()
        .filter(|(_, e)| {
            e.len() + 1 > MAX_BINS
                || e.iter().any(|v| !v.is_finite())
                || e.windows(2).any(|w| w[1] <= w[0])
        })
        .map(|(j, _)| j.to_string())
        .collect();
    if !bad.is_empty() {
        return Err(format!(
            "dependence: each {role}'s {param} must be finite, strictly increasing, and have \
             fewer than {MAX_BINS} edges, but those of {} (counting from 0) are not. Sort each \
             list, keep each edge once, and drop any edge that is not finite.",
            if bad.len() == 1 {
                format!("{role} {}", bad[0])
            } else {
                format!("{role}s {}", bad.join(", "))
            }
        ));
    }
    Ok(())
}

/// Scratch buffers reused across MI reads and updates so the hot path allocates nothing.
#[derive(Debug, Clone, Default)]
pub(crate) struct MiScratch {
    pub(crate) na: Vec<f64>,
    pub(crate) nb: Vec<f64>,
    pub(crate) x_bins: Vec<u16>,
    pub(crate) y_bins: Vec<u16>,
    pub(crate) resum: Vec<bool>,
    pub(crate) live_tw: Vec<f64>,
    pub(crate) live_nu: Vec<u64>,
}

/// Compute plug-in MI (nats) and Miller-Madow MI (when `want_mm`) from one
/// `bx x by` joint table `cells` (row-major: `a * by + b`).
pub(crate) fn mi_from_slice(
    cells: &[f64],
    bx: usize,
    by: usize,
    want_mm: bool,
    na: &mut Vec<f64>,
    nb: &mut Vec<f64>,
) -> (f64, f64) {
    na.clear();
    na.resize(bx, 0.0);
    nb.clear();
    nb.resize(by, 0.0);
    let mut n = 0.0;
    for a in 0..bx {
        let row = &cells[a * by..(a + 1) * by];
        let mut s = 0.0;
        for (b, &w) in row.iter().enumerate() {
            if w > 0.0 {
                s += w;
                nb[b] += w;
            }
        }
        na[a] = s;
        n += s;
    }
    if !n.is_finite() || n <= 0.0 {
        return (f64::NAN, f64::NAN);
    }
    let mut mi_sum = 0.0;
    let mut b_xy: usize = 0;
    for a in 0..bx {
        let wa = na[a];
        if !wa.is_finite() || wa <= 0.0 {
            continue;
        }
        let row = &cells[a * by..(a + 1) * by];
        for (b, &w_ab) in row.iter().enumerate() {
            if w_ab > 0.0 {
                b_xy += 1;
                let wb = nb[b];
                let ratio = (w_ab / wa) * (n / wb);
                let log_r = if ratio.is_finite() && ratio > 0.0 {
                    libm::log(ratio)
                } else {
                    (libm::log(w_ab) - libm::log(wa)) + (libm::log(n) - libm::log(wb))
                };
                mi_sum += w_ab * log_r;
            }
        }
    }
    let mi = (mi_sum / n).max(0.0);
    let mi_mm = if want_mm {
        let b_x = na.iter().filter(|&&v| v > 0.0).count();
        let b_y = nb.iter().filter(|&&v| v > 0.0).count();
        let corr = (b_xy as i64 - b_x as i64 - b_y as i64 + 1) as f64 / (2.0 * n);
        mi - corr
    } else {
        f64::NAN
    };
    (mi, mi_mm)
}

/// Compact ring of binned rows used when MI runs windowed without ranks.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct BinRing {
    pub(crate) len: usize,
    pub(crate) xb: VecDeque<u16>,
    pub(crate) yb: VecDeque<u16>,
    pub(crate) ws: VecDeque<f64>,
    pub(crate) stamps: VecDeque<Stamp>,
}

impl BinRing {
    pub(crate) fn new(k: usize, n_targets: usize, cap_rows: usize, clock: bool) -> Self {
        Self {
            len: 0,
            xb: VecDeque::with_capacity(cap_rows.saturating_mul(k)),
            yb: VecDeque::with_capacity(cap_rows.saturating_mul(n_targets)),
            ws: VecDeque::with_capacity(cap_rows),
            stamps: VecDeque::with_capacity(if clock { cap_rows } else { 0 }),
        }
    }

    pub(crate) fn bytes(&self, k: usize, n_targets: usize, clock: bool) -> u64 {
        Self::bytes_for(self.len, k, n_targets, clock)
    }

    pub(crate) fn bytes_for(len: usize, k: usize, n_targets: usize, clock: bool) -> u64 {
        let per_row = 2 * (k as u64 + n_targets as u64)
            + 8
            + if clock {
                std::mem::size_of::<Stamp>() as u64
            } else {
                0
            };
        (len as u64).saturating_mul(per_row)
    }

    pub(crate) fn push(&mut self, xb: &[u16], yb: &[u16], w: f64, stamp: Option<Stamp>) {
        self.xb.extend(xb.iter().copied());
        self.yb.extend(yb.iter().copied());
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
        out_x: &mut Vec<u16>,
        out_y: &mut Vec<u16>,
    ) -> Option<f64> {
        let w = self.ws.pop_front()?;
        out_x.clear();
        out_x.extend(self.xb.drain(..k));
        out_y.clear();
        out_y.extend(self.yb.drain(..n_targets));
        if !self.stamps.is_empty() {
            self.stamps.pop_front();
        }
        self.len -= 1;
        Some(w)
    }

    /// Number of front rows that have left a clock window of `span` under `closed` at `now`.
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
        tables: Option<&MiTables>,
    ) -> bool {
        if self.ws.len() != self.len
            || self.xb.len() != self.len.saturating_mul(k)
            || self.yb.len() != self.len.saturating_mul(n_targets)
            || self.stamps.len() != if clock { self.len } else { 0 }
        {
            return false;
        }
        if let Some(wr) = window_rows
            && self.len > wr
        {
            return false;
        }
        if !self.ws.iter().all(|w| w.is_finite() && *w > 0.0) {
            return false;
        }
        if let Some(mt) = tables {
            for r in 0..self.len {
                for j in 0..k {
                    let b = self.xb[r * k + j];
                    if b != NO_BIN && (b as usize) >= mt.n_x_bins(j) {
                        return false;
                    }
                }
                for t in 0..n_targets {
                    let b = self.yb[r * n_targets + t];
                    if b != NO_BIN && (b as usize) >= mt.n_y_bins(t) {
                        return false;
                    }
                }
            }
        }
        true
    }
}

/// Joint bin tables across all `(feature j, target t)` pairs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct MiTables {
    pub(crate) x_edges: Vec<Vec<f64>>,
    pub(crate) y_edges: Vec<Vec<f64>>,
    /// Offset into `weights` (and `counts`) for pair `p = j * n_targets + t`; length `k * T + 1`.
    pub(crate) pair_off: Vec<usize>,
    pub(crate) weights: Vec<f64>,
    /// Per-cell row counts (present when windowed, empty otherwise).
    pub(crate) counts: Vec<u32>,
    /// Undecayed scale under decay (`1.0` without decay).
    pub(crate) scale: f64,
    pub(crate) empty: bool,
}

impl MiTables {
    pub(crate) fn new(x_edges: Vec<Vec<f64>>, y_edges: Vec<Vec<f64>>, windowed: bool) -> Self {
        let k = x_edges.len();
        let nt = y_edges.len();
        let mut pair_off = Vec::with_capacity(k * nt + 1);
        pair_off.push(0);
        let mut total = 0usize;
        for xe in &x_edges {
            let bx = xe.len() + 1;
            for ye in &y_edges {
                let by = ye.len() + 1;
                total += bx * by;
                pair_off.push(total);
            }
        }
        let weights = vec![0.0; total];
        let counts = if windowed {
            vec![0u32; total]
        } else {
            Vec::new()
        };
        Self {
            x_edges,
            y_edges,
            pair_off,
            weights,
            counts,
            scale: 1.0,
            empty: true,
        }
    }

    #[inline]
    pub(crate) fn n_features(&self) -> usize {
        self.x_edges.len()
    }

    #[inline]
    pub(crate) fn n_targets(&self) -> usize {
        self.y_edges.len()
    }

    #[inline]
    pub(crate) fn n_x_bins(&self, j: usize) -> usize {
        self.x_edges[j].len() + 1
    }

    #[inline]
    pub(crate) fn n_y_bins(&self, t: usize) -> usize {
        self.y_edges[t].len() + 1
    }

    pub(crate) fn bytes(&self) -> u64 {
        let edge_floats: usize = self.x_edges.iter().map(Vec::len).sum::<usize>()
            + self.y_edges.iter().map(Vec::len).sum::<usize>();
        (self.weights.len() * std::mem::size_of::<f64>()
            + self.counts.len() * std::mem::size_of::<u32>()
            + edge_floats * std::mem::size_of::<f64>()
            + (self.x_edges.len() + self.y_edges.len()) * std::mem::size_of::<Vec<f64>>()
            + self.pair_off.len() * std::mem::size_of::<usize>()) as u64
    }

    pub(crate) fn max_bytes_for(
        k: usize,
        n_targets: usize,
        x_bins: &[usize],
        y_bins: &[usize],
        windowed: bool,
    ) -> u64 {
        let mut total_cells: u64 = 0;
        let mut edge_floats: u64 = 0;
        for &bx in x_bins {
            edge_floats = edge_floats.saturating_add(bx.saturating_sub(1) as u64);
            for &by in y_bins {
                total_cells = total_cells.saturating_add((bx as u64).saturating_mul(by as u64));
            }
        }
        for &by in y_bins {
            edge_floats = edge_floats.saturating_add(by.saturating_sub(1) as u64);
        }
        let cell_bytes = if windowed { 12u64 } else { 8u64 };
        total_cells
            .saturating_mul(cell_bytes)
            .saturating_add(edge_floats.saturating_mul(8))
            .saturating_add(
                ((k + n_targets) as u64).saturating_mul(std::mem::size_of::<Vec<f64>>() as u64),
            )
            .saturating_add(
                (k.saturating_mul(n_targets).saturating_add(1) as u64)
                    .saturating_mul(std::mem::size_of::<usize>() as u64),
            )
    }

    pub(crate) fn has_shape(
        &self,
        k: usize,
        n_targets: usize,
        windowed: bool,
        decays: bool,
    ) -> bool {
        if self.x_edges.len() != k || self.y_edges.len() != n_targets {
            return false;
        }
        if check_edges("feature", "bin_edges", k, &self.x_edges).is_err()
            || check_edges("target", "target_bin_edges", n_targets, &self.y_edges).is_err()
        {
            return false;
        }
        if self.pair_off.len() != k * n_targets + 1 || self.pair_off[0] != 0 {
            return false;
        }
        let mut off = 0usize;
        for j in 0..k {
            let bx = self.n_x_bins(j);
            for t in 0..n_targets {
                let by = self.n_y_bins(t);
                off = match off.checked_add(bx.saturating_mul(by)) {
                    Some(v) => v,
                    None => return false,
                };
                if self.pair_off[j * n_targets + t + 1] != off {
                    return false;
                }
            }
        }
        if self.weights.len() != off {
            return false;
        }
        if !self.weights.iter().all(|w| w.is_finite() && *w >= 0.0) {
            return false;
        }
        if windowed {
            if self.counts.len() != off {
                return false;
            }
            for (&w, &c) in self.weights.iter().zip(&self.counts) {
                if c == 0 && w != 0.0 {
                    return false;
                }
            }
        } else if !self.counts.is_empty() {
            return false;
        }
        if !(self.scale.is_finite() && self.scale > 0.0 && self.scale <= 1.0) {
            return false;
        }
        if !decays && self.scale != 1.0 {
            return false;
        }
        true
    }

    /// Decay the tables by factor `lam` in `O(1)` unless `scale * lam < RENORM_AT`.
    pub(crate) fn decay(&mut self, lam: f64) {
        if self.empty || !lam.is_finite() || lam < 0.0 || lam == 1.0 {
            return;
        }
        let s = self.scale * lam;
        if s >= RENORM_AT {
            self.scale = s;
            return;
        }
        for w in &mut self.weights {
            let v = *w * self.scale * lam;
            *w = if v >= f64::MIN_POSITIVE { v } else { 0.0 };
        }
        self.scale = 1.0;
        self.empty = self.weights.iter().all(|&w| w == 0.0);
    }

    /// Bin a row `(x, y)` into `x_bins` and `y_bins`.
    pub(crate) fn bin_row(
        &self,
        x: &[f64],
        y: &[f64],
        x_bins: &mut Vec<u16>,
        y_bins: &mut Vec<u16>,
    ) {
        x_bins.clear();
        for (j, edges) in self.x_edges.iter().enumerate() {
            x_bins.push(bin_of(edges, x[j]));
        }
        y_bins.clear();
        for (t, edges) in self.y_edges.iter().enumerate() {
            y_bins.push(bin_of(edges, y[t]));
        }
    }

    /// Add one already-binned row at weight `w > 0`.
    pub(crate) fn add_binned(&mut self, xb: &[u16], yb: &[u16], w: f64) {
        if !(w > 0.0 && w.is_finite()) {
            return;
        }
        let u = w / self.scale;
        if !u.is_finite() {
            return;
        }
        let k = self.n_features();
        let nt = self.n_targets();
        let windowed = !self.counts.is_empty();
        for (j, &a) in xb.iter().enumerate().take(k) {
            if a == NO_BIN {
                continue;
            }
            let a = a as usize;
            for (t, &b) in yb.iter().enumerate().take(nt) {
                if b == NO_BIN {
                    continue;
                }
                let by = self.y_edges[t].len() + 1;
                let idx = self.pair_off[j * nt + t] + a * by + b as usize;
                self.weights[idx] += u;
                if windowed {
                    self.counts[idx] = self.counts[idx].saturating_add(1);
                }
                self.empty = false;
            }
        }
    }

    /// Subtract one evicted binned row at weight `w > 0`, marking any pair in
    /// `needs_resum` that experienced large cancellation.
    pub(crate) fn remove_binned(
        &mut self,
        xb: &[u16],
        yb: &[u16],
        w: f64,
        needs_resum: &mut [bool],
    ) {
        if !(w > 0.0 && w.is_finite()) {
            return;
        }
        let k = self.n_features();
        let nt = self.n_targets();
        let windowed = !self.counts.is_empty();
        for (j, &a) in xb.iter().enumerate().take(k) {
            if a == NO_BIN {
                continue;
            }
            let a = a as usize;
            for (t, &b) in yb.iter().enumerate().take(nt) {
                if b == NO_BIN {
                    continue;
                }
                let p = j * nt + t;
                let by = self.y_edges[t].len() + 1;
                let idx = self.pair_off[p] + a * by + b as usize;
                if windowed {
                    self.counts[idx] = self.counts[idx].saturating_sub(1);
                    if self.counts[idx] == 0 {
                        self.weights[idx] = 0.0;
                        continue;
                    }
                }
                let rem = self.weights[idx] - w;
                if !rem.is_finite() || rem <= 0.0 || w > RESUM_RATIO * rem {
                    needs_resum[p] = true;
                } else {
                    self.weights[idx] = rem;
                }
            }
        }
    }

    /// Zero a single pair `(j, t)`'s table in preparation for resumming from a ring.
    pub(crate) fn clear_pair(&mut self, j: usize, t: usize) {
        let p = j * self.n_targets() + t;
        let range = self.pair_off[p]..self.pair_off[p + 1];
        self.weights[range.clone()].fill(0.0);
        if !self.counts.is_empty() {
            self.counts[range].fill(0);
        }
    }

    /// Add one row into pair `(j, t)` during a resum pass.
    #[inline]
    pub(crate) fn resum_add_pair(&mut self, j: usize, t: usize, a: u16, b: u16, w: f64) {
        if a == NO_BIN || b == NO_BIN || !w.is_finite() || w <= 0.0 {
            return;
        }
        let p = j * self.n_targets() + t;
        let by = self.y_edges[t].len() + 1;
        let idx = self.pair_off[p] + (a as usize) * by + (b as usize);
        self.weights[idx] += w;
        if !self.counts.is_empty() {
            self.counts[idx] = self.counts[idx].saturating_add(1);
        }
    }

    /// Read `(mi, mi_mm)` for pair `(j, t)` from the current table.
    pub(crate) fn read_pair(
        &self,
        j: usize,
        t: usize,
        want_mm: bool,
        na: &mut Vec<f64>,
        nb: &mut Vec<f64>,
    ) -> (f64, f64) {
        let p = j * self.n_targets() + t;
        let range = self.pair_off[p]..self.pair_off[p + 1];
        mi_from_slice(
            &self.weights[range],
            self.n_x_bins(j),
            self.n_y_bins(t),
            want_mm,
            na,
            nb,
        )
    }

    /// Read `(mi, mi_mm)` for pair `(j, t)` from a surviving slice of a [`BinRing`]
    /// (used only by non-mutating `predict_at` when clock-window rows have expired).
    pub(crate) fn read_pair_from_bin_ring(
        &self,
        ring: &BinRing,
        skip: usize,
        j: usize,
        t: usize,
        want_mm: bool,
        ms: &mut MiScratch,
    ) -> (f64, f64) {
        let bx = self.n_x_bins(j);
        let by = self.n_y_bins(t);
        let mut cells = vec![0.0; bx * by];
        let k = self.n_features();
        let nt = self.n_targets();
        for r in skip..ring.len {
            let a = ring.xb[r * k + j];
            let b = ring.yb[r * nt + t];
            if a != NO_BIN && b != NO_BIN {
                cells[(a as usize) * by + (b as usize)] += ring.ws[r];
            }
        }
        mi_from_slice(&cells, bx, by, want_mm, &mut ms.na, &mut ms.nb)
    }

    /// Read `(mi, mi_mm)` for pair `(j, t)` from a surviving slice of a value ring
    /// (used only by non-mutating `predict_at` when clock-window rows have expired).
    pub(crate) fn read_pair_from_value_ring(
        &self,
        ring: &super::ranks::ValueRing,
        skip: usize,
        j: usize,
        t: usize,
        want_mm: bool,
        ms: &mut MiScratch,
    ) -> (f64, f64) {
        let bx = self.n_x_bins(j);
        let by = self.n_y_bins(t);
        let mut cells = vec![0.0; bx * by];
        let k = self.n_features();
        let nt = self.n_targets();
        let xe = &self.x_edges[j];
        let ye = &self.y_edges[t];
        for r in skip..ring.len {
            let a = bin_of(xe, ring.xs[r * k + j]);
            let b = bin_of(ye, ring.ys[r * nt + t]);
            if a != NO_BIN && b != NO_BIN {
                cells[(a as usize) * by + (b as usize)] += ring.ws[r];
            }
        }
        mi_from_slice(&cells, bx, by, want_mm, &mut ms.na, &mut ms.nb)
    }
}

/// Flat buffer of warm-up rows held until MI bin edges freeze, then replayed
/// into [`MiTables`] and dropped.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct MiHold {
    pub(crate) rows: usize,
    pub(crate) xs: Vec<f64>,
    /// `f64::NAN` marks a missing target.
    pub(crate) ys: Vec<f64>,
    pub(crate) ws: Vec<f64>,
    /// Per-row decay factor when `decays`, empty otherwise.
    pub(crate) lams: Vec<f64>,
    /// Per-row stamp when `window_clock`, empty otherwise.
    pub(crate) stamps: Vec<Stamp>,
    /// Accumulated decay from zero-weight rows since the previous held row.
    pub(crate) pending_lam: f64,
    /// Stamp of the first warm-up row (used by `bin_warm_span`).
    pub(crate) first_stamp: Option<Stamp>,
}

impl PartialEq for MiHold {
    fn eq(&self, other: &Self) -> bool {
        self.rows == other.rows
            && self.xs.same_bits(&other.xs)
            && self.ys.same_bits(&other.ys)
            && self.ws.same_bits(&other.ws)
            && self.lams.same_bits(&other.lams)
            && self.stamps == other.stamps
            && self.pending_lam.same_bits(&other.pending_lam)
            && self.first_stamp == other.first_stamp
    }
}

impl MiHold {
    pub(crate) fn new(
        k: usize,
        n_targets: usize,
        cap_rows: usize,
        decays: bool,
        clock_win: bool,
    ) -> Self {
        Self {
            rows: 0,
            xs: Vec::with_capacity(cap_rows.saturating_mul(k)),
            ys: Vec::with_capacity(cap_rows.saturating_mul(n_targets)),
            ws: Vec::with_capacity(cap_rows),
            lams: Vec::with_capacity(if decays { cap_rows } else { 0 }),
            stamps: Vec::with_capacity(if clock_win { cap_rows } else { 0 }),
            pending_lam: 1.0,
            first_stamp: None,
        }
    }

    pub(crate) fn bytes(&self, k: usize, n_targets: usize, decays: bool, clock_win: bool) -> u64 {
        Self::bytes_for(self.rows, k, n_targets, decays, clock_win)
    }

    pub(crate) fn bytes_for(
        rows: usize,
        k: usize,
        n_targets: usize,
        decays: bool,
        clock_win: bool,
    ) -> u64 {
        let per_row = 8 * (k as u64 + n_targets as u64 + 1)
            + if decays { 8 } else { 0 }
            + if clock_win {
                std::mem::size_of::<Stamp>() as u64
            } else {
                0
            };
        (rows as u64).saturating_mul(per_row)
    }

    pub(crate) fn push(
        &mut self,
        x: &[f64],
        y: &[f64],
        w: f64,
        lam: Option<f64>,
        stamp: Option<Stamp>,
    ) {
        self.xs.extend_from_slice(x);
        self.ys.extend_from_slice(y);
        self.ws.push(w);
        if let Some(l) = lam {
            self.lams.push(self.pending_lam * l);
            self.pending_lam = 1.0;
        }
        if let Some(s) = stamp {
            self.stamps.push(s);
        }
        self.rows += 1;
    }

    pub(crate) fn has_shape(
        &self,
        k: usize,
        n_targets: usize,
        decays: bool,
        clock_win: bool,
    ) -> bool {
        self.ws.len() == self.rows
            && self.xs.len() == self.rows.saturating_mul(k)
            && self.ys.len() == self.rows.saturating_mul(n_targets)
            && self.lams.len() == if decays { self.rows } else { 0 }
            && self.stamps.len() == if clock_win { self.rows } else { 0 }
            && self.ws.iter().all(|w| w.is_finite() && *w > 0.0)
            && self.xs.iter().all(|v| v.is_finite())
            && self.ys.iter().all(|v| v.is_finite() || v.is_nan())
            && self
                .lams
                .iter()
                .all(|l| l.is_finite() && *l >= 0.0 && *l <= 1.0)
            && self.pending_lam.is_finite()
            && self.pending_lam >= 0.0
            && self.pending_lam <= 1.0
    }
}
