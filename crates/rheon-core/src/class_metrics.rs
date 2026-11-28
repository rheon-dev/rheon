//! Streaming binary classification metrics: ROC-AUC, PR-AUC (Average
//! Precision), Expected Calibration Error (ECE), Maximum Calibration Error
//! (MCE), Brier score, and exact uniform reliability bins.
//!
//! ROC-AUC and PR-AUC are computed from two merging [`TDigest`] sketches (one
//! for positive-class scores `y == 1` and one for negative-class scores
//! `y == 0`), keeping memory bounded by `compression` and `n_bins` regardless
//! of stream length. Because [`TDigest`] uses no platform `libm` calls, all
//! sketches and derived scores are bit-exact across platforms.
//!
//! Reliability bins partition `[0, 1]` into `n_bins` uniform intervals
//! `[b / n_bins, (b + 1) / n_bins]`, assigning clipped probability `p` to bin
//! `min(floor(p * n_bins), n_bins - 1)`. Zero-weight rows (`w == 0`) are
//! ignored cleanly (counted in neither sketch nor bins), and non-finite or
//! out-of-bound scores/targets are skipped.

use serde::{Deserialize, Serialize};

use crate::clock_span::{DEFAULT_STATE_BYTES, check_max_state_bytes};
use crate::model::INPUT_BOUND;
use crate::tdigest::{MIN_COMPRESSION, TDigest};

/// Default number of uniform reliability bins on `[0, 1]`.
pub const DEFAULT_CLASS_METRICS_BINS: usize = 10;

/// Default t-digest compression for per-class score sketches.
pub const DEFAULT_CLASS_METRICS_COMPRESSION: f64 = 200.0;

/// Running counts and weighted sums for one uniform reliability bin on `[0, 1]`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct ReliabilityBin {
    /// Positive-weight rows falling in this bin.
    pub n: u64,
    /// Total positive weight in this bin.
    pub weight_sum: f64,
    /// Weighted sum of clipped predicted probabilities in this bin.
    pub sum_p: f64,
    /// Weighted sum of binary targets (`0` or `1`) in this bin.
    pub sum_y: f64,
}

/// Summary row for one reliability bin.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReliabilityBinRow {
    pub bin: usize,
    pub bin_lo: f64,
    pub bin_hi: f64,
    pub n: u64,
    pub weight_sum: f64,
    pub mean_pred: Option<f64>,
    pub mean_actual: Option<f64>,
    pub abs_gap: Option<f64>,
}

/// Exact streaming uniform-bin reliability accumulator on `[0, 1]`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReliabilityBins {
    n_bins: usize,
    bins: Vec<ReliabilityBin>,
    total_n: u64,
    total_weight: f64,
    max_state_bytes: u64,
}

/// Check that a non-zero-weight, finite `(pred, y)` pair has a binary `0`/`1` label.
#[inline]
fn validate_binary_label(y: f64) -> Result<(), String> {
    if y != 0.0 && y != 1.0 {
        return Err(format!(
            "ClassMetrics: target must be a 0/1 value, but got {y}. \
             Pass binary 0/1 targets."
        ));
    }
    Ok(())
}

/// Whether `v` is finite and within the library's input bound.
#[inline]
fn is_usable(v: f64) -> bool {
    v.is_finite() && v.abs() <= INPUT_BOUND
}

impl ReliabilityBins {
    /// Construct a fresh uniform reliability-bin accumulator with `n_bins >= 1`
    /// bounded by `max_state_bytes` (default 256 KB, ceiling 64 MiB).
    pub fn new(n_bins: usize, max_state_bytes: Option<u64>) -> Result<Self, String> {
        if n_bins == 0 {
            return Err("ReliabilityBins: n_bins must be at least 1, but it is 0. \
                 Pass a positive number of bins, such as 10."
                .into());
        }
        check_max_state_bytes("ReliabilityBins", max_state_bytes)?;
        let cap = max_state_bytes.unwrap_or(DEFAULT_STATE_BYTES);
        let bytes = Self::max_bytes(n_bins);
        if bytes > cap {
            return Err(format!(
                "ReliabilityBins: n_bins = {n_bins} holds {bytes} bytes, past max_state_bytes \
                 of {cap}. Lower n_bins or raise max_state_bytes."
            ));
        }
        Ok(Self {
            n_bins,
            bins: vec![ReliabilityBin::default(); n_bins],
            total_n: 0,
            total_weight: 0.0,
            max_state_bytes: cap,
        })
    }

    /// Maximum heap + header bytes for `n_bins` reliability bins.
    pub fn max_bytes(n_bins: usize) -> u64 {
        (std::mem::size_of::<Self>() as u64).saturating_add(
            (n_bins as u64).saturating_mul(std::mem::size_of::<ReliabilityBin>() as u64),
        )
    }

    pub fn n_bins(&self) -> usize {
        self.n_bins
    }

    pub fn n(&self) -> u64 {
        self.total_n
    }

    pub fn weight_sum(&self) -> f64 {
        self.total_weight
    }

    pub fn max_state_bytes(&self) -> u64 {
        self.max_state_bytes
    }

    /// Fold one `(pred, y, w)` observation.
    ///
    /// - Zero-weight rows (`w == 0`), negative or non-finite weights, and
    ///   non-finite or out-of-bound `pred`/`y` are skipped cleanly.
    /// - Binary targets outside `{0, 1}` on a positive-weight usable row return
    ///   an error.
    pub fn update(&mut self, pred: f64, y: f64, w: f64) -> Result<(), String> {
        if !(is_usable(w) && w > 0.0) {
            return Ok(());
        }
        if !(is_usable(pred) && is_usable(y)) {
            return Ok(());
        }
        validate_binary_label(y)?;
        self.push_checked(pred, y, w);
        Ok(())
    }

    /// Push an already-validated `(pred, y, w)` row.
    #[inline]
    fn push_checked(&mut self, pred: f64, y: f64, w: f64) {
        let pc = pred.clamp(0.0, 1.0);
        let idx = ((pc * self.n_bins as f64).floor() as usize).min(self.n_bins - 1);
        let bin = &mut self.bins[idx];
        bin.n = bin.n.saturating_add(1);
        bin.weight_sum += w;
        bin.sum_p += w * pc;
        bin.sum_y += w * y;
        self.total_n = self.total_n.saturating_add(1);
        self.total_weight += w;
    }

    /// Merge another reliability-bin state with the same `n_bins`.
    pub fn merge(&mut self, other: &Self) -> Result<(), String> {
        if self.n_bins != other.n_bins {
            return Err(format!(
                "ReliabilityBins: cannot merge states with different n_bins ({} vs {}). \
                 Pass two states built with the same n_bins.",
                self.n_bins, other.n_bins
            ));
        }
        for (a, b) in self.bins.iter_mut().zip(&other.bins) {
            a.n = a.n.saturating_add(b.n);
            a.weight_sum += b.weight_sum;
            a.sum_p += b.sum_p;
            a.sum_y += b.sum_y;
        }
        self.total_n = self.total_n.saturating_add(other.total_n);
        self.total_weight += other.total_weight;
        Ok(())
    }

    /// Weighted Expected Calibration Error:
    /// `sum_b (W_b / W_total) * |mean_y_b - mean_p_b|` over non-empty bins.
    pub fn ece(&self) -> Option<f64> {
        if self.total_weight <= 0.0 {
            return None;
        }
        let mut acc = 0.0;
        for b in &self.bins {
            if b.weight_sum > 0.0 {
                let mean_p = b.sum_p / b.weight_sum;
                let mean_y = b.sum_y / b.weight_sum;
                acc += (b.weight_sum / self.total_weight) * (mean_y - mean_p).abs();
            }
        }
        Some(acc)
    }

    /// Maximum Calibration Error: `max_b |mean_y_b - mean_p_b|` over non-empty
    /// bins.
    pub fn mce(&self) -> Option<f64> {
        if self.total_weight <= 0.0 {
            return None;
        }
        let mut worst = 0.0_f64;
        for b in &self.bins {
            if b.weight_sum > 0.0 {
                let gap = (b.sum_y / b.weight_sum - b.sum_p / b.weight_sum).abs();
                worst = worst.max(gap);
            }
        }
        Some(worst)
    }

    /// Materialize one [`ReliabilityBinRow`] for each uniform bin `0..n_bins`.
    pub fn rows(&self) -> Vec<ReliabilityBinRow> {
        let nf = self.n_bins as f64;
        self.bins
            .iter()
            .enumerate()
            .map(|(i, b)| {
                let bin_lo = i as f64 / nf;
                let bin_hi = (i + 1) as f64 / nf;
                if b.weight_sum > 0.0 {
                    let mean_pred = b.sum_p / b.weight_sum;
                    let mean_actual = b.sum_y / b.weight_sum;
                    let abs_gap = (mean_actual - mean_pred).abs();
                    ReliabilityBinRow {
                        bin: i,
                        bin_lo,
                        bin_hi,
                        n: b.n,
                        weight_sum: b.weight_sum,
                        mean_pred: Some(mean_pred),
                        mean_actual: Some(mean_actual),
                        abs_gap: Some(abs_gap),
                    }
                } else {
                    ReliabilityBinRow {
                        bin: i,
                        bin_lo,
                        bin_hi,
                        n: b.n,
                        weight_sum: b.weight_sum,
                        mean_pred: None,
                        mean_actual: None,
                        abs_gap: None,
                    }
                }
            })
            .collect()
    }
}

/// Summary metrics extracted from a [`ClassMetrics`] accumulator.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClassMetricsSummary {
    pub n: u64,
    pub n_pos: u64,
    pub n_neg: u64,
    pub weight_sum: f64,
    pub pos_rate: Option<f64>,
    pub roc_auc: Option<f64>,
    pub pr_auc: Option<f64>,
    pub ece: Option<f64>,
    pub mce: Option<f64>,
    pub brier: Option<f64>,
}

/// Streaming binary classification metrics accumulator holding two per-class
/// [`TDigest`] sketches (`pos` for `y == 1`, `neg` for `y == 0`), exact uniform
/// reliability bins, and exact class-count / weighted Brier sums.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClassMetrics {
    pos: TDigest,
    neg: TDigest,
    reliability: ReliabilityBins,
    n_pos: u64,
    n_neg: u64,
    weight_pos: f64,
    weight_neg: f64,
    brier_sum: f64,
    max_state_bytes: u64,
}

/// Coalesce sorted `(score, weight)` points with identical scores into unique
/// ascending `(score, total_weight)` point masses, normalizing `-0.0` to `0.0`.
fn coalesced_ascending(digest: &TDigest) -> Vec<(f64, f64)> {
    let mut pts: Vec<(f64, f64)> = match digest.exact_points() {
        Some(raw) => {
            let mut v: Vec<(f64, f64)> = raw
                .iter()
                .map(|&(s, w)| (if s == 0.0 { 0.0 } else { s }, w))
                .collect();
            v.sort_by(|a, b| a.0.total_cmp(&b.0));
            v
        }
        None => digest
            .settled()
            .into_iter()
            .map(|(s, w)| (if s == 0.0 { 0.0 } else { s }, w))
            .collect(),
    };
    if pts.is_empty() {
        return pts;
    }
    let mut write = 0usize;
    for read in 1..pts.len() {
        let (s, w) = pts[read];
        if s == pts[write].0 {
            pts[write].1 += w;
        } else {
            write += 1;
            pts[write] = (s, w);
        }
    }
    pts.truncate(write + 1);
    pts
}

impl ClassMetrics {
    /// Create a fresh streaming classification metrics accumulator.
    ///
    /// Validates `n_bins >= 1`, `compression >= MIN_COMPRESSION`, and
    /// `max_state_bytes` (default 256 KB, ceiling 64 MiB) against the worst-case
    /// footprint of both t-digests plus the bin table.
    pub fn new(
        n_bins: usize,
        compression: f64,
        max_state_bytes: Option<u64>,
    ) -> Result<Self, String> {
        if n_bins == 0 {
            return Err("ClassMetrics: n_bins must be at least 1, but it is 0. \
                 Pass a positive number of bins, such as 10."
                .into());
        }
        if !(compression >= MIN_COMPRESSION && compression.is_finite()) {
            return Err(format!(
                "ClassMetrics: compression must be at least {MIN_COMPRESSION} and finite, \
                 but it is {compression}. Pass 200 for tight ROC-AUC and PR-AUC accuracy."
            ));
        }
        check_max_state_bytes("ClassMetrics", max_state_bytes)?;
        let cap = max_state_bytes.unwrap_or(DEFAULT_STATE_BYTES);
        let bytes = Self::max_bytes(compression, n_bins);
        if bytes > cap {
            return Err(format!(
                "ClassMetrics: compression = {compression} and n_bins = {n_bins} may hold up to \
                 {bytes} bytes per key, past max_state_bytes of {cap}. Lower compression or \
                 n_bins, or raise max_state_bytes."
            ));
        }
        let pos = TDigest::new(compression)?;
        let neg = TDigest::new(compression)?;
        let reliability = ReliabilityBins::new(n_bins, Some(cap))?;
        Ok(Self {
            pos,
            neg,
            reliability,
            n_pos: 0,
            n_neg: 0,
            weight_pos: 0.0,
            weight_neg: 0.0,
            brier_sum: 0.0,
            max_state_bytes: cap,
        })
    }

    /// Maximum bytes one [`ClassMetrics`] accumulator can hold for
    /// `(compression, n_bins)`.
    pub fn max_bytes(compression: f64, n_bins: usize) -> u64 {
        let digest_heap =
            TDigest::max_bytes(compression).saturating_sub(std::mem::size_of::<TDigest>() as u64);
        let bins_heap =
            (n_bins as u64).saturating_mul(std::mem::size_of::<ReliabilityBin>() as u64);
        (std::mem::size_of::<Self>() as u64)
            .saturating_add(2u64.saturating_mul(digest_heap))
            .saturating_add(bins_heap)
    }

    pub fn n_bins(&self) -> usize {
        self.reliability.n_bins()
    }

    pub fn compression(&self) -> f64 {
        self.pos.compression()
    }

    pub fn max_state_bytes(&self) -> u64 {
        self.max_state_bytes
    }

    pub fn n(&self) -> u64 {
        self.n_pos.saturating_add(self.n_neg)
    }

    pub fn n_pos(&self) -> u64 {
        self.n_pos
    }

    pub fn n_neg(&self) -> u64 {
        self.n_neg
    }

    pub fn weight_pos(&self) -> f64 {
        self.weight_pos
    }

    pub fn weight_neg(&self) -> f64 {
        self.weight_neg
    }

    pub fn weight_sum(&self) -> f64 {
        self.weight_pos + self.weight_neg
    }

    /// Fold one `(pred, y, w)` observation.
    ///
    /// - Zero-weight rows (`w == 0`), negative/non-finite weights, and
    ///   non-finite/out-of-bound scores or targets are skipped cleanly.
    /// - Targets outside `{0, 1}` on a positive-weight usable row return an
    ///   error.
    pub fn update(&mut self, pred: f64, y: f64, w: f64) -> Result<(), String> {
        if !(is_usable(w) && w > 0.0) {
            return Ok(());
        }
        if !(is_usable(pred) && is_usable(y)) {
            return Ok(());
        }
        validate_binary_label(y)?;
        self.reliability.push_checked(pred, y, w);
        let diff = pred - y;
        self.brier_sum += w * diff * diff;
        if y == 1.0 {
            self.n_pos = self.n_pos.saturating_add(1);
            self.weight_pos += w;
            self.pos.update(pred, w);
        } else {
            self.n_neg = self.n_neg.saturating_add(1);
            self.weight_neg += w;
            self.neg.update(pred, w);
        }
        Ok(())
    }

    /// Merge another [`ClassMetrics`] accumulator with matching configuration.
    pub fn merge(&mut self, other: &Self) -> Result<(), String> {
        if self.n_bins() != other.n_bins() {
            return Err(format!(
                "ClassMetrics: cannot merge states with different n_bins ({} vs {}). \
                 Pass two states built with the same n_bins.",
                self.n_bins(),
                other.n_bins()
            ));
        }
        if self.compression().to_bits() != other.compression().to_bits() {
            return Err(format!(
                "ClassMetrics: cannot merge states with different compressions ({} vs {}). \
                 Pass two states built with the same compression.",
                self.compression(),
                other.compression()
            ));
        }
        self.pos.merge_from(&other.pos)?;
        self.neg.merge_from(&other.neg)?;
        self.reliability.merge(&other.reliability)?;
        self.n_pos = self.n_pos.saturating_add(other.n_pos);
        self.n_neg = self.n_neg.saturating_add(other.n_neg);
        self.weight_pos += other.weight_pos;
        self.weight_neg += other.weight_neg;
        self.brier_sum += other.brier_sum;
        Ok(())
    }

    /// Exact ROC-AUC over the positive and negative centroid distributions:
    /// `P(S_pos > S_neg) + 0.5 * P(S_pos == S_neg)`.
    /// Returns `None` when either class has zero total weight.
    pub fn roc_auc(&self) -> Option<f64> {
        if self.weight_pos <= 0.0 || self.weight_neg <= 0.0 {
            return None;
        }
        let pos = coalesced_ascending(&self.pos);
        let neg = coalesced_ascending(&self.neg);
        let w_pos: f64 = pos.iter().map(|c| c.1).sum();
        let w_neg: f64 = neg.iter().map(|c| c.1).sum();
        if w_pos <= 0.0 || w_neg <= 0.0 {
            return None;
        }
        let mut neg_below = 0.0;
        let mut j = 0usize;
        let mut conc = 0.0;
        for &(ps, pw) in &pos {
            while j < neg.len() && neg[j].0 < ps {
                neg_below += neg[j].1;
                j += 1;
            }
            let tied = if j < neg.len() && neg[j].0 == ps {
                neg[j].1
            } else {
                0.0
            };
            conc += pw * (neg_below + 0.5 * tied);
        }
        Some((conc / (w_pos * w_neg)).clamp(0.0, 1.0))
    }

    /// Average Precision (`sum_k (R_k - R_{k-1}) * P_k`) stepping from the
    /// highest score centroid downward, matching
    /// `sklearn.metrics.average_precision_score`.
    /// Returns `None` when either class has zero total weight.
    pub fn pr_auc(&self) -> Option<f64> {
        if self.weight_pos <= 0.0 || self.weight_neg <= 0.0 {
            return None;
        }
        let pos = coalesced_ascending(&self.pos);
        let neg = coalesced_ascending(&self.neg);
        let w_pos: f64 = pos.iter().map(|c| c.1).sum();
        let w_neg: f64 = neg.iter().map(|c| c.1).sum();
        if w_pos <= 0.0 || w_neg <= 0.0 {
            return None;
        }
        let (mut pi, mut ni) = (pos.len(), neg.len());
        let (mut cum_tp, mut cum_fp, mut ap_num) = (0.0, 0.0, 0.0);
        while pi > 0 || ni > 0 {
            let s = match (pi > 0, ni > 0) {
                (true, true) => {
                    if pos[pi - 1].0 >= neg[ni - 1].0 {
                        pos[pi - 1].0
                    } else {
                        neg[ni - 1].0
                    }
                }
                (true, false) => pos[pi - 1].0,
                (false, true) => neg[ni - 1].0,
                (false, false) => break,
            };
            let tp = if pi > 0 && pos[pi - 1].0 == s {
                pi -= 1;
                pos[pi].1
            } else {
                0.0
            };
            let fp = if ni > 0 && neg[ni - 1].0 == s {
                ni -= 1;
                neg[ni].1
            } else {
                0.0
            };
            cum_tp += tp;
            cum_fp += fp;
            if tp > 0.0 {
                let denom = cum_tp + cum_fp;
                if denom > 0.0 {
                    ap_num += tp * (cum_tp / denom);
                }
            }
        }
        Some((ap_num / w_pos).clamp(0.0, 1.0))
    }

    pub fn ece(&self) -> Option<f64> {
        self.reliability.ece()
    }

    pub fn mce(&self) -> Option<f64> {
        self.reliability.mce()
    }

    pub fn brier(&self) -> Option<f64> {
        let w = self.weight_sum();
        (w > 0.0).then(|| self.brier_sum / w)
    }

    pub fn pos_rate(&self) -> Option<f64> {
        let w = self.weight_sum();
        (w > 0.0).then(|| self.weight_pos / w)
    }

    /// Compute all classification summary metrics in one call.
    pub fn summary(&self) -> ClassMetricsSummary {
        ClassMetricsSummary {
            n: self.n(),
            n_pos: self.n_pos,
            n_neg: self.n_neg,
            weight_sum: self.weight_sum(),
            pos_rate: self.pos_rate(),
            roc_auc: self.roc_auc(),
            pr_auc: self.pr_auc(),
            ece: self.ece(),
            mce: self.mce(),
            brier: self.brier(),
        }
    }

    /// Return the exact per-bin reliability rows.
    pub fn reliability_bins(&self) -> Vec<ReliabilityBinRow> {
        self.reliability.rows()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*state >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Brute-force exact weighted ROC-AUC and Average Precision (matching
    /// scikit-learn's definitions directly).
    fn exact_roc_and_ap(rows: &[(f64, f64, f64)]) -> (f64, f64) {
        let pos: Vec<(f64, f64)> = rows
            .iter()
            .filter(|&&(_, y, w)| y == 1.0 && w > 0.0)
            .map(|&(p, _, w)| (p, w))
            .collect();
        let neg: Vec<(f64, f64)> = rows
            .iter()
            .filter(|&&(_, y, w)| y == 0.0 && w > 0.0)
            .map(|&(p, _, w)| (p, w))
            .collect();
        let wp: f64 = pos.iter().map(|c| c.1).sum();
        let wn: f64 = neg.iter().map(|c| c.1).sum();
        let mut conc = 0.0;
        for &(ps, pw) in &pos {
            for &(ns, nw) in &neg {
                if ps > ns {
                    conc += pw * nw;
                } else if ps == ns {
                    conc += 0.5 * pw * nw;
                }
            }
        }
        let roc = conc / (wp * wn);

        let mut all: Vec<(f64, f64, f64)> =
            rows.iter().copied().filter(|&(_, _, w)| w > 0.0).collect();
        all.sort_by(|a, b| b.0.total_cmp(&a.0));
        let (mut cum_tp, mut cum_fp, mut ap) = (0.0, 0.0, 0.0);
        let mut i = 0;
        while i < all.len() {
            let s = all[i].0;
            let (mut tp, mut fp) = (0.0, 0.0);
            while i < all.len() && all[i].0 == s {
                if all[i].1 == 1.0 {
                    tp += all[i].2;
                } else {
                    fp += all[i].2;
                }
                i += 1;
            }
            cum_tp += tp;
            cum_fp += fp;
            if tp > 0.0 {
                ap += (tp / wp) * (cum_tp / (cum_tp + cum_fp));
            }
        }
        (roc, ap)
    }

    #[test]
    fn exact_parity_with_ties_weights_and_edge_cases() {
        let rows = vec![
            (0.9, 1.0, 2.0),
            (0.8, 1.0, 1.0),
            (0.8, 0.0, 3.0),
            (0.5, 1.0, 1.5),
            (0.5, 0.0, 0.5),
            (0.2, 0.0, 4.0),
            (0.0, 0.0, 1.0),
            (1.0, 1.0, 1.0),
            // Zero-weight and non-finite rows must be ignored cleanly (Rule 9).
            (0.99, 0.0, 0.0),
            (0.01, 1.0, 0.0),
            (f64::NAN, 1.0, 1.0),
            (0.5, f64::INFINITY, 1.0),
        ];
        let mut cm = ClassMetrics::new(5, 200.0, None).unwrap();
        for &(p, y, w) in &rows {
            cm.update(p, y, w).unwrap();
        }
        let clean: Vec<(f64, f64, f64)> = rows
            .into_iter()
            .filter(|&(p, y, w)| p.is_finite() && y.is_finite() && w > 0.0)
            .collect();
        let (want_roc, want_ap) = exact_roc_and_ap(&clean);
        let s = cm.summary();
        assert_eq!(s.n, 8);
        assert_eq!(s.n_pos, 4);
        assert_eq!(s.n_neg, 4);
        assert!((s.weight_sum - 14.0).abs() < 1e-12);
        assert!((s.roc_auc.unwrap() - want_roc).abs() < 1e-15);
        assert!((s.pr_auc.unwrap() - want_ap).abs() < 1e-15);

        // Verify bin 4 receives p == 1.0 and bin 0 receives p == 0.0.
        let bins = cm.reliability_bins();
        assert_eq!(bins.len(), 5);
        assert_eq!(bins[0].n, 1);
        assert_eq!(bins[4].n, 4); // 0.8, 0.8, 0.9, 1.0 -> floor(0.8*5)=4
        let ece_manual: f64 = bins
            .iter()
            .filter(|b| b.weight_sum > 0.0)
            .map(|b| (b.weight_sum / s.weight_sum) * b.abs_gap.unwrap())
            .sum();
        assert!((s.ece.unwrap() - ece_manual).abs() < 1e-15);
    }

    #[test]
    fn single_class_all_tied_and_perfect_separation() {
        // Single class -> roc_auc and pr_auc are None; ECE/MCE/Brier remain defined.
        let mut only_pos = ClassMetrics::new(4, 100.0, None).unwrap();
        for p in [0.6, 0.8, 0.9] {
            only_pos.update(p, 1.0, 1.0).unwrap();
        }
        let s = only_pos.summary();
        assert_eq!(s.roc_auc, None);
        assert_eq!(s.pr_auc, None);
        assert_eq!(s.pos_rate, Some(1.0));
        assert!(s.ece.is_some() && s.mce.is_some() && s.brier.is_some());

        // All tied scores -> ROC-AUC = 0.5, PR-AUC = pos_rate.
        let mut tied = ClassMetrics::new(10, 100.0, None).unwrap();
        for i in 0..20 {
            tied.update(0.4, if i < 5 { 1.0 } else { 0.0 }, 1.0)
                .unwrap();
        }
        assert!((tied.roc_auc().unwrap() - 0.5).abs() < 1e-15);
        assert!((tied.pr_auc().unwrap() - 0.25).abs() < 1e-15);

        // Perfect separation -> ROC-AUC = 1.0, PR-AUC = 1.0.
        let mut perf = ClassMetrics::new(10, 100.0, None).unwrap();
        for i in 0..10 {
            perf.update(0.1 + 0.01 * i as f64, 0.0, 1.0).unwrap();
            perf.update(0.8 + 0.01 * i as f64, 1.0, 1.0).unwrap();
        }
        assert_eq!(perf.roc_auc(), Some(1.0));
        assert_eq!(perf.pr_auc(), Some(1.0));
    }

    #[test]
    fn non_binary_target_and_cap_validation_fail_cleanly() {
        let mut cm = ClassMetrics::new(10, 200.0, None).unwrap();
        let err = cm.update(0.5, 0.5, 1.0).unwrap_err();
        assert!(err.contains("0/1"), "{err}");
        // Zero-weight row with non-binary target is ignored per Rule 9.
        assert!(cm.update(0.5, 2.5, 0.0).is_ok());
        assert_eq!(cm.n(), 0);

        assert!(ClassMetrics::new(0, 200.0, None).is_err());
        assert!(ClassMetrics::new(10, 5.0, None).is_err());
        assert!(ClassMetrics::new(10, 200.0, Some(0)).is_err());
        assert!(ClassMetrics::new(10, 200.0, Some(1024)).is_err());
    }

    #[test]
    fn continuous_ten_thousand_row_stream_matches_exact_tightly() {
        let mut rng = 42u64;
        let mut cm = ClassMetrics::new(10, 200.0, None).unwrap();
        let mut rows = Vec::with_capacity(10_000);
        for _ in 0..10_000 {
            let z = lcg(&mut rng) + lcg(&mut rng) + lcg(&mut rng) - 1.5;
            let p = 1.0 / (1.0 + (-2.2 * z).exp());
            let y = if lcg(&mut rng) < p { 1.0 } else { 0.0 };
            let w = 0.5 + lcg(&mut rng);
            cm.update(p, y, w).unwrap();
            rows.push((p, y, w));
        }
        // Verify merges actually happened (not merely buffered).
        assert!(!cm.pos.centroids().is_empty());
        assert!(!cm.neg.centroids().is_empty());
        let (want_roc, want_ap) = exact_roc_and_ap(&rows);
        let roc_err = (cm.roc_auc().unwrap() - want_roc).abs();
        let ap_err = (cm.pr_auc().unwrap() - want_ap).abs();
        assert!(roc_err < 1e-3, "roc_err={roc_err}");
        assert!(ap_err < 5e-3, "ap_err={ap_err}");
    }

    #[test]
    fn merge_combines_disjoint_accumulators() {
        let mut rng = 99u64;
        let mut whole = ClassMetrics::new(8, 100.0, None).unwrap();
        let mut left = ClassMetrics::new(8, 100.0, None).unwrap();
        let mut right = ClassMetrics::new(8, 100.0, None).unwrap();
        for i in 0..400 {
            let p = lcg(&mut rng);
            let y = if lcg(&mut rng) < p { 1.0 } else { 0.0 };
            whole.update(p, y, 1.0).unwrap();
            if i < 200 {
                left.update(p, y, 1.0).unwrap();
            } else {
                right.update(p, y, 1.0).unwrap();
            }
        }
        left.merge(&right).unwrap();
        assert_eq!(left.n(), whole.n());
        assert!((left.ece().unwrap() - whole.ece().unwrap()).abs() < 1e-14);
        assert!((left.brier().unwrap() - whole.brier().unwrap()).abs() < 1e-14);
        assert!((left.roc_auc().unwrap() - whole.roc_auc().unwrap()).abs() < 1e-14);
        assert!((left.pr_auc().unwrap() - whole.pr_auc().unwrap()).abs() < 1e-14);
    }
}
