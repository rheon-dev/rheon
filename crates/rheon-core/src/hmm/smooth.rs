//! Fixed-lag smoothing for [`super::Hmm`]: the half-size single-ring advance
//! form holding `L` scaled emission-likelihood vectors, one lagged filtered
//! state `alpha_{t-L-1}`, and the `L` clock labels of the buffered rows.
//!
//! On row `t` (once `L` seeded learned rows are held in the ring), using the
//! pre-row transition matrix `Π` and current scaled likelihood vector `f_t`:
//!
//! ```text
//! α_{t-L}  ∝ (α_{t-L-1} · Π) ⊙ f_{t-L}
//! β_t      = 1
//! β_{s-1}  ∝ Π · (f_s ⊙ β_s),   s = t, t-1, …, t-L+1
//! γ_{t-L|t} ∝ α_{t-L} ⊙ β_{t-L}
//! ```
//!
//! At fixed parameters (`params = ""` / `learn = false`) this agrees with full
//! batch forward-backward (`hmmlearn.predict_proba(X[:t+1])[t-L]`) to double
//! round-off while keeping `8 K (L + 2) + 16 L` bytes per group. Under
//! learning it runs the same online pass with the current `Π`. Zero-weight
//! rows leave the ring untouched; `clear_lags` clears the ring and anchors
//! `alpha_old` at the current filtered state.

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

use crate::ClockValue;

/// Persisted state of `hmm`'s fixed-lag smoother.
#[derive(Clone, Serialize, Deserialize)]
pub(super) struct HmmSmoother {
    lag: usize,
    /// Filtered state distribution immediately before `fring[0]`.
    alpha_old: Vec<f64>,
    /// Scaled emission likelihoods `exp(ln f_s - max ln f_s)` of the last
    /// `<= lag` seeded learned rows, oldest first.
    fring: VecDeque<Vec<f64>>,
    /// Clock labels of the rows in `fring`, oldest first.
    labels: VecDeque<Option<ClockValue>>,
    /// Clock label handed for the next stepped row (`OnlineModel::label_next`).
    next_label: Option<ClockValue>,
    /// Monotone fallback counter when no `ClockValue` is handed.
    row_index: u64,
    /// Whether the most recent `step_into` emitted a smoothed row (ephemeral).
    #[serde(skip)]
    last_valid: bool,
    /// The clock label of the row emitted by the most recent `step_into`
    /// (ephemeral).
    #[serde(skip)]
    last_emitted: Option<ClockValue>,
}

impl PartialEq for HmmSmoother {
    fn eq(&self, other: &Self) -> bool {
        self.lag == other.lag
            && self.alpha_old == other.alpha_old
            && self.fring == other.fring
            && self.labels == other.labels
            && self.next_label == other.next_label
            && self.row_index == other.row_index
    }
}

impl std::fmt::Debug for HmmSmoother {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HmmSmoother")
            .field("lag", &self.lag)
            .field("alpha_old", &self.alpha_old)
            .field("fring", &self.fring)
            .field("labels", &self.labels)
            .field("next_label", &self.next_label)
            .field("row_index", &self.row_index)
            .finish()
    }
}

impl HmmSmoother {
    /// Bytes held by one group's smoother for `k` states and lag `lag`:
    /// `8 * k * (lag + 2)` for the ring + lagged vector + working vector,
    /// plus `16 * lag` for the clock labels.
    pub(crate) fn bytes(k: usize, lag: usize) -> u64 {
        let state = 8u64
            .saturating_mul(k as u64)
            .saturating_mul((lag as u64).saturating_add(2));
        let stamps = 16u64.saturating_mul(lag as u64);
        state.saturating_add(stamps)
    }

    pub(super) fn new(k: usize, lag: usize) -> Self {
        Self {
            lag,
            alpha_old: vec![1.0 / k as f64; k],
            fring: VecDeque::with_capacity(lag),
            labels: VecDeque::with_capacity(lag),
            next_label: None,
            row_index: 0,
            last_valid: false,
            last_emitted: None,
        }
    }

    pub(super) fn label_next(&mut self, label: ClockValue) {
        self.next_label = Some(label);
    }

    /// Compute `γ_{t-L | t}` into `out[..k]` without mutating the ring.
    /// Returns `true` when the ring is full (`len == lag`) and the smoothed
    /// posterior is finite; writes `NaN` and returns `false` otherwise.
    pub(super) fn read(&self, pi: &[f64], f_cur: &[f64], out: &mut [f64]) -> bool {
        let k = self.alpha_old.len();
        debug_assert_eq!(out.len(), k);
        if self.fring.len() < self.lag {
            out.fill(f64::NAN);
            return false;
        }
        let mut alpha = vec![0.0; k];
        if !Self::advance_alpha(&self.alpha_old, pi, &self.fring[0], &mut alpha) {
            out.fill(f64::NAN);
            return false;
        }
        let mut beta = vec![1.0; k];
        let mut next = vec![0.0; k];
        if !Self::backward_step(pi, f_cur, &beta, &mut next) {
            out.fill(f64::NAN);
            return false;
        }
        std::mem::swap(&mut beta, &mut next);
        for f_s in self.fring.iter().skip(1).rev() {
            if !Self::backward_step(pi, f_s, &beta, &mut next) {
                out.fill(f64::NAN);
                return false;
            }
            std::mem::swap(&mut beta, &mut next);
        }
        let mut sum = 0.0;
        for i in 0..k {
            let v = alpha[i] * beta[i];
            out[i] = v;
            sum += v;
        }
        if !(sum > 0.0 && sum.is_finite()) {
            out.fill(f64::NAN);
            return false;
        }
        for v in out.iter_mut() {
            *v /= sum;
        }
        true
    }

    /// Fold a learned, seeded row into the ring and record whether this row
    /// emitted a smoothed posterior (and for which clock label).
    pub(super) fn advance(&mut self, pi: &[f64], f_cur: &[f64], emitted_ok: bool) {
        let k = self.alpha_old.len();
        let lbl = self
            .next_label
            .take()
            .unwrap_or(ClockValue::F64(self.row_index as f64));
        self.row_index += 1;
        if self.fring.len() == self.lag {
            let mut next_alpha = vec![0.0; k];
            if Self::advance_alpha(&self.alpha_old, pi, &self.fring[0], &mut next_alpha) {
                self.alpha_old = next_alpha;
            }
            self.fring.pop_front();
            let out_lbl = self.labels.pop_front().flatten();
            self.last_valid = emitted_ok;
            self.last_emitted = if emitted_ok { out_lbl } else { None };
        } else {
            self.last_valid = false;
            self.last_emitted = None;
        }
        self.fring.push_back(f_cur.to_vec());
        self.labels.push_back(Some(lbl));
    }

    /// A zero-weight row: leave the ring untouched, drop any handed label, and
    /// record the peeked label if the row emitted a smoothed reading.
    pub(super) fn zero_weight(&mut self, emitted_ok: bool) {
        self.next_label = None;
        if emitted_ok && self.fring.len() == self.lag {
            self.last_valid = true;
            self.last_emitted = self.labels.front().copied().flatten();
        } else {
            self.last_valid = false;
            self.last_emitted = None;
        }
    }

    /// A failed or unseeded row: leave the ring untouched and emit nothing.
    pub(super) fn discard_row(&mut self) {
        self.next_label = None;
        self.last_valid = false;
        self.last_emitted = None;
    }

    /// Clear the ring on a clock reset or `clear_lags`, anchoring `alpha_old`
    /// at the current filtered state `p`.
    pub(super) fn clear(&mut self, p: &[f64]) {
        self.fring.clear();
        self.labels.clear();
        self.alpha_old.clear();
        self.alpha_old.extend_from_slice(p);
        self.next_label = None;
        self.last_valid = false;
        self.last_emitted = None;
    }

    /// Populate [`crate::Smoothed`] for the bank's `smoothed_for` column.
    pub(super) fn fill_smoothed(&self, out: &mut crate::Smoothed) {
        out.emitted.clear();
        out.label.clear();
        out.held.clear();
        out.coef.clear();
        out.var.clear();
        out.emitted.push(self.last_valid);
        out.label.push(self.last_emitted);
        out.held.push(crate::SMOOTH_STOOD);
    }

    pub(super) fn is_valid(&self, k: usize, expected_lag: usize) -> bool {
        let non_neg = |v: f64| v.is_finite() && v >= 0.0;
        self.lag == expected_lag
            && self.lag >= 1
            && self.alpha_old.len() == k
            && self.alpha_old.iter().copied().all(non_neg)
            && self.alpha_old.iter().sum::<f64>() > 0.0
            && self.fring.len() <= self.lag
            && self.labels.len() == self.fring.len()
            && self
                .fring
                .iter()
                .all(|f| f.len() == k && f.iter().copied().all(non_neg))
    }

    /// `out ∝ (alpha · pi) ⊙ f`.
    fn advance_alpha(alpha: &[f64], pi: &[f64], f: &[f64], out: &mut [f64]) -> bool {
        let k = alpha.len();
        out.fill(0.0);
        for kk in 0..k {
            let ak = alpha[kk];
            let row = &pi[kk * k..(kk + 1) * k];
            for l in 0..k {
                out[l] += ak * row[l];
            }
        }
        let mut sum = 0.0;
        for l in 0..k {
            let v = out[l] * f[l];
            out[l] = v;
            sum += v;
        }
        if !(sum > 0.0 && sum.is_finite()) {
            return false;
        }
        for v in out.iter_mut() {
            *v /= sum;
        }
        true
    }

    /// `out ∝ pi · (f ⊙ beta)`.
    fn backward_step(pi: &[f64], f: &[f64], beta: &[f64], out: &mut [f64]) -> bool {
        let k = beta.len();
        let mut sum = 0.0;
        for i in 0..k {
            let row = &pi[i * k..(i + 1) * k];
            let mut acc = 0.0;
            for j in 0..k {
                acc += row[j] * f[j] * beta[j];
            }
            out[i] = acc;
            sum += acc;
        }
        if !(sum > 0.0 && sum.is_finite()) {
            return false;
        }
        for v in out.iter_mut() {
            *v /= sum;
        }
        true
    }
}
