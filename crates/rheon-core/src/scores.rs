//! Streaming exponentially weighted scores for point forecasts, quantiles,
//! binary probabilities, and prediction intervals.
//!
//! Kept beside [`crate::SlotMetrics`] (which holds the legacy `ic`, `r2`, and
//! `hit_rate` trio untouched) so a spec requesting only the legacy trio pays
//! zero extra state bytes and keeps bit-identical output.
//!
//! Every score here is a weighted exponential mean of a per-row quantity `s_t`
//! in mean form, read **before** the row is folded in (out-of-sample by
//! construction):
//!
//! ```text
//! W' = λ W + w
//! m' = (λ W / W') m + (w / W') s_t    (or s_t when λ W = 0)
//! ```
//!
//! - **MAE**: `s_t = |y_t − pred_t|`.
//! - **Pinball loss** at level `q ∈ (0, 1)`:
//!   `s_t = max(q (y_t − pred_t), (q − 1) (y_t − pred_t))`.
//! - **Binary log loss**:
//!   `s_t = −(y_t ln p' + (1 − y_t) ln(1 − p'))` with
//!   `p' = clip(pred_t, ε, 1 − ε)`, `ε = f64::EPSILON`, using the portable
//!   pure-Rust `libm::log` so state and outputs are bit-identical across OSes
//!   and match scikit-learn's `log_loss`.
//! - **Interval width**: `s_t = u − l` where `l = min(lo, hi)`, `u = max(lo, hi)`.
//! - **Winkler score** at nominal coverage `1 − α`:
//!   `s_t = (u − l) + (2 / α) ((l − y_t)_+ + (y_t − u)_+)`.

use serde::{Deserialize, Serialize};

/// Per-row pinball (check / quantile) loss at level `q ∈ (0, 1)`:
/// `max(q · (y − pred), (q − 1) · (y − pred))`.
#[inline]
pub fn pinball_loss(y: f64, pred: f64, q: f64) -> f64 {
    let r = y - pred;
    if r >= 0.0 { q * r } else { (q - 1.0) * r }
}

/// Per-row binary log loss with probability clipped to `[f64::EPSILON, 1 - f64::EPSILON]`
/// and pure-Rust [`libm::log`] for cross-OS bit-exactness (matching
/// scikit-learn `log_loss` clip).
#[inline]
pub fn binary_log_loss(y: f64, p: f64) -> f64 {
    let pc = p.clamp(f64::EPSILON, 1.0 - f64::EPSILON);
    -(y * libm::log(pc) + (1.0 - y) * libm::log(1.0 - pc))
}

/// Per-row un-crossed interval width and Winkler score at miscoverage `alpha = 1 - coverage`.
///
/// Bounds are un-crossed first (`l = min(lo, hi)`, `u = max(lo, hi)`),
/// matching MAPIE's `regression_mwi_score`.
#[inline]
pub fn interval_width_winkler(y: f64, lo: f64, hi: f64, alpha: f64) -> (f64, f64) {
    let (l, u) = if lo <= hi { (lo, hi) } else { (hi, lo) };
    let width = u - l;
    let under = (l - y).max(0.0);
    let over = (y - u).max(0.0);
    let winkler = width + (2.0 / alpha) * (under + over);
    (width, winkler)
}

/// Mean-form EW scalar update: keeps exact constants, avoids `w_old * m`
/// overflow, and resets cleanly when `w_old == 0.0` (first row or full decay).
#[inline]
fn ew_mean_step(m: &mut f64, s: f64, w_old: f64, w: f64, w_new: f64) {
    if w_old == 0.0 {
        *m = s;
    } else if *m != s {
        *m = (w_old / w_new) * *m + (w / w_new) * s;
    }
}

/// Streaming EW point-forecast and probability loss scores for one output slot:
/// `mae`, zero or more `pinball` levels, and `log_loss`.
///
/// Only allocated and persisted when at least one of those scores is switched
/// on. All enabled scores share one effective weight `w`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SlotScores {
    /// Shared EW weight of scored rows.
    w: f64,
    /// EW mean absolute error (`|y - pred|`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    mae: Option<f64>,
    /// Quantile levels and their EW mean pinball losses, in configured order.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pinball_levels: Vec<f64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pinball: Vec<f64>,
    /// EW mean binary log loss.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    log_loss: Option<f64>,
}

impl SlotScores {
    /// Construct a fresh accumulator for the requested scores.
    pub fn new(track_mae: bool, pinball_levels: &[f64], track_log_loss: bool) -> Self {
        Self {
            w: 0.0,
            mae: track_mae.then_some(0.0),
            pinball_levels: pinball_levels.to_vec(),
            pinball: vec![0.0; pinball_levels.len()],
            log_loss: track_log_loss.then_some(0.0),
        }
    }

    /// Clear accumulated weight and means, keeping the configured scores.
    pub fn reset(&mut self) {
        self.w = 0.0;
        if let Some(m) = &mut self.mae {
            *m = 0.0;
        }
        self.pinball.fill(0.0);
        if let Some(l) = &mut self.log_loss {
            *l = 0.0;
        }
    }

    /// Effective weight of scored rows.
    pub fn weight(&self) -> f64 {
        self.w
    }

    /// EW mean absolute error, `None` before the first scored row or if disabled.
    pub fn mae(&self) -> Option<f64> {
        if self.w > 0.0 { self.mae } else { None }
    }

    /// EW mean pinball loss at index `i`, `None` before the first scored row.
    pub fn pinball(&self, i: usize) -> Option<f64> {
        (self.w > 0.0)
            .then(|| self.pinball.get(i).copied())
            .flatten()
    }

    /// Configured pinball quantile levels.
    pub fn pinball_levels(&self) -> &[f64] {
        &self.pinball_levels
    }

    /// EW mean binary log loss, `None` before the first scored row or if disabled.
    pub fn log_loss(&self) -> Option<f64> {
        if self.w > 0.0 { self.log_loss } else { None }
    }

    /// Verify deserialized shape matches a freshly constructed accumulator
    /// for the same spec.
    pub fn matches_shape(&self, expected: &Self) -> bool {
        self.w.is_finite()
            && self.w >= 0.0
            && self.mae.is_some() == expected.mae.is_some()
            && self.mae.is_none_or(|v| v.is_finite() && v >= 0.0)
            && self.log_loss.is_some() == expected.log_loss.is_some()
            && self.log_loss.is_none_or(|v| v.is_finite() && v >= 0.0)
            && self.pinball_levels == expected.pinball_levels
            && self.pinball.len() == expected.pinball.len()
            && self.pinball.iter().all(|&v| v.is_finite() && v >= 0.0)
    }

    /// Fold one out-of-sample `(pred, y)` pair at decay factor `lam` and row
    /// weight `w`.
    ///
    /// A zero-weight row leaves weight and means untouched (hard rule 9); a
    /// non-finite `pred`/`y` or an overflowing score ages `w` without scoring.
    pub fn update(&mut self, pred: f64, y: f64, lam: f64, w: f64) {
        if !(w > 0.0 && w.is_finite()) {
            return;
        }
        if !pred.is_finite() || !y.is_finite() {
            self.w *= lam;
            return;
        }
        // Pre-compute every enabled score so an overflow leaves state finite.
        let mae_s = self.mae.map(|_| (y - pred).abs());
        if mae_s.is_some_and(|v| !v.is_finite()) {
            self.w *= lam;
            return;
        }
        let ll_s = self.log_loss.map(|_| binary_log_loss(y, pred));
        if ll_s.is_some_and(|v| !v.is_finite()) {
            self.w *= lam;
            return;
        }
        if !self.pinball_levels.is_empty() && !(y - pred).is_finite() {
            self.w *= lam;
            return;
        }
        let w_old = lam * self.w;
        let w_new = w_old + w;
        if !(w_new > 0.0 && w_new.is_finite()) {
            self.w *= lam;
            return;
        }
        if let (Some(m), Some(s)) = (&mut self.mae, mae_s) {
            ew_mean_step(m, s, w_old, w, w_new);
        }
        for (m, &q) in self.pinball.iter_mut().zip(&self.pinball_levels) {
            let s = pinball_loss(y, pred, q);
            ew_mean_step(m, s, w_old, w, w_new);
        }
        if let (Some(m), Some(s)) = (&mut self.log_loss, ll_s) {
            ew_mean_step(m, s, w_old, w, w_new);
        }
        self.w = w_new;
    }
}

/// Streaming EW interval scores (`width` and `winkler`) for one output slot
/// across one or more conformal levels.
///
/// Only allocated and persisted when `emit_metrics` includes `"width"` or
/// `"winkler"`. Each level shares one effective weight per level (so if level
/// `l`'s shown interval was finite, that level updates cleanly).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SlotIntervalScores {
    /// Target miscoverage `α_l = 1 - coverage_l` per level.
    alphas: Vec<f64>,
    /// Shared EW weight of scored rows per level.
    w: Vec<f64>,
    /// EW mean interval width per level (empty if `"width"` not enabled).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    width: Vec<f64>,
    /// EW mean Winkler score per level (empty if `"winkler"` not enabled).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    winkler: Vec<f64>,
}

impl SlotIntervalScores {
    /// Construct a fresh interval-score accumulator for conformal coverage
    /// levels `levels` (each strictly in `(0, 1)`).
    pub fn new(levels: &[f64], track_width: bool, track_winkler: bool) -> Self {
        let n = levels.len();
        Self {
            alphas: levels.iter().map(|&c| 1.0 - c).collect(),
            w: vec![0.0; n],
            width: if track_width {
                vec![0.0; n]
            } else {
                Vec::new()
            },
            winkler: if track_winkler {
                vec![0.0; n]
            } else {
                Vec::new()
            },
        }
    }

    /// Clear accumulated weights and means.
    pub fn reset(&mut self) {
        self.w.fill(0.0);
        self.width.fill(0.0);
        self.winkler.fill(0.0);
    }

    /// Number of conformal levels tracked.
    pub fn n_levels(&self) -> usize {
        self.alphas.len()
    }

    /// EW mean width at conformal level index `li`, `None` before first scored row.
    pub fn width(&self, li: usize) -> Option<f64> {
        let w = *self.w.get(li)?;
        (w > 0.0).then(|| self.width.get(li).copied()).flatten()
    }

    /// EW mean Winkler score at conformal level index `li`, `None` before first scored row.
    pub fn winkler(&self, li: usize) -> Option<f64> {
        let w = *self.w.get(li)?;
        (w > 0.0).then(|| self.winkler.get(li).copied()).flatten()
    }

    /// Verify deserialized shape matches a freshly constructed accumulator.
    pub fn matches_shape(&self, expected: &Self) -> bool {
        let n = expected.alphas.len();
        self.alphas == expected.alphas
            && self.w.len() == n
            && self.w.iter().all(|&v| v.is_finite() && v >= 0.0)
            && self.width.len() == expected.width.len()
            && self.width.iter().all(|&v| v.is_finite() && v >= 0.0)
            && self.winkler.len() == expected.winkler.len()
            && self.winkler.iter().all(|&v| v.is_finite() && v >= 0.0)
    }

    /// Fold one row's target `y` against the emitted interval `(lo, hi)` for
    /// conformal level index `li`.
    ///
    /// Bounds are un-crossed (`min`/`max`) before computing width and Winkler.
    /// Zero-weight rows leave state untouched (hard rule 9); rows without a
    /// finite target or shown interval age `w[li] *= lam`.
    pub fn update_level(&mut self, li: usize, y: f64, lo: f64, hi: f64, lam: f64, w: f64) {
        if !(w > 0.0 && w.is_finite()) {
            return;
        }
        if !(y.is_finite() && lo.is_finite() && hi.is_finite()) {
            self.w[li] *= lam;
            return;
        }
        let (wid, wink) = interval_width_winkler(y, lo, hi, self.alphas[li]);
        if !wid.is_finite() || (!self.winkler.is_empty() && !wink.is_finite()) {
            self.w[li] *= lam;
            return;
        }
        let w_old = lam * self.w[li];
        let w_new = w_old + w;
        if !(w_new > 0.0 && w_new.is_finite()) {
            self.w[li] *= lam;
            return;
        }
        if let Some(m) = self.width.get_mut(li) {
            ew_mean_step(m, wid, w_old, w, w_new);
        }
        if let Some(m) = self.winkler.get_mut(li) {
            ew_mean_step(m, wink, w_old, w, w_new);
        }
        self.w[li] = w_new;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mae_pinball_and_log_loss_match_exact_weighted_means() {
        let mut s = SlotScores::new(true, &[0.1, 0.5, 0.9], true);
        assert_eq!(s.mae(), None);
        assert_eq!(s.pinball(0), None);
        assert_eq!(s.log_loss(), None);

        let preds = [0.2, 0.8, 0.0, 1.0, 0.55];
        let ys = [0.0, 1.0, 0.0, 1.0, 1.0];
        let ws = [1.0, 2.0, 0.5, 1.5, 1.0];
        let lams = [1.0, 0.9, 0.8, 0.95, 0.85];

        // Zero-weight first row leaves state completely untouched (Rule 9).
        s.update(0.3, 1.0, 1.0, 0.0);
        assert_eq!(s.weight(), 0.0);
        assert_eq!(s.mae(), None);

        let mut total_w = 0.0;
        let mut num_mae = 0.0;
        let mut num_p = [0.0; 3];
        let mut num_ll = 0.0;
        let qs = [0.1, 0.5, 0.9];

        for i in 0..preds.len() {
            s.update(preds[i], ys[i], lams[i], ws[i]);
            total_w = lams[i] * total_w + ws[i];
            num_mae = lams[i] * num_mae + ws[i] * (ys[i] - preds[i]).abs();
            for (j, &q) in qs.iter().enumerate() {
                num_p[j] = lams[i] * num_p[j] + ws[i] * pinball_loss(ys[i], preds[i], q);
            }
            num_ll = lams[i] * num_ll + ws[i] * binary_log_loss(ys[i], preds[i]);
        }

        assert!((s.mae().unwrap() - num_mae / total_w).abs() < 1e-14);
        for (j, _) in qs.iter().enumerate() {
            assert!((s.pinball(j).unwrap() - num_p[j] / total_w).abs() < 1e-14);
        }
        assert!((s.log_loss().unwrap() - num_ll / total_w).abs() < 1e-14);
        assert!(s.matches_shape(&SlotScores::new(true, &qs, true)));
    }

    #[test]
    fn binary_log_loss_clips_at_machine_epsilon_and_is_finite() {
        let l0 = binary_log_loss(1.0, 0.0);
        let l1 = binary_log_loss(0.0, 1.0);
        let want = -libm::log(f64::EPSILON);
        assert!(l0.is_finite() && (l0 - want).abs() < 1e-12);
        assert!(l1.is_finite() && (l1 - want).abs() < 1e-12);
        let exact = binary_log_loss(1.0, 1.0);
        assert!(exact.is_finite() && exact < 1e-15);
    }

    #[test]
    fn interval_width_and_winkler_uncross_bounds() {
        // Crossed interval: lo = 5.0 > hi = 3.0 -> l = 3.0, u = 5.0, width = 2.0.
        let (w_in, wink_in) = interval_width_winkler(4.0, 5.0, 3.0, 0.1);
        assert_eq!(w_in, 2.0);
        assert_eq!(wink_in, 2.0);

        // Miss below l = 3.0 by 1.5 -> penalty = (2/0.1) * 1.5 = 30.0.
        let (w_lo, wink_lo) = interval_width_winkler(1.5, 5.0, 3.0, 0.1);
        assert_eq!(w_lo, 2.0);
        assert!((wink_lo - 32.0).abs() < 1e-12);

        // Miss above u = 5.0 by 0.5 -> penalty = (2/0.1) * 0.5 = 10.0.
        let (w_hi, wink_hi) = interval_width_winkler(5.5, 3.0, 5.0, 0.1);
        assert_eq!(w_hi, 2.0);
        assert!((wink_hi - 12.0).abs() < 1e-12);

        let mut is = SlotIntervalScores::new(&[0.8, 0.9], true, true);
        is.update_level(0, 1.5, 5.0, 3.0, 1.0, 1.0);
        is.update_level(1, 5.5, 3.0, 5.0, 1.0, 1.0);
        assert!((is.width(0).unwrap() - 2.0).abs() < 1e-15);
        assert!((is.winkler(0).unwrap() - 17.0).abs() < 1e-12); // alpha = 0.2 -> 2 + 10*1.5
        assert!((is.winkler(1).unwrap() - 12.0).abs() < 1e-12); // alpha = 0.1 -> 2 + 20*0.5
    }

    #[test]
    fn overflow_and_zero_weight_keep_scores_finite() {
        let mut s = SlotScores::new(true, &[0.5], true);
        s.update(1.0, 2.0, 1.0, 1.0);
        // Overflowing residual ages weight without poisoning state.
        s.update(-1e308, 1e308, 0.5, 1.0);
        assert!(s.mae().unwrap().is_finite());
        assert!(s.pinball(0).unwrap().is_finite());
        assert!(s.log_loss().unwrap().is_finite());
        assert!((s.weight() - 0.5).abs() < 1e-15);
    }
}
