//! Bounded Quantization Observer (QO) and Chan parallel variance statistics
//! for [`super::HoeffdingTreeRegressor`].
//!
//! Follows Mastelini and de Carvalho (2021) with a hard per-feature slot cap
//! `max_slots` (a new slot once full merges into the nearest existing slot by
//! `(|j - i|, j)`). Fixes the two bugs in `river`'s `QOSplitter`:
//!
//! 1. At the midpoint threshold between slots `i - 1` and `i`, the partition
//!    scored is `{slots <= i - 1} | {slots >= i}`, matching which side the
//!    threshold routes observations to.
//! 2. The left child receives a copy of `{slots <= i - 1}`'s statistics rather
//!    than an aliased reference to the full-leaf accumulator.

use serde::{Deserialize, Serialize};

/// Packed byte size of one QO slot: `idx: i64` (8) + `x_mean: f64` (8) +
/// `TargetVar { n, mean, s }` (24) = 40 bytes.
pub(super) const QO_SLOT_BYTES: u64 = 40;

/// Packed header bytes per feature observer (slot count `u16`).
pub(super) const QO_HEADER_BYTES: u64 = 2;

/// Weighted running target statistics (`n`, `mean`, `s` = sum of squared
/// deviations; variance `s / (n - 1)` when `n > 1`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub(super) struct TargetVar {
    /// Accumulated positive weight.
    pub n: f64,
    /// Weighted mean of target values.
    pub mean: f64,
    /// Weighted sum of squared deviations about `mean`.
    pub s: f64,
}

impl TargetVar {
    /// Update with one observation `(y, w)` of positive normal weight `w`.
    #[inline]
    pub fn update(&mut self, y: f64, w: f64) {
        let m_old = self.mean;
        let n = self.n + w;
        self.n = n;
        let ratio = w / n;
        let upd = m_old + ratio * (y - m_old);
        self.mean = if upd.is_finite() {
            upd
        } else {
            m_old * (1.0 - ratio) + y * ratio
        };
        let c = w * (y - m_old) * (y - self.mean);
        if c.is_finite() {
            if c > 0.0 {
                let ns = self.s + c;
                self.s = if ns.is_finite() { ns } else { f64::MAX };
            }
        } else if y != m_old {
            self.s = f64::MAX;
        }
    }

    /// Sample variance `s / (n - 1)` for `n > 1`, else `0.0`.
    #[inline]
    pub fn var(&self) -> f64 {
        if self.n > 1.0 {
            let v = self.s / (self.n - 1.0);
            if v.is_finite() && v > 0.0 { v } else { 0.0 }
        } else {
            0.0
        }
    }

    /// Combine two disjoint sets via Chan et al.'s parallel variance formula.
    #[inline]
    pub fn add(self, b: Self) -> Self {
        if b.n <= 0.0 {
            return self;
        }
        if self.n <= 0.0 {
            return b;
        }
        let n = self.n + b.n;
        let d = self.mean - b.mean;
        let d2 = d * d;
        let raw_cross = d2 * self.n * b.n / n;
        let cross = if raw_cross.is_finite() {
            raw_cross
        } else {
            let safe = d2 * ((self.n / n) * b.n);
            if safe.is_finite() { safe } else { f64::MAX }
        };
        let raw_mean = (self.n * self.mean + b.n * b.mean) / n;
        let mean = if raw_mean.is_finite() {
            raw_mean
        } else {
            (self.n / n) * self.mean + (b.n / n) * b.mean
        };
        let raw_s = self.s + b.s + cross;
        let s = if raw_s.is_finite() { raw_s } else { f64::MAX };
        Self { n, mean, s }
    }

    /// Structural validity check for `STATE-3` restore.
    pub fn is_valid(&self) -> bool {
        self.n.is_finite()
            && self.n >= 0.0
            && self.mean.is_finite()
            && self.s.is_finite()
            && self.s >= 0.0
    }
}

/// Variance reduction merit of splitting `pre` into `(left, right)`:
/// `var(pre) - (W_l / W) var(left) - (W_r / W) var(right)`, or `0.0` when
/// either side holds less than `min_samples_split` weight.
#[inline]
pub(super) fn variance_reduction(
    pre: TargetVar,
    left: TargetVar,
    right: TargetVar,
    min_samples_split: f64,
) -> f64 {
    if left.n >= min_samples_split && right.n >= min_samples_split && pre.n > 0.0 {
        let mut vr = pre.var();
        vr -= (left.n / pre.n) * left.var();
        vr -= (right.n / pre.n) * right.var();
        if vr.is_finite() { vr } else { 0.0 }
    } else {
        0.0
    }
}

/// One quantization bucket for a single feature.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub(super) struct QoSlot {
    /// Bucket index `floor(x / radius)`, saturated to `i64`.
    pub idx: i64,
    /// Weighted mean of the feature values assigned to this bucket.
    pub x_mean: f64,
    /// Target statistics of the observations assigned to this bucket (`y_var.n`
    /// is also the weight of `x_mean`).
    pub y_var: TargetVar,
}

impl QoSlot {
    #[inline]
    fn update(&mut self, x: f64, y: f64, w: f64) {
        let n = self.y_var.n + w;
        let ratio = w / n;
        let upd = self.x_mean + ratio * (x - self.x_mean);
        self.x_mean = if upd.is_finite() {
            upd
        } else {
            self.x_mean * (1.0 - ratio) + x * ratio
        };
        self.y_var.update(y, w);
    }
}

/// Best candidate binary split returned by a single feature's [`QoObserver`].
#[derive(Debug, Clone, Copy)]
pub(super) struct QoSplitCandidate {
    pub merit: f64,
    pub threshold: f64,
    pub left: TargetVar,
    pub right: TargetVar,
}

/// Bounded quantization observer for one feature at an active regression leaf.
///
/// Kept in strictly increasing `idx` order so lookups, nearest-slot folding
/// and split-candidate walks are all deterministic, cache-friendly and
/// allocation-free.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub(super) struct QoObserver {
    pub slots: Vec<QoSlot>,
}

/// Map `x` to bucket `floor(x / radius)`, saturating cleanly to `i64` bounds.
#[inline]
fn slot_index(x: f64, radius: f64) -> i64 {
    let q = (x / radius).floor();
    // `i64::MAX as f64` rounds up to 2^63; guard with `<` against it so
    // `q as i64` is exact across the full `i64` range.
    const MAX_F: f64 = 9_223_372_036_854_775_808.0; // 2^63
    const MIN_F: f64 = -9_223_372_036_854_775_808.0; // -2^63
    if !q.is_finite() {
        if q.is_sign_positive() {
            i64::MAX
        } else {
            i64::MIN
        }
    } else if q >= MAX_F {
        i64::MAX
    } else if q <= MIN_F {
        i64::MIN
    } else {
        q as i64
    }
}

impl QoObserver {
    /// Insert one observation `(x, y, w)` of positive normal weight `w`.
    pub fn update(&mut self, x: f64, y: f64, w: f64, radius: f64, max_slots: usize) {
        let idx = slot_index(x, radius);
        let pos = self.slots.partition_point(|s| s.idx < idx);
        if let Some(slot) = self.slots.get_mut(pos)
            && slot.idx == idx
        {
            slot.update(x, y, w);
            return;
        }
        if self.slots.len() >= max_slots {
            // Merge into the nearest existing slot by `(|j - idx|, j)`.
            let target = match (pos.checked_sub(1), (pos < self.slots.len()).then_some(pos)) {
                (Some(lo), Some(hi)) => {
                    let d_lo = self.slots[lo].idx.abs_diff(idx);
                    let d_hi = self.slots[hi].idx.abs_diff(idx);
                    if d_lo <= d_hi { lo } else { hi }
                }
                (Some(lo), None) => lo,
                (None, Some(hi)) => hi,
                (None, None) => return,
            };
            self.slots[target].update(x, y, w);
            return;
        }
        self.slots.insert(
            pos,
            QoSlot {
                idx,
                x_mean: x,
                y_var: TargetVar {
                    n: w,
                    mean: y,
                    s: 0.0,
                },
            },
        );
    }

    /// Evaluate every midpoint threshold between consecutive sorted slots and
    /// return the highest-merit split candidate, or `None` if fewer than 2
    /// slots exist.
    ///
    /// Computes exact prefix and suffix merges over the observer's own slots
    /// so `{slots <= i - 1} | {slots >= i}` is a symmetric partition with no
    /// subtraction and no inherited pre-split weight leaking onto one side.
    pub fn best_split(&self, min_samples_split: f64) -> Option<QoSplitCandidate> {
        let m = self.slots.len();
        if m < 2 {
            return None;
        }
        let mut suffix = vec![TargetVar::default(); m];
        let mut acc = TargetVar::default();
        for i in (0..m).rev() {
            acc = self.slots[i].y_var.add(acc);
            suffix[i] = acc;
        }
        let pre_obs = suffix[0];
        let mut best: Option<QoSplitCandidate> = None;
        let mut left = self.slots[0].y_var;
        let mut prev_x = self.slots[0].x_mean;
        for (&slot, &right) in self.slots[1..].iter().zip(&suffix[1..]) {
            let x = slot.x_mean;
            let raw_t = (prev_x + x) / 2.0;
            let threshold = if raw_t.is_finite() {
                raw_t
            } else {
                prev_x * 0.5 + x * 0.5
            };
            if threshold.is_finite() {
                let merit = variance_reduction(pre_obs, left, right, min_samples_split);
                if best.is_none_or(|b| merit > b.merit) {
                    best = Some(QoSplitCandidate {
                        merit,
                        threshold,
                        left,
                        right,
                    });
                }
            }
            left = left.add(slot.y_var);
            prev_x = x;
        }
        best
    }

    /// Exact packed byte size of this observer (`2 + 40 * len`).
    #[inline]
    pub fn bytes(&self) -> u64 {
        QO_HEADER_BYTES + QO_SLOT_BYTES * self.slots.len() as u64
    }

    /// Structural validity check for `STATE-3` restore.
    pub fn is_valid(&self, max_slots: usize) -> bool {
        if self.slots.len() > max_slots {
            return false;
        }
        let mut prev_idx: Option<i64> = None;
        for s in &self.slots {
            if let Some(p) = prev_idx
                && s.idx <= p
            {
                return false;
            }
            prev_idx = Some(s.idx);
            if !s.x_mean.is_finite() || !s.y_var.is_valid() || s.y_var.n <= 0.0 {
                return false;
            }
        }
        true
    }
}
