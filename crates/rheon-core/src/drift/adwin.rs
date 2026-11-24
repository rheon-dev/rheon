//! ADWIN, the adaptive window (Bifet and Gavaldà 2007, ADWIN2), as river's
//! `drift.ADWIN` computes it.
//!
//! The window `W` is an exponential histogram: level `i` holds at most `M`
//! (`max_buckets`) buckets of `2^i` rows each, a bucket being the pair (sum of
//! its values, sum of squared deviations about its own mean). A row enters
//! level 0 as `(x, 0)`; when a level holds `M + 1` buckets its two oldest
//! merge into one of the next level,
//!
//! ```text
//! (t0 + t1,  v0 + v1 + n (t0/n − t1/n)² / 2),   n = 2^i
//! ```
//!
//! and the merge cascades upward. The window's totals are kept as rows come:
//!
//! ```text
//! width += 1;   V += (width − 1) (x − S/(width − 1))² / width;   S += x
//! ```
//!
//! Every `clock` rows (counted since the window last started), once `width >
//! grace_period`, each split of the window at a bucket boundary into an older
//! `W0` (`n0` rows, sum `u0`) and a newer `W1` (`n1`, `u1`) is tested, oldest
//! boundary first:
//!
//! ```text
//! cut when  |u0/n0 − u1/n1| > ε,  n0 ≥ L and n1 ≥ L   (L = min_window_length)
//! ε  = sqrt(2 (1/m) (V/width) d) + (2/3) d (1/m)
//! d  = ln(2 ln(width) / delta)
//! 1/m = 1/(n0 − L + 1) + 1/(n1 − L + 1)
//! ```
//!
//! the paper's Bernstein form (its section 4, `δ' = δ / ln(width)`), with
//! `1/m` shifted by `L` as MOA and river shift it.
//!
//! **The range.** The bound's second term is the paper's `(2/3) R d/m`, `R`
//! the range the values lie in. The paper and river take values in `[0, 1]`
//! and write `R = 1`, which is this detector's default (`AdwinRange::Fixed`
//! of 1), river's to the bit; so it is not scale-free: the same stream in
//! other units is cut sooner or later (river finds a threefold rise in the
//! spread of `|N(0, σ)|` in 15 rows at `σ = 1`, in 655 at `σ = 0.01`, and
//! not in 8,000 at `σ = 0.001`). A fixed `R` of the caller's states the
//! values' range in their own units. `AdwinRange::Window` reads `R` as the
//! range the window holds now, `max − min` over its buckets, each bucket
//! keeping its values' least and greatest: then `ε` scales with the values,
//! as `|u0/n0 − u1/n1|` does, and the flags are the same at any scale (to
//! the bit at a power of two). A cut drops the oldest
//! bucket, of `n` rows, and takes `S` and `V` again from the buckets left,
//! oldest first, by the pairwise merge of the compression above:
//!
//! ```text
//! width −= n;   S = S_a + S_b;   V = V_a + V_b + n_a n_b / (n_a + n_b) (S_a/n_a − S_b/n_b)²
//! ```
//!
//! and the scan starts again on the shorter window; the row is flagged if any
//! cut was made. river takes the bucket's share out of `S` and `V` instead,
//! a subtraction that cancels when the bucket held most of the window's
//! spread (two outliers leaving left `V` far above 0 on a window of zeros,
//! or below it), which blinds every later test; under river's `restart` a
//! row's first cut flags it and empties the window, so the flags are
//! river's either way. Each other expression is evaluated in river's order,
//! so the flags and, until a bucket is dropped, the window's fields are
//! river's to the bit.
//!
//! Two things river does not have. `max_width` caps the window, dropping the
//! oldest bucket while `width > max_width`, so the state is bounded by the
//! spec and not by the stream: at most [`Adwin::levels_bound`] levels. And
//! `restart`: river starts from an empty window on the row after a flag
//! (`true`, the default); `false` keeps the window the cuts left, as the
//! paper's ADWIN2 does, so a change is flagged again as the old rows left in
//! `W1` are found and dropped.
//!
//! The one `libm` call, `ln` in `d`, decides a comparison and enters no state.

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

/// The settings of an [`Adwin`], river's names and defaults but for the two
/// river lacks.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdwinCfg {
    /// The confidence, in `(0, 1)`: river's 0.002.
    pub delta: f64,
    /// Rows between two scans of the window: river's 32.
    pub clock: u64,
    /// Buckets a level holds before its oldest two merge: river's 5.
    pub max_buckets: usize,
    /// The fewest rows a part of a split may hold: river's 5.
    pub min_window_length: u64,
    /// No scan while the window holds this many rows or fewer: river's 10.
    pub grace_period: u64,
    /// The most rows the window holds.
    pub max_width: u64,
    /// Start from an empty window after a flag (river), or keep the cut one
    /// (the paper).
    pub restart: bool,
    /// The range `R` in the bound's second term (module doc, "The range").
    pub range: AdwinRange,
}

/// The range `R` the cut's bound takes (module doc, "The range").
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AdwinRange {
    /// A range in the values' units: river's 1 by default.
    Fixed(f64),
    /// The range the window holds now, `max − min`: scale-free.
    Window,
}

impl Default for AdwinCfg {
    fn default() -> Self {
        Self {
            delta: 0.002,
            clock: 32,
            max_buckets: 5,
            min_window_length: 5,
            grace_period: 10,
            max_width: Adwin::DEFAULT_MAX_WIDTH,
            restart: true,
            range: AdwinRange::Fixed(1.0),
        }
    }
}

/// ADWIN on one stream of values: the module doc has the equations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Adwin {
    delta: f64,
    clock: u64,
    max_buckets: usize,
    min_window_length: u64,
    grace_period: u64,
    max_width: u64,
    restart: bool,
    /// `S`, the window's sum.
    total: f64,
    /// `V`, the window's sum of squared deviations about its mean.
    var: f64,
    /// The rows in the window.
    width: f64,
    /// Rows since the window last started: a scan runs where it is a
    /// multiple of `clock`.
    tick: u64,
    /// Level `i`'s buckets of `2^i` rows, oldest first, each `[sum, sum of
    /// squared deviations]`. Never empty: level 0 is kept with no bucket.
    levels: Vec<VecDeque<[f64; 2]>>,
    /// A fixed range `R`, 1 unless set (and 1 under the window's range);
    /// absent from a file where it is 1, so river's form writes what it did.
    #[serde(default = "unit_range", skip_serializing_if = "is_unit_range")]
    range: f64,
    /// Under [`AdwinRange::Window`], each bucket's `[least, greatest]`,
    /// shaped as `levels`; empty, and absent from the file, otherwise.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    extents: Vec<VecDeque<[f64; 2]>>,
}

fn unit_range() -> f64 {
    1.0
}

fn is_unit_range(r: &f64) -> bool {
    *r == 1.0
}

impl Adwin {
    /// `max_width` unless set: a million rows, 18 levels at `M = 5`.
    pub const DEFAULT_MAX_WIDTH: u64 = 1_000_000;

    pub fn new(cfg: AdwinCfg) -> Self {
        debug_assert!(cfg.delta > 0.0 && cfg.delta < 1.0);
        debug_assert!(cfg.clock >= 1 && cfg.max_buckets >= 1 && cfg.max_width >= 1);
        let (range, extents) = match cfg.range {
            AdwinRange::Fixed(r) => (r, Vec::new()),
            AdwinRange::Window => (1.0, vec![VecDeque::new()]),
        };
        Self {
            delta: cfg.delta,
            clock: cfg.clock,
            max_buckets: cfg.max_buckets,
            min_window_length: cfg.min_window_length,
            grace_period: cfg.grace_period,
            max_width: cfg.max_width,
            restart: cfg.restart,
            total: 0.0,
            var: 0.0,
            width: 0.0,
            tick: 0,
            levels: vec![VecDeque::new()],
            range,
            extents,
        }
    }

    /// Whether `R` is the window's range ([`AdwinRange::Window`]).
    fn windowed_range(&self) -> bool {
        !self.extents.is_empty()
    }

    /// `R` for this scan: the fixed range, or `max − min` over the window.
    fn range_now(&self) -> f64 {
        if !self.windowed_range() {
            return self.range;
        }
        let (lo, hi) = self
            .extents
            .iter()
            .flatten()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &[a, b]| {
                (lo.min(a), hi.max(b))
            });
        if hi >= lo { hi - lo } else { 0.0 }
    }

    /// The most levels a window of `width` rows can hold at `M` buckets a
    /// level: `1 + ⌊log2((width + M − 1) / M)⌋`. Every level below the top
    /// holds at least `M − 1` buckets once the level above it exists (a merge
    /// leaves `M − 1`, and only the oldest bucket, at the top, is ever
    /// dropped), so `L` levels hold at least `M·2^(L−1) − (M − 1)` rows.
    pub fn levels_bound(width: u64, max_buckets: usize) -> u64 {
        let m = max_buckets.max(1) as u128;
        let w = u128::from(width);
        let mut levels = 1u64;
        // The next level needs M·2^levels − (M − 1) rows.
        while m * (1u128 << levels) - (m - 1) <= w {
            levels += 1;
        }
        levels
    }

    /// The bytes an ADWIN's state holds at most under `max_width`, in its
    /// file's terms: each level `M + 1` buckets of two numbers and a count,
    /// and thirteen numbers of settings and totals.
    /// Under [`AdwinRange::Window`] each bucket keeps two more numbers, its
    /// extent: [`Self::state_bytes_with`].
    pub fn state_bytes(max_width: u64, max_buckets: usize) -> u64 {
        Self::state_bytes_with(max_width, max_buckets, false)
    }

    /// [`Self::state_bytes`], with each bucket's extent where `extents`.
    pub fn state_bytes_with(max_width: u64, max_buckets: usize, extents: bool) -> u64 {
        let numbers = if extents { 2 } else { 1 };
        let per_level = (max_buckets as u64)
            .saturating_add(1)
            .saturating_mul(16 * numbers)
            .saturating_add(8 * numbers);
        Self::levels_bound(max_width, max_buckets)
            .saturating_mul(per_level)
            .saturating_add(104)
    }

    /// The rows in the window.
    pub fn width(&self) -> f64 {
        self.width
    }

    /// The window's mean, 0 when it is empty (river's `estimation`).
    pub fn estimation(&self) -> f64 {
        if self.width == 0.0 {
            0.0
        } else {
            self.total / self.width
        }
    }

    /// The window's sum (river's `total`).
    pub fn total(&self) -> f64 {
        self.total
    }

    /// The window's sum of squared deviations about its mean (river's
    /// `variance`, which is this sum, not divided).
    pub fn variance(&self) -> f64 {
        self.var
    }

    /// The levels the window holds now.
    pub fn levels(&self) -> usize {
        self.levels.len()
    }

    /// Whether a restored state keeps the shape every row relies on: a level
    /// to put a row in, a top level holding the oldest bucket (the one a
    /// drop takes), no level past `M` buckets, settings in range, a width
    /// that is the rows its buckets hold and no more than `max_width`, and
    /// sums that are finite numbers, the spread at least 0.
    pub fn sound(&self) -> bool {
        let held: f64 = self
            .levels
            .iter()
            .enumerate()
            .map(|(i, l)| (l.len() as f64) * 2f64.powi(i as i32))
            .sum();
        !self.levels.is_empty()
            && self.levels.len() < 64
            && (self.levels.len() == 1 || self.levels.last().is_some_and(|l| !l.is_empty()))
            && self.levels.iter().all(|l| l.len() <= self.max_buckets)
            && self
                .levels
                .iter()
                .flatten()
                .all(|&[t, v]| t.is_finite() && v.is_finite() && v >= 0.0)
            && self.delta > 0.0
            && self.delta < 1.0
            && self.clock >= 1
            && self.max_buckets >= 1
            && self.min_window_length >= 1
            && self.max_width >= 1
            && held == self.width
            && self.width <= self.max_width as f64
            && self.total.is_finite()
            && self.var.is_finite()
            && self.var >= 0.0
            && (if self.windowed_range() {
                self.extents.len() == self.levels.len()
                    && self
                        .extents
                        .iter()
                        .zip(&self.levels)
                        .all(|(e, l)| e.len() == l.len())
                    && self
                        .extents
                        .iter()
                        .flatten()
                        .all(|&[a, b]| a.is_finite() && b.is_finite() && a <= b)
            } else {
                self.range.is_finite() && self.range > 0.0
            })
    }

    pub fn reset(&mut self) {
        self.total = 0.0;
        self.var = 0.0;
        self.width = 0.0;
        self.tick = 0;
        self.levels.clear();
        self.levels.push(VecDeque::new());
        if self.windowed_range() {
            self.extents.clear();
            self.extents.push(VecDeque::new());
        }
    }

    /// Feed one value. Returns whether the window was cut on this row; under
    /// `restart` it then starts again empty.
    pub fn update(&mut self, x: f64) -> bool {
        self.insert(x);
        // Each pass drops a bucket, so no more than the buckets held can be
        // wanted: a bound that holds the loop to a pass count it cannot reach
        // unless a pass drops nothing.
        let mut held = self.levels.iter().map(VecDeque::len).sum::<usize>();
        while self.width > self.max_width as f64 && held > 0 {
            self.delete_oldest();
            held -= 1;
        }
        let change = self.detect();
        if change && self.restart {
            self.reset();
        }
        change
    }

    fn insert(&mut self, x: f64) {
        self.levels[0].push_back([x, 0.0]);
        if self.windowed_range() {
            self.extents[0].push_back([x, x]);
        }
        self.width += 1.0;
        let mut inc = 0.0;
        if self.width > 1.0 {
            let dv = x - self.total / (self.width - 1.0);
            inc = (self.width - 1.0) * dv * dv / self.width;
        }
        // A finite value past the square root of the largest double squares
        // past it: the sums saturate there, the largest spread a double holds,
        // so the state stays finite, as a restore holds it to (`sound`).
        self.var = saturated(self.var + inc);
        self.total = saturated(self.total + x);
        self.compress();
    }

    fn compress(&mut self) {
        let m = self.max_buckets;
        let mut idx = 0;
        while self.levels[idx].len() == m + 1 {
            let n1 = (1u64 << idx) as f64;
            let [t0, v0] = self.levels[idx].pop_front().expect("M + 1 buckets");
            let [t1, v1] = self.levels[idx].pop_front().expect("M + 1 buckets");
            let mu1 = t0 / n1;
            let mu2 = t1 / n1;
            let total12 = saturated(t0 + t1);
            let temp = n1 * 0.5 * (mu1 - mu2) * (mu1 - mu2);
            let v12 = saturated(v0 + v1 + temp);
            if idx + 1 >= self.levels.len() {
                self.levels.push(VecDeque::new());
            }
            self.levels[idx + 1].push_back([total12, v12]);
            if self.windowed_range() {
                let [a0, b0] = self.extents[idx].pop_front().expect("M + 1 extents");
                let [a1, b1] = self.extents[idx].pop_front().expect("M + 1 extents");
                if idx + 1 >= self.extents.len() {
                    self.extents.push(VecDeque::new());
                }
                self.extents[idx + 1].push_back([a0.min(a1), b0.max(b1)]);
            }
            if self.levels[idx + 1].len() <= m {
                break;
            }
            idx += 1;
        }
    }

    /// Drop the oldest bucket, the first of the top level.
    fn delete_oldest(&mut self) {
        let last = self.levels.len() - 1;
        let n = (1u64 << last) as f64;
        if self.levels[last].pop_front().is_none() {
            return;
        }
        if self.windowed_range() {
            self.extents[last].pop_front();
        }
        self.width -= n;
        // The top level goes when it empties, and so does any below it left
        // empty (only at `M = 1`, where a merge leaves no bucket behind).
        while self.levels.len() > 1 && self.levels.last().is_some_and(VecDeque::is_empty) {
            self.levels.pop();
            if self.windowed_range() {
                self.extents.pop();
            }
        }
        self.resum();
    }

    /// `S` and `V` taken again from the buckets left, oldest first, by the
    /// pairwise merge of their sums and sums of squared deviations:
    ///
    /// ```text
    /// S = S_a + S_b,   V = V_a + V_b + n_a n_b / (n_a + n_b) (S_a/n_a − S_b/n_b)²
    /// ```
    ///
    /// rather than the dropped bucket's share taken out of them. That
    /// subtraction cancels where the bucket held most of the window's
    /// spread: two outliers leaving a window of zeros left `V` at `1.2e24`,
    /// or below 0, where `ε` is NaN, and either way no split was cut again.
    /// At most `M` buckets a level, so the scan this follows costs as much.
    fn resum(&mut self) {
        let (mut n, mut total, mut var) = (0.0f64, 0.0f64, 0.0f64);
        for (idx, level) in self.levels.iter().enumerate().rev() {
            let nb = (1u64 << idx) as f64;
            for &[sb, vb] in level {
                if n == 0.0 {
                    (n, total, var) = (nb, sb, vb);
                    continue;
                }
                let d = total / n - sb / nb;
                var = saturated(var + vb + n * nb / (n + nb) * d * d);
                total = saturated(total + sb);
                n += nb;
            }
        }
        self.total = total;
        self.var = var;
    }

    /// The cut test for a split of `n0` older and `n1` newer rows whose means
    /// differ by `dmean`, `d` the scan's `ln(2 ln(width) / delta)`.
    /// `r` the range `R` ([`Self::range_now`]); at river's 1 the product is
    /// river's term to the bit.
    fn cut(&self, n0: f64, n1: f64, dmean: f64, d: f64, r: f64) -> bool {
        let l = self.min_window_length as f64;
        let m_recip = 1.0 / (n0 - l + 1.0) + 1.0 / (n1 - l + 1.0);
        let eps =
            (2.0 * m_recip * (self.var / self.width) * d).sqrt() + 2.0 / 3.0 * d * m_recip * r;
        dmean.abs() > eps
    }

    fn detect(&mut self) -> bool {
        let mut change = false;
        self.tick += 1;
        if !self.tick.is_multiple_of(self.clock) || self.width <= self.grace_period as f64 {
            return false;
        }
        let l = self.min_window_length as f64;
        // A pass that cuts drops a bucket and one that does not ends the scan,
        // so the buckets held and one more are the most passes there can be.
        let passes = self.levels.iter().map(VecDeque::len).sum::<usize>() + 1;
        for _ in 0..passes {
            let mut reduce = false;
            let (mut n0, mut n1, mut u0, mut u1) = (0.0, self.width, 0.0, self.total);
            // The width holds until a cut, after which the scan starts over.
            let d = (2.0 * self.width.ln() / self.delta).ln();
            let r = self.range_now();
            'scan: for idx in (0..self.levels.len()).rev() {
                let size = (1u64 << idx) as f64;
                let cnt = self.levels[idx].len();
                for k in 0..cnt {
                    n0 += size;
                    n1 -= size;
                    let tk = self.levels[idx][k][0];
                    u0 += tk;
                    u1 -= tk;
                    // The newest bucket leaves no newer part; at one bucket
                    // a level with level 0 empty it is not level 0's, and
                    // `u1 / 0` read a rounding of the sums as infinite.
                    if (idx == 0 && k == cnt - 1) || n1 <= 0.0 {
                        break 'scan;
                    }
                    let dmean = (u0 / n0) - (u1 / n1);
                    if n1 >= l && n0 >= l && self.cut(n0, n1, dmean, d, r) {
                        reduce = true;
                        change = true;
                        if self.width > 0.0 {
                            self.delete_oldest();
                            break 'scan;
                        }
                    }
                }
            }
            if !reduce {
                break;
            }
        }
        change
    }
}

#[cfg(test)]
#[path = "adwin_tests.rs"]
mod tests;

/// `v`, or the largest double where a sum of finite values went past it.
#[inline]
fn saturated(v: f64) -> f64 {
    v.min(f64::MAX)
}
