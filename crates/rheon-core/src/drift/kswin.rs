//! KSWIN, the Kolmogorov-Smirnov windowed test (Raab, Heusinger and Schleif
//! 2020), drawn from a seeded generator so that it is reproducible.
//!
//! A ring holds the last `W` (`window_size`) values. Once it is full, each row
//! draws `r` (`stat_size`) of the older `n = W − r` values without
//! replacement and compares them with the newest `r` by the two-sample
//! Kolmogorov-Smirnov statistic
//!
//! ```text
//! D = max_x |F_old(x) − F_new(x)| = h / r,   h an integer
//! ```
//!
//! and flags, as river's `drift.KSWIN` does, where the exact two-sided
//! p-value is at most `alpha` and `D > 0.1`. With samples of equal size the
//! exact null tail is the Gnedenko-Korolyuk sum
//!
//! ```text
//! P(D ≥ h/r) = 2 Σ_{j ≥ 1, jh ≤ r} (−1)^(j+1) C(2r, r − jh) / C(2r, r)
//! C(2r, r − m) / C(2r, r) = Π_{i<m} (r − i) / (r + 1 + i)
//! ```
//!
//! which falls as `h` grows, so the test is `h ≥ h*` for the least `h*`
//! whose tail is at most `alpha`, found once from the settings: each row's
//! decision is a comparison of integers, and no `libm` result is involved.
//!
//! **The draw** is the library's own, not CPython's (river's): a SplitMix64
//! stream from `seed`, taken by Floyd's algorithm. For `j` from `n − r` to `n
//! − 1`, `t` is uniform on `[0, j]` by Lemire's bounded draw, and the older
//! value at `t` is taken unless `t` was taken already, when the one at `j` is.
//! The values are indexed oldest first. The generator is part of the state,
//! eight bytes, so a stream fed in any chunks draws the same samples.
//!
//! After a flag the detector starts again from an empty ring and the seed
//! (`restart`, river's form; its generator is re-seeded too), or keeps the
//! newest `r` values and goes on drawing (the paper's form).

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

use crate::SplitMix64;
use crate::corrchange::below_by;

/// The settings of a [`Kswin`], river's names and defaults.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KswinCfg {
    /// The test's level, in `(0, 1)`: river's 0.005.
    pub alpha: f64,
    /// The ring's rows, `W`, at least `2 r`: river's 100.
    pub window_size: usize,
    /// The rows of each sample, `r`: river's 30.
    pub stat_size: usize,
    /// The draw's seed.
    pub seed: u64,
    /// Start again empty after a flag (river), or keep the newest `r` (the
    /// paper).
    pub restart: bool,
}

impl Default for KswinCfg {
    fn default() -> Self {
        Self {
            alpha: 0.005,
            window_size: 100,
            stat_size: 30,
            seed: 0,
            restart: true,
        }
    }
}

/// What a row's test works in, rebuilt as it is needed and never saved.
#[derive(Debug, Clone, Default)]
struct Scratch {
    /// `h*`, from the settings.
    crit: Option<usize>,
    old: Vec<f64>,
    new: Vec<f64>,
    /// One bit an older value, set while it is taken.
    taken: Vec<u64>,
}

impl PartialEq for Scratch {
    /// Scratch is no state: two detectors with the same state are equal
    /// whatever their scratch holds.
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

/// KSWIN on one stream of values: the module doc has the test and the draw.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Kswin {
    alpha: f64,
    window_size: usize,
    stat_size: usize,
    seed: u64,
    restart: bool,
    /// The last `window_size` values, oldest first.
    window: VecDeque<f64>,
    rng: SplitMix64,
    #[serde(skip)]
    scratch: Scratch,
}

/// `P(D_{r,r} ≥ h/r)`, the module doc's sum in doubles, the binomial ratios
/// built by their product. 1 at `h = 0`.
pub fn ks_equal_tail(r: usize, h: usize) -> f64 {
    if h == 0 {
        return 1.0;
    }
    // ratio[m] = C(2r, r − m) / C(2r, r).
    let mut ratio = Vec::with_capacity(r + 1);
    let mut t = 1.0f64;
    ratio.push(t);
    for i in 0..r {
        t *= (r - i) as f64 / (r + 1 + i) as f64;
        ratio.push(t);
    }
    let mut sum = 0.0;
    let mut j = 1;
    while j * h <= r {
        let term = ratio[j * h];
        if j % 2 == 1 {
            sum += term;
        } else {
            sum -= term;
        }
        j += 1;
    }
    2.0 * sum
}

/// `h*`: the least `h` in `1..=r` whose tail is at most `alpha`, or `r + 1`
/// where none is.
pub fn ks_critical_h(r: usize, alpha: f64) -> usize {
    (1..=r)
        .find(|&h| ks_equal_tail(r, h) <= alpha)
        .unwrap_or(r + 1)
}

/// `r D` for two samples of `r` values each, sorted: the largest gap between
/// the counts at or below a value of either. Ties advance both samples past
/// the value before the gap is read, as `scipy.stats.ks_2samp` reads it.
fn ks_h(a: &[f64], b: &[f64]) -> usize {
    let r = a.len();
    let (mut i, mut j, mut h) = (0, 0, 0);
    // Each pass moves a cursor past at least one value, so `2r` passes end
    // the scan: a bound a pass that moved nothing would meet, not pass.
    for _ in 0..2 * r {
        if i >= r || j >= r {
            break;
        }
        let v = if a[i] <= b[j] { a[i] } else { b[j] };
        // The values equal to `v` at each cursor, counted and not looped
        // over: a cursor that stayed would be a count that is wrong, not an
        // endless loop.
        i += a[i..].iter().take_while(|&&x| x == v).count();
        j += b[j..].iter().take_while(|&&x| x == v).count();
        h = h.max(i.abs_diff(j));
    }
    h
}

impl Kswin {
    pub fn new(cfg: KswinCfg) -> Self {
        debug_assert!(cfg.alpha > 0.0 && cfg.alpha < 1.0);
        debug_assert!(cfg.stat_size >= 1 && cfg.window_size >= 2 * cfg.stat_size);
        Self {
            alpha: cfg.alpha,
            window_size: cfg.window_size,
            stat_size: cfg.stat_size,
            seed: cfg.seed,
            restart: cfg.restart,
            window: VecDeque::with_capacity(cfg.window_size),
            rng: SplitMix64::new(cfg.seed),
            scratch: Scratch::default(),
        }
    }

    /// The bytes a KSWIN's state holds: its ring, `8 W`, its generator and
    /// its settings.
    pub fn state_bytes(window_size: usize) -> u64 {
        8 * window_size as u64 + 48
    }

    /// The values in the ring, oldest first.
    pub fn window(&self) -> &VecDeque<f64> {
        &self.window
    }

    /// Whether a restored state keeps the shape every row relies on: samples
    /// of at least one row, a ring that holds twice that, no more values
    /// than it holds, and a level in range.
    pub fn sound(&self) -> bool {
        self.stat_size >= 1
            && self.window_size >= 2 * self.stat_size
            && self.window.len() <= self.window_size
            && self.alpha > 0.0
            && self.alpha < 1.0
    }

    pub fn reset(&mut self) {
        self.window.clear();
        self.rng = SplitMix64::new(self.seed);
    }

    /// Draw the older sample into the scratch, in the order drawn, moving
    /// the generator as a test does.
    fn draw(&mut self) -> &[f64] {
        let (r, n) = (self.stat_size, self.window_size - self.stat_size);
        let sc = &mut self.scratch;
        sc.taken.clear();
        sc.taken.resize(n.div_ceil(64), 0);
        sc.old.clear();
        for j in (n - r)..n {
            let t = below_by(j as u64 + 1, || self.rng.next_u64()) as usize;
            let pick = if sc.taken[t / 64] >> (t % 64) & 1 == 1 {
                j
            } else {
                t
            };
            sc.taken[pick / 64] |= 1 << (pick % 64);
            sc.old.push(self.window[pick]);
        }
        &sc.old
    }

    /// Feed one value. Returns whether this row's test rejects; before the
    /// ring is full there is no test.
    pub fn update(&mut self, x: f64) -> bool {
        if self.window.len() == self.window_size {
            self.window.pop_front();
        }
        self.window.push_back(x);
        if self.window.len() < self.window_size {
            return false;
        }
        let r = self.stat_size;
        let crit = *self
            .scratch
            .crit
            .get_or_insert_with(|| ks_critical_h(r, self.alpha));
        self.draw();
        let sc = &mut self.scratch;
        sc.new.clear();
        sc.new.extend(self.window.range(self.window_size - r..));
        sc.old.sort_unstable_by(f64::total_cmp);
        sc.new.sort_unstable_by(f64::total_cmp);
        let h = ks_h(&sc.old, &sc.new);
        // `D > 0.1` is `h / r > 1/10`, read in integers.
        let flag = h >= crit && 10 * h > r;
        if flag {
            if self.restart {
                self.reset();
            } else {
                let keep = self.window_size - r;
                self.window.drain(..keep);
            }
        }
        flag
    }

    /// This row's `h` and the samples it compared, without moving the state:
    /// for tests that hold the statistic to another computation.
    /// `None` where the ring is not full and no test runs.
    #[cfg(test)]
    fn peek(&self, x: f64) -> Option<(Vec<f64>, Vec<f64>)> {
        let mut c = self.clone();
        if c.window.len() == c.window_size {
            c.window.pop_front();
        }
        c.window.push_back(x);
        if c.window.len() < c.window_size {
            return None;
        }
        let old = c.draw().to_vec();
        let new = c
            .window
            .range(c.window_size - c.stat_size..)
            .copied()
            .collect();
        Some((old, new))
    }
}

#[cfg(test)]
#[path = "kswin_tests.rs"]
mod tests;
