//! Window-based two-sample Kolmogorov-Smirnov feature health test.
//!
//! Each feature keeps a fixed ring of its last `window` positive-weight values.
//! Once the ring is full, each read compares the older reference sample
//! (`window - recent` values) with the newer recent sample (`recent` values) by
//! the two-sample Kolmogorov-Smirnov test:
//!
//! ```text
//! D = max_x |F_ref(x) - F_recent(x)|
//! ```
//!
//! Ties advance both empirical CDFs past the tied value before reading the gap,
//! matching `scipy.stats.ks_2samp`. The two-sided p-value is computed exactly
//! by the Gnedenko-Korolyuk Horner formula when `n1 == n2` and by exact
//! lattice-exit dynamic programming when `n1 != n2` (up to `n1 * n2 <= 250_000`),
//! falling back to Smirnov's asymptotic Kolmogorov distribution with Stephens'
//! finite-sample correction for larger unequal sample products.
//!
//! **Out of sample and zero-weight transparent.** Readings come from the ring
//! before the row is pushed, and rows with weight `<= 0` or non-finite feature
//! values enter no ring. The ring holds `8 * window * k` bytes per group,
//! statically capped at [`FEATURE_KS_MAX_STATE_BYTES`] (256 KiB).

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

/// Per-group byte cap for the feature-health KS rings (256 KiB).
pub const FEATURE_KS_MAX_STATE_BYTES: u64 = crate::clock_span::DEFAULT_STATE_BYTES;

/// Maximum unequal sample-size product `n1 * n2` evaluated by exact lattice DP.
/// Equal sample sizes use the `O(n)` Horner formula for any size.
const EXACT_PRODUCT_LIMIT: u64 = 250_000;

/// Bytes held by `k` rings of `window` `f64` values. Saturates on overflow.
pub fn feature_ks_state_bytes(k: usize, window: usize) -> u64 {
    (k as u64).saturating_mul(window as u64).saturating_mul(8)
}

/// Reusable scratch buffers for zero-allocation reads. Never serialized and
/// ignored by `PartialEq`.
#[derive(Debug, Clone, Default)]
struct KsScratch {
    a: Vec<f64>,
    b: Vec<f64>,
    row: Vec<f64>,
    nxt: Vec<f64>,
    /// Per-feature `(last_h, last_p)` cache: for fixed `(n1, n2)` the p-value
    /// is a pure function of the integer numerator `h`.
    cache: Vec<Option<(u64, f64)>>,
}

impl PartialEq for KsScratch {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

/// Per-feature two-sample Kolmogorov-Smirnov window monitor.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FeatureHealthKs {
    window: usize,
    recent: usize,
    rings: Vec<VecDeque<f64>>,
    #[serde(skip)]
    scratch: KsScratch,
}

/// Greatest common divisor by Euclid's algorithm.
fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a
}

/// Two-sample KS statistic `D` and exact integer numerator `h` over `lcm(n1, n2)`
/// for two non-empty sorted slices. Ties advance both cursors past the tied
/// value before measuring `|F_1(x) - F_2(x)|`, matching `scipy.stats.ks_2samp`.
pub fn ks_2samp_stat_sorted(a: &[f64], b: &[f64]) -> (f64, u64, u64) {
    let (n1, n2) = (a.len() as u64, b.len() as u64);
    debug_assert!(n1 > 0 && n2 > 0);
    let g = gcd(n1, n2);
    let (n1g, n2g) = (n1 / g, n2 / g);
    let lcm = n1g * n2;
    let (mut i, mut j) = (0usize, 0usize);
    let mut h = 0u64;
    while i < a.len() && j < b.len() {
        let v = if a[i] <= b[j] { a[i] } else { b[j] };
        while i < a.len() && a[i] == v {
            i += 1;
        }
        while j < b.len() && b[j] == v {
            j += 1;
        }
        let gap = (i as u64 * n2g).abs_diff(j as u64 * n1g);
        if gap > h {
            h = gap;
        }
    }
    (h as f64 / lcm as f64, h, lcm)
}

/// Exact two-sided p-value for equal sample sizes `n1 == n2 == n`, given
/// integer `h = n * D` (`1 <= h <= n`), using the factored Horner form of the
/// Gnedenko-Korolyuk alternating sum (avoids overflow and cancellation).
fn exact_equal_p(n: usize, h: usize) -> f64 {
    let mut p = 0.0f64;
    for k in (1..=(n / h)).rev() {
        let base = (k - 1) * h;
        let mut p1 = 1.0f64;
        for j in 0..h {
            let num = (n - base - j) as f64;
            let den = (n + base + j + 1) as f64;
            p1 = num * p1 / den;
        }
        p = p1 * (1.0 - p);
    }
    (2.0 * p).clamp(0.0, 1.0)
}

/// Exact two-sided two-sample KS p-value for unequal `(n1, n2)` and integer
/// numerator `h` over `lcm(n1, n2)`, by Markov lattice-exit dynamic programming.
///
/// Tracks the exact probability mass of a uniform lattice path from `(0, 0)` to
/// `(n1, n2)` that remains strictly inside `|i * n2g - j * n1g| < h`, summing
/// every transition that exits the strip directly into `p_out`. All values stay
/// in `[0, 1]`: no binomial coefficients, no overflow, no catastrophic
/// cancellation even at tiny p-values.
fn exact_unequal_p(n1: usize, n2: usize, h: u64, row: &mut Vec<f64>, nxt: &mut Vec<f64>) -> f64 {
    let g = gcd(n1 as u64, n2 as u64);
    let (mut m, mut n) = (n1, n2);
    let (mut mg, mut ng) = (n1 as u64 / g, n2 as u64 / g);
    // Keep the smaller dimension in the inner loop and scratch buffers.
    if n > m {
        std::mem::swap(&mut m, &mut n);
        std::mem::swap(&mut mg, &mut ng);
    }
    row.clear();
    row.resize(n + 1, 0.0);
    nxt.clear();
    nxt.resize(n + 1, 0.0);
    row[0] = 1.0;
    let mut p_out = 0.0f64;

    let bounds = |i: usize| -> (usize, usize) {
        let center = i as u64 * ng;
        let j_lo = if center < h {
            0
        } else {
            ((center - h) / mg + 1) as usize
        };
        let j_hi = ((center + h - 1) / mg).min(n as u64) as usize;
        (j_lo, j_hi)
    };

    let (mut j_lo, mut j_hi) = bounds(0);
    for i in 0..=m {
        let (nj_lo, nj_hi) = if i < m { bounds(i + 1) } else { (1, 0) };
        for j in j_lo..=j_hi {
            let p = row[j];
            row[j] = 0.0;
            if p == 0.0 {
                continue;
            }
            let rem_i = m - i;
            let rem_j = n - j;
            let rem = rem_i + rem_j;
            if rem == 0 {
                continue;
            }
            let inv_rem = 1.0 / rem as f64;
            if rem_i > 0 {
                let pr = p * (rem_i as f64 * inv_rem);
                if j >= nj_lo && j <= nj_hi {
                    nxt[j] += pr;
                } else {
                    p_out += pr;
                }
            }
            if rem_j > 0 {
                let pu = p * (rem_j as f64 * inv_rem);
                if j < j_hi {
                    row[j + 1] += pu;
                } else {
                    p_out += pu;
                }
            }
        }
        std::mem::swap(row, nxt);
        j_lo = nj_lo;
        j_hi = nj_hi;
    }
    p_out.clamp(0.0, 1.0)
}

/// Smirnov's asymptotic two-sided KS survival function using `libm` only:
/// direct alternating sum for `z >= 0.7`, Jacobi theta transform for `z < 0.7`.
fn asymptotic_two_sided_p(n1: usize, n2: usize, d: f64) -> f64 {
    if d <= 0.0 {
        return 1.0;
    }
    if d >= 1.0 {
        return 0.0;
    }
    let (f1, f2) = (n1 as f64, n2 as f64);
    let en = libm::sqrt(f1 * f2 / (f1 + f2));
    // Stephens' (1970) continuity correction for large sample sizes.
    let z = (en + 0.12 + 0.11 / en) * d;
    if z >= 0.7 {
        let z2 = -2.0 * z * z;
        let mut sum = 0.0f64;
        for k in 1..=200u32 {
            let kf = f64::from(k);
            let term = libm::exp(z2 * kf * kf);
            sum += if k % 2 == 1 { term } else { -term };
            if term <= 0.25 * f64::EPSILON * sum.abs() {
                break;
            }
        }
        (2.0 * sum).clamp(0.0, 1.0)
    } else {
        use std::f64::consts::PI;
        let a = PI * PI / (8.0 * z * z);
        let mut sum = 0.0f64;
        for k in 1..=200u32 {
            let m = f64::from(2 * k - 1);
            let term = libm::exp(-(m * m) * a);
            sum += term;
            if term <= 0.25 * f64::EPSILON * sum {
                break;
            }
        }
        let cdf = (libm::sqrt(2.0 * PI) / z * sum).clamp(0.0, 1.0);
        (1.0 - cdf).clamp(0.0, 1.0)
    }
}

/// Two-sided two-sample KS p-value for sample sizes `(n1, n2)` and integer
/// statistic `h` over `lcm(n1, n2)`.
pub fn ks_2samp_pvalue_in(
    n1: usize,
    n2: usize,
    h: u64,
    lcm: u64,
    row: &mut Vec<f64>,
    nxt: &mut Vec<f64>,
) -> f64 {
    if h == 0 {
        return 1.0;
    }
    if n1 == n2 {
        return exact_equal_p(n1, h as usize);
    }
    if (n1 as u64).saturating_mul(n2 as u64) <= EXACT_PRODUCT_LIMIT {
        exact_unequal_p(n1, n2, h, row, nxt)
    } else {
        asymptotic_two_sided_p(n1, n2, h as f64 / lcm as f64)
    }
}

impl FeatureHealthKs {
    /// Empty rings for `k` features, comparing the older `window - recent`
    /// values against the newest `recent` values once `window` positive-weight
    /// rows have arrived.
    pub fn new(k: usize, window: usize, recent: usize) -> Self {
        debug_assert!(recent >= 1 && window > recent);
        let rings = (0..k).map(|_| VecDeque::with_capacity(window)).collect();
        let n_ref = window - recent;
        let small = n_ref.min(recent);
        let scratch = KsScratch {
            a: Vec::with_capacity(n_ref),
            b: Vec::with_capacity(recent),
            row: Vec::with_capacity(small + 1),
            nxt: Vec::with_capacity(small + 1),
            cache: vec![None; k],
        };
        Self {
            window,
            recent,
            rings,
            scratch,
        }
    }

    pub fn window(&self) -> usize {
        self.window
    }

    pub fn recent(&self) -> usize {
        self.recent
    }

    pub fn n_features(&self) -> usize {
        self.rings.len()
    }

    /// Shape check on restore: matches `k`, `window`, `recent`, respects the
    /// byte cap, and holds only finite values within capacity.
    pub fn has_shape(&self, k: usize, window: usize, recent: usize) -> bool {
        self.window == window
            && self.recent == recent
            && recent >= 1
            && window > recent
            && self.rings.len() == k
            && feature_ks_state_bytes(k, window) <= FEATURE_KS_MAX_STATE_BYTES
            && self
                .rings
                .iter()
                .all(|r| r.len() <= window && r.iter().all(|v| v.is_finite()))
    }

    /// Push one row of features if `w > 0` and finite, and every feature is
    /// finite. Zero-weight rows enter no ring (hard rule 9).
    pub fn update(&mut self, x: &[f64], w: f64) {
        if !(w > 0.0 && w.is_finite())
            || x.len() != self.rings.len()
            || !x.iter().all(|v| v.is_finite())
        {
            return;
        }
        let cap = self.window;
        for (ring, &xi) in self.rings.iter_mut().zip(x) {
            if ring.len() == cap {
                ring.pop_front();
            }
            ring.push_back(xi);
        }
    }

    /// Out-of-sample two-sample KS `(statistic, p_value)` for feature `i` from
    /// the current ring, before the incoming row is pushed. Returns `None`
    /// until the ring holds `window` values.
    pub fn read(&mut self, i: usize) -> Option<(f64, f64)> {
        let ring = self.rings.get(i)?;
        if ring.len() < self.window {
            return None;
        }
        let n_ref = self.window - self.recent;
        let sc = &mut self.scratch;
        if sc.cache.len() != self.rings.len() {
            sc.cache.resize(self.rings.len(), None);
        }
        sc.a.clear();
        sc.b.clear();
        let (front, back) = ring.as_slices();
        if n_ref <= front.len() {
            sc.a.extend_from_slice(&front[..n_ref]);
            sc.b.extend_from_slice(&front[n_ref..]);
            sc.b.extend_from_slice(back);
        } else {
            sc.a.extend_from_slice(front);
            let split = n_ref - front.len();
            sc.a.extend_from_slice(&back[..split]);
            sc.b.extend_from_slice(&back[split..]);
        }
        sc.a.sort_unstable_by(f64::total_cmp);
        sc.b.sort_unstable_by(f64::total_cmp);
        let (d, h, lcm) = ks_2samp_stat_sorted(&sc.a, &sc.b);
        let p = match sc.cache[i] {
            Some((ch, cp)) if ch == h => cp,
            _ => {
                let p = ks_2samp_pvalue_in(n_ref, self.recent, h, lcm, &mut sc.row, &mut sc.nxt);
                sc.cache[i] = Some((h, p));
                p
            }
        };
        Some((d, p))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equal_and_unequal_exact_pvalues_match_known_values() {
        // n1 = n2 = 2, completely separated: h = 2, P = 2 / C(4, 2) = 1/3.
        let mut row = Vec::new();
        let mut nxt = Vec::new();
        let p = ks_2samp_pvalue_in(2, 2, 2, 2, &mut row, &mut nxt);
        assert!((p - 1.0 / 3.0).abs() < 1e-15, "got {p}");

        // n1 = 3, n2 = 2, completely separated: h = 6, lcm = 6, P = 2 / C(5, 2) = 0.2.
        let p = ks_2samp_pvalue_in(3, 2, 6, 6, &mut row, &mut nxt);
        assert!((p - 0.2).abs() < 1e-15, "got {p}");

        // h = 0 => p = 1.0.
        assert_eq!(ks_2samp_pvalue_in(70, 30, 0, 210, &mut row, &mut nxt), 1.0);
    }

    #[test]
    fn ties_advance_both_ecdfs_before_measuring_gap() {
        let mut a = vec![1.0, 1.0, 2.0, 3.0];
        let mut b = vec![1.0, 1.0, 2.0, 3.0];
        a.sort_unstable_by(f64::total_cmp);
        b.sort_unstable_by(f64::total_cmp);
        let (d, h, _) = ks_2samp_stat_sorted(&a, &b);
        assert_eq!(d, 0.0);
        assert_eq!(h, 0);

        let mut a = vec![0.0, 0.0, 1.0, 1.0];
        let mut b = vec![0.0, 1.0, 1.0, 1.0];
        a.sort_unstable_by(f64::total_cmp);
        b.sort_unstable_by(f64::total_cmp);
        // At 0.0: F_a = 2/4 = 0.5, F_b = 1/4 = 0.25 => diff = 0.25.
        // At 1.0: F_a = 1.0, F_b = 1.0 => diff = 0.0.
        let (d, h, lcm) = ks_2samp_stat_sorted(&a, &b);
        assert!((d - 0.25).abs() < 1e-15);
        assert_eq!((h, lcm), (1, 4));
    }

    #[test]
    fn zero_weight_rows_never_enter_the_ring_and_roundtrip_matches() {
        let mut ks = FeatureHealthKs::new(2, 10, 4);
        for i in 0..10 {
            assert!(ks.read(0).is_none());
            // Zero-weight extreme outlier must be ignored.
            ks.update(&[1e9, -1e9], 0.0);
            ks.update(&[i as f64, (i % 3) as f64], 1.0);
        }
        let before = ks.read(0).expect("full");
        assert!(before.0 > 0.0 && before.0 <= 1.0);
        assert!(before.1 >= 0.0 && before.1 <= 1.0);
        // Round-trip both compact and named msgpack.
        for bytes in [
            rmp_serde::to_vec(&ks).unwrap(),
            rmp_serde::to_vec_named(&ks).unwrap(),
        ] {
            let mut restored: FeatureHealthKs = rmp_serde::from_slice(&bytes).unwrap();
            assert!(restored.has_shape(2, 10, 4));
            assert_eq!(restored.read(0), Some(before));
            assert_eq!(restored.read(1), ks.read(1));
        }
    }

    #[test]
    fn asymptotic_fallback_is_monotone_and_bounded() {
        let p_small = asymptotic_two_sided_p(1000, 300, 0.02);
        let p_mid = asymptotic_two_sided_p(1000, 300, 0.08);
        let p_big = asymptotic_two_sided_p(1000, 300, 0.25);
        assert!((0.0..=1.0).contains(&p_small));
        assert!(p_small > p_mid && p_mid > p_big && p_big >= 0.0);
    }
}
