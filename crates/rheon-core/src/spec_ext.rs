//! Additional specification and calibration diagnostics: Engle's ARCH LM test,
//! White's heteroskedasticity test, and Spiegelhalter's z-test for binary
//! probability calibration.
//!
//! Each accumulator uses exponentially weighted moments/sums with zero
//! allocation on the per-row update path, ages on missing or zero-weight rows
//! (hard rule 9), and fits inside a bounded per-group state budget.

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

use crate::solve::{KEPT_ORDER, Solves, solve_in};

/// Default per-group byte cap for specification diagnostic state (256 KiB).
pub const SPEC_DIAG_CAP_BYTES: usize = 256 * 1024;

/// Default lag order for Engle's ARCH LM test.
pub const DEFAULT_ARCH_LM_LAGS: usize = 10;

/// Largest feature count `k` whose full White test (linear + squares +
/// cross-products) fits inside [`SPEC_DIAG_CAP_BYTES`] for a single slot.
pub const MAX_WHITE_FULL_K: usize = 21;

/// Largest feature count `k` whose squares-only White test
/// (`interaction_terms = false`) fits inside [`SPEC_DIAG_CAP_BYTES`] for a
/// single slot.
pub const MAX_WHITE_SQUARES_K: usize = 253;

#[derive(Default)]
struct ReadScratch {
    solves: Solves,
    a: Vec<f64>,
    b: Vec<f64>,
    x: Vec<f64>,
}

thread_local! {
    static READ: std::cell::Cell<Option<Box<ReadScratch>>> = const { std::cell::Cell::new(None) };
}

fn with_scratch<T>(order: usize, f: impl FnOnce(&mut ReadScratch) -> T) -> T {
    let mut sc = READ.take().unwrap_or_default();
    let out = f(&mut sc);
    if order >= KEPT_ORDER {
        *sc = ReadScratch::default();
    }
    READ.set(Some(sc));
    out
}

/// Centred `R²` of column `y` regressed on columns `xs` in `m`.
fn r2_from(m: &crate::EwCov, xs: &[usize], y: usize) -> Option<f64> {
    let var_y = m.var(y);
    if var_y.is_nan() || var_y <= 0.0 {
        return None;
    }
    let p = xs.len();
    if p == 0 {
        return Some(0.0);
    }
    with_scratch(p, |sc| {
        sc.a.clear();
        sc.a.resize(p * p, 0.0);
        for (r, &i) in xs.iter().enumerate() {
            for (c, &j) in xs[..=r].iter().enumerate() {
                sc.a[r * p + c] = m.cov(i, j);
            }
        }
        sc.b.clear();
        sc.b.extend(xs.iter().map(|&i| m.cov(i, y)));
        let attempts = solve_in(&mut sc.solves, &sc.a, &sc.b, p, 1, &mut sc.x)?;
        if attempts != 0 {
            return None;
        }
        let fit: f64 = sc.x.iter().zip(&sc.b).map(|(a, b)| a * b).sum();
        Some((fit / var_y).clamp(0.0, 1.0))
    })
}

/// Engle's (1982) ARCH LM test on out-of-sample squared residuals `e_t²`:
/// auxiliary regression of `e_t²` on `(1, e_{t-1}², ..., e_{t-L}²)`, emitting
/// `LM = n_kish * R² ~ χ²(L)`.
///
/// A row enters the moments once `L` prior scored squared residuals are in the
/// ring (matching `statsmodels.stats.diagnostic.het_arch`). Zero-weight and
/// non-finite rows only age the moments and do not enter the lag ring.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ArchLm {
    lags: usize,
    /// EW centred moments of `(e_t², e_{t-1}², ..., e_{t-L}²)`, width `lags + 1`.
    moments: crate::EwCov,
    /// Last `lags` scored squared residuals, newest first.
    ring: VecDeque<f64>,
    #[serde(skip)]
    row: crate::ewcov::Scratch,
}

impl ArchLm {
    /// Empty ARCH LM accumulator of lag order `lags` (`lags >= 1`).
    pub fn new(lags: usize) -> Self {
        Self {
            lags,
            moments: crate::EwCov::new(lags + 1).without_runs(),
            ring: VecDeque::with_capacity(lags),
            row: Default::default(),
        }
    }

    /// Exact byte footprint of one slot's state at lag order `lags`.
    pub const fn state_bytes(lags: usize) -> usize {
        let d = lags + 1;
        8 * (d * (d + 1) / 2 + d + 2 + lags)
    }

    /// Whether a restored accumulator matches lag order `lags` and holds only
    /// finite entries.
    pub fn has_shape(&self, lags: usize) -> bool {
        self.lags == lags
            && lags >= 1
            && self.ring.len() <= lags
            && self.moments.has_shape(lags + 1)
            && self.moments.q_sum().is_some()
            && self.ring.iter().all(|v| v.is_finite() && *v >= 0.0)
    }

    /// Age the moments by decay factor `lam` without folding a row.
    pub fn age(&mut self, lam: f64) {
        self.moments.decay(lam);
    }

    /// Clear the lag ring across a clock break (capped gap or session change).
    pub fn clear_lags(&mut self) {
        self.ring.clear();
    }

    /// Fold one scored row's out-of-sample residual `e` at decay `lam` and
    /// weight `w`. Non-finite or zero-weight rows only age.
    pub fn update(&mut self, e: f64, lam: f64, w: f64) {
        if !(e.is_finite() && w > 0.0) {
            return self.age(lam);
        }
        let e2 = e * e;
        let guard = w * e2 * e2;
        if !(guard * guard).is_finite() {
            return self.age(lam);
        }
        if self.ring.len() == self.lags {
            self.row.0.clear();
            self.row.0.reserve(self.lags + 1);
            self.row.0.push(e2);
            self.row.0.extend(self.ring.iter().copied());
            self.moments.update(&self.row.0, lam, w);
        } else {
            self.age(lam);
        }
        if self.lags > 0 {
            self.ring.push_front(e2);
            self.ring.truncate(self.lags);
        }
    }

    /// Engle's `LM = n_kish * R²` (`χ²(lags)` under no ARCH), or `None` until
    /// `n_kish > lags + 1` and the lagged moments factorize.
    pub fn lm(&self) -> Option<f64> {
        let n = self.moments.n_kish()?;
        if n <= (self.lags + 1) as f64 {
            return None;
        }
        let xs: Vec<usize> = (1..=self.lags).collect();
        Some(n * r2_from(&self.moments, &xs, 0)?)
    }
}

/// White's (1980) heteroskedasticity test on out-of-sample squared residuals:
/// auxiliary regression of `e²` on a constant and either:
/// - `interaction_terms = true`: `(x_i, x_i x_j for i <= j)` (`q = k + k(k+1)/2`)
/// - `interaction_terms = false`: squares `(x_i²)` alone (`q = k`, matching
///   `statsmodels.stats.diagnostic.het_white(..., interaction_terms=False)`).
///
/// Emits `LM = n_kish * R² ~ χ²(q)`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WhiteTest {
    k: usize,
    interaction_terms: bool,
    /// EW centred moments of `(e², phi...)`, width `1 + q`.
    moments: crate::EwCov,
    #[serde(skip)]
    row: crate::ewcov::Scratch,
}

impl WhiteTest {
    /// Number of auxiliary regressors `q` for `k` features.
    pub const fn n_terms(k: usize, interaction_terms: bool) -> usize {
        if interaction_terms {
            k + k * (k + 1) / 2
        } else {
            k
        }
    }

    /// Exact byte footprint of one slot's state.
    pub const fn state_bytes(k: usize, interaction_terms: bool) -> usize {
        let d = Self::n_terms(k, interaction_terms) + 1;
        8 * (d * (d + 1) / 2 + d + 2)
    }

    /// Empty White test accumulator for `k` features.
    pub fn new(k: usize, interaction_terms: bool) -> Self {
        let q = Self::n_terms(k, interaction_terms);
        Self {
            k,
            interaction_terms,
            moments: crate::EwCov::new(q + 1).without_runs(),
            row: Default::default(),
        }
    }

    /// Whether a restored accumulator matches `(k, interaction_terms)`.
    pub fn has_shape(&self, k: usize, interaction_terms: bool) -> bool {
        let q = Self::n_terms(k, interaction_terms);
        self.k == k
            && self.interaction_terms == interaction_terms
            && self.moments.has_shape(q + 1)
            && self.moments.q_sum().is_some()
    }

    /// Age the moments by decay factor `lam` without folding a row.
    pub fn age(&mut self, lam: f64) {
        self.moments.decay(lam);
    }

    /// Fold one scored row `(x, e)` at decay `lam` and weight `w`. Non-finite
    /// or zero-weight rows, or rows whose terms would overflow moment sums,
    /// only age.
    pub fn update(&mut self, x: &[f64], e: f64, lam: f64, w: f64) {
        if !(e.is_finite() && w > 0.0 && x.len() == self.k && x.iter().all(|v| v.is_finite())) {
            return self.age(lam);
        }
        let e2 = e * e;
        let held = |t: f64| {
            let s = w * t * t;
            (s * s).is_finite()
        };
        if !held(e2) {
            return self.age(lam);
        }
        let q = Self::n_terms(self.k, self.interaction_terms);
        self.row.0.clear();
        self.row.0.reserve(q + 1);
        self.row.0.push(e2);
        if self.interaction_terms {
            for &xi in x {
                if !held(xi) {
                    return self.age(lam);
                }
                self.row.0.push(xi);
            }
            for i in 0..self.k {
                for j in i..self.k {
                    let prod = x[i] * x[j];
                    if !held(prod) {
                        return self.age(lam);
                    }
                    self.row.0.push(prod);
                }
            }
        } else {
            for &xi in x {
                let sq = xi * xi;
                if !held(sq) {
                    return self.age(lam);
                }
                self.row.0.push(sq);
            }
        }
        self.moments.update(&self.row.0, lam, w);
    }

    /// White's `LM = n_kish * R²` over feature subset `idx` (matching
    /// `Specification::breusch_pagan`). `None` if `idx` is empty, `n_kish` is
    /// not past `q_sub + 1`, or the moments do not factorize.
    pub fn lm(&self, idx: &[usize]) -> Option<f64> {
        let n = self.moments.n_kish()?;
        if idx.is_empty() {
            return None;
        }
        let cols: Vec<usize> = if self.interaction_terms {
            let mut v = Vec::with_capacity(idx.len() + idx.len() * (idx.len() + 1) / 2);
            // Linear terms live at 1..=k.
            for &fi in idx {
                v.push(1 + fi);
            }
            // Upper-triangle products `(a, b)` with `a <= b` live starting at `1 + k`.
            let pair_col = |a: usize, b: usize| -> usize {
                let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
                // Offset of row `lo` in an upper triangle of side `k`:
                // sum_{r=0..lo} (k - r) = lo * k - lo * (lo - 1) / 2.
                let row_off = lo * self.k - lo * lo.saturating_sub(1) / 2;
                1 + self.k + row_off + (hi - lo)
            };
            for (pos, &fi) in idx.iter().enumerate() {
                for &fj in &idx[pos..] {
                    v.push(pair_col(fi, fj));
                }
            }
            v
        } else {
            idx.iter().map(|&fi| 1 + fi).collect()
        };
        if n <= (cols.len() + 1) as f64 {
            return None;
        }
        Some(n * r2_from(&self.moments, &cols, 0)?)
    }
}

/// Spiegelhalter's (1986) z-statistic for binary probability calibration:
///
/// ```text
/// z = Σ ω_t (y_t − p_t)(1 − 2 p_t) / sqrt(Σ ω_t² (1 − 2 p_t)² p_t (1 − p_t))
/// ```
///
/// At unit weights with no decay (`lam = 1, w = 1`) this matches MAPIE's
/// `spiegelhalter_statistic`; under decay and weights the squared weights in
/// the denominator keep `Var(z) = 1` under calibrated probabilities.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Spiegelhalter {
    /// `Σ ω (y - p)(1 - 2p)`
    num: f64,
    /// `Σ ω² (1 - 2p)² p (1 - p)`
    den: f64,
    /// `Σ ω`
    w: f64,
}

impl Spiegelhalter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn has_shape(&self) -> bool {
        self.num.is_finite()
            && self.den.is_finite()
            && self.w.is_finite()
            && self.den >= 0.0
            && self.w >= 0.0
    }

    pub fn age(&mut self, lam: f64) {
        self.num *= lam;
        self.den *= lam * lam;
        self.w *= lam;
    }

    /// Fold one scored binary row `(pred, y)` at decay `lam` and weight `w`.
    /// Non-finite values, probabilities outside `[0, 1]`, or zero weights only
    /// age.
    pub fn update(&mut self, pred: f64, y: f64, lam: f64, w: f64) {
        self.age(lam);
        if !(pred.is_finite() && y.is_finite() && w > 0.0 && (0.0..=1.0).contains(&pred)) {
            return;
        }
        let a = 1.0 - 2.0 * pred;
        let dn = w * (y - pred) * a;
        let dd = (w * w) * (a * a) * pred * (1.0 - pred);
        let nn = self.num + dn;
        let nd = self.den + dd;
        let nw = self.w + w;
        if nn.is_finite() && nd.is_finite() && nw.is_finite() {
            self.num = nn;
            self.den = nd;
            self.w = nw;
        }
    }

    /// Spiegelhalter's z-statistic (`N(0, 1)` under calibrated probabilities),
    /// or `None` when the denominator is not positive.
    pub fn z(&self) -> Option<f64> {
        (self.w > 0.0 && self.den > 0.0).then(|| self.num / self.den.sqrt())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }

    #[test]
    fn arch_lm_cap_and_zero_weight_invariance() {
        assert_eq!(ArchLm::state_bytes(1), 64);
        assert_eq!(ArchLm::state_bytes(4), 208);
        assert_eq!(ArchLm::state_bytes(10), 712);
        assert_eq!(ArchLm::state_bytes(30), 4472);

        let mut st = 5201u64;
        let mut a = ArchLm::new(4);
        let mut b = ArchLm::new(4);
        let lam_step = 0.5_f64.powf(1.0 / 30.0);
        let mut carry = 1.0;
        for i in 0..150 {
            let e = lcg(&mut st) * 2.0;
            let w = if i == 0 || i % 7 == 0 { 0.0 } else { 1.0 };
            let lam = if i == 0 { 1.0 } else { lam_step };
            a.update(e, lam, w);
            if i > 0 {
                carry *= lam_step;
            }
            if w > 0.0 {
                b.update(e, carry, w);
                carry = 1.0;
            }
        }
        let za = a.lm().unwrap();
        let zb = b.lm().unwrap();
        assert!((za - zb).abs() <= 1e-12 * za.abs().max(1.0), "{za} vs {zb}");
        assert!(a.has_shape(4));
    }

    #[test]
    fn white_test_caps_and_subsets() {
        assert_eq!(WhiteTest::state_bytes(21, true), 259_088);
        assert!(WhiteTest::state_bytes(MAX_WHITE_FULL_K, true) <= SPEC_DIAG_CAP_BYTES);
        assert!(WhiteTest::state_bytes(MAX_WHITE_FULL_K + 1, true) > SPEC_DIAG_CAP_BYTES);
        assert!(WhiteTest::state_bytes(MAX_WHITE_SQUARES_K, false) <= SPEC_DIAG_CAP_BYTES);
        assert!(WhiteTest::state_bytes(MAX_WHITE_SQUARES_K + 1, false) > SPEC_DIAG_CAP_BYTES);

        let mut st = 5202u64;
        let mut full = WhiteTest::new(3, true);
        let mut sub = WhiteTest::new(2, true);
        for _ in 0..200 {
            let x = [lcg(&mut st), lcg(&mut st), lcg(&mut st)];
            let e = lcg(&mut st) * (1.0 + x[0] * x[0] + x[2] * x[2]);
            full.update(&x, e, 1.0, 1.0);
            sub.update(&[x[0], x[2]], e, 1.0, 1.0);
        }
        let lm_full_sub = full.lm(&[0, 2]).unwrap();
        let lm_sub = sub.lm(&[0, 1]).unwrap();
        assert!(
            (lm_full_sub - lm_sub).abs() <= 1e-12 * lm_sub.abs().max(1.0),
            "{lm_full_sub} vs {lm_sub}"
        );
    }

    #[test]
    fn spiegelhalter_matches_direct_formula() {
        let mut st = 5203u64;
        let mut sp = Spiegelhalter::new();
        let (mut num, mut den) = (0.0, 0.0);
        for _ in 0..200 {
            let p = 0.1 + 0.8 * ((lcg(&mut st) + 1.0) * 0.5);
            let y = if lcg(&mut st) > 0.0 { 1.0 } else { 0.0 };
            sp.update(p, y, 1.0, 1.0);
            let a = 1.0 - 2.0 * p;
            num += (y - p) * a;
            den += a * a * p * (1.0 - p);
        }
        let want = num / den.sqrt();
        let got = sp.z().unwrap();
        assert!((got - want).abs() <= 1e-13, "{got} vs {want}");
        assert!(sp.has_shape());
    }
}
