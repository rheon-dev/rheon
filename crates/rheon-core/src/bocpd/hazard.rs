//! Run-length-dependent negative-binomial hazard table for [`super::Bocpd`].
//!
//! For segment length `L ~ NegBinomial(k, p)` on `{k, k + 1, …}`:
//!
//! ```text
//! P(L = t)   = C(t − 1, k − 1) · p^k · (1 − p)^(t − k),     t ≥ k
//! P(L ≥ t)   = Σ_{j=0}^{k-1} C(t − 1, j) · p^j · (1 − p)^(t − 1 − j)
//! H(r)       = P(L = r + 1) / P(L ≥ r + 1)
//! ```
//!
//! Dividing numerator and denominator by the `j = k − 1` term reduces the
//! hazard to a finite `k`-term ratio with no binomial-coefficient overflow at
//! any run length `r`:
//!
//! ```text
//! H(r) = p / (1 + R),     R = Σ_{m=1}^{k-1} c_m,
//! c_0  = 1,               c_m = c_{m-1} · ((k − m) / (t − k + m)) · ((1 − p) / p)
//! ```
//!
//! When `r + 1 < k`, `H(r) = 0` (`ln H = −∞`, `ln(1 − H) = 0`). At `k = 1`,
//! `R = 0` and `H(r) = p` for every `r`. The table of `max_run` pairs
//! `(ln H(r), ln(1 − H(r)))` is derived from `(k, p)` and never serialized.

use super::Branches;

/// Derived lookup table `H(r)` for `r = 0 .. max_run`, rebuilt on construction
/// and load (`#[serde(skip)]`) and grown lazily so a large `max_run` (such as
/// `10^9`) never allocates unused slots.
#[derive(Debug, Clone, Default)]
pub(super) struct NbHazardTable {
    k: usize,
    p: f64,
    max_run: usize,
    entries: Vec<Branches>,
}

impl PartialEq for NbHazardTable {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

impl NbHazardTable {
    /// Build a table for `NegBinomial(k, p)` capped at `max_run`, precomputing
    /// up to `init_len` entries immediately.
    pub(super) fn build(k: usize, p: f64, max_run: usize, init_len: usize) -> Self {
        let pre = max_run.min(init_len.max(64));
        let mut entries = Vec::with_capacity(pre);
        for r in 0..pre {
            entries.push(nb_branches(k, p, r));
        }
        Self {
            k,
            p,
            max_run,
            entries,
        }
    }

    /// Ensure entries up through run length `max_r` (clamped at `max_run - 1`)
    /// are cached for O(1) lookup during a row's recursion.
    #[inline]
    pub(super) fn ensure_len(&mut self, max_r: usize) {
        if self.k == 0 {
            return;
        }
        let target = max_r.saturating_add(1).min(self.max_run);
        if self.entries.len() < target {
            self.entries.reserve(target - self.entries.len());
            for r in self.entries.len()..target {
                self.entries.push(nb_branches(self.k, self.p, r));
            }
        }
    }

    /// The branches for run length `r` (clamped at `max_run - 1`).
    #[inline]
    pub(super) fn at(&self, r: usize) -> Branches {
        let idx = r.min(self.max_run.saturating_sub(1));
        if let Some(&b) = self.entries.get(idx) {
            b
        } else {
            nb_branches(self.k, self.p, idx)
        }
    }
}

/// Exact `(ln H(r), ln(1 − H(r)))` for `NegBinomial(k, p)` at run length `r`
/// (`t = r + 1`).
pub(crate) fn nb_branches(k: usize, p: f64, r: usize) -> Branches {
    let t = r + 1;
    if t < k {
        return Branches::NONE;
    }
    if p >= 1.0 {
        return Branches {
            ln_break: 0.0,
            ln_grow: f64::NEG_INFINITY,
        };
    }
    if k == 1 {
        return Branches {
            ln_break: p.ln(),
            ln_grow: (-p).ln_1p(),
        };
    }
    let q_over_p = (1.0 - p) / p;
    // Try direct ratio accumulation first; if any term exceeds 1e150 (only
    // possible for large `k` and tiny `p`), fall back to log-space summation.
    let mut c = 1.0f64;
    let mut r_sum = 0.0f64;
    let mut overflow = false;
    for m in 1..k {
        let factor = ((k - m) as f64 / (t - k + m) as f64) * q_over_p;
        c *= factor;
        if !(c.is_finite() && c < 1e150) {
            overflow = true;
            break;
        }
        r_sum += c;
        if !r_sum.is_finite() {
            overflow = true;
            break;
        }
    }
    if !overflow {
        let ln_break = p.ln() - r_sum.ln_1p();
        let h = p / (1.0 + r_sum);
        let ln_grow = (-h).ln_1p();
        return Branches { ln_break, ln_grow };
    }
    // Log-space fallback: compute `ln(1 + R) = log_sum_exp(0, ln c_1, ...)`.
    let ln_qp = q_over_p.ln();
    let mut ln_c = 0.0f64;
    let mut ln_terms = Vec::with_capacity(k);
    ln_terms.push(0.0);
    for m in 1..k {
        ln_c += ((k - m) as f64).ln() - ((t - k + m) as f64).ln() + ln_qp;
        ln_terms.push(ln_c);
    }
    let ln_denom = super::log_sum_exp(&ln_terms);
    let ln_break = p.ln() - ln_denom;
    let h = ln_break.exp();
    let ln_grow = (-h).ln_1p();
    Branches { ln_break, ln_grow }
}

/// Scalar hazard probability `H(r)` for tests and verification.
pub fn neg_binomial_hazard(k: usize, p: f64, r: usize) -> f64 {
    let b = nb_branches(k, p, r);
    if b.ln_break.is_infinite() && b.ln_break.is_sign_negative() {
        0.0
    } else {
        b.ln_break.exp()
    }
}
