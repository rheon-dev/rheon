//! `PSquared`: Jain and Chlamtac's P² estimate of fixed quantiles of a
//! stream, five markers a quantile and no stored values (R. Jain and I.
//! Chlamtac, "The P² algorithm for dynamic calculation of quantiles and
//! histograms without storing observations", CACM 28(10), 1985, 1076-1085).
//!
//! For the level `p`, five markers carry heights `q_1..q_5` (estimates of the
//! minimum, the `p/2`, `p` and `(1+p)/2` quantiles and the maximum),
//! positions `n_1..n_5` (the rank each height is taken to have), desired
//! positions `n'_1..n'_5` and the fixed increments `dn'`:
//!
//! ```text
//! start (on the sixth value, the first five sorted):  q_i = x_(i),  n_i = i
//! n'  = [1, 1 + 2p, 1 + 4p, 3 + 2p, 5]          dn' = [0, p/2, p, (1+p)/2, 1]
//! ```
//!
//! Each value `x` from the sixth on:
//!
//! 1. The cell `k`: if `x < q_1`, `q_1 = x` and `k = 1`; else the first `k`
//!    in `1..4` with `q_k <= x < q_{k+1}`; else `q_5 = max(q_5, x)` and
//!    `k = 4`.
//! 2. `n_i += 1` for `i = k+1..5`, and `n'_i += dn'_i` for every `i`.
//! 3. For `i = 2, 3, 4` in turn, with `d = n'_i - n_i`: if `d >= 1` and
//!    `n_{i+1} - n_i > 1`, or `d <= -1` and `n_{i-1} - n_i < -1`, then with
//!    `s = sign(d)` the parabolic prediction
//!
//!    ```text
//!    q'_i = q_i + s / (n_{i+1} - n_{i-1})
//!               * [ (n_i - n_{i-1} + s) (q_{i+1} - q_i) / (n_{i+1} - n_i)
//!                 + (n_{i+1} - n_i - s) (q_i - q_{i-1}) / (n_i - n_{i-1}) ]
//!    ```
//!
//!    replaces `q_i` if `q_{i-1} < q'_i < q_{i+1}`, and otherwise the linear
//!    one `q_i + s (q_{i+s} - q_i) / (n_{i+s} - n_i)` does; then `n_i += s`.
//!
//! The estimate is `q_3`. Before the sixth value the `m` values seen are
//! kept as they came, and the estimate is the sorted values' entry at index
//! `floor(min(m - 1, m p))`, counting from 0.
//!
//! The arithmetic is river's (`river.stats.Quantile`, which runs this
//! recursion), operation for operation, so the two agree to the bit: the
//! order of the terms above, the desired positions accumulated by repeated
//! addition rather than formed from the count, the five first values sorted
//! on the sixth rather than the fifth, and the read before it. river 0.27
//! runs it in compiled code whose source is not in its wheel; the order
//! followed is that of its last pure-Python version, and the agreement is
//! measured, on every row of seeded streams, by the tests.
//!
//! **Weights.** P² has no weighted form: its positions count values. A row
//! of any positive weight counts as one value, as river's estimators and the
//! drift detectors take a row; a row of weight 0 (or a weight that is not a
//! positive number) is not seen, moves no marker and counts in no clock
//! ([`PSquared::update_weighted`]).
//!
//! P² has no error bound: the markers are a heuristic, and nothing limits
//! how far `q_3` strays from the quantile on a stream that is not
//! stationary. It never forgets a row, so it takes no span: a spec that
//! gives it a half-life, a window or another span is refused when it is
//! built ([`PSquared::check_no_span`]), naming the quantiles that forget. Its state is fixed by the levels: per
//! level the 15 numbers above (`p` aside, 128 bytes), and up to five values
//! before the sixth arrives.

use serde::{Deserialize, Serialize};

/// The P² markers of one level ([`PSquared`]): heights, positions and
/// desired positions, each 1-based in the module's equations and 0-based
/// here. Positions are whole numbers, kept as `f64` (exact below `2^53`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Markers {
    q: [f64; 5],
    n: [f64; 5],
    desired: [f64; 5],
}

impl Markers {
    /// The markers of level `p` on the five `first` values, sorted.
    fn start(first: &[f64; 5], p: f64) -> Self {
        Self {
            q: *first,
            n: [1.0, 2.0, 3.0, 4.0, 5.0],
            desired: [1.0, 1.0 + 2.0 * p, 1.0 + 4.0 * p, 3.0 + 2.0 * p, 5.0],
        }
    }

    /// The increments of the desired positions at level `p`.
    fn increments(p: f64) -> [f64; 5] {
        [0.0, p / 2.0, p, (1.0 + p) / 2.0, 1.0]
    }

    /// Steps 1 to 3 of the module's recursion for one value `x`.
    fn update(&mut self, x: f64, p: f64) {
        let q = &mut self.q;
        let k = if x < q[0] {
            q[0] = x;
            1
        } else if let Some(k) = (1..5).find(|&i| q[i - 1] <= x && x < q[i]) {
            k
        } else {
            if q[4] < x {
                q[4] = x;
            }
            4
        };
        for n in &mut self.n[k..] {
            *n += 1.0;
        }
        let dn = Self::increments(p);
        for (d, inc) in self.desired.iter_mut().zip(dn) {
            *d += inc;
        }
        let n = &mut self.n;
        for i in 1..4 {
            let d = self.desired[i] - n[i];
            if (d >= 1.0 && n[i + 1] - n[i] > 1.0) || (d <= -1.0 && n[i - 1] - n[i] < -1.0) {
                let s = if d >= 0.0 { 1.0 } else { -1.0 };
                let (ni, nm, np) = (n[i], n[i - 1], n[i + 1]);
                let outer = s / (np - nm);
                let left = (ni - nm + s) * (q[i + 1] - q[i]) / (np - ni);
                let right = (np - ni - s) * (q[i] - q[i - 1]) / (ni - nm);
                let parabolic = q[i] + outer * (left + right);
                if q[i - 1] < parabolic && parabolic < q[i + 1] {
                    q[i] = parabolic;
                } else {
                    let j = if s > 0.0 { i + 1 } else { i - 1 };
                    q[i] += s * (q[j] - q[i]) / (n[j] - ni);
                }
                n[i] += s;
            }
        }
    }
}

/// Jain and Chlamtac's P² estimate of a stream's quantile at each of a set
/// of levels: five markers a level, no values stored (the module's
/// recursion, to the bit river's).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PSquared {
    /// The levels reported, each in (0, 1).
    levels: Vec<f64>,
    /// The values seen, as they came, until a sixth arrives; then empty.
    first: Vec<f64>,
    /// Per level, its markers from the sixth value on; empty before.
    markers: Vec<Markers>,
}

impl PSquared {
    pub fn new(levels: &[f64]) -> Result<Self, String> {
        if let Some(bad) = levels.iter().find(|p| !(**p > 0.0 && **p < 1.0)) {
            return Err(format!(
                "PSquared: every level must be above 0 and below 1, but one of them is {bad}. \
                 Pass levels above 0 and below 1."
            ));
        }
        Ok(Self {
            levels: levels.to_vec(),
            first: Vec::with_capacity(5),
            markers: Vec::new(),
        })
    }

    pub fn levels(&self) -> &[f64] {
        &self.levels
    }

    /// The values seen.
    pub fn count(&self) -> f64 {
        self.markers
            .first()
            .map_or(self.first.len() as f64, |m| m.n[4])
    }

    /// One value `x`; a value that is not finite is not seen.
    pub fn update(&mut self, x: f64) {
        if !x.is_finite() {
            return;
        }
        if self.markers.is_empty() {
            if self.first.len() < 5 {
                self.first.push(x);
                return;
            }
            let mut first = [0.0; 5];
            first.copy_from_slice(&self.first);
            sort(&mut first);
            self.markers = self
                .levels
                .iter()
                .map(|&p| Markers::start(&first, p))
                .collect();
            self.first = Vec::new();
        }
        for (m, &p) in self.markers.iter_mut().zip(&self.levels) {
            m.update(x, p);
        }
    }

    /// One row `x` of weight `w`: any positive weight counts as one value
    /// (P² counts values and has no weighted form), and a weight of 0, or
    /// one that is not a positive number, leaves the estimate as it was.
    pub fn update_weighted(&mut self, x: f64, w: f64) {
        if w > 0.0 {
            self.update(x);
        }
    }

    /// Refuse a span given to P², which never forgets a row: `given` names
    /// the span's field as the spec wrote it (`half_life`, `window_size`, a
    /// rows-or-clock span), or `None` when the spec gives none.
    pub fn check_no_span(given: Option<&str>) -> Result<(), String> {
        match given {
            None => Ok(()),
            Some(field) => Err(format!(
                "PSquared: {field} is not allowed. P-squared never forgets a row, so it takes no \
                 span, half-life or window. For a quantile that forgets, use the decayed form \
                 (a half_life) or the windowed form (a window) instead."
            )),
        }
    }

    /// The estimate at level `l`, or `None` before any value.
    pub fn get(&self, l: usize) -> Option<f64> {
        if let Some(m) = self.markers.get(l) {
            return Some(m.q[2]);
        }
        if self.first.is_empty() {
            return None;
        }
        let mut seen = self.first.clone();
        sort(&mut seen);
        let m = seen.len() as f64;
        // Truncated, as the index is at least 0.
        let at = (m - 1.0).min(m * self.levels[l]) as usize;
        Some(seen[at])
    }

    /// Back to no values, at the same levels.
    pub fn reset(&mut self) {
        self.first = Vec::with_capacity(5);
        self.markers = Vec::new();
    }

    /// Whether a restored state has the shape [`Self::new`] gives it: valid
    /// levels, at most five first values, and markers for every level or
    /// none, and not both.
    pub fn has_shape(&self) -> bool {
        self.levels.iter().all(|p| *p > 0.0 && *p < 1.0)
            && self.first.len() <= 5
            && (self.markers.is_empty()
                || (self.markers.len() == self.levels.len() && self.first.is_empty()))
    }

    /// The bytes this estimate takes, its fields and what each vector has
    /// allocated.
    #[cfg(test)]
    pub(crate) fn bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + (self.levels.capacity() + self.first.capacity()) * std::mem::size_of::<f64>()
            + self.markers.capacity() * std::mem::size_of::<Markers>()
    }
}

/// Ascending and stable, `-0.0` equal to `0.0` (Python's sort of floats, as
/// river's pure-Python version sorts its first values); every value is
/// finite.
fn sort(v: &mut [f64]) {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
}

#[cfg(test)]
mod tests;
