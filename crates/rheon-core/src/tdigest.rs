//! `TDigest`: Dunning and Ertl's merging t-digest at a compression `delta`
//! (T. Dunning and O. Ertl, "Computing extremely accurate quantiles using
//! t-digests", 2019), for a quantile whose error is uniform in rank, where a
//! sketch of relative accuracy misreads (values near 0, of both signs).
//!
//! The scale function is `k_1`:
//!
//! ```text
//! k(q) = delta / (2 pi) * asin(2q - 1),    q in [0, 1]
//! ```
//!
//! which spans `delta / 2` over `q`. A centroid (mean `c`, weight `w`)
//! covers the mass from `W` (the weight before it) to `W + w`, and may span
//! at most 1 in `k`. State: the centroids ascending by mean, the minimum and
//! maximum seen, the total weight, and a buffer of rows not yet merged.
//!
//! **Merge pass**, run when the buffer holds `8 delta` rows, and on a copy at
//! a read: the old centroids and the buffered rows (each a centroid of its
//! own weight) sorted by mean (stable, centroids first on a tie); `N` their
//! total weight. Walk them in order with the running centroid `(c, w)` and
//! the weight `W` before it, and the limit `q_lim = k^-1(k(W/N) + 1)`, taken
//! as 1 where `k(W/N) + 1` is past `delta / 4`. The next `(c', w')`:
//!
//! ```text
//! (W + w + w') / N <= q_lim:  w <- w + w',  c <- c + (c' - c) w' / w
//! otherwise:                  emit (c, w),  W <- W + w,  (c, w) <- (c', w')
//! ```
//!
//! **The limit without a math library.** With `a = asin(2q - 1)` and
//! `b = 2 pi / delta`, `k^-1(k(q) + 1) = (1 + sin(a + b)) / 2`, and
//! `sin(a + b) = sin a cos b + cos a sin b` with `sin a = 2q - 1` and
//! `cos a = 2 sqrt(q (1 - q))`; the limit is 1 once `a + b >= pi / 2`, that
//! is once `2q - 1 >= cos b`. `sin b` and `cos b` are fixed by `delta` and
//! summed from their Taylor series, and `sqrt` is correctly rounded, so the
//! centroids are the same bits on every OS: no libm result decides a merge.
//!
//! **Quantile** at level `p`, `t = p N`: centroid `i`'s centre is
//! `C_i = sum_{j<i} w_j + w_i / 2`; for `C_i <= t < C_{i+1}` the read is
//! `c_i + (t - C_i) / (C_{i+1} - C_i) (c_{i+1} - c_i)`. Below the first
//! centre it interpolates from the minimum (at rank 0) to `c_1`, above the
//! last from `c_n` to the maximum (at rank `N`); one centroid reads its
//! mean. A read merges a copy of the buffer and leaves the state as it was.
//!
//! **Size.** Every pair of neighbours spans more than 1 in `k` together, and
//! `k` spans `delta / 2`, so at most `delta + 1` centroids remain after a merge
//! (between about `delta / 2` and `delta` in practice): 16 bytes each, plus
//! the buffer's `8 delta` rows at 16 bytes and a fixed header. The bound is
//! the compression's, never the stream's length ([`TDigest::max_bytes`]).
//!
//! **Rows.** A row of positive weight `w` is a centroid of weight `w`; a row
//! of weight 0 (or one that is not a positive number) is not seen and counts
//! in no clock, nor is a value that is not finite. The digest never forgets.
//!
//! **Error.** The `k_1` rule bounds each centroid's mass near the tails by
//! about `q (1 - q)` times a constant over `delta`, so the rank error
//! `|F(q_hat) - p|` is smallest at the tails and largest at the median. At
//! `N` = 100,000 on normal, lognormal, Student-t(3) and Pareto(2.5) streams
//! it was measured at most 2.4e-3 at `delta` = 100 and 1.2e-3 at 200, and the
//! tests hold it under 3.8e-3 and 1.8e-3.

use serde::{Deserialize, Serialize};

/// The smallest compression taken: under it the limit's series and the
/// digest's few centroids are no longer worth a sketch.
pub const MIN_COMPRESSION: f64 = 10.0;

/// The largest compression taken: a digest of 100,000 centroids, 14 MB with
/// its buffer, past which an exact quantile over the rows costs less.
pub const MAX_COMPRESSION: f64 = 100_000.0;

/// A merging t-digest at one compression (the module's equations).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TDigest {
    /// `delta`, in [`MIN_COMPRESSION`, `MAX_COMPRESSION`].
    compression: f64,
    /// `sin(2 pi / delta)` and `cos(2 pi / delta)`, from `delta` alone.
    sin_b: f64,
    cos_b: f64,
    /// The centroids `(mean, weight)`, ascending by mean.
    centroids: Vec<(f64, f64)>,
    /// The rows `(value, weight)` not yet merged, fewer than `8 delta`.
    buffer: Vec<(f64, f64)>,
    /// The smallest and largest value seen, `+inf` and `-inf` before any.
    lo: f64,
    hi: f64,
}

/// `sin x` and `cos x` from their Taylor series, `0 < x <= 2 pi / 10`: the
/// terms fall by at least `x^2 / 2 < 0.2` each, so 30 of them pass the last
/// bit; plain arithmetic, the same bits on every OS.
fn sin_cos(x: f64) -> (f64, f64) {
    let (mut s, mut c) = (0.0, 0.0);
    let mut term_s = x;
    let mut term_c = 1.0;
    let x2 = x * x;
    for n in 0..30 {
        s += term_s;
        c += term_c;
        let k = 2.0 * n as f64;
        term_s *= -x2 / ((k + 2.0) * (k + 3.0));
        term_c *= -x2 / ((k + 1.0) * (k + 2.0));
    }
    (s, c)
}

impl TDigest {
    pub fn new(compression: f64) -> Result<Self, String> {
        if !(MIN_COMPRESSION..=MAX_COMPRESSION).contains(&compression) {
            return Err(format!(
                "TDigest: compression must be at least {MIN_COMPRESSION} and at most \
                 {MAX_COMPRESSION}, but it is {compression}. Pass 100 for a rank error near \
                 2.5e-3, or 200 for near 1.2e-3 (twice the state)."
            ));
        }
        let (sin_b, cos_b) = sin_cos(std::f64::consts::TAU / compression);
        Ok(Self {
            compression,
            sin_b,
            cos_b,
            centroids: Vec::new(),
            buffer: Vec::new(),
            lo: f64::INFINITY,
            hi: f64::NEG_INFINITY,
        })
    }

    pub fn compression(&self) -> f64 {
        self.compression
    }

    /// The rows the buffer holds before a merge: `8 delta`.
    fn buffer_cap(&self) -> usize {
        (8.0 * self.compression) as usize
    }

    /// The most bytes a digest at `compression` can hold: the header,
    /// `delta + 1` centroids and a full buffer, 16 bytes a pair. Checked against
    /// a spec's `max_state_bytes` when the spec is built.
    pub fn max_bytes(compression: f64) -> u64 {
        let pairs = compression.floor() as u64 + 1 + (8.0 * compression) as u64;
        std::mem::size_of::<Self>() as u64 + 16 * pairs
    }

    /// Refuse a compression whose fullest state is past `cap` bytes.
    pub fn check_cap(compression: f64, cap: u64) -> Result<(), String> {
        let most = Self::max_bytes(compression);
        if most > cap {
            return Err(format!(
                "TDigest: at compression {compression} a group's digest may hold {most} bytes, \
                 past max_state_bytes {cap}. Lower the compression or raise max_state_bytes."
            ));
        }
        Ok(())
    }

    /// Refuse a span: `given` names the field a spec set (a half-life, a
    /// window), `None` when it set none. A merging digest folds a row into
    /// a centroid's mean and cannot take it out again, so it has no decayed
    /// or windowed form: run once at spec build, before any row.
    pub fn check_no_span(given: Option<&str>) -> Result<(), String> {
        match given {
            None => Ok(()),
            Some(field) => Err(format!(
                "TDigest: {field} is not allowed. A merging t-digest folds each row into a \
                 centroid's mean and cannot take it out again, so it never forgets a row and \
                 takes no span, half-life or window. For a quantile that forgets, use the \
                 windowed DDSketch (ewm_quantile with a window_size), whose buckets drop a row \
                 as it leaves the window."
            )),
        }
    }

    /// The total weight seen.
    pub fn weight(&self) -> f64 {
        self.centroids.iter().chain(&self.buffer).map(|c| c.1).sum()
    }

    /// The centroids as last merged (the buffer not folded in).
    pub fn centroids(&self) -> &[(f64, f64)] {
        &self.centroids
    }

    /// One row `x` of weight `w`; a weight of 0, one that is not a positive
    /// finite number, or a value that is not finite, is not seen.
    pub fn update(&mut self, x: f64, w: f64) {
        if !(x.is_finite() && w > 0.0 && w.is_finite()) {
            return;
        }
        self.lo = self.lo.min(x);
        self.hi = self.hi.max(x);
        if self.buffer.capacity() == 0 {
            self.buffer.reserve_exact(self.buffer_cap());
        }
        self.buffer.push((x, w));
        if self.buffer.len() >= self.buffer_cap() {
            let mut buffer = std::mem::take(&mut self.buffer);
            self.centroids = self.merged(&mut buffer);
            buffer.clear();
            self.buffer = buffer;
        }
    }

    /// `k^-1(k(q) + 1)`, as the module computes it.
    fn limit(&self, q: f64) -> f64 {
        let s = 2.0 * q - 1.0;
        if s >= self.cos_b {
            return 1.0;
        }
        let c = 2.0 * (q * (1.0 - q)).max(0.0).sqrt();
        (1.0 + s * self.cos_b + c * self.sin_b) / 2.0
    }

    /// The merge pass over the centroids and `fresh`, which it sorts.
    fn merged(&self, fresh: &mut [(f64, f64)]) -> Vec<(f64, f64)> {
        fresh.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut all = Vec::with_capacity(self.centroids.len() + fresh.len());
        let (mut i, mut j) = (0, 0);
        while i < self.centroids.len() && j < fresh.len() {
            if self.centroids[i].0.total_cmp(&fresh[j].0).is_le() {
                all.push(self.centroids[i]);
                i += 1;
            } else {
                all.push(fresh[j]);
                j += 1;
            }
        }
        all.extend_from_slice(&self.centroids[i..]);
        all.extend_from_slice(&fresh[j..]);
        let Some((&(mut c, mut w), rest)) = all.split_first() else {
            return Vec::new();
        };
        let total: f64 = all.iter().map(|x| x.1).sum();
        let mut out = Vec::with_capacity(self.compression as usize + 1);
        let mut before = 0.0;
        let mut q_lim = self.limit(0.0);
        for &(ci, wi) in rest {
            if (before + w + wi) / total <= q_lim {
                w += wi;
                c += (ci - c) * wi / w;
            } else {
                out.push((c, w));
                before += w;
                q_lim = self.limit(before / total);
                (c, w) = (ci, wi);
            }
        }
        out.push((c, w));
        out
    }

    /// The centroids with the buffer merged in, the state untouched.
    pub fn settled(&self) -> Vec<(f64, f64)> {
        if self.buffer.is_empty() {
            return self.centroids.clone();
        }
        self.merged(&mut self.buffer.clone())
    }

    /// The estimate at level `p` in [0, 1], or `None` before any row.
    pub fn quantile(&self, p: f64) -> Option<f64> {
        Self::read(&self.settled(), self.lo, self.hi, p)
    }

    /// The estimate at each of `levels`, merging the buffer's copy once.
    pub fn quantiles(&self, levels: &[f64]) -> Vec<Option<f64>> {
        let settled = self.settled();
        levels
            .iter()
            .map(|&p| Self::read(&settled, self.lo, self.hi, p))
            .collect()
    }

    /// The module's read at `p` from `centroids`, `lo` and `hi`.
    fn read(centroids: &[(f64, f64)], lo: f64, hi: f64, p: f64) -> Option<f64> {
        let (&(first, w_first), &(last, w_last)) = (centroids.first()?, centroids.last()?);
        if centroids.len() == 1 {
            return Some(first);
        }
        let total: f64 = centroids.iter().map(|c| c.1).sum();
        let t = p * total;
        if t < w_first / 2.0 {
            return Some(lo + (first - lo) * t / (w_first / 2.0));
        }
        if t > total - w_last / 2.0 {
            let r = total - t;
            return Some(hi - (hi - last) * r / (w_last / 2.0));
        }
        let mut centre = w_first / 2.0;
        for pair in centroids.windows(2) {
            let ((c0, w0), (c1, w1)) = (pair[0], pair[1]);
            let next = centre + w0 / 2.0 + w1 / 2.0;
            if t < next {
                let f = (t - centre) / (next - centre);
                return Some(c0 + f * (c1 - c0));
            }
            centre = next;
        }
        Some(last)
    }

    /// Back to no rows, at the same compression.
    pub fn reset(&mut self) {
        self.centroids = Vec::new();
        self.buffer = Vec::new();
        self.lo = f64::INFINITY;
        self.hi = f64::NEG_INFINITY;
    }

    /// Whether a restored state has the shape [`Self::new`] and
    /// [`Self::update`] give it: a compression in range with its own sine
    /// and cosine, at most `delta + 1` centroids ascending with positive finite
    /// weights, a buffer short of full, and every value within the extremes.
    pub fn has_shape(&self) -> bool {
        let Ok(fresh) = Self::new(self.compression) else {
            return false;
        };
        let within = |v: f64| v >= self.lo && v <= self.hi;
        fresh.sin_b.to_bits() == self.sin_b.to_bits()
            && fresh.cos_b.to_bits() == self.cos_b.to_bits()
            && self.centroids.len() as f64 <= self.compression.floor() + 1.0
            && self.buffer.len() < self.buffer_cap()
            && self.centroids.windows(2).all(|p| p[0].0 <= p[1].0)
            && self
                .centroids
                .iter()
                .chain(&self.buffer)
                .all(|&(v, w)| within(v) && w > 0.0 && w.is_finite())
    }

    /// The bytes this digest takes: its fields and what each vector has
    /// allocated.
    pub fn bytes(&self) -> usize {
        std::mem::size_of::<Self>() + (self.centroids.capacity() + self.buffer.capacity()) * 16
    }
}

#[cfg(test)]
mod tests;
