//! `DdSketch`: Masson, Rim and Lee's DDSketch (C. Masson, J. E. Rim and
//! H. K. Lee, "DDSketch: a fast and fully-mergeable quantile sketch with
//! relative-error guarantees", PVLDB 12(12), 2019), with the logarithmic
//! mapping of the `ddsketch` package (Datadog, Apache-2.0) and dense stores,
//! for a quantile and a tail mean within a relative accuracy `alpha` of the
//! values themselves.
//!
//! **The mapping.** With `g = 2 alpha / (1 - alpha)` and
//! `gamma = 1 + g = (1 + alpha) / (1 - alpha)`, the multiplier is
//! `mu = ln 2 / ln(1 + g)` and a value `x > 0` goes to the bucket
//!
//! ```text
//! key(x)   = ceil(log2(x) mu)                 covering (gamma^(k-1), gamma^k]
//! value(k) = 2^(k / mu) 2 / (1 + gamma)       its representative
//! ```
//!
//! `value(k) / gamma^k = 1 - alpha` and `value(k) / gamma^(k-1) = 1 + alpha`,
//! so `|value(key(x)) - x| <= alpha x` for every `x` in the bucket. A value
//! below `-m` goes to a second store by `key(-x)`, read as `-value(k)`, and
//! one within `m = f64::MIN_POSITIVE gamma` of 0 to a count of its own, read
//! as 0: so the buckets sort in value order. These are the package's rules,
//! its `LogarithmicMapping` with no offset and its `DDSketch.add`.
//!
//! **No math library decides a bucket.** `log2`, `ln(1 + g)` and `2^y` are
//! computed here from their series in plain arithmetic (`ln m = 2 atanh s`
//! with `s = (m - 1)/(m + 1)` on a mantissa in `[sqrt(1/2), sqrt(2))`, and
//! `2^f = exp(f ln 2)` on `f` in `[0, 1)`), each within a few ulps of the
//! correctly rounded result (4 at most, measured against the platform's), so a bucket's key and its representative are
//! the same bits on every OS. A key differs from the package's only for a
//! value within a few ulps of a bucket's edge, where the package's own key
//! depends on the platform's `log`.
//!
//! **Weights.** A row of positive weight `w` adds `w` to its bucket; a row of
//! weight 0, or one that is not a positive finite number, or a value that
//! is not finite, is not seen. A windowed sketch takes a row out again by
//! subtracting its weight from its bucket and counting the bucket's rows,
//! so a bucket whose rows have all left holds exactly 0 and is dropped from
//! the ends.
//!
//! **Size.** Each store keeps its live buckets alone, sparse: a 32-bit key
//! and an `f64` weight each (and a `u32` row count under a window), 12 or
//! 16 bytes. The buckets are the data's distinct ones, at most its
//! log-range over `ln gamma` (about 1,470 a store for values spanning
//! `10^-5` to `10^5` at the default `alpha`), never the stream's length; the
//! range finite doubles span is about `744 / alpha` keys a store, so the
//! bound is the data's, checked as the rows go in.

use serde::{Deserialize, Serialize};

/// The default relative accuracy, `tanh(1/128)`: `gamma = e^(1/64)`, 64
/// buckets to a factor of `e`, so `|value(key(x)) - x| <= 0.78% x`.
pub const DDSKETCH_ALPHA: f64 = 0.007_812_341_060_101_111;

/// The smallest `alpha` taken: below it a key past `2^29` in magnitude
/// would not fit the 32 bits a window's ring keeps a key in.
pub const DDSKETCH_MIN_ALPHA: f64 = 1e-6;

const SQRT_HALF: f64 = std::f64::consts::FRAC_1_SQRT_2;

/// `ln m` for `m` in `[sqrt(1/2), sqrt(2)]`: `2 atanh s`, `s = (m-1)/(m+1)`,
/// `|s| <= 0.1716`, so `s^2 <= 0.0295` and 15 terms pass the last bit.
fn ln_near_one(m: f64) -> f64 {
    let s = (m - 1.0) / (m + 1.0);
    let s2 = s * s;
    let mut term = s;
    let mut sum = 0.0;
    let mut k = 1.0;
    for _ in 0..15 {
        sum += term / k;
        term *= s2;
        k += 2.0;
    }
    2.0 * sum
}

/// `x = m 2^e` with `m` in `[sqrt(1/2), sqrt(2))`, for a positive finite
/// `x`, subnormal ones included.
fn split(x: f64) -> (f64, i32) {
    let (x, shift) = if x < f64::MIN_POSITIVE {
        (x * 18_014_398_509_481_984.0, -54) // 2^54
    } else {
        (x, 0)
    };
    let bits = x.to_bits();
    let e = ((bits >> 52) & 0x7ff) as i32 - 1023;
    let m = f64::from_bits((bits & 0x000f_ffff_ffff_ffff) | 0x3ff0_0000_0000_0000);
    if m >= std::f64::consts::SQRT_2 {
        (m / 2.0, e + 1 + shift)
    } else {
        (m, e + shift)
    }
}

/// `log2 x` for a positive finite `x`, in plain arithmetic.
pub fn log2(x: f64) -> f64 {
    let (m, e) = split(x);
    debug_assert!(m >= SQRT_HALF);
    f64::from(e) + ln_near_one(m) * std::f64::consts::LOG2_E
}

/// `ln(1 + g)` for `g > 0`, in plain arithmetic.
pub fn ln_1p(g: f64) -> f64 {
    if g <= 0.4 {
        // `2 atanh(g / (2 + g))`, the argument at most 1/6, without the
        // rounding of forming `1 + g`.
        let s = g / (2.0 + g);
        let s2 = s * s;
        let mut term = s;
        let mut sum = 0.0;
        let mut k = 1.0;
        for _ in 0..20 {
            sum += term / k;
            term *= s2;
            k += 2.0;
        }
        2.0 * sum
    } else {
        log2(1.0 + g) * std::f64::consts::LN_2
    }
}

/// `2^y` for a finite `y`, in plain arithmetic: `2^f` from the series of
/// `exp(f ln 2)`, `f` in `[0, 1)`, then scaled by `2^n` exactly (in two
/// steps where `2^n` alone would leave the doubles).
pub fn exp2(y: f64) -> f64 {
    let n = y.floor();
    let t = (y - n) * std::f64::consts::LN_2;
    // Horner's form, from the 22nd term in (`0.7^22 / 22!` is under 1e-24).
    let mut sum = 1.0;
    for k in (1..=22).rev() {
        sum = 1.0 + sum * t / f64::from(k);
    }
    let pow = |e: i32| f64::from_bits(((e + 1023) as u64) << 52);
    let n = n.clamp(-2_100.0, 2_100.0) as i32;
    match n {
        -1022..=1023 => sum * pow(n),
        _ if n > 1023 => sum * pow(1023) * pow((n - 1023).min(1023)),
        _ => sum * pow(-1022) * pow((n + 1022).max(-1022)),
    }
}

/// The logarithmic mapping of a relative accuracy (the module's).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Mapping {
    alpha: f64,
    gamma: f64,
    /// `ln 2 / ln gamma`.
    multiplier: f64,
    /// `f64::MIN_POSITIVE gamma`: a magnitude at or below it counts as 0.
    min_possible: f64,
}

impl Mapping {
    pub fn new(alpha: f64) -> Result<Self, String> {
        if !(DDSKETCH_MIN_ALPHA..1.0).contains(&alpha) {
            return Err(format!(
                "DdSketch: alpha must be at least {DDSKETCH_MIN_ALPHA} and below 1, but it is \
                 {alpha}. alpha is the relative accuracy, so 0.01 reads every quantile within \
                 1% of a value in its bucket. Give a value in that range, or leave alpha out \
                 for tanh(1/128), about 0.0078."
            ));
        }
        let g = 2.0 * alpha / (1.0 - alpha);
        let gamma = 1.0 + g;
        Ok(Self {
            alpha,
            gamma,
            multiplier: std::f64::consts::LN_2 / ln_1p(g),
            min_possible: f64::MIN_POSITIVE * gamma,
        })
    }

    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    pub fn gamma(&self) -> f64 {
        self.gamma
    }

    /// The bucket of a magnitude above [`Self::min_possible`].
    pub fn key(&self, x: f64) -> i32 {
        (log2(x) * self.multiplier).ceil() as i32
    }

    /// The representative of bucket `k`.
    pub fn value(&self, k: i32) -> f64 {
        exp2(f64::from(k) / self.multiplier) * (2.0 / (1.0 + self.gamma))
    }

    pub fn min_possible(&self) -> f64 {
        self.min_possible
    }

    /// The representative a sketch reads for a finite value `x`: its
    /// bucket's, signed, or 0 within [`Self::min_possible`] of 0.
    pub fn value_of(&self, x: f64) -> f64 {
        match Slot::of(self, x) {
            Slot::Pos(k) => self.value(k),
            Slot::Neg(k) => -self.value(k),
            Slot::Zero => 0.0,
        }
    }

    /// Whether a restored mapping is the one its `alpha` gives.
    pub fn is_own(&self) -> bool {
        Self::new(self.alpha).is_ok_and(|m| {
            m.gamma.to_bits() == self.gamma.to_bits()
                && m.multiplier.to_bits() == self.multiplier.to_bits()
                && m.min_possible.to_bits() == self.min_possible.to_bits()
        })
    }
}

/// One sign's live buckets, ascending by key: a key, a weight and, under a
/// window, a count of rows each. A bucket whose rows have all left is
/// removed, so the store holds only the buckets the rows occupy: sparse, as
/// the data's distinct buckets, not the range between its smallest and
/// largest magnitude (two values `1e-80` apart in scale are two buckets).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Store {
    keys: Vec<i32>,
    w: Vec<f64>,
    /// The rows a bucket holds: under a window only, else empty.
    n: Vec<u32>,
    /// Each bucket's representative, `Mapping::value` of its key: worked
    /// out once, as the bucket is made, so a read walks no series. Not
    /// state: [`DdSketch::refill`] works them out again after a load.
    #[serde(skip)]
    reps: Vec<f64>,
}

impl Store {
    /// The live buckets held.
    pub fn len(&self) -> usize {
        self.keys.len()
    }

    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    /// The bytes a bucket takes: its key and weight, and a count under a
    /// window.
    pub fn bucket_bytes(counted: bool) -> u64 {
        if counted { 16 } else { 12 }
    }

    /// The bytes the buckets take.
    pub fn bytes(&self, counted: bool) -> u64 {
        self.keys.len() as u64 * Self::bucket_bytes(counted)
    }

    /// The buckets the store would hold with key `k` in it.
    pub fn len_with(&self, k: i32) -> usize {
        self.keys.len() + usize::from(self.keys.binary_search(&k).is_err())
    }

    /// Add `w` to bucket `k`, making it where it is not held.
    pub fn add(&mut self, map: &Mapping, k: i32, w: f64, counted: bool) {
        let i = match self.keys.binary_search(&k) {
            Ok(i) => i,
            Err(i) => {
                self.keys.insert(i, k);
                self.reps.insert(i, map.value(k));
                self.w.insert(i, 0.0);
                if counted {
                    self.n.insert(i, 0);
                }
                i
            }
        };
        self.w[i] += w;
        if counted {
            self.n[i] += 1;
        }
    }

    /// Take a row of weight `w` out of bucket `k` (a windowed store): the
    /// last row out removes the bucket, so it holds exactly nothing.
    pub fn remove(&mut self, k: i32, w: f64) {
        let Ok(i) = self.keys.binary_search(&k) else {
            return;
        };
        self.n[i] -= 1;
        if self.n[i] == 0 {
            self.keys.remove(i);
            self.reps.remove(i);
            self.w.remove(i);
            self.n.remove(i);
        } else {
            // The rest of the bucket's rows keep a positive weight; a
            // subtraction's rounding could leave a hair under 0, which is 0.
            self.w[i] = (self.w[i] - w).max(0.0);
        }
    }

    /// The buckets as `(key, weight)`, ascending by key, empty ones skipped.
    pub fn buckets(&self) -> impl DoubleEndedIterator<Item = (i32, f64)> + '_ {
        self.keys
            .iter()
            .zip(&self.w)
            .filter(|(_, w)| **w > 0.0)
            .map(|(&k, &w)| (k, w))
    }

    /// The buckets as `(representative, weight)`, ascending by key, empty
    /// ones skipped.
    fn reps_and_weights(&self) -> impl DoubleEndedIterator<Item = (f64, f64)> + '_ {
        debug_assert_eq!(self.reps.len(), self.keys.len());
        self.reps
            .iter()
            .zip(&self.w)
            .filter(|(_, w)| **w > 0.0)
            .map(|(&r, &w)| (r, w))
    }

    /// Every bucket's weight times `f` in `(0, 1)`, one rounding each; a
    /// bucket whose weight underflows to 0 is removed (a store without a
    /// window, which holds no row counts).
    fn scale(&mut self, f: f64) {
        for w in &mut self.w {
            *w *= f;
        }
        if self.w.contains(&0.0) {
            let mut i = 0;
            while i < self.w.len() {
                if self.w[i] == 0.0 {
                    self.keys.remove(i);
                    self.w.remove(i);
                    self.reps.remove(i);
                } else {
                    i += 1;
                }
            }
        }
    }

    /// Work each bucket's representative out again, as after a load.
    fn refill(&mut self, map: &Mapping) {
        self.reps = self.keys.iter().map(|&k| map.value(k)).collect();
    }

    /// Whether a restored store has the shape a sketch gives it: keys
    /// strictly ascending, a finite weight each, positive without a window,
    /// and under one a positive count each.
    fn has_shape(&self, counted: bool) -> bool {
        let counts = if counted {
            self.n.len() == self.keys.len() && self.n.iter().all(|&c| c > 0)
        } else {
            self.n.is_empty()
        };
        self.w.len() == self.keys.len()
            && counts
            && self.keys.windows(2).all(|p| p[0] < p[1])
            && self
                .w
                .iter()
                .all(|w| w.is_finite() && (*w > 0.0 || (counted && *w == 0.0)))
    }
}

/// A DDSketch over one column: the two stores and the zero count (the
/// module's).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct DdSketch {
    pos: Store,
    neg: Store,
    zero_w: f64,
    /// The rows in the zero count, under a window; else 0.
    zero_n: u64,
}

/// Where a value goes in a sketch: a store's key, or the zero count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Slot {
    Neg(i32),
    Zero,
    Pos(i32),
}

impl Slot {
    /// The slot of a finite value under `map`.
    pub fn of(map: &Mapping, x: f64) -> Slot {
        if x > map.min_possible {
            Slot::Pos(map.key(x))
        } else if x < -map.min_possible {
            Slot::Neg(map.key(-x))
        } else {
            Slot::Zero
        }
    }

    /// The 32 bits a window's ring keeps a slot in: the key times 4 and a
    /// tag (keys are under `2^29` in magnitude at `alpha >= 1e-6`), and
    /// [`Slot::ABSENT`] for a row without the value.
    pub fn code(self) -> i32 {
        match self {
            Slot::Pos(k) => k << 2,
            Slot::Neg(k) => (k << 2) | 1,
            Slot::Zero => 2,
        }
    }

    /// A row the column was absent from.
    pub const ABSENT: i32 = 3;

    /// The slot a [`Slot::code`] names, `None` for [`Slot::ABSENT`].
    pub fn decode(code: i32) -> Option<Slot> {
        match code & 3 {
            0 => Some(Slot::Pos(code >> 2)),
            1 => Some(Slot::Neg(code >> 2)),
            2 => Some(Slot::Zero),
            _ => None,
        }
    }
}

impl DdSketch {
    /// The total weight, summed in value order (the negative store from
    /// its largest magnitude, the zero count, the positive store up).
    pub fn weight(&self) -> f64 {
        let mut total = 0.0;
        for (_, w) in self.neg.buckets().rev() {
            total += w;
        }
        total += self.zero_w;
        for (_, w) in self.pos.buckets() {
            total += w;
        }
        total
    }

    pub fn is_empty(&self) -> bool {
        self.pos.is_empty() && self.neg.is_empty() && self.zero_w == 0.0
    }

    pub fn positive(&self) -> &Store {
        &self.pos
    }

    pub fn negative(&self) -> &Store {
        &self.neg
    }

    pub fn zero_weight(&self) -> f64 {
        self.zero_w
    }

    /// The buckets it would hold, both stores, with `slot` added.
    pub fn len_with(&self, slot: Slot) -> usize {
        match slot {
            Slot::Pos(k) => self.pos.len_with(k) + self.neg.len(),
            Slot::Neg(k) => self.neg.len_with(k) + self.pos.len(),
            Slot::Zero => self.pos.len() + self.neg.len(),
        }
    }

    pub fn buckets_len(&self) -> usize {
        self.pos.len() + self.neg.len()
    }

    /// A row of weight `w > 0` in `slot`; `counted` under a window.
    pub fn add(&mut self, map: &Mapping, slot: Slot, w: f64, counted: bool) {
        match slot {
            Slot::Pos(k) => self.pos.add(map, k, w, counted),
            Slot::Neg(k) => self.neg.add(map, k, w, counted),
            Slot::Zero => {
                self.zero_w += w;
                if counted {
                    self.zero_n += 1;
                }
            }
        }
    }

    /// Take a row of weight `w` out of `slot` (a windowed sketch).
    pub fn remove(&mut self, slot: Slot, w: f64) {
        match slot {
            Slot::Pos(k) => self.pos.remove(k, w),
            Slot::Neg(k) => self.neg.remove(k, w),
            Slot::Zero => {
                self.zero_n -= 1;
                self.zero_w = if self.zero_n == 0 {
                    0.0
                } else {
                    (self.zero_w - w).max(0.0)
                };
            }
        }
    }

    /// The buckets in value order as `(representative, weight)`.
    fn ascending(&self) -> impl Iterator<Item = (f64, f64)> + '_ {
        let neg = self.neg.reps_and_weights().rev().map(|(r, w)| (-r, w));
        let zero = (self.zero_w > 0.0).then_some((0.0, self.zero_w));
        neg.chain(zero).chain(self.pos.reps_and_weights())
    }

    /// The buckets from the largest value down.
    fn descending(&self) -> impl Iterator<Item = (f64, f64)> + '_ {
        let zero = (self.zero_w > 0.0).then_some((0.0, self.zero_w));
        let neg = self.neg.reps_and_weights().map(|(r, w)| (-r, w));
        self.pos.reps_and_weights().rev().chain(zero).chain(neg)
    }

    /// Decay the sketch by `f` in `(0, 1)`: every bucket's weight and the
    /// zero count times `f`, each rounded once (the decayed form; a windowed
    /// sketch takes no decay).
    pub fn decay(&mut self, f: f64) {
        self.pos.scale(f);
        self.neg.scale(f);
        self.zero_w *= f;
    }

    /// Work every bucket's representative out again from `map`: after a
    /// load, which keeps the keys and weights alone.
    pub fn refill(&mut self, map: &Mapping) {
        self.pos.refill(map);
        self.neg.refill(map);
    }

    /// The quantile and the tail mean at each of `levels` (each in (0, 1);
    /// 0.5, whose tail has no side, reads the upper), `None` while the
    /// sketch is empty: the module's reads.
    ///
    /// The quantile at `p` is the representative of the first bucket, in
    /// value order, whose cumulative weight reaches `p W`. The tail is the
    /// lower one below 0.5, the mass `m = p W` from the smallest value,
    /// and the upper one above it, `m = (1 - p) W` from the largest; its
    /// mean takes every bucket wholly inside it and, of the bucket it ends
    /// in, only the weight it still needs:
    ///
    /// ```text
    /// ES = (sum_{k<j} c_k r_k + (m - C_{j-1}) r_j) / m
    /// ```
    pub fn reads(&self, levels: &[f64]) -> Vec<Option<(f64, f64)>> {
        if self.is_empty() {
            return vec![None; levels.len()];
        }
        let total = self.weight();
        levels
            .iter()
            .map(|&p| {
                let lower = p < 0.5;
                let (q, es_lower) = Self::walk(self.ascending(), p * total);
                let es = if lower {
                    es_lower
                } else {
                    Self::walk(self.descending(), (1.0 - p) * total).1
                };
                Some((q, es))
            })
            .collect()
    }

    /// Walk `buckets` until their weight reaches `m`: the bucket it ends in
    /// and the mean of the first `m` of weight, the end bucket's in part.
    fn walk(buckets: impl Iterator<Item = (f64, f64)>, m: f64) -> (f64, f64) {
        let (mut before, mut sum, mut last) = (0.0, 0.0, 0.0);
        for (r, c) in buckets {
            last = r;
            if before + c >= m {
                return (r, (sum + (m - before) * r) / m);
            }
            before += c;
            sum += c * r;
        }
        (last, sum / before)
    }

    /// Back to no rows.
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// The bytes it takes beside its own fields.
    pub fn heap_bytes(&self, counted: bool) -> u64 {
        self.pos.bytes(counted) + self.neg.bytes(counted)
    }

    /// Whether a restored sketch has the shape `add` and `remove` give it.
    pub fn has_shape(&self, counted: bool) -> bool {
        self.pos.has_shape(counted)
            && self.neg.has_shape(counted)
            && self.zero_w.is_finite()
            && self.zero_w >= 0.0
            && (counted || self.zero_n == 0)
            && (!counted || self.zero_n > 0 || self.zero_w == 0.0)
    }
}

#[cfg(test)]
mod tests;
