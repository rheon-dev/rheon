//! An exact running sum of doubles: Shewchuk's expansion sums.
//!
//! A double sum rounds at every addition, and the error grows with the
//! number of terms and the size of the total. A threshold read from such a
//! sum is read from its drift: at a total of 1e12 a double resolves about
//! 1e-4, so a crossing that falls on one row in exact arithmetic can fall
//! on the next. [`Expansion`] keeps the sum exactly instead, as a short list
//! of doubles whose exact sum is the value (Shewchuk 1997, "Adaptive
//! precision floating-point arithmetic and fast robust geometric
//! predicates", Discrete & Computational Geometry 18).
//!
//! # The operations
//!
//! - **Adding a double** is Knuth's two-sum, which splits `a + b` into its
//!   rounded sum and the rounding error, both doubles, run up the list
//!   (Shewchuk's grow-expansion with zero elimination, the loop of Python's
//!   `math.fsum`). It is exact.
//! - **Scaling by a power of two** is exact for a double that neither
//!   overflows nor underflows, so `m·θ` for a whole `m < 2^53` enters as at
//!   most 53 exact parts, `2^j·θ` for each set bit `j` of `m`.
//! - **Rounding the sum once, to nearest** is `math.fsum`'s last step: the
//!   top parts added down to the first inexact step, and a half-way case
//!   decided by the sign of the part below.
//!
//! No product rounds and no library function is called, so every operation
//! gives the same bits on every system: a state saved on one system
//! continues to the bit on another.
//!
//! # The canonical form, and its bound
//!
//! After each change the list is put in one canonical form:
//!
//! ```text
//! h_1 = RN(V),  h_2 = RN(V − h_1),  h_3 = RN(V − h_1 − h_2),  ...
//! ```
//!
//! each part the value left, rounded to the nearest double, until nothing
//! is left. Since `|V − h_1| ≤ ulp(h_1)/2`, each part is at most half a unit
//! in the last place of the part above it, so its exponent is at least 53
//! below. The form is that of Joldes, Marty, Muller and Popescu's
//! renormalization ("Arithmetic algorithms for extended precision using
//! floating-point expansions", IEEE Transactions on Computers, 2016), reached
//! here by rounding and exact subtraction alone. Three things follow:
//!
//! - **The top part is the sum rounded once, to nearest**, ties to even, as
//!   `math.fsum` rounds it: [`Expansion::value`] reads it with no work.
//! - **A comparison with a double is exact and reads two parts.** With `h_1`
//!   the sum rounded, `V` lies within half a gap of `h_1`, so `V` is above a
//!   double `t` below `h_1`, below a `t` above it, and at `h_1` its side of
//!   `t` is the sign of `h_2` ([`Expansion::cmp_f64`]).
//! - **The list is short.** Every part is a multiple of `2^-1074`, the
//!   smallest double, and the exponents fall by at least 53 a part, so a
//!   value below `2^e` holds at most `⌊(e + 1074) / 53⌋ + 1` parts: 28 for a
//!   value below `2^334`, which bounds a running quantity of inputs and
//!   thresholds up to 1e100, and [`MAX_PARTS`] over the whole range of
//!   doubles. On realistic data it holds one to three.

use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

/// The most parts an expansion of finite parts can hold in the canonical
/// form: the exponents span `2^-1074` to `2^1023`, and fall by at least 53 a
/// part, so `⌊(1023 + 1074) / 53⌋ + 1 = 40`.
pub const MAX_PARTS: usize = 40;

/// The most parts a value below `2^e` holds, for the bound of a state's
/// bytes: `⌊(e + 1074) / 53⌋ + 1`.
pub const fn parts_below(e: i32) -> usize {
    ((e + 1074) / 53 + 1) as usize
}

/// An exact sum of doubles, in the canonical form of the module's
/// documentation: `parts` in increasing magnitude, the last the sum rounded
/// to nearest, no part zero. The empty list is 0.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Expansion {
    parts: Vec<f64>,
}

/// `x + y` and its rounding error, exactly, when `|x| >= |y|` (Dekker's
/// fast two-sum).
#[inline]
fn fast_two_sum(x: f64, y: f64) -> (f64, f64) {
    let hi = x + y;
    (hi, y - (hi - x))
}

/// Add `x` to the nonoverlapping parts `p`, in increasing magnitude, keeping
/// them so: the inner loop of Python's `math.fsum` (Shewchuk's
/// grow-expansion, zeros dropped). Exact for finite parts whose sums do not
/// overflow.
fn grow(p: &mut Vec<f64>, mut x: f64) {
    let mut kept = 0;
    for j in 0..p.len() {
        let mut y = p[j];
        if x.abs() < y.abs() {
            std::mem::swap(&mut x, &mut y);
        }
        let (hi, lo) = fast_two_sum(x, y);
        if lo != 0.0 {
            p[kept] = lo;
            kept += 1;
        }
        x = hi;
    }
    p.truncate(kept);
    if x != 0.0 {
        p.push(x);
    }
}

/// The exact sum of the nonoverlapping parts `p`, in increasing magnitude,
/// rounded once to nearest, ties to even: the last step of Python's
/// `math.fsum`.
fn round(p: &[f64]) -> f64 {
    let mut n = p.len();
    if n == 0 {
        return 0.0;
    }
    n -= 1;
    let mut hi = p[n];
    let mut lo = 0.0;
    while n > 0 {
        n -= 1;
        let x = hi;
        let y = p[n];
        hi = x + y;
        let yr = hi - x;
        lo = y - yr;
        if lo != 0.0 {
            break;
        }
    }
    // A half-way case: `lo` is exactly half a unit of `hi`, and the parts
    // below it lean the same way, so the sum is past half way and rounds
    // away from `hi`.
    if n > 0 && ((lo < 0.0 && p[n - 1] < 0.0) || (lo > 0.0 && p[n - 1] > 0.0)) {
        let y = lo * 2.0;
        let x = hi + y;
        let yr = x - hi;
        if y == yr {
            hi = x;
        }
    }
    hi
}

/// `2^j` as a double, for `j` in `0..=1023`.
#[inline]
fn pow2(j: u32) -> f64 {
    f64::from_bits(u64::from(1023 + j) << 52)
}

/// Why [`Expansion::floor_div`] cannot give `m`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FloorDiv {
    /// `m` is `2^53` or more: a threshold below `2^-53` of the value.
    TooMany,
}

impl Expansion {
    /// The expansion of 0.
    pub const fn new() -> Self {
        Self { parts: Vec::new() }
    }

    /// The parts, in increasing magnitude; the last is the sum rounded.
    pub fn parts(&self) -> &[f64] {
        &self.parts
    }

    /// An expansion of the parts a saved state holds, which must be in the
    /// canonical form, finite and nonzero: a damaged list is refused rather
    /// than read as another value.
    pub fn from_parts(parts: Vec<f64>) -> Result<Self, String> {
        if parts.len() > MAX_PARTS || parts.iter().any(|p| !p.is_finite() || *p == 0.0) {
            return Err(
                "an exact sum holds a part that is zero, not finite, or past the most \
                        parts one holds"
                    .into(),
            );
        }
        let mut e = Self::new();
        for &p in &parts {
            grow(&mut e.parts, p);
        }
        e.canonicalize();
        if e.parts != parts {
            return Err("an exact sum's parts are not in the form this library writes".into());
        }
        Ok(e)
    }

    /// Whether the sum is exactly 0.
    pub fn is_zero(&self) -> bool {
        self.parts.is_empty()
    }

    /// The sum rounded once, to nearest, ties to even: the top part.
    pub fn value(&self) -> f64 {
        self.parts.last().copied().unwrap_or(0.0)
    }

    /// Back to 0, keeping the list's allocation.
    pub fn clear(&mut self) {
        self.parts.clear();
    }

    /// The sum's bytes in memory: the list's header and its parts.
    pub fn bytes(&self) -> usize {
        std::mem::size_of::<Vec<f64>>() + 8 * self.parts.capacity()
    }

    /// Put the parts in the canonical form: `h_1 = RN(V)` on top, then the
    /// canonical form of `V − h_1` below it. Exact: each part is rounded
    /// from what is left, and subtracted exactly.
    fn canonicalize(&mut self) {
        if self.parts.len() <= 1 {
            return;
        }
        let mut tops = [0.0f64; MAX_PARTS + 1];
        let mut n = 0;
        while !self.parts.is_empty() {
            let h = round(&self.parts);
            grow(&mut self.parts, -h);
            tops[n] = h;
            n += 1;
        }
        self.parts.extend(tops[..n].iter().rev());
    }

    /// Add a finite double, exactly.
    pub fn add(&mut self, x: f64) {
        if x == 0.0 {
            return;
        }
        grow(&mut self.parts, x);
        self.canonicalize();
    }

    /// Add a finite double, exactly, unless the sum could pass `2^1020` in
    /// magnitude, where a two-sum could overflow: then nothing is added and
    /// the answer is `false`. A running quantity bounded by its inputs never
    /// comes near it; a float sum of values past 1e300 can.
    pub fn add_checked(&mut self, x: f64) -> bool {
        const BOUND: f64 = 1.1235582092889474e307; // 2^1020
        if self.value().abs() + x.abs() >= BOUND {
            return false;
        }
        self.add(x);
        true
    }

    /// Add an integer, exactly: as its nearest double and the integer left
    /// over, both doubles, for any `|v| < 2^64`.
    pub fn add_int(&mut self, v: i128) {
        let hi = v as f64;
        // `hi` is a whole number within `2^11` of `v`, so both casts and
        // the difference are exact.
        let lo = (v - hi as i128) as f64;
        grow(&mut self.parts, hi);
        if lo != 0.0 {
            grow(&mut self.parts, lo);
        }
        self.canonicalize();
    }

    /// Change the sign, exactly.
    pub fn negate(&mut self) {
        for p in &mut self.parts {
            *p = -*p;
        }
    }

    /// The sign of the exact sum against 0.
    pub fn signum(&self) -> Ordering {
        match self.parts.last() {
            None => Ordering::Equal,
            Some(&h) if h > 0.0 => Ordering::Greater,
            Some(_) => Ordering::Less,
        }
    }

    /// The exact sum against a finite double `t`.
    ///
    /// With `h_1` the sum rounded to nearest, the sum lies within half the
    /// gap between `h_1` and the double beside it. A double `t` below `h_1`
    /// is at least one gap below, so the sum is above `t`; likewise below a
    /// `t` above `h_1`. At `t = h_1` the sum's side is the sign of the rest,
    /// which is the sign of its top part, `h_2`.
    pub fn cmp_f64(&self, t: f64) -> Ordering {
        let n = self.parts.len();
        let Some(&h) = self.parts.last() else {
            return 0.0f64.partial_cmp(&t).unwrap_or(Ordering::Equal);
        };
        match h.partial_cmp(&t) {
            Some(Ordering::Equal) | None => match n {
                1 => Ordering::Equal,
                _ if self.parts[n - 2] > 0.0 => Ordering::Greater,
                _ => Ordering::Less,
            },
            Some(o) => o,
        }
    }

    /// Subtract `m·θ` exactly, for a whole `m < 2^53` and a finite `θ`
    /// whose `2^52·θ` does not overflow: `2^j·θ` for each set bit `j`, each a
    /// scaling by a power of two.
    pub fn sub_multiple(&mut self, m: u64, theta: f64) {
        debug_assert!(m < 1 << 53);
        let mut bits = m;
        let mut j = 0;
        while bits != 0 {
            if bits & 1 == 1 {
                grow(&mut self.parts, -(theta * pow2(j)));
            }
            bits >>= 1;
            j += 1;
        }
        self.canonicalize();
    }

    /// `m = ⌊V / θ⌋` for a sum `V ≥ 0` and a finite `θ > 0`, leaving `V −
    /// m·θ`, in `[0, θ)`, in place. The estimate `⌊h_1 / θ⌋` is within two
    /// of `m`; it is subtracted, then stepped by one, up or down, until the
    /// exact test holds.
    ///
    /// # Errors
    ///
    /// [`FloorDiv::TooMany`] when `m` is `2^53` or more, leaving the sum as
    /// it was.
    pub fn floor_div(&mut self, theta: f64) -> Result<u64, FloorDiv> {
        const LIMIT: f64 = 9_007_199_254_740_992.0; // 2^53
        let est = (self.value() / theta).floor();
        // NaN, an infinity or a count past the limit.
        if est.is_nan() || est >= LIMIT - 2.0 {
            return Err(FloorDiv::TooMany);
        }
        let mut m = est as u64;
        self.sub_multiple(m, theta);
        while self.signum() == Ordering::Less {
            self.add(theta);
            m -= 1;
        }
        while self.cmp_f64(theta) != Ordering::Less {
            self.add(-theta);
            m += 1;
        }
        Ok(m)
    }
}

#[cfg(test)]
mod tests;
