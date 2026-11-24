//! A sum of squared weights kept at a power of two of its own: `Q = q ·
//! 4^s` (review 2026-11-23, REG-12).
//!
//! Kish's count `(Σw)² / Σw²` is the same at every scale of the weights,
//! but `w²` is not a double below `2^-537` (and is subnormal below
//! `2^-511`): at weights of `1e-200` every square was 0, so the count was
//! 0 for good. The sum is kept as `q` beside an exponent `s`, each row's
//! weight entering as `(w · 2^-s)²`. While the weights are near 1 the
//! exponent is 0 and every operation is the plain one, to the bit, and a
//! state writes no exponent, so its bytes are as before. Where a weight
//! taken to the sum's scale leaves `[2^-256, 2^256]` the scale moves to it
//! first, by an exact power of two: unless the sum then held is past the
//! doubles at the new scale, which makes the row's square too small beside
//! it to add anything.

/// `2^-256` and `2^256`: a weight at the sum's scale inside these squares
/// to a normal double with room to spare.
const LO: f64 = 8.636_168_555_094_445e-78;
const HI: f64 = 1.157_920_892_373_162e77;

/// `x · 2^n`, exactly where the result is a normal double: in steps of at
/// most `2^1000`, so no step leaves the doubles on its own.
pub(crate) fn ldexp(mut x: f64, mut n: i64) -> f64 {
    while n != 0 && x != 0.0 && x.is_finite() {
        let step = n.clamp(-1000, 1000);
        x *= f64::from_bits(((step + 1023) as u64) << 52);
        n -= step;
    }
    x
}

/// The sum after a row: `λ² Q + w²`, `lam` the decay (1 for an undecayed
/// sum), `w` the row's weight at its own scale.
///
/// Every row of a model that keeps such a sum runs this, so the path of a
/// sum at its own scale 0 and a weight inside `[2^-256, 2^256]` (or 0,
/// which moves no scale either) is the plain operation, inlined
/// where it is called; [`add_scaled`] takes the rest, the same arithmetic
/// it ran before the split, to the bit.
#[inline]
pub(crate) fn add(q: &mut f64, s: &mut i32, lam: f64, w: f64) {
    // A positive double's bits order as its value does, so one unsigned
    // comparison of the bits tells `[LO, HI]`; 0 and a weight that is not
    // finite move no scale either, and `add_scaled` gives them the same sum.
    let in_range = w.to_bits().wrapping_sub(LO.to_bits()) <= HI.to_bits() - LO.to_bits();
    let held = lam * lam * *q;
    // A decay that takes the sum held below the normal doubles is taken by
    // the scale (`add_scaled`): a step of no weight across 538 half-lives
    // made `λ²` 0, and Kish's size, which no decay moves, absent (review
    // 2026-11-23, COV-4).
    if *s == 0 && (in_range || w == 0.0) && (held >= f64::MIN_POSITIVE || *q == 0.0) {
        *q = held + w * w;
    } else {
        add_scaled(q, s, lam, w);
    }
}

/// [`add`] where the sum's scale is not 0 or the weight leaves the range
/// at it: rare, so out of line.
#[cold]
#[inline(never)]
fn add_scaled(q: &mut f64, s: &mut i32, lam: f64, w: f64) {
    let mut held = lam * lam * *q;
    // Where the decay would take the sum held below the normal doubles, the
    // scale takes the decay's power of two and the sum's own: `λ = f · 2^e`
    // and `q = g · 4^t`, `g` in `[1/4, 1)`, so `λ² · q · 4^s = (f² · g) ·
    // 4^(s + e + t)`, exactly, and the part kept is near 1. Only while the
    // weight the sum is of can still be a normal double, `Σw² ≤ (Σw)²`, so
    // to a scale of `2^-1022`: past it the weight itself is subnormal, and
    // the sum fades as before ([`count`] reads no size there).
    if *q > 0.0 && lam > 0.0 && lam.is_finite() && !(held >= f64::MIN_POSITIVE) {
        let (f, e) = crate::held_median::frexp(lam);
        let (_, eq) = crate::held_median::frexp(*q);
        let t = eq.div_euclid(2);
        let moved = i64::from(*s) + e + t;
        if moved >= MIN_DECAYED_SCALE && scale_fits_i64(moved) {
            held = f * f * ldexp(*q, -2 * t);
            *s = moved as i32;
        }
    }
    let mut ws = ldexp(w, -i64::from(*s));
    // `ws` is infinite where a finite weight is past the doubles at this
    // scale (a weight of 1 at a scale of 2^-1073): the scale moves to the
    // weight's own, read from the weight, not from its scaled value.
    if w.is_finite() && ws != 0.0 && !(LO..=HI).contains(&ws) {
        let (_, e) = crate::held_median::frexp(w.abs());
        let t = e - i64::from(*s);
        let moved = ldexp(held, -2 * t);
        if moved.is_finite() {
            held = moved;
            ws = ldexp(w, -e);
            *s += t as i32;
        } else {
            // The sum held is more than `2^512` times the row's square at
            // any scale both are doubles at: the row adds nothing.
            *q = held;
            return;
        }
    }
    *q = held + ws * ws;
}

/// The least scale a decay takes the sum to ([`add_scaled`]): `4^-1022`,
/// the square of the least normal double.
const MIN_DECAYED_SCALE: i64 = -1022;

/// `Q`, a double: 0 where it is below the doubles.
pub(crate) fn value(q: f64, s: i32) -> f64 {
    ldexp(q, 2 * i64::from(s))
}

/// Kish's count `W² / Q` from `W` and the sum in parts; `None` where the
/// sum is not above 0, and 0 where `W` is a subnormal double.
#[inline]
pub(crate) fn count(w: f64, q: f64, s: i32) -> Option<f64> {
    if q > 0.0 && w > 0.0 && w < f64::MIN_POSITIVE {
        // A weight past the normal doubles has lost its last bits (one
        // decayed there sticks at the least subnormal while the sum, kept
        // at its own scale, decays on): read against the sum it was a size
        // of `1e292` (review 2026-11-23, COV-4). It counts as none, as its
        // square, 0 in the doubles, always made it.
        Some(0.0)
    } else if q > 0.0 && s == 0 {
        Some(w * w / q)
    } else if q > 0.0 {
        let ws = ldexp(w, -i64::from(s));
        Some(ws * ws / q)
    } else {
        None
    }
}

/// `Q / W`, the weight of a row in Kish's sense; `None` where either is not
/// above 0, or `W` is a subnormal double.
#[inline]
pub(crate) fn per_weight(w: f64, q: f64, s: i32) -> Option<f64> {
    if !(w > 0.0 && q > 0.0) {
        None
    } else if w < f64::MIN_POSITIVE {
        // As [`count`]: a subnormal weight has lost its last bits.
        None
    } else if s == 0 {
        Some(q / w)
    } else {
        Some(ldexp(q / ldexp(w, -i64::from(s)), i64::from(s)))
    }
}

/// Whether a loaded exponent is one a weight in the doubles makes: within
/// the doubles' exponents, `2^±1100`, with room for the steps that move it.
pub(crate) fn scale_fits(s: i32) -> bool {
    scale_fits_i64(i64::from(s))
}

fn scale_fits_i64(s: i64) -> bool {
    (-1200..=1200).contains(&s)
}

/// Serde's `skip_serializing_if` for an exponent: 0 is written by no state.
pub(crate) fn is_zero(s: &i32) -> bool {
    *s == 0
}

/// A sum of squared weights that may be absent (a state written before Kish
/// sums were kept), with its power of two: `Σw² = q · 4^s`. Written as the
/// plain `q` while `s` is 0 -- the bytes it had before the power was kept,
/// in both the named and the positional encoding -- and as `[q, s]`
/// otherwise, which a load reads either way. One field, so no encoding
/// slides the fields after it.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub(crate) struct KishSum {
    pub(crate) q: Option<f64>,
    pub(crate) s: i32,
}

impl KishSum {
    pub(crate) fn of(q: Option<f64>) -> Self {
        Self { q, s: 0 }
    }
}

impl serde::Serialize for KishSum {
    fn serialize<S: serde::Serializer>(&self, ser: S) -> Result<S::Ok, S::Error> {
        match (self.q, self.s) {
            (q, 0) | (q @ None, _) => q.serialize(ser),
            (Some(q), s) => (q, s).serialize(ser),
        }
    }
}

impl<'de> serde::Deserialize<'de> for KishSum {
    fn deserialize<D: serde::Deserializer<'de>>(de: D) -> Result<Self, D::Error> {
        #[derive(serde::Deserialize)]
        #[serde(untagged)]
        enum Wire {
            Plain(Option<f64>),
            Scaled(f64, i32),
        }
        match Wire::deserialize(de)? {
            Wire::Plain(q) => Ok(Self::of(q)),
            Wire::Scaled(q, s) if scale_fits(s) => Ok(Self { q: Some(q), s }),
            Wire::Scaled(_, s) => Err(serde::de::Error::custom(format!(
                "a sum of squared weights at a power of 4^{s}, which no weight in the doubles makes"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// At weights near 1 the parts are the plain sum, to the bit, and the
    /// scale stays 0; at every power of two of the weights the count is the
    /// same, to the bit, where the plain sum's was 0.
    #[test]
    fn the_count_is_the_same_at_every_scale_of_the_weights() {
        let ws: Vec<f64> = (0..50).map(|i| 0.5 + 0.03 * f64::from(i % 17)).collect();
        let (mut q, mut s, mut plain) = (0.0, 0, 0.0f64);
        for &w in &ws {
            add(&mut q, &mut s, 0.9, w);
            plain = 0.81 * plain + w * w;
        }
        assert_eq!((q.to_bits(), s), (plain.to_bits(), 0));
        let sum = |c: f64| {
            let (mut q, mut s, mut w_sum) = (0.0, 0, 0.0);
            for &w in &ws {
                add(&mut q, &mut s, 0.9, w * c);
                w_sum = 0.9 * w_sum + w * c;
            }
            (
                count(w_sum, q, s).unwrap(),
                per_weight(w_sum, q, s).unwrap(),
            )
        };
        let (n1, p1) = sum(1.0);
        for e in [-1000, -700, -540, -300, 300, 500] {
            let c = ldexp(1.0, e);
            let (n, p) = sum(c);
            assert_eq!(n.to_bits(), n1.to_bits(), "2^{e}");
            assert_eq!(p.to_bits(), (p1 * c).to_bits(), "2^{e}");
        }
        assert_eq!(count(0.0, 0.0, 0), None);
    }

    /// The inlined path at scale 0 is the general one, to the bit, on each
    /// side of both edges of its range, at 0 of either sign, and on a
    /// weight negative, infinite or not a number: the split moved the
    /// cost of the scale's test, not a single sum or count.
    #[test]
    fn the_inlined_path_is_the_general_one_at_scale_0() {
        let up = |x: f64| f64::from_bits(x.to_bits() + 1);
        let down = |x: f64| f64::from_bits(x.to_bits() - 1);
        let ws = [
            LO,
            up(LO),
            down(LO),
            HI,
            up(HI),
            down(HI),
            0.0,
            -0.0,
            1.0,
            -1.0,
            0.37,
            1e-300,
            1e300,
            f64::MIN_POSITIVE,
            5e-324,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
        ];
        for &w in &ws {
            for (q0, lam) in [(0.0, 1.0), (2.5, 0.9), (1e-200, 0.5), (1e-300, 1e-10)] {
                let (mut qa, mut sa) = (q0, 0);
                let (mut qb, mut sb) = (q0, 0);
                add(&mut qa, &mut sa, lam, w);
                add_scaled(&mut qb, &mut sb, lam, w);
                assert_eq!((qa.to_bits(), sa), (qb.to_bits(), sb), "w {w:e}, q {q0}");
                // A subnormal weight counts as none (`count`).
                let subnormal = w > 0.0 && w < f64::MIN_POSITIVE;
                let general_count = (qb > 0.0).then(|| {
                    let x = ldexp(w, -i64::from(sb));
                    if subnormal { 0.0 } else { x * x / qb }
                });
                assert_eq!(
                    count(w, qa, sa).map(f64::to_bits),
                    general_count.map(f64::to_bits),
                    "count at w {w:e}"
                );
                let general_per = (w > 0.0 && qb > 0.0 && !subnormal)
                    .then(|| ldexp(qb / ldexp(w, -i64::from(sb)), i64::from(sb)));
                assert_eq!(
                    per_weight(w, qa, sa).map(f64::to_bits),
                    general_per.map(f64::to_bits),
                    "per_weight at w {w:e}"
                );
            }
        }
    }

    /// A tiny row beside a sum of ordinary rows adds nothing and moves no
    /// scale; ordinary rows after tiny ones take the scale back.
    #[test]
    fn rows_far_apart_in_weight() {
        let (mut q, mut s) = (0.0, 0);
        add(&mut q, &mut s, 1.0, 1.0);
        add(&mut q, &mut s, 1.0, 1e-300);
        assert_eq!((q, s), (1.0, 0));
        let (mut q, mut s) = (0.0, 0);
        for _ in 0..4 {
            add(&mut q, &mut s, 1.0, 1e-300);
        }
        assert!(s < -900, "{s}");
        assert_eq!(count(4e-300, q, s).unwrap(), 4.0);
        add(&mut q, &mut s, 1.0, 1.0);
        assert!((count(1.0 + 4e-300, q, s).unwrap() - 1.0).abs() < 1e-15);
        assert!(s.abs() < 2, "{s}");
    }

    /// A row whose weight, taken to the sum's scale, is past the doubles
    /// (a weight of 1 after a subnormal one, whose scale is 2^-1073) moves
    /// the scale to its own, as any weight outside the range does, and
    /// leaves a sum that is a number: a plain `ldexp` made it infinite,
    /// which a saved state then refused to load.
    #[test]
    fn a_weight_past_the_doubles_at_the_scale_moves_it() {
        for first in [5e-324, 1e-320, 2.2251e-309] {
            let (mut q, mut s) = (0.0, 0);
            add(&mut q, &mut s, 1.0, first);
            add(&mut q, &mut s, 1.0, 1.0);
            assert!(q.is_finite() && q > 0.0, "{first}: {q}");
            assert!(s.abs() < 2, "{first}: {s}");
            assert!((count(1.0 + first, q, s).unwrap() - 1.0).abs() < 1e-15);
            add(&mut q, &mut s, 1.0, 1e154);
            assert!(q.is_finite() && q > 0.0, "{first}: {q}");
        }
    }
}
