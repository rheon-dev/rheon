//! A segment's sums ([`Acc`]) and how two segments are joined ([`then`],
//! [`then_by`]): moved from `windows.rs` unchanged (task 561), so that file
//! stays under the source cap.

use super::{DecayMemo, Direction, KernelDef};

/// Per operator, a segment's sums: the mass-weighted value, the mass and
/// the rows with a value. For a sum or a rate `s` is the decayed sum and
/// `w` unused. For a variance (task 212) `s` is the mean itself, `hi` of a
/// compensated pair with `lo` (`rheon_core::comp`), `w` the mass `V1`,
/// `q` the mass-weighted squared deviations from the mean `M2`, and `e`
/// the masses' products in pairs, `Σ_{i<j} m_i m_j = (V1² − V2) / 2`, kept
/// as a sum of its own so the count correction cancels nothing, and is 0,
/// not a rounding of 0, while one row carries the window; the three are zero
/// for every other statistic.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub(super) struct Acc {
    pub(super) s: f64,
    pub(super) w: f64,
    pub(super) n: f64,
    pub(super) lo: f64,
    pub(super) q: f64,
    pub(super) e: f64,
}

impl Acc {
    /// A held value `x` of mass `m`: its sums as one row.
    #[inline]
    pub(super) fn held(x: f64, m: f64, var: bool) -> Acc {
        if var {
            Acc {
                s: x,
                w: m,
                n: 1.0,
                lo: 0.0,
                q: 0.0,
                e: 0.0,
            }
        } else {
            Acc {
                s: m * x,
                w: m,
                n: 1.0,
                ..Acc::default()
            }
        }
    }

    /// The sums seen `f` of a discount later: a variance's mean stays where
    /// it is, its masses scale, and their squares with `f²`.
    #[inline]
    pub(super) fn scaled(self, f: f64, var: bool) -> Acc {
        if var {
            Acc {
                s: self.s,
                lo: self.lo,
                w: f * self.w,
                n: self.n,
                q: f * self.q,
                e: f * (f * self.e),
            }
        } else {
            Acc {
                s: f * self.s,
                w: f * self.w,
                n: self.n,
                ..Acc::default()
            }
        }
    }

    /// Both segments' sums, seen from one time.
    #[inline]
    pub(super) fn plus(self, o: Acc, var: bool) -> Acc {
        if var {
            return merge(self, o);
        }
        Acc {
            s: self.s + o.s,
            w: self.w + o.w,
            n: self.n + o.n,
            ..Acc::default()
        }
    }

    /// Read from a stack's arena: three fields, or a variance-wide queue's
    /// six.
    #[inline]
    pub(super) fn read(v: &[f64], wide: bool) -> Acc {
        if wide {
            Acc {
                s: v[0],
                w: v[1],
                n: v[2],
                lo: v[3],
                q: v[4],
                e: v[5],
            }
        } else {
            Acc {
                s: v[0],
                w: v[1],
                n: v[2],
                ..Acc::default()
            }
        }
    }

    /// Written to a stack's arena, as [`Acc::read`] reads it.
    #[inline]
    pub(super) fn write(self, out: &mut [f64], wide: bool) {
        out[0] = self.s;
        out[1] = self.w;
        out[2] = self.n;
        if wide {
            out[3] = self.lo;
            out[4] = self.q;
            out[5] = self.e;
        }
    }
}

/// Two variance segments' sums as one, about their pooled mean: Chan, Golub
/// and LeVeque's pairwise update, `V1 = V1_a + V1_b`, `μ = μ_a + δ V1_b / V1`
/// and `M2 = M2_a + M2_b + δ² V1_a V1_b / V1` with `δ = μ_b − μ_a`, and the
/// pairs' products `e = e_a + e_b + V1_a V1_b`. No term
/// is a difference of two large sums, so a window of one value has `M2 = 0`
/// exactly at any level. The mean moves by its step as a compensated pair
/// (`rheon_core::comp`): a plain one stops `1/(2 V1_b/V1)` rounding steps
/// short of a value held row after row, and `M2` then settles on that gap's
/// square instead of decaying with its history. A segment of no mass moves
/// nothing but the row count, so it takes no step (`comp::add` says why);
/// an empty one is `Acc::default()`.
#[inline]
fn merge(a: Acc, b: Acc) -> Acc {
    let n = a.n + b.n;
    let q = a.q + b.q;
    let e = a.e + b.e;
    if b.w <= 0.0 {
        return Acc { n, q, e, ..a };
    }
    if a.w <= 0.0 {
        return Acc { n, q, e, ..b };
    }
    let w = a.w + b.w;
    let d = rheon_core::comp::dev(b.s, a.s, a.lo) + b.lo;
    let (mut hi, mut lo) = (a.s, a.lo);
    rheon_core::comp::add(&mut hi, &mut lo, d * (b.w / w));
    Acc {
        s: hi,
        lo,
        w,
        n,
        q: q + d * d * (a.w / w * b.w),
        e: e + a.w * b.w,
    }
}

/// `a` then `b`, `a` the earlier, each anchored at its own time: the sums
/// over both, anchored at the newer looking back and at the older looking
/// ahead, so a window's sums are always anchored at the row they are read
/// from or discounted to it once.
#[inline]
pub(super) fn then(
    k: &KernelDef,
    m: &mut DecayMemo,
    a: Acc,
    a_at: f64,
    b: Acc,
    b_at: f64,
    var: bool,
) -> (Acc, f64) {
    then_by(k, k.discount_in(b_at - a_at, m), a, a_at, b, b_at, var)
}

/// [`then`] with the discount across the two anchors given: the operators
/// of a queue are folded at the same two anchors on a row, so the caller
/// takes it once.
#[inline]
pub(super) fn then_by(
    k: &KernelDef,
    f: f64,
    a: Acc,
    a_at: f64,
    b: Acc,
    b_at: f64,
    var: bool,
) -> (Acc, f64) {
    match k.direction {
        Direction::Backward => (a.scaled(f, var).plus(b, var), b_at),
        Direction::Forward => (a.plus(b.scaled(f, var), var), a_at),
    }
}
