//! The summaries `ewm_moments` keeps over a group of legs, and Pébay's
//! merge of a summary with a row (the online update) and with another
//! summary (a window's blocks). See the parent module's docs.

use serde::{Deserialize, Serialize};

use super::{Binomials, Leg, binomial};

/// A multi-index over a group's legs: the power of each.
pub(crate) type Idx = Vec<u32>;

pub(crate) fn order(a: &[u32]) -> u32 {
    a.iter().sum()
}

/// Every multi-index at or below `a`, the first leg slowest.
pub(crate) fn below(a: &[u32]) -> Vec<Idx> {
    let mut out: Vec<Idx> = vec![Vec::new()];
    for &k in a {
        let mut next = Vec::with_capacity(out.len() * (k as usize + 1));
        for p in &out {
            for v in 0..=k {
                let mut q = p.clone();
                q.push(v);
                next.push(q);
            }
        }
        out = next;
    }
    out
}

/// `prod_j C(a_j, b_j)`.
pub(crate) fn binom(a: &[u32], b: &[u32]) -> f64 {
    let mut c = 1.0;
    for (&x, &y) in a.iter().zip(b) {
        c *= binomial(x, y);
    }
    c
}

pub(crate) fn unit(d: usize, j: usize, k: u32) -> Idx {
    (0..d).map(|i| if i == j { k } else { 0 }).collect()
}

pub(crate) fn add(a: &[u32], b: &[u32]) -> Idx {
    a.iter().zip(b).map(|(x, y)| x + y).collect()
}

pub(crate) fn sub(a: &[u32], b: &[u32]) -> Idx {
    a.iter().zip(b).map(|(x, y)| x - y).collect()
}

/// The moments a summary over a group's legs keeps.
#[derive(Debug, Clone, Default)]
pub(crate) struct Group {
    /// The legs, sorted by column and then lag.
    pub legs: Vec<Leg>,
    /// Every multi-index of order 2 or more the terms need, by order and
    /// then lexicographically.
    pub keep: Vec<Idx>,
    /// The `k` of each `r_k` kept: 2, then 3 and 4 where an unbiased third
    /// or fourth moment is asked for (none for the weight's own group).
    pub powers: Vec<u32>,
    /// Each leg's highest power.
    pub top: Vec<u32>,
    /// Per kept moment, the two legs of an order 2 moment, whose row update
    /// is `EwDiag`'s expression.
    pub two: Vec<Option<(usize, usize)>>,
}

impl Group {
    pub fn new(legs: Vec<Leg>, need: &[Idx], powers: Vec<u32>) -> Self {
        let d = legs.len();
        let mut set = std::collections::BTreeSet::new();
        for a in need {
            for b in below(a) {
                let n = order(&b);
                if n >= 2 {
                    set.insert((n, b));
                }
            }
        }
        let keep: Vec<Idx> = set.into_iter().map(|(_, b)| b).collect();
        let top = (0..d)
            .map(|j| keep.iter().map(|a| a[j]).max().unwrap_or(0))
            .collect();
        let two = keep
            .iter()
            .map(|a| {
                (order(a) == 2).then(|| {
                    let mut it = (0..d).flat_map(|j| std::iter::repeat_n(j, a[j] as usize));
                    (it.next().unwrap_or(0), it.next().unwrap_or(0))
                })
            })
            .collect();
        Self {
            legs,
            keep,
            powers,
            top,
            two,
        }
    }

    /// Where `a` is kept.
    pub fn at(&self, a: &[u32]) -> Option<usize> {
        self.keep
            .binary_search_by(|k| (order(k), k.as_slice()).cmp(&(order(a), a)))
            .ok()
    }

    /// Calls `f(β, C(a, β), where c_{a−β} is kept)` for every `β ≤ a` with
    /// `|a − β| ≠ 1` (a moment of order 1 is 0 about its own mean), the
    /// first leg slowest; the place is `None` at order 0, where `c` is 1.
    pub fn each_shift(&self, a: &[u32], mut f: impl FnMut(&[u32], f64, Option<usize>)) {
        let d = a.len();
        let tables: Vec<Vec<f64>> = a
            .iter()
            .map(|&n| {
                let mut b = Binomials::new(n);
                let mut t = Vec::with_capacity(n as usize + 1);
                t.push(b.get());
                for _ in 0..n {
                    b.step();
                    t.push(b.get());
                }
                t
            })
            .collect();
        let mut beta = vec![0u32; d];
        let mut g = a.to_vec();
        loop {
            let n = order(&g);
            if n != 1 {
                let mut coef = 1.0;
                for (t, &k) in tables.iter().zip(&beta) {
                    coef *= t[k as usize];
                }
                f(&beta, coef, if n == 0 { None } else { self.at(&g) });
            }
            let mut j = d;
            loop {
                if j == 0 {
                    return;
                }
                j -= 1;
                if beta[j] < a[j] {
                    beta[j] += 1;
                    g[j] -= 1;
                    break;
                }
                beta[j] = 0;
                g[j] = a[j];
            }
        }
    }

    /// The doubles a summary takes: `W`, the means' pairs, the moments and
    /// the `r_k`.
    pub fn numbers(&self) -> u64 {
        (1 + 2 * self.legs.len() + self.keep.len() + self.powers.len()) as u64
    }

    pub fn lagged(&self) -> bool {
        self.legs.iter().any(|l| l.lag > 0.0)
    }
}

/// `(W, m + m_lo, c, r, e)` over a group's legs; `W = 0` is empty.
///
/// `e` is the moments' scale: each `c_a` is held as `μ_a 2^(−e |a|)`, the
/// moment of the deviations measured in units of `2^e`. It is 0 for every
/// stream whose moments fit a double, which is then the arithmetic as
/// written, to the bit. A row whose deviation would carry a moment past
/// the largest double raises it, and it falls back as soon as the moments
/// fit again: the update is homogeneous in the deviations (each term of
/// `c'_a` is of degree `|a|` in `d` and the `c`), so it runs on scaled
/// deviations as it does on the deviations themselves, and a scale of a
/// power of two moves no digit (where nothing underflows).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct Summary {
    pub w: f64,
    pub m: Vec<f64>,
    pub m_lo: Vec<f64>,
    pub c: Vec<f64>,
    pub r: Vec<f64>,
    /// Last, and written only where it is not 0, so a state whose moments
    /// fit a double is written as it was before the scale.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub e: i32,
}

fn is_zero(e: &i32) -> bool {
    *e == 0
}

/// The largest scale a state may hold. A rise takes a deviation (below
/// `2^334`) below 1 and a moment (below `2^1024`) below 2, so no scale
/// passes about 850; past 2048 every moment of order 2 is below the
/// smallest double.
pub(crate) const MAX_SCALE: i32 = 2048;

/// Where every intermediate of an update must stay: below `2^1020`, short
/// of the largest double.
const ROOM: i32 = 1020;

/// The highest order a scale holds. Past it a moment's binomial
/// coefficients, and its sum of them, can pass the largest double at any
/// scale (`C(1030, 515)` is past `2^1024`), so such a moment is the
/// arithmetic as written, and withheld where it overflows.
pub(crate) const MAX_HELD_ORDER: u32 = 1000;

/// `x^0 .. x^top` by repeated multiplication.
fn powers(x: f64, top: u32) -> Vec<f64> {
    let mut out = Vec::with_capacity(top as usize + 1);
    let mut p = 1.0;
    out.push(p);
    for _ in 0..top {
        p *= x;
        out.push(p);
    }
    out
}

/// `⌊log₂ max(1, |x|)⌋` over `xs`, from the bits: 0 where every `|x|` is
/// below 2.
fn top_exponent(xs: &[f64]) -> i32 {
    xs.iter()
        .map(|x| ((x.to_bits() >> 52) & 0x7ff) as i32 - 1023)
        .fold(0, i32::max)
}

/// `⌊log₂ |x|⌋` from the bits (−1023 at 0 and below the normal range).
fn exponent(x: f64) -> i32 {
    ((x.to_bits() >> 52) & 0x7ff) as i32 - 1023
}

/// `x 2^k`, exactly while the result is a normal double.
fn times_pow2(x: f64, k: i64) -> f64 {
    libm::scalbn(x, k.clamp(-8192, 8192) as i32)
}

/// Every moment rescaled from the scale `e` to `e + by`: `c_a 2^(−by |a|)`.
fn rescale(c: &mut [f64], g: &Group, by: i32) {
    for (v, a) in c.iter_mut().zip(&g.keep) {
        *v = times_pow2(*v, -i64::from(by) * i64::from(order(a)));
    }
}

/// The deviations at the scale `e`: `d 2^(−e)`.
fn scaled(d: &[f64], e: i32) -> Vec<f64> {
    d.iter().map(|&x| times_pow2(x, -i64::from(e))).collect()
}

impl Group {
    /// How many of the kept moments, the lowest orders, a scale can hold:
    /// those of order at most [`MAX_HELD_ORDER`].
    fn held(&self) -> usize {
        self.keep.partition_point(|a| order(a) <= MAX_HELD_ORDER)
    }

    /// The highest order a scale holds.
    fn top_order(&self) -> i32 {
        self.keep[..self.held()]
            .last()
            .map_or(0, |a| order(a) as i32)
    }

    /// Whether every moment a scale holds is finite.
    fn finite(&self, c: &[f64]) -> bool {
        c[..self.held()].iter().all(|v| v.is_finite())
    }

    /// Whether an update whose moments are below `2^(c_exp + 1)` and whose
    /// deviations below `2^(d_exp + 1)` (each at least 2) keeps every
    /// intermediate finite. A moment of order `p` is a sum of at most
    /// `2^p` terms, each a binomial coefficient (at most `2^p`) times a
    /// moment times at most `p` deviations, all in a share `a` or `b` of at
    /// most 1; the order-2 form is a moment plus a square.
    fn fits(&self, c: &[f64], d_exp: i32) -> bool {
        let t = self.top_order();
        let c_exp = top_exponent(&c[..self.held()]);
        2 * t + 2 + c_exp + t * (d_exp.max(0) + 1) <= ROOM
    }

    /// The least rise of the scale after which an update keeps every
    /// moment a scale holds finite: its deviations, below `2^(d_exp + 1)`
    /// now, below 1, and each moment `c_a` below the room [`Group::fits`]
    /// leaves once divided by `2^(by |a|)`. Past an order of about 340 the
    /// bound leaves no room, and the moments are taken below 2: a moment of
    /// order `p` of deviations below 1 is then below `2^(p + 1)`, which a
    /// double holds to [`MAX_HELD_ORDER`].
    fn rise(&self, cs: &[&[f64]], d_exp: i32) -> i32 {
        let t = self.top_order();
        let room = (ROOM - 3 * t - 2).max(0);
        let mut by = (d_exp + 1).max(1);
        for c in cs {
            for (v, a) in c.iter().zip(&self.keep[..self.held()]) {
                // ⌈over / p⌉ where it is over; at most 0 where it is not.
                let p = order(a) as i32;
                let over = exponent(*v) - room;
                by = by.max((over + p - 1) / p);
            }
        }
        by
    }

    /// The scale, at most `e`, to fall back to: the lowest at which every
    /// moment a scale holds would leave an update of deviations below 16
    /// room to fit.
    fn fall(&self, c: &[f64], e: i32) -> i32 {
        let t = self.top_order();
        let room = (ROOM - 6 * t - 2).max(0);
        let mut to = 0;
        for (v, a) in c.iter().zip(&self.keep[..self.held()]) {
            if *v == 0.0 {
                continue;
            }
            let p = order(a) as i32;
            // The lowest e' with exponent(v) + (e − e') p ≤ room.
            let spare = room - exponent(*v);
            to = to.max(e - spare.div_euclid(p));
        }
        to.clamp(0, e)
    }
}

/// Pébay's merge of the moments `c` with one row, in place: deviations
/// `d` (at the moments' scale), shares `a` and `b`.
fn row_moments(c: &mut [f64], g: &Group, a: f64, b: f64, d: &[f64]) {
    let pa: Vec<Vec<f64>> = d
        .iter()
        .zip(&g.top)
        .map(|(&x, &t)| powers(-b * x, t))
        .collect();
    let pb: Vec<Vec<f64>> = d
        .iter()
        .zip(&g.top)
        .map(|(&x, &t)| powers(a * x, t))
        .collect();
    for i in (0..g.keep.len()).rev() {
        if g.two[i].is_some() {
            continue;
        }
        let mut s = 0.0;
        g.each_shift(&g.keep[i], |beta, coef, gi| {
            let (mut qa, mut qb) = (1.0, 1.0);
            for (j, &k) in beta.iter().enumerate() {
                qa *= pa[j][k as usize];
                qb *= pb[j][k as usize];
            }
            let term = match gi {
                None => a * qa + b * qb,
                Some(x) => a * c[x] * qa,
            };
            s += coef * term;
        });
        c[i] = s;
    }
    for (c, two) in c.iter_mut().zip(&g.two) {
        if let Some((p, q)) = *two {
            *c = a * *c + a * b * d[p] * d[q];
        }
    }
}

/// Pébay's merge of two summaries' moments: `ca` of the earlier rows,
/// `cb` of the later, `δ` the later's means less the earlier's (at the
/// moments' scale), `a` and `b` their shares.
fn merged_moments(g: &Group, ca: &[f64], cb: &[f64], a: f64, b: f64, delta: &[f64]) -> Vec<f64> {
    let pa: Vec<Vec<f64>> = delta
        .iter()
        .zip(&g.top)
        .map(|(&x, &t)| powers(-b * x, t))
        .collect();
    let pb: Vec<Vec<f64>> = delta
        .iter()
        .zip(&g.top)
        .map(|(&x, &t)| powers(a * x, t))
        .collect();
    let mut c = vec![0.0; g.keep.len()];
    for (ci, alpha) in c.iter_mut().zip(&g.keep) {
        let mut s = 0.0;
        g.each_shift(alpha, |beta, coef, gi| {
            let (mut qa, mut qb) = (1.0, 1.0);
            for (j, &k) in beta.iter().enumerate() {
                qa *= pa[j][k as usize];
                qb *= pb[j][k as usize];
            }
            let (xa, xb) = match gi {
                None => (1.0, 1.0),
                Some(x) => (ca[x], cb[x]),
            };
            s += coef * (a * xa * qa + b * xb * qb);
        });
        *ci = s;
    }
    c
}

impl Summary {
    pub fn empty(g: &Group) -> Self {
        let d = g.legs.len();
        Self {
            w: 0.0,
            m: vec![0.0; d],
            m_lo: vec![0.0; d],
            c: vec![0.0; g.keep.len()],
            r: vec![0.0; g.powers.len()],
            e: 0,
        }
    }

    pub fn has_shape(&self, g: &Group) -> bool {
        let d = g.legs.len();
        self.m.len() == d
            && self.m_lo.len() == d
            && self.c.len() == g.keep.len()
            && self.r.len() == g.powers.len()
            && (0..=MAX_SCALE).contains(&self.e)
    }

    /// `c_a` at the summary's scale: 1 at order 0, 0 at order 1 (about the
    /// own mean).
    pub fn cv(&self, g: &Group, a: &[u32]) -> f64 {
        match order(a) {
            0 => 1.0,
            1 => 0.0,
            _ => g.at(a).map_or(f64::NAN, |i| self.c[i]),
        }
    }

    /// `x` of degree `p` in the deviations, read at the summary's scale,
    /// in the deviations' own units: `x 2^(e p)`, `inf` past the largest
    /// double.
    pub fn unscale(&self, x: f64, p: u32) -> f64 {
        times_pow2(x, i64::from(self.e) * i64::from(p))
    }

    /// The summary at scale 0, a moment past the largest double `inf`:
    /// what a statistic that is not homogeneous in the deviations (a raw
    /// moment mixes the means in) is read from.
    pub fn unscaled(&self, g: &Group) -> Self {
        let mut out = self.clone();
        rescale(&mut out.c, g, -self.e);
        out.e = 0;
        out
    }

    /// `r_k`.
    pub fn r(&self, g: &Group, k: u32) -> f64 {
        g.powers
            .iter()
            .position(|&p| p == k)
            .map_or(f64::NAN, |i| self.r[i])
    }

    /// The decay of every weight by `f`: `W` alone moves, every `c` and
    /// `r` being a ratio a uniform rescaling leaves where it was.
    pub fn decay(&mut self, f: f64) {
        self.w *= f;
    }

    /// One row of weight `w` at `z` after a decay of `lam`: Pébay's merge
    /// with the one-row summary, order 2 by `EwDiag`'s expression. Where
    /// the weight is left at 0 the update stops at that condition.
    pub fn learn(&mut self, g: &Group, lam: f64, w: f64, z: &[f64]) {
        let w_new = lam * self.w + w;
        if w_new <= 0.0 {
            // `a` and `b` are 0/0: nothing carried and nothing added.
            self.w = w_new;
            return;
        }
        let a = lam * self.w / w_new;
        let b = w / w_new;
        if b > 0.0 {
            let d: Vec<f64> = z
                .iter()
                .zip(&self.m)
                .zip(&self.m_lo)
                .map(|((&x, &m), &lo)| crate::comp::dev(x, m, lo))
                .collect();
            self.row_at_scale(g, a, b, &d);
            let (a2, b2) = (a * a, b * b);
            for (r, &k) in self.r.iter_mut().zip(&g.powers) {
                *r = match k {
                    2 => a2 * *r + b2,
                    3 => a2 * a * *r + b2 * b,
                    _ => a2 * a2 * *r + b2 * b2,
                };
            }
            for ((m, lo), &x) in self.m.iter_mut().zip(self.m_lo.iter_mut()).zip(&d) {
                crate::comp::add(m, lo, b * x);
            }
        }
        self.w = w_new;
    }

    /// The row's moments, at the scale they fit: where the bound
    /// ([`Group::fits`]) says nothing can pass the largest double, in place
    /// at the scale as it is, the arithmetic as written; otherwise the same
    /// from a copy, and where a moment passes the largest double, again at
    /// the scale [`Group::rise`] asks for, which holds them. Then the scale
    /// falls back as far as the moments allow.
    fn row_at_scale(&mut self, g: &Group, a: f64, b: f64, d: &[f64]) {
        if g.fits(&self.c, top_exponent(d) - self.e) {
            if self.e == 0 {
                row_moments(&mut self.c, g, a, b, d);
            } else {
                row_moments(&mut self.c, g, a, b, &scaled(d, self.e));
            }
        } else {
            let old = self.c.clone();
            row_moments(&mut self.c, g, a, b, &scaled(d, self.e));
            if !g.finite(&self.c) {
                // The moments as they were, at the scale the bound asks for.
                let by = g.rise(&[&old], top_exponent(d) - self.e);
                self.c.copy_from_slice(&old);
                rescale(&mut self.c, g, by);
                self.e += by;
                row_moments(&mut self.c, g, a, b, &scaled(d, self.e));
            }
        }
        self.fall_back(g);
    }

    /// The scale lowered as far as the moments allow ([`Group::fall`]).
    fn fall_back(&mut self, g: &Group) {
        if self.e == 0 {
            return;
        }
        let to = g.fall(&self.c, self.e);
        rescale(&mut self.c, g, to - self.e);
        self.e = to;
    }

    /// Pébay's merge with `other`, a summary of later rows: an empty side
    /// gives the other unchanged. The two are read at the larger of their
    /// scales, raised where the merge would pass the largest double.
    pub fn merge(&mut self, g: &Group, other: &Summary) {
        if other.w == 0.0 {
            return;
        }
        if self.w == 0.0 {
            self.clone_from(other);
            return;
        }
        let w = self.w + other.w;
        let a = self.w / w;
        let b = other.w / w;
        let delta: Vec<f64> = (0..g.legs.len())
            .map(|j| crate::comp::dev(other.m[j], self.m[j], self.m_lo[j]) + other.m_lo[j])
            .collect();
        let mut theirs = std::borrow::Cow::Borrowed(&other.c);
        let mut e = self.e.max(other.e);
        if self.e != e {
            rescale(&mut self.c, g, e - self.e);
        }
        if other.e != e {
            rescale(theirs.to_mut(), g, e - other.e);
        }
        self.e = e;
        let mut c = if e == 0 {
            merged_moments(g, &self.c, &theirs, a, b, &delta)
        } else {
            merged_moments(g, &self.c, &theirs, a, b, &scaled(&delta, e))
        };
        if !g.finite(&c) {
            let by = g.rise(&[&self.c, &theirs], top_exponent(&delta) - e);
            rescale(&mut self.c, g, by);
            rescale(theirs.to_mut(), g, by);
            e += by;
            self.e = e;
            c = merged_moments(g, &self.c, &theirs, a, b, &scaled(&delta, e));
        }
        self.c = c;
        let (a2, b2) = (a * a, b * b);
        for ((r, &ro), &k) in self.r.iter_mut().zip(&other.r).zip(&g.powers) {
            *r = match k {
                2 => a2 * *r + b2 * ro,
                3 => a2 * a * *r + b2 * b * ro,
                _ => a2 * a2 * *r + b2 * b2 * ro,
            };
        }
        for ((m, lo), d) in self
            .m
            .iter_mut()
            .zip(&mut self.m_lo)
            .zip(&delta)
            .take(g.legs.len())
        {
            crate::comp::add(m, lo, b * d);
        }
        self.w = w;
        self.fall_back(g);
    }
}

#[cfg(test)]
#[path = "layout_tests.rs"]
mod tests;
