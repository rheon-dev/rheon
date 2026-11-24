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

/// `(W, m + m_lo, c, r)` over a group's legs; `W = 0` is empty.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct Summary {
    pub w: f64,
    pub m: Vec<f64>,
    pub m_lo: Vec<f64>,
    pub c: Vec<f64>,
    pub r: Vec<f64>,
}

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

impl Summary {
    pub fn empty(g: &Group) -> Self {
        let d = g.legs.len();
        Self {
            w: 0.0,
            m: vec![0.0; d],
            m_lo: vec![0.0; d],
            c: vec![0.0; g.keep.len()],
            r: vec![0.0; g.powers.len()],
        }
    }

    pub fn has_shape(&self, g: &Group) -> bool {
        let d = g.legs.len();
        self.m.len() == d
            && self.m_lo.len() == d
            && self.c.len() == g.keep.len()
            && self.r.len() == g.powers.len()
    }

    /// `c_a`: 1 at order 0, 0 at order 1 (about the own mean).
    pub fn cv(&self, g: &Group, a: &[u32]) -> f64 {
        match order(a) {
            0 => 1.0,
            1 => 0.0,
            _ => g.at(a).map_or(f64::NAN, |i| self.c[i]),
        }
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
                let c = &self.c;
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
                self.c[i] = s;
            }
            for (c, two) in self.c.iter_mut().zip(&g.two) {
                if let Some((p, q)) = *two {
                    *c = a * *c + a * b * d[p] * d[q];
                }
            }
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

    /// Pébay's merge with `other`, a summary of later rows: an empty side
    /// gives the other unchanged.
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
                let (ca, cb) = match gi {
                    None => (1.0, 1.0),
                    Some(x) => (self.c[x], other.c[x]),
                };
                s += coef * (a * ca * qa + b * cb * qb);
            });
            *ci = s;
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
    }
}
