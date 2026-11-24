//! Grams of disjoint row sets pooled into the Gram of their union, on a
//! subset of the columns: `rheon.gram.merge`'s arithmetic, entry
//! for entry, so the two agree to the bit where the parts are the same.
//!
//! Chan, Golub and LeVeque's update, folding the parts in in order: with
//! the weights `W_a`, `W_b` and the mean gap `d = m_b − m_a`,
//!
//! ```text
//! W = W_a + W_b
//! m = m_a + (W_b / W) d
//! C = (W_a C_a + W_b C_b) / W + (W_a W_b / W²) d dᵀ
//! ```
//!
//! and per target `t`, with its weights `w_a`, `w_b`, `α = w_a / w`,
//! `β = w_b / w` (both 0 where `w = w_a + w_b` is 0), its mean gap `δ` and
//! the gap `Δm` between its column means:
//!
//! ```text
//! var_t   = α var_a + β var_b + α β δ²         ȳ_t = ȳ_a + β δ
//! E[z y]  = (w_a E_a[z y] + w_b E_b[z y]) · (1 / w)
//! cc_t    = α cc_a + β cc_b + α β δ_ȳ Δm       m_t = α m_a + β m_b
//! Q_t     = Q_a + Q_b                          n_kish_t = w² / Q_t
//! ```
//!
//! with `ȳ` the cross-moment at the intercept (`target_means` without one)
//! and `Q = w² / n_kish` each part's sum of squared weights. One part is
//! its own union, returned as it is. The parts' co-moments may be float32
//! or packed ([`Comoments`]); the merge is in float64 either way.
//!
//! [`merge_within`] pools the parts *within* themselves instead (docs/PLAN.md
//! task 240): each part's moments about its own means, averaged by its
//! weight, without the terms in the gaps between the parts' means:
//!
//! ```text
//! W = Σ W_i          C   = Σ W_i C_i / W
//! w_t = Σ w_i        cc_t = Σ w_i cc_i / w_t      var_t = Σ w_i var_i / w_t
//! ```
//!
//! folded in order as the union is, so where the parts' means are equal the
//! two are the same numbers. [`within_cross`] takes a sign `s_i = ±1` per
//! part on its target (task 241): `cc_t = Σ w_i s_i cc_i / w_t`, what
//! negating each part's target rows gives, since that negates the part's
//! centred cross-moments and leaves its weight and variance as they are. The result is the Gram of the rows demeaned
//! part by part: its means are 0 (the intercept's 1 aside), its
//! cross-moments its centred ones, each target's mean 0, and it records per
//! target the parts it has rows in ([`OwnedGram::within`]). A regression on
//! it is one with an intercept per part, so it recovers no intercept and
//! its degrees of freedom lose one per part. A part with no weight for a
//! target adds nothing to it (`w_i = 0`).

use super::{Comoments, GramArrays};

/// A Gram held in Rust memory: what [`merge`] returns, on its columns.
#[derive(Clone, Debug, PartialEq)]
pub struct OwnedGram {
    pub k: usize,
    pub weight_sum: f64,
    pub means: Vec<f64>,
    pub comoments: Vec<f64>,
    pub cross_moments: Vec<f64>,
    pub means_by_target: Vec<f64>,
    pub cross_centred: Vec<f64>,
    pub target_weights: Vec<f64>,
    pub target_means: Vec<f64>,
    pub target_vars: Vec<f64>,
    pub target_n_kish: Vec<f64>,
    /// Per target, the parts pooled within ([`merge_within`]) that it has
    /// rows in, each with an intercept of its own; `None` for a Gram not
    /// pooled within.
    pub within: Option<Vec<usize>>,
}

impl OwnedGram {
    /// The arrays, borrowed.
    pub fn arrays(&self) -> GramArrays<'_> {
        GramArrays {
            k: self.k,
            means: &self.means,
            comoments: Comoments::Full(&self.comoments),
            cross_moments: &self.cross_moments,
            means_by_target: &self.means_by_target,
            cross_centred: &self.cross_centred,
            weight_sum: self.weight_sum,
            target_weights: &self.target_weights,
            target_means: &self.target_means,
            target_vars: &self.target_vars,
            target_n_kish: &self.target_n_kish,
            within: self.within.as_deref(),
        }
    }
}

/// Whether a weight is a positive number: false for 0 and for NaN.
#[inline]
fn weighs(w: f64) -> bool {
    w > 0.0
}

/// Each target's `ȳ`: its cross-moment at the intercept, or its own mean
/// without one.
fn ybar_of(p: &GramArrays<'_>, icept: Option<usize>) -> Vec<f64> {
    (0..p.targets())
        .map(|t| match icept {
            Some(c) => p.cross_moments[t * p.k + c],
            None => p.target_means[t],
        })
        .collect()
}

/// Each target's `Q = w² / n_kish`: 0 where the part has no weight for it,
/// and NaN where it has weight but no Kish size, so the merged size is
/// unknown rather than one that counts the part's weight and none of its
/// rows (review 2026-11-23, REG-14: a part of weight 1000 beside one of 10
/// rows made a size of 102,010). A part with weight and a size of 0 is
/// one that could not count its rows, and the same.
///
/// Each `w` is read over `scale`, a power of two a target (the largest
/// weight's, [`kish_scales`]), so `w²` neither overflows past `1.3e154` nor
/// underflows below `1.5e-154`; a power of two divides exactly, so in
/// range the result is `w² / n_kish` over `scale²` to the bit.
fn q_of(p: &GramArrays<'_>, scale: &[f64]) -> Vec<f64> {
    p.target_weights
        .iter()
        .zip(p.target_n_kish)
        .zip(scale)
        .map(|((&w, &nk), &s)| {
            if !weighs(w) {
                0.0
            } else if nk.is_finite() && nk > 0.0 {
                let ws = w / s;
                ws * ws / nk
            } else {
                f64::NAN
            }
        })
        .collect()
}

/// Per target, the power of two at or below the largest weight any part
/// has for it (1 where none has one): what [`q_of`] reads each weight over.
fn kish_scales(parts: &[GramArrays<'_>]) -> Vec<f64> {
    let m = parts[0].targets();
    (0..m)
        .map(|t| {
            let top = parts
                .iter()
                .map(|p| p.target_weights[t])
                .fold(0.0, f64::max);
            if top.is_finite() && top >= f64::MIN_POSITIVE {
                f64::from_bits(top.to_bits() & 0x7ff0_0000_0000_0000)
            } else {
                1.0
            }
        })
        .collect()
}

/// A target's merged Kish size from its weight and its `Q` over `scale²`.
#[inline]
fn kish(tw: f64, qt: f64, s: f64) -> f64 {
    if qt > 0.0 {
        let ws = tw / s;
        ws * ws / qt
    } else {
        f64::NAN
    }
}

/// The spread term's weight `W_a W_b / W²`, read in shares where the
/// product or the square leaves the doubles: past `1.3e154` or below
/// `1.5e-154` the plain form is `inf / inf` or `0 / 0` (review 2026-11-23,
/// REG-14). The plain form wherever it is a number, so a merge in range
/// keeps its bits, and `rheon.gram.merge`'s with them.
#[inline]
fn spread_weight(w: f64, wb: f64, total: f64) -> f64 {
    let plain = w * wb / (total * total);
    if plain.is_finite() && (plain > 0.0 || w == 0.0 || wb == 0.0) {
        plain
    } else {
        (w / total) * (wb / total)
    }
}

/// `(w a + wb b) / total`, read in shares where a product leaves the
/// doubles, as [`spread_weight`]; `a` alone where `wb` is 0 and `b` alone
/// where `w` is, so a part that adds nothing adds nothing even where its
/// moments are NaN (review 2026-11-23, MATH-7: `0 · NaN` is NaN).
#[inline]
fn blend(w: f64, a: f64, wb: f64, b: f64, total: f64) -> f64 {
    if wb == 0.0 {
        return a;
    }
    if w == 0.0 {
        return b;
    }
    let plain = (w * a + wb * b) / total;
    if plain.is_finite() || !(a.is_finite() && b.is_finite()) {
        plain
    } else {
        (w / total) * a + (wb / total) * b
    }
}

/// `parts` pooled, on the columns `cols` (positions in every part's
/// columns, in the order wanted), the intercept at `icept` among the
/// parts' columns or none. Every part has the same columns and targets;
/// `parts` is not empty. A part pooled within ([`merge_within`]) is its own
/// union alone: its means are 0, so pooled beside another part it would add
/// the gap to that part's means as if its rows had not been demeaned.
pub fn merge(parts: &[GramArrays<'_>], cols: &[usize], icept: Option<usize>) -> OwnedGram {
    let first = &parts[0];
    let (k, n, m) = (first.k, cols.len(), first.targets());
    let pick = |v: &[f64]| -> Vec<f64> {
        (0..m)
            .flat_map(|t| cols.iter().map(move |&i| v[t * k + i]))
            .collect()
    };
    let block = |p: &GramArrays<'_>| -> Vec<f64> {
        cols.iter()
            .flat_map(|&i| cols.iter().map(move |&j| p.comoments.get(k, i, j)))
            .collect()
    };
    let mut g = OwnedGram {
        k: n,
        weight_sum: first.weight_sum,
        means: cols.iter().map(|&i| first.means[i]).collect(),
        comoments: block(first),
        cross_moments: pick(first.cross_moments),
        means_by_target: pick(first.means_by_target),
        cross_centred: pick(first.cross_centred),
        target_weights: first.target_weights.to_vec(),
        target_means: first.target_means.to_vec(),
        target_vars: first.target_vars.to_vec(),
        target_n_kish: first.target_n_kish.to_vec(),
        within: first.within.map(<[usize]>::to_vec),
    };
    if parts.len() == 1 {
        return g;
    }
    g.within = None;
    let scales = kish_scales(parts);
    let mut q = q_of(first, &scales);
    let mut ybar = ybar_of(first, icept);
    for p in &parts[1..] {
        let (w, wb) = (g.weight_sum, p.weight_sum);
        let total = w + wb;
        if total > 0.0 {
            let d: Vec<f64> = cols
                .iter()
                .zip(&g.means)
                .map(|(&i, &mean)| p.means[i] - mean)
                .collect();
            let spread = spread_weight(w, wb, total);
            for (r, &i) in cols.iter().enumerate() {
                for (c, &j) in cols.iter().enumerate() {
                    let e = &mut g.comoments[r * n + c];
                    let pooled = blend(w, *e, wb, p.comoments.get(k, i, j), total);
                    *e = if w == 0.0 || wb == 0.0 {
                        pooled
                    } else {
                        pooled + spread * (d[r] * d[c])
                    };
                }
            }
            if w == 0.0 {
                for (mean, &i) in g.means.iter_mut().zip(cols) {
                    *mean = p.means[i];
                }
            } else if wb > 0.0 {
                for (mean, dr) in g.means.iter_mut().zip(&d) {
                    *mean += (wb / total) * dr;
                }
            }
        }
        g.weight_sum = total;
        let ybar_b = ybar_of(p, icept);
        let q_b = q_of(p, &scales);
        for t in 0..m {
            let (tw, twb) = (g.target_weights[t], p.target_weights[t]);
            let ttotal = tw + twb;
            q[t] += q_b[t];
            // A part with no weight for the target adds nothing to it, and
            // the first with weight is its moments as they are: neither is
            // multiplied in at 0, where its moments may be NaN (MATH-7).
            if !weighs(twb) {
                continue;
            }
            if !weighs(tw) {
                g.target_vars[t] = p.target_vars[t];
                g.target_means[t] = p.target_means[t];
                ybar[t] = ybar_b[t];
                for (r, &i) in cols.iter().enumerate() {
                    let (e, src) = (t * n + r, t * k + i);
                    g.cross_moments[e] = p.cross_moments[src];
                    g.cross_centred[e] = p.cross_centred[src];
                    g.means_by_target[e] = p.means_by_target[src];
                }
                g.target_weights[t] = ttotal;
                continue;
            }
            let live = ttotal > 0.0;
            let (a, b) = if live {
                (tw / ttotal, twb / ttotal)
            } else {
                (0.0, 0.0)
            };
            let dt = if live {
                p.target_means[t] - g.target_means[t]
            } else {
                0.0
            };
            g.target_vars[t] = a * g.target_vars[t] + b * p.target_vars[t] + a * b * dt * dt;
            g.target_means[t] += b * dt;
            let scale = if live { 1.0 / ttotal } else { 0.0 };
            let dy = if live { ybar_b[t] - ybar[t] } else { 0.0 };
            for (r, &i) in cols.iter().enumerate() {
                let e = t * n + r;
                let src = t * k + i;
                let plain = (tw * g.cross_moments[e] + twb * p.cross_moments[src]) * scale;
                g.cross_moments[e] = if plain.is_finite() {
                    plain
                } else {
                    a * g.cross_moments[e] + b * p.cross_moments[src]
                };
                let dm = if live {
                    p.means_by_target[src] - g.means_by_target[e]
                } else {
                    0.0
                };
                g.cross_centred[e] =
                    a * g.cross_centred[e] + b * p.cross_centred[src] + (a * b * dy) * dm;
                g.means_by_target[e] = a * g.means_by_target[e] + b * p.means_by_target[src];
            }
            ybar[t] += b * dy;
            g.target_weights[t] = ttotal;
        }
    }
    for (((nk, &tw), &qt), &sc) in g
        .target_n_kish
        .iter_mut()
        .zip(&g.target_weights)
        .zip(&q)
        .zip(&scales)
    {
        *nk = kish(tw, qt, sc);
    }
    g
}

/// `parts` pooled within themselves, on the columns `cols`, the intercept
/// at `icept` or none, as [`merge`] takes them: the module's within form.
/// A part itself pooled within adds its parts to [`OwnedGram::within`], and
/// one that was not adds itself to each target it has weight for.
pub fn merge_within(parts: &[GramArrays<'_>], cols: &[usize], icept: Option<usize>) -> OwnedGram {
    let first = &parts[0];
    let (k, n, m) = (first.k, cols.len(), first.targets());
    let mut weight_sum = first.weight_sum;
    let mut comoments: Vec<f64> = cols
        .iter()
        .flat_map(|&i| cols.iter().map(move |&j| first.comoments.get(k, i, j)))
        .collect();
    for p in &parts[1..] {
        let (w, wb) = (weight_sum, p.weight_sum);
        let total = w + wb;
        if total > 0.0 {
            for (r, &i) in cols.iter().enumerate() {
                for (c, &j) in cols.iter().enumerate() {
                    let e = &mut comoments[r * n + c];
                    *e = blend(w, *e, wb, p.comoments.get(k, i, j), total);
                }
            }
        }
        weight_sum = total;
    }
    let mut target_weights = first.target_weights.to_vec();
    let mut target_vars = first.target_vars.to_vec();
    let scales = kish_scales(parts);
    let mut q = q_of(first, &scales);
    for p in &parts[1..] {
        let q_b = q_of(p, &scales);
        for t in 0..m {
            let (tw, twb) = (target_weights[t], p.target_weights[t]);
            let ttotal = tw + twb;
            q[t] += q_b[t];
            // As [`merge`]: a part with no weight adds nothing (MATH-7).
            if !weighs(twb) {
                continue;
            }
            target_vars[t] = if tw > 0.0 {
                (tw / ttotal) * target_vars[t] + (twb / ttotal) * p.target_vars[t]
            } else {
                p.target_vars[t]
            };
            target_weights[t] = ttotal;
        }
    }
    let target_n_kish = target_weights
        .iter()
        .zip(&q)
        .zip(&scales)
        .map(|((&tw, &qt), &sc)| kish(tw, qt, sc))
        .collect();
    let within = (0..m)
        .map(|t| {
            parts
                .iter()
                .map(|p| match p.within {
                    Some(w) => w[t],
                    None => usize::from(p.target_weights[t] > 0.0),
                })
                .sum()
        })
        .collect();
    // The demeaned rows' means: 0 for every feature, and the intercept's
    // constant 1.
    let means: Vec<f64> = cols
        .iter()
        .map(|&i| if Some(i) == icept { 1.0 } else { 0.0 })
        .collect();
    let cross_centred = within_cross(parts, cols, &[]);
    OwnedGram {
        k: n,
        weight_sum,
        means_by_target: means.repeat(m),
        means,
        comoments,
        cross_moments: cross_centred.clone(),
        cross_centred,
        target_weights,
        target_means: vec![0.0; m],
        target_vars,
        target_n_kish,
        within: Some(within),
    }
}

/// Each target's centred cross-moments pooled within the parts, `m × n` on
/// the columns `cols`: `Σ w_i s_i cc_i / w_t`, folded in order as [`merge`]
/// folds them. `signs` is empty (every `s_i` +1) or one sign per part. A
/// draw of signs is a selection's null run: it needs no second merge of the
/// co-moments, which no sign moves (docs/PLAN.md task 241).
pub fn within_cross(parts: &[GramArrays<'_>], cols: &[usize], signs: &[f64]) -> Vec<f64> {
    let first = &parts[0];
    let (k, n, m) = (first.k, cols.len(), first.targets());
    let sign = |p: usize| if signs.is_empty() { 1.0 } else { signs[p] };
    let s0 = sign(0);
    let mut cc: Vec<f64> = (0..m)
        .flat_map(|t| {
            cols.iter()
                .map(move |&i| s0 * first.cross_centred[t * k + i])
        })
        .collect();
    let mut tw = first.target_weights.to_vec();
    for (pi, p) in parts.iter().enumerate().skip(1) {
        let s = sign(pi);
        for (t, w) in tw.iter_mut().enumerate() {
            let wb = p.target_weights[t];
            let ttotal = *w + wb;
            // As [`merge`]: a part with no weight adds nothing (MATH-7).
            if !weighs(wb) {
                continue;
            }
            let first = !weighs(*w);
            let (a, b) = (*w / ttotal, wb / ttotal);
            for (r, &i) in cols.iter().enumerate() {
                let e = t * n + r;
                let theirs = s * p.cross_centred[t * k + i];
                cc[e] = if first {
                    theirs
                } else {
                    a * cc[e] + b * theirs
                };
            }
            *w = ttotal;
        }
    }
    cc
}
