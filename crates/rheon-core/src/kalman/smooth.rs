//! The fixed-lag smoother beside the filter: on the row that completes a
//! lag of `L` observations of a covariance's target, the estimate of the
//! coefficients at the observation `L` before it, given every row up to
//! this one. A file of its own because `kalman.rs` stands near the
//! source-size cap (`tests/test_repo_hygiene.py`).
//!
//! **The model it smooths.** The filter's time index for the smoother is a
//! covariance's observations: the rows that observe its target at a
//! positive weight (under `share_p`, any of the targets). Between two
//! observations `n` and `n + 1` the filter applies a linear map `F` to the
//! coefficients -- each row's transition `Phi_a`, and each re-map's `Mᵀ`,
//! in the order the rows came -- and then, on the row of `n + 1`, adds the
//! process noise for the clock since `n` and the prior of any slot sized
//! there. With `b_n`, `P_n` the filtered pair after observation `n`'s
//! update and `b⁻`, `P⁻` the prior observation `n + 1`'s update starts
//! from, the Rauch-Tung-Striebel step back from `n + 1` to `n` is
//!
//! ```text
//! G_n = P_n Fᵀ (P⁻_{n+1})⁻¹
//! m_n = b_n + G_n (m_{n+1} − b⁻_{n+1})          = c_n + G_n m_{n+1},   c_n = b_n − G_n b⁻_{n+1}
//! V_n = P_n + G_n (V_{n+1} − P⁻_{n+1}) G_nᵀ      = K_n + G_n V_{n+1} G_nᵀ,  K_n = P_n − G_n P⁻_{n+1} G_nᵀ
//! ```
//!
//! started from the filter's pair at the row that emits, `m_t = b_t`, `V_t
//! = P_t`. A step reads only observation `n`'s pair and the smoothed pair
//! at `n + 1`, so `L` steps from `t` are the whole-prefix smoother read at
//! `t − L`, exactly: stopping early changes nothing.
//!
//! **What is kept.** `F` is never formed: `A = P_n Fᵀ` is, by applying to
//! each row of `P_n` every linear map the filter applies to `b` until the
//! next observation (`A`'s rows are `P_n`'s columns carried forward). On
//! that observation `P⁻` restricted to its sized slots is factored by
//! Cholesky and `G_n = A (P⁻)⁻¹` is two triangular solves a row; a slot
//! `P⁻` has not sized has a column of 0 in `G_n`, its variance and every
//! covariance with it being 0. The slot keeps `(c_n, G_n)`, and `K_n`'s
//! lower triangle when the smoothed variances are asked for, so the pass
//! needs no `F`: `m <- c_n + G_n m`, `V <- K_n + G_n V G_nᵀ`. Each slot also
//! keeps what reads its observation out: its stamp, which slots were
//! sized, whether the filter's gate let the observation's prediction
//! stand, and the anchor its coordinates are those of. A ring of `L − 1`
//! complete slots and the newest observation's pending pair is the state,
//! bounded by `L`, the slots and the targets; the backward pass runs on the
//! rows that emit, `O(L k²)` a target for the means and `O(L k³)` with the
//! variances, beside the gain's `O(k³)` an observation.
//!
//! **Nothing is fed back.** The smoother reads the filter and writes only
//! its own slots and fields: every prediction, residual and gate is the
//! filter's to the bit, with the smoother or without it.
//!
//! **Breaks.** Where the filter does not continue itself the step back
//! cannot cross: a covariance sized again after losing its definiteness, a
//! re-map refused (the anchor moves and `b`, `P` keep their numbers), and a
//! `P⁻` whose Cholesky meets a pivot that is not a positive number. The
//! slot before such a break keeps its filtered pair as its smoothed one,
//! `G_n = 0`, `c_n = b_n`, `K_n = P_n`, so the observations before it are
//! smoothed with what came before the break, and are emitted when their lag
//! completes. A session change and a capped gap are no break: the filter
//! continues through both, its noise charged for the gap.
//!
//! **What is emitted.** On a row that completes a lag, for each target of
//! the covariance: the described observation's stamp, its coefficients in
//! the caller's units (read out through that observation's anchor, as
//! `coefficients` reads the filter's), and with variances asked for, the
//! diagonal of `T V Tᵀ` times the target's noise as it stands, as
//! `coef_variance` reads `P`. A slot the filter had not sized at the
//! described observation is NaN, as is every slot of an observation the
//! filter's `min_weight` withheld, or of a target the filter had not yet
//! observed (`share_p`). A row that observes nothing -- a null target, a
//! weight of 0 -- takes no slot, counts in no lag and emits nothing.

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

use super::Kalman;
use crate::Stamp;

/// The described observation's prediction stood: its smoothed value is
/// emitted.
pub const SMOOTH_STOOD: u8 = 0;
/// The filter's `min_weight` withheld the described observation's
/// prediction, so its smoothed value is not reported either.
pub const SMOOTH_MIN_WEIGHT: u8 = 1;
/// The target had not been observed by the described observation (under
/// `share_p`, a target first seen after its covariance's first rows): its
/// coefficients were no estimate.
pub const SMOOTH_UNSEEN: u8 = 2;

/// What one row emitted, per target: whether the row completed the lag of
/// an observation of the target and, where it did, that observation's
/// stamp, whether its value stands ([`SMOOTH_STOOD`]) or why not, its
/// smoothed coefficients in the caller's units (`k_total` a target, NaN
/// where there is no value) and, with the variances asked for, their
/// variances laid out the same way.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Smoothed {
    pub emitted: Vec<bool>,
    pub stamp: Vec<Option<Stamp>>,
    pub held: Vec<u8>,
    pub coef: Vec<f64>,
    pub var: Vec<f64>,
}

/// One observation's smoothed value given every row so far, as
/// [`Kalman::smoothed_tail`] reads it from the state: the target, the
/// stamp, whether it stands, the coefficients and (with the variances
/// asked for) their variances, NaN where there is no value.
#[derive(Debug, Clone, PartialEq)]
pub struct SmoothedRow {
    pub target: usize,
    pub stamp: Stamp,
    pub held: u8,
    pub coef: Vec<f64>,
    pub var: Vec<f64>,
}

/// What reads an observation out: its stamp, which slots the filter had
/// sized (a variance not 0), whether each target's prediction stood
/// ([`SMOOTH_STOOD`] and the rest), and the anchor of its coordinates,
/// `[means' high parts, low parts, scales]`, empty without `standardize`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Described {
    stamp: Stamp,
    sized: Vec<bool>,
    held: Vec<u8>,
    anchor: Vec<f64>,
}

/// The newest observation's filtered pair, waiting for the next
/// observation's prior: `b_n` (each target's, one after another), `A = P_n
/// Fᵀ` carried forward row by row, and `P_n`'s lower triangle when the
/// variances are kept. `broken` once the filter has not continued itself.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Pending {
    of: Described,
    b: Vec<f64>,
    a: Vec<f64>,
    p: Vec<f64>,
    broken: bool,
}

/// One complete step back: `c_n` (a target after another), `G_n` (`k x k`,
/// empty at a break, where it is 0), and `K_n`'s lower triangle with the
/// variances.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Slot {
    of: Described,
    c: Vec<f64>,
    g: Vec<f64>,
    kk: Vec<f64>,
}

/// One covariance's ring: its complete slots, oldest first, and the pending
/// pair of its newest observation.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
struct Ring {
    slots: VecDeque<Slot>,
    pending: Option<Pending>,
}

/// The smoother's state: the lag, whether the variances are kept, a ring
/// per covariance, the stamp handed for the next row and the model's own
/// clock for a row handed none.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(super) struct Smoother {
    lag: usize,
    se: bool,
    rings: Vec<Ring>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    next: Option<Stamp>,
    clock: f64,
    #[serde(skip)]
    row: Work,
}

/// A row's emission and the pass's working space: never saved, so two
/// smoothers are equal whatever either holds here.
#[derive(Debug, Clone, Default)]
struct Work {
    stamp: Option<Stamp>,
    out: Smoothed,
    chol: Vec<f64>,
    idx: Vec<usize>,
    y: Vec<f64>,
    m: Vec<f64>,
    v: Vec<f64>,
    gv: Vec<f64>,
    tmp: Vec<f64>,
    /// Each observation's prior and filtered pair as the smoother read
    /// them, for the tests' smoother written from them.
    #[cfg(test)]
    log: Vec<Logged>,
}

/// One pair the smoother read: covariance `pi`'s prior (`prior`) or
/// filtered pair of an observation, each target's `b` one after another,
/// the anchor as it stood and the row's stamp.
#[cfg(test)]
#[derive(Debug, Clone)]
pub(super) struct Logged {
    pub(super) pi: usize,
    pub(super) prior: bool,
    pub(super) p: Vec<f64>,
    pub(super) b: Vec<f64>,
    pub(super) anchor: Vec<f64>,
    pub(super) stamp: Stamp,
}

impl PartialEq for Work {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

fn tri(k: usize) -> usize {
    k * (k + 1) / 2
}

/// The lower triangle of a row-major `k x k`, row by row.
fn lower_of(p: &[f64], k: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(tri(k));
    for i in 0..k {
        out.extend_from_slice(&p[i * k..=i * k + i]);
    }
    out
}

/// The symmetric `k x k` a lower triangle holds, into `out`.
fn full_of(t: &[f64], k: usize, out: &mut Vec<f64>) {
    out.clear();
    out.resize(k * k, 0.0);
    let mut at = 0;
    for i in 0..k {
        for j in 0..=i {
            out[i * k + j] = t[at];
            out[j * k + i] = t[at];
            at += 1;
        }
    }
}

/// `A = L Lᵀ` in place on the lower triangle of a row-major `n x n`;
/// `false` at a pivot that is not a positive number.
fn cholesky(a: &mut [f64], n: usize) -> bool {
    for j in 0..n {
        let mut d = a[j * n + j];
        for l in 0..j {
            d -= a[j * n + l] * a[j * n + l];
        }
        if !(d > 0.0 && d.is_finite()) {
            return false;
        }
        let d = d.sqrt();
        a[j * n + j] = d;
        for i in j + 1..n {
            let mut s = a[i * n + j];
            for l in 0..j {
                s -= a[i * n + l] * a[j * n + l];
            }
            a[i * n + j] = s / d;
        }
    }
    true
}

/// `x <- (L Lᵀ)⁻¹ x` for the factor [`cholesky`] left.
fn solve_factored(l: &[f64], n: usize, x: &mut [f64]) {
    for i in 0..n {
        let mut s = x[i];
        for j in 0..i {
            s -= l[i * n + j] * x[j];
        }
        x[i] = s / l[i * n + i];
    }
    for i in (0..n).rev() {
        let mut s = x[i];
        for j in i + 1..n {
            s -= l[j * n + i] * x[j];
        }
        x[i] = s / l[i * n + i];
    }
}

/// `out <- M v` for a row-major `k x k`.
fn mat_vec(mtx: &[f64], v: &[f64], out: &mut [f64]) {
    let k = v.len();
    for (o, row) in out.iter_mut().zip(mtx.chunks_exact(k)) {
        *o = row.iter().zip(v).map(|(a, b)| a * b).sum();
    }
}

impl Smoother {
    fn new(lag: usize, se: bool, rings: usize) -> Self {
        Self {
            lag,
            se,
            rings: vec![Ring::default(); rings],
            next: None,
            clock: 0.0,
            row: Work::default(),
        }
    }

    /// Whether a loaded smoother has the shape its filter's settings give
    /// it: `k` slots, `m` targets, `rings` covariances (one, holding every
    /// target, under `share_p`), an anchor under `standardize`.
    pub(super) fn is_valid(&self, k: usize, m: usize, rings: usize, anchored: bool) -> bool {
        let per = if rings == 1 { m } else { 1 };
        let tri_len = if self.se { tri(k) } else { 0 };
        let anchor = if anchored { 3 * k } else { 0 };
        let of_ok =
            |d: &Described| d.sized.len() == k && d.held.len() == per && d.anchor.len() == anchor;
        self.lag >= 1
            && self.clock.is_finite()
            && self.rings.len() == rings
            && self.rings.iter().all(|r| {
                r.slots.len() < self.lag
                    && r.slots.iter().all(|s| {
                        of_ok(&s.of)
                            && s.c.len() == per * k
                            && (s.g.is_empty() || s.g.len() == k * k)
                            && s.kk.len() == tri_len
                    })
                    && r.pending.as_ref().is_none_or(|p| {
                        of_ok(&p.of)
                            && p.b.len() == per * k
                            && p.a.len() == k * k
                            && p.p.len() == tri_len
                    })
            })
    }

    /// A row begins: nothing emitted yet, and its stamp the one handed, or
    /// the model's own clock.
    fn begin(&mut self, d_clock: f64, m: usize, k: usize) {
        self.clock += d_clock;
        self.row.stamp = Some(self.next.take().unwrap_or(Stamp::from(self.clock)));
        let out = &mut self.row.out;
        out.emitted.clear();
        out.emitted.resize(m, false);
        out.stamp.clear();
        out.stamp.resize(m, None);
        out.held.clear();
        out.held.resize(m, SMOOTH_STOOD);
        out.coef.clear();
        out.coef.resize(m * k, f64::NAN);
        out.var.clear();
        if self.se {
            out.var.resize(m * k, f64::NAN);
        }
    }

    /// The row's transition `Phi_a` (with the coupling `g`) applied to each
    /// pending `A`'s rows, as the filter applies it to `b`.
    pub(super) fn transit(&mut self, phi: &[f64], g: Option<&[f64]>) {
        let k = phi.len();
        for p in self.live_pending() {
            for row in p.a.chunks_exact_mut(k) {
                super::transit(row, phi, g);
            }
        }
    }

    /// A re-map's `Mᵀ` applied to each pending `A`'s rows, as the filter's
    /// re-map applies it to `b`.
    pub(super) fn remap(&mut self, a: &[f64], c: &[f64], off: usize) {
        let k = a.len();
        for p in self.live_pending() {
            for row in p.a.chunks_exact_mut(k) {
                if off == 1 {
                    let mut v0 = row[0];
                    for i in 1..k {
                        v0 += c[i] * row[i];
                    }
                    row[0] = v0;
                }
                for (v, ai) in row[off..].iter_mut().zip(&a[off..]) {
                    *v *= ai;
                }
            }
        }
    }

    fn live_pending(&mut self) -> impl Iterator<Item = &mut Pending> {
        self.rings
            .iter_mut()
            .filter_map(|r| r.pending.as_mut())
            .filter(|p| !p.broken)
    }

    /// The filter has not continued itself: covariance `pi`'s chain, or
    /// every covariance's, is cut after its newest observation.
    pub(super) fn cut(&mut self, pi: Option<usize>) {
        for (i, r) in self.rings.iter_mut().enumerate() {
            if pi.is_none_or(|pi| pi == i)
                && let Some(p) = &mut r.pending
            {
                p.broken = true;
            }
        }
    }

    /// `G_n = A (P⁻)⁻¹` on `P⁻`'s sized slots, or `None` at a break.
    fn gain(&mut self, pend: &Pending, prior: &[f64], k: usize) -> Option<Vec<f64>> {
        if pend.broken {
            return None;
        }
        let w = &mut self.row;
        w.idx.clear();
        w.idx.extend((0..k).filter(|&i| prior[i * k + i] != 0.0));
        let n = w.idx.len();
        w.chol.clear();
        w.chol.resize(n * n, 0.0);
        for (r, &i) in w.idx.iter().enumerate() {
            for (c, &j) in w.idx.iter().enumerate().take(r + 1) {
                w.chol[r * n + c] = prior[i * k + j];
            }
        }
        if !cholesky(&mut w.chol, n) {
            return None;
        }
        let mut g = vec![0.0; k * k];
        for (r, a_row) in pend.a.chunks_exact(k).enumerate() {
            if a_row.iter().all(|v| *v == 0.0) {
                continue;
            }
            w.y.clear();
            w.y.extend(w.idx.iter().map(|&s| a_row[s]));
            solve_factored(&w.chol, n, &mut w.y);
            for (v, &s) in w.y.iter().zip(&w.idx) {
                if !v.is_finite() {
                    return None;
                }
                g[r * k + s] = *v;
            }
        }
        Some(g)
    }

    /// Covariance `pi`'s next observation has its prior, `P⁻` and each
    /// target's `b⁻` (one after another): its pending pair becomes a slot.
    fn link(&mut self, pi: usize, prior: &[f64], b_prior: &[f64], k: usize) {
        let Some(pend) = self.rings[pi].pending.take() else {
            return;
        };
        let slot = match self.gain(&pend, prior, k) {
            Some(g) => {
                let mut c = pend.b;
                for (ct, bt) in c.chunks_exact_mut(k).zip(b_prior.chunks_exact(k)) {
                    for (r, cr) in ct.iter_mut().enumerate() {
                        let gb: f64 = g[r * k..(r + 1) * k]
                            .iter()
                            .zip(bt)
                            .map(|(a, b)| a * b)
                            .sum();
                        *cr -= gb;
                    }
                }
                // K = P_n − G P⁻ Gᵀ = P_n − A Gᵀ, since G P⁻ = A on the
                // sized slots and both are 0 off them.
                let mut kk = pend.p;
                if self.se {
                    let mut at = 0;
                    for i in 0..k {
                        let ai = &pend.a[i * k..(i + 1) * k];
                        for j in 0..=i {
                            let gj = &g[j * k..(j + 1) * k];
                            kk[at] -= ai.iter().zip(gj).map(|(a, b)| a * b).sum::<f64>();
                            at += 1;
                        }
                    }
                }
                Slot {
                    of: pend.of,
                    c,
                    g,
                    kk,
                }
            }
            None => Slot {
                of: pend.of,
                c: pend.b,
                g: vec![],
                kk: pend.p,
            },
        };
        self.rings[pi].slots.push_back(slot);
    }

    /// The pass back over `slots` (newest first) from `m` (each target's,
    /// one after another) and, with the variances, `v` (`k x k`), left in
    /// `w.m` and `w.v`.
    fn pass<'a>(
        w: &mut Work,
        slots: impl Iterator<Item = &'a Slot>,
        k: usize,
        se: bool,
        start_v: Option<&[f64]>,
    ) {
        if let Some(v) = start_v {
            w.v.clear();
            w.v.extend_from_slice(v);
        }
        for slot in slots {
            if slot.g.is_empty() {
                w.m.copy_from_slice(&slot.c);
                if se {
                    full_of(&slot.kk, k, &mut w.v);
                }
                continue;
            }
            w.tmp.resize(k, 0.0);
            for (mt, ct) in w.m.chunks_exact_mut(k).zip(slot.c.chunks_exact(k)) {
                mat_vec(&slot.g, mt, &mut w.tmp);
                for ((mi, ci), gi) in mt.iter_mut().zip(ct).zip(&w.tmp) {
                    *mi = ci + gi;
                }
            }
            if se {
                // gv = G V, then V = K + gv Gᵀ.
                w.gv.clear();
                w.gv.resize(k * k, 0.0);
                for i in 0..k {
                    let gi = &slot.g[i * k..(i + 1) * k];
                    for j in 0..k {
                        w.gv[i * k + j] = (0..k).map(|l| gi[l] * w.v[l * k + j]).sum();
                    }
                }
                full_of(&slot.kk, k, &mut w.v);
                for i in 0..k {
                    let gvi = &w.gv[i * k..(i + 1) * k];
                    for j in 0..k {
                        let gj = &slot.g[j * k..(j + 1) * k];
                        w.v[i * k + j] += gvi.iter().zip(gj).map(|(a, b)| a * b).sum::<f64>();
                    }
                }
            }
        }
    }
}

impl Kalman {
    /// This filter with a fixed-lag smoother of `lag` observations
    /// (`kalman/smooth.rs`), keeping the smoothed variances when `se`. The
    /// smoother starts empty. Refused at a lag of 0, and under the square
    /// root's covariance form, whose `P` it would have to form from the
    /// factor and so lose the digits that form keeps.
    pub fn with_smoother(mut self, lag: usize, se: bool) -> Result<Self, String> {
        if lag == 0 {
            return Err(
                "kalman: smooth_lag must be at least 1 observation, but it is 0. \
                        Set it to the number of observations after a row to smooth it with, \
                        or leave it out for no smoother."
                    .into(),
            );
        }
        if self.factor() {
            return Err(
                "kalman: smooth_lag needs covariance_form=\"covariance\", but it is \
                 \"square_root\": the smoother reads the covariance itself, which the square \
                 root's form does not hold. Leave covariance_form at its default to smooth."
                    .into(),
            );
        }
        self.smoother = Some(Box::new(Smoother::new(lag, se, self.p.len())));
        Ok(self)
    }

    /// The smoother's lag in observations, if there is one.
    pub fn smooth_lag(&self) -> Option<usize> {
        self.smoother.as_ref().map(|s| s.lag)
    }

    /// Whether the smoother keeps its variances.
    pub fn smooth_se(&self) -> bool {
        self.smoother.as_ref().is_some_and(|s| s.se)
    }

    /// What the last row emitted ([`Smoothed`]), if there is a smoother.
    pub fn smoothed(&self) -> Option<&Smoothed> {
        self.smoother.as_ref().map(|s| &s.row.out)
    }

    /// Each covariance's last observations, the ones whose lag the stream
    /// has not yet completed, smoothed with every row so far, oldest first,
    /// read from the state: what the rows to come would emit if they
    /// stopped now, the newest at its filtered value. `None` without a
    /// smoother.
    pub fn smoothed_tail(&self) -> Option<Vec<SmoothedRow>> {
        let s = self.smoother.as_ref()?;
        let k = self.cfg.k_total();
        let mut w = Work::default();
        let mut out = Vec::new();
        for (pi, ring) in s.rings.iter().enumerate() {
            let Some(pend) = &ring.pending else {
                continue;
            };
            let targets = self.smooth_targets(pi);
            let mut v = Vec::new();
            if s.se {
                full_of(&pend.p, k, &mut v);
            }
            let mut rows = Vec::with_capacity(ring.slots.len() + 1);
            w.m.clear();
            w.m.extend_from_slice(&pend.b);
            rows.push(self.smooth_rows(&pend.of, &w.m, s.se.then_some(&v[..]), targets.clone()));
            w.v = v;
            for slot in ring.slots.iter().rev() {
                Smoother::pass(&mut w, std::iter::once(slot), k, s.se, None);
                let v = s.se.then_some(&w.v[..]);
                rows.push(self.smooth_rows(&slot.of, &w.m, v, targets.clone()));
            }
            out.extend(rows.into_iter().rev().flatten());
        }
        Some(out)
    }

    /// The targets covariance `pi` holds the coefficients of.
    pub(super) fn smooth_targets(&self, pi: usize) -> std::ops::Range<usize> {
        if self.cfg.share_p {
            0..self.cfg.n_targets
        } else {
            pi..pi + 1
        }
    }

    /// Whether this row observes covariance `pi`'s target (any target under
    /// `share_p`) at a positive weight, as `step` charges its noise.
    fn smooth_informs(&self, pi: usize, y: &[Option<f64>], weight: f64) -> bool {
        weight > 0.0 && self.smooth_targets(pi).any(|j| y[j].is_some())
    }

    /// A row begins (`step_into`, before anything else, a refused row
    /// included).
    pub(super) fn smooth_begin(&mut self, d_clock: f64) {
        let (m, k) = (self.cfg.n_targets, self.cfg.k_total());
        if let Some(s) = self.smoother.as_deref_mut() {
            s.begin(d_clock, m, k);
        }
    }

    /// Each observed covariance's prior is formed, before the row's
    /// update: its pending pair becomes a slot.
    pub(super) fn smooth_link(&mut self, y: &[Option<f64>], weight: f64) {
        let k = self.cfg.k_total();
        let Some(mut s) = self.smoother.take() else {
            return;
        };
        let mut b_prior = std::mem::take(&mut s.row.tmp);
        for pi in 0..self.p.len() {
            if self.smooth_informs(pi, y, weight) {
                b_prior.clear();
                for j in self.smooth_targets(pi) {
                    b_prior.extend_from_slice(&self.beta[j]);
                }
                #[cfg(test)]
                s.row.log.push(Logged {
                    pi,
                    prior: true,
                    p: self.p[pi].clone(),
                    b: b_prior.clone(),
                    anchor: self.smooth_anchor(),
                    stamp: s.row.stamp.expect("a row's stamp"),
                });
                s.link(pi, &self.p[pi], &b_prior, k);
            }
        }
        s.row.tmp = b_prior;
        self.smoother = Some(s);
    }

    /// After the row's update and before its moments move: each observed
    /// covariance whose ring holds `L` slots emits its oldest observation,
    /// and the row's pair becomes the pending one. `ready` is whether
    /// `min_weight` let the row's prediction out.
    pub(super) fn smooth_observe(&mut self, y: &[Option<f64>], weight: f64, ready: bool) {
        let k = self.cfg.k_total();
        let Some(mut s) = self.smoother.take() else {
            return;
        };
        let se = s.se;
        for pi in 0..self.p.len() {
            if !self.smooth_informs(pi, y, weight) {
                continue;
            }
            let targets = self.smooth_targets(pi);
            if s.rings[pi].slots.len() == s.lag {
                let w = &mut s.row;
                w.m.clear();
                for j in targets.clone() {
                    w.m.extend_from_slice(&self.beta[j]);
                }
                let start = se.then_some(&self.p[pi][..]);
                Smoother::pass(w, s.rings[pi].slots.iter().rev(), k, se, start);
                let oldest = s.rings[pi].slots.pop_front().expect("a full ring");
                let rows =
                    self.smooth_rows(&oldest.of, &w.m, se.then_some(&w.v[..]), targets.clone());
                let out = &mut w.out;
                for r in rows {
                    let j = r.target;
                    out.emitted[j] = true;
                    out.stamp[j] = Some(r.stamp);
                    out.held[j] = r.held;
                    out.coef[j * k..(j + 1) * k].copy_from_slice(&r.coef);
                    if se {
                        out.var[j * k..(j + 1) * k].copy_from_slice(&r.var);
                    }
                }
            }
            let p = &self.p[pi];
            let mut b = Vec::with_capacity(targets.len() * k);
            for j in targets.clone() {
                b.extend_from_slice(&self.beta[j]);
            }
            let held = targets
                .map(|j| {
                    if self.wj[j] <= 0.0 {
                        SMOOTH_UNSEEN
                    } else if !ready {
                        SMOOTH_MIN_WEIGHT
                    } else {
                        SMOOTH_STOOD
                    }
                })
                .collect();
            let anchor = self.smooth_anchor();
            #[cfg(test)]
            s.row.log.push(Logged {
                pi,
                prior: false,
                p: p.clone(),
                b: b.clone(),
                anchor: anchor.clone(),
                stamp: s.row.stamp.expect("a row's stamp"),
            });
            let of = Described {
                stamp: s.row.stamp.unwrap_or(Stamp::from(s.clock)),
                sized: (0..k).map(|i| p[i * k + i] != 0.0).collect(),
                held,
                anchor,
            };
            s.rings[pi].pending = Some(Pending {
                of,
                b,
                a: p.clone(),
                p: if se { lower_of(p, k) } else { vec![] },
                broken: false,
            });
        }
        self.smoother = Some(s);
    }

    /// The smoothed pair `m` (each of `targets`' coefficients, one after
    /// another) and `v` of the observation `of` describes, read out in the
    /// caller's units: as `coefficients` and `coef_variance` read the
    /// filter's, through that observation's anchor, the variances times
    /// each target's noise as it stands.
    fn smooth_rows(
        &self,
        of: &Described,
        m: &[f64],
        v: Option<&[f64]>,
        targets: std::ops::Range<usize>,
    ) -> Vec<SmoothedRow> {
        let k = self.cfg.k_total();
        let off = usize::from(self.cfg.fit_intercept);
        let (hi, lo, sc) = if of.anchor.is_empty() {
            (&[][..], &[][..], &[][..])
        } else {
            (&of.anchor[..k], &of.anchor[k..2 * k], &of.anchor[2 * k..])
        };
        let first = targets.start;
        targets
            .map(|j| {
                let t = j - first;
                let held = of.held[t];
                let mut coef = vec![f64::NAN; k];
                let mut var = vec![f64::NAN; if v.is_some() { k } else { 0 }];
                if held == SMOOTH_STOOD {
                    let mt = &m[t * k..(t + 1) * k];
                    if sc.is_empty() {
                        coef.copy_from_slice(mt);
                    } else {
                        for i in off..k {
                            coef[i] = mt[i] / sc[i];
                        }
                        if off == 1 {
                            let mut b0 = mt[0];
                            for i in off..k {
                                b0 -= coef[i] * hi[i];
                                b0 -= coef[i] * lo[i];
                            }
                            coef[0] = b0;
                        }
                    }
                    if let Some(v) = v {
                        let ratio = self.own_noise_ratio(j);
                        for i in 0..k {
                            var[i] = if sc.is_empty() || i < off {
                                v[i * k + i]
                            } else {
                                v[i * k + i] / (sc[i] * sc[i])
                            };
                        }
                        if off == 1 && !sc.is_empty() {
                            let u: Vec<f64> = (0..k)
                                .map(|i| {
                                    if i == 0 {
                                        1.0
                                    } else {
                                        -(hi[i] + lo[i]) / sc[i]
                                    }
                                })
                                .collect();
                            let mut quad = 0.0;
                            for i in 0..k {
                                let acc: f64 = (0..k).map(|l| v[i * k + l] * u[l]).sum();
                                quad += u[i] * acc;
                            }
                            var[0] = crate::solve::clamp_rounding(quad);
                        }
                        for vi in &mut var {
                            *vi *= ratio;
                        }
                    }
                    for (i, ci) in coef.iter_mut().enumerate() {
                        if !of.sized[i] {
                            *ci = f64::NAN;
                            if let Some(vi) = var.get_mut(i) {
                                *vi = f64::NAN;
                            }
                        }
                    }
                }
                SmoothedRow {
                    target: j,
                    stamp: of.stamp,
                    held,
                    coef,
                    var,
                }
            })
            .collect()
    }

    /// The anchor as it stands, `[means' high parts, low parts, scales]`,
    /// empty without `standardize`.
    fn smooth_anchor(&self) -> Vec<f64> {
        if self.cfg.standardize {
            [&self.anchor_hi[..], &self.anchor_lo, &self.anchor_scale].concat()
        } else {
            vec![]
        }
    }

    /// The pairs the smoother read, in order.
    #[cfg(test)]
    pub(super) fn smooth_log(&self) -> &[Logged] {
        &self.smoother.as_ref().expect("a smoother").row.log
    }

    /// The stamp of the next row stepped (`OnlineModel::stamp_next`).
    pub(super) fn smooth_stamp(&mut self, stamp: Stamp) {
        if let Some(s) = self.smoother.as_deref_mut() {
            s.next = Some(stamp);
        }
    }
}
