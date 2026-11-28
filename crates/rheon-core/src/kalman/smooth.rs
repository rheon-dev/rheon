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
//! keeps what reads its observation out: its row's clock label (the
//! clock column's value as the caller hands it, or its row number), which
//! slots were
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
//! the covariance: the described observation's clock label, its coefficients in
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
use crate::ClockValue;

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
/// label, whether its value stands ([`SMOOTH_STOOD`]) or why not, its
/// smoothed coefficients in the caller's units (`k_total` a target, NaN
/// where there is no value) and, with the variances asked for, their
/// variances laid out the same way.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Smoothed {
    pub emitted: Vec<bool>,
    pub label: Vec<Option<ClockValue>>,
    pub held: Vec<u8>,
    pub coef: Vec<f64>,
    pub var: Vec<f64>,
}

/// One observation's smoothed value given every row so far, as
/// [`Kalman::smoothed_tail`] reads it from the state: the target, the
/// label, whether it stands, the coefficients and (with the variances
/// asked for) their variances, NaN where there is no value.
#[derive(Debug, Clone, PartialEq)]
pub struct SmoothedRow {
    pub target: usize,
    pub label: ClockValue,
    pub held: u8,
    pub coef: Vec<f64>,
    pub var: Vec<f64>,
}

/// What reads an observation out: its label, which slots the filter had
/// sized (a variance not 0), whether each target's prediction stood
/// ([`SMOOTH_STOOD`] and the rest), and the anchor of its coordinates,
/// `[means' high parts, low parts, scales]`, empty without `standardize`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Described {
    label: ClockValue,
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
/// per covariance, the clock label handed for the next row and the model's
/// own clock for a row handed none.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(super) struct Smoother {
    lag: usize,
    se: bool,
    rings: Vec<Ring>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    next: Option<ClockValue>,
    clock: f64,
    #[serde(skip)]
    row: Work,
}

/// A row's emission and the pass's working space: never saved, so two
/// smoothers are equal whatever either holds here.
#[derive(Debug, Clone, Default)]
struct Work {
    label: Option<ClockValue>,
    /// Which rings took a pending pair on this row: what
    /// [`Kalman::hold_smoothed`] may mark.
    fresh: Vec<bool>,
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
/// the anchor as it stood and the row's label.
#[cfg(test)]
#[derive(Debug, Clone)]
pub(super) struct Logged {
    pub(super) pi: usize,
    pub(super) prior: bool,
    pub(super) p: Vec<f64>,
    pub(super) b: Vec<f64>,
    pub(super) anchor: Vec<f64>,
    pub(super) label: ClockValue,
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

/// `a · b` over four running sums, the products taken in index order
/// within each, summed in a fixed order: the same bits on every platform,
/// and four chains of adds that overlap.
fn dot(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    let b = &b[..a.len()];
    let mut acc = [0.0; 4];
    let ((a4, ra), (b4, rb)) = (a.as_chunks::<4>(), b.as_chunks::<4>());
    for (x, y) in a4.iter().zip(b4) {
        for l in 0..4 {
            acc[l] += x[l] * y[l];
        }
    }
    let mut tail = 0.0;
    for (x, y) in ra.iter().zip(rb) {
        tail += x * y;
    }
    ((acc[0] + acc[1]) + (acc[2] + acc[3])) + tail
}

/// Scalar fallback for two dot products sharing one pass over `a`.
#[inline(always)]
fn dot2_scalar(a: &[f64], b0: &[f64], b1: &[f64]) -> [f64; 2] {
    [dot(a, b0), dot(a, b1)]
}

/// Dispatch a `pulp::WithSimd` kernel using 4-lane AVX2 on x86-64 when
/// available (matching [`crate::sgd::x86_v3`]) and `pulp::Arch::new()`
/// otherwise (2-lane NEON on aarch64, SSE2 on older x86-64).
#[inline(always)]
fn dispatch_simd<W: pulp::WithSimd>(op: W) -> W::Output {
    #[cfg(target_arch = "x86_64")]
    {
        if let Some(simd) = crate::sgd::x86_v3() {
            return pulp::Simd::vectorize(simd, op);
        }
    }
    pulp::Arch::new().dispatch(op)
}

/// SIMD-accelerated `dot(a, b)` using strictly unfused `mul_f64s` + `add_f64s`
/// on the 4-element prefix `a[..n & !3]`, bit-identical to [`dot`] for every
/// length `n` on both 4-lane (AVX2) and 2-lane (NEON / SSE2) backends.
#[inline(always)]
fn dot_simd<S: pulp::Simd>(simd: S, a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    debug_assert_eq!(b.len(), n);
    let n4 = n & !3;
    let (ah, at) = (&a[..n4], &a[n4..]);
    let (bh, bt) = (&b[..n4], &b[n4..]);
    let acc = if S::F64_LANES == 4 {
        let va = S::as_simd_f64s(ah).0;
        let vb = S::as_simd_f64s(bh).0;
        let mut c = simd.splat_f64s(0.0);
        for (&x, &y) in va.iter().zip(vb) {
            c = simd.add_f64s(c, simd.mul_f64s(x, y));
        }
        let mut out = [0.0; 4];
        S::as_mut_simd_f64s(&mut out).0[0] = c;
        out
    } else if S::F64_LANES == 2 {
        let va = S::as_simd_f64s(ah).0.as_chunks::<2>().0;
        let vb = S::as_simd_f64s(bh).0.as_chunks::<2>().0;
        let mut c0 = simd.splat_f64s(0.0);
        let mut c1 = simd.splat_f64s(0.0);
        for (&x, &y) in va.iter().zip(vb) {
            c0 = simd.add_f64s(c0, simd.mul_f64s(x[0], y[0]));
            c1 = simd.add_f64s(c1, simd.mul_f64s(x[1], y[1]));
        }
        let mut out = [0.0; 4];
        let s = S::as_mut_simd_f64s(&mut out).0;
        s[0] = c0;
        s[1] = c1;
        out
    } else {
        return dot(a, b);
    };
    let mut tail = 0.0;
    for (&x, &y) in at.iter().zip(bt) {
        tail += x * y;
    }
    ((acc[0] + acc[1]) + (acc[2] + acc[3])) + tail
}

/// SIMD-accelerated `[dot(a, b0), dot(a, b1)]` sharing one pass over `a`,
/// bit-identical to two calls to [`dot`] with unfused `mul_f64s` + `add_f64s`.
#[inline(always)]
fn dot2_simd<S: pulp::Simd>(simd: S, a: &[f64], b0: &[f64], b1: &[f64]) -> [f64; 2] {
    let n = a.len();
    debug_assert_eq!(b0.len(), n);
    debug_assert_eq!(b1.len(), n);
    let n4 = n & !3;
    let (ah, at) = (&a[..n4], &a[n4..]);
    let (b0h, b0t) = (&b0[..n4], &b0[n4..]);
    let (b1h, b1t) = (&b1[..n4], &b1[n4..]);
    let (s0, s1) = if S::F64_LANES == 4 {
        let va = S::as_simd_f64s(ah).0;
        let vb0 = S::as_simd_f64s(b0h).0;
        let vb1 = S::as_simd_f64s(b1h).0;
        let mut c0 = simd.splat_f64s(0.0);
        let mut c1 = simd.splat_f64s(0.0);
        for ((&x, &y0), &y1) in va.iter().zip(vb0).zip(vb1) {
            c0 = simd.add_f64s(c0, simd.mul_f64s(x, y0));
            c1 = simd.add_f64s(c1, simd.mul_f64s(x, y1));
        }
        let mut o0 = [0.0; 4];
        let mut o1 = [0.0; 4];
        S::as_mut_simd_f64s(&mut o0).0[0] = c0;
        S::as_mut_simd_f64s(&mut o1).0[0] = c1;
        (o0, o1)
    } else if S::F64_LANES == 2 {
        let va = S::as_simd_f64s(ah).0.as_chunks::<2>().0;
        let vb0 = S::as_simd_f64s(b0h).0.as_chunks::<2>().0;
        let vb1 = S::as_simd_f64s(b1h).0.as_chunks::<2>().0;
        let (mut c00, mut c01) = (simd.splat_f64s(0.0), simd.splat_f64s(0.0));
        let (mut c10, mut c11) = (simd.splat_f64s(0.0), simd.splat_f64s(0.0));
        for ((&x, &y0), &y1) in va.iter().zip(vb0).zip(vb1) {
            c00 = simd.add_f64s(c00, simd.mul_f64s(x[0], y0[0]));
            c01 = simd.add_f64s(c01, simd.mul_f64s(x[1], y0[1]));
            c10 = simd.add_f64s(c10, simd.mul_f64s(x[0], y1[0]));
            c11 = simd.add_f64s(c11, simd.mul_f64s(x[1], y1[1]));
        }
        let mut o0 = [0.0; 4];
        let mut o1 = [0.0; 4];
        {
            let s = S::as_mut_simd_f64s(&mut o0).0;
            s[0] = c00;
            s[1] = c01;
        }
        {
            let s = S::as_mut_simd_f64s(&mut o1).0;
            s[0] = c10;
            s[1] = c11;
        }
        (o0, o1)
    } else {
        return dot2_scalar(a, b0, b1);
    };
    let (mut t0, mut t1) = (0.0, 0.0);
    for ((&x, &y0), &y1) in at.iter().zip(b0t).zip(b1t) {
        t0 += x * y0;
        t1 += x * y1;
    }
    [
        ((s0[0] + s0[1]) + (s0[2] + s0[3])) + t0,
        ((s1[0] + s1[1]) + (s1[2] + s1[3])) + t1,
    ]
}

/// `kk[at] -= dot(a[i], g[j])` for `0 <= j <= i < k`, dispatched once over
/// SIMD lanes.
struct LinkKkSimd<'a> {
    a: &'a [f64],
    g: &'a [f64],
    kk: &'a mut [f64],
    k: usize,
}

impl pulp::WithSimd for LinkKkSimd<'_> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: pulp::Simd>(self, simd: S) {
        let Self { a, g, kk, k } = self;
        let mut at = 0;
        for (i, ai) in a.chunks_exact(k).enumerate() {
            let kk_row = &mut kk[at..at + i + 1];
            let g_sub = &g[..(i + 1) * k];
            let (kk_pairs, kk_rem) = kk_row.as_chunks_mut::<2>();
            for (kr, gp) in kk_pairs.iter_mut().zip(g_sub.chunks_exact(2 * k)) {
                let (g0, g1) = gp.split_at(k);
                let d = dot2_simd(simd, ai, g0, g1);
                kr[0] -= d[0];
                kr[1] -= d[1];
            }
            if let Some(last) = kk_rem.first_mut() {
                *last -= dot_simd(simd, ai, &g[i * k..(i + 1) * k]);
            }
            at += i + 1;
        }
    }
}

/// One full backward RTS step (`m <- c + G m` and, when `se`, `gv = G V`,
/// `V = K + gv Gᵀ` mirrored symmetrically), dispatched once over SIMD lanes.
struct PassOneSimd<'a> {
    g: &'a [f64],
    c: &'a [f64],
    kk: &'a [f64],
    m: &'a mut [f64],
    tmp: &'a mut [f64],
    v: &'a mut [f64],
    gv: &'a mut [f64],
    k: usize,
    se: bool,
}

impl pulp::WithSimd for PassOneSimd<'_> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: pulp::Simd>(self, simd: S) {
        let Self {
            g,
            c,
            kk,
            m,
            tmp,
            v,
            gv,
            k,
            se,
        } = self;
        for (mt, ct) in m.chunks_exact_mut(k).zip(c.chunks_exact(k)) {
            for (o, row) in tmp.iter_mut().zip(g.chunks_exact(k)) {
                *o = dot_simd(simd, row, mt);
            }
            for ((mi, &ci), &gi) in mt.iter_mut().zip(ct).zip(tmp.iter()) {
                *mi = ci + gi;
            }
        }
        if !se {
            return;
        }
        let k2 = 2 * k;
        let last_v = &v[(k - 1) * k..k * k];
        for (gi, out) in g.chunks_exact(k).zip(gv.chunks_exact_mut(k)) {
            let (out_pairs, out_rem) = out.as_chunks_mut::<2>();
            for (o2, vp) in out_pairs.iter_mut().zip(v.chunks_exact(k2)) {
                let (v0, v1) = vp.split_at(k);
                let d = dot2_simd(simd, gi, v0, v1);
                o2[0] = d[0];
                o2[1] = d[1];
            }
            if let Some(last) = out_rem.first_mut() {
                *last = dot_simd(simd, gi, last_v);
            }
        }
        let mut at = 0;
        for (i, (gvi, v_full_row)) in gv.chunks_exact(k).zip(v.chunks_exact_mut(k)).enumerate() {
            let kk_row = &kk[at..at + i + 1];
            let v_row = &mut v_full_row[..i + 1];
            let g_sub = &g[..(i + 1) * k];
            let (v_pairs, v_rem) = v_row.as_chunks_mut::<2>();
            let (kk_pairs, _) = kk_row.as_chunks::<2>();
            for ((vr, kr), gp) in v_pairs.iter_mut().zip(kk_pairs).zip(g_sub.chunks_exact(k2)) {
                let (g0, g1) = gp.split_at(k);
                let d = dot2_simd(simd, gvi, g0, g1);
                vr[0] = kr[0] + d[0];
                vr[1] = kr[1] + d[1];
            }
            if let Some(last) = v_rem.first_mut() {
                *last = kk_row[i] + dot_simd(simd, gvi, &g[i * k..(i + 1) * k]);
            }
            at += i + 1;
        }
        // Mirror the lower triangle into the upper so `V` stays symmetric to
        // the bit for the next step's `G V` pass.
        for i in 1..k {
            for j in 0..i {
                v[j * k + i] = v[i * k + j];
            }
        }
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

    /// A row begins: nothing emitted yet, and its label the one handed, or
    /// the model's own clock.
    fn begin(&mut self, d_clock: f64, m: usize, k: usize) {
        self.clock += d_clock;
        self.row.fresh.clear();
        self.row.fresh.resize(self.rings.len(), false);
        self.row.label = Some(self.next.take().unwrap_or(ClockValue::F64(self.clock)));
        let out = &mut self.row.out;
        out.emitted.clear();
        out.emitted.resize(m, false);
        out.label.clear();
        out.label.resize(m, None);
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
    /// Marked `#[inline(never)]` without `#[cold]` so callers marked `#[cold]`
    /// do not force size-optimized codegen on the O(k^3) gain/Cholesky loops.
    #[inline(never)]
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
                if self.se && k > 0 {
                    dispatch_simd(LinkKkSimd {
                        a: &pend.a,
                        g: &g,
                        kk: &mut kk,
                        k,
                    });
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

    /// One backward RTS step over `slot`, non-generic and shared between
    /// `smoothed_tail` (hot/public) and `Smoother::pass` so LLVM never
    /// propagates `#[cold]` size-optimization onto the `O(k^3)` loops.
    #[inline(never)]
    fn pass_one(w: &mut Work, slot: &Slot, k: usize, se: bool) {
        if slot.g.is_empty() {
            w.m.copy_from_slice(&slot.c);
            if se {
                full_of(&slot.kk, k, &mut w.v);
            }
            return;
        }
        if k == 0 {
            return;
        }
        if w.tmp.len() != k {
            w.tmp.resize(k, 0.0);
        }
        if se {
            if w.gv.len() != k * k {
                w.gv.resize(k * k, 0.0);
            }
            if w.v.len() != k * k {
                w.v.resize(k * k, 0.0);
            }
        }
        dispatch_simd(PassOneSimd {
            g: &slot.g,
            c: &slot.c,
            kk: &slot.kk,
            m: &mut w.m,
            tmp: &mut w.tmp,
            v: &mut w.v,
            gv: &mut w.gv,
            k,
            se,
        });
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
            Self::pass_one(w, slot, k, se);
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
                Smoother::pass_one(&mut w, slot, k, s.se);
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
    #[cold]
    #[inline(never)]
    pub(super) fn smooth_begin(&mut self, d_clock: f64) {
        let (m, k) = (self.cfg.n_targets, self.cfg.k_total());
        if let Some(s) = self.smoother.as_deref_mut() {
            s.begin(d_clock, m, k);
        }
    }

    /// Each observed covariance's prior is formed, before the row's
    /// update: its pending pair becomes a slot.
    #[cold]
    #[inline(never)]
    pub(super) fn smooth_link(&mut self, y: &[Option<f64>], weight: f64) {
        let Some(mut s) = self.smoother.take() else {
            return;
        };
        for pi in 0..self.p.len() {
            if self.smooth_informs(pi, y, weight) {
                self.smooth_link_in(&mut s, pi);
            }
        }
        self.smoother = Some(s);
    }

    /// Link covariance `pi`'s pending pair to its current prior `(b⁻, P⁻)`.
    #[cold]
    #[inline(never)]
    pub(super) fn smooth_link_one(&mut self, pi: usize) {
        let Some(mut s) = self.smoother.take() else {
            return;
        };
        self.smooth_link_in(&mut s, pi);
        self.smoother = Some(s);
    }

    fn smooth_link_in(&self, s: &mut Smoother, pi: usize) {
        let k = self.cfg.k_total();
        let mut b_prior = std::mem::take(&mut s.row.tmp);
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
            label: s.row.label.expect("a row's label"),
        });
        s.link(pi, &self.p[pi], &b_prior, k);
        s.row.tmp = b_prior;
    }

    /// After the row's update and before its moments move: each observed
    /// covariance whose ring holds `L` slots emits its oldest observation,
    /// and the row's pair becomes the pending one. `ready` is whether
    /// `min_weight` let the row's prediction out.
    #[cold]
    #[inline(never)]
    pub(super) fn smooth_observe(&mut self, y: &[Option<f64>], weight: f64, ready: bool) {
        let Some(mut s) = self.smoother.take() else {
            return;
        };
        for pi in 0..self.p.len() {
            if self.smooth_informs(pi, y, weight) {
                self.smooth_observe_in(&mut s, pi, ready);
            }
        }
        self.smoother = Some(s);
    }

    /// Record covariance `pi`'s filtered `(b, P)` after its update and emit
    /// its oldest smoothed observation if its ring reached lag `L`.
    #[cold]
    #[inline(never)]
    pub(super) fn smooth_observe_one(&mut self, pi: usize, ready: bool) {
        let Some(mut s) = self.smoother.take() else {
            return;
        };
        self.smooth_observe_in(&mut s, pi, ready);
        self.smoother = Some(s);
    }

    #[inline(never)]
    fn smooth_observe_in(&self, s: &mut Smoother, pi: usize, ready: bool) {
        let k = self.cfg.k_total();
        let se = s.se;
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
            let rows = self.smooth_rows(&oldest.of, &w.m, se.then_some(&w.v[..]), targets.clone());
            let out = &mut w.out;
            for r in rows {
                let j = r.target;
                out.emitted[j] = true;
                out.label[j] = Some(r.label);
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
            label: s.row.label.expect("a row's label"),
        });
        let of = Described {
            label: s.row.label.unwrap_or(ClockValue::F64(s.clock)),
            sized: (0..k).map(|i| p[i * k + i] != 0.0).collect(),
            held,
            anchor,
        };
        s.row.fresh[pi] = true;
        s.rings[pi].pending = Some(Pending {
            of,
            b,
            a: p.clone(),
            p: if se { lower_of(p, k) } else { vec![] },
            broken: false,
        });
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
                    label: of.label,
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

    /// The caller's gate withheld target `j`'s prediction on this row for
    /// its `min_weight`: the observation the row added, if it added one,
    /// is not reported when its lag completes ([`SMOOTH_MIN_WEIGHT`]).
    pub(super) fn smooth_hold(&mut self, j: usize) {
        let pi = if self.cfg.share_p { 0 } else { j };
        let t = if self.cfg.share_p { j } else { 0 };
        if let Some(s) = self.smoother.as_deref_mut()
            && s.row.fresh.get(pi).copied().unwrap_or(false)
            && let Some(p) = &mut s.rings[pi].pending
            && p.of.held[t] == SMOOTH_STOOD
        {
            p.of.held[t] = SMOOTH_MIN_WEIGHT;
        }
    }

    /// The bytes a smoother of `lag` observations takes in a state file at
    /// most, its variances kept with `se`, for a filter of `k` coefficients
    /// (the intercept's included) and `m` targets, one covariance for all
    /// of them under `share_p`, with an anchor under `standardize`: per
    /// covariance, `lag` slots (the pending pair among them) of `G` (`k²`),
    /// each target's `c` (`k`), `K`'s triangle with `se` and the anchor
    /// (`3k`), nine bytes a number as the file writes one, the sized and
    /// held marks, and 128 bytes for the clock label and the slot's names
    /// and lengths. What a spec holds to its `max_state_bytes` before a row
    /// is read.
    pub fn smoother_bytes(
        k: usize,
        m: usize,
        share_p: bool,
        standardize: bool,
        lag: usize,
        se: bool,
    ) -> u64 {
        let (rings, per) = if share_p { (1, m) } else { (m, 1) };
        let (k, per, lag) = (k as u64, per as u64, lag as u64);
        let floats = per * k
            + k * k
            + if se { k * (k + 1) / 2 } else { 0 }
            + if standardize { 3 * k } else { 0 };
        let slot = 9 * floats + k + per + 128;
        (rings as u64).saturating_mul(lag.saturating_mul(slot))
    }

    /// The label of the next row stepped (`OnlineModel::label_next`).
    pub(super) fn smooth_label(&mut self, label: ClockValue) {
        if let Some(s) = self.smoother.as_deref_mut() {
            s.next = Some(label);
        }
    }
}
