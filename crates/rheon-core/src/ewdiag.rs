//! `EwDiag`: the diagonal of [`EwCov`](crate::EwCov) — exponentially weighted means and
//! variances of a vector stream, O(k) a row.
//!
//! The same weighted Welford recursion as `ewcov.rs`, operation for
//! operation, minus the co-moments:
//!
//! ```text
//! W'    = lam * W + w
//! a     = lam * W / W'        b = w / W'        (a + b = 1)
//! delta = x - m
//! m'    = m + b * delta
//! c'_i  = a * c_i + a * b * delta_i * delta_i
//! ```
//!
//! It exists for the models that standardize their features and read nothing
//! but the diagonal — `kalman` and `sgd` — which until schema 3 carried a full
//! `EwCov` for the purpose and paid `k²` co-moment updates a row for `k`
//! variances (docs/records/PERFORMANCE-LOG.md §13). The numbers are the same to the bit:
//! every diagonal entry is updated with exactly the arithmetic `EwCov` uses
//! for it, in the same order, so a model moved from one to the other kept
//! its outputs (`tests/model_contract.rs`, the goldens).

use serde::{Deserialize, Serialize};

/// EW means and centered variances of a `k`-vector: [`EwCov`](crate::EwCov) without the
/// off-diagonal co-moments. See the module docs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "EwDiagWire")]
pub struct EwDiag {
    k: usize,
    /// EW sum of weights (the `n_eff` count).
    w_sum: f64,
    /// EW mean vector, length `k`.
    m: Vec<f64>,
    /// EW **centered** second moments, length `k`: `EwCov`'s `c[i*k+i]`.
    c: Vec<f64>,
    /// What each mean leaves out: the mean is `m[i] + m_lo[i]`, as
    /// `EwCov`'s is ([`crate::comp`]; docs/PLAN.md task 101). One per mean.
    m_lo: Vec<f64>,
}

/// The wire layout, checked on the way in. `deny_unknown_fields` is load-
/// bearing: an `EwCov` carries these names among its own, and without it a
/// map-encoded `EwCov` would deserialize as an `EwDiag` with `k²`
/// "variances", silently. The shape check covers the array encoding the
/// same way.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct EwDiagWire {
    k: usize,
    w_sum: f64,
    m: Vec<f64>,
    c: Vec<f64>,
    m_lo: Vec<f64>,
}

impl TryFrom<EwDiagWire> for EwDiag {
    type Error = String;

    fn try_from(w: EwDiagWire) -> Result<Self, String> {
        if w.m.len() != w.k || w.c.len() != w.k || w.m_lo.len() != w.k {
            return Err(format!(
                "a diagnostic's saved means and variances have the wrong shape for its {} columns. Load an \
                 undamaged copy of the state file, or rebuild it from its input.",
                w.k
            ));
        }
        // A weight below 0 made every variance negative, and a value that
        // is no number never washes out (review 2026-11-23, STATE-3). The
        // centred sums are of products of one sign, so never below 0.
        let finite = |v: &[f64]| v.iter().all(|x| x.is_finite());
        if !(w.w_sum.is_finite() && w.w_sum >= 0.0)
            || !finite(&w.m)
            || !finite(&w.m_lo)
            || !w.c.iter().all(|c| c.is_finite() && *c >= 0.0)
        {
            return Err(
                "a diagnostic's saved weight, means or variances are not ones a stream writes. \
                 Load an undamaged copy of the state file, or rebuild it from its input."
                    .to_string(),
            );
        }
        Ok(Self {
            k: w.k,
            w_sum: w.w_sum,
            m: w.m,
            c: w.c,
            m_lo: w.m_lo,
        })
    }
}

impl EwDiag {
    pub fn new(k: usize) -> Self {
        Self {
            k,
            w_sum: 0.0,
            m: vec![0.0; k],
            c: vec![0.0; k],
            m_lo: vec![0.0; k],
        }
    }

    #[inline]
    pub fn k(&self) -> usize {
        self.k
    }

    /// The means as their pairs and the centred second moments, as slices
    /// `(m, m_lo, c)`: what [`Self::mean_pair`] and [`Self::var`] read one
    /// slot at a time, for a caller that runs a loop across the slots.
    #[inline]
    pub(crate) fn parts(&self) -> (&[f64], &[f64], &[f64]) {
        (&self.m, &self.m_lo, &self.c)
    }

    #[inline]
    pub fn n_eff(&self) -> f64 {
        self.w_sum
    }

    #[inline]
    pub fn mean(&self, i: usize) -> f64 {
        self.m[i]
    }

    /// The EW mean vector, length `k`.
    pub fn means(&self) -> &[f64] {
        &self.m
    }

    /// Mean `i` as the pair it is kept as, `(hi, lo)`: the mean is `hi +
    /// lo` ([`crate::comp`]). `kalman` reads how far a mean moved from two
    /// of them, `(hi' − hi) + (lo' − lo)`, which `hi' − hi` alone would
    /// round at a level (docs/PLAN.md task 206).
    #[inline]
    pub fn mean_pair(&self, i: usize) -> (f64, f64) {
        (self.m[i], self.m_lo[i])
    }

    /// `x`'s deviation from mean `i`, the pair's ([`crate::comp::dev`]).
    #[inline]
    pub fn deviation(&self, i: usize, x: f64) -> f64 {
        crate::comp::dev(x, self.m[i], self.m_lo[i])
    }

    /// Raw (uncentered) second moment `E_w[x_i²]`, reconstructed from the
    /// centered one like [`EwCov::raw`].
    #[inline]
    pub fn raw(&self, i: usize) -> f64 {
        self.c[i] + self.m[i] * self.m[i]
    }

    /// Centered variance, floored at zero against rounding, a NaN kept
    /// ([`EwCov::var`]).
    #[inline]
    pub fn var(&self, i: usize) -> f64 {
        crate::solve::clamp_rounding(self.c[i])
    }

    /// The moments as they will stand once one more row is admitted at unit
    /// weight after a decay of `lam`, read slot by slot without touching
    /// the accumulator. `sgd` standardises a row against these ([`crate::Sgd`],
    /// and docs/PLAN.md task 74 for why the moments include the row). The
    /// arithmetic is [`EwDiag::update`]'s, operation for operation, so what
    /// this reads for `x` is to the bit what `update(x, lam, 1.0)` leaves
    /// behind. Always well defined: the total weight is at least the row's
    /// own 1, even on an empty accumulator, where the row is the whole
    /// history and its variance is zero.
    #[inline]
    pub fn including(&self, lam: f64) -> Including<'_> {
        self.including_at(lam, 1.0)
    }

    /// [`Self::including`] with the row admitted at weight `s`: the moments
    /// as `update(x, lam, s)` leaves them. A row of no weight read against
    /// no history reads the moments as they stand (hard rule 9).
    #[inline]
    pub fn including_at(&self, lam: f64, s: f64) -> Including<'_> {
        let w_new = lam * self.w_sum + s;
        if w_new <= 0.0 {
            return Including {
                sc: self,
                a: 1.0,
                b: 0.0,
            };
        }
        Including {
            sc: self,
            a: lam * self.w_sum / w_new,
            b: s / w_new,
        }
    }

    /// One observation with decay factor `lam` (from [`crate::Decay::factor`])
    /// and row weight `w`. O(k), allocation-free, and the same guards as
    /// [`EwCov::update`]: a negative weight is a caller's bug (debug assert,
    /// no-op in release), and a row that leaves the total weight at zero
    /// changes nothing (hard rule 9).
    #[inline]
    pub fn update(&mut self, x: &[f64], lam: f64, w: f64) {
        debug_assert_eq!(x.len(), self.k);
        debug_assert!(
            w >= 0.0,
            "EwDiag::update requires a non-negative weight, got {w}"
        );
        if w < 0.0 {
            return;
        }
        let w_new = lam * self.w_sum + w;
        if w_new <= 0.0 {
            // Nothing carried and nothing added: `a` and `b` are 0/0, and no
            // moment moves. The weight is 0 either way -- it was, at the
            // head of a stream, and the decay took it, from 1075 half-lives
            // on, where it used to be kept (task 115 (c), PLAN §12).
            self.w_sum = w_new;
            return;
        }
        let a = lam * self.w_sum / w_new; // weight of the old statistics
        let b = w / w_new; // weight of the new point
        // A history the decay all but forgot: each mean steps from the row
        // back (`crate::comp::step_toward`), out of line, as rare as a gap
        // of 52 half-lives.
        if a < crate::comp::FORGOTTEN && b > 0.0 {
            self.update_forgotten(x, a, b);
            self.w_sum = w_new;
            return;
        }
        // Weighted Welford, as `EwCov::update` writes its diagonal: the
        // deviation is against the OLD mean, and the expression
        // `a * c + a * b * d * d` is kept in that order so the bits agree.
        //
        // Four slots a step in AVX2's 256-bit lanes where the CPU has them
        // (pulp's check at run time, and its safe calls: no `unsafe` here),
        // from `UPDATE_LANES_FROM` slots; the loop below everywhere else: on
        // x86-64 without AVX2, on any other architecture, below the width,
        // and on a row of no weight. Each slot is its own lane and runs the
        // loop's operations in the same order, each rounded once, with no
        // fused multiply-add, so the two paths agree to the bit.
        //
        // The wide rows leave by a tail call (`update_in_lanes`, which also
        // makes the CPU's check): with the call to pulp written here, the
        // narrow rows the loop takes paid for a stack frame, three saved
        // registers and the lanes' arguments on every row, 11-14% of the
        // update at 2 and 3 slots on x86-64.
        #[cfg(target_arch = "x86_64")]
        if update_takes_lanes(b, x.len()) {
            self.w_sum = w_new;
            return self.update_in_lanes(x, a, b);
        }
        let slots = self
            .c
            .iter_mut()
            .zip(self.m.iter_mut())
            .zip(self.m_lo.iter_mut());
        // A row of weight 0 leaves the moments as they are: `a` is 1 and `b`
        // is 0, so there is no step to take (`crate::comp::add` says why a
        // step of 0 is not taken either), and only the weight decays. The
        // test is made once, outside the loop: a test in the loop kept
        // `sgd`'s standardizing from vectorizing.
        if b > 0.0 {
            for (((ci, mi), lo), &xi) in slots.zip(x) {
                let d = crate::comp::dev(xi, *mi, *lo);
                *ci = a * *ci + a * b * d * d;
                crate::comp::add(mi, lo, b * d);
            }
        }
        self.w_sum = w_new;
    }

    /// [`Self::update`]'s loop on a row whose history keeps less than
    /// [`crate::comp::FORGOTTEN`] of the weight: each mean taken from the row
    /// back (review 2026-11-23, REG-7).
    #[cold]
    #[inline(never)]
    fn update_forgotten(&mut self, x: &[f64], a: f64, b: f64) {
        for (((ci, mi), lo), &xi) in self
            .c
            .iter_mut()
            .zip(self.m.iter_mut())
            .zip(self.m_lo.iter_mut())
            .zip(x)
        {
            let d = crate::comp::dev(xi, *mi, *lo);
            *ci = a * *ci + a * b * d * d;
            crate::comp::from_row(mi, lo, xi, d, a);
        }
    }

    /// [`Self::update`], the moments read against a history of at least
    /// `floor`, up to `had`, the weight the history had before the gap:
    /// where the history's decayed weight `lam·W` is below `H = min(had,
    /// floor)` (and there is a history, and the row has a weight), the old
    /// statistics take the share `H / (H + w)` in place of `lam·W / (lam·W +
    /// w)`. The weight kept is `lam·W + w` either way, so `n_eff` is the
    /// plain recursion's (hard rule 8); where the floor does not bind this
    /// is `update`, to the bit. `true` where it bound. Uncapped, a history
    /// of `1e-6` read against a row of weight 1 after a half-life took half
    /// the share, a million times its weight. `had` is the caller's, since
    /// rows of weight 0 inside the gap have aged `W` already.
    pub(crate) fn update_keeping(
        &mut self,
        x: &[f64],
        lam: f64,
        w: f64,
        floor: f64,
        had: f64,
    ) -> bool {
        let carried = lam * self.w_sum;
        let kept = floor.min(had);
        if !(self.w_sum > 0.0 && w > 0.0 && kept > carried) {
            self.update(x, lam, w);
            return false;
        }
        debug_assert_eq!(x.len(), self.k);
        let total = kept + w;
        let a = kept / total;
        let b = w / total;
        for (((ci, mi), lo), &xi) in self
            .c
            .iter_mut()
            .zip(self.m.iter_mut())
            .zip(self.m_lo.iter_mut())
            .zip(x)
        {
            let d = crate::comp::dev(xi, *mi, *lo);
            *ci = a * *ci + a * b * d * d;
            crate::comp::step_toward(mi, lo, xi, d, a, b);
        }
        self.w_sum = carried + w;
        true
    }

    /// [`Self::update`]'s moments across vector lanes, a lane a slot, with
    /// the weights `a` and `b` it computed for the row: AVX2's four where
    /// this CPU has them (pulp's check, made once), and one at a time where
    /// it has not, which is the loop's arithmetic in the loop's order.
    #[cfg(target_arch = "x86_64")]
    #[inline(never)]
    fn update_in_lanes(&mut self, x: &[f64], a: f64, b: f64) {
        let lanes = UpdateLanes {
            a,
            b,
            x,
            c: &mut self.c,
            m: &mut self.m,
            lo: &mut self.m_lo,
        };
        match crate::sgd::x86_v3() {
            Some(simd) => pulp::Simd::vectorize(simd, lanes),
            None => pulp::Simd::vectorize(pulp::Scalar::new(), lanes),
        }
    }

    /// [`Self::update`]'s loop alone, one slot at a time, whatever the CPU:
    /// the reference the lanes are held to.
    #[cfg(test)]
    pub(crate) fn update_scalar(&mut self, x: &[f64], lam: f64, w: f64) {
        let w_new = lam * self.w_sum + w;
        if w < 0.0 || w_new <= 0.0 {
            self.w_sum = w_new;
            return;
        }
        let a = lam * self.w_sum / w_new;
        let b = w / w_new;
        if b > 0.0 {
            for (((ci, mi), lo), &xi) in self
                .c
                .iter_mut()
                .zip(self.m.iter_mut())
                .zip(self.m_lo.iter_mut())
                .zip(x)
            {
                let d = crate::comp::dev(xi, *mi, *lo);
                *ci = a * *ci + a * b * d * d;
                crate::comp::step_toward(mi, lo, xi, d, a, b);
            }
        }
        self.w_sum = w_new;
    }

    /// [`Self::update`] of a one-slot accumulator with the row `[x]`, the
    /// same arithmetic in the same order, without the loop across slots
    /// that a slice of one costs a row (a target's own spread,
    /// [`crate::spread`]).
    #[inline]
    pub(crate) fn update_one(&mut self, x: f64, lam: f64, w: f64) {
        debug_assert_eq!(self.k, 1);
        debug_assert!(
            w >= 0.0,
            "EwDiag::update requires a non-negative weight, got {w}"
        );
        if w < 0.0 {
            return;
        }
        let w_new = lam * self.w_sum + w;
        if w_new <= 0.0 {
            self.w_sum = w_new;
            return;
        }
        let a = lam * self.w_sum / w_new;
        let b = w / w_new;
        if b > 0.0
            && let (Some(ci), Some(mi), Some(lo)) = (
                self.c.first_mut(),
                self.m.first_mut(),
                self.m_lo.first_mut(),
            )
        {
            let d = crate::comp::dev(x, *mi, *lo);
            *ci = a * *ci + a * b * d * d;
            crate::comp::step_toward(mi, lo, x, d, a, b);
        }
        self.w_sum = w_new;
    }
}

/// The width from which [`EwDiag::update`] takes the vector lanes: below it
/// the dispatch and the slots past the last full vector cost more than the
/// lanes save. Measured on x86-64 (Linux, an AVX2 CPU), the update alone
/// against the loop: +6 to +23% at 6 and 7 slots, about even at 4 and 5,
/// −25% at 8, −7% at 10 (two slots past the last full vector), −25 to −35%
/// at 12 to 24 and −40 to −45% from 32.
#[cfg(target_arch = "x86_64")]
const UPDATE_LANES_FROM: usize = 8;

/// Whether [`EwDiag::update`] asks for the lanes for a row of `n` slots
/// whose new point weighs `b`: from [`UPDATE_LANES_FROM`] slots, and only
/// when there is a step to take. The CPU decides how many
/// ([`EwDiag::update_in_lanes`]).
#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn update_takes_lanes(b: f64, n: usize) -> bool {
    b > 0.0 && n >= UPDATE_LANES_FROM
}

/// [`EwDiag::update`]'s loop across vector lanes, a lane a slot.
#[cfg(target_arch = "x86_64")]
struct UpdateLanes<'a> {
    a: f64,
    b: f64,
    x: &'a [f64],
    c: &'a mut [f64],
    m: &'a mut [f64],
    lo: &'a mut [f64],
}

#[cfg(target_arch = "x86_64")]
impl pulp::WithSimd for UpdateLanes<'_> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
        let Self { a, b, x, c, m, lo } = self;
        let n = x.len();
        let (c, m, lo) = (&mut c[..n], &mut m[..n], &mut lo[..n]);
        let (xv, xt) = S::as_simd_f64s(x);
        let (cv, ct) = S::as_mut_simd_f64s(c);
        let (mv, mt) = S::as_mut_simd_f64s(m);
        let (lv, lt) = S::as_mut_simd_f64s(lo);
        let ab = a * b;
        let (va, vb, vab) = (simd.splat_f64s(a), simd.splat_f64s(b), simd.splat_f64s(ab));
        for i in 0..xv.len() {
            let (hi, l) = (mv[i], lv[i]);
            // `comp::dev`: (x - hi) - lo.
            let d = simd.sub_f64s(simd.sub_f64s(xv[i], hi), l);
            // a * c + ((a * b) * d) * d
            cv[i] = simd.add_f64s(
                simd.mul_f64s(va, cv[i]),
                simd.mul_f64s(simd.mul_f64s(vab, d), d),
            );
            // `comp::add(hi, lo, b * d)`: Kahan's step of the mean.
            let y = simd.add_f64s(simd.mul_f64s(vb, d), l);
            let t = simd.add_f64s(hi, y);
            lv[i] = simd.sub_f64s(y, simd.sub_f64s(t, hi));
            mv[i] = t;
        }
        for (((ci, mi), l), &xi) in ct.iter_mut().zip(mt.iter_mut()).zip(lt.iter_mut()).zip(xt) {
            let d = crate::comp::dev(xi, *mi, *l);
            *ci = a * *ci + ab * d * d;
            crate::comp::add(mi, l, b * d);
        }
    }
}

/// [`EwDiag::including`]: the accumulator plus one row at unit weight, one
/// slot at a time.
#[derive(Debug, Clone, Copy)]
pub struct Including<'a> {
    sc: &'a EwDiag,
    /// Weight of the old statistics, `lam * W / (lam * W + 1)`.
    a: f64,
    /// Weight of the new point, `1 / (lam * W + 1)`.
    b: f64,
}

impl Including<'_> {
    /// The two weights of the admitted row, `(a, b)`, for a caller that runs
    /// [`Self::moments_of`]'s recursion across lanes of a vector: only
    /// `sgd`'s AVX2 lanes, on x86-64.
    #[cfg(target_arch = "x86_64")]
    #[inline]
    pub(crate) fn weights(&self) -> (f64, f64) {
        (self.a, self.b)
    }

    /// `(mean, var, dev)` of slot `i` with `x` admitted: `update`'s recursion
    /// for the slot, the variance floored at zero as [`EwDiag::var`] floors
    /// it, and `x`'s deviation from the mean it leaves, the pair's
    /// ([`crate::comp::dev`]; docs/PLAN.md task 101) -- which `x − mean`
    /// would round -- or, where the history's share `a` is below
    /// [`SMALL_SHARE`], `a·d`, `d` the row's deviation from the mean before
    /// it ([`Self::deviation_of`]).
    #[inline]
    pub fn moments(&self, i: usize, x: f64) -> (f64, f64, f64) {
        self.moments_of(self.sc.m[i], self.sc.m_lo[i], self.sc.c[i], x)
    }

    /// [`Self::moments`] from the slot's own parts as [`EwDiag::parts`]
    /// gives them -- its mean's pair `(hi, lo)` and its centred moment `c`
    /// -- for a caller that runs a loop across slices of them, where an
    /// index into each of the three vectors was a bounds check a slot.
    #[inline]
    pub(crate) fn moments_of(&self, hi: f64, lo: f64, c: f64, x: f64) -> (f64, f64, f64) {
        let d = crate::comp::dev(x, hi, lo);
        let c = self.a * c + self.a * self.b * d * d;
        let (mut hi, mut lo) = (hi, lo);
        crate::comp::step_toward(&mut hi, &mut lo, x, d, self.a, self.b);
        (
            hi,
            crate::solve::clamp_rounding(c),
            self.deviation_of(x, d, hi, lo),
        )
    }

    /// `x`'s deviation from the mean the admitted row leaves, `(hi, lo)`,
    /// `d` its deviation from the mean before it: the pair's,
    /// `(x − hi) − lo`, while the history's share `a` is at least
    /// [`SMALL_SHARE`], and `a·d` below it (and above 0).
    ///
    /// The two are the same number in exact arithmetic, the row at unit
    /// weight making `a + b = 1`. The pair's carries the rounding of the
    /// step `b·d` it took, about `2⁻⁵³·|d|`, which is `2⁻⁵³/a` of the
    /// deviation `a·d`: nothing on an ordinary row, and after a long clock
    /// gap, `a` below `2⁻⁵³`, larger than the deviation itself. `z` was
    /// then rounding noise over a scale of `sqrt(a)`, exactly 0 or as large
    /// as 1e75, and `sgd` divided its step by that scale: one gap of 500
    /// half-lives left slopes near 1e75 for good (review fable2, C1). `a·d`
    /// is good to a rounding of itself at any `a`. Below [`SMALL_SHARE`]
    /// the pair's is off by up to `2⁻⁴³` of the deviation, so the switch
    /// moves a deviation by no more than that, and no row of a stream whose
    /// gaps are shorter than about 14 half-lives -- every row its moments
    /// carry ten or more rows of history on -- is moved by a bit. At `a =
    /// 0` the row is the whole history and the pair's gives `+0`, which `a·d`
    /// would give as `-0` for a negative `d`.
    #[inline(always)]
    pub(crate) fn deviation_of(&self, x: f64, d: f64, hi: f64, lo: f64) -> f64 {
        if self.a < SMALL_SHARE && self.a > 0.0 {
            self.a * d
        } else {
            crate::comp::dev(x, hi, lo)
        }
    }
}

/// The history's share `a` below which [`Including::deviation_of`] reads
/// the row's deviation as `a·d`: `2⁻¹⁰`, where the pair's form is good to
/// `2⁻⁴³` of the deviation.
pub(crate) const SMALL_SHARE: f64 = 1.0 / 1024.0;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::EwCov;

    /// A history the decay forgot (`lam = 0`) leaves the row alone: the
    /// accumulator after it is a fresh one's after the same rows, to the bit,
    /// through `update` at every width (the lanes' among them on x86-64),
    /// `update_one` and the admitted row's view. A mean at `1e18` took a row
    /// at 1 to 0 (review 2026-11-23, REG-7). And a history of `2^-60` of the
    /// weight beside a mean of `2^60` takes a row at 1 to 2, `x + a·(m − x)`.
    #[test]
    fn a_history_the_decay_forgot_leaves_the_row_alone() {
        let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        let state = |m: &EwDiag| (bits(&m.m), bits(&m.m_lo), bits(&m.c), m.w_sum.to_bits());
        for k in [1, 3, 9, 13] {
            let row = |i: usize, v: f64| -> Vec<f64> {
                (0..k).map(|j| v + (i + j) as f64 * 0.37).collect()
            };
            let mut m = EwDiag::new(k);
            m.update(&row(0, 1e18), 1.0, 1.0);
            m.update(&row(1, -3e17), 0.5, 2.0);
            let mut fresh = EwDiag::new(k);
            let (x1, x2) = (row(2, 1.0), row(3, -2.5));
            let (mean, _, _) = m.including(0.0).moments(0, x1[0]);
            assert_eq!(
                mean.to_bits(),
                x1[0].to_bits(),
                "k {k}: the admitted row's mean"
            );
            m.update(&x1, 0.0, 1.5);
            fresh.update(&x1, 1.0, 1.5);
            assert_eq!(state(&m), state(&fresh), "k {k}");
            m.update(&x2, 0.5, 1.0);
            fresh.update(&x2, 0.5, 1.0);
            assert_eq!(state(&m), state(&fresh), "k {k}");
        }
        let mut one = EwDiag::new(1);
        one.update_one(1e18, 1.0, 1.0);
        one.update_one(1.0, 0.0, 1.0);
        assert_eq!(one.mean(0), 1.0);
        let mut near = EwDiag::new(1);
        near.update(&[2f64.powi(60)], 1.0, 1.0);
        near.update(&[1.0], 2f64.powi(-60), 1.0);
        assert_eq!(near.mean(0), 2.0);
    }

    /// The update through the lanes (on an x86-64 CPU with AVX2) and the
    /// scalar loop leave the same bits in every moment, at every width
    /// around the dispatch threshold and each lane remainder, over rows
    /// whose magnitudes span many binades, with zero-weight rows (the first
    /// of a stream among them) and every kind of decay the models use.
    /// Elsewhere -- without AVX2, or on another architecture, such as an
    /// arm64 Mac -- the dispatch takes the loop, and the test holds the loop
    /// to itself.
    #[test]
    fn update_lanes_match_the_scalar_loop_to_the_bit() {
        let mut s = 0x9e37_79b9_7f4a_7c15_u64;
        let mut draw = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64
        };
        let bits = |e: &EwDiag| {
            (
                e.w_sum.to_bits(),
                e.m.iter().map(|q| q.to_bits()).collect::<Vec<_>>(),
                e.c.iter().map(|q| q.to_bits()).collect::<Vec<_>>(),
                e.m_lo.iter().map(|q| q.to_bits()).collect::<Vec<_>>(),
            )
        };
        for k in 1..=40 {
            for &lam in &[1.0, 0.999, 0.9, 0.5] {
                let (mut v, mut r) = (EwDiag::new(k), EwDiag::new(k));
                for row in 0..400 {
                    let scale = 10f64.powi((draw() * 9.0) as i32 - 3);
                    let x: Vec<f64> = (0..k)
                        .map(|i| (draw() - 0.5) * scale + if i % 3 == 0 { 1e6 } else { 0.0 })
                        .collect();
                    let w = if row == 0 || row % 17 == 5 {
                        0.0
                    } else {
                        0.25 + draw()
                    };
                    v.update(&x, lam, w);
                    r.update_scalar(&x, lam, w);
                    assert_eq!(bits(&v), bits(&r), "k = {k}, lam = {lam}, row {row}");
                }
            }
        }
    }

    /// The lanes are taken from `UPDATE_LANES_FROM` slots and not one
    /// below, and never for a row of no weight (whose loop is skipped). The
    /// update in lanes moves every slot's moments, on every x86-64 CPU (on
    /// one lane at a time without AVX2), and leaves the weight to the caller.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn update_takes_the_lanes_from_their_width_with_a_step_to_take() {
        assert!(!update_takes_lanes(0.5, UPDATE_LANES_FROM - 1));
        assert!(update_takes_lanes(0.5, UPDATE_LANES_FROM));
        assert!(update_takes_lanes(1.0, 40));
        assert!(!update_takes_lanes(0.0, UPDATE_LANES_FROM));
        assert!(!update_takes_lanes(0.0, 40));
        for n in [UPDATE_LANES_FROM, UPDATE_LANES_FROM + 3] {
            let mut e = EwDiag::new(n);
            e.update_in_lanes(&vec![2.0; n], 0.0, 1.0);
            // From an empty accumulator at `a = 0`, `b = 1`: the mean is the
            // row, and its centred moment 0.
            assert!(e.m.iter().all(|&m| m == 2.0), "{n}: {:?}", e.m);
            assert!(e.c.iter().all(|&c| c == 0.0), "{n}: {:?}", e.c);
            assert_eq!(e.w_sum, 0.0, "{n}: the weight is the caller's to set");
        }
    }

    fn lcg(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }

    /// A row of no weight leaves the moments as they were, whatever its
    /// value, and only the weight ages (hard rule 9): `update_one` as
    /// `update` does.
    #[test]
    fn a_row_of_no_weight_learns_nothing_whatever_its_value() {
        let mut base = EwDiag::new(1);
        base.update(&[0.7], 1.0, 1.0);
        base.update(&[-0.2], 0.9, 1.0);
        for x in [f64::NAN, f64::INFINITY, 1e300, 0.0] {
            let (mut one, mut many) = (base.clone(), base.clone());
            one.update_one(x, 0.9, 0.0);
            many.update(&[x], 0.9, 0.0);
            assert_eq!(one, many, "{x}");
            assert_eq!(one.mean(0).to_bits(), base.mean(0).to_bits(), "{x}");
            assert_eq!(one.var(0).to_bits(), base.var(0).to_bits(), "{x}");
        }
    }

    /// A stream with offsets and scales that differ by orders of magnitude,
    /// zero-weight rows (including the first), varying decay and a pure
    /// zero-weight run — everything `kalman` and `sgd` can feed it.
    /// Rows of `(x, lam, w)`.
    type Rows = Vec<(Vec<f64>, f64, f64)>;

    fn stream(k: usize, n: usize, seed: u64) -> Rows {
        let mut s = seed;
        (0..n)
            .map(|r| {
                let x: Vec<f64> = (0..k)
                    .map(|i| 10f64.powi(i as i32 - 2) * lcg(&mut s) + 7.0 * i as f64)
                    .collect();
                let lam = if r % 11 == 0 { 0.5 } else { 0.97 };
                let w = match r % 9 {
                    0 => 0.0,
                    1 => 2.5,
                    _ => 1.0,
                };
                let w = if r == 0 || (40..46).contains(&r) {
                    0.0
                } else {
                    w
                };
                (x, lam, w)
            })
            .collect()
    }

    /// `stream`, with slot 0 at a level of 1e8 and held there from row 100:
    /// long enough for its mean's step to fall below its rounding, where the
    /// low part carries it (docs/PLAN.md task 101). Its rows of weight 0
    /// carry another value, which moves no mean.
    fn held_stream(k: usize, n: usize, seed: u64) -> Rows {
        let mut rows = stream(k, n, seed);
        for (r, (x, _, w)) in rows.iter_mut().enumerate() {
            x[0] = 1e8
                + match (r < 100, *w > 0.0) {
                    (true, _) => x[0],
                    (false, true) => 0.37,
                    (false, false) => 5.0,
                };
        }
        rows
    }

    /// Both streams: the plain one, and the held one, on which slot 0's mean
    /// is also checked to reach the value it holds, which a plain mean does
    /// not (`held`).
    fn streams(k: usize, seed: u64) -> [(Rows, bool); 2] {
        [
            (stream(k, 300, seed), false),
            (held_stream(k, 1500, seed), true),
        ]
    }

    #[test]
    fn is_the_diagonal_of_ewcov_bit_for_bit() {
        for k in [1, 2, 5] {
            for (rows, held) in streams(k, 7 + k as u64) {
                let mut full = EwCov::new(k);
                let mut diag = EwDiag::new(k);
                for (x, lam, w) in rows {
                    full.update(&x, lam, w);
                    diag.update(&x, lam, w);
                    assert_eq!(diag.n_eff().to_bits(), full.n_eff().to_bits());
                    for i in 0..k {
                        assert_eq!(diag.mean(i).to_bits(), full.mean(i).to_bits(), "mean {i}");
                        assert_eq!(diag.var(i).to_bits(), full.var(i).to_bits(), "var {i}");
                        assert_eq!(diag.raw(i).to_bits(), full.raw(i, i).to_bits(), "raw {i}");
                    }
                    assert_eq!(diag.means(), full.means());
                    assert_eq!(diag.m_lo, full.means_lo());
                }
                if held {
                    assert_eq!(diag.mean(0), 1e8 + 0.37, "k {k}");
                }
            }
        }
    }

    /// `including` reads, for every slot and every row of the stream, the
    /// bits `update` writes for that row at unit weight -- under every decay
    /// the stream uses, from an empty accumulator on -- and moves nothing.
    #[test]
    fn including_reads_what_a_unit_row_would_leave() {
        for k in [1, 3] {
            for (rows, _) in streams(k, 11 + k as u64) {
                let mut d = EwDiag::new(k);
                for (x, lam, w) in rows {
                    let before = d.clone();
                    let inc = d.including(lam);
                    let read: Vec<(f64, f64, f64)> = (0..k).map(|i| inc.moments(i, x[i])).collect();
                    let mut unit = d.clone();
                    unit.update(&x, lam, 1.0);
                    // The history's share as `including` forms it: below
                    // `SMALL_SHARE` the deviation from the mean the row
                    // leaves is that share of its deviation from the mean
                    // before it.
                    let a = lam * d.n_eff() / (lam * d.n_eff() + 1.0);
                    for (i, &(m, v, dev)) in read.iter().enumerate() {
                        assert_eq!(m.to_bits(), unit.mean(i).to_bits(), "mean {i}");
                        assert_eq!(v.to_bits(), unit.var(i).to_bits(), "var {i}");
                        let want = if a < SMALL_SHARE && a > 0.0 {
                            a * d.deviation(i, x[i])
                        } else {
                            unit.deviation(i, x[i])
                        };
                        assert_eq!(dev.to_bits(), want.to_bits(), "dev {i}");
                    }
                    assert_eq!(d, before, "reading moved the accumulator");
                    // The stream's own weight, which may be 0 or 2.5, is what
                    // the accumulator actually takes.
                    d.update(&x, lam, w);
                }
            }
        }
        // Empty: the row is the whole history.
        let inc = EwDiag::new(2);
        let inc = inc.including(0.9);
        assert_eq!(inc.moments(0, 1e6), (1e6, 0.0, 0.0));
        assert_eq!(inc.moments(1, -3.0), (-3.0, 0.0, 0.0));
    }

    /// `update_keeping` takes the floor only where there is a history, the
    /// row has a weight and the floor is strictly above the carried weight:
    /// at the head of a stream, at weight 0 and at the tie it is `update`, to
    /// the bit, and says it did not bind; above the tie it reads the old
    /// statistics at the floor's share, and keeps the plain weight.
    #[test]
    fn update_keeping_binds_only_above_the_carried_weight() {
        let x = [1.5, -2.0];
        // No history: the floor has nothing to hold up.
        let mut kept = EwDiag::new(2);
        let mut plain = EwDiag::new(2);
        assert!(!kept.update_keeping(&x, 0.9, 1.0, 5.0, f64::INFINITY));
        plain.update(&x, 0.9, 1.0);
        assert_eq!(kept, plain);

        let mut base = EwDiag::new(2);
        base.update(&[0.25, 4.0], 1.0, 1.0);
        base.update(&[1.0, 3.0], 1.0, 1.0);
        assert_eq!(base.n_eff(), 2.0);

        // A row of no weight: nothing to learn, so nothing to keep.
        let (mut kept, mut plain) = (base.clone(), base.clone());
        assert!(!kept.update_keeping(&x, 0.5, 0.0, 9.0, 2.0));
        plain.update(&x, 0.5, 0.0);
        assert_eq!(kept, plain);

        // The tie: lam * W is 1 and so is the floor.
        let (mut kept, mut plain) = (base.clone(), base.clone());
        assert!(!kept.update_keeping(&x, 0.5, 1.0, 1.0, 2.0));
        plain.update(&x, 0.5, 1.0);
        assert_eq!(kept, plain);

        // Below the tie the floor does not bind either.
        let (mut kept, mut plain) = (base.clone(), base.clone());
        assert!(!kept.update_keeping(&x, 0.5, 1.0, 0.5, 2.0));
        plain.update(&x, 0.5, 1.0);
        assert_eq!(kept, plain);

        // Above it: the old statistics take the share H / (H + w), H the
        // floor up to the history's own weight W = 2, and the weight kept is
        // lam * W + w, the plain recursion's.
        let (floor, w, lam) = (3.0, 1.0, 0.5);
        let mut kept = base.clone();
        assert!(kept.update_keeping(&x, lam, w, floor, base.n_eff()));
        let h = floor.min(base.n_eff());
        assert_eq!(h, 2.0);
        let (a, b) = (h / (h + w), w / (h + w));
        for (i, xi) in x.iter().enumerate() {
            let d = xi - base.mean(i);
            let want_mean = base.mean(i) + b * d;
            let want_c = a * base.var(i) + a * b * d * d;
            assert!((kept.mean(i) - want_mean).abs() < 1e-14, "mean {i}");
            assert!(
                (kept.var(i) - want_c).abs() < 1e-12,
                "var {i}: {} vs {want_c}",
                kept.var(i)
            );
        }
        assert_eq!(kept.n_eff(), lam * base.n_eff() + w);
    }

    /// The history is held up to the floor, never past the weight it had:
    /// a row of weight `1e-6`, then a row of weight 1 a little past a
    /// half-life later, read the light row at half the share, a million
    /// times its weight, where a little short of a half-life the plain
    /// update gave it its weight's share (review 2026-11-23, REG-15). The
    /// history kept is `max(lam·W, min(W, floor))`, as `sgd`'s scaler keeps
    /// it.
    #[test]
    fn the_floor_holds_a_history_up_to_its_own_weight_and_no_further() {
        for lam in [0.5f64.powf(0.99), 0.5f64.powf(1.01), 0.5f64.powi(30)] {
            let mut d = EwDiag::new(1);
            d.update(&[0.0], 1.0, 1e-6);
            let mut plain = d.clone();
            d.update_keeping(&[10.0], lam, 1.0, 1.0, 1e-6);
            plain.update(&[10.0], lam, 1.0);
            let h = (lam * 1e-6f64).max(1e-6f64.min(1.0));
            let want = 10.0 * 1.0 / (h + 1.0);
            assert!(
                (d.mean(0) - want).abs() < 1e-12,
                "lam {lam}: {} against {want}",
                d.mean(0)
            );
            assert!(d.mean(0) > 9.99999, "lam {lam}");
            assert_eq!(d.n_eff(), plain.n_eff());
        }
    }

    /// The deviation is read as `a·d` where the history's share is strictly
    /// below `SMALL_SHARE`, and as the pair's at the share itself: with a
    /// history of weight 1 and a row of weight 1023, `a` is `2⁻¹⁰` exactly.
    #[test]
    fn the_small_share_switch_is_strict() {
        let mut found_a_difference = false;
        for &(m0, x) in &[
            (0.1, 0.3),
            (0.7, 0.1),
            (1.0 / 3.0, 0.9),
            (2.2, 0.37),
            (0.6, 0.15),
        ] {
            let mut d = EwDiag::new(1);
            d.update(&[m0], 1.0, 1.0);
            let inc = d.including_at(1.0, 1023.0);
            assert_eq!(inc.a, SMALL_SHARE);
            let (hi, lo, _c) = (d.m[0], d.m_lo[0], d.c[0]);
            let pre = crate::comp::dev(x, hi, lo);
            let (_, _, dev) = inc.moments(0, x);
            // The mean the row leaves, from the same recursion.
            let (mut h2, mut l2) = (hi, lo);
            crate::comp::add(&mut h2, &mut l2, inc.b * pre);
            let pair = crate::comp::dev(x, h2, l2);
            assert_eq!(dev.to_bits(), pair.to_bits(), "at the share: the pair's");
            found_a_difference |= pair.to_bits() != (inc.a * pre).to_bits();
        }
        assert!(
            found_a_difference,
            "no case told the two forms apart, so the test holds nothing"
        );
    }

    #[test]
    fn a_zero_weight_first_row_and_a_negative_weight_change_nothing() {
        let mut d = EwDiag::new(2);
        d.update(&[1e100, -3.0], 0.9, 0.0);
        assert_eq!(d, EwDiag::new(2));
        d.update(&[1.0, 2.0], 0.9, 1.0);
        let before = d.clone();
        // `debug_assert` would fire; the release-mode contract is a no-op.
        if !cfg!(debug_assertions) {
            d.update(&[5.0, 5.0], 0.9, -1.0);
            assert_eq!(d, before);
        }
        assert_eq!(d.mean(0), 1.0);
        assert_eq!(d.var(1), 0.0);
        assert_eq!(d.raw(1), 4.0);
    }

    /// A row of weight 0 carrying the mean itself: the variance term is
    /// `a·b·d·d` with `b` and `d` both zero, and stays zero, not NaN. Read
    /// from the accumulator, as `var` read a NaN as 0 until task 182.
    #[test]
    fn a_row_of_no_weight_carrying_the_mean_changes_nothing() {
        let mut d = EwDiag::new(1);
        d.update(&[2.0], 0.9, 1.0);
        d.update(&[2.0], 0.9, 0.0);
        assert_eq!((d.mean(0), d.c[0]), (2.0, 0.0));
    }

    /// A variance of NaN stays NaN, in `var` and in the moments a row would
    /// leave (`including`). The floor against rounding was `max(0.0)`,
    /// which returns its other argument when one is NaN, so a moment a NaN
    /// had reached read as a variance of 0, a column with no spread (task
    /// 182). A rounding below zero is still 0, and so is `-0.0`.
    #[test]
    fn a_variance_of_nan_stays_nan() {
        let mut d = EwDiag::new(4);
        d.update(&[1.0, 2.0, 3.0, 4.0], 1.0, 1.0);
        d.c = vec![f64::NAN, -1e-18, -0.0, 2.5];
        assert!(d.var(0).is_nan(), "{}", d.var(0));
        assert_eq!(d.var(1).to_bits(), 0f64.to_bits());
        assert_eq!(d.var(2).to_bits(), 0f64.to_bits());
        assert_eq!(d.var(3), 2.5);
        // The row at the means: no deviation, so the moments a row leaves
        // are the old ones aged by `a`, a half here.
        let inc = d.including(1.0);
        assert!(inc.moments(0, 1.0).1.is_nan());
        assert_eq!(inc.moments(1, 2.0).1.to_bits(), 0f64.to_bits());
        assert_eq!(inc.moments(2, 3.0).1.to_bits(), 0f64.to_bits());
        assert_eq!(inc.moments(3, 4.0).1, 1.25);
    }

    /// `EwDiag`'s own skip (review 2026-09-25): the rows 0.7 and
    /// 5.292162135665459 at unit weight leave the pair with a low part of a
    /// whole step, where a step of 0 would round `hi` up; a row of weight 0
    /// leaves both parts to the bit. Every other test of this compares it
    /// with `EwCov`, which skips too.
    #[test]
    fn a_row_of_no_weight_takes_no_step_in_the_pair() {
        let mut d = EwDiag::new(1);
        d.update(&[0.7], 1.0, 1.0);
        d.update(&[5.292162135665459], 1.0, 1.0);
        assert_eq!(
            (d.m[0], d.m_lo[0]),
            (2.996081067832729, 4.440892098500626e-16)
        );
        d.update(&[9.0], 1.0, 0.0);
        assert_eq!(
            (d.m[0], d.m_lo[0]),
            (2.996081067832729, 4.440892098500626e-16)
        );
    }

    #[test]
    fn serde_roundtrip_in_both_encodings() {
        let mut d = EwDiag::new(2);
        d.update(&[1.0, 2.0], 0.95, 1.3);
        d.update(&[0.5, 2.5], 0.95, 1.0);
        for bytes in [
            rmp_serde::to_vec_named(&d).unwrap(),
            rmp_serde::to_vec(&d).unwrap(),
        ] {
            let back: EwDiag = rmp_serde::from_slice(&bytes).unwrap();
            assert_eq!(back, d);
        }
    }

    /// A serialized `EwCov` must not pass for an `EwDiag` in either
    /// encoding, or a model would load with `k²` variances.
    #[test]
    fn a_full_ewcov_is_refused_in_both_encodings() {
        for k in [1, 2, 4] {
            let mut full = EwCov::new(k);
            full.update(&vec![1.0; k], 0.9, 1.0);
            full.update(&vec![2.0; k], 0.9, 1.0);
            for bytes in [
                rmp_serde::to_vec_named(&full).unwrap(),
                rmp_serde::to_vec(&full).unwrap(),
            ] {
                let got = rmp_serde::from_slice::<EwDiag>(&bytes);
                assert!(got.is_err(), "k = {k}: {got:?}");
            }
        }
        // The shape check on its own, for a hand-written map.
        let bad = serde_json::json!({
            "k": 2, "w_sum": 1.0, "m": [0.0, 0.0], "c": [0.0, 0.0, 0.0, 0.0], "m_lo": [0.0, 0.0]
        });
        let err = serde_json::from_value::<EwDiag>(bad)
            .unwrap_err()
            .to_string();
        assert!(err.contains("wrong shape"), "{err}");
    }

    /// Review 2026-11-23, STATE-3: a weight below 0, which made each
    /// variance negative, and a value that is no number are refused.
    #[test]
    fn a_weight_or_moment_no_stream_writes_is_refused() {
        let ok = serde_json::json!({
            "k": 1, "w_sum": 2.5, "m": [1.0], "c": [0.5], "m_lo": [0.0]
        });
        assert!(serde_json::from_value::<EwDiag>(ok.clone()).is_ok());
        for (key, v) in [
            ("w_sum", serde_json::json!(-2.5)),
            ("c", serde_json::json!([-0.5])),
            ("m", serde_json::json!([f64::NAN])),
        ] {
            let mut bad = ok.clone();
            bad[key] = v;
            let err = serde_json::from_value::<EwDiag>(bad).map(|_| ());
            assert!(err.is_err(), "{key}");
        }
    }
}
