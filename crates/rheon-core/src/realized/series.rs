//! One return series of a [`super::Realized`]: its compensated sums, the
//! last returns its next products are formed with, a window's terms and
//! Lee and Mykland's own window of products.

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

use super::{BNS_THETA, BV_SCALE, Held, OUTPUTS_PER_TARGET, TQ_SCALE, TV_SCALE};

/// `hi + lo += x` by Knuth's TwoSum: the rounding of `hi + x` is found
/// exactly whatever the two sizes and goes into `lo`, so a window can take
/// back a term larger than what is left (a jump leaving) and keep the
/// digits; Kahan's step (`crate::comp::add`) is exact only while the step is
/// the smaller.
pub(super) fn add(hi: &mut f64, lo: &mut f64, x: f64) {
    let s = *hi + x;
    let bb = s - *hi;
    *lo += (*hi - (s - bb)) + (x - bb);
    *hi = s;
}

/// The sums a target keeps, as indices into its pairs.
const NW: usize = 0;
const RV: usize = 1;
const BV: usize = 2;
const TV: usize = 3;
const TQ: usize = 4;
const PW: usize = 5;
/// The returns' squared weights, decayed twice over: with `n`'s weight,
/// Kish's effective count `n² / Σw²`.
const W2: usize = 6;
pub(super) const SUMS: usize = 7;

/// One row's terms of a series in a window, each at the row's weight, and
/// how many of its products were taken: 1 for its product with the return
/// before it, 2 for its triple too, which a product takes only when every
/// return in it is in the window.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(super) struct Entry {
    seq: u64,
    w: f64,
    rv: f64,
    bv: f64,
    tv: f64,
    tq: f64,
    links: u8,
}

impl Entry {
    /// Its numbers, in bytes: 7 of 8.
    pub(super) const BYTES: u64 = 8 * 7;
}

/// Lee and Mykland's window of `K`: the products of the `K − 1` returns
/// before the row, each at its later row's weight, and their compensated
/// sums.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(super) struct LmRing {
    /// `(w a_j a_{j−1}, w)`, oldest first; at most `K − 2`.
    products: VecDeque<(f64, f64)>,
    p_hi: f64,
    p_lo: f64,
    w_hi: f64,
    w_lo: f64,
}

impl LmRing {
    fn new() -> Self {
        Self {
            products: VecDeque::new(),
            p_hi: 0.0,
            p_lo: 0.0,
            w_hi: 0.0,
            w_lo: 0.0,
        }
    }

    /// The bytes of a ring of `K`: `16 (K − 2)` and its four sums.
    pub(super) fn bytes(k: usize) -> u64 {
        16 * k.saturating_sub(2) as u64 + 32
    }

    fn push(&mut self, wp: f64, w: f64, k: usize) {
        self.products.push_back((wp, w));
        add(&mut self.p_hi, &mut self.p_lo, wp);
        add(&mut self.w_hi, &mut self.w_lo, w);
        if self.products.len() > k - 2
            && let Some((wp, w)) = self.products.pop_front()
        {
            add(&mut self.p_hi, &mut self.p_lo, -wp);
            add(&mut self.w_hi, &mut self.w_lo, -w);
        }
    }

    /// `σ²`, the weighted mean of the products held.
    fn variance(&self) -> f64 {
        (self.p_hi + self.p_lo) / (self.w_hi + self.w_lo)
    }

    fn weight(&self) -> f64 {
        self.w_hi + self.w_lo
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(super) struct Series {
    /// The sums' high parts, `[n, rv, bv, tv, tq, p, Σw²]`.
    pub(super) hi: [f64; SUMS],
    /// What each high part leaves out ([`add`]).
    pub(super) lo: [f64; SUMS],
    /// `|r|` of the last return learned.
    a1: f64,
    /// `|r|^(1/3)` of the last two returns learned, the latest first.
    c1: f64,
    c2: f64,
    /// How many returns are remembered: 0, 1, or 2 for two or more.
    pub(super) seen: u8,
    /// A window's terms of this series, oldest first.
    #[serde(default)]
    held: VecDeque<Entry>,
    #[serde(default)]
    lm: Option<LmRing>,
}

/// Lee and Mykland's flag for `L` among `n` returns at confidence `β`
/// (the module docs): 1 for a jump, 0 for none, `NaN` where `L` is or `n
/// ≤ 1`.
pub(super) fn lm_jump(l: f64, n: f64, beta: f64) -> f64 {
    if !l.is_finite() || n <= 1.0 {
        return f64::NAN;
    }
    let c = (2.0 / std::f64::consts::PI).sqrt();
    let g = (2.0 * libm::log(n)).sqrt();
    let c_n = g / c - (libm::log(std::f64::consts::PI) + libm::log(libm::log(n))) / (2.0 * c * g);
    let s_n = 1.0 / (c * g);
    let critical = -libm::log(-libm::log(beta));
    if (l.abs() - c_n) / s_n > critical {
        1.0
    } else {
        0.0
    }
}

impl Series {
    /// The numbers a series holds outside a window and `lm_window`, in
    /// bytes: 16 of 8.
    pub(super) const BYTES: u64 = 8 * 18;

    pub(super) fn new(lm_window: Option<usize>) -> Self {
        Self {
            hi: [0.0; SUMS],
            lo: [0.0; SUMS],
            a1: 0.0,
            c1: 0.0,
            c2: 0.0,
            seen: 0,
            held: VecDeque::new(),
            lm: lm_window.map(|_| LmRing::new()),
        }
    }

    fn sum(&self, k: usize) -> f64 {
        self.hi[k] + self.lo[k]
    }

    /// The weight of the series' returns, the papers' `n`.
    pub(super) fn weight(&self) -> f64 {
        self.sum(NW)
    }

    /// Every sum aged by the factor `f`, the squared weights by `f²`.
    pub(super) fn decay(&mut self, f: f64) {
        if f != 1.0 {
            for v in self.hi.iter_mut().chain(self.lo.iter_mut()) {
                *v *= f;
            }
            self.hi[W2] *= f;
            self.lo[W2] *= f;
        }
    }

    /// The count of returns the estimators are of, the papers' `n`: the
    /// returns a window holds, or Kish's effective count of the returns'
    /// decayed weights, `(Σw)² / Σw²`, which is the count of returns at
    /// equal weights without a decay. 0 before a return.
    pub(super) fn count(&self, windowed: bool) -> f64 {
        if windowed {
            return self.held.len() as f64;
        }
        let w2 = self.sum(W2);
        if w2 > 0.0 {
            let w = self.sum(NW);
            w * w / w2
        } else {
            0.0
        }
    }

    fn add(&mut self, k: usize, x: f64) {
        add(&mut self.hi[k], &mut self.lo[k], x);
    }

    /// The return `r` learned at the weight `w > 0`, a window's row `seq`
    /// where there is a window.
    pub(super) fn learn(&mut self, r: f64, w: f64, seq: Option<u64>, lm_window: Option<usize>) {
        let a = r.abs();
        let c = libm::cbrt(a);
        let rv = w * (r * r);
        let bv = w * (a * self.a1);
        let t = c * self.c1 * self.c2;
        let tv = w * (t * t);
        let tq = w * ((t * t) * (t * t));
        // A product is taken where every return in it is held: under a
        // window, only those of its returns the window still has.
        let links = match seq {
            Some(_) => self.seen.min(self.held.len().min(2) as u8),
            None => self.seen,
        };
        self.add(NW, w);
        self.add(W2, w * w);
        self.add(RV, rv);
        if links >= 1 {
            self.add(BV, bv);
            self.add(PW, w);
        }
        if links >= 2 {
            self.add(TV, tv);
            self.add(TQ, tq);
        }
        if let Some(seq) = seq {
            self.held.push_back(Entry {
                seq,
                w,
                rv,
                bv,
                tv,
                tq,
                links,
            });
        }
        if let (Some(ring), Some(k)) = (self.lm.as_mut(), lm_window)
            && self.seen >= 1
        {
            ring.push(bv, w, k);
        }
        self.c2 = self.c1;
        self.c1 = c;
        self.a1 = a;
        self.seen = (self.seen + 1).min(2);
    }

    /// The window's row `seq` has left: this series' return there leaves,
    /// with the product and the triple that reached it.
    pub(super) fn drop_front(&mut self, seq: u64) {
        if self.held.front().is_none_or(|e| e.seq != seq) {
            return;
        }
        let Some(e) = self.held.pop_front() else {
            return;
        };
        self.add(NW, -e.w);
        self.add(W2, -(e.w * e.w));
        self.add(RV, -e.rv);
        if let Some(f) = self.held.front().filter(|f| f.links >= 1) {
            let (bv, w) = (f.bv, f.w);
            self.add(BV, -bv);
            self.add(PW, -w);
        }
        if let Some(g) = self.held.get(1).filter(|g| g.links >= 2) {
            let (tv, tq) = (g.tv, g.tq);
            self.add(TV, -tv);
            self.add(TQ, -tq);
        }
        if self.held.is_empty() {
            (self.hi, self.lo) = ([0.0; SUMS], [0.0; SUMS]);
        }
    }

    /// Whether a restored series has the shape its settings give it: an
    /// `lm_window` ring exactly when one is set and no longer than it, and
    /// a window's entries, in order, only of rows the window holds.
    pub(super) fn has_shape(
        &self,
        lm_window: Option<usize>,
        rows: Option<&VecDeque<Held>>,
    ) -> bool {
        let lm = match (&self.lm, lm_window) {
            (None, None) => true,
            (Some(r), Some(k)) => r.products.len() <= k.saturating_sub(2),
            _ => false,
        };
        let held = match rows {
            None => self.held.is_empty(),
            Some(rows) => {
                self.held
                    .iter()
                    .zip(self.held.iter().skip(1))
                    .all(|(a, b)| a.seq < b.seq)
                    && self.held.iter().all(|e| {
                        e.links <= 2 && rows.binary_search_by_key(&e.seq, |h| h.seq).is_ok()
                    })
            }
        };
        lm && held && self.seen <= 2
    }

    /// The series' outputs into `out`, from its state as it stands, for a
    /// row whose return is `r` (none where the row has none).
    pub(super) fn read(
        &self,
        r: Option<f64>,
        min_weight: f64,
        beta: f64,
        windowed: bool,
        out: &mut Vec<f64>,
    ) {
        if self.sum(NW) < min_weight {
            out.extend([f64::NAN; OUTPUTS_PER_TARGET]);
            return;
        }
        let n = self.count(windowed);
        let rv = self.sum(RV);
        let s_bv = self.sum(BV);
        let bv = BV_SCALE * s_bv;
        let tv = TV_SCALE * self.sum(TV);
        let tq = n * (TQ_SCALE * self.sum(TQ));
        let z = if rv > 0.0 && bv > 0.0 {
            n.sqrt() * (1.0 - bv / rv) / (BNS_THETA * (tq / (bv * bv)).max(1.0)).sqrt()
        } else {
            f64::NAN
        };
        let sigma2 = match &self.lm {
            Some(ring) if ring.weight() > 0.0 => ring.variance(),
            Some(_) => f64::NAN,
            None => s_bv / self.sum(PW),
        };
        let lm = match r {
            Some(r) if sigma2 > 0.0 => r / sigma2.sqrt(),
            _ => f64::NAN,
        };
        out.extend([rv, bv, tv, tq, z, lm, lm_jump(lm, n, beta)]);
    }
}
