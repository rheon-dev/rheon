//! The papers' estimators on a batch of returns, written from the papers
//! (Barndorff-Nielsen and Shephard 2004, eqs. of §1; 2006, eqs. 10-14;
//! Lee and Mykland 2008, Definition 1), never from the online code: the
//! oracle the online forms are held to in the Rust tests (docs/PLAN.md task
//! 499). Each sum is carried in double-double, so a value here is the
//! paper's to the rounding of its terms; the gaps the tests allow are then
//! the online sums' own.

use super::{BNS_THETA, BV_SCALE, TQ_SCALE, TV_SCALE};

/// A sum in double-double: `hi + lo`, each term added by TwoSum, so the
/// result is the exact sum of the terms to about `2^-104` of its size where
/// the terms do not cancel (every sum here is of terms of one sign).
#[derive(Clone, Copy, Default, Debug)]
pub struct Dd {
    pub hi: f64,
    pub lo: f64,
}

impl Dd {
    pub fn add(&mut self, x: f64) {
        let s = self.hi + x;
        let bb = s - self.hi;
        let err = (self.hi - (s - bb)) + (x - bb);
        self.hi = s;
        self.lo += err;
    }

    pub fn value(self) -> f64 {
        self.hi + self.lo
    }
}

/// The double-double sum of `terms`, rounded once.
pub fn sum(terms: impl IntoIterator<Item = f64>) -> f64 {
    let mut s = Dd::default();
    for t in terms {
        s.add(t);
    }
    s.value()
}

/// The papers' measures of one batch of `n` returns.
#[derive(Clone, Copy, Debug)]
pub struct Measures {
    pub n: usize,
    pub rv: f64,
    pub bv: f64,
    pub tv: f64,
    pub tq: f64,
    /// The adjusted ratio test, `√n (1 − BV/RV) / √(ϑ max(1, TQ/BV²))`.
    pub z: f64,
}

/// `Σ_{j ≥ m} Π_{i < m} a_{j−i}^p` over consecutive returns: the sum of the
/// `n − m + 1` products of `m` neighbours, each raised to `p`, factor by
/// factor as the papers write them.
fn multipower(r: &[f64], m: usize, p: f64) -> f64 {
    if r.len() < m {
        return 0.0;
    }
    sum(r.windows(m).map(|w| {
        w.iter()
            .map(|x| if p == 1.0 { x.abs() } else { x.abs().powf(p) })
            .product::<f64>()
    }))
}

/// RV, BV, TV, TQ and the adjusted ratio test on the returns `r` (BNS 2004,
/// 2006): no small-sample factor, `n` the count of returns.
pub fn measures(r: &[f64]) -> Measures {
    let n = r.len();
    let rv = sum(r.iter().map(|x| x * x));
    let bv = BV_SCALE * multipower(r, 2, 1.0);
    let tv = TV_SCALE * multipower(r, 3, 2.0 / 3.0);
    let tq = n as f64 * (TQ_SCALE * multipower(r, 3, 4.0 / 3.0));
    let z = (n as f64).sqrt() * (1.0 - bv / rv) / (BNS_THETA * (tq / (bv * bv)).max(1.0)).sqrt();
    Measures {
        n,
        rv,
        bv,
        tv,
        tq,
        z,
    }
}

/// Lee and Mykland's `L_i` for each return of `r`, with a window of `K`
/// returns: `L_i = r_i / σ_i`, `σ_i² = (1/(K − 2)) Σ_{j=i−K+2}^{i−1} |r_j|
/// |r_{j−1}|`, the `K − 2` products of the `K − 1` returns before `r_i`
/// (Definition 1; the current return is outside its own window). `NaN`
/// where fewer than `K − 1` returns precede it.
pub fn lee_mykland(r: &[f64], k: usize) -> Vec<f64> {
    assert!(k >= 3, "a window of K returns needs K >= 3");
    (0..r.len())
        .map(|i| {
            if i < k - 1 {
                return f64::NAN;
            }
            let before = &r[i + 1 - (k - 1)..i];
            let s = multipower(before, 2, 1.0) / (k - 2) as f64;
            r[i] / s.sqrt()
        })
        .collect()
}
