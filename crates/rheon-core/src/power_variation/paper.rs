//! The terms on a batch of returns, written from their definitions
//! (Barndorff-Nielsen and Shephard 2004, 2006 for the multipower variation;
//! Barndorff-Nielsen, Hansen, Lunde and Shephard 2008, 2011 and Newey and
//! West 1987 for the kernel sums; Zhang, Mykland and Aït-Sahalia 2005 for
//! the subsampled grids), never from the online code: the oracle the online
//! forms are held to in the Rust tests (docs/PLAN.md task 555). Each sum is
//! carried in double-double, the `K`-step returns too, and a centred sum is
//! taken about its mean directly, not by the online form's expansion.

/// A sum in double-double: `hi + lo`, each term added by TwoSum.
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

pub fn sum(terms: impl IntoIterator<Item = f64>) -> f64 {
    let mut s = Dd::default();
    for t in terms {
        s.add(t);
    }
    s.value()
}

/// The kernels by their formulas, written again here.
pub fn kernel(name: &str, x: f64) -> f64 {
    if !(0.0..=1.0).contains(&x) {
        return 0.0;
    }
    match name {
        "bartlett" => 1.0 - x,
        "parzen" if x <= 0.5 => 1.0 - 6.0 * x.powi(2) + 6.0 * x.powi(3),
        "parzen" => 2.0 * (1.0 - x).powi(3),
        "tukey_hanning" => (1.0 + (std::f64::consts::PI * x).cos()) / 2.0,
        "rectangular" => 1.0,
        _ => unreachable!("{name}"),
    }
}

/// `c_h` for `h = 1..=H`: `k(h/(H + 1))`, or `k((h − 1)/H)` flat-top.
pub fn weights(name: &str, h_max: usize, flat_top: bool) -> Vec<f64> {
    (1..=h_max)
        .map(|h| {
            let x = if flat_top {
                (h - 1) as f64 / h_max as f64
            } else {
                h as f64 / (h_max + 1) as f64
            };
            kernel(name, x)
        })
        .collect()
}

/// The `K`-step returns `R_j = r_{j−K+1} + ... + r_j`, `j ≥ K − 1`, each
/// at index `j`; `None` before.
fn kstep(r: &[f64], k: usize) -> Vec<Option<f64>> {
    (0..r.len())
        .map(|j| (j + 1 >= k).then(|| sum(r[j + 1 - k..=j].iter().copied())))
        .collect()
}

/// `(1/K) Σ_j ω_j Π_i |R_{j−(m−i)K}|^{p_i}`, each term at the weight `ω`
/// of its newest return.
pub fn multipower(r: &[f64], om: &[f64], powers: &[f64], k: usize) -> f64 {
    let big = kstep(r, k);
    let m = powers.len();
    let mut s = Dd::default();
    for j in 0..r.len() {
        if j + 1 < m * k {
            continue;
        }
        let mut acc = 1.0;
        for (i, &p) in powers.iter().enumerate() {
            let x = big[j - (m - 1 - i) * k].expect("formed").abs();
            acc *= if p == 1.0 { x } else { x.powf(p) };
        }
        s.add(om[j] * acc);
    }
    s.value() / k as f64
}

/// `(1/K) Σ_j ω_j [d_j² + 2 Σ_h c_h d_j d_{j−hK}]`, `d = R − m̄` with
/// `m̄` the `ω`-mean of the `R_j` where `center`, `d = R` otherwise.
pub fn kernel_sum(r: &[f64], om: &[f64], c: &[f64], center: bool, k: usize) -> f64 {
    let big = kstep(r, k);
    let mean = if center {
        let (mut a, mut b) = (Dd::default(), Dd::default());
        for (j, x) in big.iter().enumerate() {
            if let Some(x) = x {
                a.add(om[j] * x);
                b.add(om[j]);
            }
        }
        if b.value() > 0.0 {
            a.value() / b.value()
        } else {
            0.0
        }
    } else {
        0.0
    };
    let mut s = Dd::default();
    for j in 0..r.len() {
        let Some(x) = big[j] else { continue };
        let d = x - mean;
        s.add(om[j] * (d * d));
        for (h, &ch) in (1..).zip(c) {
            if j < h * k + k - 1 {
                break;
            }
            let e = big[j - h * k].expect("formed") - mean;
            s.add(om[j] * (2.0 * ch * (d * e)));
        }
    }
    s.value() / k as f64
}

/// Kish's effective count `(Σω)² / Σω²`.
pub fn kish(om: &[f64]) -> f64 {
    let w = sum(om.iter().copied());
    let v = sum(om.iter().map(|x| x * x));
    if v > 0.0 { w * w / v } else { 0.0 }
}
