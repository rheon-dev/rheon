//! One return series of a [`super::PowerVariation`]: its compensated sums,
//! the returns its next terms reach back to, and a window's credits; and
//! the [`Plan`] the settings fix.

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

use super::{Held, PvTerm, PvTermKind};

/// `hi + lo += x` by Knuth's TwoSum: the rounding of `hi + x` is found
/// exactly whatever the two sizes and goes into `lo`, so a window can take
/// back a term larger than what is left and keep the digits.
///
/// A sum that would pass the largest double is held at it, `lo` 0: the
/// largest sum a double holds, a bound on the sum, where `inf` is one no
/// decay ever brings back (its next take-back would be `inf − inf`). Below
/// it nothing moves, to the bit.
pub(super) fn add(hi: &mut f64, lo: &mut f64, x: f64) {
    let s = *hi + x;
    if s.abs() >= f64::MAX {
        *hi = f64::MAX.copysign(s);
        *lo = 0.0;
        return;
    }
    let bb = s - *hi;
    *lo += (*hi - (s - bb)) + (x - bb);
    *hi = s;
}

/// Whether taking the term `x` back from a compensated sum left `rest`
/// keeps too little of it. The pair holds a sum to about `ε²` of the
/// largest term it has held, so once a term leaves that is more than `1/ε`
/// times what is left (a return near the bound leaving a window of
/// ordinary ones), what is left is rounding of it, and it is summed again
/// from the terms held.
fn lost(x: f64, rest: f64) -> bool {
    x.abs() * f64::EPSILON > rest.abs()
}

/// `x`, or the largest double of its sign where a term passed it: a bound
/// on the term, which a sum holds as [`add`] holds its own.
fn capped(x: f64) -> f64 {
    x.clamp(-f64::MAX, f64::MAX)
}

/// `w Π |r_i|^{p_i}`, for a product of powers that passes the largest
/// double on the way (`|r|^4` of a return near the bound) or reads `0 ·
/// inf`: each factor as a fraction and a power of two, which neither
/// overflows nor underflows, the term then formed once, held at the
/// largest double where it passes it. A factor of 0 makes the term 0. A
/// power that is not whole rounds its exponent `e p`, so the term is the
/// plain product's to `ln 2 · ε · |e p|`, below `1e-13` for any exponent a
/// double has; this runs only where the plain product is not finite.
fn product_past_range(w: f64, factors: &[(f64, f64)]) -> f64 {
    let (mut m, e) = libm::frexp(w);
    let mut e = f64::from(e);
    for &(a, p) in factors {
        if a == 0.0 {
            return 0.0;
        }
        let (am, ae) = libm::frexp(a);
        let (mm, me) = libm::frexp(m * libm::pow(am, p));
        m = mm;
        e += f64::from(ae) * p + f64::from(me);
    }
    let whole = e.floor();
    let m = m * libm::exp2(e - whole);
    // Past twice the exponents a double has, the term is past it whatever
    // the fraction; the cast is then of a number an `i32` holds.
    capped(libm::scalbn(m, whole.clamp(-4096.0, 4096.0) as i32))
}

/// The returns' weight `W` and squared weights `V`, the first two sums.
const NW: usize = 0;
const W2: usize = 1;
/// The sums before the terms' own.
const BASE: usize = 2;

/// `μ_p = E|Z|^p = 2^(p/2) Γ((p + 1)/2) / Γ(1/2)` for a standard normal `Z`:
/// exact at `p = 2` (1), correctly rounded at `p = 1` (`√(2/π)`), and
/// carried down by `μ_p = (p − 1) μ_{p−2}` from above 2, so the even moments
/// (`μ_4 = 3`) are exact too.
pub fn abs_normal_moment(p: f64) -> f64 {
    if p > 2.0 {
        return (p - 1.0) * abs_normal_moment(p - 2.0);
    }
    if p == 2.0 {
        1.0
    } else if p == 1.0 {
        // √(2/π) = 0.797884560802865355879892..., correctly rounded.
        0.797_884_560_802_865_4
    } else {
        libm::pow(2.0, p / 2.0) * libm::tgamma((p + 1.0) / 2.0) / std::f64::consts::PI.sqrt()
    }
}

/// `|r|^p`, exact where `p` is 1 or 2, the `libm` crate's `pow` (the same
/// bits on every platform) otherwise.
fn power(a: f64, p: f64) -> f64 {
    if p == 1.0 {
        a
    } else if p == 2.0 {
        a * a
    } else {
        libm::pow(a, p)
    }
}

#[derive(Debug, Clone, PartialEq)]
enum Form {
    Multi {
        powers: Vec<f64>,
        /// `(Σp/2 − 1, Π μ_{p_i}^-1)` where the term is normalized.
        norm: Option<(f64, f64)>,
    },
    Kernel {
        /// `2 c_h` for `h = 1..=H`.
        c2: Vec<f64>,
        center: bool,
    },
}

#[derive(Debug, Clone, PartialEq)]
struct TermPlan {
    /// The scale's ring.
    slot: usize,
    scale: usize,
    /// The term's first sum.
    at: usize,
    form: Form,
}

/// What the settings fix, made from them and never saved: the distinct
/// scales and the `K`-step returns each keeps, the raw returns kept, each
/// term's sums and constants.
#[derive(Debug, Clone, Default, PartialEq)]
pub(super) struct Plan {
    scales: Vec<usize>,
    /// The `K`-step returns a scale's ring keeps, the newest included.
    keep: Vec<usize>,
    /// The raw returns kept before the row's: the largest scale less 1.
    raw: usize,
    terms: Vec<TermPlan>,
    /// Every sum a series keeps, `W` and `V` included.
    sums: usize,
}

impl Plan {
    /// The plan of valid terms.
    pub(super) fn of(terms: &[PvTerm]) -> Self {
        let mut plan = Plan {
            sums: BASE,
            ..Plan::default()
        };
        for t in terms {
            let k = t.scale.max(1);
            let slot = match plan.scales.iter().position(|&s| s == k) {
                Some(i) => i,
                None => {
                    plan.scales.push(k);
                    plan.keep.push(1);
                    plan.scales.len() - 1
                }
            };
            plan.raw = plan.raw.max(k - 1);
            let (lags, form, sums) = match &t.kind {
                PvTermKind::Multipower { powers, normalize } => {
                    let norm = normalize.then(|| {
                        let half: f64 = powers.iter().sum::<f64>() / 2.0 - 1.0;
                        let mu: f64 = powers.iter().map(|&p| abs_normal_moment(p)).product();
                        (half, 1.0 / mu)
                    });
                    (
                        powers.len().saturating_sub(1),
                        Form::Multi {
                            powers: powers.clone(),
                            norm,
                        },
                        1,
                    )
                }
                PvTermKind::Kernel {
                    kernel,
                    bandwidth,
                    flat_top,
                    center,
                } => {
                    let h_max = *bandwidth;
                    let c2 = (1..=h_max)
                        .map(|h| {
                            let x = if *flat_top {
                                (h - 1) as f64 / h_max as f64
                            } else {
                                h as f64 / (h_max + 1) as f64
                            };
                            2.0 * kernel.weight(x)
                        })
                        .collect();
                    (
                        h_max,
                        Form::Kernel {
                            c2,
                            center: *center,
                        },
                        if *center { 6 } else { 1 },
                    )
                }
            };
            plan.keep[slot] = plan.keep[slot].max(lags * k + 1);
            plan.terms.push(TermPlan {
                slot,
                scale: k,
                at: plan.sums,
                form,
            });
            plan.sums += sums;
        }
        plan
    }

    /// The bytes of a series outside a window: its sums, the raw returns
    /// and the scales' rings.
    pub(super) fn series_bytes(&self) -> u64 {
        16 * self.sums as u64 + 8 * self.raw as u64 + 8 * self.keep.iter().sum::<usize>() as u64
    }

    /// The bytes of a series' entry for a row a window holds: its sequence,
    /// weight and share of each term's sums.
    pub(super) fn entry_bytes(&self) -> u64 {
        16 + 16 * (self.sums - BASE) as u64
    }
}

/// A return a window holds: its row and its weight. The terms credited to
/// it are in [`Series::credits`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Entry {
    seq: u64,
    w: f64,
}

/// Where a row's terms go: each sum, and under a window the credits of the
/// return each term's oldest return is, `back` returns before the row's.
struct Sink<'a> {
    hi: &'a mut [f64],
    lo: &'a mut [f64],
    /// The window's credits and how many returns it holds; `None` without
    /// a window.
    credits: Option<(&'a mut VecDeque<f64>, usize)>,
    stride: usize,
}

impl Sink<'_> {
    fn add(&mut self, at: usize, back: usize, x: f64) {
        add(&mut self.hi[at], &mut self.lo[at], x);
        if let Some((c, held)) = self.credits.as_mut() {
            let i = (*held - 1 - back) * self.stride + 2 * (at - BASE);
            let mut pair = (c[i], c[i + 1]);
            add(&mut pair.0, &mut pair.1, x);
            (c[i], c[i + 1]) = pair;
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(super) struct Series {
    /// The sums' high parts: `W`, `V`, then each term's.
    hi: Vec<f64>,
    /// What each high part leaves out ([`add`]).
    lo: Vec<f64>,
    /// The last raw returns learned, oldest first.
    raw: VecDeque<f64>,
    /// Each scale's last `K`-step returns, oldest first.
    rings: Vec<VecDeque<f64>>,
    /// A window's returns of this series, oldest first.
    #[serde(default)]
    held: VecDeque<Entry>,
    /// The terms credited to each return `held` holds, the terms whose
    /// oldest return it is, as `(hi, lo)` pairs, one a sum after `W` and
    /// `V`: `2 (T − 2)` numbers a return for `T` sums, oldest first, in one
    /// ring so that a row takes no allocation of its own.
    #[serde(default)]
    credits: VecDeque<f64>,
}

impl Series {
    pub(super) fn new(plan: &Plan) -> Self {
        Self {
            hi: vec![0.0; plan.sums],
            lo: vec![0.0; plan.sums],
            raw: VecDeque::new(),
            rings: vec![VecDeque::new(); plan.scales.len()],
            held: VecDeque::new(),
            credits: VecDeque::new(),
        }
    }

    fn sum(&self, k: usize) -> f64 {
        self.hi[k] + self.lo[k]
    }

    /// The weight of the series' returns.
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

    /// The count of returns `n`: Kish's effective count `(Σw)² / Σw²` of
    /// the returns' weights, decayed or in the window, which is the count of
    /// returns at equal weights. 0 before a return.
    pub(super) fn count(&self) -> f64 {
        let w2 = self.sum(W2);
        if w2 > 0.0 {
            let w = self.sum(NW);
            w * w / w2
        } else {
            0.0
        }
    }

    /// The return `r` learned at the weight `w > 0`, a window's row `seq`
    /// where there is a window.
    pub(super) fn learn(&mut self, plan: &Plan, r: f64, w: f64, seq: Option<u64>) {
        let windowed = seq.is_some();
        let stride = 2 * (plan.sums - BASE);
        if let Some(seq) = seq {
            self.held.push_back(Entry { seq, w });
            self.credits.extend(std::iter::repeat_n(0.0, stride));
        }
        add(&mut self.hi[NW], &mut self.lo[NW], w);
        add(&mut self.hi[W2], &mut self.lo[W2], w * w);
        // Each scale's `K`-step return ending here, summed oldest first:
        // made once `K - 1` returns are before the row.
        let made = |raw: usize, k: usize| raw + 1 >= k;
        for (slot, &k) in plan.scales.iter().enumerate() {
            if made(self.raw.len(), k) {
                // Compensated and rounded once: the terms of a `K`-step
                // return cancel, and a plain sum's roundings are then a
                // large share of what is left.
                let (mut hi, mut lo) = (0.0, 0.0);
                for &x in self.raw.iter().skip(self.raw.len() + 1 - k) {
                    add(&mut hi, &mut lo, x);
                }
                add(&mut hi, &mut lo, r);
                let acc = hi + lo;
                let ring = &mut self.rings[slot];
                ring.push_back(acc);
                while ring.len() > plan.keep[slot] {
                    ring.pop_front();
                }
            }
        }
        let held = self.held.len();
        // A term is in a window while its oldest return, `back` before the
        // row's, is held; always without a window.
        let holds = |back: usize| !windowed || held > back;
        let raw = self.raw.len();
        let Series {
            hi,
            lo,
            rings,
            credits,
            ..
        } = self;
        let mut sink = Sink {
            hi,
            lo,
            credits: windowed.then_some((credits, held)),
            stride,
        };
        for t in &plan.terms {
            if !made(raw, t.scale) {
                continue;
            }
            let ring = &rings[t.slot];
            let (k, len) = (t.scale, ring.len());
            match &t.form {
                Form::Multi { powers, .. } => {
                    let m = powers.len();
                    let back = m * k - 1;
                    if len < (m - 1) * k + 1 || !holds(back) {
                        continue;
                    }
                    let at = |i: usize| ring[len - 1 - (m - 1 - i) * k].abs();
                    let mut acc = 1.0;
                    for (i, &p) in powers.iter().enumerate() {
                        acc *= power(at(i), p);
                    }
                    let mut x = w * acc;
                    if !x.is_finite() {
                        let factors: Vec<(f64, f64)> = powers
                            .iter()
                            .enumerate()
                            .map(|(i, &p)| (at(i), p))
                            .collect();
                        x = product_past_range(w, &factors);
                    }
                    sink.add(t.at, back, x);
                }
                Form::Kernel { c2, center } => {
                    let rt = ring[len - 1];
                    if !holds(k - 1) {
                        continue;
                    }
                    // A `K`-step return sums `K` returns, so its square
                    // times a weight can pass the largest double: held at
                    // it ([`capped`]), as every product below.
                    sink.add(t.at, k - 1, capped(w * (rt * rt)));
                    if *center {
                        sink.add(t.at + 1, k - 1, w * rt);
                        sink.add(t.at + 2, k - 1, w * rt);
                        sink.add(t.at + 3, k - 1, w);
                        sink.add(t.at + 4, k - 1, w * rt);
                        sink.add(t.at + 5, k - 1, w);
                    }
                    for (h, &c) in (1..).zip(c2) {
                        let back = (h + 1) * k - 1;
                        if len < h * k + 1 || !holds(back) {
                            break;
                        }
                        if c == 0.0 {
                            continue;
                        }
                        let x = ring[len - 1 - h * k];
                        sink.add(t.at, back, capped(w * (c * (rt * x))));
                        if *center {
                            sink.add(t.at + 1, back, w * (c * rt));
                            sink.add(t.at + 2, back, w * (c * x));
                            sink.add(t.at + 3, back, w * c);
                        }
                    }
                }
            }
        }
        if plan.raw > 0 {
            self.raw.push_back(r);
            while self.raw.len() > plan.raw {
                self.raw.pop_front();
            }
        }
    }

    /// The window's row `seq` has left: this series' return there leaves,
    /// with every term credited to it.
    pub(super) fn drop_front(&mut self, seq: u64) {
        if self.held.front().is_none_or(|e| e.seq != seq) {
            return;
        }
        let Some(e) = self.held.pop_front() else {
            return;
        };
        let mut again = vec![false; self.hi.len()];
        self.take(NW, e.w, 0.0, &mut again);
        self.take(W2, e.w * e.w, 0.0, &mut again);
        let stride = 2 * (self.hi.len() - BASE);
        for i in 0..stride / 2 {
            let (Some(h), Some(l)) = (self.credits.pop_front(), self.credits.pop_front()) else {
                break;
            };
            self.take(BASE + i, h, l, &mut again);
        }
        if self.held.is_empty() {
            self.hi.fill(0.0);
            self.lo.fill(0.0);
            self.credits.clear();
            return;
        }
        for k in (0..again.len()).filter(|&k| again[k]) {
            self.rebuild(k, stride);
        }
    }

    /// The credit `h + l` of sum `k` taken back as its return leaves the
    /// window, marked in `again` to be summed again from the credits the
    /// window still holds where taking it back cannot leave the sum: a sum
    /// held at the largest double ([`add`]) is a bound, not a sum, so a
    /// credit large enough to move it (a unit in the last place of the
    /// largest double, or more) is not taken from it, and a smaller one
    /// leaves the bound within that unit; and a credit that leaves too
    /// little of the sum for its pair to keep ([`lost`]).
    fn take(&mut self, k: usize, h: f64, l: f64, again: &mut [bool]) {
        let held = self.hi[k].abs() == f64::MAX;
        if !held {
            add(&mut self.hi[k], &mut self.lo[k], -h);
            add(&mut self.hi[k], &mut self.lo[k], -l);
        }
        if (held && h.abs() >= f64::MAX * f64::EPSILON) || (!held && lost(h, self.sum(k))) {
            again[k] = true;
        }
    }

    /// Sum `k` summed again from what the window holds: the weights of its
    /// returns, or the credits of each.
    fn rebuild(&mut self, k: usize, stride: usize) {
        let (mut hi, mut lo) = (0.0, 0.0);
        for (j, e) in self.held.iter().enumerate() {
            let (h, l) = match k {
                NW => (e.w, 0.0),
                W2 => (e.w * e.w, 0.0),
                _ => {
                    let i = j * stride + 2 * (k - BASE);
                    (self.credits[i], self.credits[i + 1])
                }
            };
            add(&mut hi, &mut lo, h);
            add(&mut hi, &mut lo, l);
        }
        (self.hi[k], self.lo[k]) = (hi, lo);
    }

    /// The numbers the series holds, counted: what [`Plan::series_bytes`]
    /// and [`Plan::entry_bytes`] bound, 8 bytes each.
    #[cfg(test)]
    pub(super) fn numbers(&self) -> usize {
        self.hi.len()
            + self.lo.len()
            + self.raw.len()
            + self.rings.iter().map(VecDeque::len).sum::<usize>()
            + 2 * self.held.len()
            + self.credits.len()
    }

    /// Whether a restored series has the shape its settings give it.
    pub(super) fn has_shape(&self, plan: &Plan, rows: Option<&VecDeque<Held>>) -> bool {
        let sums = self.hi.len() == plan.sums && self.lo.len() == plan.sums;
        let rings = self.rings.len() == plan.scales.len()
            && self
                .rings
                .iter()
                .zip(&plan.keep)
                .all(|(r, &k)| r.len() <= k)
            && self.raw.len() <= plan.raw;
        let held = match rows {
            None => self.held.is_empty() && self.credits.is_empty(),
            Some(rows) => {
                self.credits.len() == self.held.len() * 2 * (plan.sums - BASE)
                    && self
                        .held
                        .iter()
                        .zip(self.held.iter().skip(1))
                        .all(|(a, b)| a.seq < b.seq)
                    && self
                        .held
                        .iter()
                        .all(|e| rows.binary_search_by_key(&e.seq, |h| h.seq).is_ok())
            }
        };
        sums && rings && held
    }

    /// The series' outputs into `out`: each term, then `n`.
    pub(super) fn read(&self, plan: &Plan, min_weight: f64, out: &mut Vec<f64>) {
        if self.sum(NW) < min_weight {
            out.extend(std::iter::repeat_n(f64::NAN, plan.terms.len() + 1));
            return;
        }
        let n = self.count();
        // The period the sums are of, in returns: their weight `W`, which
        // is `n` at unit weights without a decay.
        let period = self.sum(NW);
        for t in &plan.terms {
            let k = t.scale as f64;
            let v = match &t.form {
                Form::Multi { norm, .. } => {
                    let s = self.sum(t.at) / k;
                    match norm {
                        // `W^e` has no value at `W = 0` for `e < 0`; at
                        // `e = 0` it is 1 (`[2]`, `[1, 1]`: no factor of
                        // `W`, as `realized`'s `rv` and `bv` read 0 there).
                        Some((e, _)) if period <= 0.0 && *e < 0.0 => f64::NAN,
                        Some((e, c)) => s * (c * libm::pow(period, *e)),
                        None => s,
                    }
                }
                Form::Kernel { center, .. } => {
                    let s = self.sum(t.at);
                    let s = if *center {
                        let m0 = self.sum(t.at + 5);
                        let m = if m0 > 0.0 {
                            self.sum(t.at + 4) / m0
                        } else {
                            0.0
                        };
                        let (a, b, c) =
                            (self.sum(t.at + 1), self.sum(t.at + 2), self.sum(t.at + 3));
                        s - m * (a + b) + (m * m) * c
                    } else {
                        s
                    };
                    s / k
                }
            };
            out.push(v);
        }
        out.push(n);
    }
}

#[cfg(test)]
#[path = "series_tests.rs"]
mod tests;
