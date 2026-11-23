//! `ewm_moments`: exponentially weighted central moments of a column to any
//! order, and the statistics read from them -- raw moments, standardized
//! moments (skew, kurtosis), cumulants, and the unbiased forms of orders 2
//! to 4 -- from one bounded state, in one pass.
//!
//! # The definitions
//!
//! After row `T`, row `t` holds the weight `ω_t = w_t · Π_{s>t} λ_s`, its
//! own weight times every later row's decay, and with `W = Σ ω_t`:
//!
//! ```text
//! mean           m       = Σ ω_t x_t / W
//! central        μ_p     = Σ ω_t (x_t − m)^p / W
//! raw            μ'_p    = Σ ω_t x_t^p / W = Σ_j C(p, j) m^(p−j) μ_j
//! standardized   s_p     = μ_p / μ_2^(p/2)
//! cumulant       κ_p     = μ_p − Σ_{j=2}^{p−2} C(p−1, j−1) κ_j μ_{p−j}
//! ```
//!
//! These are frequency-weighted plug-in moments: with every weight 1 and no
//! decay they are scipy's `moment`, `skew(bias=True)` and
//! `kurtosis(bias=True)`.
//!
//! # The update
//!
//! The state is `W`, the mean as a pair `m + m_lo` ([`crate::comp`]), the
//! normalised central moments `c_p = μ_p` for `p = 2..P`, and Kish's ratio
//! `r_2 = Σ ω_t² / W²` (`1 / r_2` is the effective number of rows), with
//! `r_3 = Σ ω³ / W³` and `r_4 = Σ ω⁴ / W⁴` where an unbiased third or fourth
//! moment is asked for. A row of weight `w` after a decay `λ` is Pébay's
//! merge (Pébay, Terriberry, Kolla and Bennett 2016) of the decayed summary
//! with the one-row summary `(w, x)`:
//!
//! ```text
//! W'   = λ W + w        a = λ W / W'        b = w / W'        (a + b = 1)
//! d    = x − m                                       (against the old mean)
//! c'_p = Σ_{β=0, p−β≠1}^{p} C(p, β) a c_{p−β} (−b d)^β  +  b (a d)^p     (c_0 = 1)
//! c'_2 = a c_2 + a b d²                              (`EwDiag`'s expression)
//! r'_k = a^k r_k + b^k
//! m'   = m + b d
//! ```
//!
//! Every term is a product of quantities at the scale of the spread -- the
//! deviation `d`, the shares `a` and `b` in `[0, 1]`, the past central
//! moments -- so nothing carries the level, and a column at `1e6` with a
//! spread of 1 keeps its fourth moment to about `1e-10`, where the raw
//! power sums keep no digit of it. Decay multiplies `W` alone: every `c_p`
//! and `r_k` is a ratio that a uniform rescaling of the weights leaves
//! where it was, so a long gap moves no moment and underflows none.
//!
//! Order 2 is [`crate::EwDiag`]'s update, operation for operation, so the
//! variance here is `ew_cov`'s to the bit; the higher orders are the merge
//! as written, each power formed by repeated multiplication.
//!
//! **A row of weight 0** decays `W` and changes nothing else: `b` is 0, so
//! no moment moves and the mean takes no step. Where `W'` is 0 -- a
//! zero-weight first row, or every row so far of weight 0 -- `a` and `b`
//! are both 0/0, so the update stops at that condition and the state stays
//! empty: no NaN enters it.
//!
//! # The unbiased forms
//!
//! For rows drawn independently under weights with ratios `r_k`, the
//! plug-in moments' expectations are
//!
//! ```text
//! E[c_2]   = (1 − r_2) μ_2
//! E[c_3]   = (1 − 3 r_2 + 2 r_3) μ_3
//! E[c_4]   = (1 − 4 r_2 + 6 r_3 − 3 r_4) μ_4 + (6 r_2 − 6 r_3 + 9 r_4 − 9 r_2²) μ_2²
//! E[c_2²]  = (r_2 − 2 r_3 + r_4) μ_4 + (1 − 3 r_2 + 2 r_3 − 3 r_4 + 3 r_2²) μ_2²
//! ```
//!
//! and inverting them gives the unbiased `M_2` (Kish), `M_3` and `M_4`
//! (Rimoldini 2014, "Weighted skewness and kurtosis unbiased by sample
//! size and Gaussian uncertainties", for reliability weights) and an
//! unbiased `μ_2²`; the unbiased fourth cumulant is `M_4 − 3 (μ_2²)`. With
//! every weight 1 these are scipy's `kstat`, `skew(bias=False)` and
//! `kurtosis(bias=False)`. Above order 4 no unbiased form is offered, and a
//! term asking for one is refused by name.
//!
//! # Withholding
//!
//! A term is withheld, with its reason ([`Withheld`]), where it is not
//! defined, and never by a count of rows: no weight yet; a weight below
//! `min_weight`; a standardized moment of a column with no spread; an
//! unbiased form whose denominator keeps no digit of the terms it is formed
//! from (one row carrying all the weight, or the weight on too few rows,
//! read from the weights themselves); a moment past a double's range.
//!
//! # The state's size
//!
//! `8 · (3 + (P − 1) + n_r)` bytes: `W`, the mean's pair, the moments of
//! orders 2 to `P` (none where every term is of order 1) and the `n_r` of
//! `r_2`, `r_3`, `r_4` kept. It grows with the terms asked for, never with
//! the stream, and is checked against `max_state_bytes` before the first
//! row. Plug-in orders 2 to 4 take 40 to 56 bytes.

use serde::{Deserialize, Serialize};

use crate::Decay;

/// The default limit on the state, as the other models' `max_state_bytes`:
/// 256 KB.
pub const EWM_MOMENTS_MAX_STATE_BYTES: u64 = 256 * 1024;

/// The highest order an unbiased form reaches: 2 (Kish), 3 and 4
/// (Rimoldini). A term asking for one above it is refused by name.
pub const MAX_UNBIASED_MOMENT_ORDER: u32 = 4;

/// A denominator of an unbiased form within this many epsilons of the sum
/// of its terms' sizes keeps no digit of them, and the term is withheld.
const NO_DIGIT: f64 = 64.0 * f64::EPSILON;

/// Which statistic a term reads from the moments.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MomentKind {
    /// `μ_p`, about the weighted mean.
    Central,
    /// `μ'_p`, about 0.
    Raw,
    /// `μ_p / μ_2^(p/2)`: skew at 3, kurtosis (not excess) at 4.
    Standardized,
    /// `κ_p`: the mean at 1, the variance at 2, excess kurtosis times
    /// `μ_2²` at 4.
    Cumulant,
}

/// One output: a statistic of the column at an order, plug-in (`bias =
/// true`) or unbiased.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MomentTerm {
    /// The output's name, which every refusal of the term names.
    pub name: String,
    /// The order `p`, from 1.
    pub order: u32,
    pub kind: MomentKind,
    /// `true` for the plug-in moment, `false` for the unbiased form (orders
    /// 2 to 4; a raw moment is unbiased either way).
    pub bias: bool,
}

impl MomentTerm {
    /// Each statistic's default `bias`, as Polars': a variance (a central
    /// moment or cumulant of order 2) is unbiased, as `ewm_var`'s default;
    /// skew, kurtosis and every other order are the plug-in, as `skew` and
    /// `kurtosis` default.
    pub fn default_bias(kind: MomentKind, order: u32) -> bool {
        !(matches!(kind, MomentKind::Central | MomentKind::Cumulant) && order == 2)
    }
}

/// The settings of [`EwmMoments`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EwmMomentsCfg {
    pub decay: Decay,
    /// The outputs, in slot order.
    pub terms: Vec<MomentTerm>,
    /// Every term is withheld while `weight_sum` is below it.
    pub min_weight: f64,
    /// The limit on the state's bytes; `None` for
    /// [`EWM_MOMENTS_MAX_STATE_BYTES`].
    pub max_state_bytes: Option<u64>,
}

impl EwmMomentsCfg {
    /// The highest central moment the state keeps: the highest order asked
    /// for, and 2 for a standardized term; 1 where no moment is kept.
    fn top(&self) -> u32 {
        self.terms
            .iter()
            .map(|t| match t.kind {
                MomentKind::Standardized => t.order.max(2),
                _ => t.order,
            })
            .max()
            .unwrap_or(1)
            .max(1)
    }

    /// How many of `r_2`, `r_3`, `r_4` the state keeps: `r_2` always, and
    /// up to the highest order of an unbiased term.
    fn n_r(&self) -> usize {
        let unbiased = self
            .terms
            .iter()
            .filter(|t| !t.bias && t.kind != MomentKind::Raw)
            .map(|t| t.order)
            .max()
            .unwrap_or(0);
        match unbiased {
            0..=2 => 1,
            3 => 2,
            _ => 3,
        }
    }

    /// The doubles the state keeps: `W`, the mean's pair, the central
    /// moments of orders 2 to the top, and the `r_k`.
    fn numbers(&self) -> u64 {
        3 + u64::from(self.top() - 1) + self.n_r() as u64
    }

    /// The state's bytes: `8 · (3 + (P − 1) + n_r)`, fixed by the terms.
    pub fn state_bytes(&self) -> u64 {
        8 * self.numbers()
    }

    fn cap(&self) -> u64 {
        self.max_state_bytes.unwrap_or(EWM_MOMENTS_MAX_STATE_BYTES)
    }

    pub fn validate(&self) -> Result<(), String> {
        let mut no = crate::Refusals::new();
        no.take(self.decay.check().map_err(|e| format!("ewm_moments: {e}")));
        if self.terms.is_empty() {
            no.refuse(
                "ewm_moments: terms names no term. Name at least one, such as a third central \
                 moment of order 3.",
            );
        }
        no.take(crate::model::check_min_weight(
            "ewm_moments",
            self.min_weight,
        ));
        if self.max_state_bytes == Some(0) {
            no.refuse(
                "ewm_moments: max_state_bytes must be above 0, but it is 0. Give a limit in \
                 bytes, or leave it unset for 256 KB.",
            );
        }
        for (i, t) in self.terms.iter().enumerate() {
            let name = &t.name;
            if self.terms[..i].iter().any(|u| u.name == *name) {
                no.refuse(format!(
                    "ewm_moments: the term {name:?} is named twice. Give each term its own name."
                ));
            }
            if t.order == 0 {
                no.refuse(format!(
                    "ewm_moments: the term {name:?} has order 0; an order is a whole number \
                     from 1."
                ));
            }
            if t.kind == MomentKind::Standardized && t.order == 1 {
                no.refuse(format!(
                    "ewm_moments: the term {name:?} is standardized at order 1, which is 0 by \
                     definition. Give it an order from 2."
                ));
            }
            if !t.bias && t.kind != MomentKind::Raw && t.order > MAX_UNBIASED_MOMENT_ORDER {
                no.refuse(format!(
                    "ewm_moments: the term {name:?} asks for bias=False at order {}; the \
                     unbiased forms stop at order {MAX_UNBIASED_MOMENT_ORDER}. Give bias=True for it.",
                    t.order
                ));
            }
        }
        no.stop_here()?;
        let (bytes, cap) = (self.state_bytes(), self.cap());
        if bytes > cap {
            no.refuse(format!(
                "ewm_moments: the state takes {bytes} bytes, over max_state_bytes ({cap}): \
                 its moments to order {} are {} doubles. Ask for lower orders, or raise \
                 max_state_bytes.",
                self.top(),
                self.numbers()
            ));
        }
        no.finish()
    }
}

/// Why a term reports no value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Withheld {
    /// No row of weight has been learned.
    NoWeight,
    /// `weight_sum` is below `min_weight`.
    BelowMinWeight,
    /// A standardized moment of a column with no spread.
    ZeroSpread,
    /// An unbiased form when one row carries all the weight (`r_2 = 1`).
    OneRow,
    /// An unbiased third moment with the weight on too few rows.
    TooFewRowsThird,
    /// An unbiased fourth moment with the weight on too few rows.
    TooFewRowsFourth,
    /// The moment is past a double's range.
    Overflow,
}

impl Withheld {
    /// The reason, in the words an output gives it.
    pub fn reason(self) -> &'static str {
        match self {
            Self::NoWeight => "no weight yet",
            Self::BelowMinWeight => "weight_sum is below min_weight",
            Self::ZeroSpread => "the spread of the column is zero",
            Self::OneRow => "one row carries all the weight, so the unbiased form is not defined",
            Self::TooFewRowsThird => "the weight sits on too few rows for an unbiased third moment",
            Self::TooFewRowsFourth => {
                "the weight sits on too few rows for an unbiased fourth moment"
            }
            Self::Overflow => "the moment overflowed a double",
        }
    }
}

/// `κ_p` as a polynomial in the central moments: each monomial (the orders
/// of its factors, sorted) with its coefficient, in the order the recursion
/// first meets it.
type Poly = Vec<(Vec<u32>, f64)>;

/// The cumulants' polynomials, one a term (empty for a term that is not a
/// plug-in cumulant of order 4 or more), formed from the settings and so no
/// part of the state: every two are equal.
#[derive(Debug, Clone, Default)]
struct Polys(Vec<Poly>);

impl PartialEq for Polys {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

impl Polys {
    fn of(cfg: &EwmMomentsCfg) -> Self {
        Self(
            cfg.terms
                .iter()
                .map(|t| {
                    if t.kind == MomentKind::Cumulant && t.bias && t.order >= 2 {
                        cumulant_poly(t.order)
                    } else {
                        Vec::new()
                    }
                })
                .collect(),
        )
    }
}

/// `C(n, k)` for `k = 0..=n` in turn: exact while the integers fit, then
/// by the floating-point recurrence.
struct Binomials {
    n: u32,
    k: u32,
    exact: Option<u128>,
    approx: f64,
}

impl Binomials {
    fn new(n: u32) -> Self {
        Self {
            n,
            k: 0,
            exact: Some(1),
            approx: 1.0,
        }
    }

    /// `C(n, k)` at the current `k`.
    fn get(&self) -> f64 {
        match self.exact {
            Some(v) => v as f64,
            None => self.approx,
        }
    }

    /// Steps to `k + 1`.
    fn step(&mut self) {
        let (num, den) = (self.n - self.k, self.k + 1);
        self.approx = self.get() * f64::from(num) / f64::from(den);
        self.exact = self
            .exact
            .and_then(|v| v.checked_mul(u128::from(num)))
            .map(|v| v / u128::from(den));
        self.k += 1;
    }
}

fn binomial(n: u32, k: u32) -> f64 {
    let mut b = Binomials::new(n);
    for _ in 0..k {
        b.step();
    }
    b.get()
}

/// `κ_p` (`p >= 2`) by the recursion of the module docs, expanded into
/// central moments.
fn cumulant_poly(p: u32) -> Poly {
    let mut k: Vec<Poly> = vec![
        Vec::new(),
        Vec::new(),
        vec![(vec![2], 1.0)],
        vec![(vec![3], 1.0)],
    ];
    for n in 4..=p {
        let mut poly: Poly = vec![(vec![n], 1.0)];
        for j in 2..n - 1 {
            let c = binomial(n - 1, j - 1);
            for (mono, v) in &k[j as usize] {
                let mut key = mono.clone();
                key.push(n - j);
                key.sort_unstable();
                match poly.iter_mut().find(|(m, _)| *m == key) {
                    Some((_, e)) => *e -= c * v,
                    None => poly.push((key, 0.0 - c * v)),
                }
            }
        }
        k.push(poly);
    }
    k.swap_remove(p as usize)
}

/// Exponentially weighted moments of one column to any order. See the
/// module docs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EwmMoments {
    cfg: EwmMomentsCfg,
    /// `W`, the decayed sum of the weights (the `n_eff` count).
    w_sum: f64,
    /// The mean, `m + m_lo` ([`crate::comp`]).
    m: f64,
    m_lo: f64,
    /// `c_p` for `p = 2..=P`, at `p − 2`.
    c: Vec<f64>,
    /// `r_2`, then `r_3` and `r_4` where kept.
    r: Vec<f64>,
    #[serde(skip)]
    polys: Polys,
}

impl EwmMoments {
    pub fn new(cfg: EwmMomentsCfg) -> Result<Self, String> {
        cfg.validate()?;
        let n_c = (cfg.top() - 1) as usize;
        let n_r = cfg.n_r();
        let polys = Polys::of(&cfg);
        Ok(Self {
            cfg,
            w_sum: 0.0,
            m: 0.0,
            m_lo: 0.0,
            c: vec![0.0; n_c],
            r: vec![0.0; n_r],
            polys,
        })
    }

    pub fn cfg(&self) -> &EwmMomentsCfg {
        &self.cfg
    }

    /// `W`: the weight before the next row and its decay (hard rule 8).
    #[inline]
    pub fn n_eff(&self) -> f64 {
        self.w_sum
    }

    /// The mean, the pair's high part.
    pub fn mean(&self) -> f64 {
        self.m
    }

    /// The plug-in central moment of order `p` (0 at order 1; NaN where the
    /// state keeps none of that order).
    pub fn central(&self, p: u32) -> f64 {
        match p {
            0 => 1.0,
            1 => 0.0,
            _ => self.c.get(p as usize - 2).copied().unwrap_or(f64::NAN),
        }
    }

    /// Kish's ratio `r_2 = Σ ω² / W²`, `1 / n_eff_kish`.
    pub fn kish_ratio(&self) -> f64 {
        self.r[0]
    }

    /// The state's bytes, [`EwmMomentsCfg::state_bytes`].
    pub fn state_bytes(&self) -> u64 {
        self.cfg.state_bytes()
    }

    /// Whether the state's vectors are at the widths its settings give them.
    pub(crate) fn has_shape(&self) -> bool {
        self.c.len() == (self.cfg.top() - 1) as usize && self.r.len() == self.cfg.n_r()
    }

    /// One row of weight `w` after a decay of `lam` (from
    /// [`Decay::factor`]). A negative weight is a caller's bug (a debug
    /// assert, and no change in release); a row that leaves the weight at 0
    /// changes nothing else (hard rule 9).
    pub fn update(&mut self, x: f64, lam: f64, w: f64) {
        debug_assert!(
            w >= 0.0,
            "EwmMoments::update needs a weight of at least 0, got {w}"
        );
        if w < 0.0 {
            return;
        }
        let w_new = lam * self.w_sum + w;
        if w_new <= 0.0 {
            // `a` and `b` are 0/0: nothing carried and nothing added.
            self.w_sum = w_new;
            return;
        }
        let a = lam * self.w_sum / w_new;
        let b = w / w_new;
        if b > 0.0 {
            let d = crate::comp::dev(x, self.m, self.m_lo);
            self.merge_row(a, b, d);
            crate::comp::add(&mut self.m, &mut self.m_lo, b * d);
        }
        self.w_sum = w_new;
    }

    /// Pébay's merge of the history (share `a`) with a row at deviation `d`
    /// (share `b`): the orders from the top down to 3, each reading the
    /// lower orders before they move, then order 2 by `EwDiag`'s
    /// expression, then the `r_k`.
    fn merge_row(&mut self, a: f64, b: f64, d: f64) {
        let top = self.c.len() as u32 + 1;
        let (na, pb) = (-b * d, a * d);
        for p in (3..=top).rev() {
            let mut s = 0.0;
            let (mut qa, mut qb) = (1.0, 1.0);
            let mut coef = Binomials::new(p);
            for beta in 0..=p {
                if beta > 0 {
                    qa *= na;
                    qb *= pb;
                    coef.step();
                }
                let g = p - beta;
                if g == 1 {
                    continue;
                }
                let term = if g == 0 {
                    a * qa + b * qb
                } else {
                    a * self.c[g as usize - 2] * qa
                };
                s += coef.get() * term;
            }
            self.c[p as usize - 2] = s;
        }
        if let Some(c2) = self.c.first_mut() {
            *c2 = a * *c2 + a * b * d * d;
        }
        let (a2, b2) = (a * a, b * b);
        self.r[0] = a2 * self.r[0] + b2;
        if let Some(r3) = self.r.get_mut(1) {
            *r3 = a2 * a * *r3 + b2 * b;
        }
        if let Some(r4) = self.r.get_mut(2) {
            *r4 = a2 * a2 * *r4 + b2 * b2;
        }
    }

    /// Every term's value, or why it is withheld, read from the state as it
    /// stands.
    pub fn read(&self) -> Vec<Result<f64, Withheld>> {
        (0..self.cfg.terms.len()).map(|i| self.value(i)).collect()
    }

    /// Every term into `out`, a withheld one as NaN.
    pub(crate) fn read_into(&self, out: &mut Vec<f64>) {
        out.clear();
        out.extend((0..self.cfg.terms.len()).map(|i| self.value(i).unwrap_or(f64::NAN)));
    }

    /// Term `i`'s value, or why it is withheld.
    pub fn value(&self, i: usize) -> Result<f64, Withheld> {
        if self.w_sum == 0.0 {
            return Err(Withheld::NoWeight);
        }
        if self.w_sum < self.cfg.min_weight {
            return Err(Withheld::BelowMinWeight);
        }
        let v = self.evaluate(i)?;
        if v.is_finite() {
            Ok(v)
        } else {
            Err(Withheld::Overflow)
        }
    }

    fn evaluate(&self, i: usize) -> Result<f64, Withheld> {
        let t = &self.cfg.terms[i];
        let p = t.order;
        match t.kind {
            MomentKind::Raw => {
                let mut val = 0.0;
                let mut coef = Binomials::new(p);
                for j in 0..=p {
                    if j > 0 {
                        coef.step();
                    }
                    let mp = self.m.powf(f64::from(p - j));
                    val += coef.get() * mp * self.central(j);
                }
                Ok(val)
            }
            MomentKind::Central if p == 1 => Ok(0.0),
            MomentKind::Central if t.bias => Ok(self.central(p)),
            MomentKind::Central => Ok(self.unbiased(p)?.0),
            MomentKind::Standardized => self.standardized(p, t.bias),
            MomentKind::Cumulant if p == 1 => Ok(self.m),
            MomentKind::Cumulant if !t.bias => {
                let (v, u22) = self.unbiased(p)?;
                Ok(if p == 4 { v - 3.0 * u22 } else { v })
            }
            MomentKind::Cumulant => {
                let mut val = 0.0;
                for (mono, coef) in &self.polys.0[i] {
                    let mut term = *coef;
                    for &q in mono {
                        term *= self.central(q);
                    }
                    val += term;
                }
                Ok(val)
            }
        }
    }

    fn standardized(&self, p: u32, bias: bool) -> Result<f64, Withheld> {
        let var = self.central(2);
        if !(var > 0.0 && var.is_finite()) {
            return Err(Withheld::ZeroSpread);
        }
        if bias || p == 2 {
            return Ok(self.central(p) / var.sqrt().powf(f64::from(p)));
        }
        let (uvar, _) = self.unbiased(2)?;
        let (mut num, u22) = self.unbiased(p)?;
        if p == 4 {
            // Fisher's k4 over the unbiased variance, plus the normal's 3:
            // `G2 + 3` at equal weights, as scipy's `kurtosis(bias=False,
            // fisher=False)`.
            let u = var / (1.0 - self.r[0]);
            num = num - 3.0 * u22 + 3.0 * ((u * u + u * u + u * u) / 3.0);
        }
        Ok(num / uvar.sqrt().powf(f64::from(p)))
    }

    /// The unbiased central moment of order `p` (2 to 4) and, at 4, the
    /// unbiased `μ_2²`; or why it is not defined.
    fn unbiased(&self, p: u32) -> Result<(f64, f64), Withheld> {
        let r2 = self.r[0];
        let c = self.central(p);
        match p {
            2 => {
                let den = 1.0 - r2;
                if !kept(den, &[1.0, r2]) {
                    return Err(Withheld::OneRow);
                }
                Ok((c / den, 0.0))
            }
            3 => {
                let r3 = self.r[1];
                let den = 1.0 - 3.0 * r2 + 2.0 * r3;
                if !kept(den, &[1.0, 3.0 * r2, 2.0 * r3]) {
                    return Err(Withheld::TooFewRowsThird);
                }
                Ok((c / den, 0.0))
            }
            _ => {
                let (r3, r4) = (self.r[1], self.r[2]);
                let a1 = 1.0 - 4.0 * r2 + 6.0 * r3 - 3.0 * r4;
                let a2 = 6.0 * r2 - 6.0 * r3 + 9.0 * r4 - 9.0 * r2 * r2;
                let b1 = r2 - 2.0 * r3 + r4;
                let b2 = 1.0 - 3.0 * r2 + 2.0 * r3 - 3.0 * r4 + 3.0 * r2 * r2;
                let det = a1 * b2 - a2 * b1;
                if !kept(det, &[a1 * b2, a2 * b1]) {
                    return Err(Withheld::TooFewRowsFourth);
                }
                let c2 = self.central(2);
                let p22 = (c2 * c2 + c2 * c2 + c2 * c2) / 3.0;
                Ok(((b2 * c - a2 * p22) / det, (a1 * p22 - b1 * c) / det))
            }
        }
    }

    /// The settings and the state checked as a state file is loaded: the
    /// settings `new` takes, and the vectors at their widths.
    pub(crate) fn checked(mut self) -> Result<Self, crate::StateError> {
        crate::model::check_cfg("ewm_moments", self.cfg.validate())?;
        if !self.has_shape() {
            return Err(crate::model::damaged(
                "ewm_moments",
                "its running sums have the wrong shape",
            ));
        }
        self.polys = Polys::of(&self.cfg);
        Ok(self)
    }
}

/// The column is the first feature; there are no targets, and a slot a
/// term. As every model's, each row's output is read from the state before
/// the row is learned.
impl crate::OnlineModel for EwmMoments {
    fn step_into(
        &mut self,
        x: &[f64],
        y: &[Option<f64>],
        d_clock: f64,
        weight: f64,
        out: &mut crate::Step,
    ) {
        if let Some(refused) = crate::model::refused_step(self, x, y, d_clock, weight) {
            *out = refused;
            return;
        }
        self.read_into(&mut out.pred);
        out.n_eff = self.w_sum;
        out.extra = None;
        let lam = self.cfg.decay.factor(d_clock);
        debug_assert!(
            !x.is_empty(),
            "ewm_moments reads its column as the first feature"
        );
        match x.first() {
            Some(&v) => self.update(v, lam, weight),
            None => self.update(0.0, lam, 0.0),
        }
    }

    fn predict(&self, x: &[f64], d_clock: f64) -> crate::Step {
        if let Some(refused) = crate::model::refused_predict(self, x, d_clock) {
            return refused;
        }
        let mut out = crate::Step {
            n_eff: self.w_sum,
            ..crate::Step::default()
        };
        self.read_into(&mut out.pred);
        out
    }

    fn state(&self) -> crate::State {
        crate::State::new(crate::ModelState::EwmMoments(Box::new(self.clone())))
    }

    fn restore(s: &crate::State) -> Result<Self, crate::StateError> {
        Self::restore_owned(s.clone())
    }

    fn restore_owned(s: crate::State) -> Result<Self, crate::StateError> {
        crate::check_schema(&s)?;
        match s.model {
            crate::ModelState::EwmMoments(m) => m.checked(),
            other => Err(crate::StateError::WrongModel {
                expected: "ewm_moments",
                found: other.kind(),
            }),
        }
    }

    fn n_targets(&self) -> usize {
        0
    }

    fn n_features(&self) -> usize {
        1
    }

    fn n_outputs(&self) -> usize {
        self.cfg.terms.len()
    }
}

/// Whether a denominator keeps a digit of the terms it is formed from.
fn kept(den: f64, terms: &[f64]) -> bool {
    den > NO_DIGIT * terms.iter().fold(0.0, |s, t| s + t.abs())
}

#[cfg(test)]
mod tests;
