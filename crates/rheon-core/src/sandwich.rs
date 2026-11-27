//! `Sandwich`: heteroskedasticity- and autocorrelation-robust covariances of
//! an exponentially weighted least-squares fit, from its out-of-sample
//! residuals (docs/PLAN.md task 221 (c)).
//!
//! The fit is `β = B⁻¹ Σ ω z y`, with `z = (1, x − c)` -- the features about
//! an origin `c`, the first row's, which keeps the sums well conditioned
//! far from zero -- and `ω` each row's weight times the decay since it.
//! Its error is `B⁻¹ Σ ω z ε`, so its covariance is the sandwich
//!
//! ```text
//! B   = Σ ω z z'                                       the bread
//! M   = Σ_t Σ_s ω_t ω_s E[e_t e_s] z_t z_s'            the meat
//! Cov = B⁻¹ M B⁻¹
//! ```
//!
//! White's (1980) HC0 keeps the meat's diagonal, `Σ ω² e² z z'`; Newey and
//! West (1987) add each lag `l` up to `L` at Bartlett's weight `1 − l/(L+1)`:
//!
//! ```text
//! Γ_l = Σ_t ω_t ω_{t−l} e_t e_{t−l} z_t z_{t−l}'
//! M   = Γ_0 + Σ_{l=1..L} (1 − l/(L+1)) (Γ_l + Γ_l')
//! ```
//!
//! `t − l` is the `l`-th folded row before `t`. Each sum is kept as it
//! stands at the current row: on each step `B` decays by `λ`, `Γ_l` by
//! `λ²`, and each of the last `L` scores `u = ω e z` by `λ`, so the lag
//! products are those of the two rows' present weights. `e` is the row's
//! out-of-sample residual, the one the model had not learned, where the
//! classical sandwich has the in-sample residual of the final fit: larger
//! in expectation by its estimation error, `sqrt(1 + h)` on a row of
//! leverage `h`, which errs large while the fit is young and fades as it
//! settles. So the standard errors tend to read above the classical ones,
//! though not on every stream: run once on 1,500 rows over 50 seeds, the
//! ratio to statsmodels' ran 0.990-1.034 (HC0) and 0.985-1.059 (HAC).
//!
//! The covariance is mapped to the features' own units: the slopes are the
//! fit's, and the intercept `β_0 − c'β_x`, whose variance is
//! `v_00 − 2 c'v_0x + c'V_xx c`. With no decay and unit weights it is
//! `statsmodels`' `OLS.fit(cov_type="HC0")` and `cov_type="HAC"` with
//! `maxlags = L` on the same rows and residuals.

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

use crate::solve::{KEPT_ORDER, Solves, factorize_kept, kept_work, solve_in, solve_kept};

thread_local! {
    /// A row's `z` and its score `u`, as [`Sandwich::update`] forms them:
    /// kept by the thread and written over, no part of any state.
    static ROW: std::cell::RefCell<(Vec<f64>, Vec<f64>)> = const { std::cell::RefCell::new((Vec::new(), Vec::new())) };
}

/// The buffers a covariance is formed in ([`Sandwich::covariance`] and what
/// reads it), one set a thread, kept between calls and written over:
/// nothing in it is read between calls, so it is no part of any state.
#[derive(Default)]
struct CovScratch {
    solves: Solves,
    pos: Vec<usize>,
    bread: Vec<f64>,
    meat: Vec<f64>,
    col: Vec<f64>,
    left: Vec<f64>,
    left_t: Vec<f64>,
    v: Vec<f64>,
    out: Vec<f64>,
    c: Vec<f64>,
    row0: Vec<f64>,
    sub: Vec<f64>,
    x: Vec<f64>,
}

thread_local! {
    /// Lent to a read and left here after it ([`with_cov`]), never borrowed:
    /// a read of 256 coefficients or more solves on the pool's threads, and
    /// a thread waiting there takes other work, another read among it.
    static COV: std::cell::Cell<Option<Box<CovScratch>>> = const { std::cell::Cell::new(None) };
}

/// `f` in this thread's [`CovScratch`], which a covariance of `order`
/// coefficients or more leaves empty: a thread keeps buffers for the orders
/// a bank usually reads, not for a thousand features'.
///
/// The scratch is taken out of the thread's cell for the call and set back
/// after it, so nothing is borrowed while `f` runs: a read that starts on
/// this thread meanwhile (a pool thread waiting in a parallel solve takes
/// other work) finds the cell empty and forms its own, which the first then
/// replaces.
fn with_cov<T>(order: usize, f: impl FnOnce(&mut CovScratch) -> T) -> T {
    let mut sc = COV.take().unwrap_or_default();
    let out = f(&mut sc);
    if order >= KEPT_ORDER {
        *sc = CovScratch::default();
    }
    COV.set(Some(sc));
    out
}

/// Which heteroskedasticity-consistent (HC) covariance estimator a [`Sandwich`]
/// reads.
///
/// At row time, with out-of-sample residual `e` and pre-fold leverage
/// `g = w z' (λ B)⁻¹ z` (`h = g / (1 + g)`, `e_in = e / (1 + g)`):
///
/// - [`Self::Hc0`]: `e_in² = e² / (1 + g)²`
/// - [`Self::Hc1`]: `Hc0` scaled by `n / (n − p)` at Kish effective sample size `n = W² / Q`
/// - [`Self::Hc2`]: `e_in² / (1 − h) = e² / (1 + g)`
/// - [`Self::Hc3`]: `(e_in / (1 − h))² = e²` (the raw out-of-sample meat)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RobustSeType {
    Hc0,
    Hc1,
    Hc2,
    Hc3,
}

impl RobustSeType {
    /// Every HC variant in canonical order.
    pub const ALL: [Self; 4] = [Self::Hc0, Self::Hc1, Self::Hc2, Self::Hc3];

    /// The word a spec spells for this variant.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Hc0 => "hc0",
            Self::Hc1 => "hc1",
            Self::Hc2 => "hc2",
            Self::Hc3 => "hc3",
        }
    }

    /// Parse a spec word into a variant.
    pub fn from_name(s: &str) -> Option<Self> {
        match s {
            "hc0" => Some(Self::Hc0),
            "hc1" => Some(Self::Hc1),
            "hc2" => Some(Self::Hc2),
            "hc3" => Some(Self::Hc3),
            _ => None,
        }
    }
}

/// Optional row-time HC0–HC2 state kept beside [`Sandwich`]'s shared bread and
/// raw out-of-sample meat (`HC3*`). Absent when `robust_se_types` is unset so
/// existing states serialize to identical bytes and pay zero overhead.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct SandwichVariants {
    /// Active feature indices for this slot (`None` = all `0..k`).
    active: Option<Vec<usize>>,
    /// Keep `meat_hc0` (shared by `hc0` and `hc1`).
    need_hc0_meat: bool,
    /// Track `w_sum`, `q_sum`, `q_scale` for `hc1`'s Kish `n / (n - p)` factor.
    need_hc1_scale: bool,
    /// Keep `meat_hc2`.
    need_hc2_meat: bool,
    /// Packed upper-triangular Cholesky factor `R_b` of the active bread
    /// (`B_act = R_bᵀ R_b`), maintained in `O(p²)` via Givens rotations when
    /// `need_hc0_meat || need_hc2_meat`.
    bread_r: Vec<f64>,
    /// `Σ (ω e / (1 + g))² z z'`, row-major `d × d`.
    meat_hc0: Vec<f64>,
    /// `Σ ω² (e² / (1 + g)) z z'`, row-major `d × d`.
    meat_hc2: Vec<f64>,
    /// Decayed weight sum `W = Σ ω`.
    w_sum: f64,
    /// Decayed squared-weight sum `Q = Σ ω²` (`sqsum` scaled).
    q_sum: f64,
    q_scale: i32,
    /// Positive-weight rows folded into the bread.
    folded: u64,
    /// Positive-weight rows folded into the leverage-adjusted meats.
    meat_folded: u64,
}

impl SandwichVariants {
    fn new(k: usize, intercept: bool, types: &[RobustSeType], active: Option<Vec<usize>>) -> Self {
        let d = k + usize::from(intercept);
        let p = active.as_ref().map_or(k, Vec::len) + usize::from(intercept);
        let need_hc0_meat =
            types.contains(&RobustSeType::Hc0) || types.contains(&RobustSeType::Hc1);
        let need_hc1_scale = types.contains(&RobustSeType::Hc1);
        let need_hc2_meat = types.contains(&RobustSeType::Hc2);
        let need_chol = need_hc0_meat || need_hc2_meat;
        Self {
            active,
            need_hc0_meat,
            need_hc1_scale,
            need_hc2_meat,
            bread_r: if need_chol {
                vec![0.0; p * (p + 1) / 2]
            } else {
                Vec::new()
            },
            meat_hc0: if need_hc0_meat {
                vec![0.0; d * d]
            } else {
                Vec::new()
            },
            meat_hc2: if need_hc2_meat {
                vec![0.0; d * d]
            } else {
                Vec::new()
            },
            w_sum: 0.0,
            q_sum: 0.0,
            q_scale: 0,
            folded: 0,
            meat_folded: 0,
        }
    }

    fn active_dim(&self, k: usize, intercept: bool) -> usize {
        self.active.as_ref().map_or(k, Vec::len) + usize::from(intercept)
    }

    fn has_shape(
        &self,
        k: usize,
        intercept: bool,
        types: &[RobustSeType],
        active: Option<&[usize]>,
    ) -> bool {
        let d = k + usize::from(intercept);
        let need_hc0_meat =
            types.contains(&RobustSeType::Hc0) || types.contains(&RobustSeType::Hc1);
        let need_hc1_scale = types.contains(&RobustSeType::Hc1);
        let need_hc2_meat = types.contains(&RobustSeType::Hc2);
        let need_chol = need_hc0_meat || need_hc2_meat;
        let p = active.map_or(k, <[usize]>::len) + usize::from(intercept);
        self.active.as_deref() == active
            && self.need_hc0_meat == need_hc0_meat
            && self.need_hc1_scale == need_hc1_scale
            && self.need_hc2_meat == need_hc2_meat
            && self.bread_r.len() == if need_chol { p * (p + 1) / 2 } else { 0 }
            && self.meat_hc0.len() == if need_hc0_meat { d * d } else { 0 }
            && self.meat_hc2.len() == if need_hc2_meat { d * d } else { 0 }
            && self.w_sum.is_finite()
            && self.w_sum >= 0.0
            && self.q_sum.is_finite()
            && self.q_sum >= 0.0
            && crate::sqsum::scale_fits(self.q_scale)
            && self
                .bread_r
                .iter()
                .chain(&self.meat_hc0)
                .chain(&self.meat_hc2)
                .all(|v| v.is_finite())
    }
}

/// Packed upper-triangular Givens rank-1 update: `R'ᵀ R' = root² Rᵀ R + z zᵀ`.
/// Mirrors `rls::fold` for `O(p²)` orthogonal updates with `hypot` overflow/subnormal safety.
fn givens_fold_upper(r: &mut [f64], z: &mut [f64], root: f64, decays: bool) {
    let p = z.len();
    let mut rest = r;
    for i in 0..p {
        let (row, tail) = std::mem::take(&mut rest).split_at_mut(p - i);
        rest = tail;
        let zi = z[i];
        if zi == 0.0 {
            if decays {
                for v in row.iter_mut() {
                    *v *= root;
                }
            }
            continue;
        }
        let mut rii = row[0];
        if decays {
            rii *= root;
        }
        let sq = rii * rii + zi * zi;
        let mut rho = sq.sqrt();
        if !(sq >= f64::MIN_POSITIVE && sq.is_finite()) {
            rho = rii.hypot(zi);
            if !(rho > 0.0 && rho.is_finite()) {
                if decays {
                    for v in row.iter_mut() {
                        *v *= root;
                    }
                }
                continue;
            }
        }
        let c = rii / rho;
        let s = zi / rho;
        row[0] = rho;
        z[i] = 0.0;
        for (rij, zj) in row[1..].iter_mut().zip(z[i + 1..].iter_mut()) {
            let mut a = *rij;
            if decays {
                a *= root;
            }
            let b = *zj;
            *rij = c * a + s * b;
            *zj = c * b - s * a;
        }
    }
}

/// Pre-fold leverage `g = (w / lam) ‖R_b⁻ᵀ z_act‖²` against the packed upper
/// factor `bread_r`, or `None` if `bread_r` is rank-deficient or ill-conditioned.
fn prefold_leverage(bread_r: &[f64], z_act: &[f64], lam: f64, w: f64) -> Option<f64> {
    let p = z_act.len();
    if p == 0 || !(lam > 0.0 && lam.is_finite()) {
        return None;
    }
    // Check that every diagonal pivot of `R_b` is positive and non-degenerate.
    let mut max_diag = 0.0f64;
    let mut min_diag = f64::INFINITY;
    let mut at = 0usize;
    for i in 0..p {
        let d = bread_r[at];
        if !(d > 0.0 && d.is_finite()) {
            return None;
        }
        max_diag = max_diag.max(d);
        min_diag = min_diag.min(d);
        at += p - i;
    }
    if min_diag <= 1e-12 * max_diag || !min_diag.is_finite() {
        return None;
    }
    let mut tmp = vec![0.0; p];
    if !crate::lincon::solve_upper_t_packed(bread_r, p, z_act, &mut tmp) {
        return None;
    }
    let norm2: f64 = tmp.iter().map(|v| v * v).sum();
    let g = (w / lam) * norm2;
    (g.is_finite() && g >= 0.0).then_some(g)
}

/// The sandwich of one slot's fit. See the module docs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Sandwich {
    /// Features, and whether `z` leads with the intercept's 1.
    k: usize,
    intercept: bool,
    /// Newey and West's lags, `0` for HC0 alone.
    lags: usize,
    /// The origin the features are taken about: the first folded row's,
    /// empty before it (and always without an intercept, which cannot
    /// absorb a shift).
    origin: Vec<f64>,
    /// `B`, `Γ_0` and each `Γ_l`, row-major over `z`.
    bread: Vec<f64>,
    meat: Vec<f64>,
    gammas: Vec<Vec<f64>>,
    /// The last `lags` scores `u = ω e z`, newest first, at their present
    /// weights.
    ring: VecDeque<Vec<f64>>,
    /// Optional row-time HC0–HC2 state; last and the sole skipping field on
    /// `Sandwich` so compact and named msgpack both round-trip cleanly and
    /// pre-existing states keep identical bytes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    variants: Option<SandwichVariants>,
}

impl Sandwich {
    /// An empty sandwich over `k` features, with or without an intercept,
    /// keeping `lags` lags (legacy out-of-sample HC0 / HAC only).
    pub fn new(k: usize, intercept: bool, lags: usize) -> Self {
        let d = k + usize::from(intercept);
        Self {
            k,
            intercept,
            lags,
            origin: Vec::new(),
            bread: vec![0.0; d * d],
            meat: vec![0.0; d * d],
            gammas: vec![vec![0.0; d * d]; lags],
            ring: VecDeque::with_capacity(lags),
            variants: None,
        }
    }

    /// An empty sandwich tracking the requested row-time HC variants (`types`).
    /// `active` names the subset of `0..k` this slot's fit uses (`None` = all).
    pub fn with_types(
        k: usize,
        intercept: bool,
        lags: usize,
        types: &[RobustSeType],
        active: Option<Vec<usize>>,
    ) -> Self {
        let mut s = Self::new(k, intercept, lags);
        s.variants = Some(SandwichVariants::new(k, intercept, types, active));
        s
    }

    /// Heap bytes one `Sandwich` reserves for `(k, intercept, lags, types, active_k)`.
    pub fn state_bytes(
        k: usize,
        intercept: bool,
        lags: usize,
        types: Option<&[RobustSeType]>,
        active_k: Option<usize>,
    ) -> u64 {
        let d = (k + usize::from(intercept)) as u64;
        let d2 = d.saturating_mul(d);
        let l = lags as u64;
        // Base: bread + meat + L * (gamma_l + ring score).
        let base_f64 = 2u64
            .saturating_mul(d2)
            .saturating_add(l.saturating_mul(d2.saturating_add(d)));
        let extra_f64 = match types {
            None => 0u64,
            Some(ts) => {
                let need_hc0 = ts.contains(&RobustSeType::Hc0) || ts.contains(&RobustSeType::Hc1);
                let need_hc2 = ts.contains(&RobustSeType::Hc2);
                let p = (active_k.unwrap_or(k) + usize::from(intercept)) as u64;
                let chol = if need_hc0 || need_hc2 {
                    p.saturating_mul(p.saturating_add(1)) / 2
                } else {
                    0
                };
                let meats = u64::from(need_hc0)
                    .saturating_add(u64::from(need_hc2))
                    .saturating_mul(d2);
                meats.saturating_add(chol)
            }
        };
        base_f64.saturating_add(extra_f64).saturating_mul(8)
    }

    fn dim(&self) -> usize {
        self.k + usize::from(self.intercept)
    }

    /// Whether a restored sandwich is shaped as this configuration's: its
    /// sums sized for `z`, its lags and ring within them, every number
    /// finite.
    pub fn has_shape(&self, k: usize, intercept: bool, lags: usize) -> bool {
        self.has_shape_with_types(k, intercept, lags, None, None)
    }

    /// [`Self::has_shape`] checking optional `types` and `active` feature indices too.
    pub fn has_shape_with_types(
        &self,
        k: usize,
        intercept: bool,
        lags: usize,
        types: Option<&[RobustSeType]>,
        active: Option<&[usize]>,
    ) -> bool {
        let d = k + usize::from(intercept);
        let variants_ok = match (&self.variants, types) {
            (None, None) => true,
            (Some(v), Some(ts)) => v.has_shape(k, intercept, ts, active),
            _ => false,
        };
        variants_ok
            && self.k == k
            && self.intercept == intercept
            && self.lags == lags
            && (self.origin.is_empty() || self.origin.len() == k)
            && self.bread.len() == d * d
            && self.meat.len() == d * d
            && self.gammas.len() == lags
            && self.gammas.iter().all(|g| g.len() == d * d)
            && self.ring.len() <= lags
            && self.ring.iter().all(|u| u.len() == d)
            && self
                .bread
                .iter()
                .chain(&self.meat)
                .chain(self.gammas.iter().flatten())
                .chain(self.ring.iter().flatten())
                .chain(&self.origin)
                .all(|v| v.is_finite())
    }

    /// The clock moves on by a step whose decay is `lam`, with nothing
    /// folded: the sums keep their rows at their present weights.
    pub fn age(&mut self, lam: f64) {
        let lam2 = lam * lam;
        self.bread.iter_mut().for_each(|v| *v *= lam);
        self.meat.iter_mut().for_each(|v| *v *= lam2);
        self.gammas.iter_mut().flatten().for_each(|v| *v *= lam2);
        self.ring.iter_mut().flatten().for_each(|v| *v *= lam);
        if let Some(v) = &mut self.variants {
            v.meat_hc0.iter_mut().for_each(|m| *m *= lam2);
            v.meat_hc2.iter_mut().for_each(|m| *m *= lam2);
            if !v.bread_r.is_empty() && lam != 1.0 {
                let root = lam.sqrt();
                v.bread_r.iter_mut().for_each(|r| *r *= root);
            }
            if v.need_hc1_scale {
                v.w_sum *= lam;
                crate::sqsum::add(&mut v.q_sum, &mut v.q_scale, lam, 0.0);
            }
        }
    }

    /// The rows behind this one are no longer adjacent to it -- a capped gap
    /// or a session change -- so none of them is a lag of the next.
    pub fn clear_lags(&mut self) {
        self.ring.clear();
    }

    /// One row: its features, its out-of-sample residual, the step's decay
    /// and its weight. A residual or a feature that is not finite, or a
    /// weight of 0, only ages (CLAUDE.md hard rule 9), and so does a row
    /// whose score `u = w e z` or bread term `w z z'` would put a sum past
    /// `f64`: kept, its `inf` would stay in the sums for good.
    ///
    /// Each sum's decay and its row's term are one pass over it, `S ← λS +
    /// term`: the two roundings [`Self::age`] and an addition would make, in
    /// the same order, and the same bits, with the sums read once rather than
    /// twice. The row's `z` and `u` are written into buffers the thread
    /// keeps, and `u` goes on to the ring in the buffer of the score it
    /// pushes out, so a row allocates nothing once the ring is full.
    pub fn update(&mut self, x: &[f64], e: f64, lam: f64, w: f64) {
        if !(e.is_finite() && w > 0.0 && x.len() == self.k && x.iter().all(|v| v.is_finite())) {
            return self.age(lam);
        }
        if self.intercept && self.origin.is_empty() {
            self.origin = x.to_vec();
        }
        let d = self.dim();
        crate::scratch::with_borrow_mut(&ROW, |(z, u)| {
            z.clear();
            if self.intercept {
                z.push(1.0);
                z.extend(x.iter().zip(&self.origin).map(|(v, c)| v - c));
            } else {
                z.extend_from_slice(x);
            }
            let we = w * e;
            u.clear();
            u.extend(z.iter().map(|v| we * v));
            // The bread and the meat are sums of positive semidefinite
            // terms, so each entry is bounded by its diagonal ones
            // (Cauchy-Schwarz), and each lag sum by the meat's diagonal at
            // the two rows' present weights: a finite diagonal after the row
            // keeps every sum finite (docs/PLAN.md task 270, review round 9
            // B-2). The diagonals are read as the decay leaves them.
            let lam2 = lam * lam;
            let base_fits = (0..d).all(|i| {
                (self.bread[i * d + i] * lam + w * z[i] * z[i]).is_finite()
                    && (self.meat[i * d + i] * lam2 + u[i] * u[i]).is_finite()
            });
            let extra_fits = self.variants.as_ref().is_none_or(|v| {
                (!v.need_hc1_scale || (v.w_sum * lam + w).is_finite())
                    && (!v.need_hc0_meat
                        || (0..d).all(|i| (v.meat_hc0[i * d + i] * lam2 + u[i] * u[i]).is_finite()))
                    && (!v.need_hc2_meat
                        || (0..d).all(|i| (v.meat_hc2[i * d + i] * lam2 + u[i] * u[i]).is_finite()))
            });
            if !(base_fits && extra_fits) {
                return self.age(lam);
            }
            if let Some(v) = &mut self.variants {
                if !v.bread_r.is_empty() {
                    let off = usize::from(self.intercept);
                    let p = v.active_dim(self.k, self.intercept);
                    let mut z_act = Vec::with_capacity(p);
                    match &v.active {
                        None => z_act.extend_from_slice(&z[..d]),
                        Some(idx) => {
                            if self.intercept {
                                z_act.push(1.0);
                            }
                            for &fi in idx {
                                z_act.push(z[off + fi]);
                            }
                        }
                    }
                    let g = if v.folded >= p as u64 {
                        prefold_leverage(&v.bread_r, &z_act, lam, w)
                    } else {
                        None
                    };
                    if let Some(g) = g {
                        let inv_1pg = 1.0 / (1.0 + g);
                        if v.need_hc0_meat {
                            let c = inv_1pg;
                            for i in 0..d {
                                let u0i = c * u[i];
                                let at = i * d..(i + 1) * d;
                                for (m, &uj) in v.meat_hc0[at].iter_mut().zip(u.iter()) {
                                    *m = *m * lam2 + u0i * (c * uj);
                                }
                            }
                        }
                        if v.need_hc2_meat {
                            let c2 = inv_1pg;
                            for i in 0..d {
                                let u2i = c2 * u[i];
                                let at = i * d..(i + 1) * d;
                                for (m, &uj) in v.meat_hc2[at].iter_mut().zip(u.iter()) {
                                    *m = *m * lam2 + u2i * uj;
                                }
                            }
                        }
                        v.meat_folded = v.meat_folded.saturating_add(1);
                    } else {
                        v.meat_hc0.iter_mut().for_each(|m| *m *= lam2);
                        v.meat_hc2.iter_mut().for_each(|m| *m *= lam2);
                    }
                    let sw = w.sqrt();
                    for val in &mut z_act {
                        *val *= sw;
                    }
                    let decays = lam != 1.0;
                    let root = if decays { lam.sqrt() } else { 1.0 };
                    givens_fold_upper(&mut v.bread_r, &mut z_act, root, decays);
                }
                v.folded = v.folded.saturating_add(1);
                if v.need_hc1_scale {
                    v.w_sum = lam * v.w_sum + w;
                    crate::sqsum::add(&mut v.q_sum, &mut v.q_scale, lam, w);
                }
            }
            for i in 0..d {
                let (wz, ui) = (w * z[i], u[i]);
                let at = i * d..(i + 1) * d;
                for (b, &zj) in self.bread[at.clone()].iter_mut().zip(z.iter()) {
                    *b = *b * lam + wz * zj;
                }
                for (m, &uj) in self.meat[at].iter_mut().zip(u.iter()) {
                    *m = *m * lam2 + ui * uj;
                }
            }
            for (l, g) in self.gammas.iter_mut().enumerate() {
                match self.ring.get_mut(l) {
                    Some(back) => {
                        // The score as the step's decay leaves it.
                        back.iter_mut().for_each(|v| *v *= lam);
                        for (i, &ui) in u.iter().enumerate() {
                            for (g, &bj) in g[i * d..(i + 1) * d].iter_mut().zip(back.iter()) {
                                *g = *g * lam2 + ui * bj;
                            }
                        }
                    }
                    None => g.iter_mut().for_each(|v| *v *= lam2),
                }
            }
            if self.lags > 0 {
                let mut kept = if self.ring.len() >= self.lags {
                    self.ring.pop_back().unwrap_or_default()
                } else {
                    Vec::with_capacity(d)
                };
                kept.clear();
                kept.extend_from_slice(u);
                self.ring.push_front(kept);
            }
        });
    }

    /// The covariance of the coefficients over the features `idx`
    /// (positions in `x`), in the features' own units -- the intercept
    /// first where there is one, then each of `idx` -- with Newey and
    /// West's lags (`hac`) or HC0's diagonal alone, `p × p` row-major.
    /// `None` until a row is folded, and while the bread over `idx` does not
    /// factorize: fewer rows than coefficients, or a feature with no spread.
    pub fn covariance(&self, idx: &[usize], hac: bool) -> Option<Vec<f64>> {
        with_cov(idx.len() + usize::from(self.intercept), |sc| {
            self.covariance_with_meat(sc, idx, &self.meat, hac, false)?;
            Some(sc.out.clone())
        })
    }

    /// Covariance of the coefficients over `idx` for a specific row-time HC
    /// variant (`HC0*`–`HC3*`), in the features' own units (`p × p` row-major).
    pub fn covariance_of(&self, idx: &[usize], kind: RobustSeType) -> Option<Vec<f64>> {
        let p = idx.len() + usize::from(self.intercept);
        let (meat, scale) = self.variant_meat_and_scale(kind, p)?;
        with_cov(p, |sc| {
            self.covariance_with_meat(sc, idx, meat, false, false)?;
            if scale != 1.0 {
                for v in &mut sc.out {
                    *v *= scale;
                }
            }
            Some(sc.out.clone())
        })
    }

    /// Resolve the meat slice and scalar multiplier for a row-time HC variant.
    fn variant_meat_and_scale(&self, kind: RobustSeType, p: usize) -> Option<(&[f64], f64)> {
        match kind {
            RobustSeType::Hc3 => Some((&self.meat, 1.0)),
            RobustSeType::Hc0 => {
                let v = self.variants.as_ref()?;
                (v.need_hc0_meat && v.meat_folded > 0).then_some((v.meat_hc0.as_slice(), 1.0))
            }
            RobustSeType::Hc1 => {
                let v = self.variants.as_ref()?;
                if !(v.need_hc0_meat && v.need_hc1_scale && v.meat_folded > 0) {
                    return None;
                }
                let n = crate::sqsum::count(v.w_sum, v.q_sum, v.q_scale)?;
                let pf = p as f64;
                if !(n.is_finite() && n > pf) {
                    return None;
                }
                Some((v.meat_hc0.as_slice(), n / (n - pf)))
            }
            RobustSeType::Hc2 => {
                let v = self.variants.as_ref()?;
                (v.need_hc2_meat && v.meat_folded > 0).then_some((v.meat_hc2.as_slice(), 1.0))
            }
        }
    }

    /// [`Self::covariance`] into `sc.out`, in `sc`'s buffers: each matrix
    /// is written over one the thread kept, the lag sums are read where
    /// they lie rather than picked out into a matrix each, and the bread's
    /// factor serves both solves. The arithmetic is that of the allocating
    /// form, operation for operation. `centred` leaves the intercept about
    /// the features' origin, the frame the sums are kept in, rather than
    /// moving it to the features' own units.
    fn covariance_in(
        &self,
        sc: &mut CovScratch,
        idx: &[usize],
        hac: bool,
        centred: bool,
    ) -> Option<()> {
        self.covariance_with_meat(sc, idx, &self.meat, hac, centred)
    }

    fn covariance_with_meat(
        &self,
        sc: &mut CovScratch,
        idx: &[usize],
        meat_src: &[f64],
        hac: bool,
        centred: bool,
    ) -> Option<()> {
        let d = self.dim();
        let off = usize::from(self.intercept);
        let CovScratch {
            solves,
            pos,
            bread,
            meat,
            col,
            left,
            left_t,
            v,
            out,
            c,
            row0,
            ..
        } = sc;
        pos.clear();
        pos.extend((0..off).chain(idx.iter().map(|&i| i + off)));
        let p = pos.len();
        if p == 0 {
            return None;
        }
        let pick = |m: &[f64], into: &mut Vec<f64>| {
            into.clear();
            into.extend(
                pos.iter()
                    .flat_map(|&i| pos.iter().map(move |&j| m[i * d + j])),
            );
        };
        pick(&self.bread, bread);
        pick(meat_src, meat);
        if hac {
            let lags = self.lags as f64;
            for (l, g) in self.gammas.iter().enumerate() {
                let b = 1.0 - (l + 1) as f64 / (lags + 1.0);
                // Entries `(i, j)` and `(j, i)` add the one number `b (g_ij +
                // g_ji)`: a sum of the same two terms either way round.
                for i in 0..p {
                    for j in 0..=i {
                        let t = b * (g[pos[i] * d + pos[j]] + g[pos[j] * d + pos[i]]);
                        meat[i * p + j] += t;
                        if j != i {
                            meat[j * p + i] += t;
                        }
                    }
                }
            }
        }
        let work = kept_work(solves, p, p);
        if factorize_kept(bread, p, work)? != 0 {
            return None;
        }
        // `B⁻¹ M B⁻¹`: the meat's columns solved, transposed (`M B⁻¹`, `M`
        // symmetric), and solved again. Column-major in and out.
        col.clear();
        col.extend((0..p * p).map(|n| meat[(n % p) * p + n / p]));
        solve_kept(col, p, p, work, left);
        left_t.clear();
        left_t.extend((0..p * p).map(|n| left[(n % p) * p + n / p]));
        solve_kept(left_t, p, p, work, v);
        // Column-major `v`: entry (i, j) at `j·p + i`; symmetric.
        out.clear();
        out.extend((0..p * p).map(|n| {
            let (i, j) = (n / p, n % p);
            0.5 * (v[j * p + i] + v[i * p + j])
        }));
        if self.intercept && !centred {
            // The intercept in the features' own units is `β₀ − c'β_x`:
            // `T V T'` with `T = [[1, −c'], [0, I]]`.
            c.clear();
            c.push(0.0);
            c.extend(
                idx.iter()
                    .map(|&i| self.origin.get(i).copied().unwrap_or(0.0)),
            );
            // Row 0 of `T V`: `v_0j − Σ_a c_a v_aj`.
            row0.clear();
            row0.extend(
                (0..p).map(|j| out[j] - (1..p).map(|a| c[a] * out[a * p + j]).sum::<f64>()),
            );
            let v00 = row0[0] - (1..p).map(|b| c[b] * row0[b]).sum::<f64>();
            for j in 1..p {
                out[j] = row0[j];
                out[j * p] = row0[j];
            }
            out[0] = v00;
        }
        Some(())
    }

    /// The standard errors of the coefficients over the features `idx`
    /// (positions in `x`), in the layout `coef` reads -- the intercept
    /// first where there is one, then each of the `k` features, NaN for one
    /// outside `idx` -- with Newey and West's lags (`hac`) or HC0's
    /// diagonal alone. `None` as [`Self::covariance`] is.
    pub fn standard_errors(&self, idx: &[usize], hac: bool) -> Option<Vec<f64>> {
        let off = usize::from(self.intercept);
        let p = idx.len() + off;
        with_cov(p, |sc| {
            self.covariance_with_meat(sc, idx, &self.meat, hac, false)?;
            self.diag_se(&sc.out, idx, 1.0)
        })
    }

    /// Standard errors over `idx` for a specific row-time HC variant
    /// (`HC0*`–`HC3*`), laid out like `coef`.
    pub fn standard_errors_of(&self, idx: &[usize], kind: RobustSeType) -> Option<Vec<f64>> {
        let off = usize::from(self.intercept);
        let p = idx.len() + off;
        let (meat, scale) = self.variant_meat_and_scale(kind, p)?;
        with_cov(p, |sc| {
            self.covariance_with_meat(sc, idx, meat, false, false)?;
            self.diag_se(&sc.out, idx, scale)
        })
    }

    fn diag_se(&self, v: &[f64], idx: &[usize], scale: f64) -> Option<Vec<f64>> {
        let off = usize::from(self.intercept);
        let p = idx.len() + off;
        let mut out = vec![f64::NAN; self.k + off];
        // A rounding below 0 is 0, but a NaN variance stays NaN:
        // `f64::max` read it as 0.0, a standard error that says the
        // coefficient is exact (review round 9, B-2).
        for (a, &i) in idx.iter().enumerate() {
            out[off + i] = crate::solve::clamp_rounding(v[(a + off) * p + a + off] * scale).sqrt();
        }
        if self.intercept {
            out[0] = crate::solve::clamp_rounding(v[0] * scale).sqrt();
        }
        out.iter().any(|v| v.is_finite()).then_some(out)
    }

    /// Wald's statistic `d' V⁻¹ d` for the coefficients at `pick` --
    /// positions in [`Self::covariance`]'s layout over `idx` -- against the
    /// distance `d` of their estimates from the null, `V` the covariance
    /// with Newey and West's lags (`hac`). `None` where the covariance is
    /// not there, or its part over `pick` does not factorize.
    ///
    /// A test of every coefficient, the intercept among them, is read in
    /// the frame the sums are kept in, about the features' origin `c`:
    /// with `T = [[1, −c'], [0, I]]` taking it to the features' own units,
    /// `d' (T V_c T')⁻¹ d = (T⁻¹ d)' V_c⁻¹ (T⁻¹ d)`, `T⁻¹ d = [d₀ + c'd_x,
    /// d_x]`. `T V_c T'` grows the covariance's condition by the square of
    /// the features' level over their spread: a calibration of predictions
    /// at `1e8` with a spread of 1 did not factorize, and read `None` where
    /// the test is 37.5.
    pub fn wald(&self, idx: &[usize], hac: bool, pick: &[usize], d: &[f64]) -> Option<f64> {
        let p = idx.len() + usize::from(self.intercept);
        let every =
            self.intercept && pick.len() == p && pick.iter().enumerate().all(|(i, &j)| i == j);
        with_cov(p, |sc| {
            self.covariance_in(sc, idx, hac, every)?;
            let q = pick.len();
            if q == 0 || d.len() != q || pick.iter().any(|&i| i >= p) {
                return None;
            }
            // `T⁻¹ d`, on the stack for the calibration's two coefficients
            // and the few a test of every one has.
            let (mut short, mut long) = ([0.0f64; 8], Vec::new());
            let d = if every {
                let shifted = d[0]
                    + idx
                        .iter()
                        .zip(&d[1..])
                        .map(|(&i, &di)| self.origin.get(i).copied().unwrap_or(0.0) * di)
                        .sum::<f64>();
                let into: &mut [f64] = if q <= short.len() {
                    &mut short[..q]
                } else {
                    long.resize(q, 0.0);
                    &mut long
                };
                into[0] = shifted;
                into[1..].copy_from_slice(&d[1..]);
                &*into
            } else {
                d
            };
            let CovScratch {
                solves,
                out,
                sub,
                x,
                ..
            } = sc;
            let cov = &*out;
            sub.clear();
            sub.extend(
                pick.iter()
                    .flat_map(|&i| pick.iter().map(move |&j| cov[i * p + j])),
            );
            let attempts = solve_in(solves, sub, d, q, 1, x)?;
            let w: f64 = x.iter().zip(d).map(|(a, b)| a * b).sum();
            (attempts == 0 && w.is_finite()).then_some(w.max(0.0))
        })
    }
}

#[cfg(test)]
pub(crate) mod rounding;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::oracle::CanonBits;

    /// Whether this thread's covariance scratch holds anything.
    fn cov_kept() -> bool {
        let sc = COV.take();
        let kept = sc
            .as_ref()
            .is_some_and(|sc| !sc.solves.is_empty() || sc.bread.capacity() > 0);
        COV.set(sc);
        kept
    }

    /// A covariance of fewer than `KEPT_ORDER` coefficients leaves its
    /// buffers with the thread, one of that many or more leaves none --
    /// counting the intercept, and without one, the features alone.
    #[test]
    fn the_covariance_scratch_is_kept_below_the_order_and_released_at_it() {
        let mut st = 5u64;
        let mut fold = |k: usize, intercept: bool| {
            let mut s = Sandwich::new(k, intercept, 1);
            for _ in 0..k + 10 {
                let x: Vec<f64> = (0..k).map(|_| lcg(&mut st)).collect();
                s.update(&x, lcg(&mut st), 1.0, 1.0);
            }
            s
        };
        let small = fold(3, true);
        assert!(small.covariance(&[0, 1, 2], false).is_some());
        assert!(cov_kept());
        let big = fold(KEPT_ORDER - 1, true);
        let idx: Vec<usize> = (0..KEPT_ORDER - 1).collect();
        big.covariance(&idx, false);
        assert!(!cov_kept(), "{KEPT_ORDER} coefficients with the intercept");
        assert!(small.covariance(&[0, 1, 2], false).is_some());
        assert!(cov_kept());
        let bare = fold(KEPT_ORDER, false);
        let idx: Vec<usize> = (0..KEPT_ORDER).collect();
        bare.covariance(&idx, false);
        assert!(!cov_kept(), "{KEPT_ORDER} features without one");
        // One coefficient short of it, kept.
        let short = fold(KEPT_ORDER - 2, true);
        let idx: Vec<usize> = (0..KEPT_ORDER - 2).collect();
        short.covariance(&idx, false);
        assert!(cov_kept(), "{} coefficients", KEPT_ORDER - 1);
    }

    /// Task 458: a covariance of 256 coefficients or more solves on the
    /// pool's threads, and a thread waiting there takes other work -- a
    /// read of its own, which must not find this thread's scratch borrowed
    /// across the solve ("RefCell already borrowed", a panic in a bank on
    /// several threads) or write over it.
    #[test]
    fn a_covariance_read_on_a_pool_is_not_met_by_one_waiting_in_its_solve() {
        let k = 260;
        let mut st = 458u64;
        let mut s = Sandwich::new(k, false, 0);
        for _ in 0..2 * k {
            let x: Vec<f64> = (0..k).map(|_| lcg(&mut st)).collect();
            s.update(&x, lcg(&mut st), 1.0, 1.0);
        }
        let idx: Vec<usize> = (0..k).collect();
        let want = s.standard_errors(&idx, false).expect("a covariance");
        let (a, b) = crate::test_pool::read_met_by_a_read(|| {
            s.standard_errors(&idx, false).expect("a covariance")
        });
        for got in [a, b] {
            assert_eq!(got.len(), want.len());
            for (g, w) in got.iter().zip(&want) {
                assert_eq!(g.to_bits(), w.to_bits());
            }
        }
        // Each read leaves nothing behind at this order.
        assert!(!cov_kept());
    }

    /// A sum that the decay brings back under `f64`'s range takes the
    /// row: the check reads each diagonal as the decay leaves it, `λS + t`,
    /// not as `S` and `λ` and `t` in some other combination (a sum of
    /// the three, a quotient by the decay).
    #[test]
    fn a_row_is_judged_by_the_sums_as_the_decay_leaves_them() {
        let mut s = Sandwich::new(1, false, 0);
        s.update(&[1e154], 1.0, 1.0, 1.0);
        assert_eq!((s.bread[0], s.meat[0]), (1e154 * 1e154, 1e154 * 1e154));
        // The bread: 0.5e308 + 1.21e308 and the meat 0.25e308 + 1.46e308,
        // each under f64::MAX, each over it unless decayed first.
        let (x, e) = (1.1e154, 1.1);
        s.update(&[x], e, 0.5, 1.0);
        assert_eq!(s.bread[0], 1e308 * 0.5 + x * x);
        assert_eq!(s.meat[0], 1e308 * 0.25 + (e * x) * (e * x));
    }

    fn lcg(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    }

    /// The sandwich by its definition, from every folded row at its present
    /// weight: `X'ΩX`, the meat's double sum over pairs at most `L` folded
    /// rows apart at Bartlett's weights, inverted by faer's LU
    /// (`crate::oracle`), about the level `(50, −7)` and mapped to the
    /// features' own units -- not the model's origin, its first row -- under
    /// a decay, uneven weights, zeros and residuals that carry a lag.
    ///
    /// Task 274: each standard error within a bound on both sides'
    /// rounding ([`rounding::covariance_bound`]), the root halving its
    /// variance's relative error, where it was `1e-8` relative. Summed about
    /// 0, the definition's bread held the level's square, and its own
    /// rounding, `u` of entries 2,500 times the spread, was all it could
    /// see: 3e-11 of the model's numbers, against the model's 1e-15.
    /// Measured on aarch64 (macOS): at most 0.012 of the bound.
    #[test]
    fn the_sandwich_is_its_definition() {
        use rounding::{DefRow, Track, covariance_bound};
        let lags = 3usize;
        let lam = 0.99;
        let mut s = Sandwich::new(2, true, lags);
        let mut track = Track::new(2, lags);
        let mut st = 13u64;
        // Folded rows only, in order.
        let mut rows: Vec<DefRow> = Vec::new();
        let mut prev = 0.0;
        for i in 0..300 {
            let x = [lcg(&mut st) + 50.0, 2.0 * lcg(&mut st) - 7.0];
            let shock = lcg(&mut st) * (1.0 + (x[0] - 50.0).abs());
            let e = 0.6 * prev + shock;
            prev = e;
            let w = if i % 11 == 5 {
                0.0
            } else {
                0.5 + lcg(&mut st) + 0.5
            };
            s.update(&x, e, lam, w);
            track.update(&x, e, lam, w);
            rows.iter_mut().for_each(|r| r.w *= lam);
            if w > 0.0 {
                rows.push(DefRow {
                    x: x.to_vec(),
                    e,
                    w,
                    born: i,
                });
            }
            if i < 20 || i % 41 != 0 {
                continue;
            }
            for hac in [false, true] {
                let reach = if hac { lags } else { 0 };
                let b = covariance_bound(&track, &rows, i, &[0.0, 50.0, -7.0], (reach, lags));
                let (v, dv) = (b.v, b.dv);
                let got = s.standard_errors(&[0, 1], hac).unwrap();
                for a in 0..3 {
                    let want = v[a * 3 + a].sqrt();
                    let bound =
                        want * (0.5 * dv[a * 3 + a] / v[a * 3 + a] + 2.0 * crate::oracle::U);
                    crate::oracle::assert_within(
                        got[a],
                        want,
                        bound,
                        format_args!("row {i}, hac {hac}, coefficient {a}"),
                    );
                }
            }
        }
        assert!(s.has_shape(2, true, lags));
    }

    /// A feature set reads its own sub-block, NaN outside it; no intercept
    /// reads the raw features; a zero weight, the first included, only
    /// ages; a cleared ring adds no lag product across the break.
    #[test]
    fn sets_origins_zero_weights_and_breaks() {
        let mut s = Sandwich::new(2, false, 1);
        s.update(&[1.0, 2.0], 0.5, 0.9, 0.0);
        assert!(s.standard_errors(&[0, 1], true).is_none());
        assert_eq!(s, Sandwich::new(2, false, 1));
        let mut st = 3u64;
        for _ in 0..50 {
            let x = [lcg(&mut st), lcg(&mut st)];
            s.update(&x, lcg(&mut st), 1.0, 1.0);
        }
        let one = s.standard_errors(&[1], false).unwrap();
        assert!(one[0].is_nan() && one[1].is_finite());
        let before = s.gammas[0].clone();
        s.clear_lags();
        s.update(&[0.3, -0.2], 0.4, 1.0, 1.0);
        assert_eq!(s.gammas[0], before, "no lag product across a break");
        assert!(s.has_shape(2, false, 1));
    }

    /// A row whose score `u = w e z` would put a sum past `f64` -- one
    /// wild row whose square overflows, or a row the meat's running sum
    /// cannot take -- only ages, as a residual that is not finite does: the
    /// sandwich after it is the one that aged past it, and its errors stay
    /// numbers. Folded, its `inf` stayed in the meat for good and every
    /// later standard error read `NaN.max(0.0) = 0.0`, an exact coefficient
    /// (review round 9, B-2; docs/PLAN.md task 270).
    #[test]
    fn a_score_past_f64_is_refused() {
        let mut st = 29u64;
        let rows: Vec<(f64, f64)> = (0..80).map(|_| (lcg(&mut st), lcg(&mut st))).collect();
        let (mut s, mut aged) = (Sandwich::new(1, true, 2), Sandwich::new(1, true, 2));
        for (i, &(x, e)) in rows.iter().enumerate() {
            if i == 40 {
                s.update(&[1e60], 1e100, 1.0, 1.0);
                aged.update(&[1e60], f64::NAN, 1.0, 1.0);
            }
            s.update(&[x], e, 1.0, 1.0);
            aged.update(&[x], e, 1.0, 1.0);
        }
        assert_eq!(s, aged, "the wild row only aged");
        for hac in [false, true] {
            let se = s.standard_errors(&[0], hac).unwrap();
            assert!(se.iter().all(|v| v.is_finite() && *v > 0.0), "{se:?}");
        }
        // Each square finite, their sum not: the second is refused.
        let mut s = Sandwich::new(1, true, 0);
        s.update(&[0.0], 1e154, 1.0, 1.0);
        let one = s.clone();
        s.update(&[0.0], 1e154, 1.0, 1.0);
        assert_eq!(s, one);
        assert!(s.has_shape(1, true, 0));
    }

    /// The guard reads each sum's own diagonal after the row, `B_ii + w
    /// z_i²` and `M_ii + u_i²`, and refuses the row only when one would be
    /// past `f64`. With no intercept, `z = x` on two features, the second
    /// at `1e154`: its square `1e308` fits, a second row's does not, and
    /// the row only ages, while the off-diagonal `Σ w z_1 z_2`, at `1e154`,
    /// would take it; the same of the meat, with `u_2 = 1e154` from a
    /// feature of `1e100` and a residual of `1e54`. Two diagonals of
    /// `1e200` sum to `2e200`, a number, and the second row is folded,
    /// though their product is not one. A weight of `1e-10` on a feature of
    /// `1e155` gives a term `(w z) z = 1e300`, folded, though `z²` alone is
    /// past `f64`.
    #[test]
    fn a_row_is_refused_by_its_own_diagonals_after_it() {
        let refused = |x: [f64; 2], e: f64, w: f64| {
            let mut s = Sandwich::new(2, false, 0);
            s.update(&x, e, 1.0, w);
            let one = s.clone();
            assert!(s.has_shape(2, false, 0), "the first row is folded");
            s.update(&x, e, 1.0, w);
            assert!(s.has_shape(2, false, 0), "{x:?}, e {e}: every sum a number");
            s == one
        };
        assert!(refused([1.0, 1e154], 1e-200, 1.0), "the bread's diagonal");
        assert!(refused([1.0, 1e100], 1e54, 1.0), "the meat's diagonal");
        assert!(!refused([1.0, 1e100], 1e-200, 1.0), "two terms of 1e200");
        let mut s = Sandwich::new(2, false, 0);
        s.update(&[1.0, 1e100], 1e-200, 1.0, 1.0);
        s.update(&[1.0, 1e100], 1e-200, 1.0, 1.0);
        assert_eq!(s.bread[3], 2e200);
        let mut s = Sandwich::new(2, false, 0);
        s.update(&[1.0, 1e155], 1e-200, 1.0, 1e-10);
        assert_eq!(s.bread[3], 1e-10 * 1e155 * 1e155);
        assert!(s.bread[3] > 1e299 && s.bread[3].is_finite());
    }

    /// A variance that is not a number -- here `B⁻¹ M B⁻¹` past `f64`, its
    /// intercept mapped by `inf − inf` -- is no standard error, not one of
    /// 0.0, which reads as an exact coefficient (review round 9, B-2).
    #[test]
    fn a_variance_that_is_not_a_number_is_no_standard_error() {
        let mut s = Sandwich::new(1, true, 0);
        let mut st = 31u64;
        for _ in 0..20 {
            s.update(&[5.0 + 1e-9 * lcg(&mut st)], 1e150 * lcg(&mut st), 1.0, 1.0);
        }
        let v = s.covariance(&[0], false).unwrap();
        assert!(
            v.iter().any(|v| v.is_nan()),
            "the case: a NaN variance, {v:?}"
        );
        let se = s.standard_errors(&[0], false);
        let se = se.as_deref().unwrap_or(&[]);
        assert!(se.iter().all(|v| *v != 0.0), "{se:?}");
    }

    /// [`Sandwich::update`] as it was before its decay and its row's term
    /// shared a pass: the decay ([`Sandwich::age`]), then each sum's term
    /// added, the row's `z` and `u` each a vector of their own, the ring's
    /// new score pushed and the oldest truncated.
    fn update_by_parts(s: &mut Sandwich, x: &[f64], e: f64, lam: f64, w: f64) {
        s.age(lam);
        if !(e.is_finite() && w > 0.0 && x.len() == s.k && x.iter().all(|v| v.is_finite())) {
            return;
        }
        if s.intercept && s.origin.is_empty() {
            s.origin = x.to_vec();
        }
        let d = s.dim();
        let mut z = Vec::with_capacity(d);
        if s.intercept {
            z.push(1.0);
            z.extend(x.iter().zip(&s.origin).map(|(v, c)| v - c));
        } else {
            z.extend_from_slice(x);
        }
        let u: Vec<f64> = z.iter().map(|v| w * e * v).collect();
        let fits = (0..d).all(|i| {
            (s.bread[i * d + i] + w * z[i] * z[i]).is_finite()
                && (s.meat[i * d + i] + u[i] * u[i]).is_finite()
        });
        if !fits {
            return;
        }
        for i in 0..d {
            for j in 0..d {
                s.bread[i * d + j] += w * z[i] * z[j];
                s.meat[i * d + j] += u[i] * u[j];
            }
        }
        for (g, back) in s.gammas.iter_mut().zip(&s.ring) {
            for i in 0..d {
                for j in 0..d {
                    g[i * d + j] += u[i] * back[j];
                }
            }
        }
        if s.lags > 0 {
            s.ring.push_front(u);
            s.ring.truncate(s.lags);
        }
    }

    /// Every number a sandwich holds, as bits.
    fn bits(s: &Sandwich) -> Vec<u64> {
        s.origin
            .iter()
            .chain(&s.bread)
            .chain(&s.meat)
            .chain(s.gammas.iter().flatten())
            .chain(s.ring.iter().flatten())
            .map(|v| v.canon_bits())
            .collect()
    }

    /// A row's decay and its term in one pass give the sums the two passes
    /// gave, to the bit, whatever the row: with and without an intercept,
    /// at no lags and several, under a decay of 1 or below, through zero
    /// weights, a residual or a feature that is not finite, a row whose
    /// score overflows a sum, and a cleared ring.
    #[test]
    fn a_row_in_one_pass_is_the_row_aged_then_added_to_the_bit() {
        for (k, intercept, lags) in [(3, true, 0), (3, true, 4), (2, false, 2), (1, true, 1)] {
            let (mut fused, mut parts) = (
                Sandwich::new(k, intercept, lags),
                Sandwich::new(k, intercept, lags),
            );
            let mut st = 41u64;
            for i in 0..300 {
                let mut x: Vec<f64> = (0..k).map(|_| lcg(&mut st) + 30.0).collect();
                let mut e = lcg(&mut st);
                let mut w = 0.5 + lcg(&mut st);
                let lam = [0.97, 1.0, 0.5, 0.999][i % 4];
                match i % 29 {
                    3 => w = 0.0,
                    5 => e = f64::NAN,
                    7 => x[0] = f64::INFINITY,
                    11 => {
                        x[0] = 1e160;
                        e = 1e160;
                    }
                    13 => w = -1.0,
                    _ => {}
                }
                if i == 150 {
                    fused.clear_lags();
                    parts.clear_lags();
                }
                fused.update(&x, e, lam, w);
                update_by_parts(&mut parts, &x, e, lam, w);
                assert_eq!(
                    bits(&fused),
                    bits(&parts),
                    "row {i}, {k} {intercept} {lags}"
                );
            }
            assert!(fused.has_shape(k, intercept, lags));
        }
    }

    /// [`Sandwich::covariance`] and [`Sandwich::wald`] as they were written
    /// before their buffers were kept: each matrix a vector of its own, the
    /// bread factorized by [`crate::SpdFactor::of`] and solved against
    /// twice, each lag sum picked out.
    impl Sandwich {
        fn covariance_alloc(&self, idx: &[usize], hac: bool) -> Option<Vec<f64>> {
            self.covariance_alloc_in(idx, hac, false)
        }

        fn covariance_alloc_in(&self, idx: &[usize], hac: bool, centred: bool) -> Option<Vec<f64>> {
            let d = self.dim();
            let off = usize::from(self.intercept);
            let pos: Vec<usize> = (0..off).chain(idx.iter().map(|&i| i + off)).collect();
            let p = pos.len();
            if p == 0 {
                return None;
            }
            let pick = |m: &[f64]| -> Vec<f64> {
                pos.iter()
                    .flat_map(|&i| pos.iter().map(move |&j| m[i * d + j]))
                    .collect()
            };
            let bread = pick(&self.bread);
            let mut meat = pick(&self.meat);
            if hac {
                let lags = self.lags as f64;
                for (l, g) in self.gammas.iter().enumerate() {
                    let b = 1.0 - (l + 1) as f64 / (lags + 1.0);
                    let g = pick(g);
                    for i in 0..p {
                        for j in 0..p {
                            meat[i * p + j] += b * (g[i * p + j] + g[j * p + i]);
                        }
                    }
                }
            }
            let factor = crate::SpdFactor::of(&bread, p).filter(|f| f.attempts() == 0)?;
            // `B⁻¹ M B⁻¹`: the meat's columns solved, transposed (`M B⁻¹`, `M`
            // symmetric), and solved again. Column-major in and out.
            let col =
                |m: &[f64]| -> Vec<f64> { (0..p * p).map(|n| m[(n % p) * p + n / p]).collect() };
            let left = factor.solve(&col(&meat), p, p);
            let left_t: Vec<f64> = (0..p * p).map(|n| left[(n % p) * p + n / p]).collect();
            let v = factor.solve(&left_t, p, p);
            // Column-major `v`: entry (i, j) at `j·p + i`; symmetric.
            let mut out: Vec<f64> = (0..p * p)
                .map(|n| {
                    let (i, j) = (n / p, n % p);
                    0.5 * (v[j * p + i] + v[i * p + j])
                })
                .collect();
            if self.intercept && !centred {
                // The intercept in the features' own units is `β₀ − c'β_x`:
                // `T V T'` with `T = [[1, −c'], [0, I]]`.
                let c: Vec<f64> = std::iter::once(0.0)
                    .chain(
                        idx.iter()
                            .map(|&i| self.origin.get(i).copied().unwrap_or(0.0)),
                    )
                    .collect();
                // Row 0 of `T V`: `v_0j − Σ_a c_a v_aj`.
                let row0: Vec<f64> = (0..p)
                    .map(|j| out[j] - (1..p).map(|a| c[a] * out[a * p + j]).sum::<f64>())
                    .collect();
                let v00 = row0[0] - (1..p).map(|b| c[b] * row0[b]).sum::<f64>();
                for j in 1..p {
                    out[j] = row0[j];
                    out[j * p] = row0[j];
                }
                out[0] = v00;
            }
            Some(out)
        }

        fn wald_alloc(&self, idx: &[usize], hac: bool, pick: &[usize], d: &[f64]) -> Option<f64> {
            let p = idx.len() + usize::from(self.intercept);
            // A test of every coefficient reads the frame the sums are kept
            // in, `d` moved there by `T⁻¹`.
            let every = self.intercept && pick.iter().copied().eq(0..p);
            let v = self.covariance_alloc_in(idx, hac, every)?;
            let q = pick.len();
            if q == 0 || d.len() != q || pick.iter().any(|&i| i >= p) {
                return None;
            }
            let moved: Vec<f64>;
            let d = if every {
                let c0: f64 = idx
                    .iter()
                    .zip(&d[1..])
                    .map(|(&i, &di)| self.origin.get(i).copied().unwrap_or(0.0) * di)
                    .sum();
                moved = std::iter::once(d[0] + c0)
                    .chain(d[1..].iter().copied())
                    .collect();
                &moved[..]
            } else {
                d
            };
            let v = &v;
            let sub: Vec<f64> = pick
                .iter()
                .flat_map(|&i| pick.iter().map(move |&j| v[i * p + j]))
                .collect();
            let (x, attempts) = crate::solve_spd(&sub, d, q, 1)?;
            let w: f64 = x.iter().zip(d).map(|(a, b)| a * b).sum();
            (attempts == 0 && w.is_finite()).then_some(w.max(0.0))
        }
    }

    /// The covariance, the standard errors and Wald's statistic read in the
    /// thread's kept buffers are the allocating forms', to the bit, over
    /// every subset of features, HC0 and Newey and West, with and without an
    /// intercept, on a sandwich young and old, one that does not factorize
    /// (a constant feature, an exact copy) and one folded through rows that
    /// only age.
    #[test]
    fn the_covariance_in_kept_buffers_is_the_allocating_one_to_the_bit() {
        let same = |a: Option<&[f64]>, b: Option<&[f64]>| {
            a.map(|v| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>())
                == b.map(|v| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>())
        };
        let mut seen = 0;
        for (intercept, lags, copy, constant) in [
            (true, 0, false, false),
            (true, 3, false, false),
            (false, 2, false, false),
            (true, 2, true, false),
            (true, 2, false, true),
        ] {
            let mut s = Sandwich::new(4, intercept, lags);
            let mut st = 53u64;
            for i in 0..200 {
                let a = lcg(&mut st);
                let x = [
                    a,
                    lcg(&mut st) + 5.0,
                    if copy { a } else { lcg(&mut st) },
                    if constant { 2.0 } else { lcg(&mut st) },
                ];
                let w = if i % 13 == 4 { 0.0 } else { 0.5 + lcg(&mut st) };
                s.update(&x, lcg(&mut st) * (1.0 + a.abs()), 0.98, w);
                if i % 9 != 0 && i > 5 {
                    continue;
                }
                for idx in [&[0, 1, 2, 3][..], &[2, 0], &[1], &[3, 1, 0], &[]] {
                    for hac in [false, true] {
                        let (got, want) = (s.covariance(idx, hac), s.covariance_alloc(idx, hac));
                        assert!(same(got.as_deref(), want.as_deref()), "row {i} {idx:?}");
                        seen += usize::from(got.is_some());
                        let off = usize::from(intercept);
                        let p = idx.len() + off;
                        // Picks at the last coefficient, every one, and a
                        // position past them (refused).
                        for pick in [vec![p.saturating_sub(1)], (0..p).collect(), vec![p]] {
                            if p == 0 {
                                continue;
                            }
                            let d: Vec<f64> = pick.iter().map(|&j| 0.1 * (j + 1) as f64).collect();
                            let (g, w) = (
                                s.wald(idx, hac, &pick, &d),
                                s.wald_alloc(idx, hac, &pick, &d),
                            );
                            assert_eq!(
                                g.map(f64::to_bits),
                                w.map(f64::to_bits),
                                "wald {i} {idx:?}"
                            );
                        }
                    }
                }
            }
        }
        assert!(seen > 400, "the covariances reached numbers: {seen}");
    }

    /// A sandwich over `k` rows of `(1, x)` at `lags` lags, its ring full.
    fn folded(k: usize, lags: usize, rows: usize, seed: u64) -> Sandwich {
        let mut s = Sandwich::new(k, true, lags);
        let mut st = seed;
        for _ in 0..rows {
            let x: Vec<f64> = (0..k).map(|_| lcg(&mut st) + 3.0).collect();
            s.update(&x, lcg(&mut st), 0.95, 1.0);
        }
        s
    }

    /// The shape a restored sandwich must hold, one part broken at a time:
    /// the configuration it was made for, each sum's size, the ring's depth
    /// and width, and every number finite. A fresh one and one with its
    /// ring full both hold it.
    #[test]
    fn each_part_of_the_shape_is_checked_alone() {
        assert!(Sandwich::new(2, true, 2).has_shape(2, true, 2));
        let s = folded(2, 2, 6, 17);
        assert_eq!((s.origin.len(), s.ring.len()), (2, 2));
        assert!(s.has_shape(2, true, 2));
        type Edit<'a> = (&'a str, &'a dyn Fn(&mut Sandwich));
        let broken: [Edit; 15] = [
            ("k", &|s| s.k = 3),
            ("intercept", &|s| s.intercept = false),
            ("lags", &|s| s.lags = 3),
            ("origin", &|s| {
                s.origin.pop();
            }),
            ("bread", &|s| {
                s.bread.pop();
            }),
            ("meat", &|s| {
                s.meat.pop();
            }),
            ("gammas", &|s| {
                s.gammas.pop();
            }),
            ("a gamma", &|s| {
                s.gammas[1].pop();
            }),
            ("ring depth", &|s| s.ring.push_back(vec![0.0; 3])),
            ("ring width", &|s| s.ring[1].push(0.0)),
            ("bread NaN", &|s| s.bread[4] = f64::NAN),
            ("meat inf", &|s| s.meat[8] = f64::INFINITY),
            ("gamma NaN", &|s| s.gammas[1][3] = f64::NAN),
            ("ring NaN", &|s| s.ring[1][2] = f64::NAN),
            ("origin NaN", &|s| s.origin[1] = f64::NAN),
        ];
        for (what, edit) in broken {
            let mut t = s.clone();
            edit(&mut t);
            assert!(!t.has_shape(2, true, 2), "{what}");
        }
        assert!(!s.has_shape(2, true, 3) && !s.has_shape(2, false, 2));
    }

    /// Wald's statistic is `d' V⁻¹ d` over the picked block of the
    /// covariance, solved by faer's LU (`crate::oracle`); it refuses an
    /// empty pick, a distance of another length, a position outside the
    /// covariance, and a block that does not factorize -- a fit whose
    /// residuals were all 0, so the meat is.
    #[test]
    fn wald_is_its_definition_and_refuses_what_it_cannot_test() {
        let s = folded(2, 2, 40, 23);
        let idx = [0, 1];
        let v = s.covariance(&idx, true).unwrap();
        let pick = [1, 2];
        let d = [0.3, -0.2];
        let v = &v;
        let sub: Vec<f64> = pick
            .iter()
            .flat_map(|&i| pick.iter().map(move |&j| v[i * 3 + j]))
            .collect();
        let x = crate::oracle::solve(&sub, &d);
        let want: f64 = x.iter().zip(&d).map(|(a, b)| a * b).sum();
        let got = s.wald(&idx, true, &pick, &d).unwrap();
        // Task 274: the model's Cholesky form of the same block against the
        // oracle's LU form, each side's rounding bounded (`crate::oracle`);
        // it was `1e-10` relative.
        // Measured on aarch64 (macOS): equal to the bit; the result moved by 100u of itself
        // fails it (6.5 times the bound).
        let bound = crate::oracle::quad_error(&sub, &d, &crate::oracle::cholesky_backward(&sub))
            + crate::oracle::quad_form_error(&sub, &d);
        crate::oracle::assert_within(got, want, bound, "wald");
        assert!(s.wald(&idx, true, &[], &[]).is_none(), "nothing picked");
        assert!(s.wald(&idx, true, &pick, &d[..1]).is_none(), "d too short");
        assert!(s.wald(&idx, true, &[0, 3], &d).is_none(), "outside");
        let mut quiet = Sandwich::new(2, true, 2);
        let mut st = 29u64;
        for _ in 0..40 {
            let x = [lcg(&mut st), lcg(&mut st)];
            quiet.update(&x, 0.0, 0.95, 1.0);
        }
        let v = quiet.covariance(&idx, true).unwrap();
        assert!(v.iter().all(|&a| a == 0.0), "no residual, no meat");
        assert!(quiet.wald(&idx, true, &pick, &d).is_none());
    }

    /// Task 252: with no intercept and no feature asked for there is no
    /// coefficient, so no covariance, no standard error and no Wald test,
    /// however many rows were folded; the same sandwich over its feature
    /// has all three.
    #[test]
    fn no_coefficient_has_no_covariance() {
        let mut s = Sandwich::new(1, false, 2);
        for i in 0..20 {
            let x = f64::from(i % 5) - 2.0;
            s.update(&[x], 0.1 * x + 0.05 * f64::from(i % 3), 1.0, 1.0);
        }
        assert!(s.covariance(&[], false).is_none());
        assert!(s.covariance(&[], true).is_none());
        assert!(s.standard_errors(&[], true).is_none());
        assert!(s.wald(&[], true, &[0], &[1.0]).is_none());
        assert!(s.covariance(&[0], true).is_some(), "the control");
    }

    #[test]
    fn hc0_to_hc3_match_exact_row_time_definitions_and_order() {
        let mut s_all = Sandwich::with_types(2, true, 0, &RobustSeType::ALL, None);
        let mut s_legacy = Sandwich::new(2, true, 0);
        let mut s_hc3_only = Sandwich::with_types(2, true, 0, &[RobustSeType::Hc3], None);
        let mut st = 42u64;
        let lam = 0.98f64;
        let p = 3usize;
        // Brute-force reference maintaining B, M_hc0, M_hc2, M_hc3, W, Q.
        let mut b = vec![0.0; p * p];
        let mut m0 = vec![0.0; p * p];
        let mut m2 = vec![0.0; p * p];
        let mut m3 = vec![0.0; p * p];
        let (mut w_sum, mut q_sum) = (0.0f64, 0.0f64);
        let mut folded = 0u64;
        for row in 0..120 {
            let x = [lcg(&mut st) * 2.0, lcg(&mut st) * 1.5 + 0.5];
            let z = [1.0, x[0], x[1]];
            let e = (0.5 + x[0].abs()) * lcg(&mut st);
            let w = if row % 19 == 7 {
                0.0
            } else {
                0.5 + (lcg(&mut st) + 1.0)
            };
            s_all.update(&x, e, lam, w);
            s_legacy.update(&x, e, lam, w);
            s_hc3_only.update(&x, e, lam, w);

            // Reference update.
            for v in &mut b {
                *v *= lam;
            }
            for v in &mut m0 {
                *v *= lam * lam;
            }
            for v in &mut m2 {
                *v *= lam * lam;
            }
            for v in &mut m3 {
                *v *= lam * lam;
            }
            w_sum *= lam;
            q_sum *= lam * lam;
            if w > 0.0 {
                // Pre-fold leverage g = w z' B_prev^-1 z (after aging B by lam).
                if folded >= p as u64 {
                    let fac = crate::SpdFactor::of(&b, p).unwrap();
                    let inv_z = fac.solve_lower_cols(&z, p, 1);
                    let g: f64 = w * inv_z.iter().map(|v| v * v).sum::<f64>();
                    let denom = 1.0 + g;
                    let c0 = (w * e / denom).powi(2);
                    let c2 = (w * e).powi(2) / denom;
                    for i in 0..p {
                        for j in 0..p {
                            m0[i * p + j] += c0 * z[i] * z[j];
                            m2[i * p + j] += c2 * z[i] * z[j];
                        }
                    }
                }
                w_sum += w;
                q_sum += w * w;
                let c3 = (w * e).powi(2);
                for i in 0..p {
                    for j in 0..p {
                        b[i * p + j] += w * z[i] * z[j];
                        m3[i * p + j] += c3 * z[i] * z[j];
                    }
                }
                folded += 1;
            }
        }
        let idx = [0, 1];
        // Legacy HC0 and HC3-only must agree to the bit.
        let legacy = s_legacy.standard_errors(&idx, false).unwrap();
        let hc3_only = s_hc3_only
            .standard_errors_of(&idx, RobustSeType::Hc3)
            .unwrap();
        let hc3 = s_all.standard_errors_of(&idx, RobustSeType::Hc3).unwrap();
        for j in 0..p {
            assert_eq!(legacy[j].to_bits(), hc3_only[j].to_bits());
            assert_eq!(legacy[j].to_bits(), hc3[j].to_bits());
        }
        let hc0 = s_all.standard_errors_of(&idx, RobustSeType::Hc0).unwrap();
        let hc1 = s_all.standard_errors_of(&idx, RobustSeType::Hc1).unwrap();
        let hc2 = s_all.standard_errors_of(&idx, RobustSeType::Hc2).unwrap();
        // Verify against brute-force B^-1 M B^-1.
        let fac = crate::SpdFactor::of(&b, p).unwrap();
        let ref_se = |m: &[f64]| -> Vec<f64> {
            let mut binv = vec![0.0; p * p];
            for col in 0..p {
                let mut e_col = vec![0.0; p];
                e_col[col] = 1.0;
                let tmp = fac.solve_lower_cols(&e_col, p, 1);
                let sol = fac.solve_upper_transpose_cols(&tmp, p, 1);
                for row in 0..p {
                    binv[row * p + col] = sol[row];
                }
            }
            (0..p)
                .map(|j| {
                    let mut acc = 0.0;
                    for r in 0..p {
                        for c in 0..p {
                            acc += binv[j * p + r] * m[r * p + c] * binv[j * p + c];
                        }
                    }
                    acc.sqrt()
                })
                .collect()
        };
        let r0 = ref_se(&m0);
        let r2 = ref_se(&m2);
        let r3 = ref_se(&m3);
        let n_kish = w_sum * w_sum / q_sum;
        let hc1_mult = (n_kish / (n_kish - p as f64)).sqrt();
        for j in 0..p {
            assert!((hc0[j] - r0[j]).abs() < 1e-12 * (1.0 + r0[j]), "hc0[{j}]");
            assert!(
                (hc1[j] - r0[j] * hc1_mult).abs() < 1e-12 * (1.0 + r0[j]),
                "hc1[{j}]"
            );
            assert!((hc2[j] - r2[j]).abs() < 1e-12 * (1.0 + r2[j]), "hc2[{j}]");
            assert!((hc3[j] - r3[j]).abs() < 1e-12 * (1.0 + r3[j]), "hc3[{j}]");
            assert!(
                hc0[j] < hc1[j] && hc0[j] < hc2[j] && hc2[j] < hc3[j],
                "ordering failed at {j}: hc0={} hc1={} hc2={} hc3={}",
                hc0[j],
                hc1[j],
                hc2[j],
                hc3[j]
            );
        }

        // Active subset [0, 2] on a 3-feature sandwich must match a standalone 2-feature sandwich.
        let mut s_sub = Sandwich::with_types(3, true, 0, &RobustSeType::ALL, Some(vec![0, 2]));
        let mut s_2 = Sandwich::with_types(2, true, 0, &RobustSeType::ALL, None);
        let mut st2 = 91u64;
        for _ in 0..50 {
            let x3 = [lcg(&mut st2), lcg(&mut st2) * 5.0, lcg(&mut st2) - 0.3];
            let e = lcg(&mut st2);
            s_sub.update(&x3, e, 0.96, 1.0);
            s_2.update(&[x3[0], x3[2]], e, 0.96, 1.0);
        }
        for kind in RobustSeType::ALL {
            let v_sub = s_sub.standard_errors_of(&[0, 2], kind).unwrap();
            let v_2 = s_2.standard_errors_of(&[0, 1], kind).unwrap();
            assert!((v_sub[0] - v_2[0]).abs() < 1e-13);
            assert!((v_sub[1] - v_2[1]).abs() < 1e-13);
            assert!(v_sub[2].is_nan());
            assert!((v_sub[3] - v_2[2]).abs() < 1e-13);
        }
    }
}
