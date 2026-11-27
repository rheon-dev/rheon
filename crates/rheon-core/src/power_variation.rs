//! Power variation and kernel-weighted autocovariance sums of a stream of
//! returns, one pass: the building block that realized variance, bipower
//! and tripower variation, the realized kernel, two-scale realized variance
//! and the HAC long-run variance are made of.
//!
//! **The definitions**, for `n` returns `r_j` of one target and a scale
//! `K ≥ 1`, with `R_j = r_{j−K+1} + ... + r_j` the `K`-step return ending at
//! `j` (`R_j = r_j` at `K = 1`), each term of the model one of:
//!
//! ```text
//! multipower [p_1..p_m]:  MPV = (1/K) Σ_j Π_{i=1}^{m} |R_{j−(m−i)K}|^{p_i}
//! kernel (k, H):          KS  = (1/K) Σ_j [ R_j² + 2 Σ_{h=1}^{H} c_h R_j R_{j−hK} ]
//!                         c_h = k(h/(H + 1)),  or k((h − 1)/H) with flat_top
//! ```
//!
//! The powers are listed oldest first, `p_1` on the oldest return of the
//! run. A term is summed over every `j` where each return in it exists, so
//! at `K = 1` the multipower sum of `m` powers has `n − m + 1` terms and the
//! kernel's lag `h` has `n − h` products. Dividing by `K` averages the
//! term over the `K` offset grids of `K`-step returns, each `j` belonging
//! to one of them: at `K > 1`, `[2]` is Zhang, Mykland and Aït-Sahalia's
//! `[Y, Y]^(K)`, from which their two-scale estimator is composed. `k` is
//! the Bartlett, Parzen, Tukey-Hanning or rectangular kernel; the default
//! argument `h/(H + 1)` is Barndorff-Nielsen, Hansen, Lunde and Shephard's
//! (2011) and Newey and West's (Bartlett at lag `H` is `1 − h/(H + 1)`), and
//! `flat_top` is the flat-top realized kernel of Barndorff-Nielsen, Hansen,
//! Lunde and Shephard (2008), which gives lag 1 the full weight.
//!
//! **Options.** `normalize` multiplies a multipower term by `W^(Σp/2 − 1)
//! Π μ_{p_i}^-1`, `μ_p = E|Z|^p = 2^(p/2) Γ((p + 1)/2) / Γ(1/2)` for a
//! standard normal `Z` and `W` the returns' weight: the multipower
//! variation of Barndorff-Nielsen and Shephard, whose factor is the count
//! `n` at unit weights without a decay, so `[2]` is realized variance,
//! `[1, 1]` bipower variation, `[2/3, 2/3, 2/3]` tripower variation and
//! `[4/3, 4/3, 4/3]` tripower quarticity. Under unequal or decayed weights
//! the factor is the weight, the period the sums are of: at a constant
//! volatility the weighted sum of `Σp`-th powers is about `W σ^Σp
//! δ^(Σp/2) Π μ`, so the quarticity over the bipower variation squared is
//! 1 there only with `W` as the factor, as the bipower test needs, while
//! the test's `√n` in front is Kish's count, the variance of `1 − BV/RV`
//! being `ϑ Σw² / (Σw)²`. `realized` reads the same. `center` takes a kernel term about the mean `m̄` of the
//! `R_j` it is of: each product `(R_j − m̄)(R_{j−hK} − m̄)`, the sum the
//! HAC long-run variance divides by `n`. Neither carries a small-sample
//! factor such as `n/(n − h)`: a factor of `n` has no meaning on decayed
//! weights.
//!
//! **The online form.** A row is a clock step, one return per target (a
//! target absent on a row has no return there: its terms reach across the
//! row to its last return) and a weight `w ≥ 0`. Per target, with `f =
//! e^(−ln 2 · d / half_life)` the row's decay, every sum decays and then
//! takes the row's terms, each at the weight of the latest row in it:
//!
//! ```text
//! W ← f W + w          V ← f² V + w²       the returns' weight and squared weights
//! S ← f S + w · term   for each term ending at the row
//! ```
//!
//! and a row reads `S / K`, scaled where `normalize` says. A centered kernel
//! term keeps, beside its products `S`, the sums `A` of `c R_j`, `B` of `c
//! R_{j−hK}` and `C` of `c` over the same products (`c = 1` at lag 0 and
//! `2 c_h` beyond) and the weight `M₀` and sum `M₁` of the `R_j`, and reads
//! `S − m̄ (A + B) + m̄² C`, `m̄ = M₁ / M₀`: the sum about the mean as it
//! stands, exactly, not about each row's earlier mean. `n` is the count of
//! returns: Kish's effective count `W² / V` of the returns' weights,
//! decayed or in the window, which is the count at equal weights; each
//! target reports it beside its terms.
//!
//! **Out of sample** (hard rule 2): a row reports from the state before its
//! own return and before its own decay. `weight_sum` is every row's weight
//! and each target's `W` that of its returns, both before the row (hard
//! rule 8); a target reports `NaN` while its `W` is under `min_weight`.
//!
//! **A row of weight 0** learns nothing (hard rule 9): no sum takes a term
//! and no return is remembered, so the next term reaches across it.
//!
//! **A window** (`window_rows = W`, or `window_clock = D`) in place of a
//! decay: the terms of the returns of the last `W` rows of positive weight,
//! or of the rows of the last `D` clock units, alone. A term is in the
//! window while every return in it is: each is credited to its oldest
//! return and leaves with it. The sums are compensated (TwoSum), and set to
//! exactly 0 when a target's window empties. `D` is read on the rows'
//! exact stamps under `closed` (`Right` drops a row exactly `D` old, `Both`
//! keeps it). A window forgets by dropping, so it takes no decay.
//!
//! **Past the largest double.** A term of returns near the input bound can
//! pass the largest double: `|r|⁴` of a return of `1e100` is `1e400`, and
//! the square of a `K`-step return of them times a weight passes it at a
//! `K` above 13. Such a term, and a sum it takes past the largest double,
//! is held at it, the largest a double holds: a bound on the term, where
//! `inf` would never decay (`f · inf` is `inf`) and a window's next
//! take-back of it, `inf − inf`, would be NaN. A product of powers that
//! passes the largest double on the way, or reads `0 · inf`, is formed
//! again from each factor's fraction and power of two, so a term that fits
//! is kept and a factor of 0 makes it 0. Each output reads its own sums,
//! so one held at the bound moves no other. Below the largest double
//! nothing moves, to the bit.
//!
//! A window takes a term back from a sum held at the bound by summing the
//! credits it still holds again, and does the same where a term leaving is
//! more than `1/ε` times what it leaves (a return near the bound leaving a
//! window of ordinary ones): the compensated pair keeps a sum to about
//! `ε²` of the largest term it has held, so what it leaves is otherwise
//! the rounding of that term, and it would never leave.
//!
//! **The state.** Per target, the sums (16 bytes each: `W`, `V`, one a
//! multipower or plain kernel term, six a centered kernel term), the last
//! `K_max − 1` returns and, per scale, the last `K`-step returns its lags
//! reach (8 bytes each); bounded by the settings, whatever the stream's
//! length. A window adds `32` bytes, and a row it holds `16` (a stamp more
//! under a clock window) and per target `16` and `16` a sum. The fixed part
//! and a full `window_rows` window are checked against `max_state_bytes`
//! when the model is built; a clock window as the rows go in
//! ([`OnlineModel::state_over_cap`]).

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

use crate::clock_span::Place;
use crate::model::{ModelState, OnlineModel, State, StateError, Step, check_schema};
use crate::{Decay, Stamp, WindowClosed};

mod series;

use series::{Plan, Series};

/// A kernel of the kernel-weighted autocovariance sum, `k(0) = 1` and
/// `k(x) = 0` beyond 1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PvKernel {
    /// `1 − x`.
    Bartlett,
    /// `1 − 6x² + 6x³` to `1/2`, `2 (1 − x)³` to 1.
    Parzen,
    /// `(1 + cos πx) / 2`.
    TukeyHanning,
    /// `1` to 1, the truncated kernel.
    Rectangular,
}

impl PvKernel {
    /// `k(x)` for `x ≥ 0`.
    pub fn weight(self, x: f64) -> f64 {
        if !(0.0..=1.0).contains(&x) {
            return 0.0;
        }
        match self {
            PvKernel::Bartlett => 1.0 - x,
            PvKernel::Parzen => crate::parzen(x),
            PvKernel::TukeyHanning => 0.5 * (1.0 + libm::cos(std::f64::consts::PI * x)),
            PvKernel::Rectangular => 1.0,
        }
    }
}

/// What a term sums (the module docs).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PvTermKind {
    /// `Π |R|^{p_i}` over `m` consecutive `K`-step returns, oldest first.
    Multipower {
        powers: Vec<f64>,
        /// Multiply by `n^(Σp/2 − 1) Π μ_{p_i}^-1`.
        #[serde(default)]
        normalize: bool,
    },
    /// `R_j² + 2 Σ_h c_h R_j R_{j−hK}`.
    Kernel {
        kernel: PvKernel,
        /// `H`, the last lag with weight.
        bandwidth: usize,
        /// `c_h = k((h − 1)/H)` in place of `k(h/(H + 1))`.
        #[serde(default)]
        flat_top: bool,
        /// About the mean of the `R_j`.
        #[serde(default)]
        center: bool,
    },
}

fn one() -> usize {
    1
}

/// One term a target reports.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PvTerm {
    pub kind: PvTermKind,
    /// `K`: the term of the `K`-step returns, averaged over the `K` offsets.
    #[serde(default = "one")]
    pub scale: usize,
}

impl PvTerm {
    /// The returns a term ending at a return reaches back over, the return
    /// itself not counted: `mK − 1` for `m` powers, `(H + 1)K − 1` for a
    /// kernel of bandwidth `H`. `None` past [`crate::MAX_LAG`].
    pub fn reach(&self) -> Option<usize> {
        let lags = match &self.kind {
            PvTermKind::Multipower { powers, .. } => powers.len().max(1),
            PvTermKind::Kernel { bandwidth, .. } => bandwidth.checked_add(1)?,
        };
        let r = lags.checked_mul(self.scale)?.checked_sub(1)?;
        (r <= crate::MAX_LAG).then_some(r)
    }
}

/// The outputs a target reports: one a term, then `n`.
pub fn outputs_per_target(terms: &[PvTerm]) -> usize {
    terms.len() + 1
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PowerVariationCfg {
    /// The return series, one per target.
    pub n_targets: usize,
    /// The terms each target reports, in order.
    pub terms: Vec<PvTerm>,
    /// The forgetting, per clock unit; `Halflife(inf)` keeps every return.
    pub decay: Decay,
    /// The weight of a target's returns before it reports.
    pub min_weight: f64,
    /// The terms of the last this many rows of positive weight alone, in
    /// place of a decay.
    #[serde(default)]
    pub window_rows: Option<usize>,
    /// The terms of the rows of the last this many clock units alone, in
    /// place of a decay.
    #[serde(default)]
    pub window_clock: Option<f64>,
    /// Which edge of a clock window holds the row exactly `window_clock`
    /// old: `Right`, the default, drops it, `Both` keeps it.
    #[serde(default)]
    pub closed: WindowClosed,
    /// The bytes a group's state may hold, a window's rows included; the
    /// default is [`crate::clock_span::DEFAULT_STATE_BYTES`].
    #[serde(default)]
    pub max_state_bytes: Option<u64>,
}

const KIND: &str = "power_variation";

impl PowerVariationCfg {
    pub fn validate(&self) -> Result<(), String> {
        let mut no = crate::Refusals::new();
        if self.n_targets == 0 {
            no.refuse(
                "power_variation: the model needs at least one target, a series of returns, but \
                 it has none. Name at least one column of returns in targets.",
            );
        }
        if self.terms.is_empty() {
            no.refuse(
                "power_variation: the model needs at least one term, but it has none. Give a \
                 multipower term (its powers) or a kernel term (its kernel and bandwidth).",
            );
        }
        let mut shaped = true;
        for (i, t) in self.terms.iter().enumerate() {
            let at = i + 1;
            if t.scale == 0 {
                shaped = false;
                no.refuse(format!(
                    "power_variation: term {at}'s scale must be at least 1, but it is 0. Give \
                     the number of returns each of its returns sums, 1 for the returns as they \
                     are."
                ));
            }
            match &t.kind {
                PvTermKind::Multipower { powers, normalize } => {
                    if powers.is_empty() {
                        shaped = false;
                        no.refuse(format!(
                            "power_variation: term {at} has no powers. Give one power for each \
                             return of its product, oldest first, such as [1, 1]."
                        ));
                    }
                    if let Some(p) = powers.iter().find(|p| !(p.is_finite() && **p > 0.0)) {
                        no.refuse(format!(
                            "power_variation: term {at}'s powers must be finite and above 0, \
                             but one is {p}. Leave out a return of power 0: it multiplies by 1."
                        ));
                    }
                    if *normalize && t.scale > 1 {
                        no.refuse(format!(
                            "power_variation: term {at} is normalized at a scale of {}, but \
                             normalize is defined on the returns as they are. Set its scale to \
                             1, or leave normalize out.",
                            t.scale
                        ));
                    }
                }
                PvTermKind::Kernel {
                    bandwidth,
                    flat_top,
                    ..
                } => {
                    if *flat_top && *bandwidth == 0 {
                        shaped = false;
                        no.refuse(format!(
                            "power_variation: term {at} is flat-top with a bandwidth of 0, but \
                             the flat-top weights k((h - 1)/H) need H of 1 or more. Give a \
                             bandwidth of at least 1."
                        ));
                    }
                }
            }
            if t.scale > 0 && t.reach().is_none() {
                shaped = false;
                no.refuse(format!(
                    "power_variation: term {at} reaches back more than {} returns (its powers or \
                     bandwidth times its scale). Lower its scale, powers or bandwidth.",
                    crate::MAX_LAG
                ));
            }
        }
        if let Err(e) = self.decay.check() {
            no.refuse(format!("power_variation: {e}"));
        }
        no.take(crate::model::check_min_weight(KIND, self.min_weight));
        if self.window_rows == Some(0) {
            no.refuse(
                "power_variation: window_rows must be at least 1, but it is 0. Give the number \
                 of rows the terms are summed over, or leave it out for no window.",
            );
        }
        if let Some(d) = self.window_clock {
            if !(d.is_finite() && d > 0.0) {
                no.refuse(format!(
                    "power_variation: the clock window must be a finite number of clock units \
                     above 0, but it is {d}. Give the span of clock the terms are summed over, \
                     or leave it out for no window."
                ));
            }
            if self.window_rows.is_some() {
                no.refuse(
                    "power_variation: a window of clock units and window_rows are two windows, \
                     and the terms hold one. Leave one of them out.",
                );
            }
        } else if self.closed != WindowClosed::Right {
            no.refuse(
                "power_variation: closed = \"both\" says which edge of a clock window keeps the \
                 row exactly its span old, but there is no clock window. Give the window's \
                 span, or leave closed out.",
            );
        }
        if self.windowed() && self.decay.check().is_ok() && self.decay.factor(1.0) != 1.0 {
            no.refuse(
                "power_variation: a window forgets by dropping the oldest rows, so it takes no \
                 decay, but half_life (or lam) decays too. Set half_life to inf, or leave out \
                 the window.",
            );
        }
        match crate::clock_span::check_max_state_bytes(KIND, self.max_state_bytes) {
            Err(e) => no.refuse(e),
            Ok(()) if self.n_targets > 0 && shaped && !self.terms.is_empty() => {
                let plan = Plan::of(&self.terms);
                let (bytes, cap) = (self.fixed_bytes(&plan), self.cap());
                if bytes > cap {
                    no.refuse(self.past_the_cap(&plan, bytes, cap));
                }
            }
            Ok(()) => {}
        }
        no.finish()
    }

    /// The refusal of settings whose fixed state passes the cap, naming the
    /// largest window that fits where a window is what passes it.
    fn past_the_cap(&self, plan: &Plan, bytes: u64, cap: u64) -> String {
        let base = PowerVariation::state_bytes(plan, self.n_targets, Some(0), false);
        let row = PowerVariation::window_row_bytes(plan, self.n_targets, false);
        match self.window_rows {
            Some(w) if base <= cap => format!(
                "power_variation: its state would hold {bytes} bytes a group, past \
                 max_state_bytes of {cap}: a window of {w} rows keeps {row} bytes a row for {}. \
                 The largest window_rows that fits is {}. Lower window_rows, or raise \
                 max_state_bytes.",
                crate::model::count(self.n_targets, "target"),
                (cap - base) / row,
            ),
            _ => format!(
                "power_variation: its state would hold {bytes} bytes a group, past \
                 max_state_bytes of {cap}: {} bytes for each of {}, most of it the returns its \
                 terms reach back over. Give fewer targets or terms a spec, lower a term's \
                 scale or bandwidth, or raise max_state_bytes.",
                bytes / (self.n_targets as u64).max(1),
                crate::model::count(self.n_targets, "target"),
            ),
        }
    }

    fn windowed(&self) -> bool {
        self.window_rows.is_some() || self.window_clock.is_some()
    }

    /// The bytes the settings fix: the state with a full `window_rows`
    /// window, or with an empty clock window.
    fn fixed_bytes(&self, plan: &Plan) -> u64 {
        PowerVariation::state_bytes(
            plan,
            self.n_targets,
            self.window_rows.or(self.window_clock.map(|_| 0)),
            self.window_clock.is_some(),
        )
    }

    /// The bytes a group's state may hold.
    fn cap(&self) -> u64 {
        self.max_state_bytes
            .unwrap_or(crate::clock_span::DEFAULT_STATE_BYTES)
    }
}

/// A row a window holds: its place in the stream, its weight and, in a
/// window of clock units, its stamp.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Held {
    seq: u64,
    w: f64,
    #[serde(default)]
    stamp: Option<Stamp>,
}

/// The rows a window holds, oldest first, the weight inside it, the next
/// row's sequence and, in a window of clock units, the clock.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Window {
    rows: VecDeque<Held>,
    w_hi: f64,
    w_lo: f64,
    next_seq: u64,
    #[serde(default)]
    place: Option<Place>,
}

impl Window {
    fn new(clocked: bool) -> Self {
        Self {
            rows: VecDeque::new(),
            w_hi: 0.0,
            w_lo: 0.0,
            next_seq: 0,
            place: clocked.then(Place::default),
        }
    }

    fn weight(&self) -> f64 {
        self.w_hi + self.w_lo
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PowerVariation {
    cfg: PowerVariationCfg,
    series: Vec<Series>,
    /// Every row's weight, decayed: the emitted `weight_sum` under a decay.
    w_sum: f64,
    window: Option<Window>,
    /// The bytes a clock window's next row would have taken the state to,
    /// past the cap: from then on no row is learned.
    #[serde(default)]
    over: Option<u64>,
    /// The half-of-the-cap warning has been raised (once a group).
    #[serde(default)]
    warned: bool,
    /// The warning raised and not yet taken. Never state.
    #[serde(skip)]
    warning: Option<u64>,
    /// What the settings fix: the scales, the kernels' weights, the
    /// normalizing constants. Never state; made from `cfg`.
    #[serde(skip)]
    plan: Plan,
}

impl PowerVariation {
    pub fn new(cfg: PowerVariationCfg) -> Result<Self, String> {
        cfg.validate()?;
        let plan = Plan::of(&cfg.terms);
        Ok(Self {
            series: vec![Series::new(&plan); cfg.n_targets],
            w_sum: 0.0,
            window: cfg
                .windowed()
                .then(|| Window::new(cfg.window_clock.is_some())),
            over: None,
            warned: false,
            warning: None,
            plan,
            cfg,
        })
    }

    pub fn cfg(&self) -> &PowerVariationCfg {
        &self.cfg
    }

    /// The bytes a window keeps a row it holds: its sequence and weight, a
    /// stamp under a clock window, and for each target its sequence, weight
    /// and its share of every sum.
    fn window_row_bytes(plan: &Plan, n_targets: usize, clocked: bool) -> u64 {
        let stamp = if clocked {
            std::mem::size_of::<Stamp>() as u64
        } else {
            0
        };
        16 + stamp + plan.entry_bytes() * n_targets as u64
    }

    /// The bytes of the numbers the state holds: `8`, each target's
    /// [`Plan::series_bytes`] and, with a window holding `rows` rows, its
    /// fixed `32` and [`Self::window_row_bytes`] a row.
    fn state_bytes(plan: &Plan, n_targets: usize, rows: Option<usize>, clocked: bool) -> u64 {
        8 + plan.series_bytes() * n_targets as u64
            + rows.map_or(0, |r| {
                32 + r as u64 * Self::window_row_bytes(plan, n_targets, clocked)
            })
    }

    /// The bytes the state holds now, a window's rows as they are.
    pub fn bytes(&self) -> u64 {
        self.bytes_with(0)
    }

    fn bytes_with(&self, extra: usize) -> u64 {
        Self::state_bytes(
            &self.plan,
            self.cfg.n_targets,
            self.window.as_ref().map(|w| w.rows.len() + extra),
            self.cfg.window_clock.is_some(),
        )
    }

    /// The weight the terms are of (hard rule 8): every row's, decayed, or
    /// the weight inside the window.
    pub fn n_eff(&self) -> f64 {
        self.window.as_ref().map_or(self.w_sum, Window::weight)
    }

    /// The stamp of the newest row a clock window holds, `None` before one
    /// and without a clock window.
    pub fn last_stamp(&self) -> Option<Stamp> {
        self.window.as_ref()?.rows.back()?.stamp
    }

    /// The rows a window holds; 0 without one.
    pub fn window_len(&self) -> usize {
        self.window.as_ref().map_or(0, |w| w.rows.len())
    }

    fn read_into(&self, pred: &mut Vec<f64>) {
        pred.clear();
        for s in &self.series {
            s.read(&self.plan, self.cfg.min_weight, pred);
        }
    }

    fn held_cap(&mut self) -> crate::clock_span::HeldCap<'_> {
        crate::clock_span::HeldCap {
            over: &mut self.over,
            warned: &mut self.warned,
            warning: &mut self.warning,
        }
    }

    /// The oldest row out of the window, and each target's terms credited
    /// to its return there.
    fn drop_oldest(&mut self) {
        let Some(w) = self.window.as_mut() else {
            return;
        };
        let Some(old) = w.rows.pop_front() else {
            return;
        };
        series::add(&mut w.w_hi, &mut w.w_lo, -old.w);
        if w.rows.is_empty() {
            (w.w_hi, w.w_lo) = (0.0, 0.0);
        }
        for s in &mut self.series {
            s.drop_front(old.seq);
        }
    }

    /// The rows of a clock window that have left at `now`, oldest first.
    fn trim(&mut self, now: Stamp) {
        let (Some(span), closed) = (self.cfg.window_clock, self.cfg.closed) else {
            return;
        };
        while self
            .window
            .as_ref()
            .and_then(|w| w.rows.front())
            .and_then(|h| h.stamp)
            .is_some_and(|s| closed.has_left(now, s, span))
        {
            self.drop_oldest();
        }
    }
}

impl OnlineModel for PowerVariation {
    fn step_into(
        &mut self,
        _x: &[f64],
        y: &[Option<f64>],
        d_clock: f64,
        weight: f64,
        out: &mut Step,
    ) {
        if let Some(refused) = crate::model::refused_step(self, &[], y, d_clock, weight) {
            *out = refused;
            return;
        }
        // Read before the row's return and before its decay, or before the
        // rows that leave a window at this row have left (hard rules 2, 8).
        self.read_into(&mut out.pred);
        out.n_eff = self.n_eff();
        out.extra = None;
        if self.over.is_some() {
            if let Some(p) = self.window.as_mut().and_then(|w| w.place.as_mut()) {
                p.discard_next();
            }
            return;
        }
        if weight > 0.0
            && let Some(now) = self
                .window
                .as_ref()
                .and_then(|w| w.place.as_ref())
                .map(|p| p.peek(d_clock))
        {
            let dropped = match (self.cfg.window_clock, &self.window) {
                (Some(span), Some(w)) => w
                    .rows
                    .iter()
                    .take_while(|h| {
                        h.stamp
                            .is_some_and(|s| self.cfg.closed.has_left(now, s, span))
                    })
                    .count(),
                _ => 0,
            };
            let rows = self.window.as_ref().map(|w| w.rows.len() - dropped + 1);
            let bytes = Self::state_bytes(&self.plan, self.cfg.n_targets, rows, true);
            let cap = self.cfg.cap();
            if !self.held_cap().admit(bytes, cap) {
                if let Some(p) = self.window.as_mut().and_then(|w| w.place.as_mut()) {
                    p.discard_next();
                }
                return;
            }
        }
        let f = self.cfg.decay.factor(d_clock);
        self.w_sum = f * self.w_sum + weight;
        for s in &mut self.series {
            s.decay(f);
        }
        let now = self
            .window
            .as_mut()
            .and_then(|w| w.place.as_mut())
            .map(|p| p.step(d_clock));
        if let Some(now) = now {
            self.trim(now);
        }
        // A row of weight 0 learns nothing (hard rule 9).
        if weight <= 0.0 {
            return;
        }
        let seq = match self.window.as_mut() {
            Some(w) => {
                let seq = w.next_seq;
                w.next_seq += 1;
                w.rows.push_back(Held {
                    seq,
                    w: weight,
                    stamp: now,
                });
                series::add(&mut w.w_hi, &mut w.w_lo, weight);
                Some(seq)
            }
            None => None,
        };
        for (j, s) in self.series.iter_mut().enumerate() {
            if let Some(r) = y.get(j).copied().flatten() {
                s.learn(&self.plan, r, weight, seq);
            }
        }
        if let Some(cap) = self.cfg.window_rows {
            while self.window_len() > cap {
                self.drop_oldest();
            }
        }
        if now.is_some() {
            let (bytes, cap) = (self.bytes_with(0), self.cfg.cap());
            self.held_cap().note(bytes, cap);
        }
    }

    fn predict(&self, x: &[f64], d_clock: f64) -> Step {
        self.predict_with(x, &[], d_clock)
    }

    /// Every output is of the returns before the row; the row's own are not
    /// read.
    fn predict_with(&self, _x: &[f64], y: &[Option<f64>], d_clock: f64) -> Step {
        if let Some(refused) = crate::model::refused_predict_with(self, &[], y, d_clock) {
            return refused;
        }
        let mut pred = Vec::with_capacity(self.n_outputs());
        self.read_into(&mut pred);
        Step {
            pred,
            n_eff: self.n_eff(),
            extra: None,
        }
    }

    fn stamp_next(&mut self, stamp: Stamp) {
        if let Some(p) = self.window.as_mut().and_then(|w| w.place.as_mut()) {
            p.hand(stamp);
        }
    }

    fn take_state_warning(&mut self) -> Option<(u64, u64)> {
        self.warning.take().map(|bytes| (bytes, self.cfg.cap()))
    }

    fn state_over_cap(&self) -> Option<(u64, u64)> {
        self.over.map(|bytes| (bytes, self.cfg.cap()))
    }

    fn target_n_eff_into(&self, out: &mut Vec<f64>) -> bool {
        out.clear();
        out.extend(self.series.iter().map(Series::weight));
        true
    }

    fn state(&self) -> State {
        State::new(ModelState::PowerVariation(Box::new(self.clone())))
    }

    fn restore(s: &State) -> Result<Self, StateError> {
        check_schema(s)?;
        match &s.model {
            ModelState::PowerVariation(m) => {
                let mut m = (**m).clone();
                crate::model::check_cfg(KIND, m.cfg.validate())?;
                m.plan = Plan::of(&m.cfg.terms);
                let rows = m.window.as_ref().map(|w| &w.rows);
                let shaped = m.series.len() == m.cfg.n_targets
                    && m.series.iter().all(|s| s.has_shape(&m.plan, rows))
                    && match (&m.window, m.cfg.windowed()) {
                        (None, false) => true,
                        (Some(w), true) => {
                            w.place.is_some() == m.cfg.window_clock.is_some()
                                && w.rows.iter().all(|h| {
                                    h.stamp.is_some() == m.cfg.window_clock.is_some()
                                        && h.seq < w.next_seq
                                })
                                && m.cfg.window_rows.is_none_or(|c| w.rows.len() <= c)
                        }
                        _ => false,
                    };
                if !shaped {
                    return Err(crate::model::damaged(
                        KIND,
                        "its running sums have the wrong shape",
                    ));
                }
                Ok(m)
            }
            other => Err(StateError::WrongModel {
                expected: KIND,
                found: other.kind(),
            }),
        }
    }

    fn n_targets(&self) -> usize {
        self.cfg.n_targets
    }

    /// The series are the targets; the model reads no features.
    fn n_features(&self) -> usize {
        0
    }

    fn n_outputs(&self) -> usize {
        outputs_per_target(&self.cfg.terms) * self.cfg.n_targets
    }
}

#[cfg(test)]
mod paper;
#[cfg(test)]
mod tests;
