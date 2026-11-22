//! Realized-volatility estimators and the jump tests built on them, for a
//! stream of returns, one pass.
//!
//! **The definitions** (Barndorff-Nielsen and Shephard 2004, 2006; Lee and
//! Mykland 2008), for `n` returns `r_j` of one stretch of the clock, `a_j =
//! |r_j|`, and `μ_p = E|Z|^p = 2^(p/2) Γ((p + 1)/2) / Γ(1/2)` for a standard
//! normal `Z`:
//!
//! ```text
//! RV = Σ_j r_j²                                        realized variance
//! BV = μ_1^-2 Σ_{j≥2} a_j a_{j−1}                      bipower variation, μ_1^-2 = π/2
//! TV = μ_{2/3}^-3 Σ_{j≥3} (a_j a_{j−1} a_{j−2})^(2/3)  tripower variation
//! TQ = n μ_{4/3}^-3 Σ_{j≥3} (a_j a_{j−1} a_{j−2})^(4/3) tripower quarticity
//! Z  = √n (1 − BV/RV) / √(ϑ max(1, TQ/BV²))           the adjusted ratio test, ϑ = π²/4 + π − 5
//! L_i = r_i / σ_i,  σ_i² = mean of a_j a_{j−1} over the products before row i
//! ```
//!
//! `RV`, `BV` and `TV` estimate the integrated variance, `BV` and `TV` with
//! the jumps left out; `TQ` the integrated quarticity. `Z` is N(0, 1) without
//! jumps, and large and positive with them. `L_i` is Lee and Mykland's
//! statistic of the row's own return against the volatility of the returns
//! before it: under no jump `c L_i` is N(0, 1), `c = √(2/π)`. None carries a
//! small-sample factor such as `n/(n − 1)`: the papers' estimators have none.
//! A window of `W` returns holds the `W − 1` products of its consecutive
//! returns, and the `W − 2` triples, as the papers' sums over `W` returns do.

//!
//! **The online form.** A row is a clock step, one return per target (a
//! target absent on a row has no return there) and a weight `w ≥ 0`. Per
//! target, with `f = e^(−ln 2 · d / half_life)` the row's decay, every sum
//! decays and then takes the row's terms, each term weighted by the weight
//! of the latest row in it:
//!
//! ```text
//! n ← f n + w                       the return weight, the papers' n
//! s_rv ← f s_rv + w r_j²
//! s_bv ← f s_bv + w a_j a_{j−1}     p ← f p + w      (from the second return on)
//! s_tv ← f s_tv + w (c_j c_{j−1} c_{j−2})²            c = a^(1/3)
//! s_tq ← f s_tq + w (c_j c_{j−1} c_{j−2})⁴            (from the third return on)
//! ```
//!
//! and a row reads `RV = s_rv`, `BV = (π/2) s_bv`, `TV = μ_{2/3}^-3 s_tv`,
//! `TQ = n μ_{4/3}^-3 s_tq`, `Z` with `n` for the papers' count, and `σ_i² =
//! s_bv / p`, the weighted mean of the products before the row. With no
//! decay (`half_life = inf`) and unit weights these are the papers' values
//! on the returns so far, `n − 1` products and `n − 2` triples. A weight
//! counts returns: the estimators are sums, so they scale with it, and `n`
//! is a count of returns at unit weight.
//!
//! **Out of sample** (hard rule 2): a row reports from the state before its
//! own return and before its own decay, so the variations are those of the
//! returns before it, and `L_i` is the row's return against their
//! volatility. `weight_sum` is the weight of every row, and each target's
//! own `n` the weight of its returns, both before the row's update and
//! before its decay (hard rule 8); a target reports `NaN` while its `n` is
//! under `min_weight`, the test where `RV` or `BV` is 0, and `L_i` where
//! `σ_i` is 0 or the row has no return for it.
//!
//! **A row of weight 0** learns nothing (hard rule 9): no sum takes a term
//! and the last returns are kept, so the next product reaches across it to
//! the last return learned. The stream hands no such row to the model, so
//! it counts in no clock; handed one directly, the model ages its sums over
//! the row's step, as every model does.
//!
//! **The sums are compensated**: each is a pair (`crate::comp`), its high
//! part and what that leaves out, so a sum over a day of 23,400 returns
//! keeps the rounding of its terms rather than a random walk of 23,400
//! roundings (40 to 130 units in the last place, measured, against 0 to 5).
//! The cube roots are the `libm` crate's, the same bits on every platform.
//!
//! **The state** is `16` numbers a target (six sums and their low parts,
//! the last return's size, the last two cube roots and how many returns
//! are remembered) and the shared weight: `8 + 128 m` bytes for `m`
//! targets, whatever the stream's length or the half-life.

use serde::{Deserialize, Serialize};

use crate::model::{ModelState, OnlineModel, State, StateError, Step, check_schema};
use crate::{Decay, comp};

/// `μ_1^-2 = π/2`, the scale of bipower variation.
pub const BV_SCALE: f64 = std::f64::consts::FRAC_PI_2;

/// `μ_{2/3}^-3`, the scale of tripower variation, `μ_{2/3} = 2^(1/3) Γ(5/6) /
/// Γ(1/2)`; correctly rounded from 40 digits.
pub const TV_SCALE: f64 = 1.935_792_404_880_348_3;

/// `μ_{4/3}^-3`, the scale of tripower quarticity, `μ_{4/3} = 2^(2/3) Γ(7/6) /
/// Γ(1/2)`; correctly rounded from 40 digits.
pub const TQ_SCALE: f64 = 1.743_472_074_531_984_5;

/// `ϑ = π²/4 + π − 5`, the asymptotic variance of the ratio statistic
/// `√n (1 − BV/RV)` per unit of `TQ/BV²`; correctly rounded from 40 digits.
pub const BNS_THETA: f64 = 0.608_993_753_862_133;

/// The outputs a target reports, in order: `rv`, `bv`, `tv`, `tq`, `bns`,
/// `lm`.
pub const OUTPUTS_PER_TARGET: usize = 6;

/// The sums a target keeps, as indices into its pairs.
const NW: usize = 0;
const RV: usize = 1;
const BV: usize = 2;
const TV: usize = 3;
const TQ: usize = 4;
const PW: usize = 5;
const SUMS: usize = 6;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RealizedCfg {
    /// The return series, one per target.
    pub n_targets: usize,
    /// The forgetting, per clock unit; `Halflife(inf)` keeps every return.
    pub decay: Decay,
    /// The weight of a target's returns before it reports.
    pub min_weight: f64,
    /// The bytes a group's state may hold; the default is
    /// [`crate::clock_span::DEFAULT_STATE_BYTES`].
    #[serde(default)]
    pub max_state_bytes: Option<u64>,
}

impl RealizedCfg {
    pub fn validate(&self) -> Result<(), String> {
        let mut no = crate::Refusals::new();
        if self.n_targets == 0 {
            no.refuse(
                "realized: the model needs at least one target, a series of returns, but it \
                 has none. Name at least one column of returns in targets.",
            );
        }
        if let Err(e) = self.decay.check() {
            no.refuse(format!("realized: {e}"));
        }
        no.take(crate::model::check_min_weight("realized", self.min_weight));
        match crate::clock_span::check_max_state_bytes("realized", self.max_state_bytes) {
            Err(e) => no.refuse(e),
            Ok(()) => {
                let (bytes, cap) = (Realized::state_bytes(self.n_targets), self.cap());
                if self.n_targets > 0 && bytes > cap {
                    no.refuse(format!(
                        "realized: its state would hold {bytes} bytes a group, past \
                         max_state_bytes of {cap}: {} bytes for each of {}. Give fewer \
                         targets a spec, or raise max_state_bytes.",
                        Series::BYTES,
                        crate::model::count(self.n_targets, "target"),
                    ));
                }
            }
        }
        no.finish()
    }

    /// The bytes a group's state may hold.
    fn cap(&self) -> u64 {
        self.max_state_bytes
            .unwrap_or(crate::clock_span::DEFAULT_STATE_BYTES)
    }
}

/// One return series: its compensated sums and the last returns it has
/// learned, which the next products are formed with.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Series {
    /// The sums' high parts, `[n, rv, bv, tv, tq, p]`.
    hi: [f64; SUMS],
    /// What each high part leaves out ([`comp::add`]).
    lo: [f64; SUMS],
    /// `|r|` of the last return learned.
    a1: f64,
    /// `|r|^(1/3)` of the last two returns learned, the latest first.
    c1: f64,
    c2: f64,
    /// How many returns are remembered: 0, 1, or 2 for two or more.
    seen: u8,
}

impl Series {
    /// The numbers a series holds, in bytes: 16 of 8 bytes.
    const BYTES: u64 = 8 * 16;

    fn new() -> Self {
        Self {
            hi: [0.0; SUMS],
            lo: [0.0; SUMS],
            a1: 0.0,
            c1: 0.0,
            c2: 0.0,
            seen: 0,
        }
    }

    fn sum(&self, k: usize) -> f64 {
        self.hi[k] + self.lo[k]
    }

    /// Every sum aged by the factor `f`.
    fn decay(&mut self, f: f64) {
        if f != 1.0 {
            for v in self.hi.iter_mut().chain(self.lo.iter_mut()) {
                *v *= f;
            }
        }
    }

    fn add(&mut self, k: usize, x: f64) {
        comp::add(&mut self.hi[k], &mut self.lo[k], x);
    }

    /// The return `r` learned at the weight `w > 0`.
    fn learn(&mut self, r: f64, w: f64) {
        let a = r.abs();
        let c = libm::cbrt(a);
        self.add(NW, w);
        self.add(RV, w * (r * r));
        if self.seen >= 1 {
            self.add(BV, w * (a * self.a1));
            self.add(PW, w);
        }
        if self.seen >= 2 {
            let t = c * self.c1 * self.c2;
            let t2 = t * t;
            self.add(TV, w * t2);
            self.add(TQ, w * (t2 * t2));
        }
        self.c2 = self.c1;
        self.c1 = c;
        self.a1 = a;
        self.seen = (self.seen + 1).min(2);
    }

    /// The series' six outputs into `out`, from its state as it stands,
    /// for a row whose return is `r` (none where the row has none).
    fn read(&self, r: Option<f64>, min_weight: f64, out: &mut Vec<f64>) {
        let n = self.sum(NW);
        if n < min_weight {
            out.extend([f64::NAN; OUTPUTS_PER_TARGET]);
            return;
        }
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
        let p = self.sum(PW);
        let lm = match r {
            Some(r) if p > 0.0 && s_bv > 0.0 => r / (s_bv / p).sqrt(),
            _ => f64::NAN,
        };
        out.extend([rv, bv, tv, tq, z, lm]);
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Realized {
    cfg: RealizedCfg,
    series: Vec<Series>,
    /// Every row's weight, decayed: the emitted `weight_sum`.
    w_sum: f64,
}

impl Realized {
    pub fn new(cfg: RealizedCfg) -> Result<Self, String> {
        cfg.validate()?;
        Ok(Self {
            series: vec![Series::new(); cfg.n_targets],
            w_sum: 0.0,
            cfg,
        })
    }

    pub fn cfg(&self) -> &RealizedCfg {
        &self.cfg
    }

    /// The bytes of the numbers a model of `n_targets` series holds: `8 +
    /// 128 m`, whatever the stream's length or the decay.
    pub fn state_bytes(n_targets: usize) -> u64 {
        8 + Series::BYTES * n_targets as u64
    }

    /// The weight of every row, decayed (hard rule 8).
    pub fn n_eff(&self) -> f64 {
        self.w_sum
    }

    /// Each target's outputs into `pred`, from the state as it stands, for
    /// a row whose returns are `y`.
    fn read_into(&self, y: &[Option<f64>], pred: &mut Vec<f64>) {
        pred.clear();
        for (j, s) in self.series.iter().enumerate() {
            s.read(y.get(j).copied().flatten(), self.cfg.min_weight, pred);
        }
    }
}

impl OnlineModel for Realized {
    fn step_into(
        &mut self,
        _x: &[f64],
        y: &[Option<f64>],
        d_clock: f64,
        weight: f64,
        out: &mut Step,
    ) {
        // A return or a weight that is not usable, by the rule every model
        // keeps (`OnlineModel`): such a return is absent, and such a weight
        // learns nothing. The model reads no features.
        if let Some(refused) = crate::model::refused_step(self, &[], y, d_clock, weight) {
            *out = refused;
            return;
        }
        // Read before the row's return and before its decay (hard rules 2
        // and 8).
        self.read_into(y, &mut out.pred);
        out.n_eff = self.w_sum;
        out.extra = None;
        let f = self.cfg.decay.factor(d_clock);
        self.w_sum = f * self.w_sum + weight;
        for (j, s) in self.series.iter_mut().enumerate() {
            s.decay(f);
            // A row of weight 0 learns nothing, and keeps the last returns
            // as they are (hard rule 9).
            if weight > 0.0
                && let Some(r) = y.get(j).copied().flatten()
            {
                s.learn(r, weight);
            }
        }
    }

    fn predict(&self, x: &[f64], d_clock: f64) -> Step {
        self.predict_with(x, &[], d_clock)
    }

    /// The row's returns are read for `lm`, the row's own statistic; every
    /// other output is of the returns before the row.
    fn predict_with(&self, _x: &[f64], y: &[Option<f64>], d_clock: f64) -> Step {
        if let Some(refused) = crate::model::refused_predict_with(self, &[], y, d_clock) {
            return refused;
        }
        let mut pred = Vec::with_capacity(self.n_outputs());
        self.read_into(y, &mut pred);
        Step {
            pred,
            n_eff: self.w_sum,
            extra: None,
        }
    }

    fn target_n_eff_into(&self, out: &mut Vec<f64>) -> bool {
        out.clear();
        out.extend(self.series.iter().map(|s| s.sum(NW)));
        true
    }

    fn state(&self) -> State {
        State::new(ModelState::Realized(Box::new(self.clone())))
    }

    fn restore(s: &State) -> Result<Self, StateError> {
        check_schema(s)?;
        match &s.model {
            ModelState::Realized(m) => {
                let m = (**m).clone();
                crate::model::check_cfg("realized", m.cfg.validate())?;
                if m.series.len() != m.cfg.n_targets || m.series.iter().any(|s| s.seen > 2) {
                    return Err(crate::model::damaged(
                        "realized",
                        "its running sums have the wrong shape",
                    ));
                }
                Ok(m)
            }
            other => Err(StateError::WrongModel {
                expected: "realized",
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
        OUTPUTS_PER_TARGET * self.cfg.n_targets
    }
}

#[cfg(test)]
mod paper;
#[cfg(test)]
mod tests;
