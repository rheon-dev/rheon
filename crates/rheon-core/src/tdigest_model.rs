//! `TDigestModel`: one merging t-digest ([`crate::TDigest`]) per target
//! column, read at a list of quantile levels before each row is added. The
//! bank's `tdigest` kind.
//!
//! For target `t` with digest `D_t`, level `p` and the row's value `y`:
//!
//! ```text
//! q_p,t  = D_t.quantile(p)        read before the row (out of sample)
//! es_p,t = D_t.tail_mean(p)       the same digest's centroids past the
//!                                 level, the boundary one split: the lower
//!                                 tail below 0.5, the upper above it (0.5
//!                                 is refused)
//! D_t   <- D_t + (y, w)           a row of weight w > 0 with y present
//! W_t   <- W_t + w,   S_t <- S_t + w^2
//! ```
//!
//! **When a level is withheld.** `n_t = W_t^2 / S_t` is Kish's effective
//! size of the rows the digest holds: the number of rows of equal weight
//! that carry the same information, `W_t` itself for rows of weight 1, and
//! the same whatever scale every weight is given. The sample's `i`-th
//! smallest of `n` rows sits at expected level `i / (n + 1)` of the
//! distribution it is drawn from (the plotting position of Weibull's
//! definition, Hyndman and Fan's sixth); a level below `1 / (n + 1)` or
//! above `n / (n + 1)` lies past the smallest or largest row in
//! expectation, so the rows do not identify it: no rows bracket it, and the
//! digest's read there is the minimum or the maximum, not the level. The
//! level is reported where
//!
//! ```text
//! p (n_t + 1) >= 1   and   (1 - p) (n_t + 1) >= 1
//! ```
//!
//! and withheld otherwise, as NaN, with its tail mean (a tail whose level
//! is not identified holds no row in expectation), for the stream to name
//! ([`TDigestModel::beyond_sample`]). The rule is the condition itself, at
//! every sample size; no count of rows is fixed. Levels of 0.25 and 0.75 are
//! reported from the 3rd row of equal weight, 0.05 and 0.95 from the 19th,
//! 0.01 and 0.99 from the 99th.
//!
//! **The weight.** `n_eff` is every learned row's weight, with no decay (a
//! digest never forgets), before the row: the shared weight; each target's
//! own `W_t` is its rows' alone. A row of weight 0 learns nothing and
//! counts in no clock; a target that is absent or not usable is not added.
//!
//! **State**, bounded by the settings alone: `n_targets` digests of at most
//! [`crate::TDigest::max_bytes`] bytes each, plus two numbers a target.
//! Checked against `max_state_bytes` when the model is built.

use serde::{Deserialize, Serialize};

use crate::TDigest;
use crate::clock_span::{DEFAULT_STATE_BYTES, check_max_state_bytes};
use crate::model::{ModelState, OnlineModel, State, StateError, Step, check_schema};

/// The settings of a [`TDigestModel`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TDigestModelCfg {
    /// One digest per target column.
    pub n_targets: usize,
    /// The digest's `delta` ([`crate::TDigest`]): 100 by default.
    pub compression: f64,
    /// The levels read, each in (0, 1), no two alike; per target, one
    /// quantile a level, then one tail mean a level, in this order.
    pub levels: Vec<f64>,
    /// The bytes a group's state may hold; `None` is 256 KB.
    pub max_state_bytes: Option<u64>,
}

/// The compression a `tdigest` spec takes when it gives none.
pub const TDIGEST_COMPRESSION: f64 = 100.0;

impl TDigestModelCfg {
    fn cap(&self) -> u64 {
        self.max_state_bytes.unwrap_or(DEFAULT_STATE_BYTES)
    }

    /// The most bytes the state can hold: each target's fullest digest and
    /// its two sums, and the header.
    pub fn max_bytes(&self) -> u64 {
        std::mem::size_of::<TDigestModel>() as u64
            + self.n_targets as u64 * (TDigest::max_bytes(self.compression) + 16)
            + 8 * self.levels.len() as u64
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.n_targets == 0 {
            return Err(
                "tdigest: it needs at least one target, a column whose quantiles it \
                 reads. Name the column in targets."
                    .into(),
            );
        }
        TDigest::new(self.compression).map_err(|e| e.replace("TDigest:", "tdigest:"))?;
        if self.levels.is_empty() {
            return Err(
                "tdigest: quantiles must list at least one level, but it is empty. \
                 Give the levels to read, such as [0.05, 0.25, 0.95]."
                    .into(),
            );
        }
        if let Some(p) = self.levels.iter().find(|p| !(**p > 0.0 && **p < 1.0)) {
            return Err(format!(
                "tdigest: each of quantiles must be above 0 and below 1, but one is {p}. A \
                 level of 0 or 1 is the smallest or largest row, which no sample identifies \
                 as a quantile. Give levels strictly between 0 and 1."
            ));
        }
        if self.levels.contains(&0.5) {
            return Err(
                "tdigest: quantiles lists 0.5, whose tail is neither the lower nor the upper, \
                 so its expected shortfall has no side. Give a level below or above 0.5, such \
                 as 0.49 or 0.51; the median itself is the level a tail's mean is not taken at."
                    .into(),
            );
        }
        for (i, p) in self.levels.iter().enumerate() {
            if self.levels[..i].iter().any(|q| q.to_bits() == p.to_bits()) {
                return Err(format!(
                    "tdigest: quantiles lists {p} twice, and each level names an output. \
                     List each level once."
                ));
            }
        }
        check_max_state_bytes("tdigest", self.max_state_bytes)?;
        let (most, cap) = (self.max_bytes(), self.cap());
        if most > cap {
            return Err(format!(
                "tdigest: at compression {} a group's {} digests may hold {most} bytes, past \
                 max_state_bytes {cap}. Lower the compression, read fewer targets in this spec, \
                 or raise max_state_bytes.",
                self.compression, self.n_targets
            ));
        }
        Ok(())
    }
}

/// A t-digest a target, read at the configured levels (the module's
/// equations).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TDigestModel {
    cfg: TDigestModelCfg,
    digests: Vec<TDigest>,
    /// Every learned row's weight: the shared `n_eff`.
    w_sum: f64,
    /// Per target, the weight `W_t` and the squared weights `S_t` of the
    /// rows its digest holds.
    target_w: Vec<f64>,
    target_w2: Vec<f64>,
}

/// Whether level `p` is identified by `n` rows of equal weight: within the
/// plotting positions of the smallest and largest, `1/(n+1)` and
/// `n/(n+1)` (the module's rule).
pub fn level_identified(p: f64, n: f64) -> bool {
    p * (n + 1.0) >= 1.0 && (1.0 - p) * (n + 1.0) >= 1.0
}

impl TDigestModel {
    pub fn new(cfg: TDigestModelCfg) -> Result<Self, String> {
        cfg.validate()?;
        let digest = TDigest::new(cfg.compression)?;
        let n = cfg.n_targets;
        Ok(Self {
            digests: vec![digest; n],
            w_sum: 0.0,
            target_w: vec![0.0; n],
            target_w2: vec![0.0; n],
            cfg,
        })
    }

    pub fn cfg(&self) -> &TDigestModelCfg {
        &self.cfg
    }

    /// Every learned row's weight, with no decay: the shared `n_eff`.
    pub fn n_eff(&self) -> f64 {
        self.w_sum
    }

    /// Target `j`'s digest.
    pub fn digest(&self, j: usize) -> &TDigest {
        &self.digests[j]
    }

    /// Kish's effective size of target `j`'s rows, `W^2 / S`; 0 before any.
    pub fn effective_rows(&self, j: usize) -> f64 {
        let (w, s) = (self.target_w[j], self.target_w2[j]);
        if s > 0.0 { w / s * w } else { 0.0 }
    }

    /// Whether the next row reads some level of some target as not
    /// identified by its rows (the module's rule), so withheld.
    pub fn beyond_sample(&self) -> bool {
        (0..self.cfg.n_targets).any(|j| {
            let n = self.effective_rows(j);
            self.cfg.levels.iter().any(|&p| !level_identified(p, n))
        })
    }

    /// The bytes the state holds now.
    pub fn bytes(&self) -> u64 {
        std::mem::size_of::<Self>() as u64
            + self.digests.iter().map(|d| d.bytes() as u64).sum::<u64>()
            + 16 * self.cfg.n_targets as u64
            + 8 * self.cfg.levels.len() as u64
    }

    /// The levels and their tail means read before the next row, target by
    /// target (the quantiles at `2 nl j + l`, the tail means at
    /// `2 nl j + nl + l`), NaN where a level is withheld.
    fn read_into(&self, out: &mut Step) {
        let nl = self.cfg.levels.len();
        out.pred.clear();
        out.pred.resize(self.cfg.n_targets * 2 * nl, f64::NAN);
        out.n_eff = self.w_sum;
        out.extra = None;
        for (j, d) in self.digests.iter().enumerate() {
            let n = self.effective_rows(j);
            if !self.cfg.levels.iter().any(|&p| level_identified(p, n)) {
                continue;
            }
            let reads = d.quantiles_and_tails(&self.cfg.levels);
            for (li, (&p, r)) in self.cfg.levels.iter().zip(reads).enumerate() {
                if let Some((q, es)) = r.filter(|_| level_identified(p, n)) {
                    out.pred[2 * nl * j + li] = q;
                    out.pred[2 * nl * j + nl + li] = es;
                }
            }
        }
    }

    /// Whether a restored state has the shape [`Self::new`] gives it.
    fn has_shape(&self) -> bool {
        let n = self.cfg.n_targets;
        self.digests.len() == n
            && self.target_w.len() == n
            && self.target_w2.len() == n
            && self.digests.iter().all(|d| {
                d.compression().to_bits() == self.cfg.compression.to_bits() && d.has_shape()
            })
            && self
                .target_w
                .iter()
                .chain(&self.target_w2)
                .chain(std::iter::once(&self.w_sum))
                .all(|w| w.is_finite() && *w >= 0.0)
    }
}

impl OnlineModel for TDigestModel {
    fn step_into(
        &mut self,
        _x: &[f64],
        y: &[Option<f64>],
        d_clock: f64,
        weight: f64,
        out: &mut Step,
    ) {
        // A value or a weight that is not usable, by the rule every model
        // keeps (`OnlineModel`): such a value is absent, and such a weight
        // learns nothing. The model reads no features.
        if let Some(refused) = crate::model::refused_step(self, &[], y, d_clock, weight) {
            *out = refused;
            return;
        }
        self.read_into(out);
        if weight <= 0.0 {
            return;
        }
        self.w_sum += weight;
        for (j, d) in self.digests.iter_mut().enumerate() {
            if let Some(v) = y.get(j).copied().flatten() {
                d.update(v, weight);
                self.target_w[j] += weight;
                self.target_w2[j] += weight * weight;
            }
        }
    }

    fn predict(&self, _x: &[f64], _d_clock: f64) -> Step {
        let mut out = Step::default();
        self.read_into(&mut out);
        out
    }

    fn target_n_eff_into(&self, out: &mut Vec<f64>) -> bool {
        out.clear();
        out.extend_from_slice(&self.target_w);
        true
    }

    fn state(&self) -> State {
        State::new(ModelState::TDigest(Box::new(self.clone())))
    }

    fn restore(s: &State) -> Result<Self, StateError> {
        check_schema(s)?;
        match &s.model {
            ModelState::TDigest(m) => {
                let m = (**m).clone();
                crate::model::check_cfg("tdigest", m.cfg.validate())?;
                if !m.has_shape() {
                    return Err(crate::model::damaged(
                        "tdigest",
                        "its digests have the wrong shape",
                    ));
                }
                Ok(m)
            }
            other => Err(StateError::WrongModel {
                expected: "tdigest",
                found: other.kind(),
            }),
        }
    }

    fn n_targets(&self) -> usize {
        self.cfg.n_targets
    }

    /// The digests are the targets'; the model reads no features.
    fn n_features(&self) -> usize {
        0
    }

    fn n_outputs(&self) -> usize {
        self.cfg.n_targets * 2 * self.cfg.levels.len()
    }
}

#[cfg(test)]
#[path = "tdigest/model_tests.rs"]
mod tests;
