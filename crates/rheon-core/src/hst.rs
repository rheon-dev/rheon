//! Streaming Half-Space Trees anomaly detector (Tan, Ting and Liu 2011), in the
//! padded mass-profile formulation matching `river.anomaly.HalfSpaceTrees`.
//!
//! ## Construction
//!
//! Each of `n_trees` complete binary trees of height `height` (`2^(h+1) - 1`
//! nodes, `2^h - 1` internal) is built once from Philox4x64-10 lane 2 (`n = 0`)
//! in pre-order (root, left subtree, right subtree), using two uniforms per
//! internal node over the node's feature box `[a_i, b_i]`:
//!
//! ```text
//! feature  q = bisect_right(cumsum(b_i - a_i), u1 * total)
//! split    lo = a_q + 0.15 * (b_q - a_q),   hi = b_q - 0.15 * (b_q - a_q)
//!          s  = lo + (hi - lo) * u2
//! ```
//!
//! Because construction depends solely on `(seed, group_hash, slot, limits,
//! n_trees, height)` via pure IEEE arithmetic and integer comparisons, the
//! splits are not stored in serialized state and are rebuilt bit-identically on
//! restore on every OS.
//!
//! ## Learning and scoring
//!
//! On each learned row (`w > 0`), `l_mass += w` along the row's root-to-leaf
//! path (`x_q < s` goes left, else right) in every tree. After `window_rows`
//! learned rows, the tumbling window pivots: `r_mass = l_mass`, `l_mass = 0`.
//!
//! Before the row is learned (out of sample), during the first tumbling window
//! (`r_mass` empty) or while `weight_sum < min_weight`, the score is `NaN`
//! (withheld). Otherwise, with `ref_mass = r_mass[tree 0, root]` (equal to
//! `window_rows` at unit weights, and weight-scale invariant under arbitrary
//! positive weights):
//!
//! ```text
//! per tree from root (depth 0):
//!     S += r_node * 2^depth
//!     stop if r_node < 0.1 * ref_mass; else descend to depth height
//! score = 1 - (sum_trees S) / (n_trees * ref_mass * (2^(height + 1) - 1))
//! ```
//!
//! Higher scores (near 1) indicate anomalies; lower scores (near 0) indicate
//! dense regions.

use serde::{Deserialize, Serialize};

use crate::model::{ModelState, OnlineModel, State, StateError, Step, check_schema};
use crate::philox::{DrawStream, LANE_HST_BUILD, LaneReader};

/// Default per-group state cap for [`HalfSpaceTrees`]: 2 MiB.
pub const HST_MAX_STATE_BYTES: u64 = 2 * 1024 * 1024;
/// Maximum allowed tree height (prevents shift overflow).
pub const HST_MAX_HEIGHT: usize = 16;

const PADDING: f64 = 0.15;

/// Configuration for [`HalfSpaceTrees`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HalfSpaceTreesCfg {
    pub n_features: usize,
    pub n_trees: usize,
    pub height: usize,
    /// Number of positive-weight learned rows per tumbling mass window.
    pub window_rows: usize,
    /// Declared `(lo, hi)` bounds per feature (`lo < hi`, finite).
    pub limits: Vec<(f64, f64)>,
    pub min_weight: f64,
    pub seed: u64,
    #[serde(default)]
    pub group_hash: u64,
    #[serde(default)]
    pub slot: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_state_bytes: Option<u64>,
}

impl HalfSpaceTreesCfg {
    #[inline]
    pub fn n_internal(&self) -> usize {
        (1usize << self.height) - 1
    }

    #[inline]
    pub fn n_nodes(&self) -> usize {
        (1usize << (self.height + 1)) - 1
    }

    /// Fixed per-group serialized state bytes (mass arrays + counters; splits
    /// are rebuilt deterministically from seed on restore).
    pub fn state_bytes(&self) -> u64 {
        if self.height > HST_MAX_HEIGHT {
            return u64::MAX;
        }
        let n_nodes = (1u64 << (self.height + 1)) - 1;
        (self.n_trees as u64)
            .saturating_mul(n_nodes)
            .saturating_mul(16)
            .saturating_add(25)
    }

    pub fn cap_bytes(&self) -> u64 {
        self.max_state_bytes.unwrap_or(HST_MAX_STATE_BYTES)
    }

    pub fn validate(&self) -> Result<(), String> {
        let mut no = crate::Refusals::new();
        if self.n_features == 0 {
            no.refuse(
                "half_space_trees: the model needs at least one feature, but it has none. Name \
                 at least one column in features.",
            );
        }
        if self.n_trees == 0 {
            no.refuse(
                "half_space_trees: n_trees must be at least 1, but it is 0. Set n_trees to a \
                 positive integer, such as the default 10.",
            );
        }
        if self.height == 0 || self.height > HST_MAX_HEIGHT {
            no.refuse(format!(
                "half_space_trees: height must be from 1 to {HST_MAX_HEIGHT}, but it is {}. Set \
                 height to a positive integer in that range, such as the default 8.",
                self.height
            ));
        }
        if self.window_rows == 0 {
            no.refuse(
                "half_space_trees: window_rows must be at least 1, but it is 0. Set window_rows \
                 to the number of learned rows per tumbling mass window, such as the default 250.",
            );
        }
        if self.limits.len() != self.n_features {
            no.refuse(format!(
                "half_space_trees: limits must hold one (lo, hi) pair per feature ({}), but it \
                 holds {}. Pass one (lo, hi) pair for each feature.",
                self.n_features,
                self.limits.len()
            ));
        } else {
            for (i, &(lo, hi)) in self.limits.iter().enumerate() {
                if !(lo.is_finite() && hi.is_finite() && lo < hi) {
                    no.refuse(format!(
                        "half_space_trees: limits for feature {i} must be finite with lo < hi, \
                         but it is ({lo}, {hi}). Set finite bounds with lower strictly below upper."
                    ));
                    break;
                }
            }
        }
        no.take(crate::model::check_min_weight(
            "half_space_trees",
            self.min_weight,
        ));
        no.take(crate::clock_span::check_max_state_bytes(
            "half_space_trees",
            self.max_state_bytes,
        ));
        if self.n_trees > 0 && self.height >= 1 && self.height <= HST_MAX_HEIGHT {
            let cap = self.cap_bytes();
            let bytes = self.state_bytes();
            if cap > 0 && cap <= crate::clock_span::STATE_BYTES_CEILING && bytes > cap {
                no.refuse(format!(
                    "half_space_trees: state holds {bytes} bytes for {} trees of height {}, past \
                     max_state_bytes of {cap}. Lower n_trees or height, or raise max_state_bytes.",
                    self.n_trees, self.height
                ));
            }
        }
        no.finish()
    }
}

fn bisect_right(cum: &[f64], target: f64) -> usize {
    let mut lo = 0usize;
    let mut hi = cum.len().saturating_sub(1);
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if target < cum[mid] {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    lo
}

fn build_node<F: FnMut() -> f64>(
    feature: &mut [u32],
    split: &mut [f64],
    cum: &mut [f64],
    lim: &mut [(f64, f64)],
    i: usize,
    rem_height: usize,
    uniform: &mut F,
) {
    if rem_height == 0 {
        return;
    }
    let mut acc = 0.0;
    for (c, &(a, b)) in cum.iter_mut().zip(lim.iter()) {
        acc += b - a;
        *c = acc;
    }
    let total = acc;
    let u1 = uniform();
    let q = bisect_right(cum, u1 * total);
    let (a, b) = lim[q];
    let lo = a + PADDING * (b - a);
    let hi = b - PADDING * (b - a);
    let s = lo + (hi - lo) * uniform();
    feature[i] = q as u32;
    split[i] = s;
    lim[q] = (a, s);
    build_node(feature, split, cum, lim, 2 * i + 1, rem_height - 1, uniform);
    lim[q] = (s, b);
    build_node(feature, split, cum, lim, 2 * i + 2, rem_height - 1, uniform);
    lim[q] = (a, b);
}

fn build_trees_from<F: FnMut() -> f64>(
    cfg: &HalfSpaceTreesCfg,
    mut uniform: F,
) -> (Vec<u32>, Vec<f64>) {
    let n_int = cfg.n_internal();
    let mut feature = vec![0u32; cfg.n_trees * n_int];
    let mut split = vec![0.0; cfg.n_trees * n_int];
    let mut cum = vec![0.0; cfg.n_features];
    let mut lim = cfg.limits.clone();
    for t in 0..cfg.n_trees {
        let base = t * n_int;
        build_node(
            &mut feature[base..base + n_int],
            &mut split[base..base + n_int],
            &mut cum,
            &mut lim,
            0,
            cfg.height,
            &mut uniform,
        );
    }
    (feature, split)
}

fn build_trees(cfg: &HalfSpaceTreesCfg) -> (Vec<u32>, Vec<f64>) {
    let stream = DrawStream::new(cfg.seed, cfg.group_hash, cfg.slot);
    let mut reader: LaneReader = stream.reader(LANE_HST_BUILD, 0, 0);
    build_trees_from(cfg, || reader.next_u01())
}

/// Streaming Half-Space Trees anomaly detector.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HalfSpaceTrees {
    cfg: HalfSpaceTreesCfg,
    /// Latest tumbling window mass per tree and node (`n_trees * n_nodes`).
    l: Vec<f64>,
    /// Reference tumbling window mass per tree and node (`n_trees * n_nodes`).
    r: Vec<f64>,
    /// Learned rows accumulated in the current window (`0..window_rows`).
    counter: usize,
    /// Total positive-weight learned rows.
    n_learned: u64,
    /// Total accumulated weight (`n_eff`).
    w_sum: f64,
    /// True until the first tumbling window pivots into `r`.
    first_window: bool,
    /// Split feature per internal node (rebuilt deterministically on restore).
    #[serde(skip)]
    feature: Vec<u32>,
    /// Split threshold per internal node (rebuilt deterministically on restore).
    #[serde(skip)]
    split: Vec<f64>,
}

impl HalfSpaceTrees {
    pub fn new(cfg: HalfSpaceTreesCfg) -> Result<Self, String> {
        cfg.validate()?;
        let n_all = cfg.n_trees * cfg.n_nodes();
        let (feature, split) = build_trees(&cfg);
        Ok(Self {
            l: vec![0.0; n_all],
            r: vec![0.0; n_all],
            counter: 0,
            n_learned: 0,
            w_sum: 0.0,
            first_window: true,
            feature,
            split,
            cfg,
        })
    }

    pub fn cfg(&self) -> &HalfSpaceTreesCfg {
        &self.cfg
    }

    /// True during the first tumbling window, before any reference profile
    /// exists (`withheld_reason = "below_warm_rows"`).
    #[inline]
    pub fn warming(&self) -> bool {
        self.first_window
    }

    pub fn n_eff(&self) -> f64 {
        self.w_sum
    }

    pub fn n_learned(&self) -> u64 {
        self.n_learned
    }

    #[inline]
    fn child(&self, t_int: usize, i: usize, x: &[f64]) -> usize {
        let feat = self.feature[t_int + i] as usize;
        if x[feat] < self.split[t_int + i] {
            2 * i + 1
        } else {
            2 * i + 2
        }
    }

    fn score_of(&self, x: &[f64]) -> f64 {
        if self.first_window || self.w_sum < self.cfg.min_weight {
            return f64::NAN;
        }
        let ref_mass = self.r[0];
        if !(ref_mass > 0.0 && ref_mass.is_finite()) {
            return f64::NAN;
        }
        let size_limit = 0.1 * ref_mass;
        let n_int = self.cfg.n_internal();
        let n_nodes = self.cfg.n_nodes();
        let h = self.cfg.height;
        let mut total = 0.0;
        for t in 0..self.cfg.n_trees {
            let t_int = t * n_int;
            let t_all = t * n_nodes;
            let mut i = 0usize;
            let mut pw = 1u64;
            let mut s = 0.0;
            let mut stopped = false;
            for _ in 0..h {
                let rm = self.r[t_all + i];
                s += rm * (pw as f64);
                if rm < size_limit {
                    stopped = true;
                    break;
                }
                i = self.child(t_int, i, x);
                pw <<= 1;
            }
            if !stopped {
                s += self.r[t_all + i] * (pw as f64);
            }
            total += s;
        }
        let max_score = (self.cfg.n_trees as f64) * ref_mass * (n_nodes as f64);
        1.0 - total / max_score
    }
}

impl OnlineModel for HalfSpaceTrees {
    fn step_into(
        &mut self,
        x: &[f64],
        y: &[Option<f64>],
        d_clock: f64,
        weight: f64,
        out: &mut Step,
    ) {
        if let Some(refused) = crate::model::refused_step(self, x, y, d_clock, weight) {
            *out = refused;
            return;
        }
        out.pred.clear();
        out.pred.push(self.score_of(x));
        out.n_eff = self.w_sum;
        out.extra = None;

        if weight > 0.0 {
            let n_int = self.cfg.n_internal();
            let n_nodes = self.cfg.n_nodes();
            let h = self.cfg.height;
            for t in 0..self.cfg.n_trees {
                let t_int = t * n_int;
                let t_all = t * n_nodes;
                let mut i = 0usize;
                for _ in 0..h {
                    self.l[t_all + i] += weight;
                    i = self.child(t_int, i, x);
                }
                self.l[t_all + i] += weight;
            }
            self.w_sum += weight;
            self.n_learned += 1;
            self.counter += 1;
            if self.counter == self.cfg.window_rows {
                self.r.copy_from_slice(&self.l);
                self.l.fill(0.0);
                self.first_window = false;
                self.counter = 0;
            }
        }
    }

    fn predict(&self, x: &[f64], d_clock: f64) -> Step {
        if let Some(refused) = crate::model::refused_predict(self, x, d_clock) {
            return refused;
        }
        Step {
            pred: vec![self.score_of(x)],
            n_eff: self.w_sum,
            extra: None,
        }
    }

    fn n_features(&self) -> usize {
        self.cfg.n_features
    }

    fn n_targets(&self) -> usize {
        0
    }

    fn n_outputs(&self) -> usize {
        1
    }

    fn state(&self) -> State {
        State::new(ModelState::HalfSpaceTrees(Box::new(self.clone())))
    }

    fn restore(state: &State) -> Result<Self, StateError> {
        check_schema(state)?;
        let mut m = match &state.model {
            ModelState::HalfSpaceTrees(h) => (**h).clone(),
            other => {
                return Err(StateError::WrongModel {
                    expected: "half_space_trees",
                    found: other.kind(),
                });
            }
        };
        crate::model::check_cfg("half_space_trees", m.cfg.validate())?;
        let n_all = m.cfg.n_trees * m.cfg.n_nodes();
        if m.l.len() != n_all || m.r.len() != n_all {
            return Err(crate::model::damaged(
                "half_space_trees",
                "its mass arrays have the wrong shape",
            ));
        }
        if m.counter >= m.cfg.window_rows {
            return Err(crate::model::damaged(
                "half_space_trees",
                "its window counter is past window_rows",
            ));
        }
        if !(m.w_sum.is_finite() && m.w_sum >= 0.0) {
            return Err(crate::model::damaged(
                "half_space_trees",
                "its accumulated weight is not a non-negative finite number",
            ));
        }
        if !m.l.iter().all(|&v| v.is_finite() && v >= 0.0)
            || !m.r.iter().all(|&v| v.is_finite() && v >= 0.0)
        {
            return Err(crate::model::damaged(
                "half_space_trees",
                "its mass arrays contain a negative or non-finite value",
            ));
        }
        let (feature, split) = build_trees(&m.cfg);
        m.feature = feature;
        m.split = split;
        Ok(m)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> HalfSpaceTreesCfg {
        HalfSpaceTreesCfg {
            n_features: 2,
            n_trees: 10,
            height: 6,
            window_rows: 32,
            limits: vec![(0.0, 1.0), (0.0, 1.0)],
            min_weight: 0.0,
            seed: 42,
            group_hash: 0,
            slot: 0,
            max_state_bytes: None,
        }
    }

    #[test]
    fn first_window_withholds_then_separates_anomalies_and_rebuilds_trees_on_restore() {
        let mut h = HalfSpaceTrees::new(cfg()).unwrap();
        for i in 0..32 {
            let x = [0.5 + 0.01 * ((i % 5) as f64 - 2.0), 0.5];
            let s = h.step(&x, &[], 1.0, 1.0);
            assert!(
                s.pred[0].is_nan(),
                "row {i} should withhold during window 1"
            );
            assert!(h.warming() == (i < 31));
        }
        assert!(!h.warming());
        let normal_score = h.predict(&[0.5, 0.5], 1.0).pred[0];
        let outlier_score = h.predict(&[0.01, 0.99], 1.0).pred[0];
        assert!(
            outlier_score > normal_score + 0.3,
            "outlier {outlier_score} should exceed normal {normal_score}"
        );

        // Restore rebuilds splits identically without saving them.
        let saved = h.state();
        let mut restored = HalfSpaceTrees::restore(&saved).unwrap();
        assert_eq!(restored.feature, h.feature);
        assert_eq!(restored.split, h.split);
        for x in [[0.5, 0.5], [0.01, 0.99], [0.25, 0.75]] {
            assert_eq!(restored.step(&x, &[], 1.0, 1.0), h.step(&x, &[], 1.0, 1.0));
        }
    }

    #[test]
    fn weight_scaling_leaves_scores_invariant() {
        let mut h1 = HalfSpaceTrees::new(cfg()).unwrap();
        let mut h2 = HalfSpaceTrees::new(cfg()).unwrap();
        for i in 0..100 {
            let x = [
                ((i * 17) % 100) as f64 / 100.0,
                ((i * 31) % 100) as f64 / 100.0,
            ];
            let w = if i % 4 == 0 { 0.5 } else { 1.5 };
            let s1 = h1.step(&x, &[], 1.0, w);
            let s2 = h2.step(&x, &[], 1.0, w * 7.25);
            if s1.pred[0].is_nan() {
                assert!(s2.pred[0].is_nan());
            } else {
                assert!(
                    (s1.pred[0] - s2.pred[0]).abs() < 1e-13,
                    "row {i}: {} vs {}",
                    s1.pred[0],
                    s2.pred[0]
                );
                assert!((0.0..=1.0).contains(&s2.pred[0]));
            }
        }
    }

    #[test]
    fn a_state_of_the_wrong_shape_is_refused() {
        let mut h = HalfSpaceTrees::new(cfg()).unwrap();
        h.step(&[0.5, 0.5], &[], 1.0, 1.0);
        let mut bad = h.clone();
        bad.l.pop();
        assert!(HalfSpaceTrees::restore(&bad.state()).is_err());

        let mut bad2 = h.clone();
        bad2.counter = bad2.cfg.window_rows;
        assert!(HalfSpaceTrees::restore(&bad2.state()).is_err());
    }
}
