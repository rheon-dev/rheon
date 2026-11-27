//! `HoeffdingTreeRegressor`: online regression tree (FIMT-DD mean leaves,
//! Ikonomovska, Gama and Dzeroski 2011) with bounded Quantization Observers
//! (QO, Mastelini and de Carvalho 2021) and an exact per-group byte cap.
//!
//! Each target owns one tree. Each leaf keeps the target's weighted running
//! statistics `(W, mean, S)` (variance `S / (W - 1)`) and, while active, one
//! bounded QO observer per feature (`radius`, at most `max_slots` buckets;
//! excess buckets fold into the nearest existing bucket). Every `grace_period`
//! of learned weight the leaf scores the midpoint between each pair of
//! consecutive sorted QO slots by variance reduction:
//!
//! ```text
//! VR = var(S) - (W_l / W) * var(S_l) - (W_r / W) * var(S_r)
//!      (0 when W_l < min_samples_split or W_r < min_samples_split)
//! split when  VR(second) / VR(best) < 1 - eps   or   eps < tau,   with VR(best) > 0
//! eps = sqrt(ln(1 / delta) / (2 * W))
//! ```
//!
//! Each leaf predicts its weighted target mean (`mean`). With
//! `merit_preprune = true`, the null split enters at merit `0.0`. Both QO bugs
//! documented against `river`'s `QOSplitter` (off-by-one partition and aliased
//! left-child accumulator) are fixed here.
//!
//! **Per-group memory cap (`max_state_bytes`).** Default 256 KB, ceiling
//! 64 MiB (1 MB recommended for `n_features >= 20`). Multi-target specs split
//! the group cap equally across targets (`cap / n_targets` per tree). Packed
//! bytes per tree are `40 + 20B + 35L + sum(2 + 40 * slots)` (`<= struct +
//! k * (2 + 40 * max_slots) * A`). Splits stop once `struct_bytes(L + 1,
//! B + 1)` would pass the tree's cap; after every split and every 1,000
//! learned rows, eligible leaves are ranked by SSE promise (`S`) and at most
//! `(tree_cap - struct_bytes) / (k * (2 + 40 * max_slots))` stay active.
//! Cap-deactivated leaves keep updating their target statistics and predicting
//! their mean, and reactivate with fresh observers if their promise ranks back
//! in. At `max_depth`, a leaf drops its observers immediately (zero effect on
//! mean predictions).
//!
//! **Portability and contracts.** All transcendentals (`log`) go through the
//! pure-Rust `libm` crate, so trees and saved states are bit-identical across
//! macOS, Linux and Windows. Weights are frequency weights. Zero-weight rows
//! predict out-of-sample and touch no state, observer or counter. No decay
//! and no rolling window.

use serde::{Deserialize, Serialize};

use crate::clock_span::{DEFAULT_STATE_BYTES, check_max_state_bytes};
use crate::model::{
    ModelState, OnlineModel, State, StateError, Step, check_dims, check_min_weight, check_schema,
};

mod qo;
use qo::{QO_HEADER_BYTES, QO_SLOT_BYTES, QoObserver, TargetVar};

/// Packed tree header bytes.
const REG_HEADER_BYTES: u64 = 40;
/// Packed branch bytes (`feature: u32`, `threshold: f64`, `left: i32`, `right: i32`).
const REG_BRANCH_BYTES: u64 = 20;
/// Packed leaf bytes (`n`, `mean`, `s`, `last` = 32, `depth: u16` = 2, `flags: u8` = 1).
const REG_LEAF_BYTES: u64 = 35;
/// Learned rows per tree between memory-cap promise re-rankings.
const RECHECK_INTERVAL: u64 = 1_000;
/// Default effective depth ceiling when `max_depth` is `None`.
const DEFAULT_MAX_DEPTH: usize = u16::MAX as usize;

/// Configuration for [`HoeffdingTreeRegressor`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HoeffdingTreeRegressorCfg {
    /// Number of numeric features (`>= 1`).
    pub n_features: usize,
    /// Number of targets (`>= 1`; one independent tree per target).
    pub n_targets: usize,
    /// Learned weight at a leaf between split attempts (`> 0`, finite; default `200.0`).
    pub grace_period: f64,
    /// Hoeffding bound error probability in `(0, 1)` (default `1e-7`).
    pub delta: f64,
    /// Tie-breaking threshold `tau >= 0`, finite (default `0.05`).
    pub tau: f64,
    /// Minimum weight on each side of a candidate split (`>= 1`, finite; default `5.0`).
    pub min_samples_split: f64,
    /// Quantization observer bucket radius (`> 0`, finite; default `0.25`).
    pub radius: f64,
    /// Hard cap on QO slots per feature per active leaf (`>= 2`; default `32`).
    pub max_slots: usize,
    /// Include the null split (merit `0.0`) as a pre-pruning competitor (default `true`).
    pub merit_preprune: bool,
    /// Optional tree depth cap (`None` = unbounded up to `u16::MAX`).
    pub max_depth: Option<usize>,
    /// Minimum accumulated target weight before predicting (`>= 0`; default `0.0`).
    pub min_weight: f64,
    /// Per-group packed state cap in bytes (`None` = 256 KB; max 64 MiB).
    pub max_state_bytes: Option<u64>,
}

impl Default for HoeffdingTreeRegressorCfg {
    fn default() -> Self {
        Self {
            n_features: 1,
            n_targets: 1,
            grace_period: 200.0,
            delta: 1e-7,
            tau: 0.05,
            min_samples_split: 5.0,
            radius: 0.25,
            max_slots: 32,
            merit_preprune: true,
            max_depth: None,
            min_weight: 0.0,
            max_state_bytes: None,
        }
    }
}

/// Packed structural bytes of one regression tree (excluding active observers).
pub fn reg_struct_bytes(n_leaves: usize, n_branches: usize) -> u64 {
    REG_HEADER_BYTES
        .saturating_add(REG_BRANCH_BYTES.saturating_mul(n_branches as u64))
        .saturating_add(REG_LEAF_BYTES.saturating_mul(n_leaves as u64))
}

/// Worst-case packed observer bytes reserved for one active regression leaf.
pub fn reg_obs_reserve(n_features: usize, max_slots: usize) -> u64 {
    let per_feat = QO_HEADER_BYTES.saturating_add(QO_SLOT_BYTES.saturating_mul(max_slots as u64));
    per_feat.saturating_mul(n_features as u64)
}

impl HoeffdingTreeRegressorCfg {
    /// Active per-group cap in bytes (`DEFAULT_STATE_BYTES` when unset).
    pub fn cap(&self) -> u64 {
        self.max_state_bytes.unwrap_or(DEFAULT_STATE_BYTES)
    }

    /// Cap allotted to each target's tree (`cap / n_targets`).
    pub fn per_tree_cap(&self) -> u64 {
        self.cap() / (self.n_targets.max(1) as u64)
    }

    /// Minimum bytes needed for one active root leaf with observers per target.
    pub fn min_state_bytes(&self) -> u64 {
        let one_tree =
            reg_struct_bytes(1, 0).saturating_add(reg_obs_reserve(self.n_features, self.max_slots));
        one_tree.saturating_mul(self.n_targets as u64)
    }

    fn effective_max_depth(&self) -> usize {
        self.max_depth.unwrap_or(DEFAULT_MAX_DEPTH)
    }

    pub fn validate(&self) -> Result<(), String> {
        let k_name = "hoeffding_tree_regressor";
        let mut no = crate::Refusals::new();
        no.take(check_dims(k_name, self.n_features, self.n_targets));
        if !(self.grace_period.is_finite() && self.grace_period > 0.0) {
            no.refuse(format!(
                "{k_name}: grace_period must be finite and above 0, but it is {}. Set \
                 grace_period to the positive weight a leaf gathers between split attempts.",
                self.grace_period
            ));
        }
        if !(self.delta.is_finite() && self.delta > 0.0 && self.delta < 1.0) {
            no.refuse(format!(
                "{k_name}: delta must be finite, above 0 and below 1, but it is {}. Set delta \
                 to the Hoeffding bound error probability, such as 1e-7.",
                self.delta
            ));
        }
        if !(self.tau.is_finite() && self.tau >= 0.0) {
            no.refuse(format!(
                "{k_name}: tau must be finite and at least 0, but it is {}. Set tau to the \
                 tie-breaking threshold, such as 0.05.",
                self.tau
            ));
        }
        if !(self.min_samples_split.is_finite() && self.min_samples_split >= 1.0) {
            no.refuse(format!(
                "{k_name}: min_samples_split must be finite and at least 1, but it is {}. Set \
                 min_samples_split to the minimum weight each child must receive.",
                self.min_samples_split
            ));
        }
        if !(self.radius.is_finite() && self.radius > 0.0) {
            no.refuse(format!(
                "{k_name}: radius must be finite and above 0, but it is {}. Set radius to the \
                 quantization bucket width, such as 0.25.",
                self.radius
            ));
        }
        if self.max_slots < 2 {
            no.refuse(format!(
                "{k_name}: max_slots must be at least 2, but it is {}. Set max_slots to the \
                 maximum number of quantization buckets per feature.",
                self.max_slots
            ));
        }
        if let Some(d) = self.max_depth
            && d > DEFAULT_MAX_DEPTH
        {
            no.refuse(format!(
                "{k_name}: max_depth may be at most {DEFAULT_MAX_DEPTH}, but it is {d}. Lower \
                 max_depth or leave it out."
            ));
        }
        no.take(check_min_weight(k_name, self.min_weight));
        no.take(check_max_state_bytes(k_name, self.max_state_bytes));
        if self.n_features >= 1
            && self.n_targets >= 1
            && self.max_slots >= 2
            && self.max_state_bytes != Some(0)
        {
            let cap = self.cap();
            let one_tree = reg_struct_bytes(1, 0)
                .saturating_add(reg_obs_reserve(self.n_features, self.max_slots));
            if cap <= crate::clock_span::STATE_BYTES_CEILING && one_tree > self.per_tree_cap() {
                let need = one_tree.saturating_mul(self.n_targets as u64);
                no.refuse(format!(
                    "{k_name}: one active leaf per target ({} targets, {} features, max_slots {}) \
                     needs {need} bytes, past max_state_bytes ({cap}). Raise max_state_bytes to \
                     at least {need}, lower max_slots, or fit fewer targets in one spec.",
                    self.n_targets, self.n_features, self.max_slots
                ));
            }
        }
        no.finish()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum NodeRef {
    Branch(u32),
    Leaf(u32),
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
struct RegBranch {
    feature: u32,
    threshold: f64,
    left: NodeRef,
    right: NodeRef,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct RegLeaf {
    stats: TargetVar,
    obs: Option<Vec<QoObserver>>,
    depth: u16,
    last_split_attempt: f64,
    active: bool,
    capped: bool,
}

impl RegLeaf {
    fn new(depth: u16, stats: TargetVar) -> Self {
        Self {
            last_split_attempt: stats.n,
            stats,
            obs: None,
            depth,
            active: true,
            capped: false,
        }
    }
}

struct RegSplitCand {
    feature: usize,
    threshold: f64,
    left: TargetVar,
    right: TargetVar,
}

/// One target's regression tree.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct RegTree {
    leaves: Vec<RegLeaf>,
    branches: Vec<RegBranch>,
    root: NodeRef,
    target_weight: f64,
    rows_learned: u64,
}

impl RegTree {
    fn new() -> Self {
        Self {
            leaves: vec![RegLeaf::new(0, TargetVar::default())],
            branches: Vec::new(),
            root: NodeRef::Leaf(0),
            target_weight: 0.0,
            rows_learned: 0,
        }
    }

    #[inline]
    fn leaf_of(&self, x: &[f64]) -> usize {
        let mut node = self.root;
        while let NodeRef::Branch(bi) = node {
            let b = self.branches[bi as usize];
            node = if x[b.feature as usize] <= b.threshold {
                b.left
            } else {
                b.right
            };
        }
        match node {
            NodeRef::Leaf(li) => li as usize,
            NodeRef::Branch(_) => unreachable!(),
        }
    }

    fn state_bytes(&self) -> u64 {
        let mut b = reg_struct_bytes(self.leaves.len(), self.branches.len());
        for lf in &self.leaves {
            if let Some(obs) = &lf.obs {
                for ob in obs {
                    b += ob.bytes();
                }
            }
        }
        b
    }

    fn reserved_bytes(&self, cfg: &HoeffdingTreeRegressorCfg) -> u64 {
        let active = self.leaves.iter().filter(|l| l.active).count() as u64;
        reg_struct_bytes(self.leaves.len(), self.branches.len())
            + active * reg_obs_reserve(cfg.n_features, cfg.max_slots)
    }

    fn may_grow(&self, tree_cap: u64) -> bool {
        reg_struct_bytes(self.leaves.len() + 1, self.branches.len() + 1) <= tree_cap
    }

    fn enforce_cap(&mut self, cfg: &HoeffdingTreeRegressorCfg, tree_cap: u64) {
        let sb = reg_struct_bytes(self.leaves.len(), self.branches.len());
        let reserve = reg_obs_reserve(cfg.n_features, cfg.max_slots);
        let budget = tree_cap.saturating_sub(sb);
        let max_active = (budget / reserve) as usize;
        let mut elig: Vec<usize> = self
            .leaves
            .iter()
            .enumerate()
            .filter_map(|(i, l)| (l.active || l.capped).then_some(i))
            .collect();
        // Promise = leaf sum of squared deviations S (Study 503 §5.3 winner).
        elig.sort_by(|&a, &b| self.leaves[a].stats.s.total_cmp(&self.leaves[b].stats.s));
        let cutoff = elig.len().saturating_sub(max_active);
        for &i in &elig[..cutoff] {
            let lf = &mut self.leaves[i];
            if lf.active {
                lf.active = false;
                lf.capped = true;
                lf.obs = None;
            }
        }
        for &i in &elig[cutoff..] {
            let lf = &mut self.leaves[i];
            if lf.capped {
                lf.active = true;
                lf.capped = false;
                lf.obs = None;
            }
        }
    }

    fn learn(&mut self, x: &[f64], y: f64, w: f64, cfg: &HoeffdingTreeRegressorCfg, tree_cap: u64) {
        self.target_weight += w;
        self.rows_learned += 1;
        if self.rows_learned.is_multiple_of(RECHECK_INTERVAL) {
            self.enforce_cap(cfg, tree_cap);
        }
        let mut node = self.root;
        let mut parent: Option<usize> = None;
        let mut went_right = false;
        while let NodeRef::Branch(bi) = node {
            let b = self.branches[bi as usize];
            parent = Some(bi as usize);
            if x[b.feature as usize] <= b.threshold {
                node = b.left;
                went_right = false;
            } else {
                node = b.right;
                went_right = true;
            }
        }
        let li = match node {
            NodeRef::Leaf(i) => i as usize,
            NodeRef::Branch(_) => unreachable!(),
        };
        let max_d = cfg.effective_max_depth();
        let k = cfg.n_features;
        let leaf = &mut self.leaves[li];
        leaf.stats.update(y, w);
        if !leaf.active {
            return;
        }
        if (leaf.depth as usize) >= max_d {
            // At max_depth a regression leaf never splits and predicts its mean;
            // drop observers immediately to reclaim budget.
            leaf.active = false;
            leaf.obs = None;
            return;
        }
        let obs = leaf
            .obs
            .get_or_insert_with(|| vec![QoObserver::default(); k]);
        for (f, &xf) in x.iter().enumerate().take(k) {
            obs[f].update(xf, y, w, cfg.radius, cfg.max_slots);
        }
        if (leaf.stats.n - leaf.last_split_attempt) >= cfg.grace_period
            && !self.attempt_split(li, parent, went_right, cfg, tree_cap)
        {
            self.leaves[li].last_split_attempt = self.leaves[li].stats.n;
        }
    }

    fn attempt_split(
        &mut self,
        li: usize,
        parent: Option<usize>,
        went_right: bool,
        cfg: &HoeffdingTreeRegressorCfg,
        tree_cap: u64,
    ) -> bool {
        let pre = self.leaves[li].stats;
        if pre.n <= 0.0 {
            return false;
        }
        let mut sugg: Vec<(f64, Option<RegSplitCand>)> = Vec::with_capacity(cfg.n_features + 1);
        if cfg.merit_preprune {
            sugg.push((0.0, None));
        }
        if let Some(obs) = self.leaves[li].obs.as_deref() {
            for (f, ob) in obs.iter().enumerate() {
                match ob.best_split(cfg.min_samples_split) {
                    Some(c) => sugg.push((
                        c.merit,
                        Some(RegSplitCand {
                            feature: f,
                            threshold: c.threshold,
                            left: c.left,
                            right: c.right,
                        }),
                    )),
                    None => sugg.push((f64::NEG_INFINITY, None)),
                }
            }
        }
        sugg.sort_by(|a, b| a.0.total_cmp(&b.0));
        let should = if sugg.len() < 2 {
            sugg.first().is_some_and(|s| s.0 > 0.0)
        } else {
            let eps = (libm::log(1.0 / cfg.delta) / (2.0 * pre.n)).sqrt();
            let best_m = sugg[sugg.len() - 1].0;
            let second_m = sugg[sugg.len() - 2].0;
            best_m > 0.0 && (second_m / best_m < 1.0 - eps || eps < cfg.tau)
        };
        if !should {
            return false;
        }
        let Some(best) = sugg.pop().and_then(|(_, c)| c) else {
            let lf = &mut self.leaves[li];
            lf.active = false;
            lf.obs = None;
            return false;
        };
        if !self.may_grow(tree_cap) {
            return false;
        }
        let next_depth = self.leaves[li].depth.saturating_add(1);
        let new_l = RegLeaf::new(next_depth, best.left);
        let new_r = RegLeaf::new(next_depth, best.right);
        let right_idx = self.leaves.len();
        self.leaves[li] = new_l;
        self.leaves.push(new_r);
        let bid = self.branches.len() as u32;
        self.branches.push(RegBranch {
            feature: best.feature as u32,
            threshold: best.threshold,
            left: NodeRef::Leaf(li as u32),
            right: NodeRef::Leaf(right_idx as u32),
        });
        match parent {
            None => self.root = NodeRef::Branch(bid),
            Some(pi) if went_right => self.branches[pi].right = NodeRef::Branch(bid),
            Some(pi) => self.branches[pi].left = NodeRef::Branch(bid),
        }
        self.enforce_cap(cfg, tree_cap);
        true
    }

    fn has_shape(&self, cfg: &HoeffdingTreeRegressorCfg, tree_cap: u64) -> bool {
        if !(self.target_weight.is_finite() && self.target_weight >= 0.0) {
            return false;
        }
        let n_leaves = self.leaves.len();
        let n_branches = self.branches.len();
        if n_leaves == 0 || n_branches + 1 != n_leaves {
            return false;
        }
        let k = cfg.n_features;
        let max_d = cfg.effective_max_depth();
        let mut seen_branch = vec![false; n_branches];
        let mut seen_leaf = vec![false; n_leaves];
        let mut stack: Vec<(NodeRef, usize)> = Vec::with_capacity(n_leaves);
        stack.push((self.root, 0));
        while let Some((node, depth)) = stack.pop() {
            if depth > max_d {
                return false;
            }
            match node {
                NodeRef::Branch(bi) => {
                    let i = bi as usize;
                    if i >= n_branches || seen_branch[i] {
                        return false;
                    }
                    seen_branch[i] = true;
                    let b = self.branches[i];
                    if (b.feature as usize) >= k || !b.threshold.is_finite() {
                        return false;
                    }
                    stack.push((b.right, depth + 1));
                    stack.push((b.left, depth + 1));
                }
                NodeRef::Leaf(li) => {
                    let i = li as usize;
                    if i >= n_leaves || seen_leaf[i] {
                        return false;
                    }
                    seen_leaf[i] = true;
                    if self.leaves[i].depth as usize != depth {
                        return false;
                    }
                }
            }
        }
        if !seen_branch.iter().all(|&s| s) || !seen_leaf.iter().all(|&s| s) {
            return false;
        }
        for lf in &self.leaves {
            if !lf.stats.is_valid()
                || !(lf.last_split_attempt.is_finite() && lf.last_split_attempt >= 0.0)
                || (lf.active && lf.capped)
            {
                return false;
            }
            if !lf.active && lf.obs.is_some() {
                return false;
            }
            if let Some(obs) = &lf.obs
                && (obs.len() != k || !obs.iter().all(|o| o.is_valid(cfg.max_slots)))
            {
                return false;
            }
        }
        self.reserved_bytes(cfg) <= tree_cap
    }
}

/// Online Hoeffding tree regressor (see module documentation).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HoeffdingTreeRegressor {
    cfg: HoeffdingTreeRegressorCfg,
    trees: Vec<RegTree>,
    /// Accumulated positive weight over all accepted rows: `n_eff`.
    w_sum: f64,
}

impl HoeffdingTreeRegressor {
    pub fn new(cfg: HoeffdingTreeRegressorCfg) -> Result<Self, String> {
        cfg.validate()?;
        let trees = (0..cfg.n_targets).map(|_| RegTree::new()).collect();
        Ok(Self {
            cfg,
            trees,
            w_sum: 0.0,
        })
    }

    pub fn cfg(&self) -> &HoeffdingTreeRegressorCfg {
        &self.cfg
    }

    /// Accumulated weight before the next row (`n_eff`, hard rule 8).
    pub fn n_eff(&self) -> f64 {
        self.w_sum
    }

    /// Accumulated weight of target `j` over the rows it was present on.
    pub fn target_weight(&self, j: usize) -> f64 {
        self.trees[j].target_weight
    }

    /// Total nodes across all target trees.
    pub fn n_nodes(&self) -> usize {
        self.trees
            .iter()
            .map(|t| t.leaves.len() + t.branches.len())
            .sum()
    }

    /// Number of leaves in target `j`'s tree.
    pub fn n_leaves(&self, j: usize) -> usize {
        self.trees[j].leaves.len()
    }

    /// Number of currently active leaves in target `j`'s tree.
    pub fn n_active_leaves(&self, j: usize) -> usize {
        self.trees[j].leaves.iter().filter(|l| l.active).count()
    }

    /// Exact packed state bytes currently used across all targets.
    pub fn state_bytes(&self) -> u64 {
        self.trees.iter().map(RegTree::state_bytes).sum()
    }

    /// Worst-case packed bytes reserved across all targets.
    pub fn reserved_bytes(&self) -> u64 {
        self.trees.iter().map(|t| t.reserved_bytes(&self.cfg)).sum()
    }

    fn read_into(&self, x: &[f64], out: &mut Step) {
        out.pred.clear();
        for tree in &self.trees {
            if tree.target_weight < self.cfg.min_weight || tree.target_weight <= 0.0 {
                out.pred.push(f64::NAN);
            } else {
                let lf = &tree.leaves[tree.leaf_of(x)];
                out.pred.push(if lf.stats.n > 0.0 {
                    lf.stats.mean
                } else {
                    f64::NAN
                });
            }
        }
        out.n_eff = self.w_sum;
        out.extra = None;
    }

    fn has_shape(&self) -> bool {
        if !(self.w_sum.is_finite() && self.w_sum >= 0.0) || self.trees.len() != self.cfg.n_targets
        {
            return false;
        }
        let tree_cap = self.cfg.per_tree_cap();
        self.trees.iter().all(|t| t.has_shape(&self.cfg, tree_cap))
    }
}

impl OnlineModel for HoeffdingTreeRegressor {
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
        // Out-of-sample prediction before any update (hard rule 2).
        self.read_into(x, out);
        // Zero-weight (or subnormal) rows learn nothing and touch no counter (hard rule 9).
        if weight < f64::MIN_POSITIVE {
            return;
        }
        self.w_sum += weight;
        let tree_cap = self.cfg.per_tree_cap();
        for (tree, yj) in self.trees.iter_mut().zip(y.iter()) {
            let Some(yv) = *yj else { continue };
            tree.learn(x, yv, weight, &self.cfg, tree_cap);
        }
    }

    fn predict(&self, x: &[f64], d_clock: f64) -> Step {
        if let Some(refused) = crate::model::refused_predict(self, x, d_clock) {
            return refused;
        }
        let mut out = Step::default();
        self.read_into(x, &mut out);
        out
    }

    fn predict_with_floor(
        &mut self,
        x: &[f64],
        y: &[Option<f64>],
        d_clock: f64,
        floor: f64,
    ) -> Step {
        let saved = std::mem::replace(&mut self.cfg.min_weight, floor);
        let out = OnlineModel::predict_with(self, x, y, d_clock);
        self.cfg.min_weight = saved;
        out
    }

    fn target_n_eff_into(&self, out: &mut Vec<f64>) -> bool {
        out.clear();
        out.extend(self.trees.iter().map(|t| t.target_weight));
        true
    }

    fn state(&self) -> State {
        State::new(ModelState::HoeffdingTreeRegressor(Box::new(self.clone())))
    }

    fn restore(s: &State) -> Result<Self, StateError> {
        check_schema(s)?;
        match &s.model {
            ModelState::HoeffdingTreeRegressor(m) => {
                let m = (**m).clone();
                crate::model::check_cfg("hoeffding_tree_regressor", m.cfg.validate())?;
                if !m.has_shape() {
                    return Err(crate::model::damaged(
                        "hoeffding_tree_regressor",
                        "its trees have the wrong shape",
                    ));
                }
                Ok(m)
            }
            other => Err(StateError::WrongModel {
                expected: "hoeffding_tree_regressor",
                found: other.kind(),
            }),
        }
    }

    fn n_targets(&self) -> usize {
        self.cfg.n_targets
    }

    fn n_features(&self) -> usize {
        self.cfg.n_features
    }
}

#[cfg(test)]
#[path = "hoeffding_tree_regressor/tests.rs"]
mod tests;
