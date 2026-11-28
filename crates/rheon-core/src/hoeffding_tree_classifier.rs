//! `HoeffdingTreeClassifier`: online decision tree classifier (VFDT, Domingos
//! and Hulten 2000) with per-feature Gaussian observers and an exact per-group
//! byte cap.
//!
//! Each leaf maintains class weights `n_c` and, while active, a Gaussian
//! observer `(W, mean, S, min, max)` per `(feature, class)` pair (Welford
//! weighted variance `S / (W - 1)`). Every `grace_period` of learned weight
//! the leaf scores `n_splits` equally spaced thresholds strictly between each
//! feature's observed min and max using the Gaussian CDF to partition each
//! class's weight left and right, skips candidates leaving either branch at or
//! below `min_branch_fraction`, and evaluates `split_criterion` (`info_gain`,
//! `gini` or `hellinger`). The leaf splits on the best feature when:
//!
//! ```text
//! G(best) - G(second) > eps   or   eps < tau
//! eps = sqrt(R^2 * ln(1 / delta) / (2 * n)),   n = sum_c n_c
//! R   = log2(max(observed_classes, 2))   (info_gain, hellinger);  1   (gini)
//! ```
//!
//! With `merit_preprune = true`, the null split ("do not split") enters the
//! candidate list at merit `0.0` (Domingos and Hulten's `X_0`), preventing a
//! single noisy feature from splitting on every attempt. A leaf whose majority
//! share exceeds `max_share_to_split` or where the null split wins is
//! permanently deactivated (it drops its observers and predicts its majority
//! class).
//!
//! **Leaf prediction (`leaf_prediction`).**
//! - `mc`: class proportions `n_c / sum n`.
//! - `nb`: Gaussian naive Bayes `ln(n_c / n) + sum_f ln pdf_c(x_f)` normalized
//!   by softmax once the leaf weight reaches `nb_threshold`.
//! - `nba` (default): predicts with `nb` while its prequential correct weight
//!   is at least `mc`'s, else `mc`. Inactive leaves always predict `mc`.
//!
//! **Per-group memory cap (`max_state_bytes`).** Default 256 KB, ceiling
//! 64 MiB. Packed state bytes are `40 + 8C + 20B + (8C + 27)L + 40 k C A`
//! (`L` leaves, `B = L - 1` branches, `A` active leaves with observers, `k`
//! features, `C` classes). Splits stop once `struct_bytes(L + 1, B + 1)`
//! would pass the cap; after every split and, while any leaf is capped,
//! every `ceil(grace_period) * max(1, active_leaves)` learned rows,
//! eligible leaves are ranked by promise (`sum n_c - max n_c`) and at most
//! `(cap - struct_bytes) / (40 k C)` remain active (reserving observers at
//! full size so packed bytes never pass the cap). Cap-deactivated leaves keep
//! updating their class counts and predicting `mc`, and reactivate with fresh
//! observers if their promise ranks back in.
//!
//! **Portability and contracts.** All transcendentals (`erf`, `log`, `log2`,
//! `exp`) call the pure-Rust `libm` crate, so trees, split thresholds,
//! probabilities and saved states are bit-identical across macOS, Linux and
//! Windows. Weights are frequency weights (`grace_period`, the Hoeffding bound
//! and `W - 1` are in weight units). A zero-weight row predicts out-of-sample
//! and touches no state, observer or counter. No decay and no rolling window.

use serde::{Deserialize, Serialize};

use crate::clock_span::{DEFAULT_STATE_BYTES, check_max_state_bytes};
use crate::model::{
    ModelState, OnlineModel, State, StateError, Step, check_min_weight, check_schema,
};

mod observer;
use observer::{GAUSSIAN_OBS_BYTES, GaussianObs, split_merit};

/// Packed header bytes (fixed fields excluding per-class metadata).
const CLF_HEADER_BYTES: u64 = 40;
/// Packed branch bytes (`feature: u32`, `threshold: f64`, `left: i32`, `right: i32`).
const CLF_BRANCH_BYTES: u64 = 20;
/// Packed fixed leaf bytes (`last`, `mc`, `nb` = 24, `depth: u16` = 2, `flags: u8` = 1).
const CLF_LEAF_FIXED_BYTES: u64 = 27;
/// Default effective depth ceiling when `max_depth` is `None`.
const DEFAULT_MAX_DEPTH: usize = u16::MAX as usize;

/// Split criterion for [`HoeffdingTreeClassifier`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SplitCriterion {
    /// Information gain (entropy reduction in bits).
    #[default]
    InfoGain,
    /// Gini impurity reduction.
    Gini,
    /// Hellinger distance (binary classification only).
    Hellinger,
}

impl SplitCriterion {
    pub fn name(self) -> &'static str {
        match self {
            SplitCriterion::InfoGain => "info_gain",
            SplitCriterion::Gini => "gini",
            SplitCriterion::Hellinger => "hellinger",
        }
    }

    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "info_gain" => Ok(SplitCriterion::InfoGain),
            "gini" => Ok(SplitCriterion::Gini),
            "hellinger" => Ok(SplitCriterion::Hellinger),
            other => Err(format!(
                "split_criterion must be \"info_gain\", \"gini\" or \"hellinger\", but it is \
                 {other:?}. Use one of those words."
            )),
        }
    }
}

/// Leaf prediction strategy for [`HoeffdingTreeClassifier`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LeafPrediction {
    /// Majority class proportions (`n_c / sum n`).
    #[serde(rename = "mc")]
    MajorityClass,
    /// Gaussian naive Bayes over the leaf's feature observers (once weight >= `nb_threshold`).
    #[serde(rename = "nb")]
    NaiveBayes,
    /// Naive Bayes Adaptive: whichever of `nb` and `mc` has higher prequential correct weight.
    #[default]
    #[serde(rename = "nba")]
    NaiveBayesAdaptive,
}

impl LeafPrediction {
    pub fn name(self) -> &'static str {
        match self {
            LeafPrediction::MajorityClass => "mc",
            LeafPrediction::NaiveBayes => "nb",
            LeafPrediction::NaiveBayesAdaptive => "nba",
        }
    }

    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "mc" => Ok(LeafPrediction::MajorityClass),
            "nb" => Ok(LeafPrediction::NaiveBayes),
            "nba" => Ok(LeafPrediction::NaiveBayesAdaptive),
            other => Err(format!(
                "leaf_prediction must be \"mc\", \"nb\" or \"nba\", but it is {other:?}. \
                 Use one of those words."
            )),
        }
    }
}

/// Configuration for [`HoeffdingTreeClassifier`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HoeffdingTreeClassifierCfg {
    /// Number of numeric features (`>= 1`).
    pub n_features: usize,
    /// Number of classes (`>= 2`; must be `2` for [`SplitCriterion::Hellinger`]).
    pub n_classes: usize,
    /// Learned weight at a leaf between split attempts (`> 0`, finite; default `200.0`).
    pub grace_period: f64,
    /// Hoeffding bound error probability in `(0, 1)` (default `1e-7`).
    pub delta: f64,
    /// Tie-breaking threshold `tau >= 0`, finite (default `0.05`).
    pub tau: f64,
    /// Criterion scoring candidate splits (default `InfoGain`).
    pub split_criterion: SplitCriterion,
    /// Leaf predictor (default `NaiveBayesAdaptive`).
    pub leaf_prediction: LeafPrediction,
    /// Minimum leaf weight before `nb` activates (`>= 0`, finite; default `0.0`).
    pub nb_threshold: f64,
    /// Candidate thresholds evaluated per feature (`>= 1`; default `10`).
    pub n_splits: usize,
    /// Minimum weight fraction on each side of a candidate split in `[0, 0.5)` (default `0.01`).
    pub min_branch_fraction: f64,
    /// Deactivate a leaf once its majority class share exceeds this in `(0, 1]` (default `0.99`).
    pub max_share_to_split: f64,
    /// Include the null split (merit `0.0`) as a pre-pruning competitor (default `true`).
    pub merit_preprune: bool,
    /// Optional tree depth cap (`None` = unbounded up to `u16::MAX`).
    pub max_depth: Option<usize>,
    /// Minimum accumulated labeled weight before predicting (`>= 0`; default `0.0`).
    pub min_weight: f64,
    /// Per-group packed state cap in bytes (`None` = 256 KB; max 64 MiB).
    pub max_state_bytes: Option<u64>,
}

impl Default for HoeffdingTreeClassifierCfg {
    fn default() -> Self {
        Self {
            n_features: 1,
            n_classes: 2,
            grace_period: 200.0,
            delta: 1e-7,
            tau: 0.05,
            split_criterion: SplitCriterion::InfoGain,
            leaf_prediction: LeafPrediction::NaiveBayesAdaptive,
            nb_threshold: 0.0,
            n_splits: 10,
            min_branch_fraction: 0.01,
            max_share_to_split: 0.99,
            merit_preprune: true,
            max_depth: None,
            min_weight: 0.0,
            max_state_bytes: None,
        }
    }
}

/// Packed structural bytes of a classifier tree (excluding active observers).
pub fn clf_struct_bytes(n_classes: usize, n_leaves: usize, n_branches: usize) -> u64 {
    let c = n_classes as u64;
    let l = n_leaves as u64;
    let b = n_branches as u64;
    let leaf_bytes = CLF_LEAF_FIXED_BYTES.saturating_add(8u64.saturating_mul(c));
    CLF_HEADER_BYTES
        .saturating_add(8u64.saturating_mul(c))
        .saturating_add(CLF_BRANCH_BYTES.saturating_mul(b))
        .saturating_add(leaf_bytes.saturating_mul(l))
}

/// Packed observer reserve for one active classifier leaf (`40 * k * C`).
pub fn clf_obs_reserve(n_features: usize, n_classes: usize) -> u64 {
    GAUSSIAN_OBS_BYTES
        .saturating_mul(n_features as u64)
        .saturating_mul(n_classes as u64)
}

impl HoeffdingTreeClassifierCfg {
    /// Active per-group cap in bytes (`DEFAULT_STATE_BYTES` when unset).
    pub fn cap(&self) -> u64 {
        self.max_state_bytes.unwrap_or(DEFAULT_STATE_BYTES)
    }

    /// Minimum bytes needed for a single active root leaf with observers.
    pub fn min_state_bytes(&self) -> u64 {
        clf_struct_bytes(self.n_classes, 1, 0)
            .saturating_add(clf_obs_reserve(self.n_features, self.n_classes))
    }

    fn effective_max_depth(&self) -> usize {
        self.max_depth.unwrap_or(DEFAULT_MAX_DEPTH)
    }

    pub fn validate(&self) -> Result<(), String> {
        let k_name = "hoeffding_tree_classifier";
        let mut no = crate::Refusals::new();
        if self.n_features == 0 {
            no.refuse(format!(
                "{k_name}: the model needs at least one feature, but it has none. Name at least \
                 one column in features."
            ));
        }
        if self.n_classes < 2 {
            no.refuse(format!(
                "{k_name}: the model needs at least 2 classes, but it has {}. List 2 or more \
                 classes in classes.",
                self.n_classes
            ));
        }
        if self.split_criterion == SplitCriterion::Hellinger && self.n_classes != 2 {
            no.refuse(format!(
                "{k_name}: split_criterion = \"hellinger\" is defined for 2 classes, but classes \
                 lists {}. Use \"info_gain\" or \"gini\" for multi-class labels.",
                self.n_classes
            ));
        }
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
        if !(self.nb_threshold.is_finite() && self.nb_threshold >= 0.0) {
            no.refuse(format!(
                "{k_name}: nb_threshold must be finite and at least 0, but it is {}. Set \
                 nb_threshold to the leaf weight needed before naive Bayes predicts.",
                self.nb_threshold
            ));
        }
        if self.n_splits == 0 {
            no.refuse(format!(
                "{k_name}: n_splits must be at least 1, but it is 0. Set n_splits to the \
                 number of candidate thresholds per feature."
            ));
        }
        if !(self.min_branch_fraction.is_finite()
            && self.min_branch_fraction >= 0.0
            && self.min_branch_fraction < 0.5)
        {
            no.refuse(format!(
                "{k_name}: min_branch_fraction must be finite, at least 0 and below 0.5, but it \
                 is {}. Set min_branch_fraction to the minimum share of weight each child must \
                 receive.",
                self.min_branch_fraction
            ));
        }
        if !(self.max_share_to_split.is_finite()
            && self.max_share_to_split > 0.0
            && self.max_share_to_split <= 1.0)
        {
            no.refuse(format!(
                "{k_name}: max_share_to_split must be finite, above 0 and at most 1, but it is \
                 {}. Set max_share_to_split to the majority-class share above which a leaf stops \
                 splitting.",
                self.max_share_to_split
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
        if self.n_features >= 1 && self.n_classes >= 2 && self.max_state_bytes != Some(0) {
            let need = self.min_state_bytes();
            let cap = self.cap();
            if cap <= crate::clock_span::STATE_BYTES_CEILING && need > cap {
                no.refuse(format!(
                    "{k_name}: one active leaf with {} features and {} classes needs {need} \
                     bytes, past max_state_bytes ({cap}). Raise max_state_bytes to at least \
                     {need} or use fewer features or classes.",
                    self.n_features, self.n_classes
                ));
            }
        }
        no.finish()
    }
}

/// Reference to either an internal branch or a leaf in a tree's arena.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum NodeRef {
    Branch(u32),
    Leaf(u32),
}

/// Internal split node in [`HoeffdingTreeClassifier`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
struct ClfBranch {
    feature: u32,
    threshold: f64,
    left: NodeRef,
    right: NodeRef,
}

/// Leaf node in [`HoeffdingTreeClassifier`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct ClfLeaf {
    /// Class weight distribution (`len == n_classes`).
    counts: Vec<f64>,
    /// Per-feature, per-class Gaussian observers (`len == n_features * n_classes`),
    /// allocated lazily on the leaf's first learned row while `active`.
    obs: Option<Vec<GaussianObs>>,
    /// Depth of this leaf from the root (`0` at root).
    depth: u16,
    /// Total leaf weight at the last split attempt.
    last_split_attempt: f64,
    /// Cumulative weight where majority-class predicted correctly (`nba`).
    mc_correct: f64,
    /// Cumulative weight where naive Bayes predicted correctly (`nba`).
    nb_correct: f64,
    /// Whether this leaf is currently updating observers and eligible to split.
    active: bool,
    /// Whether this leaf was deactivated by the memory cap (and may reactivate).
    capped: bool,
}

impl ClfLeaf {
    fn new(depth: u16, counts: Vec<f64>) -> Self {
        let last = sum_slice(&counts);
        Self {
            counts,
            obs: None,
            depth,
            last_split_attempt: last,
            mc_correct: 0.0,
            nb_correct: 0.0,
            active: true,
            capped: false,
        }
    }

    #[inline]
    fn total(&self) -> f64 {
        sum_slice(&self.counts)
    }

    /// Promise metric: weight the majority class gets wrong (`total - max(counts)`).
    #[inline]
    fn promise(&self) -> f64 {
        let mut tot = 0.0;
        let mut mx = 0.0;
        for &c in &self.counts {
            tot += c;
            if c > mx {
                mx = c;
            }
        }
        tot - mx
    }
}

#[inline]
fn sum_slice(xs: &[f64]) -> f64 {
    let mut s = 0.0;
    for &x in xs {
        s += x;
    }
    s
}

struct SplitCand {
    merit: f64,
    feature: usize,
    threshold: f64,
    left: Vec<f64>,
    right: Vec<f64>,
}

/// Online Hoeffding tree classifier (see module documentation).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HoeffdingTreeClassifier {
    cfg: HoeffdingTreeClassifierCfg,
    leaves: Vec<ClfLeaf>,
    branches: Vec<ClfBranch>,
    root: NodeRef,
    /// Accumulated positive weight over all accepted rows (labeled or not): `n_eff`.
    w_sum: f64,
    /// Accumulated positive weight over rows with a valid class label.
    labeled_weight: f64,
    /// Count of labeled positive-weight rows learned (drives deterministic cap rechecks).
    rows_learned: u64,
}

impl HoeffdingTreeClassifier {
    pub fn new(cfg: HoeffdingTreeClassifierCfg) -> Result<Self, String> {
        cfg.validate()?;
        let root_leaf = ClfLeaf::new(0, vec![0.0; cfg.n_classes]);
        Ok(Self {
            cfg,
            leaves: vec![root_leaf],
            branches: Vec::new(),
            root: NodeRef::Leaf(0),
            w_sum: 0.0,
            labeled_weight: 0.0,
            rows_learned: 0,
        })
    }

    pub fn cfg(&self) -> &HoeffdingTreeClassifierCfg {
        &self.cfg
    }

    /// Accumulated weight before the next row (`n_eff`, hard rule 8).
    pub fn n_eff(&self) -> f64 {
        self.w_sum
    }

    /// Accumulated weight of labeled rows.
    pub fn labeled_weight(&self) -> f64 {
        self.labeled_weight
    }

    /// Total nodes (`leaves + branches`).
    pub fn n_nodes(&self) -> usize {
        self.leaves.len() + self.branches.len()
    }

    /// Number of leaves in the tree.
    pub fn n_leaves(&self) -> usize {
        self.leaves.len()
    }

    /// Number of currently active leaves.
    pub fn n_active_leaves(&self) -> usize {
        self.leaves.iter().filter(|l| l.active).count()
    }

    /// Exact packed state bytes currently used by the tree.
    pub fn state_bytes(&self) -> u64 {
        let with_obs = self.leaves.iter().filter(|l| l.obs.is_some()).count() as u64;
        clf_struct_bytes(self.cfg.n_classes, self.leaves.len(), self.branches.len())
            + with_obs * clf_obs_reserve(self.cfg.n_features, self.cfg.n_classes)
    }

    /// Worst-case packed bytes reserved for all currently active leaves.
    pub fn reserved_bytes(&self) -> u64 {
        let active = self.n_active_leaves() as u64;
        clf_struct_bytes(self.cfg.n_classes, self.leaves.len(), self.branches.len())
            + active * clf_obs_reserve(self.cfg.n_features, self.cfg.n_classes)
    }

    /// Output names matching `ew_class`: `class`, then `p_<name>` per class.
    pub fn labels(classes: &[String]) -> Vec<String> {
        let mut out = Vec::with_capacity(1 + classes.len());
        out.push("class".to_string());
        out.extend(classes.iter().map(|c| format!("p_{c}")));
        out
    }

    #[inline]
    fn parse_label(&self, y: &[Option<f64>]) -> Option<usize> {
        let v = y.first().copied().flatten()?;
        if v.is_finite() && v >= 0.0 && v.fract() == 0.0 && v < self.cfg.n_classes as f64 {
            Some(v as usize)
        } else {
            None
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

    #[inline]
    fn use_nb(&self, leaf: &ClfLeaf) -> bool {
        if !leaf.active {
            return false;
        }
        match self.cfg.leaf_prediction {
            LeafPrediction::MajorityClass => false,
            LeafPrediction::NaiveBayes => leaf.total() >= self.cfg.nb_threshold,
            LeafPrediction::NaiveBayesAdaptive => leaf.nb_correct >= leaf.mc_correct,
        }
    }

    /// Argmax class under Gaussian naive Bayes without allocating.
    fn nb_class(&self, leaf: &ClfLeaf, x: &[f64]) -> Option<usize> {
        let tot = leaf.total();
        if tot <= 0.0 {
            return None;
        }
        let nc = self.cfg.n_classes;
        let k = self.cfg.n_features;
        let mut best: Option<usize> = None;
        let mut best_v = f64::NEG_INFINITY;
        for c in 0..nc {
            let cw = leaf.counts[c];
            if cw <= 0.0 {
                continue;
            }
            let mut v = libm::log(cw / tot);
            if let Some(obs) = leaf.obs.as_deref() {
                for (f, &xf) in x.iter().enumerate().take(k) {
                    let lp = obs[f * nc + c].log_pdf(xf);
                    if lp.is_finite() {
                        v += lp;
                    }
                }
            }
            if v > best_v {
                best_v = v;
                best = Some(c);
            }
        }
        best
    }

    /// Write `[class, p_0, ..., p_{C-1}]` into `out` without allocating.
    fn score_into(&self, x: &[f64], out: &mut Step) {
        let nc = self.cfg.n_classes;
        out.n_eff = self.w_sum;
        out.extra = None;
        out.pred.clear();
        if self.labeled_weight < self.cfg.min_weight || self.labeled_weight <= 0.0 {
            out.pred.resize(1 + nc, f64::NAN);
            return;
        }
        let leaf = &self.leaves[self.leaf_of(x)];
        let tot = leaf.total();
        if tot <= 0.0 {
            out.pred.resize(1 + nc, f64::NAN);
            return;
        }
        out.pred.resize(1 + nc, 0.0);
        if self.use_nb(leaf) {
            let k = self.cfg.n_features;
            let mut best = 0usize;
            let mut mx = f64::NEG_INFINITY;
            for c in 0..nc {
                let cw = leaf.counts[c];
                if cw <= 0.0 {
                    out.pred[1 + c] = f64::NEG_INFINITY;
                    continue;
                }
                let mut v = libm::log(cw / tot);
                if let Some(obs) = leaf.obs.as_deref() {
                    for (f, &xf) in x.iter().enumerate().take(k) {
                        let lp = obs[f * nc + c].log_pdf(xf);
                        if lp.is_finite() {
                            v += lp;
                        }
                    }
                }
                out.pred[1 + c] = v;
                if v > mx {
                    mx = v;
                    best = c;
                }
            }
            if !mx.is_finite() {
                out.pred.fill(f64::NAN);
                return;
            }
            let mut norm = 0.0;
            for c in 0..nc {
                let v = out.pred[1 + c];
                if v.is_finite() {
                    norm += libm::exp(v - mx);
                }
            }
            if !(norm.is_finite() && norm > 0.0) {
                out.pred.fill(f64::NAN);
                return;
            }
            out.pred[0] = best as f64;
            for c in 0..nc {
                let v = out.pred[1 + c];
                out.pred[1 + c] = if v.is_finite() {
                    libm::exp(v - mx) / norm
                } else {
                    0.0
                };
            }
        } else {
            let mut best = 0usize;
            for c in 1..nc {
                if leaf.counts[c] > leaf.counts[best] {
                    best = c;
                }
            }
            out.pred[0] = best as f64;
            for c in 0..nc {
                out.pred[1 + c] = leaf.counts[c] / tot;
            }
        }
    }

    fn best_candidate(&self, leaf: &ClfLeaf, f: usize) -> Option<SplitCand> {
        let obs_all = leaf.obs.as_deref()?;
        let nc = self.cfg.n_classes;
        let obs = &obs_all[f * nc..(f + 1) * nc];
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for o in obs {
            if o.n > 0.0 {
                if o.min < lo {
                    lo = o.min;
                }
                if o.max > hi {
                    hi = o.max;
                }
            }
        }
        if !lo.is_finite() || !hi.is_finite() || hi <= lo {
            return None;
        }
        let denom = (self.cfg.n_splits + 1) as f64;
        let bin_size = (hi - lo) / denom;
        let mut best: Option<SplitCand> = None;
        let mut left = vec![0.0; nc];
        let mut right = vec![0.0; nc];
        for i in 1..=self.cfg.n_splits {
            let t = if bin_size.is_finite() {
                lo + bin_size * (i as f64)
            } else {
                let r = (i as f64) / denom;
                lo * (1.0 - r) + hi * r
            };
            if !t.is_finite() {
                continue;
            }
            left.fill(0.0);
            right.fill(0.0);
            for (ci, o) in obs.iter().enumerate() {
                let wt = o.n;
                if wt == 0.0 {
                    continue;
                }
                if t < o.min {
                    right[ci] = wt;
                } else if t >= o.max {
                    left[ci] = wt;
                } else {
                    let lw_c = o.cdf(t) * wt;
                    left[ci] = lw_c;
                    right[ci] = wt - lw_c;
                }
            }
            let lw = sum_slice(&left);
            let rw = sum_slice(&right);
            let tot = lw + rw;
            if tot <= 0.0
                || lw / tot <= self.cfg.min_branch_fraction
                || rw / tot <= self.cfg.min_branch_fraction
            {
                continue;
            }
            let merit = split_merit(
                &leaf.counts,
                &left,
                &right,
                lw,
                rw,
                self.cfg.split_criterion,
            );
            if merit.is_finite() && best.as_ref().is_none_or(|b| merit > b.merit) {
                best = Some(SplitCand {
                    merit,
                    feature: f,
                    threshold: t,
                    left: left.clone(),
                    right: right.clone(),
                });
            }
        }
        best
    }

    fn may_grow(&self) -> bool {
        clf_struct_bytes(
            self.cfg.n_classes,
            self.leaves.len() + 1,
            self.branches.len() + 1,
        ) <= self.cfg.cap()
    }

    fn recheck_interval(&self) -> u64 {
        let grace = self.cfg.grace_period.ceil().max(1.0);
        let gp = if grace >= u64::MAX as f64 {
            u64::MAX
        } else {
            grace as u64
        };
        let active = self.leaves.iter().filter(|l| l.active).count().max(1) as u64;
        gp.saturating_mul(active).max(1)
    }

    fn enforce_cap(&mut self) {
        let cap = self.cfg.cap();
        let sb = clf_struct_bytes(self.cfg.n_classes, self.leaves.len(), self.branches.len());
        let reserve = clf_obs_reserve(self.cfg.n_features, self.cfg.n_classes);
        let budget = cap.saturating_sub(sb);
        let max_active = (budget / reserve) as usize;
        let mut elig: Vec<usize> = self
            .leaves
            .iter()
            .enumerate()
            .filter_map(|(i, l)| (l.active || l.capped).then_some(i))
            .collect();
        // Stable sort by promise ascending; ties preserve leaf index order.
        elig.sort_by(|&a, &b| {
            self.leaves[a]
                .promise()
                .total_cmp(&self.leaves[b].promise())
        });
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

    fn attempt_split(&mut self, li: usize, parent: Option<usize>, went_right: bool) -> bool {
        let nc = self.cfg.n_classes;
        let tot = self.leaves[li].total();
        if tot <= 0.0 {
            return false;
        }
        let mut maj = 0.0;
        let mut observed = 0usize;
        for &v in &self.leaves[li].counts {
            if v > maj {
                maj = v;
            }
            if v > 0.0 {
                observed += 1;
            }
        }
        if observed < 2 {
            return false;
        }
        if maj / tot > self.cfg.max_share_to_split {
            let lf = &mut self.leaves[li];
            lf.active = false;
            lf.obs = None;
            return false;
        }
        let mut sugg: Vec<(f64, Option<SplitCand>)> = Vec::with_capacity(self.cfg.n_features + 1);
        if self.cfg.merit_preprune {
            // Domingos & Hulten's null split X_0: merit 0.0.
            sugg.push((0.0, None));
        }
        for f in 0..self.cfg.n_features {
            let has_obs = self.leaves[li]
                .obs
                .as_deref()
                .is_some_and(|obs| obs[f * nc..(f + 1) * nc].iter().any(|o| o.n > 0.0));
            if has_obs {
                match self.best_candidate(&self.leaves[li], f) {
                    Some(c) => sugg.push((c.merit, Some(c))),
                    None => sugg.push((f64::NEG_INFINITY, None)),
                }
            }
        }
        sugg.sort_by(|a, b| a.0.total_cmp(&b.0));
        let should = if sugg.len() < 2 {
            !sugg.is_empty()
        } else {
            let best_m = sugg[sugg.len() - 1].0;
            let second_m = sugg[sugg.len() - 2].0;
            let range = match self.cfg.split_criterion {
                SplitCriterion::Gini => 1.0,
                SplitCriterion::InfoGain | SplitCriterion::Hellinger => {
                    libm::log2(observed.max(2) as f64)
                }
            };
            let eps = ((range * range * libm::log(1.0 / self.cfg.delta)) / (2.0 * tot)).sqrt();
            (best_m - second_m) > eps || eps < self.cfg.tau
        };
        if !should {
            return false;
        }
        let Some(best) = sugg.pop().and_then(|(_, c)| c) else {
            // Null split won: permanently deactivate leaf.
            let lf = &mut self.leaves[li];
            lf.active = false;
            lf.obs = None;
            return false;
        };
        if !self.may_grow() {
            return false;
        }
        let next_depth = self.leaves[li].depth.saturating_add(1);
        let new_l = ClfLeaf::new(next_depth, best.left);
        let new_r = ClfLeaf::new(next_depth, best.right);
        let right_idx = self.leaves.len();
        self.leaves[li] = new_l;
        self.leaves.push(new_r);
        let bid = self.branches.len() as u32;
        self.branches.push(ClfBranch {
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
        self.enforce_cap();
        true
    }

    /// `STATE-3` structural invariant verification for restored states.
    fn has_shape(&self) -> bool {
        if !(self.w_sum.is_finite()
            && self.w_sum >= 0.0
            && self.labeled_weight.is_finite()
            && self.labeled_weight >= 0.0)
        {
            return false;
        }
        let n_leaves = self.leaves.len();
        let n_branches = self.branches.len();
        if n_leaves == 0 || n_branches + 1 != n_leaves {
            return false;
        }
        let k = self.cfg.n_features;
        let nc = self.cfg.n_classes;
        let max_d = self.cfg.effective_max_depth();
        // Verify tree topology: connected, acyclic, every node visited exactly once.
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
        // Validate every leaf's fields and observers.
        for lf in &self.leaves {
            if lf.counts.len() != nc
                || !lf.counts.iter().all(|c| c.is_finite() && *c >= 0.0)
                || !(lf.last_split_attempt.is_finite() && lf.last_split_attempt >= 0.0)
                || !(lf.mc_correct.is_finite() && lf.mc_correct >= 0.0)
                || !(lf.nb_correct.is_finite() && lf.nb_correct >= 0.0)
                || (lf.active && lf.capped)
            {
                return false;
            }
            if !lf.active && lf.obs.is_some() {
                return false;
            }
            if let Some(obs) = &lf.obs
                && (obs.len() != k * nc || !obs.iter().all(GaussianObs::is_valid))
            {
                return false;
            }
        }
        self.reserved_bytes() <= self.cfg.cap()
    }
}

impl OnlineModel for HoeffdingTreeClassifier {
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
        self.score_into(x, out);
        // Zero-weight (or subnormal) rows learn nothing and touch no counter (hard rule 9).
        if weight < f64::MIN_POSITIVE {
            return;
        }
        self.w_sum += weight;
        let Some(c) = self.parse_label(y) else {
            return;
        };
        self.labeled_weight += weight;
        self.rows_learned += 1;
        if self.leaves.iter().any(|l| l.capped)
            && self.rows_learned.is_multiple_of(self.recheck_interval())
        {
            self.enforce_cap();
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
        let nc = self.cfg.n_classes;
        let k = self.cfg.n_features;
        if self.leaves[li].active && self.cfg.leaf_prediction == LeafPrediction::NaiveBayesAdaptive
        {
            let tot = self.leaves[li].total();
            let mut maj = 0usize;
            for cand in 1..nc {
                if self.leaves[li].counts[cand] > self.leaves[li].counts[maj] {
                    maj = cand;
                }
            }
            let nb_pred = self.nb_class(&self.leaves[li], x);
            let leaf = &mut self.leaves[li];
            if tot == 0.0 || maj == c {
                leaf.mc_correct += weight;
            }
            if nb_pred == Some(c) {
                leaf.nb_correct += weight;
            }
        }
        let leaf = &mut self.leaves[li];
        leaf.counts[c] += weight;
        if leaf.active {
            let obs = leaf
                .obs
                .get_or_insert_with(|| vec![GaussianObs::default(); k * nc]);
            for (f, &xf) in x.iter().enumerate().take(k) {
                obs[f * nc + c].update(xf, weight);
            }
        }
        let max_d = self.cfg.effective_max_depth();
        if self.leaves[li].active
            && (self.leaves[li].depth as usize) < max_d
            && (self.leaves[li].total() - self.leaves[li].last_split_attempt)
                >= self.cfg.grace_period
            && !self.attempt_split(li, parent, went_right)
        {
            self.leaves[li].last_split_attempt = self.leaves[li].total();
        }
    }

    fn predict(&self, x: &[f64], d_clock: f64) -> Step {
        if let Some(refused) = crate::model::refused_predict(self, x, d_clock) {
            return refused;
        }
        let mut out = Step::default();
        self.score_into(x, &mut out);
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
        out.push(self.labeled_weight);
        true
    }

    fn state(&self) -> State {
        State::new(ModelState::HoeffdingTreeClassifier(Box::new(self.clone())))
    }

    fn restore(s: &State) -> Result<Self, StateError> {
        check_schema(s)?;
        match &s.model {
            ModelState::HoeffdingTreeClassifier(m) => {
                let m = (**m).clone();
                crate::model::check_cfg("hoeffding_tree_classifier", m.cfg.validate())?;
                if !m.has_shape() {
                    return Err(crate::model::damaged(
                        "hoeffding_tree_classifier",
                        "its tree has the wrong shape",
                    ));
                }
                Ok(m)
            }
            other => Err(StateError::WrongModel {
                expected: "hoeffding_tree_classifier",
                found: other.kind(),
            }),
        }
    }

    fn n_targets(&self) -> usize {
        1
    }

    fn n_features(&self) -> usize {
        self.cfg.n_features
    }

    fn n_outputs(&self) -> usize {
        1 + self.cfg.n_classes
    }
}

#[cfg(test)]
#[path = "hoeffding_tree_classifier/tests.rs"]
mod tests;
