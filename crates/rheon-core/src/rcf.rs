//! Robust Random Cut Forest streaming anomaly detector (Guha, Mishra, Roy and
//! Schrijvers 2016), using the paper's exact leaf-cut rule.
//!
//! ## Paper's cut and strict separation invariant
//!
//! At a node `v` with bounding box `[lo, hi]` (for a leaf, `lo = hi = point`),
//! extending to a new point `p` gives `[lo', hi']` with `span_i = hi'_i - lo'_i`.
//! A dimension `q` is drawn with probability proportional to `span_q` and a cut
//! `c` uniformly in `[lo'_q, hi'_q)` using Philox4x64-10 lane 3 with counter
//! `j = (tree << 32) | block` and `n` the learned-row count before the row:
//!
//! - if `p_q < lo_q` and `c < lo_q`: create a branch `(q, c)` with `p` left and `v` right;
//! - if `p_q > hi_q` and `c >= hi_q`: create a branch `(q, c)` with `v` left and `p` right;
//! - otherwise descend left if `p_{q_v} <= c_v`, else right.
//!
//! Every branch strictly separates its children's points (`left_q <= c < right_q`),
//! so every leaf is reachable by tree descent.
//!
//! ## Scoring, FIFO eviction, and `predict == step` parity
//!
//! The anomaly score of `x` is the forest mean of its collusive displacement:
//!
//! ```text
//! CoDisp(x) = max_{v from leaf(x) up to root's child}  n(sibling(v)) / n(v)
//! ```
//!
//! On `step(w > 0)`, `x` is inserted, scored against all points currently in the
//! tree, and only then is the oldest point evicted when the FIFO ring exceeds
//! `tree_size`. On `predict` and `step(w == 0)`, `CoDisp(x)` is evaluated
//! read-only along the identical draw sequence without mutating any tree, so
//! `predict(x) == step(x)` to the bit on every row and zero-weight rows leave
//! state byte-identical.

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

use crate::model::{ModelState, OnlineModel, State, StateError, Step, check_schema};
use crate::philox::{DrawStream, LANE_RCF_CUT, LaneReader};

/// Default per-group state cap for [`RandomCutForest`]: 9 MiB (accommodates the
/// default `40 x 256` forest up to `d_eff = 32`).
pub const RCF_MAX_STATE_BYTES: u64 = 9 * 1024 * 1024;

const NONE: u32 = u32::MAX;

/// Configuration for [`RandomCutForest`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RandomCutForestCfg {
    pub n_features: usize,
    pub n_trees: usize,
    /// FIFO learned-point capacity per tree (`>= 2`).
    pub tree_size: usize,
    /// Consecutive learned rows stacked into each shingled point (`>= 1`).
    pub shingle_size: usize,
    pub min_weight: f64,
    pub seed: u64,
    #[serde(default)]
    pub group_hash: u64,
    #[serde(default)]
    pub slot: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_state_bytes: Option<u64>,
}

impl RandomCutForestCfg {
    #[inline]
    pub fn d_eff(&self) -> Option<usize> {
        self.n_features.checked_mul(self.shingle_size)
    }

    /// Fixed per-group state bytes at full capacity (`tree_size` points).
    pub fn state_bytes(&self) -> u64 {
        let Some(d) = self.d_eff() else {
            return u64::MAX;
        };
        let d = d as u64;
        let ts = self.tree_size as u64;
        let nt = self.n_trees as u64;
        let branch = 28u64.saturating_add(16u64.saturating_mul(d));
        let leaf = 12u64;
        let per_tree = ts
            .saturating_mul(branch)
            .saturating_add(ts.saturating_add(1).saturating_mul(leaf));
        let ring = ts.saturating_mul(8u64.saturating_mul(d).saturating_add(8));
        let shingle = (self.shingle_size.saturating_sub(1) as u64)
            .saturating_mul(self.n_features as u64)
            .saturating_mul(8);
        nt.saturating_mul(per_tree)
            .saturating_add(ring)
            .saturating_add(shingle)
            .saturating_add(8)
    }

    pub fn cap_bytes(&self) -> u64 {
        self.max_state_bytes.unwrap_or(RCF_MAX_STATE_BYTES)
    }

    pub fn validate(&self) -> Result<(), String> {
        let mut no = crate::Refusals::new();
        if self.n_features == 0 {
            no.refuse(
                "random_cut_forest: the model needs at least one feature, but it has none. Name \
                 at least one column in features.",
            );
        }
        if self.n_trees == 0 {
            no.refuse(
                "random_cut_forest: n_trees must be at least 1, but it is 0. Set n_trees to a \
                 positive integer, such as the default 40.",
            );
        }
        if self.tree_size < 2 {
            no.refuse(format!(
                "random_cut_forest: tree_size must be at least 2, but it is {}. Set tree_size to \
                 the number of points each tree holds, such as the default 256.",
                self.tree_size
            ));
        }
        if self.shingle_size == 0 {
            no.refuse(
                "random_cut_forest: shingle_size must be at least 1, but it is 0. Set \
                 shingle_size to 1 for row-by-row points, or larger to stack consecutive rows.",
            );
        }
        no.take(crate::model::check_min_weight(
            "random_cut_forest",
            self.min_weight,
        ));
        no.take(crate::clock_span::check_max_state_bytes(
            "random_cut_forest",
            self.max_state_bytes,
        ));
        if self.n_features > 0 && self.n_trees > 0 && self.tree_size >= 2 && self.shingle_size > 0 {
            let cap = self.cap_bytes();
            let bytes = self.state_bytes();
            if cap > 0 && cap <= crate::clock_span::STATE_BYTES_CEILING && bytes > cap {
                no.refuse(format!(
                    "random_cut_forest: state holds {bytes} bytes for {} trees of size {} and \
                     width {}, past max_state_bytes of {cap}. Lower n_trees, tree_size or \
                     shingle_size, or raise max_state_bytes.",
                    self.n_trees,
                    self.tree_size,
                    self.d_eff().unwrap_or(usize::MAX)
                ));
            }
        }
        no.finish()
    }
}

/// Largest representable `f64` strictly less than `x` (for finite `x`).
#[inline]
fn prev_f64(x: f64) -> f64 {
    x.next_down()
}

/// Draw a cut `(q, c)` from the extended bounding box `[min(lo, x), max(hi, x)]`.
/// Guarantees `lo2[q] <= c < hi2[q]` whenever `lo2 != hi2`.
#[inline]
fn draw_cut(lo: &[f64], hi: &[f64], x: &[f64], reader: &mut LaneReader) -> (usize, f64) {
    let d = x.len();
    let mut total = 0.0;
    for i in 0..d {
        let lo2 = lo[i].min(x[i]);
        let hi2 = hi[i].max(x[i]);
        total += hi2 - lo2;
    }
    let u = reader.next_u01_open();
    let r = total * u;
    let mut acc = 0.0;
    let mut chosen = None;
    let mut chosen_prev = 0.0;
    let mut last_pos = 0usize;
    let mut last_prev = 0.0;
    for i in 0..d {
        let lo2 = lo[i].min(x[i]);
        let hi2 = hi[i].max(x[i]);
        let span = hi2 - lo2;
        if span > 0.0 {
            let prev = acc;
            acc += span;
            last_pos = i;
            last_prev = prev;
            if chosen.is_none() && r < acc {
                chosen = Some(i);
                chosen_prev = prev;
            }
        }
    }
    let (q, prev) = match chosen {
        Some(i) => (i, chosen_prev),
        None => (last_pos, last_prev),
    };
    let lo2 = lo[q].min(x[q]);
    let hi2 = hi[q].max(x[q]);
    let mut c = lo2 + (r - prev);
    if c < lo2 {
        c = lo2;
    }
    if c >= hi2 {
        c = prev_f64(hi2).max(lo2);
    }
    (q, c)
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
struct Node {
    is_leaf: bool,
    parent: u32,
    left: u32,
    right: u32,
    count: u32,
    /// For branches: cut dimension (`0..d_eff`). For leaves: ring slot (`0..=tree_size`).
    dim_or_slot: u32,
    /// For branches: index into `CutTree::boxes`. Unused (`NONE`) for leaves.
    box_idx: u32,
    /// For branches: cut threshold. Unused (`0.0`) for leaves.
    cut: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct CutTree {
    root: u32,
    nodes: Vec<Node>,
    free_nodes: Vec<u32>,
    /// Flat bounding boxes per branch slot: `[lo_0..lo_{d-1}, hi_0..hi_{d-1}]`.
    boxes: Vec<f64>,
    free_boxes: Vec<u32>,
    /// Maps ring point slot (`0..=tree_size`) to its leaf node index (or `NONE`).
    slot_leaf: Vec<u32>,
}

impl CutTree {
    fn new(tree_size: usize, d_eff: usize) -> Self {
        let max_nodes = 2 * tree_size + 1;
        Self {
            root: NONE,
            nodes: Vec::with_capacity(max_nodes),
            free_nodes: Vec::new(),
            boxes: Vec::with_capacity(tree_size * 2 * d_eff),
            free_boxes: Vec::new(),
            slot_leaf: vec![NONE; tree_size + 1],
        }
    }

    #[inline]
    fn box_slices(&self, box_idx: u32, d_eff: usize) -> (&[f64], &[f64]) {
        let base = (box_idx as usize) * 2 * d_eff;
        (
            &self.boxes[base..base + d_eff],
            &self.boxes[base + d_eff..base + 2 * d_eff],
        )
    }

    #[inline]
    fn node_bounds<'a>(&'a self, v: u32, ring: &'a [f64], d_eff: usize) -> (&'a [f64], &'a [f64]) {
        let n = &self.nodes[v as usize];
        if n.is_leaf {
            let p = ring_point(ring, n.dim_or_slot, d_eff);
            (p, p)
        } else {
            self.box_slices(n.box_idx, d_eff)
        }
    }

    fn alloc_node(&mut self, node: Node) -> u32 {
        if let Some(idx) = self.free_nodes.pop() {
            self.nodes[idx as usize] = node;
            idx
        } else {
            let idx = self.nodes.len() as u32;
            self.nodes.push(node);
            idx
        }
    }

    fn alloc_box(&mut self, d_eff: usize) -> u32 {
        if let Some(idx) = self.free_boxes.pop() {
            idx
        } else {
            let idx = (self.boxes.len() / (2 * d_eff)) as u32;
            self.boxes.resize(self.boxes.len() + 2 * d_eff, 0.0);
            idx
        }
    }

    #[inline]
    fn query_leaf(&self, x: &[f64]) -> u32 {
        let mut v = self.root;
        while !self.nodes[v as usize].is_leaf {
            let n = self.nodes[v as usize];
            v = if x[n.dim_or_slot as usize] <= n.cut {
                n.left
            } else {
                n.right
            };
        }
        v
    }

    fn count_up(&mut self, mut v: u32, delta: i32) {
        while v != NONE {
            let n = &mut self.nodes[v as usize];
            n.count = (n.count as i32 + delta) as u32;
            v = n.parent;
        }
    }

    fn recompute_box(&mut self, v: u32, ring: &[f64], d_eff: usize) {
        let n = self.nodes[v as usize];
        debug_assert!(!n.is_leaf);
        let box_base = (n.box_idx as usize) * 2 * d_eff;
        let a = self.nodes[n.left as usize];
        let b = self.nodes[n.right as usize];
        for i in 0..d_eff {
            let (a_lo, a_hi) = if a.is_leaf {
                let val = ring[(a.dim_or_slot as usize) * d_eff + i];
                (val, val)
            } else {
                let ab = (a.box_idx as usize) * 2 * d_eff;
                (self.boxes[ab + i], self.boxes[ab + d_eff + i])
            };
            let (b_lo, b_hi) = if b.is_leaf {
                let val = ring[(b.dim_or_slot as usize) * d_eff + i];
                (val, val)
            } else {
                let bb = (b.box_idx as usize) * 2 * d_eff;
                (self.boxes[bb + i], self.boxes[bb + d_eff + i])
            };
            self.boxes[box_base + i] = a_lo.min(b_lo);
            self.boxes[box_base + d_eff + i] = a_hi.max(b_hi);
        }
    }

    fn box_up(&mut self, mut v: u32, ring: &[f64], d_eff: usize) {
        while v != NONE {
            self.recompute_box(v, ring, d_eff);
            v = self.nodes[v as usize].parent;
        }
    }

    /// Read-only evaluation of `CoDisp(x)` as if `x` were inserted with
    /// `reader` — bit-identical to `insert` followed by `codisp` + `forget`.
    fn codisp_if_inserted(&self, x: &[f64], ring: &[f64], reader: &mut LaneReader) -> f64 {
        let d_eff = x.len();
        if self.root == NONE {
            return 0.0;
        }
        let near = self.query_leaf(x);
        let near_slot = self.nodes[near as usize].dim_or_slot;
        if ring_point(ring, near_slot, d_eff) == x {
            if near == self.root {
                return 0.0;
            }
            let mut best = 0.0_f64;
            let mut has_best = false;
            let mut v = near;
            while self.nodes[v as usize].parent != NONE {
                let p = self.nodes[v as usize].parent;
                let pn = self.nodes[p as usize];
                let sib = if pn.left == v { pn.right } else { pn.left };
                let ratio = (self.nodes[sib as usize].count as f64)
                    / ((self.nodes[v as usize].count + 1) as f64);
                best = if has_best { best.max(ratio) } else { ratio };
                has_best = true;
                v = p;
            }
            return best;
        }

        let mut v = self.root;
        loop {
            let (lo, hi) = self.node_bounds(v, ring, d_eff);
            let (q, c) = draw_cut(lo, hi, x, reader);
            let separates_left = x[q] < lo[q] && c < lo[q];
            let separates_right = x[q] > hi[q] && c >= hi[q];
            if separates_left || separates_right {
                // New branch above `v` with children `new_leaf (count=1)` and `v`.
                let mut best = self.nodes[v as usize].count as f64;
                let mut cur = v;
                while self.nodes[cur as usize].parent != NONE {
                    let p = self.nodes[cur as usize].parent;
                    let pn = self.nodes[p as usize];
                    let sib = if pn.left == cur { pn.right } else { pn.left };
                    let ratio = (self.nodes[sib as usize].count as f64)
                        / ((self.nodes[cur as usize].count + 1) as f64);
                    best = best.max(ratio);
                    cur = p;
                }
                return best;
            }
            let n = self.nodes[v as usize];
            debug_assert!(!n.is_leaf, "leaf cut must always separate distinct points");
            v = if x[n.dim_or_slot as usize] <= n.cut {
                n.left
            } else {
                n.right
            };
        }
    }

    /// Insert point stored at `ring[slot]` and return its `CoDisp`.
    fn insert_and_codisp(
        &mut self,
        slot: u32,
        ring: &[f64],
        d_eff: usize,
        reader: &mut LaneReader,
    ) -> f64 {
        let x = ring_point(ring, slot, d_eff);
        if self.root == NONE {
            let leaf = self.alloc_node(Node {
                is_leaf: true,
                parent: NONE,
                left: NONE,
                right: NONE,
                count: 1,
                dim_or_slot: slot,
                box_idx: NONE,
                cut: 0.0,
            });
            self.root = leaf;
            self.slot_leaf[slot as usize] = leaf;
            return 0.0;
        }
        let near = self.query_leaf(x);
        let near_slot = self.nodes[near as usize].dim_or_slot;
        if ring_point(ring, near_slot, d_eff) == x {
            self.count_up(near, 1);
            self.slot_leaf[slot as usize] = near;
            return self.codisp_of_leaf(near);
        }

        let mut v = self.root;
        let mut parent = NONE;
        let mut is_right = false;
        loop {
            let (lo, hi) = self.node_bounds(v, ring, d_eff);
            let (q, c) = draw_cut(lo, hi, x, reader);
            let separates_left = x[q] < lo[q] && c < lo[q];
            let separates_right = x[q] > hi[q] && c >= hi[q];
            if separates_left || separates_right {
                let new_leaf = self.alloc_node(Node {
                    is_leaf: true,
                    parent: NONE,
                    left: NONE,
                    right: NONE,
                    count: 1,
                    dim_or_slot: slot,
                    box_idx: NONE,
                    cut: 0.0,
                });
                let box_idx = self.alloc_box(d_eff);
                let (left, right) = if separates_left {
                    (new_leaf, v)
                } else {
                    (v, new_leaf)
                };
                let br = self.alloc_node(Node {
                    is_leaf: false,
                    parent,
                    left,
                    right,
                    count: self.nodes[v as usize].count + 1,
                    dim_or_slot: q as u32,
                    box_idx,
                    cut: c,
                });
                self.nodes[v as usize].parent = br;
                self.nodes[new_leaf as usize].parent = br;
                if parent == NONE {
                    self.root = br;
                } else if is_right {
                    self.nodes[parent as usize].right = br;
                } else {
                    self.nodes[parent as usize].left = br;
                }
                self.count_up(parent, 1);
                self.box_up(br, ring, d_eff);
                self.slot_leaf[slot as usize] = new_leaf;
                return self.codisp_of_leaf(new_leaf);
            }
            let n = self.nodes[v as usize];
            debug_assert!(!n.is_leaf, "leaf cut must always separate distinct points");
            parent = v;
            if x[n.dim_or_slot as usize] <= n.cut {
                v = n.left;
                is_right = false;
            } else {
                v = n.right;
                is_right = true;
            }
        }
    }

    fn codisp_of_leaf(&self, leaf: u32) -> f64 {
        if leaf == self.root {
            return 0.0;
        }
        let mut best = 0.0_f64;
        let mut has_best = false;
        let mut v = leaf;
        while self.nodes[v as usize].parent != NONE {
            let p = self.nodes[v as usize].parent;
            let pn = self.nodes[p as usize];
            let sib = if pn.left == v { pn.right } else { pn.left };
            let ratio =
                (self.nodes[sib as usize].count as f64) / (self.nodes[v as usize].count as f64);
            best = if has_best { best.max(ratio) } else { ratio };
            has_best = true;
            v = p;
        }
        best
    }

    /// Forget the point stored at `ring[slot]`. If the leaf still holds
    /// duplicates (`count > 1`) and referenced `slot`, rebind its `dim_or_slot`
    /// to another live slot mapping to the same leaf (`replacement`).
    fn forget(&mut self, slot: u32, fifo: &VecDeque<u32>, ring: &[f64], d_eff: usize) {
        let leaf = self.slot_leaf[slot as usize];
        self.slot_leaf[slot as usize] = NONE;
        debug_assert!(leaf != NONE);
        if self.nodes[leaf as usize].count > 1 {
            if self.nodes[leaf as usize].dim_or_slot == slot {
                // Find any remaining live FIFO slot pointing to this leaf.
                if let Some(&other) = fifo.iter().find(|&&s| self.slot_leaf[s as usize] == leaf) {
                    self.nodes[leaf as usize].dim_or_slot = other;
                }
            }
            self.count_up(leaf, -1);
            return;
        }
        if leaf == self.root {
            self.root = NONE;
            self.free_nodes.push(leaf);
            return;
        }
        let p = self.nodes[leaf as usize].parent;
        let pn = self.nodes[p as usize];
        let sib = if pn.left == leaf { pn.right } else { pn.left };
        let g = pn.parent;
        self.nodes[sib as usize].parent = g;
        if g == NONE {
            self.root = sib;
        } else if self.nodes[g as usize].left == p {
            self.nodes[g as usize].left = sib;
        } else {
            self.nodes[g as usize].right = sib;
        }
        self.free_boxes.push(pn.box_idx);
        self.free_nodes.push(leaf);
        self.free_nodes.push(p);
        self.count_up(g, -1);
        self.box_up(g, ring, d_eff);
    }

    /// Validate structural integrity of a deserialized tree against `fifo`
    /// (prevents panics on tampered/corrupted state files).
    fn validate(
        &self,
        tree_size: usize,
        d_eff: usize,
        live_slots: &[bool],
    ) -> Option<&'static str> {
        if self.slot_leaf.len() != tree_size + 1 {
            return Some("its slot map has the wrong length");
        }
        if !self.boxes.len().is_multiple_of(2 * d_eff) {
            return Some("its bounding boxes have the wrong shape");
        }
        if !self.boxes.iter().all(|v| v.is_finite()) {
            return Some("its bounding boxes contain a non-finite number");
        }
        let n_nodes = self.nodes.len();
        let n_boxes = self.boxes.len() / (2 * d_eff);
        let mut is_free = vec![false; n_nodes];
        for &f in &self.free_nodes {
            let fi = f as usize;
            if fi >= n_nodes || is_free[fi] {
                return Some("its free node list is invalid");
            }
            is_free[fi] = true;
        }
        let mut box_free = vec![false; n_boxes];
        for &fb in &self.free_boxes {
            let fi = fb as usize;
            if fi >= n_boxes || box_free[fi] {
                return Some("its free box list is invalid");
            }
            box_free[fi] = true;
        }
        let n_live_fifo = live_slots.iter().filter(|&&b| b).count();
        if n_live_fifo == 0 {
            if self.root != NONE {
                return Some("its root is set while the FIFO ring is empty");
            }
            if self.slot_leaf.iter().any(|&v| v != NONE) {
                return Some("its slot map is non-empty while the FIFO ring is empty");
            }
            return None;
        }
        if (self.root as usize) >= n_nodes || is_free[self.root as usize] {
            return Some("its root index is invalid");
        }
        if self.nodes[self.root as usize].parent != NONE {
            return Some("its root has a parent");
        }
        // DFS check: acyclic, valid links, counts, cut dims, boxes.
        let mut visited = vec![false; n_nodes];
        let mut stack = vec![self.root];
        let mut used_boxes = vec![false; n_boxes];
        while let Some(v) = stack.pop() {
            let vi = v as usize;
            if vi >= n_nodes || is_free[vi] || visited[vi] {
                return Some("its tree links are cyclic or invalid");
            }
            visited[vi] = true;
            let n = self.nodes[vi];
            if n.count == 0 {
                return Some("a node has count 0");
            }
            if n.is_leaf {
                let s = n.dim_or_slot as usize;
                if s >= live_slots.len() || !live_slots[s] {
                    return Some("a leaf references an inactive ring slot");
                }
            } else {
                if (n.dim_or_slot as usize) >= d_eff || !n.cut.is_finite() {
                    return Some("a branch cut is out of range or non-finite");
                }
                let bi = n.box_idx as usize;
                if bi >= n_boxes || box_free[bi] || used_boxes[bi] {
                    return Some("a branch box index is invalid");
                }
                used_boxes[bi] = true;
                let li = n.left as usize;
                let ri = n.right as usize;
                if li >= n_nodes || ri >= n_nodes || li == ri || is_free[li] || is_free[ri] {
                    return Some("a branch child index is invalid");
                }
                if self.nodes[li].parent != v || self.nodes[ri].parent != v {
                    return Some("a child's parent pointer disagrees with its branch");
                }
                if self.nodes[li].count.saturating_add(self.nodes[ri].count) != n.count {
                    return Some("a branch count disagrees with its children");
                }
                stack.push(n.left);
                stack.push(n.right);
            }
        }
        if self.nodes[self.root as usize].count as usize != n_live_fifo {
            return Some("its root count disagrees with the FIFO ring");
        }
        for (s, &leaf) in self.slot_leaf.iter().enumerate() {
            if live_slots[s] {
                let li = leaf as usize;
                if li >= n_nodes || !visited[li] || !self.nodes[li].is_leaf {
                    return Some("a live ring slot does not map to a reachable leaf");
                }
            } else if leaf != NONE {
                return Some("an inactive ring slot maps to a leaf");
            }
        }
        None
    }
}

#[inline]
fn ring_point(ring: &[f64], slot: u32, d_eff: usize) -> &[f64] {
    let base = (slot as usize) * d_eff;
    &ring[base..base + d_eff]
}

/// Reusable per-row scratch buffer for the shingled point. Not serialized, and
/// ignored by `PartialEq` and `Debug` so zero-weight rows and restores compare equal.
#[derive(Clone, Default)]
struct RcfWork(Vec<f64>);

impl std::fmt::Debug for RcfWork {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("RcfWork")
    }
}

impl PartialEq for RcfWork {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}

/// Robust Random Cut Forest streaming anomaly detector (paper's cut).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RandomCutForest {
    cfg: RandomCutForestCfg,
    trees: Vec<CutTree>,
    /// Flat storage of `tree_size + 1` point slots, each of length `d_eff`.
    ring: Vec<f64>,
    /// FIFO queue of active point slot indices (length `<= tree_size`).
    fifo: VecDeque<u32>,
    /// Free point slot indices in `0..=tree_size`.
    free_slots: Vec<u32>,
    /// Lag buffer of the last `shingle_size - 1` learned feature rows (cleared
    /// by `clear_lags()` on session boundaries and capped clock gaps).
    shingle: VecDeque<Vec<f64>>,
    /// Total positive-weight shingled points inserted into the forest.
    n_learned: u64,
    /// Accumulated weight (`n_eff`).
    w_sum: f64,
    #[serde(skip)]
    point_buf: RcfWork,
}

impl RandomCutForest {
    pub fn new(cfg: RandomCutForestCfg) -> Result<Self, String> {
        cfg.validate()?;
        let d_eff = cfg.d_eff().ok_or_else(|| {
            "random_cut_forest: n_features * shingle_size overflows usize.".to_string()
        })?;
        let trees = (0..cfg.n_trees)
            .map(|_| CutTree::new(cfg.tree_size, d_eff))
            .collect();
        let ring = vec![0.0; (cfg.tree_size + 1) * d_eff];
        let free_slots: Vec<u32> = (0..=(cfg.tree_size as u32)).rev().collect();
        Ok(Self {
            trees,
            ring,
            fifo: VecDeque::with_capacity(cfg.tree_size + 1),
            free_slots,
            shingle: VecDeque::with_capacity(cfg.shingle_size.saturating_sub(1)),
            n_learned: 0,
            w_sum: 0.0,
            point_buf: RcfWork(vec![0.0; d_eff]),
            cfg,
        })
    }

    pub fn cfg(&self) -> &RandomCutForestCfg {
        &self.cfg
    }

    /// True while `shingle_size > 1` has fewer than `shingle_size - 1` prior
    /// consecutive rows buffered (`withheld_reason = "below_warm_rows"`).
    #[inline]
    pub fn warming(&self) -> bool {
        self.shingle.len() + 1 < self.cfg.shingle_size
    }

    pub fn n_eff(&self) -> f64 {
        self.w_sum
    }

    pub fn n_learned(&self) -> u64 {
        self.n_learned
    }

    #[inline]
    fn draw_stream(&self) -> DrawStream {
        DrawStream::new(self.cfg.seed, self.cfg.group_hash, self.cfg.slot)
    }

    fn fill_shingle(&self, x: &[f64], buf: &mut [f64]) {
        let d = self.cfg.n_features;
        let mut at = 0usize;
        for past in &self.shingle {
            buf[at..at + d].copy_from_slice(past);
            at += d;
        }
        buf[at..at + d].copy_from_slice(x);
    }

    fn score_readonly(&self, point: &[f64]) -> f64 {
        if self.w_sum < self.cfg.min_weight {
            return f64::NAN;
        }
        let stream = self.draw_stream();
        let nt = self.cfg.n_trees;
        let mut sum = 0.0;
        for (t, tree) in self.trees.iter().enumerate() {
            let mut reader = stream.reader(LANE_RCF_CUT, self.n_learned, (t as u64) << 32);
            sum += tree.codisp_if_inserted(point, &self.ring, &mut reader);
        }
        sum / (nt as f64)
    }

    fn push_shingle_row(&mut self, x: &[f64]) {
        if self.cfg.shingle_size <= 1 {
            return;
        }
        let need = self.cfg.shingle_size - 1;
        if self.shingle.len() == need {
            if let Some(mut old) = self.shingle.pop_front() {
                old.copy_from_slice(x);
                self.shingle.push_back(old);
            }
        } else {
            self.shingle.push_back(x.to_vec());
        }
    }
}

impl OnlineModel for RandomCutForest {
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
        let d_eff = self.cfg.n_features * self.cfg.shingle_size;
        if self.point_buf.0.len() != d_eff {
            self.point_buf.0.resize(d_eff, 0.0);
        }
        out.pred.clear();
        out.n_eff = self.w_sum;
        out.extra = None;

        if self.warming() {
            out.pred.push(f64::NAN);
            if weight > 0.0 {
                self.w_sum += weight;
                self.push_shingle_row(x);
            }
            return;
        }

        if weight <= 0.0 {
            let mut buf = std::mem::take(&mut self.point_buf.0);
            self.fill_shingle(x, &mut buf);
            let score = self.score_readonly(&buf);
            self.point_buf.0 = buf;
            out.pred.push(score);
            return;
        }

        // Evict-after-score order: insert point into a free slot, compute mean
        // CoDisp across trees (identical to read-only predict), push to FIFO,
        // then evict the oldest point if the FIFO now exceeds tree_size.
        let slot = self.free_slots.pop().expect("ring has tree_size + 1 slots");
        let base = (slot as usize) * d_eff;
        let mut buf = std::mem::take(&mut self.point_buf.0);
        self.fill_shingle(x, &mut buf);
        self.ring[base..base + d_eff].copy_from_slice(&buf);
        self.point_buf.0 = buf;

        let stream = self.draw_stream();
        let nt = self.cfg.n_trees;
        let mut sum = 0.0;
        for (t, tree) in self.trees.iter_mut().enumerate() {
            let mut reader = stream.reader(LANE_RCF_CUT, self.n_learned, (t as u64) << 32);
            sum += tree.insert_and_codisp(slot, &self.ring, d_eff, &mut reader);
        }
        let score = if self.w_sum < self.cfg.min_weight {
            f64::NAN
        } else {
            sum / (nt as f64)
        };
        out.pred.push(score);

        self.fifo.push_back(slot);
        if self.fifo.len() > self.cfg.tree_size {
            let old = self.fifo.pop_front().expect("fifo non-empty");
            for tree in &mut self.trees {
                tree.forget(old, &self.fifo, &self.ring, d_eff);
            }
            self.free_slots.push(old);
        }
        self.w_sum += weight;
        self.n_learned += 1;
        self.push_shingle_row(x);
    }

    fn predict(&self, x: &[f64], d_clock: f64) -> Step {
        if let Some(refused) = crate::model::refused_predict(self, x, d_clock) {
            return refused;
        }
        if self.warming() {
            return Step {
                pred: vec![f64::NAN],
                n_eff: self.w_sum,
                extra: None,
            };
        }
        let d_eff = self.cfg.n_features * self.cfg.shingle_size;
        let mut buf = vec![0.0; d_eff];
        self.fill_shingle(x, &mut buf);
        Step {
            pred: vec![self.score_readonly(&buf)],
            n_eff: self.w_sum,
            extra: None,
        }
    }

    fn clear_lags(&mut self) {
        self.shingle.clear();
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
        State::new(ModelState::RandomCutForest(Box::new(self.clone())))
    }

    fn restore(state: &State) -> Result<Self, StateError> {
        check_schema(state)?;
        let mut m = match &state.model {
            ModelState::RandomCutForest(r) => (**r).clone(),
            other => {
                return Err(StateError::WrongModel {
                    expected: "random_cut_forest",
                    found: other.kind(),
                });
            }
        };
        crate::model::check_cfg("random_cut_forest", m.cfg.validate())?;
        let d_eff = m
            .cfg
            .d_eff()
            .ok_or_else(|| crate::model::damaged("random_cut_forest", "its dimensions overflow"))?;
        if !(m.w_sum.is_finite() && m.w_sum >= 0.0) {
            return Err(crate::model::damaged(
                "random_cut_forest",
                "its accumulated weight is not a non-negative finite number",
            ));
        }
        if m.trees.len() != m.cfg.n_trees {
            return Err(crate::model::damaged(
                "random_cut_forest",
                "its forest has the wrong number of trees",
            ));
        }
        if m.ring.len() != (m.cfg.tree_size + 1) * d_eff {
            return Err(crate::model::damaged(
                "random_cut_forest",
                "its point ring has the wrong shape",
            ));
        }
        if m.fifo.len() > m.cfg.tree_size {
            return Err(crate::model::damaged(
                "random_cut_forest",
                "its FIFO queue exceeds tree_size",
            ));
        }
        if m.shingle.len() >= m.cfg.shingle_size {
            return Err(crate::model::damaged(
                "random_cut_forest",
                "its shingle buffer is too long",
            ));
        }
        for row in &m.shingle {
            if row.len() != m.cfg.n_features || !row.iter().all(|v| v.is_finite()) {
                return Err(crate::model::damaged(
                    "random_cut_forest",
                    "its shingle buffer has the wrong shape or a non-finite entry",
                ));
            }
        }
        let mut live = vec![false; m.cfg.tree_size + 1];
        for &s in &m.fifo {
            let si = s as usize;
            if si >= live.len() || live[si] {
                return Err(crate::model::damaged(
                    "random_cut_forest",
                    "its FIFO ring contains an out-of-range or duplicate slot",
                ));
            }
            live[si] = true;
            if !ring_point(&m.ring, s, d_eff).iter().all(|v| v.is_finite()) {
                return Err(crate::model::damaged(
                    "random_cut_forest",
                    "its point ring contains a non-finite value",
                ));
            }
        }
        let mut free_seen = vec![false; m.cfg.tree_size + 1];
        for &s in &m.free_slots {
            let si = s as usize;
            if si >= free_seen.len() || live[si] || free_seen[si] {
                return Err(crate::model::damaged(
                    "random_cut_forest",
                    "its free slot list is inconsistent with its FIFO ring",
                ));
            }
            free_seen[si] = true;
        }
        if m.fifo.len() + m.free_slots.len() != m.cfg.tree_size + 1 {
            return Err(crate::model::damaged(
                "random_cut_forest",
                "its slot accounting does not match tree_size + 1",
            ));
        }
        for tree in &m.trees {
            if let Some(reason) = tree.validate(m.cfg.tree_size, d_eff, &live) {
                return Err(crate::model::damaged("random_cut_forest", reason));
            }
        }
        m.point_buf = RcfWork(vec![0.0; d_eff]);
        Ok(m)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(shingle_size: usize) -> RandomCutForestCfg {
        RandomCutForestCfg {
            n_features: 2,
            n_trees: 8,
            tree_size: 32,
            shingle_size,
            min_weight: 0.0,
            seed: 3,
            group_hash: 0,
            slot: 0,
            max_state_bytes: None,
        }
    }

    /// Verify every paper invariant on every tree in the forest:
    /// 1. Strict separation at every branch (`left_q <= cut < right_q`).
    /// 2. Every live leaf is reachable via `query_leaf`.
    /// 3. Bounding boxes equal exact min/max of descendant leaves.
    /// 4. Node counts equal exact sum of leaf counts.
    fn assert_paper_invariants(rcf: &RandomCutForest) {
        let d_eff = rcf.cfg.n_features * rcf.cfg.shingle_size;
        for (ti, tree) in rcf.trees.iter().enumerate() {
            if rcf.fifo.is_empty() {
                assert_eq!(tree.root, NONE);
                continue;
            }
            // Collect leaves per subtree and verify boxes, counts, separation.
            fn check_subtree(
                tree: &CutTree,
                ring: &[f64],
                d_eff: usize,
                v: u32,
            ) -> (Vec<f64>, Vec<f64>, u32, Vec<u32>) {
                let n = tree.nodes[v as usize];
                if n.is_leaf {
                    let p = ring_point(ring, n.dim_or_slot, d_eff);
                    return (p.to_vec(), p.to_vec(), n.count, vec![v]);
                }
                let (l_lo, l_hi, l_cnt, mut l_leaves) = check_subtree(tree, ring, d_eff, n.left);
                let (r_lo, r_hi, r_cnt, r_leaves) = check_subtree(tree, ring, d_eff, n.right);
                let q = n.dim_or_slot as usize;
                assert!(
                    l_hi[q] <= n.cut,
                    "left violation: l_hi[{q}]={} > cut={}",
                    l_hi[q],
                    n.cut
                );
                assert!(
                    r_lo[q] > n.cut,
                    "right violation: r_lo[{q}]={} <= cut={}",
                    r_lo[q],
                    n.cut
                );
                assert_eq!(n.count, l_cnt + r_cnt);
                let lo: Vec<f64> = l_lo.iter().zip(&r_lo).map(|(a, b)| a.min(*b)).collect();
                let hi: Vec<f64> = l_hi.iter().zip(&r_hi).map(|(a, b)| a.max(*b)).collect();
                let (box_lo, box_hi) = tree.box_slices(n.box_idx, d_eff);
                assert_eq!(box_lo, lo.as_slice());
                assert_eq!(box_hi, hi.as_slice());
                l_leaves.extend(r_leaves);
                (lo, hi, n.count, l_leaves)
            }
            let (_, _, total, leaves) = check_subtree(tree, &rcf.ring, d_eff, tree.root);
            assert_eq!(total as usize, rcf.fifo.len(), "tree {ti} count mismatch");
            for leaf in leaves {
                let slot = tree.nodes[leaf as usize].dim_or_slot;
                let p = ring_point(&rcf.ring, slot, d_eff);
                assert_eq!(
                    tree.query_leaf(p),
                    leaf,
                    "tree {ti} leaf {leaf} unreachable by query"
                );
            }
        }
    }

    #[test]
    fn paper_cut_invariants_and_predict_step_parity_hold_through_evictions_and_duplicates() {
        let mut rcf = RandomCutForest::new(cfg(1)).unwrap();
        for i in 0..120 {
            // Mix distinct points, exact duplicates, and zero-weight rows.
            let x = if i == 90 {
                [50.0, -50.0] // sharp anomaly
            } else if i % 15 == 0 {
                [0.25, -0.25]
            } else {
                [(i as f64 * 0.21).sin(), (i as f64 * 0.37).cos()]
            };
            let pred = rcf.predict(&x, 1.0);
            let before = rcf.state();
            let s0 = rcf.step(&x, &[], 1.0, 0.0);
            assert_eq!(s0, pred);
            assert_eq!(rcf.state(), before);

            let step = rcf.step(&x, &[], 1.0, 1.0);
            assert_eq!(step.pred[0].to_bits(), pred.pred[0].to_bits(), "row {i}");
            assert_eq!(step.n_eff.to_bits(), pred.n_eff.to_bits(), "row {i}");
            if i % 10 == 0 || i == 90 {
                assert_paper_invariants(&rcf);
            }
            if i == 90 {
                assert!(
                    step.pred[0] > 10.0,
                    "injected spike should get high CoDisp, got {}",
                    step.pred[0]
                );
            }
        }
        assert_paper_invariants(&rcf);
    }

    #[test]
    fn shingling_clears_lags_without_touching_trees() {
        let mut rcf = RandomCutForest::new(cfg(3)).unwrap();
        assert!(rcf.warming());
        assert!(rcf.step(&[1.0, 2.0], &[], 1.0, 1.0).pred[0].is_nan());
        assert!(rcf.step(&[1.1, 2.1], &[], 1.0, 1.0).pred[0].is_nan());
        assert!(!rcf.warming());
        let s2 = rcf.step(&[1.2, 2.2], &[], 1.0, 1.0);
        assert!(s2.pred[0].is_finite());
        let trees_before = rcf.trees.clone();
        rcf.clear_lags();
        assert!(rcf.warming());
        assert_eq!(rcf.trees, trees_before);
        assert!(rcf.predict(&[1.3, 2.3], 1.0).pred[0].is_nan());
    }

    #[test]
    fn a_state_of_the_wrong_shape_is_refused() {
        let mut rcf = RandomCutForest::new(cfg(2)).unwrap();
        for i in 0..10 {
            rcf.step(&[i as f64 * 0.1, -(i as f64) * 0.1], &[], 1.0, 1.0);
        }
        let mut bad = rcf.clone();
        bad.trees.pop();
        assert!(RandomCutForest::restore(&bad.state()).is_err());

        let mut bad2 = rcf.clone();
        bad2.trees[0].nodes[0].count = 0;
        assert!(RandomCutForest::restore(&bad2.state()).is_err());

        let mut bad3 = rcf.clone();
        bad3.fifo.push_back(999);
        assert!(RandomCutForest::restore(&bad3.state()).is_err());
    }
}
