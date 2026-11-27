//! Exact windowed quantiles (`rolling_quantile`) and ranks (`rolling_rank`):
//! an order-statistic AVL tree beside the window core's two-stack queue.
//!
//! **Complexity and determinism.** Every insert, remove, k-th selection and
//! rank query takes `O(log U)` time where `U <= R` is the number of distinct
//! values in the window, with zero randomness and zero `unsafe`. Values are
//! ordered by `f64::total_cmp` so `-0.0` precedes `+0.0` in quantile reads
//! (matching Polars' `rolling_quantile_by`), while rank queries treat `-0.0`
//! and `+0.0` as tied (matching Polars' `rolling_rank_by`).
//!
//! **Quantile interpolations.** Over `n` sorted values `v[0] <= ... <= v[n-1]`
//! at level `q in [0, 1]` with `h = (n - 1) * q`:
//! - `nearest`: `v[round(h)]` (halves round up)
//! - `lower`: `v[floor(h)]`
//! - `higher`: `v[ceil(h)]`
//! - `midpoint`: `(v[floor(h)] + v[ceil(h)]) / 2`
//! - `linear`: `(h - floor(h)) * (v[ceil(h)] - v[floor(h)]) + v[floor(h)]`
//! - `equiprobable`: `v[max(ceil(n * q) - 1, 0)]`
//!
//! **Rank methods.** For a value `x` with `l` window values strictly below `x`,
//! `e` window values equal to `x`, and `d` distinct window values strictly
//! below `x`:
//! - `average`: `l + (e + 1) / 2`
//! - `min`: `l + 1`
//! - `max`: `l + e`
//! - `dense`: `d + 1`
//!
//! **State serialization.** `OrdSide` serializes as the sorted `(value, count)`
//! pairs and rebuilds a minimum-height balanced tree in `O(U)` on load, so
//! saved bytes depend only on the window's multiset and never on rotation
//! history.

use std::cmp::Ordering;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::extrema::STATE_BYTES_CEILING;
use super::{Direction, Interp, KernelDef, OpDef, RankMethod, Stat};

/// Approximate per-row heap footprint of one exact-order operator's AVL node
/// (worst case, all values distinct): 40-byte node + 8-byte allocator header.
pub const ORD_ROW_BYTES: usize = 48;

/// Validate an exact-order operator (`rolling_quantile`, `rolling_rank`) and
/// forbid its parameters on other operators.
pub(super) fn check_op(o: usize, op: &OpDef, k: &KernelDef) -> Result<(), String> {
    if !op.stat.is_exact_order() {
        if op.interpolation.is_some() {
            return Err(format!(
                "operator {o}: interpolation is given, but only rolling_quantile takes it. \
                 Leave out interpolation."
            ));
        }
        if op.rank_method.is_some() {
            return Err(format!(
                "operator {o}: method is given, but only rolling_rank takes it. \
                 Leave out method."
            ));
        }
        return Ok(());
    }
    if !k.half_life.is_infinite() {
        return Err(format!(
            "operator {o}: an exact window quantile or rank weighs no row, so its kernel's \
             half-life must be inf. Give it a kernel of its own with half_life = inf."
        ));
    }
    if k.window_size.is_none() {
        return Err(format!(
            "operator {o}: window_size is required for rolling_quantile and rolling_rank so \
             the window's state stays bounded. Set window_size to a finite positive number of \
             clock units or a duration."
        ));
    }
    if let Some(c) = op.max_state_bytes {
        if c == 0 {
            return Err(format!(
                "operator {o}: max_state_bytes must be above 0, but it is 0. Give the bytes a \
                 group's window may hold, or leave it out for 256 KB."
            ));
        }
        if c > STATE_BYTES_CEILING {
            return Err(format!(
                "operator {o}: max_state_bytes may be at most {STATE_BYTES_CEILING} (64 MiB) a \
                 group, but it is {c}. Lower it: a group's window is held in memory beside every \
                 other group's."
            ));
        }
    }
    match op.stat {
        Stat::ExactQuantile => {
            if op.rank_method.is_some() {
                return Err(format!(
                    "operator {o}: method is given on rolling_quantile, which takes interpolation. \
                     Leave out method."
                ));
            }
            match op.quantile {
                Some(p) if (0.0..=1.0).contains(&p) => Ok(()),
                p => Err(format!(
                    "operator {o}: quantile must be from 0 to 1, but it is {}. Give a quantile \
                     between 0 and 1.",
                    p.map_or_else(|| "not set".to_string(), |p| p.to_string())
                )),
            }
        }
        Stat::ExactRank => {
            if op.quantile.is_some() {
                return Err(format!(
                    "operator {o}: quantile is given on rolling_rank. Leave out quantile."
                ));
            }
            if op.interpolation.is_some() {
                return Err(format!(
                    "operator {o}: interpolation is given on rolling_rank. Leave out interpolation."
                ));
            }
            if op.rank_method == Some(RankMethod::Ordinal) {
                return Err(format!(
                    "operator {o}: rolling_rank method must be \"average\", \"min\", \"max\" or \
                     \"dense\", but it is \"ordinal\". Set method to one of those four words."
                ));
            }
            if !k.closed.near(k.direction) {
                let allowed = match k.direction {
                    Direction::Backward => "\"right\" or \"both\"",
                    Direction::Forward => "\"left\" or \"both\"",
                };
                return Err(format!(
                    "operator {o}: rolling_rank requires the row's own stamp to be inside the \
                     window, so closed must be {allowed}, but it is {:?}. Set closed to {allowed}.",
                    k.closed.name()
                ));
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

#[derive(Debug, Clone)]
struct Node {
    key: f64,
    count: u32,
    height: u8,
    sub_total: u32,
    sub_distinct: u32,
    left: Option<Box<Node>>,
    right: Option<Box<Node>>,
}

impl Node {
    #[inline]
    fn new(key: f64, count: u32) -> Box<Self> {
        Box::new(Self {
            key,
            count,
            height: 1,
            sub_total: count,
            sub_distinct: 1,
            left: None,
            right: None,
        })
    }

    #[inline]
    fn height_of(n: &Option<Box<Node>>) -> u8 {
        n.as_ref().map_or(0, |n| n.height)
    }

    #[inline]
    fn total_of(n: &Option<Box<Node>>) -> u32 {
        n.as_ref().map_or(0, |n| n.sub_total)
    }

    #[inline]
    fn distinct_of(n: &Option<Box<Node>>) -> u32 {
        n.as_ref().map_or(0, |n| n.sub_distinct)
    }

    #[inline]
    fn refresh(&mut self) {
        self.height = 1 + Self::height_of(&self.left).max(Self::height_of(&self.right));
        self.sub_total = self.count + Self::total_of(&self.left) + Self::total_of(&self.right);
        self.sub_distinct = 1 + Self::distinct_of(&self.left) + Self::distinct_of(&self.right);
    }

    #[inline]
    fn balance_factor(&self) -> i32 {
        i32::from(Self::height_of(&self.left)) - i32::from(Self::height_of(&self.right))
    }

    fn rotate_right(mut y: Box<Self>) -> Box<Self> {
        let mut x = y.left.take().expect("rotate_right needs left child");
        y.left = x.right.take();
        y.refresh();
        x.right = Some(y);
        x.refresh();
        x
    }

    fn rotate_left(mut x: Box<Self>) -> Box<Self> {
        let mut y = x.right.take().expect("rotate_left needs right child");
        x.right = y.left.take();
        x.refresh();
        y.left = Some(x);
        y.refresh();
        y
    }

    fn rebalance(mut n: Box<Self>) -> Box<Self> {
        n.refresh();
        let bf = n.balance_factor();
        if bf > 1 {
            if n.left.as_ref().is_some_and(|l| l.balance_factor() < 0) {
                n.left = Some(Self::rotate_left(n.left.take().expect("left child")));
            }
            return Self::rotate_right(n);
        }
        if bf < -1 {
            if n.right.as_ref().is_some_and(|r| r.balance_factor() > 0) {
                n.right = Some(Self::rotate_right(n.right.take().expect("right child")));
            }
            return Self::rotate_left(n);
        }
        n
    }

    fn insert(node: Option<Box<Self>>, key: f64) -> Box<Self> {
        let Some(mut n) = node else {
            return Self::new(key, 1);
        };
        match key.total_cmp(&n.key) {
            Ordering::Less => n.left = Some(Self::insert(n.left.take(), key)),
            Ordering::Greater => n.right = Some(Self::insert(n.right.take(), key)),
            Ordering::Equal => {
                n.count += 1;
                n.sub_total += 1;
                return n;
            }
        }
        Self::rebalance(n)
    }

    fn take_min(mut n: Box<Self>) -> (f64, u32, Option<Box<Self>>) {
        match n.left.take() {
            Some(left) => {
                let (k, c, new_left) = Self::take_min(left);
                n.left = new_left;
                (k, c, Some(Self::rebalance(n)))
            }
            None => (n.key, n.count, n.right.take()),
        }
    }

    fn remove(node: Option<Box<Self>>, key: f64) -> Option<Box<Self>> {
        let mut n = node?;
        match key.total_cmp(&n.key) {
            Ordering::Less => n.left = Self::remove(n.left.take(), key),
            Ordering::Greater => n.right = Self::remove(n.right.take(), key),
            Ordering::Equal => {
                if n.count > 1 {
                    n.count -= 1;
                    n.sub_total -= 1;
                    return Some(n);
                }
                return match (n.left.take(), n.right.take()) {
                    (None, r) => r,
                    (l, None) => l,
                    (Some(l), Some(r)) => {
                        let (sk, sc, new_r) = Self::take_min(r);
                        n.key = sk;
                        n.count = sc;
                        n.left = Some(l);
                        n.right = new_r;
                        Some(Self::rebalance(n))
                    }
                };
            }
        }
        Some(Self::rebalance(n))
    }

    fn kth(&self, mut idx: u32) -> f64 {
        let mut cur = self;
        loop {
            let left_n = Self::total_of(&cur.left);
            if idx < left_n {
                cur = cur.left.as_deref().expect("idx < left_n");
            } else if idx < left_n + cur.count {
                return cur.key;
            } else {
                idx -= left_n + cur.count;
                cur = cur.right.as_deref().expect("idx within right subtree");
            }
        }
    }

    /// Count `(less_total, equal_count, distinct_less)` relative to `key` under
    /// `f64::total_cmp`.
    fn query_total_cmp(&self, key: f64) -> (u32, u32, u32) {
        let mut cur = Some(self);
        let (mut less, mut eq, mut dless) = (0u32, 0u32, 0u32);
        while let Some(n) = cur {
            match key.total_cmp(&n.key) {
                Ordering::Less => cur = n.left.as_deref(),
                Ordering::Greater => {
                    less += Self::total_of(&n.left) + n.count;
                    dless += Self::distinct_of(&n.left) + 1;
                    cur = n.right.as_deref();
                }
                Ordering::Equal => {
                    less += Self::total_of(&n.left);
                    dless += Self::distinct_of(&n.left);
                    eq = n.count;
                    break;
                }
            }
        }
        (less, eq, dless)
    }

    fn count_of(&self, key: f64) -> u32 {
        let mut cur = Some(self);
        while let Some(n) = cur {
            match key.total_cmp(&n.key) {
                Ordering::Less => cur = n.left.as_deref(),
                Ordering::Greater => cur = n.right.as_deref(),
                Ordering::Equal => return n.count,
            }
        }
        0
    }

    fn collect_sorted(&self, out: &mut Vec<(f64, u32)>) {
        if let Some(l) = &self.left {
            l.collect_sorted(out);
        }
        out.push((self.key, self.count));
        if let Some(r) = &self.right {
            r.collect_sorted(out);
        }
    }

    fn from_sorted(entries: &[(f64, u32)]) -> Option<Box<Self>> {
        if entries.is_empty() {
            return None;
        }
        let mid = entries.len() / 2;
        let (key, count) = entries[mid];
        let mut n = Self::new(key, count);
        n.left = Self::from_sorted(&entries[..mid]);
        n.right = Self::from_sorted(&entries[mid + 1..]);
        n.refresh();
        Some(n)
    }
}

/// One exact-order operator's multiset inside a window queue.
#[derive(Debug, Clone, Default)]
pub(crate) struct OrdSide {
    root: Option<Box<Node>>,
}

impl OrdSide {
    #[inline]
    pub(crate) fn insert(&mut self, x: f64) {
        if !x.is_nan() {
            self.root = Some(Node::insert(self.root.take(), x));
        }
    }

    #[inline]
    pub(crate) fn remove(&mut self, x: f64) {
        if !x.is_nan() {
            self.root = Node::remove(self.root.take(), x);
        }
    }

    #[inline]
    pub(crate) fn clear(&mut self) {
        self.root = None;
    }

    #[inline]
    pub(crate) fn len(&self) -> u32 {
        Node::total_of(&self.root)
    }

    /// 0-indexed `idx`-th element of the window, with optional `open` value
    /// (for a forward window's unpushed open row) included without mutating the
    /// tree.
    fn kth_with(&self, idx: u32, open: Option<f64>) -> f64 {
        let Some(ox) = open.filter(|v| !v.is_nan()) else {
            return self.root.as_deref().expect("non-empty window").kth(idx);
        };
        let Some(root) = self.root.as_deref() else {
            return ox;
        };
        let (less, _, _) = root.query_total_cmp(ox);
        if idx < less {
            root.kth(idx)
        } else if idx == less {
            ox
        } else {
            root.kth(idx - 1)
        }
    }

    /// Exact quantile matching Polars' `rolling_quantile_by`.
    pub(crate) fn quantile(&self, q: f64, interp: Interp, open: Option<f64>) -> f64 {
        let open = open.filter(|v| !v.is_nan());
        let n = self.len() + u32::from(open.is_some());
        if n == 0 {
            return f64::NAN;
        }
        let h = (f64::from(n) - 1.0) * q;
        let last = n - 1;
        match interp {
            Interp::Nearest => self.kth_with((h.round() as u32).min(last), open),
            Interp::Lower => self.kth_with((h.floor() as u32).min(last), open),
            Interp::Higher => self.kth_with((h.ceil() as u32).min(last), open),
            Interp::Equiprobable => {
                let idx = ((f64::from(n) * q).ceil() - 1.0).max(0.0) as u32;
                self.kth_with(idx.min(last), open)
            }
            Interp::Midpoint | Interp::Linear => {
                let lo = (h.floor() as u32).min(last);
                let hi = (h.ceil() as u32).min(last);
                let a = self.kth_with(lo, open);
                let b = if hi == lo { a } else { self.kth_with(hi, open) };
                if a == b {
                    a
                } else if interp == Interp::Midpoint {
                    (a + b) / 2.0
                } else {
                    (h - f64::from(lo)) * (b - a) + a
                }
            }
        }
    }

    /// Exact rank of `x` within the window (matching Polars' `rolling_rank_by`,
    /// where `-0.0 == +0.0` are tied). Returns NaN when `x` is NaN or window is
    /// empty.
    pub(crate) fn rank(&self, x: f64, method: RankMethod, open: Option<f64>) -> f64 {
        if x.is_nan() {
            return f64::NAN;
        }
        let open = open.filter(|v| !v.is_nan());
        let n = self.len() + u32::from(open.is_some());
        if n == 0 {
            return f64::NAN;
        }
        // Query under numeric equality (`-0.0 == +0.0`).
        let (mut less, mut eq, mut dless) = match self.root.as_deref() {
            None => (0, 0, 0),
            Some(root) => {
                if x == 0.0 {
                    let (l_neg, eq_neg, dl_neg) = root.query_total_cmp(-0.0);
                    let eq_pos = root.count_of(0.0);
                    (l_neg, eq_neg + eq_pos, dl_neg)
                } else {
                    let (l, e, mut dl) = root.query_total_cmp(x);
                    // If both -0.0 and +0.0 appear strictly below x, they form
                    // one distinct value for `dense` rank.
                    if x > 0.0 && root.count_of(-0.0) > 0 && root.count_of(0.0) > 0 {
                        dl -= 1;
                    }
                    (l, e, dl)
                }
            }
        };
        if let Some(ox) = open {
            if ox < x {
                less += 1;
                let has_ox = match self.root.as_deref() {
                    None => false,
                    Some(root) => {
                        if ox == 0.0 {
                            root.count_of(-0.0) > 0 || root.count_of(0.0) > 0
                        } else {
                            root.count_of(ox) > 0
                        }
                    }
                };
                if !has_ox {
                    dless += 1;
                }
            } else if ox == x {
                eq += 1;
            }
        }
        if eq == 0 {
            return f64::NAN;
        }
        match method {
            RankMethod::Average => f64::from(less) + f64::from(eq + 1) * 0.5,
            RankMethod::Min => f64::from(less + 1),
            RankMethod::Max => f64::from(less + eq),
            RankMethod::Dense => f64::from(dless + 1),
            RankMethod::Ordinal => f64::NAN,
        }
    }

    fn entries(&self) -> Vec<(f64, u32)> {
        let mut out = Vec::with_capacity(Node::distinct_of(&self.root) as usize);
        if let Some(r) = &self.root {
            r.collect_sorted(&mut out);
        }
        out
    }

    /// Verify that this tree holds the exact multiset of non-NaN values in
    /// `values` (checked when loading a saved windows state).
    pub(crate) fn has_shape(&self, values: impl Iterator<Item = f64>) -> bool {
        let mut expected: Vec<f64> = values.filter(|v| !v.is_nan()).collect();
        if expected.len() != self.len() as usize {
            return false;
        }
        expected.sort_by(f64::total_cmp);
        let mut grouped: Vec<(u64, u32)> = Vec::new();
        for x in expected {
            let bits = x.to_bits();
            match grouped.last_mut() {
                Some((b, c)) if *b == bits => *c += 1,
                _ => grouped.push((bits, 1)),
            }
        }
        let actual: Vec<(u64, u32)> = self
            .entries()
            .into_iter()
            .map(|(k, c)| (k.to_bits(), c))
            .collect();
        actual == grouped
    }
}

impl PartialEq for OrdSide {
    fn eq(&self, other: &Self) -> bool {
        if self.len() != other.len() {
            return false;
        }
        let a = self.entries();
        let b = other.entries();
        a.len() == b.len()
            && a.iter()
                .zip(&b)
                .all(|((ak, ac), (bk, bc))| ak.to_bits() == bk.to_bits() && ac == bc)
    }
}

impl Serialize for OrdSide {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.entries().serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for OrdSide {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let entries: Vec<(f64, u32)> = Vec::deserialize(deserializer)?;
        for (i, &(k, c)) in entries.iter().enumerate() {
            if k.is_nan() || c == 0 {
                return Err(serde::de::Error::custom("invalid order-tree entry"));
            }
            if i > 0 && entries[i - 1].0.total_cmp(&k) != Ordering::Less {
                return Err(serde::de::Error::custom(
                    "order-tree entries must be strictly ascending",
                ));
            }
        }
        Ok(Self {
            root: Node::from_sorted(&entries),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn avl_tree_matches_sorted_slice_quantiles_and_ranks() {
        let mut state: u64 = 0x518_1234_5678;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = ((state >> 11) as f64) / ((1u64 << 53) as f64);
            ((u * 20.0).round() - 10.0) * 0.5
        };
        let xs: Vec<f64> = (0..120).map(|_| next()).collect();
        let win = 19usize;
        let mut ord = OrdSide::default();
        for i in 0..xs.len() {
            ord.insert(xs[i]);
            if i >= win {
                ord.remove(xs[i - win]);
            }
            let lo = (i + 1).saturating_sub(win);
            let mut sorted: Vec<f64> = xs[lo..=i].to_vec();
            sorted.sort_by(f64::total_cmp);
            assert_eq!(ord.len() as usize, sorted.len());
            assert!(ord.has_shape(xs[lo..=i].iter().copied()));
            for interp in [
                Interp::Nearest,
                Interp::Lower,
                Interp::Higher,
                Interp::Midpoint,
                Interp::Linear,
                Interp::Equiprobable,
            ] {
                for &q in &[0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0] {
                    let n = sorted.len();
                    let h = (n as f64 - 1.0) * q;
                    let want = match interp {
                        Interp::Nearest => sorted[(h.round() as usize).min(n - 1)],
                        Interp::Lower => sorted[(h.floor() as usize).min(n - 1)],
                        Interp::Higher => sorted[(h.ceil() as usize).min(n - 1)],
                        Interp::Equiprobable => {
                            sorted[(((n as f64 * q).ceil() - 1.0).max(0.0) as usize).min(n - 1)]
                        }
                        Interp::Midpoint => {
                            let a = sorted[(h.floor() as usize).min(n - 1)];
                            let b = sorted[(h.ceil() as usize).min(n - 1)];
                            if a == b { a } else { (a + b) / 2.0 }
                        }
                        Interp::Linear => {
                            let l = (h.floor() as usize).min(n - 1);
                            let a = sorted[l];
                            let b = sorted[(h.ceil() as usize).min(n - 1)];
                            if a == b {
                                a
                            } else {
                                (h - l as f64) * (b - a) + a
                            }
                        }
                    };
                    assert_eq!(
                        ord.quantile(q, interp, None).to_bits(),
                        want.to_bits(),
                        "row {i} q {q} {interp:?}"
                    );
                }
            }
            let x = xs[i];
            let less = sorted.iter().filter(|&&v| v < x).count();
            let eq = sorted.iter().filter(|&&v| v == x).count();
            let mut distinct_below: Vec<f64> = sorted.iter().copied().filter(|&v| v < x).collect();
            distinct_below.dedup();
            let dl = distinct_below.len();
            assert_eq!(
                ord.rank(x, RankMethod::Average, None),
                less as f64 + (eq as f64 + 1.0) * 0.5
            );
            assert_eq!(ord.rank(x, RankMethod::Min, None), (less + 1) as f64);
            assert_eq!(ord.rank(x, RankMethod::Max, None), (less + eq) as f64);
            assert_eq!(ord.rank(x, RankMethod::Dense, None), (dl + 1) as f64);
        }
        // Round-trip through msgpack.
        let bytes = rmp_serde::to_vec_named(&ord).unwrap();
        let back: OrdSide = rmp_serde::from_slice(&bytes).unwrap();
        assert_eq!(ord, back);
    }

    #[test]
    fn signed_zeros_and_open_row_match_polars() {
        let mut ord = OrdSide::default();
        ord.insert(-0.0);
        ord.insert(0.0);
        ord.insert(-0.0);
        ord.insert(1.0);
        // Quantile distinguishes -0.0 < +0.0.
        assert_eq!(
            ord.quantile(0.0, Interp::Nearest, None).to_bits(),
            (-0.0_f64).to_bits()
        );
        // Rank treats -0.0 and +0.0 as tied, and dense counts them as one distinct value.
        assert_eq!(ord.rank(0.0, RankMethod::Average, None), 2.0);
        assert_eq!(ord.rank(-0.0, RankMethod::Average, None), 2.0);
        assert_eq!(ord.rank(1.0, RankMethod::Dense, None), 2.0);
        assert_eq!(ord.rank(1.0, RankMethod::Average, None), 4.0);
        // Open row included without mutating the tree.
        assert_eq!(ord.rank(1.0, RankMethod::Average, Some(0.5)), 5.0);
        assert_eq!(ord.rank(1.0, RankMethod::Dense, Some(0.5)), 3.0);
        assert_eq!(ord.quantile(0.5, Interp::Linear, Some(0.5)), 0.0);
        assert_eq!(ord.len(), 4);
    }
}
