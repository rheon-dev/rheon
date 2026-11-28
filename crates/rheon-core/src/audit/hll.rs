//! HyperLogLog++ and sliding-window HyperLogLog++ distinct-count sketches for
//! `audit` (Heule, Nunkesser & Hall 2013; Chabchoub & Hébrail 2010).
//!
//! **Hash and registers.** Each usable value is keyed by its `f64` bits with
//! `-0.0` mapped to `0` (`audit::key`), then mixed with the SplitMix64
//! finalizer ([`splitmix64`]). Precision `p` in `8..=16` gives `m = 2^p`
//! registers: register index `j = h & (m - 1)` and rank `rho` = position of
//! the leftmost 1-bit in the remaining `64 - p` bits (`1..=(65 - p)`),
//! matching `datasketch.HyperLogLogPlusPlus(p, hashfunc=splitmix)` entry for
//! entry.
//!
//! **Estimate.** Dense HLL++ with the empirical bias tables in
//! `hll_tables.rs`: linear counting `m * ln(m / V)` via `libm::log` while
//! empty registers `V > 0` and `lc <= threshold[p]`, else raw estimate
//! `alpha_m * m^2 / sum_j 2^(-M_j)` minus the 6-nearest-neighbor interpolated
//! bias while `E <= 5 m`, else `E`. Maintaining an exact integer histogram of
//! active register ranks (`rank_counts: [u32; 58]`) evaluates `sum_j 2^(-M_j)`
//! in at most 57 exact power-of-two terms (zero floating-point drift,
//! independent of `p`) and caches the estimate until a register changes.
//!
//! **Sliding window (Chabchoub & Hébrail 2010).** Under `distinct_window_rows`
//! or `distinct_window_size`, each register `j` maintains its Pareto-optimal
//! frontier (list of `(rank, stamp)` pairs with strictly decreasing ranks and
//! strictly increasing stamps, at most `65 - p` entries per register). Stored
//! in an intrusive doubly-linked slab (8 bytes per register header plus one
//! node per live entry) paired with a bounded lazy min-heap over register
//! fronts for amortized `O(log m)` expiry and `O(1)` cached reads.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use serde::{Deserialize, Serialize};

use super::hll_tables;
use crate::Stamp;

/// Smallest supported HyperLogLog++ precision (`m = 256` registers, ~6.5% RSE).
pub const MIN_DISTINCT_PRECISION: u8 = 8;

/// Largest supported HyperLogLog++ precision (`m = 65_536` registers, ~0.41% RSE).
pub const MAX_DISTINCT_PRECISION: u8 = 16;

/// Largest possible rank `rho = 65 - p` across `p = 8..=16` (`65 - 8 = 57`).
const MAX_RANK: usize = 65 - MIN_DISTINCT_PRECISION as usize;

/// Sentinel index for empty intrusive slab links.
const NONE: u32 = u32::MAX;

/// Byte footprint of one live entry in a row-windowed sliding HLL++ slab.
pub const HLL_ROW_ENTRY_BYTES: u64 = 24;

/// Byte footprint of one live entry in a clock-windowed sliding HLL++ slab.
pub const HLL_CLOCK_ENTRY_BYTES: u64 = 40;

/// SplitMix64 finalizer: a bijection of 64-bit words with full avalanche on
/// sequential integers, cent grids, and floats. Fixed constants (no runtime
/// seed), so every process and platform maps a key to the same register and
/// rank. Pinned: `splitmix64(0) == 0xe220_a839_7b1d_cdaf`.
#[inline(always)]
pub fn splitmix64(k: u64) -> u64 {
    let mut z = k.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// Register index `j in 0..(1 << p)` and rank `rho in 1..=(65 - p)` for `key`.
#[inline]
pub(crate) fn reg_and_rank(key: u64, p: u8) -> (usize, u8) {
    let h = splitmix64(key);
    let j = (h & ((1u64 << p) - 1)) as usize;
    let rho = ((h >> p).leading_zeros() - u32::from(p) + 1) as u8;
    (j, rho)
}

/// Exact `2^(-r)` as an IEEE-754 double for `r in 0..=MAX_RANK`.
#[inline]
fn pow2_neg(r: usize) -> f64 {
    debug_assert!(r <= MAX_RANK);
    f64::from_bits((1023u64 - r as u64) << 52)
}

/// HyperLogLog `alpha_m` constant for `p >= 8` (`m >= 256`).
#[inline]
fn alpha_m(p: u8) -> f64 {
    0.7213 / (1.0 + 1.079 / (1u64 << p) as f64)
}

/// HLL++ 6-nearest-neighbor empirical bias interpolation (Heule et al. 2013),
/// matching `datasketch.HyperLogLogPlusPlus._estimate_bias`.
fn estimate_bias(e: f64, p: u8) -> f64 {
    let idx = usize::from(p - MIN_DISTINCT_PRECISION);
    let est = hll_tables::RAW_ESTIMATE[idx];
    let bias = hll_tables::BIAS[idx];
    // `est` is strictly ascending, so the 6 nearest entries always lie within
    // 6 positions on either side of `e`'s insertion point (at most 12 entries).
    let pos = est.partition_point(|&x| x < e);
    let lo = pos.saturating_sub(6);
    let hi = (pos + 6).min(est.len());
    let mut cand: [(f64, usize); 12] = [(f64::INFINITY, 0); 12];
    let len = hi - lo;
    for (slot, i) in (lo..hi).enumerate() {
        let d = e - est[i];
        cand[slot] = (d * d, i);
    }
    cand[..len].sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    let k = len.min(6);
    let mut sum = 0.0;
    for &(_, i) in &cand[..k] {
        sum += bias[i];
    }
    sum / (k as f64)
}

/// Dense HLL++ cardinality estimate from an exact rank histogram.
///
/// Uses `libm::log` (no platform `libm` call) for linear counting. An empty
/// sketch (`rank_counts[0] == m`) yields `0.0`.
pub(crate) fn count_from_rank_counts(rank_counts: &[u32; MAX_RANK + 1], p: u8) -> f64 {
    let m = (1u64 << p) as f64;
    let zeros = rank_counts[0];
    if zeros > 0 {
        let lc = m * libm::log(m / f64::from(zeros));
        if lc <= hll_tables::THRESHOLDS[usize::from(p - MIN_DISTINCT_PRECISION)] {
            return lc;
        }
    }
    let max_rank = 65 - usize::from(p);
    let mut inv_sum = 0.0;
    for (r, &cnt) in rank_counts.iter().enumerate().take(max_rank + 1) {
        if cnt > 0 {
            inv_sum += f64::from(cnt) * pow2_neg(r);
        }
    }
    let e = alpha_m(p) * (m * m) / inv_sum;
    if e <= 5.0 * m {
        e - estimate_bias(e, p)
    } else {
        e
    }
}

/// Total order on [`Stamp`] values within a stream (same variant).
#[inline]
pub(crate) fn stamp_ord(a: Stamp, b: Stamp) -> Ordering {
    match (a, b) {
        (Stamp::Ns(x), Stamp::Ns(y)) => x.cmp(&y),
        (Stamp::Raw(x, rx), Stamp::Raw(y, ry)) => ((x - y) - (rx - ry)).total_cmp(&0.0),
        (Stamp::Int(x, fx), Stamp::Int(y, fy)) => match x.cmp(&y) {
            Ordering::Equal => fx.total_cmp(&fy),
            ord if x.abs_diff(y) >= 2 => ord,
            _ => ((x - y) as f64 + (fx - fy)).total_cmp(&0.0),
        },
        _ => a.cmp_span(b, 0.0),
    }
}

/// Whether a [`Stamp`] holds finite numbers (rejects NaN/inf in corrupt state).
#[inline]
pub(crate) fn stamp_is_finite(s: Stamp) -> bool {
    match s {
        Stamp::Ns(_) => true,
        Stamp::Raw(a, b) => a.is_finite() && b.is_finite(),
        Stamp::Int(_, f) => f.is_finite(),
    }
}

// ---------------------------------------------------------------------------
// Plain (unwindowed) HyperLogLog++
// ---------------------------------------------------------------------------

/// Dense HyperLogLog++ sketch (`2^p` one-byte registers).
///
/// Serializes as a flat byte array (`Vec<u8>`); the rank histogram and cached
/// estimate are rebuilt on load.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(from = "Vec<u8>", into = "Vec<u8>")]
pub struct Hll {
    regs: Vec<u8>,
    rank_counts: [u32; MAX_RANK + 1],
    cached: f64,
    dirty: bool,
}

impl PartialEq for Hll {
    fn eq(&self, other: &Self) -> bool {
        self.regs == other.regs
    }
}

impl From<Vec<u8>> for Hll {
    fn from(regs: Vec<u8>) -> Self {
        let mut rank_counts = [0u32; MAX_RANK + 1];
        for &r in &regs {
            if usize::from(r) <= MAX_RANK {
                rank_counts[usize::from(r)] += 1;
            }
        }
        Self {
            regs,
            rank_counts,
            cached: 0.0,
            dirty: true,
        }
    }
}

impl From<Hll> for Vec<u8> {
    fn from(h: Hll) -> Self {
        h.regs
    }
}

impl Hll {
    /// Fresh empty sketch at precision `p` (`8..=16`).
    pub fn new(p: u8) -> Self {
        let m = 1usize << p;
        let mut rank_counts = [0u32; MAX_RANK + 1];
        rank_counts[0] = m as u32;
        Self {
            regs: vec![0u8; m],
            rank_counts,
            cached: 0.0,
            dirty: false,
        }
    }

    /// Raw register slice (length `2^p`).
    pub fn registers(&self) -> &[u8] {
        &self.regs
    }

    /// Insert one key (`audit::key` bits) into the sketch.
    #[inline]
    pub fn add_key(&mut self, key: u64, p: u8) {
        let (j, rho) = reg_and_rank(key, p);
        let old = self.regs[j];
        if rho > old {
            self.regs[j] = rho;
            self.rank_counts[usize::from(old)] -= 1;
            self.rank_counts[usize::from(rho)] += 1;
            self.dirty = true;
        }
    }

    /// Merge another sketch of the same precision by register-wise maximum.
    pub fn merge(&mut self, other: &Hll) {
        let mut changed = false;
        for (a, &b) in self.regs.iter_mut().zip(&other.regs) {
            if b > *a {
                self.rank_counts[usize::from(*a)] -= 1;
                self.rank_counts[usize::from(b)] += 1;
                *a = b;
                changed = true;
            }
        }
        if changed {
            self.dirty = true;
        }
    }

    /// Current HLL++ cardinality estimate (cached when registers have not moved).
    pub fn count(&self, p: u8) -> f64 {
        if !self.dirty {
            return self.cached;
        }
        count_from_rank_counts(&self.rank_counts, p)
    }

    /// Current HLL++ cardinality estimate, refreshing the cache if dirty.
    pub fn count_cached(&mut self, p: u8) -> f64 {
        if self.dirty {
            self.cached = count_from_rank_counts(&self.rank_counts, p);
            self.dirty = false;
        }
        self.cached
    }

    /// Validate register count and rank bounds for precision `p`.
    pub fn has_shape(&self, p: u8) -> bool {
        let m = 1usize << p;
        let max_rank = 65 - p;
        self.regs.len() == m && self.regs.iter().all(|&r| r <= max_rank)
    }
}

// ---------------------------------------------------------------------------
// Sliding-window HyperLogLog++ (Chabchoub & Hébrail 2010)
// ---------------------------------------------------------------------------

/// Stamp trait for [`SlidingHll`]: `u64` positive-weight row index for
/// `distinct_window_rows`, or [`Stamp`] for `distinct_window_size`.
pub(crate) trait HllStamp: Copy + PartialEq + std::fmt::Debug {
    fn ord(a: Self, b: Self) -> Ordering;
    fn is_valid(self) -> bool;
}

impl HllStamp for u64 {
    #[inline]
    fn ord(a: Self, b: Self) -> Ordering {
        a.cmp(&b)
    }
    #[inline]
    fn is_valid(self) -> bool {
        self >= 1
    }
}

impl HllStamp for Stamp {
    #[inline]
    fn ord(a: Self, b: Self) -> Ordering {
        stamp_ord(a, b)
    }
    #[inline]
    fn is_valid(self) -> bool {
        stamp_is_finite(self)
    }
}

#[derive(Debug, Clone, Copy)]
struct Node<S: Copy> {
    rank: u8,
    stamp: S,
    prev: u32,
    next: u32,
}

/// Min-heap element ordered by earliest `stamp` (then `reg`).
#[derive(Debug, Clone, Copy)]
struct HeapEntry<S: HllStamp> {
    stamp: S,
    reg: u32,
}

impl<S: HllStamp> PartialEq for HeapEntry<S> {
    fn eq(&self, other: &Self) -> bool {
        self.reg == other.reg && S::ord(self.stamp, other.stamp) == Ordering::Equal
    }
}

impl<S: HllStamp> Eq for HeapEntry<S> {}

impl<S: HllStamp> Ord for HeapEntry<S> {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reversed for min-heap behavior in `BinaryHeap`.
        S::ord(other.stamp, self.stamp).then_with(|| other.reg.cmp(&self.reg))
    }
}

impl<S: HllStamp> PartialOrd for HeapEntry<S> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Canonical serialized form of a [`SlidingHll`]: precision `p` and only the
/// live `(reg, rank, stamp)` triples sorted by `(reg, stamp)`.
#[derive(Serialize, Deserialize)]
struct SlidingHllFile<S> {
    p: u8,
    entries: Vec<(u32, u8, S)>,
}

/// Sliding-window HyperLogLog++ sketch (Chabchoub & Hébrail 2010 LFPM).
///
/// Each register `j in 0..2^p` stores its Pareto frontier of `(rank, stamp)`
/// pairs: ranks strictly decrease and stamps strictly increase from front
/// (oldest, highest rank) to back (newest, lowest rank).
#[derive(Debug, Clone)]
pub(crate) struct SlidingHll<S: HllStamp> {
    p: u8,
    heads: Vec<u32>,
    tails: Vec<u32>,
    nodes: Vec<Node<S>>,
    free: Vec<u32>,
    live_entries: usize,
    rank_counts: [u32; MAX_RANK + 1],
    heap: BinaryHeap<HeapEntry<S>>,
    cached: f64,
    dirty: bool,
    /// Set on load if the serialized entries violated ordering, bounds, or
    /// Pareto-frontier invariants; rejected by `Audit::restore`.
    corrupt: bool,
}

impl<S: HllStamp> PartialEq for SlidingHll<S> {
    fn eq(&self, other: &Self) -> bool {
        if self.p != other.p
            || self.live_entries != other.live_entries
            || self.corrupt != other.corrupt
            || self.heads.len() != other.heads.len()
        {
            return false;
        }
        for j in 0..self.heads.len() {
            let mut a = self.heads[j];
            let mut b = other.heads[j];
            while a != NONE && b != NONE {
                let na = self.nodes[a as usize];
                let nb = other.nodes[b as usize];
                if na.rank != nb.rank || na.stamp != nb.stamp {
                    return false;
                }
                a = na.next;
                b = nb.next;
            }
            if a != b {
                return false;
            }
        }
        true
    }
}

impl<S: HllStamp + Serialize> Serialize for SlidingHll<S> {
    fn serialize<Ser: serde::Serializer>(&self, serializer: Ser) -> Result<Ser::Ok, Ser::Error> {
        let mut entries = Vec::with_capacity(self.live_entries);
        for (j, &head) in self.heads.iter().enumerate() {
            let mut cur = head;
            while cur != NONE {
                let n = self.nodes[cur as usize];
                entries.push((j as u32, n.rank, n.stamp));
                cur = n.next;
            }
        }
        let file = SlidingHllFile { p: self.p, entries };
        file.serialize(serializer)
    }
}

impl<'de, S: HllStamp + Deserialize<'de>> Deserialize<'de> for SlidingHll<S> {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let file = SlidingHllFile::<S>::deserialize(deserializer)?;
        if !(MIN_DISTINCT_PRECISION..=MAX_DISTINCT_PRECISION).contains(&file.p) {
            let mut empty = Self::new(MIN_DISTINCT_PRECISION);
            empty.corrupt = true;
            return Ok(empty);
        }
        let mut h = Self::new(file.p);
        let m = 1usize << file.p;
        let max_rank = 65 - file.p;
        let mut prev_entry: Option<(u32, u8, S)> = None;
        for (reg, rank, stamp) in file.entries {
            if (reg as usize) >= m || rank == 0 || rank > max_rank || !stamp.is_valid() {
                h.corrupt = true;
                return Ok(h);
            }
            if let Some((pr, prank, pstamp)) = prev_entry {
                if reg < pr {
                    h.corrupt = true;
                    return Ok(h);
                }
                if reg == pr && (rank >= prank || S::ord(pstamp, stamp) != Ordering::Less) {
                    h.corrupt = true;
                    return Ok(h);
                }
            }
            prev_entry = Some((reg, rank, stamp));
            let j = reg as usize;
            let prev_idx = h.tails[j];
            let idx = h.alloc(Node {
                rank,
                stamp,
                prev: prev_idx,
                next: NONE,
            });
            if prev_idx == NONE {
                h.heads[j] = idx;
                h.rank_counts[0] -= 1;
                h.rank_counts[usize::from(rank)] += 1;
                h.heap.push(HeapEntry { stamp, reg });
            } else {
                h.nodes[prev_idx as usize].next = idx;
            }
            h.tails[j] = idx;
        }
        h.dirty = true;
        Ok(h)
    }
}

impl<S: HllStamp> SlidingHll<S> {
    /// Fresh empty sliding HLL++ at precision `p` (`8..=16`).
    pub fn new(p: u8) -> Self {
        let m = 1usize << p;
        let mut rank_counts = [0u32; MAX_RANK + 1];
        rank_counts[0] = m as u32;
        Self {
            p,
            heads: vec![NONE; m],
            tails: vec![NONE; m],
            nodes: Vec::new(),
            free: Vec::new(),
            live_entries: 0,
            rank_counts,
            heap: BinaryHeap::new(),
            cached: 0.0,
            dirty: false,
            corrupt: false,
        }
    }

    #[inline]
    pub fn live_entries(&self) -> usize {
        self.live_entries
    }

    #[inline]
    fn alloc(&mut self, node: Node<S>) -> u32 {
        self.live_entries += 1;
        if let Some(idx) = self.free.pop() {
            self.nodes[idx as usize] = node;
            idx
        } else {
            let idx = self.nodes.len() as u32;
            self.nodes.push(node);
            idx
        }
    }

    #[inline]
    fn release(&mut self, idx: u32) {
        self.live_entries -= 1;
        self.free.push(idx);
    }

    /// Rebuild the lazy min-heap from live register heads so stale entries
    /// never accumulate unboundedly (e.g. on a constant stream).
    fn rebuild_heap(&mut self) {
        self.heap.clear();
        for (j, &head) in self.heads.iter().enumerate() {
            if head != NONE {
                self.heap.push(HeapEntry {
                    stamp: self.nodes[head as usize].stamp,
                    reg: j as u32,
                });
            }
        }
    }

    /// Clear all entries (e.g. when a clock-windowed audit restarts on a new
    /// clock epoch).
    pub fn clear(&mut self) {
        self.heads.fill(NONE);
        self.tails.fill(NONE);
        self.nodes.clear();
        self.free.clear();
        self.live_entries = 0;
        self.rank_counts = [0u32; MAX_RANK + 1];
        self.rank_counts[0] = 1u32 << self.p;
        self.heap.clear();
        self.cached = 0.0;
        self.dirty = false;
    }

    /// Drop every entry whose stamp satisfies `is_expired`, oldest first.
    pub fn trim(&mut self, is_expired: impl Fn(S) -> bool) {
        while let Some(&top) = self.heap.peek() {
            let j = top.reg as usize;
            let head = self.heads[j];
            if head == NONE || S::ord(self.nodes[head as usize].stamp, top.stamp) != Ordering::Equal
            {
                self.heap.pop();
                continue;
            }
            if !is_expired(top.stamp) {
                break;
            }
            self.heap.pop();
            let old_node = self.nodes[head as usize];
            let next = old_node.next;
            self.heads[j] = next;
            if next == NONE {
                self.tails[j] = NONE;
            } else {
                self.nodes[next as usize].prev = NONE;
            }
            self.release(head);
            let new_rank = if next == NONE {
                0usize
            } else {
                let nxt = self.nodes[next as usize];
                self.heap.push(HeapEntry {
                    stamp: nxt.stamp,
                    reg: j as u32,
                });
                usize::from(nxt.rank)
            };
            let old_rank = usize::from(old_node.rank);
            if new_rank != old_rank {
                self.rank_counts[old_rank] -= 1;
                self.rank_counts[new_rank] += 1;
                self.dirty = true;
            }
        }
        if self.live_entries == 0 && !self.nodes.is_empty() {
            self.nodes.clear();
            self.free.clear();
        }
    }

    /// Insert one key at `now` (non-decreasing relative to existing stamps).
    #[inline]
    pub fn add_key(&mut self, key: u64, now: S) {
        let (j, rho) = reg_and_rank(key, self.p);
        let old_head = self.heads[j];
        let old_front_rank = if old_head == NONE {
            0usize
        } else {
            usize::from(self.nodes[old_head as usize].rank)
        };

        // At an equal stamp, a lower-or-equal rank is strictly dominated.
        if self.tails[j] != NONE {
            let tail_node = self.nodes[self.tails[j] as usize];
            if S::ord(tail_node.stamp, now) == Ordering::Equal && tail_node.rank >= rho {
                return;
            }
        }

        // Pop dominated entries from the back (`rank <= rho`).
        let mut head_changed = old_head == NONE;
        while self.tails[j] != NONE {
            let t = self.tails[j];
            let tail_node = self.nodes[t as usize];
            if tail_node.rank <= rho {
                let prev = tail_node.prev;
                self.tails[j] = prev;
                if prev == NONE {
                    self.heads[j] = NONE;
                    head_changed = true;
                } else {
                    self.nodes[prev as usize].next = NONE;
                }
                self.release(t);
            } else {
                break;
            }
        }

        let prev = self.tails[j];
        let idx = self.alloc(Node {
            rank: rho,
            stamp: now,
            prev,
            next: NONE,
        });
        if prev == NONE {
            self.heads[j] = idx;
            head_changed = true;
        } else {
            self.nodes[prev as usize].next = idx;
        }
        self.tails[j] = idx;

        if head_changed {
            let new_front_rank = usize::from(self.nodes[self.heads[j] as usize].rank);
            if new_front_rank != old_front_rank {
                self.rank_counts[old_front_rank] -= 1;
                self.rank_counts[new_front_rank] += 1;
                self.dirty = true;
            }
            self.heap.push(HeapEntry {
                stamp: now,
                reg: j as u32,
            });
            let cap = 2 * self.heads.len().max(64);
            if self.heap.len() > cap {
                self.rebuild_heap();
            }
        }
    }

    /// Current HLL++ estimate over all live entries (no expiration check).
    pub fn count_current(&self) -> f64 {
        if !self.dirty {
            return self.cached;
        }
        count_from_rank_counts(&self.rank_counts, self.p)
    }

    /// Current HLL++ estimate, refreshing the cache if dirty.
    pub fn count_current_cached(&mut self) -> f64 {
        if self.dirty {
            self.cached = count_from_rank_counts(&self.rank_counts, self.p);
            self.dirty = false;
        }
        self.cached
    }

    /// Read-only HLL++ estimate after hypothetically dropping entries matching
    /// `is_expired`. O(1) when the global minimum heap stamp is not expired.
    pub fn count_excluding(&self, is_expired: impl Fn(S) -> bool) -> f64 {
        if self.live_entries == 0 {
            return 0.0;
        }
        match self.heap.peek() {
            None => return 0.0,
            Some(top) if !is_expired(top.stamp) => return self.count_current(),
            Some(_) => {}
        }
        let mut rc = self.rank_counts;
        for &head in &self.heads {
            if head == NONE {
                continue;
            }
            let hnode = self.nodes[head as usize];
            if !is_expired(hnode.stamp) {
                continue;
            }
            let old_r = usize::from(hnode.rank);
            let mut cur = hnode.next;
            while cur != NONE && is_expired(self.nodes[cur as usize].stamp) {
                cur = self.nodes[cur as usize].next;
            }
            let new_r = if cur == NONE {
                0usize
            } else {
                usize::from(self.nodes[cur as usize].rank)
            };
            if new_r != old_r {
                rc[old_r] -= 1;
                rc[new_r] += 1;
            }
        }
        count_from_rank_counts(&rc, self.p)
    }

    /// Exact number of live entries that would remain after trimming with
    /// `is_expired` and optionally inserting `(key, now)` — without mutating
    /// `self` (for atomic `HeldCap` boundary checks).
    pub fn entries_after(&self, insert: Option<(u64, S)>, is_expired: impl Fn(S) -> bool) -> usize {
        let target = insert.map(|(k, now)| {
            let (j, rho) = reg_and_rank(k, self.p);
            (j, rho, now)
        });
        let mut dropped: usize = 0;
        // Count expired entries across all registers when any could be expired.
        let check_expiry = self.heap.peek().is_some_and(|top| is_expired(top.stamp));
        if check_expiry {
            for &head in &self.heads {
                let mut cur = head;
                while cur != NONE && is_expired(self.nodes[cur as usize].stamp) {
                    dropped += 1;
                    cur = self.nodes[cur as usize].next;
                }
            }
        }
        let Some((j, rho, now)) = target else {
            return self.live_entries - dropped;
        };
        // Find the first non-expired node in register `j`.
        let mut first_live = self.heads[j];
        while first_live != NONE && is_expired(self.nodes[first_live as usize].stamp) {
            first_live = self.nodes[first_live as usize].next;
        }
        if first_live == NONE {
            // Register `j` will be empty after trimming, so inserting adds 1.
            return self.live_entries - dropped + 1;
        }
        // Check equal-stamp domination at tail.
        let tail = self.tails[j];
        let tnode = self.nodes[tail as usize];
        if S::ord(tnode.stamp, now) == Ordering::Equal && tnode.rank >= rho {
            return self.live_entries - dropped;
        }
        // Count non-expired tail entries popped by `rank <= rho`.
        let mut dominated: usize = 0;
        let mut cur = tail;
        while cur != NONE {
            let n = self.nodes[cur as usize];
            if is_expired(n.stamp) {
                break;
            }
            if n.rank <= rho {
                dominated += 1;
            } else {
                break;
            }
            if cur == first_live {
                break;
            }
            cur = n.prev;
        }
        self.live_entries - dropped - dominated + 1
    }

    /// Materialize active registers (front rank of each register, 0 if empty)
    /// — useful in unit tests to verify register-for-register equality against
    /// a fresh plain [`Hll`].
    #[cfg(test)]
    pub fn active_registers(&self) -> Vec<u8> {
        self.heads
            .iter()
            .map(|&h| {
                if h == NONE {
                    0
                } else {
                    self.nodes[h as usize].rank
                }
            })
            .collect()
    }

    /// Merge another sliding HLL++ of the same precision, dropping entries
    /// that fail `keep` (e.g. expired at `max(last_a, last_b)`), preserving
    /// the exact Pareto frontier of the union of both streams.
    pub fn merge(&mut self, other: &SlidingHll<S>, keep: impl Fn(S) -> bool) {
        let m = self.heads.len();
        debug_assert_eq!(self.p, other.p);
        debug_assert_eq!(m, other.heads.len());
        let mut combined: Vec<(u8, S)> = Vec::new();
        let mut fresh = SlidingHll::new(self.p);
        for j in 0..m {
            combined.clear();
            let mut cur = self.heads[j];
            while cur != NONE {
                let n = self.nodes[cur as usize];
                if keep(n.stamp) {
                    combined.push((n.rank, n.stamp));
                }
                cur = n.next;
            }
            let mut cur = other.heads[j];
            while cur != NONE {
                let n = other.nodes[cur as usize];
                if keep(n.stamp) {
                    combined.push((n.rank, n.stamp));
                }
                cur = n.next;
            }
            if combined.is_empty() {
                continue;
            }
            // Sort by stamp ascending; for equal stamps put higher rank first
            // so the lower rank is naturally dominated.
            combined.sort_by(|a, b| S::ord(a.1, b.1).then_with(|| b.0.cmp(&a.0)));
            for &(rank, stamp) in &combined {
                fresh.push_reg(j, rank, stamp);
            }
        }
        *self = fresh;
    }

    /// Direct register insert used during merge (assumes stamps non-decreasing
    /// within register `j`).
    fn push_reg(&mut self, j: usize, rho: u8, now: S) {
        let old_head = self.heads[j];
        let old_front_rank = if old_head == NONE {
            0usize
        } else {
            usize::from(self.nodes[old_head as usize].rank)
        };
        if self.tails[j] != NONE {
            let tail_node = self.nodes[self.tails[j] as usize];
            if S::ord(tail_node.stamp, now) == Ordering::Equal && tail_node.rank >= rho {
                return;
            }
        }
        let mut head_changed = old_head == NONE;
        while self.tails[j] != NONE {
            let t = self.tails[j];
            let tail_node = self.nodes[t as usize];
            if tail_node.rank <= rho {
                let prev = tail_node.prev;
                self.tails[j] = prev;
                if prev == NONE {
                    self.heads[j] = NONE;
                    head_changed = true;
                } else {
                    self.nodes[prev as usize].next = NONE;
                }
                self.release(t);
            } else {
                break;
            }
        }
        let prev = self.tails[j];
        let idx = self.alloc(Node {
            rank: rho,
            stamp: now,
            prev,
            next: NONE,
        });
        if prev == NONE {
            self.heads[j] = idx;
            head_changed = true;
        } else {
            self.nodes[prev as usize].next = idx;
        }
        self.tails[j] = idx;
        if head_changed {
            let new_front_rank = usize::from(self.nodes[self.heads[j] as usize].rank);
            if new_front_rank != old_front_rank {
                self.rank_counts[old_front_rank] -= 1;
                self.rank_counts[new_front_rank] += 1;
                self.dirty = true;
            }
            self.heap.push(HeapEntry {
                stamp: now,
                reg: j as u32,
            });
        }
    }

    /// Validate loaded shape, bounds, and predicate `valid_stamp` on every
    /// live entry (e.g. non-future and non-expired).
    pub fn has_shape(&self, p: u8, valid_stamp: impl Fn(S) -> bool) -> bool {
        if self.corrupt || self.p != p || self.heads.len() != (1usize << p) {
            return false;
        }
        let max_rank = 65 - p;
        let mut count = 0usize;
        for &head in &self.heads {
            let mut cur = head;
            let mut prev: Option<(u8, S)> = None;
            while cur != NONE {
                let n = self.nodes[cur as usize];
                if n.rank == 0 || n.rank > max_rank || !n.stamp.is_valid() || !valid_stamp(n.stamp)
                {
                    return false;
                }
                if let Some((pr, ps)) = prev
                    && (n.rank >= pr || S::ord(ps, n.stamp) != Ordering::Less)
                {
                    return false;
                }
                prev = Some((n.rank, n.stamp));
                count += 1;
                cur = n.next;
            }
        }
        count == self.live_entries
    }
}

#[cfg(test)]
#[path = "hll_mutant_tests.rs"]
mod mutant_tests;
