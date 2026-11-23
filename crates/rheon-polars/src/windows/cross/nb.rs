//! A cross operator over neighbors: each row reads its stamp's rows of the
//! groups its own list names, not every group's.
//!
//! A row's neighbors are the group keys in its list column, each counted
//! once, in the order first listed; a key no group has yet, or a null, names
//! no row. The row's own group is a neighbor only where its list names it.
//! Under `"right"` the neighbors' rows are read at the close of the row's
//! own stamp, so a neighbor later in the stamp counts; under `"left"` they
//! are the rows of the stamp before, read as the row arrives.
//!
//! The rows read are the neighbors' members of that stamp (a value and a
//! weight above 0), in input order, and the statistic is the one the
//! operator computes over every group: the weighted moments in Welford's
//! form, the rank of the row's own value, or the quantile. Fewer than two
//! members read null, as every cross-section does.

use serde::{Deserialize, Serialize};

use super::{Acc, CrossDef, CrossStat, order};

/// A row of the open stamp: its `seq`, group, value and weight (NaN and 0
/// where it has none), and under `"right"` its neighbors' keys.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(super) struct NbRow {
    pub seq: u64,
    pub group: usize,
    pub x: f64,
    pub w: f64,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub keys: Vec<String>,
}

impl NbRow {
    pub fn member(&self) -> bool {
        !self.x.is_nan() && self.w > 0.0
    }
}

/// A member of a closed stamp: its group, `seq`, value and weight.
pub(super) type Member = (usize, u64, f64, f64);

/// The bytes a row of the open stamp is counted at, its keys aside: its
/// `seq`, group, value and weight, and its list's own length.
pub(super) const ROW_BYTES: u64 = 40;

/// The bytes a member of the stamp before is counted at.
pub(super) const MEMBER_BYTES: u64 = 32;

/// A neighbor key's bytes: its text and 8 for its length.
fn key_bytes(k: &str) -> u64 {
    8 + k.len() as u64
}

/// The bytes of a row's list as it arrives.
pub(super) fn keys_bytes(keys: Option<&[Option<String>]>) -> u64 {
    keys.map_or(0, |ks| ks.iter().flatten().map(|k| key_bytes(k)).sum())
}

pub(super) fn owned_keys_bytes(keys: &[String]) -> u64 {
    keys.iter().map(|k| key_bytes(k)).sum()
}

/// A row's list, each key once in the order first listed, nulls left out.
pub(super) fn dedup(keys: Option<&[Option<String>]>) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for k in keys.into_iter().flatten().flatten() {
        if !out.contains(k) {
            out.push(k.clone());
        }
    }
    out
}

/// The groups the keys name, where the core has them.
pub(super) fn resolve(keys: &[String], find: &dyn Fn(&str) -> Option<usize>) -> Vec<usize> {
    keys.iter().filter_map(|k| find(k)).collect()
}

/// A stamp's members, by group and then input order.
pub(super) fn members_of(stamp: &[NbRow]) -> Vec<Member> {
    let mut m: Vec<Member> = stamp
        .iter()
        .filter(|r| r.member())
        .map(|r| (r.group, r.seq, r.x, r.w))
        .collect();
    m.sort_unstable_by_key(|&(g, s, _, _)| (g, s));
    m
}

/// The members of `groups` among `members`, in input order.
pub(super) fn gather(members: &[Member], groups: &[usize]) -> Vec<(u64, f64, f64)> {
    let mut out = Vec::new();
    for &g in groups {
        let from = members.partition_point(|m| m.0 < g);
        let to = members.partition_point(|m| m.0 <= g);
        out.extend(members[from..to].iter().map(|&(_, s, x, w)| (s, x, w)));
    }
    out.sort_unstable_by_key(|r| r.0);
    out
}

/// The statistic over the gathered members, for a row of value `x`; `own`
/// is the row's `seq` where it is a member of what it reads.
pub(super) fn value(gathered: &[(u64, f64, f64)], x: f64, own: Option<u64>, def: &CrossDef) -> f64 {
    let pairs: Vec<(f64, f64)> = gathered.iter().map(|&(_, x, w)| (x, w)).collect();
    match def.stat {
        CrossStat::Mean | CrossStat::Std => {
            let mut acc = Acc::default();
            for &(x, w) in &pairs {
                acc.add(x, w);
            }
            acc.value(def)
        }
        CrossStat::Quantile => order::quantile_of(&order::members(&pairs), def),
        CrossStat::Rank => {
            let sorted = order::members(&pairs);
            match own.and_then(|s| gathered.iter().position(|r| r.0 == s)) {
                Some(at) => order::read(&pairs, &sorted, def)[at],
                None => order::rank_of(&order::table(&sorted), x, def),
            }
        }
        CrossStat::Ref => unreachable!("a reference read has no neighbors"),
    }
}

/// Under `"right"`: what each row of the closed stamp reads, in input order.
pub(super) fn read(
    stamp: &[NbRow],
    members: &[Member],
    def: &CrossDef,
    find: &dyn Fn(&str) -> Option<usize>,
) -> Vec<f64> {
    stamp
        .iter()
        .map(|r| {
            let groups = resolve(&r.keys, find);
            let own = r.member().then_some(r.seq);
            value(&gather(members, &groups), r.x, own, def)
        })
        .collect()
}
