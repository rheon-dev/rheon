//! The cross operators that read a stamp's values in order: a row's rank
//! within the cross-section, and a quantile of it.
//!
//! **Rank** (Polars' `rank(method, descending)` within the stamp, the row
//! counted in it): with `l` the members ranked before the row's value, `e`
//! the members tied with it, the row among them, and `p` the row's place
//! among those ties in input order (from 0),
//!
//! ```text
//! average = l + (e + 1) / 2,   min = l + 1,   max = l + e,
//! dense = (distinct values ranked before) + 1,   ordinal = l + p + 1
//! ```
//!
//! Under `closed = "left"` the row is not a member of the stamp it reads:
//! it is ranked as though it were added to it, last in input order.
//!
//! **Quantile** at level `q` of the `n` members sorted, `v[0] <= ... <=
//! v[n-1]`, with `h = (n - 1) q`, as Polars reads it: `nearest` takes
//! `v[round(h)]`, `lower` `v[floor(h)]`, `higher` `v[ceil(h)]`, `midpoint`
//! their average and `linear` `v[floor(h)] + (h - floor(h)) (v[ceil(h)] -
//! v[floor(h)])`; `equiprobable` takes `v[max(ceil(n q) - 1, 0)]`. With
//! weights only the last has a form: the smallest value whose cumulative
//! weight share reaches the level, `v[k]` for the first `k` with `(w_0 + ...
//! + w_k) / W >= q`, NumPy's weighted `inverted_cdf`.

use super::{CrossDef, CrossStat, Interp, RankMethod};

/// A stamp's members, the rows with a value and a weight above 0, sorted by
/// value, the earlier row first on a tie.
pub(super) fn members(stamp: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let mut m: Vec<(f64, f64)> = stamp
        .iter()
        .copied()
        .filter(|&(x, w)| !x.is_nan() && w > 0.0)
        .collect();
    // No NaN is left, so the order is total; the sort is stable.
    m.sort_by(|a, b| a.0.partial_cmp(&b.0).expect("no NaN among members"));
    m
}

/// A stamp's members as a rank reads them: each distinct value, ascending,
/// with the count of members at or below it.
pub(super) fn table(members: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let mut t: Vec<(f64, f64)> = Vec::new();
    for (k, &(x, _)) in members.iter().enumerate() {
        match t.last_mut() {
            Some(last) if last.0 == x => last.1 = (k + 1) as f64,
            _ => t.push((x, (k + 1) as f64)),
        }
    }
    t
}

/// The rank of `x` in the members of `table`: as one of them, `seen` of
/// its ties before it in input order, or (`member` false) as though it
/// were added last. Null where `x` is missing or the cross-section has
/// fewer than two members.
fn rank(table: &[(f64, f64)], x: f64, member: bool, seen: usize, def: &CrossDef) -> f64 {
    let total = table.last().map_or(0.0, |t| t.1);
    if x.is_nan() || total < 2.0 {
        return f64::NAN;
    }
    let i = table.partition_point(|t| t.0 < x);
    let found = i < table.len() && table[i].0 == x;
    let below = if i > 0 { table[i - 1].1 } else { 0.0 };
    let tied = if found { table[i].1 - below } else { 0.0 };
    let (l, distinct) = if def.descending {
        (total - below - tied, table.len() - i - usize::from(found))
    } else {
        (below, i)
    };
    let (e, p) = if member {
        (tied, seen as f64)
    } else {
        (tied + 1.0, tied)
    };
    match def.rank {
        RankMethod::Average => l + (e + 1.0) / 2.0,
        RankMethod::Min => l + 1.0,
        RankMethod::Max => l + e,
        RankMethod::Dense => distinct as f64 + 1.0,
        RankMethod::Ordinal => l + p + 1.0,
    }
}

/// Under `"left"`: the rank of `x` added to the stamp before, whose
/// [`table`] is `prev`.
pub(super) fn rank_of(prev: &[(f64, f64)], x: f64, def: &CrossDef) -> f64 {
    rank(prev, x, false, 0, def)
}

/// Under `"right"`: what each row of the closed stamp reads, in input order.
pub(super) fn read(stamp: &[(f64, f64)], members: &[(f64, f64)], def: &CrossDef) -> Vec<f64> {
    match def.stat {
        CrossStat::Rank => {
            let t = table(members);
            let mut seen = vec![0usize; t.len()];
            stamp
                .iter()
                .map(|&(x, w)| {
                    let member = !x.is_nan() && w > 0.0;
                    let mut before = 0;
                    if member {
                        let i = t.partition_point(|e| e.0 < x);
                        before = seen[i];
                        seen[i] += 1;
                    }
                    rank(&t, x, member, before, def)
                })
                .collect()
        }
        _ => vec![quantile_of(members, def); stamp.len()],
    }
}

/// The quantile of the sorted `members`; null for fewer than two.
pub(super) fn quantile_of(members: &[(f64, f64)], def: &CrossDef) -> f64 {
    let n = members.len();
    if n < 2 {
        return f64::NAN;
    }
    let q = def.quantile;
    let v = |k: usize| members[k.min(n - 1)].0;
    if def.weight.is_some() {
        // Only `equiprobable` is accepted with weights.
        let total: f64 = members.iter().map(|m| m.1).sum();
        let mut cum = 0.0;
        // The last member's share is 1 to the rounding, so it closes the
        // walk at any level.
        return members
            .iter()
            .find(|&&(_, w)| {
                cum += w;
                cum / total >= q
            })
            .map_or(members[n - 1].0, |m| m.0);
    }
    let h = (n as f64 - 1.0) * q;
    match def.interpolation {
        Interp::Nearest => v(h.round() as usize),
        Interp::Lower => v(h as usize),
        Interp::Higher => v(h.ceil() as usize),
        Interp::Equiprobable => v(((n as f64 * q).ceil() - 1.0).max(0.0) as usize),
        Interp::Midpoint | Interp::Linear => {
            let lo = h as usize;
            let (a, b) = (v(lo), v(h.ceil() as usize));
            if a == b {
                a
            } else if def.interpolation == Interp::Midpoint {
                (a + b) / 2.0
            } else {
                (h - lo as f64) * (b - a) + a
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::{CROSS_STATE_BYTES, CrossClosed};
    use super::*;

    fn def(stat: CrossStat, rank: RankMethod, descending: bool) -> CrossDef {
        CrossDef {
            stat,
            closed: CrossClosed::Right,
            input: 0,
            weight: None,
            bias: false,
            cap: CROSS_STATE_BYTES,
            rank,
            descending,
            quantile: 0.5,
            interpolation: Interp::Nearest,
            neighbors: None,
            pick: None,
        }
    }

    fn stamp(xs: &[f64]) -> Vec<(f64, f64)> {
        xs.iter().map(|&x| (x, 1.0)).collect()
    }

    /// Polars' ranks of `[3, 1, 3, 2, 3, 1]`, each method in both
    /// directions, as `pl.Series(...).rank(method, descending=...)` gives
    /// them (written out by hand from that call).
    #[test]
    fn a_stamp_ranks_as_polars_ranks_it() {
        let s = stamp(&[3.0, 1.0, 3.0, 2.0, 3.0, 1.0]);
        let cases: [(RankMethod, [f64; 6], [f64; 6]); 5] = [
            (
                RankMethod::Ordinal,
                [4.0, 1.0, 5.0, 3.0, 6.0, 2.0],
                [1.0, 5.0, 2.0, 4.0, 3.0, 6.0],
            ),
            (
                RankMethod::Dense,
                [3.0, 1.0, 3.0, 2.0, 3.0, 1.0],
                [1.0, 3.0, 1.0, 2.0, 1.0, 3.0],
            ),
            (
                RankMethod::Average,
                [5.0, 1.5, 5.0, 3.0, 5.0, 1.5],
                [2.0, 5.5, 2.0, 4.0, 2.0, 5.5],
            ),
            (
                RankMethod::Min,
                [4.0, 1.0, 4.0, 3.0, 4.0, 1.0],
                [1.0, 5.0, 1.0, 4.0, 1.0, 5.0],
            ),
            (
                RankMethod::Max,
                [6.0, 2.0, 6.0, 3.0, 6.0, 2.0],
                [3.0, 6.0, 3.0, 4.0, 3.0, 6.0],
            ),
        ];
        for (method, up, down) in cases {
            for (desc, want) in [(false, up), (true, down)] {
                let d = def(CrossStat::Rank, method, desc);
                let m = members(&s);
                assert_eq!(read(&s, &m, &d), want, "{method:?} descending={desc}");
            }
        }
    }

    /// A row added to a stamp ranks as the last row of the stamp with it.
    #[test]
    fn a_row_added_ranks_as_the_last_row_of_the_stamp_with_it() {
        let before = [3.0, 1.0, 3.0, 2.0, 1.0];
        for method in [
            RankMethod::Average,
            RankMethod::Min,
            RankMethod::Max,
            RankMethod::Dense,
            RankMethod::Ordinal,
        ] {
            for desc in [false, true] {
                let d = def(CrossStat::Rank, method, desc);
                let t = table(&members(&stamp(&before)));
                for x in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0] {
                    let mut with = before.to_vec();
                    with.push(x);
                    let s = stamp(&with);
                    let want = *read(&s, &members(&s), &d).last().unwrap();
                    assert_eq!(rank_of(&t, x, &d), want, "{method:?} {desc} {x}");
                }
            }
        }
    }

    /// A missing value reads null and is no member; one member is withheld.
    #[test]
    fn a_missing_value_reads_null_and_one_member_is_withheld() {
        let d = def(CrossStat::Rank, RankMethod::Average, false);
        let s = stamp(&[2.0, f64::NAN, 1.0]);
        let r = read(&s, &members(&s), &d);
        assert_eq!((r[0], r[2]), (2.0, 1.0));
        assert!(r[1].is_nan());
        let s = stamp(&[2.0, f64::NAN]);
        assert!(read(&s, &members(&s), &d).iter().all(|v| v.is_nan()));
        let mut q = def(CrossStat::Quantile, RankMethod::Average, false);
        assert!(quantile_of(&members(&s), &q).is_nan());
        q.weight = Some(1);
        assert!(quantile_of(&[(1.0, 2.0)], &q).is_nan());
    }

    /// Each interpolation at a level between two values, and at the ends.
    #[test]
    fn each_interpolation_reads_as_polars_defines_it() {
        let m = members(&stamp(&[4.0, 1.0, 3.0, 2.0]));
        let at = |interp: Interp, q: f64| {
            let mut d = def(CrossStat::Quantile, RankMethod::Average, false);
            d.interpolation = interp;
            d.quantile = q;
            quantile_of(&m, &d)
        };
        // h = 3 * 0.4 = 1.2: between v[1] = 2 and v[2] = 3.
        assert_eq!(at(Interp::Nearest, 0.4), 2.0);
        assert_eq!(at(Interp::Lower, 0.4), 2.0);
        assert_eq!(at(Interp::Higher, 0.4), 3.0);
        assert_eq!(at(Interp::Midpoint, 0.4), 2.5);
        assert_eq!(at(Interp::Linear, 0.4), (1.2f64 - 1.0) * 1.0 + 2.0);
        // ceil(4 * 0.4) - 1 = 1.
        assert_eq!(at(Interp::Equiprobable, 0.4), 2.0);
        for interp in [
            Interp::Nearest,
            Interp::Lower,
            Interp::Higher,
            Interp::Midpoint,
            Interp::Linear,
            Interp::Equiprobable,
        ] {
            assert_eq!(at(interp, 0.0), 1.0, "{interp:?}");
            assert_eq!(at(interp, 1.0), 4.0, "{interp:?}");
        }
    }

    /// The weighted level: the first value whose cumulative share reaches it.
    #[test]
    fn the_weighted_quantile_is_the_first_value_whose_share_reaches_the_level() {
        let mut d = def(CrossStat::Quantile, RankMethod::Average, false);
        d.weight = Some(1);
        d.interpolation = Interp::Equiprobable;
        let m = members(&[(3.0, 1.0), (1.0, 1.0), (2.0, 2.0), (9.0, 0.0)]);
        assert_eq!(m.len(), 3, "the row of weight 0 is no member");
        for (q, want) in [
            (0.0, 1.0),
            (0.25, 1.0),
            (0.26, 2.0),
            (0.75, 2.0),
            (0.76, 3.0),
            (1.0, 3.0),
        ] {
            d.quantile = q;
            assert_eq!(quantile_of(&m, &d), want, "q = {q}");
        }
    }
}
