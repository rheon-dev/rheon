//! Targeted unit tests for `audit/hll.rs` and `audit/hll_tables.rs`
//! (Wave 2 mutation survivors).

use super::*;

fn fnv1a_f64s(vals: &[f64]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &v in vals {
        for b in v.to_bits().to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

/// Pin every entry bit-for-bit across `THRESHOLDS`, `RAW_ESTIMATE`, and `BIAS`
/// (including every negative sign in `BIAS_8..=BIAS_16`).
#[test]
fn hll_tables_exact_digests_lengths_and_monotonicity() {
    assert_eq!(hll_tables::THRESHOLDS.len(), 9);
    assert_eq!(fnv1a_f64s(&hll_tables::THRESHOLDS), 0x67b4_3e43_d64d_ea07);

    let expected: [(usize, u64, u64, usize); 9] = [
        (200, 0xc9c8_18a7_8472_d0e4, 0x5c62_876c_cedd_372c, 3),
        (201, 0xb4e5_34d7_d471_0b2c, 0x8c30_7e36_5d23_abfe, 52),
        (200, 0x67f8_5219_aac9_2223, 0xb171_e48e_446b_4d9a, 32),
        (201, 0x42a4_64ae_b95b_4edd, 0x3523_f8b4_bf06_d19d, 16),
        (201, 0x59a8_3896_295f_5ad0, 0x5bfd_c280_9e1c_8ab8, 27),
        (200, 0xba73_1960_db02_6700, 0x794d_76c3_8047_b113, 33),
        (201, 0x82f6_52a6_fe29_f0f0, 0xdbc9_006b_91dc_e2c7, 26),
        (201, 0x0dec_86f4_c7cd_6a56, 0x0b28_ef1f_2302_ace7, 28),
        (200, 0x8dcc_3c0e_b699_481e, 0x5fb6_6c4c_d7b6_cad7, 22),
    ];
    for (idx, &(len, raw_hash, bias_hash, neg_count)) in expected.iter().enumerate() {
        let raw = hll_tables::RAW_ESTIMATE[idx];
        let bias = hll_tables::BIAS[idx];
        assert_eq!(raw.len(), len, "raw len idx={idx}");
        assert_eq!(bias.len(), len, "bias len idx={idx}");
        assert!(
            raw.windows(2).all(|w| w[0] < w[1] && w[0] > 0.0),
            "raw strictly ascending idx={idx}"
        );
        assert_eq!(
            bias.iter().filter(|&&b| b < 0.0).count(),
            neg_count,
            "bias negative count idx={idx}"
        );
        assert_eq!(fnv1a_f64s(raw), raw_hash, "raw digest idx={idx}");
        assert_eq!(fnv1a_f64s(bias), bias_hash, "bias digest idx={idx}");
    }
}

#[test]
fn max_rank_alpha_m_estimate_bias_and_rank_count_edges() {
    assert_eq!(MAX_RANK, 57);
    assert!((alpha_m(8) - 0.7213 / (1.0 + 1.079 / 256.0)).abs() < 1e-15);
    assert!((alpha_m(12) - 0.7213 / (1.0 + 1.079 / 4096.0)).abs() < 1e-15);

    // Brute-force 6-NN reference for `estimate_bias`.
    for (p, e) in [(8u8, 650.0), (12u8, 10_000.0)] {
        let idx = usize::from(p - MIN_DISTINCT_PRECISION);
        let raw = hll_tables::RAW_ESTIMATE[idx];
        let bias = hll_tables::BIAS[idx];
        let mut all: Vec<(f64, usize)> = raw
            .iter()
            .enumerate()
            .map(|(i, &x)| ((e - x) * (e - x), i))
            .collect();
        all.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        let want: f64 = all[..6].iter().map(|&(_, i)| bias[i]).sum::<f64>() / 6.0;
        let got = estimate_bias(e, p);
        assert!((got - want).abs() < 1e-12, "p={p} got={got} want={want}");
    }

    // `p = 8`: `zeros = 70` gives `lc = 256 * ln(256/70) ≈ 331.86`, which is
    // above THRESHOLDS[0] = 220.0 (takes HLL path) but below THRESHOLDS[1] = 400.0.
    let mut rc = [0u32; MAX_RANK + 1];
    rc[0] = 70;
    rc[1] = 186;
    let lc = 256.0 * libm::log(256.0 / 70.0);
    let est = count_from_rank_counts(&rc, 8);
    assert!(
        (est - lc).abs() > 1.0,
        "expected HLL path above p=8 threshold, got {est} vs lc={lc}"
    );

    // `p = 8`, all 256 registers at `max_rank = 57`: dropping rank 57 yields +inf.
    let mut rc_max = [0u32; MAX_RANK + 1];
    rc_max[57] = 256;
    let est_max = count_from_rank_counts(&rc_max, 8);
    let want_max = alpha_m(8) * (256.0 * 256.0) / (256.0 * pow2_neg(57));
    assert!(est_max.is_finite());
    assert!((est_max - want_max).abs() <= 1e-12 * want_max);

    // `p = 9`, `max_rank = 56`: `rc[57]` sits past `max_rank` and must be ignored.
    let mut rc_p9 = [0u32; MAX_RANK + 1];
    rc_p9[56] = 512;
    rc_p9[57] = 512;
    let est_p9 = count_from_rank_counts(&rc_p9, 9);
    let want_p9 = alpha_m(9) * (512.0 * 512.0) / (512.0 * pow2_neg(56));
    assert!((est_p9 - want_p9).abs() <= 1e-12 * want_p9);
}

#[test]
fn stamp_ord_all_variants_and_tie_breakers() {
    // Raw: `-0.0` vs `+0.0` distinguishes `total_cmp` from `cmp_span`'s `partial_cmp`.
    assert_eq!(
        stamp_ord(Stamp::Raw(-0.0, 0.0), Stamp::Raw(0.0, 0.0)),
        Ordering::Less
    );
    // Raw with non-zero removed times: `(10-8) - (2-5) = +5 > 0`.
    assert_eq!(
        stamp_ord(Stamp::Raw(10.0, 2.0), Stamp::Raw(8.0, 5.0)),
        Ordering::Greater
    );
    // Int equal whole part: `-0.0 < +0.0` via `total_cmp` (differs from fallback).
    assert_eq!(
        stamp_ord(Stamp::Int(5, -0.0), Stamp::Int(5, 0.0)),
        Ordering::Less
    );
    // Int `abs_diff == 2`: takes whole-part `ord` guard without reading fractions.
    assert_eq!(
        stamp_ord(Stamp::Int(12, 0.0), Stamp::Int(10, 3.0)),
        Ordering::Greater
    );
    // Int `abs_diff == 1`: crosses whole boundary via fractions (`-1 + 1.3 = +0.3`).
    assert_eq!(
        stamp_ord(Stamp::Int(10, 0.9), Stamp::Int(11, -0.4)),
        Ordering::Greater
    );
    // Int `abs_diff == 1`: `10 - 11 = -1` vs `10 / 11 = 0`.
    assert_eq!(
        stamp_ord(Stamp::Int(10, 0.2), Stamp::Int(11, 0.1)),
        Ordering::Less
    );
}

#[test]
fn hll_eq_cache_invariants_stamp_validity_and_heap_entry_eq() {
    let mut a = Hll::new(8);
    let b = Hll::new(8);
    a.add_key(1, 8);
    assert_ne!(a, b);

    // Duplicate key or idempotent merge leaves `dirty == false`.
    let _ = a.count_cached(8);
    assert!(!a.dirty);
    a.add_key(1, 8);
    assert!(!a.dirty, "duplicate key must not dirty cache");
    let a_copy = a.clone();
    a.merge(&a_copy);
    assert!(!a.dirty, "idempotent merge must not dirty cache");

    // `HllStamp::is_valid` for `u64` and `Stamp`.
    assert!(!0u64.is_valid());
    assert!(1u64.is_valid());
    assert!(!Stamp::Raw(f64::NAN, 0.0).is_valid());
    assert!(Stamp::Raw(0.0, 0.0).is_valid());

    // `HeapEntry::eq` on both `reg` and `stamp`.
    let e0 = HeapEntry {
        stamp: 10u64,
        reg: 3,
    };
    let e_same = HeapEntry {
        stamp: 10u64,
        reg: 3,
    };
    let e_reg = HeapEntry {
        stamp: 10u64,
        reg: 4,
    };
    let e_stamp = HeapEntry {
        stamp: 11u64,
        reg: 3,
    };
    assert_eq!(e0, e_same);
    assert_ne!(e0, e_reg);
    assert_ne!(e0, e_stamp);
}

#[test]
fn sliding_hll_eq_distinguishes_every_field_and_node() {
    let base: SlidingHll<u64> = SlidingHll::new(8);
    assert_eq!(base, base.clone());

    let mut diff_p = base.clone();
    diff_p.p = 9;
    assert_ne!(base, diff_p);

    let mut diff_live = base.clone();
    diff_live.live_entries = 1;
    assert_ne!(base, diff_live);

    let mut diff_corrupt = base.clone();
    diff_corrupt.corrupt = true;
    assert_ne!(base, diff_corrupt);

    let mut diff_heads = base.clone();
    diff_heads.heads.push(NONE);
    assert_ne!(base, diff_heads);

    // Single entry in register 0 vs register 1 (`a != NONE && b == NONE`).
    let mut r0: SlidingHll<u64> = SlidingHll::new(8);
    r0.push_reg(0, 3, 1);
    let mut r1: SlidingHll<u64> = SlidingHll::new(8);
    r1.push_reg(1, 3, 1);
    assert_ne!(r0, r1);

    // Same register and stamp, different rank.
    let mut diff_rank: SlidingHll<u64> = SlidingHll::new(8);
    diff_rank.push_reg(0, 4, 1);
    assert_ne!(r0, diff_rank);

    // Same register and rank, different stamp.
    let mut diff_stamp: SlidingHll<u64> = SlidingHll::new(8);
    diff_stamp.push_reg(0, 3, 2);
    assert_ne!(r0, diff_stamp);
}

#[test]
fn sliding_hll_deserialize_rejects_every_corrupt_case() {
    let decode = |p: u8, entries: Vec<(u32, u8, u64)>| -> SlidingHll<u64> {
        let file = SlidingHllFile { p, entries };
        let bytes = rmp_serde::to_vec(&file).unwrap();
        rmp_serde::from_slice(&bytes).unwrap()
    };

    // Valid max rank (`65 - 8 = 57`) loads cleanly.
    assert!(!decode(8, vec![(0, 57, 1)]).corrupt);
    // Out-of-range rank (`58 > 57`).
    assert!(decode(8, vec![(0, 58, 1)]).corrupt);
    // Zero rank.
    assert!(decode(8, vec![(0, 0, 1)]).corrupt);
    // Out-of-range register (`256 >= 256`).
    assert!(decode(8, vec![(256, 1, 1)]).corrupt);
    // Invalid stamp (`0u64`).
    assert!(decode(8, vec![(0, 1, 0)]).corrupt);
    // Same register: non-decreasing rank with strictly increasing stamps.
    assert!(decode(8, vec![(0, 3, 1), (0, 3, 2)]).corrupt);
    // Same register: strictly decreasing rank with non-increasing stamps.
    assert!(decode(8, vec![(0, 5, 2), (0, 3, 2)]).corrupt);
}

#[test]
fn sliding_hll_heap_compaction_exact_boundary_and_trim_cleanup() {
    let mut h: SlidingHll<u64> = SlidingHll::new(8);
    // `cap = 2 * heads.len().max(64) = 2 * 256 = 512`.
    for s in 1..=512u64 {
        h.add_key(42, s);
    }
    assert_eq!(h.heap.len(), 512, "must not compact at heap.len() == cap");
    h.add_key(42, 513);
    assert_eq!(h.heap.len(), 1, "must compact to live heads at cap + 1");
    assert_eq!(h.live_entries(), 1);

    h.trim(|s| s <= 513);
    assert_eq!(h.live_entries(), 0);
    assert!(h.nodes.is_empty() && h.free.is_empty());
}

#[test]
fn sliding_hll_trim_count_excluding_add_key_push_reg_and_has_shape_edges() {
    let p = 8u8;
    // Find two keys with the same rank in distinct registers.
    let (k1, r1, j1) = (0u64..10_000)
        .map(|k| {
            let (j, r) = reg_and_rank(k, p);
            (k, r, j)
        })
        .find(|&(_, r, _)| r == 1)
        .unwrap();
    let k2 = (k1 + 1..10_000)
        .find(|&k| {
            let (j, r) = reg_and_rank(k, p);
            r == r1 && j != j1
        })
        .unwrap();

    let mut h: SlidingHll<u64> = SlidingHll::new(p);
    h.add_key(k1, 1);
    h.add_key(k2, 2);
    assert_eq!(h.rank_counts[usize::from(r1)], 2);

    // `count_excluding` decrements `rc[old_r]` from 2 to 1.
    let mut ref1 = Hll::new(p);
    ref1.add_key(k2, p);
    assert_eq!(
        h.count_excluding(|s| s <= 1).to_bits(),
        ref1.count(p).to_bits()
    );

    // `trim` decrements `rank_counts[old_rank]` from 2 to 1.
    h.trim(|s| s <= 1);
    assert_eq!(h.rank_counts[usize::from(r1)], 1);
    assert_eq!(h.count_current().to_bits(), ref1.count(p).to_bits());

    // Find keys in the same register with ranks `rho_hi > rho_lo`.
    let k_lo = k1;
    let k_hi = (0u64..100_000)
        .find(|&k| {
            let (j, r) = reg_and_rank(k, p);
            j == j1 && r > r1
        })
        .unwrap();
    let (_, rho_hi) = reg_and_rank(k_hi, p);

    // Equal-stamp lower rank is dominated without adding an entry.
    let mut same_ts: SlidingHll<u64> = SlidingHll::new(p);
    same_ts.add_key(k_hi, 1);
    same_ts.add_key(k_lo, 1);
    assert_eq!(same_ts.live_entries(), 1);
    assert_eq!(same_ts.active_registers()[j1], rho_hi);

    // Equal-stamp higher rank replaces lower rank.
    let mut same_ts2: SlidingHll<u64> = SlidingHll::new(p);
    same_ts2.add_key(k_lo, 1);
    same_ts2.add_key(k_hi, 1);
    assert_eq!(same_ts2.live_entries(), 1);
    assert_eq!(same_ts2.active_registers()[j1], rho_hi);

    // Appending a lower rank at a later stamp must NOT push onto `heap`.
    let mut append: SlidingHll<u64> = SlidingHll::new(p);
    append.add_key(k_hi, 1);
    append.add_key(k_lo, 2);
    assert_eq!(append.live_entries(), 2);
    assert_eq!(append.heap.len(), 1);

    // `push_reg` edges: equal-stamp domination, later-stamp append, mid-list pop.
    let mut pr: SlidingHll<u64> = SlidingHll::new(p);
    pr.push_reg(0, 5, 10);
    pr.push_reg(0, 3, 10);
    assert_eq!(pr.live_entries(), 1);
    pr.push_reg(0, 2, 20);
    assert_eq!(pr.live_entries(), 2);
    assert_eq!(pr.heap.len(), 1);
    pr.push_reg(0, 3, 30); // pops `(2, 20)` whose `prev != NONE`
    assert_eq!(pr.live_entries(), 2);
    assert_eq!(pr.heap.len(), 1);
    assert!(pr.has_shape(p, |_| true));

    // `has_shape` edge checks.
    let mut bad_corrupt: SlidingHll<u64> = SlidingHll::new(p);
    bad_corrupt.corrupt = true;
    assert!(!bad_corrupt.has_shape(p, |_| true));

    let mut bad_heads: SlidingHll<u64> = SlidingHll::new(p);
    bad_heads.heads.pop();
    assert!(!bad_heads.has_shape(p, |_| true));

    // Exact `max_rank = 57` is valid (`>` not `>=`).
    let mut exact_max: SlidingHll<u64> = SlidingHll::new(p);
    exact_max.push_reg(0, 57, 1);
    assert!(exact_max.has_shape(p, |_| true));

    // Corrupt second node rank only, then stamp only.
    let mut bad_r = pr.clone();
    let second = bad_r.nodes[bad_r.heads[0] as usize].next;
    bad_r.nodes[second as usize].rank = 5; // >= pr (5), stamps still 10 < 30
    assert!(!bad_r.has_shape(p, |_| true));

    let mut bad_s = pr.clone();
    let second_s = bad_s.nodes[bad_s.heads[0] as usize].next;
    bad_s.nodes[second_s as usize].stamp = 10; // not > 10, ranks still 5 > 3
    assert!(!bad_s.has_shape(p, |_| true));
}

#[test]
fn sliding_hll_entries_after_matches_exact_trim_and_add_across_all_branches() {
    let p = 8u8;
    let target_reg = 0usize;
    let find_key = |reg: usize, want_rank: u8| -> u64 {
        (0u64..2_000_000)
            .find(|&k| reg_and_rank(k, p) == (reg, want_rank))
            .unwrap()
    };
    let k5 = find_key(target_reg, 5);
    let k4 = find_key(target_reg, 4);
    let k3 = find_key(target_reg, 3);
    let k2 = find_key(target_reg, 2);
    let k1 = find_key(target_reg, 1);

    let mut h: SlidingHll<u64> = SlidingHll::new(p);
    // Register 0: 1 expired (`stamp=1`) + 3 live (`stamp=10, 11, 12`).
    h.add_key(k5, 1);
    h.add_key(k4, 10);
    h.add_key(k3, 11);
    h.add_key(k2, 12);
    // 2 expired entries in registers 1..=2 (`dropped` total = 3).
    h.add_key(find_key(1, 1), 1);
    h.add_key(find_key(2, 1), 2);
    // 4 live entries in registers 3..=6 (`live_entries` total = 10).
    for reg in 3..=6 {
        h.add_key(find_key(reg, 1), 10);
    }
    assert_eq!(h.live_entries(), 10);

    let check = |insert: Option<(u64, u64)>, expected: usize| {
        let got = h.entries_after(insert, |s| s <= 5);
        let mut actual = h.clone();
        actual.trim(|s| s <= 5);
        if let Some((k, now)) = insert {
            actual.add_key(k, now);
        }
        assert_eq!(got, actual.live_entries(), "insert={insert:?}");
        assert_eq!(got, expected, "insert={insert:?}");
    };

    // 1. `insert = None`: `10 - 3 = 7`.
    check(None, 7);
    // 2. `first_live == NONE` (register 1's only entry expired): `10 - 3 + 1 = 8`.
    check(Some((find_key(1, 1), 13)), 8);
    // 3. Equal-stamp domination at tail (`stamp = 12`, rank 1 <= 2): `10 - 3 = 7`.
    check(Some((k1, 12)), 7);
    // 4. Dominates 2 non-expired tail nodes (`k2`, `k3`) while stopping before `k4`
    //    (`live=10, dropped=3, dominated=2 => 10 - 3 - 2 + 1 = 6`).
    check(Some((k3, 13)), 6);
}
