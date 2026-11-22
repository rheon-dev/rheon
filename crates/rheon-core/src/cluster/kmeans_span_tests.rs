//! `kmeans`' spans of the clock (docs/PLAN.md task 538): the warm-up over a
//! span (`warm_clock`) and the two cadences on the clock (`update_every`,
//! `split_merge_every`), each against its count of rows on a clock of one
//! unit a row, both sides of each edge, the warm-up's cap and a state saved
//! inside the span.

use super::*;
use crate::Decay;

const S: i128 = 1_000_000_000;

fn cfg(k: usize) -> KMeansCfg {
    KMeansCfg {
        n_features: 2,
        k,
        decay: Decay::Halflife(200.0),
        min_weight: 0.0,
        warm_rows: 6,
        seed_rule: SeedRule::Lloyd,
        seed: 3,
        update_every_rows: 1,
        split_merge: 0.5,
        split_merge_every_rows: 20,
        dead_frac: 0.05,
        standardize: true,
        scale_floor: 0.1,
        warm_clock: None,
        update_every: None,
        split_merge_every: None,
    }
}

/// Three blobs, a cluster appearing late, rows interleaved.
fn rows(n: usize, seed: u64) -> Vec<[f64; 2]> {
    let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    let mut u = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (s >> 11) as f64 / (1u64 << 53) as f64
    };
    (0..n)
        .map(|i| {
            let c = match i % 4 {
                0 | 1 => [0.0, 0.0],
                2 => [6.0, 1.0],
                _ if i > n / 2 => [-4.0, 7.0],
                _ => [1.0, 5.0],
            };
            [c[0] + u() - 0.5, c[1] + u() - 0.5]
        })
        .collect()
}

/// Every output of a row, as bits.
fn bits(m: &mut KMeans, x: &[f64; 2]) -> Vec<u64> {
    let out = m.step(x, &[], 1.0, 1.0);
    let mut b: Vec<u64> = out.pred.iter().map(|v| v.to_bits()).collect();
    b.push(out.n_eff.to_bits());
    b
}

/// Run `a` unstamped and `b` stamped on a clock of one unit a row (the
/// row clock, or one second a row in nanoseconds), each row's outputs
/// equal to the bit.
fn agree(a: KMeansCfg, b: KMeansCfg, n: usize, what: &str) {
    for temporal in [false, true] {
        let (mut ma, mut mb) = (
            KMeans::new(a.clone()).unwrap(),
            KMeans::new(b.clone()).unwrap(),
        );
        for (i, x) in rows(n, 5).iter().enumerate() {
            let t = i as i128 + 1;
            if temporal {
                mb.stamp_next(Stamp::Ns(t * S));
            } else {
                mb.stamp_next_on_rows(Stamp::from(t as f64));
            }
            assert_eq!(
                bits(&mut ma, x),
                bits(&mut mb, x),
                "{what}, temporal {temporal}, row {i}"
            );
            assert_eq!(
                ma.clusters.is_empty(),
                mb.clusters.is_empty(),
                "{what}, row {i}"
            );
            assert_eq!(
                (ma.since, ma.since_sm),
                (mb.since, mb.since_sm),
                "{what}, row {i}"
            );
        }
        assert!(!mb.clusters.is_empty(), "{what}: seeded");
    }
}

#[test]
fn a_warm_span_of_d_is_d_plus_one_rows_on_a_regular_clock() {
    for d in [5.0, 40.0] {
        let rows = KMeansCfg {
            warm_rows: d as usize + 1,
            ..cfg(3)
        };
        let span = KMeansCfg {
            warm_rows: 3,
            warm_clock: Some(d),
            ..cfg(3)
        };
        agree(rows, span, 300, &format!("warm span {d}"));
    }
}

#[test]
fn a_cadence_of_n_is_n_rows_on_a_regular_clock() {
    for n in [1u32, 3, 7] {
        let rows = KMeansCfg {
            update_every_rows: n,
            split_merge_every_rows: 4 * n,
            ..cfg(3)
        };
        let clock = KMeansCfg {
            update_every_rows: u32::MAX,
            split_merge_every_rows: u32::MAX,
            update_every: Some(f64::from(n)),
            split_merge_every: Some(f64::from(4 * n)),
            ..cfg(3)
        };
        agree(rows, clock, 300, &format!("cadence {n}"));
    }
    // `0` is every row, as `update_every_rows = 1` is.
    let every = KMeansCfg {
        update_every_rows: u32::MAX,
        update_every: Some(0.0),
        ..cfg(3)
    };
    agree(cfg(3), every, 120, "cadence 0");
}

/// A warm-up's span closes at its length to the nanosecond and not before;
/// one that has run with fewer than `k` rows waits for the `k`-th.
#[test]
fn the_warm_span_closes_at_its_length_and_not_a_nanosecond_before() {
    let c = KMeansCfg {
        warm_rows: 3,
        warm_clock: Some(10.0),
        ..cfg(3)
    };
    let data = rows(8, 6);
    for (last, seeded) in [(10 * S - 1, false), (10 * S, true)] {
        let mut m = KMeans::new(c.clone()).unwrap();
        for (i, x) in data[..5].iter().enumerate() {
            m.stamp_next(Stamp::Ns(if i == 4 { last } else { i as i128 * S }));
            m.step(x, &[], 1.0, 1.0);
        }
        assert_eq!(!m.clusters.is_empty(), seeded, "last at {last}");
    }
    let mut few = KMeans::new(c).unwrap();
    for (i, x) in data[..3].iter().enumerate() {
        few.stamp_next(Stamp::Ns([0, 20, 21][i] * S));
        few.step(x, &[], 1.0, 1.0);
        assert_eq!(!few.clusters.is_empty(), i == 2, "row {i}");
    }
}

/// A cadence on the clock fires at its length after the seeding, and not a
/// nanosecond before.
#[test]
fn a_clock_cadence_fires_at_its_length_and_not_a_nanosecond_before() {
    let c = KMeansCfg {
        warm_rows: 3,
        update_every_rows: u32::MAX,
        update_every: Some(5.0),
        ..cfg(3)
    };
    let data = rows(8, 7);
    for (last, fired) in [(7 * S - 1, false), (7 * S, true)] {
        let mut m = KMeans::new(c.clone()).unwrap();
        // Seeded at the third row, stamped 2 s; the next row 5 s after it.
        for (i, x) in data[..4].iter().enumerate() {
            m.stamp_next(Stamp::Ns(if i == 3 { last } else { i as i128 * S }));
            m.step(x, &[], 1.0, 1.0);
        }
        assert!(!m.clusters.is_empty());
        assert_eq!(m.since == 0, fired, "last at {last}");
    }
}

/// The warm-up's rows are the data's: past half of its cap a warning, once;
/// the row that would pass it is not learned, nor any after it.
#[test]
fn the_warm_span_is_warned_of_at_half_its_cap_and_stopped_at_it() {
    let c = KMeansCfg {
        warm_rows: 3,
        warm_clock: Some(1e12),
        ..cfg(3)
    };
    let mut m = KMeans::new(c).unwrap();
    let row = m.warm_bytes(2) - m.warm_bytes(1);
    let cap = m.warm_bytes(100);
    m.set_cap_for_test(cap);
    let mut warned = 0;
    for (i, x) in rows(150, 8).iter().enumerate() {
        m.stamp_next(Stamp::Ns(i as i128 * S));
        m.step(x, &[], 1.0, 1.0);
        if let Some((bytes, got)) = m.take_state_warning() {
            warned += 1;
            assert_eq!(got, cap);
            assert!(2 * bytes > cap && 2 * (bytes - row) <= cap, "{bytes}");
        }
    }
    assert_eq!(warned, 1);
    assert_eq!(m.state_over_cap(), Some((m.warm_bytes(101), cap)));
    assert_eq!(m.buf.len(), 100);
    assert!(m.clusters.is_empty());
}

/// A state saved inside the warm-up and between cadence events, through
/// the file's encoding, goes on to the bit; one with no span writes none of
/// its fields.
#[test]
fn a_state_saved_inside_the_span_goes_on_to_the_bit() {
    let c = KMeansCfg {
        warm_rows: 3,
        warm_clock: Some(30.0),
        update_every_rows: u32::MAX,
        update_every: Some(4.0),
        split_merge_every_rows: u32::MAX,
        split_merge_every: Some(25.0),
        ..cfg(3)
    };
    let data = rows(200, 9);
    for cut in [10, 60] {
        let mut a = KMeans::new(c.clone()).unwrap();
        for (i, x) in data[..cut].iter().enumerate() {
            a.stamp_next(Stamp::Ns(i as i128 * 700_000_000));
            a.step(x, &[], 1.0, 1.0);
        }
        let bytes = rmp_serde::to_vec_named(&a.state()).unwrap();
        let mut b = KMeans::restore(&rmp_serde::from_slice(&bytes).unwrap()).unwrap();
        for (i, x) in data[cut..].iter().enumerate() {
            let t = Stamp::Ns((cut + i) as i128 * 700_000_000);
            a.stamp_next(t);
            b.stamp_next(t);
            assert_eq!(bits(&mut a, x), bits(&mut b, x), "cut {cut}, row {i}");
        }
    }
    let plain = rmp_serde::to_vec_named(&KMeans::new(cfg(3)).unwrap().state()).unwrap();
    for f in [
        "warm_clock",
        "update_every",
        "split_merge_every",
        "place",
        "warm_first",
        "since_update",
        "since_split",
        "over",
        "warned",
    ] {
        let mut key = vec![0xa0 | f.len() as u8];
        key.extend_from_slice(f.as_bytes());
        assert!(!plain.windows(key.len()).any(|w| w == key), "{f}");
    }
}

#[test]
fn the_spans_are_checked() {
    for (c, words) in [
        (
            KMeansCfg {
                warm_clock: Some(0.0),
                ..cfg(3)
            },
            "warm_span must be a finite number",
        ),
        (
            KMeansCfg {
                warm_clock: Some(f64::INFINITY),
                ..cfg(3)
            },
            "warm_span must be a finite number",
        ),
        (
            KMeansCfg {
                update_every: Some(-1.0),
                ..cfg(3)
            },
            "update_every must be at least 0",
        ),
        (
            KMeansCfg {
                split_merge_every: Some(f64::NAN),
                ..cfg(3)
            },
            "split_merge_every must be at least 0",
        ),
    ] {
        let e = KMeans::new(c).unwrap_err();
        assert!(e.contains(words), "{e}");
    }
}
