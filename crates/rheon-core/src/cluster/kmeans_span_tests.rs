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
            "update_every must be a finite number of clock units, at least 0",
        ),
        (
            KMeansCfg {
                split_merge_every: Some(f64::NAN),
                ..cfg(3)
            },
            "split_merge_every must be a finite number of clock units, at least 0",
        ),
        // `inf` never comes round (review 2026-11-23, SPEC-4).
        (
            KMeansCfg {
                update_every: Some(f64::INFINITY),
                ..cfg(3)
            },
            "update_every must be a finite number of clock units, at least 0",
        ),
        (
            KMeansCfg {
                split_merge_every: Some(f64::INFINITY),
                ..cfg(3)
            },
            "split_merge_every must be a finite number of clock units, at least 0",
        ),
    ] {
        let e = KMeans::new(c).unwrap_err();
        assert!(e.contains(words), "{e}");
    }
}

/// The row clock's stamps (`stamp_next_on_rows`) are the clock the spans
/// read, whatever their spacing: on a clock of 2.5 a row the span of 10 closes
/// at the fifth row and not at the fourth, and a cadence of 5 fires two rows
/// after the seeding and not one. Unstamped, a row is one unit and neither
/// moves, so only a stamp the model reads tells them apart.
#[test]
fn the_row_clock_is_read_at_its_own_spacing() {
    let data = rows(12, 12);
    let warm = KMeansCfg {
        warm_rows: 3,
        warm_clock: Some(10.0),
        ..cfg(3)
    };
    let mut m = KMeans::new(warm).unwrap();
    for (i, x) in data[..5].iter().enumerate() {
        m.stamp_next_on_rows(Stamp::from(i as f64 * 2.5));
        m.step(x, &[], 1.0, 1.0);
        assert_eq!(!m.clusters.is_empty(), i == 4, "row {i}");
    }
    let cadence = KMeansCfg {
        warm_rows: 3,
        update_every_rows: u32::MAX,
        update_every: Some(5.0),
        ..cfg(3)
    };
    let mut m = KMeans::new(cadence).unwrap();
    let mut fired = Vec::new();
    for (i, x) in data[..8].iter().enumerate() {
        m.stamp_next_on_rows(Stamp::from(i as f64 * 2.5));
        m.step(x, &[], 1.0, 1.0);
        fired.push(!m.clusters.is_empty() && m.since == 0);
    }
    // Seeded at row 2; the cadence's 5 units are rows 3 and 4 after it.
    assert_eq!(
        fired,
        [false, false, true, false, true, false, true, false],
        "{fired:?}"
    );
}

/// The warm-up's byte count and cap, and the shape checks of a saved state
/// that the cadences and the span add, each alone.
#[test]
fn the_warm_up_is_counted_and_a_saved_state_is_held_to_its_cfg() {
    let m = KMeans::new(cfg(3)).unwrap();
    assert_eq!(m.warm_cap(), 256 * 1024 * 1024);
    assert_eq!(
        m.warm_bytes(1000),
        (1000 * (std::mem::size_of::<Vec<f64>>() + 3 * 8)) as u64
    );

    let spans = KMeansCfg {
        warm_clock: Some(1e9),
        update_every_rows: u32::MAX,
        update_every: Some(5.0),
        split_merge_every_rows: u32::MAX,
        split_merge_every: Some(25.0),
        ..cfg(3)
    };
    let mut full = KMeans::new(spans).unwrap();
    for (i, x) in rows(10, 13).iter().enumerate() {
        full.stamp_next(Stamp::Ns(i as i128 * S));
        full.step(x, &[], 1.0, 1.0);
    }
    let mut plain = KMeans::new(cfg(3)).unwrap();
    for x in rows(40, 13).iter() {
        plain.step(x, &[], 1.0, 1.0);
    }
    assert!(full.since_update.is_some() && full.since_split.is_some());
    let damaged = |m: &KMeans, damage: fn(&mut KMeans)| {
        let mut s = m.state();
        let crate::ModelState::KMeans(inner) = &mut s.model else {
            unreachable!()
        };
        damage(inner);
        KMeans::restore(&s)
    };
    for m in [&full, &plain] {
        assert!(damaged(m, |_| {}).is_ok());
    }
    let wrong_shape = |r: Result<KMeans, StateError>, what: &str| match r {
        Err(StateError::Invalid(e)) => assert!(e.contains("wrong shape"), "{what}: {e}"),
        other => panic!("{what}: {:?}", other.map(|_| ())),
    };
    wrong_shape(damaged(&full, |m| m.place = None), "no place");
    wrong_shape(
        damaged(&plain, |m| m.place = Some(Default::default())),
        "a place",
    );
    wrong_shape(damaged(&full, |m| m.since_update = None), "no update clock");
    wrong_shape(damaged(&full, |m| m.since_split = None), "no split clock");
    wrong_shape(
        damaged(&plain, |m| m.since_update = full_since()),
        "an update clock",
    );
    wrong_shape(
        damaged(&plain, |m| m.since_split = full_since()),
        "a split clock",
    );
}

fn full_since() -> Option<crate::since::Since> {
    Some(Default::default())
}

/// The warning's flag is state: a model saved after its half-of-the-cap
/// warning restores with it raised, so it does not warn a second time.
#[test]
fn a_warned_flag_survives_a_save() {
    let c = KMeansCfg {
        warm_rows: 3,
        warm_clock: Some(1e12),
        ..cfg(3)
    };
    let mut m = KMeans::new(c).unwrap();
    m.set_cap_for_test(m.warm_bytes(100));
    for (i, x) in rows(60, 8).iter().enumerate() {
        m.stamp_next(Stamp::Ns(i as i128 * S));
        m.step(x, &[], 1.0, 1.0);
    }
    assert!(m.warned && m.state_over_cap().is_none());
    let bytes = rmp_serde::to_vec_named(&m.state()).unwrap();
    let key = b"\xa6warned";
    assert!(bytes.windows(key.len()).any(|w| w == key));
    let back = KMeans::restore(&rmp_serde::from_slice(&bytes).unwrap()).unwrap();
    assert!(back.warned);
}

/// The warm-up's cap is asked of a row that would be learned into the
/// warm-up, and of no other: a row of no weight takes no room, and once the
/// model is seeded its buffer is gone and no cap holds a row back.
#[test]
fn the_warm_up_cap_is_asked_of_learned_warm_up_rows_only() {
    let c = KMeansCfg {
        warm_rows: 3,
        warm_clock: Some(4.0),
        ..cfg(3)
    };
    // A row of no weight, with a cap of one byte: nothing is held, nothing
    // is over it.
    let mut m = KMeans::new(c.clone()).unwrap();
    m.set_cap_for_test(1);
    m.stamp_next(Stamp::Ns(0));
    m.step(&[0.5, 0.5], &[], 1.0, 0.0);
    assert_eq!(m.state_over_cap(), None);
    // The same row with weight is the one past the cap.
    let mut m = KMeans::new(c.clone()).unwrap();
    m.set_cap_for_test(1);
    m.stamp_next(Stamp::Ns(0));
    m.step(&[0.5, 0.5], &[], 1.0, 1.0);
    assert!(m.state_over_cap().is_some());

    // A seeded model learns on, whatever cap is set after the seeding.
    let data = rows(60, 14);
    let (mut capped, mut free) = (KMeans::new(c.clone()).unwrap(), KMeans::new(c).unwrap());
    for (i, x) in data.iter().enumerate() {
        if i == 20 {
            assert!(!capped.clusters.is_empty());
            capped.set_cap_for_test(1);
        }
        capped.stamp_next(Stamp::Ns(i as i128 * S));
        free.stamp_next(Stamp::Ns(i as i128 * S));
        assert_eq!(bits(&mut capped, x), bits(&mut free, x), "row {i}");
    }
    assert_eq!(capped.state_over_cap(), None);
}
