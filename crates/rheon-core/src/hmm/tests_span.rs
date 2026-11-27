//! `hmm`'s warm-up over a span of the clock (`warm_clock`, docs/PLAN.md
//! task 538): `warm_rows = D + 1` on a clock of one unit a row, both sides
//! of the span's end, its cap as the rows come, and a state saved inside it.

use super::*;
use crate::{OnlineModel, Stamp};

const S: i128 = 1_000_000_000;

fn cfg(d: usize, k: usize) -> HmmCfg {
    HmmCfg {
        n_features: d,
        k,
        decay: Decay::Halflife(300.0),
        covariance: Covariance::Full,
        precision_prior: 1e-3,
        min_weight: 0.0,
        learn: true,
        transition_prior: 1.0,
        transition: None,
        means: None,
        covs: None,
        warm_rows: 20,
        seed_rule: SeedRule::Lloyd,
        seed: 7,
        tvtp: None,
        prior_rows: None,
        warm_clock: None,
        extras: None,
    }
}

/// Two blobs, visited in runs of `run` rows.
fn stream(n: usize, seed: u64, run: usize) -> Vec<Vec<f64>> {
    let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    let mut u = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (s >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    };
    (0..n)
        .map(|i| {
            let c = if (i / run).is_multiple_of(2) {
                -3.0
            } else {
                3.0
            };
            vec![c + u(), c + u()]
        })
        .collect()
}

fn bits(m: &mut Hmm, x: &[f64]) -> Vec<u64> {
    let out = m.step(x, &[], 1.0, 1.0);
    let mut b: Vec<u64> = out.pred.iter().map(|v| v.to_bits()).collect();
    b.push(out.n_eff.to_bits());
    b
}

#[test]
fn a_warm_span_of_d_is_d_plus_one_rows_on_a_regular_clock() {
    for covariance in [Covariance::Full, Covariance::Diagonal] {
        for d in [3.0, 30.0] {
            let rows = HmmCfg {
                warm_rows: d as usize + 1,
                covariance,
                ..cfg(2, 2)
            };
            let span = HmmCfg {
                warm_rows: 2,
                warm_clock: Some(d),
                ..rows.clone()
            };
            for temporal in [false, true] {
                let (mut a, mut b) = (
                    Hmm::new(rows.clone()).unwrap(),
                    Hmm::new(span.clone()).unwrap(),
                );
                for (i, x) in stream(200, 3, 9).iter().enumerate() {
                    let t = i as i128 + 1;
                    if temporal {
                        b.stamp_next(Stamp::Ns(t * S));
                    } else {
                        b.stamp_next_on_rows(Stamp::from(t as f64));
                    }
                    assert_eq!(
                        bits(&mut a, x),
                        bits(&mut b, x),
                        "{covariance:?} {d} {temporal} row {i}"
                    );
                    assert_eq!(a.seeded, b.seeded, "row {i}");
                }
                assert!(b.seeded);
            }
        }
    }
}

#[test]
fn the_warm_span_closes_at_its_length_and_not_a_nanosecond_before() {
    let c = HmmCfg {
        warm_rows: 2,
        warm_clock: Some(10.0),
        ..cfg(2, 2)
    };
    let data = stream(8, 4, 2);
    for (last, seeded) in [(10 * S - 1, false), (10 * S, true)] {
        let mut m = Hmm::new(c.clone()).unwrap();
        for (i, x) in data[..5].iter().enumerate() {
            m.stamp_next(Stamp::Ns(if i == 4 { last } else { i as i128 * S }));
            m.step(x, &[], 1.0, 1.0);
        }
        assert_eq!(m.seeded, seeded, "last at {last}");
    }
    // Three states: past its span with two rows, it waits for the third.
    let mut few = Hmm::new(HmmCfg {
        k: 3,
        warm_rows: 3,
        ..c
    })
    .unwrap();
    for (i, x) in data[..3].iter().enumerate() {
        few.stamp_next(Stamp::Ns([0, 20, 21][i] * S));
        few.step(x, &[], 1.0, 1.0);
        assert_eq!(few.seeded, i == 2, "row {i}");
    }
}

#[test]
fn the_warm_span_is_warned_of_at_half_its_cap_and_stopped_at_it() {
    let c = HmmCfg {
        warm_rows: 2,
        warm_clock: Some(1e12),
        ..cfg(2, 2)
    };
    let mut m = Hmm::new(c).unwrap();
    let row = m.warm_bytes(2) - m.warm_bytes(1);
    let cap = m.warm_bytes(100);
    m.set_cap_for_test(cap);
    let mut warned = 0;
    for (i, x) in stream(150, 5, 7).iter().enumerate() {
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
    assert_eq!(m.buffer.len(), 100);
    assert!(!m.seeded);
}

#[test]
fn a_state_saved_inside_the_span_goes_on_to_the_bit() {
    let c = HmmCfg {
        warm_rows: 2,
        warm_clock: Some(25.0),
        ..cfg(2, 2)
    };
    let data = stream(120, 6, 8);
    let mut a = Hmm::new(c).unwrap();
    for x in &data[..10] {
        a.step(x, &[], 0.9, 1.0);
    }
    let bytes = rmp_serde::to_vec_named(&a.state()).unwrap();
    let mut b = Hmm::restore(&rmp_serde::from_slice(&bytes).unwrap()).unwrap();
    for (i, x) in data[10..].iter().enumerate() {
        let (p, q) = (a.step(x, &[], 0.9, 1.0), b.step(x, &[], 0.9, 1.0));
        let bits = |s: &crate::Step| s.pred.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&p), bits(&q), "row {i}");
    }
    assert!(b.seeded);
    let plain = rmp_serde::to_vec_named(&Hmm::new(cfg(2, 2)).unwrap().state()).unwrap();
    for f in ["warm_clock", "place", "warm_first", "over", "warned"] {
        let mut key = vec![0xa0 | f.len() as u8];
        key.extend_from_slice(f.as_bytes());
        assert!(!plain.windows(key.len()).any(|w| w == key), "{f}");
    }
}

#[test]
fn the_warm_span_is_checked() {
    for d in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        let e = Hmm::new(HmmCfg {
            warm_clock: Some(d),
            ..cfg(2, 2)
        })
        .unwrap_err();
        assert!(e.contains("warm_span must be a finite number"), "{e}");
    }
}

// ---- mutants-w11: the warm-up's cap, asked of its own rows only ----

/// The cap holds the rows a warm-up over a span buffers, and no other: a model
/// with no span, or one already seeded, learns on under a cap of one byte as
/// a free one does, to the bit.
#[test]
fn the_warm_up_cap_is_asked_of_a_span_that_has_not_seeded_only() {
    let data = stream(80, 3, 9);
    let controlled = |c: HmmCfg, seeded_at: usize| {
        let (mut capped, mut free) = (Hmm::new(c.clone()).unwrap(), Hmm::new(c).unwrap());
        for (i, x) in data.iter().enumerate() {
            if i == seeded_at {
                capped.set_cap_for_test(1);
            }
            for m in [&mut capped, &mut free] {
                m.stamp_next(Stamp::Ns(i as i128 * S));
            }
            assert_eq!(bits(&mut capped, x), bits(&mut free, x), "row {i}");
        }
        assert_eq!(capped.state_over_cap(), None);
    };
    // No span: unseeded rows are buffered for 20 rows, with a cap of one byte.
    controlled(cfg(2, 2), 0);
    // A span that has seeded.
    controlled(
        HmmCfg {
            warm_rows: 2,
            warm_clock: Some(4.0),
            ..cfg(2, 2)
        },
        30,
    );
}

/// A row refused at the cap leaves `n_eff` and every other field where it stood:
/// refusal is atomic.
#[test]
fn a_refused_warm_up_row_leaves_n_eff_unchanged() {
    let c = HmmCfg {
        warm_rows: 2,
        warm_clock: Some(1e12),
        decay: Decay::Lam(0.9),
        ..cfg(2, 2)
    };
    let mut m = Hmm::new(c).unwrap();
    m.set_cap_for_test(m.warm_bytes(100));
    for (i, x) in stream(5, 4, 3).iter().enumerate() {
        m.stamp_next(Stamp::Ns(i as i128 * S));
        m.step(x, &[], 1.0, 1.0);
    }
    let before = m.n_eff();
    assert!(before > 1.0);
    m.set_cap_for_test(1);
    m.stamp_next(Stamp::Ns(6 * S));
    m.step(&[0.5, 0.5], &[], 1.0, 1.0);
    assert!(m.state_over_cap().is_some());
    assert_eq!(m.n_eff().to_bits(), before.to_bits());
    assert_eq!(m.buffer.len(), 5);
}

/// The warning's flag is state: saved after the warning, it restores raised.
#[test]
fn a_warned_flag_survives_a_save() {
    let c = HmmCfg {
        warm_rows: 2,
        warm_clock: Some(1e12),
        ..cfg(2, 2)
    };
    let mut m = Hmm::new(c).unwrap();
    m.set_cap_for_test(m.warm_bytes(100));
    for (i, x) in stream(70, 5, 7).iter().enumerate() {
        m.stamp_next(Stamp::Ns(i as i128 * S));
        m.step(x, &[], 1.0, 1.0);
    }
    assert!(m.warned && m.state_over_cap().is_none());
    let bytes = rmp_serde::to_vec_named(&m.state()).unwrap();
    let key = b"\xa6warned";
    assert!(bytes.windows(key.len()).any(|w| w == key));
    let back = Hmm::restore(&rmp_serde::from_slice(&bytes).unwrap()).unwrap();
    assert!(back.warned);
}

/// The warm-up's cap is `kmeans`' budget in bytes, and its rows are counted
/// by `kmeans`' own width.
#[test]
fn the_warm_up_budget_is_counted_in_bytes() {
    let m = Hmm::new(cfg(2, 2)).unwrap();
    assert_eq!(m.warm_cap(), 256 * 1024 * 1024);
    assert_eq!(
        m.warm_bytes(1000),
        (1000 * (std::mem::size_of::<Vec<f64>>() + 3 * 8)) as u64
    );
}

/// STATE-3: an `hmm` warm-up span whose buffered rows exceed its cap (or a
/// non-span state recording `over` or `warm_first`) is refused on load.
#[test]
fn state_3_a_warm_span_past_its_cap_is_refused_on_load() {
    let mut m = Hmm::new(HmmCfg {
        warm_rows: 2,
        warm_clock: Some(1e12),
        ..cfg(2, 2)
    })
    .unwrap();
    for (i, x) in stream(6, 17, 3).iter().enumerate() {
        m.stamp_next(Stamp::Ns(i as i128 * S));
        m.step(x, &[], 1.0, 1.0);
    }
    assert_eq!(m.buffer.len(), 6);
    m.set_cap_for_test(m.warm_bytes(6));
    assert!(Hmm::restore(&m.state()).is_ok());

    m.set_cap_for_test(m.warm_bytes(5));
    match Hmm::restore(&m.state()) {
        Err(crate::StateError::Invalid(e)) => assert!(e.contains("warm-up cap"), "{e}"),
        other => panic!("over cap: {other:?}"),
    }

    let plain = Hmm::new(cfg(2, 2)).unwrap();
    for (name, edit) in [
        ("over", (|h: &mut Hmm| h.over = Some(1)) as fn(&mut Hmm)),
        ("warm_first", |h: &mut Hmm| {
            h.warm_first = Some(Stamp::from(0.0));
        }),
    ] {
        let mut damaged = plain.clone();
        edit(&mut damaged);
        assert!(Hmm::restore(&damaged.state()).is_err(), "{name}");
    }
}
