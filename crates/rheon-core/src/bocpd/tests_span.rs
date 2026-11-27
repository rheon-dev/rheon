//! `bocpd`'s warm-up over a span of the clock (`warm_clock`, docs/PLAN.md
//! task 538): `warm_rows = D + 1` on a clock of one unit a row, both sides
//! of the span's end, its cap as the rows come, and a state saved inside it.

use super::tests::{Normals, cfg};
use super::*;
use crate::{OnlineModel, Stamp};

const S: i128 = 1_000_000_000;

/// The prior from the first rows: a warm-up.
fn warming(d: usize) -> BocpdCfg {
    BocpdCfg {
        prior_mean: None,
        prior_scale: None,
        prior_nu: None,
        ..cfg(d)
    }
}

/// A level that moves twice, with noise.
fn rows(n: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut g = Normals::new(seed);
    (0..n)
        .map(|i| {
            let level = if i < n / 3 {
                0.0
            } else if i < 2 * n / 3 {
                3.0
            } else {
                -1.0
            };
            vec![level + g.normal(), 0.5 * level + g.normal()]
        })
        .collect()
}

fn bits(m: &mut Bocpd, x: &[f64]) -> Vec<u64> {
    let out = m.step(x, &[], 1.0, 1.0);
    let mut b: Vec<u64> = out.pred.iter().map(|v| v.to_bits()).collect();
    b.push(out.n_eff.to_bits());
    b
}

#[test]
fn a_warm_span_of_d_is_d_plus_one_rows_on_a_regular_clock() {
    for emission in [BocpdEmission::Diag, BocpdEmission::Gaussian] {
        for d in [3.0, 25.0] {
            let rows_cfg = BocpdCfg {
                emission,
                warm_rows: Some(d as usize + 1),
                ..warming(2)
            };
            let span = BocpdCfg {
                warm_rows: None,
                warm_clock: Some(d),
                ..rows_cfg.clone()
            };
            for temporal in [false, true] {
                let (mut a, mut b) = (
                    Bocpd::new(rows_cfg.clone()).unwrap(),
                    Bocpd::new(span.clone()).unwrap(),
                );
                for (i, x) in rows(120, 3).iter().enumerate() {
                    let t = i as i128 + 1;
                    if temporal {
                        b.stamp_next(Stamp::Ns(t * S));
                    } else {
                        b.stamp_next_on_rows(Stamp::from(t as f64));
                    }
                    assert_eq!(
                        bits(&mut a, x),
                        bits(&mut b, x),
                        "{emission:?} {d} {temporal} row {i}"
                    );
                    assert_eq!(a.warming(), b.warming(), "row {i}");
                }
                assert!(!b.warming());
            }
        }
    }
}

#[test]
fn the_warm_span_closes_at_its_length_and_not_a_nanosecond_before() {
    let c = BocpdCfg {
        warm_clock: Some(10.0),
        ..warming(2)
    };
    let data = rows(8, 4);
    for (last, warm) in [(10 * S - 1, true), (10 * S, false)] {
        let mut m = Bocpd::new(c.clone()).unwrap();
        for (i, x) in data[..4].iter().enumerate() {
            m.stamp_next(Stamp::Ns(if i == 3 { last } else { i as i128 * S }));
            m.step(x, &[], 1.0, 1.0);
        }
        assert_eq!(m.warming(), warm, "last at {last}");
    }
    // Past its span with one row, it waits for the second.
    let mut few = Bocpd::new(c).unwrap();
    for (i, x) in data[..2].iter().enumerate() {
        few.stamp_next(Stamp::Ns([0, 20][i] * S));
        if i == 1 {
            // The first row's span has run; one row is no variance.
            assert!(few.warming());
        }
        few.step(x, &[], 1.0, 1.0);
    }
    assert!(!few.warming());
}

#[test]
fn the_warm_span_is_warned_of_at_half_its_cap_and_stopped_at_it() {
    let c = BocpdCfg {
        warm_clock: Some(1e12),
        ..warming(2)
    };
    let mut m = Bocpd::new(c).unwrap();
    let row = m.warm_bytes(1);
    let cap = m.warm_bytes(100);
    m.set_cap_for_test(cap);
    let mut warned = 0;
    for (i, x) in rows(150, 5).iter().enumerate() {
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
    assert_eq!(m.warm.len(), 100);
    assert!(m.warming());
}

#[test]
fn a_state_saved_inside_the_span_goes_on_to_the_bit() {
    let c = BocpdCfg {
        warm_clock: Some(12.0),
        ..warming(2)
    };
    let data = rows(80, 6);
    let mut a = Bocpd::new(c).unwrap();
    for x in &data[..6] {
        a.step(x, &[], 0.75, 1.0);
    }
    assert!(a.warming());
    let bytes = rmp_serde::to_vec_named(&a.state()).unwrap();
    let mut b = Bocpd::restore(&rmp_serde::from_slice(&bytes).unwrap()).unwrap();
    for (i, x) in data[6..].iter().enumerate() {
        let (p, q) = (a.step(x, &[], 0.75, 1.0), b.step(x, &[], 0.75, 1.0));
        let bits = |s: &crate::Step| s.pred.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&p), bits(&q), "row {i}");
    }
    assert!(!b.warming());
    let plain = rmp_serde::to_vec_named(&Bocpd::new(warming(2)).unwrap().state()).unwrap();
    for f in ["warm_clock", "place", "warm_first", "over", "warned"] {
        let mut key = vec![0xa0 | f.len() as u8];
        key.extend_from_slice(f.as_bytes());
        assert!(!plain.windows(key.len()).any(|w| w == key), "{f}");
    }
}

#[test]
fn the_warm_span_is_checked() {
    let refuse = |c: BocpdCfg, words: &str| {
        let e = Bocpd::new(c).unwrap_err();
        assert!(e.contains(words), "{e}");
    };
    for d in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        refuse(
            BocpdCfg {
                warm_clock: Some(d),
                ..warming(2)
            },
            "warm_span must be a finite number",
        );
    }
    refuse(
        BocpdCfg {
            warm_clock: Some(5.0),
            warm_rows: Some(10),
            ..warming(2)
        },
        "two warm-ups",
    );
    refuse(
        BocpdCfg {
            warm_clock: Some(5.0),
            ..cfg(2)
        },
        "warm_span sets the prior from the first rows",
    );
}

/// STATE-3: a `bocpd` warm-up span whose buffered rows exceed its cap (or a
/// non-span state recording `over` or `warm_first`) is refused on load.
#[test]
fn state_3_a_warm_span_past_its_cap_is_refused_on_load() {
    let mut m = Bocpd::new(BocpdCfg {
        warm_clock: Some(1e12),
        ..warming(2)
    })
    .unwrap();
    for (i, x) in rows(6, 17).iter().enumerate() {
        m.stamp_next(Stamp::Ns(i as i128 * S));
        m.step(x, &[], 1.0, 1.0);
    }
    assert_eq!(m.warm.len(), 6);
    m.set_cap_for_test(m.warm_bytes(6));
    assert!(Bocpd::restore(&m.state()).is_ok());

    m.set_cap_for_test(m.warm_bytes(5));
    match Bocpd::restore(&m.state()) {
        Err(crate::StateError::Invalid(e)) => assert!(e.contains("warm-up cap"), "{e}"),
        other => panic!("over cap: {other:?}"),
    }

    let plain = Bocpd::new(warming(2)).unwrap();
    for (name, edit) in [
        ("over", (|b: &mut Bocpd| b.over = Some(1)) as fn(&mut Bocpd)),
        ("warm_first", |b: &mut Bocpd| {
            b.warm_first = Some(Stamp::from(0.0));
        }),
    ] {
        let mut damaged = plain.clone();
        edit(&mut damaged);
        assert!(Bocpd::restore(&damaged.state()).is_err(), "{name}");
    }
}
