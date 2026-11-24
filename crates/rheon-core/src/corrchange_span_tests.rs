//! `corrchange`'s redraw on the clock (`permute_every`, docs/PLAN.md task
//! 538): `permute_every_rows = n` on a clock of one unit a row, both sides
//! of the cadence's length, a state saved between two draws, the refusal.

use super::*;
use crate::{OnlineModel, Stamp};

const S: i128 = 1_000_000_000;

fn cfg() -> CorrChangeCfg {
    CorrChangeCfg {
        n_features: 2,
        kind: CorrChangeKind::Window,
        span_rows: 20,
        alpha: 0.05,
        alpha_adjust: "bonferroni".into(),
        bandwidth: None,
        scalar: false,
        decay: Decay::Halflife(f64::INFINITY),
        crit: None,
        n_perm: 40,
        permute_every_rows: 25,
        perm_block: 2,
        norm: ChangeNorm::L1,
        seed: 3,
        reset: false,
        monitor_rows: 0,
        boundary_gamma: 0.0,
        permute_every: None,
        span_clock: None,
        monitor_clock: None,
        max_state_bytes: None,
    }
}

/// Pairs whose correlation moves at a third and at two thirds.
fn rows(n: usize, seed: u64) -> Vec<[f64; 2]> {
    let mut g = SplitMix64::new(seed);
    let mut u = || ((g.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
    (0..n)
        .map(|i| {
            let rho = [0.0, 0.8, -0.5][3 * i / n];
            let (a, b) = (u() - 0.5, u() - 0.5);
            [a, rho * a + (1.0 - rho * rho).sqrt() * b]
        })
        .collect()
}

fn bits(m: &mut CorrChange, x: &[f64; 2]) -> Vec<u64> {
    let out = m.step(x, &[], 1.0, 1.0);
    let mut b: Vec<u64> = out.pred.iter().map(|v| v.to_bits()).collect();
    b.push(m.perm_crit.unwrap_or(f64::NAN).to_bits());
    b
}

#[test]
fn a_redraw_every_n_is_every_n_reports_on_a_regular_clock() {
    for n in [1usize, 4, 15] {
        let rows_cfg = CorrChangeCfg {
            permute_every_rows: n,
            ..cfg()
        };
        let clock = CorrChangeCfg {
            permute_every_rows: usize::MAX,
            permute_every: Some(n as f64),
            ..cfg()
        };
        for temporal in [false, true] {
            let (mut a, mut b) = (
                CorrChange::new(rows_cfg.clone()).unwrap(),
                CorrChange::new(clock.clone()).unwrap(),
            );
            for (i, x) in rows(240, 5).iter().enumerate() {
                let t = i as i128 + 1;
                if temporal {
                    b.stamp_next(Stamp::Ns(t * S));
                } else {
                    b.stamp_next_on_rows(Stamp::from(t as f64));
                }
                assert_eq!(
                    bits(&mut a, x),
                    bits(&mut b, x),
                    "every {n}, temporal {temporal}, row {i}"
                );
            }
        }
    }
}

/// The redraw falls on the row at the cadence's length after the draw, and
/// not on one a nanosecond short of it.
#[test]
fn a_redraw_falls_at_its_length_and_not_a_nanosecond_before() {
    let c = CorrChangeCfg {
        permute_every_rows: usize::MAX,
        permute_every: Some(5.0),
        ..cfg()
    };
    let data = rows(60, 6);
    // The ring is full at row 40 (two windows of 20), drawn then at 39 s.
    for (last, redrawn) in [(44 * S - 1, false), (44 * S, true)] {
        let mut m = CorrChange::new(c.clone()).unwrap();
        for (i, x) in data[..40].iter().enumerate() {
            m.stamp_next(Stamp::Ns(i as i128 * S));
            m.step(x, &[], 1.0, 1.0);
        }
        assert!(m.perm_crit.is_some() && m.since_perm == 0);
        m.stamp_next(Stamp::Ns(last));
        m.step(&data[40], &[], 1.0, 1.0);
        assert_eq!(m.since_perm == 0, redrawn, "last at {last}");
    }
}

#[test]
fn a_state_saved_between_two_draws_goes_on_to_the_bit() {
    let c = CorrChangeCfg {
        permute_every_rows: usize::MAX,
        permute_every: Some(7.5),
        ..cfg()
    };
    let data = rows(200, 7);
    let mut a = CorrChange::new(c).unwrap();
    for (i, x) in data[..50].iter().enumerate() {
        a.stamp_next(Stamp::Ns(i as i128 * 900_000_000));
        a.step(x, &[], 1.0, 1.0);
    }
    let bytes = rmp_serde::to_vec_named(&a.state()).unwrap();
    let mut b = CorrChange::restore(&rmp_serde::from_slice(&bytes).unwrap()).unwrap();
    for (i, x) in data[50..].iter().enumerate() {
        let t = Stamp::Ns((50 + i) as i128 * 900_000_000);
        a.stamp_next(t);
        b.stamp_next(t);
        assert_eq!(bits(&mut a, x), bits(&mut b, x), "row {i}");
    }
    let plain = rmp_serde::to_vec_named(&CorrChange::new(cfg()).unwrap().state()).unwrap();
    for f in ["permute_every", "since_draw"] {
        let mut key = vec![0xa0 | f.len() as u8];
        key.extend_from_slice(f.as_bytes());
        assert!(!plain.windows(key.len()).any(|w| w == key), "{f}");
    }
}

#[test]
fn the_redraw_cadence_is_checked() {
    // `inf` too, which never comes round (review 2026-11-23, SPEC-4).
    for d in [-1.0, f64::NAN, f64::INFINITY] {
        let e = CorrChange::new(CorrChangeCfg {
            permute_every: Some(d),
            ..cfg()
        })
        .unwrap_err();
        assert!(
            e.contains("permute_every must be a finite number of clock units, at least 0"),
            "{e}"
        );
    }
    for d in [0.0, 5.0] {
        CorrChange::new(CorrChangeCfg {
            permute_every: Some(d),
            ..cfg()
        })
        .unwrap();
    }
}
