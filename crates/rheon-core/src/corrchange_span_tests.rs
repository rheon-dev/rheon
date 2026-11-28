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
        emit_n_flagged: false,
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
    for d in [-1.0, f64::NAN] {
        let e = CorrChange::new(CorrChangeCfg {
            permute_every: Some(d),
            ..cfg()
        })
        .unwrap_err();
        assert!(e.contains("permute_every must be at least 0"), "{e}");
    }
}

/// Clock-window cap check: zero-weight rows do not breach at cap, `dropped_at`
/// uses `2.0 * span` (not `2.0 + span` or `0`), and `ring.len() - dropped_at + monitoring + 1`
/// accounts for evicted rows cleanly.
#[test]
fn window_clock_cap_zero_weight_and_dropped_at_span_edges() {
    let base = CorrChangeCfg {
        kind: CorrChangeKind::Window,
        span_rows: 0,
        span_clock: Some(5.0),
        ..cfg()
    };
    let mut probe = CorrChange::new(base.clone()).unwrap();
    let cap = probe.held_bytes_for(4);
    let c = CorrChangeCfg {
        max_state_bytes: Some(cap),
        ..base
    };
    probe = CorrChange::new(c).unwrap();
    let data = rows(10, 42);
    // Fill 4 rows at t = 0, 1, 2, 3 (ring at cap = 4 rows).
    for (i, x) in data[..4].iter().enumerate() {
        probe.stamp_next(Stamp::Raw(i as f64, 0.0));
        probe.step(x, &[], 1.0, 1.0);
        assert_eq!(probe.state_over_cap(), None);
    }
    assert_eq!(probe.ring.len(), 4);

    // 1. Zero-weight row at cap must NOT breach (`weight > 0.0`, not `>=`).
    let mut zero = probe.clone();
    zero.stamp_next(Stamp::Raw(4.0, 0.0));
    zero.step(&data[4], &[], 1.0, 0.0);
    assert_eq!(zero.state_over_cap(), None);

    // 2. Positive-weight row at t = 8.0: `2.0 * span = 10.0` drops 0 rows
    //    (whereas `2.0 + span = 7.0` would drop 2 rows, and `+ monitoring` -> `*` would give 1).
    //    Must breach `cap`!
    let mut still_in = probe.clone();
    still_in.stamp_next(Stamp::Raw(8.0, 0.0));
    still_in.step(&data[4], &[], 5.0, 1.0);
    assert!(
        still_in.state_over_cap().is_some(),
        "at t=8.0 < 2*5=10.0 no row leaves, so 5th row breaches cap"
    );

    // 3. Positive-weight row at t = 11.0: `11 - 0 >= 10` and `11 - 1 >= 10` -> 2 rows leave,
    //    `rows = 4 - 2 + 0 + 1 = 3 <= 4` -> ADMITTED!
    let mut after_drop = probe;
    after_drop.stamp_next(Stamp::Raw(11.0, 0.0));
    after_drop.step(&data[4], &[], 8.0, 1.0);
    assert_eq!(
        after_drop.state_over_cap(),
        None,
        "at t=11.0 >= 2*5=10.0 two rows leave so row fits"
    );
}
