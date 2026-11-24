//! The STL ring: its embargo's defaults, its bytes and their refusal, what
//! each `closed` reads, and that missing rows count in no lag.

use super::*;
use crate::stl::StlCfg;

fn stl(period: usize, seasonal: usize) -> Decomposition {
    Decomposition::Stl(Stl::new(&StlCfg::new(period, seasonal)).unwrap())
}

fn mstl(seasons: &[(usize, usize)]) -> Decomposition {
    Decomposition::Mstl(Mstl::new(seasons, &StlCfg::new(2, 3), 2).unwrap())
}

/// A series with a season of 4 and of 6, a trend and a few jumps, from
/// arithmetic alone.
fn series(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            [1.5, -0.5, 2.0, -3.0][i % 4]
                + [0.3, -0.1, 0.7, 0.2, -0.6, -0.5][i % 6]
                + i as f64 * 0.05
                + ((i * 37) % 11) as f64 * 0.125
        })
        .collect()
}

#[test]
fn the_embargo_is_the_seasonal_reach() {
    // (p, s) -> h = p (s - 1) / 2, W = 2 h + 1; the design's table.
    for (seasons, h, w) in [
        (vec![(24, 7)], 72, 145),
        (vec![(168, 7)], 504, 1009),
        (vec![(1440, 7)], 4320, 8641),
        (vec![(24, 11), (168, 15)], 1176, 2353),
    ] {
        assert_eq!(default_lag(&seasons, Closed::Right), h);
        assert_eq!(default_lag(&seasons, Closed::Left), h + 1);
        assert_eq!(default_window(&seasons), w);
    }
    // Not half the trend span: STL(24, 7)'s trend is 47, whose (47 - 1) / 2 is 23.
    let op = StlOperator::new(stl(24, 7), None, None, Closed::Left).unwrap();
    assert_eq!((op.window(), op.lag(), op.position()), (145, 73, 72));
    let op = StlOperator::new(stl(24, 7), None, None, Closed::Right).unwrap();
    assert_eq!((op.window(), op.lag(), op.position()), (145, 72, 72));
}

#[test]
fn a_rings_bytes_are_8w_plus_24_and_past_the_cap_are_refused() {
    assert_eq!(state_bytes(145), 1_184);
    assert_eq!(state_bytes(2_353), 18_848);
    assert_eq!(state_bytes(141_121), 1_128_992);
    assert!(check_state_bytes("mstl", 2_353, 256 * 1024).is_ok());
    let err = check_state_bytes("mstl", 141_121, 256 * 1024).unwrap_err();
    for part in ["mstl", "141121 rows", "1128992 bytes", "262144"] {
        assert!(err.contains(part), "{err}");
    }
    // At the cap exactly: held.
    assert!(check_state_bytes("stl", 100, state_bytes(100)).is_ok());
    assert!(check_state_bytes("stl", 101, state_bytes(100)).is_err());
}

#[test]
fn the_ring_holds_the_last_w_rows_oldest_first() {
    let mut r = StlRing::new(4);
    assert!(r.is_empty() && r.last_clock().is_nan());
    for i in 0..3 {
        r.push(i as f64, 10.0 + i as f64);
    }
    assert_eq!((r.len(), r.is_full()), (3, false));
    assert_eq!(r.window(), vec![0.0, 1.0, 2.0]);
    for i in 3..7 {
        r.push(i as f64, 10.0 + i as f64);
    }
    assert!(r.is_full());
    assert_eq!(r.window(), vec![3.0, 4.0, 5.0, 6.0]);
    assert_eq!(r.last_clock(), 16.0);
    r.clear();
    assert!(r.is_empty() && r.window().is_empty() && r.last_clock().is_nan());
}

/// Runs an operator over a series with `None` for missing rows.
fn run(op: &StlOperator, xs: &[Option<f64>]) -> Vec<Option<Components>> {
    let mut ring = op.ring();
    xs.iter()
        .enumerate()
        .map(|(t, &x)| op.step(&mut ring, x, t as f64))
        .collect()
}

#[test]
fn right_emits_the_window_ending_at_the_row() {
    let y = series(60);
    let s = Stl::new(&StlCfg::new(4, 7)).unwrap();
    let op = StlOperator::new(
        Decomposition::Stl(s.clone()),
        Some(24),
        Some(5),
        Closed::Right,
    )
    .unwrap();
    let out = run(&op, &y.iter().map(|&v| Some(v)).collect::<Vec<_>>());
    for t in 0..60 {
        if t + 1 < 24 {
            assert!(out[t].is_none(), "row {t} before the ring is full");
            continue;
        }
        let f = s.fit(&y[t + 1 - 24..=t]);
        let c = out[t].as_ref().unwrap();
        // Position W - 1 - lag = 18: row t - 5.
        assert_eq!(
            (c.seasonal[0], c.trend, c.resid),
            (f.seasonal[18], f.trend[18], f.resid[18])
        );
    }
}

#[test]
fn left_emits_the_window_before_the_row_and_never_reads_it() {
    let y = series(60);
    let s = Stl::new(&StlCfg::new(4, 7)).unwrap();
    let op = StlOperator::new(
        Decomposition::Stl(s.clone()),
        Some(24),
        Some(5),
        Closed::Left,
    )
    .unwrap();
    let out = run(&op, &y.iter().map(|&v| Some(v)).collect::<Vec<_>>());
    for t in 0..60 {
        if t < 24 {
            assert!(out[t].is_none(), "row {t}: the ring before it is not full");
            continue;
        }
        let f = s.fit(&y[t - 24..t]);
        let c = out[t].as_ref().unwrap();
        // Position W - lag = 19 of the window before row t: row t - 5.
        assert_eq!(
            (c.seasonal[0], c.trend, c.resid),
            (f.seasonal[19], f.trend[19], f.resid[19])
        );
    }
    // Row t's own value, and every row after it, move nothing row t emits.
    for t in [30, 45] {
        let mut z: Vec<Option<f64>> = y.iter().map(|&v| Some(v)).collect();
        for v in z.iter_mut().skip(t) {
            *v = Some(1e6);
        }
        assert_eq!(run(&op, &z)[..=t], out[..=t]);
    }
}

#[test]
fn right_never_reads_a_row_after_its_own() {
    let y = series(60);
    let op = StlOperator::new(stl(4, 7), Some(24), Some(0), Closed::Right).unwrap();
    let base = run(&op, &y.iter().map(|&v| Some(v)).collect::<Vec<_>>());
    let mut z: Vec<Option<f64>> = y.iter().map(|&v| Some(v)).collect();
    for v in z.iter_mut().skip(41) {
        *v = Some(1e6);
    }
    assert_eq!(run(&op, &z)[..=40], base[..=40]);
}

#[test]
fn a_missing_row_counts_in_no_lag() {
    let y = series(70);
    for closed in [Closed::Left, Closed::Right] {
        let op = StlOperator::new(mstl(&[(4, 7), (6, 7)]), Some(30), Some(6), closed).unwrap();
        let dense = run(&op, &y.iter().map(|&v| Some(v)).collect::<Vec<_>>());
        // The same rows with a missing row after every fifth.
        let mut sparse = Vec::new();
        let mut at = Vec::new();
        for (i, &v) in y.iter().enumerate() {
            at.push(sparse.len());
            sparse.push(Some(v));
            if i % 5 == 4 {
                sparse.push(None);
            }
        }
        let out = run(&op, &sparse);
        for (i, &j) in at.iter().enumerate() {
            assert_eq!(out[j], dense[i], "{closed:?}: row {i}");
            if closed == Closed::Right && i % 5 == 4 {
                assert_eq!(out[j + 1], out[j], "a missing row repeats the row before");
            }
        }
    }
}

#[test]
fn mstl_emits_each_season() {
    let y = series(80);
    let m = Mstl::new(&[(6, 7), (4, 7)], &StlCfg::new(2, 3), 2).unwrap();
    let op = StlOperator::new(
        Decomposition::Mstl(m.clone()),
        Some(36),
        Some(9),
        Closed::Right,
    )
    .unwrap();
    let out = run(&op, &y.iter().map(|&v| Some(v)).collect::<Vec<_>>());
    let f = m.fit(&y[80 - 36..]);
    let c = out[79].as_ref().unwrap();
    assert_eq!(c.seasonal, vec![f.seasonal[0][26], f.seasonal[1][26]]);
    assert_eq!((c.trend, c.resid), (f.trend[26], f.resid[26]));
}

#[test]
fn refusals_name_the_operator_and_the_bound() {
    let e = StlOperator::new(stl(24, 7), Some(47), None, Closed::Left).unwrap_err();
    assert!(e.contains("stl") && e.contains("48"), "{e}");
    let e = StlOperator::new(mstl(&[(4, 7), (6, 7)]), Some(12), Some(1), Closed::Left).unwrap_err();
    assert!(e.contains("mstl") && e.contains("period 6"), "{e}");
    let e = StlOperator::new(stl(4, 7), Some(24), Some(0), Closed::Left).unwrap_err();
    assert!(e.contains("lag 0") && e.contains("from 1 to 24"), "{e}");
    let e = StlOperator::new(stl(4, 7), Some(24), Some(24), Closed::Right).unwrap_err();
    assert!(e.contains("from 0 to 23"), "{e}");
    assert!(StlOperator::new(stl(4, 7), Some(24), Some(24), Closed::Left).is_ok());
}
