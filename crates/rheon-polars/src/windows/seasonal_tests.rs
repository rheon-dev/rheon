//! The STL operators through the window core: each row's value is the
//! decomposition of its group's window, read at the lag; a missing row
//! counts in no lag; a session change empties the ring; the state checks
//! and round-trips to the byte at every row; a ring past its cap is
//! refused when the run is built.

use super::super::{RowIn, Windows};
use super::*;
use rheon_core::{ClockCfg, ClockValue, Mstl, Stl, StlCfg};

fn def(seasons: &[(u32, u32)], window_rows: u32, lag: u32, closed: RingClosed) -> StlDef {
    StlDef {
        seasons: seasons.to_vec(),
        iterate: 2,
        trend: None,
        low_pass: None,
        seasonal_deg: 1,
        trend_deg: 1,
        low_pass_deg: 1,
        seasonal_jump: 1,
        trend_jump: 1,
        low_pass_jump: 1,
        robust: false,
        inner_iter: None,
        outer_iter: None,
        window_rows,
        lag,
        closed,
        component: StlComponent::Trend,
        input: 0,
        cap: 256 * 1024,
    }
}

fn core(defs: Vec<StlDef>) -> Windows {
    let cfg = ClockCfg {
        gap_cap: 1e9,
        ..ClockCfg::default()
    };
    let mut c = Windows::with_seasonals(
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        defs,
        cfg,
    )
    .unwrap();
    c.set_grouped(true);
    c
}

fn value(k: usize) -> f64 {
    [1.5, -0.5, 2.0, -3.0][k % 4] + [0.3, -0.1, 0.7, 0.2, -0.6, -0.5][k % 6] + k as f64 * 0.05
}

/// Feeds rows `(clock, group, value, session)`, checking the state and
/// round-tripping it at every row; returns each operator's column.
fn feed(c: &mut Windows, rows: &[(f64, usize, f64, Option<u64>)], n: usize) -> Vec<Vec<f64>> {
    let mut out = vec![Vec::new(); n];
    for &(t, g, x, session) in rows {
        let gi = c.group(Some(&format!("g{g}")));
        c.push(&RowIn {
            group: gi,
            clock: Some(ClockValue::F64(t)),
            session,
            values: &[x],
            accept: true,
            lists: &[],
        })
        .unwrap();
        c.check().unwrap();
        let bytes = rmp_serde::to_vec_named(&*c).unwrap();
        let back: Windows = rmp_serde::from_slice(&bytes).unwrap();
        back.check().unwrap();
        assert_eq!(rmp_serde::to_vec_named(&back).unwrap(), bytes);
        for (o, col) in c.drain().values.iter().enumerate() {
            out[o].extend(col);
        }
    }
    for (o, col) in c.finish().values.iter().enumerate() {
        out[o].extend(col);
    }
    out
}

fn same(got: f64, want: f64) -> bool {
    got.to_bits() == want.to_bits() || (got.is_nan() && want.is_nan())
}

#[test]
fn each_row_reads_its_groups_window_at_the_lag() {
    let left = def(&[(4, 7)], 24, 5, RingClosed::Left);
    let right = StlDef {
        closed: RingClosed::Right,
        component: StlComponent::Seasonal(0),
        ..def(&[(4, 7)], 24, 5, RingClosed::Right)
    };
    let mstl = StlDef {
        component: StlComponent::Seasonal(1),
        ..def(&[(6, 7), (4, 7)], 30, 6, RingClosed::Left)
    };
    let mut c = core(vec![left, right, mstl]);
    // Two groups interleaved, a missing value in each now and then.
    let mut rows = Vec::new();
    let mut seen = [Vec::new(), Vec::new()];
    for k in 0..140usize {
        let g = usize::from(k % 3 == 0);
        let x = if k % 11 == 7 { f64::NAN } else { value(k) };
        rows.push((k as f64, g, x, None));
        seen[g].push(x);
    }
    let out = feed(&mut c, &rows, 3);
    let stl = Stl::new(&StlCfg::new(4, 7)).unwrap();
    let m = Mstl::new(&[(6, 7), (4, 7)], &StlCfg::new(4, 7), 2).unwrap();
    let mut learned = [Vec::new(), Vec::new()];
    for (i, &(_, g, x, _)) in rows.iter().enumerate() {
        let before = learned[g].clone();
        if !x.is_nan() {
            learned[g].push(x);
        }
        let after = &learned[g];
        // Left: the 24 learned rows before this one, at position 19.
        let want = if before.len() >= 24 {
            stl.fit(&before[before.len() - 24..]).trend[19]
        } else {
            f64::NAN
        };
        assert!(
            same(out[0][i], want),
            "row {i} left: {} against {want}",
            out[0][i]
        );
        // Right: the 24 learned rows ending here, at position 18.
        let want = if after.len() >= 24 {
            stl.fit(&after[after.len() - 24..]).seasonal[18]
        } else {
            f64::NAN
        };
        assert!(
            same(out[1][i], want),
            "row {i} right: {} against {want}",
            out[1][i]
        );
        // MSTL's second season (period 6), left, window 30, lag 6.
        let want = if before.len() >= 30 {
            m.fit(&before[before.len() - 30..]).seasonal[1][24]
        } else {
            f64::NAN
        };
        assert!(
            same(out[2][i], want),
            "row {i} mstl: {} against {want}",
            out[2][i]
        );
    }
}

#[test]
fn a_session_change_empties_the_ring() {
    let mut c = core(vec![def(&[(4, 7)], 16, 1, RingClosed::Left)]);
    let rows: Vec<_> = (0..60usize)
        .map(|k| (k as f64, 0, value(k), Some(u64::from(k >= 30))))
        .collect();
    let out = feed(&mut c, &rows, 1);
    // Rows 30 to 45: the new session's ring is not yet full.
    assert!(out[0][16..30].iter().all(|v| !v.is_nan()));
    assert!(out[0][30..46].iter().all(|v| v.is_nan()));
    let stl = Stl::new(&StlCfg::new(4, 7)).unwrap();
    let ys: Vec<f64> = (30..46).map(value).collect();
    assert!(same(out[0][46], stl.fit(&ys).trend[15]));
}

#[test]
fn a_ring_past_its_cap_is_refused_when_the_run_is_built() {
    let big = StlDef {
        cap: 1_000,
        ..def(&[(4, 7)], 200, 1, RingClosed::Left)
    };
    let cfg = ClockCfg::default();
    let err = Windows::with_seasonals(
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        vec![big],
        cfg,
    )
    .unwrap_err();
    assert!(
        err.contains("stl") && err.contains("1624 bytes") && err.contains("1000"),
        "{err}"
    );
    let bad = StlDef {
        component: StlComponent::Seasonal(1),
        ..def(&[(4, 7)], 24, 1, RingClosed::Left)
    };
    assert!(bad.check().unwrap_err().contains("seasonal component 1"));
}
