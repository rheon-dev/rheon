//! `ewm_moments`' window read through two stacks (PLAN 554, part 5): every
//! read against a fresh fold of the window's rows, across flips, a window
//! inside one block and a clock gap that empties several blocks at once;
//! a state saved anywhere goes on to the bit; and the time a row at windows
//! of 1,000 and 10,000 rows. The block merge the stacks replaced measured
//! 12.7 to 214 times slower on the same streams (the module docs).

use super::*;
use crate::OnlineModel;

/// Skew and kurtosis of `x` and the correlation of `x` with `y`.
fn window_terms() -> Vec<MomentTerm> {
    vec![
        term("s3", 3, MomentKind::Standardized, true),
        term("k4", 4, MomentKind::Standardized, true),
        MomentTerm {
            name: "xy".into(),
            powers: vec![(Leg::of(0), 1), (Leg::of(1), 1)],
            kind: MomentKind::Standardized,
            bias: true,
        },
    ]
}

/// `n` rows of `x` at a level of `1e3` with a spread of about 1, `y` tied
/// to it, weights in `[1/4, 5/4)` (0 on every 37th row), the clock stepping
/// `[1/2, 3/2)` and `gap` on every 101st row.
fn window_rows(seed: u64, n: usize, gap: f64) -> Vec<(f64, f64, f64, f64)> {
    let mut s = seed;
    (0..n)
        .map(|i| {
            let (u, v, q, c) = (lcg(&mut s), lcg(&mut s), lcg(&mut s), lcg(&mut s));
            let x = 1e3 + 4.0 * (u - 0.5) * (u - 0.5) * (u - 0.5) + (v - 0.5);
            let w = if i % 37 == 36 { 0.0 } else { 0.25 + q };
            let d = if i % 101 == 100 { gap } else { 0.5 + c };
            (x, 0.5 * x + q, w, d)
        })
        .collect()
}

fn window_cfg(rows: Option<usize>, clock: Option<f64>) -> EwmMomentsCfg {
    EwmMomentsCfg {
        max_state_bytes: Some(64 << 20),
        clocked: clock.is_some(),
        window_rows: rows,
        window_clock: clock,
        ..EwmMomentsCfg::new(Decay::Halflife(25.0), window_terms())
    }
}

/// The worst relative gap of every output to a fresh left fold of the
/// window's rows (relative to the fold's size, or to 1 below it), each
/// withheld where the other is.
fn gap_to_fold(m: &EwmMoments) -> f64 {
    let r = m.ring.as_ref().unwrap();
    let end = r.first + (r.vals.len() / m.stride()) as u64;
    let mut fold = m.fold(r.first, end).sums;
    fold.pop();
    let want = m.read_sums(&fold);
    let mut worst = 0.0f64;
    for (g, w) in m.read().iter().zip(&want) {
        match (g, w) {
            (Ok(g), Ok(w)) => worst = worst.max(gap(*g, *w)),
            (g, w) => assert_eq!(g.is_ok(), w.is_ok(), "{g:?} against {w:?}"),
        }
    }
    worst
}

/// Windows of 1, 2, 3, 7 and 50 rows (blocks of 1, 1, 1, 2 and 7 rows),
/// and clock windows of 0.75, 3 and 40 with a gap of 30 every 101 rows
/// (which empties the window but for the row, or takes several blocks out
/// at once): after every row the window's outputs are a fresh fold's
/// within 1e-12, and a state saved every 13th row goes on to the bit.
/// Measured 2026-11-23: at most 2.9e-15.
#[test]
fn the_two_stacks_read_the_windows_rows() {
    let rows = window_rows(5545, 700, 30.0);
    let mut forms: Vec<EwmMomentsCfg> = [1, 2, 3, 7, 50]
        .iter()
        .map(|&w| window_cfg(Some(w), None))
        .collect();
    forms.extend([0.75, 3.0, 40.0].iter().map(|&c| window_cfg(None, Some(c))));
    let mut worst = 0.0f64;
    for cfg in forms {
        let mut whole = EwmMoments::new(cfg.clone()).unwrap();
        let mut saved = EwmMoments::new(cfg.clone()).unwrap();
        for (i, &(x, y, w, d)) in rows.iter().enumerate() {
            let a = whole.step(&[x, y], &[], d, w);
            if i % 13 == 6 {
                let bytes = rmp_serde::to_vec_named(&saved.state()).unwrap();
                let s: crate::State = rmp_serde::from_slice(&bytes).unwrap();
                saved = EwmMoments::restore(&s).unwrap();
            }
            let b = saved.step(&[x, y], &[], d, w);
            let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&a.pred), bits(&b.pred), "{cfg:?}, row {i}");
            let g = gap_to_fold(&whole);
            assert!(g < 1e-12, "{cfg:?}, row {i}: {g:e}");
            worst = worst.max(g);
        }
        assert_eq!(saved.state(), whole.state());
    }
    println!("worst gap to a fresh fold: {worst:.1e}");
}

/// Nanoseconds a step at windows of 1,000 and 10,000 rows, on rows and on
/// a clock (no zero-weight rows, no gaps), the minimum of three rounds
/// after a warm-up of three windows, and the gap to a fresh fold at the
/// end.
#[test]
#[ignore = "a measurement: cargo test --release -p rheon-core window_timing -- --ignored --nocapture"]
fn window_timing() {
    for w in [1_000usize, 10_000] {
        for clocked in [false, true] {
            let (warm, timed) = (3 * w, 20_000);
            let rows: Vec<_> = window_rows(554 + w as u64, warm + timed, 1.0)
                .into_iter()
                .map(|(x, y, _, d)| (x, y, 1.0, d))
                .collect();
            let cfg = match clocked {
                false => window_cfg(Some(w), None),
                true => window_cfg(None, Some(w as f64)),
            };
            let (mut best, mut g) = (f64::INFINITY, 0.0);
            for _ in 0..3 {
                let mut m = EwmMoments::new(cfg.clone()).unwrap();
                for &(x, y, wt, d) in &rows[..warm] {
                    m.step(&[x, y], &[], d, wt);
                }
                let t0 = std::time::Instant::now();
                for &(x, y, wt, d) in &rows[warm..] {
                    std::hint::black_box(m.step(&[x, y], &[], d, wt));
                }
                best = best.min(t0.elapsed().as_nanos() as f64 / timed as f64);
                g = gap_to_fold(&m);
            }
            let on = if clocked { "on a clock" } else { "rows" };
            println!("window {w} {on}: {best:.0} ns a row; gap to a fresh fold {g:.1e}");
        }
    }
}
