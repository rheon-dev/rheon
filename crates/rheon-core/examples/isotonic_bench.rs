//! A learned row's cost in `Isotonic`, by the distinct feature values its
//! fit holds (no window, the values repeating) and by a row window's
//! length (a continuous feature), with and without a decay: the figures of
//! the CHANGELOG's entry (task 573). The minimum of five runs of 20,000
//! rows each, after a warm-up that fills the fit.
//!
//! `cargo run --release -p rheon-core --example isotonic_bench`

use std::hint::black_box;
use std::time::Instant;

use rheon_core::{Decay, Isotonic, IsotonicCfg, IsotonicOutOfBounds, OnlineModel, WindowClosed};

fn lcg(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*s >> 11) as f64 / (1u64 << 53) as f64
}

fn cfg(window_rows: Option<usize>, decays: bool) -> IsotonicCfg {
    IsotonicCfg {
        n_targets: 1,
        increasing: true,
        y_min: None,
        y_max: None,
        out_of_bounds: IsotonicOutOfBounds::Clip,
        min_weight: 0.0,
        window_rows,
        window_clock: None,
        closed: WindowClosed::Right,
        decay: Decay::Halflife(if decays { 2000.0 } else { f64::INFINITY }),
        max_state_bytes: Some(64 << 20),
    }
}

/// Nanoseconds a row over `rows` rows of a feature on `levels` values
/// (continuous where 0), after `warm` rows.
fn time(c: IsotonicCfg, levels: usize, warm: usize, rows: usize) -> f64 {
    let mut s = 573u64;
    let row = |s: &mut u64| {
        let u = lcg(s);
        let x = if levels == 0 {
            u
        } else {
            (u * levels as f64).floor() / levels as f64
        };
        (x, x * x + 0.3 * (lcg(s) - 0.5))
    };
    let mut best = f64::INFINITY;
    for _ in 0..5 {
        let mut m = Isotonic::new(c.clone()).unwrap();
        for _ in 0..warm {
            let (x, y) = row(&mut s);
            m.step(&[x], &[Some(y)], 1.0, 1.0);
        }
        let start = Instant::now();
        for _ in 0..rows {
            let (x, y) = row(&mut s);
            black_box(m.step(black_box(&[x]), &[Some(y)], 1.0, 1.0));
        }
        best = best.min(start.elapsed().as_nanos() as f64 / rows as f64);
    }
    best
}

fn main() {
    for decays in [false, true] {
        for levels in [10, 100, 1_000, 8_000] {
            let ns = time(cfg(None, decays), levels, 20 * levels, 20_000);
            println!("pooled  decay={decays:5} distinct={levels:6}  {ns:9.1} ns a row");
        }
        for n in [100, 1_000, 8_000] {
            let ns = time(cfg(Some(n), decays), 0, 2 * n, 20_000);
            println!("window  decay={decays:5} rows={n:10}  {ns:9.1} ns a row");
        }
    }
}
