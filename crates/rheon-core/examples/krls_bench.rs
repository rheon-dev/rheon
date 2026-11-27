//! Arm64 release per-row cost (ABBA) and exact state bytes for `Krls`:
//! - ALD mode (`M = 50, 100`, `k = 5, 20`, `lam = 1.0` Givens `O(m^2)` and
//!   `lam < 1` Cholesky `O(m^3 / 3)`)
//! - Sliding-window mode (`W = 50, 100`, `k = 5`, Givens `O(W^2)`)
//!
//! `cargo run --release -p rheon-core --example krls_bench`

use std::hint::black_box;
use std::time::Instant;

use rheon_core::{Decay, KernelKind, Krls, KrlsCfg, OnlineModel, WindowClosed};

fn lcg(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

fn fill_row(s: &mut u64, x: &mut [f64]) -> f64 {
    for v in x.iter_mut() {
        *v = lcg(s) * 1.5;
    }
    libm::sin(x[0]) + 0.5 * x[1] * x[1] + 0.05 * lcg(s)
}

fn cfg_ald(k: usize, m: usize, decays: bool) -> KrlsCfg {
    KrlsCfg {
        n_features: k,
        n_targets: 1,
        fit_intercept: true,
        kernel: KernelKind::Rbf,
        gamma: None,
        degree: None,
        coef0: None,
        ridge: 0.1,
        max_atoms: Some(m),
        ald_threshold: Some(0.0),
        window_rows: None,
        window_clock: None,
        closed: WindowClosed::Right,
        decay: if decays {
            Decay::Lam(0.99)
        } else {
            Decay::Halflife(f64::INFINITY)
        },
        min_weight: 0.0,
        emit_n_atoms: false,
        max_state_bytes: Some(64 << 20),
    }
}

fn cfg_window(k: usize, w: usize) -> KrlsCfg {
    KrlsCfg {
        n_features: k,
        n_targets: 1,
        fit_intercept: true,
        kernel: KernelKind::Rbf,
        gamma: None,
        degree: None,
        coef0: None,
        ridge: 0.1,
        max_atoms: None,
        ald_threshold: None,
        window_rows: Some(w),
        window_clock: None,
        closed: WindowClosed::Right,
        decay: Decay::Halflife(f64::INFINITY),
        min_weight: 0.0,
        emit_n_atoms: false,
        max_state_bytes: Some(64 << 20),
    }
}

fn bench_case(label: &str, cfg: KrlsCfg, warm: usize, rows: usize) {
    let k = cfg.n_features;
    let mut s = 502u64;
    let mut xs = vec![vec![0.0; k]; warm + rows];
    let mut ys = vec![0.0; warm + rows];
    for i in 0..warm + rows {
        ys[i] = fill_row(&mut s, &mut xs[i]);
    }
    let mut samples_ns = Vec::with_capacity(4);
    let mut final_bytes = 0u64;
    let mut final_atoms = 0usize;
    for _ in 0..4 {
        let mut m = Krls::new(cfg.clone()).unwrap();
        let mut step = rheon_core::Step::default();
        for i in 0..warm {
            m.step_into(&xs[i], &[Some(ys[i])], 1.0, 1.0, &mut step);
        }
        let start = Instant::now();
        for i in warm..warm + rows {
            m.step_into(black_box(&xs[i]), &[Some(ys[i])], 1.0, 1.0, &mut step);
            black_box(step.pred[0]);
        }
        let ns = start.elapsed().as_nanos() as f64 / rows as f64;
        samples_ns.push(ns);
        final_bytes = m.bytes();
        final_atoms = m.n_atoms();
    }
    samples_ns.sort_by(f64::total_cmp);
    let median = 0.5 * (samples_ns[1] + samples_ns[2]);
    let min = samples_ns[0];
    println!(
        "{label:32}  median={median:8.1} ns/row ({:6.2} us)  min={min:8.1} ns/row  atoms={final_atoms:3}  state={final_bytes:7} B",
        median / 1000.0
    );
}

fn main() {
    for &k in &[5usize, 20] {
        for &m in &[50usize, 100] {
            bench_case(
                &format!("ald lam=1.0  k={k:2} M={m:3}"),
                cfg_ald(k, m, false),
                2 * m,
                2_000,
            );
            bench_case(
                &format!("ald lam=0.99 k={k:2} M={m:3}"),
                cfg_ald(k, m, true),
                2 * m,
                1_000,
            );
        }
    }
    for &w in &[50usize, 100] {
        bench_case(
            &format!("window       k= 5 W={w:3}"),
            cfg_window(5, w),
            2 * w,
            2_000,
        );
    }
}
