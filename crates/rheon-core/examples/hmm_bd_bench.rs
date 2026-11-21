//! Study C4 (Bd): `hmm`'s time a row under today's ridge and under
//! stabilized forgetting (`Hmm::with_stabilized`), over a seeded model.
//!
//!     cargo run --release -p rheon-core --example hmm_bd_bench -- \
//!         VARIANT COV D HALF_LIFE [ROWS] [REPS]
//!
//! `VARIANT` is `A` (today), `Y` (stabilized, prior worth `D` rows, its
//! factor form moving every coordinate at once), `Y32` or `Y64` (`Y` at a
//! tolerance of 1/32 or 1/64), `F` (the axis form, a coordinate at a time;
//! `F64` at 1/64) or `E` (the exact recursion through co-moments); with the
//! `study-c4` feature, `U` (the axis form with the cursor that holds its
//! target on average, tolerance 1/2) and `Z` (`Y` with `ŝ` fixed);
//! `COV` is `full` or `diag`. Two states, a steady two-regime stream with
//! regimes in runs of about 20 rows. The model is seeded on 200 rows (not
//! timed); each rep steps a copy of it over `ROWS` rows (default 10,000),
//! and the best of `REPS` (default 7) is printed as ns a row, one JSON line.
//! On a commit without stabilized forgetting, keep the `A` arm alone.

use rheon_core::{Covariance, Decay, Hmm, HmmCfg, OnlineModel, SeedRule, Step};
use std::time::Instant;

fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*state >> 11) as f64 / (1u64 << 53) as f64
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let variant = a[1].as_str();
    let cov = Covariance::parse(&a[2]).unwrap();
    let d: usize = a[3].parse().unwrap();
    let hl: f64 = a[4].parse().unwrap();
    let rows: usize = a.get(5).map_or(10_000, |r| r.parse().unwrap());
    let reps: usize = a.get(6).map_or(7, |r| r.parse().unwrap());
    let cfg = HmmCfg {
        n_features: d,
        k: 2,
        decay: Decay::Halflife(hl),
        covariance: cov,
        precision_prior: 1e-3,
        min_weight: 0.0,
        learn: true,
        transition_prior: 1.0,
        transition: None,
        means: None,
        covs: None,
        warm_rows: 200,
        seed_rule: SeedRule::Kmeanspp,
        seed: 7,
        tvtp: None,
    };
    let mut m = match variant {
        "A" => Hmm::new(cfg).unwrap(),
        "F" => {
            let mut m = Hmm::with_stabilized(cfg, d as f64).unwrap();
            m.set_stabilized_axis();
            m
        }
        "Y" => Hmm::with_stabilized(cfg, d as f64).unwrap(),
        "Y32" | "Y64" | "F64" => {
            let mut m = Hmm::with_stabilized(cfg, d as f64).unwrap();
            if variant == "F64" {
                m.set_stabilized_axis();
            }
            m.set_stabilized_tau(if variant == "Y32" {
                1.0 / 32.0
            } else {
                1.0 / 64.0
            });
            m
        }
        "E" => Hmm::with_stabilized_comoments(cfg, d as f64).unwrap(),
        #[cfg(feature = "study-c4")]
        "U" => {
            let mut m = Hmm::with_stabilized(cfg, d as f64).unwrap();
            m.set_stabilized_axis();
            m.set_stabilized_tau(0.5);
            m.set_stabilized_study(false, true, false);
            m
        }
        #[cfg(feature = "study-c4")]
        "Z" => {
            let mut m = Hmm::with_stabilized(cfg, d as f64).unwrap();
            m.set_stabilized_study(false, false, true);
            m
        }
        _ => panic!("variant {variant}"),
    };
    let mut s = 11u64;
    let mut state = 0usize;
    let mut row = |s: &mut u64| -> Vec<f64> {
        if lcg(s) < 0.05 {
            state ^= 1;
        }
        let c = if state == 0 {
            0.0
        } else {
            3.0 / (d as f64).sqrt()
        };
        // A sum of four uniforms: near Gaussian, variance 1/3.
        (0..d)
            .map(|_| c + (lcg(s) + lcg(s) + lcg(s) + lcg(s) - 2.0))
            .collect()
    };
    let mut step = Step::default();
    for _ in 0..200 {
        let x = row(&mut s);
        m.step_into(&x, &[], 1.0, 1.0, &mut step);
    }
    assert!(m.seeded());
    let data: Vec<Vec<f64>> = (0..rows).map(|_| row(&mut s)).collect();
    let mut best = f64::INFINITY;
    let mut check = 0.0;
    for _ in 0..reps {
        let mut c = m.clone();
        let t = Instant::now();
        for x in &data {
            c.step_into(x, &[], 1.0, 1.0, &mut step);
        }
        best = best.min(t.elapsed().as_secs_f64());
        check = step.pred[5] + c.solve_failures as f64;
    }
    println!(
        "{{\"variant\": \"{variant}\", \"cov\": \"{}\", \"d\": {d}, \"hl\": {hl}, \"ns_per_row\": {:.1}, \"last\": {check}}}",
        a[2],
        best * 1e9 / rows as f64
    );
}
