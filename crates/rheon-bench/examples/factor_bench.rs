//! Task 479's harness: the models that factor a `k x k` matrix on a learned
//! row, timed through `AnyModel::step_into` as the bank steps them, at the
//! release benchmark's settings and over a sweep of `k`, so a build's speed
//! and its crossover order can be read on each platform.
//!
//!     cargo run --release -p rheon-bench --example factor_bench -- [reps] [filter...]
//!
//! Prints one line per case: the best and the median over `reps` (default 7)
//! of the measured block, in ns a row and rows a second, and an FNV digest
//! of every output bit of the block (`pred` and `n_eff` of every row), which
//! says whether two builds give the same numbers. A filter keeps the cases
//! whose name contains any of the words given (`hmm`, `ew_class k=20`).
//! Each repetition builds the model afresh, steps the warm-up rows, then
//! times the block, so every repetition does the same work.
//!
//! The file uses only `rheon_polars`' public API and its own data, so it can
//! be copied into another checkout's `crates/rheon-bench/examples/` to time
//! that build on the same rows.
use std::hint::black_box;
use std::time::Instant;

use rheon_polars::{Spec, build_models};
use serde_json::{Value, json};

/// Knuth's MMIX LCG, the top 53 bits: the same bits on every platform.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    fn signed(&mut self) -> f64 {
        2.0 * self.next() - 1.0
    }
}

#[derive(Clone, Copy)]
enum Shape {
    /// Features around one of three centres six apart (clusters).
    Blobs,
    /// Features around one of three centres two apart, the centre's
    /// number as the target (a label).
    Labelled,
    /// Features in `[-1, 1)` and a target linear in them.
    Linear,
}

struct Rows {
    k: usize,
    m: usize,
    x: Vec<f64>,
    y: Vec<Option<f64>>,
    d: Vec<f64>,
}

impl Rows {
    fn generate(shape: Shape, k: usize, m: usize, n: usize, seed: u64) -> Self {
        let mut g = Lcg(seed);
        let (mut x, mut y, mut d) = (Vec::new(), Vec::new(), Vec::new());
        for i in 0..n {
            let row = x.len();
            match shape {
                Shape::Blobs => {
                    let label = (3.0 * g.next()).floor() as usize;
                    x.extend((0..k).map(|j| g.signed() + if j % 3 == label { 6.0 } else { 0.0 }));
                    y.extend((0..m).map(|_| Some(g.signed())));
                }
                Shape::Labelled => {
                    let label = (3.0 * g.next()).floor() as usize;
                    x.extend((0..k).map(|j| g.signed() + if j % 3 == label { 2.0 } else { 0.0 }));
                    y.extend((0..m).map(|_| Some(label as f64)));
                }
                Shape::Linear => {
                    x.extend((0..k).map(|_| g.signed()));
                    let fit: f64 = x[row..]
                        .iter()
                        .enumerate()
                        .map(|(j, v)| v / (j + 1) as f64)
                        .sum();
                    y.extend((0..m).map(|j| Some(0.5 + fit + j as f64 + 0.3 * g.signed())));
                }
            }
            d.push(if i == 0 {
                0.0
            } else {
                0.5 * (1.0 + (4.0 * g.next()).floor().min(3.0))
            });
        }
        Self { k, m, x, y, d }
    }

    fn row(&self, i: usize) -> (&[f64], &[Option<f64>], f64) {
        (
            &self.x[i * self.k..(i + 1) * self.k],
            &self.y[i * self.m..(i + 1) * self.m],
            self.d[i],
        )
    }
}

fn names(prefix: &str, n: usize) -> Vec<String> {
    (0..n).map(|j| format!("{prefix}{j}")).collect()
}

struct Case {
    name: String,
    spec: Value,
    shape: Shape,
    warm: usize,
    rows: usize,
}

fn cases() -> Vec<Case> {
    let mut out = Vec::new();
    let decayed = |model: Value, k: usize, target: bool| {
        let mut s = json!({
            "name": "m",
            "model": model,
            "features": names("x", k),
            "clock": "t",
            "half_life": 1000.0,
            "gap_cap": 100.0,
        });
        if target {
            s["targets"] = json!(["y0"]);
        }
        s
    };
    for k in [2, 4, 6, 8, 10, 12, 13, 16, 20, 32] {
        out.push(Case {
            name: format!("hmm k={k} K=2 full"),
            spec: decayed(
                json!({"type": "hmm", "n_components": 2, "precision_prior": 1.0}),
                k,
                false,
            ),
            shape: Shape::Blobs,
            warm: 2_000,
            rows: 20_000,
        });
    }
    for k in [2, 4, 6, 8, 10, 12, 13, 16, 20, 32] {
        out.push(Case {
            name: format!("ew_class k={k} full"),
            spec: decayed(
                json!({"type": "ew_class", "classes": ["0", "1", "2"], "precision_prior": 1.0}),
                k,
                true,
            ),
            shape: Shape::Labelled,
            warm: 2_000,
            rows: 20_000,
        });
    }
    for k in [2, 4, 8, 12, 13, 16, 20, 32] {
        out.push(Case {
            name: format!("ew_class k={k} tied"),
            spec: decayed(
                json!({"type": "ew_class", "classes": ["0", "1", "2"], "precision_prior": 1.0,
                       "covariance_type": "tied"}),
                k,
                true,
            ),
            shape: Shape::Labelled,
            warm: 2_000,
            rows: 20_000,
        });
    }
    for k in [2, 4, 6, 8, 10, 12, 13, 16, 20, 32] {
        out.push(Case {
            name: format!("ew_cov k={k} mahal"),
            spec: decayed(
                json!({"type": "ew_cov", "stats": ["mean", "mahal"], "precision_prior": 1.0,
                       "mahal_quantiles": [0.99]}),
                k,
                false,
            ),
            shape: Shape::Linear,
            warm: 2_000,
            rows: 20_000,
        });
    }
    for k in [2, 4, 6, 8, 12, 16] {
        out.push(Case {
            name: format!("bocpd k={k} full max_run=200"),
            spec: json!({
                "name": "m",
                "model": {"type": "bocpd", "covariance_type": "full",
                          "degrees_of_freedom_prior": (k as f64 + 2.0).max(6.0),
                          "prune_below": 1e-6, "max_run": 200},
                "features": names("x", k),
                "clock": "t",
                "gap_cap": 100.0,
            }),
            shape: Shape::Blobs,
            warm: 2_000,
            rows: 4_000,
        });
    }
    for k in [20, 200, 500] {
        out.push(Case {
            name: format!("lasso k={k} 3 penalties"),
            spec: decayed(
                json!({"type": "lasso", "penalties": [0.1, 0.01, 0.0]}),
                k,
                true,
            ),
            shape: Shape::Linear,
            // As `scripts/wide_benchmark.py` runs it at the wide widths: a
            // stream of `4k` rows from the first, warm-up included.
            warm: if k > 100 { 0 } else { 2_000 },
            rows: if k > 100 { 4 * k } else { 20_000 },
        });
    }
    out
}

fn fnv(h: &mut u64, v: f64) {
    for b in v.to_bits().to_le_bytes() {
        *h ^= b as u64;
        *h = h.wrapping_mul(0x100000001b3);
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (reps, filters): (usize, Vec<String>) = match args.first().and_then(|a| a.parse().ok()) {
        Some(r) => (r, args[1..].to_vec()),
        None => (7, args),
    };
    for case in cases() {
        if !filters.is_empty() && !filters.iter().any(|f| case.name.contains(f.as_str())) {
            continue;
        }
        let mut spec: Spec = serde_json::from_value(case.spec.clone())
            .unwrap_or_else(|e| panic!("{}: {e}", case.name));
        spec.check()
            .unwrap_or_else(|e| panic!("{}: {e}", case.name));
        let rows = Rows::generate(case.shape, spec.k(), spec.m(), case.warm + case.rows, 7);
        let mut times = Vec::with_capacity(reps);
        let mut digest = None;
        for _ in 0..reps {
            let (_, mut model) = build_models(&spec)
                .unwrap_or_else(|e| panic!("{}: {e}", case.name))
                .pop()
                .expect("one instance");
            let mut step = rheon_polars::rheon_core::Step::default();
            for i in 0..case.warm {
                let (x, y, d) = rows.row(i);
                model.step_into(x, y, d, 1.0, &mut step);
            }
            rayon::current_num_threads();
            let mut h = 0xcbf29ce484222325u64;
            let t = Instant::now();
            for i in case.warm..case.warm + case.rows {
                let (x, y, d) = rows.row(i);
                model.step_into(black_box(x), y, d, 1.0, &mut step);
                if digest.is_none() {
                    for &p in &step.pred {
                        fnv(&mut h, p);
                    }
                    fnv(&mut h, step.n_eff);
                }
                black_box(&step);
            }
            let dt = t.elapsed().as_secs_f64();
            // The first repetition also digests, so it is timed apart.
            if digest.is_none() {
                digest = Some(h);
            } else {
                times.push(dt);
            }
        }
        if times.is_empty() {
            times.push(f64::NAN);
        }
        times.sort_by(f64::total_cmp);
        let best = times[0] / case.rows as f64 * 1e9;
        let median = times[times.len() / 2] / case.rows as f64 * 1e9;
        println!(
            "{:<34} best {:>10.1} ns/row  median {:>10.1} ns/row  {:>11.0} rows/s  digest {:016x}",
            case.name,
            best,
            median,
            1e9 / best,
            digest.unwrap_or(0)
        );
    }
}
