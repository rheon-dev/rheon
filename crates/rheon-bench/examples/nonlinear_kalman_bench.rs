//! Nonlinear `kalman` filters, timed: linear filter, EKF, `scalar_index` UKF,
//! and full `merwe` UKF at 5 and 20 features (6 and 21 coefficients), one
//! target, through `AnyModel::step_into` as the bank steps them.
//!
//!     cargo run --release -p rheon-bench --example nonlinear_kalman_bench -- [reps] [rows]
//!
//! Prints one line per case: the median and the best over `reps` (default 7)
//! of a block of `rows` rows (default 20,000) after 2,000 rows of warm-up,
//! in ns a row. Each repetition builds the model afresh, so every
//! repetition does the same work.
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
}

fn spec(k: usize, extra: Value) -> Spec {
    let mut model = json!({"type": "kalman"});
    if let (Value::Object(m), Value::Object(add)) = (&mut model, extra) {
        m.extend(add);
    }
    let mut s: Spec = serde_json::from_value(json!({
        "name": "m",
        "model": model,
        "targets": ["y0"],
        "features": (0..k).map(|j| format!("x{j}")).collect::<Vec<_>>(),
        "clock": "t",
        "half_life": 1000.0,
        "gap_cap": 100.0,
        "min_weight": k as f64,
    }))
    .expect("a spec");
    s.check().expect("a valid spec");
    s
}

fn main() {
    let args: Vec<usize> = std::env::args()
        .skip(1)
        .map(|a| a.parse().expect("a count"))
        .collect();
    let reps = args.first().copied().unwrap_or(7);
    let rows = args.get(1).copied().unwrap_or(20_000);
    let warm = 2_000;
    for k in [5, 20] {
        let q = vec![1e-4; k + 1];
        for (label, count_y, extra) in [
            ("linear", false, json!({"coef_half_life": 2000.0})),
            (
                "ekf poisson",
                true,
                json!({"family": "poisson", "filter": "ekf", "q": q, "p0": 1.0}),
            ),
            (
                "ukf scalar_index",
                true,
                json!({
                    "family": "poisson",
                    "filter": "ukf",
                    "ukf_rule": "scalar_index",
                    "q": q,
                    "p0": 1.0,
                }),
            ),
            (
                "ukf merwe",
                true,
                json!({
                    "family": "poisson",
                    "filter": "ukf",
                    "ukf_rule": "merwe",
                    "q": q,
                    "p0": 1.0,
                }),
            ),
        ] {
            let s = spec(k, extra);
            let mut g = Lcg(7);
            let n = warm + rows;
            let x: Vec<f64> = (0..n * k).map(|_| 0.7 * (2.0 * g.next() - 1.0)).collect();
            let y: Vec<Option<f64>> = (0..n)
                .map(|i| {
                    let fit: f64 = x[i * k..(i + 1) * k]
                        .iter()
                        .enumerate()
                        .map(|(j, v)| v / (j + 1) as f64)
                        .sum();
                    if count_y {
                        Some(((0.4 + 0.5 * fit).exp() + g.next()).floor().max(0.0))
                    } else {
                        Some(0.5 + fit + 0.3 * (2.0 * g.next() - 1.0))
                    }
                })
                .collect();
            let mut times = Vec::with_capacity(reps);
            for _ in 0..reps {
                let (_, mut model) = build_models(&s).expect("a model").pop().expect("one");
                let mut step = rheon_polars::rheon_core::Step::default();
                for i in 0..warm {
                    let d = if i == 0 { 0.0 } else { 1.0 };
                    model.step_into(&x[i * k..(i + 1) * k], &y[i..=i], d, 1.0, &mut step);
                }
                let t = Instant::now();
                for i in warm..n {
                    model.step_into(
                        black_box(&x[i * k..(i + 1) * k]),
                        &y[i..=i],
                        1.0,
                        1.0,
                        &mut step,
                    );
                    black_box(&step);
                }
                times.push(t.elapsed().as_secs_f64() / rows as f64 * 1e9);
            }
            times.sort_by(f64::total_cmp);
            println!(
                "k={k:>2} {label:<18} median {:>8.0} ns/row  best {:>8.0} ns/row",
                times[times.len() / 2],
                times[0]
            );
        }
    }
}
