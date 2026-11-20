//! Study C4's driver (`scripts/study_c4.py` runs it): one model over one
//! stream, under a ridge variant, writing each row's posterior.
//!
//!     cargo run --release -p rheon-core --features study-c4 --example study_c4 -- \
//!         MODEL COV VARIANT D K HALF_LIFE PRIOR WARM IN OUT [REPS]
//!
//! * `MODEL`: `hmm` or `ewclass`; `COV`: `full`, `diag` or `tied`.
//! * `VARIANT`: `A`, `B:<prior rows>` or `C:<floor>` ([`rheon_core::study_c4`]).
//! * `D` features, `K` states or classes, `HALF_LIFE` in clock units (`inf`:
//!   no decay), `PRIOR` the `precision_prior`, `WARM` the hmm's warm rows.
//! * `IN`: little-endian doubles, `3 + D` a row: the clock step, the weight,
//!   the label (ew_class's; ignored by hmm), the features.
//! * `OUT`: `1 + K` doubles a row: hmm's row loglik and its filtered
//!   posterior after the row; ew_class's NaN and its posterior before the row.
//! * `REPS` (default 1): run the stream this many times on fresh models and
//!   report the best time.
//!
//! Prints one JSON line: ns a row (best of `REPS`), the state's doubles, the
//! solve failures.

use rheon_core::study_c4::Variant;
use rheon_core::{
    Covariance, Decay, EwClass, EwClassCfg, Hmm, HmmCfg, OnlineModel, SeedRule, Step,
};
use std::time::Instant;

fn read(path: &str) -> Vec<f64> {
    let b = std::fs::read(path).expect("input");
    b.as_chunks::<8>()
        .0
        .iter()
        .map(|c| f64::from_le_bytes(*c))
        .collect()
}

fn variant(s: &str) -> Variant {
    match s.split_once(':') {
        None if s == "A" => Variant::A,
        Some(("B", v)) => Variant::B {
            prior_rows: v.parse().unwrap(),
        },
        Some(("C", v)) => Variant::C {
            floor: v.parse().unwrap(),
        },
        Some(("D", v)) => Variant::D {
            prior_rows: v.parse().unwrap(),
        },
        _ => panic!("variant {s}"),
    }
}

enum Model {
    Hmm(Box<Hmm>),
    Class(Box<EwClass>),
}

#[allow(clippy::too_many_arguments)]
fn build(
    model: &str,
    cov: Covariance,
    v: Variant,
    d: usize,
    k: usize,
    decay: Decay,
    prior: f64,
    warm: usize,
) -> Model {
    match model {
        "hmm" => {
            let mut m = Hmm::new(HmmCfg {
                n_features: d,
                k,
                decay,
                covariance: cov,
                precision_prior: prior,
                min_weight: 0.0,
                learn: true,
                transition_prior: 1.0,
                transition: None,
                means: None,
                covs: None,
                warm_rows: warm,
                seed_rule: SeedRule::Kmeanspp,
                seed: 7,
                tvtp: None,
            })
            .unwrap();
            m.set_study(v);
            Model::Hmm(Box::new(m))
        }
        "ewclass" => {
            let mut m = EwClass::new(EwClassCfg {
                n_features: d,
                n_classes: k,
                decay,
                min_weight: 0.0,
                covariance: cov,
                precision_prior: prior,
                window: None,
                window_every: None,
                max_rows_between_snapshots: None,
            })
            .unwrap();
            m.set_study(v);
            Model::Class(Box::new(m))
        }
        _ => panic!("model {model}"),
    }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (model, cov, v) = (&a[1], Covariance::parse(&a[2]).unwrap(), variant(&a[3]));
    let d: usize = a[4].parse().unwrap();
    let k: usize = a[5].parse().unwrap();
    let hl: f64 = a[6].parse().unwrap();
    let prior: f64 = a[7].parse().unwrap();
    let warm: usize = a[8].parse().unwrap();
    let data = read(&a[9]);
    let reps: usize = a.get(11).map_or(1, |r| r.parse().unwrap());
    let decay = Decay::Halflife(hl);
    let width = 3 + d;
    let rows = data.len() / width;
    let mut out = Vec::with_capacity(rows * (1 + k));
    let mut best = f64::INFINITY;
    let mut state_len = 0;
    let mut failures = 0;
    for rep in 0..reps.max(1) {
        let mut m = build(model, cov, v, d, k, decay, prior, warm);
        let mut step = Step::default();
        let keep = rep == 0;
        let t = Instant::now();
        for r in data.chunks_exact(width) {
            let (dt, w, label, x) = (r[0], r[1], r[2], &r[3..]);
            match &mut m {
                Model::Hmm(h) => {
                    h.step_into(x, &[], dt, w, &mut step);
                    if keep {
                        out.push(step.pred[2 * k + 1]);
                        out.extend_from_slice(h.filtered());
                    }
                }
                Model::Class(c) => {
                    let y = [(label >= 0.0).then_some(label)];
                    c.step_into(x, &y, dt, w, &mut step);
                    if keep {
                        out.push(f64::NAN);
                        out.extend_from_slice(&step.pred[1..=k]);
                    }
                }
            }
        }
        best = best.min(t.elapsed().as_secs_f64());
        (state_len, failures) = match &m {
            Model::Hmm(h) => (h.study_state_len(), h.solve_failures),
            Model::Class(c) => (
                c.study().state_len() + k * (d * (d + 1) / 2 + 2 * d + 5),
                c.solve_failures,
            ),
        };
    }
    let bytes: Vec<u8> = out.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(&a[10], bytes).expect("output");
    println!(
        "{{\"ns_per_row\": {}, \"state_doubles\": {}, \"solve_failures\": {}, \"rows\": {}}}",
        best * 1e9 / rows as f64,
        state_len,
        failures,
        rows
    );
}
