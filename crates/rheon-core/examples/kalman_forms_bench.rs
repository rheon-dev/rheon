//! `Kalman::step` alone, no Polars, in each covariance form (task 281
//! (3)): the default covariance form, the same form again as an
//! identical-code control, and the square root. Rounds in ABBA
//! order, each from a clone of one warmed filter over the same rows, so
//! every form times the same work; the bank's setting (standardized, `P`
//! in units of the noise), with and without drift.
//!
//!     cargo run --release -p rheon-core --example kalman_forms_bench [rounds]
use rheon_core::*;
use std::hint::black_box;
use std::time::Instant;

fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn main() {
    let rounds: usize = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(8);
    let forms = [
        ("covariance", CovarianceForm::Covariance),
        ("control", CovarianceForm::Covariance),
        ("square_root", CovarianceForm::SquareRoot),
    ];
    for h in [2000.0, f64::INFINITY] {
        for k in [5usize, 20, 50] {
            let rows = 4_000_000 / (k * k).max(100);
            let warm = 2_000usize;
            let mut s = 281u64;
            let x: Vec<f64> = (0..(rows + warm) * k).map(|_| lcg(&mut s)).collect();
            let y: Vec<f64> = (0..rows + warm)
                .map(|i| x[i * k] - 0.5 * x[i * k + k - 1] + 0.3 * lcg(&mut s))
                .collect();
            let warmed: Vec<Kalman> = forms
                .iter()
                .map(|(_, f)| {
                    let mut m = Kalman::new(KalmanCfg {
                        horizons: Vec::new(),
                        n_features: k,
                        n_targets: 1,
                        fit_intercept: true,
                        decay: Decay::Halflife(500.0),
                        half_life: vec![h],
                        q: None,
                        obs_var: None,
                        p0: 1.0,
                        share_p: false,
                        min_weight: 0.0,
                        revert_half_life: vec![f64::INFINITY],
                        standardize: true,
                        min_weight_holds_learning: false,
                        noise_units: true,
                        obs: Default::default(),
                    })
                    .unwrap()
                    .with_covariance_form(*f)
                    .unwrap();
                    for i in 0..warm {
                        m.step(&x[i * k..(i + 1) * k], &[Some(y[i])], 1.0, 1.0);
                    }
                    m
                })
                .collect();
            let mut times = vec![Vec::new(); forms.len()];
            let mut control = Vec::new();
            for r in 0..rounds {
                let order: Vec<usize> = if r % 2 == 0 {
                    vec![0, 1, 2]
                } else {
                    vec![2, 1, 0]
                };
                let mut this = [0.0; 3];
                for v in order {
                    let mut m = warmed[v].clone();
                    let t0 = Instant::now();
                    for i in warm..warm + rows {
                        black_box(m.step(&x[i * k..(i + 1) * k], &[Some(y[i])], 1.0, 1.0));
                    }
                    let ns = t0.elapsed().as_secs_f64() * 1e9 / rows as f64;
                    times[v].push(ns);
                    this[v] = ns;
                }
                control.push(this[1] / this[0]);
            }
            let mins: Vec<f64> = times
                .iter()
                .map(|t| t.iter().copied().fold(f64::INFINITY, f64::min))
                .collect();
            let meds: Vec<f64> = times.iter_mut().map(|t| median(t)).collect();
            let (clo, chi) = control
                .iter()
                .fold((f64::INFINITY, 0.0f64), |(a, b), c| (a.min(*c), b.max(*c)));
            print!("h={h:>6} k={k:>2}:");
            for (v, (name, _)) in forms.iter().enumerate() {
                print!(
                    " {name} {:.0} ns ({:.2}x min, {:.2}x med) |",
                    mins[v],
                    mins[v] / mins[0],
                    meds[v] / meds[0]
                );
            }
            println!(" control per round {clo:.3}-{chi:.3}");
        }
    }
}
