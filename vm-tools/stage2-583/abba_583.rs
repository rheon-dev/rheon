//! Native timing driver for Stage 2 (PLAN 583): compiles identically against
//! `v2.0.0` and `HEAD` using `rheon_bench`'s public budget entry points.
//!
//! Usage:
//!   abba_583 <reps> <v2|all> [substring_filter...]

use std::hint::black_box;
use std::time::Instant;

use rheon_bench::{
    BANK_CASES, CLOCK_CASES, MODEL_CASES, ROWS, WIDE_CASES, WIDE_ROWS, WINDOW_CASES, bank_setup,
    clock_setup, model_setup, rheon_budget_bank_rows, rheon_budget_clock_rows,
    rheon_budget_model_rows, rheon_budget_windows_rows, windows_setup,
};

const V2_MODELS: &[&str] = &[
    "ewridge_k5",
    "ewridge_k20",
    "ewridge_k200",
    "ewridge_gram_t10",
    "huber_k20",
    "huber_k20_sets",
    "huber_k20_sum",
    "huber_k20_refit",
    "sgd",
    "sgd_huber",
    "sgd_huber_refit",
    "pa",
    "seqtest",
    "bocpd",
    "kmeans",
    "micro",
    "kalman",
    "ew_cov",
    "lasso_k20",
    "rls",
    "quantile",
    "quantile_refit",
    "ftrl",
    "holt",
    "marginal",
    "marginal_bins_huber",
    "ew_class",
    "deco",
    "hmm",
    "rcov",
    "audit_continuous",
    "bocpd_full",
    "ew_cov_lags",
    "corrchange_monitor_w2500",
    "corrchange_window_w100",
];

const V2_CLOCKS: &[&str] = &["clock_float", "clock_temporal"];

const V2_BANKS: &[&str] = &[
    "bank_seqtest",
    "bank_ewridge_k5",
    "bank_seqtest_groups",
    "bank_ewridge_k5_groups",
    "formula_target_held1k",
];

const V2_WIDES: &[&str] = &["bank_ew_cov_wide_t1"];

const V2_WINDOWS: &[&str] = &[
    "window_ewm_mean_w10k",
    "window_ewm_sum_w10k",
    "window_ewm_quantile_w10k",
    "window_ewm_quantile_running",
    "window_ewm_sum_groups10k",
];

fn matches(id: &str, filters: &[String]) -> bool {
    filters.is_empty() || filters.iter().any(|f| id.contains(f.as_str()))
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let reps: usize = args.first().and_then(|s| s.parse().ok()).unwrap_or(5);
    let suite = args.get(1).map(String::as_str).unwrap_or("v2");
    let filters = if args.len() > 2 { &args[2..] } else { &[] };

    let (models, clocks, banks, wides, windows) = if suite == "all" {
        (MODEL_CASES, CLOCK_CASES, BANK_CASES, WIDE_CASES, WINDOW_CASES)
    } else {
        (V2_MODELS, V2_CLOCKS, V2_BANKS, V2_WIDES, V2_WINDOWS)
    };

    for &id in models {
        if !matches(id, filters) {
            continue;
        }
        let mut best_ns = f64::INFINITY;
        let mut digest = 0u64;
        for _ in 0..reps {
            let mut run = model_setup(id);
            let t0 = Instant::now();
            let blk = black_box(rheon_budget_model_rows(black_box(&mut run)));
            let ns = t0.elapsed().as_nanos() as f64 / ROWS as f64;
            if ns < best_ns {
                best_ns = ns;
            }
            digest = blk.checksum.to_bits() ^ (blk.predicted as u64).rotate_left(17);
        }
        println!("{id:<34} best {best_ns:10.2} ns/row  digest {digest:016x}");
    }

    for &id in clocks {
        if !matches(id, filters) {
            continue;
        }
        let mut best_ns = f64::INFINITY;
        let mut digest = 0u64;
        for _ in 0..reps {
            let mut run = clock_setup(id);
            let t0 = Instant::now();
            let blk = black_box(rheon_budget_clock_rows(black_box(&mut run)));
            let ns = t0.elapsed().as_nanos() as f64 / ROWS as f64;
            if ns < best_ns {
                best_ns = ns;
            }
            digest = blk.checksum.to_bits() ^ (blk.predicted as u64).rotate_left(17);
        }
        println!("{id:<34} best {best_ns:10.2} ns/row  digest {digest:016x}");
    }

    for &id in banks {
        if !matches(id, filters) {
            continue;
        }
        let mut best_ns = f64::INFINITY;
        let mut digest = 0u64;
        for _ in 0..reps {
            let mut run = bank_setup(id);
            let pool = rheon_polars::pool().expect("bank pool");
            let t0 = Instant::now();
            let cols = black_box(pool.install(|| rheon_budget_bank_rows(black_box(&mut run))));
            let ns = t0.elapsed().as_nanos() as f64 / ROWS as f64;
            if ns < best_ns {
                best_ns = ns;
            }
            digest = cols
                .iter()
                .fold(cols.len() as u64, |acc, c| acc.wrapping_mul(131).wrapping_add(c.len() as u64));
        }
        println!("{id:<34} best {best_ns:10.2} ns/row  digest {digest:016x}");
    }

    for &id in wides {
        if !matches(id, filters) {
            continue;
        }
        let mut best_ns = f64::INFINITY;
        let mut digest = 0u64;
        for _ in 0..reps {
            let mut run = bank_setup(id);
            let pool = rheon_polars::pool().expect("bank pool");
            let t0 = Instant::now();
            let cols = black_box(pool.install(|| rheon_budget_bank_rows(black_box(&mut run))));
            let ns = t0.elapsed().as_nanos() as f64 / WIDE_ROWS as f64;
            if ns < best_ns {
                best_ns = ns;
            }
            digest = cols
                .iter()
                .fold(cols.len() as u64, |acc, c| acc.wrapping_mul(131).wrapping_add(c.len() as u64));
        }
        println!("{id:<34} best {best_ns:10.2} ns/row  digest {digest:016x}");
    }

    for &id in windows {
        if !matches(id, filters) {
            continue;
        }
        let mut best_ns = f64::INFINITY;
        let mut digest = 0u64;
        for _ in 0..reps {
            let mut run = windows_setup(id);
            let t0 = Instant::now();
            let df = black_box(rheon_budget_windows_rows(black_box(&mut run)));
            let ns = t0.elapsed().as_nanos() as f64 / ROWS as f64;
            if ns < best_ns {
                best_ns = ns;
            }
            digest = (df.height() as u64)
                .wrapping_mul(131)
                .wrapping_add(df.width() as u64);
        }
        println!("{id:<34} best {best_ns:10.2} ns/row  digest {digest:016x}");
    }
}
