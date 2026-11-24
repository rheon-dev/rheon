//! The workloads behind the instruction-count budgets (`budgets.toml`).
//!
//! Each case is a fixed block of [`ROWS`] rows through one hot path
//! ([`WIDE_ROWS`] for the wide bank cases), after [`WARM_ROWS`] rows that
//! bring the state to where a long stream keeps it: past `min_weight` and
//! the first solves, a full run vector, settled centres. A case is built by
//! a *setup* function and run by one of three measured functions,
//! [`rheon_budget_model_rows`], [`rheon_budget_clock_rows`]
//! and [`rheon_budget_bank_rows`]. `benches/instructions.rs` hands
//! each to Callgrind with the measured function as its entry point, so the
//! count is those rows and nothing else: not the setup, not the data, not
//! the drop. The functions carry fixed, unmangled names because Callgrind
//! finds an entry point by its symbol. The one case on several threads is
//! counted from a client request around the call instead, on every thread
//! (`benches/instructions.rs`).
//!
//! The data are generated here, from a fixed seed, so a count depends on the
//! code and the toolchain alone. `cargo test` runs every case natively on
//! every platform and checks that it reaches the path it is named for.

pub mod scaling;

use polars::prelude::*;
use rheon_polars::rheon_core::bars::BarKind;
use rheon_polars::rheon_core::{ClockCfg, ClockState, ClockValue, ExactCaps};
use rheon_polars::{
    AggKind, AggSpec, AnyModel, Bank, BarThreshold, Bars, BarsConfig, Spec, WindowsConfig,
    WindowsRun, build_models,
};
use serde_json::{Value, json};

/// Rows each case steps before the measured block, outside the count.
pub const WARM_ROWS: usize = 2_000;

/// Rows in the measured block. A case's budget is for the block; its count
/// per row is the budget over this.
pub const ROWS: usize = 1_000;

/// The model cases, by the id the bench and `budgets.toml` give them.
pub const MODEL_CASES: &[&str] = &[
    "ewridge_k5",
    "ewridge_k20",
    // A wide fit, 200 features on the default schedule: each solve's
    // factorization, its condition check and its rank-one updates at the
    // width where the k² and k³ work is the row's cost (docs/PLAN.md task
    // 306).
    "ewridge_k200",
    "ewridge_gram_t10",
    "huber_k20",
    "huber_k20_sets",
    "huber_k20_sum",
    // The refit of the first rows (docs/PLAN.md task 277) inside the
    // measured block: its buffer of 2,500 rows fills on the block's 500th.
    "huber_k20_refit",
    "sgd",
    "sgd_huber",
    "sgd_huber_refit",
    "pa",
    // A Poisson fit of the labels 0-2 as counts: a Newton step a row, the
    // Cholesky of `G + R I` at 21 coefficients; the default refit of the
    // first 210 rows runs in the warm-up.
    "glm_k20",
    // An upper expectile in one pass: its side judged at the fit before
    // the row, a rank-one update of the mean-form Gram and a Cholesky of 21
    // coefficients a row; the default refit of the first 210 rows runs in
    // the warm-up.
    "expectile_k20",
    // A window of 500 rows at 5 features refit exactly on every row,
    // warm-started at the fit before: a WLS solve a pass over the window.
    "expectile_window",
    "seqtest",
    "bocpd",
    "kmeans",
    "micro",
    "kalman",
    // The fixed-lag smoother on top of the filter: a lag of 10 observations
    // with its standard errors, one target, 6 and 21 coefficients. A row's
    // gain (a Cholesky of the next prior and two triangular solves a row of
    // `A`), `K`'s triangle, and the backward pass of 10 slots, means and
    // covariance.
    "kalman_smooth_k5",
    "kalman_smooth_k20",
    "ew_cov",
    "lasso_k20",
    // One case per model kind that had none, and the paths the review of
    // 2026-10-23 found costly (task 270 (e): J-3, slice M's table, F-2).
    "rls",
    "quantile",
    // Its refit of the first rows (docs/PLAN.md task 278, D-Q2) inside the
    // measured block, as `huber_k20_refit`'s.
    "quantile_refit",
    "ftrl",
    "holt",
    "marginal",
    // Its bins under Huber's location (docs/PLAN.md task 272 (c)).
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
    // One node, three kernels: its row's solve of a 4 x 4 system for the
    // intensity it reports before the row's event (task 498).
    "hawkes",
    // One series after its start: a row's variance, score and 3 x 3 step
    // (task 492); and FIGARCH, its O(M) pass of 1,000 lags a row.
    "garch",
    "garch_figarch",
    // Three series' correlation, the asymmetric model learned after its
    // start: a row's 3 x 3 factor, inverse and three R⁻¹ ∂R (task 493); and
    // the same through GARCH margins.
    "dcc",
    "dcc_margins",
    // Two return series under a decay, Lee and Mykland's own window of 100:
    // a row's compensated sums and ring (task 499).
    "realized",
    // One component of 20 features by the incremental SVD: a row rotates
    // the 2 kept directions and the row's deviation until orthogonal.
    "pca",
    // Two return series under a decay, a term of each kind: a row's
    // multipower products, two kernel sums of 20 and 10 lags and a scale of
    // 10 (task 555).
    "power_variation",
    // One column's digest at compression 100, read at three levels: a row's
    // merge of a copy of its buffer (task 544).
    "tdigest",
    // One column's DDSketch at the default accuracy, read at three levels
    // with their tail means: a row's walk of its buckets (task 544).
    "ddsketch",
    // An AR(5) with no features: a row's ring, design, rank-one update of a
    // 6 x 6 Gram and its Cholesky.
    "ar",
    // Sixteen periods on an even clock: a rotation and an add each a row.
    "spectral",
    // One target on one feature over a window of 200 rows under a decay: a
    // row's insert into the sorted rows, the pooling of its ties and a
    // whole pool-adjacent-violators pass over the window (task 573).
    "isotonic",
    // A VAR(2) of three series with a Granger test, solving every row: a
    // row's ring, the rank-one update of a 10 x 10 Gram, the fit's solve
    // and the read of the test from one Cholesky factor.
    "vector_ar",
];

/// The one model case that reports on a single row of its block:
/// `corrchange`'s monitor tests its span when the span's last row arrives,
/// and its span of 2,500 rows closes once in the block, at its 500th row.
/// That close is the cost the case is for (docs/PLAN.md task 270, D-1).
pub const SPAN_CASE: &str = "corrchange_monitor_w2500";

/// The model cases whose model predicts nothing by design: `Step::pred` is
/// empty on every row, so the native test holds each to the path it is
/// named for by another reading (`tests/workloads.rs`).
pub const SILENT_CASES: &[&str] = &[
    "marginal",
    "marginal_bins_huber",
    "rcov",
    "audit_continuous",
];

/// The clock cases: a float clock, and a temporal one on the exact path.
pub const CLOCK_CASES: &[&str] = &["clock_float", "clock_temporal"];

/// The bank cases: a frame through `Bank::fit_predict`, every per-row step
/// of the bank around the model, for the cheapest model and a regression;
/// and the same two over [`GROUPS`] groups of one row each, on a pool of one
/// thread, where what a `(spec, group)` task costs whatever its rows is most
/// of the count (docs/PLAN.md task 270, G-1); and a regression whose target
/// is a window expression looking 1,000 rows ahead under an embargo of the
/// same span, so each row waits behind about a thousand held rows
/// (docs/PLAN.md task 270, M-3).
pub const BANK_CASES: &[&str] = &[
    "bank_seqtest",
    "bank_ewridge_k5",
    "bank_seqtest_groups",
    "bank_ewridge_k5_groups",
    "formula_target_held1k",
];

/// Groups in a grouped bank case: its block is one row of each.
pub const GROUPS: usize = 1_000;

/// Rows of a grouped case's warm-up: ten a group, past `min_weight` and the
/// first solve of each group's fit.
pub const GROUPED_WARM_ROWS: usize = 10 * GROUPS;

fn grouped(id: &str) -> bool {
    id.ends_with("_groups")
}

/// The window cases: a frame through the window operators' core
/// (`WindowsRun::feed`, as `with_windows` runs it), one operator over a
/// window of 12,500 clock units, about 10,000 rows at the cases' mean step
/// of 1.25. The warm-up and the block are inside the stream's first window,
/// where nothing has left it yet and a mean cuts its oldest row's held
/// interval at the window's edge on every row: the path that cost a mean
/// O(rows in the window) a row where the sum costs O(1) (docs/PLAN.md task
/// 270, M-2). The two are held within twice each other. And the quantile
/// (task 272 (d)): over the same window, its sketches walked on every row,
/// and with no window, the running sketch an EW-MAD reads. And a running
/// sum over [`WINDOW_GROUPS`] groups, one row of each in turn, with no cap
/// on a group's silence, so every group stays on the window core's list of
/// groups by last row: a row's cost must not grow with the groups behind it
/// (task 325: the walk from the list's head made it 80 µs a row at 10,000).
/// And the clock step (task 564): the decayed mean of the steps of the
/// stream's own clock, ungrouped (one group), half-life 1000, beside a
/// running `ewm_mean` of the feature at the same half-life, its baseline:
/// the two run the same rows through the same core, so their difference
/// over [`ROWS`] is what a row of the clock step costs above a running
/// mean.
pub const WINDOW_CASES: &[&str] = &[
    "window_ewm_mean_w10k",
    "window_ewm_sum_w10k",
    "window_ewm_quantile_w10k",
    "window_ewm_quantile_running",
    "window_ewm_sum_groups10k",
    "window_invert_cubic",
    "window_ewm_mean_running",
    "window_clock_step_running",
];

/// The bar cases: a frame of trades through the bars operator's core
/// (`Bars::feed`, as `rp.stream.bars` runs it), volume bars of 1,000 over
/// sizes of 1 to 100, about 20 rows a bar, each with the open, high, low
/// and close of the price, the volume and the rows: a row adds its size to
/// the exact running total, tests the crossing and folds six aggregates,
/// and about one row in 20 closes a bar into the output. Ungrouped, and
/// over [`BAR_GROUPS`] groups, one row of each in turn, ten rows a group in
/// the block, so about half the groups close a bar in it: a row finds its
/// group's open bar among the others.
pub const BAR_CASES: &[&str] = &["bars_volume", "bars_volume_groups100"];

/// Groups in the grouped bar case.
pub const BAR_GROUPS: usize = 100;

/// The invert case (PLAN 563): the root in `u` of `u^3 + u + exp(u / 2) =
/// x0` on `[-10, 10]`, solved on every row by the formula's own interpreter,
/// beside a running sum of x0, since a run computes at least one window
/// operator; the sum keeps no row, so the block's cost is the solve's.
/// `windows_setup` builds it and `tests/workloads.rs` runs it natively. It
/// joins [`WINDOW_CASES`] and `benches/instructions.rs` together with its
/// entry in budgets.toml, which every counted case needs.
pub const INVERT_CASE: &str = "window_invert_cubic";

/// The invert case's node: `u^3 + u + exp(u / 2) = x0` on `[-10, 10]`,
/// increasing, at the default tolerance.
fn invert_tree() -> Value {
    let u = json!(["unknown"]);
    json!(["invert",
        ["+", ["+", ["**", u, ["lit", 3.0]], u], ["exp", ["*", ["lit", 0.5], u]]],
        ["col", "x0"], ["lit", -10.0], ["lit", 10.0],
        {"increasing": true}])
}

/// Groups in the grouped window case: its warm-up is two rows of each.
pub const WINDOW_GROUPS: usize = 10_000;

/// The wide bank cases: a frame of [`WIDE_ROWS`] rows of the `ew_cov` case's
/// 230 statistics through `Bank::fit_predict`, on a pool of one thread.
/// The frame is several runs long (about a thousand rows a run at this
/// width), so each run is copied into the output as it ends while the
/// stream learns the next (`rayon::join`, which on one thread runs the two
/// in turn), and it is past `PAR_MIN_ROWS`, so the output columns are
/// allocated and finished in parallel. `bank_setup` also knows
/// `bank_ew_cov_wide_t4`, the same on a pool of four, which the native test
/// holds to the one-thread case's output; it has no count (benches/
/// instructions.rs says why).
pub const WIDE_CASES: &[&str] = &["bank_ew_cov_wide_t1"];

/// Rows in a wide case's measured block.
pub const WIDE_ROWS: usize = 8_192;

/// A clock step of a second on average, irregular: 0.5, 1, 1.5 or 2.
fn step_of(draw: f64) -> f64 {
    0.5 * (1.0 + (4.0 * draw).floor().min(3.0))
}

/// A seeded generator: Knuth's MMIX constants, the top 53 bits as a
/// uniform draw in `[0, 1)`. Integer arithmetic only, so the data are the
/// same bits on every platform.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }

    /// A draw in `[-1, 1)`.
    fn signed(&mut self) -> f64 {
        2.0 * self.next() - 1.0
    }
}

/// What a case's rows look like.
#[derive(Clone, Copy)]
enum Shape {
    /// `k` features in `[-1, 1)`, and a target linear in them plus noise.
    Linear,
    /// Four features around one of three centres six apart, as
    /// `scripts/benchmark.py`'s blobs: what a clustering model is for.
    Blobs,
    /// `k` features around one of three centres two apart, and the centre's
    /// number as the target: a label a classifier learns, not always right.
    Labelled,
    /// As `Labelled`, over this many classes: feature `j` is raised for the
    /// classes `c` with `j % classes == c` (a class past the features has
    /// none raised, and its rows still carry its label).
    Classes(usize),
    /// Each target a series of returns of a GARCH(1,1) at `(0.05, 0.15,
    /// 0.8)`, normal innovations (twelve uniforms less six), its variance
    /// started at its long-run level: what `garch` is for. A start on
    /// returns with no conditional variance waits, so on `Linear` targets a
    /// `garch` case would time its start's buffer, not its step (PLAN 548).
    Garch,
    /// The targets a DCC process at `(a, b) = (0.04, 0.92)`, every target
    /// correlation 0.5 (twelve uniforms less six, through the Cholesky
    /// factor of each row's `R`), and with the flag each scaled by its own
    /// GARCH(1,1) at `(0.05, 0.15, 0.8)`: what `dcc` is for. A start on
    /// series of one constant correlation waits (PLAN 493), so on `Linear`
    /// targets a learned `dcc` case would time its start's buffer.
    Dcc(bool),
}

/// Rows in the layout a model steps them: features row-major, targets
/// row-major, and each row's clock step and clock value. Over `groups`
/// groups (0: none), row `i` is group `i % groups`'s, and each group keeps
/// a clock of its own, a unit a row.
pub struct Rows {
    groups: usize,
    k: usize,
    m: usize,
    x: Vec<f64>,
    y: Vec<Option<f64>>,
    d: Vec<f64>,
    t: Vec<f64>,
}

impl Rows {
    fn generate(shape: Shape, k: usize, m: usize, n: usize, groups: usize, seed: u64) -> Self {
        let mut g = Lcg(seed);
        let (mut x, mut y) = (Vec::with_capacity(n * k), Vec::with_capacity(n * m));
        let (mut d, mut t) = (Vec::with_capacity(n), Vec::with_capacity(n));
        let mut clock = 1_000.0;
        // each target's variance and last return, for `Shape::Garch`
        let mut garch: Vec<(f64, f64)> = vec![(1.0, 0.0); m];
        // the DCC process's Q, for `Shape::Dcc`
        let qbar: Vec<f64> = (0..m * m)
            .map(|ij| if ij / m == ij % m { 1.0 } else { 0.5 })
            .collect();
        let mut q = qbar.clone();
        for i in 0..n {
            let row = x.len();
            match shape {
                Shape::Linear => x.extend((0..k).map(|_| g.signed())),
                Shape::Blobs => {
                    let label = (3.0 * g.next()).floor() as usize;
                    x.extend((0..k).map(|j| g.signed() + if j % 3 == label { 6.0 } else { 0.0 }));
                }
                Shape::Labelled => {
                    let label = (3.0 * g.next()).floor() as usize;
                    x.extend((0..k).map(|j| g.signed() + if j % 3 == label { 2.0 } else { 0.0 }));
                    y.extend((0..m).map(|_| Some(label as f64)));
                }
                Shape::Classes(c) => {
                    let label = ((c as f64 * g.next()).floor() as usize).min(c - 1);
                    x.extend((0..k).map(|j| g.signed() + if j % c == label { 2.0 } else { 0.0 }));
                    y.extend((0..m).map(|_| Some(label as f64)));
                }
                Shape::Garch => {
                    x.extend((0..k).map(|_| g.signed()));
                    for (s, e) in garch.iter_mut() {
                        *s = 0.05 + 0.15 * *e * *e + 0.8 * *s;
                        let z = (0..12).map(|_| g.next()).sum::<f64>() - 6.0;
                        *e = (*s).sqrt() * z;
                        y.push(Some(*e));
                    }
                }
                Shape::Dcc(scaled) => {
                    x.extend((0..k).map(|_| g.signed()));
                    let mut l = vec![0.0; m * m];
                    for a in 0..m {
                        for b in 0..=a {
                            let r = if a == b {
                                1.0
                            } else {
                                q[a * m + b] / (q[a * m + a] * q[b * m + b]).sqrt()
                            };
                            let acc = r - (0..b).map(|c| l[a * m + c] * l[b * m + c]).sum::<f64>();
                            l[a * m + b] = if a == b {
                                acc.sqrt()
                            } else {
                                acc / l[b * m + b]
                            };
                        }
                    }
                    let u: Vec<f64> = (0..m)
                        .map(|_| (0..12).map(|_| g.next()).sum::<f64>() - 6.0)
                        .collect();
                    let z: Vec<f64> = (0..m)
                        .map(|a| (0..=a).map(|b| l[a * m + b] * u[b]).sum())
                        .collect();
                    for a in 0..m {
                        for b in 0..m {
                            let ab = a * m + b;
                            q[ab] = 0.04 * qbar[ab] + 0.04 * z[a] * z[b] + 0.92 * q[ab];
                        }
                    }
                    for (j, (s, e)) in garch.iter_mut().enumerate() {
                        if scaled {
                            *s = 0.05 + 0.15 * *e * *e + 0.8 * *s;
                            *e = (*s).sqrt() * z[j];
                        } else {
                            *e = z[j];
                        }
                        y.push(Some(*e));
                    }
                }
            }
            let fit: f64 = x[row..]
                .iter()
                .enumerate()
                .map(|(j, v)| v / (j + 1) as f64)
                .sum();
            // With no features, a column whose sign a sequential test
            // watches: either sign, a small drift up.
            let (level, noise) = if k == 0 { (0.05, 1.0) } else { (0.5, 0.3) };
            if !matches!(
                shape,
                Shape::Labelled | Shape::Classes(_) | Shape::Garch | Shape::Dcc(_)
            ) {
                y.extend((0..m).map(|j| Some(level + fit + j as f64 + noise * g.signed())));
            }
            let step = if i == 0 { 0.0 } else { step_of(g.next()) };
            clock += step;
            d.push(step);
            t.push(match i.checked_div(groups) {
                None => clock,
                Some(row) => 1_000.0 + row as f64,
            });
        }
        Self {
            groups,
            k,
            m,
            x,
            y,
            d,
            t,
        }
    }

    fn len(&self) -> usize {
        self.d.len()
    }

    /// No rows: what a scaling case keeps once its rows are dropped.
    fn empty() -> Self {
        Self {
            groups: 0,
            k: 0,
            m: 0,
            x: Vec::new(),
            y: Vec::new(),
            d: Vec::new(),
            t: Vec::new(),
        }
    }

    fn row(&self, i: usize) -> (&[f64], &[Option<f64>], f64) {
        (
            &self.x[i * self.k..(i + 1) * self.k],
            &self.y[i * self.m..(i + 1) * self.m],
            self.d[i],
        )
    }

    /// Rows `from..to` as a frame: `x0..`, `y0..` and the clock `t`.
    fn frame(&self, from: usize, to: usize) -> DataFrame {
        let mut cols: Vec<Column> = (0..self.k)
            .map(|j| {
                let v: Vec<f64> = (from..to).map(|i| self.x[i * self.k + j]).collect();
                Column::new(format!("x{j}").into(), v)
            })
            .collect();
        cols.extend((0..self.m).map(|j| {
            let v: Vec<Option<f64>> = (from..to).map(|i| self.y[i * self.m + j]).collect();
            Column::new(format!("y{j}").into(), v)
        }));
        cols.push(Column::new("t".into(), self.t[from..to].to_vec()));
        if self.groups > 0 {
            let g: Vec<i64> = (from..to).map(|i| (i % self.groups) as i64).collect();
            cols.push(Column::new("g".into(), g));
        }
        DataFrame::new_infer_height(cols).expect("columns of one length")
    }
}

fn names(prefix: &str, n: usize) -> Vec<String> {
    (0..n).map(|j| format!("{prefix}{j}")).collect()
}

/// A case's spec, as `rheon.spec` writes it, with the shape of its rows. The
/// sizes and settings are `scripts/benchmark.py`'s rows of the same names,
/// so a count and a throughput figure describe one workload.
fn case_spec(id: &str) -> (Value, Shape) {
    let fit = |model: Value, k: usize| {
        json!({
            "name": "m",
            "model": model,
            "targets": ["y0"],
            "features": names("x", k),
            "clock": "t",
            "half_life": 1000.0,
            "gap_cap": 100.0,
            "min_weight": k as f64,
        })
    };
    let four = |model: Value, decays: bool| {
        let mut s = json!({
            "name": "m",
            "model": model,
            "features": names("x", 4),
            "clock": "t",
            "gap_cap": 100.0,
        });
        if decays {
            s["half_life"] = json!(1000.0);
        }
        s
    };
    match id {
        "ewridge_k5" | "bank_ewridge_k5" => (fit(json!({"type": "ewridge"}), 5), Shape::Linear),
        // A solve every 4 rows of a group's clock: at the default cadence a
        // fit ten rows old solves on nearly every row, and the case is for
        // the task's constant, not a young fit's solves. The cadence's
        // first period, which solves on the default schedule (task 278,
        // "Default cadence first"), ends on a group's 5th row, and its
        // solves fall on the 5th, 9th and 13th: the measured 11th solves
        // none, as the old cadence of 20 from the first solve did not.
        "bank_ewridge_k5_groups" => {
            let mut s = fit(json!({"type": "ewridge", "solve_every": 4.0}), 5);
            s["group"] = json!("g");
            (s, Shape::Linear)
        }
        "ewridge_k20" => (fit(json!({"type": "ewridge"}), 20), Shape::Linear),
        // `min_weight` 200, crossed about 220 rows in at this clock's
        // steps: the warm-up's 2,000 rows are past it, so the block solves
        // and predicts on the default weight-share schedule.
        "ewridge_k200" => (fit(json!({"type": "ewridge"}), 200), Shape::Linear),
        // The target the mean of `y0` over the next 1,250 clock units
        // (about 1,000 rows), resolved by the bank's own window core, each
        // row learned once its window has closed and an embargo of the
        // same span has passed: about a thousand rows held at every row.
        "formula_target_held1k" => {
            let mut s = fit(json!({"type": "ewridge"}), 5);
            s["targets"] = json!([{
                "name": "fwd",
                "formula": ["rewm_mean", ["col", "y0"], {"half_life": 1000.0, "window_size": 1250.0}],
            }]);
            s["embargo"] = json!(1250.0);
            (s, Shape::Linear)
        }
        // The blocked Gram with ten targets folded into its merge
        // (docs/PLAN.md task 257): 50 features, a 64-row block, one Gram
        // over every row, a solve every 400 clock units.
        "ewridge_gram_t10" => {
            let model = json!({
                "type": "ewridge",
                "gram_block_rows": 64,
                "target_gaps": "pairwise",
                "solve_every": 400.0,
            });
            let mut s = fit(model, 50);
            s["targets"] = json!(names("y", 10));
            (s, Shape::Linear)
        }
        // Huber's cut in the MAD scale, the default since task 271: a
        // sketch of `|e|` per target, aged and fed each row.
        "huber_k20" => (fit(json!({"type": "huber"}), 20), Shape::Linear),
        // Three nested feature sets of 5, 10 and 20 features from the one
        // target's sums, each solved on the cadence, the rows weighed by
        // the largest set's residual (docs/PLAN.md task 272 (a)).
        // RLS's sum-scale prior on the target's raw moments, the
        // intercept penalized (docs/PLAN.md task 272 (a)).
        "huber_k20_sum" => (
            fit(json!({"type": "huber", "ridge_scale": "sum"}), 20),
            Shape::Linear,
        ),
        // The default refit runs in the warm-up's first 500 rows, outside
        // every block; here it holds 2,500 rows and runs on the block's
        // 500th, so the block pays for it once (docs/PLAN.md task 277).
        "huber_k20_refit" => (
            fit(json!({"type": "huber", "huber_refit_rows": 2500}), 20),
            Shape::Linear,
        ),
        "huber_k20_sets" => {
            let sets = json!([
                ["x5", names("x", 5)],
                ["x10", names("x", 10)],
                ["all", names("x", 20)],
            ]);
            (
                fit(json!({"type": "huber", "feature_sets": sets}), 20),
                Shape::Linear,
            )
        }
        "sgd" => (
            fit(json!({"type": "sgd", "learning_rate": 0.01}), 20),
            Shape::Linear,
        ),
        "sgd_huber" => (
            fit(
                json!({"type": "sgd", "learning_rate": 0.01, "loss": "huber"}),
                20,
            ),
            Shape::Linear,
        ),
        "sgd_huber_refit" => (
            fit(
                json!({"type": "sgd", "learning_rate": 0.01, "loss": "huber", "huber_refit_rows": 2500}),
                20,
            ),
            Shape::Linear,
        ),
        "pa" => (fit(json!({"type": "pa"}), 20), Shape::Linear),
        "glm_k20" => (fit(json!({"type": "glm"}), 20), Shape::Labelled),
        "expectile_k20" => (
            fit(json!({"type": "expectile", "expectile": 0.9}), 20),
            Shape::Linear,
        ),
        "expectile_window" => {
            let mut s = fit(
                json!({"type": "expectile", "expectile": 0.9, "window_rows": 500}),
                5,
            );
            s["half_life"] = json!("inf");
            (s, Shape::Linear)
        }
        // The penalty path's selection on every row and its solves on the
        // default cadence (docs/PLAN.md task 270, C-2); the path is
        // `scripts/wide_benchmark.py`'s.
        "lasso_k20" => (
            fit(json!({"type": "lasso", "penalties": [0.1, 0.01, 0.0]}), 20),
            Shape::Linear,
        ),
        "kalman" => (
            fit(json!({"type": "kalman", "coef_half_life": 2000.0}), 20),
            Shape::Linear,
        ),
        "kalman_smooth_k5" | "kalman_smooth_k20" => (
            fit(
                json!({"type": "kalman", "coef_half_life": 2000.0, "smooth_lag": 10,
                       "emit_smoothed_se": true}),
                if id == "kalman_smooth_k5" { 5 } else { 20 },
            ),
            Shape::Linear,
        ),
        "bank_seqtest_groups" => {
            let (mut s, shape) = case_spec("seqtest");
            s["group"] = json!("g");
            (s, shape)
        }
        // The start's rows fit within the warm-up (500, doubled at most
        // twice); FIGARCH's 2,000 under a cap that holds no doubling. Each
        // on a GARCH series, whose start tells it from a constant variance.
        "garch" => (
            json!({
                "name": "m",
                "model": {"type": "garch", "start_rows": 500},
                "targets": ["y0"],
                "features": [],
            }),
            Shape::Garch,
        ),
        "garch_figarch" => (
            json!({
                "name": "m",
                "model": {"type": "garch", "vol": "figarch", "max_state_bytes": 40000},
                "targets": ["y0"],
                "features": [],
            }),
            Shape::Garch,
        ),
        // Both starts within the warm-up: the correlation's 500 rows, after
        // the margins' 300 (doubled at most twice) where it has them.
        "dcc" => (
            json!({
                "name": "m",
                "model": {"type": "dcc", "margins": "none", "asymmetric": true},
                "targets": ["y0", "y1", "y2"],
                "features": [],
                "half_life": "inf",
            }),
            Shape::Dcc(false),
        ),
        "dcc_margins" => (
            json!({
                "name": "m",
                "model": {"type": "dcc", "asymmetric": true, "margin_start_rows": 300},
                "targets": ["y0", "y1", "y2"],
                "features": [],
                "half_life": "inf",
            }),
            Shape::Dcc(true),
        ),
        "realized" => (
            json!({
                "name": "m",
                "model": {"type": "realized", "lm_window": 100},
                "targets": ["y0", "y1"],
                "features": [],
                "clock": "t",
                "half_life": 1000.0,
                "gap_cap": 100.0,
            }),
            Shape::Linear,
        ),
        "power_variation" => (
            json!({
                "name": "m",
                "model": {
                    "type": "power_variation",
                    "terms": [
                        {"name": "rv", "powers": [2.0]},
                        {"name": "bv", "powers": [1.0, 1.0], "normalize": true},
                        {"name": "lrv", "kernel": "parzen", "bandwidth": 20, "center": true},
                        {"name": "rk", "kernel": "bartlett", "bandwidth": 10, "flat_top": true},
                        {"name": "sub", "powers": [2.0], "scale": 10}
                    ],
                },
                "targets": ["y0", "y1"],
                "features": [],
                "clock": "t",
                "half_life": 1000.0,
                "gap_cap": 100.0,
            }),
            Shape::Linear,
        ),
        "tdigest" => (
            json!({
                "name": "m",
                "model": {"type": "tdigest", "quantiles": [0.05, 0.25, 0.95]},
                "targets": ["y0"],
                "features": [],
            }),
            Shape::Linear,
        ),
        "ddsketch" => (
            json!({
                "name": "m",
                "model": {"type": "ddsketch", "quantiles": [0.05, 0.25, 0.95]},
                "targets": ["y0"],
                "features": [],
            }),
            Shape::Linear,
        ),
        "hawkes" => (
            json!({
                "name": "m",
                "model": {"type": "hawkes", "kernel_decays": [0.5, 2.0, 8.0]},
                "targets": ["y0"],
                "features": [],
                "clock": "t",
                "half_life": 1000.0,
                "gap_cap": 100.0,
            }),
            Shape::Linear,
        ),
        "seqtest" | "bank_seqtest" => (
            json!({
                "name": "m",
                "model": {"type": "seqtest"},
                "targets": ["y0"],
                "features": [],
                "clock": "t",
                "gap_cap": 100.0,
            }),
            Shape::Linear,
        ),
        "pca" => {
            // Rows around two centres: one direction of a larger variance,
            // so its component is written, where independent features of
            // one spread (and three symmetric centres) are tied and withheld.
            let mut s = fit(json!({"type": "pca", "n_components": 1}), 20);
            let s = s.as_object_mut().expect("an object");
            s.remove("targets");
            s.remove("min_weight");
            (Value::Object(s.clone()), Shape::Classes(2))
        }
        "ew_cov" | "bank_ew_cov_wide_t1" | "bank_ew_cov_wide_t4" => {
            let mut s = fit(json!({"type": "ew_cov"}), 20);
            let s = s.as_object_mut().expect("an object");
            s.remove("targets");
            s.remove("min_weight");
            (Value::Object(s.clone()), Shape::Linear)
        }
        "bocpd" => (
            four(
                json!({"type": "bocpd", "prune_below": 1e-6, "max_run": 200}),
                false,
            ),
            Shape::Blobs,
        ),
        "kmeans" => (
            four(json!({"type": "kmeans", "n_clusters": 8}), true),
            Shape::Blobs,
        ),
        "micro" => (
            four(json!({"type": "micro", "eps": 1.0}), true),
            Shape::Blobs,
        ),
        // Four columns, six pairs, a span of 2,500 rows: the warm-up's 2,000
        // and the block's first 500 (docs/PLAN.md task 270, D-1).
        "corrchange_monitor_w2500" => (
            four(
                json!({"type": "corrchange", "kind": "monitor", "window_rows": 2500}),
                false,
            ),
            Shape::Linear,
        ),
        // Two adjacent windows of 100 rows over four columns, a permutation
        // critical value of 100 draws redrawn every 250 rows (docs/PLAN.md
        // task 270, D-2).
        "corrchange_window_w100" => (
            four(
                json!({"type": "corrchange", "kind": "window", "window_rows": 100,
                       "n_perm": 100, "permute_every_rows": 250}),
                false,
            ),
            Shape::Linear,
        ),
        // From here, the cases added with the review of 2026-10-23 (task
        // 270 (e)), each `scripts/benchmark.py`'s row of the same kind.
        "rls" => (fit(json!({"type": "rls", "delta": 1.0}), 20), Shape::Linear),
        "quantile" => (
            fit(json!({"type": "quantile", "quantile": 0.5}), 20),
            Shape::Linear,
        ),
        // The default refit runs in the warm-up's first 500 rows, outside
        // every block; here it holds 2,500 rows and runs on the block's
        // 500th, so the block pays for it once (task 278, D-Q2).
        "quantile_refit" => (
            fit(
                json!({"type": "quantile", "quantile": 0.5, "quantile_refit_rows": 2500}),
                20,
            ),
            Shape::Linear,
        ),
        "ftrl" => (fit(json!({"type": "ftrl"}), 20), Shape::Linear),
        "spectral" => {
            let periods: Vec<f64> = (0..16).map(|k| 2.0 + k as f64).collect();
            let mut s = fit(json!({"type": "spectral", "periods": periods}), 0);
            s.as_object_mut().expect("an object").remove("min_weight");
            (s, Shape::Linear)
        }
        "isotonic" => {
            let mut s = fit(
                json!({"type": "isotonic", "window_rows": 200, "out_of_bounds": "clip"}),
                1,
            );
            s.as_object_mut().expect("an object").remove("min_weight");
            (s, Shape::Linear)
        }
        "ar" => {
            let mut s = fit(json!({"type": "ar", "lags": 5}), 0);
            s.as_object_mut().expect("an object").remove("min_weight");
            (s, Shape::Linear)
        }
        "vector_ar" => {
            let mut s = fit(
                json!({"type": "vector_ar", "lags": 2, "max_rows_between_solves": 1,
                       "causality": [{"caused": ["y0"]}]}),
                0,
            );
            s["targets"] = json!(["y0", "y1", "y2"]);
            s.as_object_mut().expect("an object").remove("min_weight");
            (s, Shape::Linear)
        }
        // Level and trend of one column, no features.
        "holt" => {
            let mut s = fit(json!({"type": "holt"}), 0);
            s.as_object_mut().expect("an object").remove("min_weight");
            (s, Shape::Linear)
        }
        // Twenty features against one target, every pair's moments.
        "marginal" => (fit(json!({"type": "marginal"}), 20), Shape::Linear),
        // The same, with four given bins a feature and each bin's location
        // Huber's (docs/PLAN.md task 272 (c)): a cell's step, its judge
        // and each target's scale on every row.
        "marginal_bins_huber" => (
            fit(
                json!({
                    "type": "marginal",
                    "bin_edges": vec![vec![-0.5, 0.0, 0.5]; 20],
                    "bin_stat": "huber",
                }),
                20,
            ),
            Shape::Linear,
        ),
        "ew_class" => {
            let model = json!({
                "type": "ew_class",
                "classes": ["0", "1", "2"],
                "precision_prior": 1.0,
            });
            let mut s = fit(model, 20);
            s.as_object_mut().expect("an object").remove("min_weight");
            (s, Shape::Labelled)
        }
        "deco" | "ew_cov_lags" => {
            let model = if id == "deco" {
                json!({"type": "deco"})
            } else {
                json!({"type": "ew_cov", "stats": ["mean", "cov"], "lags": [1, 2, 3, 4, 5]})
            };
            let mut s = fit(model, 20);
            let o = s.as_object_mut().expect("an object");
            o.remove("targets");
            o.remove("min_weight");
            (s, Shape::Linear)
        }
        "hmm" => (
            four(
                json!({"type": "hmm", "n_components": 2, "precision_prior": 1.0}),
                true,
            ),
            Shape::Blobs,
        ),
        // The kernel estimator, as `scripts/benchmark.py`'s row.
        // A group that closes is required; the case counts the rows the
        // block takes in, not its close, which the bank calls.
        "rcov" => {
            let mut s = four(
                json!({"type": "rcov", "kind": "kernel", "block_rows": 2000}),
                false,
            );
            s["group"] = json!("g");
            s["group_close"] = json!("monotone");
            (s, Shape::Blobs)
        }
        // A continuous column at the default cap of 256 counters: nearly
        // every value new, so past the cap on nearly every row (F-2).
        "audit_continuous" => (
            json!({
                "name": "m",
                "model": {"type": "audit"},
                "features": ["x0"],
                "clock": "t",
                "gap_cap": 100.0,
            }),
            Shape::Linear,
        ),
        // The full covariance, whose runs each factor a 4 x 4 scale a row.
        "bocpd_full" => (
            four(
                json!({
                    "type": "bocpd",
                    "covariance_type": "full",
                    "degrees_of_freedom_prior": 6.0,
                    "prune_below": 1e-6,
                    "max_run": 200,
                }),
                false,
            ),
            Shape::Blobs,
        ),
        other => panic!("no case {other:?}"),
    }
}

/// A case's spec, filled and checked as a bank checks it.
pub fn spec(id: &str) -> Spec {
    let (value, _) = case_spec(id);
    let mut spec: Spec = serde_json::from_value(value).unwrap_or_else(|e| panic!("{id}: {e}"));
    spec.check().unwrap_or_else(|e| panic!("{id}: {e}"));
    spec
}

/// A case's rows: warm-up and measured block, from one seed.
fn rows(id: &str, spec: &Spec) -> Rows {
    let (_, shape) = case_spec(id);
    let block = if WIDE_CASES.contains(&id) || id == "bank_ew_cov_wide_t4" {
        WIDE_ROWS
    } else {
        ROWS
    };
    let groups = if grouped(id) { GROUPS } else { 0 };
    Rows::generate(shape, spec.k(), spec.m(), warm_rows(id) + block, groups, 7)
}

/// Rows a case steps before its measured block.
fn warm_rows(id: &str) -> usize {
    if grouped(id) {
        GROUPED_WARM_ROWS
    } else {
        WARM_ROWS
    }
}

/// A model case, warmed up: the model, the rows of its measured block, and
/// the `Step` it is stepped into, as the bank keeps one per model.
pub struct ModelRun {
    pub model: AnyModel,
    rows: Rows,
    step: rheon_polars::rheon_core::Step,
}

/// The setup of a model case: the model the spec builds, stepped through
/// the warm-up rows.
pub fn model_setup(id: &str) -> ModelRun {
    let spec = spec(id);
    let mut built = build_models(&spec).unwrap_or_else(|e| panic!("{id}: {e}"));
    assert_eq!(built.len(), 1, "{id}: one instance");
    let (_, mut model) = built.pop().expect("one instance");
    let rows = rows(id, &spec);
    let mut step = rheon_polars::rheon_core::Step::default();
    for i in 0..WARM_ROWS {
        let (x, y, d) = rows.row(i);
        model.step_into(x, y, d, 1.0, &mut step);
    }
    // Rayon's global pool, which a solve asks for its size, built here at
    // the latest rather than by the first solve of the block:
    // `sgd_huber_refit`'s first is its refit, inside the block, where
    // building the pool read the environment (at a cost that grows with its
    // size) and spawned the pool's threads, whose allocations moved the heap
    // the block then ran on, about 0.01% with the size of the process's
    // environment (task 292, P8). After the warm-up, not before the model:
    // a case whose warm-up already built it keeps the heap it had, where
    // building it first laid the heap out in another order (the VM's
    // measurement, variant A).
    rayon::current_num_threads();
    ModelRun { model, rows, step }
}

/// What a measured block reports: a checksum, so no row's work can be
/// optimized away, and how many rows predicted a number.
#[derive(Debug, Default, Clone, Copy)]
pub struct Block {
    pub checksum: f64,
    pub predicted: usize,
}

/// The measured block of a model case: [`ROWS`] rows of
/// `AnyModel::step_into`, each with weight 1, into the one `Step` the case
/// keeps, as the bank steps a model (docs/PLAN.md task 270, M-4).
#[unsafe(no_mangle)]
#[inline(never)]
pub fn rheon_budget_model_rows(run: &mut ModelRun) -> Block {
    let mut out = Block::default();
    for i in WARM_ROWS..run.rows.len() {
        let (x, y, d) = run.rows.row(i);
        run.model.step_into(x, y, d, 1.0, &mut run.step);
        let step = &run.step;
        let finite: f64 = step.pred.iter().filter(|p| p.is_finite()).sum();
        out.predicted += usize::from(step.pred.iter().any(|p| p.is_finite()));
        out.checksum += finite + step.n_eff;
    }
    out
}

/// A clock case, warmed up.
pub struct ClockRun {
    pub state: ClockState,
    cfg: ClockCfg,
    caps: ExactCaps,
    clocks: Vec<ClockValue>,
}

/// The setup of a clock case: `ClockState::advance_stamped`, as the bank
/// calls it on every row, over a float clock in seconds or a temporal one
/// in nanoseconds since 2024, each with a cap of 100 seconds.
pub fn clock_setup(id: &str) -> ClockRun {
    let mut g = Lcg(11);
    let mut seconds = 1_000.0_f64;
    let mut ns: i64 = 1_704_067_200_000_000_000;
    let mut clocks = Vec::with_capacity(WARM_ROWS + ROWS);
    for i in 0..WARM_ROWS + ROWS {
        let step = if i == 0 { 0.0 } else { step_of(g.next()) };
        seconds += step;
        ns += (step * 1e9) as i64;
        clocks.push(match id {
            "clock_float" => ClockValue::F64(seconds),
            "clock_temporal" => ClockValue::Ns(ns),
            other => panic!("no case {other:?}"),
        });
    }
    let cfg = ClockCfg {
        gap_cap: 100.0,
        ..ClockCfg::default()
    };
    let caps = ExactCaps {
        gap_cap_ns: (id == "clock_temporal").then_some(100_000_000_000),
        session_gap_ns: None,
    };
    let mut state = ClockState::new();
    for &c in &clocks[..WARM_ROWS] {
        state.advance_stamped(&cfg, &caps, Some(c), None, true);
    }
    ClockRun {
        state,
        cfg,
        caps,
        clocks,
    }
}

/// The measured block of a clock case: [`ROWS`] accepted rows.
#[unsafe(no_mangle)]
#[inline(never)]
pub fn rheon_budget_clock_rows(run: &mut ClockRun) -> Block {
    let mut out = Block::default();
    for &c in &run.clocks[WARM_ROWS..] {
        let adv = run
            .state
            .advance_stamped(&run.cfg, &run.caps, Some(c), None, true);
        out.checksum += adv.d_clock;
        out.predicted += usize::from(adv.stamp.is_some());
    }
    out
}

/// A bank case, warmed up: the bank and the frame of its measured block.
pub struct BankRun {
    pub bank: Bank,
    frame: DataFrame,
}

/// The setup of a bank case: a bank of the case's one spec, fed the
/// warm-up rows as one frame. A wide or grouped case's bank is given its
/// pool size:
/// one thread is the process default's pool when that has one thread, as
/// under the bench, so the rows are learned on the thread that calls.
pub fn bank_setup(id: &str) -> BankRun {
    let spec = spec(id);
    let rows = rows(id, &spec);
    let mut bank = Bank::new(vec![spec]).unwrap_or_else(|e| panic!("{id}: {e}"));
    let threads = match id {
        "bank_ew_cov_wide_t1" => Some(1),
        "bank_ew_cov_wide_t4" => Some(4),
        id if grouped(id) => Some(1),
        _ => None,
    };
    if threads.is_some() {
        rheon_polars::pool().unwrap_or_else(|e| panic!("{id}: {e}"));
        bank.set_threads(threads)
            .unwrap_or_else(|e| panic!("{id}: {e}"));
    }
    let warm = warm_rows(id);
    bank.fit_predict(&rows.frame(0, warm))
        .unwrap_or_else(|e| panic!("{id}: {e}"));
    let frame = rows.frame(warm, rows.len());
    BankRun { bank, frame }
}

/// The measured block of a bank case: one `Bank::fit_predict` of [`ROWS`]
/// rows, below the bank's parallel threshold, so every row is learned on
/// the thread that calls it. The bench calls this on the bank's own pool,
/// where `fit_predict`'s hand-off to the pool runs inline; Callgrind counts
/// the thread an entry point runs on, and nothing of the waiting caller.
/// The output is returned, so its drop falls outside the count. A wide
/// case's block is [`WIDE_ROWS`] rows, on its bank's pool of one thread or
/// four.
#[unsafe(no_mangle)]
#[inline(never)]
pub fn rheon_budget_bank_rows(run: &mut BankRun) -> Vec<Column> {
    run.bank
        .fit_predict(&run.frame)
        .unwrap_or_else(|e| panic!("{e}"))
}

/// A window case, warmed up: the core and the frame of its measured block.
pub struct WindowsCase {
    pub run: WindowsRun,
    frame: DataFrame,
}

/// A window case's formula, as `rp.ops` writes it.
fn window_tree(id: &str) -> Value {
    let op = match id {
        "window_ewm_mean_w10k" => "ewm_mean",
        "window_ewm_sum_w10k" => "ewm_sum",
        "window_ewm_quantile_w10k" => {
            return json!(["ewm_quantile", ["col", "x0"], {"half_life": 1000.0, "window_size": 12_500.0, "quantile": 0.5}]);
        }
        "window_ewm_quantile_running" => {
            return json!(["ewm_quantile", ["abs", ["col", "x0"]], {"half_life": 1000.0, "quantile": 0.5}]);
        }
        "window_ewm_sum_groups10k" => {
            return json!(["ewm_sum", ["col", "x0"], {"half_life": 1000.0}]);
        }
        "window_invert_cubic" => return invert_tree(),
        "window_ewm_mean_running" => {
            return json!(["ewm_mean", ["col", "x0"], {"half_life": 1000.0}]);
        }
        "window_clock_step_running" => {
            return json!(["clock_step", ["col", "t"], {"half_life": 1000.0, "stat": "mean"}]);
        }
        other => panic!("no case {other:?}"),
    };
    json!([op, ["col", "x0"], {"half_life": 1000.0, "window_size": 12_500.0}])
}

/// The setup of a window case: the core over one feature column and the
/// clock, fed the warm-up rows as one frame. The grouped case adds a group
/// column, the row's place modulo [`WINDOW_GROUPS`], on the same irregular
/// clock, so each row is a stamp of its own; its cap of 1e12 never cuts a
/// group, and its warm-up of two rows a group puts every group on the list.
/// The invert case ([`INVERT_CASE`]) adds a second formula, a running sum,
/// as a run of inverts alone is refused.
pub fn windows_setup(id: &str) -> WindowsCase {
    let grouped = id.ends_with("_groups10k");
    let mut config = json!({
        "formulas": [{"name": "w", "tree": window_tree(id)}],
        "clock": "t",
        "gap_cap": 100.0,
    });
    if id == INVERT_CASE {
        let sum = json!(["ewm_sum", ["col", "x0"], {"half_life": 1000.0}]);
        config["formulas"]
            .as_array_mut()
            .expect("a list of formulas")
            .push(json!({"name": "s", "tree": sum}));
    }
    let warm_rows = if grouped {
        config["gap_cap"] = json!(1e12);
        config["group"] = json!("g");
        2 * WINDOW_GROUPS
    } else {
        WARM_ROWS
    };
    let config: WindowsConfig =
        serde_json::from_value(config).unwrap_or_else(|e| panic!("{id}: {e}"));
    let rows = Rows::generate(Shape::Linear, 1, 0, warm_rows + ROWS, 0, 7);
    let frame = |from: usize, to: usize| {
        let mut f = rows.frame(from, to);
        if grouped {
            let g: Vec<i64> = (from..to).map(|i| (i % WINDOW_GROUPS) as i64).collect();
            f.with_column(Column::new("g".into(), g))
                .expect("a column of the frame's height");
        }
        f
    };
    let warm = frame(0, warm_rows);
    let mut run = WindowsRun::new(config, warm.schema()).unwrap_or_else(|e| panic!("{id}: {e}"));
    run.feed(&warm, None)
        .unwrap_or_else(|e| panic!("{id}: {e}"));
    let frame = frame(warm_rows, rows.len());
    WindowsCase { run, frame }
}

/// The measured block of a window case: one `WindowsRun::feed` of [`ROWS`]
/// rows. A backward window emits each row as it is read, so the output is
/// the block's values, returned so its drop falls outside the count.
#[unsafe(no_mangle)]
#[inline(never)]
pub fn rheon_budget_windows_rows(run: &mut WindowsCase) -> DataFrame {
    run.run
        .feed(&run.frame, None)
        .unwrap_or_else(|e| panic!("{e}"))
}

/// A bar case, warmed up: the core and the frame of its measured block.
pub struct BarsCase {
    pub bars: Bars,
    frame: DataFrame,
}

/// The trades of a bar case: an irregular clock `t`, a price `p` on a grid
/// of 1/64 walking a tick at most, a size `q` of 1 to 100, and the group
/// `g`, the row's place modulo [`BAR_GROUPS`].
fn bar_trades(n: usize, seed: u64) -> DataFrame {
    let mut r = Lcg(seed);
    let (mut t, mut p, mut q) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    );
    let (mut clock, mut price) = (0.0, 100.0);
    for _ in 0..n {
        clock += step_of(r.next());
        price += (3.0 * r.next()).floor() / 64.0 - 1.0 / 64.0;
        t.push(clock);
        p.push(price);
        q.push((100.0 * r.next()).floor() + 1.0);
    }
    let g: Vec<i64> = (0..n).map(|i| (i % BAR_GROUPS) as i64).collect();
    df!("t" => t, "p" => p, "q" => q, "g" => g).expect("columns of one height")
}

/// A bar case's call: volume bars of 1,000 with six aggregates.
pub fn bars_config(id: &str) -> BarsConfig {
    let agg = |kind: AggKind, column: Option<&str>, name: &str| AggSpec {
        kind,
        column: column.map(str::to_string),
        name: name.into(),
    };
    BarsConfig {
        kind: BarKind::Sum,
        clock: "t".into(),
        threshold: BarThreshold::Value(1_000.0),
        quantity: Some("q".into()),
        group: id.ends_with("_groups100").then(|| "g".into()),
        weight: None,
        session: None,
        gap_cap: None,
        max_span: None,
        restart_after_step_back: None,
        keep_partial: true,
        monotone: false,
        max_state_bytes: 256 * 1024,
        max_total_bytes: 1 << 30,
        aggregates: vec![
            agg(AggKind::First, Some("p"), "open"),
            agg(AggKind::Max, Some("p"), "high"),
            agg(AggKind::Min, Some("p"), "low"),
            agg(AggKind::Last, Some("p"), "close"),
            agg(AggKind::Sum, Some("q"), "volume"),
            agg(AggKind::Len, None, "rows"),
        ],
        call: [("clock".to_string(), "t".to_string())]
            .into_iter()
            .collect(),
    }
}

/// The setup of a bar case: the core fed the [`WARM_ROWS`] warm-up rows as
/// one frame, 20 a group in the grouped case, so every group has closed a
/// bar and holds an open one when the block starts.
pub fn bars_setup(id: &str) -> BarsCase {
    let warm = WARM_ROWS;
    let trades = bar_trades(warm + ROWS, 11);
    let first = trades.slice(0, warm);
    let mut bars =
        Bars::new(bars_config(id), first.schema()).unwrap_or_else(|e| panic!("{id}: {e}"));
    bars.feed(&first, None)
        .unwrap_or_else(|e| panic!("{id}: {e}"));
    let frame = trades.slice(warm as i64, ROWS);
    BarsCase { bars, frame }
}

/// The measured block of a bar case: one `Bars::feed` of [`ROWS`] rows. The
/// bars it closes are returned, so their drop falls outside the count.
#[unsafe(no_mangle)]
#[inline(never)]
pub fn rheon_budget_bars_rows(run: &mut BarsCase) -> DataFrame {
    run.bars
        .feed(&run.frame, None)
        .unwrap_or_else(|e| panic!("{e}"))
}
