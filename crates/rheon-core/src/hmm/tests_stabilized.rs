//! Stabilized forgetting ([`Stabilized`]), the default under `full` and
//! `diag` in a model that learns: review fable2's C2, C3 and C4
//! reproductions, which a build before it failed, its factor form against
//! the exact recursion, and the contract a row keeps (predict, zero-weight
//! rows).

use std::cell::Cell;

use super::*;
use crate::OnlineModel;

thread_local! {
    /// Set by a test: [`Hmm::new`] builds no prior, as a build before
    /// stabilized forgetting did, every state under the decaying ridge.
    pub(super) static NO_PRIOR: Cell<bool> = const { Cell::new(false) };
}

/// `f` run as a build before stabilized forgetting ran it ([`NO_PRIOR`]).
pub(super) fn before_the_prior<T>(f: impl FnOnce() -> T) -> T {
    NO_PRIOR.with(|c| c.set(true));
    let out = f();
    NO_PRIOR.with(|c| c.set(false));
    out
}

fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

/// The review's configuration: two states, seeded from the first 40 rows.
pub(super) fn cfg(d: usize, hl: f64, covariance: Covariance) -> HmmCfg {
    HmmCfg {
        n_features: d,
        k: 2,
        decay: Decay::Halflife(hl),
        covariance,
        precision_prior: 1e-3,
        min_weight: 0.0,
        learn: true,
        transition_prior: 1.0,
        transition: None,
        means: None,
        covs: None,
        warm_rows: 40,
        seed_rule: SeedRule::First,
        seed: 7,
        tvtp: None,
        prior_rows: None,
        warm_clock: None,
    }
}

/// The review's stream: two regimes at `−1` and `1` in runs of 7 rows with
/// unit-range noise, `edit` changing a row and giving its clock step. Its
/// NaN rows among the last 500 and the solve failures, for the model as
/// built (`stabilized`, at the default `P₀ = d`) or as a build before it,
/// under the decaying ridge alone.
pub(super) fn run(
    cfg: HmmCfg,
    stabilized: bool,
    n: usize,
    edit: impl Fn(usize, &mut Vec<f64>) -> f64,
) -> (usize, u64) {
    let d = cfg.n_features;
    let mut m = if stabilized {
        Hmm::new(cfg).unwrap()
    } else {
        before_the_prior(|| Hmm::new(cfg).unwrap())
    };
    assert_eq!(m.stabilized.is_some(), stabilized);
    let mut s = 5u64;
    let mut nan_tail = 0;
    for i in 0..n {
        let c = if (i / 7) % 2 == 0 { -1.0 } else { 1.0 };
        let mut x: Vec<f64> = (0..d).map(|_| c + lcg(&mut s)).collect();
        let d_clock = edit(i, &mut x);
        let st = m.step(&x, &[], d_clock, 1.0);
        if i + 500 >= n && st.pred.iter().any(|v| v.is_nan()) {
            nan_tail += 1;
        }
    }
    (nan_tail, m.solve_failures)
}

/// C2a: under `diag`, a feature that never varies has a variance of 0 in
/// every state. The decaying ridge of a build before the prior went to 0
/// with it, and every row was NaN from about 1,065 half-lives; the prior's
/// floor keeps the variance positive. That build still fails here, so the
/// stream reproduces the fault.
#[test]
fn c2a_a_constant_column_reads_under_the_prior() {
    let edit = |_: usize, x: &mut Vec<f64>| {
        x[2] = 0.0;
        1.0
    };
    let before = run(cfg(3, 2.0, Covariance::Diagonal), false, 3000, edit);
    assert!(
        before.0 > 0,
        "the reproduction no longer fails without the prior: {before:?}"
    );
    let (nan, fails) = run(cfg(3, 2.0, Covariance::Diagonal), true, 3000, edit);
    assert_eq!((nan, fails), (0, 0));
}

/// C2b: a feature varies, sticks for 1,200 half-lives, and moves again.
/// Without the prior every row after it moves was NaN; under it none is.
#[test]
fn c2b_a_feature_that_sticks_and_moves_again_reads_under_the_prior() {
    let edit = |i: usize, x: &mut Vec<f64>| {
        if (200..2600).contains(&i) {
            x[2] = 0.25;
        }
        1.0
    };
    for cov in [Covariance::Full, Covariance::Diagonal] {
        let before = run(cfg(3, 2.0, cov), false, 4600, edit);
        assert!(
            before.0 > 0,
            "{cov:?}: no longer fails without the prior: {before:?}"
        );
        let (nan, fails) = run(cfg(3, 2.0, cov), true, 4600, edit);
        assert_eq!((nan, fails), (0, 0), "{cov:?}");
    }
}

/// C3a: six rows each 300 half-lives apart. Without the prior the factor
/// form (13 features) kept pivots near `1e-170` and every later row was
/// NaN; under it each gap's read stands on the prior, at 12 features and 13.
#[test]
fn c3a_a_burst_of_gaps_reads_under_the_prior() {
    let edit = |i: usize, _: &mut Vec<f64>| if (100..106).contains(&i) { 6000.0 } else { 1.0 };
    let before = run(cfg(13, 20.0, Covariance::Full), false, 1000, edit);
    assert!(
        before.0 > 0,
        "no longer fails without the prior: {before:?}"
    );
    for d in [12, 13] {
        let (nan, fails) = run(cfg(d, 20.0, Covariance::Full), true, 1000, edit);
        assert_eq!((nan, fails), (0, 0), "{d}");
    }
}

/// C3b: the last feature sticks at `0.25` from row 100 on. Without the
/// prior the factor form went NaN at about 2,050 half-lives; under it the
/// model reads on.
#[test]
fn c3b_a_stuck_feature_reads_under_the_prior() {
    let edit = |d: usize| {
        move |i: usize, x: &mut Vec<f64>| {
            if i >= 100 {
                x[d - 1] = 0.25;
            }
            1.0
        }
    };
    let before = run(cfg(13, 2.0, Covariance::Full), false, 6000, edit(13));
    assert!(
        before.0 > 0,
        "no longer fails without the prior: {before:?}"
    );
    for d in [12, 13] {
        let (nan, fails) = run(cfg(d, 2.0, Covariance::Full), true, 6000, edit(d));
        assert_eq!((nan, fails), (0, 0), "{d}");
    }
}

/// C4: regime B (at 6) is absent for 20 half-lives and returns for 400
/// rows. Without the prior `full` and `diag` filtered none of its returning
/// rows into its own state; under it nearly all, at 3 features and at 13.
#[test]
fn c4_a_returning_regime_is_recognised_under_the_prior() {
    for d in [3, 13] {
        for cov in [Covariance::Full, Covariance::Diagonal] {
            let (hl, absent) = (20.0, 400);
            let mut m = Hmm::new(cfg(d, hl, cov)).unwrap();
            let mut s = 9u64;
            let (mut b_state, mut in_b, mut b_rows) = (0, 0, 0);
            for i in 0..400 + absent + 400 {
                let g = if i < 400 {
                    (i / 20) % 2
                } else {
                    usize::from(i >= 400 + absent)
                };
                let c = if g == 0 { 0.0 } else { 6.0 };
                let x: Vec<f64> = (0..d).map(|_| c + lcg(&mut s)).collect();
                m.step(&x, &[], 1.0, 1.0);
                if i == 399 {
                    b_state = usize::from(
                        (m.state_means(1)[0] - 6.0).abs() < (m.state_means(0)[0] - 6.0).abs(),
                    );
                }
                if i >= 400 + absent + 5 {
                    b_rows += 1;
                    let f = m.filtered();
                    in_b += usize::from((f[1] > f[0]) == (b_state == 1));
                }
            }
            assert!(in_b * 100 >= b_rows * 99, "{d} {cov:?}: {in_b}/{b_rows}");
        }
    }
}

use super::stabilized::{EAGER, TAU};
use super::tests_factor::{blobs, wide};
use crate::oracle::exact::Recursion;
use crate::solve::QuadWork;

/// `ln Σ p̃ₗ e^(fₗ)` and the posterior it normalizes, about the largest
/// term, as `read_into` forms them.
fn filtered(pred: &[f64], logf: &[f64]) -> (f64, Vec<f64>) {
    let terms: Vec<f64> = pred
        .iter()
        .zip(logf)
        .map(|(p, f)| {
            if *p > 0.0 {
                p.ln() + f
            } else {
                f64::NEG_INFINITY
            }
        })
        .collect();
    let top = terms.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = terms.iter().map(|v| (v - top).exp()).collect();
    let z: f64 = exps.iter().sum();
    (top + z.ln(), exps.iter().map(|e| e / z).collect())
}

fn max(v: &[f64]) -> f64 {
    v.iter().cloned().fold(0f64, f64::max)
}

/// The factor form's reads against two recursions carried in 512 bits from
/// each state's mean-form data co-moments `M ← a·M + a·b·δδᵀ` (the model's
/// own weights and responsibilities, as `tests_factor`'s track carries
/// them), each read at the model's aged weight `W`:
///
/// - its own, `(W·M + diag(f))/(W + P₀)` with `f` the prior it holds: per
///   row read, each state's `σ·L Lᵀ` off it (units of rounding in its
///   largest diagonal entry), and the near log densities' error;
/// - stabilized forgetting's, `(W·M + P₀ψ)/(W + P₀)`: the near log
///   densities', the posterior's and the score's distance from it, which
///   is the prior not yet topped up (`eager` tops every coordinate up
///   before every read; one whose `ψ` fell since keeps the larger prior).
#[derive(Default, Debug)]
struct Track {
    /// The factored prior's distance from its target at each read before
    /// the first long gap, as a share of the target: the largest over the
    /// coordinates.
    band: Vec<f64>,
    off: Vec<f64>,
    logf: Vec<f64>,
    gap_logf: Vec<f64>,
    post: Vec<f64>,
    score: Vec<f64>,
}

fn track(d: usize, decay: Decay, n: usize, seed: u64, eager: bool) -> Track {
    EAGER.with(|c| c.set(eager));
    let p0 = d as f64;
    let mut m = Hmm::new(wide(d, decay)).unwrap();
    let mut e = Hmm::new_comoments(wide(d, decay)).unwrap();
    let base = -0.5 * d as f64 * std::f64::consts::TAU.ln();
    let mut quad = QuadWork::default();
    let mut exact: Vec<Recursion> = Vec::new();
    let mut out = Track::default();
    for (i, (x, dt)) in blobs(d, n, seed).iter().enumerate() {
        let w = if i % 23 == 11 { 0.0 } else { 1.0 };
        // Gaps of 10 half-lives (about), of 1,060 (a subnormal decay) and
        // of 1,500 (none left): the last two start every state again.
        let dt = match i {
            _ if i == n / 3 => 1060.0 * 20.0,
            _ if i == 2 * n / 3 => 1500.0 * 20.0,
            _ if i % 97 == 50 => 200.0,
            _ => *dt,
        };
        let before = m.clone();
        let step = m.step(x, &[], dt, w);
        if !before.seeded {
            e.step(x, &[], dt, w);
            if m.seeded {
                assert!(m.factored() && !e.factored());
                exact = e
                    .states
                    .iter()
                    .map(|c| Recursion::of(&c.comoments().to_full()))
                    .collect();
            }
            continue;
        }
        let lam = decay.factor(dt);
        let mut aged = before.clone();
        aged.age_stabilized(lam, &mut RowWork::default(), w == 0.0);
        let st = aged.stabilized().unwrap();
        let target = st.targets();
        let held = st.held.clone();
        if i < n / 3 {
            out.band.push(
                held.iter()
                    .zip(&target)
                    .map(|(h, t)| (h / t - 1.0).abs())
                    .fold(0.0, f64::max),
            );
        }
        let mut deltas = Vec::new();
        let (mut lf, mut le) = ([0.0; 2], [0.0; 2]);
        for s in 0..2 {
            let f = &aged.factor_states[s];
            let delta: Vec<f64> = x
                .iter()
                .enumerate()
                .map(|(j, &v)| f.deviation(j, v))
                .collect();
            let own = exact[s].shrunk(f.n_eff(), p0, &held);
            let (q_o, ld_o) = own.quad_log_det(&delta);
            let ex = exact[s].shrunk(f.n_eff(), p0, &target);
            let (q, ld) = ex.quad_log_det(&delta);
            let (ld_f, q_f) = f.read(&delta, &mut quad);
            out.off
                .push(own.off_factor(&f.factor().lower_rows(), f.sigma()));
            if q_o < 3.0 * d as f64 {
                out.logf.push(0.5 * (ld_f + q_f - ld_o - q_o).abs());
            }
            // The gap before the first long gap: after one, `ŝ` starts
            // again from a row and `ψ` falls faster than `f` decays, which
            // the form takes no prior back from (the module docs).
            if q < 3.0 * d as f64 && i < n / 3 {
                out.gap_logf.push(0.5 * (ld_f + q_f - ld - q).abs());
            }
            lf[s] = base - 0.5 * ld_f - 0.5 * q_f;
            le[s] = base - 0.5 * ld - 0.5 * q;
            deltas.push(delta);
        }
        let read = step.pred[5].is_finite();
        assert!(read, "{d} {decay:?} row {i}: not read");
        let pred = &step.pred[2..4];
        let (score_f, post_f) = filtered(pred, &lf);
        assert_eq!(
            score_f.to_bits(),
            step.pred[5].to_bits(),
            "row {i}: the model's score"
        );
        let (score_e, post_e) = filtered(pred, &le);
        if i >= n / 3 {
            continue_shadow(&mut exact, &aged, &m, w, &deltas);
            continue;
        }
        out.post.push(
            post_f
                .iter()
                .zip(&post_e)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max),
        );
        out.score.push((score_f - score_e).abs());
        continue_shadow(&mut exact, &aged, &m, w, &deltas);
    }
    EAGER.with(|c| c.set(false));
    out
}

/// The data's step at the weight the model's filter gave each state.
fn continue_shadow(exact: &mut [Recursion], aged: &Hmm, m: &Hmm, w: f64, deltas: &[Vec<f64>]) {
    for s in 0..2 {
        let ws = w * m.p[s];
        let wa = aged.factor_states[s].n_eff();
        if ws > 0.0 {
            let w_new = wa + ws;
            let (a, b) = (wa / w_new, ws / w_new);
            if a == 0.0 {
                exact[s].reset(0.0);
            } else {
                exact[s].step(a, a * b, &deltas[s]);
            }
        }
        assert_eq!(
            m.factor_states[s].n_eff(),
            if ws > 0.0 { wa + ws } else { wa }
        );
    }
}

/// Topped up before every read, the factor form holds the exact recursion
/// to rounding, as today's factor form holds its own (`tests_factor`): its
/// matrix within a few thousand units of rounding and its densities within
/// `2e-11`, its own recursion's and stabilized forgetting's alike. Lazily --
/// the form as built -- it holds its own recursion as closely, and reads off
/// stabilized forgetting's by the prior not yet moved to its target: within
/// `τ` of each coordinate's prior at every read (held here), which is
/// `P₀/(W + P₀)` of the matrix. Measured 2026-11-20 (13 and 20 features,
/// half-lives of 20 and 500 rows, 2,000 rows, arm64): eagerly 145 units of
/// rounding, densities within 1e-12 of either recursion; lazily 131 units
/// of rounding, up to 0.33 nats in a near density at a half-life of 20 rows
/// (0.35 at 500), and 8.9e-3 in a posterior.
fn hold_the_exact_recursion(widths: &[usize], n: usize) {
    for &d in widths {
        for decay in [Decay::Halflife(20.0), Decay::Halflife(500.0)] {
            for eager in [true, false] {
                let t = track(d, decay, n, 3, eager);
                println!(
                    "{d} {decay:?} eager {eager}: off {:.0}, logf {:.1e}; from \
                     stabilized forgetting: logf {:.1e}, post {:.1e}, score {:.1e}",
                    max(&t.off),
                    max(&t.logf),
                    max(&t.gap_logf),
                    max(&t.post),
                    max(&t.score)
                );
                assert!(max(&t.off) < 4096.0, "{d} {decay:?} {eager}");
                assert!(max(&t.logf) < 2e-11, "{d} {decay:?} {eager}");
                // Every read stands on a prior within `τ` of its target in
                // every coordinate; topped up before every read, on the
                // target itself.
                let band = if eager { 0.0 } else { TAU };
                assert!(
                    max(&t.band) <= band,
                    "{d} {decay:?} {eager}: {:e} of the target off",
                    max(&t.band)
                );
                if eager {
                    assert!(max(&t.gap_logf) < 2e-11, "{d} {decay:?}");
                    assert!(max(&t.post) < 1e-12, "{d} {decay:?}");
                } else {
                    assert!(max(&t.gap_logf) < 1.0, "{d} {decay:?}");
                    assert!(max(&t.post) < 0.05, "{d} {decay:?}");
                }
            }
        }
    }
}

#[test]
fn the_factor_form_holds_the_exact_recursion() {
    hold_the_exact_recursion(&[13], 600);
}

/// [`the_factor_form_holds_the_exact_recursion`] over the grid its bounds
/// were measured on: 13 and 20 features, 2,000 rows a run.
#[test]
#[ignore = "extended: the stabilized factor form against the 512-bit recursion over its grid"]
fn the_factor_form_holds_the_exact_recursion_over_the_grid() {
    hold_the_exact_recursion(&[13, 20], 2000);
}

/// Rows of [`blobs`], a zero-weight one every 13th and a gap of 30 clock
/// units every 50th.
fn rows(d: usize, n: usize) -> Vec<(Vec<f64>, f64, f64)> {
    blobs(d, n, 9)
        .into_iter()
        .enumerate()
        .map(|(i, (x, dt))| {
            let w = if i % 13 == 6 { 0.0 } else { 1.0 };
            (x, if i % 50 == 25 { 30.0 } else { dt }, w)
        })
        .collect()
}

fn shapes() -> [(usize, Covariance); 4] {
    [
        (4, Covariance::Diagonal),
        (4, Covariance::Full),
        (13, Covariance::Full),
        (20, Covariance::Diagonal),
    ]
}

/// A model under the prior.
fn model(d: usize, cov: Covariance) -> Hmm {
    Hmm::new(HmmCfg {
        covariance: cov,
        ..wide(d, Decay::Halflife(20.0))
    })
    .unwrap()
}

/// `predict` is the step's read without the step, to the bit, though the
/// states age before the read: it ages a copy.
#[test]
fn predict_is_the_step_without_the_step_under_the_prior() {
    for (d, cov) in shapes() {
        let mut m = model(d, cov);
        for (i, (x, dt, w)) in rows(d, 600).into_iter().enumerate() {
            let p = m.predict(&x, dt);
            let kept = m.clone();
            let s = m.step(&x, &[], dt, w);
            let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&p.pred), bits(&s.pred), "{d} {cov:?} row {i}");
            assert_eq!(kept.predict(&x, dt).pred.len(), p.pred.len());
        }
        assert!(m.stabilized().is_some() && m.factored() == (d >= 13 && cov == Covariance::Full));
    }
}

/// Hard rule 9 under the prior: a zero-weight row inside a gap and the same
/// stream with its clock carried into the next row read alike, to rounding,
/// in every form: the factor form's top-ups on a zero-weight row serve its
/// read alone.
#[test]
fn a_zero_weight_row_is_clock_alone_under_the_prior() {
    for (d, cov) in shapes() {
        let factored = d >= 13 && cov == Covariance::Full;
        let (mut with, mut without) = (model(d, cov), model(d, cov));
        let mut carried = 0.0;
        let mut worst = 0f64;
        for (i, (x, dt, _)) in rows(d, 900).into_iter().enumerate() {
            if i > 150 && matches!(i % 13, 5 | 6) {
                with.step(&x, &[], dt, 0.0);
                carried += dt;
                continue;
            }
            let a = with.step(&x, &[], dt, 1.0);
            let b = without.step(&x, &[], dt + carried, 1.0);
            carried = 0.0;
            for (p, q) in a.pred.iter().zip(&b.pred) {
                if !(p.is_nan() && q.is_nan()) {
                    worst = worst.max((p - q).abs() / (1.0 + q.abs()));
                }
            }
        }
        println!("{d} {cov:?}: {worst:e}");
        assert!(
            worst <= 1e-11,
            "{d} {cov:?} (factored {factored}): {worst:e}"
        );
    }
}

/// Only `full` and `diag` take the prior, and only a model that learns;
/// `prior_rows` is refused where nothing takes it, and where it is not a
/// finite number above 0. Its default is the number of features.
#[test]
fn the_prior_is_for_a_learning_full_or_diag_model() {
    let c = |cov, learn, prior_rows| HmmCfg {
        covariance: cov,
        learn,
        means: Some(vec![-3.0, -3.0, 3.0, 3.0]),
        covs: Some(vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]),
        prior_rows,
        ..wide(2, Decay::Halflife(20.0))
    };
    for cov in [Covariance::Full, Covariance::Diagonal] {
        let m = Hmm::new(c(cov, true, Some(2.5))).unwrap();
        // Given states: `ŝ` from their variances at once.
        let st = m.stabilized().unwrap();
        assert_eq!(st.scale(), (&[1.0, 1.0][..], 2.0));
        assert_eq!(st.prior_rows(), 2.5);
        let m = Hmm::new(c(cov, true, None)).unwrap();
        assert_eq!(m.stabilized().unwrap().prior_rows(), 2.0);
        assert!(Hmm::new(c(cov, false, None)).unwrap().stabilized.is_none());
        let e = Hmm::new(c(cov, false, Some(2.0))).unwrap_err();
        assert!(
            e.contains("prior_rows does not apply with learn off"),
            "{e}"
        );
    }
    assert!(
        Hmm::new(c(Covariance::Shared, true, None))
            .unwrap()
            .stabilized
            .is_none()
    );
    let e = Hmm::new(c(Covariance::Shared, true, Some(2.0))).unwrap_err();
    assert!(
        e.contains("prior_rows does not apply under covariance_type \"tied\""),
        "{e}"
    );
    for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        let e = Hmm::new(c(Covariance::Full, true, Some(bad))).unwrap_err();
        assert!(
            e.contains("prior_rows must be finite and above 0"),
            "{bad}: {e}"
        );
    }
}

/// What [`loaded_under_the_prior`] measured: the continuations of the
/// converted states against the old build's, over the seeded runs.
#[derive(Default, Debug)]
struct Loaded {
    /// The largest `|Δp|` of a filtered or predicted posterior.
    post: f64,
    /// The share of rows whose reported state differs.
    flipped: f64,
    /// Seeded runs converted, of them in the factor form, and warm-up runs.
    seeded: usize,
    factored: usize,
    warming: usize,
}

/// A state of schema 69 or earlier under `full` or `diag` that learns
/// (rule 5's twenty-second exception): it was saved by a build before the
/// prior, under the decaying ridge, in co-moments (schema 68, or below 13
/// features, or `diag`) or in the factor form under the ridge (schema 69,
/// 13 features or more), and keeps no `ŝ`, `V` or `f`. It loads as this
/// build would continue from its moments:
///
/// - `ŝ` the states' variances pooled by weight, `V` their total weight
///   (a factor form's variances read from `σ·L Lᵀ` less its ridge);
/// - a factor form taken up again from its co-moments, at `P₀ψ`:
///   `σ·L Lᵀ = (W·C + P₀Ψ)/(W + P₀)` within rounding, and `f = P₀ψ`;
/// - saved inside its warm-up, it continues to the bit of this build's run.
///
/// It goes on under the prior: neither the old build's continuation to the
/// bit nor within rounding of it. What is held: its `weight_sum` is the old
/// build's to the bit, every output is a number wherever the old build's
/// was, and the state re-saves bytes that load back the same and continue
/// to the bit. `--nocapture` prints how far the posteriors part.
fn loaded_under_the_prior(widths: &[(usize, Covariance)], n: usize) -> Loaded {
    use super::tests_factor::{blobs, wide};
    let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    let mut out = Loaded::default();
    let (mut rows_seen, mut rows_flipped) = (0usize, 0usize);
    for &(d, covariance) in widths {
        for decay in [
            Decay::Halflife(20.0),
            Decay::Halflife(400.0),
            Decay::Halflife(f64::INFINITY),
        ] {
            let factor_width = covariance == Covariance::Full && d >= Hmm::FACTOR_FROM;
            let schemas: &[u32] = if factor_width { &[68, 69] } else { &[69] };
            for &schema in schemas {
                for at in [60, 600] {
                    let cfg = HmmCfg {
                        covariance,
                        ..wide(d, decay)
                    };
                    let what = format!("{d} {covariance:?} {decay:?} schema {schema} at {at}");
                    let rows = blobs(d, n, 41 + d as u64);
                    let w_of = |i: usize| if i % 19 == 7 { 0.0 } else { 1.0 };
                    // The old build: no prior; at 68, no factor form either.
                    let step_old = |m: &mut Hmm, x: &[f64], dt: f64, w: f64| {
                        if schema == 68 {
                            super::tests_factor::old(|| m.step(x, &[], dt, w))
                        } else {
                            m.step(x, &[], dt, w)
                        }
                    };
                    let mut was = before_the_prior(|| Hmm::new(cfg.clone()).unwrap());
                    let mut fresh = Hmm::new(cfg.clone()).unwrap();
                    for (i, (x, dt)) in rows[..at].iter().enumerate() {
                        step_old(&mut was, x, *dt, w_of(i));
                        fresh.step(x, &[], *dt, w_of(i));
                    }
                    assert!(was.stabilized.is_none(), "{what}");
                    let mut state = was.state();
                    state.schema_version = schema;
                    let mut back = Hmm::restore(&state).unwrap();
                    let st = back.stabilized.as_deref().expect("a prior");
                    assert_eq!(st.prior_rows(), d as f64, "{what}");
                    if !was.seeded {
                        // Inside the warm-up: this build's run, to the bit.
                        out.warming += 1;
                        assert_eq!(st.scale().1, 0.0, "{what}");
                        for (i, (x, dt)) in rows[at..].iter().enumerate() {
                            let w = w_of(at + i);
                            let (a, b) = (back.step(x, &[], *dt, w), fresh.step(x, &[], *dt, w));
                            assert_eq!(bits(&a.pred), bits(&b.pred), "{what}, row {i}");
                        }
                        assert_eq!(back, fresh, "{what}");
                        continue;
                    }
                    out.seeded += 1;
                    // The moments the old build kept: each state's weight
                    // and co-moments `C` (a factor form's `σ·L Lᵀ` less the
                    // ridge it has reached).
                    let moments: Vec<(f64, Vec<f64>)> = (0..2)
                        .map(|s| match (was.state_cov(s), was.state_factor(s)) {
                            (Some(c), _) => (c.n_eff(), c.comoments().to_full()),
                            (None, Some(f)) => {
                                let r = was.cfg.precision_prior * f.precision_scale();
                                let mut m = f.matrix();
                                (0..d).for_each(|i| m[i * d + i] -= r);
                                (f.n_eff(), m)
                            }
                            (None, None) => {
                                let c = &was.diag_states[s];
                                let mut m = vec![0.0; d * d];
                                (0..d).for_each(|i| m[i * d + i] = c.var(i));
                                (c.n_eff(), m)
                            }
                        })
                        .collect();
                    let total: f64 = moments.iter().map(|(w, _)| w).sum();
                    let (scale, v) = st.scale();
                    assert_eq!(v.to_bits(), total.to_bits(), "{what}: V");
                    for (i, &si) in scale.iter().enumerate() {
                        let want = moments
                            .iter()
                            .map(|(w, c)| w * c[i * d + i].max(0.0))
                            .sum::<f64>()
                            / total;
                        assert!(
                            (si - want).abs() <= 1e-13 * want.abs(),
                            "{what}: ŝ[{i}] {si} against {want}"
                        );
                    }
                    assert_eq!(back.factored(), factor_width, "{what}");
                    if factor_width {
                        out.factored += 1;
                        let psi = st.psi().to_vec();
                        let held = st.held.clone();
                        assert_eq!(held, st.targets(), "{what}: f at its target");
                        for (s, (w, c)) in moments.iter().enumerate() {
                            let f = &back.factor_states[s];
                            assert_eq!((f.sigma(), f.n_eff()), (1.0, *w), "{what}");
                            let got = f.matrix();
                            let p0 = d as f64;
                            let mut top = 0f64;
                            let mut apart = 0f64;
                            for i in 0..d {
                                for j in 0..d {
                                    let prior = if i == j { p0 * psi[i] } else { 0.0 };
                                    let want = (w * c[i * d + j] + prior) / (w + p0);
                                    apart = apart.max((got[i * d + j] - want).abs());
                                    top = top.max(want.abs());
                                }
                            }
                            assert!(apart <= 1e-13 * top, "{what}: {apart:e} of {top:e}");
                        }
                    }
                    // A converted state saves bytes that load back the same.
                    let bytes = rmp_serde::to_vec_named(&back.state()).unwrap();
                    let again: crate::State = rmp_serde::from_slice(&bytes).unwrap();
                    let mut twin = Hmm::restore(&again).unwrap();
                    assert_eq!(twin, back, "{what}");
                    for (i, (x, dt)) in rows[at..].iter().enumerate() {
                        let w = w_of(at + i);
                        let a = back.step(x, &[], *dt, w);
                        let b = step_old(&mut was, x, *dt, w);
                        let t = twin.step(x, &[], *dt, w);
                        assert_eq!(bits(&a.pred), bits(&t.pred), "{what}, row {i}: the twin");
                        assert_eq!(a.n_eff.to_bits(), b.n_eff.to_bits(), "{what}, row {i}");
                        for (p, q) in a.pred.iter().zip(&b.pred) {
                            assert!(!p.is_nan() || q.is_nan(), "{what}, row {i}: {p} for {q}");
                        }
                        for (p, q) in a.pred[..4].iter().zip(&b.pred[..4]) {
                            out.post = out.post.max((p - q).abs());
                        }
                        rows_seen += 1;
                        rows_flipped += usize::from(a.pred[4] != b.pred[4]);
                    }
                }
            }
        }
    }
    out.flipped = rows_flipped as f64 / rows_seen.max(1) as f64;
    println!("{widths:?}: {out:?}");
    out
}

#[test]
fn a_state_before_schema_70_loads_under_the_prior() {
    let out = loaded_under_the_prior(
        &[
            (4, Covariance::Diagonal),
            (12, Covariance::Full),
            (13, Covariance::Full),
        ],
        900,
    );
    assert!(
        out.seeded > 0 && out.factored > 0 && out.warming > 0,
        "{out:?}"
    );
    assert!(out.post > 0.0, "nothing moved: the prior never acted");
}

/// [`a_state_before_schema_70_loads_under_the_prior`] at 20 and 32 features
/// too, 2,000 rows a run: the grid exception 22's numbers were measured on.
#[test]
#[ignore = "extended: states before schema 70 loaded under the prior, over the grid"]
fn a_state_before_schema_70_loads_under_the_prior_over_the_grid() {
    let out = loaded_under_the_prior(
        &[
            (4, Covariance::Diagonal),
            (20, Covariance::Diagonal),
            (12, Covariance::Full),
            (13, Covariance::Full),
            (20, Covariance::Full),
            (32, Covariance::Full),
        ],
        2000,
    );
    assert!(
        out.seeded > 0 && out.factored > 0 && out.warming > 0,
        "{out:?}"
    );
}

// ---- the pieces of the read and the factor form, one at a time ----

/// A model of `cfg` fed `n` rows of blobs, in the form its width gives it.
fn fed(cfg: HmmCfg, n: usize) -> Hmm {
    let d = cfg.n_features;
    let mut m = Hmm::new(cfg).unwrap();
    for (x, dt) in &super::tests_factor::blobs(d, n, 17) {
        m.step(x, &[], *dt, 1.0);
    }
    m
}

/// Two states given with different shapes: the first `var·I`, the second
/// `var` on the diagonal and `var/20` beside it.
fn shaped_cfg(d: usize, var: f64) -> HmmCfg {
    let mut covs = vec![0.0; 2 * d * d];
    for i in 0..d {
        for j in 0..d {
            covs[i * d + j] = if i == j { var } else { 0.0 };
            covs[d * d + i * d + j] = if i == j { var } else { var / 20.0 };
        }
    }
    HmmCfg {
        means: Some([vec![-3.0; d], vec![3.0; d]].concat()),
        covs: Some(covs),
        ..super::tests_factor::wide(d, Decay::Halflife(50.0))
    }
}

/// `(W·C + P₀Ψ)/(W + P₀)` for state `s`'s co-moments `c` (row-major), its
/// weight, `ψ` and, under a shape, the matrix `G` given for it: row-major.
fn stabilized_matrix(c: &[f64], w: f64, p0: f64, psi: &[f64], g: Option<&[f64]>) -> Vec<f64> {
    let d = psi.len();
    let mut out = vec![0.0; d * d];
    for i in 0..d {
        for j in 0..d {
            let prior = match g {
                Some(g) => p0 * (psi[i] * psi[j]).sqrt() * g[i * d + j],
                None if i == j => p0 * psi[i],
                None => 0.0,
            };
            out[i * d + j] = (w * c[i * d + j] + prior) / (w + p0);
        }
    }
    out
}

fn assert_close(got: &[f64], want: &[f64], what: &str) {
    let top = want.iter().fold(0.0f64, |a, b| a.max(b.abs()));
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= 1e-12 * top,
            "{what}, entry {i}: {g} against {w}"
        );
    }
}

/// The matrix a state reads, in the co-moment form, is `(W·C + P₀Ψ)/(W +
/// P₀)`: its lower triangle, entry by entry, under the diagonal prior and
/// under each state's own shape, at a width where the entries' places in the
/// row-major square tell apart.
#[test]
fn the_state_matrix_is_the_comoments_beside_the_prior() {
    let d = 3;
    // The diagonal prior.
    let m = fed(super::tests_stabilized::cfg(d, 20.0, Covariance::Full), 120);
    assert!(m.seeded && !m.factored());
    let st = m.stabilized().unwrap();
    for s in 0..2 {
        let cov = &m.states[s];
        let got = m.state_matrix(s);
        assert_eq!(got.len(), d * d);
        let want = stabilized_matrix(
            &cov.comoments().to_full(),
            cov.n_eff(),
            st.prior_rows,
            st.psi(),
            None,
        );
        for i in 0..d {
            for j in 0..d {
                if j <= i {
                    assert_close(&got[i * d + j..][..1], &want[i * d + j..][..1], "lower");
                } else {
                    assert_eq!(got[i * d + j], 0.0, "the upper triangle is not written");
                }
            }
        }
    }
    // Each state's own shape.
    let cfg = shaped_cfg(d, 0.5);
    let covs = cfg.covs.clone().unwrap();
    let m = fed(cfg, 60);
    let st = m.stabilized().unwrap();
    assert!(st.shape().is_some());
    for s in 0..2 {
        let cov = &m.states[s];
        let got = m.state_matrix(s);
        let g = &covs[s * d * d..(s + 1) * d * d];
        let want = stabilized_matrix(
            &cov.comoments().to_full(),
            cov.n_eff(),
            st.prior_rows,
            st.psi(),
            Some(g),
        );
        for i in 0..d {
            for j in 0..=i {
                assert_close(
                    &got[i * d + j..][..1],
                    &want[i * d + j..][..1],
                    "shaped lower",
                );
            }
        }
    }
}

/// The factors a model takes up are `(W·C + P₀Ψ)/(W + P₀)` of the
/// co-moments they were made from, at the prior's scale `ψ` it holds then:
/// under the diagonal prior and under a shape.
#[test]
fn the_factors_taken_up_read_the_prior_beside_the_comoments() {
    use super::tests_factor::old;
    let d = 13;
    let cases: [(&str, HmmCfg); 2] = [
        (
            "diagonal",
            super::tests_factor::wide(d, Decay::Halflife(50.0)),
        ),
        ("shaped", shaped_cfg(d, 0.5)),
    ];
    for (what, cfg) in cases {
        let covs = cfg.covs.clone();
        let mut a = old(|| fed(cfg, 150));
        assert!(a.seeded && !a.factored(), "{what}");
        // A scale of the prior that is not 1, so that `ψ` and `1/ψ` differ.
        a.stabilized
            .as_deref_mut()
            .unwrap()
            .seed(&vec![2.0; d], 5.0);
        let mut b = a.clone();
        b.take_up_factors();
        assert!(b.factored(), "{what}");
        let st = a.stabilized().unwrap();
        for s in 0..2 {
            let cov = &a.states[s];
            let g = covs.as_ref().map(|c| &c[s * d * d..(s + 1) * d * d]);
            let want = stabilized_matrix(
                &cov.comoments().to_full(),
                cov.n_eff(),
                st.prior_rows,
                st.psi(),
                g,
            );
            assert_close(&b.factor_states[s].matrix(), &want, what);
        }
    }
}

/// The factors a co-moment model caches are made for each state at its
/// first read, under a shape too: a cache entry each, the factor of the
/// matrix the state reads.
#[test]
fn the_cached_factors_are_made_under_the_diagonal_prior_and_a_shape() {
    let d = 3;
    let cfgs: [(&str, HmmCfg); 2] = [
        (
            "diagonal",
            super::tests_stabilized::cfg(d, 20.0, Covariance::Full),
        ),
        ("shaped", shaped_cfg(d, 0.5)),
    ];
    for (what, cfg) in cfgs {
        let mut m = fed(cfg, 120);
        assert!(m.seeded && !m.factored(), "{what}");
        m.factors.clear();
        assert!(m.factors.peek(0).is_none() && m.factors.peek(1).is_none());
        m.ensure_factors_stabilized();
        for s in 0..2 {
            let f = m
                .factors
                .peek(s)
                .unwrap_or_else(|| panic!("{what}: state {s}"));
            // Its `ln det` is that of the matrix the state reads.
            let sq = m.state_matrix(s);
            let mut full = sq.clone();
            for i in 0..d {
                for j in 0..i {
                    full[j * d + i] = sq[i * d + j];
                }
            }
            let want = crate::oracle::log_det(&full);
            assert!(
                (f.log_det() - want).abs() <= 1e-10,
                "{what}: {} {want}",
                f.log_det()
            );
        }
    }
    // A model still in its warm-up builds none, and so does one that keeps
    // no full covariance.
    let mut m = Hmm::new(super::tests_stabilized::cfg(d, 20.0, Covariance::Full)).unwrap();
    for (x, dt) in super::tests_factor::blobs(d, 10, 17) {
        m.step(&x, &[], dt, 1.0);
    }
    assert!(!m.seeded);
    m.factors.clear();
    m.ensure_factors_stabilized();
    assert!(m.factors.peek(0).is_none() && m.factors.peek(1).is_none());
}

/// A factored prior moved where a state's factor cannot take the move starts
/// that state again at the prior, as factored, over its weight and the
/// prior's, and counts the failure.
#[test]
fn a_move_of_the_prior_that_leaves_no_factor_restarts_the_state_and_counts_it() {
    let d = 13;
    let mut m = fed(super::tests_factor::wide(d, Decay::Halflife(50.0)), 200);
    assert!(m.factored());
    let before = m.solve_failures;
    // A prior held a billion times its target: bringing it back takes more
    // from the matrix than the matrix holds.
    let st = m.stabilized.as_deref_mut().unwrap();
    st.held.iter_mut().for_each(|h| *h *= 1e9);
    let saved = m.age_stabilized(0.99, &mut RowWork::default(), false);
    assert!(saved.is_none());
    assert_eq!(m.solve_failures, before + 2, "a failure a state");
    let st = m.stabilized().unwrap();
    let p0 = st.prior_rows;
    for (s, f) in m.factor_states.iter().enumerate().take(2) {
        let got = f.matrix();
        for i in 0..d {
            for j in 0..d {
                let want = if i == j {
                    st.held[i] / (f.n_eff() + p0)
                } else {
                    0.0
                };
                assert!(
                    (got[i * d + j] - want).abs() <= 1e-12 * want.max(1e-300).max(got[0].abs()),
                    "state {s}, ({i}, {j}): {} against {want}",
                    got[i * d + j]
                );
            }
        }
    }
}

/// A learned row whose rotation leaves a state no factor counts the failure
/// and starts that state again at its prior; the others learn the row.
#[test]
fn a_row_that_leaves_a_state_no_factor_is_counted() {
    #[derive(serde::Serialize)]
    struct Raw {
        w_sum: f64,
        m: Vec<f64>,
        m_lo: Vec<f64>,
        precision_scale: f64,
        sigma: f64,
        l: Vec<f64>,
    }
    let d = 13;
    let mut m = fed(super::tests_factor::wide(d, Decay::Halflife(50.0)), 200);
    assert!(m.factored());
    let real = &m.factor_states[0];
    // A factor with a pivot of 1e-300 over an entry and a pivot of the
    // largest double: rotating a row through the first leaves the second no
    // finite pivot.
    let mut l = Vec::new();
    for j in 0..d {
        for i in j..d {
            l.push(match (i, j) {
                (0, 0) => 1e-300,
                (1, 0) => f64::MAX,
                (1, 1) => f64::MAX,
                (a, b) if a == b => 1.0,
                _ => 0.0,
            });
        }
    }
    let raw = Raw {
        w_sum: real.n_eff(),
        m: real.means().to_vec(),
        m_lo: vec![0.0; d],
        precision_scale: 1.0,
        sigma: 1.0,
        l,
    };
    let bytes = rmp_serde::to_vec_named(&raw).unwrap();
    m.factor_states[0] = rmp_serde::from_slice(&bytes).unwrap();
    let before = m.solve_failures;
    let x: Vec<f64> = m.factor_states[0].means().iter().map(|v| v + 1.0).collect();
    let mut work = RowWork {
        post: vec![0.5, 0.5],
        ..Default::default()
    };
    m.learn_stabilized(&x, 1.0, 1.0, &mut work);
    assert_eq!(m.solve_failures, before + 1);
    // That state is started again at the prior it holds, its share of the
    // weight `1/(W' + P₀)`.
    let f = &m.factor_states[0];
    let st = m.stabilized().unwrap();
    let want = st.held[0] / (f.n_eff() + st.prior_rows);
    assert!(
        (f.matrix()[0] - want).abs() <= 1e-12 * want,
        "{} {want}",
        f.matrix()[0]
    );
}

/// A saved prior that holds a factored prior where the states hold none is
/// refused, though the rest of it is as a state of this build writes it.
#[test]
fn a_saved_prior_that_does_not_fit_its_states_is_refused() {
    let m = fed(super::tests_stabilized::cfg(3, 20.0, Covariance::Full), 120);
    assert!(!m.factored());
    let state = m.state();
    assert!(Hmm::restore(&state).is_ok());
    let mut bad = state.clone();
    if let crate::ModelState::Hmm(h) = &mut bad.model {
        h.stabilized.as_deref_mut().unwrap().held = vec![1.0; 3];
    } else {
        panic!("an hmm state");
    }
    let e = Hmm::restore(&bad).unwrap_err();
    assert!(e.to_string().contains("saved prior does not fit"), "{e}");
}
