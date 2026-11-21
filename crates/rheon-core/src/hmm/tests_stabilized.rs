//! Stabilized forgetting ([`Stabilized`]): review fable2's C2, C3 and C4
//! reproductions under it, its factor form against the exact recursion,
//! and the contract a row keeps (predict, zero-weight rows).

use super::*;
use crate::OnlineModel;

fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

/// The review's configuration: two states, seeded from the first 40 rows.
fn cfg(d: usize, hl: f64, covariance: Covariance) -> HmmCfg {
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
    }
}

/// The review's stream: two regimes at `−1` and `1` in runs of 7 rows with
/// unit-range noise, `edit` changing a row and giving its clock step. Its
/// NaN rows among the last 500 and the solve failures, for a model under
/// stabilized forgetting at `P₀ = d` (`stabilized`) or today's ridge.
fn run(
    cfg: HmmCfg,
    stabilized: bool,
    n: usize,
    edit: impl Fn(usize, &mut Vec<f64>) -> f64,
) -> (usize, u64) {
    let d = cfg.n_features;
    let mut m = if stabilized {
        Hmm::with_stabilized(cfg, d as f64).unwrap()
    } else {
        Hmm::new(cfg).unwrap()
    };
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
/// every state. Today's ridge decays to 0 with it and every row is NaN from
/// about 1,065 half-lives; the prior's floor keeps the variance positive.
#[test]
fn c2a_a_constant_column_reads_under_the_prior() {
    let edit = |_: usize, x: &mut Vec<f64>| {
        x[2] = 0.0;
        1.0
    };
    let today = run(cfg(3, 2.0, Covariance::Diagonal), false, 3000, edit);
    assert!(
        today.0 > 0,
        "the reproduction no longer fails today: {today:?}"
    );
    let (nan, fails) = run(cfg(3, 2.0, Covariance::Diagonal), true, 3000, edit);
    assert_eq!((nan, fails), (0, 0));
}

/// C2b: a feature varies, sticks for 1,200 half-lives, and moves again.
/// Today every row after it moves is NaN; under the prior none is.
#[test]
fn c2b_a_feature_that_sticks_and_moves_again_reads_under_the_prior() {
    let edit = |i: usize, x: &mut Vec<f64>| {
        if (200..2600).contains(&i) {
            x[2] = 0.25;
        }
        1.0
    };
    for cov in [Covariance::Full, Covariance::Diagonal] {
        let today = run(cfg(3, 2.0, cov), false, 4600, edit);
        assert!(today.0 > 0, "{cov:?}: no longer fails today: {today:?}");
        let (nan, fails) = run(cfg(3, 2.0, cov), true, 4600, edit);
        assert_eq!((nan, fails), (0, 0), "{cov:?}");
    }
}

/// C3a: six rows each 300 half-lives apart. Today the factor form (13
/// features) keeps pivots near `1e-170` and every later row is NaN; under
/// the prior each gap's read stands on the prior, at 12 features and 13.
#[test]
fn c3a_a_burst_of_gaps_reads_under_the_prior() {
    let edit = |i: usize, _: &mut Vec<f64>| if (100..106).contains(&i) { 6000.0 } else { 1.0 };
    let today = run(cfg(13, 20.0, Covariance::Full), false, 1000, edit);
    assert!(today.0 > 0, "no longer fails today: {today:?}");
    for d in [12, 13] {
        let (nan, fails) = run(cfg(d, 20.0, Covariance::Full), true, 1000, edit);
        assert_eq!((nan, fails), (0, 0), "{d}");
    }
}

/// C3b: the last feature sticks at `0.25` from row 100 on. Today the factor
/// form goes NaN at about 2,050 half-lives; under the prior it reads on.
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
    let today = run(cfg(13, 2.0, Covariance::Full), false, 6000, edit(13));
    assert!(today.0 > 0, "no longer fails today: {today:?}");
    for d in [12, 13] {
        let (nan, fails) = run(cfg(d, 2.0, Covariance::Full), true, 6000, edit(d));
        assert_eq!((nan, fails), (0, 0), "{d}");
    }
}

/// C4: regime B (at 6) is absent for 20 half-lives and returns for 400
/// rows. Today `full` and `diag` filter none of its returning rows into its
/// own state; under the prior nearly all, at 3 features and at 13.
#[test]
fn c4_a_returning_regime_is_recognised_under_the_prior() {
    for d in [3, 13] {
        for cov in [Covariance::Full, Covariance::Diagonal] {
            let (hl, absent) = (20.0, 400);
            let mut m = Hmm::with_stabilized(cfg(d, hl, cov), d as f64).unwrap();
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

use super::stabilized::EAGER;
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
    off: Vec<f64>,
    logf: Vec<f64>,
    gap_logf: Vec<f64>,
    post: Vec<f64>,
    score: Vec<f64>,
}

fn track(d: usize, decay: Decay, n: usize, seed: u64, eager: bool) -> Track {
    EAGER.with(|c| c.set(eager));
    let p0 = d as f64;
    let mut m = Hmm::with_stabilized(wide(d, decay), p0).unwrap();
    let mut e = Hmm::with_stabilized_comoments(wide(d, decay), p0).unwrap();
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
        aged.age_stabilized(x, lam, &mut RowWork::default(), w == 0.0);
        let st = aged.stabilized().unwrap();
        let target: Vec<f64> = (0..d).map(|j| st.target(j)).collect();
        let held = st.held.clone();
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
/// `τ` of each coordinate's prior, which is `P₀/(W + P₀)` of the matrix.
/// Measured 2026-11-20 (13 and 20 features, half-lives of 20 and 500 rows,
/// 2,000 rows, arm64): eagerly 1.8e3 units of rounding, densities within
/// 1e-12 of either; lazily up to 0.45 nats in a near density at a half-life
/// of 20 rows (0.23 at 500) and 1.2e-2 in a posterior.
fn hold_the_exact_recursion(widths: &[usize], n: usize) {
    for &d in widths {
        for decay in [Decay::Halflife(20.0), Decay::Halflife(500.0)] {
            for eager in [true, false] {
                let t = track(d, decay, n, 3, eager);
                println!(
                    "{d} {decay:?} eager {eager}: off {:.0}, logf {:.1e}; from stabilized \
                     forgetting: logf {:.1e}, post {:.1e}, score {:.1e}",
                    max(&t.off),
                    max(&t.logf),
                    max(&t.gap_logf),
                    max(&t.post),
                    max(&t.score)
                );
                assert!(max(&t.off) < 4096.0, "{d} {decay:?} {eager}");
                assert!(max(&t.logf) < 2e-11, "{d} {decay:?} {eager}");
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

fn model(d: usize, cov: Covariance) -> Hmm {
    let c = HmmCfg {
        covariance: cov,
        ..wide(d, Decay::Halflife(20.0))
    };
    Hmm::with_stabilized(c, d as f64).unwrap()
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

/// Only `full` and `diag` take the prior, and only a model that learns.
#[test]
fn the_prior_is_for_a_learning_full_or_diag_model() {
    let c = |cov, learn| HmmCfg {
        covariance: cov,
        learn,
        means: Some(vec![-3.0, -3.0, 3.0, 3.0]),
        covs: Some(vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]),
        ..wide(2, Decay::Halflife(20.0))
    };
    for cov in [Covariance::Full, Covariance::Diagonal] {
        let m = Hmm::with_stabilized(c(cov, true), 2.0).unwrap();
        // Given states: `ŝ` from their variances at once.
        let st = m.stabilized().unwrap();
        assert_eq!(st.scale(), (&[1.0, 1.0][..], 2.0));
        assert!(
            Hmm::with_stabilized(c(cov, false), 2.0)
                .unwrap()
                .stabilized()
                .is_none()
        );
    }
    assert!(
        Hmm::with_stabilized(c(Covariance::Shared, true), 2.0)
            .unwrap()
            .stabilized()
            .is_none()
    );
    for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(Hmm::with_stabilized(c(Covariance::Full, true), bad).is_err());
    }
}
