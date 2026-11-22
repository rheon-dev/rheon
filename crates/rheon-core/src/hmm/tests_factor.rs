//! The factor form of a wide `full` state (task 479; [`EwFactor`]): where it
//! starts, what it keeps, how it reads against the co-moments it replaces
//! and against the exact recursion. The co-moment form's tests are `tests`'.
//! Its form under the stabilized prior, the default, is held by
//! `tests_stabilized`; the tests here that read the rotations against the
//! decaying ridge build a model as a build before the prior did
//! ([`ridge`]), which is how a model's factor form learned until schema 70.

use std::cell::Cell;

use super::*;
use crate::OnlineModel;

thread_local! {
    /// Set by a test: `take_up_factors` takes up nothing, so a model keeps
    /// its co-moments at every width, as a build before schema 69 did.
    pub(super) static COMOMENTS_ONLY: Cell<bool> = const { Cell::new(false) };
}

/// `f` run as a build before the factor form ran it ([`COMOMENTS_ONLY`]).
pub(super) fn old<T>(f: impl FnOnce() -> T) -> T {
    COMOMENTS_ONLY.with(|c| c.set(true));
    let out = f();
    COMOMENTS_ONLY.with(|c| c.set(false));
    out
}

/// A model of `cfg` as a build before the stabilized prior built it: every
/// state under the decaying ridge alone.
fn ridge(cfg: HmmCfg) -> Hmm {
    super::tests_stabilized::before_the_prior(|| Hmm::new(cfg).unwrap())
}

fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

/// Rows of `d` features around one of three centres six apart, in runs of
/// 40 rows, with clock steps of 0.5 to 2 (`factor_bench`'s blobs).
pub(super) fn blobs(d: usize, n: usize, seed: u64) -> Vec<(Vec<f64>, f64)> {
    let mut s = seed;
    (0..n)
        .map(|i| {
            let label = (i / 40) % 3;
            let x = (0..d)
                .map(|j| lcg(&mut s) + if j % 3 == label { 6.0 } else { 0.0 })
                .collect();
            let dt = 0.5 * (1.0 + (2.0 * (lcg(&mut s) + 1.0)).floor().min(3.0));
            (x, dt)
        })
        .collect()
}

pub(super) fn wide(d: usize, decay: Decay) -> HmmCfg {
    HmmCfg {
        n_features: d,
        k: 2,
        decay,
        covariance: Covariance::Full,
        precision_prior: 1.0,
        min_weight: 0.0,
        learn: true,
        transition_prior: 1.0,
        transition: None,
        means: None,
        covs: None,
        warm_rows: 100,
        seed_rule: SeedRule::First,
        seed: 7,
        tvtp: None,
        prior_rows: None,
        warm_clock: None,
    }
}

/// Two states given at `-3` and `3` in every feature, each covariance
/// `var·I`.
pub(super) fn given(d: usize, var: f64) -> HmmCfg {
    let mut covs = vec![0.0; 2 * d * d];
    for s in 0..2 {
        for i in 0..d {
            covs[s * d * d + i * d + i] = var;
        }
    }
    HmmCfg {
        means: Some([vec![-3.0; d], vec![3.0; d]].concat()),
        covs: Some(covs),
        ..wide(d, Decay::Halflife(50.0))
    }
}

fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// The factor form starts at [`Hmm::FACTOR_FROM`] features under `full`,
/// once the states are seeded, and nowhere else: 12 features keep the
/// co-moments and build the cache's factors, 13 and 20 hold a factor a
/// state and no co-moments; `tied` and `diag` at 13 hold none, nor does a
/// model still in its warm-up. Given states are taken up at once, and a
/// filter that does not learn never moves them. (Before task 479 this was
/// the cache's test alone: `only_a_seeded_full_model_builds_factors`.)
#[test]
fn the_factor_form_starts_at_thirteen_features_once_seeded() {
    assert_eq!(Hmm::FACTOR_FROM, 13);
    for d in [12, 13, 20] {
        let mut m = Hmm::new(wide(d, Decay::Halflife(50.0))).unwrap();
        for (x, dt) in &blobs(d, 300, 3)[..99] {
            m.step(x, &[], *dt, 1.0);
        }
        assert!(!m.seeded && !m.factored() && m.states.len() == 2, "{d}");
        for (x, dt) in &blobs(d, 300, 3)[99..] {
            m.step(x, &[], *dt, 1.0);
        }
        assert!(m.seeded, "{d}");
        if d >= Hmm::FACTOR_FROM {
            assert!(m.factored() && m.states.is_empty(), "{d}");
            assert!(m.factor_states.iter().all(|f| f.order() == d));
            assert!(
                m.factors.peek(0).is_none(),
                "{d}: no cache beside the factors"
            );
            assert!(m.state_cov(0).is_none() && m.state_factor(1).is_some());
        } else {
            assert!(!m.factored() && m.states.len() == 2, "{d}");
            m.ensure_factors();
            assert!(m.factors.peek(0).is_some(), "{d}: the cache is built");
            assert!(m.state_cov(0).is_some() && m.state_factor(1).is_none());
        }
        assert_eq!(m.solve_failures, 0);
    }
    for covariance in [Covariance::Shared, Covariance::Diagonal] {
        let mut m = Hmm::new(HmmCfg {
            covariance,
            ..wide(13, Decay::Halflife(50.0))
        })
        .unwrap();
        for (x, dt) in &blobs(13, 200, 4) {
            m.step(x, &[], *dt, 1.0);
        }
        assert!(m.seeded && !m.factored(), "{covariance:?}");
    }
    let mut scorer = Hmm::new(HmmCfg {
        learn: false,
        ..given(13, 1.0)
    })
    .unwrap();
    assert!(scorer.factored());
    let held = scorer.factor_states.clone();
    for (x, dt) in &blobs(13, 50, 5) {
        scorer.step(x, &[], *dt, 1.0);
    }
    assert_eq!(scorer.factor_states, held, "a filter that does not learn");
    // The given states' factors are the cache's at 12: the same reads.
    let narrow = Hmm::new(HmmCfg {
        learn: false,
        ..given(12, 1.0)
    })
    .unwrap();
    assert!(!narrow.factored());
}

/// The factor a state takes up is the one the co-moments' read factorized,
/// to the bit, so every row up to the first one learned after the seeding
/// reads what a build before the factor form read; from there each read
/// is the moved factor's. The means and weights are the co-moments' own on
/// every row: one code steps them.
#[test]
fn the_taken_up_factor_reads_as_the_comoments_did() {
    for d in [13, 20] {
        for decay in [Decay::Halflife(60.0), Decay::Halflife(f64::INFINITY)] {
            let cfg = wide(d, decay);
            let mut new = ridge(cfg.clone());
            let mut was = ridge(cfg);
            let (mut same, mut differ, mut moved) = (0, 0, false);
            for (i, (x, dt)) in blobs(d, 400, 9).iter().enumerate() {
                let w = if i % 17 == 3 { 0.0 } else { 1.0 };
                let taken = new.factored();
                let a = new.step(x, &[], *dt, w);
                let b = old(|| was.step(x, &[], *dt, w));
                // Read before it is learned: the last row to read the
                // factor as taken up is the first learned after it.
                let first_moving = taken && w > 0.0 && !moved;
                if !moved {
                    moved = first_moving;
                    assert_eq!(bits(&a.pred), bits(&b.pred), "{d} {decay:?}, row {i}");
                    same += 1;
                } else if bits(&a.pred) != bits(&b.pred) {
                    differ += 1;
                    for (p, q) in a.pred.iter().zip(&b.pred) {
                        assert!(
                            (p - q).abs() <= 1e-9 * (1.0 + q.abs()),
                            "{d} {decay:?}, row {i}: {p} against {q}"
                        );
                    }
                }
                // Until a read has parted, each state learns at the same
                // weight in both forms, so its means, weight and ridge scale
                // are the co-moments' to the bit (from there the posteriors
                // that weigh the rows part in their last bits).
                for s in 0..2 {
                    if let (Some(f), Some(c), false) =
                        (new.state_factor(s), was.state_cov(s), differ > 0)
                    {
                        assert_eq!(f.means(), c.means(), "{d}, row {i}");
                        assert_eq!(f.n_eff().to_bits(), c.n_eff().to_bits());
                        assert_eq!(f.precision_scale().to_bits(), c.precision_scale().to_bits());
                    }
                }
            }
            assert!(new.factored() && !was.factored());
            assert!(
                same > 100 && differ > 200,
                "{d}: {same} same, {differ} moved"
            );
        }
    }
}

/// A state saved mid-stream in the factor form loads, continues to the
/// bit of the run that was never saved, and saves again byte for byte, in
/// both encodings: the factor is what is saved, and nothing is factorized
/// on the load. A state saved inside its warm-up takes its factors up at
/// its seeding after the load, as the run that was never saved does.
#[test]
fn a_factor_state_saved_mid_stream_continues_to_the_bit() {
    for decay in [Decay::Halflife(80.0), Decay::Halflife(f64::INFINITY)] {
        for at in [50, 450] {
            let d = 14;
            let rows = blobs(d, 900, 9);
            let mut whole = Hmm::new(wide(d, decay)).unwrap();
            for (x, dt) in &rows[..at] {
                whole.step(x, &[], *dt, 1.0);
            }
            assert_eq!(whole.factored(), at > 100);
            for named in [true, false] {
                let bytes = if named {
                    rmp_serde::to_vec_named(&whole.state()).unwrap()
                } else {
                    rmp_serde::to_vec(&whole.state()).unwrap()
                };
                let state: crate::State = rmp_serde::from_slice(&bytes).unwrap();
                let mut back = Hmm::restore(&state).unwrap();
                assert_eq!(back, whole);
                let mut run = whole.clone();
                for (i, (x, dt)) in rows[at..].iter().enumerate() {
                    let w = if i % 13 == 2 { 0.0 } else { 1.0 };
                    let (a, b) = (run.step(x, &[], *dt, w), back.step(x, &[], *dt, w));
                    assert_eq!(bits(&a.pred), bits(&b.pred), "{decay:?} {at}, row {i}");
                    assert_eq!(a.n_eff.to_bits(), b.n_eff.to_bits());
                }
                assert!(run.factored());
                assert_eq!(
                    rmp_serde::to_vec_named(&run.state()).unwrap(),
                    rmp_serde::to_vec_named(&back.state()).unwrap()
                );
            }
        }
    }
}

/// A row of no weight moves no factor; a gap that takes every state's
/// whole history makes the next row a first row for each state it gives a
/// weight to, which starts its `M` at the ridge, `√r` on the diagonal; a
/// state it gives none keeps its matrix, as the co-moments would.
#[test]
fn a_factor_stands_on_a_row_of_no_weight_and_restarts_after_a_total_decay() {
    let d = 13;
    let mut m = ridge(HmmCfg {
        warm_rows: 30,
        ..wide(d, Decay::Halflife(5.0))
    });
    let rows = blobs(d, 200, 4);
    for (x, dt) in &rows {
        m.step(x, &[], *dt, 1.0);
    }
    let held = m.factor_states.clone();
    m.step(&rows[0].0, &[], 3.0, 0.0);
    let lam = Decay::Halflife(5.0).factor(3.0);
    for (f, h) in m.factor_states.iter().zip(&held) {
        assert_eq!(
            (f.factor(), f.sigma(), f.means()),
            (h.factor(), h.sigma(), h.means())
        );
        assert_eq!(f.n_eff(), h.n_eff() * lam);
    }
    m.step(&rows[1].0, &[], 5e5, 1.0);
    let fresh = SpdFactor::scaled_identity(d, 1.0).unwrap();
    for (s, (f, h)) in m.factor_states.iter().zip(&held).enumerate() {
        if m.p[s] > 0.0 {
            assert_eq!((f.factor(), f.sigma()), (&fresh, 1.0), "{s}");
        } else {
            assert_eq!(f.factor(), h.factor(), "{s}");
        }
    }
    assert!(m.p.iter().any(|p| *p > 0.0));
    assert_eq!(m.solve_failures, 0);
}

/// `predict` is the step without the step in the factor form, to the bit,
/// on rows of weight 0 and 1, and leaves the state as it found it.
#[test]
fn predict_is_the_step_without_the_step_in_the_factor_form() {
    let d = 13;
    let mut m = Hmm::new(wide(d, Decay::Halflife(40.0))).unwrap();
    for (i, (x, dt)) in blobs(d, 500, 2).iter().enumerate() {
        let before = rmp_serde::to_vec(&m.state()).unwrap();
        let p = m.predict(x, *dt);
        assert_eq!(rmp_serde::to_vec(&m.state()).unwrap(), before);
        let w = if i % 9 == 4 { 0.0 } else { 1.0 };
        let s = m.step(x, &[], *dt, w);
        assert_eq!(bits(&p.pred), bits(&s.pred), "row {i}");
        assert_eq!(p.n_eff.to_bits(), s.n_eff.to_bits(), "row {i}");
    }
    assert!(m.factored());
}

/// A row whose feature is not usable is the same row at weight 0, byte for
/// byte, in the factor form, as in every other (`OnlineModel`).
#[test]
fn a_row_that_is_not_a_number_moves_no_factor() {
    let d = 13;
    let mut m = Hmm::new(wide(d, Decay::Halflife(10.0))).unwrap();
    for (x, dt) in blobs(d, 200, 6) {
        m.step(&x, &[], dt, 1.0);
    }
    assert!(m.factored());
    let mut finite = m.clone();
    let mut x = blobs(d, 1, 7).remove(0).0;
    finite.step(&x, &[], 1.0, 0.0);
    x[5] = f64::NAN;
    let out = m.step(&x, &[], 1.0, 1.0);
    assert!(out.pred.iter().all(|v| v.is_nan()));
    assert_eq!(
        rmp_serde::to_vec(&m.state()).unwrap(),
        rmp_serde::to_vec(&finite.state()).unwrap()
    );
}

/// Both sides of [`Hmm::FACTOR_FROM`] give the same bits on pools of 1, 3
/// and 8 threads: 12 features through the cache's factorizations, 13 and
/// 14 through the factors' rotations.
#[test]
fn both_sides_of_the_factor_width_on_a_threaded_pool() {
    for d in [12, 13, 14] {
        let rows = blobs(d, 400, 21);
        let run = || {
            let mut m = Hmm::new(wide(d, Decay::Halflife(50.0))).unwrap();
            let out: Vec<u64> = rows
                .iter()
                .flat_map(|(x, dt)| bits(&m.step(x, &[], *dt, 1.0).pred))
                .collect();
            (out, m.factored())
        };
        let one = crate::test_pool::pool(1).install(run);
        assert_eq!(one.1, d >= Hmm::FACTOR_FROM, "{d}");
        for n in [3, 8] {
            let got = crate::test_pool::pool(n).install(run);
            assert!(got == one, "{d} features, {n} threads");
        }
    }
}

/// What a learned row reads, per state, from the model before it: the row's
/// deviation from each state's means.
fn deltas(m: &Hmm, x: &[f64]) -> Vec<Vec<f64>> {
    (0..m.cfg.k)
        .map(|s| {
            x.iter()
                .enumerate()
                .map(|(i, &v)| match (m.state_factor(s), m.state_cov(s)) {
                    (Some(f), _) => f.deviation(i, v),
                    (None, Some(c)) => c.deviation(i, v),
                    (None, None) => unreachable!("a full state"),
                })
                .collect()
        })
        .collect()
}

/// One run's checks of three forms of each state's matrix against the
/// recursion they both round, carried in 512 bits
/// ([`crate::oracle::exact::Recursion`]): the co-moments with their ridge
/// (read through a fresh factor, as the co-moment form's read does), the
/// factor form's `σ·L Lᵀ`, and the exact one. Each check reads a row as
/// the model does, before learning it: per state, each form's matrix off
/// the exact one (units of rounding in its largest diagonal entry) and its
/// log density's error `|Δ(ln det M + δᵀM⁻¹δ)|/2`, over the states the row
/// could have come from (its exact `δᵀM⁻¹δ` under `3d`; a far state's
/// posterior is `e^(−q/2)`, nothing, whatever its density's last digits);
/// per row, the filtered posterior's and the score's error, each form's
/// densities put through the predicted state probabilities the model read.
#[derive(Default)]
struct Track {
    comoment_off: Vec<f64>,
    factor_off: Vec<f64>,
    comoment_logf: Vec<f64>,
    factor_logf: Vec<f64>,
    comoment_post: Vec<f64>,
    factor_post: Vec<f64>,
    comoment_score: Vec<f64>,
    factor_score: Vec<f64>,
}

/// `ln Σ p̃ₗ e^(fₗ)` and the posterior it normalizes, as `read_into` forms
/// them (about the largest term).
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

/// [`Track`] over `n` rows of [`blobs`], checking every `every`-th learned
/// row after the states are taken up.
fn track(d: usize, decay: Decay, n: usize, every: usize, seed: u64) -> Track {
    use crate::oracle::exact::Recursion;
    use crate::solve::{FactorWork, QuadWork};
    let cfg = wide(d, decay);
    let prior = cfg.precision_prior;
    let base = -0.5 * d as f64 * std::f64::consts::TAU.ln();
    let mut m = ridge(cfg);
    let (mut work, mut quad) = (FactorWork::default(), QuadWork::default());
    let mut out = Track::default();
    let mut shadow: Vec<EwCov> = Vec::new();
    let mut exact: Vec<Recursion> = Vec::new();
    let mut learned = 0usize;
    for (i, (x, dt)) in blobs(d, n, seed).iter().enumerate() {
        let w = if i % 23 == 11 { 0.0 } else { 1.0 };
        let before = m.clone();
        let lam = before.cfg.decay.factor(*dt);
        // The states' seeding happens inside the step: the co-moments it
        // fills are the shadow's start, and the exact recursion's.
        let seeding = !before.seeded;
        let step = m.step(x, &[], *dt, w);
        if !m.factored() {
            continue;
        }
        if seeding {
            let comoments = old(|| {
                let mut c = before.clone();
                c.step(x, &[], *dt, w);
                c
            });
            shadow = comoments.states.clone();
            exact = (0..2)
                .map(|s| Recursion::of(&comoments.state_matrix(s)))
                .collect();
            continue;
        }
        let ds = deltas(&before, x);
        if w > 0.0 && learned % every == every - 1 && step.pred[4].is_finite() {
            let pred = &step.pred[2..4];
            let (mut lc, mut lf, mut le) = ([0.0; 2], [0.0; 2], [0.0; 2]);
            for s in 0..2 {
                let delta = &ds[s];
                let (q, ld) = exact[s].quad_log_det(delta);
                le[s] = base - 0.5 * ld - 0.5 * q;
                let near = q < 3.0 * d as f64;
                let r = prior * shadow[s].precision_scale();
                let mut sq = Vec::new();
                shadow[s].comoments().ridged_lower_into(&mut sq, r);
                out.comoment_off.push(exact[s].off(&sq));
                let f = work
                    .factor_ridged(shadow[s].comoments().packed(), r, d)
                    .unwrap();
                let (ld_c, q_c) = (f.log_det(), f.quad_form(delta, &mut quad));
                work.give(f);
                lc[s] = base - 0.5 * ld_c - 0.5 * q_c;
                let held = &before.factor_states[s];
                out.factor_off
                    .push(exact[s].off_factor(&held.factor().lower_rows(), held.sigma()));
                let (ld_f, q_f) = held.read(delta, &mut quad);
                lf[s] = base - 0.5 * ld_f - 0.5 * q_f;
                if near {
                    out.comoment_logf.push(0.5 * (ld_c + q_c - ld - q).abs());
                    out.factor_logf.push(0.5 * (ld_f + q_f - ld - q).abs());
                }
            }
            // The model's own score is the factor form's, to the bit.
            let (score_f, post_f) = filtered(pred, &lf);
            assert_eq!(score_f.to_bits(), step.pred[5].to_bits(), "row {i}");
            let (score_c, post_c) = filtered(pred, &lc);
            let (score_e, post_e) = filtered(pred, &le);
            let dev = |p: &[f64]| {
                p.iter()
                    .zip(&post_e)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f64, f64::max)
            };
            out.comoment_post.push(dev(&post_c));
            out.factor_post.push(dev(&post_f));
            out.comoment_score.push((score_c - score_e).abs());
            out.factor_score.push((score_f - score_e).abs());
        }
        if w > 0.0 {
            learned += 1;
        }
        // Each state learns the row at the weight the model's filter gave
        // it, `w·pₗ` with `p` the posterior the step left.
        for s in 0..2 {
            let ws = w * m.p[s];
            let cov = &shadow[s];
            match cov.factor_step_of(lam, ws) {
                None => {}
                Some((0.0, _)) => exact[s].reset(prior),
                Some((a, ab)) => exact[s].step(a, ab, &ds[s]),
            }
            shadow[s].update(x, lam, ws);
            assert_eq!(
                shadow[s].means(),
                m.state_means(s),
                "the means are one code"
            );
        }
    }
    assert!(!out.factor_off.is_empty(), "{d}: no row checked");
    out
}

impl EwCov {
    /// What [`EwCov::update`]`(x, lam, w)` does to `M = C + s·r·I`, read
    /// before it runs: `None` where `M` stands, `(0, _)` where it starts
    /// again at the ridge, else `(a, a·b)` of `M' = a·M + a·b·δδᵀ`.
    fn factor_step_of(&self, lam: f64, w: f64) -> Option<(f64, f64)> {
        let w_sum = self.n_eff();
        if w < 0.0 || (w == 0.0 && lam * w_sum == 0.0) {
            return None;
        }
        let w_new = lam * w_sum + w;
        if w_new <= 0.0 {
            return None;
        }
        let (a, b) = (lam * w_sum / w_new, w / w_new);
        if a <= 0.0 {
            return Some((0.0, 0.0));
        }
        (a != 1.0 || a * b != 0.0).then_some((a, a * b))
    }
}

fn max(v: &[f64]) -> f64 {
    v.iter().cloned().fold(0f64, f64::max)
}

/// The factor form holds the exact recursion's matrix about as well as the
/// co-moments do, under decay and without it, and reads its densities,
/// posteriors and scores as close to the exact ones (the bounds from the
/// study of 2026-11-20 over 4 to 32 features, below; the whole grid is
/// [`the_factor_holds_the_exact_recursion_over_the_grid`]). The rotations
/// round three or four times an entry a row where the co-moments' step
/// rounds twice, and without decay nothing contracts what they leave, so
/// the factor's matrix is the further off; its densities are not, since it
/// is never factorized afresh.
#[test]
fn the_factor_holds_the_exact_recursion() {
    for decay in [Decay::Halflife(20.0), Decay::Halflife(f64::INFINITY)] {
        let t = track(13, decay, 500, 7, 21);
        // Measured 2026-11-20 over the extended grid (13, 20 and 32
        // features, 4,000 rows a case, two seeds, arm64): the factor's
        // matrix within 680 units of rounding, the co-moments' within 95;
        // the near log densities within 4.6e-12 and 3.1e-12, the posteriors
        // within 2.9e-13 and 4.2e-13, the scores within 8.5e-13 and
        // 8.7e-12, the factor's and the co-moments' in turn.
        assert!(
            max(&t.factor_off) < 2048.0,
            "{decay:?}: {:?}",
            max(&t.factor_off)
        );
        assert!(max(&t.comoment_off) < 512.0, "{decay:?}");
        assert!(
            max(&t.factor_logf) < 2e-11,
            "{decay:?}: {:e}",
            max(&t.factor_logf)
        );
        assert!(max(&t.comoment_logf) < 2e-11, "{decay:?}");
        assert!(
            max(&t.factor_post) < 1e-12,
            "{decay:?}: {:e}",
            max(&t.factor_post)
        );
        assert!(
            max(&t.factor_score) < 4e-12,
            "{decay:?}: {:e}",
            max(&t.factor_score)
        );
        assert!(max(&t.comoment_score) < 4e-11, "{decay:?}");
    }
}

/// [`the_factor_holds_the_exact_recursion`] over the study's grid: 13, 20
/// and 32 features, half-lives of 1000 rows, 20 rows and none, two seeds of
/// 4,000 rows, every learned row checked (every fourth from 20 features).
/// `--nocapture` prints each case's worst numbers.
#[test]
#[ignore = "extended: the factor form against the exact recursion over the study's grid"]
fn the_factor_holds_the_exact_recursion_over_the_grid() {
    for d in [13, 20, 32] {
        for decay in [
            Decay::Halflife(1000.0),
            Decay::Halflife(20.0),
            Decay::Halflife(f64::INFINITY),
        ] {
            let mut all = Track::default();
            for seed in [1, 2] {
                let t = track(d, decay, 4000, if d >= 20 { 4 } else { 1 }, seed);
                all.comoment_off.extend(t.comoment_off);
                all.factor_off.extend(t.factor_off);
                all.comoment_logf.extend(t.comoment_logf);
                all.factor_logf.extend(t.factor_logf);
                all.factor_post.extend(t.factor_post);
                all.comoment_post.extend(t.comoment_post);
                all.factor_score.extend(t.factor_score);
                all.comoment_score.extend(t.comoment_score);
            }
            println!(
                "{d} {decay:?}: off {:.0} ({:.0}), logf {:.1e} ({:.1e}), post {:.1e} ({:.1e}), \
                 score {:.1e} ({:.1e}): the factor's (the co-moments')",
                max(&all.factor_off),
                max(&all.comoment_off),
                max(&all.factor_logf),
                max(&all.comoment_logf),
                max(&all.factor_post),
                max(&all.comoment_post),
                max(&all.factor_score),
                max(&all.comoment_score),
            );
            assert!(max(&all.factor_off) < 2048.0, "{d} {decay:?}");
            assert!(max(&all.factor_logf) < 2e-11, "{d} {decay:?}");
            assert!(max(&all.factor_post) < 1e-12, "{d} {decay:?}");
            assert!(max(&all.factor_score) < 4e-12, "{d} {decay:?}");
        }
    }
}

/// The factor form's state is the smaller, bounded by the shape alone: a
/// state's triangle `d(d + 1)/2`, its means and their low parts, and four
/// numbers, where the co-moments write their whole `d × d` matrix and more;
/// beside the states, the factor form keeps its prior as factored, `d`
/// numbers a model, which the co-moments do not. `--nocapture` prints both
/// forms' bytes, in the state file's encoding.
#[test]
fn the_factor_form_is_the_smaller_state() {
    for d in [13, 20, 32] {
        let rows = blobs(d, 300, 5);
        let run = |m: &mut Hmm| {
            for (x, dt) in &rows {
                m.step(x, &[], *dt, 1.0);
            }
            rmp_serde::to_vec_named(&m.state()).unwrap().len()
        };
        let mut new = Hmm::new(wide(d, Decay::Halflife(50.0))).unwrap();
        let mut was = Hmm::new(wide(d, Decay::Halflife(50.0))).unwrap();
        let (factor, comoments) = (run(&mut new), old(|| run(&mut was)));
        println!("{d} features, 2 states: {factor} bytes, {comoments} in co-moments");
        assert!(new.factored() && !was.factored());
        // Each state: `d(d+1)/2 + 2d + 4` doubles of 9 bytes, where the
        // co-moments write `d² + 2d + 5` and their runs, at most; and the
        // model's factored prior, `d` more.
        let floats = |n: usize| 9 * n;
        assert!(
            factor + floats(2 * (d * d - d * (d + 1) / 2)) <= comoments + floats(d),
            "{d}: {factor} against {comoments}"
        );
    }
}
