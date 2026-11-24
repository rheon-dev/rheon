//! The stabilized prior of **given** states ([`Stabilized`], "Given
//! states"): each state forgets toward the covariance it was given, every
//! feature rescaled by how far the rows have moved from it. The reads
//! against the definition through a third-party eigensolver
//! (`crate::oracle`), the factor form against the exact recursion, the
//! contract a row keeps, and the stream that showed the diagonal prior
//! erasing regimes that differ in correlation alone.

use std::cell::Cell;

use super::stabilized::{EAGER, Shape, TAU};
use super::tests_factor::wide;
use super::*;
use crate::OnlineModel;

thread_local! {
    /// Set by a test: given states take the diagonal prior, as a build
    /// before the shape gave them.
    pub(super) static NO_SHAPE: Cell<bool> = const { Cell::new(false) };
}

/// `f` run as a build before the shape ran it ([`NO_SHAPE`]).
fn before_the_shape<T>(f: impl FnOnce() -> T) -> T {
    NO_SHAPE.with(|c| c.set(true));
    let out = f();
    NO_SHAPE.with(|c| c.set(false));
    out
}

fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

/// About a standard normal: twelve uniforms.
fn normal(state: &mut u64) -> f64 {
    (0..12).map(|_| lcg(state)).sum::<f64>() * 0.5
}

/// Two states' covariances, `d·d` row-major each: state 0 of variance 1
/// with neighbours at correlation 0.3, state 1 of variances 4, 2.25, 4, ...
/// with neighbours at correlation −0.4. Both positive definite at every
/// width.
fn two_shapes(d: usize) -> Vec<f64> {
    let mut covs = vec![0.0; 2 * d * d];
    for (s, rho) in [(0, 0.3), (1, -0.4)] {
        let sd = |i: usize| match (s, i % 2) {
            (0, _) => 1.0,
            (_, 0) => 2.0,
            _ => 1.5,
        };
        for i in 0..d {
            for j in 0..d {
                let c = match i.abs_diff(j) {
                    0 => 1.0,
                    1 => rho,
                    _ => 0.0,
                };
                covs[s * d * d + i * d + j] = c * sd(i) * sd(j);
            }
        }
    }
    covs
}

/// Two states given at `−1` and `1` in every feature under
/// [`two_shapes`], learning.
fn shaped(d: usize, covariance: Covariance, decay: Decay) -> HmmCfg {
    HmmCfg {
        covariance,
        means: Some([vec![-1.0; d], vec![1.0; d]].concat()),
        covs: Some(two_shapes(d)),
        precision_prior: 1e-3,
        ..wide(d, decay)
    }
}

/// Rows drawn from the two given states in runs of 30 (each feature its
/// state's `sd·(a·zᵢ + b·zᵢ₊₁)`, `a² + b² = 1` and `a·b` the neighbours'
/// correlation, so the rows have [`two_shapes`]' covariances), with a clock
/// step of 0.5 to 2, a zero-weight row every 13th and a gap of 60 every
/// 50th.
fn rows(d: usize, n: usize, seed: u64) -> Vec<(Vec<f64>, f64, f64)> {
    let mut s = seed;
    (0..n)
        .map(|i| {
            let state = (i / 30) % 2;
            let (mean, a, b) = if state == 0 {
                (-1.0, 0.9535, 0.3146)
            } else {
                (1.0, 0.8944, -0.4472)
            };
            let z: Vec<f64> = (0..=d).map(|_| normal(&mut s)).collect();
            let x = (0..d)
                .map(|j| {
                    let sd = match (state, j % 2) {
                        (0, _) => 1.0,
                        (_, 0) => 2.0,
                        _ => 1.5,
                    };
                    mean + sd * (a * z[j] + b * z[j + 1])
                })
                .collect();
            let dt = match i {
                0 => 0.0,
                _ if i % 50 == 25 => 60.0,
                _ => 0.5 * (1.0 + (2.0 * (lcg(&mut s) + 1.0)).floor().min(3.0)),
            };
            (x, dt, if i % 13 == 6 { 0.0 } else { 1.0 })
        })
        .collect()
}

fn forms() -> [(usize, Covariance); 3] {
    [
        (3, Covariance::Diagonal),
        (4, Covariance::Full),
        (13, Covariance::Full),
    ]
}

/// The score a row reads from the predicted state and each state's log
/// density, about the largest term, with the posterior it normalizes.
fn filtered(pred: &[f64], logf: &[f64]) -> (f64, Vec<f64>) {
    let terms: Vec<f64> = pred.iter().zip(logf).map(|(p, f)| p.ln() + f).collect();
    let top = terms.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = terms.iter().map(|v| (v - top).exp()).collect();
    let z: f64 = exps.iter().sum();
    (top + z.ln(), exps.iter().map(|e| e / z).collect())
}

/// State `s`'s covariance as given, a square.
fn given_of(cfg: &HmmCfg, s: usize) -> Vec<f64> {
    let d = cfg.n_features;
    let mut g = cfg.covs.as_ref().unwrap()[s * d * d..(s + 1) * d * d].to_vec();
    if matches!(cfg.covariance, Covariance::Diagonal) {
        for i in 0..d {
            for j in 0..d {
                if i != j {
                    g[i * d + j] = 0.0;
                }
            }
        }
    }
    g
}

/// Which matrices keep a shape: any that are not the same multiple of the
/// identity for every state. That multiple is the diagonal prior exactly,
/// and keeps none; under `diag` the entries off the diagonal are not read.
/// The same diagonal with variances that differ keeps one (review
/// 2026-11-23, HMM-3), which a build before did not.
#[test]
fn one_multiple_of_the_identity_for_every_state_keeps_no_shape() {
    let identity = [2.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 2.0];
    assert!(Shape::of(&identity, 2, 2, false).is_none());
    assert!(Shape::of(&identity, 2, 2, true).is_none());
    let off = [2.0, 0.5, 0.5, 2.0, 2.0, 0.0, 0.0, 2.0];
    assert!(Shape::of(&off, 2, 2, true).is_none());
    assert!(Shape::of(&off, 2, 2, false).is_some_and(|g| g.kept_before()));
    let same = [2.0, 0.0, 0.0, 3.0, 2.0, 0.0, 0.0, 3.0];
    for diagonal in [false, true] {
        let g = Shape::of(&same, 2, 2, diagonal).unwrap();
        assert_eq!((g.diag(0), g.diag(1)), (&[2.0, 3.0][..], &[2.0, 3.0][..]));
        assert!(!g.kept_before());
    }
    let variances = [2.0, 0.0, 0.0, 3.0, 2.0, 0.0, 0.0, 5.0];
    for diagonal in [false, true] {
        let g = Shape::of(&variances, 2, 2, diagonal).unwrap();
        assert_eq!((g.diag(0), g.diag(1)), (&[2.0, 3.0][..], &[2.0, 5.0][..]));
        assert!(g.kept_before());
    }
    let correlated = [2.0, 1.0, 1.0, 3.0, 2.0, 0.0, 0.0, 3.0];
    let g = Shape::of(&correlated, 2, 2, false).unwrap();
    assert_eq!(g.tri(0), &[2.0, 1.0, 3.0]);
    assert_eq!(g.tri(1), &[2.0, 0.0, 3.0]);
    // Under `diag` it is the same diagonal, `(2, 3)`, in both.
    assert!(Shape::of(&correlated, 2, 2, true).is_some_and(|g| !g.kept_before()));
    // In the model: none before the states are given, none for one
    // diagonal, one where they differ; and none in a filter that does not
    // learn, which takes no prior.
    for (d, cov) in forms() {
        let seeded = Hmm::new(HmmCfg {
            covariance: cov,
            ..wide(d, Decay::Halflife(50.0))
        })
        .unwrap();
        assert!(seeded.stabilized.as_ref().unwrap().shape().is_none());
        let plain = Hmm::new(HmmCfg {
            covariance: cov,
            ..super::tests_factor::given(d, 2.0)
        })
        .unwrap();
        assert!(plain.stabilized().unwrap().shape().is_none());
        let m = Hmm::new(shaped(d, cov, Decay::Halflife(50.0))).unwrap();
        let st = m.stabilized().unwrap();
        assert!(st.shape().is_some(), "{d} {cov:?}");
        // In units of the given variances: 1, at the states' weight.
        assert_eq!(st.scale(), (&vec![1.0; d][..], 2.0));
        let still = Hmm::new(HmmCfg {
            learn: false,
            ..shaped(d, cov, Decay::Halflife(50.0))
        })
        .unwrap();
        assert!(still.stabilized.is_none());
    }
}

/// On the first row each state reads the covariance it was given:
/// `(1·G + P₀·G)/(1 + P₀)`. The diagonal prior read `(G + P₀·diag(ψ))/(1 +
/// P₀)` there, two thirds of a diagonal matrix at two features. The score
/// against the Gaussians of the given matrices (`crate::oracle`'s
/// eigensolver), every form.
#[test]
fn the_first_row_reads_the_given_covariances() {
    for (d, cov) in forms() {
        let cfg = shaped(d, cov, Decay::Halflife(50.0));
        let mut m = Hmm::new(cfg.clone()).unwrap();
        let x: Vec<f64> = (0..d).map(|i| 0.3 * i as f64 - 0.7).collect();
        let step = m.step(&x, &[], 0.0, 1.0);
        let logf: Vec<f64> = (0..2)
            .map(|s| {
                let mean = if s == 0 { -1.0 } else { 1.0 };
                let delta: Vec<f64> = x.iter().map(|v| v - mean).collect();
                crate::oracle::gaussian_log_density(&given_of(&cfg, s), &delta)
            })
            .collect();
        let (score, _) = filtered(&[0.5, 0.5], &logf);
        let got = step.pred[5];
        assert!(
            (got - score).abs() <= 1e-12 * score.abs(),
            "{d} {cov:?}: {got} {score}"
        );
        // And the diagonal prior did not: the same row, a build before.
        let mut old = before_the_shape(|| Hmm::new(cfg.clone()).unwrap());
        assert!(old.stabilized().unwrap().shape().is_none());
        let was = old.step(&x, &[], 0.0, 1.0).pred[5];
        assert!((was - score).abs() > 1e-3, "{d} {cov:?}: {was} {score}");
    }
}

/// Every row's read is the definition, from the model's own moments: with
/// `W`, `M` and `μ` a state's weight after the row's decay, co-moments and
/// means, `Σ_s = (W·M + P₀·R·G_s·R)/(W + P₀)`, `R = diag(√ψ)`, each density
/// through `crate::oracle`'s eigensolver; and `ŝ` is its recursion, `eᵢ =
/// Σₛ pₛ·(xᵢ − μₛᵢ)²/G_s,ii` at ten times the half-life, written out here.
/// The forms that keep co-moments or variances (the factor form is held to
/// them below).
#[test]
fn a_row_reads_each_state_under_its_own_shape() {
    for (d, cov) in [(3, Covariance::Diagonal), (4, Covariance::Full)] {
        let decay = Decay::Halflife(40.0);
        let cfg = shaped(d, cov, decay);
        let p0 = d as f64;
        let mut m = Hmm::new(cfg.clone()).unwrap();
        let (mut s_hat, mut v) = (vec![1.0; d], 2.0);
        let mut worst = 0f64;
        for (i, (x, dt, w)) in rows(d, 700, 21).into_iter().enumerate() {
            let lam = decay.factor(dt);
            let mut aged = m.clone();
            aged.age_stabilized(lam, &mut RowWork::default(), w == 0.0);
            let st = aged.stabilized().unwrap();
            let psi = st.psi().to_vec();
            let state = |s: usize| -> (f64, Vec<f64>, Vec<f64>) {
                match cov {
                    Covariance::Diagonal => {
                        let c = &aged.diag_states[s];
                        let mut full = vec![0.0; d * d];
                        for j in 0..d {
                            full[j * d + j] = c.var(j);
                        }
                        (c.n_eff(), c.means().to_vec(), full)
                    }
                    _ => {
                        let c = &aged.states[s];
                        (c.n_eff(), c.means().to_vec(), c.comoments().to_full())
                    }
                }
            };
            let mut logf = [0.0; 2];
            let mut devs = Vec::new();
            for (s, lf) in logf.iter_mut().enumerate() {
                let (weight, means, comoments) = state(s);
                let g = given_of(&cfg, s);
                let mut sigma = vec![0.0; d * d];
                for a in 0..d {
                    for b in 0..d {
                        let prior = (psi[a] * psi[b]).sqrt() * g[a * d + b];
                        sigma[a * d + b] =
                            (weight * comoments[a * d + b] + p0 * prior) / (weight + p0);
                    }
                }
                let delta: Vec<f64> = x.iter().zip(&means).map(|(x, m)| x - m).collect();
                *lf = crate::oracle::gaussian_log_density(&sigma, &delta);
                devs.push(delta);
            }
            let step = m.step(&x, &[], dt, w);
            let (score, post) = filtered(&step.pred[2..4], &logf);
            worst = worst.max((step.pred[5] - score).abs() / (1.0 + score.abs()));
            // `ŝ`, by its recursion: the squared deviations from the means
            // before the row, in units of each state's given variance.
            let slow = Decay::Halflife(400.0).factor(dt);
            if w > 0.0 {
                let v_new = slow * v + w;
                for j in 0..d {
                    let e: f64 = (0..2)
                        .map(|s| post[s] * devs[s][j] * devs[s][j] / given_of(&cfg, s)[j * d + j])
                        .sum();
                    s_hat[j] += (w / v_new) * (e - s_hat[j]);
                }
                v = v_new;
            } else {
                v *= slow;
            }
            let (got, got_v) = m.stabilized().unwrap().scale();
            for j in 0..d {
                assert!(
                    (got[j] - s_hat[j]).abs() <= 1e-11 * s_hat[j],
                    "{d} {cov:?} row {i}: {} {}",
                    got[j],
                    s_hat[j]
                );
            }
            assert!((got_v - v).abs() <= 1e-11 * v, "{d} {cov:?} row {i}");
        }
        println!("{d} {cov:?}: {worst:e}");
        assert!(worst <= 1e-11, "{d} {cov:?}: {worst:e}");
    }
}

/// The factor form under a shape against the same recursion read through
/// co-moments ([`Hmm::new_comoments`], one factorization a state a row,
/// held to the definition above at 4 features), each learning from its own
/// reads. Its prior moved to the target before every read, it is that
/// recursion to rounding; as built, lazily, each read stands on a prior
/// within `τ` of its target in every coordinate, and the two part by no
/// more than the diagonal prior's bounds (a nat in a score, 0.05 in a
/// posterior). Measured 2026-11-20 (13 and 20 features, half-lives of 20
/// and 500 rows, 2,000 rows drawn from the given states, arm64): eagerly
/// 2e-15 in a posterior and 2e-13 nats in a score; lazily 9.8e-4 and 0.66
/// nats, where the diagonal prior's form on the same rows parted from its
/// own recursion by 4.5e-2 and 1.4 nats.
fn hold_the_exact_recursion(widths: &[usize], n: usize) {
    for &d in widths {
        for hl in [20.0, 500.0] {
            for eager in [true, false] {
                EAGER.with(|c| c.set(eager));
                let cfg = shaped(d, Covariance::Full, Decay::Halflife(hl));
                let mut m = Hmm::new(cfg.clone()).unwrap();
                let mut e = Hmm::new_comoments(cfg).unwrap();
                assert!(m.factored() && !e.factored());
                let (mut post, mut score, mut band) = (0f64, 0f64, 0f64);
                for (x, dt, w) in rows(d, n, 5) {
                    let mut aged = m.clone();
                    aged.age_stabilized(m.cfg.decay.factor(dt), &mut RowWork::default(), w == 0.0);
                    let st = aged.stabilized().unwrap();
                    for (h, t) in st.factored_prior().iter().zip(st.targets()) {
                        band = band.max((h / t - 1.0).abs());
                    }
                    let (a, b) = (m.step(&x, &[], dt, w), e.step(&x, &[], dt, w));
                    for (p, q) in a.pred[..4].iter().zip(&b.pred[..4]) {
                        post = post.max((p - q).abs());
                    }
                    score = score.max((a.pred[5] - b.pred[5]).abs());
                }
                EAGER.with(|c| c.set(false));
                println!("{d} hl {hl} eager {eager}: post {post:.1e}, score {score:.1e}");
                assert_eq!(m.solve_failures + e.solve_failures, 0);
                if eager {
                    assert!(band <= 1e-12, "{d} {hl}: {band:e}");
                    assert!(post < 1e-9 && score < 1e-8, "{d} {hl}: {post:e} {score:e}");
                } else {
                    assert!(band <= TAU, "{d} {hl}: {band:e} of the target off");
                    assert!(post < 0.05 && score < 1.0, "{d} {hl}: {post:e} {score:e}");
                }
            }
        }
    }
}

#[test]
fn the_factor_form_holds_the_exact_recursion_under_a_shape() {
    hold_the_exact_recursion(&[13], 600);
}

#[test]
#[ignore = "extended: the shaped factor form against the exact recursion at two widths"]
fn the_factor_form_holds_the_exact_recursion_under_a_shape_over_the_grid() {
    hold_the_exact_recursion(&[13, 20], 2000);
}

/// `predict` is the step's read without the step, to the bit, under a
/// shape in every form.
#[test]
fn predict_is_the_step_without_the_step_under_a_shape() {
    for (d, cov) in forms() {
        let mut m = Hmm::new(shaped(d, cov, Decay::Halflife(20.0))).unwrap();
        for (i, (x, dt, w)) in rows(d, 500, 9).into_iter().enumerate() {
            let p = m.predict(&x, dt);
            let s = m.step(&x, &[], dt, w);
            let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&p.pred), bits(&s.pred), "{d} {cov:?} row {i}");
        }
        assert!(m.stabilized().unwrap().shape().is_some());
        assert_eq!(m.factored(), d >= 13);
    }
}

/// Hard rule 9 under a shape: a zero-weight row inside a gap and the same
/// stream with its clock carried into the next row read alike, to
/// rounding, in every form.
#[test]
fn a_zero_weight_row_is_clock_alone_under_a_shape() {
    for (d, cov) in forms() {
        let new = || Hmm::new(shaped(d, cov, Decay::Halflife(20.0))).unwrap();
        let (mut with, mut without) = (new(), new());
        let mut carried = 0.0;
        let mut worst = 0f64;
        for (i, (x, dt, _)) in rows(d, 900, 9).into_iter().enumerate() {
            if i > 150 && matches!(i % 13, 5 | 6) {
                with.step(&x, &[], dt, 0.0);
                carried += dt;
                continue;
            }
            let a = with.step(&x, &[], dt, 1.0);
            let b = without.step(&x, &[], dt + carried, 1.0);
            carried = 0.0;
            for (p, q) in a.pred.iter().zip(&b.pred) {
                worst = worst.max((p - q).abs() / (1.0 + q.abs()));
            }
        }
        println!("{d} {cov:?}: {worst:e}");
        assert!(worst <= 1e-11, "{d} {cov:?}: {worst:e}");
    }
}

/// A saved state loads with its shape, which is derived from the
/// configuration and not written, and goes on to the bit; the bytes hold
/// what a state without a shape holds.
#[test]
fn a_saved_state_goes_on_under_its_shape() {
    for (d, cov) in forms() {
        let mut m = Hmm::new(shaped(d, cov, Decay::Halflife(20.0))).unwrap();
        let all = rows(d, 400, 9);
        for (x, dt, w) in &all[..250] {
            m.step(x, &[], *dt, *w);
        }
        let saved = m.state();
        let mut back = Hmm::restore(&saved).unwrap();
        assert_eq!(back, m);
        assert!(back.stabilized().unwrap().shape().is_some());
        let bytes = |m: &Hmm| rmp_serde::to_vec_named(&m.state()).unwrap();
        assert_eq!(bytes(&back), rmp_serde::to_vec_named(&saved).unwrap());
        for (x, dt, w) in &all[250..] {
            let (a, b) = (m.step(x, &[], *dt, *w), back.step(x, &[], *dt, *w));
            let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&a.pred), bits(&b.pred), "{d} {cov:?}");
        }
        assert_eq!(bytes(&back), bytes(&m));
    }
}

/// The faults the prior ended stay ended under a shape (review fable2's C2
/// and C3, `tests_stabilized`'s streams and edits, the states given under
/// [`two_shapes`] where those are seeded): a constant column and a feature
/// that sticks and moves again under `diag` and `full`, a burst of gaps and
/// a stuck feature on both sides of the factor form. No row among the last
/// 500 is NaN and no solve fails. The shape's variances are positive and
/// `ψ` is floored, so a feature with no variance keeps a positive one.
#[test]
fn the_faults_the_prior_ended_stay_ended_under_a_shape() {
    use super::tests_stabilized::{cfg, run};
    let given = |d: usize, hl: f64, cov: Covariance| HmmCfg {
        means: Some([vec![-1.0; d], vec![1.0; d]].concat()),
        covs: Some(two_shapes(d)),
        ..cfg(d, hl, cov)
    };
    for cov in [Covariance::Diagonal, Covariance::Full] {
        // C2a: a constant column.
        let constant = |_: usize, x: &mut Vec<f64>| {
            x[2] = 0.0;
            1.0
        };
        assert_eq!(
            run(given(3, 2.0, cov), true, 3000, constant),
            (0, 0),
            "{cov:?}"
        );
        // C2b: a feature sticks for 1,200 half-lives and moves again.
        let sticks = |i: usize, x: &mut Vec<f64>| {
            if (200..2600).contains(&i) {
                x[2] = 0.25;
            }
            1.0
        };
        assert_eq!(
            run(given(3, 2.0, cov), true, 4600, sticks),
            (0, 0),
            "{cov:?}"
        );
    }
    for d in [12, 13] {
        // C3a: six rows each 300 half-lives apart.
        let gaps = |i: usize, _: &mut Vec<f64>| if (100..106).contains(&i) { 6000.0 } else { 1.0 };
        assert_eq!(
            run(given(d, 20.0, Covariance::Full), true, 1000, gaps),
            (0, 0),
            "{d}"
        );
        // C3b: the last feature sticks from row 100 on.
        let stuck = move |i: usize, x: &mut Vec<f64>| {
            if i >= 100 {
                x[d - 1] = 0.25;
            }
            1.0
        };
        assert_eq!(
            run(given(d, 2.0, Covariance::Full), true, 6000, stuck),
            (0, 0),
            "{d}"
        );
    }
}

/// A state of schema 69 or earlier whose states were given (rule 5's
/// twenty-second exception) kept no prior: it loads under each state's own
/// shape, `ŝ` its states' variances over the given ones, pooled by weight,
/// and `V` their total weight; a factor form held under the ridge is taken
/// up again as `(W·C + P₀·R·G_s·R)/(W + P₀)`, `R = diag(√ψ)`, with `f` at
/// its target. It then saves bytes that load back the same and go on to the
/// bit, with a number wherever the old build had one.
#[test]
fn a_state_before_schema_70_loads_under_its_shape() {
    use super::tests_stabilized::before_the_prior;
    for (d, cov) in forms() {
        let cfg = shaped(d, cov, Decay::Halflife(40.0));
        let all = rows(d, 500, 33);
        let mut was = before_the_prior(|| Hmm::new(cfg.clone()).unwrap());
        for (x, dt, w) in &all[..300] {
            was.step(x, &[], *dt, *w);
        }
        assert!(was.stabilized.is_none());
        let mut state = was.state();
        state.schema_version = 69;
        let mut back = Hmm::restore(&state).unwrap();
        let st = back.stabilized.as_deref().expect("a prior");
        assert!(st.shape().is_some(), "{d} {cov:?}");
        // The moments the old build kept: each state's weight and
        // co-moments (a factor form's `σ·L Lᵀ` less the ridge it reached).
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
        assert_eq!(v.to_bits(), total.to_bits(), "{d} {cov:?}: V");
        for (i, &si) in scale.iter().enumerate() {
            let want = moments
                .iter()
                .enumerate()
                .map(|(s, (w, c))| w * c[i * d + i].max(0.0) / given_of(&cfg, s)[i * d + i])
                .sum::<f64>()
                / total;
            assert!(
                (si - want).abs() <= 1e-12 * want,
                "{d} {cov:?}: {si} {want}"
            );
        }
        assert_eq!(back.factored(), d >= 13);
        if back.factored() {
            let (psi, p0) = (st.psi().to_vec(), d as f64);
            assert_eq!(st.factored_prior(), &st.targets()[..]);
            for (s, (w, c)) in moments.iter().enumerate() {
                let f = &back.factor_states[s];
                assert_eq!((f.sigma(), f.n_eff()), (1.0, *w));
                let (got, g) = (f.matrix(), given_of(&cfg, s));
                let (mut top, mut apart) = (0f64, 0f64);
                for i in 0..d {
                    for j in 0..d {
                        let prior = p0 * (psi[i] * psi[j]).sqrt() * g[i * d + j];
                        let want = (w * c[i * d + j] + prior) / (w + p0);
                        apart = apart.max((got[i * d + j] - want).abs());
                        top = top.max(want.abs());
                    }
                }
                assert!(apart <= 1e-12 * top, "{d}: {apart:e} of {top:e}");
            }
        }
        let bytes = rmp_serde::to_vec_named(&back.state()).unwrap();
        let again: crate::State = rmp_serde::from_slice(&bytes).unwrap();
        let mut twin = Hmm::restore(&again).unwrap();
        assert_eq!(twin, back, "{d} {cov:?}");
        for (x, dt, w) in &all[300..] {
            let (a, t) = (back.step(x, &[], *dt, *w), twin.step(x, &[], *dt, *w));
            let b = was.step(x, &[], *dt, *w);
            let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&a.pred), bits(&t.pred), "{d} {cov:?}: the twin");
            assert_eq!(a.n_eff.to_bits(), b.n_eff.to_bits(), "{d} {cov:?}");
            for (p, q) in a.pred.iter().zip(&b.pred) {
                assert!(!p.is_nan() || q.is_nan(), "{d} {cov:?}: {p} for {q}");
            }
        }
    }
}

/// Rows of two features at correlation `rho`, zero mean, unit variance.
fn correlated(n: usize, rho: f64, s: &mut u64) -> Vec<Vec<f64>> {
    (0..n)
        .map(|_| {
            let (u, v) = (normal(s), normal(s));
            vec![u, rho * u + (1.0 - rho * rho).sqrt() * v]
        })
        .collect()
}

/// Two regimes that differ in correlation alone, 0.1 and 0.8, given at the
/// truth and learning on, with the per-row chance of a switch given too.
fn correlation_regimes(hl: f64, prior_rows: Option<f64>) -> HmmCfg {
    HmmCfg {
        means: Some(vec![0.0; 4]),
        covs: Some(vec![1.0, 0.1, 0.1, 1.0, 1.0, 0.8, 0.8, 1.0]),
        transition: Some(vec![0.99, 0.01, 0.02, 0.98]),
        precision_prior: 0.01,
        prior_rows,
        ..wide(2, Decay::Halflife(hl))
    }
}

/// The share of `xs`' rows the filter puts in state `want` after each.
fn share(m: &mut Hmm, xs: &[Vec<f64>], want: usize) -> f64 {
    let mut hits = 0;
    for x in xs {
        m.step(x, &[], 1.0, 1.0);
        hits += usize::from((m.p[1] > 0.5) == (want == 1));
    }
    hits as f64 / xs.len() as f64
}

/// The mean and the least of `runs`.
fn mean_min(runs: &[f64]) -> (f64, f64) {
    (
        runs.iter().sum::<f64>() / runs.len() as f64,
        runs.iter().cloned().fold(f64::INFINITY, f64::min),
    )
}

/// A model of `cfg`, as built or as a build before the shape built it.
fn built(cfg: HmmCfg, old: bool) -> Hmm {
    let m = if old {
        before_the_shape(|| Hmm::new(cfg).unwrap())
    } else {
        Hmm::new(cfg).unwrap()
    };
    let shaped = m.stabilized.as_ref().is_some_and(|st| st.shape().is_some());
    assert_eq!(shaped, !old && m.cfg.learn);
    m
}

/// The stream that showed the fault: regimes that differ in correlation
/// alone, the states given at the truth, and every row from the first
/// regime. Under the diagonal prior the given pair, at one row's weight,
/// read two thirds of a matrix with no correlation on the first rows; the
/// state that wins no rows kept reading it, took the rows that look like
/// neither, and learned the present regime: the filter then split one
/// regime's rows between two states that had become alike. Under its own
/// shape the absent state stays the regime it was given. Eight streams of
/// 4,000 rows at a half-life of 4,000; measured 2026-11-20 (arm64), the
/// mean share of rows in the true state and the worst stream's are printed
/// (`--nocapture`).
#[test]
fn regimes_that_differ_in_correlation_alone_stay_apart_under_their_shapes() {
    let streams: Vec<Vec<Vec<f64>>> = (0..8u64)
        .map(|i| correlated(4000, 0.1, &mut (77 + 1000 * i)))
        .collect();
    let run = |cfg: HmmCfg, old: bool| {
        let shares: Vec<f64> = streams
            .iter()
            .map(|xs| share(&mut built(cfg.clone(), old), xs, 0))
            .collect();
        mean_min(&shares)
    };
    let fixed = run(
        HmmCfg {
            learn: false,
            ..correlation_regimes(4000.0, None)
        },
        false,
    );
    let shape = run(correlation_regimes(4000.0, None), false);
    let diagonal = run(correlation_regimes(4000.0, None), true);
    let light = run(correlation_regimes(4000.0, Some(0.01)), true);
    println!("filter {fixed:.3?}, shape {shape:.3?}, diagonal {diagonal:.3?}, light {light:.3?}");
    assert!(fixed.0 > 0.96, "{fixed:?}");
    // Learning under the shape keeps the filter's share, in the mean and in
    // the worst stream; the diagonal prior lost a tenth of the rows in the
    // mean and a third in the worst stream.
    assert!(
        shape.0 > fixed.0 - 0.02 && shape.1 > fixed.1 - 0.03,
        "{shape:?} {fixed:?}"
    );
    assert!(
        diagonal.0 < shape.0 - 0.05 && diagonal.1 < shape.1 - 0.15,
        "{diagonal:?}"
    );
    // And no worse than the diagonal prior at a hundredth of a row, which
    // is the decaying ridge of a build before either.
    assert!(
        shape.0 >= light.0 && shape.1 >= light.1,
        "{shape:?} {light:?}"
    );
}

/// And through switches: the same regimes, 1,200 rows each in turn, at a
/// half-life of 500 rows, so the absent state has little data left when
/// its regime returns and reads its prior. Under its shape that is the
/// regime; under the diagonal prior it was a matrix with no correlation,
/// nearer the weakly correlated regime than its own. Eight streams.
#[test]
fn a_returning_correlation_regime_is_read_from_its_shape() {
    let run = |old: bool| {
        let shares: Vec<f64> = (0..8u64)
            .map(|i| {
                let mut s = 91 + 1000 * i;
                let mut m = built(correlation_regimes(500.0, None), old);
                let parts: Vec<f64> = [0.1, 0.8, 0.1, 0.8]
                    .iter()
                    .enumerate()
                    .map(|(j, &rho)| share(&mut m, &correlated(1200, rho, &mut s), j % 2))
                    .collect();
                parts.iter().sum::<f64>() / parts.len() as f64
            })
            .collect();
        mean_min(&shares)
    };
    let (shape, diagonal) = (run(false), run(true));
    println!("shape {shape:.3?}, diagonal {diagonal:.3?}");
    assert!(shape.0 > 0.95 && shape.1 > 0.93, "{shape:?}");
    assert!(diagonal.0 < shape.0 - 0.05, "{diagonal:?} {shape:?}");
}
