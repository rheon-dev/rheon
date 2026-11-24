//! The exact row step under the inverse and power links (PLAN 581): its
//! root against the equation it solves, its step against the stationary
//! condition of the row's penalized deviance (solved by the oracle's LU),
//! the links it leaves alone to the bit, the start's cap it lifts, and the
//! null predictions it ends after a refit.

use super::tests::cfg_of;
use super::*;
use crate::oracle;

pub(super) fn lcg(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*s >> 11) as f64 + 0.5) / (1u64 << 53) as f64
}

/// The family, link and power of each canonical pair the step serves.
const CANONICAL: [(GlmFamily, GlmLink, f64); 3] = [
    (GlmFamily::Gamma, GlmLink::Inverse, 0.0),
    (GlmFamily::Tweedie, GlmLink::Power, 1.5),
    (GlmFamily::Tweedie, GlmLink::Power, 3.0),
];

fn canonical_cfg(family: GlmFamily, link: GlmLink, p: f64, nf: usize, refit: u32) -> GlmCfg {
    GlmCfg {
        link,
        var_power: if family == GlmFamily::Tweedie { p } else { 1.5 },
        ..cfg_of(family, nf, refit)
    }
}

/// A well-specified stream under the canonical link: `η = b0 + 0.15 x1 −
/// 0.1 x2` (`x` uniform on `[−1, 1]`, further features 0.05 each), `μ` at
/// that `η`, and a target of that mean: a Gamma of shape 2; under the
/// Tweedie at `p < 2` 0 a quarter of the time and an exponential of mean
/// `μ/0.75` otherwise. `b0` sets how near 0 the `η`s run.
fn canonical_stream(
    family: GlmFamily,
    p: f64,
    nf: usize,
    n: usize,
    b0: f64,
    seed: u64,
) -> Vec<(Vec<f64>, f64)> {
    let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ 581;
    (0..n)
        .map(|_| {
            let x: Vec<f64> = (0..nf).map(|_| 2.0 * lcg(&mut s) - 1.0).collect();
            let slopes = [0.15, -0.1];
            let eta = b0
                + x.iter()
                    .enumerate()
                    .map(|(i, v)| v * slopes.get(i).copied().unwrap_or(0.05))
                    .sum::<f64>();
            assert!(eta > 0.0);
            let (mu, y) = if family == GlmFamily::Gamma {
                let mu = 1.0 / eta;
                (mu, -0.5 * mu * (lcg(&mut s).ln() + lcg(&mut s).ln()))
            } else {
                let mu = eta.powf(1.0 / (1.0 - p));
                let y = if p < 2.0 && lcg(&mut s) < 0.25 {
                    0.0
                } else if p < 2.0 {
                    -mu / 0.75 * lcg(&mut s).ln()
                } else {
                    -0.5 * mu * (lcg(&mut s).ln() + lcg(&mut s).ln())
                };
                (mu, y)
            };
            assert!(mu > 0.0);
            (x, y)
        })
        .collect()
}

/// `f(t) = t − η + c (y − t^(−m))`, from its definition.
fn f_of(t: f64, eta: f64, c: f64, y: f64, m: f64) -> f64 {
    t - eta + c * (y - t.powf(-m))
}

/// **The root solves its equation, between `η` and the link at `y`,** over
/// `p` from 1.01 to 50 and `η`, `c` and `y` across thirty orders of
/// magnitude, `η` at, below and above 0: `f` changes sign across a relative
/// step of `1e-10 max(1, 1/m)` about it, wide enough to clear the rounding
/// of `f`'s terms there (`f'(u) u ≥ min(1, m) (u + c μ)`, and the terms are
/// at most `2 (u + c μ)` at the root), and the root's mean is `u^(−m)`.
/// The Gamma's quadratic in the form that cancels nothing is held at `a =
/// η − c y` of `−1e8` beside `c = 1`, where `(a + √(a² + 4c))/2` is 0. No
/// root runs to the pass cap: the most passes taken is 11 for `p` from 1.1
/// to 10, and 34 at the extremes, `p` of 1.01 (`m = 100`, where a loose
/// bound on `u` is a factor of `3^100` on `μ`) and 50 (2026-11-24). In a
/// steady stream after a refit it takes 2 or 3.
#[test]
fn the_proximal_root_solves_its_equation() {
    let ps = [1.01, 1.1, 1.5, 1.9, 2.0, 2.5, 3.0, 10.0, 50.0];
    let etas = [-1e3, -1.0, -1e-6, 0.0, 1e-9, 1e-3, 0.5, 1.0, 1e3, 1e8];
    let cs = [1e-12, 1e-6, 1e-2, 1.0, 1e2, 1e6];
    let ys = [0.0, 1e-9, 1e-3, 1.0, 1e3, 1e9];
    // The most passes at each power: the moderate powers' and the extremes'.
    let (mut most, mut most_extreme) = (0, 0);
    let (mut checked, mut outside) = (0, 0);
    for &p in &ps {
        let m = 1.0 / (p - 1.0);
        for &eta in &etas {
            for &c in &cs {
                for &y in &ys {
                    if y == 0.0 && p >= 2.0 {
                        continue;
                    }
                    let what = format!("p {p} η {eta} c {c} y {y}");
                    // The root by bisection on `f`'s sign over `ln u`,
                    // which shares no arithmetic with Newton's passes.
                    let (mut lo, mut hi) = (f64::MIN_POSITIVE.ln(), f64::MAX.ln());
                    for _ in 0..200 {
                        let mid = 0.5 * (lo + hi);
                        if f_of(mid.exp(), eta, c, y, m) < 0.0 {
                            lo = mid;
                        } else {
                            hi = mid;
                        }
                    }
                    let ub = (0.5 * (lo + hi)).exp();
                    let Some(r) = prox_root(eta, c, y, m) else {
                        // None only where the root or its mean is past
                        // the doubles: below the least normal double, or
                        // a mean that underflows.
                        let mb = ub.powf(-m);
                        assert!(
                            ub <= 2.0 * f64::MIN_POSITIVE || mb < f64::MIN_POSITIVE || mb.is_nan(),
                            "{what}: no root, bisection's {ub:e} of mean {mb:e}"
                        );
                        outside += 1;
                        continue;
                    };
                    let u = r.u;
                    assert!(u > 0.0 && u.is_finite(), "{what}: {u}");
                    assert!(r.passes < PROX_PASSES, "{what}: ran to the cap");
                    if (1.1..=10.0).contains(&p) {
                        most = most.max(r.passes);
                    } else {
                        most_extreme = most_extreme.max(r.passes);
                    }
                    let want_mu = u.powf(-m);
                    assert!(
                        (r.mu - want_mu).abs() <= 1e-12 * (1.0 + m) * want_mu,
                        "{what}: μ {} vs u^(−m) {want_mu}",
                        r.mu
                    );
                    // To the problem's own conditioning: `f`'s terms'
                    // rounding over its slope there.
                    let mu_b = ub.powf(-m);
                    let terms = ub + eta.abs() + c * y + c * mu_b;
                    let slope = 1.0 + c * m * mu_b / ub;
                    assert!(
                        (u - ub).abs() * slope <= 1e-12 * terms,
                        "{what}: u {u:e}, bisection's {ub:e}"
                    );
                    // Between `η` and `g(y) = y^(1−p)` (`∞` at `y = 0`).
                    let gy = if y > 0.0 {
                        y.powf(1.0 - p)
                    } else {
                        f64::INFINITY
                    };
                    let (lo, hi) = (eta.min(gy), eta.max(gy));
                    assert!(
                        u >= lo * (1.0 - 1e-9) - 1e-300 && u <= hi * (1.0 + 1e-9),
                        "{what}: u {u} outside [{lo}, {hi}]"
                    );
                    checked += 1;
                }
            }
        }
    }
    // 2,940 cases: 2,844 roots and 96 past the doubles.
    assert!(checked > 2800 && outside < 100, "{checked}, {outside}");
    assert!(
        most <= 12 && most_extreme <= 40,
        "the most passes: {most}, {most_extreme}"
    );
    // The Gamma's cancellation: `u = 2c/(√(a² + 4c) − a)`, `1e-8` here.
    let r = prox_root(-1e8, 1.0, 0.0, 1.0).unwrap();
    assert!((r.u - 1e-8).abs() <= 1e-8 * 1e-12, "{}", r.u);
    assert_eq!(r.mu, 1.0 / r.u);
    assert_eq!(r.passes, 0);
    // A start past the doubles has no root.
    assert!(prox_root(f64::NAN, 1.0, 1.0, 2.0).is_none());
    assert!(prox_root(1.0, 0.0, 1.0, 2.0).is_none());
    assert!(prox_root(1.0, f64::INFINITY, 1.0, 2.0).is_none());
    assert!(prox_root(1.0, 1.0, f64::INFINITY, 0.5).is_none());
}

/// The full matrix of `G`'s packed lower triangle, plus `r` on the
/// diagonal.
pub(super) fn full(g: &[f64], k: usize, r: f64) -> Vec<f64> {
    let mut a = vec![0.0; k * k];
    for i in 0..k {
        for m in 0..=i {
            a[i * k + m] = g[at(i, m)];
            a[m * k + i] = g[at(i, m)];
        }
        a[i * k + i] += r;
    }
    a
}

/// **The step is the minimum of the row's penalized deviance.** Its
/// stationary condition, `(G + R I)(β₁ − β₀) + w (y − μ₁)/(p − 1) x̃ = 0`
/// (`μ₁` the mean at the new `η`; `w (μ − y)/(p − 1)` is the half
/// deviance's slope in `η` under the canonical link), is solved for `β₁ −
/// β₀` by the oracle's LU, which shares no arithmetic with the model's
/// Cholesky, and the model's move is held to it at `1e-9` relative; and `G`
/// takes the row at the smaller of the family's own Fisher weights at the
/// old and the new `η` ([`Family::working`]), `1e-12` relative. Every row
/// after a refit of 40, under a half-life of 50 rows (so `G` decays first),
/// weights from 0.5 to 3, a target 1000 times its mean every 37th row and
/// one 1000 times below it every 41st, so the moves run both ways and far.
#[test]
fn the_exact_step_is_the_minimum_of_the_rows_penalized_deviance() {
    for (family, link, p) in CANONICAL {
        let c = GlmCfg {
            decay: Decay::Halflife(50.0),
            ..canonical_cfg(family, link, p, 2, 40)
        };
        let fam = c.family();
        let p1 = if family == GlmFamily::Gamma {
            1.0
        } else {
            p - 1.0
        };
        let lam = c.decay.factor(1.0);
        let mut m = Glm::new(c.clone()).unwrap();
        let rows = canonical_stream(family, p, 2, 400, 0.5, 3);
        let (mut checked, mut far) = (0, 0);
        for (i, (x, y)) in rows.iter().enumerate() {
            let y = if i % 37 == 0 {
                y * 1000.0
            } else if i % 41 == 0 {
                y / 1000.0
            } else {
                *y
            };
            let w = [0.5, 1.0, 3.0][i % 3];
            let done = m.refit[0].done;
            let (b0, g0) = (m.beta[0].clone(), m.gram[0].clone());
            m.step(x, &[Some(y)], 1.0, w);
            if !done {
                continue;
            }
            let xt: Vec<f64> = std::iter::once(1.0).chain(x.iter().copied()).collect();
            let g0: Vec<f64> = g0.iter().map(|v| v * lam).collect();
            let b1 = &m.beta[0];
            let eta0 = dot_aug(&b0, x, true);
            let eta1 = dot_aug(b1, x, true);
            let mu1 = fam.mean(eta1).expect("the step's η gives a mean").1;
            let s = w * (y - mu1) / p1;
            let rhs: Vec<f64> = xt.iter().map(|t| -s * t).collect();
            let want = oracle::solve(&full(&g0, 3, c.ridge), &rhs);
            for (j, wj) in want.iter().enumerate() {
                let got = b1[j] - b0[j];
                assert!(
                    (got - wj).abs() <= 1e-9 * (wj.abs() + b0[j].abs() + 1e-3),
                    "{family:?} p {p} row {i}: coefficient {j} moved {got:e}, the condition says {wj:e}"
                );
            }
            let at_eta0 = fam.working(eta0, y).map_or(f64::INFINITY, |wk| wk.w);
            let wmin = at_eta0.min(fam.working(eta1, y).unwrap().w);
            for i2 in 0..3 {
                for m2 in 0..=i2 {
                    let want = g0[at(i2, m2)] + w * wmin * xt[i2] * xt[m2];
                    let got = m.gram[0][at(i2, m2)];
                    assert!(
                        (got - want).abs() <= 1e-12 * want.abs().max(g0[at(i2, m2)].abs()),
                        "{family:?} p {p} row {i}: G {got:e} vs {want:e}"
                    );
                }
            }
            far += usize::from((eta1 - eta0).abs() > 0.1 * eta0.abs());
            checked += 1;
        }
        assert!(
            checked > 300 && far > 5,
            "{family:?} p {p}: {checked} rows, {far} far"
        );
        assert_eq!(m.solve_failures(), 0);
    }
}

/// **The log and logit links take the Fisher step, to the bit.** The exact
/// step is the inverse and power links' alone: under every other link each
/// learned row is the decayed `G`, then [`Glm::learn`] with the family's
/// weight and working response at the row's `η`, as before the exact step
/// existed. Each row is replayed on a clone through that path and the
/// model's `β`, `G` and prediction must match it bit for bit: with the
/// refit waiting, run and given up, without and with a cap, under decay.
/// (The digests of 1,440 such runs, every output's bits, were equal before
/// and after the change, 2026-11-24; the frozen states and goldens of
/// these links hold the same across commits.)
#[test]
fn the_log_and_logit_links_take_the_fisher_step_to_the_bit() {
    let pairs = [
        (GlmFamily::Poisson, GlmLink::Log),
        (GlmFamily::Gamma, GlmLink::Log),
        (GlmFamily::Binomial, GlmLink::Logit),
        (GlmFamily::Tweedie, GlmLink::Log),
    ];
    for (family, link) in pairs {
        for (refit, give_up, cap) in [
            (0, false, 1.0),
            (30, false, f64::INFINITY),
            (30, true, f64::INFINITY),
            (30, false, 0.2),
        ] {
            let c = GlmCfg {
                link,
                decay: Decay::Halflife(80.0),
                step_cap: cap,
                ..cfg_of(family, 2, refit)
            };
            let fam = c.family();
            let lam = c.decay.factor(1.0);
            let mut m = Glm::new(c).unwrap();
            if give_up {
                m.refit_give_up();
            }
            let mut s = 5u64 + refit as u64;
            for i in 0..300 {
                let x = [2.0 * lcg(&mut s) - 1.0, 2.0 * lcg(&mut s) - 1.0];
                let mu = (0.3 + 0.4 * x[0] - 0.2 * x[1]).exp();
                let y = match family {
                    GlmFamily::Binomial => f64::from(lcg(&mut s) < mu / (1.0 + mu)),
                    GlmFamily::Poisson => (2.0 * mu * lcg(&mut s)).floor(),
                    _ => -mu * lcg(&mut s).ln(),
                };
                let waiting = m.refit.first().is_some_and(|r| !r.done);
                let mut e = m.clone();
                let got = m.step(&x, &[Some(y)], 1.0, 1.0).pred[0];
                if waiting {
                    continue;
                }
                let eta = dot_aug(&e.beta[0], &x, true);
                let want = fam.mean(eta).map_or(f64::NAN, |(_, mu)| mu);
                e.gram[0].iter_mut().for_each(|v| *v *= lam);
                e.ensure_work();
                e.work.x[1..].copy_from_slice(&x);
                if let Some(wk) = fam.working(eta, y) {
                    let target = (link == GlmLink::Log).then_some(Target {
                        eta: wk.eta,
                        mu: wk.mu,
                        y,
                        w: 1.0,
                        p: variance_power(&fam),
                    });
                    e.learn(0, wk.w, wk.w * (wk.z - wk.eta), target);
                }
                let what = format!("{family:?} {link:?} refit {refit} {give_up} cap {cap} row {i}");
                assert_eq!(got.to_bits(), want.to_bits(), "{what}");
                assert!(
                    e.beta[0]
                        .iter()
                        .zip(&m.beta[0])
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "{what}: {:?} vs {:?}",
                    e.beta[0],
                    m.beta[0]
                );
                assert!(
                    e.gram[0]
                        .iter()
                        .zip(&m.gram[0])
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "{what}"
                );
            }
        }
    }
}

/// **The start's cap of 1 no longer holds these links' rows; `step_cap`
/// does.** A refit given up keeps a log link's rows at the start's cap for
/// good (`glm_3_a_given_up_refit_keeps_the_starts_cap`), since its one-pass
/// start can run away; the exact step cannot, so under the inverse and
/// power links a given-up target's first row moves its own `η` from 0 to
/// the root, beyond 1 here (targets of a hundredth of the design's, a ridge
/// of 0.01), uncapped. A spec's own `step_cap`
/// still holds each row's move to it: to 0.25 here, every row.
#[test]
fn the_starts_cap_is_lifted_and_step_cap_holds() {
    for (family, link, p) in CANONICAL {
        // Targets a hundredth of the design's, so the link at each is
        // tens to hundreds and the first rows' roots lie far past 1.
        let rows: Vec<_> = canonical_stream(family, p, 2, 300, 0.5, 8)
            .into_iter()
            .map(|(x, y)| (x, y * 0.01))
            .collect();
        for cap in [f64::INFINITY, 0.25] {
            let mut m = Glm::new(GlmCfg {
                step_cap: cap,
                ridge: 0.01,
                ..canonical_cfg(family, link, p, 2, 50)
            })
            .unwrap();
            m.refit_give_up();
            let mut worst = 0.0_f64;
            for (x, y) in &rows {
                let before = dot_aug(&m.beta[0], x, true);
                m.step(x, &[Some(*y)], 1.0, 1.0);
                worst = worst.max((dot_aug(&m.beta[0], x, true) - before).abs());
            }
            if cap.is_finite() {
                assert!(worst <= cap * (1.0 + 1e-12), "{family:?} {p}: {worst}");
                assert!(worst >= cap * (1.0 - 1e-12), "{family:?} {p}: {worst}");
            } else {
                assert!(
                    worst > 1.0,
                    "{family:?} {p}: the most a row moved was {worst}"
                );
            }
        }
    }
}

/// **A given-up refit under these links learns, from its first row.** Every
/// coefficient 0 gives no mean, so the Fisher step never started (51671e2e:
/// null forever); the exact step starts from `η = 0` and lands each row at
/// a positive `η`, so every prediction after the first is a positive mean,
/// and the fit comes to the refitted fit's neighbourhood: its intercept
/// within 2% of the refitted one's at 3,000 rows (measured 0.05% Gamma,
/// 0.4% and 0.05% Tweedie at `p` 1.5 and 3).
#[test]
fn a_given_up_refit_learns_from_its_first_row() {
    for (family, link, p) in CANONICAL {
        let rows = canonical_stream(family, p, 2, 3000, 0.5, 21);
        let mut given_up = Glm::new(canonical_cfg(family, link, p, 2, 200)).unwrap();
        given_up.refit_give_up();
        let mut refitted = Glm::new(canonical_cfg(family, link, p, 2, 200)).unwrap();
        let mut nulls = 0;
        for (i, (x, y)) in rows.iter().enumerate() {
            let pred = given_up.step(x, &[Some(*y)], 1.0, 1.0).pred[0];
            assert!(i > 0 || pred.is_nan(), "{family:?} {p}: {pred}");
            nulls += usize::from(pred.is_nan());
            refitted.step(x, &[Some(*y)], 1.0, 1.0);
        }
        let (a, b) = (given_up.beta[0][0], refitted.beta[0][0]);
        assert_eq!(nulls, 1, "{family:?} {p}");
        assert!(
            (a - b).abs() <= 0.02 * b.abs(),
            "{family:?} {p}: {a} vs {b}"
        );
    }
}

/// **After a refit, one far target leaves no run of null predictions.**
/// A target 1000 times its mean at row 1,500, after the refit has run: the
/// Fisher step scored it at the fit's `η` and pushed `β` past the domain's
/// edge, and the rows after it had no mean, each teaching the recursion
/// nothing, so none brought it back. On these 18 streams (Gamma; Tweedie
/// at `p` 1.5 and 3; six seeds) it left 197 to 998 null predictions of
/// the 2,800 after the refit in 11 of them (49414c1d). The exact step moves
/// that row's own `η` only to the root, above 0, and a row the step left
/// at or below 0 is itself learned and lands above 0: at most 22 here. The
/// other side, a 2% category whose `η` sits near 0 (means 2 to 10 times
/// the rest's) with no outlier: at most 18 before, 1 now. Held: every learned
/// row's own mean exists after its step, and the null predictions after
/// the refit are fewer than 2% of the rows.
#[test]
fn after_a_refit_one_far_target_leaves_no_run_of_nulls() {
    for (family, link, p) in CANONICAL {
        for seed in 0..6u64 {
            for rare in [false, true] {
                let mut rows = canonical_stream(family, p, 2, 3000, 0.5, 100 + seed);
                if rare {
                    // A 2% category on the second feature with a slope
                    // that takes its `η` to 0.05 above 0 at the least.
                    let mut s = seed ^ 0x5eed;
                    for (x, y) in rows.iter_mut() {
                        let on = lcg(&mut s) < 0.02;
                        x[1] = f64::from(u8::from(on));
                        let eta = 0.5 + 0.1 * x[0] - 0.35 * x[1];
                        let mu = if family == GlmFamily::Gamma {
                            1.0 / eta
                        } else {
                            eta.powf(1.0 / (1.0 - p))
                        };
                        *y = -mu * lcg(&mut s).ln();
                    }
                } else {
                    rows[1500].1 *= 1000.0;
                }
                let mut m = Glm::new(canonical_cfg(family, link, p, 2, 200)).unwrap();
                let mut nulls = 0;
                for (i, (x, y)) in rows.iter().enumerate() {
                    let pred = m.step(x, &[Some(*y)], 1.0, 1.0).pred[0];
                    if i < 200 {
                        continue;
                    }
                    nulls += usize::from(pred.is_nan());
                    assert!(
                        m.mean_of(0, x).is_finite(),
                        "{family:?} {p} seed {seed} rare {rare} row {i}: no mean after its step"
                    );
                }
                assert!(
                    nulls * 50 < rows.len() - 200,
                    "{family:?} {p} seed {seed} rare {rare}: {nulls} null predictions"
                );
            }
        }
    }
}

/// **A root that runs to the pass cap stands, and is counted.** At `p =
/// 1.001` (`m = 1000`), `η = 0`, `c = 1e-14` and `y = 0`, Newton's passes in
/// the mean climb from a bound `3^1000` too loose and run to the cap
/// (found by a search over `p` from 1.001 to 500 and thirty orders of `η`,
/// `c` and `y`; nothing short of `p = 1.001` came near it). Each pass stays
/// below the root, so the step toward it stands, and the model counts it
/// in `solve_failures`, as a refit run to its cap is. The same row through
/// a model: a given-up target, every coefficient 0, a ridge of `1e17` so
/// that `c = w q/(p − 1) = 1e-14`.
#[test]
fn a_root_at_the_pass_cap_stands_and_is_counted() {
    let r = prox_root(0.0, 1e-14, 0.0, 1000.0).unwrap();
    assert_eq!(r.passes, PROX_PASSES);
    let mut m = Glm::new(GlmCfg {
        ridge: 1e17,
        ..canonical_cfg(GlmFamily::Tweedie, GlmLink::Power, 1.001, 1, 10)
    })
    .unwrap();
    m.refit_give_up();
    m.step(&[0.0], &[Some(0.0)], 1.0, 1.0);
    assert_eq!(m.solve_failures(), 1);
    assert!(m.beta[0][0] > 0.0, "{:?}", m.beta[0]);
}

/// **A solve that fails is counted, and the row teaches nothing.** `G + R
/// I` with a pivot of 0 is not positive definite: the step has no `P x̃`,
/// so neither `β` nor `G` moves, and `solve_failures` counts it, as the
/// Fisher step's failed solves are.
#[test]
fn a_failed_solve_is_counted_and_teaches_nothing() {
    for (family, link, p) in CANONICAL {
        let mut m = Glm::new(canonical_cfg(family, link, p, 2, 30)).unwrap();
        let rows = canonical_stream(family, p, 2, 31, 0.5, 4);
        for (x, y) in &rows[..30] {
            m.step(x, &[Some(*y)], 1.0, 1.0);
        }
        assert!(m.refit[0].done && m.solve_failures() == 0);
        m.gram[0][0] = -m.cfg.ridge;
        let (b, g) = (m.beta[0].clone(), m.gram[0].clone());
        m.step(&rows[30].0, &[Some(rows[30].1)], 1.0, 1.0);
        assert_eq!(m.solve_failures(), 1, "{family:?} {p}");
        assert_eq!((&m.beta[0], &m.gram[0]), (&b, &g), "{family:?} {p}");
    }
}

pub(super) fn normal(s: &mut u64) -> f64 {
    let (u, v) = (lcg(s), lcg(s));
    (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
}

/// A Gamma draw of shape `a` and scale `b` (Marsaglia and Tsang; below 1,
/// boosted by `U^(1/a)`).
pub(super) fn gamma_draw(s: &mut u64, a: f64, b: f64) -> f64 {
    if a < 1.0 {
        return gamma_draw(s, a + 1.0, b) * lcg(s).powf(1.0 / a);
    }
    let d = a - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let z = normal(s);
        let v = (1.0 + c * z).powi(3);
        if v > 0.0 && lcg(s).ln() < 0.5 * z * z + d - d * v + d * v.ln() {
            return d * v * b;
        }
    }
}

/// A Poisson draw by inversion, for the small means here.
pub(super) fn poisson_draw(s: &mut u64, mu: f64) -> f64 {
    let l = (-mu).exp();
    let (mut k, mut p) = (0.0, 1.0);
    loop {
        p *= lcg(s);
        if p <= l {
            return k;
        }
        k += 1.0;
    }
}

/// A Tweedie amount of mean `mu`, power `p` and dispersion 2: the compound
/// Poisson-Gamma below 2, a Gamma of variance `2 μ^p` above.
pub(super) fn tweedie_draw(s: &mut u64, mu: f64, p: f64) -> f64 {
    let phi = 2.0;
    if p < 2.0 {
        let n = poisson_draw(s, mu.powf(2.0 - p) / (phi * (2.0 - p)));
        if n == 0.0 {
            return 0.0;
        }
        gamma_draw(
            s,
            n * (2.0 - p) / (p - 1.0),
            phi * (p - 1.0) * mu.powf(p - 1.0),
        )
    } else {
        gamma_draw(s, 1.0 / (phi * mu.powf(p - 2.0)), phi * mu.powf(p - 1.0))
    }
}

/// A Tweedie stream under the power link: an intercept and two standard
/// normal features, `η = 0.7 + 0.08 x1 − 0.06 x2` held above 0.2.
fn tweedie_stream(p: f64, n: usize, seed: u64) -> Vec<(Vec<f64>, f64)> {
    let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ 0x0005_817e;
    (0..n)
        .map(|_| {
            loop {
                let x = vec![normal(&mut s), normal(&mut s)];
                let eta = 0.7 + 0.08 * x[0] - 0.06 * x[1];
                if eta > 0.2 {
                    let y = tweedie_draw(&mut s, eta.powf(1.0 / (1.0 - p)), p);
                    break (x, y);
                }
            }
        })
        .collect()
}

/// The rows through the Fisher step from `start`, no refit and no cap:
/// the step before the exact one, at this link.
fn fisher_path(c: &GlmCfg, rows: &[(Vec<f64>, f64)], start: &[f64]) -> Vec<f64> {
    let mut m = Glm::new(c.clone()).unwrap();
    m.refit.clear();
    m.beta[0].copy_from_slice(start);
    let fam = c.family();
    m.ensure_work();
    for (x, y) in rows {
        let eta = dot_aug(&m.beta[0], x, true);
        m.work.x[1..].copy_from_slice(x);
        if let Some(wk) = fam.working(eta, *y) {
            m.learn(0, wk.w, wk.w * (wk.z - wk.eta), None);
        }
    }
    m.beta[0].clone()
}

/// The same rows through the model's own step from `start`.
fn exact_path(c: &GlmCfg, rows: &[(Vec<f64>, f64)], start: Option<&[f64]>) -> Vec<f64> {
    let mut m = Glm::new(c.clone()).unwrap();
    match start {
        Some(b) => {
            m.refit.clear();
            m.beta[0].copy_from_slice(b);
        }
        None => m.refit_give_up(),
    }
    for (x, y) in rows {
        m.step(x, &[Some(*y)], 1.0, 1.0);
    }
    m.beta[0].clone()
}

/// The batch fit of the rows and its standard errors: `β*` by Newton on
/// full systems solved by the oracle's LU, the errors from the Fisher
/// information `Σ W x̃x̃'` at `β*` and the Pearson dispersion.
fn batch_with_se(c: &GlmCfg, rows: &[(Vec<f64>, f64)]) -> (Vec<f64>, Vec<f64>) {
    let mut buf = Vec::new();
    for (x, y) in rows {
        buf.extend_from_slice(x);
        buf.push(*y);
        buf.push(1.0);
    }
    let fam = c.family();
    let flat = GlmCfg {
        ridge: 1e-300,
        ..c.clone()
    };
    let mut b = vec![
        fam.start_eta(rows.iter().map(|r| r.1).sum::<f64>() / rows.len() as f64),
        0.0,
        0.0,
    ];
    b = super::review_tests::batch(&flat, &buf, &b);
    let k = b.len();
    let mut info = vec![0.0; k * k];
    let mut pearson = 0.0;
    for (x, y) in rows {
        let xt: Vec<f64> = std::iter::once(1.0).chain(x.iter().copied()).collect();
        let wk = fam.working(dot_aug(&b, x, true), *y).unwrap();
        for i in 0..k {
            for j in 0..k {
                info[i * k + j] += wk.w * xt[i] * xt[j];
            }
        }
        pearson += (y - wk.mu).powi(2) / fam.variance(wk.mu);
    }
    let phi = pearson / (rows.len() - k) as f64;
    let inv = oracle::inverse(&info);
    let se = (0..k).map(|i| (phi * inv[i * k + i]).sqrt()).collect();
    (b, se)
}

/// Over `seeds` Tweedie streams of 1,000 rows under the power link at
/// power `p`: the share of seeds on which the Fisher step from the link
/// at the targets' mean, the exact step from it, and the exact step from
/// every coefficient 0 (a given-up refit) ran away -- a coefficient not
/// finite, or more than 20 of the batch fit's standard errors from it --
/// and the exact steps' median gap, in those errors.
fn runaway_shares(p: f64, seeds: std::ops::Range<u64>) -> ([usize; 3], [f64; 2]) {
    let c = canonical_cfg(GlmFamily::Tweedie, GlmLink::Power, p, 2, 10);
    let fam = c.family();
    let mut away = [0; 3];
    let (mut from_mean, mut from_zero) = (Vec::new(), Vec::new());
    for seed in seeds {
        let rows = tweedie_stream(p, 1000, seed);
        let (want, se) = batch_with_se(&c, &rows);
        let ybar = rows.iter().map(|r| r.1).sum::<f64>() / rows.len() as f64;
        let start = [fam.start_eta(ybar), 0.0, 0.0];
        let gap = |b: &[f64]| {
            if b.iter().all(|v| v.is_finite()) {
                b.iter()
                    .zip(&want)
                    .zip(&se)
                    .map(|((a, w), s)| (a - w).abs() / s)
                    .fold(0.0, f64::max)
            } else {
                f64::INFINITY
            }
        };
        let gaps = [
            gap(&fisher_path(&c, &rows, &start)),
            gap(&exact_path(&c, &rows, Some(&start))),
            gap(&exact_path(&c, &rows, None)),
        ];
        for (n, g) in away.iter_mut().zip(gaps) {
            *n += usize::from(g > 20.0);
        }
        from_mean.push(gaps[1]);
        from_zero.push(gaps[2]);
    }
    let median = |v: &mut Vec<f64>| {
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };
    (away, [median(&mut from_mean), median(&mut from_zero)])
}

/// **Without a refit or a cap, the exact step does not run away where the
/// Fisher step does** (PLAN 581, from Rheon C's oracle work: the Tweedie
/// recursion with neither ran away on about 1 in 5 of its seeded streams).
/// Under the power link a refit is required, so a stream learns with
/// neither only once its refit is given up; here the Fisher step is run
/// that way by hand (`fisher_path`, `Glm::learn` with the refit cleared)
/// from the link at the targets' mean, as the refit starts, beside the
/// exact step from the same start and from every coefficient 0. A seed
/// runs away where a coefficient is not finite or is more than 20 of the
/// batch fit's standard errors from it (the batch fit, unpenalized, by
/// Newton on systems the oracle's LU solves; its errors from the Fisher
/// information and the Pearson dispersion): far outside the batch fit's
/// own sampling error, which the exact step's median seed sits within.
/// Measured 2026-11-24 over 100 seeds a power (`p` 1.2, 1.5, 1.8, 2.5, 3;
/// dispersion 2; 1,000 rows): the Fisher step ran away on 36%, 45%, 31%,
/// 8% and 6% of them (126 of 500, a quarter; its median seed 3.5 to 15
/// errors off), the exact step on none, from either start. The exact
/// step's median seed was 0.12 to 0.75 errors off from the mean's link and
/// 0.14 to 1.04 from 0, its worst 0.44, 0.92, 2.1, 14.7 and 10.9. Above 2
/// its worst seeds stay several errors off without a refit, the first
/// rows' curvature frozen in `G`, which the refit (the default) replaces:
/// there the grid of `tests/test_second_opinion_glm_canonical.py` holds `p
/// = 3` within 0.37. The essentials run 2 seeds a power (the Fisher step
/// runs away on two of the ten); the extended form 100, where it runs away
/// at every power. Under the log link the exact step does not apply: there
/// the Fisher step with neither remedy is what it was, and the refit or
/// the step cap is the remedy (its default).
#[test]
fn without_a_refit_the_tweedie_power_link_does_not_run_away() {
    tweedie_runaways(0..2);
}

#[test]
#[ignore = "extended: 100 seeded streams a power, five powers, a batch fit each"]
fn without_a_refit_the_tweedie_power_link_does_not_run_away_over_100_seeds() {
    tweedie_runaways(0..100);
}

fn tweedie_runaways(seeds: std::ops::Range<u64>) {
    let mut fisher = 0;
    for p in [1.2, 1.5, 1.8, 2.5, 3.0] {
        let ([old, from_mean, from_zero], medians) = runaway_shares(p, seeds.clone());
        assert_eq!(
            (from_mean, from_zero),
            (0, 0),
            "p {p}: the exact step ran away"
        );
        assert!(
            medians.iter().all(|m| *m <= 1.5),
            "p {p}: the exact step's median seed {medians:?} errors off"
        );
        fisher += old;
        if seeds.end >= 100 {
            assert!(old > 0, "p {p}: the Fisher step ran away on no seed");
        }
    }
    assert!(fisher > 0, "the Fisher step ran away on no seed");
}
