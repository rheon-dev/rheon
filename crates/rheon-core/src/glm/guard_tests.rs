//! GLM-4, the target guard under the log link (the module docs, *The
//! target guard*): its root against the equation it solves; a guarded
//! step against the row's proximal problem, `q` and the step's direction
//! from the oracle's LU; the rows it does not fire on, the Fisher step's to
//! the bit; and the runaways it ends, with no refit and no cap and after
//! the default spec's refit.

use super::prox_tests::{full, gamma_draw, lcg, normal, poisson_draw, tweedie_draw};
use super::tests::cfg_of;
use super::*;
use crate::oracle;

/// The log-link families the guard serves, each with its variance power.
const LOG_FAMILIES: [(GlmFamily, f64); 7] = [
    (GlmFamily::Poisson, 1.0),
    (GlmFamily::Gamma, 2.0),
    (GlmFamily::Tweedie, 1.2),
    (GlmFamily::Tweedie, 1.5),
    (GlmFamily::Tweedie, 1.8),
    (GlmFamily::Tweedie, 2.5),
    (GlmFamily::Tweedie, 3.0),
];

fn log_cfg(family: GlmFamily, p: f64, refit: u32) -> GlmCfg {
    GlmCfg {
        var_power: if family == GlmFamily::Tweedie { p } else { 1.5 },
        ..cfg_of(family, 2, refit)
    }
}

/// `h(u) = u − η + wq e^((1−p)u)(e^u − y)`, the root's equation written
/// from its definition, `f'(u) = μ^(1−p)(μ − y)` at `μ = e^u`, its two
/// exponents summed (`y e^((1−p)u) (e^(u − ln y) − 1)`) so that the far
/// targets below stay in range.
fn h_of(u: f64, eta: f64, wq: f64, ln_y: f64, p: f64) -> f64 {
    u - eta + wq * (ln_y + (1.0 - p) * u).exp() * (u - ln_y).exp_m1()
}

/// The root of [`h_of`] in `[η, ln y]` by bisection to the doubles'
/// rounding: an algorithm the model does not run.
fn bisect(eta: f64, wq: f64, ln_y: f64, p: f64) -> f64 {
    let (mut lo, mut hi) = (eta, ln_y);
    loop {
        let mid = lo + 0.5 * (hi - lo);
        if !(mid > lo && mid < hi) {
            return mid;
        }
        if h_of(mid, eta, wq, ln_y, p) > 0.0 {
            hi = mid;
        } else {
            lo = mid;
        }
    }
}

/// The most passes [`guard_root`] took over the draws below (2026-11-24,
/// macOS arm64; 4.4 on average). Newton's passes on `h` alone, from `ln
/// y`, took up to 56 with `ln y − η` to 30, and with it to 700 reached the
/// cap of 64 short of the root: a pass on a wrong slope fails here.
const MOST_ROOT_PASSES: u32 = 21;

/// **The guard's root solves its equation, in `(η, ln y]`, in a few
/// passes.** Over every power, `η` from −30 (the clip) to 5, `ln y − η`
/// from 1e-12 to 700 (a target near the doubles' largest) and `wq` from
/// 1e-12 to 1e12: `h(u)` is within the rounding of its own terms (each
/// `exp` good to its argument's size in ulps) of 0, `u` matches a
/// bisection to the doubles' rounding within that rounding over `h`'s
/// slope, at least 1, and no root takes more than [`MOST_ROOT_PASSES`]
/// passes. Where `wq` is not a positive, finite number the penalty has
/// vanished and the root is `ln y`.
#[test]
fn the_guards_root_solves_its_equation() {
    let mut s = 4u64;
    let eps = f64::EPSILON;
    let (mut most, mut total, mut roots) = (0, 0, 0);
    for (_, p) in LOG_FAMILIES {
        for _ in 0..2000 {
            let eta = -30.0 + 35.0 * lcg(&mut s);
            let gap = if lcg(&mut s) < 0.1 {
                1e-12
            } else {
                700f64.powf(lcg(&mut s))
            };
            let wq = 10f64.powf(-12.0 + 24.0 * lcg(&mut s));
            let ln_y = (eta + gap).exp().ln();
            if !(ln_y > eta && ln_y.is_finite()) {
                continue;
            }
            let (u, passes) = guard_root(eta, wq, p, ln_y);
            (most, total, roots) = (most.max(passes), total + passes, roots + 1);
            let what = format!("p {p} eta {eta} ln y {ln_y} wq {wq}: u {u}");
            assert!(u >= eta && u <= ln_y, "{what}");
            let e = (ln_y + (1.0 - p) * u).exp();
            let terms = u.abs() + eta.abs() + wq * e * (2.0 + (u - ln_y).exp());
            let tol = 16.0 * eps * terms * (2.0 + ln_y.abs() + u.abs() * p.max(1.0));
            let h = h_of(u, eta, wq, ln_y, p);
            assert!(h.abs() <= tol, "{what}: h(u) {h}, tol {tol}");
            let b = bisect(eta, wq, ln_y, p);
            assert!(
                (u - b).abs() <= 2.0 * tol + 4.0 * eps * u.abs(),
                "{what} vs {b}"
            );
        }
        // The far ends: a target at the doubles' largest, a mean at the
        // clip, and no penalty left.
        for (eta, y) in [(-30.0, 1e300), (5.0, 1e300), (-30.0, 1e-12)] {
            let ln_y: f64 = f64::ln(y);
            for wq in [1e-300, 1.0, 1e300] {
                let (u, _) = guard_root(eta, wq, p, ln_y);
                assert!(
                    u >= eta && u <= ln_y,
                    "p {p} eta {eta} y {y} wq {wq}: u {u}"
                );
            }
        }
        for wq in [0.0, -1.0, f64::INFINITY, f64::NAN] {
            assert_eq!(guard_root(0.0, wq, p, 10f64.ln()), (10f64.ln(), 0), "{wq}");
        }
    }
    if std::env::var_os("GLM4_PRINT").is_some() {
        eprintln!(
            "passes: most {most}, mean {}",
            f64::from(total) / f64::from(roots)
        );
    }
    assert!(most <= MOST_ROOT_PASSES, "{most} passes");
}

/// C's design (Rheon C's oracle test): an intercept and two standard
/// normal features, `μ = e^(0.5 + 0.3 x1 − 0.2 x2)`; Poisson counts, Gamma
/// amounts of shape 1/2 (dispersion 2), Tweedie amounts at dispersion 2;
/// 300 rows. With `outlier`, row 250's target (at least 1) is 1000 times
/// its draw.
fn c_stream(family: GlmFamily, p: f64, seed: u64, outlier: bool) -> Vec<(Vec<f64>, f64)> {
    let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ 0x6c4d_0004 ^ (p * 10.0) as u64;
    let mut rows: Vec<(Vec<f64>, f64)> = (0..300)
        .map(|_| {
            let x = vec![normal(&mut s), normal(&mut s)];
            let mu = (0.5 + 0.3 * x[0] - 0.2 * x[1]).exp();
            let y = match family {
                GlmFamily::Poisson => poisson_draw(&mut s, mu),
                GlmFamily::Gamma => gamma_draw(&mut s, 0.5, 2.0 * mu),
                _ => tweedie_draw(&mut s, mu, p),
            };
            (x, y)
        })
        .collect();
    if outlier {
        rows[250].1 = rows[250].1.max(1.0) * 1000.0;
    }
    rows
}

/// The full `x̃'`.
fn aug(x: &[f64]) -> Vec<f64> {
    std::iter::once(1.0).chain(x.iter().copied()).collect()
}

/// **A guarded step lands the row's `η` on its proximal root, along the
/// Fisher step.** After 40 rows of each family's stream, no refit and no
/// cap, a row whose target is a million times its mean: the Fisher step
/// carries `η` past `ln y` (the model with the guard off), and the guarded
/// model's `η` lands on the root of `h(u) = u − η + w q f'(u)` in `(η, ln
/// y]`, found by bisection, with `q = x̃'(λG + R I)⁻¹x̃` before the row from
/// the oracle's LU rather than by Sherman-Morrison; its move is the Fisher
/// step's direction, `(G_t + R I)⁻¹x̃` by the LU, scaled to that root.
/// Under no decay and a half-life of 50 rows, weights 1 and 2.5. Bounds:
/// `q` from the Cholesky and Sherman-Morrison against the LU's, a relative
/// 1e-12 at these conditions, moves the root by less than that over the
/// root's slope in `q`; held at 1e-9 relative.
#[test]
fn a_guarded_step_lands_eta_on_the_rows_proximal_root() {
    for (family, p) in LOG_FAMILIES {
        for half in [f64::INFINITY, 50.0] {
            for w in [1.0, 2.5] {
                let c = GlmCfg {
                    decay: if half.is_finite() {
                        Decay::Halflife(half)
                    } else {
                        Decay::Lam(1.0)
                    },
                    ..log_cfg(family, p, 0)
                };
                let fam = c.family();
                let lam = c.decay.factor(1.0);
                let rows = c_stream(family, p, 3, false);
                let mut m = Glm::new(c.clone()).unwrap();
                for (x, y) in &rows[..40] {
                    m.step(x, &[Some(*y)], 1.0, 1.0);
                }
                let x = rows[40].0.clone();
                let xt = aug(&x);
                let eta = dot_aug(&m.beta[0], &x, true);
                let (_, mu) = fam.mean(eta).unwrap();
                let y = 1e6 * mu;
                let what = format!("{family:?} {p} half {half} w {w}");
                let g: Vec<f64> = m.gram[0].iter().map(|v| v * lam).collect();
                let q = oracle::quad_form(&full(&g, 3, c.ridge), &xt);
                let u = bisect(eta, w * q, y.ln(), p);
                let wk = fam.working(eta, y).unwrap();
                let mut gt = g.clone();
                for i in 0..3 {
                    for j in 0..=i {
                        gt[at(i, j)] += w * wk.w * xt[i] * xt[j];
                    }
                }
                let d = oracle::solve(&full(&gt, 3, c.ridge), &xt);
                let qp: f64 = xt.iter().zip(&d).map(|(a, b)| a * b).sum();
                let before = m.beta[0].clone();
                let mut off = m.clone();
                off.work.unguarded = true;
                off.step(&x, &[Some(y)], 1.0, w);
                assert!(
                    dot_aug(&off.beta[0], &x, true) > y.ln(),
                    "{what}: the Fisher step stops short of ln y"
                );
                let fired = m.guarded_rows();
                m.step(&x, &[Some(y)], 1.0, w);
                assert_eq!(m.guarded_rows(), fired + 1, "{what}");
                let got = dot_aug(&m.beta[0], &x, true);
                assert!(u > eta && u <= y.ln(), "{what}");
                assert!(
                    (got - u).abs() <= 1e-9 * (1.0 + u.abs()),
                    "{what}: eta {got} against the root {u}"
                );
                for i in 0..3 {
                    let want = before[i] + (u - eta) / qp * d[i];
                    assert!(
                        (m.beta[0][i] - want).abs() <= 1e-9 * (1.0 + want.abs()),
                        "{what}: coef {i} {} against {want}",
                        m.beta[0][i]
                    );
                }
                // `G` takes the row as the Fisher step does, at `η`.
                assert!(
                    m.gram[0]
                        .iter()
                        .zip(&off.gram[0])
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "{what}"
                );
            }
        }
    }
}

/// **Every row the guard does not fire on is the Fisher step's, to the
/// bit, and it fires only where it should.** The guarded model and the
/// same with the guard off run side by side over each log-link family's
/// stream, calm and with a target 1000 times its draw at row 250; no
/// refit, a refit of 40 rows run and given up; no cap and a cap of 1; no
/// decay and a half-life of 50 rows; weights 1, or 0 to 3. Until a row
/// fires, each row's prediction, coefficients and `G` are the same bits;
/// a row fires only on an upward step (`y > μ`) and never while the
/// start's cap holds the target (the refit waiting or given up); and both
/// sides are reached: streams on which it never fires (whole, to the
/// bit), streams on which it does, and upward rows on which it does not.
/// The logit link never fires.
#[test]
fn the_rows_the_guard_leaves_are_the_fisher_steps_to_the_bit() {
    let (mut calm, mut fired, mut upward_left) = (0, 0, 0);
    for (family, p) in LOG_FAMILIES {
        for (refit, give_up) in [(0, false), (40, false), (40, true)] {
            for cap in [f64::INFINITY, 1.0] {
                for half in [f64::INFINITY, 50.0] {
                    for (seed, outlier, weighted) in
                        [(0, false, false), (1, true, false), (2, true, true)]
                    {
                        let c = GlmCfg {
                            decay: if half.is_finite() {
                                Decay::Halflife(half)
                            } else {
                                Decay::Lam(1.0)
                            },
                            step_cap: cap,
                            ..log_cfg(family, p, refit)
                        };
                        let fam = c.family();
                        let mut m = Glm::new(c).unwrap();
                        if give_up {
                            m.refit_give_up();
                        }
                        let mut e = m.clone();
                        e.work.unguarded = true;
                        let mut s = seed + 77;
                        let mut parted = false;
                        for (i, (x, y)) in c_stream(family, p, seed, outlier).iter().enumerate() {
                            let w = if weighted {
                                [0.0, 0.5, 1.0, 3.0][(4.0 * lcg(&mut s)) as usize]
                            } else {
                                1.0
                            };
                            let held = m.refit.first().is_some_and(|r| !r.done || r.over_budget);
                            let mu = fam.mean(dot_aug(&m.beta[0], x, true)).unwrap().1;
                            let before = m.guarded_rows();
                            let a = m.step(x, &[Some(*y)], 1.0, w).pred[0];
                            let b = e.step(x, &[Some(*y)], 1.0, w).pred[0];
                            if parted {
                                continue;
                            }
                            let what = format!(
                                "{family:?} {p} refit {refit} {give_up} cap {cap} half {half} \
                                 seed {seed} row {i}"
                            );
                            assert_eq!(a.to_bits(), b.to_bits(), "{what}");
                            if m.guarded_rows() > before {
                                assert!(!held, "{what}: fired under the start's cap");
                                assert!(*y > mu && w > 0.0, "{what}: fired downward");
                                parted = true;
                                continue;
                            }
                            if *y > mu && w > 0.0 && !held {
                                upward_left += 1;
                            }
                            let same = |u: &[f64], v: &[f64]| {
                                u.iter().zip(v).all(|(a, b)| a.to_bits() == b.to_bits())
                            };
                            assert!(same(&m.beta[0], &e.beta[0]), "{what}");
                            assert!(same(&m.gram[0], &e.gram[0]), "{what}");
                        }
                        if parted {
                            fired += 1;
                        } else {
                            calm += 1;
                        }
                    }
                }
            }
        }
    }
    assert!(
        calm > 0 && fired > 0 && upward_left > 0,
        "{calm} {fired} {upward_left}"
    );
    // The logit link: a target's score is `y − μ`, at most 1 either way.
    let c = cfg_of(GlmFamily::Binomial, 2, 0);
    let mut m = Glm::new(c).unwrap();
    let mut s = 9u64;
    for _ in 0..300 {
        let x = [normal(&mut s), normal(&mut s)];
        let mu = 1.0 / (1.0 + (-(-6.0 + 2.0 * x[0])).exp());
        m.step(&x, &[Some(f64::from(lcg(&mut s) < mu))], 1.0, 1.0);
    }
    assert_eq!(m.guarded_rows(), 0);
}

/// Over `seeds` of each log-link family's stream ([`c_stream`]): the
/// seeds on which the Fisher step with the guard off passes `|coef|` 10
/// (or leaves the doubles) at any row, the same with it on, and the rows
/// it fired on. `default` runs the default spec (its refit of
/// [`default_glm_refit_rows`] rows, no cap after it) on the streams with
/// the outlier at row 250; otherwise no refit and no cap on the calm ones.
fn runaways(seeds: std::ops::Range<u64>, default: bool) -> Vec<(usize, usize, u64)> {
    LOG_FAMILIES
        .iter()
        .map(|&(family, p)| {
            let refit = if default {
                default_glm_refit_rows(2, 1)
            } else {
                0
            };
            let c = GlmCfg {
                step_cap: default_glm_step_cap(refit),
                ..log_cfg(family, p, refit)
            };
            let c = if default {
                c
            } else {
                GlmCfg {
                    step_cap: f64::INFINITY,
                    ..c
                }
            };
            let (mut off, mut on, mut fires) = (0, 0, 0);
            for seed in seeds.clone() {
                let rows = c_stream(family, p, seed, default);
                for guarded in [false, true] {
                    let mut m = Glm::new(c.clone()).unwrap();
                    m.work.unguarded = !guarded;
                    let mut away = false;
                    for (x, y) in &rows {
                        m.step(x, &[Some(*y)], 1.0, 1.0);
                        away |= !m.beta[0].iter().all(|b| b.abs() < 10.0);
                    }
                    if guarded {
                        on += usize::from(away);
                        fires += m.guarded_rows();
                    } else {
                        off += usize::from(away);
                    }
                }
            }
            (off, on, fires)
        })
        .collect()
}

/// The seeds, of those run, on which the Fisher step with the guard off
/// ran away, per family in [`LOG_FAMILIES`]' order (Poisson, Gamma,
/// Tweedie 1.2, 1.5, 1.8, 2.5, 3), measured 2026-11-24: with no refit and
/// no cap, and under the default spec with the outlier at row 250.
const ESSENTIAL_SEEDS: std::ops::Range<u64> = 0..8;
const MEASURED_NO_REFIT_8: [usize; 7] = [0, 1, 0, 1, 0, 3, 3];
const MEASURED_DEFAULT_8: [usize; 7] = [3, 3, 2, 2, 3, 6, 3];
const MEASURED_NO_REFIT_100: [usize; 7] = [0, 8, 0, 5, 7, 40, 57];
const MEASURED_DEFAULT_100: [usize; 7] = [23, 35, 26, 29, 27, 43, 44];

fn hold_runaways(seeds: std::ops::Range<u64>, default: bool, measured: [usize; 7]) {
    let got = runaways(seeds.clone(), default);
    if std::env::var_os("GLM4_PRINT").is_some() {
        eprintln!("{seeds:?} default {default}: {got:?}");
    }
    let off: Vec<usize> = got.iter().map(|g| g.0).collect();
    let on: Vec<usize> = got.iter().map(|g| g.1).collect();
    assert_eq!(off, measured, "the unguarded runaways moved");
    assert!(off.iter().sum::<usize>() > 0, "nothing to stop");
    assert_eq!(on, [0; 7], "the guarded step ran away");
    assert!(
        got.iter().all(|g| g.2 > 0),
        "a family the guard never fired on"
    );
}

/// **The guard ends the log link's runaways**, which the Fisher step's
/// upward steps cause (`y μ^(1−p)` grows without bound as `μ` falls):
/// with no refit and no cap, and under the default spec after its refit,
/// where the cap is off, on a stream with one target 1000 times its draw.
/// Over C's design ([`c_stream`]), a seed runs away where a coefficient
/// passes 10 at any row; the counts with the guard off are asserted as
/// measured, so the test cannot pass on streams with nothing to stop.
/// The essentials run 8 seeds a family; the extended form 100.
#[test]
fn the_guard_ends_the_log_links_runaways() {
    hold_runaways(ESSENTIAL_SEEDS, false, MEASURED_NO_REFIT_8);
    hold_runaways(ESSENTIAL_SEEDS, true, MEASURED_DEFAULT_8);
}

#[test]
#[ignore = "extended: 100 seeded streams a family, seven families, two specs"]
fn the_guard_ends_the_log_links_runaways_over_100_seeds() {
    hold_runaways(0..100, false, MEASURED_NO_REFIT_100);
    hold_runaways(0..100, true, MEASURED_DEFAULT_100);
}
