use super::*;
use crate::oracle;

fn lcg(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*s >> 11) as f64 + 0.5) / (1u64 << 53) as f64
}

fn normal(s: &mut u64) -> f64 {
    let (u, v) = (lcg(s), lcg(s));
    (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
}

/// A Poisson draw by inversion (Knuth), for the small means here.
fn poisson(s: &mut u64, mu: f64) -> f64 {
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

/// The study's design: an intercept and `k − 1` standard normal features,
/// `η` of spread about 0.5 around 0.5; Poisson counts, or Gamma amounts of
/// shape 2 (a sum of two exponentials).
fn stream(family: GlmFamily, k: usize, n: usize, seed: u64) -> Vec<(Vec<f64>, f64)> {
    let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ 500;
    let beta: Vec<f64> = (0..k - 1)
        .map(|_| normal(&mut s) * 0.5 / ((k - 1) as f64).sqrt())
        .collect();
    (0..n)
        .map(|_| {
            let x: Vec<f64> = (0..k - 1).map(|_| normal(&mut s)).collect();
            let mu = (0.5 + x.iter().zip(&beta).map(|(a, b)| a * b).sum::<f64>()).exp();
            let y = match family {
                GlmFamily::Poisson => poisson(&mut s, mu),
                GlmFamily::Gamma => -0.5 * mu * (lcg(&mut s).ln() + lcg(&mut s).ln()),
                _ => unreachable!(),
            };
            (x, y)
        })
        .collect()
}

fn cfg(family: GlmFamily, nf: usize, refit_rows: u32) -> GlmCfg {
    GlmCfg {
        n_features: nf,
        n_targets: 1,
        fit_intercept: true,
        decay: Decay::Lam(1.0),
        family,
        link: family.default_link(),
        var_power: 1.5,
        ridge: 1.0,
        min_weight: 0.0,
        refit_rows,
        step_cap: f64::INFINITY,
    }
}

fn run(m: &mut Glm, rows: &[(Vec<f64>, f64)], w: impl Fn(usize) -> f64) -> Vec<f64> {
    rows.iter()
        .enumerate()
        .map(|(i, (x, y))| m.step(x, &[Some(*y)], 1.0, w(i)).pred[0])
        .collect()
}

/// The batch target, `argmin Σ v d(y, μ)/2 + R/2 |b|²`, by Newton with step
/// halving on full square systems solved by the oracle's LU (no arithmetic
/// of the model's Cholesky), from 0, to a step of 1e-14.
fn batch(c: &GlmCfg, rows: &[(Vec<f64>, f64)], v: &[f64]) -> Vec<f64> {
    let fam = c.family();
    let k = c.k_total();
    let obj = |b: &[f64]| -> f64 {
        let mut f = 0.5 * c.ridge * b.iter().map(|t| t * t).sum::<f64>();
        for ((x, y), vi) in rows.iter().zip(v) {
            let (_, mu) = fam.mean(dot_aug(b, x, true)).unwrap();
            f += 0.5 * vi * fam.unit_deviance(*y, mu);
        }
        f
    };
    let mut b = vec![0.0; k];
    for _ in 0..200 {
        let mut hess = vec![0.0; k * k];
        let mut g: Vec<f64> = b.iter().map(|t| -c.ridge * t).collect();
        for i in 0..k {
            hess[i * k + i] = c.ridge;
        }
        for ((x, y), vi) in rows.iter().zip(v) {
            let xt: Vec<f64> = std::iter::once(1.0).chain(x.iter().copied()).collect();
            let wk = fam.working(dot_aug(&b, x, true), *y).unwrap();
            for i in 0..k {
                g[i] += vi * wk.w * (wk.z - wk.eta) * xt[i];
                for m in 0..k {
                    hess[i * k + m] += vi * wk.w * xt[i] * xt[m];
                }
            }
        }
        let d = oracle::solve(&hess, &g);
        let f0 = obj(&b);
        let mut t = 1.0;
        let mut nb: Vec<f64> = b.iter().zip(&d).map(|(a, e)| a + e).collect();
        while obj(&nb) > f0 && t > 1e-14 {
            t *= 0.5;
            nb = b.iter().zip(&d).map(|(a, e)| a + t * e).collect();
        }
        b = nb;
        if d.iter().all(|e| (t * e).abs() < 1e-14) {
            break;
        }
    }
    b
}

fn max_gap(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

/// Without decay the increment form is the sums form `β = (G + R I)⁻¹ h`,
/// `h += w W z x̃` (the module docs), checked row by row against the sums
/// form written longhand with the oracle's LU, a capped row resetting `h`
/// to `(G + R I) β`: the two share no solve, and agree to rounding
/// (measured 2e-15 over 300 rows; the bound is 1e-10 of the fit's size).
#[test]
fn the_increment_form_is_the_sums_form() {
    for family in [GlmFamily::Poisson, GlmFamily::Gamma] {
        let rows = stream(family, 4, 300, 3);
        let c = GlmCfg {
            step_cap: 1.0,
            ..cfg(family, 3, 0)
        };
        let mut m = Glm::new(c.clone()).unwrap();
        let fam = c.family();
        let k = 4;
        let mut g = vec![0.0; k * k];
        let mut h = vec![0.0; k];
        let mut b = vec![0.0; k];
        for (x, y) in &rows {
            let xt: Vec<f64> = std::iter::once(1.0).chain(x.iter().copied()).collect();
            let wk = fam.working(dot_aug(&b, x, true), *y).unwrap();
            for i in 0..k {
                for j in 0..k {
                    g[i * k + j] += wk.w * xt[i] * xt[j];
                }
                h[i] += wk.w * wk.z * xt[i];
            }
            let mut a = g.clone();
            for i in 0..k {
                a[i * k + i] += 1.0;
            }
            let mut nb = oracle::solve(&a, &h);
            let moved: f64 = xt
                .iter()
                .zip(nb.iter().zip(&b))
                .map(|(t, (n, o))| t * (n - o))
                .sum();
            if moved.abs() > 1.0 {
                for i in 0..k {
                    nb[i] = b[i] + (nb[i] - b[i]) / moved.abs();
                }
                h = (0..k)
                    .map(|i| (0..k).map(|j| a[i * k + j] * nb[j]).sum())
                    .collect();
            }
            b = nb;
            m.step(x, &[Some(*y)], 1.0, 1.0);
            let got = &m.coefficients()[0];
            let size = b.iter().fold(1.0_f64, |a, v| a.max(v.abs()));
            assert!(
                max_gap(got, &b) <= 1e-10 * size,
                "{family:?}: {got:?} vs {b:?}"
            );
        }
    }
}

/// With the buffer as long as the stream, the refit's answer is the batch
/// target itself: Newton by the oracle's LU from 0 agrees to 1e-9 (the two
/// stop at steps of 1e-11 and 1e-14 of a fit of order 1).
#[test]
fn the_refit_is_the_batch_fit_of_its_rows() {
    for family in [GlmFamily::Poisson, GlmFamily::Gamma] {
        let rows = stream(family, 4, 400, 7);
        let c = cfg(family, 3, 400);
        let mut m = Glm::new(c.clone()).unwrap();
        run(&mut m, &rows, |_| 1.0);
        let st = m.refit_status()[0];
        assert!(
            st.done && !st.failed && st.passes > 1 && st.passes < 30,
            "{st:?}"
        );
        let want = batch(&c, &rows, &vec![1.0; rows.len()]);
        let got = &m.coefficients()[0];
        assert!(
            max_gap(got, &want) < 1e-9,
            "{family:?}: {got:?} vs {want:?}"
        );
        assert_eq!(m.solve_failures(), 0);
    }
}

/// Under decay the refit weights each buffered row by the decays since it
/// and a row of weight 0 only ages the buffer: the batch target at those
/// weights, to 1e-9.
#[test]
fn the_refit_reads_the_decayed_weights() {
    let rows = stream(GlmFamily::Poisson, 3, 300, 11);
    let lam: f64 = 0.99;
    let c = GlmCfg {
        decay: Decay::Lam(lam),
        ..cfg(GlmFamily::Poisson, 2, 250)
    };
    let w = |i: usize| match i % 7 {
        3 => 0.0,
        5 => 2.0,
        _ => 1.0,
    };
    let mut m = Glm::new(c.clone()).unwrap();
    // The 250th row of positive weight fills the buffer.
    let mut learned = 0;
    let n = rows
        .iter()
        .enumerate()
        .position(|(i, _)| {
            learned += usize::from(w(i) > 0.0);
            learned == 250
        })
        .unwrap()
        + 1;
    run(&mut m, &rows[..n], w);
    assert!(m.refit_status()[0].done);
    let v: Vec<f64> = (0..n)
        .map(|i| w(i) * lam.powi((n - 1 - i) as i32))
        .collect();
    let want = batch(&c, &rows[..n], &v);
    assert!(max_gap(&m.coefficients()[0], &want) < 1e-9);
}

/// The online fit after the refit tracks the batch fit of every row: at
/// 3,000 rows and 4 coefficients within 0.02 of it (the study's gap at 10k
/// rows and 5 coefficients was 1e-3 in the worst of 50 seeds; at 3,000
/// rows a seed here is about 5e-3).
#[test]
fn the_online_fit_tracks_the_batch_fit() {
    for family in [GlmFamily::Poisson, GlmFamily::Gamma] {
        let rows = stream(family, 4, 3000, 5);
        let c = cfg(family, 3, default_glm_refit_rows(3, 1));
        let mut m = Glm::new(c.clone()).unwrap();
        run(&mut m, &rows, |_| 1.0);
        let want = batch(&c, &rows, &vec![1.0; rows.len()]);
        let gap = max_gap(&m.coefficients()[0], &want);
        assert!(gap < 0.02, "{family:?}: {gap}");
    }
}

/// Without decay a row of weight 0 changes nothing: the stream with those
/// rows, the first among them, is the stream without them to the bit, with
/// the refit and without it.
#[test]
fn a_zero_weight_row_is_no_row_without_decay() {
    let rows = stream(GlmFamily::Gamma, 3, 600, 2);
    for refit in [0, 200] {
        let c = GlmCfg {
            step_cap: default_glm_step_cap(refit),
            ..cfg(GlmFamily::Gamma, 2, refit)
        };
        let w = |i: usize| if i.is_multiple_of(5) { 0.0 } else { 1.0 };
        let mut with = Glm::new(c.clone()).unwrap();
        run(&mut with, &rows, w);
        let kept: Vec<_> = rows
            .iter()
            .enumerate()
            .filter(|(i, _)| w(*i) > 0.0)
            .map(|(_, r)| r.clone())
            .collect();
        let mut without = Glm::new(c).unwrap();
        run(&mut without, &kept, |_| 1.0);
        assert_eq!(with.gram, without.gram);
        assert_eq!(with.beta, without.beta);
        assert_eq!(with.refit, without.refit);
    }
}

/// Under decay a zero-weight row moves no coefficient and divides nothing,
/// the first row included.
#[test]
fn a_zero_weight_first_row_under_decay_is_finite() {
    let mut m = Glm::new(GlmCfg {
        decay: Decay::Halflife(10.0),
        ..cfg(GlmFamily::Poisson, 2, 50)
    })
    .unwrap();
    let s = m.step(&[0.5, -1.0], &[Some(3.0)], 1.0, 0.0);
    assert_eq!(s.n_eff, 0.0);
    assert!(m.beta[0].iter().chain(&m.gram[0]).all(|v| *v == 0.0));
    m.step(&[0.5, -1.0], &[Some(3.0)], 1.0, 1.0);
    assert!(m.beta[0].iter().all(|v| v.is_finite() && *v != 0.0));
    let b = m.beta.clone();
    m.step(&[0.5, -1.0], &[Some(3.0)], 7.0, 0.0);
    assert_eq!(m.beta, b);
}

/// A target outside the family's support teaches nothing and counts
/// nothing toward its weight, as an absent one.
#[test]
fn a_target_outside_the_support_is_absent() {
    let rows = stream(GlmFamily::Gamma, 3, 50, 4);
    let c = cfg(GlmFamily::Gamma, 2, 20);
    let mut bad = Glm::new(c.clone()).unwrap();
    let mut none = Glm::new(c).unwrap();
    for (i, (x, y)) in rows.iter().enumerate() {
        let (b, n) = if i % 3 == 1 {
            (Some(0.0), None)
        } else {
            (Some(*y), Some(*y))
        };
        assert_eq!(bad.step(x, &[b], 1.0, 1.0), none.step(x, &[n], 1.0, 1.0));
    }
    assert_eq!(bad, none);
}

/// A capped row moves its own `η` by exactly the cap.
#[test]
fn the_cap_holds_a_rows_own_step() {
    let c = GlmCfg {
        step_cap: 0.25,
        ..cfg(GlmFamily::Gamma, 2, 0)
    };
    let mut m = Glm::new(c).unwrap();
    let x = [1.5, -0.5];
    let before = dot_aug(&m.beta[0], &x, true);
    m.step(&x, &[Some(40.0)], 1.0, 1.0);
    let after = dot_aug(&m.beta[0], &x, true);
    assert!(
        ((after - before) - 0.25).abs() < 1e-15,
        "{}",
        after - before
    );
}

/// The start-up failure the refit and the cap exist for: Gamma with 20
/// coefficients from zero runs away on the plain recursion on this seed
/// (an `η` at the clip), and with either remedy stays near the truth.
#[test]
fn the_remedies_hold_the_start() {
    let rows = stream(GlmFamily::Gamma, 20, 2000, 6);
    let fit = |refit: u32, cap: f64| {
        let mut m = Glm::new(GlmCfg {
            step_cap: cap,
            ..cfg(GlmFamily::Gamma, 19, refit)
        })
        .unwrap();
        run(&mut m, &rows, |_| 1.0);
        (m.coefficients()[0][0] - 0.5).abs()
    };
    assert!(
        fit(0, f64::INFINITY) > 1.0,
        "the plain recursion held on this seed"
    );
    assert!(fit(200, f64::INFINITY) < 0.1);
    assert!(fit(0, 1.0) < 0.1);
}

#[test]
fn predict_is_the_steps_prediction() {
    let rows = stream(GlmFamily::Poisson, 3, 400, 8);
    let mut m = Glm::new(GlmCfg {
        min_weight: 30.0,
        decay: Decay::Halflife(50.0),
        ..cfg(GlmFamily::Poisson, 2, 100)
    })
    .unwrap();
    for (x, y) in &rows {
        let p = m.predict(x, 1.0);
        let s = m.step(x, &[Some(*y)], 1.0, 1.0);
        assert_eq!(p.pred[0].to_bits(), s.pred[0].to_bits());
        assert_eq!(p.n_eff, s.n_eff);
    }
}

#[test]
fn the_canonical_links_fit_through_the_refit() {
    // Gamma's inverse link: the recursion's start has no mean, so nothing
    // is predicted before the refit, and after it the fit predicts.
    let rows = stream(GlmFamily::Gamma, 3, 400, 9);
    let mut m = Glm::new(GlmCfg {
        link: GlmLink::Inverse,
        ..cfg(GlmFamily::Gamma, 2, 100)
    })
    .unwrap();
    let preds = run(&mut m, &rows, |_| 1.0);
    assert!(preds[..100].iter().all(|p| p.is_nan()));
    assert!(preds[100..].iter().all(|p| p.is_finite() && *p > 0.0));
    let st = m.refit_status()[0];
    assert!(st.done && !st.failed, "{st:?}");
}

#[test]
fn cfg_validation_refuses_each_bad_field() {
    let base = cfg(GlmFamily::Poisson, 2, 10);
    let refused = |c: GlmCfg, what: &str| {
        let e = Glm::new(c).unwrap_err();
        assert!(e.contains(what), "{what}: {e}");
    };
    refused(
        GlmCfg {
            ridge: 0.0,
            ..base.clone()
        },
        "ridge must be above 0",
    );
    refused(
        GlmCfg {
            step_cap: 0.0,
            ..base.clone()
        },
        "step_cap must be above 0",
    );
    refused(
        GlmCfg {
            link: GlmLink::Logit,
            ..base.clone()
        },
        "takes the link log",
    );
    refused(
        GlmCfg {
            refit_rows: crate::MAX_REFIT_ROWS + 1,
            ..base.clone()
        },
        "past the cap of 262144 bytes",
    );
    refused(
        GlmCfg {
            family: GlmFamily::Gamma,
            link: GlmLink::Inverse,
            refit_rows: 0,
            ..base.clone()
        },
        "needs refit_rows above 0",
    );
    refused(
        GlmCfg {
            family: GlmFamily::Gamma,
            link: GlmLink::Inverse,
            fit_intercept: false,
            ..base.clone()
        },
        "fit_intercept is off",
    );
    refused(
        GlmCfg {
            family: GlmFamily::Tweedie,
            var_power: 2.0,
            ..base
        },
        "var_power must be above 1",
    );
}

#[test]
fn a_state_of_the_wrong_shape_is_refused() {
    let m = Glm::new(cfg(GlmFamily::Poisson, 2, 10)).unwrap();
    let mut s = m.state();
    if let ModelState::Glm(g) = &mut s.model {
        g.beta[0].pop();
    }
    let e = Glm::restore(&s).unwrap_err().to_string();
    assert!(e.contains("wrong shape"), "{e}");
    let mut s = m.state();
    if let ModelState::Glm(g) = &mut s.model {
        g.refit.clear();
    }
    assert!(Glm::restore(&s).is_err());
    let mut s = m.state();
    if let ModelState::Glm(g) = &mut s.model {
        g.refit[0].rows.push(1.0);
    }
    assert!(Glm::restore(&s).is_err());
}

/// A state saved mid-buffer and after the refit goes on as the model it
/// was saved from, to the bit.
#[test]
fn a_saved_state_goes_on_to_the_bit() {
    let rows = stream(GlmFamily::Poisson, 3, 300, 12);
    for cut in [0, 50, 150, 299] {
        let mut m = Glm::new(GlmCfg {
            decay: Decay::Halflife(80.0),
            ..cfg(GlmFamily::Poisson, 2, 100)
        })
        .unwrap();
        run(&mut m, &rows[..cut], |_| 1.0);
        let bytes = rmp_serde::to_vec_named(&m.state()).unwrap();
        let mut back = Glm::restore(&rmp_serde::from_slice(&bytes).unwrap()).unwrap();
        for (x, y) in &rows[cut..] {
            assert_eq!(
                back.step(x, &[Some(*y)], 1.0, 1.0),
                m.step(x, &[Some(*y)], 1.0, 1.0)
            );
        }
        assert_eq!(back, m);
    }
}

#[test]
fn giving_the_refit_up_frees_its_rows() {
    let rows = stream(GlmFamily::Poisson, 3, 50, 1);
    let mut m = Glm::new(cfg(GlmFamily::Poisson, 2, 200)).unwrap();
    assert_eq!(m.refit_reservation(), 200 * 4 * 8);
    run(&mut m, &rows, |_| 1.0);
    m.refit_give_up();
    let st = m.refit_status()[0];
    assert!(st.done && st.over_budget && m.refit[0].rows.is_empty());
    run(&mut m, &rows, |_| 1.0);
    assert!(m.refit[0].rows.is_empty());
}

#[test]
fn the_default_refit_and_cap() {
    assert_eq!(default_glm_refit_rows(1, 1), 200);
    assert_eq!(default_glm_refit_rows(49, 1), 500);
    assert_eq!(default_glm_refit_rows(101, 1), 0);
    // Held to the group's cap: 100 features, 262,144 / (8 · 102) = 321
    // rows where `10 k` is 1,010; at 2 targets 160 rows, under `2 k`, off.
    assert_eq!(default_glm_refit_rows(100, 1), 321);
    assert_eq!(default_glm_refit_rows(100, 2), 0);
    for (nf, t) in [(1, 1), (20, 1), (20, 8), (50, 3), (100, 1), (100, 2)] {
        let c = GlmCfg {
            n_targets: t,
            ..cfg(GlmFamily::Poisson, nf, default_glm_refit_rows(nf, t))
        };
        assert!(c.refit_bytes() <= GLM_GROUP_REFIT_CAP, "{nf} {t}");
        Glm::new(c).unwrap();
    }
    assert_eq!(default_glm_step_cap(0), 1.0);
    assert!(default_glm_step_cap(200).is_infinite());
}

/// The refit's buffers are the one part of the state whose size is not the
/// model's shape: a configuration past the group's cap is refused as it is
/// built, naming the bytes, and one at the cap holds no more than it says
/// however many rows run past the count.
#[test]
fn the_refit_buffer_is_held_to_the_groups_cap() {
    let at_cap = (GLM_GROUP_REFIT_CAP / (8 * 2 * 5)) as u32;
    let over = GlmCfg {
        n_targets: 2,
        ..cfg(GlmFamily::Poisson, 3, at_cap + 1)
    };
    let e = Glm::new(over).unwrap_err();
    assert!(
        e.contains("262160 bytes a group, past the cap of 262144") && e.contains("at most 3276,"),
        "{e}"
    );
    let c = GlmCfg {
        n_targets: 2,
        ..cfg(GlmFamily::Poisson, 3, at_cap)
    };
    assert_eq!(c.refit_bytes(), 262_080);
    let mut m = Glm::new(c.clone()).unwrap();
    let rows = stream(GlmFamily::Poisson, 4, 2 * at_cap as usize, 3);
    let mut most = 0;
    for (i, (x, y)) in rows.iter().enumerate() {
        m.step(x, &[Some(*y), Some(*y)], 1.0, 1.0);
        let held: usize = m.refit.iter().map(|r| 8 * r.rows.len()).sum();
        most = most.max(held);
        assert!(held <= c.refit_bytes(), "row {i}: {held}");
    }
    assert_eq!(
        most,
        8 * 2 * 5 * (at_cap as usize - 1),
        "the buffer filled to one row short"
    );
    assert!(m.refit.iter().all(|r| r.done && r.rows.is_empty()));
}
