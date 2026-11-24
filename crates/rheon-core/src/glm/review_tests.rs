//! The 2026-11-23 review's findings on `glm` (PLAN 578b): each test either
//! failed on 51671e2e for the reason the review gives, and holds the fix,
//! or shows where the claim does not hold. The finding's ID leads each
//! test's doc.

use super::tests::{cfg_of, stream_of};
use super::*;
use crate::family::ETA_CLIP;
use crate::oracle;

fn lcg(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*s >> 11) as f64 + 0.5) / (1u64 << 53) as f64
}

/// A refit buffer of rows `(x, y, w)`: the features, the target and the
/// weight, as the model keeps them.
fn buffer(rows: &[(Vec<f64>, f64, f64)]) -> Vec<f64> {
    let mut out = Vec::new();
    for (x, y, w) in rows {
        out.extend_from_slice(x);
        out.push(*y);
        out.push(*w);
    }
    out
}

/// The batch target, `argmin Σ v d(y, μ)/2 + R/2 |b|²`, by Newton with step
/// halving on full square systems solved by the oracle's LU (no arithmetic
/// of the model's Cholesky), from `start`, for 200 passes.
fn batch(c: &GlmCfg, buf: &[f64], start: &[f64]) -> Vec<f64> {
    let fam = c.family();
    let (k, nf) = (c.k_total(), c.n_features);
    let obj = |b: &[f64]| objective(c, &fam, buf, nf, b);
    let mut b = start.to_vec();
    for _ in 0..200 {
        let mut hess = vec![0.0; k * k];
        let mut g: Vec<f64> = b.iter().map(|t| -c.ridge * t).collect();
        for i in 0..k {
            hess[i * k + i] = c.ridge;
        }
        for row in buf.chunks_exact(nf + 2) {
            let xt: Vec<f64> = std::iter::once(1.0)
                .chain(row[..nf].iter().copied())
                .collect();
            let wk = fam.working(dot_aug(&b, &row[..nf], true), row[nf]).unwrap();
            for i in 0..k {
                g[i] += row[nf + 1] * wk.w * (wk.z - wk.eta) * xt[i];
                for m in 0..k {
                    hess[i * k + m] += row[nf + 1] * wk.w * xt[i] * xt[m];
                }
            }
        }
        let d = oracle::solve(&hess, &g);
        let mut t = 1.0;
        let mut nb: Vec<f64> = b.iter().zip(&d).map(|(a, e)| a + e).collect();
        if d.iter().any(|e| e.abs() > 1e-6) {
            let f0 = obj(&b);
            while (obj(&nb).is_nan() || obj(&nb) > f0) && t > 1e-14 {
                t *= 0.5;
                nb = b.iter().zip(&d).map(|(a, e)| a + t * e).collect();
            }
        }
        b = nb;
    }
    b
}

/// GLM-1 (1): a Poisson target past the log link's clip, `e^30`. The
/// objective reads the clipped mean, flat in `η` past 30, but the score
/// read the clipped mean's residual, as if it were not: so the refit
/// started past the clip, at `ln y`, and the steps the slopes accepted
/// raised the objective's penalty pass after pass, the intercept going to
/// 32.33 at `y = 1e14` and to 1e6 at `1e25`, there reported as converged
/// (51671e2e). Now the score past the clip is the objective's, 0, the
/// start is held to the clip, and the fit stops at the clip's edge, the
/// closest mean the link gives. The other side, a target inside the clip: the fit
/// converges to `ln y`, the batch fit's.
#[test]
fn glm_1_a_target_past_the_clip_never_raises_the_refits_objective() {
    for y in [1e12, 1e14, 1e25] {
        let c = cfg_of(GlmFamily::Poisson, 1, 20);
        let fam = c.family();
        let mut s = 7u64;
        let rows: Vec<_> = (0..20)
            .map(|_| (vec![0.2 * (lcg(&mut s) - 0.5)], y, 1.0))
            .collect();
        let buf = buffer(&rows);
        let start = |b0: f64| objective(&c, &fam, &buf, 1, &[b0, 0.0]);
        let f0 = start(fam.start_eta(y)).min(start(fam.start_eta(y).min(ETA_CLIP)));
        let (b, passes, converged) = refit_fit(&c, &buf, 1).unwrap();
        let f = objective(&c, &fam, &buf, 1, &b);
        // `F`'s own rounding: its terms are of order `y`, and their sum
        // far smaller near the fit.
        let rounding = 64.0 * f64::EPSILON * 20.0 * (y + ETA_CLIP.exp().min(y));
        assert!(
            f <= f0 + rounding,
            "y = {y:e}: the objective rose from {f0} to {f} ({b:?}, {passes} passes)"
        );
        if y < ETA_CLIP.exp() {
            assert!(converged && passes < GLM_REFIT_PASSES, "{passes}");
            assert!((b[0] - y.ln()).abs() < 1e-9 && b[1].abs() < 1e-9, "{b:?}");
        } else {
            assert!(
                (b[0] - ETA_CLIP).abs() < 1e-6 && b[1].abs() < 1e-6,
                "y = {y:e}: {b:?}, {passes} passes, converged {converged}"
            );
        }
    }
}

/// GLM-1 (2): a line search that finds no step that lowers the objective
/// is no convergence. Under the Gamma's inverse link every halving of this
/// pass's step leaves a row's `η` below 0, where the link gives no mean,
/// and the refit returned its start, a fit `10¹⁵` times worse than one in
/// reach, as converged (51671e2e). Converged now means the fit is a
/// minimiser: no worse than `[1, −1 + 10⁻¹⁵]`, whose objective is the
/// penalty alone.
#[test]
fn glm_1_a_line_search_that_finds_no_fall_is_not_converged() {
    let c = GlmCfg {
        link: GlmLink::Inverse,
        ..cfg_of(GlmFamily::Gamma, 1, 2)
    };
    let fam = c.family();
    let buf = buffer(&[(vec![1.0], 1e15, 1.0), (vec![0.0], 1.0, 1e16)]);
    let (b, passes, converged) = refit_fit(&c, &buf, 1).unwrap();
    let f = objective(&c, &fam, &buf, 1, &b);
    let reach = objective(&c, &fam, &buf, 1, &[1.0, -1.0 + 1e-15]);
    assert!(reach < 2.0);
    assert!(
        !converged || f <= reach * (1.0 + 1e-9),
        "converged at {b:?} after {passes} passes, objective {f} against {reach} in reach"
    );
}

/// GLM-1 (3): the trapezoid rule on the slopes decides only a step whose
/// predicted fall is under `1e-10 (1 + |F|)`, below what `F` resolves, so
/// on a family whose deviance is not convex in `η` (Tweedie, log link, `p
/// = 3`, where `μ > 2y`) and on the Gamma's log link it never stands for a
/// rise the objective can show: each refit lands on the oracle's batch fit
/// (Newton by faer's LU from the refit's own start). Not reproduced.
#[test]
fn glm_1_the_trapezoid_rule_holds_on_non_quadratic_deviances() {
    for (family, power) in [(GlmFamily::Tweedie, 3.0), (GlmFamily::Gamma, 0.0)] {
        for seed in 1..=4u64 {
            let c = GlmCfg {
                var_power: power,
                ..cfg_of(family, 2, 80)
            };
            let rows: Vec<_> = stream_of(GlmFamily::Gamma, 3, 80, seed)
                .into_iter()
                // Means far above the targets on some rows: the non-convex
                // part of the Tweedie's deviance.
                .map(|(x, y)| (x, y * if seed % 2 == 0 { 0.2 } else { 1.0 }, 1.0))
                .collect();
            let buf = buffer(&rows);
            let (b, _, converged) = refit_fit(&c, &buf, 2).unwrap();
            let fam = c.family();
            let (mut sv, mut svy) = (0.0, 0.0);
            for (_, y, w) in &rows {
                sv += w;
                svy += w * y;
            }
            let want = batch(&c, &buf, &[fam.start_eta(svy / sv), 0.0, 0.0]);
            let gap = b
                .iter()
                .zip(&want)
                .fold(0.0_f64, |m, (p, q)| m.max((p - q).abs()));
            assert!(
                converged && gap < 1e-10,
                "{family:?} seed {seed}: {b:?} vs {want:?}"
            );
        }
    }
}

/// GLM-2 (2): under decay the recursion's sums are not those of the
/// objective the module docs gave, `F_t = Σ w λ^age d/2 + λ^(age_0) R/2
/// |β|²`, whose stationarity reads `h_t = λ h_{t−1} + w W z x̃` for `h_t =
/// (G_t + R_t I) β_t`. Since `G` decays and the step's `R` does not, the
/// recursion's identity is `h_t = λ h_{t−1} + (1 − λ) R β_{t−1} + w W z
/// x̃`, `h_t = (G_t + R I) β_t`: a proximal term pulling to the past fits,
/// which vanishes as the fit stops moving. The identity, row by row, to 1e-10 of
/// its terms' size; the documented one misses by far more.
#[test]
fn glm_2_the_recursions_identity_under_decay_is_the_proximal_one() {
    let lam: f64 = 0.9;
    let c = GlmCfg {
        decay: Decay::Lam(lam),
        step_cap: f64::INFINITY,
        ..cfg_of(GlmFamily::Poisson, 2, 0)
    };
    let fam = c.family();
    let (k, r) = (3, c.ridge);
    let mut m = Glm::new(c).unwrap();
    let full = |g: &[f64]| -> Vec<f64> {
        let mut a = vec![0.0; k * k];
        for i in 0..k {
            for j in 0..=i {
                a[i * k + j] = g[at(i, j)];
                a[j * k + i] = g[at(i, j)];
            }
            a[i * k + i] += r;
        }
        a
    };
    let times = |a: &[f64], b: &[f64]| -> Vec<f64> {
        (0..k)
            .map(|i| (0..k).map(|j| a[i * k + j] * b[j]).sum())
            .collect()
    };
    let mut worst_doc = 0.0_f64;
    for (i, (x, y)) in stream_of(GlmFamily::Poisson, 3, 60, 3).iter().enumerate() {
        let (b0, g0) = (m.beta[0].clone(), m.gram[0].clone());
        let wk = fam.working(dot_aug(&b0, x, true), *y).unwrap();
        let w = 1.0 + (i % 3) as f64 * 0.5;
        m.step(x, &[Some(*y)], 1.0, w);
        let h = times(&full(&m.gram[0]), &m.beta[0]);
        let h0 = times(&full(&g0), &b0);
        let xt = [1.0, x[0], x[1]];
        let size = h.iter().chain(&h0).fold(1.0_f64, |s, v| s.max(v.abs()));
        for t in 0..k {
            let score = w * wk.w * wk.z * xt[t];
            let want = lam * h0[t] + (1.0 - lam) * r * b0[t] + score;
            assert!(
                (h[t] - want).abs() <= 1e-10 * size,
                "row {i}: {} vs {want}",
                h[t]
            );
            worst_doc = worst_doc.max((h[t] - (lam * h0[t] + score)).abs() / size);
        }
    }
    assert!(worst_doc > 1e-3, "{worst_doc}");
}

/// GLM-3 (1): a group past `refit_budget` gives its refit up before its
/// first row, and under the Gamma's inverse link (and the Tweedie's power
/// one) every coefficient 0 gave no mean, so the group never learned and
/// read NaN forever (51671e2e; held here as a `should_panic` until the
/// fix). Fixed by the exact row step (PLAN 581), which starts from `η = 0`
/// and lands each row at a positive `η`: the group predicts from its
/// second row (`prox_tests::a_given_up_refit_learns_from_its_first_row`
/// holds its fit). The other side: a group whose refit runs fits
/// (`the_canonical_links_fit_through_the_refit`).
#[test]
fn glm_3_a_given_up_refit_under_a_link_undefined_at_zero_learns() {
    for (family, link, power) in [
        (GlmFamily::Gamma, GlmLink::Inverse, 0.0),
        (GlmFamily::Tweedie, GlmLink::Power, 1.5),
    ] {
        let c = GlmCfg {
            link,
            var_power: power,
            ..cfg_of(family, 2, 50)
        };
        let mut m = Glm::new(c).unwrap();
        m.refit_give_up();
        let mut finite = 0;
        for (x, y) in stream_of(GlmFamily::Gamma, 3, 400, 21) {
            finite += usize::from(m.step(&x, &[Some(y)], 1.0, 1.0).pred[0].is_finite());
        }
        assert_eq!(
            finite,
            399,
            "{family:?} {link:?}: a given-up refit learns from its first row, {:?}",
            m.coefficients()
        );
    }
}

/// GLM-3 (2): a spec that leaves `step_cap` out where the refit runs gets
/// no cap, the refit replacing what the capped start left; a group that
/// gives its refit up never has it replaced, and kept no cap from its first
/// row (51671e2e), the runaway the start's cap is there to stop. It now
/// keeps the start's cap of 1 for good. The other side: once a refit has
/// run, a row may move its own `η` by more than 1.
#[test]
fn glm_3_a_given_up_refit_keeps_the_starts_cap() {
    let c = cfg_of(GlmFamily::Gamma, 2, 200);
    assert_eq!(c.step_cap, f64::INFINITY);
    let rows = stream_of(GlmFamily::Gamma, 3, 300, 2);
    let moved = |m: &mut Glm, x: &[f64], y: f64| -> f64 {
        let before = dot_aug(&m.beta[0], x, true);
        m.step(x, &[Some(y)], 1.0, 1.0);
        (dot_aug(&m.beta[0], x, true) - before).abs()
    };
    let mut m = Glm::new(c.clone()).unwrap();
    m.refit_give_up();
    let worst = rows
        .iter()
        .map(|(x, y)| moved(&mut m, x, *y))
        .fold(0.0_f64, f64::max);
    assert!(worst <= 1.0 + 1e-12, "a row moved its own η by {worst}");
    let mut m = Glm::new(c).unwrap();
    for (x, y) in &rows[..200] {
        m.step(x, &[Some(*y)], 1.0, 1.0);
    }
    assert!(m.refit_status()[0].done && !m.refit_status()[0].over_budget);
    let x = &rows[200].0;
    let mu = m.predict(x, 1.0).pred[0];
    assert!(moved(&mut m, x, 1e3 * mu) > 1.0);
}

/// GLM-4 (3): the Tweedie deviance at `p = 30` and `μ = e^−30` has two
/// terms past the largest double, `+∞` and `−∞`, and read NaN (51671e2e);
/// its exact value is past the largest double too, so it is `+∞`. The other
/// side: where both terms are finite, the deviance is as before, to the bit.
#[test]
fn glm_4_a_tweedie_deviance_past_the_largest_double_is_infinite() {
    let f = Family::new("glm", GlmFamily::Tweedie, GlmLink::Log, 30.0).unwrap();
    assert_eq!(f.unit_deviance(1.0, (-30.0_f64).exp()), f64::INFINITY);
    let mu = (-20.0_f64).exp();
    let p = 30.0_f64;
    let plain = 2.0
        * (1.0 / ((1.0 - p) * (2.0 - p)) - mu.powf(1.0 - p) / (1.0 - p)
            + mu.powf(2.0 - p) / (2.0 - p));
    assert_eq!(f.unit_deviance(1.0, mu).to_bits(), plain.to_bits());
}

/// GLM-4 (4): the Gamma deviance's `y/μ` underflowed to 0 at `y = 1e-200`,
/// `μ = 1e150`, and read `+∞` (51671e2e), where it is `2 (350 ln 10 − 1 +
/// 1e-350)`, 1609.8. The other side: a ratio in range keeps its bits.
#[test]
fn glm_4_a_gamma_deviance_whose_ratio_underflows_is_finite() {
    let f = Family::new("glm", GlmFamily::Gamma, GlmLink::Inverse, 0.0).unwrap();
    let want = 2.0 * (350.0 * std::f64::consts::LN_10 - 1.0);
    let got = f.unit_deviance(1e-200, 1e150);
    assert!((got - want).abs() <= 1e-12 * want, "{got} vs {want}");
    let (y, mu) = (1.7_f64, 2.3_f64);
    let plain = 2.0 * (-(y / mu).ln() + (y - mu) / mu);
    assert_eq!(f.unit_deviance(y, mu).to_bits(), plain.to_bits());
}

/// GLM-4 (5): the review's rows for a step past the largest double: a
/// ridge of 1e-300, no cap, rows of weight 1e100 at a feature of 1e-205
/// and a target of 1e100. The slope reaches 1.7e279 on the first row and
/// stays there, every coefficient finite over 40 rows: the solve's
/// `(G + R I)⁻¹` keeps the step's size where the feature is that small.
/// Not reproduced.
#[test]
fn glm_4_the_reviews_rows_keep_every_coefficient_finite() {
    let c = GlmCfg {
        ridge: 1e-300,
        step_cap: f64::INFINITY,
        ..cfg_of(GlmFamily::Gamma, 1, 0)
    };
    let mut m = Glm::new(c).unwrap();
    for _ in 0..40 {
        m.step(&[1e-205], &[Some(1e100)], 1.0, 1e100);
        assert!(m.beta[0].iter().all(|v| v.is_finite()), "{:?}", m.beta[0]);
    }
}

/// GLM-4 (5): no family takes a target that is not finite. The bank never
/// hands one (a value past 1e100 is no target), but `Family` is public.
#[test]
fn glm_4_no_family_takes_an_infinite_target() {
    for (family, link, p) in [
        (GlmFamily::Poisson, GlmLink::Log, 0.0),
        (GlmFamily::Gamma, GlmLink::Log, 0.0),
        (GlmFamily::Binomial, GlmLink::Logit, 0.0),
        (GlmFamily::Tweedie, GlmLink::Log, 1.5),
        (GlmFamily::Tweedie, GlmLink::Log, 2.5),
    ] {
        let f = Family::new("glm", family, link, p).unwrap();
        assert!(!f.takes(f64::INFINITY) && !f.takes(f64::NAN), "{family:?}");
        assert!(f.takes(1.0), "{family:?}");
    }
}

/// GLM-5: a row whose `G` is finite but whose score is not (`x̃ = [1,
/// 1e100]` at `η = −30`: `s x̃ = [1.07e211, ∞]`). The step failed and moved
/// no coefficient, yet the row's `x̃x̃'` stayed in `G`, 1e298 on the slope's
/// diagonal, so without decay that slope never moved again (51671e2e). Such
/// a row now teaches nothing, as one whose `G` overflows does. The other
/// side: the next ordinary row is learned as by a model that never saw it;
/// and a solve that fails on a finite system still keeps its row's `G`
/// (`mutant_tests.rs`, `a_solve_that_fails_in_a_learned_row_is_counted`).
#[test]
fn glm_5_a_row_whose_score_overflows_leaves_the_gram_as_it_was() {
    let c = GlmCfg {
        step_cap: f64::INFINITY,
        ..cfg_of(GlmFamily::Gamma, 1, 0)
    };
    let mut m = Glm::new(c).unwrap();
    m.beta[0] = vec![-30.0, 0.0];
    let twin = m.clone();
    m.step(&[1e100], &[Some(1e100)], 1.0, 1e98);
    assert_eq!(m.beta[0], twin.beta[0]);
    assert_eq!(m.gram[0], twin.gram[0], "the row's x̃x̃' stayed in G");
    let mut twin = twin;
    let (a, b) = (
        m.step(&[0.5], &[Some(2.0)], 1.0, 1.0),
        twin.step(&[0.5], &[Some(2.0)], 1.0, 1.0),
    );
    assert_eq!(a.pred[0].to_bits(), b.pred[0].to_bits());
    assert_eq!(m.beta[0], twin.beta[0]);
}

/// GLM-7: the stamp handed for a row is taken by that row whatever its
/// weight: a row of weight 0, or one refused for a feature that is not a
/// number (which is stepped again at weight 0), leaves no stamp for a
/// later row that is handed none. That row closes the span on the clock
/// summed from the steps. The review read the take as under `weight > 0`;
/// it is not. Not reproduced.
#[test]
fn glm_7_a_stamp_is_taken_by_its_own_row_whatever_its_weight() {
    let c = GlmCfg {
        refit_clock: Some(5.0),
        ..cfg_of(GlmFamily::Poisson, 1, 0)
    };
    let rows = stream_of(GlmFamily::Poisson, 2, 30, 4);
    for odd in [0.0, f64::NAN] {
        let mut m = Glm::new(c.clone()).unwrap();
        m.stamp_next(Stamp::from(1000.0));
        if odd.is_nan() {
            m.step(&[f64::NAN], &[Some(1.0)], 1.0, 1.0);
        } else {
            m.step(&rows[0].0, &[Some(1.0)], 1.0, 0.0);
        }
        assert!(m.next.is_none());
        // Unstamped rows a unit apart: the span closes on the sixth.
        for (i, (x, y)) in rows.iter().enumerate().take(6) {
            assert!(!m.refit_status()[0].done, "row {i}");
            m.step(x, &[Some(*y)], 1.0, 1.0);
        }
        assert!(m.refit_status()[0].done);
    }
}

/// GLM-EXP-LOW (3): a weight below 0 that a caller hands the model directly
/// learns nothing, as one of 0 does, and adds nothing to the weights either
/// (51671e2e added it to `n_eff` and the target's weight).
#[test]
fn low_3_a_negative_weight_is_a_weight_of_0() {
    let rows = stream_of(GlmFamily::Poisson, 2, 20, 5);
    let c = cfg_of(GlmFamily::Poisson, 1, 0);
    let (mut a, mut b) = (Glm::new(c.clone()).unwrap(), Glm::new(c).unwrap());
    for (i, (x, y)) in rows.iter().enumerate() {
        let w = if i == 7 { -1.0 } else { 1.0 };
        let p = a.step(x, &[Some(*y)], 1.0, w);
        let q = b.step(x, &[Some(*y)], 1.0, w.max(0.0));
        assert_eq!(p.pred[0].to_bits(), q.pred[0].to_bits());
        assert_eq!(p.n_eff, q.n_eff, "row {i}");
    }
    assert_eq!(a.target_weights(), b.target_weights());
}

/// GLM-EXP-LOW (5): a state whose span refit holds rows but no first stamp,
/// or records a stop at the cap where no span could stop it, is refused as
/// damaged rather than run.
#[test]
fn low_5_a_state_missing_its_span_start_or_stopped_without_a_span_is_refused() {
    let c = GlmCfg {
        refit_clock: Some(50.0),
        ..cfg_of(GlmFamily::Poisson, 1, 0)
    };
    let mut m = Glm::new(c).unwrap();
    for (x, y) in stream_of(GlmFamily::Poisson, 2, 5, 6) {
        m.step(&x, &[Some(y)], 1.0, 1.0);
    }
    assert!(Glm::restore(&m.state()).is_ok());
    let mut bad = m.clone();
    bad.refit[0].first = None;
    assert!(Glm::restore(&bad.state()).is_err());
    let mut m = Glm::new(cfg_of(GlmFamily::Poisson, 1, 0)).unwrap();
    for (x, y) in stream_of(GlmFamily::Poisson, 2, 5, 6) {
        m.step(&x, &[Some(y)], 1.0, 1.0);
    }
    let mut bad = m.clone();
    bad.over = Some(1);
    assert!(Glm::restore(&bad.state()).is_err());
}

/// GLM-EXP-LOW (6): the refusal of a link undefined at 0 without a refit
/// names both refits, a count of rows and a span of the clock.
#[test]
fn low_6_the_links_refusal_names_both_refits() {
    let c = GlmCfg {
        link: GlmLink::Inverse,
        ..cfg_of(GlmFamily::Gamma, 1, 0)
    };
    let e = c.validate().unwrap_err();
    assert!(e.contains("refit_rows") && e.contains("refit_span"), "{e}");
}

/// GLM-4 (1), a confirmed defect whose fix is the user's to allow: the
/// score `W (z − η)` is formed through `z = η + (y − μ) g'(μ)`, so at `η =
/// 30` and a residual of order 1 it keeps two digits of `y − μ`, the rest
/// lost to `η`'s rounding. The fix, a closed-form score in `Working`,
/// moves every `glm` output's last bits; this test fails once it lands, and
/// is then turned into a plain test.
#[test]
#[should_panic(expected = "digits")]
fn glm_4_the_score_at_the_clip_loses_its_digits() {
    let f = Family::new("glm", GlmFamily::Poisson, GlmLink::Log, 0.0).unwrap();
    let wk = f.working(ETA_CLIP, ETA_CLIP.exp().round() + 1.0).unwrap();
    let y = ETA_CLIP.exp().round() + 1.0;
    let got = wk.w * (wk.z - wk.eta);
    // Exact: `y` and `μ` are within a factor of 2 (Sterbenz).
    let want = y - wk.mu;
    let rel = (got - want).abs() / want.abs();
    assert!(
        rel <= 1e-12,
        "the score keeps {:.1} digits: {got} vs {want}",
        -rel.log10()
    );
}

/// GLM-4 (2), a confirmed defect whose fix is the user's to allow: the
/// binomial's weight `μ (1 − μ)` at `η = 30` forms `1 − μ` by subtraction,
/// `1e-13` from 1, so it keeps three digits. The exact weight is `e^−η /
/// (1 + e^−η)²`. The fix moves every binomial `glm` output's last bits.
#[test]
#[should_panic(expected = "digits")]
fn glm_4_the_binomials_weight_at_the_clip_loses_its_digits() {
    let f = Family::new("glm", GlmFamily::Binomial, GlmLink::Logit, 0.0).unwrap();
    let wk = f.working(ETA_CLIP, 1.0).unwrap();
    let e = (-ETA_CLIP).exp();
    let want = e / ((1.0 + e) * (1.0 + e));
    let rel = (wk.w - want).abs() / want;
    assert!(
        rel <= 1e-12,
        "the weight keeps {:.1} digits: {} vs {want}",
        -rel.log10(),
        wk.w
    );
}
