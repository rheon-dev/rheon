//! Tests that hold `glm.rs`'s lines to numbers a mutation of them would
//! change (docs/PLAN.md task 546): each pins a value from its definition,
//! beside an ordinary case where the line has one.

use super::tests::{cfg_of, stream_of};
use super::*;

/// A buffer of `n` rows of `nf + 2` numbers: the feature(s), the target and
/// the weight, `x_i = (i + 1)/2 - n/4` in each feature column.
fn buffer(nf: usize, ys: &[f64], ws: &[f64]) -> Vec<f64> {
    let n = ys.len();
    let mut out = Vec::new();
    for i in 0..n {
        for j in 0..nf {
            out.push((i + 1) as f64 / 2.0 - n as f64 / 4.0 + j as f64 * 0.1);
        }
        out.push(ys[i]);
        out.push(ws[i]);
    }
    out
}

/// A default `refit_rows` is `max(200, 10 k)` held to what the cap holds,
/// and off below `2 k` rows: at one feature (`k = 2`) the cap holds `4`
/// rows at 96 bytes (`8 · 3 · 4`), exactly `2 k`, which is kept, and 3 at
/// 95, which is not.
#[test]
fn a_default_refit_of_two_rows_a_coefficient_is_kept() {
    assert_eq!(default_glm_refit_rows_within(1, 1, 96), 4);
    assert_eq!(default_glm_refit_rows_within(1, 1, 95), 0);
    assert_eq!(default_glm_refit_rows_within(1, 1, 24 * 200), 200);
}

/// A waiting buffer holds fewer rows than its count: one that holds as many
/// is not the shape a saved state can be in (the refit ran when it filled).
#[test]
fn a_waiting_buffer_one_short_of_its_count_is_the_shape_and_a_full_one_is_not() {
    let rows = |learned: u32| GlmRefit {
        rows: vec![0.0; learned as usize * 4],
        learned,
        ..GlmRefit::new()
    };
    assert!(rows(5).fits(4, 6, false));
    assert!(!rows(6).fits(4, 6, false));
    assert!(rows(7).fits(4, 6, true), "a span holds what its span has");
    // A count of 0 reads as 1: the empty buffer fits, one row does not.
    assert!(rows(0).fits(4, 0, false));
    assert!(!rows(1).fits(4, 0, false));
    let done = GlmRefit {
        done: true,
        ..GlmRefit::new()
    };
    assert!(done.fits(4, 6, false));
}

/// `(A + r I) x = b` by the Cholesky factor: `A = [[4, 2], [2, 3]]`, `b =
/// (2, 1)` gives `x = (1/2, 0)`, and at `r = 1`, `A + I = [[5, 2], [2, 4]]`
/// of determinant 16, `x = (6, 1)/16`. A pivot that is 0 (a singular `A`),
/// or infinite, is no solve.
#[test]
fn the_ridged_solve_is_the_systems_and_refuses_a_pivot_that_is_not_a_number() {
    let (mut l, mut x) = (Vec::new(), Vec::new());
    let a = [4.0, 2.0, 3.0];
    assert!(solve_ridged(&a, 0.0, &[2.0, 1.0], 2, &mut l, &mut x));
    assert!((x[0] - 0.5).abs() < 1e-15 && x[1].abs() < 1e-15, "{x:?}");
    assert!(solve_ridged(&a, 1.0, &[2.0, 1.0], 2, &mut l, &mut x));
    assert!((x[0] - 0.375).abs() < 1e-15 && (x[1] - 0.0625).abs() < 1e-15);
    // Singular: the second pivot is 1 − 1 = 0.
    assert!(!solve_ridged(
        &[1.0, 1.0, 1.0],
        0.0,
        &[1.0, 1.0],
        2,
        &mut l,
        &mut x
    ));
    // An infinite pivot would give a factor whose solve is 0, finite.
    assert!(!solve_ridged(
        &[f64::INFINITY],
        0.0,
        &[1.0],
        1,
        &mut l,
        &mut x
    ));
    assert!(!solve_ridged(&[0.0], 0.0, &[1.0], 1, &mut l, &mut x));
}

fn two_target_model() -> Glm {
    Glm::new(GlmCfg {
        n_targets: 2,
        decay: Decay::Lam(0.5),
        refit_rows: 0,
        step_cap: 1.0,
        ..cfg_of(GlmFamily::Poisson, 1, 0)
    })
    .unwrap()
}

/// Each target's own weight is `w ← λ w + v` on the rows that carried it in
/// the support, and the trait method gives the same list.
#[test]
fn a_targets_weight_is_the_decayed_weight_of_its_rows() {
    let mut m = two_target_model();
    assert_eq!(m.target_weights(), &[0.0, 0.0]);
    m.step(&[0.3], &[Some(1.0), None], 1.0, 2.0);
    assert_eq!(m.target_weights(), &[2.0, 0.0]);
    // A count of -1 is outside the support: it only ages the first.
    m.step(&[0.3], &[Some(-1.0), Some(3.0)], 1.0, 0.5);
    assert_eq!(m.target_weights(), &[1.0, 0.5]);
    m.step(&[0.3], &[Some(2.0), Some(0.0)], 1.0, 1.0);
    assert_eq!(m.target_weights(), &[1.5, 1.25]);
    let mut out = vec![9.0];
    assert!(m.target_n_eff_into(&mut out));
    assert_eq!(out, vec![1.5, 1.25]);
}

/// A solve that fails is counted, and the count is the accessor's: a
/// ridge of `1e-300` is lost beside a first row's `G = x̃x̃'`, rank 1, so
/// its second pivot is `1 + 1e-300 − 1 = 0`; the second row, at another
/// `x`, makes `G` definite and fails nothing.
#[test]
fn a_solve_that_fails_in_a_learned_row_is_counted() {
    let mut m = Glm::new(GlmCfg {
        ridge: 1e-300,
        refit_rows: 0,
        step_cap: 1.0,
        ..cfg_of(GlmFamily::Poisson, 1, 0)
    })
    .unwrap();
    assert_eq!(m.solve_failures(), 0);
    m.step(&[1.0], &[Some(1.0)], 1.0, 1.0);
    assert_eq!(m.solve_failures(), 1);
    m.step(&[2.0], &[Some(3.0)], 1.0, 1.0);
    assert_eq!(m.solve_failures(), 1);
    m.step(&[1.0], &[Some(1.0)], 1.0, 1.0);
    assert_eq!(m.solve_failures(), 1, "G is definite from here on");
}

/// A refit that cannot solve its first pass (three rows at one `x`, a
/// ridge of `1e-300`) is failed and counted, after the failures the rows'
/// own solves already counted (so a count that multiplies shows). The
/// weights 1, 3 and 12 make `G` 1, 4 and 16 times `x̃x̃'`, and the targets
/// all 1 keep `W = 1`, so each pivot is a perfect square and the second is
/// exactly 0, not rounding's.
#[test]
fn a_refit_that_cannot_solve_is_failed_and_counted() {
    let mut m = Glm::new(GlmCfg {
        ridge: 1e-300,
        ..cfg_of(GlmFamily::Poisson, 1, 3)
    })
    .unwrap();
    // The refit is the third row's: it adds one to that row's own failure.
    for (w, failures) in [(1.0, 1), (3.0, 2), (12.0, 4)] {
        m.step(&[1.0], &[Some(1.0)], 1.0, w);
        assert_eq!(m.solve_failures(), failures);
    }
    let st = m.refit_status()[0];
    assert!(st.done && st.failed, "{st:?}");
}

/// The bytes the refit buffers can hold: `8 · targets · refit_rows ·
/// (n_features + 2)`, at features other than 2 (where `n + 2` is `2 n`).
#[test]
fn the_refit_reservation_is_the_buffers_bytes() {
    let m = Glm::new(GlmCfg {
        n_targets: 2,
        ..cfg_of(GlmFamily::Poisson, 3, 10)
    })
    .unwrap();
    assert_eq!(m.refit_reservation(), 2 * 8 * 10 * 5);
    let m = Glm::new(GlmCfg {
        n_targets: 3,
        ..cfg_of(GlmFamily::Poisson, 1, 7)
    })
    .unwrap();
    assert_eq!(m.refit_reservation(), 3 * 8 * 7 * 3);
}

/// After the refit the sums are those of its rows at the fit: `G = Σ v W(η)
/// x̃x̃'`, `W` the family's Fisher weight at the row's `η` under the
/// fitted coefficients, written out from the family's own `working`.
#[test]
fn the_refit_rebuilds_the_gram_from_the_rows_at_the_fit() {
    let n = 40;
    let c = cfg_of(GlmFamily::Poisson, 3, n as u32);
    let rows = stream_of(GlmFamily::Poisson, 4, n, 21);
    let v = |i: usize| 0.5 + 0.25 * (i % 4) as f64;
    let mut m = Glm::new(c.clone()).unwrap();
    for (i, (x, y)) in rows.iter().enumerate() {
        m.step(x, &[Some(*y)], 1.0, v(i));
    }
    assert!(m.refit_status()[0].done);
    let (b, fam, k) = (m.coefficients()[0].clone(), c.family(), 4);
    let mut want = vec![0.0; k * (k + 1) / 2];
    for (i, (x, y)) in rows.iter().enumerate() {
        let wk = fam.working(dot_aug(&b, x, true), *y).unwrap();
        let xt: Vec<f64> = std::iter::once(1.0).chain(x.iter().copied()).collect();
        for a in 0..k {
            for z in 0..=a {
                want[at(a, z)] += v(i) * wk.w * xt[a] * xt[z];
            }
        }
    }
    let got = &m.gram[0];
    let size = want.iter().fold(0.0_f64, |s, t| s.max(t.abs()));
    assert!(
        max_gap(got, &want) <= 1e-12 * size,
        "{:e}: {got:?} vs {want:?}",
        max_gap(got, &want)
    );
}

fn max_gap(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

/// The penalized objective `Σ v d(y, μ)/2 + R/2 |b|²`, the Poisson
/// deviance `2 (y ln(y/μ) − (y − μ))` written out: two rows at `x = ±1`,
/// `b = (0.2, 0.5)`, `R = 2`.
#[test]
fn the_objective_is_the_deviance_and_the_ridge_penalty() {
    let c = GlmCfg {
        ridge: 2.0,
        ..cfg_of(GlmFamily::Poisson, 1, 10)
    };
    let rows = [1.0, 2.0, 0.5, -1.0, 0.0, 2.0];
    let b = [0.2, 0.5];
    let (mu1, mu2) = ((0.2_f64 + 0.5).exp(), (0.2_f64 - 0.5).exp());
    let d1 = 2.0 * (2.0 * (2.0 / mu1).ln() - (2.0 - mu1));
    let d2 = 2.0 * mu2;
    let want = 0.5 * 0.5 * d1 + 0.5 * 2.0 * d2 + 0.5 * 2.0 * (0.2 * 0.2 + 0.5 * 0.5);
    let got = objective(&c, &c.family(), &rows, 1, &b);
    assert!((got - want).abs() < 1e-14 * want, "{got} vs {want}");
    // A row whose `η` gives no mean makes it infinite.
    let inverse = GlmCfg {
        link: GlmLink::Inverse,
        family: GlmFamily::Gamma,
        ..c
    };
    assert!(objective(&inverse, &inverse.family(), &rows, 1, &[-1.0, 0.0]).is_infinite());
}

/// The refit's start is the intercept at the link of the weighted mean,
/// from the buffer's rows: where the start is the fit, a pass finds nothing
/// to move. Features all 0 and a ridge of `1e-12` leave the intercept
/// `ln ȳ`; its first step is below `1e-11`, its second at the rounding.
#[test]
fn the_start_is_the_links_value_at_the_buffers_mean() {
    let c = GlmCfg {
        ridge: 1e-12,
        ..cfg_of(GlmFamily::Poisson, 3, 10)
    };
    let ys = [1.0, 4.0, 2.0, 0.0, 3.0, 6.0, 2.0, 1.0];
    let ws = [1.0, 2.0, 1.0, 1.0, 0.5, 1.0, 1.0, 2.0];
    let mut rows = Vec::new();
    for (y, w) in ys.iter().zip(ws) {
        rows.extend_from_slice(&[0.0, 0.0, 0.0, *y, w]);
    }
    let (b, passes, converged) = refit_fit(&c, &rows, 3).unwrap();
    let (sv, svy): (f64, f64) = ys
        .iter()
        .zip(ws)
        .fold((0.0, 0.0), |(s, t), (y, w)| (s + w, t + w * y));
    assert!(converged && passes <= 2, "{passes} passes");
    assert!((b[0] - (svy / sv).ln()).abs() < 1e-9, "{b:?}");
    assert!(b[1..].iter().all(|v| v.abs() < 1e-12), "{b:?}");
}

/// Rows of weight 0 carry no mean: the start is 0, which is the fit
/// (nothing but the ridge), found in the first pass.
#[test]
fn a_buffer_of_zero_weights_starts_and_stays_at_zero() {
    let c = cfg_of(GlmFamily::Poisson, 1, 10);
    let rows = buffer(1, &[1.0, 3.0, 2.0], &[0.0, 0.0, 0.0]);
    let (b, passes, converged) = refit_fit(&c, &rows, 1).unwrap();
    assert!(converged);
    assert_eq!((passes, b), (1, vec![0.0, 0.0]));
}

/// Without an intercept the start is 0 for every coefficient, however the
/// targets run: a Poisson fit on `x = ±1, ±2` with targets 5, 2, 2, 5 has
/// `Σ (y − 1) x = 0`, so `β = 0` is the fit and the first pass moves
/// nothing (the mean 3.5 would put `ln 3.5` in the slope).
#[test]
fn without_an_intercept_the_start_is_zero() {
    let c = GlmCfg {
        fit_intercept: false,
        ..cfg_of(GlmFamily::Poisson, 1, 10)
    };
    let rows = [-2.0, 5.0, 1.0, -1.0, 2.0, 1.0, 1.0, 2.0, 1.0, 2.0, 5.0, 1.0];
    let (b, passes, converged) = refit_fit(&c, &rows, 1).unwrap();
    assert!(converged);
    assert_eq!((passes, b), (1, vec![0.0]));
}

/// A buffered row's weight is aged by every row after it: each row's
/// `λ` multiplies the weights already held, at features other than 2.
#[test]
fn the_buffers_weights_decay_row_by_row() {
    let lam: f64 = 0.9;
    for nf in [1, 3] {
        let mut m = Glm::new(GlmCfg {
            decay: Decay::Lam(lam),
            ..cfg_of(GlmFamily::Poisson, nf, 50)
        })
        .unwrap();
        let w = [1.0, 2.0, 0.5, 1.5, 1.0];
        for (i, wi) in w.iter().enumerate() {
            let x: Vec<f64> = (0..nf).map(|j| 0.1 * (i + j) as f64).collect();
            m.step(&x, &[Some(1.0 + i as f64)], 1.0, *wi);
        }
        let held = &m.refit[0].rows;
        assert_eq!(held.len(), 5 * (nf + 2));
        for (i, wi) in w.iter().enumerate() {
            let want = wi * lam.powi((w.len() - 1 - i) as i32);
            let got = held[i * (nf + 2) + nf + 1];
            assert!((got - want).abs() < 1e-15, "nf {nf} row {i}: {got} {want}");
            assert_eq!(held[i * (nf + 2) + nf], 1.0 + i as f64, "the target");
        }
    }
}

/// A state saved while a refit's buffer is filling restores, at features
/// other than 2, where the row's width `n + 2` and `2 n` are not the same.
#[test]
fn a_state_saved_mid_buffer_restores_at_any_width() {
    for nf in [1, 3] {
        let rows = stream_of(GlmFamily::Poisson, nf + 1, 60, 3);
        let mut m = Glm::new(cfg_of(GlmFamily::Poisson, nf, 100)).unwrap();
        for (x, y) in &rows[..40] {
            m.step(x, &[Some(*y)], 1.0, 1.0);
        }
        assert_eq!(m.refit[0].learned, 40);
        let back = Glm::restore(&m.state()).unwrap();
        assert_eq!(back, m, "nf {nf}");
    }
}

/// A prediction under a floor is the model's with that floor, and the
/// model's own floor is put back.
#[test]
fn a_prediction_under_a_floor_is_the_models_with_that_floor() {
    let rows = stream_of(GlmFamily::Poisson, 3, 30, 5);
    let mut m = Glm::new(GlmCfg {
        min_weight: 1e9,
        ..cfg_of(GlmFamily::Poisson, 2, 0)
    })
    .unwrap();
    for (x, y) in &rows {
        m.step(x, &[Some(*y)], 1.0, 1.0);
    }
    let x = [0.2, -0.4];
    assert!(m.predict(&x, 1.0).pred[0].is_nan());
    let open = m.predict_with_floor(&x, &[None], 1.0, 0.0);
    assert!(open.pred[0].is_finite() && open.pred[0] > 0.0, "{open:?}");
    let mut twin = m.clone();
    twin.cfg.min_weight = 0.0;
    assert_eq!(open, twin.predict(&x, 1.0));
    let shut = m.predict_with_floor(&x, &[None], 1.0, 1e12);
    assert!(shut.pred[0].is_nan());
    assert_eq!(m.cfg.min_weight, 1e9, "the model's own floor is back");
}

/// A refit that runs its 100 passes without settling is counted as a
/// failure and says so: Fisher scoring under the Gamma's log link
/// converges linearly, at a rate that rises with the targets' spread, and
/// targets whose logarithm has a standard deviation of 5 (13 orders of
/// magnitude between the extremes) take more than 100 passes. The fit it
/// leaves is the pass cap's, finite, not an error.
#[test]
fn a_refit_that_runs_to_its_pass_cap_is_counted_as_a_failure() {
    let mut s = 12345u64;
    let mut lcg = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    };
    let mut normal = || {
        let (u, v) = (lcg(), lcg());
        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
    };
    let n = 200;
    let mut m = Glm::new(cfg_of(GlmFamily::Gamma, 3, n)).unwrap();
    for _ in 0..n {
        let x: Vec<f64> = (0..3).map(|_| normal()).collect();
        let y = (5.0 * normal()).exp();
        m.step(&x, &[Some(y)], 1.0, 1.0);
    }
    let st = m.refit_status()[0];
    assert!(st.done && !st.failed, "{st:?}");
    assert_eq!(st.passes, GLM_REFIT_PASSES);
    assert_eq!(m.solve_failures(), 1);
    assert!(m.coefficients()[0].iter().all(|b| b.is_finite()));
}

/// A step that leaves the link's range is halved until it does not: the
/// Gamma family under its inverse link (a mean only where `η > 0`) on
/// features of spread 2 and every tenth target 50 times the others', whose
/// first full step takes `η` below 0 for some row, and which halves it
/// once. The fit it ends at is the penalized objective's optimum, whose
/// score `Σ v W (z − η) x̃ − R b` is 0: it is held to 1e-9 of the score's
/// terms (measured 1e-12), where a line search that took no step left the
/// start's, which is of their size.
#[test]
fn a_step_out_of_the_links_range_is_halved_to_the_fit() {
    let (nf, n) = (3, 100);
    let c = GlmCfg {
        link: GlmLink::Inverse,
        ..cfg_of(GlmFamily::Gamma, nf, n as u32)
    };
    let mut buf = Vec::new();
    for (i, (x, y)) in stream_of(GlmFamily::Poisson, 4, n, 7).iter().enumerate() {
        buf.extend(x.iter().map(|v| 2.0 * v));
        buf.push(if i % 10 == 0 { 50.0 * y + 1.0 } else { y + 1.0 });
        buf.push(1.0);
    }
    let (b, passes, converged) = refit_fit(&c, &buf, nf).unwrap();
    assert!(converged && passes < GLM_REFIT_PASSES, "{passes}");
    let fam = c.family();
    let mut score = vec![0.0; 4];
    let mut terms = 0.0_f64;
    for row in buf.chunks_exact(nf + 2) {
        let wk = fam.working(dot_aug(&b, &row[..nf], true), row[nf]).unwrap();
        let xt = [1.0, row[0], row[1], row[2]];
        for (s, t) in score.iter_mut().zip(xt) {
            *s += row[nf + 1] * wk.w * (wk.z - wk.eta) * t;
            terms = terms.max((row[nf + 1] * wk.w * (wk.z - wk.eta) * t).abs());
        }
    }
    for (s, bi) in score.iter_mut().zip(&b) {
        *s -= c.ridge * bi;
    }
    assert!(
        score.iter().all(|s| s.abs() <= 1e-9 * terms),
        "score {score:?} against terms of {terms}"
    );
}

// ---- mutants-w11: the refit over a span's cap, its bytes and its flag ----

const S: i128 = 1_000_000_000;

fn spanned(nf: usize, targets: usize) -> Glm {
    let mut c = cfg_of(GlmFamily::Poisson, nf, 0);
    c.n_targets = targets;
    c.refit_clock = Some(1e12);
    Glm::new(c).unwrap()
}

fn stamp_bytes() -> u64 {
    std::mem::size_of::<Option<Stamp>>() as u64
}

/// The refit buffers' bytes: a stamp a target, and `n_features + 2` numbers
/// of 8 bytes a row held and a row to come, each target's.
#[test]
fn the_refit_buffers_bytes_are_counted_by_target_row_and_width() {
    let mut m = spanned(3, 2);
    assert_eq!(m.refit_held_bytes(0), 2 * stamp_bytes());
    assert_eq!(m.refit_held_bytes(5), 8 * 5 * (3 + 2) + 2 * stamp_bytes());
    let data = stream_of(GlmFamily::Poisson, 4, 4, 7);
    for (i, (x, y)) in data.iter().enumerate() {
        m.stamp_next(Stamp::Ns(i as i128 * S));
        m.step(x, &[Some(*y), Some(*y)], 1.0, 1.0);
    }
    // Four rows in each of two buffers.
    assert_eq!(
        m.refit_held_bytes(1),
        8 * (2 * 4 * (3 + 2) + (3 + 2)) + 2 * stamp_bytes()
    );
}

/// The warning's flag is state: saved after the warning, it restores raised.
#[test]
fn a_warned_flag_survives_a_save() {
    let mut c = cfg_of(GlmFamily::Poisson, 2, 0);
    c.refit_clock = Some(1e12);
    c.max_state_bytes = Some(2000);
    let mut m = Glm::new(c).unwrap();
    for (i, (x, y)) in stream_of(GlmFamily::Poisson, 3, 40, 5).iter().enumerate() {
        m.stamp_next(Stamp::Ns(i as i128 * S));
        m.step(x, &[Some(*y)], 1.0, 1.0);
    }
    assert!(m.warned && m.over.is_none());
    let bytes = rmp_serde::to_vec_named(&m.state()).unwrap();
    let key = b"\xa6warned";
    assert!(bytes.windows(key.len()).any(|w| w == key));
    let back = Glm::restore(&rmp_serde::from_slice(&bytes).unwrap()).unwrap();
    assert!(back.warned);
}

/// The cap is asked of a row that would be learned into a refit still
/// waiting, on a span, and of no other.
#[test]
fn the_refit_cap_is_asked_of_learned_rows_of_open_targets_on_a_span_only() {
    let data = stream_of(GlmFamily::Poisson, 3, 30, 9);
    let (x, y) = &data[0];
    // A row of no weight takes no room: a cap of the stamps alone holds it.
    let mut m = spanned(2, 1);
    m.cfg.max_state_bytes = Some(stamp_bytes());
    m.stamp_next(Stamp::Ns(0));
    m.step(x, &[Some(*y)], 1.0, 0.0);
    assert_eq!(m.state_over_cap(), None);
    // A row for a target it does not carry takes none either.
    let mut m = spanned(2, 1);
    m.cfg.max_state_bytes = Some(stamp_bytes() + 16);
    m.stamp_next(Stamp::Ns(0));
    m.step(x, &[None], 1.0, 1.0);
    assert_eq!(m.state_over_cap(), None);
    // Nor does a row once the refit is done: its buffer is empty.
    let mut m = spanned(2, 1);
    m.cfg.refit_clock = Some(3.0);
    for (i, (x, y)) in data[..8].iter().enumerate() {
        m.stamp_next(Stamp::Ns(i as i128 * S));
        m.step(x, &[Some(*y)], 1.0, 1.0);
    }
    assert!(m.refit_status()[0].done);
    m.cfg.max_state_bytes = Some(stamp_bytes() + 16);
    m.stamp_next(Stamp::Ns(9 * S));
    m.step(x, &[Some(*y)], 1.0, 1.0);
    assert_eq!(m.state_over_cap(), None);
    // And a model with no span checks no cap: it learns on as a free one.
    let c = cfg_of(GlmFamily::Poisson, 2, 6);
    let (mut capped, mut free) = (Glm::new(c.clone()).unwrap(), Glm::new(c.clone()).unwrap());
    capped.cfg.max_state_bytes = Some(1);
    for (x, y) in &data {
        let (a, b) = (
            capped.step(x, &[Some(*y)], 1.0, 1.0),
            free.step(x, &[Some(*y)], 1.0, 1.0),
        );
        assert_eq!(
            a.pred.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            b.pred.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }
    assert_eq!(capped.state_over_cap(), None);
}

/// A row refused at the cap is predicted by the fit's mean where the
/// target's weight has reached `min_weight`, and by nothing before.
#[test]
fn a_row_refused_at_the_cap_is_predicted_where_the_weight_has_reached_the_floor() {
    let data = stream_of(GlmFamily::Poisson, 3, 30, 11);
    let mut c = cfg_of(GlmFamily::Poisson, 2, 0);
    c.refit_clock = Some(1e12);
    c.min_weight = 2.0;
    let mut m = Glm::new(c).unwrap();
    // Room for five rows and the stamp.
    m.cfg.max_state_bytes = Some(stamp_bytes() + 5 * 32);
    for (i, (x, y)) in data[..5].iter().enumerate() {
        m.stamp_next(Stamp::Ns(i as i128 * S));
        m.step(x, &[Some(*y)], 1.0, 1.0);
    }
    assert_eq!(m.state_over_cap(), None);
    m.stamp_next(Stamp::Ns(5 * S));
    let refused = m.step(&data[5].0, &[Some(data[5].1)], 1.0, 1.0);
    assert!(m.state_over_cap().is_some());
    assert!(refused.pred[0].is_finite(), "{:?}", refused.pred);
    // Before the floor: nothing to say.
    let mut c = cfg_of(GlmFamily::Poisson, 2, 0);
    c.refit_clock = Some(1e12);
    c.min_weight = 2.0;
    let mut m = Glm::new(c).unwrap();
    m.cfg.max_state_bytes = Some(stamp_bytes());
    m.stamp_next(Stamp::Ns(0));
    let first = m.step(&data[0].0, &[Some(data[0].1)], 1.0, 1.0);
    assert!(m.state_over_cap().is_some());
    assert!(first.pred[0].is_nan(), "{:?}", first.pred);
}
