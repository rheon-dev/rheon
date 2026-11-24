//! The 2026-11-23 review's findings on `expectile` (PLAN 578b): each test
//! either failed on 51671e2e for the reason the review gives, and holds the
//! fix, or shows where the claim does not hold. The finding's ID leads each
//! test's doc.

use super::tests::{cfg_of, stream_of};
use super::*;
use crate::oracle;

fn z_of(x: &[f64]) -> Vec<f64> {
    std::iter::once(1.0).chain(x.iter().copied()).collect()
}

/// The penalized WLS fit `(Σ c z z' + r V D) β = Σ c z y` by faer's LU on
/// the sums form, `D` the slopes' diagonal and `V` the rows' weight: no
/// arithmetic shared with the model's Cholesky or its centred solves.
fn lu_ridged(rows: &[(Vec<f64>, f64, f64, f64)], ridge: f64) -> Vec<f64> {
    let p = rows[0].0.len();
    let (mut g, mut b) = (vec![0.0; p * p], vec![0.0; p]);
    let mut v = 0.0;
    for (z, y, w, a) in rows {
        let c = w * a;
        v += w;
        for i in 0..p {
            for j in 0..p {
                g[i * p + j] += c * z[i] * z[j];
            }
            b[i] += c * z[i] * y;
        }
    }
    for i in 1..p {
        g[i * p + i] += ridge * v;
    }
    oracle::solve(&g, &b)
}

/// The penalized expectile regression `argmin Σ v a(r) r² + r V |β_s|²` by
/// its definition: the penalized WLS fit (by the LU) whose sides agree with
/// its own residuals, iterated from least squares until no side changes.
fn lu_expectile(rows: &[(Vec<f64>, f64, f64)], tau: f64, ridge: f64) -> Vec<f64> {
    let half: Vec<_> = rows
        .iter()
        .map(|(z, y, v)| (z.clone(), *y, *v, 0.5))
        .collect();
    let mut b = lu_ridged(&half, ridge);
    for _ in 0..100 {
        let sided: Vec<_> = rows
            .iter()
            .map(|(z, y, v)| (z.clone(), *y, *v, side_weight(y - dot(z, &b), tau)))
            .collect();
        let nb = lu_ridged(&sided, ridge);
        let same = rows
            .iter()
            .all(|(z, y, _)| (y - dot(z, &nb) > 0.0) == (y - dot(z, &b) > 0.0));
        // Rows the fit interpolates trade sides at rounding with no effect.
        let size = b.iter().fold(1.0_f64, |m, v| m.max(v.abs()));
        let still = nb
            .iter()
            .zip(&b)
            .all(|(p, q)| (p - q).abs() <= 1e-13 * size);
        b = nb;
        if same || still {
            return b;
        }
    }
    panic!("the oracle's fixed point did not settle");
}

/// The `τ`-expectile of a sample, by its definition (Newey and Powell
/// 1987): the `e` with `Σ v |τ − 1{y ≤ e}| (y − e) = 0`, found by
/// bisection, which reads nothing of the model.
fn sample_expectile(ys: &[f64], tau: f64) -> f64 {
    let score = |e: f64| -> f64 { ys.iter().map(|y| side_weight(y - e, tau) * (y - e)).sum() };
    let (mut lo, mut hi) = ys
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &y| {
            (a.min(y), b.max(y))
        });
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if score(mid) > 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol * (1.0 + b.abs())
}

/// EXP-1, case 1: a clock window every row of which has left by the next
/// row holds no rows, so the fit is gone at every level, as it was at
/// `τ = 1/2`. At `τ = 0.9` the refit handed back the fit of the expired
/// rows (51671e2e). The other side: a row by which only some rows have
/// left reads the exact fit of the rows left.
#[test]
fn exp_1_an_emptied_clock_window_reads_no_fit_at_every_level() {
    for tau in [0.5, 0.9] {
        let mut c = cfg_of(1, tau);
        c.window_clock = Some(5.0);
        c.ridge = 1e-6;
        let rows = stream_of(1, 10, 3);
        let fed = |m: &mut Expectile| {
            for (i, (x, y)) in rows.iter().enumerate() {
                m.stamp_next(Stamp::from(i as f64 * 0.5));
                m.step(x, &[Some(*y)], 0.5, 1.0);
            }
        };
        let x = [0.3];
        // Every row left by 14.
        let mut m = Expectile::new(c.clone()).unwrap();
        fed(&mut m);
        let ahead = m.predict(&x, 14.0 - 4.5);
        let failures = m.solve_failures();
        m.stamp_next(Stamp::from(14.0));
        let got = m.step(&x, &[Some(1.0)], 14.0 - 4.5, 1.0);
        // The fit it had is lost, which counts, at every level.
        assert_eq!(m.solve_failures(), failures + 1, "tau {tau}");
        assert!(
            got.pred[0].is_nan() && ahead.pred[0].is_nan(),
            "tau {tau}: an empty window read {} (predict {})",
            got.pred[0],
            ahead.pred[0]
        );
        // By 7, the rows at 2.0 and before have left (Right drops a row 5 old).
        let mut m = Expectile::new(c).unwrap();
        fed(&mut m);
        m.stamp_next(Stamp::from(7.0));
        let got = m.step(&x, &[Some(1.0)], 7.0 - 4.5, 1.0);
        let kept: Vec<_> = rows
            .iter()
            .enumerate()
            .filter(|(i, _)| *i as f64 * 0.5 > 2.0)
            .map(|(_, (x, y))| (z_of(x), *y, 1.0))
            .collect();
        let want = dot(&z_of(&x), &lu_expectile(&kept, tau, 1e-6));
        assert!(close(got.pred[0], want, 1e-9), "{} vs {want}", got.pred[0]);
    }
}

/// EXP-1, case 2: a row window whose rows carry the target on fewer rows
/// than coefficients determines no fit without a ridge, at every level. At
/// `τ = 0.9` the fit of the rows that left was kept (51671e2e). The other
/// side: a row per coefficient is the exact fit of those rows.
#[test]
fn exp_1_a_window_with_too_few_targets_reads_no_fit_at_every_level() {
    for tau in [0.5, 0.9] {
        for present in [2usize, 3] {
            let mut c = cfg_of(2, tau);
            c.window_rows = Some(10);
            c.min_weight = 1.0;
            let mut m = Expectile::new(c).unwrap();
            let rows = stream_of(2, 10 + 10 - present, 4);
            // Ten rows with the target, then rows without it, so the window
            // of the last ten holds `present` rows that carry it.
            for (i, (x, y)) in rows.iter().enumerate() {
                let y = (i < 10).then_some(*y);
                m.step(x, &[y], 1.0, 1.0);
            }
            let x = [0.2, -0.4];
            let got = m.step(&x, &[None], 1.0, 1.0).pred[0];
            if present < 3 {
                assert!(got.is_nan(), "tau {tau}, {present} rows: {got}");
            } else {
                let kept: Vec<_> = rows[10 - present..10]
                    .iter()
                    .map(|(x, y)| (z_of(x), *y, 1.0))
                    .collect();
                let want = dot(&z_of(&x), &lu_expectile(&kept, tau, 0.0));
                assert!(close(got, want, 1e-9), "tau {tau}: {got} vs {want}");
            }
        }
    }
}

/// EXP-1, case 3, and EXP-2 (1): a feature constant at -1500 under the
/// default ridge. The one pass solves it about the features' means; the
/// exact refit, of a window or of the first rows, had no such fallback, so
/// a window read NaN on every row and the refit failed (51671e2e). Each
/// now reads the `τ`-expectile of its rows' targets, the constant feature
/// carrying nothing.
#[test]
fn exp_2_a_constant_feature_is_fitted_by_the_window_and_the_refit() {
    let rows: Vec<(Vec<f64>, f64)> = stream_of(1, 40, 5)
        .into_iter()
        .map(|(_, y)| (vec![-1500.0], y))
        .collect();
    for tau in [0.5, 0.9] {
        let mut c = cfg_of(1, tau);
        c.ridge = 1e-6;
        c.window_rows = Some(20);
        let mut m = Expectile::new(c).unwrap();
        for (i, (x, y)) in rows.iter().enumerate() {
            let got = m.step(x, &[Some(*y)], 1.0, 1.0).pred[0];
            if i >= 2 {
                let ys: Vec<f64> = rows[i.saturating_sub(20)..i].iter().map(|r| r.1).collect();
                let want = sample_expectile(&ys, tau);
                assert!(
                    close(got, want, 1e-9),
                    "tau {tau}, row {i}: the window read {got}, the sample's expectile is {want}"
                );
            }
        }
        let mut c = cfg_of(1, tau);
        c.ridge = 1e-6;
        c.refit_rows = 30;
        let mut m = Expectile::new(c).unwrap();
        for (x, y) in &rows[..30] {
            m.step(x, &[Some(*y)], 1.0, 1.0);
        }
        let st = m.refit_status()[0];
        assert!(st.done && !st.failed, "tau {tau}: {st:?}");
        let ys: Vec<f64> = rows[..30].iter().map(|r| r.1).collect();
        let got = m.predict(&[-1500.0], 1.0).pred[0];
        let want = sample_expectile(&ys, tau);
        assert!(close(got, want, 1e-9), "tau {tau}: refit {got} vs {want}");
    }
}

/// EXP-2 (2), the window: a feature at a level of 1e6 that moves by 1e-3.
/// Its raw Gram's pivot, about 1e-18 of the level squared, is under the
/// floor, so the window read NaN on every row (51671e2e); about the rows'
/// own means it is well conditioned, and the window reads the exact fit,
/// the oracle's on the rows shifted by 1e6 (exactly, by Sterbenz), which
/// moves no fitted value.
#[test]
fn exp_2_a_feature_far_from_zero_is_fitted_by_the_window() {
    let raw = stream_of(1, 80, 6);
    let rows: Vec<(Vec<f64>, f64)> = raw
        .iter()
        .map(|(x, y)| (vec![1e6 + 1e-3 * x[0]], 3.0 + 2000.0 * 1e-3 * x[0] + y))
        .collect();
    for tau in [0.5, 0.7] {
        let mut c = cfg_of(1, tau);
        c.ridge = 1e-6;
        c.window_rows = Some(50);
        let mut m = Expectile::new(c).unwrap();
        for (i, (x, y)) in rows.iter().enumerate() {
            let got = m.step(x, &[Some(*y)], 1.0, 1.0).pred[0];
            if i < 5 {
                continue;
            }
            let kept: Vec<_> = rows[i.saturating_sub(50)..i]
                .iter()
                .map(|(x, y)| (vec![1.0, x[0] - 1e6], *y, 1.0))
                .collect();
            let b = lu_expectile(&kept, tau, 1e-6);
            let want = b[0] + b[1] * (x[0] - 1e6);
            // The model's coefficients are on the raw feature, its fitted
            // value the difference of two numbers of order 2000 * 1e6: that
            // cancellation is the parametrization's, not the solve's.
            let tol = 1e-12 * (b[1].abs() * 1e6 + want.abs());
            assert!(
                (got - want).abs() <= tol,
                "tau {tau}, row {i}: {got} vs {want}"
            );
        }
    }
}

/// EXP-2 (3): a feature that is always 0 beside one whose spread is 1e7.
/// Under the default ridge the system is positive definite, the dead
/// feature's pivot being the ridge alone, but that pivot is under 1e-12 of
/// the other feature's diagonal, and the floor read against the largest
/// diagonal refused every solve: no fit, ever (51671e2e). Read against its
/// own diagonal, the pivot is the whole of it. The one pass at `τ = 1/2` is
/// the ridged least-squares fit, the oracle's; the window, the oracle's
/// expectile of its rows.
#[test]
fn exp_2_a_dead_feature_beside_a_wide_one_is_fitted() {
    let raw = stream_of(1, 60, 7);
    let rows: Vec<(Vec<f64>, f64)> = raw
        .iter()
        .map(|(x, y)| (vec![3162.0 * x[0], 0.0], 1.0 + 2.0 * 3162.0 * x[0] + y))
        .collect();
    let ridge = 1e-6;
    let mut c = cfg_of(2, 0.5);
    c.ridge = ridge;
    let mut m = Expectile::new(c).unwrap();
    for (x, y) in &rows {
        m.step(x, &[Some(*y)], 1.0, 1.0);
    }
    let all: Vec<_> = rows.iter().map(|(x, y)| (z_of(x), *y, 1.0, 0.5)).collect();
    let want = lu_ridged(&all, ridge);
    let got = &m.coefficients()[0];
    for (g, w) in got.iter().zip(&want) {
        assert!(close(*g, *w, 1e-8), "{got:?} vs {want:?}");
    }
    for tau in [0.5, 0.8] {
        let mut c = cfg_of(2, tau);
        c.ridge = ridge;
        c.window_rows = Some(30);
        let mut m = Expectile::new(c).unwrap();
        for (i, (x, y)) in rows.iter().enumerate() {
            let got = m.step(x, &[Some(*y)], 1.0, 1.0).pred[0];
            if i < 3 {
                continue;
            }
            let kept: Vec<_> = rows[i.saturating_sub(30)..i]
                .iter()
                .map(|(x, y)| (z_of(x), *y, 1.0))
                .collect();
            let want = dot(&z_of(x), &lu_expectile(&kept, tau, ridge));
            assert!(
                close(got, want, 1e-8),
                "tau {tau}, row {i}: {got} vs {want}"
            );
        }
    }
}

/// EXP-3: on the row clock, a read `d` rows on keys its window as a clock
/// of numbers keys it `d` units on: the two models, one stamped on the row
/// clock and one with the same numbers, read the same bits for every `d`
/// (51671e2e read one row on whatever `d`). And a row of weight 0 on the
/// row clock reads as `predict` reads it, as on any clock.
#[test]
fn exp_3_a_read_on_the_row_clock_keys_its_window_d_rows_on() {
    let mk = || {
        let mut c = cfg_of(1, 0.8);
        c.window_clock = Some(6.0);
        c.ridge = 1e-6;
        Expectile::new(c).unwrap()
    };
    let (mut rows_m, mut num_m) = (mk(), mk());
    for (i, (x, y)) in stream_of(1, 20, 8).iter().enumerate() {
        let s = Stamp::from((i + 1) as f64);
        rows_m.stamp_next_on_rows(s);
        rows_m.step(x, &[Some(*y)], 1.0, 1.0);
        num_m.stamp_next(s);
        num_m.step(x, &[Some(*y)], 1.0, 1.0);
    }
    let x = [0.7];
    for d in [1.0, 3.0, 5.0, 7.0] {
        let (a, b) = (rows_m.predict(&x, d), num_m.predict(&x, d));
        assert_eq!(
            (a.pred[0].to_bits(), a.n_eff.to_bits()),
            (b.pred[0].to_bits(), b.n_eff.to_bits()),
            "d = {d}: the row clock read {a:?}, the number clock {b:?}"
        );
    }
    for d in [1.0, 4.0] {
        let mut zero = rows_m.clone();
        zero.stamp_next_on_rows(Stamp::from(21.0));
        let got = zero.step(&x, &[Some(0.0)], d, 0.0);
        let want = rows_m.predict(&x, d);
        assert_eq!(got.pred[0].to_bits(), want.pred[0].to_bits(), "d = {d}");
    }
}

/// EXP-4: a clock window's `min_weight` gate reads the weight it reports,
/// the window's before the row's own clock step (hard rule 8; `hawkes`
/// reads it so too). The rows at 0 to 4 weigh 5; at 8 only the row at 4
/// is left. The gate read 1 and withheld the row while the row reported
/// 5, past the floor of 3 (51671e2e): a null prediction no gate named.
/// Now the row reads the fit of the row left. The other side: a floor of 6
/// withholds it, and the weight reported says why.
#[test]
fn exp_4_the_gate_reads_the_weight_the_row_reports() {
    for floor in [3.0, 6.0] {
        let mut c = cfg_of(1, 0.7);
        c.window_clock = Some(5.0);
        c.min_weight = floor;
        c.ridge = 1e-6;
        let mut m = Expectile::new(c).unwrap();
        let rows = stream_of(1, 5, 9);
        for (i, (x, y)) in rows.iter().enumerate() {
            m.stamp_next(Stamp::from(i as f64));
            m.step(x, &[Some(*y)], 1.0, 1.0);
        }
        let mut tw = Vec::new();
        m.target_n_eff_into(&mut tw);
        let x = [0.1];
        let ahead = m.predict(&x, 4.0);
        m.stamp_next(Stamp::from(8.0));
        let got = m.step(&x, &[Some(0.0)], 4.0, 1.0);
        assert_eq!((tw[0], got.n_eff), (5.0, 5.0));
        assert_eq!(got.pred[0].to_bits(), ahead.pred[0].to_bits());
        if floor <= tw[0] {
            // One row left: the intercept alone fits it, the slope penalized.
            let want = lu_expectile(&[(z_of(&rows[4].0), rows[4].1, 1.0)], 0.7, 1e-6);
            let want = dot(&z_of(&x), &want);
            assert!(close(got.pred[0], want, 1e-9), "{} vs {want}", got.pred[0]);
        } else {
            assert!(got.pred[0].is_nan());
        }
    }
}

/// EXP-5 (1): a warm start whose loss overflows (`β_1 = -1e160` against
/// features of order 1: every residual squared is past the largest double).
/// The review read the overflow as a false convergence on the first pass;
/// it is not one: the first pass's loss is finite, its scale is the new
/// fit's, and the refit goes on to the same fixed point a cold start finds,
/// stationary to rounding. Not reproduced.
#[test]
fn exp_5_an_overflowing_warm_start_still_reaches_the_fixed_point() {
    let raw = stream_of(1, 30, 10);
    let mut fit = Fit::default();
    for (x, y) in &raw {
        fit.z.extend_from_slice(&[1.0, x[0]]);
        fit.y.push(*y);
        fit.v.push(1.0);
    }
    let cold = fit.refit(2, 0.8, None).unwrap();
    let warm = fit.refit(2, 0.8, Some(&[0.0, -1e160])).unwrap();
    assert!(cold.converged && warm.converged, "{cold:?} {warm:?}");
    for (a, b) in cold.beta.iter().zip(&warm.beta) {
        assert!(close(*a, *b, 1e-12), "{cold:?} vs {warm:?}");
    }
}

/// EXP-5 (2): a weight of `5e-324` at the side `1 − τ = 0.1` rounds its
/// asymmetric weight to 0, so the row teaches nothing, though it adds its
/// weight to `n_eff`. That weight is below every other row's last bit:
/// every prediction and coefficient after it is the stream's without it, to
/// the bit, and `n_eff` with it rounds to the same number once a row of
/// weight 1 is in it. Not a defect.
#[test]
fn exp_5_a_subnormal_weight_moves_no_output() {
    let mut c = cfg_of(1, 0.9);
    c.ridge = 1e-6;
    let rows = stream_of(1, 40, 11);
    let (mut with, mut without) = (
        Expectile::new(c.clone()).unwrap(),
        Expectile::new(c).unwrap(),
    );
    for (i, (x, y)) in rows.iter().enumerate() {
        if i == 20 {
            // A target far below the fit: its side is `1 − τ`.
            with.step(x, &[Some(-1e6)], 1.0, 5e-324);
            without.step(x, &[Some(-1e6)], 1.0, 0.0);
        }
        let (a, b) = (
            with.step(x, &[Some(*y)], 1.0, 1.0),
            without.step(x, &[Some(*y)], 1.0, 1.0),
        );
        assert_eq!(a.pred[0].to_bits(), b.pred[0].to_bits(), "row {i}");
        if i > 20 {
            assert_eq!(a.n_eff.to_bits(), b.n_eff.to_bits(), "row {i}");
        }
    }
}

/// GLM-EXP-LOW (5): a state whose refit over a span holds rows but no first
/// stamp, or that records a run stopped at the cap where no span or clock
/// window could have stopped it, is refused as damaged rather than run.
#[test]
fn low_5_a_state_missing_its_span_start_or_stopped_without_a_span_is_refused() {
    let mut c = cfg_of(1, 0.7);
    c.refit_clock = Some(50.0);
    let mut m = Expectile::new(c).unwrap();
    for (x, y) in stream_of(1, 5, 12) {
        m.step(&x, &[Some(y)], 1.0, 1.0);
    }
    assert!(Expectile::restore(&m.state()).is_ok());
    let mut bad = m.clone();
    bad.refit[0].first = None;
    assert!(Expectile::restore(&bad.state()).is_err());
    let mut m = Expectile::new(cfg_of(1, 0.7)).unwrap();
    for (x, y) in stream_of(1, 5, 12) {
        m.step(&x, &[Some(y)], 1.0, 1.0);
    }
    let mut bad = m.clone();
    bad.over = Some(1);
    assert!(Expectile::restore(&bad.state()).is_err());
}

/// EXP-2 (2), the one pass, a confirmed defect whose fix is the user's to
/// allow: a feature at `1e6 + 1e-3 N(0, 1)`. The one pass keeps the
/// mean-form Gram over `[1, x]`, whose centred block `E[x²] − E[x]²` is a
/// spread of 1e-6 read off a level of 1e12, every digit of it lost, so the
/// solve about the means fails or reads noise. Only a state that keeps the
/// features' centred moments (Welford's) fits it: a layout change. This
/// test fails once that lands, and is then turned into a plain test.
#[test]
#[should_panic(expected = "the one pass")]
fn exp_2_the_one_pass_fits_a_feature_far_from_zero() {
    let rows: Vec<(Vec<f64>, f64)> = stream_of(1, 200, 6)
        .iter()
        .map(|(x, y)| (vec![1e6 + 1e-3 * x[0]], 3.0 + 2000.0 * 1e-3 * x[0] + y))
        .collect();
    let mut c = cfg_of(1, 0.5);
    c.ridge = 1e-6;
    let mut m = Expectile::new(c).unwrap();
    for (x, y) in &rows {
        m.step(x, &[Some(*y)], 1.0, 1.0);
    }
    let shifted: Vec<_> = rows
        .iter()
        .map(|(x, y)| (vec![1.0, x[0] - 1e6], *y, 1.0, 0.5))
        .collect();
    let b = lu_ridged(&shifted, 1e-6);
    let x = 1e6 + 1e-3;
    let want = b[0] + b[1] * 1e-3;
    let got = m.predict(&[x], 1.0).pred[0];
    let tol = 1e-9 * (b[1].abs() * 1e6 + want.abs());
    assert!(
        (got - want).abs() <= tol,
        "the one pass reads {got} where the ridged least-squares fit reads {want}"
    );
}
