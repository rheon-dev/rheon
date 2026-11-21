//! The expectile model's unit tests: the recursion against its definition
//! and `crate::oracle`'s LU, the refit and the window against the
//! stationarity condition and a fixed point found by the LU, rows of weight
//! 0, the state's bytes and its shape.

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

/// `n` rows of `k` standard normal features and a target on a line with
/// skewed noise (an exponential less its mean), so the expectiles at
/// different levels differ by more than rounding.
fn stream(k: usize, n: usize, seed: u64) -> Vec<(Vec<f64>, f64)> {
    let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ 501;
    (0..n)
        .map(|_| {
            let x: Vec<f64> = (0..k).map(|_| normal(&mut s)).collect();
            let lin = 1.0
                + x.iter()
                    .enumerate()
                    .map(|(i, v)| (i as f64 - 0.5) * v)
                    .sum::<f64>();
            (x, lin + (-lcg(&mut s).ln() - 1.0) * 1.5)
        })
        .collect()
}

fn cfg(k: usize, tau: f64) -> ExpectileCfg {
    ExpectileCfg {
        n_features: k,
        n_targets: 1,
        fit_intercept: true,
        decay: Decay::Lam(1.0),
        expectile: tau,
        min_weight: 0.0,
        refit_rows: 0,
        window_rows: None,
        window_clock: None,
        closed: WindowClosed::Right,
        max_state_bytes: None,
    }
}

fn z_of(x: &[f64]) -> Vec<f64> {
    std::iter::once(1.0).chain(x.iter().copied()).collect()
}

/// The WLS fit of rows `(z, y, c)` by faer's LU on the sums form: no
/// arithmetic shared with the model's mean-form Cholesky.
fn lu_wls(rows: &[(Vec<f64>, f64, f64)]) -> Vec<f64> {
    let p = rows[0].0.len();
    let (mut g, mut b) = (vec![0.0; p * p], vec![0.0; p]);
    for (z, y, c) in rows {
        for i in 0..p {
            for j in 0..p {
                g[i * p + j] += c * z[i] * z[j];
            }
            b[i] += c * z[i] * y;
        }
    }
    oracle::solve(&g, &b)
}

/// The expectile regression by its definition: the WLS fit (by the LU)
/// whose sides agree with its own residuals, found by iterating from least
/// squares until no side changes.
fn lu_expectile(rows: &[(Vec<f64>, f64, f64)], tau: f64) -> Vec<f64> {
    let half: Vec<_> = rows
        .iter()
        .map(|(z, y, v)| (z.clone(), *y, 0.5 * v))
        .collect();
    let mut b = lu_wls(&half);
    for _ in 0..100 {
        let sided: Vec<_> = rows
            .iter()
            .map(|(z, y, v)| (z.clone(), *y, v * side_weight(y - dot(z, &b), tau)))
            .collect();
        let nb = lu_wls(&sided);
        let same = rows
            .iter()
            .all(|(z, y, _)| (y - dot(z, &nb) > 0.0) == (y - dot(z, &b) > 0.0));
        b = nb;
        if same {
            return b;
        }
    }
    panic!("the oracle's fixed point did not settle");
}

/// The largest entry of `Σ v a(r) r z` over `sqrt(Σ v y²) sqrt(Σ v z'z)`:
/// the stationarity condition, scaled.
fn stationarity(rows: &[(Vec<f64>, f64, f64)], b: &[f64], tau: f64) -> f64 {
    let p = b.len();
    let mut g = vec![0.0; p];
    let (mut sy, mut sz) = (0.0, 0.0);
    for (z, y, v) in rows {
        let r = y - dot(z, b);
        for i in 0..p {
            g[i] += v * side_weight(r, tau) * r * z[i];
        }
        sy += v * y * y;
        sz += v * dot(z, z);
    }
    g.iter().fold(0.0_f64, |m, v| m.max(v.abs())) / (sy.sqrt() * sz.sqrt())
}

fn rel(a: &[f64], b: &[f64]) -> f64 {
    let top = b.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    a.iter()
        .zip(b)
        .fold(0.0_f64, |m, (x, y)| m.max((x - y).abs()))
        / top
}

/// At `τ = 1/2` every side weighs 1/2, so one pass is the decayed weighted
/// least-squares fit, to rounding (the study's E3: 3.3e-15).
#[test]
fn at_one_half_one_pass_is_the_decayed_least_squares_fit() {
    for lam in [1.0, 0.99] {
        let mut c = cfg(3, 0.5);
        c.decay = Decay::Lam(lam);
        let mut m = Expectile::new(c).unwrap();
        let rows = stream(3, 400, 1);
        let mut kept = Vec::new();
        let mut s = 7u64;
        for (i, (x, y)) in rows.iter().enumerate() {
            let w = 0.5 + lcg(&mut s);
            m.step(x, &[Some(*y)], if i == 0 { 0.0 } else { 1.0 }, w);
            for r in kept.iter_mut() {
                let r: &mut (Vec<f64>, f64, f64) = r;
                r.2 *= lam;
            }
            kept.push((z_of(x), *y, w));
        }
        let want = lu_wls(&kept);
        let got = &m.coefficients()[0];
        assert!(
            rel(got, &want) < 1e-12,
            "lam {lam}: {got:?} against {want:?}"
        );
    }
}

/// A row's own update cannot move its residual across zero (the module
/// docs): the side judged before the row is the side after it.
#[test]
fn a_row_never_crosses_zero_on_its_own_update() {
    for tau in [0.1, 0.9] {
        let mut m = Expectile::new(cfg(2, tau)).unwrap();
        let mut checked = 0;
        for (x, y) in stream(2, 2000, 2) {
            let before = m.predict(&x, 1.0).pred[0];
            m.step(&x, &[Some(y)], 1.0, 1.0);
            let after = m.predict(&x, 1.0).pred[0];
            if before.is_finite() {
                let (rb, ra) = (y - before, y - after);
                assert!(ra == 0.0 || (rb > 0.0) == (ra > 0.0), "{rb} then {ra}");
                checked += 1;
            }
        }
        assert!(checked > 1900);
    }
}

/// The exact refit of a set of rows is the fit its definition names: the
/// LU's self-consistent WLS, to 1e-12, with its stationarity at rounding,
/// cold and warm, at levels from 0.05 to 0.95, with weights that include 0.
#[test]
fn the_refit_is_the_fixed_point_of_its_definition() {
    for (seed, tau) in [(3u64, 0.05), (4, 0.2), (5, 0.5), (6, 0.8), (7, 0.95)] {
        let mut s = seed;
        let rows: Vec<(Vec<f64>, f64, f64)> = stream(4, 300, seed)
            .into_iter()
            .map(|(x, y)| {
                let w = if lcg(&mut s) < 0.2 {
                    0.0
                } else {
                    lcg(&mut s) * 2.0
                };
                (z_of(&x), y, w)
            })
            .collect();
        let mut fit = Fit::default();
        for (z, y, v) in &rows {
            fit.z.extend_from_slice(z);
            fit.y.push(*y);
            fit.v.push(*v);
        }
        let want = lu_expectile(&rows, tau);
        let cold = fit.refit(5, tau, None).unwrap();
        assert!(cold.converged);
        assert!(
            rel(&cold.beta, &want) < 1e-12,
            "tau {tau}: {:?} {want:?}",
            cold.beta
        );
        assert!(stationarity(&rows, &cold.beta, tau) < 1e-12);
        let warm = fit.refit(5, tau, Some(&[0.3; 5])).unwrap();
        assert!(rel(&warm.beta, &want) < 1e-12);
        assert!(cold.passes <= 10, "{} passes", cold.passes);
    }
}

/// An exact fit -- targets on a line -- stops on the rounding floor or an
/// unchanged side pattern, never at the cap: rows whose residuals are at
/// rounding level trade sides with no effect on the fit (the study's
/// first stress run cycled for 40 passes without the floor).
#[test]
fn an_exact_fit_stops_without_reaching_the_cap() {
    let mut fit = Fit::default();
    for i in 0..40 {
        let x = f64::from(i);
        fit.z.extend_from_slice(&[1.0, x]);
        fit.y.push(3.0 + 2.0 * x);
        fit.v.push(1.0);
    }
    for tau in [0.01, 0.3, 0.99] {
        let got = fit.refit(2, tau, None).unwrap();
        assert!(got.converged && got.passes < 10, "{got:?}");
        assert!((got.beta[0] - 3.0).abs() < 1e-12 && (got.beta[1] - 2.0).abs() < 1e-12);
    }
}

/// Fewer rows than coefficients determine no fit: no refit, not a failure.
#[test]
fn too_few_rows_are_no_fit() {
    let mut fit = Fit::default();
    fit.z.extend_from_slice(&[1.0, 0.5, 0.2, 1.0, 1.0, 2.0]);
    fit.y.extend_from_slice(&[1.0, 2.0]);
    fit.v.extend_from_slice(&[1.0, 1.0]);
    assert!(fit.refit(3, 0.7, None).is_none());
}

/// The first rows' refit: once the buffer fills, the fit is the exact fit
/// of those rows at their decayed weights, and the sums it leaves solve to
/// it; the buffer is freed.
#[test]
fn the_first_rows_refit_is_the_exact_fit_of_its_rows() {
    for lam in [1.0, 0.995] {
        let mut c = cfg(3, 0.85);
        c.refit_rows = 150;
        c.decay = Decay::Lam(lam);
        let mut m = Expectile::new(c).unwrap();
        let mut kept: Vec<(Vec<f64>, f64, f64)> = Vec::new();
        for (x, y) in stream(3, 150, 8) {
            m.step(&x, &[Some(y)], 1.0, 1.0);
            for r in kept.iter_mut() {
                r.2 *= lam;
            }
            kept.push((z_of(&x), y, 1.0));
        }
        let st = &m.refit_status()[0];
        assert!(st.done && !st.failed && st.passes > 0, "{st:?}");
        assert!(m.refit[0].rows.is_empty());
        let want = lu_expectile(&kept, 0.85);
        let got = &m.coefficients()[0];
        assert!(rel(got, &want) < 1e-12, "{got:?} {want:?}");
        let mut x = Vec::new();
        assert!(chol_solve(
            &m.gram[0],
            &m.cross[0],
            4,
            &mut Vec::new(),
            &mut x
        ));
        assert!(rel(&x, &want) < 1e-12);
    }
}

/// A window of `N` rows: each row's prediction is the exact fit of the `N`
/// rows before it, the oracle's fixed point on exactly those rows.
#[test]
fn a_row_window_predicts_from_the_exact_fit_of_its_rows() {
    let n = 60;
    let mut c = cfg(2, 0.8);
    c.window_rows = Some(n);
    let mut m = Expectile::new(c).unwrap();
    let rows = stream(2, 200, 9);
    let mut checked = 0;
    for (i, (x, y)) in rows.iter().enumerate() {
        let got = m.step(x, &[Some(*y)], 1.0, 1.0);
        if i >= n {
            let held: Vec<_> = rows[i - n..i]
                .iter()
                .map(|(x, y)| (z_of(x), *y, 1.0))
                .collect();
            let b = lu_expectile(&held, 0.8);
            let want = dot(&z_of(x), &b);
            assert!(
                (got.pred[0] - want).abs() <= 1e-11 * (1.0 + want.abs()),
                "row {i}: {} against {want}",
                got.pred[0]
            );
            assert_eq!(got.n_eff, n as f64);
            checked += 1;
        }
    }
    assert_eq!(checked, 200 - n);
    assert_eq!(m.window_rows(), n);
}

/// A window of `D` clock units keeps the rows less than `D` old under
/// `Right` and exactly `D` old too under `Both`, decided on the stamps.
#[test]
fn a_clock_window_keeps_the_rows_of_its_span() {
    for (closed, held) in [(WindowClosed::Right, 3), (WindowClosed::Both, 4)] {
        let mut c = cfg(1, 0.3);
        c.window_clock = Some(4.0);
        c.closed = closed;
        let mut m = Expectile::new(c).unwrap();
        let rows = stream(1, 30, 10);
        for (i, (x, y)) in rows.iter().enumerate() {
            m.stamp_next(Stamp::Ns(i as i128 * 1_000_000_000));
            m.step(x, &[Some(*y)], 1.0, 1.0);
        }
        // The next row, 1 s on: the rows less than 4 s old (or 4 s) by then.
        let x = vec![0.25];
        m.stamp_next(Stamp::Ns(30 * 1_000_000_000));
        let got = m.step(&x, &[Some(0.0)], 1.0, 1.0);
        let kept: Vec<_> = rows[30 - held..]
            .iter()
            .map(|(x, y)| (z_of(x), *y, 1.0))
            .collect();
        let want = dot(&z_of(&x), &lu_expectile(&kept, 0.3));
        assert!(
            (got.pred[0] - want).abs() <= 1e-11 * (1.0 + want.abs()),
            "{closed:?}: {} against {want}",
            got.pred[0]
        );
        // Read before the row's own clock step: the rows that left at it
        // still weigh in `n_eff` (hard rule 8).
        assert_eq!(
            got.n_eff,
            if closed == WindowClosed::Right {
                4.0
            } else {
                5.0
            }
        );
    }
}

/// A clock window's state is the data's: the first row to take it past half
/// of `max_state_bytes` is said once, and the row that would pass the cap is
/// not learned, nor any after it.
#[test]
fn a_clock_window_is_warned_of_at_half_its_cap_and_stopped_at_it() {
    let mut c = cfg(2, 0.6);
    c.window_clock = Some(1e9);
    let row = Expectile::window_row_bytes(2, 1, true);
    let fixed = Expectile::clock_state_bytes(2, 1, true, 0);
    let cap = fixed + 100 * row;
    c.max_state_bytes = Some(cap);
    let mut m = Expectile::new(c).unwrap();
    let mut warned_at = None;
    for (i, (x, y)) in stream(2, 150, 11).iter().enumerate() {
        m.step(x, &[Some(*y)], 1.0, 1.0);
        if let Some((bytes, c)) = m.take_state_warning() {
            assert!(warned_at.is_none(), "warned twice");
            assert_eq!(c, cap);
            assert!(2 * bytes > cap);
            warned_at = Some(i);
        }
    }
    let warned_at = warned_at.expect("warned");
    assert!((40..=60).contains(&warned_at), "warned at {warned_at}");
    let (bytes, c) = m.state_over_cap().expect("stopped at the cap");
    assert!(bytes > cap && c == cap);
    assert_eq!(m.window_rows(), 100);
}

/// The configuration's own refusals, each named.
#[test]
fn the_settings_are_checked() {
    let refuse = |c: ExpectileCfg, words: &str| {
        let e = Expectile::new(c).unwrap_err();
        assert!(e.contains(words), "{e}");
    };
    refuse(cfg(2, 1.0), "the level must be above 0 and below 1");
    refuse(cfg(2, 0.0), "the level must be above 0 and below 1");
    let mut c = cfg(2, 0.5);
    c.window_rows = Some(2);
    refuse(c, "at least one row per coefficient");
    let mut c = cfg(2, 0.5);
    c.window_rows = Some(20);
    c.decay = Decay::Halflife(10.0);
    refuse(c, "takes no decay");
    let mut c = cfg(2, 0.5);
    c.window_rows = Some(20);
    c.refit_rows = 30;
    refuse(c, "nothing to add");
    let mut c = cfg(2, 0.5);
    c.window_rows = Some(20);
    c.window_clock = Some(3.0);
    refuse(c, "two windows");
    let mut c = cfg(2, 0.5);
    c.closed = WindowClosed::Both;
    refuse(c, "there is no clock window");
    let mut c = cfg(2, 0.5);
    c.max_state_bytes = Some(0);
    refuse(c, "must be above 0");
    let mut c = cfg(2, 0.5);
    c.max_state_bytes = Some(EXPECTILE_STATE_BYTES_CEILING + 1);
    refuse(c, "at most 67108864 (64 MiB)");
    let mut c = cfg(2, 0.5);
    c.window_rows = Some(100_000);
    refuse(c, "past max_state_bytes of 262144");
    let mut c = cfg(2, 0.5);
    c.refit_rows = 100_000;
    refuse(c, "Lower refit_rows");
}

/// The bytes the settings fix, as the module docs give them.
#[test]
fn the_fixed_bytes_are_the_shape_s() {
    let stamp = std::mem::size_of::<Option<Stamp>>() as u64;
    // k = 5 features, 2 targets, p = 6: 6·7/2 + 12 + 4 = 37 numbers a target.
    assert_eq!(
        Expectile::sums_bytes(5, 2, true, false),
        8 * (2 * 37 + 6) + 2 * stamp
    );
    let mut c = cfg(5, 0.7);
    c.n_targets = 2;
    c.refit_rows = 200;
    assert_eq!(
        c.fixed_bytes(),
        Expectile::sums_bytes(5, 2, true, false) + 8 * 2 * 200 * 7
    );
    let mut c = cfg(5, 0.7);
    c.n_targets = 2;
    c.window_rows = Some(500);
    assert_eq!(
        c.fixed_bytes(),
        Expectile::sums_bytes(5, 2, true, true) + 500 * (8 * 8 + 2)
    );
}

/// The default refit is `max(200, 10 k)` held to the cap, and off under a
/// window or where the cap holds fewer than `2 k` rows.
#[test]
fn the_default_refit_fits_the_cap() {
    let cap = EXPECTILE_MAX_STATE_BYTES;
    assert_eq!(default_expectile_refit_rows(5, 1, true, false, cap), 200);
    assert_eq!(default_expectile_refit_rows(40, 1, true, false, cap), 410);
    assert_eq!(default_expectile_refit_rows(5, 1, true, true, cap), 0);
    let small = default_expectile_refit_rows(5, 40, true, false, cap);
    assert!((12..200).contains(&small), "{small}");
    let mut c = cfg(5, 0.7);
    c.n_targets = 40;
    c.refit_rows = small;
    assert!(c.validate().is_ok());
    assert_eq!(default_expectile_refit_rows(5, 4000, true, false, cap), 0);
    assert_eq!(
        default_expectile_refit_rows(101, 1, true, false, 1 << 26),
        0
    );
}

/// Rows of weight 0 count in no clock (hard rule 9): the weighted rows of a
/// stream with them report what the stream without them reports, to the
/// bit, whether the caller hands temporal stamps (the gap read once from the
/// stamps), the row clock's (no step at all), or none (the steps summed,
/// whole numbers here). Their own rows report a prediction with no NaN left
/// behind, the first rows of the stream among them.
#[test]
fn rows_of_weight_zero_leave_the_weighted_rows_as_they_were() {
    #[derive(Clone, Copy, Debug)]
    enum Clocking {
        Temporal,
        Rows,
        Unstamped,
    }
    let variants: Vec<(&str, ExpectileCfg)> = {
        let mut decayed = cfg(2, 0.8);
        decayed.decay = Decay::Halflife(7.0);
        let mut refit = decayed.clone();
        refit.refit_rows = 40;
        let mut rows = cfg(2, 0.8);
        rows.window_rows = Some(30);
        let mut clock = cfg(2, 0.8);
        clock.window_clock = Some(25.0);
        clock.closed = WindowClosed::Both;
        vec![
            ("decay", decayed),
            ("refit", refit),
            ("window_rows", rows),
            ("window_clock", clock),
        ]
    };
    for (name, c) in variants {
        for clocking in [Clocking::Temporal, Clocking::Rows, Clocking::Unstamped] {
            let rows = stream(2, 160, 12);
            let zero = |i: usize| i < 3 || i % 7 == 3 || (50..58).contains(&i);
            // Each row's clock: irregular steps of 1 to 3.
            let mut t = 0i128;
            let at: Vec<i128> = (0..rows.len())
                .map(|i| {
                    t += 1 + (i as i128 * 7919) % 3;
                    t
                })
                .collect();
            let run = |keep_zero: bool| -> Vec<Step> {
                let mut m = Expectile::new(c.clone()).unwrap();
                let (mut out, mut prev, mut place) = (Vec::new(), None::<i128>, 0i128);
                for (i, (x, y)) in rows.iter().enumerate() {
                    let w = if zero(i) { 0.0 } else { 1.0 };
                    if w == 0.0 && !keep_zero {
                        continue;
                    }
                    place += 1;
                    let d = match clocking {
                        Clocking::Rows => {
                            if prev.is_some() {
                                1.0
                            } else {
                                0.0
                            }
                        }
                        _ => prev.map_or(0.0, |p| (at[i] - p) as f64),
                    };
                    match clocking {
                        Clocking::Temporal => m.stamp_next(Stamp::Ns(at[i] * 1_000_000_000)),
                        Clocking::Rows => m.stamp_next_on_rows(Stamp::from(place as f64)),
                        Clocking::Unstamped => {}
                    }
                    prev = Some(at[i]);
                    let s = m.step(x, &[Some(*y)], d, w);
                    if w == 0.0 {
                        assert!(
                            s.n_eff.is_finite(),
                            "{name} {clocking:?}: row {i} left a NaN"
                        );
                    } else {
                        out.push(s);
                    }
                }
                assert!(m.state_over_cap().is_none());
                out
            };
            let (with, without) = (run(true), run(false));
            assert_eq!(with.len(), without.len());
            let mut finite = 0;
            for (i, (a, b)) in with.iter().zip(&without).enumerate() {
                let same = a.n_eff.to_bits() == b.n_eff.to_bits()
                    && a.pred.len() == b.pred.len()
                    && a.pred
                        .iter()
                        .zip(&b.pred)
                        .all(|(p, q)| p.to_bits() == q.to_bits());
                assert!(
                    same,
                    "{name} {clocking:?}: weighted row {i}: {a:?} against {b:?}"
                );
                finite += usize::from(a.pred[0].is_finite());
            }
            assert!(finite > 100, "{name} {clocking:?}: {finite}");
        }
    }
}

/// A row of weight 0 at the stream's start divides no 0/0: the state stays
/// finite, and nothing is predicted until the first fit.
#[test]
fn a_first_row_of_weight_zero_leaves_no_nan() {
    let mut m = Expectile::new(cfg(1, 0.9)).unwrap();
    let s = m.step(&[1.0], &[Some(2.0)], 0.0, 0.0);
    assert!(s.pred[0].is_nan() && s.n_eff == 0.0);
    assert!(m.s[0] == 0.0 && m.gram[0].iter().all(|v| *v == 0.0));
    for (x, y) in stream(1, 10, 13) {
        m.step(&x, &[Some(y)], 1.0, 1.0);
    }
    assert!(m.coefficients()[0].iter().all(|v| v.is_finite()));
}

/// A state saved at any point continues as the model it was saved from, to
/// the bit, with and without each window and the refit mid-buffer.
#[test]
fn a_restored_state_continues_to_the_bit() {
    let mut refit = cfg(2, 0.7);
    refit.refit_rows = 50;
    let mut rows = cfg(2, 0.7);
    rows.window_rows = Some(20);
    let mut clock = cfg(2, 0.7);
    clock.window_clock = Some(15.0);
    for c in [cfg(2, 0.7), refit, rows, clock] {
        let data = stream(2, 120, 14);
        let mut a = Expectile::new(c).unwrap();
        for (x, y) in &data[..30] {
            a.step(x, &[Some(*y)], 1.0, 1.0);
        }
        let mut b = Expectile::restore(&a.state()).unwrap();
        for (x, y) in &data[30..] {
            let (p, q) = (
                a.step(x, &[Some(*y)], 1.0, 1.0),
                b.step(x, &[Some(*y)], 1.0, 1.0),
            );
            assert_eq!(p.pred[0].to_bits(), q.pred[0].to_bits());
        }
    }
}

/// A state whose sums are not the shape its settings give is refused,
/// never indexed.
#[test]
fn a_state_of_the_wrong_shape_is_refused() {
    let mut c = cfg(2, 0.7);
    c.window_rows = Some(10);
    for c in [cfg(2, 0.7), c] {
        let mut m = Expectile::new(c).unwrap();
        for (x, y) in stream(2, 20, 15) {
            m.step(&x, &[Some(y)], 1.0, 1.0);
        }
        let mut bad = m.clone();
        bad.beta[0].pop();
        let e = Expectile::restore(&bad.state()).unwrap_err();
        assert!(e.to_string().contains("wrong shape"), "{e}");
        let mut bad = m.clone();
        bad.fitted.push(true);
        assert!(Expectile::restore(&bad.state()).is_err());
        if let Some(w) = bad.window.as_mut() {
            w.w.push_back(1.0);
            assert!(Expectile::restore(&bad.state()).is_err());
        }
    }
}
