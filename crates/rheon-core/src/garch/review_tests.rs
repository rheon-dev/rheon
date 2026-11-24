//! The 2026-11-23 review's batch a (PLAN 578a), the GARCH family: one test
//! or more a finding, named by its ID. Each failed on 51671e2e for the
//! review's reason where the outcome says "confirmed"
//! (`docs/records/REVIEW-2026-11-23-OUTCOMES.md`, "Batch a").

use super::filter::Filter;
use super::*;

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

/// A GJR series (`γ = 0` is a GARCH) of `n` rows after 500 discarded,
/// normal innovations scaled by `scale`.
fn gjr_series(theta: [f64; 4], n: usize, seed: u64, scale: f64) -> Vec<f64> {
    let mut s = seed;
    let (w, a, g, b) = (theta[0], theta[1], theta[2], theta[3]);
    let mut v = w / (1.0 - a - 0.5 * g - b);
    let mut e: f64 = 0.0;
    let mut out = Vec::with_capacity(n);
    for t in 0..n + 500 {
        v = w + (a + if e < 0.0 { g } else { 0.0 }) * e * e + b * v;
        e = v.sqrt() * normal(&mut s);
        if t >= 500 {
            out.push(e * scale);
        }
    }
    out
}

/// A GARCH(1,1) series with standardized Student-t innovations of 6
/// degrees of freedom, scaled by `scale`.
fn t_series(n: usize, seed: u64, scale: f64) -> Vec<f64> {
    let mut s = seed;
    let (w, a, b) = (0.05, 0.08, 0.9);
    let mut v = w / (1.0 - a - b);
    let mut e: f64 = 0.0;
    let mut out = Vec::with_capacity(n);
    for t in 0..n + 500 {
        v = w + a * e * e + b * v;
        let chi: f64 = (0..6).map(|_| normal(&mut s).powi(2)).sum();
        let z = normal(&mut s) / (chi / 6.0).sqrt() * (4.0f64 / 6.0).sqrt();
        e = v.sqrt() * z;
        if t >= 500 {
            out.push(e * scale);
        }
    }
    out
}

/// A FIGARCH series of truncation 100 at `(0.03, 0.2, 0.4, 0.5)`, normal
/// innovations, after 3,000 discarded rows.
fn figarch_series(n: usize, seed: u64) -> Vec<f64> {
    let mut s = seed;
    let th = [0.03, 0.2, 0.4, 0.5];
    let mut f = Filter::new(GarchVol::Figarch, 4, 100, 1.0);
    let mut dl = [0.0; 4];
    let mut out = Vec::with_capacity(n);
    for t in 0..n + 3000 {
        let v = f.predict(GarchVol::Figarch, &th, &[], &mut dl);
        let e = v.sqrt() * normal(&mut s);
        f.push(GarchVol::Figarch, e, v, &dl);
        if t >= 3000 {
            out.push(e);
        }
    }
    out
}

fn cfg(vol: GarchVol, dist: GarchDist) -> GarchCfg {
    GarchCfg {
        n_targets: 1,
        vol,
        dist,
        truncation: (vol == GarchVol::Figarch).then_some(100),
        variance_targeting: false,
        start_rows: if vol == GarchVol::Figarch { 2000 } else { 500 },
        start_span: None,
        horizons: Vec::new(),
        max_horizon_work: None,
        n_inputs: 0,
        min_weight: 0.0,
        max_state_bytes: (vol == GarchVol::Figarch).then_some(40_000),
    }
}

fn run(m: &mut Garch, ys: &[f64], w: f64) -> Vec<Vec<f64>> {
    ys.iter()
        .map(|y| m.step(&[], &[Some(*y)], 1.0, w).pred)
        .collect()
}

fn bits(v: &[Vec<f64>]) -> Vec<Vec<u64>> {
    v.iter()
        .map(|r| r.iter().map(|x| x.to_bits()).collect())
        .collect()
}

/// GARCH-1: a projected step whose quadratic model predicts a loss is not
/// read from the slopes. On a concave quadratic in `(x, y)` with its
/// maximum at `(1, −10)`, plus a cubic `40 (x − ½)³₊` that bends `f` up
/// past `x = ½`, over `{x ≤ 1, y ≥ 0}`: the Newton step from `(0, 0)`
/// projects to the corner `(1, 0)`, where `f` is 3.5 below the start and
/// the quadratic model predicts a loss of 8.5; the slopes' trapezoid there
/// is `−8 + 21 > 0`. The set's maximum is `(−8, 0)`, where `f = −9.5`.
#[test]
fn garch_1_a_step_the_quadratic_model_says_loses_is_read_from_f() {
    use super::optim::{Eval, maximize};
    let (a, c) = ([1.0, 0.9, 0.9, 1.0], [1.0, -10.0]);
    let obj = move |x: &[f64], _grad: bool| {
        let d = [x[0] - c[0], x[1] - c[1]];
        let ad = [a[0] * d[0] + a[1] * d[1], a[2] * d[0] + a[3] * d[1]];
        let bend = (x[0] - 0.5).max(0.0);
        Eval {
            f: -0.5 * (d[0] * ad[0] + d[1] * ad[1]) + 40.0 * bend * bend * bend,
            g: vec![-ad[0] + 120.0 * bend * bend, -ad[1]],
            h: a.to_vec(),
        }
    };
    let project = |x: &[f64]| vec![x[0].min(1.0), x[1].max(0.0)];
    let start = obj(&[0.0, 0.0], false).f;
    assert_eq!(start, -41.5);
    let fit = maximize(&obj, &project, &[vec![0.0, 0.0]], &[1.0, 1.0]).unwrap();
    assert!(
        fit.f >= start,
        "the fit fell from {start} to {} at {:?}",
        fit.f,
        fit.theta
    );
    assert!(
        (fit.theta[0] + 8.0).abs() < 1e-9 && fit.theta[1] == 0.0,
        "{:?}",
        fit.theta
    );
}

/// GARCH-3: from a point that breaks EGARCH's guard to one that breaks it
/// too, the cut lands where the filter contracts. `μ = β − (α |z| + γ z)/2`
/// is negative on residuals of `|z| ≈ 5` at `α = 0.8`: lowering `β` makes
/// `|μ|` larger there, so `β = 0` is no floor (the review's two cases).
#[test]
fn garch_3_a_cut_from_a_broken_point_keeps_the_filter_contracting() {
    use super::guard::{admits, bin, cut, mean_log_mu};
    for (z, from, to) in [
        (5.0, [0.0, 0.8, 0.0, 0.90], [0.0, 0.8, 0.0, 0.92]),
        (3.375, [0.0, 0.8, 0.0, 0.30], [0.0, 0.8, 0.0, 0.34]),
        (-5.0, [0.0, 0.8, -0.3, 0.90], [0.0, 0.8, -0.3, 0.92]),
    ] {
        let mut counts = vec![0.0; super::guard::BINS];
        counts[bin(z)] = 1.0;
        assert!(!admits(&from, &counts) && !admits(&to, &counts), "{z}");
        let got = cut(&from, &to, &counts);
        assert!(
            admits(&got, &counts),
            "{z}: the cut {got:?} reads {} (the step's end {})",
            mean_log_mu(&got, &counts),
            mean_log_mu(&to, &counts)
        );
        // still in the box
        assert!(got[1] >= got[2].abs() && got[1] <= 1.0 && (0.0..1.0).contains(&got[3]));
    }
}

/// GARCH-4 (2): a t's score at a residual whose `z²` overflows. A series
/// at `10⁻⁵⁵` met by a return of `10¹⁰⁰` (`z ≈ 10¹⁵⁵`, both inside the
/// inputs' bound) leaves the information matrix a number, and the model
/// goes on learning: its parameters move on the rows after.
#[test]
fn garch_4_a_t_score_past_the_largest_square_leaves_the_state_finite() {
    for dist in [GarchDist::T, GarchDist::Skewt] {
        let ys = t_series(2400, 5, 1e-55);
        let mut m = Garch::new(cfg(GarchVol::Garch, dist)).unwrap();
        run(&mut m, &ys[..2200], 1.0);
        assert!(m.started()[0], "{dist:?}: {:?}", m.start_report());
        m.step(&[], &[Some(1e100)], 1.0, 1.0);
        let h = m.series[0].h.clone();
        assert!(h.iter().all(|v| v.is_finite()), "{dist:?}: {h:?}");
        let before = m.series[0].theta.clone();
        let out = run(&mut m, &ys[2200..], 1.0);
        assert!(m.series[0].h.iter().all(|v| v.is_finite()), "{dist:?}");
        assert!(
            out.iter().all(|r| r[0].is_finite() && r[0] > 0.0),
            "{dist:?}"
        );
        assert_ne!(m.series[0].theta, before, "{dist:?}: frozen");
    }
}

/// GARCH-4 (1): a series at a scale of `10⁻⁸⁰` (returns well inside the
/// inputs' bound) never starts. `∂ ln s/∂ω = 1/s` is `10¹⁶⁰` there, so the
/// start's information `½ (∂ ln s/∂ω)²` and its curvature overflow, no
/// Newton step is formed, the fit is never at its optimum, and every row is
/// withheld; near `10⁸⁰` the same entry falls below the normal doubles. The review's poisoning
/// of a started model's `H` is not reached: no such model starts. Left for
/// the user (a power-of-two rescaling of each series at its start is a
/// layout change).
#[test]
#[should_panic(expected = "never started")]
fn garch_4_a_series_at_a_scale_of_1e_minus_80_starts() {
    let ys = gjr_series([0.05, 0.08, 0.0, 0.9], 2600, 7, 1e-80);
    let mut m = Garch::new(cfg(GarchVol::Garch, GarchDist::Normal)).unwrap();
    let out = run(&mut m, &ys, 1.0);
    assert!(m.started()[0], "never started: {:?}", m.start_report());
    assert!(out.last().unwrap()[0].is_finite());
}

/// GARCH-4 (2), the density itself: `log f(z)` and the scores at `z` past
/// `√(f64::MAX)` are numbers, and continue those just below it.
#[test]
fn garch_4_the_t_density_and_scores_past_the_largest_square_are_numbers() {
    use super::dist::{log_density, scores};
    for (nu, lam) in [(6.0, 0.0), (4.5, -0.3), (30.0, 0.5)] {
        let skew = lam != 0.0;
        for z in [1e150, -1e150, 1e156, -1e156, 1e300] {
            let lf = log_density(z, nu, lam);
            let (w, s_nu, s_lam) = scores(z, nu, lam, skew);
            assert!(
                lf.is_finite() && w.is_finite() && s_nu.is_finite() && s_lam.is_finite(),
                "{nu} {lam} {z}: {lf} {w} {s_nu} {s_lam}"
            );
            // far out, the variance score's factor is ν: (ν + 1) z²/(ν − 2 + z²) − 1
            assert!((w - nu).abs() < 1e-9 * nu, "{nu} {lam} {z}: {w}");
        }
    }
}

/// GARCH-5 (1): EGARCH's upper bound reads `hi + ln s − ln hi`, `arch`'s
/// `var_bounds[t, 1] + log(sigma2) − log(var_bounds[t, 1])`, which never
/// forms `s/hi`: at `hi = 10⁻⁶` and `ln s = 700` that ratio overflows.
#[test]
fn garch_5_the_egarch_upper_bound_does_not_overflow() {
    use super::math;
    let f = Filter::new(GarchVol::Egarch, 4, 0, 1e-12);
    let mut dl = [0.0; 4];
    let s = f.predict(GarchVol::Egarch, &[700.0, 0.0, 0.0, 0.0], &[], &mut dl);
    let hi = 1e-12 * 1e6;
    let want = hi + 700.0 - math::ln(hi);
    assert!(
        s.is_finite() && (s - want).abs() <= 1e-15 * want,
        "{s} against {want}"
    );
}

/// GARCH-5 (2): where EGARCH's upper bound binds, the variance still moves
/// with the parameters (`∂s/∂θ = ∂ ln s_raw/∂θ` below the exponent's cap),
/// so `∂ ln s/∂θ` is that over `s`, not 0. Over two rows from a fresh
/// filter, both bounded above, against a central difference of the whole
/// run, its first row's lags recomputed at each parameter.
#[test]
fn garch_5_the_egarch_sensitivity_past_the_upper_bound_is_its_slope() {
    let ln_s = |theta: &[f64; 4], dl: &mut [f64; 4]| {
        let mut f = Filter::new(GarchVol::Egarch, 4, 0, 1.0);
        let s0 = f.predict(GarchVol::Egarch, theta, &[], dl);
        f.push(GarchVol::Egarch, 1.3e3, s0, dl);
        f.predict(GarchVol::Egarch, theta, &[], dl).ln()
    };
    let theta = [20.0, 0.1, -0.05, 0.3];
    let mut dl = [0.0; 4];
    let at = ln_s(&theta, &mut dl);
    assert!(at.exp() > 1e6, "the bound binds: {}", at.exp());
    for i in 0..4 {
        let h = 1e-6;
        let (mut up, mut dn) = (theta, theta);
        up[i] += h;
        dn[i] -= h;
        let mut scratch = [0.0; 4];
        let num = (ln_s(&up, &mut scratch) - ln_s(&dn, &mut scratch)) / (2.0 * h);
        assert!(
            (dl[i] - num).abs() <= 1e-6 * num.abs().max(1e-9),
            "{i}: {} against {num}",
            dl[i]
        );
    }
}

/// GARCH-5 (3): an EGARCH whose first 75 rows are 0 has a backcast of 0,
/// whose logarithm is the filter's first `ln s` lag; it starts all the same,
/// on the rows' mean square.
#[test]
fn garch_5_an_egarch_whose_first_rows_are_zero_starts() {
    let c = cfg(GarchVol::Egarch, GarchDist::Normal);
    let mut ys = vec![0.0; 75];
    ys.extend(gjr_series([0.05, 0.08, 0.0, 0.9], 3000, 9, 1.0));
    let mut m = Garch::new(c).unwrap();
    let out = run(&mut m, &ys, 1.0);
    assert!(m.started()[0], "{:?}", m.start_report());
    assert!(out.last().unwrap()[0].is_finite());
}

/// GARCH-5 (3), online: a run of zero residuals long enough for the
/// 0.94-weighted mean of `e²` to underflow to 0 (about 12,050 rows) leaves
/// EGARCH reporting a variance on every row, and its parameters numbers.
#[test]
fn garch_5_a_long_run_of_zeros_leaves_egarch_reporting() {
    let c = cfg(GarchVol::Egarch, GarchDist::Normal);
    let ys = gjr_series([0.05, 0.08, 0.0, 0.9], 3000, 10, 1.0);
    let mut m = Garch::new(c).unwrap();
    run(&mut m, &ys[..2000], 1.0);
    assert!(m.started()[0]);
    let mut out = run(&mut m, &vec![0.0; 13_000], 1.0);
    out.extend(run(&mut m, &ys[2000..], 1.0));
    assert!(m.coefficients()[0].iter().all(|v| v.is_finite()));
    for (t, r) in out.iter().enumerate() {
        assert!(
            r[0].is_finite() && r[0] > 0.0 && r[1].is_finite(),
            "{t}: {r:?}"
        );
    }
}

/// A started model of `c` on its own series, for the damaged states below.
fn started(c: &GarchCfg) -> Garch {
    for seed in 4..12 {
        let mut m = Garch::new(c.clone()).unwrap();
        run(&mut m, &series(c, 2600, seed), 1.0);
        if m.started()[0] {
            return m;
        }
    }
    panic!("{c:?}: no seed started");
}

/// The series a form is tested on: its own process for FIGARCH, t
/// innovations for a t.
fn series(c: &GarchCfg, n: usize, seed: u64) -> Vec<f64> {
    if c.vol == GarchVol::Figarch {
        figarch_series(n, seed)
    } else if c.dist == GarchDist::Normal {
        gjr_series([0.05, 0.08, 0.0, 0.9], n, seed, 1.0)
    } else {
        t_series(n, seed, 1.0)
    }
}

/// GARCH-6: a damaged state is refused at the load: a shape out of its
/// range (`ν = −10¹⁷`, on which `ln Γ`'s shift ran forever, or `ν = 2`,
/// where the t has no variance), a FIGARCH at `β = 1`, parameters that are
/// not numbers, a weight below 0, and a buffer past `max_state_bytes`.
#[test]
fn garch_6_a_damaged_state_is_refused() {
    let t = started(&cfg(GarchVol::Garch, GarchDist::T));
    let refused = |m: &Garch, what: &str| {
        let e = Garch::restore(&m.state());
        assert!(e.is_err(), "{what}: loaded");
    };
    for nu in [-1e17, 2.0, f64::NAN, 1e9] {
        let mut bad = t.clone();
        bad.series[0].theta[3] = nu;
        refused(&bad, &format!("nu {nu}"));
    }
    let mut bad = t.clone();
    bad.series[0].theta[1] = f64::NAN;
    refused(&bad, "alpha NaN");
    let mut bad = t.clone();
    bad.series[0].w_sum = -1.0;
    refused(&bad, "w_sum");
    let mut bad = t.clone();
    bad.w_sum = f64::NAN;
    refused(&bad, "the model's w_sum");
    let s = started(&cfg(GarchVol::Garch, GarchDist::Skewt));
    let mut bad = s.clone();
    bad.series[0].theta[4] = 1.5;
    refused(&bad, "lambda");
    let f = started(&cfg(GarchVol::Figarch, GarchDist::Normal));
    let mut bad = f.clone();
    bad.series[0].theta[3] = 1.0;
    refused(&bad, "figarch beta");
    // a start of a span holding more than the cap
    let mut c = cfg(GarchVol::Garch, GarchDist::Normal);
    c.start_span = Some(1e9);
    c.max_state_bytes = Some(4000);
    let mut m = Garch::new(c).unwrap();
    run(&mut m, &[0.1; 100], 1.0);
    let mut bad = m.clone();
    bad.series[0].buffer.extend(vec![[0.1, 1.0]; 400]);
    bad.series[0].w_sum += 400.0;
    refused(&bad, "a buffer past the cap");
    let mut bad = m.clone();
    bad.series[0].buffer[3][1] = -1.0;
    refused(&bad, "a buffered weight");
    // and the healthy ones load
    for m in [&t, &s, &f, &m] {
        assert!(Garch::restore(&m.state()).is_ok());
    }
}

/// GARCH-6, `ln Γ` and `ψ` away from their domain: a number at once, not a
/// shift by ones that never moves a value of `10¹⁷` (run on a thread, so the
/// old loop fails the test instead of hanging it).
#[test]
fn garch_6_ln_gamma_outside_its_domain_returns() {
    use super::dist::{digamma, ln_gamma};
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let got = [-1e17, f64::NEG_INFINITY, 0.0, -3.0]
            .map(|x| (ln_gamma(x).is_nan(), digamma(x).is_nan()));
        let _ = tx.send(got);
    });
    let got = rx
        .recv_timeout(std::time::Duration::from_secs(5))
        .expect("ln Γ ran on past 5 s");
    assert!(got.iter().all(|&(a, b)| a && b), "{got:?}");
}

/// GARCH-7: a target's own weight, which `min_weight` gates, is the weight
/// its fit rests on. A start that kept its latest rows after a try that did
/// not start (its buffer at its share of the cap) rests on those rows: the
/// rows its ring let go taught it nothing, as a window's rows that left
/// teach a windowed fit nothing. So the review's case, a first 500 rows
/// with no process and a cap of 500 rows, gates on the 500 rows of its
/// fit, while `weight_sum` counts every row's weight.
#[test]
fn garch_7_a_targets_weight_is_the_weight_its_fit_rests_on() {
    let mut c = cfg(GarchVol::Garch, GarchDist::Normal);
    c.min_weight = 600.0;
    c.max_state_bytes = Some(Garch::state_bytes(&c, 500));
    let mut s = 3;
    let mut ys: Vec<f64> = (0..500).map(|_| normal(&mut s)).collect();
    ys.extend(gjr_series([0.05, 0.08, 0.0, 0.9], 700, 12, 1.0));
    let mut m = Garch::new(c).unwrap();
    for (t, y) in ys.iter().enumerate() {
        let out = m.step(&[], &[Some(*y)], 1.0, 1.0);
        assert_eq!(out.n_eff, t as f64, "{t}");
        if t == 999 {
            assert!(m.started()[0], "the second try starts");
        }
        let mut w = Vec::new();
        m.target_n_eff_into(&mut w);
        if t >= 999 {
            // the 500 rows of the fit, and each row learned since
            assert_eq!(w[0], (t + 1 - 500) as f64, "{t}");
            assert_eq!(out.pred[0].is_finite(), t >= 1100, "{t}");
        }
    }
}

/// GARCH-8 (1): a FIGARCH start of a span at a cap its ring and model fill
/// exactly buffers its first row: a buffered row adds nothing to the bytes
/// while the buffer is no larger than the ring.
#[test]
fn garch_8_a_figarch_span_start_at_its_exact_cap_buffers() {
    let c = GarchCfg {
        truncation: Some(1000),
        start_rows: 2000,
        start_span: Some(100.0),
        max_state_bytes: Some(8232),
        ..cfg(GarchVol::Figarch, GarchDist::Normal)
    };
    assert_eq!(Garch::state_bytes(&c, 0), 8232);
    let mut m = Garch::new(c).unwrap();
    for t in 0..10 {
        m.stamp_next(crate::Stamp::from(t as f64));
        m.step(&[], &[Some(0.3)], 1.0, 1.0);
        assert!(m.state_over_cap().is_none(), "{t}");
        assert_eq!(m.series[0].buffer.len(), t + 1);
    }
}

/// GARCH-8 (2): a FIGARCH start of a span fits on as few rows as any
/// form's span start, 50, where a start of rows is refused under 2,000
/// for FIGARCH. Left for the user: a span start holding FIGARCH to 2,000
/// rows is a fixed row count.
#[test]
#[should_panic(expected = "a FIGARCH start of a span tried on")]
fn garch_8_a_figarch_span_start_tries_on_no_fewer_rows_than_a_start_of_rows() {
    let c = GarchCfg {
        start_span: Some(60.0),
        ..cfg(GarchVol::Figarch, GarchDist::Normal)
    };
    let ys = figarch_series(200, 6);
    let mut m = Garch::new(c).unwrap();
    for (t, y) in ys.iter().enumerate() {
        m.stamp_next(crate::Stamp::from(t as f64));
        m.step(&[], &[Some(*y)], 1.0, 1.0);
        let tried = m.series[0].tried;
        assert!(
            m.start_report()[0].0 == 0 && !m.started()[0] || tried >= FIGARCH_MIN_START_ROWS as u64,
            "a FIGARCH start of a span tried on {tried} rows at row {t}"
        );
    }
}

/// GARCH-9: every weight scaled by a power of two (`2⁻³⁴`, `2⁻⁶⁰`, `2⁴⁰`),
/// which every product carries exactly, changes nothing but `weight_sum`:
/// each form's reports are the unit weights' to the bit, the
/// variance-targeted GARCH's and FIGARCH's among them.
#[test]
fn garch_9_a_power_of_two_on_every_weight_moves_no_report() {
    let mut cfgs: Vec<GarchCfg> = [
        (GarchVol::Garch, GarchDist::Normal),
        (GarchVol::Gjr, GarchDist::Normal),
        (GarchVol::Egarch, GarchDist::Normal),
        (GarchVol::Figarch, GarchDist::Normal),
        (GarchVol::Garch, GarchDist::T),
        (GarchVol::Gjr, GarchDist::T),
        (GarchVol::Garch, GarchDist::Skewt),
    ]
    .map(|(v, d)| cfg(v, d))
    .to_vec();
    cfgs.push(GarchCfg {
        variance_targeting: true,
        ..cfg(GarchVol::Garch, GarchDist::Normal)
    });
    for c in cfgs {
        let n = if c.vol == GarchVol::Figarch {
            2300
        } else {
            1200
        };
        let mut tried = 0;
        for seed in 3..12 {
            let ys = series(&c, n, seed);
            let mut a = Garch::new(c.clone()).unwrap();
            let oa = run(&mut a, &ys, 1.0);
            if !a.started()[0] {
                continue;
            }
            tried += 1;
            if tried > 2 {
                break;
            }
            for p in [-34, -60, 40] {
                let mut b = Garch::new(c.clone()).unwrap();
                let ob = run(&mut b, &ys, (2.0f64).powi(p));
                assert_eq!(bits(&oa), bits(&ob), "{c:?} {seed} 2^{p}");
            }
        }
        assert!(tried >= 2, "{c:?}: {tried} seeds started");
    }
}

/// GARCH-10: the projection in a metric is the problem's minimizer, by its
/// optimality conditions (`crate::oracle::project_qp`, an active-set
/// enumeration on faer's LU): on GARCH's set at `ω ≈ 10⁸`, on GJR's, and on
/// FIGARCH's, at metrics whose condition `κ` reaches `10⁸` and more, every
/// point it returns is within `10⁻¹² + 10 κ ε` of the minimizer in each
/// coordinate's units (both solves lose `κ`'s digits), and outside the set
/// by no more than `10⁻¹²` of them (the slack a caller makes exact for the
/// bounds that keep the variance positive, GARCH-12). Dykstra's sweeps,
/// which it ran before, were off by up to 1 and outside by as much at a
/// condition of `10⁸`.
#[test]
fn garch_10_the_projection_in_a_metric_is_the_minimizer() {
    use super::fit::{figarch_set, garch_set, gjr_set};
    let mut s = 77u64;
    let (mut worst, mut outside, mut kappa) = (0.0f64, 0.0f64, 0.0f64);
    for case in 0..300 {
        let (set, n, x, scale) = match case % 3 {
            0 => {
                let w = 1e8;
                let x = vec![
                    w * (0.5 + lcg(&mut s)),
                    0.3 * normal(&mut s),
                    0.9 + 0.2 * lcg(&mut s),
                ];
                (garch_set(1e-6 * w), 3, x, [w, 1.0, 1.0, 1.0])
            }
            1 => {
                let x = vec![
                    0.1,
                    0.05 * normal(&mut s),
                    0.1 * normal(&mut s),
                    0.8 + 0.3 * lcg(&mut s),
                ];
                (gjr_set(1e-7), 4, x, [1.0; 4])
            }
            _ => {
                let x = vec![
                    0.03,
                    0.3 * lcg(&mut s) - 0.05,
                    0.6 * lcg(&mut s),
                    0.2 + 0.8 * lcg(&mut s),
                ];
                (figarch_set(1e-7), 4, x, [1.0; 4])
            }
        };
        // a metric `D⁻¹ B Bᵀ D⁻¹`, `B`'s first column 10⁻⁴ of the rest, so
        // its condition is about 10⁸ along a direction no axis is
        let mut b = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                b[i * n + j] = normal(&mut s) * if j == 0 { 1e-4 } else { 1.0 }
                    + if i == j && j > 0 { 0.5 } else { 0.0 };
            }
        }
        let mut m = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                let v: f64 = (0..n).map(|k| b[i * n + k] * b[j * n + k]).sum();
                m[i * n + j] = v / (scale[i] * scale[j]);
            }
        }
        let scaled: Vec<f64> = (0..n * n)
            .map(|ij| m[ij] * scale[ij / n] * scale[ij % n])
            .collect();
        kappa = kappa.max(crate::oracle::condition(&scaled));
        let Some(y) = set.project_metric(&x, &m) else {
            continue;
        };
        // the oracle on the problem in each coordinate's unit, `y = D z`
        let xs: Vec<f64> = (0..n).map(|i| x[i] / scale[i]).collect();
        let a_s: Vec<f64> = (0..set.a.len())
            .map(|ij| set.a[ij] * scale[ij % n])
            .collect();
        let Some(z) = crate::oracle::project_qp(&scaled, &xs, &a_s, &set.b, 1e-6) else {
            panic!("{case}: no minimizer found")
        };
        let want: Vec<f64> = (0..n).map(|i| z[i] * scale[i]).collect();
        let mut e = 0.0f64;
        for i in 0..n {
            let err = (y[i] - want[i]).abs() / (scale[i] * (1.0 + (want[i] / scale[i]).abs()));
            e = e.max(err);
        }
        let mut o = 0.0f64;
        for (r, row) in set.a.chunks(n).enumerate() {
            let slack: f64 = row.iter().zip(&y).map(|(a, y)| a * y).sum::<f64>() - set.b[r];
            let unit: f64 = row.iter().zip(&scale).map(|(a, u)| (a * u).abs()).sum();
            o = o.max(-slack / unit);
        }
        // both solves lose the condition's digits along its narrow
        // direction: the bound is ten times `κ ε`
        let k = crate::oracle::condition(&scaled);
        worst = worst.max(e / (1e-12 + 10.0 * k * f64::EPSILON));
        outside = outside.max(o);
    }
    eprintln!("GARCH-10: worst {worst:e}, outside {outside:e}, condition up to {kappa:e}");
    assert!(kappa > 1e8, "{kappa:e}");
    assert!(
        worst <= 1.0 && outside < 1e-12,
        "GARCH-10: worst {worst:e} of its bound, outside {outside:e}"
    );
}

/// GARCH-12: a step's projection keeps the shocks' weights at 0 or above
/// exactly -- GJR's `α` and `α + γ`, GARCH's `α` and `β` -- so a return as
/// large as the inputs allow never takes the variance below 0. Over a GJR
/// whose `α` is 0, its steps on that bound, every row's parameters.
#[test]
fn garch_12_a_step_keeps_every_shock_weight_at_zero_or_above() {
    let c = cfg(GarchVol::Gjr, GarchDist::Normal);
    for seed in 1..4 {
        let ys = gjr_series([0.05, 0.0, 0.15, 0.88], 4000, seed, 1.0);
        let mut m = Garch::new(c.clone()).unwrap();
        for (t, y) in ys.iter().enumerate() {
            m.step(&[], &[Some(*y)], 1.0, 1.0);
            if m.started()[0] {
                let th = &m.series[0].theta;
                assert!(
                    th[1] >= 0.0 && th[1] + th[2] >= 0.0 && th[3] >= 0.0,
                    "{seed} {t}: {th:?}"
                );
            }
        }
    }
}

/// GARCH-12: a variance at or below 0 is never reported, nor learned from.
/// A GJR at `α + γ = −4.1·10⁻¹⁵`, the slack the old projection left, meets
/// `e = −10⁸`; a FIGARCH at `λ₁ = −9·10⁻¹³` meets `e = 10⁹`: the next
/// row's variance is a positive number or withheld, and a GJR with a t
/// learning from it keeps its information matrix a number.
#[test]
fn garch_12_no_variance_at_or_below_zero_is_reported() {
    for (vol, dist) in [
        (GarchVol::Gjr, GarchDist::Normal),
        (GarchVol::Gjr, GarchDist::T),
        (GarchVol::Figarch, GarchDist::Normal),
    ] {
        let mut m = started(&cfg(vol, dist));
        let th = &mut m.series[0].theta;
        let shock = if vol == GarchVol::Gjr {
            th[2] = -th[1] - 4.108e-15;
            -1e8
        } else {
            th[3] = th[1] + th[2] + 9e-13;
            1e9
        };
        let k = vol.k();
        let mut dl = [0.0; 4];
        let s = m.series[0].read(&m.cfg, &[], &mut dl[..k]);
        m.series[0].filter.push(vol, shock, s, &dl[..k]);
        let p = m.predict(&[], 1.0).pred;
        assert!(p[0].is_nan() || p[0] > 0.0, "{vol:?} {dist:?}: {}", p[0]);
        let out = m.step(&[], &[Some(0.5)], 1.0, 1.0);
        assert!(
            out.pred[0].is_nan() || out.pred[0] > 0.0,
            "{vol:?} {dist:?}"
        );
        assert!(
            m.series[0].h.iter().all(|v| v.is_finite()),
            "{vol:?} {dist:?}"
        );
    }
}

/// GARCH-2: a start whose fit is at its optimum on a bound of the shape --
/// a Gaussian series fitted with a t, its `ν` at 300 -- starts at its first
/// try. The unconstrained decrement there is large (the score in `ν` points
/// out of the box), but soundness is read from the decrement inside the set
/// in the curvature's metric, which is the rounding's. Of twelve 2,000-row
/// series, nine fit `ν = 300`; none doubles its start.
#[test]
fn garch_2_a_t_fit_at_its_shape_bound_starts_at_its_first_try() {
    let mut on_bound = 0;
    for seed in 0..12u64 {
        let ys = gjr_series([0.05, 0.08, 0.0, 0.9], 2000, 100 + seed, 1.0);
        let c = GarchCfg {
            start_rows: 2000,
            ..cfg(GarchVol::Garch, GarchDist::T)
        };
        let mut m = Garch::new(c).unwrap();
        run(&mut m, &ys, 1.0);
        assert!(m.started()[0], "{seed}: {:?}", m.start_report());
        assert_eq!(m.start_report()[0], (0, true), "{seed}");
        on_bound += usize::from(m.series[0].theta[3] == super::dist::NU_HI);
    }
    assert!(on_bound >= 6, "{on_bound} of 12 on the bound");
}

/// GARCH-11 (3): a start whose rows are all 0 waits as `not_identified`
/// (`no_effect`): its rows tell no variance process from a constant
/// variance, 0, so more rows of 0 cannot start it, which is not the start's
/// own warm-up.
#[test]
fn garch_11_a_start_of_zeros_waits_as_not_identified() {
    for (vol, dist) in [
        (GarchVol::Garch, GarchDist::Normal),
        (GarchVol::Egarch, GarchDist::Normal),
        (GarchVol::Garch, GarchDist::T),
    ] {
        let mut m = Garch::new(cfg(vol, dist)).unwrap();
        run(&mut m, &[0.0; 499], 1.0);
        assert!(
            m.starting() && !m.no_effect(),
            "{vol:?} {dist:?}: the start's rows"
        );
        run(&mut m, &[0.0; 1], 1.0);
        assert!(!m.started()[0], "{vol:?} {dist:?}");
        assert!(
            m.no_effect() && !m.starting(),
            "{vol:?} {dist:?}: after its try"
        );
        assert!(m.coefficients()[0].iter().all(|v| v.is_nan()));
    }
}

/// GARCH-12, the bounds made exact: a point the projection leaves a hair
/// outside each bound that keeps a shock's weight at 0 or above is moved
/// onto it exactly, and a point inside is left to the bit. GJR's `α + γ`
/// is then 0 exactly, as a sum, so `α e² + γ e²` is 0 for any `e`.
#[test]
fn garch_12_the_shock_weights_bounds_are_made_exact() {
    use super::series::exact;
    let floor = 1e-7;
    for (vol, inside, outside, check) in [
        (
            GarchVol::Garch,
            vec![0.05, 0.08, 0.9],
            vec![1e-9, -1e-17, -2e-17],
            (|y: &[f64]| y[0] == 1e-7 && y[1] == 0.0 && y[2] == 0.0) as fn(&[f64]) -> bool,
        ),
        (
            GarchVol::Gjr,
            vec![0.05, 0.03, 0.1, 0.85],
            vec![1e-9, 0.05, -0.05 - 4.108e-15, -1e-18],
            |y: &[f64]| {
                y[0] == 1e-7 && y[1] == 0.05 && y[1] + y[2] == 0.0 && y[3] == 0.0 && {
                    let e2 = 1e16;
                    y[1] * e2 + y[2] * e2 == 0.0
                }
            },
        ),
        (
            GarchVol::Gjr,
            vec![0.05, 0.03, 0.1, 0.85],
            vec![0.05, -1e-16, 0.2, 0.7],
            |y: &[f64]| y[1] == 0.0 && y[2] == 0.2,
        ),
        (
            GarchVol::Figarch,
            vec![0.03, 0.2, 0.4, 0.5],
            vec![0.03, 0.2, 0.4, 0.6 + 9e-13],
            |y: &[f64]| y[3] == y[1] + y[2],
        ),
        (
            GarchVol::Figarch,
            vec![0.03, 0.2, 0.4, 0.5],
            vec![-1.0, -1e-17, 1.0, -1e-17],
            |y: &[f64]| y[0] == 1e-7 && y[2] == 1.0 - 1e-4 && y[1] == 0.0 && y[3] == 0.0,
        ),
        (
            GarchVol::Figarch,
            vec![0.03, 0.2, 0.4, 0.5],
            vec![0.03, 0.4, 0.4, 0.5],
            |y: &[f64]| y[1] == 0.5 * (1.0 - 0.4) && y[3] == 0.5,
        ),
        (
            GarchVol::Egarch,
            vec![0.01, 0.15, -0.08, 0.97],
            vec![0.01, 1.5, -1.6, 1.0],
            |y: &[f64]| y[1] == 1.0 && y[2] == -1.0 && y[3] == 1.0 - 1e-4 && y[0] == 0.01,
        ),
        (
            GarchVol::Egarch,
            vec![0.01, 0.15, -0.08, 0.97],
            vec![0.01, -0.1, 0.3, -0.2],
            |y: &[f64]| y[1] == 0.0 && y[2] == 0.0 && y[3] == 0.0,
        ),
    ] {
        let mut y = inside.clone();
        exact(vol, &mut y, floor);
        assert_eq!(y, inside, "{vol:?}: a point inside moved");
        let mut y = outside.clone();
        exact(vol, &mut y, floor);
        assert!(check(&y), "{vol:?}: {outside:?} became {y:?}");
    }
    // a point that is not a number is left for the caller to refuse
    let mut y = vec![f64::NAN, 0.1, 0.8];
    exact(GarchVol::Garch, &mut y, floor);
    assert!(y[0].is_nan() && y[1] == 0.1);
}

/// W14-15: a state saved after a row at the input bound loads, and goes on
/// to the bit. At `|e| = 10¹⁰⁰` the step's `ω` is near `10¹⁹⁵`, and the
/// projection in its metric left GARCH's and GJR's `α` and `β` at its
/// rounding, `10¹⁶⁷` to `10¹⁸⁰`, outside the set the load reads; and under
/// variance targeting the information matrix took a square past `10³⁰⁸`,
/// `∞` for good. Each form and shape, each sign, saved after the bound row
/// and after the row past it. The other side: a state outside the set, or
/// with an information matrix that is not a number, is still refused.
#[test]
fn a_state_after_a_row_at_the_input_bound_loads_and_goes_on_to_the_bit() {
    let mut forms = Vec::new();
    for vol in [
        GarchVol::Garch,
        GarchVol::Gjr,
        GarchVol::Egarch,
        GarchVol::Figarch,
    ] {
        for dist in [GarchDist::Normal, GarchDist::T, GarchDist::Skewt] {
            let c = cfg(vol, dist);
            if c.validate().is_ok() {
                forms.push(c);
            }
        }
    }
    forms.push(GarchCfg {
        variance_targeting: true,
        ..cfg(GarchVol::Garch, GarchDist::Normal)
    });
    let after = gjr_series([0.05, 0.08, 0.0, 0.9], 40, 99, 1.0);
    for c in &forms {
        let m0 = started(c);
        for bound in [crate::INPUT_BOUND, -crate::INPUT_BOUND] {
            for past in 0..2 {
                let mut m = m0.clone();
                run(&mut m, &[bound], 1.0);
                run(&mut m, &after[..past], 1.0);
                let what = format!(
                    "{:?} {:?} vt {} at {bound:e}, {past} row past",
                    c.vol, c.dist, c.variance_targeting
                );
                let mut back =
                    Garch::restore(&m.state()).unwrap_or_else(|e| panic!("{what}: refused: {e}"));
                assert_eq!(back, m, "{what}");
                let set = super::fit::set_of(c.vol, c.q(), 0.0);
                let k = c.k();
                // in the set to the projection's tolerance, `10⁻¹²` of each
                // constraint's size, as every step leaves it
                assert!(
                    set.holds(&m.series[0].theta[..k], 1e-12),
                    "{what}: {:?}",
                    m.series[0].theta
                );
                let (a, b) = (run(&mut m, &after, 1.0), run(&mut back, &after, 1.0));
                assert_eq!(bits(&a), bits(&b), "{what}");
                assert!(a.iter().flatten().all(|v| !v.is_infinite()), "{what}");
            }
        }
    }
    // the other side: outside the set, or an information matrix that is
    // not a number, is damage
    let refused = |m: &Garch, what: &str| {
        assert!(Garch::restore(&m.state()).is_err(), "{what}: loaded");
    };
    let g = started(&cfg(GarchVol::Garch, GarchDist::Normal));
    let mut bad = g.clone();
    bad.series[0].theta[1] = 0.2;
    bad.series[0].theta[2] = 0.9;
    refused(&bad, "garch alpha + beta 1.1");
    let mut bad = g.clone();
    bad.series[0].theta[2] = 1.8e167;
    refused(&bad, "garch beta 1.8e167");
    let j = started(&cfg(GarchVol::Gjr, GarchDist::Normal));
    let mut bad = j.clone();
    bad.series[0].theta[1..4].copy_from_slice(&[8.8e179, -8.8e179, 0.0]);
    refused(&bad, "gjr alpha + gamma / 2 8.8e179 / 2");
    let v = started(&GarchCfg {
        variance_targeting: true,
        ..cfg(GarchVol::Garch, GarchDist::Normal)
    });
    let mut bad = v.clone();
    bad.series[0].h[0] = f64::INFINITY;
    refused(&bad, "an information matrix of inf");
}
