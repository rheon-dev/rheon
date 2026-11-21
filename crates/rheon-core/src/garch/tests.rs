use super::filter::{Filter, figarch_weights};
use super::fit::backcast;
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

/// A GJR series (`gamma = 0` is a GARCH) of `n` rows after 500 discarded,
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

/// A GARCH(1,1) series with Student-t innovations of 6 degrees of freedom,
/// standardized: `N/√(χ²₆/6) · √(4/6)`.
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
        let v = f.predict(GarchVol::Figarch, &th, &mut dl);
        let e = v.sqrt() * normal(&mut s);
        f.push(GarchVol::Figarch, e, v, &dl);
        if t >= 3000 {
            out.push(e);
        }
    }
    out
}

/// The series a form is tested on: t innovations for a t form.
fn series_for(dist: GarchDist, n: usize, seed: u64) -> Vec<f64> {
    match dist {
        GarchDist::Normal => gjr_series([0.05, 0.08, 0.0, 0.9], n, seed, 1.0),
        _ => t_series(n, seed, 1.0),
    }
}

/// The series a config is tested on: its own process for FIGARCH.
fn series_of(c: &GarchCfg, n: usize, seed: u64) -> Vec<f64> {
    if c.vol == GarchVol::Figarch {
        figarch_series(n, seed)
    } else {
        series_for(c.dist, n, seed)
    }
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
        horizon: 1,
        min_weight: 0.0,
        // FIGARCH's start too big to double, so its tests stay short
        max_state_bytes: (vol == GarchVol::Figarch).then_some(40_000),
    }
}

const FORMS: [(GarchVol, GarchDist); 7] = [
    (GarchVol::Garch, GarchDist::Normal),
    (GarchVol::Gjr, GarchDist::Normal),
    (GarchVol::Egarch, GarchDist::Normal),
    (GarchVol::Figarch, GarchDist::Normal),
    (GarchVol::Garch, GarchDist::T),
    (GarchVol::Gjr, GarchDist::T),
    (GarchVol::Garch, GarchDist::Skewt),
];

/// The forms, and the variance-targeted GARCH.
fn every_cfg() -> Vec<GarchCfg> {
    let mut out: Vec<GarchCfg> = FORMS.iter().map(|&(v, d)| cfg(v, d)).collect();
    out.push(GarchCfg {
        variance_targeting: true,
        ..cfg(GarchVol::Garch, GarchDist::Normal)
    });
    out
}

/// The rows a config's start holds at most, after its refits.
fn start(c: &GarchCfg) -> usize {
    if c.vol == GarchVol::Figarch {
        c.start_rows
    } else {
        4 * c.start_rows
    }
}

/// The first row a run reported a variance on, every row before it
/// reporting none and every row from it one.
fn first_report(out: &[Vec<f64>]) -> usize {
    let f = out.iter().position(|r| r[0].is_finite()).expect("a start");
    assert!(out[f..].iter().all(|r| r[0] > 0.0 && r[0].is_finite()));
    f
}

fn state_bits(m: &Garch) -> Vec<u8> {
    rmp_serde::to_vec_named(&m.state()).unwrap()
}

fn run(m: &mut Garch, ys: &[f64], ws: &[f64]) -> Vec<Vec<f64>> {
    ys.iter()
        .zip(ws)
        .map(|(y, w)| m.step(&[], &[Some(*y)], 1.0, *w).pred)
        .collect()
}

fn bits(v: &[Vec<f64>]) -> Vec<Vec<u64>> {
    v.iter()
        .map(|r| r.iter().map(|x| x.to_bits()).collect())
        .collect()
}

/// FIGARCH's weights against the definition: `λ(L) = 1 − (1 − φL)(1 −
/// L)^d / (1 − βL)`, the binomial series of `(1 − L)^d` multiplied out and
/// divided by `1 − βL` term by term -- another route to the same numbers.
#[test]
fn figarch_weights_are_the_fractional_differences_expansion() {
    let m = 400;
    for th in [
        [0.0, 0.2, 0.4, 0.5],
        [0.0, 0.0, 0.9, 0.3],
        [0.0, 0.45, 0.1, 0.05],
    ] {
        let (phi, d, beta) = (th[1], th[2], th[3]);
        let mut pi = vec![1.0; m + 1];
        for j in 1..=m {
            pi[j] = pi[j - 1] * (j as f64 - 1.0 - d) / j as f64;
        }
        let c: Vec<f64> = (0..=m)
            .map(|j| pi[j] - if j > 0 { phi * pi[j - 1] } else { 0.0 })
            .collect();
        let mut got = vec![0.0; m + 1];
        figarch_weights(&th, m, false, |j, lam, _, _, _| got[j] = lam);
        let mut bp = vec![1.0; m + 1];
        for i in 1..=m {
            bp[i] = bp[i - 1] * beta;
        }
        for j in 1..=m {
            let q: f64 = (0..=j).map(|i| c[j - i] * bp[i]).sum();
            let want = -q;
            assert!(
                (got[j] - want).abs() < 1e-14,
                "{th:?} j {j}: {} {want}",
                got[j]
            );
        }
        // the weights are non-negative and sum to less than 1
        let s: f64 = got.iter().sum();
        assert!(
            got[1..].iter().all(|l| *l >= -1e-15) && s < 1.0,
            "{th:?}: {s}"
        );
    }
}

/// The weights' derivatives against central differences of the weights.
#[test]
fn figarch_weight_derivatives_are_the_weights_slopes() {
    let m = 300;
    let th = [0.03, 0.2, 0.4, 0.5];
    let mut d = vec![[0.0; 3]; m + 1];
    figarch_weights(&th, m, true, |j, _, a, b, c| d[j] = [a, b, c]);
    for (p, idx) in [(0, 1), (1, 2), (2, 3)] {
        let h = 1e-6;
        let (mut up, mut dn) = (th, th);
        up[idx] += h;
        dn[idx] -= h;
        let mut lu = vec![0.0; m + 1];
        let mut ld = vec![0.0; m + 1];
        figarch_weights(&up, m, false, |j, l, _, _, _| lu[j] = l);
        figarch_weights(&dn, m, false, |j, l, _, _, _| ld[j] = l);
        for j in 1..=m {
            let num = (lu[j] - ld[j]) / (2.0 * h);
            assert!((d[j][p] - num).abs() < 1e-8 * (1.0 + num.abs()), "{p} {j}");
        }
    }
}

/// Each filter's variance at fixed parameters against its recursion
/// written out longhand from the definition, from `arch`'s backcast.
#[test]
fn each_filter_is_its_recursion() {
    let ys = gjr_series([0.03, 0.05, 0.1, 0.87], 400, 7, 1.0);
    let rows: Vec<[f64; 2]> = ys.iter().map(|y| [*y, 1.0]).collect();
    let bc = backcast(&rows);
    // GARCH and GJR
    for (vol, th) in [
        (GarchVol::Garch, vec![0.05, 0.08, 0.9]),
        (GarchVol::Gjr, vec![0.03, 0.05, 0.1, 0.87]),
    ] {
        let k = th.len();
        let mut f = Filter::new(vol, k, 0, bc);
        let (mut e2, mut asym, mut s) = (bc, 0.5 * bc, bc);
        let mut dl = vec![0.0; k];
        for y in &ys {
            let got = f.predict(vol, &th, &mut dl);
            let want = if k == 3 {
                th[0] + th[1] * e2 + th[2] * s
            } else {
                th[0] + th[1] * e2 + th[2] * asym + th[3] * s
            };
            assert!((got - want).abs() <= 1e-14 * want, "{vol:?}");
            f.push(vol, *y, got, &dl);
            (e2, asym, s) = (y * y, if *y < 0.0 { y * y } else { 0.0 }, want);
        }
    }
    // EGARCH
    let th = [0.0, 0.15, -0.08, 0.97];
    let mut f = Filter::new(GarchVol::Egarch, 4, 0, bc);
    let (mut z, mut l, mut fresh) = (0.0f64, bc.ln(), true);
    let mut dl = [0.0; 4];
    for y in &ys {
        let got = f.predict(GarchVol::Egarch, &th, &mut dl);
        let news = if fresh {
            0.0
        } else {
            th[1] * (z.abs() - (2.0 / std::f64::consts::PI).sqrt()) + th[2] * z
        };
        let want = (th[0] + news + th[3] * l).exp();
        assert!((got - want).abs() <= 1e-13 * want);
        f.push(GarchVol::Egarch, *y, got, &dl);
        (z, l, fresh) = (y / want.sqrt(), want.ln(), false);
    }
    // FIGARCH, the weights by the definition's expansion and the lags past
    // the first row the backcast
    let m = 50;
    let th = [0.03, 0.2, 0.4, 0.5];
    let mut lam = vec![0.0; m + 1];
    figarch_weights(&th, m, false, |j, l, _, _, _| lam[j] = l);
    let mut f = Filter::new(GarchVol::Figarch, 4, m, bc);
    let mut dl = [0.0; 4];
    for (t, y) in ys.iter().enumerate() {
        let got = f.predict(GarchVol::Figarch, &th, &mut dl);
        let want = th[0] / (1.0 - th[3])
            + (1..=m)
                .map(|j| lam[j] * if j <= t { ys[t - j] * ys[t - j] } else { bc })
                .sum::<f64>();
        assert!((got - want).abs() <= 1e-13 * want, "{t}");
        f.push(GarchVol::Figarch, *y, got, &dl);
    }
}

/// Each filter's sensitivity `∂ln s/∂θ` against central differences of
/// its variance path, at every row.
#[test]
fn each_sensitivity_is_its_variances_slope() {
    let ys = gjr_series([0.03, 0.05, 0.1, 0.87], 300, 8, 1.0);
    let rows: Vec<[f64; 2]> = ys.iter().map(|y| [*y, 1.0]).collect();
    let bc = backcast(&rows);
    for (vol, th, m) in [
        (GarchVol::Garch, vec![0.05, 0.08, 0.9], 0),
        (GarchVol::Gjr, vec![0.03, 0.05, 0.1, 0.87], 0),
        (GarchVol::Egarch, vec![-0.01, 0.15, -0.08, 0.97], 0),
        (GarchVol::Figarch, vec![0.03, 0.2, 0.4, 0.5], 60),
    ] {
        let k = th.len();
        let path = |th: &[f64]| -> Vec<(f64, Vec<f64>)> {
            let mut f = Filter::new(vol, k, m, bc);
            let mut dl = vec![0.0; k];
            ys.iter()
                .map(|y| {
                    let s = f.predict(vol, th, &mut dl);
                    f.push(vol, *y, s, &dl);
                    (s, dl.clone())
                })
                .collect()
        };
        let at = path(&th);
        for p in 0..k {
            let h = 1e-6 * th[p].abs().max(1e-2);
            let (mut up, mut dn) = (th.clone(), th.clone());
            up[p] += h;
            dn[p] -= h;
            let (pu, pd) = (path(&up), path(&dn));
            for t in 0..ys.len() {
                let num = (pu[t].0.ln() - pd[t].0.ln()) / (2.0 * h);
                let got = at[t].1[p];
                assert!(
                    (got - num).abs() < 1e-5 * (1.0 + num.abs()),
                    "{vol:?} p {p} t {t}: {got} {num}"
                );
            }
        }
    }
}

/// The rows of the start report nothing; the first after it reports; the
/// variance a row reports does not read the row's own residual (hard rule
/// 2), and its standardized residual does.
#[test]
fn a_row_is_predicted_before_it_is_learned() {
    for c in every_cfg() {
        let n0 = start(&c);
        let ys = series_of(&c, n0 + 200, 11);
        let mut m = Garch::new(c.clone()).unwrap();
        let out = run(&mut m, &ys[..n0 + 100], &vec![1.0; n0 + 100]);
        let f = first_report(&out);
        assert!(
            f >= c.start_rows && f.is_multiple_of(c.start_rows),
            "{c:?}: {f}"
        );
        assert!(out[..f].iter().all(|r| r[1].is_nan()), "{c:?}");
        assert!(out[f..].iter().all(|r| r[1].is_finite()), "{c:?}");
        let mut twin = m.clone();
        let y = ys[n0 + 100];
        let a = m.step(&[], &[Some(y)], 1.0, 1.0);
        let b = twin.step(&[], &[Some(100.0 * y)], 1.0, 1.0);
        assert_eq!(a.pred[0].to_bits(), b.pred[0].to_bits(), "{c:?}");
        assert!((b.pred[1] - 100.0 * a.pred[1]).abs() < 1e-9 * b.pred[1].abs());
        // predict is the step without the step
        let p = m.predict_with(&[], &[Some(ys[n0 + 101])], 1.0);
        let s = m.step(&[], &[Some(ys[n0 + 101])], 1.0, 1.0);
        assert_eq!(bits(&[p.pred]), bits(&[s.pred]));
        assert_eq!(p.n_eff, s.n_eff);
    }
}

/// A row of weight 0, or with its residual absent, steps nothing: the
/// stream with such rows and the stream without them report the same on
/// every row they share, to the bit, and end in the same state.
#[test]
fn a_zero_weight_row_steps_no_recursion() {
    for c in every_cfg() {
        let ys = series_of(&c, start(&c) + 400, 12);
        let mut plain = Garch::new(c.clone()).unwrap();
        let mut gappy = Garch::new(c.clone()).unwrap();
        // a zero-weight row first, then one every 7 rows (before, inside
        // and after the start), wild ones among them, and absent residuals
        let z = gappy.step(&[], &[Some(5.0)], 1.0, 0.0);
        assert!(z.pred[0].is_nan() && z.n_eff == 0.0);
        for (t, y) in ys.iter().enumerate() {
            if t % 7 == 3 {
                gappy.step(&[], &[Some(1e6 * (t as f64))], 1.0, 0.0);
                gappy.step(&[], &[None], 1.0, 1.0);
            }
            let a = plain.step(&[], &[Some(*y)], 1.0, 1.0);
            let b = gappy.step(&[], &[Some(*y)], 1.0, 1.0);
            assert_eq!(bits(&[a.pred]), bits(&[b.pred]), "{c:?} row {t}");
        }
        assert_eq!(state_bits(&plain).len(), state_bits(&gappy).len(), "{c:?}");
        let (p, g) = (plain.series.clone(), gappy.series.clone());
        assert_eq!(
            rmp_serde::to_vec(&p).unwrap(),
            rmp_serde::to_vec(&g).unwrap(),
            "{c:?}"
        );
    }
}

/// Saved and restored at every phase -- empty, mid-start, just started,
/// mid-stream -- a model goes on to the bit.
#[test]
fn a_saved_state_goes_on_to_the_bit() {
    for c in every_cfg() {
        let n0 = start(&c);
        let ys = series_of(&c, n0 + 300, 13);
        for cut in [0, n0 / 2, n0, n0 + 150] {
            let mut m = Garch::new(c.clone()).unwrap();
            run(&mut m, &ys[..cut], &vec![1.0; cut]);
            let bytes = rmp_serde::to_vec_named(&m.state()).unwrap();
            let state: State = rmp_serde::from_slice(&bytes).unwrap();
            let mut back = Garch::restore(&state).unwrap();
            let rest = &ys[cut..];
            let a = run(&mut m, rest, &vec![1.0; rest.len()]);
            let b = run(&mut back, rest, &vec![1.0; rest.len()]);
            assert_eq!(bits(&a), bits(&b), "{c:?} cut {cut}");
        }
    }
}

/// Scaling the returns by `c` scales every variance by `c²` and leaves the
/// standardized residuals and the dynamics alone; scaling every weight
/// changes nothing but `weight_sum`.
#[test]
fn the_fit_is_equivariant_in_the_scale_and_the_weights() {
    for c in every_cfg() {
        let n0 = start(&c);
        let ys = series_of(&c, n0 + 1000, 14);
        let mut base = Garch::new(c.clone()).unwrap();
        let a = run(&mut base, &ys, &vec![1.0; ys.len()]);
        // by powers of two, which every product carries exactly: what is
        // left is the logarithm's shift and the stopping rule's `1 + |f|`
        let scaled: Vec<f64> = ys.iter().map(|y| y * 1024.0).collect();
        let mut big = Garch::new(c.clone()).unwrap();
        let b = run(&mut big, &scaled, &vec![1.0; ys.len()]);
        let mut heavy = Garch::new(c.clone()).unwrap();
        let w = run(&mut heavy, &ys, &vec![4.0; ys.len()]);
        let f = first_report(&a);
        assert_eq!((first_report(&b), first_report(&w)), (f, f), "{c:?}");
        // EGARCH's level is a logarithm, which a scale shifts, and
        // FIGARCH's start mixes ω with the rest in its projection's metric:
        // their starts' differences, stopping rules and projections read
        // them a few digits differently
        let tol = match c.vol {
            GarchVol::Egarch | GarchVol::Figarch => 1e-4,
            _ => 1e-6,
        };
        for t in f..ys.len() {
            let (s, sb, sw) = (a[t][0], b[t][0], w[t][0]);
            assert!(
                (sb / 1_048_576.0 / s - 1.0).abs() < tol,
                "{c:?} {t}: {s} {sb}"
            );
            assert!((sw / s - 1.0).abs() < tol, "{c:?} {t}: {s} {sw}");
        }
        let (ta, tb) = (&base.coefficients()[0], &big.coefficients()[0]);
        for p in 1..ta.len() {
            assert!(
                (ta[p] - tb[p]).abs() < tol * (1.0 + ta[p].abs()),
                "{c:?} {p}"
            );
        }
    }
}

/// On a long GARCH(1,1) and a GJR series the online fit settles near the
/// process that made it (the Python oracle tests hold it to `arch`'s batch
/// fit over seeds).
#[test]
fn the_fit_finds_the_process() {
    let ys = gjr_series([0.05, 0.08, 0.0, 0.9], 20_000, 15, 1.0);
    for c in [
        cfg(GarchVol::Garch, GarchDist::Normal),
        GarchCfg {
            variance_targeting: true,
            ..cfg(GarchVol::Garch, GarchDist::Normal)
        },
    ] {
        let mut m = Garch::new(c).unwrap();
        run(&mut m, &ys, &vec![1.0; ys.len()]);
        let th = &m.coefficients()[0];
        assert!(
            (th[1] - 0.08).abs() < 0.03 && (th[2] - 0.9).abs() < 0.04,
            "{th:?}"
        );
    }
    let ys = gjr_series([0.03, 0.05, 0.1, 0.87], 20_000, 16, 1.0);
    let mut m = Garch::new(cfg(GarchVol::Gjr, GarchDist::Normal)).unwrap();
    run(&mut m, &ys, &vec![1.0; ys.len()]);
    let th = &m.coefficients()[0];
    assert!(
        (th[2] - 0.1).abs() < 0.05 && (th[3] - 0.87).abs() < 0.05,
        "{th:?}"
    );
}

/// The variance's level is identified by the plain score with a free `ω`:
/// a model started at a quarter of the series' variance finds it again.
/// (A clipped score centred on its own running mean would not: it has mean
/// 0 at every level, which is why the clip is used only under targeting.)
#[test]
fn a_free_omega_finds_the_level_from_a_wrong_start() {
    let ys = gjr_series([0.05, 0.08, 0.0, 0.9], 30_500, 30, 1.0);
    let mut m = Garch::new(cfg(GarchVol::Garch, GarchDist::Normal)).unwrap();
    // the start on rows a half the size, then the series as it is
    let low: Vec<f64> = ys[..500].iter().map(|y| 0.5 * y).collect();
    run(&mut m, &low, &[1.0; 500]);
    run(&mut m, &ys[500..], &vec![1.0; 30_000]);
    let th = &m.coefficients()[0];
    let level = th[0] / (1.0 - th[1] - th[2]);
    assert!(
        (level - 2.5).abs() < 0.4,
        "{th:?}: the long-run variance {level}, the truth 2.5"
    );
}

/// EGARCH's steps keep `α ≥ |γ|` and `α ≤ 1` on every row after the start,
/// through a wild row, and its variance a number.
#[test]
fn egarch_stays_invertible() {
    let c = cfg(GarchVol::Egarch, GarchDist::Normal);
    let mut ys = series_of(&c, 3000, 31);
    ys[1200] = 300.0;
    let mut m = Garch::new(c).unwrap();
    for (t, y) in ys.iter().enumerate() {
        m.step(&[], &[Some(*y)], 1.0, 1.0);
        if t >= 500 {
            let th = &m.coefficients()[0];
            assert!(
                th[1] >= th[2].abs() - 1e-12 && th[1] <= 1.0 + 1e-12,
                "{t}: {th:?}"
            );
        }
    }
    assert!(m.invertibility()[0].is_finite());
}

/// A start that does not settle -- here every residual the same size, so
/// the fit's `α` sits on its bound at 0 -- doubles its rows and refits,
/// twice, then steps from its last fit and says so.
#[test]
fn a_start_on_a_bound_doubles_and_refits() {
    let c = cfg(GarchVol::Gjr, GarchDist::Normal);
    let n0 = c.start_rows;
    let ys: Vec<f64> = (0..4 * n0 + 10)
        .map(|t| if t % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
    let mut m = Garch::new(c).unwrap();
    for (t, y) in ys.iter().enumerate() {
        m.step(&[], &[Some(*y)], 1.0, 1.0);
        let (refits, _) = m.start_report()[0];
        let want = if t + 1 < n0 {
            0
        } else if t + 1 < 2 * n0 {
            1
        } else {
            2
        };
        assert_eq!(refits, want.min(2), "{t}");
        assert_eq!(m.started()[0], t + 1 >= 4 * n0, "{t}");
    }
    assert_eq!(m.start_report()[0], (2, false));
}

/// A row a million standard deviations out, and a run of zero residuals,
/// leave every form's state finite and reporting.
#[test]
fn wild_rows_leave_the_state_finite() {
    for c in every_cfg() {
        let n0 = start(&c);
        let mut ys = series_of(&c, n0 + 700, 17);
        ys[n0 + 200] = 1e6;
        for y in &mut ys[n0 + 300..n0 + 350] {
            *y = 0.0;
        }
        let mut m = Garch::new(c.clone()).unwrap();
        let out = run(&mut m, &ys, &vec![1.0; ys.len()]);
        assert!(first_report(&out) <= n0, "{c:?}");
        assert!(m.coefficients()[0].iter().all(|v| v.is_finite()), "{c:?}");
    }
}

/// The forecast against its recursion written out: `ω + P f` for GARCH and
/// GJR, and FIGARCH's weights over the forecasts and the ring, longhand.
#[test]
fn the_forecast_is_the_variance_recursion_run_ahead() {
    for vol in [GarchVol::Garch, GarchVol::Gjr, GarchVol::Figarch] {
        let mut c = cfg(vol, GarchDist::Normal);
        c.horizon = 5;
        let n0 = start(&c);
        let ys = gjr_series([0.05, 0.08, 0.0, 0.9], n0 + 200, 18, 1.0);
        let mut m = Garch::new(c).unwrap();
        run(&mut m, &ys[..n0 + 150], &vec![1.0; n0 + 150]);
        assert!(m.started()[0]);
        let out = m.predict(&[], 1.0).pred;
        let (s, fc) = (out[0], out[2]);
        let th = m.coefficients()[0].clone();
        let want = if vol == GarchVol::Figarch {
            let mm = 100;
            let mut lam = vec![0.0; mm + 1];
            figarch_weights(&th, mm, false, |j, l, _, _, _| lam[j] = l);
            let ring = &m.series[0].filter;
            let back = |j: usize| ring.ring[(ring.head + j - 1) % mm];
            let mut f = vec![s];
            for i in 1..5 {
                let mut h = th[0] / (1.0 - th[3]);
                for j in 1..=mm {
                    h += lam[j] * if j <= i { f[i - j] } else { back(j - i) };
                }
                f.push(h);
            }
            f[4]
        } else {
            let p = if vol == GarchVol::Gjr {
                th[1] + 0.5 * th[2] + th[3]
            } else {
                th[1] + th[2]
            };
            (0..4).fold(s, |f, _| th[0] + p * f)
        };
        assert!((fc - want).abs() < 1e-12 * want, "{vol:?}: {fc} {want}");
        // the forecast reverts toward the long-run variance
        assert!(fc.is_finite() && fc > 0.0);
    }
}

/// The state's bytes are the formula's, and a FIGARCH past the cap, or a
/// cap past 64 MiB, is refused at the call.
#[test]
fn the_state_is_bounded_by_its_settings() {
    let c = cfg(GarchVol::Garch, GarchDist::Normal);
    assert_eq!(Garch::series_bytes(&c, 0), 168);
    assert_eq!(Garch::series_bytes(&c, 500), 168 + 8000);
    let t = GarchCfg {
        variance_targeting: true,
        ..c.clone()
    };
    assert_eq!(Garch::series_bytes(&t, 0), 144);
    let mut f = cfg(GarchVol::Figarch, GarchDist::Normal);
    f.truncation = None;
    f.max_state_bytes = None;
    // the ring of 1,000 lags, and the start's 2,000 rows held before it
    assert_eq!(Garch::series_bytes(&f, 0), 216 + 8000);
    assert_eq!(Garch::series_bytes(&f, 2000), 216 + 32_000);
    assert_eq!(
        Garch::series_bytes(&cfg(GarchVol::Gjr, GarchDist::T), 0),
        264
    );
    assert_eq!(
        Garch::series_bytes(&cfg(GarchVol::Egarch, GarchDist::Normal), 0),
        216 + 512
    );
    // the largest M a group of one target holds under the default cap
    f.truncation = Some(32_000);
    assert!(Garch::new(f.clone()).is_ok());
    f.truncation = Some(40_000);
    let e = Garch::new(f.clone()).unwrap_err();
    assert!(
        e.contains("past max_state_bytes of 262144") && e.contains("truncation"),
        "{e}"
    );
    f.max_state_bytes = Some(1 << 20);
    assert!(Garch::new(f.clone()).is_ok());
    f.max_state_bytes = Some((64 << 20) + 1);
    assert!(Garch::new(f).unwrap_err().contains("at most 67108864"));
    // the state holds what the formula says, after the start and before
    let ys = gjr_series([0.05, 0.08, 0.0, 0.9], 9000, 19, 1.0);
    let mut m = Garch::new(cfg(GarchVol::Figarch, GarchDist::Normal)).unwrap();
    run(&mut m, &ys[..300], &[1.0; 300]);
    assert_eq!(m.bytes(), 216 + 16 * 300 + 16);
    run(&mut m, &ys[300..], &[1.0; 8700]);
    assert!(m.started()[0]);
    assert_eq!(m.bytes(), 216 + 8 * 100 + 16);
}

/// A start of a clock span buffers what the span holds, so its bytes are
/// read as the rows come: a warning past half of `max_state_bytes`, once,
/// and a refusal at it.
#[test]
fn a_clock_span_start_is_warned_of_at_half_the_cap_and_stopped_at_it() {
    let mut c = cfg(GarchVol::Garch, GarchDist::Normal);
    c.start_span = Some(1e9);
    c.max_state_bytes = Some(4000);
    let mut m = Garch::new(c).unwrap();
    let ys = gjr_series([0.05, 0.08, 0.0, 0.9], 400, 20, 1.0);
    let mut warned_at = None;
    let mut over_at = None;
    for (t, y) in ys.iter().enumerate() {
        m.step(&[], &[Some(*y)], 1.0, 1.0);
        if warned_at.is_none() && m.take_state_warning().is_some() {
            warned_at = Some(t);
        }
        if over_at.is_none() && m.state_over_cap().is_some() {
            over_at = Some(t);
        }
    }
    // a row is held to 168 + 16 + 16 (r + 1) bytes with r before it: past
    // 2,000 at r = 113, past 4,000 at r = 238
    assert_eq!(warned_at, Some(113));
    assert_eq!(over_at, Some(238));
    assert!(m.take_state_warning().is_none(), "warned once");
    assert!(m.bytes() <= 4000);
}

/// A start of a clock span closes at the first row at or past the span
/// from its first, once it holds 50 rows, on the stamps handed; a refit
/// waits for twice the span.
#[test]
fn a_clock_span_start_closes_at_its_span() {
    let ys = gjr_series([0.05, 0.08, 0.0, 0.9], 1000, 21, 1.0);
    // (rows apart, the span): the rows a fit is made on, at each refit
    for (apart, span, want) in [(2.0, 100.0, [51, 101, 201]), (5.0, 100.0, [50, 51, 81])] {
        let mut c = cfg(GarchVol::Garch, GarchDist::Normal);
        c.start_span = Some(span);
        let mut m = Garch::new(c).unwrap();
        let mut fits = Vec::new();
        let mut last = (0, false);
        for (t, y) in ys.iter().enumerate() {
            m.stamp_next(Stamp::from(apart * t as f64));
            m.step(&[], &[Some(*y)], apart, 1.0);
            let now = (m.start_report()[0].0, m.started()[0]);
            if now != last {
                fits.push(t + 1);
                last = now;
            }
            if now.1 {
                break;
            }
        }
        // every decision falls at a span or its 50th row, whichever is later
        let n = fits.len();
        assert_eq!(fits[..], want[..n], "{apart}: {fits:?}");
    }
}

#[test]
fn a_state_of_the_wrong_shape_is_refused() {
    let ys = t_series(900, 22, 1.0);
    let mut m = Garch::new(GarchCfg {
        start_rows: 100,
        ..cfg(GarchVol::Gjr, GarchDist::T)
    })
    .unwrap();
    run(&mut m, &ys, &[1.0; 900]);
    assert!(m.started()[0]);
    for cut in 0..4 {
        let mut bad = m.clone();
        match cut {
            0 => bad.series[0].theta.pop().map(|_| ()).unwrap(),
            1 => bad.series[0].h.push(0.0),
            2 => bad.series[0].filter.sens.clear(),
            _ => bad.series.push(bad.series[0].clone()),
        }
        let e = Garch::restore(&bad.state()).unwrap_err().to_string();
        assert!(e.contains("wrong shape"), "{cut}: {e}");
    }
}

#[test]
fn the_settings_are_checked() {
    let refused = |f: &dyn Fn(&mut GarchCfg), word: &str| {
        let mut c = cfg(GarchVol::Garch, GarchDist::Normal);
        f(&mut c);
        let e = Garch::new(c).unwrap_err();
        assert!(e.contains(word), "{word}: {e}");
    };
    refused(&|c| c.n_targets = 0, "at least one target");
    refused(
        &|c| {
            c.vol = GarchVol::Egarch;
            c.dist = GarchDist::T
        },
        "takes vol",
    );
    refused(
        &|c| {
            c.vol = GarchVol::Figarch;
            c.dist = GarchDist::Skewt
        },
        "takes vol",
    );
    refused(&|c| c.truncation = Some(10), "truncation is FIGARCH's");
    refused(
        &|c| {
            c.vol = GarchVol::Figarch;
            c.truncation = Some(0)
        },
        "at least 1",
    );
    refused(&|c| c.start_rows = 49, "at least 50");
    refused(
        &|c| {
            c.vol = GarchVol::Figarch;
            c.start_rows = 1999
        },
        "at least 2000",
    );
    refused(&|c| c.start_span = Some(0.0), "start_span");
    refused(
        &|c| {
            c.vol = GarchVol::Gjr;
            c.dist = GarchDist::Skewt
        },
        "does not take vol",
    );
    refused(
        &|c| {
            c.vol = GarchVol::Gjr;
            c.variance_targeting = true
        },
        "variance_targeting",
    );
    refused(
        &|c| {
            c.dist = GarchDist::T;
            c.variance_targeting = true
        },
        "variance_targeting",
    );
    refused(&|c| c.horizon = 0, "horizon");
    refused(
        &|c| {
            c.vol = GarchVol::Egarch;
            c.horizon = 2
        },
        "one row ahead",
    );
    refused(&|c| c.min_weight = -1.0, "min_weight");
    refused(&|c| c.max_state_bytes = Some(0), "above 0");
}

/// `min_weight` gates each target on its own weight; `weight_sum` is every
/// row's weight before the row.
#[test]
fn min_weight_gates_each_target_on_its_own_weight() {
    let mut c = cfg(GarchVol::Garch, GarchDist::Normal);
    c.n_targets = 2;
    c.min_weight = 700.0;
    c.start_rows = 100;
    let ys = gjr_series([0.05, 0.08, 0.0, 0.9], 800, 23, 1.0);
    let mut m = Garch::new(c).unwrap();
    for (t, y) in ys.iter().enumerate() {
        // the second target misses every other row
        let second = (t % 2 == 0).then_some(*y);
        let s = m.step(&[], &[Some(*y), second], 1.0, 1.0);
        assert_eq!(s.n_eff, t as f64);
        assert_eq!(s.pred[0].is_nan(), t < 700, "{t}");
        assert!(s.pred[2].is_nan(), "{t}: the second has 400 rows of weight");
    }
    let mut w = Vec::new();
    assert!(m.target_n_eff_into(&mut w));
    assert_eq!(w, vec![800.0, 400.0]);
}
