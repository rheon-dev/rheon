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

/// A level of 3, a cycle of period 8 and one of period 3.2, and noise.
fn signal(n: usize, seed: u64) -> Vec<f64> {
    let mut s = seed;
    (0..n)
        .map(|i| {
            let t = i as f64;
            3.0 + (std::f64::consts::TAU * t / 8.0).sin()
                + 0.5 * (std::f64::consts::TAU * t / 3.2 + 1.0).sin()
                + normal(&mut s)
        })
        .collect()
}

fn cfg(freqs: Vec<f64>, decay: Decay) -> SpectralCfg {
    SpectralCfg {
        n_targets: 1,
        frequencies: freqs,
        spacing: 1.0,
        decay,
        scaling: SpectralScaling::Density,
        detrend: true,
        min_weight: 0.0,
        max_state_bytes: None,
        window: None,
        window_every: None,
        max_rows_between_snapshots: None,
    }
}

#[allow(clippy::too_many_arguments)]
/// The definition written out over the rows, by a direct sum with every
/// amplitude `w λ^{age}` and every phase `2π f τ` taken from the row's own
/// clock, sharing no recursion with the model; `trig` is the `sin`/`cos`.
fn longhand(
    f: f64,
    x: &[f64],
    tau: &[f64],
    w: &[f64],
    h: f64,
    spacing: f64,
    scaling: SpectralScaling,
    trig: fn(f64) -> (f64, f64),
) -> f64 {
    let t = *tau.last().unwrap();
    let a: Vec<f64> = (0..x.len())
        .map(|s| w[s] * (-(t - tau[s]) / h).exp2())
        .collect();
    let (sa, sax, saa) = (
        crate::oracle::sum(a.iter().copied()),
        crate::oracle::sum(a.iter().zip(x).map(|(a, x)| a * x)),
        crate::oracle::sum(a.iter().map(|a| a * a)),
    );
    let m = sax / sa;
    let (mut re, mut im) = (0.0, 0.0);
    for s in 0..x.len() {
        let (sn, cs) = trig(std::f64::consts::TAU * f * tau[s]);
        re += a[s] * (x[s] - m) * cs;
        im -= a[s] * (x[s] - m) * sn;
    }
    let nyq = 0.5 / spacing;
    let c = if f == 0.0 || f == nyq { 1.0 } else { 2.0 };
    match scaling {
        SpectralScaling::Density => c * (re * re + im * im) * spacing / saa,
        SpectralScaling::Spectrum => c * (re * re + im * im) / (sa * sa),
        SpectralScaling::LombScargle => unreachable!("written out in ls_longhand"),
    }
}

/// The model's own trigonometry, `rotation`, as the longhand reads it.
fn platform(t: f64) -> (f64, f64) {
    let [c, s] = rotation(t);
    (s, c)
}

/// The recursion is the definition, on an even clock and on an uneven one
/// with weights, under decay and without, both scalings, at the DC, at
/// frequencies on and off the cycles and at the Nyquist frequency.
#[test]
fn the_recursion_is_the_definition() {
    let n = 2000;
    let x = signal(n, 3);
    let mut s = 5;
    let freqs = vec![0.0, 0.05, 0.125, 1.0 / 3.2, 0.4, 0.5];
    for uneven in [false, true] {
        let steps: Vec<f64> = (0..n)
            .map(|i| {
                if i == 0 {
                    0.0
                } else if uneven {
                    0.5 + lcg(&mut s)
                } else {
                    1.0
                }
            })
            .collect();
        let w: Vec<f64> = (0..n)
            .map(|_| if uneven { 0.5 + lcg(&mut s) } else { 1.0 })
            .collect();
        let tau: Vec<f64> = steps
            .iter()
            .scan(0.0, |c, d| {
                *c += d;
                Some(*c)
            })
            .collect();
        for h in [f64::INFINITY, 150.0] {
            for scaling in [SpectralScaling::Density, SpectralScaling::Spectrum] {
                let mut m = Spectral::new(SpectralCfg {
                    scaling,
                    ..cfg(freqs.clone(), Decay::Halflife(h))
                })
                .unwrap();
                for i in 0..n {
                    m.step(&[], &[Some(x[i])], steps[i], w[i]);
                }
                let got = &m.powers()[0];
                for (k, f) in freqs.iter().enumerate() {
                    let want = longhand(*f, &x, &tau, &w, h, 1.0, scaling, platform);
                    let tol = 1e-9
                        * want
                            .abs()
                            .max(1e-6 * got.iter().fold(0.0_f64, |a, v| a.max(*v)));
                    assert!(
                        (got[k] - want).abs() <= tol,
                        "uneven {uneven}, h {h}, {scaling:?}, f {f}: {} vs {want}",
                        got[k]
                    );
                }
            }
        }
    }
}

/// The cycles are found: at their frequencies the power stands far above
/// the noise's, and off them it does not.
#[test]
fn the_cycles_stand_out() {
    let x = signal(4000, 7);
    let mut m = Spectral::new(cfg(
        vec![0.05, 0.125, 1.0 / 3.2, 0.2],
        Decay::Halflife(1000.0),
    ))
    .unwrap();
    for v in &x {
        m.step(&[], &[Some(*v)], 1.0, 1.0);
    }
    let p = &m.powers()[0];
    assert!(p[1] > 20.0 * p[0] && p[1] > 20.0 * p[3], "{p:?}");
    assert!(p[2] > 5.0 * p[0] && p[2] > 5.0 * p[3], "{p:?}");
}

/// The transform's trigonometry is the `libm` crate's, so its state is the
/// same bits on every OS. The platform's last bit was measured to move the
/// power by up to 140,089 ulps over 20,000 rows with no decay (a one-ulp
/// offset in the rotation on every row of an even clock), so no platform
/// library may creep back in.
#[test]
fn the_trigonometry_is_libms() {
    let mut s = 21;
    for _ in 0..10_000 {
        let th = (lcg(&mut s) - 0.5) * 1e3;
        let [c, si] = rotation(th);
        assert_eq!(
            (c.to_bits(), si.to_bits()),
            (libm::cos(th).to_bits(), libm::sin(th).to_bits())
        );
    }
}

/// A break empties the sums, the weight stays; a row of weight 0 learns
/// nothing but the clock; a saved state goes on to the bit.
#[test]
fn breaks_zero_weights_and_saves() {
    let x = signal(300, 17);
    let mut m = Spectral::new(cfg(vec![0.125, 0.25], Decay::Halflife(50.0))).unwrap();
    for v in &x[..150] {
        m.step(&[], &[Some(*v)], 1.0, 1.0);
    }
    let w = m.n_eff();
    m.clear_lags();
    assert_eq!(m.n_eff(), w);
    assert!(m.powers()[0].iter().all(|p| p.is_nan()));
    let mut zero = m.clone();
    zero.step(&[], &[Some(1e9)], 0.0, 0.0);
    assert!(zero.powers()[0].iter().all(|p| p.is_nan()));
    let bytes = rmp_serde::to_vec(&m.state()).unwrap();
    let mut back = Spectral::restore(&rmp_serde::from_slice(&bytes).unwrap()).unwrap();
    for v in &x[150..] {
        let a = m.step(&[], &[Some(*v)], 1.0, 1.0);
        let b = back.step(&[], &[Some(*v)], 1.0, 1.0);
        assert_eq!(
            a.pred.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            b.pred.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }
    let fresh = rmp_serde::to_vec(&m.state()).unwrap().len() as u64;
    assert!(fresh <= Spectral::state_bytes(m.cfg()) + 400, "{fresh}");
}

/// Its settings, refused by name, on both sides of each limit.
#[test]
fn it_refuses_its_settings() {
    let base = cfg(vec![0.5], Decay::Halflife(10.0));
    assert!(
        Spectral::new(base.clone()).is_ok(),
        "the Nyquist itself is taken"
    );
    let err = |c: SpectralCfg| Spectral::new(c).unwrap_err();
    assert!(
        err(SpectralCfg {
            frequencies: vec![0.5000001],
            ..base.clone()
        })
        .contains("Nyquist")
    );
    assert!(
        err(SpectralCfg {
            frequencies: vec![],
            ..base.clone()
        })
        .contains("at least one")
    );
    assert!(
        err(SpectralCfg {
            spacing: 0.0,
            ..base.clone()
        })
        .contains("spacing")
    );
    let many = vec![0.1; MAX_FREQUENCIES];
    assert!(
        Spectral::new(SpectralCfg {
            frequencies: many.clone(),
            max_state_bytes: Some(64 << 20),
            ..base.clone()
        })
        .is_ok()
    );
    let mut more = many;
    more.push(0.1);
    assert!(
        err(SpectralCfg {
            frequencies: more,
            max_state_bytes: Some(64 << 20),
            ..base.clone()
        })
        .contains("at most")
    );
    let c = cfg(vec![0.1, 0.2], Decay::Halflife(10.0));
    let b = Spectral::state_bytes(&c);
    assert!(
        Spectral::new(SpectralCfg {
            max_state_bytes: Some(b),
            ..c.clone()
        })
        .is_ok()
    );
    assert!(
        err(SpectralCfg {
            max_state_bytes: Some(b - 1),
            ..c
        })
        .contains("max_state_bytes")
    );
}

/// The Lomb-Scargle power written out: the weighted least squares of `x`
/// on `cos θ`, `sin θ` (and a constant with `detrend`), `θ = 2π f τ`, by
/// faer's LU (`crate::oracle`), with every amplitude `w λ^{age}`; its
/// explained share of the weighted sum of squares, centred with `detrend`.
fn ls_longhand(f: f64, x: &[f64], tau: &[f64], w: &[f64], h: f64, detrend: bool) -> f64 {
    let t = *tau.last().unwrap();
    let a: Vec<f64> = (0..x.len())
        .map(|s| w[s] * (-(t - tau[s]) / h).exp2())
        .collect();
    let cols = |s: usize| {
        let th = std::f64::consts::TAU * f * tau[s];
        let mut z = vec![th.cos(), th.sin()];
        if detrend {
            z.push(1.0);
        }
        z
    };
    let k = if detrend { 3 } else { 2 };
    let mut g = vec![0.0; k * k];
    let mut b = vec![0.0; k];
    for i in 0..k {
        for j in 0..k {
            g[i * k + j] = crate::oracle::sum((0..x.len()).map(|s| a[s] * cols(s)[i] * cols(s)[j]));
        }
        b[i] = crate::oracle::sum((0..x.len()).map(|s| a[s] * cols(s)[i] * x[s]));
    }
    let beta = crate::oracle::solve(&g, &b);
    let rss = crate::oracle::sum((0..x.len()).map(|s| {
        let e = x[s] - cols(s).iter().zip(&beta).map(|(z, b)| z * b).sum::<f64>();
        a[s] * e * e
    }));
    let m = if detrend {
        crate::oracle::sum((0..x.len()).map(|s| a[s] * x[s]))
            / crate::oracle::sum(a.iter().copied())
    } else {
        0.0
    };
    let tss = crate::oracle::sum((0..x.len()).map(|s| a[s] * (x[s] - m) * (x[s] - m)));
    1.0 - rss / tss
}

/// Lomb-Scargle is its definition: on an uneven clock with weights, under
/// decay and without, with and without `detrend`, at frequencies on and off
/// the cycle and past the even clock's Nyquist frequency; between 0 and 1,
/// and high on the clock's cycle.
#[test]
fn lomb_scargle_is_the_weighted_least_squares_of_a_sinusoid() {
    let n = 1500;
    let noise = signal(n, 23);
    let mut s = 31;
    let steps: Vec<f64> = (0..n)
        .map(|i| if i == 0 { 0.0 } else { 0.2 + 1.6 * lcg(&mut s) })
        .collect();
    let w: Vec<f64> = (0..n).map(|_| 0.5 + lcg(&mut s)).collect();
    let tau: Vec<f64> = steps
        .iter()
        .scan(0.0, |c, d| {
            *c += d;
            Some(*c)
        })
        .collect();
    // A cycle of period 8 on the clock, under the noise's own level and
    // row-indexed cycles.
    let x: Vec<f64> = (0..n)
        .map(|i| noise[i] + 2.0 * (std::f64::consts::TAU * tau[i] / 8.0).sin())
        .collect();
    let freqs = vec![0.05, 0.125, 1.0 / 3.2, 0.4, 0.9];
    for h in [f64::INFINITY, 200.0] {
        for detrend in [true, false] {
            let mut m = Spectral::new(SpectralCfg {
                scaling: SpectralScaling::LombScargle,
                detrend,
                ..cfg(freqs.clone(), Decay::Halflife(h))
            })
            .unwrap();
            for i in 0..n {
                m.step(&[], &[Some(x[i])], steps[i], w[i]);
            }
            let got = &m.powers()[0];
            for (k, f) in freqs.iter().enumerate() {
                let want = ls_longhand(*f, &x, &tau, &w, h, detrend);
                assert!(
                    (got[k] - want).abs() <= 1e-9 * want.max(1e-3),
                    "h {h}, detrend {detrend}, f {f}: {} vs {want}",
                    got[k]
                );
                assert!((0.0..=1.0).contains(&got[k]));
            }
            if detrend {
                assert!(got[1] > 0.2 && got[0] < 0.02, "{got:?}");
            }
        }
    }
}

/// The early rows are guarded by the fit's spread, not a count: one row
/// cannot tell a cosine from a sine, and its power is withheld; once the
/// phases spread, it is read. A target held at one value has no spread of
/// its own and is withheld throughout.
#[test]
fn lomb_scargle_withholds_an_undetermined_fit() {
    let ls = |detrend| {
        Spectral::new(SpectralCfg {
            scaling: SpectralScaling::LombScargle,
            detrend,
            ..cfg(vec![0.1], Decay::Halflife(f64::INFINITY))
        })
        .unwrap()
    };
    for detrend in [false, true] {
        let mut m = ls(detrend);
        let first = m.step(&[], &[Some(1.0)], 0.0, 1.0);
        assert!(first.pred[0].is_nan());
        m.step(&[], &[Some(2.0)], 1.0, 1.0);
        // two rows: without a constant, two columns and two phases apart
        // are determined; with one, the three columns are not
        assert_eq!(m.powers()[0][0].is_nan(), detrend, "detrend {detrend}");
        for (i, v) in [0.5, 3.0, -1.0].iter().enumerate() {
            m.step(&[], &[Some(*v)], 1.0 + 0.1 * i as f64, 1.0);
        }
        assert!(m.powers()[0][0].is_finite());
    }
    let mut flat = ls(true);
    for _ in 0..50 {
        flat.step(&[], &[Some(4.0)], 1.0, 1.0);
    }
    assert!(flat.powers()[0][0].is_nan());
}

/// Under Lomb-Scargle a period below twice `spacing` is taken, an even
/// clock's limit that does not apply, and an infinite one is refused; the
/// state holds `8 (6F + 4)` a target, and goes on to the bit.
#[test]
fn lomb_scargle_settings_and_state() {
    let c = SpectralCfg {
        scaling: SpectralScaling::LombScargle,
        ..cfg(vec![0.1, 3.0], Decay::Halflife(40.0))
    };
    let mut m = Spectral::new(c.clone()).unwrap();
    assert_eq!(Spectral::state_bytes(&c), 8 * (6 * 2 + 4) + 8);
    let e = Spectral::new(SpectralCfg {
        frequencies: vec![0.0],
        ..c.clone()
    })
    .unwrap_err();
    assert!(e.contains("lomb_scargle"), "{e}");
    let x = signal(200, 41);
    for v in &x[..100] {
        m.step(&[], &[Some(*v)], 0.7, 1.0);
    }
    let bytes = rmp_serde::to_vec(&m.state()).unwrap();
    let mut back = Spectral::restore(&rmp_serde::from_slice(&bytes).unwrap()).unwrap();
    for v in &x[100..] {
        let a = m.step(&[], &[Some(*v)], 1.3, 1.0);
        let b = back.step(&[], &[Some(*v)], 1.3, 1.0);
        assert_eq!(a.pred[1].to_bits(), b.pred[1].to_bits());
        assert_eq!(a.pred[0].to_bits(), b.pred[0].to_bits());
    }
    let len = rmp_serde::to_vec(&m.state()).unwrap().len() as u64;
    assert!(len <= Spectral::state_bytes(m.cfg()) + 400, "{len}");
}

/// A spectrum over a hard window of `w` clock units, a snapshot every row.
fn windowed(freqs: Vec<f64>, h: f64, w: f64, scaling: SpectralScaling) -> Spectral {
    Spectral::new(SpectralCfg {
        scaling,
        window: Some(w),
        ..cfg(freqs, Decay::Halflife(h))
    })
    .unwrap()
}

/// The clock of each row from its step.
fn clocks(steps: &[f64]) -> Vec<f64> {
    steps
        .iter()
        .scan(0.0, |c, d| {
            *c += d;
            Some(*c)
        })
        .collect()
}

/// Under a window the read is the definition over the rows inside it
/// alone, those less than `w` old (a row exactly `w` old out): on an uneven
/// clock with weights, under decay and without, for each scaling and for
/// Lomb-Scargle, at several ends of the stream, many windows in.
#[test]
fn a_window_is_the_definition_over_its_rows() {
    let n = 3000;
    let x = signal(n, 43);
    let mut s = 47;
    let steps: Vec<f64> = (0..n)
        .map(|i| if i == 0 { 0.0 } else { 0.5 + lcg(&mut s) })
        .collect();
    let wts: Vec<f64> = (0..n).map(|_| 0.5 + lcg(&mut s)).collect();
    let tau = clocks(&steps);
    let freqs = vec![0.05, 0.125, 1.0 / 3.2, 0.4];
    let w = 150.0;
    for h in [f64::INFINITY, 90.0] {
        for scaling in [
            SpectralScaling::Density,
            SpectralScaling::Spectrum,
            SpectralScaling::LombScargle,
        ] {
            let mut m = windowed(freqs.clone(), h, w, scaling);
            for i in 0..n {
                m.step(&[], &[Some(x[i])], steps[i], wts[i]);
                if !matches!(i, 120 | 1777 | 2999) {
                    continue;
                }
                let t = tau[i];
                let from = (0..=i).find(|&s| t - tau[s] < w).unwrap();
                let r = from..i + 1;
                let got = &m.powers()[0];
                for (k, f) in freqs.iter().enumerate() {
                    let want = match scaling {
                        SpectralScaling::LombScargle => ls_longhand(
                            *f,
                            &x[r.clone()],
                            &tau[r.clone()],
                            &wts[r.clone()],
                            h,
                            true,
                        ),
                        _ => longhand(
                            *f,
                            &x[r.clone()],
                            &tau[r.clone()],
                            &wts[r.clone()],
                            h,
                            1.0,
                            scaling,
                            platform,
                        ),
                    };
                    assert!(
                        (got[k] - want).abs() <= 1e-9 * want.abs().max(1e-6),
                        "h {h}, {scaling:?}, row {i}, f {f}: {} vs {want}",
                        got[k]
                    );
                }
                // The weight read is the window's (hard rule 8).
                let a: f64 = r.clone().map(|s| wts[s] * (-(t - tau[s]) / h).exp2()).sum();
                assert!((m.n_eff() - a).abs() <= 1e-9 * a, "{} vs {a}", m.n_eff());
            }
        }
    }
}

/// Under a window a saved state goes on to the bit, and a coarse
/// cadence's boundary never keeps a row older than the window: its read is
/// the definition over the rows after its boundary, which are all inside.
#[test]
fn a_window_saves_and_its_coarse_boundary_keeps_no_older_row() {
    let n = 900;
    let x = signal(n, 53);
    let mut m = Spectral::new(SpectralCfg {
        window: Some(100.0),
        window_every: Some(7.0),
        ..cfg(vec![0.125, 0.3], Decay::Halflife(200.0))
    })
    .unwrap();
    for v in &x[..400] {
        m.step(&[], &[Some(*v)], 1.0, 1.0);
    }
    let bytes = rmp_serde::to_vec(&m.state()).unwrap();
    let mut back = Spectral::restore(&rmp_serde::from_slice(&bytes).unwrap()).unwrap();
    for v in &x[400..] {
        let a = m.step(&[], &[Some(*v)], 1.0, 1.0);
        let b = back.step(&[], &[Some(*v)], 1.0, 1.0);
        assert_eq!(
            a.pred.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            b.pred.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        assert_eq!(a.n_eff.to_bits(), b.n_eff.to_bits());
    }
    // Rows 1 a clock unit apart: the window holds at most 100 of them and,
    // with a snapshot every 7, at least 100 - 7.
    let held = m.n_eff();
    let full: f64 = (0..100).map(|a| (-(a as f64) / 200.0).exp2()).sum();
    let least: f64 = (0..93).map(|a| (-(a as f64) / 200.0).exp2()).sum();
    assert!(
        held <= full * (1.0 + 1e-12) && held >= least * (1.0 - 1e-12),
        "{held}"
    );
}

/// A break under a window empties the transform, the boundary's sums
/// with it, and the rows after it are read alone; the shared weight stays.
#[test]
fn a_break_under_a_window_starts_the_transform_again() {
    let x = signal(400, 59);
    let mut m = windowed(
        vec![0.125, 0.25],
        f64::INFINITY,
        60.0,
        SpectralScaling::Spectrum,
    );
    for v in &x[..200] {
        m.step(&[], &[Some(*v)], 1.0, 1.0);
    }
    m.clear_lags();
    assert!(m.powers()[0].iter().all(|p| p.is_nan()));
    let tau: Vec<f64> = (0..400).map(|i| i as f64).collect();
    let ones = vec![1.0; 400];
    for i in 200..400 {
        m.step(&[], &[Some(x[i])], 1.0, 1.0);
        if matches!(i, 230 | 399) {
            let from = (i + 1).saturating_sub(60).max(200);
            let got = &m.powers()[0];
            for (k, f) in [0.125, 0.25].iter().enumerate() {
                let want = longhand(
                    *f,
                    &x[from..=i],
                    &tau[from..=i],
                    &ones[from..=i],
                    f64::INFINITY,
                    1.0,
                    SpectralScaling::Spectrum,
                    platform,
                );
                assert!(
                    (got[k] - want).abs() <= 1e-9 * want,
                    "row {i}, f {f}: {} vs {want}",
                    got[k]
                );
            }
        }
    }
}

/// Ten million rows through a window, its clock starting near 0 and near
/// `2^40`: each block's sums are at its own origin and turned once to a
/// reference within two windows of it, so neither the stream's length nor
/// the clock's size enters a phase but through a difference of nearby
/// clocks. With no decay on an uneven clock, at a thousand to ten million
/// rows, the read is the direct sum over the window's rows, its phases
/// from the same differences, to `1e-9` of the power.
#[test]
#[ignore = "extended: ten million rows, twice"]
fn a_window_holds_its_digits_over_ten_million_rows() {
    let n = 10_000_000usize;
    let w = 500.0;
    let freqs = [0.01, 0.125, 0.4];
    let mut worst = 0.0_f64;
    for origin in [0.0, 2f64.powi(40)] {
        let mut m = windowed(freqs.to_vec(), f64::INFINITY, w, SpectralScaling::Spectrum);
        let mut s = 61;
        let (mut t, mut tail) = (origin, std::collections::VecDeque::new());
        for i in 0..n {
            let d = if i == 0 { origin } else { 0.5 + lcg(&mut s) };
            if i > 0 {
                t += d;
            }
            let x = 3.0 + (std::f64::consts::TAU * (t - origin) / 8.0).sin() + normal(&mut s);
            m.step(&[], &[Some(x)], d, 1.0);
            tail.push_back((t, x));
            while tail.front().is_some_and(|&(u, _)| t - u >= w) {
                tail.pop_front();
            }
            if matches!(i + 1, 1_000 | 10_000 | 100_000 | 1_000_000 | 10_000_000) {
                let t0 = tail.front().unwrap().0;
                let xs: Vec<f64> = tail.iter().map(|p| p.1).collect();
                let ts: Vec<f64> = tail.iter().map(|p| p.0 - t0).collect();
                let ones = vec![1.0; xs.len()];
                let got = &m.powers()[0];
                let mut here = 0.0_f64;
                for (k, f) in freqs.iter().enumerate() {
                    let want = longhand(
                        *f,
                        &xs,
                        &ts,
                        &ones,
                        f64::INFINITY,
                        1.0,
                        SpectralScaling::Spectrum,
                        platform,
                    );
                    here = here.max((got[k] - want).abs() / want);
                }
                worst = worst.max(here);
                eprintln!("origin {origin:e}, row {}: rel {here:e}", i + 1);
            }
        }
    }
    assert!(worst < 1e-9, "{worst:e}");
}

/// Past a thinning budget neighboring blocks merge, the younger turned
/// once to the older's origin, and the window still never keeps an older
/// row: with no decay and a row a clock unit, it holds `n_eff` rows, at
/// most the window's, and its power is the direct sum over those last
/// rows.
#[test]
fn a_thinned_window_is_the_definition_over_the_rows_it_holds() {
    let x = signal(2000, 67);
    let mut m = windowed(
        vec![0.125, 0.3],
        f64::INFINITY,
        200.0,
        SpectralScaling::Spectrum,
    );
    m.set_window_budget(Some(crate::WindowBudget::Thin(0.01)));
    let tau: Vec<f64> = (0..2000).map(|i| i as f64).collect();
    let ones = vec![1.0; 2000];
    for i in 0..2000 {
        m.step(&[], &[Some(x[i])], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        if matches!(i, 700 | 1999) {
            let held = m.n_eff().round() as usize;
            assert!((1..=200).contains(&held), "{held}");
            let win = m.win.as_ref().unwrap();
            assert!(
                win.blocks.len() < 100 && win.ring.cadence().rows > 1,
                "{}",
                win.blocks.len()
            );
            let from = i + 1 - held;
            let got = &m.powers()[0];
            for (k, f) in [0.125, 0.3].iter().enumerate() {
                let want = longhand(
                    *f,
                    &x[from..=i],
                    &tau[from..=i],
                    &ones[from..=i],
                    f64::INFINITY,
                    1.0,
                    SpectralScaling::Spectrum,
                    platform,
                );
                assert!((got[k] - want).abs() <= 1e-9 * want, "{} vs {want}", got[k]);
            }
        }
    }
    assert!(m.window_over_budget().is_none());
}

/// Regression for TEST-10 proptest: a row at `INPUT_BOUND` (`1e100`) with
/// weight `1e100` puts `s.x` near `1e200` and `s.q` near `1e200`. Squaring
/// `re`/`im` before scaling overflowed to `+inf` even though the true power
/// `O(1e200)` fits in `f64` (and similarly in `Spectrum` and `LombScargle`).
#[test]
fn power_at_the_input_bound_scales_before_squaring_and_never_emits_inf() {
    let base = SpectralCfg {
        n_targets: 2,
        frequencies: vec![0.0, 0.2, 0.5],
        spacing: 1.0,
        decay: Decay::Halflife(20.0),
        scaling: SpectralScaling::Density,
        detrend: true,
        min_weight: 0.0,
        max_state_bytes: None,
        window: None,
        window_every: None,
        max_rows_between_snapshots: None,
    };
    for scaling in [
        SpectralScaling::Density,
        SpectralScaling::Spectrum,
        SpectralScaling::LombScargle,
    ] {
        let mut c = base.clone();
        c.scaling = scaling;
        if scaling == SpectralScaling::LombScargle {
            c.frequencies = vec![0.1, 0.2, 0.5];
        }
        let mut m = Spectral::new(c).unwrap();
        let s0 = m.step(&[], &[None, Some(1e100)], 0.0, 1e100);
        let s1 = m.step(&[], &[None, None], 1.0, 1.0);
        let s2 = m.step(&[], &[None, None], 1.0, 1.0);
        for (row, s) in [(0, &s0), (1, &s1), (2, &s2)] {
            assert!(
                s.pred.iter().all(|v| !v.is_infinite()),
                "{scaling:?} row {row}: {:?}",
                s.pred
            );
        }
        // Single detrended point: mean equals the point itself, so power is 0
        // (up to floating-point rounding of `s.x - m * s.u` at `1e200`, which
        // stays finite rather than overflowing to `+inf`).
        if scaling != SpectralScaling::LombScargle {
            assert!(
                s2.pred[3..6].iter().all(|v| v.is_finite()),
                "{scaling:?}: {:?}",
                &s2.pred[3..6]
            );
        }
    }
    // Non-detrended single point at `y = 1e100, w = 1e100`: exact density at
    // DC is `|w y|^2 * 1 / w^2 = y^2 = 1e200` and at interior `f = 0.2` is
    // `2 * 1e200`.
    let mut c = base;
    c.detrend = false;
    for scaling in [SpectralScaling::Density, SpectralScaling::Spectrum] {
        c.scaling = scaling;
        let mut m = Spectral::new(c.clone()).unwrap();
        m.step(&[], &[None, Some(1e100)], 0.0, 1e100);
        let s = m.step(&[], &[None, None], 1.0, 1.0);
        assert!(
            (s.pred[3] - 1e200).abs() <= 1e-12 * 1e200,
            "{scaling:?} DC: {}",
            s.pred[3]
        );
        assert!(
            (s.pred[4] - 2e200).abs() <= 1e-12 * 2e200,
            "{scaling:?} f=0.2: {}",
            s.pred[4]
        );
    }
}
