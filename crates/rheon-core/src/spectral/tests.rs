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
