//! GARCH-X: the variance equations with inputs (PLAN 559 part 4,
//! docs/records/HORIZONS.md §10 and §16).

use super::filter::{ABS_Z, Filter};
use super::fit::{collinear, lr_cut, restrictions, set_of};
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

/// `q` inputs a row over `n` rows: each an AR(1) of unit variance and
/// coefficient 0.9, squared and halved under GARCH and GJR (so at least 0,
/// mean ½), as it is under EGARCH.
fn inputs(vol: GarchVol, q: usize, n: usize, seed: u64) -> Vec<f64> {
    let mut s = seed;
    let mut a = vec![0.0; q];
    let mut out = Vec::with_capacity(n * q);
    for _ in 0..n {
        for v in a.iter_mut() {
            *v = 0.9 * *v + (1.0f64 - 0.81).sqrt() * normal(&mut s);
            out.push(if vol == GarchVol::Egarch {
                *v
            } else {
                0.5 * *v * *v
            });
        }
    }
    out
}

/// The variance equation written out, one row at a time, from the
/// backcast `b` (the filter's lags before the first row): each row's
/// variance, its terms in the filter's order, the inputs' last.
fn definition(vol: GarchVol, theta: &[f64], e: &[f64], xs: &[f64], q: usize, b: f64) -> Vec<f64> {
    let vk = vol.k();
    let mut out = Vec::with_capacity(e.len());
    let (mut e2, mut neg, mut s) = (b, 0.5 * b, b);
    let (mut z, mut ln_s, mut fresh) = (0.0f64, b.ln(), true);
    for (t, &et) in e.iter().enumerate() {
        let x = &xs[t * q..(t + 1) * q];
        let v = match vol {
            GarchVol::Garch | GarchVol::Gjr => {
                let mut v = theta[0] + theta[1] * e2;
                if vk == 4 {
                    v += theta[2] * neg;
                }
                v += theta[vk - 1] * s;
                for i in 0..q {
                    v += theta[vk + i] * x[i];
                }
                v
            }
            _ => {
                let (az, zz) = if fresh {
                    (0.0, 0.0)
                } else {
                    (z.abs() - ABS_Z, z)
                };
                let mut l = theta[0] + theta[1] * az + theta[2] * zz + theta[3] * ln_s;
                for i in 0..q {
                    l += theta[4 + i] * x[i];
                }
                l.exp()
            }
        };
        out.push(v);
        e2 = et * et;
        neg = if et < 0.0 { e2 } else { 0.0 };
        s = v;
        z = et / v.sqrt();
        ln_s = v.ln();
        fresh = false;
    }
    out
}

/// A series of `n` rows of the process at `theta` with `q` inputs `xs`
/// (`n · q`), normal innovations, from the backcast's lags at its long-run
/// variance `v0`.
fn simulate(vol: GarchVol, theta: &[f64], xs: &[f64], q: usize, n: usize, seed: u64) -> Vec<f64> {
    let mut s = seed;
    let vk = vol.k();
    let mut out = Vec::with_capacity(n);
    let (mut e, mut v) = (0.0f64, 1.0f64);
    let mut ln_v = 0.0;
    for t in 0..n {
        let x = &xs[t * q..(t + 1) * q];
        let px: f64 = (0..q).map(|i| theta[vk + i] * x[i]).sum();
        if vol == GarchVol::Egarch {
            let z = if t == 0 { 0.0 } else { e / v.sqrt() };
            let news = if t == 0 {
                0.0
            } else {
                theta[1] * (z.abs() - ABS_Z) + theta[2] * z
            };
            ln_v = theta[0] + news + theta[3] * ln_v + px;
            v = ln_v.exp();
        } else {
            let g = if vk == 4 && e < 0.0 { theta[2] } else { 0.0 };
            v = theta[0] + (theta[1] + g) * e * e + theta[vk - 1] * v + px;
        }
        e = v.sqrt() * normal(&mut s);
        out.push(e);
    }
    out
}

fn cfg_x(vol: GarchVol, q: usize) -> GarchCfg {
    GarchCfg {
        n_targets: 1,
        vol,
        dist: GarchDist::Normal,
        truncation: None,
        variance_targeting: false,
        start_rows: 2000,
        start_span: None,
        horizons: Vec::new(),
        max_horizon_work: None,
        min_weight: 0.0,
        max_state_bytes: None,
        n_inputs: q,
        adaptive_start: false,
    }
}

const TRUTH_GARCH: [f64; 5] = [0.02, 0.05, 0.85, 0.1, 0.04];
const TRUTH_GJR: [f64; 6] = [0.02, 0.03, 0.06, 0.85, 0.1, 0.04];
const TRUTH_EGARCH: [f64; 6] = [-0.02, 0.15, -0.08, 0.95, 0.05, -0.03];

fn truth(vol: GarchVol) -> &'static [f64] {
    match vol {
        GarchVol::Garch => &TRUTH_GARCH,
        GarchVol::Gjr => &TRUTH_GJR,
        _ => &TRUTH_EGARCH,
    }
}

const X_FORMS: [GarchVol; 3] = [GarchVol::Garch, GarchVol::Gjr, GarchVol::Egarch];

#[test]
fn the_filter_is_the_definition_and_its_sensitivity_the_derivative() {
    let q = 2;
    for vol in X_FORMS {
        let th = truth(vol);
        let n = 300;
        let xs = inputs(vol, q, n, 3);
        let e = simulate(vol, th, &xs, q, n, 5);
        let b = 0.8;
        let want = definition(vol, th, &e, &xs, q, b);
        let k = vol.k() + q;
        let run = |theta: &[f64]| -> (Vec<f64>, Vec<Vec<f64>>) {
            let mut f = Filter::new(vol, k, 0, b);
            let mut dl = vec![0.0; k];
            let (mut s, mut d) = (Vec::new(), Vec::new());
            for (t, &et) in e.iter().enumerate() {
                let v = f.predict(vol, theta, &xs[t * q..(t + 1) * q], &mut dl);
                s.push(v);
                d.push(dl.clone());
                f.push(vol, et, v, &dl);
            }
            (s, d)
        };
        let (got, dl) = run(th);
        for t in 0..n {
            // the same terms in the same order: to the bit under GARCH and
            // GJR; EGARCH's `exp` is the `libm` crate's, the definition's
            // the platform's
            let tol = if vol == GarchVol::Egarch { 1e-13 } else { 0.0 };
            assert!(
                (got[t] - want[t]).abs() <= tol * want[t],
                "{vol:?} row {t}: {} {}",
                got[t],
                want[t]
            );
        }
        // ∂ln s/∂π_i by central differences of the definition
        for i in 0..q {
            let h = 1e-6;
            let (mut up, mut dn) = (th.to_vec(), th.to_vec());
            up[vol.k() + i] += h;
            dn[vol.k() + i] -= h;
            let (su, sd) = (
                definition(vol, &up, &e, &xs, q, b),
                definition(vol, &dn, &e, &xs, q, b),
            );
            for t in [1, 50, 299] {
                let fd = (su[t].ln() - sd[t].ln()) / (2.0 * h);
                let an = dl[t][vol.k() + i];
                assert!(
                    (fd - an).abs() < 1e-6 * (1.0 + an.abs()),
                    "{vol:?} π{i} row {t}: {fd} {an}"
                );
            }
        }
    }
}

#[test]
fn the_sets_bound_pi_below_by_0_under_garch_and_gjr_alone() {
    for vol in X_FORMS {
        let set = set_of(vol, 2, 0.0);
        let mut th = truth(vol).to_vec();
        th.truncate(vol.k() + 2);
        assert!(set.holds(&th, 0.0) || vol == GarchVol::Egarch, "{vol:?}");
        let mut neg = th.clone();
        neg[vol.k()] = -0.01;
        // π below 0: outside under GARCH and GJR, inside under EGARCH
        assert_eq!(set.holds(&neg, 0.0), vol == GarchVol::Egarch, "{vol:?}");
    }
    // without inputs the set is the process's own
    let (a, b) = (
        set_of(GarchVol::Gjr, 0, 1e-3),
        set_of(GarchVol::Gjr, 0, 1e-3),
    );
    assert_eq!((a.a, a.b, a.n), (b.a.clone(), b.b.clone(), 4));
}

/// `P(χ²_r > x)` for integer `r`: the Poisson sum for even `r`, and for
/// odd `r` `erfc(√(x/2))` plus its odd terms.
fn chi2_sf(x: f64, r: usize) -> f64 {
    let h = 0.5 * x;
    if r.is_multiple_of(2) {
        let mut term = (-h).exp();
        let mut sum = term;
        for j in 1..r / 2 {
            term *= h / j as f64;
            sum += term;
        }
        sum
    } else {
        let mut term = (h / std::f64::consts::PI).sqrt() * 2.0 * (-h).exp();
        let mut sum = 0.0;
        for j in 0..(r - 1) / 2 {
            if j > 0 {
                term *= h / (j as f64 + 0.5);
            }
            sum += term;
        }
        libm::erfc(h.sqrt()) + sum
    }
}

#[test]
fn the_cut_is_chi2_at_0999_for_2_plus_q_restrictions() {
    for q in 0..=MAX_INPUTS {
        assert_eq!(restrictions(GarchVol::Garch, q), 2 + q);
        assert_eq!(restrictions(GarchVol::Gjr, q), 3 + q);
        assert_eq!(restrictions(GarchVol::Egarch, q), 3 + q);
        for vol in X_FORMS {
            let r = restrictions(vol, q);
            let p = chi2_sf(lr_cut(vol, q), r);
            assert!((p - 1e-3).abs() < 1e-14, "{vol:?} q {q}: {p}");
        }
    }
}

#[test]
fn an_input_collinear_with_omega_or_another_is_found() {
    // unit-diagonal information with two exactly collinear columns, and one
    // only near
    let h = [
        1.0, 0.3, 1.0, //
        0.3, 1.0, 0.3, //
        1.0, 0.3, 1.0,
    ];
    assert!(collinear(&h, 3, 3));
    let h = [
        2.0, 0.3, 1.0, //
        0.3, 1.0, 0.3, //
        1.0, 0.3, 1.0,
    ];
    assert!(!collinear(&h, 3, 3));
    // the first `k` of a larger matrix, its diagonal scaled out
    let h = [
        4.0, 0.0, 2.0, 0.0, //
        0.0, 1.0, 0.0, 0.0, //
        2.0, 0.0, 1.0, 0.0, //
        0.0, 0.0, 0.0, 9.0,
    ];
    assert!(collinear(&h, 4, 3));
    assert!(!collinear(&h, 4, 2));
}

/// The model's run over `e` and `xs`, each row's report, and the model.
fn run(c: &GarchCfg, e: &[f64], xs: &[f64]) -> (Vec<Vec<f64>>, Garch) {
    let q = c.q();
    let mut m = Garch::new(c.clone()).unwrap();
    let mut out = Vec::with_capacity(e.len());
    for (t, &et) in e.iter().enumerate() {
        let st = m.step(&xs[t * q..(t + 1) * q], &[Some(et)], 1.0, 1.0);
        out.push(st.pred);
    }
    (out, m)
}

#[test]
fn the_estimate_finds_the_truth() {
    // 30,000 rows of each form with two inputs; after the start of 2,000
    // rows and the steps, every parameter near the truth. The tolerances
    // are a few of the batch fit's standard errors at this length (scipy's
    // fit of the same rows, tests/test_second_opinion_garch_x.py).
    let (q, n) = (2, 30_000);
    for vol in X_FORMS {
        let th = truth(vol);
        let xs = inputs(vol, q, n, 11);
        let e = simulate(vol, th, &xs, q, n, 13);
        let (_, m) = run(&cfg_x(vol, q), &e, &xs);
        let got = &m.coefficients()[0];
        for (i, (g, w)) in got.iter().zip(th).enumerate() {
            let tol = if i == 0 { 0.03 } else { 0.05 };
            assert!((g - w).abs() < tol, "{vol:?} {i}: {got:?} {th:?}");
        }
    }
}

#[test]
fn a_negative_input_under_garch_or_gjr_is_a_row_of_weight_0_that_reports_nothing() {
    for vol in X_FORMS {
        let (q, n) = (1, 2_600);
        let c = cfg_x(vol, q);
        let th = truth(vol);
        let xs = inputs(vol, q, n, 21);
        let e = simulate(vol, &th[..vol.k() + q], &xs, q, n, 23);
        let (_, mut a) = run(&c, &e, &xs);
        let mut b = a.clone();
        let x = [-0.5];
        let got = a.step(&x, &[Some(0.3)], 1.0, 1.0);
        let zero = b.step(&[0.0], &[Some(0.3)], 1.0, 0.0);
        if vol == GarchVol::Egarch {
            // any sign is read
            assert!(got.pred.iter().all(|v| v.is_finite()), "{got:?}");
            assert!(!c.refuses_inputs(&x));
            continue;
        }
        assert!(c.refuses_inputs(&x));
        assert!(got.pred.iter().all(|v| v.is_nan()), "{vol:?} {got:?}");
        assert_eq!(got.n_eff, zero.n_eff);
        assert_eq!(a, b, "{vol:?}: the row was learned");
        let p = a.predict(&x, 1.0);
        assert!(p.pred.iter().all(|v| v.is_nan()), "{vol:?}");
        // -0.0 is not below 0
        assert!(!c.refuses_inputs(&[-0.0]));
    }
}

#[test]
fn a_zero_weight_row_moves_no_mean_and_no_lag() {
    let (q, n) = (2, 2_300);
    for vol in X_FORMS {
        let th = truth(vol);
        let xs = inputs(vol, q, n, 31);
        let e = simulate(vol, th, &xs, q, n, 37);
        let (_, mut a) = run(&cfg_x(vol, q), &e, &xs);
        let before = a.clone();
        a.step(&[3.0, 7.0], &[Some(5.0)], 1.0, 0.0);
        assert_eq!(a, before, "{vol:?}");
    }
}

#[test]
fn the_forecasts_hold_the_inputs_at_their_learned_mean() {
    let (q, n) = (2, 2_500);
    for vol in [GarchVol::Garch, GarchVol::Gjr] {
        let th = truth(vol);
        let xs = inputs(vol, q, n, 41);
        let e = simulate(vol, th, &xs, q, n, 43);
        let c = GarchCfg {
            horizons: vec![1, 2, 7, 400],
            ..cfg_x(vol, q)
        };
        let (_, m) = run(&c, &e, &xs);
        let x = [0.7, 0.2];
        let st = m.predict(&x, 1.0);
        let s = &m.series[0];
        // x̄: the learned rows' mean, every row weighing 1, the start's included
        let rows = s.w_sum as usize;
        for i in 0..q {
            let mean: f64 = (n - rows..n).map(|t| xs[t * q + i]).sum::<f64>() / rows as f64;
            assert!((s.xbar[i] - mean).abs() < 1e-12, "{vol:?} {i}");
        }
        let tt = &s.theta;
        let vk = vol.k();
        let omega = tt[0] + tt[vk] * s.xbar[0] + tt[vk + 1] * s.xbar[1];
        let persist = if vk == 4 {
            tt[1] + 0.5 * tt[2] + tt[3]
        } else {
            tt[1] + tt[2]
        };
        let mut f = st.pred[0];
        let (mut sum, mut at) = (0.0, Vec::new());
        for h in 1..=400 {
            if h > 1 {
                f = omega + persist * f;
            }
            sum += f;
            at.push((f, sum));
        }
        for (slot, h) in [1usize, 2, 7, 400].iter().enumerate() {
            let (wf, ws) = at[h - 1];
            assert_eq!(st.pred[2 + 2 * slot], wf, "{vol:?} h {h}");
            assert!(
                (st.pred[3 + 2 * slot] - ws).abs() < 1e-12 * ws,
                "{vol:?} h {h}"
            );
        }
        // the row's own variance reads the row's own input, and the far
        // horizon reverts to the long-run variance at x̄
        let lr = omega / (1.0 - persist);
        assert!((st.pred[2 + 2 * 3] - lr).abs() < 1e-3 * lr, "{vol:?}");
    }
}

#[test]
fn a_constant_input_waits_as_not_identified() {
    // The input is 1 on every row: its π is ω again, and the start waits.
    let (q, n) = (1, 3_000);
    let xs = vec![1.0; n];
    let th = [0.02, 0.05, 0.85, 0.05];
    let e = simulate(GarchVol::Garch, &th, &xs, q, n, 51);
    let c = GarchCfg {
        max_state_bytes: Some(80_000),
        ..cfg_x(GarchVol::Garch, q)
    };
    let (out, m) = run(&c, &e, &xs);
    assert!(m.no_effect(), "{:?}", m.coefficients());
    assert!(out.iter().all(|p| p[0].is_nan()));
    // the same series with an input that moves starts
    let xs = inputs(GarchVol::Garch, q, n, 53);
    let e = simulate(GarchVol::Garch, &th, &xs, q, n, 51);
    let (_, m) = run(&c, &e, &xs);
    assert!(m.started()[0] && !m.no_effect());
}

#[test]
fn a_collinear_pair_of_inputs_waits_and_a_well_conditioned_pair_starts() {
    // The relative pivot rule on the start's information: two inputs that
    // are one input twice over, exactly or to 1e-7 of its level, leave a
    // pivot at the rounding and the start waits; two that move apart start.
    let (q, n) = (2, 3_000);
    let th = [0.02, 0.05, 0.85, 0.05, 0.05];
    let c = GarchCfg {
        max_state_bytes: Some(80_000),
        ..cfg_x(GarchVol::Garch, q)
    };
    let one = inputs(GarchVol::Garch, 1, n, 71);
    let mut s = 73;
    for jitter in [0.0, 1e-7] {
        let xs: Vec<f64> = one
            .iter()
            .flat_map(|&x| [x, 2.0 * x + jitter * lcg(&mut s)])
            .collect();
        let e = simulate(GarchVol::Garch, &th, &xs, q, n, 79);
        let (out, m) = run(&c, &e, &xs);
        assert!(m.no_effect(), "{jitter}: {:?}", m.coefficients());
        assert!(out.iter().all(|p| p[0].is_nan()), "{jitter}");
    }
    let xs = inputs(GarchVol::Garch, q, n, 71);
    let e = simulate(GarchVol::Garch, &th, &xs, q, n, 79);
    let (out, m) = run(&c, &e, &xs);
    assert!(m.started()[0] && !m.no_effect());
    assert!(out[n - 1][0].is_finite());
}

#[test]
fn the_state_round_trips_mid_start_and_after() {
    let (q, n) = (2, 2_400);
    for vol in X_FORMS {
        let th = truth(vol);
        let xs = inputs(vol, q, n, 61);
        let e = simulate(vol, th, &xs, q, n, 67);
        for stop in [700, n] {
            let (_, m) = run(&cfg_x(vol, q), &e[..stop], &xs[..stop * q]);
            let s = m.state();
            let bytes = rmp_serde::to_vec_named(&s).unwrap();
            let back: crate::model::State = rmp_serde::from_slice(&bytes).unwrap();
            // compared by its bytes: before the start `theta` is not a number
            let r = Garch::restore(&back).unwrap();
            assert_eq!(
                rmp_serde::to_vec_named(&r.state()).unwrap(),
                bytes,
                "{vol:?} {stop}"
            );
            // and as an array, as the contract's probe writes it
            let arr = rmp_serde::to_vec(&s).unwrap();
            let back: crate::model::State = rmp_serde::from_slice(&arr).unwrap();
            let r = Garch::restore(&back).unwrap();
            assert_eq!(
                rmp_serde::to_vec(&r.state()).unwrap(),
                arr,
                "{vol:?} {stop}"
            );
            assert_eq!(
                m.series[0].xbuf.len(),
                if stop < 2000 { stop * q } else { 0 }
            );
        }
    }
    // a buffer of inputs that does not match its rows is damaged
    let xs = inputs(GarchVol::Garch, q, 100, 1);
    let e = simulate(GarchVol::Garch, &TRUTH_GARCH, &xs, q, 100, 2);
    let (_, mut m) = run(&cfg_x(GarchVol::Garch, q), &e, &xs);
    m.series[0].xbuf.pop();
    assert!(Garch::restore(&m.state()).is_err());
}

#[test]
fn the_config_refuses_inputs_where_none_is_defined() {
    let mut c = cfg_x(GarchVol::Figarch, 1);
    c.start_rows = 2000;
    assert!(c.validate().unwrap_err().contains("figarch"));
    let c = GarchCfg {
        variance_targeting: true,
        ..cfg_x(GarchVol::Garch, 1)
    };
    assert!(c.validate().unwrap_err().contains("variance_targeting"));
    let c = cfg_x(GarchVol::Garch, MAX_INPUTS + 1);
    assert!(c.validate().unwrap_err().contains("at most 8"));
    assert!(cfg_x(GarchVol::Garch, MAX_INPUTS).validate().is_ok());
    // the state's bytes count the inputs: 8 (2k + q + m + n(n+1)/2 + 9)
    // and (16 + 8q) a buffered row
    let c = cfg_x(GarchVol::Garch, 2);
    assert_eq!(Garch::series_bytes(&c, 0), 8 * (2 * 5 + 2 + 15 + 9));
    assert_eq!(
        Garch::series_bytes(&c, 10) - Garch::series_bytes(&c, 0),
        10 * 32
    );
    let mut m = Garch::new(c.clone()).unwrap();
    for t in 0..10 {
        m.step(&[0.1, 0.2], &[Some(0.3 * t as f64 - 1.0)], 1.0, 1.0);
    }
    assert_eq!(m.bytes(), Garch::state_bytes(&c, 10));
}
