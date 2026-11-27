//! Unit tests for `bocpd` extensions:
//! - Negative-binomial run-length hazard (`nb_hazard`)
//! - Known-variance normal emission (`BocpdEmission::KnownVariance`, `obs_var`)
//! - Runtime live-run byte cap (`max_state_bytes`)

use super::tests::cfg;
use super::*;
use crate::OnlineModel;

/// Exact negative-binomial hazard `P(L = r + 1) / P(L >= r + 1)` computed via
/// direct log-binomial sums for small `k, r`.
fn exact_nb_hazard(k: usize, p: f64, r: usize) -> f64 {
    let t = r + 1;
    if t < k {
        return 0.0;
    }
    if p >= 1.0 {
        return 1.0;
    }
    let log_comb = |n: usize, j: usize| -> f64 {
        ln_gamma((n + 1) as f64) - ln_gamma((j + 1) as f64) - ln_gamma((n - j + 1) as f64)
    };
    let ln_pmf = log_comb(t - 1, k - 1) + (k as f64) * p.ln() + ((t - k) as f64) * (1.0 - p).ln();
    let ln_terms: Vec<f64> = (0..k)
        .map(|j| log_comb(t - 1, j) + (j as f64) * p.ln() + ((t - 1 - j) as f64) * (1.0 - p).ln())
        .collect();
    (ln_pmf - log_sum_exp(&ln_terms)).exp()
}

#[test]
fn neg_binomial_hazard_matches_definition_and_k1_equals_constant() {
    for &p in &[0.01, 0.05, 0.2, 0.75, 1.0] {
        for r in [0usize, 1, 2, 5, 20, 200, 5_000] {
            let h1 = neg_binomial_hazard(1, p, r);
            assert!((h1 - p).abs() < 1e-14, "k=1 p={p} r={r}: {h1}");
        }
    }
    for k in [2usize, 3, 5, 12] {
        for &p in &[0.02, 0.1, 0.4] {
            for r in [0usize, 1, 2, 3, 5, 10, 40, 200] {
                let got = neg_binomial_hazard(k, p, r);
                let want = exact_nb_hazard(k, p, r);
                assert!(
                    (got - want).abs() <= 1e-12 * (1.0 + want),
                    "k={k} p={p} r={r}: got {got} want {want}"
                );
            }
            // Support minimum: H(r) == 0 for r + 1 < k.
            for r in 0..(k - 1) {
                assert_eq!(neg_binomial_hazard(k, p, r), 0.0);
            }
            // Monotone non-decreasing in r and bounded by p.
            let mut prev = 0.0;
            for r in 0..300 {
                let h = neg_binomial_hazard(k, p, r);
                assert!(
                    h + 1e-15 >= prev && h <= p + 1e-15,
                    "k={k} p={p} r={r}: {h}"
                );
                prev = h;
            }
        }
    }
    // High k, tiny p exercises log-space fallback without NaN/Inf.
    let h = neg_binomial_hazard(50, 1e-4, 500);
    assert!(h.is_finite() && (0.0..=1e-4).contains(&h), "{h}");
}

#[test]
fn neg_binomial_k1_matches_constant_per_row_hazard_stream() {
    let p = 0.04;
    let c_const = BocpdCfg {
        hazard: 1.0 / p,
        prune_below: 0.0,
        max_run: 100,
        ..cfg(2)
    };
    let c_nb = BocpdCfg {
        hazard: 250.0,
        prune_below: 0.0,
        max_run: 100,
        extras: Some(Box::new(BocpdCfgExtras {
            nb_hazard: Some((1, p)),
            ..Default::default()
        })),
        ..cfg(2)
    };
    let mut a = Bocpd::new(c_const).unwrap();
    let mut b = Bocpd::new(c_nb).unwrap();
    for i in 0..50 {
        let shift = if i >= 25 { 4.0 } else { 0.0 };
        let x = [shift + (i % 5) as f64 * 0.2, -shift + (i % 3) as f64 * 0.3];
        let sa = a.step(&x, &[], 1.0, 1.0);
        let sb = b.step(&x, &[], 1.0, 1.0);
        for (u, v) in sa.pred.iter().zip(&sb.pred) {
            assert!((u - v).abs() < 1e-12, "row {i}: {u} vs {v}");
        }
    }
}

#[test]
fn known_variance_emission_matches_exact_2_pow_t_enumeration() {
    // Enumerate all 2^T changepoint patterns for T = 8 rows under BOTH constant
    // hazard and NegBinomial(k=3, p=0.25) with KnownVariance(d=2).
    let d = 2;
    let mu0 = [0.25, -0.5];
    let tau2 = [2.0, 0.75];
    let obs_var = [0.5, 1.25];
    let xs = [
        [0.1, -0.4],
        [0.4, -0.6],
        [-0.2, -0.3],
        [3.1, 2.2],
        [3.4, 2.5],
        [2.9, 2.1],
        [3.2, 2.4],
        [3.0, 2.0],
    ];
    let exact_pred = |obs: &[[f64; 2]], x: &[f64; 2]| -> (f64, [f64; 2]) {
        let n = obs.len() as f64;
        let mut lp = 0.0;
        let mut mean = [0.0; 2];
        for i in 0..d {
            let xbar = if obs.is_empty() {
                0.0
            } else {
                obs.iter().map(|r| r[i]).sum::<f64>() / n
            };
            let v = 1.0 / (1.0 / tau2[i] + n / obs_var[i]);
            let m = v * (mu0[i] / tau2[i] + n * xbar / obs_var[i]);
            let pv = v + obs_var[i];
            lp += -0.5 * (std::f64::consts::TAU * pv).ln() - 0.5 * (x[i] - m).powi(2) / pv;
            mean[i] = m;
        }
        (lp, mean)
    };

    for nb in [None, Some((3usize, 0.25))] {
        let c = BocpdCfg {
            n_features: d,
            hazard: 10.0,
            hazard_from_row: false,
            emission: BocpdEmission::KnownVariance,
            prior_mean: Some(mu0.to_vec()),
            prior_kappa: 1.0,
            prior_nu: None,
            prior_scale: Some(tau2.to_vec()),
            robust_beta: 0.0,
            prune_below: 0.0,
            max_run: 100,
            min_weight: 0.0,
            warm_rows: None,
            hazard_on_clock: false,
            warm_clock: None,
            extras: Some(Box::new(BocpdCfgExtras {
                nb_hazard: nb,
                obs_var: Some(obs_var.to_vec()),
                max_state_bytes: None,
            })),
        };
        let mut m = Bocpd::new(c).unwrap();
        // Maintain exact joint `log P(r_t = r, x_{1:t})` in longhand.
        let mut exact_joint = vec![0.0f64]; // r = 0 before row 0
        for (t, x) in xs.iter().enumerate() {
            let step = m.step(x, &[], 1.0, 1.0);
            // Verify every run keeps m2 empty under KnownVariance.
            assert!(m.runs.iter().all(|r| r.m2.is_empty()));
            let z_pre = log_sum_exp(&exact_joint);
            let pre: Vec<f64> = exact_joint.iter().map(|l| (l - z_pre).exp()).collect();
            let mut want_pred = [0.0; 2];
            let mut lmix = Vec::new();
            let mut lpi = Vec::new();
            for r in 0..=t {
                let (lp, rm) = exact_pred(&xs[t - r..t], x);
                lpi.push(lp);
                for i in 0..d {
                    want_pred[i] += pre[r] * rm[i];
                }
                if pre[r] > 0.0 {
                    lmix.push(pre[r].ln() + lp);
                }
            }
            let want_ll = log_sum_exp(&lmix);
            let mut next_j = vec![f64::NEG_INFINITY; t + 2];
            let mut cp = Vec::new();
            for r in 0..=t {
                let h = match nb {
                    None => 0.1,
                    Some((k, p)) => neg_binomial_hazard(k, p, r),
                };
                let base = exact_joint[r] + lpi[r];
                next_j[r + 1] = base
                    + if h < 1.0 {
                        (1.0 - h).ln()
                    } else {
                        f64::NEG_INFINITY
                    };
                cp.push(base + if h > 0.0 { h.ln() } else { f64::NEG_INFINITY });
            }
            next_j[0] = log_sum_exp(&cp);
            let z_post = log_sum_exp(&next_j);
            let want_p_change = (log_sum_exp(&next_j[..2]) - z_post).exp();
            assert!(
                (step.pred[0] - want_p_change).abs() < 1e-13,
                "nb={nb:?} t={t}"
            );
            for (i, &wp) in want_pred.iter().enumerate() {
                assert!(
                    (step.pred[3 + i] - wp).abs() < 1e-13,
                    "nb={nb:?} t={t} i={i}"
                );
            }
            assert!((step.pred[5] - want_ll).abs() < 1e-13, "nb={nb:?} t={t}");
            exact_joint = next_j;
        }
    }
}

#[test]
fn runtime_live_run_cap_warns_at_half_and_refuses_atomically() {
    let d = 2;
    let bytes_per_run = 8 * (3 + 3 * d) as u64; // Diag: 72 B/run
    let max_runs_fit = 10u64;
    let cap = max_runs_fit * bytes_per_run;
    let c = BocpdCfg {
        prune_below: 0.0,
        max_run: 1000,
        extras: Some(Box::new(BocpdCfgExtras {
            max_state_bytes: Some(cap),
            ..Default::default()
        })),
        ..cfg(d)
    };
    let mut m = Bocpd::new(c).unwrap();
    let mut warned = 0;
    // Starts with 1 run; each stationary row with prune_below=0 adds 1 run.
    // Row 0 -> 2 runs, ..., row 8 -> 10 runs (at cap), row 9 -> 11 runs (breach!).
    for i in 0..9 {
        m.step(&[0.1 * (i % 3) as f64, -0.1], &[], 1.0, 1.0);
        if let Some((b, got_cap)) = m.take_state_warning() {
            warned += 1;
            assert_eq!(got_cap, cap);
            assert!(2 * b > cap);
        }
        assert_eq!(m.state_over_cap(), None, "row {i} must fit");
    }
    assert_eq!(warned, 1);
    assert_eq!(m.runs.len(), 10);
    // Snapshot state before breaching row.
    let pre_runs = m.runs.clone();
    let pre_joint = m.logjoint.clone();
    let pre_neff = m.n_eff;
    let pre_pred = m.predict(&[0.5, -0.2], 1.0);

    // Row 9 would grow to 11 runs > cap: refused atomically!
    let out = m.step(&[0.5, -0.2], &[], 1.0, 1.0);
    assert_eq!(m.state_over_cap(), Some((11 * bytes_per_run, cap)));
    assert_eq!(m.runs, pre_runs);
    assert_eq!(m.logjoint, pre_joint);
    assert_eq!(m.n_eff, pre_neff);
    assert_eq!(out.pred, pre_pred.pred);

    // Subsequent rows after breach also leave everything untouched.
    m.step(&[10.0, 10.0], &[], 1.0, 1.0);
    assert_eq!(m.runs, pre_runs);
    assert_eq!(m.state_over_cap(), Some((11 * bytes_per_run, cap)));
}

#[test]
fn bocpd_523_refusals_are_clear() {
    let check = |c: BocpdCfg, needle: &str| {
        let e = Bocpd::new(c).unwrap_err();
        assert!(e.contains(needle), "expected {needle:?} in {e:?}");
    };
    // hazard_k == 0 and hazard_p out of (0, 1].
    check(
        BocpdCfg {
            extras: Some(Box::new(BocpdCfgExtras {
                nb_hazard: Some((0, 0.1)),
                ..Default::default()
            })),
            ..cfg(1)
        },
        "hazard_k must be at least 1",
    );
    for bad_p in [0.0, -0.1, 1.01, f64::NAN, f64::INFINITY] {
        check(
            BocpdCfg {
                extras: Some(Box::new(BocpdCfgExtras {
                    nb_hazard: Some((2, bad_p)),
                    ..Default::default()
                })),
                ..cfg(1)
            },
            "hazard_p must be finite",
        );
    }
    // nb_hazard beside clock or row hazard.
    check(
        BocpdCfg {
            hazard_on_clock: true,
            extras: Some(Box::new(BocpdCfgExtras {
                nb_hazard: Some((2, 0.1)),
                ..Default::default()
            })),
            ..cfg(1)
        },
        "clock hazard",
    );
    check(
        BocpdCfg {
            hazard_from_row: true,
            extras: Some(Box::new(BocpdCfgExtras {
                nb_hazard: Some((2, 0.1)),
                ..Default::default()
            })),
            ..cfg(1)
        },
        "hazard_col",
    );
    // KnownVariance missing obs_var, non-positive obs_var, wrong len, or beside prior_nu / robust_beta.
    check(
        BocpdCfg {
            emission: BocpdEmission::KnownVariance,
            prior_nu: None,
            ..cfg(2)
        },
        "needs obs_var",
    );
    check(
        BocpdCfg {
            emission: BocpdEmission::KnownVariance,
            prior_nu: None,
            extras: Some(Box::new(BocpdCfgExtras {
                obs_var: Some(vec![1.0, -0.5]),
                ..Default::default()
            })),
            ..cfg(2)
        },
        "obs_var must be finite and above 0",
    );
    check(
        BocpdCfg {
            emission: BocpdEmission::KnownVariance,
            prior_nu: Some(3.0),
            extras: Some(Box::new(BocpdCfgExtras {
                obs_var: Some(vec![1.0]),
                ..Default::default()
            })),
            ..cfg(2)
        },
        "degrees_of_freedom_prior",
    );
    check(
        BocpdCfg {
            extras: Some(Box::new(BocpdCfgExtras {
                max_state_bytes: Some(0),
                ..Default::default()
            })),
            ..cfg(2)
        },
        "max_state_bytes must be above 0",
    );
}
