//! The statistics' emit: the correlations, the partial correlations and the
//! lagged correlations take their quotient whatever the denominator is and
//! choose after, so the compiler divides two at a time. Each is held to the
//! form that divides only where the denominator is positive, to the bit, on
//! the values a stream reaches: signed zeros, subnormals, overflow, NaN and
//! infinity, a column with no spread.

use super::*;
use crate::Decay;
use crate::model::OnlineModel;
use crate::oracle::CanonBits;
use std::hint::black_box;

fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

/// A number of the kinds the emit meets: mostly ordinary, often a value
/// that decides a branch or a clamp.
fn awkward(state: &mut u64) -> f64 {
    let u = lcg(state);
    let pick = lcg(state).abs();
    if pick < 0.05 {
        0.0
    } else if pick < 0.1 {
        -0.0
    } else if pick < 0.13 {
        f64::NAN
    } else if pick < 0.16 {
        f64::INFINITY
    } else if pick < 0.19 {
        f64::NEG_INFINITY
    } else if pick < 0.23 {
        1e-310 * u
    } else if pick < 0.27 {
        1e160 * u
    } else if pick < 0.3 {
        1.0
    } else if pick < 0.33 {
        -1.0
    } else {
        u
    }
}

/// The correlation as every build computed it before the quotient was
/// taken ahead of the choice.
fn correlation_by_branch(cij: f64, si: f64, sj: f64) -> f64 {
    let d = si * sj;
    if d > 0.0 {
        (cij / d).clamp(-1.0, 1.0)
    } else {
        f64::NAN
    }
}

#[test]
fn the_correlation_is_the_branching_form_to_the_bit() {
    let values = [
        0.0,
        -0.0,
        1.0,
        -1.0,
        0.5,
        -3.25,
        1e-310,
        -1e-310,
        1e-160,
        1e154,
        -1e154,
        1e300,
        f64::MAX,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
        -f64::NAN,
    ];
    let (mut nan, mut up, mut down, mut inside, mut negative_zero) = (0, 0, 0, 0, 0);
    for &cij in &values {
        for &si in &values {
            for &sj in &values {
                let (cij, si, sj) = (black_box(cij), black_box(si), black_box(sj));
                let want = correlation_by_branch(cij, si, sj);
                let got = correlation(cij, si, sj);
                assert_eq!(
                    got.canon_bits(),
                    want.canon_bits(),
                    "correlation({cij:e}, {si:e}, {sj:e})"
                );
                nan += usize::from(got.is_nan());
                up += usize::from(got == 1.0);
                down += usize::from(got == -1.0);
                inside += usize::from(got.abs() < 1.0 && got != 0.0);
                negative_zero += usize::from(got == 0.0 && got.is_sign_negative());
            }
        }
    }
    // Every kind of answer was reached.
    assert!(nan > 100 && up > 10 && down > 10 && inside > 10 && negative_zero > 0);
}

#[test]
fn the_correlations_of_a_triangle_are_the_branching_form_to_the_bit() {
    // The packed triangle's pairs through the method the emit calls: every
    // pair `(i, j)` once, at its place, for the widths that leave over a
    // different count of rows from the fours.
    let mut state = 11;
    for k in (1..=26).chain([33, 50]) {
        for round in 0..6 {
            let mut tri = LowerTriangle::zeros(k);
            let mut full = vec![0.0; k * k];
            for i in 0..k {
                for j in 0..=i {
                    let v = awkward(&mut state);
                    full[i * k + j] = v;
                    full[j * k + i] = v;
                }
            }
            tri.copy_from(&full);
            let std: Vec<f64> = (0..k).map(|_| awkward(&mut state)).collect();
            let mut got = vec![f64::MAX; k * (k - 1) / 2];
            tri.upper_rows_map(k, &mut got, &std, correlation);
            let mut at = 0;
            for i in 0..k {
                for j in (i + 1)..k {
                    let want = correlation_by_branch(tri.get(k, j, i), std[i], std[j]);
                    assert_eq!(
                        got[at].to_bits(),
                        want.to_bits(),
                        "k = {k}, round {round}, pair ({i}, {j})"
                    );
                    at += 1;
                }
            }
            assert_eq!(at, got.len());
        }
    }
}

#[test]
fn the_partial_correlations_are_the_pair_at_a_time_form_to_the_bit() {
    let mut state = 23;
    let (mut nan, mut clamped, mut inside) = (0, 0, 0);
    for k in (1..=26).chain([33, 50]) {
        for _ in 0..6 {
            let p: Vec<f64> = (0..k * k).map(|_| awkward(&mut state)).collect();
            // Sentinels where the row is longer than the scratch it reads.
            let mut got = vec![f64::MAX; k * (k - 1) / 2];
            let mut scratch = vec![7.0; 3];
            partial_corrs_into(&p, k, &mut got, &mut scratch);
            assert_eq!(scratch.len(), k, "the scratch holds the diagonal's roots");
            let mut at = 0;
            for i in 0..k {
                for j in (i + 1)..k {
                    let want = partial_corr(&p, k, i, j);
                    assert_eq!(
                        got[at].to_bits(),
                        want.to_bits(),
                        "k = {k}, pair ({i}, {j})"
                    );
                    nan += usize::from(want.is_nan());
                    clamped += usize::from(want.abs() == 1.0);
                    inside += usize::from(want.abs() < 1.0 && want != 0.0);
                    at += 1;
                }
            }
            assert_eq!(at, got.len());
        }
    }
    assert!(nan > 1000 && clamped > 100 && inside > 100);
}

#[test]
fn the_lagged_correlations_are_the_element_at_a_time_form_to_the_bit() {
    let mut state = 37;
    let (mut nan, mut finite) = (0, 0);
    for k in 1..=13 {
        for n_lags in 1..=4 {
            for _ in 0..4 {
                let c: Vec<f64> = (0..n_lags * k * k).map(|_| awkward(&mut state)).collect();
                let std: Vec<f64> = (0..k).map(|_| awkward(&mut state)).collect();
                let mut got = vec![f64::MAX; n_lags * k * k];
                lag_corrs_into(&c, &std, k, &mut got);
                let mut at = 0;
                for li in 0..n_lags {
                    for a in 0..k {
                        for b in 0..k {
                            let d = std[a] * std[b];
                            let want = if d > 0.0 {
                                c[li * k * k + a * k + b] / d
                            } else {
                                f64::NAN
                            };
                            assert_eq!(
                                got[at].canon_bits(),
                                want.canon_bits(),
                                "k = {k}, lag {li}, ({a}, {b})"
                            );
                            nan += usize::from(want.is_nan());
                            finite += usize::from(want.is_finite());
                            at += 1;
                        }
                    }
                }
                assert_eq!(at, got.len());
            }
        }
    }
    assert!(nan > 500 && finite > 500);
}

#[test]
fn the_slots_of_a_row_are_the_stored_moments_read_the_long_way_to_the_bit() {
    // The whole path, a model's `predict` over a stream with a column of no
    // spread, a column of zeros, a large one and a tiny one: each slot of the
    // correlations, the partial correlations and the lagged correlations is
    // the stored accumulator read pair by pair, as every earlier build did.
    for k in [2usize, 3, 4, 5, 6, 7, 8, 9, 13, 20] {
        let lags = vec![1usize, 3];
        let stats = vec![
            EwCovStat::Mean,
            EwCovStat::Corr,
            EwCovStat::PartialCorr,
            EwCovStat::LagCorr,
            EwCovStat::Std,
        ];
        let mut m = EwCovModel::new(EwCovCfg {
            bias: false,
            n_features: k,
            decay: crate::Decay::Halflife(30.0),
            stats: stats.clone(),
            min_weight: 2.0,
            precision_prior: Some(1.0),
            mahal_quantiles: Vec::new(),
            pca: 0,
            pca_every: 0.0,
            max_rows_between_pca: u32::MAX,
            lags: lags.clone(),
            window: None,
            window_every: None,
            max_rows_between_snapshots: None,
        })
        .unwrap();
        let mut state = 5 + k as u64;
        let mut x = vec![0.0; k];
        for _ in 0..120 {
            for (j, v) in x.iter_mut().enumerate() {
                *v = match j % 5 {
                    1 => 3.0,
                    2 => 0.0,
                    3 => 1e100 * lcg(&mut state),
                    4 => 1e-170 * lcg(&mut state),
                    _ => lcg(&mut state),
                };
            }
            m.step(&x, &[], 1.0, 1.0);
        }
        let got = m.predict(&x, 1.0).pred;
        let cov = m.cov();
        let lag = m.lag().expect("lags are kept");
        let std: Vec<f64> = (0..k).map(|i| cov.var(i).sqrt()).collect();
        let pairs = k * (k - 1) / 2;
        let precision = cov.precision();
        let mut want = Vec::new();
        want.extend((0..k).map(|i| cov.mean(i)));
        for i in 0..k {
            for j in (i + 1)..k {
                want.push(correlation_by_branch(cov.cov(j, i), std[i], std[j]));
            }
        }
        for i in 0..k {
            for j in (i + 1)..k {
                want.push(match &precision {
                    Some(p) => partial_corr(p, k, i, j),
                    None => f64::NAN,
                });
            }
        }
        for li in 0..lags.len() {
            for a in 0..k {
                for b in 0..k {
                    let d = std[a] * std[b];
                    want.push(if d > 0.0 {
                        lag.get(li, a, b) / d
                    } else {
                        f64::NAN
                    });
                }
            }
        }
        let unbiased = cov.unbiased_factor();
        want.extend((0..k).map(|i| (unbiased * cov.var(i)).sqrt()));
        assert_eq!(got.len(), want.len(), "k = {k}");
        assert_eq!(
            got.len(),
            k + 2 * pairs + lags.len() * k * k + k,
            "the slots are the stats' widths"
        );
        for (at, (g, w)) in got.iter().zip(&want).enumerate() {
            assert_eq!(g.canon_bits(), w.canon_bits(), "k = {k}, slot {at}");
        }
        // The stream reached a NaN and a number in the correlations.
        let corr = &got[k..k + pairs];
        assert!(corr.iter().any(|c| c.is_nan()), "k = {k}");
        // Two ordinary columns, the first and the sixth, make a correlation.
        if k >= 6 {
            assert!(corr.iter().any(|c| c.is_finite()), "k = {k}");
        }
    }
}

/// PLAN 517: `raw_cov` emits `E_w[x_i x_j] = C_ij + m_i m_j` for all `i <= j`
/// (divisor `W` regardless of `bias`, legal for `k >= 1`), and `loglik` emits
/// `-0.5 * (k ln(2π) + ln|S| + q)` with `S = c (C + s·prior·I)` matching
/// `mahal` and agreeing whether or not `partial_corr` is also requested.
#[test]
fn raw_cov_and_loglik_match_definition_and_share_factorization() {
    let mut state = 91u64;
    let mut next = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((state >> 11) as f64) / ((1u64 << 53) as f64) - 0.5
    };
    // k = 1 is legal for both raw_cov and loglik.
    let mut m1 = EwCovModel::new(EwCovCfg {
        n_features: 1,
        decay: Decay::Halflife(20.0),
        stats: vec![EwCovStat::RawCov, EwCovStat::Mahal, EwCovStat::Loglik],
        min_weight: 2.0,
        bias: false,
        precision_prior: Some(1e-3),
        mahal_quantiles: vec![],
        pca: 0,
        pca_every: 0.0,
        max_rows_between_pca: u32::MAX,
        lags: vec![],
        window: None,
        window_every: None,
        max_rows_between_snapshots: None,
    })
    .unwrap();
    for r in 0..30 {
        let x = [3.0 + next()];
        let out = m1.step(&x, &[], 1.0, 1.0);
        if r >= 3 {
            assert_eq!(out.pred.len(), 3);
            assert!(out.pred.iter().all(|v| v.is_finite()), "row {r}");
        }
    }

    // k = 3: compare bias=false and bias=true, with and without partial_corr.
    let k = 3usize;
    let rows: Vec<([f64; 3], f64, f64)> = (0..80)
        .map(|i| {
            let x = [10.0 + 2.0 * next(), -5.0 + next(), 3.0 + 0.5 * next()];
            let dt = 1.0 + (i % 3) as f64 * 0.5;
            let w = if i == 0 || i % 11 == 0 {
                0.0
            } else {
                0.5 + (i % 4) as f64 * 0.4
            };
            (x, dt, w)
        })
        .collect();
    for bias in [false, true] {
        let mk = |stats: Vec<EwCovStat>| {
            EwCovModel::new(EwCovCfg {
                n_features: k,
                decay: Decay::Halflife(25.0),
                stats,
                min_weight: 2.0,
                bias,
                precision_prior: Some(1e-3),
                mahal_quantiles: vec![],
                pca: 0,
                pca_every: 0.0,
                max_rows_between_pca: u32::MAX,
                lags: vec![],
                window: None,
                window_every: None,
                max_rows_between_snapshots: None,
            })
            .unwrap()
        };
        let mut a = mk(vec![EwCovStat::RawCov, EwCovStat::Mahal, EwCovStat::Loglik]);
        let mut b = mk(vec![
            EwCovStat::PartialCorr,
            EwCovStat::RawCov,
            EwCovStat::Mahal,
            EwCovStat::Loglik,
        ]);
        let mut c_bias = EwCovModel::new(EwCovCfg {
            bias: !bias,
            ..a.cfg().clone()
        })
        .unwrap();
        for (idx, (x, dt, w)) in rows.iter().enumerate() {
            let sa = a.step(x, &[], *dt, *w);
            let sb = b.step(x, &[], *dt, *w);
            let sc = c_bias.step(x, &[], *dt, *w);
            if idx >= 10 {
                // raw_cov slots: 0..6 in a and sc, 3..9 in b
                let raw_a = &sa.pred[0..6];
                let raw_b = &sb.pred[3..9];
                let raw_c = &sc.pred[0..6];
                for s in 0..6 {
                    assert_eq!(raw_a[s].canon_bits(), raw_b[s].canon_bits());
                    assert_eq!(
                        raw_a[s].canon_bits(),
                        raw_c[s].canon_bits(),
                        "raw_cov ignores bias"
                    );
                }
                // mahal & loglik match between shared-factor and partial_corr paths.
                let (mahal_a, ll_a) = (sa.pred[6], sa.pred[7]);
                let (mahal_b, ll_b) = (sb.pred[9], sb.pred[10]);
                assert!((mahal_a - mahal_b).abs() < 1e-11, "mahal row {idx}");
                assert!(
                    (ll_a - ll_b).abs() < 1e-11,
                    "ll row {idx}: {ll_a} vs {ll_b}"
                );
                // Independent oracle check at row 60 against two-pass weighted raw & logpdf.
                if idx == 60 {
                    // Check loglik accessor agrees with emitted loglik on next x.
                    let x_next = rows[idx + 1].0;
                    let ll_acc = a.loglik(&x_next);
                    let m_acc = a.mahal(&x_next);
                    let next_pred = a.predict(&x_next, rows[idx + 1].1).pred;
                    assert_eq!(ll_acc.canon_bits(), next_pred[7].canon_bits());
                    assert_eq!(m_acc.canon_bits(), next_pred[6].canon_bits());
                }
            }
        }
        // State round-trip preserves raw_cov and loglik to the bit.
        let restored = EwCovModel::restore(&a.state()).unwrap();
        let probe = [10.5, -4.5, 3.2];
        assert_eq!(
            a.predict(&probe, 1.0).pred,
            restored.predict(&probe, 1.0).pred
        );
    }

    // Labels check.
    let names = vec!["a".into(), "b".into()];
    let labels = EwCovModel::labels(&names, &[EwCovStat::RawCov, EwCovStat::Loglik], &[], 0, &[]);
    assert_eq!(
        labels,
        vec!["raw_cov_a_a", "raw_cov_a_b", "raw_cov_b_b", "loglik"]
    );

    // Validation refuses loglik without precision_prior.
    let err = EwCovModel::new(EwCovCfg {
        n_features: 2,
        decay: Decay::Halflife(10.0),
        stats: vec![EwCovStat::Loglik],
        min_weight: 2.0,
        bias: false,
        precision_prior: None,
        mahal_quantiles: vec![],
        pca: 0,
        pca_every: 0.0,
        max_rows_between_pca: u32::MAX,
        lags: vec![],
        window: None,
        window_every: None,
        max_rows_between_snapshots: None,
    })
    .unwrap_err();
    assert!(
        err.contains("\"loglik\"") && err.contains("precision_prior"),
        "{err}"
    );
}
