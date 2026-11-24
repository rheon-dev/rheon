//! `ew_cov` with shrinkage (docs/PLAN.md task 511): every second moment
//! reads `(1 − s) S + s T`, the `shrinkage` slot is `s`, and a row that
//! learns nothing moves neither.

use super::*;
use crate::{OnlineModel, Shrinkage};

fn rows(seed: u64, n: usize, k: usize) -> Vec<Vec<f64>> {
    let mut g = crate::SplitMix64::new(seed);
    (0..n)
        .map(|_| {
            let z: Vec<f64> = (0..k).map(|_| g.uniform() * 2.0 - 1.0).collect();
            (0..k)
                .map(|i| 4.0 + (1.0 + 0.4 * i as f64) * (z[i] + 0.7 * z[(i + 1) % k]))
                .collect()
        })
        .collect()
}

fn cfg(k: usize, stats: Vec<EwCovStat>, shrinkage: Option<Shrinkage>) -> EwCovCfg {
    EwCovCfg {
        bias: true,
        n_features: k,
        decay: crate::Decay::Halflife(f64::INFINITY),
        stats,
        min_weight: 0.0,
        precision_prior: Some(1e-9),
        mahal_quantiles: Vec::new(),
        pca: 0,
        pca_every: 0.0,
        max_rows_between_pca: u32::MAX,
        lags: Vec::new(),
        window: None,
        window_every: None,
        max_rows_between_snapshots: None,
        shrinkage,
    }
}

const ALL: [EwCovStat; 8] = [
    EwCovStat::Mean,
    EwCovStat::Var,
    EwCovStat::Std,
    EwCovStat::Cov,
    EwCovStat::Corr,
    EwCovStat::PartialCorr,
    EwCovStat::Mahal,
    EwCovStat::Shrinkage,
];

/// The biased covariance of `xs`, two-pass, row-major.
fn emp(xs: &[Vec<f64>]) -> (Vec<f64>, Vec<f64>) {
    let n = xs.len() as f64;
    let k = xs[0].len();
    let m: Vec<f64> = (0..k)
        .map(|i| xs.iter().map(|x| x[i]).sum::<f64>() / n)
        .collect();
    let mut s = vec![0.0; k * k];
    for x in xs {
        for i in 0..k {
            for j in 0..k {
                s[i * k + j] += (x[i] - m[i]) * (x[j] - m[j]) / n;
            }
        }
    }
    (m, s)
}

/// At unit weights and no decay, each statistic is that of the shrunk
/// two-pass covariance, at the intensity the `shrinkage` slot reports,
/// which is the estimator's on the rows so far (`crate::shrink`'s tests
/// hold the intensities to scikit-learn's arithmetic). Measured here, the
/// worst over the three estimators: the moments 6.3e-16 of the matrix's
/// largest entry, the correlations 1.7e-16, the partial correlations
/// 2.2e-16 and the distance 1.5e-16 relative (against `crate::oracle`'s
/// LU), the intensity 1.4e-16. Each bound is about a hundred times its
/// worst.
#[test]
fn every_second_moment_reads_the_shrunk_matrix() {
    let k = 4;
    let xs = rows(1, 60, k);
    let fixed = Shrinkage::Fixed {
        intensity: 0.3,
        target: None,
    };
    for sh in [Shrinkage::LedoitWolf, Shrinkage::Oas, fixed] {
        let mut m = EwCovModel::new(cfg(k, ALL.to_vec(), Some(sh.clone()))).unwrap();
        for x in &xs {
            m.step(x, &[], 1.0, 1.0);
        }
        let q = [0.5, -1.0, 2.0, 7.0];
        let out = m.predict(&q, 1.0).pred;
        let (mean, s) = emp(&xs);
        let n = xs.len();
        let (tr, frob2) = (
            (0..k).map(|i| s[i * k + i]).sum::<f64>(),
            s.iter().map(|v| v * v).sum::<f64>(),
        );
        let fourth: f64 = xs
            .iter()
            .map(|x| {
                let z: f64 = x.iter().zip(&mean).map(|(a, b)| (a - b) * (a - b)).sum();
                z * z
            })
            .sum::<f64>()
            / n as f64;
        let want_s = match &sh {
            Shrinkage::LedoitWolf => {
                crate::shrink::ledoit_wolf(k, tr, frob2, fourth, Some(n as f64))
            }
            Shrinkage::Oas => crate::shrink::oas(k, tr, frob2, Some(n as f64)),
            Shrinkage::Fixed { intensity, .. } => *intensity,
        };
        let got_s = *out.last().unwrap();
        assert!((got_s - want_s).abs() < 2e-14, "{sh:?} {got_s} {want_s}");
        assert!(got_s > 0.0 && got_s < 1.0);
        let mu = tr / k as f64;
        let mut shrunk = s.clone();
        for v in shrunk.iter_mut() {
            *v *= 1.0 - got_s;
        }
        for i in 0..k {
            shrunk[i * k + i] += got_s * mu;
        }
        let top = shrunk.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
        let close = |a: f64, b: f64| (a - b).abs() <= 1e-13 * top.max(1.0);
        let mut at = 0;
        for i in 0..k {
            assert!(close(out[at + i], mean[i]), "mean");
        }
        at += k;
        for i in 0..k {
            assert!(close(out[at + i], shrunk[i * k + i]), "var");
            assert!(close(out[at + k + i], shrunk[i * k + i].sqrt()), "std");
        }
        at += 2 * k;
        let pairs = k * (k - 1) / 2;
        let mut p = 0;
        for i in 0..k {
            for j in (i + 1)..k {
                assert!(close(out[at + p], shrunk[i * k + j]), "cov");
                let corr = shrunk[i * k + j] / (shrunk[i * k + i] * shrunk[j * k + j]).sqrt();
                assert!((out[at + pairs + p] - corr).abs() < 2e-14, "corr");
                p += 1;
            }
        }
        at += 2 * pairs;
        // The partial correlations and the distance from the shrunk
        // matrix with its ridge, through `crate::oracle`'s LU.
        let mut ridged = shrunk.clone();
        let ridge = 1e-9 * m.cov().precision_scale();
        for i in 0..k {
            ridged[i * k + i] += ridge;
        }
        let prec = crate::oracle::inverse(&ridged);
        p = 0;
        for i in 0..k {
            for j in (i + 1)..k {
                let pc = -prec[i * k + j] / (prec[i * k + i] * prec[j * k + j]).sqrt();
                assert!(
                    (out[at + p] - pc).abs() < 2e-14,
                    "partial {}",
                    out[at + p] - pc
                );
                p += 1;
            }
        }
        at += pairs;
        let delta: Vec<f64> = q.iter().zip(&mean).map(|(a, b)| a - b).collect();
        let d = crate::oracle::quad_form(&ridged, &delta).sqrt();
        assert!((out[at] - d).abs() < 2e-14 * d, "mahal {} {d}", out[at]);
    }
}

/// A row of weight 0 after a gap ages `W` and `Q` and moves no moment: the
/// intensity and every shrunk output are as before it, to the bit (hard
/// rule 9), because Kish's size is kept from the last learned row.
#[test]
fn a_row_that_learns_nothing_moves_no_shrunk_output() {
    let k = 3;
    let xs = rows(2, 40, k);
    for sh in [Shrinkage::LedoitWolf, Shrinkage::Oas] {
        let mut c = cfg(k, ALL.to_vec(), Some(sh));
        c.decay = crate::Decay::Halflife(7.0);
        let mut m = EwCovModel::new(c).unwrap();
        for (r, x) in xs.iter().enumerate() {
            m.step(x, &[], 1.0 + (r % 3) as f64 * 0.37, 0.5 + (r % 5) as f64);
        }
        let q = [1.0, 2.0, 3.0];
        let before = m.predict(&q, 0.0).pred;
        for gap in [0.3, 2.0, 11.0] {
            m.step(&xs[0], &[], gap, 0.0);
            let after = m.predict(&q, 0.0).pred;
            assert_eq!(
                before.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                after.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
        }
    }
}

/// Saved and loaded mid-stream, the model goes on to the bit, and
/// `predict` is what `step` reports.
#[test]
fn a_shrunk_model_continues_through_its_state() {
    let k = 5;
    let xs = rows(3, 80, k);
    let target = {
        let mut t = vec![0.0; k * (k + 1) / 2];
        for i in 0..k {
            t[i * (i + 1) / 2 + i] = 1.0 + i as f64;
        }
        t
    };
    for sh in [
        Shrinkage::LedoitWolf,
        Shrinkage::Oas,
        Shrinkage::Fixed {
            intensity: 0.2,
            target: Some(target.clone()),
        },
    ] {
        let mut c = cfg(k, ALL.to_vec(), Some(sh));
        c.decay = crate::Decay::Halflife(15.0);
        c.pca = 2;
        let mut a = EwCovModel::new(c).unwrap();
        for x in &xs[..40] {
            a.step(x, &[], 1.0, 1.0);
        }
        let bytes = rmp_serde::to_vec_named(&a.state()).unwrap();
        let state: crate::State = rmp_serde::from_slice(&bytes).unwrap();
        let mut b = EwCovModel::restore(&state).unwrap();
        assert_eq!(a, b);
        for x in &xs[40..] {
            let p = a.predict(x, 1.0).pred;
            let sa = a.step(x, &[], 1.0, 1.0).pred;
            let sb = b.step(x, &[], 1.0, 1.0).pred;
            let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&sa), bits(&sb));
            assert_eq!(bits(&sa), bits(&p));
        }
    }
}

/// The components of `(1 − s) S + s μ I` are `S`'s, each variance moved to
/// `(1 − s) λ + s μ`, along the same axes: against a twin without
/// shrinkage. The two matrices are decomposed separately; measured, the
/// variances agree to 1.9e-16 of the largest and the axes to 4.4e-16 of a
/// unit dot product, and each bound is about a hundred times that.
#[test]
fn the_components_of_the_shrunk_matrix_are_the_moments_moved_toward_the_mean() {
    let k = 4;
    let xs = rows(4, 50, k);
    let s = 0.35;
    let mk = |sh| {
        let mut c = cfg(k, vec![EwCovStat::Var], sh);
        c.pca = k;
        EwCovModel::new(c).unwrap()
    };
    let mut plain = mk(None);
    let mut shrunk = mk(Some(Shrinkage::Fixed {
        intensity: s,
        target: None,
    }));
    for x in &xs {
        plain.step(x, &[], 1.0, 1.0);
        shrunk.step(x, &[], 1.0, 1.0);
    }
    let (a, b) = (plain.pca().unwrap(), shrunk.pca().unwrap());
    let mu = a.trace / k as f64;
    for j in 0..k {
        let want = (1.0 - s) * a.eig[j] + s * mu;
        assert!((b.eig[j] - want).abs() < 2e-14 * a.eig[0], "{j}");
        // The same axis; its sign is each model's own continuity's, from
        // the refreshes before, where the order of two components could
        // differ.
        let along: f64 = (0..k)
            .map(|i| a.loadings[j * k + i] * b.loadings[j * k + i])
            .sum();
        assert!((along.abs() - 1.0).abs() < 5e-14, "loading {j}: {along}");
    }
    assert!((b.trace - a.trace).abs() < 1e-12 * a.trace);
}

/// One feature: both estimators give 0 and the variance is the moments'.
#[test]
fn one_feature_is_its_own_target() {
    let xs = rows(5, 30, 1);
    for sh in [Shrinkage::LedoitWolf, Shrinkage::Oas] {
        let mut m =
            EwCovModel::new(cfg(1, vec![EwCovStat::Var, EwCovStat::Shrinkage], Some(sh))).unwrap();
        let mut plain = EwCovModel::new(cfg(1, vec![EwCovStat::Var], None)).unwrap();
        for x in &xs {
            let a = m.step(x, &[], 1.0, 1.0).pred;
            let b = plain.step(x, &[], 1.0, 1.0).pred;
            assert_eq!(a[0].to_bits(), b[0].to_bits());
            assert_eq!(a[1], 0.0);
        }
    }
}

#[test]
fn shrinkage_is_refused_where_it_cannot_be_read() {
    let e = EwCovModel::new(cfg(2, vec![EwCovStat::Shrinkage], None)).unwrap_err();
    assert!(e.contains("\"shrinkage\" in stats needs shrinkage"), "{e}");
    let mut c = cfg(2, vec![EwCovStat::Var], Some(Shrinkage::LedoitWolf));
    c.window = Some(10.0);
    let e = EwCovModel::new(c).unwrap_err();
    assert!(e.contains("window_size and shrinkage"), "{e}");
    let e = EwCovModel::new(cfg(
        2,
        vec![EwCovStat::Var],
        Some(Shrinkage::Fixed {
            intensity: 1.5,
            target: None,
        }),
    ))
    .unwrap_err();
    assert!(e.contains("at most 1"), "{e}");
    // A state whose shrinkage parts do not fit its cfg is refused.
    let m = EwCovModel::new(cfg(2, vec![EwCovStat::Var], Some(Shrinkage::Oas))).unwrap();
    let mut s = m.state();
    if let crate::ModelState::EwCovModel(b) = &mut s.model {
        b.shrink = Some(crate::ShrinkState::new(&Shrinkage::LedoitWolf, 2));
    }
    assert!(EwCovModel::restore(&s).is_err());
    if let crate::ModelState::EwCovModel(b) = &mut s.model {
        b.shrink = None;
    }
    assert!(EwCovModel::restore(&s).is_err());
}
