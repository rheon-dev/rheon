//! `ew_class` with shrinkage (docs/PLAN.md task 511): each class's
//! covariance shrunk at its own intensity before the ridge, the `tied`
//! shape pooling the shrunk matrices, against a from-scratch definition.

use super::*;
use crate::Shrinkage;

fn cfg(k: usize, nc: usize, covariance: Covariance, sh: Option<Shrinkage>) -> EwClassCfg {
    EwClassCfg {
        n_features: k,
        n_classes: nc,
        decay: Decay::Halflife(f64::INFINITY),
        min_weight: 0.0,
        covariance,
        precision_prior: 1e-3,
        window: None,
        window_every: None,
        max_rows_between_snapshots: None,
        shrinkage: sh,
    }
}

/// `n` labelled rows, three classes about three centres, correlated
/// features.
fn rows(n: usize, k: usize, seed: u64) -> Vec<(Vec<f64>, usize)> {
    let mut g = crate::SplitMix64::new(seed);
    (0..n)
        .map(|r| {
            let c = r % 3;
            let z: Vec<f64> = (0..k).map(|_| g.uniform() * 2.0 - 1.0).collect();
            let x = (0..k)
                .map(|i| c as f64 * (1.0 + i as f64) * 0.7 + z[i] + 0.5 * z[(i + 1) % k])
                .collect();
            (x, c)
        })
        .collect()
}

/// The posteriors by definition: per class the biased covariance of its
/// rows, shrunk at the estimator's intensity on those rows, the ridge
/// `prior · s_c` added (`s_c` the class's scale, `1/n_c` at unit weights
/// and no decay), pooled by the priors under `tied`; then Gaussian scores
/// through `crate::oracle`'s LU and a softmax.
fn definition(
    seen: &[(Vec<f64>, usize)],
    x: &[f64],
    k: usize,
    nc: usize,
    covariance: Covariance,
    sh: &Shrinkage,
    prior: f64,
) -> Vec<f64> {
    let mut mats = Vec::new();
    let mut means = Vec::new();
    let mut counts = Vec::new();
    for c in 0..nc {
        let xs: Vec<&Vec<f64>> = seen
            .iter()
            .filter(|(_, l)| *l == c)
            .map(|(x, _)| x)
            .collect();
        let n = xs.len() as f64;
        let m: Vec<f64> = (0..k)
            .map(|i| xs.iter().map(|x| x[i]).sum::<f64>() / n)
            .collect();
        let mut s = vec![0.0; k * k];
        let mut f = 0.0;
        for x in &xs {
            let z: Vec<f64> = x.iter().zip(&m).map(|(a, b)| a - b).collect();
            let zz: f64 = z.iter().map(|v| v * v).sum();
            f += zz * zz / n;
            for i in 0..k {
                for j in 0..k {
                    s[i * k + j] += z[i] * z[j] / n;
                }
            }
        }
        let tr: f64 = (0..k).map(|i| s[i * k + i]).sum();
        let frob2: f64 = s.iter().map(|v| v * v).sum();
        let sc = match sh {
            Shrinkage::LedoitWolf => crate::shrink::ledoit_wolf(k, tr, frob2, f, Some(n)),
            Shrinkage::Oas => crate::shrink::oas(k, tr, frob2, Some(n)),
            Shrinkage::Fixed { intensity, .. } => *intensity,
        };
        let mut shrunk: Vec<f64> = s.iter().map(|v| (1.0 - sc) * v).collect();
        for i in 0..k {
            shrunk[i * k + i] += sc * tr / k as f64 + prior / n;
        }
        mats.push(shrunk);
        means.push(m);
        counts.push(n);
    }
    let total: f64 = counts.iter().sum();
    let pooled: Vec<f64> = (0..k * k)
        .map(|ij| (0..nc).map(|c| counts[c] / total * mats[c][ij]).sum())
        .collect();
    let ell: Vec<f64> = (0..nc)
        .map(|c| {
            let m = if matches!(covariance, Covariance::Shared) {
                &pooled
            } else {
                &mats[c]
            };
            let d: Vec<f64> = x.iter().zip(&means[c]).map(|(a, b)| a - b).collect();
            (counts[c] / total).ln()
                - 0.5 * crate::oracle::log_det(m)
                - 0.5 * crate::oracle::quad_form(m, &d)
        })
        .collect();
    let top = ell.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let z: f64 = ell.iter().map(|l| (l - top).exp()).sum();
    ell.iter().map(|l| (l - top).exp() / z).collect()
}

/// Each estimator, in the `full` and `tied` shapes, against the
/// definition. Measured, the posteriors agree to 3.1e-15 at worst (the
/// `full` shape under Ledoit and Wolf's); the bound is about fifteen times
/// that, the definition's LU against the model's Cholesky.
#[test]
fn each_class_is_shrunk_then_pooled() {
    let (k, nc) = (4, 3);
    let data = rows(150, k, 7);
    let fixed = Shrinkage::Fixed {
        intensity: 0.4,
        target: None,
    };
    for covariance in [Covariance::Full, Covariance::Shared] {
        for sh in [Shrinkage::LedoitWolf, Shrinkage::Oas, fixed.clone()] {
            let mut m = EwClass::new(cfg(k, nc, covariance, Some(sh.clone()))).unwrap();
            let mut worst = 0.0_f64;
            for (r, (x, c)) in data.iter().enumerate() {
                let out = m.step(x, &[Some(*c as f64)], 1.0, 1.0).pred;
                if r >= 3 * 5 {
                    let want = definition(&data[..r], x, k, nc, covariance, &sh, 1e-3);
                    for (a, b) in out[1..].iter().zip(&want) {
                        worst = worst.max((a - b).abs());
                    }
                }
            }
            assert!(worst < 5e-14, "{covariance:?} {sh:?}: {worst}");
        }
    }
}

/// A class that does not learn a row keeps its intensity and its factor:
/// the `full` shape's cache, emptied before every row, gives every output
/// to the bit, under decay and rows of weight 0.
#[test]
fn a_class_that_learns_nothing_keeps_its_factor() {
    let (k, nc) = (3, 3);
    for sh in [Shrinkage::LedoitWolf, Shrinkage::Oas] {
        let mut c = cfg(k, nc, Covariance::Full, Some(sh));
        c.decay = Decay::Halflife(9.0);
        let mut m = EwClass::new(c).unwrap();
        for (r, (x, l)) in rows(200, k, 8).iter().enumerate() {
            let mut cold = m.clone();
            cold.factors = Factors::default();
            let w = if r % 7 == 3 { 0.0 } else { 1.0 };
            let y = [Some(*l as f64)];
            let a = m.step(x, &y, 1.0 + (r % 3) as f64, w);
            let b = cold.step(x, &y, 1.0 + (r % 3) as f64, w);
            let bits = |v: &[f64]| v.iter().map(|p| p.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&a.pred), bits(&b.pred), "row {r}");
        }
    }
}

#[test]
fn the_state_goes_on_to_the_bit() {
    let (k, nc) = (3, 3);
    for covariance in [Covariance::Full, Covariance::Shared] {
        let mut a = EwClass::new(cfg(k, nc, covariance, Some(Shrinkage::LedoitWolf))).unwrap();
        let data = rows(90, k, 9);
        for (x, c) in &data[..45] {
            a.step(x, &[Some(*c as f64)], 1.0, 1.0);
        }
        let bytes = rmp_serde::to_vec_named(&a.state()).unwrap();
        let mut b = EwClass::restore(&rmp_serde::from_slice(&bytes).unwrap()).unwrap();
        for (x, c) in &data[45..] {
            let p = a.step(x, &[Some(*c as f64)], 1.0, 1.0).pred;
            let q = b.step(x, &[Some(*c as f64)], 1.0, 1.0).pred;
            assert_eq!(
                p.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                q.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
        }
    }
}

#[test]
fn shrinkage_is_refused_where_there_is_no_matrix() {
    let e = EwClass::new(cfg(2, 2, Covariance::Diagonal, Some(Shrinkage::Oas))).unwrap_err();
    assert!(e.contains("covariance_type \"full\" or \"tied\""), "{e}");
    let mut c = cfg(2, 2, Covariance::Full, Some(Shrinkage::LedoitWolf));
    c.window = Some(10.0);
    assert!(
        EwClass::new(c)
            .unwrap_err()
            .contains("window_size and shrinkage")
    );
    // A state whose shrinkage parts do not fit is refused.
    let m = EwClass::new(cfg(2, 2, Covariance::Full, Some(Shrinkage::Oas))).unwrap();
    let mut s = m.state();
    if let ModelState::EwClass(b) = &mut s.model {
        b.shrink.as_mut().unwrap().pop();
    }
    assert!(EwClass::restore(&s).is_err());
}
