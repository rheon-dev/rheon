//! The forecasts at horizons (task 559): `predict` reads what `step` writes,
//! to the bit, in every form; `Δ = 0` is the row's prediction; under the
//! random walk the forecast does not move, its integral is `Δ` times it,
//! and its variance grows with the span; under reversion the forecast
//! shrinks toward the reverted level and its integral is the forecasts'.

use super::*;
use crate::OnlineModel;

const K: usize = 3;
const SPANS: [f64; 4] = [0.0, 1.5, 7.0, 40.0];

fn cfg(revert: Vec<f64>, standardize: bool, share_p: bool) -> KalmanCfg {
    KalmanCfg {
        horizons: SPANS.to_vec(),
        min_weight_holds_learning: false,
        noise_units: true,
        n_features: K,
        n_targets: 2,
        fit_intercept: true,
        decay: Decay::Halflife(200.0),
        half_life: vec![50.0],
        q: None,
        obs_var: None,
        p0: 1.0,
        share_p,
        min_weight: 5.0,
        revert_half_life: revert,
        standardize,
        obs: Default::default(),
    }
}

fn configs() -> Vec<(KalmanCfg, CovarianceForm)> {
    let per_slot: Vec<f64> = (0..=K).map(|i| 30.0 * (i + 1) as f64).collect();
    let mut out = Vec::new();
    for revert in [vec![f64::INFINITY], vec![25.0], per_slot] {
        for (standardize, share_p) in [(true, false), (false, false), (true, true)] {
            for form in [CovarianceForm::Covariance, CovarianceForm::SquareRoot] {
                out.push((cfg(revert.clone(), standardize, share_p), form));
            }
        }
    }
    out
}

/// `n` rows: features, two targets (one null now and then), the clock step.
#[allow(clippy::type_complexity)]
fn rows(n: usize, seed: u64) -> Vec<([f64; K], [Option<f64>; 2], f64)> {
    let mut s = seed;
    (0..n)
        .map(|i| {
            let x = [lcg(&mut s), 2.0 + lcg(&mut s), 0.5 * lcg(&mut s)];
            let y0 = 1.0 + x[0] - 0.5 * x[1] + 0.1 * lcg(&mut s);
            let y1 = -2.0 + 0.3 * x[2] + 0.2 * lcg(&mut s);
            let y1 = (i % 7 != 3).then_some(y1);
            let d = if i == 0 {
                0.0
            } else {
                0.25 + lcg(&mut s).abs() * 2.0
            };
            (x, [Some(y0), y1], d)
        })
        .collect()
}

#[test]
fn predict_reads_what_step_writes_and_zero_is_the_prediction() {
    for (c, form) in configs() {
        let mut m = Kalman::new(c.clone())
            .unwrap()
            .with_covariance_form(form)
            .unwrap();
        assert_eq!(m.n_outputs(), 2 * (1 + 3 * SPANS.len()));
        let mut seen = 0;
        for (i, (x, y, d)) in rows(200, 11).iter().enumerate() {
            let read = m.predict(x, *d).pred;
            let got = m.step(x, y, *d, 1.0).pred;
            assert_eq!(got.len(), 2 + 2 * 3 * SPANS.len(), "{c:?}");
            let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&got), bits(&read), "{c:?} {form:?}, row {i}");
            for j in 0..2 {
                let at = 2 + j * 3 * SPANS.len();
                assert_eq!(
                    got[at].to_bits(),
                    got[j].to_bits(),
                    "Δ = 0 is the prediction"
                );
                if got[j].is_finite() {
                    assert_eq!(got[at + 1], 0.0);
                    seen += 1;
                }
            }
        }
        assert!(seen > 300, "{c:?}: forecasts were read ({seen})");
    }
}

#[test]
fn the_random_walk_holds_its_forecast_and_grows_its_variance() {
    let c = cfg(vec![f64::INFINITY], true, false);
    let mut m = Kalman::new(c).unwrap();
    let mut checked = 0;
    for (x, y, d) in rows(300, 12) {
        let got = m.step(&x, &y, d, 1.0).pred;
        for j in 0..2 {
            let at = 2 + j * 3 * SPANS.len();
            if !got[j].is_finite() || !got[at + 2].is_finite() {
                continue;
            }
            for (h, &dt) in SPANS.iter().enumerate() {
                let (f, area, var) = (got[at + 3 * h], got[at + 3 * h + 1], got[at + 3 * h + 2]);
                assert_eq!(f.to_bits(), got[j].to_bits());
                assert_eq!(area.to_bits(), (dt * got[j]).to_bits());
                if h > 0 {
                    assert!(var > got[at + 3 * (h - 1) + 2], "the span adds drift");
                }
            }
            checked += 1;
        }
    }
    assert!(checked > 400, "{checked}");
}

/// Under one reversion half-life `r` on every slot, unstandardized, the
/// forecast is `2^(−Δ/r)` times the prediction and its integral
/// `r/ln 2 (1 − 2^(−Δ/r))` times it.
#[test]
fn one_reversion_shrinks_the_forecast_and_integrates_it() {
    let r = 25.0;
    let c = cfg(vec![r], false, false);
    let mut m = Kalman::new(c).unwrap();
    let mut checked = 0;
    for (x, y, d) in rows(300, 13) {
        let got = m.step(&x, &y, d, 1.0).pred;
        for j in 0..2 {
            let at = 2 + j * 3 * SPANS.len();
            if !got[j].is_finite() {
                continue;
            }
            for (h, &dt) in SPANS.iter().enumerate() {
                let shrink = (-(dt / r)).exp2();
                let want = shrink * got[j];
                let f = got[at + 3 * h];
                assert!((f - want).abs() <= 1e-14 * (1.0 + want.abs()), "{f} {want}");
                let bar = r / std::f64::consts::LN_2 * (1.0 - shrink);
                let area = got[at + 3 * h + 1];
                let want = bar * got[j];
                assert!(
                    (area - want).abs() <= 1e-13 * (1.0 + want.abs()),
                    "{area} {want}"
                );
            }
            checked += 1;
        }
    }
    assert!(checked > 400, "{checked}");
}
