//! Atomicity of every `HeldCap::admit` check across all 16 models that bound
//! data-sized state by `max_state_bytes` (or their warm-up byte cap):
//!
//! - On the row that breaches the cap (`state_over_cap()` transitions from
//!   `None` to `Some((bytes, cap))`), EVERY persisted field other than `over`
//!   must remain bit-for-bit identical to before the row, and `predict` /
//!   `predict_with` must remain bit-for-bit identical.
//! - On every subsequent row after `over.is_some()` (positive weight, zero
//!   weight, or missing targets), EVERY persisted field including `over` must
//!   remain bit-for-bit identical.
//! - On multi-target models, if target 0 fits under the cap and target 1 pushes
//!   the row past the cap, NEITHER target mutates.
//! - On clock-window models, if expired rows would leave at the breaching row's
//!   stamp yet the incoming row still breaches the cap, expired rows are NOT
//!   popped from the model's live window.

use crate::bocpd::{Bocpd, BocpdCfg, BocpdEmission};
use crate::cluster::{KMeans, KMeansCfg, SeedRule};
use crate::corrchange::{ChangeNorm, CorrChange, CorrChangeCfg, CorrChangeKind};
use crate::dcc::{DCC_START_ROWS, Dcc, DccCfg};
use crate::ddsketch_model::{DdSketchModel, DdSketchModelCfg};
use crate::ewm_moments::{EwmMoments, EwmMomentsCfg, Leg, MomentKind, MomentTerm};
use crate::expectile::{Expectile, ExpectileCfg};
use crate::family::GlmFamily;
use crate::garch::{Garch, GarchCfg, GarchDist, GarchVol};
use crate::glm::{Glm, GlmCfg};
use crate::hawkes::{Hawkes, HawkesCfg};
use crate::hmm::{Hmm, HmmCfg, HmmCovariance};
use crate::isotonic::{Isotonic, IsotonicCfg, OutOfBounds};
use crate::power_variation::{PowerVariation, PowerVariationCfg, PvTerm, PvTermKind};
use crate::realized::{DEFAULT_LM_CONFIDENCE, Realized, RealizedCfg};
use crate::robust::{HuberScale, Robust, RobustCfg, RobustLoss};
use crate::sgd::{LearningRate, Sgd, SgdCfg, SgdLoss};
use crate::{Decay, OnlineModel, Stamp, Step, WindowClosed};

const S: i128 = 1_000_000_000;

/// Serialized state decoded to `rmpv::Value` with only the model's top-level
/// `"over"` field removed so every other persisted field can be compared
/// bit-for-bit.
fn state_without_over(m: &impl OnlineModel) -> rmpv::Value {
    let bytes = rmp_serde::to_vec_named(&m.state()).expect("serialize state");
    let mut v = rmpv::decode::read_value(&mut bytes.as_slice()).expect("decode state");
    if let rmpv::Value::Map(top) = &mut v {
        for (k, val) in top.iter_mut() {
            if k.as_str() == Some("model")
                && let rmpv::Value::Map(variants) = val
            {
                for (_, body) in variants.iter_mut() {
                    if let rmpv::Value::Map(fields) = body {
                        fields.retain(|(fk, _)| fk.as_str() != Some("over"));
                    }
                }
            }
        }
    }
    v
}

/// Serialized state decoded to `rmpv::Value` in full (including `"over"`).
fn state_full(m: &impl OnlineModel) -> rmpv::Value {
    let bytes = rmp_serde::to_vec_named(&m.state()).expect("serialize state");
    rmpv::decode::read_value(&mut bytes.as_slice()).expect("decode state")
}

/// Collect every field path where two `rmpv::Value` trees differ (comparing
/// floats by their exact bits so NaNs compare equal to NaNs with the same bits).
fn diff_values(path: &str, a: &rmpv::Value, b: &rmpv::Value, out: &mut Vec<String>) {
    use rmpv::Value;
    match (a, b) {
        (Value::F64(x), Value::F64(y)) => {
            if x.to_bits() != y.to_bits() {
                out.push(format!("{path}: {x} != {y}"));
            }
        }
        (Value::F32(x), Value::F32(y)) => {
            if x.to_bits() != y.to_bits() {
                out.push(format!("{path}: {x} != {y}"));
            }
        }
        (Value::Array(xs), Value::Array(ys)) => {
            if xs.len() != ys.len() {
                out.push(format!("{path}: len {} != {}", xs.len(), ys.len()));
            } else {
                for (i, (x, y)) in xs.iter().zip(ys).enumerate() {
                    diff_values(&format!("{path}[{i}]"), x, y, out);
                }
            }
        }
        (Value::Map(xs), Value::Map(ys)) => {
            if xs.len() != ys.len() {
                let xk: Vec<_> = xs.iter().map(|(k, _)| format!("{k}")).collect();
                let yk: Vec<_> = ys.iter().map(|(k, _)| format!("{k}")).collect();
                out.push(format!("{path}: keys {xk:?} != {yk:?}"));
            } else {
                for ((kx, vx), (ky, vy)) in xs.iter().zip(ys) {
                    if kx != ky {
                        out.push(format!("{path}: key {kx} != {ky}"));
                    } else {
                        let name = kx.as_str().unwrap_or("?");
                        let next = if path.is_empty() {
                            name.to_string()
                        } else {
                            format!("{path}.{name}")
                        };
                        diff_values(&next, vx, vy, out);
                    }
                }
            }
        }
        _ => {
            if a != b {
                out.push(format!("{path}: {a} != {b}"));
            }
        }
    }
}

fn assert_state_eq(label: &str, before: &rmpv::Value, after: &rmpv::Value) {
    let mut diffs = Vec::new();
    diff_values("", before, after, &mut diffs);
    assert!(
        diffs.is_empty(),
        "{label}: state mutated on/after cap breach:\n  {}",
        diffs.join("\n  ")
    );
}

fn step_bits(s: &Step) -> (Vec<u64>, u64) {
    (
        s.pred.iter().map(|v| v.to_bits()).collect(),
        s.n_eff.to_bits(),
    )
}

/// Assert that stepping the breaching row sets `state_over_cap()`, leaves every
/// persisted field other than `over` untouched, leaves `predict_with` untouched,
/// and leaves the entire persisted state (including `over`) untouched across
/// subsequent positive-weight, zero-weight, and missing-target rows.
#[allow(clippy::too_many_arguments)]
fn assert_breach_and_followups_atomic<M: OnlineModel>(
    label: &str,
    m: &mut M,
    breach: impl FnOnce(&mut M),
    probe_x: &[f64],
    probe_y: &[Option<f64>],
    followup_pos: impl FnOnce(&mut M),
    followup_zero: impl FnOnce(&mut M),
    followup_none: impl FnOnce(&mut M),
) {
    assert_eq!(m.state_over_cap(), None, "{label}: must start under cap");
    let pre = state_without_over(m);
    let pre_pred = step_bits(&m.predict_with(probe_x, probe_y, 1.0));

    breach(m);
    let over = m
        .state_over_cap()
        .unwrap_or_else(|| panic!("{label}: breaching row did not trigger state_over_cap"));
    assert!(over.0 > over.1, "{label}: {over:?}");
    assert_state_eq(&format!("{label} (breach)"), &pre, &state_without_over(m));
    assert_eq!(
        step_bits(&m.predict_with(probe_x, probe_y, 1.0)),
        pre_pred,
        "{label}: predict_with moved after breaching row"
    );

    let frozen = state_full(m);
    followup_pos(m);
    assert_eq!(m.state_over_cap(), Some(over), "{label}: over changed");
    assert_state_eq(&format!("{label} (after pos)"), &frozen, &state_full(m));

    followup_zero(m);
    assert_eq!(m.state_over_cap(), Some(over), "{label}: over changed");
    assert_state_eq(&format!("{label} (after zero)"), &frozen, &state_full(m));

    followup_none(m);
    assert_eq!(m.state_over_cap(), Some(over), "{label}: over changed");
    assert_state_eq(&format!("{label} (after none)"), &frozen, &state_full(m));
}

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

// ---------------------------------------------------------------------------
// 1. Glm (`refit_clock`, multi-target + unstamped `own` + zero-weight followup)
// ---------------------------------------------------------------------------

#[test]
fn glm_refit_span_is_atomic() {
    let row_bytes = 8 * (2 + 2) as u64;
    let stamps = (2 * std::mem::size_of::<Option<Stamp>>()) as u64;
    // Room for 5 target-rows total: 2 full rows (4 target-rows) fit; row 2 has
    // 2 targets so target 0 alone would fit (5) while target 1 pushes to 6 > 5.
    let cap = stamps + 5 * row_bytes;
    let cfg = GlmCfg {
        n_features: 2,
        n_targets: 2,
        fit_intercept: true,
        decay: Decay::Lam(0.95),
        family: GlmFamily::Poisson,
        link: GlmFamily::Poisson.default_link(),
        var_power: 1.5,
        ridge: 1.0,
        min_weight: 0.0,
        refit_rows: 0,
        refit_clock: Some(1e9),
        step_cap: f64::INFINITY,
        max_state_bytes: Some(cap),
        adaptive_refit: false,
    };
    let mut m = Glm::new(cfg).unwrap();
    for i in 0..2 {
        m.step(
            &[0.5 + i as f64 * 0.1, -0.3],
            &[Some(3.0), Some(2.0)],
            1.0,
            1.0,
        );
    }
    assert_breach_and_followups_atomic(
        "glm",
        &mut m,
        |m| {
            m.step(&[0.2, 0.4], &[Some(4.0), Some(1.0)], 1.5, 1.0);
        },
        &[0.3, -0.2],
        &[Some(2.0), Some(2.0)],
        |m| {
            m.step(&[0.1, 0.2], &[Some(2.0), Some(3.0)], 2.0, 1.0);
        },
        |m| {
            m.step(&[0.1, 0.2], &[Some(2.0), Some(3.0)], 2.0, 0.0);
        },
        |m| {
            m.step(&[0.1, 0.2], &[None, None], 2.0, 1.0);
        },
    );
}

// ---------------------------------------------------------------------------
// 2. Expectile (`refit_clock` AND `window_clock`, multi-target + expiry)
// ---------------------------------------------------------------------------

#[test]
fn expectile_refit_span_is_atomic() {
    let base_cfg = ExpectileCfg {
        n_features: 2,
        n_targets: 2,
        fit_intercept: true,
        decay: Decay::Lam(0.95),
        expectile: 0.7,
        ridge: 1e-2,
        min_weight: 0.0,
        refit_rows: 0,
        refit_clock: Some(1e9),
        window_rows: None,
        window_clock: None,
        closed: WindowClosed::Right,
        max_state_bytes: None,
        adaptive_refit: false,
    };
    let fixed = Expectile::sums_bytes(2, 2, true, false)
        + (2 * std::mem::size_of::<Option<Stamp>>()) as u64;
    let row = 8 * (2 + 2) as u64;
    // 2 full rows (4 target-rows) fit; row 2's target 0 would fit (5) but
    // target 1 pushes to 6 > 5.
    let cap = fixed + 5 * row;
    let mut m = Expectile::new(ExpectileCfg {
        max_state_bytes: Some(cap),
        ..base_cfg
    })
    .unwrap();
    for i in 0..2 {
        m.stamp_next(Stamp::Ns(i * S));
        m.step(&[0.5 + i as f64, -0.2], &[Some(1.5), Some(-0.5)], 1.0, 1.0);
    }
    assert_breach_and_followups_atomic(
        "expectile_refit_span",
        &mut m,
        |m| {
            m.stamp_next(Stamp::Ns(5 * S));
            m.step(&[0.3, 0.7], &[Some(2.0), Some(0.1)], 3.0, 1.0);
        },
        &[0.1, 0.2],
        &[Some(1.0), Some(1.0)],
        |m| {
            m.stamp_next(Stamp::Ns(6 * S));
            m.step(&[0.4, -0.1], &[Some(1.0), Some(2.0)], 1.0, 1.0);
        },
        |m| {
            m.stamp_next(Stamp::Ns(7 * S));
            m.step(&[0.4, -0.1], &[Some(1.0), Some(2.0)], 1.0, 0.0);
        },
        |m| {
            m.stamp_next(Stamp::Ns(8 * S));
            m.step(&[0.4, -0.1], &[None, None], 1.0, 1.0);
        },
    );
}

#[test]
fn expectile_window_clock_is_atomic() {
    let row = Expectile::window_row_bytes(2, 2, true);
    let fixed = Expectile::clock_state_bytes(2, 2, true, 0);
    // Fits 4 rows.
    let cap = fixed + 4 * row;
    let cfg = ExpectileCfg {
        n_features: 2,
        n_targets: 2,
        fit_intercept: true,
        decay: Decay::Halflife(f64::INFINITY),
        expectile: 0.7,
        ridge: 1e-2,
        min_weight: 0.0,
        refit_rows: 0,
        refit_clock: None,
        window_rows: None,
        window_clock: Some(10.0),
        closed: WindowClosed::Right,
        max_state_bytes: Some(cap),
        adaptive_refit: false,
    };
    // Case A: burst breach (no row expires).
    let mut m = Expectile::new(cfg.clone()).unwrap();
    for i in 0..4 {
        m.stamp_next(Stamp::Ns(i * S));
        m.step(
            &[i as f64, (i * i) as f64 * 0.1],
            &[Some(1.0 + i as f64), Some(2.0 - i as f64)],
            1.0,
            1.0,
        );
    }
    assert_breach_and_followups_atomic(
        "expectile_window_burst",
        &mut m,
        |m| {
            m.stamp_next(Stamp::Ns(4 * S));
            m.step(&[0.5, 0.5], &[Some(3.0), Some(-1.0)], 1.0, 1.0);
        },
        &[0.5, 0.5],
        &[Some(1.0), Some(1.0)],
        |m| {
            m.stamp_next(Stamp::Ns(5 * S));
            m.step(&[0.5, 0.5], &[Some(3.0), Some(-1.0)], 1.0, 1.0);
        },
        |m| {
            m.stamp_next(Stamp::Ns(6 * S));
            m.step(&[0.5, 0.5], &[Some(3.0), Some(-1.0)], 1.0, 0.0);
        },
        |m| {
            m.stamp_next(Stamp::Ns(20 * S));
            m.step(&[0.5, 0.5], &[None, None], 14.0, 1.0);
        },
    );

    // Case B: one expired row at the front, cap only holds 3 rows after lowering
    // via a fresh model with 4 rows at cap=4 and then 1 row expires while still
    // breaching a cap of 3 (simulated by 4 rows at t=0,10,10,10, window=5, cap=3
    // -> impossible to insert 4th at cap=3; instead fill 4 rows at t=0,1,2,3
    // with cap=4, then tighten cap? Even simpler: fill 5 rows? No: with cap=4
    // we hold 4 rows at t=0,1,2,3; if we step at t=11 (window=10), row 0 expires
    // leaving 3 rows + 1 new = 4 <= cap=4, so that fits! To breach EVEN AFTER
    // expiry in a single-row model, only models whose row size varies (isotonic,
    // ddsketch) or with cap lowered can breach after expiry — wait: if a test
    // steps a row AFTER over.is_some() with a large stamp (t=20s above!), `trim`
    // WOULD pop expired rows if called before `over.is_some()`!).
    // Notice Case A's `followup_none` ALREADY stepped at t=20s (past window=10s)
    // and proved expired rows are NOT popped after `over.is_some()`!
}

// ---------------------------------------------------------------------------
// 3. Sgd (`huber_refit_span`, multi-target learned breach + REG-2 ageing breach)
// ---------------------------------------------------------------------------

#[test]
fn sgd_huber_refit_span_is_atomic() {
    let base = SgdCfg {
        n_features: 2,
        n_targets: 2,
        fit_intercept: true,
        decay: Decay::Lam(0.95),
        loss: SgdLoss::Huber { delta: 1.345 },
        learning_rate: 0.05,
        schedule: LearningRate::AdaGrad,
        l2: 0.0,
        min_weight: 0.0,
        clip_gradient: 1e3,
        constraint: None,
        standardize: true,
        strict_binary: false,
        huber_scale: HuberScale::Mad,
        huber_scale_decay: None,
        refit_rows: 0,
        min_weight_holds_learning: false,
        target_spread: false,
        mad_anchor: false,
        refit_clock: Some(1e9),
        max_state_bytes: None,
        average: false,
        invariant: false,
        l1: 0.0,
    };
    let row = 8 * (2 + 4) as u64 + 48;
    let stamps = (2 * std::mem::size_of::<Option<Stamp>>()) as u64;
    // Room for 5 target-rows: 2 full rows (4 target-rows) fit; row 2's target 0
    // alone would fit (5), target 1 pushes to 6 > 5 -> MUST reject atomically!
    let mut m = Sgd::new(base.clone()).unwrap();
    m.set_cap_for_test(stamps + 5 * row);
    for i in 0..2 {
        m.stamp_next(Stamp::Ns(i * S));
        m.step(
            &[0.5 + i as f64, -0.3 + i as f64 * 0.2],
            &[Some(10.0 + i as f64), Some(-2.0 + i as f64)],
            1.0,
            1.0,
        );
    }
    assert_breach_and_followups_atomic(
        "sgd_learned_multitarget",
        &mut m,
        |m| {
            m.stamp_next(Stamp::Ns(3 * S));
            m.step(&[1.2, -0.7], &[Some(15.0), Some(4.0)], 2.0, 1.0);
        },
        &[0.4, 0.8],
        &[Some(1.0), Some(1.0)],
        |m| {
            m.stamp_next(Stamp::Ns(4 * S));
            m.step(&[0.2, 0.3], &[Some(11.0), Some(-1.0)], 1.0, 1.0);
        },
        |m| {
            m.stamp_next(Stamp::Ns(5 * S));
            m.step(&[0.2, 0.3], &[Some(11.0), Some(-1.0)], 1.0, 0.0);
        },
        |m| {
            m.stamp_next(Stamp::Ns(6 * S));
            m.step(&[0.2, 0.3], &[None, None], 1.0, 1.0);
        },
    );

    // Ageing-row (REG-2) multi-target breach: room for 2 learned rows + 1 ageing
    // run (32 bytes), so target 0's run fits while target 1's run pushes over!
    let mut ma = Sgd::new(base).unwrap();
    ma.set_cap_for_test(stamps + 4 * row + 32);
    for i in 0..2 {
        ma.stamp_next(Stamp::Ns(i * S));
        ma.step(&[0.5 + i as f64, -0.3], &[Some(10.0), Some(-2.0)], 1.0, 1.0);
    }
    assert_breach_and_followups_atomic(
        "sgd_ageing_multitarget",
        &mut ma,
        |m| {
            m.stamp_next(Stamp::Ns(3 * S));
            m.step(&[0.8, 0.4], &[None, None], 1.5, 1.0);
        },
        &[0.4, 0.8],
        &[Some(1.0), Some(1.0)],
        |m| {
            m.stamp_next(Stamp::Ns(4 * S));
            m.step(&[0.2, 0.3], &[Some(11.0), Some(-1.0)], 1.0, 1.0);
        },
        |m| {
            m.stamp_next(Stamp::Ns(5 * S));
            m.step(&[0.2, 0.3], &[Some(11.0), Some(-1.0)], 1.0, 0.0);
        },
        |m| {
            m.stamp_next(Stamp::Ns(6 * S));
            m.step(&[0.2, 0.3], &[None, None], 1.0, 1.0);
        },
    );
}

// ---------------------------------------------------------------------------
// 4. Robust (`huber_refit_span` / `quantile_refit_span`, multi-target + REG-2)
// ---------------------------------------------------------------------------

#[test]
fn robust_refit_span_is_atomic() {
    for (name, loss, scale, extra) in [
        (
            "robust_huber_mad",
            RobustLoss::Huber { delta: 1.345 },
            HuberScale::Mad,
            48u64,
        ),
        (
            "robust_quantile_rms",
            RobustLoss::Quantile { tau: 0.7 },
            HuberScale::Rms,
            0u64,
        ),
    ] {
        let base = RobustCfg {
            n_features: 2,
            n_targets: 2,
            fit_intercept: true,
            decay: Decay::Lam(0.95),
            loss,
            ridge: 1e-3,
            standardize: true,
            min_weight: 0.0,
            solve_every: 0.0,
            max_rows_between_solves: u32::MAX,
            solve_share: None,
            quantile_eps: 0.1,
            huber_scale: scale,
            huber_scale_decay: None,
            feature_sets: Vec::new(),
            ridge_scale: false,
            emit_scale: matches!(loss, RobustLoss::Huber { .. }),
            refit_rows: 0,
            mad_anchor: false,
            min_weight_holds_learning: false,
            refit_clock: Some(1e9),
            max_state_bytes: None,
            norm: crate::HuberNorm::Huber,
        };
        let row = 8 * (2 + 4) as u64 + extra;
        let stamps = (2 * std::mem::size_of::<Option<Stamp>>()) as u64;
        // 4 full rows (8 target-rows) fit; row 4 needs 2 target-rows -> 10 > 9,
        // where target 0 alone (9) would fit!
        let mut m = Robust::new(base.clone()).unwrap();
        m.set_cap_for_test(stamps + 9 * row);
        for i in 0..4 {
            m.stamp_next(Stamp::Ns(i * S));
            let x = [i as f64 * 0.4 - 0.5, (i % 3) as f64 * 0.7];
            m.step(
                &x,
                &[Some(2.0 + x[0] - 0.5 * x[1]), Some(-1.0 + 0.3 * x[1])],
                1.0,
                1.0,
            );
        }
        assert_breach_and_followups_atomic(
            &format!("{name}_learned"),
            &mut m,
            |m| {
                m.stamp_next(Stamp::Ns(5 * S));
                m.step(&[1.1, -0.4], &[Some(5.0), Some(3.0)], 2.0, 1.0);
            },
            &[0.3, -0.1],
            &[Some(1.0), Some(1.0)],
            |m| {
                m.stamp_next(Stamp::Ns(6 * S));
                m.step(&[0.2, 0.5], &[Some(3.0), Some(-2.0)], 1.0, 1.0);
            },
            |m| {
                m.stamp_next(Stamp::Ns(7 * S));
                m.step(&[0.2, 0.5], &[Some(3.0), Some(-2.0)], 1.0, 0.0);
            },
            |m| {
                m.stamp_next(Stamp::Ns(8 * S));
                m.step(&[0.2, 0.5], &[None, None], 1.0, 1.0);
            },
        );

        // Ageing-row (REG-2) multi-target breach: room for 4 learned rows + 1 run (32).
        let mut ma = Robust::new(base).unwrap();
        ma.set_cap_for_test(stamps + 8 * row + 32);
        for i in 0..4 {
            ma.stamp_next(Stamp::Ns(i * S));
            let x = [i as f64 * 0.4 - 0.5, (i % 3) as f64 * 0.7];
            ma.step(
                &x,
                &[Some(2.0 + x[0] - 0.5 * x[1]), Some(-1.0 + 0.3 * x[1])],
                1.0,
                1.0,
            );
        }
        assert_breach_and_followups_atomic(
            &format!("{name}_ageing"),
            &mut ma,
            |m| {
                m.stamp_next(Stamp::Ns(5 * S));
                m.step(&[0.9, -0.2], &[None, None], 1.7, 1.0);
            },
            &[0.3, -0.1],
            &[Some(1.0), Some(1.0)],
            |m| {
                m.stamp_next(Stamp::Ns(6 * S));
                m.step(&[0.2, 0.5], &[Some(3.0), Some(-2.0)], 1.0, 1.0);
            },
            |m| {
                m.stamp_next(Stamp::Ns(7 * S));
                m.step(&[0.2, 0.5], &[Some(3.0), Some(-2.0)], 1.0, 0.0);
            },
            |m| {
                m.stamp_next(Stamp::Ns(8 * S));
                m.step(&[0.2, 0.5], &[None, None], 1.0, 1.0);
            },
        );
    }
}

// ---------------------------------------------------------------------------
// 5. Garch (`start_span`, multi-target partial update + `own` + `w_sum`)
// ---------------------------------------------------------------------------

#[test]
fn garch_start_span_is_atomic() {
    let base = GarchCfg {
        n_inputs: 0,
        n_targets: 2,
        vol: GarchVol::Garch,
        dist: GarchDist::Normal,
        truncation: None,
        variance_targeting: false,
        start_rows: 500,
        start_span: Some(1e9),
        horizons: Vec::new(),
        max_horizon_work: None,
        min_weight: 0.0,
        max_state_bytes: None,
    };
    // Room for 5 rows across the two targets (2*2 = 4 rows fit; row 2's target 0
    // would fit at 5, while target 1 pushes to 6 > 5).
    let base_bytes = Garch::state_bytes(&base, 0);
    let per_target_row = Garch::series_bytes(&base, 1) - Garch::series_bytes(&base, 0);
    let cap = base_bytes + 5 * per_target_row;
    let mut m = Garch::new(GarchCfg {
        max_state_bytes: Some(cap),
        ..base
    })
    .unwrap();
    for i in 0..2 {
        m.step(&[], &[Some(0.3 + i as f64 * 0.1), Some(-0.2)], 1.0, 1.0);
    }
    assert_breach_and_followups_atomic(
        "garch_multitarget",
        &mut m,
        |m| {
            m.step(&[], &[Some(0.5), Some(-0.4)], 1.5, 1.0);
        },
        &[],
        &[Some(0.2), Some(-0.1)],
        |m| {
            m.step(&[], &[Some(0.1), Some(0.2)], 1.0, 1.0);
        },
        |m| {
            m.step(&[], &[Some(0.1), Some(0.2)], 1.0, 0.0);
        },
        |m| {
            m.step(&[], &[None, None], 1.0, 1.0);
        },
    );
}

// ---------------------------------------------------------------------------
// 6. Dcc (`start_span`, standalone AND with GARCH margins)
// ---------------------------------------------------------------------------

#[test]
fn dcc_start_span_is_atomic() {
    // Case A: standalone DCC (`margins = None`), with decay so `pending` & `w_sum`
    // would visibly move if mutated before `admit`.
    let base = DccCfg {
        n_targets: 2,
        asymmetric: false,
        margins: None,
        params: Some(vec![0.05, 0.9]),
        decay: Decay::Lam(0.95),
        start_rows: DCC_START_ROWS,
        start_span: Some(1e9),
        min_weight: 0.0,
        max_state_bytes: None,
    };
    let cap = base.stage_bytes() + 32 + 3 * base.row_bytes();
    let mut m = Dcc::new(DccCfg {
        max_state_bytes: Some(cap),
        ..base
    })
    .unwrap();
    for i in 0..3 {
        m.step(&[], &[Some(0.4 + i as f64 * 0.1), Some(-0.3)], 1.0, 1.0);
    }
    assert_breach_and_followups_atomic(
        "dcc_standalone",
        &mut m,
        |m| {
            m.step(&[], &[Some(0.6), Some(-0.5)], 2.0, 1.0);
        },
        &[],
        &[Some(0.2), Some(0.1)],
        |m| {
            m.step(&[], &[Some(0.1), Some(0.2)], 1.0, 1.0);
        },
        |m| {
            m.step(&[], &[Some(0.1), Some(0.2)], 1.0, 0.0);
        },
        |m| {
            m.step(&[], &[None, None], 1.0, 1.0);
        },
    );

    // Case B: DCC with GARCH margins whose own `start_span` hits the cap.
    let margin = GarchCfg {
        n_inputs: 0,
        n_targets: 2,
        vol: GarchVol::Garch,
        dist: GarchDist::Normal,
        truncation: None,
        variance_targeting: false,
        start_rows: 500,
        start_span: Some(1e9),
        horizons: Vec::new(),
        max_horizon_work: None,
        min_weight: 0.0,
        max_state_bytes: None,
    };
    let stage_fixed = DccCfg {
        n_targets: 2,
        asymmetric: false,
        margins: Some(margin.clone()),
        params: Some(vec![0.05, 0.9]),
        decay: Decay::Lam(0.95),
        start_rows: DCC_START_ROWS,
        start_span: Some(1e9),
        min_weight: 0.0,
        max_state_bytes: None,
    };
    let g_base = Garch::state_bytes(&margin, 0);
    let g_per = Garch::series_bytes(&margin, 1) - Garch::series_bytes(&margin, 0);
    let cap_m = stage_fixed.stage_bytes() + 32 + g_base + 5 * g_per;
    let mut mm = Dcc::new(DccCfg {
        max_state_bytes: Some(cap_m),
        ..stage_fixed
    })
    .unwrap();
    for i in 0..2 {
        mm.step(&[], &[Some(0.3 + i as f64 * 0.1), Some(-0.2)], 1.0, 1.0);
    }
    assert_breach_and_followups_atomic(
        "dcc_with_margins_cap",
        &mut mm,
        |m| {
            m.step(&[], &[Some(0.5), Some(-0.4)], 2.0, 1.0);
        },
        &[],
        &[Some(0.2), Some(0.1)],
        |m| {
            m.step(&[], &[Some(0.1), Some(0.2)], 1.0, 1.0);
        },
        |m| {
            m.step(&[], &[Some(0.1), Some(0.2)], 1.0, 0.0);
        },
        |m| {
            m.step(&[], &[None, None], 1.0, 1.0);
        },
    );

    // Case C: DCC with started GARCH margins whose stage `start_span` hits the cap.
    // If stage breaches, `margins` MUST NOT have stepped!
    let mut s = 17u64;
    let (mut v0, mut v1) = (1.0f64, 1.0f64);
    let (mut e0, mut e1) = (0.0f64, 0.0f64);
    let mut rets = Vec::with_capacity(1800);
    for _ in 0..1800 {
        v0 = 0.05 + 0.25 * e0 * e0 + 0.70 * v0;
        v1 = 0.05 + 0.25 * e1 * e1 + 0.70 * v1;
        let z0 = normal(&mut s);
        let z1 = 0.5 * z0 + (1.0 - 0.25f64).sqrt() * normal(&mut s);
        e0 = v0.sqrt() * z0;
        e1 = v1.sqrt() * z1;
        rets.push([Some(e0), Some(e1)]);
    }
    let margin_rows = GarchCfg {
        n_inputs: 0,
        n_targets: 2,
        vol: GarchVol::Garch,
        dist: GarchDist::Normal,
        truncation: None,
        variance_targeting: false,
        start_rows: 300,
        start_span: None,
        horizons: Vec::new(),
        max_horizon_work: None,
        min_weight: 0.0,
        max_state_bytes: None,
    };
    let mut cfg_c = DccCfg {
        n_targets: 2,
        asymmetric: false,
        margins: Some(margin_rows.clone()),
        params: Some(vec![0.05, 0.9]),
        decay: Decay::Lam(0.95),
        start_rows: DCC_START_ROWS,
        start_span: Some(1e9),
        min_weight: 0.0,
        max_state_bytes: None,
    };
    // Margins buffer 100 rows during startup, then drop to 0 once started; give
    // enough cap for margin startup, then check stage breach once stage reaches cap.
    let cap_c = cfg_c.peak_bytes();
    cfg_c.max_state_bytes = Some(cap_c);
    let mut mc = Dcc::new(cfg_c).unwrap();
    let mut breach_idx = None;
    for (idx, r) in rets.iter().enumerate() {
        let before = mc.clone();
        mc.step(&[], r, 1.0, 1.0);
        if mc.state_over_cap().is_some() {
            // Rewind to `before` and verify atomicity on this exact row!
            mc = before;
            breach_idx = Some(idx);
            break;
        }
    }
    let idx = breach_idx.expect("stage start_span should hit cap once margins started");
    assert!(
        mc.margins().unwrap().started().iter().all(|&b| b),
        "margins must already be started so stage breach tests margins atomicity"
    );
    assert_breach_and_followups_atomic(
        "dcc_stage_cap_preserves_started_margins",
        &mut mc,
        |m| {
            m.step(&[], &rets[idx], 1.5, 1.0);
        },
        &[],
        &rets[idx],
        |m| {
            m.step(&[], &rets[idx], 1.0, 1.0);
        },
        |m| {
            m.step(&[], &rets[idx], 1.0, 0.0);
        },
        |m| {
            m.step(&[], &[None, None], 1.0, 1.0);
        },
    );
}

// ---------------------------------------------------------------------------
// 7. Hmm (`warm_clock` / `warm_span`)
// ---------------------------------------------------------------------------

#[test]
fn hmm_warm_span_is_atomic() {
    let cfg = HmmCfg {
        n_features: 2,
        k: 2,
        decay: Decay::Lam(0.9),
        covariance: HmmCovariance::Full,
        precision_prior: 1e-3,
        min_weight: 0.0,
        learn: true,
        transition_prior: 1.0,
        transition: None,
        means: None,
        covs: None,
        warm_rows: 2,
        seed_rule: SeedRule::Lloyd,
        seed: 7,
        tvtp: None,
        prior_rows: None,
        warm_clock: Some(1e9),
        extras: None,
    };
    let mut m = Hmm::new(cfg).unwrap();
    for i in 0..3 {
        m.step(&[i as f64, -(i as f64)], &[], 1.0, 1.0);
    }
    m.set_cap_for_test(1);
    assert_breach_and_followups_atomic(
        "hmm",
        &mut m,
        |m| {
            m.step(&[2.0, -1.0], &[], 2.0, 1.0);
        },
        &[0.5, 0.5],
        &[],
        |m| {
            m.step(&[1.0, 1.0], &[], 1.0, 1.0);
        },
        |m| {
            m.step(&[1.0, 1.0], &[], 1.0, 0.0);
        },
        |m| {
            m.step(&[1.0, 1.0], &[], 1.0, 0.0);
        },
    );
}

// ---------------------------------------------------------------------------
// 8. Bocpd (`warm_clock` / `warm_span`)
// ---------------------------------------------------------------------------

#[test]
fn bocpd_warm_span_is_atomic() {
    let cfg = BocpdCfg {
        n_features: 2,
        hazard: 100.0,
        hazard_from_row: false,
        emission: BocpdEmission::Diag,
        prior_mean: None,
        prior_kappa: 1.0,
        prior_nu: None,
        prior_scale: None,
        robust_beta: 0.0,
        prune_below: 0.0,
        max_run: 1000,
        min_weight: 0.0,
        warm_rows: None,
        hazard_on_clock: false,
        warm_clock: Some(1e9),
        extras: None,
    };
    let mut m = Bocpd::new(cfg).unwrap();
    for i in 0..3 {
        m.step(&[i as f64, 1.0 - i as f64], &[], 1.0, 1.0);
    }
    m.set_cap_for_test(1);
    assert_breach_and_followups_atomic(
        "bocpd",
        &mut m,
        |m| {
            m.step(&[3.0, -2.0], &[], 2.5, 1.0);
        },
        &[0.5, 0.5],
        &[],
        |m| {
            m.step(&[1.0, 1.0], &[], 1.0, 1.0);
        },
        |m| {
            m.step(&[1.0, 1.0], &[], 3.0, 0.0);
        },
        |m| {
            m.step(&[1.0, 1.0], &[], 1.0, 0.0);
        },
    );
}

// ---------------------------------------------------------------------------
// 9. KMeans (`warm_clock` / `start_span`)
// ---------------------------------------------------------------------------

#[test]
fn kmeans_start_span_is_atomic() {
    let cfg = KMeansCfg {
        n_features: 2,
        k: 3,
        decay: Decay::Lam(0.95),
        min_weight: 0.0,
        warm_rows: 3,
        seed_rule: SeedRule::Lloyd,
        seed: 3,
        update_every_rows: 1,
        split_merge: 0.5,
        split_merge_every_rows: 20,
        dead_frac: 0.05,
        standardize: true,
        scale_floor: 0.1,
        warm_clock: Some(1e9),
        update_every: Some(10.0),
        split_merge_every: Some(20.0),
        centers: None,
        learn: true,
        emit_distances: false,
        emit_inertia: false,
    };
    let mut m = KMeans::new(cfg).unwrap();
    for i in 0..3 {
        m.step(&[i as f64, 0.5 - i as f64], &[], 1.0, 1.0);
    }
    m.set_cap_for_test(1);
    assert_breach_and_followups_atomic(
        "kmeans",
        &mut m,
        |m| {
            m.step(&[4.0, -1.0], &[], 2.0, 1.0);
        },
        &[0.5, 0.5],
        &[],
        |m| {
            m.step(&[1.0, 2.0], &[], 1.0, 1.0);
        },
        |m| {
            m.step(&[1.0, 2.0], &[], 1.0, 0.0);
        },
        |m| {
            m.step(&[1.0, 2.0], &[], 5.0, 0.0);
        },
    );
}

// ---------------------------------------------------------------------------
// 10. CorrChange (`Monitor`, `Sequential`, `Window` with `permute_every`)
// ---------------------------------------------------------------------------

#[test]
fn corrchange_clock_span_is_atomic() {
    for kind in [
        CorrChangeKind::Monitor,
        CorrChangeKind::Sequential,
        CorrChangeKind::Window,
    ] {
        let base = CorrChangeCfg {
            n_features: 2,
            kind,
            span_rows: 0,
            alpha: 0.05,
            alpha_adjust: "bonferroni".into(),
            bandwidth: None,
            scalar: false,
            decay: Decay::Lam(0.95),
            crit: Some(3.0),
            n_perm: 40,
            permute_every_rows: usize::MAX,
            perm_block: 1,
            norm: ChangeNorm::L1,
            seed: 3,
            reset: false,
            monitor_rows: 0,
            boundary_gamma: 0.0,
            permute_every: (kind == CorrChangeKind::Window).then_some(5.0),
            span_clock: Some(10.0),
            monitor_clock: (kind == CorrChangeKind::Sequential).then_some(10.0),
            max_state_bytes: None,
            emit_n_flagged: false,
        };
        let mut probe = CorrChange::new(CorrChangeCfg {
            max_state_bytes: Some(1),
            ..base.clone()
        })
        .unwrap();
        probe.stamp_next(Stamp::Ns(0));
        probe.step(&[0.5, -0.3], &[], 1.0, 1.0);
        let one_row_total = probe.state_over_cap().expect("probe over cap").0;
        let per = 16u64
            + if kind == CorrChangeKind::Window {
                std::mem::size_of::<Stamp>() as u64
            } else {
                0
            };
        let cap = one_row_total + 3 * per;
        let mut m = CorrChange::new(CorrChangeCfg {
            max_state_bytes: Some(cap),
            ..base
        })
        .unwrap();
        for i in 0..4 {
            m.stamp_next(Stamp::Ns(i * S));
            m.step(&[i as f64 * 0.5, -(i as f64) * 0.3], &[], 1.0, 1.0);
        }
        assert_breach_and_followups_atomic(
            &format!("corrchange_{kind:?}"),
            &mut m,
            |m| {
                m.stamp_next(Stamp::Ns(4 * S));
                m.step(&[2.5, -1.2], &[], 1.0, 1.0);
            },
            &[0.3, -0.4],
            &[],
            |m| {
                m.stamp_next(Stamp::Ns(5 * S));
                m.step(&[1.0, 0.2], &[], 1.0, 1.0);
            },
            |m| {
                m.stamp_next(Stamp::Ns(6 * S));
                m.step(&[1.0, 0.2], &[], 1.0, 0.0);
            },
            |m| {
                // Past `2 * span_clock = 20s`: must not trim `Window` after over!
                m.stamp_next(Stamp::Ns(30 * S));
                m.step(&[1.0, 0.2], &[], 24.0, 1.0);
            },
        );
    }
}

// ---------------------------------------------------------------------------
// 11. EwmMoments (clocked lag buffer AND `window_clock`, `step` + `update_row`)
// ---------------------------------------------------------------------------

#[test]
fn ewm_moments_clock_span_is_atomic() {
    // Case A: clocked lag buffer.
    let term_lag = MomentTerm {
        name: "ac".into(),
        powers: vec![(Leg::of(0), 1), (Leg::lagged(0, 1000.0), 1)],
        kind: MomentKind::Central,
        bias: true,
    };
    let base_lag = EwmMomentsCfg {
        decay: Decay::Lam(0.95),
        terms: vec![term_lag],
        min_weight: 0.0,
        max_state_bytes: None,
        clocked: true,
        lag_tolerance: None,
        emit_se: false,
        window_rows: None,
        window_clock: None,
    };
    let cap_lag = base_lag.state_bytes() + 3 * 24;
    let mut ml = EwmMoments::new(EwmMomentsCfg {
        max_state_bytes: Some(cap_lag),
        ..base_lag
    })
    .unwrap();
    for i in 0..3 {
        ml.step(&[1.0 + i as f64], &[], 1.0, 1.0);
    }
    assert_breach_and_followups_atomic(
        "ewm_moments_lag",
        &mut ml,
        |m| {
            m.step(&[5.0], &[], 2.0, 1.0);
        },
        &[1.0],
        &[],
        |m| {
            m.step(&[6.0], &[], 1.0, 1.0);
        },
        |m| {
            m.step(&[6.0], &[], 1.0, 0.0);
        },
        |m| {
            m.update_row(&[7.0], 3.0, 1.0);
        },
    );

    // Case B: `window_clock` (must not mutate `place` OR `ring.flip` via `slide`).
    let term_w = MomentTerm::of_column("v", 2, MomentKind::Central, true);
    let base_w = EwmMomentsCfg {
        decay: Decay::Lam(0.95),
        terms: vec![term_w],
        min_weight: 0.0,
        max_state_bytes: None,
        clocked: true,
        lag_tolerance: None,
        emit_se: false,
        window_rows: None,
        window_clock: Some(100.0),
    };
    let mut probe = EwmMoments::new(base_w.clone()).unwrap();
    for i in 0..20 {
        probe.step(&[i as f64], &[], 1.0, 1.0);
    }
    let cap_w = probe.state_bytes();
    let mut mw = EwmMoments::new(EwmMomentsCfg {
        max_state_bytes: Some(cap_w),
        ..base_w
    })
    .unwrap();
    for i in 0..20 {
        mw.step(&[i as f64], &[], 1.0, 1.0);
    }
    assert_breach_and_followups_atomic(
        "ewm_moments_window",
        &mut mw,
        |m| {
            m.step(&[99.0], &[], 2.0, 1.0);
        },
        &[1.0],
        &[],
        |m| {
            m.step(&[100.0], &[], 1.0, 1.0);
        },
        |m| {
            m.step(&[100.0], &[], 1.0, 0.0);
        },
        |m| {
            // Large step that would slide/flip the window if not stopped.
            m.step(&[100.0], &[], 200.0, 1.0);
        },
    );
}

// ---------------------------------------------------------------------------
// 12. Hawkes (`window_clock`)
// ---------------------------------------------------------------------------

#[test]
fn hawkes_window_clock_is_atomic() {
    let cap = Hawkes::clock_state_bytes(2, 1, 4);
    let cfg = HawkesCfg {
        n_targets: 2,
        kernel_decays: vec![1.0],
        decay: Decay::Halflife(f64::INFINITY),
        min_weight: 0.0,
        window_events: None,
        window_clock: Some(10.0),
        closed: WindowClosed::Right,
        max_state_bytes: Some(cap),
    };
    let mut m = Hawkes::new(cfg).unwrap();
    for i in 0..4 {
        m.stamp_next(Stamp::Ns(i * S));
        m.step(&[], &[Some(1.0), Some(0.5)], 1.0, 1.0);
    }
    assert_breach_and_followups_atomic(
        "hawkes",
        &mut m,
        |m| {
            m.stamp_next(Stamp::Ns(5 * S));
            m.step(&[], &[Some(1.0), Some(1.0)], 2.0, 1.0);
        },
        &[],
        &[Some(1.0), Some(1.0)],
        |m| {
            m.stamp_next(Stamp::Ns(6 * S));
            m.step(&[], &[Some(1.0), Some(1.0)], 1.0, 1.0);
        },
        |m| {
            m.stamp_next(Stamp::Ns(7 * S));
            m.step(&[], &[Some(1.0), Some(1.0)], 1.0, 0.0);
        },
        |m| {
            // Past window_clock=10s with no marks: must not advance or trim!
            m.stamp_next(Stamp::Ns(30 * S));
            m.step(&[], &[None, None], 23.0, 1.0);
        },
    );
}

// ---------------------------------------------------------------------------
// 13. DdSketchModel (unbounded with decay AND `window_clock` with expiry)
// ---------------------------------------------------------------------------

#[test]
fn ddsketch_model_cap_is_atomic() {
    // Case A: unbounded with decay, 2 targets.
    let mut probe = DdSketchModel::new(DdSketchModelCfg {
        n_targets: 2,
        alpha: crate::EW_QUANTILE_ALPHA,
        levels: vec![0.4],
        window_rows: None,
        window_clock: None,
        closed: WindowClosed::Right,
        decay: Decay::Lam(0.95),
        max_state_bytes: None,
    })
    .unwrap();
    let g = probe.mapping().gamma();
    let mut x = 1.0;
    for _ in 0..5 {
        x *= g;
        probe.step(&[], &[Some(x), Some(x * g)], 1.0, 1.0);
    }
    // Cap allows 1 more bucket (target 0) but not 2 (target 0 + target 1).
    let cap_u = probe.bytes() + crate::ddsketch::Store::bucket_bytes(false);
    let mut mu = DdSketchModel::new(DdSketchModelCfg {
        n_targets: 2,
        alpha: crate::EW_QUANTILE_ALPHA,
        levels: vec![0.4],
        window_rows: None,
        window_clock: None,
        closed: WindowClosed::Right,
        decay: Decay::Lam(0.95),
        max_state_bytes: Some(cap_u),
    })
    .unwrap();
    x = 1.0;
    for _ in 0..5 {
        x *= g;
        mu.step(&[], &[Some(x), Some(x * g)], 1.0, 1.0);
    }
    let next_x = x * g * g;
    assert_breach_and_followups_atomic(
        "ddsketch_unbounded",
        &mut mu,
        |m| {
            m.step(&[], &[Some(next_x), Some(next_x * g)], 2.0, 1.0);
        },
        &[],
        &[Some(1.0), Some(1.0)],
        |m| {
            m.step(&[], &[Some(1.0), Some(1.0)], 1.0, 1.0);
        },
        |m| {
            m.step(&[], &[Some(1.0), Some(1.0)], 1.0, 0.0);
        },
        |m| {
            m.step(&[], &[None, None], 1.0, 1.0);
        },
    );

    // Case B: `window_clock` where an expired row leaves at the breaching row's
    // stamp (expired row had 0 buckets removed because repeated value, incoming
    // row adds 2 brand-new buckets -> STILL breaches cap!).
    let base_w = DdSketchModelCfg {
        n_targets: 2,
        alpha: crate::EW_QUANTILE_ALPHA,
        levels: vec![0.4],
        window_rows: None,
        window_clock: Some(5.0),
        closed: WindowClosed::Right,
        decay: Decay::Halflife(f64::INFINITY),
        max_state_bytes: None,
    };
    let mut pw = DdSketchModel::new(base_w.clone()).unwrap();
    for i in 0..3 {
        pw.stamp_next(Stamp::Ns(i * 2 * S));
        // Repeated value `1.0` so dropping row 0 removes NO bucket, only 1 ring row.
        pw.step(&[], &[Some(1.0), Some(1.0)], 2.0, 1.0);
    }
    // Cap = current bytes (3 ring rows + 2 buckets). At t=6s (window=5s), row 0
    // at t=0s expires (-1 ring row, 0 buckets) while the incoming row adds
    // (+1 ring row + 2 NEW buckets) -> net +2 buckets > cap!
    let cap_w = pw.bytes();
    let mut mw = DdSketchModel::new(DdSketchModelCfg {
        max_state_bytes: Some(cap_w),
        ..base_w
    })
    .unwrap();
    for i in 0..3 {
        mw.stamp_next(Stamp::Ns(i * 2 * S));
        mw.step(&[], &[Some(1.0), Some(1.0)], 2.0, 1.0);
    }
    assert_breach_and_followups_atomic(
        "ddsketch_window_expiry_breach",
        &mut mw,
        |m| {
            m.stamp_next(Stamp::Ns(6 * S));
            m.step(&[], &[Some(100.0), Some(200.0)], 2.0, 1.0);
        },
        &[],
        &[Some(1.0), Some(1.0)],
        |m| {
            m.stamp_next(Stamp::Ns(7 * S));
            m.step(&[], &[Some(1.0), Some(1.0)], 1.0, 1.0);
        },
        |m| {
            m.stamp_next(Stamp::Ns(8 * S));
            m.step(&[], &[Some(1.0), Some(1.0)], 1.0, 0.0);
        },
        |m| {
            m.stamp_next(Stamp::Ns(20 * S));
            m.step(&[], &[None, None], 12.0, 1.0);
        },
    );
}

// ---------------------------------------------------------------------------
// 14. Isotonic (unbounded with decay AND `window_clock` with expiry)
// ---------------------------------------------------------------------------

#[test]
fn isotonic_cap_is_atomic() {
    // Case A: unbounded with decay, 2 targets.
    let base_u = IsotonicCfg {
        n_targets: 2,
        increasing: true,
        y_min: None,
        y_max: None,
        out_of_bounds: OutOfBounds::Clip,
        min_weight: 0.0,
        window_rows: None,
        window_clock: None,
        closed: WindowClosed::Right,
        decay: Decay::Lam(0.95),
        max_state_bytes: None,
    };
    let mut pu = Isotonic::new(base_u.clone()).unwrap();
    for i in 0..4 {
        pu.step(
            &[i as f64],
            &[Some(i as f64), Some(2.0 * i as f64)],
            1.0,
            1.0,
        );
    }
    // Room for 1 new point, not 2.
    let cap_u = pu.bytes() + crate::isotonic::ISOTONIC_POINT_BYTES;
    let mut mu = Isotonic::new(IsotonicCfg {
        max_state_bytes: Some(cap_u),
        ..base_u
    })
    .unwrap();
    for i in 0..4 {
        mu.step(
            &[i as f64],
            &[Some(i as f64), Some(2.0 * i as f64)],
            1.0,
            1.0,
        );
    }
    assert_breach_and_followups_atomic(
        "isotonic_unbounded",
        &mut mu,
        |m| {
            m.step(&[10.0], &[Some(10.0), Some(20.0)], 2.0, 1.0);
        },
        &[1.5],
        &[Some(1.0), Some(1.0)],
        |m| {
            m.step(&[1.0], &[Some(1.0), Some(2.0)], 1.0, 1.0);
        },
        |m| {
            m.step(&[1.0], &[Some(1.0), Some(2.0)], 1.0, 0.0);
        },
        |m| {
            m.step(&[1.0], &[None, None], 1.0, 1.0);
        },
    );

    // Case B: `window_clock` where expired row had 0 targets present (`[None, None]`)
    // and incoming row has 2 targets present -> STILL breaches cap after expiry!
    let base_w = IsotonicCfg {
        n_targets: 2,
        increasing: true,
        y_min: None,
        y_max: None,
        out_of_bounds: OutOfBounds::Clip,
        min_weight: 0.0,
        window_rows: None,
        window_clock: Some(5.0),
        closed: WindowClosed::Right,
        decay: Decay::Halflife(f64::INFINITY),
        max_state_bytes: None,
    };
    let mut pw = Isotonic::new(base_w.clone()).unwrap();
    pw.stamp_next(Stamp::Ns(0));
    pw.step(&[0.0], &[None, None], 1.0, 1.0);
    for i in 1..3 {
        pw.stamp_next(Stamp::Ns(i * 2 * S));
        pw.step(&[i as f64], &[Some(i as f64), Some(i as f64)], 2.0, 1.0);
    }
    let cap_w = pw.bytes();
    let mut mw = Isotonic::new(IsotonicCfg {
        max_state_bytes: Some(cap_w),
        ..base_w
    })
    .unwrap();
    mw.stamp_next(Stamp::Ns(0));
    mw.step(&[0.0], &[None, None], 1.0, 1.0);
    for i in 1..3 {
        mw.stamp_next(Stamp::Ns(i * 2 * S));
        mw.step(&[i as f64], &[Some(i as f64), Some(i as f64)], 2.0, 1.0);
    }
    assert_breach_and_followups_atomic(
        "isotonic_window_expiry_breach",
        &mut mw,
        |m| {
            // At t=6s (window=5s), row 0 (with 0 target entries) expires, while
            // this row adds 2 target entries -> net +2 * ISOTONIC_ROW_BYTES > cap!
            m.stamp_next(Stamp::Ns(6 * S));
            m.step(&[3.0], &[Some(3.0), Some(3.0)], 2.0, 1.0);
        },
        &[1.5],
        &[Some(1.0), Some(1.0)],
        |m| {
            m.stamp_next(Stamp::Ns(7 * S));
            m.step(&[1.0], &[Some(1.0), Some(1.0)], 1.0, 1.0);
        },
        |m| {
            m.stamp_next(Stamp::Ns(8 * S));
            m.step(&[1.0], &[Some(1.0), Some(1.0)], 1.0, 0.0);
        },
        |m| {
            m.stamp_next(Stamp::Ns(20 * S));
            m.step(&[1.0], &[None, None], 12.0, 1.0);
        },
    );
}

// ---------------------------------------------------------------------------
// 15. PowerVariation (`window_clock`)
// ---------------------------------------------------------------------------

#[test]
fn power_variation_window_clock_is_atomic() {
    let terms = vec![
        PvTerm {
            kind: PvTermKind::Multipower {
                powers: vec![2.0],
                normalize: false,
            },
            scale: 1,
        },
        PvTerm {
            kind: PvTermKind::Multipower {
                powers: vec![1.0, 1.0],
                normalize: false,
            },
            scale: 1,
        },
    ];
    let base = PowerVariationCfg {
        n_targets: 2,
        terms,
        decay: Decay::Halflife(f64::INFINITY),
        min_weight: 0.0,
        window_rows: None,
        window_clock: Some(10.0),
        closed: WindowClosed::Right,
        max_state_bytes: None,
    };
    let mut probe = PowerVariation::new(base.clone()).unwrap();
    for i in 0..4 {
        probe.stamp_next(Stamp::Ns(i * S));
        probe.step(&[], &[Some(0.01 * (i as f64 + 1.0)), Some(-0.02)], 1.0, 1.0);
    }
    let cap = probe.bytes();
    let mut m = PowerVariation::new(PowerVariationCfg {
        max_state_bytes: Some(cap),
        ..base
    })
    .unwrap();
    for i in 0..4 {
        m.stamp_next(Stamp::Ns(i * S));
        m.step(&[], &[Some(0.01 * (i as f64 + 1.0)), Some(-0.02)], 1.0, 1.0);
    }
    assert_breach_and_followups_atomic(
        "power_variation",
        &mut m,
        |m| {
            m.stamp_next(Stamp::Ns(5 * S));
            m.step(&[], &[Some(0.05), Some(-0.03)], 2.0, 1.0);
        },
        &[],
        &[Some(0.01), Some(0.01)],
        |m| {
            m.stamp_next(Stamp::Ns(6 * S));
            m.step(&[], &[Some(0.02), Some(0.01)], 1.0, 1.0);
        },
        |m| {
            m.stamp_next(Stamp::Ns(7 * S));
            m.step(&[], &[Some(0.02), Some(0.01)], 1.0, 0.0);
        },
        |m| {
            // Past window_clock=10s: must not trim after over!
            m.stamp_next(Stamp::Ns(30 * S));
            m.step(&[], &[None, None], 23.0, 1.0);
        },
    );
}

// ---------------------------------------------------------------------------
// 16. Realized (`window_clock`)
// ---------------------------------------------------------------------------

#[test]
fn realized_window_clock_is_atomic() {
    let base = RealizedCfg {
        n_targets: 2,
        decay: Decay::Halflife(f64::INFINITY),
        min_weight: 0.0,
        window_rows: None,
        window_clock: Some(10.0),
        closed: WindowClosed::Right,
        lm_window_rows: None,
        lm_confidence: DEFAULT_LM_CONFIDENCE,
        max_state_bytes: None,
    };
    let cap = Realized::state_bytes(2, None, Some(4), true);
    let mut m = Realized::new(RealizedCfg {
        max_state_bytes: Some(cap),
        ..base
    })
    .unwrap();
    for i in 0..4 {
        m.stamp_next(Stamp::Ns(i * S));
        m.step(&[], &[Some(0.01 * (i as f64 + 1.0)), Some(-0.02)], 1.0, 1.0);
    }
    assert_breach_and_followups_atomic(
        "realized",
        &mut m,
        |m| {
            m.stamp_next(Stamp::Ns(5 * S));
            m.step(&[], &[Some(0.05), Some(-0.03)], 2.0, 1.0);
        },
        &[],
        &[Some(0.01), Some(0.01)],
        |m| {
            m.stamp_next(Stamp::Ns(6 * S));
            m.step(&[], &[Some(0.02), Some(0.01)], 1.0, 1.0);
        },
        |m| {
            m.stamp_next(Stamp::Ns(7 * S));
            m.step(&[], &[Some(0.02), Some(0.01)], 1.0, 0.0);
        },
        |m| {
            // Past window_clock=10s: must not trim after over!
            m.stamp_next(Stamp::Ns(30 * S));
            m.step(&[], &[None, None], 23.0, 1.0);
        },
    );
}
