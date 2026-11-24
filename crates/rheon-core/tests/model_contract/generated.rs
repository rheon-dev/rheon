// ---------------------------------------------------------------------------
// The contract over generated streams (T-D2 in Rust, docs/PLAN.md task 121).
// ---------------------------------------------------------------------------

// --- states written before a representation grew ---------------------------

/// A state as a msgpack value, edited, and read back: in the named encoding a
/// bank writes, where an infinite parameter stays infinite (serde_json would
/// write it as null and refuse it on the way back).
fn edit_state(state: &State, edit: impl FnOnce(&mut rmpv::Value)) -> State {
    let bytes = rmp_serde::to_vec_named(state).unwrap();
    let mut v = rmpv::decode::read_value(&mut bytes.as_slice()).unwrap();
    edit(&mut v);
    let mut out = Vec::new();
    rmpv::encode::write_value(&mut out, &v).unwrap();
    rmp_serde::from_slice(&out).unwrap()
}

/// Every map entry whose key ends in `_lo`, removed, at any depth: the
/// means' low parts (docs/PLAN.md task 101), which a state written before
/// them does not carry. The count removed.
fn strip_low_parts(v: &mut rmpv::Value) -> usize {
    match v {
        rmpv::Value::Map(entries) => {
            let before = entries.len();
            entries.retain(|(k, _)| !k.as_str().is_some_and(|k| k.ends_with("_lo")));
            let mut n = before - entries.len();
            for (_, x) in entries.iter_mut() {
                n += strip_low_parts(x);
            }
            n
        }
        rmpv::Value::Array(a) => a.iter_mut().map(strip_low_parts).sum(),
        _ => 0,
    }
}

/// A stream at a level, so the means carry something below their doubles,
/// with a target now and then absent and weights that move.
fn leveled(targets: usize, binary: bool) -> Vec<Row> {
    let mut s = 20260927u64;
    (0..160)
        .map(|i| {
            let x: Vec<f64> = (0..K)
                .map(|j| 1e3 * (j as f64 + 1.0) + lcg(&mut s))
                .collect();
            let y = (0..targets)
                .map(|j| {
                    let v = 3.0 + x[0] - 0.5 * x[1] + 0.3 * lcg(&mut s);
                    (i % 9 != 4 + j).then_some(if binary { f64::from(v > 3.0) } else { v })
                })
                .collect();
            Row {
                x,
                y,
                w: 0.5 + (lcg(&mut s) + 1.0) / 2.0,
                extreme: false,
            }
        })
        .collect()
}

/// A model's state at row 80 with every low part removed -- a state written
/// before them -- read back: the number of low parts the state had, and
/// whether the model refused it, by the decoding or by its restore.
fn refuses_without_low_parts<M: OnlineModel>(make: impl Fn() -> M, rows: &[Row]) -> (usize, bool) {
    let mut m = make();
    for (i, r) in rows.iter().enumerate().take(80) {
        m.step(&r.x, &r.y, if i == 0 { 0.0 } else { 1.0 }, r.w);
    }
    let state = m.state();
    let bytes = rmp_serde::to_vec_named(&state).unwrap();
    let mut v = rmpv::decode::read_value(&mut bytes.as_slice()).unwrap();
    let stripped = strip_low_parts(&mut v);
    let mut out = Vec::new();
    rmpv::encode::write_value(&mut out, &v).unwrap();
    let refused = match rmp_serde::from_slice::<State>(&out) {
        Err(_) => true,
        Ok(old) => M::restore(&old).is_err(),
    };
    (stripped, refused)
}

/// Every model refuses a state without the means' low parts, where it
/// loaded one and went on with each part at zero (docs/PLAN.md task 110):
/// such a state is of a layout before this build's schema, which the
/// version refuses (docs/PLAN.md task 198), and what the repair reached
/// was a damaged file. `tests/state_repairs.rs` damages each low part
/// alone, a vector at a time.
#[test]
fn every_model_refuses_a_state_without_the_low_parts() {
    let two = leveled(2, false);
    let one = leveled(1, false);
    let label = leveled(1, true);
    let label2 = leveled(2, true);
    let none = leveled(0, false);
    let results: Vec<(&str, (usize, bool))> = vec![
        (
            "ewridge",
            refuses_without_low_parts(|| EwRidge::new(ewridge_cfg()).unwrap(), &two),
        ),
        (
            "rls",
            refuses_without_low_parts(|| Rls::new(rls_cfg()).unwrap(), &two),
        ),
        (
            "lasso",
            refuses_without_low_parts(|| Lasso::new(lasso_cfg()).unwrap(), &one),
        ),
        (
            "kalman",
            refuses_without_low_parts(|| Kalman::new(kalman_cfg()).unwrap(), &two),
        ),
        (
            "kalman_revert",
            refuses_without_low_parts(|| Kalman::new(kalman_revert_cfg()).unwrap(), &two),
        ),
        (
            "huber",
            refuses_without_low_parts(|| Robust::new(robust_cfg(ROBUST_LOSSES[0])).unwrap(), &two),
        ),
        (
            "quantile",
            refuses_without_low_parts(|| Robust::new(robust_cfg(ROBUST_LOSSES[1])).unwrap(), &two),
        ),
        (
            "ftrl",
            refuses_without_low_parts(|| Ftrl::new(ftrl_cfg()).unwrap(), &label2),
        ),
        (
            "sgd",
            refuses_without_low_parts(|| Sgd::new(sgd_cfg()).unwrap(), &two),
        ),
        (
            "pa",
            refuses_without_low_parts(|| Pa::new(pa_cfg()).unwrap(), &two),
        ),
        (
            "holt",
            refuses_without_low_parts(|| Holt::new(holt_cfg()).unwrap(), &two),
        ),
        (
            "ew_cov",
            refuses_without_low_parts(|| EwCovModel::new(ew_cov_model_cfg()).unwrap(), &none),
        ),
        (
            "kmeans",
            refuses_without_low_parts(|| KMeans::new(kmeans_cfg()).unwrap(), &none),
        ),
        (
            "micro",
            refuses_without_low_parts(|| Micro::new(micro_cfg()).unwrap(), &none),
        ),
        (
            "ew_class",
            refuses_without_low_parts(|| EwClass::new(ew_class_cfg()).unwrap(), &label),
        ),
        (
            "seqtest",
            refuses_without_low_parts(|| SeqTest::new(seqtest_cfg()).unwrap(), &two),
        ),
        (
            "marginal",
            refuses_without_low_parts(|| Marginal::new(marginal_cfg()).unwrap(), &two),
        ),
        (
            "deco",
            refuses_without_low_parts(|| Deco::new(deco_cfg()).unwrap(), &none),
        ),
        (
            "rcov",
            refuses_without_low_parts(|| Rcov::new(rcov_cfg()).unwrap(), &none),
        ),
        (
            "hmm",
            refuses_without_low_parts(|| Hmm::new(hmm_cfg()).unwrap(), &none),
        ),
        (
            "corrchange",
            refuses_without_low_parts(|| CorrChange::new(corrchange_cfg()).unwrap(), &none),
        ),
        (
            "bocpd",
            refuses_without_low_parts(|| Bocpd::new(bocpd_cfg()).unwrap(), &none),
        ),
        (
            "audit",
            refuses_without_low_parts(|| Audit::new(audit_cfg()).unwrap(), &none),
        ),
        (
            "glm",
            refuses_without_low_parts(|| Glm::new(glm_cfg()).unwrap(), &two),
        ),
        (
            "hawkes",
            refuses_without_low_parts(|| Hawkes::new(hawkes_cfg()).unwrap(), &two),
        ),
        (
            "expectile",
            refuses_without_low_parts(|| Expectile::new(expectile_cfg()).unwrap(), &two),
        ),
        (
            "pca",
            refuses_without_low_parts(|| StreamPca::new(pca_cfg()).unwrap(), &none),
        ),
    ];
    // Each model that keeps a mean had parts to drop, so its refusal is of
    // a real state: a model that lost its means, or a renamed field, shows
    // here. The rest keep no low part, and their state is whole.
    let keeps_means = [
        "ewridge",
        "lasso",
        "kalman",
        "kalman_revert",
        "huber",
        "quantile",
        "pa",
        "ew_cov",
        "kmeans",
        "micro",
        "ew_class",
        "marginal",
        "deco",
        "hmm",
        "corrchange",
        "bocpd",
        "audit",
        "pca",
    ];
    for (name, (stripped, refused)) in &results {
        if keeps_means.contains(name) {
            assert!(*stripped > 0, "{name}: no low part in its state");
            assert!(*refused, "{name}: a state without its low parts loaded");
        } else {
            assert_eq!(*stripped, 0, "{name}: list it among the models with means");
        }
    }
}

/// A windowed `ewridge` or `lasso` saved at schema 16, when each target's
/// own feature mean was kept as its offset `d` from the all-row mean, is
/// refused: by its version, and by its layout, which names no `mj`. It
/// loaded at 17 with the offsets made means once (review 2026-09-27, G3),
/// a loader the floor at the schema shipped retired (docs/PLAN.md task
/// 198).
#[test]
fn a_schema_16_state_of_offsets_is_refused() {
    fn to_offsets(v: &mut rmpv::Value, n: &mut usize) {
        match v {
            rmpv::Value::Map(entries) => {
                entries.retain(|(k, _)| k.as_str() != Some("mj_lo"));
                for (k, x) in entries.iter_mut() {
                    if k.as_str() == Some("mj") {
                        *k = rmpv::Value::from("d");
                        *n += 1;
                    } else {
                        to_offsets(x, n);
                    }
                }
            }
            rmpv::Value::Array(a) => a.iter_mut().for_each(|x| to_offsets(x, n)),
            _ => {}
        }
    }
    fn check<M: OnlineModel>(name: &str, make: impl Fn() -> M, rows: &[Row]) {
        let mut m = make();
        for (i, r) in rows.iter().enumerate().take(80) {
            m.step(&r.x, &r.y, if i == 0 { 0.0 } else { 1.0 }, r.w);
        }
        let state = m.state();
        let mut old = state.clone();
        old.schema_version = 16;
        assert!(
            matches!(
                M::restore(&old),
                Err(StateError::SchemaVersion { found: 16, .. })
            ),
            "{name}: refused by its version"
        );
        let bytes = rmp_serde::to_vec_named(&state).unwrap();
        let mut v = rmpv::decode::read_value(&mut bytes.as_slice()).unwrap();
        let mut n = 0;
        to_offsets(&mut v, &mut n);
        assert!(n >= 2, "{name}: {n} offset vectors written");
        let mut out = Vec::new();
        rmpv::encode::write_value(&mut out, &v).unwrap();
        assert!(
            rmp_serde::from_slice::<State>(&out).is_err(),
            "{name}: refused by its layout"
        );
    }
    let rows = leveled(2, false);
    for gaps in [TargetGaps::OwnRows, TargetGaps::Pairwise] {
        let ridge = || {
            let mut c = ewridge_cfg();
            c.window = Some(30.0);
            c.max_rows_between_snapshots = Some(4);
            c.target_gaps = gaps;
            EwRidge::new(c).unwrap()
        };
        check(&format!("ewridge {gaps:?}"), ridge, &rows);
    }
    let one = leveled(1, false);
    let lasso = || {
        let mut c = lasso_cfg();
        c.window = Some(30.0);
        c.max_rows_between_snapshots = Some(4);
        c.target_gaps = TargetGaps::Pairwise;
        Lasso::new(c).unwrap()
    };
    check("lasso", lasso, &one);
}

/// `tests/test_properties.py` asserts invariants on the bank over streams
/// Hypothesis generates. This asserts three clauses of the contract above on
/// every model directly, over streams proptest generates and, on a failure,
/// shrinks to a short one: `predict_with` is the step without the update; a
/// state saved and restored at any row continues exactly as the model that
/// was not; and nothing a model reports is infinite, whatever the inputs
/// inside the bound. Each is checked above on one fixed stream; here the
/// stream is the variable. 128 streams a model in the suite, half proptest's
/// own count: 16 under the essentials gate's `PROPTEST_CASES=32`
/// (docs/TESTING.md, "Two tiers"). 2,000 a model, run once when this was
/// written (docs/PLAN.md task 121), found nothing.
/// The streams are drawn afresh each run (proptest seeds from the OS): a
/// failure prints the shrunk stream, which is what to keep as a fixed case,
/// and `PROPTEST_RNG_SEED=<n>` repeats a run (review 2026-09-26, C8).
mod generated {
    use super::*;
    use proptest::prelude::*;

    /// One row: the features, each target or its absence, the clock since
    /// the previous row, and the weight.
    #[derive(Debug, Clone)]
    struct GenRow {
        x: Vec<f64>,
        y: Vec<Option<f64>>,
        d: f64,
        w: f64,
    }

    /// The side of zero a stream's values sit on (docs/PLAN.md task 209
    /// (a)). A stream of mixed signs was all this drew: it could draw a
    /// one-sided stream and never did. A one-sided stream reads each value
    /// `v` as `sign · |v| + offset`, all positive at a level of `1e3` or all
    /// negative at `-1e3`, and keeps the bound at the bound: `1e100 + 1e3`
    /// is `1e100`. It catches a NaN, an overflow or a divergence at a level,
    /// not a wrong fit, which `held_values.rs` holds.
    #[derive(Debug, Clone, Copy)]
    enum Sign {
        Mixed,
        Positive,
        Negative,
    }

    /// The level a one-sided stream sits at.
    const LEVEL: f64 = 1e3;

    impl Sign {
        fn apply(self, v: f64) -> f64 {
            match self {
                Sign::Mixed => v,
                Sign::Positive => v.abs() + LEVEL,
                Sign::Negative => -v.abs() - LEVEL,
            }
        }
    }

    /// A value a stream may carry: mostly ordinary, often a repeat that holds
    /// a column still, sometimes wide, sometimes at the input bound itself;
    /// on `sign`'s side of zero.
    fn value(sign: Sign) -> impl Strategy<Value = f64> {
        prop_oneof![
            6 => -3.0..3.0f64,
            2 => Just(0.5),
            1 => -1e6..1e6f64,
            1 => (-1.0..1.0f64).prop_map(|u| u * 1e50),
            1 => prop_oneof![
                Just(INPUT_BOUND),
                Just(-INPUT_BOUND),
                (-1.0..1.0f64).prop_map(|u| u * INPUT_BOUND)
            ],
        ]
        .prop_map(move |v| sign.apply(v))
    }

    /// A row of `k` features and `targets` targets; `binary` targets are 0 or 1, the label a
    /// classifier and a logistic loss read. The features sit on `sign`'s
    /// side of zero, and the targets too unless they are labels, which read
    /// the side of zero a value drawn of mixed signs fell on.
    fn row_of(k: usize, targets: usize, binary: bool, sign: Sign) -> impl Strategy<Value = GenRow> {
        let target_sign = if binary { Sign::Mixed } else { sign };
        (
            prop::collection::vec(value(sign), k),
            prop::collection::vec(prop::option::weighted(0.85, value(target_sign)), targets),
            prop_oneof![5 => Just(1.0), 1 => Just(0.0), 1 => 0.0..30.0f64],
            // `-0.0` is a weight of 0 too (hard rule 9), and is drawn by
            // name (docs/PLAN.md task 209 (e)).
            prop_oneof![
                5 => Just(1.0),
                1 => prop_oneof![Just(0.0), Just(-0.0)],
                2 => 0.01..4.0f64,
                1 => Just(1e-100),
                1 => Just(1e100)
            ],
        )
            .prop_map(move |(x, y, d, w)| GenRow {
                x,
                y: if binary {
                    y.into_iter()
                        .map(|v| v.map(|v| f64::from(v > 0.0)))
                        .collect()
                } else {
                    y
                },
                d,
                w,
            })
    }

    /// A stream of 1 to 59 rows, its sign drawn first: a third of the cases
    /// of mixed signs, a third all positive, a third all negative, so each
    /// model's 128 cases run about 43 of each (all 128 missing one has odds
    /// of `3 · (2/3)^128`, about `1e-22`) at the cost of the 128 they ran.
    fn stream(targets: usize, binary: bool) -> impl Strategy<Value = Vec<GenRow>> {
        stream_of(K, targets, binary)
    }

    /// [`stream`] of `k` features: the width a model's wide paths start
    /// from, which the contract's two features never reach.
    fn stream_of(k: usize, targets: usize, binary: bool) -> impl Strategy<Value = Vec<GenRow>> {
        prop_oneof![
            Just(Sign::Mixed),
            Just(Sign::Positive),
            Just(Sign::Negative)
        ]
        .prop_flat_map(move |sign| prop::collection::vec(row_of(k, targets, binary, sign), 1..60))
    }

    fn equal(a: &Step, b: &Step) -> bool {
        let same = |u: &f64, v: &f64| u.to_bits() == v.to_bits() || (u.is_nan() && v.is_nan());
        a.pred.len() == b.pred.len()
            && a.pred.iter().zip(&b.pred).all(|(u, v)| same(u, v))
            && same(&a.n_eff, &b.n_eff)
            && a.extra == b.extra
    }

    /// The three clauses over one stream, with the save and restore at row
    /// `split` (past the end: never).
    fn contract<M: OnlineModel>(
        build: impl Fn() -> M,
        rows: &[GenRow],
        split: usize,
    ) -> Result<(), TestCaseError> {
        let (mut whole, mut parted) = (build(), build());
        for (i, r) in rows.iter().enumerate() {
            if i == split {
                let bytes = rmp_serde::to_vec(&parted.state()).unwrap();
                parted = M::restore(&rmp_serde::from_slice(&bytes).unwrap()).unwrap();
            }
            let d = if i == 0 { 0.0 } else { r.d };
            let p = whole.predict_with(&r.x, &r.y, d);
            let a = whole.step(&r.x, &r.y, d, r.w);
            let b = parted.step(&r.x, &r.y, d, r.w);
            prop_assert!(equal(&p, &a), "row {}: predict {:?}, step {:?}", i, p, a);
            prop_assert!(
                equal(&a, &b),
                "row {}, restored at {}: {:?} against {:?}",
                i,
                split,
                b,
                a
            );
            prop_assert!(
                a.pred
                    .iter()
                    .chain(std::iter::once(&a.n_eff))
                    .all(|v| !v.is_infinite()),
                "row {}: {:?}",
                i,
                a
            );
        }
        Ok(())
    }

    /// The stream the quantile `robust` failed the contract on once the
    /// values reached the bound (review 2026-09-26, G2): a target of `1e100`,
    /// then features of `1e100` at weights from `1e-100`, and the prediction
    /// at `1e100` read `-inf` (`robust.rs` has the mechanism and the fix).
    #[test]
    fn robust_quantile_through_the_bound() {
        let row = |x0: f64, y0: Option<f64>, w: f64| GenRow {
            x: vec![x0, 0.0],
            y: vec![y0, None],
            d: 1.0,
            w,
        };
        let rows = vec![
            row(0.0, None, 1.0),
            row(0.0, Some(0.0), 1.0),
            row(0.0, Some(0.0), 1.0),
            row(0.0, Some(0.0), 1.0),
            row(0.0, Some(0.0), 1.1785268524789025),
            row(0.0, Some(0.0), 1.0),
            row(0.0, Some(1e100), 2.0699847121199655),
            row(0.0, Some(0.0), 1.4511913482607992),
            row(0.0, Some(0.0), 1.0),
            row(1.4589775263789706, Some(0.0), 1.0),
            row(1e100, Some(0.0), 1e-100),
            row(0.0, Some(0.0), 3.8874731502776667),
            row(1e100, Some(0.0), 1.0),
            row(1e100, None, 1.0),
        ];
        contract(
            || Robust::new(robust_cfg(ROBUST_LOSSES[1])).unwrap(),
            &rows,
            0,
        )
        .unwrap();
    }

    /// The stream the windowed `lasso` failed the contract on once the
    /// values reached the bound (review 2026-09-26, G3): a target absent on
    /// a row whose feature stood at `-2.6e99`, then rows at `1e100`, and the
    /// prediction read `-inf` -- windowed or not, and through `ewridge`,
    /// which shares the cross accumulator (`gaps.rs` has the mechanism).
    #[test]
    fn a_target_absent_on_a_row_at_the_bound_through_the_bound() {
        let row = |x: [f64; 2], y0: Option<f64>, w: f64| GenRow {
            x: x.to_vec(),
            y: vec![y0],
            d: 1.0,
            w,
        };
        let rows = vec![
            row([0.0, -2.570413849510098e99], None, 1.0),
            row([0.0, 0.0], None, 1.0),
            row([0.0, 0.0], Some(0.0), 1.0),
            row(
                [0.0, -0.39761046383362925],
                Some(-5.567947375697575e49),
                1.0,
            ),
            row([1.4616143340058023, 1e100], Some(0.0), 0.01),
            row([1e100, 0.0], None, 1.0),
        ];
        for window in [None, Some(7.0)] {
            contract(
                || {
                    let mut c = lasso_cfg();
                    c.window = window;
                    Lasso::new(c).unwrap()
                },
                &rows,
                0,
            )
            .unwrap();
        }
        let two: Vec<GenRow> = rows
            .iter()
            .map(|r| GenRow {
                y: vec![r.y[0], r.y[0]],
                ..r.clone()
            })
            .collect();
        contract(|| EwRidge::new(ewridge_cfg()).unwrap(), &two, 0).unwrap();
    }

    /// The stream the windowed `lasso` failed the contract on (review
    /// 2026-09-27, G5): rows at `1e100` before the window, one of weight
    /// `1e100` inside it, and the prediction at `x1 = 1e100` read `-inf`
    /// (`crate::truncated` has the mechanism). Through `ewridge` too.
    #[test]
    fn a_window_the_live_state_cannot_resolve_through_the_bound() {
        let row = |x: [f64; 2], y0: Option<f64>, d: f64, w: f64| GenRow {
            x: x.to_vec(),
            y: vec![y0],
            d,
            w,
        };
        let rows = vec![
            row([0.0, 0.0], Some(0.0), 1.0, 1.0),
            row([0.0, 1e100], Some(1e100), 0.0, 1.0),
            row([0.0, 0.0], Some(0.0), 1.0, 0.714463635904142),
            row([0.0, 0.0], Some(0.0), 1.0, 1.0),
            row([0.0, 0.0], None, 1.0, 1.0),
            row([0.0, 0.0], None, 1.0, 1.0),
            row([1e100, -7.726806526869417e99], Some(0.0), 1.0, 1.0),
            row([0.0, 0.0], Some(0.0), 12.875843092015453, 0.01),
            row([0.0, 0.0], Some(0.0), 1.0, 1.0),
            row([0.0, 0.0], Some(0.0), 1.0, 1.0),
            row(
                [-2.9807652788918353, 1.021121247753338],
                Some(0.0),
                1.0,
                1e100,
            ),
            row([0.0, 0.0], None, 1.0, 1.0),
            row([0.0, 0.0], None, 0.0, 1.0),
            row([0.0, 0.0], None, 1.0, 1.0),
            row([0.0, 1e100], None, 1.0, 1.0),
        ];
        let build = || {
            let mut c = lasso_cfg();
            c.window = Some(7.0);
            Lasso::new(c).unwrap()
        };
        contract(build, &rows, 0).unwrap();
        let two: Vec<GenRow> = rows
            .iter()
            .map(|r| GenRow {
                y: vec![r.y[0], r.y[0]],
                ..r.clone()
            })
            .collect();
        contract(
            || {
                let mut c = ewridge_cfg();
                c.window = Some(7.0);
                EwRidge::new(c).unwrap()
            },
            &two,
            0,
        )
        .unwrap();
    }

    /// Task 209's stream at a level, its values `v` read as `level + v -
    /// 1000`: a row of weight `1e100` (row 1) leaves a window of 7 at row 9
    /// while another (row 6) stays, and the last row's feature is at the
    /// bound. Its first row was the one-sided stream the windowed `lasso`
    /// failed the contract on (docs/PLAN.md task 209's report; task 216, D4).
    fn dominant_row_leaving(level: f64) -> Vec<GenRow> {
        let s = level - 1000.0;
        let row = |x0: f64, x1: f64, y0: Option<f64>, d: f64, w: f64| GenRow {
            x: vec![x0 + s, x1 + s],
            y: vec![y0.map(|v| v + s)],
            d,
            w,
        };
        vec![
            row(1000.0, 526586.3282990033, Some(1000.0), 1.0, 0.01),
            row(
                1000.1221927147707,
                1000.4436148121216,
                Some(1000.0),
                1.0,
                1e100,
            ),
            row(1000.0, 1000.0, None, 1.0, 1.0),
            row(1000.0, 1000.0, None, 0.0, 1.0),
            row(1000.0, 1000.0, None, 1.0, 1.0),
            row(1000.0, 1000.0, None, 1.0, 1.0),
            row(
                1000.0218939707881,
                1000.8139282141994,
                Some(1.9261315645396117e49),
                1.0,
                1e100,
            ),
            row(1000.0, 1000.0, None, 1.0, 1.0),
            row(1000.0, 1000.0, Some(1000.0), 1.0, 0.01),
            row(1000.0, 1000.0, None, 1.0, 1.0),
            row(1000.0, 1000.0, None, 1.0, 1.0),
            row(1000.0, 1e100, None, 1.0, 1.0),
        ]
    }

    /// **A windowed fit at a level reads what it reads at level 0 once a
    /// dominant row has left the window** (docs/PLAN.md task 217, D4). The
    /// row of weight `1e100` that leaves takes the window's spread to about
    /// `1e-106`, and the subtraction left a remainder of the means' rounding
    /// at the level: 1e-14 at 1,000 and 1.3e-9 at 1e8, which passed for a
    /// spread (`crate::truncated` has the bound and the numbers). On the
    /// base, at ±1,000 the windowed `lasso` grew by about `1e45` a row and
    /// predicted `inf` on the last, and `ewridge` held slopes of -6.6e41
    /// where the kept rows give 4.5e-49; at ±1e8 both were finite with
    /// slopes of 2.6e49 and 3.4e46. Now the coefficients after the row has
    /// left (rows 9 to 11) are level 0's to the bit, under both snapshot
    /// cadences; before it (row 8), with a snapshot every row, each is the
    /// fit of the rows inside the window to 1e-9, read from a model fed
    /// those rows alone; and every level keeps the contract.
    #[test]
    fn a_dominant_row_leaving_a_window_at_a_level_reads_as_level_0() {
        type Coefs = Vec<f64>;
        let lasso = |every: Option<usize>, window: Option<f64>| {
            let mut c = lasso_cfg();
            c.window = window;
            c.max_rows_between_snapshots = every;
            Lasso::new(c).unwrap()
        };
        let ridge = |every: Option<usize>, window: Option<f64>| {
            let mut c = ewridge_cfg();
            c.window = window;
            c.max_rows_between_snapshots = every;
            EwRidge::new(c).unwrap()
        };
        let two = |rows: &[GenRow]| -> Vec<GenRow> {
            rows.iter()
                .map(|r| GenRow {
                    y: vec![r.y[0], r.y[0]],
                    ..r.clone()
                })
                .collect()
        };
        let lasso_coefs = |m: &Lasso| -> Coefs { m.coefficients().unwrap().concat().concat() };
        let ridge_coefs = |m: &EwRidge| -> Coefs { m.coefficients().unwrap().concat() };
        // Each model's coefficients after rows 8 to 11, and the fit of the
        // rows inside the window after row 8 (rows 1 to 8: row 0 is 8 clock
        // units old there, past the window of 7).
        let fits = |level: f64, every: Option<usize>| -> [(Vec<Coefs>, Coefs); 2] {
            let rows = dominant_row_leaving(level);
            let mut l = lasso(every, Some(7.0));
            let mut e = ridge(every, Some(7.0));
            let (mut lc, mut ec) = (Vec::new(), Vec::new());
            for (i, (r, r2)) in rows.iter().zip(two(&rows)).enumerate() {
                let d = if i == 0 { 0.0 } else { r.d };
                l.step(&r.x, &r.y, d, r.w);
                e.step(&r2.x, &r2.y, d, r2.w);
                if i >= 8 {
                    lc.push(lasso_coefs(&l));
                    ec.push(ridge_coefs(&e));
                }
            }
            let (mut ls, mut es) = (lasso(None, None), ridge(None, None));
            for (i, (r, r2)) in rows[1..=8].iter().zip(two(&rows[1..=8])).enumerate() {
                let d = if i == 0 { 0.0 } else { r.d };
                ls.step(&r.x, &r.y, d, r.w);
                es.step(&r2.x, &r2.y, d, r2.w);
            }
            [(lc, lasso_coefs(&ls)), (ec, ridge_coefs(&es))]
        };
        for every in [None, Some(3)] {
            let base = fits(0.0, every);
            for level in [0.0, 1e3, -1e3, 1e8, -1e8] {
                let got = fits(level, every);
                for (name, (at, inside), (at0, _)) in
                    [("lasso", &got[0], &base[0]), ("ewridge", &got[1], &base[1])]
                {
                    let case = format!("{name} at level {level}, snapshots {every:?}");
                    // A snapshot every 3 rows draws the boundary later, past
                    // row 1 already: its window is not rows 1 to 8.
                    let kept = if every.is_none() {
                        at[0].as_slice()
                    } else {
                        &[]
                    };
                    // Normwise, over each fit's intercept and slopes
                    // (docs/PLAN.md task 273): the window's system is near
                    // singular (two rows of weight `1e100` and one of 0.01),
                    // so a forward error is a share of the vector's size,
                    // and an entry 1e-14 of it (`lasso`'s second slope, 2e36
                    // beside 2e50) carries no digit. Entrywise it kept its
                    // sign with Apple's `exp2`, and flipped with it moved by
                    // 1 ulp. Measured normwise: `ewridge` 4.6e-11 at every
                    // level and `lasso` 1.4e-13, the same with `exp2` moved
                    // by up to 4 ulps.
                    let fits = kept.chunks(K + 1).zip(inside.chunks(K + 1));
                    for (slot, (us, vs)) in fits.enumerate() {
                        let size = us.iter().chain(vs).fold(1.0f64, |a, x| a.max(x.abs()));
                        for (u, v) in us.iter().zip(vs) {
                            assert!(
                                (u - v).abs() <= 1e-9 * size,
                                "{case}: row 8, fit {slot}, {:?} where the rows inside the window fit {inside:?}",
                                at[0]
                            );
                        }
                    }
                    for (row, (c, c0)) in at.iter().zip(at0).enumerate().skip(1) {
                        assert!(
                            c.iter().zip(c0).all(|(u, v)| u.to_bits() == v.to_bits()),
                            "{case}: row {}, {c:?} where level 0 has {c0:?}",
                            row + 8
                        );
                    }
                }
                let rows = dominant_row_leaving(level);
                contract(|| lasso(every, Some(7.0)), &rows, 0).unwrap();
                contract(|| ridge(every, Some(7.0)), &two(&rows), 0).unwrap();
            }
        }
    }

    /// **A windowed `marginal` pair is the moments of the rows inside the
    /// window, or none** (docs/PLAN.md task 217). Its truncation had no
    /// bound at all (`crate::truncated` has G5's and the level's): on
    /// [`dominant_row_leaving`]'s stream, once the row of weight `1e100` has
    /// left, the window's spreads are about `1e-106` (the feature) and
    /// `4e-4` against a history of `9e97` (the target), and the pair read the
    /// subtraction's rounding as moments. On the base, at level 0, `var_y`
    /// was `6.1e82` where the rows give `4.0e-4`, and `beta` `-1.9e50` where
    /// they give `8.8e50`, `corr` -1 where they give 1; at 1,000, `var_x`
    /// was 6.8e-15 and `beta` -9.6e49; at 1e8 `var_x` 1.3e-9 and `beta`
    /// 2.6e49. Each of `var_x`, `var_y` and `cov`, at every level and row,
    /// is now the fit's of the rows inside the window, read from a pair fed
    /// them alone, to 1e-6, or 0, and every level reads level 0's pair to
    /// the bit once the row has left.
    #[test]
    fn a_windowed_marginal_pair_is_the_rows_inside_or_none() {
        let cfg = |window: Option<f64>| MarginalCfg {
            n_targets: 1,
            min_weight: vec![0.0],
            lags: vec![],
            window,
            ..marginal_cfg()
        };
        let feed = |m: &mut Marginal, rows: &[GenRow]| {
            for (i, r) in rows.iter().enumerate() {
                let d = if i == 0 { 0.0 } else { r.d };
                m.step(&r.x, &r.y, d, r.w);
            }
        };
        let mut at0 = Vec::new();
        for level in [0.0, 1e3, -1e3, 1e8, -1e8] {
            let rows = dominant_row_leaving(level);
            let mut got = Vec::new();
            for (upto, first) in [(8, 1), (9, 2), (10, 4), (11, 5)] {
                let mut win = Marginal::new(cfg(Some(7.0))).unwrap();
                feed(&mut win, &rows[..=upto]);
                let mut inside = Marginal::new(cfg(None)).unwrap();
                feed(&mut inside, &rows[first..=upto]);
                for j in 0..K {
                    let (p, q) = (win.pair(0, j), inside.pair(0, j));
                    for (what, u, v) in [
                        ("var_x", p.var_x, q.var_x),
                        ("var_y", p.var_y, q.var_y),
                        ("cov", p.cov, q.cov),
                    ] {
                        assert!(
                            u == 0.0 || (u - v).abs() <= 1e-6 * v.abs(),
                            "level {level}, row {upto}, x{j}: {what} {u:e} where the rows inside \
                             the window give {v:e}"
                        );
                        if upto > 8 {
                            got.push(u.to_bits());
                        }
                    }
                }
            }
            if level == 0.0 {
                at0 = got;
            } else {
                assert_eq!(got, at0, "level {level} against level 0, rows 9 to 11");
            }
        }
    }

    /// **A windowed `marginal` target with none of its rows inside the window
    /// reads nothing there** (docs/PLAN.md task 217): its weight 0, its pairs
    /// empty, as a target never seen. Absent for 300 rows after 60 at a
    /// weight of `1e-300`, it kept a weight stuck a few subnormal steps above
    /// 0, and the window's, that less the snapshot's, was a remainder of
    /// rounding. `marginal` counts each target's learned rows, live and in
    /// each snapshot, so the window holds none of them exactly.
    #[test]
    fn a_windowed_marginal_target_with_no_rows_inside_reads_nothing() {
        let cfg = MarginalCfg {
            min_weight: vec![0.0; 2],
            lags: vec![],
            decay: Decay::Lam(0.75),
            window: Some(5.0),
            ..marginal_cfg()
        };
        let mut m = Marginal::new(cfg).unwrap();
        let mut s = 5u64;
        for i in 0..360 {
            let x = [lcg(&mut s), lcg(&mut s)];
            let y1 = (i < 60).then(|| x[1] + 0.1 * lcg(&mut s));
            let w = if i < 60 { 1e-300 } else { 1.0 };
            m.step(&x, &[Some(x[0]), y1], if i == 0 { 0.0 } else { 1.0 }, w);
        }
        assert_eq!(m.target_weight(1).to_bits(), 0f64.to_bits(), "the weight");
        for j in 0..K {
            let p = m.pair(1, j);
            assert!(
                p.n_eff == 0.0 && p.var_x.is_nan() && p.var_y.is_nan() && p.cov.is_nan(),
                "x{j}: weight {:e}, var_x {:e}, var_y {:e}, cov {:e} from a window holding none \
                 of the target's rows",
                p.n_eff,
                p.var_x,
                p.var_y,
                p.cov
            );
        }
        // The target with rows inside the window reads them.
        assert!(m.target_weight(0) > 0.0 && m.pair(0, 0).var_x > 0.0);
    }

    /// Two rows of weight `1e100` inside a window of 7, their features apart
    /// and their targets equal, after a third has left: the feature has a
    /// spread in the window, the target none the subtraction can resolve.
    fn equal_targets_inside(level: f64) -> Vec<GenRow> {
        let s = level - 1000.0;
        let row = |x0: f64, x1: f64, y0: Option<f64>, d: f64, w: f64| GenRow {
            x: vec![x0 + s, x1 + s],
            y: vec![y0.map(|v| v + s)],
            d,
            w,
        };
        vec![
            row(1000.3, 999.8, Some(1001.0), 1.0, 1.0),
            row(1000.12, 1000.44, Some(1000.5), 1.0, 1e100),
            row(1000.0, 1000.0, None, 1.0, 1.0),
            row(1000.0, 1000.0, None, 0.0, 1.0),
            row(1000.0, 1000.0, None, 1.0, 1.0),
            row(1000.0, 1000.0, None, 1.0, 1.0),
            row(1000.5, 1000.0, Some(1000.25), 1.0, 1e100),
            row(999.5, 1000.3, Some(1000.25), 1.0, 1e100),
            row(1000.0, 1000.0, Some(1003.0), 1.0, 1.0),
            row(1000.0, 1000.0, None, 1.0, 1.0),
            row(1000.0, 1000.0, None, 1.0, 1.0),
            row(1000.0, 1000.3, None, 1.0, 1.0),
        ]
    }

    /// **A target with no spread in the window has no covariance with a
    /// feature there** (Cauchy-Schwarz; docs/PLAN.md task 217): its
    /// cross-moments are 0 where the window's target variance is, as a
    /// feature's are where its variance in the Gram is. On
    /// [`equal_targets_inside`]'s stream the rows give slopes of about
    /// `1e-95`; on the base the cross-moments' remainder, the target's mean
    /// rounded at the level, set them at -4.8e-14 and -1.6e-13 at level 0,
    /// -1.4e-9 and -4.6e-9 at 1,000, and 2.0e-4 and 6.6e-4 at 1e8, where the
    /// intercept stood 85,704 from the target's mean. Now the slopes after
    /// the row has left (rows 9 to 11) are 0 at every level, for `ewridge`
    /// and `lasso`, and the intercept is the target's mean.
    #[test]
    fn a_target_with_no_spread_in_a_window_has_no_slope_there() {
        let two = |rows: &[GenRow]| -> Vec<GenRow> {
            rows.iter()
                .map(|r| GenRow {
                    y: vec![r.y[0], r.y[0]],
                    ..r.clone()
                })
                .collect()
        };
        for level in [0.0, 1e3, -1e3, 1e8, -1e8] {
            let rows = equal_targets_inside(level);
            let mut l = {
                let mut c = lasso_cfg();
                c.window = Some(7.0);
                c.min_weight = 0.0;
                Lasso::new(c).unwrap()
            };
            let mut e = {
                let mut c = ewridge_cfg();
                c.window = Some(7.0);
                c.min_weight = 0.0;
                EwRidge::new(c).unwrap()
            };
            for (i, (r, r2)) in rows.iter().zip(two(&rows)).enumerate() {
                let d = if i == 0 { 0.0 } else { r.d };
                l.step(&r.x, &r.y, d, r.w);
                e.step(&r2.x, &r2.y, d, r2.w);
                if i < 9 {
                    continue;
                }
                let mean = level + 0.25;
                for (name, c) in [
                    ("lasso", l.coefficients().unwrap().concat()),
                    ("ewridge", e.coefficients().unwrap().to_vec()),
                ] {
                    for c in c {
                        assert!(
                            c[1] == 0.0 && c[2] == 0.0,
                            "{name} at level {level}, row {i}: slopes {:?}",
                            &c[1..]
                        );
                        assert!(
                            (c[0] - mean).abs() <= 64.0 * f64::EPSILON * mean.abs().max(1.0),
                            "{name} at level {level}, row {i}: intercept {} against the \
                             target's mean {mean}",
                            c[0]
                        );
                    }
                }
            }
        }
    }

    /// The case that failed the scheduled mutation pass of 2026-10-04 at its
    /// baseline (run 37194203887, shard 13), as proptest shrank it: on row
    /// 7 a feature at the input bound, standardized against a scale the
    /// earlier rows set, times its coefficient overflowed, and `kalman`
    /// predicted `inf` for the second target. Kept as a named test, since
    /// proptest cannot find this file's source to save its own regression
    /// file (docs/PLAN.md task 158).
    #[test]
    fn kalman_at_the_input_bound_predicts_a_number_or_nothing() {
        let row = |x: [f64; 2], y: Option<f64>, d: f64, w: f64| GenRow {
            x: x.to_vec(),
            y: vec![None, y],
            d,
            w,
        };
        let rows = [
            row([-1.197488167584902, 0.0], Some(0.0), 1.0, 2.785109721135609),
            row([0.0, 0.0], None, 4.697822926878844, 0.0),
            row(
                [-2.982070606445263, 4.622005473946053e49],
                Some(0.0),
                1.0,
                1.0,
            ),
            row([-0.6358232128145591, 0.0], Some(INPUT_BOUND), 1.0, 1e100),
            row([0.0, 0.0], None, 1.0, 0.0),
            row([0.0, 0.2675076319235289], Some(0.0), 1.0, 1.0),
            row([-INPUT_BOUND, 0.5], Some(0.0), 0.0, 1.0),
            row([0.0, INPUT_BOUND], None, 1.0, 1.0),
        ];
        contract(|| Kalman::new(kalman_cfg()).unwrap(), &rows, 0).unwrap();
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(ProptestConfig::default().cases / 2))]

        #[test]
        fn ewridge(rows in stream(2, false), split in 0usize..60) {
            contract(|| EwRidge::new(ewridge_cfg()).unwrap(), &rows, split)?;
        }

        /// The window's ring, its snapshots and their serde, under both
        /// cadences; and the blocked Gram with pairwise gaps, whose held
        /// rows the state carries (review 2026-09-26, C8).
        #[test]
        fn ewridge_windowed(rows in stream(2, false), split in 0usize..60, every in 0usize..2) {
            contract(|| {
                let mut c = ewridge_cfg();
                c.window = Some(7.0);
                c.max_rows_between_snapshots = [None, Some(3)][every];
                EwRidge::new(c).unwrap()
            }, &rows, split)?;
        }

        #[test]
        fn ewridge_blocked_pairwise(rows in stream(2, false), split in 0usize..60) {
            contract(|| {
                let mut c = ewridge_cfg();
                // A block needs a solve cadence, or it never holds a second row.
                c.gram_block_rows = 4;
                c.solve_every = 3.0;
                c.max_rows_between_solves = 8;
                c.target_gaps = rheon_core::TargetGaps::Pairwise;
                EwRidge::new(c).unwrap()
            }, &rows, split)?;
        }

        #[test]
        fn rls(rows in stream(2, false), split in 0usize..60) {
            contract(|| Rls::new(rls_cfg()).unwrap(), &rows, split)?;
        }

        #[test]
        fn lasso(rows in stream(1, false), split in 0usize..60) {
            contract(|| Lasso::new(lasso_cfg()).unwrap(), &rows, split)?;
        }

        /// At 13, 20, 39 and 40 features, windowed and not (review
        /// fable2, C8): coefficients finite but large after rows at the
        /// input bound made `β·x` overflow on a row inside it, and the
        /// prediction read `±inf` from 13 features on, where two never
        /// reach. 39 and 40 straddle the split sweep's width.
        #[test]
        fn lasso_wide(
            (k, rows) in prop_oneof![Just(13usize), Just(20), Just(39), Just(40)]
                .prop_flat_map(|k| (Just(k), stream_of(k, 1, false))),
            split in 0usize..60,
            window in 0usize..2,
        ) {
            contract(|| {
                let mut c = lasso_cfg();
                c.n_features = k;
                c.window = [None, Some(7.0)][window];
                Lasso::new(c).unwrap()
            }, &rows, split)?;
        }

        /// One-sided streams too, since docs/PLAN.md task 217 (D4). Drawn
        /// one-sided, this failed one run in 18 before it: once a row of
        /// weight `1e100` left the window of 7 at a level of ±1,000, the
        /// windowed lasso's coefficients grew by about `1e45` a row and a
        /// feature at the bound predicted `inf`
        /// ([`a_dominant_row_leaving_a_window_at_a_level_reads_as_level_0`]).
        /// 12,000 cases drawn as here passed with the fix.
        #[test]
        fn lasso_windowed(rows in stream(1, false), split in 0usize..60, every in 0usize..2) {
            contract(|| {
                let mut c = lasso_cfg();
                c.window = Some(7.0);
                c.max_rows_between_snapshots = [None, Some(3)][every];
                Lasso::new(c).unwrap()
            }, &rows, split)?;
        }

        #[test]
        fn kalman(rows in stream(2, false), split in 0usize..60) {
            contract(|| Kalman::new(kalman_cfg()).unwrap(), &rows, split)?;
        }

        #[test]
        fn kalman_reverting(rows in stream(2, false), split in 0usize..60) {
            contract(|| Kalman::new(kalman_revert_cfg()).unwrap(), &rows, split)?;
        }

        /// The form a state saved before task 278 goes on in, the noise
        /// estimate in the gain, reverting.
        #[test]
        fn kalman_reverting_noise_in_the_gain(rows in stream(2, false), split in 0usize..60) {
            contract(|| Kalman::new(KalmanCfg { noise_units: false, ..kalman_revert_cfg() }).unwrap(), &rows, split)?;
        }

        #[test]
        fn robust(rows in stream(2, false), split in 0usize..60, which in 0usize..ROBUST_LOSSES.len()) {
            contract(|| Robust::new(robust_cfg(ROBUST_LOSSES[which])).unwrap(), &rows, split)?;
        }

        #[test]
        fn ftrl(rows in stream(2, false), split in 0usize..60) {
            contract(|| Ftrl::new(ftrl_cfg()).unwrap(), &rows, split)?;
        }

        #[test]
        fn sgd(rows in stream(2, false), split in 0usize..60) {
            contract(|| Sgd::new(sgd_cfg()).unwrap(), &rows, split)?;
        }

        #[test]
        fn pa(rows in stream(2, false), split in 0usize..60) {
            contract(|| Pa::new(pa_cfg()).unwrap(), &rows, split)?;
        }

        /// `pa` and `sgd` standardize by default since docs/PLAN.md task 195
        /// (U2), so the contract holds the scaled step too.
        #[test]
        fn pa_standardized(rows in stream(2, false), split in 0usize..60) {
            contract(|| Pa::new(PaCfg { standardize: true, ..pa_cfg() }).unwrap(), &rows, split)?;
        }

        #[test]
        fn sgd_standardized(rows in stream(2, false), split in 0usize..60) {
            contract(
                || Sgd::new(SgdCfg { standardize: true, ..sgd_cfg() }).unwrap(),
                &rows,
                split,
            )?;
        }

        #[test]
        fn holt(rows in stream(2, false), split in 0usize..60) {
            contract(|| Holt::new(holt_cfg()).unwrap(), &rows, split)?;
        }

        #[test]
        fn ew_cov(rows in stream(0, false), split in 0usize..60) {
            contract(|| EwCovModel::new(ew_cov_model_cfg()).unwrap(), &rows, split)?;
        }

        /// A window keeps no lags (the two rings do not combine), so the
        /// windowed `ew_cov` is the moments alone.
        #[test]
        fn ew_cov_windowed(rows in stream(0, false), split in 0usize..60, every in 0usize..2) {
            contract(|| {
                let mut c = ew_cov_model_cfg();
                c.lags = vec![];
                c.stats.retain(|s| *s != EwCovStat::LagCorr);
                c.window = Some(7.0);
                c.max_rows_between_snapshots = [None, Some(3)][every];
                EwCovModel::new(c).unwrap()
            }, &rows, split)?;
        }

        #[test]
        fn marginal(rows in stream(2, false), split in 0usize..60) {
            contract(|| Marginal::new(marginal_cfg()).unwrap(), &rows, split)?;
        }

        /// A window keeps no lags, so the windowed marginal is the moments
        /// alone.
        #[test]
        fn marginal_windowed(rows in stream(2, false), split in 0usize..60, every in 0usize..2) {
            contract(|| {
                let mut c = marginal_cfg();
                c.lags = vec![];
                c.window = Some(7.0);
                c.max_rows_between_snapshots = [None, Some(3)][every];
                Marginal::new(c).unwrap()
            }, &rows, split)?;
        }

        #[test]
        fn kmeans(rows in stream(0, false), split in 0usize..60) {
            contract(|| KMeans::new(kmeans_cfg()).unwrap(), &rows, split)?;
        }

        #[test]
        fn micro(rows in stream(0, false), split in 0usize..60) {
            contract(|| Micro::new(micro_cfg()).unwrap(), &rows, split)?;
        }

        #[test]
        fn ew_class(rows in stream(1, true), split in 0usize..60) {
            contract(|| EwClass::new(ew_class_cfg()).unwrap(), &rows, split)?;
        }

        #[test]
        fn ew_class_windowed(rows in stream(1, true), split in 0usize..60, every in 0usize..2) {
            contract(|| {
                let mut c = ew_class_cfg();
                c.window = Some(7.0);
                c.max_rows_between_snapshots = [None, Some(3)][every];
                EwClass::new(c).unwrap()
            }, &rows, split)?;
        }

        #[test]
        fn seqtest(rows in stream(2, false), split in 0usize..60) {
            contract(|| SeqTest::new(seqtest_cfg()).unwrap(), &rows, split)?;
        }

        #[test]
        fn bocpd(rows in stream(0, false), split in 0usize..60) {
            contract(|| Bocpd::new(bocpd_cfg()).unwrap(), &rows, split)?;
        }

        #[test]
        fn bocpd_on_the_clock(rows in stream(0, false), split in 0usize..60) {
            contract(|| Bocpd::new(bocpd_on_the_clock_cfg()).unwrap(), &rows, split)?;
        }

        #[test]
        fn bocpd_from_the_data(rows in stream(0, false), split in 0usize..60) {
            contract(|| Bocpd::new(bocpd_from_the_data_cfg()).unwrap(), &rows, split)?;
        }

        #[test]
        fn corrchange(rows in stream(0, false), split in 0usize..60) {
            contract(|| CorrChange::new(corrchange_cfg()).unwrap(), &rows, split)?;
        }

        #[test]
        fn hmm(rows in stream(0, false), split in 0usize..60) {
            contract(|| Hmm::new(hmm_cfg()).unwrap(), &rows, split)?;
        }

        #[test]
        fn rcov(rows in stream(0, false), split in 0usize..60) {
            contract(|| Rcov::new(rcov_cfg()).unwrap(), &rows, split)?;
        }

        #[test]
        fn deco(rows in stream(0, false), split in 0usize..60) {
            contract(|| Deco::new(deco_cfg()).unwrap(), &rows, split)?;
        }

        #[test]
        fn audit(rows in stream(0, false), split in 0usize..60) {
            contract(|| Audit::new(audit_cfg()).unwrap(), &rows, split)?;
        }

        #[test]
        fn glm(rows in stream(2, false), split in 0usize..60) {
            contract(|| Glm::new(glm_cfg()).unwrap(), &rows, split)?;
        }

        #[test]
        fn hawkes(rows in stream(2, false), split in 0usize..60) {
            contract(|| Hawkes::new(hawkes_cfg()).unwrap(), &rows, split)?;
        }

        #[test]
        fn pca(rows in stream(0, false), split in 0usize..60) {
            contract(|| StreamPca::new(pca_cfg()).unwrap(), &rows, split)?;
        }

        #[test]
        fn pca_oja(rows in stream(0, false), split in 0usize..60) {
            contract(|| StreamPca::new(pca_oja_cfg()).unwrap(), &rows, split)?;
        }

        #[test]
        fn expectile(rows in stream(2, false), split in 0usize..60) {
            contract(|| Expectile::new(expectile_cfg()).unwrap(), &rows, split)?;
        }

        #[test]
        fn expectile_window_rows(rows in stream(2, false), split in 0usize..60) {
            contract(|| Expectile::new(expectile_rows_cfg()).unwrap(), &rows, split)?;
        }

        /// The window drops a row a row once full, and forms its older
        /// block every few rows, so a split lands in either block.
        #[test]
        fn hawkes_windowed(rows in stream(2, false), split in 0usize..60) {
            contract(|| Hawkes::new(hawkes_windowed_cfg()).unwrap(), &rows, split)?;
        }

        /// A window of clock units leaves rows at the steps' own, which the
        /// stream's gaps and zero steps put on and off its edge.
        #[test]
        fn hawkes_clock_windowed(rows in stream(2, false), split in 0usize..60) {
            contract(|| Hawkes::new(hawkes_clock_cfg()).unwrap(), &rows, split)?;
        }
    }
}

/// Two return series under the shared decay, with Lee and Mykland's own
/// window of 8 returns.
fn realized_cfg() -> RealizedCfg {
    RealizedCfg {
        n_targets: 2,
        decay: decay(),
        min_weight: 3.0,
        window_rows: None,
        window_clock: None,
        closed: WindowClosed::Right,
        lm_window: Some(8),
        lm_beta: 0.99,
        max_state_bytes: None,
    }
}

/// `pca` on the probes' two columns: one component emitted, both kept, the
/// incremental SVD; its gate is the eigengap test, so `min_weight` 0.
fn pca_cfg() -> StreamPcaCfg {
    StreamPcaCfg {
        n_features: K,
        n_components: 1,
        method: PcaMethod::Svd,
        decay: decay(),
        min_weight: 0.0,
        learning_rate: None,
        max_state_bytes: None,
    }
}

/// A window of six rows, so the probes run through its turnovers; a window
/// takes no decay, and its `n_eff` is the weight inside it.
fn realized_rows_cfg() -> RealizedCfg {
    RealizedCfg {
        decay: Decay::Halflife(f64::INFINITY),
        window_rows: Some(6),
        lm_window: None,
        ..realized_cfg()
    }
}

/// The same over a span of the clock: the probes' rows are a unit apart, so
/// a window of 12.5 units holds thirteen rows.
fn realized_clock_cfg() -> RealizedCfg {
    RealizedCfg {
        decay: Decay::Halflife(f64::INFINITY),
        window_clock: Some(12.5),
        ..realized_cfg()
    }
}

fn pca_oja_cfg() -> StreamPcaCfg {
    StreamPcaCfg {
        method: PcaMethod::Oja,
        ..pca_cfg()
    }
}

fn pca_ccipca_cfg() -> StreamPcaCfg {
    StreamPcaCfg {
        method: PcaMethod::Ccipca,
        ..pca_cfg()
    }
}

#[test]
fn realized() {
    // Realized volatility: one return series per target, no features, seven
    // outputs a series, and `n_eff` every row's weight on the shared
    // recursion.
    let m = Realized::new(realized_cfg()).unwrap();
    assert_eq!(m.n_features(), 0, "realized reads no features");
    let r = probe_with(m, 2, Some(&Realized::n_eff));
    assert_eq!(r.n_features, 0);
    assert_eq!(r.kind, "realized");
    assert_eq!(r.n_targets, 2);
    assert_eq!(r.n_outputs, 14);
    assert_eq!(r.pred_len, 14);
    assert_eq!(r.n_eff[0], 0.0);
    assert_eq!(r.n_eff[1], 1.0);
    assert!((r.n_eff[2] - (0.5f64.powf(1.0 / HALFLIFE) + 1.0)).abs() < 1e-12);
    assert!((r.after_gap - (r.before_gap * 0.5f64.powi(10) + 1.0)).abs() < 1e-9);
    assert!(r.roundtrips);
    // Under a window `n_eff` is the weight inside it (hard rule 8): the six
    // rows' of a window of six, the thirteen of a span of 12.5 units a step
    // apart; a gap does not move a window of rows, and a gap past the clock
    // window's span empties it at the row the gap ends on, after that row
    // reports.
    let r = probe_windowed(
        Realized::new(realized_rows_cfg()).unwrap(),
        2,
        Some(&Realized::n_eff),
    );
    assert_eq!(r.n_eff, vec![0.0, 1.0, 2.0, 6.0]);
    assert_eq!(r.after_gap, r.before_gap, "a gap does not move the window");
    assert!(r.roundtrips);
    let r = probe_windowed(
        Realized::new(realized_clock_cfg()).unwrap(),
        2,
        Some(&Realized::n_eff),
    );
    assert_eq!(r.n_eff, vec![0.0, 1.0, 2.0, 13.0]);
    assert!(r.roundtrips);
}

#[test]
fn realized_predict_is_the_step() {
    predict_is_the_step_without_the_step(|| Realized::new(realized_cfg()).unwrap(), 2, false);
    predict_is_the_step_without_the_step(|| Realized::new(realized_rows_cfg()).unwrap(), 2, false);
    predict_is_the_step_without_the_step(|| Realized::new(realized_clock_cfg()).unwrap(), 2, false);
}

#[test]
fn pca() {
    // Principal components, not a fit: no targets, `k + 3` slots a
    // component, and `n_eff` the decayed weight, under each method.
    for cfg in [pca_cfg(), pca_oja_cfg(), pca_ccipca_cfg()] {
        let m = StreamPca::new(cfg).unwrap();
        assert_eq!(m.n_targets(), 0);
        assert_eq!(m.n_features(), K);
        assert_eq!(m.n_outputs(), K + 3);
        let r = probe_with(m, 0, Some(&StreamPca::n_eff));
        assert_eq!(r.kind, "pca");
        assert_eq!(r.pred_len, r.n_outputs);
        assert_eq!(r.n_eff[0], 0.0);
        assert_eq!(r.n_eff[1], 1.0);
        assert!((r.n_eff[2] - (0.5f64.powf(1.0 / HALFLIFE) + 1.0)).abs() < 1e-12);
        assert!((r.after_gap - (r.before_gap * 0.5f64.powi(10) + 1.0)).abs() < 1e-9);
        assert!(r.roundtrips);
    }
}

#[test]
fn pca_predict_is_the_step() {
    predict_is_the_step_without_the_step(|| StreamPca::new(pca_cfg()).unwrap(), 0, false);
    predict_is_the_step_without_the_step(|| StreamPca::new(pca_oja_cfg()).unwrap(), 0, false);
    predict_is_the_step_without_the_step(|| StreamPca::new(pca_ccipca_cfg()).unwrap(), 0, false);
}

/// Two return series under the shared decay, with a term of each form: a
/// sum of squares, a normalized bipower term, a centered Parzen sum and a
/// flat-top Bartlett sum of two-step returns.
fn power_variation_cfg() -> PowerVariationCfg {
    let mp = |powers: &[f64], normalize: bool| PvTerm {
        kind: PvTermKind::Multipower {
            powers: powers.to_vec(),
            normalize,
        },
        scale: 1,
    };
    let ker = |kernel, bandwidth, flat_top, center, scale| PvTerm {
        kind: PvTermKind::Kernel {
            kernel,
            bandwidth,
            flat_top,
            center,
        },
        scale,
    };
    PowerVariationCfg {
        n_targets: 2,
        terms: vec![
            mp(&[2.0], false),
            mp(&[1.0, 1.0], true),
            ker(PvKernel::Parzen, 3, false, true, 1),
            ker(PvKernel::Bartlett, 2, true, false, 2),
        ],
        decay: decay(),
        min_weight: 3.0,
        window_rows: None,
        window_clock: None,
        closed: WindowClosed::Right,
        max_state_bytes: None,
    }
}

/// A window of six rows, so the probes run through its turnovers.
fn power_variation_rows_cfg() -> PowerVariationCfg {
    PowerVariationCfg {
        decay: Decay::Halflife(f64::INFINITY),
        window_rows: Some(6),
        ..power_variation_cfg()
    }
}

/// The same over a span of the clock: 12.5 units hold thirteen rows.
fn power_variation_clock_cfg() -> PowerVariationCfg {
    PowerVariationCfg {
        decay: Decay::Halflife(f64::INFINITY),
        window_clock: Some(12.5),
        ..power_variation_cfg()
    }
}

#[test]
fn power_variation() {
    // One return series per target, no features, a term each and `n` a
    // series, and `n_eff` every row's weight on the shared recursion.
    let m = PowerVariation::new(power_variation_cfg()).unwrap();
    assert_eq!(m.n_features(), 0, "power_variation reads no features");
    let r = probe_with(m, 2, Some(&PowerVariation::n_eff));
    assert_eq!(r.n_features, 0);
    assert_eq!(r.kind, "power_variation");
    assert_eq!(r.n_targets, 2);
    assert_eq!(r.n_outputs, 10);
    assert_eq!(r.pred_len, 10);
    assert_eq!(r.n_eff[0], 0.0);
    assert_eq!(r.n_eff[1], 1.0);
    assert!((r.n_eff[2] - (0.5f64.powf(1.0 / HALFLIFE) + 1.0)).abs() < 1e-12);
    assert!((r.after_gap - (r.before_gap * 0.5f64.powi(10) + 1.0)).abs() < 1e-9);
    assert!(r.roundtrips);
    // Under a window `n_eff` is the weight inside it (hard rule 8).
    let r = probe_windowed(
        PowerVariation::new(power_variation_rows_cfg()).unwrap(),
        2,
        Some(&PowerVariation::n_eff),
    );
    assert_eq!(r.n_eff, vec![0.0, 1.0, 2.0, 6.0]);
    assert_eq!(r.after_gap, r.before_gap, "a gap does not move the window");
    assert!(r.roundtrips);
    let r = probe_windowed(
        PowerVariation::new(power_variation_clock_cfg()).unwrap(),
        2,
        Some(&PowerVariation::n_eff),
    );
    assert_eq!(r.n_eff, vec![0.0, 1.0, 2.0, 13.0]);
    assert!(r.roundtrips);
}

#[test]
fn power_variation_predict_is_the_step() {
    predict_is_the_step_without_the_step(
        || PowerVariation::new(power_variation_cfg()).unwrap(),
        2,
        false,
    );
    predict_is_the_step_without_the_step(
        || PowerVariation::new(power_variation_rows_cfg()).unwrap(),
        2,
        false,
    );
    predict_is_the_step_without_the_step(
        || PowerVariation::new(power_variation_clock_cfg()).unwrap(),
        2,
        false,
    );
}

/// Two targets read at three levels with their tail means, a coarse
/// accuracy, no window: the contract's recursion is the sum.
fn ddsketch_cfg() -> DdSketchModelCfg {
    DdSketchModelCfg {
        n_targets: 2,
        alpha: 0.02,
        levels: vec![0.05, 0.25, 0.95],
        window_rows: None,
        window_clock: None,
        closed: WindowClosed::Right,
        decay: Decay::Halflife(f64::INFINITY),
        max_state_bytes: None,
    }
}

#[test]
fn ddsketch() {
    // A sketch per target, no features: per target one quantile a level,
    // then one tail mean a level, and `n_eff` every learned row's weight.
    let m = DdSketchModel::new(ddsketch_cfg()).unwrap();
    assert_eq!(m.n_features(), 0, "ddsketch reads no features");
    let r = probe_with(m, 2, Some(&DdSketchModel::n_eff));
    assert_eq!(r.kind, "ddsketch");
    assert_eq!(r.n_targets, 2);
    assert_eq!(r.n_outputs, 12);
    assert_eq!(r.pred_len, 12);
    assert!(r.roundtrips);
}

#[test]
fn ddsketch_predict_is_the_step() {
    predict_is_the_step_without_the_step(|| DdSketchModel::new(ddsketch_cfg()).unwrap(), 2, false);
}

/// The decayed form keeps the step's parity: `predict` reads what the step
/// would, before the row's decay.
#[test]
fn ddsketch_decayed_predict_is_the_step() {
    predict_is_the_step_without_the_step(
        || {
            DdSketchModel::new(DdSketchModelCfg {
                decay: decay(),
                levels: vec![0.25, 0.75],
                ..ddsketch_cfg()
            })
            .unwrap()
        },
        2,
        false,
    );
}

/// The windowed forms keep the step's parity too: a row window and a
/// clock window, each read ahead as the step reads, at levels a few rows
/// identify.
#[test]
fn ddsketch_windowed_predict_is_the_step() {
    predict_is_the_step_without_the_step(
        || {
            DdSketchModel::new(DdSketchModelCfg {
                window_rows: Some(7),
                levels: vec![0.25, 0.75],
                ..ddsketch_cfg()
            })
            .unwrap()
        },
        2,
        false,
    );
    predict_is_the_step_without_the_step(
        || {
            DdSketchModel::new(DdSketchModelCfg {
                window_clock: Some(40.0),
                levels: vec![0.25, 0.75],
                ..ddsketch_cfg()
            })
            .unwrap()
        },
        2,
        false,
    );
}

/// An AR(2) with the probe's two features as exogenous regressors, over
/// two targets, the probes' half-life. A small ridge, as `ewridge_cfg`
/// has: the probes' streams predict from the first rows, where five
/// coefficients have fewer learned rows than that and least squares alone
/// is singular.
fn ar_cfg() -> ArCfg {
    ArCfg {
        n_targets: 2,
        n_features: K,
        lags: vec![1, 2],
        fit_intercept: true,
        decay: Decay::Halflife(HALFLIFE),
        ridge: 0.01,
        min_weight: 3.0,
        max_state_bytes: None,
        select: None,
        window: None,
        window_every: None,
        max_rows_between_snapshots: None,
    }
}

#[test]
fn ar() {
    // A prediction per target from its own lags and the shared feature;
    // `n_eff` every row's weight, decayed (rule 8), each fit taking every
    // row; the lag rings are what `clear_lags` empties.
    let r = probe_with(Ar::new(ar_cfg()).unwrap(), 2, Some(&Ar::n_eff));
    assert_eq!(r.kind, "ar");
    assert_eq!(r.n_targets, 2);
    assert_eq!(r.n_outputs, 2);
    assert_eq!(r.pred_len, 2);
    assert!(r.roundtrips);
    assert!(!r.lags_are_a_no_op);
    let pure = ArCfg {
        lags: vec![1, 4],
        fit_intercept: false,
        ridge: 0.5,
        ..ar_cfg()
    };
    assert!(probe_with(Ar::new(pure).unwrap(), 2, Some(&Ar::n_eff)).roundtrips);
    let chosen = ArCfg {
        lags: vec![1, 2, 3],
        ridge: 0.0,
        select: Some(ArSelect::Aic),
        ..ar_cfg()
    };
    assert!(probe_with(Ar::new(chosen).unwrap(), 2, Some(&Ar::n_eff)).roundtrips);
    // Under a hard window its weight is the window's, read by the caller.
    let windowed = ArCfg {
        window: Some(15.0),
        ..ar_cfg()
    };
    let r = probe_windowed(Ar::new(windowed).unwrap(), 2, Some(&Ar::n_eff));
    assert!(r.roundtrips && !r.lags_are_a_no_op);
}

#[test]
fn ar_predict_is_the_step() {
    predict_is_the_step_without_the_step(|| Ar::new(ar_cfg()).unwrap(), 2, false);
    predict_is_the_step_without_the_step(
        || {
            Ar::new(ArCfg {
                lags: vec![1, 2, 3],
                ridge: 0.0,
                select: Some(ArSelect::Bic),
                ..ar_cfg()
            })
            .unwrap()
        },
        2,
        false,
    );
    predict_is_the_step_without_the_step(
        || {
            Ar::new(ArCfg {
                window: Some(25.0),
                ..ar_cfg()
            })
            .unwrap()
        },
        2,
        false,
    );
    predict_is_the_step_without_the_step(
        || {
            Ar::new(ArCfg {
                min_weight: 6.0,
                ..ar_cfg()
            })
            .unwrap()
        },
        2,
        false,
    );
}

/// Two series at three frequencies, the DC and the Nyquist among them, the
/// probes' half-life.
fn spectral_cfg() -> SpectralCfg {
    SpectralCfg {
        n_targets: 2,
        frequencies: vec![0.0, 0.2, 0.5],
        spacing: 1.0,
        decay: Decay::Halflife(HALFLIFE),
        scaling: SpectralScaling::Density,
        detrend: true,
        min_weight: 0.0,
        max_state_bytes: None,
        window: None,
        window_every: None,
        max_rows_between_snapshots: None,
    }
}

#[test]
fn spectral() {
    // A power per target and frequency, read before the row; `n_eff` every
    // row's weight, decayed (rule 8); the transform is what a break clears.
    let m = Spectral::new(spectral_cfg()).unwrap();
    assert_eq!(m.n_features(), 0, "spectral reads no features");
    let r = probe_with(m, 2, Some(&Spectral::n_eff));
    assert_eq!(r.kind, "spectral");
    assert_eq!(r.n_outputs, 6);
    assert_eq!(r.pred_len, 6);
    assert!(r.roundtrips);
    assert!(!r.lags_are_a_no_op);
    let other = SpectralCfg {
        scaling: SpectralScaling::Spectrum,
        detrend: false,
        ..spectral_cfg()
    };
    assert!(probe_with(Spectral::new(other).unwrap(), 2, Some(&Spectral::n_eff)).roundtrips);
}

#[test]
fn spectral_predict_is_the_step() {
    predict_is_the_step_without_the_step(|| Spectral::new(spectral_cfg()).unwrap(), 2, false);
    predict_is_the_step_without_the_step(
        || {
            Spectral::new(SpectralCfg {
                min_weight: 3.0,
                scaling: SpectralScaling::Spectrum,
                ..spectral_cfg()
            })
            .unwrap()
        },
        2,
        false,
    );
}

/// A SARIMA(1,1,1)(0,1,1,4) with the probe's two features as regressors,
/// over two targets, at fixed parameters: each target's own.
fn sarima_cfg() -> SarimaCfg {
    let params = |beta: f64| SarimaParams {
        exog: vec![beta, -0.5],
        ar: vec![0.4],
        ma: vec![0.3],
        seasonal_ar: vec![],
        seasonal_ma: vec![-0.5],
        sigma2: 0.8,
    };
    SarimaCfg {
        n_targets: 2,
        n_features: K,
        order: [1, 1, 1],
        seasonal_order: [0, 1, 1, 4],
        params: vec![params(1.0), params(0.7)],
        spacing: 1.0,
        min_weight: 3.0,
        max_state_bytes: None,
    }
}

#[test]
fn sarima() {
    // A prediction per target from its filter, before the row; `n_eff`
    // every row's weight summed, with no decay (rule 8); the rings and the
    // filter's state are what `clear_lags` starts again.
    let r = probe_with(Sarima::new(sarima_cfg()).unwrap(), 2, Some(&Sarima::n_eff));
    assert_eq!(r.kind, "sarima");
    assert_eq!(r.n_targets, 2);
    assert_eq!(r.n_outputs, 2);
    assert_eq!(r.pred_len, 2);
    assert!(r.roundtrips);
    assert!(!r.lags_are_a_no_op);
    let arma = SarimaCfg {
        order: [2, 0, 1],
        seasonal_order: [0, 0, 0, 0],
        params: vec![
            SarimaParams {
                exog: vec![1.0, -0.5],
                ar: vec![0.5, -0.2],
                ma: vec![0.3],
                seasonal_ar: vec![],
                seasonal_ma: vec![],
                sigma2: 1.0,
            };
            2
        ],
        ..sarima_cfg()
    };
    assert!(probe_with(Sarima::new(arma).unwrap(), 2, Some(&Sarima::n_eff)).roundtrips);
}

/// `sarima_cfg` with no differencing: the stream's clock steps of several
/// slots leave slots with no features, whose values a differencing over
/// regressors cannot form, so that spec reports on too few rows for the
/// parity's count (`sarima/tests.rs` holds it there, on a stream without
/// regressors).
fn sarima_arma_cfg() -> SarimaCfg {
    let params = |beta: f64| SarimaParams {
        exog: vec![beta, -0.5],
        ar: vec![0.4],
        ma: vec![0.3],
        seasonal_ar: vec![],
        seasonal_ma: vec![-0.5],
        sigma2: 0.8,
    };
    SarimaCfg {
        order: [1, 0, 1],
        seasonal_order: [0, 0, 1, 4],
        params: vec![params(1.0), params(0.7)],
        ..sarima_cfg()
    }
}

#[test]
fn sarima_predict_is_the_step() {
    predict_is_the_step_without_the_step(|| Sarima::new(sarima_arma_cfg()).unwrap(), 2, false);
    predict_is_the_step_without_the_step(
        || {
            Sarima::new(SarimaCfg {
                min_weight: 0.0,
                ..sarima_arma_cfg()
            })
            .unwrap()
        },
        2,
        false,
    );
}
