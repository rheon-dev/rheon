//! Every model must round-trip through **both** msgpack encodings, with each
//! of its optional parts present and absent.
//!
//! `rmp_serde` writes a struct two ways: `to_vec_named` as a map of field
//! name to value, and `to_vec` (compact) as a bare array, position only. The
//! bank's state files are named, so a new field with a `default` is
//! backward-compatible there. The compact form has no such slack: a
//! `skip_serializing_if` field simply disappears from the array and
//! everything after it slides down one. So **at most one field may skip, and
//! it must be last** -- otherwise a state with the skipped field absent and
//! a later field present decodes the later field into the earlier field's
//! slot, and the failure is a type error at a distance.
//!
//! This is not hypothetical: E66's lag moments were added with a skip in
//! front of the window's, and a marginal with a window but no lags stopped
//! round-tripping compactly. Nothing shipped broke, because bank files are
//! named -- which is exactly why a test is needed rather than a habit.
//!
//! The rule is crate-wide; `marginal` is where it is checked, because it is
//! the model with the most optional parts (lags, bins and a window) and so
//! the only one that can get the ordering wrong in more than one way.

use rheon_core::{Decay, MarginalCfg, OnlineModel};

/// Both encodings, or the reason one failed.
fn roundtrip<T>(m: &T, what: &str)
where
    T: serde::Serialize + serde::de::DeserializeOwned + PartialEq + std::fmt::Debug,
{
    let compact = rmp_serde::to_vec(m).unwrap();
    let named = rmp_serde::to_vec_named(m).unwrap();
    let from_compact: T = rmp_serde::from_slice(&compact)
        .unwrap_or_else(|e| panic!("{what}: compact encoding does not round-trip: {e}"));
    let from_named: T = rmp_serde::from_slice(&named)
        .unwrap_or_else(|e| panic!("{what}: named encoding does not round-trip: {e}"));
    assert_eq!(
        &from_compact, m,
        "{what}: compact round-trip changed the state"
    );
    assert_eq!(&from_named, m, "{what}: named round-trip changed the state");
}

/// A window's cadence each way (docs/PLAN.md task 162): every row, the
/// clock, the rows, and both -- with no spacing and a row cap the shape a
/// skipped `window_every` would read into the wrong slot -- and no window.
const WINDOWS: [(Option<f64>, Option<f64>, Option<usize>); 5] = [
    (None, None, None),
    (Some(50.0), None, None),
    (Some(50.0), Some(2.5), None),
    (Some(50.0), None, Some(3)),
    (Some(50.0), Some(2.5), Some(3)),
];

#[test]
fn marginal_round_trips_with_every_optional_part_present_or_absent() {
    // `cross_lags` (E70) three ways: absent, one of the lags, and none.
    let crosses = [None, Some(vec![2usize]), Some(vec![])];
    for lags in [vec![], vec![1usize, 2]] {
        for (window, every, rows) in WINDOWS {
            for bins in [None, Some(4usize)] {
                for cross_lags in crosses.clone() {
                    // A window and bins are refused together (a snapshot of the
                    // histogram would be bins times the size of one), and so are
                    // a window and lags (the lag ring keeps no snapshot; review
                    // 2026-09-12, C18), so those pairings have no state to encode.
                    if window.is_some() && (bins.is_some() || !lags.is_empty()) {
                        continue;
                    }
                    // `cross_lags` chooses among the lags, and is refused
                    // without them.
                    if lags.is_empty() && cross_lags.is_some() {
                        continue;
                    }
                    let cfg = MarginalCfg {
                        n_features: 2,
                        n_targets: 1,
                        decay: Decay::Halflife(10.0),
                        min_weight: vec![0.0],
                        lags: lags.clone(),
                        serial_rule: None,
                        cross_lags: cross_lags.clone(),
                        bins: bins.map(|n| {
                            Box::new(rheon_core::BinCfg {
                                n_bins: n,
                                edges: None,
                                rule: rheon_core::BinRule::Quantile,
                                warm_rows: 4,
                                stat: rheon_core::BinStat::Mean,
                                refit_rows: 0,
                                budget_mib: None,
                                mad_anchor: false,
                            })
                        }),
                        window,
                        window_every: every,
                        max_rows_between_snapshots: rows,
                        feature_moments: rheon_core::FeatureMomentLayout::PerTarget,
                    };
                    let mut m = rheon_core::Marginal::new(cfg).unwrap();
                    // Bins have two states worth encoding: the warm-up hold with
                    // rows in it (3 of the 4 it waits for), and the histogram
                    // it becomes. Both must survive, or a bank saved during the
                    // warm-up loses the rows it was holding.
                    for i in 0..3 {
                        let v = i as f64;
                        OnlineModel::step(&mut m, &[v, -v], &[Some(v * 0.5)], v, 1.0);
                    }
                    let what = format!(
                        "marginal lags={lags:?} cross_lags={cross_lags:?} window={window:?} \
                     every={every:?} rows={rows:?} bins={bins:?}"
                    );
                    roundtrip(&m, &format!("{what} (held)"));
                    for i in 3..8 {
                        let v = i as f64;
                        OnlineModel::step(&mut m, &[v, -v], &[Some(v * 0.5)], v, 1.0);
                    }
                    roundtrip(&m, &what);
                }
            }
        }
    }
}

/// The same rule on `ew_cov`, whose optional parts are the lag ring and the
/// window. `EwCovModel` skipped both `lag` and `win` when `None`, `lag`
/// first, so a windowed model without lags wrote `[…, win]` compactly and
/// read `win` back into `lag`'s slot; and the window's snapshots
/// (`window::Moments`) skipped `q` -- `Some` only where a Kish sum is kept,
/// which `ew_cov`'s is not -- ahead of `m` and `c`, so a snapshot decoded
/// its mean into `q` (the review of 2026-09-18, B6 and B7). Bank files are
/// named and never broke; the compact form is the rule's tripwire.
#[test]
fn ew_cov_round_trips_with_every_optional_part_present_or_absent() {
    for lags in [vec![], vec![1usize]] {
        for (window, every, rows) in WINDOWS {
            // A window and lags are refused together (the lag ring keeps no
            // snapshot; review 2026-09-12, C18).
            if window.is_some() && !lags.is_empty() {
                continue;
            }
            let cfg = rheon_core::EwCovCfg {
                bias: true,
                n_features: 2,
                decay: Decay::Halflife(10.0),
                stats: vec![rheon_core::EwCovStat::Mean, rheon_core::EwCovStat::Corr],
                min_weight: 0.0,
                precision_prior: None,
                mahal_quantiles: vec![],
                pca: 0,
                pca_every: 0.0,
                max_rows_between_pca: u32::MAX,
                lags: lags.clone(),
                window,
                window_every: every,
                max_rows_between_snapshots: rows,
            };
            let mut m = rheon_core::EwCovModel::new(cfg).unwrap();
            for i in 0..10 {
                let v = i as f64;
                OnlineModel::step(&mut m, &[v, -0.5 * v + 1.0], &[], v, 1.0);
            }
            roundtrip(
                &m,
                &format!("ew_cov lags={lags:?} window={window:?} every={every:?} rows={rows:?}"),
            );
        }
    }
}

/// The rule at the type that broke it: a snapshot without a Kish sum is
/// `Moments { q: None, .. }`, and with `q` skipped and not last the compact
/// form wrote `[w, m, c]` and read `m` back into `q` (review 2026-09-18,
/// B7). No model writes one today -- `EwCov::new` starts `q_sum` at
/// `Some(0)`, which is why the windowed `ew_cov` above never tripped it --
/// so this is the tripwire for the rule, not a reproduction through a model.
#[test]
fn a_snapshot_without_a_kish_sum_round_trips_compactly() {
    let snap = rheon_core::Moments {
        w: 3.0,
        q: None,
        m: vec![1.0, -2.0],
        c: vec![4.0, 0.5, 0.5, 9.0].into(),
        rows: Some(7),
    };
    roundtrip(&snap, "Moments without q");
}

/// A `humanfloat`-annotated field must be **byte-identical** to a bare `f64`
/// in msgpack, and tagged in JSON.
///
/// This is the whole safety argument for `crate::humanfloat`. The helpers
/// exist so a JSON export can carry `NaN` and `±inf`, which `serde_json`
/// otherwise writes as `null`; they key on `is_human_readable()`, which
/// msgpack reports as `false`. If that were ever untrue -- a serde change, a
/// different msgpack crate -- every state file ever written would stop
/// loading, silently, because the field would decode as a string. So the
/// bytes are pinned here rather than assumed.
#[test]
fn a_human_readable_float_does_not_move_the_msgpack() {
    use serde::{Deserialize, Serialize};

    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct Bare {
        a: f64,
        b: Vec<f64>,
        c: Option<f64>,
    }

    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct Tagged {
        #[serde(with = "rheon_core::humanfloat::f64_or_tag")]
        a: f64,
        #[serde(with = "rheon_core::humanfloat::vec_f64_or_tag")]
        b: Vec<f64>,
        #[serde(with = "rheon_core::humanfloat::opt_f64_or_tag")]
        c: Option<f64>,
    }

    for (a, b, c) in [
        (1.5, vec![2.0, 3.0], Some(4.0)),
        (
            f64::INFINITY,
            vec![f64::NEG_INFINITY, 1.0],
            Some(f64::INFINITY),
        ),
        (f64::NAN, vec![f64::NAN], None),
        (-0.0, vec![], Some(-0.0)),
    ] {
        let bare = Bare { a, b: b.clone(), c };
        let tagged = Tagged { a, b: b.clone(), c };

        for (what, bare_bytes, tag_bytes) in [
            (
                "named",
                rmp_serde::to_vec_named(&bare).unwrap(),
                rmp_serde::to_vec_named(&tagged).unwrap(),
            ),
            (
                "compact",
                rmp_serde::to_vec(&bare).unwrap(),
                rmp_serde::to_vec(&tagged).unwrap(),
            ),
        ] {
            assert_eq!(
                bare_bytes, tag_bytes,
                "{what} msgpack moved for a={a} b={b:?} c={c:?}"
            );
        }

        // and it reads back, through msgpack and through JSON
        let back: Tagged =
            rmp_serde::from_slice(&rmp_serde::to_vec_named(&tagged).unwrap()).unwrap();
        let json = serde_json::to_string(&tagged).unwrap();
        let from_json: Tagged = serde_json::from_str(&json).unwrap();
        // Bit for bit, but a NaN's payload: `==` took `0.0` for `-0.0`, so a
        // sign lost in either encoding passed, and `b` and `c` were checked
        // for their length and presence only (review 2026-10-06, CF9).
        let same = |g: f64, w: f64| {
            if w.is_nan() {
                g.is_nan()
            } else {
                g.to_bits() == w.to_bits()
            }
        };
        for (label, got) in [("msgpack", &back), ("json", &from_json)] {
            assert!(same(got.a, a), "{label}: a {:?} against {a:?}", got.a);
            assert_eq!(got.b.len(), b.len(), "{label}: b length");
            for (g, w) in got.b.iter().zip(&b) {
                assert!(same(*g, *w), "{label}: b {:?} against {b:?}", got.b);
            }
            assert_eq!(got.c.is_some(), c.is_some(), "{label}: c presence");
            if let (Some(g), Some(w)) = (got.c, c) {
                assert!(same(g, w), "{label}: c {g:?} against {w:?}");
            }
        }

        // JSON says the non-finite ones out loud rather than writing null
        let json = serde_json::to_string(&tagged).unwrap();
        if !a.is_finite() {
            assert!(
                json.contains("\"inf\"") || json.contains("\"-inf\"") || json.contains("\"nan\""),
                "json did not tag a non-finite: {json}"
            );
            // `"c":null` is legal -- that is `Option::None`. `a` is a plain
            // `f64`, so a null there is the loss this module exists to stop.
            assert!(
                !json.contains("\"a\":null"),
                "json wrote a plain f64 as null: {json}"
            );
        }
    }
}

/// A bins budget is the one field of `BinCfg` that skips, and it is last, so
/// a model with it set and one without both round-trip in both encodings,
/// and one without writes no budget (docs/PLAN.md task 131). The bins'
/// statistic is written whatever it is, ahead of the budget (task 272 (c)),
/// and a Huber histogram -- its cells' robust weights and its targets'
/// scales -- round-trips in both encodings too.
#[test]
fn a_bins_budget_round_trips_in_both_encodings() {
    let build = |budget_mib, stat, mad_anchor, refit_rows| {
        let cfg = MarginalCfg {
            n_features: 2,
            n_targets: 1,
            decay: Decay::Halflife(10.0),
            min_weight: vec![0.0],
            lags: Vec::new(),
            serial_rule: None,
            cross_lags: None,
            bins: Some(Box::new(rheon_core::BinCfg {
                n_bins: 2,
                edges: None,
                rule: rheon_core::BinRule::Quantile,
                warm_rows: 4,
                stat,
                refit_rows,
                budget_mib,
                mad_anchor,
            })),
            feature_moments: rheon_core::FeatureMomentLayout::PerTarget,
            window: None,
            window_every: None,
            max_rows_between_snapshots: None,
        };
        let mut m = rheon_core::Marginal::new(cfg).unwrap();
        for i in 0..8 {
            let v = i as f64;
            OnlineModel::step(&mut m, &[v, -v], &[Some(v * 0.5)], v, 1.0);
        }
        m
    };
    let huber = rheon_core::BinStat::Huber { delta: 1.345 };
    for budget in [None, Some(512.0), Some(f64::INFINITY)] {
        // The Huber scales' sketches anchored and not (task 278).
        for (stat, anchor) in [
            (rheon_core::BinStat::Mean, false),
            (huber, false),
            (huber, true),
        ] {
            roundtrip(
                &build(budget, stat, anchor, 0),
                &format!("marginal bin_budget={budget:?} {stat:?} mad_anchor={anchor}"),
            );
        }
        // The Huber refit (task 278): waiting with rows held past the
        // edges' warm-up (a count of 20 after 8 rows), and run (6).
        for refit in [20, 6] {
            roundtrip(
                &build(budget, huber, true, refit),
                &format!("marginal bin_budget={budget:?} refit {refit}"),
            );
        }
    }
    // Absent, the budget is not written at all.
    let bins = rheon_core::BinCfg {
        n_bins: 2,
        edges: None,
        rule: rheon_core::BinRule::Quantile,
        warm_rows: 4,
        stat: rheon_core::BinStat::Mean,
        refit_rows: 0,
        budget_mib: None,
        mad_anchor: false,
    };
    let compact = rmp_serde::to_vec(&bins).unwrap();
    assert_eq!(
        compact[0], 0x97,
        "seven fields: the statistic, the refit's count, the anchor, and no budget"
    );
}

/// Regression variants (evidence on `ewridge`, `positive` on `lasso`,
/// `huber_norm` on `robust`, `average` on `pa`, and `l1`/`Optimal`/`average`/
/// `invariant` on `sgd`) must round-trip in both compact (positional) and
/// named msgpack encodings for every combination of trailing optional fields.
#[test]
fn regression_variants_round_trip_in_both_encodings() {
    fn roundtrip_model<M>(m: &M, next_rows: &[([f64; 2], f64)], what: &str)
    where
        M: OnlineModel + Clone + serde::Serialize + serde::de::DeserializeOwned,
    {
        let compact = rmp_serde::to_vec(m).unwrap();
        let named = rmp_serde::to_vec_named(m).unwrap();
        let raw_compact: M = rmp_serde::from_slice(&compact)
            .unwrap_or_else(|e| panic!("{what}: compact encoding does not round-trip: {e}"));
        let raw_named: M = rmp_serde::from_slice(&named)
            .unwrap_or_else(|e| panic!("{what}: named encoding does not round-trip: {e}"));
        assert_eq!(
            rmp_serde::to_vec(&raw_compact).unwrap(),
            compact,
            "{what}: compact re-encode differed"
        );
        assert_eq!(
            rmp_serde::to_vec_named(&raw_named).unwrap(),
            named,
            "{what}: named re-encode differed"
        );
        let st = m.state();
        let st_compact: rheon_core::State =
            rmp_serde::from_slice(&rmp_serde::to_vec(&st).unwrap()).unwrap();
        let st_named: rheon_core::State =
            rmp_serde::from_slice(&rmp_serde::to_vec_named(&st).unwrap()).unwrap();
        let mut from_compact = M::restore(&st_compact).unwrap();
        let mut from_named = M::restore(&st_named).unwrap();
        let mut orig = m.clone();
        for (x, y) in next_rows {
            let so = orig.step(x, &[Some(*y)], 1.0, 1.0);
            let sc = from_compact.step(x, &[Some(*y)], 1.0, 1.0);
            let sn = from_named.step(x, &[Some(*y)], 1.0, 1.0);
            for (tag, s) in [("compact", &sc), ("named", &sn)] {
                assert_eq!(so.pred.len(), s.pred.len(), "{what} ({tag}): pred len");
                for (a, b) in so.pred.iter().zip(&s.pred) {
                    if a.is_nan() {
                        assert!(b.is_nan(), "{what} ({tag}): expected NaN, got {b}");
                    } else {
                        assert_eq!(a.to_bits(), b.to_bits(), "{what} ({tag}): pred mismatch");
                    }
                }
                assert_eq!(
                    so.n_eff.to_bits(),
                    s.n_eff.to_bits(),
                    "{what} ({tag}): n_eff"
                );
                assert_eq!(so.extra, s.extra, "{what} ({tag}): extra");
            }
        }
    }

    let next_rows = [([0.4, -0.3], 1.8), ([-0.7, 0.9], -0.4), ([1.1, 0.2], 2.5)];

    // 1. EwRidge: combinations of evidence and max_rows_between_snapshots.
    for evidence in [
        None,
        Some(rheon_core::EvidenceCfg::default()),
        Some(rheon_core::EvidenceCfg::uniform(0.0)),
    ] {
        for (window, every, rows) in WINDOWS {
            let cfg = rheon_core::EwRidgeCfg {
                n_features: 2,
                n_targets: 1,
                fit_intercept: true,
                decay: Decay::Halflife(20.0),
                ridge: vec![if evidence.is_some() { 0.0 } else { 1e-3 }],
                feature_sets: vec![],
                standardize: false,
                ridge_scale: false,
                coef_prior: None,
                session_shrink: None,
                long_half_life: None,
                min_weight: 2.0,
                solve_every: 0.0,
                max_rows_between_solves: 1,
                solve_share: None,
                gram_block_rows: 0,
                target_gaps: rheon_core::TargetGaps::OwnRows,
                window,
                window_every: every,
                max_rows_between_snapshots: rows,
                evidence,
            };
            let mut m = rheon_core::EwRidge::new(cfg).unwrap();
            for i in 0..12 {
                let x0 = (i as f64) * 0.3 - 1.5;
                let x1 = ((i * 7) % 11) as f64 * 0.2 - 1.0;
                let y = 1.2 + 2.0 * x0 - 0.5 * x1 + 0.05 * ((i % 3) as f64 - 1.0);
                OnlineModel::step(
                    &mut m,
                    &[x0, x1],
                    &[Some(y)],
                    if i == 0 { 0.0 } else { 1.0 },
                    1.0,
                );
            }
            roundtrip_model(
                &m,
                &next_rows,
                &format!("ewridge evidence={evidence:?} window={window:?} rows={rows:?}"),
            );
        }
    }

    // 2. Lasso: combinations of positive and max_rows_between_snapshots.
    for positive in [false, true] {
        for (window, every, rows) in WINDOWS {
            let cfg = rheon_core::LassoCfg {
                n_features: 2,
                n_targets: 1,
                fit_intercept: true,
                decay: Decay::Halflife(20.0),
                lasso_path: vec![0.5, 0.1],
                l1_ratio: 0.8,
                select_half_life: None,
                min_weight: 2.0,
                target_min_weight: Vec::new(),
                min_weight_holds_learning: false,
                solve_every: 0.0,
                max_rows_between_solves: 1,
                solve_share: None,
                window,
                window_every: every,
                max_rows_between_snapshots: rows,
                max_iter: 100,
                tol: 1e-10,
                target_gaps: rheon_core::TargetGaps::OwnRows,
                positive,
            };
            let mut m = rheon_core::Lasso::new(cfg).unwrap();
            for i in 0..10 {
                let x0 = (i as f64) * 0.25 - 1.0;
                let x1 = ((i * 5) % 9) as f64 * 0.2 - 0.8;
                let y = 0.5 + 1.5 * x0 - 1.2 * x1;
                OnlineModel::step(
                    &mut m,
                    &[x0, x1],
                    &[Some(y)],
                    if i == 0 { 0.0 } else { 1.0 },
                    1.0,
                );
            }
            roundtrip_model(
                &m,
                &next_rows,
                &format!("lasso positive={positive} window={window:?} rows={rows:?}"),
            );
        }
    }

    // 3. Robust (Huber): combinations of huber_norm, refit_clock, max_state_bytes.
    for norm in [
        rheon_core::HuberNorm::Huber,
        rheon_core::HuberNorm::Tukey,
        rheon_core::HuberNorm::Andrews,
        rheon_core::HuberNorm::Hampel,
        rheon_core::HuberNorm::Ramsay,
    ] {
        for (refit_clock, max_state_bytes) in [
            (None, None),
            (Some(10.0), None),
            (None, Some(128 * 1024)),
            (Some(10.0), Some(128 * 1024)),
        ] {
            let cfg = rheon_core::RobustCfg {
                n_features: 2,
                n_targets: 1,
                fit_intercept: true,
                decay: Decay::Halflife(20.0),
                loss: rheon_core::RobustLoss::Huber {
                    delta: norm.default_delta(),
                },
                ridge: 1e-3,
                standardize: false,
                min_weight: 3.0,
                solve_every: 0.0,
                max_rows_between_solves: 1,
                solve_share: None,
                quantile_eps: 1e-3,
                huber_scale: rheon_core::HuberScale::Mad,
                huber_scale_decay: None,
                feature_sets: vec![],
                ridge_scale: false,
                emit_scale: true,
                refit_rows: if refit_clock.is_some() { 0 } else { 6 },
                mad_anchor: true,
                min_weight_holds_learning: false,
                refit_clock,
                max_state_bytes,
                huber_norm: norm,
            };
            let mut m = rheon_core::Robust::new(cfg).unwrap();
            for i in 0..12 {
                let x0 = (i as f64) * 0.3 - 1.5;
                let x1 = ((i * 3) % 7) as f64 * 0.25 - 0.7;
                let y = 1.0 + 1.5 * x0 - 0.8 * x1 + if i == 9 { 25.0 } else { 0.05 };
                OnlineModel::step(
                    &mut m,
                    &[x0, x1],
                    &[Some(y)],
                    if i == 0 { 0.0 } else { 1.0 },
                    1.0,
                );
            }
            roundtrip_model(
                &m,
                &next_rows,
                &format!(
                    "robust norm={norm:?} refit_clock={refit_clock:?} cap={max_state_bytes:?}"
                ),
            );
        }
    }

    // 4. PA: average false & true.
    for average in [false, true] {
        for standardize in [false, true] {
            let cfg = rheon_core::PaCfg {
                n_features: 2,
                n_targets: 1,
                fit_intercept: true,
                decay: Decay::Halflife(20.0),
                mode: rheon_core::PaMode::Pa1,
                c: 1.0,
                eps: 0.01,
                min_weight: 3.0,
                constraint: None,
                standardize,
                average,
            };
            let mut m = rheon_core::Pa::new(cfg).unwrap();
            for i in 0..15 {
                let x0 = (i as f64) * 0.2 - 1.0;
                let x1 = ((i * 5) % 8) as f64 * 0.3 - 1.0;
                OnlineModel::step(
                    &mut m,
                    &[x0, x1],
                    &[Some(0.5 + 1.2 * x0 - 0.7 * x1)],
                    if i == 0 { 0.0 } else { 1.0 },
                    1.0,
                );
            }
            roundtrip_model(
                &m,
                &next_rows,
                &format!("pa average={average} standardize={standardize}"),
            );
        }
    }

    // 5. SGD: combinations of schedule, l1, average, invariant, refit_clock, max_state_bytes.
    for (schedule, l1, l2, average, invariant) in [
        (rheon_core::LearningRate::Constant, 0.0, 0.0, false, false),
        (rheon_core::LearningRate::Constant, 0.01, 0.0, false, false),
        (rheon_core::LearningRate::Constant, 0.01, 0.02, true, false),
        (rheon_core::LearningRate::Optimal, 0.0, 0.05, false, false),
        (rheon_core::LearningRate::Optimal, 0.01, 0.05, true, false),
        (rheon_core::LearningRate::Constant, 0.0, 0.01, false, true),
        (rheon_core::LearningRate::Optimal, 0.01, 0.05, true, true),
    ] {
        for standardize in [false, true] {
            let cfg = rheon_core::SgdCfg {
                n_features: 2,
                n_targets: 1,
                fit_intercept: true,
                decay: Decay::Halflife(50.0),
                loss: rheon_core::SgdLoss::Squared,
                learning_rate: 0.05,
                schedule,
                l2,
                min_weight: 3.0,
                clip_gradient: 1e3,
                constraint: None,
                standardize,
                strict_binary: false,
                huber_scale: rheon_core::HuberScale::Rms,
                huber_scale_decay: None,
                refit_rows: 0,
                min_weight_holds_learning: false,
                target_spread: true,
                mad_anchor: true,
                refit_clock: None,
                max_state_bytes: None,
                l1,
                average,
                invariant,
            };
            let mut m = rheon_core::Sgd::new(cfg).unwrap();
            for i in 0..15 {
                let x0 = (i as f64) * 0.2 - 1.0;
                let x1 = ((i * 3) % 7) as f64 * 0.25 - 0.8;
                OnlineModel::step(
                    &mut m,
                    &[x0, x1],
                    &[Some(0.5 + 1.5 * x0 - 0.6 * x1)],
                    if i == 0 { 0.0 } else { 1.0 },
                    1.0,
                );
            }
            roundtrip_model(
                &m,
                &next_rows,
                &format!(
                    "sgd sched={schedule:?} l1={l1} l2={l2} avg={average} inv={invariant} std={standardize}"
                ),
            );
        }
    }
}
