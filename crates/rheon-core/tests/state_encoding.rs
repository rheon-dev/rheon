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

/// `KalmanCfg` skips its trailing optional fields (`noise_units`, `forecast_spans`,
/// `obs`) when at their defaults, while writing every earlier optional field
/// whenever a later one is written so both compact positional and named
/// msgpack encodings round-trip cleanly in every combination.
#[test]
fn kalman_round_trips_with_every_optional_part_present_or_absent() {
    use rheon_core::{
        Kalman, KalmanCfg, KalmanFamily, KalmanFilter, KalmanLink, KalmanObs, UkfNoise, UkfRule,
    };
    let obs_cases = [
        KalmanObs::default(),
        KalmanObs {
            family: KalmanFamily::Poisson,
            link: KalmanLink::Log,
            filter: KalmanFilter::Ekf,
            ..KalmanObs::default()
        },
        KalmanObs {
            family: KalmanFamily::Gamma,
            link: KalmanLink::Inverse,
            dispersion: 0.5,
            filter: KalmanFilter::Ukf,
            ukf_rule: UkfRule::ScalarIndex,
            ukf_noise: UkfNoise::Expected,
            ..KalmanObs::default()
        },
        KalmanObs {
            family: KalmanFamily::Gaussian,
            link: KalmanLink::Logit,
            filter: KalmanFilter::Ukf,
            ukf_rule: UkfRule::Merwe,
            alpha: 0.8,
            beta: 2.0,
            kappa: 0.0,
            ukf_noise: UkfNoise::Plugin,
            ..KalmanObs::default()
        },
    ];
    for noise_units in [false, true] {
        for horizons in [vec![], vec![2.0, 5.0]] {
            for obs in &obs_cases {
                if !obs.is_gaussian_identity() && !horizons.is_empty() {
                    continue;
                }
                let is_nl = !obs.is_gaussian_identity();
                type ExtraCase = (Option<Vec<Vec<f64>>>, bool, bool);
                let extra_cases: &[ExtraCase] = if is_nl {
                    &[
                        (None, false, false),
                        (Some(vec![vec![1.5, 0.2, -0.1]]), false, false),
                    ]
                } else {
                    &[
                        (None, false, false),
                        (Some(vec![vec![0.5, 1.0, -0.5]]), false, false),
                        (None, true, false),
                        (None, false, true),
                        (Some(vec![vec![0.5, 1.0, -0.5]]), true, true),
                    ]
                };
                for (coef_prior, emit_loglik, emit_mahal) in extra_cases {
                    let cfg = KalmanCfg {
                        n_features: 2,
                        n_targets: 1,
                        fit_intercept: true,
                        decay: Decay::Halflife(50.0),
                        half_life: if is_nl { vec![] } else { vec![25.0] },
                        q: if is_nl { Some(vec![1e-4; 3]) } else { None },
                        obs_var: if is_nl && obs.family == KalmanFamily::Gaussian {
                            Some(0.25)
                        } else {
                            None
                        },
                        p0: if obs.link.defined_at_zero() {
                            1.0
                        } else {
                            0.05
                        },
                        share_p: false,
                        min_weight: 0.0,
                        revert_half_life: vec![f64::INFINITY],
                        standardize: true,
                        min_weight_holds_learning: false,
                        noise_units,
                        horizons: horizons.clone(),
                        obs: obs.clone(),
                        coef_prior: coef_prior.clone(),
                        emit_loglik: *emit_loglik,
                        emit_mahal: *emit_mahal,
                    };
                    let mut m = Kalman::new(cfg).unwrap();
                    for i in 0..8 {
                        let x = [(i as f64) * 0.1, -(i as f64) * 0.05];
                        let y = match obs.family {
                            KalmanFamily::Gaussian => 0.4 + 0.05 * i as f64,
                            KalmanFamily::Poisson => (i % 4) as f64,
                            KalmanFamily::Binomial => (i % 2) as f64,
                            KalmanFamily::Gamma | KalmanFamily::Tweedie => 0.5 + 0.1 * i as f64,
                        };
                        OnlineModel::step(
                            &mut m,
                            &x,
                            &[Some(y)],
                            if i == 0 { 0.0 } else { 1.0 },
                            1.0,
                        );
                    }
                    let what = format!(
                        "kalman noise_units={noise_units} horizons={horizons:?} obs={:?}/{:?} prior={} ll={emit_loglik} mh={emit_mahal}",
                        obs.filter,
                        obs.family,
                        coef_prior.is_some()
                    );
                    roundtrip(m.cfg(), &format!("{what} cfg"));
                    let compact = rmp_serde::to_vec(&m).unwrap();
                    let named = rmp_serde::to_vec_named(&m).unwrap();
                    let mut from_compact: Kalman =
                        rmp_serde::from_slice(&compact).unwrap_or_else(|e| {
                            panic!("{what}: compact encoding does not round-trip: {e}")
                        });
                    let mut from_named: Kalman =
                        rmp_serde::from_slice(&named).unwrap_or_else(|e| {
                            panic!("{what}: named encoding does not round-trip: {e}")
                        });
                    assert_eq!(
                        rmp_serde::to_vec(&from_compact).unwrap(),
                        compact,
                        "{what}: compact bytes changed"
                    );
                    assert_eq!(
                        rmp_serde::to_vec_named(&from_named).unwrap(),
                        named,
                        "{what}: named bytes changed"
                    );
                    let x = [0.25, -0.15];
                    let y = [Some(0.6)];
                    let s0 = OnlineModel::step(&mut m, &x, &y, 1.0, 1.0);
                    let sc = OnlineModel::step(&mut from_compact, &x, &y, 1.0, 1.0);
                    let sn = OnlineModel::step(&mut from_named, &x, &y, 1.0, 1.0);
                    let bits = |v: &[f64]| v.iter().map(|z| z.to_bits()).collect::<Vec<_>>();
                    assert_eq!(
                        bits(&sc.pred),
                        bits(&s0.pred),
                        "{what}: compact continuation diverged"
                    );
                    assert_eq!(
                        bits(&sn.pred),
                        bits(&s0.pred),
                        "{what}: named continuation diverged"
                    );
                    if let (
                        Some(rheon_core::Extra::Kalman { values: v0 }),
                        Some(rheon_core::Extra::Kalman { values: vc }),
                        Some(rheon_core::Extra::Kalman { values: vn }),
                    ) = (&s0.extra, &sc.extra, &sn.extra)
                    {
                        assert_eq!(bits(vc), bits(v0), "{what}: compact extra diverged");
                        assert_eq!(bits(vn), bits(v0), "{what}: named extra diverged");
                    }
                }
            }
        }
    }
}

/// Exhaustive compact + named round-trip across every combination of new
/// `HmmCfgExtras` / `HmmExtras` options (`spherical`, `smooth_lag`, `params`,
/// `means_weight`, `covars_prior`, `covars_weight`, `warm_clock`, `prior_rows`).
#[test]
fn hmm_round_trips_with_every_optional_part_in_both_encodings() {
    use rheon_core::{ClockValue, Hmm, HmmCfg, HmmCfgExtras, HmmCovariance, SeedRule, State};

    // Round-trip through `State` + `Hmm::restore` so `#[serde(skip)]` derived
    // fields (such as `Stabilized::prior_rows`) are rebuilt identically under
    // both compact and named msgpack encodings.
    let check_hmm = |m: &Hmm, what: &str| {
        let st = m.state();
        let compact = rmp_serde::to_vec(&st).unwrap();
        let named = rmp_serde::to_vec_named(&st).unwrap();
        let sc: State = rmp_serde::from_slice(&compact)
            .unwrap_or_else(|e| panic!("{what}: compact encoding does not decode: {e}"));
        let sn: State = rmp_serde::from_slice(&named)
            .unwrap_or_else(|e| panic!("{what}: named encoding does not decode: {e}"));
        let rc =
            Hmm::restore(&sc).unwrap_or_else(|e| panic!("{what}: compact restore failed: {e}"));
        let rn = Hmm::restore(&sn).unwrap_or_else(|e| panic!("{what}: named restore failed: {e}"));
        assert_eq!(rc, rn, "{what}: compact and named restores diverged");
        assert_eq!(
            rmp_serde::to_vec_named(&rc.state()).unwrap(),
            named,
            "{what}: compact round-trip changed serialized state"
        );
        assert_eq!(
            rmp_serde::to_vec_named(&rn.state()).unwrap(),
            named,
            "{what}: named round-trip changed serialized state"
        );
    };

    let base_given = |cov: HmmCovariance, extras: Option<Box<HmmCfgExtras>>| HmmCfg {
        n_features: 2,
        k: 2,
        decay: Decay::Halflife(25.0),
        covariance: cov,
        precision_prior: 1e-2,
        min_weight: 0.0,
        learn: true,
        transition_prior: 1.0,
        transition: Some(vec![0.9, 0.1, 0.1, 0.9]),
        means: Some(vec![-1.5, -1.5, 1.5, 1.5]),
        covs: Some(if matches!(cov, HmmCovariance::Spherical) {
            vec![1.0, 1.0]
        } else {
            vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]
        }),
        warm_rows: 10,
        seed_rule: SeedRule::First,
        seed: 7,
        tvtp: None,
        prior_rows: None,
        warm_clock: None,
        extras,
    };

    for cov in [
        HmmCovariance::Full,
        HmmCovariance::Shared,
        HmmCovariance::Diagonal,
        HmmCovariance::Spherical,
    ] {
        for smooth_lag in [None, Some(3usize)] {
            for params in [
                None,
                Some("t"),
                Some("m"),
                Some("c"),
                Some("mc"),
                Some("tmc"),
            ] {
                for use_conj in [false, true] {
                    if use_conj && matches!(params, Some("t") | Some("m")) {
                        continue;
                    }
                    let (mw, cp, cw) = if use_conj {
                        (
                            if matches!(params, Some("c")) {
                                None
                            } else {
                                Some(2.5)
                            },
                            Some(vec![0.25]),
                            Some(3.0),
                        )
                    } else {
                        (None, None, None)
                    };
                    let ex = HmmCfgExtras {
                        smooth_lag,
                        max_state_bytes: smooth_lag.map(|_| 65_536),
                        params: params.map(str::to_string),
                        means_weight: mw,
                        covars_prior: cp,
                        covars_weight: cw,
                    };
                    let extras = (!ex.is_empty()).then(|| Box::new(ex));
                    let mut m = Hmm::new(base_given(cov, extras)).unwrap();
                    for i in 0..8 {
                        m.label_next(ClockValue::F64(i as f64));
                        let sign = if i < 4 { -1.5 } else { 1.5 };
                        m.step(
                            &[sign + 0.1 * i as f64, sign - 0.1 * i as f64],
                            &[],
                            1.0,
                            1.0,
                        );
                    }
                    check_hmm(
                        &m,
                        &format!(
                            "hmm cov={cov:?} smooth={smooth_lag:?} params={params:?} conj={use_conj}"
                        ),
                    );
                }
            }
        }
    }

    // Warming spherical HMM with warm_clock + smooth_lag + max_state_bytes
    // (before and after seeding).
    let mut w = Hmm::new(HmmCfg {
        n_features: 2,
        k: 2,
        decay: Decay::Halflife(20.0),
        covariance: HmmCovariance::Spherical,
        precision_prior: 1e-2,
        min_weight: 0.0,
        learn: true,
        transition_prior: 1.0,
        transition: None,
        means: None,
        covs: None,
        warm_rows: 2,
        seed_rule: SeedRule::First,
        seed: 3,
        tvtp: None,
        prior_rows: Some(3.0),
        warm_clock: Some(5.0),
        extras: Some(Box::new(HmmCfgExtras {
            smooth_lag: Some(2),
            max_state_bytes: Some(131_072),
            params: None,
            means_weight: None,
            covars_prior: None,
            covars_weight: None,
        })),
    })
    .unwrap();
    for i in 0..3 {
        w.label_next(ClockValue::F64(i as f64));
        w.step(&[i as f64, -(i as f64)], &[], 1.0, 1.0);
    }
    check_hmm(&w, "hmm spherical warm_clock+smooth_lag (warming)");
    for i in 3..10 {
        w.label_next(ClockValue::F64(i as f64));
        w.step(&[i as f64, -(i as f64)], &[], 1.0, 1.0);
    }
    check_hmm(&w, "hmm spherical warm_clock+smooth_lag (seeded)");
}

/// Exhaustive compact + named round-trip across every combination of new
/// `BocpdCfgExtras` options (`nb_hazard`, `obs_var`, `max_state_bytes`,
/// `warm_rows`, `hazard_on_clock`).
#[test]
fn bocpd_round_trips_with_every_optional_part_in_both_encodings() {
    use rheon_core::{Bocpd, BocpdCfg, BocpdCfgExtras, BocpdEmission};

    for (emission, obs_var) in [
        (BocpdEmission::Diag, None),
        (BocpdEmission::Gaussian, None),
        (BocpdEmission::KnownVariance, Some(vec![0.5, 1.25])),
        (BocpdEmission::KnownVariance, Some(vec![1.0])),
    ] {
        for nb_hazard in [None, Some((1usize, 0.05)), Some((4usize, 0.2))] {
            for hazard_on_clock in [false, true] {
                if nb_hazard.is_some() && hazard_on_clock {
                    continue;
                }
                for max_state_bytes in [None, Some(65_536u64)] {
                    for warming in [false, true] {
                        let ex = BocpdCfgExtras {
                            nb_hazard,
                            obs_var: obs_var.clone(),
                            max_state_bytes,
                        };
                        let extras = (!ex.is_empty()).then(|| Box::new(ex));
                        let cfg = BocpdCfg {
                            n_features: 2,
                            hazard: 25.0,
                            hazard_from_row: false,
                            emission,
                            prior_mean: (!warming).then(|| vec![0.0, 0.0]),
                            prior_kappa: 1.0,
                            prior_nu: None,
                            prior_scale: (!warming).then(|| vec![1.0]),
                            robust_beta: 0.0,
                            prune_below: 1e-6,
                            max_run: 50,
                            min_weight: 0.0,
                            warm_rows: warming.then_some(5),
                            hazard_on_clock,
                            warm_clock: None,
                            extras,
                        };
                        let mut m = Bocpd::new(cfg).unwrap();
                        for i in 0..3 {
                            m.step(&[i as f64 * 0.3, -0.2 * i as f64], &[], 1.0, 1.0);
                        }
                        let what = format!(
                            "bocpd {emission:?} nb={nb_hazard:?} clock={hazard_on_clock} \
                             cap={max_state_bytes:?} warm={warming}"
                        );
                        roundtrip(&m, &format!("{what} (early)"));
                        for i in 3..10 {
                            m.step(&[2.0 + i as f64 * 0.1, -1.0], &[], 1.0, 1.0);
                        }
                        roundtrip(&m, &format!("{what} (late)"));
                    }
                }
            }
        }
    }
}

#[test]
fn ewridge_rls_and_sandwich_round_trip_with_constraints_and_variants() {
    let lc = rheon_core::LinearConstraints {
        matrix: vec![vec![1.0, 1.0]],
        rhs: vec![1.0],
    };
    // All four combinations of trailing optional fields on `EwRidgeCfg`:
    // `(max_rows_between_snapshots, constraints)` must round-trip in BOTH
    // compact and named msgpack!
    for (window, rows) in [(None, None), (Some(50.0), Some(3usize))] {
        for constraints in [None, Some(lc.clone())] {
            let cfg = rheon_core::EwRidgeCfg {
                n_features: 2,
                n_targets: 1,
                fit_intercept: true,
                decay: Decay::Halflife(15.0),
                ridge: vec![1e-3],
                feature_sets: vec![],
                standardize: false,
                ridge_scale: false,
                coef_prior: None,
                session_shrink: None,
                long_half_life: None,
                min_weight: 0.0,
                solve_every: 0.0,
                max_rows_between_solves: 1,
                solve_share: None,
                gram_block_rows: 0,
                target_gaps: rheon_core::TargetGaps::OwnRows,
                window,
                window_every: None,
                max_rows_between_snapshots: rows,
                constraints: constraints.clone(),
                evidence: None,
            };
            let mut m = rheon_core::EwRidge::new(cfg).unwrap();
            m.set_keep_factor(true);
            for i in 0..12 {
                let x0 = (i as f64 + 1.0) * 0.3;
                let x1 = ((i * 7 % 11) as f64 - 5.0) * 0.2;
                let y = 0.5 + 0.6 * x0 + 0.4 * x1;
                OnlineModel::step(&mut m, &[x0, x1], &[Some(y)], i as f64, 1.0);
            }
            m.settle_readiness();
            let what = format!(
                "ewridge rows={rows:?} constraints={}",
                constraints.is_some()
            );
            let compact = rmp_serde::to_vec(&m).unwrap();
            let named = rmp_serde::to_vec_named(&m).unwrap();
            let from_compact: rheon_core::EwRidge = rmp_serde::from_slice(&compact)
                .unwrap_or_else(|e| panic!("{what}: compact decode failed: {e}"));
            let from_named: rheon_core::EwRidge = rmp_serde::from_slice(&named)
                .unwrap_or_else(|e| panic!("{what}: named decode failed: {e}"));
            assert_eq!(from_compact.cfg(), m.cfg(), "{what}: compact cfg");
            assert_eq!(from_named.cfg(), m.cfg(), "{what}: named cfg");
            assert_eq!(
                rmp_serde::to_vec_named(&from_compact).unwrap(),
                named,
                "{what}: compact re-encode"
            );
            assert_eq!(
                rmp_serde::to_vec_named(&from_named).unwrap(),
                named,
                "{what}: named re-encode"
            );
            let mut r_compact = rheon_core::EwRidge::restore(&rheon_core::State::new(
                rheon_core::ModelState::EwRidge(Box::new(from_compact)),
            ))
            .unwrap();
            let mut r_named = rheon_core::EwRidge::restore(&rheon_core::State::new(
                rheon_core::ModelState::EwRidge(Box::new(from_named)),
            ))
            .unwrap();
            let s0 = OnlineModel::step(&mut m, &[0.7, -0.4], &[Some(0.8)], 12.0, 1.0);
            let s1 = OnlineModel::step(&mut r_compact, &[0.7, -0.4], &[Some(0.8)], 12.0, 1.0);
            let s2 = OnlineModel::step(&mut r_named, &[0.7, -0.4], &[Some(0.8)], 12.0, 1.0);
            assert_eq!(s1.pred, s0.pred, "{what}: compact step");
            assert_eq!(s2.pred, s0.pred, "{what}: named step");
        }
    }
    // `Rls` unconstrained and constrained.
    for constraints in [None, Some(lc)] {
        let cfg = rheon_core::RlsCfg {
            n_features: 2,
            n_targets: 1,
            fit_intercept: true,
            decay: Decay::Halflife(15.0),
            delta: 0.5,
            coef_prior: None,
            min_weight: 0.0,
            constraints: constraints.clone(),
        };
        let mut m = rheon_core::Rls::new(cfg).unwrap();
        for i in 0..12 {
            let x0 = (i as f64 + 1.0) * 0.3;
            let x1 = ((i * 7 % 11) as f64 - 5.0) * 0.2;
            let y = 0.5 + 0.6 * x0 + 0.4 * x1;
            OnlineModel::step(&mut m, &[x0, x1], &[Some(y)], i as f64, 1.0);
        }
        let what = format!("rls constraints={}", constraints.is_some());
        let compact = rmp_serde::to_vec(&m).unwrap();
        let named = rmp_serde::to_vec_named(&m).unwrap();
        let from_compact: rheon_core::Rls = rmp_serde::from_slice(&compact)
            .unwrap_or_else(|e| panic!("{what}: compact decode failed: {e}"));
        let from_named: rheon_core::Rls = rmp_serde::from_slice(&named)
            .unwrap_or_else(|e| panic!("{what}: named decode failed: {e}"));
        assert_eq!(from_compact.cfg(), m.cfg(), "{what}: compact cfg");
        assert_eq!(from_named.cfg(), m.cfg(), "{what}: named cfg");
        assert_eq!(
            rmp_serde::to_vec_named(&from_compact).unwrap(),
            named,
            "{what}: compact re-encode"
        );
        assert_eq!(
            rmp_serde::to_vec_named(&from_named).unwrap(),
            named,
            "{what}: named re-encode"
        );
        let mut r_compact = rheon_core::Rls::restore(&rheon_core::State::new(
            rheon_core::ModelState::Rls(Box::new(from_compact)),
        ))
        .unwrap();
        let mut r_named = rheon_core::Rls::restore(&rheon_core::State::new(
            rheon_core::ModelState::Rls(Box::new(from_named)),
        ))
        .unwrap();
        let s0 = OnlineModel::step(&mut m, &[0.7, -0.4], &[Some(0.8)], 12.0, 1.0);
        let s1 = OnlineModel::step(&mut r_compact, &[0.7, -0.4], &[Some(0.8)], 12.0, 1.0);
        let s2 = OnlineModel::step(&mut r_named, &[0.7, -0.4], &[Some(0.8)], 12.0, 1.0);
        assert_eq!(s1.pred, s0.pred, "{what}: compact step");
        assert_eq!(s2.pred, s0.pred, "{what}: named step");
    }
    // `Sandwich` plain and with HC0-HC3 variants (full & subset active).
    let subsets: &[&[rheon_core::RobustSeType]] = &[
        &[rheon_core::RobustSeType::Hc0],
        &[rheon_core::RobustSeType::Hc1],
        &[rheon_core::RobustSeType::Hc2],
        &[rheon_core::RobustSeType::Hc3],
        &rheon_core::RobustSeType::ALL,
    ];
    let mut plain = rheon_core::Sandwich::new(3, true, 2);
    for i in 0..20 {
        let x = [
            (i as f64 + 1.0) * 0.2,
            ((i * 5 % 13) as f64 - 6.0) * 0.15,
            ((i * 3 % 7) as f64 - 3.0) * 0.3,
        ];
        plain.update(&x, (i as f64 * 0.37).sin(), 0.95, 1.0);
    }
    roundtrip(&plain, "sandwich plain");
    for &types in subsets {
        for active in [None, Some(vec![0usize, 2usize])] {
            let mut s = rheon_core::Sandwich::with_types(3, true, 1, types, active.clone());
            for i in 0..20 {
                let x = [
                    (i as f64 + 1.0) * 0.2,
                    ((i * 5 % 13) as f64 - 6.0) * 0.15,
                    ((i * 3 % 7) as f64 - 3.0) * 0.3,
                ];
                s.update(&x, (i as f64 * 0.37).sin(), 0.95, 1.0);
            }
            roundtrip(&s, &format!("sandwich types={types:?} active={active:?}"));
        }
    }
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
                constraints: None,
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
