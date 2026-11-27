//! Unit tests for stamp-gated and floor-gated prediction methods across every
//! `rheon-core` model whose read decides a clock window's edge: `Hawkes`,
//! `EwmMoments`, `Expectile`, `Isotonic`, `DdSketchModel`, and `CorrChange`.

use rheon_core::{
    ChangeNorm, CorrChange, CorrChangeCfg, CorrChangeKind, DdSketchModel, DdSketchModelCfg, Decay,
    EwmMoments, EwmMomentsCfg, Expectile, ExpectileCfg, Hawkes, HawkesCfg, Isotonic, IsotonicCfg,
    IsotonicOutOfBounds, MomentKind, MomentLeg, MomentTerm, OnlineModel, Stamp, WindowClosed,
};

fn state_bytes<M: OnlineModel>(m: &M) -> Vec<u8> {
    rmp_serde::to_vec_named(&m.state()).unwrap()
}

fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn hawkes_predict_with_floor_and_at_stamp() {
    let make = |mw: f64| {
        Hawkes::new(HawkesCfg {
            n_targets: 1,
            kernel_decays: vec![0.8, 3.0],
            decay: Decay::Halflife(f64::INFINITY),
            min_weight: mw,
            window_events: None,
            window_clock: Some(0.9),
            closed: WindowClosed::Right,
            max_state_bytes: None,
        })
        .unwrap()
    };
    let mut hi = make(50.0);
    let mut lo = make(0.0);
    let mut mid = make(4.0);
    let times = [0.0, 0.05, 0.1, 0.15, 0.2];
    for (k, &ts) in times.iter().enumerate() {
        let d = if k == 0 { 0.0 } else { ts - times[k - 1] };
        let st = Stamp::Raw(ts, 0.0);
        for m in [&mut hi, &mut lo, &mut mid] {
            m.stamp_next(st);
            m.step(&[], &[Some(1.0)], d, 1.0);
        }
    }
    let d_edge = 0.9 - 0.2;
    let stamp_edge = Stamp::Raw(0.9, 0.0);
    let before = state_bytes(&hi);
    // Floor below n_eff=5 unlocks, above withholds; read-ahead vs exact stamp splits.
    assert!(hi.predict(&[], d_edge).pred[0].is_nan());
    let ahead_lo = hi.predict_with_floor(&[], &[], d_edge, 0.0);
    let at_lo = hi.predict_with_floor_at(&[], d_edge, stamp_edge, 0.0);
    let at_mid = hi.predict_with_floor_at(&[], d_edge, stamp_edge, 4.0);
    let at_shut = hi.predict_with_floor_at(&[], d_edge, stamp_edge, 6.0);
    assert_eq!(state_bytes(&hi), before);
    assert_eq!(bits(&ahead_lo.pred), bits(&lo.predict(&[], d_edge).pred));
    assert_eq!(
        bits(&at_lo.pred),
        bits(&lo.predict_at(&[], d_edge, stamp_edge).pred)
    );
    assert_eq!(
        bits(&at_mid.pred),
        bits(&mid.predict_at(&[], d_edge, stamp_edge).pred)
    );
    assert!(at_lo.pred[0].is_finite());
    assert!(at_shut.pred[0].is_nan());
    assert_ne!(bits(&ahead_lo.pred), bits(&at_lo.pred));
}

#[test]
fn ewm_moments_predict_with_floor_and_at_stamp_including_se() {
    let make = |mw: f64| {
        EwmMoments::new(EwmMomentsCfg {
            decay: Decay::Halflife(f64::INFINITY),
            terms: vec![MomentTerm {
                name: "var".into(),
                powers: vec![(MomentLeg::of(0), 2)],
                kind: MomentKind::Central,
                bias: false,
            }],
            min_weight: mw,
            max_state_bytes: None,
            clocked: true,
            lag_tolerance: None,
            se: true,
            window_rows: None,
            window_clock: Some(0.9),
        })
        .unwrap()
    };
    let mut hi = make(50.0);
    let mut lo = make(0.0);
    let rows = [
        (0.0, 50.0),
        (0.05, 1.0),
        (0.1, 2.0),
        (0.15, 3.0),
        (0.2, 4.0),
    ];
    for (k, &(ts, v)) in rows.iter().enumerate() {
        let d = if k == 0 { 0.0 } else { ts - rows[k - 1].0 };
        let st = Stamp::Raw(ts, 0.0);
        for m in [&mut hi, &mut lo] {
            m.stamp_next(st);
            m.step(&[v], &[], d, 1.0);
        }
    }
    let d_edge = 0.9 - 0.2;
    let stamp_edge = Stamp::Raw(0.9, 0.0);
    let before = state_bytes(&hi);
    // Both moment and SE are withheld at min_weight=50.
    let gated = hi.predict(&[0.0], d_edge);
    assert!(gated.pred.iter().all(|v| v.is_nan()));
    let ahead_lo = hi.predict_with_floor(&[0.0], &[], d_edge, 0.0);
    let at_lo = hi.predict_with_floor_at(&[0.0], d_edge, stamp_edge, 0.0);
    assert_eq!(state_bytes(&hi), before);
    assert_eq!(bits(&ahead_lo.pred), bits(&lo.predict(&[0.0], d_edge).pred));
    assert_eq!(
        bits(&at_lo.pred),
        bits(&lo.predict_at(&[0.0], d_edge, stamp_edge).pred)
    );
    assert!(at_lo.pred.iter().all(|v| v.is_finite()));
    assert_ne!(bits(&ahead_lo.pred), bits(&at_lo.pred));
}

#[test]
fn expectile_predict_at_and_floor_and_on_rows() {
    let make = |mw: f64| {
        Expectile::new(ExpectileCfg {
            n_features: 1,
            n_targets: 1,
            expectile: 0.75,
            decay: Decay::Halflife(f64::INFINITY),
            ridge: 1e-6,
            fit_intercept: true,
            min_weight: mw,
            refit_rows: 0,
            refit_clock: None,
            window_rows: None,
            window_clock: Some(0.9),
            closed: WindowClosed::Right,
            max_state_bytes: None,
        })
        .unwrap()
    };
    let mut hi = make(50.0);
    let mut lo = make(0.0);
    let rows = [
        (0.0, -1.0, 20.0),
        (0.05, 0.0, 1.0),
        (0.1, 1.0, 2.0),
        (0.2, 2.0, 3.0),
    ];
    for (k, &(ts, x, y)) in rows.iter().enumerate() {
        let d = if k == 0 { 0.0 } else { ts - rows[k - 1].0 };
        let st = Stamp::Raw(ts, 0.0);
        for m in [&mut hi, &mut lo] {
            m.stamp_next(st);
            m.step(&[x], &[Some(y)], d, 1.0);
        }
    }
    let d_edge = 0.9 - 0.2;
    let stamp_edge = Stamp::Raw(0.9, 0.0);
    let before = state_bytes(&hi);
    assert!(hi.predict(&[0.5], d_edge).pred[0].is_nan());
    let ahead_lo = hi.predict_with_floor(&[0.5], &[None], d_edge, 0.0);
    let at_lo = hi.predict_with_floor_at(&[0.5], d_edge, stamp_edge, 0.0);
    assert_eq!(state_bytes(&hi), before);
    assert_eq!(
        bits(&at_lo.pred),
        bits(&lo.predict_at(&[0.5], d_edge, stamp_edge).pred)
    );
    assert!(at_lo.pred[0].is_finite());
    assert_ne!(bits(&ahead_lo.pred), bits(&at_lo.pred));

    // On row clock (`stamp_next_on_rows`), `predict_at` preserves row-count keys
    // instead of comparing `ClockState`'s 0-based stamp against 1-based `counted`.
    let mut rc = Expectile::new(ExpectileCfg {
        window_clock: Some(3.0),
        ..lo.cfg().clone()
    })
    .unwrap();
    for k in 0..6 {
        rc.stamp_next_on_rows(Stamp::from(k as f64));
        rc.step(&[k as f64], &[Some(k as f64 * 2.0 + 1.0)], 1.0, 1.0);
    }
    let plain = rc.predict(&[2.5], 1.0);
    let stamped = rc.predict_at(&[2.5], 1.0, Stamp::from(6.0));
    assert_eq!(bits(&plain.pred), bits(&stamped.pred));
}

#[test]
fn isotonic_predict_at_floor_trims_below_and_on_rows() {
    let make = |mw: f64| {
        Isotonic::new(IsotonicCfg {
            n_targets: 1,
            increasing: true,
            y_min: None,
            y_max: None,
            out_of_bounds: IsotonicOutOfBounds::Clip,
            decay: Decay::Halflife(f64::INFINITY),
            min_weight: mw,
            window_rows: None,
            window_clock: Some(0.9),
            closed: WindowClosed::Right,
            max_state_bytes: None,
        })
        .unwrap()
    };
    let mut hi = make(50.0);
    let mut lo = make(0.0);
    let rows = [
        (0.0, 0.0, 10.0),
        (0.05, 1.0, 1.0),
        (0.1, 2.0, 2.0),
        (0.2, 3.0, 3.0),
    ];
    for (k, &(ts, x, y)) in rows.iter().enumerate() {
        let d = if k == 0 { 0.0 } else { ts - rows[k - 1].0 };
        let st = Stamp::Raw(ts, 0.0);
        for m in [&mut hi, &mut lo] {
            m.stamp_next(st);
            m.step(&[x], &[Some(y)], d, 1.0);
        }
    }
    let d_edge = 0.9 - 0.2;
    let stamp_edge = Stamp::Raw(0.9, 0.0);
    let before = state_bytes(&hi);
    assert!(hi.predict(&[1.5], d_edge).pred[0].is_nan());
    let ahead_lo = hi.predict_with_floor(&[1.5], &[None], d_edge, 0.0);
    let at_lo = hi.predict_with_floor_at(&[1.5], d_edge, stamp_edge, 0.0);
    assert_eq!(state_bytes(&hi), before);
    assert_eq!(
        bits(&at_lo.pred),
        bits(&lo.predict_at(&[1.5], d_edge, stamp_edge).pred)
    );
    assert!(at_lo.pred[0].is_finite());
    assert_ne!(bits(&ahead_lo.pred), bits(&at_lo.pred));
    assert_ne!(ahead_lo.n_eff.to_bits(), at_lo.n_eff.to_bits());
    // `trims_below`: window holds weight 4; trimming at 0.9 drops weight 1 -> 3.
    assert!(!lo.trims_below(d_edge, Some(stamp_edge), 3.0));
    assert!(lo.trims_below(d_edge, Some(stamp_edge), 4.0));
    assert!(!lo.trims_below(0.1, Some(Stamp::Raw(0.3, 0.0)), 4.0));

    // Row-clock preservation.
    let mut rc = Isotonic::new(IsotonicCfg {
        window_clock: Some(3.0),
        ..lo.cfg().clone()
    })
    .unwrap();
    for k in 0..6 {
        rc.stamp_next_on_rows(Stamp::from(k as f64));
        rc.step(&[k as f64], &[Some(k as f64)], 1.0, 1.0);
    }
    let plain = rc.predict(&[2.5], 1.0);
    let stamped = rc.predict_at(&[2.5], 1.0, Stamp::from(6.0));
    assert_eq!(bits(&plain.pred), bits(&stamped.pred));
    assert_eq!(plain.n_eff.to_bits(), stamped.n_eff.to_bits());
}

#[test]
fn ddsketch_and_corrchange_predict_at_and_on_rows() {
    // DdSketchModel
    let mut dd = DdSketchModel::new(DdSketchModelCfg {
        n_targets: 1,
        levels: vec![0.4],
        alpha: 0.01,
        decay: Decay::Halflife(f64::INFINITY),
        window_rows: None,
        window_clock: Some(0.9),
        closed: WindowClosed::Right,
        max_state_bytes: None,
    })
    .unwrap();
    let rows = [
        (0.0, 100.0),
        (0.05, 1.0),
        (0.1, 2.0),
        (0.15, 3.0),
        (0.2, 4.0),
    ];
    for (k, &(ts, y)) in rows.iter().enumerate() {
        let d = if k == 0 { 0.0 } else { ts - rows[k - 1].0 };
        dd.stamp_next(Stamp::Raw(ts, 0.0));
        dd.step(&[], &[Some(y)], d, 1.0);
    }
    let d_edge = 0.9 - 0.2;
    let stamp_edge = Stamp::Raw(0.9, 0.0);
    let before = state_bytes(&dd);
    let ahead = dd.predict(&[], d_edge);
    let at = dd.predict_at(&[], d_edge, stamp_edge);
    assert_eq!(state_bytes(&dd), before);
    assert_ne!(ahead.n_eff.to_bits(), at.n_eff.to_bits());
    assert_eq!(at.n_eff, 4.0);
    assert_eq!(ahead.n_eff, 5.0);

    let mut dd_rc = DdSketchModel::new(DdSketchModelCfg {
        window_clock: Some(3.0),
        ..dd.cfg().clone()
    })
    .unwrap();
    for k in 0..6 {
        dd_rc.stamp_next_on_rows(Stamp::from(k as f64));
        dd_rc.step(&[], &[Some(k as f64 + 1.0)], 1.0, 1.0);
    }
    assert_eq!(
        dd_rc.predict(&[], 1.0).n_eff.to_bits(),
        dd_rc.predict_at(&[], 1.0, Stamp::from(6.0)).n_eff.to_bits()
    );

    // CorrChange (Monitor)
    let mut cc = CorrChange::new(CorrChangeCfg {
        n_features: 2,
        kind: CorrChangeKind::Monitor,
        span_rows: 0,
        alpha: 0.05,
        alpha_adjust: "bonferroni".into(),
        bandwidth: None,
        scalar: false,
        decay: Decay::Halflife(f64::INFINITY),
        crit: None,
        n_perm: 200,
        permute_every_rows: 20,
        perm_block: 1,
        norm: ChangeNorm::LInf,
        seed: 0,
        reset: false,
        monitor_rows: 0,
        boundary_gamma: 0.0,
        permute_every: None,
        span_clock: Some(0.9),
        monitor_clock: None,
        max_state_bytes: None,
        emit_n_flagged: false,
    })
    .unwrap();
    let pts = [
        (0.0, [0.1, 0.2]),
        (0.03, [0.5, 0.4]),
        (0.06, [-0.2, -0.3]),
        (0.09, [0.8, 0.7]),
        (0.12, [-0.6, 0.5]),
        (0.15, [0.4, -0.5]),
        (0.18, [-0.7, 0.8]),
        (0.2, [0.3, -0.2]),
    ];
    for (k, &(ts, x)) in pts.iter().enumerate() {
        let d = if k == 0 { 0.0 } else { ts - pts[k - 1].0 };
        cc.stamp_next(Stamp::Raw(ts, 0.0));
        cc.step(&x, &[], d, 1.0);
    }
    let before = state_bytes(&cc);
    let ahead = cc.predict(&[0.2, -0.3], d_edge);
    let at = cc.predict_at(&[0.2, -0.3], d_edge, stamp_edge);
    assert_eq!(state_bytes(&cc), before);
    assert!(ahead.pred[0].is_nan());
    assert!(at.pred[0].is_finite());

    let mut cc_rc = CorrChange::new(CorrChangeCfg {
        span_clock: Some(3.0),
        ..cc.cfg().clone()
    })
    .unwrap();
    for k in 0..6 {
        cc_rc.stamp_next_on_rows(Stamp::from(k as f64));
        cc_rc.step(&[k as f64, (k as f64) * 0.5 + 0.1], &[], 1.0, 1.0);
    }
    let plain = cc_rc.predict(&[1.0, 2.0], 1.0);
    let stamped = cc_rc.predict_at(&[1.0, 2.0], 1.0, Stamp::from(6.0));
    assert_eq!(bits(&plain.pred), bits(&stamped.pred));
}
