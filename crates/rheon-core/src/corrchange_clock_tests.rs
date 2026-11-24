//! `corrchange`'s spans on the clock (`window_size`, `monitor_span`,
//! docs/PLAN.md task 553): each kind on the row clock against its count of
//! rows to the bit, the span's end on both sides to the nanosecond, rows of
//! weight 0 in no clock, the window's held critical value rescaled to each
//! report's two windows, a cycle's critical value at its own rows, the
//! state's cap, a state saved mid-span, short spans tested with no floor,
//! and the refusals.

use super::*;
use crate::{OnlineModel, Stamp};

const S: i128 = 1_000_000_000;

fn cfg(kind: CorrChangeKind) -> CorrChangeCfg {
    CorrChangeCfg {
        n_features: 2,
        kind,
        span_rows: 0,
        alpha: 0.05,
        alpha_adjust: "bonferroni".into(),
        bandwidth: None,
        scalar: false,
        decay: Decay::Halflife(f64::INFINITY),
        crit: None,
        n_perm: 40,
        permute_every_rows: usize::MAX,
        perm_block: 1,
        norm: ChangeNorm::L1,
        seed: 3,
        reset: false,
        monitor_rows: 0,
        boundary_gamma: 0.0,
        permute_every: None,
        span_clock: None,
        monitor_clock: None,
        max_state_bytes: None,
    }
}

/// Pairs whose correlation moves from 0.2 to 0.85 and to -0.4 at a third and
/// two thirds: a realistic case, every statistic identified.
fn rows(n: usize, seed: u64) -> Vec<[f64; 2]> {
    let mut g = SplitMix64::new(seed);
    let mut normal = || {
        let u = ((g.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
        let v = ((g.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
        (-2.0 * libm::log(u)).sqrt() * libm::cos(std::f64::consts::TAU * v)
    };
    (0..n)
        .map(|i| {
            let rho = [0.2, 0.85, -0.4][3 * i / n];
            let (a, b) = (normal(), normal());
            [a, rho * a + (1.0 - rho * rho).sqrt() * b]
        })
        .collect()
}

fn bits(out: &crate::Step) -> Vec<u64> {
    out.pred.iter().map(|v| v.to_bits()).collect()
}

/// `monitor` and `sequential` under `window_size = D` (and `monitor_span =
/// D`) on the row clock are `window_rows = D + 1` (and `monitor_rows = D +
/// 1`) to the bit, every output on every row: a span of the clock ends at
/// the row that reaches its length from its first, which on the row clock
/// is its `D + 1`-th row, as every span of the clock that holds rows reads
/// it. Every fourth row weighs 0 and counts in no clock: on the row clock
/// it is no step, and it is in no span.
#[test]
fn a_clock_span_on_the_row_clock_is_its_count_of_rows_plus_one() {
    for (kind, d) in [
        (CorrChangeKind::Monitor, 39usize),
        (CorrChangeKind::Monitor, 9),
        (CorrChangeKind::Sequential, 49),
        (CorrChangeKind::Sequential, 11),
    ] {
        let seq = kind == CorrChangeKind::Sequential;
        let by_rows = CorrChangeCfg {
            span_rows: d + 1,
            monitor_rows: if seq { d + 1 } else { 0 },
            ..cfg(kind)
        };
        let on_clock = CorrChangeCfg {
            span_clock: Some(d as f64),
            monitor_clock: seq.then_some(d as f64),
            ..cfg(kind)
        };
        let (mut a, mut b) = (
            CorrChange::new(by_rows).unwrap(),
            CorrChange::new(on_clock).unwrap(),
        );
        let (mut counted, mut reports, mut flags) = (0u64, 0, 0);
        for (i, x) in rows(900, 5).iter().enumerate() {
            let w = if i % 4 == 3 { 0.0 } else { 1.0 };
            counted += u64::from(w > 0.0);
            b.stamp_next_on_rows(Stamp::from(counted as f64));
            let (p, q) = (a.step(x, &[], 1.0, w), b.step(x, &[], 1.0, w));
            assert_eq!(bits(&p), bits(&q), "{kind:?} D {d}, row {i}");
            reports += usize::from(p.pred[0].is_finite());
            flags += usize::from(p.pred[2] == 1.0);
        }
        // The breaks are found where the spans are long enough to see
        // them: not by spans of ten rows.
        assert!(
            reports > 5 && (d < 30 || flags > 0),
            "{kind:?} D {d}: {reports}, {flags}"
        );
    }
}

/// The span's end on a temporal clock, on both sides of its length to the
/// nanosecond: a span of 3 s begun at 10 s holds the row at 13 s as its last
/// and not the row a nanosecond before it.
#[test]
fn a_span_ends_at_its_length_and_not_a_nanosecond_before() {
    let c = CorrChangeCfg {
        span_clock: Some(3.0),
        ..cfg(CorrChangeKind::Monitor)
    };
    let data = rows(20, 6);
    for (last, closes) in [(13 * S - 1, false), (13 * S, true), (13 * S + 1, true)] {
        let mut m = CorrChange::new(c.clone()).unwrap();
        let stamps = [10 * S, 10 * S + S / 2, 11 * S, 12 * S, 12 * S + S / 3];
        for (t, x) in stamps.iter().zip(&data) {
            m.stamp_next(Stamp::Ns(*t));
            assert!(m.step(x, &[], 0.5, 1.0).pred[0].is_nan());
        }
        m.stamp_next(Stamp::Ns(last));
        let out = m.step(&data[5], &[], 0.5, 1.0);
        assert_eq!(out.pred[1].is_finite(), closes, "last at {last}");
        assert_eq!(m.depth() == 0, closes, "the span closed at {last}");
        if closes {
            // Six rows: the grid correction reads them.
            assert_eq!(out.pred[1], m.cfg.monitor_crit(6));
        }
    }
}

/// No floor on a clock span's rows: a span that happens to hold three rows
/// is tested at its three rows, and cannot flag, by the statistic's own
/// bound (`a_short_span_cannot_flag_by_its_own_bound`); one of a single row
/// has no correlation and reports none.
#[test]
fn a_clock_span_of_few_rows_is_tested_with_no_floor() {
    let c = CorrChangeCfg {
        span_clock: Some(10.0),
        bandwidth: Some(1),
        ..cfg(CorrChangeKind::Monitor)
    };
    let data = rows(30, 8);
    let mut m = CorrChange::new(c).unwrap();
    // Three rows in the first span, one in the second, then the third's.
    let stamps = [0, 4, 10, 20, 31, 33, 41];
    let mut outs = Vec::new();
    for (t, x) in stamps.iter().zip(&data) {
        m.stamp_next(Stamp::Ns(*t as i128 * S));
        outs.push(m.step(x, &[], 1.0, 1.0).pred);
    }
    assert!(outs[2][0].is_finite() && outs[2][2] == 0.0, "{:?}", outs[2]);
    assert_eq!(outs[2][1], m.cfg.monitor_crit(3));
    // The span begun at 20 s closes at 31 s with two rows: collinear, so no
    // statistic, and no flag.
    assert!(outs[4][0].is_nan() && outs[4][2] == 0.0, "{:?}", outs[4]);
}

/// The window kind on the row clock reads the windows `window_rows`
/// reads: under `window_size = n` its windows are `(t − 2n, t − n]` and `(t
/// − n, t]`, `n` rows each, so every report from row `2n + 1` on (the first
/// row at which a row has left both windows) is `window_rows = n`'s to the
/// bit under a fixed critical value. With permutation draws, the two kinds'
/// values are drawn a row apart, so each holds its own; but on a regular
/// clock the windows never change their counts, and a held value is never
/// rescaled: it is reported as drawn.
#[test]
fn a_clock_window_on_the_row_clock_is_its_count_of_rows() {
    let n = 15usize;
    let data = rows(300, 9);
    let by_rows = CorrChangeCfg {
        span_rows: n,
        crit: Some(0.3),
        ..cfg(CorrChangeKind::Window)
    };
    let on_clock = CorrChangeCfg {
        span_clock: Some(n as f64),
        crit: Some(0.3),
        ..cfg(CorrChangeKind::Window)
    };
    let (mut a, mut b) = (
        CorrChange::new(by_rows).unwrap(),
        CorrChange::new(on_clock).unwrap(),
    );
    for (i, x) in data.iter().enumerate() {
        b.stamp_next_on_rows(Stamp::from((i + 1) as f64));
        let (p, q) = (a.step(x, &[], 1.0, 1.0), b.step(x, &[], 1.0, 1.0));
        if i >= 2 * n {
            // `since_flag` aside: the count reports at row `2n` too, and a
            // flag there restarts its count where the clock form, which has
            // not yet reported, does not.
            let (mut p, mut q) = (bits(&p), bits(&q));
            p.remove(3);
            q.remove(3);
            assert_eq!(p, q, "row {i}");
        } else {
            assert!(q.pred[0].is_nan(), "row {i}: both windows have run");
        }
    }
    let mut m = CorrChange::new(CorrChangeCfg {
        span_clock: Some(n as f64),
        permute_every: Some(n as f64),
        ..cfg(CorrChangeKind::Window)
    })
    .unwrap();
    let mut held = 0;
    for (i, x) in data.iter().enumerate() {
        m.stamp_next_on_rows(Stamp::from((i + 1) as f64));
        let before = m.perm_crit;
        let out = m.step(x, &[], 1.0, 1.0);
        if let Some(c) = before {
            assert_eq!(out.pred[1], c, "row {i}: as drawn");
            assert_eq!(m.perm_counts, Some([n as u64, n as u64]));
            held += 1;
        }
    }
    assert!(held > 200, "{held}");
}

/// On an irregular clock the two windows hold different rows, and the held
/// value is rescaled to each report's: `c·√((1/n₁ + 1/n₂)/(1/n₁⁰ +
/// 1/n₂⁰))`, the value as drawn where the counts are those it was drawn
/// at. Rows that come at half the rate halve both counts and raise the
/// value by `√2`, where held as drawn it turned 5% into 17%.
#[test]
fn a_held_value_is_rescaled_to_each_reports_windows() {
    let c = CorrChangeCfg {
        span_clock: Some(20.0),
        permute_every: Some(1e9),
        ..cfg(CorrChangeKind::Window)
    };
    let data = rows(400, 10);
    let mut m = CorrChange::new(c).unwrap();
    // One row a second, then one every two seconds.
    let mut t = 0i128;
    let (mut rescaled, mut checked) = (0, 0);
    for (i, x) in data.iter().enumerate() {
        t += if i < 200 { S } else { 2 * S };
        m.stamp_next(Stamp::Ns(t));
        let key = m.clock.as_ref().unwrap().key(1.0);
        let windows = m.windows_at(Some(key));
        let (c0, counts) = (m.perm_crit, m.perm_counts);
        let out = m.step(x, &[], 1.0, 1.0);
        let (Some(c0), Some([a, b]), Some((_, n1, n2))) = (c0, counts, windows) else {
            continue;
        };
        let s = |p: f64, q: f64| 1.0 / p + 1.0 / q;
        let want = if (a, b) == (n1 as u64, n2 as u64) {
            c0
        } else {
            c0 * (s(n1 as f64, n2 as f64) / s(a as f64, b as f64)).sqrt()
        };
        assert_eq!(out.pred[1], want, "row {i}: {n1}, {n2} against {a}, {b}");
        rescaled += usize::from((a, b) != (n1 as u64, n2 as u64));
        checked += 1;
        if i > 300 {
            assert_eq!((n1, n2), (10, 10), "row {i}");
            assert_eq!((a, b), (20, 20), "drawn at the old rate");
            let r = out.pred[1] / c0;
            assert!((r - 2f64.sqrt()).abs() < 1e-15, "{r}");
        }
    }
    assert!(rescaled > 50 && checked > 300, "{rescaled}, {checked}");
}

/// A `sequential` cycle on an irregular clock: its history is the rows of
/// `window_size` from its first, `m` of them; its period plans `K =
/// ⌊m·monitor_span/window_size⌋` rows and ends at the first of those and its
/// own span; and its critical value is W&G's at the period's information
/// `(K − 3)/(m − 3)`, each cycle its own.
#[test]
fn a_clock_cycle_plans_its_period_from_its_history() {
    let c = CorrChangeCfg {
        span_clock: Some(30.0),
        monitor_clock: Some(45.0),
        crit: None,
        ..cfg(CorrChangeKind::Sequential)
    };
    let data = rows(600, 11);
    let mut m = CorrChange::new(c.clone()).unwrap();
    let mut g = SplitMix64::new(4);
    let mut t = 0i128;
    let mut cycles = std::collections::BTreeSet::new();
    for (i, x) in data.iter().enumerate() {
        t += (S / 4) * (1 + (g.next_u64() % 7) as i128);
        m.stamp_next(Stamp::Ns(t));
        let out = m.step(x, &[], 1.0, 1.0);
        if let Some(mon) = &m.monitoring {
            let (mh, k) = mon.rows_of(&m.cfg);
            assert_eq!(k, ((mh as f64 * 45.0 / 30.0).floor() as usize).max(1));
            assert!(mon.rows.len() <= k, "row {i}");
            let want =
                crate::boundary::sequential_crit(0.05, 0.0, (k as f64 - 3.0) / (mh as f64 - 3.0));
            assert_eq!(mon.crit.to_bits(), want.to_bits(), "row {i}");
            if out.pred[1].is_finite() {
                cycles.insert(mh);
            }
        }
    }
    assert!(cycles.len() > 2, "cycles of several histories: {cycles:?}");
}

/// The rows a span of the clock holds are checked as they come: past half
/// of `max_state_bytes` a warning, once, and the row that would pass it not
/// learned, nor any after it.
#[test]
fn the_cap_warns_past_half_and_refuses_past_the_cap() {
    for kind in [
        CorrChangeKind::Monitor,
        CorrChangeKind::Sequential,
        CorrChangeKind::Window,
    ] {
        let seq = (kind == CorrChangeKind::Sequential).then_some(1e6);
        let probe = CorrChange::new(CorrChangeCfg {
            span_clock: Some(1e6),
            monitor_clock: seq,
            ..cfg(kind)
        })
        .unwrap();
        // Room for about 60 rows past the model's fixed part.
        let per = probe.held_bytes(1) - probe.held_bytes(0);
        let cap = probe.held_bytes(0) + 60 * per;
        let mut m = CorrChange::new(CorrChangeCfg {
            span_clock: Some(1e6),
            monitor_clock: seq,
            max_state_bytes: Some(cap),
            ..cfg(kind)
        })
        .unwrap();
        let (mut warned, mut learned) = (None, 0);
        for (i, x) in rows(100, 12).iter().enumerate() {
            m.stamp_next(Stamp::Ns(i as i128 * S));
            let before = m.n_eff();
            m.step(x, &[], 1.0, 1.0);
            if let Some(w) = m.take_state_warning() {
                assert!(warned.is_none(), "{kind:?}: warned once");
                assert!(2 * w.0 > cap && w.1 == cap, "{kind:?}: {w:?}");
                warned = Some(i);
            }
            learned += usize::from(m.n_eff() > before);
        }
        assert!(warned.is_some_and(|i| i < 60), "{kind:?}: {warned:?}");
        assert_eq!(learned, 60, "{kind:?}: rows up to the cap");
        assert!(m.state_over_cap().is_some_and(|(b, c)| b > c && c == cap));
    }
}

/// A clock form's state, saved in the middle of a span, of a history, of a
/// period and of two windows, goes on to the bit through the named
/// encoding; and a state of a form without a clock writes none of the
/// clock's keys.
#[test]
fn a_clock_state_saved_mid_span_goes_on_to_the_bit() {
    for kind in [
        CorrChangeKind::Monitor,
        CorrChangeKind::Sequential,
        CorrChangeKind::Window,
    ] {
        let c = CorrChangeCfg {
            span_clock: Some(17.5),
            monitor_clock: (kind == CorrChangeKind::Sequential).then_some(20.0),
            ..cfg(kind)
        };
        let data = rows(400, 13);
        let mut g = SplitMix64::new(9);
        let stamps: Vec<i128> = (0..400)
            .scan(0i128, |t, _| {
                *t += S / 2 + (g.next_u64() % (2 * S as u64)) as i128;
                Some(*t)
            })
            .collect();
        for save_at in [37usize, 113, 250] {
            let mut a = CorrChange::new(c.clone()).unwrap();
            for (x, t) in data[..save_at].iter().zip(&stamps) {
                a.stamp_next(Stamp::Ns(*t));
                a.step(x, &[], 1.0, 1.0);
            }
            let bytes = rmp_serde::to_vec_named(&a.state()).unwrap();
            let mut b = CorrChange::restore(&rmp_serde::from_slice(&bytes).unwrap()).unwrap();
            assert_eq!(rmp_serde::to_vec_named(&b.state()).unwrap(), bytes);
            for (x, t) in data[save_at..].iter().zip(&stamps[save_at..]) {
                a.stamp_next(Stamp::Ns(*t));
                b.stamp_next(Stamp::Ns(*t));
                let (p, q) = (a.step(x, &[], 1.0, 1.0), b.step(x, &[], 1.0, 1.0));
                assert_eq!(bits(&p), bits(&q), "{kind:?} saved at {save_at}");
            }
        }
        let plain = CorrChange::new(CorrChangeCfg {
            span_rows: 20,
            monitor_rows: if kind == CorrChangeKind::Sequential {
                20
            } else {
                0
            },
            ..cfg(kind)
        })
        .unwrap();
        let plain = rmp_serde::to_vec_named(&plain.state()).unwrap();
        for f in [
            "span_clock",
            "monitor_clock",
            "max_state_bytes",
            "clock",
            "first",
            "keys",
            "perm_counts",
            "over",
            "warned",
            "history",
            "horizon",
        ] {
            let mut key = vec![0xa0 | f.len() as u8];
            key.extend_from_slice(f.as_bytes());
            assert!(!plain.windows(key.len()).any(|w| w == key), "{kind:?}: {f}");
        }
    }
}

/// `predict` is the step's report without the step, on the clock too: on
/// every row of a stamped stream, the report a copy predicts is the one the
/// model steps to.
#[test]
fn predict_is_the_step_on_the_clock() {
    for kind in [
        CorrChangeKind::Monitor,
        CorrChangeKind::Sequential,
        CorrChangeKind::Window,
    ] {
        let mut m = CorrChange::new(CorrChangeCfg {
            span_clock: Some(12.0),
            monitor_clock: (kind == CorrChangeKind::Sequential).then_some(9.0),
            ..cfg(kind)
        })
        .unwrap();
        let mut t = 0.0;
        for (i, x) in rows(300, 14).iter().enumerate() {
            let d = if i % 5 == 0 { 2.5 } else { 0.75 };
            t += d;
            let p = m.predict(x, d);
            let q = m.step(x, &[], d, 1.0);
            assert_eq!(bits(&p), bits(&q), "{kind:?} row {i} at {t}");
        }
    }
}

/// Every misuse of the clock forms is refused by name.
#[test]
fn the_clock_forms_are_checked() {
    let refused = |c: CorrChangeCfg, says: &str| {
        let e = CorrChange::new(c).unwrap_err();
        assert!(e.contains(says), "{e}");
    };
    for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        refused(
            CorrChangeCfg {
                span_clock: Some(bad),
                ..cfg(CorrChangeKind::Monitor)
            },
            "window_size must be a span of the clock above 0",
        );
        refused(
            CorrChangeCfg {
                span_clock: Some(5.0),
                monitor_clock: Some(bad),
                ..cfg(CorrChangeKind::Sequential)
            },
            "monitor_span must be a span of the clock above 0",
        );
    }
    refused(
        CorrChangeCfg {
            span_rows: 20,
            span_clock: Some(5.0),
            ..cfg(CorrChangeKind::Monitor)
        },
        "window_rows and window_size are both set",
    );
    refused(
        CorrChangeCfg {
            span_rows: 20,
            monitor_rows: 20,
            monitor_clock: Some(5.0),
            ..cfg(CorrChangeKind::Sequential)
        },
        "monitor_span needs window_size",
    );
    refused(
        CorrChangeCfg {
            span_clock: Some(5.0),
            monitor_rows: 20,
            monitor_clock: Some(5.0),
            ..cfg(CorrChangeKind::Sequential)
        },
        "monitor_rows and monitor_span are both set",
    );
    refused(
        CorrChangeCfg {
            span_clock: Some(5.0),
            monitor_clock: Some(5.0),
            ..cfg(CorrChangeKind::Monitor)
        },
        "monitor_span applies only under kind=\"sequential\"",
    );
    refused(
        CorrChangeCfg {
            span_clock: Some(5.0),
            perm_block: 0,
            ..cfg(CorrChangeKind::Window)
        },
        "perm_block must be at least 1",
    );
    refused(
        CorrChangeCfg {
            span_clock: Some(5.0),
            max_state_bytes: Some(0),
            ..cfg(CorrChangeKind::Window)
        },
        "max_state_bytes must be above 0",
    );
    // A block longer than a clock window's rows is allowed: its rows are the
    // data's.
    assert!(
        CorrChange::new(CorrChangeCfg {
            span_clock: Some(5.0),
            perm_block: 1000,
            ..cfg(CorrChangeKind::Window)
        })
        .is_ok()
    );
}

/// `permute_every` is the window's cadence and nothing else's: refused under
/// `monitor` and `sequential`, as the bank's spec refuses it, where it went
/// unchecked (a NaN or a negative span taken) and stepped a clock no draw
/// reads. A state's clock since the last draw is there exactly when the cfg
/// has the cadence: a window state with one and not the other is refused,
/// where it loaded and drew by the clock never, or kept a clock it never
/// read (review 2026-11-23, REGIME-1).
#[test]
fn permute_every_is_the_windows_and_its_clock_matches_it() {
    for kind in [CorrChangeKind::Monitor, CorrChangeKind::Sequential] {
        for d in [5.0, f64::NAN, -1.0] {
            let e = CorrChange::new(CorrChangeCfg {
                span_rows: 20,
                monitor_rows: 20,
                permute_every: Some(d),
                ..cfg(kind)
            })
            .unwrap_err();
            assert!(
                e.contains("permute_every applies only under kind=\"window\""),
                "{e}"
            );
        }
    }
    let window = CorrChangeCfg {
        span_rows: 10,
        permute_every: Some(5.0),
        ..cfg(CorrChangeKind::Window)
    };
    let mut m = CorrChange::new(window.clone()).unwrap();
    for (i, x) in rows(30, 4).iter().enumerate() {
        m.step(x, &[], if i == 0 { 0.0 } else { 1.0 }, 1.0);
    }
    assert!(CorrChange::restore(&m.state()).is_ok());
    let mut without = m.clone();
    without.since_draw = None;
    let mut extra = CorrChange::new(CorrChangeCfg {
        permute_every: None,
        ..window
    })
    .unwrap();
    extra.since_draw = Some(Since::default());
    for (what, bad) in [("no clock", without), ("a clock", extra)] {
        match CorrChange::restore(&bad.state()) {
            Err(crate::StateError::Invalid(e)) => assert!(e.contains("wrong shape"), "{what}: {e}"),
            other => panic!("{what}: {other:?}"),
        }
    }
}
