//! `ewm_moments`' lags, windows and standard errors (PLAN 554, part 4):
//! held to the Python reference on seeded streams of two columns; the
//! as-of rule at `lag_tolerance` from both sides; a gap longer than the lag
//! closing no pair; each new form saved and restored to the bit; a row of
//! weight 0 on a row clock and on a time clock against `ew_cov`; the state
//! formula against the encoding; the clock span's cap.

use super::*;
use crate::OnlineModel;

/// A row of `stream2`: `x`, `y`, the weight, and the clock's step to it.
type Row2 = (f64, f64, f64, f64);

/// The stream `scripts/ewm_moments_literals_554.py`'s `stream2` feeds the
/// reference: four draws a row, `x` as `stream`'s, `y = 1/2 + x/2 + q'`,
/// the weight 0 on `zero_rows`, the clock stepping 1, 1.5, 2, ... and 9.5
/// on `gap_row`. Every clock is a sum of halves, so summed it is exact.
fn stream2(seed: u64, zero_rows: &[usize], gap_row: Option<usize>) -> Vec<Row2> {
    let mut s = seed;
    (0..60)
        .map(|i| {
            let (u, v, q, q2) = (lcg(&mut s), lcg(&mut s), lcg(&mut s), lcg(&mut s));
            let x = 3.0 + 4.0 * u * u * u + 0.5 * v;
            let y = 0.5 + 0.5 * x + q2;
            let w = if zero_rows.contains(&i) {
                0.0
            } else {
                0.25 + q
            };
            let d = if Some(i) == gap_row {
                9.5
            } else {
                1.0 + (i % 3) as f64 * 0.5
            };
            (x, y, w, d)
        })
        .collect()
}

/// `"x@3.5"` as a leg: `x` column 0, `y` column 1.
fn leg(s: &str) -> Leg {
    let (col, lag) = match s.split_once('@') {
        Some((c, l)) => (c, l.parse().unwrap()),
        None => (s, 0.0),
    };
    Leg::lagged(usize::from(col == "y"), lag)
}

fn kind(s: &str) -> MomentKind {
    match s {
        "central" => MomentKind::Central,
        "raw" => MomentKind::Raw,
        "standardized" => MomentKind::Standardized,
        _ => MomentKind::Cumulant,
    }
}

/// The script's terms at one `bias`, suffixed; under `bias = false` only
/// those the reference keeps (no raw term, order 4 at most).
fn terms_of(spec: &Spec, bias: bool, suffix: &str) -> Vec<MomentTerm> {
    spec.iter()
        .filter(|(_, legs, k)| bias || (*k != "raw" && legs.iter().map(|l| l.1).sum::<u32>() <= 4))
        .map(|&(name, legs, k)| MomentTerm {
            name: format!("{name}{suffix}"),
            powers: legs.iter().map(|&(l, p)| (leg(l), p)).collect(),
            kind: kind(k),
            bias,
        })
        .collect()
}

const ROWS4: [usize; 5] = [2, 13, 31, 44, 59];

/// Feeds `rows` as the stream does: a row of weight 0 is never stepped,
/// and on a clock its step is folded into the next row's; on rows each
/// learned row is one step. Every output at `ROWS4`, read after the row,
/// by the reference's name (`se_<term>` for a standard error).
fn run4(cfg: EwmMomentsCfg, rows: &[Row2]) -> Vec<(usize, String, f64)> {
    let clocked = cfg.clocked;
    let mut m = EwmMoments::new(cfg).unwrap();
    let mut names: Vec<String> = m.cfg.terms.iter().map(|t| t.name.clone()).collect();
    if m.cfg.se {
        let se: Vec<String> = names.iter().map(|n| format!("se_{n}")).collect();
        names.extend(se);
    }
    let mut pending = 0.0;
    let mut out = Vec::new();
    for (i, &(x, y, w, d)) in rows.iter().enumerate() {
        pending += d;
        if w > 0.0 {
            m.update_row(&[x, y], if clocked { pending } else { 1.0 }, w);
            pending = 0.0;
        }
        if ROWS4.contains(&i) {
            for (n, v) in names.iter().zip(m.read()) {
                out.push((i, n.clone(), v.unwrap_or(f64::NAN)));
            }
        }
    }
    out
}

fn base(decay: Decay, terms: Vec<MomentTerm>) -> EwmMomentsCfg {
    EwmMomentsCfg::new(decay, terms)
}

/// Row lags and cross terms, plug-in and unbiased, under `lam = 0.9` with
/// zero-weight rows: within 5e-14 of the reference relative to the size
/// (or to 1 below it), withheld where it withholds. Measured 2026-11-23:
/// 1.8e-15. They part by rounding: the Rust keeps each mean as a pair.
#[test]
fn row_lags_match_the_python_reference() {
    let rows = stream2(5542, &[0, 17, 40], None);
    let mut got = run4(base(Decay::Lam(0.9), terms_of(LAG_TERMS, true, "")), &rows);
    got.extend(run4(
        base(Decay::Lam(0.9), terms_of(LAG_TERMS, false, "u")),
        &rows,
    ));
    let (worst, _) = against(REF_LAG_ROWS, &got);
    assert!(worst < 5e-14, "{worst:e}");
}

/// Clock lags, as-of, on a clock with a gap of 9.5 at row 30 (longer than
/// either lag, 2 and 3.5) and zero-weight rows: the default tolerance (the
/// lag) and a tolerance of 0.5, each within 5e-14 of the reference.
/// Measured 2026-11-23: 4.4e-16 and 4.4e-16.
#[test]
fn clock_lags_match_the_python_reference() {
    let rows = stream2(5543, &[0, 17, 18, 40], Some(30));
    let cfg = EwmMomentsCfg {
        clocked: true,
        ..base(Decay::Halflife(7.0), terms_of(CLOCK_TERMS, true, ""))
    };
    let (worst, _) = against(REF_LAG_CLOCK, &run4(cfg.clone(), &rows));
    assert!(worst < 5e-14, "{worst:e}");
    let tol = EwmMomentsCfg {
        lag_tolerance: Some(0.5),
        ..cfg
    };
    let (worst, _) = against(REF_LAG_TOL, &run4(tol, &rows));
    assert!(worst < 5e-14, "{worst:e}");
}

/// A window of 12 rows under `lam = 0.95`, and a clock window of 9 under a
/// half-life of 20 (a gap at row 30 empties it but for the row): within
/// 5e-14 of the reference, which folds a window's blocks left to right
/// where the Rust merges blocks of `⌊√12⌋ = 3` rows (16 on a clock).
/// Measured 2026-11-23: 2.3e-15 and 4.3e-16.
#[test]
fn windows_match_the_python_reference() {
    let rows = stream2(5542, &[0, 17, 40], None);
    let cfg = EwmMomentsCfg {
        window_rows: Some(12),
        ..base(Decay::Lam(0.95), terms_of(WINDOW_TERMS, true, ""))
    };
    let (worst, _) = against(REF_WIN_ROWS, &run4(cfg, &rows));
    assert!(worst < 5e-14, "{worst:e}");
    let clock = stream2(5543, &[0, 17, 18, 40], Some(30));
    let cfg = EwmMomentsCfg {
        clocked: true,
        window_clock: Some(9.0),
        ..base(Decay::Halflife(20.0), terms_of(WINDOW_TERMS, true, ""))
    };
    let (worst, _) = against(REF_WIN_CLOCK, &run4(cfg, &clock));
    assert!(worst < 5e-14, "{worst:e}");
}

/// Each term's standard error -- central, standardized, raw, cumulant,
/// cross, lagged -- within 5e-14 of the reference's delta method at Kish's
/// `n`, withheld where it withholds (a correlation of two rows, exactly 1,
/// keeps no digit of its error). Measured 2026-11-23: 6.4e-15.
#[test]
fn standard_errors_match_the_python_reference() {
    let rows = stream2(5542, &[0, 17, 40], None);
    let cfg = EwmMomentsCfg {
        se: true,
        ..base(Decay::Lam(0.97), terms_of(SE_TERMS, true, ""))
    };
    let (worst, _) = against(REF_SE, &run4(cfg, &rows));
    assert!(worst < 5e-14, "{worst:e}");
}

/// The standard error of a variance under equal weights and no decay is
/// the textbook `sqrt((μ_4 − μ_2²) / n)`, to rounding.
#[test]
fn the_variances_standard_error_is_the_textbook_form() {
    let cfg = EwmMomentsCfg {
        se: true,
        ..base(
            Decay::Lam(1.0),
            vec![MomentTerm::of_column("c2", 2, MomentKind::Central, true)],
        )
    };
    let mut m = EwmMoments::new(cfg).unwrap();
    let xs: Vec<f64> = stream(3, &[], false).iter().map(|r| r.0).collect();
    for &x in &xs {
        m.update(x, 1.0, 1.0);
    }
    let n = xs.len() as f64;
    let mean = xs.iter().sum::<f64>() / n;
    let mu = |p: i32| xs.iter().map(|x| (x - mean).powi(p)).sum::<f64>() / n;
    let want = ((mu(4) - mu(2) * mu(2)) / n).sqrt();
    let got = m.read()[1].unwrap();
    assert!((got - want).abs() < 1e-13 * want, "{got} {want}");
}

/// A clocked model of the lagged mean `x@2` (a raw term of order 1).
fn lag_mean(tol: Option<f64>) -> EwmMoments {
    let t = MomentTerm {
        name: "lagmean".into(),
        powers: vec![(Leg::lagged(0, 2.0), 1)],
        kind: MomentKind::Raw,
        bias: true,
    };
    EwmMoments::new(EwmMomentsCfg {
        clocked: true,
        lag_tolerance: tol,
        ..base(Decay::Halflife(f64::INFINITY), vec![t])
    })
    .unwrap()
}

/// The as-of rule at `lag_tolerance`, from both sides: lag 2, rows at 0,
/// 1, 4 and 7. At 4 the as-of point is 2 and the latest row at or before
/// it is at 1, one before the point; at 7 the point is 5 and the row at 4
/// is one before it. A tolerance of 1 pairs both (at the tolerance), one of
/// `1 − 1e-9` neither, and the default (the lag) pairs both. The reach is
/// the lag plus the tolerance, one span of the clock against the stamps'
/// one subtraction.
#[test]
fn lag_tolerance_pairs_at_it_and_not_past_it() {
    let feed = |mut m: EwmMoments| {
        let mut out = Vec::new();
        for (t, x) in [(0.0, 1.0), (1.0, 3.0), (4.0, 2.0), (7.0, 5.0)] {
            m.stamp_next(Stamp::from(t));
            m.update_row(&[x], 1.0, 1.0);
            out.push(m.read()[0]);
        }
        out
    };
    let none = Err(Withheld::NoLaggedPair);
    assert_eq!(feed(lag_mean(Some(1.0))), [none, none, Ok(3.0), Ok(2.5)]);
    assert_eq!(feed(lag_mean(None)), [none, none, Ok(3.0), Ok(2.5)]);
    let below = 1.0 - 1e-9;
    assert_eq!(feed(lag_mean(Some(below))), [none; 4]);
    // A lag tolerance is a clock's: on rows a lag pairs with that row.
    let mut c = lag_mean(Some(1.0)).cfg.clone();
    c.clocked = false;
    assert!(
        c.validate()
            .unwrap_err()
            .contains("lag_tolerance is the as-of tolerance")
    );
}

/// A gap longer than the lag closes no pair: the row after it leaves the
/// lagged term where it was (decayed alone, so its value does not move)
/// and still counts for the unlagged one; the next row in reach pairs.
/// A break clears the buffer and the moments go on.
#[test]
fn a_gap_longer_than_the_lag_leaves_no_pair() {
    let terms = vec![
        MomentTerm {
            name: "ac".into(),
            powers: vec![(Leg::of(0), 1), (Leg::lagged(0, 1.0), 1)],
            kind: MomentKind::Central,
            bias: true,
        },
        MomentTerm::of_column("mean", 1, MomentKind::Raw, true),
    ];
    let mut m = EwmMoments::new(EwmMomentsCfg {
        clocked: true,
        ..base(Decay::Halflife(5.0), terms)
    })
    .unwrap();
    let mut out = Vec::new();
    let mut last = 0.0;
    for (t, x) in [
        (0.0, 1.0),
        (1.0, 3.0),
        (2.0, 2.0),
        (3.0, 5.0),
        (10.0, 4.0),
        (11.0, 7.0),
    ] {
        m.update_row(&[x], t - last, 1.0);
        last = t;
        out.push(m.read());
    }
    assert_eq!(out[0][0], Err(Withheld::NoLaggedPair));
    assert_eq!(out[4][0], out[3][0], "the row after the gap closes no pair");
    assert_ne!(out[4][1], out[3][1], "and counts for the mean");
    assert_ne!(out[5][0], out[4][0], "the next row in reach pairs");
    m.clear_lags();
    assert!(
        m.bufs
            .iter()
            .all(|b| b.vals.is_empty() && b.stamps.is_empty())
    );
    let before = m.read();
    m.update_row(&[8.0], 1.0, 1.0);
    let after = m.read();
    assert_eq!(after[0], before[0], "the cleared buffer has no partner");
    assert_ne!(after[1], before[1]);
}

/// Every form part 4 adds: row lags, clock lags, a row window, a clock
/// window, the standard errors.
fn forms() -> Vec<(&'static str, EwmMomentsCfg)> {
    let lag = |c: usize, k: f64| Leg::lagged(c, k);
    let terms = || {
        vec![
            MomentTerm {
                name: "ac".into(),
                powers: vec![(Leg::of(0), 1), (lag(0, 2.0), 1)],
                kind: MomentKind::Standardized,
                bias: true,
            },
            MomentTerm {
                name: "xy".into(),
                powers: vec![(Leg::of(0), 2), (lag(1, 1.0), 1)],
                kind: MomentKind::Central,
                bias: true,
            },
            MomentTerm::of_column("k4", 4, MomentKind::Cumulant, false),
            MomentTerm::of_column("s3", 3, MomentKind::Standardized, true),
        ]
    };
    let ew = base(Decay::Halflife(9.0), terms());
    vec![
        ("row lags", ew.clone()),
        (
            "clock lags",
            EwmMomentsCfg {
                clocked: true,
                lag_tolerance: Some(1.5),
                ..ew.clone()
            },
        ),
        (
            "row window",
            EwmMomentsCfg {
                window_rows: Some(20),
                ..ew.clone()
            },
        ),
        (
            "clock window",
            EwmMomentsCfg {
                clocked: true,
                window_clock: Some(13.0),
                ..ew.clone()
            },
        ),
        (
            "se",
            EwmMomentsCfg {
                se: true,
                ..ew.clone()
            },
        ),
    ]
}

fn steps(m: &mut EwmMoments, rows: &[Row2]) -> Vec<u64> {
    let mut out = Vec::new();
    for &(x, y, w, d) in rows {
        let s = m.step(&[x, y], &[], d, w);
        out.extend(s.pred.iter().map(|v| v.to_bits()));
        out.push(s.n_eff.to_bits());
    }
    out
}

/// Each form, saved and loaded at every row (as a state file, the window's
/// block summaries rebuilt from the ring) and read between rows, goes on to
/// the same bits as the run never saved: the outputs, `n_eff` and the
/// final state.
#[test]
fn each_form_saves_and_goes_on_to_the_bit() {
    let rows = stream2(77, &[5, 6, 33], Some(25));
    for (name, cfg) in forms() {
        let mut whole = EwmMoments::new(cfg.clone()).unwrap();
        let want = steps(&mut whole, &rows);
        let mut m = EwmMoments::new(cfg).unwrap();
        let mut got = Vec::new();
        for (i, r) in rows.iter().enumerate() {
            if i % 3 == 1 {
                let _ = m.predict(&[r.0, r.1], r.3);
                let bytes = rmp_serde::to_vec_named(&m.state()).unwrap();
                let s: crate::State = rmp_serde::from_slice(&bytes).unwrap();
                m = EwmMoments::restore(&s).unwrap();
            }
            got.extend(steps(&mut m, std::slice::from_ref(r)));
        }
        assert_eq!(got, want, "{name}");
        assert_eq!(m.state(), whole.state(), "{name}");
    }
}

/// `predict` is the step's read without the step, every form.
#[test]
fn predict_reads_what_the_step_reads() {
    let rows = stream2(78, &[4], Some(20));
    for (name, cfg) in forms() {
        let mut m = EwmMoments::new(cfg).unwrap();
        for &(x, y, w, d) in &rows {
            let p = m.predict(&[x, y], d);
            let s = m.step(&[x, y], &[], d, w);
            let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&p.pred), bits(&s.pred), "{name}");
            assert_eq!(p.n_eff.to_bits(), s.n_eff.to_bits(), "{name}");
        }
    }
}

/// The state's bytes against its encoding, every form, as rows go in: the
/// numbers the encoding holds outside the settings are
/// [`EwmMoments::saved_bytes`] (the state less a window's block summaries,
/// a cache), and before the first row the state is what the settings fix.
#[test]
fn each_forms_state_is_its_formula() {
    let rows = stream2(79, &[3], Some(30));
    for (name, cfg) in forms() {
        let mut m = EwmMoments::new(cfg.clone()).unwrap();
        assert_eq!(m.state_bytes(), cfg.state_bytes(), "{name}");
        for (i, &(x, y, w, d)) in rows.iter().enumerate() {
            m.step(&[x, y], &[], d, w);
            // A row lag's buffer and a row window's ring are counted full
            // from the first row: the bound, which they reach once as many
            // rows of weight are in.
            let saved = 8 * saved_numbers(&m);
            if cfg.clocked || i >= 25 {
                assert_eq!(saved, m.saved_bytes(), "{name}, row {i}");
            } else {
                assert!(saved <= m.saved_bytes(), "{name}, row {i}");
            }
        }
    }
    // The formula, spelled: two columns, the row-lag form. Summaries: `ac`
    // over (x, x@2), orders to 2: 1 + 4 + 3 + 1 = 9; `xy` over (x, y@1),
    // (2, 1)'s closure (2,0), (1,1), (2,1): 1 + 4 + 3 + 1 = 9; `k4` and
    // `s3` over x, orders 2 to 4 and r_2..r_4: 1 + 2 + 3 + 3 = 9. Buffers:
    // x, 1 + 2; y, 1 + 1. The weight: 1. 27 + 5 + 1 = 33 doubles.
    let (_, ew) = &forms()[0];
    assert_eq!(ew.state_bytes(), 8 * 33);
    // The row window of 20 rows: the weight, the ring's two counters, 20
    // rows of a decay, a weight and three legs (x, x@2, y@1), and the two
    // buffers' 5: 1 + 2 + 100 + 5 = 108 saved; and blocks of ⌊√20⌋ = 4
    // rows, (20 / 4 + 2) (27 + 2) = 203 working.
    let (_, win) = &forms()[2];
    assert_eq!(win.state_bytes(), 8 * (108 + 203));
}

/// A clock span's state is checked as each row goes in: the first row past
/// half of `max_state_bytes` raises a warning, once; a row that would take
/// the state past it is not learned, nor any after it, and is reported.
/// A form the settings fix is refused before the first row, by name.
#[test]
fn a_clock_spans_state_warns_past_half_and_refuses_at_the_cap() {
    let t = MomentTerm {
        name: "ac".into(),
        powers: vec![(Leg::of(0), 1), (Leg::lagged(0, 1000.0), 1)],
        kind: MomentKind::Central,
        bias: true,
    };
    let cfg = EwmMomentsCfg {
        clocked: true,
        max_state_bytes: Some(2048),
        ..base(Decay::Lam(0.99), vec![t])
    };
    let fixed = cfg.state_bytes();
    let mut m = EwmMoments::new(cfg).unwrap();
    let mut warned_at = None;
    let mut refused_at = None;
    for i in 0..200 {
        m.step(&[i as f64], &[], 1.0, 1.0);
        if let Some((bytes, cap)) = m.take_state_warning() {
            assert!(warned_at.is_none(), "warned once");
            assert!(2 * bytes > cap && 2 * (bytes - 24) <= cap, "{bytes}");
            warned_at = Some(i);
        }
        if refused_at.is_none() && m.state_over_cap().is_some() {
            refused_at = Some(i);
        }
    }
    // Each row holds 3 doubles in the buffer: past 1024 bytes at the row
    // that takes it there, past 2048 at the row after the last that fits.
    let per = 24;
    let fits = (2048 - fixed) / per;
    assert_eq!(refused_at, Some(fits as usize));
    let (bytes, cap) = m.state_over_cap().unwrap();
    assert_eq!((bytes, cap), (fixed + (fits + 1) * per, 2048));
    assert!(m.state_bytes() <= 2048);
    assert_eq!(warned_at, Some(((1024 - fixed) / per) as usize));
    // The row window's ring is fixed by the settings: refused before the
    // first row, naming the bytes and the limit.
    let w = EwmMomentsCfg {
        window_rows: Some(1000),
        max_state_bytes: Some(4096),
        ..forms()[0].1.clone()
    };
    let err = EwmMoments::new(w).unwrap_err();
    assert!(
        err.contains("over max_state_bytes (4096)") && err.contains("1000 window rows"),
        "{err}"
    );
}

/// A row of weight 0, as the model is handed one: on a time clock its step
/// decays the weight as `ew_cov`'s does, to the bit, and the variance and
/// the means do not move; on a row clock, which the stream steps no
/// weightless row on, it enters no lag buffer and no window, so the next
/// row's partner is the last row of weight.
#[test]
fn a_zero_weight_row_decays_as_ew_cov_and_counts_in_no_row_span() {
    let decay = Decay::Halflife(4.0);
    let mut ew = crate::EwCov::new(1);
    let mut m = EwmMoments::new(base(
        decay,
        vec![MomentTerm::of_column("c2", 2, MomentKind::Central, true)],
    ))
    .unwrap();
    let rows = [
        (1.0, 1.0, 1.0),
        (4.0, 0.5, 1.5),
        (9.0, 0.0, 1.5),
        (2.0, 2.0, 1.0),
        (5.0, 0.0, 2.0),
        (3.0, 1.0, 0.5),
    ];
    for (x, w, d) in rows {
        ew.update(&[x], decay.factor(d), w);
        m.update_row(&[x], d, w);
        assert_eq!(m.n_eff().to_bits(), ew.n_eff().to_bits());
        assert_eq!(m.central(2).to_bits(), ew.var(0).to_bits());
    }
    // On rows, the lag skips a row of weight 0.
    let t = MomentTerm {
        name: "lag".into(),
        powers: vec![(Leg::lagged(0, 1.0), 1)],
        kind: MomentKind::Raw,
        bias: true,
    };
    let mut m = EwmMoments::new(EwmMomentsCfg {
        window_rows: Some(2),
        ..base(Decay::Lam(1.0), vec![t])
    })
    .unwrap();
    m.update_row(&[1.0], 1.0, 1.0);
    m.update_row(&[2.0], 1.0, 1.0);
    m.update_row(&[100.0], 1.0, 0.0);
    m.update_row(&[3.0], 1.0, 1.0);
    // The window's two rows are 2 and 3, their partners 1 and 2.
    assert_eq!(m.read()[0], Ok(1.5));
}

/// The refusals part 4 adds, each naming its term or setting.
#[test]
fn the_refusals_name_the_problem() {
    let lagged = |k: f64| MomentTerm {
        name: "ac".into(),
        powers: vec![(Leg::of(0), 1), (Leg::lagged(0, k), 1)],
        kind: MomentKind::Central,
        bias: true,
    };
    let refuse = |c: EwmMomentsCfg| c.validate().unwrap_err();
    let joint = MomentTerm {
        kind: MomentKind::Cumulant,
        ..lagged(1.0)
    };
    assert!(
        refuse(base(Decay::Lam(0.9), vec![joint])).contains("\"ac\" asks for a joint cumulant")
    );
    assert!(refuse(base(Decay::Lam(0.9), vec![lagged(1.5)])).contains("not a whole number"));
    assert!(refuse(base(Decay::Lam(0.9), vec![lagged(-1.0)])).contains("at least 0"));
    let twice = MomentTerm {
        powers: vec![(Leg::of(0), 1), (Leg::of(0), 2)],
        ..lagged(1.0)
    };
    assert!(refuse(base(Decay::Lam(0.9), vec![twice])).contains("names a leg twice"));
    assert!(
        refuse(EwmMomentsCfg {
            window_clock: Some(5.0),
            ..base(Decay::Lam(0.9), vec![lagged(1.0)])
        })
        .contains("window_clock needs a clock")
    );
    assert!(
        refuse(EwmMomentsCfg {
            window_rows: Some(5),
            window_clock: Some(5.0),
            clocked: true,
            ..base(Decay::Lam(0.9), vec![lagged(1.0)])
        })
        .contains("both given")
    );
    // A fractional lag on a clock is a span, and builds.
    assert!(
        EwmMomentsCfg {
            clocked: true,
            ..base(Decay::Lam(0.9), vec![lagged(1.5)])
        }
        .validate()
        .is_ok()
    );
}
