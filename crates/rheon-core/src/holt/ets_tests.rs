//! Tests of `holt`'s fixed forms of seasonal exponential smoothing (task
//! 488): each form against its definition written out, the phase read from
//! the raw clock, the starts, both sides of every threshold, rows of weight
//! 0, and a save continued to the bit. statsmodels is the oracle in
//! `tests/test_second_opinion_ets.py`.

use super::*;
use crate::ClockValue;
use crate::oracle::gamma;

fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

const M: usize = 4;

/// A known start of the form: one target, the level's, trend's, season's
/// and damping's half-lives given, `m = 4` over 8 clock units, so on a
/// clock of 1 each bucket holds two rows.
fn known(season: HoltSeason, trend: bool, damped: bool) -> HoltCfg {
    let seasonal = season != HoltSeason::None;
    let (one, amp) = if season == HoltSeason::Mul {
        (1.0, 0.1)
    } else {
        (0.0, 0.5)
    };
    HoltCfg {
        n_targets: 1,
        level_half_life: 3.0,
        trend_half_life: 7.0,
        min_weight: 0.0,
        trend,
        season,
        season_periods: if seasonal { M } else { 0 },
        period: if seasonal { 8.0 } else { 0.0 },
        seasonal_half_life: if seasonal { 20.0 } else { f64::INFINITY },
        damping_half_life: if damped { 9.0 } else { f64::INFINITY },
        start: HoltStart::Known,
        initial_level: vec![10.0],
        initial_trend: if trend { vec![0.3] } else { Vec::new() },
        initial_seasonal: if seasonal {
            vec![(0..M).map(|p| one + amp * (p as f64 - 1.5)).collect()]
        } else {
            Vec::new()
        },
        max_state_bytes: None,
        horizons: Vec::new(),
    }
}

const FORMS: [(HoltSeason, bool, bool); 9] = [
    (HoltSeason::None, false, false),
    (HoltSeason::None, true, false),
    (HoltSeason::None, true, true),
    (HoltSeason::Add, false, false),
    (HoltSeason::Add, true, false),
    (HoltSeason::Add, true, true),
    (HoltSeason::Mul, false, false),
    (HoltSeason::Mul, true, false),
    (HoltSeason::Mul, true, true),
];

/// A stream with steps of 1, 2, 0.5 and 3, weights of 1, 0.5, 2.5 and 0,
/// and a null target, about a rising seasonal series.
fn stream(n: usize, seed: u64) -> Vec<(Option<f64>, f64, f64)> {
    let mut s = seed;
    let mut clock = 0.0;
    (0..n)
        .map(|i| {
            let d = match i {
                0 => 0.0,
                _ if i % 7 == 3 => 2.0,
                _ if i % 11 == 5 => 0.5,
                _ if i % 13 == 8 => 3.0,
                _ => 1.0,
            };
            clock += d;
            let y = 10.0
                + 0.3 * clock
                + (clock * std::f64::consts::TAU / 8.0).sin()
                + 0.2 * lcg(&mut s);
            let w = match i % 9 {
                2 => 0.5,
                5 => 2.5,
                7 => 0.0,
                _ => 1.0,
            };
            ((i % 10 != 6).then_some(y), d, w)
        })
        .collect()
}

/// The model written out from its definitions: each update a weighted mean
/// of the state carried over the step and what the row observes, the held
/// weight the one unit rows at the step saturate at, `1/(1 − λ)`; `φ_s`
/// from the platform's `powf`, not the model's `exp2`; the phase from the
/// sum of the steps. Returns each row's prediction and the final level,
/// trend and seasonal values.
fn written_out(cfg: &HoltCfg, rows: &[(Option<f64>, f64, f64)]) -> (Vec<f64>, f64, f64, Vec<f64>) {
    let half = |s: f64, h: f64| 0.5f64.powf(s / h);
    let seasonal = cfg.season != HoltSeason::None;
    let m = cfg.season_periods;
    let p_len = cfg.period;
    let mut l = cfg.initial_level[0];
    let mut b = cfg.initial_trend.first().copied().unwrap_or(0.0);
    let mut seas = cfg.initial_seasonal.first().cloned().unwrap_or_default();
    let mut at: Vec<Option<f64>> = vec![None; m];
    let (mut clock, mut origin, mut since) =
        (0.0, None, if seasonal { p_len / m as f64 } else { 1.0 });
    let mut preds = Vec::new();
    for &(y, d, w) in rows {
        let c = clock + d;
        if origin.is_none() && w > 0.0 {
            origin = Some(c);
        }
        let p = match origin {
            Some(o) if seasonal => {
                (((c - o) * m as f64 / p_len).floor() as i64).rem_euclid(m as i64) as usize
            }
            _ => 0,
        };
        clock = c;
        let s = since + d;
        let phi_s = if cfg.damping_half_life.is_finite() {
            let phi = half(1.0, cfg.damping_half_life);
            phi * (1.0 - phi.powf(s)) / (1.0 - phi)
        } else {
            s
        };
        let big_l = l + if cfg.trend { phi_s * b } else { 0.0 };
        preds.push(match cfg.season {
            HoltSeason::None => big_l,
            HoltSeason::Add => big_l + seas[p],
            HoltSeason::Mul => big_l * seas[p],
        });
        let Some(y) = y.filter(|_| w > 0.0) else {
            since = s;
            continue;
        };
        let mean = |old: f64, new: f64, lam: f64| {
            let held = lam / (1.0 - lam);
            (held * old + w * new) / (held + w)
        };
        let deseason = match cfg.season {
            HoltSeason::None => y,
            HoltSeason::Add => y - seas[p],
            HoltSeason::Mul => y / seas[p],
        };
        let level = mean(big_l, deseason, half(s, cfg.level_half_life));
        if cfg.trend {
            let carried = if cfg.damping_half_life.is_finite() {
                b * half(s, cfg.damping_half_life)
            } else {
                b
            };
            b = mean(carried, (level - l) / s, half(s, cfg.trend_half_life));
        }
        if seasonal {
            let gap = at[p].map_or(p_len, |t| clock - t);
            let observed = if cfg.season == HoltSeason::Add {
                y - big_l
            } else {
                y / big_l
            };
            seas[p] = mean(seas[p], observed, half(gap, cfg.seasonal_half_life));
            at[p] = Some(clock);
        }
        l = level;
        since = 0.0;
    }
    (preds, l, b, seas)
}

/// The relative distance of `a` from `b`, on the scale of `b` and at least
/// 1.
fn off(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs().max(1.0)
}

/// Every one of the nine forms, under the known start, on an uneven clock
/// with weights, a weight of 0, nulls and several rows a bucket, against
/// its definition written out another way. The bound: each row is about 40
/// roundings and four calls to the math library, and the level, trend and
/// seasonal values contract their errors, so `γ_{60·n}` over `n` rows
/// bounds the run with room (measured, Linux x86-64: at most 1.2e-15,
/// 9.1e-4 of the bound's 1.3e-12 at 200 rows).
#[test]
fn every_form_is_its_definition_written_out() {
    let rows = stream(200, 7);
    let bound = gamma(60 * rows.len());
    for (season, trend, damped) in FORMS {
        let cfg = known(season, trend, damped);
        let (want, l, b, seas) = written_out(&cfg, &rows);
        let mut m = Holt::new(cfg).unwrap();
        for (i, &(y, d, w)) in rows.iter().enumerate() {
            let got = m.step(&[], &[y], d, w).pred[0];
            assert!(
                off(got, want[i]) <= bound,
                "{season:?} trend {trend} damped {damped}, row {i}: {got} vs {}",
                want[i]
            );
        }
        let what = format!("{season:?} trend {trend} damped {damped}");
        assert!(off(m.level()[0], l) <= bound, "{what}: level");
        assert!(off(m.trend()[0], b) <= bound, "{what}: trend");
        for (g, w) in m.seasonal(0).iter().zip(&seas) {
            assert!(off(*g, *w) <= bound, "{what}: season {g} vs {w}");
        }
    }
}

/// At a step of one clock unit and a weight of 1 every gain is the fixed
/// one to the bit, and the update is statsmodels' `ExponentialSmoothing`'s,
/// operation for operation: `α·y − α·S + (1 − α)·L` with `α = 1 − λ`.
#[test]
fn a_unit_row_takes_the_fixed_gains_to_the_bit() {
    for (season, trend, damped) in FORMS {
        let cfg = known(season, trend, damped);
        let alpha = 1.0 - (-(1.0 / cfg.level_half_life)).exp2();
        let mut m = Holt::new(cfg.clone()).unwrap();
        // The first row is one step after the initial states: a season's
        // bucket, 2 clock units here, so step it with 2 first.
        m.since[0] = 1.0;
        let (l, b, s0) = (m.level()[0], m.trend()[0], m.seasonal(0).first().copied());
        let phi = if damped {
            (-(1.0 / 9.0f64)).exp2()
        } else {
            1.0
        };
        let big_l = if trend { l + b * phi } else { l };
        let y = 12.5;
        m.step(&[], &[Some(y)], 0.0, 1.0);
        let want = match season {
            HoltSeason::None => alpha * y + (1.0 - alpha) * big_l,
            HoltSeason::Add => alpha * y - alpha * s0.unwrap() + (1.0 - alpha) * big_l,
            HoltSeason::Mul => alpha * y / s0.unwrap() + (1.0 - alpha) * big_l,
        };
        assert_eq!(
            m.level()[0].to_bits(),
            want.to_bits(),
            "{season:?} {trend} {damped}"
        );
        if trend {
            let beta = 1.0 - (-(1.0 / 7.0f64)).exp2();
            let want_b = beta * (want - l) + (1.0 - beta) * (b * phi);
            assert_eq!(m.trend()[0].to_bits(), want_b.to_bits());
        }
    }
}

/// The phase is read from the row's raw clock, in integers on a temporal
/// clock: hourly rows over a day of 24 buckets, the steps the model decays
/// by capped at 2 hours so a gap shortens them, and every row learns the
/// bucket of its own hour, whatever the steps between.
#[test]
fn the_phase_is_the_raw_clocks_not_the_steps() {
    const H: i64 = 3_600_000_000_000;
    let cfg = HoltCfg {
        season: HoltSeason::Add,
        season_periods: 24,
        period: 86_400.0,
        seasonal_half_life: 86_400.0 * 3.0,
        initial_seasonal: vec![vec![0.0; 24]],
        ..known(HoltSeason::Add, true, false)
    };
    let mut m = Holt::new(cfg).unwrap();
    let start = 1_700_000_000_000_000_000i64 - 1_700_000_000_000_000_000i64 % (24 * H) + 5 * H;
    let mut last = start;
    let mut s = 3u64;
    for hour in (0..200).filter(|h| !(40..70).contains(h)) {
        let raw = start + hour * H;
        let d = (((raw - last) / H) as f64).min(2.0) * 3600.0;
        last = raw;
        let before = m.seasonal(0).to_vec();
        m.clock_next(ClockValue::Ns(raw));
        m.step(&[], &[Some(10.0 + lcg(&mut s))], d, 1.0);
        let moved: Vec<usize> = (0..24).filter(|&p| m.seasonal(0)[p] != before[p]).collect();
        assert_eq!(moved, vec![(hour % 24) as usize], "hour {hour}");
    }
}

/// The integer and the float paths of the phase, and both sides of a
/// bucket's edge: on an integer clock with `P = 10` and `m = 4` the edges
/// are at 2.5, 5 and 7.5 units, so 2 is phase 0 and 3 is phase 1 exactly;
/// a float clock reads them the same, and a period that is not a whole
/// number of an integer clock's units takes the float path.
#[test]
fn a_phase_on_each_side_of_a_buckets_edge() {
    let cfg = HoltCfg {
        period: 10.0,
        ..known(HoltSeason::Add, false, false)
    };
    let mut m = Holt::new(cfg.clone()).unwrap();
    m.origin = Some(ClockValue::I64(100));
    let ints: Vec<usize> = [100, 102, 103, 104, 105, 107, 108, 109, 110, 99, 90]
        .iter()
        .map(|&c| m.phase(ClockValue::I64(c)))
        .collect();
    assert_eq!(ints, [0, 0, 1, 1, 2, 2, 3, 3, 0, 3, 0]);
    m.origin = Some(ClockValue::F64(100.0));
    let floats: Vec<usize> = [100.0, 102.4, 102.5, 104.99, 105.0, 109.99, 110.0, 97.5]
        .iter()
        .map(|&c| m.phase(ClockValue::F64(c)))
        .collect();
    assert_eq!(floats, [0, 0, 1, 1, 2, 3, 0, 3]);
    let mut odd = Holt::new(HoltCfg { period: 2.5, ..cfg }).unwrap();
    odd.origin = Some(ClockValue::I64(0));
    let phases: Vec<usize> = (0..6).map(|c| odd.phase(ClockValue::I64(c))).collect();
    assert_eq!(
        phases,
        [0, 1, 3, 0, 2, 0],
        "P = 2.5 over 4 buckets of 0.625"
    );
}

/// A row of weight 0, and one whose weight is below the smallest normal
/// double, learns nothing: the level, trend, seasonal values and weights
/// stay, and only the clock it was handed passes.
#[test]
fn a_row_of_weight_zero_learns_nothing() {
    for w in [0.0, 1e-310] {
        for (season, trend, damped) in FORMS {
            let mut m = Holt::new(known(season, trend, damped)).unwrap();
            for (y, d, w) in stream(30, 1)
                .into_iter()
                .map(|(y, d, w)| (y, d, w.max(0.5)))
            {
                m.step(&[], &[y], d, w);
            }
            let before = m.clone();
            let step = m.step(&[], &[Some(-50.0)], 1.0, w);
            assert!(step.pred[0].is_finite(), "it is predicted");
            assert_eq!(m.level, before.level);
            assert_eq!(m.trend, before.trend);
            assert_eq!(m.season, before.season);
            assert_eq!(m.season_at, before.season_at);
            assert_eq!(m.w_level, before.w_level);
            assert_eq!(m.since[0], before.since[0] + 1.0, "its step is carried");
            assert!(!m.not_positive(), "and it is no refusal");
        }
    }
}

/// A multiplicative season divides by the target: a target of 0 is
/// predicted and not learned, and named; the smallest positive one is
/// learned. So is an update that would leave the level at 0 or below.
#[test]
fn a_multiplicative_target_of_zero_is_predicted_and_not_learned() {
    for y in [0.0, -1.0, f64::MIN_POSITIVE] {
        let mut m = Holt::new(known(HoltSeason::Mul, true, false)).unwrap();
        m.step(&[], &[Some(10.0)], 0.0, 1.0);
        let before = m.clone();
        let step = m.step(&[], &[Some(y)], 1.0, 1.0);
        assert!(step.pred[0].is_finite());
        if y > 0.0 {
            assert!(!m.not_positive(), "{y} is learned");
            assert_ne!(m.level, before.level);
        } else {
            assert!(m.not_positive(), "{y} is not learned");
            assert_eq!(m.level, before.level);
            assert_eq!(m.season, before.season);
            assert_eq!(m.since[0], 1.0);
        }
    }
    // A trend that carries the level below 0: the update is not taken.
    for (trend, taken) in [(-10.0, false), (-9.0, true)] {
        let mut m = Holt::new(HoltCfg {
            initial_trend: vec![trend],
            ..known(HoltSeason::Mul, true, false)
        })
        .unwrap();
        m.since[0] = 0.0;
        m.step(&[], &[Some(5.0)], 1.0, 1.0);
        assert_eq!(!m.not_positive(), taken, "trend {trend}");
        assert_eq!(m.level()[0] != 10.0, taken, "trend {trend}");
    }
}

/// The heuristic start holds exactly the rows its definition reads -- 10
/// without a season, else 5 seasons, raised where `m` is small -- and
/// predicts from the row after; a row it cannot hold (a null) is not
/// counted.
#[test]
fn the_heuristic_start_holds_its_rows_and_no_more() {
    let rows_for = |season: HoltSeason, m: usize| {
        HoltCfg {
            season,
            season_periods: m,
            ..HoltCfg::default()
        }
        .start_rows()
    };
    assert_eq!(rows_for(HoltSeason::None, 0), 10);
    assert_eq!(rows_for(HoltSeason::Add, 2), 12);
    assert_eq!(rows_for(HoltSeason::Add, 3), 15);
    assert_eq!(rows_for(HoltSeason::Mul, 4), 20);
    assert_eq!(rows_for(HoltSeason::Add, 12), 60);
    for (season, m) in [
        (HoltSeason::None, 0),
        (HoltSeason::Add, 3),
        (HoltSeason::Mul, 4),
    ] {
        let mut cfg = known(season, true, true);
        cfg.start = HoltStart::Heuristic;
        cfg.initial_level.clear();
        cfg.initial_trend.clear();
        cfg.initial_seasonal.clear();
        if season != HoltSeason::None {
            cfg.season_periods = m;
            cfg.period = m as f64;
        }
        let need = cfg.start_rows();
        let mut model = Holt::new(cfg).unwrap();
        let mut s = 9u64;
        let mut held = 0;
        let mut i = 0;
        while held < need {
            let y = (i % 5 != 4).then(|| 20.0 + 0.1 * i as f64 + lcg(&mut s));
            assert!(model.starting(), "row {i}: still holding");
            let pred = model
                .step(&[], &[y], if i == 0 { 0.0 } else { 1.0 }, 1.0)
                .pred[0];
            assert!(pred.is_nan(), "row {i}: no prediction while it holds");
            held += usize::from(y.is_some());
            i += 1;
        }
        assert!(!model.starting(), "{season:?}: fitted at its {need}-th row");
        let pred = model.step(&[], &[Some(20.0)], 1.0, 1.0).pred[0];
        assert!(pred.is_finite(), "{season:?}: the next row predicts");
    }
}

/// The heuristic's definition on series whose answer is known: a line
/// through the first 10 rows without a season, `1 + 2x` at `x = 1..10`;
/// with a season, a line plus a periodic term, which the centred moving
/// average over a season removes, so the seasonal values are the term's
/// offsets about their mean and the line is read off the average.
#[test]
fn the_heuristic_reads_a_line_and_a_season_exactly() {
    let y: Vec<f64> = (0..10).map(|i| 3.0 + 2.0 * i as f64).collect();
    let (l, b, s) = super::ets::heuristic(&y, &[0; 10], HoltSeason::None, 0);
    assert!(
        (l - 1.0).abs() < 1e-13 && (b - 2.0).abs() < 1e-13 && s.is_empty(),
        "{l} {b}"
    );
    for m in [3usize, 4] {
        let offs: Vec<f64> = (0..m).map(|p| [1.0, -2.0, 0.5, 0.5][p]).collect();
        let mean = offs.iter().sum::<f64>() / m as f64;
        let n = 5 * m.max(3);
        let y: Vec<f64> = (0..n).map(|i| 3.0 + 2.0 * i as f64 + offs[i % m]).collect();
        let phase: Vec<u32> = (0..n).map(|i| (i % m) as u32).collect();
        let (l, b, s) = super::ets::heuristic(&y, &phase, HoltSeason::Add, m);
        for (p, v) in s.iter().enumerate() {
            assert!((v - (offs[p] - mean)).abs() < 1e-12, "m {m} phase {p}: {v}");
        }
        // The average's first value sits at row `⌊m/2⌋` (x = 1 there), so
        // its line is `3 + 2(⌊m/2⌋ + x − 1) + mean`.
        let first = 3.0 + 2.0 * (m / 2) as f64 + mean;
        assert!((b - 2.0).abs() < 1e-12, "m {m}: trend {b}");
        assert!(
            (l - (first - 2.0)).abs() < 1e-11,
            "m {m}: level {l} vs {}",
            first - 2.0
        );
    }
}

/// A state saved part way -- the heuristic start holding, and fitted, and
/// the known start -- loads and goes on to the bit beside the model that
/// was not saved.
#[test]
fn a_saved_state_goes_on_to_the_bit() {
    let rows = stream(120, 4);
    for (season, trend, damped) in FORMS {
        for start in [HoltStart::Known, HoltStart::Heuristic] {
            let mut cfg = known(season, trend, damped);
            if start == HoltStart::Heuristic {
                cfg.start = start;
                cfg.initial_level.clear();
                cfg.initial_trend.clear();
                cfg.initial_seasonal.clear();
            }
            for cut in [5, 60] {
                let mut a = Holt::new(cfg.clone()).unwrap();
                for &(y, d, w) in &rows[..cut] {
                    a.step(&[], &[y], d, w);
                }
                let bytes = rmp_serde::to_vec_named(&a.state()).unwrap();
                let mut b = Holt::restore(&rmp_serde::from_slice(&bytes).unwrap()).unwrap();
                for &(y, d, w) in &rows[cut..] {
                    let (pa, pb) = (a.step(&[], &[y], d, w), b.step(&[], &[y], d, w));
                    assert_eq!(
                        pa.pred[0].to_bits(),
                        pb.pred[0].to_bits(),
                        "{season:?} {start:?} {cut}"
                    );
                }
                assert_eq!(
                    rmp_serde::to_vec_named(&a.state()).unwrap(),
                    rmp_serde::to_vec_named(&b.state()).unwrap()
                );
            }
        }
    }
}

/// A state whose seasonal values are not the config's shape is refused.
#[test]
fn a_season_of_the_wrong_shape_is_refused() {
    let m = Holt::new(known(HoltSeason::Add, true, false)).unwrap();
    for cut in ["season", "season_at"] {
        let mut s = m.state();
        let ModelState::Holt(inner) = &mut s.model else {
            unreachable!()
        };
        match cut {
            "season" => inner.season[0].pop(),
            _ => inner.season_at[0].pop(),
        };
        match Holt::restore(&s) {
            Err(StateError::Invalid(e)) => assert!(e.contains("wrong shape"), "{e}"),
            other => panic!("{cut}: {other:?}"),
        }
    }
}

/// Each refusal, beside the config it is the edge of.
#[test]
fn each_form_is_refused_where_it_has_no_definition() {
    let ok = |c: HoltCfg| Holt::new(c).map(|_| ()).map_err(|e| e.to_string());
    let base = known(HoltSeason::Add, true, false);
    // The warm start has no season and no damping.
    let warm = HoltCfg {
        start: HoltStart::Warm,
        initial_level: Vec::new(),
        initial_trend: Vec::new(),
        initial_seasonal: Vec::new(),
        ..base.clone()
    };
    assert!(ok(warm.clone()).unwrap_err().contains("no warm start"));
    assert!(
        ok(HoltCfg {
            season: HoltSeason::None,
            season_periods: 0,
            period: 0.0,
            seasonal_half_life: f64::INFINITY,
            ..warm.clone()
        })
        .is_ok()
    );
    // A season of 1 value, and of 2.
    let two = HoltCfg {
        season_periods: 2,
        initial_seasonal: vec![vec![0.0, 0.0]],
        ..base.clone()
    };
    assert!(ok(two.clone()).is_ok());
    assert!(
        ok(HoltCfg {
            season_periods: 1,
            initial_seasonal: vec![vec![0.0]],
            ..two
        })
        .unwrap_err()
        .contains("at least 2")
    );
    // Damping without a trend.
    let flat = HoltCfg {
        trend: false,
        initial_trend: Vec::new(),
        ..base.clone()
    };
    assert!(ok(flat.clone()).is_ok());
    assert!(
        ok(HoltCfg {
            damping_half_life: 9.0,
            ..flat
        })
        .unwrap_err()
        .contains("damping_half_life")
    );
    // An infinite half-life is a defined recursion, a gain of 0: allowed.
    assert!(
        ok(HoltCfg {
            level_half_life: f64::INFINITY,
            seasonal_half_life: f64::INFINITY,
            damping_half_life: f64::INFINITY,
            ..base.clone()
        })
        .is_ok()
    );
    // A known start's states of the wrong shape, and a factor of 0.
    assert!(
        ok(HoltCfg {
            initial_seasonal: vec![vec![0.0; 3]],
            ..base.clone()
        })
        .unwrap_err()
        .contains("initial_seasonal")
    );
    let mul = known(HoltSeason::Mul, true, false);
    assert!(ok(mul.clone()).is_ok());
    let mut zero = mul;
    zero.initial_seasonal[0][2] = 0.0;
    assert!(ok(zero).unwrap_err().contains("above 0"));
}

/// The state's size against `max_state_bytes`, on both sides: a cap of
/// exactly the bytes the season and the start's held rows take builds, a
/// byte less is refused with a message giving the bytes needed.
#[test]
fn the_state_is_held_to_its_cap_on_both_sides() {
    let cfg = HoltCfg {
        n_targets: 3,
        season: HoltSeason::Add,
        season_periods: 8760,
        period: 8760.0,
        seasonal_half_life: 1e5,
        start: HoltStart::Heuristic,
        ..HoltCfg::default()
    };
    let bytes = cfg.state_bytes();
    assert_eq!(bytes, 64 + 3 * (81 + 16 * 8760 + 36 * 5 * 8760));
    let refused = Holt::new(cfg.clone()).map(|_| ()).unwrap_err();
    assert!(
        refused.contains(&format!("Set max_state_bytes to {bytes} or more")),
        "{refused}"
    );
    assert!(
        Holt::new(HoltCfg {
            max_state_bytes: Some(bytes),
            ..cfg.clone()
        })
        .is_ok()
    );
    assert!(
        Holt::new(HoltCfg {
            max_state_bytes: Some(bytes - 1),
            ..cfg
        })
        .is_err()
    );
    // Under the default cap, a season of a week's hours with a known start.
    let week = HoltCfg {
        season_periods: 168,
        period: 168.0,
        start: HoltStart::Known,
        initial_level: vec![0.0],
        initial_trend: vec![0.0],
        initial_seasonal: vec![vec![0.0; 168]],
        n_targets: 1,
        seasonal_half_life: 1e3,
        season: HoltSeason::Add,
        ..HoltCfg::default()
    };
    assert!(week.state_bytes() < HOLT_MAX_STATE_BYTES);
    assert!(Holt::new(week).is_ok());
}

/// A row at the clock of the last learned row teaches nothing: its gains,
/// `1 − λ⁰`, are 0, and a move over no clock has no slope.
#[test]
fn a_row_at_the_last_rows_clock_teaches_nothing() {
    for (season, trend, damped) in FORMS {
        let mut m = Holt::new(known(season, trend, damped)).unwrap();
        m.step(&[], &[Some(11.0)], 0.0, 1.0);
        let before = (m.level.clone(), m.trend.clone(), m.season.clone());
        m.step(&[], &[Some(40.0)], 0.0, 1.0);
        assert_eq!((m.level.clone(), m.trend.clone(), m.season.clone()), before);
        assert!(m.level[0].is_finite() && m.trend[0].is_finite());
    }
}
