//! The windows: `window_rows`, `window_clock` and `lm_window` (docs/PLAN.md
//! task 499, part 3), against the papers' estimators on the returns each
//! window holds.

use super::paper;
use super::tests::sv_day;
use super::*;

type Row = (f64, Vec<Option<f64>>, f64);

fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64 + 0.5) / (1u64 << 53) as f64
}

fn cfg(n: usize) -> RealizedCfg {
    RealizedCfg {
        n_targets: n,
        decay: Decay::Halflife(f64::INFINITY),
        min_weight: 0.0,
        window_rows: None,
        window_clock: None,
        closed: WindowClosed::Right,
        lm_window: None,
        lm_beta: DEFAULT_LM_BETA,
        max_state_bytes: None,
    }
}

fn run(m: &mut Realized, rows: &[Row]) -> Vec<Step> {
    rows.iter()
        .map(|(d, y, w)| m.step(&[], y, *d, *w))
        .collect()
}

fn unit_rows(r: &[f64]) -> Vec<Row> {
    r.iter().map(|&x| (1.0, vec![Some(x)], 1.0)).collect()
}

fn ulps(a: f64, b: f64) -> f64 {
    if a.is_nan() && b.is_nan() {
        return 0.0;
    }
    let ulp = f64::from_bits(b.abs().to_bits() + 1) - b.abs();
    (a - b).abs() / ulp
}

fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// The gaps of a row's outputs to the papers' measures `p`: RV, BV, TV and
/// TQ in ulps, `Z` in units of `cond · ε`.
fn gaps(o: &[f64], p: &paper::Measures) -> [f64; 5] {
    let cond = (p.rv + p.bv) / (p.rv - p.bv).abs();
    [
        ulps(o[0], p.rv),
        ulps(o[1], p.bv),
        ulps(o[2], p.tv),
        ulps(o[3], p.tq),
        (o[4] - p.z).abs() / (cond * f64::EPSILON * p.z.abs().max(1.0)),
    ]
}

fn hold(worst: &mut [f64; 5], g: [f64; 5]) {
    for (w, v) in worst.iter_mut().zip(g) {
        *w = w.max(v);
    }
}

/// A window of `W` rows reports, at row `i`, the papers' measures of the
/// `W` returns before it: `W − 1` products and `W − 2` triples, the first
/// return's product with the one before the window left out. Its sums take
/// each leaving term back, compensated, so the gap is the rounding of the
/// window's terms, not of the 23,400 rows the stream ran: measured (ulps)
/// RV 0, BV 0, TV 10, TQ 21, `Z` 2.1 `cond · ε` (on the clock window too).
#[test]
fn a_window_of_rows_is_the_papers_measures_of_its_returns() {
    let mut worst = [0.0f64; 5];
    for (n, seed, w) in [(23_400, 21, 100), (23_400, 22, 1_000), (2_340, 23, 390)] {
        let r = sv_day(n, seed, true);
        let mut c = cfg(1);
        c.window_rows = Some(w);
        let mut m = Realized::new(c).unwrap();
        let steps = run(&mut m, &unit_rows(&r));
        for i in (3..n).step_by(89).chain([w - 1, w, w + 1, n - 1]) {
            let from = i.saturating_sub(w);
            hold(
                &mut worst,
                gaps(&steps[i].pred, &paper::measures(&r[from..i])),
            );
            assert_eq!(steps[i].n_eff, (i - from) as f64, "row {i}");
        }
        assert_eq!(m.window_len(), w);
        let mut own = Vec::new();
        m.target_n_eff_into(&mut own);
        assert_eq!(own, vec![w as f64]);
    }
    let bound = [4.0, 4.0, 16.0, 32.0, 8.0];
    for (k, (w, b)) in worst.iter().zip(bound).enumerate() {
        assert!(w <= &b, "output {k}: {w} ulps, bound {b} ({worst:?})");
    }
}

/// The rows of a window of `D` clock units, on an irregular clock whose
/// steps are 64ths, so that many rows are exactly `D` old: under `closed =
/// "right"` a row exactly `D` old has left, under `"both"` it is kept. At
/// row `i` the window is that of the row before, and the outputs are the
/// papers' measures of its returns, on both sides of the edge.
#[test]
fn a_window_of_clock_units_holds_the_rows_of_its_span_on_either_edge() {
    let n = 3_000;
    let r = sv_day(n, 31, true);
    let mut s = 4;
    let d: Vec<f64> = (0..n)
        .map(|_| (1.0 + (lcg(&mut s) * 128.0).floor()) / 64.0)
        .collect();
    let t: Vec<f64> = d
        .iter()
        .scan(0.0, |acc, x| {
            *acc += x;
            Some(*acc)
        })
        .collect();
    let span = 40.0;
    let mut ties = 0;
    let mut worst = [0.0f64; 5];
    for closed in [WindowClosed::Right, WindowClosed::Both] {
        let mut c = cfg(1);
        c.window_clock = Some(span);
        c.closed = closed;
        let mut m = Realized::new(c).unwrap();
        let rows: Vec<Row> = (0..n).map(|i| (d[i], vec![Some(r[i])], 1.0)).collect();
        let steps = run(&mut m, &rows);
        for i in 1..n {
            let last = t[i - 1];
            let from = (0..i)
                .find(|&j| {
                    let age = last - t[j];
                    age < span || (closed == WindowClosed::Both && age == span)
                })
                .unwrap();
            ties += usize::from(last - t[from.saturating_sub(1)] == span);
            if i % 7 == 0 || from > 0 && last - t[from - 1] == span {
                hold(
                    &mut worst,
                    gaps(&steps[i].pred, &paper::measures(&r[from..i])),
                );
                assert_eq!(steps[i].n_eff, (i - from) as f64, "row {i}");
            }
        }
    }
    assert!(ties > 20, "the edge is met exactly: {ties}");
    let bound = [4.0, 4.0, 16.0, 32.0, 8.0];
    for (k, (w, b)) in worst.iter().zip(bound).enumerate() {
        assert!(w <= &b, "output {k}: {w} ulps, bound {b} ({worst:?})");
    }
}

/// Unequal weights and a target absent on some rows, in a window of rows:
/// the window's weight is its rows', each target's `n` its returns', and
/// the measures those of the definition on the target's returns in the
/// window, a product reaching across a row it is absent on but never past
/// the window's first row.
#[test]
fn a_window_weighs_its_rows_and_skips_an_absent_return() {
    let n = 2_000;
    let (r0, r1) = (sv_day(n, 41, true), sv_day(n, 42, false));
    let mut s = 9;
    let w: Vec<f64> = (0..n).map(|_| 0.5 + lcg(&mut s)).collect();
    let present = |i: usize| i % 4 != 2;
    let rows: Vec<Row> = (0..n)
        .map(|i| (1.0, vec![Some(r0[i]), present(i).then_some(r1[i])], w[i]))
        .collect();
    let mut c = cfg(2);
    c.window_rows = Some(150);
    let mut m = Realized::new(c).unwrap();
    let steps = run(&mut m, &rows);
    for i in (160..n).step_by(37) {
        let from = i - 150;
        let defn = |rs: &[f64], keep: &dyn Fn(usize) -> bool| {
            let held: Vec<(f64, f64, f64)> = (from..i)
                .filter(|&j| keep(j))
                .map(|j| (j as f64, rs[j], w[j]))
                .collect();
            // A window's `n` is its count of returns.
            let count = held.len() as f64;
            paper::weighted_counted(&held, f64::INFINITY, Some(count))
        };
        let (p0, n0, _) = defn(&r0, &|_| true);
        let (p1, n1, _) = defn(&r1, &present);
        let o = &steps[i].pred;
        for (k, g) in gaps(&o[..7], &p0).into_iter().enumerate() {
            assert!(g <= 32.0, "row {i} target 0 output {k}: {g}");
        }
        for (k, g) in gaps(&o[7..], &p1).into_iter().enumerate() {
            assert!(g <= 32.0, "row {i} target 1 output {k}: {g}");
        }
        let in_window = paper::sum(w[from..i].iter().copied());
        assert!(ulps(steps[i].n_eff, in_window) <= 4.0, "row {i}");
        let _ = (n0, n1);
    }
    let mut own = Vec::new();
    m.target_n_eff_into(&mut own);
    let last = n - 150..n;
    let n1: f64 = paper::sum(last.clone().filter(|&j| present(j)).map(|j| w[j]));
    assert!(ulps(own[1], n1) <= 4.0 && ulps(own[0], paper::sum(w[last].iter().copied())) <= 4.0);
}

/// Rows of weight 0, carrying returns that are not numbers or are huge,
/// take no place in a window of rows and none in a clock window at their
/// own clock: the real rows report what they report without them, to the
/// bit.
#[test]
fn a_row_of_weight_zero_takes_no_place_in_a_window() {
    let r = sv_day(1_500, 51, true);
    let clean = unit_rows(&r);
    let mut with = Vec::new();
    let mut at = Vec::new();
    let junk = [f64::NAN, 1e300, f64::NEG_INFINITY, 0.5];
    for (i, row) in clean.iter().enumerate() {
        if i % 2 == 0 {
            with.push((0.0, vec![Some(junk[i % 4])], 0.0));
        }
        with.push(row.clone());
        at.push(with.len() - 1);
    }
    for clock in [false, true] {
        let mut c = cfg(1);
        if clock {
            c.window_clock = Some(120.0);
        } else {
            c.window_rows = Some(120);
        }
        c.lm_window = Some(50);
        let a = run(&mut Realized::new(c.clone()).unwrap(), &clean);
        let b = run(&mut Realized::new(c).unwrap(), &with);
        for (i, &j) in at.iter().enumerate() {
            assert_eq!(bits(&a[i].pred), bits(&b[j].pred), "clock {clock} row {i}");
            assert_eq!(a[i].n_eff.to_bits(), b[j].n_eff.to_bits());
        }
    }
}

/// Lee and Mykland's own window: `σ_i` from the `K − 1` returns before the
/// row, whatever the estimators' window or decay, to the rounding of the
/// paper's `L_i` (measured 2 ulps); before `K − 1` returns, from those
/// there are.
#[test]
fn lm_window_is_the_papers_statistic() {
    let n = 5_000;
    let r = sv_day(n, 61, true);
    for (k, decay, rows) in [
        (100, f64::INFINITY, None),
        (270, 200.0, None),
        (30, f64::INFINITY, Some(500)),
    ] {
        let mut c = cfg(1);
        c.decay = Decay::Halflife(decay);
        c.window_rows = rows;
        c.lm_window = Some(k);
        let mut m = Realized::new(c).unwrap();
        let steps = run(&mut m, &unit_rows(&r));
        let l = paper::lee_mykland(&r, k);
        for i in k - 1..n {
            assert!(ulps(steps[i].pred[5], l[i]) <= 8.0, "K {k} row {i}");
        }
        // Before K − 1 returns: the mean of the products there are.
        let early = paper::lee_mykland(&r[..k / 2 + 1], k / 2 + 1)[k / 2];
        assert!(ulps(steps[k / 2].pred[5], early) <= 8.0);
    }
}

/// The flag is the paper's test at confidence `β`: `β* = −ln(−ln β)`, 4.6001
/// at 0.99, against `(|L| − C_n)/S_n`; a statistic just past the critical
/// value is flagged and one just short of it is not, and the planted jumps
/// of a day are flagged where its quiet returns are not.
#[test]
fn the_flag_is_the_papers_test_on_either_side_of_its_critical_value() {
    let beta: f64 = 0.99;
    let critical = -(-beta.ln()).ln();
    assert!((critical - 4.600_149_226_776_9).abs() < 1e-12, "{critical}");
    for n in [78.0, 390.0, 23_400.0] {
        let c = (2.0 / std::f64::consts::PI).sqrt();
        let g = (2.0 * f64::ln(n)).sqrt();
        let c_n = g / c - (std::f64::consts::PI.ln() + n.ln().ln()) / (2.0 * c * g);
        let s_n = 1.0 / (c * g);
        let edge = c_n + s_n * critical;
        assert_eq!(series::lm_jump(edge * (1.0 + 1e-9), n, beta), 1.0);
        assert_eq!(series::lm_jump(-edge * (1.0 + 1e-9), n, beta), 1.0);
        assert_eq!(series::lm_jump(edge * (1.0 - 1e-9), n, beta), 0.0);
    }
    assert!(series::lm_jump(f64::NAN, 100.0, beta).is_nan());
    assert!(series::lm_jump(10.0, 1.0, beta).is_nan());
    // A day of 2,340 returns, each window of 390: its five jumps flagged.
    let n = 2_340;
    let r = sv_day(n, 62, true);
    let mut c = cfg(1);
    c.window_rows = Some(390);
    c.lm_window = Some(200);
    let steps = run(&mut Realized::new(c).unwrap(), &unit_rows(&r));
    let jumps: Vec<usize> = [0.1, 0.3, 0.5, 0.7, 0.9]
        .iter()
        .map(|a| (a * n as f64) as usize)
        .filter(|&j| j >= 200)
        .collect();
    for &j in &jumps {
        assert_eq!(steps[j].pred[6], 1.0, "the jump at {j}");
    }
    let quiet = (200..n)
        .filter(|i| !jumps.contains(i))
        .filter(|&i| steps[i].pred[6] == 1.0);
    assert!(quiet.count() <= 5, "few false flags at 99%");
}

/// A save at any row, the window empty, filling, full or emptied, goes on to
/// the bit, on either window and with `lm_window`; a clock window's stamps
/// handed by the caller included.
#[test]
fn a_saved_window_goes_on_to_the_bit() {
    let n = 700;
    let (r0, r1) = (sv_day(n, 71, true), sv_day(n, 72, true));
    let mut s = 5;
    let rows: Vec<Row> = (0..n)
        .map(|i| {
            let d = if i == 300 {
                500.0
            } else {
                (lcg(&mut s) * 4.0).floor() / 2.0
            };
            (
                d,
                vec![Some(r0[i]), (i % 5 != 1).then_some(r1[i])],
                0.5 + lcg(&mut s),
            )
        })
        .collect();
    for clock in [false, true] {
        let mut c = cfg(2);
        if clock {
            c.window_clock = Some(60.0);
        } else {
            c.window_rows = Some(80);
        }
        c.lm_window = Some(25);
        let stamp =
            |i: usize| Stamp::Ns(rows[..=i].iter().map(|r| (r.0 * 1e9) as i128).sum::<i128>());
        for cut in [0, 1, 2, 3, 40, 80, 81, 299, 301, 520] {
            let mut m = Realized::new(c.clone()).unwrap();
            for (i, (d, y, w)) in rows[..cut].iter().enumerate() {
                m.stamp_next(stamp(i));
                m.step(&[], y, *d, *w);
            }
            let bytes = rmp_serde::to_vec(&m.state()).unwrap();
            let mut back = Realized::restore(&rmp_serde::from_slice(&bytes).unwrap()).unwrap();
            assert_eq!(m, back, "clock {clock} cut {cut}");
            for (i, (d, y, w)) in rows.iter().enumerate().skip(cut) {
                m.stamp_next(stamp(i));
                back.stamp_next(stamp(i));
                let (a, b) = (m.step(&[], y, *d, *w), back.step(&[], y, *d, *w));
                assert_eq!(
                    bits(&a.pred),
                    bits(&b.pred),
                    "clock {clock} cut {cut} row {i}"
                );
            }
            assert_eq!(m, back);
        }
    }
}

/// The state's bytes: a window of rows is checked when the model is built,
/// and refused past the cap naming the largest window that fits, which is
/// taken where one more row is not; it then holds no more than that.
#[test]
fn a_window_of_rows_is_checked_against_the_cap_when_built() {
    let mut c = cfg(2);
    c.lm_window = Some(50);
    c.window_rows = Some(1_000_000);
    let e = Realized::new(c.clone()).unwrap_err();
    let fits: usize = e
        .split("largest window_rows that fits is ")
        .nth(1)
        .and_then(|s| s.split('.').next())
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| panic!("{e}"));
    c.window_rows = Some(fits);
    let mut m = Realized::new(c.clone()).unwrap();
    assert!(Realized::state_bytes(2, Some(50), Some(fits), false) <= 256 * 1024);
    c.window_rows = Some(fits + 1);
    assert!(Realized::new(c).is_err());
    let r = sv_day(fits + 500, 81, false);
    let rows: Vec<Row> = r.iter().map(|&x| (1.0, vec![Some(x); 2], 1.0)).collect();
    run(&mut m, &rows);
    assert_eq!(m.window_len(), fits);
    assert_eq!(
        m.bytes_with(0),
        Realized::state_bytes(2, Some(50), Some(fits), false)
    );
}

/// A window of clock units holds as many rows as its span has: a burst is
/// warned of once, past half of `max_state_bytes`, and the row that would
/// pass the cap is not learned, nor any after it, and is reported.
#[test]
fn a_burst_in_a_clock_window_is_warned_of_at_half_the_cap_and_stopped_at_it() {
    let mut c = cfg(1);
    c.window_clock = Some(1e6);
    c.max_state_bytes = Some(20_000);
    let mut m = Realized::new(c).unwrap();
    let row = Realized::window_row_bytes(1, true);
    let base = Realized::state_bytes(1, None, Some(0), true);
    let (half, cap) = ((10_000 - base) / row + 1, (20_000 - base) / row);
    let r = sv_day(1_000, 91, false);
    let mut warned_at = None;
    for (i, &x) in r.iter().enumerate() {
        m.step(&[], &[Some(x)], 0.001, 1.0);
        if let Some((bytes, of)) = m.take_state_warning() {
            assert!(warned_at.is_none(), "said once");
            assert!(2 * bytes > of && of == 20_000);
            warned_at = Some(i + 1);
        }
        if (i as u64) < cap {
            assert!(m.state_over_cap().is_none(), "row {i}");
        }
    }
    assert_eq!(warned_at, Some(half as usize));
    let (bytes, of) = m.state_over_cap().expect("refused at the cap");
    assert!(bytes > of && of == 20_000);
    assert_eq!(m.window_len() as u64, cap, "nothing learned past the cap");
}

/// What the settings refuse about the windows, each naming the setting.
#[test]
fn the_window_settings_are_checked() {
    let refuse = |f: &dyn Fn(&mut RealizedCfg), says: &str| {
        let mut c = cfg(1);
        f(&mut c);
        let e = Realized::new(c).unwrap_err();
        assert!(e.contains(says), "{e}");
    };
    refuse(
        &|c| c.window_rows = Some(0),
        "window_rows must be at least 1",
    );
    refuse(
        &|c| c.window_clock = Some(0.0),
        "finite number of clock units",
    );
    refuse(
        &|c| c.window_clock = Some(f64::INFINITY),
        "finite number of clock units",
    );
    refuse(
        &|c| {
            c.window_rows = Some(5);
            c.window_clock = Some(5.0);
        },
        "two windows",
    );
    refuse(
        &|c| c.closed = WindowClosed::Both,
        "there is no clock window",
    );
    refuse(
        &|c| {
            c.window_rows = Some(5);
            c.decay = Decay::Halflife(10.0);
        },
        "takes no decay",
    );
    refuse(&|c| c.lm_window = Some(2), "lm_window must be at least 3");
    refuse(&|c| c.lm_beta = 1.0, "lm_beta must be above 0 and below 1");
    refuse(&|c| c.lm_beta = 0.0, "lm_beta must be above 0 and below 1");
    let mut c = cfg(1);
    c.lm_window = Some(3);
    c.lm_beta = 0.5;
    c.window_rows = Some(1);
    assert!(Realized::new(c).is_ok());
}
