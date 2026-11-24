//! The window's clock and the nodes it holds (HAWKES-7, the review of
//! 2026-11-23, and the finding beside it): a gap after a long quiet stretch
//! keeps its digits, and a node with no excitation over the observation
//! leaves the fit unidentified, so nothing is reported.

use super::*;
use crate::oracle;

fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64 + 0.5) / (1u64 << 53) as f64
}

fn cfg(n: usize, betas: &[f64]) -> HawkesCfg {
    HawkesCfg {
        n_targets: n,
        kernel_decays: betas.to_vec(),
        decay: Decay::Halflife(f64::INFINITY),
        min_weight: 0.0,
        window_events: None,
        window_clock: None,
        closed: WindowClosed::Right,
        max_state_bytes: None,
    }
}

/// `max |a − b| / max |b|`.
fn rel(a: &[f64], b: &[f64]) -> f64 {
    let scale = b
        .iter()
        .fold(0.0f64, |m, v| m.max(v.abs()))
        .max(f64::MIN_POSITIVE);
    a.iter()
        .zip(b)
        .fold(0.0f64, |m, (x, y)| m.max((x - y).abs()))
        / scale
}

fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

// ---- a gap after a long quiet stretch ----

/// A window of clock units long enough to hold the rows on both sides of a
/// quiet stretch of `1e16`: the rows before it leave at a later row, and the
/// newer block, formed into the older then, holds the rows after the
/// stretch. Their gaps of 0.25 to 0.75 are the fit's whole content (their
/// kernels are `e^(−2·6e15)`, nothing, at the read), and a time counted
/// from the block's start, `1e16 + 2.25`, rounds them to 0 or 2: `b` was
/// formed afresh from the wrong gaps. The definition's system over the
/// same rows, on the clock moved to the first row after the stretch, is
/// the reference.
#[test]
fn a_clock_window_across_a_quiet_stretch_keeps_the_gaps_after_it() {
    let span = 1.5e16;
    let mut m = Hawkes::new(HawkesCfg {
        window_clock: Some(span),
        ..cfg(1, &[2.0])
    })
    .unwrap();
    let steps = [0.0, 1.0, 1.0, 1e16, 0.25, 0.5, 0.25, 0.75, 6e15];
    for (i, d) in steps.iter().enumerate() {
        let y = if i + 1 < steps.len() { Some(1.0) } else { None };
        m.step(&[], &[y], *d, 1.0);
    }
    // The rows at 0, 1 and 2 have left; the five after the stretch stay.
    assert_eq!(m.window_rows(), 5);
    // Moved to the first row after the stretch, every clock is exact.
    let window: Vec<(f64, Vec<f64>)> = [0.0, 0.25, 0.75, 1.0, 1.75]
        .iter()
        .map(|t| (*t, vec![1.0]))
        .collect();
    let end = 1.75 + 6e15;
    let (g_want, b_want) = oracle::hawkes_system(&window, &[2.0], 0.0, end - span, end);
    let (g, b) = m.system();
    let err = rel(&g, &g_want).max(rel(&b, &b_want));
    assert!(
        err < 1e-12,
        "G/b {err:e}: {g:?} against {g_want:?}, {b:?} against {b_want:?}"
    );
}

/// The other side: a quiet stretch a double holds exactly, `2^30` on a clock
/// of quarters, moves no bit of what the rows after it report against a
/// stretch twice as long, once the row the window dropped last is past it.
/// Every time the window holds is exact either way, so this held before
/// the window kept its gaps and holds after.
#[test]
fn a_quiet_stretch_a_double_holds_exactly_moves_no_bit() {
    for window_events in [3usize, 5, 8] {
        let run = |quiet: f64| {
            let mut m = Hawkes::new(HawkesCfg {
                window_events: Some(window_events),
                ..cfg(2, &[0.5])
            })
            .unwrap();
            let mut s = 41u64;
            let mut out = Vec::new();
            // The rows with an event from the stretch on: once the window
            // has dropped one of them, it holds nothing from before.
            let mut after = 0;
            for i in 0..80 {
                let d = match i {
                    0 => 0.0,
                    20 => quiet,
                    _ => 0.25 * (1.0 + (lcg(&mut s) * 6.0).floor()),
                };
                let y = [
                    (lcg(&mut s) < 0.6).then_some(1.0),
                    (lcg(&mut s) < 0.5).then_some(2.0),
                ];
                if i >= 20 && y.iter().any(Option::is_some) {
                    after += 1;
                }
                let step = m.step(&[], &y, d, 1.0);
                out.push((
                    step.pred,
                    m.coefficients().concat(),
                    after > window_events + 1,
                ));
            }
            out
        };
        let (a, b) = (run(2f64.powi(30)), run(2f64.powi(31)));
        let mut compared = 0;
        for (i, (x, y)) in a.iter().zip(&b).enumerate() {
            if !x.2 {
                continue;
            }
            assert_eq!(
                bits(&x.0),
                bits(&y.0),
                "{window_events} rows, row {i}: intensities"
            );
            assert_eq!(
                bits(&x.1),
                bits(&y.1),
                "{window_events} rows, row {i}: coefficients"
            );
            compared += usize::from(x.1.iter().all(|v| v.is_finite()));
        }
        assert!(
            compared > 30,
            "{window_events} rows: {compared} fits compared"
        );
    }
}

// ---- a node with no excitation over the observation ----

/// The rows a window of `cap` rows with an event holds after row `i`, from
/// the definition: (clock, marks) of each, oldest first.
fn held_after(line: &[(f64, Vec<f64>)], i: usize, cap: usize) -> Vec<(f64, Vec<f64>)> {
    let mut kept: Vec<(f64, Vec<f64>)> = line[..=i]
        .iter()
        .filter(|(_, m)| m.iter().any(|v| *v > 0.0))
        .cloned()
        .collect();
    let drop = kept.len().saturating_sub(cap);
    kept.drain(..drop);
    kept
}

/// Whether every node has an event among `rows` at a clock before `at`:
/// else its excitation is 0 over the whole observation, its column of `G`
/// is 0, and its kernels' parameters are not identified.
fn every_node_excited(rows: &[(f64, Vec<f64>)], n: usize, at: f64) -> bool {
    (0..n).all(|j| rows.iter().any(|(t, m)| m[j] > 0.0 && *t < at))
}

/// A stream of two nodes on an irregular clock, with fractional marks, in
/// which the second node falls silent for stretches longer than a window
/// of three rows with an event.
fn silent_stretches(rows: usize, seed: u64) -> Vec<(f64, Vec<Option<f64>>)> {
    let mut s = seed;
    (0..rows)
        .map(|i| {
            let d = if i == 0 {
                0.0
            } else {
                -lcg(&mut s).ln() * 0.37
            };
            let quiet = (i / 9) % 3 == 1;
            let a = (lcg(&mut s) < 0.7).then(|| 0.3 + 1.7 * lcg(&mut s));
            let b = (!quiet && lcg(&mut s) < 0.5).then(|| 0.2 + 2.3 * lcg(&mut s));
            (d, vec![a, b])
        })
        .collect()
}

/// Where a window holds no event of a node before the read, the fit is not
/// identified: the intensities the row reports, and the coefficients read
/// after its events, are not numbers, and the row says why
/// ([`Hawkes::not_identified`]). The subtractions that took the node's
/// events back left rounding dust in its sums, which the scaling to a unit
/// diagonal hid from the pivot test, so a fit of the dust was reported;
/// the dust is now cleared as the node's last event leaves, and its
/// column of `G` is 0 exactly. Where every node is excited the fit is
/// reported, as before.
#[test]
fn a_window_without_a_nodes_events_reports_no_fit() {
    let cap = 3;
    let mut withheld = (0, 0);
    let mut reported = 0;
    for seed in [3u64, 7, 11, 19] {
        let rows = silent_stretches(400, seed);
        let mut t = 0.0;
        let line: Vec<(f64, Vec<f64>)> = rows
            .iter()
            .map(|(d, y)| {
                t += d;
                (t, y.iter().map(|v| Hawkes::mark(*v)).collect())
            })
            .collect();
        let mut m = Hawkes::new(HawkesCfg {
            window_events: Some(cap),
            ..cfg(2, &[1.0, 4.0])
        })
        .unwrap();
        for (i, (d, y)) in rows.iter().enumerate() {
            let before = if i == 0 {
                Vec::new()
            } else {
                held_after(&line, i - 1, cap)
            };
            let step = m.step(&[], y, *d, 1.0);
            if !every_node_excited(&before, 2, line[i].0) {
                assert!(
                    step.pred.iter().all(|v| v.is_nan()),
                    "seed {seed}, row {i}: {step:?}"
                );
                assert!(m.not_identified(&step), "seed {seed}, row {i}: {step:?}");
                withheld.0 += 1;
            } else {
                // A fit of too few events is not identified either; one
                // that is reported is no such fit.
                let fit = step.pred.iter().all(|v| v.is_finite());
                assert_eq!(m.not_identified(&step), !fit, "seed {seed}, row {i}");
                reported += usize::from(fit);
            }
            let after = held_after(&line, i, cap);
            let coef = m.coefficients().concat();
            if !every_node_excited(&after, 2, line[i].0) {
                assert!(
                    coef.iter().all(|v| v.is_nan()),
                    "seed {seed}, row {i}: {coef:?}"
                );
                withheld.1 += 1;
                // The silent node's column of `G`, and its sums, are 0.
                if let Some(j) = (0..2).find(|&j| !after.iter().any(|(_, mk)| mk[j] > 0.0)) {
                    let (g, b) = m.system();
                    let p = 5;
                    for u in 0..2 {
                        let a = 1 + 2 * j + u;
                        assert!(
                            (0..p).all(|c| g[a * p + c] == 0.0),
                            "seed {seed}, row {i}: G's column {a}: {g:?}"
                        );
                        assert!((0..2).all(|ii| b[ii * p + a] == 0.0), "{b:?}");
                    }
                    assert!(b[j * p..(j + 1) * p].iter().all(|v| *v == 0.0), "{b:?}");
                }
            }
        }
    }
    assert!(withheld.0 > 50 && withheld.1 > 50, "{withheld:?}");
    assert!(reported > 500, "{reported}");
}

/// A node whose only events in the window are on the rows at the read's own
/// clock has excitation over none of the observation: its column of `G` is
/// 0 by its definition, however its terms round. The coefficients read
/// right after its events are not reported, nor a read at the same clock;
/// one clock unit on, they are. This held before the window counted its
/// nodes' events (the terms round to a pivot the test refuses), and holds
/// now by the count, with the reason.
#[test]
fn a_nodes_events_at_the_reads_own_clock_excite_nothing_yet() {
    for window_events in [Some(4), None] {
        let mut m = Hawkes::new(HawkesCfg {
            window_events,
            ..cfg(2, &[1.0, 4.0])
        })
        .unwrap();
        let mut s = 5u64;
        for i in 0..12 {
            let d = if i == 0 { 0.0 } else { 0.3 + lcg(&mut s) };
            m.step(&[], &[Some(0.7 + lcg(&mut s)), None], d, 1.0);
        }
        // The second node's first events: two rows at one clock.
        m.step(&[], &[None, Some(1.3)], 0.6, 1.0);
        m.step(&[], &[Some(1.1), Some(0.9)], 0.0, 1.0);
        assert!(
            m.coefficients().concat().iter().all(|v| v.is_nan()),
            "{window_events:?}: {:?}",
            m.coefficients()
        );
        let at = m.predict(&[], 0.0);
        assert!(at.pred.iter().all(|v| v.is_nan()), "{window_events:?}");
        assert!(m.not_identified(&at));
        let on = m.step(&[], &[None, None], 1.0, 1.0);
        assert!(
            on.pred.iter().all(|v| v.is_finite()),
            "{window_events:?}: {on:?}"
        );
        assert!(!m.not_identified(&on));
    }
}

/// A fit withheld by `min_weight` is not one the events fail to identify:
/// the floor names it.
#[test]
fn a_fit_below_min_weight_is_not_called_unidentified() {
    let mut m = Hawkes::new(HawkesCfg {
        min_weight: 1e9,
        ..cfg(1, &[1.0])
    })
    .unwrap();
    for i in 0..20 {
        let s = m.step(&[], &[Some(1.0)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        assert!(s.pred[0].is_nan());
        assert!(!m.not_identified(&s), "row {i}");
    }
}
