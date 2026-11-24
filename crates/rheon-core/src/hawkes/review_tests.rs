//! The 2026-11-23 review's `hawkes` findings (PLAN 578f), each test first:
//! it failed on the build before the fix for the review's reason.

use super::*;
use crate::oracle;

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

/// HAWKES-2: a window of clock units that every event has left holds
/// nothing, so its sums are 0 exactly, not what the subtractions of its
/// rows rounded to. Marks of `1e8` and 1 in the older block left `Q =
/// 1e16 + 1 − 1e16 − 1 = −1` (the sum had rounded the 1 away), and the
/// burst after it fitted from `Q = 9` where its ten events make 10. The
/// fit after the window empties is the definition's on the events in the
/// window, as everywhere else ([`oracle::hawkes_system`]).
#[test]
fn a_clock_window_that_empties_keeps_nothing_of_the_rows_that_left() {
    let (span, betas) = (2.0, [2.0]);
    let mut c = cfg(1, &betas);
    c.window_clock = Some(span);
    let mut m = Hawkes::new(c).unwrap();
    // One row to leave first, so the older block is formed of the two
    // after it; a row with no event that forms it; a quiet stretch that
    // drops both; and a burst of ten events.
    let mut rows: Vec<(f64, f64)> = vec![(0.0, 1.0), (0.1, 1e8), (0.1, 1.0), (1.95, 0.0)];
    rows.push((10.0, 0.0));
    rows.extend(std::iter::repeat_n((0.15, 1.0), 10));
    let mut line: Vec<(f64, Vec<f64>)> = Vec::new();
    let mut t = 0.0;
    let mut checked = 0;
    for (i, &(d, mark)) in rows.iter().enumerate() {
        t += d;
        line.push((t, vec![mark]));
        m.step(&[], &[Some(mark)], d, 1.0);
        if i == 4 {
            assert_eq!(m.window_rows(), 0);
            assert!(
                m.r.iter().chain(&m.b).chain(&m.q).all(|v| *v == 0.0),
                "an empty window holds R {:?}, b {:?}, Q {:?}",
                m.r,
                m.b,
                m.q
            );
        }
        if i < 6 {
            continue;
        }
        let window: Vec<(f64, Vec<f64>)> = line
            .iter()
            .filter(|(at, mk)| t - at < span && mk[0] > 0.0)
            .cloned()
            .collect();
        assert_eq!(m.window_rows(), window.len(), "row {i}");
        let (g_want, b_want) = oracle::hawkes_system(&window, &betas, 0.0, t - span, t);
        let (g, b) = m.system();
        let err = rel(&g, &g_want).max(rel(&b, &b_want));
        assert!(err < 1e-12, "row {i}: G/b off the definition by {err:e}");
        checked += 1;
    }
    assert_eq!(checked, 9);
}

/// HAWKES-2's second part: the clock since the newer block's start runs
/// on only while the window holds a row. A window emptied by the older
/// block's last rows leaving kept its clock, and a quiet stretch of `1e16`
/// after it put the next rows at `1e16 + 0.25` and `1e16 + 0.5`, both
/// `1e16`: their gaps were rounded away.
#[test]
fn an_empty_clock_window_counts_no_clock() {
    let mut c = cfg(1, &[1.0]);
    c.window_clock = Some(3.0);
    let mut m = Hawkes::new(c).unwrap();
    for (d, mark) in [(0.0, 1.0), (1.0, 1.0), (0.5, 1.0), (1.7, 0.0), (1.4, 0.0)] {
        m.step(&[], &[Some(mark)], d, 1.0);
    }
    assert_eq!(m.window_rows(), 0);
    m.step(&[], &[Some(0.0)], 1e16, 1.0);
    for _ in 0..4 {
        m.step(&[], &[Some(1.0)], 0.25, 1.0);
    }
    let w = m.window.as_ref().unwrap();
    // The newer block holds the first row at the block's start and each
    // later row by its gap to the row before it (HAWKES-7), so the stretch
    // of `1e16` is in no gap: the three after the first are 0.25 each.
    let time: Vec<f64> = w.time.iter().copied().collect();
    assert_eq!(time, [0.0, 0.25, 0.25, 0.25], "times {time:?}");
}

/// HAWKES-3 (1) and (2): `max_state_bytes` is held to the ceiling every
/// model's is, 64 MiB a group, and the default cap of 256 KB is checked
/// against a window of events as a given cap is.
#[test]
fn the_state_cap_is_held_to_the_ceiling_and_the_default_is_checked() {
    let mut c = cfg(2, &[1.0, 5.0]);
    c.max_state_bytes = Some(crate::clock_span::STATE_BYTES_CEILING);
    assert!(Hawkes::new(c.clone()).is_ok());
    c.max_state_bytes = Some(crate::clock_span::STATE_BYTES_CEILING + 1);
    let e = Hawkes::new(c.clone()).unwrap_err();
    assert!(e.contains("may be at most 67108864"), "{e}");
    c.max_state_bytes = Some(u64::MAX);
    assert!(Hawkes::new(c.clone()).is_err());
    // The default: 2 nodes and 2 decays keep 64 bytes a row.
    c.max_state_bytes = None;
    let fits = ((MAX_STATE_BYTES - Hawkes::state_bytes(2, 2, Some(0))) / 64) as usize;
    c.window_events = Some(fits);
    assert!(Hawkes::new(c.clone()).is_ok());
    c.window_events = Some(fits + 1);
    let e = Hawkes::new(c).unwrap_err();
    assert!(e.contains("max_state_bytes of 262144"), "{e}");
}

/// HAWKES-3 (3): a saved window of clock units holding more rows than its
/// cap takes is a damaged state, refused on load; the same state at a cap
/// that takes it loads.
#[test]
fn a_restored_clock_window_past_its_cap_is_refused() {
    let mut c = cfg(1, &[1.0]);
    c.window_clock = Some(100.0);
    let mut m = Hawkes::new(c).unwrap();
    for i in 0..40 {
        m.step(&[], &[Some(1.0)], if i == 0 { 0.0 } else { 0.5 }, 1.0);
    }
    assert_eq!(m.window_rows(), 40);
    let held = m.clock_bytes_with(0);
    for (cap, loads) in [(held, true), (held - 1, false)] {
        let mut s = m.state();
        let ModelState::Hawkes(inner) = &mut s.model else {
            unreachable!()
        };
        inner.cfg.max_state_bytes = Some(cap);
        match Hawkes::restore(&s) {
            Ok(_) => assert!(loads, "cap {cap}: loaded"),
            Err(StateError::Invalid(e)) => {
                assert!(!loads, "cap {cap}: {e}");
                assert!(e.contains("max_state_bytes"), "{e}");
            }
            Err(e) => panic!("{e:?}"),
        }
    }
}

/// HAWKES-4: a forgetting rate or a kernel's arithmetic that leaves the
/// doubles is refused: a half-life of `1e-310` gave `γ = ∞`, and `G₀₀`'s
/// first step `(1 − e^(−∞·0)) / ∞` was NaN, which no later row washes
/// out; decays of `1e308` and `9e307` sum past the largest double, and
/// `G`'s entries divide by that sum.
#[test]
fn a_rate_or_a_kernel_past_the_doubles_is_refused() {
    let mut c = cfg(1, &[1.0]);
    c.decay = Decay::Halflife(1e-310);
    let e = Hawkes::new(c.clone()).unwrap_err();
    assert!(e.contains("half_life"), "{e}");
    // Both sides: a rate that stays finite is taken.
    c.decay = Decay::Halflife(1e-300);
    assert!(Hawkes::new(c).is_ok());
    for betas in [vec![1e308, 9e307], vec![1e160, 1e150]] {
        let e = Hawkes::new(cfg(1, &betas)).unwrap_err();
        assert!(e.contains("kernel_decays"), "{betas:?}: {e}");
    }
    assert!(Hawkes::new(cfg(1, &[1e150, 1e140])).is_ok());
}

/// HAWKES-5: a row only scored reads a window of clock units at its own
/// stamp, as its step would be handed it ([`Hawkes::predict_at`]), not at
/// the last stamp moved on by the step: on a number clock in tenths `0.2 +
/// 0.7` is not `0.9`, so a row exactly a span old was read on the other
/// side of the edge.
#[test]
fn a_scored_row_reads_the_window_at_its_own_stamp() {
    let tenths: Vec<i64> = vec![1, 2, 9, 10, 12, 17, 20, 21, 29, 30, 33, 38, 41, 49];
    let mut mismatched = 0;
    for span10 in 1..=29i64 {
        let span = span10 as f64 / 10.0;
        let mut c = cfg(1, &[1.0, 3.0]);
        c.window_clock = Some(span);
        let mut m = Hawkes::new(c).unwrap();
        let mut last: Option<f64> = None;
        for (i, &t10) in tenths.iter().enumerate() {
            let t = t10 as f64 / 10.0;
            let d = last.map_or(0.0, |l| t - l);
            let stamp = Stamp::Raw(t, 0.0);
            if i % 3 == 2 {
                let mut stepped = m.clone();
                stepped.stamp_next(stamp);
                let want = stepped.step(&[], &[None], d, 0.0);
                let got = m.predict_at(&[], d, stamp);
                let bits = |s: &Step| s.pred.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
                assert_eq!(bits(&got), bits(&want), "span {span}, row {i}");
                assert_eq!(got.n_eff.to_bits(), want.n_eff.to_bits());
                mismatched += usize::from(bits(&m.predict(&[], d)) != bits(&want));
            } else {
                m.stamp_next(stamp);
                m.step(&[], &[Some(1.0)], d, 1.0);
                last = Some(t);
            }
        }
    }
    // The read ahead from the last stamp parts from the step at some edge.
    assert!(mismatched > 0, "no row sat on an edge");
}
