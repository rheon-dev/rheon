//! Tests that hold `hawkes.rs`'s lines to numbers a mutation of them would
//! change (docs/PLAN.md task 546): each states the value from its
//! definition, beside an ordinary case where the line has one.

use super::*;

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

/// The forgetting rate `γ`: `ln 2 / half_life`, and `−ln λ` for a factor
/// `λ` per clock unit, both above 0; none at `λ = 1` or an infinite
/// half-life.
#[test]
fn the_forgetting_rate_is_positive_for_a_decay() {
    assert_eq!(rate(Decay::Halflife(2.0)), std::f64::consts::LN_2 / 2.0);
    assert_eq!(rate(Decay::Halflife(f64::INFINITY)), 0.0);
    assert_eq!(rate(Decay::Lam(1.0)), 0.0);
    assert_eq!(rate(Decay::Lam(0.5)), std::f64::consts::LN_2);
    let g = rate(Decay::Lam(0.99));
    assert!((g - 0.010_050_335_853_501_45).abs() < 1e-15, "{g}");
}

/// A kernel's decay must be above 0: 0 is no kernel, and is refused with
/// the others that are not positive.
#[test]
fn a_kernel_decay_of_zero_is_refused() {
    let e = cfg(1, &[1.0, 0.0]).validate().unwrap_err();
    assert!(e.contains("above 0, but one is 0"), "{e}");
    assert!(cfg(1, &[1.0, 1e-300]).validate().is_ok());
}

/// The state's bytes against `max_state_bytes`: a state of exactly the cap
/// is allowed and one byte under it is not, and a model with no target, or
/// no kernel, is refused for that alone, with no size to go with it.
#[test]
fn the_state_cap_is_a_byte_count_a_model_may_reach() {
    let bytes = Hawkes::state_bytes(2, 3, None);
    let at = |cap: u64| HawkesCfg {
        max_state_bytes: Some(cap),
        ..cfg(2, &[1.0, 4.0, 15.0])
    };
    assert!(at(bytes).validate().is_ok());
    let e = at(bytes - 1).validate().unwrap_err();
    assert!(e.contains(&format!("would hold {bytes} bytes")), "{e}");
    let windowed = |cap: u64| HawkesCfg {
        window_events: Some(10),
        ..at(cap)
    };
    let bytes = Hawkes::state_bytes(2, 3, Some(10));
    assert!(windowed(bytes).validate().is_ok());
    assert!(windowed(bytes - 1).validate().is_err());
    let clocked = |cap: u64| HawkesCfg {
        window_clock: Some(5.0),
        ..at(cap)
    };
    let bytes = Hawkes::clock_state_bytes(2, 3, 0);
    assert!(clocked(bytes).validate().is_ok());
    assert!(clocked(bytes - 1).validate().is_err());

    for c in [
        HawkesCfg {
            max_state_bytes: Some(1),
            ..cfg(0, &[1.0])
        },
        HawkesCfg {
            max_state_bytes: Some(1),
            ..cfg(1, &[])
        },
    ] {
        let e = c.validate().unwrap_err();
        assert!(!e.contains("would hold"), "a size for no model: {e}");
    }
}

/// `n` rows' worth of a clock window's bytes, written from its parts: the
/// sums `8 (k + D (1 + k) + D (D + 1)/2 + 2)`, the rows `8 (rows (2 + D +
/// k) + k + 3)`, a stamp for each row and the one the clock tracks, and the
/// clock summed.
#[test]
fn a_clock_windows_bytes_are_its_parts() {
    let stamp = std::mem::size_of::<Stamp>() as u64;
    for (n, u, rows) in [(2u64, 3u64, 5u64), (1, 2, 0), (3, 1, 11)] {
        let k = n * u;
        let sums = 8 * (k + n * (1 + k) + n * (n + 1) / 2 + 2);
        let held = 8 * (rows * (2 + n + k) + k + 3);
        let want = sums + held + (rows + 1) * stamp + 8;
        assert_eq!(
            Hawkes::clock_state_bytes(n as usize, u as usize, rows as usize),
            want,
            "{n} nodes, {u} decays, {rows} rows"
        );
    }
}

/// A mark is the target's value where above 0, else 0 -- a plain `+0.0`
/// for `-0.0` too, which is no event.
#[test]
fn a_mark_is_a_positive_value_or_plain_zero() {
    assert_eq!(Hawkes::mark(Some(2.5)), 2.5);
    for v in [Some(0.0), Some(-0.0), Some(-1.0), Some(f64::NAN), None] {
        assert_eq!(Hawkes::mark(v).to_bits(), 0, "{v:?}");
    }
}

/// A read of order `p` keeps its scratch for the thread up to
/// `KEPT_ORDER` and leaves it empty past it.
fn kept() -> bool {
    let sc = READ.take();
    let kept = sc.as_ref().is_some_and(|sc| sc.g.capacity() > 0);
    READ.set(sc);
    kept
}

#[test]
fn the_read_scratch_is_kept_up_to_the_order_and_released_past_it() {
    for (order, want) in [
        (KEPT_ORDER, true),
        (KEPT_ORDER + 1, false),
        (3, true),
        (0, true),
    ] {
        with_scratch(order, |sc| sc.g.resize(9, 1.0));
        assert_eq!(kept(), want, "order {order}");
        with_scratch(0, |sc| *sc = ReadScratch::default());
    }
}

/// Each read -- a step's, a `predict`'s, and a `predict` across rows that
/// leave a clock window -- holds the scratch of a small model and drops
/// that of one of more than `KEPT_ORDER` coefficients. The order is `1 +
/// nodes · decays` read from the sums' length over the nodes: 19 at 2
/// nodes and 9 decays (a node count in its place, or the sums' length
/// itself, would be past 64), 65 at 32 decays (a remainder would be 0).
#[test]
fn every_read_keeps_a_small_models_scratch_and_drops_a_large_ones() {
    let betas = |u: usize| (1..=u).map(|i| i as f64).collect::<Vec<_>>();
    for (u, want) in [(9usize, true), (32, false)] {
        let rows = [
            (0.0, vec![Some(1.0), Some(2.0)]),
            (0.5, vec![Some(1.0), None]),
            (0.5, vec![None, Some(1.0)]),
            (1.0, vec![Some(3.0), Some(1.0)]),
        ];
        // A step, then a read ahead of the same state.
        let mut m = Hawkes::new(cfg(2, &betas(u))).unwrap();
        for (d, y) in &rows {
            let _ = READ.take();
            m.step(&[], y, *d, 1.0);
            assert_eq!(kept(), want, "step, {u} decays");
        }
        let _ = READ.take();
        m.predict(&[], 0.5);
        assert_eq!(kept(), want, "predict, {u} decays");
        // A clock window whose oldest row is out at the read: the copy
        // advanced across it is read in place.
        let mut m = Hawkes::new(HawkesCfg {
            window_clock: Some(1.5),
            ..cfg(2, &betas(u))
        })
        .unwrap();
        let mut at = 0.0;
        for (d, y) in &rows {
            at += d;
            m.stamp_next(Stamp::from(at));
            m.step(&[], y, *d, 1.0);
        }
        // At 2.5 the row at 1.0 has left, and the row at 2.0, of both
        // nodes, is in the window: the read solves.
        assert!(m.oldest_has_left(Stamp::from(2.5)));
        let _ = READ.take();
        m.predict(&[], 0.5);
        assert_eq!(kept(), want, "predict past the window, {u} decays");
    }
}

/// With nine nodes a row's marks do not fit the stack array of eight and
/// take the list: every node's event is learned (`b₀ = Σ m`, the first
/// column of each node's right-hand side), through both paths' boundary.
#[test]
fn a_row_of_more_nodes_than_the_stack_array_holds_is_learned() {
    for n in [8usize, 9, 12] {
        let mut m = Hawkes::new(cfg(n, &[1.0])).unwrap();
        let y: Vec<Option<f64>> = (0..n).map(|j| Some(1.0 + j as f64)).collect();
        m.step(&[], &y, 0.0, 1.0);
        m.step(&[], &y, 0.5, 1.0);
        let (_, b) = m.system();
        let p = 1 + n;
        for j in 0..n {
            assert_eq!(b[j * p], 2.0 * (1.0 + j as f64), "{n} nodes, node {j}");
        }
        let q: f64 = m.q.iter().sum();
        let marks: Vec<f64> = (0..n).map(|j| 1.0 + j as f64).collect();
        // `Q = Σ_rows m m'`, its lower triangle summed, twice over.
        let mut want = 0.0;
        for j in 0..n {
            for l in 0..=j {
                want += 2.0 * marks[j] * marks[l];
            }
        }
        assert!((q - want).abs() < 1e-9 * want, "{n} nodes: {q} vs {want}");
    }
}

/// A saved state restores at every node count, under no window and a window
/// of events: the packed `Q` holds `n (n + 1)/2` numbers, which is 3 at two
/// nodes, where `n + (n + 1)/2` is 3 too, and 1, 6 and 10 at one, three and
/// four, where it is not.
#[test]
fn a_saved_state_restores_at_every_node_count() {
    for n in [1usize, 2, 3, 4] {
        for window in [None, Some(6usize)] {
            let mut m = Hawkes::new(HawkesCfg {
                window_events: window,
                ..cfg(n, &[1.0, 4.0])
            })
            .unwrap();
            let mut s = 3u64;
            for i in 0..40 {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let y: Vec<Option<f64>> = (0..n)
                    .map(|j| ((s >> (8 + j)) & 1 == 1).then_some(1.0 + (i % 3) as f64))
                    .collect();
                m.step(&[], &y, 0.3 + 0.1 * (i % 4) as f64, 1.0);
            }
            assert_eq!(m.q.len(), n * (n + 1) / 2);
            let back = Hawkes::restore(&m.state()).unwrap();
            assert_eq!(back, m, "{n} nodes, window {window:?}");
        }
    }
}

/// A window the settings do not ask for is not a shape the state can have,
/// whatever its own numbers are.
#[test]
fn a_window_the_settings_do_not_ask_for_is_refused() {
    let mut m = Hawkes::new(cfg(2, &[1.0, 4.0])).unwrap();
    m.step(&[], &[Some(1.0), None], 0.0, 1.0);
    let (_, _, k) = m.cfg.dims();
    for clocked in [false, true] {
        let mut s = m.state();
        let ModelState::Hawkes(inner) = &mut s.model else {
            unreachable!()
        };
        inner.window = Some(EventWindow::new(2, k, clocked));
        match Hawkes::restore(&s) {
            Err(StateError::Invalid(e)) => assert!(e.contains("wrong shape"), "{e}"),
            other => panic!("clocked {clocked}: {other:?}"),
        }
    }
}

/// A clock window that holds rows with the event rows of a long stream is
/// bounded by those rows: a trim that finds the oldest row left takes one
/// out each pass and ends, whatever one pass did (a pass that took none
/// out would end the trim by its count, leaving the rows in).
#[test]
fn a_clock_window_trims_every_row_that_has_left() {
    let mut m = Hawkes::new(HawkesCfg {
        window_clock: Some(1.0),
        ..cfg(1, &[1.0, 3.0])
    })
    .unwrap();
    for i in 0..8 {
        m.stamp_next(Stamp::from(0.25 * i as f64));
        m.step(&[], &[Some(1.0)], if i == 0 { 0.0 } else { 0.25 }, 1.0);
    }
    // Rows at 0, 0.25 ... 1.75: those under a span of 1 old at 1.75 are
    // the rows at 1.0 and later, 4 of them.
    assert_eq!(m.window_rows(), 4);
    // A long gap leaves none.
    m.stamp_next(Stamp::from(20.0));
    m.step(&[], &[None], 18.25, 1.0);
    assert_eq!(m.window_rows(), 0);
}

/// HAWKES-7 (the 2026-11-23 review, PLAN 578d): a window of events holds
/// its newer block's times from the block's start, which only the forming
/// of an older block moved. After a quiet stretch of `1e16` the newer
/// rows' times were `1e16 + dt`, and their gaps of 0.25 rounded away when
/// they became the older block: the observation `G₀₀` read 0, the solve
/// failed, and every intensity was NaN. The same events after a quiet
/// stretch of 100, where the kernels' `e^(−2·100)` is as nothing, are the
/// reference: once the window's dropped row is past the stretch (from row
/// 6), its fit over the last two events is the one after `1e16`.
///
/// Fixed by the gaps: a newer row holds its gap to the row before it, so a
/// stretch adds nothing to a gap after it.
#[test]
fn a_window_of_events_after_a_long_quiet_stretch_keeps_their_gaps() {
    let run = |quiet: f64| {
        let mut m = Hawkes::new(HawkesCfg {
            window_events: Some(2),
            ..cfg(1, &[2.0])
        })
        .unwrap();
        let mut out = Vec::new();
        for d in [0.0, 1.0, 1.0, quiet, 0.25, 0.5, 0.25, 0.75] {
            out.push(m.step(&[], &[Some(1.0)], d, 1.0).pred);
        }
        (out, m.g0)
    };
    let (far, g_far) = run(1e16);
    let (near, g_near) = run(100.0);
    assert!(g_far > 0.0 && g_far == g_near, "{g_far} against {g_near}");
    for (i, (a, b)) in far.iter().zip(&near).enumerate().skip(6) {
        for (x, y) in a.iter().zip(b) {
            assert!(
                x.is_nan() == y.is_nan()
                    && (x.is_nan() || (x - y).abs() <= 1e-9 * y.abs().max(1.0)),
                "row {i}: {a:?} against {b:?}"
            );
        }
    }
}

/// HAWKES-7's second path (PLAN 578d, found against the 60-digit fit while
/// the review's fix was held to it): the quiet stretch falls inside the
/// newer block, not before its first row. With three rows held, the fifth
/// row is the newer block's first, the stretch comes before the sixth, and
/// the sixth and seventh rows' times are `1 + 1e16 + dt`: their gap of 0.25
/// rounds away when they become the older block. Moving the older block's
/// times to the newer block's first row (the review's fix) leaves this as
/// it is, since that row is before the stretch. Through the bank, on an
/// integer clock and `window_events = 8`, the fit after a stretch of `1e16`
/// is 1.6e-2 of its largest coefficient off the fit of the window's events
/// at 60 digits, and 2.5 times it after `1e18`, with the review's fix and
/// without it. From row 9 the row dropped last is past the stretch, and the
/// fit is that of the same rows after a stretch of 100. Row 8 is observed
/// from row 4's clock, before the stretch, so its `G₀₀` holds the stretch
/// and its fit is not the near run's: it is held to the definition's
/// system over its three rows, on the clock moved to row 5.
///
/// Fixed by the same gaps, and the clock since the newest row beside the
/// clock since the block's start: a newer row's time is its gap to the row
/// before it, whichever row the stretch falls after.
#[test]
fn a_quiet_stretch_inside_the_newer_block_keeps_the_later_gaps() {
    let run = |quiet: f64| {
        let mut m = Hawkes::new(HawkesCfg {
            window_events: Some(3),
            ..cfg(1, &[2.0])
        })
        .unwrap();
        let mut out = Vec::new();
        for d in [0.0, 1.0, 1.0, 1.0, 1.0, quiet, 0.25, 0.5, 0.25, 0.75, 0.5] {
            out.push(m.step(&[], &[Some(1.0)], d, 1.0).pred);
        }
        out
    };
    let (far, near) = (run(1e16), run(100.0));
    assert!(
        near[8..].iter().flatten().all(|x| x.is_finite()),
        "{near:?}"
    );
    // Row 8: rows 5, 6 and 7 at 0, 0.25 and 0.75, read at 1.0, observed
    // from row 4 at `−1e16`.
    let window: Vec<(f64, Vec<f64>)> = [0.0, 0.25, 0.75].iter().map(|t| (*t, vec![1.0])).collect();
    let (g, b) = crate::oracle::hawkes_system(&window, &[2.0], 0.0, -1e16, 1.0);
    let theta = crate::oracle::solve_columns(&g, &b, 1);
    let excite: f64 = [0.0f64, 0.25, 0.75]
        .iter()
        .map(|t| 2.0 * (-2.0 * (1.0 - t)).exp())
        .sum();
    let want = theta[0] + theta[1] * excite;
    assert!(
        (far[8][0] - want).abs() <= 1e-9 * want.abs(),
        "row 8: {} against the definition's {want}",
        far[8][0]
    );
    for (i, (a, b)) in far.iter().zip(&near).enumerate().skip(9) {
        for (x, y) in a.iter().zip(b) {
            assert!(
                x.is_nan() == y.is_nan()
                    && (x.is_nan() || (x - y).abs() <= 1e-9 * y.abs().max(1.0)),
                "row {i}: {a:?} against {b:?}"
            );
        }
    }
}
