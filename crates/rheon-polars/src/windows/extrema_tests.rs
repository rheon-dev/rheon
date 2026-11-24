//! The extrema (task 561) against their definition: the `since` event read
//! by stamp, both sides of each threshold, the bound on a queue's bytes, and
//! a resume at every row to the bit. The extrema without an event are also
//! in the brute force of `windows.rs`'s tests (`all_kernels`).
use rheon_core::{ClockCfg, OnClockReset};

use super::*;

fn cfg() -> ClockCfg {
    ClockCfg {
        gap_cap: 1e6,
        on_clock_reset: OnClockReset::Error,
        session_gap: None,
        min_backwards_jump: 0.0,
    }
}

fn kernel(window: Option<f64>, closed: Closed, since: Option<usize>) -> KernelDef {
    KernelDef {
        direction: Direction::Backward,
        half_life: f64::INFINITY,
        window_size: window,
        window_ns: window.map(|w| (w * 1e9).round() as i64),
        closed,
        since,
    }
}

fn op(stat: Stat, partial: Partial, cap: Option<u64>) -> OpDef {
    OpDef {
        kernel: 0,
        stat,
        input: 0,
        min_samples: 1,
        partial,
        bias: false,
        quantile: None,
        max_state_bytes: cap,
    }
}

/// A row: its stamp, its value (NaN null) and whether it is an event.
type R = (f64, f64, bool);

/// The core's outputs, saved and loaded between rows at `save_every`, and
/// drained every `every` rows.
fn run(k: &KernelDef, ops: &[OpDef], rows: &[R], every: usize, save_every: usize) -> Vec<Vec<f64>> {
    let mut core = Windows::new(vec![k.clone()], ops.to_vec(), cfg()).unwrap();
    let mut out = vec![Vec::new(); ops.len()];
    let take = |e: Emitted, out: &mut Vec<Vec<f64>>| {
        for (a, b) in out.iter_mut().zip(e.values) {
            a.extend(b);
        }
    };
    for (i, &(t, x, ev)) in rows.iter().enumerate() {
        let group = core.group(None);
        core.push(&RowIn {
            group,
            clock: Some(ClockValue::F64(t)),
            session: None,
            values: &[x, if ev { 1.0 } else { 0.0 }],
            accept: true,
            lists: &[],
        })
        .unwrap();
        if (i + 1) % every == 0 {
            take(core.drain(), &mut out);
        }
        if (i + 1) % save_every == 0 {
            let bytes = rmp_serde::to_vec_named(&core).unwrap();
            core = rmp_serde::from_slice(&bytes).unwrap();
            core.check().unwrap();
        }
    }
    take(core.finish(), &mut out);
    out
}

/// The definition: the extremum of the values at the stamps the window
/// holds, at or after the last stamp holding an event (any of its rows), with
/// the earliest row winning a tie; null with none, and under
/// `partial = "null"` where the span since the stream's start or the event
/// is short of the window.
fn brute(k: &KernelDef, o: &OpDef, rows: &[R]) -> Vec<f64> {
    let max = o.stat == Stat::Max;
    let t0 = rows[0].0;
    rows.iter()
        .map(|&(t, _, _)| {
            let ev = k.since.and_then(|_| {
                rows.iter()
                    .filter(|r| r.2 && r.0 <= t)
                    .map(|r| r.0)
                    .fold(None, |a: Option<f64>, s| Some(a.map_or(s, |a| a.max(s))))
            });
            let start = ev.unwrap_or(t0);
            let near = k.closed.near(Direction::Backward);
            let far = k.closed.far(Direction::Backward);
            let inside = |s: f64| {
                let before = if near { s <= t } else { s < t };
                let after = match k.window_size {
                    None => true,
                    Some(w) => {
                        if far {
                            s >= t - w
                        } else {
                            s > t - w
                        }
                    }
                };
                before && after && ev.is_none_or(|e| s >= e)
            };
            let v = rows
                .iter()
                .filter(|r| inside(r.0) && crate::stream::usable(r.1))
                .fold(f64::NAN, |a, r| extrema::pick(max, false, a, r.1));
            let whole = k.window_size.is_none_or(|w| t - start >= w);
            if !whole && o.partial == Partial::Null {
                f64::NAN
            } else {
                v
            }
        })
        .collect()
}

fn stream(seed: u64, n: usize) -> Vec<R> {
    let mut s = seed;
    let mut next = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (s >> 33) as usize
    };
    let mut t = 0.0;
    (0..n)
        .map(|_| {
            // A repeated stamp a third of the time.
            if next() % 3 != 0 {
                t += [0.5, 1.0, 2.0][next() % 3];
            }
            let x = match next() % 9 {
                0 => f64::NAN,
                1 => 0.0,
                2 => -0.0,
                3 => f64::INFINITY,
                v => (v as f64 - 5.0) * [1.0, 0.5][next() % 2],
            };
            (t, x, next() % 7 == 0)
        })
        .collect()
}

fn same_bits(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

#[test]
fn since_matches_its_definition_by_stamp_in_every_form() {
    for seed in 1..6 {
        let rows = stream(seed, 300);
        for window in [None, Some(3.0), Some(0.5)] {
            for closed in [Closed::Right, Closed::Left, Closed::Both, Closed::Neither] {
                for since in [None, Some(1)] {
                    let k = kernel(window, closed, since);
                    let partial = if window.is_some() {
                        Partial::Null
                    } else {
                        Partial::Keep
                    };
                    let ops = [
                        op(Stat::Max, Partial::Keep, None),
                        op(Stat::Min, partial, None),
                    ];
                    let whole = run(&k, &ops, &rows, rows.len() + 1, rows.len() + 1);
                    for (o, got) in ops.iter().zip(&whole) {
                        let want = brute(&k, o, &rows);
                        let at = got
                            .iter()
                            .zip(&want)
                            .position(|(a, b)| a.to_bits() != b.to_bits());
                        assert!(
                            at.is_none(),
                            "seed {seed} {window:?} {closed:?} since {since:?} {:?} row {:?}: \
                             {:?} against {:?}",
                            o.stat,
                            at,
                            at.map(|i| got[i]),
                            at.map(|i| want[i])
                        );
                    }
                    // Chunked a row at a time and resumed from a save at
                    // every row, and every seventh: the same bits.
                    for (every, save) in [(1, 1), (7, 7), (50, 3)] {
                        let other = run(&k, &ops, &rows, every, save);
                        for (a, b) in whole.iter().zip(&other) {
                            assert!(same_bits(a, b), "seed {seed} {window:?} {closed:?} {every}");
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn an_event_anywhere_in_a_stamp_starts_the_run_at_that_stamp() {
    let k = kernel(None, Closed::Right, Some(1));
    let ops = [op(Stat::Max, Partial::Keep, None)];
    // The event is the stamp's last row: its first row, 9.0, is in the run,
    // and the earlier stamp's 20.0 is not.
    let rows = [
        (0.0, 20.0, false),
        (1.0, 9.0, false),
        (1.0, 3.0, true),
        (2.0, 4.0, false),
    ];
    assert_eq!(run(&k, &ops, &rows, 1, 100)[0], vec![20.0, 9.0, 9.0, 9.0]);
    // Under "left" the event's stamp is null: its own rows are out of its
    // window, and the event emptied the rest.
    let k = kernel(None, Closed::Left, Some(1));
    let got = run(&k, &ops, &rows, 1, 100)[0].clone();
    assert!(
        got[0].is_nan() && got[1].is_nan() && got[2].is_nan(),
        "{got:?}"
    );
    assert_eq!(got[3], 9.0);
}

#[test]
fn a_window_since_an_event_is_whole_from_one_window_on() {
    let ops = [op(Stat::Max, Partial::Null, None)];
    let k = kernel(Some(2.0), Closed::Right, Some(1));
    // The stream's start at 0 and an event at 10: a row exactly 2.0 after
    // either is whole, one a nanosecond short of it is not.
    let short = 12.0 - 1e-9;
    let rows = [
        (0.0, 1.0, false),
        (2.0 - 1e-9, 1.0, false),
        (2.0, 1.0, false),
        (10.0, 5.0, true),
        (short, 3.0, false),
        (12.0, 2.0, false),
    ];
    let got = run(&k, &ops, &rows, 1, 100)[0].clone();
    assert!(got[0].is_nan() && got[1].is_nan(), "{got:?}");
    assert_eq!(got[2], 1.0);
    assert!(got[3].is_nan() && got[4].is_nan(), "{got:?}");
    assert_eq!(got[5], 3.0);
}

#[test]
fn a_queue_past_its_cap_is_refused_at_the_row_that_takes_it_there() {
    let k = kernel(Some(1e6), Closed::Right, None);
    let row = extrema::row_bytes(1, false, 1);
    // A cap of exactly `rows` rows takes them, and refuses the next.
    let rows_cap = 10;
    let cap = (row * rows_cap) as u64;
    let mut core = Windows::new(
        vec![k],
        vec![op(Stat::Max, Partial::Keep, Some(cap))],
        cfg(),
    )
    .unwrap();
    for i in 0..=rows_cap {
        let group = core.group(None);
        let r = core.push(&RowIn {
            group,
            clock: Some(ClockValue::F64(i as f64)),
            session: None,
            values: &[1.0],
            accept: true,
            lists: &[],
        });
        if i < rows_cap {
            assert!(r.is_ok(), "row {i}");
        } else {
            match r {
                Err(Refusal::OverCap {
                    rows,
                    row_bytes,
                    cap: c,
                    ..
                }) => assert_eq!((rows, row_bytes, c), (rows_cap + 1, row, cap)),
                other => panic!("{other:?}"),
            }
        }
    }
    assert_eq!(core.extremum_caps(), vec![(0, row, cap)]);
}

#[test]
fn what_an_extremum_cannot_run_is_refused_by_name() {
    let ahead = KernelDef {
        direction: Direction::Forward,
        ..kernel(Some(2.0), Closed::Right, Some(1))
    };
    let e = ahead.check().unwrap_err();
    assert!(e.contains("reverse=True"), "{e}");
    let decayed = KernelDef {
        half_life: 3.0,
        ..kernel(Some(2.0), Closed::Right, None)
    };
    let e = Windows::new(
        vec![decayed],
        vec![op(Stat::Max, Partial::Keep, None)],
        cfg(),
    )
    .unwrap_err();
    assert!(e.contains("half-life must be inf"), "{e}");
    let e = Windows::new(
        vec![kernel(None, Closed::Right, Some(1))],
        vec![op(Stat::Sum, Partial::Keep, None)],
        cfg(),
    )
    .unwrap_err();
    assert!(
        e.contains("only rolling_max and rolling_min take since"),
        "{e}"
    );
    for (cap, says) in [(0, "above 0"), ((64 << 20) + 1, "at most")] {
        let e = Windows::new(
            vec![kernel(Some(2.0), Closed::Right, None)],
            vec![op(Stat::Max, Partial::Keep, Some(cap))],
            cfg(),
        )
        .unwrap_err();
        assert!(e.contains(says), "{e}");
    }
    // 64 MiB itself is taken.
    Windows::new(
        vec![kernel(Some(2.0), Closed::Right, None)],
        vec![op(Stat::Max, Partial::Keep, Some(64 << 20))],
        cfg(),
    )
    .unwrap();
}
