use super::*;

fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*state >> 11) as f64 / (1u64 << 53) as f64
}

fn cfg() -> AdwinCfg {
    AdwinCfg::default()
}

/// The window is the stream's last `width` rows: its sum, its sum of squared
/// deviations and its buckets' widths against a longhand over those rows,
/// while nothing is cut and while `max_width` drops the oldest buckets.
#[test]
fn the_window_holds_the_last_width_rows() {
    let mut a = Adwin::new(AdwinCfg {
        max_width: 300,
        delta: 1e-12,
        ..cfg()
    });
    let mut rows = Vec::new();
    let mut s = 5u64;
    for i in 0..3000 {
        let x = 10.0 + lcg(&mut s);
        rows.push(x);
        assert!(!a.update(x), "row {i}: no cut on a steady stream");
        let w = a.width() as usize;
        if i < 300 {
            assert_eq!(w, i + 1, "row {i}");
        } else {
            assert!(w <= 300 && w > 150, "row {i}: width {w}");
        }
        let held: usize = a
            .levels
            .iter()
            .enumerate()
            .map(|(lv, b)| b.len() << lv)
            .sum();
        assert_eq!(held, w, "row {i}: the buckets hold the window");
        let last = &rows[rows.len() - w..];
        let mean = last.iter().sum::<f64>() / w as f64;
        let sq: f64 = last.iter().map(|x| (x - mean) * (x - mean)).sum();
        assert!(
            (a.total() - last.iter().sum::<f64>()).abs() < 1e-9,
            "row {i}"
        );
        assert!((a.estimation() - mean).abs() < 1e-12, "row {i}");
        assert!((a.variance() - sq).abs() < 1e-8 * (1.0 + sq), "row {i}");
    }
}

/// The levels never pass their bound at the width held, nor the bound at
/// `max_width`, and the state's bytes follow.
#[test]
fn the_levels_are_bounded_by_the_width() {
    for m in [1usize, 2, 3, 5, 8] {
        for cap in [7u64, 100, 1000, 5000] {
            let mut a = Adwin::new(AdwinCfg {
                max_buckets: m,
                max_width: cap,
                delta: 1e-12,
                ..cfg()
            });
            let mut s = 11u64;
            for i in 0..20_000 {
                a.update(lcg(&mut s));
                let w = a.width() as u64;
                assert!(
                    a.levels() as u64 <= Adwin::levels_bound(w, m),
                    "M = {m}, cap = {cap}, row {i}: {} levels at width {w}",
                    a.levels()
                );
                assert!(a.levels() as u64 <= Adwin::levels_bound(cap, m));
                assert!(a.levels.iter().all(|l| l.len() <= m));
            }
        }
    }
    // The bound is reached: at M = 5, 8 levels hold at least 5·2^7 − 4 = 636
    // rows, and 9 at least 1,276.
    assert_eq!(Adwin::levels_bound(635, 5), 7);
    assert_eq!(Adwin::levels_bound(636, 5), 8);
    assert_eq!(Adwin::levels_bound(1275, 5), 8);
    assert_eq!(Adwin::levels_bound(1276, 5), 9);
    assert_eq!(Adwin::levels_bound(1_000_000, 5), 18);
    assert_eq!(Adwin::state_bytes(1_000_000, 5), 18 * 104 + 104);
}

/// A step in the mean is cut: flagged on a scan row soon after it, and under
/// `restart` the window starts again; kept (the paper's form), the window
/// holds the rows after the cut and the same change is flagged again as the
/// old rows left in it are found.
#[test]
fn a_step_is_cut() {
    for restart in [true, false] {
        let mut a = Adwin::new(AdwinCfg { restart, ..cfg() });
        let mut s = 13u64;
        for _ in 0..5000 {
            assert!(!a.update(lcg(&mut s)));
        }
        let mut flags = Vec::new();
        for i in 0..2000 {
            if a.update(lcg(&mut s) + 0.5) {
                flags.push(i);
                if restart {
                    assert_eq!(a.width(), 0.0, "started again");
                } else {
                    assert!(a.width() > 0.0, "kept");
                }
            }
        }
        assert!(!flags.is_empty() && flags[0] < 64, "{flags:?}");
        // Rows since the start are counted from 1 at the first of 5,000:
        // each scan falls on a multiple of 32 of them.
        assert!(flags.iter().all(|f| (5000 + f + 1) % 32 == 0), "{flags:?}");
        if restart {
            assert_eq!(flags.len(), 1, "{flags:?}");
        } else {
            assert!(flags.len() > 1, "{flags:?}");
        }
    }
}

/// The grace period, both sides: a window of `grace_period` rows is not
/// scanned, one of `grace_period + 1` is. The clock, both sides: a scan runs
/// on a multiple of `clock` rows and on no other.
#[test]
fn the_grace_period_and_the_clock_gate_the_scan() {
    // Ten zeros then a large value: any scan cuts.
    let run = |grace: u64, clock: u64, rows: usize| {
        let mut a = Adwin::new(AdwinCfg {
            grace_period: grace,
            clock,
            min_window_length: 1,
            delta: 0.9,
            ..cfg()
        });
        let mut flagged = None;
        for i in 1..=rows {
            let x = if i <= 10 { 0.0 } else { 1000.0 };
            if a.update(x) {
                flagged = Some(i);
                break;
            }
        }
        flagged
    };
    assert_eq!(run(11, 1, 11), None, "width 11 = grace: no scan");
    assert_eq!(run(10, 1, 11), Some(11), "width 11 > grace 10: cut");
    assert_eq!(run(0, 4, 11), None, "row 11 is no multiple of 4");
    assert_eq!(run(0, 4, 12), Some(12), "row 12 is");
}

/// The shortest part, both sides: a split is tested only where each part
/// holds at least `min_window_length` rows. Thirty-eight 0s, then 1,000s: at
/// row 40 the split after the 0s has two rows newer, which `L = 2` tests and
/// cuts at and `L = 3` does not test (its other splits' gaps are under their
/// bounds there).
#[test]
fn min_window_length_gates_the_splits() {
    let run = |l: u64| {
        let mut a = Adwin::new(AdwinCfg {
            grace_period: 0,
            clock: 1,
            min_window_length: l,
            max_buckets: 100,
            ..cfg()
        });
        (1..=60).find(|&i| a.update(if i <= 38 { 0.0 } else { 1000.0 }))
    };
    assert_eq!(run(2), Some(40));
    assert!(run(3).is_none_or(|i| i > 40));
}

/// A window dropped whole by `max_width` leaves nothing behind: no `0 / 0`
/// enters the sums.
#[test]
fn a_window_dropped_whole_stays_finite() {
    let mut a = Adwin::new(AdwinCfg {
        max_buckets: 1,
        max_width: 1,
        ..cfg()
    });
    for i in 0..200 {
        a.update(f64::from(i));
        assert!(a.total().is_finite() && a.variance().is_finite(), "row {i}");
        assert!(a.width() <= 1.0);
    }
}

/// A detector saved mid-stream goes on to the bit.
#[test]
fn a_saved_detector_goes_on_to_the_bit() {
    let mut a = Adwin::new(AdwinCfg {
        clock: 1,
        max_width: 2000,
        ..cfg()
    });
    let mut s = 17u64;
    for _ in 0..1500 {
        a.update(lcg(&mut s));
    }
    let mut b: Adwin = rmp_serde::from_slice(&rmp_serde::to_vec_named(&a).unwrap()).unwrap();
    let mut flags = 0;
    for i in 0..6000 {
        let x = lcg(&mut s) + if i > 3000 { 0.4 } else { 0.0 };
        let (fa, fb) = (a.update(x), b.update(x));
        assert_eq!(fa, fb, "row {i}");
        flags += usize::from(fa);
    }
    assert!(flags > 0);
    assert_eq!(a, b);
}

/// The levels held are the exponential histogram's: one with no rows, two
/// at the first merge (the sixth row at `M = 5`), and the count a model of
/// the buckets' counts gives as it grows.
#[test]
fn the_levels_held_follow_the_merges() {
    let mut a = Adwin::new(AdwinCfg {
        delta: 1e-12,
        ..cfg()
    });
    assert_eq!(a.levels(), 1);
    let (mut counts, mut s) = (vec![0usize], 41u64);
    for i in 1..=400usize {
        assert!(!a.update(lcg(&mut s)));
        counts[0] += 1;
        let mut lv = 0;
        while counts[lv] == 6 {
            counts[lv] -= 2;
            if lv + 1 == counts.len() {
                counts.push(0);
            }
            counts[lv + 1] += 1;
            lv += 1;
        }
        assert_eq!(a.levels(), counts.len(), "row {i}");
        assert_eq!(a.levels() == 1, i <= 5, "row {i}");
    }
    assert!(a.levels() >= 4);
}

/// `sound` refuses each way a state can be out of shape, alone, and takes the
/// shape just inside each bound.
#[test]
fn a_restored_windows_shape_is_checked_bound_by_bound() {
    let mut a = Adwin::new(AdwinCfg {
        delta: 1e-12,
        ..cfg()
    });
    let mut s = 3u64;
    for _ in 0..90 {
        a.update(lcg(&mut s));
    }
    assert!(a.sound());
    let with = |f: &dyn Fn(&mut Adwin)| {
        let mut b = a.clone();
        f(&mut b);
        b.sound()
    };
    // Settings: open at 0 and at 1, closed inside.
    assert!(!with(&|b| b.delta = 0.0));
    assert!(with(&|b| b.delta = 1e-300));
    assert!(!with(&|b| b.delta = 1.0));
    assert!(with(&|b| b.delta = 1.0 - f64::EPSILON));
    assert!(!with(&|b| b.delta = f64::NAN));
    // The most levels: 63 are held, 64 are not.
    let empty = |n: usize| {
        let mut b = Adwin::new(cfg());
        b.levels = vec![VecDeque::new(); n];
        b.sound()
    };
    assert!(empty(63) && !empty(64) && !empty(65) && !empty(0));
    assert!(!with(&|b| b.clock = 0) && with(&|b| b.clock = 1));
    assert!(!with(&|b| b.max_buckets = 0));
    assert!(!with(&|b| b.max_width = 0));
    assert!(!with(&|b| b.width += 1.0));
}

/// The cut's bound, term by term: both sides of `ε` to the last bit, for
/// parts of unequal size and `L > 1`, so each of the two `1/m` terms and the
/// shift by `L` is read, and a gap of `-ε` is a cut as `+ε` is.
#[test]
fn the_cut_is_the_bernstein_bound_to_the_bit() {
    let mut a = Adwin::new(AdwinCfg {
        clock: 1 << 40,
        min_window_length: 5,
        ..cfg()
    });
    let mut s = 29u64;
    for _ in 0..40 {
        a.update(lcg(&mut s) * 3.0);
    }
    assert!(a.variance() > 1.0);
    for (n0, n1, d) in [(20.0, 7.0, 4.0), (6.0, 34.0, 5.5), (9.0, 9.0, 2.0)] {
        let m_recip = 1.0 / (n0 - 5.0 + 1.0) + 1.0 / (n1 - 5.0 + 1.0);
        let eps = (2.0 * m_recip * (a.variance() / a.width()) * d).sqrt() + 2.0 / 3.0 * d * m_recip;
        let above = f64::from_bits(eps.to_bits() + 1);
        let below = f64::from_bits(eps.to_bits() - 1);
        for sign in [1.0, -1.0] {
            assert!(!a.cut(n0, n1, sign * eps, d), "{n0} {n1} at the bound");
            assert!(!a.cut(n0, n1, sign * below, d), "{n0} {n1} below");
            assert!(a.cut(n0, n1, sign * above, d), "{n0} {n1} above");
        }
    }
}

/// The window as a list of rows, scanned longhand: every row its own bucket
/// (`M` past the rows fed), the oldest split `W0 | W1` whose gap passes the
/// bound cut, the oldest row dropped, the scan over again on the shorter
/// window. The detector's flags and the window's width are this scan's, row
/// by row, for several `L`, deltas and both `restart`s: so `d`, `1/m`, the
/// order of the splits, the repeat after a cut and the skipped empty `W1`
/// are each held, not only the flags' count.
#[test]
fn the_scan_is_the_papers_longhand() {
    fn first_cut(win: &[f64], delta: f64, l: u64) -> bool {
        let w = win.len() as f64;
        let mean = win.iter().sum::<f64>() / w;
        let var: f64 = win.iter().map(|x| (x - mean) * (x - mean)).sum();
        let d = (2.0 * w.ln() / delta).ln();
        (1..win.len()).any(|k| {
            let (n0, n1) = (k as f64, (win.len() - k) as f64);
            if (k as u64) < l || ((win.len() - k) as u64) < l {
                return false;
            }
            let lf = l as f64;
            let m = 1.0 / (n0 - lf + 1.0) + 1.0 / (n1 - lf + 1.0);
            let eps = (2.0 * m * (var / w) * d).sqrt() + 2.0 / 3.0 * d * m;
            let gap = win[..k].iter().sum::<f64>() / n0 - win[k..].iter().sum::<f64>() / n1;
            gap.abs() > eps
        })
    }
    for (delta, l, restart) in [
        (0.05, 1, true),
        (0.05, 1, false),
        (0.3, 3, false),
        (0.002, 5, false),
        (0.9, 2, true),
    ] {
        let mut a = Adwin::new(AdwinCfg {
            delta,
            clock: 1,
            max_buckets: 5000,
            min_window_length: l,
            grace_period: 4,
            restart,
            ..cfg()
        });
        let (mut win, mut s, mut flags) = (Vec::<f64>::new(), 31u64, 0);
        for i in 0..700 {
            let level = match i / 150 {
                0 | 2 => 0.0,
                1 => 1.5,
                _ => 0.6,
            };
            let x = level + lcg(&mut s);
            win.push(x);
            let mut expect = false;
            if win.len() > 4 {
                while first_cut(&win, delta, l) {
                    win.remove(0);
                    expect = true;
                }
            }
            let got = a.update(x);
            if restart && expect {
                win.clear();
            }
            assert_eq!(got, expect, "delta {delta} L {l} row {i}");
            assert_eq!(a.width() as usize, win.len(), "delta {delta} L {l} row {i}");
            flags += usize::from(got);
        }
        assert!(flags >= 2, "delta {delta} L {l}: {flags} cuts");
    }
}

/// `L = 0` lets a part of no rows through the length test, and no split is
/// tested with an empty newer part, whose mean is a sum left over by
/// rounding over no rows: 0.1 added three times and taken off three times
/// leaves 2.7e-17, which divided by 0 would be infinite and cut every
/// constant stream.
#[test]
fn a_split_with_no_newer_rows_is_never_tested() {
    let mut a = Adwin::new(AdwinCfg {
        delta: 0.5,
        clock: 1,
        grace_period: 0,
        min_window_length: 0,
        ..cfg()
    });
    for i in 0..6 {
        assert!(!a.update(0.1), "row {i}");
    }
    assert_eq!(a.width(), 6.0);
}

/// A window past `max_width` loses buckets until it is not, and no scan
/// goes on past the buckets held: the widths follow the cap row by row.
#[test]
fn the_cap_is_held_row_by_row_and_a_scan_ends() {
    let mut a = Adwin::new(AdwinCfg {
        max_width: 40,
        max_buckets: 2,
        delta: 1e-12,
        ..cfg()
    });
    let mut s = 37u64;
    for i in 0..500 {
        a.update(lcg(&mut s));
        assert!(a.width() <= 40.0, "row {i}: {}", a.width());
        assert!(i < 100 || a.width() >= 12.0, "row {i}: {}", a.width());
    }
}

/// Two outliers dropped by `max_width` leave no variance behind: once the
/// window holds only zeros its sum and its sum of squared deviations are 0,
/// neither below 0 (where `ε` is NaN and no split is ever cut) nor a ghost
/// left by cancellation above it (where `ε` is huge), and a shift after
/// them is cut where the window is wide enough to see it (review
/// 2026-11-23, DRIFT-2: the review's two cases).
#[test]
fn outliers_that_leave_the_window_leave_no_variance_behind() {
    for (outliers, max_width) in [
        ([1e20, 3.3e19], 25u64),
        ([1e20, 3.3e19], 64),
        ([3.3e19, 1e20], 25),
        ([-1e20, 7e19], 40),
        ([1e15, 1e-3], 25),
        ([1e20, 3.3e19], 300),
        ([-1e20, 7e19], 500),
    ] {
        for restart in [true, false] {
            let mut a = Adwin::new(AdwinCfg {
                max_width,
                restart,
                ..cfg()
            });
            for x in outliers {
                a.update(x);
            }
            // Every row from `max_width` on is one fed after the outliers.
            let (gone, shift) = (max_width as usize, max_width as usize + 100);
            let mut flags = Vec::new();
            for i in 0..shift + 1000 {
                let x = if i < shift { 0.0 } else { 1.0 };
                if a.update(x) {
                    flags.push(i);
                }
                assert!(
                    a.variance() >= 0.0,
                    "{outliers:?} at {max_width}, row {i}: V = {}",
                    a.variance()
                );
                if (gone..shift).contains(&i) {
                    assert_eq!(
                        (a.total(), a.variance()),
                        (0.0, 0.0),
                        "{outliers:?} at {max_width}, restart {restart}, row {i}: zeros only"
                    );
                }
            }
            // A unit step is inside the bound of a window under about 60
            // rows whatever came before.
            assert!(
                max_width < 300 || flags.iter().any(|&i| (shift..shift + 300).contains(&i)),
                "{outliers:?} at {max_width}, restart {restart}: the shift at {shift} is cut: \
                 {flags:?}"
            );
        }
    }
}
