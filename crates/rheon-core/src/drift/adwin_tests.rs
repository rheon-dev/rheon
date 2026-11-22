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
