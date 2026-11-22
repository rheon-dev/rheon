//! The refit over a span of the clock (`refit_clock`, docs/PLAN.md task
//! 538): the same refit as `refit_rows`, closed by the first row at least
//! the span after a target's first learned row, decided on the exact
//! stamps; its rows checked against the cap as they come.

use super::tests::{cfg_of, stream_of};
use super::*;

const S: i128 = 1_000_000_000;

/// Each row's prediction and the coefficients after it, as bits.
fn bits(m: &mut Expectile, x: &[f64], y: f64, w: f64) -> Vec<u64> {
    let mut out: Vec<u64> = m
        .step(x, &[Some(y)], 1.0, w)
        .pred
        .iter()
        .map(|v| v.to_bits())
        .collect();
    out.extend(m.coefficients()[0].iter().map(|v| v.to_bits()));
    out
}

/// On the row clock, and on a temporal clock of one-second steps, a span
/// of `D` holds the target's first `D + 1` rows: the refit is
/// `refit_rows = D + 1`'s, every output and coefficient to the bit, the
/// refit's own row included -- under no decay and under a decay.
#[test]
fn a_span_of_d_is_d_plus_one_rows_on_a_regular_clock() {
    for lam in [1.0, 0.99] {
        for d in [5.0, 40.0] {
            let mut rows = cfg_of(2, 0.8);
            rows.decay = Decay::Lam(lam);
            rows.refit_rows = d as u32 + 1;
            let mut span = rows.clone();
            span.refit_rows = 0;
            span.refit_clock = Some(d);
            for temporal in [false, true] {
                let (mut a, mut b) = (
                    Expectile::new(rows.clone()).unwrap(),
                    Expectile::new(span.clone()).unwrap(),
                );
                for (i, (x, y)) in stream_of(2, 120, 21).iter().enumerate() {
                    let n = i as i128 + 1;
                    if temporal {
                        b.stamp_next(Stamp::Ns(n * S));
                    } else {
                        b.stamp_next_on_rows(Stamp::from(n as f64));
                    }
                    assert_eq!(
                        bits(&mut a, x, *y, 1.0),
                        bits(&mut b, x, *y, 1.0),
                        "lam {lam}, span {d}, temporal {temporal}, row {i}"
                    );
                    let (sa, sb) = (&a.refit_status()[0], &b.refit_status()[0]);
                    assert_eq!(sa.done, sb.done, "row {i}");
                    assert_eq!(sa.learned, sb.learned, "row {i}");
                }
                assert!(b.refit_status()[0].done);
            }
        }
    }
}

/// Both sides of the span's end: a row a nanosecond short of `D` after the
/// first does not close it, the row at exactly `D` does; and a span that
/// has run before it holds a row per coefficient waits for that row.
#[test]
fn the_span_closes_at_its_length_and_not_a_nanosecond_before() {
    let mut c = cfg_of(2, 0.7);
    c.refit_clock = Some(10.0);
    let data = stream_of(2, 8, 22);
    let mut short = Expectile::new(c.clone()).unwrap();
    let mut on = Expectile::new(c.clone()).unwrap();
    // Rows at 0, 1, 2, 3 s, then one at 10 s less a nanosecond or at 10 s.
    for (i, (x, y)) in data[..4].iter().enumerate() {
        for m in [&mut short, &mut on] {
            m.stamp_next(Stamp::Ns(i as i128 * S));
            m.step(x, &[Some(*y)], 1.0, 1.0);
        }
    }
    let (x, y) = &data[4];
    short.stamp_next(Stamp::Ns(10 * S - 1));
    short.step(x, &[Some(*y)], 1.0, 1.0);
    on.stamp_next(Stamp::Ns(10 * S));
    on.step(x, &[Some(*y)], 1.0, 1.0);
    assert!(!short.refit_status()[0].done);
    assert_eq!(short.refit_status()[0].learned, 5);
    assert!(on.refit_status()[0].done);
    assert_eq!(on.refit_status()[0].learned, 5);

    // Two rows, the second past the span: three coefficients want a third.
    let mut few = Expectile::new(c).unwrap();
    for (i, (x, y)) in data[..3].iter().enumerate() {
        few.stamp_next(Stamp::Ns([0, 20, 21][i] * S));
        few.step(x, &[Some(*y)], 1.0, 1.0);
        assert_eq!(few.refit_status()[0].done, i == 2, "row {i}");
    }
}

/// A row of weight 0 is no row of the span: on the row clock the stream
/// does not count it, so a span of `D` still holds `D + 1` rows that
/// counted, the same refit as the stream without the row.
#[test]
fn a_row_of_weight_zero_is_no_row_of_the_span() {
    let mut c = cfg_of(1, 0.6);
    c.refit_clock = Some(6.0);
    let (mut with, mut without) = (
        Expectile::new(c.clone()).unwrap(),
        Expectile::new(c).unwrap(),
    );
    let data = stream_of(1, 30, 23);
    let mut n = 0.0;
    for (i, (x, y)) in data.iter().enumerate() {
        if i == 3 {
            // The stream hands a weightless row no stamp of its own.
            with.step(x, &[Some(*y)], 1.0, 0.0);
        }
        n += 1.0;
        with.stamp_next_on_rows(Stamp::from(n));
        without.stamp_next_on_rows(Stamp::from(n));
        assert_eq!(
            bits(&mut with, x, *y, 1.0),
            bits(&mut without, x, *y, 1.0),
            "row {i}"
        );
    }
    assert!(with.refit_status()[0].done);
}

/// The span's rows are the data's: past half of `max_state_bytes` a
/// warning, once; the row that would pass the cap is learned by no target,
/// nor any row after it, and the overrun is reported.
#[test]
fn the_spans_rows_are_warned_of_at_half_the_cap_and_stopped_at_it() {
    let mut c = cfg_of(2, 0.6);
    c.refit_clock = Some(1e12);
    let row = 8 * (2 + 2) as u64;
    let fixed = Expectile::new(c.clone()).unwrap().refit_clock_bytes(0);
    let cap = fixed + 100 * row;
    c.max_state_bytes = Some(cap);
    let mut m = Expectile::new(c).unwrap();
    let mut warned_at = None;
    for (i, (x, y)) in stream_of(2, 150, 24).iter().enumerate() {
        m.stamp_next(Stamp::Ns(i as i128 * S));
        m.step(x, &[Some(*y)], 1.0, 1.0);
        if let Some((bytes, got)) = m.take_state_warning() {
            assert!(warned_at.is_none(), "warned twice");
            assert_eq!(got, cap);
            assert!(2 * bytes > cap && 2 * (bytes - row) <= cap, "{bytes}");
            warned_at = Some(i);
        }
    }
    assert!(warned_at.is_some());
    let (bytes, got) = m.state_over_cap().expect("stopped at the cap");
    assert_eq!((bytes, got), (cap + row, cap));
    assert_eq!(m.refit_status()[0].learned, 100);
    assert!(!m.refit_status()[0].done);
}

/// A state saved mid-span, through the file's encoding, goes on to the bit;
/// and the span is written only where it is set, so a state without one
/// keeps its bytes.
#[test]
fn a_state_saved_mid_span_goes_on_to_the_bit() {
    let mut c = cfg_of(2, 0.7);
    c.refit_clock = Some(50.0);
    let data = stream_of(2, 120, 25);
    let mut a = Expectile::new(c).unwrap();
    for (i, (x, y)) in data[..30].iter().enumerate() {
        a.stamp_next(Stamp::Ns(i as i128 * S));
        a.step(x, &[Some(*y)], 1.0, 1.0);
    }
    let bytes = rmp_serde::to_vec_named(&a.state()).unwrap();
    let mut b = Expectile::restore(&rmp_serde::from_slice(&bytes).unwrap()).unwrap();
    for (i, (x, y)) in data[30..].iter().enumerate() {
        let t = Stamp::Ns((30 + i) as i128 * S);
        a.stamp_next(t);
        b.stamp_next(t);
        assert_eq!(bits(&mut a, x, *y, 1.0), bits(&mut b, x, *y, 1.0));
    }
    assert!(b.refit_status()[0].done);
    let plain = rmp_serde::to_vec_named(&Expectile::new(cfg_of(2, 0.7)).unwrap().state()).unwrap();
    let text = String::from_utf8_lossy(&plain);
    assert!(!text.contains("refit_clock") && !text.contains("first"));
}

/// The span's own refusals: not finite or not above 0, beside a count of
/// rows, beside a window.
#[test]
fn the_span_is_checked() {
    let refuse = |c: ExpectileCfg, words: &str| {
        let e = Expectile::new(c).unwrap_err();
        assert!(e.contains(words), "{e}");
    };
    for d in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        let mut c = cfg_of(1, 0.5);
        c.refit_clock = Some(d);
        refuse(c, "refit_span must be a finite number");
    }
    let mut c = cfg_of(1, 0.5);
    c.refit_clock = Some(5.0);
    c.refit_rows = 10;
    refuse(c, "two refits of the first rows");
    let mut c = cfg_of(1, 0.5);
    c.refit_clock = Some(5.0);
    c.window_rows = Some(10);
    refuse(c, "but refit_span is set");
}
