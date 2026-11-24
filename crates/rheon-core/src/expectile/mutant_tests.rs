//! Tests that hold `expectile.rs`'s lines to numbers a mutation of them
//! would change (the survivors of W11's changed-lines pass): the refit
//! buffer's shape, the bytes the refit holds and reserves, what a span's row
//! is charged and when, the warning's flag, and the centred fallback's guard.

use super::tests::{cfg_of, stream_of};
use super::*;

const S: i128 = 1_000_000_000;

fn stamp_bytes() -> u64 {
    std::mem::size_of::<Option<Stamp>>() as u64
}

fn spanned(k: usize, targets: usize) -> Expectile {
    let mut c = cfg_of(k, 0.7);
    c.n_targets = targets;
    c.refit_clock = Some(1e12);
    Expectile::new(c).unwrap()
}

/// A waiting buffer holds fewer rows than its count, none once it is done; a
/// buffer with a first stamp is a span's.
#[test]
fn a_refit_buffers_shape_is_its_counts() {
    let rows = |learned: u32| Refit {
        rows: vec![0.0; learned as usize * 4],
        learned,
        ..Refit::new()
    };
    assert!(rows(5).fits(4, 6, false));
    assert!(!rows(6).fits(4, 6, false), "as many as its count: it ran");
    assert!(!rows(7).fits(4, 6, false));
    assert!(rows(7).fits(4, 6, true), "a span holds what its span has");
    assert!(rows(0).fits(4, 0, false));
    assert!(!rows(1).fits(4, 0, false), "a count of 0 reads as 1");
    let done = Refit {
        done: true,
        ..Refit::new()
    };
    assert!(done.fits(4, 6, false));
    let mut wrong_width = rows(3);
    wrong_width.rows.pop();
    assert!(!wrong_width.fits(4, 6, false));
    assert!(!wrong_width.fits(4, 6, true));
    // A first stamp belongs to a span alone.
    let stamped = Refit {
        first: Some(Stamp::Ns(0)),
        ..rows(2)
    };
    assert!(stamped.fits(4, 6, true));
    assert!(!stamped.fits(4, 6, false));
    assert!(
        !Refit {
            first: Some(Stamp::Ns(0)),
            ..done
        }
        .fits(4, 6, false)
    );
}

/// The bytes a refit may hold: `8 · targets · refit_rows · (k + 2)` on a count,
/// what the cap leaves beside the sums on a span, and none with the refit off.
#[test]
fn the_refit_reservation_is_counted_by_target_row_and_width() {
    let off = Expectile::new(cfg_of(3, 0.7)).unwrap();
    assert_eq!(off.refit_reservation(), 0);
    let mut c = cfg_of(3, 0.7);
    c.n_targets = 2;
    c.refit_rows = 50;
    let m = Expectile::new(c).unwrap();
    assert_eq!(m.refit_reservation(), 8 * 2 * 50 * (3 + 2));
    let mut c = cfg_of(3, 0.7);
    c.n_targets = 2;
    c.refit_clock = Some(10.0);
    c.max_state_bytes = Some(5000);
    let m = Expectile::new(c).unwrap();
    assert_eq!(
        m.refit_reservation() as u64,
        5000 - Expectile::sums_bytes(3, 2, true, false)
    );
}

/// A span's bytes: the sums, `k + 2` numbers a row held and a row to come, and
/// a stamp a target.
#[test]
fn a_spans_bytes_are_counted_by_target_row_and_width() {
    let mut m = spanned(3, 2);
    let sums = Expectile::sums_bytes(3, 2, true, false);
    assert_eq!(m.refit_clock_bytes(0), sums + 2 * stamp_bytes());
    assert_eq!(
        m.refit_clock_bytes(3),
        sums + 8 * 3 * (3 + 2) + 2 * stamp_bytes()
    );
    let data = stream_of(3, 4, 7);
    for (i, (x, y)) in data.iter().enumerate() {
        m.stamp_next(Stamp::Ns(i as i128 * S));
        m.step(x, &[Some(*y), Some(*y)], 1.0, 1.0);
    }
    assert_eq!(
        m.refit_clock_bytes(1),
        sums + 8 * (2 * 4 * (3 + 2) + (3 + 2)) + 2 * stamp_bytes()
    );
}

/// The warning's flag is state: saved after the warning, it restores raised.
#[test]
fn a_warned_flag_survives_a_save() {
    let mut c = cfg_of(2, 0.7);
    c.refit_clock = Some(1e12);
    c.max_state_bytes = Some(Expectile::sums_bytes(2, 1, true, false) + 2000);
    let mut m = Expectile::new(c).unwrap();
    for (i, (x, y)) in stream_of(2, 40, 5).iter().enumerate() {
        m.stamp_next(Stamp::Ns(i as i128 * S));
        m.step(x, &[Some(*y)], 1.0, 1.0);
    }
    assert!(m.warned && m.over.is_none());
    let bytes = rmp_serde::to_vec_named(&m.state()).unwrap();
    let key = b"\xa6warned";
    assert!(bytes.windows(key.len()).any(|w| w == key));
    let back = Expectile::restore(&rmp_serde::from_slice(&bytes).unwrap()).unwrap();
    assert!(back.warned);
}

/// A span's cap is asked of the rows that join a waiting buffer, and of none
/// other: a target the row lacks, a buffer that has run.
#[test]
fn the_cap_is_asked_of_rows_that_join_a_waiting_buffer_only() {
    let data = stream_of(2, 12, 9);
    let (x, y) = &data[0];
    let tight = |m: &mut Expectile| {
        m.cfg.max_state_bytes = Some(Expectile::sums_bytes(2, 1, true, false) + stamp_bytes() + 16);
    };
    let mut m = spanned(2, 1);
    tight(&mut m);
    m.stamp_next(Stamp::Ns(0));
    m.step(x, &[None], 1.0, 1.0);
    assert_eq!(m.state_over_cap(), None, "no target, no room");
    let mut m = spanned(2, 1);
    m.cfg.refit_clock = Some(3.0);
    for (i, (x, y)) in data[..8].iter().enumerate() {
        m.stamp_next(Stamp::Ns(i as i128 * S));
        m.step(x, &[Some(*y)], 1.0, 1.0);
    }
    assert!(m.refit_status()[0].done);
    tight(&mut m);
    m.stamp_next(Stamp::Ns(9 * S));
    m.step(x, &[Some(*y)], 1.0, 1.0);
    assert_eq!(m.state_over_cap(), None, "a buffer that has run holds none");
}

/// With no feature besides the intercept there is nothing to centre: the
/// fallback solve is none.
#[test]
fn a_model_of_an_intercept_alone_has_no_centred_solve() {
    assert_eq!(centred_solve(&[1.0], &[2.0], 1, 0.1), None);
    assert_eq!(centred_solve(&[], &[], 0, 0.1), None);
    assert!(centred_solve(&[1.0, 0.5, 2.0], &[1.0, 1.0], 2, 0.1).is_some());
}

/// Where the raw system fails its pivot floor (a feature at a level of 1e6,
/// its spread 0.29 a pivot below 1e-12 of the level squared), a ridged model
/// solves about the features' means, with the penalty `r W / S` the raw
/// system would have had (`2 r` at `tau = 1/2`): the slope is the covariance
/// over the variance plus that penalty. Without a ridge the fallback is not tried.
#[test]
fn the_fallback_solve_carries_the_ridge_penalty_and_only_under_a_ridge() {
    let level = -1.0e6;
    let mut s = 17u64;
    let mut u = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    };
    let rows: Vec<(f64, f64)> = (0..50)
        .map(|_| {
            let d = u();
            (level + d, 3.0 + 2.0 * d)
        })
        .collect();
    let (x_new, d_new) = (level + 0.5, 0.5);
    let n = rows.len() as f64;
    let (mx, my) = (
        rows.iter().map(|r| r.0).sum::<f64>() / n,
        rows.iter().map(|r| r.1).sum::<f64>() / n,
    );
    let var = rows.iter().map(|r| (r.0 - mx) * (r.0 - mx)).sum::<f64>() / n;
    let cov = rows.iter().map(|r| (r.0 - mx) * (r.1 - my)).sum::<f64>() / n;
    let fit = |ridge: f64| {
        let mut c = cfg_of(1, 0.5);
        c.ridge = ridge;
        let mut m = Expectile::new(c).unwrap();
        for (x, y) in &rows {
            m.step(&[*x], &[Some(*y)], 1.0, 1.0);
        }
        m.step(&[x_new], &[Some(3.0 + 2.0 * d_new)], 1.0, 1.0).pred[0]
    };
    for ridge in [0.1, 1.0] {
        // At `tau = 1/2` every row's side weight is 1/2, so `S = W / 2`.
        let want = my + cov / (var + 2.0 * ridge) * (x_new - mx);
        let got = fit(ridge);
        assert!(
            (got - want).abs() < 1e-3 * want.abs(),
            "ridge {ridge}: {got} against {want}"
        );
    }
    assert!(fit(0.0).is_nan(), "no ridge, no fallback");
}
