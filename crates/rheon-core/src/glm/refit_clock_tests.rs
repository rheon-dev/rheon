//! The refit over a span of the clock (`refit_clock`, docs/PLAN.md task
//! 538): `refit_rows`' refit closed by the first row at least the span after
//! a target's first learned row, on the exact stamps; its buffers checked
//! against the group's cap as they fill.

use super::tests::{cfg_of, stream_of};
use super::*;

const S: i128 = 1_000_000_000;

fn bits(m: &mut Glm, x: &[f64], y: f64) -> Vec<u64> {
    let mut out: Vec<u64> = m
        .step(x, &[Some(y)], 1.0, 1.0)
        .pred
        .iter()
        .map(|v| v.to_bits())
        .collect();
    out.extend(m.coefficients()[0].iter().map(|v| v.to_bits()));
    out
}

/// On the row clock and on a one-second temporal clock a span of `D` holds
/// a target's first `D + 1` rows: `refit_rows = D + 1`'s refit and every
/// output to the bit, for each family, with and without decay.
#[test]
fn a_span_of_d_is_d_plus_one_rows_on_a_regular_clock() {
    for family in [GlmFamily::Poisson, GlmFamily::Gamma] {
        for lam in [1.0, 0.995] {
            let d = 30.0;
            let mut rows = cfg_of(family, 2, d as u32 + 1);
            rows.decay = Decay::Lam(lam);
            let mut span = rows.clone();
            span.refit_rows = 0;
            span.refit_clock = Some(d);
            for temporal in [false, true] {
                let (mut a, mut b) = (
                    Glm::new(rows.clone()).unwrap(),
                    Glm::new(span.clone()).unwrap(),
                );
                for (i, (x, y)) in stream_of(family, 3, 90, 31).iter().enumerate() {
                    let n = i as i128 + 1;
                    if temporal {
                        b.stamp_next(Stamp::Ns(n * S));
                    } else {
                        b.stamp_next_on_rows(Stamp::from(n as f64));
                    }
                    assert_eq!(
                        bits(&mut a, x, *y),
                        bits(&mut b, x, *y),
                        "{family:?}, lam {lam}, temporal {temporal}, row {i}"
                    );
                    assert_eq!(
                        a.refit_status()[0].done,
                        b.refit_status()[0].done,
                        "row {i}"
                    );
                }
                assert!(b.refit_status()[0].done);
            }
        }
    }
}

/// Both sides of the span's end, and the wait for a row per coefficient
/// where the span runs first.
#[test]
fn the_span_closes_at_its_length_and_not_a_nanosecond_before() {
    let mut c = cfg_of(GlmFamily::Poisson, 2, 0);
    c.refit_clock = Some(10.0);
    let data = stream_of(GlmFamily::Poisson, 3, 8, 32);
    for (last, done) in [(10 * S - 1, false), (10 * S, true)] {
        let mut m = Glm::new(c.clone()).unwrap();
        for (i, (x, y)) in data[..5].iter().enumerate() {
            m.stamp_next(Stamp::Ns(if i == 4 { last } else { i as i128 * S }));
            m.step(x, &[Some(*y)], 1.0, 1.0);
        }
        assert_eq!(m.refit_status()[0].done, done, "last at {last}");
    }
    let mut few = Glm::new(c).unwrap();
    for (i, (x, y)) in data[..3].iter().enumerate() {
        few.stamp_next(Stamp::Ns([0, 20, 21][i] * S));
        few.step(x, &[Some(*y)], 1.0, 1.0);
        assert_eq!(few.refit_status()[0].done, i == 2, "row {i}");
    }
}

/// Past half of the group's refit cap a warning, once; the row that would
/// pass it is learned by no target, nor any row after it.
#[test]
fn the_spans_rows_are_warned_of_at_half_the_cap_and_stopped_at_it() {
    let mut c = cfg_of(GlmFamily::Poisson, 2, 0);
    c.refit_clock = Some(1e12);
    let mut m = Glm::new(c).unwrap();
    let row = 8 * (2 + 2) as u64;
    let cap = GLM_GROUP_REFIT_CAP as u64;
    let fits = (cap - m.refit_held_bytes(0)) / row;
    let data = stream_of(GlmFamily::Poisson, 3, fits as usize + 50, 33);
    let mut warned = 0;
    for (i, (x, y)) in data.iter().enumerate() {
        m.stamp_next(Stamp::Ns(i as i128 * S));
        m.step(x, &[Some(*y)], 1.0, 1.0);
        if let Some((bytes, got)) = m.take_state_warning() {
            warned += 1;
            assert_eq!(got, cap);
            assert!(2 * bytes > cap && 2 * (bytes - row) <= cap, "{bytes}");
        }
    }
    assert_eq!(warned, 1);
    let (bytes, got) = m.state_over_cap().expect("stopped at the cap");
    assert!(bytes > cap && bytes <= cap + row && got == cap);
    assert_eq!(u64::from(m.refit_status()[0].learned), fits);
    let before = m.coefficients();
    let n_eff = m.n_eff();
    let (x, y) = &data[0];
    m.stamp_next(Stamp::Ns(1_000_000 * S));
    m.step(x, &[Some(*y)], 1.0, 1.0);
    assert_eq!(m.coefficients(), before, "nothing learned past the cap");
    assert_eq!(m.n_eff(), n_eff);
}

/// A state saved mid-span, through the file's encoding, goes on to the bit;
/// a state without a span writes none of its fields.
#[test]
fn a_state_saved_mid_span_goes_on_to_the_bit() {
    let mut c = cfg_of(GlmFamily::Gamma, 2, 0);
    c.refit_clock = Some(40.0);
    let data = stream_of(GlmFamily::Gamma, 3, 100, 34);
    let mut a = Glm::new(c).unwrap();
    for (x, y) in &data[..20] {
        // No stamps: the clock summed from the steps is the row's place,
        // and it is state.
        a.step(x, &[Some(*y)], 1.0, 1.0);
    }
    let bytes = rmp_serde::to_vec_named(&a.state()).unwrap();
    let mut b = Glm::restore(&rmp_serde::from_slice(&bytes).unwrap()).unwrap();
    for (x, y) in &data[20..] {
        assert_eq!(bits(&mut a, x, *y), bits(&mut b, x, *y));
    }
    assert!(b.refit_status()[0].done);
    let plain =
        rmp_serde::to_vec_named(&Glm::new(cfg_of(GlmFamily::Gamma, 2, 10)).unwrap().state())
            .unwrap();
    // Each key as msgpack writes it: its length in a fixstr's byte, then
    // the name ("over" is also the start of `over_budget`).
    for f in ["refit_clock", "first", "own", "over", "warned"] {
        let mut key = vec![0xa0 | f.len() as u8];
        key.extend_from_slice(f.as_bytes());
        assert!(!plain.windows(key.len()).any(|w| w == key), "{f}");
    }
}

#[test]
fn the_span_is_checked() {
    let refuse = |c: GlmCfg, words: &str| {
        let e = Glm::new(c).unwrap_err();
        assert!(e.contains(words), "{e}");
    };
    for d in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        let mut c = cfg_of(GlmFamily::Poisson, 1, 0);
        c.refit_clock = Some(d);
        refuse(c, "refit_span must be a finite number");
    }
    let mut c = cfg_of(GlmFamily::Poisson, 1, 10);
    c.refit_clock = Some(5.0);
    refuse(c, "two refits of the first rows");
    // A link with no mean at 0 needs a refit: a span is one.
    let mut c = cfg_of(GlmFamily::Gamma, 1, 0);
    c.link = GlmLink::Inverse;
    c.refit_clock = Some(5.0);
    assert!(Glm::new(c.clone()).is_ok());
    c.refit_clock = None;
    refuse(c, "needs refit_rows above 0");
}
