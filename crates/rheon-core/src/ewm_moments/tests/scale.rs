//! Reads of a summary held at a scale (PLAN 578j, TEST-10's `ewm_moments`):
//! each kind of term at its degree in the deviations, against the same
//! model on the same stream scaled by a power of two, whose moments fit a
//! double and need no scale. The moments are homogeneous, so a term of
//! degree `p` of the scaled stream is the term of the stream times `2^(−s
//! p)`, and the mean `2^(−s)` of it, up to the parts below the smallest
//! double that both drop.

use super::*;
use crate::OnlineModel;

fn terms() -> Vec<MomentTerm> {
    vec![
        term("var", 2, MomentKind::Central, true),
        term("var_u", 2, MomentKind::Central, false),
        term("m4", 4, MomentKind::Central, true),
        term("m8", 8, MomentKind::Central, true),
        term("skew", 3, MomentKind::Standardized, true),
        term("kurt_u", 4, MomentKind::Standardized, false),
        term("mean", 1, MomentKind::Cumulant, true),
        term("k4", 4, MomentKind::Cumulant, true),
        term("k4_u", 4, MomentKind::Cumulant, false),
        term("raw2", 2, MomentKind::Raw, true),
        term("m1", 1, MomentKind::Central, true),
    ]
}

/// Each term's degree in the deviations, and the standard error's.
const DEGREE: [(Option<i32>, Option<i32>); 11] = [
    (Some(2), Some(2)),
    (Some(2), Some(2)),
    (Some(4), Some(4)),
    (Some(8), Some(8)),
    (Some(0), Some(0)),
    (Some(0), Some(0)),
    (Some(1), Some(1)),
    (Some(4), Some(4)),
    (Some(4), Some(4)),
    (Some(2), Some(2)),
    (Some(1), Some(1)),
];

/// 400 rows of heavy-tailed draws around 2, one at `1e45` at row 300.
fn rows() -> Vec<(f64, f64)> {
    let mut s = 578u64;
    (0..400)
        .map(|i| {
            let u = lcg(&mut s) - 0.5;
            let x = if i == 300 {
                1e45
            } else {
                2.0 + u / (0.05 + lcg(&mut s))
            };
            (x, 0.5 + lcg(&mut s))
        })
        .collect()
}

#[test]
fn a_term_at_a_scale_is_read_at_its_degree() {
    let shift = 150;
    let mut cfg = cfg(Decay::Halflife(30.0), terms());
    cfg.emit_se = true;
    let mut m = EwmMoments::new(cfg.clone()).unwrap();
    let mut small = EwmMoments::new(cfg).unwrap();
    let mut raised = false;
    let mut compared = 0;
    for (i, &(x, w)) in rows().iter().enumerate() {
        let d = if i == 0 { 0.0 } else { 1.0 };
        let a = m.step(&[x], &[], d, w);
        let b = small.step(&[libm::scalbn(x, -shift)], &[], d, w);
        raised |= m.sums[0].e > 0;
        assert_eq!(
            small.sums[0].e, 0,
            "row {i}: the scaled stream fits a double"
        );
        if i < 300 {
            // Before the row at 1e45 is learned the scale is 0 and the
            // outputs are the plain arithmetic's.
            assert_eq!(m.sums[0].e, 0, "row {i}");
            continue;
        }
        if i == 300 {
            // Read before the row: still the plain arithmetic's.
            continue;
        }
        assert!(m.sums[0].e > 0, "row {i}: 1e45^8 holds the scale up");
        for (k, (pa, pb)) in a.pred.iter().zip(&b.pred).enumerate() {
            let (t, se) = (k % 11, k >= 11);
            let deg = if se { DEGREE[t].1 } else { DEGREE[t].0 };
            let want = libm::scalbn(*pb, shift * deg.unwrap());
            if t == 3 && !se {
                // 1e45^8 is past a double: withheld as overflowed.
                assert!(pa.is_nan(), "row {i}: m8 is {pa}");
                assert_eq!(m.value(3), Err(Withheld::Overflow));
                continue;
            }
            if pb.is_nan() || !want.is_finite() {
                assert!(
                    pa.is_nan(),
                    "row {i}, slot {k}: {pa} where the scaled stream's is {want}"
                );
                continue;
            }
            let err = (pa - want).abs() / want.abs().max(f64::MIN_POSITIVE);
            assert!(err < 1e-12, "row {i}, slot {k}: {pa} vs {want} ({err:.1e})");
            compared += 1;
        }
    }
    assert!(raised);
    assert!(compared >= 99 * 19, "{compared} outputs compared");
    // Its state restores as itself, the scale with it.
    let back = EwmMoments::restore(&m.state()).unwrap();
    assert_eq!(back.sums[0].e, m.sums[0].e);
    assert_eq!(format!("{:?}", back.state()), format!("{:?}", m.state()));
}

/// A scale past the most a state may hold, or below 0, is damage.
#[test]
fn a_scale_out_of_range_is_refused_on_load() {
    let mut m = EwmMoments::new(cfg(Decay::Halflife(30.0), terms())).unwrap();
    for &(x, w) in &rows() {
        m.step(&[x], &[], 1.0, w);
    }
    assert!(m.sums[0].e > 0);
    for (e, ok) in [
        (0, true),
        (layout::MAX_SCALE, true),
        (layout::MAX_SCALE + 1, false),
        (-1, false),
    ] {
        let mut bad = m.clone();
        bad.sums[0].e = e;
        assert_eq!(
            EwmMoments::restore(&bad.state()).is_ok(),
            ok,
            "a scale of {e}"
        );
    }
}
