//! The 2026-11-23 review's `hmm` findings (PLAN 578f), each test first: it
//! failed on the build before the fix for the review's reason, or, marked
//! `should_panic`, holds a confirmed defect whose fix is the user's call.

use super::stabilized::Shape;
use super::*;
use crate::{OnlineModel, Stamp};

fn cfg(d: usize) -> HmmCfg {
    HmmCfg {
        n_features: d,
        k: 2,
        decay: Decay::Halflife(50.0),
        covariance: Covariance::Full,
        precision_prior: 1e-3,
        min_weight: 0.0,
        learn: true,
        transition_prior: 1.0,
        transition: None,
        means: None,
        covs: None,
        warm_rows: 10,
        seed_rule: SeedRule::First,
        seed: 7,
        tvtp: None,
        prior_rows: None,
        warm_clock: None,
    }
}

/// Two given states in one feature: state 0 of variance `g0`, state 1 of
/// variance 1, both at 0.
fn given(g0: f64, covariance: Covariance) -> HmmCfg {
    HmmCfg {
        covariance,
        means: Some(vec![0.0, 0.0]),
        covs: Some(vec![g0, 1.0]),
        ..cfg(1)
    }
}

/// SPEC-2: `tvtp_coef` is read through a softmax on every row, so a value
/// that is not finite makes every transition row NaN; it is refused, and a
/// large finite one is taken.
#[test]
fn a_tvtp_coefficient_that_is_not_finite_is_refused() {
    let with = |a: f64, b: f64| HmmCfg {
        tvtp: Some((vec![0.0, a, 0.0, 0.0], vec![0.0, 0.0, b, 0.0])),
        ..cfg(1)
    };
    for (a, b) in [
        (f64::NAN, 0.0),
        (0.0, f64::INFINITY),
        (f64::NEG_INFINITY, 1.0),
    ] {
        let e = Hmm::new(with(a, b)).unwrap_err();
        assert!(e.contains("tvtp_coef must be finite"), "{a} {b}: {e}");
    }
    assert!(Hmm::new(with(1e300, -1e300)).is_ok());
}

/// HMM-1: under given states a row whose squared deviation, in units of a
/// state's given variance, passes the largest double (`1e80` from a state
/// given `1e-150`) made `p·∞`: NaN where the state's posterior is 0, and
/// `ŝ = ∞` where it is not, which the next row turned to NaN. Either way
/// the state was one no load would take. Such a row does not move `ŝ`; it
/// ages it, as a row of no weight does, and the state saves and loads.
#[test]
fn a_deviation_past_the_doubles_leaves_the_prior_loadable() {
    for (post, covariance) in [
        ([0.0, 1.0], Covariance::Full),
        ([0.5, 0.5], Covariance::Full),
        ([0.0, 1.0], Covariance::Diagonal),
        ([0.5, 0.5], Covariance::Diagonal),
    ] {
        let mut m = Hmm::new(given(1e-150, covariance)).unwrap();
        assert!(m.stabilized().is_some_and(|s| s.shape().is_some()));
        let before = m.stabilized().unwrap().scale().1;
        let mut work = RowWork {
            post: post.to_vec(),
            ..Default::default()
        };
        m.learn_stabilized(&[1e80], 1.0, 1.0, &mut work);
        let (scale, w) = m.stabilized().unwrap().scale();
        assert!(
            scale.iter().all(|v| v.is_finite()),
            "{post:?} {covariance:?}: ŝ {scale:?}"
        );
        // Where the row is in state 0 its deviation there is past the
        // doubles, and `V` ages as at a row of no weight; where it is not,
        // state 0 adds nothing and the row is learned in state 1's units.
        if post[0] > 0.0 {
            assert!(w <= before, "{post:?}: V {w} from {before}");
        } else {
            assert!(
                w > before && scale[0] > 1e159,
                "{post:?}: V {w}, ŝ {scale:?}"
            );
        }
        let back = Hmm::restore(&m.state());
        assert!(back.is_ok(), "{post:?} {covariance:?}: {:?}", back.err());
        // And the model goes on: a row near the states reads finite.
        let mut back = back.unwrap();
        let out = back.step(&[0.5], &[], 1.0, 1.0);
        assert!(out.pred.iter().all(|v| v.is_finite()), "{:?}", out.pred);
    }
}

/// HMM-2: under given states `ψ` is a multiple of each state's given
/// covariance, 1 as given; where every `ŝ` is 0 -- a gap that leaves `V` no
/// weight, then a row exactly on the means -- it fell back to
/// `precision_prior`, a variance, and read each given covariance at a
/// thousandth of its size. It falls back to 1, the covariances as given.
#[test]
fn with_given_states_no_scale_falls_back_to_the_given_covariances() {
    let shape = Shape::of(&[2.0, 3.0], 2, 1, false);
    assert!(shape.is_some());
    let mut st = Stabilized::new(1.0, 1, Decay::Halflife(5.0), 1e-3, shape);
    st.seed(&[1.0], 2.0);
    assert_eq!(st.psi(), [1.0]);
    // Ten thousand half-lives: the slow decay leaves `V` nothing.
    st.observe(5.0 * 10.0 * 1e4, 1.0, &[0.0]);
    assert_eq!(st.scale().0, [0.0]);
    assert_eq!(st.psi(), [1.0], "under a shape ψ is a multiple of G");
    // Without a shape `ψ` is a variance, and the ridge is its fallback.
    let mut plain = Stabilized::new(1.0, 1, Decay::Halflife(5.0), 1e-3, None);
    plain.seed(&[1.0], 2.0);
    plain.observe(5.0 * 10.0 * 1e4, 1.0, &[0.0]);
    assert_eq!(plain.psi(), [1e-3]);
}

/// HMM-3: covariances given as the same anisotropic diagonal in every
/// state, `diag(1e-20, 1e20)`, took no shape, so `ψ` was the variances
/// themselves and its floor, `2⁻¹⁰⁰` of the largest, lifted `1e-20` to
/// `7.9e-11` on the first row. By the definition a state reads the
/// covariance it was given on the first row, `Ψ_s = G_s`: they keep the
/// shape, read `1e-20` and `1e20`, as the same covariances one ulp apart in
/// one state did.
#[test]
fn the_same_anisotropic_diagonal_in_every_state_reads_as_given_on_the_first_row() {
    let read = |covs: Vec<f64>| {
        let m = Hmm::new(anisotropic(covs, Covariance::Diagonal)).unwrap();
        let st = m.stabilized().unwrap();
        let g = st.shape().expect("a shape is kept");
        [st.psi()[0] * g.diag(0)[0], st.psi()[1] * g.diag(0)[1]]
    };
    let same = read(same_diagonal());
    let mut covs = same_diagonal();
    covs[7] = f64::from_bits(1e20f64.to_bits() + 1);
    let nudged = read(covs);
    assert_eq!(same, [1e-20, 1e20], "the first row reads {same:?}");
    assert_eq!(same, nudged);
}

/// `diag(1e-20, 1e20)` in both states of two features, means `(0, 0)` and
/// `(1e-10, 1e10)`.
fn same_diagonal() -> Vec<f64> {
    vec![1e-20, 0.0, 0.0, 1e20, 1e-20, 0.0, 0.0, 1e20]
}

fn anisotropic(covs: Vec<f64>, covariance: Covariance) -> HmmCfg {
    HmmCfg {
        covariance,
        means: Some(vec![0.0, 0.0, 1e-10, 1e10]),
        covs: Some(covs),
        ..cfg(2)
    }
}

/// HMM-3's flag: a state of the same anisotropic diagonal written by a
/// build that kept no shape for it holds `ŝ` as variances and no `shaped`;
/// it loads without the shape and continues as that build would, to the
/// bit, and saves without the flag again. A state this build writes keeps
/// the shape, says so, and loads with it.
#[test]
fn a_state_saved_without_the_shape_reads_on_as_it_was_saved() {
    use super::tests_shape::NO_SHAPE;
    let mut s = 23u64;
    let rows: Vec<[f64; 2]> = (0..60)
        .map(|i| {
            let (u, v) = (lcg(&mut s), lcg(&mut s));
            if (i / 15) % 2 == 0 {
                [1e-10 * u, 1e10 * v]
            } else {
                [1e-10 * (1.0 + u), 1e10 * (1.0 + v)]
            }
        })
        .collect();
    for covariance in [Covariance::Diagonal, Covariance::Full] {
        let c = anisotropic(same_diagonal(), covariance);
        // The build before: no shape, `ŝ` the given variances pooled.
        NO_SHAPE.with(|f| f.set(true));
        let mut before = Hmm::new(c.clone()).unwrap();
        for x in &rows[..30] {
            before.step(x, &[], 1.0, 1.0);
        }
        let saved = before.state();
        let mut want = Vec::new();
        for x in &rows[30..] {
            want.push(before.step(x, &[], 1.0, 1.0).pred);
        }
        NO_SHAPE.with(|f| f.set(false));
        let written = |st: &crate::State| rmp_serde::to_vec_named(st).unwrap();
        assert!(!written(&saved).windows(6).any(|w| w == b"shaped"));
        let mut back = Hmm::restore(&saved).unwrap();
        assert!(
            back.stabilized().unwrap().shape().is_none(),
            "{covariance:?}"
        );
        assert_eq!(
            written(&back.state()),
            written(&saved),
            "{covariance:?}: re-saved"
        );
        for (t, x) in rows[30..].iter().enumerate() {
            let got = back.step(x, &[], 1.0, 1.0).pred;
            assert_eq!(
                got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                want[t].iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "{covariance:?}, row {t}"
            );
        }
        // This build's state keeps the shape, and loads with it.
        let mut now = Hmm::new(c).unwrap();
        for x in &rows[..30] {
            now.step(x, &[], 1.0, 1.0);
        }
        assert!(
            written(&now.state()).windows(6).any(|w| w == b"shaped"),
            "{covariance:?}"
        );
        // Read and written again with no load between, as an export's
        // check does: the flag goes back as it was read.
        let named = rmp_serde::to_vec_named(&now).unwrap();
        let raw: Hmm = rmp_serde::from_slice(&named).unwrap();
        assert_eq!(
            rmp_serde::to_vec_named(&raw).unwrap(),
            named,
            "{covariance:?}"
        );
        let mut again = Hmm::restore(&now.state()).unwrap();
        assert!(again.stabilized().unwrap().shape().is_some());
        for x in &rows[30..] {
            let (a, b) = (again.step(x, &[], 1.0, 1.0), now.step(x, &[], 1.0, 1.0));
            assert_eq!(
                a.pred.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                b.pred.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
        }
    }
}

/// A state written under a shape is refused by a configuration that gives
/// none (the same multiple of the identity in every state): its `ŝ` is in
/// units of covariances the configuration does not hold.
#[test]
fn a_shaped_state_under_a_configuration_with_no_shape_is_refused() {
    let mut m = Hmm::new(anisotropic(same_diagonal(), Covariance::Diagonal)).unwrap();
    m.step(&[0.0, 0.0], &[], 1.0, 1.0);
    m.cfg.covs = Some(vec![2.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 2.0]);
    match Hmm::restore(&m.state()) {
        Err(crate::StateError::Invalid(e)) => assert!(e.contains("prior"), "{e}"),
        other => panic!("{other:?}"),
    }
}

fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

/// HMM-4: a row of weight 0 handed to the model directly moves the prior
/// for its read alone and puts it back; a state the move fails in is put
/// back too, so the failure is not counted: it is the next learned row's,
/// which makes the move again.
#[test]
fn a_failed_move_on_a_row_of_weight_0_is_not_counted() {
    let d = 13;
    let mut m = Hmm::new(super::tests_factor::wide(d, Decay::Halflife(50.0))).unwrap();
    for (x, dt) in super::tests_factor::blobs(d, 200, 3) {
        m.step(&x, &[], dt, 1.0);
    }
    assert!(m.factored());
    let before = m.solve_failures;
    let st = m.stabilized.as_deref_mut().unwrap();
    st.held.iter_mut().for_each(|h| *h *= 1e9);
    let saved = m.age_stabilized(0.99, &mut RowWork::default(), true);
    assert!(saved.is_some(), "the moves are handed back");
    m.restore_stabilized(saved);
    assert_eq!(
        m.solve_failures, before,
        "a row of weight 0 counted a failure"
    );
    // The same move on a learned row is counted, once a state.
    let saved = m.age_stabilized(0.99, &mut RowWork::default(), false);
    assert!(saved.is_none());
    assert_eq!(m.solve_failures, before + 2);
}

/// HMM-5 (1): a warm-up over a span of the clock is refused beside given
/// states, which are not seeded from the first rows, as the spec refuses it.
#[test]
fn a_warm_span_beside_given_states_is_refused() {
    let mut c = given(2.0, Covariance::Full);
    c.warm_clock = Some(5.0);
    let e = Hmm::new(c.clone()).unwrap_err();
    assert!(e.contains("warm_span does not apply"), "{e}");
    c.warm_clock = Some(f64::NAN);
    assert!(Hmm::new(c).is_err());
}

/// HMM-5 (2): a state whose place on the clock does not match its
/// warm-up -- one with a span and no place, or a place and no span -- is
/// damaged, and refused on load, as `bocpd`'s is.
#[test]
fn a_restored_place_matches_the_warm_span() {
    let span = HmmCfg {
        warm_clock: Some(5.0),
        warm_rows: 2,
        ..cfg(1)
    };
    for c in [span, cfg(1)] {
        let mut m = Hmm::new(c).unwrap();
        m.step(&[0.5], &[], 1.0, 1.0);
        assert!(Hmm::restore(&m.state()).is_ok());
        m.place = match m.place {
            Some(_) => None,
            None => Some(Place::default()),
        };
        match Hmm::restore(&m.state()) {
            Err(crate::StateError::Invalid(e)) => assert!(e.contains("wrong shape"), "{e}"),
            other => panic!("{other:?}"),
        }
    }
}

/// HMM-5 (3): the row that seeds the states from a warm-up over a span of
/// the clock is the row that took the buffer past half its cap, and the
/// half-cap warning is raised on it. The seeding emptied the buffer before
/// the bytes were read, so it read 0 and said nothing.
#[test]
fn the_seeding_row_raises_the_half_cap_warning() {
    let c = HmmCfg {
        warm_clock: Some(10.0),
        warm_rows: 2,
        ..cfg(2)
    };
    let mut m = Hmm::new(c).unwrap();
    // Eleven rows, at 0 to 10: the eleventh seeds. The cap is one byte
    // short of twice its buffer, so only that row passes half of it.
    let cap = 2 * m.warm_bytes(11) - 1;
    assert!(2 * m.warm_bytes(10) <= cap);
    m.set_cap_for_test(cap);
    for i in 0..11 {
        m.stamp_next(Stamp::from(i as f64));
        let x = [(i % 3) as f64, (i % 4) as f64 - 1.0];
        m.step(&x, &[], 1.0, 1.0);
        if i < 10 {
            assert!(m.take_state_warning().is_none(), "row {i}");
        }
    }
    assert!(m.seeded);
    assert_eq!(m.take_state_warning(), Some((m.warm_bytes(11), cap)));
}

/// HMM-5 (4) and BOCPD-4: the compact (positional) msgpack encoding does
/// not round-trip a state whose skipped field is not its last: `hmm` under
/// `tied` writes no prior and then its warm-up's place. Every file the
/// library writes is named msgpack, which does, so this is the user's call
/// (writing every field always changes the bytes of every saved state).
#[test]
#[should_panic(expected = "compact")]
fn a_compact_state_with_a_skipped_middle_field_round_trips() {
    let mut m = Hmm::new(HmmCfg {
        covariance: Covariance::Shared,
        warm_clock: Some(5.0),
        warm_rows: 2,
        ..cfg(1)
    })
    .unwrap();
    m.stamp_next(Stamp::from(0.0));
    m.step(&[0.5], &[], 1.0, 1.0);
    assert!(m.stabilized.is_none() && m.place.is_some());
    let bytes = rmp_serde::to_vec_named(&m).unwrap();
    let named: Hmm = rmp_serde::from_slice(&bytes).unwrap();
    assert_eq!(rmp_serde::to_vec_named(&named).unwrap(), bytes);
    let compact = rmp_serde::from_slice::<Hmm>(&rmp_serde::to_vec(&m).unwrap());
    let back = compact
        .as_ref()
        .map(|c| rmp_serde::to_vec_named(c).unwrap());
    assert!(
        back.as_ref().is_ok_and(|b| *b == bytes),
        "the compact encoding read back {:?}",
        compact.err()
    );
}

/// STATE-3: impossible weights, transition counts, posterior entries, or
/// per-state accumulator weights in an `hmm` state are refused on load;
/// a buffered row whose weight decayed to 0 is still accepted.
#[test]
fn state_3_a_weight_no_stream_writes_is_refused_on_load() {
    fn mutate_msgpack_state_field(m: &Hmm, list_key: &str, field: &str, val: rmpv::Value) -> Hmm {
        let bytes = rmp_serde::to_vec_named(m).unwrap();
        let mut v = rmpv::decode::read_value(&mut bytes.as_slice()).unwrap();
        let rmpv::Value::Map(entries) = &mut v else {
            panic!("map");
        };
        let (_, list) = entries
            .iter_mut()
            .find(|(k, _)| k.as_str() == Some(list_key))
            .unwrap();
        let rmpv::Value::Array(items) = list else {
            panic!("array");
        };
        let rmpv::Value::Map(st) = &mut items[0] else {
            panic!("state map");
        };
        let (_, slot) = st
            .iter_mut()
            .find(|(k, _)| k.as_str() == Some(field))
            .unwrap();
        *slot = val;
        let mut out = Vec::new();
        rmpv::encode::write_value(&mut out, &v).unwrap();
        rmp_serde::from_slice(&out).unwrap()
    }

    for (cov, list_key) in [
        (Covariance::Full, "states"),
        (Covariance::Diagonal, "diag_states"),
    ] {
        let mut m = Hmm::new(given(2.0, cov)).unwrap();
        for i in 0..8 {
            m.step(&[if i % 2 == 0 { -2.0 } else { 2.0 }], &[], 1.0, 1.0);
        }
        assert!(Hmm::restore(&m.state()).is_ok(), "{cov:?}");

        for bad in [-1.0, f64::NAN, f64::INFINITY] {
            for (name, edit) in [
                ("n_eff", (|h: &mut Hmm, v| h.n_eff = v) as fn(&mut Hmm, f64)),
                ("p[0]", |h: &mut Hmm, v| h.p[0] = v),
                ("a[0]", |h: &mut Hmm, v| h.a[0] = v),
            ] {
                let mut damaged = m.clone();
                edit(&mut damaged, bad);
                match Hmm::restore(&damaged.state()) {
                    Err(crate::StateError::Invalid(e)) => {
                        assert!(e.contains("damaged"), "{cov:?} {name} {bad}: {e}");
                    }
                    other => panic!("{cov:?} {name} = {bad}: {other:?}"),
                }
            }
            for field in ["w_sum", "precision_scale", "q_sum"] {
                let damaged =
                    mutate_msgpack_state_field(&m, list_key, field, rmpv::Value::F64(bad));
                match Hmm::restore(&damaged.state()) {
                    Err(crate::StateError::Invalid(e)) => {
                        assert!(
                            e.contains("damaged"),
                            "{cov:?} {list_key}.{field} {bad}: {e}"
                        );
                    }
                    other => panic!("{cov:?} {list_key}.{field} = {bad}: {other:?}"),
                }
            }
        }

        let mut zero_p = m.clone();
        zero_p.p.fill(0.0);
        assert!(Hmm::restore(&zero_p.state()).is_err(), "{cov:?} zero p");

        let mut seeded_with_buf = m.clone();
        seeded_with_buf.buffer.push((vec![0.0], 1.0));
        assert!(
            Hmm::restore(&seeded_with_buf.state()).is_err(),
            "{cov:?} seeded with buffer"
        );
    }

    // Unseeded buffer: decayed-to-zero weight is legal; negative/NaN/inf refused.
    let mut unseeded = Hmm::new(cfg(1)).unwrap();
    unseeded.step(&[0.5], &[], 1.0, 1.0);
    assert!(!unseeded.seeded && unseeded.buffer.len() == 1);
    let mut zero_w = unseeded.clone();
    zero_w.buffer[0].1 = 0.0;
    assert!(Hmm::restore(&zero_w.state()).is_ok());
    for bad in [-1.0, f64::NAN, f64::INFINITY] {
        let mut damaged = unseeded.clone();
        damaged.buffer[0].1 = bad;
        assert!(Hmm::restore(&damaged.state()).is_err(), "buffer w {bad}");
    }

    // Factor-mode `Hmm`: `EwFactorState` refuses negative/non-finite weights
    // and ridge scales at deserialization time.
    let d = Hmm::FACTOR_FROM;
    let mut fm = Hmm::new(cfg(d)).unwrap();
    for i in 0..40 {
        let x: Vec<f64> = (0..d)
            .map(|j| ((i * (j + 1) + j) as f64 * 0.37).sin())
            .collect();
        fm.step(&x, &[], 1.0, 1.0);
    }
    assert!(fm.factored(), "expected factor_states at d = {d}");
    assert!(Hmm::restore(&fm.state()).is_ok());
    let fm_bytes = rmp_serde::to_vec_named(&fm).unwrap();
    for bad in [-1.0, f64::NAN, f64::INFINITY] {
        for field in ["w_sum", "precision_scale"] {
            let mut v = rmpv::decode::read_value(&mut fm_bytes.as_slice()).unwrap();
            let rmpv::Value::Map(entries) = &mut v else {
                panic!("map");
            };
            let (_, list) = entries
                .iter_mut()
                .find(|(k, _)| k.as_str() == Some("factor_states"))
                .unwrap();
            let rmpv::Value::Array(items) = list else {
                panic!("array");
            };
            let rmpv::Value::Map(st) = &mut items[0] else {
                panic!("state map");
            };
            let (_, slot) = st
                .iter_mut()
                .find(|(k, _)| k.as_str() == Some(field))
                .unwrap();
            *slot = rmpv::Value::F64(bad);
            let mut out = Vec::new();
            rmpv::encode::write_value(&mut out, &v).unwrap();
            assert!(
                rmp_serde::from_slice::<Hmm>(&out).is_err(),
                "factor_states.{field} = {bad} should fail deserialization"
            );
        }
    }
}
