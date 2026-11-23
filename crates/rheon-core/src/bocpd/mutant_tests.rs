//! Tests that hold each line of `bocpd` the changed-lines mutation pass found
//! no test for (the survivors of W11's push): a saved state's shape checks one
//! at a time, and the warm-up's byte counts.

use super::tests::{Normals, cfg};
use super::*;
use crate::{ModelState, OnlineModel, StateError};

fn warming(d: usize) -> BocpdCfg {
    BocpdCfg {
        prior_mean: None,
        prior_scale: None,
        prior_nu: None,
        ..cfg(d)
    }
}

/// A model after `n` rows of a two-feature stream.
fn stepped(c: BocpdCfg, n: usize) -> Bocpd {
    let mut g = Normals::new(11);
    let mut m = Bocpd::new(c).unwrap();
    for _ in 0..n {
        m.step(&[g.normal(), g.normal()], &[], 1.0, 1.0);
    }
    m
}

/// The state of `m` after `damage` has touched it, restored.
fn restored(m: &Bocpd, damage: impl FnOnce(&mut Bocpd)) -> Result<Bocpd, StateError> {
    let mut s = m.state();
    let ModelState::Bocpd(inner) = &mut s.model else {
        unreachable!()
    };
    damage(inner);
    Bocpd::restore(&s)
}

fn refused(m: &Bocpd, what: &str, damage: impl FnOnce(&mut Bocpd)) {
    match restored(m, damage) {
        Err(StateError::Invalid(e)) => assert!(e.contains("wrong shape"), "{what}: {e}"),
        other => panic!("{what}: {:?}", other.map(|_| ())),
    }
}

fn warm_row() -> WarmRow {
    WarmRow {
        x: vec![0.0, 0.0],
        hazard: None,
        d_clock: 1.0,
        weight: 1.0,
    }
}

/// Each shape check on its own: a state damaged in one way and no other is
/// refused, and undamaged it loads. `||` joins the checks, so a test that
/// damages two at once (as a `logjoint` push beside an empty run list does)
/// cannot tell a check that stopped counting.
#[test]
fn each_shape_check_of_a_saved_state_refuses_alone() {
    let warm = stepped(warming(2), 2);
    let done = stepped(warming(2), 12);
    let span = stepped(
        BocpdCfg {
            warm_clock: Some(1e9),
            ..warming(2)
        },
        2,
    );
    assert!(warm.warming() && !done.warming());
    assert!(!warm.warm.is_empty() && done.warm.is_empty() && done.data_prior.is_some());
    for (name, m) in [("warm", &warm), ("done", &done), ("span", &span)] {
        assert!(restored(m, |_| {}).is_ok(), "{name} loads undamaged");
    }

    refused(&done, "no run", |m| {
        m.runs.clear();
        m.logjoint.clear();
    });
    refused(&done, "a log-joint short", |m| {
        m.logjoint.pop();
    });
    refused(&done, "a run's mean", |m| m.runs[0].mean.push(0.0));
    refused(&done, "a run's scatter", |m| m.runs[0].m2.push(0.0));
    refused(&done, "a run's low mean", |m| m.runs[0].mean_lo.push(0.0));
    refused(&done, "the prior's mean", |m| {
        m.data_prior.as_mut().unwrap().mean.push(0.0);
    });
    refused(&done, "the prior's scale", |m| {
        m.data_prior.as_mut().unwrap().scale.push(0.0);
    });
    refused(&warm, "a warm-up row's width", |m| {
        m.warm[0].x.push(0.0);
    });
    // Rows held after the prior is set.
    refused(&done, "warm rows beside the prior", |m| {
        m.warm.push(warm_row());
    });
    // A full buffer that should have set the prior, on rows (a span's is
    // closed by the clock, not by a count).
    refused(&warm, "a full warm-up buffer", |m| {
        while m.warm.len() < m.cfg.warm_rows_or_default() {
            m.warm.push(warm_row());
        }
    });
    let not_full = |m: &mut Bocpd| {
        while m.warm.len() < m.cfg.warm_rows_or_default() - 1 {
            m.warm.push(warm_row());
        }
    };
    assert!(restored(&warm, not_full).is_ok(), "one short of full loads");
    assert!(
        restored(&span, |m| {
            for _ in 0..8 {
                m.warm.push(warm_row());
            }
        })
        .is_ok(),
        "a span's buffer is not bounded by a count"
    );
    // The place of a span: held exactly where the cfg has one.
    refused(&span, "a span without its place", |m| m.place = None);
    refused(&warm, "a place without a span", |m| {
        m.place = Some(Place::default());
    });
}

#[test]
fn the_warm_up_counts_its_bytes_by_row_and_width() {
    for d in [2usize, 3, 5] {
        let m = Bocpd::new(BocpdCfg {
            warm_clock: Some(5.0),
            ..warming(d)
        })
        .unwrap();
        for rows in [0u64, 1, 5, 7] {
            assert_eq!(m.warm_bytes(rows as usize), rows * (d as u64 + 4) * 8);
        }
    }
    let m = Bocpd::new(warming(2)).unwrap();
    assert_eq!(m.warm_cap(), 256 * 1024 * 1024);
}
