// ---------------------------------------------------------------------------
// `predict` is the step without the step (docs/ENHANCEMENTS.md E31).
// ---------------------------------------------------------------------------

/// Equal slot by slot, with NaN equal to NaN: the "not ready" marker has to
/// agree too.
fn same_step(kind: &str, i: usize, p: &Step, s: &Step) {
    assert_eq!(p.pred.len(), s.pred.len(), "{kind}: slot count at row {i}");
    for (slot, (a, b)) in p.pred.iter().zip(&s.pred).enumerate() {
        assert!(
            a == b || (a.is_nan() && b.is_nan()),
            "{kind}: slot {slot} at row {i}: predict said {a}, step said {b}"
        );
    }
    assert!(
        p.n_eff == s.n_eff,
        "{kind}: n_eff at row {i}: predict said {}, step said {}",
        p.n_eff,
        s.n_eff
    );
    assert_eq!(p.extra, s.extra, "{kind}: extra at row {i}");
}

/// Over a stream with missing targets, zero-weight rows, uneven weights and
/// clock gaps, `predict(x, d)` called before each `step(x, y, d, w)` must
/// return exactly what the step returns -- the same numbers, not close ones --
/// and, being `&self`, it cannot have moved the state. That equality is the
/// whole definition of `predict`; a model that computes its prediction any
/// other way in one of the two places fails here.
fn predict_is_the_step_without_the_step<M: OnlineModel + Clone>(
    build: impl Fn() -> M,
    targets: usize,
    binary: bool,
) {
    predict_is_the_step_naming_waits(build, targets, binary, None);
}

/// Target `j`'s reason for reporting nothing, as the model names it, or
/// `None` where it is not withholding.
type Waits<'a, M> = &'a dyn Fn(&M, usize) -> Option<&'static str>;

/// The parity above; with `waits`, a row whose slots are not all ready
/// counts towards the ready count where every slot that is NaN is a
/// target the model names a reason for before the row (a start not yet
/// accepted): the row is accounted for, not lost.
fn predict_is_the_step_naming_waits<M: OnlineModel + Clone>(
    build: impl Fn() -> M,
    targets: usize,
    binary: bool,
    waits: Option<Waits<'_, M>>,
) {
    let mut m = build();
    let kind = m.state().model.kind();
    // A twin stepped through `step_into` on one kept buffer, as the bank
    // steps every model (task 270, M-4).
    let mut twin = build();
    let mut kept = Step::default();
    let mut s = 20260902u64;
    let mut ready = 0usize;
    // The rows every slot of which is a number, waits or not.
    let mut live = 0usize;
    // `garch` models a variance, and its start waits on a target with none
    // (PLAN 548): its targets are each a GARCH(1,1)'s returns at (0.05,
    // 0.6, 0.35), each row's variance and return kept here.
    let mut returns = vec![(1.0f64, 0.0f64); targets];
    for i in 0..400 {
        let x: Vec<f64> = (0..K).map(|_| lcg(&mut s) * 3.0).collect();
        let y: Vec<Option<f64>> = (0..targets)
            .map(|j| {
                if lcg(&mut s) > 0.8 {
                    return None;
                }
                if kind == "garch" {
                    let (v, e) = &mut returns[j];
                    *v = 0.05 + 0.6 * *e * *e + 0.35 * *v;
                    let z = (0..12).map(|_| 0.5 * lcg(&mut s)).sum::<f64>();
                    *e = v.sqrt() * z;
                    return Some(*e);
                }
                let lin = 0.5 * (j as f64 + 1.0) + x[0] - 0.5 * x[1] + 0.1 * lcg(&mut s);
                Some(if binary { f64::from(lin > 0.5) } else { lin })
            })
            .collect();
        let u = lcg(&mut s);
        let w = if u < -0.9 {
            0.0
        } else if u < -0.5 {
            0.5
        } else if u > 0.7 {
            2.0
        } else {
            1.0
        };
        let d = match (i, lcg(&mut s)) {
            (0, _) => 0.0,
            (_, v) if v > 0.95 => 25.0,
            (_, v) if v > 0.8 => 3.0,
            _ => 1.0,
        };
        // `predict_with`, not `predict`: two models read a number out of
        // the targets slot, and the parity that matters is against the
        // answer the step gives for the same row (C1).
        let named: Vec<Option<&'static str>> = match waits {
            Some(f) => (0..targets).map(|j| f(&m, j)).collect(),
            None => vec![None; targets],
        };
        let p = m.predict_with(&x, &y, d);
        let step = m.step(&x, &y, d, w);
        same_step(kind, i, &p, &step);
        if i % 3 == 0 {
            kept = stale_step(i / 3);
        }
        twin.step_into(&x, &y, d, w, &mut kept);
        assert!(
            same_bits(&kept, &step),
            "{kind}: row {i}: step_into wrote {kept:?}, step returned {step:?}"
        );
        assert!(
            state_bytes(&twin) == state_bytes(&m),
            "{kind}: row {i}: step_into left another state than step"
        );
        let flagged_only = |slot: usize| ONLY_ON_A_FLAG.contains(&(kind, slot));
        if step
            .pred
            .iter()
            .enumerate()
            .all(|(slot, v)| v.is_finite() || flagged_only(slot))
        {
            ready += 1;
            live += 1;
        } else if waits.is_some()
            && step.pred.iter().enumerate().all(|(slot, v)| {
                v.is_finite() || named.get(slot).copied().flatten().is_some_and(|r| !r.is_empty())
            })
        {
            ready += 1;
        }
    }
    // The stream must actually have exercised the ready path, or the test
    // would pass on NaN == NaN alone. A model that reports only where a
    // span closes has far fewer such rows, and that is the point of it:
    // `SPARSE_OUTPUT` says how many are enough.
    let want = if SPARSE_OUTPUT.contains(&kind) {
        10
    } else if kind == "pca" {
        // This stream's two columns are independent and of one spread, so
        // its components are tied, and `pca` withholds a tie: it reports
        // nothing here by design. `streampca/tests.rs`'s
        // `predict_is_the_step_on_a_separated_stream` holds the ready path
        // on a stream whose components differ, with the same weights and
        // gaps.
        0
    } else {
        300
    };
    assert!(
        ready > want || (want == 0 && ready == 0),
        "{kind}: only {ready} rows had every slot ready or named its wait"
    );
    // Where waits count, the ready path is exercised still: at least a
    // third of the stream reports every slot.
    assert!(
        waits.is_none() || 3 * live > 400,
        "{kind}: only {live} rows had every slot ready"
    );
    // Of mixed signs, and all positive and all negative at a level of 1,000
    // (docs/PLAN.md task 209 (a)'s one-sided streams).
    for level in [0.0, 1e3, -1e3] {
        zero_weight_rows_leave_no_nan(&build, targets, binary, kind, level);
    }
    a_zero_weight_row_past_the_underflow_forgets(&build, targets, binary, kind);
    unusable_values_are_refused(&build, targets, binary, kind);
}

/// Task 115 (c), decided 2026-09-28: a zero-weight row whose decay
/// underflows to exactly 0 -- `2^-1075` rounds to 0, so from 1075 half-lives
/// on -- forgets the history, as the decay does. It used to keep it: the
/// mean-form update is `0/0` there, the guard skipped the row whole, and the
/// next row saw the old count (PLAN §12). "As the decay does" is the same row
/// one half-life short, whose factor `2^-1074` is the smallest double: it
/// ages the history to nothing, and the two streams must agree from there
/// on. The row after the one that forgot reports no weight, unless the
/// model's weight does not decay at all.
///
/// Predictions are compared too, except where the reference is not "the
/// history aged to nothing": `kalman` adds `Q·d` to its covariance and
/// `holt` carries its level along its trend, so a half-life more of gap is a
/// different state for both; `ew_class` counts a class of subnormal weight
/// as present and one of weight 0 as absent, a threshold rather than an age;
/// and `hmm` one half-life short is not a sane state -- a subnormal history
/// share leaves its co-moments and precision prior a few bits, and its
/// densities NaN for 39 of the next 40 rows, which do not count towards its
/// `n_eff` either (from about 1025 half-lives; PLAN task 115 (c), measured
/// 2026-09-28 and raised, not fixed). `hmm` keeps the first check alone.
/// A `lasso` that solves on the clock is compared as every other model is:
/// it solves on the first row after the gap, from that row alone, and one
/// half-life short the history's remains, `2^-1074` of its weight, gave
/// each feature a spread of its own, which the standardized descent divided
/// by and so read the old fit back -- 4.66 apart on this stream
/// (docs/PLAN.md task 214, raised there). A history in the subnormal range
/// of the row it meets is forgotten now, as the decay that underflows
/// forgets it (task 215).
fn a_zero_weight_row_past_the_underflow_forgets<M: OnlineModel>(
    build: &impl Fn() -> M,
    targets: usize,
    binary: bool,
    kind: &'static str,
) {
    // `hawkes` reads the clock without an event as evidence -- `G₀₀` is the
    // time observed -- so where nothing decays (its window) a half-life more
    // of gap is a longer quiet spell and a lower base rate.
    const NOT_COMPARED: [&str; 5] = ["kalman", "holt", "ew_class", "hmm", "hawkes"];
    let row = |s: &mut u64| {
        let x: Vec<f64> = (0..K).map(|_| lcg(s) * 3.0).collect();
        let y: Vec<Option<f64>> = (0..targets)
            .map(|j| {
                let lin = 0.5 * (j as f64 + 1.0) + x[0] - 0.5 * x[1] + 0.1 * lcg(s);
                Some(if binary { f64::from(lin > 0.5) } else { lin })
            })
            .collect();
        (x, y)
    };
    let run = |gap: f64| -> Vec<Step> {
        let mut m = build();
        let mut s = 20260928u64;
        for i in 0..40 {
            let (x, y) = row(&mut s);
            m.step(&x, &y, if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        let (x, y) = row(&mut s);
        m.step(&x, &y, gap, 0.0);
        (0..40)
            .map(|_| {
                let (x, y) = row(&mut s);
                m.step(&x, &y, 1.0, 1.0)
            })
            .collect()
    };
    assert_eq!(decay().factor(1075.0 * HALFLIFE), 0.0);
    assert!(decay().factor(1074.0 * HALFLIFE) > 0.0);
    let forgot = run(1075.0 * HALFLIFE);
    let aged = run(1074.0 * HALFLIFE);
    assert!(
        forgot[0].n_eff == 0.0 || forgot[0].n_eff == aged[0].n_eff,
        "{kind}: the row after a zero-weight row 1075 half_lives on reports n_eff {}, a \
         history the decay forgot",
        forgot[0].n_eff
    );
    let close = |a: f64, b: f64| {
        a == b || (a.is_nan() && b.is_nan()) || (a - b).abs() <= 1e-9 * (1.0 + b.abs())
    };
    if kind == "hmm" {
        return;
    }
    let exempt = NOT_COMPARED.contains(&kind);
    for (i, (a, b)) in forgot.iter().zip(&aged).enumerate() {
        let preds = exempt
            || a.pred.len() == b.pred.len()
                && a.pred.iter().zip(&b.pred).all(|(p, q)| close(*p, *q));
        assert!(
            close(a.n_eff, b.n_eff) && preds,
            "{kind}: row {i} after the zero-weight row: n_eff {} and {:?} where its decay \
             underflowed, n_eff {} and {:?} one half_life short of that",
            a.n_eff,
            a.pred,
            b.n_eff,
            b.pred
        );
    }
}

/// Hard rule 9 for a model handed a row of weight 0 directly: the row,
/// the first with a prediction among them, divides no 0/0 and leaves no NaN
/// behind, so every row the stream without it predicts, the stream with it
/// predicts too, and its weight stays finite. The bank hands a model no such
/// row: a row of weight 0 counts in no clock, and the next row steps from
/// the last that counted, which `crates/rheon-polars/tests/zero_weight.rs`
/// holds for every kind against the stream without such rows, to the bit
/// on every OS (PLAN 539). The numbers themselves are therefore not held
/// here: this stream carries the zero-weight row's step into the next as a
/// sum, and a model's decay over two steps differs from its decay over
/// their sum in the last bits, by more on some OSes (`glm` at a level of
/// 1,000 parted by 1.5e-9 on Windows).
fn zero_weight_rows_leave_no_nan<M: OnlineModel>(
    build: &impl Fn() -> M,
    targets: usize,
    binary: bool,
    kind: &'static str,
    level: f64,
) {
    let side = |v: f64| {
        if level == 0.0 {
            v
        } else {
            level.signum() * v.abs() + level
        }
    };
    let row = |s: &mut u64, i: usize| {
        let raw: Vec<f64> = (0..K).map(|_| lcg(s) * 3.0).collect();
        let x: Vec<f64> = raw.iter().map(|&v| side(v)).collect();
        let y: Vec<Option<f64>> = (0..targets)
            .map(|j| {
                let label = 0.5 * (j as f64 + 1.0) + raw[0] - 0.5 * raw[1];
                let lin = 0.5 * (j as f64 + 1.0) + x[0] - 0.5 * x[1];
                Some(if binary { f64::from(label > 0.5) } else { lin })
            })
            .collect();
        let d = match i {
            0 => 0.0,
            _ if i.is_multiple_of(5) => 3.0,
            _ => 1.0,
        };
        (x, y, d)
    };
    // Every seventh row has weight 0, and so does the first row with a
    // prediction -- where a weight a model keeps beside `n_eff` can still be
    // 0, and a zero-weight row forms `0/0` in any division left unguarded
    // (review 2026-09-12, C7: `lasso`'s selection). That row is found on a
    // probe of the same stream.
    let zero = |i: usize| i % 7 == 3;
    let first = {
        let (mut probe, mut s) = (build(), 20260906u64);
        (0..80).find(|&i| {
            let (x, y, d) = row(&mut s, i);
            let w = if zero(i) { 0.0 } else { 1.0 };
            probe.step(&x, &y, d, w).pred.iter().any(|p| p.is_finite())
        })
    };
    let (mut with, mut without) = (build(), build());
    let mut s = 20260906u64;
    let mut carried = 0.0;
    for i in 0..80 {
        let (x, y, d) = row(&mut s, i);
        if zero(i) || first == Some(i) {
            with.step(&x, &y, d, 0.0);
            carried += d;
            continue;
        }
        let a = with.step(&x, &y, d, 1.0);
        let b = without.step(&x, &y, d + carried, 1.0);
        // `n_eff` is read *before* the row's own decay, so on the row that
        // carries the skipped one's delta the two are one decay apart by
        // construction; they must agree on every other row, and the row
        // after the carry is the one that says the streams re-converged.
        let compare = carried == 0.0;
        carried = 0.0;
        assert!(
            a.n_eff.is_finite(),
            "{kind} at level {level}: row {i}: n_eff {} after a zero-weight row",
            a.n_eff
        );
        // Nor may it leave a NaN behind: a prediction the stream without it
        // makes, the stream with it makes too (hard rule 9).
        let finite = |p: &[f64]| p.iter().map(|v| v.is_finite()).collect::<Vec<_>>();
        assert!(
            !compare || finite(&a.pred) == finite(&b.pred),
            "{kind} at level {level}: row {i}: predictions {:?} in the stream with a zero-weight \
             row, {:?} in the one without it",
            a.pred,
            b.pred
        );
    }
}
