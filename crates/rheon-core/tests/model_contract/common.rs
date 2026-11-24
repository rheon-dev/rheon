/// This contract's own source, the root and every section of it: the roll-call
/// tests read it for the functions they require of each model.
fn contract_source() -> String {
    [
        include_str!("../model_contract.rs"),
        include_str!("common.rs"),
        include_str!("models.rs"),
        include_str!("states.rs"),
        include_str!("more_models.rs"),
        include_str!("refusals.rs"),
        include_str!("bounded.rs"),
        include_str!("predict_is_the_step.rs"),
        include_str!("unusable_values.rs"),
        include_str!("lower_floor.rs"),
        include_str!("generated.rs"),
        include_str!("recovery_since_2_0.rs"),
    ]
    .join("\n")
}

fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

const K: usize = 2;
const HALFLIFE: f64 = 20.0;

/// What a model must report about itself, checked against the config it was
/// built from and against its own behaviour over a fixed stream.
struct Report {
    kind: &'static str,
    n_features: usize,
    n_targets: usize,
    n_outputs: usize,
    /// `n_eff` before the first row, and after 1, 2 and 40 rows.
    n_eff: Vec<f64>,
    /// Prediction-slot count actually returned by `step`.
    pred_len: usize,
    /// The weight reported on the row that carries a long clock gap, and the
    /// weight reported on the row after it.
    before_gap: f64,
    after_gap: f64,
    /// State round-trips through msgpack and continues identically.
    roundtrips: bool,
    /// `clear_lags` on a warm model left every byte of the state where it
    /// was. True for every model that keeps nothing indexed by rows back,
    /// which is the default; a model with a ring says so in `KEEPS_LAGS`.
    lags_are_a_no_op: bool,
}

/// `n_eff_of` reads the model's own `n_eff()` accessor, which is inherent
/// rather than part of the trait, so it has to be handed in. The accessor's
/// value read before a row must equal the `n_eff` that row reports: they are
/// the same number, and reporting two different ones would make `min_weight`
/// mean something different from what a caller inspecting the model sees.
fn probe_with<M: OnlineModel + Clone>(
    m: M,
    targets: usize,
    n_eff_of: Option<&dyn Fn(&M) -> f64>,
) -> Report {
    probe_checked(m, targets, n_eff_of, true, 0.0)
}

/// The spread of the noise [`probe_noisy`] adds to each target: the line's
/// slopes are 1 and -0.5 on features of spread about 0.58, so the noise is
/// about a fifth of the signal's spread, a spread every residual-scale output
/// reads as a number and not as 0.
const NOISE: f64 = 0.2;

/// [`probe_with`] on a target that is the same line plus noise of
/// [`NOISE`] (uniform, from the same generator): every check of the contract
/// -- the weights, the gap, the round trip, `clear_lags` -- on a stream whose
/// residuals are not 0. The noiseless line stays beside it: it is the one
/// whose exact answer the tests of each model hold.
fn probe_noisy<M: OnlineModel + Clone>(
    m: M,
    targets: usize,
    n_eff_of: Option<&dyn Fn(&M) -> f64>,
) -> Report {
    probe_checked(m, targets, n_eff_of, true, NOISE)
}

/// [`probe_with`] for a model run under a window, whose `n_eff` is the
/// weight inside the window and so not rule 8's decayed recursion: the
/// recursion's assertion is left to the caller, which holds the window's
/// own numbers (and `the_n_eff_accessor_is_the_steps_under_a_window` holds
/// the accessor to the step).
fn probe_windowed<M: OnlineModel + Clone>(
    m: M,
    targets: usize,
    n_eff_of: Option<&dyn Fn(&M) -> f64>,
) -> Report {
    probe_checked(m, targets, n_eff_of, false, 0.0)
}

fn probe_checked<M: OnlineModel + Clone>(
    mut m: M,
    targets: usize,
    n_eff_of: Option<&dyn Fn(&M) -> f64>,
    rule_8_recursion: bool,
    noise: f64,
) -> Report {
    let kind = m.state().model.kind();
    let mut n_eff = vec![];
    let mut s = 20260830u64;
    let mut pred_len = 0;

    let row = |m: &mut M, s: &mut u64, d: f64| -> Step {
        let x: Vec<f64> = (0..K).map(|_| lcg(s)).collect();
        let y: Vec<Option<f64>> = (0..targets)
            .map(|j| {
                let e = if noise == 0.0 { 0.0 } else { noise * lcg(s) };
                Some(0.5 * (j as f64 + 1.0) + x[0] - 0.5 * x[1] + e)
            })
            .collect();
        m.step(&x, &y, d, 1.0)
    };

    for i in 0..40 {
        let before = n_eff_of.map(|f| f(&m));
        let step = row(&mut m, &mut s, if i == 0 { 0.0 } else { 1.0 });
        if let Some(before) = before {
            assert!(
                (before - step.n_eff).abs() < 1e-12,
                "row {i}: accessor said {before}, the step reported {}",
                step.n_eff
            );
        }
        if matches!(i, 0..=2) {
            n_eff.push(step.n_eff);
        }
        pred_len = step.pred.len();
    }
    let after_40 = row(&mut m, &mut s, 1.0).n_eff;
    n_eff.push(after_40);

    // A value that is not usable, on the warm model, with the model's own
    // accessor for the `n_eff` the refused row reports (task 183). On
    // copies: `m` goes on below as it was.
    let mut s4 = s;
    let rows: Vec<Row3> = (0..13)
        .map(|_| {
            let x: Vec<f64> = (0..K).map(|_| lcg(&mut s4)).collect();
            let y = (0..targets)
                .map(|j| {
                    let e = if noise == 0.0 {
                        0.0
                    } else {
                        noise * lcg(&mut s4)
                    };
                    Some(0.5 * (j as f64 + 1.0) + x[0] - 0.5 * x[1] + e)
                })
                .collect();
            (x, y, 1.0)
        })
        .collect();
    refuses_unusable_values(&m, &rows[0], &rows[1..], kind, n_eff_of);

    // Serialize here, before the two branches diverge.
    let bytes = rmp_serde::to_vec(&m.state()).unwrap();
    let mut restored = M::restore(&rmp_serde::from_slice(&bytes).unwrap()).unwrap();

    let mut s2 = s;
    let undecayed = row(&mut m, &mut s, 1.0);
    let continued = row(&mut restored, &mut s2, 1.0);
    // Every slot, not only `n_eff`: a state that restores its weights but
    // loses a ring reports the same `n_eff` and a different row (the
    // `ew_cov` lag ring did exactly that, docs/REVIEW-E54-E64.md L1).
    let roundtrips = (continued.n_eff - undecayed.n_eff).abs() < 1e-12
        && continued.pred.len() == undecayed.pred.len()
        && continued
            .pred
            .iter()
            .zip(&undecayed.pred)
            .all(|(a, b)| a == b || (a.is_nan() && b.is_nan()));

    // `n_eff` is read before the row's own decay, so a gap shows up on the row
    // *after* the one that carries it.
    let before_gap = row(&mut restored, &mut s2, 10.0 * HALFLIFE).n_eff;
    let after_gap = row(&mut restored, &mut s2, 1.0).n_eff;

    // Task 47: `clear_lags` drops what is indexed by rows back and *nothing
    // else*. For a model with no such state that is the whole of it, so the
    // bytes must not move; for one with a ring, only the ring may.
    let before_clear = rmp_serde::to_vec(&m.state()).unwrap();
    m.clear_lags();
    let lags_are_a_no_op = rmp_serde::to_vec(&m.state()).unwrap() == before_clear;
    // Rule 8, here rather than only in `check`, so that every model probed
    // is held to it, the models that assert individually and a model added
    // without its own assertions alike (review S6: `pa` reading the weight
    // after its own decay passed once its hand-written probe was gone).
    // `n_eff` is the weight before the row's update and its decay: 0 before
    // the first row, 1 after one row of weight 1, `lam + 1` after two, and
    // a ten-half-life gap decays it by 2^-10 on the row after the gap. A
    // model that does not decay ([`NO_DECAY`]) sums the weights instead.
    if !rule_8_recursion {
        // A window's weight: the caller's assertions hold it.
    } else if NO_DECAY.contains(&kind) {
        assert_eq!(n_eff[..3], [0.0, 1.0, 2.0], "{kind}: the weights summed");
        assert_eq!(after_gap, before_gap + 1.0, "{kind}: no decay over a gap");
    } else {
        assert_eq!(n_eff[0], 0.0, "{kind}: nothing seen before the first row");
        assert_eq!(n_eff[1], 1.0, "{kind}: one row of weight 1");
        let two = 0.5f64.powf(1.0 / HALFLIFE) + 1.0;
        assert!(
            (n_eff[2] - two).abs() < 1e-12,
            "{kind}: n_eff after two rows is {}, not the weight before the row's decay {two}",
            n_eff[2]
        );
        let want = before_gap * 0.5f64.powi(10) + 1.0;
        assert!(
            (after_gap - want).abs() < 1e-9,
            "{kind}: gap decay got {after_gap} want {want}"
        );
    }

    // Asserted here rather than in `check`, so that the models with outputs
    // of their own -- which assert individually instead of calling it -- are
    // held to it too.
    assert_eq!(
        lags_are_a_no_op,
        !KEEPS_LAGS.contains(&kind),
        "{kind}: clear_lags moved state that is not a lag ring (or a model \
         with a ring is missing from KEEPS_LAGS)"
    );

    // A state saved with an *empty* ring must resume like the model it was
    // copied from. Read a ring's depth from a container's capacity instead
    // of from the configuration and it never refills after this copy, which
    // is what `ew_cov`'s lags did (docs/REVIEW-E54-E64.md L1).
    let mut after_clear =
        M::restore(&rmp_serde::from_slice(&rmp_serde::to_vec(&m.state()).unwrap()).unwrap())
            .unwrap();
    let mut s3 = s;
    for i in 0..8 {
        let a = row(&mut m, &mut s, 1.0);
        let b = row(&mut after_clear, &mut s3, 1.0);
        assert!(
            a.pred.len() == b.pred.len()
                && a.pred
                    .iter()
                    .zip(&b.pred)
                    .all(|(x, y)| x == y || (x.is_nan() && y.is_nan())),
            "{kind}: a state saved just after clear_lags diverged by row {i}"
        );
    }

    Report {
        kind,
        n_features: m.n_features(),
        n_targets: m.n_targets(),
        n_outputs: m.n_outputs(),
        n_eff,
        pred_len,
        before_gap,
        after_gap,
        roundtrips,
        lags_are_a_no_op,
    }
}

/// The models that keep state indexed by *rows back*, and so may legally
/// move under `clear_lags`. Everything else must not: `clear_lags` is not a
/// reset, and a model that quietly threw away a mean here would look like a
/// decay bug three chunks later (docs/PLAN.md task 47).
const KEEPS_LAGS: &[&str] = &["rcov", "corrchange", "ew_cov", "marginal", "ar", "spectral"];

/// The model that counts a value that is not usable rather than refusing
/// it: `audit`, whose counts of nulls, NaNs and infinities are what it is for
/// (docs/PLAN.md task 223 (b)). `refuses_unusable_values` passes it by, and
/// its own tests (`audit/tests.rs`) hold what it does with each.
const COUNTS_UNUSABLE: &[&str] = &["audit"];

/// Models that report on some rows and not others by design -- a span-based
/// test writes its statistic where the span closes, and `garch`'s start of 50
/// rows (a cap too small to double it) waits until a buffer's rows tell its
/// variance process from a constant one (PLAN 548), so it reports from some
/// buffer after the first -- so the predict-parity helper cannot ask for 300
/// rows with every slot ready.
const SPARSE_OUTPUT: &[&str] = &["corrchange", "garch"];

/// The models whose `n_eff` is every row's weight summed, with no decay:
/// `audit`, a record of the rows; `bocpd`, whose recursion over run lengths
/// does not decay; `rcov`, a block estimator; `seqtest`, an e-process that
/// would not be one if it forgot; `garch`, whose recursion is per
/// observation and whose estimator weighs every row it learned; `tdigest`,
/// a merging digest that cannot take a row out again; `ddsketch`, whose
/// decayed form is refused (a window is its form that forgets). `probe_with` holds them to the sum where
/// it holds every other model to rule 8's decayed recursion, so a model
/// added here is a choice made in this file, not a default.
const NO_DECAY: &[&str] = &[
    "audit", "bocpd", "ddsketch", "garch", "rcov", "seqtest", "tdigest",
];

/// Slots null by design except on a flag, which the ready count leaves out:
/// `corrchange`'s `since_change` dates a change only where one is flagged
/// (docs/PLAN.md task 114); `garch`'s standardized residual is one only
/// where the row has its residual.
/// `dcc`'s log-density is one only where every series has its residual
/// (slot 1 with two series and no margins).
const ONLY_ON_A_FLAG: &[(&str, usize)] = &[("corrchange", 4), ("garch", 1), ("dcc", 1)];

fn check(r: &Report, kind: &str, targets: usize, combos: usize) {
    assert_eq!(r.kind, kind, "state kind");
    assert!(r.lags_are_a_no_op || KEEPS_LAGS.contains(&kind));
    assert_eq!(r.n_features, K, "{kind}: n_features");
    assert_eq!(r.n_targets, targets, "{kind}: n_targets");
    assert_eq!(
        r.n_outputs,
        targets * combos,
        "{kind}: n_outputs is targets x grid combos"
    );
    assert_eq!(
        r.pred_len, r.n_outputs,
        "{kind}: step must fill exactly n_outputs slots"
    );

    // `n_eff` is the weight *before* the row's update and before its decay, so
    // it starts at zero and lags the row count by one. Every model reports it
    // the same way; that uniformity is what makes `min_weight` portable.
    assert_eq!(r.n_eff[0], 0.0, "{kind}: nothing seen before the first row");
    assert_eq!(r.n_eff[1], 1.0, "{kind}: one row of weight 1");
    let two = 0.5f64.powf(1.0 / HALFLIFE) + 1.0;
    assert!(
        (r.n_eff[2] - two).abs() < 1e-12,
        "{kind}: decayed weight, got {}",
        r.n_eff[2]
    );
    // It saturates at 1/(1 - lam) rather than growing without bound.
    let ceiling = 1.0 / (1.0 - 0.5f64.powf(1.0 / HALFLIFE));
    assert!(
        r.n_eff[3] > 20.0 && r.n_eff[3] < ceiling,
        "{kind}: n_eff after 40 rows is {} (ceiling {ceiling})",
        r.n_eff[3]
    );

    // A ten-half-life gap must decay it by 2^-10, not reset or ignore it.
    let want = r.before_gap * 0.5f64.powi(10) + 1.0;
    assert!(
        (r.after_gap - want).abs() < 1e-9,
        "{kind}: gap decay got {} want {want}",
        r.after_gap
    );

    assert!(r.roundtrips, "{kind}: state did not round-trip");
}
