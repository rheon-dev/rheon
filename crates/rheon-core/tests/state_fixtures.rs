//! Frozen state fixtures: one state of every [`ModelState`] variant, saved
//! mid-stream with what it keeps, embedded as bytes beside the rows that
//! follow it and the numbers they reported (docs/PLAN.md task 198; review
//! round 4, D1 and CF1). Each is held three ways:
//!
//! 1. **it loads**: the bytes decode and the model's `restore` takes them;
//! 2. **it goes on to the bit**: the rows after the save report what they
//!    reported when the fixture was written -- to the bit on the platform
//!    that wrote it, and elsewhere to the golden files' tolerance, since a
//!    decay or a log-likelihood runs through libm, whose last bit differs by
//!    platform (CLAUDE.md, no libm in the state); a NaN is a NaN on both;
//! 3. **it saves its bytes again**: the model restored from them writes the
//!    same bytes.
//!
//! The bank's own fixtures -- a bank file, a `with_windows` state and a
//! `refresh_time` state -- are `rheon-polars`' `tests/state_fixtures.rs`,
//! on the same three checks.
//!
//! **The fixtures are the schema's.** Every schema from `MIN_SCHEMA_VERSION`
//! to `SCHEMA_VERSION` has a set, and `the_fixtures_cover_the_schemas_this_build_loads`
//! refuses a range one does not cover: a layout change cannot land without
//! its fixtures. Before 1.0 a layout change regenerates the set and raises
//! the minimum with the version (hard rule 5's waiver):
//!
//! ```text
//! PRINT_STATE_FIXTURES=1 cargo test -p rheon-core --test state_fixtures
//! ```
//!
//! rewrites every file under `tests/state_fixtures/` from the cases in
//! `tests/cases/mod.rs`, and a run without the variable checks them. No
//! index lists the files: `build.rs` discovers every `<case>.rs` there, and
//! every one under each `v<N>/`, on each build, so two branches that each
//! add a case add two files and edit no shared list. A
//! change that moves a number moves a continuation too: regenerate after
//! confirming the move is intended, as for `tests/golden.rs`.
//!
//! **From 1.0** the minimum stays at 1.0's schema, and a layout change
//! keeps the fixtures of the schema before it (docs/PLAN.md §18 D1; review
//! round 5, D1 and D2). The first 1.x layout change, from schema 44 to 45,
//! goes like this:
//!
//! 1. copy the current set, every `<case>.rs`, to
//!    `tests/state_fixtures/v44/` as it is -- the files are the schema's,
//!    and the directory is the kept set, which `build.rs` finds;
//! 2. raise `SCHEMA_VERSION` to 45 and write the loader the change needs,
//!    in the model's `Deserialize` (`check_schema`'s doc);
//! 3. regenerate the current set, which is then schema 45's.
//!
//! Each kept set is held by `every_previous_fixture_loads_goes_on_and_converts_to_the_current_bytes`:
//! every fixture loads through the loader, goes on to the bit against its
//! own frozen continuation, and -- loaded, then saved -- writes the current
//! schema's fixture bytes of the same case, since the two are the same case
//! over the same rows (the exact-conversion check). A later change repeats
//! the steps with `v45/`: the loaders of 1.x stay.
//!
//! The variant names are the files' tags (the named encoding writes
//! `{"EwRidge": {...}}`), so they are frozen here too: a renamed variant
//! breaks every saved file.

mod cases;

/// The fixtures, discovered: `build.rs` lists every file under
/// `tests/state_fixtures/` and each `v<N>/` beside them, so a case added is
/// a file added and no list is edited by hand.
mod frozen {
    use crate::cases::Fixture;
    include!(concat!(env!("OUT_DIR"), "/state_fixtures_index.rs"));
}

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use cases::{Case, Fixture, Out, agree, all, hex, load, named, saved, writer};
use rheon_core::{MIN_SCHEMA_VERSION, ModelState, SCHEMA_VERSION, State};

/// The schemas before the current one whose fixtures are kept, each under
/// `tests/state_fixtures/v<N>/` and loaded by this build: `previous` lists
/// them, and `previous_sets` is each set as `build.rs` found it in its
/// directory. None before 1.0,
/// where a layout change regenerates the set instead (the module doc has
/// the steps from 1.0). 61, 1.0's, kept since 62 moved a blocked
/// `ewridge`'s layout (docs/PLAN.md task 257): its states load through the
/// field's default, a blocked one's targets current through its held rows.
/// No kept case is blocked, so each converts to the current bytes exactly;
/// a blocked state saved mid-block at 61 goes on from its held rows within
/// the fold's bound of the definition, not to the bits a 62 build reaches
/// from the same rows (`gaps::fold_tests`). 62 kept since 63 gave `huber`
/// and `sgd(loss="huber")` a robust scale (docs/PLAN.md task 271): a 62
/// state has no `huber_scale` and was fitted under the RMS, so it loads as
/// `huber_scale = "rms"` (`HuberScale::of_a_state_before_63`), with no
/// sketch, and goes on to the bit; the `huber` and `sgd_huber` cases are
/// the RMS's, so each converts to the current bytes exactly, and the MAD's
/// form is the new cases `huber_mad` and `sgd_huber_mad`.
///
/// 63 kept since 64 gave `huber` and `sgd(loss = "huber")` the refit of
/// each target's first rows (docs/PLAN.md task 277): a 63 state has no
/// `refit_rows` and no buffers and was fitted without a refit, so it loads
/// with `refit_rows = 0` and none, and goes on to the bit, never refitting
/// rows its stream has long passed; every kept case is the refit's off
/// form, so each converts to the current bytes exactly, and the refit's
/// forms are the new cases `huber_refit` and `sgd_huber_refit`, saved
/// with their buffers half full, and `huber_refit_done` and
/// `sgd_huber_refit_done`, saved after the refit. 64 gave `kalman` its
/// covariance in units of the noise and a diffuse intercept too (task
/// 278): a 63 state has no `noise_units` and loads as the form it was
/// fitted in, `P` absolute and the noise estimate in the gain, and goes on
/// to the bit; the kept `kalman` and `kalman_gap` cases are that form, so
/// each converts to the current bytes exactly, and the new form is the
/// case `kalman_noise_units`, saved with its intercepts still diffuse. 64
/// gave `kalman` and `lasso` `min_weight_holds_learning` too (task 278,
/// D10): a 61-63 state has none and loads as `true`, the gate holding
/// learning back as it was fitted, and goes on as before; the kept cases
/// are that form, so each converts to the current bytes as it did, and the
/// form a spec builds, the gate withholding the output alone, is the cases
/// `kalman_output_only` and `lasso_floor`. 64 gave `ewridge`
/// `min_weight_output_only` too (task 278): a 61-63 state has none and
/// loads as `false`, its residual variance and first solve held back to
/// `min_weight` as they were fitted, and goes on to the bit; every kept
/// `ewridge` case is that form, and re-saved writes the mark `false`, as
/// the current case of the same name does, so each converts to the current
/// bytes; the form a spec with a `min_weight` builds is the case
/// `ewridge_output_only`. 64 gave `ewridge`, `lasso`, `huber` and
/// `quantile` `first_period_by_share` too (task 278, "Default cadence
/// first"): a 61-63 state has none and loads as `false`, its explicit
/// cadence counted from its first solve as it was fitted, and goes on to
/// the bit; every kept case is that form, and re-saved writes the mark
/// `false`, as the current case of the same name does, so each converts to
/// the current bytes; the form a spec with an explicit `solve_every` builds
/// is the cases `ewridge_first_period`, `lasso_first_period` and
/// `huber_first_period`, saved inside the first period, and
/// `ewridge_never`, at `solve_every = inf`, the form with no first period. 64 gave
/// `marginal(bin_stat = "huber")` its own refit of the first rows (task 278): a 63 state has no
/// `refit_rows` in its bins' cfg and no refit record, loads with
/// `refit_rows = 0` and goes on to the bit as the one pass it was fitted
/// in; the kept `marginal_bins` and `marginal_bins_huber` cases are that
/// form, so each converts to the current bytes exactly, and the refit's
/// forms are the new cases `marginal_bins_refit`, saved holding its rows
/// beside given edges, and `marginal_bins_refit_done`, saved after it ran
/// past a learned warm-up.
///
/// The windowed cases convert otherwise since the running products
/// (schema 63 before its release, docs/PLAN.md task 275): a ring saved
/// without them loads with none, so the first read after loading is the
/// saved state's to the bit, and its snapshots go on decaying by the summed
/// clock until they leave. Once a snapshot taken after the load is the
/// boundary, its products decay it and the reads part from the old build's
/// in the last bits (from row 12 of 16 in `ewridge_window`), as a fresh
/// run's do, within the golden tolerance; and the state it saves is not
/// the current fixture's, whose reads took the products from the first
/// row. So a windowed case is held to its first row exactly, the rest to
/// the golden tolerance, and to saving bytes that load back to the same.
///
/// The `kmeans` cases convert otherwise since task 278 (before 63's
/// release; rule 5's second exception, the user's, 2026-10-25): `init =
/// "lloyd"` starts from greedy k-means++ and runs each start to its fixed
/// point, where it took ten Lloyd passes from plain k-means++. A `kmeans`
/// state of schema 61-63 saved before it seeded seeds by the new rule after
/// loading: the same centres and distances to the bit, the cluster labels
/// possibly permuted, held to one consistent relabelling of the frozen
/// labels on every row (`kmeans_warming`, which seeds four rows after the
/// save; its bytes are the current set's). A state saved after seeding
/// continues to the bit (`kmeans`); its centres are in the old rule's
/// order, so its bytes are not the current set's, which the new rule
/// seeded, and it is held to saving bytes that load back to the same.
///
/// The `holt` case converts otherwise since task 278 (D-H, schema 64 before
/// its release): a state of schema 61-63 keeps no line fit, so it loads
/// without one and goes on with the textbook gains alone, the recursion it
/// was fitted under -- to the bit until task 291a moved their rounding,
/// within [`MOVED`]'s bound since. The current set's `holt` was fitted with
/// the line fit, so a kept state's bytes are not its, and it is held to
/// saving bytes that load back to the same, as `kmeans` is.
///
/// The `lasso` cases convert otherwise since task 278 too (rule 5's third
/// exception, the user's, 2026-10-25): the descent's stopping rule reads a
/// sweep's move against `tol` times the target's spread, where it read it
/// against `tol` itself, so where the spread is not 1 a descent stops on
/// another sweep. A `lasso` state of schema 61-63 continues within 1e-9
/// relative, not to the bit, and re-saves the same bytes; [`MOVED`] lists
/// the cases and their bound, and holds an unwindowed one's converted
/// state to the current fixture's field for field, its floats within the
/// bound.
///
/// The `ew_cov` cases continue otherwise since task 294 (rule 5's sixth
/// exception, the user's, 2026-11-01): an accumulator's co-moment matrix is
/// kept as its lower triangle, where a build before updated it whole, each
/// entry `(a·b·d_i)·d_j` and its mirror `(a·b·d_j)·d_i` rounded apart. A
/// state of schema 61-63 loads as its lower triangle, mirrored -- the
/// triangle every build computed alike -- so it re-saves the current
/// fixture's bytes exactly, but an output read from the upper triangle,
/// a covariance or a correlation, reads the mirror where the frozen
/// continuation read the entry rounded on its own; [`CONTINUES_WITHIN`]
/// lists the cases and their bound.
///
/// 64 kept since 65 stopped choosing a residual sketch's anchor from the
/// residuals one pass sees (task 323): while a refit waits, the MAD is
/// read exactly from the `|e|` its buffer keeps beside each row, the
/// sketch taking none; a sketch no refit anchors keeps its grid fixed at
/// one. A kept state's sketches are anchored at their first value, or await
/// it, and its refit buffers hold no `|e|`, so its model's restore marks
/// each anchored sketch to keep the old rule, reads its scale from the
/// sketch it feeds until its refit, and it goes on to the bit. Converted,
/// every anchored sketch carries the mark and every open buffer says it
/// holds no `|e|`, which a state this build makes does not, and the rows
/// before the save were scaled by the old rule, so such a case -- one
/// whose kept state holds an anchored sketch, or whose current state holds
/// an `|e|` or deviations for a refit -- is held to its marks and to
/// saving bytes that load back to the same ([`kept_scale_rule`]), not to
/// the current fixture's bytes.
///
/// 65 kept since 66 made two changes (tasks 327 and 328). First, `hmm` and
/// `ew_class` under `covariance_type = "diag"` keep their variances alone
/// (task 327): a 65 state keeps each state's or class's whole co-moments,
/// and each window snapshot's, of which the shape read the diagonal alone,
/// so it loads as their diagonals, goes on to the bit, and saves the
/// current fixture's bytes of the same case -- the windowed
/// `ew_class_window` too, its snapshots carrying their running products
/// since 63 (the windowed rule below is for the sets before them). The
/// cases `hmm_diag` and `ew_class_diag` were written into `v65/` by the
/// schema-65 build, so the conversion of each diagonal form is held,
/// windowed and not. Second, `corrchange(kind = "window")` keeps the count
/// of rows its ring has taken, `ring_rows`, which numbers the rows its
/// merged sums are cut by (task 328). It is written only where it is not 0,
/// so a monitor or sequential state's bytes are as they were; a window's
/// state of 65 has none and is numbered from its ring's oldest row. The
/// case `corrchange_window` was written into `v65/` by the schema-65
/// build, so that conversion is held: [`RENUMBERED`] says how.
///
/// 66 kept since 67 made two changes (task 340), of which one touches a
/// model's state: `marginal`'s mark `before_65` -- carried by a histogram
/// still to be built from a state of schema 64 or earlier, which the freeze
/// then builds by that build's rule -- is cleared at the freeze, where it
/// acts last, so from there the state saves the current form; a build
/// before wrote it on every later save. A state of 65 or 66 saved past its
/// freeze carries the mark still and loads without it, going on to the
/// bit: the case `marginal_bins_refit_first_value_frozen`, saved past its
/// freeze and before its refit, was written into `v66/` by the schema-66
/// build, marked, and converts to the current fixture's bytes exactly;
/// `marginal_bins_refit_first_value`, saved inside its warm-up, loads
/// marked as before, and its kept file is the current one. The other
/// change, the notices a bank's stream keeps, is the bank harness's. Every
/// other kept case converts to the current bytes exactly.
///
/// 67 kept since 68 let `kalman` hold its covariance as the square root's
/// factor (docs/PLAN.md task 281 (3)), written with the field
/// `covariance_form` only where it is not the covariance form: a 67 state
/// has none and loads as the covariance form it is, so every kept case
/// converts to the current bytes exactly. The new form is the case
/// `kalman_square_root`, which has no kept file.
///
/// 68 kept since 69 made a wide `full` `hmm` state keep the Cholesky
/// factor of its matrix as the state (task 479; rule 5's twentieth
/// exception, the user's, 2026-11-20): at 13 features or more, once seeded,
/// a state is its weight, means and `M = σ·L Lᵀ`, written in
/// `factor_states` only where it is not empty, and a learned row moves the
/// factor by rotations. A 68 state of that width holds co-moments; it loads
/// as their factor and goes on in the factor form. The cases `hmm_full`
/// (13 features) and `hmm_full_narrow` (12, which keeps the co-moments)
/// were written into `v68/` by the schema-68 build, so the conversion and
/// the width it starts at are held. Since 70 both go on under the
/// stabilized prior ([`UNDER_THE_PRIOR`]), where 69's build held the wide
/// one within rounding of its frozen continuation.
///
/// 69 kept since 70 put each `full` and `diag` `hmm` state of a model that
/// learns under a stabilized prior (rule 5's twenty-second exception, the
/// user's, 2026-11-20): the model keeps `ŝ`, `V` and, in the factor form,
/// `f`, in `stabilized`, written only where the model has a prior, and
/// writes `factor_states` always, empty or not, since it is no longer the
/// last field. A state of 69 or earlier holds no prior and loads with one
/// seeded from its moments ([`UNDER_THE_PRIOR`]); the `tied` case `hmm`
/// takes none, and converts to the current bytes exactly, as every other
/// kept case does. The case `hmm_full_warming`, saved inside its warm-up
/// with a `prior_rows` of its own, has no kept file.
///
/// And since review C1 (rule 5's twenty-third exception, the user's,
/// 2026-11-20), with no layout change and no schema bump, `sgd` and `pa`
/// under `standardize` read a row in the stream's own unit of weight. Every
/// kept `sgd_*` and `pa*` case is weighted, so the current set's were
/// regenerated and a kept state of those cases of 68 or earlier continues
/// by the new read, within [`READ_ANEW`]'s bounds of its frozen
/// continuation, not to the bit; it saves bytes that load back to the same.
/// The kept `v69/` set of those cases was written by the schema-69 build
/// with C1's read (no release carried 69 before C1), so it converts to the
/// current bytes exactly and continues to the bit, as every other kept case
/// does.
///
/// 71 kept since 72 (PLAN 542's envelope, then PLAN 551 before any release
/// carried 72): `deco`'s level is the ratio of the means, in `ratio`, where
/// `rho_bar` kept the EW mean of `u`. A `deco` state of 71 or earlier loads
/// with its level restarted (rule 5's twenty-sixth exception,
/// [`RESTARTED_LEVEL`]): every kept `deco` file is held to that, not to the
/// current bytes. The ratio's forms are the cases `deco`, `deco_linear` and
/// `deco_warming`.
///
/// The kept schemas, each a `v<N>/` directory found beside the current set.
fn previous() -> Vec<u32> {
    frozen::PREVIOUS_SETS.iter().map(|(s, _)| *s).collect()
}

/// Each kept schema's set, as `build.rs` found it on disk.
fn previous_sets() -> Vec<(u32, &'static [&'static Fixture])> {
    frozen::PREVIOUS_SETS.to_vec()
}

/// The cases whose numbers a later fix moved without touching the layout,
/// each with how far its continuation may part from a previous set's
/// frozen one. A previous fixture of such a case was fitted under the old
/// arithmetic, so neither its continuation nor its converted state is the
/// current build's to the bit: it is held to loading, to saving bytes that
/// load back to the same, and to its continuation within the bound, by
/// [`agree`]'s relative measure. The current set is regenerated as for any
/// change that moves a number, and holds them to the bit.
///
/// Its converted state is held to the current fixture's field for field:
/// the same keys, lengths and every number but a float the same, each
/// float within the bound ([`same_but_floats`]), so the loader's work is
/// still checked whole -- but under a window, whose converted ring differs
/// by design (the doc above `previous`).
///
/// - `lasso`, `lasso_window`: the descent's stopping rule compares a
///   sweep's move with `tol` times the target's spread, not with `tol`
///   itself (task 278, D3), so a descent stops on another sweep where the
///   spread is not 1. On these rows that moved the predictions by at most
///   5.8e-11 (`lasso`) and 1.7e-11 (`lasso_window`) of `1 + |pred|`,
///   measured 2026-10-25 on both kept sets; the bound is 17 times the
///   larger.
/// - `holt`: the recursion divides into each denominator once and keeps
///   the reciprocal to multiply by (task 291a, the user's, 2026-11-01), so
///   the textbook gains a state of schema 61-63 goes on with round
///   otherwise in their last bits. On these rows that moved the
///   predictions by at most 7.5e-17 of `1 + |pred|` (2.3e-16 relative),
///   measured 2026-11-01 on all three kept sets; the bound is 133 times
///   that.
/// - `bocpd`, `bocpd_warming`: two features or more under the diagonal
///   covariance take the features' `ln ψₙ` and `ln(1 + z²/νₙ)` as the logs of
///   products of blocks of 8, and `ln(νₙπ var)` as `ln ψₙ + ln(π(κₙ+1)/κₙ)`
///   (rule 5's sixteenth exception, the user's, 2026-11-17), so a log density
///   parts from the old arithmetic's in its last bits and a kept state's
///   log joint, which the rows before its save made, is the old build's.
///   On these rows that moved 9 and 10 of 112 numbers by at most 1.4e-15 of
///   `1 + |value|`, measured 2026-11-17 as the set before the change
///   against the regenerated one (the kept sets' continuations are the
///   former's); the bound is 70 times that.
/// - `corrchange_window`: the permutation critical value is drawn by Lemire's
///   multiply-shift reduction in place of a remainder (rule 5's seventeenth
///   exception, the user's, 2026-11-17), so a state's saved `perm_crit` and
///   every critical value the continuation redraws part from the old
///   draws', by a few percent in general and as the rows are few here by
///   more. On these rows `perm_crit` moved by 0.115 and the redrawn `crit`
///   by 0.140 of `1 + |crit|` (schemas 65 and 66, over the rows after the save,
///   measured 2026-11-18 as each kept set against the regenerated one);
///   no `flag` flipped. The bound, 0.25, is under the 0.5 a flipped flag
///   would be, so a flip fails the test. A schema-65 state is also
///   renumbered ([`RENUMBERED`]).
/// - `micro`: the rows' mean weight is taken at the old mean's and the
///   row's shares, where the Welford step cancelled (rule 5's thirtieth
///   exception, the user's, 2026-11-23: "Exception 30, full fix"), so a
///   kept state's `w_mean` holds the old step's last bits. On these rows it
///   moved by one ulp (2.2e-16 relative), measured 2026-11-23 on every kept
///   set; the bound is 45 times that.
const MOVED: &[(&str, f64)] = &[
    ("lasso", 1e-9),
    ("lasso_window", 1e-9),
    ("holt", 1e-14),
    ("bocpd", 1e-13),
    ("bocpd_warming", 1e-13),
    ("corrchange_window", 0.25),
    ("micro", 1e-14),
];

/// The [`MOVED`] cases whose bound covers only some numbers, each with the
/// output slots and the state keys it covers; every other number of the
/// case is held as an unmoved case's is (review S5: `corrchange_window`'s
/// 0.25, justified for the permutation critical value alone, covered every
/// float of the converted state and every slot of the continuation, so a
/// loader that mis-scaled the weight passed).
///
/// - `corrchange_window`: `crit`, slot 1 of `stat`, `crit`, `flag`,
///   `since_flag`, `since_change`, and the state's `perm_crit`, both the
///   draws'; the statistic, the flag, the counts, the weight and the rest
///   of the state continue as an unmoved case's do.
const MOVED_ONLY: &[(&str, std::ops::Range<usize>, &[&str])] =
    &[("corrchange_window", 1..2, &["perm_crit"])];

/// A [`MOVED`] case's bound at the state's field `path` (as
/// [`same_but_floats`] names it): the whole bound, or under
/// [`MOVED_ONLY`] the bound on its keys and 0 elsewhere.
fn moved_bound_at(name: &str, bound: f64, path: &str) -> f64 {
    match MOVED_ONLY.iter().find(|(n, _, _)| *n == name) {
        None => bound,
        Some((_, _, keys)) => {
            let last = path.rsplit('.').next().unwrap_or(path).trim_matches('"');
            if keys.contains(&last) { bound } else { 0.0 }
        }
    }
}

/// A [`MOVED`] case's bound at output slot `n`: the whole bound, or under
/// [`MOVED_ONLY`] the bound on its slots and `base` elsewhere.
fn moved_slot_bound(name: &str, bound: f64, base: f64, n: usize) -> f64 {
    match MOVED_ONLY.iter().find(|(c, _, _)| *c == name) {
        Some((_, slots, _)) if !slots.contains(&n) => base,
        _ => base.max(bound),
    }
}

/// The cases whose state converts exactly but whose continuation a later
/// change moved, each with how far it may part from a previous set's
/// frozen one by [`agree`]'s measure: unlike a [`MOVED`] case, its
/// converted state is still held to the current fixture's bytes.
///
/// - `ew_cov`, `ew_cov_window`: the co-moment matrix is kept as its lower
///   triangle (task 294), so a covariance or correlation of pair `(i, j)`,
///   `i < j`, reads entry `(j, i)`, where the frozen continuation read
///   entry `(i, j)`, rounded apart from it. On these rows that moved the
///   outputs by at most 1.9e-16 (`ew_cov`) and 7.1e-16 (`ew_cov_window`)
///   of `1 + |out|`, from the first row after the load, measured
///   2026-11-01 on all three kept sets; the bound is the user's, 1e-15.
///   It holds the `corr` slots alone, numbers 6 to 8 of both cases (after
///   three means and three variances; `cases::all`), and no other: the
///   means, variances, Mahalanobis distances, its quantiles, the lagged
///   correlations and the component read the lower triangle only, and
///   continue to the bit (task 316, review 11's S2-1).
/// - `ew_cov_accumulator`: the bare accumulator's slots are its two means
///   and then its 2 × 2 co-moments row by row (`cases::Accumulator`), so
///   number 3 is `cov(0, 1)`, an upper entry, read through the mirror
///   from `(1, 0)` since task 294 and read directly before it. On these
///   rows the two entries are the same double, so it continues to the bit;
///   that is the data's doing, not the arithmetic's, so it is held to the
///   same 1e-15 as the model's `corr` (task 316, review 11's S2-3).
const CONTINUES_WITHIN: &[(&str, f64, std::ops::Range<usize>)] = &[
    ("ew_cov", 1e-15, 6..9),
    ("ew_cov_window", 1e-15, 6..9),
    ("ew_cov_accumulator", 1e-15, 3..4),
];

/// The `corrchange(kind = "window")` cases of schema 65, each with how far
/// its statistic may part from the frozen one by [`agree`]'s measure, and
/// the slots that statistic fills. A state of 65 keeps no `ring_rows`
/// (task 328), so it loads numbered from its ring's oldest row: converted,
/// it is the current fixture's field for field but that count, which is the
/// ring's length where the current fixture's counts every row the ring has
/// taken ([`renumbered_rule`]). The statistic is merged from blocks cut by
/// that numbering, where the schema-65 build computed it in two passes over
/// the ring, so it continues within the merged sums' rounding of the frozen
/// one; every other slot -- `crit`, `flag`, `since_flag`, `since_change`
/// and the weight -- continues to the bit.
///
/// - `corrchange_window`: two windows of 16 rows over two columns, four
///   blocks a window, the ring saved past its first row (32 rows held, 36
///   taken). On these rows the statistic moved by at most 4.1e-16 of
///   `1 + |stat|` (a few units in its last place), measured 2026-11-05 on
///   macOS arm64; the bound is 24 times that.
const RENUMBERED: &[(&str, f64, std::ops::Range<usize>)] = &[("corrchange_window", 1e-14, 0..1)];

/// `garch`'s states of schema 71 and before, whose forecast was one
/// `horizon` (task 559): each loads as the one horizon it named, and goes
/// on reporting its sum beside it, so a target's slots are four where they
/// were three (the variance, the standardized residual, the forecast, and
/// now the sum). The sums, which the old build did not report, are dropped
/// from the comparison ([`narrowed`]); every other number goes on to the
/// bit, the forecast's arithmetic being the old build's. This is such a
/// row in schema 71's layout from this build's, the sums dropped: each
/// target's four slots to the three of before.
fn narrowed(got: &Out) -> Out {
    let mut out = got.clone();
    out.pred = got
        .pred
        .chunks(4)
        .flat_map(|c| c[..3].iter().copied())
        .collect();
    out
}

/// The `hmm` cases a kept set holds under `covariance_type` `"full"` and
/// `"diag"` with `learn` on, which go on under the stabilized prior (schema
/// 70; rule 5's twenty-second exception, the user's, 2026-11-20). A state
/// of 69 or earlier keeps no prior: it loads with one seeded from its
/// states' moments -- `ŝ` their variances pooled by weight, `V` their total
/// weight -- and a factor form (`hmm_full`, 13 features, which a set before
/// 69 keeps in co-moments) taken up again under it, at its target. So
/// neither its continuation nor its converted state is the old build's, nor
/// this build's from the same rows, whose prior acted from the seeding:
/// each is held to loading, to saving bytes that load back the same and
/// hold the prior ([`under_the_prior_rule`]), to the frozen continuation's
/// `weight_sum` to the bit, and to a number wherever the frozen
/// continuation has one. The `tied` case `hmm` takes no prior, and goes on
/// to the bit.
const UNDER_THE_PRIOR: &[&str] = &["hmm_diag", "hmm_full", "hmm_full_narrow"];

/// The `deco` cases a kept set holds, whose level restarts on load (PLAN
/// 551; rule 5's twenty-sixth exception, the user's "Switch, restarting the
/// level", 2026-11-22). A state of 71 or earlier keeps the EW mean of `u`
/// (`rho_bar`), from which the ratio's sums cannot be had: it loads with its
/// standardiser as saved and its level's sums empty ([`restarted_level_rule`]),
/// so its continuation is neither the old build's nor this build's from the
/// same rows. It is held to saving bytes that load back the same; to the
/// frozen continuation's `weight_sum` and `u` (slot 0), which the
/// standardiser alone gives, to the bit; and to `rho` and `loglik` null on
/// the rows before the level relearns, a number on the rows after, the
/// relearning reached within the continuation. `deco.rs`'s
/// `a_state_from_before_the_ratio_restarts_its_level` holds the restarted
/// level to a model emptied by hand, to the bit.
const RESTARTED_LEVEL: &[&str] = &["deco"];

/// The `huber` and `quantile` cases of schema 71 or earlier (rule 5's
/// twenty-ninth exception, the user's "Exception 29: start fresh",
/// 2026-11-23; review 2026-11-23, REG-18): under `standardize` the state
/// keeps each target's features' moments over every row, which a state of
/// 71 or earlier holds none of. It loads with them empty and starts them at
/// its first row after the load, its scales the band's until they give one,
/// so its continuation is neither the old build's nor this build's from the
/// same rows. It is held to [`started_fresh_rule`] -- saving bytes that load
/// back the same, its moments empty and in the current layout -- and to
/// its frozen continuation's `weight_sum` to the bit, each prediction a
/// number where the frozen one is, and within [`STARTED_FRESH_BOUND`] of it
/// relatively; a case without `standardize` keeps none and goes on to the
/// bit.
fn started_fresh(name: &str, schema: u32) -> bool {
    schema < 72 && (name.starts_with("huber") || name.starts_with("quantile"))
}

/// What a fresh start of the scales moves a kept `huber` or `quantile`
/// state's predictions by, relative to the larger of each and 1, over the
/// 16 rows after its load ([`started_fresh`]): the largest measured over
/// `v61/` to `v71/` is 1.56e-5 (`huber`, its fourth row; 3,540 of 4,554
/// predictions moved, every null kept). The scales move the ridge's
/// weight on each slope, `1e-6` of its standardized square, and the
/// columns kept, which no kept case drops.
const STARTED_FRESH_BOUND: f64 = 2e-5;

/// The weighted `sgd` and `pa` cases under `standardize` (rule 5's
/// twenty-third exception, the user's, 2026-11-20; review C1), each with
/// how far its continuation may part from a previous set's frozen one by
/// [`agree`]'s measure. The scaler reads a row in the stream's own unit of
/// weight, one row's `ŵ = Σw²/Σw` of the warm-up's rows, while it warms up
/// and the smaller of that and the row's own weight once the fit is held,
/// where it read every row at 1, and keeps `ŵ·n/(n + 1)` and then `ŵ` of
/// history into a row; at unit weights both are 1 and nothing moves. These
/// cases' rows are weighted, so the current set's states, written by this
/// build, are not what a kept state converts to (no layout changed: it
/// loads as it was), and a kept state continues by the new read from the
/// old read's moments and fit. It is held to saving bytes that load back to
/// the same, and its continuation within the bound. Measured 2026-11-20 on
/// macOS arm64 over every kept set (v61 to v68), the largest move of any
/// number after the save, in the comment; the bounds are about three times
/// those. A held fit's prediction reads no moment, and the first number
/// that moved is on row 3 after the save (row 4 for `sgd_resting`); a
/// warming case's prediction reads the moments, and moves from row 0.
/// `sgd_unstable` (a constant rate of 2, past the bound its gate holds)
/// reports every number after the save to the bit, measured, so its bound
/// is 0; its state moved, so it is listed.
const READ_ANEW: &[(&str, f64)] = &[
    ("sgd", 0.0003),                   // 1.1e-4
    ("sgd_warming", 0.1),              // 3.1e-2
    ("sgd_huber", 0.0025),             // 8.2e-4
    ("sgd_huber_mad", 0.005),          // 1.5e-3
    ("sgd_huber_mad_anchored", 0.005), // 1.9e-3
    ("sgd_huber_refit", 0.0006),       // 2.1e-4
    ("sgd_huber_refit_done", 0.0025),  // 8.5e-4
    ("sgd_spread", 0.0025),            // 8.1e-4
    ("sgd_unstable", 0.0),             // 0
    ("sgd_resting", 0.005),            // 1.6e-3
    ("sgd_spread_adagrad", 0.0035),    // 1.2e-3
    ("sgd_eps", 0.0012),               // 4.1e-4
    ("sgd_logistic", 0.0003),          // 1.1e-4
    ("pa", 0.002),                     // 6.8e-4
    ("pa_warming", 0.25),              // 8.6e-2
];

/// The `audit` cases, whose kept states counted the values of the rows of
/// weight 0 before the save (the user, 2026-11-21: an audit stops counting
/// such rows, hard rule 9). No layout changed, and what a kept state counted
/// cannot be told apart from the rest, so it loads as it was: it continues
/// to the bit (an audit's only output is `n_eff`, which never took such a
/// row's weight), saves bytes that load back to the same, and differs from
/// the current fixture's state in its columns' and pairs' counts and, since
/// PLAN 539 (a zero-weight row counts in no clock), its clock's, which
/// counted those rows' steps ([`counted_zero_weight_rule`]). The sets of
/// schema 70 and earlier; schema 71's was written by the build that counts
/// none, and converts to the current bytes exactly.
const COUNTED_ZERO_WEIGHT: &[&str] = &["audit", "audit_exact", "audit_scaled"];

/// The `kalman` case whose kept states were saved after rows that observed
/// no target (rule 5's thirty-fifth exception, the user's, 2026-11-24: "No
/// concern for continuing differently here"; review 2026-11-23, KAL-6).
/// Under `standardize` the moments' floor reads the moments' own gap, the
/// clock since the last row of weight above 0, which a state keeps as
/// `moments_gap` from schema 72; a state of 71 or earlier holds none, and
/// reads the clock since a target was observed, which is not the moments'
/// gap after such rows (here 2, where this build keeps 0). The rows are not
/// in the state, so no loader can know it. Such a state loads, continues to
/// the bit of its frozen continuation, and saves bytes that load back the
/// same; it is not held to the current fixture's bytes. The sets `v64/` to
/// `v71/`.
const MOMENTS_GAP: &[&str] = &["kalman_noise_units"];

/// The `corrchange` cases whose statistics were replaced with no layout
/// changed (rule 5's twenty-seventh exception, the user's, 2026-11-22:
/// "Switch to the new statistics"; PLAN 553): `monitor` is the linear
/// CUSUM against the grid-corrected critical value, `sequential` the
/// Fisher-`z` detector. A kept state holds no number the old statistics
/// computed -- the span's or the history's rows, the history's correlation
/// and long-run deviation, the monitored rows' sums -- so it loads as it was
/// and goes on under the new statistics, and its frozen continuation, the
/// old statistics', is not what it is held to. What it keeps of the old
/// ones is where they flagged. What is held:
///
/// - `corrchange_monitor`: no span of these rows flagged under either
///   statistic before the save, so the kept state converts to the current
///   fixture's bytes exactly, and goes on to the current fixture's frozen
///   continuation to the bit: every output as this build computes it from
///   the same rows.
/// - `corrchange_sequential`: the old detector last flagged 8 learned rows
///   before the save and the new one 19, so the kept state's cycle began
///   at the old flag, and it is not the current fixture's. It is held to
///   saving bytes that load back to the same, and to its frozen
///   continuation's `weight_sum` to the bit; its reports are the new
///   detector's over the cycle the old one began, which no build reaches
///   from these rows, so no output of them is held to a number.
const NEW_STATISTICS: &[&str] = &["corrchange_monitor", "corrchange_sequential"];

/// The windowed `ew_class` cases of schema 71 or earlier (rule 5's
/// thirty-second exception, the user's, 2026-11-24: "No concern for
/// continuing differently here"; review 2026-11-23, COV-3): a window's
/// ridge is the prior's weight over the window's weight, `s·W / W_R`, where
/// it was the whole history's share `s`. No layout changed, so a kept state
/// converts to the current fixture's bytes exactly; its frozen
/// continuation, read with the old ridge, is not what it is held to. It
/// goes on as the current fixture does, every output to the bit.
const WINDOW_RIDGE: &[&str] = &["ew_class_window"];

/// The windowed `marginal` lag cases of schema 71 or earlier (review
/// 2026-11-23, MARG-1, the user's decision of 2026-11-24, "(a), exact"):
/// the lag moments keep, since, the sides now over the rows a lagged one
/// was matched with and their weights, which a kept state holds none of.
/// It loads with them started at 0 (`MarginalLags::start_matching`), and
/// its window reads them once its boundary is a snapshot taken after the
/// load, and the old way until then: so it goes on to its frozen
/// continuation to the bit, and converts to the current fixture's state
/// but for those sides ([`without_matched`]), which the current state
/// holds from its first row.
const MATCHED_FROM_LOAD: &[&str] = &["marginal_window_lags"];

/// `v` with the sides over the matched rows taken out of every map
/// ([`MATCHED_FROM_LOAD`]).
fn without_matched(v: &rmpv::Value) -> rmpv::Value {
    match v {
        rmpv::Value::Map(m) => rmpv::Value::Map(
            m.iter()
                .filter(|(k, _)| k.as_str() != Some("matched"))
                .map(|(k, v)| (k.clone(), without_matched(v)))
                .collect(),
        ),
        rmpv::Value::Array(a) => rmpv::Value::Array(a.iter().map(without_matched).collect()),
        v => v.clone(),
    }
}

/// The `garch` cases of schema 71 saved within a start that had cleared its
/// buffer: the start's rule of that schema tried at one, two and four times
/// `start_rows` and then let every row go and buffered afresh, which no
/// count of tries ends now (Fable's W13 review, A2, for `garch`: a try that
/// does not start keeps its rows). The rows let go are not in the state,
/// so no build reaches the current fixture's state from it. It loads as it
/// was, its buffer a first one (its next try at `start_rows`, at once where
/// it holds them), and goes on under the current rule: held to saving bytes
/// that load back to the same, to its frozen continuation's `weight_sum` to
/// the bit, and to a null wherever that continuation has one, since the
/// start is still waiting. No release wrote such a state: `garch` is new
/// since 2.0.0, whose schema was 67. Proposed as rule 5's twenty-eighth
/// exception (`docs/records/STATE-CONTINUATION.md`), for the user's word.
const KEPT_START: &[&str] = &[
    "garch_egarch",
    "garch_figarch",
    "garch_skewt",
    "garch_started",
    "garch_targeted",
];

/// How far a continuation may part from the one frozen, on a platform other
/// than the writer's: the golden files' tolerance (`tests/golden.rs`).
const TOL: f64 = 1e-12;

const REGENERATE: &str = "PRINT_STATE_FIXTURES=1 cargo test -p rheon-core --test state_fixtures";

fn regenerating() -> bool {
    std::env::var("PRINT_STATE_FIXTURES").is_ok_and(|v| v == "1")
}

fn case_of(f: &Fixture) -> Case {
    all()
        .into_iter()
        .find(|c| c.name == f.name)
        .unwrap_or_else(|| {
            panic!(
                "{}: no case of that name; regenerate with `{REGENERATE}`",
                f.name
            )
        })
}

fn same_out(what: &str, got: &Out, want: &Out, tol: f64) {
    same_out_by(what, got, want, |_| tol);
}

/// [`same_out`], each number `i` (the predictions', then `n_eff`, then the
/// extras') within its own `tol(i)`.
fn same_out_by(what: &str, got: &Out, want: &Out, tol: impl Fn(usize) -> f64) {
    let pairs = got
        .pred
        .iter()
        .zip(&want.pred)
        .chain([(&got.n_eff, &want.n_eff)])
        .chain(got.extra.iter().zip(&want.extra));
    assert_eq!(
        (got.pred.len(), got.extra.len()),
        (want.pred.len(), want.extra.len()),
        "{what}: the row reports another number of slots"
    );
    for (i, (g, w)) in pairs.enumerate() {
        assert!(
            agree(*g, *w, tol(i)),
            "{what}, number {i}: got {g:e} ({:#018x}), frozen {w:e} ({:#018x})",
            g.to_bits(),
            w.to_bits()
        );
    }
}

#[test]
fn every_fixture_loads() {
    if regenerating() {
        return;
    }
    for f in frozen::ALL {
        let case = case_of(f);
        let s: State = rmp_serde::from_slice(&f.bytes())
            .unwrap_or_else(|e| panic!("{}: the bytes do not decode: {e}", f.name));
        assert_eq!(s.schema_version, f.schema, "{}", f.name);
        let variant = format!("{:?}", s.model);
        assert!(
            variant.starts_with(&format!("{}(", f.variant)),
            "{}: a state of {}",
            f.name,
            variant.split('(').next().unwrap()
        );
        if let Err(e) = (case.restore)(&s) {
            panic!("{}: the state does not load: {e}", f.name);
        }
    }
}

#[test]
fn every_fixture_goes_on_to_the_bit() {
    if regenerating() {
        return;
    }
    for f in frozen::ALL {
        let case = case_of(f);
        let mut m = load(&case, &f.bytes()).unwrap_or_else(|e| panic!("{}: {e}", f.name));
        let tol = if f.writer == writer() { 0.0 } else { TOL };
        for (i, (row, want)) in f.rows().iter().zip(f.outs()).enumerate() {
            let got = m.step(row);
            same_out(
                &format!("{}, row {i} after the save", f.name),
                &got,
                &want,
                tol,
            );
        }
    }
}

#[test]
fn every_fixture_saves_its_bytes_again() {
    if regenerating() {
        return;
    }
    for f in frozen::ALL {
        let case = case_of(f);
        let bytes = f.bytes();
        let m = load(&case, &bytes).unwrap_or_else(|e| panic!("{}: {e}", f.name));
        assert!(
            named(&m.state()) == bytes,
            "{}: the state restored from the fixture saves other bytes",
            f.name
        );
    }
}

/// The `.rs` files directly in `dir`, counted: what `build.rs` discovers
/// there, read again at run time, so a test binary built before a fixture
/// was added or removed is caught rather than trusted.
fn fixtures_on_disk(dir: &Path) -> Result<usize, String> {
    Ok(std::fs::read_dir(dir)
        .map_err(|e| format!("{}: {e}", dir.display()))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_file() && p.extension().is_some_and(|x| x == "rs"))
        .count())
}

/// The set a kept schema has on disk: the fixtures under `dir/v<schema>/`,
/// counted; or why there is none.
fn set_on_disk(dir: &Path, schema: u32) -> Result<usize, String> {
    let set = dir.join(format!("v{schema}"));
    if !set.is_dir() {
        return Err(format!(
            "schema {schema} has no set under {}: a kept schema keeps its fixtures there",
            set.display()
        ));
    }
    match fixtures_on_disk(&set)? {
        0 => Err(format!(
            "schema {schema}: {} holds no fixture",
            set.display()
        )),
        n => Ok(n),
    }
}

/// The schema of the current set: the one every fixture in it was written
/// at.
fn current_schema() -> u32 {
    let schemas: Vec<u32> = frozen::ALL.iter().map(|f| f.schema).collect();
    assert!(
        !schemas.is_empty(),
        "no current fixture was discovered under {}: regenerate with `{REGENERATE}`",
        dir().display()
    );
    assert!(
        schemas.iter().all(|s| *s == schemas[0]),
        "one schema a set: {schemas:?}"
    );
    schemas[0]
}

/// Every schema this build claims to load has its fixtures, and every
/// fixture is of a schema it loads: so `SCHEMA_VERSION` cannot move without
/// a set of its own, and `MIN_SCHEMA_VERSION` cannot sit below the oldest
/// set kept. A kept schema's set is on disk, under `v<N>/`, every fixture
/// in it of schema `N`, and each set compiled in holds every file its
/// directory does (review round 5, D2; discovery, task 281 (4)).
#[test]
fn the_fixtures_cover_the_schemas_this_build_loads() {
    if regenerating() {
        return;
    }
    let schema = current_schema();
    assert_eq!(
        schema, SCHEMA_VERSION,
        "SCHEMA_VERSION is {SCHEMA_VERSION} and the frozen state fixtures are of {schema}: a \
         layout change ships its fixtures. Before 1.0, regenerate them with `{REGENERATE}` and \
         raise MIN_SCHEMA_VERSION (and the bank's minimum) to the new schema; from 1.0, keep \
         this set as the previous schema's (the module doc), write its loader, then regenerate"
    );
    assert_eq!(
        fixtures_on_disk(&dir()),
        Ok(frozen::ALL.len()),
        "the current set compiled in is not the files on disk: rebuild the test"
    );
    let previous = previous();
    let covered: Vec<u32> = previous.iter().copied().chain([schema]).collect();
    let claimed: Vec<u32> = (MIN_SCHEMA_VERSION..=SCHEMA_VERSION).collect();
    assert_eq!(
        claimed, covered,
        "this build loads schemas {MIN_SCHEMA_VERSION}..={SCHEMA_VERSION} and fixtures cover \
         {covered:?}: each schema it loads needs a frozen set, held to its loader"
    );
    for (schema, set) in &previous_sets() {
        let on_disk = set_on_disk(&dir(), *schema).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(
            on_disk,
            set.len(),
            "schema {schema}: v{schema}/ holds {on_disk} fixtures and {} were compiled in",
            set.len()
        );
        for f in *set {
            assert_eq!(
                f.schema, *schema,
                "v{schema}/{}.rs is another schema's",
                f.name
            );
        }
    }
}

/// How many fixtures each kept schema's set holds, frozen when the set was
/// copied (review S4). The kept sets are the only test of an old schema's
/// loader, and the checks above hold only that a set is not empty and
/// that what was compiled in is what is on disk, so a deleted file or a
/// partial copy passed. A new kept set adds its line here, with the count
/// of the current set it was copied from. `v71/` was copied with 104 and
/// holds 102: rule 5's thirty-sixth exception (the user's, 2026-11-24)
/// removed `glm_cap` and `glm_refit_done` from it, states no release wrote
/// whose generating streams the log link's target guard now fires on.
const KEPT_COUNTS: &[(u32, usize)] = &[
    (61, 41),
    (62, 42),
    (63, 47),
    (64, 75),
    (65, 81),
    (66, 82),
    (67, 82),
    (68, 85),
    (69, 85),
    (70, 86),
    (71, 102),
];

/// The names one set holds that the next does not, each set given as a
/// label and its fixtures' names, oldest first: a later set is copied from
/// the current one, which only gains cases, so each set holds every name
/// of the one before it.
fn names_dropped(sets: &[(String, Vec<&str>)]) -> Vec<String> {
    let mut out = Vec::new();
    for pair in sets.windows(2) {
        let (old, new) = (&pair[0], &pair[1]);
        for name in &old.1 {
            if !new.1.contains(name) {
                out.push(format!("{}/{name}.rs is not in {}", old.0, new.0));
            }
        }
    }
    out
}

/// Each kept set is whole: its count is the one frozen in [`KEPT_COUNTS`],
/// and its names hold every name of the set before it, the current set
/// every name of the newest kept one (review S4).
#[test]
fn every_kept_set_is_whole() {
    if regenerating() {
        return;
    }
    let sets = previous_sets();
    let counts: Vec<(u32, usize)> = sets.iter().map(|(s, set)| (*s, set.len())).collect();
    assert_eq!(
        counts, KEPT_COUNTS,
        "the kept sets' counts of fixtures (schema, count) are not the frozen ones: a kept \
         file was deleted or added. A new kept set adds its line to KEPT_COUNTS"
    );
    let names = |set: &[&Fixture]| set.iter().map(|f| f.name).collect::<Vec<&str>>();
    let mut chain: Vec<(String, Vec<&str>)> = sets
        .iter()
        .map(|(s, set)| (format!("v{s}"), names(set)))
        .collect();
    chain.push(("the current set".to_owned(), names(frozen::ALL)));
    let dropped = names_dropped(&chain);
    assert!(
        dropped.is_empty(),
        "each set holds every case of the set before it:\n{}",
        dropped.join("\n")
    );
}

/// The chain check on its own: a name the next set lacks is named.
#[test]
fn a_set_missing_a_case_of_the_one_before_is_named() {
    let set = |label: &str, names: &[&'static str]| (label.to_owned(), names.to_vec());
    let whole = [
        set("v1", &["a", "b"]),
        set("v2", &["a", "b", "c"]),
        set("now", &["a", "b", "c", "d"]),
    ];
    assert!(names_dropped(&whole).is_empty());
    let partial = [
        set("v1", &["a", "b"]),
        set("v2", &["a", "c"]),
        set("now", &["a", "c"]),
    ];
    assert_eq!(names_dropped(&partial), ["v1/b.rs is not in v2"]);
}

/// The on-disk check behind the coverage test, proven on a directory of
/// its own (review round 5, D2): a schema with no set, or an empty one, is
/// refused; a set with fixtures passes and is counted.
#[test]
fn a_kept_schema_without_its_set_on_disk_is_refused() {
    let root =
        std::env::temp_dir().join(format!("rheon-state-fixture-sets-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    let missing = set_on_disk(&root, 44).unwrap_err();
    assert!(
        missing.contains("schema 44 has no set") && missing.contains("v44"),
        "{missing}"
    );
    let set = root.join("v44");
    std::fs::create_dir_all(&set).unwrap();
    std::fs::write(set.join("notes.txt"), "not a fixture\n").unwrap();
    let empty = set_on_disk(&root, 44).unwrap_err();
    assert!(empty.contains("no fixture"), "{empty}");
    std::fs::write(set.join("ewridge.rs"), "// a fixture\n").unwrap();
    assert_eq!(set_on_disk(&root, 44), Ok(1));
    std::fs::write(set.join("sgd.rs"), "// a fixture\n").unwrap();
    assert_eq!(set_on_disk(&root, 44), Ok(2));
    assert!(set_on_disk(&root, 43).is_err(), "another schema has no set");
    std::fs::remove_dir_all(&root).unwrap();
}

/// Each kept schema's fixtures, held to the loader the change after it
/// shipped (the module doc; review round 5, D1 and D2): every one loads,
/// goes on to the bit against its own frozen continuation -- as it did
/// when written -- and, loaded then saved, writes the current schema's
/// fixture bytes of the same case: the same case over the same rows, so
/// the converted state is the one this build writes, field for field (the
/// exact-conversion check). The sets must share a writer, since a libm
/// bit in a decayed sum differs by platform. Nothing to run before 1.0,
/// where no set is kept.
#[test]
fn every_previous_fixture_loads_goes_on_and_converts_to_the_current_bytes() {
    if regenerating() {
        return;
    }
    for (schema, set) in previous_sets() {
        assert!(!set.is_empty(), "schema {schema}: an empty set");
        for f in set {
            let what = format!("v{schema}/{}", f.name);
            assert_eq!(f.schema, schema, "{what}: a fixture of schema {}", f.schema);
            let case = case_of(f);
            let mut m = load(&case, &f.bytes()).unwrap_or_else(|e| {
                panic!("{what}: the state does not load through this build's loader: {e}")
            });
            let current = frozen::ALL
                .iter()
                .find(|c| c.name == f.name)
                .unwrap_or_else(|| {
                    panic!("{what}: the current set has no fixture of that case; a kept case stays")
                });
            assert_eq!(
                current.writer, f.writer,
                "{what}: written on {}, the current set on {}: the sets are written on one \
                 platform, or their bytes differ in libm's last bit",
                f.writer, current.writer
            );
            let windowed = holds_a_window(&f.bytes());
            let moved = MOVED.iter().find(|(n, _)| *n == f.name).map(|&(_, b)| b);
            let renumbered = RENUMBERED
                .iter()
                .find(|(n, _, _)| *n == f.name && schema == 65)
                .map(|(_, b, slots)| (*b, slots.clone()));
            let within = CONTINUES_WITHIN
                .iter()
                .find(|(n, _, _)| *n == f.name)
                .map(|(_, b, slots)| (*b, slots.clone()))
                .or_else(|| renumbered.clone());
            // Seeded by the `lloyd` rule before task 278 (plain k-means++
            // starts, ten Lloyd passes): the rule since finds the same
            // centres in another order (the doc above `previous`).
            let reseeded = f.name == "kmeans_warming";
            // A `full` or `diag` `hmm` of 69 or earlier goes on under the
            // prior ([`UNDER_THE_PRIOR`]).
            let under_prior = schema < 70 && UNDER_THE_PRIOR.contains(&f.name);
            // A `deco` of 71 or earlier restarts its level ([`RESTARTED_LEVEL`]).
            let restarted = schema < 72 && RESTARTED_LEVEL.contains(&f.name);
            // A `garch` of 71 within a start that cleared its rows goes on
            // under the current start rule ([`KEPT_START`]).
            let kept_start = schema < 72 && KEPT_START.contains(&f.name);
            // A weighted `sgd` or `pa` under `standardize` of 68 or earlier
            // reads its rows in the stream's own unit of weight since review
            // C1 ([`READ_ANEW`]); a set of 69 was written by C1's read.
            let read_anew = READ_ANEW
                .iter()
                .find(|(n, _)| *n == f.name && schema < 69)
                .map(|&(_, b)| b);
            let kept = rmpv::decode::read_value(&mut &f.bytes()[..]).unwrap();
            let fresh = started_fresh(f.name, schema);
            // A `holt` of 61-63 keeps no line fit (the doc above `previous`).
            if fresh {
                let saved = named(&m.state());
                let again = load(&case, &saved).unwrap_or_else(|e| panic!("{what}: {e}"));
                assert!(
                    named(&again.state()) == saved,
                    "{what}: saved, loaded, saved"
                );
                let decode = |b: &[u8]| rmpv::decode::read_value(&mut &b[..]).unwrap();
                started_fresh_rule(&kept, &decode(&saved), &decode(&current.bytes()), &what);
            } else if under_prior || restarted || kept_start {
                let saved = named(&m.state());
                let again = load(&case, &saved).unwrap_or_else(|e| panic!("{what}: {e}"));
                assert!(
                    named(&again.state()) == saved,
                    "{what}: saved, loaded, saved"
                );
                let decode = |b: &[u8]| rmpv::decode::read_value(&mut &b[..]).unwrap();
                if restarted {
                    restarted_level_rule(&kept, &decode(&saved), &decode(&current.bytes()), &what);
                } else if under_prior {
                    under_the_prior_rule(&kept, &decode(&saved), &what);
                }
            } else if f.name == "kmeans" || f.name == "holt" {
                let saved = named(&m.state());
                let again = load(&case, &saved).unwrap_or_else(|e| panic!("{what}: {e}"));
                assert!(
                    named(&again.state()) == saved,
                    "{what}: saved, loaded, saved"
                );
            } else if schema <= 64
                && (sketches(&kept, "anchor") > 0 || holds_for_a_refit(&current.bytes()))
            {
                kept_scale_rule(&case, &kept, &named(&m.state()), &what);
            } else if (windowed && schema < 65) || moved.is_some() {
                // A window's snapshots from before task 275 carry no running
                // products (the doc above `previous`): converted, they say
                // so, and the state saves bytes that load back to the same.
                // A moved case's (`MOVED`) state is the old arithmetic's,
                // and saves bytes that load back to the same.
                let saved = named(&m.state());
                assert_eq!(holds_a_window(&saved), windowed, "{what}: the window");
                let again = load(&case, &saved).unwrap_or_else(|e| panic!("{what}: {e}"));
                assert!(
                    named(&again.state()) == saved,
                    "{what}: saved, loaded, saved"
                );
                // A window's converted ring lacks the running products, as
                // above, so a windowed case is not held field for field.
                if let Some(bound) = moved.filter(|_| !windowed) {
                    let decode = |b: &[u8]| rmpv::decode::read_value(&mut &b[..]).unwrap();
                    let at = |path: &str| moved_bound_at(f.name, bound, path);
                    if renumbered.is_some() {
                        // A schema-65 window's count is the ring's length
                        // ([`RENUMBERED`]), and its floats are within the bound.
                        renumbered_rule(&case, &saved, &current.bytes(), &at, &what);
                    } else {
                        same_but_floats(&decode(&saved), &decode(&current.bytes()), &at, &what);
                    }
                }
            } else if read_anew.is_some() {
                // Its state is the old read's, not the current fixture's
                // (whose rows before the save were read anew), so it is held
                // to saving bytes that load back to the same.
                let saved = named(&m.state());
                let again = load(&case, &saved).unwrap_or_else(|e| panic!("{what}: {e}"));
                assert!(
                    named(&again.state()) == saved,
                    "{what}: saved, loaded, saved"
                );
            } else if schema <= 70 && COUNTED_ZERO_WEIGHT.contains(&f.name) {
                let saved = named(&m.state());
                let again = load(&case, &saved).unwrap_or_else(|e| panic!("{what}: {e}"));
                assert!(
                    named(&again.state()) == saved,
                    "{what}: saved, loaded, saved"
                );
                counted_zero_weight_rule(&saved, &current.bytes(), &what);
            } else if NEW_STATISTICS.contains(&f.name) && named(&m.state()) != current.bytes() {
                // The old detector's cycle ([`NEW_STATISTICS`]): the
                // sequential case alone, its state the kept one's.
                assert_eq!(f.name, "corrchange_sequential", "{what}: converts exactly");
                let saved = named(&m.state());
                let again = load(&case, &saved).unwrap_or_else(|e| panic!("{what}: {e}"));
                assert!(
                    named(&again.state()) == saved,
                    "{what}: saved, loaded, saved"
                );
            } else if schema < 72 && MATCHED_FROM_LOAD.contains(&f.name) {
                let saved = named(&m.state());
                let again = load(&case, &saved).unwrap_or_else(|e| panic!("{what}: {e}"));
                assert!(
                    named(&again.state()) == saved,
                    "{what}: saved, loaded, saved"
                );
                let decode = |b: &[u8]| rmpv::decode::read_value(&mut &b[..]).unwrap();
                assert_eq!(
                    without_matched(&decode(&saved)),
                    without_matched(&decode(&current.bytes())),
                    "{what}: converted, but for the sides over the matched rows"
                );
            } else if schema <= 71 && MOMENTS_GAP.contains(&f.name) {
                // The moments' gap ([`MOMENTS_GAP`]): today's reading, held
                // to loading back the same.
                let saved = named(&m.state());
                let again = load(&case, &saved).unwrap_or_else(|e| panic!("{what}: {e}"));
                assert!(
                    named(&again.state()) == saved,
                    "{what}: saved, loaded, saved"
                );
                assert!(saved != current.bytes(), "{what}: now converts exactly");
            } else if renumbered.is_some() {
                assert_eq!(schema, 65, "{what}: RENUMBERED is for schema 65's windows");
                renumbered_rule(&case, &named(&m.state()), &current.bytes(), &|_| 0.0, &what);
            } else {
                assert!(
                    named(&m.state()) == current.bytes(),
                    "{what}: loaded, the state does not save the current schema's fixture bytes \
                     of the same case: the loader converts it to another state than this build \
                     reaches from the same rows"
                );
            }
            let tol = if f.writer == writer() { 0.0 } else { TOL };
            let mut relabel: Vec<(u64, u64)> = Vec::new();
            let mut relearned = false;
            // Under the new statistics ([`NEW_STATISTICS`]): a state that
            // converts to the current fixture's goes on as that fixture does,
            // every output to the bit; one that does not, by its frozen
            // `weight_sum`.
            let new_statistics = NEW_STATISTICS.contains(&f.name);
            let exact = new_statistics && named(&m.state()) == current.bytes();
            let current_outs = current.outs();
            // The window's ridge ([`WINDOW_RIDGE`]): as the current fixture.
            let window_ridge = schema < 72 && WINDOW_RIDGE.contains(&f.name);
            for (i, (row, want)) in f.rows().iter().zip(f.outs()).enumerate() {
                if window_ridge {
                    let got = m.step(row);
                    let tol = if current.writer == writer() { 0.0 } else { TOL };
                    // A ring from before the running products (schema 61
                    // to 64) reads in its last bits as a fresh run's does.
                    let tol = if windowed && schema < 65 && i > 0 {
                        tol.max(TOL)
                    } else {
                        tol
                    };
                    same_out(
                        &format!("{what}, row {i} after the save"),
                        &got,
                        &current_outs[i],
                        tol,
                    );
                    continue;
                }
                if new_statistics {
                    let got = m.step(row);
                    let what = format!("{what}, row {i} after the save");
                    if exact {
                        let tol = if current.writer == writer() { 0.0 } else { TOL };
                        same_out(&what, &got, &current_outs[i], tol);
                    } else {
                        assert!(agree(got.n_eff, want.n_eff, tol), "{what}: weight_sum");
                    }
                    continue;
                }
                if reseeded {
                    let got = m.step(row);
                    let (g, w) = (got.pred[0], want.pred[0]);
                    if !(g.is_nan() && w.is_nan()) {
                        let pair = (w.to_bits(), g.to_bits());
                        assert!(
                            relabel.iter().all(|&(a, b)| (a == pair.0) == (b == pair.1)),
                            "{what}, row {i} after the save: label {w} read as {g}, not one \
                             relabelling of the frozen labels ({relabel:?})"
                        );
                        relabel.push(pair);
                    }
                    let mut want = want.clone();
                    want.pred[0] = g;
                    same_out(&format!("{what}, row {i} after the save"), &got, &want, tol);
                    continue;
                }
                // Under a window, the first read is the saved state's to the
                // bit; once a snapshot taken after the load is the boundary,
                // its products decay it, and the reads part from the old
                // build's in their last bits, as a fresh run's do.
                let tol = if windowed && schema < 65 && i > 0 {
                    tol.max(TOL)
                } else {
                    tol
                };
                let base = read_anew.map_or(tol, |b| tol.max(b));
                let tol = moved.map_or(base, |b| base.max(b));
                let got = m.step(row);
                // A `garch` of before the horizons reported one forecast
                // (3 slots a target); this build reports its sum beside it
                // (4), which the old build did not, so the sums are dropped
                // from every comparison below and the rest goes on to the
                // bit (`narrowed`).
                let one_horizon = f.variant == "Garch"
                    && schema <= 71
                    && got.pred.len() == want.pred.len() / 3 * 4
                    && got.pred.len() != want.pred.len();
                let got = if one_horizon { narrowed(&got) } else { got };
                if fresh {
                    let what = format!("{what}, row {i} after the save");
                    assert!(agree(got.n_eff, want.n_eff, tol), "{what}: weight_sum");
                    assert_eq!(got.pred.len(), want.pred.len(), "{what}");
                    for (g, w) in got.pred.iter().zip(&want.pred) {
                        assert_eq!(g.is_nan(), w.is_nan(), "{what}: {g} against {w}");
                        if w.is_finite() {
                            let rel = (g - w).abs() / w.abs().max(1.0);
                            assert!(rel <= STARTED_FRESH_BOUND, "{what}: {g} against {w}");
                        }
                    }
                    continue;
                }
                if restarted {
                    // `weight_sum` and `u` to the bit; `rho` and `loglik` null
                    // until the level relearns, then numbers.
                    let what = format!("{what}, row {i} after the save");
                    assert!(agree(got.n_eff, want.n_eff, tol), "{what}: weight_sum");
                    assert_eq!(got.pred.len(), 3, "{what}: an unblocked case");
                    assert!(agree(got.pred[0], want.pred[0], tol), "{what}: u");
                    relearned |= got.pred[1].is_finite();
                    assert_eq!(
                        got.pred[1].is_finite(),
                        relearned,
                        "{what}: rho {}",
                        got.pred[1]
                    );
                    if got.pred[1].is_nan() {
                        assert!(got.pred[2].is_nan(), "{what}: loglik without a level");
                    }
                    continue;
                }
                if under_prior || kept_start {
                    // `weight_sum` to the bit, and a number wherever the
                    // frozen continuation has one (and, for a kept start,
                    // nowhere else either: it still waits).
                    let what = format!("{what}, row {i} after the save");
                    assert!(agree(got.n_eff, want.n_eff, tol), "{what}: weight_sum");
                    assert_eq!(got.pred.len(), want.pred.len(), "{what}");
                    for (g, w) in got.pred.iter().zip(&want.pred) {
                        assert_eq!(g.is_nan(), w.is_nan(), "{what}: {g} against {w}");
                    }
                    continue;
                }
                if let Some((_, slots)) = &within {
                    assert!(
                        slots.end <= got.pred.len(),
                        "{what}: CONTINUES_WITHIN names slots past the predictions"
                    );
                }
                same_out_by(
                    &format!("{what}, row {i} after the save"),
                    &got,
                    &want,
                    |n| {
                        // A moved case's bound on the slots it covers
                        // ([`MOVED_ONLY`]), the unmoved tolerance elsewhere.
                        let tol = moved.map_or(tol, |b| moved_slot_bound(f.name, b, base, n));
                        match &within {
                            Some((b, slots)) if slots.contains(&n) => tol.max(*b),
                            _ => tol,
                        }
                    },
                );
            }
            assert!(!restarted || relearned, "{what}: the level never relearned");
        }
    }
}

/// The maps in `v`, at any depth, that hold the key `key`: the sketches a
/// state keeps that are anchored (`"anchor"`), or marked to anchor at their
/// first value (`"anchor_by_first_value"`).
fn sketches(v: &rmpv::Value, key: &str) -> usize {
    match v {
        rmpv::Value::Map(m) => {
            // A quantile sketch's: beside its `band` (a window's operator
            // keeps an `anchor` of its own).
            let has = |name: &str| m.iter().any(|(k, _)| k.as_str() == Some(name));
            usize::from(has(key) && has("band"))
                + m.iter().map(|(_, v)| sketches(v, key)).sum::<usize>()
        }
        rmpv::Value::Array(a) => a.iter().map(|v| sketches(v, key)).sum(),
        _ => 0,
    }
}

/// Whether a state's refit buffers keep each row's `|e|` (`resid`, a list,
/// empty once the refit is done) or a histogram holds its deviations
/// (`held`, with its `pending` decay): the forms a state of schema 64 has
/// none of, and the mark of a case whose rows before the save were scaled
/// by the rule since.
fn holds_for_a_refit(bytes: &[u8]) -> bool {
    fn any(v: &rmpv::Value) -> bool {
        match v {
            rmpv::Value::Map(m) => m.iter().any(|(k, v)| {
                // A refit buffer's: beside its `learned` count.
                (k.as_str() == Some("resid")
                    && v.is_array()
                    && m.iter().any(|(k, _)| k.as_str() == Some("learned")))
                    || (k.as_str() == Some("held") && at(v, &["pending"]).is_some())
                    || any(v)
            }),
            rmpv::Value::Array(a) => a.iter().any(any),
            _ => false,
        }
    }
    any(&rmpv::decode::read_value(&mut &bytes[..]).unwrap())
}

/// A kept state of schema 64, `kept`, converted to `saved`, whose scale the
/// old rule kept (the doc above `previous`): every anchored sketch is
/// marked to keep anchoring at its first value and nothing else is; no
/// buffer or histogram holds an `|e|` or deviations for a refit; and the
/// state saves bytes that load back to the same.
/// A kept `audit` state, converted (`saved`), against the current
/// fixture's ([`COUNTED_ZERO_WEIGHT`]): the same but for its columns,
/// pairs and clock, which counted the rows of weight 0 the current state
/// leaves out, so each column read at least as many rows and the clock at
/// least as many steps.
fn counted_zero_weight_rule(saved: &[u8], current: &[u8], what: &str) {
    use rmpv::Value;
    fn audit(v: &Value) -> Option<&Vec<(Value, Value)>> {
        match v {
            Value::Map(m) if m.iter().any(|(k, _)| k.as_str() == Some("columns")) => Some(m),
            Value::Map(m) => m.iter().find_map(|(_, v)| audit(v)),
            Value::Array(a) => a.iter().find_map(audit),
            _ => None,
        }
    }
    fn rows(m: &[(Value, Value)]) -> Vec<u64> {
        let (_, Value::Array(cols)) = m
            .iter()
            .find(|(k, _)| k.as_str() == Some("columns"))
            .unwrap()
        else {
            panic!("columns are a list")
        };
        cols.iter()
            .map(|c| {
                let Value::Map(c) = c else {
                    panic!("a column is a map")
                };
                let get = |k: &str| {
                    c.iter()
                        .find(|(n, _)| n.as_str() == Some(k))
                        .map(|(_, v)| v)
                        .unwrap_or_else(|| panic!("{k}"))
                };
                let Value::Map(moments) = get("moments") else {
                    panic!("moments")
                };
                let n = moments
                    .iter()
                    .find(|(k, _)| k.as_str() == Some("n"))
                    .and_then(|(_, v)| v.as_u64())
                    .unwrap();
                ["null", "nan", "pos_inf", "neg_inf", "beyond"]
                    .iter()
                    .map(|k| get(k).as_u64().unwrap())
                    .sum::<u64>()
                    + n
            })
            .collect()
    }
    let decode = |b: &[u8]| rmpv::decode::read_value(&mut &b[..]).unwrap();
    let (kept, now) = (decode(saved), decode(current));
    let (kept, now) = (audit(&kept).unwrap(), audit(&now).unwrap());
    assert_eq!(kept.len(), now.len(), "{what}: the fields");
    for ((k, a), (_, b)) in kept.iter().zip(now) {
        if !matches!(k.as_str(), Some("columns" | "pairs" | "clock")) {
            assert_eq!(a, b, "{what}: {k}");
        }
    }
    // And its clock, which counted the steps of the rows of weight 0 before
    // the save (PLAN 539: such a row counts in no clock).
    let steps = |m: &[(Value, Value)]| {
        m.iter()
            .find(|(k, _)| k.as_str() == Some("clock"))
            .and_then(|(_, c)| match c {
                Value::Map(c) => c
                    .iter()
                    .find(|(k, _)| k.as_str() == Some("steps"))
                    .and_then(|(_, v)| v.as_u64()),
                _ => None,
            })
    };
    let (sa, sb) = (steps(kept), steps(now));
    assert!(
        sa >= sb,
        "{what}: the kept clock counted {sa:?} steps, the current {sb:?}"
    );
    let (a, b) = (rows(kept), rows(now));
    assert!(
        a.iter().zip(&b).all(|(a, b)| a >= b) && a != b,
        "{what}: the kept columns read {a:?} rows, the current {b:?}"
    );
}

fn kept_scale_rule(case: &Case, kept: &rmpv::Value, saved: &[u8], what: &str) {
    const MARK: &str = "anchor_by_first_value";
    let converted = rmpv::decode::read_value(&mut &saved[..]).unwrap();
    let anchored = sketches(kept, "anchor");
    assert_eq!(
        sketches(&converted, MARK),
        anchored,
        "{what}: each of the {anchored} anchored sketches is marked, and no other"
    );
    assert_eq!(
        sketches(kept, MARK),
        0,
        "{what}: a kept state carries no mark"
    );
    assert!(
        !holds_for_a_refit(saved),
        "{what}: held for a refit by the new rule"
    );
    let again = load(case, saved).unwrap_or_else(|e| panic!("{what}: {e}"));
    assert!(
        named(&again.state()) == saved,
        "{what}: saved, loaded, saved"
    );
}

/// A `corrchange(kind = "window")` state of schema 65, converted to
/// `saved` ([`RENUMBERED`]): numbered from its ring's oldest row, so its
/// `ring_rows` is the ring's length, and every other field is the current
/// fixture's, `current`, exactly -- each float within `bound` of its twin
/// by [`agree`]'s measure, 0 for the bit -- and the state saves bytes that
/// load back to the same.
fn renumbered_rule(
    case: &Case,
    saved: &[u8],
    current: &[u8],
    bound: &dyn Fn(&str) -> f64,
    what: &str,
) {
    let decode = |b: &[u8]| rmpv::decode::read_value(&mut &b[..]).unwrap();
    let (got, want) = (decode(saved), decode(current));
    let model = ["model", "CorrChange"];
    let count =
        |v: &rmpv::Value| at(v, &[model[0], model[1], "ring_rows"]).and_then(rmpv::Value::as_u64);
    let ring = at(&got, &[model[0], model[1], "ring"])
        .and_then(rmpv::Value::as_array)
        .map_or(0, Vec::len) as u64;
    assert_eq!(
        count(&got),
        Some(ring),
        "{what}: numbered from the ring's oldest row"
    );
    let taken = count(&want).unwrap_or_else(|| panic!("{what}: the current fixture has no count"));
    assert!(
        taken > ring,
        "{what}: the current fixture's ring has taken no more rows than it holds"
    );
    // The count aside, the converted state is the current one's.
    fn same_but_count(
        got: &rmpv::Value,
        want: &rmpv::Value,
        bound: &dyn Fn(&str) -> f64,
        what: &str,
    ) {
        use rmpv::Value as V;
        match (got, want) {
            (V::Map(g), V::Map(w)) => {
                assert_eq!(g.len(), w.len(), "{what}: a map of another size");
                for ((gk, gv), (wk, wv)) in g.iter().zip(w) {
                    assert_eq!(gk, wk, "{what}: another key");
                    if gk.as_str() != Some("ring_rows") {
                        same_but_count(gv, wv, bound, &format!("{what}.{gk}"));
                    }
                }
            }
            (V::Array(g), V::Array(w)) => {
                assert_eq!(g.len(), w.len(), "{what}: an array of another length");
                for (i, (gv, wv)) in g.iter().zip(w).enumerate() {
                    same_but_count(gv, wv, bound, &format!("{what}[{i}]"));
                }
            }
            (V::F64(g), V::F64(w)) => {
                let b = bound(what);
                assert!(
                    agree(*g, *w, b),
                    "{what}: {g:e} against {w:e} (bound {b:e})"
                );
            }
            _ => assert_eq!(got, want, "{what}"),
        }
    }
    same_but_count(&got, &want, bound, what);
    let again = load(case, saved).unwrap_or_else(|e| panic!("{what}: {e}"));
    assert!(
        named(&again.state()) == saved,
        "{what}: saved, loaded, saved"
    );
}

/// `got` is `want` field for field -- the same map keys in the same order,
/// the same array lengths, every value but a float equal -- and each float
/// within `bound` of its twin by [`agree`]'s measure: a [`MOVED`] case's
/// converted state beside the current fixture's.
fn same_but_floats(got: &rmpv::Value, want: &rmpv::Value, bound: &dyn Fn(&str) -> f64, what: &str) {
    use rmpv::Value as V;
    match (got, want) {
        (V::Map(g), V::Map(w)) => {
            assert_eq!(g.len(), w.len(), "{what}: a map of another size");
            for ((gk, gv), (wk, wv)) in g.iter().zip(w) {
                assert_eq!(gk, wk, "{what}: another key");
                same_but_floats(gv, wv, bound, &format!("{what}.{gk}"));
            }
        }
        (V::Array(g), V::Array(w)) => {
            assert_eq!(g.len(), w.len(), "{what}: an array of another length");
            for (i, (gv, wv)) in g.iter().zip(w).enumerate() {
                same_but_floats(gv, wv, bound, &format!("{what}[{i}]"));
            }
        }
        (V::F64(g), V::F64(w)) => {
            let b = bound(what);
            assert!(
                agree(*g, *w, b),
                "{what}: {g:e} against {w:e} (bound {b:e})"
            );
        }
        _ => assert_eq!(got, want, "{what}"),
    }
}

/// Whether a saved state holds a window's ring of snapshots: a model's
/// `win.snaps.ring`, non-empty.
fn holds_a_window(bytes: &[u8]) -> bool {
    let v = rmpv::decode::read_value(&mut &bytes[..]).unwrap();
    let Some(model) = at(&v, &["model"]).and_then(rmpv::Value::as_map) else {
        return false;
    };
    model.iter().any(|(_, m)| {
        at(m, &["win", "snaps", "ring"])
            .and_then(rmpv::Value::as_array)
            .is_some_and(|r| !r.is_empty())
    })
}

/// The variant names of [`ModelState`], as serde lists them: the one place
/// the list exists outside the enum.
fn variants() -> Vec<String> {
    let err = serde_json::from_str::<ModelState>(r#"{"Nope": null}"#)
        .unwrap_err()
        .to_string();
    let quoted: Vec<&str> = err.split('`').skip(1).step_by(2).collect();
    assert_eq!(quoted[0], "Nope", "{err}");
    quoted[1..].iter().map(|s| s.to_string()).collect()
}

/// The variant names are the tags every state file carries, so a rename
/// breaks every file written before it: they are frozen. A new model adds a
/// name here, and a fixture.
#[test]
fn the_variant_names_are_frozen() {
    const FROZEN: [&str; 38] = [
        "EwCov",
        "EwRidge",
        "Rls",
        "Lasso",
        "Kalman",
        "Robust",
        "Ftrl",
        "EwCovModel",
        "Sgd",
        "Pa",
        "Holt",
        "KMeans",
        "Micro",
        "EwClass",
        "SeqTest",
        "Marginal",
        "Deco",
        "Rcov",
        "Hmm",
        "CorrChange",
        "Bocpd",
        "Audit",
        "Glm",
        "Hawkes",
        "Expectile",
        "Garch",
        "Dcc",
        "Realized",
        "StreamPca",
        "PowerVariation",
        "TDigest",
        "DdSketch",
        "EwmMoments",
        "Ar",
        "Spectral",
        "Isotonic",
        "VectorAr",
        "Krls",
        "Dependence",
    ];
    assert_eq!(
        variants(),
        FROZEN,
        "a ModelState variant was renamed, removed or added: its name is the tag in every \
         state file, so a rename needs a loader for the old tag; a new one adds its name here \
         and a case in tests/cases/mod.rs"
    );
}

/// Every variant has a fixture, and every case is frozen: a case added or
/// renamed in `tests/cases/mod.rs` needs the fixtures regenerated.
#[test]
fn every_variant_and_every_case_has_a_fixture() {
    if regenerating() {
        return;
    }
    // The fixtures are discovered in file-name order and the cases are in
    // file order: compared as sets, and counted, so a discovery that found
    // nothing fails here too.
    let mut names: Vec<&str> = frozen::ALL.iter().map(|f| f.name).collect();
    let mut cases: Vec<&str> = all().iter().map(|c| c.name).collect();
    names.sort_unstable();
    cases.sort_unstable();
    assert!(!cases.is_empty(), "no case in tests/cases/mod.rs");
    assert_eq!(
        names, cases,
        "the frozen fixtures are not the cases: regenerate with `{REGENERATE}`"
    );
    for v in variants() {
        assert!(
            frozen::ALL.iter().any(|f| f.variant == v),
            "{v}: no frozen state; add a case to tests/cases/mod.rs and regenerate"
        );
    }
    for c in all() {
        assert!(
            c.before > 0 && c.rows().len() == c.before + cases::CONTINUATION,
            "{}",
            c.name
        );
    }
}

/// `v` at a path of map keys, or `None` where a key is absent.
fn at<'a>(v: &'a rmpv::Value, path: &[&str]) -> Option<&'a rmpv::Value> {
    path.iter()
        .try_fold(v, |v, key| match (v, key.parse::<usize>()) {
            // a number steps into a list
            (rmpv::Value::Array(a), Ok(i)) => a.get(i),
            _ => v
                .as_map()?
                .iter()
                .find(|(k, _)| k.as_str() == Some(key))
                .map(|(_, v)| v),
        })
}

/// How much of a layout `v` holds: its finite, non-zero numbers, through
/// any nesting, and a non-zero 128-bit integer (a temporal [`Stamp`], which
/// msgpack writes as 16 bytes). An empty layout writes an empty vector, a
/// zero (`rls`'s `s2` before a row) or a NaN (a share no solve has fit).
fn held(v: &rmpv::Value) -> usize {
    match v {
        rmpv::Value::Array(a) => a.iter().map(held).sum(),
        rmpv::Value::Map(m) => m.iter().map(|(_, v)| held(v)).sum(),
        rmpv::Value::Binary(b) => usize::from(b.iter().any(|&x| x != 0)),
        v => usize::from(v.as_f64().is_some_and(|f| f.is_finite() && f != 0.0)),
    }
}

/// A layout a schema bump moved is written by a fixture that holds it, so
/// a change to its tags or fields cannot pass the harness unseen (review
/// round 5, D3): `sgd`'s per-loss state -- the residual scale under the
/// Huber loss (schema 40), the target's spread under the epsilon-insensitive
/// one (44) -- `ewridge`'s kept systems (38, under `set_keep_factor`),
/// `bocpd`'s warm-up rows (40), `rls`'s second weight sum `s₂` and the
/// robust models' data shares (45, docs/PLAN.md task 116). Each was empty
/// in every fixture before. The standardizing models' warm-up (47,
/// docs/PLAN.md task 206) is written twice for each of `sgd` and `pa`:
/// saved inside it, its count short of 22 and not switched, and saved past
/// it, switched, where the coefficients mean another thing. `kalman`'s
/// anchor, the squared innovations its prior is sized from and its clock
/// since an observation (49, docs/PLAN.md task 211): the last written by
/// `kalman_gap`, saved after a row whose targets are null, where `kalman`'s
/// clocks are 0, its last row having observed both targets. The clusters'
/// centres as pairs (51, docs/PLAN.md task 215, D2): each summary in a list
/// keeps its own low parts, so the check runs through every element, and
/// some of `kmeans`' centres and of `micro`'s must hold one. The row counts
/// of `ewridge`'s and `lasso`'s cross-moments (52, docs/PLAN.md task 217),
/// per target and over every row, live and in every snapshot a window
/// holds. `audit`'s state (54, docs/PLAN.md task 223 (b)): its pairs, its
/// clock's regular steps, and per column its lag pairs and a digest with
/// both centroids and values waiting; its counters' two forms, and its
/// moments over a power of two (`audit_scaled`), are
/// `an_audits_counters_are_written_in_both_forms`'. Its smallest change per
/// column and the changes at it (59, task 233). A blocked `ewridge`'s held
/// targets (62, task 257), `acc.held`, written only while a block holds a
/// target's values: `ewridge_blocked` is saved mid-block. The Huber
/// scale's two forms (63, task 271): `huber_mad` and `sgd_huber_mad` keep
/// a sketch of `|e|` per target, `huber_mad` its own scale memory, and each
/// cfg names its scale, `"mad"` there and `"rms"` in `huber` and
/// `sgd_huber`; each MAD sketch names its band (task 271 (b)). `huber`'s
/// feature sets (63, task 272 (a)): `huber_sets` names them in its cfg and
/// holds a fit per target and set; `huber_sum` asks for the sum-scale
/// prior and the emitted scale. `marginal`'s Huber location per bin (63,
/// task 272 (c)): `marginal_bins_huber` keeps each cell's robust weight and
/// each target's location and scale, and its cfg names its statistic and
/// cut. `holt`'s line fit (64, task 278, D-H): its weight, mean age and
/// spread per target, under a half-life (`holt`) and without one
/// (`holt_inf`). `marginal`'s windowed lag moments about an origin (64,
/// task 280 phase 3): `marginal_window_lags` keeps the origin and the
/// lagged sides' sums. `corrchange`'s count of the rows its window ring has
/// taken (66, task 328): `corrchange_window` is saved past the ring's
/// first row, so the count is more than the ring holds.
#[test]
fn every_layout_a_schema_moved_is_written_by_a_fixture() {
    if regenerating() {
        return;
    }
    const FORMS: [(&str, &[&str]); 243] = [
        // PLAN 569, at schema 72: `ftrl`'s scaler and warm-up under
        // `standardize`, past the warm-up's count and inside it.
        ("ftrl_standardized", &["model", "Ftrl", "scaler", "m"]),
        ("ftrl_standardized", &["model", "Ftrl", "warmup", "w"]),
        (
            "ftrl_standardized_warming",
            &["model", "Ftrl", "scaler", "c"],
        ),
        (
            "ftrl_standardized_warming",
            &["model", "Ftrl", "warmup", "w2"],
        ),
        // Schema 72, unreleased (task 488): `holt`'s fixed forms -- each season, the
        // damping, both starts, a start holding its rows, and the clocks
        // the seasonal values were learned at.
        ("holt_season_add", &["model", "Holt", "season"]),
        ("holt_season_add", &["model", "Holt", "season_at"]),
        (
            "holt_season_add",
            &["model", "Holt", "cfg", "initial_seasonal"],
        ),
        ("holt_season_mul_damped", &["model", "Holt", "season"]),
        (
            "holt_season_mul_damped",
            &["model", "Holt", "cfg", "damping_half_life"],
        ),
        ("holt_heuristic", &["model", "Holt", "season"]),
        ("holt_heuristic", &["model", "Holt", "clock"]),
        ("holt_heuristic_holding", &["model", "Holt", "held"]),
        // `power_variation` (task 555): the sums, the last raw returns, the
        // scales' rings, a window's credits and rows, a clock window's
        // stamps and clock, and a cap's overrun.
        (
            "power_variation",
            &["model", "PowerVariation", "series", "1", "hi"],
        ),
        (
            "power_variation",
            &["model", "PowerVariation", "series", "1", "raw"],
        ),
        (
            "power_variation",
            &["model", "PowerVariation", "series", "1", "rings", "1"],
        ),
        (
            "power_variation_window",
            &["model", "PowerVariation", "series", "0", "held"],
        ),
        (
            "power_variation_window",
            &["model", "PowerVariation", "series", "1", "credits"],
        ),
        (
            "power_variation_window",
            &["model", "PowerVariation", "window", "rows"],
        ),
        (
            "power_variation_clock_window",
            &["model", "PowerVariation", "window", "rows", "0", "stamp"],
        ),
        (
            "power_variation_clock_window",
            &["model", "PowerVariation", "window", "place"],
        ),
        (
            "power_variation_clock_capped",
            &["model", "PowerVariation", "over"],
        ),
        // `realized` (task 499): the sums and the last returns, Lee and
        // Mykland's ring of products, a window's terms and rows, a clock
        // window's stamps and clock, and a cap's warning and overrun.
        ("realized", &["model", "Realized", "series", "1", "hi"]),
        (
            "realized",
            &["model", "Realized", "series", "1", "lm", "products"],
        ),
        (
            "realized_window",
            &["model", "Realized", "series", "0", "held"],
        ),
        (
            "realized_window",
            &["model", "Realized", "series", "1", "held"],
        ),
        ("realized_window", &["model", "Realized", "window", "rows"]),
        (
            "realized_clock_window",
            &["model", "Realized", "window", "rows", "0", "stamp"],
        ),
        (
            "realized_clock_window",
            &["model", "Realized", "window", "place"],
        ),
        // PLAN 551, at schema 72: the ratio level's sums, the form a spec
        // builds, under each dynamics, with a pair's second denominator, and
        // the standardiser's share inside its warm-up.
        ("deco", &["model", "Deco", "ratio", "num"]),
        ("deco", &["model", "Deco", "ratio", "den"]),
        ("deco", &["model", "Deco", "ratio", "w2_share"]),
        ("deco_linear", &["model", "Deco", "ratio", "den_b"]),
        ("deco_linear", &["model", "Deco", "ratio", "lin_num"]),
        ("deco_linear", &["model", "Deco", "ratio", "lin_den"]),
        ("deco_linear", &["model", "Deco", "ratio", "lin_den_b"]),
        ("deco_warming", &["model", "Deco", "ratio", "w2_share"]),
        // Schema 72 (task 554, parts 4 and 5): `ewm_moments`' summaries over
        // legs, its lag buffers (and their stamps on a clock), the clock's
        // place and a window's ring (and its stamps on a clock), and the
        // boundary of the window's two stacks, `flip`.
        ("ewm_moments_lags", &["model", "EwmMoments", "sums"]),
        // PLAN 578j, at schema 72 before any release: a summary's scale, the
        // form a stream whose moments pass the largest double leaves.
        (
            "ewm_moments_scaled",
            &["model", "EwmMoments", "sums", "0", "e"],
        ),
        ("ewm_moments_lags", &["model", "EwmMoments", "bufs"]),
        (
            "ewm_moments_clock_lags",
            &["model", "EwmMoments", "bufs", "0", "stamps"],
        ),
        ("ewm_moments_clock_lags", &["model", "EwmMoments", "place"]),
        (
            "ewm_moments_window",
            &["model", "EwmMoments", "ring", "vals"],
        ),
        (
            "ewm_moments_clock_window",
            &["model", "EwmMoments", "ring", "vals"],
        ),
        (
            "ewm_moments_clock_window",
            &["model", "EwmMoments", "ring", "stamps"],
        ),
        (
            "ewm_moments_window",
            &["model", "EwmMoments", "ring", "flip"],
        ),
        (
            "ewm_moments_clock_window",
            &["model", "EwmMoments", "ring", "flip"],
        ),
        // Schema 70 (rule 5's twenty-second exception): the stabilized
        // prior's `ŝ` and `V` under `full` in each form and under `diag`,
        // the factor form's `f`, and a spec's own `prior_rows`.
        ("hmm_full", &["model", "Hmm", "stabilized", "scale"]),
        ("hmm_full", &["model", "Hmm", "stabilized", "scale_weight"]),
        (
            "hmm_full",
            &["model", "Hmm", "stabilized", "factored_prior"],
        ),
        ("hmm_full_narrow", &["model", "Hmm", "stabilized", "scale"]),
        ("hmm_diag", &["model", "Hmm", "stabilized", "scale"]),
        ("hmm_diag", &["model", "Hmm", "stabilized", "scale_weight"]),
        ("hmm_full_warming", &["model", "Hmm", "cfg", "prior_rows"]),
        // Schema 72 (review 2026-11-23, HMM-3): given states of the same
        // diagonal with variances that differ, `ŝ` in units of them.
        ("hmm_given", &["model", "Hmm", "stabilized", "scale"]),
        ("hmm_given", &["model", "Hmm", "stabilized", "scale_weight"]),
        // Schema 69 (task 479): a wide `full` state's factor.
        ("hmm_full", &["model", "Hmm", "factor_states"]),
        // Schema 66 (task 327): a diagonal shape's variances alone.
        ("hmm_diag", &["model", "Hmm", "diag_states"]),
        ("ew_class_diag", &["model", "EwClass", "diag_classes"]),
        ("ew_class_window", &["model", "EwClass", "diag_classes"]),
        // Schema 66 (task 328): the rows a window's ring has taken.
        ("corrchange_window", &["model", "CorrChange", "ring_rows"]),
        (
            "marginal_window_lags",
            &["model", "Marginal", "lag", "origin", "x"],
        ),
        (
            "marginal_window_lags",
            &["model", "Marginal", "lag", "origin", "bx"],
        ),
        (
            "marginal_window_lags",
            &["model", "Marginal", "lag", "origin", "by"],
        ),
        ("holt", &["model", "Holt", "fit_weight"]),
        ("holt", &["model", "Holt", "fit_age"]),
        ("holt", &["model", "Holt", "fit_spread"]),
        ("holt_inf", &["model", "Holt", "fit_weight"]),
        ("holt_inf", &["model", "Holt", "fit_age"]),
        ("holt_inf", &["model", "Holt", "fit_spread"]),
        ("huber_sets", &["model", "Robust", "cfg", "feature_sets"]),
        (
            "marginal_bins_huber",
            &["model", "Marginal", "bins", "hist", "om"],
        ),
        (
            "marginal_bins_huber",
            &["model", "Marginal", "bins", "hist", "scales", "w"],
        ),
        (
            "marginal_bins_huber",
            &["model", "Marginal", "bins", "hist", "scales", "mad"],
        ),
        (
            "marginal_bins_huber",
            &["model", "Marginal", "cfg", "bins", "stat", "huber", "delta"],
        ),
        ("huber_mad", &["model", "Robust", "mad"]),
        (
            "huber_mad",
            &["model", "Robust", "cfg", "huber_scale_decay"],
        ),
        ("sgd_huber_mad", &["model", "Sgd", "mad"]),
        ("ewridge_blocked", &["model", "EwRidge", "acc", "held", "y"]),
        (
            "ewridge_blocked",
            &["model", "EwRidge", "acc", "held", "w_end"],
        ),
        ("ewridge", &["model", "EwRidge", "acc", "cross", "nj"]),
        ("ewridge", &["model", "EwRidge", "acc", "cross", "n"]),
        (
            "ewridge_window",
            &["model", "EwRidge", "acc", "cross", "nj"],
        ),
        ("ewridge_window", &["model", "EwRidge", "acc", "cross", "n"]),
        ("lasso", &["model", "Lasso", "acc", "cross", "nj"]),
        ("lasso", &["model", "Lasso", "acc", "cross", "n"]),
        ("lasso_window", &["model", "Lasso", "acc", "cross", "nj"]),
        ("lasso_window", &["model", "Lasso", "acc", "cross", "n"]),
        ("sgd", &["model", "Sgd", "warmup"]),
        ("sgd_warming", &["model", "Sgd", "warmup"]),
        ("pa", &["model", "Pa", "warmup"]),
        ("pa_warming", &["model", "Pa", "warmup"]),
        ("kalman", &["model", "Kalman", "anchor_hi"]),
        ("kalman", &["model", "Kalman", "anchor_lo"]),
        ("kalman", &["model", "Kalman", "anchor_scale"]),
        ("kalman", &["model", "Kalman", "basis"]),
        ("kalman_gap", &["model", "Kalman", "elapsed"]),
        ("sgd_huber", &["model", "Sgd", "sig2"]),
        ("sgd_huber", &["model", "Sgd", "wsig"]),
        ("sgd_eps", &["model", "Sgd", "spread"]),
        (
            "ewridge_leverage",
            &["model", "EwRidge", "ready", "systems"],
        ),
        ("bocpd_warming", &["model", "Bocpd", "warm"]),
        ("rls", &["model", "Rls", "s2"]),
        ("huber", &["model", "Robust", "support"]),
        ("quantile", &["model", "Robust", "support"]),
        ("audit", &["model", "Audit", "pairs"]),
        ("audit", &["model", "Audit", "clock", "regular"]),
        ("glm_refit", &["model", "Glm", "refit"]),
        ("glm_refit_done", &["model", "Glm", "refit"]),
        ("glm_cap", &["model", "Glm", "gram"]),
        // A `hawkes` window in both of its blocks (task 498): the older's
        // forward sums, the newer's sum `B`.
        ("hawkes", &["model", "Hawkes", "q"]),
        ("hawkes_window", &["model", "Hawkes", "window", "time"]),
        ("hawkes_window", &["model", "Hawkes", "window", "fwd"]),
        ("hawkes_window", &["model", "Hawkes", "window", "new_sum"]),
        ("hawkes_window", &["model", "Hawkes", "window", "weight"]),
        // Saved on a row with no event while the newer block holds three
        // rows (HAWKES-7): the clock since the newest row, and the newer
        // rows' gaps.
        (
            "hawkes_window_quiet",
            &["model", "Hawkes", "window", "last"],
        ),
        (
            "hawkes_window_quiet",
            &["model", "Hawkes", "window", "time"],
        ),
        // A window of clock units, its rows stamped: a stamp a row, and the
        // stamp the clock was last handed.
        (
            "hawkes_clock_window",
            &["model", "Hawkes", "window", "stamp"],
        ),
        (
            "hawkes_clock_window",
            &["model", "Hawkes", "window", "track", "last"],
        ),
        // `expectile` (task 501): the one pass's sums, the refit's buffer,
        // a window of rows and one of clock units with its stamps, and the
        // clock rows of weight 0 left uncounted.
        ("expectile", &["model", "Expectile", "gram"]),
        ("expectile", &["model", "Expectile", "clock", "carry"]),
        ("expectile_refit", &["model", "Expectile", "refit"]),
        ("expectile_refit_done", &["model", "Expectile", "refit"]),
        ("expectile_window", &["model", "Expectile", "window", "x"]),
        ("expectile_window", &["model", "Expectile", "window", "y"]),
        (
            "expectile_clock_window",
            &["model", "Expectile", "window", "stamp"],
        ),
        (
            "expectile_clock_window",
            &["model", "Expectile", "clock", "last"],
        ),
        // `garch`'s series (task 492), a path into the list by index: its
        // start's buffer, then each process's lags once started, the
        // targeted GARCH's Huber score, a skewed t's shape, EGARCH's
        // binned residuals and FIGARCH's ring, and a start of a clock span's
        // stamp. Started on realistic series (PLAN 550): the degenerate
        // cases' starts now wait (PLAN 548), and `garch_started` holds the
        // fit such a wait withheld.
        ("garch", &["model", "Garch", "series", "1", "buffer"]),
        ("garch_started", &["model", "Garch", "series", "0", "theta"]),
        // the forecasts' horizons (task 559): several, out of order, and a
        // work cap of the spec's own
        (
            "garch_realistic",
            &["model", "Garch", "cfg", "forecast_rows"],
        ),
        (
            "garch_figarch_realistic",
            &["model", "Garch", "cfg", "max_horizon_work"],
        ),
        ("garch_realistic", &["model", "Garch", "series", "0", "h"]),
        (
            "garch_realistic",
            &["model", "Garch", "series", "0", "filter", "sens"],
        ),
        ("garch_t_realistic", &["model", "Garch", "series", "1", "h"]),
        (
            "garch_targeted_realistic",
            &["model", "Garch", "series", "0", "kappa"],
        ),
        (
            "garch_targeted_realistic",
            &["model", "Garch", "series", "0", "slope"],
        ),
        (
            "garch_skewt_realistic",
            &["model", "Garch", "series", "0", "theta"],
        ),
        (
            "garch_egarch_realistic",
            &["model", "Garch", "series", "0", "zbins"],
        ),
        (
            "garch_figarch_realistic",
            &["model", "Garch", "series", "0", "filter", "ring"],
        ),
        ("garch_span", &["model", "Garch", "series", "0", "buffer"]),
        ("garch_bound", &["model", "Garch", "series", "0", "buffer"]),
        // `dcc`'s stage (task 493): its start's buffer, then the targets,
        // `Q`, its sensitivities and the information, the decay carried to
        // the next row, given parameters, the margins, and a start of a
        // clock span's stamp.
        ("dcc", &["model", "Dcc", "stage", "buffer"]),
        ("dcc_started", &["model", "Dcc", "stage", "qbar"]),
        ("dcc_started", &["model", "Dcc", "stage", "nbar"]),
        ("dcc_started", &["model", "Dcc", "stage", "q"]),
        ("dcc_started", &["model", "Dcc", "stage", "dq"]),
        ("dcc_started", &["model", "Dcc", "stage", "h"]),
        ("dcc_started", &["model", "Dcc", "stage", "theta"]),
        ("dcc_started", &["model", "Dcc", "stage", "lr"]),
        ("dcc_withheld", &["model", "Dcc", "stage", "buffer"]),
        ("dcc_bound", &["model", "Dcc", "stage", "buffer"]),
        ("dcc_given", &["model", "Dcc", "cfg", "params"]),
        ("dcc_margins", &["model", "Dcc", "margins", "series"]),
        ("dcc_margins", &["model", "Dcc", "stage", "buffer"]),
        ("dcc_span", &["model", "Dcc", "place"]),
        // `tdigest`'s digests (PLAN 544): a buffer before the first
        // merge, then the centroids beside it, and the weights.
        ("tdigest", &["model", "TDigest", "digests", "1", "buffer"]),
        ("tdigest", &["model", "TDigest", "target_w2"]),
        (
            "tdigest_merged",
            &["model", "TDigest", "digests", "0", "centroids"],
        ),
        (
            "tdigest_merged",
            &["model", "TDigest", "digests", "0", "buffer"],
        ),
        // `ddsketch`'s sketches (PLAN 544, part 7): both stores, then a row
        // window's ring and counts, then a clock window's stamps and clock.
        (
            "ddsketch",
            &["model", "DdSketch", "sketches", "0", "pos", "w"],
        ),
        (
            "ddsketch",
            &["model", "DdSketch", "sketches", "1", "neg", "w"],
        ),
        ("ddsketch_rows", &["model", "DdSketch", "ring", "codes"]),
        (
            "ddsketch_rows",
            &["model", "DdSketch", "sketches", "0", "pos", "n"],
        ),
        ("ddsketch_clock", &["model", "DdSketch", "ring", "stamp"]),
        ("ddsketch_decayed", &["model", "DdSketch", "cfg", "decay"]),
        (
            "ddsketch_clock",
            &["model", "DdSketch", "cfg", "window_clock"],
        ),
        // GARCH-X (task 559, part 4): the inputs' count, their running
        // mean once started, and the start's inputs within it.
        ("garch_x_realistic", &["model", "Garch", "cfg", "n_inputs"]),
        (
            "garch_x_realistic",
            &["model", "Garch", "series", "1", "xbar"],
        ),
        ("garch_x_start", &["model", "Garch", "series", "0", "xbuf"]),
        // A refit over a span of the clock (task 538): the span in each
        // cfg, glm's clock summed from the steps, and each buffer.
        ("glm_refit_clock", &["model", "Glm", "cfg", "refit_clock"]),
        ("glm_refit_clock", &["model", "Glm", "own"]),
        ("glm_refit_clock", &["model", "Glm", "refit", "0", "rows"]),
        (
            "expectile_refit_clock",
            &["model", "Expectile", "cfg", "refit_clock"],
        ),
        (
            "expectile_refit_clock",
            &["model", "Expectile", "refit", "0", "rows"],
        ),
        // Task 538, part 2: kmeans' warm-up over a span and its cadences.
        ("kmeans_spans_warming", &["model", "KMeans", "place"]),
        ("kmeans_spans_warming", &["model", "KMeans", "buf"]),
        ("kmeans_spans", &["model", "KMeans", "cfg", "warm_clock"]),
        ("kmeans_spans", &["model", "KMeans", "cfg", "update_every"]),
        (
            "kmeans_spans",
            &["model", "KMeans", "cfg", "split_merge_every"],
        ),
        ("kmeans_spans", &["model", "KMeans", "since_update"]),
        ("kmeans_spans", &["model", "KMeans", "since_split"]),
        ("hmm_warm_span", &["model", "Hmm", "cfg", "warm_clock"]),
        ("hmm_warm_span", &["model", "Hmm", "place"]),
        ("hmm_warm_span", &["model", "Hmm", "buffer"]),
        ("bocpd_warm_span", &["model", "Bocpd", "cfg", "warm_clock"]),
        ("bocpd_warm_span", &["model", "Bocpd", "place"]),
        ("bocpd_warm_span", &["model", "Bocpd", "warm"]),
        (
            "corrchange_permute_every",
            &["model", "CorrChange", "cfg", "permute_every"],
        ),
        (
            "corrchange_permute_every",
            &["model", "CorrChange", "since_draw"],
        ),
        // Task 553: each kind's span on the clock.
        (
            "corrchange_monitor_clock",
            &["model", "CorrChange", "cfg", "span_clock"],
        ),
        (
            "corrchange_monitor_clock",
            &["model", "CorrChange", "cfg", "max_state_bytes"],
        ),
        (
            "corrchange_monitor_clock",
            &["model", "CorrChange", "clock"],
        ),
        (
            "corrchange_monitor_clock",
            &["model", "CorrChange", "first"],
        ),
        (
            "corrchange_sequential_clock",
            &["model", "CorrChange", "cfg", "monitor_clock"],
        ),
        (
            "corrchange_sequential_clock",
            &["model", "CorrChange", "monitoring", "history"],
        ),
        (
            "corrchange_sequential_clock",
            &["model", "CorrChange", "monitoring", "horizon"],
        ),
        ("corrchange_window_clock", &["model", "CorrChange", "keys"]),
        (
            "corrchange_window_clock",
            &["model", "CorrChange", "perm_counts"],
        ),
        (
            "huber_refit_span",
            &["model", "Robust", "cfg", "refit_clock"],
        ),
        ("huber_refit_span", &["model", "Robust", "place"]),
        (
            "huber_refit_span",
            &["model", "Robust", "refit", "0", "rows"],
        ),
        (
            "quantile_refit_span",
            &["model", "Robust", "refit", "0", "rows"],
        ),
        (
            "sgd_huber_refit_span",
            &["model", "Sgd", "cfg", "refit_clock"],
        ),
        ("sgd_huber_refit_span", &["model", "Sgd", "place"]),
        (
            "sgd_huber_refit_span",
            &["model", "Sgd", "refit", "0", "rows"],
        ),
        // Task 538: the span's cap, settable (537's rule).
        (
            "glm_refit_clock",
            &["model", "Glm", "cfg", "max_state_bytes"],
        ),
        (
            "huber_refit_span",
            &["model", "Robust", "cfg", "max_state_bytes"],
        ),
        (
            "sgd_huber_refit_span",
            &["model", "Sgd", "cfg", "max_state_bytes"],
        ),
        // `ar` (task 486): each target's ring of lags, full and filling,
        // and its fit, solved and not yet.
        ("ar", &["model", "Ar", "rings", "1"]),
        ("ar", &["model", "Ar", "fits", "1", "beta"]),
        ("ar", &["model", "Ar", "cfg", "lags"]),
        ("ar_select", &["model", "Ar", "chosen", "1"]),
        ("ar_select", &["model", "Ar", "chosen", "0"]),
        ("ar_window", &["model", "Ar", "fits", "0", "win"]),
        // `spectral` (task 491): each target's transform and its weights'.
        ("spectral", &["model", "Spectral", "sums", "1", "x"]),
        ("spectral", &["model", "Spectral", "sums", "0", "u"]),
        // Under Lomb-Scargle, the weights' transform at twice each
        // frequency.
        ("spectral_ls", &["model", "Spectral", "sums", "1", "v"]),
        // Under a window, its blocks and the closed blocks' total.
        ("spectral_window", &["model", "Spectral", "win", "blocks"]),
        ("spectral_window", &["model", "Spectral", "win", "total"]),
        ("ar_filling", &["model", "Ar", "rings", "0"]),
        ("ar_filling", &["model", "Ar", "fits", "0", "acc"]),
        // `vector_ar`: its ring, full and filling, its fit, and what the
        // last solve read: each test's statistic, the variances, and the
        // chosen coefficients under `select`.
        ("vector_ar", &["model", "VectorAr", "ring"]),
        ("vector_ar", &["model", "VectorAr", "fit", "beta"]),
        ("vector_ar", &["model", "VectorAr", "last", "wald"]),
        ("vector_ar", &["model", "VectorAr", "last", "coef_var"]),
        ("vector_ar", &["model", "VectorAr", "cfg", "causality"]),
        ("vector_ar_select", &["model", "VectorAr", "last", "chosen"]),
        ("vector_ar_window", &["model", "VectorAr", "fit", "win"]),
        ("vector_ar_filling", &["model", "VectorAr", "ring"]),
        // `audit` HLL++ and sliding-window HLL++ (PLAN 509): plain registers,
        // sliding row-window Pareto entries, and sliding clock-window entries
        // and place.
        ("audit_hll", &["model", "Audit", "cfg", "distinct"]),
        (
            "audit_hll",
            &["model", "Audit", "hll", "sketches", "Plain", "0"],
        ),
        ("audit_hll_rows", &["model", "Audit", "cfg", "distinct"]),
        (
            "audit_hll_rows",
            &[
                "model",
                "Audit",
                "hll",
                "sketches",
                "SlidingRows",
                "cols",
                "0",
                "entries",
            ],
        ),
        ("audit_hll_clock", &["model", "Audit", "cfg", "distinct"]),
        (
            "audit_hll_clock",
            &[
                "model",
                "Audit",
                "hll",
                "sketches",
                "SlidingClock",
                "cols",
                "0",
                "entries",
            ],
        // `dependence`: decayed MI tables & scale, warm-up hold, row-windowed
        // value ring & held ranks, and clock-windowed stamps & cadence.
        (
            "dependence_mi_decayed",
            &["model", "Dependence", "mi_tables", "weights"],
        ),
        (
            "dependence_mi_decayed",
            &["model", "Dependence", "mi_tables", "x_edges"],
        ),
        (
            "dependence_mi_decayed",
            &["model", "Dependence", "mi_tables", "scale"],
        ),
        (
            "dependence_warming",
            &["model", "Dependence", "mi_hold", "xs"],
        ),
        (
            "dependence_window_rows",
            &["model", "Dependence", "val_ring", "xs"],
        ),
        (
            "dependence_window_rows",
            &["model", "Dependence", "held_spearman"],
        ),
        (
            "dependence_window_rows",
            &["model", "Dependence", "held_xi"],
        ),
        (
            "dependence_window_clock",
            &["model", "Dependence", "val_ring", "stamps"],
        ),
        (
            "dependence_window_clock",
            &["model", "Dependence", "rank_since"],
        ),
    ];
    // Each refit's first stamp, written only under a span (task 538): in
    // glm on the summed clock, in expectile an integer stamp.
    for (name, variant) in [
        ("glm_refit_clock", "Glm"),
        ("expectile_refit_clock", "Expectile"),
    ] {
        let f = frozen::ALL
            .iter()
            .find(|f| f.name == name)
            .unwrap_or_else(|| panic!("{name}: no fixture"));
        let v = rmpv::decode::read_value(&mut f.bytes().as_slice()).unwrap();
        let first = at(&v, &["model", variant, "refit", "0", "first"]).expect("a first stamp");
        assert!(!first.is_nil(), "{name}: no first stamp");
    }
    // kmeans' warm-up's first stamp (task 538, part 2), the first row's.
    let f = frozen::ALL
        .iter()
        .find(|f| f.name == "kmeans_spans_warming")
        .expect("kmeans_spans_warming");
    let v = rmpv::decode::read_value(&mut f.bytes().as_slice()).unwrap();
    let first = at(&v, &["model", "KMeans", "warm_first"]).expect("a warm-up's first stamp");
    assert!(!first.is_nil(), "kmeans_spans_warming: no first stamp");
    // The span's first stamp, an integer stamp msgpack writes as integers
    // (which `held` does not count): present, not nil.
    let f = frozen::ALL
        .iter()
        .find(|f| f.name == "garch_span")
        .expect("garch_span");
    let v = rmpv::decode::read_value(&mut f.bytes().as_slice()).unwrap();
    let first = at(&v, &["model", "Garch", "series", "0", "first"]).expect("a first stamp");
    assert!(!first.is_nil(), "garch_span: no first stamp");
    // garch's start at its bound: a ring whose oldest row is not the first
    // slot, rows let go, and the rows of its last try and since it, each
    // an integer `held` does not count.
    let f = frozen::ALL
        .iter()
        .find(|f| f.name == "garch_bound")
        .expect("garch_bound");
    let v = rmpv::decode::read_value(&mut f.bytes().as_slice()).unwrap();
    for j in ["0", "1"] {
        let series = |key: &str| {
            at(&v, &["model", "Garch", "series", j, key])
                .and_then(rmpv::Value::as_u64)
                .unwrap_or(0)
        };
        for key in ["head", "tried", "since", "dropped"] {
            assert!(series(key) > 0, "garch_bound: series {j}'s {key} is 0");
        }
    }
    // dcc's start over a span: its first stamp, the clock's 0 on this
    // stream, so present and not nil rather than held.
    let f = frozen::ALL
        .iter()
        .find(|f| f.name == "dcc_span")
        .expect("dcc_span");
    let v = rmpv::decode::read_value(&mut f.bytes().as_slice()).unwrap();
    let first = at(&v, &["model", "Dcc", "stage", "first"]).expect("a first stamp");
    assert!(!first.is_nil(), "dcc_span: no first stamp");
    // dcc's withheld start: the flag set and a refit counted (a bool and an
    // integer, which `held` does not count); its statistic is 0 where the
    // fit sits at a = 0, so present rather than held.
    let f = frozen::ALL
        .iter()
        .find(|f| f.name == "dcc_withheld")
        .expect("dcc_withheld");
    let v = rmpv::decode::read_value(&mut f.bytes().as_slice()).unwrap();
    let stage = |key: &str| at(&v, &["model", "Dcc", "stage", key]).cloned();
    assert_eq!(stage("withheld"), Some(rmpv::Value::Boolean(true)));
    assert!(
        stage("refits")
            .and_then(|r| r.as_u64())
            .is_some_and(|r| r >= 1)
    );
    assert!(stage("lr").is_some_and(|l| l.as_f64().is_some_and(f64::is_finite)));
    // dcc's start at its bound: a ring whose oldest row is not the first
    // slot, rows let go, and the rows of its last try and since it, each
    // an integer `held` does not count.
    let f = frozen::ALL
        .iter()
        .find(|f| f.name == "dcc_bound")
        .expect("dcc_bound");
    let v = rmpv::decode::read_value(&mut f.bytes().as_slice()).unwrap();
    let stage = |key: &str| {
        at(&v, &["model", "Dcc", "stage", key])
            .and_then(rmpv::Value::as_u64)
            .unwrap_or(0)
    };
    for key in ["head", "tried", "since", "dropped"] {
        assert!(stage(key) > 0, "dcc_bound: {key} is 0");
    }
    // A clock window's cap (task 499): the bytes of the row that would have
    // passed it, an integer, and the warning raised.
    let f = frozen::ALL
        .iter()
        .find(|f| f.name == "realized_clock_capped")
        .expect("realized_clock_capped");
    let v = rmpv::decode::read_value(&mut f.bytes().as_slice()).unwrap();
    let over = at(&v, &["model", "Realized", "over"]).expect("the overrun");
    assert!(over.as_u64().is_some_and(|b| b > 1_500), "{over}");
    let warned = at(&v, &["model", "Realized", "warned"]).and_then(rmpv::Value::as_bool);
    assert_eq!(warned, Some(true));
    // `deco_warming` is saved inside the standardiser's warm-up (PLAN 551):
    // its share at a fifth or more, so Kish's size at 5 or less, and no level;
    // `deco` past it.
    for (name, warm) in [("deco_warming", false), ("deco", true)] {
        let f = frozen::ALL
            .iter()
            .find(|f| f.name == name)
            .unwrap_or_else(|| panic!("{name}: no fixture"));
        let v = rmpv::decode::read_value(&mut f.bytes().as_slice()).unwrap();
        let share = at(&v, &["model", "Deco", "ratio", "w2_share"])
            .and_then(rmpv::Value::as_f64)
            .expect("a share");
        let rho = at(&v, &["model", "Deco", "rho", "0"])
            .and_then(rmpv::Value::as_f64)
            .expect("a level");
        assert_eq!(share < 0.2, warm, "{name}: share {share}");
        assert_eq!(rho.is_finite(), warm, "{name}: level {rho}");
    }
    for (name, path) in FORMS {
        let f = frozen::ALL
            .iter()
            .find(|f| f.name == name)
            .unwrap_or_else(|| panic!("{name}: no fixture; add the case and regenerate"));
        let v = rmpv::decode::read_value(&mut f.bytes().as_slice()).unwrap();
        let Some(field) = at(&v, path) else {
            panic!("{name}: the state has no {}", path.join("."));
        };
        assert!(
            held(field) > 0,
            "{name}: {} is empty, so the fixture holds nothing of the layout it is there for",
            path.join(".")
        );
        if path.last() == Some(&"warmup") {
            let switched = at(field, &["switched"]).and_then(rmpv::Value::as_bool);
            assert_eq!(
                switched,
                Some(!name.ends_with("_warming")),
                "{name}: saved on the wrong side of the switch"
            );
        }
    }
    // Each shape keeps one list (66, task 327): the diagonal ones their
    // variances alone and no whole matrix, in the model and in every window
    // snapshot; the others their matrices and no diagonal list.
    for (name, variant, list, diag) in [
        ("hmm_diag", "Hmm", "states", true),
        ("hmm", "Hmm", "states", false),
        ("ew_class_diag", "EwClass", "classes", true),
        ("ew_class_window", "EwClass", "classes", true),
        ("ew_class", "EwClass", "classes", false),
    ] {
        let f = frozen::ALL
            .iter()
            .find(|f| f.name == name)
            .unwrap_or_else(|| panic!("{name}: no fixture; add the case and regenerate"));
        let v = rmpv::decode::read_value(&mut f.bytes().as_slice()).unwrap();
        let len = |key: &str| {
            at(&v, &["model", variant, key])
                .and_then(rmpv::Value::as_array)
                .unwrap_or_else(|| panic!("{name}: the state has no {key}"))
                .len()
        };
        let (whole, part) = (len(list), len(&format!("diag_{list}")));
        assert_eq!(
            (whole == 0, part == 0),
            (diag, !diag),
            "{name}: {whole} {part}"
        );
        if let Some(ring) =
            at(&v, &["model", variant, "win", "snaps", "ring"]).and_then(rmpv::Value::as_array)
        {
            assert!(!ring.is_empty(), "{name}: the window holds no snapshot");
            for (i, item) in ring.iter().enumerate() {
                let moments = item.as_array().and_then(|t| t.get(2)).unwrap();
                let n = |key: &str| {
                    at(moments, &[key])
                        .and_then(rmpv::Value::as_array)
                        .map(Vec::len)
                };
                assert_eq!(n("classes"), Some(0), "{name}: snapshot {i}");
                assert_eq!(n("diag_classes"), Some(3), "{name}: snapshot {i}");
                assert!(
                    held(at(moments, &["diag_classes"]).unwrap()) > 0,
                    "{name}: snapshot {i}"
                );
            }
        }
    }
    // `huber_sum` asks for the sum-scale prior and the emitted scale, and
    // the rest do not (63, task 272 (a)).
    for (name, want) in [("huber_sum", true), ("huber", false), ("huber_sets", false)] {
        let f = frozen::ALL
            .iter()
            .find(|f| f.name == name)
            .unwrap_or_else(|| panic!("{name}: no fixture; add the case and regenerate"));
        let v = rmpv::decode::read_value(&mut f.bytes().as_slice()).unwrap();
        for field in ["ridge_scale", "emit_scale"] {
            let got = at(&v, &["model", "Robust", "cfg", field]).and_then(rmpv::Value::as_bool);
            assert_eq!(got, Some(want), "{name}: cfg.{field}");
        }
    }
    // Each Huber form names its scale, and the RMS's keeps no sketch.
    for (name, variant, scale) in [
        ("huber", "Robust", "rms"),
        ("huber_mad", "Robust", "mad"),
        ("sgd_huber", "Sgd", "rms"),
        ("sgd_huber_mad", "Sgd", "mad"),
    ] {
        let f = frozen::ALL
            .iter()
            .find(|f| f.name == name)
            .unwrap_or_else(|| panic!("{name}: no fixture; add the case and regenerate"));
        let v = rmpv::decode::read_value(&mut f.bytes().as_slice()).unwrap();
        let got = at(&v, &["model", variant, "cfg", "huber_scale"]).and_then(rmpv::Value::as_str);
        assert_eq!(got, Some(scale), "{name}: cfg.huber_scale");
        let sketches = at(&v, &["model", variant, "mad"])
            .and_then(rmpv::Value::as_array)
            .unwrap_or_else(|| panic!("{name}: the state has no mad"));
        assert_eq!(sketches.is_empty(), scale == "rms", "{name}: mad");
        // A MAD's sketch is bounded and names its band (docs/PLAN.md task
        // 271 (b)); a sketch without one loads unbounded.
        for q in sketches {
            let band = at(q, &["band"]).and_then(rmpv::Value::as_u64);
            let want = u64::from(rheon_core::MAD_BAND_OCTAVES);
            assert_eq!(band, Some(want), "{name}: mad.band");
        }
    }
    // Every snapshot a window holds keeps the counts as they stood before
    // its row; past the first rows, a target's are above 0.
    for (name, variant) in [("ewridge_window", "EwRidge"), ("lasso_window", "Lasso")] {
        let f = frozen::ALL
            .iter()
            .find(|f| f.name == name)
            .unwrap_or_else(|| panic!("{name}: no fixture; add the case and regenerate"));
        let v = rmpv::decode::read_value(&mut f.bytes().as_slice()).unwrap();
        let ring = at(&v, &["model", variant, "win", "snaps", "ring"])
            .and_then(rmpv::Value::as_array)
            .unwrap_or_else(|| panic!("{name}: the state has no window ring"));
        assert!(!ring.is_empty(), "{name}: the window holds no snapshot");
        for (i, item) in ring.iter().enumerate() {
            let moments = item.as_array().and_then(|t| t.get(2)).unwrap_or_else(|| {
                panic!("{name}: snapshot {i} is no (clock, stamp, moments, products)")
            });
            for field in ["nj", "n"] {
                let Some(count) = at(moments, &["acc", "cross", field]) else {
                    panic!("{name}: snapshot {i} has no acc.cross.{field}");
                };
                assert!(
                    held(count) > 0,
                    "{name}: snapshot {i}'s acc.cross.{field} is empty"
                );
            }
        }
    }
    // Every windowed model keeps the running products of its decays, one
    // per decay its window is read with -- two for `lasso`, its selection's
    // beside its moments' -- and every snapshot the products it was taken
    // at (docs/PLAN.md task 275).
    for (name, variant, decays) in [
        ("ewridge_window", "EwRidge", 1),
        ("lasso_window", "Lasso", 2),
        ("ew_cov_window", "EwCovModel", 1),
        ("ew_class_window", "EwClass", 1),
        ("marginal_window", "Marginal", 1),
    ] {
        let f = frozen::ALL
            .iter()
            .find(|f| f.name == name)
            .unwrap_or_else(|| panic!("{name}: no fixture; add the case and regenerate"));
        let v = rmpv::decode::read_value(&mut f.bytes().as_slice()).unwrap();
        let snaps = at(&v, &["model", variant, "win", "snaps"])
            .unwrap_or_else(|| panic!("{name}: the state has no window"));
        let live = at(snaps, &["decay"])
            .and_then(rmpv::Value::as_array)
            .unwrap_or_else(|| panic!("{name}: the window has no running products"));
        assert_eq!(live.len(), decays, "{name}: the running products");
        assert!(held(&rmpv::Value::Array(live.clone())) > 0, "{name}: decay");
        let ring = at(snaps, &["ring"])
            .and_then(rmpv::Value::as_array)
            .unwrap();
        assert!(!ring.is_empty(), "{name}: the window holds no snapshot");
        for (i, item) in ring.iter().enumerate() {
            let products = item
                .as_array()
                .and_then(|t| t.get(3))
                .and_then(rmpv::Value::as_array)
                .unwrap_or_else(|| panic!("{name}: snapshot {i} keeps no products"));
            assert_eq!(products.len(), decays, "{name}: snapshot {i}'s products");
            assert!(
                held(item.as_array().unwrap().get(3).unwrap()) > 0,
                "{name}: snapshot {i}"
            );
        }
    }
    type Listed<'a> = (&'a str, &'a [&'a str], &'a [&'a str]);
    const LISTED: [Listed; 8] = [
        ("kmeans", &["model", "KMeans", "clusters"], &["c_lo"]),
        // Schema 59 (task 233, C-1): each column's smallest change and the
        // changes at it, continuous values and a few values both.
        (
            "audit",
            &["model", "Audit", "columns"],
            &["min_change_count"],
        ),
        (
            "audit_exact",
            &["model", "Audit", "columns"],
            &["min_change_count"],
        ),
        ("micro", &["model", "Micro", "mc"], &["s", "c_lo"]),
        (
            "audit",
            &["model", "Audit", "columns"],
            &["digest", "centroids"],
        ),
        (
            "audit",
            &["model", "Audit", "columns"],
            &["digest", "buffer"],
        ),
        ("audit", &["model", "Audit", "columns"], &["lag"]),
        (
            "audit_exact",
            &["model", "Audit", "columns"],
            &["digest", "buffer"],
        ),
    ];
    for (name, list, inner) in LISTED {
        let f = frozen::ALL
            .iter()
            .find(|f| f.name == name)
            .unwrap_or_else(|| panic!("{name}: no fixture; add the case and regenerate"));
        let v = rmpv::decode::read_value(&mut f.bytes().as_slice()).unwrap();
        let Some(items) = at(&v, list).and_then(rmpv::Value::as_array) else {
            panic!("{name}: the state has no list {}", list.join("."));
        };
        let mut lows = 0;
        for item in items {
            let Some(field) = at(item, inner) else {
                panic!(
                    "{name}: an element of {} has no {}",
                    list.join("."),
                    inner.join(".")
                );
            };
            lows += held(field);
        }
        assert!(
            lows > 0,
            "{name}: no {} in {} holds a number, so the fixture holds nothing of the layout",
            inner.join("."),
            list.join(".")
        );
    }
}

/// The refit's forms (schema 64, docs/PLAN.md task 277): `huber_refit` and
/// `sgd_huber_refit` are saved with every target's buffer part full -- rows
/// held, fewer learned than the configuration's count -- and their
/// continuations fill them, so the refit runs on a row the fixture holds;
/// `huber_refit_done` and `sgd_huber_refit_done` are saved after it, every
/// buffer done and empty. `quantile_refit` and `quantile_refit_done` are
/// the quantile refit's same two forms (task 278, D-Q2). The cases with the
/// refit off write no buffer.
#[test]
fn the_refit_is_written_part_full_and_done() {
    if regenerating() {
        return;
    }
    let buffers = |bytes: &[u8], variant: &str| -> (u64, Vec<(bool, u64, usize)>) {
        let v = rmpv::decode::read_value(&mut &bytes[..]).unwrap();
        let cap = at(&v, &["model", variant, "cfg", "refit_rows"])
            .and_then(rmpv::Value::as_u64)
            .unwrap_or_else(|| panic!("{variant}: cfg.refit_rows"));
        let list = at(&v, &["model", variant, "refit"])
            .and_then(rmpv::Value::as_array)
            .unwrap_or_else(|| panic!("{variant}: refit"))
            .iter()
            .map(|b| {
                let done = at(b, &["done"]).and_then(rmpv::Value::as_bool).unwrap();
                let learned = at(b, &["learned"]).and_then(rmpv::Value::as_u64).unwrap();
                (done, learned, held(at(b, &["rows"]).unwrap()))
            })
            .collect();
        (cap, list)
    };
    for (name, variant) in [
        ("huber_refit", "Robust"),
        ("sgd_huber_refit", "Sgd"),
        ("quantile_refit", "Robust"),
    ] {
        let f = frozen::ALL.iter().find(|f| f.name == name).unwrap();
        let (cap, list) = buffers(&f.bytes(), variant);
        assert_eq!(list.len(), 2, "{name}: a buffer a target");
        for (done, learned, rows) in &list {
            assert!(
                !done && *learned > 0 && *learned < cap && *rows > 0,
                "{name}: {list:?}"
            );
        }
        let mut m = load(&case_of(f), &f.bytes()).unwrap();
        for row in f.rows() {
            m.step(&row);
        }
        let (_, after) = buffers(&named(&m.state()), variant);
        assert!(
            after.iter().all(|(done, _, rows)| *done && *rows == 0),
            "{name}: the continuation does not reach the refit: {after:?}"
        );
    }
    for (name, variant) in [
        ("huber_refit_done", "Robust"),
        ("sgd_huber_refit_done", "Sgd"),
        ("quantile_refit_done", "Robust"),
    ] {
        let f = frozen::ALL.iter().find(|f| f.name == name).unwrap();
        let (cap, list) = buffers(&f.bytes(), variant);
        assert_eq!(list.len(), 2, "{name}: a buffer a target");
        for (done, learned, rows) in &list {
            assert!(*done && *learned == cap && *rows == 0, "{name}: {list:?}");
        }
    }
    for (name, variant) in [
        ("huber", "Robust"),
        ("sgd_huber", "Sgd"),
        ("quantile", "Robust"),
        ("quantile_mad", "Robust"),
    ] {
        let f = frozen::ALL.iter().find(|f| f.name == name).unwrap();
        assert_eq!(buffers(&f.bytes(), variant), (0, vec![]), "{name}");
    }
}

/// The bins' refit forms (schema 64, docs/PLAN.md task 278):
/// `marginal_bins_refit` is saved with its edges given and its rows held,
/// fewer than its count, and its continuation reaches the count, so the
/// refit runs on a row the fixture holds; `marginal_bins_refit_done` is
/// saved after its refit ran past a learned warm-up, its rows let go and
/// each target's passes kept. The cases with the refit off write a count of
/// 0, no passes and no held rows once the edges exist.
#[test]
fn the_bins_refit_is_written_holding_and_done() {
    if regenerating() {
        return;
    }
    let parts = |bytes: &[u8]| -> (u64, bool, usize, usize) {
        let v = rmpv::decode::read_value(&mut &bytes[..]).unwrap();
        let cap = at(&v, &["model", "Marginal", "cfg", "bins", "refit_rows"])
            .and_then(rmpv::Value::as_u64)
            .expect("cfg.bins.refit_rows");
        let refit = at(&v, &["model", "Marginal", "bins", "refit"]).expect("bins.refit");
        let done = at(refit, &["done"]).and_then(rmpv::Value::as_bool).unwrap();
        let passes = at(refit, &["passes"])
            .and_then(rmpv::Value::as_array)
            .map_or(0, |p| p.iter().filter(|x| x.as_u64() > Some(0)).count());
        let held = at(&v, &["model", "Marginal", "bins", "held"])
            .and_then(rmpv::Value::as_array)
            .map_or(0, Vec::len);
        (cap, done, passes, held)
    };
    let f = frozen::ALL
        .iter()
        .find(|f| f.name == "marginal_bins_refit")
        .unwrap();
    let (cap, done, passes, held) = parts(&f.bytes());
    assert!(
        cap == 30 && !done && passes == 0 && held > 0 && held < 30,
        "marginal_bins_refit: {:?}",
        (cap, done, passes, held)
    );
    let mut m = load(&case_of(f), &f.bytes()).unwrap();
    for row in f.rows() {
        m.step(&row);
    }
    let after = parts(&named(&m.state()));
    assert!(
        after.1 && after.2 == 2 && after.3 == 0,
        "marginal_bins_refit: the continuation does not reach the refit: {after:?}"
    );
    let f = frozen::ALL
        .iter()
        .find(|f| f.name == "marginal_bins_refit_done")
        .unwrap();
    assert_eq!(
        parts(&f.bytes()),
        (20, true, 2, 0),
        "marginal_bins_refit_done"
    );
    let off = frozen::ALL
        .iter()
        .find(|f| f.name == "marginal_bins_huber")
        .unwrap();
    assert_eq!(parts(&off.bytes()), (0, false, 0, 0), "marginal_bins_huber");
}

/// The gate's two forms and the quantile band's scale (schema 64,
/// docs/PLAN.md task 278, D10 and D-Q): every case a state written before
/// 64 converts to says `min_weight` held learning back (`true`, the field
/// such a state lacks), and the forms since -- the refit cases and
/// `quantile_mad` -- say it withholds the output alone (`false`), the
/// robust ones each holding the floor's count of its targets' first rows.
/// `quantile_mad` draws its band in the MAD, a sketch of `|e|` a target and
/// each holding residuals; `quantile` in the RMS, with none.
#[test]
fn the_gates_forms_and_the_quantile_scale_are_written() {
    if regenerating() {
        return;
    }
    let decode = |name: &str| {
        let f = frozen::ALL.iter().find(|f| f.name == name).unwrap();
        rmpv::decode::read_value(&mut &f.bytes()[..]).unwrap()
    };
    for (name, variant, holds) in [
        ("huber", "Robust", true),
        ("huber_mad", "Robust", true),
        ("quantile", "Robust", true),
        ("sgd_huber", "Sgd", true),
        ("sgd_huber_mad", "Sgd", true),
        ("huber_refit", "Robust", false),
        ("huber_refit_done", "Robust", false),
        ("huber_joining", "Robust", false),
        ("quantile_mad", "Robust", false),
        ("quantile_refit", "Robust", false),
        ("quantile_refit_done", "Robust", false),
        ("sgd_huber_refit", "Sgd", false),
        ("sgd_huber_refit_done", "Sgd", false),
        ("sgd_spread", "Sgd", false),
        ("sgd_unstable", "Sgd", false),
        ("sgd_spread_adagrad", "Sgd", false),
        ("sgd_resting", "Sgd", false),
    ] {
        let v = decode(name);
        let got = at(&v, &["model", variant, "cfg", "min_weight_holds_learning"])
            .and_then(rmpv::Value::as_bool);
        assert_eq!(got, Some(holds), "{name}: cfg.min_weight_holds_learning");
        // The robust floor's counts: one a target under the form since,
        // none under the old. Each is above 0 by the save, and at the floor
        // (a row per coefficient: two features and the intercept) but in
        // `huber_joining`, whose second target joined on the row before the
        // save and holds one row of the three: the form a count below the
        // floor takes (review round 10, S2-3).
        if variant == "Robust" {
            let counts: Vec<u64> = at(&v, &["model", variant, "seen"])
                .and_then(rmpv::Value::as_array)
                .map_or(Vec::new(), |a| {
                    a.iter()
                        .map(|c| c.as_u64().unwrap_or_else(|| panic!("{name}: {c}")))
                        .collect()
                });
            assert_eq!(counts.len(), if holds { 0 } else { 2 }, "{name}: seen");
            let want: &[u64] = match (holds, name) {
                (true, _) => &[],
                (false, "huber_joining") => &[3, 1],
                (false, _) => &[3, 3],
            };
            assert_eq!(counts, want, "{name}: seen");
        }
    }
    for (name, scale, sketches) in [("quantile", "rms", 0), ("quantile_mad", "mad", 2)] {
        let v = decode(name);
        let got = at(&v, &["model", "Robust", "cfg", "huber_scale"]).and_then(rmpv::Value::as_str);
        assert_eq!(got, Some(scale), "{name}: cfg.huber_scale");
        let mad = at(&v, &["model", "Robust", "mad"])
            .and_then(rmpv::Value::as_array)
            .unwrap_or_else(|| panic!("{name}: mad"));
        assert_eq!(mad.len(), sketches, "{name}: a sketch a target");
        assert!(
            mad.iter().all(|q| held(q) > 0),
            "{name}: each sketch holds residuals"
        );
    }
}

/// `sgd`'s target-spread forms (schema 64, docs/PLAN.md task 278: D1, D2,
/// D8, D-S): `sgd_spread` keeps each target's spread under the squared loss,
/// its step gains (with their rows' weight, task 282 M1) and its level,
/// every one holding a number; `sgd_unstable`
/// is saved with its gate withholding, so its continuation reads the gains
/// from the file and withholds every row; `sgd_resting` is saved with its
/// gate resting (task 314); `sgd_spread_adagrad` keeps the
/// spread and the level and no gains, AdaGrad's step being bounded. The
/// cases without the flag write none of them.
#[test]
fn the_target_spread_forms_are_written() {
    if regenerating() {
        return;
    }
    let decoded = |name: &str| -> rmpv::Value {
        let f = frozen::ALL.iter().find(|f| f.name == name).unwrap();
        rmpv::decode::read_value(&mut f.bytes().as_slice()).unwrap()
    };
    let field = |name: &str, key: &str| -> rmpv::Value {
        at(&decoded(name), &["model", "Sgd", key])
            .cloned()
            .unwrap_or_else(|| panic!("{name}: no {key}"))
    };
    let len = |v: &rmpv::Value| v.as_array().map_or(0, Vec::len);
    for name in [
        "sgd_spread",
        "sgd_unstable",
        "sgd_spread_adagrad",
        "sgd_resting",
    ] {
        let on = at(&decoded(name), &["model", "Sgd", "cfg", "target_spread"])
            .and_then(rmpv::Value::as_bool);
        assert_eq!(on, Some(true), "{name}: cfg.target_spread");
        for key in ["spread", "level"] {
            assert_eq!(len(&field(name, key)), 2, "{name}: {key}, one a target");
            assert!(held(&field(name, key)) > 0, "{name}: {key} holds nothing");
        }
    }
    // The gate's two forms (task 314): awake in `sgd_spread` and
    // `sgd_unstable`, resting in `sgd_resting`, each target's.
    for (name, rests) in [
        ("sgd_spread", false),
        ("sgd_unstable", false),
        ("sgd_resting", true),
    ] {
        for g in field(name, "gain").as_array().unwrap() {
            let resting = at(g, &["resting"]).and_then(rmpv::Value::as_bool);
            assert_eq!(resting, Some(rests), "{name}: gain {g}");
            // The gate's count, the second new field, written non-zero
            // where the gate rests: the rows let pass since it rested
            // (task 316, review 11's S2-4).
            if rests {
                let rested = at(g, &["rested"]).and_then(rmpv::Value::as_u64);
                assert!(rested.is_some_and(|r| r > 0), "{name}: gain {g}");
            }
        }
    }
    for name in ["sgd_spread", "sgd_unstable", "sgd_resting"] {
        assert_eq!(len(&field(name, "gain")), 2, "{name}: gains, one a target");
        assert!(
            held(&field(name, "gain")) > 0,
            "{name}: the gains hold nothing"
        );
        // Each target's gains carry their rows' weight, `n`, beside the two
        // sums: the running mean each gain is bounded by.
        for g in field(name, "gain").as_array().unwrap() {
            let n = at(g, &["n"]).and_then(rmpv::Value::as_f64);
            assert!(n.is_some_and(|n| n > 1.0), "{name}: gain {g}");
        }
    }
    assert_eq!(len(&field("sgd_spread_adagrad", "gain")), 0);
    for name in ["sgd", "sgd_huber", "sgd_logistic"] {
        for key in ["gain", "level"] {
            assert_eq!(len(&field(name, key)), 0, "{name}: {key} without the flag");
        }
    }
    // The gate: the stable case predicts after its save, the unstable one
    // withholds every row, read from the gains in the file.
    for (name, withheld) in [
        ("sgd_spread", false),
        ("sgd_unstable", true),
        ("sgd_resting", false),
    ] {
        let f = frozen::ALL.iter().find(|f| f.name == name).unwrap();
        let mut m = load(&case_of(f), &f.bytes()).unwrap();
        let preds: Vec<f64> = f.rows().iter().flat_map(|r| m.step(r).pred).collect();
        assert_eq!(
            preds.iter().all(|p| p.is_nan()),
            withheld,
            "{name}: {preds:?}"
        );
    }
}

/// `min_weight`'s two forms in `kalman` and `lasso` (schema 64,
/// docs/PLAN.md task 278, D10): every case a state written before 64
/// converts to says it held learning back (`true`, the field such a state
/// lacks), and `kalman_output_only` and `lasso_floor` say it withholds the
/// output alone (`false`). Each of those two is saved where the gate
/// withholds the continuation's first rows and lets out the rest, and the
/// rows it withholds are learned from: flipped to the old form, the same
/// state goes on to other numbers (the explicit `q`, read over the residual
/// variance, carries `kalman`'s into its predictions; `lasso`'s selection
/// reports its own).
#[test]
fn the_gates_two_forms_are_written_and_act() {
    if regenerating() {
        return;
    }
    let decode = |name: &str| {
        let f = frozen::ALL.iter().find(|f| f.name == name).unwrap();
        (f, rmpv::decode::read_value(&mut &f.bytes()[..]).unwrap())
    };
    for (name, variant, holds) in [
        ("kalman", "Kalman", true),
        ("kalman_gap", "Kalman", true),
        ("kalman_noise_units", "Kalman", true),
        ("kalman_output_only", "Kalman", false),
        ("lasso", "Lasso", true),
        ("lasso_window", "Lasso", true),
        ("lasso_floor", "Lasso", false),
    ] {
        let (f, mut v) = decode(name);
        let path = ["model", variant, "cfg", "min_weight_holds_learning"];
        let got = at(&v, &path).and_then(rmpv::Value::as_bool);
        assert_eq!(got, Some(holds), "{name}: cfg.min_weight_holds_learning");
        if holds {
            continue;
        }
        let outs = f.outs();
        let withheld = outs
            .iter()
            .filter(|o| o.pred.iter().all(|p| p.is_nan()))
            .count();
        let shown = outs
            .iter()
            .filter(|o| o.pred.iter().all(|p| p.is_finite()))
            .count();
        assert!(
            withheld >= 1 && shown >= 1,
            "{name}: the continuation withholds {withheld} rows and shows {shown}"
        );
        // The same state in the old form.
        let field = path.iter().try_fold(&mut v, |v, key| match v {
            rmpv::Value::Map(m) => m
                .iter_mut()
                .find(|(k, _)| k.as_str() == Some(key))
                .map(|(_, v)| v),
            _ => None,
        });
        *field.unwrap() = rmpv::Value::from(true);
        let mut bytes = Vec::new();
        rmpv::encode::write_value(&mut bytes, &v).unwrap();
        let mut old = load(&case_of(f), &bytes).unwrap();
        let moved = f
            .rows()
            .iter()
            .zip(&outs)
            .filter(|(row, want)| {
                let got = old.step(row);
                let bits = |o: &Out| {
                    o.pred
                        .iter()
                        .chain(&o.extra)
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>()
                };
                bits(&got) != bits(want)
            })
            .count();
        assert!(moved > 0, "{name}: the gate's form moves nothing");
    }
}

/// `ewridge`'s `min_weight` withholding the output alone (schema 64,
/// docs/PLAN.md task 278): `ewridge_output_only` writes the mark `true` (as
/// `ewridge_first_period`, the form a spec builds, does), and every other
/// `ewridge` case `false`, the form a state saved before it,
/// which has no mark, loads as: the residual variance and the first solve
/// held back to `min_weight`. The fixture is saved where the gate withholds the
/// continuation's first rows and lets out the rest; its variance folds the
/// withheld rows at the model's own floor, so the same state with the mark
/// taken out goes on to another variance, and the same predictions (no
/// output reads it; the first solve is long past).
#[test]
fn ewridges_output_only_form_is_written_and_acts() {
    if regenerating() {
        return;
    }
    let mark = ["model", "EwRidge", "min_weight_output_only"];
    for f in frozen::ALL.iter().filter(|f| f.variant == "EwRidge") {
        let v = rmpv::decode::read_value(&mut f.bytes().as_slice()).unwrap();
        let got = at(&v, &mark).and_then(rmpv::Value::as_bool);
        // The first period's and the never cadence's cases are the form a
        // spec builds, with this mark.
        let want = Some(matches!(
            f.name,
            "ewridge_output_only" | "ewridge_first_period" | "ewridge_never"
        ));
        assert_eq!(got, want, "{}: min_weight_output_only", f.name);
    }
    let f = frozen::ALL
        .iter()
        .find(|f| f.name == "ewridge_output_only")
        .expect("ewridge_output_only: no fixture; add the case and regenerate");
    let outs = f.outs();
    let withheld = outs
        .iter()
        .filter(|o| o.pred.iter().all(|p| p.is_nan()))
        .count();
    let shown = outs
        .iter()
        .filter(|o| o.pred.iter().all(|p| p.is_finite()))
        .count();
    assert!(
        withheld >= 1 && shown >= 1,
        "the continuation withholds {withheld} rows and shows {shown}"
    );
    let mut v = rmpv::decode::read_value(&mut f.bytes().as_slice()).unwrap();
    if let Some(rmpv::Value::Map(m)) = at(&v, &["model", "EwRidge"]).cloned().as_mut() {
        m.retain(|(k, _)| k.as_str() != Some("min_weight_output_only"));
        let model = match &mut v {
            rmpv::Value::Map(top) => top
                .iter_mut()
                .find(|(k, _)| k.as_str() == Some("model"))
                .map(|(_, v)| v)
                .unwrap(),
            _ => unreachable!("a state is a map"),
        };
        *model = rmpv::Value::Map(vec![("EwRidge".into(), rmpv::Value::Map(m.clone()))]);
    } else {
        panic!("an EwRidge model");
    }
    let mut bytes = Vec::new();
    rmpv::encode::write_value(&mut bytes, &v).unwrap();
    let case = case_of(f);
    let (mut new, mut old) = (
        load(&case, &f.bytes()).unwrap(),
        load(&case, &bytes).unwrap(),
    );
    for (row, want) in f.rows().iter().zip(&outs) {
        let (a, b) = (new.step(row), old.step(row));
        let bits = |o: &Out| o.pred.iter().map(|p| p.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&a), bits(want), "the frozen continuation");
        assert_eq!(bits(&a), bits(&b), "the mark moved a prediction");
    }
    let sig2 = |m: &dyn cases::Run| {
        let s = named(&m.state());
        let v = rmpv::decode::read_value(&mut s.as_slice()).unwrap();
        format!("{}", at(&v, &["model", "EwRidge", "sig2"]).expect("sig2"))
    };
    let (a, b) = (sig2(new.as_ref()), sig2(old.as_ref()));
    println!("sig2 with the mark {a}, without it {b}");
    assert_ne!(a, b, "the mark moved no residual variance");
}

/// An explicit cadence's first period on the default weight-share schedule
/// (schema 64, task 278, "Default cadence first"): each `*_first_period`
/// case writes the mark `true`, and every other case of its variant
/// `false`, the form a state saved before it, which has no mark, loads as.
/// Each is saved inside the period, which its continuation's row 42 ends:
/// the period's solves move its predictions there, so the same state with
/// the mark taken out goes on to other numbers, and the state it saves at
/// the end has the mark cleared. `ewridge_never`, a cadence that never
/// comes due (`inf`), is the spec's form that takes no mark.
#[test]
fn the_first_periods_form_is_written_and_acts() {
    if regenerating() {
        return;
    }
    let forms = [
        ("ewridge_first_period", "EwRidge"),
        ("lasso_first_period", "Lasso"),
        ("huber_first_period", "Robust"),
    ];
    for (_, variant) in forms {
        let mark = ["model", variant, "first_period_by_share"];
        for f in frozen::ALL.iter().filter(|f| f.variant == variant) {
            let v = rmpv::decode::read_value(&mut f.bytes().as_slice()).unwrap();
            let got = at(&v, &mark).and_then(rmpv::Value::as_bool);
            let want = Some(forms.iter().any(|(name, _)| *name == f.name));
            assert_eq!(got, want, "{}: first_period_by_share", f.name);
        }
    }
    // A cadence that never comes due takes no mark, though the spec sets
    // it: `ewridge_never`, at `solve_every = inf`, written as the double.
    let never = frozen::ALL
        .iter()
        .find(|f| f.name == "ewridge_never")
        .expect("ewridge_never: no fixture; add the case and regenerate");
    let v = rmpv::decode::read_value(&mut never.bytes().as_slice()).unwrap();
    let every = at(&v, &["model", "EwRidge", "cfg", "solve_every"]).and_then(rmpv::Value::as_f64);
    assert_eq!(every, Some(f64::INFINITY), "ewridge_never: cfg.solve_every");
    let mark = at(&v, &["model", "EwRidge", "first_period_by_share"]);
    assert_eq!(mark.and_then(rmpv::Value::as_bool), Some(false));
    for (name, variant) in forms {
        let f = frozen::ALL
            .iter()
            .find(|f| f.name == name)
            .unwrap_or_else(|| panic!("{name}: no fixture; add the case and regenerate"));
        let mut v = rmpv::decode::read_value(&mut f.bytes().as_slice()).unwrap();
        let field =
            ["model", variant, "first_period_by_share"]
                .iter()
                .try_fold(&mut v, |v, key| match v {
                    rmpv::Value::Map(m) => m
                        .iter_mut()
                        .find(|(k, _)| k.as_str() == Some(key))
                        .map(|(_, v)| v),
                    _ => None,
                });
        *field.unwrap() = rmpv::Value::from(false);
        let mut bytes = Vec::new();
        rmpv::encode::write_value(&mut bytes, &v).unwrap();
        let case = case_of(f);
        let (mut new, mut old) = (
            load(&case, &f.bytes()).unwrap(),
            load(&case, &bytes).unwrap(),
        );
        let bits = |o: &Out| o.pred.iter().map(|p| p.to_bits()).collect::<Vec<_>>();
        let mut moved = 0;
        for (row, want) in f.rows().iter().zip(&f.outs()) {
            let (a, b) = (new.step(row), old.step(row));
            assert_eq!(bits(&a), bits(want), "{name}: the frozen continuation");
            moved += usize::from(bits(&a) != bits(&b));
        }
        println!("{name}: the period's solves moved {moved} predictions");
        assert!(moved > 0, "{name}: the mark moved no prediction");
        let end = rmpv::decode::read_value(&mut named(&new.state()).as_slice()).unwrap();
        let mark = at(&end, &["model", variant, "first_period_by_share"]);
        assert_eq!(
            mark.and_then(rmpv::Value::as_bool),
            Some(false),
            "{name}: the period ended on row 42 and cleared the mark"
        );
    }
}

/// `kalman`'s covariance in units of the noise (schema 64, docs/PLAN.md task
/// 278): `kalman_noise_units` names the form in its cfg and is saved with
/// every covariance's intercept still diffuse -- its row and column 0 --
/// after rows that moved the moments and the clock since an observation,
/// and its continuation takes the diffuse step. `kalman` and `kalman_gap`
/// are the form before, which writes no `noise_units`, so a state saved
/// before it loads as that form.
#[test]
fn kalmans_noise_units_form_is_written_with_its_intercept_open() {
    if regenerating() {
        return;
    }
    let intercepts = |v: &rmpv::Value| -> Vec<f64> {
        at(v, &["model", "Kalman", "p"])
            .and_then(rmpv::Value::as_array)
            .expect("p")
            .iter()
            .map(|p| p.as_array().unwrap()[0].as_f64().unwrap())
            .collect()
    };
    let f = frozen::ALL
        .iter()
        .find(|f| f.name == "kalman_noise_units")
        .expect("kalman_noise_units: no fixture; add the case and regenerate");
    let v = rmpv::decode::read_value(&mut f.bytes().as_slice()).unwrap();
    let units = at(&v, &["model", "Kalman", "cfg", "noise_units"]).and_then(rmpv::Value::as_bool);
    assert_eq!(units, Some(true), "cfg.noise_units");
    let open = intercepts(&v);
    assert!(
        open.len() == 2 && open.iter().all(|p| *p == 0.0),
        "{open:?}"
    );
    let elapsed = at(&v, &["model", "Kalman", "elapsed"]).expect("elapsed");
    assert!(held(elapsed) > 0, "the clock since an observation moved");
    let mut m = load(&case_of(f), &f.bytes()).unwrap();
    for row in f.rows() {
        m.step(&row);
    }
    let after = rmpv::decode::read_value(&mut named(&m.state()).as_slice()).unwrap();
    let closed = intercepts(&after);
    assert!(
        closed.iter().all(|p| *p > 0.0),
        "the continuation takes the step: {closed:?}"
    );
    for name in ["kalman", "kalman_gap"] {
        let f = frozen::ALL.iter().find(|f| f.name == name).unwrap();
        let v = rmpv::decode::read_value(&mut f.bytes().as_slice()).unwrap();
        assert!(
            at(&v, &["model", "Kalman", "cfg", "noise_units"]).is_none(),
            "{name}"
        );
    }
}

/// The forms of `kalman`'s fixed-lag smoother (schema 72, before any
/// release), read from a state by case name: `kalman_smooth` holds slots
/// with a gain and `K`'s triangle and a pending pair; `kalman_smooth_shared`
/// one ring of both targets' means and no `K`; `kalman_smooth_early` a
/// ring short of its lag with a slope not yet sized and a row `min_weight`
/// held; `kalman_smooth_break` a slot whose gain is empty, the chain cut by
/// a refused re-map.
fn smoother_forms(state_of: &dyn Fn(&str) -> rmpv::Value) {
    let ring = |name: &str| -> rmpv::Value {
        let v = state_of(name);
        let rings = at(&v, &["model", "Kalman", "smoother", "rings"])
            .and_then(rmpv::Value::as_array)
            .unwrap_or_else(|| panic!("{name}: no smoother"))
            .clone();
        assert!(!rings.is_empty(), "{name}");
        rings[0].clone()
    };
    let slots = |r: &rmpv::Value| -> Vec<rmpv::Value> {
        at(r, &["slots"])
            .and_then(rmpv::Value::as_array)
            .cloned()
            .unwrap_or_default()
    };
    let len = |v: &rmpv::Value, key: &str| {
        at(v, &[key])
            .and_then(rmpv::Value::as_array)
            .map_or(0, Vec::len)
    };
    let full = ring("kalman_smooth");
    let s = slots(&full);
    assert_eq!(s.len(), 7, "a full ring holds lag - 1 slots");
    assert!(s.iter().all(|x| len(x, "g") == 9 && len(x, "kk") == 6));
    assert!(at(&full, &["pending"]).is_some_and(|p| !p.is_nil()));
    let shared = ring("kalman_smooth_shared");
    let s = slots(&shared);
    assert!(!s.is_empty() && s.iter().all(|x| len(x, "c") == 6 && len(x, "kk") == 0));
    let early = ring("kalman_smooth_early");
    let described: Vec<rmpv::Value> = slots(&early)
        .iter()
        .chain(at(&early, &["pending"]))
        .filter_map(|x| at(x, &["of"]).cloned())
        .collect();
    let any_of = |key: &str, hit: &dyn Fn(&rmpv::Value) -> bool| {
        described.iter().any(|d| {
            at(d, &[key])
                .and_then(rmpv::Value::as_array)
                .is_some_and(|v| v.iter().any(hit))
        })
    };
    assert!(
        any_of("sized", &|b| b.as_bool() == Some(false)),
        "an unsized slot"
    );
    assert!(any_of("held", &|b| b.as_u64() != Some(0)), "a held row");
    let cut = ring("kalman_smooth_break");
    assert!(
        slots(&cut)
            .iter()
            .any(|x| len(x, "g") == 0 && len(x, "kk") == 6)
    );
}

/// The smoother's cases hold each of its forms when saved, before any
/// file is written: what the frozen files will freeze.
#[test]
fn kalmans_smoother_cases_hold_each_form() {
    smoother_forms(&|name| {
        let case = all().into_iter().find(|c| c.name == name).expect("a case");
        let (m, _) = saved(&case);
        rmpv::decode::read_value(&mut named(&m.state()).as_slice()).unwrap()
    });
}

/// The frozen files write every form of the smoother, and every other
/// `kalman` case writes no `smoother`.
#[test]
fn kalmans_smoother_is_written_in_each_form() {
    if regenerating() {
        return;
    }
    smoother_forms(&|name| {
        let f = frozen::ALL
            .iter()
            .find(|f| f.name == name)
            .unwrap_or_else(|| panic!("{name}: no fixture; add the case and regenerate"));
        rmpv::decode::read_value(&mut f.bytes().as_slice()).unwrap()
    });
    for f in frozen::ALL {
        if f.variant == "Kalman" && !f.name.starts_with("kalman_smooth") {
            let v = rmpv::decode::read_value(&mut f.bytes().as_slice()).unwrap();
            assert!(
                at(&v, &["model", "Kalman", "smoother"]).is_none(),
                "{}",
                f.name
            );
        }
    }
}

/// `kalman`'s square root (schema 68, docs/PLAN.md task 281 (3)):
/// `kalman_square_root` names its form, `covariance_form = "square_root"`,
/// and holds each covariance as the factor `L = Cᵀ`, lower triangular with
/// its upper triangle 0 to the bit, a positive diagonal and an entry below
/// it that is not 0, so the factor's every write has acted; its
/// continuation keeps it so. Every other `kalman` case, the covariance
/// form, writes no `covariance_form`, which a state saved before 68 lacks.
#[test]
fn kalmans_square_root_is_written_as_its_factor() {
    if regenerating() {
        return;
    }
    let k = 3;
    let factors = |v: &rmpv::Value| -> Vec<Vec<f64>> {
        at(v, &["model", "Kalman", "p"])
            .and_then(rmpv::Value::as_array)
            .expect("p")
            .iter()
            .map(|p| {
                p.as_array()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_f64().unwrap())
                    .collect()
            })
            .collect()
    };
    let lower = |l: &[f64]| {
        l.len() == k * k
            && (0..k * k).all(|ij| ij / k >= ij % k || l[ij].to_bits() == 0)
            && (0..k).all(|i| l[i * k + i] > 0.0)
            && (0..k * k).any(|ij| ij / k > ij % k && l[ij] != 0.0)
    };
    let f = frozen::ALL
        .iter()
        .find(|f| f.name == "kalman_square_root")
        .expect("kalman_square_root: no fixture; add the case and regenerate");
    let v = rmpv::decode::read_value(&mut f.bytes().as_slice()).unwrap();
    let form = at(&v, &["model", "Kalman", "covariance_form"]).and_then(rmpv::Value::as_str);
    assert_eq!(form, Some("square_root"), "covariance_form");
    let held = factors(&v);
    assert!(held.len() == 2 && held.iter().all(|l| lower(l)), "{held:?}");
    let mut m = load(&case_of(f), &f.bytes()).unwrap();
    for row in f.rows() {
        m.step(&row);
    }
    let after = rmpv::decode::read_value(&mut named(&m.state()).as_slice()).unwrap();
    assert!(factors(&after).iter().all(|l| lower(l)), "the continuation");
    // The others write no form, or, where the moments' gap is written
    // after it (review 2026-11-23, KAL-6), the default one.
    for f in frozen::ALL.iter().filter(|f| f.variant == "Kalman") {
        if f.name == "kalman_square_root" {
            continue;
        }
        let v = rmpv::decode::read_value(&mut f.bytes().as_slice()).unwrap();
        let form = at(&v, &["model", "Kalman", "covariance_form"]).and_then(rmpv::Value::as_str);
        let gap = at(&v, &["model", "Kalman", "moments_gap"]);
        assert!(
            form.is_none() || (form == Some("covariance") && gap.is_some()),
            "{}",
            f.name
        );
        // The moments' gap (schema 72, rule 5's thirty-fifth exception):
        // written in the case saved after rows with no target, the clock
        // since the last row that moved the moments, 0; in no other.
        let want = (f.name == "kalman_noise_units").then_some(0.0);
        assert_eq!(gap.and_then(rmpv::Value::as_f64), want, "{}", f.name);
    }
}

/// `hmm`'s factor form (schema 69, task 479): `hmm_full`, 13 features
/// under `full`, holds each state as its weight, means, ridge scale, `σ` in
/// `[1/4, 1]` and `L`, the lower triangle of a Cholesky factor of order 13
/// (91 numbers, every pivot positive, an entry below one not 0), and no
/// co-moments; the factor has moved since its take-up, so some `σ` is
/// below 1, and its continuation keeps the form. `hmm_full_narrow`, 12
/// features, and every other `hmm` case write `factor_states` empty (since
/// schema 70; a state saved before 69 lacks it). Under the prior the
/// factor form's `f` has moved off its target, `P₀ψ`, by its decay and the
/// moves since its take-up; the `tied` case `hmm` writes no prior.
#[test]
fn a_wide_full_hmm_is_written_as_its_factors() {
    if regenerating() {
        return;
    }
    let d = 13;
    let forms = |v: &rmpv::Value| -> Vec<(f64, Vec<f64>)> {
        at(v, &["model", "Hmm", "factor_states"])
            .and_then(rmpv::Value::as_array)
            .expect("factor_states")
            .iter()
            .map(|f| {
                let sigma = at(f, &["sigma"]).and_then(rmpv::Value::as_f64).unwrap();
                let l = at(f, &["l"])
                    .and_then(rmpv::Value::as_array)
                    .unwrap()
                    .iter()
                    .map(|x| x.as_f64().unwrap())
                    .collect();
                (sigma, l)
            })
            .collect()
    };
    let factor = |(sigma, l): &(f64, Vec<f64>)| {
        let col = |j: usize| j * d - j * j.saturating_sub(1) / 2;
        (0.25..=1.0).contains(sigma)
            && l.len() == d * (d + 1) / 2
            && (0..d).all(|j| l[col(j)] > 0.0)
            && l.iter().all(|x| x.is_finite())
            && (0..d).any(|j| (j + 1..d).any(|i| l[col(j) + i - j] != 0.0))
    };
    let f = frozen::ALL
        .iter()
        .find(|f| f.name == "hmm_full")
        .expect("hmm_full: no fixture; add the case and regenerate");
    let v = rmpv::decode::read_value(&mut f.bytes().as_slice()).unwrap();
    let held = forms(&v);
    assert!(held.len() == 2 && held.iter().all(factor), "{held:?}");
    assert!(
        held.iter().any(|(sigma, _)| *sigma < 1.0),
        "no factor has moved"
    );
    let states = at(&v, &["model", "Hmm", "states"]).and_then(rmpv::Value::as_array);
    assert_eq!(
        states.map(Vec::len),
        Some(0),
        "co-moments beside the factors"
    );
    let mut m = load(&case_of(f), &f.bytes()).unwrap();
    for row in f.rows() {
        m.step(&row);
    }
    let after = rmpv::decode::read_value(&mut named(&m.state()).as_slice()).unwrap();
    assert!(forms(&after).iter().all(factor), "the continuation");
    // The factored prior has moved off its target since the take-up: the
    // prior's moves and decay are in the state.
    let list = |path: &[&str]| -> Vec<f64> {
        at(&v, path)
            .and_then(rmpv::Value::as_array)
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap())
            .collect()
    };
    let scale = list(&["model", "Hmm", "stabilized", "scale"]);
    let prior = list(&["model", "Hmm", "stabilized", "factored_prior"]);
    assert_eq!((scale.len(), prior.len()), (d, d));
    let p0 = d as f64;
    assert!(
        scale
            .iter()
            .zip(&prior)
            .any(|(s, f)| (f - p0 * s).abs() > 1e-9 * p0 * s),
        "f at its target: {prior:?}"
    );
    for f in frozen::ALL
        .iter()
        .filter(|f| f.variant == "Hmm" && f.name != "hmm_full")
    {
        let v = rmpv::decode::read_value(&mut f.bytes().as_slice()).unwrap();
        assert_eq!(
            at(&v, &["model", "Hmm", "factor_states"])
                .and_then(rmpv::Value::as_array)
                .map(Vec::len),
            Some(0),
            "{}",
            f.name
        );
        assert_eq!(
            at(&v, &["model", "Hmm", "stabilized"]).is_some(),
            f.name != "hmm",
            "{}: the prior",
            f.name
        );
    }
}

/// `audit`'s counters in both of their forms (schema 54, docs/PLAN.md task
/// 223 (b)): a summary past the cap, with decrements, and exact counts below
/// it. Integers, which [`held`] does not count, so read here.
#[test]
fn an_audits_counters_are_written_in_both_forms() {
    if regenerating() {
        return;
    }
    for (name, over) in [("audit", true), ("audit_exact", false)] {
        let f = frozen::ALL
            .iter()
            .find(|f| f.name == name)
            .unwrap_or_else(|| panic!("{name}: no fixture; add the case and regenerate"));
        let v = rmpv::decode::read_value(&mut f.bytes().as_slice()).unwrap();
        let columns = at(&v, &["model", "Audit", "columns"])
            .and_then(rmpv::Value::as_array)
            .unwrap_or_else(|| panic!("{name}: the state has no columns"));
        for c in columns {
            let counts = at(c, &["counts"]).unwrap();
            assert_eq!(
                at(counts, &["over"]).and_then(rmpv::Value::as_bool),
                Some(over),
                "{name}"
            );
            let held_counts = at(counts, &["counts"])
                .and_then(rmpv::Value::as_array)
                .map_or(0, Vec::len);
            assert!(held_counts > 0, "{name}: no counter is held");
            let decrements = at(counts, &["decrements"]).and_then(rmpv::Value::as_u64);
            assert_eq!(decrements.is_some_and(|d| d > 0), over, "{name}");
        }
    }
    // And its moments over a power of two above 1, for values past 2^65.
    let f = frozen::ALL
        .iter()
        .find(|f| f.name == "audit_scaled")
        .expect("audit_scaled: no fixture; add the case and regenerate");
    let v = rmpv::decode::read_value(&mut f.bytes().as_slice()).unwrap();
    let columns = at(&v, &["model", "Audit", "columns"])
        .and_then(rmpv::Value::as_array)
        .expect("audit_scaled: the state has no columns");
    for c in columns {
        let scale = at(c, &["moments", "scale"]).and_then(rmpv::Value::as_f64);
        assert!(
            scale.is_some_and(|s| s > 1.0),
            "audit_scaled: scale {scale:?}"
        );
    }
}

/// `ew_cov`'s `bias` in both of its values (schema 61, docs/PLAN.md task
/// 244): `false`, the default, in `ew_cov`, whose statistics and components
/// the factor scales, and `true` in `ew_cov_window`. A bool, which [`held`]
/// does not count, so read here.
#[test]
fn ew_covs_bias_is_written_in_both_values() {
    if regenerating() {
        return;
    }
    for (name, bias) in [("ew_cov", false), ("ew_cov_window", true)] {
        let f = frozen::ALL
            .iter()
            .find(|f| f.name == name)
            .unwrap_or_else(|| panic!("{name}: no fixture; add the case and regenerate"));
        let v = rmpv::decode::read_value(&mut f.bytes().as_slice()).unwrap();
        let got = at(&v, &["model", "EwCovModel", "cfg", "bias"]).and_then(rmpv::Value::as_bool);
        assert_eq!(got, Some(bias), "{name}");
    }
}

// --- regeneration -----------------------------------------------------------

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/state_fixtures")
}

/// `bits` as a Rust array of hex literals, eight a line.
fn words(out: &mut String, name: &str, bits: &[u64]) {
    writeln!(out, "    {name}: &[").unwrap();
    for chunk in bits.chunks(4) {
        let line: Vec<String> = chunk.iter().map(|b| format!("{b:#018x}")).collect();
        writeln!(out, "        {},", line.join(", ")).unwrap();
    }
    writeln!(out, "    ],").unwrap();
}

fn file_of(case: &Case) -> String {
    let (mut m, rows) = saved(case);
    let bytes = named(&m.state());
    let after = &rows[case.before..];
    let outs: Vec<Out> = after.iter().map(|r| m.step(r)).collect();
    // The state saved before the continuation, loaded, goes on as the model
    // that never stopped, and saves its bytes again: a fixture that fails
    // its own checks is not written.
    let mut back = load(case, &bytes).unwrap_or_else(|e| panic!("{}: {e}", case.name));
    assert!(
        named(&back.state()) == bytes,
        "{}: re-saves other bytes",
        case.name
    );
    for (r, want) in after.iter().zip(&outs) {
        same_out(case.name, &back.step(r), want, 0.0);
    }
    let n_pred = outs[0].pred.len();
    let n_extra = outs[0].extra.len();
    let n_targets = after[0].y.len();
    let s: State = rmp_serde::from_slice(&bytes).unwrap();
    let mut out = String::new();
    writeln!(
        out,
        "// @generated by `{REGENERATE}`: do not edit by hand.\n\
         // The `{}` case of tests/cases/mod.rs after {} rows, and its next {} rows.\n\
         pub const FIXTURE: crate::cases::Fixture = crate::cases::Fixture {{",
        case.name,
        case.before,
        after.len()
    )
    .unwrap();
    writeln!(out, "    name: {:?},", case.name).unwrap();
    writeln!(out, "    variant: {:?},", case.variant).unwrap();
    writeln!(out, "    schema: {},", s.schema_version).unwrap();
    writeln!(out, "    writer: {:?},", writer()).unwrap();
    writeln!(out, "    state: \"\\").unwrap();
    let h = hex(&bytes);
    for (i, chunk) in h.as_bytes().chunks(96).enumerate() {
        let tail = if (i + 1) * 96 >= h.len() { "\"," } else { "\\" };
        writeln!(out, "        {}{tail}", std::str::from_utf8(chunk).unwrap()).unwrap();
    }
    let x: Vec<u64> = after
        .iter()
        .flat_map(|r| r.x.iter().map(|v| v.to_bits()))
        .collect();
    words(&mut out, "x", &x);
    writeln!(out, "    y: &[").unwrap();
    for r in after.iter().filter(|r| !r.y.is_empty()) {
        let ys: Vec<String> =
            r.y.iter()
                .map(|v| match v {
                    Some(v) => format!("Some({:#018x})", v.to_bits()),
                    None => "None".into(),
                })
                .collect();
        writeln!(out, "        {},", ys.join(", ")).unwrap();
    }
    writeln!(out, "    ],").unwrap();
    let d: Vec<u64> = after.iter().map(|r| r.d.to_bits()).collect();
    let w: Vec<u64> = after.iter().map(|r| r.w.to_bits()).collect();
    words(&mut out, "d", &d);
    words(&mut out, "w", &w);
    let pred: Vec<u64> = outs
        .iter()
        .flat_map(|o| o.pred.iter().map(|v| v.to_bits()))
        .collect();
    let n_eff: Vec<u64> = outs.iter().map(|o| o.n_eff.to_bits()).collect();
    let extra: Vec<u64> = outs
        .iter()
        .flat_map(|o| o.extra.iter().map(|v| v.to_bits()))
        .collect();
    words(&mut out, "pred", &pred);
    words(&mut out, "n_eff", &n_eff);
    words(&mut out, "extra", &extra);
    writeln!(
        out,
        "    n_features: {},\n    n_targets: {n_targets},\n    n_pred: {n_pred},\n    n_extra: {n_extra},\n}};",
        case.n_features
    )
    .unwrap();
    out
}

/// Rewrites every fixture when `PRINT_STATE_FIXTURES=1`.
#[test]
fn regenerate_when_asked() {
    if !regenerating() {
        return;
    }
    let dir = dir();
    std::fs::create_dir_all(&dir).unwrap();
    for entry in std::fs::read_dir(&dir).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_some_and(|e| e == "rs") {
            std::fs::remove_file(path).unwrap();
        }
    }
    // No index: `build.rs` discovers the files on the next build.
    for case in &all() {
        let text = file_of(case);
        std::fs::write(dir.join(format!("{}.rs", case.name)), &text).unwrap();
        println!("wrote {}.rs ({} bytes)", case.name, text.len());
    }
    // A run that rewrote the files tested nothing: the other tests return
    // early under the variable. It fails, so it never stands for a test run
    // (review S20: a broken build regenerated its fixtures and passed).
    panic!(
        "PRINT_STATE_FIXTURES=1 rewrote the current fixtures and tested nothing; run again \
         without it to test them"
    );
}

/// Where `got` is not `want` field for field -- the same map keys in the
/// same order, the same array lengths, every value but a float equal, and
/// each float within `bound` by [`agree`] -- the first place it is not:
/// [`same_but_floats`] as a finding rather than a panic.
fn first_difference(got: &rmpv::Value, want: &rmpv::Value, bound: f64, at: &str) -> Option<String> {
    use rmpv::Value as V;
    match (got, want) {
        (V::Map(g), V::Map(w)) => {
            if g.len() != w.len() {
                return Some(format!("{at}: {} keys against {}", g.len(), w.len()));
            }
            g.iter().zip(w).find_map(|((gk, gv), (wk, wv))| {
                if gk == wk {
                    first_difference(gv, wv, bound, &format!("{at}.{gk}"))
                } else {
                    Some(format!("{at}: key {gk} against {wk}"))
                }
            })
        }
        (V::Array(g), V::Array(w)) => {
            if g.len() != w.len() {
                return Some(format!("{at}: {} entries against {}", g.len(), w.len()));
            }
            g.iter()
                .zip(w)
                .enumerate()
                .find_map(|(i, (gv, wv))| first_difference(gv, wv, bound, &format!("{at}[{i}]")))
        }
        (V::F64(g), V::F64(w)) => {
            (!agree(*g, *w, bound)).then(|| format!("{at}: {g:e} against {w:e}"))
        }
        _ => (got != want).then(|| format!("{at}: {got} against {want}")),
    }
}

/// Where the current build's run of `case` is not the committed fixture
/// `f`, which another platform wrote: the state's layout exactly and its
/// numbers within the golden files' tolerance, the continuation's inputs
/// to the bit and its reports within the same tolerance. A decay or a
/// log-likelihood runs through libm, whose last bit is the platform's.
/// `garch` calls no platform library (its `ln`, `exp` and `ln_1p` are the
/// `libm` crate's), nor does `tdigest` (its sine and cosine are series and
/// its `sqrt` correctly rounded), nor `ddsketch` (its logarithm and power
/// are series), so their cases are held to the bit: a
/// start fitted on another platform's last bits once took another path, 28%
/// away.
fn differs_from_another_writer(case: &Case, f: &Fixture) -> Option<String> {
    // `ddsketch`'s decayed form multiplies by the platform's `exp2`.
    let exact_ddsketch = f.variant == "DdSketch" && f.name != "ddsketch_decayed";
    let tol = if matches!(f.variant, "Garch" | "TDigest") || exact_ddsketch {
        0.0
    } else {
        TOL
    };
    let (mut m, rows) = saved(case);
    let decode = |b: &[u8]| rmpv::decode::read_value(&mut &b[..]).unwrap();
    let state = first_difference(
        &decode(&named(&m.state())),
        &decode(&f.bytes()),
        tol,
        "state",
    );
    if state.is_some() {
        return state;
    }
    let after = &rows[case.before..];
    let frozen = f.rows();
    if after.len() != frozen.len() {
        return Some(format!(
            "{} continuation rows against {}",
            after.len(),
            frozen.len()
        ));
    }
    let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    let ybits = |v: &[Option<f64>]| v.iter().map(|x| x.map(f64::to_bits)).collect::<Vec<_>>();
    for (i, ((r, fr), want)) in after.iter().zip(&frozen).zip(f.outs()).enumerate() {
        let same_row = bits(&r.x) == bits(&fr.x)
            && ybits(&r.y) == ybits(&fr.y)
            && r.d.to_bits() == fr.d.to_bits()
            && r.w.to_bits() == fr.w.to_bits();
        if !same_row {
            return Some(format!("row {i} after the save: another input"));
        }
        let got = m.step(r);
        let numbers = |o: &Out| {
            o.pred
                .iter()
                .copied()
                .chain([o.n_eff])
                .chain(o.extra.iter().copied())
                .collect::<Vec<_>>()
        };
        let (g, w) = (numbers(&got), numbers(&want));
        if g.len() != w.len() || g.iter().zip(&w).any(|(a, b)| !agree(*a, *b, tol)) {
            return Some(format!("row {i} after the save: another report"));
        }
    }
    None
}

/// **The committed fixtures are what this build writes.** Every case of
/// the current schema is run as `PRINT_STATE_FIXTURES=1` runs it, in
/// memory, and set beside its committed file: where the platform that
/// wrote the file is this one, the text [`file_of`] would write must be
/// the file's, byte for byte; elsewhere the state's layout must be the
/// same and its numbers and the continuation's within the golden
/// tolerance ([`differs_from_another_writer`]). The other checks load a
/// fixture, go on from it and save it again, which a file written before
/// a change in what a case saves still passes: it loads and re-saves its
/// own bytes. This one fails on such a file, and names it, so a change
/// that moves what a case writes regenerates the set in the same change.
#[test]
fn the_committed_fixtures_are_what_this_build_writes() {
    if regenerating() {
        return;
    }
    let mut stale = Vec::new();
    for case in &all() {
        let file = format!("{}.rs", case.name);
        let f = frozen::ALL
            .iter()
            .find(|f| f.name == case.name)
            .unwrap_or_else(|| {
                panic!("{file}: no committed fixture; regenerate with `{REGENERATE}`")
            });
        let why = if f.writer == writer() {
            let committed = std::fs::read_to_string(dir().join(&file))
                .unwrap_or_else(|e| panic!("{file}: {e}"));
            (file_of(case) != committed).then(|| "the text this build writes is another".to_owned())
        } else {
            differs_from_another_writer(case, f)
        };
        if let Some(why) = why {
            stale.push(format!("{file} ({why})"));
        }
    }
    assert!(
        stale.is_empty(),
        "{} committed fixture(s) are not what this build writes: {}. If the change that moved \
         them is intended, regenerate the set with `{REGENERATE}`",
        stale.len(),
        stale.join("; ")
    );
}

/// The anchored MAD sketches' forms (schema 64, task 278, the user's
/// 2026-10-26; schema 65, task 323). `huber_mad_anchored`,
/// `sgd_huber_mad_anchored`, `marginal_bins_huber_anchored`, `quantile_mad`
/// and `huber_mad_awaiting` set the flag with the refit off, every sketch
/// on the grid fixed at 1; `quantile_refit` is saved while its refit
/// waits, its sketches untouched on the grid, and `quantile_refit_done`
/// after it, anchored at the converged residuals' median. The cases
/// without the flag write it `false` and keep no anchor, the form a state
/// saved before schema 64 goes on in. A sketch a state of 64 or earlier
/// held is marked to keep anchoring at its first value:
/// `huber_mad_awaiting_first_value` writes that mark on every sketch,
/// saved awaiting, and `huber_refit_first_value` on every sketch, saved
/// mid-buffer, anchored at its first residual; no other case writes it.
#[test]
fn the_anchored_sketch_forms_are_written() {
    if regenerating() {
        return;
    }
    let decoded = |name: &str| -> rmpv::Value {
        let f = frozen::ALL
            .iter()
            .find(|f| f.name == name)
            .unwrap_or_else(|| panic!("{name}: no fixture; add the case and regenerate"));
        rmpv::decode::read_value(&mut f.bytes().as_slice()).unwrap()
    };
    let robust = ["model", "Robust"];
    let sgd = ["model", "Sgd"];
    let bins = ["model", "Marginal", "bins", "hist", "scales"];
    let robust_cfg = ["model", "Robust", "cfg"];
    let sgd_cfg = ["model", "Sgd", "cfg"];
    let bins_cfg = ["model", "Marginal", "cfg", "bins"];
    #[derive(Clone, Copy, Debug)]
    enum Anchors {
        None,
        Grid,
        Set,
        Awaiting,
    }
    // (case, where its sketches are, where its flag is, whether it is set,
    // its anchors)
    type Form<'a> = (&'a str, &'a [&'a str], &'a [&'a str], bool, Anchors);
    let cases: [Form; 13] = [
        (
            "huber_mad_anchored",
            &robust,
            &robust_cfg,
            true,
            Anchors::Grid,
        ),
        (
            "huber_mad_awaiting_first_value",
            &robust,
            &robust_cfg,
            true,
            Anchors::Awaiting,
        ),
        (
            "huber_refit_first_value",
            &robust,
            &robust_cfg,
            true,
            Anchors::Set,
        ),
        ("quantile_mad", &robust, &robust_cfg, true, Anchors::Grid),
        ("quantile_refit", &robust, &robust_cfg, true, Anchors::Grid),
        (
            "quantile_refit_done",
            &robust,
            &robust_cfg,
            true,
            Anchors::Set,
        ),
        (
            "huber_mad_awaiting",
            &robust,
            &robust_cfg,
            true,
            Anchors::Grid,
        ),
        (
            "sgd_huber_mad_anchored",
            &sgd,
            &sgd_cfg,
            true,
            Anchors::Grid,
        ),
        (
            "marginal_bins_huber_anchored",
            &bins,
            &bins_cfg,
            true,
            Anchors::Grid,
        ),
        (
            "marginal_bins_refit_done",
            &bins,
            &bins_cfg,
            true,
            Anchors::Set,
        ),
        ("huber_mad", &robust, &robust_cfg, false, Anchors::None),
        ("sgd_huber_mad", &sgd, &sgd_cfg, false, Anchors::None),
        (
            "marginal_bins_huber",
            &bins,
            &bins_cfg,
            false,
            Anchors::None,
        ),
    ];
    // The histogram a state of 64 builds at its freeze anchors each sketch
    // at its first deviation, marked so; `marginal_bins_refit_first_value_frozen`
    // is saved past that freeze, its mark `before_65` cleared there (schema
    // 67) and its two sketches marked.
    let marked_cases = [
        "huber_mad_awaiting_first_value",
        "huber_refit_first_value",
        "marginal_bins_refit_first_value_frozen",
    ];
    for (name, place, flag_at, flag, anchors) in cases {
        let v = decoded(name);
        let path: Vec<&str> = flag_at.iter().copied().chain(["mad_anchor"]).collect();
        let got = at(&v, &path).and_then(rmpv::Value::as_bool);
        assert_eq!(got, Some(flag), "{name}: cfg.mad_anchor");
        let path: Vec<&str> = place.iter().copied().chain(["mad"]).collect();
        let sketches = at(&v, &path)
            .and_then(rmpv::Value::as_array)
            .unwrap_or_else(|| panic!("{name}: no sketches"));
        assert_eq!(sketches.len(), 2, "{name}: a sketch a target");
        for q in sketches {
            let mark = at(q, &["anchor_by_first_value"]).and_then(rmpv::Value::as_bool);
            assert_eq!(
                mark,
                marked_cases.contains(&name).then_some(true),
                "{name}: anchor_by_first_value"
            );
            let a = at(q, &["anchor"]).and_then(rmpv::Value::as_f64);
            match anchors {
                Anchors::None => assert_eq!(a, None, "{name}: an anchor without the flag"),
                Anchors::Grid => assert_eq!(a, Some(1.0), "{name}: the grid fixed at 1"),
                Anchors::Set => assert!(
                    a.is_some_and(|a| a > 0.0 && a != 1.0),
                    "{name}: anchor {a:?}"
                ),
                Anchors::Awaiting => assert_eq!(a, Some(0.0), "{name}: awaiting"),
            }
        }
    }
    // While a refit waits its buffers hold each row's `|e|`, a finite
    // number at least 0 or -1 where the scale took none, and the sketch
    // holds nothing; a state of 64 read by this build holds none, its
    // sketch fed.
    for (name, place, keeps) in [
        ("huber_refit", &robust, true),
        ("quantile_refit", &robust, true),
        ("sgd_huber_refit", &sgd, true),
        ("huber_refit_first_value", &robust, false),
    ] {
        let v = decoded(name);
        let path: Vec<&str> = place.iter().copied().chain(["refit"]).collect();
        let bufs = at(&v, &path).and_then(rmpv::Value::as_array).unwrap();
        assert_eq!(bufs.len(), 2, "{name}: a buffer a target");
        for b in bufs {
            let rows = at(b, &["rows"]).and_then(rmpv::Value::as_array).unwrap();
            let learned = at(b, &["learned"]).and_then(rmpv::Value::as_u64).unwrap();
            assert!(!rows.is_empty() && learned > 0, "{name}: an open buffer");
            let resid = at(b, &["resid"]).unwrap();
            if keeps {
                let e = resid.as_array().unwrap();
                assert_eq!(e.len() as u64, learned, "{name}: an |e| a row");
                assert!(
                    e.iter().filter_map(rmpv::Value::as_f64).any(|e| e > 0.0),
                    "{name}: {e:?}"
                );
            } else {
                assert!(resid.is_nil(), "{name}: {resid:?}");
            }
        }
        let path: Vec<&str> = place.iter().copied().chain(["mad"]).collect();
        let fed = at(&v, &path)
            .and_then(rmpv::Value::as_array)
            .unwrap()
            .iter()
            .any(|q| {
                at(q, &["buckets"])
                    .and_then(rmpv::Value::as_array)
                    .is_some_and(|b| !b.is_empty())
            });
        assert_eq!(fed, !keeps, "{name}: the sketch fed");
    }
    // A histogram whose refit waits holds each row's deviations from each
    // target's location; one a state of 64 left inside its warm-up builds
    // its histogram by that build's rule, and says so.
    let v = decoded("marginal_bins_refit");
    let held = at(&v, &["model", "Marginal", "bins", "hist", "scales", "held"]).unwrap();
    let e = at(held, &["e"]).and_then(rmpv::Value::as_array).unwrap();
    let w = at(held, &["w"]).and_then(rmpv::Value::as_array).unwrap();
    assert!(!w.is_empty() && e.len() == 2 * w.len(), "{held:?}");
    assert!(e.iter().filter_map(rmpv::Value::as_f64).any(|e| e > 0.0));
    let v = decoded("marginal_bins_refit_first_value");
    let b = at(&v, &["model", "Marginal", "bins"]).unwrap();
    assert!(at(b, &["hist"]).is_some_and(rmpv::Value::is_nil));
    assert_eq!(
        at(b, &["before_65"]).and_then(rmpv::Value::as_bool),
        Some(true)
    );
    assert!(
        at(b, &["held"])
            .and_then(rmpv::Value::as_array)
            .is_some_and(|h| !h.is_empty())
    );
    // Past its freeze the mark is cleared (schema 67, task 340): the
    // histogram is built, the refit's rows are held on, and the field is
    // not written; the schema-66 build's file of the same case carries it.
    let v = decoded("marginal_bins_refit_first_value_frozen");
    let b = at(&v, &["model", "Marginal", "bins"]).unwrap();
    assert!(at(b, &["hist"]).is_some_and(|h| !h.is_nil()));
    assert!(at(b, &["before_65"]).is_none(), "cleared at the freeze");
    assert!(
        at(b, &["held"])
            .and_then(rmpv::Value::as_array)
            .is_some_and(|h| !h.is_empty()),
        "saved before its refit"
    );
    let kept = frozen::v66::ALL
        .iter()
        .find(|f| f.name == "marginal_bins_refit_first_value_frozen")
        .expect("the 66 build's file of the case");
    let v = rmpv::decode::read_value(&mut kept.bytes().as_slice()).unwrap();
    let b = at(&v, &["model", "Marginal", "bins"]).unwrap();
    assert!(at(b, &["hist"]).is_some_and(|h| !h.is_nil()));
    assert_eq!(
        at(b, &["before_65"]).and_then(rmpv::Value::as_bool),
        Some(true),
        "the kept file is marked past its freeze"
    );
    for f in frozen::ALL {
        if f.name != "marginal_bins_refit_first_value" {
            let v = rmpv::decode::read_value(&mut f.bytes().as_slice()).unwrap();
            assert_eq!(sketches(&v, "before_65"), 0, "{}", f.name);
        }
    }
    // No other fixture writes the mark.
    for f in frozen::ALL {
        let v = rmpv::decode::read_value(&mut f.bytes().as_slice()).unwrap();
        let marked = sketches(&v, "anchor_by_first_value");
        let want = if marked_cases.contains(&f.name) { 2 } else { 0 };
        assert_eq!(marked, want, "{}: marked sketches", f.name);
    }
}

/// A fixture written on another platform is compared number by number, and
/// a configuration can hold an infinity (`solve_every`, `revert_half_life`):
/// equal infinities agree at any tolerance, and unequal numbers at none.
#[test]
fn equal_infinities_agree_at_any_tolerance() {
    for tol in [0.0, 1e-12] {
        assert!(agree(f64::INFINITY, f64::INFINITY, tol));
        assert!(agree(f64::NEG_INFINITY, f64::NEG_INFINITY, tol));
        assert!(!agree(f64::INFINITY, f64::NEG_INFINITY, tol));
        assert!(!agree(f64::INFINITY, 1e300, tol));
        assert!(!agree(1e300, f64::INFINITY, tol));
        assert!(agree(f64::NAN, -f64::NAN, tol));
        assert!(!agree(f64::NAN, 1.0, tol));
    }
    assert!(!agree(1.0, 1.0 + 1e-15, 0.0));
    assert!(agree(1.0, 1.0 + 1e-15, 1e-12));
}

/// A kept `full` or `diag` `hmm` state put under the prior
/// ([`UNDER_THE_PRIOR`]): the kept state holds none, and the converted one
/// holds `ŝ` and `V`, `ŝ` one number a feature, and `f` beside the factor
/// form alone, one a feature; a seeded state's `ŝ` and `V` not 0, and a
/// factor form's states each `σ = 1`, taken up again.
fn under_the_prior_rule(kept: &rmpv::Value, saved: &rmpv::Value, what: &str) {
    assert!(
        at(kept, &["model", "Hmm", "stabilized"]).is_none(),
        "{what}: the kept state holds a prior"
    );
    let d = at(saved, &["model", "Hmm", "cfg", "n_features"])
        .and_then(rmpv::Value::as_u64)
        .unwrap() as usize;
    let list = |path: &[&str]| -> Vec<f64> {
        at(saved, path)
            .and_then(rmpv::Value::as_array)
            .unwrap_or_else(|| panic!("{what}: no {}", path.join(".")))
            .iter()
            .map(|x| x.as_f64().unwrap())
            .collect()
    };
    let scale = list(&["model", "Hmm", "stabilized", "scale"]);
    let prior = list(&["model", "Hmm", "stabilized", "factored_prior"]);
    let weight = at(saved, &["model", "Hmm", "stabilized", "scale_weight"])
        .and_then(rmpv::Value::as_f64)
        .unwrap();
    let seeded = at(saved, &["model", "Hmm", "seeded"])
        .and_then(rmpv::Value::as_bool)
        .unwrap();
    let factors = at(saved, &["model", "Hmm", "factor_states"])
        .and_then(rmpv::Value::as_array)
        .map_or(0, Vec::len);
    assert_eq!(scale.len(), d, "{what}: ŝ");
    assert_eq!(prior.len(), if factors > 0 { d } else { 0 }, "{what}: f");
    assert_eq!(
        seeded,
        weight > 0.0 && scale.iter().all(|s| *s > 0.0),
        "{what}: ŝ and V"
    );
    for f in at(saved, &["model", "Hmm", "factor_states"])
        .and_then(rmpv::Value::as_array)
        .into_iter()
        .flatten()
    {
        assert_eq!(
            at(f, &["sigma"]).and_then(rmpv::Value::as_f64),
            Some(1.0),
            "{what}: taken up again"
        );
    }
}

/// A `huber` or `quantile` state of 71 or earlier, loaded and saved
/// ([`started_fresh`]): every key the kept state held, in the current
/// fixture's layout key for key, with each target's moments over every row
/// empty -- no weight, every moment 0 -- and its scales none under
/// `standardize`, and neither without it.
fn started_fresh_rule(kept: &rmpv::Value, saved: &rmpv::Value, current: &rmpv::Value, what: &str) {
    let model = |v: &rmpv::Value| -> Vec<(String, rmpv::Value)> {
        at(v, &["model", "Robust"])
            .and_then(rmpv::Value::as_map)
            .unwrap_or_else(|| panic!("{what}: no Robust"))
            .iter()
            .map(|(k, v)| (k.as_str().unwrap().to_owned(), v.clone()))
            .collect()
    };
    let keys = |m: &[(String, rmpv::Value)]| m.iter().map(|(k, _)| k.clone()).collect::<Vec<_>>();
    let (s, c) = (model(saved), model(current));
    assert_eq!(keys(&s), keys(&c), "{what}: the current layout");
    let get = |m: &[(String, rmpv::Value)], k: &str| {
        m.iter().find(|(n, _)| n == k).map(|(_, v)| v.clone())
    };
    for (k, _) in model(kept) {
        assert!(get(&s, &k).is_some(), "{what}: {k} dropped");
    }
    let standardize = at(saved, &["model", "Robust", "cfg", "standardize"])
        .and_then(rmpv::Value::as_bool)
        .unwrap();
    let spread = get(&s, "spread").unwrap();
    let scales = get(&s, "spread_scale").unwrap();
    let (spread, scales) = (spread.as_array().unwrap(), scales.as_array().unwrap());
    if standardize {
        assert!(
            !spread.is_empty() && spread.len() == scales.len(),
            "{what}: one a target"
        );
        for d in spread {
            assert_eq!(
                held(d) - usize::from(at(d, &["k"]).is_some()),
                0,
                "{what}: {d}"
            );
        }
        assert!(
            scales
                .iter()
                .all(|v| v.as_array().is_some_and(Vec::is_empty)),
            "{what}"
        );
    } else {
        assert!(spread.is_empty() && scales.is_empty(), "{what}");
    }
}

/// A `deco` state of 71 or earlier, loaded and saved ([`RESTARTED_LEVEL`]):
/// the kept state held the older level (`rho_bar`, `rho_bar_lo`) and no
/// sums; the saved one holds the kept standardiser, cfg and blocks as they
/// were, every `rho` NaN, every weight 0, the sums empty with the share at 1,
/// and the current fixture's layout, key for key.
fn restarted_level_rule(
    kept: &rmpv::Value,
    saved: &rmpv::Value,
    current: &rmpv::Value,
    what: &str,
) {
    let path = |k: &str| ["model", "Deco", k].map(String::from);
    let get = |v: &rmpv::Value, k: &str| at(v, &path(k).each_ref().map(String::as_str)).cloned();
    assert!(
        get(kept, "rho_bar").is_some() && get(kept, "ratio").is_none(),
        "{what}: the kept layout"
    );
    assert!(
        get(saved, "rho_bar").is_none() && get(saved, "rho_bar_lo").is_none(),
        "{what}: rho_bar kept"
    );
    for key in ["cfg", "diag", "blocks"] {
        assert_eq!(get(saved, key), get(kept, key), "{what}: {key} moved");
    }
    let floats = |v: Option<rmpv::Value>| -> Vec<f64> {
        v.and_then(|v| v.as_array().cloned())
            .unwrap_or_default()
            .iter()
            .map(|x| x.as_f64().unwrap())
            .collect()
    };
    let rho = floats(get(saved, "rho"));
    assert!(
        !rho.is_empty() && rho.iter().all(|v| v.is_nan()),
        "{what}: rho {rho:?}"
    );
    assert!(
        floats(get(saved, "rho_w")).iter().all(|&w| w == 0.0),
        "{what}: the weights"
    );
    let ratio = get(saved, "ratio").unwrap_or_else(|| panic!("{what}: no sums"));
    assert_eq!(
        at(&ratio, &["w2_share"]).and_then(rmpv::Value::as_f64),
        Some(1.0),
        "{what}"
    );
    for key in ["num", "den"] {
        let v = floats(at(&ratio, &[key]).cloned());
        assert!(
            v.len() == rho.len() && v.iter().all(|x| x.is_nan()),
            "{what}: {key} {v:?}"
        );
    }
    let keys = |v: &rmpv::Value| -> Vec<String> {
        at(v, &["model", "Deco"])
            .and_then(rmpv::Value::as_map)
            .unwrap()
            .iter()
            .map(|(k, _)| k.as_str().unwrap().to_owned())
            .collect()
    };
    assert_eq!(keys(saved), keys(current), "{what}: the current layout");
}

/// The Kish sums' exponents (schema 72, review 2026-11-23, REG-12): each
/// `*_tiny_weights` case is fed every weight times `2^-664`, so each sum of
/// squared weights it keeps writes a non-zero exponent beside it -- the
/// warm-up's and every level's `w2_scale` in `sgd`, `rls`'s `s2_scale`,
/// and every Gram's `q_sum` in `ewridge`, written as `[q, s]` there -- and
/// the cases at ordinary weights write none, the bytes they had.
#[test]
fn the_kish_sums_exponents_are_written() {
    if regenerating() {
        return;
    }
    fn found(v: &rmpv::Value, key: &str, out: &mut Vec<i64>) {
        match v {
            rmpv::Value::Map(m) => {
                for (k, v) in m {
                    if k.as_str() == Some(key) {
                        // `q_sum` is a float, or `[q, s]` beside a power.
                        match v {
                            rmpv::Value::Array(a) => out.push(a[1].as_i64().unwrap()),
                            v if key == "q_sum" => {
                                assert!(v.is_f64() || v.is_nil(), "{v}");
                            }
                            v => out.push(v.as_i64().unwrap()),
                        }
                    }
                    found(v, key, out);
                }
            }
            rmpv::Value::Array(a) => a.iter().for_each(|v| found(v, key, out)),
            _ => {}
        }
    }
    let scales = |name: &str, key: &str| {
        let f = frozen::ALL.iter().find(|f| f.name == name).unwrap();
        let v = rmpv::decode::read_value(&mut f.bytes().as_slice()).unwrap();
        let mut out = Vec::new();
        found(&v, key, &mut out);
        out
    };
    for (name, key, at_least) in [
        ("sgd_tiny_weights", "w2_scale", 3),
        ("rls_tiny_weights", "s2_scale", 1),
        ("ewridge_tiny_weights", "q_sum", 1),
    ] {
        let s = scales(name, key);
        assert!(
            s.len() >= at_least && s.iter().all(|&v| v != 0),
            "{name}: {key} {s:?}"
        );
    }
    for (name, key) in [
        ("sgd_spread", "w2_scale"),
        ("rls", "s2_scale"),
        ("ewridge", "q_scale"),
    ] {
        assert_eq!(scales(name, key), Vec::<i64>::new(), "{name}: {key}");
    }
}

/// The prior's shape flag (schema 72, review 2026-11-23, HMM-3): given
/// states of the same diagonal with variances that differ keep their shape
/// and write `shaped`, `true`; the cases with no shape write none, the bytes
/// they had, and a state without it loads as the build before read it.
#[test]
fn the_priors_shape_flag_is_written_where_a_shape_is_kept() {
    if regenerating() {
        return;
    }
    let shaped = |name: &str| {
        let f = frozen::ALL
            .iter()
            .find(|f| f.name == name)
            .unwrap_or_else(|| panic!("{name}: no fixture"));
        let v = rmpv::decode::read_value(&mut f.bytes().as_slice()).unwrap();
        assert!(
            at(&v, &["model", "Hmm", "stabilized", "scale"]).is_some(),
            "{name}: no prior"
        );
        at(&v, &["model", "Hmm", "stabilized", "shaped"]).cloned()
    };
    assert_eq!(shaped("hmm_given"), Some(rmpv::Value::Boolean(true)));
    for name in [
        "hmm_diag",
        "hmm_full",
        "hmm_full_narrow",
        "hmm_full_warming",
    ] {
        assert_eq!(shaped(name), None, "{name}");
    }
}

/// The scales from every row (schema 72, review 2026-11-23, REG-18): each
/// standardized `huber` and `quantile` case keeps, per target, its
/// features' moments over every row it learned, holding weight, and the
/// scales its band last moved at, one per feature; a case without
/// `standardize` keeps neither.
#[test]
fn the_scales_from_every_row_are_written() {
    if regenerating() {
        return;
    }
    let mut kept = 0;
    for f in frozen::ALL
        .iter()
        .filter(|f| f.name.starts_with("huber") || f.name.starts_with("quantile"))
    {
        let v = rmpv::decode::read_value(&mut f.bytes().as_slice()).unwrap();
        let get = |k: &str| at(&v, &["model", "Robust", k]).cloned().unwrap();
        let standardize = get("cfg")
            .as_map()
            .unwrap()
            .iter()
            .find(|(k, _)| k.as_str() == Some("standardize"))
            .and_then(|(_, v)| v.as_bool())
            .unwrap();
        let (spread, scales) = (get("spread"), get("spread_scale"));
        let (spread, scales) = (spread.as_array().unwrap(), scales.as_array().unwrap());
        if standardize {
            assert!(
                !spread.is_empty() && spread.len() == scales.len(),
                "{}",
                f.name
            );
            for d in spread {
                let w = at(d, &["w_sum"]).and_then(rmpv::Value::as_f64).unwrap();
                assert!(w > 0.0, "{}: {d}", f.name);
            }
            assert!(
                scales
                    .iter()
                    .any(|s| s.as_array().is_some_and(|a| !a.is_empty())),
                "{}: no scales",
                f.name
            );
            kept += 1;
        } else {
            assert!(spread.is_empty() && scales.is_empty(), "{}", f.name);
        }
    }
    assert!(kept >= 10, "{kept} standardized cases");
}

/// A windowed `marginal`'s lag moments keep, beside the lagged sides about
/// the origin, the sides now over the rows a lagged one was matched with
/// and those rows' weights, in one vector, `matched` (review 2026-11-23,
/// MARG-1): every current set of a windowed lag case writes it, in the live
/// moments and in every snapshot, non-empty, and the case whose second
/// target joins late holds a weight over the matched rows below its rows'
/// where the rows before it lacked a partner. A set of 71 or earlier
/// writes none, the form a state saved before them takes (its load starts
/// them).
#[test]
fn the_sides_over_the_matched_rows_are_written() {
    if regenerating() {
        return;
    }
    fn found<'v>(v: &'v rmpv::Value, out: &mut Vec<&'v rmpv::Value>) {
        match v {
            rmpv::Value::Map(m) => {
                for (k, v) in m {
                    if k.as_str() == Some("matched") {
                        out.push(v);
                    }
                    found(v, out);
                }
            }
            rmpv::Value::Array(a) => a.iter().for_each(|v| found(v, out)),
            _ => {}
        }
    }
    let decoded =
        |f: &crate::cases::Fixture| rmpv::decode::read_value(&mut f.bytes().as_slice()).unwrap();
    let floats = |m: &rmpv::Value| -> Vec<f64> {
        m.as_array()
            .unwrap_or_else(|| panic!("{m}"))
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect()
    };
    for name in ["marginal_window_lags", "marginal_window_lags_joining"] {
        let f = frozen::ALL.iter().find(|f| f.name == name).unwrap();
        let v = decoded(f);
        let mut out = Vec::new();
        found(&v, &mut out);
        // The live moments and at least one snapshot.
        assert!(out.len() >= 2, "{name}: written {} times", out.len());
        assert!(out.iter().all(|m| !floats(m).is_empty()), "{name}");
        if name == "marginal_window_lags_joining" {
            // Some value strictly inside (0, 1): a weight over the matched
            // rows below the rows' own (the weights are each target's share).
            let inside = out
                .iter()
                .flat_map(|m| floats(m))
                .any(|w| w > 0.0 && w < 0.999);
            assert!(
                inside,
                "{name}: no weight over the matched rows below the rows'"
            );
        }
    }
    for (schema, set) in previous_sets() {
        for f in set.iter().filter(|f| f.name == "marginal_window_lags") {
            let v = decoded(f);
            let mut out = Vec::new();
            found(&v, &mut out);
            assert!(
                out.is_empty(),
                "v{schema}/{}: matched in a set before it",
                f.name
            );
        }
    }
}

/// Every nonlinear `kalman` fixture writes its `obs` configuration non-empty
/// (`filter`, `family`, `link`, `ukf_rule`, `ukf_noise`), while every linear
/// `kalman` fixture writes no `obs` key at all.
#[test]
fn kalman_nonlinear_fixtures_write_their_obs_forms() {
    if regenerating() {
        return;
    }
    let decoded =
        |f: &crate::cases::Fixture| rmpv::decode::read_value(&mut f.bytes().as_slice()).unwrap();
    let expected = [
        (
            "kalman_ekf_poisson",
            "poisson",
            "log",
            "ekf",
            "scalar_index",
            "expected",
        ),
        (
            "kalman_ukf_gamma_inverse",
            "gamma",
            "inverse",
            "ukf",
            "scalar_index",
            "expected",
        ),
        (
            "kalman_ukf_gaussian_logit",
            "gaussian",
            "logit",
            "ukf",
            "scalar_index",
            "plugin",
        ),
        (
            "kalman_ukf_merwe_binomial",
            "binomial",
            "logit",
            "ukf",
            "merwe",
            "expected",
        ),
    ];
    for (name, fam, lnk, flt, rule, noise) in expected {
        let f = frozen::ALL
            .iter()
            .find(|f| f.name == name)
            .unwrap_or_else(|| panic!("{name}: missing fixture"));
        let v = decoded(f);
        let obs = at(&v, &["model", "Kalman", "cfg", "obs"])
            .unwrap_or_else(|| panic!("{name}: cfg.obs missing"));
        assert_eq!(
            at(obs, &["family"]).and_then(rmpv::Value::as_str),
            Some(fam),
            "{name}"
        );
        assert_eq!(
            at(obs, &["link"]).and_then(rmpv::Value::as_str),
            Some(lnk),
            "{name}"
        );
        assert_eq!(
            at(obs, &["filter"]).and_then(rmpv::Value::as_str),
            Some(flt),
            "{name}"
        );
        assert_eq!(
            at(obs, &["ukf_rule"]).and_then(rmpv::Value::as_str),
            Some(rule),
            "{name}"
        );
        assert_eq!(
            at(obs, &["ukf_noise"]).and_then(rmpv::Value::as_str),
            Some(noise),
            "{name}"
        );
    }
    for name in [
        "kalman",
        "kalman_gap",
        "kalman_noise_units",
        "kalman_output_only",
        "kalman_square_root",
        "kalman_smooth",
    ] {
        let f = frozen::ALL.iter().find(|f| f.name == name).unwrap();
        let v = decoded(f);
        assert!(
            at(&v, &["model", "Kalman", "cfg", "obs"]).is_none(),
            "{name}: linear kalman fixture must not write cfg.obs"
        );
    }
}
