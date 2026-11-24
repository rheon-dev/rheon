//! Frozen state fixtures for the files a bank and its helpers write: a bank
//! file (several specs, a group closed and drained, rows held under an
//! `embargo`, a temporal clock, a window), a second bank file of the
//! diagnostics with a memory of their own (`bank_diagnosed`), a third of an autocorrelation's
//! lagged values (`bank_autocorr`), a `with_windows` state holding
//! rows for a window that looks ahead, a `refresh_time` state part-way
//! through an interval, and one holding the rest of a tick's pair points
//! that a slice left out (`refresh_time_held`), two states of a merge of
//! streams (`merge`, holding rows at a tie; `merge_by`, under `by` and a
//! restart) -- each embedded as its bytes beside the input that
//! follows it and the frames that input gave (docs/PLAN.md task 198; review
//! round 4, D1). The models' own states are `rheon-core`'s
//! `tests/state_fixtures.rs`, on the same three checks:
//!
//! 1. **it loads**;
//! 2. **it goes on to the bit**: the next input gives the frames it gave
//!    when the fixture was written -- to the bit on the platform that wrote
//!    it, to the golden pipeline's tolerance elsewhere (libm's last bit
//!    differs by platform); a null is a null, and every other value equal;
//! 3. **it saves its bytes again**.
//!
//! The inputs and outputs are frozen as Arrow IPC (a format polars reads
//! back however it writes), in hex, as the states are: no binary file
//! (CLAUDE.md hard rule 1). Regenerate with
//!
//! ```text
//! PRINT_STATE_FIXTURES=1 cargo test -p rheon-polars --test state_fixtures
//! ```
//!
//! before 1.0, whenever the bank's schema, the windows state's version or
//! the `refresh_time` state's moves, or a number does; the bank's minimum
//! schema moves with the schema until 1.0 and stays at 1.0's after, as
//! `rheon-core`'s does. The windows state and the formula tree were
//! labelled unstable (docs/PLAN.md task 198, D5) until PLAN 542 wrote every
//! state file in one envelope: since then a window run's state, the
//! formula tree inside it and a pipeline checkpoint are kept with a loader
//! like the rest.
//!
//! **From 1.0** a layout change keeps the fixtures of the schema before it
//! (docs/PLAN.md §18 D1; review round 5, D1 and D2), as `rheon-core`'s
//! harness does. The first 1.x layout change, from schema 44 to 45:
//!
//! 1. copy the kinds the promise covers -- every current file, since PLAN
//!    542 the window runs' and the checkpoint's too -- to
//!    `tests/state_fixtures/v44/`, the directory the kept set, which
//!    `build.rs` finds;
//! 2. raise `SCHEMA_VERSION` to 45 and write the loader;
//! 3. regenerate the current set.
//!
//! No index lists the files: `build.rs` discovers every `<kind>.rs` under
//! `tests/state_fixtures/`, and every one under each `v<N>/`, on each
//! build, so two branches that each add a kind add two files and edit no
//! shared list.
//!
//! Each kept set is held by `every_previous_fixture_loads_goes_on_and_converts_to_the_current_bytes`:
//! every fixture loads through the loader, goes on to the bit against its
//! own frozen frames, and -- loaded, then saved -- writes the current
//! schema's fixture bytes of the same kind, the two being the same stream
//! saved at the same row (the exact-conversion check).

/// The fixtures, discovered: `build.rs` lists every file under
/// `tests/state_fixtures/` and each `v<N>/` beside them, so a kind added is
/// a file added and no list is edited by hand.
mod frozen {
    use crate::Fixture;
    include!(concat!(env!("OUT_DIR"), "/state_fixtures_index.rs"));
}

use std::fmt::Write as _;
use std::io::Cursor;
use std::path::{Path, PathBuf};

use polars::prelude::*;
use rheon_polars::rheon_core::SCHEMA_VERSION;
use rheon_polars::rheon_core::bars::BarKind;
use rheon_polars::{
    AggKind, AggSpec, Bank, BarThreshold, Bars, BarsConfig, MIN_BANK_SCHEMA_VERSION, Merge,
    MergeConfig, RefreshCols, RefreshTime, Spec, StateKind, WindowsConfig, WindowsRun,
    checkpoint_bytes, load_checkpoint,
};

/// One frozen fixture, as `tests/state_fixtures/<name>.rs` holds it.
pub struct Fixture {
    pub name: &'static str,
    /// The schema of the build that wrote it (`rheon_core::SCHEMA_VERSION`);
    /// a helper's state carries a version of its own in its bytes, which
    /// its loader checks.
    pub schema: u32,
    pub writer: &'static str,
    pub state: &'static str,
    /// The input the continuation is fed, as Arrow IPC in hex.
    pub input: &'static str,
    /// What it gave, each frame named, as Arrow IPC in hex.
    pub output: &'static [(&'static str, &'static str)],
}

const REGENERATE: &str = "PRINT_STATE_FIXTURES=1 cargo test -p rheon-polars --test state_fixtures";

/// The schemas before the current one whose fixtures are kept, each under
/// `tests/state_fixtures/v<N>/` and loaded by this build: `previous` lists
/// them, and `previous_sets` is each set as `build.rs` found it in its
/// directory. None before 1.0
/// (the module doc has the steps from 1.0). 61, 1.0's, kept since 62
/// (docs/PLAN.md task 257): the bank files and `refresh_time` states. 62
/// kept since 63 moved `huber`'s and `sgd`'s layout (docs/PLAN.md task
/// 271): no bank fixture holds either model, so each converts to the
/// current bytes exactly. The bank files that hold a window (`bank`,
/// `bank_diagnosed`) convert otherwise since the windows' running products
/// (schema 63 before its release, docs/PLAN.md task 275; the core's
/// `state_fixtures.rs` has why): their first output row is the old build's
/// to the bit, the rest to the golden tolerance, and the state they save is
/// not compared with the current fixture's. 63 kept since 64 gave `huber`
/// and `sgd(loss = "huber")` the refit of each target's first rows
/// (docs/PLAN.md task 277): no bank fixture holds either model, so each
/// converts to the current bytes exactly, the windowed ones as above.
/// Since task 278 a spec's `quantile` draws its band in the MAD and its
/// `min_weight` holds back no learning (D-Q, D10), and `bank_diagnosed`
/// holds a quantile fit: a kept one loads in the old form (the RMS, the
/// gate holding learning back) and goes on in it to the bit, so the current
/// fixture's fit is set to that form before its first row
/// ([`kept_quantile`]) and the kept ones convert to its bytes exactly.
/// Likewise, since task 278's batch 17 a stream a spec builds folds into its
/// residual spread each slot's own prediction on the rows a gate withholds,
/// and weighs its residual autocorrelation by the row's weight
/// (`Persisted::ungated_spread`, `Persisted::weighted_autocorr`); a kept
/// state loads with neither and goes on so to the bit, so the current
/// `bank`, `bank_diagnosed` and `bank_autocorr` streams are set to that form
/// before their first row ([`kept_rules`]), and `bank_gated` freezes the
/// form a spec builds. So too, since task 278, a spec's drift detector
/// folds each slot's own residual on the rows a gate withholds
/// (`Persisted::ungated_drift`), and a spec's `ewridge` with a `min_weight`
/// learns its residual variance and first solve at its own floor
/// (`min_weight_output_only`): [`kept_rules`] leaves both out of the kept
/// forms, `bank_reset` freezes the first, and `bank_gated` the second.
///
/// 64 kept since 65 stopped choosing a MAD sketch's anchor from the
/// residuals one pass sees (task 323): while a refit waits the MAD is read
/// exactly from the `|e|` its buffer keeps beside each row, and a sketch no
/// refit anchors keeps the grid fixed at 1. A kept bank's anchored sketches
/// are anchored at their first value, or await it, and its refit buffers
/// hold no `|e|`: each model's restore marks its anchored sketches to keep
/// that rule and reads its scale from the sketch until its refit, so it
/// goes on to the bit, but the rows before the save were scaled by the old
/// rule, so its state is not the current fixture's. A kind whose kept bank
/// holds an anchored sketch, or whose current one holds an `|e|` for a
/// refit (`bank_refit`), is held to its marks and to saving bytes that load
/// back to the same ([`kept_scale_rule`]).
///
/// 65 kept since 66 made two changes (tasks 327 and 328): `hmm` and
/// `ew_class` under `covariance_type = "diag"` keep their variances alone,
/// and `corrchange(kind = "window")` keeps the count of rows its ring has
/// taken (`ring_rows`). No bank file holds a diagonal `hmm` or `ew_class`
/// (the core's fixtures hold both conversions). `bank_diagnosed`'s
/// `corr_window`, ten rows a window over two columns, is read whole, as
/// before, so it goes on to the bit; a kept one has no count and is
/// numbered from its ring's oldest row, and that file holds a window, so
/// its state is held as above. Its spec leaves `permute_every_rows` unset,
/// whose default is now the window's rows: a kept state's config keeps the
/// 50 it was saved with, and goes on with it.
///
/// 66 kept since 67 made two changes (task 340). A stream keeps the
/// readiness notices a call that learned and then failed left in it
/// (`Notified::pending`), which a build before held in memory alone, so a
/// bank saved and loaded between such a call and its next never said them;
/// a 66 file holds none and loads with none, so every kept kind converts
/// to the current bytes exactly, the windowed ones as above. The kind that
/// holds some, `bank_notice_pending`, is new at 67 and has no kept file:
/// one the 66 build wrote would hold the flags that the notices were raised
/// and not their text, another stream than the current fixture's (the
/// loader of such a file is held by `bank/notices_tests.rs`). And
/// `marginal`'s mark `before_65` is cleared once its histogram is built,
/// which no bank file holds (the core's fixtures hold that conversion).
///
/// 67 kept since 68 let `kalman` hold its covariance as the square root's
/// factor (docs/PLAN.md task 281 (3)), written only where it is not the
/// covariance form; no bank file holds a `kalman`, and a 67 file loads as
/// it was, so every kept kind converts to the current bytes exactly.
///
/// And since review P1 (rule 5's twenty-first exception, the user's,
/// 2026-11-20), under `label_clock` a row of weight 0 is held and learned at
/// its label time, as `embargo` learns it, where a build before scored it
/// and held nothing. A state saved before holds none of its zero-weight
/// rows, so the two kinds whose frame has them, `bank_label_clock` and
/// `bank_conformal_auto`, continue by the new rule from the rows they hold,
/// not to their frozen frames nor to the current fixture's; no layout
/// changed. PLAN 539 then counts such a row in no clock, and holds it for
/// nothing ([`ZERO_WEIGHT_CLOCK`]).
///
/// 69 kept since 70 put `hmm` under `"full"` and `"diag"` that learns under
/// a stabilized prior (rule 5's twenty-second exception, the user's,
/// 2026-11-20): its state gains the prior, and every `hmm` state writes its
/// factor list and its configuration's `prior_rows`, and a spec its
/// `prior_rows`. The bank files hold one `hmm`, under `"tied"`, which takes
/// no prior and goes on to the bit, so every kept kind converts to the
/// current bytes exactly. The 69 set was written by the build that held the
/// zero-weight rows, so its two label kinds converted exactly too, until
/// PLAN 539 ([`ZERO_WEIGHT_CLOCK`]).
///
/// 70 kept since 71 ended a bank file and a window run's state (version 11)
/// with a checksum of their bytes (review P7). A kept file has none and
/// loads as it did, so every kept kind goes on to the bit and converts to
/// the current bytes exactly, the windows states included. No bank file
/// here holds a formula target; a 70 bank whose target's core is version 10
/// is held by `state_checksum.rs`.
///
/// 71 kept since 72 wrote every state file in one envelope (PLAN 542): its
/// first entries are `magic` (`"rheon-state"`), `kind` and `version`, its
/// last the checksum; a window run's state is version 12 and a
/// `refresh_time` state version 3, which gains the checksum and names its
/// clock's kind `clock_kind`. A kept file begins with its kind's old magic
/// and loads through its loader, so every kept kind goes on to the bit and
/// converts to the current bytes exactly. Two kinds are new at 72 and have
/// no kept file: `bank_formula`, a bank holding a formula target's window
/// core, the form a bank file embeds a window run's state in, and
/// `checkpoint`, a pipeline checkpoint of a `refresh_time` state, a window
/// run's and that bank's, and, from PLAN 541, a merge's too. A merge's
/// state is a kind of its own, version 1, new at 72 with `merge` and
/// `merge_by`; and, from PLAN 556, `merge_carry`, a merge carrying values
/// forward, and `refresh_time_age` and `refresh_time_age_held`, grids that
/// give each value's age, each new at 72 in a version that did not move,
/// since a state without either writes the bytes it did. The 71 set was written by the build that counts a
/// zero-weight row in no clock (PLAN 539), so [`ZERO_WEIGHT_CLOCK`] is for
/// the sets before it.
///
/// The kept schemas, each a `v<N>/` directory found beside the current set.
fn previous() -> Vec<u32> {
    frozen::PREVIOUS_SETS.iter().map(|(s, _)| *s).collect()
}

/// Each kept schema's set, as `build.rs` found it on disk.
fn previous_sets() -> Vec<(u32, &'static [&'static Fixture])> {
    frozen::PREVIOUS_SETS.to_vec()
}

/// The golden pipeline's tolerance, off the writer's platform.
const TOL: f64 = 1e-12;

fn regenerating() -> bool {
    std::env::var("PRINT_STATE_FIXTURES").is_ok_and(|v| v == "1")
}

fn writer() -> String {
    format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH)
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn unhex(s: &str) -> Vec<u8> {
    let s: Vec<u8> = s.bytes().filter(|b| !b.is_ascii_whitespace()).collect();
    s.chunks(2)
        .map(|p| u8::from_str_radix(std::str::from_utf8(p).unwrap(), 16).unwrap())
        .collect()
}

fn ipc(df: &DataFrame) -> Vec<u8> {
    let mut out = Vec::new();
    IpcWriter::new(&mut out)
        .finish(&mut df.clone())
        .expect("a frame writes as IPC");
    out
}

fn from_ipc(hex_text: &str) -> DataFrame {
    IpcReader::new(Cursor::new(unhex(hex_text)))
        .finish()
        .expect("a frozen frame reads back")
}

// --- the streams ------------------------------------------------------------

/// The bank's specs: a temporal clock with duration parameters and a weight,
/// rows held under an `embargo`, groups closed as the key grows, a window,
/// and the integer clock's forms (task 200; review round 5, B1): an `Int64`
/// clock under a fractional `gap_cap` and `embargo` (rows held, each with
/// an integer stamp), saved right after a skipped row, so the removed and
/// pending ticks hold time and are written; a `UInt32`
/// clock that starts over at each session, under a `session_gap`, for the
/// width and the elapsed clock's removed ticks; a group closed on
/// session, whose PCA continuity is kept per group (`pca_prev_by_group`);
/// and a `min_weight` above the ceiling its half-life allows, saved inside
/// the wait its readiness notice counts (task 208, schema 46): the floor
/// and the noise gate have withheld every row since the stream was 95%
/// settled, a half-life not yet past, so the notice comes in the
/// continuation. And a windowed `ewridge` with its residual spread (task
/// 217, schema 52): the row counts its cross-moments keep, live and in its
/// window's snapshots, and the spread ring's own, per slot. The `audit`
/// is in the second file's specs ([`diagnosed_specs`]).
fn specs() -> Vec<Spec> {
    [
        r#"{"name": "dated", "model": {"type": "ewridge"}, "targets": ["y"], "features": ["x0", "x1"], "clock": "ts", "half_life": "10m", "gap_cap": "1h", "weight": "w"}"#,
        r#"{"name": "embargoed", "model": {"type": "ewridge"}, "targets": ["y"], "features": ["x0", "x1"], "clock": "t", "half_life": 40.0, "gap_cap": 50.0, "embargo": 3.0}"#,
        r#"{"name": "closing", "model": {"type": "ew_cov"}, "targets": ["x0"], "features": ["x0", "x1"], "clock": "t", "half_life": 20.0, "gap_cap": 50.0, "group": "g", "group_close": "monotone"}"#,
        r#"{"name": "windowed", "model": {"type": "ew_cov", "window_size": 12.0, "bias": true}, "targets": ["x0"], "features": ["x0", "x1"], "clock": "t", "half_life": 20.0, "gap_cap": 50.0}"#,
        r#"{"name": "int_clock", "model": {"type": "ewridge"}, "targets": ["y"], "features": ["x0", "xi"], "clock": "ti", "half_life": 160.0, "gap_cap": 6.5, "embargo": 12.5, "weight": "w"}"#,
        r#"{"name": "uint_clock", "model": {"type": "ew_cov"}, "targets": ["x0"], "features": ["x0", "x1"], "clock": "tu", "half_life": 80.0, "gap_cap": 6.5, "session": "s", "session_gap": 3.5}"#,
        r#"{"name": "sessioned", "model": {"type": "ew_cov", "pca": 1}, "targets": ["x0"], "features": ["x0", "x1"], "clock": "t", "half_life": 20.0, "gap_cap": 50.0, "group": "g", "session": "s", "group_close": "session"}"#,
        r#"{"name": "waiting", "model": {"type": "ewridge"}, "targets": ["y"], "features": ["x0", "x1"], "half_life": 16.0, "min_weight": 30.0}"#,
        r#"{"name": "spread_window", "model": {"type": "ewridge", "window_size": 12.0}, "targets": ["y"], "features": ["x0", "x1"], "clock": "t", "half_life": 20.0, "gap_cap": 50.0, "emit_sigma": true}"#,
    ]
    .iter()
    .map(|text| serde_json::from_str(text).unwrap_or_else(|e| panic!("{text}: {e}")))
    .collect()
}

/// The specs of the second bank file, `bank_diagnosed`: the diagnostics
/// with a memory of their own (docs/PLAN.md task 221, schema 55), in a file
/// of their own so that neither file passes the 250 KB a source file is
/// held to. A half-life of 10 clock units settles the stream 95% well
/// before the save, from which the diagnostics fold (task 232 (1)). And an
/// `audit` with its pairs (task 223 (b), schema 54): a spec the bank's file
/// names, which reads every row. It moved here from the first file when the
/// fourth `withheld_reason` (task 239) took that file past the cap. And
/// the keys and words a spec took at 1.0 (task 244, schema 61), each in the
/// file's specs: `kmeans`' `n_clusters` and `init = "k-means++"`, `hmm`'s
/// `n_components` and `covariance_type = "tied"`, `bocpd`'s
/// `covariance_type = "full"` and its four priors, `corrchange`'s
/// `window_rows` and `change_action = "reset"`, and `ew_cov`'s
/// `max_rows_between_pcas` and `snapshot_every`.
fn diagnosed_specs() -> Vec<Spec> {
    [
        r#"{"name": "diagnosed", "model": {"type": "ewridge"}, "targets": ["y"], "features": ["x0", "x1"], "clock": "t", "half_life": 10.0, "gap_cap": 50.0, "embargo": 2.0, "horizon_rows": 2, "emit_calibration": true, "calibration_half_life": 90.0, "emit_breaks": true, "emit_robust_se": true, "robust_se_lags": 2, "coef_every": 4.0, "emit_specification": true, "ljung_box_lags": 3, "emit_tails": true, "emit_influence": true, "emit_feature_health": true}"#,
        r#"{"name": "quantiled", "model": {"type": "quantile", "quantile": 0.8}, "targets": ["y"], "features": ["x0", "x1"], "clock": "t", "half_life": 10.0, "gap_cap": 50.0, "embargo": 2.0, "horizon_rows": 2, "emit_calibration": true, "emit_breaks": true}"#,
        r#"{"name": "audited", "model": {"type": "audit", "pairs": true}, "targets": ["x0"], "features": ["x0", "x1", "y"], "clock": "t", "gap_cap": 50.0}"#,
        r#"{"name": "clustered", "model": {"type": "kmeans", "n_clusters": 2, "init": "k-means++", "warm_rows": 8}, "targets": ["x0"], "features": ["x0", "x1"], "clock": "t", "half_life": 10.0, "gap_cap": 50.0}"#,
        r#"{"name": "regimes", "model": {"type": "hmm", "n_components": 2, "covariance_type": "tied", "precision_prior": 0.01, "warm_rows": 8, "init": "first"}, "targets": ["x0"], "features": ["x0", "x1"], "clock": "t", "half_life": 10.0, "gap_cap": 50.0}"#,
        r#"{"name": "changing", "model": {"type": "bocpd", "covariance_type": "full", "mean_prior": [0.0, 0.0], "mean_precision_prior": 1.0, "degrees_of_freedom_prior": 5.0, "covariance_prior": [1.0], "max_run": 40}, "targets": ["x0"], "features": ["x0", "x1"], "clock": "t", "gap_cap": 50.0}"#,
        r#"{"name": "corr_window", "model": {"type": "corrchange", "kind": "window", "window_rows": 10, "change_action": "reset", "n_perm": 20}, "targets": ["x0"], "features": ["x0", "x1"], "clock": "t", "gap_cap": 50.0}"#,
        r#"{"name": "components", "model": {"type": "ew_cov", "pca": 1, "max_rows_between_pcas": 5, "window_size": 12.0, "snapshot_every": 3.0}, "targets": ["x0"], "features": ["x0", "x1"], "clock": "t", "half_life": 20.0, "gap_cap": 50.0}"#,
    ]
    .iter()
    .map(|text| serde_json::from_str(text).unwrap_or_else(|e| panic!("{text}: {e}")))
    .collect()
}

/// The spec of the third bank file, `bank_autocorr`: the residuals' EW
/// autocorrelation at a lag of 3, so that the values it keeps for the lag
/// are frozen non-empty (review round 9 of tasks 270 and 271, R3-3). In a
/// file of its own: a spec added to `bank_diagnosed` would change the
/// stream the kept schemas' `bank_diagnosed` files were written from, and
/// they could no longer convert to the current bytes; and that file is
/// near the 250 KB cap.
fn autocorr_specs() -> Vec<Spec> {
    [
        r#"{"name": "autocorrelated", "model": {"type": "ewridge"}, "targets": ["y"], "features": ["x0", "x1"], "clock": "t", "half_life": 10.0, "gap_cap": 50.0, "emit_autocorr": true, "resid_autocorr_lag": 3}"#,
    ]
    .iter()
    .map(|text| serde_json::from_str(text).unwrap_or_else(|e| panic!("{text}: {e}")))
    .collect()
}

/// The spec whose closed groups the continuation drains.
const CLOSING: usize = 2;

/// Rows of the bank's stream, and the row it is saved at.
const N: usize = 120;
const SPLIT: usize = 80;

/// A seeded generator in `[-1, 1)`, integer arithmetic alone.
fn lcg(seed: u64) -> impl FnMut() -> f64 {
    let mut s = seed;
    move || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }
}

/// Nanoseconds since the Unix epoch of 2024-01-01T00:00:00.
const T0_NS: i64 = 1_704_067_200_000_000_000;

/// Features `x0`, `x1` (null now and then), a target `y`, weights `w` with
/// zeros among them, a drifting level `mid`, a number clock `t` in quarter
/// steps, the same instants as a `Datetime` `ts`, and a group key `g` that
/// only grows, blocks of 25. Beside them, for the integer clock (task 200):
/// the same clock in whole units, `ti = 4t` as `Int64`; a session `s` of 20
/// rows, and `tu` as `UInt32`, `ti` since the session began, so it steps
/// back at each session change; `xi`, `x1`'s value null on the last row of
/// each session, so the row before the save ([`SPLIT`]) is skipped; and a
/// running count `cum`, an integer input to `increment`. None of them
/// draws on the generator, so the other columns are what they were.
fn frame() -> DataFrame {
    let mut r = lcg(198);
    let (mut t, mut ts, mut g, mut x0, mut x1) = (vec![], vec![], vec![], vec![], vec![]);
    let (mut y, mut w, mut mid) = (vec![], vec![], vec![]);
    let (mut ti, mut tu, mut s, mut xi, mut cum) = (vec![], vec![], vec![], vec![], vec![]);
    let (mut clock, mut level) = (0.0f64, 100.0f64);
    let (mut session_start, mut count) = (0i64, 0i64);
    for i in 0..N {
        if i > 0 {
            clock += 0.25 * (1.0 + (4.0 * (r() + 1.0)).floor());
        }
        t.push(clock);
        ts.push(T0_NS + (clock * 1e9) as i64);
        g.push((i / 25) as i64);
        let (a, b) = (r(), r());
        x0.push((i % 17 != 5).then_some(a));
        x1.push((i % 23 != 9).then_some(b));
        y.push(2.0 * a - b + 0.1 * r());
        w.push(if i % 13 == 6 { 0.0 } else { 1.0 + 0.5 * r() });
        level += 0.1 * r();
        mid.push(level);
        let whole = (clock * 4.0) as i64;
        if i % 20 == 0 {
            session_start = whole;
        }
        ti.push(whole);
        tu.push((whole - session_start) as u32);
        s.push((i / 20) as i64);
        xi.push((i % 20 != 19).then_some(b));
        count += 1 + (i % 3) as i64;
        cum.push(count);
    }
    assert!(
        xi[SPLIT - 1].is_none(),
        "the row before the save is skipped, so the pending ticks hold its step"
    );
    let ts = Series::new("ts".into(), ts)
        .cast(&DataType::Datetime(TimeUnit::Nanoseconds, None))
        .unwrap();
    df!("t" => t, "ts" => ts, "g" => g, "x0" => x0, "x1" => x1, "y" => y, "w" => w, "mid" => mid,
        "ti" => ti, "tu" => tu, "s" => s, "xi" => xi, "cum" => cum)
    .unwrap()
}

/// Three series ticking at their own pace on a number clock, long form.
fn ticks() -> DataFrame {
    let mut r = lcg(11);
    let (mut series, mut t, mut v) = (vec![], vec![], vec![]);
    let mut clock = 0.0;
    for _ in 0..120 {
        clock += 0.5 * (1.0 + (2.0 * (r() + 1.0)).floor());
        let pick = r();
        series.push(if pick < -0.2 {
            "a"
        } else if pick < 0.5 {
            "b"
        } else {
            "c"
        });
        t.push(clock);
        v.push(100.0 + r());
    }
    df!("series" => series, "t" => t, "v" => v).unwrap()
}

/// The same ticks on an integer clock: `t` doubled, as `Int64` (task 200).
fn ticks_int() -> DataFrame {
    let mut df = ticks();
    let t = (df.column("t").unwrap().as_materialized_series() * 2.0)
        .cast(&DataType::Int64)
        .unwrap();
    df.with_column(t.into()).unwrap();
    df
}

fn refresh_cols() -> RefreshCols<'static> {
    RefreshCols {
        series: "series",
        clock: "t",
        value: "v",
        group: None,
        keep: &[],
    }
}

fn refresh_names() -> Vec<String> {
    ["a", "b", "c"].map(str::to_string).to_vec()
}

/// A backward mean and a mean that looks ahead, so rows are held across the
/// save; a standard deviation on the mean's kernel, which makes its queue a
/// variance's, six wide, and a biased variance with no window, whose running
/// sums are six wide too (task 212).
fn windows_config() -> WindowsConfig {
    serde_json::from_str(
        r#"{"formulas": [
               {"name": "level", "tree": ["ewm_mean", ["col", "mid"], {"half_life": 5.0, "window_size": 10.0}]},
               {"name": "ahead", "tree": ["-", ["rewm_mean", ["col", "mid"], {"half_life": 5.0, "window_size": 10.0}], ["col", "mid"]]},
               {"name": "spread", "tree": ["ewm_std", ["col", "mid"], {"half_life": 5.0, "window_size": 10.0}]},
               {"name": "var", "tree": ["ewm_var", ["col", "mid"], {"half_life": 5.0, "bias": true}]}
             ],
             "clock": "t", "gap_cap": 50.0}"#,
    )
    .unwrap()
}

fn window_input(df: &DataFrame) -> DataFrame {
    df.select(["t", "mid"]).unwrap()
}

/// The quantile's three sketches (docs/PLAN.md task 272 (d), windows state
/// 10): a running one with no window, a windowed one, whose `front` rows
/// keep what their buckets held before them, and one looking ahead, so rows
/// are held across the save; each with its level.
fn windows_quantile_config() -> WindowsConfig {
    serde_json::from_str(
        r#"{"formulas": [
               {"name": "median", "tree": ["ewm_quantile", ["col", "mid"], {"half_life": 5.0, "quantile": 0.5}]},
               {"name": "high", "tree": ["ewm_quantile", ["abs", ["col", "mid"]], {"half_life": 5.0, "window_size": 10.0, "quantile": 0.9}]},
               {"name": "ahead", "tree": ["rewm_quantile", ["col", "mid"], {"half_life": 5.0, "window_size": 10.0, "quantile": 0.25}]}
             ],
             "clock": "t", "gap_cap": 50.0}"#,
    )
    .unwrap()
}

/// The moments operator (task 554): a skew under a window, whose model keeps
/// its ring of rows, and an autocorrelation at a lag of the clock, whose
/// model keeps its lag buffer, each a group's model in the state.
fn windows_moments_config() -> WindowsConfig {
    serde_json::from_str(
        r#"{"formulas": [
               {"name": "skew", "tree": ["ewm_moments", ["legs", ["col", "mid"]], {"half_life": 5.0, "window_size": 10.0, "powers": [3], "lags": [0.0], "kind": "standardized", "bias": true}]},
               {"name": "ac", "tree": ["ewm_moments", ["legs", ["col", "mid"], ["col", "mid"]], {"half_life": 5.0, "powers": [1, 1], "lags": [0.0, 2.0], "kind": "standardized", "bias": true}]}
             ],
             "clock": "t", "gap_cap": 50.0}"#,
    )
    .unwrap()
}

/// The extrema (task 561, windows state 13): a windowed maximum and a
/// minimum under `"left"`, a maximum looking ahead, so rows are held across
/// the save, a running maximum since an event, and a windowed minimum since
/// one with its own `max_state_bytes`.
fn windows_extrema_config() -> WindowsConfig {
    serde_json::from_str(
        r#"{"formulas": [
               {"name": "hi", "tree": ["rolling_max", ["col", "mid"], {"window_size": 10.0}]},
               {"name": "lo", "tree": ["rolling_min", ["col", "mid"], {"window_size": 10.0, "closed": "left"}]},
               {"name": "ahead", "tree": ["rrolling_max", ["col", "mid"], {"window_size": 10.0}]},
               {"name": "peak", "tree": ["-", ["col", "mid"], ["rolling_max", ["col", "mid"], {"since": [">", ["col", "mid"], ["lit", 100.3]]}]]},
               {"name": "low", "tree": ["rolling_min", ["col", "mid"], {"window_size": 10.0, "since": [">", ["col", "mid"], ["lit", 100.3]], "max_state_bytes": 1000000}]}
             ],
             "clock": "t", "gap_cap": 50.0}"#,
    )
    .unwrap()
}

/// The row operators of task 564 (windows state 14): a run length exact
/// and under a `tol`, a clock lag and drift whose sums are open across the
/// save, a clock step's mean, corrected spread and count of repeats, beside
/// an increment and a window looking ahead, so rows are held. The arrival
/// is the clock, and the event a lag behind it that moves.
fn windows_rows_config() -> WindowsConfig {
    serde_json::from_str(
        r#"{"formulas": [
               {"name": "held", "tree": ["run_length", [">", ["col", "mid"], ["lit", 100.0]]]},
               {"name": "near", "tree": ["run_length", ["col", "mid"], {"abs_tol": 0.5}]},
               {"name": "lag", "tree": ["clock_lag", ["col", "t"], {"event": ["-", ["col", "t"], ["*", ["abs", ["col", "mid"]], ["lit", 0.001]]], "half_life": 5.0}]},
               {"name": "drift", "tree": ["clock_drift", ["col", "t"], {"event": ["-", ["col", "t"], ["*", ["abs", ["col", "mid"]], ["lit", 0.001]]], "half_life": 5.0}]},
               {"name": "step", "tree": ["increment", ["col", "mid"]]},
               {"name": "cadence", "tree": ["clock_step", ["col", "t"], {"half_life": 5.0, "stat": "mean"}]},
               {"name": "jitter", "tree": ["clock_step", ["col", "t"], {"half_life": 5.0, "stat": "std", "bias": false}]},
               {"name": "repeats", "tree": ["clock_step", ["col", "mid"], {"half_life": 5.0, "stat": "zeros"}]},
               {"name": "ahead", "tree": ["rewm_mean", ["col", "mid"], {"half_life": 5.0, "window_size": 10.0}]}
             ],
             "clock": "t", "gap_cap": 50.0}"#,
    )
    .unwrap()
}

/// The invert node (PLAN 563, windows state 14): a root of the row beside
/// a mean looking ahead, so rows are held across the save, and a mean of
/// the roots, every parameter of the node written.
fn windows_invert_config() -> WindowsConfig {
    serde_json::from_str(
        r#"{"formulas": [
               {"name": "ahead", "tree": ["rewm_mean", ["col", "mid"], {"half_life": 5.0, "window_size": 10.0}]},
               {"name": "root", "tree": ["invert", ["+", ["**", ["unknown"], ["lit", 3.0]], ["unknown"]], ["col", "mid"], ["lit", -10.0], ["lit", 10.0], {"increasing": true, "xtol": 1e-10, "rtol": 1e-12, "detail": false}]},
               {"name": "smooth", "tree": ["ewm_mean", ["invert", ["-", ["exp", ["unknown"]], ["lit", 1.0]], ["col", "mid"], ["lit", 0.0], ["lit", 10.0], {"increasing": true}], {"half_life": 5.0}]}
             ],
             "clock": "t", "gap_cap": 50.0}"#,
    )
    .unwrap()
}

/// The same two means on the `Int64` clock, under a fractional `gap_cap`,
/// and an `increment` of an integer input (task 200; review round 5, B1):
/// the rows' `off` in the integer form, and the increment's previous value
/// kept as an integer.
fn windows_int_config() -> WindowsConfig {
    serde_json::from_str(
        r#"{"formulas": [
               {"name": "level", "tree": ["ewm_mean", ["col", "mid"], {"half_life": 20.0, "window_size": 40.0}]},
               {"name": "ahead", "tree": ["-", ["rewm_mean", ["col", "mid"], {"half_life": 20.0, "window_size": 40.0}], ["col", "mid"]]},
               {"name": "step", "tree": ["increment", ["col", "cum"]]}
             ],
             "clock": "ti", "gap_cap": 200.5}"#,
    )
    .unwrap()
}

/// Two periodic profiles (PLAN 560) on a zoned clock: a weighted mean read
/// with the stamp's rows, so rows are held across the save, and a biased
/// variance read before them, from an offset; each group's buckets.
fn windows_profile_config() -> WindowsConfig {
    serde_json::from_str(
        r#"{"formulas": [
               {"name": "level", "tree": ["ewm_profile", ["col", "mid"], {"period": "10s", "buckets": 5, "half_life": "20s", "weight": ["col", "w"]}]},
               {"name": "spread", "tree": ["ewm_profile", ["col", "mid"], {"period": "10s", "buckets": 5, "half_life": "20s", "stat": "var", "bias": true, "closed": "left", "offset": "3s"}]},
               {"name": "median", "tree": ["ewm_profile", ["col", "mid"], {"period": "4s", "buckets": 2, "half_life": "20s", "stat": "quantile", "quantile": 0.5, "window_size": "5s", "weight": ["col", "w"]}]},
               {"name": "high", "tree": ["ewm_profile", ["col", "mid"], {"period": "10s", "buckets": 5, "half_life": "20s", "stat": "quantile", "quantile": 0.9}]}
             ],
             "clock": "ts", "gap_cap": "50s", "group": "g"}"#,
    )
    .unwrap()
}

/// Two tick signs and an increment. The first signs `mid` in
/// tenths, an integer input whose step is often 0, so the sign carried over
/// a step of 0 is read across the save; the second signs `mid` itself inside
/// a decayed sum, the signed flow. Each tick sign's carried sign rides in
/// its group's `prev` past the increments' previous values.
fn windows_tick_config() -> WindowsConfig {
    serde_json::from_str(
        r#"{"formulas": [
               {"name": "sign", "tree": ["tick_sign", ["cast", ["*", ["col", "mid"], ["lit", 10.0]], "Int64"]]},
               {"name": "flow", "tree": ["ewm_sum", ["*", ["tick_sign", ["col", "mid"]], ["col", "cum"]], {"half_life": 5.0, "window_size": 10.0}]},
               {"name": "step", "tree": ["increment", ["col", "cum"]]}
             ],
             "clock": "t", "gap_cap": 50.0}"#,
    )
    .unwrap()
}

/// The clock in Amsterdam's zone, the same instants.
fn window_profile_input(df: &DataFrame) -> DataFrame {
    let zone = TimeZone::opt_try_new(Some("Europe/Amsterdam")).unwrap();
    let ts = df
        .column("ts")
        .unwrap()
        .to_physical_repr()
        .as_materialized_series()
        .i64()
        .unwrap()
        .clone()
        .into_datetime(TimeUnit::Nanoseconds, zone.clone())
        .into_column();
    assert_eq!(
        ts.dtype(),
        &DataType::Datetime(TimeUnit::Nanoseconds, zone),
        "a zoned clock"
    );
    let mut out = df.select(["ts", "g", "mid", "w"]).unwrap();
    out.with_column(ts).unwrap();
    out
}

/// The same profiles' forms on a number clock: a period in clock units.
fn windows_profile_units_config() -> WindowsConfig {
    serde_json::from_str(
        r#"{"formulas": [
               {"name": "level", "tree": ["ewm_profile", ["col", "mid"], {"period": 6.0, "buckets": 3, "half_life": 12.0, "weight": ["col", "w"], "stat": "std"}]},
               {"name": "recent", "tree": ["ewm_profile", ["col", "mid"], {"period": 6.0, "buckets": 3, "half_life": 12.0, "window_size": 20.0, "closed": "both", "min_samples": 2}]},
               {"name": "shape", "tree": ["ewm_profile", ["col", "mid"], {"period": 6.0, "harmonics": 1, "half_life": 12.0, "window_size": 4.0, "stat": "var", "weight": ["col", "w"]}]}
             ],
             "clock": "t", "gap_cap": 50.0, "group": "g"}"#,
    )
    .unwrap()
}

fn window_tick_input(df: &DataFrame) -> DataFrame {
    df.select(["t", "mid", "cum"]).unwrap()
}

fn window_int_input(df: &DataFrame) -> DataFrame {
    df.select(["ti", "mid", "cum"]).unwrap()
}

// --- the three kinds --------------------------------------------------------

/// A state loaded and continued: the bytes it saved before it went on, and
/// the frames the input gave, named; or why it did not load.
type Ran = Result<(Vec<u8>, Vec<(String, DataFrame)>), String>;

/// What a kind's state is: how it is made, loaded, saved and continued.
struct Kind {
    name: &'static str,
    /// The state after the first part, and the input of the continuation.
    first: fn() -> (Vec<u8>, DataFrame),
    /// The state loaded from bytes, continued on the input: the frames it
    /// gave, named, and the bytes the loaded state saved before it went on.
    run: fn(&[u8], &DataFrame) -> Ran,
}

fn bank_first() -> (Vec<u8>, DataFrame) {
    let df = frame();
    let mut bank = kept_rules_bank(specs(), &df);
    bank.fit_predict(&df.slice(0, SPLIT)).unwrap();
    // A group closed before the save is drained, and the next is open
    // across it.
    let drained = bank.closed_groups(Some(CLOSING), true).unwrap();
    assert!(drained.height() > 0, "a group closed before the save");
    (
        kept_throughout(bank.save_bytes().unwrap()),
        df.slice(SPLIT as i64, N - SPLIT),
    )
}

fn bank_run(bytes: &[u8], input: &DataFrame) -> Ran {
    let mut bank = Bank::load_bytes(bytes, Some(&specs()))?;
    let again = bank.save_bytes()?;
    let cols = bank.fit_predict(input).map_err(|e| e.to_string())?;
    let out = DataFrame::new(input.height(), cols).map_err(|e| e.to_string())?;
    let closed = bank.closed_groups(Some(CLOSING), false)?;
    Ok((
        again,
        vec![
            ("fit_predict".into(), out),
            ("closed_groups".into(), closed),
        ],
    ))
}

fn diagnosed_first() -> (Vec<u8>, DataFrame) {
    let df = frame();
    let mut fresh = Bank::new(diagnosed_specs()).unwrap();
    // An empty chunk makes each stream, its models unfed.
    fresh.fit_predict(&df.slice(0, 0)).unwrap();
    let mut bank = Bank::load_bytes(
        &kept_rules(&kept_quantile(&fresh.save_bytes().unwrap())),
        Some(&diagnosed_specs()),
    )
    .unwrap();
    bank.fit_predict(&df.slice(0, SPLIT)).unwrap();
    (
        kept_throughout(bank.save_bytes().unwrap()),
        df.slice(SPLIT as i64, N - SPLIT),
    )
}

/// A bank of `specs` whose streams are in the form a state written before
/// schema 64 holds, set before its first row ([`kept_rules`]): an empty
/// chunk makes each ungrouped stream, its models unfed.
fn kept_rules_bank(specs: Vec<Spec>, df: &DataFrame) -> Bank {
    let mut fresh = Bank::new(specs.clone()).unwrap();
    fresh.fit_predict(&df.slice(0, 0)).unwrap();
    Bank::load_bytes(&kept_rules(&fresh.save_bytes().unwrap()), Some(&specs)).unwrap()
}

/// The residual spread and autocorrelation's rules in the form a state
/// written before schema 64 holds (task 278, batch 17): the spread folding
/// only the residuals the bank reports, the autocorrelation weighing each
/// residual 1 -- the two flags a spec sets, `ungated_spread` and
/// `weighted_autocorr`, left out, as such a state leaves them; and the drift
/// detector's and `ewridge`'s `min_weight` rules (task 278), `ungated_drift`
/// and `min_weight_output_only`, likewise. So the kept
/// fixtures of these kinds, which load in that form, go on to this
/// fixture's numbers and convert to its bytes; the form since is frozen by
/// `bank_gated`.
fn kept_rules(bytes: &[u8]) -> Vec<u8> {
    use rmpv::Value;
    fn walk(v: &mut Value, found: &mut usize) {
        match v {
            Value::Map(m) => {
                let before = m.len();
                m.retain(|(k, _)| {
                    !matches!(
                        k.as_str(),
                        Some(
                            "ungated_spread"
                                | "weighted_autocorr"
                                | "ungated_drift"
                                | "min_weight_output_only"
                        )
                    )
                });
                *found += before - m.len();
                m.iter_mut().for_each(|(_, v)| walk(v, found));
            }
            Value::Array(a) => a.iter_mut().for_each(|v| walk(v, found)),
            _ => {}
        }
    }
    let mut v = rmpv::decode::read_value(&mut &bytes[..]).unwrap();
    let mut found = 0;
    walk(&mut v, &mut found);
    assert!(
        found > 0,
        "a stream that forms a residual sets the spread's flag"
    );
    let mut out = Vec::new();
    rmpv::encode::write_value(&mut out, &v).unwrap();
    // Edited on purpose, so sealed again (schema 71's checksum).
    rheon_polars::reseal(&out)
}

/// `bytes`, held to every stream having stayed in the kept form: no stream
/// a later row made sets either flag ([`kept_rules`]).
fn kept_throughout(bytes: Vec<u8>) -> Vec<u8> {
    let mut all = Vec::new();
    leaves(
        &rmpv::decode::read_value(&mut &bytes[..]).unwrap(),
        "",
        &mut all,
    );
    let set: Vec<&String> = all
        .iter()
        .filter(|(p, v)| {
            [
                ".ungated_spread",
                ".weighted_autocorr",
                ".ungated_drift",
                ".min_weight_output_only",
            ]
            .iter()
            .any(|flag| p.ends_with(flag))
                && v.as_bool() == Some(true)
        })
        .map(|(p, _)| p)
        .collect();
    assert!(set.is_empty(), "streams in the current form: {set:?}");
    bytes
}

/// `bank_diagnosed`'s quantile fit in the form a state written before
/// schema 64 holds, set before its first row: its band drawn in the RMS
/// and `min_weight` holding its learning back (docs/PLAN.md task 278, D-Q
/// and D10), where a spec builds the MAD and a gate on the output alone,
/// its refit's buffers keeping no `|e|`, as the RMS's do.
/// So the kept fixtures of this kind, which load in that form, go on to
/// this fixture's numbers and convert to its bytes; the model's form since
/// is frozen by the core's `quantile_mad` fixture.
fn kept_quantile(bytes: &[u8]) -> Vec<u8> {
    use rmpv::Value;
    fn key<'a>(m: &'a mut [(Value, Value)], k: &str) -> Option<&'a mut Value> {
        m.iter_mut()
            .find(|(n, _)| n.as_str() == Some(k))
            .map(|(_, v)| v)
    }
    fn walk(v: &mut Value, found: &mut usize) {
        match v {
            Value::Map(m) => {
                if let Some(Value::Map(robust)) = key(m, "Robust") {
                    let Some(Value::Map(cfg)) = key(robust, "cfg") else {
                        panic!("a robust state without its cfg")
                    };
                    *key(cfg, "huber_scale").expect("huber_scale") = Value::from("rms");
                    *key(cfg, "min_weight_holds_learning").expect("the gate's form") =
                        Value::from(true);
                    *key(robust, "mad").expect("mad") = Value::Array(Vec::new());
                    // The RMS's buffers keep no `|e|` (task 323).
                    if let Some(Value::Array(bufs)) = key(robust, "refit") {
                        for b in bufs {
                            if let Value::Map(b) = b
                                && let Some(e) = key(b, "resid")
                            {
                                *e = Value::Nil;
                            }
                        }
                    }
                    *key(robust, "seen").expect("the floor's counts") = Value::Array(Vec::new());
                    *found += 1;
                    return;
                }
                m.iter_mut().for_each(|(_, v)| walk(v, found));
            }
            Value::Array(a) => a.iter_mut().for_each(|v| walk(v, found)),
            _ => {}
        }
    }
    let mut v = rmpv::decode::read_value(&mut &bytes[..]).unwrap();
    let mut found = 0;
    walk(&mut v, &mut found);
    assert_eq!(found, 1, "bank_diagnosed holds one quantile fit");
    let mut out = Vec::new();
    rmpv::encode::write_value(&mut out, &v).unwrap();
    // Edited on purpose, so sealed again (schema 71's checksum).
    rheon_polars::reseal(&out)
}

fn diagnosed_run(bytes: &[u8], input: &DataFrame) -> Ran {
    let mut bank = Bank::load_bytes(bytes, Some(&diagnosed_specs()))?;
    let again = bank.save_bytes()?;
    let cols = bank.fit_predict(input).map_err(|e| e.to_string())?;
    let out = DataFrame::new(input.height(), cols).map_err(|e| e.to_string())?;
    Ok((again, vec![("fit_predict".into(), out)]))
}

fn autocorr_first() -> (Vec<u8>, DataFrame) {
    let df = frame();
    let mut bank = kept_rules_bank(autocorr_specs(), &df);
    bank.fit_predict(&df.slice(0, SPLIT)).unwrap();
    (
        kept_throughout(bank.save_bytes().unwrap()),
        df.slice(SPLIT as i64, N - SPLIT),
    )
}

/// The spec of the sixth bank file, `bank_gated`: the residual spread and
/// autocorrelation in the form a spec builds (task 278, batch 17, schema
/// 64). An `ewridge` over a grid of two half-lives under a `min_weight` of
/// 18, which the half-life of 8 never meets on this stream (its weight
/// tops out near 16) and the half-life of 40 meets on row 23, with uneven
/// weights, an `embargo` holding rows across the save -- each held row's
/// record carrying the slot's own prediction the gate withheld, which the
/// release folds into the spread -- and the selection, the average, `sigma`
/// and the weighted autocorrelation that read it.
fn gated_specs() -> Vec<Spec> {
    [
        r#"{"name": "gated", "model": {"type": "ewridge"}, "targets": ["y"], "features": ["x0", "x1"], "clock": "t", "half_life": [8.0, 40.0], "gap_cap": 50.0, "embargo": 2.0, "weight": "w", "min_weight": 18.0, "emit_selected": true, "emit_averaged": true, "emit_sigma": true, "emit_autocorr": true, "resid_autocorr_lag": 2}"#,
    ]
    .iter()
    .map(|text| serde_json::from_str(text).unwrap_or_else(|e| panic!("{text}: {e}")))
    .collect()
}

fn gated_first() -> (Vec<u8>, DataFrame) {
    let df = frame();
    let mut bank = Bank::new(gated_specs()).unwrap();
    bank.fit_predict(&df.slice(0, SPLIT)).unwrap();
    (
        bank.save_bytes().unwrap(),
        df.slice(SPLIT as i64, N - SPLIT),
    )
}

fn gated_run(bytes: &[u8], input: &DataFrame) -> Ran {
    let mut bank = Bank::load_bytes(bytes, Some(&gated_specs()))?;
    let again = bank.save_bytes()?;
    let cols = bank.fit_predict(input).map_err(|e| e.to_string())?;
    let out = DataFrame::new(input.height(), cols).map_err(|e| e.to_string())?;
    Ok((again, vec![("fit_predict".into(), out)]))
}

fn autocorr_run(bytes: &[u8], input: &DataFrame) -> Ran {
    let mut bank = Bank::load_bytes(bytes, Some(&autocorr_specs()))?;
    let again = bank.save_bytes()?;
    let cols = bank.fit_predict(input).map_err(|e| e.to_string())?;
    let out = DataFrame::new(input.height(), cols).map_err(|e| e.to_string())?;
    Ok((again, vec![("fit_predict".into(), out)]))
}

/// The spec of the fourth bank file, `bank_reset`: `rls` under
/// `drift_action = "reset"`, whose prior a step on the row after a reset
/// would decay (docs/PLAN.md tasks 278 and 277).
fn reset_specs() -> Vec<Spec> {
    [
        r#"{"name": "reset", "model": {"type": "rls"}, "targets": ["y"], "features": ["x0", "x1"], "clock": "t", "gap_cap": 10.0, "half_life": 100.0, "min_weight": 10.0, "emit_drift": true, "drift_threshold": 20.0, "drift_action": "reset"}"#,
    ]
    .iter()
    .map(|text| serde_json::from_str(text).unwrap_or_else(|e| panic!("{text}: {e}")))
    .collect()
}

/// Rows of `bank_reset`'s stream: at its middle both slopes flip and the
/// level moves by 6 against noise of 0.1, which fires the detector.
const RESET_N: usize = 300;

fn reset_frame() -> DataFrame {
    let mut r = lcg(7);
    let (mut t, mut x0, mut x1, mut y) = (vec![], vec![], vec![], vec![]);
    let mut clock = 0.0;
    for i in 0..RESET_N {
        clock += 1.0 + 0.5 * r();
        let (a, b) = (r(), r());
        let (sign, level) = if i < RESET_N / 2 {
            (1.0, 0.3)
        } else {
            (-1.0, 6.3)
        };
        t.push(clock);
        x0.push(a);
        x1.push(b);
        y.push(level + sign * (2.0 * a - 1.5 * b) + 0.1 * r());
    }
    df!("t" => t, "x0" => x0, "x1" => x1, "y" => y).unwrap()
}

/// Saved on the row that fired: the rebuilt model has learned no row, and
/// the instance's restart mark is set, so the continuation's first row
/// reads a step of 0 from the state (schema 64, `Persisted::restarted`).
fn reset_first() -> (Vec<u8>, DataFrame) {
    let df = reset_frame();
    let mut probe = Bank::new(reset_specs()).unwrap();
    let cols = probe.fit_predict(&df).unwrap();
    let out = DataFrame::new(df.height(), cols).unwrap();
    let drift = out
        .column("reset")
        .unwrap()
        .struct_()
        .unwrap()
        .field_by_name("drift_y")
        .unwrap();
    let fired = (0..df.height())
        .find(|&i| drift.bool().unwrap().get(i) == Some(true))
        .expect("the detector fires");
    assert!(fired + 20 < RESET_N, "rows after the reset to continue on");
    let mut bank = Bank::new(reset_specs()).unwrap();
    bank.fit_predict(&df.slice(0, fired + 1)).unwrap();
    (bank.save_bytes().unwrap(), df.slice((fired + 1) as i64, 20))
}

/// The specs of `bank_drift_kinds` (PLAN 508): each drift detector other
/// than Page-Hinkley, which `bank_reset` and the others hold, on
/// `bank_reset`'s stream, whose break at its middle each flags. Each kind
/// is written under its name (`rheon_core::DriftDetector`): a CUSUM's sums,
/// an ADWIN's window of several levels, a DDM's and an EDDM's counts behind
/// their error cut, and a KSWIN's ring and generator, kept after a flag.
fn drift_kinds_specs() -> Vec<Spec> {
    [
        r#"{"name": "cusum", "model": {"type": "rls"}, "targets": ["y"], "features": ["x0", "x1"], "clock": "t", "gap_cap": 10.0, "half_life": 100.0, "emit_sigma": true, "emit_drift": true, "drift_kind": "cusum", "drift_delta": 0.25, "drift_threshold": 3.0}"#,
        r#"{"name": "adwin", "model": {"type": "rls"}, "targets": ["y"], "features": ["x0", "x1"], "half_life": 100.0, "emit_drift": true, "drift_kind": "adwin", "drift_action": "reset", "drift_options": {"clock": 2, "max_buckets": 2, "max_width": 400}}"#,
        r#"{"name": "ddm", "model": {"type": "rls"}, "targets": ["y"], "features": ["x0", "x1"], "half_life": 100.0, "emit_drift": true, "drift_kind": "ddm", "drift_options": {"error_above": 0.15, "warm_start": 10}}"#,
        r#"{"name": "eddm", "model": {"type": "rls"}, "targets": ["y"], "features": ["x0", "x1"], "half_life": 100.0, "emit_drift": true, "drift_kind": "eddm", "drift_options": {"error_above": 0.15, "warm_start": 5, "beta": 0.95}}"#,
        r#"{"name": "kswin", "model": {"type": "rls"}, "targets": ["y"], "features": ["x0", "x1"], "half_life": 100.0, "emit_drift": true, "drift_kind": "kswin", "drift_options": {"window_size": 40, "stat_size": 12, "alpha": 0.05, "seed": 7, "restart": false}}"#,
    ]
    .iter()
    .map(|text| serde_json::from_str(text).unwrap_or_else(|e| panic!("{text}: {e}")))
    .collect()
}

/// Saved at row 220, after the break each kind has flagged; the next 80
/// rows continue.
fn drift_kinds_first() -> (Vec<u8>, DataFrame) {
    let df = reset_frame();
    let mut bank = Bank::new(drift_kinds_specs()).unwrap();
    let cols = bank.fit_predict(&df.slice(0, 220)).unwrap();
    let out = DataFrame::new(220, cols).unwrap();
    for spec in drift_kinds_specs() {
        let drift = out
            .column(spec.name.as_str())
            .unwrap()
            .struct_()
            .unwrap()
            .field_by_name("drift_y")
            .unwrap();
        assert!(
            drift.bool().unwrap().sum().unwrap_or(0) > 0,
            "{} flags before the save",
            spec.name
        );
    }
    (bank.save_bytes().unwrap(), df.slice(220, RESET_N - 220))
}

fn drift_kinds_run(bytes: &[u8], input: &DataFrame) -> Ran {
    bank_run_of(&drift_kinds_specs(), bytes, input)
}

/// The specs of `bank_drift_ranges` (review 2026-11-23, DRIFT-1): ADWIN's
/// two forms of range other than river's 1, a number given and the
/// window's, whose state writes the range and each bucket's extent.
fn drift_ranges_specs() -> Vec<Spec> {
    [
        r#"{"name": "fixed", "model": {"type": "rls"}, "targets": ["y"], "features": ["x0", "x1"], "half_life": 100.0, "emit_drift": true, "drift_kind": "adwin", "drift_action": "reset", "drift_options": {"clock": 2, "max_buckets": 2, "max_width": 400, "range": 0.5}}"#,
        r#"{"name": "window", "model": {"type": "rls"}, "targets": ["y"], "features": ["x0", "x1"], "half_life": 100.0, "emit_drift": true, "drift_kind": "adwin", "drift_options": {"clock": 2, "max_buckets": 2, "max_width": 400, "range": "window", "restart": false}}"#,
    ]
    .iter()
    .map(|text| serde_json::from_str(text).unwrap_or_else(|e| panic!("{text}: {e}")))
    .collect()
}

/// Saved at row 220, after the break; the next rows continue.
fn drift_ranges_first() -> (Vec<u8>, DataFrame) {
    let df = reset_frame();
    let mut bank = Bank::new(drift_ranges_specs()).unwrap();
    bank.fit_predict(&df.slice(0, 220)).unwrap();
    (bank.save_bytes().unwrap(), df.slice(220, RESET_N - 220))
}

fn drift_ranges_run(bytes: &[u8], input: &DataFrame) -> Ran {
    bank_run_of(&drift_ranges_specs(), bytes, input)
}

/// The specs of `bank_refit_span` (PLAN 538): `glm` and `expectile`, each
/// refitting a target's first rows over a span of the clock, grouped on
/// `bank_refit`'s stream. Saved at its split, inside every group's span,
/// so each buffer holds rows and its first stamp; the continuation closes
/// the spans of `a` and `b`.
fn refit_span_specs() -> Vec<Spec> {
    [
        r#"{"name": "counts", "model": {"type": "glm", "refit_span": 60.0, "max_state_bytes": 200000}, "targets": ["c"], "features": ["x0", "x1"], "clock": "t", "gap_cap": 10.0, "half_life": "inf", "group": "g"}"#,
        r#"{"name": "upper", "model": {"type": "expectile", "expectile": 0.8, "refit_span": 60.0}, "targets": ["y"], "features": ["x0", "x1"], "clock": "t", "gap_cap": 10.0, "half_life": "inf", "group": "g"}"#,
    ]
    .iter()
    .map(|text| serde_json::from_str(text).unwrap_or_else(|e| panic!("{text}: {e}")))
    .collect()
}

/// `bank_refit`'s stream with a count beside its target, `|y|` rounded,
/// for the `glm`.
fn refit_span_frame() -> DataFrame {
    let df = refit_frame();
    let c: Vec<f64> = df
        .column("y")
        .unwrap()
        .f64()
        .unwrap()
        .into_no_null_iter()
        .map(|v| v.abs().round())
        .collect();
    let mut df = df;
    df.with_column(Column::new("c".into(), c)).unwrap();
    df
}

fn refit_span_first() -> (Vec<u8>, DataFrame) {
    let df = refit_span_frame();
    let mut bank = Bank::new(refit_span_specs()).unwrap();
    bank.fit_predict(&df.slice(0, REFIT_SPLIT)).unwrap();
    (
        bank.save_bytes().unwrap(),
        df.slice(REFIT_SPLIT as i64, REFIT_N - REFIT_SPLIT),
    )
}

fn refit_span_run(bytes: &[u8], input: &DataFrame) -> Ran {
    bank_run_of(&refit_span_specs(), bytes, input)
}

/// The specs of `bank_bocpd_hazard` (PLAN 547): `bocpd`'s two hazards,
/// grouped on `bank_refit`'s stream and number clock, the per-row count
/// `hazard_rows` and a number `hazard`, clock units, which a number clock
/// reads as the step's chance of a break before each row.
fn bocpd_hazard_specs() -> Vec<Spec> {
    [
        r#"{"name": "per_row", "model": {"type": "bocpd", "hazard_rows": 30.0, "mean_prior": [0.0, 0.0], "covariance_prior": [1.0]}, "features": ["x0", "x1"], "clock": "t", "gap_cap": 10.0, "group": "g", "min_weight": 0.0}"#,
        r#"{"name": "on_clock", "model": {"type": "bocpd", "hazard": 20.0, "mean_prior": [0.0, 0.0], "covariance_prior": [1.0]}, "features": ["x0", "x1"], "clock": "t", "gap_cap": 10.0, "group": "g", "min_weight": 0.0}"#,
    ]
    .iter()
    .map(|text| serde_json::from_str(text).unwrap_or_else(|e| panic!("{text}: {e}")))
    .collect()
}

fn bocpd_hazard_first() -> (Vec<u8>, DataFrame) {
    let df = refit_frame();
    let mut bank = Bank::new(bocpd_hazard_specs()).unwrap();
    bank.fit_predict(&df.slice(0, REFIT_SPLIT)).unwrap();
    (
        bank.save_bytes().unwrap(),
        df.slice(REFIT_SPLIT as i64, REFIT_N - REFIT_SPLIT),
    )
}

fn bocpd_hazard_run(bytes: &[u8], input: &DataFrame) -> Ran {
    bank_run_of(&bocpd_hazard_specs(), bytes, input)
}

/// The spec of the bank file `bank_target_weight`: `emit_target_weight`
/// (docs/PLAN.md task 568, at schema 72 before its release), which writes
/// the spec's switch and each stream's last row's `target_weight`, both
/// skipped when off. An `ewridge` over two targets, `y2` null on one row in
/// three and on the twenty rows before the save, on a grid of two
/// half-lives with uneven weights, so each instance's last row holds a
/// weight per target and `y2`'s is short of the shared one.
fn target_weight_specs() -> Vec<Spec> {
    [
        r#"{"name": "own_weights", "model": {"type": "ewridge"}, "targets": ["y", "y2"], "features": ["x0", "x1"], "clock": "t", "half_life": [8.0, 40.0], "gap_cap": 50.0, "weight": "w", "min_weight": 6.0, "emit_target_weight": true}"#,
    ]
    .iter()
    .map(|text| serde_json::from_str(text).unwrap_or_else(|e| panic!("{text}: {e}")))
    .collect()
}

/// [`frame`] with a second target, `y2`: `0.5 y + 1`, null on one row in
/// three and on rows 60-79, the last before [`SPLIT`].
fn target_weight_frame() -> DataFrame {
    let mut df = frame();
    let y = df.column("y").unwrap().f64().unwrap().clone();
    let y2: Vec<Option<f64>> = y
        .iter()
        .enumerate()
        .map(|(i, v)| {
            v.filter(|_| i % 3 != 1 && !(60..SPLIT).contains(&i))
                .map(|v| 0.5 * v + 1.0)
        })
        .collect();
    df.with_column(Column::new("y2".into(), y2)).unwrap();
    df
}

fn target_weight_first() -> (Vec<u8>, DataFrame) {
    let df = target_weight_frame();
    let mut bank = Bank::new(target_weight_specs()).unwrap();
    bank.fit_predict(&df.slice(0, SPLIT)).unwrap();
    (
        bank.save_bytes().unwrap(),
        df.slice(SPLIT as i64, N - SPLIT),
    )
}

fn target_weight_run(bytes: &[u8], input: &DataFrame) -> Ran {
    bank_run_of(&target_weight_specs(), bytes, input)
}

/// The `bank_target_weight` fixture freezes the forms `emit_target_weight`
/// writes, each skipped when the switch is off: the spec's switch, and in
/// each stream's last row one weight a target for each instance, `y2`'s
/// below `y`'s (hard rule 5: a test holds the new forms non-empty).
#[test]
fn the_target_weight_fixture_freezes_its_forms() {
    if regenerating() {
        return;
    }
    let f = frozen::ALL
        .iter()
        .find(|f| f.name == "bank_target_weight")
        .expect("the target_weight fixture");
    let v = rmpv::decode::read_value(&mut unhex(f.state).as_slice()).unwrap();
    let mut all = Vec::new();
    leaves(&v, "", &mut all);
    assert!(
        all.iter()
            .any(|(p, v)| p.ends_with(".emit_target_weight") && v.as_bool() == Some(true)),
        "the spec's switch is not in the file"
    );
    let weights: Vec<f64> = all
        .iter()
        .filter(|(p, _)| p.contains(".last_row.target_weight."))
        .map(|(_, v)| v.as_f64().expect("a weight is a number"))
        .collect();
    // Two instances of two targets, laid out instance by instance.
    assert_eq!(weights.len(), 4, "{weights:?}");
    for pair in weights.chunks(2) {
        assert!(pair[1] > 0.0 && pair[1] < pair[0], "{weights:?}");
    }
}

/// The specs of the fifth bank file, `bank_refit`: `huber` and
/// `sgd(loss = "huber")` grouped, each refitting a target's first 20 rows
/// (docs/PLAN.md task 277, schema 64). `huber`'s `refit_budget` holds two
/// groups' buffers, so the third group to appear, and every later one, gives
/// its refit up: the bank keeps the spec's reservation and each such
/// stream's mark.
fn refit_specs() -> Vec<Spec> {
    [
        r#"{"name": "robust", "model": {"type": "huber", "huber_refit_rows": 20, "refit_budget": 0.008}, "targets": ["y"], "features": ["x0", "x1"], "clock": "t", "gap_cap": 10.0, "half_life": 60.0, "group": "g"}"#,
        r#"{"name": "stepped", "model": {"type": "sgd", "loss": "huber", "huber_refit_rows": 20}, "targets": ["y"], "features": ["x0", "x1"], "clock": "t", "gap_cap": 10.0, "half_life": 60.0, "group": "g"}"#,
    ]
    .iter()
    .map(|text| serde_json::from_str(text).unwrap_or_else(|e| panic!("{text}: {e}")))
    .collect()
}

/// Rows of `bank_refit`'s stream: group `a` on every other row, `b` on every
/// fourth, `c` and `d` on every eighth, and `e` in `d`'s place from row 72,
/// so `e` first appears after the split. A wild row, 8 above, on every
/// seventh.
const REFIT_N: usize = 96;
const REFIT_SPLIT: usize = 48;

fn refit_frame() -> DataFrame {
    let mut r = lcg(11);
    let pattern = ["a", "b", "a", "c", "a", "b", "a", "d"];
    let (mut t, mut g, mut x0, mut x1, mut y) = (vec![], vec![], vec![], vec![], vec![]);
    for i in 0..REFIT_N {
        let (a, b) = (r(), r());
        let key = match pattern[i % 8] {
            "d" if i >= 72 => "e",
            k => k,
        };
        t.push(i as f64);
        g.push(key);
        x0.push(a);
        x1.push(b);
        let wild = if i % 7 == 3 { 8.0 } else { 0.0 };
        y.push(1.0 + 2.0 * a - 1.5 * b + 0.1 * r() + wild);
    }
    df!("t" => t, "g" => g, "x0" => x0, "x1" => x1, "y" => y).unwrap()
}

/// Saved at the split: `a` has refit and freed its rows, `b` holds its 12,
/// and `c` and `d` gave their refits up to the budget; `e`, new after the
/// load, gives its up too.
fn refit_first() -> (Vec<u8>, DataFrame) {
    let df = refit_frame();
    let mut bank = Bank::new(refit_specs()).unwrap();
    bank.fit_predict(&df.slice(0, REFIT_SPLIT)).unwrap();
    (
        bank.save_bytes().unwrap(),
        df.slice(REFIT_SPLIT as i64, REFIT_N - REFIT_SPLIT),
    )
}

/// `bank_refit` as the build of schema 64 saved it (`v64/bank_refit.rs`),
/// loaded and saved by this build (task 323): each anchored MAD sketch
/// marked to keep anchoring at its first value, and `b`'s buffer, which
/// holds no `|e|`, read through the sketch it feeds until its refit, which
/// falls in the continuation. The forms a 64 bank takes in a bank this
/// build saves.
fn refit_64_first() -> (Vec<u8>, DataFrame) {
    let kept = previous_sets()
        .into_iter()
        .find(|(s, _)| *s == 64)
        .and_then(|(_, set)| set.iter().find(|f| f.name == "bank_refit"))
        .expect("the kept bank_refit of schema 64");
    let bank = Bank::load_bytes(&unhex(kept.state), Some(&refit_specs())).unwrap();
    (bank.save_bytes().unwrap(), from_ipc(kept.input))
}

fn refit_run(bytes: &[u8], input: &DataFrame) -> Ran {
    let mut bank = Bank::load_bytes(bytes, Some(&refit_specs()))?;
    let again = bank.save_bytes()?;
    let cols = bank.fit_predict(input).map_err(|e| e.to_string())?;
    let out = DataFrame::new(input.height(), cols).map_err(|e| e.to_string())?;
    let summary = bank.summary(0, None)?;
    Ok((
        again,
        vec![("fit_predict".into(), out), ("summary".into(), summary)],
    ))
}

fn reset_run(bytes: &[u8], input: &DataFrame) -> Ran {
    bank_run_of(&reset_specs(), bytes, input)
}

/// A bank of `specs` loaded from `bytes` and fed `input`.
fn bank_run_of(specs: &[Spec], bytes: &[u8], input: &DataFrame) -> Ran {
    let mut bank = Bank::load_bytes(bytes, Some(specs))?;
    let again = bank.save_bytes()?;
    let cols = bank.fit_predict(input).map_err(|e| e.to_string())?;
    let out = DataFrame::new(input.height(), cols).map_err(|e| e.to_string())?;
    Ok((again, vec![("fit_predict".into(), out)]))
}

/// The spec of the seventh bank file, `bank_reset_held`: `bank_reset`'s,
/// under an `embargo`, on a stream with skipped rows (review round 10,
/// S7-1 and S7-3): a held row after a skipped one carries the step it reads
/// as a fresh stream's first, and a chunk that ends on a skipped row keeps
/// the clock a fresh stream begun there would.
fn reset_held_specs() -> Vec<Spec> {
    [
        r#"{"name": "reset", "model": {"type": "rls"}, "targets": ["y"], "features": ["x0", "x1"], "clock": "t", "gap_cap": 10.0, "half_life": 100.0, "min_weight": 10.0, "emit_drift": true, "drift_threshold": 20.0, "drift_action": "reset", "embargo": 3.0, "emit_clocks": true}"#,
    ]
    .iter()
    .map(|text| serde_json::from_str(text).unwrap_or_else(|e| panic!("{text}: {e}")))
    .collect()
}

/// `bank_reset`'s stream with `x0` null on rows 20 and 23 of every 25.
fn reset_held_frame() -> DataFrame {
    let df = reset_frame();
    let x0: Vec<Option<f64>> = df
        .column("x0")
        .unwrap()
        .f64()
        .unwrap()
        .iter()
        .enumerate()
        .map(|(i, v)| v.filter(|_| !matches!(i % 25, 20 | 23)))
        .collect();
    let mut out = df.clone();
    out.with_column(Series::new("x0".into(), x0).into())
        .unwrap();
    out
}

/// Saved on the first skipped row of the 23rd kind after the detection:
/// the rows held then include the one after row 20's skip.
fn reset_held_first() -> (Vec<u8>, DataFrame) {
    let df = reset_held_frame();
    let mut probe = Bank::new(reset_held_specs()).unwrap();
    let cols = probe.fit_predict(&df).unwrap();
    let out = DataFrame::new(df.height(), cols).unwrap();
    let drift = out
        .column("reset")
        .unwrap()
        .struct_()
        .unwrap()
        .field_by_name("drift_y")
        .unwrap();
    let fired = (0..df.height())
        .find(|&i| drift.bool().unwrap().get(i) == Some(true))
        .expect("the detector fires");
    let save = (fired..).find(|i| i % 25 == 23).unwrap();
    assert!(save + 20 < RESET_N, "rows after the save to continue on");
    let mut bank = Bank::new(reset_held_specs()).unwrap();
    bank.fit_predict(&df.slice(0, save + 1)).unwrap();
    (bank.save_bytes().unwrap(), df.slice((save + 1) as i64, 20))
}

fn reset_held_run(bytes: &[u8], input: &DataFrame) -> Ran {
    bank_run_of(&reset_held_specs(), bytes, input)
}

/// The spec of the eighth bank file, `bank_unstable`: an `sgd` at a
/// constant rate past its stability bound on two features (task 278, D8),
/// whose instance has said so (`Notified::unstable`; review round 10,
/// S7-4).
fn unstable_specs() -> Vec<Spec> {
    [
        r#"{"name": "unstable", "model": {"type": "sgd", "learning_rate": 0.5}, "targets": ["y"], "features": ["x0", "x1"], "clock": "t", "gap_cap": 10.0, "half_life": 60.0}"#,
    ]
    .iter()
    .map(|text| serde_json::from_str(text).unwrap_or_else(|e| panic!("{text}: {e}")))
    .collect()
}

fn unstable_first() -> (Vec<u8>, DataFrame) {
    let df = frame();
    let mut bank = Bank::new(unstable_specs()).unwrap();
    bank.fit_predict(&df.slice(0, SPLIT)).unwrap();
    (
        bank.save_bytes().unwrap(),
        df.slice(SPLIT as i64, N - SPLIT),
    )
}

fn unstable_run(bytes: &[u8], input: &DataFrame) -> Ran {
    bank_run_of(&unstable_specs(), bytes, input)
}

/// The spec of the tenth bank file, `bank_notice_pending`: a stream saved
/// holding readiness notices raised and not yet taken (`Notified::pending`,
/// schema 67, task 340). Every call that completes takes them before the
/// bank can be saved, so a stream holds some only after a call that learned
/// rows and then returned an error (an output it could not assemble, which
/// no input reaches: the bank's own hook makes one). Under `group_close =
/// "session"` a notice names its span, so the file holds both forms a
/// pending notice takes: group 1's span `"a"` raised its notice in the
/// call and was closed by span `"b"` in it, so span `"b"`'s stream holds
/// the notice named ` session "a"`; group 2 raised its own in its open
/// span, which is named only when the notice is taken. The continuation's
/// first call says both, in the groups' order, and the frame of what it
/// said is frozen beside its output.
fn notice_specs() -> Vec<Spec> {
    [
        r#"{"name": "noticed", "model": {"type": "ewridge", "ridge": 0.5}, "targets": ["y"], "features": ["x0", "x1", "x2"], "half_life": 4.0, "coef_every": 1, "group": "g", "session": "s", "group_close": "session"}"#,
    ]
    .iter()
    .map(|text| serde_json::from_str(text).unwrap_or_else(|e| panic!("{text}: {e}")))
    .collect()
}

/// Rows `from..from + n` of group `g` in session `s`: under a ridge of 0.5
/// the duplicated feature `x2` reads a support below 0.5, and at a
/// half-life of 4 rows the span is 95% settled from its row 19 (0-based),
/// where the support notice is raised (`tests/notice_paths.rs`).
fn notice_rows(g: i64, s: &str, from: usize, n: usize) -> DataFrame {
    let x0: Vec<f64> = (from..from + n)
        .map(|i| ((i * 7) % 11) as f64 - 5.0)
        .collect();
    let x1: Vec<f64> = (from..from + n)
        .map(|i| ((i * 5) % 9) as f64 - 4.0)
        .collect();
    let y: Vec<f64> = (from..from + n)
        .map(|i| x0[i - from] - x1[i - from] + ((i * 3) % 7) as f64 * 0.1)
        .collect();
    df!("g" => vec![g; n], "s" => vec![s; n], "x0" => &x0, "x1" => &x1, "x2" => &x0, "y" => &y)
        .unwrap()
}

fn stacked(parts: &[DataFrame]) -> DataFrame {
    let mut out = parts[0].clone();
    for p in &parts[1..] {
        out.vstack_mut(p).unwrap();
    }
    out
}

fn notice_first() -> (Vec<u8>, DataFrame) {
    let mut bank = Bank::new(notice_specs()).unwrap();
    bank.fit_predict(&stacked(&[
        notice_rows(1, "a", 0, 19),
        notice_rows(2, "a", 0, 19),
    ]))
    .unwrap();
    assert!(
        bank.take_notices().is_empty(),
        "nothing raised before row 19"
    );
    // Group 1's span "a" raises its notice at its row 19 and is closed by
    // span "b", which carries it; group 2 raises its own; then the call
    // fails, leaving both in their streams.
    bank.fail_next_call_after_learning();
    assert!(
        bank.fit_predict(&stacked(&[
            notice_rows(1, "a", 19, 1),
            notice_rows(1, "b", 0, 2),
            notice_rows(2, "a", 19, 1),
        ]))
        .is_err(),
        "the call fails after learning"
    );
    assert!(
        bank.take_notices().is_empty(),
        "none taken by the failed call"
    );
    (
        bank.save_bytes().unwrap(),
        stacked(&[notice_rows(1, "b", 2, 4), notice_rows(2, "a", 20, 4)]),
    )
}

/// The specs of the eleventh bank file, `bank_label_clock` (schema 67,
/// task 341): rows held for their label times, on an integer clock and on
/// a `Datetime` one. The buffer is ordered by label time, so at the save it
/// holds a row released after one that arrived later; it holds rows past
/// the end of the first run, which the continuation learns; and the
/// counts of labels that never arrived (a null label time) and of rows a
/// full buffer dropped (`held_overflow = "drop_oldest"` at two rows) are
/// both above 0. The continuation's `summary()` is frozen beside its
/// output.
fn label_specs() -> Vec<Spec> {
    [
        r#"{"name": "labelled", "model": {"type": "ewridge"}, "targets": ["y"], "features": ["x0", "x1"], "clock": "t", "half_life": 30.0, "gap_cap": 50.0, "weight": "w", "label_clock": "avail", "max_held_rows": 2, "held_overflow": "drop_oldest", "conformal": 0.9, "emit_metrics": true, "emit_sigma": true}"#,
        r#"{"name": "labelled_ts", "model": {"type": "rls"}, "targets": ["y"], "features": ["x0"], "clock": "ts", "half_life": "1m", "gap_cap": "10m", "label_clock": ["avail_ts"]}"#,
    ]
    .iter()
    .map(|text| serde_json::from_str(text).unwrap_or_else(|e| panic!("{text}: {e}")))
    .collect()
}

const LABEL_N: usize = 60;
const LABEL_SPLIT: usize = 40;

/// Rows two clock units apart, each label time its clock plus a delay of 0
/// to 20 by the row's place in a cycle of eight, null on every ninth row
/// from the fifth; a weight of 0 on every tenth from the eighth.
fn label_frame() -> DataFrame {
    let t: Vec<i64> = (0..LABEL_N as i64).map(|i| 2 * i).collect();
    let delay = |i: usize| [7i64, 1, 11, 3, 0, 20, 5, 2][i % 8];
    let avail: Vec<Option<i64>> = (0..LABEL_N)
        .map(|i| (i % 9 != 4).then(|| t[i] + delay(i)))
        .collect();
    let ts: Vec<i64> = t.iter().map(|v| T0_NS + v * 1_000_000_000).collect();
    let avail_ts: Vec<Option<i64>> = avail
        .iter()
        .map(|a| a.map(|v| T0_NS + v * 1_000_000_000))
        .collect();
    let datetime = |name: &str, v: Vec<Option<i64>>| {
        Series::new(name.into(), v)
            .cast(&DataType::Datetime(TimeUnit::Nanoseconds, None))
            .unwrap()
    };
    let x0: Vec<f64> = (0..LABEL_N).map(|i| ((i * 7) % 11) as f64 - 5.0).collect();
    let x1: Vec<f64> = (0..LABEL_N).map(|i| ((i * 5) % 9) as f64 - 4.0).collect();
    let y: Vec<f64> = (0..LABEL_N)
        .map(|i| 0.5 * x0[i] - x1[i] + ((i * 3) % 7) as f64 * 0.1)
        .collect();
    let w: Vec<f64> = (0..LABEL_N)
        .map(|i| {
            if i % 10 == 7 {
                0.0
            } else {
                1.0 + (i % 3) as f64 * 0.5
            }
        })
        .collect();
    df!("t" => t, "avail" => avail, "ts" => datetime("ts", ts.into_iter().map(Some).collect()),
        "avail_ts" => datetime("avail_ts", avail_ts), "x0" => x0, "x1" => x1, "y" => y, "w" => w)
    .unwrap()
}

fn label_first() -> (Vec<u8>, DataFrame) {
    let df = label_frame();
    let mut bank = Bank::new(label_specs()).unwrap();
    bank.fit_predict(&df.slice(0, LABEL_SPLIT)).unwrap();
    let summary = bank.summary(0, None).unwrap();
    for name in ["labels_missing", "labels_dropped"] {
        let n = summary.column(name).unwrap().i64().unwrap().get(0).unwrap();
        assert!(n > 0, "{name} is {n} at the save");
    }
    (
        bank.save_bytes().unwrap(),
        df.slice(LABEL_SPLIT as i64, LABEL_N - LABEL_SPLIT),
    )
}

fn label_run(bytes: &[u8], input: &DataFrame) -> Ran {
    let mut bank = Bank::load_bytes(bytes, Some(&label_specs()))?;
    let again = bank.save_bytes()?;
    let cols = bank.fit_predict(input).map_err(|e| e.to_string())?;
    let out = DataFrame::new(input.height(), cols).map_err(|e| e.to_string())?;
    let summary = bank.summary(0, None)?;
    Ok((
        again,
        vec![("fit_predict".into(), out), ("summary".into(), summary)],
    ))
}

/// The specs of the twelfth bank file, `bank_conformal_auto` (schema 67,
/// task 342b): a conformal interval under `conformal_rate = "auto"` on rows
/// held for their label times, past its warm start at the save, so its
/// tracker writes the mean of the rows waiting at each release
/// (`held_mean`), which a tracker with nothing waiting does not write; and a
/// second whose predictions a `min_weight` held back until late, so at the
/// save its tracker still waits for its start and writes the count of rows
/// it has been given (`released`, task 368). The rows of
/// `bank_label_clock`, a buffer with no bound.
fn auto_specs() -> Vec<Spec> {
    [
        r#"{"name": "auto", "model": {"type": "ewridge"}, "targets": ["y"], "features": ["x0", "x1"], "clock": "t", "half_life": 30.0, "gap_cap": 50.0, "weight": "w", "label_clock": "avail", "conformal": 0.9, "conformal_rate": "auto", "emit_sigma": true}"#,
        r#"{"name": "waiting", "model": {"type": "ewridge"}, "targets": ["y"], "features": ["x0", "x1"], "clock": "t", "half_life": 30.0, "gap_cap": 50.0, "weight": "w", "label_clock": "avail", "conformal": 0.9, "conformal_rate": "auto", "min_weight": 19.0, "emit_sigma": true}"#,
    ]
    .iter()
    .map(|text| serde_json::from_str(text).unwrap_or_else(|e| panic!("{text}: {e}")))
    .collect()
}

fn auto_first() -> (Vec<u8>, DataFrame) {
    let df = label_frame();
    let mut bank = Bank::new(auto_specs()).unwrap();
    let cols = bank.fit_predict(&df.slice(0, LABEL_SPLIT)).unwrap();
    let out = DataFrame::new(LABEL_SPLIT, cols).unwrap();
    let lo = out.column("auto").unwrap().struct_().unwrap();
    let lo = lo.field_by_name("lo_y").unwrap();
    assert!(
        lo.f64().unwrap().iter().flatten().count() > 10,
        "past the warm start"
    );
    let bytes = bank.save_bytes().unwrap();
    assert!(
        bytes.windows(8).any(|b| b == b"released"),
        "a tracker still waits for its start"
    );
    (
        bank.save_bytes().unwrap(),
        df.slice(LABEL_SPLIT as i64, LABEL_N - LABEL_SPLIT),
    )
}

fn auto_run(bytes: &[u8], input: &DataFrame) -> Ran {
    let mut bank = Bank::load_bytes(bytes, Some(&auto_specs()))?;
    let again = bank.save_bytes()?;
    let cols = bank.fit_predict(input).map_err(|e| e.to_string())?;
    let out = DataFrame::new(input.height(), cols).map_err(|e| e.to_string())?;
    Ok((again, vec![("fit_predict".into(), out)]))
}

fn notice_run(bytes: &[u8], input: &DataFrame) -> Ran {
    let mut bank = Bank::load_bytes(bytes, Some(&notice_specs()))?;
    let again = bank.save_bytes()?;
    let cols = bank.fit_predict(input).map_err(|e| e.to_string())?;
    let out = DataFrame::new(input.height(), cols).map_err(|e| e.to_string())?;
    let said = df!("notice" => bank.take_notices()).map_err(|e| e.to_string())?;
    Ok((
        again,
        vec![("fit_predict".into(), out), ("notices".into(), said)],
    ))
}

fn windows_first_of(config: WindowsConfig, input: DataFrame) -> (Vec<u8>, DataFrame) {
    let mut run = WindowsRun::new(config, input.schema()).unwrap();
    run.feed(&input.slice(0, SPLIT), None).unwrap();
    assert!(run.held() > 0, "rows held across the save");
    (
        run.save_bytes().unwrap(),
        input.slice(SPLIT as i64, N - SPLIT),
    )
}

fn windows_run_of(config: WindowsConfig, bytes: &[u8], input: &DataFrame) -> Ran {
    let mut run = WindowsRun::load_bytes(bytes, config, input.schema())?;
    let again = run.save_bytes()?;
    let fed = run.feed(input, None).map_err(|e| e.to_string())?;
    let finished = run.finish().map_err(|e| e.to_string())?;
    Ok((
        again,
        vec![("feed".into(), fed), ("finish".into(), finished)],
    ))
}

fn windows_first() -> (Vec<u8>, DataFrame) {
    windows_first_of(windows_config(), window_input(&frame()))
}

fn windows_run(bytes: &[u8], input: &DataFrame) -> Ran {
    windows_run_of(windows_config(), bytes, input)
}

fn windows_profile_first() -> (Vec<u8>, DataFrame) {
    windows_first_of(windows_profile_config(), window_profile_input(&frame()))
}

fn windows_profile_run(bytes: &[u8], input: &DataFrame) -> Ran {
    windows_run_of(windows_profile_config(), bytes, input)
}

fn windows_profile_units_first() -> (Vec<u8>, DataFrame) {
    let df = frame().select(["t", "g", "mid", "w"]).unwrap();
    windows_first_of(windows_profile_units_config(), df)
}

fn windows_profile_units_run(bytes: &[u8], input: &DataFrame) -> Ran {
    windows_run_of(windows_profile_units_config(), bytes, input)
}

fn windows_quantile_first() -> (Vec<u8>, DataFrame) {
    windows_first_of(windows_quantile_config(), window_input(&frame()))
}

fn windows_quantile_run(bytes: &[u8], input: &DataFrame) -> Ran {
    windows_run_of(windows_quantile_config(), bytes, input)
}

fn windows_moments_first() -> (Vec<u8>, DataFrame) {
    windows_first_of(windows_moments_config(), window_input(&frame()))
}

fn windows_moments_run(bytes: &[u8], input: &DataFrame) -> Ran {
    windows_run_of(windows_moments_config(), bytes, input)
}

fn windows_extrema_first() -> (Vec<u8>, DataFrame) {
    windows_first_of(windows_extrema_config(), window_input(&frame()))
}

fn windows_extrema_run(bytes: &[u8], input: &DataFrame) -> Ran {
    windows_run_of(windows_extrema_config(), bytes, input)
}

fn windows_tick_first() -> (Vec<u8>, DataFrame) {
    windows_first_of(windows_tick_config(), window_tick_input(&frame()))
}

fn windows_tick_run(bytes: &[u8], input: &DataFrame) -> Ran {
    windows_run_of(windows_tick_config(), bytes, input)
}

fn windows_rows_first() -> (Vec<u8>, DataFrame) {
    windows_first_of(windows_rows_config(), window_input(&frame()))
}

fn windows_rows_run(bytes: &[u8], input: &DataFrame) -> Ran {
    windows_run_of(windows_rows_config(), bytes, input)
}

fn windows_invert_first() -> (Vec<u8>, DataFrame) {
    windows_first_of(windows_invert_config(), window_input(&frame()))
}

fn windows_invert_run(bytes: &[u8], input: &DataFrame) -> Ran {
    windows_run_of(windows_invert_config(), bytes, input)
}

fn windows_int_first() -> (Vec<u8>, DataFrame) {
    windows_first_of(windows_int_config(), window_int_input(&frame()))
}

fn windows_int_run(bytes: &[u8], input: &DataFrame) -> Ran {
    windows_run_of(windows_int_config(), bytes, input)
}

fn refresh_first_of(ticks: DataFrame) -> (Vec<u8>, DataFrame) {
    let half = ticks.height() / 2;
    let mut r = RefreshTime::new(refresh_names(), false).unwrap();
    r.feed(&ticks.slice(0, half), &refresh_cols()).unwrap();
    (
        r.save_bytes().unwrap(),
        ticks.slice(half as i64, ticks.height() - half),
    )
}

fn refresh_run(bytes: &[u8], input: &DataFrame) -> Ran {
    let mut r = RefreshTime::load_bytes(bytes, refresh_names(), false, false)?;
    let again = r.save_bytes()?;
    let fed = r.feed(input, &refresh_cols()).map_err(|e| e.to_string())?;
    Ok((again, vec![("feed".into(), fed)]))
}

fn refresh_first() -> (Vec<u8>, DataFrame) {
    refresh_first_of(ticks())
}

fn refresh_int_first() -> (Vec<u8>, DataFrame) {
    refresh_first_of(ticks_int())
}

/// The ticks with a pair grid's columns: a group, a zoned clock, and a
/// `keep` column of each form a held row's value takes -- a boolean, a
/// signed and an unsigned integer, a float, a string, a null, a date, a
/// duration, a time of day, and a list, which is kept as Arrow IPC (task
/// 321).
fn ticks_held() -> DataFrame {
    let df = ticks();
    let n = df.height();
    let tz = TimeZone::opt_try_new(Some("Europe/Amsterdam")).unwrap();
    let row: Vec<i64> = (0..n as i64).collect();
    let t = (df.column("t").unwrap().as_materialized_series() * 1e6)
        .cast(&DataType::Int64)
        .unwrap()
        .into_datetime(TimeUnit::Microseconds, tz);
    let cols = vec![
        df.column("series").unwrap().clone(),
        t.into_column(),
        df.column("v").unwrap().clone(),
        Column::new(
            "g".into(),
            row.iter()
                .map(|r| ["p", "q"][(r % 2) as usize])
                .collect::<Vec<_>>(),
        ),
        Column::new("i".into(), row.clone()),
        Column::new(
            "flag".into(),
            row.iter().map(|r| r % 3 == 0).collect::<Vec<_>>(),
        ),
        Column::new(
            "u".into(),
            row.iter().map(|r| *r as u32 + 7).collect::<Vec<_>>(),
        ),
        Column::new(
            "x".into(),
            row.iter().map(|r| *r as f64 / 8.0).collect::<Vec<_>>(),
        ),
        Column::new(
            "label".into(),
            row.iter().map(|r| format!("row {r}")).collect::<Vec<_>>(),
        ),
        Column::full_null("none".into(), n, &DataType::Int64),
        Column::new(
            "day".into(),
            row.iter().map(|r| 19_000 + *r as i32).collect::<Vec<_>>(),
        )
        .cast(&DataType::Date)
        .unwrap(),
        Column::new(
            "wait".into(),
            row.iter().map(|r| r * 1_000).collect::<Vec<_>>(),
        )
        .cast(&DataType::Duration(TimeUnit::Milliseconds))
        .unwrap(),
        Column::new(
            "at".into(),
            row.iter().map(|r| r * 1_000_000_000).collect::<Vec<_>>(),
        )
        .cast(&DataType::Time)
        .unwrap(),
        Series::new(
            "list".into(),
            row.iter()
                .map(|r| Series::new("".into(), [*r, -r]))
                .collect::<Vec<_>>(),
        )
        .into_column(),
    ];
    DataFrame::new(n, cols).unwrap()
}

const HELD_KEEP: &[&str] = &[
    "i", "flag", "u", "x", "label", "none", "day", "wait", "at", "list",
];

fn held_keep() -> Vec<String> {
    HELD_KEEP.iter().map(|k| (*k).to_string()).collect()
}

fn held_cols(keep: &[String]) -> RefreshCols<'_> {
    RefreshCols {
        series: "series",
        clock: "t",
        value: "v",
        group: Some("g"),
        keep,
    }
}

/// A pair grid sliced among one tick's points (task 321): the state holds
/// the tick's point the slice left out, and the run loaded from it returns
/// that point first.
fn refresh_held_first() -> (Vec<u8>, DataFrame) {
    let df = ticks_held();
    let keep = held_keep();
    let whole = RefreshTime::new(refresh_names(), true)
        .unwrap()
        .feed(&df, &held_cols(&keep))
        .unwrap();
    let i = whole.column("i").unwrap().i64().unwrap();
    // A slice ending after the first point of a tick past the input's middle.
    let half = (df.height() / 2) as i64;
    let at = (1..i.len())
        .find(|&r| i.get(r) == i.get(r - 1) && i.get(r).unwrap() > half)
        .expect("a tick completing two pairs past the middle");
    let mut r = RefreshTime::new(refresh_names(), true).unwrap();
    r.feed_limited(&df, &held_cols(&keep), Some(at)).unwrap();
    let bytes = r.save_bytes().unwrap();
    let fed = i.get(at).unwrap() as usize + 1;
    (bytes, df.slice(fed as i64, df.height() - fed))
}

fn refresh_held_run(bytes: &[u8], input: &DataFrame) -> Ran {
    let keep = held_keep();
    let mut r = RefreshTime::load_bytes(bytes, refresh_names(), true, true)?;
    let again = r.save_bytes()?;
    let fed = r
        .feed(input, &held_cols(&keep))
        .map_err(|e| e.to_string())?;
    Ok((again, vec![("feed".into(), fed)]))
}

/// A formula target, whose window looks ahead, so each group's window core
/// holds rows across the save; a bank file embeds that core as a window
/// run's state (PLAN 542's form of a bank holding one). Groups by `g`,
/// whose blocks of 25 rows only grow, so a core per group seen.
fn formula_specs() -> Vec<Spec> {
    [
        r#"{"name": "ahead", "model": {"type": "ewridge"}, "features": ["x0"], "group": "g",
            "targets": [{"name": "fwd", "formula": ["-", ["rewm_mean", ["col", "mid"],
                {"half_life": 2.0, "window_size": 4.0}], ["col", "mid"]]}],
            "clock": "t", "gap_cap": 50.0, "half_life": 20.0, "embargo": 4.0}"#,
    ]
    .iter()
    .map(|text| serde_json::from_str(text).unwrap_or_else(|e| panic!("{text}: {e}")))
    .collect()
}

fn formula_input(df: &DataFrame) -> DataFrame {
    df.select(["t", "g", "x0", "mid"]).unwrap()
}

fn formula_first() -> (Vec<u8>, DataFrame) {
    let df = formula_input(&frame());
    let mut bank = Bank::new(formula_specs()).unwrap();
    bank.fit_predict(&df.slice(0, SPLIT)).unwrap();
    (
        bank.save_bytes().unwrap(),
        df.slice(SPLIT as i64, N - SPLIT),
    )
}

fn formula_run(bytes: &[u8], input: &DataFrame) -> Ran {
    bank_run_of(&formula_specs(), bytes, input)
}

// --- rp.stream.merge (PLAN 541) ---------------------------------------------

/// The column of a merge fixture's input that says which stream a row is
/// of: the continuation's streams travel as one frame.
const MERGE_INPUT: &str = "__input";

/// Three streams on a float clock, ties within and across them; `news` has
/// no row before the cut, so its state holds no clock.
fn merge_streams() -> Vec<(&'static str, DataFrame)> {
    let quotes = [0.0, 0.5, 1.0, 1.0, 2.0, 2.5, 3.0, 3.0, 4.0, 5.0, 5.5, 6.0];
    let trades = [0.25, 1.0, 1.0, 2.0, 3.0, 3.5, 4.0, 4.5, 6.0];
    let news = [3.0, 5.0, 6.5];
    vec![
        (
            "quotes",
            df!(
                "t" => quotes,
                "bid" => quotes.iter().map(|t| 100.0 + t / 8.0).collect::<Vec<_>>(),
            )
            .unwrap(),
        ),
        (
            "trades",
            df!(
                "t" => trades,
                "price" => trades.iter().map(|t| 100.1 + t / 16.0).collect::<Vec<_>>(),
                "size" => (1..=trades.len() as i64).collect::<Vec<_>>(),
            )
            .unwrap(),
        ),
        (
            "news",
            df!("t" => news, "headline" => ["open", "halt", "close"]).unwrap(),
        ),
    ]
}

/// Two streams on a session clock (a `Duration` since each session's open)
/// grouped by a symbol, with a null one; each starts again at its second
/// session, a step back past `restart_after_step_back` = 10m.
fn merge_by_streams() -> Vec<(&'static str, DataFrame)> {
    let minutes = |m: &[i64]| {
        Series::new("t".into(), m.iter().map(|m| m * 60_000).collect::<Vec<_>>())
            .cast(&DataType::Duration(TimeUnit::Milliseconds))
            .unwrap()
            .into_column()
    };
    let bids = DataFrame::new(
        7,
        vec![
            Column::new(
                "sym".into(),
                [
                    Some("A"),
                    Some("B"),
                    None,
                    Some("A"),
                    Some("A"),
                    Some("B"),
                    Some("A"),
                ],
            ),
            minutes(&[0, 5, 10, 20, 2, 7, 15]),
            Column::new("bid".into(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]),
        ],
    )
    .unwrap();
    let asks = DataFrame::new(
        5,
        vec![
            Column::new("sym".into(), ["B", "A", "B", "A", "B"]),
            minutes(&[3, 12, 25, 1, 9]),
            Column::new("ask".into(), [1.5, 2.5, 3.5, 4.5, 5.5]),
        ],
    )
    .unwrap();
    vec![("bids", bids), ("asks", asks)]
}

fn merge_cfg(by: bool) -> MergeConfig {
    MergeConfig {
        clock: "t".into(),
        source: "source".into(),
        by: by.then(|| "sym".into()),
        restart: by.then_some(600e9),
        restart_text: by.then(|| "10m".into()),
        carry: Vec::new(),
    }
}

fn merge_of(
    streams: &[(&'static str, DataFrame)],
    by: bool,
    state: Option<&[u8]>,
) -> Result<Merge, String> {
    let names = streams.iter().map(|(n, _)| (*n).to_owned()).collect();
    let schemas = streams
        .iter()
        .map(|(_, d)| d.schema().as_ref().clone())
        .collect();
    match state {
        None => Merge::new(names, schemas, merge_cfg(by)),
        Some(b) => Merge::load_bytes(b, names, schemas, merge_cfg(by)),
    }
    .map_err(|e| e.to_string())
}

/// Each stream fed two rows at a time, pulled as the Python source pulls,
/// up to `limit` rows; `hold` holds the rows at the last clock.
fn merge_drive(
    m: &mut Merge,
    parts: &[DataFrame],
    limit: Option<usize>,
    hold: bool,
) -> Result<DataFrame, String> {
    let e = |e: PolarsError| e.to_string();
    let mut next = vec![0usize; parts.len()];
    let mut out = DataFrame::empty_with_schema(m.schema());
    loop {
        let (df, want) = m.pull(limit.map(|l| l - out.height()), hold).map_err(e)?;
        out.vstack_mut(&df).map_err(e)?;
        if limit.is_some_and(|l| out.height() >= l) {
            return Ok(out);
        }
        let Some(i) = want else { return Ok(out) };
        if next[i] < parts[i].height() {
            m.push(i, parts[i].slice(next[i] as i64, 2)).map_err(e)?;
            next[i] += 2;
        } else {
            m.end(i);
        }
    }
}

/// The continuation's streams as one frame, each row tagged with its
/// stream's index, and back.
fn merge_join(parts: &[DataFrame]) -> DataFrame {
    let mut schema = Schema::default();
    for d in parts {
        for (n, t) in d.schema().iter() {
            if schema.get(n).is_none() {
                schema.insert(n.clone(), t.clone());
            }
        }
    }
    let mut out = DataFrame::empty_with_schema(&schema);
    out.with_column(Column::new_empty(MERGE_INPUT.into(), &DataType::UInt32))
        .unwrap();
    for (i, d) in parts.iter().enumerate() {
        let h = d.height();
        let mut cols: Vec<Column> = schema
            .iter()
            .map(|(n, t)| {
                d.column(n)
                    .cloned()
                    .unwrap_or_else(|_| Column::full_null(n.clone(), h, t))
            })
            .collect();
        cols.push(Column::new(MERGE_INPUT.into(), vec![i as u32; h]));
        out.vstack_mut(&DataFrame::new(h, cols).unwrap()).unwrap();
    }
    out.rechunk_mut();
    out
}

fn merge_split(
    streams: &[(&'static str, DataFrame)],
    input: &DataFrame,
) -> Result<Vec<DataFrame>, String> {
    let e = |e: PolarsError| e.to_string();
    let tag = input
        .column(MERGE_INPUT)
        .map_err(e)?
        .u32()
        .map_err(e)?
        .clone();
    streams
        .iter()
        .enumerate()
        .map(|(i, (_, d))| {
            let rows = input.filter(&tag.equal(i as u32)).map_err(e)?;
            rows.select(d.get_column_names_owned()).map_err(e)
        })
        .collect()
}

/// `merge`: the streams cut at 3.0, inside its tie -- `quotes`' first row
/// at 3.0 before the cut, its second and `trades`' after -- and the first
/// part merged to its end by a run that holds the tie: the state holds
/// `quotes`' row at 3.0, and `news` no clock.
fn merge_first() -> (Vec<u8>, DataFrame) {
    let streams = merge_streams();
    let before = [7usize, 4, 0];
    let mut m = merge_of(&streams, false, None).unwrap();
    let first: Vec<DataFrame> = streams
        .iter()
        .zip(before)
        .map(|((_, d), k)| d.head(Some(k)))
        .collect();
    merge_drive(&mut m, &first, None, true).unwrap();
    let rest: Vec<DataFrame> = streams
        .iter()
        .zip(before)
        .map(|((_, d), k)| d.slice(k as i64, d.height() - k))
        .collect();
    (m.save_bytes().unwrap(), merge_join(&rest))
}

fn merge_run_of(by: bool, bytes: &[u8], input: &DataFrame) -> Ran {
    let streams = if by {
        merge_by_streams()
    } else {
        merge_streams()
    };
    let mut m = merge_of(&streams, by, Some(bytes))?;
    let again = m.save_bytes()?;
    let parts = merge_split(&streams, input)?;
    let out = merge_drive(&mut m, &parts, None, false)?;
    Ok((again, vec![("merged".into(), out)]))
}

fn merge_run(bytes: &[u8], input: &DataFrame) -> Ran {
    merge_run_of(false, bytes, input)
}

/// `merge_by`: the session streams under `by` and a restart, sliced after
/// the merged stream's ninth row, inside the second session: each stream
/// has started again, every group has merged rows, the null one included,
/// and nothing is held.
fn merge_by_first() -> (Vec<u8>, DataFrame) {
    let streams = merge_by_streams();
    let parts: Vec<DataFrame> = streams.iter().map(|(_, d)| d.clone()).collect();
    let mut m = merge_of(&streams, true, None).unwrap();
    let head = merge_drive(&mut m, &parts, Some(9), false).unwrap();
    let src = head.column("source").unwrap().str().unwrap();
    let rest: Vec<DataFrame> = streams
        .iter()
        .map(|(n, d)| {
            let took = src.iter().filter(|s| *s == Some(*n)).count();
            d.slice(took as i64, d.height() - took)
        })
        .collect();
    (m.save_bytes().unwrap(), merge_join(&rest))
}

fn merge_by_run(bytes: &[u8], input: &DataFrame) -> Ran {
    merge_run_of(true, bytes, input)
}

/// The `merge_by` streams with a column of text, null now and then, for
/// `carry` (PLAN 556) to fill forward with the bids and asks.
fn merge_carry_streams() -> Vec<(&'static str, DataFrame)> {
    let mut streams = merge_by_streams();
    streams[0]
        .1
        .with_column(Column::new(
            "note".into(),
            [
                Some("open"),
                None,
                Some("odd"),
                None,
                None,
                Some("late"),
                None,
            ],
        ))
        .unwrap();
    streams
}

const MERGE_CARRY: [&str; 3] = ["bid", "ask", "note"];

fn merge_carry_of(state: Option<&[u8]>) -> Result<Merge, String> {
    let streams = merge_carry_streams();
    let names = streams.iter().map(|(n, _)| (*n).to_owned()).collect();
    let schemas = streams
        .iter()
        .map(|(_, d)| d.schema().as_ref().clone())
        .collect();
    let mut cfg = merge_cfg(true);
    cfg.carry = MERGE_CARRY.iter().map(|c| (*c).to_owned()).collect();
    match state {
        None => Merge::new(names, schemas, cfg),
        Some(b) => Merge::load_bytes(b, names, schemas, cfg),
    }
    .map_err(|e| e.to_string())
}

/// `merge_carry`: `merge_by`'s streams and cut, carrying the bid, the ask
/// and the note (PLAN 556): every group carries a value of the second
/// session, the null one included, and the note none in some.
fn merge_carry_first() -> (Vec<u8>, DataFrame) {
    let streams = merge_carry_streams();
    let parts: Vec<DataFrame> = streams.iter().map(|(_, d)| d.clone()).collect();
    let mut m = merge_carry_of(None).unwrap();
    let head = merge_drive(&mut m, &parts, Some(9), false).unwrap();
    let src = head.column("source").unwrap().str().unwrap();
    let rest: Vec<DataFrame> = streams
        .iter()
        .map(|(n, d)| {
            let took = src.iter().filter(|s| *s == Some(*n)).count();
            d.slice(took as i64, d.height() - took)
        })
        .collect();
    (m.save_bytes().unwrap(), merge_join(&rest))
}

fn merge_carry_run(bytes: &[u8], input: &DataFrame) -> Ran {
    let mut m = merge_carry_of(Some(bytes))?;
    let again = m.save_bytes()?;
    let parts = merge_split(&merge_carry_streams(), input)?;
    let out = merge_drive(&mut m, &parts, None, false)?;
    Ok((again, vec![("merged".into(), out)]))
}

/// `refresh_time_age`: the `refresh_time` grid with each value's age (PLAN
/// 556), part-way through an interval, so the series seen hold their last
/// tick's clock.
fn refresh_age_first() -> (Vec<u8>, DataFrame) {
    let ticks = ticks();
    let half = ticks.height() / 2;
    let mut r = RefreshTime::new(refresh_names(), false)
        .unwrap()
        .with_age(true);
    r.feed(&ticks.slice(0, half), &refresh_cols()).unwrap();
    (
        r.save_bytes().unwrap(),
        ticks.slice(half as i64, ticks.height() - half),
    )
}

fn refresh_age_run(bytes: &[u8], input: &DataFrame) -> Ran {
    let mut r = RefreshTime::load_bytes_with_age(bytes, refresh_names(), false, false, true)?;
    let again = r.save_bytes()?;
    let fed = r.feed(input, &refresh_cols()).map_err(|e| e.to_string())?;
    Ok((again, vec![("feed".into(), fed)]))
}

/// `refresh_time_age_held`: `refresh_time_held`'s pair grid and slice with
/// each value's age (PLAN 556): the held points keep their values' clocks,
/// on a zoned temporal clock.
fn refresh_age_held_first() -> (Vec<u8>, DataFrame) {
    let df = ticks_held();
    let keep = held_keep();
    let new = || {
        RefreshTime::new(refresh_names(), true)
            .unwrap()
            .with_age(true)
    };
    let whole = new().feed(&df, &held_cols(&keep)).unwrap();
    let i = whole.column("i").unwrap().i64().unwrap();
    let half = (df.height() / 2) as i64;
    let at = (1..i.len())
        .find(|&r| i.get(r) == i.get(r - 1) && i.get(r).unwrap() > half)
        .expect("a tick completing two pairs past the middle");
    let mut r = new();
    r.feed_limited(&df, &held_cols(&keep), Some(at)).unwrap();
    let bytes = r.save_bytes().unwrap();
    let fed = i.get(at).unwrap() as usize + 1;
    (bytes, df.slice(fed as i64, df.height() - fed))
}

fn refresh_age_held_run(bytes: &[u8], input: &DataFrame) -> Ran {
    let keep = held_keep();
    let mut r = RefreshTime::load_bytes_with_age(bytes, refresh_names(), true, true, true)?;
    let again = r.save_bytes()?;
    let fed = r
        .feed(input, &held_cols(&keep))
        .map_err(|e| e.to_string())?;
    Ok((again, vec![("feed".into(), fed)]))
}

/// The stages of the `checkpoint` fixture, in order: a `refresh_time`
/// grid, a window run, a bank and a merge of streams (PLAN 541), each its
/// kind's fixture's state, and each continued on that fixture's input. The
/// bank's input is the fixture's own; the others are made from the same
/// generators.
const STAGES: [&str; 4] = ["grid", "windows", "bank", "merge"];

fn checkpoint_first() -> (Vec<u8>, DataFrame) {
    let (grid, _) = refresh_first();
    let (windows, _) = windows_first();
    let (bank, input) = formula_first();
    let (merge, _) = merge_first();
    let stages: Vec<(&str, &[u8])> = STAGES
        .into_iter()
        .zip([&grid[..], &windows, &bank, &merge])
        .collect();
    (checkpoint_bytes(&stages).unwrap(), input)
}

fn checkpoint_run(bytes: &[u8], input: &DataFrame) -> Ran {
    let stages = load_checkpoint(bytes)?;
    let names: Vec<(&str, StateKind)> = stages.iter().map(|s| (s.name.as_str(), s.kind)).collect();
    if names
        != STAGES
            .into_iter()
            .zip([
                StateKind::RefreshTime,
                StateKind::Windows,
                StateKind::Bank,
                StateKind::Merge,
            ])
            .collect::<Vec<_>>()
    {
        return Err(format!("the checkpoint's stages are {names:?}"));
    }
    let ticks = ticks();
    let half = ticks.height() / 2;
    let runs = [
        refresh_run(
            &stages[0].state,
            &ticks.slice(half as i64, ticks.height() - half),
        )?,
        windows_run(
            &stages[1].state,
            &window_input(&frame()).slice(SPLIT as i64, N - SPLIT),
        )?,
        formula_run(&stages[2].state, input)?,
        merge_run(&stages[3].state, &merge_first().1)?,
    ];
    let again: Vec<(&str, &[u8])> = STAGES
        .into_iter()
        .zip(runs.iter().map(|(b, _)| &b[..]))
        .collect();
    let again = checkpoint_bytes(&again)?;
    let frames = STAGES
        .into_iter()
        .zip(runs)
        .flat_map(|(stage, (_, frames))| {
            frames
                .into_iter()
                .map(move |(name, df)| (format!("{stage}/{name}"), df))
        })
        .collect();
    Ok((again, frames))
}

// --- bars ---------------------------------------------------------------------

/// The aggregation `kind` of `column`, written as `name`.
fn bar_agg(kind: AggKind, column: Option<&str>, name: &str) -> AggSpec {
    AggSpec {
        kind,
        column: column.map(str::to_string),
        name: name.into(),
    }
}

/// The three calls of the bars fixtures: `bars`, ungrouped, `"sum"` on a
/// float clock; `bars_by`, grouped, `"imbalance"` of an integer past 2^53 on
/// an integer clock, with a threshold column, a weight, a session and the
/// clock's cuts; `bars_count`, `"count"` on a temporal clock under a
/// monotone close and `max_span`.
fn bars_cfg(name: &str) -> BarsConfig {
    let mut cfg = BarsConfig {
        kind: BarKind::Sum,
        clock: "t".into(),
        threshold: BarThreshold::Value(10.0),
        quantity: Some("q".into()),
        group: None,
        weight: None,
        session: None,
        gap_cap: None,
        max_span: None,
        restart_after_step_back: None,
        keep_partial: true,
        monotone: false,
        max_state_bytes: 256 * 1024,
        max_total_bytes: 1 << 30,
        aggregates: vec![
            bar_agg(AggKind::First, Some("p"), "open"),
            bar_agg(AggKind::Max, Some("p"), "high"),
            bar_agg(AggKind::Min, Some("x"), "low"),
            bar_agg(AggKind::Last, Some("p"), "close"),
            bar_agg(AggKind::Sum, Some("x"), "sx"),
            bar_agg(AggKind::Sum, Some("n"), "sn"),
            bar_agg(AggKind::Count, Some("p"), "np"),
            bar_agg(AggKind::Len, None, "rows"),
        ],
        call: [("clock".to_string(), "t".to_string())]
            .into_iter()
            .collect(),
    };
    match name {
        "bars_by" => {
            cfg.kind = BarKind::Imbalance;
            cfg.quantity = Some("n".into());
            cfg.threshold = BarThreshold::Column("th".into());
            cfg.group = Some("g".into());
            cfg.weight = Some("w".into());
            cfg.session = Some("day".into());
            cfg.gap_cap = Some(6.0);
            cfg.restart_after_step_back = Some(100.0);
            cfg.aggregates = vec![
                bar_agg(AggKind::Last, Some("d"), "last_d"),
                bar_agg(AggKind::Min, Some("b"), "all_b"),
                bar_agg(AggKind::Sum, Some("b"), "n_b"),
                bar_agg(AggKind::Max, Some("u"), "max_u"),
            ];
        }
        "bars_count" => {
            cfg.kind = BarKind::Count;
            cfg.quantity = None;
            cfg.threshold = BarThreshold::Value(3.0);
            cfg.group = Some("day".into());
            cfg.monotone = true;
            cfg.keep_partial = false;
            cfg.max_span = Some(5e9);
            cfg.aggregates = vec![
                bar_agg(AggKind::First, Some("u"), "first_u"),
                bar_agg(AggKind::Sum, Some("u"), "sum_u"),
            ];
        }
        _ => {}
    }
    cfg
}

/// Each fixture's whole input, and the row its first part ends at.
fn bars_input(name: &str) -> (DataFrame, usize) {
    let tiny = 2f64.powi(-60);
    match name {
        "bars" => {
            // The open bar at the cut holds a total of 3 + 2^-60 + 1e-30
            // in three parts, a null first `p`, a NaN in `x` and a sum of
            // `x` in two parts.
            let t: Vec<f64> = (0..12).map(|i| f64::from(i) * 0.5).collect();
            let q = [
                4.0,
                6.0,
                3.0,
                tiny,
                1e-30,
                9.0,
                0.5,
                25.0,
                f64::NAN,
                1.0,
                2.0,
                3.0,
            ];
            let p = [
                Some(1.0),
                Some(2.0),
                None,
                Some(1.5),
                Some(0.5),
                Some(3.0),
                None,
                Some(2.5),
                Some(2.0),
                None,
                Some(1.0),
                Some(4.0),
            ];
            let x = [
                1.0,
                -2.0,
                1e10,
                1e-10,
                f64::NAN,
                0.25,
                3.0,
                -1.0,
                2.0,
                f64::INFINITY,
                0.5,
                1.0,
            ];
            let n: Vec<i32> = (0..12).map(|i| i * 7 - 30).collect();
            let df = df!("t" => t, "q" => q, "p" => p, "x" => x, "n" => n).unwrap();
            (df, 5)
        }
        "bars_by" => {
            let big = (1i64 << 53) + 1;
            let t: Vec<i64> = vec![0, 1, 2, 3, 3, 4, 12, 13, 14, 15, 16, 17];
            let g = ["a", "b", "a", "b", "a", "b", "a", "b", "a", "b", "a", "b"];
            let n = [big, -5, -big, 7, 3, -4, big, 1, 2, -big, 5, 9];
            let th = [
                1e16, 6.0, 1e16, 6.0, 1e16, 6.0, 1e16, 6.0, 1e16, 6.0, 1e16, 6.0,
            ];
            let w = [1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0];
            let day = [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2];
            let d: Vec<i64> = (0..12).map(|i| 1_700_000_000_000_000 + i * 1_000).collect();
            let b = [
                Some(true),
                Some(false),
                None,
                Some(true),
                Some(true),
                Some(true),
                Some(false),
                None,
                Some(true),
                Some(true),
                Some(false),
                Some(true),
            ];
            let u: Vec<u64> = (0..12).map(|i| u64::MAX - 100 * i).collect();
            let mut df = df!("t" => t, "g" => g, "n" => n, "th" => th, "w" => w, "day" => day,
                "d" => d, "b" => b, "u" => u)
            .unwrap();
            df.apply("d", |s| {
                s.cast(&DataType::Datetime(TimeUnit::Microseconds, None))
                    .unwrap()
            })
            .unwrap();
            (df, 6)
        }
        _ => {
            let t: Vec<i64> = (0..12)
                .map(|i| 1_700_000_000_000_000_000 + i * 2_000_000_000)
                .collect();
            let day: Vec<i64> = vec![1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3];
            let u: Vec<u64> = (0..12).map(|i| 3 * i + 1).collect();
            let mut df = df!("t" => t, "day" => day, "u" => u).unwrap();
            df.apply("t", |s| {
                s.cast(&DataType::Datetime(TimeUnit::Nanoseconds, None))
                    .unwrap()
            })
            .unwrap();
            (df, 7)
        }
    }
}

fn bars_first_of(name: &str) -> (Vec<u8>, DataFrame) {
    let (df, cut) = bars_input(name);
    let mut b = Bars::new(bars_cfg(name), df.schema()).unwrap();
    b.feed(&df.head(Some(cut)), None).unwrap();
    (
        b.save_bytes().unwrap(),
        df.slice(cut as i64, df.height() - cut),
    )
}

fn bars_run_of(name: &str, bytes: &[u8], input: &DataFrame) -> Ran {
    let mut b = Bars::load_bytes(bytes, bars_cfg(name), input.schema())?;
    let again = b.save_bytes()?;
    let out = b.feed(input, None).map_err(|e| e.to_string())?;
    let open = b.finish().map_err(|e| e.to_string())?;
    Ok((again, vec![("bars".into(), out), ("open".into(), open)]))
}

fn bars_first() -> (Vec<u8>, DataFrame) {
    bars_first_of("bars")
}

fn bars_run(bytes: &[u8], input: &DataFrame) -> Ran {
    bars_run_of("bars", bytes, input)
}

fn bars_by_first() -> (Vec<u8>, DataFrame) {
    bars_first_of("bars_by")
}

fn bars_by_run(bytes: &[u8], input: &DataFrame) -> Ran {
    bars_run_of("bars_by", bytes, input)
}

fn bars_count_first() -> (Vec<u8>, DataFrame) {
    bars_first_of("bars_count")
}

fn bars_count_run(bytes: &[u8], input: &DataFrame) -> Ran {
    bars_run_of("bars_count", bytes, input)
}

fn kinds() -> Vec<Kind> {
    vec![
        Kind {
            name: "bank",
            first: bank_first,
            run: bank_run,
        },
        Kind {
            name: "bank_diagnosed",
            first: diagnosed_first,
            run: diagnosed_run,
        },
        Kind {
            name: "bank_autocorr",
            first: autocorr_first,
            run: autocorr_run,
        },
        Kind {
            name: "bank_reset",
            first: reset_first,
            run: reset_run,
        },
        Kind {
            name: "bank_refit",
            first: refit_first,
            run: refit_run,
        },
        Kind {
            name: "bank_refit_64",
            first: refit_64_first,
            run: refit_run,
        },
        Kind {
            name: "bank_gated",
            first: gated_first,
            run: gated_run,
        },
        Kind {
            name: "bank_reset_held",
            first: reset_held_first,
            run: reset_held_run,
        },
        Kind {
            name: "bank_unstable",
            first: unstable_first,
            run: unstable_run,
        },
        Kind {
            name: "bank_notice_pending",
            first: notice_first,
            run: notice_run,
        },
        Kind {
            name: "bank_label_clock",
            first: label_first,
            run: label_run,
        },
        Kind {
            name: "bank_conformal_auto",
            first: auto_first,
            run: auto_run,
        },
        Kind {
            name: "bank_drift_kinds",
            first: drift_kinds_first,
            run: drift_kinds_run,
        },
        Kind {
            name: "bank_drift_ranges",
            first: drift_ranges_first,
            run: drift_ranges_run,
        },
        Kind {
            name: "bank_refit_span",
            first: refit_span_first,
            run: refit_span_run,
        },
        Kind {
            name: "bank_bocpd_hazard",
            first: bocpd_hazard_first,
            run: bocpd_hazard_run,
        },
        Kind {
            name: "bank_target_weight",
            first: target_weight_first,
            run: target_weight_run,
        },
        Kind {
            name: "with_windows",
            first: windows_first,
            run: windows_run,
        },
        Kind {
            name: "with_windows_int",
            first: windows_int_first,
            run: windows_int_run,
        },
        Kind {
            name: "with_windows_quantile",
            first: windows_quantile_first,
            run: windows_quantile_run,
        },
        Kind {
            name: "with_windows_moments",
            first: windows_moments_first,
            run: windows_moments_run,
        },
        Kind {
            name: "with_windows_profile",
            first: windows_profile_first,
            run: windows_profile_run,
        },
        Kind {
            name: "with_windows_profile_units",
            first: windows_profile_units_first,
            run: windows_profile_units_run,
        },
        Kind {
            name: "with_windows_extrema",
            first: windows_extrema_first,
            run: windows_extrema_run,
        },
        Kind {
            name: "with_windows_tick",
            first: windows_tick_first,
            run: windows_tick_run,
        },
        Kind {
            name: "with_windows_rows",
            first: windows_rows_first,
            run: windows_rows_run,
        },
        Kind {
            name: "with_windows_invert",
            first: windows_invert_first,
            run: windows_invert_run,
        },
        Kind {
            name: "refresh_time",
            first: refresh_first,
            run: refresh_run,
        },
        Kind {
            name: "refresh_time_int",
            first: refresh_int_first,
            run: refresh_run,
        },
        Kind {
            name: "refresh_time_held",
            first: refresh_held_first,
            run: refresh_held_run,
        },
        Kind {
            name: "bank_formula",
            first: formula_first,
            run: formula_run,
        },
        Kind {
            name: "merge",
            first: merge_first,
            run: merge_run,
        },
        Kind {
            name: "merge_by",
            first: merge_by_first,
            run: merge_by_run,
        },
        Kind {
            name: "checkpoint",
            first: checkpoint_first,
            run: checkpoint_run,
        },
        Kind {
            name: "merge_carry",
            first: merge_carry_first,
            run: merge_carry_run,
        },
        Kind {
            name: "refresh_time_age",
            first: refresh_age_first,
            run: refresh_age_run,
        },
        Kind {
            name: "refresh_time_age_held",
            first: refresh_age_held_first,
            run: refresh_age_held_run,
        },
        Kind {
            name: "bars",
            first: bars_first,
            run: bars_run,
        },
        Kind {
            name: "bars_by",
            first: bars_by_first,
            run: bars_by_run,
        },
        Kind {
            name: "bars_count",
            first: bars_count_first,
            run: bars_count_run,
        },
    ]
}

fn kind_of(f: &Fixture) -> Kind {
    kinds()
        .into_iter()
        .find(|k| k.name == f.name)
        .unwrap_or_else(|| panic!("{}: no such kind; regenerate with `{REGENERATE}`", f.name))
}

// --- comparison -------------------------------------------------------------

/// Every column of `got` against `want`: the nulls in the same rows, the
/// numbers within `tol` of each other relative to `1 + |want|` (the same
/// bits at 0; a NaN on both sides agrees), and every other value equal.
fn same(what: &str, got: &DataFrame, want: &DataFrame, tol: f64) {
    same_within(what, got, want, tol, &[]);
}

/// [`same`], with each column `within` names held to `tol` or that name's
/// bound, if larger ([`CONTINUES_WITHIN`]): the whole column where the
/// entry names no field prefix, and otherwise, in a struct column, only
/// the fields whose names start with one of its prefixes.
fn same_within(what: &str, got: &DataFrame, want: &DataFrame, tol: f64, within: &[Within]) {
    assert_eq!(got.shape(), want.shape(), "{what}: shape");
    for (g, w) in got.columns().iter().zip(want.columns()) {
        assert_eq!(g.name(), w.name(), "{what}: column names");
        let (gs, ws) = (g.as_materialized_series(), w.as_materialized_series());
        let what = format!("{what}.{}", w.name());
        match within.iter().find(|(n, _, _)| *n == w.name().as_str()) {
            Some(&(_, prefixes, b)) if !prefixes.is_empty() => {
                let (DataType::Struct(gf), DataType::Struct(wf)) = (gs.dtype(), ws.dtype()) else {
                    panic!("{what}: field prefixes name a column that is not a struct");
                };
                let names = |f: &[Field]| f.iter().map(|f| f.name().clone()).collect::<Vec<_>>();
                assert_eq!(names(gf), names(wf), "{what}: fields");
                assert_eq!(gs.len(), ws.len(), "{what}: length");
                let (gst, wst) = (gs.struct_().unwrap(), ws.struct_().unwrap());
                for (gf, wf) in gst.fields_as_series().iter().zip(wst.fields_as_series()) {
                    let moves = prefixes.iter().any(|p| wf.name().starts_with(p));
                    let tol = if moves { tol.max(b) } else { tol };
                    same_series(&format!("{what}.{}", wf.name()), gf, &wf, tol);
                }
            }
            entry => {
                let tol = entry.map_or(tol, |&(_, _, b)| tol.max(b));
                same_series(&what, gs, ws, tol);
            }
        }
    }
}

/// A [`CONTINUES_WITHIN`] entry: the column, the prefixes of the struct
/// fields that move (none: the whole column), and the bound.
type Within = (&'static str, &'static [&'static str], f64);

fn same_series(what: &str, got: &Series, want: &Series, tol: f64) {
    // `withheld_reason`'s names only ever grow at the end -- the code a
    // state's last row keeps is a name's position, so a new reason takes
    // the next one (`WITHHELD_REASONS`; `ill_conditioned`, task 278 (d4)).
    // So a frozen output's enum is a prefix of today's, and its values are
    // compared by name.
    if let (DataType::Enum(g, _), DataType::Enum(w, _)) = (got.dtype(), want.dtype()) {
        let gc: Vec<&str> = g.categories().values_iter().collect();
        let wc: Vec<&str> = w.categories().values_iter().collect();
        assert!(
            wc.len() <= gc.len() && gc[..wc.len()] == wc[..],
            "{what}: the frozen names {wc:?} are not the start of {gc:?}"
        );
        let (gs, ws) = (
            got.cast(&DataType::String).unwrap(),
            want.cast(&DataType::String).unwrap(),
        );
        assert!(gs.equals_missing(&ws), "{what}: {gs:?} against {ws:?}");
        return;
    }
    // A struct by its fields' names, each field then by its own dtype, so
    // that an enum inside one is read by the rule above.
    if let (DataType::Struct(gf), DataType::Struct(wf)) = (got.dtype(), want.dtype()) {
        let names = |f: &[Field]| f.iter().map(|f| f.name().clone()).collect::<Vec<_>>();
        assert_eq!(names(gf), names(wf), "{what}: fields");
        assert_eq!(got.len(), want.len(), "{what}: length");
        let (g, w) = (got.struct_().unwrap(), want.struct_().unwrap());
        for (gf, wf) in g.fields_as_series().iter().zip(w.fields_as_series()) {
            same_series(&format!("{what}.{}", wf.name()), gf, &wf, tol);
        }
        return;
    }
    assert_eq!(got.dtype(), want.dtype(), "{what}: dtype");
    assert_eq!(got.len(), want.len(), "{what}: length");
    match want.dtype() {
        DataType::List(inner) if **inner == DataType::Float64 => {
            let (g, w) = (got.list().unwrap(), want.list().unwrap());
            for i in 0..want.len() {
                match (g.get_as_series(i), w.get_as_series(i)) {
                    (Some(a), Some(b)) => same_series(&format!("{what}[{i}]"), &a, &b, tol),
                    (None, None) => {}
                    (a, b) => panic!("{what}, row {i}: {a:?} against {b:?}"),
                }
            }
        }
        DataType::Float64 => {
            let (g, w) = (got.f64().unwrap(), want.f64().unwrap());
            for (i, (a, b)) in g.iter().zip(w.iter()).enumerate() {
                let ok = match (a, b) {
                    (None, None) => true,
                    (Some(a), Some(b)) if a.is_nan() || b.is_nan() => a.is_nan() && b.is_nan(),
                    (Some(a), Some(b)) if tol == 0.0 => a.to_bits() == b.to_bits(),
                    // an infinity the same on both sides: their difference is NaN
                    (Some(a), Some(b)) if a == b => true,
                    (Some(a), Some(b)) => (a - b).abs() <= tol * (1.0 + b.abs()),
                    _ => false,
                };
                assert!(ok, "{what}, row {i}: got {a:?}, frozen {b:?}");
            }
        }
        _ => assert!(got.equals_missing(want), "{what}: {got:?} against {want:?}"),
    }
}

// --- the checks -------------------------------------------------------------

#[test]
fn every_fixture_loads_goes_on_to_the_bit_and_saves_its_bytes_again() {
    if regenerating() {
        return;
    }
    // The fixtures are discovered in file-name order and the kinds are in
    // file order: compared as sets, and counted, so a discovery that found
    // nothing fails here too.
    let mut names: Vec<&str> = frozen::ALL.iter().map(|f| f.name).collect();
    let mut kinds_named: Vec<&str> = kinds().iter().map(|k| k.name).collect();
    names.sort_unstable();
    kinds_named.sort_unstable();
    assert!(!kinds_named.is_empty(), "no kind to freeze");
    assert_eq!(
        names, kinds_named,
        "the frozen fixtures are not the kinds: regenerate with `{REGENERATE}`"
    );
    assert_eq!(
        fixtures_on_disk(&dir()),
        Ok(frozen::ALL.len()),
        "the current set compiled in is not the files on disk: rebuild the test"
    );
    for f in frozen::ALL {
        let kind = kind_of(f);
        let bytes = unhex(f.state);
        let input = from_ipc(f.input);
        let (again, frames) = (kind.run)(&bytes, &input).unwrap_or_else(|e| {
            panic!(
                "{}: the frozen state does not load: {e}\n(a layout or version change ships \
                 its fixtures: before 1.0, regenerate with `{REGENERATE}`)",
                f.name
            )
        });
        assert!(
            again == bytes,
            "{}: the loaded state saves other bytes",
            f.name
        );
        let tol = if f.writer == writer() { 0.0 } else { TOL };
        assert_eq!(frames.len(), f.output.len(), "{}", f.name);
        for ((name, got), (frozen_name, want)) in frames.iter().zip(f.output) {
            assert_eq!(name, frozen_name, "{}", f.name);
            let want = from_ipc(want);
            // Every frame has rows to compare: a group closes in the
            // continuation, the held rows are released, the grid has points.
            assert!(want.height() > 0, "{}: {name} is empty", f.name);
            same(&format!("{}: {name}", f.name), got, &want, tol);
        }
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

/// Every schema a bank file loads from has its fixtures: the bank's minimum
/// cannot sit below the oldest set kept, nor the schema move without one.
/// A kept schema's set is on disk, under `v<N>/`, holds a bank file, every
/// fixture in it of schema `N`, and each set compiled in holds every file
/// its directory does (review round 5, D2; discovery, task 281 (4)).
#[test]
fn the_bank_fixtures_cover_the_schemas_a_bank_loads() {
    if regenerating() {
        return;
    }
    let bank = frozen::ALL
        .iter()
        .find(|f| f.name == "bank")
        .expect("a bank fixture");
    assert_eq!(
        bank.schema, SCHEMA_VERSION,
        "SCHEMA_VERSION is {SCHEMA_VERSION} and the frozen bank file is of {}: before 1.0, \
         regenerate with `{REGENERATE}` and raise MIN_BANK_SCHEMA_VERSION with it; from 1.0, \
         keep this set as the previous schema's (the module doc), write its loader, then \
         regenerate",
        bank.schema
    );
    let previous = previous();
    let covered: Vec<u32> = previous.iter().copied().chain([bank.schema]).collect();
    let claimed: Vec<u32> = (MIN_BANK_SCHEMA_VERSION..=SCHEMA_VERSION).collect();
    assert_eq!(
        claimed, covered,
        "a bank loads schemas {MIN_BANK_SCHEMA_VERSION}..={SCHEMA_VERSION} and fixtures cover \
         {covered:?}: each needs a frozen file, held to its loader"
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
        assert!(
            set.iter().any(|f| f.name == "bank"),
            "schema {schema}: the kept set has no bank file, the kind the minimum is about"
        );
    }
}

/// How many files each kept schema's set holds, frozen when the set was
/// copied (review S4). The kept sets are the only test of an old schema's
/// loader, and the check above holds only that a set has a bank file and
/// that what was compiled in is what is on disk, so a deleted file or a
/// partial copy passed. A new kept set adds its line here, with the count
/// of the current set it was copied from.
const KEPT_COUNTS: &[(u32, usize)] = &[
    (61, 4),
    (62, 6),
    (63, 8),
    (64, 11),
    (65, 12),
    (66, 15),
    (67, 18),
    (68, 18),
    (69, 18),
    (70, 18),
    (71, 19),
];

/// The one place a set lacks a case of the set before it: the window
/// runs' fixtures were not copied into `v64/` and `v65/` when those sets
/// were kept, so `v63/`'s three are not in `v64/`; `v66/` has them again.
/// Found by review S4's check; the states of schemas 64 and 65 cannot be
/// written now, so the gap is named rather than filled.
const KNOWN_GAPS: &[(&str, &str)] = &[
    ("v63", "with_windows"),
    ("v63", "with_windows_int"),
    ("v63", "with_windows_quantile"),
];

/// The names one set holds that the next does not, each set given as a
/// label and its fixtures' names, oldest first, less [`KNOWN_GAPS`]: a
/// later set is copied from the current one, which only gains cases.
fn names_dropped(sets: &[(String, Vec<&str>)], known: &[(&str, &str)]) -> Vec<String> {
    let mut out = Vec::new();
    for pair in sets.windows(2) {
        let (old, new) = (&pair[0], &pair[1]);
        for name in &old.1 {
            if !new.1.contains(name) && !known.contains(&(old.0.as_str(), *name)) {
                out.push(format!("{}/{name}.rs is not in {}", old.0, new.0));
            }
        }
    }
    out
}

/// Each kept set is whole: its count is the one frozen in [`KEPT_COUNTS`],
/// and its names hold every name of the set before it (less
/// [`KNOWN_GAPS`]), the current set every name of the newest kept one
/// (review S4).
#[test]
fn every_kept_set_is_whole() {
    if regenerating() {
        return;
    }
    let sets = previous_sets();
    let counts: Vec<(u32, usize)> = sets.iter().map(|(s, set)| (*s, set.len())).collect();
    assert_eq!(
        counts, KEPT_COUNTS,
        "the kept sets' counts of files (schema, count) are not the frozen ones: a kept file \
         was deleted or added. A new kept set adds its line to KEPT_COUNTS"
    );
    let names = |set: &[&Fixture]| set.iter().map(|f| f.name).collect::<Vec<&str>>();
    let mut chain: Vec<(String, Vec<&str>)> = sets
        .iter()
        .map(|(s, set)| (format!("v{s}"), names(set)))
        .collect();
    chain.push(("the current set".to_owned(), names(frozen::ALL)));
    let dropped = names_dropped(&chain, KNOWN_GAPS);
    assert!(
        dropped.is_empty(),
        "each set holds every case of the set before it:\n{}",
        dropped.join("\n")
    );
}

/// The chain check on its own: a name the next set lacks is named, unless
/// it is a known gap.
#[test]
fn a_set_missing_a_case_of_the_one_before_is_named() {
    let set = |label: &str, names: &[&'static str]| (label.to_owned(), names.to_vec());
    let partial = [
        set("v1", &["a", "b"]),
        set("v2", &["a", "c"]),
        set("now", &["a", "c"]),
    ];
    assert_eq!(names_dropped(&partial, &[]), ["v1/b.rs is not in v2"]);
    assert!(names_dropped(&partial, &[("v1", "b")]).is_empty());
}

/// The on-disk check behind the coverage test, proven on a directory of
/// its own (review round 5, D2): a schema with no set, or an empty one, is
/// refused; a set with fixtures passes and is counted.
#[test]
fn a_kept_schema_without_its_set_on_disk_is_refused() {
    let root = std::env::temp_dir().join(format!("rheon-bank-fixture-sets-{}", std::process::id()));
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
    std::fs::write(set.join("bank.rs"), "// a fixture\n").unwrap();
    assert_eq!(set_on_disk(&root, 44), Ok(1));
    std::fs::write(set.join("refresh_time.rs"), "// a fixture\n").unwrap();
    assert_eq!(set_on_disk(&root, 44), Ok(2));
    assert!(set_on_disk(&root, 43).is_err(), "another schema has no set");
    std::fs::remove_dir_all(&root).unwrap();
}

/// The envelope's head (PLAN 542): `magic`, `kind` and `version` as the
/// file's first three entries, read from its first bytes as the loaders
/// read them, or `None`.
fn envelope_head(bytes: &[u8]) -> Option<(String, String, u64)> {
    let v = rmpv::decode::read_value(&mut &bytes[..]).ok()?;
    let m = v.as_map()?;
    let entry = |i: usize, key: &str| m.get(i).filter(|(k, _)| k.as_str() == Some(key));
    Some((
        entry(0, "magic")?.1.as_str()?.to_owned(),
        entry(1, "kind")?.1.as_str()?.to_owned(),
        entry(2, "version")?.1.as_u64()?,
    ))
}

/// The forms schemas 71 and 72 added, held in the files (review 5, B1 and
/// D3): every current fixture is written in the envelope -- the head of its
/// kind and version first, the checksum last -- a checkpoint's stages each a
/// whole state of its own kind, and a bank's formula target's core a whole
/// window run's state; every kept file of 71 begins with its kind's old
/// magic, its bank files and window states sealed and its `refresh_time`
/// states not; and no kept file of 70 or earlier is sealed.
#[test]
fn every_current_fixture_is_written_in_the_envelope() {
    const TAIL: &[u8] = b"\xa8checksum\xc4\x08";
    let sealed = |bytes: &[u8]| {
        bytes.len() > TAIL.len() + 8
            && &bytes[bytes.len() - 8 - TAIL.len()..bytes.len() - 8] == TAIL
    };
    let kind_of = |name: &str| {
        if name == "checkpoint" {
            ("checkpoint", 1)
        } else if name.starts_with("bars") {
            ("bars", 1)
        } else if name.starts_with("refresh_time") {
            ("refresh_time", 3)
        } else if name.starts_with("merge") {
            ("merge", 1)
        } else if name.starts_with("with_windows") {
            ("windows", 14)
        } else {
            ("bank", u64::from(SCHEMA_VERSION))
        }
    };
    let mut kinds = std::collections::BTreeMap::new();
    for f in frozen::ALL {
        let bytes = unhex(f.state);
        let (kind, version) = kind_of(f.name);
        assert_eq!(
            envelope_head(&bytes),
            Some(("rheon-state".to_owned(), kind.to_owned(), version)),
            "{}",
            f.name
        );
        assert!(sealed(&bytes), "{}", f.name);
        *kinds.entry(kind).or_insert(0) += 1;
    }
    assert_eq!(
        kinds.into_iter().collect::<Vec<_>>(),
        [
            ("bank", 18),
            ("bars", 3),
            ("checkpoint", 1),
            ("merge", 3),
            ("refresh_time", 5),
            ("windows", 10)
        ]
    );
    // A checkpoint's stages, each a whole state of its kind; and the bank's
    // formula target's cores inside its bank stage, each a window run's.
    let f = frozen::ALL.iter().find(|f| f.name == "checkpoint").unwrap();
    let stages = load_checkpoint(&unhex(f.state)).unwrap();
    for (s, (kind, version)) in stages.iter().zip([
        ("refresh_time", 3),
        ("windows", 14),
        ("bank", 72),
        ("merge", 1),
    ]) {
        assert_eq!(
            envelope_head(&s.state),
            Some(("rheon-state".to_owned(), kind.to_owned(), version)),
            "{}",
            s.name
        );
    }
    let f = frozen::ALL
        .iter()
        .find(|f| f.name == "bank_formula")
        .unwrap();
    let v = rmpv::decode::read_value(&mut &unhex(f.state)[..]).unwrap();
    let resolvers = v
        .as_map()
        .unwrap()
        .iter()
        .find(|(k, _)| k.as_str() == Some("resolvers"))
        .expect("the bank holds its formula target's cores")
        .1
        .clone();
    let mut cores = 0;
    for spec in resolvers.as_array().unwrap() {
        for group in spec.as_array().unwrap()[1].as_array().unwrap() {
            let core = &group.as_array().unwrap()[1];
            let bytes: Vec<u8> = match core {
                rmpv::Value::Binary(b) => b.clone(),
                rmpv::Value::Array(a) => a.iter().map(|x| x.as_u64().unwrap() as u8).collect(),
                other => panic!("a core is bytes, not {other}"),
            };
            assert_eq!(
                envelope_head(&bytes),
                Some(("rheon-state".to_owned(), "windows".to_owned(), 14))
            );
            assert!(sealed(&bytes));
            cores += 1;
        }
    }
    assert!(cores > 0, "a core held across the save");
    for (schema, set) in previous_sets() {
        for f in set {
            let bytes = unhex(f.state);
            assert_eq!(envelope_head(&bytes), None, "v{schema}/{}", f.name);
            let want = schema == 71 && !f.name.starts_with("refresh_time");
            assert_eq!(sealed(&bytes), want, "v{schema}/{}", f.name);
        }
    }
}

/// Each kept schema's fixtures, held to the loader the change after it
/// shipped (the module doc; review round 5, D1 and D2): every one loads,
/// goes on to the bit against its own frozen frames -- as it did when
/// written -- and, loaded then saved, writes the current schema's fixture
/// bytes of the same kind: the same stream saved at the same row, so the
/// converted state is the one this build writes (the exact-conversion
/// check). The sets must share a writer, since a libm bit in a decayed
/// sum differs by platform. Nothing to run before 1.0, where no set
/// is kept.
///
/// What it cannot see (review 8, E-7; recorded by the user's decision,
/// 2026-10-16): a previous fixture whose map *key* is corrupted, when the
/// field it named defaults to the value the fixture froze. The persisted
/// state structs do not `deny_unknown_fields`, so the unknown key is
/// skipped, the field takes its `#[serde(default)]`, and the state loads,
/// goes on and converts to the current bytes as if nothing were wrong
/// (one hex digit of `v61/bank.rs`'s `prev_session` key passed all six
/// tests). A corrupted *value*, or any byte of a current fixture, fails.
/// The structs stay open on purpose: a field removed since a previous
/// schema is an unknown key in that schema's files, and denying unknown
/// fields would refuse every one of them.
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
            let kind = kind_of(f);
            if schema <= 70 && ZERO_WEIGHT_CLOCK.iter().any(|z| z.0 == f.name) {
                zero_weight_clock_rule(&kind, f, &what);
                continue;
            }
            let (again, frames) =
                (kind.run)(&unhex(f.state), &from_ipc(f.input)).unwrap_or_else(|e| {
                    panic!("{what}: the state does not load through this build's loader: {e}")
                });
            let current = frozen::ALL
                .iter()
                .find(|c| c.name == f.name)
                .unwrap_or_else(|| {
                    panic!("{what}: the current set has no fixture of that kind; a kept kind stays")
                });
            assert_eq!(
                current.writer, f.writer,
                "{what}: written on {}, the current set on {}: the sets are written on one \
                 platform, or their bytes differ in libm's last bit",
                f.writer, current.writer
            );
            // A window's snapshots from before task 275 carry no running
            // products (the doc above `previous`): its continuation goes on
            // from the saved state's reads, the first to the bit, and parts
            // from the old build's in the last bits once a snapshot taken
            // after the load is the boundary, as a fresh run's reads do; the
            // state it saves is then not the current fixture's, whose window
            // read its products from the first row.
            let windowed = holds_a_window(&unhex(f.state));
            let kept = rmpv::decode::read_value(&mut &unhex(f.state)[..]).unwrap();
            let old_scale = schema <= 64
                && (sketches(&kept, "anchor") > 0 || holds_for_a_refit(&unhex(current.state)));
            if old_scale {
                kept_scale_rule(&kind, &kept, &again, &from_ipc(f.input), &what);
            } else if moved_by_the_held_mass(f.name)
                || started_fresh(f.name, schema)
                || moved_by_the_mean_weight(schema, f.name)
            {
                // Rule 5's fifteenth exception: the queue's sums hold the old
                // form's bits for the rows before the save, so the state this
                // build saves after the continued rows is not the current
                // fixture's (the same bytes where no row after the load has a
                // gap of a hundredth of a half-life): it saves bytes that load
                // back the same.
                let (resaved, _) = (kind.run)(&again, &from_ipc(f.input))
                    .unwrap_or_else(|e| panic!("{what}: {e}"));
                assert!(
                    again == unhex(current.state) || resaved == again,
                    "{what}: saved, loaded, saved"
                );
            } else if !windowed {
                assert!(
                    again == unhex(current.state),
                    "{what}: loaded, the state does not save the current schema's fixture bytes \
                     of the same kind: the loader converts it to another state than this build \
                     reaches from the same stream"
                );
            }
            if seeded_lag0(schema, f.name) {
                // Rule 5's thirty-first exception: the kept state has no lag
                // 0 of its own, so it is not the current fixture's; it saves
                // bytes that load back the same.
                let (resaved, _) = (kind.run)(&again, &from_ipc(f.input))
                    .unwrap_or_else(|e| panic!("{what}: {e}"));
                assert!(resaved == again, "{what}: saved, loaded, saved");
            }
            let tol = if f.writer == writer() { 0.0 } else { TOL };
            assert_eq!(frames.len(), f.output.len(), "{what}");
            assert_eq!(current.output.len(), f.output.len(), "{what}");
            for (((name, got), (frozen_name, want)), (_, now)) in
                frames.iter().zip(f.output).zip(current.output)
            {
                assert_eq!(name, frozen_name, "{what}");
                let kept_frame = from_ipc(want);
                let want = moved_since(&kept_frame, &from_ipc(now));
                let want = if seeded_lag0(schema, f.name) {
                    fallback_cells(&kept_frame, &want, got)
                } else {
                    want
                };
                let want = whole_at_the_end_of_the_input(f.name, name, &want, &from_ipc(now));
                let want = if old_scale {
                    new_after_load(f.name, name, &from_ipc(f.input), &want, &from_ipc(now))
                } else {
                    want
                };
                if windowed {
                    same_within(
                        &format!("{what}: {name}, the first row"),
                        &got.slice(0, 1),
                        &want.slice(0, 1),
                        tol,
                        CONTINUES_WITHIN,
                    );
                }
                let tol = if windowed
                    || moved_by_the_held_mass(f.name)
                    || moved_by_the_mean_weight(schema, f.name)
                {
                    tol.max(TOL)
                } else {
                    tol
                };
                same_within(
                    &format!("{what}: {name}"),
                    got,
                    &want,
                    tol,
                    CONTINUES_WITHIN,
                );
            }
        }
    }
}

/// The kinds whose frames hold rows of weight 0, which a build before PLAN
/// 539 counted in the clock (a step of every clock, a row of the row clock,
/// a gap of its own under `gap_cap`, a session's and a step back's rule,
/// and held under `label_clock`), and which this build counts in none (hard
/// rule 9; the user, 2026-11-21: "the next row should decay as if they
/// were not present"). Whether a state saved by such a build continues
/// under the new rule is the user's decision (PLAN 539, step 4), so the
/// kept states are held to what was measured, not to a rule: each loads,
/// saves bytes that load back the same, and its continuation parts from
/// its frozen one, which the old rule wrote, by no more than the bound
/// beside its kind, relative to `1 + |frozen|` over every number of every
/// frame (`RHEON_PRINT_MOVES=1` prints each set's figure), and in no more
/// than the count of other cells -- a null, a flag, a count -- beside it.
/// Measured 2026-11-21 on macOS arm64. The moves are those of the
/// continued rows' own zero-weight rows as much as the saved clock's: the
/// old build counted both. The current fixtures were regenerated.
///
/// Widened 2026-11-23 (rule 5's twenty-fifth exception, the user: "Widen
/// exception 25"): a row of weight 0 under `embargo` or `label_clock` is now
/// scored as `predict` scores it and releases no held row (review
/// 2026-11-23, STRM-1), so a kept state whose saved rows wait on such a row
/// continues further from its frozen frame. Each bound is the figure
/// measured on macOS arm64 that day, rounded up, and `bank_gated` parts in
/// two other cells of its `gated` frame. The schema-71 set, written by the build
/// that counted such rows in no clock but released at them, still converts
/// to the current bytes; its frozen frames were rewritten to what its
/// unchanged state now continues to.
const ZERO_WEIGHT_CLOCK: &[(&str, f64, usize)] = &[
    // 3.77e-2, `int_clock`'s `pred_y`; `dated` 2.28e-3.
    ("bank", 0.04, 0),
    // 1.67e-1, and 2 other cells (was 1.67e-1 and none).
    ("bank_gated", 0.17, 2),
    // 4.11e-2 for `v67/` and `v68/` (was 2.86e-3), 1.64e-1 for `v69/` and
    // `v70/`.
    ("bank_label_clock", 0.17, 0),
    // 1.79e-2 for `v67/` and `v68/` (was 2.86e-3), 1.17e-1 for `v69/` and
    // `v70/` (was 4.53e-2).
    ("bank_conformal_auto", 0.12, 0),
];

/// [`ZERO_WEIGHT_CLOCK`]'s hold on the kept fixture `f` of `kind`.
fn zero_weight_clock_rule(kind: &Kind, f: &Fixture, what: &str) {
    let input = from_ipc(f.input);
    let (again, frames) =
        (kind.run)(&unhex(f.state), &input).unwrap_or_else(|e| panic!("{what}: {e}"));
    let (resaved, _) = (kind.run)(&again, &input).unwrap_or_else(|e| panic!("{what}: {e}"));
    assert!(resaved == again, "{what}: saved, loaded, saved");
    let &(_, bound, cells) = ZERO_WEIGHT_CLOCK
        .iter()
        .find(|z| z.0 == f.name)
        .expect("listed");
    assert_eq!(frames.len(), f.output.len(), "{what}");
    let (mut most, mut other, mut moved) = (0.0f64, 0usize, Vec::new());
    for ((name, got), (frozen_name, want)) in frames.iter().zip(f.output) {
        assert_eq!(name, frozen_name, "{what}");
        let want = from_ipc(want);
        for c in want.columns() {
            let g = got
                .column(c.name())
                .unwrap_or_else(|_| panic!("{what}: {name}.{} is gone", c.name()));
            let (m, o) = parted_by(g.as_materialized_series(), c.as_materialized_series());
            if m > 0.0 || o > 0 {
                moved.push(format!("{name}.{}: {m:.2e}, {o} other", c.name()));
            }
            most = most.max(m);
            other += o;
        }
    }
    if std::env::var_os("RHEON_PRINT_MOVES").is_some() {
        eprintln!(
            "{what}: at most {most:.3e} of 1 + |frozen|, {other} other cells: {}",
            moved.join("; ")
        );
    }
    assert!(
        most <= bound && other <= cells,
        "{what}: the continuation parts from its frozen one by {most:.3e} of 1 + |frozen| \
         (bound {bound}) and in {other} other cells (bound {cells}): {}",
        moved.join("; ")
    );
}

/// How far `got` parts from `want`: the largest `|a − b| / (1 + |b|)` over
/// the numbers both hold, and the count of other cells that differ (a null
/// on one side, a NaN, a flag, a count, a label), a struct by its fields.
fn parted_by(got: &Series, want: &Series) -> (f64, usize) {
    if let (DataType::Struct(_), DataType::Struct(_)) = (got.dtype(), want.dtype()) {
        let (g, w) = (got.struct_().unwrap(), want.struct_().unwrap());
        return g
            .fields_as_series()
            .iter()
            .zip(w.fields_as_series())
            .map(|(a, b)| parted_by(a, &b))
            .fold((0.0, 0), |(m, o), (a, b)| (m.max(a), o + b));
    }
    match (got.dtype(), want.dtype()) {
        (DataType::Float64, DataType::Float64) => {
            let (g, w) = (got.f64().unwrap(), want.f64().unwrap());
            g.iter()
                .zip(w.iter())
                .fold((0.0, 0), |(m, o), (a, b)| match (a, b) {
                    (Some(a), Some(b)) if a.is_finite() && b.is_finite() => {
                        (m.max((a - b).abs() / (1.0 + b.abs())), o)
                    }
                    (Some(a), Some(b)) if a.to_bits() == b.to_bits() => (m, o),
                    (Some(a), Some(b)) if a.is_nan() && b.is_nan() => (m, o),
                    (None, None) => (m, o),
                    _ => (m, o + 1),
                })
        }
        (DataType::List(_), DataType::List(_)) => {
            let (g, w) = (got.list().unwrap(), want.list().unwrap());
            (0..want.len().min(got.len())).fold((0.0, 0), |(m, o), i| {
                match (g.get_as_series(i), w.get_as_series(i)) {
                    (Some(a), Some(b)) if a.len() == b.len() => {
                        let (x, y) = parted_by(&a, &b);
                        (m.max(x), o + y)
                    }
                    (None, None) => (m, o),
                    _ => (m, o + 1),
                }
            })
        }
        _ => {
            let (g, w) = (
                got.cast(&DataType::String).unwrap(),
                want.cast(&DataType::String).unwrap(),
            );
            let o = g
                .str()
                .unwrap()
                .iter()
                .zip(w.str().unwrap().iter())
                .filter(|(a, b)| a != b)
                .count();
            (0.0, o)
        }
    }
}

/// The window-run fixtures whose continuation a later change moved within the
/// golden tolerance (rule 5's fifteenth exception, the user's, 2026-11-17,
/// task 457): a row's held mass is taken from the discount's `exp2` from a
/// hundredth of a half-life up, so a kept state continues, after its load,
/// within about 1e-13 relative of the frames it was frozen with, not to the
/// bit. The current set is held to the bit as ever.
/// The kinds whose banks hold a `huber` or `quantile` under `standardize`
/// (rule 5's twenty-ninth exception, the user's "Exception 29: start
/// fresh", 2026-11-23; review 2026-11-23, REG-18). The model now keeps each
/// target's features' moments over every row, which a state of 71 or
/// earlier holds none of: it starts them at its first row after the load,
/// its scales the band's until they give one. So the state it saves after
/// its continuation is not the current fixture's: it is held to saving
/// bytes that load back the same. Its frames go on to the bit of its frozen
/// ones (measured 2026-11-23 on macOS arm64: the fresh start moves no cell
/// of these frames over their rows). The current fixtures were
/// regenerated.
const STARTED_FRESH: &[&str] = &["bank_diagnosed", "bank_refit", "bank_refit_64"];

fn started_fresh(name: &str, schema: u32) -> bool {
    schema < 72 && STARTED_FRESH.contains(&name)
}

fn moved_by_the_held_mass(name: &str) -> bool {
    name.starts_with("with_windows")
}

/// The fixtures of schema 71 or earlier whose conformal band's mean weight
/// a later change moved (rule 5's thirtieth exception, the user's,
/// 2026-11-23: "Exception 30, full fix"): the band's mean weight is taken
/// at the old mean's and the row's shares, `(R/R')·w̄ + (1/R')·w`, where the
/// Welford step `w̄ + (w − w̄)/R'` cancelled, so a kept state's mean weight
/// holds the old step's last bits. It saves bytes that load back the same,
/// and its continuation is held within [`TOL`] (1e-12 relative) of the
/// frames it was frozen with, not to the bit. The current set is held to
/// the bit as ever. The kinds are [`MEAN_WEIGHT_MOVED`]'s, the kept ones
/// whose band's rows have uneven weights.
const MEAN_WEIGHT_MOVED: &[&str] = &["bank_conformal_auto", "bank_label_clock"];

/// The fixtures of schema 71 or earlier whose breaks or coverage under a
/// horizon read a long-run variance (rule 5's thirty-first exception, the
/// user's, 2026-11-23): where the null's lag 0 left that variance at or below
/// 0, the old build read `None`, and this build reads the fallback, the
/// terms' own lag 0, which a kept state starts at the null's `Σ ω²` and then
/// decays. Every row whose frozen value is a number reads it to the bit, as
/// the null's form is unchanged; a row the old build left `None` in
/// [`FALLBACK_FIELDS`] is held only to a finite number
/// ([`fallback_cells`]).
fn seeded_lag0(schema: u32, name: &str) -> bool {
    schema <= 71 && name == "bank_diagnosed"
}

/// The fields a long-run variance divides: the CUSUM (not the CUSUM of
/// squares, whose lag 0 is the measured fourth moment's), the break Wald
/// distance, and a quantile fit's coverage Wald.
const FALLBACK_FIELDS: &[&str] = &["cusum_", "break_wald_", "calibration_wald_"];

/// `want` with each [`FALLBACK_FIELDS`] cell that `kept`, the kept fixture's
/// own frozen frame, left null taken from `got` where that is a finite
/// number: the rows where the old build read no variance.
fn fallback_cells(kept: &DataFrame, want: &DataFrame, got: &DataFrame) -> DataFrame {
    let falls_back =
        |n: &str| FALLBACK_FIELDS.iter().any(|p| n.starts_with(p)) && !n.starts_with("cusum_sq_");
    let columns: Vec<Column> = want
        .columns()
        .iter()
        .map(|w| {
            let ws = w.as_materialized_series();
            let (DataType::Struct(_), Ok(k), Ok(g)) =
                (ws.dtype(), kept.column(w.name()), got.column(w.name()))
            else {
                return w.clone();
            };
            let (wst, kst, gst) = (
                ws.struct_().unwrap(),
                k.as_materialized_series().struct_().unwrap().clone(),
                g.as_materialized_series().struct_().unwrap().clone(),
            );
            let fields: Vec<Series> = wst
                .fields_as_series()
                .into_iter()
                .map(|f| {
                    let (Some(kf), Some(gf)) = (
                        kst.field_by_name(f.name()).ok(),
                        gst.field_by_name(f.name()).ok(),
                    ) else {
                        return f;
                    };
                    if !falls_back(f.name()) || kf.len() != f.len() || gf.len() != f.len() {
                        return f;
                    }
                    let finite = gf
                        .f64()
                        .map(|c| c.is_not_nan() & c.is_not_null())
                        .unwrap_or_else(|_| gf.is_not_null());
                    let take = kf.is_null() & finite;
                    gf.zip_with(&take, &f).unwrap()
                })
                .collect();
            StructChunked::from_series(ws.name().clone(), ws.len(), fields.iter())
                .unwrap()
                .with_outer_validity(wst.rechunk_validity())
                .into_series()
                .into_column()
        })
        .collect();
    DataFrame::new(want.height(), columns).unwrap()
}

fn moved_by_the_mean_weight(schema: u32, name: &str) -> bool {
    schema <= 71 && MEAN_WEIGHT_MOVED.contains(&name)
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

/// Whether a bank's refit buffers keep each row's `|e|` (`resid`, a list)
/// or a histogram holds its deviations (`held`, with its `pending` decay):
/// the forms a bank of schema 64 has none of.
fn holds_for_a_refit(bytes: &[u8]) -> bool {
    fn any(v: &rmpv::Value) -> bool {
        match v {
            rmpv::Value::Map(m) => m.iter().any(|(k, v)| {
                // A refit buffer's: beside its `learned` count.
                (k.as_str() == Some("resid")
                    && v.is_array()
                    && m.iter().any(|(k, _)| k.as_str() == Some("learned")))
                    || (k.as_str() == Some("held")
                        && v.as_map()
                            .is_some_and(|h| h.iter().any(|(k, _)| k.as_str() == Some("pending"))))
                    || any(v)
            }),
            rmpv::Value::Array(a) => a.iter().any(any),
            _ => false,
        }
    }
    any(&rmpv::decode::read_value(&mut &bytes[..]).unwrap())
}

/// A kept bank of schema 64, `kept`, converted to `saved`, whose scales the
/// old rule kept (the doc above `previous`): every anchored sketch is
/// marked to keep anchoring at its first value and nothing else is; no
/// buffer or histogram holds an `|e|` or deviations for a refit; and the
/// bank saves bytes that load back to the same.
fn kept_scale_rule(kind: &Kind, kept: &rmpv::Value, saved: &[u8], input: &DataFrame, what: &str) {
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
        "{what}: a kept bank carries no mark"
    );
    assert!(
        !holds_for_a_refit(saved),
        "{what}: held for a refit by the new rule"
    );
    let (again, _) = (kind.run)(saved, input).unwrap_or_else(|e| panic!("{what}: {e}"));
    assert!(again == saved, "{what}: saved, loaded, saved");
}

/// The fields whose numbers a fix moved after a kept schema's fixtures were
/// written, each a declared output change: a kept fixture goes on to the
/// current fixture's numbers in them -- the same stream, saved at the same
/// row, fed the same input -- and to its own frozen numbers in every other
/// field. A field is named by its prefix. `cusum_<t>` and `cusum_sq_<t>`
/// (task 278, D-B5): the windowed CUSUMs' null share is read as far as
/// their sums have filled, where it was the filled share from the first
/// row.
const MOVED_SINCE: &[&str] = &["cusum_"];

/// The specs whose outputs a later change moved within a bound while their
/// state converts exactly, each with the bound, by the golden measure
/// `|got − frozen| ≤ bound·(1 + |frozen|)`: the kept bank fixtures' `ew_cov`
/// specs (rule 5's sixth exception, the user's, 2026-11-01, task 294). The
/// co-moment matrix is kept as its lower triangle, so a covariance or
/// correlation of pair `(i, j)`, `i < j`, reads entry `(j, i)`, where the
/// frozen frames read entry `(i, j)`, rounded apart from it; the state
/// loads as its lower triangle mirrored, the triangle every build computed
/// alike, and saves the current fixture's bytes. Only the fields that read
/// an upper entry are held to the bound; every other field, the
/// predictions among them, goes on to the bit (task 316, review 11's
/// S2-2): an `ew_cov` spec's `corr_` and `cov_` fields (its means,
/// standard deviations and component read the lower triangle only), a
/// closed group's `comoments`, the matrix's upper triangle row by row
/// (1.3e-17 here), and the `diagnosed` spec's specification
/// (`ljung_box_`, `breusch_pagan_`, `ramsey_reset_`), calibration
/// (`calibration_`), break (`break_wald_`) and `ic_` statistics, which
/// solve from one triangle of such a matrix and read their cross terms
/// from the other: held to the golden tolerance, as the rule says
/// (`breusch_pagan_y` moved by 4.4e-16 and `ramsey_reset_y` by 7.7e-17
/// on the first row, measured 2026-11-01; the rows after it are the
/// window's, held to the golden tolerance already).
///
/// And the `corr_window` spec's `crit` in the kept `bank_diagnosed` files
/// (rule 5's seventeenth exception, the user's, 2026-11-17): the permutation
/// critical value is drawn by Lemire's multiply-shift reduction in place of
/// a remainder, so a kept state's redraws, from its saved generator, give
/// other values than the frozen frames', by 0.063 of `1 + |crit|` at most
/// over the rows after the save (measured 2026-11-18, `v65/` and `v66/`
/// alike, as the kept set against the regenerated one). The bound, 0.25, is
/// under the 0.5 a flipped `flag` would read (a field not named here, so
/// held to the bit): every other field of the spec goes on to the bit.
const CONTINUES_WITHIN: &[Within] = &[
    ("comoments", &[], 1e-15),
    (
        "diagnosed",
        &[
            "ljung_box_",
            "breusch_pagan_",
            "ramsey_reset_",
            "calibration_",
            "break_wald_",
            "ic_",
        ],
        TOL,
    ),
    // Rule 5's eighteenth exception (the user's, 2026-11-18, task 469): the
    // break distance is formed from the difference of the fits' inverse
    // Grams, so every kept fixture's continuation parts from its frozen
    // `break_wald_` by rounding (5.5e-16 relative here, on rows well past
    // the `2k` wait), held to the golden tolerance.
    ("quantiled", &["break_wald_"], TOL),
    ("closing", COV_FIELDS, 1e-15),
    ("windowed", COV_FIELDS, 1e-15),
    ("uint_clock", COV_FIELDS, 1e-15),
    ("sessioned", COV_FIELDS, 1e-15),
    ("components", COV_FIELDS, 1e-15),
    ("corr_window", &["crit"], 0.25),
];

/// The fields of an `ew_cov` spec that read an upper entry of its
/// co-moment matrix ([`CONTINUES_WITHIN`]).
const COV_FIELDS: &[&str] = &["corr_", "cov_"];

/// The columns a frozen frame has gained since a kept schema: `summary()`'s
/// counts of labels missing and dropped (schema 67, task 341), which a kept
/// fixture's frame was written without. A kept frame takes each from the
/// current fixture's frame of the same name, where that has it.
const ADDED_SINCE: &[&str] = &["labels_missing", "labels_dropped"];

/// `kept`, a kept fixture's frozen frame, with the struct fields that
/// [`MOVED_SINCE`] names taken from `now`, the current fixture's frame of
/// the same name, and the columns [`ADDED_SINCE`] names that it lacks
/// taken from `now` at their place there.
fn moved_since(kept: &DataFrame, now: &DataFrame) -> DataFrame {
    let kept = &added_since(kept, now);
    let moved = |name: &str| MOVED_SINCE.iter().any(|p| name.starts_with(p));
    let columns: Vec<Column> = kept
        .columns()
        .iter()
        .map(|k| {
            let ks = k.as_materialized_series();
            let (DataType::Struct(_), Some(ns)) = (
                ks.dtype(),
                now.column(k.name())
                    .ok()
                    .map(|c| c.as_materialized_series()),
            ) else {
                return k.clone();
            };
            let (kst, nst) = (ks.struct_().unwrap(), ns.struct_().unwrap());
            let fields: Vec<Series> = kst
                .fields_as_series()
                .into_iter()
                .map(|f| {
                    if moved(f.name()) {
                        nst.field_by_name(f.name()).unwrap_or(f)
                    } else {
                        f
                    }
                })
                .collect();
            StructChunked::from_series(ks.name().clone(), ks.len(), fields.iter())
                .unwrap()
                .with_outer_validity(kst.rechunk_validity())
                .into_series()
                .into_column()
        })
        .collect();
    DataFrame::new(kept.height(), columns).unwrap()
}

/// [`moved_since`]'s columns gained: `kept` with each column
/// [`ADDED_SINCE`] names that `now` has and `kept` lacks, in `now`'s order.
fn added_since(kept: &DataFrame, now: &DataFrame) -> DataFrame {
    let lacks = |n: &str| ADDED_SINCE.contains(&n) && kept.column(n).is_err();
    if !now.columns().iter().any(|c| lacks(c.name())) || now.height() != kept.height() {
        return kept.clone();
    }
    let columns: Vec<Column> = now
        .columns()
        .iter()
        .filter_map(|c| {
            if lacks(c.name()) {
                Some(c.clone())
            } else {
                kept.column(c.name()).ok().cloned()
            }
        })
        .collect();
    DataFrame::new(kept.height(), columns).unwrap()
}

/// The window runs' `finish` frames, in which a window that looks ahead
/// whose far edge is the input's last row is whole since PLAN 565 and was
/// null before it. The windows' state is the same bytes, so a kept state
/// continues to the same numbers but for those cells: `kept` (a kept
/// fixture's frame `frame` of kind `kind`) takes `now`'s value, the
/// current fixture's, in each cell it holds null where `now` holds a value,
/// in the column `ahead`, and is the same everywhere else.
fn whole_at_the_end_of_the_input(
    kind: &str,
    frame: &str,
    kept: &DataFrame,
    now: &DataFrame,
) -> DataFrame {
    if !kind.starts_with("with_windows") || frame != "finish" || kept.height() != now.height() {
        return kept.clone();
    }
    let (Ok(k), Ok(n)) = (kept.column("ahead"), now.column("ahead")) else {
        return kept.clone();
    };
    let (Ok(k), Ok(n)) = (
        k.as_materialized_series().f64(),
        n.as_materialized_series().f64(),
    ) else {
        return kept.clone();
    };
    let taken: Float64Chunked = k
        .iter()
        .zip(n.iter())
        .map(|(k, n)| if k.is_none() { n } else { k })
        .collect_ca(k.name().clone());
    let mut out = kept.clone();
    out.replace("ahead", taken.into_column()).unwrap();
    out
}

/// The groups a kind's continuation first sees, after the load: each is a
/// stream this build makes from its spec, under its own rules, in the
/// kept fixture's continuation and the current one's alike, so where a
/// rule moved since the kept schema (`kept_scale_rule`) its rows are held
/// to the current fixture's numbers: `bank_refit`'s group `e`, from row
/// 72 of the 96, whose MAD before its refit is read exactly from its
/// buffer, where the 64 build read the sketch.
const NEW_AFTER_LOAD: &[(&str, &str, &[&str])] = &[("bank_refit", "g", &["e"])];

/// `want`, a kept fixture's frame `frame` (with [`moved_since`] applied),
/// with the rows of the groups [`NEW_AFTER_LOAD`] names for kind `kind`
/// taken from `now`, the current fixture's frame of the same name; `input`
/// is the continuation's input, whose rows the frame's are.
fn new_after_load(
    kind: &str,
    frame: &str,
    input: &DataFrame,
    want: &DataFrame,
    now: &DataFrame,
) -> DataFrame {
    let Some((_, col, groups)) = NEW_AFTER_LOAD.iter().find(|(k, _, _)| *k == kind) else {
        return want.clone();
    };
    if frame != "fit_predict" {
        return want.clone();
    }
    let keys = input.column(col).unwrap().str().unwrap();
    let new: BooleanChunked = keys
        .iter()
        .map(|g| Some(g.is_some_and(|g| groups.contains(&g))))
        .collect();
    assert!(new.any(), "{kind}: a group new after the load");
    let columns: Vec<Column> = want
        .columns()
        .iter()
        .map(|k| {
            let n = now.column(k.name()).unwrap();
            // A kept frame's `withheld_reason` names are a prefix of today's
            // (`same_series`): read in today's dtype, as its values are.
            let k = k.as_materialized_series().cast(n.dtype()).unwrap();
            n.as_materialized_series()
                .zip_with(&new, &k)
                .unwrap()
                .into_column()
        })
        .collect();
    DataFrame::new(want.height(), columns).unwrap()
}

/// Whether a saved state holds a window's snapshots: a non-empty `ring`
/// under a `snaps` anywhere in it.
fn holds_a_window(bytes: &[u8]) -> bool {
    fn find(v: &rmpv::Value) -> bool {
        match v {
            rmpv::Value::Map(m) => m.iter().any(|(k, v)| {
                (k.as_str() == Some("snaps")
                    && v.as_map().is_some_and(|m| {
                        m.iter().any(|(k, r)| {
                            k.as_str() == Some("ring")
                                && r.as_array().is_some_and(|r| !r.is_empty())
                        })
                    }))
                    || find(v)
            }),
            rmpv::Value::Array(a) => a.iter().any(find),
            _ => false,
        }
    }
    find(&rmpv::decode::read_value(&mut &bytes[..]).unwrap())
}

/// Every leaf of a msgpack value: its dotted path and its value.
fn leaves(v: &rmpv::Value, path: &str, out: &mut Vec<(String, rmpv::Value)>) {
    match v {
        rmpv::Value::Map(m) => {
            for (k, v) in m {
                let key = k.as_str().map_or_else(|| k.to_string(), str::to_string);
                leaves(v, &format!("{path}.{key}"), out);
            }
        }
        rmpv::Value::Array(a) => {
            for (i, v) in a.iter().enumerate() {
                leaves(v, &format!("{path}.{i}"), out);
            }
        }
        _ => out.push((path.to_string(), v.clone())),
    }
}

/// The forms a schema moved in the bank's states are each written by a
/// fixture, so a change to their tags or layouts cannot pass the harness
/// unseen (review round 5, B1). The integer clock's (docs/PLAN.md task
/// 200, schema 42 and windows state 8): the bank's `ClockValue::I64`,
/// `Stamp::Int`, the three `Ticks` fields -- skipped when zero, so written
/// only by a state saved while they hold time -- a fraction in one of them,
/// `ClockDtype::Int` at two widths, and the per-group PCA continuity; the
/// windows state's `OffForm::Int`, its group's stamp as a
/// `ClockValue::I64` and an increment's integer previous value; the refresh
/// state's `Instant::Int`. And the readiness notices' waits (schema 46,
/// task 208), skipped while none runs, so written only by a stream saved
/// inside one: a target's `floor_since` and the noise gate's `gate_since`,
/// each a decay time.
#[test]
fn every_form_a_schema_moved_is_written_by_a_fixture() {
    if regenerating() {
        return;
    }
    type Holds = fn(&str, &rmpv::Value) -> bool;
    const FORMS: &[(&str, &str, Holds)] = &[
        ("bank", "ClockValue::I64", |p, _| {
            p.ends_with(".prev_clock.I64")
        }),
        ("bank", "Stamp::Int", |p, _| p.contains(".stamp.Int.")),
        ("bank", "Ticks: removed_int", |p, _| {
            p.ends_with(".removed_int.whole")
        }),
        ("bank", "Ticks: elapsed_removed_int", |p, _| {
            p.ends_with(".elapsed_removed_int.whole")
        }),
        ("bank", "Ticks: pending_int", |p, _| {
            p.ends_with(".pending_int.whole")
        }),
        ("bank", "a Ticks fraction", |p, v| {
            p.ends_with("_int.frac") && v.as_f64().is_some_and(|f| f != 0.0)
        }),
        ("bank", "ClockDtype::Int(I64)", |p, v| {
            p.contains(".clock_dtypes.") && p.ends_with(".Int") && v.as_str() == Some("I64")
        }),
        ("bank", "ClockDtype::Int(U32)", |p, v| {
            p.contains(".clock_dtypes.") && p.ends_with(".Int") && v.as_str() == Some("U32")
        }),
        ("bank", "pca_prev_by_group", |p, _| {
            p.starts_with(".pca_prev_by_group.")
        }),
        // Schema 64 (docs/PLAN.md task 277): an instance restarted by a
        // drift reset and not yet past its next row.
        ("bank_reset", "a restart mark", |p, v| {
            p.contains(".restarted.") && v.as_bool() == Some(true)
        }),
        // Schema 64 (docs/PLAN.md task 277): a spec's refit reservation,
        // a stream given up to the refit budget, and a target's buffer
        // before its refit.
        ("bank_refit", "a refit reservation", |p, v| {
            p.starts_with(".refit_reserved.") && v.as_f64().is_some_and(|f| f > 0.0)
        }),
        ("bank_refit", "a stream over the refit budget", |p, v| {
            p.ends_with(".refit_over_budget") && v.as_bool() == Some(true)
        }),
        // Schema 65 (task 323): a buffer's one-pass `|e|` beside each row
        // while the refit waits, and a sketch on the grid fixed at 1 (a
        // stream over the budget, its refit given up before a row); a 64
        // bank's buffer holding none, and its sketches kept at their first
        // value.
        ("bank_refit", "a refit buffer's |e|", |p, v| {
            p.contains(".resid.") && v.as_f64().is_some_and(|f| f > 0.0)
        }),
        ("bank_refit", "a sketch on the grid fixed at 1", |p, v| {
            p.ends_with(".anchor") && v.as_f64() == Some(1.0)
        }),
        // Schema 67 (task 340): the notices a failed call left in a stream,
        // one naming the closed span it was raised in, one its body alone.
        ("bank_notice_pending", "a pending notice's body", |p, v| {
            p.contains(".pending.")
                && p.ends_with(".body")
                && v.as_str().is_some_and(|s| s.contains("support_coef"))
        }),
        ("bank_notice_pending", "a pending notice's span", |p, v| {
            p.contains(".pending.") && p.ends_with(".span") && v.as_str() == Some(" session \"a\"")
        }),
        // Schema 67 (task 341): rows held for their label times, on an
        // integer clock and a temporal one; the clock of the labels
        // learned; and the counts of labels missing and dropped. The
        // buffer's order by label time is held by
        // `a_label_buffer_is_frozen_out_of_arrival_order`.
        (
            "bank_label_clock",
            "a held row's integer label time",
            |p, _| p.contains(".pending.") && p.ends_with(".label.I64"),
        ),
        (
            "bank_label_clock",
            "a held row's temporal label time",
            |p, _| p.contains(".pending.") && p.ends_with(".label.Ns"),
        ),
        (
            "bank_label_clock",
            "the clock of the labels learned",
            |p, _| p.contains(".label_learned."),
        ),
        ("bank_label_clock", "a count of labels missing", |p, v| {
            p.ends_with(".labels_missing") && v.as_u64().is_some_and(|n| n > 0)
        }),
        ("bank_label_clock", "a count of labels dropped", |p, v| {
            p.ends_with(".labels_dropped") && v.as_u64().is_some_and(|n| n > 0)
        }),
        // Schema 67 (task 342b): a conformal tracker's mean of the rows
        // waiting at each release, under `conformal_rate = "auto"`.
        (
            "bank_conformal_auto",
            "a conformal mean of the rows held",
            |p, v| p.ends_with(".held_mean") && v.as_f64().is_some_and(|f| f > 0.0),
        ),
        // Schema 67 (task 368): a conformal tracker waiting for its start,
        // the rows it has been given counted.
        (
            "bank_conformal_auto",
            "a conformal tracker's count of the rows before its start",
            |p, v| p.ends_with(".released") && v.as_f64().is_some_and(|f| f > 0.0),
        ),
        // Schema 66 (task 328): the rows a window's ring has taken.
        (
            "bank_diagnosed",
            "a window ring's count of its rows",
            |p, v| p.ends_with(".ring_rows") && v.as_u64().is_some_and(|n| n > 0),
        ),
        (
            "bank_refit_64",
            "a buffer of schema 64, with no |e|",
            |p, v| p.ends_with(".resid") && v.is_nil(),
        ),
        (
            "bank_refit_64",
            "a sketch kept at its first value",
            |p, v| p.ends_with(".anchor_by_first_value") && v.as_bool() == Some(true),
        ),
        // Schema 64 (task 278, batch 17): the spread folding the slots' own
        // predictions, the autocorrelation weighed by the rows, and a held
        // row's record carrying the prediction a gate withheld: one slot an
        // instance, so a second value in a record is that block.
        ("bank_gated", "the spread's flag", |p, v| {
            p.ends_with(".ungated_spread") && v.as_bool() == Some(true)
        }),
        ("bank_gated", "the autocorrelation's flag", |p, v| {
            p.ends_with(".weighted_autocorr") && v.as_bool() == Some(true)
        }),
        // Task 278: `ewridge`'s `min_weight` withholding the output alone,
        // and the drift detector folding what a gate withholds.
        ("bank_gated", "ewridge's output-only floor", |p, v| {
            p.ends_with(".min_weight_output_only") && v.as_bool() == Some(true)
        }),
        // PLAN 547: `bocpd`'s per-row count in a spec, and a number hazard
        // on a number clock read on the clock.
        ("bank_bocpd_hazard", "a spec's hazard_rows", |p, v| {
            p.ends_with(".hazard_rows") && v.as_f64() == Some(30.0)
        }),
        (
            "bank_bocpd_hazard",
            "a number hazard on the clock",
            |p, v| p.ends_with(".cfg.hazard_on_clock") && v.as_bool() == Some(true),
        ),
        // PLAN 538: a refit over a span of the clock, in a spec and in each
        // model's buffer, with its first stamp.
        ("bank_refit_span", "a spec's max_state_bytes", |p, v| {
            p.ends_with(".max_state_bytes") && v.as_u64() == Some(200_000)
        }),
        ("bank_refit_span", "a spec's refit_span", |p, v| {
            p.ends_with(".refit_span") && v.as_f64() == Some(60.0)
        }),
        ("bank_refit_span", "a model's refit_clock", |p, v| {
            p.ends_with(".cfg.refit_clock") && v.as_f64() == Some(60.0)
        }),
        ("bank_refit_span", "a glm buffer's first stamp", |p, v| {
            p.contains(".Glm.refit.") && p.contains(".first.") && !v.is_nil()
        }),
        (
            "bank_refit_span",
            "an expectile buffer's first stamp",
            |p, v| p.contains(".Expectile.refit.") && p.contains(".first.") && !v.is_nil(),
        ),
        // PLAN 508: each drift detector other than Page-Hinkley, under its
        // name, each holding what its kind keeps.
        ("bank_drift_kinds", "a CUSUM's sums", |p, v| {
            p.contains(".Cusum.")
                && (p.ends_with(".up") || p.ends_with(".down"))
                && v.as_f64().is_some_and(|f| f > 0.0)
        }),
        ("bank_drift_kinds", "an ADWIN's second level", |p, v| {
            p.contains(".Adwin.levels.1.") && v.as_f64().is_some()
        }),
        // Rule 5's thirty-first exception: under a horizon the breaks' and
        // a quantile fit's coverage's empirical lag 0.
        ("bank_diagnosed", "a CUSUM's empirical lag 0", |p, v| {
            p.ends_with(".g1") && v.as_f64().is_some_and(|f| f > 0.0)
        }),
        (
            "bank_diagnosed",
            "a quantile fit's residuals' lag 0",
            |p, v| p.ends_with(".gz") && v.as_f64().is_some_and(|f| f > 0.0),
        ),
        ("bank_diagnosed", "a coverage's empirical lag 0", |p, v| {
            p.ends_with(".g") && v.as_f64().is_some_and(|f| f > 0.0)
        }),
        ("bank_drift_ranges", "an ADWIN's fixed range", |p, v| {
            p.contains(".Adwin.") && p.ends_with(".range") && v.as_f64() == Some(0.5)
        }),
        ("bank_drift_ranges", "an ADWIN bucket's extent", |p, v| {
            p.contains(".Adwin.extents.1.") && v.as_f64().is_some()
        }),
        ("bank_drift_kinds", "a DDM's least rate", |p, v| {
            p.contains(".Ddm.detector.")
                && p.ends_with(".p_min")
                && v.as_f64().is_some_and(f64::is_finite)
        }),
        ("bank_drift_kinds", "a DDM's error cut", |p, v| {
            p.contains(".Ddm.") && p.ends_with(".error_above") && v.as_f64() == Some(0.15)
        }),
        ("bank_drift_kinds", "an EDDM's distances", |p, v| {
            p.contains(".Eddm.detector.")
                && p.ends_with(".p2s_max")
                && v.as_f64().is_some_and(|f| f > 0.0)
        }),
        ("bank_drift_kinds", "a KSWIN's ring", |p, v| {
            p.contains(".Kswin.window.") && v.as_f64().is_some()
        }),
        ("bank_drift_kinds", "a KSWIN's generator", |p, v| {
            p.contains(".Kswin.rng") && v.as_u64().is_some_and(|n| n != 7)
        }),
        ("bank_reset", "the drift detector's flag", |p, v| {
            p.ends_with(".ungated_drift") && v.as_bool() == Some(true)
        }),
        // Schema 64 (review round 10): the rows a restarted instance has
        // learned, which bound its notices' rounding (S7-7); under a drift
        // reset, the clock a fresh stream begun at a skipped row keeps, and
        // a held row's step as that stream's first (S7-3); and an `sgd`
        // instance that has said its rate is past its bound (S7-4).
        ("bank_reset", "the rows learned since a restart", |p, v| {
            p.ends_with(".learned_since_restart") && v.as_u64().is_some()
        }),
        ("bank_reset_held", "a fresh stream's clock", |p, _| {
            p.contains(".restart_fold.")
        }),
        ("bank_reset_held", "a held row's fresh step", |p, v| {
            p.contains(".pending.")
                && p.ends_with(".d_fresh")
                && v.as_f64().is_some_and(|f| f > 0.0)
        }),
        ("bank_unstable", "an unstable rate said", |p, v| {
            p.contains(".notified.") && p.ends_with(".unstable") && v.as_bool() == Some(true)
        }),
        (
            "bank_gated",
            "a record with the withheld prediction",
            |p, v| {
                p.contains(".score_pred.")
                    && p.ends_with(".1")
                    && v.as_f64().is_some_and(f64::is_finite)
            },
        ),
        ("with_windows_int", "OffForm::Int", |p, v| {
            p.ends_with(".form") && v.as_str() == Some("Int")
        }),
        ("with_windows_int", "ClockValue::I64", |p, _| {
            p.ends_with(".prev_clock.I64")
        }),
        (
            "with_windows_int",
            "the group's stamp as ClockValue::I64",
            |p, _| p.ends_with(".stamp.I64"),
        ),
        // An `i128` rides as sixteen bytes, so the leaf is a binary, not an
        // integer: present is what is checked.
        ("with_windows_int", "an increment's prev_int", |p, v| {
            p.contains(".prev_int.") && !v.is_nil()
        }),
        // Task 212: a variance's queue, its arenas six wide, and an
        // operator's `bias`.
        ("with_windows", "a variance's queue, six wide", |p, v| {
            p.ends_with(".front.wide") && v.as_bool() == Some(true)
        }),
        ("with_windows", "a variance's running sums", |p, v| {
            p.contains(".total.") && v.as_f64().is_some_and(|f| f != 0.0)
        }),
        ("with_windows", "an operator's bias", |p, v| {
            p.ends_with(".bias") && v.as_bool() == Some(true)
        }),
        // Windows state 10 (task 272 (d)): a quantile's level, its running
        // sketch's buckets, and a windowed quantile's `front` rows, each
        // keeping what its bucket held before it.
        ("with_windows_quantile", "an operator's quantile", |p, v| {
            p.ends_with(".quantile") && v.as_f64() == Some(0.9)
        }),
        ("with_windows_quantile", "a sketch's buckets", |p, v| {
            p.contains(".quant.")
                && p.contains(".back.pos.w.")
                && v.as_f64().is_some_and(|f| f > 0.0)
        }),
        (
            "with_windows_quantile",
            "a front row's kept weights",
            |p, v| {
                p.contains(".quant.") && p.contains(".prev.") && v.as_f64().is_some_and(|f| f > 0.0)
            },
        ),
        // PLAN 560: a profile's definition with its zone and spans in
        // nanoseconds or clock units, and a group's buckets, last moved at
        // a temporal or a number clock.
        ("with_windows_profile", "a profile's zone", |p, v| {
            p.ends_with(".zone") && v.as_str() == Some("Europe/Amsterdam")
        }),
        ("with_windows_profile", "a period in nanoseconds", |p, _| {
            p.ends_with(".period.ns")
        }),
        (
            "with_windows_profile",
            "a bucket moved at a temporal clock",
            |p, _| p.contains(".profile.") && p.ends_with(".at.Ns"),
        ),
        (
            "with_windows_profile",
            "a left profile's stamp value",
            |p, v| {
                p.contains(".profile.")
                    && p.ends_with(".left")
                    && v.as_f64().is_some_and(f64::is_finite)
            },
        ),
        (
            "with_windows_profile",
            "a windowed quantile's kept rows",
            |p, _| p.contains(".win.") && p.ends_with(".rows.0.tau"),
        ),
        (
            "with_windows_profile",
            "a front row's kept sketch weights",
            |p, v| {
                p.contains(".win.")
                    && p.contains(".q.prev.")
                    && v.as_f64().is_some_and(f64::is_finite)
            },
        ),
        (
            "with_windows_profile",
            "an unbounded sketch's row count",
            |p, v| p.contains(".win.") && p.ends_with(".n") && v.as_u64().is_some_and(|n| n > 0),
        ),
        (
            "with_windows_profile_units",
            "a window's front moments",
            |p, _| p.contains(".win.") && p.ends_with(".suffix.0.m2"),
        ),
        (
            "with_windows_profile_units",
            "a harmonic window's front Grams",
            |p, v| {
                p.contains(".harm.suffix.")
                    && p.contains(".g.")
                    && v.as_f64().is_some_and(|f| f != 0.0)
            },
        ),
        (
            "with_windows_profile_units",
            "a harmonic profile's phase",
            |p, v| p.ends_with(".phi") && v.as_f64().is_some_and(|f| (0.0..1.0).contains(&f)),
        ),
        (
            "with_windows_profile_units",
            "a profile's harmonics",
            |p, v| p.ends_with(".harmonics") && v.as_u64() == Some(1),
        ),
        (
            "with_windows_profile_units",
            "a profile's min_samples",
            |p, v| p.ends_with(".min_samples") && v.as_u64() == Some(2),
        ),
        (
            "with_windows_profile_units",
            "a period in clock units",
            |p, _| p.ends_with(".period.units"),
        ),
        (
            "with_windows_profile_units",
            "a bucket moved at a number clock",
            |p, _| p.contains(".profile.") && p.ends_with(".at.F64"),
        ),
        // Windows state 13 (task 561): an extremum's `front` rows, its
        // running form's count, a kernel's `since` and its last event's
        // stamp, and an operator's `max_state_bytes`.
        (
            "with_windows_extrema",
            "an extremum's front rows",
            |p, v| {
                p.contains(".ext.")
                    && p.contains(".front.")
                    && v.as_f64().is_some_and(f64::is_finite)
            },
        ),
        (
            "with_windows_extrema",
            "a running extremum's count",
            |p, v| {
                p.contains(".ext.")
                    && (p.ends_with(".run_n") || p.ends_with(".cur_n"))
                    && v.as_f64().is_some_and(|f| f > 0.0)
            },
        ),
        ("with_windows_extrema", "a kernel's since", |p, v| {
            p.ends_with(".since") && v.as_u64().is_some()
        }),
        ("with_windows_extrema", "a queue's last event", |p, v| {
            p.ends_with(".since_off") && v.as_i64().is_some()
        }),
        (
            "with_windows_extrema",
            "an operator's max_state_bytes",
            |p, v| p.ends_with(".max_state_bytes") && v.as_u64() == Some(1_000_000),
        ),
        // A tick sign's carried sign, past the three slots of the
        // previous values (two tick signs and an increment), each +1 or -1.
        ("with_windows_tick", "a tick sign's carried sign", |p, v| {
            p.starts_with(".increments.")
                && (p.ends_with(".prev.3") || p.ends_with(".prev.4"))
                && v.as_f64().is_some_and(|f| f.abs() == 1.0)
        }),
        // Windows state 14 (task 564): a group's row operators' states, a
        // run length's run past its first row and its tolerance, a drift's
        // sums with a spread and the arrival they are centred on, and an
        // increment's place beside them.
        ("with_windows_rows", "a run length's run", |p, v| {
            p.contains(".rows.") && p.ends_with(".Run.n") && v.as_u64().is_some_and(|n| n > 1)
        }),
        ("with_windows_rows", "a run length's tol", |p, v| {
            p.ends_with(".Run.tol") && v.as_f64() == Some(0.5)
        }),
        ("with_windows_rows", "a run length's last value", |p, _| {
            p.contains(".Run.last.Float")
        }),
        ("with_windows_rows", "a drift's sums", |p, v| {
            p.contains(".Drift.sums.")
                && p.ends_with(".c_aa")
                && v.as_f64().is_some_and(|f| f > 0.0)
        }),
        ("with_windows_rows", "a drift's origin", |p, _| {
            p.contains(".Drift.origin.Float")
        }),
        ("with_windows_rows", "a clock step's moments", |p, v| {
            p.contains(".Steps.stats.moments.")
                && p.ends_with(".w_sum")
                && v.as_f64().is_some_and(|f| f > 0.0)
        }),
        ("with_windows_rows", "a clock step's last time", |p, _| {
            p.contains(".Steps.prev.Float")
        }),
        ("with_windows_rows", "a clock step's statistic", |p, v| {
            p.ends_with(".Steps.stats.stat") && v.as_str() == Some("std")
        }),
        ("with_windows_rows", "an increment's place", |p, v| {
            p.contains(".rows.") && v.as_str() == Some("Step")
        }),
        // Windows state 14 (PLAN 563): a formula's invert node, its unknown,
        // and an operator whose input is one.
        ("with_windows_invert", "an invert node", |p, v| {
            p.contains(".formulas.") && v.as_str() == Some("invert")
        }),
        ("with_windows_invert", "the unknown", |p, v| {
            p.contains(".formulas.") && v.as_str() == Some("unknown")
        }),
        (
            "with_windows_invert",
            "an operator over an invert",
            |p, v| p.contains(".formulas.2.tree.1.") && v.as_str() == Some("invert"),
        ),
        // PLAN 541: a merge's state, in each form it takes -- an input's
        // last clock as a float and as an integer (a temporal clock's
        // physical one) and none, the rows held at a tie, an input started
        // again, and the merged rows' last clock in each group, the null
        // key's and the one key of a merge without `by`.
        ("merge", "Stamp::Float", |p, _| {
            p.starts_with(".inputs.") && p.ends_with(".last.Float")
        }),
        ("merge", "an input with no clock", |p, v| {
            p.starts_with(".inputs.") && p.ends_with(".last") && v.is_nil()
        }),
        ("merge", "the rows held at a tie", |p, v| {
            p.starts_with(".inputs.")
                && p.ends_with(".held")
                && v.as_slice().is_some_and(|b| b.starts_with(b"ARROW1"))
        }),
        ("merge", "the one key without by", |p, v| {
            p == ".groups.0.0" && v.as_str() == Some("")
        }),
        ("merge_by", "Stamp::Int", |p, _| {
            p.starts_with(".inputs.") && p.ends_with(".last.Int")
        }),
        ("merge_by", "an input started again", |p, v| {
            p.starts_with(".inputs.") && p.ends_with(".restarts") && v.as_u64() == Some(1)
        }),
        ("merge_by", "the null group", |p, v| {
            p.starts_with(".groups.") && p.ends_with(".0") && v.is_nil()
        }),
        ("merge_by", "a group's last clock", |p, _| {
            p.starts_with(".groups.") && p.ends_with(".1.Int")
        }),
        // PLAN 556: a merge's carried values -- each group's, the null
        // one's too, of a day started again, a column with none yet in a
        // group, and the value as a number and as text -- and a grid's ages:
        // the clock of each seen series' last tick, a number and a temporal
        // one, and each held point's.
        ("merge_carry", "a carried value's clock", |p, _| {
            p.starts_with(".carry.groups.") && p.ends_with(".at.Int")
        }),
        ("merge_carry", "a value of a day started again", |p, v| {
            p.starts_with(".carry.groups.") && p.ends_with(".restarts") && v.as_u64() == Some(1)
        }),
        ("merge_carry", "the null group's values", |p, v| {
            p.starts_with(".carry.groups.")
                && p.ends_with(".0")
                && p.split('.').count() == 5
                && v.is_nil()
        }),
        ("merge_carry", "a column with no value yet", |p, v| {
            let seg: Vec<&str> = p.split('.').collect();
            seg.len() == 6 && seg[1] == "carry" && seg[4] == "1" && v.is_nil()
        }),
        ("merge_carry", "a carried Cell::Float", |p, _| {
            p.starts_with(".carry.groups.") && p.ends_with(".value.Float")
        }),
        ("merge_carry", "a carried Cell::Str", |p, _| {
            p.starts_with(".carry.groups.") && p.ends_with(".value.Str")
        }),
        ("refresh_time_age", "a run with ages", |p, v| {
            p == ".age" && v.as_bool() == Some(true)
        }),
        (
            "refresh_time_age",
            "a series' last tick's Instant::Number",
            |p, _| p.starts_with(".states.") && p.contains(".ats.") && p.ends_with(".Number"),
        ),
        (
            "refresh_time_age",
            "a series unseen since the point, without a clock",
            |p, v| p.starts_with(".states.") && p.contains(".ats.") && v.is_nil(),
        ),
        (
            "refresh_time_age_held",
            "a series' last tick's Instant::Nanos",
            |p, _| p.starts_with(".states.") && p.contains(".ats.") && p.ends_with(".Nanos"),
        ),
        ("refresh_time_age_held", "a held point's clocks", |p, _| {
            p.starts_with(".held.ats.0.") && p.ends_with(".Nanos")
        }),
        // Task 554: a moments operator's term among the core's definitions,
        // and each group's model of it, a window's ring and a lag's buffer.
        (
            "with_windows_moments",
            "a moments operator's term",
            |p, v| {
                p.contains(".moments.")
                    && p.ends_with(".kind")
                    && v.as_str() == Some("standardized")
            },
        ),
        ("with_windows_moments", "a group's moments model", |p, v| {
            p.contains(".model.") && p.ends_with(".w_sum") && v.as_f64().is_some_and(|f| f > 0.0)
        }),
        ("with_windows_moments", "a moments window's ring", |p, v| {
            p.contains(".model.ring.vals.") && v.as_f64().is_some()
        }),
        ("with_windows_moments", "a moments lag's buffer", |p, v| {
            p.contains(".model.bufs.") && p.contains(".vals.") && v.as_f64().is_some()
        }),
        ("refresh_time_int", "Instant::Int", |p, _| {
            p.contains(".last_time.") && p.ends_with(".Int")
        }),
        // Task 321: a pair grid sliced among one tick's points holds the
        // rest, the file then version 2 (3, with or without them, since
        // PLAN 542's envelope; `v71/` keeps the 2), and the tick's row in
        // each form a held value takes.
        ("refresh_time_held", "version 3 holding points", |p, v| {
            p == ".version" && v.as_u64() == Some(3)
        }),
        // PLAN 542: the clock's kind, named `clock_kind` beside the
        // envelope's own `kind`, in each of its two forms.
        ("refresh_time", "a number clock's clock_kind", |p, v| {
            p == ".clock_kind" && v.as_str() == Some("Numeric")
        }),
        (
            "refresh_time_held",
            "a temporal clock's clock_kind",
            |p, v| p == ".clock_kind" && v.as_str() == Some("Temporal"),
        ),
        ("refresh_time_held", "a held point", |p, v| {
            p.starts_with(".held.points.0.2.") && v.as_f64().is_some()
        }),
        ("refresh_time_held", "Cell::Null", |p, v| {
            p.starts_with(".held.row.") && v.as_str() == Some("Null")
        }),
        ("refresh_time_held", "Cell::Bool", |p, _| {
            p.starts_with(".held.row.") && p.ends_with(".Bool")
        }),
        ("refresh_time_held", "Cell::Int", |p, _| {
            p.starts_with(".held.row.") && p.ends_with(".Int")
        }),
        ("refresh_time_held", "Cell::UInt", |p, _| {
            p.starts_with(".held.row.") && p.ends_with(".UInt")
        }),
        ("refresh_time_held", "Cell::Float", |p, _| {
            p.starts_with(".held.row.") && p.ends_with(".Float")
        }),
        ("refresh_time_held", "Cell::Str", |p, _| {
            p.starts_with(".held.row.") && p.ends_with(".Str")
        }),
        ("refresh_time_held", "Cell::Date", |p, _| {
            p.starts_with(".held.row.") && p.ends_with(".Date")
        }),
        ("refresh_time_held", "a zoned Cell::Datetime", |p, v| {
            p.starts_with(".held.row.") && p.ends_with(".Datetime.2") && v.as_str().is_some()
        }),
        ("refresh_time_held", "Cell::Duration", |p, v| {
            p.starts_with(".held.row.") && p.ends_with(".Duration.1") && v.as_str() == Some("Ms")
        }),
        ("refresh_time_held", "Cell::Time", |p, _| {
            p.starts_with(".held.row.") && p.ends_with(".Time")
        }),
        ("refresh_time_held", "Cell::Arrow, as a binary", |p, v| {
            p.starts_with(".held.row.")
                && p.ends_with(".Arrow")
                && v.as_slice().is_some_and(|b| b.starts_with(b"ARROW1"))
        }),
        ("bank", "a target's floor_since", |p, v| {
            p.contains(".notified.") && p.contains(".floor_since.") && v.as_f64().is_some()
        }),
        ("bank", "the noise gate's gate_since", |p, v| {
            p.contains(".notified.") && p.ends_with(".gate_since") && v.as_f64().is_some()
        }),
        // Task 217 (schema 52): the rows of positive weight a windowed
        // model's cross-moments count per target, in a snapshot of its
        // window, and the residual spread's ring's counts per slot, live and
        // in a snapshot.
        (
            "bank",
            "a window snapshot's per-target row count",
            |p, v| {
                p.contains(".snaps.ring.")
                    && p.contains(".cross.nj.")
                    && v.as_u64().is_some_and(|n| n > 0)
            },
        ),
        ("bank", "the residual ring's row count", |p, v| {
            p.contains(".resid_win.") && p.contains(".rows.") && v.as_u64().is_some_and(|n| n > 0)
        }),
        ("bank", "a residual snapshot's row count", |p, v| {
            p.contains(".resid_win.")
                && p.contains(".ring.")
                && p.contains(".n.")
                && v.as_u64().is_some_and(|n| n > 0)
        }),
        // Task 221 (schema 55): the diagnostics with a memory of their own,
        // per instance and slot.
        ("bank_diagnosed", "a slot's CUSUM sums", |p, v| {
            p.contains(".checks.") && p.contains(".breaks.") && v.as_f64().is_some_and(|f| f != 0.0)
        }),
        ("bank_diagnosed", "a target's twin fits", |p, v| {
            p.contains(".checks.") && p.contains(".twin.") && v.as_f64().is_some_and(|f| f != 0.0)
        }),
        (
            "bank_diagnosed",
            "a slot's sandwich and its lags",
            |p, v| {
                p.contains(".checks.")
                    && p.contains(".sandwich.")
                    && p.contains(".gammas.")
                    && v.as_f64().is_some_and(|f| f != 0.0)
            },
        ),
        ("bank_diagnosed", "a sandwich's origin", |p, v| {
            p.contains(".sandwich.") && p.contains(".origin.") && v.as_f64().is_some()
        }),
        // Schema 56: the specification tests' lag sums and ring.
        ("bank_diagnosed", "a slot's Ljung-Box pairs", |p, v| {
            p.contains(".specification.")
                && p.contains(".pairs.")
                && v.as_f64().is_some_and(|f| f != 0.0)
        }),
        ("bank_diagnosed", "a slot's RESET origin", |p, v| {
            p.contains(".specification.") && p.ends_with(".origin") && v.as_f64().is_some()
        }),
        // Schema 57 (review round 6, B-5): the tails' central sums and
        // mean, where raw power sums were.
        ("bank_diagnosed", "a slot's tails' central sums", |p, v| {
            p.contains(".tails.") && p.contains(".m.") && v.as_f64().is_some_and(|f| f != 0.0)
        }),
        ("bank_diagnosed", "a slot's tails' mean", |p, v| {
            p.contains(".tails.") && p.ends_with(".mean") && v.as_f64().is_some_and(|f| f != 0.0)
        }),
        // Schema 57 (review round 6, A-6): a held row's score record keeps
        // the row's error inflation after its prediction -- the spec has
        // one slot and no conformal interval, so it is the record's second
        // value.
        ("bank_diagnosed", "a held row's scored inflation", |p, v| {
            p.contains(".score_pred.")
                && p.ends_with(".1")
                && v.as_f64().is_some_and(f64::is_finite)
        }),
        // Schema 60 (task 232 (3)): under a horizon, the CUSUMs' lag products
        // and ring, and the calibration's, Breusch and Pagan's and RESET's
        // sandwiches.
        ("bank_diagnosed", "a slot's CUSUM lag products", |p, v| {
            p.contains(".breaks.") && p.contains(".p1.") && v.as_f64().is_some_and(|f| f != 0.0)
        }),
        // Task 232 (5): the twin fits' weighted sums of `z z'` -- fast,
        // slow and crossed -- and their origin.
        ("bank_diagnosed", "a target's twin cross sums", |p, v| {
            p.contains(".twin.") && p.contains(".hfs.") && v.as_f64().is_some_and(|f| f != 0.0)
        }),
        ("bank_diagnosed", "a target's twin origin", |p, v| {
            p.contains(".twin.") && p.contains(".origin.") && v.as_f64().is_some()
        }),
        // Task 232 (4): the CUSUM of squares' fourth moment at the slow
        // memory.
        // Task 232 (6): a quantile fit's coverage, under a horizon, and its
        // CUSUM over indicators.
        (
            "bank_diagnosed",
            "a quantile fit's coverage lag products",
            |p, v| {
                p.contains(".coverage.")
                    && p.contains(".p.")
                    && v.as_f64().is_some_and(|f| f != 0.0)
            },
        ),
        (
            "bank_diagnosed",
            "a quantile fit's indicator CUSUM",
            |p, v| {
                p.contains(".breaks.") && p.ends_with(".c2") && v.as_f64().is_some_and(|f| f != 0.0)
            },
        ),
        // Schema 61 (task 244): a spec's keys and words as they stand at
        // 1.0.
        ("bank_diagnosed", "kmeans' n_clusters", |p, v| {
            p.ends_with(".model.n_clusters") && v.as_u64() == Some(2)
        }),
        ("bank_diagnosed", "init = \"k-means++\"", |p, v| {
            p.ends_with(".model.init") && v.as_str() == Some("k-means++")
        }),
        ("bank_diagnosed", "hmm's n_components", |p, v| {
            p.ends_with(".model.n_components") && v.as_u64() == Some(2)
        }),
        ("bank_diagnosed", "covariance_type = \"tied\"", |p, v| {
            p.ends_with(".model.covariance_type") && v.as_str() == Some("tied")
        }),
        ("bank_diagnosed", "covariance_type = \"full\"", |p, v| {
            p.ends_with(".model.covariance_type") && v.as_str() == Some("full")
        }),
        ("bank_diagnosed", "bocpd's mean_prior", |p, v| {
            p.contains(".model.mean_prior.") && v.as_f64().is_some()
        }),
        ("bank_diagnosed", "bocpd's mean_precision_prior", |p, v| {
            p.ends_with(".model.mean_precision_prior") && v.as_f64().is_some()
        }),
        (
            "bank_diagnosed",
            "bocpd's degrees_of_freedom_prior",
            |p, v| p.ends_with(".model.degrees_of_freedom_prior") && v.as_f64().is_some(),
        ),
        ("bank_diagnosed", "bocpd's covariance_prior", |p, v| {
            p.contains(".model.covariance_prior.") && v.as_f64().is_some()
        }),
        ("bank_diagnosed", "corrchange's window_rows", |p, v| {
            p.ends_with(".model.window_rows") && v.as_u64() == Some(10)
        }),
        ("bank_diagnosed", "change_action = \"reset\"", |p, v| {
            p.ends_with(".model.change_action") && v.as_str() == Some("reset")
        }),
        (
            "bank_diagnosed",
            "ew_cov's max_rows_between_pcas",
            |p, v| p.ends_with(".model.max_rows_between_pcas") && v.as_u64() == Some(5),
        ),
        ("bank_diagnosed", "a window's snapshot_every", |p, v| {
            p.ends_with(".model.snapshot_every") && !v.is_nil()
        }),
        // Schema 61 (review round 7, A-2): a quantile fit's residuals' lag
        // products and ring under a horizon, beside its indicators'.
        (
            "bank_diagnosed",
            "a quantile fit's residual lag products",
            |p, v| {
                p.contains(".breaks.") && p.contains(".pz.") && v.as_f64().is_some_and(|f| f != 0.0)
            },
        ),
        (
            "bank_diagnosed",
            "a quantile fit's residual ring",
            |p, v| {
                p.contains(".breaks.")
                    && p.contains(".ringz.")
                    && v.as_f64().is_some_and(|f| f != 0.0)
            },
        ),
        ("bank_diagnosed", "a slot's fourth moment", |p, v| {
            p.contains(".breaks.") && p.ends_with(".z4") && v.as_f64().is_some_and(|f| f != 0.0)
        }),
        ("bank_diagnosed", "a slot's CUSUM ring", |p, v| {
            p.contains(".breaks.") && p.contains(".ring2.") && v.as_f64().is_some_and(|f| f != 0.0)
        }),
        ("bank_diagnosed", "a calibration's sandwich", |p, v| {
            p.contains(".calibration.")
                && p.contains(".hac.")
                && p.contains(".gammas.")
                && v.as_f64().is_some_and(|f| f != 0.0)
        }),
        ("bank_diagnosed", "Breusch and Pagan's sandwich", |p, v| {
            p.contains(".het_hac.") && p.contains(".meat.") && v.as_f64().is_some_and(|f| f != 0.0)
        }),
        ("bank_diagnosed", "RESET's sandwich", |p, v| {
            p.contains(".reset_hac.")
                && p.contains(".ring.")
                && v.as_f64().is_some_and(|f| f != 0.0)
        }),
        ("bank_diagnosed", "a slot's influence scale", |p, v| {
            p.contains(".influence.") && p.ends_with(".q") && v.as_f64().is_some_and(|f| f != 0.0)
        }),
        (
            "bank_diagnosed",
            "the features' fast and slow moments",
            |p, v| {
                p.contains(".checks.")
                    && p.contains(".health.slow.")
                    && v.as_f64().is_some_and(|f| f != 0.0)
            },
        ),
        // Task 270 (i), R1-2: a target's twin fits' count of the rows
        // since the history was last wiped numerically.
        (
            "bank_diagnosed",
            "a target's twin fits' rows since a wipe",
            |p, v| {
                p.contains(".twin.")
                    && p.ends_with(".fresh_rows")
                    && v.as_u64().is_some_and(|n| n > 0 && n < u64::MAX)
            },
        ),
        // A bars run's state, in each form it takes: a running quantity
        // of several parts, a count, an open bar's null first value and
        // non-finite float sum, a float sum of several parts, an integer
        // sum, each kind of clock, a threshold column, a weight and a
        // session, the one key of an ungrouped run, and a monotone close's
        // high-water key.
        ("bars", "a running quantity of several parts", |p, v| {
            p.ends_with(".crossing.running.Exact.2") && v.as_f64().is_some()
        }),
        ("bars", "a null first value", |p, v| {
            p.ends_with(".First") && v.as_str() == Some("Null")
        }),
        ("bars", "a float sum's NaN", |p, v| {
            p.ends_with(".SumFloat.nan") && v.as_bool() == Some(true)
        }),
        ("bars", "a float sum of several parts", |p, v| {
            p.ends_with(".SumFloat.exact.1") && v.as_f64().is_some()
        }),
        ("bars", "an integer sum", |p, v| {
            p.ends_with(".SumInt") && !v.is_nil()
        }),
        ("bars", "Instant::Number", |p, _| {
            p.ends_with(".time_open.Number")
        }),
        ("bars", "the one key without a group", |p, v| {
            p == ".groups.0.0" && v.as_str() == Some("")
        }),
        ("bars_by", "Instant::Int", |p, _| {
            p.ends_with(".last_clock.Int")
        }),
        ("bars_by", "a threshold column", |p, v| {
            p == ".config.threshold.column" && v.as_str() == Some("th")
        }),
        ("bars_by", "a session", |p, v| {
            p.ends_with(".session") && v.as_u64().is_some()
        }),
        ("bars_by", "an unsigned value", |p, _| p.ends_with(".U")),
        ("bars_by", "a Boolean value", |p, _| p.ends_with(".B")),
        ("bars_count", "Running::Count", |p, v| {
            p.ends_with(".crossing.running.Count") && v.as_u64().is_some_and(|n| n > 0)
        }),
        ("bars_count", "Instant::Nanos", |p, _| {
            p.ends_with(".time_open.Nanos")
        }),
        ("bars_count", "a monotone close's high-water key", |p, v| {
            p == ".high_water" && v.as_str().is_some()
        }),
        // R3-3: the values an autocorrelation keeps for its lag.
        (
            "bank_autocorr",
            "an autocorrelation's lagged values",
            |p, v| {
                p.contains(".autocorr.")
                    && p.contains(".buf.")
                    && v.as_f64().is_some_and(|f| f != 0.0)
            },
        ),
        ("bank_diagnosed", "a slot's calibration moments", |p, v| {
            p.contains(".checks.")
                && p.contains(".calibration.")
                && v.as_f64().is_some_and(|f| f != 0.0)
        }),
    ];
    for (kind, form, holds) in FORMS {
        let f = frozen::ALL
            .iter()
            .find(|f| f.name == *kind)
            .unwrap_or_else(|| panic!("{kind}: no fixture; add the kind and regenerate"));
        let v = rmpv::decode::read_value(&mut unhex(f.state).as_slice()).unwrap();
        let mut all = Vec::new();
        leaves(&v, "", &mut all);
        assert!(
            all.iter().any(|(p, v)| holds(p, v)),
            "{kind}: no leaf of the state holds {form}, so the fixture freezes nothing of it"
        );
    }
}

/// The `bank_label_clock` fixture freezes a buffer in label-time order that
/// is not its arrival order: some held row sits after one that arrived
/// later, so a loader that read the rows back in arrival order would release
/// them in another order (schema 67, task 341).
#[test]
fn a_label_buffer_is_frozen_out_of_arrival_order() {
    if regenerating() {
        return;
    }
    let f = frozen::ALL
        .iter()
        .find(|f| f.name == "bank_label_clock")
        .expect("the label_clock fixture");
    let v = rmpv::decode::read_value(&mut unhex(f.state).as_slice()).unwrap();
    let mut all = Vec::new();
    leaves(&v, "", &mut all);
    // Each held row's arrival clock, in buffer order, by the stream it is in.
    let mut by_stream: std::collections::BTreeMap<String, Vec<i64>> = Default::default();
    for (p, v) in &all {
        if let Some(at) = p.find(".pending.")
            && p.ends_with(".clock.I64")
        {
            by_stream
                .entry(p[..at].to_string())
                .or_default()
                .push(v.as_i64().unwrap());
        }
    }
    assert!(
        by_stream
            .values()
            .any(|c| c.windows(2).any(|w| w[1] < w[0])),
        "no held row sits after one that arrived later: {by_stream:?}"
    );
}

/// A row of weight 0 is held for nothing (hard rule 9; PLAN 539, which
/// supersedes review P1's holding of it), so a frozen buffer holds none:
/// `bank_conformal_auto`'s, whose buffer has no bound, held `label_frame`'s
/// row 37 (label time 94, in flight at row 40), and holds the rows of
/// positive weight in flight alone.
#[test]
fn a_label_buffer_is_frozen_holding_no_zero_weight_row() {
    if regenerating() {
        return;
    }
    let f = frozen::ALL
        .iter()
        .find(|f| f.name == "bank_conformal_auto")
        .expect("the conformal_auto fixture");
    let v = rmpv::decode::read_value(&mut unhex(f.state).as_slice()).unwrap();
    let mut all = Vec::new();
    leaves(&v, "", &mut all);
    let weights: Vec<f64> = all
        .iter()
        .filter(|(p, _)| p.contains(".pending.") && p.ends_with(".w"))
        .map(|(_, v)| v.as_f64().unwrap())
        .collect();
    assert!(
        weights.len() > 2 && !weights.contains(&0.0),
        "a held row of weight 0: {weights:?}"
    );
}

// --- regeneration -----------------------------------------------------------

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/state_fixtures")
}

/// `text` as a Rust string literal of hex, wrapped.
fn hex_literal(out: &mut String, indent: &str, text: &str) {
    writeln!(out, "\"\\").unwrap();
    let chunks: Vec<&[u8]> = text.as_bytes().chunks(96).collect();
    for (i, chunk) in chunks.iter().enumerate() {
        let tail = if i + 1 == chunks.len() { "\"" } else { "\\" };
        writeln!(out, "{indent}{}{tail}", std::str::from_utf8(chunk).unwrap()).unwrap();
    }
}

fn file_of(kind: &Kind) -> String {
    let (bytes, input) = (kind.first)();
    let (again, frames) =
        (kind.run)(&bytes, &input).unwrap_or_else(|e| panic!("{}: {e}", kind.name));
    assert!(again == bytes, "{}: re-saves other bytes", kind.name);
    // Two runs of the first part write the same bytes, which is what makes
    // a frozen file comparable at all.
    assert!(
        (kind.first)().0 == bytes,
        "{}: two runs saved other bytes",
        kind.name
    );
    let mut out = format!(
        "// @generated by `{REGENERATE}`: do not edit by hand.\n\
         pub const FIXTURE: crate::Fixture = crate::Fixture {{\n    name: {:?},\n    schema: {SCHEMA_VERSION},\n    writer: {:?},\n    state: ",
        kind.name,
        writer()
    );
    hex_literal(&mut out, "        ", &hex(&bytes));
    out.push_str("    ,\n    input: ");
    hex_literal(&mut out, "        ", &hex(&ipc(&input)));
    out.push_str("    ,\n    output: &[\n");
    for (name, df) in &frames {
        write!(out, "        (\n            {name:?},\n            ").unwrap();
        hex_literal(&mut out, "            ", &hex(&ipc(df)));
        out.push_str("        ),\n");
    }
    out.push_str("    ],\n};\n");
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
    // Every current fixture is rewritten and any other removed (a kept set
    // under `v<N>/` is not touched); no index: `build.rs` discovers the
    // files on the next build.
    for entry in std::fs::read_dir(&dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_file() && path.extension().is_some_and(|e| e == "rs") {
            std::fs::remove_file(path).unwrap();
        }
    }
    for kind in &kinds() {
        let text = file_of(kind);
        std::fs::write(dir.join(format!("{}.rs", kind.name)), &text).unwrap();
        println!("wrote {}.rs ({} bytes)", kind.name, text.len());
    }
    // A run that rewrote the files tested nothing: the other tests return
    // early under the variable. It fails, so it never stands for a test run
    // (review S20: a broken build regenerated its fixtures and passed).
    panic!(
        "PRINT_STATE_FIXTURES=1 rewrote the current fixtures and tested nothing; run again \
         without it to test them"
    );
}

/// A value that holds a whole state file of the envelope (a checkpoint's
/// stage, a bank's formula core: msgpack bytes, stored as `bin` or as an
/// array of byte values) opened into the value it encodes, its `checksum`
/// stripped, as the top level of a file is for a cross-OS comparison. `None`
/// for anything else; it is tried only on bytes that begin as the envelope
/// does, so a list of numbers that is not a file is never decoded.
fn embedded_state(v: &rmpv::Value) -> Option<rmpv::Value> {
    const HEAD: &[u8] = b"\xa5magic\xabrheon-state";
    let bytes: Vec<u8> = match v {
        rmpv::Value::Binary(b) => b.clone(),
        rmpv::Value::Array(a) if a.len() > HEAD.len() + 3 => a
            .iter()
            .map(|x| x.as_u64().and_then(|n| u8::try_from(n).ok()))
            .collect::<Option<_>>()?,
        _ => return None,
    };
    // A map header of one byte (up to 15 entries) or of three (`map 16`).
    let at = match bytes.first()? {
        0x80..=0x8f => 1,
        0xde => 3,
        _ => return None,
    };
    bytes[at..].starts_with(HEAD).then_some(())?;
    let mut opened = rmpv::decode::read_value(&mut &bytes[..]).ok()?;
    if let rmpv::Value::Map(m) = &mut opened {
        m.retain(|(k, _)| k.as_str() != Some("checksum"));
    }
    Some(opened)
}

/// Where `got` is not `want` field for field -- the same map keys in the
/// same order, the same array lengths, every value but a float equal, and
/// each float within `bound` of its twin relative to `1 + |want|` (a NaN
/// on both sides agrees) -- the first place it is not. A state file held
/// inside the value (a checkpoint's stage, a bank's formula core) is opened
/// and compared the same way, its own checksum left out: its bytes carry
/// the libm fields' last bits and the hash over them, which differ by OS
/// as the outer file's do.
fn first_difference(got: &rmpv::Value, want: &rmpv::Value, bound: f64, at: &str) -> Option<String> {
    use rmpv::Value as V;
    if let (Some(g), Some(w)) = (embedded_state(got), embedded_state(want)) {
        return first_difference(&g, &w, bound, &format!("{at}(state)"));
    }
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
            let agree = if g.is_nan() || w.is_nan() {
                g.is_nan() && w.is_nan()
            } else {
                // An infinity agrees only with itself: against an infinite
                // `w` the relative bound would admit any number.
                g.to_bits() == w.to_bits()
                    || (g.is_finite() && w.is_finite() && (g - w).abs() <= bound * (1.0 + w.abs()))
            };
            (!agree).then(|| format!("{at}: {g:e} against {w:e}"))
        }
        _ => (got != want).then(|| format!("{at}: {got} against {want}")),
    }
}

/// Where the current build's run of `kind` is not the committed fixture
/// `f`, which another platform wrote: the state's layout exactly and its
/// numbers within the golden files' tolerance, the continuation's input to
/// the bit and the frames it gave within the same tolerance ([`same`]). A
/// decay or a log-likelihood runs through libm, whose last bit is the
/// platform's.
fn differs_from_another_writer(kind: &Kind, f: &Fixture) -> Option<String> {
    let (bytes, input) = (kind.first)();
    // The checksum hashes the other bytes, and a libm field's last bits may
    // differ by OS, so the hash may too; the fields it covers are compared
    // within the tolerance below, which is what the checksum stands for.
    let decode = |b: &[u8]| {
        let mut v = rmpv::decode::read_value(&mut &b[..]).unwrap();
        if let rmpv::Value::Map(m) = &mut v {
            m.retain(|(k, _)| k.as_str() != Some("checksum"));
        }
        v
    };
    if let Some(d) = first_difference(&decode(&bytes), &decode(&unhex(f.state)), TOL, "state") {
        return Some(d);
    }
    let compared = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        same("input", &input, &from_ipc(f.input), 0.0);
        let (_, frames) = (kind.run)(&bytes, &input).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(frames.len(), f.output.len(), "the frames");
        for ((name, got), (_, want)) in frames.iter().zip(f.output) {
            same(name, got, &from_ipc(want), TOL);
        }
    }));
    compared
        .is_err()
        .then(|| "the continuation is another (its panic above says where)".to_owned())
}

/// **The committed fixtures are what this build writes.** Every kind of
/// the current schema is run as `PRINT_STATE_FIXTURES=1` runs it, in
/// memory, and set beside its committed file: where the platform that
/// wrote the file is this one, the text [`file_of`] would write must be
/// the file's, byte for byte; elsewhere the state's layout must be the
/// same and its numbers and the continuation's within the golden
/// tolerance ([`differs_from_another_writer`]). The other checks load a
/// fixture, go on from it and save it again, which a file written before
/// a change in what a kind saves still passes: it loads and re-saves its
/// own bytes (`bank_refit` and `bank_diagnosed` stood so through a change
/// that reserved a refit's held residuals). This one fails on such a file,
/// and names it, so a change that moves what a kind writes regenerates the
/// set in the same change.
#[test]
fn the_committed_fixtures_are_what_this_build_writes() {
    if regenerating() {
        return;
    }
    let mut stale = Vec::new();
    for kind in &kinds() {
        let file = format!("{}.rs", kind.name);
        let f = frozen::ALL
            .iter()
            .find(|f| f.name == kind.name)
            .unwrap_or_else(|| {
                panic!("{file}: no committed fixture; regenerate with `{REGENERATE}`")
            });
        let why = if f.writer == writer() {
            let committed = std::fs::read_to_string(dir().join(&file))
                .unwrap_or_else(|e| panic!("{file}: {e}"));
            (file_of(kind) != committed).then(|| "the text this build writes is another".to_owned())
        } else {
            differs_from_another_writer(kind, f)
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

/// A checkpoint's stage is a whole state file stored as bytes, so another
/// OS's checkpoint differs from this one's in the stage's libm fields' last
/// bit and in the stage's own checksum, bytes that a plain comparison reads as
/// a different file (the CI failure of `checkpoint.rs` on Linux and Windows:
/// nine floats of the window run's queues (`front.vals`, `front.sums`) and
/// `held_values`, each within 3 ulps of the macOS-written one, read from the
/// Linux bytes in the failed run; the comparison's bound, 1e-12, is the
/// golden tolerance the outer files already use).
/// The stage is opened and held to the golden tolerance like the outer
/// file: a last-bit difference in a float passes, a larger one and any other
/// field's difference does not.
#[test]
fn a_checkpoint_stage_is_compared_as_a_state_not_as_bytes() {
    if regenerating() {
        return;
    }
    let f = frozen::ALL
        .iter()
        .find(|f| f.name == "checkpoint")
        .expect("the checkpoint fixture");
    let file = rmpv::decode::read_value(&mut &unhex(f.state)[..]).unwrap();
    fn entry<'a>(v: &'a mut rmpv::Value, key: &str) -> &'a mut rmpv::Value {
        match v {
            rmpv::Value::Map(m) => {
                &mut m
                    .iter_mut()
                    .find(|(k, _)| k.as_str() == Some(key))
                    .unwrap()
                    .1
            }
            _ => panic!("a map holds {key}"),
        }
    }
    fn items(v: &mut rmpv::Value) -> &mut Vec<rmpv::Value> {
        match v {
            rmpv::Value::Array(a) => a,
            _ => panic!("an array"),
        }
    }
    // The file with its second stage's float `held_values[2]` moved by
    // `nudge` and, where asked, its `held_values` one entry short.
    let with_stage_changed = |nudge: f64, short: bool| {
        let mut file = file.clone();
        let stage = &mut items(entry(&mut file, "stages"))[1];
        let state = entry(stage, "state");
        let mut inner = embedded_state(state).expect("the stage is a state file");
        let h = items(entry(entry(&mut inner, "core"), "held_values"));
        let x = h[2].as_f64().expect("a float");
        h[2] = rmpv::Value::F64(x + nudge * (1.0 + x.abs()));
        if short {
            h.pop();
        }
        let mut bytes = Vec::new();
        rmpv::encode::write_value(&mut bytes, &inner).unwrap();
        // Stored as the checkpoint stores it, bytes as an array of values.
        *state = rmpv::Value::Array(bytes.iter().map(|&b| rmpv::Value::from(b)).collect());
        file
    };
    let at = "state";
    // Unchanged (the stage's checksum is gone from the copy, so its bytes
    // differ from the original's): the same.
    assert_eq!(
        first_difference(&with_stage_changed(0.0, false), &file, TOL, at),
        None
    );
    // A last-bit difference: within the tolerance.
    assert_eq!(
        first_difference(&with_stage_changed(1e-15, false), &file, TOL, at),
        None
    );
    // A real difference in a float, and in the shape of a field.
    let d = first_difference(&with_stage_changed(1e-6, false), &file, TOL, at).expect("a float");
    assert!(d.contains("\"held_values\"[2]"), "{d}");
    let d = first_difference(&with_stage_changed(0.0, true), &file, TOL, at).expect("a length");
    assert!(d.contains("held_values"), "{d}");
}

/// The two crates discover their fixtures with one `build.rs`, kept as a
/// copy in each so each package builds on its own: the copies are the same
/// file, byte for byte.
#[test]
fn the_fixture_discovery_is_one_build_script() {
    assert!(
        include_str!("../build.rs") == include_str!("../../rheon-core/build.rs"),
        "crates/rheon-polars/build.rs and crates/rheon-core/build.rs differ: copy one over the \
         other"
    );
}

/// A conformal tracker saved before the multiplier (schema 66 or earlier,
/// CLAUDE.md rule 5's exception of 2026-11-09, task 368) holds the radius
/// `q` where one now holds `m`. No kept fixture holds a conformal tracker,
/// so the form is written here from a current bank, its trackers' `m`
/// renamed to `q` (a radius of 1.75) and the file's schema set to 66: it
/// loads, saves its `q` again, shows that radius on the next row to the bit,
/// and from the first row with an error scale holds `m = q / σ`, stepped,
/// its radius `m·σ` for the `σ` each later row reads, and writes no `q`.
#[test]
fn a_conformal_radius_of_schema_66_is_shown_then_converted() {
    if regenerating() {
        return;
    }
    let spec: Spec = serde_json::from_str(
        r#"{"name": "m", "model": {"type": "ewridge"}, "targets": ["y"], "features": ["x0"],
            "half_life": 50.0, "conformal": 0.9, "emit_sigma": true}"#,
    )
    .unwrap();
    let n = 80usize;
    let mut r = lcg(12);
    let x0: Vec<f64> = (0..n).map(|_| r()).collect();
    let y: Vec<f64> = (0..n).map(|i| 2.0 * x0[i] + 0.5 * r()).collect();
    let df = df!("x0" => x0, "y" => y).unwrap();
    let mut bank = Bank::new(vec![spec.clone()]).unwrap();
    bank.fit_predict(&df.slice(0, 60)).unwrap();
    // The file as a build before the multiplier wrote it.
    let mut v = rmpv::decode::read_value(&mut &bank.save_bytes().unwrap()[..]).unwrap();
    fn to_66(v: &mut rmpv::Value, found: &mut usize) {
        match v {
            rmpv::Value::Map(m) => {
                let tracker = m.iter().any(|(k, _)| k.as_str() == Some("cov_w"))
                    && m.iter().any(|(k, _)| k.as_str() == Some("m"));
                for (k, val) in m.iter_mut() {
                    if tracker && k.as_str() == Some("m") {
                        *k = rmpv::Value::from("q");
                        *val = rmpv::Value::from(1.75);
                        *found += 1;
                    } else if k.as_str() == Some("schema_version") {
                        *val = rmpv::Value::from(66);
                    } else {
                        to_66(val, found);
                    }
                }
            }
            rmpv::Value::Array(a) => a.iter_mut().for_each(|x| to_66(x, found)),
            _ => {}
        }
    }
    let mut found = 0;
    to_66(&mut v, &mut found);
    assert_eq!(found, 1, "one tracker");
    // A file of 66 begins with the bank's magic before the envelope, names
    // its schema `schema_version`, and ends with no checksum (schema 71's).
    if let rmpv::Value::Map(m) = &mut v {
        m.retain(|(k, _)| !matches!(k.as_str(), Some("checksum" | "kind")));
        for (k, val) in m.iter_mut() {
            match k.as_str() {
                Some("magic") => *val = "polars-online-bank".into(),
                Some("version") => {
                    *k = "schema_version".into();
                    *val = 66.into();
                }
                _ => {}
            }
        }
    }
    let mut old = Vec::new();
    rmpv::encode::write_value(&mut old, &v).unwrap();
    let mut bank = Bank::load_bytes(&old, Some(std::slice::from_ref(&spec))).unwrap();
    let again = bank.save_bytes().unwrap();
    let field_of = |bytes: &[u8], key: &str| -> Vec<f64> {
        fn walk(v: &rmpv::Value, key: &str, out: &mut Vec<f64>) {
            match v {
                rmpv::Value::Map(m) => {
                    let tracker = m.iter().any(|(k, _)| k.as_str() == Some("cov_w"));
                    for (k, val) in m {
                        if tracker && k.as_str() == Some(key) {
                            out.push(val.as_f64().unwrap_or(f64::NAN));
                        }
                        walk(val, key, out);
                    }
                }
                rmpv::Value::Array(a) => a.iter().for_each(|x| walk(x, key, out)),
                _ => {}
            }
        }
        let mut out = Vec::new();
        walk(
            &rmpv::decode::read_value(&mut &bytes[..]).unwrap(),
            key,
            &mut out,
        );
        out
    };
    assert_eq!(field_of(&again, "q"), vec![1.75], "the radius kept");
    let read = |out: &DataFrame, name: &str| -> f64 {
        out.column("m")
            .unwrap()
            .struct_()
            .unwrap()
            .field_by_name(name)
            .unwrap()
            .f64()
            .unwrap()
            .get(0)
            .unwrap()
    };
    let row = |bank: &mut Bank, i: usize| {
        DataFrame::new(1, bank.fit_predict(&df.slice(i as i64, 1)).unwrap()).unwrap()
    };
    // The next row is shown the radius saved, to the bit.
    let first = row(&mut bank, 60);
    let (pred, sigma) = (read(&first, "pred_y"), read(&first, "sigma_y"));
    assert!(sigma > 0.0);
    assert_eq!(read(&first, "hi_y").to_bits(), (pred + 1.75).to_bits());
    assert_eq!(read(&first, "lo_y").to_bits(), (pred - 1.75).to_bits());
    // That row had an error scale: the tracker is a multiplier, stepped
    // from 1.75 / σ by one step of the rate.
    let saved = bank.save_bytes().unwrap();
    assert!(field_of(&saved, "q").is_empty(), "no radius written");
    let m = field_of(&saved, "m")[0];
    // Unit weights, so `w/w̄` is 1.
    let miss = f64::from(u8::from(
        (df.column("y").unwrap().f64().unwrap().get(60).unwrap() - pred).abs() > 1.75,
    ));
    let want = (1.75 / sigma + 0.05 * 1.0 * (miss - (1.0 - 0.9))).max(0.0);
    assert_eq!(m.to_bits(), want.to_bits(), "{m} vs {want}");
    // Every later row reads `m·σ`.
    let next = row(&mut bank, 61);
    let (pred, sigma) = (read(&next, "pred_y"), read(&next, "sigma_y"));
    assert_eq!(read(&next, "hi_y").to_bits(), (pred + m * sigma).to_bits());
}
