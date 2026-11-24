//! The contract every model in this crate keeps, asserted once for all of them.
//!
//! Why this file exists. Each model has thorough tests of *its own* behaviour,
//! but the small shared surface -- the shape accessors, `n_eff`, the prediction
//! slot count, state round-tripping, and the "before this row" reading of
//! everything -- was asserted nowhere in Rust. A `cargo mutants` pass made that
//! visible: `n_eff -> 0.0`, `n_targets -> 1` and `n_features -> 0` survived in
//! nearly every file, because the only tests exercising them run in Python
//! through the compiled extension, which `cargo test` cannot see.
//!
//! This is the core-level counterpart of `tests/test_semantics_all_models.py`.
//! Anything genuinely model-specific belongs in that model's own unit tests.

use rheon_core::*;

// The contract is one test binary in sections, each a file of `model_contract/`
// (the file was 263 KB at one piece). The sections are included, not modules,
// so every test keeps its name, `cargo test model_contract <name>` finds it as
// before, and each section reads the shared helpers of `common.rs` unqualified.
include!("model_contract/common.rs");
include!("model_contract/models.rs");
include!("model_contract/states.rs");
include!("model_contract/more_models.rs");
include!("model_contract/refusals.rs");
include!("model_contract/bounded.rs");
include!("model_contract/predict_is_the_step.rs");
include!("model_contract/unusable_values.rs");
include!("model_contract/lower_floor.rs");
include!("model_contract/generated.rs");
include!("model_contract/recovery_since_2_0.rs");
