# Hoeffding trees — design as built

*Status: built on 2026-11-27 for schema 72 (PLAN 503), following `STUDY-503-TREES.md`.*

## 1. Scope and model kinds

Two model kinds are added, matching Rheon's separation of regression and classification specs:

1. **`hoeffding_tree_classifier`** (`rheon.spec.hoeffding_tree_classifier`, `ModelKind::HoeffdingTreeClassifier`) — Very Fast Decision Tree (VFDT, Domingos & Hulten 2000) with per-feature per-class Gaussian observers (`n_splits` equally spaced thresholds between `[min, max]`), split criteria `"info_gain"`, `"gini"`, and `"hellinger"` (2-class only), and leaf prediction modes `"majority"`, `"nb"`, and `"nba"` (default). Matches `ew_class`'s output contract (`class`, `p_<label>` per declared class, `weight_sum`, `settled_frac`, `withheld_reason`, no `coef`).
2. **`hoeffding_tree_regressor`** (`rheon.spec.hoeffding_tree_regressor`, `ModelKind::HoeffdingTreeRegressor`) — FIMT-style mean-leaf regression tree (Ikonomovska, Gama & Dzeroski 2011) with bounded Quantization Observers (QO, Mastelini & de Carvalho 2021; `radius = 0.25`, `max_slots = 32` by default), variance-reduction splits (`min_samples_split = 5.0`), and one independent tree per target. Matches Rheon's regression contract (`pred_<t>`, `resid_<t>`, multi-target, formula targets, all residual diagnostics; linear-only readings refused; no `coef`).

Tree inspection (`bank.trees()`) is deferred to a follow-up task.

## 2. Intentional departures from `river`

1. **Null-split merit (`merit_preprune = true`).** Set to `0.0` (Domingos & Hulten 2000), not `-inf` as in `river`'s Python path. A single pure-noise feature no longer forces splits at every grace period while `eps >= tau`.
2. **Zero-weight rows (CLAUDE.md hard rule 9).** A zero-weight row predicts out-of-sample and leaves counts, observers, min/max bounds, and recheck counters untouched.
3. **Permanent deactivation.** Leaves deactivated by `max_share_to_split` or winning null splits stay inactive permanently and are never reactivated by the memory cap manager.
4. **Declared class tie-breaking.** Ties in majority-class and naive Bayes argmax break deterministically by the declared `classes` list order, never by first-seen dict insertion order.
5. **Hellinger with arbitrary labels.** Works for any two string labels and when one class has zero mass on one side of a threshold (fixing `river` 0.26.1's integer-key `KeyError`).
6. **Bounded QO fixes.** At the midpoint between consecutive sorted slots `i - 1` and `i`, Rheon scores the exact symmetric partition `{slots <= i - 1} | {slots >= i}` via prefix/suffix Chan merges over the observer's own slots (no subtraction from inherited parent statistics, no off-by-one overlap, and copied left-child stats instead of aliased accumulators). Over 50 seeded nonlinear regression streams this cuts mean tail MAE by ~10% vs parent-subtraction and ~17% vs `river`'s unbounded TE-BST.

## 3. Per-group memory cap (`max_state_bytes`)

- Default: `262_144` bytes (256 KiB); ceiling: `67_108_864` bytes (64 MiB).
- Multi-target regressors divide the group cap equally across targets (`cap / n_targets` per tree).
- Build-time refusal when `cap` cannot hold at least one active root leaf with observers per target (`40 + 8C + (8C + 27) + 40kC` for classifier; `T * (40 + 35 + k * (2 + 40 * max_slots))` for regressor).
- `ReadinessWarning` emitted by `rheon.spec.hoeffding_tree_regressor` when `len(features) >= 20` and `max_state_bytes < 1_048_576` (suggesting at least 1 MiB).
- Graceful degradation at cap: splits pause when `struct_bytes(L + 1, B + 1) > tree_cap`; after every split and every 1,000 learned rows (`rows_learned` persisted in state for bit-exact save/resume), eligible leaves are ranked by promise (`total - max(counts)` for classifier; leaf SSE `S` for regressor) and only as many keep observers as `(tree_cap - struct_bytes) / obs_reserve` allows. Capped leaves keep updating their class counts / target mean and may reactivate with fresh observers if their promise ranks back in. Neither model hard-refuses rows via `HeldCap`.

## 4. Decay, windows, weights, and portability

- Exponential decay (`half_life`, `lam`, `min_settled_frac`) and hard windows (`window_size`, `window_rows`) are refused: standard Hoeffding bounds assume i.i.d. cumulative leaf weights, and no third-party oracle defines decayed Hoeffding trees cleanly.
- Weights are frequency weights (registered in `tests/test_weight_scale.py::EXCEPTIONS`), as leaf weight enters `grace_period` and `eps = sqrt(R^2 ln(1/delta) / (2 W))`.
- All transcendentals (`erf`, `log`, `log2`, `exp`) route through pure-Rust `libm` so splits, probabilities, predictions, and frozen state fixtures are bit-identical across macOS, Linux, and Windows.
