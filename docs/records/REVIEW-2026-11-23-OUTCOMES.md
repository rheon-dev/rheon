<!-- kind: record -->
# The 2026-11-23 review: what was done about each finding

`REVIEW-2026-11-23.md` (pushed at 555b45e5, reviewed at c4e7c7d2) is a
whole-project review in two passes. Its reviewer ran no `cargo`, so every
claim was read, not run. PLAN 578 settles each claim test first: a test
that fails on the build before the change, for the reason the review gives,
confirms it, and the fix follows in the same batch; a test that passes
where the review expected a failure is recorded as not reproduced; a claim
that contradicts a recorded decision goes to the user with both sides.

Each batch adds its table below.

## Batch c: the stream, embargo and labels, clock and spans, `rp.stream.merge`, the bank's caps and notices (PLAN 578c)

On `review-578c`, cut from `integration` at 1e0e0945. "Confirmed" means
the named test failed on 1e0e0945 for the review's reason.

| ID | verdict | test | commit |
|---|---|---|---|
| GLM-METRICS | confirmed, fixed: a binomial `glm` scored by the 0.5 threshold and read by `rp.eval` as a classifier; Poisson, Gamma and Tweedie (and a spec with no family) null `hit_rate` | `tests/test_glm.py::test_each_familys_hit_rate_is_its_own_test_in_the_bank_and_in_eval` | 31ad0f5d |
| STRM-1 | confirmed (claims 1 and 2), fixed by the user's decision (2026-11-23: "Score as predict"): under `embargo` or `label_clock` a row of weight 0 is scored as `predict` scores it and releases no held row, so it moves no live state; it has no place on the elapsed clock, which settles claim 3. The cost: such a row does not see the labels matured at its own clock. Claim 4 is the rule `predict` keeps (task 120) | `tests/test_label_delay.py::test_a_zero_weight_row_releases_no_label_a_later_row_before_it_would_not_see`, `::test_a_stream_ending_on_a_zero_weight_row_saves_what_it_would_without_it`, `::test_a_number_clock_and_a_temporal_one_agree_at_a_zero_weight_row_that_starts_a_session`; `crates/rheon-polars/tests/zero_weight.rs::a_stream_ending_on_zero_weight_rows_saves_what_it_would_without_them`, `::a_zero_weight_row_is_scored_as_predict_scores_it` (now under `embargo` and `label_clock` too) | 31ad0f5d, 5a763d77, fa550472, fed9d7f3 |
| REG-2 | confirmed by reading and by test, fixed: an ageing row's new run is held to the span's cap | `crates/rheon-core/src/robust/span_tests.rs::an_ageing_row_is_held_to_the_spans_cap`, `sgd/span_tests.rs::an_ageing_row_is_held_to_the_spans_cap` | 549908ef |
| REG-3 | **not reproduced as stated** (the coordinator, 2026-11-23: no decay). The slope that stays near 0 is the step's weight under the default constant rate, as scikit-learn's `sample_weight` multiplies the step: weights of 1e-6 on every row, with no fall and `ŵ` exactly 1e-6, stay as near 0, standardized or not | `tests/test_sgd.py::test_a_fall_in_the_weights_slows_the_fit_as_scikit_learn_s_sample_weight_does` | 1db677d9 |
| MERGE-1 | confirmed (a NaN and a step back past the slice), fixed by the break the review's Pass 2 gives | `tests/test_stream_merge.py::test_a_slice_reads_no_chunk_past_its_last_row` | 31ad0f5d |
| STRM-2 | confirmed, fixed: a fresh-session row of weight 0 reads a fresh stream's diagnostics and `learned_clock`; `predict`'s `learned_clock` too. `predict` already scored the fresh class with a fresh stream's diagnostics, so that half of the Pass-2 note was not reproduced | `tests/test_zero_weight_rows.py::test_a_zero_weight_row_that_starts_a_fresh_session_reads_a_fresh_streams_diagnostics` | 31ad0f5d |
| STRM-3 | (2) confirmed, fixed (`weight_sum_settled`'s mean weight). (1) rule kept (the coordinator, 2026-11-23: no defect is shown): `ModelBank.last_row`'s docstring says a row of weight 0 replaces it, the last row the model predicted. (3) not changed: the unstable-rate notice reports the state the row is read against, which a row of weight 0 reads as the next row will, and no output of a row of positive weight moves. (4), (5) the rule kept: `ZERO-WEIGHT-AUDIT.md` leaves the data summary and the clock range describing every row fed, and W9 made a row of weight 0 no session boundary. (5)'s refit reservation is BANK-6's question | `tests/test_zero_weight_rows.py::test_rows_of_weight_0_leave_the_settled_weight_where_it_was` | 31ad0f5d |
| CLOCK-1 | confirmed on two finite steps (the review's one step is refused already, PB7), fixed by a refusal by row, on a bound of the removed time | `crates/rheon-polars/tests/bank.rs::a_clock_whose_removed_time_passes_the_largest_double_is_refused_by_row` | 31ad0f5d |
| BANK-1 | (1) confirmed, fixed in Rust and Python. (2) the rule kept: a row of weight 0 is in no clock, so the bank holds no place for one after its last clock, and replaying it moves nothing. (3) deferred: specs that group or weigh differently are PLAN 566's documented rule | `tests/test_skip_learned.py::test_a_monotone_close_drops_the_groups_it_closed`, `crates/rheon-polars/tests/bank.rs::skip_learned_keeps_an_infinite_clock_and_drops_a_closed_group` | 31ad0f5d |
| STRM-4 | (1) confirmed, fixed: `EwClass::target_n_eff_into`, the classes' weights. (2) confirmed, fixed; `tests/test_released_state.py` reads the move on every release's file (`_learned_clock_kept`). The Pass-2 addition (a held row sets `learned_clock` when queued) not reproduced by reading: a held row's plan has `learn = false`, and its release sets it | `tests/test_zero_weight_rows.py::test_ew_class_withholds_until_its_labelled_rows_reach_min_weight`, `::test_the_learned_clock_is_the_last_row_with_a_target_learned` | 4a79f8ac, 31ad0f5d, d6fa6a2c |
| REG-14 | confirmed (both parts), fixed in `gramfit::merge`; the same `Σ w²` fabrication in Python's `gram.merge` (`_target_q`) is MATH-3's, batch g | `crates/rheon-core/src/gramfit/merge_tests.rs::a_merge_is_the_same_at_every_scale_of_weight`, `::a_weighted_part_without_a_kish_size_leaves_the_merged_one_unknown` | c692174b |
| MATH-7 | confirmed, fixed in `merge`, `merge_within` and `within_cross` | `merge_tests.rs::a_part_with_no_weight_for_a_target_adds_nothing_even_where_its_moments_are_nan` | c692174b |
| SPAN-1 | deferred: carrying a span's nanoseconds into `clock_span` touches every span-capable model's configuration; the loss is 1 ns at spans past 97 days | -- | -- |
| SPAN-2 | deferred (the `glm` and `expectile` span caps; with SPAN-3 and REG-11, one change to make every span cap atomic) | -- | -- |
| REG-11 | deferred with SPAN-2 | -- | -- |
| MERGE-2 | deferred | -- | -- |
| MERGE-3 | confirmed by a run (a state holding `a: [2.0], b: [2.0]`, `.head(1)` on `a2 = [3.0]`, `b2 = [3.0]`: the slice returns the held `(2.0, "a")`, `b`'s row stays held, and `a2.slice(1)` drops `3.0`); deferred: the fix is a choice of surface, a count of each input's rows read beside the output or a refusal of a slice over a loaded state that holds a tie, besides the docstring | -- | -- |
| BANK-5 | confirmed by reading, fixed | `tests/test_glm.py::test_the_refit_budget_gives_the_later_groups_up` | 31ad0f5d |
| BANK-7 | rule kept (the coordinator, 2026-11-23: no defect is shown): `refit_budget`'s docstring says a half-life grid reserves once per value; budgeting `max_state_bytes` per group across a grid changes what the setting means | -- | -- |
| SPAN-3 | deferred with SPAN-2 | -- | -- |
| BANK-2 | confirmed, fixed in Rust and Python | `tests/test_skip_learned.py::test_a_clock_that_is_not_a_finite_number_is_kept_for_the_bank_to_refuse`, `bank.rs::skip_learned_keeps_an_infinite_clock_and_drops_a_closed_group` | 31ad0f5d |
| BANK-4 | deferred | -- | -- |
| BANK-6 | rule kept (the coordinator, 2026-11-23: no defect is shown): `refit_budget`'s docstring keeps a reservation for good ("once the bound is reached no later group refits"); handing it back at `drop_groups` lets later groups refit | -- | -- |
| BANK-8 | confirmed by reading, fixed (`group_label`, one place) | `tests/test_error_messages.py::test_a_notice_for_the_null_group_names_it` | 31ad0f5d |
| BANK-9 | confirmed by reading, fixed: `format_duration` takes any `i128`; `audit`'s duration columns give a span past 64 bits as null | `tests/test_error_messages.py::test_a_step_back_past_292_years_is_said_at_its_length` | 31ad0f5d |
| PY-7 | name collision confirmed, fixed. `inf * 0` is the rule kept: the bank skips a row whose weight is not usable, so both copies are skipped as the native `embargo` skips the row | `tests/test_label_delay.py::test_the_doubled_stream_takes_an_input_holding_its_helper_columns_names` | 31ad0f5d |
| BANK-11 | deferred | -- | -- |
| BANK-12 | deferred | -- | -- |
| PY-14 | deferred | -- | -- |
| TEST-9 | deferred to batch a (`garch`) and j (tests) | -- | -- |
| CI-5 | deferred to batch j | -- | -- |
| MERGE-4 | deferred | -- | -- |
| PY-11 | confirmed by reading, fixed | `tests/test_check.py::TestValues::test_an_expectiles_ridge_is_told_its_own_way_out` | 31ad0f5d |

STRM-3's fix moves `summary()["weight_sum_settled"]` on a stream with rows
of weight 0, which the bank fixture `bank_label_clock` freezes: the current
set is regenerated, and the schema-71 set's frozen frame of that one field
is updated to what its state, loaded and continued, now reports; its state
bytes are untouched. `scripts/compare_release.py --against 2.0.0` says
every move declared (the probe reaches none of this batch's moves; each is
declared in the CHANGELOG).

### What the coordinator and the user decided (2026-11-23)

- **STRM-1**, the user: "Score as predict". Fixed as above; the record of
  the zero-weight audit, `ZERO-WEIGHT-AUDIT.md`, has the change and its
  cost.
- **STRM-3 (1), BANK-6, BANK-7**, the coordinator: the documented rules
  kept, since no defect against them is shown.
- **The v71 `bank_label_clock` readout**, the user: "Accept", as a moved
  readout, not an exception (`STATE-CONTINUATION.md`).

- **STRM-1's move on the kept states**, the user: "Widen exception 25"
  (rule 5's twenty-fifth exception widened, no new number;
  `STATE-CONTINUATION.md`).
- **REG-3**, the coordinator: no decay. The review's mechanism is not the
  cause, below.

### REG-3: not reproduced as stated

The review reads the slope that stays near 0 after the weights fall a
million-fold as the frozen unit of weight `ŵ = Σw² / Σw` freezing the
scaler. Under the default `schedule="constant"`, `sgd` multiplies a row's
gradient by its weight, as scikit-learn's `SGDRegressor` multiplies it by
`sample_weight`, so 275 rows at weight 1e-6 move the slope about a millionth
as far as rows at weight 1, whatever `ŵ` is: 7.2e-6 standardized and 8.2e-6
not after the fall; 7.8e-6 and 8.3e-6 on weights of 1e-6 on every row (no
fall, `ŵ` 1e-6 from the first row); 2.79 and 2.82 on weights of 1.
scikit-learn's `SGDRegressor` (constant rate 0.01) gives 9.2e-6 on the same
rows at `sample_weight=1e-6` and 2.87 at 1. The test pinning this is
`tests/test_sgd.py::test_a_fall_in_the_weights_slows_the_fit_as_scikit_learn_s_sample_weight_does`.

Open, to measure over seeds later and not a fix: where the frozen `ŵ` does
act is the scaler's floor, under `schedule="adagrad"`, whose step is
scale-free: after the fall the standardized slope read 0.301 against 0.321
unstandardized, about 7% lower (one seed, 300 rows), where even weights read
about 0.32 either way.

## Batch g: the regressions: `ewridge`, `sgd`, `ftrl`, `huber`/`quantile` and their refits, `holt`, `lasso` and LARS, `constraint`, the Gram merge (PLAN 578g)

On `review-578g`, cut from `integration` at 1e0e0945. "Confirmed" means
the named test failed on 1e0e0945 for the review's reason. REG-2, REG-3,
REG-14 and MATH-7 are batch c's. Test paths are under
`crates/rheon-core/src/` unless they start with `tests/`.

| ID | verdict | test | commit |
|---|---|---|---|
| REG-1 | confirmed (both parts: `frexp(0)` panics in a debug build on an ageing row across a gap of 2,000 half-lives, and a learned row across it writes a factor of 0 the load refuses as damaged), fixed: a run of decays of 0 lets every held row go, and a buffer's factors may be 0 | `held_median_tests.rs::a_run_of_decays_of_zero_forgets_every_row`, `robust/refit_tests.rs::a_gap_past_the_decays_underflow_with_a_refit_open` (huber and quantile), `sgd/refit_tests.rs::a_gap_past_the_decays_underflow_with_a_refit_open`, `huber_refit.rs::tests::a_buffer_that_was_not_written_so_does_not_fit` (a factor of 0 now fits) | c2b55214, 725cf5ef |
| LASSO-1 | confirmed at the descent (on `[[1, 2], [2, 1]]`, not positive semidefinite, both sweeps and the Gram path settle at `[inf, -inf]`); the existing `descend_on_a_dense_problem_is_the_serial_descent_to_the_bit` had settled at infinities too, at 90 features. Fixed: a settled descent converges only at finite coefficients | `lasso.rs::tests::a_descent_that_diverges_is_not_converged`, `gramfit/tests.rs::a_descent_that_diverges_is_not_converged` | b99f605b |
| REG-4 | confirmed (`y = 0, 10` a step of `1e-308` apart: level NaN from the next row), fixed: a slope past the doubles is read as a step of no clock, and a mean whose terms overflow is formed as the mean of shares; a finite trend and level keep their bits | `holt.rs::tests::a_slope_past_the_largest_double_leaves_the_fit_finite` | f1408c3b |
| REG-5 | confirmed (weights `2^-660`: the forecast of a line off by 41%; `2^330` beside a step of `1e60`: no slope), fixed where `S0·w·ā²` is not a normal double, its bits kept elsewhere (measured: 240,000 values of a 2,000-row, 8-target stream under five half-life pairs identical to the bit). The review's mean-form state was not taken: it changes the layout and every output | `holt.rs::tests::scaling_every_weight_by_a_power_of_two_moves_no_prediction` | c2bc3dde |
| REG-6 | confirmed (a refit on the warm-up's switching row: intercept -3.08 against the projection's -1.62), fixed: the switching row's map is the scaler's | `sgd/refit_tests.rs::a_refit_on_the_switching_row_is_projected_with_the_scalers_means` | f37c8414 |
| REG-8 | confirmed (`sgd(standardize, coef_sum=1)` on features near `1e-155`: coefficients `1e155` and off the sum), fixed with REG-19 | `constraint.rs::tests::scales_at_the_ends_of_the_doubles_project_as_ordinary_ones`, `sgd/gap_tests.rs::a_sum_on_features_near_the_smallest_doubles_holds` | 8b2f1ede |
| REG-19 | confirmed (two coordinates `10^156` apart: the point off the sum by a third), fixed: where the direct projection is not sound (a step not a normal double, a finite bound's breakpoint lost, or the point off the sum) it is made again by scale groups, in the caller's units. The oracle is the definition, two coordinates solved in closed form at scale ratios `10^80` to `10^300`. The review's third mode (`coef_min = 1e-220` at a scale of `1e-110` rounding to 0 in the standardized coordinates) is the representation's: the state holds `c·s`, which is not a double there | `constraint.rs::tests::two_coordinates_far_apart_meet_the_definition`, `::scales_far_apart_project_onto_the_sum` | 8b2f1ede |
| REG-9 | confirmed (`ftrl`: `n = inf` after two gradients of `1.2e154`; `sgd(adagrad)` at an infinite clip: a sum of `inf`), fixed: a row whose sum would pass the doubles is skipped, under `ftrl` by a bound on the largest sum (a factor of 2 conservative at the edge), which also covers the step at a small `alpha`; `ftrl`'s largest gradient is formed in the update's order | `ftrl.rs::tests::a_row_that_would_overflow_a_sum_is_skipped`, `sgd/gap_tests.rs::adagrads_sums_never_pass_the_largest_double` | 381e853c |
| REG-10 | confirmed (`σ²` NaN after a weighted square past the doubles and a gap), fixed: the mean of shares where the plain mean overflows, and the weight aged where the square itself does | `ewridge/tests.rs::a_weighted_square_past_the_largest_double_leaves_sigma2_finite` | 89e37d20 |
| MATH-8 | confirmed (weights `1e200`: 2 knots against 5), fixed: the weights taken to a power of two near 1, exactly; weights more than `2^256` apart refused by name, since no common scale holds their correlations | `gramfit/tests.rs::penalty_weights_near_the_ends_of_the_doubles_walk_the_scaled_path` | bbb0d67a |
| LASSO-3 | confirmed (one solve failure from a warm start of `±inf` at 2 and at 45 features), fixed: a coordinate whose start is not a number starts at 0 | `lasso.rs::tests::a_warm_start_past_the_largest_double_starts_at_zero` | 76144cf2 |
| REG-18 | **a decision for the user.** Confirmed: `y = 2x₀ + 20x₁`, `x₁` in {0, 1}, two outliers, the refit on row 60: `[0.0005, 2.0005, 19.999]` unstandardized, `[16.67, 0.29, 0.0]` under `standardize=True`. It contradicts PLAN §7 ("a feature with ~0 variance is dropped from the solve for that step (coefficient 0)"), which `robust.rs` keeps under `standardize` with tests pinned to it (`a_band_row_that_changes_the_kept_columns_drops_the_factor`, `a_nudge_against_a_band_system_of_no_column_reads_the_intercepts_leverage`, `a_band_system_with_no_column_holds_no_factor_and_loads_back`, `each_damaged_band_system_is_refused_on_its_own`). The review's side: the quantile loss's band Gram holds only the rows near the fit, so its variance is not the feature's, and a 0/1 feature constant inside the band is dropped while the rows outside hold it at 20. The rule's side: a column of no spread has nothing to standardize by. Tried: keeping such a column at a scale of 1, as `standardize=False` does, fixes the case and fails those four tests; it changes the band systems' kept columns, which states hold. The review's other fix (scales from every row) needs a new accumulator in the state. The Newton stall read as converged follows from the dropped column; a majorizer pass drops it too | -- (the case above, not committed: it fails until the decision) | -- |
| LASSO-2 | confirmed (at 50 features a half-life of 30, which settles at 87 rows against the bound of 115, excused on rows 1 to 3), fixed: the decay per row solved from the Kish size and the product of decays together, the first row at no clock. The docstring's "a scale of 1" for a window corrected to 0. The existing test's comment, "about row 65", was row 139 | `gramfit/tests.rs::a_gram_whose_first_row_came_at_no_clock_is_read_at_its_limit` | 7f09e765 |
| SOLVE-1 | confirmed (`diag(1e308, 1e308)` failed every rung), fixed: the first rung adds nothing whatever the base, and an infinite trace's base is the sum of each entry over `k` | `solve.rs::tests::a_trace_past_the_largest_double_factorizes` | eb20682d |
| REG-12 | not started in this batch: the fix the review gives keeps a power-of-two scale of the weights in `Warmup`, `Level`, `Rls::s2` and `EwCov::q_sum`, a layout change in four state structs with fixtures to regenerate and continuations to check | -- | -- |
| REG-13 | not started in this batch | -- | -- |
| MATH-4 | confirmed (an intercept 10 off least squares': `resid_var` unchanged), fixed: the intercept's error squared added, 0 to a rounding for `solve`'s coefficients; docs/GRAM.md's formula with it | `tests/test_gram_module.py::TestCoefStats::test_an_intercept_off_the_fit_counts_in_the_residuals` | 4bc0d459 |
| REG-20 | **not reproduced for the review's reason; the gate's rule is a recorded design.** `sgd(loss="logistic", learning_rate=80, inv_scaling)` on a separable stream is withheld on all of rows 900 to 999, as the review says, but the gate took 615 nonzero gains in that stretch: its verdict is the rule's own on live gains (`q2 = 17.6 > 2q = 15.5`), not a freeze from the early return on `q == 0`. The module docs record that a gain of 0 is left out. The review's fix, decaying `q` and `q2` on such rows, scales both alike and cannot change `q2 > 2q` | -- (a probe run, not committed) | -- |
| REG-16 | confirmed (a span's state without its place loaded), fixed: refused as damaged, as `sgd`'s is | `robust/span_tests.rs::a_state_whose_place_does_not_match_its_span_is_refused` | cfafe5d9 |
| MATH-3 | confirmed (a part with a target of no weight and NaN moments made the pooled moments NaN, `within` or not, with and without an intercept), fixed: such a part's moments read 0. The `Σ w²` fabrication REG-14 names is `_target_q`'s, unchanged here | `tests/test_gram_module.py::TestMerge::test_a_part_with_no_weight_on_a_target_takes_nothing_from_it` | bb5058cf |
| REG-17 | not reproduced: against the same rows shifted back exactly, the quantile refit's slope moves by `1.9e-3` at a feature of `1e14 + N(0, 1)` and `2.4e-7` at `1e10`, inside the `64 · shift · ε` every centred moment of such a feature carries | `robust/equivariance_tests.rs::a_shift_of_the_feature_moves_the_quantile_refits_slope_by_rounding` | 2b993c11 |
| MATH-6 | not started in this batch | -- | -- |
