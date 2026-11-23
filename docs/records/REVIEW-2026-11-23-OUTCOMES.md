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
