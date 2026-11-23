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
| STRM-1 | confirmed (claims 1 and 2); **decision for the user**, below. Claim 3 rides on the same mechanism; claim 4 is the rule `predict` keeps (task 120) | `tests/test_label_delay.py::test_a_zero_weight_row_releases_no_label_a_later_row_before_it_would_not_see`, `::test_a_stream_ending_on_a_zero_weight_row_saves_what_it_would_without_it` (strict xfail) | 31ad0f5d |
| REG-2 | confirmed by reading and by test, fixed: an ageing row's new run is held to the span's cap | `crates/rheon-core/src/robust/span_tests.rs::an_ageing_row_is_held_to_the_spans_cap`, `sgd/span_tests.rs::an_ageing_row_is_held_to_the_spans_cap` | 549908ef |
| REG-3 | confirmed; **decision for the user**, below | `tests/test_sgd.py::test_a_fall_in_the_weights_does_not_freeze_the_fit` (strict xfail) | e4bc73e4 |
| MERGE-1 | confirmed (a NaN and a step back past the slice), fixed by the break the review's Pass 2 gives | `tests/test_stream_merge.py::test_a_slice_reads_no_chunk_past_its_last_row` | 31ad0f5d |
| STRM-2 | confirmed, fixed: a fresh-session row of weight 0 reads a fresh stream's diagnostics and `learned_clock`; `predict`'s `learned_clock` too. `predict` already scored the fresh class with a fresh stream's diagnostics, so that half of the Pass-2 note was not reproduced | `tests/test_zero_weight_rows.py::test_a_zero_weight_row_that_starts_a_fresh_session_reads_a_fresh_streams_diagnostics` | 31ad0f5d |
| STRM-3 | (2) confirmed, fixed (`weight_sum_settled`'s mean weight). (1) **decision**: `ModelBank.last_row`'s docstring says a row of weight 0 replaces it. (3) not changed: the unstable-rate notice reports the state the row is read against, which a row of weight 0 reads as the next row will, and no output of a row of positive weight moves. (4), (5) the rule kept: `ZERO-WEIGHT-AUDIT.md` leaves the data summary and the clock range describing every row fed, and W9 made a row of weight 0 no session boundary. (5)'s refit reservation is BANK-6's question | `tests/test_zero_weight_rows.py::test_rows_of_weight_0_leave_the_settled_weight_where_it_was` | 31ad0f5d |
| CLOCK-1 | confirmed on two finite steps (the review's one step is refused already, PB7), fixed by a refusal by row, on a bound of the removed time | `crates/rheon-polars/tests/bank.rs::a_clock_whose_removed_time_passes_the_largest_double_is_refused_by_row` | 31ad0f5d |
| BANK-1 | (1) confirmed, fixed in Rust and Python. (2) the rule kept: a row of weight 0 is in no clock, so the bank holds no place for one after its last clock, and replaying it moves nothing. (3) deferred: specs that group or weigh differently are PLAN 566's documented rule | `tests/test_skip_learned.py::test_a_monotone_close_drops_the_groups_it_closed`, `crates/rheon-polars/tests/bank.rs::skip_learned_keeps_an_infinite_clock_and_drops_a_closed_group` | 31ad0f5d |
| STRM-4 | (1) confirmed, fixed: `EwClass::target_n_eff_into`, the classes' weights. (2) confirmed, fixed. The Pass-2 addition (a held row sets `learned_clock` when queued) not reproduced by reading: a held row's plan has `learn = false`, and its release sets it | `tests/test_zero_weight_rows.py::test_ew_class_withholds_until_its_labelled_rows_reach_min_weight`, `::test_the_learned_clock_is_the_last_row_with_a_target_learned` | 4a79f8ac, 31ad0f5d |
| REG-14 | confirmed (both parts), fixed in `gramfit::merge`; the same `Σ w²` fabrication in Python's `gram.merge` (`_target_q`) is MATH-3's, batch g | `crates/rheon-core/src/gramfit/merge_tests.rs::a_merge_is_the_same_at_every_scale_of_weight`, `::a_weighted_part_without_a_kish_size_leaves_the_merged_one_unknown` | c692174b |
| MATH-7 | confirmed, fixed in `merge`, `merge_within` and `within_cross` | `merge_tests.rs::a_part_with_no_weight_for_a_target_adds_nothing_even_where_its_moments_are_nan` | c692174b |
| SPAN-1 | deferred: carrying a span's nanoseconds into `clock_span` touches every span-capable model's configuration; the loss is 1 ns at spans past 97 days | -- | -- |
| SPAN-2 | deferred (the `glm` and `expectile` span caps; with SPAN-3 and REG-11, one change to make every span cap atomic) | -- | -- |
| REG-11 | deferred with SPAN-2 | -- | -- |
| MERGE-2 | deferred | -- | -- |
| MERGE-3 | confirmed by a run (a state holding `a: [2.0], b: [2.0]`, `.head(1)` on `a2 = [3.0]`, `b2 = [3.0]`: the slice returns the held `(2.0, "a")`, `b`'s row stays held, and `a2.slice(1)` drops `3.0`); deferred: the fix is a choice of surface, a count of each input's rows read beside the output or a refusal of a slice over a loaded state that holds a tie, besides the docstring | -- | -- |
| BANK-5 | confirmed by reading, fixed | `tests/test_glm.py::test_the_refit_budget_gives_the_later_groups_up` | 31ad0f5d |
| BANK-7 | **decision**: `refit_budget`'s docstring says a half-life grid reserves once per value; budgeting `max_state_bytes` per group across a grid changes what the setting means | -- | -- |
| SPAN-3 | deferred with SPAN-2 | -- | -- |
| BANK-2 | confirmed, fixed in Rust and Python | `tests/test_skip_learned.py::test_a_clock_that_is_not_a_finite_number_is_kept_for_the_bank_to_refuse`, `bank.rs::skip_learned_keeps_an_infinite_clock_and_drops_a_closed_group` | 31ad0f5d |
| BANK-4 | deferred | -- | -- |
| BANK-6 | **decision**: `refit_budget`'s docstring keeps a reservation for good ("once the bound is reached no later group refits"); handing it back at `drop_groups` lets later groups refit | -- | -- |
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

### The decisions, both sides

**STRM-1: a row of weight 0 under `embargo` or `label_clock` releases the
held rows into the live models.** `ZERO-WEIGHT-AUDIT.md` chose it: such a
row is scored from the labels known at its clock, so it releases the held
rows whose wait it has seen out first, "as the next row would". The
review's side: when the next row of positive weight is not later (a row
of weight 0 counts in no step-back rule, hard rule 9), that row is scored
by a model that has learned a label the embargo still holds at its clock
(rule 2), and a stream that ends on such a row saves a state that has
learned what the stream without it holds (rule 9). Options: (a) the
review's: release into a scoring copy of each instance, made once for a
run of rows of weight 0 and dropped at the next row that counts; the
scoring stays the record's, the live state never moves, at the cost of a
copy of every instance's state per such run (to be measured); (b) score
the row from the state as it stands, as `predict` does: no copy, rule 2
and 9 hold exactly, but such a row no longer sees the labels matured at
its clock, and `label_clock.rs`'s doubled-stream oracle moves; (c) keep
the rule and document the step back.

**REG-3: the row-weight unit `ŵ = Σw² / Σw` is frozen at the warm-up**
(review fable2, C1's rule A: a stream whose weights are all scaled by `c`
is read as the unscaled one). After the weights fall a million-fold, each
row steps `sgd`'s fit by a millionth of a row, and the slope stays at the
old regime's (2e-5 where the new rows say 3, after fourteen half-lives).
The review proposes decaying `Σw` and `Σw²` on the model's clock so the
unit follows the weights. That keeps scale invariance but moves every
`sgd` and `pa` output on a stream of uneven weights, and on even ones by
rounding unless guarded.

**STRM-3 (1): `ModelBank.last_row` after a row of weight 0.** The
docstring says such a row replaces it ("the last one the model
predicted"); the review reads the plan's "the last row it learned from".

**BANK-6 and BANK-7: the refit budget.** Each is a change to a rule the
docstrings state (a reservation kept for good; a grid reserving per
value), not a defect against them.
