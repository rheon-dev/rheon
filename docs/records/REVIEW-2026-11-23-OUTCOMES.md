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

## Batch e: the drift detectors, checks, audit, calibration, conformal, breaks, `rp.eval` and the metrics (PLAN 578e)

On `review-578e`, cut from `integration` at 1e0e0945. "Confirmed" means
the named test failed on 1e0e0945 for the review's reason. A test marked
`should_panic` holds a confirmed defect whose fix is the user's to allow;
it fails once the fix lands, and is then turned into a plain test.
GLM-METRICS is batch c's.

| ID | verdict | test | commit |
|---|---|---|---|
| DRIFT-4 | confirmed in part, fixed. The endless loop was gone before this batch: de1be89a (mutants-w11) bounded `ks_h`'s scan. A NaN still loaded (`Kswin::sound` read no value) and `Kswin::update` took one, and the scan then read a wrong statistic. Both refused now | `crates/rheon-core/src/drift/kswin_tests.rs::a_restored_ring_holds_only_finite_values`, `::a_non_finite_value_is_no_row`, `::the_scan_ends_on_a_nan` | 5cac6698 |
| AUDIT-1 | confirmed (the review's own case panicked at `audit.rs:603` in a debug build), fixed: a counter below the Misra-Gries cut goes | `crates/rheon-core/src/audit/tests.rs::a_counter_below_the_cut_goes_and_wraps_nothing` | d8797848 |
| DRIFT-2 | confirmed (the ghost `V = 1.2e24` on a window of zeros), fixed: `S` and `V` are taken again from the buckets left on each drop. river's flags are kept under its `restart`: `tests/test_second_opinion_drift_kinds.py` passes, all 33 | `crates/rheon-core/src/drift/adwin_tests.rs::outliers_that_leave_the_window_leave_no_variance_behind` | 6c29a3e0 |
| HAWKES-1 | confirmed (a debug build panicked on each multiplication; a release build wraps), fixed: the bytes saturate at `u64::MAX`, twice `stat_size` is checked, and the casts saturate | `crates/rheon-polars/src/spec/drift_tests.rs::a_ring_past_what_its_bytes_can_count_is_refused`, `crates/rheon-polars/tests/spec_refusals.rs::a_hawkes_window_past_its_state_cap_is_refused` | 4692ef27 |
| DRIFT-5 | confirmed in part, fixed. `Adwin::sound` took an empty top level, and `restores_as` checked no kind but ADWIN and KSWIN. Every kind is now held to its shape. The endless loop was gone before this batch (the eviction loop is bounded by the buckets held); the window stayed past its cap instead | `crates/rheon-core/src/drift/detector.rs::tests::every_kind_is_held_to_its_shape`, `::an_empty_top_level_is_no_window`; `drift.rs::tests::a_restored_detectors_numbers_are_finite`; `cusum.rs::tests::a_restored_cusums_numbers_are_finite` | 42bc17d2 |
| DIAG-11 | confirmed by measurement: at `expectile(0.9)` on an exact Gaussian fit, `calibration_wald` and `cusum` rejected on 100% of rows 1,000-4,000; on an exact Poisson `glm`, `breusch_pagan` and `jarque_bera` rejected on 100% and `ramsey_reset` on 24%. Fixed by refusal: `glm` refuses `emit_specification` and `emit_tails`, and `expectile` off 0.5 refuses `emit_calibration` and `emit_breaks`. Both models are unreleased. The checks they keep hold their level | `tests/test_expectile.py::test_an_expectile_off_the_mean_refuses_the_mean_checks`, `::test_the_mean_expectile_keeps_every_check_at_its_level`; `tests/test_glm.py::test_a_glm_refuses_the_checks_of_a_constant_normal_residual`, `::test_a_glms_mean_checks_hold_their_level_on_an_exact_fit` | 857bff13, aaa5d7f8 |
| DIAG-1 | (1) **confirmed; fixed by the user's fallback form (2026-11-23), rule 5's thirty-first exception.** The long-run variance takes the null's lag 0 wherever that leaves it above 0, to the bit as before, and the terms' own lag 0, Newey and West's, only where it does not, where it read `None`. On a stream whose residuals alternate at ±30 after a quiet spell, 78 rows of 170 fall back, and each reads statsmodels' `S_hac_simple` at the same lags to 1e-9. First built with the empirical lag 0 alone (f4a9e87a, held back), which kept the size and lost power. The tables, on the same simulations (Rust, run-once size over 2,000 null streams of MA(h−1) residuals at `L = 2h`, a `cusum` past ±1.96 on the last row; power over 300 streams at a half-life of 50 rows and 4 lags, the mean rows to a `cusum` past ±3 after a sustained shift): size at h = 1, 3, 5 is 4.6%, 6.3%, 6.6% under the null's form and the fallback alike (4.3%, 5.8%, 6.7% under the empirical alone); rows to a flag at shifts of 1, 3, 10 and 1000 sd are 25.6, 17.5, 12.2, 0.5 under the null's form and the fallback alike (28.8, 22.1, 22.3, 197 under the empirical alone). `Coverage::wald` takes the same fallback: the review's `[hit, miss, hit]` at `q = 0.9` under 10 lags reads a statistic. (2) **not reproduced** as a break unseen: a 1000-sd break under 20 lags at a memory of 20 reads 36 on its row and past 1.96 on each of the 180 rows after it | `crates/rheon-core/src/breaks.rs::review_2026_11_23::the_null_lag_0_is_read_where_positive_and_the_empirical_one_where_not`, `::the_long_run_variance_stays_positive_under_alternating_shocks` (was `should_panic`), `::a_large_break_passes_the_cusums_boundary_under_a_horizon`, `tests::a_long_run_variance_the_lags_cancel_reads_the_terms_own_lag_0`; `calibration.rs::tests::a_coverage_variance_the_lags_cancel_reads_the_indicators_own_lag_0`; `tests/test_second_opinion_diagnostics.py::TestTheCusumsLongRunVarianceFallsBackToNeweyWest` | a040084d, f4a9e87a, 28d7d543 |
| DIAG-2 | confirmed: at a target level of `1e6` the HAC `calibration_wald` read `3.9e-5` off statsmodels', and at `1e8` 0.722 where it is 0.593. Fixed: a Wald test of every coefficient is read in the frame the sandwich's sums are kept in, `T⁻¹ d` against `V_c`. Held to statsmodels' `S_hac_simple` in the centred frame to `1e-6` at both levels; elsewhere it moves by rounding (`bank_diagnosed`'s frozen row 0 by one ulp) | `tests/test_second_opinion_diagnostics.py::TestUnderAHorizonTheTestsAreNeweyWest::test_the_calibration_at_a_large_level_is_statsmodels_in_the_centred_frame`; `crates/rheon-core/src/sandwich.rs::tests::the_covariance_in_kept_buffers_is_the_allocating_one_to_the_bit` | 75af1b1a, 81be5d5f |
| DIAG-3 | (2) confirmed, fixed: a prediction `2e51` from the origin, or a residual of `2e77`, left the sums infinite and the state unloadable. Such a row now only ages. (1) **confirmed, deferred (the coordinator, 2026-11-23) as a state change for a later batch**: RESET about a first prediction `1e4` or `1e5` from where the predictions settle reads `None`, and at `1e3` and `1e6` it reads off. The review's fix, solving in correlation form, was tried and does not recover it: the powers' centred moments have lost the curvature in rounding. A fix moves the origin (re-centring the moments of `q`, `q²`, `q³` by their binomial transform while they are still exact), which is state | `crates/rheon-core/src/specification.rs::review_2026_11_23::a_row_whose_powers_overflow_leaves_the_state_loadable`, `::reset_reads_alike_about_a_far_origin` (`should_panic`) | c25249b8 |
| DIAG-4 | confirmed, all three, fixed: `jarque_bera` is `None` where `Σω²` leaves the normal doubles (it read `inf` after 538 half-lives); a residual whose fourth power overflows only ages; `skew` and `kurtosis` are `None` where their denominators or `m_4` leave the normal doubles (they read NaN near `1e-90`). The review's form `(m_4/m_2)/m_2` was not taken: `m_4` itself underflows there, so it would read `-3` | `crates/rheon-core/src/tails.rs::tests::a_memory_decayed_past_its_kish_size_reads_no_statistic`, `::a_residual_whose_fourth_power_overflows_leaves_the_state_loadable`, `::tiny_residuals_read_their_shape_or_none` | 7be09ef1 |
| DIAG-5 | **not reproduced** as a defect. The band keeps the buckets between a sketch's levels by design (`EwQuantile`'s "The band"). The review's value of `1e-300` lies between a level at 0 and one at 1, and kept there the lower level steps off 0 onto the unbounded sketch's bucket, to the bit. Every bounded sketch the library keeps has one level (`huber`'s and `quantile`'s MAD, the bins' medians), and the same rows hold it to `one_level_buckets` | `crates/rheon-core/src/stats/band_tests.rs::a_value_between_two_levels_is_kept_and_one_level_holds_its_band` | af89c70b |
| PY-1 | confirmed, both clocks, fixed: the edge is the clock less its floor remainder, in the clock's own units where `every` is whole in them, and an edge below the type's least value is refused | `tests/test_eval_scoring.py::test_an_int64_clock_whose_window_edge_passes_its_least_value_is_refused`, `::test_a_duration_clock_past_292_years_is_windowed_in_its_own_units` | f4a34b25 |
| CLUST-3 | confirmed. First held as a rule-5 matter (8ca008d5, the `micro` guard alone). Then fixed by the user's decision (2026-11-23: "Exception 30, full fix"): the mean weight is taken at the old mean's and the row's shares in `micro` and in the conformal band, and rule 5's thirtieth exception holds the kept `v61`-`v71` `micro` and `v71` `bank_conformal_auto`/`bank_label_clock` sets to bounds (`STATE-CONTINUATION.md`). Python's replays (`tests/test_conformal.py::replay`, `tests/reference_cluster.py`) take the same form | `crates/rheon-core/src/cluster/micro.rs::tests::a_mean_weight_cancelled_to_nothing_admits_no_far_point`, `::the_mean_weight_survives_a_heavy_row_decaying_away`; `conformal.rs::tests::the_mean_weight_survives_a_heavy_row_decaying_away` | 8ca008d5, 3dd1d528, e819c617 |
| DRIFT-1 | **contradicted a recorded decision (task 508: ADWIN bit-equal to river); then built by the user's decision** (2026-11-23: "Allow both by parameter"). river 0.26.1's `drift.ADWIN` has the same `R = 1`: on the absolute value of `N(0, σ)` with a threefold rise at row 2,000 it found the rise in 15 rows at `σ = 1` and `1e3`, in 655 at `1e-2`, and not in 8,000 rows at `1e-3`. `drift_options`' `range` takes river's 1 (the default, held to river by `test_adwin_is_rivers`, `range1` among its cases), a number in the target's units, or `"window"`, the range the window holds, kept as each bucket's least and greatest value. No other library computes a range-adaptive ADWIN: anomstream-core (Rust) takes a range the caller declares, in a Hoeffding form, and irithyll-core and menelaus take none. So the window's form is held by its scale invariance (the same flags at `c` of `2^±k`, `1e-3` and `7.3`, to the bit), the hand-written Bernstein bound at a known `R`, and its range against a longhand over the rows held. The docs' "scale-free" now says KSWIN is, and ADWIN is only under `range="window"`. `bank_drift_ranges` freezes both new forms | `crates/rheon-core/src/drift/adwin_tests.rs::a_fixed_range_is_the_bernstein_bound_at_that_range`, `::the_windows_range_is_the_rows_it_holds`, `::under_the_windows_range_the_flags_do_not_depend_on_the_unit`, `::the_windows_range_is_saved_and_rivers_form_writes_no_range`; `crates/rheon-polars/src/spec/drift_tests.rs::range_builds_adwins_range`; `tests/test_second_opinion_drift_kinds.py::test_adwin_is_rivers[range1]` | 90dc007e |
| DRIFT-3 | **contradicted a recorded decision (task 508); the user kept `d (z − k)` (2026-11-23), documented, no output or state moved.** At a mean step of 1, `h = 5`, `k = 0.5`, two-sided, the ARL in rows (2,000 runs each, error about 2%) is 469 at unit steps, 79 at `Exp(1)` steps and 30 at bursty steps (0.1 at 90%, 9.1 at 10%). The `√d z − d k` form the user first chose was simulated and not built: it reads 472, 577 and 1,840, and at constant steps of 0.1 and 2 it reads 2,090 rows (209 clock units) and 370 rows (740 units). So it holds the false-alarm run neither in rows nor in clock time across step patterns, since the sums are reflected at 0 and tested against `h` only at the rows. The CUSUM's docstring, `rheon.spec`'s and `DIAGNOSTICS.md`'s *Has it broken?* give the measured runs and recommend a regular or a row clock where a steady false-alarm rate matters; a test holds the direction | `crates/rheon-core/src/drift/cusum.rs::tests::irregular_steps_run_to_a_false_alarm_sooner_than_regular_ones` | 4d13e43b |
| DRIFT-6 | confirmed, all three, fixed: an overflowing `r / σ` is the largest double and flags; a step of 0 adds nothing to CUSUM; Page-Hinkley starts its sums again at an excess overflowing below. No finite path moves, so no state continuation moves | `crates/rheon-core/src/drift/detector.rs::tests::overflowing_rows_flag_or_pass_and_leave_the_detectors_reading` | f25e0346 |
| DRIFT-7 | confirmed, fixed: `ddm`'s `drift_threshold` must be above 1 (at 0.5, 10 flags in 310 steady rows); `Ddm::sound` follows; the field's doc said `p_min + s_min` | `crates/rheon-polars/src/spec/drift_tests.rs::each_setting_is_held_to_its_range_on_both_sides`; `crates/rheon-core/src/drift/ddm.rs::tests::a_threshold_at_or_under_one_flags_a_steady_stream_every_warm_start` | a5fbb55d |
| REG-15 | confirmed, fixed: `update_keeping` holds the history to `min(W, floor)`, `W` the history's weight before the gap. Kalman reads it as the weight now over the gap's decay, so a gap split by rows of weight 0 is one gap (`a_gap_split_by_zero_weight_rows_is_one_gap` holds). The review's `floor.min(self.w_sum)` broke that test, since the zero-weight rows had already aged `W` | `crates/rheon-core/src/ewdiag.rs::tests::the_floor_holds_a_history_up_to_its_own_weight_and_no_further`, `kalman/tests/gap_moments.rs::update_keeping_is_update_but_where_the_floor_binds` | de4952c4 |
| DIAG-6 | confirmed, fixed: the upper quantile is read from the lower tail only where `1 − α/2` rounds to 1, so every other level keeps its bits; a weight that is not finite only ages | `crates/rheon-core/src/conformal.rs::tests::a_coverage_just_under_one_and_a_weight_that_is_no_number_keep_it_finite` | 880b92bb |
| DIAG-12 | confirmed, fixed, through a prediction far past its target, which no bound holds. Inside the bound a sum takes about 45 million rows of `w e² = 4e300` to overflow, so the review's "valid ≤ 1e100 inputs" needs a prediction outside it. Such a row is aged and not scored; `Influence::of` reads `e·sqrt(1 − 1/infl²)` where `infl²` overflows | `crates/rheon-core/src/influence.rs::tests::overflowing_squares_leave_the_spread_finite_and_the_influence_read`; `stats.rs::tests::a_weighted_square_that_overflows_leaves_the_mse_finite` | 57b708b1 |
| AUDIT-3 | (1) confirmed, fixed: `unit_root_t` on `[0, 1e-155, 0, 1]` read `-0.0`; where the ratio or its denominator leaves the normal doubles it is taken in square roots. (2) deferred (the coordinator, 2026-11-23) as a state change for a later batch: scaling below `2^-64` in `scale_for` changes the moments' scale rule, which is state | `crates/rheon-core/src/audit/tests.rs::the_unit_root_t_of_a_tiny_spread_is_the_scaled_ones` | 1b4afe2d |
| DRIFT-9 | confirmed for a direct caller (`min_window_length = 0`, refused by the spec and now by `sound`): a steady 0.1 at one bucket a level flagged 83 times in 500 rows. Fixed: the scan ends at a split with no newer rows | `crates/rheon-core/src/drift/adwin_tests.rs::a_split_with_nothing_newer_is_no_cut_at_any_shortest_part` | b13b14dd |
| SPEC-8 | confirmed both, fixed: an `expectile` window (`window_size`, or `window_rows` without a clock column) is the fit's memory for the diagnostics; `window_rows` on a clock says its rows are no span of the clock; `garch` is told it does not decay | `crates/rheon-polars/tests/spec_refusals.rs::an_expectile_window_is_feature_healths_memory_and_garch_is_told_it_does_not_decay` | 679bf201 |
| DIAG-7 | **confirmed, deferred as a state change**: at a residual bias of `1e8`, `ljung_box` reads `None` where the unbiased residuals read 4.57. The fix keeps the sums about an origin, which is state | `crates/rheon-core/src/specification.rs::review_2026_11_23_diag7::ljung_box_reads_alike_under_a_bias` (`should_panic`) | 571e3c09 |
| DIAG-8 | confirmed, fixed: a row after a decay of 0 is studentized by nothing; a studentized fourth power, or a twin fit's `w² zᵢ zⱼ`, that overflows only ages | `crates/rheon-core/src/breaks.rs::review_2026_11_23_diag8::a_row_after_a_total_forgetting_is_studentized_by_nothing`, `::a_residual_whose_fourth_power_overflows_leaves_the_sums_finite`, `::a_twin_fit_row_past_the_doubles_only_ages` | 92e5fbc6 |
| DIAG-9 | **confirmed, deferred as a state change**: the EW lag-1 autocorrelation of `(−1)^t` reads −0.5 at `lam = 0.5`. The fix keeps the lagged leg's own mean, which is state; documenting the shrinkage is the other way | `crates/rheon-core/src/stats/band_tests.rs::the_ew_autocorrelation_of_an_alternating_stream_is_minus_one` (`should_panic`) | 571e3c09 |
| PY-6 | confirmed, all four, fixed: `diebold_mariano`/`clark_west` check `min_samples`; `half_life=True`, `every=True` or `None`, and `seqtest(targets="ret")` are refused by name | `tests/test_eval_inference.py::test_min_samples_is_held_to_a_whole_number_of_rows`, `::test_a_half_life_of_true_is_refused`, `::test_window_metrics_refuses_every_of_a_bool_or_none`, `::test_seqtest_refuses_targets_given_as_one_string` | c7c0bff6 |
| PY-15 | (1) **not reproduced**: with `emit_selected` and `emit_averaged` on, a grid's candidates read the same `mse`, `n` and `rank` (both are scored on the candidates' rows). (2), (3) confirmed, fixed: `seqtest` checks `min_weight`; `binary` refuses a string. (4) not reproduced as stated: `rheon.corr.shrink(R, x=x_nan)` returns R with `alpha = 0` rather than blaming `alpha` (not changed here). (5), (6) not reached | `tests/test_grid_search.py::test_the_selected_and_averaged_fields_move_no_candidates_score`; `tests/test_eval_inference.py::test_seqtest_refuses_a_min_weight_that_is_no_weight`, `::test_binary_given_as_a_string_is_refused` | 6724c5bf, de87d26d |
| DRIFT-8 | confirmed by reading (`r^{3/2}` operations and a vector a step), fixed: the ratios are built once, `h*` the same to the bit | `crates/rheon-core/src/drift/kswin_tests.rs::the_critical_h_of_a_large_sample_is_the_least_whose_tail_passes` | b1a4b91a |
| AUDIT-2 | (1) confirmed, fixed: audits merge only under the same clock settings. (2) confirmed, fixed: the module doc said a weight-0 row's step is counted. (3) **not reproduced** as an asymmetry: a negative weight is an unusable weight, whose row is counted with no weight, as `a_weight_that_is_not_usable_adds_no_weight` holds for NaN and the bound | `crates/rheon-core/src/audit/tests.rs::audits_under_other_clock_settings_do_not_merge`, `::a_negative_weight_is_counted_as_an_unusable_one_is` | c08db97a |
| DIAG-10 | (1) confirmed by reading, fixed: the twin fits' scratch threshold counts the intercept. (4) confirmed, fixed: a restored conformal band is held to its shape. (2), (3) not reached | `crates/rheon-core/src/conformal.rs::tests::a_restored_trackers_shape_is_checked` | 17c56d8e |
| PY-10 | not reached | | |

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
| MATH-6 | confirmed and fixed in the second round: see "Batch g, continued" below | d12cd115 (PLAN 578g, MATH-6: five edge cases of `rheon.corr` and `rheon.gram`, each test first) | see that table |

## Batch g, continued (PLAN 578g, second round)

On `review-578g-2`, cut from b7c81d64. The decisions: REG-18 the user's
(2026-11-23, "Scales from every row"), REG-12 and REG-13 the
coordinator's.

| ID | verdict | test | commit |
|---|---|---|---|
| REG-13 | confirmed (a target present on a quarter of the rows: `se_coef` over `sigma` 0.021 against statsmodels WLS's 0.041 on its rows), fixed: under `"pairwise"` each target's errors are read at its own rows' Kish size and column means, live or in the window | `tests/test_second_opinion.py::TestEwridgePairwiseStandardErrors::test_each_error_is_wls_on_the_targets_rows`, `crates/rheon-core/src/ewridge/tests.rs::pairwise_errors_are_read_at_the_targets_own_rows`; `tests/test_gram_threads.py`'s pairwise spec now takes `max_error_inflation=2` (300 features at 200 rows, the target's own size 5% below the Gram's, held past `sqrt(2)` on most rows) | 553aa82f, 6d0a1f89 |
| MATH-6 | confirmed, all five (NaN and `inf` counts gave NaN edges; an unsymmetric matrix read by its lower half; a NaN pair misnamed; a stale `cross_moments` width; `True` and `0.5` read as positions), fixed | `tests/test_corr_paths.py::test_the_noise_floor_refuses_a_count_that_is_not_finite`, `::test_an_unsymmetric_matrix_is_read_as_its_symmetric_part`, `::test_a_nan_off_its_mirror_is_named_where_it_is`; `tests/test_gram_module.py::TestMath6` | d12cd115 |
| REG-12 | confirmed (at weights `2^-664`: the warm-up count 0 and no switch, `rls`'s Kish ratio and every `EwCov`'s Kish size absent), fixed: `Warmup`, `sgd`'s `Level`, `Rls::s2` and `EwCov::q_sum` keep their sums of squared weights beside a power of two of their own (`sqsum`), exponent 0 and unwritten at ordinary weights (`q_sum` is written as `[q, s]` beside a power, one field, so the positional encoding the generated contract tests round-trip through does not slide), so every earlier fixture's bytes and continuations are unchanged. New fixtures `ewridge_tiny_weights`, `rls_tiny_weights` and `sgd_tiny_weights` freeze the new forms. The review's `sgd` 0.9 quantile held at the mean is not the count: at such weights AdaGrad's epsilon outweighs every gradient, a weight being a step size there (`tests/test_weight_scale.py`) | `sqsum.rs::tests`, `warmup.rs::tests::weights_too_small_to_square_count_as_rows`, `sgd/gap_tests.rs::weights_too_small_to_square_count_as_rows`, `rls.rs::tests::the_kish_ratio_holds_at_weights_too_small_to_square`, `ewcov/tests.rs::kish_size_holds_at_weights_too_small_to_square`, `tests/state_fixtures.rs::the_kish_sums_exponents_are_written` | 8c6936b3, 2f6e52e1 |
| REG-18 | confirmed; fixed by the user's decisions (2026-11-23: "Scales from every row", and rule 5's twenty-ninth exception, "Exception 29: start fresh"). Under `standardize`, `huber` and `quantile` read each feature's scale from its moments over every row the target is learned on with a positive weight (`Robust::spread`), refreshed at each row that moves the band (`spread_scale`), so a kept band system and the next solve read the same scales; a feature with no spread over those rows reads the band's, as before, and is dropped where neither has one. The review's case fits as the unstandardized fit and statsmodels' `QuantReg` within the band's half-width (0.0023). A state of 71 or earlier starts the moments at its first row after the load (`docs/records/STATE-CONTINUATION.md`, the twenty-ninth); the kept `huber*`/`quantile*` sets continue within 2e-5 relative (measured 1.56e-5), the kept banks to the bit. Goldens `quantile` and `quantile_skewed` regenerated (moves 2e-5 and 2e-3 relative; the skewed case's own check against the 0.7 quantile's shift holds). `robust_ref` takes the rule. Cost: 0.3-2.7% a row, `8(3k+2) + 8k` bytes a target | `crates/rheon-core/src/robust/quantile_refit_tests.rs::a_binary_feature_keeps_its_coefficient_under_standardize`, `tests/test_quantile_refit.py::test_a_binary_feature_keeps_its_coefficient_under_standardize`, `tests/state_fixtures.rs::the_scales_from_every_row_are_written` | 0be33dee, 377d2825 |

## Batch d: the window operators and their state (PLAN 578d)

On `review-578d`, cut from `vm/w13-integration` at 51671e2e. "Confirmed"
means the named test failed on 51671e2e for the review's reason. A test
marked `should_panic` holds a confirmed defect whose fix moves outputs
past 5 ulps, or is a choice of surface, and so waits on the user; it
fails once the fix lands and is then made a plain test. Each such fix is
built and kept as a patch under `.claude/coord/reviews/R1123/`. Rust test
paths are under `crates/rheon-polars/src/` unless they name `rheon-core`;
`far_tests` is `windows/far_tests.rs`. DRIFT-4 is batch e's.

| ID | verdict | test | commit |
|---|---|---|---|
| WIN-1 | **confirmed, settled by oracle and fixed in the second round** ("Batch d, continued" below): each interval's mass is `λ^(T − b) M(b − a)` (ahead `λ^(a − T) M(b − a)`) under a finite half-life. `t = [0, 54]`, `x = [42, null]`, a window of 200: the windowed mean and variance were null; `t = [0, 1, 54]`: the mean 199.99999999999997 against 150; ahead, null and 99.99999999999999 against 42 and 150 | `far_tests::a_backward_window_read_far_past_its_rows_keeps_their_proportions`, `::a_forward_window_read_far_ahead_of_its_rows_keeps_their_proportions`; `tests/test_windows.py::test_a_mean_read_far_from_its_values_is_the_definition`, `::test_windows_across_null_runs_are_the_definition` | a55c62e1 (tests), 27098311 |
| WIN-2 | (1), (2) confirmed, fixed: a quantile's sketch is anchored at the nearest row with a value, and a forward one whose anchor row left, its rest under `2^-64` of it, is made again from the oldest of them. No output of the core's streams or of `window_digest`'s 236 cases moves; the anchor moves a sketch's stored weights where the end row is null, so state bytes move and `window_digest`'s `BITS` with them (2800076a). (3) confirmed (`t = [1198, 2300, 2401]`, `x = [null, 42, null]` ahead: null against 42), **fixed in the second round**: a mean looking ahead is read where its sums are anchored, at the first row in the window with a value, and its open row's mass is seen from there | `far_tests::a_quantile_far_from_its_anchor_is_the_quantile_of_the_rows_moved_close`; `::a_forward_mean_closed_far_from_its_row_is_its_value`; `tests/test_windows.py::test_a_mean_a_thousand_half_lives_from_its_values_is_the_definition` | a55c62e1, 27098311 |
| HAWKES-7 | **confirmed (`G₀₀` 0.25 against 0.75 after a quiet `1e16`); held to the 60-digit fit in the second round, the built fix is not the whole of it, and it goes back to the user** ("Batch d, continued"). The fix moves the older block's times to the newer block's first row; a quiet stretch after that row is as before. Patch `review-578d-HAWKES-7-rebase.patch` | `rheon-core/src/hawkes/mutant_tests.rs::a_window_of_events_after_a_long_quiet_stretch_keeps_their_gaps`, `::a_quiet_stretch_inside_the_newer_block_keeps_the_later_gaps` (both `should_panic`) | a55c62e1 (test), the commit after 27098311 (the second test) |
| CWIN-1 | confirmed (`since` 0 for a factor of `2^-513`), fixed: a normal factor is taken with two exact scalings; a subnormal one still forgets | `rheon-core/src/window.rs::tests::a_normal_factor_below_the_scaling_is_kept` | a55c62e1 |
| WIN-8 | confirmed, both. (1) Unbounded, over a null run: the unbiased variance 32.000000000055934 at row 519 (the pairs' sum `e` subnormal), the mean 33.0 on `[42, 50]` at row 1076 and null at 1077. (2) Under `"left"`: the mean is WIN-1's cancellation, and with WIN-1's fix the variance is still null at 540 (`e` underflows). **A decision for the user**, with WIN-4, 5 and 10 below | `far_tests::a_mean_over_null_rows_holds_however_far_they_run`, `::a_left_closed_window_far_from_its_rows_keeps_their_proportions` (`should_panic`) | a55c62e1 (tests) |
| WIN-9 | confirmed (`t = [-1e308, 0, 1e308]`, `gap_cap=1e308`: every output null from the third row, and its state refused as damaged), fixed: the row whose distance from its windows' first row passes the largest double is refused by name (the run ends, as at a state cap); `check` holds the policy time finite, and the run's block paths end their block at such a row so that it reaches the refusal. From an origin at 0 the same clock runs on | `far_tests::a_clock_past_the_largest_double_from_its_origin_is_refused`; `windows_frame/load_tests.rs::a_clock_past_the_largest_double_is_refused_through_each_block_path`; `tests/test_windows.py::test_a_clock_past_the_largest_double_from_its_windows_start_is_refused` | a55c62e1, the block paths in the next commit |
| WIN-10 | confirmed (`x = [1e100, 1e100]` at a half-life of `1e210`, and `[10, 20]` at `1e308`: the mean `inf`). **A decision for the user**, with WIN-4, 5 and 8 | `far_tests::a_half_life_near_the_largest_double_weighs_its_rows_alike` (`should_panic`) | a55c62e1 (test) |
| CWIN-2 | confirmed (`corrchange`'s window statistic 0 against 0.566 at columns scaled by `1e80`), fixed at the six sites by `pair_corr`: the root of the product wherever the product is a normal double, so those bits do not move, and the two roots apart elsewhere. The existing test that held NaN for a pair whose product underflows now holds the pair's correlation | `rheon-core/src/corrchange_tests.rs::every_kind_is_free_of_its_columns_scale_at_the_ends_of_the_doubles`, `::a_windows_correlations_are_each_pairs_own` | a55c62e1 |
| WIN-4 | confirmed (a half-life of `1.5e308`: the mean null and the rate 0). **A decision for the user**, with WIN-5, 8 and 10 | the same `should_panic` test | a55c62e1 (test) |
| WIN-5 | confirmed (a half-life of `1e200`: the unbounded unbiased variance `inf`). **A decision for the user**, with WIN-4, 8 and 10 | the same `should_panic` test | a55c62e1 (test) |
| WIN-7 | (a), (b) confirmed, fixed: the shape check holds what a pop puts back (finite and at least 0 for a row with a value, NaN for one without) and refuses a weight with no anchor; the existing test of the front now reads it in its own order, newest first. (c) confirmed by reading, **deferred**: an emptied arm keeps its peak buffers until it next fills, as a `Vec` does; letting them go on every turnover allocates again at each one for a value range past four octaves, a speed and memory trade to measure first | `window_quantile.rs::tests::what_a_pop_puts_back_and_a_weights_anchor_are_in_the_shape` | a55c62e1 |
| WIN-11 | (1) confirmed (an increment list cut short loaded; the next row panicked), fixed: each group's lists are held to the call's width, one state per key and one for the null key. (2) `check` now holds each group's policy time finite; the queues' sums are not checked, since no panic from them is shown and the checksum refuses damage | `windows_frame/load_tests.rs::a_state_whose_increments_do_not_fit_its_call_is_refused`; the core's damage list in `windows.rs` | a55c62e1 |
| WIN-3 | **the rule kept, now written down** (`docs/WINDOWS.md`, "Which rows a window holds"): looking ahead a row holds its value to the next row, so a window one grid step long under `"right"` holds only its far edge's row and is null; the mean of the rows themselves is a ratio of two sums. Part 2 was the review's own false positive | `tests/test_windows.py::test_a_mean_ahead_one_step_long_on_a_grid_holds_only_its_far_edge` | a55c62e1 |
| WIN-6 | **not reproduced**: under `save_state` the backward windows waiting on the last stamp are read as they stand at the input's end (`w.settle()`), so a saved run returns the rows a run without a state returns, to the bit, and `WINDOWS.md` and the docstring say so | `tests/test_windows.py::test_a_saved_run_returns_the_last_stamps_rows` | a55c62e1 |
| CWIN-3 | **confirmed (at a level of `1e8`, a window holding 1% of the weight reads 1.5e-7 off the exact value; the centred form 1e-14); settled by exact rationals in the second round, and held on the `v71` set's frozen frame** ("Batch d, continued"). Patch `review-578d-CWIN-3-centred-v2.patch` | `rheon-core/src/window.rs::tests::a_truncated_mean_keeps_the_spreads_digits_at_a_level`, `::a_truncated_mean_of_equal_means_is_that_mean` (both `should_panic`) | a55c62e1 (test), the commit after 27098311 (the cases and the second test) |
| DRIFT-4 | batch e's (5cac6698) | -- | -- |

## Batch d, continued (PLAN 578d, second round): the held fixes, settled by oracle

On `review-578d`, from c8ddeb66 (27098311 and the commit after it). The user's rule for a fix that moves
outputs past 5 ulps: establish the right answer first, by a third-party
oracle, an exact or high-precision recursion, a simulation or a limit
case; a fix the evidence settles proceeds with that evidence in its tests
and a `Moves:` entry, and only what stays open goes back. Each oracle here
is written from the definition in `decimal` or `fractions`, on the
operands' own doubles, so it has no rounding to share with the code.

| ID | the oracle | before | after | outcome |
|---|---|---|---|---|
| WIN-1 | The two forms of `mass_between` against 120-digit decimal, 32,640 draws: half-lives `1e-3` to `1e9`, intervals `1e-12` to `1e4` half-lives long, 0 to 1,000 half-lives from the row. Then `with_windows` against the definition at 80 digits (`tests/reference_windows.py`): two values read `T` half-lives on, each `closed`, both directions; and streams with runs of 40 null rows, eight operators, half-lives 0.5, 4, 50 and none, windows of 7, 60 and 400 | The difference: 4 ulps at one half-life, 2,400 at 10, 2.5e6 at 20, 1.9e9 at 30, 2.3e12 at 40, all of the mass from 54. The mean of 100 and 200: 1.2e-14 of 200 off at 10 half-lives, 8.7e-9 at 30, 1.5e-5 at 40, 4.0e-4 at 45, 0.25 from 54. Null-run streams: 0.25 (mean), 0.098 (variance), 0.31 (deviation) at 120 half-lives, 4e-13 at 15, 5e-6 at 100; 104 rows null on one side alone | The product: 2 ulps to one half-life, 5 at 10, 20 at 54, 90 at 200, 430 at 1,000, and in no draw more than 4 ulps further off than the difference. The mean of 100 and 200: at most 5.4 ulps of the largest value to 663 half-lives, 17.4 at 1,000. Null-run streams: at most 9.5e-15; one cell of 96 further off than before (9.0e-16 to 1.8e-15); 8 rows null on one side alone, all a variance at 800 half-lives (WIN-8's) | **applied**. 179 of `window_digest`'s 236 cases move (every case with a window but five refused at a step back; none without one), its null masks do not; the current `with_windows`, `with_windows_int`, `bank_formula` and `checkpoint` fixtures are regenerated, and every kept set goes on and converts as before. The release probe's fields do not move (its formula target has no decay) |
| WIN-2 (3) | The same 80-digit definition, a row 1,000, 1,102 and 3,000 half-lives before its window's two values, each `closed` | null (the sums discounted to the row are 0 from 1,074 half-lives) | 0.13 ulps of the largest value | **applied** with WIN-1: a ratio is read where its sums are anchored. With no decay the mass is taken as it was |
| HAWKES-7 | The windowed least-squares fit from its definition at 60 digits (`G = ∫ φ φᵀ`, `b = Σ m φ(t−)` over the window's events alone, solved by elimination), against the bank's `coef` on tick's simulated streams: one node and one decay, two nodes and two decays; `window_events` 3 to 150, clock windows of 3 to 80; quiet stretches of `1e3` to `1e18` on a number clock and on an integer clock, whose stamps stay exact. The oracle agrees with the build to 1e-15 at the median where the Gram is well conditioned (tick's own minimiser is the suite's second opinion there) | Ordinary streams, the rows where the 60-digit fit exists: median 3.7e-16 to 4.5e-14, worst 2.6e-6 (`window_events=3`, two nodes), 1.5e-7 (5), 1.0e-10 (8). After a quiet stretch on an integer clock, `window_events=8`: 1.6e-2 of the largest coefficient off at `1e16`, 2.5 times it at `1e18` (0.81 for two nodes) | With the built fix: the same medians and the same worst errors on ordinary streams, the fits moved by at most 1.2e-8 (3), 4.6e-10 (5), 1.7e-12 (8) and neither build the nearer (94 rows against 97, 173 against 165, 172 against 157). **After a quiet stretch: 1.6e-2 and 2.5, as before** | **not applied; open** (below) |
| CWIN-3 | `truncated_scalar`'s two forms against exact rationals (`fractions`), 180,000 draws in 90 cells: levels 0 to `1e12` and `-1e8`, spreads `1e-6` to `1e3`, windows holding a half to a millionth of the weight. Both forms are plain IEEE arithmetic, so Python's doubles are Rust's | The uncentred form: its error grows as `ε (w_now / w) · level`. At a level of `1e8` and a spread of 1: 3.2e-8 of the spread with half the weight in the window, 2.1e-6 with 1%, 1.9e-2 with a millionth; at `1e12`, 2.1e2 spreads with a millionth; `w_now·m` overflows at `m = 1e305` and reads NaN | The centred form: half an ulp of the level in those cells (7.4e-9 of the spread at `1e8`). Its largest and median error are at or under the uncentred form's in each of the 90 cells; within `3 ε r abs(m − m_old)` and one ulp of the value on every draw; exact where the two means are equal or nothing is subtracted | **settled, not applied; held on rule 5** (below) |

### WIN-1 and WIN-2 (3): what was built

- `windows/gap.rs`: `mass_between`, `mass_between_in` and
  `mass_between_close` take the product under a finite half-life and the
  difference with none; `mass_between_fill` and `mass_held_to` already
  read an interval ending at the row. `windows.rs`'s close reads a mean's
  sums unscaled and its open row's mass from their anchor.
- Two memo slots, neither written to a state: `disc_cut`, the cut's
  discount, which shared `discount` with the fold across a queue's anchors
  and, with several operators in a kernel, missed on every ask of either
  (a mean, a deviation, a sum and a variance of one kernel ran 1.076 times
  the build before with one slot, 1.033 with two); and `disc_mean`, the
  discount of a mean's open row to its anchor.
- The Rust brute force's mass (`mass_bf`) was the same difference, and is
  now the integral itself by five-point Gauss-Legendre quadrature on panels
  of at most half a half-life, every term above 0. The Python definition
  at 80 digits is new (`tests/reference_windows.py`).
- Speed, arm64 only (release, load 5, an ABBA of 12 runs a build with the
  identical binary's beside it): a windowed mean on an irregular clock
  1.022 (identical binary 0.993), on a regular one 1.002 (0.998), with 30%
  nulls 1.027 (0.996); a deviation 1.008 (1.004); four operators of a
  kernel 1.033 (1.016); a mean ahead 1.020 (1.001) and 1.015 (1.014); no
  window 0.989 (0.979). Instructions retired: +54 a row for the mean on an
  irregular clock (2,471 to 2,525), +6 on a regular one, +51 for the
  deviation, −16 for the mean ahead. `instructions.window_ewm_mean_w10k`
  is raised from 2,172,254 to 2,226,254 on that count, **arm64's, not
  CI's**: its first count on the change is the one to hold. Not timed on
  x86-64, which the workflow asks of a change to a per-row path before it
  merges.

### What waits on the user

- **HAWKES-7: the built fix is half of it, and its fixture breaks a kept
  set.** Three things the 60-digit fit showed.
  1. *The 0.36 move is between two answers that do not exist.* The largest
     moves (0.27 at `window_events=5`, 0.046 at 3, 0.055 under a clock
     window of 10 after a quiet stretch) are all on rows whose Gram is
     singular at 60 digits: a node whose only event in the window is the
     reading row's own, so its excitation is 0 over the whole observation
     and its kernels' coefficients are not identified. Both builds report
     a fit there (rounding dust left by the window's subtractions, which
     the scaling to a unit diagonal hides from the pivot test), and the fix
     moves one dust to another. Where the 60-digit fit exists the two
     builds are equally near it. So the move is neither toward the truth
     nor away from it, and the fix is harmless to ordinary fits.
  2. *The fix covers one of the defect's two paths.* It moves the block's
     start to the newer block's first row. A quiet stretch after that row
     still puts `1e16 + dt` in the later rows' times, and their gaps still
     round away when the block turns over:
     `a_quiet_stretch_inside_the_newer_block_keeps_the_later_gaps` fails
     with the fix and without it, and through the bank the errors above
     (1.6e-2 at `1e16`, 2.5 at `1e18`) are the same in both builds. The
     whole fix keeps each newer row's gap to the row before it, and the
     clock since the newest row beside the clock since the block's start:
     one more double in the window's state (8 bytes a group, none a row),
     and a row's `time` read as a gap while its block is the newer.
  3. *Rule 5.* With the fix, `hawkes_window` regenerates and
     `v71/hawkes_window` no longer converts to its bytes (the kept state
     loads and goes on to the bit, but in the old block start, which is as
     valid a state as the new one and another set of bytes). The whole fix
     changes the same field's meaning, so either a load rebases a kept
     state (exact where the stamps' sums are, as the fixture's are) or the
     kept set is compared as `bank`'s windowed kinds are, an exception of
     the user's.
  Options: (a) the whole fix, with the new field and a rebase at load,
  moving ordinary windowed fits once, by the rounding measured above
  (1.2e-8 at three events, 1.7e-12 at eight, 1.1e-13 at twelve); (b)
  the built fix now and the rest later, moving them twice; (c) neither,
  and a quiet stretch so long that a gap after it is under its rounding
  is documented as losing that gap. The recommendation is (a), as a
  task of its own.
  *A finding of its own, not the review's:* `hawkes` reports a fit for a
  window in which a node has no excitation (above). Over the simulated
  streams: 18 of 158 such rows under a clock window of 3 with one node, 3
  of 18 under 10, 2 of 264 at `window_events=5` and 1 to 3 of 797 at 3 with
  two nodes; the tiny case written by hand (the earlier events dropped
  exactly) is withheld as it should be, so it is the subtraction's dust
  that passes. The guard is the condition itself: a diagonal of `G` that
  is rounding against its own terms (`β b + β² Q` against `R²`) is 0, and
  the fit is withheld as not identified. No test is committed for it;
  `hawkes_all.py`'s method (every row against the 60-digit Gram) is the
  one to write it from.
- **CWIN-3: settled, and held on `v71/bank`'s frozen frame.** The centred
  form is the more accurate everywhere it was measured and exact where the
  uncentred form rounds. It moves `spread_window.sigma_y` by one ulp on
  the first row after a load (0.06739974994500428 to …426) in the current
  `bank` fixture, which regenerates, and in `v71/bank`'s frame, which a
  kept set holds to the bit on that row (hard rule 5: "its first output
  row is the old build's to the bit"). No core fixture moves. The kept
  frame is not edited here; landing the fix takes the user's word that
  `v71/bank`'s windowed spread is compared as the rows after its first are
  (the golden tolerance), an exception of the user's to count.
- **WIN-4, WIN-5, WIN-8, WIN-10: masses and their squares at the ends of
  the doubles, measured.** Nothing is built. Today a held row is 40 bytes
  and 64 an operator (112 an operator in a kernel with a variance); a
  kernel with no window keeps 24 or 48 bytes an operator a group. The
  options, each with its cost on arm64 (a merge is what a row's push does;
  the unbounded mean's row is 18.5 ns, a windowed mean's 86.5, a windowed
  deviation's 105.5) and what it moves (emulated in doubles against the
  exact recursion at 60 digits, 12,000 rows at a half-life of 5 with 15%
  nulls, and 6,000 at a half-life of 2,000):

  | option | fixes | bytes | time | outputs moved |
  |---|---|---|---|---|
  | (a) refuse a half-life from `2^511` | WIN-4, WIN-10, and WIN-5 where the half-life is the cause | 0 | 0 | none; a call legal today is refused (the suite's own mass test at `1e300`) |
  | (b) a power of two a kernel as the masses' unit, 1 below `2^511` | the same, with no refusal: all nine half-lives of `far_tests`' case, `1e150` to the largest double | 0 in the state (a constant of the kernel) | none measurable: a mass 5.15 ns with the unit multiplied in against 5.12 to 5.15 | none below `2^511`, state bytes included. Alone, at the largest half-lives, a variance over steps under `1e-9` is null (the pairs' products underflow) and the mass's argument `t ln 2 / h` is subnormal from steps of `1e-12` (a mean 1e-4 off): the mass needs its length where that argument is under `2^-53` |
  | (c) for the mean: the mean itself in place of `m x` | WIN-10 to `1.2e308`, not WIN-4 above it (`h / ln 2` overflows either way) | 0 | **+3.7 ns a merge** (4.3 to 8.0, a division; control ±0.02): a fifth of the unbounded mean's row, 4% to 9% of a windowed one's | 80% of means, a median of 1 ulp and at most 8 where the mean is not near 0 (a median of 34 and at most 102 at a half-life of 2,000), each nearer the exact recursion (0.3 to 0.6 ulps at the median against 0.8) |
  | (c) for the variance: `r = 2e / w²` in place of the pairs' sum, and the variance `M2 / w` in place of `M2` | WIN-5 whatever the cause (no square is kept), and the variance's part of WIN-8's drift while the mass is above 0 | 0 (the same six doubles, two with another meaning: a windows state version and a loader) | +0.3 to +0.5 ns a merge for `r` alone, +0.1 to +0.3 with the variance too (8.7 to 9.1; control −0.03 to +0.14): under half a percent of a windowed deviation's row | 80% of unbiased variances, a median of 1 to 2 ulps and at most 15 (21 and 111 at a half-life of 2,000), as near the exact recursion or nearer (0.71 to 0.92 ulps at the median against 0.88 to 0.90). Past 5 ulps at the tail, so the user's |
  | (d) read a mean and a variance where their sums are anchored, and let a row that adds nothing move no anchor (WIN-2 (3)'s reading, looking back) | WIN-8, both parts, for a kernel whose operators read one column; with an anchor an operator, for any | 0, or 8 an operator a row for the anchors | not measured in the engine; it removes the read's discount | half of means, a median of 0 to 1 ulp and at most 7 where the mean is not near 0, as near the exact recursion |

  The recommendation: **(b)**, with the length for a subnormal argument,
  for WIN-4 and WIN-10: it is free and moves nothing. **(c) for the
  variance only**, both `r` and the variance, for WIN-5: under half a
  percent, no bytes, and with (b) every half-life of the test passes
  (emulated: today fails from `1e154`, (b) alone passes all nine, (c)
  alone fails above `1.2e308`). **Not (c) for the mean**: a division a
  merge for an overflow (b) already ends. **(d) for WIN-8, as a batch of
  its own**, shared anchors first: it is the only one of the four that
  touches the drift, since a unit and a ratio both still discount the mass
  to 0 over 1,074 half-lives of null rows. (c) and (d) move outputs past 5
  ulps at their tails and (c) changes what two of the state's doubles
  mean, so both are the user's; (b) needs no decision.
