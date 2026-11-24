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

## Batch f: the state-space models: `hawkes`, `hmm`, `kalman`, `bocpd` (PLAN 578f)

On `review-578f`, cut from `main` at 13f5ffe4. "Confirmed" means the
named test failed on 13f5ffe4 for the review's reason. A test marked
`should_panic` holds a confirmed defect whose fix is the user's call; it
fails once the fix lands, and is then turned into a plain test. Test
paths are under `crates/rheon-core/src/` unless they start with
`crates/rheon-polars/` or `tests/`. HAWKES-1 is batch e's and HAWKES-7
batch d's.

| ID | verdict | test | commit |
|---|---|---|---|
| HAWKES-2 | confirmed (marks `1e8` and 1 dropped from the older block left `Q = -1`; a window emptied by the older block's rows kept its clock, and a quiet stretch of `1e16` put the next rows' times at `1e16` each, their gaps of 0.25 lost), fixed: an empty window's `R`, `b`, `Q` and `B` are 0 and its clock is 0, and the clock runs only while the window holds a row. The fit after the window empties is the definition's (`oracle::hawkes_system`) to `1e-12` | `hawkes/review_tests.rs::a_clock_window_that_empties_keeps_nothing_of_the_rows_that_left`, `::an_empty_clock_window_counts_no_clock` | 9ac7dbc0 |
| HAWKES-3 | confirmed, all three, fixed: `max_state_bytes` is held to the 64 MiB ceiling in the core (`clock_span::check_max_state_bytes`) and in the spec; the core checks the default cap of 256 KB as a given one; a restored window of clock units past its cap is refused as damaged. `hawkes` is not in 2.0.0, so no released state meets the ceiling | `hawkes/review_tests.rs::the_state_cap_is_held_to_the_ceiling_and_the_default_is_checked`, `::a_restored_clock_window_past_its_cap_is_refused`; `crates/rheon-polars/tests/spec_refusals.rs::a_hawkes_window_past_its_state_cap_is_refused`; `tests/test_hawkes.py::test_the_window_is_held_to_the_groups_state_cap` | 9ac7dbc0 |
| HAWKES-4 | confirmed (`half_life = 1e-310` built, and `G₀₀` was NaN from row 0), fixed by refusal: a rate `ln 2 / half_life` past the doubles, and decays two of which sum or multiply past them (`[1e160, 1e150]`); `1e-300` and `[1e150, 1e140]` are taken | `hawkes/review_tests.rs::a_rate_or_a_kernel_past_the_doubles_is_refused` | 9ac7dbc0 |
| HAWKES-5 | confirmed (on a clock in tenths a row of weight 0 read the window at the last stamp moved on by its step; at a span of 0.9 the bank gave null where the step reads 50.94), fixed: `Hawkes::predict_at` reads the window at the row's own stamp, and the bank routes a scored row of a `hawkes` under `window_size` to it, as `ewm_moments`' | `hawkes/review_tests.rs::a_scored_row_reads_the_window_at_its_own_stamp`; `crates/rheon-polars/tests/hawkes_clock_window.rs::a_zero_weight_row_reads_the_window_at_its_own_stamp` | 9ac7dbc0 |
| HAWKES-6 | confirmed, fixed: 64 and 200 bytes, 6,551 and 3,273 rows (each recomputed from `Hawkes::state_bytes`) | -- (docs/MODELS.md) | 9ac7dbc0 |
| SPEC-2 | confirmed in Rust (a Rust or TOML spec took it; Python's builder refused it already, by its check of every list of numbers), fixed: `HmmCfg::validate` refuses a `tvtp_coef` value that is not finite | `hmm/review_tests.rs::a_tvtp_coefficient_that_is_not_finite_is_refused`; `tests/test_hmm.py::test_a_bad_spec_is_refused_by_name` (the builder's refusal) | 4e422563 |
| HMM-1 | confirmed (`ŝ` NaN after a row `1e80` from a state given `1e-150` at a posterior of 0; infinite at a posterior of 0.5), fixed: a state the row is not in adds nothing, a row whose deviations leave the doubles ages `V` as a row of no weight does, and the seeding pools no state of no weight. The share forms the review gives for `take_up_factors` and `shrink` were not taken: `P₀ψ` overflows only at `prior_rows` past `1e108`, since `ψ` is at most `INPUT_BOUND²`, and the share form moves every prior by rounding | `hmm/review_tests.rs::a_deviation_past_the_doubles_leaves_the_prior_loadable` | 4e422563 |
| HMM-2 | confirmed (`ψ` fell back to `precision_prior`, `1e-3`, a variance, under a shape), fixed: the fallback is 1 under a shape, the covariances as given; unserialized, so no state moves | `hmm/review_tests.rs::with_given_states_no_scale_falls_back_to_the_given_covariances` | 4e422563 |
| HMM-3 | confirmed (`diag(1e-20, 1e20)` in both states read `7.9e-11` on the first row, where the same one ulp apart read `1e-20`), fixed in the follow-ups by the coordinator's option (a): `Shape::of` keeps no shape only for the same multiple of the identity in every state, and `Saved` writes `shaped` where a shape is kept (`serde(default)`, last); a state without it loads with the shape the build before kept, so one it saved for the same diagonal reads on as saved, to the bit | `hmm/review_tests.rs::the_same_anisotropic_diagonal_in_every_state_reads_as_given_on_the_first_row`, `::a_state_saved_without_the_shape_reads_on_as_it_was_saved`, `::a_shaped_state_under_a_configuration_with_no_shape_is_refused`; `tests/state_fixtures.rs::the_priors_shape_flag_is_written_where_a_shape_is_kept` | 4e422563, f97b20c7 |
| HMM-4 | (2) confirmed (a failed move on a row of weight 0 counted two failures), fixed: counted only where the restart is kept. (1) the rule kept: a row that starts a state again keeps every move (the docstring says so), and the same gap on the next learned row would age that state out too | `hmm/review_tests.rs::a_failed_move_on_a_row_of_weight_0_is_not_counted` | 4e422563 |
| HMM-5 | (1) confirmed in the core (the spec refused it), fixed: `warm_span` beside given states is refused. (2) confirmed, fixed: a place that does not match `warm_span` is damaged. (3) confirmed, fixed: the seeding row's bytes are read before the seeding empties the buffer. (4) confirmed for the compact encoding, **left for the user** with BOCPD-4 | `hmm/review_tests.rs::a_warm_span_beside_given_states_is_refused`, `::a_restored_place_matches_the_warm_span`, `::the_seeding_row_raises_the_half_cap_warning`, `::a_compact_state_with_a_skipped_middle_field_round_trips` (`should_panic`) | 4e422563 |
| KAL-1 | not taken: core only (the bank never hands a model a row of weight 0), and past about 1,075 half-lives every weight underflows to 0 under any rule; `kalman/tests/gap_moments.rs::a_gap_split_by_zero_weight_rows_is_one_gap` states the boundary | -- | -- |
| KAL-2 | confirmed (at `2^-530` the factor's update was off in the fifth digit; at `1.3e154` the row is zeroed), fixed: the rotation scales the pair by its larger where `l² + x²` is not a normal double, to the bit elsewhere | `kalman/tests/review.rs::the_factors_rank_one_update_holds_at_every_scale` | 6dd64679 |
| KAL-3 | confirmed (`R/w = 1e-250/1e100 = 0`: the factor's `P` after the row was `1.3e-250` where the covariance form's is `2.0e-251`), fixed: Carlson's update at `α = 0` is its limit, a column with `f_j = 0` kept and the first other zeroed, as the covariance form's exact observation. The review's gate (`r > 0`) was not taken: it would refuse a row the covariance form learns | `kalman/tests/review.rs::an_exact_observation_is_the_same_in_both_forms` | 6dd64679 |
| KAL-4 | confirmed (back from the square root, `P`'s diagonal held its roots), fixed: squared again, within an ulp of the start | `kalman/tests/review.rs::a_form_given_and_given_back_leaves_the_covariance_as_it_was` | 6dd64679 |
| KAL-5 | confirmed on a state set by hand (the review's note: unreachable from the bank), fixed: the readout's gap noise stays off an open intercept | `kalman/tests/review.rs::a_diffuse_intercepts_readout_takes_no_noise_of_the_gap` | 6dd64679 |
| KAL-6 | (2) confirmed, fixed in the follow-ups (below, "Follow-ups"; rule 5's thirty-fifth exception): rows that learn the features and observe no target opened the moments' floor on 16 of 20 such rows at a half-life of 5. (1) the documented rule: on a grid coarser than the half-life every row is "more than one half-life after a target was last learned" | `kalman/tests/review.rs::rows_with_no_target_leave_the_moments_floor_shut` (`should_panic`) | 6dd64679 |
| KAL-7 | confirmed on a state set by hand (`c = 2000`, `P = v vᵀ` with `v ⊥ (1, c)`: `-9.3e-10`; a stream that shifted a feature by `1e5` of its scale did not reproduce it), fixed: `current_diagonal`'s intercept and `prior_quad` clamp a rounding below 0, as `coef_variance_of` does | `kalman/tests/review.rs::the_intercepts_variance_in_the_current_coordinates_is_never_below_0` | 6dd64679 |
| BOCPD-1 | confirmed, both (a row of no share left the joint NaN; under `robust` the empty run's weight of 0 made every row after the first a failure), fixed: a weight of 0 passes its run on as it is, and a second pass whose joint is not a number refuses the row. The review's mode check is a `debug_assert`: once BOCPD-5's variance is a finite number above 0 the mode is too | `bocpd/review_tests.rs::a_row_of_no_share_leaves_the_joint_a_number`, `::a_robust_run_of_weight_0_passes_on_as_it_is` | 6d073513, 2a70fcc7 |
| BOCPD-2 | confirmed (0.404 nats off on the review's values), fixed: a block's partial products are read where its least value is below `2^-127`, the only way one falls below the normal numbers (an overflowing partial stays infinite and the last one's range refuses it). The review's `[1e-35, 1e35]` was not taken: it moves every block outside it | `bocpd/review_tests.rs::a_sum_of_logs_holds_through_a_product_that_passes_the_subnormals` | 6d073513, 2a70fcc7 |
| BOCPD-3 | confirmed (four failures at `d/τ = ∞`), fixed: a certain break, `ln H = 0`, `ln(1 − H) = −∞`; to the bit the reads of a step of `d/τ = 1e290` | `bocpd/review_tests.rs::a_clock_step_past_the_doubles_is_a_certain_break` | 6d073513 |
| BOCPD-4 | confirmed for the compact encoding, which no file of the library is written in; **left for the user** (below) | `bocpd/review_tests.rs::a_compact_state_with_a_skipped_middle_field_round_trips` (`should_panic`) | 6d073513 |
| BOCPD-5 | confirmed (12 failures in 12 rows at `ψ₀ = 1e308`; NaN from row 16 at `κ₀ = 1e307`), fixed in the share forms where the first forms leave the doubles, to the bit elsewhere: the variance, its log, the prior's pull, and the full scatter's scale. The oracle is the model's own scale equivariance: at `ψ₀ = 1e308` the run posterior is the one at `1e308 · 2^-1000`, to the rounding of a log, in every emission | `bocpd/review_tests.rs::a_large_prior_scale_reads_as_the_same_prior_at_a_smaller_scale`, `::a_heavy_prior_on_data_at_its_mean_reads_a_number` | 6d073513, 2a70fcc7 |

**What moved.** `hawkes` alone: a window whose rows have all left now
holds exactly nothing, a window's clock starts at its first event, and
a row of weight 0 reads a window of clock units at its own stamp
(`Moves: hawkes.intensity_*`). Every other fix leaves the bits of every
row within the doubles as they were: each takes its new form only where
the old one left them (a sum of squares, a variance, a product, a pull
not a normal double; a weight of exactly 0; `α = 0`). Every state
fixture loads, continues to the bit and re-saves byte for byte, v71 and
older included; no fixture was regenerated.

**Per-row cost** (arm64, release, the case's binary built from 13f5ffe4
and from this branch, minimum of 6 to 12 interleaved runs of 4,000 rows,
load 6 to 13, both code layouts): `bocpd` with 4 features, `max_run`
200, prune `1e-6`: diag 1.02 to 1.03 (the identical binary's spread
1.02 to 1.05), full 1.01 to 1.03 (1.04), robust 1.03 to 1.04 (1.02 to
1.03). A first build that read every block's partial products cost
diag 1.13; the guards now ride on the `ψₙ` the read already forms.
`hmm`, `hawkes` and `kalman` (both forms): level, within the control's
spread. Not measured on x86.

### Left for the user (batch f)

- **HMM-3** (done in the follow-ups, option (a); below): given
  covariances that are the same anisotropic diagonal in
  every state take no shape, so `ψ` is a variance and its floor (`2^-100`
  of the largest) lifts the small one: `diag(1e-20, 1e20)` reads
  `7.9e-11` on the first row, where a state one ulp apart reads `1e-20`.
  `Saved` does not say which form `ŝ` is in (a variance without a shape,
  a multiple of each given one with one), so changing `Shape::of` alone
  would read a saved variance as a multiple: a schema-70-to-72 state of
  that form would load with its variances squared. Options: (a) `Shape::of`
  returns `None` only for the same multiple of the identity in every
  state, with a `shaped` flag written in `Saved` where a shape is kept
  (`serde(default)`: a state without it loads as today's form and keeps
  it); the current fixtures of given states regenerate, and outputs move
  from the first row for the configurations this reaches; (b) keep the
  form and say in `covs`' docstring that features whose given variances
  differ by more than `2^100` are floored. **Recommendation: (a)**: the
  shape is what `covs` says, and the flag keeps every saved state.
- **KAL-6 (2)** (option (a) chosen; landed under rule 5's thirty-fifth
  exception, below): under
  `standardize`, rows that learn the features but
  observe no target leave the moments without a gap, yet the floor on
  their history reads the clock since a target was observed, so it opens
  on such rows from a half-life on (16 of 20 at a half-life of 5) and
  holds the moments' memory to the row's own weight. The fix keeps the
  moments' own gap, a new `f64` in the state (schema 72 unreleased; a
  state of 71 or older loads it as the clock since a target was observed,
  today's reading), and moves the outputs of streams with missing targets
  under `standardize` from the first such run longer than a half-life.
  Options: (a) the moments' own gap; (b) keep the rule and say in the
  module doc that a row with no target does not close the gap.
  **Recommendation: (a).** Claim (1), on a grid coarser than the
  half-life, is the documented rule; keeping it is recommended.
- **BOCPD-4 and HMM-5 (4)**: a struct whose skipped field is not its last
  does not round-trip the compact (positional) msgpack encoding: `bocpd`
  with no failures and a warm span, `hmm` under `tied` with a warm span.
  Every file the library writes is named msgpack, which round-trips, and
  the compact form is used only to count bytes (`rheon-bench`). Options:
  (a) keep, and say in the two structs' comments that their states are
  named msgpack only; (b) write every field always, which adds a few
  bytes to every such state and regenerates the current fixtures.
  **Recommendation: (a).**

### Follow-ups (batch f), on `fix-578f-followups`

The coordinator chose option (a) for both HMM-3 and KAL-6 (2).

- **HMM-3: done** (f97b20c7). The case the definition settles: on the
  first row a state reads the covariance it was given, `Ψ_s = G_s`, so
  `diag(1e-20, 1e20)` reads `1e-20` and `1e20`, as the one-ulp-apart
  covariances did. What moved: only given covariances that are the same
  diagonal in every state with variances that differ, under `"full"` or
  `"diag"` in a model that learns; there by rounding where the floor did
  not bind (2,000 rows, every output of `hmm`: at most 2.9e-14 relative
  at `diag(0.25, 4)`, 5.7e-14 at `diag(1e-6, 1e6)`), and to the given
  variances where it did. No test, golden or release workload holds such
  covariances (the second-opinion oracle's `given_shape` takes the rule).
  New fixture `hmm_given` (schema 72, no bump) freezes `shaped: true`; no
  other fixture's bytes moved, v71 and older included. A schema-69 state
  takes the shape the build before kept. Per-row cost (arm64, release,
  minimum of 6 interleaved runs, identical-binary control, load 6 to 15):
  the affected configuration at 4 features pays the shaped path, diag
  1.01 and full 1.02 against a control of 1.00 and 1.01; every other
  configuration's path is unchanged.
- **KAL-6 (2): landed** (fec71253, merged), after a stop on a fixture,
  under the user's word (2026-11-24: "No concern for continuing
  differently here"), rule 5's thirty-fifth exception: `MOMENTS_GAP`
  holds `kalman_noise_units`' kept sets to loading, continuing to the bit
  and saving bytes that load back the same. The fix: the moments' own
  gap, the clock since the last row of weight above 0, kept as `moments_gap`, and
  written, last and after the covariance form, only where it is not the
  clock since a target was observed (so the positional encoding never
  reads one in the other's place); a state without it reads that clock,
  today's reading. The review's test is a plain test held to an `EwDiag`
  fed the same rows, to the bit; on the old reading it fails at row 29,
  where a row weighing more than the history's decayed weight is read
  past the floor. Outputs move only where the floor binds: a row after a
  run of missing targets longer than a half-life that weighs more than
  the history's decayed weight (with even weights on a regular clock,
  none past a stream's first rows). The stop: `kalman_noise_units` is
  saved after three rows with no target, so this build writes a gap of 0
  where its clock since a target is 2, and its v64 to v71 files, which
  load with today's reading (2), continue to the bit but convert to other
  bytes than the regenerated current fixture
  (`every_previous_fixture_loads_goes_on_and_converts_to_the_current_bytes`
  failed on them alone). No loader can know the gap, which an old state
  does not hold; reading it as 0 (true of every state a bank wrote, since
  the bank hands no row of weight 0) breaks the other cases, saved after
  rows of weight 0. Per-row cost (as above): `kalman`, 4 features,
  `standardize`, a target missing one row in five, 0.97 against a
  control of 1.01, no measurable change.
- **CWIN-3 (batch d's): landed here** under the same word, rule 5's
  thirty-fourth exception: the centred truncated mean
  (`review-578d-CWIN-3-centred-v2.patch`, adapted: its two tests, held by
  `should_panic` on `review-578d`, are carried here as plain tests). No
  core fixture moved; the current `bank` fixture regenerated (its frames'
  last bits), and `v71/bank` is held at 1e-15 of `1 + |frozen|` on
  `spread_window.sigma_y` alone (`moved_by_the_centred_mean`).

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

## Batch b: `glm` and `expectile` (PLAN 578b)

On `review-578b`, cut from `vm/w13-integration` at 51671e2e. "Confirmed"
means the named test failed on 51671e2e for the review's reason. Test
paths are under `crates/rheon-core/src/` unless they start with `tests/`;
`glm/review_tests.rs` and `expectile/review_tests.rs` hold one test or
more a finding, named by its ID. A test marked `should_panic` holds a
confirmed defect whose fix is the user's to allow, as in batch e.

| ID | verdict | test | commit |
|---|---|---|---|
| EXP-1 | confirmed (cases 1 and 2: an emptied clock window, and a row window with fewer targets than coefficients, read the fit of the rows that left at `τ = 0.9`, where `τ = 1/2` read none), fixed: a first pass that fails from a warm start starts cold, from least squares, or finds no fit. A target that loses its fit counts in `solve_failures` at every level, as the field says ("refits that ... found no fit") and as `τ ≠ 1/2` counted before, so every fixture keeps its bytes. Case 3 (the first rows' refit failing its first pass) was EXP-2's constant feature, fixed there | `expectile/review_tests.rs::exp_1_an_emptied_clock_window_reads_no_fit_at_every_level`, `::exp_1_a_window_with_too_few_targets_reads_no_fit_at_every_level`, `::exp_2_a_constant_feature_is_fitted_by_the_window_and_the_refit`; `tests/test_expectile.py::test_a_clock_window_withholds_only_where_its_weight_says_so` | e6a1eee8 |
| GLM-1 | (1) confirmed (the refit started at `ln y` past the log link's clip, where the objective is flat, and read the clipped mean's residual as the score: the intercept went to 32.33 at `y = 1e14` and to 1e6 at `1e25`, reported converged), fixed: a row past the clip has no score or curvature in the refit, and the start is held to the clip, so the fit stops at the clip's edge, the nearest mean the link gives. (2) confirmed (the Gamma's inverse link, every halving leaving a row's `η` below 0: the start returned as converged, `9.5e14` against an objective of 1.0 in reach), fixed: a line search that finds no fall is convergence only where the full step's predicted fall is below what `F` shows; otherwise the fit stands, counted unconverged. (3) not reproduced: the trapezoid rule decides only steps whose predicted fall is below `1e-10` of one plus the objective's size, and the Tweedie at `p = 3` (not convex in `η`) and the Gamma's log link land on the oracle's batch fit (Newton by faer's LU) to 1e-10 | `glm/review_tests.rs::glm_1_a_target_past_the_clip_never_raises_the_refits_objective`, `::glm_1_a_line_search_that_finds_no_fall_is_not_converged`, `::glm_1_the_trapezoid_rule_holds_on_non_quadratic_deviances` | e6a1eee8 |
| EXP-2 | (1) confirmed (a feature constant at -1500: a window read NaN on every row, the first rows' refit kept the one pass's fit at `τ = 0.9`), fixed: the exact refit's WLS falls back, where its raw pivot fails, to the solve about the rows' own weighted means, formed from the rows (`Fit::centred_wls`). (2) the window and refit: confirmed (a feature at `1e6 + 1e-3 N(0, 1)`: NaN on every row), fixed by the same fallback, exact on the rows; the one pass: confirmed (its solve fails on 118 of 200 rows, and the fit before stands, up to 4.9 from the ridged least-squares fit against noise of spread 1.5), a decision for the user (below). (3) confirmed (a feature always 0 beside one of spread 1e7: the one pass froze at an early row's fit, the intercept 1.42 where the ridged least-squares fit is 2.33), fixed: the centred solves read the pivot floor against each pivot's own diagonal, which every system the raw floor accepted also passes, so no stream that fitted moves | `expectile/review_tests.rs::exp_2_a_constant_feature_is_fitted_by_the_window_and_the_refit`, `::exp_2_a_feature_far_from_zero_is_fitted_by_the_window`, `::exp_2_a_dead_feature_beside_a_wide_one_is_fitted`, `::exp_2_the_one_pass_fits_a_feature_far_from_zero` (`should_panic`) | e6a1eee8, 7495cf53 |
| GLM-2 | (2) confirmed: under decay the recursion's sums are `h_t = λ h_{t−1} + (1 − λ) R β_{t−1} + w W z x̃`, a penalty toward 0 aged as the first row plus a pull toward the fits before, not the documented `F_t`'s; the module docs, `glm`'s docstring and docs/MODELS.md corrected. (1) the refit's undecayed `R` beside decayed weights is confirmed as stated and was documented as such; kept, and now said plainly in the docs: over 200 seeds of the review's case (binomial, 200 rows, `R = 1`), the full `R` gives the lowest excess log-loss against the truth at a half-life of 5 rows (mean 0.057, against 0.114 for the decayed likelihood's maximum and for `R` aged as the doc's `F_t`), and is within 0.002 of the best at 20 and 50 | `glm/review_tests.rs::glm_2_the_recursions_identity_under_decay_is_the_proximal_one`; `glm/tests.rs::the_refit_reads_the_decayed_weights` pins (1) | e6a1eee8 |
| EXP-3 | confirmed (a read `d` rows on, on the row clock, keyed its window one row on whatever `d`; a row of weight 0 read the window without the step), fixed: `predict` and a row of weight 0 key the window `d_clock` rows on, as a clock of numbers keys it `d_clock` units on. The bank scores a row of weight 0 by `predict`, so it moves only `predict` under `embargo` on a row-clock window | `expectile/review_tests.rs::exp_3_a_read_on_the_row_clock_keys_its_window_d_rows_on` | e6a1eee8 |
| EXP-4 | confirmed (a floor of 3, the window holding 5 before the row and 1 after the rows that left: `pred` null, `weight_sum` 5, no reason), fixed: the gate reads the weight the row reports, the window's before the rows that leave at the row, as `n_eff` and `hawkes` read it (hard rule 8) | `expectile/review_tests.rs::exp_4_the_gate_reads_the_weight_the_row_reports`; `tests/test_expectile.py::test_a_clock_window_withholds_only_where_its_weight_says_so` | e6a1eee8 |
| GLM-3 | (1) confirmed (a group past `refit_budget` under the Gamma's inverse link or the Tweedie's power link: every coefficient 0, no mean, null forever), a decision for the user (below). (2) confirmed (a row moved its own `η` by 2.58 with the cap defaulted off for a refit that never ran), fixed: a given-up refit keeps the start's cap for good | `glm/review_tests.rs::glm_3_a_given_up_refit_under_a_link_undefined_at_zero_never_learns` (`should_panic`), `::glm_3_a_given_up_refit_keeps_the_starts_cap` | e6a1eee8, fa02efe6 |
| GLM-5 | confirmed (a row of finite `G` and a score of `[1.07e211, ∞]` left 1e298 in `G` with no step), fixed: a row whose score is not finite teaches nothing, as one whose `G` overflows does. A solve that fails on a finite system still keeps its row's `G`, which the rows before a definite `G` need under a vanishing ridge (the review's fix, keeping `G` only on a solved step, failed `mutant_tests.rs::a_solve_that_fails_in_a_learned_row_is_counted`) | `glm/review_tests.rs::glm_5_a_row_whose_score_overflows_leaves_the_gram_as_it_was` | e6a1eee8 |
| EXP-5 | (1) not reproduced: a warm start whose every residual squared overflows (`β_1 = -1e160`) reaches the cold start's fixed point, converged; the first pass's loss and scale are the new fit's, both finite. A scale that is not finite is now no yardstick for "moved" (a guard). (2) not a defect: a weight of `5e-324` at the side `0.1` teaches nothing, and every output after it is the stream's without it, to the bit | `expectile/review_tests.rs::exp_5_an_overflowing_warm_start_still_reaches_the_fixed_point`, `::exp_5_a_subnormal_weight_moves_no_output` | e6a1eee8 |
| GLM-4 | (3) confirmed (the Tweedie deviance NaN at `p = 30`, `μ = e^−30`), fixed: `+∞`, factored only where the plain form is NaN. (4) confirmed (the Gamma deviance `+∞` at `y = 1e-200`, `μ = 1e150`), fixed: `ln y − ln μ` only where `y/μ` leaves the doubles. (5) `takes(∞)` confirmed, fixed; the review's rows for `β` past the largest double not reproduced (the slope stays at 1.7e279 over 40 rows); the rebuilt `G` and `moved` guards not added, no case reaching them being found. (1) and (2) confirmed by test, a decision for the user (below). The Pass-2 Tweedie power-link case not reproduced: `working` already returns no weight where `z` is not finite | `glm/review_tests.rs::glm_4_a_tweedie_deviance_past_the_largest_double_is_infinite`, `::glm_4_a_gamma_deviance_whose_ratio_underflows_is_finite`, `::glm_4_no_family_takes_an_infinite_target`, `::glm_4_the_reviews_rows_keep_every_coefficient_finite`, `::glm_4_the_score_at_the_clip_loses_its_digits` (`should_panic`), `::glm_4_the_binomials_weight_at_the_clip_loses_its_digits` (`should_panic`) | e6a1eee8 |
| GLM-7 | not reproduced: `Glm::step_into` takes the handed stamp on every row (51671e2e's line 894, before any weight test), and a row refused for a feature is stepped again at weight 0, which takes it | `glm/review_tests.rs::glm_7_a_stamp_is_taken_by_its_own_row_whatever_its_weight` | e6a1eee8 |
| GLM-EXP-LOW | (1) confirmed, fixed: `expectile`'s docstring says the ridge fits from the first row. (2) confirmed: `default_glm_refit_rows` counts the intercept whether or not the spec fits one; the docstring now says `k` is the features plus one (a code change would move the default for specs without an intercept above 19 features). (3) confirmed, fixed: a negative weight a caller hands `Glm` directly is a weight of 0 (the bank refuses one). (4) fixed: `τ = 1/2` solves once. (5) confirmed, fixed: `checked_mul`, and a span's buffer without its first stamp or a stop at the cap without a span refused as damaged. (6) confirmed, fixed: the refusal names `refit_span` | `glm/review_tests.rs::low_3_a_negative_weight_is_a_weight_of_0`, `::low_5_...`, `::low_6_the_links_refusal_names_both_refits`; `expectile/review_tests.rs::low_5_...` | e6a1eee8 |

### Left for the user (batch b)

- **EXP-2 (2), the one pass and a feature far from 0.** The one pass keeps
  the mean-form Gram over `[1, x]`, so a feature whose level is about 1e9
  times its spread (`1e6 + 1e-3 N(0, 1)`) has a centred block read off a
  level of 1e12 with its digits lost: the solve fails on 118 of 200 rows
  (31 under a half-life of 69 rows, the review's 124 of 200), the fit
  before stands, and a prediction reads up to 4.9 from the ridged
  least-squares fit's against noise of spread 1.5. Options: (a) keep the raw
  moments and say in the docs that such a feature is centred before the
  bank (the window and the refits now fit it exactly from their rows);
  (b) keep each target's weighted feature means beside a centred Gram
  (Welford's update): `k` more numbers a target, a few flops a row, a
  layout change whose loader can only form the centred block from the raw
  one, with the loss it has, and every one-pass `expectile` output moving
  in its last bits. Recommended: (a). Held by
  `expectile/review_tests.rs::exp_2_the_one_pass_fits_a_feature_far_from_zero`
  (`should_panic`).
- **GLM-4 (1) and (2), digits at the clip.** The score `W (z − η)` keeps
  two digits of `y − μ` at `η = 30` and a residual of order 1, and the
  binomial's `1 − μ` three digits at `η = 30`. Options: (a) keep; (b) a
  closed-form score in `Working` (`(y − μ)/(g'(μ) V(μ))`, `y − μ` under
  the canonical links) and `1 − μ = 1/(1 + e^η)` above 0. (b) is right,
  but it moves every `glm` output's last bits, so every kept `glm` state
  fixture needs rule 5's exception to continue, while the digits it gains
  are at the clip's edge alone. Recommended: (a) until a release that
  moves `glm`'s bits for another reason. Held by the two `should_panic`
  tests named above.
- **GLM-3 (1), a given-up refit under a link with no mean at 0.** Such a
  group never learns. Tried here and set aside: starting its one-pass fit
  from the intercept at the link of its first target, as the refit
  starts, under the start's cap of 1. Against statsmodels' `GLM` (Gamma,
  `InversePower`) on six streams of a well-specified inverse-link Gamma it
  fit within 0.26, 0.26, 0.53, 0.60 and 0.45 standard errors, and 40 on
  the sixth at 3,000 rows (the refitted fits: 0.05 to 0.16); a cap that
  holds a row's mean within a factor `e` (`|Δη| ≤ (1 − e^−|q|) η`) left
  that stream at 35 and took another to 4.0. The one-pass start without
  its refit is not sound under these links, which is why the spec refuses
  them without a refit. Options: (a) never give a refit up under such a
  link, so `refit_budget` does not bind such a spec (each group's buffer
  is still held to `max_state_bytes` and freed after its refit); (b) give
  it up and withhold its predictions with a named reason, where today
  they are null with none; (c) the start above, with its tail; (d) refuse
  a finite `refit_budget` beside such a link. Recommended: (a), since the
  link's own refusal says a refit is what it needs. Held by
  `glm_3_a_given_up_refit_under_a_link_undefined_at_zero_never_learns`
  (`should_panic`).
- **GLM-2 (1), the refit's ridge** is kept (above): the review's 54% slope
  cut at a half-life of 5 rows is the ridge's shrinkage over 7.7 rows'
  weight, which lowers the excess log-loss; the user may prefer the
  decayed likelihood's maximum.
- **The log link's clip, `e^30`**, caps the mean at 1.07e13: a Gamma or
  Tweedie amount above it cannot be fitted (GLM-1's case is such a
  target). The clip keeps `μ` finite; under the log link any clip up to
  about 700 does. Raising it moves outputs only on rows past 30. Not
  changed here.
- **EXP-4's direction.** The gate now reads the weight before the rows
  that leave at the row, as `n_eff` and `hawkes` do. The other reading,
  the weight after them in the field, the gate and the accessor, is the
  weight the fit is read from, but the bank reads each target's weight
  before the row's step and has no stamp to trim it at: it needs a trait
  method that reads the weights at a stamp.

## Batch i: the surfaces and the state file: spec validation, Python, PyO3, envelope and checkpoints, atomic IO, checksum, CLI (PLAN 578i)

On `review-578i`, cut from 51671e2e. "Confirmed" means the named test
failed on 51671e2e for the review's reason (a Rust test whose check did not
exist there is marked so: it was written against the code as it stood, and
fails without the fix by construction). The coordinator's word at
2026-11-24 (the machine's load at 35) cut the batch's builds to four jobs
and two test processes. No output field of any model moved,
and no state layout changed: every fix refuses, names or documents.

| ID | verdict | test | commit |
|---|---|---|---|
| SPEC-1 | confirmed (a `feature_sets` name, a `deco` block and an `ew_class` class each panicked the Arrow export in Python on 51671e2e, `NulError`, after the bank had learned the chunk; a feature's NUL was already refused at the frame), fixed: `Spec::validate` refuses a NUL in a feature and in every column a spec names (`clock`, `session`, `weight`, `group`, `label_clock`, `hazard_col`, `exog_tvtp`), and `Spec::check` refuses any output field whose name holds one, naming the field and what it is of | `crates/rheon-polars/tests/review_surface.rs::a_nul_in_any_name_a_spec_carries_is_refused` | cbf3e558 |
| SPEC-4 | the cadence part **contradicted a recorded decision and was not changed**: task 538 takes `inf` on purpose (the clock never brings the cadence round and its count alone fires it, `_INF_OK`), so updates are not frozen; `test_the_inf_table_matches_the_rust_side` holds it, and a first fix that refused `inf` broke it. The JSON part confirmed (`resolved_defaults` refused the spec as a defect; the state's JSON export wrote `null`), fixed: the three fields are `humanfloat` tags, `"inf"` in JSON and a bare double in msgpack, so no state's bytes move | `review_surface.rs::an_infinite_cadence_reads_back_from_json` | 064c5b84 |
| SPEC-5 | confirmed on all five, fixed: a count given beside a span is refused at 0 too, by both keys and both values | `review_surface.rs::a_refit_count_beside_its_span_is_refused_at_zero_too` | cbf3e558 |
| SPEC-6 | confirmed (`audit` and `deco` refused in Python on 51671e2e; `bocpd`, `corrchange`, `rcov`, `marginal` in the Rust test), fixed by an allowlist, `ModelKind::names_its_intercept`. The review's list of eleven missed `pa` and `ar`, whose terms name `intercept` too (`coef_fields`), so the list has twelve | `review_surface.rs::intercept_is_refused_only_where_the_fit_names_its_own_constant` | cbf3e558 |
| SPEC-7 | confirmed (`Spec::validate` took both under `loss="squared"`; the model refused them only after every other problem), fixed in the Rust and the Python tables; a span refit at an infinite cut, and beside `ridge_scale="sum"`, names its span in `huber`, `quantile` and `sgd` | `review_surface.rs::sgds_huber_span_and_cap_are_the_huber_losss` | cbf3e558 |
| SPEC-9 | (1) confirmed for `hawkes` and `bocpd` under a clock `hazard`, fixed: refused under `label_clock`, as `holt`, `kalman`, `hmm` and `micro` are, each prediction reading the clock since the newest learned row. Featureless `garch` not changed: it reads no clock; its recursion runs in the order rows are learned, which under `label_clock` is the labels' order, as every model's learning is (see the open list). (2) confirmed (`model.l1_ratio` named on `lasso` against `ewridge`), fixed: specs of two models differ at `model.type` | `review_surface.rs::label_clock_refuses_the_kinds_whose_prediction_reads_the_learned_clock`, `::specs_of_two_models_differ_at_the_type` | cbf3e558 |
| SPEC-10 | (1) confirmed (both repeats reached the duplicate-field refusal), fixed: refused by their own key. (2) confirmed for a repeat (`"5m", "300s"` blamed `kernel_decays` with a raw decay) and for a half-life whose decay overflows (1e-310: `inf`, blaming `kernel_decays`); **not reproduced** at the review's 1e-308, whose decay, 6.9e307, is finite and is taken. Fixed: refused by `kernel_half_lives`. (3) confirmed on all five kinds; fixed on `glm` and `expectile`, which no release has carried; **left for the user** on `huber`, `quantile` and `sgd` (below), held by a `should_panic` test | `review_surface.rs::a_repeated_level_or_kernel_is_refused_by_its_own_key`, `::a_refit_budget_with_no_refit_is_refused`, `::a_refit_budget_with_no_refit_on_a_released_kind_is_refused` (`should_panic`) | cbf3e558 |
| SPEC-11 | confirmed (`rewm_quantile`, `rrolling_max` and `rrolling_min` unnamed), fixed: the list is the operators whose direction is forward, read from the table of heads | `review_surface.rs::the_formula_refusal_names_every_operator_looking_ahead` | cbf3e558 |
| GLM-6 | confirmed (a Poisson `glm` learned a chunk whose resolved targets were below 0), fixed: a `glm`'s resolved formula targets are held to its family's support before any stream learns them, refused naming the target, the value and the row, and the chunk refused whole (the resolvers' snapshot is taken for a `glm` spec). The review's other half, refusing formula targets on a binomial `glm` as `sgd(loss="logistic")` is refused, is not taken: a fraction in [0, 1] is in the binomial family's support, as glum takes one | `review_surface.rs::a_glm_formula_target_outside_its_support_is_refused_by_row` | cbf3e558 |
| STATE-1 | confirmed by reading (`kind_of` took `Seal::Absent`), fixed: a stage is an enveloped state whose checksum holds; a file from before the envelope is told to be loaded and saved again | `crates/rheon-polars/src/checkpoint.rs::tests::a_stage_without_its_checksum_is_refused_and_the_old_checkpoint_kept` | 1dd7d879 |
| STATE-2 | (2) confirmed by reading (two IPC reads without `catch_unwind`; the byte sweep over a `refresh_time` value's IPC is the test), fixed through one reader, `envelope::read_ipc`. (1) confirmed by reading, fixed: a claimed length reserves at most 4 KiB of bytes (`Bin`, `ByteBuf`) or 1 MiB of elements (the state's `humanfloat` sequences, which had the same pattern); no test can see a reservation, and none allocates 4 GiB on this machine | `crates/rheon-polars/src/refresh/held.rs::tests::a_damaged_arrow_value_is_refused_and_never_panics`, `::an_array_claiming_more_bytes_than_it_holds_is_refused` | 1dd7d879 |
| IO-1 | confirmed (the block grew to the write's size), fixed: a write of a block or more goes straight through, the bytes and the digest those of a block-at-a-time pass on every split | `crates/rheon-polars/src/checksum.rs::tests::a_large_write_is_not_copied_into_the_block` | 1dd7d879 |
| IO-2 | (1) confirmed (a link planted at the temporary's name was written through and renamed over the destination), (2) confirmed (the temporary was 644 while written over a 600 file), (3) confirmed by reading (`& 0o7777`): fixed, the temporary is created new with the destination's permission bits and the owner's write, a taken name passed over up to `TEMP_TRIES` times, and the mode copied is `& 0o777`. (4) confirmed by reading, fixed: only the library's own `InvalidInput` (no OS code) is a `ValueError` from `save_checkpoint`. (5) confirmed, fixed with CLI-5 | `crates/rheon-polars/src/atomic.rs::tests::a_temporary_is_created_new_and_never_through_a_planted_link`, `::the_temporary_is_as_private_as_the_destination_while_written`, `::a_failed_write_leaves_the_destination_alone` (both sides of `TEMP_TRIES`) | 1dd7d879 |
| STATE-3 | confirmed in part, fixed in part. Fixed (each check absent on 51671e2e): an `audit` column's minimum above its maximum (the `clamp` panic); `micro`'s cluster count against its potential summaries' labels (the underflow); `rcov`'s rings past their depth (`fin`, `head`, `pre_ring`); `ftrl`'s negative sums and weights; every `EwDiag`'s negative weight, negative centred sum or non-finite mean. **Open, not done in this batch** (confirmed by reading, no test yet): the span caps' `held_bytes <= cap` on load in `glm`, `expectile`, `sgd`, `robust`, `bocpd`, `hmm`; `rls`' zero pivots; `StepGain { resting, n: 0 }`; the weights of `glm`, `expectile`, `bocpd`, `hmm`, `lasso` | `crates/rheon-core/src/audit/tests.rs::a_state_of_the_wrong_shape_is_refused` (a seventeenth edit), `cluster/micro.rs::tests::a_cluster_count_that_is_not_the_labels_is_refused`, `rcov.rs::tests::a_state_of_the_wrong_shape_is_refused` (two rings), `ftrl.rs::tests::a_negative_sum_is_refused`, `ewdiag.rs::tests::a_weight_or_moment_no_stream_writes_is_refused` | 1dd7d879 |
| UTIL-1 | confirmed (`b"group-1"` and `b"group-1\x07"` hash alike), **not changed** (the coordinator's to overrule): every map keyed through the hasher compares the keys whose hashes meet, so it costs a probe at most, and a fix moves every hash and adds a word to each key's hashing on the per-row path. The module doc says it is no fingerprint, and a test holds the collision | `crates/rheon-core/src/hasher.rs::tests::a_short_tail_hashes_as_the_word_its_length_completes` | 1dd7d879 |
| CLI-2 | **not reproduced as a break**: the CLI wrote a passed-through `List(Datetime(us, zone))` in NDJSON, as the zone's wall time with its offset, `2024-01-01T00:00:00-05:00`, beside the same instant in a column written in UTC. Made consistent: a list's and an array's elements are written in UTC as a column is (NDJSON only; a CSV refuses a list of instants as before) | `crates/rheon-polars/tests/runner.rs::a_list_of_zoned_instants_is_written_in_utc_in_ndjson` | f5824dd6 |
| CLI-3 | confirmed (a group whose last row was skipped left the closing lines), fixed in `LastReasons::note`: a row the spec skipped (null `weight_sum`) is not noted. A row of weight 0 has a `weight_sum` and is noted, so the review's warning about it holds; `LastReasons` is kept, since `last_row` is the last learned row and would lose a predicted row of a null target, the common last rows of a forecast | `crates/rheon-polars/src/runner.rs::tests::a_skipped_last_row_keeps_the_reason_before_it` | f5824dd6 |
| CLI-4 | confirmed in a form the review did not name: `model_key` has read each key alone beside its `type` since PC11, and a missing required key gives the same "missing field" message with any key, so the first was blamed (`at spec "m": model.l1_ratio` on a `lasso` without `penalties`). Fixed: a key is blamed only if the model without it no longer gives the message | `crates/rheon-cli/tests/run.rs::a_missing_model_key_blames_no_key_given` | f5824dd6 |
| CLI-5 | confirmed by reading and in Python, fixed: a path ending in a separator names no file, refused by `temp_name` (every writer), `check_parent` (the run and the dry run, output and state) and Python's `_save_path` (every plan form), which also refuses `""` | `crates/rheon-cli/tests/run.rs::a_path_ending_in_a_separator_is_refused_before_the_run`; `atomic.rs::tests::a_path_ending_in_a_separator_is_refused_at_once`; `tests/test_frame.py::test_a_save_state_that_names_no_file_is_refused_before_the_stream` | 1dd7d879, f5824dd6 |
| PY-3 | confirmed (`123` and `"ab"` raised `dict()`'s raw errors; a read-only mapping's `inf` passed), fixed: `Mapping` is read by the shape check, its description, the finite check, `_lists`, `_json` and `_bad_word` | `tests/test_spec_validation.py::test_drift_options_are_checked_as_any_mapping_is` | dd80ef75 |
| PY-13 | (2) confirmed on all six, fixed: each is told its floor of 1 for a negative value, and 0 is left to the Rust side, which says why. (3) confirmed (`{"delta": True}` reached serde), fixed: each `drift_options` key is held to the Rust type, counts to 0 or more. (4) confirmed, fixed. (1) **confirmed, deferred**: `check()` reports nothing for a Poisson `glm` on `x = 50,000 ± 0.3`, whose predictions sat at 2.17 while the rates spanned 0.15 to 1.6e1; `glm`'s level and spread limits need the measurement the other models' had (out-of-sample loss of 0.05 at half-lives 20, 200 and infinite) before it joins `_UNCENTRED_LIMITS`, and so does `fit_intercept=False` | `tests/test_spec_validation.py::test_a_count_below_its_floor_is_told_the_floor`, `::test_each_drift_option_is_held_to_its_own_type`; `tests/test_unstable.py::test_the_warnings_own_list_is_the_unstable_surfaces` | dd80ef75 |
| PY-4 | confirmed (the warning came before a refusal, in Python's quotes), fixed | `tests/test_hawkes.py::test_the_window_warning_follows_the_specs_own_checks` | dd80ef75 |
| PY-5 | confirmed (25 docstrings), fixed: each says the one class | `tests/test_exception_types.py::test_no_docstring_says_an_error_is_both_a_type_and_a_value_error` | dd80ef75 |
| PY-12 | (1) **not reproduced**: batch e's `test_grid_search.py::test_the_selected_and_averaged_fields_move_no_candidates_score` holds that the switches move no candidate's rows, so no innocent candidate is blamed for them. (2) confirmed, fixed in `huber`'s and `sgd`'s `huber_refit_span` | (2) by reading | dd80ef75 |
| PYO3-1 | confirmed by reading (a `&self` getter: PyO3 raises its own `RuntimeError` there), fixed: read under the borrow, refused with `busy("threads")` | `tests/test_error_messages.py::test_reading_threads_while_the_bank_learns_says_so` | dd80ef75 |
| PYO3-2 | (3) confirmed (`target-cpu=x86-64-v3` recorded nothing; `-avx2` after `+avx2` was ignored), fixed: the psABI levels' features, then each `+`/`-` in turn. (2) the POSIX map closed when `mprotect` fails, fixed. **Open, for the VM**: (1) the `OSXSAVE`/`XGETBV` test and (2) Windows' `PAGE_READWRITE` then `VirtualProtect`, machine code and Windows calls that only an x86-64 and a Windows run can test; a wrong byte there would kill the import | `tests/test_cpu_check.py::test_the_build_script_records_the_features_rustc_was_told` (seven new cases) | dd80ef75 |
| PYO3-3 | confirmed by reading, fixed: the bank's `save`, `save_bytes`, `save_json_string` and `load_bytes` run without the GIL; `gramfit`'s `k²` is checked, and a float64 buffer it cannot read is the Gram's error, not a bare `BufferError` | by reading (a timing test of the GIL would be flaky at this machine's load) | dd80ef75 |
| TEST-11 | confirmed (`glm` and `expectile` refit at 210 rows, past every stream the suites draw), fixed: `TestRefitProperties` runs `quantile`, `glm`, `expectile` and the span forms of `huber` and `glm` at counts and spans a 40-row stream reaches (`glm` on 0/1 targets, as the sweeps feed it), each with its own activation check; all 28 pass. **Open**: `SWITCHES`' post-2.0 modes and `garch` in the temporal suites, left to batch j | `tests/test_properties.py::TestRefitProperties` | dd80ef75 |

### Left for the user (batch i)

- **SPEC-10 (3) on `huber`, `quantile` and `sgd(loss="huber")`**: a
  `refit_budget` with no refit (`*_refit_rows = 0`, `huber_delta = inf`,
  `ridge_scale = "sum"`) is taken and bounds nothing. Every door, the load
  of a state among them, checks its specs, so refusing it would refuse to
  load a 2.0.0 state saved with such a spec. Options: (a) refuse it at every
  door, breaking those loads (rule 5's waiver ended at 1.0); (b) refuse it
  for a new spec and take it on load, a second rule for one setting;
  (c) keep taking it and say in the docstrings that it bounds nothing with
  the refit off. **Recommendation: (c)** now, and (a) at the next major
  version. The `should_panic` test turns plain if (a) or (b) lands.
- **UTIL-1**: whether to change the hasher (every hash moves, a word more a
  key on the per-row path) though no map is wrong. Recommendation: leave
  it, as documented.

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

## Batch a: the GARCH family (PLAN 578a)

On `review-578a`, cut from 51671e2e. "Confirmed" means the named test
failed on 51671e2e for the review's reason. Test paths are under
`crates/rheon-core/src/garch/` unless they start with `tests/` or
`crates/`; `review_tests.rs` holds one test or more a finding, named by
its ID. A test marked `should_panic` holds a confirmed defect whose fix is
the user's to allow. Every fix below leaves the goldens and every state
fixture's continuation as they were; the layout change (BANK-10 (2))
regenerated the fifteen current `garch*` and `dcc_margins` fixtures, whose
continuations did not move, and two degenerate cases' withheld fits
(`garch_started`, `garch_egarch`, `garch_skewt`, `garch_figarch`: the fit
of rows with no process, kept in the state and never reported) moved with
GARCH-1's line search. The kept `v71/` sets convert and continue as before.

| ID | verdict | test | commit |
|---|---|---|---|
| GARCH-1 | confirmed (a concave quadratic bent up past `x = ½` over `{x ≤ 1, y ≥ 0}`: the projected Newton step to the corner, its model predicting a loss of 8.5, was taken on the slopes' trapezoid while `f` fell from −41.5 to −45, and the fit ended there), fixed: the slopes decide only a step whose two model terms are each at most twice the rounding floor (an unprojected Newton step under the floor always is); any other step is read from `f`, and taken only with `gᵀΔ > 0` and Armijo's rise. The review's fix as given | `optim.rs`'s tests: `a_step_along_which_f_falls_is_never_taken`; `review_tests.rs::garch_1_a_step_the_quadratic_model_says_loses_is_read_from_f` | 7d41ce7c |
| GARCH-2 | not reproduced: of twelve 2,000-row Gaussian GARCH(1,1) series fitted with a t, nine fit `ν = 300` and every one started at its first try. Soundness is read from the decrement inside the set (`decrement_in`, the step projected onto the set with the shape's box in the curvature's metric), which at an optimum on the `ν` face is the rounding's; the review read the unconstrained decrement alone | `review_tests.rs::garch_2_a_t_fit_at_its_shape_bound_starts_at_its_first_try` | 7d41ce7c |
| GARCH-3 | confirmed (both of the review's cases: from and to breaking the guard at `\|z\| ≈ 5` and `3.375`, the cut took `β = 0`, mean `ln \|μ\|` 0.72 against the step's end's 0.12), fixed: `(α, γ, β)` shrunk together by the largest factor that keeps the guard, `ω` held. `μ` is linear in them, so the mean is `ln t` plus its value at the step's end: monotone, and at `t = 0` far below the bound. The review's Pass-2 objection (no homogeneity) is to another `μ`; the guard's is homogeneous of degree 1 | `review_tests.rs::garch_3_a_cut_from_a_broken_point_keeps_the_filter_contracting`; `guard.rs`'s test updated to the new rule | 7d41ce7c |
| GARCH-4 | (2) confirmed (the t and skewed t at `z ≈ 10¹⁵⁵`: density `−∞`, scores NaN; a t series at `10⁻⁵⁵` met by a return of `10¹⁰⁰` never started on the old build), fixed: `ln(1 + m²/n₂)` and the scores' ratios read in `1/m` where `m²` overflows; continuous with the plain form, and the variance score's factor tends to `ν`. (1) not reproduced as stated: a series at `10⁻⁸⁰` never starts (`½ (∂ ln s/∂ω)²` overflows in the start's information and curvature), so no started `H` is reached; that series reporting nothing is a decision for the user (below) | `review_tests.rs::garch_4_the_t_density_and_scores_past_the_largest_square_are_numbers`, `::garch_4_a_t_score_past_the_largest_square_leaves_the_state_finite`, `::garch_4_a_series_at_a_scale_of_1e_minus_80_starts` (`should_panic`) | 7d41ce7c |
| GARCH-5 | (1) confirmed (`hi = 10⁻⁶`, `ln s = 700`: an infinite variance), fixed: `hi + ln s − ln hi`, `arch`'s `egarch_recursion` in its own order. (2) the slope confirmed (0 where a central difference of the two-row run reads `9.6·10⁻⁷`), fixed: `∂ ln s_raw/∂θ` over the bounded `s`, 0 past the exponent's cap; the "dimensional mismatch" is `arch`'s own formula and kept. (3) the backcast confirmed (75 zeros, then an EGARCH series: 3 tries, no start), fixed: EGARCH's backcast of 0 read as the rows' mean square. The online claim not reproduced: the 0.94-weighted mean of `e²` never reaches 0 (0.94 times 8 subnormal ulps rounds back to 8), and through 13,000 zeros the variance follows the lower bound down to `1.5·10⁻³²³`, positive, and reports on every row | `review_tests.rs::garch_5_the_egarch_upper_bound_does_not_overflow`, `::garch_5_the_egarch_sensitivity_past_the_upper_bound_is_its_slope`, `::garch_5_an_egarch_whose_first_rows_are_zero_starts`, `::garch_5_a_long_run_of_zeros_leaves_egarch_reporting`; `mutant_tests.rs`'s bound test updated to the slope | 7d41ce7c |
| GARCH-6 | confirmed (`ν = −10¹⁷`, `ν = 2`, NaN, `ν = 10⁹`, `λ = 1.5`, FIGARCH `β = 1`, a weight below 0, a buffer past the cap: each loaded; `ln Γ(−10¹⁷)` ran past 5 s), fixed: `Garch::restore` refuses a state whose values are out of range (`Series::has_values`: weights and running means finite, weights at 0 or above, buffered weights above 0, and once started the parameters in the process's set to `10⁻⁹` of each bound, the shape in its box) or whose bytes pass `max_state_bytes`; `ln_gamma` and `digamma` return NaN off `(0, ∞)` at once. Every kept fixture loads | `review_tests.rs::garch_6_a_damaged_state_is_refused`, `::garch_6_ln_gamma_outside_its_domain_returns` | 7d41ce7c |
| GARCH-7 | the mechanism confirmed, not a defect: a target's own weight, which `min_weight` gates, is the weight its fit rests on, and a start's ring lets its oldest rows go unlearned, as a window's rows that left teach a windowed fit nothing (hard rule 8's window clause). The review's case gates on the 500 rows of its fit while `weight_sum` counts 1,000; pinned. A choice for the user (below) | `review_tests.rs::garch_7_a_targets_weight_is_the_weight_its_fit_rests_on` | 7d41ce7c |
| GARCH-8 | (1) not reproduced at 51671e2e: the row's bytes are `series_bytes(rows + 1)`, the larger of the buffer and the ring, so a FIGARCH span start at a cap of 8,232 bytes buffers its rows. (2) confirmed (a FIGARCH span start tried on 61 rows, where a start of rows is refused under 2,000), a decision for the user (below) | `review_tests.rs::garch_8_a_figarch_span_start_at_its_exact_cap_buffers`, `::garch_8_a_figarch_span_start_tries_on_no_fewer_rows_than_a_start_of_rows` (`should_panic`) | 7d41ce7c |
| GARCH-9 | not reproduced: every form, the targeted GARCH and FIGARCH among them, reports the same bits at weights `2⁻³⁴`, `2⁻⁶⁰` and `2⁴⁰` as at 1 (two seeds each). The floor `1 + \|f\|` does reach more steps' gains at tiny weights, but the slopes and Armijo's rule accept the same steps there | `review_tests.rs::garch_9_a_power_of_two_on_every_weight_moves_no_report` | 7d41ce7c |
| GARCH-10 | confirmed (300 random metrics of condition `10⁸` to `5·10¹⁴` on GARCH's set at `ω ≈ 10⁸`, GJR's and FIGARCH's: the metric projection up to 1 off the minimizer `crate::oracle::project_qp` finds, an active-set enumeration on faer's LU, and outside the set by as much), fixed: where Dykstra's sweeps do not reach their stop, the projection is solved from its optimality conditions (`HalfSpaces::project_exact`); against the minimizer in exact rational arithmetic (Python's `fractions`) on every such call in the state fixtures' runs, the old path was up to 1.5 off in the metric and the exact solve within `5·10⁻¹²`. A point whose sweeps stopped is kept: there the old path was within `10⁻¹⁰` and the exact solve no closer, so the goldens and continuations do not move. The Euclidean sweeps' stop is now `10⁻¹⁴` of the point's largest coordinate (at least 1), so a variance level of `10⁸` stops, bit for bit the old stop for coordinates within 1. The cleanup's single pass is left: the bounds that matter are made exact after it (GARCH-12) | `review_tests.rs::garch_10_the_projection_in_a_metric_is_the_minimizer` | 7d41ce7c |
| GARCH-11 | (1) the wording confirmed and corrected: the statistic's degrees of freedom are the parameters the process adds to one variance, FIGARCH's `d = 0`, `φ = β` two equations leaving its four parameters one variance, so 3 (`fit.rs`, the module docs, the docstring, docs/MODELS.md). (2) confirmed, corrected: docs/MODELS.md says FIGARCH holds the larger of the buffer and the ring. (3) confirmed, fixed: a start of zeros waits as `not_identified` | `review_tests.rs::garch_11_a_start_of_zeros_waits_as_not_identified` | 7d41ce7c |
| GARCH-12 | confirmed (GJR at `α + γ = −4.1·10⁻¹⁵` and `e = −10⁸`: a variance of −38.7 reported; FIGARCH at `λ₁ = −9·10⁻¹³` and `e = 10⁹`: negative; the GJR t's `H` NaN; `project_metric` left GJR's `α` at `−1.6·10⁻¹⁴`), fixed: after each step's projection and at the start's fit, the bounds that keep each shock's weight at 0 or above are made exact (`series::exact`; GJR's as `γ ≥ −α`, whose sum rounds to an exact 0, EGARCH's box beside them), FIGARCH's weights are read at 0 or above, and a variance that is not a positive number is neither reported nor learned. Over three GJR series with `α = 0` the old build's steps kept the bounds (a pin, not evidence) | `review_tests.rs::garch_12_no_variance_at_or_below_zero_is_reported`, `::garch_12_the_shock_weights_bounds_are_made_exact`, `::garch_12_a_step_keeps_every_shock_weight_at_zero_or_above` | 7d41ce7c |
| BANK-10 | (1) confirmed (a broken bank given a clock of another type: `fit_predict_arrow` and `predict_arrow` named the clock, `fit_predict` and `predict` the break), fixed: both ask first, and the Python binding asks before it converts the frame. (2) confirmed, fixed: `Garch::next` and `warning` are `#[serde(skip)]`, as every other model's are; the layout loses two keys at schema 72 (unreleased), the current fixtures regenerated, the `v71/` sets converting to them | `tests/test_exception_types.py::test_a_broken_bank_refuses_every_entry_point_before_it_reads_the_chunk`, `crates/rheon-polars/tests/arrow_chunk.rs::a_broken_bank_says_so_first_at_the_arrow_entry_points` | 7d41ce7c |

Per-row cost (arm64, release build of the library's tests, one series
after its 2,000-row start, 50,000 rows, best of seven, 51671e2e then this
branch, load 4.6 to 8): `garch` 316.5 / 315.0 ns, `gjr` 346.8 / 350.5,
`egarch` 562.8 / 568.4, `figarch` (1,000 lags) 2,766.6 / 2,803.4, `garch`
t 397.8 / 399.4, skewed t 511.8 / 506.2: within 1.3% either way, the
spread of such runs. The row adds the exact bounds' few comparisons after
each step, a test of the variance's sign, and for FIGARCH a `max` a lag;
the exact projection and the guard's new cut run only on the rare rows
that need them. Not timed on x86-64. State: two keys fewer a `garch`
(15 bytes).

W12's mutants (the coordinator, 2026-11-24): five survivors of CI's
changed-lines pass in `optim.rs`'s `maximize` (`/` → `*` in the step's
scale, `+` → `*` in the rounding floor, `>` → `>=` at the floor, `+` → `-`
in Armijo's line, `+` → `*` in the stall bound), each killed by a test on
an objective the line search's decision alone settles:
`the_rounding_floor_is_read_in_each_parameters_unit`,
`a_gain_at_the_rounding_floor_is_read_from_the_slopes` (an exact tie, the
gain `10⁻¹⁰` to the bit), `a_step_short_of_armijos_gain_is_halved`,
`a_step_that_no_longer_contracts_at_the_rounding_ends_the_fit`; and
GARCH-1's new lines by `a_step_along_which_f_falls_is_never_taken`. A
local `cargo mutants --in-diff` of `maximize` (its changed lines and the
five survivors' lines, the library's tests alone) caught 39 of 43, the
five among them; the three it missed, the gain read as `slope + curve` and
the stall bound read as `10⁻¹¹/(1 + size)` or `10⁻¹¹ (1 − size)`, each
have a case of their own in the last two tests (a gain of half the floor,
a stall about 1,000).

### Left for the user (batch a)

- **GARCH-8 (2), FIGARCH's start under `start_span`.** A start of rows is
  refused under 2,000 rows for FIGARCH (its ring of 1,000 lags is mostly
  the backcast on fewer, and `d` fits low); a start of a span fits on as
  few as 50 rows, any form's floor. Options: (a) hold a FIGARCH span start
  to the same 2,000 rows, the review's fix: consistent, but a fixed row
  count; (b) keep 50 and say so; (c) replace both floors by a condition,
  such as `d`'s standard error from the start's information at the rows in
  hand, which no row count stands in for. Recommended: (c) as its own task,
  and (a) meanwhile only if the user accepts the row count it already has
  for starts of rows. Held by
  `garch_8_a_figarch_span_start_tries_on_no_fewer_rows_than_a_start_of_rows`
  (`should_panic`).
- **GARCH-4 (1), a series far from unit scale.** `∂ ln s/∂ω = 1/s` sits in
  the start's information squared, so below about `10⁻⁷⁷` in the returns
  it overflows and the start is never at its optimum (every row withheld,
  `below_warm_rows`), and near `10⁸⁰` the entry falls below the normal
  doubles (it starts, its `ω` steps on lost digits). Options: (a) rescale
  each series by a power of two at its start (exact, so every result in
  range keeps its bits), a field per target and a layout change with a
  fixture for the new form; (b) document the range, `10⁻⁷⁵` to `10⁷⁵`, and
  let a caller rescale. Recommended: (b) now, (a) if a use appears.
  Held by `garch_4_a_series_at_a_scale_of_1e_minus_80_starts`
  (`should_panic`).
- **GARCH-7, what a target's `min_weight` counts.** Today the weight its fit
  rests on: after a ring let rows go unlearned, fewer than `weight_sum`.
  The review's alternative counts every row the target was given, a new
  field a target and a layout change. Recommended: keep, as hard rule 8
  reads a window's weight.

## Batch h: covariance and clustering: `ew_cov`, `ew_class`, `rheon.corr`, `marginal`, `micro`, `kmeans`, `deco`, `corrchange`, refresh time, moments (PLAN 578h)

On `review-578h`, cut from 13f5ffe4. "Confirmed" means the named test
failed on 13f5ffe4 for the review's reason (a Rust check that did not exist
there is marked so: it fails without the fix by construction). Rust test
paths are under `crates/rheon-core/src/` unless they say otherwise. The
coordinator's word at 07:28 (the machine's load at 58) cut the batch's
builds to four jobs and two test processes. Every previous schema's fixture
loads, continues to the bit and converts to the current bytes: no fixture
moved, and none was regenerated.

Per-row cost, arm64 (`factor_bench`, release builds of 13f5ffe4 and the
branch, ABBA with an identical-binary control, best of 7 to 9, load 6 to
7): every case's output digest identical; `hmm` at k = 4, 10 and 20
1.01-1.03, `ew_class` (full, tied), `ew_cov` (mahal) and `lasso` at k = 20
and 200 inside the control's spread (A against A: -4% to +4%). The new
work on an ordinary row is a comparison a row (the history's share
against `2^-52`; the Kish sum's held value against the least normal
double); the from-the-row loops are out of line (`comp::from_rows`):
inline, beside `hmm`'s factor update, they cost it 1.25-1.55 at k >= 10.
No x86 figure yet (the VM's), and no budget is raised.

MARG-1's cost (arm64, `examples/marg_bench.rs`, which takes a `window`
now: 2 targets, 20 features, lags 1, 2 and 5, every lag crossed, 10% of
the second target absent, 100,000 rows, best of 5, ABBA with a control,
load 8 to 16): a window of 200 rows 1.13 (125.7 against 142.5 ms; control
0.98-1.02), no window 0.98-0.99 (inside the control). Memory: a window's
snapshot holds `6·L·T + 3·(L + C)·p·T` lag doubles where it held `2·L·T +
2·(L + C)·p·T`, 396 against 252 at that shape (the whole snapshot, with
its `(3p + 5)·T` pair moments, 526 against 382, +38%), and the live
moments the same again once. The sides are kept in one vector, so a
snapshot copies them in one allocation: kept as six families of vectors,
the copies cost the windowed case 1.92.

| ID | verdict | test | commit |
|---|---|---|---|
| REG-7 | confirmed (a mean at `1e18` took a row at 1, after a gap that forgot it, to 0; `ewridge` predicted 0 where the rows read 1), and wider than the review's `a = 0`: at a history's share of `2^-60` beside a mean of `2^60` a row at 1 went to 0 where the recursion gives 2. Fixed: below a history's share of `2^-52` a mean takes the row from the row back, `x − a·d` (`comp::step_toward`), which is the same number in exact arithmetic and good to a rounding of the result; at `a = 0` it is a fresh accumulator's first step, to the bit. Applied wherever such a mean steps: `EwDiag` (row, kept floor, one slot, the admitted row's view; `sgd`'s x86 lanes take the loop on such a row, as they do not hold the second form), `EwCov` (row and block merge), `EwCovDiag`, `TargetMoments`, `gaps`' `Cross`, `robust`'s target means, `EwAutoCorr`, `ewm_moments` and `hmm`'s factor, which hold to `EwDiag`/`EwCov` to the bit. A row whose history keeps more than `2^-52` of the weight moves no bit (no golden or fixture moved) | `comp.rs::tests::a_step_toward_a_row_that_forgot_the_mean_is_the_row`, `ewdiag.rs::tests::a_history_the_decay_forgot_leaves_the_row_alone`, `ewcov/tests.rs::a_history_the_decay_forgot_leaves_the_row_alone`, `ewridge/tests.rs::a_gap_that_forgot_the_history_fits_the_rows_after_it`, `robust/tests.rs::a_gap_that_forgot_the_history_leaves_the_targets_mean_at_the_row`, `stats.rs::tests::an_autocorrelations_mean_after_a_forgetting_step_is_the_row` | f114897d |
| MARG-1 | confirmed (one target missing in a window of 50 over a series at 1000 after a first row at 0 read `lag_corr_yy` 20.6 against its definition's -0.12), fixed by the user's decision (below, (a)): the lag moments keep, beside each lag, the sides now over the rows a lagged one was taken on and those rows' weights (`ax`, `ayx`, `wx`, `ay`, `wy`, `axy`), stepped on every path the lagged sides are (the row, and the sharded batch), and the read re-centres with `C − e_a·side_b − e_b·side_a + e_a·e_b·w_P`, the definition over the matched rows. New forms at schema 72, frozen in `marginal_window_lags` and a new case whose second target joins late, `marginal_window_lags_joining`; a state saved before them starts them at its load and reads them once its window's boundary is a snapshot taken after it (the kept sets go on to the bit and convert to the current state but for the six sides, `MATCHED_FROM_LOAD`). Cost above | `marginal/tests.rs::a_windowed_lag_correlation_with_an_unmatched_row_is_its_definition`, `marginal/tests.rs::the_lag_moments_add_what_window_lags_says`, `crates/rheon-core/tests/state_fixtures.rs::the_sides_over_the_matched_rows_are_written` | a0e8c850 (test), the MARG-1 commit |
| COV-1 | confirmed (a first row of weight 5 against a `min_weight` of 1: `pc0_var` NaN for the whole cadence), fixed: components whose variances are NaN, refreshed while the unbiased scale was not defined, are refreshed again on the next row. Taking no refresh there instead (the review's fix) broke the recorded rule that the unbiased model's loadings are its biased twin's on every row (`a_component_refreshed_on_one_row_of_weight_recovers_at_the_next`, `the_unbiased_moments_are_the_definition`) | `ewcov/bias_tests.rs::components_wait_for_a_defined_scale` | dd2c1f7f |
| DECO-1 | confirmed (a row of weight 0 at the standardiser's mean and one away from it parted `rho` for good), fixed: a row of weight 0 decays every level's weight and learns nothing, whatever its values | `deco.rs::tests::a_row_of_no_weight_at_the_mean_learns_what_one_away_from_it_does` | dd2c1f7f |
| REFRESH-1 | confirmed (held points saved on a `Datetime` clock resumed on a `Float64` or `Int64` one came out cast; an `Int64` one on a `Datetime` came out as 1970), fixed: a chunk's columns, the clock's kind read from its type among them, are checked before held points are drained, a chunk of no rows included | `tests/test_refresh_time.py::test_points_held_are_resumed_only_on_the_clock_they_were_saved_on` | f114897d |
| MATH-1 | confirmed (`[[-10, 1], [1, -10]]` stopped at iteration 2 with the identity), fixed: each change is relative to the new iterate, as Higham's (4.1), and a change from 0 is not none. The answer, all ones, is each entry's nearest value a correlation can take, at any diagonal weight | `tests/test_corr.py::test_nearest_runs_on_from_a_first_projection_of_nothing` | 3f593edc |
| MATH-2 | confirmed (two negative long-run variances read a finite correlation), fixed: `nan` in the row and column of a column whose long-run variance is not above 0 | `tests/test_corr.py::test_epps_invert_has_no_correlation_where_a_long_run_variance_is_not_above_zero` | 3f593edc |
| COV-3 | confirmed in three parts. (a) `"tied"` and `"diag"` read the ridge off the whole history's accumulators while scoring on the window's: fixed, they read the view's, as `"full"` does (no output moves while (b) stands). (b) Every window's ridge is the whole history's share `s`, where the window's own is `s·W/W_R`: after 394 rows aged out of a window of 6, `"full"` read 0.99 for the class a fresh model on the window's rows reads at 0.50. Fixed by the user's decision, rule 5's thirty-second exception: the window's scale is `s·W/W_R` (`ewcov::window_precision_scale`), and the kept windowed `ew_class` sets go on as the current fixture does (`WINDOW_RIDGE`); the `ew_class_windowed` golden moved by up to `3.5e-4`. (c) `"diag"` read `0/0` for good on a class-constant feature once the ridge faded to 0 (past about 1,075 half-lives): fixed, such a class takes the jitter `"full"`'s factorization ladder takes (`solve::diagonal_jitter`) | `ewclass.rs::tests::a_window_reads_the_ridge_its_own_rows_give`, `::a_class_constant_feature_scores_after_the_ridge_has_faded` | a0e8c850, f114897d, the exceptions' commit |
| MARG-3 | confirmed (`n_kish = inf` and `t = -inf` on noise after a missing target across 538 half-lives), fixed: Kish's size is NaN where the target's squared weights faded to 0 and its weight did not, as the windowed read has it, and so is every statistic read at it | `marginal/tests.rs::a_kish_size_whose_squared_weights_faded_is_unknown` | a0e8c850 |
| REGIME-1 | confirmed (a `monitor` with `permute_every = NaN` built; a window state without its clock since the last draw loaded), fixed: refused under `monitor` and `sequential`, as the bank's spec refuses it, and a state whose clock does not match its cfg's cadence is refused as damaged | `corrchange_clock_tests.rs::permute_every_is_the_windows_and_its_clock_matches_it` | 55ace792 |
| CLUST-1 | confirmed (a half-life of 1 at `beta_mu = 72`: `Tp` 1 and `ξ(∞)` 2; at 0.01, 1), fixed: `Tp` is not rounded up to a whole clock unit, so the horizon is the same in any unit and `ξ(∞) = beta_mu`. DenStream rounds it because its timestamps are integers | `cluster/micro.rs::tests::the_prune_horizon_is_in_the_clocks_own_units` | 92e5517e |
| CLUST-2 | confirmed (a dead cluster re-placed on its own far rows came back at half the weight it was dead at, 0.5, the far rows' 5 dropped), fixed: it takes the far rows' weight | `cluster/kmeans.rs::tests::a_dead_cluster_re_placed_on_its_own_far_rows_takes_their_weight` | 92e5517e |
| MARG-2 | confirmed (each of five damaged states loaded), fixed: refused as damaged where the runs are off under a window, kept without one, of another width, or a snapshot's counts of learned rows are short a target | `marginal/tests.rs::each_part_of_a_state_is_checked_alone` (five new cases) | 55ace792 |
| REFRESH-2 | confirmed: the Rust half by reading (no check of a group's last clock; a Rust check that did not exist), the Python half by test (no order warning). Fixed: a last clock that is NaN, infinite, of the other kind, or with no kind recorded is refused as damaged; `rp.stream.refresh_time` warns of an order hazard and of a spent single-use source, names itself in `explain`, and is not pure where it saves a state, as `merge` and `with_windows` are. The held row's columns are checked where it is used, against the input's (`Held::frame`), which already refuses them by name | `crates/rheon-polars/src/refresh.rs::tests::a_state_whose_last_clock_is_damaged_is_refused`, `tests/test_refresh_time.py::test_refresh_time_checks_its_plan_as_a_bank_does` | f114897d |
| MATH-5 | confirmed on all three, fixed: `z_mse` refuses an `n` that is not a number above 3 (`True` was read as 1, and `n = 2` gave a negative loss); `qlike` is `nan` unless both matrices are positive definite (`-I` scored -4 against `I`); `minvar` is `nan` for a forecast that is not positive definite or a realized matrix with a negative eigenvalue, and refuses a `mu` of the wrong length by name. `test_corr_paths.py`'s coverage test had scored an indefinite forecast (20/9); its zero-normaliser case now uses `mu = 0` | `tests/test_corr.py::test_each_loss_reads_only_what_defines_it`, `tests/test_corr_paths.py::test_a_minimum_variance_weight_with_a_zero_normaliser_has_no_loss` | 3f593edc |
| COV-4 | confirmed (`n/(n − 1)` NaN after a step of no weight across 538 half-lives; the review's `w < 1.5e-154` half was REG-12's, fixed in batch g), fixed: the Kish sum's power of two takes a decay that would take the sum below the normal doubles, down to the scale of the least normal weight, and a subnormal weight counts as 0 rows (a weight stuck at the least subnormal read 1e292 rows against a sum kept at its scale). `rls` shares the sum. Two `readiness.rs` tests and `rls`'s had pinned Kish's size at 0 across 5,000 rows of no weight, an artefact of `Σw²` fading first: they now hold the size across the decay and reach 0 once the weight is subnormal | `ewcov/tests.rs::the_unbiased_factor_holds_where_the_squared_weights_underflow`, `rls.rs::tests::a_squared_weight_sum_faded_to_0_reads_no_kish_size`, `crates/rheon-core/tests/readiness.rs::lasso_reads_infinite_where_the_gram_has_no_weight`, `::ewridge_coef_variance_is_none_where_the_gram_has_no_weight` | dd2c1f7f, f114897d |
| COV-2 | confirmed (`EwCovDiag::from_moments` at a weight of 100 kept a squared weight of 0), fixed in both: a weight above 0 given without its squares has none known (`None`), as a state from before they were kept | `ewcov/tests.rs::moments_given_at_a_weight_have_no_kish_size` | dd2c1f7f |
| COV-5 | (1) confirmed (a row at `1e80` after rows near `1e-100`: `u` NaN in `deco`, the row out of `corrchange`'s scalar CUSUM), fixed: the equicorrelation, a ratio of the squares' degree, is read from each overflowed block's sums at a power of two of their own; `deco`'s level does not learn from such a row, whose sums are past the doubles. Not over a variance below the normal doubles, which a history the decay all but forgot leaves with its digits gone: there the row keeps no estimate, as the step with no history at all (`model_contract`'s `corrchange_predict_is_the_step` holds the two alike). (2) batch i's STATE-2 (1dd7d879) caps the four reservations; not repeated here. (3) confirmed (stretches of 3 returns at `m = 2` read otherwise backwards; 40 do not), fixed by the user's decision, rule 5's thirty-third exception: the lead is held until the stretch has `2m` returns, a stretch of `2m − 1` closes with its one jittered return (`Rcov::fused`), and a shorter one has none; a kept state inside such a stretch closes it at its load (no kept set holds one) | `deco.rs::tests::a_row_far_past_its_spread_has_an_estimate`, `corrchange_tests.rs::a_row_far_past_its_spread_is_scalarised`, `rcov.rs::tests::a_short_stretch_reads_the_same_backwards` | 2d8a0ef0, the exceptions' commit |

### Decided by the user (batch h, 2026-11-24)

The three below went to the user, who chose: MARG-1 (a), exact, as new
forms at schema 72; COV-3 (b) as rule 5's thirty-second exception and
COV-5 (3) as its thirty-third (`docs/records/STATE-CONTINUATION.md`; "No
concern for continuing differently here"). The options as they were put:

- **MARG-1, a windowed `marginal`'s lag moments with unmatched rows.**
  Under a window the lag moments are raw about the series' first value and
  re-centred at the read by `C − e_a·side_b`, exact only where every row in
  the window has its lagged partner. A row without one (a target absent
  `ℓ` rows back, the first rows after a session break) leaves `−e_b ·
  Σ_unmatched w (a − m_a)/W` behind, which grows with the series' drift
  from its origin. Options: (a) keep, per lag, the now-side sums over the
  matched rows and the matched weights, for the rows whose `x` partner
  exists and those whose `y` partner exists too, and re-centre exactly:
  `p·T` more doubles per lag for `cxx`'s side, `p·T` per cross lag for
  `cxy`'s, and `3T` per lag, in the live moments and in every window
  snapshot (about half again of today's `4·p·T + 2T` per crossed lag), and
  a multiply-add more per pair per lag per row; a schema-72 form, with
  fixtures; (b) read all four lag statistics of a pair over the rows whose
  `y` partner exists, which needs only `p·T + 2T` per lag but moves
  `lag_corr_xx` on any stream with a missing target, windowed or not; (c)
  document the bias. Recommendation: (a), the definition at the cost the
  window's other moments already pay; it needs the user's word on the
  state growth.
- **COV-3 (b), the window's ridge.** A window's precision prior is the
  whole history's share `s` (the prior read as one row at the stream's
  start), read against the window's co-moments: its weight in sum form is
  `s·W`, so over the window's weight it is `s·W/W_R`, which with no decay
  is a fresh model's on the window's rows (the test holds that). Built and
  backed out: it moves the continuation of every saved windowed `ew_class`
  state, `v61` to `v71` and the current set, by up to `1e-3` of a
  probability (and the `ew_class_windowed` golden by `3.5e-4`), which needs
  rule 5's next exception. The review's `q_R/w_R²` (one row of the window's
  Kish size) agrees with no decay and not under one, and jumps where the
  window first lets a row go. Recommendation: `s·W/W_R`, with the
  exception ("windowed `ew_class` and `ew_cov` states continue with the
  window's ridge").
- **COV-5 (3), `rcov`'s kernel estimate on short stretches.** A stretch of
  fewer than `2m` returns emits its leading jittered return at its `m`-th
  return and then cannot close its trailing one (the module doc records
  that the lead cannot be un-emitted), so the estimate is not the same
  read backwards. Options: (a) hold the lead until the stretch reaches
  `2m − 1` returns and emit one jittered return there (`mean(X_{n−m+1..n}) −
  mean(X_0..X_{m−1})`), which moves the estimate on stretches cut short by
  a break and the continuation of a kernel state saved mid-stretch; (b)
  keep, as documented. Recommendation: (a), behind rule 5's word for the
  states saved inside a short stretch.
