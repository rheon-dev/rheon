use super::*;

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
        // `kalman`'s fixed-lag smoother (schema 72, before any release).
        ("bank_smoothed", "a smoother slot's gain", smoothed_gain),
        ("bank_smoothed", "a smoother's clock label", smoothed_label),
        ("bank_smoothed", "a row the gate held", smoothed_held),
        (
            "bank_smoothed",
            "the last row's smoothed fields",
            smoothed_last,
        ),
        // Embargoed `smooth_lag` list-typed `LastRow` fields (kalman + hmm).
        (
            "bank_smoothed_embargo",
            "last row's smoothed_coef_list",
            smoothed_coef_list_last,
        ),
        (
            "bank_smoothed_embargo",
            "last row's smoothed_se_list",
            smoothed_se_list_last,
        ),
        (
            "bank_smoothed_embargo",
            "last row's smoothed_state_list",
            smoothed_state_list_last,
        ),
        (
            "bank_smoothed_embargo",
            "last row's smoothed_for_list",
            smoothed_for_list_last,
        ),
        // `hmm` spherical / smoother / conjugate priors and `bocpd` NB + known variance.
        ("bank_hmm_bocpd_ext", "hmm spherical states", |p, v| {
            p.contains(".spherical_states.") && p.ends_with(".var") && v.as_f64().is_some()
        }),
        ("bank_hmm_bocpd_ext", "hmm smoother ring", |p, v| {
            p.contains(".smoother.fring.") && v.as_f64().is_some()
        }),
        ("bank_hmm_bocpd_ext", "hmm smoother label", |p, v| {
            p.contains(".smoother.labels.") && p.ends_with(".F64") && v.as_f64().is_some()
        }),
        ("bank_hmm_bocpd_ext", "hmm conjugate data_w", |p, v| {
            p.contains(".extras.data_w.") && v.as_f64().is_some_and(|f| f > 0.0)
        }),
        ("bank_hmm_bocpd_ext", "bocpd nb_hazard", |p, v| {
            p.contains(".nb_hazard.") && v.as_u64() == Some(3)
        }),
        ("bank_hmm_bocpd_ext", "bocpd obs_var", |p, v| {
            p.contains(".obs_var.") && v.as_f64() == Some(0.5)
        }),
        // Clocked-embargo calibration HAC (schema 72, before any release).
        (
            "bank_calibration_clock_embargo",
            "Calibration clock_hac bread",
            |p, v| {
                p.contains(".calibration.")
                    && p.contains(".clock_hac.bread.")
                    && v.as_f64().is_some_and(|x| x > 0.0)
            },
        ),
        (
            "bank_calibration_clock_embargo",
            "Calibration clock_hac temporal bandwidth_ns",
            |p, v| {
                p.contains(".calibration.") && p.ends_with(".clock_hac.bandwidth_ns") && !v.is_nil()
            },
        ),
        (
            "bank_calibration_clock_embargo",
            "Calibration clock_hac temporal Stamp::Ns",
            |p, _| {
                p.contains(".calibration.")
                    && p.contains(".clock_hac.ring.")
                    && p.ends_with(".stamp.Ns")
            },
        ),
        (
            "bank_calibration_clock_embargo",
            "Coverage clock_hac ring",
            |p, v| {
                p.contains(".coverage.")
                    && p.contains(".clock_hac.ring.")
                    && p.ends_with(".u")
                    && v.as_f64().is_some_and(f64::is_finite)
            },
        ),
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
        // Windows state 15 (task 78): a row-only run carries an empty
        // clock-only core with zero outputs and populated group clocks.
        (
            "with_windows_row_only",
            "a clock-only core's zero outputs",
            |p, v| p == ".core.n_outputs" && v.as_u64() == Some(0),
        ),
        (
            "with_windows_row_only",
            "a clock-only core's group policy time",
            |p, v| {
                p.starts_with(".core.groups.")
                    && p.ends_with(".tau")
                    && v.as_f64().is_some_and(|f| f > 0.0)
            },
        ),
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
        // Windows state 14 (PLAN 490): the STL operators and each form a
        // group's ring takes at the save.
        ("with_windows_stl", "an STL operator's window", |p, v| {
            p.contains(".seasonals.") && p.ends_with(".window_rows") && v.as_u64() == Some(6)
        }),
        ("with_windows_stl", "a full ring", |p, v| {
            p.contains(".seasonal.") && p.ends_with(".count") && v.as_u64() == Some(6)
        }),
        (
            "with_windows_stl",
            "a ring wrapped past its window",
            |p, v| p.contains(".seasonal.") && p.ends_with(".head") && v.as_u64() == Some(2),
        ),
        ("with_windows_stl", "a filling ring", |p, v| {
            p.contains(".seasonal.") && p.ends_with(".count") && v.as_u64() == Some(3)
        }),
        ("with_windows_stl", "an empty or emptied ring", |p, v| {
            p.contains(".seasonal.") && p.ends_with(".count") && v.as_u64() == Some(0)
        }),
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
        // The kinds since 2.0 no bank file held (review 2026-11-23, CI-5):
        // a started `garch`'s fit beside a start still buffering its rows,
        // a `hawkes` window's stamps, and `kalman`'s square-root factor.
        ("bank_since_2_0", "a started garch", |p, v| {
            p.ends_with(".Garch.series.0.started") && v.as_bool() == Some(true)
        }),
        ("bank_since_2_0", "a garch start's rows", |p, _| {
            p.contains(".Garch.series.0.buffer.")
        }),
        ("bank_since_2_0", "a hawkes window's stamps", |p, _| {
            p.contains(".Hawkes.window.stamp.")
        }),
        ("bank_since_2_0", "kalman's square-root form", |p, v| {
            p.ends_with(".Kalman.covariance_form") && v.as_str() == Some("square_root")
        }),
        // PLAN 509: `audit` HLL++ plain, row-windowed, and clock-windowed.
        ("bank_audit_hll", "a plain HLL++ sketch", |p, _| {
            p.contains(".Audit.hll.sketches.Plain.")
        }),
        ("bank_audit_hll", "a row-windowed sliding HLL++", |p, _| {
            p.contains(".Audit.hll.sketches.SlidingRows.cols.0.entries.")
        }),
        (
            "bank_audit_hll",
            "a clock-windowed sliding HLL++",
            |p, _| p.contains(".Audit.hll.sketches.SlidingClock.cols.0.entries."),
        ),
        (
            "bank_kalman_nonlinear",
            "kalman's EKF Poisson obs",
            |p, v| p.ends_with(".Kalman.cfg.obs.filter") && v.as_str() == Some("ekf"),
        ),
        (
            "bank_kalman_nonlinear",
            "kalman's UKF scalar_index obs",
            |p, v| p.ends_with(".Kalman.cfg.obs.ukf_rule") && v.as_str() == Some("scalar_index"),
        ),
        ("bank_kalman_nonlinear", "kalman's UKF merwe obs", |p, v| {
            p.ends_with(".Kalman.cfg.obs.ukf_rule") && v.as_str() == Some("merwe")
        }),
        (
            "bank_kalman_nonlinear",
            "kalman's UKF plugin noise",
            |p, v| p.ends_with(".Kalman.cfg.obs.ukf_noise") && v.as_str() == Some("plugin"),
        ),
        ("bank_kalman_extras", "kalman's coef_prior", |p, v| {
            p.contains(".Kalman.cfg.coef_prior.") && v.as_f64().is_some()
        }),
        ("bank_kalman_extras", "kalman's emit_loglik", |p, v| {
            p.ends_with(".Kalman.cfg.emit_loglik") && v.as_bool() == Some(true)
        }),
        ("bank_kalman_extras", "kalman's emit_mahal", |p, v| {
            p.ends_with(".Kalman.cfg.emit_mahal") && v.as_bool() == Some(true)
        }),
        ("bank_krls", "krls ALD dictionary atoms", |p, v| {
            p.contains(".Krls.mode.Ald.atoms.") && v.as_f64().is_some()
        }),
        ("bank_krls", "krls window Cholesky factor", |p, v| {
            p.contains(".Krls.mode.Window.l_w.") && v.as_f64().is_some_and(|x| x > 0.0)
        }),
        ("bank_krls", "krls clock window stamps", |p, _| {
            p.contains(".Krls.mode.Window.stamp.")
        }),
        // PLAN 525: kmeans fixed centers + inertia, ew_class opts, holt known +
        // horizons, corrchange holm + n_flagged, and standalone feature KS.
        ("bank_plan525", "kmeans EW inertia state", |p, v| {
            p.ends_with(".KMeans.inertia.w") && v.as_f64().is_some_and(|w| w > 0.0)
        }),
        (
            "bank_plan525",
            "ew_class fixed priors and reg_param",
            |p, v| p.ends_with(".EwClass.opts.reg_param") && v.as_f64() == Some(0.15),
        ),
        ("bank_plan525", "ew_class var_smoothing", |p, v| {
            p.ends_with(".EwClass.opts.var_smoothing") && v.as_f64() == Some(0.01)
        }),
        (
            "bank_plan525",
            "corrchange holm + emit_n_flagged",
            |p, v| p.ends_with(".CorrChange.cfg.emit_n_flagged") && v.as_bool() == Some(true),
        ),
        ("bank_plan525", "standalone feature KS window", |p, v| {
            p.ends_with(".checks.0.health_ks.window") && v.as_u64() == Some(16)
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

/// Windows state 15 (task 78): the `with_windows_row_only` fixture freezes
/// an empty clock-only core -- zero kernels, zero operators, zero members and
/// zero group queues -- instead of the legacy one-row placeholder sum kernel.
#[test]
fn the_row_only_fixture_freezes_an_empty_clock_only_core() {
    if regenerating() {
        return;
    }
    let f = frozen::ALL
        .iter()
        .find(|f| f.name == "with_windows_row_only")
        .expect("the with_windows_row_only fixture");
    let v = rmpv::decode::read_value(&mut unhex(f.state).as_slice()).unwrap();
    let core = v
        .as_map()
        .unwrap()
        .iter()
        .find(|(k, _)| k.as_str() == Some("core"))
        .unwrap()
        .1
        .as_map()
        .unwrap();
    let get = |name: &str| {
        &core
            .iter()
            .find(|(k, _)| k.as_str() == Some(name))
            .unwrap()
            .1
    };
    assert!(get("kernels").as_array().unwrap().is_empty(), "0 kernels");
    assert!(get("ops").as_array().unwrap().is_empty(), "0 ops");
    assert!(get("members").as_array().unwrap().is_empty(), "0 members");
    assert_eq!(get("n_outputs").as_u64(), Some(0));
    let groups = get("groups").as_array().unwrap();
    assert!(groups.len() >= 2, "multiple groups tracked");
    for g in groups {
        let gm = g.as_map().unwrap();
        let queues = gm
            .iter()
            .find(|(k, _)| k.as_str() == Some("queues"))
            .unwrap()
            .1
            .as_array()
            .unwrap();
        assert!(queues.is_empty(), "no placeholder queue in any group");
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

pub(super) fn dir() -> PathBuf {
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
        include_str!("../../build.rs") == include_str!("../../../rheon-core/build.rs"),
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

/// Specs of `bank_calibration_clock_embargo`: numeric and temporal clocked
/// `embargo` with `emit_calibration=true` and no explicit `horizon_rows`
/// (exercising both `Calibration::clock_hac` and `Coverage::clock_hac`, plus
/// `Stamp::Raw` and `Stamp::Ns`).
pub(super) fn calibration_clock_embargo_specs() -> Vec<Spec> {
    [
        r#"{"name": "cal_num", "model": {"type": "ewridge"}, "targets": ["y"], "features": ["x0", "x1"], "clock": "t", "half_life": 10.0, "gap_cap": 50.0, "weight": "w", "min_weight": 0.0, "min_settled_frac": 0.0, "max_error_inflation": "inf", "embargo": 3.0, "emit_calibration": true, "max_state_bytes": 65536}"#,
        r#"{"name": "cal_ts", "model": {"type": "ewridge"}, "targets": ["y"], "features": ["x0", "x1"], "clock": "ts", "half_life": "10s", "gap_cap": "1m", "weight": "w", "min_weight": 0.0, "min_settled_frac": 0.0, "max_error_inflation": "inf", "embargo": "3s", "emit_calibration": true, "max_state_bytes": 65536}"#,
        r#"{"name": "cov_num", "model": {"type": "quantile", "quantile": 0.8}, "targets": ["y"], "features": ["x0", "x1"], "clock": "t", "half_life": 10.0, "gap_cap": 50.0, "weight": "w", "min_weight": 0.0, "min_settled_frac": 0.0, "embargo": 3.0, "emit_calibration": true, "max_state_bytes": 65536}"#,
    ]
    .iter()
    .map(|text| serde_json::from_str(text).unwrap_or_else(|e| panic!("{text}: {e}")))
    .collect()
}

pub(super) fn calibration_clock_embargo_first() -> (Vec<u8>, DataFrame) {
    let df = frame();
    let mut bank = Bank::new(calibration_clock_embargo_specs()).unwrap();
    bank.fit_predict(&df.slice(0, SPLIT)).unwrap();
    (
        bank.save_bytes().unwrap(),
        df.slice(SPLIT as i64, N - SPLIT),
    )
}

pub(super) fn calibration_clock_embargo_run(bytes: &[u8], input: &DataFrame) -> Ran {
    bank_run_of(&calibration_clock_embargo_specs(), bytes, input)
}
