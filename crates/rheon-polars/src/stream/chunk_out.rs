//! A stream's output buffers for one run of rows ([`ChunkOut`]), moved out
//! of `stream.rs`, which had passed the size a source file is held to
//! (task 270, G-1).

use super::*;
use std::mem::size_of;

/// One row of a `List(List(Float64))` smoothed output (`None` for null,
/// `Some(emissions)` for one or more emitting releases in release order).
pub type NestedListRow = Option<Vec<Vec<f64>>>;

/// One row of a `List(<clock>)` smoothed label output (`None` for null,
/// `Some(labels)` parallel to the row's smoothed lists).
pub type ClockListRow = Option<Vec<Option<ClockValue>>>;

/// Flat output buffers for one (stream, chunk) task (docs/records/PERFORMANCE-LOG.md P1).
///
/// One allocation per output *column* for the whole chunk, rather than the
/// ~11 `Vec`s per row the previous `RowOut` needed. Every numeric buffer is
/// `n_slots * n_rows`, slot-major, so a slot's values are contiguous and the
/// scatter into the final column is a straight walk. NaN is null.
///
/// `processed` distinguishes "this row was skipped" from "this row produced a
/// NaN", which matters only for `drift` (a bool, with no NaN to spare) and for
/// `n_eff`, which is otherwise always finite.
pub struct ChunkOut {
    /// Absolute row indices this task wrote, in order.
    pub rows: Vec<usize>,
    /// Per row: false = skipped, every output is null.
    pub processed: Vec<bool>,
    /// `n_models * n_slots * n_rows` unless noted; NaN = null.
    pub pred: Vec<f64>,
    pub resid: Vec<f64>,
    pub sigma: Vec<f64>,
    pub zscore: Vec<f64>,
    pub autocorr: Vec<f64>,
    /// `(ic, r2, hit_rate)`, model-major: `n_models * 3 * n_slots * n_rows`.
    pub metrics: Vec<f64>,
    /// Point-loss scores (`mae`, `pinball*`, `log_loss`), model-major:
    /// `n_models * n_scores * n_slots * n_rows`.
    pub scores: Vec<f64>,
    /// Interval scores (`width`, `winkler` per conformal level), model-major:
    /// `n_models * (2 * n_conf_levels) * n_slots * n_rows`.
    pub interval_scores: Vec<f64>,
    /// `(lo, hi, coverage)` per conformal level, model-major:
    /// `n_models * (3 * n_conf_levels) * n_slots * n_rows`.
    pub conformal: Vec<f64>,
    /// Standalone per-row conformal p-value: `n_models * n_slots * n_rows`.
    pub conformal_p: Vec<f64>,
    /// Task 221's values, `n_checks` per slot, laid out like `metrics`:
    /// `n_models * n_checks * n_slots * n_rows` ([`crate::checks`]).
    pub checks: Vec<f64>,
    /// Feature health under `emit_feature_health`, per instance and
    /// feature, not per slot: `n_models * 2 * k * n_rows`, the spread
    /// ratios' block and then the mean shifts' (task 221 (g)).
    pub health: Vec<f64>,
    /// Windowed two-sample KS feature health under `feature_ks_window_rows` /
    /// `feature_ks_recent_rows`, per instance and feature: `n_models * 2 * k * n_rows`,
    /// the statistics' block and then the p-values'.
    pub health_ks: Vec<f64>,
    /// Model-major: `n_models * n_levels * n_slots * n_rows`.
    pub resid_q: Vec<f64>,
    pub drift: Vec<bool>,
    /// `n_models * n_rows`.
    pub n_eff: Vec<f64>,
    /// `n_models * n_targets * n_rows`: lasso's selected penalty, or
    /// `huber`'s residual scale under `emit_huber_scale`; empty otherwise.
    pub lam_selected: Vec<f64>,
    /// `n_models * n_targets * n_rows` under `emit_target_weight`, else
    /// empty: the weight each target's `min_weight` gate read on the row
    /// (docs/PLAN.md task 568).
    pub target_weight: Vec<f64>,
    /// Emitted on a cadence rather than every row, so it stays boxed:
    /// `[model][row]`.
    pub coef: Vec<Vec<Option<Vec<f64>>>>,
    /// `n_models * n_rows`: how settled the instance was before the row
    /// (docs/WARMUP-AND-CONVERGENCE.md §3); NaN = null, where nothing decays.
    pub settled: Vec<f64>,
    /// `n_models * n_rows`: why the row's predictions were withheld, as an
    /// index into [`WITHHELD_REASONS`] plus one; 0 = none (null).
    pub reason: Vec<u8>,
    /// `n_models * n_slots * n_rows` under `emit_error_inflation`, else
    /// empty: the row's own error inflation per slot.
    pub inflation: Vec<f64>,
    /// Each coefficient's data share, on `coef`'s cadence: `[model][row]`.
    pub support_coef: Vec<Vec<Option<Vec<f64>>>>,
    /// Each coefficient's standard error under `emit_se_coef`, on `coef`'s
    /// cadence: `[model][row]` (docs/PLAN.md task 116, F).
    pub se_coef: Vec<Vec<Option<Vec<f64>>>>,
    /// The robust standard errors under `emit_robust_se`, HC0..HC3 and Newey
    /// and West's, laid out like `se_coef`.
    pub se_coef_hc0: Vec<Vec<Option<Vec<f64>>>>,
    pub se_coef_hc1: Vec<Vec<Option<Vec<f64>>>>,
    pub se_coef_hc2: Vec<Vec<Option<Vec<f64>>>>,
    pub se_coef_hc3: Vec<Vec<Option<Vec<f64>>>>,
    pub se_coef_hac: Vec<Vec<Option<Vec<f64>>>>,
    /// Per-coefficient DFBETAS under `emit_dfbetas`, laid out like `coef`,
    /// written on every scored row (empty when `emit_dfbetas` is false).
    pub dfbetas: Vec<Vec<Option<Vec<f64>>>>,
    /// `n_rows` each under `emit_clocks`, else empty: the row's own clock,
    /// and the newest learned row's when it was scored (task 152).
    pub scored_clock: Vec<Option<ClockValue>>,
    pub learned_clock: Vec<Option<ClockValue>>,
    /// `kalman`'s fixed-lag smoother under `smooth_lag` without delayed
    /// learning, else empty: the smoothed coefficients and (under
    /// `emit_smoothed_se`) their standard errors, `[model][row]` on the rows
    /// that complete a lag, and the clock label of the row each target's value
    /// describes, `n_models * smoothed_for_slots * n_rows`.
    pub smoothed_coef: Vec<Vec<Option<Vec<f64>>>>,
    pub smoothed_se: Vec<Vec<Option<Vec<f64>>>>,
    pub smoothed_for: Vec<Option<ClockValue>>,
    /// Fixed-lag smoother outputs under delayed learning (`embargo`,
    /// `label_clock`, or a formula target), else empty: each arriving row can
    /// release zero, one, or many held rows, so each row holds `None` (zero
    /// emissions) or `Some(vec)` with one entry per emitting release in
    /// release order.
    pub smoothed_coef_list: Vec<Vec<NestedListRow>>,
    pub smoothed_se_list: Vec<Vec<NestedListRow>>,
    pub smoothed_state_list: Vec<Vec<Option<Vec<f64>>>>,
    pub smoothed_for_list: Vec<Vec<ClockListRow>>,
    /// Slot counts this layout was built for.
    pub n_models: usize,
    pub n_slots: usize,
    pub n_levels: usize,
    pub n_conf_levels: usize,
    pub n_scores: usize,
    pub n_checks: usize,
}

/// Which optional buffers a spec fills: read by `ChunkOut::new` to size them
/// and by `ChunkOut::run_rows` to count the values a row writes.
pub(super) struct Buffers {
    pub(super) residuals: bool,
    pub(super) extras: bool,
    n_levels: usize,
    /// Values per row from the model's step extras: one per target for
    /// lasso's selected penalty or `huber`'s scale under
    /// `emit_huber_scale`, and `vector_ar`'s tests and order.
    extra_width: usize,
}

impl Buffers {
    pub(super) fn of(spec: &Spec) -> Self {
        Self {
            // A model with no target to predict has no residual field, so its
            // `resid` buffer would be filled, written once per slot per row
            // and never read: for a 230-statistic `ew_cov` that was a fifth
            // of the row (docs/records/PERFORMANCE-LOG.md §13).
            residuals: !spec.model.predicts_no_target(),
            // `sigma` is also the loss that `emit_selected` and
            // `emit_averaged` rank slots by (E13/E14 reuse E12's tracked
            // error), so it has to be materialized for them even when it is
            // not itself an output field.
            extras: spec.emit_sigma || spec.emit_zscore || spec.emit_selected || spec.emit_averaged,
            n_levels: spec.resid_quantiles.as_ref().map_or(0, Vec::len),
            extra_width: crate::bank::extra_width(spec),
        }
    }

    /// Whether every row a stream processes writes every slot of the
    /// buffers [`ChunkOut::renew_run`] leaves unfilled: a spec with no
    /// `embargo`, no `label_clock` and no formula target schedules each
    /// accepted row as one that emits, where any of the three replays held
    /// rows that do not (`Stream::apply_label_delay` returns the plans as
    /// they are for a spec with none).
    pub(super) fn dense(spec: &Spec) -> bool {
        spec.embargo.is_none() && spec.label_clock.is_none() && !spec.targets.any_formula()
    }

    /// `f64` values one row writes across every buffer, for `n_models`
    /// instances of `n_slots` slots.
    fn per_row(&self, spec: &Spec, n_models: usize, n_slots: usize) -> usize {
        let per = n_models * n_slots;
        let on = |flag: bool| if flag { per } else { 0 };
        let n_conf = spec.conformal_levels().len();
        let n_scores = spec.score_block_count();
        per + on(self.residuals)
            + 2 * on(self.extras)
            + on(spec.emit_autocorr)
            + 3 * on(spec.emit_metrics.has_legacy())
            + n_scores * per
            + if spec.emit_metrics.has_interval_scores() {
                2 * n_conf * per
            } else {
                0
            }
            + 3 * n_conf * per
            + on(spec.emit_conformal_p)
            + crate::checks::value_names(spec).len() * per
            + if spec.emit_feature_health {
                n_models * 2 * spec.k()
            } else {
                0
            }
            + if spec.has_feature_ks() {
                n_models * 2 * spec.k()
            } else {
                0
            }
            + self.n_levels * per
            + n_models
            + n_models
            + on(spec.emit_error_inflation)
            + if spec.emit_target_weight {
                n_models * spec.m()
            } else {
                0
            }
            + n_models * self.extra_width
    }
}

impl ChunkOut {
    /// Buffers for `n_rows` rows of one stream, all null until written.
    pub fn new(spec: &Spec, n_models: usize, n_slots: usize, n_rows: usize) -> Self {
        let mut out = Self::empty();
        out.renew(spec, n_models, n_slots, n_rows);
        out
    }

    /// No buffers at all, for [`Self::renew`] to size.
    fn empty() -> Self {
        Self {
            rows: Vec::new(),
            processed: Vec::new(),
            pred: Vec::new(),
            resid: Vec::new(),
            sigma: Vec::new(),
            zscore: Vec::new(),
            autocorr: Vec::new(),
            metrics: Vec::new(),
            scores: Vec::new(),
            interval_scores: Vec::new(),
            conformal: Vec::new(),
            conformal_p: Vec::new(),
            checks: Vec::new(),
            health: Vec::new(),
            health_ks: Vec::new(),
            resid_q: Vec::new(),
            drift: Vec::new(),
            n_eff: Vec::new(),
            lam_selected: Vec::new(),
            target_weight: Vec::new(),
            coef: Vec::new(),
            settled: Vec::new(),
            reason: Vec::new(),
            inflation: Vec::new(),
            support_coef: Vec::new(),
            se_coef: Vec::new(),
            se_coef_hc0: Vec::new(),
            se_coef_hc1: Vec::new(),
            se_coef_hc2: Vec::new(),
            se_coef_hc3: Vec::new(),
            se_coef_hac: Vec::new(),
            dfbetas: Vec::new(),
            scored_clock: Vec::new(),
            learned_clock: Vec::new(),
            smoothed_coef: Vec::new(),
            smoothed_se: Vec::new(),
            smoothed_for: Vec::new(),
            smoothed_coef_list: Vec::new(),
            smoothed_se_list: Vec::new(),
            smoothed_state_list: Vec::new(),
            smoothed_for_list: Vec::new(),
            n_models: 0,
            n_slots: 0,
            n_levels: 0,
            n_conf_levels: 0,
            n_scores: 0,
            n_checks: 0,
        }
    }

    /// [`Self::new`]'s buffers, in the memory these already hold: every
    /// buffer cleared and filled as `new` fills it, allocating only where
    /// one has less room than it needs. `new` is this on empty buffers, so
    /// the two cannot differ. A bank of many small groups makes a run's
    /// buffers for every group of every chunk, and allocating and freeing
    /// them was the largest share of such a task (task 270, G-1).
    pub fn renew(&mut self, spec: &Spec, n_models: usize, n_slots: usize, n_rows: usize) {
        self.renew_with(spec, n_models, n_slots, n_rows, false);
    }

    /// [`Self::renew`] for a run that `Stream::process_chunk` is about to
    /// fill: where every row the run learns or scores writes every slot of
    /// the buffers a row writes in full (`pred`, `resid`, `sigma`,
    /// `zscore`, `n_eff`, `settled`, `reason`), those are left as they were,
    /// not filled with NaN again. A row the run does not process leaves its
    /// values unread -- every reader of a run asks `processed` first -- and
    /// the others are filled as `renew` fills them. Whether a spec's rows
    /// all write is [`Buffers::dense`]; for any other the two are the same.
    pub fn renew_run(&mut self, spec: &Spec, n_models: usize, n_slots: usize, n_rows: usize) {
        self.renew_with(spec, n_models, n_slots, n_rows, Self::keeps_buffers(spec));
    }

    /// Whether [`Self::renew_run`] leaves a spec's buffers as they were
    /// ([`Buffers::dense`]).
    pub(crate) fn keeps_buffers(spec: &Spec) -> bool {
        Buffers::dense(spec)
    }

    fn renew_with(
        &mut self,
        spec: &Spec,
        n_models: usize,
        n_slots: usize,
        n_rows: usize,
        dense: bool,
    ) {
        fn fill<T: Clone>(v: &mut Vec<T>, n: usize, x: T) {
            v.clear();
            v.resize(n, x);
        }
        // A buffer the run writes in full where it processes a row: the
        // length is set, the old values stay (or, where the buffer is new
        // or longer, are `x`).
        fn written<T: Clone>(dense: bool, v: &mut Vec<T>, n: usize, x: T) {
            if dense {
                v.truncate(n);
                v.resize(n, x);
            } else {
                fill(v, n, x);
            }
        }
        fn lists<T: Clone>(v: &mut Vec<Vec<Option<T>>>, n_outer: usize, n_rows: usize) {
            v.truncate(n_outer);
            for inner in v.iter_mut() {
                fill(inner, n_rows, None);
            }
            v.resize_with(n_outer, || vec![None; n_rows]);
        }
        let per = n_models * n_slots * n_rows;
        let on = |flag: bool| if flag { per } else { 0 };
        let Buffers {
            residuals,
            extras,
            n_levels,
            extra_width,
        } = Buffers::of(spec);
        let n_checks = crate::checks::value_names(spec).len();
        self.rows.clear();
        self.rows.reserve(n_rows);
        fill(&mut self.processed, n_rows, false);
        written(dense, &mut self.pred, per, f64::NAN);
        written(dense, &mut self.resid, on(residuals), f64::NAN);
        written(dense, &mut self.sigma, on(extras), f64::NAN);
        written(dense, &mut self.zscore, on(extras), f64::NAN);
        fill(&mut self.autocorr, on(spec.emit_autocorr), f64::NAN);
        let n_conf_levels = spec.conformal_levels().len();
        let n_scores = spec.score_block_count();
        fill(
            &mut self.metrics,
            3 * on(spec.emit_metrics.has_legacy()),
            f64::NAN,
        );
        fill(&mut self.scores, n_scores * per, f64::NAN);
        fill(
            &mut self.interval_scores,
            if spec.emit_metrics.has_interval_scores() {
                2 * n_conf_levels * per
            } else {
                0
            },
            f64::NAN,
        );
        fill(&mut self.conformal, 3 * n_conf_levels * per, f64::NAN);
        fill(&mut self.conformal_p, on(spec.emit_conformal_p), f64::NAN);
        fill(&mut self.checks, n_checks * per, f64::NAN);
        let health = if spec.emit_feature_health {
            n_models * 2 * spec.k() * n_rows
        } else {
            0
        };
        fill(&mut self.health, health, f64::NAN);
        let health_ks = if spec.has_feature_ks() {
            n_models * 2 * spec.k() * n_rows
        } else {
            0
        };
        fill(&mut self.health_ks, health_ks, f64::NAN);
        fill(&mut self.resid_q, n_levels * per, f64::NAN);
        fill(&mut self.drift, on(spec.emit_drift), false);
        written(dense, &mut self.n_eff, n_models * n_rows, f64::NAN);
        let lam = n_models * extra_width * n_rows;
        fill(&mut self.lam_selected, lam, f64::NAN);
        let target_weight = if spec.emit_target_weight {
            n_models * spec.m() * n_rows
        } else {
            0
        };
        fill(&mut self.target_weight, target_weight, f64::NAN);
        lists(&mut self.coef, n_models, n_rows);
        written(dense, &mut self.settled, n_models * n_rows, f64::NAN);
        written(dense, &mut self.reason, n_models * n_rows, 0);
        fill(&mut self.inflation, on(spec.emit_error_inflation), f64::NAN);
        lists(&mut self.support_coef, n_models, n_rows);
        lists(&mut self.se_coef, n_models, n_rows);
        lists(&mut self.se_coef_hc0, n_models, n_rows);
        lists(
            &mut self.se_coef_hc1,
            if spec.emits_robust_se_type(rheon_core::RobustSeType::Hc1) {
                n_models
            } else {
                0
            },
            n_rows,
        );
        lists(
            &mut self.se_coef_hc2,
            if spec.emits_robust_se_type(rheon_core::RobustSeType::Hc2) {
                n_models
            } else {
                0
            },
            n_rows,
        );
        lists(
            &mut self.se_coef_hc3,
            if spec.emits_robust_se_type(rheon_core::RobustSeType::Hc3) {
                n_models
            } else {
                0
            },
            n_rows,
        );
        lists(&mut self.se_coef_hac, n_models, n_rows);
        let dfb = if spec.emit_dfbetas && spec.has_row_error_inflation() {
            n_models
        } else {
            0
        };
        lists(&mut self.dfbetas, dfb, n_rows);
        let clocks = if spec.emit_clocks { n_rows } else { 0 };
        fill(&mut self.scored_clock, clocks, None);
        fill(&mut self.learned_clock, clocks, None);
        let sm_list = spec.smoothed_is_list();
        let kalman_sm = if spec.is_kalman_smoother() {
            n_models
        } else {
            0
        };
        let kalman_flat = if sm_list { 0 } else { kalman_sm };
        let kalman_delayed = if sm_list { kalman_sm } else { 0 };
        lists(&mut self.smoothed_coef, kalman_flat, n_rows);
        lists(
            &mut self.smoothed_se,
            if spec.emit_smoothed_se() {
                kalman_flat
            } else {
                0
            },
            n_rows,
        );
        fill(
            &mut self.smoothed_for,
            if sm_list {
                0
            } else {
                n_models * spec.smoothed_for_slots() * n_rows
            },
            None,
        );
        lists(&mut self.smoothed_coef_list, kalman_delayed, n_rows);
        lists(
            &mut self.smoothed_se_list,
            if spec.emit_smoothed_se() {
                kalman_delayed
            } else {
                0
            },
            n_rows,
        );
        lists(
            &mut self.smoothed_state_list,
            n_models * spec.hmm_smoothed_list_slots(),
            n_rows,
        );
        lists(
            &mut self.smoothed_for_list,
            if sm_list {
                n_models * spec.smoothed_for_slots()
            } else {
                0
            },
            n_rows,
        );
        self.n_models = n_models;
        self.n_slots = n_slots;
        self.n_levels = n_levels;
        self.n_conf_levels = n_conf_levels;
        self.n_scores = n_scores;
        self.n_checks = n_checks;
    }

    /// The bytes these buffers hold, the struct itself included: what a
    /// bank keeps when it keeps them for a later run ([`Self::renew`]).
    pub fn heap_bytes(&self) -> usize {
        self.flat_bytes() + self.list_bytes()
    }

    /// The bytes these buffers hold, if they are at most `limit`. The flat
    /// buffers are summed first, from their headers alone, so a long run,
    /// whose flat buffers pass a small `limit` by themselves, is answered
    /// without walking its `coef`-like lists, one `Option` per model and row
    /// each (task 270, R2-1).
    pub fn heap_bytes_within(&self, limit: usize) -> Option<usize> {
        let flat = self.flat_bytes();
        if flat > limit {
            return None;
        }
        Some(flat + self.list_bytes()).filter(|&bytes| bytes <= limit)
    }

    /// The bytes held by the struct and its flat buffers: every buffer but
    /// the `coef`-like lists, read from the headers alone.
    fn flat_bytes(&self) -> usize {
        size_of::<Self>()
            + cap(&self.rows)
            + cap(&self.processed)
            + [
                &self.pred,
                &self.resid,
                &self.sigma,
                &self.zscore,
                &self.autocorr,
                &self.metrics,
                &self.scores,
                &self.interval_scores,
                &self.conformal,
                &self.conformal_p,
                &self.checks,
                &self.health,
                &self.health_ks,
                &self.resid_q,
                &self.n_eff,
                &self.lam_selected,
                &self.target_weight,
                &self.settled,
                &self.inflation,
            ]
            .into_iter()
            .map(cap)
            .sum::<usize>()
            + cap(&self.drift)
            + cap(&self.reason)
            + cap(&self.scored_clock)
            + cap(&self.learned_clock)
            + cap(&self.smoothed_for)
    }

    /// The bytes held by the `coef`-like lists: each list's header and
    /// every vector it holds.
    fn list_bytes(&self) -> usize {
        fn lists(v: &Vec<Vec<Option<Vec<f64>>>>) -> usize {
            cap(v)
                + v.iter()
                    .map(|inner| cap(inner) + inner.iter().flatten().map(cap).sum::<usize>())
                    .sum::<usize>()
        }
        fn nested_lists(v: &Vec<Vec<NestedListRow>>) -> usize {
            cap(v)
                + v.iter()
                    .map(|inner| {
                        cap(inner)
                            + inner
                                .iter()
                                .flatten()
                                .map(|rows| cap(rows) + rows.iter().map(cap).sum::<usize>())
                                .sum::<usize>()
                    })
                    .sum::<usize>()
        }
        fn clock_lists(v: &Vec<Vec<ClockListRow>>) -> usize {
            cap(v)
                + v.iter()
                    .map(|inner| cap(inner) + inner.iter().flatten().map(cap).sum::<usize>())
                    .sum::<usize>()
        }
        [
            &self.coef,
            &self.support_coef,
            &self.se_coef,
            &self.se_coef_hc0,
            &self.se_coef_hc1,
            &self.se_coef_hc2,
            &self.se_coef_hc3,
            &self.se_coef_hac,
            &self.dfbetas,
            &self.smoothed_coef,
            &self.smoothed_se,
            &self.smoothed_state_list,
        ]
        .into_iter()
        .map(lists)
        .sum::<usize>()
            + nested_lists(&self.smoothed_coef_list)
            + nested_lists(&self.smoothed_se_list)
            + clock_lists(&self.smoothed_for_list)
    }

    /// How many rows one learning run covers (docs/records/PERFORMANCE-LOG.md §13).
    ///
    /// The buffers are slot-major, so a row is one store per slot, each a
    /// slot's stride apart. Two things about that stride decide how fast a
    /// wide model runs, and both are set here rather than by whoever chose
    /// the chunk size:
    ///
    /// - The run's buffers stay within `BUDGET` bytes, so the lines a wide
    ///   model's row touches are still in cache when the next rows complete
    ///   them; over a 400k-row chunk, a 230-statistic `ew_cov` spent a third
    ///   of its time on stores to lines that had already been evicted. The
    ///   run's feature rows count against it too.
    /// - A slot's stride is an odd number of 128-byte lines, so no two slots
    ///   of a row share a cache set. A power-of-two chunk (65 536 rows, the
    ///   natural streaming batch) put every slot of that `ew_cov` in the
    ///   same set and ran it 2-3x slower than a chunk 16 rows longer.
    ///
    /// A narrow model writes a few values per row and gets runs of a hundred
    /// thousand rows: a chunk is one or two, and nothing changes for it. A
    /// model too wide for the budget gets the floor, five lines. Chunk
    /// invariance is what makes the split invisible.
    pub fn run_rows(spec: &Spec, n_models: usize, n_slots: usize) -> usize {
        const BUDGET: usize = 2 << 20;
        /// `f64`s in a 128-byte line, the longest of the two OSes' lines.
        const LINE: usize = 16;
        /// The row's schedule, written in pass 1 and read in pass 2, in
        /// `f64`s: 128 bytes since the exact stamps joined it, against 56
        /// before. Left out of the budget, a narrow model's run of a hundred
        /// thousand rows held 13 to 16 MB of it, past the cache, and a `seqtest`
        /// row spent a large share storing and reloading it (task 250).
        const PLAN: usize = std::mem::size_of::<RowPlan>().div_ceil(8);
        // The run's features as rows, which the bank copies from the
        // chunk's columns just before the run and the passes read back: a
        // `f64` a feature (docs/records/PERFORMANCE-LOG.md §42). Inside the budget they
        // are read from cache; the whole chunk in rows was written to
        // memory and read back, 160 MB at 20,000 rows of 1,000 features.
        let width = Buffers::of(spec).per_row(spec, n_models, n_slots).max(1) + PLAN + spec.k();
        let lines = BUDGET / (8 * width * LINE);
        let lines = if lines.is_multiple_of(2) {
            lines.saturating_sub(1)
        } else {
            lines
        };
        lines.max(5) * LINE
    }

    /// Offset of `(model, slot)` at row `ri`, in any `n_models * n_slots *
    /// n_rows` buffer. The single place the layout is spelled out: each
    /// instance's block is written by `run_instance` at `slot * n_rows + ri`,
    /// and the sink in `bank/assembly.rs` reads the whole buffer through this.
    #[inline]
    pub fn at(n_slots: usize, n_rows: usize, mi: usize, slot: usize, ri: usize) -> usize {
        (mi * n_slots + slot) * n_rows + ri
    }

    /// The first row when this task's rows are one unbroken run `base..base +
    /// n`: a single group, or a group that arrives in blocks. `rows` is
    /// strictly increasing by construction (`group_indices` walks the chunk
    /// in order and each row belongs to one group), so the two ends decide.
    pub fn contiguous(&self) -> Option<usize> {
        let (&first, &last) = (self.rows.first()?, self.rows.last()?);
        debug_assert!(
            self.rows.windows(2).all(|w| w[0] < w[1]),
            "rows out of order"
        );
        (last - first + 1 == self.rows.len()).then_some(first)
    }
}

/// The bytes a vector's buffer holds: its capacity, not its length.
fn cap<T>(v: &Vec<T>) -> usize {
    v.capacity() * size_of::<T>()
}
