//! What a stream's run is made of beyond the models: the instances split
//! out of the stream's state ([`build_instances`]), the plans' list a thread
//! keeps between runs, and the slots' lists joined into one. Moved out of
//! `stream.rs`, which is held under the size a source file is.

use super::*;

/// Split per-instance state and output into disjoint pieces, so instances
/// can run concurrently. Every `ChunkOut` buffer is laid out model-major,
/// which is what makes an instance's region one contiguous slice; the
/// state vectors are already `[mi]`-indexed.
#[allow(clippy::too_many_arguments)]
pub(super) fn build_instances<'a>(
    spec: &'a Spec,
    models: impl Iterator<Item = ModelRef<'a>>,
    mut rings: impl Iterator<Item = Option<SpreadRef<'a>>>,
    decays: &[Decay],
    check_cfgs: &'a [CheckCfg],
    diag: Diagnostics<'a>,
    out: &'a mut ChunkOut,
    n_rows: usize,
    rows_before: u64,
) -> Instances<'a> {
    let n = decays.len();
    let block = out.n_slots * n_rows;
    let n_targets = if n == 0 || n_rows == 0 || out.lam_selected.is_empty() {
        0
    } else {
        out.lam_selected.len() / (n * n_rows)
    };

    let mut drift = diag.drift.iter_mut();
    let mut resid_q = diag.resid_q.iter_mut();
    let mut autocorr = diag.autocorr.iter_mut();
    let mut metrics = diag.metrics.iter_mut();
    let mut scores = diag.scores.iter_mut();
    let mut interval_scores = diag.interval_scores.iter_mut();
    let mut conformal = diag.conformal.iter_mut();
    let mut conformal_multi = diag.conformal_multi.iter_mut();
    let mut conformal_p = diag.conformal_p.iter_mut();
    let mut checks = diag.checks.iter_mut();
    let mut o_pred = out.pred.chunks_mut(block.max(1));
    let mut o_resid = out.resid.chunks_mut(block.max(1));
    let mut o_sigma = out.sigma.chunks_mut(block.max(1));
    let mut o_resid_z = out.zscore.chunks_mut(block.max(1));
    let mut o_autocorr = out.autocorr.chunks_mut(block.max(1));
    let mut o_metrics = out.metrics.chunks_mut((3 * block).max(1));
    let mut o_scores = out.scores.chunks_mut((out.n_scores * block).max(1));
    let mut o_interval_scores = out
        .interval_scores
        .chunks_mut((2 * out.n_conf_levels * block).max(1));
    let mut o_conformal = out
        .conformal
        .chunks_mut((3 * out.n_conf_levels * block).max(1));
    let mut o_conformal_p = out.conformal_p.chunks_mut(block.max(1));
    let mut o_checks = out.checks.chunks_mut((out.n_checks * block).max(1));
    let health_block = if spec.emit_feature_health {
        2 * spec.k() * n_rows
    } else {
        0
    };
    let mut o_health = out.health.chunks_mut(health_block.max(1));
    let health_ks_block = if spec.has_feature_ks() {
        2 * spec.k() * n_rows
    } else {
        0
    };
    let mut o_health_ks = out.health_ks.chunks_mut(health_ks_block.max(1));
    let mut o_resid_q = out.resid_q.chunks_mut((out.n_levels * block).max(1));
    let mut o_drift = out.drift.chunks_mut(block.max(1));
    let mut o_n_eff = out.n_eff.chunks_mut(n_rows.max(1));
    let mut o_lam = out.lam_selected.chunks_mut((n_targets * n_rows).max(1));
    // Each target's own weight under `emit_target_weight`, a block of
    // `m * n_rows` an instance (docs/PLAN.md task 568).
    let tw_block = if spec.emit_target_weight {
        spec.m() * n_rows
    } else {
        0
    };
    let mut o_target_weight = out.target_weight.chunks_mut(tw_block.max(1));
    let mut o_coef = out.coef.iter_mut();
    let mut o_settled = out.settled.chunks_mut(n_rows.max(1));
    let mut o_reason = out.reason.chunks_mut(n_rows.max(1));
    let mut o_inflation = out.inflation.chunks_mut(block.max(1));
    let mut o_support_coef = out.support_coef.iter_mut();
    let mut o_se_coef = out.se_coef.iter_mut();
    let mut o_se_hc0 = out.se_coef_hc0.iter_mut();
    let mut o_se_hc1 = out.se_coef_hc1.iter_mut();
    let mut o_se_hc2 = out.se_coef_hc2.iter_mut();
    let mut o_se_hc3 = out.se_coef_hc3.iter_mut();
    let mut o_se_hac = out.se_coef_hac.iter_mut();
    let mut o_dfbetas = out.dfbetas.iter_mut();
    let mut o_smoothed = out.smoothed_coef.iter_mut();
    let mut o_smoothed_se = out.smoothed_se.iter_mut();
    let mut o_smoothed_for = out
        .smoothed_for
        .chunks_mut((spec.smoothed_for_slots() * n_rows).max(1));
    let mut o_smoothed_list = out.smoothed_coef_list.iter_mut();
    let mut o_smoothed_se_list = out.smoothed_se_list.iter_mut();
    let mut o_smoothed_state_list = out
        .smoothed_state_list
        .chunks_mut(spec.hmm_smoothed_list_slots().max(1));
    let sm_for_list_slots = if spec.smoothed_is_list() {
        spec.smoothed_for_slots()
    } else {
        0
    };
    let mut o_smoothed_for_list = out.smoothed_for_list.chunks_mut(sm_for_list_slots.max(1));

    let n_slots = out.n_slots;
    let mut decays = decays.iter();
    let mut check_cfgs = check_cfgs.iter();
    let mut resid_var = diag.resid_var.iter_mut();
    let mut resid_w = diag.resid_w.iter_mut();
    let mut scratch = diag.scratch.iter_mut();
    let mut score_pred = diag.score_pred.iter_mut();
    let mut decay_time = diag.decay_time.iter_mut();
    let mut restarted = diag.restarted.iter_mut();
    let mut pending_clock = diag.pending_clock.iter_mut();
    let mut notified = diag.notified.iter_mut();

    // Pulled in lockstep: each iterator yields disjoint `&mut`s, so every
    // Instance owns its own piece of everything.
    let mut built = models.map(|model| {
        let decay = *decays.next().expect("one per instance");
        Instance {
            steps: rows_before,
            spec,
            shards: marginal_shards(spec, model.get()),
            model,
            resid_win: rings.next().expect("one per instance"),
            decay,
            check_cfg: check_cfgs.next().expect("one per instance"),
            residuals: !spec.model.predicts_no_target(),
            resid_var: resid_var.next().expect("one per instance"),
            resid_w: resid_w.next().expect("one per instance"),
            drift: drift.next(),
            resid_q: resid_q.next(),
            autocorr: autocorr.next(),
            metrics: metrics.next(),
            scores: scores.next(),
            interval_scores: interval_scores.next(),
            conformal: conformal.next(),
            conformal_multi: conformal_multi.next(),
            conformal_p: conformal_p.next(),
            checks: checks.next(),
            scratch: scratch.next().expect("one per instance"),
            score_pred: score_pred.next().expect("one per instance"),
            n_slots,
            n_rows,
            o_pred: o_pred.next().unwrap_or_default(),
            o_resid: o_resid.next().unwrap_or_default(),
            o_sigma: o_sigma.next().unwrap_or_default(),
            o_resid_z: o_resid_z.next().unwrap_or_default(),
            o_autocorr: o_autocorr.next().unwrap_or_default(),
            o_metrics: o_metrics.next().unwrap_or_default(),
            o_scores: o_scores.next().unwrap_or_default(),
            o_interval_scores: o_interval_scores.next().unwrap_or_default(),
            o_conformal: o_conformal.next().unwrap_or_default(),
            o_conformal_p: o_conformal_p.next().unwrap_or_default(),
            o_checks: o_checks.next().unwrap_or_default(),
            o_health: o_health.next().unwrap_or_default(),
            o_health_ks: o_health_ks.next().unwrap_or_default(),
            o_resid_q: o_resid_q.next().unwrap_or_default(),
            o_drift: o_drift.next().unwrap_or_default(),
            o_n_eff: o_n_eff.next().unwrap_or_default(),
            o_lam: o_lam.next().unwrap_or_default(),
            o_target_weight: o_target_weight.next().unwrap_or_default(),
            o_coef: o_coef.next().expect("one per instance"),
            o_settled: o_settled.next().unwrap_or_default(),
            o_reason: o_reason.next().unwrap_or_default(),
            o_inflation: o_inflation.next().unwrap_or_default(),
            o_support_coef: o_support_coef.next().expect("one per instance"),
            o_se_coef: o_se_coef.next().expect("one per instance"),
            o_se_hc0: o_se_hc0.next().expect("one per instance"),
            o_se_hc1: o_se_hc1.next(),
            o_se_hc2: o_se_hc2.next(),
            o_se_hc3: o_se_hc3.next(),
            o_se_hac: o_se_hac.next().expect("one per instance"),
            o_dfbetas: o_dfbetas.next(),
            o_smoothed: o_smoothed.next(),
            o_smoothed_se: o_smoothed_se.next(),
            o_smoothed_for: o_smoothed_for.next().unwrap_or_default(),
            o_smoothed_list: o_smoothed_list.next(),
            o_smoothed_se_list: o_smoothed_se_list.next(),
            o_smoothed_state_list: o_smoothed_state_list.next().unwrap_or_default(),
            o_smoothed_for_list: o_smoothed_for_list.next().unwrap_or_default(),
            decay_time: decay_time.next().expect("one per instance"),
            restarted: restarted.next().expect("one per instance"),
            notified: notified.next().expect("one per instance"),
            refit_over_budget: diag.refit_over_budget,
            ungated_spread: diag.ungated_spread,
            weighted_autocorr: diag.weighted_autocorr,
            ungated_drift: diag.ungated_drift,
            // None without an embargo, which keeps no held rows' clock.
            pending_clock: pending_clock.next(),
            group_hash: diag.group_hash,
        }
    });
    // One instance, the commonest, is held where it is made: a list of one
    // was an allocation of its own for every group of every chunk, five
    // hundred bytes each, and the call that frees it.
    let Some(first) = built.next() else {
        return Instances::Many(Vec::new());
    };
    let Some(second) = built.next() else {
        return Instances::One(first);
    };
    let mut all = Vec::with_capacity(n.max(2));
    all.push(first);
    all.push(second);
    all.extend(built);
    Instances::Many(all)
}

/// A stream's instances for one run: the one it has, or the list of them.
// The size difference is the point: boxing the one would be the allocation
// it is held here to avoid.
#[allow(clippy::large_enum_variant)]
pub(super) enum Instances<'a> {
    One(Instance<'a>),
    Many(Vec<Instance<'a>>),
}

impl<'a> Instances<'a> {
    pub(super) fn as_mut_slice(&mut self) -> &mut [Instance<'a>] {
        match self {
            Self::One(inst) => std::slice::from_mut(inst),
            Self::Many(all) => all,
        }
    }
}

thread_local! {
    /// The plans' buffer of the last run this thread scheduled, empty, for
    /// the next ([`plan_buffer`]).
    static PLANS: std::cell::Cell<Vec<RowPlan>> = const { std::cell::Cell::new(Vec::new()) };
}

/// The most plans a thread keeps room for between runs: a run of a group's
/// few rows is the case that made them (a hundred and twenty-eight bytes
/// each, and a call to allocate them for every group of every chunk); a
/// long run's are freed as before.
const PLANS_KEPT: usize = 4096;

/// An empty list for a run's plans: the one this thread kept, if any.
pub(super) fn plan_buffer() -> Vec<RowPlan> {
    PLANS.take()
}

/// Keep a run's plans' room for the next run on this thread, where it is
/// at most [`PLANS_KEPT`] plans.
pub(super) fn keep_plans(mut plans: Vec<RowPlan>) {
    if plans.capacity() <= PLANS_KEPT {
        plans.clear();
        PLANS.set(plans);
    }
}

/// The slots' lists as one list, slot after slot: what `into_iter().flatten()
/// .collect()` gives, without the list it was collected into when the model
/// has one slot (the one a target's coefficients are), and with its size
/// known where it has several.
pub(super) fn joined(mut slots: Vec<Vec<f64>>) -> Vec<f64> {
    match slots.len() {
        0 => Vec::new(),
        1 => slots.pop().expect("one slot"),
        _ => {
            let mut all = Vec::with_capacity(slots.iter().map(Vec::len).sum());
            for slot in slots {
                all.extend(slot);
            }
            all
        }
    }
}

/// Null every Kalman innovation diagnostic (`loglik`, `mahal`) when the whole
/// row is withheld by `min_settled_frac`.
pub(super) fn withhold_kalman_all(extra: &mut Option<rheon_core::Extra>) {
    if let Some(rheon_core::Extra::Kalman { values }) = extra.as_mut() {
        values.fill(f64::NAN);
    }
}

/// An instance's list-typed `kalman` smoother outputs under delayed learning.
pub(super) struct SmoothedListOut<'a> {
    pub(super) coef: Option<&'a mut Vec<NestedListRow>>,
    pub(super) se: Option<&'a mut Vec<NestedListRow>>,
    pub(super) labels: &'a mut [Vec<ClockListRow>],
    pub(super) n_rows: usize,
}

/// Append one released observation's `kalman` fixed-lag smoother emission to
/// arriving row `at`'s parallel lists, in exact release order. Rows with zero
/// emitting releases stay `None` (materialized as `null`, never `[]`).
pub(super) fn write_smoothed_list(
    model: &AnyModel,
    buf: &mut rheon_core::Smoothed,
    out: SmoothedListOut<'_>,
    at: usize,
) {
    if at >= out.n_rows || !model.smoothed_into(buf) || !buf.emitted.contains(&true) {
        return;
    }
    for (j, slot) in out.labels.iter_mut().enumerate() {
        let label = if buf.emitted[j] { buf.label[j] } else { None };
        slot[at].get_or_insert_with(Vec::new).push(label);
    }
    let k = buf.coef.len() / buf.emitted.len().max(1);
    let make_vec = |vals: &[f64], root: bool| -> Vec<f64> {
        let mut row = vec![f64::NAN; vals.len()];
        for (j, e) in buf.emitted.iter().enumerate() {
            if *e {
                for (o, v) in row[j * k..(j + 1) * k].iter_mut().zip(&vals[j * k..]) {
                    *o = if root && !v.is_nan() {
                        v.max(0.0).sqrt()
                    } else {
                        *v
                    };
                }
            }
        }
        row
    };
    if let Some(o) = out.coef {
        o[at]
            .get_or_insert_with(Vec::new)
            .push(make_vec(&buf.coef, false));
    }
    if let Some(o) = out.se {
        o[at]
            .get_or_insert_with(Vec::new)
            .push(make_vec(&buf.var, true));
    }
}

/// Append one released observation's `hmm` fixed-lag smoother emission to
/// arriving row `at`'s parallel `smoothed_state<j>` and `smoothed_for` lists,
/// in exact release order.
pub(super) fn write_hmm_smoothed_list(
    model: &AnyModel,
    buf: &mut rheon_core::Smoothed,
    pred: &[f64],
    states: &mut [Vec<Option<Vec<f64>>>],
    labels: &mut [Vec<ClockListRow>],
    at: usize,
    n_rows: usize,
) {
    let k = states.len();
    if k == 0
        || at >= n_rows
        || pred.len() < k
        || !model.smoothed_into(buf)
        || !buf.emitted.first().copied().unwrap_or(false)
    {
        return;
    }
    let start = pred.len() - k;
    for (s, slot) in states.iter_mut().enumerate() {
        slot[at].get_or_insert_with(Vec::new).push(pred[start + s]);
    }
    if let Some(lbl_slot) = labels.first_mut() {
        lbl_slot[at]
            .get_or_insert_with(Vec::new)
            .push(buf.label.first().copied().flatten());
    }
}

/// Null target `tj`'s Kalman innovation diagnostics (`loglik`, `mahal`) when
/// a per-target readiness gate (`min_weight`, `max_error_inflation`) withholds
/// `tj`.
pub(super) fn withhold_kalman_target(
    extra: &mut Option<rheon_core::Extra>,
    m_targets: usize,
    tj: usize,
) {
    if let Some(rheon_core::Extra::Kalman { values }) = extra.as_mut()
        && m_targets > 0
        && tj < m_targets
    {
        for block in values.chunks_exact_mut(m_targets) {
            block[tj] = f64::NAN;
        }
    }
}
