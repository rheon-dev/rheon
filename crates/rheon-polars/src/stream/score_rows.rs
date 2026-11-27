//! Per-row streaming score (`mae`, `pinball`, `log_loss`), interval score
//! (`width`, `winkler`), multi-level / asymmetric conformal, and standalone
//! conformal p-value work, out of `run_instance`'s hot loop (`stream.rs`).
//!
//! Marked `#[inline(never)]` so specs that do not enable these diagnostics pay
//! zero code-size or register-pressure cost in `run_instance`.

use rheon_core::{
    Conformal, ConformalPRing, ConformalSlot, Decay, SlotIntervalScores, SlotScores, Stamp,
    sort_offsets_non_crossing,
};

use super::conformal_rows::At;
use super::*;
use crate::spec::MetricKind;

/// Unpacked score-time slices from a released row's `score_pred` record.
pub(super) struct UnpackedDelay {
    pub shown_radius: Option<Vec<f64>>,
    pub shown_multi: Option<Vec<f64>>,
    pub shown_sigma: Option<Vec<f64>>,
    pub kept: bool,
}

/// Build the initial `scores`, `interval_scores`, `conformal`,
/// `conformal_multi`, and `conformal_p` per-instance diagnostic vectors.
#[allow(clippy::type_complexity)]
pub(super) fn init_score_diags(
    spec: &Spec,
    slots: &[usize],
) -> Result<
    (
        Vec<Vec<SlotScores>>,
        Vec<Vec<SlotIntervalScores>>,
        Vec<Vec<Conformal>>,
        Vec<Vec<ConformalSlot>>,
        Vec<Vec<ConformalPRing>>,
    ),
    String,
> {
    let scores = if spec.emit_metrics.has_scores() {
        let qs = spec.pinball_levels();
        let proto = SlotScores::new(
            spec.emit_metrics.has(MetricKind::Mae),
            &qs,
            spec.emit_metrics.has(MetricKind::LogLoss),
        );
        slots.iter().map(|&n| vec![proto.clone(); n]).collect()
    } else {
        Vec::new()
    };
    let interval_scores = if spec.emit_metrics.has_interval_scores() {
        let proto = SlotIntervalScores::new(
            &spec.conformal_levels(),
            spec.emit_metrics.has(MetricKind::Width),
            spec.emit_metrics.has(MetricKind::Winkler),
        );
        slots.iter().map(|&n| vec![proto.clone(); n]).collect()
    } else {
        Vec::new()
    };
    let conformal = if spec.uses_legacy_conformal() {
        let level = spec.conformal_levels()[0];
        let proto = Conformal::new(level, spec.conformal_rate_or_default())?;
        slots.iter().map(|&n| vec![proto.clone(); n]).collect()
    } else {
        Vec::new()
    };
    let conformal_multi = if spec.uses_multi_conformal() {
        let proto = ConformalSlot::new(
            &spec.conformal_levels(),
            spec.conformal_rate_or_default(),
            spec.conformal_symmetric,
        )?;
        slots.iter().map(|&n| vec![proto.clone(); n]).collect()
    } else {
        Vec::new()
    };
    let conformal_p = match spec.conformal_p_window() {
        Some(win) => {
            let proto = ConformalPRing::new(win, spec.conformal_p_per_ring_cap());
            slots.iter().map(|&n| vec![proto.clone(); n]).collect()
        }
        None => Vec::new(),
    };
    Ok((
        scores,
        interval_scores,
        conformal,
        conformal_multi,
        conformal_p,
    ))
}

/// Record raw pre-row `(q_lo, q_hi)` offsets per `(slot, level)` for a held
/// row under `embargo` / `label_clock` when multi-level or asymmetric
/// conformal is active (`2 * n_levels * n_slots` floats).
#[inline(never)]
pub(super) fn shown_multi_offsets(
    record: &mut Vec<f64>,
    cs: &[ConformalSlot],
    ring: Option<&ResidWindow>,
    decay: Decay,
    spread: Option<(&[f64], &[f64])>,
) {
    let n_levels = cs.first().map_or(0, ConformalSlot::len);
    let mut q_lo = Vec::with_capacity(n_levels);
    let mut q_hi = Vec::with_capacity(n_levels);
    for (slot, c) in cs.iter().enumerate() {
        let sig = spread.map_or(f64::NAN, |s| pre_row_sigma(ring, decay, s, slot));
        c.raw_offsets(sig, &mut q_lo, &mut q_hi);
        for li in 0..n_levels {
            record.push(q_lo[li]);
            record.push(q_hi[li]);
        }
    }
}

/// Record pre-row `sigma` per slot for a held row under `embargo` /
/// `label_clock` when `emit_conformal_p` is active (`n_slots` floats).
#[inline(never)]
pub(super) fn shown_conformal_p_sigma(
    record: &mut Vec<f64>,
    n_slots: usize,
    ring: Option<&ResidWindow>,
    decay: Decay,
    spread: Option<(&[f64], &[f64])>,
) {
    for slot in 0..n_slots {
        let sig = spread.map_or(f64::NAN, |s| pre_row_sigma(ring, decay, s, slot));
        record.push(sig);
    }
}

/// Pack a held row's score-time prediction, shown conformal bounds/sigma,
/// row inflation, and ungated predictions into `score_pred`.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub(super) fn pack_score_pred(
    score_pred: &mut VecDeque<Vec<f64>>,
    conformal: Option<&[Conformal]>,
    conformal_multi: Option<&[ConformalSlot]>,
    has_cp: bool,
    ring: Option<&ResidWindow>,
    decay: Decay,
    spread: Option<(&[f64], &[f64])>,
    pred: &[f64],
    row_infl: &[f64],
    ungated: &[f64],
    has_row_infl: bool,
    kept: bool,
    place: Option<(u32, u32)>,
) {
    let multi_conf_parts = conformal_multi
        .and_then(|cs| cs.first())
        .map_or(0, |c| 2 * c.len());
    let len = pred.len()
        + conformal.map_or(0, <[Conformal]>::len)
        + multi_conf_parts * pred.len()
        + if has_cp { pred.len() } else { 0 }
        + if has_row_infl { row_infl.len() } else { 0 }
        + if kept { ungated.len() } else { 0 };
    let mut record = Vec::with_capacity(len);
    record.extend_from_slice(pred);
    if let Some(cs) = conformal {
        conformal_rows::shown_radii(&mut record, cs, ring, decay, spread);
    }
    if let Some(cs) = conformal_multi {
        shown_multi_offsets(&mut record, cs, ring, decay, spread);
    }
    if has_cp {
        shown_conformal_p_sigma(&mut record, pred.len(), ring, decay, spread);
    }
    if has_row_infl {
        record.extend_from_slice(row_infl);
    }
    if kept {
        record.extend_from_slice(ungated);
    }
    match place {
        Some((at, drop)) => {
            score_pred.insert(at as usize, record);
            if drop != u32::MAX {
                score_pred.remove(drop as usize);
            }
        }
        None => score_pred.push_back(record),
    }
}

/// Pop and unpack a released row's `score_pred` record.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub(super) fn unpack_score_pred(
    score_pred: &mut VecDeque<Vec<f64>>,
    conformal: bool,
    multi_conf_parts: usize,
    has_cp: bool,
    ungated_ok: bool,
    pred: &mut [f64],
    row_infl: &mut Vec<f64>,
    ungated: &mut Vec<f64>,
    has_row_infl: bool,
) -> UnpackedDelay {
    let n = pred.len();
    let parts = 1
        + usize::from(conformal)
        + multi_conf_parts
        + usize::from(has_cp)
        + usize::from(has_row_infl);
    let mut out = UnpackedDelay {
        shown_radius: None,
        shown_multi: None,
        shown_sigma: None,
        kept: false,
    };
    match score_pred.pop_front() {
        Some(p) if p.len() == parts * n || (ungated_ok && p.len() == (parts + 1) * n) => {
            let mut off = 0;
            pred.copy_from_slice(&p[off..off + n]);
            off += n;
            if conformal {
                out.shown_radius = Some(p[off..off + n].to_vec());
                off += n;
            }
            if multi_conf_parts > 0 {
                let mlen = multi_conf_parts * n;
                out.shown_multi = Some(p[off..off + mlen].to_vec());
                off += mlen;
            }
            if has_cp {
                out.shown_sigma = Some(p[off..off + n].to_vec());
                off += n;
            }
            if has_row_infl {
                row_infl.clear();
                row_infl.extend_from_slice(&p[off..off + n]);
                off += n;
            }
            if p.len() > off {
                ungated.clear();
                ungated.extend_from_slice(&p[off..]);
                out.kept = true;
            }
        }
        _ => pred.fill(f64::NAN),
    }
    out
}

/// Row slices read by [`step_conformal_multi`]: `(pred, resid, sigma, ys)`.
pub(super) type ScoreRowSlices<'a> = (&'a [f64], &'a [f64], &'a [f64], &'a [Option<f64>]);

/// Step multi-level or asymmetric conformal trackers and (when active)
/// `SlotIntervalScores` for one row.
///
/// Outputs in `o_conformal` are laid out as `3 * n_levels` blocks per instance:
/// for each level `li`, `(3 * li + 0)` = `lo`, `(3 * li + 1)` = `hi`,
/// `(3 * li + 2)` = `coverage`.
/// Outputs in `o_interval_scores` (when non-empty) are laid out as
/// `2 * n_levels` blocks per instance: `(2 * li + 0)` = `width`,
/// `(2 * li + 1)` = `winkler`.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub(super) fn step_conformal_multi(
    cs: &mut [ConformalSlot],
    mut iscores: Option<&mut [SlotIntervalScores]>,
    o_conformal: &mut [f64],
    o_interval_scores: &mut [f64],
    at: At,
    (pred, resid, sig, ys): ScoreRowSlices<'_>,
    nc: usize,
    (emit, learn): (bool, bool),
    shown_multi: Option<&[f64]>,
    (lam, w): (f64, f64),
    held: usize,
) {
    let n_levels = cs.first().map_or(0, ConformalSlot::len);
    if n_levels == 0 {
        return;
    }
    let mut raw_lo = Vec::with_capacity(n_levels);
    let mut raw_hi = Vec::with_capacity(n_levels);
    let mut sorted_lo = vec![f64::NAN; n_levels];
    let mut sorted_hi = vec![f64::NAN; n_levels];

    for (slot, c) in cs.iter_mut().enumerate() {
        let p = pred[slot];
        let s = sig.get(slot).copied().unwrap_or(f64::NAN);

        // Compute pre-row raw & sorted offsets for emission and direct learning.
        if emit || (learn && shown_multi.is_none()) {
            c.raw_offsets(s, &mut raw_lo, &mut raw_hi);
            sorted_lo.copy_from_slice(&raw_lo);
            sorted_hi.copy_from_slice(&raw_hi);
            sort_offsets_non_crossing(&mut sorted_lo, &mut sorted_hi);
        }

        if emit {
            let base_slot = slot * at.n_rows + at.ri;
            for li in 0..n_levels {
                let b = 3 * li * at.block + base_slot;
                o_conformal[b] = p - sorted_lo[li];
                o_conformal[b + at.block] = p + sorted_hi[li];
                o_conformal[b + 2 * at.block] = c.coverage(li).unwrap_or(f64::NAN);
            }
            if let Some(isc) = iscores.as_deref() {
                for li in 0..n_levels {
                    let b = 2 * li * at.block + base_slot;
                    o_interval_scores[b] = isc[slot].width(li).unwrap_or(f64::NAN);
                    o_interval_scores[b + at.block] = isc[slot].winkler(li).unwrap_or(f64::NAN);
                }
            }
        }

        if learn {
            let yj = ys.get(slot / nc).copied().flatten().unwrap_or(f64::NAN);
            if let Some(sm) = shown_multi {
                let slice = &sm[slot * 2 * n_levels..(slot + 1) * 2 * n_levels];
                for li in 0..n_levels {
                    sorted_lo[li] = slice[2 * li];
                    sorted_hi[li] = slice[2 * li + 1];
                }
                sort_offsets_non_crossing(&mut sorted_lo, &mut sorted_hi);
                c.update_against(
                    resid[slot],
                    s,
                    lam,
                    w,
                    Some(slice),
                    Some((&sorted_lo, &sorted_hi)),
                    held,
                );
            } else {
                c.update_against(
                    resid[slot],
                    s,
                    lam,
                    w,
                    None,
                    Some((&sorted_lo, &sorted_hi)),
                    held,
                );
            }
            if let Some(isc) = iscores.as_deref_mut() {
                for li in 0..n_levels {
                    let lo = p - sorted_lo[li];
                    let hi = p + sorted_hi[li];
                    isc[slot].update_level(li, yj, lo, hi, lam, w);
                }
            }
        }
    }
}

/// Step `SlotIntervalScores` paired with legacy single-level symmetric
/// `Conformal` (when `uses_legacy_conformal` and `width`/`winkler` are on).
///
/// Must be called BEFORE `step_conformal` mutates `cs` so direct rows read the
/// exact same pre-row radius `c.radius(sig)` that `step_conformal` emitted.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub(super) fn step_interval_scores_legacy(
    cs: &[Conformal],
    iscores: &mut [SlotIntervalScores],
    o_interval_scores: &mut [f64],
    at: At,
    (pred, sig, ys): (&[f64], &[f64], &[Option<f64>]),
    nc: usize,
    (emit, learn): (bool, bool),
    shown_radius: Option<&[f64]>,
    (lam, w): (f64, f64),
) {
    for (slot, (c, isc)) in cs.iter().zip(iscores.iter_mut()).enumerate() {
        if emit {
            let base = slot * at.n_rows + at.ri;
            o_interval_scores[base] = isc.width(0).unwrap_or(f64::NAN);
            o_interval_scores[at.block + base] = isc.winkler(0).unwrap_or(f64::NAN);
        }
        if learn {
            let p = pred[slot];
            let q = match shown_radius {
                Some(qs) => qs[slot],
                None => c
                    .radius(sig.get(slot).copied().unwrap_or(f64::NAN))
                    .unwrap_or(f64::NAN),
            };
            let yj = ys.get(slot / nc).copied().flatten().unwrap_or(f64::NAN);
            isc.update_level(0, yj, p - q, p + q, lam, w);
        }
    }
}

/// Step point-loss scores (`mae`, `pinball`, `log_loss`) for one row.
///
/// Output blocks are written in canonical order matching `bank/fields.rs`:
/// `[mae? , pinball[0..P], log_loss?]`.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub(super) fn step_scores(
    scores: &mut [SlotScores],
    o_scores: &mut [f64],
    at: At,
    (has_mae, n_pinball, has_log_loss): (bool, usize, bool),
    (pred, ys): (&[f64], &[Option<f64>]),
    nc: usize,
    (emit, learn): (bool, bool),
    (lam, w): (f64, f64),
) {
    for (slot, sc) in scores.iter_mut().enumerate() {
        if emit {
            let mut b = 0usize;
            let base = slot * at.n_rows + at.ri;
            if has_mae {
                o_scores[b * at.block + base] = sc.mae().unwrap_or(f64::NAN);
                b += 1;
            }
            for pi in 0..n_pinball {
                o_scores[b * at.block + base] = sc.pinball(pi).unwrap_or(f64::NAN);
                b += 1;
            }
            if has_log_loss {
                o_scores[b * at.block + base] = sc.log_loss().unwrap_or(f64::NAN);
            }
        }
        if learn {
            let yj = ys.get(slot / nc).copied().flatten().unwrap_or(f64::NAN);
            sc.update(pred[slot], yj, lam, w);
        }
    }
}

/// Step standalone per-row conformal p-value rings for one row.
///
/// Reads `p_t` out-of-sample before pushing the row's own score, respects
/// zero-weight Rule 9 inside `ConformalPRing::push`, and drains any once-per-
/// group state-cap half-warning into `notified`.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub(super) fn step_conformal_p(
    rings: &mut [ConformalPRing],
    o_conformal_p: &mut [f64],
    at: At,
    spec: &Spec,
    decay: Decay,
    (resid, sig): (&[f64], &[f64]),
    shown_sigma: Option<&[f64]>,
    (emit, learn): (bool, bool),
    (d_clock, score_d, stamp, w): (f64, f64, Option<Stamp>, f64),
    notified: &mut Notified,
) {
    let n_slots_u64 = (rings.len() as u64).max(1);
    let n_rings = n_slots_u64 * (spec.decays().map_or(1, |d| d.len()) as u64).max(1);
    for (slot, ring) in rings.iter_mut().enumerate() {
        if emit {
            let rv = resid.get(slot).copied().unwrap_or(f64::NAN);
            let sd = sig.get(slot).copied().unwrap_or(f64::NAN);
            let a_t = if rv.is_finite() && sd.is_finite() && sd > 0.0 {
                rv.abs() / sd
            } else {
                f64::NAN
            };
            o_conformal_p[slot * at.n_rows + at.ri] = ring
                .p_value_at(a_t, score_d, stamp, decay)
                .unwrap_or(f64::NAN);
        }
        if learn {
            let rv = resid.get(slot).copied().unwrap_or(f64::NAN);
            let sd = shown_sigma
                .and_then(|ss| ss.get(slot).copied())
                .unwrap_or_else(|| sig.get(slot).copied().unwrap_or(f64::NAN));
            let a_t = if rv.is_finite() && sd.is_finite() && sd > 0.0 {
                rv.abs() / sd
            } else {
                f64::NAN
            };
            ring.push(a_t, d_clock, stamp, w);
            if let Some((bytes, _per_cap)) = ring.take_state_warning() {
                let group_cap = spec.conformal_p_cap_bytes();
                let group_bytes = bytes.saturating_mul(n_rings);
                notified.raise(state_cap_notice(spec, group_bytes, group_cap));
            }
        }
    }
}
