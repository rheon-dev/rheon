//! Hierarchical `prior_from` DAG resolution, causality validation, topological
//! execution layers, dynamic `coef_prior` column extraction, and parent/child
//! coefficient scatter/gather for [`Bank`](super::Bank).

use polars::prelude::*;
use polars_arrow::array::Float64Array;

use crate::arrow::ArrowChunk;
use crate::spec::Spec;
use crate::stream::{ParentCapture, PriorRows, write_ready_coefs};

use super::{Groups, Layout, SpecDerived, Streams, place_of};

/// Resolve all `prior_from` edges, enforce parent/child compatibility and
/// embargo / `label_clock` causality, refuse cycles, and return topological
/// execution layers covering both `prior_from` and `compare` (`seqtest`) edges.
pub(super) fn resolve_dag(
    specs: &[Spec],
    derived: &mut [SpecDerived],
) -> Result<Vec<Vec<usize>>, String> {
    let n = specs.len();
    for si in 0..n {
        let Some(parent_name) = specs[si].model.prior_from() else {
            continue;
        };
        let child = &specs[si];
        let Some(pi) = specs.iter().position(|s| s.name == parent_name) else {
            let have: Vec<&str> = specs.iter().map(|s| s.name.as_str()).collect();
            return Err(format!(
                "spec {:?}: prior_from={parent_name:?} names no spec of this bank, whose specs \
                 are {have:?}. Set prior_from to the name of one of those specs.",
                child.name
            ));
        };
        if pi == si {
            return Err(format!(
                "spec {:?}: prior_from cannot name the spec itself ({parent_name:?}). Name a \
                 different parent spec in the same ModelBank.",
                child.name
            ));
        }
        let parent = &specs[pi];
        if !parent.model.has_linear_feature_coef() {
            return Err(format!(
                "spec {:?}: prior_from={parent_name:?} names a {} model, which does not fit \
                 linear feature coefficients. Point prior_from at a linear regression spec such \
                 as ewridge, rls, kalman, lasso, or huber.",
                child.name,
                parent.model.kind_name()
            ));
        }
        if derived[pi].n_models != 1 || derived[pi].nc != 1 {
            return Err(format!(
                "spec {:?}: prior_from={parent_name:?} has a grid of {} instances and {} combos, \
                 so its parent coefficient vector is ambiguous. Give {parent_name:?} a single \
                 half_life/lam and a single ridge/feature set.",
                child.name, derived[pi].n_models, derived[pi].nc
            ));
        }
        if parent.fit_intercept != child.fit_intercept {
            return Err(format!(
                "spec {:?}: prior_from={parent_name:?} has fit_intercept={}, but {:?} has \
                 fit_intercept={}. Set the same fit_intercept on both parent and child.",
                child.name, parent.fit_intercept, child.name, child.fit_intercept
            ));
        }
        if parent.features != child.features {
            return Err(format!(
                "spec {:?}: prior_from={parent_name:?} has features {:?}, which does not match \
                 {:?}'s features {:?}. Give parent and child the exact same features in the same \
                 order.",
                child.name, parent.features, child.name, child.features
            ));
        }
        if parent.targets != child.targets {
            return Err(format!(
                "spec {:?}: prior_from={parent_name:?} has targets {:?}, which does not match \
                 {:?}'s targets {:?}. Give parent and child the exact same targets in the same \
                 order.",
                child.name,
                parent.targets.as_slice(),
                child.name,
                child.targets.as_slice()
            ));
        }
        // Strict causality under embargo / label_clock: parent must never
        // release or learn a target earlier than the child does.
        check_delay_causality(child, parent)?;

        derived[si].prior_parent = Some(pi);
        derived[pi].is_prior_parent = true;
    }

    // Kahn's topological sort over both `prior_from` and `compare` edges.
    let mut indeg = vec![0usize; n];
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for si in 0..n {
        if let Some(pi) = derived[si].prior_parent {
            adj[pi].push(si);
            indeg[si] += 1;
        }
        if let Some((a, b)) = derived[si].compare {
            adj[a].push(si);
            indeg[si] += 1;
            if b != a {
                adj[b].push(si);
                indeg[si] += 1;
            }
        }
    }
    let mut layers: Vec<Vec<usize>> = Vec::new();
    let mut frontier: Vec<usize> = (0..n).filter(|&i| indeg[i] == 0).collect();
    let mut visited = 0usize;
    while !frontier.is_empty() {
        visited += frontier.len();
        let mut next = Vec::new();
        for &u in &frontier {
            for &v in &adj[u] {
                indeg[v] -= 1;
                if indeg[v] == 0 {
                    next.push(v);
                }
            }
        }
        next.sort_unstable();
        layers.push(frontier);
        frontier = next;
    }
    if visited < n {
        let stuck: Vec<&str> = (0..n)
            .filter(|&i| indeg[i] > 0)
            .map(|i| specs[i].name.as_str())
            .collect();
        return Err(format!(
            "specs: prior_from dependencies form a cycle involving {stuck:?}. Remove or redirect \
             one prior_from edge so the dependency graph is acyclic."
        ));
    }
    Ok(layers)
}

/// Enforce that `parent` cannot leak future targets into `child` when `child`
/// delays target availability via `embargo` or `label_clock`.
fn check_delay_causality(child: &Spec, parent: &Spec) -> Result<(), String> {
    if let Some(cl) = &child.label_clock {
        let same_label = parent
            .label_clock
            .as_ref()
            .is_some_and(|pl| pl.column() == cl.column());
        if !same_label || parent.clock != child.clock {
            return Err(format!(
                "spec {:?}: child uses label_clock={:?} (clock={:?}), but prior_from={:?} uses \
                 label_clock={:?} (clock={:?}), which would leak future targets into the child. \
                 Give both specs the same clock and label_clock.",
                child.name,
                cl.column(),
                child.clock,
                parent.name,
                parent.label_clock.as_ref().map(|l| l.column()),
                parent.clock
            ));
        }
    }
    if let Some(ce) = &child.embargo {
        let Some(pe) = &parent.embargo else {
            return Err(format!(
                "spec {:?}: child sets embargo={ce}, but prior_from={:?} has no embargo, which \
                 would leak future targets into the child. Give {:?} embargo >= {ce} on the same \
                 clock.",
                child.name, parent.name, parent.name
            ));
        };
        if parent.clock != child.clock {
            return Err(format!(
                "spec {:?}: child sets embargo={ce} on clock={:?}, but prior_from={:?} uses \
                 clock={:?}. Give parent and child the same clock column so their embargoes can \
                 be compared.",
                child.name, child.clock, parent.name, parent.clock
            ));
        }
        if pe.is_duration() != ce.is_duration() {
            return Err(format!(
                "spec {:?}: child embargo={ce} and prior_from={:?} embargo={pe} use incompatible \
                 units. Give both embargoes in the same units.",
                child.name, parent.name
            ));
        }
        if pe.value() < ce.value() {
            return Err(format!(
                "spec {:?}: child sets embargo={ce}, but prior_from={:?} has shorter \
                 embargo={pe}, which would leak future targets into the child. Increase {:?}'s \
                 embargo to at least {ce}.",
                child.name, parent.name, parent.name
            ));
        }
    }
    Ok(())
}

/// Extract a dynamic `coef_prior` `List(Float64)` column into layout-ordered
/// `PriorRows` (`pre` empty for strict zero lookahead, `post` populated).
pub(super) fn extract_prior_column(
    chunk: &ArrowChunk,
    spec: &Spec,
    scoring: bool,
    layout: Layout<'_>,
) -> PolarsResult<Option<PriorRows>> {
    let Some(col_name) = spec.model.coef_prior_column() else {
        return Ok(None);
    };
    let n = chunk.height();
    let m = spec.m();
    let k_total = spec.k() + usize::from(spec.fit_intercept);
    let width = m * k_total;
    if scoring && !chunk.has(col_name) {
        return Ok(None);
    }
    let arr = chunk.list_f64(spec, "coef_prior", col_name)?;
    let offsets = arr.offsets().as_slice();
    let values_any = arr.values().as_any();
    let values = values_any.downcast_ref::<Float64Array>().ok_or_else(|| {
        polars_err!(InvalidOperation:
            "spec {:?}: coef_prior column {:?} does not hold Float64 list elements. Cast the \
             column to List(Float64).",
            spec.name, col_name
        )
    })?;
    let val_slice = values.values().as_slice();
    let val_valid = values.validity();
    let list_valid = arr.validity();

    let mut post_chunk = vec![f64::NAN; n * width];
    for r in 0..n {
        if list_valid.is_some_and(|v| !v.get_bit(r)) {
            continue;
        }
        let start = offsets[r] as usize;
        let end = offsets[r + 1] as usize;
        let len = end - start;
        let dst = &mut post_chunk[r * width..(r + 1) * width];
        if len == width {
            for (i, slot) in dst.iter_mut().enumerate() {
                let idx = start + i;
                if val_valid.is_none_or(|v| v.get_bit(idx)) {
                    *slot = val_slice[idx];
                }
            }
        } else if len == k_total && m > 1 {
            // Broadcast a single k_total vector across all targets.
            for j in 0..m {
                for i in 0..k_total {
                    let idx = start + i;
                    if val_valid.is_none_or(|v| v.get_bit(idx)) {
                        dst[j * k_total + i] = val_slice[idx];
                    }
                }
            }
        } else {
            let expected = if m == 1 {
                format!("{k_total}")
            } else {
                format!("{k_total} or {width}")
            };
            polars_bail!(ShapeMismatch:
                "spec {:?}: coef_prior column {:?} has {} entries at row {} (counting from 0), \
                 but expected {} ({}). Provide one vector of length {} per row.",
                spec.name,
                col_name,
                len,
                chunk.rows().at(r),
                expected,
                if spec.fit_intercept {
                    "intercept first, then features"
                } else {
                    "one per feature"
                },
                expected
            );
        }
    }
    let post = match layout {
        None => post_chunk,
        Some(perm) => {
            let mut out = vec![f64::NAN; n * width];
            for (pos, &r) in perm.iter().enumerate() {
                out[pos * width..(pos + 1) * width]
                    .copy_from_slice(&post_chunk[r * width..(r + 1) * width]);
            }
            out
        }
    };
    Ok(Some(PriorRows {
        width,
        pre: Vec::new(),
        post,
    }))
}

/// Scatter a parent spec's per-task [`ParentCapture`] buffers into chunk-order
/// `(pre, post)` buffers (`n * width`).
pub(super) fn scatter_parent_captures(
    captures: &[ParentCapture],
    n: usize,
    width: usize,
) -> (Vec<f64>, Vec<f64>) {
    let mut pre = vec![f64::NAN; n * width];
    let mut post = vec![f64::NAN; n * width];
    for cap in captures {
        debug_assert_eq!(cap.width, width);
        for (ri, &row) in cap.rows.iter().enumerate() {
            let src_off = ri * width;
            let dst_off = row * width;
            pre[dst_off..dst_off + width].copy_from_slice(&cap.pre[src_off..src_off + width]);
            post[dst_off..dst_off + width].copy_from_slice(&cap.post[src_off..src_off + width]);
        }
    }
    (pre, post)
}

/// Gather chunk-order `(pre, post)` into a child spec's layout order.
pub(super) fn gather_prior_rows(
    pre_chunk: &[f64],
    post_chunk: &[f64],
    width: usize,
    layout: Layout<'_>,
) -> PriorRows {
    let n = pre_chunk.len() / width.max(1);
    match layout {
        None => PriorRows {
            width,
            pre: pre_chunk.to_vec(),
            post: post_chunk.to_vec(),
        },
        Some(perm) => {
            let mut pre = vec![f64::NAN; n * width];
            let mut post = if post_chunk.is_empty() {
                Vec::new()
            } else {
                vec![f64::NAN; n * width]
            };
            for (pos, &r) in perm.iter().enumerate() {
                let d = pos * width;
                let s = r * width;
                pre[d..d + width].copy_from_slice(&pre_chunk[s..s + width]);
                if !post.is_empty() {
                    post[d..d + width].copy_from_slice(&post_chunk[s..s + width]);
                }
            }
            PriorRows { width, pre, post }
        }
    }
}

/// For `&self` scoring (`Bank::predict`), collect a parent spec's live ready
/// coefficients for every row `0..n` in chunk order without mutating state.
pub(super) fn collect_live_parent_prior(
    parent_spec: &Spec,
    parent_states: &Streams,
    parent_groups: &Groups<'_>,
    upstream_pre: Option<&PriorRows>,
    n: usize,
) -> Vec<f64> {
    let m = parent_spec.m();
    let k_total = parent_spec.k() + usize::from(parent_spec.fit_intercept);
    let width = m * k_total;
    let mut pre = vec![f64::NAN; n * width];
    let min_settled = parent_spec.min_settled_frac_or_default();
    let mut tn = Vec::new();
    let mut coef_buf = vec![f64::NAN; width];
    let mut base = 0usize;
    for (key, rows) in parent_groups.iter() {
        if let Some(at) = place_of(parent_states, key) {
            let stream = &parent_states[at];
            if let Some((_, model)) = stream.first_model() {
                let settled = stream.first_settled();
                let up = upstream_pre.and_then(|u| {
                    if u.pre.is_empty() {
                        None
                    } else {
                        u.pre.get(base * width..(base + 1) * width)
                    }
                });
                write_ready_coefs(
                    model,
                    m,
                    k_total,
                    stream.min_weight_floors(),
                    min_settled,
                    settled,
                    up,
                    &mut tn,
                    &mut coef_buf,
                );
                for &r in rows {
                    pre[r * width..(r + 1) * width].copy_from_slice(&coef_buf);
                }
            }
        } else if let Some(up_rows) = upstream_pre
            && !up_rows.pre.is_empty()
            && parent_spec.model.has_dynamic_prior()
            && parent_spec.min_periods_or_default() == 0.0
        {
            // Unseen intermediate parent in a 3-level hierarchy forwards its
            // own upstream parent's live coefficients.
            for (ri, &r) in rows.iter().enumerate() {
                let s = (base + ri) * width;
                pre[r * width..(r + 1) * width].copy_from_slice(&up_rows.pre[s..s + width]);
            }
        }
        base += rows.len();
    }
    pre
}
