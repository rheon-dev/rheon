//! Hierarchical dynamic-prior capture (parent spec) and view (child spec)
//! types for [`Stream`](super::Stream).
//!
//! Kept in its own submodule so `stream.rs` stays well below the repository
//! file-size cap and non-hierarchical streams pay zero extra code in their hot
//! paths.

use super::AnyModel;

/// Per-row dynamic priors in a spec's layout order (`n_rows * width` floats).
///
/// - Under `prior_from`, both `pre` (parent coefficients strictly before row
///   `t`) and `post` (parent coefficients after row `t`, for solving `beta(t)`
///   to predict row `t+1`) are populated.
/// - Under a dynamic `coef_prior` column, `pre` is empty (ensuring strict zero
///   lookahead even when an upstream spec's post-row `coef` column is passed
///   directly) and `post` holds the column values.
#[derive(Debug, Clone)]
pub struct PriorRows {
    /// Floats per row: `n_targets * (n_features + fit_intercept)`.
    pub width: usize,
    /// Pre-row parent coefficients (`n * width`, or empty in column mode).
    pub pre: Vec<f64>,
    /// Post-row coefficients (`n * width`, or empty in scoring-only mode).
    pub post: Vec<f64>,
}

impl PriorRows {
    /// Borrow a view starting at `base` in layout order.
    #[inline]
    pub fn view(&self, base: usize) -> PriorView<'_> {
        PriorView {
            width: self.width,
            base,
            pre: &self.pre,
            post: &self.post,
        }
    }
}

/// Borrowed slice view of [`PriorRows`] for one stream run starting at `base`.
#[derive(Debug, Clone, Copy)]
pub struct PriorView<'a> {
    pub width: usize,
    pub base: usize,
    pub pre: &'a [f64],
    pub post: &'a [f64],
}

impl<'a> PriorView<'a> {
    /// Pre-row slice for run-local row `ri`, if available.
    #[inline]
    pub fn pre_at(&self, ri: usize) -> Option<&'a [f64]> {
        if self.pre.is_empty() {
            return None;
        }
        let start = (self.base + ri) * self.width;
        self.pre.get(start..start + self.width)
    }

    /// Post-row slice for run-local row `ri`, if available.
    #[inline]
    pub fn post_at(&self, ri: usize) -> Option<&'a [f64]> {
        if self.post.is_empty() {
            return None;
        }
        let start = (self.base + ri) * self.width;
        self.post.get(start..start + self.width)
    }
}

/// Per-run parent coefficient capture buffers, written by `run_instance` when
/// the spec is referenced by a downstream child's `prior_from`.
#[derive(Debug, Default)]
pub struct ParentCapture {
    /// Absolute chunk row indices matching `pre`/`post`.
    pub rows: Vec<usize>,
    /// Floats per row: `n_targets * k_total`.
    pub width: usize,
    /// Pre-row parent coefficients (`rows.len() * width`), `NaN` where unavailable.
    pub pre: Vec<f64>,
    /// Post-row parent coefficients (`rows.len() * width`), `NaN` where unavailable.
    pub post: Vec<f64>,
}

impl ParentCapture {
    pub fn new(width: usize) -> Self {
        Self {
            rows: Vec::new(),
            width,
            pre: Vec::new(),
            post: Vec::new(),
        }
    }

    /// Reserve space for a run of `n_rows` rows initialized to `NaN`, returning
    /// the starting offset (`row_start`) into `self.pre`/`self.post` (in rows).
    pub fn extend_run(&mut self, rows: &[usize]) -> usize {
        let start = self.rows.len();
        self.rows.extend_from_slice(rows);
        let add = rows.len() * self.width;
        self.pre.resize(self.pre.len() + add, f64::NAN);
        self.post.resize(self.post.len() + add, f64::NAN);
        start
    }

    /// Slice for run-local `ri` inside a run that began at `row_start`.
    #[inline]
    pub fn pre_mut(&mut self, row_start: usize, ri: usize) -> &mut [f64] {
        let off = (row_start + ri) * self.width;
        &mut self.pre[off..off + self.width]
    }

    /// Slice for run-local `ri` inside a run that began at `row_start`.
    #[inline]
    pub fn post_mut(&mut self, row_start: usize, ri: usize) -> &mut [f64] {
        let off = (row_start + ri) * self.width;
        &mut self.post[off..off + self.width]
    }
}

/// Write a parent model's ready coefficients into `dst` (`m * k_total`),
/// leaving any unready, warming, or non-finite target slice as `NaN`.
///
/// For a 3-level hierarchy (`global -> sector -> ticker`), if an intermediate
/// `ewridge` parent is itself at cold start (`w_j == 0`) with `min_weight == 0`
/// and holds a valid finite upstream `prior_pre` for target `j`, it forwards
/// its cold-start prior so the grandchild inherits it causally on row 0 too.
#[allow(clippy::too_many_arguments)]
pub(crate) fn write_ready_coefs(
    model: &AnyModel,
    m: usize,
    k_total: usize,
    min_weight: &[f64],
    min_settled: f64,
    settled: f64,
    upstream_pre: Option<&[f64]>,
    tn_buf: &mut Vec<f64>,
    dst: &mut [f64],
) {
    dst.fill(f64::NAN);
    debug_assert_eq!(dst.len(), m * k_total);
    if min_settled > 0.0 && settled.is_finite() && settled < min_settled {
        return;
    }
    let own_weights = model.target_n_eff_into(tn_buf);
    let shared_w = model.n_eff();
    let coefs = model.coefficients();
    for j in 0..m {
        let w_j = if own_weights {
            tn_buf.get(j).copied().unwrap_or(shared_w)
        } else {
            shared_w
        };
        let floor_j = min_weight.get(j).copied().unwrap_or(0.0);
        if w_j < floor_j {
            continue;
        }
        let target_dst = &mut dst[j * k_total..(j + 1) * k_total];
        if w_j > 0.0 {
            if let Some(all) = coefs.as_deref()
                && let Some(slot) = all.get(j)
                && slot.len() == k_total
                && slot.iter().all(|v| v.is_finite())
            {
                target_dst.copy_from_slice(slot);
            }
        } else if let Some(up) = upstream_pre
            && let Some(slot) = up.get(j * k_total..(j + 1) * k_total)
            && slot.iter().all(|v| v.is_finite())
            && let AnyModel::EwRidge(ew) = model
            && ew.cfg().prior_weight.is_some()
            && (!ew.cfg().fit_intercept || ew.cfg().prior_include_intercept)
        {
            // Forward cold-start prior through a multi-level hierarchy.
            target_dst.copy_from_slice(slot);
        }
    }
}
