//! The rows a closed group leaves behind and the frames they become: the
//! `Gram` and `PairRow` a group's moments are read as, the `ClosedRow` a
//! closed group is emitted as, its checks against the specs, and the frame it
//! is built into (moved out of `bank.rs` under the 250 KB cap a source file is
//! held to).

#[allow(unused_imports)]
use super::*;

/// The axes a spec's [`Gram`] is indexed by: its columns (the features,
/// with the intercept in front when the spec has one) and its targets. The
/// Python `ModelBank.gram` builds the same two lists from the spec dict, and
/// `tests/test_closed_groups.py` holds the two to each other.
pub(crate) fn gram_axes(spec: &Spec) -> (Vec<String>, Vec<String>) {
    // `ew_cov` accumulates over the features alone -- no target, and so no
    // constant column to regress one on. It is the only model without a
    // target that keeps a Gram (review 2026-09-12, D7).
    let unsupervised = matches!(spec.model, ModelKind::EwCov { .. });
    let mut columns = spec.features.clone();
    if spec.fit_intercept && !unsupervised {
        columns.insert(0, "intercept".to_string());
    }
    let targets = if unsupervised {
        Vec::new()
    } else {
        spec.targets.to_vec()
    };
    (columns, targets)
}

/// The upper triangle of a row-major `k * k` symmetric matrix, with the
/// diagonal, row by row: `k(k+1)/2` numbers instead of `k^2`, which is what
/// the closed row carries (`rheon.gram.from_row` expands it). A co-moment
/// matrix is symmetric to 1e-16 and not to the bit (E48), so the half that
/// is kept is named: the upper one. `ModelBank.gram(layout="packed")`
/// packs the same way (docs/PLAN.md task 229).
pub fn vech(m: &[f64], k: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(k * (k + 1) / 2);
    for i in 0..k {
        for j in i..k {
            out.push(m[i * k + j]);
        }
    }
    out
}

/// One model instance's [`Gram`]s: one per Gram its fit reads (several only
/// under `target_gaps = "own_rows"`, task 81), none for a kind that keeps no
/// co-moment matrix (`rls` and `kalman` track an inverse).
///
/// The one builder behind both [`Bank::gram`] and a [`ClosedRow`]'s Gram
/// block, so "a closed row equals `gram()` read at the same point" holds by
/// construction and the test is the tripwire that keeps it so.
pub(crate) fn gram_of(key: &GroupKey, label: &str, model: &AnyModel) -> Vec<Gram> {
    // Under a `window` every number here is the window's -- the accumulators
    // the fit `coef` reports was solved from -- so the Gram solves to that fit;
    // it read the live ones, so a windowed spec's Gram and closed row carried
    // the whole history beside a `coef` solved on the window (review
    // 2026-09-12, S19). The target moments are the window's too (task 136):
    // the snapshots carry them, and a snapshot written before they did gives
    // `None`, "this state cannot say", until the ring rolls over.
    let (parts, tm) = match model {
        AnyModel::EwRidge(m) => m.gram_parts(),
        AnyModel::Lasso(m) => m.gram_parts(),
        AnyModel::EwCov(m) => {
            // No targets: the matrix is the whole output, and empty target
            // moments say so, where `None` would say "cannot tell".
            let cov = m.windowed_cov();
            let cov = cov.flushed();
            let lag = m.lag();
            return vec![Gram {
                group: key.clone(),
                instance: label.to_string(),
                k: cov.k(),
                n_eff: cov.n_eff(),
                n_kish: cov.n_kish(),
                means: cov.means().to_vec(),
                comoments: cov.comoments().to_full(),
                targets: Vec::new(),
                cross_moments: Vec::new(),
                means_by_target: Vec::new(),
                cross_centred: Vec::new(),
                target_means: Some(Vec::new()),
                target_vars: Some(Vec::new()),
                target_n_kish: Some(Vec::new()),
                target_weights: Vec::new(),
                lags: lag.map(|l| l.lags().to_vec()),
                lag_comoments: lag.map(|l| l.comoments().to_vec()),
            }];
        }
        _ => return Vec::new(),
    };
    parts
        .into_iter()
        .map(|p| {
            let pick = |v: &[f64]| p.targets.iter().map(|&j| v[j]).collect::<Vec<f64>>();
            // Kish's count from the part's own target weights -- the
            // window's under a window, which the moments are -- over `Q_t`.
            let kish = tm.as_ref().map(|t| {
                p.targets
                    .iter()
                    .zip(&p.target_weights)
                    .map(|(&j, &w)| (t.q()[j] > 0.0).then(|| w * w / t.q()[j]))
                    .collect::<Vec<Option<f64>>>()
            });
            Gram {
                group: key.clone(),
                instance: label.to_string(),
                k: p.cov.k(),
                n_eff: p.cov.n_eff(),
                n_kish: p.cov.n_kish(),
                means: p.cov.means().to_vec(),
                comoments: p.cov.comoments().to_full(),
                target_means: tm.as_ref().map(|t| pick(t.means())),
                target_vars: tm.as_ref().map(|t| pick(t.vars())),
                target_n_kish: kish,
                cross_moments: p.cross_moments,
                means_by_target: p.means_by_target,
                cross_centred: p.cross_centred,
                target_weights: p.target_weights,
                targets: p.targets,
                lags: None,
                lag_comoments: None,
            }
        })
        .collect()
}

/// One instance's EW accumulators, as returned by [`Bank::gram`].
///
/// Values are in the features' original units. `comoments` is **centered**
/// (E11b, which is what makes it accurate at large offsets). `cross_moments`
/// is **uncentered**, and `cross_centred` is the same centred at each
/// target's means, which is what `rheon.gram.solve` reads: re-centring the raw
/// form kept `L²·ε` of it at a level `L` (review 2026-09-12, N4). The raw
/// form is bridged to the fit by one identity, and getting it wrong is a
/// silent wrong answer rather than an error:
///
/// ```text
/// raw[i][j] = comoments[i*k+j] + means[i]*means[j]
/// raw · beta[t] = cross_moments[t]        (up to the ridge term)
/// ```
///
/// The intercept, when the spec has one, is column 0: a constant 1, so it has
/// zero variance in `comoments` and `raw[0][j] == means[j]`.
///
/// The model itself keeps its cross-moments centred and solves the centred
/// system (review 2026-09-12, N1), so at a level `L` the raw ones here carry
/// `L²·ε` of rounding that its own fit does not.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Gram {
    pub group: GroupKey,
    /// The decay instance's suffix (`"@h500"`, or `""` for a single instance).
    pub instance: String,
    /// Columns, including the intercept when the spec has one.
    pub k: usize,
    /// Accumulated weight behind these moments.
    pub n_eff: f64,
    /// Kish's effective sample size over the feature rows, `n_eff^2 / Sum w^2`
    /// (docs/ENHANCEMENTS.md E45): the number of equally weighted rows these
    /// moments are worth, which is what a standard error divides by. `None`
    /// before the first row and for a state written before task 38.
    pub n_kish: Option<f64>,
    /// EW column means, length `k`.
    pub means: Vec<f64>,
    /// Centered co-moments, row-major `k*k`.
    pub comoments: Vec<f64>,
    /// The targets fitted from these moments (indices into the spec's), in
    /// the order of the per-target fields: all, or under `target_gaps =
    /// "own_rows"` the ones present on this Gram's rows (PLAN task 81).
    pub targets: Vec<usize>,
    /// Per-target uncentered cross-moments, each `k` long. Empty for
    /// `ew_cov`.
    pub cross_moments: Vec<Vec<f64>>,
    /// Per target, the column means over its rows, which its fit is centred
    /// at: `means` under `"own_rows"`, its own under `"pairwise"`.
    pub means_by_target: Vec<Vec<f64>>,
    /// Per target, its cross-moments centred at [`Self::means_by_target`] and
    /// its own mean: what the fit is solved from, and what `rheon.gram.solve`
    /// reads. `cross_moments` less `m_t·ȳ_t` in exact arithmetic; formed in
    /// floating point, that difference keeps `L²·ε` of it at a level `L`
    /// (review 2026-09-12, N4). Empty for `ew_cov`.
    pub cross_centred: Vec<Vec<f64>>,
    /// Per-target accumulated weight. Empty for `ew_cov`.
    pub target_weights: Vec<f64>,
    /// Per-target EW mean of the target. Empty for `ew_cov` (no targets);
    /// `None` for a state written before task 38.
    pub target_means: Option<Vec<f64>>,
    /// Per-target EW centered variance of the target, alongside
    /// [`Self::target_means`].
    pub target_vars: Option<Vec<f64>>,
    /// Per-target Kish effective sample size, `W_t^2 / Q_t`; an entry is
    /// `None` for a target that has not seen a weighted row.
    pub target_n_kish: Option<Vec<Option<f64>>>,
    /// The lags an `ew_cov(lags = ...)` accumulates at, in output order, and
    /// their cross-moments as `L * k * k` row-major within a lag
    /// (docs/ENHANCEMENTS.md E56). `None` for a spec without lags.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lags: Option<Vec<usize>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lag_comoments: Option<Vec<f64>>,
}

/// One (feature, target) pair of a closed `marginal` group, the columns
/// [`Bank::marginal`] reports for it: the linear block always, the lag
/// block (E66) where the spec asked for `lags`, the binned block (E67)
/// where it asked for `bins` or `bin_edges`. The lists are empty and the
/// scalars NaN where the spec did not ask, and a row written before the two
/// blocks existed reads back the same way (bank files are named msgpack).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PairRow {
    pub feature: String,
    pub target: String,
    pub n_eff: f64,
    pub n_kish: f64,
    pub mean_x: f64,
    pub var_x: f64,
    pub mean_y: f64,
    pub var_y: f64,
    pub cov: f64,
    pub corr: f64,
    pub beta: f64,
    /// The correlation's t-statistic, `t_stat` in `marginal()` (review round
    /// 4, CD13: `t` read as every example's clock column).
    pub t_stat: f64,
    #[serde(default)]
    pub lag_corr_xx: Vec<f64>,
    #[serde(default)]
    pub lag_corr_yy: Vec<f64>,
    #[serde(default)]
    pub lag_corr_xy: Vec<f64>,
    #[serde(default)]
    pub lag_corr_yx: Vec<f64>,
    #[serde(default = "nan")]
    pub n_serial: f64,
    #[serde(default = "nan")]
    pub t_serial: f64,
    #[serde(default = "nan")]
    pub phi_x: f64,
    #[serde(default = "nan")]
    pub phi_y: f64,
    #[serde(default)]
    pub bin_edges: Vec<f64>,
    #[serde(default)]
    pub bin_n: Vec<f64>,
    #[serde(default)]
    pub bin_mean_y: Vec<f64>,
    #[serde(default)]
    pub bin_var_y: Vec<f64>,
    #[serde(default = "nan")]
    pub split_gain: f64,
    #[serde(default = "nan")]
    pub split_at: f64,
    #[serde(default = "nan")]
    pub split_gain_t: f64,
}

/// The core's "not defined", for a field a row written before it existed
/// does not carry.
fn nan() -> f64 {
    f64::NAN
}

/// One group's accumulators at the moment the bank could prove that no
/// further row would join it (docs/ENHANCEMENTS.md E54), one per decay
/// instance.
///
/// The row is the state *as it stood*, so it equals the [`Bank::gram`] a
/// driver would have read at the same point, field for field and bit for
/// bit -- both are built by [`gram_of`], which is what keeps that true. The
/// stream is dropped once the row is made, which is the point: a bank over
/// an unbounded key space stays bounded.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClosedRow {
    /// Index of the spec in the bank; the frame carries its name.
    pub spec: usize,
    pub group: GroupKey,
    /// The decay instance's suffix (`"@h500"`, or `""` for one instance).
    pub instance: String,
    /// The session value of the span just closed, under `group_close =
    /// "session"`; `None` under `"monotone"`, and for a null session value.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session: Option<String>,
    pub n_eff: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n_kish: Option<f64>,
    /// Rows fed and rows learned from over the span, from the stream's own
    /// [`DataSummary`]; `None` only for a stream restored from a file
    /// written before task 35's summary existed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rows_fed: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rows_learned: Option<u64>,
    /// The span's clock range as its stream kept it (review round 4, N18):
    /// the frame gives it in the clock column's own dtype, exactly.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub clock_min: Option<ClockValue>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub clock_max: Option<ClockValue>,
    /// The accumulators, for a kind that keeps them.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gram: Option<Gram>,
    /// The flat `coef` list as of the last solve, for a kind that reports
    /// one; the exact solve on the closed Gram is
    /// `rheon.gram.solve(rheon.gram.from_row(row))`. NaN in a slot no solve has
    /// fit (review round 4, CC1), so the export tags it.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        with = "rheon_core::humanfloat::opt_vec_f64_or_tag"
    )]
    pub coef: Option<Vec<f64>>,
    /// `ew_cov(pca = r)`: the eigenvalues of this row's own `comoments`,
    /// descending, and the loadings row-major `r * k`. Filled by the bank
    /// as the row is queued, so the sign continuity is a function of queue
    /// order alone.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub eig_vals: Option<Vec<f64>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub eig_vecs: Option<Vec<f64>>,
    /// A `marginal` spec's pairs, in [`Bank::marginal`] order.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub pairs: Vec<PairRow>,
    /// An `rcov` spec's block (docs/ENHANCEMENTS.md E57): the realised
    /// covariance and correlation as `vech` of the upper triangle, the
    /// effective return count, the estimator, and the noise diagnostics
    /// behind an automatic bandwidth.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rcov: Option<RcovRow>,
    /// Position of the closing row within the chunk that closed it: the
    /// first row of the new session, or the first row of the greater key.
    /// The queue's order, and not part of the row itself.
    #[serde(skip)]
    pub at: usize,
}

/// A `list[f64]` column: one list per row, `None` where the row has none,
/// and null inside for a value that is not finite -- `coef`'s rule.
fn list_f64(name: &str, rows: &[ClosedRow], f: impl Fn(&ClosedRow) -> Option<Vec<f64>>) -> Column {
    list_f64_by(name, rows, f64::is_finite, f)
}

/// [`list_f64`] under `marginal()`'s rule: null inside only for NaN, the
/// core's "undefined" (a constant column's `corr`), while `±inf` stays --
/// `t` on a perfect fit is `±inf` there and must be `±inf` here.
fn pair_f64(name: &str, rows: &[ClosedRow], f: impl Fn(&ClosedRow) -> Option<Vec<f64>>) -> Column {
    list_f64_by(name, rows, |x| !x.is_nan(), f)
}

fn list_f64_by(
    name: &str,
    rows: &[ClosedRow],
    keep: impl Fn(f64) -> bool,
    f: impl Fn(&ClosedRow) -> Option<Vec<f64>>,
) -> Column {
    let mut b = ListPrimitiveChunkedBuilder::<Float64Type>::new(
        name.into(),
        rows.len(),
        8,
        DataType::Float64,
    );
    for r in rows {
        match f(r) {
            Some(v) => b.append_iter(v.iter().map(|x| keep(*x).then_some(*x))),
            None => b.append_null(),
        }
    }
    b.finish().into_series().into()
}

/// A `list[str]` column, as [`list_f64`].
fn list_str(
    name: &str,
    rows: &[ClosedRow],
    f: impl Fn(&ClosedRow) -> Option<Vec<String>>,
) -> Column {
    let mut b = ListStringChunkedBuilder::new(name.into(), rows.len(), 8);
    for r in rows {
        match f(r) {
            Some(v) => b.append_values_iter(v.iter().map(String::as_str)),
            None => b.append_null(),
        }
    }
    b.finish().into_series().into()
}

/// A `list[list[f64]]` column: one list per row with one inner list per
/// pair, `None` where the row has no such block, and null inside an inner
/// list for NaN, as [`pair_f64`]. This is how a closed `marginal` row
/// carries what [`Bank::marginal`] reports as a list column per pair -- the
/// lag correlations and the bin histogram, whose inner lengths differ from
/// pair to pair by design.
fn list_list_f64<'a>(
    name: &str,
    rows: &'a [ClosedRow],
    f: impl Fn(&'a ClosedRow) -> Option<Vec<&'a [f64]>>,
) -> PolarsResult<Column> {
    let inner = DataType::List(Box::new(DataType::Float64));
    let mut b = AnonymousOwnedListBuilder::new(name.into(), rows.len(), Some(inner));
    for r in rows {
        match f(r) {
            Some(lists) => {
                let mut ib = ListPrimitiveChunkedBuilder::<Float64Type>::new(
                    PlSmallStr::EMPTY,
                    lists.len(),
                    8,
                    DataType::Float64,
                );
                for v in lists {
                    ib.append_iter(v.iter().map(|x| (!x.is_nan()).then_some(*x)));
                }
                b.append_series(&ib.finish().into_series())?;
            }
            None => b.append_null(),
        }
    }
    Ok(b.finish().into_series().into())
}

/// Whether a `marginal` spec asked for the lag block (E66).
pub(super) fn spec_lags(s: &Spec) -> bool {
    matches!(&s.model, ModelKind::Marginal { lags: Some(l), .. } if !l.is_empty())
}

/// Whether a `marginal` spec with lags keeps a cross term at any of them:
/// every lag unless `cross_lags` names fewer, and none under `[]` (E70).
pub(super) fn spec_cross_lags(s: &Spec) -> bool {
    spec_lags(s)
        && !matches!(
            &s.model,
            ModelKind::Marginal { cross_lags: Some(c), .. } if c.is_empty()
        )
}

/// Whether a `marginal` spec asked for the binned block (E67), learned or
/// given.
pub(super) fn spec_bins(s: &Spec) -> bool {
    matches!(
        &s.model,
        ModelKind::Marginal { bins: Some(_), .. }
            | ModelKind::Marginal {
                bin_edges: Some(_),
                ..
            }
    )
}

/// Whether a closed row read from a state file fits the bank's specs:
/// everything [`closed_frame`] reads it by -- its spec, which must close
/// groups, the targets its Gram names, and the Gram's shape, `k` columns of
/// the spec's and one entry per named target in every per-target list.
/// Checked at load, so a damaged file is refused there rather than loaded,
/// saved again and found by a panic at the first drain (task 160, PA2).
pub(super) fn check_closed_row(specs: &[Spec], r: &ClosedRow) -> Result<(), String> {
    let damaged = |what: String| {
        Err(format!(
            "state file: it is damaged, since a closed row of group {} {what}. Load an undamaged copy of the state file, or rebuild it from its input.",
            r.group
        ))
    };
    let Some(spec) = specs.get(r.spec) else {
        return damaged(format!(
            "belongs to spec {} of a bank of {}",
            r.spec,
            specs.len()
        ));
    };
    if spec.group_close.is_none() {
        return damaged(format!(
            "belongs to spec {:?}, which closes no group",
            spec.name
        ));
    }
    let Some(g) = &r.gram else {
        return Ok(());
    };
    if !matches!(
        spec.model,
        ModelKind::EwRidge { .. } | ModelKind::Lasso { .. } | ModelKind::EwCov { .. }
    ) {
        return damaged(format!(
            "carries a Gram, and spec {:?} keeps none",
            spec.name
        ));
    }
    let (columns, targets) = gram_axes(spec);
    let k = columns.len();
    if g.k != k || g.means.len() != k || g.comoments.len() != k * k {
        return damaged(format!(
            "has a Gram of {} columns with {} means and {} co-moments, where spec {:?} has \
             {k} columns",
            g.k,
            g.means.len(),
            g.comoments.len(),
            spec.name
        ));
    }
    if let Some(j) = g.targets.iter().find(|&&j| j >= targets.len()) {
        return damaged(format!(
            "names target {j} of spec {:?}, which has {}",
            spec.name,
            targets.len()
        ));
    }
    let m = g.targets.len();
    let lists_fit = [&g.cross_moments, &g.means_by_target, &g.cross_centred]
        .iter()
        .all(|l| l.len() == m && l.iter().all(|v| v.len() == k))
        && g.target_weights.len() == m
        && g.target_means.as_ref().is_none_or(|v| v.len() == m)
        && g.target_vars.as_ref().is_none_or(|v| v.len() == m)
        && g.target_n_kish.as_ref().is_none_or(|v| v.len() == m);
    if !lists_fit {
        return damaged(format!(
            "names {m} targets and carries per-target lists of another length"
        ));
    }
    match (&g.lags, &g.lag_comoments) {
        (None, None) => Ok(()),
        (Some(l), Some(c)) if c.len() == l.len() * k * k => Ok(()),
        _ => damaged(format!(
            "carries lag co-moments that are not one {k} x {k} block per lag"
        )),
    }
}

/// The decay of `spec`'s instance named by its field suffix (`"@h500"`, or
/// `""` for one instance) as `output_index` gives it: `(half_life, lam)`,
/// both `None` for a model that takes no decay (`seqtest`, `rcov`,
/// `corrchange`, `bocpd`, `audit`), and an infinite half-life as `inf`.
pub(crate) fn instance_decay(spec: &Spec, instance: &str) -> (Option<f64>, Option<f64>) {
    if spec.half_life.is_none() && spec.lam.is_none() {
        return (None, None);
    }
    let decays = spec.decays().unwrap_or_default();
    match decays.iter().find(|(s, _)| s == instance).map(|(_, d)| d) {
        Some(rheon_core::Decay::Halflife(h)) => (Some(*h), None),
        Some(rheon_core::Decay::Lam(l)) => (None, Some(*l)),
        None => (None, None),
    }
}

/// The frame [`Bank::closed_groups`] returns: the common columns always,
/// and a kind's block when **any** spec of the bank closes groups and is of
/// that kind, null on the rows of other kinds (docs/ENHANCEMENTS.md E54).
///
/// One schema per bank, not per call: a driver concatenating a run's drains
/// needs every frame to have the same columns, whatever happened to close in
/// that chunk. The clock range is in `clock`'s dtype, the clock column's own
/// ([`ClockDtype::column`], review round 4, N18).
pub(super) fn closed_frame(
    specs: &[Spec],
    rows: &[ClosedRow],
    clock: Option<&ClockDtype>,
) -> PolarsResult<DataFrame> {
    let closing: Vec<&Spec> = specs.iter().filter(|s| s.group_close.is_some()).collect();
    let any = |f: fn(&Spec) -> bool| closing.iter().any(|s| f(s));
    let has_gram = any(|s| {
        matches!(
            s.model,
            ModelKind::EwRidge { .. } | ModelKind::Lasso { .. } | ModelKind::EwCov { .. }
        )
    });
    let has_pca = any(|s| matches!(s.model, ModelKind::EwCov { pca: Some(_), .. }));
    let has_pairs = any(|s| matches!(s.model, ModelKind::Marginal { .. }));
    let has_lags = any(spec_lags);
    let has_cross_lags = any(spec_cross_lags);
    let has_bins = any(spec_bins);
    let has_rcov = any(|s| matches!(s.model, ModelKind::Rcov { .. }));
    let opt = |v: f64| v.is_finite().then_some(v);

    let mut cols: Vec<Column> = vec![
        Column::new(
            "spec".into(),
            rows.iter()
                .map(|r| specs[r.spec].name.as_str())
                .collect::<Vec<_>>(),
        ),
        Column::new(
            "group".into(),
            rows.iter().map(|r| r.group.as_str()).collect::<Vec<_>>(),
        ),
        Column::new(
            "instance".into(),
            rows.iter().map(|r| r.instance.as_str()).collect::<Vec<_>>(),
        ),
        // The instance's decay beside its suffix, as `output_index` and
        // `coef()` give it (task 244).
        Column::new(
            "half_life".into(),
            rows.iter()
                .map(|r| instance_decay(&specs[r.spec], &r.instance).0)
                .collect::<Vec<_>>(),
        ),
        Column::new(
            "lam".into(),
            rows.iter()
                .map(|r| instance_decay(&specs[r.spec], &r.instance).1)
                .collect::<Vec<_>>(),
        ),
        Column::new(
            "session".into(),
            rows.iter()
                .map(|r| r.session.as_deref())
                .collect::<Vec<_>>(),
        ),
        Column::new(
            "weight_sum".into(),
            rows.iter().map(|r| opt(r.n_eff)).collect::<Vec<_>>(),
        ),
        Column::new(
            "n_kish".into(),
            rows.iter()
                .map(|r| r.n_kish.and_then(opt))
                .collect::<Vec<_>>(),
        ),
        // An `Int64`, as every count of every frame (task 244): a count
        // saturated at the top of a `u64` reads `i64::MAX`, never -1
        // (`summary::count`; review round 4, PA3).
        counts("rows_fed", rows.iter().map(|r| r.rows_fed)),
        counts("rows_learned", rows.iter().map(|r| r.rows_learned)),
        ClockDtype::column(
            clock,
            "clock_min",
            &rows.iter().map(|r| r.clock_min).collect::<Vec<_>>(),
        )?,
        ClockDtype::column(
            clock,
            "clock_max",
            &rows.iter().map(|r| r.clock_max).collect::<Vec<_>>(),
        )?,
    ];
    if has_gram {
        let axes = |r: &ClosedRow| gram_axes(&specs[r.spec]);
        cols.push(list_str("columns", rows, |r| {
            r.gram.as_ref().map(|_| axes(r).0)
        }));
        cols.push(list_f64("means", rows, |r| {
            r.gram.as_ref().map(|g| g.means.clone())
        }));
        cols.push(list_f64("comoments", rows, |r| {
            r.gram.as_ref().map(|g| vech(&g.comoments, g.k))
        }));
        cols.push(list_str("targets", rows, |r| {
            r.gram.as_ref().map(|g| {
                let names = axes(r).1;
                g.targets.iter().map(|&j| names[j].clone()).collect()
            })
        }));
        cols.push(list_f64("target_means", rows, |r| {
            r.gram.as_ref().and_then(|g| g.target_means.clone())
        }));
        cols.push(list_f64("target_vars", rows, |r| {
            r.gram.as_ref().and_then(|g| g.target_vars.clone())
        }));
        cols.push(list_f64("target_weights", rows, |r| {
            r.gram.as_ref().map(|g| g.target_weights.clone())
        }));
        cols.push(list_f64("target_n_kish", rows, |r| {
            r.gram.as_ref().and_then(|g| {
                g.target_n_kish
                    .as_ref()
                    .map(|v| v.iter().map(|x| x.unwrap_or(f64::NAN)).collect())
            })
        }));
        cols.push(list_f64("cross_moments", rows, |r| {
            r.gram
                .as_ref()
                .map(|g| g.cross_moments.iter().flatten().copied().collect())
        }));
        cols.push(list_f64("means_by_target", rows, |r| {
            r.gram
                .as_ref()
                .map(|g| g.means_by_target.iter().flatten().copied().collect())
        }));
        cols.push(list_f64("cross_centered", rows, |r| {
            r.gram
                .as_ref()
                .map(|g| g.cross_centred.iter().flatten().copied().collect())
        }));
    }
    if any(|s| matches!(s.model, ModelKind::EwCov { lags: Some(_), .. })) {
        let mut b = ListPrimitiveChunkedBuilder::<Int64Type>::new(
            "lags".into(),
            rows.len(),
            4,
            DataType::Int64,
        );
        for r in rows {
            match r.gram.as_ref().and_then(|g| g.lags.as_ref()) {
                Some(v) => b.append_values_iter(v.iter().map(|&l| l as i64)),
                None => b.append_null(),
            }
        }
        cols.push(b.finish().into_series().into());
        cols.push(list_f64("lag_comoments", rows, |r| {
            r.gram.as_ref().and_then(|g| g.lag_comoments.clone())
        }));
    }
    cols.push(list_f64("coef", rows, |r| r.coef.clone()));
    if has_pca {
        cols.push(list_f64("eig_vals", rows, |r| r.eig_vals.clone()));
        cols.push(list_f64("eig_vecs", rows, |r| r.eig_vecs.clone()));
    }
    if has_pairs {
        let empty = |r: &ClosedRow| r.pairs.is_empty();
        cols.push(list_str("pair_feature", rows, |r| {
            (!empty(r)).then(|| r.pairs.iter().map(|p| p.feature.clone()).collect())
        }));
        cols.push(list_str("pair_target", rows, |r| {
            (!empty(r)).then(|| r.pairs.iter().map(|p| p.target.clone()).collect())
        }));
        for (name, f) in [
            (
                "pair_weight_sum",
                (|p: &PairRow| p.n_eff) as fn(&PairRow) -> f64,
            ),
            ("pair_n_kish", |p| p.n_kish),
            ("pair_mean_x", |p| p.mean_x),
            ("pair_var_x", |p| p.var_x),
            ("pair_mean_y", |p| p.mean_y),
            ("pair_var_y", |p| p.var_y),
            ("pair_cov", |p| p.cov),
            ("pair_corr", |p| p.corr),
            ("pair_beta", |p| p.beta),
            ("pair_t_stat", |p| p.t_stat),
        ] {
            cols.push(pair_f64(name, rows, move |r| {
                (!empty(r)).then(|| r.pairs.iter().map(f).collect())
            }));
        }
        // The two optional blocks, in `marginal()`'s order and under its
        // rule: present when any closing spec asked, and on a row whose
        // spec did not ask, null -- the column `marginal()` would not have.
        let asked = |r: &ClosedRow, f: fn(&Spec) -> bool| !empty(r) && f(&specs[r.spec]);
        if has_lags {
            for (name, f) in [
                (
                    "pair_lag_corr_xx",
                    (|p: &PairRow| p.lag_corr_xx.as_slice()) as fn(&PairRow) -> &[f64],
                ),
                ("pair_lag_corr_yy", |p| p.lag_corr_yy.as_slice()),
            ] {
                cols.push(list_list_f64(name, rows, move |r| {
                    asked(r, spec_lags).then(|| r.pairs.iter().map(f).collect())
                })?);
            }
        }
        // The cross pair under its own rule: absent when no closing spec
        // keeps a cross term (`cross_lags = []`), as `marginal()` leaves it.
        if has_cross_lags {
            for (name, f) in [
                (
                    "pair_lag_corr_xy",
                    (|p: &PairRow| p.lag_corr_xy.as_slice()) as fn(&PairRow) -> &[f64],
                ),
                ("pair_lag_corr_yx", |p| p.lag_corr_yx.as_slice()),
            ] {
                cols.push(list_list_f64(name, rows, move |r| {
                    asked(r, spec_cross_lags).then(|| r.pairs.iter().map(f).collect())
                })?);
            }
        }
        if has_lags {
            for (name, f) in [
                (
                    "pair_n_serial",
                    (|p: &PairRow| p.n_serial) as fn(&PairRow) -> f64,
                ),
                ("pair_t_stat_serial", |p| p.t_serial),
                ("pair_phi_x", |p| p.phi_x),
                ("pair_phi_y", |p| p.phi_y),
            ] {
                cols.push(pair_f64(name, rows, move |r| {
                    asked(r, spec_lags).then(|| r.pairs.iter().map(f).collect())
                }));
            }
        }
        if has_bins {
            for (name, f) in [
                (
                    "pair_bin_edges",
                    (|p: &PairRow| p.bin_edges.as_slice()) as fn(&PairRow) -> &[f64],
                ),
                ("pair_bin_n", |p| p.bin_n.as_slice()),
                ("pair_bin_mean_y", |p| p.bin_mean_y.as_slice()),
                ("pair_bin_var_y", |p| p.bin_var_y.as_slice()),
            ] {
                cols.push(list_list_f64(name, rows, move |r| {
                    asked(r, spec_bins).then(|| r.pairs.iter().map(f).collect())
                })?);
            }
            for (name, f) in [
                (
                    "pair_split_gain",
                    (|p: &PairRow| p.split_gain) as fn(&PairRow) -> f64,
                ),
                ("pair_split_at", |p| p.split_at),
                ("pair_split_gain_t", |p| p.split_gain_t),
            ] {
                cols.push(pair_f64(name, rows, move |r| {
                    asked(r, spec_bins).then(|| r.pairs.iter().map(f).collect())
                }));
            }
        }
    }
    if has_rcov {
        let block =
            |f: fn(&RcovRow) -> Option<Vec<f64>>| move |r: &ClosedRow| r.rcov.as_ref().and_then(f);
        cols.push(list_f64("rcov", rows, block(|b| b.rcov.clone())));
        cols.push(list_f64("rcorr", rows, block(|b| b.rcorr.clone())));
        cols.push(Column::new(
            "rcov_n".into(),
            rows.iter()
                .map(|r| r.rcov.as_ref().map(|b| b.n))
                .collect::<Vec<_>>(),
        ));
        // An `Enum` over the estimators, in `RcovKind`'s order (task 244).
        let kinds = [
            rheon_core::RcovKind::Plain,
            rheon_core::RcovKind::Kernel,
            rheon_core::RcovKind::Preavg,
        ]
        .map(rheon_core::RcovKind::as_str);
        cols.push(
            Column::new(
                "rcov_kind".into(),
                rows.iter()
                    .map(|r| r.rcov.as_ref().map(|b| b.kind.as_str()))
                    .collect::<Vec<_>>(),
            )
            .cast(&DataType::from_frozen_categories(
                FrozenCategories::new(kinds).expect("distinct names"),
            ))?,
        );
        // Every column of the block carries the kind's prefix, as
        // `marginal`'s block carries `pair_` (review round 4, PA13).
        cols.push(Column::new(
            "rcov_bandwidth_used".into(),
            rows.iter()
                .map(|r| r.rcov.as_ref().and_then(|b| b.bandwidth_used))
                .collect::<Vec<_>>(),
        ));
        cols.push(list_f64("rcov_omega2", rows, block(|b| b.omega2.clone())));
        cols.push(list_f64(
            "rcov_iv_sparse",
            rows,
            block(|b| b.iv_sparse.clone()),
        ));
        cols.push(list_f64("rcov_iq", rows, block(|b| b.iq.clone())));
        cols.push(Column::new(
            "rcov_psd_repaired".into(),
            rows.iter()
                .map(|r| r.rcov.as_ref().and_then(|b| b.psd_repaired))
                .collect::<Vec<_>>(),
        ));
    }
    DataFrame::new(rows.len(), cols)
}
