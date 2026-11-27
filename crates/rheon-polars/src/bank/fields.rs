//! The output index: every field a spec writes, with the machine values
//! its name encodes and the buffer `assemble` fills it from. Moved whole
//! out of `bank.rs`, which stood at the source-size cap
//! (`tests/test_repo_hygiene.py`).

use super::*;

/// One output field with the machine values its name encodes.
///
/// This is the antidote to string formatting as API: a caller filters this
/// table for `kind == "pred" && target == "y" && ridge == Some(0.5)` instead
/// of constructing `"pred_y__r0.5@h500"` by hand — which would require
/// reimplementing `num_label`'s float rendering. Produced by the same code
/// that renders the names, so the two cannot drift.
#[derive(Debug, Clone, serde::Serialize)]
pub struct FieldMeta {
    pub field: String,
    /// pred / resid / sigma / zscore / ic / r2 / hit_rate / abs_resid_q /
    /// autocorr / drift / n_eff / coef / lam_selected / selected /
    /// pred_selected / pred_averaged — or an `ew_cov` statistic name.
    pub kind: String,
    pub target: Option<String>,
    /// The instance's field suffix (`"@h500"`, or `""` for a single
    /// instance), as `coef()`, `gram()` and `closed_groups()` name it;
    /// `None` on a field that chooses across the instances
    /// (`emit_selected`, `emit_averaged`) or is the spec's own (the clocks).
    pub instance: Option<String>,
    /// The instance's decay, as configured (present even when the suffix is
    /// empty because there is a single instance).
    pub half_life: Option<f64>,
    pub lam: Option<f64>,
    pub ridge: Option<f64>,
    pub feature_set: Option<String>,
    /// Lasso path point.
    pub penalty: Option<f64>,
    /// Quantile level (`abs_resid_q*` fields).
    pub quantile: Option<f64>,
    /// Lag, in learned rows, of an `ew_cov` `lag_corr_*` field
    /// (docs/ENHANCEMENTS.md E56); `None` for every other field.
    pub lag: Option<usize>,
    /// The principal component an `ew_cov` `pc<j>_*` field is about, or a
    /// `spectral` field's period, by its place in `periods`; `None` for
    /// every other field.
    pub component: Option<usize>,
    /// The hidden state an `hmm` posterior is about, `filtered_state<s>`
    /// and `predicted_state<s>`; `None` for every other field. With `lag`,
    /// `component` and `class`, it makes every field's row unique apart
    /// from its name.
    pub state: Option<usize>,
    /// The class an `ew_class` posterior `p_<class>` is about; `None` for
    /// every other field.
    pub class: Option<String>,
    /// The arm a `bandit` per-arm score `score_<arm>` is about; `None` for
    /// every other field.
    pub arm: Option<String>,
    /// Columns an `ew_cov` statistic is over.
    pub columns: Option<Vec<String>>,
    /// The polars dtype the field is materialized with, as its string form
    /// (`f64`, `i64`, `bool`, `enum`, `list[f64]`, `clock`).
    /// Set from `src`, so it is the same
    /// table `assemble` fills the buffers from, so a declared output struct and
    /// the values that land in it cannot disagree (docs/IMPROVEMENTS.md C1 — a
    /// name-prefix guess once declared `drift_*` as `f64` while the bank
    /// produced `bool`, and polars refused the struct).
    pub dtype: String,
    /// Which assembled buffer, and where in it, this field's values come from.
    /// Private and not serialized: it is how `assemble` walks this schema
    /// instead of rebuilding the same nested loops with its own `format!`
    /// calls (docs/SIMPLIFICATION.md S1).
    #[serde(skip)]
    pub(super) src: Source,
    /// The categories of an `Enum` field in their order: `ew_class`'s
    /// classes for `class`, the grid's slots for `selected_<t>`, the
    /// reasons the spec can name for `withheld_reason`. `None` for every
    /// other field.
    #[serde(skip)]
    pub(super) categories: Option<Vec<String>>,
}

/// The buffer a field's values are scattered into, with the index into it.
/// One variant per buffer `assemble` allocates.
#[derive(Debug, Clone, Default, PartialEq)]
pub(super) enum Source {
    /// Assigned before the field is pushed; never observed.
    #[default]
    Unset,
    Pred(usize),
    /// An integer a row reports beside its predictions, in the extras
    /// buffer as `LamSelected` is: `vector_ar`'s order.
    ExtraId(usize),
    Resid(usize),
    Sigma(usize),
    ResidZ(usize),
    Drift(usize),
    /// `(which of ic/r2/hit_rate, index)`.
    Metric(usize, usize),
    /// `(which of lo/hi/coverage, index)`, laid out like `Metric`.
    Conformal(usize, usize),
    /// `(which of task 221's values, index)`, laid out like `Metric` with
    /// `n_checks` blocks per instance ([`crate::checks::value_names`]).
    Check(usize, usize),
    /// `(which of spread_ratio/mean_shift, instance, feature)`, in the
    /// `health` buffer (task 221 (g)).
    Health(usize, usize, usize),
    Quantile(usize),
    Autocorr(usize),
    NEff(usize),
    Coef(usize),
    /// `settled_frac`, per instance, in the `settled` buffer
    /// (docs/WARMUP-AND-CONVERGENCE.md §3).
    Settled(usize),
    /// `withheld_reason`, per instance: a code in the `reason` buffer,
    /// materialized as a dictionary-encoded string (a categorical).
    Reason(usize),
    /// `error_inflation_<slot>`, per slot, in the `inflation` buffer.
    Inflation(usize),
    /// `support_coef`, per instance, laid out like `Coef`.
    SupportCoef(usize),
    /// `se_coef`, per instance, laid out like `Coef` (task 116).
    SeCoef(usize),
    /// `se_coef_hc0` and `se_coef_hac`, per instance, laid out like `Coef`
    /// (task 221 (c)).
    SeCoefHc0(usize),
    SeCoefHac(usize),
    /// `scored_clock` and `learned_clock`, one pair per spec, in the clock
    /// column's own type (docs/PLAN.md task 152).
    ScoredClock,
    LearnedClock,
    /// `kalman`'s fixed-lag smoother, per instance: `smoothed_coef` and
    /// `smoothed_se_coef`, laid out like `Coef`, and `smoothed_for_<t>`,
    /// per instance and target (`instance * m + target`), in the clock
    /// column's own type.
    SmoothedCoef(usize),
    SmoothedSe(usize),
    SmoothedFor(usize),
    LamSelected(usize),
    /// `weight_<target>` under `emit_target_weight`: the weight the
    /// target's `min_weight` gate read, per instance and target, laid out
    /// like `LamSelected` in the `target_weight` buffer (docs/PLAN.md task
    /// 568).
    TargetWeight(usize),
    SelPred(usize),
    SelName(usize),
    AvgPred(usize),
    /// An `ew_cov` statistic, or a `kmeans` / `micro` distance, which rides
    /// in the `pred` buffer.
    Stat(usize),
    /// Every integer a model emits -- a cluster label, a `micro` id or
    /// count, an `hmm` state, a run length: the `pred` buffer holds it as an
    /// f64 (NaN = null), materialized as `i64`, Polars' default integer, so
    /// every integer output has one dtype (task 238).
    Id(usize),
    /// A `micro` flag: `1.0` / `0.0` in the `pred` buffer (NaN = null),
    /// materialized as `Boolean`.
    Flag(usize),
    /// An `ew_class` prediction: the class's position in `classes` as an
    /// f64 in the `pred` buffer (NaN = null), materialized as the class name.
    Label(usize),
}

impl FieldMeta {
    fn new(field: String, kind: &str) -> Self {
        Self {
            field,
            kind: kind.to_string(),
            target: None,
            instance: None,
            half_life: None,
            lam: None,
            ridge: None,
            feature_set: None,
            penalty: None,
            quantile: None,
            lag: None,
            component: None,
            state: None,
            class: None,
            arm: None,
            columns: None,
            dtype: String::new(),
            src: Source::Unset,
            categories: None,
        }
    }
    /// An `Enum` field over these categories, in this order; set before
    /// [`Self::src`], which renders the dtype.
    fn categories(mut self, names: &[String]) -> Self {
        self.categories = Some(names.to_vec());
        self
    }
    fn src(mut self, src: Source) -> Self {
        // The clock fields take the clock column's own type, known only
        // when a chunk arrives (task 152).
        let is_clock = matches!(
            src,
            Source::ScoredClock | Source::LearnedClock | Source::SmoothedFor(_)
        );
        self.src = src;
        self.dtype = if is_clock {
            "clock".to_string()
        } else {
            self.dtype().to_string()
        };
        self
    }

    /// The dtype `assemble` materializes this field with.
    pub fn dtype(&self) -> DataType {
        let enum_of = |names: &[String]| {
            DataType::from_frozen_categories(
                polars::prelude::FrozenCategories::new(names.iter().map(String::as_str))
                    .expect("distinct names"),
            )
        };
        match self.src {
            Source::Drift(_) => DataType::Boolean,
            // An `Enum` over the slots or the classes, in their order: a
            // value is one of them, and a sort or a comparison goes by that
            // order (task 244).
            Source::SelName(_) | Source::Label(_) => {
                enum_of(self.categories.as_deref().unwrap_or_default())
            }
            Source::Coef(_)
            | Source::SupportCoef(_)
            | Source::SeCoef(_)
            | Source::SeCoefHc0(_)
            | Source::SeCoefHac(_)
            | Source::SmoothedCoef(_)
            | Source::SmoothedSe(_) => DataType::List(Box::new(DataType::Float64)),
            // The reasons the spec can name, set from
            // `stream::withheld_reasons` where the field is pushed.
            Source::Reason(_) => enum_of(self.categories.as_deref().unwrap_or_default()),
            Source::Id(_) | Source::ExtraId(_) => DataType::Int64,
            Source::Flag(_) => DataType::Boolean,
            Source::Unset => unreachable!("every field is given a source in output_index"),
            _ => DataType::Float64,
        }
    }
    /// The instance the field belongs to: its suffix and its decay.
    fn decay(mut self, suffix: &str, d: &rheon_core::Decay) -> Self {
        self.instance = Some(suffix.to_string());
        match d {
            rheon_core::Decay::Halflife(h) => self.half_life = Some(*h),
            rheon_core::Decay::Lam(l) => self.lam = Some(*l),
        }
        self
    }
    fn target(mut self, t: &str) -> Self {
        self.target = Some(t.to_string());
        self
    }
    fn combo(mut self, c: &crate::stream::Combo) -> Self {
        self.ridge = c.ridge;
        self.feature_set = c.feature_set.clone();
        self.penalty = c.lambda;
        self
    }
}

/// Every coefficient a spec reports, in `coef` list order, with the name
/// [`CoefField`] gives it. Empty for `ew_cov` and `seqtest`, which have
/// none, and for `micro`, whose `coef` is one `[id, label, n, radius, c_1 ..
/// c_p]` row per potential summary -- as many as there are, so no position
/// has a name.
///
/// The layout is the models' (`rheon_core`): per instance, one `coef` list
/// holding `(target, combo)` slots in the order the `pred` fields declare
/// them, each slot the full term vector -- `intercept` first when the spec
/// has one, then every feature, zeros for features a feature set leaves out
/// (`EwRidgeModel::solve` scatters each combo's solution into `k_total`
/// columns) -- or `level`, `trend` for `holt`. Rendered here, beside the
/// field names, from the same combos and suffixes, so the two cannot drift.
///
/// `kmeans` reports its centres here: `k` slots named `cluster{j}` in place
/// of the targets, each the centre's coordinate per feature, so
/// `coef_cluster0_x1` is centre 0's `x1` and `coef_index` lays the list out
/// as `(cluster, feature)`, the centre in `cluster` and no target. `hmm`
/// does the same with its state means (`coef_state0_x1`, `state`), and
/// `ew_class` with its class means: one slot per class, named by the class,
/// so `coef_a_x1` is class `a`'s mean of `x1`, the class in `class` and the
/// label column in `target`.
pub fn coef_fields(spec: &Spec) -> Vec<CoefField> {
    if matches!(
        spec.model,
        crate::ModelKind::EwCov { .. }
            | crate::ModelKind::Micro { .. }
            | crate::ModelKind::SeqTest { .. }
            | crate::ModelKind::Marginal { .. }
            | crate::ModelKind::Rcov { .. }
            | crate::ModelKind::CorrChange { .. }
            | crate::ModelKind::Bocpd { .. }
            | crate::ModelKind::Audit { .. }
            | crate::ModelKind::Realized { .. }
            | crate::ModelKind::Pca { .. }
            | crate::ModelKind::PowerVariation { .. }
            | crate::ModelKind::TDigest { .. }
            | crate::ModelKind::DdSketch { .. }
            | crate::ModelKind::EwmMoments { .. }
            | crate::ModelKind::Spectral(_)
            | crate::ModelKind::Isotonic { .. }
            | crate::ModelKind::Krls(_)
            | crate::ModelKind::Dependence(_)
            | crate::ModelKind::HoeffdingTreeClassifier(_)
            | crate::ModelKind::HoeffdingTreeRegressor(_)
            | crate::ModelKind::HalfSpaceTrees(_)
            | crate::ModelKind::RandomCutForest(_)
    ) {
        return Vec::new();
    }
    // dcc's `coef` is its parameters, `a`, `b` and the asymmetric model's
    // `g`, one slot each and no target: `coef_a`.
    if let crate::ModelKind::Dcc { asymmetric, .. } = &spec.model {
        let names: &[&str] = if asymmetric.unwrap_or(false) {
            &["a", "b", "g"]
        } else {
            &["a", "b"]
        };
        let mut out = Vec::new();
        for (suffix, d) in spec.decays().expect("validated") {
            for (position, n) in names.iter().enumerate() {
                out.push(CoefField {
                    field: format!("coef{suffix}"),
                    position,
                    name: format!("coef_{n}{suffix}"),
                    target: None,
                    instance: suffix.clone(),
                    half_life: match d {
                        rheon_core::Decay::Halflife(h) => Some(h),
                        rheon_core::Decay::Lam(_) => None,
                    },
                    lam: match d {
                        rheon_core::Decay::Lam(l) => Some(l),
                        rheon_core::Decay::Halflife(_) => None,
                    },
                    ridge: None,
                    feature_set: None,
                    penalty: None,
                    cluster: None,
                    state: None,
                    class: None,
                    arm: None,
                    term: n.to_string(),
                });
            }
        }
        return out;
    }
    // deco's `coef` is the correlation values themselves, one slot each,
    // named for the block or the pair of blocks they belong to.
    if matches!(spec.model, crate::ModelKind::Deco { .. }) {
        let names = crate::stream::deco_block_names(spec);
        let labels = rheon_core::Deco::labels(&names);
        let values: Vec<&String> = labels
            .iter()
            .filter(|l| l.as_str() == "rho" || l.starts_with("rho_"))
            .collect();
        let mut out = Vec::new();
        for (suffix, d) in spec.decays().expect("validated") {
            for (position, l) in values.iter().enumerate() {
                let slot = l.strip_prefix("rho_").unwrap_or("rho");
                out.push(CoefField {
                    field: format!("coef{suffix}"),
                    position,
                    name: format!("coef_{slot}{suffix}"),
                    target: Some(slot.to_string()),
                    instance: suffix.clone(),
                    half_life: match d {
                        rheon_core::Decay::Halflife(h) => Some(h),
                        rheon_core::Decay::Lam(_) => None,
                    },
                    lam: match d {
                        rheon_core::Decay::Lam(l) => Some(l),
                        rheon_core::Decay::Halflife(_) => None,
                    },
                    ridge: None,
                    feature_set: None,
                    penalty: None,
                    cluster: None,
                    state: None,
                    class: None,
                    arm: None,
                    term: "rho".to_string(),
                });
            }
        }
        return out;
    }
    // Each slot: the name its columns take, and the target, centre, state,
    // class or arm it is.
    struct Slot {
        name: String,
        target: Option<String>,
        cluster: Option<usize>,
        state: Option<usize>,
        class: Option<String>,
        arm: Option<String>,
    }
    let slot = |name: String| Slot {
        name,
        target: None,
        cluster: None,
        state: None,
        class: None,
        arm: None,
    };
    let slots: Vec<Slot> = match &spec.model {
        crate::ModelKind::KMeans { k, .. } => (0..*k)
            .map(|j| Slot {
                cluster: Some(j),
                ..slot(format!("cluster{j}"))
            })
            .collect(),
        // One slot per hidden state, named as `kmeans` names its centres.
        crate::ModelKind::Hmm { k, .. } => (0..*k)
            .map(|j| Slot {
                state: Some(j),
                ..slot(format!("state{j}"))
            })
            .collect(),
        crate::ModelKind::EwClass { classes, .. } => classes
            .iter()
            .map(|c| Slot {
                target: Some(spec.targets[0].clone()),
                class: Some(c.clone()),
                ..slot(c.clone())
            })
            .collect(),
        crate::ModelKind::Bandit(b) => b
            .arms
            .iter()
            .map(|a| Slot {
                target: Some(spec.targets[0].clone()),
                arm: Some(a.clone()),
                ..slot(a.clone())
            })
            .collect(),
        _ => spec
            .targets
            .iter()
            .map(|t| Slot {
                target: Some(t.clone()),
                ..slot(t.clone())
            })
            .collect(),
    };
    let terms: Vec<String> = if matches!(spec.model, crate::ModelKind::Holt { .. }) {
        vec!["level".into(), "trend".into()]
    } else if matches!(spec.model, crate::ModelKind::Hawkes { .. }) {
        // `μ`, then `α` from each source node at each kernel, the source
        // outer: the order of `Hawkes::coefficients`.
        let kernels = spec.model.hawkes_kernels();
        std::iter::once("mu".to_string())
            .chain(spec.targets.iter().flat_map(|src| {
                kernels
                    .iter()
                    .map(move |(_, label)| format!("alpha_{src}_{label}"))
            }))
            .collect()
    } else if matches!(spec.model, crate::ModelKind::Garch { .. }) {
        // The process's parameters, each input's `π` as `pi_<feature>`,
        // then the shape: `Garch::coefficients`.
        let cfg = crate::stream::garch_cfg(spec).expect("validated");
        cfg.vol
            .names()
            .iter()
            .map(|n| n.to_string())
            .chain(spec.features.iter().map(|f| format!("pi_{f}")))
            .chain(cfg.dist.names().iter().map(|n| n.to_string()))
            .collect()
    } else if let crate::ModelKind::VectorAr(v) = &spec.model {
        // The constant, the features, then each lag of every series:
        // `VectorAr::coefficients`, statsmodels' `params` order.
        let mut t = Vec::new();
        if spec.fit_intercept {
            t.push("intercept".to_string());
        }
        t.extend(spec.features.iter().cloned());
        for l in 1..=v.lags {
            t.extend(spec.targets.iter().map(|s| format!("lag{l}_{s}")));
        }
        t
    } else if let crate::ModelKind::Ar { lags, .. } = &spec.model {
        // The constant, each lag, then the features: `Ar::coefficients`.
        let mut t = Vec::with_capacity(spec.features.len() + 1);
        if spec.fit_intercept {
            t.push("intercept".to_string());
        }
        t.extend(lags.lags().iter().map(|l| format!("lag{l}")));
        t.extend(spec.features.iter().cloned());
        t
    } else if matches!(
        spec.model,
        crate::ModelKind::KMeans { .. }
            | crate::ModelKind::EwClass { .. }
            | crate::ModelKind::Hmm { .. }
    ) {
        spec.features.clone()
    } else {
        let mut t = Vec::with_capacity(spec.features.len() + 1);
        if spec.fit_intercept {
            t.push("intercept".to_string());
        }
        t.extend(spec.features.iter().cloned());
        t
    };
    let combos = crate::stream::combos(spec);
    let decays = spec.decays().expect("validated");
    let mut out = Vec::new();
    for (suffix, d) in &decays {
        let (half_life, lam) = match d {
            rheon_core::Decay::Halflife(h) => (Some(*h), None),
            rheon_core::Decay::Lam(l) => (None, Some(*l)),
        };
        let mut position = 0;
        for t in &slots {
            for c in &combos {
                for term in &terms {
                    out.push(CoefField {
                        field: format!("coef{suffix}"),
                        position,
                        name: format!("coef_{}_{term}{}{suffix}", t.name, c.label),
                        target: t.target.clone(),
                        instance: suffix.clone(),
                        half_life,
                        lam,
                        ridge: c.ridge,
                        feature_set: c.feature_set.clone(),
                        penalty: c.lambda,
                        cluster: t.cluster,
                        state: t.state,
                        class: t.class.clone(),
                        arm: t.arm.clone(),
                        term: term.clone(),
                    });
                    position += 1;
                }
            }
        }
    }
    out
}

/// The refusal for two outputs that render to one field name, naming the
/// inputs that collided -- a target, a feature, a grid label -- since the
/// rendered names are the user's handle on every output, and a duplicate
/// inside one struct would otherwise surface much later as a polars error.
/// `Spec::validate` refuses every way the grammar has of making one that it
/// can see (a feature-set name given twice got through until review
/// 2026-09-12, S7; a target `z_y` beside `y` under `emit_resid_z` until task
/// 144, which renamed the field); the names that remain are the columns'
/// own, as `ew_cov`'s `corr_a_b_c` over `a_b, c` and over `a, b_c`.
pub fn duplicate_field(spec: &Spec) -> Option<String> {
    let index = output_index(spec);
    let mut seen: HashMap<&str, &FieldMeta> = HashMap::with_capacity(index.len());
    for f in &index {
        let Some(first) = seen.insert(f.field.as_str(), f) else {
            continue;
        };
        return Some(format!(
            "spec {:?}: two outputs render to the same field name {:?}: {} and {}; rename a \
             target, feature or grid label to tell them apart",
            spec.name,
            f.field,
            described(first),
            described(f)
        ));
    }
    None
}

/// The first output field whose name holds a NUL character, which the Arrow
/// export cannot carry: it panicked there, after the bank had learned the
/// chunk (review 2026-11-23, SPEC-1). The spec's own names are refused by
/// name in `Spec::validate`; this catches the names a model makes of its
/// settings: a feature set, a `deco` block, an `ew_class` class.
pub(crate) fn nul_field(spec: &Spec) -> Option<String> {
    let f = output_index(spec)
        .into_iter()
        .find(|f| f.field.contains('\0'))?;
    Some(format!(
        "spec {:?}: the output field {:?} ({}) holds a NUL character (\\0), which no column \
         name of an Arrow export can hold. Rename the setting that names it without one.",
        spec.name,
        f.field,
        described(&f)
    ))
}

/// An output field in words: its kind and what it is of.
fn described(m: &FieldMeta) -> String {
    let mut s = format!("{:?}", m.kind);
    if let Some(t) = &m.target {
        s.push_str(&format!(" of target {t:?}"));
    }
    if let Some(c) = &m.columns {
        s.push_str(&format!(" over columns {c:?}"));
    }
    if let Some(fs) = &m.feature_set {
        s.push_str(&format!(" in feature set {fs:?}"));
    }
    if let Some(r) = m.ridge {
        s.push_str(&format!(" at ridge {r}"));
    }
    if let Some(h) = m.half_life {
        s.push_str(&format!(" at half_life {h}"));
    }
    s
}

/// A `list[str]` column of these lists, null where a row has none.
fn str_lists(name: &str, rows: &[Option<&Vec<String>>]) -> Column {
    let mut b = ListStringChunkedBuilder::new(name.into(), rows.len(), 4);
    for r in rows {
        match r {
            Some(v) => b.append_values_iter(v.iter().map(String::as_str)),
            None => b.append_null(),
        }
    }
    b.finish().into_series().into()
}

/// An `Int64` column of these indices, null where a row has none. Every
/// integer a frame of this library writes is an `Int64`, polars' default.
fn ints(name: &str, values: impl Iterator<Item = Option<usize>>) -> Column {
    Column::new(
        name.into(),
        values
            .map(|v| v.map(|v| i64::try_from(v).unwrap_or(i64::MAX)))
            .collect::<Vec<Option<i64>>>(),
    )
}

/// [`output_index`] as the frame `rheon.spec.output_index` returns,
/// typed here rather than through JSON, which has no infinity: an infinite
/// half-life is `inf` in `half_life`, where JSON read it as null.
///
/// # Errors
///
/// None in practice: every column has one row per field.
pub fn output_index_frame(spec: &Spec) -> PolarsResult<DataFrame> {
    let idx = output_index(spec);
    let str_col = |name: &str, f: &dyn Fn(&FieldMeta) -> Option<&str>| {
        Column::new(name.into(), idx.iter().map(f).collect::<Vec<_>>())
    };
    let f64_col = |name: &str, f: &dyn Fn(&FieldMeta) -> Option<f64>| {
        Column::new(name.into(), idx.iter().map(f).collect::<Vec<_>>())
    };
    let columns: Vec<Option<&Vec<String>>> = idx.iter().map(|m| m.columns.as_ref()).collect();
    DataFrame::new(
        idx.len(),
        vec![
            str_col("field", &|m| Some(m.field.as_str())),
            str_col("kind", &|m| Some(m.kind.as_str())),
            str_col("target", &|m| m.target.as_deref()),
            str_col("instance", &|m| m.instance.as_deref()),
            f64_col("half_life", &|m| m.half_life),
            f64_col("lam", &|m| m.lam),
            f64_col("ridge", &|m| m.ridge),
            str_col("feature_set", &|m| m.feature_set.as_deref()),
            f64_col("penalty", &|m| m.penalty),
            f64_col("quantile", &|m| m.quantile),
            ints("lag", idx.iter().map(|m| m.lag)),
            ints("component", idx.iter().map(|m| m.component)),
            ints("state", idx.iter().map(|m| m.state)),
            str_col("class", &|m| m.class.as_deref()),
            str_col("arm", &|m| m.arm.as_deref()),
            str_lists("columns", &columns),
            str_col("dtype", &|m| Some(m.dtype.as_str())),
        ],
    )
}

/// [`coef_fields`] as the frame `rheon.spec.coef_fields` returns,
/// typed here, as [`output_index_frame`] is. Empty, with every column, for
/// a spec with no named coefficient.
///
/// # Errors
///
/// None in practice: every column has one row per coefficient.
pub fn coef_fields_frame(spec: &Spec) -> PolarsResult<DataFrame> {
    let cf = coef_fields(spec);
    let str_col = |name: &str, f: &dyn Fn(&CoefField) -> Option<&str>| {
        Column::new(name.into(), cf.iter().map(f).collect::<Vec<_>>())
    };
    let f64_col = |name: &str, f: &dyn Fn(&CoefField) -> Option<f64>| {
        Column::new(name.into(), cf.iter().map(f).collect::<Vec<_>>())
    };
    DataFrame::new(
        cf.len(),
        vec![
            str_col("field", &|c| Some(c.field.as_str())),
            ints("position", cf.iter().map(|c| Some(c.position))),
            str_col("name", &|c| Some(c.name.as_str())),
            str_col("target", &|c| c.target.as_deref()),
            str_col("instance", &|c| Some(c.instance.as_str())),
            f64_col("half_life", &|c| c.half_life),
            f64_col("lam", &|c| c.lam),
            f64_col("ridge", &|c| c.ridge),
            str_col("feature_set", &|c| c.feature_set.as_deref()),
            f64_col("penalty", &|c| c.penalty),
            ints("cluster", cf.iter().map(|c| c.cluster)),
            ints("state", cf.iter().map(|c| c.state)),
            str_col("class", &|c| c.class.as_deref()),
            str_col("arm", &|c| c.arm.as_deref()),
            str_col("term", &|c| Some(c.term.as_str())),
        ],
    )
}

/// Output field names for a spec, in struct order (used by Python for dtypes).
pub fn output_fields(spec: &Spec) -> Vec<String> {
    output_index(spec).into_iter().map(|m| m.field).collect()
}

/// Every output field with its metadata, in struct order.
///
/// The readiness fields (docs/WARMUP-AND-CONVERGENCE.md §3) ride on every
/// model that writes a row: `settled_frac` and `withheld_reason` follow
/// each instance's `n_eff`, and `support_coef` follows `coef` where the
/// model has one -- inserted here, once, rather than in each model's own
/// layout. The state-only models (`marginal`, `rcov`, `audit`) write no
/// row and get none.
pub fn output_index(spec: &Spec) -> Vec<FieldMeta> {
    let base = output_index_base(spec);
    if matches!(
        spec.model,
        crate::ModelKind::Marginal { .. }
            | crate::ModelKind::Rcov { .. }
            | crate::ModelKind::Audit { .. }
    ) {
        return with_clocks(spec, no_decay(spec, base));
    }
    let mut fields = Vec::with_capacity(base.len() + 3 * spec.decays().map_or(1, |d| d.len()));
    for f in base {
        // The instance the field belongs to, as its `n_eff` or `coef`
        // carries it: the same suffix and the same decay (none, for a model
        // that does not decay -- `seqtest`'s fields carry no half-life).
        let (src, suffix, instance, half_life, lam) = (
            f.src.clone(),
            f.field.clone(),
            f.instance.clone(),
            f.half_life,
            f.lam,
        );
        fields.push(f);
        let like = |mut m: FieldMeta| {
            m.instance = instance.clone();
            m.half_life = half_life;
            m.lam = lam;
            m
        };
        match src {
            Source::NEff(mi) => {
                let suffix = suffix.strip_prefix("weight_sum").unwrap_or("");
                // Each target's own weight beside the shared one, opt-in
                // (docs/PLAN.md task 568): the number its `min_weight` reads.
                if spec.emit_target_weight {
                    for (t_i, t) in spec.targets.iter().enumerate() {
                        fields.push(like(
                            FieldMeta::new(format!("weight_{t}{suffix}"), "weight")
                                .target(t)
                                .src(Source::TargetWeight(mi * spec.m() + t_i)),
                        ));
                    }
                }
                fields.push(like(
                    FieldMeta::new(format!("settled_frac{suffix}"), "settled_frac")
                        .src(Source::Settled(mi)),
                ));
                let reasons: Vec<String> = crate::stream::withheld_reasons(spec)
                    .iter()
                    .map(|r| (*r).to_string())
                    .collect();
                fields.push(like(
                    FieldMeta::new(format!("withheld_reason{suffix}"), "withheld_reason")
                        .categories(&reasons)
                        .src(Source::Reason(mi)),
                ));
            }
            Source::Coef(mi) => {
                let suffix = suffix.strip_prefix("coef").unwrap_or("");
                if spec.has_support_coef() {
                    fields.push(like(
                        FieldMeta::new(format!("support_coef{suffix}"), "support_coef")
                            .src(Source::SupportCoef(mi)),
                    ));
                }
                // Each coefficient's standard error, opt-in (task 116, F).
                if spec.emit_se_coef && spec.has_se_coef() {
                    fields.push(like(
                        FieldMeta::new(format!("se_coef{suffix}"), "se_coef")
                            .src(Source::SeCoef(mi)),
                    ));
                }
                // The fixed-lag smoother's, on the rows that complete a lag.
                if spec.smooth_lag().is_some() {
                    fields.push(like(
                        FieldMeta::new(format!("smoothed_coef{suffix}"), "smoothed_coef")
                            .src(Source::SmoothedCoef(mi)),
                    ));
                    if spec.emit_smoothed_se() {
                        fields.push(like(
                            FieldMeta::new(format!("smoothed_se_coef{suffix}"), "smoothed_se_coef")
                                .src(Source::SmoothedSe(mi)),
                        ));
                    }
                    for (t_i, t) in spec.targets.iter().enumerate() {
                        fields.push(like(
                            FieldMeta::new(format!("smoothed_for_{t}{suffix}"), "smoothed_for")
                                .target(t)
                                .src(Source::SmoothedFor(mi * spec.m() + t_i)),
                        ));
                    }
                }
                // The robust ones beside it, opt-in (task 221 (c)).
                if spec.emit_robust_se && spec.has_robust_se() {
                    fields.push(like(
                        FieldMeta::new(format!("se_coef_hc0{suffix}"), "se_coef_hc0")
                            .src(Source::SeCoefHc0(mi)),
                    ));
                    if spec.robust_se_lags_or_default() > 0 {
                        fields.push(like(
                            FieldMeta::new(format!("se_coef_hac{suffix}"), "se_coef_hac")
                                .src(Source::SeCoefHac(mi)),
                        ));
                    }
                }
            }
            _ => {}
        }
    }
    with_clocks(spec, no_decay(spec, fields))
}

/// A model that takes no decay -- `seqtest`, `rcov`, `corrchange`,
/// `bocpd`, `audit` -- runs one undecayed instance, which `Spec::decays`
/// gives as an infinite half-life. Its fields carry no decay, as its spec
/// configures none; a spec that says `half_life = inf` carries `inf`.
fn no_decay(spec: &Spec, mut fields: Vec<FieldMeta>) -> Vec<FieldMeta> {
    if spec.half_life.is_none() && spec.lam.is_none() {
        for f in &mut fields {
            f.half_life = None;
            f.lam = None;
        }
    }
    fields
}

/// The clock fields, last, under `emit_clocks` (docs/PLAN.md task 152).
fn with_clocks(spec: &Spec, mut fields: Vec<FieldMeta>) -> Vec<FieldMeta> {
    if spec.emit_clocks {
        fields.push(FieldMeta::new("scored_clock".into(), "scored_clock").src(Source::ScoredClock));
        fields.push(
            FieldMeta::new("learned_clock".into(), "learned_clock").src(Source::LearnedClock),
        );
    }
    fields
}

/// [`output_index`] before the readiness fields: each model's own layout.
fn output_index_base(spec: &Spec) -> Vec<FieldMeta> {
    let decays = spec.decays().expect("validated");
    // deco is not a regression either: its slots are the row's own
    // equicorrelation estimate, the level before the row and the row's
    // log-density under it -- one of each unblocked, one `u_*` and one
    // `rho_*` per value and a single `loglik` with blocks.
    if matches!(spec.model, crate::ModelKind::Deco { .. }) {
        let names = crate::stream::deco_block_names(spec);
        let labels = rheon_core::Deco::labels(&names);
        let n_slots = labels.len();
        // The columns a slot is over: its block, its pair of blocks, or all
        // of them for `loglik` and the unblocked scalars.
        let blocks = crate::stream::deco_cfg(spec).expect("validated");
        let all = spec.features.clone();
        let cols_of = |slot: usize| -> Vec<String> {
            if names.is_empty() {
                return all.clone();
            }
            let k = names.len();
            let m = k + k * (k - 1) / 2;
            if slot >= 2 * m {
                return all.clone();
            }
            let v = slot % m;
            let of = |b: usize| -> Vec<String> {
                blocks.blocks[b].iter().map(|&i| all[i].clone()).collect()
            };
            if v < k {
                return of(v);
            }
            let mut p = k;
            for i in 0..k {
                for j in (i + 1)..k {
                    if p == v {
                        let mut cols = of(i);
                        cols.extend(of(j));
                        return cols;
                    }
                    p += 1;
                }
            }
            all.clone()
        };
        let mut fields = Vec::new();
        for (mi, (suffix, d)) in decays.iter().enumerate() {
            for (slot, l) in labels.iter().enumerate() {
                // The kind is the label without its block suffix, so
                // `u_tech` and `u_banks` are both `u` to a reader of the
                // index.
                let kind = l.split('_').next().unwrap_or(l).to_string();
                let mut m = FieldMeta::new(format!("{l}{suffix}"), &kind)
                    .decay(suffix, d)
                    .src(Source::Stat(mi * n_slots + slot));
                m.columns = Some(cols_of(slot));
                fields.push(m);
            }
            fields.push(
                FieldMeta::new(format!("weight_sum{suffix}"), "weight_sum")
                    .decay(suffix, d)
                    .src(Source::NEff(mi)),
            );
            fields.push(
                FieldMeta::new(format!("coef{suffix}"), "coef")
                    .decay(suffix, d)
                    .src(Source::Coef(mi)),
            );
        }
        return fields;
    }
    // ew_cov is not a regression: its slots are named statistics, not
    // pred/resid pairs, and it has no targets or coefficients.
    if let crate::ModelKind::EwCov {
        stats,
        mahal_quantiles,
        pca,
        lags,
        ..
    } = &spec.model
    {
        let names = stats
            .clone()
            .unwrap_or_else(|| vec!["mean".into(), "std".into(), "corr".into()]);
        let kinds: Vec<rheon_core::EwCovStat> = names
            .iter()
            .map(|s| match s.as_str() {
                "mean" => rheon_core::EwCovStat::Mean,
                "var" => rheon_core::EwCovStat::Var,
                "std" => rheon_core::EwCovStat::Std,
                "cov" => rheon_core::EwCovStat::Cov,
                "partial_corr" => rheon_core::EwCovStat::PartialCorr,
                "mahal" => rheon_core::EwCovStat::Mahal,
                "lag_corr" => rheon_core::EwCovStat::LagCorr,
                _ => rheon_core::EwCovStat::Corr,
            })
            .collect();
        let levels: Vec<f64> = mahal_quantiles.clone().unwrap_or_default();
        let r = pca.unwrap_or(0);
        let lags: Vec<usize> = lags.clone().unwrap_or_default();
        let labels = rheon_core::EwCovModel::labels(&spec.features, &kinds, &levels, r, &lags);
        // Statistic kind, the columns it is over and its quantile level, in
        // label order: the same walk `labels` makes (per stat: each column,
        // each i<j pair, or all of them; then the levels; then `k + 3` per
        // component, all over every column).
        let all = spec.features.clone();
        // Statistic kind, the columns it is over, its quantile level, its
        // lag and its principal component, in label order.
        type StatMeta = (
            String,
            Vec<String>,
            Option<f64>,
            Option<usize>,
            Option<usize>,
        );
        let mut meta: Vec<StatMeta> = Vec::new();
        for (name, kind) in names.iter().zip(&kinds) {
            match kind {
                rheon_core::EwCovStat::Mean
                | rheon_core::EwCovStat::Var
                | rheon_core::EwCovStat::Std => {
                    for col in &spec.features {
                        meta.push((name.clone(), vec![col.clone()], None, None, None));
                    }
                }
                rheon_core::EwCovStat::Mahal => {
                    meta.push((name.clone(), all.clone(), None, None, None));
                }
                // One entry per (lag, ordered pair), the auto terms
                // included: both orientations, since the lagged matrix is
                // not symmetric.
                rheon_core::EwCovStat::LagCorr => {
                    for &l in &lags {
                        for a in &spec.features {
                            for b in &spec.features {
                                meta.push((
                                    name.clone(),
                                    vec![a.clone(), b.clone()],
                                    None,
                                    Some(l),
                                    None,
                                ));
                            }
                        }
                    }
                }
                _ => {
                    for i in 0..spec.features.len() {
                        for j in (i + 1)..spec.features.len() {
                            meta.push((
                                name.clone(),
                                vec![spec.features[i].clone(), spec.features[j].clone()],
                                None,
                                None,
                                None,
                            ));
                        }
                    }
                }
            }
        }
        for &q in &levels {
            meta.push(("mahal_q".into(), all.clone(), Some(q), None, None));
        }
        for j in 0..r {
            let pc = Some(j);
            meta.push(("pc_var".into(), all.clone(), None, None, pc));
            meta.push(("pc_share".into(), all.clone(), None, None, pc));
            for col in &spec.features {
                meta.push(("pc_loading".into(), vec![col.clone()], None, None, pc));
            }
            meta.push(("pc_score".into(), all.clone(), None, None, pc));
        }
        debug_assert_eq!(meta.len(), labels.len());
        let n_slots = labels.len();
        let mut fields = Vec::new();
        for (mi, (suffix, d)) in decays.iter().enumerate() {
            for (slot, (l, (kind, cols, q, lag, component))) in labels.iter().zip(&meta).enumerate()
            {
                let mut m = FieldMeta::new(format!("{l}{suffix}"), kind)
                    .decay(suffix, d)
                    .src(Source::Stat(mi * n_slots + slot));
                m.columns = Some(cols.clone());
                m.quantile = *q;
                m.lag = *lag;
                m.component = *component;
                fields.push(m);
            }
            fields.push(
                FieldMeta::new(format!("weight_sum{suffix}"), "weight_sum")
                    .decay(suffix, d)
                    .src(Source::NEff(mi)),
            );
        }
        return fields;
    }
    // pca is not a regression: per component its variance, its share, its
    // loadings and the row's score, named as `ew_cov`'s components are, and
    // `n_eff`; no targets or coefficients.
    if matches!(spec.model, crate::ModelKind::Pca { .. }) {
        let r = crate::stream::pca_cfg(spec, rheon_core::Decay::Halflife(f64::INFINITY))
            .map_or(0, |c| c.n_components);
        let all = spec.features.clone();
        let labels = rheon_core::EwCovModel::labels(&spec.features, &[], &[], r, &[]);
        let mut meta: Vec<(&str, Vec<String>, usize)> = Vec::new();
        for j in 0..r {
            meta.push(("pc_var", all.clone(), j));
            meta.push(("pc_share", all.clone(), j));
            for col in &spec.features {
                meta.push(("pc_loading", vec![col.clone()], j));
            }
            meta.push(("pc_score", all.clone(), j));
        }
        let n_slots = labels.len();
        let mut fields = Vec::new();
        for (mi, (suffix, d)) in decays.iter().enumerate() {
            for (slot, (l, (kind, cols, j))) in labels.iter().zip(&meta).enumerate() {
                let mut m = FieldMeta::new(format!("{l}{suffix}"), kind)
                    .decay(suffix, d)
                    .src(Source::Stat(mi * n_slots + slot));
                m.columns = Some(cols.clone());
                m.component = Some(*j);
                fields.push(m);
            }
            fields.push(
                FieldMeta::new(format!("weight_sum{suffix}"), "weight_sum")
                    .decay(suffix, d)
                    .src(Source::NEff(mi)),
            );
        }
        return fields;
    }
    // kmeans is not a regression either: per instance, the nearest centre
    // and two distances (read before the row is learned), `n_eff`, and the
    // centres as `coef`.
    if matches!(spec.model, crate::ModelKind::KMeans { .. }) {
        let n_slots = 3;
        let mut fields = Vec::new();
        for (mi, (suffix, d)) in decays.iter().enumerate() {
            let over = |mut f: FieldMeta| {
                f.columns = Some(spec.features.clone());
                f
            };
            fields.push(over(
                FieldMeta::new(format!("cluster{suffix}"), "cluster")
                    .decay(suffix, d)
                    .src(Source::Id(mi * n_slots)),
            ));
            fields.push(over(
                FieldMeta::new(format!("dist{suffix}"), "dist")
                    .decay(suffix, d)
                    .src(Source::Stat(mi * n_slots + 1)),
            ));
            fields.push(over(
                FieldMeta::new(format!("dist_second{suffix}"), "dist_second")
                    .decay(suffix, d)
                    .src(Source::Stat(mi * n_slots + 2)),
            ));
            fields.push(
                FieldMeta::new(format!("weight_sum{suffix}"), "weight_sum")
                    .decay(suffix, d)
                    .src(Source::NEff(mi)),
            );
            fields.push(
                FieldMeta::new(format!("coef{suffix}"), "coef")
                    .decay(suffix, d)
                    .src(Source::Coef(mi)),
            );
        }
        return fields;
    }
    // micro: per instance, the nearest cluster's label and distance, the
    // micro-cluster id the row goes to, an outlier flag, two counts (all
    // read before the row is learned), `n_eff`, and the potential summaries
    // as `coef`.
    if matches!(spec.model, crate::ModelKind::Micro { .. }) {
        let n_slots = 6;
        let mut fields = Vec::new();
        for (mi, (suffix, d)) in decays.iter().enumerate() {
            let over = |mut f: FieldMeta| {
                f.columns = Some(spec.features.clone());
                f
            };
            let at = |slot: usize| mi * n_slots + slot;
            fields.push(over(
                FieldMeta::new(format!("cluster{suffix}"), "cluster")
                    .decay(suffix, d)
                    .src(Source::Id(at(0))),
            ));
            fields.push(over(
                FieldMeta::new(format!("dist{suffix}"), "dist")
                    .decay(suffix, d)
                    .src(Source::Stat(at(1))),
            ));
            fields.push(over(
                FieldMeta::new(format!("micro_id{suffix}"), "micro_id")
                    .decay(suffix, d)
                    .src(Source::Id(at(2))),
            ));
            fields.push(over(
                FieldMeta::new(format!("outlier{suffix}"), "outlier")
                    .decay(suffix, d)
                    .src(Source::Flag(at(3))),
            ));
            fields.push(over(
                FieldMeta::new(format!("n_clusters{suffix}"), "n_clusters")
                    .decay(suffix, d)
                    .src(Source::Id(at(4))),
            ));
            fields.push(over(
                FieldMeta::new(format!("n_micro{suffix}"), "n_micro")
                    .decay(suffix, d)
                    .src(Source::Id(at(5))),
            ));
            fields.push(
                FieldMeta::new(format!("weight_sum{suffix}"), "weight_sum")
                    .decay(suffix, d)
                    .src(Source::NEff(mi)),
            );
            fields.push(
                FieldMeta::new(format!("coef{suffix}"), "coef")
                    .decay(suffix, d)
                    .src(Source::Coef(mi)),
            );
        }
        return fields;
    }
    // bocpd's value is a posterior over run lengths: per instance, the
    // changepoint mass, the run-length mode and mean, a predictive mean per
    // column and the row's log score.
    if matches!(spec.model, crate::ModelKind::Bocpd { .. }) {
        let labels = rheon_core::Bocpd::labels(&spec.features);
        let n_slots = labels.len();
        // `pred_<column>` follows the three run-length fields, one per
        // feature in order: each is about its own column alone.
        let per_feature = 3..3 + spec.features.len();
        let mut fields = Vec::new();
        for (mi, (suffix, d)) in decays.iter().enumerate() {
            for (slot, l) in labels.iter().enumerate() {
                let at = mi * n_slots + slot;
                let src = if l == "run_mode" {
                    Source::Id(at)
                } else {
                    Source::Stat(at)
                };
                // The run-length fields keep their full stems; a predictive
                // mean is `pred` about its column, as `columns` says.
                let kind = if per_feature.contains(&slot) {
                    "pred"
                } else {
                    l.as_str()
                };
                let mut m = FieldMeta::new(format!("{l}{suffix}"), kind)
                    .decay(suffix, d)
                    .src(src);
                m.columns = Some(if per_feature.contains(&slot) {
                    vec![spec.features[slot - 3].clone()]
                } else {
                    spec.features.clone()
                });
                fields.push(m);
            }
            fields.push(
                FieldMeta::new(format!("weight_sum{suffix}"), "weight_sum")
                    .decay(suffix, d)
                    .src(Source::NEff(mi)),
            );
        }
        return fields;
    }
    // corrchange is a test, not a model of the data: per instance the
    // statistic, its critical value, the flag, the rows since the last one
    // and, on a flag, the rows since the change it dates. Nothing is
    // reported except where a statistic is due.
    if matches!(spec.model, crate::ModelKind::CorrChange { .. }) {
        let labels = rheon_core::CorrChange::labels();
        let n_slots = labels.len();
        let mut fields = Vec::new();
        for (mi, (suffix, d)) in decays.iter().enumerate() {
            for (slot, l) in labels.iter().enumerate() {
                let at = mi * n_slots + slot;
                let src = match l.as_str() {
                    "flag" => Source::Flag(at),
                    "since_flag" | "since_change" => Source::Id(at),
                    _ => Source::Stat(at),
                };
                let mut m = FieldMeta::new(format!("{l}{suffix}"), l)
                    .decay(suffix, d)
                    .src(src);
                m.columns = Some(spec.features.clone());
                fields.push(m);
            }
            fields.push(
                FieldMeta::new(format!("weight_sum{suffix}"), "weight_sum")
                    .decay(suffix, d)
                    .src(Source::NEff(mi)),
            );
        }
        return fields;
    }
    // hmm's state is hidden: per instance, the filtered and predicted
    // posteriors, the state, the row's log-likelihood, `n_eff` and the
    // state means as `coef`.
    if let crate::ModelKind::Hmm { k, .. } = &spec.model {
        let labels = rheon_core::Hmm::labels(*k);
        let n_slots = labels.len();
        let mut fields = Vec::new();
        for (mi, (suffix, d)) in decays.iter().enumerate() {
            for (slot, l) in labels.iter().enumerate() {
                let at = mi * n_slots + slot;
                // `state` is a small count, so it rides in `pred` and comes
                // out as an `i64`, as every integer output does.
                let src = if l == "state" {
                    Source::Id(at)
                } else {
                    Source::Stat(at)
                };
                // `filtered_state<s>` and `predicted_state<s>`: the kind, and
                // the state they are about.
                let (kind, state) = match l.split_once("_state") {
                    Some((kind, s)) => (kind, s.parse().ok()),
                    None => (l.as_str(), None),
                };
                let mut m = FieldMeta::new(format!("{l}{suffix}"), kind)
                    .decay(suffix, d)
                    .src(src);
                m.columns = Some(spec.features.clone());
                m.state = state;
                fields.push(m);
            }
            fields.push(
                FieldMeta::new(format!("weight_sum{suffix}"), "weight_sum")
                    .decay(suffix, d)
                    .src(Source::NEff(mi)),
            );
            fields.push(
                FieldMeta::new(format!("coef{suffix}"), "coef")
                    .decay(suffix, d)
                    .src(Source::Coef(mi)),
            );
        }
        return fields;
    }
    // Classification models predict a label, not a number: per instance, the
    // class the row is assigned to and one posterior per class (read before
    // the row is learned), `weight_sum`, and (where `has_coef()`) `coef`.
    if let Some(classes) = match &spec.model {
        crate::ModelKind::EwClass { classes, .. } => Some(classes),
        crate::ModelKind::HoeffdingTreeClassifier(t) => Some(&t.classes),
        _ => None,
    } {
        let n_slots = 1 + classes.len();
        let mut fields = Vec::new();
        for (mi, (suffix, d)) in decays.iter().enumerate() {
            let over = |mut f: FieldMeta| {
                f.columns = Some(spec.features.clone());
                f
            };
            fields.push(over(
                FieldMeta::new(format!("class{suffix}"), "class")
                    .decay(suffix, d)
                    .target(&spec.targets[0])
                    .categories(classes)
                    .src(Source::Label(mi * n_slots)),
            ));
            for (c, class) in classes.iter().enumerate() {
                let mut p = over(
                    FieldMeta::new(format!("p_{class}{suffix}"), "p")
                        .decay(suffix, d)
                        .target(&spec.targets[0])
                        .src(Source::Stat(mi * n_slots + 1 + c)),
                );
                p.class = Some(class.clone());
                fields.push(p);
            }
            fields.push(
                FieldMeta::new(format!("weight_sum{suffix}"), "weight_sum")
                    .decay(suffix, d)
                    .src(Source::NEff(mi)),
            );
            if spec.model.has_coef() {
                fields.push(
                    FieldMeta::new(format!("coef{suffix}"), "coef")
                        .decay(suffix, d)
                        .src(Source::Coef(mi)),
                );
            }
        }
        return fields;
    }
    // bandit recommends an arm and scores every arm (read before the row is
    // learned), plus `prob` under policy="eps_greedy", `weight_sum`, and per-arm
    // linear coefficients in `coef`.
    if let crate::ModelKind::Bandit(b) = &spec.model {
        let eps = crate::spec::is_eps_greedy(b);
        let n_slots = 2 + b.arms.len() + usize::from(eps);
        let reward = &spec.targets[0];
        let mut fields = Vec::new();
        for (mi, (suffix, d)) in decays.iter().enumerate() {
            let over = |mut f: FieldMeta| {
                f.columns = Some(spec.features.clone());
                f
            };
            fields.push(over(
                FieldMeta::new(format!("arm{suffix}"), "arm")
                    .decay(suffix, d)
                    .target(reward)
                    .categories(&b.arms)
                    .src(Source::Label(mi * n_slots)),
            ));
            fields.push(over(
                FieldMeta::new(format!("score{suffix}"), "score")
                    .decay(suffix, d)
                    .target(reward)
                    .src(Source::Stat(mi * n_slots + 1)),
            ));
            for (ai, arm) in b.arms.iter().enumerate() {
                let mut f = over(
                    FieldMeta::new(format!("score_{arm}{suffix}"), "score")
                        .decay(suffix, d)
                        .target(reward)
                        .src(Source::Stat(mi * n_slots + 2 + ai)),
                );
                f.arm = Some(arm.clone());
                fields.push(f);
            }
            if eps {
                fields.push(over(
                    FieldMeta::new(format!("prob{suffix}"), "prob")
                        .decay(suffix, d)
                        .target(reward)
                        .src(Source::Stat(mi * n_slots + 2 + b.arms.len())),
                ));
            }
            fields.push(
                FieldMeta::new(format!("weight_sum{suffix}"), "weight_sum")
                    .decay(suffix, d)
                    .src(Source::NEff(mi)),
            );
            fields.push(
                FieldMeta::new(format!("coef{suffix}"), "coef")
                    .decay(suffix, d)
                    .src(Source::Coef(mi)),
            );
        }
        return fields;
    }
    // Streaming anomaly detectors (`half_space_trees` and `random_cut_forest`)
    // emit one unsupervised anomaly `score` per row (read before learning) and
    // `weight_sum`. No `coef`.
    if matches!(
        spec.model,
        crate::ModelKind::HalfSpaceTrees(_) | crate::ModelKind::RandomCutForest(_)
    ) {
        let mut fields = Vec::new();
        for (mi, (suffix, d)) in decays.iter().enumerate() {
            let mut s = FieldMeta::new(format!("score{suffix}"), "score")
                .decay(suffix, d)
                .src(Source::Stat(mi));
            s.columns = Some(spec.features.clone());
            fields.push(s);
            fields.push(
                FieldMeta::new(format!("weight_sum{suffix}"), "weight_sum")
                    .decay(suffix, d)
                    .src(Source::NEff(mi)),
            );
        }
        return fields;
    }
    // hawkes predicts no target's value: per instance, each node's
    // intensity before the row's events, then `n_eff` and the fit as `coef`.
    if matches!(spec.model, crate::ModelKind::Hawkes { .. }) {
        let m = spec.targets.len();
        let mut fields = Vec::new();
        for (mi, (suffix, d)) in decays.iter().enumerate() {
            for (t_i, t) in spec.targets.iter().enumerate() {
                fields.push(
                    FieldMeta::new(format!("intensity_{t}{suffix}"), "intensity")
                        .decay(suffix, d)
                        .target(t)
                        .src(Source::Stat(mi * m + t_i)),
                );
            }
            fields.push(
                FieldMeta::new(format!("weight_sum{suffix}"), "weight_sum")
                    .decay(suffix, d)
                    .src(Source::NEff(mi)),
            );
            fields.push(
                FieldMeta::new(format!("coef{suffix}"), "coef")
                    .decay(suffix, d)
                    .src(Source::Coef(mi)),
            );
        }
        return fields;
    }
    // dcc predicts no target's value either: with margins each target's
    // variance, then each pair's correlation and the row's log-density
    // (read before the row is learned); then `n_eff` and the parameters as
    // `coef`. An instance a decay.
    if matches!(spec.model, crate::ModelKind::Dcc { .. }) {
        let margins = !matches!(
            &spec.model,
            crate::ModelKind::Dcc { margins: Some(m), .. } if m == "none"
        );
        let k = spec.targets.len();
        let n_slots = if margins { k } else { 0 } + k * (k - 1) / 2 + 1;
        let mut fields = Vec::new();
        for (mi, (suffix, d)) in decays.iter().enumerate() {
            let mut at = mi * n_slots;
            if margins {
                for t in &spec.targets {
                    fields.push(
                        FieldMeta::new(format!("variance_{t}{suffix}"), "variance")
                            .decay(suffix, d)
                            .target(t)
                            .src(Source::Stat(at)),
                    );
                    at += 1;
                }
            }
            for i in 0..k {
                for j in (i + 1)..k {
                    let (a, b) = (&spec.targets[i], &spec.targets[j]);
                    let mut m = FieldMeta::new(format!("corr_{a}_{b}{suffix}"), "corr")
                        .decay(suffix, d)
                        .src(Source::Stat(at));
                    m.columns = Some(vec![a.clone(), b.clone()]);
                    fields.push(m);
                    at += 1;
                }
            }
            let mut m = FieldMeta::new(format!("loglik{suffix}"), "loglik")
                .decay(suffix, d)
                .src(Source::Stat(at));
            m.columns = Some(spec.targets.iter().cloned().collect());
            fields.push(m);
            fields.push(
                FieldMeta::new(format!("weight_sum{suffix}"), "weight_sum")
                    .decay(suffix, d)
                    .src(Source::NEff(mi)),
            );
            fields.push(
                FieldMeta::new(format!("coef{suffix}"), "coef")
                    .decay(suffix, d)
                    .src(Source::Coef(mi)),
            );
        }
        return fields;
    }
    // realized predicts no target's value: per instance and target, the
    // estimators of the returns before the row, Lee and Mykland's statistic
    // of the row's own return and its flag; then `n_eff`. No `coef`.
    if matches!(spec.model, crate::ModelKind::Realized { .. }) {
        const NAMES: [&str; rheon_core::REALIZED_OUTPUTS] =
            ["rv", "bv", "tv", "tq", "bns", "lm", "lm_jump"];
        let n_slots = NAMES.len();
        let mut fields = Vec::new();
        for (mi, (suffix, d)) in decays.iter().enumerate() {
            let base = mi * spec.targets.len() * n_slots;
            for (t_i, t) in spec.targets.iter().enumerate() {
                for (slot, name) in NAMES.iter().enumerate() {
                    let at = base + t_i * n_slots + slot;
                    let src = if *name == "lm_jump" {
                        Source::Flag(at)
                    } else {
                        Source::Stat(at)
                    };
                    fields.push(
                        FieldMeta::new(format!("{name}_{t}{suffix}"), name)
                            .decay(suffix, d)
                            .target(t)
                            .src(src),
                    );
                }
            }
            fields.push(
                FieldMeta::new(format!("weight_sum{suffix}"), "weight_sum")
                    .decay(suffix, d)
                    .src(Source::NEff(mi)),
            );
        }
        return fields;
    }
    // power_variation predicts no target's value: per instance and target,
    // each named term of the returns before the row, then their count `n`;
    // then `n_eff`. No `coef`.
    if let crate::ModelKind::PowerVariation { terms, .. } = &spec.model {
        let names: Vec<&str> = terms.iter().map(|t| t.name.as_str()).chain(["n"]).collect();
        let n_slots = names.len();
        let mut fields = Vec::new();
        for (mi, (suffix, d)) in decays.iter().enumerate() {
            let base = mi * spec.targets.len() * n_slots;
            for (t_i, t) in spec.targets.iter().enumerate() {
                for (slot, name) in names.iter().enumerate() {
                    fields.push(
                        FieldMeta::new(format!("{name}_{t}{suffix}"), name)
                            .decay(suffix, d)
                            .target(t)
                            .src(Source::Stat(base + t_i * n_slots + slot)),
                    );
                }
            }
            fields.push(
                FieldMeta::new(format!("weight_sum{suffix}"), "weight_sum")
                    .decay(suffix, d)
                    .src(Source::NEff(mi)),
            );
        }
        return fields;
    }
    // ewm_moments predicts no target's value: per instance, each term in the
    // order the spec names them, then under `se` each term's standard error
    // in the same order, then `n_eff` (the weight inside the window under
    // one). Each is read before the row is learned; no `coef`.
    if let crate::ModelKind::EwmMoments { terms, se, .. } = &spec.model {
        let se = se.unwrap_or(false);
        let n_slots = terms.0.len() * if se { 2 } else { 1 };
        let mut fields = Vec::new();
        for (mi, (suffix, d)) in decays.iter().enumerate() {
            let columns = |legs: &crate::spec::MomentLegs| {
                let mut cols: Vec<String> = Vec::new();
                for (key, _) in &legs.0 {
                    let col = crate::spec::moment_leg_column(&spec.features, key);
                    if !cols.contains(&col) {
                        cols.push(col);
                    }
                }
                cols
            };
            let names = terms.0.iter().map(|(t, legs)| (t.clone(), "moment", legs));
            let ses = terms
                .0
                .iter()
                .filter(|_| se)
                .map(|(t, legs)| (format!("se_{t}"), "se_moment", legs));
            for (slot, (name, kind, legs)) in names.chain(ses).enumerate() {
                let mut f = FieldMeta::new(format!("{name}{suffix}"), kind)
                    .decay(suffix, d)
                    .src(Source::Stat(mi * n_slots + slot));
                f.columns = Some(columns(legs));
                fields.push(f);
            }
            fields.push(
                FieldMeta::new(format!("weight_sum{suffix}"), "weight_sum")
                    .decay(suffix, d)
                    .src(Source::NEff(mi)),
            );
        }
        return fields;
    }
    // garch predicts no target's value: per target, the variance and the
    // standardized residual (read before the row is learned) and, at each
    // horizon, the variance there and the sum of the variances to it; then
    // `n_eff` and the parameters as `coef`. One undecayed instance.
    if let crate::ModelKind::Garch { horizons, .. } = &spec.model {
        let labels: Vec<String> = horizons
            .iter()
            .flatten()
            .map(crate::Horizon::label)
            .collect();
        let n_slots = 2 + 2 * labels.len();
        let mut fields = Vec::new();
        for (mi, (suffix, d)) in decays.iter().enumerate() {
            let base = mi * spec.targets.len() * n_slots;
            for (t_i, t) in spec.targets.iter().enumerate() {
                let at = |slot: usize| base + t_i * n_slots + slot;
                fields.push(
                    FieldMeta::new(format!("variance_{t}{suffix}"), "variance")
                        .decay(suffix, d)
                        .target(t)
                        .src(Source::Stat(at(0))),
                );
                fields.push(
                    FieldMeta::new(format!("std_resid_{t}{suffix}"), "std_resid")
                        .decay(suffix, d)
                        .target(t)
                        .src(Source::Stat(at(1))),
                );
                for (k, l) in labels.iter().enumerate() {
                    fields.push(
                        FieldMeta::new(format!("variance_{l}_{t}{suffix}"), "variance_at")
                            .decay(suffix, d)
                            .target(t)
                            .src(Source::Stat(at(2 + 2 * k))),
                    );
                    fields.push(
                        FieldMeta::new(format!("variance_sum_{l}_{t}{suffix}"), "variance_sum")
                            .decay(suffix, d)
                            .target(t)
                            .src(Source::Stat(at(3 + 2 * k))),
                    );
                }
            }
            fields.push(
                FieldMeta::new(format!("weight_sum{suffix}"), "weight_sum")
                    .decay(suffix, d)
                    .src(Source::NEff(mi)),
            );
            fields.push(
                FieldMeta::new(format!("coef{suffix}"), "coef")
                    .decay(suffix, d)
                    .src(Source::Coef(mi)),
            );
        }
        return fields;
    }
    // tdigest and ddsketch predict no target's value: per target, one quantile a level,
    // `q<level>_<target>`, then the tail mean past it a level,
    // `es<level>_<target>` (both read before the row is added), then
    // `n_eff`. One undecayed instance; no `coef`.
    if let crate::ModelKind::TDigest { quantiles, .. }
    | crate::ModelKind::DdSketch { quantiles, .. } = &spec.model
    {
        let mut fields = Vec::new();
        let nl = quantiles.len();
        for (mi, (suffix, d)) in decays.iter().enumerate() {
            let base = mi * spec.targets.len() * 2 * nl;
            for (t_i, t) in spec.targets.iter().enumerate() {
                for (part, role) in ["q", "es"].into_iter().enumerate() {
                    for (li, q) in quantiles.iter().enumerate() {
                        let name = format!("{role}{}_{t}{suffix}", crate::spec::num_label(*q));
                        let mut f = FieldMeta::new(name, role)
                            .decay(suffix, d)
                            .target(t)
                            .src(Source::Stat(base + (2 * t_i + part) * nl + li));
                        f.quantile = Some(*q);
                        fields.push(f);
                    }
                }
            }
            fields.push(
                FieldMeta::new(format!("weight_sum{suffix}"), "weight_sum")
                    .decay(suffix, d)
                    .src(Source::NEff(mi)),
            );
        }
        return fields;
    }
    // spectral: per target and period, the power read before the row, then
    // `n_eff`; no `coef`. Its periods are named by their place in the list.
    if let crate::ModelKind::Spectral(s) = &spec.model {
        let n_slots = s.periods.spans().len();
        let mut fields = Vec::new();
        for (mi, (suffix, d)) in decays.iter().enumerate() {
            let base = mi * spec.targets.len() * n_slots;
            for (t_i, t) in spec.targets.iter().enumerate() {
                for k in 0..n_slots {
                    let mut m = FieldMeta::new(format!("power_{t}_p{k}{suffix}"), "power")
                        .decay(suffix, d)
                        .target(t)
                        .src(Source::Stat(base + t_i * n_slots + k));
                    // The period's place in `periods` tells the fields apart.
                    m.component = Some(k);
                    fields.push(m);
                }
            }
            fields.push(
                FieldMeta::new(format!("weight_sum{suffix}"), "weight_sum")
                    .decay(suffix, d)
                    .src(Source::NEff(mi)),
            );
        }
        return fields;
    }
    // dependence: per (measure, feature, target), the nonlinear dependence
    // measure read before the row, then `weight_sum`; no `coef`.
    // Internally `Step::pred` is target-outer: `(t * n_measures + m_idx) * k + j`.
    if let crate::ModelKind::Dependence(d) = &spec.model {
        let measures = d.measures_for_fields();
        let nm = measures.len();
        let k = spec.features.len();
        let stride = spec.targets.len() * nm * k;
        let mut fields = Vec::new();
        for (mi, (suffix, decay)) in decays.iter().enumerate() {
            let base = mi * stride;
            for (m_idx, measure) in measures.iter().enumerate() {
                let kind = measure.as_str();
                for (j, f) in spec.features.iter().enumerate() {
                    for (t_i, t) in spec.targets.iter().enumerate() {
                        let mut fm = FieldMeta::new(format!("{kind}_{f}_{t}{suffix}"), kind)
                            .decay(suffix, decay)
                            .target(t)
                            .src(Source::Stat(base + (t_i * nm + m_idx) * k + j));
                        fm.columns = Some(vec![f.clone()]);
                        fields.push(fm);
                    }
                }
            }
            fields.push(
                FieldMeta::new(format!("weight_sum{suffix}"), "weight_sum")
                    .decay(suffix, decay)
                    .src(Source::NEff(mi)),
            );
        }
        return fields;
    }
    // seqtest predicts nothing: per target, the two log e-values and the two
    // sign counts they are staked on (read before the row is learned), then
    // `n_eff`. One undecayed instance, so no suffix and no decay on the
    // fields; no `coef`. A comparison names its fields by the two sides.
    if let crate::ModelKind::SeqTest { .. } = &spec.model {
        let n_slots = rheon_core::SEQTEST_SLOTS;
        let names: [&str; 4] = if spec.model.compares().is_some() {
            ["log_e_a", "log_e_b", "wins_a", "wins_b"]
        } else {
            ["log_e_pos", "log_e_neg", "n_pos", "n_neg"]
        };
        // One undecayed instance: its suffix is empty, as a single
        // instance's is.
        let one = |mut m: FieldMeta| {
            m.instance = Some(String::new());
            m
        };
        let mut fields = Vec::new();
        for (t_i, t) in spec.targets.iter().enumerate() {
            let at = |slot: usize| t_i * n_slots + slot;
            fields.push(one(FieldMeta::new(format!("{}_{t}", names[0]), names[0])
                .target(t)
                .src(Source::Stat(at(0)))));
            fields.push(one(FieldMeta::new(format!("{}_{t}", names[1]), names[1])
                .target(t)
                .src(Source::Stat(at(1)))));
            fields.push(one(FieldMeta::new(format!("{}_{t}", names[2]), names[2])
                .target(t)
                .src(Source::Id(at(2)))));
            fields.push(one(FieldMeta::new(format!("{}_{t}", names[3]), names[3])
                .target(t)
                .src(Source::Id(at(3)))));
        }
        fields.push(one(
            FieldMeta::new("weight_sum".into(), "weight_sum").src(Source::NEff(0))
        ));
        return fields;
    }
    // audit with HLL++ enabled (`distinct_precision` set) emits each audited
    // column's out-of-sample estimated distinct count (`approx_n_unique_<col>`)
    // followed by `weight_sum`. Without `distinct_precision` it stays
    // state-only below (`weight_sum` alone).
    if let crate::ModelKind::Audit {
        distinct_precision: Some(_),
        ..
    } = &spec.model
    {
        let mut fields = Vec::with_capacity(spec.features.len() + 1);
        for (mi, (suffix, d)) in decays.iter().enumerate() {
            let base = mi * spec.features.len();
            for (ci, col) in spec.features.iter().enumerate() {
                let mut m =
                    FieldMeta::new(format!("approx_n_unique_{col}{suffix}"), "approx_n_unique")
                        .decay(suffix, d)
                        .src(Source::Stat(base + ci));
                m.columns = Some(vec![col.clone()]);
                fields.push(m);
            }
            fields.push(
                FieldMeta::new(format!("weight_sum{suffix}"), "weight_sum")
                    .decay(suffix, d)
                    .src(Source::NEff(mi)),
            );
        }
        return fields;
    }
    // rcov, marginal and audit emit nothing per row but `n_eff`, one per
    // instance: rcov's value is the block it emits at the group's close,
    // marginal's the pairs `Bank::marginal` reads from the state, audit's
    // the counts `Bank::audit` reads.
    if matches!(
        spec.model,
        crate::ModelKind::Marginal { .. }
            | crate::ModelKind::Rcov { .. }
            | crate::ModelKind::Audit { .. }
    ) {
        return decays
            .iter()
            .enumerate()
            .map(|(mi, (suffix, d))| {
                FieldMeta::new(format!("weight_sum{suffix}"), "weight_sum")
                    .decay(suffix, d)
                    .src(Source::NEff(mi))
            })
            .collect();
    }
    let combos = crate::stream::combos(spec);
    let (nc, m, n_models) = (combos.len(), spec.m(), decays.len());
    let mut fields = Vec::new();
    for (mi, (suffix, d)) in decays.iter().enumerate() {
        // `dst` is the flat (instance, target, combo) index every per-slot
        // buffer in `assemble` is laid out by. Computing it here, once, is
        // what lets `assemble` be a walk over this vector instead of the same
        // nested loops written a second time.
        let mk = |kind: &str, t: &str, c: &crate::stream::Combo, src: Source| {
            FieldMeta::new(format!("{kind}_{t}{}{suffix}", c.label), kind)
                .decay(suffix, d)
                .target(t)
                .combo(c)
                .src(src)
        };
        // An instance's slots: its predictions, then its forecasts at
        // horizons (`holt`, `kalman`), each target's together.
        let horizons = spec.forecast_labels();
        let stride = m * nc + spec.forecast_slots();
        let dst = |t_i: usize, c_i: usize| mi * stride + t_i * nc + c_i;
        for (t_i, t) in spec.targets.iter().enumerate() {
            for (c_i, c) in combos.iter().enumerate() {
                fields.push(mk("pred", t, c, Source::Pred(dst(t_i, c_i))));
                fields.push(mk("resid", t, c, Source::Resid(dst(t_i, c_i))));
            }
        }
        let kinds = spec.forecast_kinds();
        for (t_i, t) in spec.targets.iter().enumerate() {
            let at = mi * stride + m * nc + t_i * kinds.len() * horizons.len();
            for (k, l) in horizons.iter().enumerate() {
                for (slot, kind) in kinds.iter().enumerate() {
                    fields.push(
                        FieldMeta::new(format!("{kind}_{l}_{t}{suffix}"), kind)
                            .decay(suffix, d)
                            .target(t)
                            .src(Source::Stat(at + kinds.len() * k + slot)),
                    );
                }
            }
        }
        if spec.emit_error_inflation {
            for (t_i, t) in spec.targets.iter().enumerate() {
                for (c_i, c) in combos.iter().enumerate() {
                    fields.push(mk(
                        "error_inflation",
                        t,
                        c,
                        Source::Inflation(dst(t_i, c_i)),
                    ));
                }
            }
        }
        if spec.emit_sigma {
            for (t_i, t) in spec.targets.iter().enumerate() {
                for (c_i, c) in combos.iter().enumerate() {
                    fields.push(mk("sigma", t, c, Source::Sigma(dst(t_i, c_i))));
                }
            }
        }
        if spec.emit_zscore {
            for (t_i, t) in spec.targets.iter().enumerate() {
                for (c_i, c) in combos.iter().enumerate() {
                    fields.push(mk("zscore", t, c, Source::ResidZ(dst(t_i, c_i))));
                }
            }
        }
        if spec.conformal.is_some() {
            // Not `pred_lo`: `pred_` is the prefix that marks a prediction,
            // for `eval.unpack` and for the README's grammar alike.
            for (k, name) in ["lo", "hi", "coverage"].into_iter().enumerate() {
                for (t_i, t) in spec.targets.iter().enumerate() {
                    for (c_i, c) in combos.iter().enumerate() {
                        fields.push(mk(name, t, c, Source::Conformal(k, dst(t_i, c_i))));
                    }
                }
            }
        }
        if spec.emit_metrics {
            for (k, name) in ["ic", "r2", "hit_rate"].into_iter().enumerate() {
                for (t_i, t) in spec.targets.iter().enumerate() {
                    for (c_i, c) in combos.iter().enumerate() {
                        fields.push(mk(name, t, c, Source::Metric(k, dst(t_i, c_i))));
                    }
                }
            }
        }
        if let Some(levels) = &spec.resid_quantiles {
            for (li, q) in levels.iter().enumerate() {
                for (t_i, t) in spec.targets.iter().enumerate() {
                    for (c_i, c) in combos.iter().enumerate() {
                        let name = format!(
                            "abs_resid_q{}_{t}{}{suffix}",
                            crate::spec::num_label(*q),
                            c.label
                        );
                        let idx = (li * n_models + mi) * m * nc + t_i * nc + c_i;
                        let mut f = FieldMeta::new(name, "abs_resid_q")
                            .decay(suffix, d)
                            .target(t)
                            .combo(c)
                            .src(Source::Quantile(idx));
                        f.quantile = Some(*q);
                        fields.push(f);
                    }
                }
            }
        }
        if spec.emit_autocorr {
            for (t_i, t) in spec.targets.iter().enumerate() {
                for (c_i, c) in combos.iter().enumerate() {
                    fields.push(mk("autocorr", t, c, Source::Autocorr(dst(t_i, c_i))));
                }
            }
        }
        if spec.emit_drift {
            for (t_i, t) in spec.targets.iter().enumerate() {
                for (c_i, c) in combos.iter().enumerate() {
                    fields.push(mk("drift", t, c, Source::Drift(dst(t_i, c_i))));
                }
            }
        }
        for (k, name) in crate::checks::value_names(spec).into_iter().enumerate() {
            for (t_i, t) in spec.targets.iter().enumerate() {
                for (c_i, c) in combos.iter().enumerate() {
                    fields.push(mk(name, t, c, Source::Check(k, dst(t_i, c_i))));
                }
            }
        }
        if spec.emit_feature_health {
            for (which, kind) in ["spread_ratio", "mean_shift"].into_iter().enumerate() {
                for (fi, f) in spec.features.iter().enumerate() {
                    // The feature it reads, so the index tells the fields
                    // apart (review round 6, F-9).
                    let mut m = FieldMeta::new(format!("{kind}_{f}{suffix}"), kind)
                        .decay(suffix, d)
                        .src(Source::Health(which, mi, fi));
                    m.columns = Some(vec![f.clone()]);
                    fields.push(m);
                }
            }
        }
        fields.push(
            FieldMeta::new(format!("weight_sum{suffix}"), "weight_sum")
                .decay(suffix, d)
                .src(Source::NEff(mi)),
        );
        // `isotonic` predicts a target from a curve, not coefficients: no
        // `coef`.
        if spec.model.has_coef() {
            fields.push(
                FieldMeta::new(format!("coef{suffix}"), "coef")
                    .decay(suffix, d)
                    .src(Source::Coef(mi)),
            );
        }
        if matches!(spec.model, crate::ModelKind::Lasso { .. }) {
            for (t_i, t) in spec.targets.iter().enumerate() {
                fields.push(
                    FieldMeta::new(format!("penalty_selected_{t}{suffix}"), "penalty_selected")
                        .decay(suffix, d)
                        .target(t)
                        .src(Source::LamSelected(mi * m + t_i)),
                );
            }
        }
        // `vector_ar`'s tests, a statistic and a p-value each, then the
        // order under `select`, in the same buffer as lasso's penalty.
        if let crate::ModelKind::VectorAr(v) = &spec.model {
            let w = extra_width(spec);
            let tests = v.causality.as_deref().unwrap_or_default();
            for i in 0..tests.len() {
                for (part, kind) in ["stat", "p"].into_iter().enumerate() {
                    let mut f = FieldMeta::new(
                        format!("causality{i}_{kind}{suffix}"),
                        &format!("causality_{kind}"),
                    )
                    .decay(suffix, d)
                    .src(Source::LamSelected(mi * w + 2 * i + part));
                    // The test's place in `causality`.
                    f.component = Some(i);
                    fields.push(f);
                }
            }
            if v.select.is_some() {
                fields.push(
                    FieldMeta::new(format!("order{suffix}"), "order")
                        .decay(suffix, d)
                        .src(Source::ExtraId(mi * w + w - 1)),
                );
            }
        }
        if matches!(&spec.model, crate::ModelKind::Krls(k) if k.emit_n_atoms) {
            fields.push(
                FieldMeta::new(format!("n_atoms{suffix}"), "n_atoms")
                    .decay(suffix, d)
                    .src(Source::ExtraId(mi)),
            );
        }
        // `huber`'s residual scale per target, in the same buffer as
        // lasso's selected penalty (docs/PLAN.md task 272 (a)).
        if matches!(
            spec.model,
            crate::ModelKind::Huber {
                emit_huber_scale: true,
                ..
            }
        ) {
            for (t_i, t) in spec.targets.iter().enumerate() {
                fields.push(
                    FieldMeta::new(format!("huber_scale_{t}{suffix}"), "huber_scale")
                        .decay(suffix, d)
                        .target(t)
                        .src(Source::LamSelected(mi * m + t_i)),
                );
            }
        }
    }
    if spec.emit_selected {
        for (t_i, t) in spec.targets.iter().enumerate() {
            fields.push(
                FieldMeta::new(format!("pred_{t}__selected"), "pred_selected")
                    .target(t)
                    .src(Source::SelPred(t_i)),
            );
            fields.push(
                FieldMeta::new(format!("selected_{t}"), "selected")
                    .target(t)
                    .categories(&slot_labels(spec))
                    .src(Source::SelName(t_i)),
            );
        }
    }
    if spec.emit_averaged {
        for (t_i, t) in spec.targets.iter().enumerate() {
            fields.push(
                FieldMeta::new(format!("pred_{t}__averaged"), "pred_averaged")
                    .target(t)
                    .src(Source::AvgPred(t_i)),
            );
        }
    }
    fields
}

/// Labels for every prediction slot of one target, across half-life instances
/// and combos, in the order the slots appear.
pub(super) fn slot_labels(spec: &Spec) -> Vec<String> {
    let decays = spec.decays().expect("validated");
    let combos = combo_labels(spec);
    let mut out = Vec::new();
    for (suffix, _) in &decays {
        for c in &combos {
            let label = format!("{c}{suffix}");
            out.push(if label.is_empty() {
                "default".to_string()
            } else {
                label.trim_start_matches("__").to_string()
            });
        }
    }
    out
}

/// Values a row writes per instance into the extras buffer: one per target
/// for `lasso`'s selected penalty and `huber`'s scale, and `vector_ar`'s
/// statistic and p-value per test with its order under `select`.
pub fn extra_width(spec: &Spec) -> usize {
    match &spec.model {
        crate::ModelKind::Lasso { .. }
        | crate::ModelKind::Huber {
            emit_huber_scale: true,
            ..
        } => spec.m(),
        crate::ModelKind::VectorAr(v) => {
            2 * v.causality.as_ref().map_or(0, Vec::len) + usize::from(v.select.is_some())
        }
        crate::ModelKind::Krls(k) if k.emit_n_atoms => 1,
        _ => 0,
    }
}
