//! Building a spec's models: one instance per half-life grid entry, each
//! model kind from its spec, and the defaults the Python layer leaves to us
//! (moved out of `stream.rs` under the 250 KB cap a source file is held to).

#[allow(unused_imports)]
use super::*;

/// The default `bins` the Python layer leaves to us: enough resolution to see
/// a threshold or a V, and small enough that the histogram stays a rounding
/// error next to the pair moments it rides along with. With `bin_edges` the
/// count is the edges' and this is not read.
const DEFAULT_BINS: usize = 16;

/// Learned rows held before the bin edges are fixed. A thousand rows put
/// roughly sixty in each of sixteen quantile bins -- enough for edges that do
/// not move much -- and hold 8 KB per feature while they wait.
const DEFAULT_BIN_WARM_ROWS: usize = 1_000;

/// The cut of `bin_stat = "huber"` in units of the target's robust scale
/// when the spec names none: `huber`'s default, the cut at which Huber's
/// estimator keeps 95% of the mean's efficiency on normal errors.
pub(crate) const DEFAULT_BIN_HUBER_DELTA: f64 = 1.345;

/// Build the model instances for a spec: one per half-life grid entry. A
/// model's own refusal names the spec, as `Spec::validate` names it for
/// `bocpd`'s and the other cfg builders': in a bank of several specs
/// `sgd: clip_gradient must be > 0` did not say which (review 2026-10-06,
/// YA4). `decays` names it already.
pub fn build_models(spec: &Spec) -> Result<Vec<(String, AnyModel)>, String> {
    let decays = spec.decays()?;
    decays
        .into_iter()
        .map(|(suffix, decay)| {
            // A model reports every problem with its settings, one to a
            // line; the spec names them once.
            let m = build_one(spec, decay).map_err(|e| {
                let mut no = rheon_core::Refusals::new();
                no.take(Err(e));
                crate::spec::render_refusal(&spec.name, spec.model.kind_name(), &no).unwrap_err()
            })?;
            Ok((suffix, m))
        })
        .collect()
}

/// `kmeans`' and `hmm`'s `init` as the core's [`SeedRule`]: scikit-learn's
/// words, `"k-means++"` among them (task 244), `"lloyd"` when absent.
pub(crate) fn init_rule(init: Option<&str>) -> Result<SeedRule, String> {
    match init {
        None | Some("lloyd") => Ok(SeedRule::Lloyd),
        Some("k-means++") => Ok(SeedRule::Kmeanspp),
        Some("farthest") => Ok(SeedRule::Farthest),
        Some("first") => Ok(SeedRule::First),
        Some(other) => Err(crate::words_refusal(
            "init",
            &crate::spec::INIT_WORDS,
            other,
        )),
    }
}

/// The slope constraint of `sgd` / `pa` from the spec's `coef_min`,
/// `coef_max` and `coef_sum`: a scalar bound is broadcast to every feature,
/// a list is taken as given (its length is checked by the model's config,
/// which names the offence). `None` when nothing is constrained.
/// A spec's `huber_scale` word as the model takes it: `"mad"` where none
/// is given, the default since task 271 (the word was validated with the
/// spec).
pub(super) fn huber_scale_of(word: Option<&str>) -> HuberScale {
    match word {
        Some("rms") => HuberScale::Rms,
        _ => HuberScale::Mad,
    }
}

pub(super) fn constraint(
    k: usize,
    coef_min: &Option<FloatOrList>,
    coef_max: &Option<FloatOrList>,
    coef_sum: Option<f64>,
) -> Option<Constraint> {
    if coef_min.is_none() && coef_max.is_none() && coef_sum.is_none() {
        return None;
    }
    let bounds = |b: &Option<FloatOrList>, none: f64| match b {
        None => vec![none; k],
        Some(FloatOrList::Float(v)) => vec![v.0; k],
        Some(list) => list.to_vec(),
    };
    let c = Constraint {
        lo: bounds(coef_min, f64::NEG_INFINITY),
        hi: bounds(coef_max, f64::INFINITY),
        sum: coef_sum,
    };
    // A list of the wrong length goes to the model's check, which names
    // it, whatever its bounds: dropped as trivial first, `coef_min =
    // ["-inf"]` with two features, or `[]`, passed (task 160, PB4).
    if c.lo.len() != k || c.hi.len() != k {
        return Some(c);
    }
    (!c.is_trivial()).then_some(c)
}

/// One model instance, its window bounded as its spec says
/// (`ModelKind::window_budget`). The budget is configuration, which a state
/// does not carry, so every model built -- a stream's, a reset's -- gets it
/// here, and every model restored gets it in [`Stream::restore`].
pub(super) fn build_one(spec: &Spec, decay: Decay) -> Result<AnyModel, String> {
    let mut m = build_bare(spec, decay)?;
    m.set_window_budget(spec.model.window_budget());
    // Threads are configuration too (docs/PLAN.md task 225).
    m.set_gram_threads(spec.model.gram_threads());
    // The window's edge travels with the state, so it is set once, here,
    // before the first row (docs/PLAN.md task 196, N17).
    m.set_window_closed(spec.model.window_edge());
    // An explicit solve cadence's first period runs on the default
    // weight-share schedule (task 278, "Default cadence first"); a model
    // with no explicit cadence leaves the mark off. It travels with the
    // state, so a state written before it goes on as it was fitted.
    m.set_first_period_by_share(true);
    Ok(m)
}

/// The ring a windowed spread is cut from (review 2026-09-12, S1), for a
/// windowed model that predicts a target, when anything reads the spread:
/// the output fields, the ranking of `emit_selected` and `emit_averaged`,
/// the drift detector or the conformal band. Without a reader it would be
/// memory, and a budget to fail, for nothing.
pub(super) fn resid_window(spec: &Spec) -> Result<Option<ResidWindow>, String> {
    let Some((window, cadence)) = spec.model.window_and_cadence() else {
        return Ok(None);
    };
    let read = Buffers::of(spec).extras || spec.emit_drift || spec.conformal.is_some();
    if spec.model.predicts_no_target() || !read {
        return Ok(None);
    }
    ResidWindow::new(
        window,
        cadence,
        spec.model.window_edge(),
        spec.model.window_budget(),
    )
    .map(Some)
}

fn build_bare(spec: &Spec, decay: Decay) -> Result<AnyModel, String> {
    match &spec.model {
        ModelKind::EwRidge {
            ridge,
            feature_sets,
            standardize,
            ridge_scale,
            coef_prior,
            session_shrink,
            long_half_life,
            solve_every,
            max_rows_between_solves,
            gram_block_rows,
            gram_threads: _,
            target_gaps,
            window_size: window,
            snapshot_every,
            max_rows_between_snapshots,
            window_budget: _,
            closed: _,
        } => {
            let fs = feature_sets
                .as_ref()
                .map(|sets| {
                    sets.iter()
                        .map(|(name, cols)| {
                            let idx = cols
                                .iter()
                                .map(|c| spec.features.iter().position(|f| f == c).unwrap())
                                .collect();
                            (name.clone(), idx)
                        })
                        .collect()
                })
                .unwrap_or_default();
            let cfg = EwRidgeCfg {
                n_features: spec.k(),
                n_targets: spec.m(),
                fit_intercept: spec.fit_intercept,
                decay,
                ridge: ridge
                    .as_ref()
                    .map(FloatOrList::to_vec)
                    .unwrap_or_else(|| vec![1e-6]),
                feature_sets: fs,
                standardize: *standardize,
                ridge_scale: *ridge_scale == crate::spec::RidgeScale::Sum,
                session_shrink: *session_shrink,
                long_half_life: long_half_life.as_ref().map(Span::value),
                coef_prior: coef_prior.clone(),
                // The spec's default is 0, the noise gate being this model's
                // (docs/WARMUP-AND-CONVERGENCE.md §2.1); the model's own
                // floor -- its first solve, and its own gate -- is then a
                // row per unknown, as the old default was. An explicit value
                // is the user's, above or below that.
                min_weight: match spec.min_weight {
                    Some(_) => spec.min_periods_or_default(),
                    None => (spec.k() + usize::from(spec.fit_intercept)) as f64,
                },
                solve_every: solve_every.as_ref().map_or_else(
                    || spec.solve_every_default(decay),
                    model_cfgs::solve_every_of,
                ),
                max_rows_between_solves: max_rows_between_solves.unwrap_or(u32::MAX),
                solve_share: spec.solve_share_default(solve_every.as_ref(), decay),
                gram_block_rows: gram_block_rows.unwrap_or(0),
                target_gaps: *target_gaps,
                window: window.as_ref().map(Span::value),
                window_every: snapshot_every.as_ref().map(Span::value),
                max_rows_between_snapshots: max_rows_between_snapshots.map(|r| r as usize),
            };
            let mut m = EwRidge::new(cfg)?;
            // The per-row leverage needs the factors kept (§2.1), and so do
            // the coefficients' standard errors (task 116, F).
            // So does the breaks' studentized residual (task 221 (b)).
            m.set_keep_factor(spec.emit_se_coef || spec.reads_row_inflation());
            // Each target's own threshold, for its own first solve (task
            // 195, S9b); left out, every target's is the model's.
            // And `min_weight` withholds the output alone: the model's own
            // noise and its first solve read its own floor (task 278).
            if spec.min_weight.is_some() {
                m.set_target_min_weight(spec.min_periods_per_target())?;
                m.set_min_weight_output_only(true);
            }
            Ok(AnyModel::EwRidge(Box::new(m)))
        }
        ModelKind::Rls { delta, coef_prior } => {
            let cfg = RlsCfg {
                n_features: spec.k(),
                n_targets: spec.m(),
                fit_intercept: spec.fit_intercept,
                decay,
                delta: delta.unwrap_or(1.0),
                coef_prior: coef_prior.clone(),
                min_weight: spec.min_periods_or_default(),
            };
            Ok(AnyModel::Rls(Box::new(Rls::new(cfg)?)))
        }
        ModelKind::Lasso {
            penalties,
            l1_ratio,
            select_half_life,
            solve_every,
            max_rows_between_solves,
            max_iter,
            tol,
            target_gaps,
            window_size: window,
            snapshot_every,
            max_rows_between_snapshots,
            window_budget: _,
            closed: _,
        } => {
            let cfg = LassoCfg {
                n_features: spec.k(),
                n_targets: spec.m(),
                fit_intercept: spec.fit_intercept,
                decay,
                lasso_path: penalties.clone(),
                l1_ratio: l1_ratio.unwrap_or(1.0),
                select_half_life: select_half_life.as_ref().map(Span::value),
                min_weight: spec.min_periods_or_default(),
                // Each target's own, which a state written before schema 64
                // counted its selection's errors from (CA3); this layer
                // gates the output on it either way.
                target_min_weight: spec.min_periods_per_target(),
                // `min_weight` withholds the output alone: the selection
                // scores every row from the model's floor, a row per
                // coefficient (task 278, D10); a state saved before keeps
                // its own.
                min_weight_holds_learning: false,
                solve_every: solve_every.as_ref().map_or_else(
                    || spec.solve_every_default(decay),
                    model_cfgs::solve_every_of,
                ),
                max_rows_between_solves: max_rows_between_solves.unwrap_or(u32::MAX),
                solve_share: spec.solve_share_default(solve_every.as_ref(), decay),
                window: window.as_ref().map(Span::value),
                window_every: snapshot_every.as_ref().map(Span::value),
                max_rows_between_snapshots: max_rows_between_snapshots.map(|r| r as usize),
                max_iter: max_iter.unwrap_or(100),
                tol: tol.unwrap_or(1e-10),
                target_gaps: *target_gaps,
            };
            Ok(AnyModel::Lasso(Box::new(Lasso::new(cfg)?)))
        }
        ModelKind::Kalman {
            coef_half_life,
            q,
            obs_var,
            p0,
            share_p,
            revert_half_life,
            standardize,
            covariance_form,
            horizons,
            ..
        } => {
            let cfg = KalmanCfg {
                horizons: horizons
                    .iter()
                    .flatten()
                    .map(|h| h.span().value())
                    .collect(),
                n_features: spec.k(),
                n_targets: spec.m(),
                fit_intercept: spec.fit_intercept,
                decay,
                // None beside a `q`, which the model reads instead (review
                // 2026-10-06, PC6).
                half_life: coef_half_life
                    .as_ref()
                    .map_or_else(Vec::new, SpanList::to_vec),
                q: q.as_ref().map(|v| v.iter().map(|n| n.0).collect()),
                obs_var: *obs_var,
                p0: p0.unwrap_or(1.0),
                share_p: *share_p,
                min_weight: spec.min_periods_or_default(),
                // `min_weight` withholds the output alone: the residual
                // variance learns from every row the filter predicts (task
                // 278, D10); a state saved before keeps its own.
                min_weight_holds_learning: false,
                revert_half_life: revert_half_life
                    .as_ref()
                    .map_or_else(|| vec![f64::INFINITY], SpanList::to_vec),
                standardize: *standardize,
                // `P` in units of the noise, the intercept's prior diffuse
                // (docs/PLAN.md task 278); a state saved before keeps its own.
                noise_units: true,
            };
            let mut model = Kalman::new(cfg)?.with_covariance_form(*covariance_form)?;
            if let Some(lag) = spec.smooth_lag() {
                model = model.with_smoother(lag, spec.emit_smoothed_se())?;
            }
            Ok(AnyModel::Kalman(Box::new(model)))
        }
        ModelKind::Huber { .. } => huber_model(spec, decay),
        ModelKind::Quantile { .. } => quantile_model(spec, decay),
        ModelKind::Ftrl {
            alpha,
            beta,
            l1,
            l2,
            strict_binary,
            loss,
            standardize,
        } => {
            let loss = match loss.as_deref() {
                None | Some("logistic") => FtrlLoss::Logistic,
                Some("squared") => FtrlLoss::Squared,
                Some(other) => {
                    return Err(crate::words_refusal(
                        "loss",
                        &["logistic", "squared"],
                        other,
                    ));
                }
            };
            let cfg = FtrlCfg {
                n_features: spec.k(),
                n_targets: spec.m(),
                fit_intercept: spec.fit_intercept,
                decay,
                alpha: alpha.unwrap_or(0.1),
                beta: beta.unwrap_or(1.0),
                l1: l1.unwrap_or(0.0),
                l2: l2.unwrap_or(1.0),
                min_weight: spec.min_periods_or_default(),
                strict_binary: *strict_binary,
                loss,
                standardize: *standardize,
            };
            Ok(AnyModel::Ftrl(Box::new(Ftrl::new(cfg)?)))
        }
        ModelKind::EwCov {
            stats,
            bias,
            precision_prior,
            mahal_quantiles,
            pca,
            pca_every,
            max_rows_between_pcas,
            lags,
            gram_threads: _,
            window_size: window,
            snapshot_every,
            max_rows_between_snapshots,
            window_budget: _,
            closed: _,
        } => {
            let names = stats
                .clone()
                .unwrap_or_else(|| vec!["mean".into(), "std".into(), "corr".into()]);
            let stats = names
                .iter()
                .map(|s| match s.as_str() {
                    "mean" => Ok(EwCovStat::Mean),
                    "var" => Ok(EwCovStat::Var),
                    "std" => Ok(EwCovStat::Std),
                    "cov" => Ok(EwCovStat::Cov),
                    "corr" => Ok(EwCovStat::Corr),
                    "partial_corr" => Ok(EwCovStat::PartialCorr),
                    "mahal" => Ok(EwCovStat::Mahal),
                    "lag_corr" => Ok(EwCovStat::LagCorr),
                    other => Err(format!(
                        "stats lists {other:?}, which is not an ew_cov statistic. Use \
                         \"mean\", \"var\", \"std\", \"cov\", \"corr\", \"partial_corr\", \
                         \"mahal\" or \"lag_corr\"."
                    )),
                })
                .collect::<Result<Vec<_>, String>>()?;
            let cfg = EwCovCfg {
                n_features: spec.k(),
                decay,
                stats,
                // Floored at 2 (`gate_floors`): a covariance below two rows
                // has nothing to report (review 2026-09-18, minor).
                min_weight: gate_floors(spec)[0],
                bias: *bias,
                precision_prior: *precision_prior,
                mahal_quantiles: mahal_quantiles.clone().unwrap_or_default(),
                pca: pca.unwrap_or(0),
                // The regressions' schedule (task 161): the clock, the rows,
                // or every row when neither is given.
                pca_every: match (pca_every, max_rows_between_pcas) {
                    (Some(e), _) => e.value(),
                    (None, Some(_)) => f64::INFINITY,
                    (None, None) => 0.0,
                },
                max_rows_between_pca: max_rows_between_pcas.unwrap_or(u32::MAX),
                lags: lags.clone().unwrap_or_default(),
                window: window.as_ref().map(Span::value),
                window_every: snapshot_every.as_ref().map(Span::value),
                max_rows_between_snapshots: max_rows_between_snapshots.map(|r| r as usize),
            };
            Ok(AnyModel::EwCov(Box::new(EwCovModel::new(cfg)?)))
        }
        ModelKind::Sgd { .. } => sgd_model(spec, decay),
        ModelKind::Pa {
            mode,
            c,
            eps,
            coef_min,
            coef_max,
            coef_sum,
            standardize,
        } => {
            let mode = match mode.as_deref().unwrap_or("pa1") {
                "pa" => PaMode::Pa,
                "pa1" => PaMode::Pa1,
                "pa2" => PaMode::Pa2,
                other => {
                    return Err(format!(
                        "mode must be \"pa\", \"pa1\" or \"pa2\", but it is {other:?}. Set mode \
                         to one of those words."
                    ));
                }
            };
            let cfg = PaCfg {
                n_features: spec.k(),
                n_targets: spec.m(),
                fit_intercept: spec.fit_intercept,
                decay,
                mode,
                // PA-I's cap, in the target's units over `‖z‖²`'s.
                c: c.map_or(1.0, |n| n.0),
                // As `sgd`'s tube: 1% of the target's own spread (task 203),
                // kept by task 207's sweep.
                eps: eps.unwrap_or(0.01),
                min_weight: spec.min_periods_or_default(),
                constraint: constraint(spec.k(), coef_min, coef_max, *coef_sum),
                standardize: *standardize,
            };
            Ok(AnyModel::Pa(Box::new(Pa::new(cfg)?)))
        }
        ModelKind::Holt {
            trend_half_life,
            trend,
            horizons,
            seasonal,
            seasonal_periods,
            period,
            seasonal_half_life,
            damping_half_life,
            start,
            initial_level,
            initial_trend,
            initial_seasonal,
            max_state_bytes,
        } => {
            // The level's half-life is the spec's own, so `half_life` means
            // the same thing here as it does for every other model; it had a
            // second name, `level_half_life`, until task 196.
            let level = match decay {
                Decay::Halflife(h) => h,
                // `lam = 1` forgets nothing, as for every other model; its
                // log is 0, and the division made it `-inf` (review
                // 2026-09-12, S30).
                Decay::Lam(1.0) => f64::INFINITY,
                Decay::Lam(l) => -std::f64::consts::LN_2 / l.ln(),
            };
            let cfg = HoltCfg {
                horizons: horizons
                    .iter()
                    .flatten()
                    .map(|h| h.span().value())
                    .collect(),
                n_targets: spec.m(),
                level_half_life: level,
                trend_half_life: trend_half_life.as_ref().map_or(level * 4.0, Span::value),
                min_weight: spec.min_periods_or_default(),
                trend: trend.unwrap_or(true),
                season: holt_season(seasonal.as_deref())?,
                season_periods: seasonal_periods.unwrap_or(0),
                period: period.as_ref().map_or(0.0, Span::value),
                seasonal_half_life: seasonal_half_life
                    .as_ref()
                    .map_or(f64::INFINITY, Span::value),
                damping_half_life: damping_half_life
                    .as_ref()
                    .map_or(f64::INFINITY, Span::value),
                start: holt_start(start.as_deref())?,
                initial_level: initial_level
                    .as_ref()
                    .map_or_else(Vec::new, FloatOrList::to_vec),
                initial_trend: initial_trend
                    .as_ref()
                    .map_or_else(Vec::new, FloatOrList::to_vec),
                initial_seasonal: holt_seasonal(initial_seasonal.as_deref()),
                max_state_bytes: *max_state_bytes,
            };
            Ok(AnyModel::Holt(Box::new(Holt::new(cfg)?)))
        }
        ModelKind::KMeans {
            k,
            warm_rows,
            init,
            seed,
            update_every_rows,
            split_merge,
            split_merge_every_rows,
            dead_frac,
            standardize,
            scale_floor,
            warm_span,
            update_every,
            split_merge_every,
        } => {
            let seed_rule = crate::stream::init_rule(init.as_deref())?;
            let cfg = KMeansCfg {
                n_features: spec.k(),
                k: *k,
                decay,
                // One `min_weight` for the whole model: the spec's first
                // (and only) entry, defaulting like the regressions do.
                min_weight: spec.min_periods_per_target()[0],
                // At least `k`: a smaller one given is refused, and left out
                // it is the buffer `max(warm_rows, k)` always was (review
                // 2026-10-06, PC8).
                // Under a span the buffer holds the span's rows, at least
                // `k`; a count beside it is the spec's to refuse.
                warm_rows: warm_rows.unwrap_or_else(|| {
                    if warm_span.is_some() {
                        *k
                    } else {
                        (*k).max(500)
                    }
                }),
                seed_rule,
                seed: seed.unwrap_or(0),
                // A clock cadence alone has no row cap.
                update_every_rows: update_every_rows.unwrap_or(if update_every.is_some() {
                    u32::MAX
                } else {
                    1
                }),
                split_merge: split_merge.unwrap_or(0.5),
                split_merge_every_rows: split_merge_every_rows.unwrap_or(
                    if split_merge_every.is_some() {
                        u32::MAX
                    } else {
                        100
                    },
                ),
                // The dead rule runs at a split-merge check, so with none it
                // is 0, where a value above 0 is refused (CF6).
                dead_frac: dead_frac.unwrap_or(if *split_merge == Some(0.0) { 0.0 } else { 0.05 }),
                standardize: standardize.unwrap_or(true),
                scale_floor: scale_floor.unwrap_or(0.1),
                warm_clock: warm_span.as_ref().map(Span::value),
                update_every: update_every.as_ref().map(Span::value),
                split_merge_every: split_merge_every.as_ref().map(Span::value),
            };
            Ok(AnyModel::KMeans(Box::new(KMeans::new(cfg)?)))
        }
        ModelKind::Micro {
            eps,
            beta_mu,
            max_clusters,
            prune_every,
            max_rows_between_prunes,
            macro_link,
            standardize,
            scale_floor,
        } => {
            let cfg = MicroCfg {
                n_features: spec.k(),
                decay,
                min_weight: spec.min_periods_per_target()[0],
                eps: *eps,
                beta_mu: beta_mu.unwrap_or(3.0),
                max_clusters: max_clusters.unwrap_or(200),
                // The regressions' schedule (task 163): the clock, the
                // learned rows, or, with neither, every 100 learned rows.
                prune_every: prune_every.as_ref().map_or(f64::INFINITY, Span::value),
                max_rows_between_prunes: match (prune_every, max_rows_between_prunes) {
                    (_, Some(r)) => *r,
                    (None, None) => 100,
                    (Some(_), None) => u32::MAX,
                },
                macro_link: *macro_link,
                standardize: standardize.unwrap_or(true),
                scale_floor: scale_floor.unwrap_or(0.1),
            };
            Ok(AnyModel::Micro(Box::new(Micro::new(cfg)?)))
        }
        ModelKind::EwClass {
            classes,
            covariance_type,
            precision_prior,
            window_size: window,
            snapshot_every,
            max_rows_between_snapshots,
            window_budget: _,
            closed: _,
        } => {
            let cfg = EwClassCfg {
                n_features: spec.k(),
                n_classes: classes.len(),
                decay,
                min_weight: spec.min_periods_per_target()[0],
                covariance: match covariance_type {
                    Some(c) => Covariance::parse(c)?,
                    None => Covariance::Full,
                },
                precision_prior: *precision_prior,
                window: window.as_ref().map(Span::value),
                window_every: snapshot_every.as_ref().map(Span::value),
                max_rows_between_snapshots: max_rows_between_snapshots.map(|r| r as usize),
            };
            Ok(AnyModel::EwClass(Box::new(EwClass::new(cfg)?)))
        }
        // No decay to build with: `decays()` gives one undecayed instance.
        ModelKind::SeqTest { .. } => {
            let cfg = SeqTestCfg {
                n_targets: spec.m(),
                min_weight: spec.min_periods_or_default(),
            };
            Ok(AnyModel::SeqTest(Box::new(SeqTest::new(cfg)?)))
        }
        ModelKind::Marginal {
            window_size: window,
            snapshot_every,
            max_rows_between_snapshots,
            window_budget: _,
            closed: _,
            // A spec-level acceptance of the price, checked in `validate`.
            window_lags: _,
            feature_moments,
            lags,
            serial_rule,
            cross_lags,
            bins,
            bin_rule,
            bin_warm_rows,
            bin_edges,
            bin_budget,
            bin_stat,
            bin_huber_delta,
            bin_refit_rows,
            // The bank's, against every group's reservation (`reserve_refits`).
            refit_budget: _,
            // How the pairs are run, not what they are (`marginal_shards`).
            shards: _,
        } => {
            // The per-target thresholds go to the model whole: it is the
            // reader of its own state, so it gates each target's pairs
            // itself where a regression's stream layer would.
            let cfg = MarginalCfg {
                n_features: spec.k(),
                n_targets: spec.m(),
                decay,
                min_weight: spec.min_periods_per_target(),
                lags: lags.clone().unwrap_or_default(),
                cross_lags: cross_lags.clone(),
                serial_rule: match serial_rule.as_deref() {
                    None => None,
                    Some("truncated") => Some(rheon_core::SerialRule::Truncated),
                    Some("geometric") => Some(rheon_core::SerialRule::Geometric),
                    Some("bartlett") => Some(rheon_core::SerialRule::Bartlett),
                    Some(other) => {
                        return Err(crate::words_refusal(
                            "serial_rule",
                            &["truncated", "bartlett", "geometric"],
                            other,
                        ));
                    }
                },
                bins: match (bins, bin_edges) {
                    (None, None) => None,
                    (_, edges) => Some(Box::new(rheon_core::BinCfg {
                        n_bins: bins.unwrap_or(DEFAULT_BINS),
                        edges: edges.clone(),
                        rule: match bin_rule.as_deref() {
                            None | Some("quantile") => rheon_core::BinRule::Quantile,
                            Some("fixed") => rheon_core::BinRule::Fixed,
                            Some(other) => {
                                return Err(crate::words_refusal(
                                    "bin_rule",
                                    &["quantile", "fixed"],
                                    other,
                                ));
                            }
                        },
                        warm_rows: bin_warm_rows.unwrap_or(DEFAULT_BIN_WARM_ROWS),
                        stat: match bin_stat.as_deref() {
                            None | Some("mean") => rheon_core::BinStat::Mean,
                            Some("huber") => rheon_core::BinStat::Huber {
                                delta: bin_huber_delta.map_or(DEFAULT_BIN_HUBER_DELTA, |n| n.0),
                            },
                            Some(other) => {
                                return Err(crate::words_refusal(
                                    "bin_stat",
                                    &["mean", "huber"],
                                    other,
                                ));
                            }
                        },
                        // On by default under a finite Huber cut, off
                        // otherwise (`validate` refuses a count given there).
                        refit_rows: match (bin_stat.as_deref(), bin_huber_delta) {
                            // `max(500, 30 · bins)`, the widest feature's
                            // bins (docs/PLAN.md task 278): 500 at the
                            // default 16, which the learned edges' warm-up of
                            // 1,000 rows covers, so by default the refit runs
                            // when the edges are fixed and holds no row of
                            // its own.
                            (Some("huber"), d) if d.is_none_or(|d| d.0.is_finite()) => {
                                bin_refit_rows.unwrap_or_else(|| {
                                    rheon_core::default_bin_refit_rows(match edges {
                                        Some(e) => e.iter().map(|f| f.len() + 1).max().unwrap_or(1),
                                        None => bins.unwrap_or(DEFAULT_BINS),
                                    })
                                })
                            }
                            _ => 0,
                        },
                        // Under `bin_stat = "huber"`, the target's MAD
                        // sketch in units of its first deviation.
                        mad_anchor: true,
                        budget_mib: bin_budget.map(|n| n.0),
                    })),
                },
                feature_moments: match feature_moments.as_deref() {
                    None | Some("per_target") => rheon_core::FeatureMomentLayout::PerTarget,
                    Some("shared") => rheon_core::FeatureMomentLayout::Shared,
                    Some(other) => {
                        return Err(format!(
                            "feature_moments must be \"per_target\" or \"shared\", but it is \
                             {other:?}. Set feature_moments to one of those words."
                        ));
                    }
                },
                window: window.as_ref().map(Span::value),
                window_every: snapshot_every.as_ref().map(Span::value),
                max_rows_between_snapshots: max_rows_between_snapshots.map(|r| r as usize),
            };
            Ok(AnyModel::Marginal(Box::new(Marginal::new(cfg)?)))
        }
        ModelKind::Deco { .. } => {
            let mut cfg = deco_cfg(spec)?;
            cfg.decay = decay;
            Ok(AnyModel::Deco(Box::new(Deco::new(cfg)?)))
        }
        // No decay to build with, so the one undecayed instance `decays()`
        // gives is the only one.
        ModelKind::Rcov { .. } => Ok(AnyModel::Rcov(Box::new(Rcov::new(rcov_cfg(spec)?)?))),
        ModelKind::Hmm { .. } => {
            let mut cfg = hmm_cfg(spec)?;
            cfg.decay = decay;
            Ok(AnyModel::Hmm(Box::new(Hmm::new(cfg)?)))
        }
        ModelKind::CorrChange { .. } => {
            let mut cfg = corrchange_cfg(spec)?;
            cfg.decay = decay;
            Ok(AnyModel::CorrChange(Box::new(CorrChange::new(cfg)?)))
        }
        // No decay: the run-length posterior is what forgets.
        ModelKind::Bocpd { .. } => Ok(AnyModel::Bocpd(Box::new(Bocpd::new(bocpd_cfg(spec)?)?))),
        // No decay: an audit counts every row.
        ModelKind::Audit { .. } => Ok(AnyModel::Audit(Box::new(rheon_core::Audit::new(
            audit_cfg(spec)?,
        )?))),
        ModelKind::Glm { .. } => {
            let mut cfg = glm_cfg(spec)?;
            cfg.decay = decay;
            Ok(AnyModel::Glm(Box::new(rheon_core::Glm::new(cfg)?)))
        }
        ModelKind::Hawkes { .. } => Ok(AnyModel::Hawkes(Box::new(rheon_core::Hawkes::new(
            hawkes_cfg(spec, decay)?,
        )?))),
        ModelKind::Expectile { .. } => Ok(AnyModel::Expectile(Box::new(
            rheon_core::Expectile::new(expectile_cfg(spec, decay)?)?,
        ))),
        // No decay: the recursion is per observation.
        ModelKind::Garch { .. } => Ok(AnyModel::Garch(Box::new(rheon_core::Garch::new(
            garch_cfg(spec)?,
        )?))),
        ModelKind::Dcc { .. } => Ok(AnyModel::Dcc(Box::new(rheon_core::Dcc::new(dcc_cfg(
            spec, decay,
        )?)?))),
        ModelKind::Realized { .. } => Ok(AnyModel::Realized(Box::new(rheon_core::Realized::new(
            realized_cfg(spec, decay)?,
        )?))),

        ModelKind::Pca { .. } => Ok(AnyModel::Pca(Box::new(rheon_core::StreamPca::new(
            pca_cfg(spec, decay)?,
        )?))),
        ModelKind::PowerVariation { .. } => Ok(AnyModel::PowerVariation(Box::new(
            rheon_core::PowerVariation::new(power_variation_cfg(spec, decay)?)?,
        ))),
        // No decay: a merging digest cannot take a row out.
        ModelKind::TDigest { .. } => Ok(AnyModel::TDigest(Box::new(
            rheon_core::TDigestModel::new(tdigest_cfg(spec)?)?,
        ))),
        // A decay (rounded) or a window (exact), not both.
        ModelKind::Isotonic { .. } => Ok(AnyModel::Isotonic(Box::new(rheon_core::Isotonic::new(
            isotonic_cfg(spec, decay)?,
        )?))),
        ModelKind::DdSketch { .. } => Ok(AnyModel::DdSketch(Box::new(
            rheon_core::DdSketchModel::new(ddsketch_cfg(spec, decay)?)?,
        ))),
        ModelKind::EwmMoments { .. } => Ok(AnyModel::EwmMoments(Box::new(
            rheon_core::EwmMoments::new(crate::spec::ewm_moments_cfg(spec, decay)?)?,
        ))),
        ModelKind::Spectral(_) => Ok(AnyModel::Spectral(Box::new(rheon_core::Spectral::new(
            spectral_cfg(spec, decay)?,
        )?))),
        ModelKind::Ar { .. } => {
            let mut m = rheon_core::Ar::new(ar_cfg(spec, decay)?)?;
            // Each fit's factor, which its standard errors are read from.
            m.set_keep_factor(spec.emit_se_coef);
            Ok(AnyModel::Ar(Box::new(m)))
        }
        ModelKind::VectorAr(_) => {
            let mut m = rheon_core::VectorAr::new(crate::spec::vector_ar_cfg(spec, decay)?)?;
            // The coefficients' variances, which `se_coef` reports.
            m.set_keep_factor(spec.emit_se_coef);
            Ok(AnyModel::VectorAr(Box::new(m)))
        }
    }
}

/// `holt`'s `seasonal`: absent, `"add"` or `"mul"`.
fn holt_season(word: Option<&str>) -> Result<rheon_core::HoltSeason, String> {
    match word {
        None => Ok(rheon_core::HoltSeason::None),
        Some("add") => Ok(rheon_core::HoltSeason::Add),
        Some("mul") => Ok(rheon_core::HoltSeason::Mul),
        Some(other) => Err(format!(
            "holt: seasonal must be \"add\" or \"mul\", but it is {other:?}. Use one of those \
             words."
        )),
    }
}

/// `holt`'s `start`: absent (the warm start), `"warm"`, `"known"` or
/// `"heuristic"`.
fn holt_start(word: Option<&str>) -> Result<rheon_core::HoltStart, String> {
    match word {
        None | Some("warm") => Ok(rheon_core::HoltStart::Warm),
        Some("known") => Ok(rheon_core::HoltStart::Known),
        Some("heuristic") => Ok(rheon_core::HoltStart::Heuristic),
        Some(other) => Err(format!(
            "holt: start must be \"warm\", \"known\" or \"heuristic\", but it is {other:?}. \
             Use one of those words."
        )),
    }
}

/// `holt`'s `initial_seasonal`: one list of numbers, for a single target,
/// or a list per target.
fn holt_seasonal(given: Option<&[FloatOrList]>) -> Vec<Vec<f64>> {
    match given {
        None => Vec::new(),
        Some(v) if v.iter().all(|e| matches!(e, FloatOrList::Float(_))) => {
            vec![v.iter().flat_map(FloatOrList::to_vec).collect()]
        }
        Some(v) => v.iter().map(FloatOrList::to_vec).collect(),
    }
}
