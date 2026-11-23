//! The configurations of the models a spec does not build field by field:
//! `audit`, `bocpd`, `corrchange`, `hmm`, `rcov` and `deco`, each read from
//! its spec, and `deco`'s block names; and the models `huber`, `quantile`
//! and `sgd` build from their specs; and the floor a model is read at below
//! its `min_weight` (moved out of `stream.rs` under the 250 KB cap a source
//! file is held to).

#[allow(unused_imports)]
use super::*;

/// A given `solve_every` as a model's configuration takes it: its value in
/// clock units, and `inf` for any value that never comes due
/// ([`rheon_core::SOLVE_NEVER`]; task 278), so a spec written with `1e18`
/// builds the model one written with `inf` does, to the bit.
pub(crate) fn solve_every_of(span: &Span) -> f64 {
    let v = span.value();
    if rheon_core::solve_never(v) {
        f64::INFINITY
    } else {
        v
    }
}

/// The refusal of a spec of another kind, passed to one kind's builder: a
/// caller's mistake in Rust, which a spec built by the bank never makes.
fn not_a(spec: &Spec, call: &str, kind: &str) -> String {
    format!(
        "{call}: spec {:?} is a {} spec, not a {kind} spec. Pass {call} a {kind} spec.",
        spec.name,
        spec.model.kind_name()
    )
}

/// The refusal a check stops at, with those found before it: the checks
/// after it read the word it refuses.
fn stop_at(mut no: rheon_core::Refusals, problem: String) -> String {
    no.refuse(problem);
    no.stop_here().err().unwrap_or_default()
}

/// An `audit` spec's [`rheon_core::AuditCfg`]: its columns are the spec's
/// features, and the clock's `gap_cap` is the stream's.
pub fn audit_cfg(spec: &Spec) -> Result<rheon_core::AuditCfg, String> {
    let ModelKind::Audit {
        pairs,
        distinct_cap,
    } = &spec.model
    else {
        return Err(not_a(spec, "audit_cfg", "audit"));
    };
    let cfg = rheon_core::AuditCfg {
        n_columns: spec.features.len(),
        pairs: pairs.unwrap_or(false),
        distinct_cap: distinct_cap.unwrap_or(rheon_core::DISTINCT_CAP),
        gap_cap: spec.gap_cap.as_ref().map(Span::value),
        has_clock: spec.clock.is_some(),
    };
    cfg.validate()?;
    Ok(cfg)
}

/// A `hawkes` spec's [`rheon_core::HawkesCfg`]: its nodes are the spec's
/// targets, and every parameter check is the model's.
pub fn hawkes_cfg(spec: &Spec, decay: rheon_core::Decay) -> Result<rheon_core::HawkesCfg, String> {
    let ModelKind::Hawkes {
        window_events,
        window_size,
        closed,
        max_state_bytes,
        ..
    } = &spec.model
    else {
        return Err(not_a(spec, "hawkes_cfg", "hawkes"));
    };
    let cfg = rheon_core::HawkesCfg {
        n_targets: spec.m(),
        kernel_decays: spec
            .model
            .hawkes_kernels()
            .into_iter()
            .map(|(b, _)| b)
            .collect(),
        decay,
        min_weight: spec.min_periods_or_default(),
        window_events: *window_events,
        window_clock: window_size.as_ref().map(Span::value),
        closed: closed.edge(),
        max_state_bytes: *max_state_bytes,
    };
    cfg.validate()?;
    Ok(cfg)
}

/// An `expectile` spec's [`rheon_core::ExpectileCfg`] at `decay`: the
/// refit's default from the features, the targets and the state's cap
/// ([`rheon_core::default_expectile_refit_rows`]); every parameter check is
/// the model's, the state's bytes against `max_state_bytes` among them.
pub fn expectile_cfg(
    spec: &Spec,
    decay: rheon_core::Decay,
) -> Result<rheon_core::ExpectileCfg, String> {
    let ModelKind::Expectile {
        expectile,
        ridge,
        refit_rows,
        refit_span,
        window_rows,
        window_size,
        closed,
        max_state_bytes,
        ..
    } = &spec.model
    else {
        return Err(not_a(spec, "expectile_cfg", "expectile"));
    };
    let windowed = window_rows.is_some() || window_size.is_some();
    let cap = max_state_bytes.unwrap_or(rheon_core::EXPECTILE_MAX_STATE_BYTES);
    Ok(rheon_core::ExpectileCfg {
        n_features: spec.k(),
        n_targets: spec.m(),
        fit_intercept: spec.fit_intercept,
        decay,
        expectile: *expectile,
        ridge: ridge.unwrap_or(1e-6),
        min_weight: spec.min_periods_or_default(),
        // A span in place of the count: the count's default is off beside
        // it, and one given is the model's to refuse.
        refit_rows: refit_rows.unwrap_or_else(|| {
            if refit_span.is_some() {
                return 0;
            }
            rheon_core::default_expectile_refit_rows(
                spec.k(),
                spec.m(),
                spec.fit_intercept,
                windowed,
                cap,
            )
        }),
        refit_clock: refit_span.as_ref().map(Span::value),
        window_rows: *window_rows,
        window_clock: window_size.as_ref().map(Span::value),
        closed: closed.edge(),
        max_state_bytes: *max_state_bytes,
    })
}

/// A `realized` spec's [`rheon_core::RealizedCfg`] at `decay`: one series a
/// target. Every value is the model's own to check, the state's bytes
/// against `max_state_bytes` among them.
pub fn realized_cfg(
    spec: &Spec,
    decay: rheon_core::Decay,
) -> Result<rheon_core::RealizedCfg, String> {
    let ModelKind::Realized {
        window_rows,
        window_size,
        closed,
        lm_window,
        lm_beta,
        max_state_bytes,
    } = &spec.model
    else {
        return Err(not_a(spec, "realized_cfg", "realized"));
    };
    Ok(rheon_core::RealizedCfg {
        n_targets: spec.m(),
        decay,
        min_weight: spec.min_periods_or_default(),
        window_rows: *window_rows,
        window_clock: window_size.as_ref().map(Span::value),
        closed: closed.edge(),
        lm_window: *lm_window,
        lm_beta: lm_beta.unwrap_or(rheon_core::REALIZED_LM_BETA),
        max_state_bytes: *max_state_bytes,
    })
}

/// A `power_variation` spec's [`rheon_core::PowerVariationCfg`] at
/// `decay`: one series a target, each named term as the core's. A term's
/// shape (powers or a kernel, the options each takes) and its name are
/// checked here; every other value is the model's own to check.
pub fn power_variation_cfg(
    spec: &Spec,
    decay: rheon_core::Decay,
) -> Result<rheon_core::PowerVariationCfg, String> {
    use rheon_core::{PvTerm, PvTermKind};
    let ModelKind::PowerVariation {
        terms,
        window_rows,
        window_size,
        closed,
        max_state_bytes,
    } = &spec.model
    else {
        return Err(not_a(spec, "power_variation_cfg", "power_variation"));
    };
    let mut no = rheon_core::Refusals::new();
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::with_capacity(terms.len());
    for t in terms {
        let name = &t.name;
        if name.is_empty() {
            no.refuse(
                "power_variation: a term's name is empty. Give each term a name: its fields are \
                 <name>_<target>.",
            );
        } else if name == "n" || name == "weight_sum" {
            no.refuse(format!(
                "power_variation: a term is named {name:?}, which names a field every \
                 power_variation reports. Give the term another name."
            ));
        } else if !seen.insert(name.as_str()) {
            no.refuse(format!(
                "power_variation: two terms are named {name:?}, and a name names a term's \
                 fields. Give each term its own name."
            ));
        }
        let scale = t.scale.unwrap_or(1);
        match (&t.powers, t.kernel) {
            (Some(powers), None) => {
                if t.bandwidth.is_some() || t.flat_top || t.center {
                    no.refuse(format!(
                        "power_variation: term {name:?} has powers, and bandwidth, flat_top \
                         and center belong to a kernel term. Leave them out, or give the term \
                         a kernel in place of its powers."
                    ));
                }
                out.push(PvTerm {
                    kind: PvTermKind::Multipower {
                        powers: powers.clone(),
                        normalize: t.normalize,
                    },
                    scale,
                });
            }
            (None, Some(kernel)) => {
                if t.normalize {
                    no.refuse(format!(
                        "power_variation: term {name:?} is a kernel term, and normalize \
                         belongs to a multipower term. Leave normalize out."
                    ));
                }
                match t.bandwidth {
                    Some(bandwidth) => out.push(PvTerm {
                        kind: PvTermKind::Kernel {
                            kernel,
                            bandwidth,
                            flat_top: t.flat_top,
                            center: t.center,
                        },
                        scale,
                    }),
                    None => no.refuse(format!(
                        "power_variation: term {name:?} has a kernel and no bandwidth. Give \
                         its bandwidth H, the last lag with weight."
                    )),
                }
            }
            (Some(_), Some(_)) => no.refuse(format!(
                "power_variation: term {name:?} has both powers and a kernel, and a term is \
                 one of the two. Split it into two terms."
            )),
            (None, None) => no.refuse(format!(
                "power_variation: term {name:?} has neither powers nor a kernel. Give its \
                 powers, such as [1, 1], or a kernel and a bandwidth."
            )),
        }
    }
    no.finish()?;
    Ok(rheon_core::PowerVariationCfg {
        n_targets: spec.m(),
        terms: out,
        decay,
        min_weight: spec.min_periods_or_default(),
        window_rows: *window_rows,
        window_clock: window_size.as_ref().map(Span::value),
        closed: closed.edge(),
        max_state_bytes: *max_state_bytes,
    })
}

/// A `garch` spec's [`rheon_core::GarchCfg`]: one series a target, the
/// start 2,000 rows unless the spec gives rows or a span. Every value is
/// the model's own to check.
pub fn garch_cfg(spec: &Spec) -> Result<rheon_core::GarchCfg, String> {
    use rheon_core::{GarchDist, GarchVol};
    let ModelKind::Garch {
        vol,
        dist,
        truncation,
        variance_targeting,
        start_rows,
        start_span,
        horizon,
        max_state_bytes,
    } = &spec.model
    else {
        return Err(not_a(spec, "garch_cfg", "garch"));
    };
    let vol = match vol.as_deref().unwrap_or("garch") {
        "garch" => GarchVol::Garch,
        "gjr" => GarchVol::Gjr,
        "egarch" => GarchVol::Egarch,
        "figarch" => GarchVol::Figarch,
        other => {
            return Err(format!(
                "garch: vol must be \"garch\", \"gjr\", \"egarch\" or \"figarch\", but it \
                 is {other:?}. Use one of those words."
            ));
        }
    };
    let dist = match dist.as_deref().unwrap_or("normal") {
        "normal" => GarchDist::Normal,
        "t" => GarchDist::T,
        "skewt" => GarchDist::Skewt,
        other => {
            return Err(format!(
                "garch: dist must be \"normal\", \"t\" or \"skewt\", but it is {other:?}. Use \
                 one of those words."
            ));
        }
    };
    let cfg = rheon_core::GarchCfg {
        n_targets: spec.m(),
        vol,
        dist,
        truncation: *truncation,
        variance_targeting: variance_targeting.unwrap_or(false),
        start_rows: start_rows.unwrap_or(GARCH_START_ROWS),
        start_span: start_span.as_ref().map(Span::value),
        horizon: horizon.unwrap_or(1),
        min_weight: spec.min_periods_or_default(),
        max_state_bytes: *max_state_bytes,
    };
    cfg.validate()?;
    Ok(cfg)
}

/// A `tdigest` spec's [`rheon_core::TDigestModelCfg`]: one digest a target,
/// at compression 100 unless the spec gives one. Every value is the
/// model's own to check.
pub fn tdigest_cfg(spec: &Spec) -> Result<rheon_core::TDigestModelCfg, String> {
    let ModelKind::TDigest {
        compression,
        quantiles,
        max_state_bytes,
    } = &spec.model
    else {
        return Err(not_a(spec, "tdigest_cfg", "tdigest"));
    };
    let cfg = rheon_core::TDigestModelCfg {
        n_targets: spec.m(),
        compression: compression.unwrap_or(rheon_core::TDIGEST_COMPRESSION),
        levels: quantiles.clone(),
        max_state_bytes: *max_state_bytes,
    };
    cfg.validate()?;
    Ok(cfg)
}

/// A `ddsketch` spec's [`rheon_core::DdSketchModelCfg`] under `decay`: one
/// sketch a target, at `alpha = tanh(1/128)` unless the spec gives one, its
/// window in rows or in clock units. Every value is the model's own to check.
pub fn ddsketch_cfg(
    spec: &Spec,
    decay: rheon_core::Decay,
) -> Result<rheon_core::DdSketchModelCfg, String> {
    let ModelKind::DdSketch {
        alpha,
        quantiles,
        window_rows,
        window_size,
        closed,
        max_state_bytes,
    } = &spec.model
    else {
        return Err(not_a(spec, "ddsketch_cfg", "ddsketch"));
    };
    let cfg = rheon_core::DdSketchModelCfg {
        n_targets: spec.m(),
        alpha: alpha.unwrap_or(rheon_core::EW_QUANTILE_ALPHA),
        levels: quantiles.clone(),
        window_rows: *window_rows,
        window_clock: window_size.as_ref().map(Span::value),
        closed: closed.edge(),
        decay,
        max_state_bytes: *max_state_bytes,
    };
    cfg.validate()?;
    Ok(cfg)
}

/// A `garch` spec's start, in rows, when it gives neither rows nor a span.
pub const GARCH_START_ROWS: usize = 2000;

/// A `dcc` spec's [`rheon_core::DccCfg`] under `decay`: the targets the
/// series, a GARCH margin each unless `margins` is `"none"` (its start
/// 2,000 rows unless the spec gives rows or a span), the correlation's
/// start 500 rows unless the spec gives rows or a span, and `(a, b[, g])`
/// held where the spec gives them. Every value is the model's own to check.
pub fn dcc_cfg(spec: &Spec, decay: Decay) -> Result<rheon_core::DccCfg, String> {
    use rheon_core::{GarchDist, GarchVol};
    let ModelKind::Dcc {
        asymmetric,
        margins,
        vol,
        dist,
        truncation,
        variance_targeting,
        margin_start_rows,
        margin_start_span,
        a,
        b,
        g,
        start_rows,
        start_span,
        max_state_bytes,
    } = &spec.model
    else {
        return Err(not_a(spec, "dcc_cfg", "dcc"));
    };
    let asymmetric = asymmetric.unwrap_or(false);
    let with_margins = match margins.as_deref().unwrap_or("garch") {
        "garch" => true,
        "none" => false,
        other => {
            return Err(format!(
                "dcc: margins must be \"garch\" or \"none\", but it is {other:?}. Use one of \
                 those words."
            ));
        }
    };
    let margin_keys_given = vol.is_some()
        || dist.is_some()
        || truncation.is_some()
        || variance_targeting.is_some()
        || margin_start_rows.is_some()
        || margin_start_span.is_some();
    if !with_margins && margin_keys_given {
        return Err(
            "dcc: margins = \"none\" runs no GARCH margin, but the spec sets one of vol, dist, \
             truncation, variance_targeting, margin_start_rows or margin_start_span. Leave them \
             out, or set margins to \"garch\"."
                .into(),
        );
    }
    let vol = match vol.as_deref().unwrap_or("garch") {
        "garch" => GarchVol::Garch,
        "gjr" => GarchVol::Gjr,
        "egarch" => GarchVol::Egarch,
        "figarch" => GarchVol::Figarch,
        other => {
            return Err(format!(
                "dcc: vol must be \"garch\", \"gjr\", \"egarch\" or \"figarch\", but it is \
                 {other:?}. Use one of those words."
            ));
        }
    };
    let dist = match dist.as_deref().unwrap_or("normal") {
        "normal" => GarchDist::Normal,
        "t" => GarchDist::T,
        "skewt" => GarchDist::Skewt,
        other => {
            return Err(format!(
                "dcc: dist must be \"normal\", \"t\" or \"skewt\", but it is {other:?}. Use \
                 one of those words."
            ));
        }
    };
    let params = match (a, b, g) {
        (None, None, None) => None,
        (Some(a), Some(b), None) if !asymmetric => Some(vec![*a, *b]),
        (Some(a), Some(b), Some(g)) if asymmetric => Some(vec![*a, *b, *g]),
        _ => {
            return Err(format!(
                "dcc: the spec gives {} of the parameters, but they are held only all together. \
                 Give {}, or leave every one out to estimate them.",
                [("a", a), ("b", b), ("g", g)]
                    .iter()
                    .filter(|(_, v)| v.is_some())
                    .map(|(n, _)| *n)
                    .collect::<Vec<_>>()
                    .join(", ")
                    + if !asymmetric && g.is_some() {
                        " (g is the asymmetric model's: set asymmetric to true)"
                    } else {
                        ""
                    },
                if asymmetric { "a, b and g" } else { "a and b" },
            ));
        }
    };
    let cfg = rheon_core::DccCfg {
        n_targets: spec.m(),
        asymmetric,
        margins: with_margins.then(|| rheon_core::GarchCfg {
            n_targets: spec.m(),
            vol,
            dist,
            truncation: *truncation,
            variance_targeting: variance_targeting.unwrap_or(false),
            start_rows: margin_start_rows.unwrap_or(GARCH_START_ROWS),
            start_span: margin_start_span.as_ref().map(Span::value),
            horizon: 1,
            min_weight: 0.0,
            max_state_bytes: None,
        }),
        params,
        decay,
        start_rows: start_rows.unwrap_or(rheon_core::DCC_START_ROWS),
        start_span: start_span.as_ref().map(Span::value),
        min_weight: spec.min_periods_or_default(),
        max_state_bytes: *max_state_bytes,
    };
    // A margin's refusal names the margins' parameters as the spec spells
    // them.
    cfg.validate()
        .map_err(|e| match e.strip_prefix("dcc: garch: ") {
            Some(rest) => format!(
                "dcc: {}",
                rest.replace("start_rows", "margin_start_rows")
                    .replace("start_span", "margin_start_span")
            ),
            None => e,
        })?;
    Ok(cfg)
}

/// A `bocpd` spec's expected rows between changepoints, where it gives
/// neither `hazard_rows` nor `hazard`.
pub const BOCPD_HAZARD_ROWS: f64 = 250.0;

/// A `bocpd` spec's [`BocpdCfg`]; every parameter check is the model's.
pub fn bocpd_cfg(spec: &Spec) -> Result<BocpdCfg, String> {
    let ModelKind::Bocpd {
        hazard,
        hazard_col,
        covariance_type,
        mean_prior,
        mean_precision_prior,
        degrees_of_freedom_prior,
        covariance_prior,
        robust_beta,
        prune_below,
        max_run,
        warm_rows,
        warm_span,
        hazard_rows,
    } = &spec.model
    else {
        return Err(not_a(spec, "bocpd_cfg", "bocpd"));
    };
    let emission = match (covariance_type.as_deref(), robust_beta) {
        (None | Some("diag"), None) => BocpdEmission::Diag,
        (None | Some("diag"), Some(b)) => {
            // Measured (`bocpd.rs`, `a_sustained_shift_survives_a_small_beta_
            // _and_not_a_large_one`): 0.1 ignores a 20-sigma row outright and
            // still finds a real shift within five rows; above ~0.2 nothing
            // is ever detected. Left out, the predictive is not tempered.
            if !(b.is_finite() && *b > 0.0) {
                return Err(format!(
                    "bocpd: robust_beta must be finite and above 0, but it is {b}. Set \
                     robust_beta to a small positive value such as 0.1, which ignores a row 20 \
                     standard deviations out and still finds a shift of 4 standard deviations, or leave it out to \
                     weigh every row in full."
                ));
            }
            BocpdEmission::Robust
        }
        (Some("full"), None) => BocpdEmission::Gaussian,
        (Some("full"), Some(_)) => {
            return Err(
                "bocpd: robust_beta applies only under covariance_type=\"diag\", but \
                 covariance_type is \"full\", which fits one distribution over all the \
                 features. Set covariance_type=\"diag\", or leave robust_beta out."
                    .into(),
            );
        }
        (Some(other), _) => {
            return Err(format!(
                "bocpd: {}",
                crate::words_refusal("covariance_type", &["full", "diag"], other)
            ));
        }
    };
    Ok(BocpdCfg {
        n_features: spec.k(),
        // `hazard` is the expected clock time between changepoints, in clock
        // units: rows without a clock, the column's units on a number clock,
        // seconds on a temporal one (tasks 179 and 547). `hazard_rows`, or
        // neither, the expected rows between them; `Spec::validate` refuses
        // both.
        hazard: hazard
            .as_ref()
            .map(Span::value)
            .or(*hazard_rows)
            .unwrap_or(BOCPD_HAZARD_ROWS),
        hazard_from_row: hazard_col.is_some(),
        // The covariance's words and the tempering, which `emission` once
        // said in one word: `"gaussian"` was `"full"`, `"robust"` the
        // per-feature pair tempered (task 244, P-2).
        emission,
        prior_mean: mean_prior.clone(),
        prior_kappa: mean_precision_prior.unwrap_or(1.0),
        prior_nu: *degrees_of_freedom_prior,
        prior_scale: covariance_prior.clone(),
        robust_beta: robust_beta.unwrap_or(0.0),
        prune_below: prune_below.unwrap_or(1e-6),
        max_run: max_run.unwrap_or(10_000),
        min_weight: spec.min_periods_or_default(),
        warm_rows: *warm_rows,
        hazard_on_clock: hazard.is_some(),
        warm_clock: warm_span.as_ref().map(Span::value),
    })
}

/// A `corrchange` spec's [`CorrChangeCfg`]; every parameter check is the
/// model's.
pub fn corrchange_cfg(spec: &Spec) -> Result<CorrChangeCfg, String> {
    let ModelKind::CorrChange {
        kind,
        window_rows: span_rows,
        alpha,
        alpha_adjust,
        bandwidth,
        scalar,
        crit,
        n_perm,
        permute_every_rows,
        perm_block,
        norm,
        seed,
        change_action,
        monitor_rows,
        boundary_gamma,
        permute_every,
        window_size,
        monitor_span,
        max_state_bytes,
    } = &spec.model
    else {
        return Err(not_a(spec, "corrchange_cfg", "corrchange"));
    };
    // Every problem at once, but a kind outside the three stops the
    // checks: each after it asks what the kind reads.
    let mut no = rheon_core::Refusals::new();
    // Unset, the kind is the default, which a refusal of window_rows does
    // not name: the reader wrote no kind.
    let kind_given = kind.is_some();
    let (kind, name) = match kind.as_deref() {
        None | Some("monitor") => (CorrChangeKind::Monitor, "monitor"),
        Some("window") => (CorrChangeKind::Window, "window"),
        Some("sequential") => (CorrChangeKind::Sequential, "sequential"),
        Some(other) => {
            return Err(stop_at(
                no,
                format!(
                    "corrchange: {}",
                    crate::words_refusal("kind", &["monitor", "sequential", "window"], other)
                ),
            ));
        }
    };
    let span_rows = match (*span_rows, window_size) {
        (Some(rows), _) => rows,
        // On the clock: the model refuses the two together, and reads the
        // span from `span_clock`.
        (None, Some(_)) => 0,
        (None, None) => {
            // A subject either way: the spec's render drops "corrchange: ".
            let kind_said = if kind_given {
                format!("kind={name:?} ")
            } else {
                "the model ".to_string()
            };
            no.refuse(format!(
                "corrchange: {kind_said}needs window_rows, {}, but it is not set. Set window_rows \
                 to a number of rows, {} or more, or window_size to a span of the clock.",
                match kind {
                    CorrChangeKind::Sequential =>
                        "the rows of history each monitored period is tested against",
                    CorrChangeKind::Window => "the rows in each of the two windows compared",
                    CorrChangeKind::Monitor => "the rows of each span tested",
                },
                if kind == CorrChangeKind::Window { 3 } else { 8 }
            ));
            0
        }
    };
    let reset = match change_action.as_deref() {
        None | Some("flag") => false,
        Some("reset") => true,
        Some(other) => {
            no.refuse(format!(
                "corrchange: {}",
                crate::words_refusal("change_action", &["flag", "reset"], other)
            ));
            false
        }
    };
    let norm = match norm.as_deref() {
        None | Some("l1") => ChangeNorm::L1,
        Some("linf") => ChangeNorm::LInf,
        Some(other) => {
            no.refuse(format!(
                "corrchange: {}",
                crate::words_refusal("norm", &["l1", "linf"], other)
            ));
            ChangeNorm::L1
        }
    };
    // A parameter that belongs to another kind is refused, not ignored: a
    // `crit` given to `"monitor"` changed nothing, in silence. A value equal
    // to the builders' default (`norm = "l1"`, `change_action = "flag"`) is
    // taken as unset, since the Python builder writes those whether or not
    // asked.
    let mut refuse = |set: bool, param: &str, kinds: &str, why: &str| {
        if set {
            no.refuse(format!(
                "corrchange: {param} applies only under kind={kinds}, but kind is \
                 {name:?}{why}. Leave {param} out, or set kind={kinds}."
            ));
        }
    };
    let (monitor, window, sequential) = (
        kind == CorrChangeKind::Monitor,
        kind == CorrChangeKind::Window,
        kind == CorrChangeKind::Sequential,
    );
    if !window {
        refuse(n_perm.is_some(), "n_perm", "\"window\"", "");
        refuse(
            permute_every_rows.is_some(),
            "permute_every_rows",
            "\"window\"",
            "",
        );
        refuse(perm_block.is_some(), "perm_block", "\"window\"", "");
        refuse(permute_every.is_some(), "permute_every", "\"window\"", "");
        refuse(seed.is_some(), "seed", "\"window\"", "");
        refuse(norm != ChangeNorm::L1, "norm", "\"window\"", "");
        refuse(
            reset,
            "change_action=\"reset\"",
            "\"window\"",
            if monitor {
                ", whose span already ends at its flag"
            } else {
                ", whose monitored period already ends at its flag"
            },
        );
    }
    refuse(
        monitor && crit.is_some(),
        "crit",
        "\"window\" or \"sequential\"",
        ", whose critical value is the Kolmogorov quantile",
    );
    refuse(
        window && bandwidth.is_some(),
        "bandwidth",
        "\"monitor\" or \"sequential\"",
        "",
    );
    // `"window"` takes its permutation quantile at `alpha` itself, so a
    // spread over the pairs changed nothing there (task 160, CD4). The
    // builders' default, "bonferroni", is taken as unset.
    refuse(
        window && alpha_adjust.as_deref().is_some_and(|a| a != "bonferroni"),
        "alpha_adjust",
        "\"monitor\" or \"sequential\"",
        ", which takes its permutation quantile at alpha itself",
    );
    if !sequential {
        refuse(monitor_rows.is_some(), "monitor_rows", "\"sequential\"", "");
        refuse(monitor_span.is_some(), "monitor_span", "\"sequential\"", "");
        refuse(
            boundary_gamma.is_some(),
            "boundary_gamma",
            "\"sequential\"",
            "",
        );
    }
    // The rows a span of rows holds are bounded by its count; only a span
    // of the clock holds what the data sizes.
    if max_state_bytes.is_some() && window_size.is_none() {
        no.refuse(
            "corrchange: max_state_bytes applies only under window_size, where the rows a span \
             of the clock holds are the data's to number; window_rows bounds them already. \
             Leave max_state_bytes out, or give window_size in place of window_rows.",
        );
    }
    no.finish()?;
    Ok(CorrChangeCfg {
        n_features: spec.k(),
        kind,
        span_rows,
        alpha: alpha.unwrap_or(0.05),
        alpha_adjust: alpha_adjust.clone().unwrap_or_else(|| "bonferroni".into()),
        bandwidth: *bandwidth,
        scalar: scalar.unwrap_or(false),
        decay: rheon_core::Decay::Lam(1.0),
        crit: *crit,
        n_perm: n_perm.unwrap_or(200),
        // One critical value in force for one shift of the windows: the
        // draws then cost `n_perm · k²` a row whatever `window_rows` is.
        // A clock cadence alone has no count of reports, and under
        // `window_size` the shift is a span of the clock (below).
        permute_every_rows: permute_every_rows.unwrap_or(
            if permute_every.is_some() || window_size.is_some() {
                usize::MAX
            } else {
                span_rows
            },
        ),
        perm_block: perm_block.unwrap_or(1),
        norm,
        seed: seed.unwrap_or(0),
        reset,
        // W&G's `T = 1`: as many rows monitored as the history has, or
        // under `window_size` a period as long as the history's span.
        monitor_rows: match (sequential, window_size) {
            (false, _) => 0,
            (true, None) => monitor_rows.unwrap_or(span_rows),
            (true, Some(_)) => monitor_rows.unwrap_or(0),
        },
        boundary_gamma: boundary_gamma.unwrap_or(0.0),
        // Under `window_size` the value is redrawn once the windows have
        // moved on by a window, as `window_rows`' default redraws once
        // they have moved on by its rows.
        permute_every: permute_every.as_ref().map(Span::value).or_else(|| {
            window_size
                .as_ref()
                .filter(|_| window && permute_every_rows.is_none())
                .map(Span::value)
        }),
        span_clock: window_size.as_ref().map(Span::value),
        monitor_clock: match (sequential, window_size, monitor_span) {
            (true, _, Some(s)) => Some(s.value()),
            (true, Some(w), None) if monitor_rows.is_none() => Some(w.value()),
            (_, _, s) => s.as_ref().map(Span::value),
        },
        max_state_bytes: *max_state_bytes,
    })
}

/// An `hmm` spec's [`HmmCfg`]. The decay is the caller's; every other check
/// is the model's, so `Spec::validate` gets the same messages.
pub fn hmm_cfg(spec: &Spec) -> Result<HmmCfg, String> {
    let ModelKind::Hmm {
        k,
        covariance_type,
        precision_prior,
        learn,
        transition_prior,
        transition,
        means,
        covs,
        warm_rows,
        init,
        seed,
        exog_tvtp,
        tvtp_coef,
        prior_rows,
        warm_span,
    } = &spec.model
    else {
        return Err(not_a(spec, "hmm_cfg", "hmm"));
    };
    // Every problem at once; none of these reads what another refuses.
    let mut no = rheon_core::Refusals::new();
    let tvtp = match (exog_tvtp, tvtp_coef) {
        (Some(_), Some(ab)) if ab.len() == 2 => Some((ab[0].clone(), ab[1].clone())),
        (Some(_), got) => {
            no.refuse(format!(
                "hmm: exog_tvtp needs tvtp_coef, two n_components by n_components matrices, \
                 each flattened by rows, but {}. Give tvtp_coef as a list of the two matrices, \
                 or leave exog_tvtp out.",
                got.as_ref().map_or_else(
                    || "tvtp_coef is not set".to_string(),
                    |ab| format!(
                        "tvtp_coef holds {} matri{}",
                        ab.len(),
                        if ab.len() == 1 { "x" } else { "ces" }
                    )
                )
            ));
            None
        }
        (None, Some(_)) => {
            no.refuse(
                "hmm: tvtp_coef needs exog_tvtp, the column its transitions read, but \
                 exog_tvtp is not set. Set exog_tvtp, or leave tvtp_coef out.",
            );
            None
        }
        (None, None) => None,
    };
    // A parameter its mode does not read is refused, not ignored (review
    // 2026-10-06, CE4): under `tvtp_coef` no transition count is learned,
    // and with the states given nothing is seeded. A value given is told
    // from the default here, where both are `None` until filled.
    for (param, given) in [
        ("transition", transition.is_some()),
        ("transition_prior", transition_prior.is_some()),
    ] {
        if given && tvtp.is_some() {
            no.refuse(format!(
                "hmm: {param} does not apply under tvtp_coef, which reads each row's \
                 transitions from exog_tvtp and learns no transition counts. Leave {param} out, \
                 or leave out tvtp_coef."
            ));
        }
    }
    if warm_rows.is_some() && warm_span.is_some() {
        no.refuse(
            "hmm: warm_span and warm_rows are two warm-ups, and the states are seeded once. \
             Leave one of them out.",
        );
    }
    for (param, given) in [
        ("warm_rows", warm_rows.is_some()),
        ("warm_span", warm_span.is_some()),
        ("init", init.is_some()),
        ("seed", seed.is_some()),
    ] {
        if given && means.is_some() && covs.is_some() {
            no.refuse(format!(
                "hmm: {param} does not apply when means and covs are given, since the states \
                 are then not seeded from the first rows. Leave {param} out, or leave out means \
                 and covs."
            ));
        }
    }
    let covariance = covariance_type
        .as_deref()
        .map_or(Ok(Covariance::Full), Covariance::parse);
    let seed_rule = crate::stream::init_rule(init.as_deref());
    no.take(covariance.as_ref().map(|_| ()).map_err(Clone::clone));
    no.take(seed_rule.as_ref().map(|_| ()).map_err(Clone::clone));
    no.finish()?;
    Ok(HmmCfg {
        n_features: spec.k(),
        k: *k,
        decay: rheon_core::Decay::Lam(1.0),
        covariance: covariance?,
        precision_prior: *precision_prior,
        min_weight: spec.min_periods_or_default(),
        learn: learn.unwrap_or(true),
        transition_prior: transition_prior.unwrap_or(1.0),
        transition: transition.clone(),
        means: means.clone(),
        covs: covs.clone(),
        // 1,000 rows, or `k` where that is more: the buffer must hold every
        // regime several times over for the seeding to place a state on each
        // (task 278, D9: at 50, regimes lasting 50 rows were never found).
        // Under a span the buffer holds the span's rows, at least `k`.
        warm_rows: warm_rows.unwrap_or_else(|| {
            if warm_span.is_some() {
                *k
            } else {
                (*k).max(1000)
            }
        }),
        seed_rule: seed_rule?,
        seed: seed.unwrap_or(0),
        tvtp,
        prior_rows: *prior_rows,
        warm_clock: warm_span.as_ref().map(Span::value),
    })
}

/// An `rcov` spec's [`RcovCfg`]. Every parameter check is the model's, so
/// `Spec::validate` and the CLI get one set of messages.
pub fn rcov_cfg(spec: &Spec) -> Result<RcovCfg, String> {
    let ModelKind::Rcov {
        kind,
        kernel,
        bandwidth,
        jitter,
        theta,
        psd,
        block_rows,
        max_bandwidth,
        preavg_rows,
        noise_stride,
        iv_stride,
    } = &spec.model
    else {
        return Err(not_a(spec, "rcov_cfg", "rcov"));
    };
    let kind = match kind.as_deref() {
        None | Some("kernel") => RcovKind::Kernel,
        Some("preavg") => RcovKind::Preavg,
        Some("plain") => RcovKind::Plain,
        Some(other) => {
            return Err(stop_at(
                rheon_core::Refusals::new(),
                format!(
                    "rcov: {}",
                    crate::words_refusal("kind", &["kernel", "preavg", "plain"], other)
                ),
            ));
        }
    };
    let mut no = rheon_core::Refusals::new();
    // A parameter its kind does not read is refused, not ignored, as
    // `bandwidth` and `preavg_rows` are (review 2026-10-06, CE4): the jitter
    // and the ring are the kernel's, `theta` sets the pre-averaging window.
    // The builders leave both `None` unless given.
    for (param, given, owner) in [
        ("jitter", jitter.is_some(), RcovKind::Kernel),
        ("max_bandwidth", max_bandwidth.is_some(), RcovKind::Kernel),
        ("theta", theta.is_some(), RcovKind::Preavg),
    ] {
        if given && kind != owner {
            no.refuse(format!(
                "rcov: {param} applies only under kind={:?}, but kind is {:?}. Leave {param} \
                 out, or set kind={:?}.",
                owner.as_str(),
                kind.as_str(),
                owner.as_str()
            ));
        }
    }
    // `preavg_rows` fixes the window, which is all `theta` sets: a `theta`
    // beside it is read by nothing, and is refused, as the Python builder
    // refuses it.
    if let (Some(theta), Some(_)) = (theta, preavg_rows) {
        no.refuse(format!(
            "rcov: theta={theta:?} has no effect beside preavg_rows, which fixes the \
             pre-averaging span theta would set. Leave theta out."
        ));
    }
    no.finish()?;
    Ok(RcovCfg {
        n_features: spec.k(),
        kind,
        kernel: kernel.clone().unwrap_or_else(|| "parzen".into()),
        bandwidth: *bandwidth,
        jitter: jitter.unwrap_or(2),
        theta: theta.unwrap_or(1.0),
        psd: psd.unwrap_or(true),
        block_rows: *block_rows,
        max_bandwidth: *max_bandwidth,
        preavg_rows: *preavg_rows,
        noise_stride: noise_stride.unwrap_or(1),
        iv_stride: iv_stride.unwrap_or(20),
    })
}

/// The named blocks of a `deco` spec, if it has any.
fn blocks_named(model: &ModelKind) -> Option<&Vec<(String, Vec<String>)>> {
    match model {
        ModelKind::Deco { blocks, .. } => blocks.as_ref(),
        _ => None,
    }
}

/// A `deco` spec's [`DecoCfg`], with the block *names* resolved to feature
/// positions. The decay is the caller's (one instance per half-life); every
/// other check is `DecoCfg::validate`'s, so `Spec::validate` gets the same
/// messages the model would give.
pub fn deco_cfg(spec: &Spec) -> Result<DecoCfg, String> {
    let ModelKind::Deco {
        dynamics,
        alpha,
        beta,
        blocks,
    } = &spec.model
    else {
        return Err(not_a(spec, "deco_cfg", "deco"));
    };
    // Every problem at once; none of these reads what another refuses.
    let mut no = rheon_core::Refusals::new();
    let dynamics = match dynamics.as_deref() {
        None | Some("ew") => DecoDynamics::Ew,
        Some("linear") => DecoDynamics::Linear,
        Some(other) => {
            no.refuse(format!(
                "deco: {}",
                crate::words_refusal("dynamics", &["ew", "linear"], other)
            ));
            DecoDynamics::Ew
        }
    };
    let blocks = match blocks {
        None => Vec::new(),
        Some(named) if named.is_empty() => {
            // An empty list is not "one block of everything" and not the
            // unblocked form either; whichever was meant, say which
            // (docs/REVIEW-E54-E64.md D1).
            no.refuse(
                "deco: blocks must name at least one block, but it names none. Leave blocks \
                 out for one correlation shared by every pair of features, or name at least one \
                 block.",
            );
            Vec::new()
        }
        Some(named) => named
            .iter()
            .map(|(name, cols)| {
                cols.iter()
                    .filter_map(|c| {
                        let at = spec.features.iter().position(|f| f == c);
                        if at.is_none() {
                            no.refuse(format!(
                                "deco: block {name:?} lists {c:?}, which is not one of \
                                 features. Add {c:?} to features, or remove it from the block."
                            ));
                        }
                        at
                    })
                    .collect::<Vec<usize>>()
            })
            .collect::<Vec<_>>(),
    };
    // The blocks' rules said by name, where the reader wrote names: the
    // core's own check (`DecoCfg::validate`) knows only positions, and
    // stays as the Rust API's. Once every name is a feature.
    if let Some(named) = blocks_named(&spec.model)
        && !named.is_empty()
        && named
            .iter()
            .all(|(_, cols)| cols.iter().all(|c| spec.features.contains(c)))
    {
        no.take(deco_blocks_by_name(&spec.features, named));
    } else if blocks_named(&spec.model).is_some_and(|named| !named.is_empty()) {
        // A block naming another column leaves the by-name rules (a
        // feature in no block among them) unchecked: say so.
        if !no.is_empty() {
            let _ = no.stop_here();
        }
    }
    // A JSON or TOML spec writes `blocks` as an array of pairs, so two of
    // them can carry the same name; the collision then surfaces as two
    // output fields called `u_u` rather than as the block list's problem.
    if let Some(named) = blocks_named(&spec.model) {
        let mut seen = std::collections::HashSet::new();
        for (dup, _) in named.iter().filter(|(n, _)| !seen.insert(n.as_str())) {
            no.refuse(format!(
                "deco: blocks names {dup:?} more than once, but a block's name labels its \
                 output fields. Rename all but one of them."
            ));
        }
    }
    no.finish()?;
    Ok(DecoCfg {
        n_features: spec.k(),
        decay: rheon_core::Decay::Lam(1.0),
        dynamics,
        alpha: *alpha,
        beta: *beta,
        blocks,
        min_weight: spec.min_periods_or_default(),
    })
}

/// `deco`'s blocks, each of whose columns is a feature, held to the
/// model's rules by name: each block lists each feature once and holds at
/// least two, and each feature is in exactly one block. Every problem at
/// once.
fn deco_blocks_by_name(features: &[String], named: &[(String, Vec<String>)]) -> Result<(), String> {
    let quoted = |v: &[&String]| {
        let q: Vec<String> = v.iter().map(|c| format!("{c:?}")).collect();
        match q.split_last() {
            Some((last, rest)) if !rest.is_empty() => format!("{} and {last}", rest.join(", ")),
            _ => q.join(""),
        }
    };
    let mut no = rheon_core::Refusals::new();
    let mut owner: Vec<(&String, Vec<&String>)> = Vec::new();
    for (name, cols) in named {
        let mut distinct: Vec<&String> = Vec::new();
        let mut twice: Vec<&String> = Vec::new();
        for c in cols {
            if distinct.contains(&c) {
                if !twice.contains(&c) {
                    twice.push(c);
                }
            } else {
                distinct.push(c);
            }
        }
        if !twice.is_empty() {
            no.refuse(format!(
                "deco: block {name:?} lists {} more than once. List each feature once in a block.",
                quoted(&twice)
            ));
        }
        if distinct.len() < 2 {
            no.refuse(format!(
                "deco: block {name:?} must hold at least two features, since its correlation is \
                 between them, but it holds {}. Add a feature to it, or merge it with another \
                 block.",
                if distinct.is_empty() {
                    "none".to_string()
                } else {
                    format!("one, {}", quoted(&distinct))
                }
            ));
        }
        for c in distinct {
            match owner.iter_mut().find(|(f, _)| *f == c) {
                Some((_, blocks)) => blocks.push(name),
                None => owner.push((c, vec![name])),
            }
        }
    }
    for (c, blocks) in &owner {
        if blocks.len() > 1 {
            no.refuse(format!(
                "deco: feature {c:?} is in blocks {}, but each feature belongs to one block. \
                 Remove it from all but one.",
                quoted(blocks)
            ));
        }
    }
    let missing: Vec<&String> = features
        .iter()
        .filter(|f| !owner.iter().any(|(c, _)| c == f))
        .collect();
    if !missing.is_empty() {
        no.refuse(format!(
            "deco: every feature must be in exactly one block, but {} {} in none. Add {} to a \
             block, or remove {} from features.",
            quoted(&missing),
            if missing.len() == 1 { "is" } else { "are" },
            if missing.len() == 1 { "it" } else { "each" },
            if missing.len() == 1 { "it" } else { "them" },
        ));
    }
    no.finish()
}

/// The block names of a `deco` spec, in emission order; empty when it has
/// none, which is the unblocked model.
pub fn deco_block_names(spec: &Spec) -> Vec<String> {
    match &spec.model {
        ModelKind::Deco {
            blocks: Some(named),
            ..
        } => named.iter().map(|(n, _)| n.clone()).collect(),
        _ => Vec::new(),
    }
}

/// A `huber` spec's model: `ewridge`'s sums per target, each row at its Huber weight.
pub fn huber_model(spec: &Spec, decay: Decay) -> Result<AnyModel, String> {
    let ModelKind::Huber {
        huber_delta,
        huber_scale,
        huber_scale_half_life,
        ridge,
        standardize,
        solve_every,
        max_rows_between_solves,
        feature_sets: _,
        ridge_scale: _,
        emit_huber_scale,
        huber_refit_rows,
        huber_refit_span,
        max_state_bytes,
        refit_budget: _,
    } = &spec.model
    else {
        return Err(not_a(spec, "huber_model", "huber"));
    };
    let cfg = RobustCfg {
        n_features: spec.k(),
        n_targets: spec.m(),
        fit_intercept: spec.fit_intercept,
        decay,
        // The 95%-efficiency constant (Huber 1981; statsmodels'
        // `HuberT`, scikit-learn's 1.35): task 195, U3.
        loss: RobustLoss::Huber {
            delta: huber_delta.map_or(1.345, |n| n.0),
        },
        ridge: ridge.unwrap_or(1e-6),
        standardize: *standardize,
        min_weight: spec.min_periods_or_default(),
        solve_every: solve_every
            .as_ref()
            .map_or_else(|| spec.solve_every_default(decay), solve_every_of),
        max_rows_between_solves: max_rows_between_solves.unwrap_or(u32::MAX),
        solve_share: spec.solve_share_default(solve_every.as_ref(), decay),
        quantile_eps: 1e-3,
        huber_scale: huber_scale_of(huber_scale.as_deref()),
        huber_scale_decay: huber_scale_half_life
            .as_ref()
            .map(|h| Decay::Halflife(h.value())),
        feature_sets: huber_feature_sets(spec),
        ridge_scale: huber_ridge_scale_sum(spec),
        emit_scale: *emit_huber_scale,
        // On by default, but under the sum-scale prior, which the refit
        // does not hold, and an infinite cut, least squares, with no
        // outliers to refit for (`RobustCfg::refit_rows`): there it is off
        // unless asked for, and asked for it is refused.
        refit_rows: huber_refit_rows.unwrap_or(
            if huber_refit_span.is_some()
                || huber_ridge_scale_sum(spec)
                || huber_delta.is_some_and(|d| d.0.is_infinite())
            {
                0
            } else {
                rheon_core::default_refit_rows(spec.k())
            },
        ),
        // Each MAD sketch in units of its first residual, so the scale
        // moves with a rescaled target to rounding; a state written
        // without it loads with it off.
        mad_anchor: true,
        min_weight_holds_learning: false,
        refit_clock: huber_refit_span.as_ref().map(Span::value),
        max_state_bytes: *max_state_bytes,
    };
    let mut m = Robust::new(cfg)?;
    // Each target's own threshold, for its own first solve (S9b).
    m.set_target_min_weight(spec.min_periods_per_target())?;
    Ok(AnyModel::Robust(Box::new(m)))
}

/// A `huber` spec's feature sets as the model takes them: each set's
/// columns as positions in the spec's features, which `Spec::validate` has
/// held to name them; none without sets.
fn huber_feature_sets(spec: &Spec) -> Vec<(String, Vec<usize>)> {
    let ModelKind::Huber {
        feature_sets: Some(sets),
        ..
    } = &spec.model
    else {
        return Vec::new();
    };
    sets.iter()
        .map(|(name, cols)| {
            let idx = cols
                .iter()
                .filter_map(|c| spec.features.iter().position(|f| f == c))
                .collect();
            (name.clone(), idx)
        })
        .collect()
}

/// Whether a `huber` spec asks for RLS's sum-scale prior.
fn huber_ridge_scale_sum(spec: &Spec) -> bool {
    matches!(
        spec.model,
        ModelKind::Huber {
            ridge_scale: crate::spec::RidgeScale::Sum,
            ..
        }
    )
}

/// A `quantile` spec's model.
pub fn quantile_model(spec: &Spec, decay: Decay) -> Result<AnyModel, String> {
    let ModelKind::Quantile {
        quantile,
        ridge,
        standardize,
        solve_every,
        max_rows_between_solves,
        quantile_eps,
        quantile_refit_rows,
        quantile_refit_span,
        max_state_bytes,
        refit_budget: _,
    } = &spec.model
    else {
        return Err(not_a(spec, "quantile_model", "quantile"));
    };
    let cfg = RobustCfg {
        n_features: spec.k(),
        n_targets: spec.m(),
        fit_intercept: spec.fit_intercept,
        decay,
        loss: RobustLoss::Quantile { tau: *quantile },
        ridge: ridge.unwrap_or(1e-6),
        standardize: *standardize,
        min_weight: spec.min_periods_or_default(),
        solve_every: solve_every
            .as_ref()
            .map_or_else(|| spec.solve_every_default(decay), solve_every_of),
        max_rows_between_solves: max_rows_between_solves.unwrap_or(u32::MAX),
        solve_share: spec.solve_share_default(solve_every.as_ref(), decay),
        quantile_eps: quantile_eps.unwrap_or(0.2),
        // The band is drawn in the residuals' MAD, on the model's clock:
        // the RMS let the wild rows widen the band drawn to bound their
        // pull (task 278, D-Q).
        huber_scale: HuberScale::Mad,
        huber_scale_decay: None,
        feature_sets: Vec::new(),
        ridge_scale: false,
        emit_scale: false,
        // On by default, at `huber`'s count: the rows the warm-up took as
        // least squares and the first bands held refit to the converged
        // fit, once (task 278, D-Q2).
        refit_rows: quantile_refit_rows.unwrap_or_else(|| {
            if quantile_refit_span.is_some() {
                0
            } else {
                rheon_core::default_refit_rows(spec.k())
            }
        }),
        // No sketch under the RMS band; anchored once a sketch is kept.
        mad_anchor: true,
        min_weight_holds_learning: false,
        refit_clock: quantile_refit_span.as_ref().map(Span::value),
        max_state_bytes: *max_state_bytes,
    };
    let mut m = Robust::new(cfg)?;
    // Each target's own threshold, for its own first solve (S9b).
    m.set_target_min_weight(spec.min_periods_per_target())?;
    Ok(AnyModel::Robust(Box::new(m)))
}

/// An `sgd` spec's model, its loss and schedule read from their words.
pub fn sgd_model(spec: &Spec, decay: Decay) -> Result<AnyModel, String> {
    let ModelKind::Sgd {
        loss,
        huber_delta,
        huber_scale,
        huber_scale_half_life,
        huber_refit_rows,
        huber_refit_span,
        max_state_bytes,
        refit_budget: _,
        quantile,
        eps,
        learning_rate,
        schedule,
        power,
        l2,
        clip_gradient,
        standardize,
        coef_min,
        coef_max,
        strict_binary,
        coef_sum,
    } = &spec.model
    else {
        return Err(not_a(spec, "sgd_model", "sgd"));
    };
    // Both words at once: neither check reads the other.
    let mut no = rheon_core::Refusals::new();
    let loss: Result<SgdLoss, String> = (|| {
        Ok(match loss.as_deref().unwrap_or("squared") {
            "squared" => SgdLoss::Squared,
            // In units of the target's residual std, `huber`'s constant
            // under `huber`'s name (task 195, U1 and U3).
            "huber" => SgdLoss::Huber {
                delta: huber_delta.map_or(1.345, |n| n.0),
            },
            "quantile" => SgdLoss::Quantile {
                tau: quantile.ok_or(
                    "sgd: loss=\"quantile\" needs quantile, the level to fit, but it is not set. \
                 Set quantile to a level above 0 and below 1, such as 0.5 for the median.",
                )?,
            },
            // 1% of the target's own spread (task 203), kept by task
            // 207's sweep, which measured it against the residual's
            // spread capped by the target's.
            "epsilon_insensitive" => SgdLoss::EpsilonInsensitive {
                eps: eps.unwrap_or(0.01),
            },
            "poisson" => SgdLoss::Poisson,
            "logistic" => SgdLoss::Logistic,
            other => {
                return Err(format!(
                    "sgd: {}",
                    crate::words_refusal(
                        "loss",
                        &[
                            "squared",
                            "huber",
                            "quantile",
                            "epsilon_insensitive",
                            "poisson",
                            "logistic"
                        ],
                        other
                    )
                ));
            }
        })
    })();
    let sched = match schedule.as_deref().unwrap_or("constant") {
        "constant" => Ok(LearningRate::Constant),
        "inv_scaling" => Ok(LearningRate::InvScaling {
            power: power.unwrap_or(0.5),
        }),
        "adagrad" => Ok(LearningRate::AdaGrad),
        other => Err(format!(
            "sgd: {}",
            crate::words_refusal("schedule", &["constant", "inv_scaling", "adagrad"], other)
        )),
    };
    let loss = no.value(loss);
    let sched = no.value(sched);
    no.finish()?;
    let (Some(loss), Some(sched)) = (loss, sched) else {
        unreachable!("a refused word is in the refusal just returned");
    };
    let cfg = SgdCfg {
        n_features: spec.k(),
        n_targets: spec.m(),
        fit_intercept: spec.fit_intercept,
        decay,
        loss,
        learning_rate: learning_rate.unwrap_or(0.01),
        schedule: sched,
        l2: l2.unwrap_or(0.0),
        min_weight: spec.min_periods_or_default(),
        // Finite by default: see SgdCfg::clip_gradient.
        clip_gradient: clip_gradient.map_or(1e3, |n| n.0),
        standardize: *standardize,
        constraint: constraint(spec.k(), coef_min, coef_max, *coef_sum),
        strict_binary: *strict_binary,
        // `huber`'s scale under the Huber loss; every other loss
        // reads none, and the spec refuses one beside it.
        huber_scale: if matches!(loss, SgdLoss::Huber { .. }) {
            huber_scale_of(huber_scale.as_deref())
        } else {
            HuberScale::Rms
        },
        huber_scale_decay: huber_scale_half_life
            .as_ref()
            .filter(|_| matches!(loss, SgdLoss::Huber { .. }))
            .map(|h| Decay::Halflife(h.value())),
        // `huber`'s refit under the Huber loss, on by default; every other
        // loss has none, and the spec refuses one beside it.
        // Off under an infinite cut, the squared loss, with no outliers to
        // refit for.
        refit_rows: match loss {
            SgdLoss::Huber { delta } if delta.is_finite() => {
                huber_refit_rows.unwrap_or_else(|| {
                    if huber_refit_span.is_some() {
                        0
                    } else {
                        rheon_core::default_refit_rows(spec.k())
                    }
                })
            }
            SgdLoss::Huber { .. } => huber_refit_rows.unwrap_or(0),
            _ => 0,
        },
        min_weight_holds_learning: false,
        // The target's own units and the unstable rate's gate (task 278:
        // D1, D2, D8, D-S), always on from a spec; a state written before
        // them loads with them off.
        target_spread: true,
        // `huber`'s anchored MAD sketch under the Huber loss.
        mad_anchor: true,
        refit_clock: huber_refit_span.as_ref().map(Span::value),
        max_state_bytes: *max_state_bytes,
    };
    Ok(AnyModel::Sgd(Box::new(Sgd::new(cfg)?)))
}

/// The floor a model is read at on a row its `min_weight` may withhold
/// ([`Persisted::ungated_spread`]): the floor the model keeps with no
/// `min_weight` given. That is 0 for every model but `ewridge`, whose own
/// floor is then a row per unknown (`build_bare`): below it its system has
/// fewer rows than unknowns, and the spec with no `min_weight` reports
/// nothing there, the noise gate being its readiness gate. An explicit
/// `min_weight` below that lowers the model's own floor, so the read is at
/// the lower of the two, and a `min_weight` above a row per unknown
/// withholds what the spec without one reports and nothing else. The floor
/// counts weight, as the model's does. `ewridge` runs no solve its weight
/// share `s` brings due -- `solve_share`, or `ln 2 / 50` in the default
/// schedule and an explicit cadence's first period -- while its weight is
/// below `(1 - s)(1 - 2⁻¹⁶)` of the least of this floor and its targets'
/// own, and runs every one for `s > 1/2` (`rheon_core`'s
/// `ewridge/unread.rs`; tasks 311 and 313): this read is one reason a row
/// per unknown -- `k + 1`, `k` without an intercept -- is in it.
pub(crate) fn read_floor(spec: &Spec, min_weight: &[f64]) -> f64 {
    match spec.model {
        ModelKind::EwRidge { .. } => {
            let own = (spec.k() + usize::from(spec.fit_intercept)) as f64;
            min_weight.iter().copied().fold(own, f64::min)
        }
        _ => 0.0,
    }
}

/// The words a `glm`'s `family` takes, the default first.
pub const GLM_FAMILIES: [&str; 4] = ["poisson", "gamma", "binomial", "tweedie"];

/// The words a `glm`'s `link` takes.
pub const GLM_LINKS: [&str; 4] = ["log", "logit", "inverse", "power"];

/// A `glm` spec's configuration, its decay `Lam(1)` for the caller to set:
/// the family and link read from their words, the refit's default from the
/// features ([`rheon_core::default_glm_refit_rows`]) and the cap's from the
/// refit ([`rheon_core::default_glm_step_cap`]).
pub fn glm_cfg(spec: &Spec) -> Result<rheon_core::GlmCfg, String> {
    use rheon_core::{GlmFamily, GlmLink};
    let ModelKind::Glm {
        family,
        link,
        var_power,
        ridge,
        refit_rows,
        refit_span,
        max_state_bytes,
        step_cap,
        ..
    } = &spec.model
    else {
        return Err(not_a(spec, "glm_cfg", "glm"));
    };
    let words = |param: &str, words: &[&str], got: &str| {
        crate::spec::spec_words_refusal(&spec.name, param, words, got)
    };
    let family = match family.as_deref().unwrap_or("poisson") {
        "poisson" => GlmFamily::Poisson,
        "gamma" => GlmFamily::Gamma,
        "binomial" => GlmFamily::Binomial,
        "tweedie" => GlmFamily::Tweedie,
        other => return Err(words("family", &GLM_FAMILIES, other)),
    };
    let link = match link.as_deref() {
        None => family.default_link(),
        Some("log") => GlmLink::Log,
        Some("logit") => GlmLink::Logit,
        Some("inverse") => GlmLink::Inverse,
        Some("power") => GlmLink::Power,
        Some(other) => return Err(words("link", &GLM_LINKS, other)),
    };
    // A span in place of the count: the count's default is off beside it,
    // and one given is the model's to refuse.
    let refit_rows = refit_rows.unwrap_or_else(|| {
        if refit_span.is_some() {
            0
        } else {
            rheon_core::default_glm_refit_rows_within(
                spec.k(),
                spec.m(),
                max_state_bytes.unwrap_or(rheon_core::GLM_GROUP_REFIT_CAP as u64),
            )
        }
    });
    Ok(rheon_core::GlmCfg {
        n_features: spec.k(),
        n_targets: spec.m(),
        fit_intercept: spec.fit_intercept,
        decay: Decay::Lam(1.0),
        family,
        link,
        var_power: var_power.unwrap_or(1.5),
        ridge: ridge.unwrap_or(1.0),
        min_weight: spec.min_periods_or_default(),
        refit_rows,
        refit_clock: refit_span.as_ref().map(Span::value),
        max_state_bytes: *max_state_bytes,
        step_cap: step_cap.map_or_else(
            || rheon_core::default_glm_step_cap(refit_rows.max(u32::from(refit_span.is_some()))),
            |n| n.0,
        ),
    })
}

/// A `pca` spec's [`rheon_core::StreamPcaCfg`]: the features its columns,
/// two components unless the spec gives a count (one for one feature),
/// the incremental SVD unless it names a method. Every value is the model's
/// own to check.
pub fn pca_cfg(spec: &Spec, decay: Decay) -> Result<rheon_core::StreamPcaCfg, String> {
    use rheon_core::PcaMethod;
    let ModelKind::Pca {
        n_components,
        method,
        learning_rate,
        max_state_bytes,
    } = &spec.model
    else {
        return Err(not_a(spec, "pca_cfg", "pca"));
    };
    let method = match method.as_deref().unwrap_or("svd") {
        "svd" => PcaMethod::Svd,
        "ccipca" => PcaMethod::Ccipca,
        "oja" => PcaMethod::Oja,
        other => {
            return Err(format!(
                "pca: method must be \"svd\", \"ccipca\" or \"oja\", but it is {other:?}. \
                 Use one of those words."
            ));
        }
    };
    Ok(rheon_core::StreamPcaCfg {
        n_features: spec.k(),
        n_components: n_components.unwrap_or(spec.k().min(2)),
        method,
        decay,
        min_weight: spec.min_periods_or_default(),
        learning_rate: *learning_rate,
        max_state_bytes: *max_state_bytes,
    })
}

#[cfg(test)]
mod refit_tests {
    //! The refit's default (docs/PLAN.md task 277): on at
    //! `default_refit_rows(k)` for `huber` and `sgd(loss = "huber")`, off
    //! where the refit does not apply -- the sum-scale prior, an infinite
    //! cut, another loss -- and as given where given, refused there.
    use super::*;

    fn spec(model: &str) -> Spec {
        let text = format!(
            "name = \"m\"\ntargets = [\"y\"]\nfeatures = [\"x\"]\nhalf_life = 10.0\n[model]\n{model}\n"
        );
        toml::from_str(&text).unwrap_or_else(|e| panic!("did not parse: {e}\n{text}"))
    }

    fn huber(model: &str) -> Result<u32, String> {
        match huber_model(
            &spec(&format!("type = \"huber\"\n{model}")),
            Decay::Halflife(10.0),
        )? {
            AnyModel::Robust(m) => Ok(m.cfg().refit_rows),
            _ => unreachable!(),
        }
    }

    fn sgd(model: &str) -> Result<u32, String> {
        match sgd_model(
            &spec(&format!("type = \"sgd\"\n{model}")),
            Decay::Halflife(10.0),
        )? {
            AnyModel::Sgd(m) => Ok(m.cfg().refit_rows),
            _ => unreachable!(),
        }
    }

    #[test]
    fn the_refit_is_on_by_default_where_it_applies() {
        let on = rheon_core::default_refit_rows(1);
        assert_eq!(huber(""), Ok(on));
        assert_eq!(huber("huber_refit_rows = 7"), Ok(7));
        assert_eq!(huber("huber_refit_rows = 0"), Ok(0));
        assert_eq!(huber("ridge_scale = \"sum\""), Ok(0));
        assert_eq!(huber("huber_delta = inf"), Ok(0));
        assert!(huber("ridge_scale = \"sum\"\nhuber_refit_rows = 7").is_err());
        assert!(huber("huber_delta = inf\nhuber_refit_rows = 7").is_err());
        assert_eq!(sgd("loss = \"huber\""), Ok(on));
        assert_eq!(sgd("loss = \"huber\"\nhuber_refit_rows = 7"), Ok(7));
        assert_eq!(sgd("loss = \"huber\"\nhuber_delta = inf"), Ok(0));
        assert!(sgd("loss = \"huber\"\nhuber_delta = inf\nhuber_refit_rows = 7").is_err());
        assert_eq!(sgd(""), Ok(0));
        assert_eq!(sgd("loss = \"quantile\"\nquantile = 0.5"), Ok(0));
        // `quantile`'s (task 278, D-Q2): on at the same count, as given
        // where given, and refused inside the warm-up's three rows a
        // coefficient (6 here, with the intercept).
        assert_eq!(quantile(""), Ok(on));
        assert_eq!(quantile("quantile_refit_rows = 0"), Ok(0));
        assert_eq!(quantile("quantile_refit_rows = 6"), Ok(6));
        assert!(quantile("quantile_refit_rows = 5").is_err());
    }

    fn quantile(model: &str) -> Result<u32, String> {
        match quantile_model(
            &spec(&format!("type = \"quantile\"\nquantile = 0.5\n{model}")),
            Decay::Halflife(10.0),
        )? {
            AnyModel::Robust(m) => Ok(m.cfg().refit_rows),
            _ => unreachable!(),
        }
    }
}
