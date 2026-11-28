//! The words for a state the data sizes (`hawkes`' and `realized`'s windows
//! of clock units, `garch`'s start over a span of the clock),
//! which the settings cannot bound at the call: the notice that it is past
//! half of its cap, and what the bank says when a row would pass the cap.

use super::*;

/// A span of the clock that holds rows, of a kind with a fixed cap a group
/// (docs/PLAN.md task 538): the parameter as the spec writes it, what it
/// holds, the way to hold fewer, and the cap's name, which no setting
/// raises.
pub(crate) struct FixedSpan {
    pub span: String,
    pub holds: &'static str,
    pub shorten: &'static str,
    pub cap: &'static str,
}

/// The [`FixedSpan`] of a spec, if it has one.
pub(crate) fn fixed_span(spec: &Spec) -> Option<FixedSpan> {
    let it = |span: String, holds, shorten, cap| {
        Some(FixedSpan {
            span,
            holds,
            shorten,
            cap,
        })
    };
    match &spec.model {
        ModelKind::KMeans {
            warm_span: Some(s), ..
        } => it(
            format!("warm_span = {s}"),
            "rows of its warm-up",
            "Shorten warm_span, or give warm_rows",
            "the 256 MiB a warm-up may hold",
        ),
        ModelKind::Bocpd {
            warm_span: Some(s),
            max_state_bytes: None,
            ..
        } => it(
            format!("warm_span = {s}"),
            "rows of its warm-up",
            "Shorten warm_span, or give warm_rows",
            "the 256 MiB a warm-up may hold",
        ),
        ModelKind::Hmm {
            warm_span: Some(s),
            max_state_bytes: None,
            ..
        } => it(
            format!("warm_span = {s}"),
            "rows of its warm-up",
            "Shorten warm_span, or give warm_rows",
            "the 256 MiB a warm-up may hold",
        ),
        _ => None,
    }
}

/// What a spec's window is, in its own words, and the ways to hold fewer
/// events: `window_size = 5m`, shortened.
fn window_words(spec: &Spec) -> (String, &'static str, &'static str) {
    if let Some(f) = fixed_span(spec) {
        return (f.span, f.shorten, "rows");
    }
    if let Some(w) = &spec.conformal_p_window_size {
        return (
            format!("conformal_p_window_size = {w}"),
            "Shorten conformal_p_window_size, or give conformal_p_window_rows",
            "rows",
        );
    }
    match &spec.model {
        ModelKind::Hawkes {
            window_size: Some(w),
            ..
        } => (
            format!("window_size = {w}"),
            "Shorten window_size",
            "events",
        ),
        ModelKind::Expectile {
            window_size: Some(w),
            ..
        }
        | ModelKind::Realized {
            window_size: Some(w),
            ..
        }
        | ModelKind::PowerVariation {
            window_size: Some(w),
            ..
        }
        | ModelKind::EwmMoments {
            window_size: Some(w),
            ..
        } => (format!("window_size = {w}"), "Shorten window_size", "rows"),
        ModelKind::CorrChange {
            window_size: Some(w),
            monitor_span,
            ..
        } => (
            match monitor_span {
                Some(m) => format!("window_size = {w} and monitor_span = {m}"),
                None => format!("window_size = {w}"),
            },
            "Shorten window_size, or give window_rows",
            "rows",
        ),
        ModelKind::Garch {
            start_span: Some(s),
            ..
        }
        | ModelKind::Dcc {
            start_span: Some(s),
            ..
        } => (format!("start_span = {s}"), "Shorten start_span", "rows"),
        ModelKind::Dcc {
            margin_start_span: Some(s),
            ..
        } => (
            format!("margin_start_span = {s}"),
            "Shorten margin_start_span",
            "rows",
        ),
        ModelKind::DdSketch {
            window_size: Some(w),
            ..
        } => (
            format!("window_size = {w}"),
            "Shorten window_size, or raise alpha",
            "rows and buckets",
        ),
        ModelKind::DdSketch { .. } => (
            "the range of its values".to_string(),
            "Raise alpha",
            "buckets",
        ),
        ModelKind::Isotonic {
            window_size: Some(w),
            ..
        }
        | ModelKind::Krls(crate::spec::KrlsSpec {
            window_size: Some(w),
            ..
        }) => (format!("window_size = {w}"), "Shorten window_size", "rows"),
        ModelKind::Isotonic { .. } => (
            "the feature column".to_string(),
            "Round the feature to fewer values in the query, or give window_rows",
            "distinct values",
        ),
        ModelKind::Dependence(d) if d.window_size.is_some() => (
            format!(
                "window_size = {}",
                d.window_size.as_ref().expect("checked is_some")
            ),
            "Shorten window_size, or give window_rows",
            "rows",
        ),
        ModelKind::Dependence(d) if d.bin_warm_span.is_some() => (
            format!(
                "bin_warm_span = {}",
                d.bin_warm_span.as_ref().expect("checked is_some")
            ),
            "Shorten bin_warm_span, or give bin_warm_rows",
            "rows",
        ),
        ModelKind::Expectile {
            refit_span: Some(s),
            ..
        }
        | ModelKind::Glm {
            refit_span: Some(s),
            ..
        } => (
            format!("refit_span = {s}"),
            "Shorten refit_span, or give refit_rows",
            "rows",
        ),
        ModelKind::Huber {
            huber_refit_span: Some(s),
            ..
        }
        | ModelKind::Sgd {
            huber_refit_span: Some(s),
            ..
        } => (
            format!("huber_refit_span = {s}"),
            "Shorten huber_refit_span, or give huber_refit_rows",
            "rows",
        ),
        ModelKind::Quantile {
            quantile_refit_span: Some(s),
            ..
        } => (
            format!("quantile_refit_span = {s}"),
            "Shorten quantile_refit_span, or give quantile_refit_rows",
            "rows",
        ),
        // No window on the clock: the lag buffers are what the data sizes.
        ModelKind::EwmMoments { .. } => (
            "the reach of its lags".to_string(),
            "Shorten the lags or lag_tolerance",
            "rows",
        ),
        ModelKind::Audit {
            distinct_window_size: Some(w),
            ..
        } => (
            format!("distinct_window_size = {w}"),
            "Shorten distinct_window_size or lower distinct_precision",
            "distinct-sketch entries",
        ),
        ModelKind::Audit {
            distinct_window_rows: Some(n),
            ..
        } => (
            format!("distinct_window_rows = {n}"),
            "Shorten distinct_window_rows or lower distinct_precision",
            "distinct-sketch entries",
        ),
        ModelKind::Bocpd { .. } => (
            "its live runs".to_string(),
            "Raise prune_below or lower max_run",
            "live runs",
        ),
        ModelKind::Hmm {
            warm_span: Some(s), ..
        } => (
            format!("warm_span = {s}"),
            "Shorten warm_span, or give warm_rows",
            "rows",
        ),
        _ => ("its window".to_string(), "Shorten the window", "events"),
    }
}

/// What a spec's cap is called, and the way out by raising it: a warm-up's
/// fixed limit, which no setting raises, or `max_state_bytes`.
pub(crate) fn cap_words(spec: &Spec) -> (&'static str, &'static str) {
    if let Some(f) = fixed_span(spec) {
        return (f.cap, "");
    }
    ("max_state_bytes", ", or raise max_state_bytes")
}

/// A group's state is past half of `max_state_bytes`: the model holds `bytes`
/// of events, and more arrive. Raised once a group by the first row to pass
/// it; the bank prefixes the spec and the group.
pub(super) fn state_cap_notice(spec: &Spec, bytes: u64, cap: u64) -> String {
    let (window, shorten, what) = window_words(spec);
    let (called, raise) = cap_words(spec);
    format!(
        "the state holds {bytes} bytes, past half of {called} ({cap}), and it grows with the \
         {what} inside {window}. At {cap} bytes the bank stops and refuses the run. \
         {shorten}{raise}, to stay clear of it."
    )
}

/// Clocked-embargo `emit_calibration` HAC ring is past half of `max_state_bytes`.
pub(super) fn calibration_cap_notice(spec: &Spec, bytes: u64, cap: u64) -> String {
    let emb = spec
        .embargo
        .as_ref()
        .map_or_else(|| "embargo".to_string(), |e| format!("embargo = {e}"));
    format!(
        "emit_calibration holds {bytes} bytes, past half of max_state_bytes ({cap}), and it \
         grows with the residuals inside 2 * embargo ({emb}). At {cap} bytes the bank stops and \
         refuses the run. Give horizon_rows, shorten embargo, or raise max_state_bytes, to stay \
         clear of it."
    )
}
