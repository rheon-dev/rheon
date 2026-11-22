//! The words for a state the data sizes (`hawkes`' window of clock units),
//! which the settings cannot bound at the call: the notice that it is past
//! half of its cap, and what the bank says when a row would pass the cap.

use super::*;

/// What a spec's window is, in its own words, and the ways to hold fewer
/// events: `window_size = 5m`, shortened.
fn window_words(spec: &Spec) -> (String, &'static str, &'static str) {
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
        } => (format!("window_size = {w}"), "Shorten window_size", "rows"),
        _ => ("its window".to_string(), "Shorten the window", "events"),
    }
}

/// A group's state is past half of `max_state_bytes`: the model holds `bytes`
/// of events, and more arrive. Raised once a group by the first row to pass
/// it; the bank prefixes the spec and the group.
pub(super) fn state_cap_notice(spec: &Spec, bytes: u64, cap: u64) -> String {
    let (window, shorten, what) = window_words(spec);
    format!(
        "the state holds {bytes} bytes, past half of max_state_bytes ({cap}), and it grows with \
         the {what} inside {window}. At {cap} bytes the bank stops and refuses the run. \
         {shorten}, or raise max_state_bytes, to stay clear of it."
    )
}
