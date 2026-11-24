//! The notice `dcc` raises once when a GARCH margin's Student-t degrees of
//! freedom are found at or below 4 (`rheon_core::Dcc::take_heavy_tail`).

use super::*;

/// The words: which target's margin, its `ν`, and what follows from it.
pub(super) fn heavy_tail_notice(spec: &Spec, target: usize, nu: f64) -> String {
    let name = spec.targets.get(target).map_or("", String::as_str);
    format!(
        "the GARCH margin of target {name:?} has Student-t degrees of freedom nu = {nu:.4}, at \
         most 4, so its standardized residuals have no fourth moment. Then z z' has no \
         variance: the correlation's recursion is not covariance stationary, and the \
         estimate of a, b (and g) has no standard error, though it still runs. Check the \
         series for outliers or a level shift, or model it with dist = \"normal\" for a \
         quasi-likelihood margin."
    )
}
