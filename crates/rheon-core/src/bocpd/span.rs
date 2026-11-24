//! `bocpd`'s warm-up over a span of the clock (docs/PLAN.md task 538): the
//! learned rows from the first through the first at least `warm_clock`
//! clock units after it, once they are at least two
//! ([`crate::clock_span::start_due`]), held to [`super::WARM_BUDGET_MIB`] as
//! they come ([`crate::clock_span::HeldCap`]). Out of `bocpd.rs`, which
//! stands at the repository's size cap for a source file.

use super::{Bocpd, BocpdCfg, WARM_BUDGET_MIB};
use crate::clock_span::HeldCap;

impl BocpdCfg {
    /// The span's own refusals: not finite or not above 0, beside both
    /// priors, beside `warm_rows`.
    pub(super) fn check_warm_clock(&self, no: &mut crate::Refusals) {
        if let Some(d) = self.warm_clock {
            if !(d.is_finite() && d > 0.0) {
                no.refuse(format!(
                    "bocpd: warm_span must be a finite number of clock units above 0, but it \
                 is {d}. Give the span of clock whose rows set the prior, or leave it out \
                 for warm_rows."
                ));
            }
            if !self.warms_up() {
                no.refuse(
                    "bocpd: warm_span sets the prior from the first rows, but mean_prior and \
                 covariance_prior are both given, so it has nothing to set. Leave out \
                 warm_span, or one of the two priors.",
                );
            }
            if self.warm_rows.is_some() {
                no.refuse(
                    "bocpd: warm_span and warm_rows are two warm-ups, and the prior is set \
                 once. Leave one of them out.",
                );
            }
        }
    }
}

impl Bocpd {
    /// The bytes of the warm-up's rows at `rows` rows, `(d + 4) · 8` a row,
    /// as the limit counts them.
    pub(super) fn warm_bytes(&self, rows: usize) -> u64 {
        rows as u64 * (self.cfg.n_features as u64 + 4) * 8
    }

    /// The warm-up's cap in bytes ([`WARM_BUDGET_MIB`]).
    pub(super) fn warm_cap(&self) -> u64 {
        self.cap_for_test
            .unwrap_or((WARM_BUDGET_MIB * f64::from(1 << 20)) as u64)
    }

    /// Hold the warm-up to `bytes` in place of its cap, for a test.
    #[cfg(test)]
    pub(crate) fn set_cap_for_test(&mut self, bytes: u64) {
        self.cap_for_test = Some(bytes);
    }

    pub(super) fn held_cap(&mut self) -> HeldCap<'_> {
        HeldCap {
            over: &mut self.over,
            warned: &mut self.warned,
            warning: &mut self.warning,
        }
    }
}
