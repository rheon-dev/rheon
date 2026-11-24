//! `kalman`'s fixed-lag smoother on the spec: `smooth_lag`,
//! `emit_smoothed_se` and the bytes it may hold (`max_state_bytes`), and
//! their refusals.
//!
//! The smoother emits, on the row that completes a lag of `L` observations
//! of a target, the estimate of the coefficients at the observation `L`
//! before it given every row up to this one (`rheon_core::Kalman`'s
//! `kalman/smooth.rs` states the recursion). It is never fed back: every
//! prediction is the filter's. Its state is a ring of `L` slots a
//! covariance, each of `k²` numbers and more, so its size is known from the
//! settings and is held to `max_state_bytes` before a row is read, as every
//! model whose settings fix its state is.

use super::{ModelKind, Spec};

/// The default bytes a group's smoother may hold, as every model's cap.
pub const SMOOTH_MAX_STATE_BYTES: u64 = 256 * 1024;

/// The most `max_state_bytes` may be set to: 64 MiB, as every model's cap.
pub const SMOOTH_STATE_BYTES_CEILING: u64 = 64 * 1024 * 1024;

impl Spec {
    /// `kalman`'s `smooth_lag`, in observations; `None` for no smoother and
    /// for every other model.
    pub fn smooth_lag(&self) -> Option<usize> {
        match &self.model {
            ModelKind::Kalman {
                smooth_lag: Some(l),
                ..
            } => Some(usize::try_from(*l).unwrap_or(usize::MAX)),
            _ => None,
        }
    }

    /// Whether `kalman`'s smoother emits its standard errors.
    pub fn emit_smoothed_se(&self) -> bool {
        matches!(
            self.model,
            ModelKind::Kalman {
                emit_smoothed_se: true,
                ..
            }
        )
    }

    /// The bytes a group's smoothers take, one per instance, at most
    /// ([`rheon_core::Kalman::smoother_bytes`]); 0 without a smoother.
    pub fn smoother_bytes(&self) -> u64 {
        let (
            Some(lag),
            ModelKind::Kalman {
                share_p,
                standardize,
                ..
            },
        ) = (self.smooth_lag(), &self.model)
        else {
            return 0;
        };
        let k = self.k() + usize::from(self.fit_intercept);
        let one = rheon_core::Kalman::smoother_bytes(
            k,
            self.m(),
            *share_p,
            *standardize,
            lag,
            self.emit_smoothed_se(),
        );
        let instances = self.decays().map_or(1, |d| d.len()) as u64;
        one.saturating_mul(instances)
    }

    /// The smoother's settings: a lag of at least 1, the covariance form,
    /// `emit_smoothed_se` and `max_state_bytes` only beside a lag, the cap
    /// within its range, and the ring's bytes within the cap.
    pub(super) fn check_smoother(&self) -> Result<(), String> {
        let ModelKind::Kalman {
            smooth_lag,
            emit_smoothed_se,
            max_state_bytes,
            covariance_form,
            ..
        } = &self.model
        else {
            return Ok(());
        };
        let name = &self.name;
        let mut no = rheon_core::Refusals::new();
        if *smooth_lag == Some(0) {
            no.refuse(format!(
                "spec {name:?}: smooth_lag must be at least 1 observation, but it is 0. Set it \
                 to the number of observations after a row to smooth it with, or leave it out \
                 for no smoother."
            ));
        }
        if smooth_lag.is_none() {
            for (key, set) in [
                ("emit_smoothed_se", *emit_smoothed_se),
                ("max_state_bytes", max_state_bytes.is_some()),
            ] {
                if set {
                    no.refuse(format!(
                        "spec {name:?}: {key} is set, but smooth_lag is not, and {key} is \
                         about the smoother alone. Set smooth_lag, or leave out {key}."
                    ));
                }
            }
            return no.finish();
        }
        // A held row is learned on the row whose clock releases it, and one
        // row can release several, each of which can complete a lag: the
        // row's line has room for one value a target, so the others would
        // be lost.
        for (key, set) in [
            ("embargo", self.embargo.is_some()),
            ("label_clock", self.label_clock.is_some()),
        ] {
            if set {
                no.refuse(format!(
                    "spec {name:?}: smooth_lag cannot be used with {key}: {key} holds each \
                     row until a later row releases it, one row can release several, and the \
                     smoothed value of each that completes its lag would need the same line. \
                     Leave out {key}, or leave out smooth_lag."
                ));
            }
        }
        if !covariance_form.is_covariance() {
            no.refuse(format!(
                "spec {name:?}: smooth_lag needs covariance_form=\"covariance\", but it is \
                 \"square_root\": the smoother reads the covariance itself, which the square \
                 root's form does not hold. Leave covariance_form at its default to smooth, \
                 or leave out smooth_lag."
            ));
        }
        let cap = max_state_bytes.unwrap_or(SMOOTH_MAX_STATE_BYTES);
        if cap == 0 || cap > SMOOTH_STATE_BYTES_CEILING {
            no.refuse(format!(
                "spec {name:?}: max_state_bytes must be above 0 and at most \
                 {SMOOTH_STATE_BYTES_CEILING} (64 MiB), but it is {cap}. Give the bytes a \
                 group's smoother may hold, or leave it out for 262144 (256 KB)."
            ));
        } else if let Some(lag) = smooth_lag.filter(|l| *l > 0) {
            let bytes = self.smoother_bytes();
            if bytes > cap {
                let k = self.k() + usize::from(self.fit_intercept);
                no.refuse(format!(
                    "spec {name:?}: smooth_lag = {lag} holds up to {bytes} bytes a group, past \
                     max_state_bytes of {cap}: {lag} slots of {k} coefficients' gains for each \
                     of the model's covariances and instances. Shorten smooth_lag, or raise \
                     max_state_bytes (at most 64 MiB)."
                ));
            }
        }
        no.finish()
    }
}
