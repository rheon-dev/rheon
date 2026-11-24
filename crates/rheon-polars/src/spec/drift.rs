//! `drift_kind` and `drift_options`: which detector `emit_drift` keeps on
//! each slot's errors, and its settings (docs/PLAN.md task 508). Split from
//! `spec.rs` to keep it under the repository's file-size cap.
//!
//! `page_hinkley` (the default) and `cusum` read the shared `drift_delta` and
//! `drift_threshold`; the others take their own settings, by river's names,
//! in `drift_options`. Each setting given to a kind that does not read it is
//! refused, as every switch's knob is.

use rheon_core::{
    Adwin, AdwinCfg, AdwinRange, Cusum, Ddm, DriftDetector, Eddm, ErrorRate, Kswin, KswinCfg,
    PageHinkley, Refusals,
};
use serde::{Deserialize, Serialize};

use super::{Spec, num_label};

/// The detectors `drift_kind` names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DriftKind {
    PageHinkley,
    Cusum,
    Adwin,
    Ddm,
    Eddm,
    Kswin,
}

impl DriftKind {
    /// Every kind's name, in the order the docs give them.
    pub const NAMES: [&'static str; 6] = ["page_hinkley", "cusum", "adwin", "ddm", "eddm", "kswin"];

    pub fn of(name: &str) -> Option<Self> {
        Some(match name {
            "page_hinkley" => Self::PageHinkley,
            "cusum" => Self::Cusum,
            "adwin" => Self::Adwin,
            "ddm" => Self::Ddm,
            "eddm" => Self::Eddm,
            "kswin" => Self::Kswin,
            _ => return None,
        })
    }

    pub fn name(self) -> &'static str {
        Self::NAMES[self as usize]
    }

    /// Whether it reads `drift_delta` and `drift_threshold` and sums its
    /// excess over the clock.
    pub fn clocked(self) -> bool {
        matches!(self, Self::PageHinkley | Self::Cusum)
    }

    /// The `drift_options` keys it reads.
    pub fn options(self) -> &'static [&'static str] {
        match self {
            Self::PageHinkley | Self::Cusum => &[],
            Self::Adwin => &[
                "delta",
                "clock",
                "max_buckets",
                "min_window_length",
                "grace_period",
                "max_width",
                "restart",
                "max_state_bytes",
                "range",
            ],
            Self::Ddm => &["error_above", "warm_start", "drift_threshold"],
            Self::Eddm => &["error_above", "warm_start", "beta"],
            Self::Kswin => &[
                "alpha",
                "window_size",
                "stat_size",
                "seed",
                "restart",
                "max_state_bytes",
            ],
        }
    }
}

/// `drift_options`: every setting a detector other than `page_hinkley` and
/// `cusum` takes, by river's names. A key the spec's `drift_kind` does not
/// read is refused. Written only where given.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DriftOptions {
    /// ADWIN's confidence, in `(0, 1)`. Default 0.002.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub delta: Option<f64>,
    /// ADWIN's rows between scans of its window, at least 1. Default 32.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub clock: Option<u64>,
    /// ADWIN's buckets a level, from 1 to 64. Default 5.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_buckets: Option<u64>,
    /// ADWIN's fewest rows in either part of a split, at least 1. Default 5.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_window_length: Option<u64>,
    /// ADWIN's rows before its first scan. Default 10.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub grace_period: Option<u64>,
    /// ADWIN's most rows, at least 1. Default 1,000,000.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_width: Option<u64>,
    /// ADWIN's range `R` in its bound: a positive number in the residuals'
    /// units, or `"window"`, the range the window holds. Default 1, river's.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub range: Option<RangeOption>,
    /// ADWIN's and KSWIN's choice after a flag: start again empty (`true`,
    /// the default) or keep what the flag left.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub restart: Option<bool>,
    /// The bytes ADWIN's or KSWIN's detectors may hold in a group, every slot
    /// together; above 0 and at most 64 MiB. Default 256 KB.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_state_bytes: Option<u64>,
    /// DDM's and EDDM's error: a residual whose size is past this, in the
    /// target's units. Required.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error_above: Option<f64>,
    /// DDM's and EDDM's rows before a flag. Default 30.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub warm_start: Option<u64>,
    /// DDM's standard deviations past the best rate that flag. Default 3.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub drift_threshold: Option<f64>,
    /// EDDM's ratio to the largest `p' + 2s'` under which it flags, in
    /// `(0, 1]`. Default 0.9.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub beta: Option<f64>,
    /// KSWIN's test level, in `(0, 1)`. Default 0.005.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub alpha: Option<f64>,
    /// KSWIN's ring of rows, at least twice `stat_size`. Default 100.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub window_size: Option<u64>,
    /// KSWIN's rows a sample, at least 1. Default 30.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stat_size: Option<u64>,
    /// KSWIN's draw's seed. Default 0.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
}

/// `drift_options`' `range`: a number, or the name `"window"`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum RangeOption {
    Fixed(f64),
    Named(String),
}

/// The default cap on a group's ADWIN or KSWIN detectors, every slot
/// together: the per-group 256 KB every bounded state is held to.
pub const DRIFT_MAX_STATE_BYTES: u64 = 256 * 1024;

/// The most `max_state_bytes` may be set to: 64 MiB, as every model's cap.
pub const DRIFT_STATE_BYTES_CEILING: u64 = 64 * 1024 * 1024;

impl DriftOptions {
    /// The keys given, by name.
    fn given(&self) -> Vec<&'static str> {
        [
            ("delta", self.delta.is_some()),
            ("clock", self.clock.is_some()),
            ("max_buckets", self.max_buckets.is_some()),
            ("min_window_length", self.min_window_length.is_some()),
            ("grace_period", self.grace_period.is_some()),
            ("max_width", self.max_width.is_some()),
            ("range", self.range.is_some()),
            ("restart", self.restart.is_some()),
            ("max_state_bytes", self.max_state_bytes.is_some()),
            ("error_above", self.error_above.is_some()),
            ("warm_start", self.warm_start.is_some()),
            ("drift_threshold", self.drift_threshold.is_some()),
            ("beta", self.beta.is_some()),
            ("alpha", self.alpha.is_some()),
            ("window_size", self.window_size.is_some()),
            ("stat_size", self.stat_size.is_some()),
            ("seed", self.seed.is_some()),
        ]
        .into_iter()
        .filter(|(_, on)| *on)
        .map(|(k, _)| k)
        .collect()
    }

    fn adwin(&self) -> AdwinCfg {
        let d = AdwinCfg::default();
        AdwinCfg {
            delta: self.delta.unwrap_or(d.delta),
            clock: self.clock.unwrap_or(d.clock),
            max_buckets: self
                .max_buckets
                .map_or(d.max_buckets, |m| usize::try_from(m).unwrap_or(usize::MAX)),
            min_window_length: self.min_window_length.unwrap_or(d.min_window_length),
            grace_period: self.grace_period.unwrap_or(d.grace_period),
            max_width: self.max_width.unwrap_or(d.max_width),
            restart: self.restart.unwrap_or(d.restart),
            range: match &self.range {
                None => d.range,
                Some(RangeOption::Fixed(r)) => AdwinRange::Fixed(*r),
                Some(RangeOption::Named(_)) => AdwinRange::Window,
            },
        }
    }

    fn kswin(&self) -> KswinCfg {
        let d = KswinCfg::default();
        KswinCfg {
            alpha: self.alpha.unwrap_or(d.alpha),
            window_size: self
                .window_size
                .map_or(d.window_size, |w| usize::try_from(w).unwrap_or(usize::MAX)),
            stat_size: self
                .stat_size
                .map_or(d.stat_size, |r| usize::try_from(r).unwrap_or(usize::MAX)),
            seed: self.seed.unwrap_or(d.seed),
            restart: self.restart.unwrap_or(d.restart),
        }
    }
}

impl Spec {
    /// The detector `emit_drift` keeps: `drift_kind`, `page_hinkley` unless
    /// set. `None` for a name no kind has, which [`Spec::validate`] refuses.
    pub fn drift_kind_or_default(&self) -> Option<DriftKind> {
        self.drift_kind
            .as_deref()
            .map_or(Some(DriftKind::PageHinkley), DriftKind::of)
    }

    fn drift_options_or_default(&self) -> DriftOptions {
        self.drift_options.clone().unwrap_or_default()
    }

    /// One slot's detector, fresh, as the spec sets it; `None` without
    /// `emit_drift`. A spec [`Spec::validate`] passed.
    pub fn drift_detector(&self) -> Option<DriftDetector> {
        if !self.emit_drift {
            return None;
        }
        let o = self.drift_options_or_default();
        let (delta, threshold) = (
            self.drift_delta_or_default(),
            self.drift_threshold_or_default(),
        );
        Some(match self.drift_kind_or_default()? {
            DriftKind::PageHinkley => {
                DriftDetector::PageHinkley(PageHinkley::new(delta, threshold))
            }
            DriftKind::Cusum => DriftDetector::Cusum(Cusum::new(delta, threshold)),
            DriftKind::Adwin => DriftDetector::Adwin(Adwin::new(o.adwin())),
            DriftKind::Kswin => DriftDetector::Kswin(Kswin::new(o.kswin())),
            DriftKind::Ddm => DriftDetector::Ddm(ErrorRate {
                error_above: o.error_above.unwrap_or(f64::INFINITY),
                detector: Ddm::new(o.warm_start.unwrap_or(30), o.drift_threshold.unwrap_or(3.0)),
            }),
            DriftKind::Eddm => DriftDetector::Eddm(ErrorRate {
                error_above: o.error_above.unwrap_or(f64::INFINITY),
                detector: Eddm::new(o.warm_start.unwrap_or(30), o.beta.unwrap_or(0.9)),
            }),
        })
    }

    /// The bytes one slot's detector may hold, at most: ADWIN's at its
    /// `max_width`, KSWIN's ring; the rest a few numbers.
    pub fn drift_slot_bytes(&self) -> u64 {
        let o = self.drift_options_or_default();
        match self.drift_kind_or_default() {
            Some(DriftKind::Adwin) => {
                let c = o.adwin();
                Adwin::state_bytes_with(
                    c.max_width,
                    c.max_buckets,
                    matches!(c.range, AdwinRange::Window),
                )
            }
            Some(DriftKind::Kswin) => Kswin::state_bytes(o.kswin().window_size),
            _ => 72,
        }
    }

    /// `drift_kind` and `drift_options`, each problem added to `no`. The
    /// shared knobs' checks stay with the rest of `emit_drift`'s.
    pub(super) fn check_drift_kind(&self, no: &mut Refusals) {
        let name = &self.name;
        let kind = match self.drift_kind.as_deref() {
            None => DriftKind::PageHinkley,
            Some(k) => match DriftKind::of(k) {
                Some(kind) => kind,
                None => {
                    no.refuse(super::spec_words_refusal(
                        name,
                        "drift_kind",
                        &DriftKind::NAMES,
                        k,
                    ));
                    return;
                }
            },
        };
        if !self.emit_drift {
            return;
        }
        if !kind.clocked() {
            for (key, set) in [
                ("drift_delta", self.drift_delta.is_some()),
                ("drift_threshold", self.drift_threshold.is_some()),
            ] {
                if set {
                    no.refuse(format!(
                        "spec {name:?}: {key} applies to drift_kind \"page_hinkley\" and \
                         \"cusum\", but drift_kind is {:?}, whose settings are in \
                         drift_options. Leave out {key}, and give the detector's own settings \
                         in drift_options.",
                        kind.name()
                    ));
                }
            }
        }
        let o = self.drift_options_or_default();
        let foreign: Vec<&str> = o
            .given()
            .into_iter()
            .filter(|k| !kind.options().contains(k))
            .collect();
        if !foreign.is_empty() {
            let reads = if kind.options().is_empty() {
                format!(
                    "{:?} reads none, taking drift_delta and drift_threshold instead",
                    kind.name()
                )
            } else {
                format!("{:?} reads {}", kind.name(), kind.options().join(", "))
            };
            no.refuse(format!(
                "spec {name:?}: drift_options gives {}, which drift_kind {:?} does not read: \
                 {reads}. Leave out {}.",
                foreign.join(", "),
                kind.name(),
                if foreign.len() == 1 {
                    "that key"
                } else {
                    "those keys"
                }
            ));
        }
        let at_least = |no: &mut Refusals, key: &str, v: Option<u64>, least: u64, what: &str| {
            if let Some(v) = v.filter(|v| *v < least) {
                no.refuse(format!(
                    "spec {name:?}: drift_options' {key} must be at least {least}, but it is \
                     {v}. Set {key} to {what}."
                ));
            }
        };
        let open_unit = |no: &mut Refusals, key: &str, v: Option<f64>, what: &str| {
            if let Some(v) = v.filter(|v| !(*v > 0.0 && *v < 1.0)) {
                no.refuse(format!(
                    "spec {name:?}: drift_options' {key} must be above 0 and below 1, but it \
                     is {}. Set {key} to {what}.",
                    num_label(v)
                ));
            }
        };
        match kind {
            DriftKind::PageHinkley | DriftKind::Cusum => {}
            DriftKind::Adwin => {
                open_unit(
                    no,
                    "delta",
                    o.delta,
                    "the confidence of a cut, such as 0.002",
                );
                at_least(
                    no,
                    "clock",
                    o.clock,
                    1,
                    "the rows between two scans, such as 32",
                );
                at_least(
                    no,
                    "max_buckets",
                    o.max_buckets,
                    1,
                    "the buckets a level holds, such as 5",
                );
                if let Some(m) = o.max_buckets.filter(|m| *m > 64) {
                    no.refuse(format!(
                        "spec {name:?}: drift_options' max_buckets must be at most 64, but it \
                         is {m}: each level of the window holds that many buckets, and past \
                         64 a level is too long a list to stay a summary. Set max_buckets \
                         to 64 or fewer, such as 5."
                    ));
                }
                at_least(
                    no,
                    "min_window_length",
                    o.min_window_length,
                    1,
                    "the fewest rows a part of a split holds, such as 5",
                );
                at_least(
                    no,
                    "max_width",
                    o.max_width,
                    1,
                    "the most rows the window holds, such as 1000000",
                );
                match &o.range {
                    Some(RangeOption::Fixed(r)) if !(r.is_finite() && *r > 0.0) => {
                        no.refuse(format!(
                            "spec {name:?}: drift_options' range must be a finite number above \
                             0 or \"window\", but it is {}. Set range to the range the \
                             residuals' sizes lie in, in the target's units (1, river's, by \
                             default), or to \"window\" for the range the window holds, which \
                             makes the test the same in any unit.",
                            num_label(*r)
                        ));
                    }
                    Some(RangeOption::Named(n)) if n != "window" => {
                        no.refuse(format!(
                            "spec {name:?}: drift_options' range must be a finite number above \
                             0 or \"window\", but it is {n:?}. Set range to a number in the \
                             target's units, or to \"window\"."
                        ));
                    }
                    _ => {}
                }
            }
            DriftKind::Kswin => {
                open_unit(no, "alpha", o.alpha, "the test's level, such as 0.005");
                let c = o.kswin();
                at_least(
                    no,
                    "stat_size",
                    o.stat_size,
                    1,
                    "the rows each sample holds, such as 30",
                );
                if c.stat_size >= 1 && c.window_size / 2 < c.stat_size {
                    no.refuse(format!(
                        "spec {name:?}: drift_options' window_size must be at least twice \
                         stat_size ({}), but it is {}: each test draws stat_size rows from the \
                         older window_size - stat_size. Raise window_size, or lower stat_size.",
                        u128::from(c.stat_size as u64) * 2,
                        c.window_size
                    ));
                }
            }
            DriftKind::Ddm | DriftKind::Eddm => {
                match o.error_above {
                    None => no.refuse(format!(
                        "spec {name:?}: drift_kind {:?} counts errors, so drift_options must \
                         say what an error is, but error_above is not set. Set error_above to \
                         the size of residual past which a row is an error, in the target's \
                         units: 0.5 for a probability of a 0/1 target, which makes an error a \
                         misclassification.",
                        kind.name()
                    )),
                    Some(v) if !(v.is_finite() && v >= 0.0) => no.refuse(format!(
                        "spec {name:?}: drift_options' error_above must be finite and at least \
                         0, but it is {}. Set error_above to the size of residual past which a \
                         row is an error, in the target's units.",
                        num_label(v)
                    )),
                    Some(_) => {}
                }
                if let Some(v) = o.drift_threshold.filter(|v| !(v.is_finite() && *v > 1.0)) {
                    no.refuse(format!(
                        "spec {name:?}: drift_options' drift_threshold must be finite and above \
                         1, but it is {}: at 1 or under, the row that sets the best error rate \
                         is already past it, so a stream that never changes flags once a \
                         warm_start. Set drift_threshold to the standard deviations past the \
                         best error rate that flag a drift, such as 3.",
                        num_label(v)
                    ));
                }
                if let Some(v) = o.beta.filter(|v| !(*v > 0.0 && *v <= 1.0)) {
                    no.refuse(format!(
                        "spec {name:?}: drift_options' beta must be above 0 and at most 1, but \
                         it is {}. Set beta to the ratio under which a drift is flagged, such \
                         as 0.9.",
                        num_label(v)
                    ));
                }
            }
        }
        if matches!(kind, DriftKind::Adwin | DriftKind::Kswin) {
            self.check_drift_bytes(no, &o);
        }
    }

    /// ADWIN's or KSWIN's detectors, every slot of a group, against
    /// `max_state_bytes`: known here from the settings, so a group is never
    /// refused as it runs.
    fn check_drift_bytes(&self, no: &mut Refusals, o: &DriftOptions) {
        let name = &self.name;
        let cap = o.max_state_bytes.unwrap_or(DRIFT_MAX_STATE_BYTES);
        if cap == 0 || cap > DRIFT_STATE_BYTES_CEILING {
            no.refuse(format!(
                "spec {name:?}: drift_options' max_state_bytes must be above 0 and at most \
                 {DRIFT_STATE_BYTES_CEILING} (64 MiB), but it is {cap}. Give the bytes a \
                 group's drift detectors may hold, or leave it out for 256 KB."
            ));
            return;
        }
        let Ok(decays) = self.decays() else {
            return;
        };
        let slots = (decays.len() * self.targets.len() * crate::stream::combos(self).len()) as u64;
        let each = self.drift_slot_bytes();
        let bytes = slots.saturating_mul(each);
        if bytes > cap {
            let (what, lower) = match self.drift_kind_or_default() {
                Some(DriftKind::Adwin) => ("ADWIN's window at its max_width", "max_width"),
                _ => ("KSWIN's ring of window_size rows", "window_size"),
            };
            no.refuse(format!(
                "spec {name:?}: the drift detectors would hold up to {bytes} bytes in each \
                 group, past max_state_bytes of {cap}: {each} bytes for {what}, on each of \
                 {slots} slots. Lower {lower}, or raise max_state_bytes in drift_options."
            ));
        }
    }
}
