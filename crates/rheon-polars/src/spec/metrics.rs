//! Streaming evaluation metrics and loss score configuration (`emit_metrics`,
//! `metric_quantiles`). Split from `spec.rs` to keep it under the repository's
//! file-size cap.

use serde::{Deserialize, Serialize};
use std::fmt;

use super::{ModelKind, Spec, num_label, words_body};

/// Every streaming metric name accepted inside `emit_metrics = [...]`.
pub const METRIC_KINDS: [&str; 9] = [
    "ic", "r2", "hit_rate", "mae", "pinball", "log_loss", "coverage", "width", "winkler",
];

/// One streaming evaluation metric requested via `emit_metrics`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MetricKind {
    Ic,
    R2,
    HitRate,
    Mae,
    Pinball,
    LogLoss,
    Coverage,
    Width,
    Winkler,
}

impl MetricKind {
    /// Canonical string spelling of this metric.
    pub const fn name(self) -> &'static str {
        match self {
            MetricKind::Ic => "ic",
            MetricKind::R2 => "r2",
            MetricKind::HitRate => "hit_rate",
            MetricKind::Mae => "mae",
            MetricKind::Pinball => "pinball",
            MetricKind::LogLoss => "log_loss",
            MetricKind::Coverage => "coverage",
            MetricKind::Width => "width",
            MetricKind::Winkler => "winkler",
        }
    }

    fn parse(s: &str) -> Option<Self> {
        match s {
            "ic" => Some(MetricKind::Ic),
            "r2" => Some(MetricKind::R2),
            "hit_rate" => Some(MetricKind::HitRate),
            "mae" => Some(MetricKind::Mae),
            "pinball" => Some(MetricKind::Pinball),
            "log_loss" => Some(MetricKind::LogLoss),
            "coverage" => Some(MetricKind::Coverage),
            "width" => Some(MetricKind::Width),
            "winkler" => Some(MetricKind::Winkler),
            _ => None,
        }
    }
}

impl<'de> Deserialize<'de> for MetricKind {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct V;
        impl serde::de::Visitor<'_> for V {
            type Value = MetricKind;
            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("a metric name")
            }
            fn visit_str<E: serde::de::Error>(self, v: &str) -> Result<MetricKind, E> {
                MetricKind::parse(v).ok_or_else(|| E::custom(words_body(&METRIC_KINDS, v)))
            }
        }
        d.deserialize_any(V)
    }
}

/// `emit_metrics` configuration: either a legacy boolean (`false` = none,
/// `true` = `["ic", "r2", "hit_rate"]`) or an explicit list of metric names.
///
/// Serializes `Bool(b)` as a plain JSON/msgpack boolean so every existing spec
/// and saved state file keeps its exact bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EmitMetrics {
    Bool(bool),
    List(Vec<MetricKind>),
}

impl Default for EmitMetrics {
    fn default() -> Self {
        EmitMetrics::Bool(false)
    }
}

impl Serialize for EmitMetrics {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            EmitMetrics::Bool(b) => s.serialize_bool(*b),
            EmitMetrics::List(v) => v.serialize(s),
        }
    }
}

impl<'de> Deserialize<'de> for EmitMetrics {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct V;
        impl<'de> serde::de::Visitor<'de> for V {
            type Value = EmitMetrics;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("a boolean or a list of metric names")
            }

            fn visit_bool<E: serde::de::Error>(self, v: bool) -> Result<EmitMetrics, E> {
                Ok(EmitMetrics::Bool(v))
            }

            fn visit_seq<A: serde::de::SeqAccess<'de>>(
                self,
                mut seq: A,
            ) -> Result<EmitMetrics, A::Error> {
                let mut out = Vec::new();
                while let Some(s) = seq.next_element::<String>()? {
                    let kind = MetricKind::parse(&s)
                        .ok_or_else(|| serde::de::Error::custom(words_body(&METRIC_KINDS, &s)))?;
                    out.push(kind);
                }
                Ok(EmitMetrics::List(out))
            }
        }
        d.deserialize_any(V)
    }
}

impl EmitMetrics {
    /// True if any streaming metric is requested (including an invalid empty
    /// list only as false; validation refuses empty lists explicitly).
    pub fn is_any(&self) -> bool {
        match self {
            EmitMetrics::Bool(b) => *b,
            EmitMetrics::List(v) => !v.is_empty(),
        }
    }

    /// True if this metric kind is enabled.
    pub fn has(&self, k: MetricKind) -> bool {
        match self {
            EmitMetrics::Bool(false) => false,
            EmitMetrics::Bool(true) => {
                matches!(k, MetricKind::Ic | MetricKind::R2 | MetricKind::HitRate)
            }
            EmitMetrics::List(v) => v.contains(&k),
        }
    }

    /// True when any of the legacy trio (`ic`, `r2`, `hit_rate`) is active.
    pub fn has_legacy(&self) -> bool {
        self.has(MetricKind::Ic) || self.has(MetricKind::R2) || self.has(MetricKind::HitRate)
    }

    /// True when any point-loss score (`mae`, `pinball`, `log_loss`) is active.
    pub fn has_scores(&self) -> bool {
        self.has(MetricKind::Mae) || self.has(MetricKind::Pinball) || self.has(MetricKind::LogLoss)
    }

    /// True when any interval-quality score (`width`, `winkler`) is active.
    pub fn has_interval_scores(&self) -> bool {
        self.has(MetricKind::Width) || self.has(MetricKind::Winkler)
    }
}

impl Spec {
    /// True for binary probability models where streaming `log_loss` is valid:
    /// `sgd(loss="logistic")`, `ftrl(loss="logistic")` (its default), and
    /// `glm(family="binomial")`.
    pub fn is_binary_logistic(&self) -> bool {
        match &self.model {
            ModelKind::Sgd { loss, .. } => loss.as_deref() == Some("logistic"),
            ModelKind::Ftrl { loss, .. } => loss.as_deref().unwrap_or("logistic") == "logistic",
            ModelKind::Glm { family, .. } => family.as_deref() == Some("binomial"),
            _ => false,
        }
    }

    /// The model's own quantile level, if this is a quantile regression model
    /// (`quantile`, or `sgd(loss="quantile")`).
    pub fn model_quantile(&self) -> Option<f64> {
        match &self.model {
            ModelKind::Quantile { quantile, .. } => Some(*quantile),
            ModelKind::Sgd { loss, quantile, .. } if loss.as_deref() == Some("quantile") => {
                Some(quantile.unwrap_or(0.5))
            }
            _ => None,
        }
    }

    /// Quantile levels scored by streaming `pinball` (empty when `pinball` is
    /// not active).
    pub fn pinball_levels(&self) -> Vec<f64> {
        if !self.emit_metrics.has(MetricKind::Pinball) {
            return Vec::new();
        }
        if let Some(qs) = &self.metric_quantiles {
            return qs.clone();
        }
        self.model_quantile().into_iter().collect()
    }

    /// Whether pinball fields carry explicit `_q<level>` suffixes (true when
    /// `metric_quantiles` is given, false when defaulting to a quantile model's
    /// single level).
    pub fn pinball_is_explicit(&self) -> bool {
        self.metric_quantiles.is_some()
    }

    /// Number of per-slot blocks in `ChunkOut::scores` (`mae`, each pinball
    /// level, `log_loss`).
    pub fn score_block_count(&self) -> usize {
        usize::from(self.emit_metrics.has(MetricKind::Mae))
            + self.pinball_levels().len()
            + usize::from(self.emit_metrics.has(MetricKind::LogLoss))
    }

    /// Validate `emit_metrics` and `metric_quantiles`.
    pub(super) fn validate_metrics(&self, no: &mut rheon_core::Refusals) {
        let name = &self.name;
        if let EmitMetrics::List(list) = &self.emit_metrics {
            if list.is_empty() {
                no.refuse(format!(
                    "spec {name:?}: emit_metrics cannot be an empty list. Give one or more \
                     metric names ({}), or set emit_metrics=false.",
                    METRIC_KINDS.join(", ")
                ));
            }
            let mut seen = Vec::with_capacity(list.len());
            for &m in list {
                if seen.contains(&m) {
                    no.refuse(format!(
                        "spec {name:?}: emit_metrics lists {:?} more than once. List each metric \
                         name at most once.",
                        m.name()
                    ));
                } else {
                    seen.push(m);
                }
            }
        }
        if self.metric_quantiles.is_some() && !self.emit_metrics.has(MetricKind::Pinball) {
            no.refuse(format!(
                "spec {name:?}: metric_quantiles applies only when \"pinball\" is in \
                 emit_metrics, but it is not. Add \"pinball\" to emit_metrics, or leave out \
                 metric_quantiles."
            ));
        }
        if let Some(qs) = &self.metric_quantiles {
            if qs.is_empty() {
                no.refuse(format!(
                    "spec {name:?}: metric_quantiles cannot be empty. List one or more quantile \
                     levels above 0 and below 1 (such as [0.1, 0.5, 0.9]), or leave out \
                     metric_quantiles."
                ));
            }
            let bad: Vec<String> = qs
                .iter()
                .filter(|q| !(**q > 0.0 && **q < 1.0))
                .map(|q| num_label(*q))
                .collect();
            if !bad.is_empty() {
                no.refuse(format!(
                    "spec {name:?}: metric_quantiles must each be above 0 and below 1, but {} {}. \
                     Set each level above 0 and below 1.",
                    if bad.len() == 1 {
                        "one of them is"
                    } else {
                        "these are not:"
                    },
                    bad.join(", ")
                ));
            }
            let mut seen_bits: Vec<u64> = Vec::with_capacity(qs.len());
            for &q in qs {
                if q > 0.0 && q < 1.0 {
                    let b = q.to_bits();
                    if seen_bits.contains(&b) {
                        no.refuse(format!(
                            "spec {name:?}: metric_quantiles lists {} more than once. List each \
                             quantile level at most once.",
                            num_label(q)
                        ));
                        break;
                    }
                    seen_bits.push(b);
                }
            }
        }
        if !self.model.predicts_no_target() {
            if self.emit_metrics.has(MetricKind::Pinball)
                && self.metric_quantiles.is_none()
                && self.model_quantile().is_none()
            {
                no.refuse(format!(
                    "spec {name:?}: emit_metrics includes \"pinball\", but {} has no quantile \
                     level of its own. Set metric_quantiles (such as [0.1, 0.5, 0.9]), or use \
                     quantile or sgd(loss=\"quantile\").",
                    self.model.kind_name()
                ));
            }
            if self.emit_metrics.has(MetricKind::LogLoss) && !self.is_binary_logistic() {
                no.refuse(format!(
                    "spec {name:?}: emit_metrics includes \"log_loss\", which applies only to \
                     binary probability models (sgd or ftrl with loss=\"logistic\", or glm with \
                     family=\"binomial\"), not {}. Remove \"log_loss\" from emit_metrics.",
                    self.model.kind_name()
                ));
            }
            let interval_asked: Vec<&str> =
                [MetricKind::Coverage, MetricKind::Width, MetricKind::Winkler]
                    .into_iter()
                    .filter(|&k| self.emit_metrics.has(k))
                    .map(MetricKind::name)
                    .collect();
            if !interval_asked.is_empty() && self.conformal.is_none() {
                let listed = interval_asked.join(", ");
                no.refuse(format!(
                    "spec {name:?}: emit_metrics includes {listed}, which requires conformal to \
                     be set. Set conformal (such as 0.9 or [0.8, 0.9]), or remove {listed} from \
                     emit_metrics."
                ));
            }
        }
    }
}
