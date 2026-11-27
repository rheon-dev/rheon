//! `ewm_moments`' terms as a spec writes them, and the model's settings read
//! from a spec (docs/PLAN.md task 554).
//!
//! A term is named by the user and maps each leg to its power:
//! `{"m3": {"x": 3}, "xy2": {"x": 1, "y": 2}, "ac1": {"x": 1, "x@1": 1}}`.
//! A leg is a feature column, or a column lagged by `k` written `x@k`: `k`
//! learned rows without a clock, clock units with one (a duration such as
//! `x@5m` on a temporal clock). The terms keep the order the spec writes
//! them in, which is the order of the outputs.

use std::fmt;

use serde::de::{self, MapAccess, Visitor};
use serde::ser::SerializeMap;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::{ModelKind, Spec, spec_words_refusal};
use crate::span::{Duration, Span};

/// The words a term's `kind` takes, the default first.
pub const MOMENT_KINDS: [&str; 4] = ["central", "raw", "standardized", "cumulant"];

/// A term's legs, each a column (or `column@lag`) and its power, in the
/// order the spec writes them.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct MomentLegs(pub Vec<(String, u32)>);

/// The terms, by name, in the order the spec writes them.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct MomentTerms(pub Vec<(String, MomentLegs)>);

/// `kind` as a spec writes it: one word for every term, or a table from a
/// term's name to its word, a term it leaves out taking `"central"`.
#[derive(Debug, Clone, PartialEq)]
pub enum MomentKinds {
    One(String),
    Each(Vec<(String, String)>),
}

/// An ordered map's entries, as a visitor reads them: serde's own maps
/// would sort them.
struct Entries<V>(std::marker::PhantomData<V>, &'static str);

impl<'de, V: Deserialize<'de>> Visitor<'de> for Entries<V> {
    type Value = Vec<(String, V)>;

    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(self.1)
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
        let mut out = Vec::new();
        while let Some((k, v)) = map.next_entry::<String, V>()? {
            out.push((k, v));
        }
        Ok(out)
    }
}

fn serialize_entries<S: Serializer, V: Serialize>(
    s: S,
    entries: &[(String, V)],
) -> Result<S::Ok, S::Error> {
    let mut m = s.serialize_map(Some(entries.len()))?;
    for (k, v) in entries {
        m.serialize_entry(k, v)?;
    }
    m.end()
}

impl Serialize for MomentLegs {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        serialize_entries(s, &self.0)
    }
}

impl<'de> Deserialize<'de> for MomentLegs {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        d.deserialize_map(Entries(
            std::marker::PhantomData,
            "a table from each leg (a column, or column@lag) to its power, a whole number \
             from 1, such as {\"x\": 1, \"x@1\": 1}",
        ))
        .map(MomentLegs)
    }
}

impl Serialize for MomentTerms {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        serialize_entries(s, &self.0)
    }
}

impl<'de> Deserialize<'de> for MomentTerms {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        d.deserialize_map(Entries(
            std::marker::PhantomData,
            "a table from each term's name to its legs, such as {\"m3\": {\"x\": 3}}",
        ))
        .map(MomentTerms)
    }
}

impl Serialize for MomentKinds {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            MomentKinds::One(k) => s.serialize_str(k),
            MomentKinds::Each(e) => serialize_entries(s, e),
        }
    }
}

struct KindsVisitor;

impl<'de> Visitor<'de> for KindsVisitor {
    type Value = MomentKinds;

    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(
            "a kind for every term (\"central\", \"raw\", \"standardized\" or \"cumulant\"), \
             or a table from a term's name to its kind",
        )
    }

    fn visit_str<E: de::Error>(self, v: &str) -> Result<MomentKinds, E> {
        Ok(MomentKinds::One(v.to_string()))
    }

    fn visit_map<A: MapAccess<'de>>(self, map: A) -> Result<MomentKinds, A::Error> {
        Entries::<String>(std::marker::PhantomData, "")
            .visit_map(map)
            .map(MomentKinds::Each)
    }
}

impl<'de> Deserialize<'de> for MomentKinds {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        d.deserialize_any(KindsVisitor)
    }
}

/// A leg as written, `x` or `x@k`, read against the spec's features: the
/// feature's place and the lag (`None` unlagged). A key that is a feature's
/// name is that feature unlagged, so a column whose name holds `@` is read
/// whole; otherwise the text after the last `@` is the lag.
pub(crate) fn parse_leg(features: &[String], key: &str) -> Result<(usize, Option<Span>), String> {
    if let Some(i) = features.iter().position(|f| f == key) {
        return Ok((i, None));
    }
    let unknown = |col: &str| {
        format!(
            "the leg {key:?} names the column {col:?}, which is not among the features ({}). \
             Name a feature, or add {col:?} to features.",
            features
                .iter()
                .map(|f| format!("{f:?}"))
                .collect::<Vec<_>>()
                .join(", ")
        )
    };
    let Some((col, lag)) = key.rsplit_once('@') else {
        return Err(unknown(key));
    };
    let Some(i) = features.iter().position(|f| f == col) else {
        return Err(unknown(col));
    };
    let span = match lag.trim().parse::<f64>() {
        Ok(v) => Span::Units(v),
        Err(_) => Span::Duration(Duration::parse(lag).map_err(|e| {
            format!(
                "the leg {key:?} lags {col:?} by {lag:?}, which is neither a number nor a \
                 duration: {e}"
            )
        })?),
    };
    Ok((i, Some(span)))
}

/// The column a leg reads, as written: the key itself where it is a
/// feature's name, otherwise the text before the last `@`.
pub(crate) fn moment_leg_column(features: &[String], key: &str) -> String {
    if features.iter().any(|f| f == key) {
        return key.to_string();
    }
    match key.rsplit_once('@') {
        Some((col, _)) => col.to_string(),
        None => key.to_string(),
    }
}

impl ModelKind {
    /// Every lag the terms of an `ewm_moments` spec write, as a span of the
    /// clock (a number of rows without one); empty for any other kind, and
    /// for a leg that names no feature, which the model's check refuses.
    pub fn moment_lags(&self, features: &[String]) -> Vec<Span> {
        let ModelKind::EwmMoments { terms, .. } = self else {
            return Vec::new();
        };
        terms
            .0
            .iter()
            .flat_map(|(_, legs)| legs.0.iter())
            .filter_map(|(key, _)| parse_leg(features, key).ok().and_then(|(_, lag)| lag))
            .collect()
    }
}

impl Spec {
    /// Every column the terms of an `ewm_moments` spec read, in the order
    /// they first name them, lags taken off: the features such a spec
    /// fills in when it names none.
    pub(crate) fn moment_columns(&self) -> Vec<String> {
        let ModelKind::EwmMoments { terms, .. } = &self.model else {
            return Vec::new();
        };
        let mut out: Vec<String> = Vec::new();
        for (key, _) in terms.0.iter().flat_map(|(_, legs)| legs.0.iter()) {
            let col = moment_leg_column(&[], key);
            if !out.contains(&col) {
                out.push(col);
            }
        }
        out
    }
}

/// An `ewm_moments` spec's [`rheon_core::EwmMomentsCfg`] at `decay`: the
/// terms read against the features, each term's kind and bias, the clock,
/// the window and the state's cap. The settings the model checks are its
/// own to refuse ([`rheon_core::EwmMomentsCfg::validate`]); this refuses
/// what it cannot read: a leg that names no feature, a kind that is no
/// kind, a window or a tolerance in the wrong form.
pub fn ewm_moments_cfg(
    spec: &Spec,
    decay: rheon_core::Decay,
) -> Result<rheon_core::EwmMomentsCfg, String> {
    use rheon_core::{MomentKind, MomentLeg, MomentTerm};
    let ModelKind::EwmMoments {
        terms,
        kind,
        bias,
        se,
        lag_tolerance,
        window_rows,
        window_size,
        max_state_bytes,
    } = &spec.model
    else {
        return Err(format!(
            "ewm_moments_cfg: spec {:?} is a {} spec, not an ewm_moments spec. Pass \
             ewm_moments_cfg an ewm_moments spec.",
            spec.name,
            spec.model.kind_name()
        ));
    };
    let mut no = rheon_core::Refusals::new();
    let word = |w: &str| -> Result<MomentKind, String> {
        Ok(match w {
            "central" => MomentKind::Central,
            "raw" => MomentKind::Raw,
            "standardized" => MomentKind::Standardized,
            "cumulant" => MomentKind::Cumulant,
            _ => {
                return Err(spec_words_refusal(&spec.name, "kind", &MOMENT_KINDS, w)
                    .trim_start_matches(&format!("spec {:?}: ", spec.name))
                    .to_string());
            }
        })
    };
    let one = match kind {
        Some(MomentKinds::One(w)) => no.value(word(w)).unwrap_or(MomentKind::Central),
        _ => MomentKind::Central,
    };
    if let Some(MomentKinds::Each(each)) = kind {
        for (name, _) in each {
            if !terms.0.iter().any(|(t, _)| t == name) {
                no.refuse(format!(
                    "spec {:?}: kind names the term {name:?}, which terms does not have. Name a \
                     term of terms, or leave {name:?} out of kind.",
                    spec.name
                ));
            }
        }
    }
    if window_rows.is_some() && window_size.is_some() {
        no.refuse(format!(
            "spec {:?}: window_rows (a count of rows) and window_size (a span of the clock) both \
             set the window. Give one of them.",
            spec.name
        ));
    }
    let clocked = spec.clock.is_some();
    if let Some(w) = window_size {
        if !clocked {
            no.refuse(format!(
                "spec {:?}: window_size = {w} is a span of the clock, but the spec has no clock. \
                 Give window_rows for a window of learned rows, or set clock.",
                spec.name
            ));
        } else if !(w.value().is_finite() && w.value() > 0.0) {
            no.refuse(format!(
                "spec {:?}: window_size must be a finite span of the clock above 0, but it is \
                 {w}. Give the span the moments keep rows of, or leave it out for no window.",
                spec.name
            ));
        }
    }
    let mut out = Vec::with_capacity(terms.0.len());
    for (name, legs) in &terms.0 {
        let k = match kind {
            Some(MomentKinds::Each(each)) => match each.iter().find(|(t, _)| t == name) {
                Some((_, w)) => match word(w) {
                    Ok(k) => k,
                    Err(_) => {
                        let param = format!("the term {name:?}'s kind");
                        no.refuse(spec_words_refusal(&spec.name, &param, &MOMENT_KINDS, w));
                        MomentKind::Central
                    }
                },
                None => MomentKind::Central,
            },
            _ => one,
        };
        let mut powers = Vec::with_capacity(legs.0.len());
        for (key, p) in &legs.0 {
            match parse_leg(&spec.features, key) {
                Ok((col, lag)) => powers.push((
                    lag.map_or(MomentLeg::of(col), |l| MomentLeg::lagged(col, l.value())),
                    *p,
                )),
                Err(e) => no.refuse(format!("the term {name:?}: {e}")),
            }
        }
        let order = powers.iter().map(|&(_, p)| p).sum();
        out.push(MomentTerm {
            name: name.clone(),
            powers,
            kind: k,
            bias: bias.unwrap_or_else(|| MomentTerm::default_bias(k, order)),
        });
    }
    no.finish()?;
    Ok(rheon_core::EwmMomentsCfg {
        decay,
        terms: out,
        min_weight: spec.min_periods_or_default(),
        max_state_bytes: spec.model_max_state_bytes(*max_state_bytes),
        clocked,
        lag_tolerance: lag_tolerance.as_ref().map(Span::value),
        se: se.unwrap_or(false),
        window_rows: *window_rows,
        window_clock: window_size.as_ref().map(Span::value),
    })
}
