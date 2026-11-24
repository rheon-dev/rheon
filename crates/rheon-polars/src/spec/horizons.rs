//! The forecast horizons a spec's `forecast_spans` and `forecast_rows` are written in: spans of the
//! clock for `holt` and `kalman` (a span, a duration, or a `[name, span]`
//! pair, one or a list of them), and counts of rows for `garch` (a count or
//! a `[name, count]` pair), each with the label its fields carry. Split from
//! `spec.rs` to keep it under the repository's file-size cap.

use serde::{Deserialize, Serialize};

use super::num_label;
use crate::span::Span;

/// One forecast horizon on the clock (`holt`'s `forecast_spans`): a span, its
/// fields labelled `h<number>` or by the duration as written, or a `[name,
/// span]` pair labelled by the name.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(untagged)]
pub enum SpanHorizon {
    Span(Span),
    Named(String, Span),
}

impl<'de> Deserialize<'de> for SpanHorizon {
    /// A span as [`Span`] reads one (its errors, a word that is no duration
    /// named as such), or a `[name, span]` pair.
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct V;
        impl<'de> serde::de::Visitor<'de> for V {
            type Value = SpanHorizon;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a span of the clock, or a [name, span] pair")
            }
            fn visit_f64<E: serde::de::Error>(self, v: f64) -> Result<SpanHorizon, E> {
                Ok(SpanHorizon::Span(Span::Units(v)))
            }
            fn visit_i64<E: serde::de::Error>(self, v: i64) -> Result<SpanHorizon, E> {
                Ok(SpanHorizon::Span(Span::Units(v as f64)))
            }
            fn visit_u64<E: serde::de::Error>(self, v: u64) -> Result<SpanHorizon, E> {
                Ok(SpanHorizon::Span(Span::Units(v as f64)))
            }
            fn visit_str<E: serde::de::Error>(self, v: &str) -> Result<SpanHorizon, E> {
                use serde::de::IntoDeserializer;
                let d: serde::de::value::StrDeserializer<'_, E> = v.into_deserializer();
                Span::deserialize(d).map(SpanHorizon::Span)
            }
            fn visit_seq<A: serde::de::SeqAccess<'de>>(
                self,
                mut seq: A,
            ) -> Result<SpanHorizon, A::Error> {
                use serde::de::Error;
                let name: String = seq
                    .next_element()?
                    .ok_or_else(|| A::Error::invalid_length(0, &"a [name, span] pair"))?;
                let span: Span = seq
                    .next_element()?
                    .ok_or_else(|| A::Error::invalid_length(1, &"a [name, span] pair"))?;
                Ok(SpanHorizon::Named(name, span))
            }
        }
        d.deserialize_any(V)
    }
}

/// `forecast_spans` read as one span or a list of them (a span or a `[name,
/// span]` pair each), as `kernel_half_lives` reads one or a list.
pub(super) fn one_or_more_horizons<'de, D: serde::Deserializer<'de>>(
    d: D,
) -> Result<Option<Vec<SpanHorizon>>, D::Error> {
    struct V;
    impl<'de> serde::de::Visitor<'de> for V {
        type Value = Vec<SpanHorizon>;
        fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            f.write_str("a span of the clock, or a list of spans or [name, span] pairs")
        }
        fn visit_f64<E: serde::de::Error>(self, v: f64) -> Result<Self::Value, E> {
            Ok(vec![SpanHorizon::Span(Span::Units(v))])
        }
        fn visit_i64<E: serde::de::Error>(self, v: i64) -> Result<Self::Value, E> {
            Ok(vec![SpanHorizon::Span(Span::Units(v as f64))])
        }
        fn visit_u64<E: serde::de::Error>(self, v: u64) -> Result<Self::Value, E> {
            Ok(vec![SpanHorizon::Span(Span::Units(v as f64))])
        }
        fn visit_str<E: serde::de::Error>(self, v: &str) -> Result<Self::Value, E> {
            use serde::de::IntoDeserializer;
            let d: serde::de::value::StrDeserializer<'_, E> = v.into_deserializer();
            Span::deserialize(d).map(|s| vec![SpanHorizon::Span(s)])
        }
        fn visit_seq<A: serde::de::SeqAccess<'de>>(
            self,
            mut seq: A,
        ) -> Result<Self::Value, A::Error> {
            let mut out = Vec::new();
            while let Some(h) = seq.next_element::<SpanHorizon>()? {
                out.push(h);
            }
            Ok(out)
        }
    }
    d.deserialize_any(V).map(Some)
}

impl SpanHorizon {
    pub fn span(&self) -> &Span {
        match self {
            SpanHorizon::Span(s) | SpanHorizon::Named(_, s) => s,
        }
    }

    /// The label its fields carry: `h<number>`, the duration as written,
    /// or its name.
    pub fn label(&self) -> String {
        match self {
            SpanHorizon::Span(Span::Units(v)) => format!("h{}", num_label(*v)),
            SpanHorizon::Span(s) => s.label(),
            SpanHorizon::Named(name, _) => name.clone(),
        }
    }
}

/// The labels of a list of horizons checked: each non-empty, none beginning
/// with `sum` (which `<quantity>_sum_<label>` would confuse) and none given
/// twice. `who` names the model in the message.
pub fn check_horizon_labels(who: &str, labels: &[String]) -> Result<(), String> {
    for (i, l) in labels.iter().enumerate() {
        if l.is_empty() || l.starts_with("sum") {
            return Err(format!(
                "{who}: a horizon's name must be non-empty and not begin with \"sum\", but one \
                 is {l:?}. It names a field and the sum's field beside it, which a name \
                 beginning with \"sum\" would confuse. Rename it."
            ));
        }
        if labels[..i].contains(l) {
            return Err(format!(
                "{who}: two horizons are labeled {l:?}. Each labels its fields, so each label is \
                 given once. Rename one, or leave out the second."
            ));
        }
    }
    Ok(())
}

/// One forecast horizon of a spec (`garch`'s `forecast_rows`): a count of rows,
/// its fields labelled `h<count>`, or a `[name, count]` pair, labelled by
/// the name (a Python dict's item).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Horizon {
    Rows(usize),
    Named(String, usize),
}

impl Horizon {
    /// The rows it counts, the row read included.
    pub fn rows(&self) -> usize {
        match self {
            Horizon::Rows(n) | Horizon::Named(_, n) => *n,
        }
    }

    /// The label its fields carry: `h<count>`, or its name.
    pub fn label(&self) -> String {
        match self {
            Horizon::Rows(n) => format!("h{n}"),
            Horizon::Named(name, _) => name.clone(),
        }
    }
}
