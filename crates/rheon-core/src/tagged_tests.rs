use super::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, VecDeque};

fn to_json<T: Serialize>(v: &T) -> String {
    let mut out = Vec::new();
    v.serialize(Ser(&mut serde_json::Serializer::new(&mut out)))
        .unwrap();
    String::from_utf8(out).unwrap()
}

fn from_json<T: for<'de> Deserialize<'de>>(s: &str) -> Result<T, serde_json::Error> {
    T::deserialize(De(&mut serde_json::Deserializer::from_str(s)))
}

fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

#[derive(Debug, Serialize, Deserialize)]
enum Shape {
    Unit,
    New(f64),
    Tuple(f64, f32),
    Fields { a: f64, b: Vec<f64> },
}

/// Every place a float can sit that a state uses: a field, a list, a ring,
/// an option, a tuple, a map's value, each kind of enum variant, a newtype.
#[derive(Debug, Serialize, Deserialize)]
struct State {
    x: f64,
    xs: Vec<f64>,
    ring: VecDeque<f64>,
    opt: Option<f64>,
    none: Option<f64>,
    pair: (f64, u32),
    nested: Vec<Vec<f64>>,
    map: BTreeMap<String, f64>,
    shapes: Vec<Shape>,
    wrapped: Newtype,
    word: String,
    #[serde(with = "crate::humanfloat::f64_or_tag")]
    annotated: f64,
    small: f32,
}

#[derive(Debug, Serialize, Deserialize)]
struct Newtype(f64);

const ODD_NAN: u64 = 0x7ff0_0000_0000_0001;

fn odd() -> [f64; 7] {
    [
        f64::NAN,
        -f64::NAN,
        f64::from_bits(ODD_NAN),
        f64::from_bits(0xfff8_0000_0000_0000),
        f64::INFINITY,
        f64::NEG_INFINITY,
        -0.0,
    ]
}

fn state() -> State {
    let o = odd();
    State {
        x: -f64::NAN,
        xs: o.to_vec(),
        ring: o.iter().copied().collect(),
        opt: Some(f64::NEG_INFINITY),
        none: None,
        pair: (f64::from_bits(ODD_NAN), 3),
        nested: vec![vec![], o.to_vec()],
        map: [("a".to_string(), f64::NAN), ("b".to_string(), 1.5)].into(),
        shapes: vec![
            Shape::Unit,
            Shape::New(f64::INFINITY),
            Shape::Tuple(f64::NAN, f32::NEG_INFINITY),
            Shape::Fields {
                a: -f64::NAN,
                b: o.to_vec(),
            },
        ],
        wrapped: Newtype(f64::NAN),
        word: "nan".into(),
        annotated: -f64::NAN,
        small: f32::INFINITY,
    }
}

/// The round trip through JSON gives the state's msgpack byte for byte:
/// every NaN keeps its sign and payload, every infinity its sign, and a
/// string that spells a tag stays a string.
#[test]
fn every_float_round_trips_through_json_to_the_bit() {
    let s = state();
    let text = to_json(&s);
    // The one `null` is the `None`.
    assert_eq!(text.matches("null").count(), 1, "{text}");
    assert!(text.contains(r#""-nan""#), "{text}");
    assert!(text.contains(r#""nan:0x7ff0000000000001""#), "{text}");
    let back: State = from_json(&text).unwrap();
    assert_eq!(
        rmp_serde::to_vec_named(&back).unwrap(),
        rmp_serde::to_vec_named(&s).unwrap()
    );
    assert_eq!(bits(&back.xs), bits(&s.xs));
    assert_eq!(back.word, "nan");
    assert_eq!(back.none, None);
}

/// msgpack through the wrappers is msgpack without them: they act on a
/// human-readable format only, so the state file's bytes cannot move.
#[test]
fn msgpack_through_the_wrappers_is_unchanged() {
    let s = state();
    let plain = rmp_serde::to_vec_named(&s).unwrap();
    let mut wrapped = Vec::new();
    s.serialize(Ser(
        &mut rmp_serde::Serializer::new(&mut wrapped).with_struct_map()
    ))
    .unwrap();
    assert_eq!(wrapped, plain);
    let back = State::deserialize(De(&mut rmp_serde::Deserializer::new(&plain[..]))).unwrap();
    assert_eq!(rmp_serde::to_vec_named(&back).unwrap(), plain);
}

/// An annotated field writes the text it wrote without the wrapper, so a
/// spec's JSON and a state's read alike.
#[test]
fn an_annotated_field_writes_the_same_text() {
    #[derive(Serialize)]
    struct A {
        #[serde(with = "crate::humanfloat::f64_or_tag")]
        a: f64,
        #[serde(with = "crate::humanfloat::vec_f64_or_tag")]
        v: Vec<f64>,
    }
    let a = A {
        a: f64::INFINITY,
        v: vec![f64::NAN, 2.0, f64::NEG_INFINITY],
    };
    assert_eq!(to_json(&a), serde_json::to_string(&a).unwrap());
}

/// What is not a float is left alone, and a value of the wrong kind where
/// a float belongs is still refused, naming what was expected.
#[test]
fn a_wrong_value_is_still_refused() {
    #[derive(Debug, Deserialize)]
    struct F {
        #[allow(dead_code)]
        f: f64,
    }
    for bad in [
        r#"{"f": null}"#,
        r#"{"f": "nanx"}"#,
        r#"{"f": "nan:0x0000000000000001"}"#,
    ] {
        let e = from_json::<F>(bad).expect_err(bad).to_string();
        assert!(e.contains(r#"expected a number, or "inf""#), "{bad}: {e}");
    }
    let f: F = from_json(r#"{"f": 3}"#).unwrap();
    assert_eq!(f.f, 3.0);
}

/// Each tag reads back as the float it was written from, to the bit.
#[test]
fn each_tag_reads_back_to_the_bit() {
    use crate::humanfloat::{tag, untagged};
    for v in odd().into_iter().filter(|v| !v.is_finite()) {
        let t = tag(v);
        assert_eq!(untagged(&t).map(f64::to_bits), Some(v.to_bits()), "{t}");
    }
    assert_eq!(tag(f64::NAN), "nan");
    assert_eq!(tag(f64::from_bits(0xfff8_0000_0000_0000)), "-nan");
    // A hex form that is not a NaN, or not sixteen digits, is no tag.
    assert_eq!(untagged("nan:0x7ff0000000000000"), None);
    assert_eq!(untagged("nan:0x7ff8"), None);
    assert_eq!(untagged("nan:0x7ff000000000000g"), None);
}
