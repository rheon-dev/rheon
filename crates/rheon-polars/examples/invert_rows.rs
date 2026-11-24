//! The `invert` node over a frame read from stdin, for
//! `tests/test_second_opinion_invert.py` to hold to scipy's `brentq`.
//!
//! Stdin is one JSON object, `{"node": [...], "columns": {name: [values]}}`,
//! the node in its list form and each column a list of numbers or nulls.
//! The frame is built from the columns and the node run over it as a Polars
//! expression, under `detail`; stdout is one JSON object with a list per
//! field of the detail struct (`root` and the bracket null where withheld,
//! `reason` null where a root is given).

use std::io::Read;

use polars::prelude::*;
use rheon_polars::{INVERT_DETAIL_FIELDS, Invert};
use serde_json::{Map, Value, json};

fn main() {
    let mut text = String::new();
    std::io::stdin()
        .read_to_string(&mut text)
        .expect("stdin is readable");
    let input: Value = serde_json::from_str(&text).expect("stdin is one JSON object");
    let inv = Invert::from_json(&input["node"]).unwrap_or_else(|e| panic!("{e}"));
    assert!(inv.detail(), "the node is run with detail: true");
    let Value::Object(cols) = &input["columns"] else {
        panic!("\"columns\" is an object of lists");
    };
    let columns: Vec<Column> = cols
        .iter()
        .map(|(name, v)| {
            let vals: Vec<Option<f64>> = v
                .as_array()
                .expect("a column is a list")
                .iter()
                .map(Value::as_f64)
                .collect();
            Column::new(name.as_str().into(), vals)
        })
        .collect();
    let df = DataFrame::new_infer_height(columns).expect("columns of one length");
    let out = df
        .lazy()
        .select([inv.to_expr().alias("out")])
        .collect()
        .unwrap_or_else(|e| panic!("{e}"));
    let s = out
        .column("out")
        .expect("the output")
        .struct_()
        .expect("a struct");
    let mut result = Map::new();
    for name in INVERT_DETAIL_FIELDS {
        let field = s.field_by_name(name).expect("a detail field");
        let values: Vec<Value> = match field.dtype() {
            DataType::Float64 => field.f64().unwrap().iter().map(|v| json!(v)).collect(),
            DataType::UInt32 => field.u32().unwrap().iter().map(|v| json!(v)).collect(),
            _ => field.str().unwrap().iter().map(|v| json!(v)).collect(),
        };
        result.insert(name.to_string(), Value::Array(values));
    }
    println!("{}", Value::Object(result));
}
