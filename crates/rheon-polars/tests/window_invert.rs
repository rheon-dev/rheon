//! The invert node as a node of the window formula grammar (PLAN 563, part
//! 2c): read by `Node::from_json` and written back by `to_json`, run inside
//! `with_windows` beside a window operator and as an operator's input, the
//! same bits however the input is chunked and across a save and a load, and
//! the refusals that keep it to what is built.

use polars::prelude::*;
use rheon_polars::{Invert, Node, WindowsConfig, WindowsRun};
use serde_json::{Value, json};

/// `u^3 + u = mid` on `[-10, 10]`, increasing: one root a row.
fn invert() -> Value {
    json!(["invert",
        ["+", ["**", ["unknown"], ["lit", 3.0]], ["unknown"]],
        ["col", "mid"], ["lit", -10.0], ["lit", 10.0],
        {"increasing": true}])
}

fn config() -> WindowsConfig {
    serde_json::from_value(json!({
        "formulas": [
            {"name": "level", "tree": ["ewm_mean", ["col", "mid"], {"half_life": 5.0}]},
            {"name": "ahead", "tree": ["rewm_mean", ["col", "mid"], {"half_life": 5.0, "window_size": 10.0}]},
            {"name": "root", "tree": invert()},
            {"name": "smooth", "tree": ["ewm_mean", invert(), {"half_life": 3.0}]},
        ],
        "clock": "t", "gap_cap": 50.0,
    }))
    .unwrap()
}

/// A clock that steps by a fraction, `mid` a walk with a null and a value
/// out of the bracket's range now and then.
fn frame(n: usize) -> DataFrame {
    let mut s = 0x9e37_79b9_7f4a_7c15u64;
    let mut r = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64
    };
    let (mut t, mut mid, mut clock, mut level) = (vec![], vec![], 0.0, 1.0);
    for i in 0..n {
        clock += 0.25 + r();
        level += r() - 0.5;
        t.push(clock);
        mid.push(match i % 37 {
            5 => None,
            11 => Some(5000.0),
            _ => Some(level),
        });
    }
    df!("t" => t, "mid" => mid).unwrap()
}

fn run_in(config: WindowsConfig, df: &DataFrame, chunk: usize) -> DataFrame {
    let mut run = WindowsRun::new(config, df.schema()).unwrap();
    let mut out = DataFrame::empty();
    let mut at = 0;
    while at < df.height() {
        let n = chunk.min(df.height() - at);
        out.vstack_mut(&run.feed(&df.slice(at as i64, n), None).unwrap())
            .unwrap();
        at += n;
    }
    out.vstack_mut(&run.finish().unwrap()).unwrap();
    out
}

fn bits(df: &DataFrame, name: &str) -> Vec<Option<u64>> {
    df.column(name)
        .unwrap()
        .f64()
        .unwrap()
        .iter()
        .map(|v| v.map(f64::to_bits))
        .collect()
}

#[test]
fn the_node_is_read_and_written_back_by_the_grammar() {
    let node = Node::from_json(&invert()).unwrap();
    assert!(matches!(node, Node::Invert(_)), "{node:?}");
    let written = node.to_json();
    assert_eq!(written[5]["xtol"], json!(rheon_core::DEFAULT_XTOL));
    assert_eq!(written[5]["rtol"], json!(rheon_core::MIN_RTOL));
    assert_eq!(written[5]["detail"], json!(false));
    assert_eq!(
        written.as_array().unwrap()[1..5],
        invert().as_array().unwrap()[1..5]
    );
    assert_eq!(Node::from_json(&written).unwrap(), node);
    assert_eq!(node.columns(), ["mid"]);
    assert!(node.operators().is_empty() && node.contains_call("invert"));
    // A tree that holds one, under arithmetic.
    let tree = json!(["-", ["col", "mid"], invert()]);
    let node = Node::from_json(&tree).unwrap();
    assert_eq!(
        node.to_json(),
        Node::from_json(&node.to_json()).unwrap().to_json()
    );
}

#[test]
fn a_run_gives_the_nodes_own_roots_beside_its_windows() {
    let df = frame(300);
    let out = run_in(config(), &df, 300);
    assert_eq!(out.height(), df.height());
    let direct = Invert::from_json(&invert()).unwrap();
    let want = df
        .clone()
        .lazy()
        .select([direct.to_expr().alias("root")])
        .collect()
        .unwrap();
    assert_eq!(bits(&out, "root"), bits(&want, "root"));
    let roots = bits(&out, "root");
    for (i, (r, m)) in roots
        .iter()
        .zip(df.column("mid").unwrap().f64().unwrap().iter())
        .enumerate()
    {
        match (i % 37, r, m) {
            (5 | 11, None, _) => {}
            (_, Some(r), Some(m)) => {
                let u = f64::from_bits(*r);
                assert!(
                    (u * u * u + u - m).abs() < 1e-9 * (1.0 + m.abs()),
                    "row {i}"
                );
            }
            other => panic!("row {i}: {other:?}"),
        }
    }
    // The window over the roots is the window of the column of them.
    let level = run_in(
        serde_json::from_value(json!({
            "formulas": [{"name": "smooth", "tree": ["ewm_mean", ["col", "root"], {"half_life": 3.0}]}],
            "clock": "t", "gap_cap": 50.0,
        }))
        .unwrap(),
        &df.hstack(&[want.column("root").unwrap().clone()]).unwrap(),
        300,
    );
    assert_eq!(bits(&out, "smooth"), bits(&level, "smooth"));
}

#[test]
fn a_run_with_an_invert_gives_the_same_bits_however_it_is_chunked() {
    let df = frame(1000);
    let one = run_in(config(), &df, 1000);
    for chunk in [1, 7, 333] {
        let other = run_in(config(), &df, chunk);
        for name in ["level", "ahead", "root", "smooth"] {
            assert_eq!(bits(&one, name), bits(&other, name), "{name} at {chunk}");
        }
    }
}

#[test]
fn a_saved_run_with_an_invert_loads_and_goes_on_to_the_bit() {
    let df = frame(400);
    let whole = run_in(config(), &df, 400);
    let mut first = WindowsRun::new(config(), df.schema()).unwrap();
    let mut out = first.feed(&df.slice(0, 250), None).unwrap();
    assert!(first.held() > 0, "rows held across the save");
    let bytes = first.save_bytes().unwrap();
    let text = String::from_utf8_lossy(&bytes);
    assert!(
        text.contains("invert") && text.contains("unknown"),
        "the tree is in the state"
    );
    let mut second = WindowsRun::load_bytes(&bytes, config(), df.schema()).unwrap();
    assert_eq!(
        second.save_bytes().unwrap(),
        bytes,
        "re-saved byte for byte"
    );
    out.vstack_mut(&second.feed(&df.slice(250, 150), None).unwrap())
        .unwrap();
    out.vstack_mut(&second.finish().unwrap()).unwrap();
    for name in ["level", "ahead", "root", "smooth"] {
        assert_eq!(bits(&out, name), bits(&whole, name), "{name}");
    }
    // The call that saved it, and no other: an invert with another target.
    let mut other = config();
    other.formulas[2].tree = Node::from_json(&json!(["invert",
        ["+", ["**", ["unknown"], ["lit", 3.0]], ["unknown"]],
        ["*", ["col", "mid"], ["lit", 2.0]], ["lit", -10.0], ["lit", 10.0], {"increasing": true}]))
    .unwrap();
    assert!(WindowsRun::load_bytes(&bytes, other, df.schema()).is_err());
}

#[test]
fn what_a_run_refuses_of_an_invert() {
    let df = frame(10);
    let alone: WindowsConfig = serde_json::from_value(json!({
        "formulas": [{"name": "root", "tree": invert()}],
        "clock": "t", "gap_cap": 50.0,
    }))
    .unwrap();
    let Err(err) = WindowsRun::new(alone, df.schema()) else {
        panic!("a run of inverts alone is refused");
    };
    let err = err.to_string();
    assert!(err.contains("no window to compute"), "{err}");
    let inner = json!(["invert", ["-", ["unknown"], ["ewm_mean", ["col", "mid"], {"half_life": 5.0}]],
        ["lit", 0.0], ["lit", -10.0], ["lit", 10.0], {"increasing": true}]);
    let err = Node::from_json(&inner).unwrap_err();
    assert!(
        err.contains("\"ewm_mean\"") && err.contains("as a column first"),
        "{err}"
    );
    let err = Node::from_json(&json!(["invert", ["unknown"]])).unwrap_err();
    assert!(err.contains("takes 5 arguments"), "{err}");
}
